"""Flask web app: serves the 3D HAWK 6 viewer + a JSON API.

Endpoints
---------
GET /                     -> the 3D viewer (index.html)
GET /api/tles             -> raw TLE records for the HAWK 6 cluster
GET /api/snapshot         -> current lat/lon/alt/velocity for every sat
                             optional ?at=<ISO-8601> to query an arbitrary time
                             optional ?trails=1 to include ±45 min orbit trails
GET /api/orbit/<name>     -> orbit trail for a single sat
                             optional ?from=<ISO>&to=<ISO>&step=<seconds>
GET /healthz              -> liveness probe

Run locally:
    pip install -r requirements.txt
    python app.py            # http://127.0.0.1:8000
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from flask import Flask, abort, jsonify, request, send_from_directory

import hawk6_tracker as tracker

app = Flask(__name__, static_folder=".", static_url_path="")

# Cheap in-process cache so we don't hammer CelesTrak. The TLE feed only
# updates a handful of times per day; 1 hour is plenty.
_TLE_CACHE: dict = {"records": None, "fetched_at": 0.0}
_TLE_TTL_SECONDS = 3600


def _get_records(force: bool = False) -> list[tracker.TLERecord]:
    import time as _t

    now = _t.time()
    if (
        force
        or _TLE_CACHE["records"] is None
        or now - _TLE_CACHE["fetched_at"] > _TLE_TTL_SECONDS
    ):
        _TLE_CACHE["records"] = tracker.get_hawk6_tles()
        _TLE_CACHE["fetched_at"] = now
    return _TLE_CACHE["records"]


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    # Tolerate trailing Z (Python <3.11 doesn't parse it natively).
    cleaned = value.replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        abort(400, description=f"Invalid ISO-8601 timestamp: {value!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
# Static viewer
# ---------------------------------------------------------------------------


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/api/tles")
def api_tles():
    records = _get_records(force=request.args.get("refresh") == "1")
    return jsonify(
        {
            "fetched_at": datetime.fromtimestamp(
                _TLE_CACHE["fetched_at"], tz=timezone.utc
            ).isoformat().replace("+00:00", "Z"),
            "satellites": [r.to_dict() for r in records],
        }
    )


@app.get("/api/snapshot")
def api_snapshot():
    when = _parse_iso(request.args.get("at"))
    include_trails = request.args.get("trails") in ("1", "true", "yes")
    # Re-use the cached records so each /snapshot call is cheap.
    records = _get_records()
    epoch = when or datetime.now(timezone.utc)
    sats = []
    for r in records:
        entry = r.to_dict()
        entry["position"] = tracker.position_at(r, epoch).to_dict()
        if include_trails:
            entry["trail"] = [p.to_dict() for p in tracker.orbit_trail(r)]
        sats.append(entry)
    return jsonify(
        {
            "generated_at": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "epoch": epoch.isoformat().replace("+00:00", "Z"),
            "source": tracker.CELESTRAK_URL,
            "satellites": sats,
        }
    )


@app.get("/api/orbit/<name>")
def api_orbit(name: str):
    records = _get_records()
    target = next((r for r in records if r.name.upper() == name.upper()), None)
    if not target:
        abort(404, description=f"Unknown satellite: {name!r}")
    start = _parse_iso(request.args.get("from"))
    end = _parse_iso(request.args.get("to"))
    step = float(request.args.get("step", 30.0))
    trail = tracker.orbit_trail(target, start=start, end=end, step_seconds=step)
    return jsonify(
        {
            "name": target.name,
            "norad_id": target.norad_id,
            "points": [p.to_dict() for p in trail],
        }
    )


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
