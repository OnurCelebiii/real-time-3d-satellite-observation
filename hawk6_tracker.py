"""Real-time tracker for the HawkEye 360 HAWK 6 satellite cluster.

This module exposes the small set of functions you need to:

* fetch live two-line elements (TLEs) from CelesTrak, filtered to HAWK 6,
* parse them into ``TLERecord`` objects,
* propagate each satellite to a given UTC instant (lat / lon / altitude /
  velocity) using SGP4 via Skyfield,
* generate an orbit trail (polyline of positions over a time window),
* dump a full constellation snapshot as a JSON-serialisable dict.

The module has no global state and every function is independently usable,
so it slots cleanly into a Flask / FastAPI / Django app, a CLI script, a
Jupyter notebook, or a scheduled job that writes a static JSON file.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from typing import Iterable, Optional

import requests
from skyfield.api import EarthSatellite, load, wgs84

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: CelesTrak GP query that returns every catalog entry whose NAME contains
#: "HAWK". We filter the response client-side for the HAWK 6 cluster.
CELESTRAK_URL = (
    "https://celestrak.org/NORAD/elements/gp.php?NAME=HAWK&FORMAT=tle"
)

#: HAWK 6 name pattern. CelesTrak names them ``HAWK-6A`` / ``HAWK-6B`` /
#: ``HAWK-6C`` (hyphenated), but we tolerate ``HAWK 6`` and ``HAWK6`` too.
HAWK6_NAME_PATTERN = re.compile(r"^HAWK[\s\-]*6", re.IGNORECASE)

#: Mean Earth radius in km, used for altitude conversions / sanity checks.
EARTH_RADIUS_KM = 6371.0

#: Embedded snapshot taken from CelesTrak on 2026-04-29. Used as the
#: ultimate fallback if the network is unreachable; the live fetch is
#: always tried first.
FALLBACK_TLE_TEXT = """\
HAWK-6A
1 55327U 23011D   26119.45216630  .00004994  00000+0  21925-3 0  9992
2 55327  40.4991 274.6944 0002584   1.9581 358.1238 15.22489875180426
HAWK-6B
1 55324U 23011A   26119.51836108  .00005273  00000+0  23105-3 0  9997
2 55324  40.4994 275.8448 0002150  20.3427 339.7468 15.22489134180429
HAWK-6C
1 55326U 23011C   26119.12415408  .00005426  00000+0  23748-3 0  9995
2 55326  40.4980 274.9866 0002906   4.4998 355.5837 15.22488460180376
"""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TLERecord:
    """A single three-line TLE entry (name + two element lines)."""

    name: str
    line1: str
    line2: str

    @property
    def norad_id(self) -> str:
        return self.line1[2:7].strip()

    @property
    def inclination_deg(self) -> float:
        return float(self.line2[8:16])

    @property
    def mean_motion_rev_per_day(self) -> float:
        return float(self.line2[52:63])

    @property
    def period_minutes(self) -> float:
        n = self.mean_motion_rev_per_day
        return 1440.0 / n if n > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "line1": self.line1,
            "line2": self.line2,
            "norad_id": self.norad_id,
            "inclination_deg": self.inclination_deg,
            "period_minutes": self.period_minutes,
        }


@dataclass(frozen=True)
class Position:
    """Geodetic position + speed of a satellite at a single instant."""

    name: str
    norad_id: str
    timestamp: str  # ISO-8601 UTC
    latitude_deg: float
    longitude_deg: float
    altitude_km: float
    velocity_kms: float

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# TLE fetch / parse / filter
# ---------------------------------------------------------------------------


class TLEFetchError(RuntimeError):
    """Raised when no live TLE source can be reached."""


def fetch_tle_text(
    url: str = CELESTRAK_URL,
    timeout: float = 20.0,
    retries: int = 3,
    backoff: float = 2.0,
) -> str:
    """Download raw TLE text from CelesTrak with simple retry/backoff."""
    last_err: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "hawk6-tracker/1.0"},
            )
            r.raise_for_status()
            text = r.text.strip()
            if not text or "Invalid query" in text.splitlines()[0]:
                raise TLEFetchError(f"CelesTrak rejected query: {text[:120]}")
            return text
        except (requests.RequestException, TLEFetchError) as exc:
            last_err = exc
            if attempt < retries - 1:
                time.sleep(backoff * (2**attempt))
    raise TLEFetchError(f"Could not fetch TLE feed: {last_err}")


def parse_tle_text(text: str) -> list[TLERecord]:
    """Parse the standard CelesTrak 3-line-per-satellite TLE format."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    records: list[TLERecord] = []
    i = 0
    while i + 2 < len(lines) + 1:
        if i + 2 >= len(lines):
            break
        name, l1, l2 = lines[i], lines[i + 1], lines[i + 2]
        if l1.startswith("1 ") and l2.startswith("2 "):
            records.append(TLERecord(name=name, line1=l1, line2=l2))
            i += 3
        else:
            i += 1
    return records


def filter_hawk6(records: Iterable[TLERecord]) -> list[TLERecord]:
    """Keep only entries that belong to the HAWK 6 cluster."""
    return [r for r in records if HAWK6_NAME_PATTERN.match(r.name)]


def get_hawk6_tles(
    *,
    url: str = CELESTRAK_URL,
    use_fallback: bool = True,
    timeout: float = 20.0,
) -> list[TLERecord]:
    """High-level helper: fetch + parse + filter, with embedded fallback."""
    try:
        text = fetch_tle_text(url, timeout=timeout)
        source = "live"
    except TLEFetchError:
        if not use_fallback:
            raise
        text = FALLBACK_TLE_TEXT
        source = "fallback"
    records = filter_hawk6(parse_tle_text(text))
    if not records and use_fallback and source == "live":
        # Live feed was reachable but empty / format-changed -> try fallback
        records = filter_hawk6(parse_tle_text(FALLBACK_TLE_TEXT))
    if not records:
        raise TLEFetchError("No HAWK 6 satellites in TLE feed")
    return records


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------


# Skyfield's timescale is reusable and a bit expensive to build, so we
# cache it module-level. ``builtin=True`` avoids a network download.
_TS = load.timescale(builtin=True)


def to_skyfield(record: TLERecord) -> EarthSatellite:
    """Convert a ``TLERecord`` to a Skyfield ``EarthSatellite``."""
    return EarthSatellite(record.line1, record.line2, record.name, _TS)


def _ensure_aware_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def position_at(record: TLERecord, when: Optional[datetime] = None) -> Position:
    """Propagate ``record`` to ``when`` (UTC) and return a ``Position``."""
    when = _ensure_aware_utc(when or datetime.now(timezone.utc))
    sat = to_skyfield(record)
    t = _TS.from_datetime(when)
    geocentric = sat.at(t)
    subpoint = wgs84.subpoint_of(geocentric)
    altitude_km = wgs84.height_of(geocentric).km
    # Velocity magnitude from the geocentric ICRF velocity vector (km/s).
    vx, vy, vz = geocentric.velocity.km_per_s
    velocity = (vx * vx + vy * vy + vz * vz) ** 0.5
    return Position(
        name=record.name,
        norad_id=record.norad_id,
        timestamp=when.isoformat().replace("+00:00", "Z"),
        latitude_deg=float(subpoint.latitude.degrees),
        longitude_deg=float(subpoint.longitude.degrees),
        altitude_km=float(altitude_km),
        velocity_kms=float(velocity),
    )


def orbit_trail(
    record: TLERecord,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    step_seconds: float = 30.0,
) -> list[Position]:
    """Sample the ground track of ``record`` between ``start`` and ``end``.

    Defaults to ±45 minutes around "now", which covers roughly one orbit
    for a typical LEO satellite (HAWK 6 has a ~94 minute period).
    """
    now = datetime.now(timezone.utc)
    start = _ensure_aware_utc(start or now - timedelta(minutes=45))
    end = _ensure_aware_utc(end or now + timedelta(minutes=45))
    if end <= start:
        raise ValueError("end must be after start")
    step = timedelta(seconds=step_seconds)
    positions: list[Position] = []
    t = start
    while t <= end:
        positions.append(position_at(record, t))
        t += step
    return positions


# ---------------------------------------------------------------------------
# Snapshots
# ---------------------------------------------------------------------------


def snapshot(
    *,
    when: Optional[datetime] = None,
    use_fallback: bool = True,
    include_trails: bool = False,
    trail_step_seconds: float = 30.0,
) -> dict:
    """Build a JSON-friendly snapshot of the whole HAWK 6 constellation."""
    when = _ensure_aware_utc(when or datetime.now(timezone.utc))
    records = get_hawk6_tles(use_fallback=use_fallback)
    out = {
        "generated_at": datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z"),
        "epoch": when.isoformat().replace("+00:00", "Z"),
        "source": CELESTRAK_URL,
        "satellites": [],
    }
    for r in records:
        entry = r.to_dict()
        entry["position"] = position_at(r, when).to_dict()
        if include_trails:
            entry["trail"] = [p.to_dict() for p in orbit_trail(r)]
        out["satellites"].append(entry)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Print a JSON snapshot of the HAWK 6 constellation.",
    )
    parser.add_argument(
        "--trails",
        action="store_true",
        help="Include ±45 min orbit trails for each satellite.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Fail instead of using the embedded TLE fallback.",
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Write JSON to this path instead of stdout.",
    )
    args = parser.parse_args()
    data = snapshot(
        include_trails=args.trails,
        use_fallback=not args.no_fallback,
    )
    text = json.dumps(data, indent=2)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


if __name__ == "__main__":
    _cli()
