"""Refresh ``data/hawk6.json`` from CelesTrak.

Intended to run on a schedule (cron / GitHub Actions). The output file is
consumed by the static 3D viewer when no Python backend is available, so
the GitHub Pages build always has up-to-date ephemeris.

Usage:
    python scripts/refresh_tles.py [--output PATH] [--trails]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the repo root importable when this script is run directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import hawk6_tracker as tracker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "-o",
        default=str(ROOT / "data" / "hawk6.json"),
        help="Where to write the JSON snapshot.",
    )
    parser.add_argument(
        "--trails",
        action="store_true",
        help="Include ±45 min orbit trails for each satellite.",
    )
    args = parser.parse_args()

    try:
        data = tracker.snapshot(include_trails=args.trails)
    except tracker.TLEFetchError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    sat_names = ", ".join(s["name"] for s in data["satellites"])
    print(f"Wrote {out} ({len(data['satellites'])} sats: {sat_names})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
