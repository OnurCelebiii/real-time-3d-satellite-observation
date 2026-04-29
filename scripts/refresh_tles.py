"""Refresh ``data/hawk6.json`` from CelesTrak when it's older than 6 hours.

Intended to run on a schedule (cron / GitHub Actions) **and** on demand
(e.g. as a startup hook for the Flask app). If the existing cache file
is still fresh, the script is a no-op so it's safe to call frequently.

Usage:
    python scripts/refresh_tles.py                    # refresh if stale
    python scripts/refresh_tles.py --force            # always refetch
    python scripts/refresh_tles.py --max-age-hours 1  # tighter freshness
    python scripts/refresh_tles.py --trails           # include orbit trails
"""

from __future__ import annotations

import argparse
import sys
from datetime import timedelta
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
        default=str(tracker.DEFAULT_CACHE_PATH),
        help="Where to write the JSON snapshot.",
    )
    parser.add_argument(
        "--trails",
        action="store_true",
        help="Include ±45 min orbit trails for each satellite.",
    )
    parser.add_argument(
        "--max-age-hours",
        type=float,
        default=tracker.DEFAULT_MAX_CACHE_AGE.total_seconds() / 3600,
        help="Re-fetch only if the existing file is older than this. Default: 6h.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Always re-fetch, ignoring any existing cache file.",
    )
    args = parser.parse_args()

    try:
        data = tracker.load_or_refresh_snapshot(
            args.output,
            max_age=timedelta(hours=args.max_age_hours),
            force=args.force,
            include_trails=args.trails,
        )
    except tracker.TLEFetchError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 1

    cache = data.get("cache", {})
    sat_names = ", ".join(s["name"] for s in data["satellites"])
    if cache.get("refreshed"):
        print(
            f"Refreshed {args.output} "
            f"({len(data['satellites'])} sats: {sat_names})"
        )
    else:
        age_min = (cache.get("age_seconds") or 0) // 60
        print(
            f"Cache is fresh ({age_min} min old, max "
            f"{args.max_age_hours:g}h) — kept {args.output}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
