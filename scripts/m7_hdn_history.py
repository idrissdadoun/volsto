"""Build the 2022 H2 SPX surface history from the HDN sample (SPEC §15 Part 2 / real-data run).

Runs the importer day by day (``build_hdn_history``), writes the snapshot YAMLs to
``--snapshots`` and the tidy history table plus the per-date failures to ``--out``.  eSSVI days
go through the calendar repair (M10 Part 0) unless ``--no-calendar-repair``.

The snapshot directory defaults to ``<out>/snapshots/hdn_2022H2`` (under ``outputs/``, git
ignored): regenerating the committed snapshots under ``configs/`` changes the anchor surfaces
and possibly leverage cache keys, so it needs ``--allow-configs`` explicitly (M10 Part 0 guard).

Usage::

    .venv/bin/python scripts/m7_hdn_history.py [--root DIR] [--limit N] [--tag _x]
    .venv/bin/python scripts/m7_hdn_history.py --tag _repaired --out outputs/essvi_gate \\
        --snapshots /tmp/essvi_rep
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from volsto.calibration.history import build_hdn_history, hdn_available_dates
from volsto.market.import_hdn import DEFAULT_CALENDAR_REPAIR

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/hdn_sample/options_sample_2022H2")
    ap.add_argument("--out", default="outputs/m7")
    ap.add_argument(
        "--snapshots",
        default=None,
        help="snapshot directory (default <out>/snapshots/hdn_2022H2; '_ssvi' appended with "
        "--no-essvi)",
    )
    ap.add_argument(
        "--allow-configs",
        action="store_true",
        help="permit a snapshot directory under configs/ (regenerates committed snapshots)",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-essvi", action="store_true", help="plain SSVI per day (eSSVI default)")
    ap.add_argument(
        "--no-calendar-repair",
        action="store_true",
        help="eSSVI without the M10 calendar repair (a violating day fails, as before M10)",
    )
    ap.add_argument("--tag", default="", help="suffix of the output files")
    args = ap.parse_args()
    dates = hdn_available_dates(args.root)
    if args.limit:
        dates = dates[: args.limit]
    t0 = time.perf_counter()
    base = args.snapshots or str(Path(args.out) / "snapshots" / "hdn_2022H2")
    snap = Path(base + ("_ssvi" if args.no_essvi else ""))
    if snap.resolve().is_relative_to(CONFIGS.resolve()) and not args.allow_configs:
        sys.exit(f"refusing to write snapshots under {CONFIGS} without --allow-configs: {snap}")
    repair = None if args.no_calendar_repair else DEFAULT_CALENDAR_REPAIR
    history, failures = build_hdn_history(
        args.root, dates, snap, essvi=not args.no_essvi, calendar_repair=repair
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history.frame.to_csv(out / f"hdn_history{args.tag}.csv", index=False)
    (out / f"hdn_history{args.tag}_failures.json").write_text(json.dumps(failures, indent=2))
    print(history)
    print(
        f"{len(dates)} dates requested, {history.n_dates} imported, {len(failures)} failed, "
        f"snapshots in {snap}, calendar repair {'off' if repair is None else 'on'}, "
        f"recalibrated: no, wall clock {time.perf_counter() - t0:.0f} s"
    )
    for d, e in failures.items():
        print(f"  {d}: {e}")


if __name__ == "__main__":
    main()
