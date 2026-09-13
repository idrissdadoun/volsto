"""Build the 2022 H2 SPX surface history from the HDN sample (SPEC §15 Part 2 / real-data run).

Runs the importer day by day (``build_hdn_history``), writes the snapshot YAMLs to
``configs/surfaces/snapshots/hdn_2022H2`` and the tidy history table plus the per-date failures
to ``outputs/m7/``.  Usage: ``python scripts/m7_hdn_history.py [--root DIR] [--limit N]``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from volsto.calibration.history import build_hdn_history, hdn_available_dates


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/hdn_sample/options_sample_2022H2")
    ap.add_argument("--out", default="outputs/m7")
    ap.add_argument("--snapshots", default="configs/surfaces/snapshots/hdn_2022H2")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-essvi", action="store_true", help="plain SSVI per day (eSSVI default)")
    ap.add_argument("--tag", default="", help="suffix of the output files")
    args = ap.parse_args()
    dates = hdn_available_dates(args.root)
    if args.limit:
        dates = dates[: args.limit]
    t0 = time.perf_counter()
    snap = Path(args.snapshots + ("_ssvi" if args.no_essvi else ""))
    history, failures = build_hdn_history(args.root, dates, snap, essvi=not args.no_essvi)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    history.frame.to_csv(out / f"hdn_history{args.tag}.csv", index=False)
    (out / f"hdn_history{args.tag}_failures.json").write_text(json.dumps(failures, indent=2))
    print(history)
    print(
        f"{len(dates)} dates requested, {history.n_dates} imported, {len(failures)} failed, "
        f"{time.perf_counter() - t0:.0f} s"
    )
    for d, e in failures.items():
        print(f"  {d}: {e}")


if __name__ == "__main__":
    main()
