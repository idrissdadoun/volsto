"""M4b acceptance (SPEC §12): the second-order SV step judged on the coarse schedule.

    .venv/bin/python scripts/m4b_acceptance.py --n-particles 200000 800000

For each particle count: calibrate the 1F reference LSV (ω = 3, κ = 1.5, ρ = −0.7, 3y horizon)
on the schedule 1/1460 – 1/365 – 1/250, reprice the §4.2 pillars and variance swaps on the same
schedule, and print the variance-swap and put-wing errors with standard errors and the wall
times.  Entries go through the leverage cache so the headline rerun reuses them.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from pathlib import Path

import pandas as pd

from volsto.calibration import LeverageCache
from volsto.config import CalibrationSpec, StepSchedule, load_yaml

ROOT = Path(__file__).resolve().parents[1]
COARSE = StepSchedule(breaks=(0.25, 2.0), dts=(1.0 / 1460.0, 1.0 / 365.0, 1.0 / 250.0))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec", default=str(ROOT / "configs/studies/lsv_reference_1f.yaml"))
    ap.add_argument("--n-particles", type=int, nargs="+", default=[200_000, 800_000])
    ap.add_argument("--diag-paths", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "m4b"))
    ap.add_argument("--fine", action="store_true", help="use the default (fine) schedule instead")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base = load_yaml(args.spec, CalibrationSpec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = LeverageCache(args.cache)
    pd.set_option("display.width", 200)
    for n in args.n_particles:
        sim = base.sim if args.fine else dataclasses.replace(base.sim, dt_max=COARSE)
        spec = dataclasses.replace(
            base, particle=dataclasses.replace(base.particle, n_particles=n), sim=sim
        )
        diag_sim = dataclasses.replace(sim, n_paths=args.diag_paths, seed=args.seed)
        t0 = time.perf_counter()
        hit = cache.has(spec)
        model, report = cache.get_or_calibrate(spec, run_diagnostics=True, diagnostics_sim=diag_sim)
        t_total = time.perf_counter() - t0
        assert report is not None
        tag = f"N{n}_{'fine' if args.fine else 'coarse'}"
        report.vanillas.to_csv(out / f"vanillas_{tag}.csv", index=False)
        report.varswaps.to_csv(out / f"varswaps_{tag}.csv", index=False)
        print(
            f"\n=== {tag}: cache {'hit' if hit else 'miss'}; total {t_total:.0f} s "
            f"(diagnostics {report.wall_time:.0f} s) ==="
        )
        print(report.varswaps.to_string())
        print(report.vanillas.to_string())
        print(report.summary())
        print(
            "passes(0.15, 2y, |k|<=0.2, z=3, max_std=2.5):",
            report.passes(0.15, 2.0, 0.2, 3.0, max_std=2.5),
        )
        print("model:", model)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
