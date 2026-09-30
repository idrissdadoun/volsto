"""The payoff study, stage 1 (``volsto/studies/payoff.py``): prices and model risk, Greeks and
ladders, marking dials, rotation with the P1 parameters held, the hedge comparison.

    caffeinate -is .venv/bin/python scripts/payoff_study.py --parts prices,greeks,dials,rotation
    caffeinate -is .venv/bin/python scripts/payoff_study.py --parts hedge

Calibrates the leverages the cache lacks (the 1F reference model, the dial marks, the rotated
states, the Greeks' bumped surfaces, the hedger's vega bump) unless ``--no-calibrate``; every
part is resumable (a written table is skipped, the hedge part resumes run by run).  Prints each
part's wall clock and the calibrations it triggered.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from volsto.studies.payoff import (
    DEFAULT_CACHE,
    DEFAULT_OUT,
    MARKING_FIT,
    PARTS,
    PayoffConfig,
    PayoffEnvironment,
    run_part,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--parts", default=",".join(PARTS), help=f"comma list of {PARTS}")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--marking-fit", type=Path, default=MARKING_FIT)
    ap.add_argument("--n-particles", type=int, default=800_000)
    ap.add_argument("--price-paths", type=int, default=100_000)
    ap.add_argument("--greek-paths", type=int, default=50_000)
    ap.add_argument("--hedge-paths", type=int, default=20_000)
    ap.add_argument("--world-paths", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--no-calibrate", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        ap.error(f"unknown parts {unknown}; choose from {PARTS}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    cfg = PayoffConfig(
        out=args.out,
        cache=args.cache,
        marking_fit=args.marking_fit,
        n_particles=args.n_particles,
        seed=args.seed,
        price_paths=args.price_paths,
        greek_paths=args.greek_paths,
        hedge_paths=args.hedge_paths,
        world_paths=args.world_paths,
        allow_calibrate=not args.no_calibrate,
        verbose=args.verbose,
    )
    env = PayoffEnvironment(cfg)
    t0 = time.perf_counter()
    for part in parts:
        res = run_part(env, part)
        print(
            f"[payoff] {part}: wall {res.wall_s:.0f} s, {res.calibrations} calibration(s); "
            f"table {res.path}",
            flush=True,
        )
    print(
        f"[payoff] done: wall {time.perf_counter() - t0:.0f} s; recalibrated: "
        f"{'yes' if env.calibrated else 'no'} ({len(env.calibrated)} leverage(s))",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
