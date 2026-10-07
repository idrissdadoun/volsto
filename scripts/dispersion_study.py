"""The dispersion study, stage 1 (``volsto/studies/dispersion.py``): the market world's prices
and sensitivities, the expectations grid of the framework, the basket's 2006–2026 history.

    .venv/bin/python scripts/dispersion_study.py [--parts sensitivities,expectations,history]
        [--n-paths 100000] [--out outputs/dispersion]

No leverage calibration (Black–Scholes and local-vol names); every part is resumable.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from volsto.studies.dispersion import (
    DEFAULT_OUT,
    PARTS,
    DispersionConfig,
    Environment,
    run_part,
)
from volsto.studies.history_stats import HISTORY_DIR


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--parts", default=",".join(PARTS), help=f"comma list of {PARTS}")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--history", type=Path, default=HISTORY_DIR)
    ap.add_argument("--n-paths", type=int, default=100_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--anchor", default="2022-12-30")
    args = ap.parse_args(argv)
    parts = [p.strip() for p in args.parts.split(",") if p.strip()]
    unknown = [p for p in parts if p not in PARTS]
    if unknown:
        ap.error(f"unknown parts {unknown}; choose from {PARTS}")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    env = Environment(
        DispersionConfig(
            out=args.out,
            history=args.history,
            n_paths=args.n_paths,
            seed=args.seed,
            anchor=args.anchor,
        )
    )
    print(
        f"[dispersion] market world: COR3M {env.setup['cor3m_implied']:.3f}, realised 1y corr "
        f"{env.setup['corr_realised_1y']:.3f}, vols {[round(v, 3) for v in env.market.vols]}, "
        f"call strike {env.call_strike:.4f}, package scale {env.package_scale:.3f}",
        flush=True,
    )
    t0 = time.perf_counter()
    for part in parts:
        res = run_part(env, part)
        print(f"[dispersion] {part}: wall {res.wall_s:.0f} s; table {res.path}", flush=True)
    print(f"[dispersion] done: wall {time.perf_counter() - t0:.0f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
