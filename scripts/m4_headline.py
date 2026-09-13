"""M4 headline table: LV vs 1F LSV (ω = 1, 2, 3) vs 2F Table 8.2 on the reference surface.

    .venv/bin/python scripts/m4_headline.py --n-paths 400000 --seed 2024 --out outputs/m4

Writes ``headline.csv``, ``wing.csv``, the forward smiles and ``headline.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from pathlib import Path

from volsto.calibration import LeverageCache
from volsto.config import CalibrationSpec, load_yaml
from volsto.studies.m4 import headline_models, run_headline

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-paths", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "m4"))
    ap.add_argument("--spec-1f", default=str(ROOT / "configs/studies/lsv_reference_1f.yaml"))
    ap.add_argument("--spec-2f", default=str(ROOT / "configs/studies/lsv_reference_2f.yaml"))
    ap.add_argument("--no-2f", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base = load_yaml(args.spec_1f, CalibrationSpec)
    spec_2f = None if args.no_2f else load_yaml(args.spec_2f, CalibrationSpec)
    t0 = time.perf_counter()
    models, _ = headline_models(LeverageCache(args.cache), base, spec_2f)
    logging.info("models ready in %.0f s", time.perf_counter() - t0)
    sim = dataclasses.replace(base.sim, n_paths=args.n_paths, seed=args.seed)
    result = run_headline(models, sim)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out / "headline.csv", index=False)
    result.wing.to_csv(out / "wing.csv", index=False)
    for name, smile in result.smiles.items():
        smile.as_frame().to_csv(
            out / f"smile_{name.replace(' ', '_').replace('=', '')}.csv", index=False
        )
    md = result.to_markdown()
    (out / "headline.md").write_text(md + "\n")
    print(md)
    logging.info("done in %.0f s", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
