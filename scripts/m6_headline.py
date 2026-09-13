"""M6 headline rows: the 3y annual autocall and the 3y Phoenix (SPEC §9.1 headline additions)
under LV (ω = 0), the 1F LSV (ω = 1, 2, 3) and the 2F Table 8.2 set on the reference surface.

    .venv/bin/python scripts/m6_headline.py --n-paths 400000 --seed 2024 --n-particles 800000

Models come from :func:`volsto.studies.m4.headline_models` (cache hits at 8·10⁵ particles for
the 1F ω = 1, 2, 3 and the 2F set; ω = 0 is the pure local vol).  Writes ``headline.csv``,
``headline.md`` and the SPEC §11 reproducibility sidecar ``headline.meta.json`` (code version —
the git commit —, seed, paths, particles, the spec paths and their leverage-cache keys, the
simulation settings) under ``--out`` (default ``outputs/m6/headline``).  Production run: hours at
8·10⁵ particles / 4·10⁵ paths when a calibration is missing from the cache — the integrator runs
it after the owner's decision on the Phoenix coupon (``volsto.studies.m6.PHOENIX_COUPON``).
Checked end to end (two Black–Scholes models monkeypatched in for the calibrations) by
``tests/test_m6_headline.py::test_script_main_writes_the_artefacts``.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path

from volsto.calibration import LeverageCache, code_version
from volsto.calibration.cache import spec_key
from volsto.config import CalibrationSpec, SimConfig, load_yaml
from volsto.studies.m4 import (
    HEADLINE_OMEGAS,
    LV_NAME,
    TWO_FACTOR_NAME,
    headline_models,
    one_factor_variants,
)
from volsto.studies.m6 import M6HeadlineResult, run_m6_headline, run_m6_headline_seeds

ROOT = Path(__file__).resolve().parents[1]


def _with_particles(spec: CalibrationSpec, n_particles: int) -> CalibrationSpec:
    """The spec as :func:`headline_models` calibrates it (its ``n_particles`` override)."""
    particle = dataclasses.replace(spec.particle, n_particles=n_particles)
    return dataclasses.replace(spec, particle=particle)


def run_metadata(
    args: argparse.Namespace,
    base: CalibrationSpec,
    spec_2f: CalibrationSpec | None,
    sim: SimConfig,
    result: M6HeadlineResult,
    model_names: list[str],
    wall_s: float,
) -> dict[str, object]:
    """The ``headline.meta.json`` payload (module docstring): what reproduces the table — code
    version, seed, paths, particles, spec paths and keys (``spec_key`` at ``--n-particles``, the
    leverage-cache addresses of the headline calibrations, calibration code tag included), the
    models and products of the table and the simulation settings."""
    base_p = _with_particles(base, args.n_particles)
    cache_keys = {
        name: spec_key(spec) for name, spec in one_factor_variants(base_p, HEADLINE_OMEGAS).items()
    }
    spec_2f_p = None if spec_2f is None else _with_particles(spec_2f, args.n_particles)
    if spec_2f_p is not None:
        cache_keys[TWO_FACTOR_NAME] = spec_key(spec_2f_p)
    return {
        "study": "m6_headline",
        "written_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "code_version": code_version(),
        "n_paths": int(args.n_paths),
        "seed": int(args.seed),
        "n_particles": int(args.n_particles),
        "spec_1f": str(Path(args.spec_1f).resolve()),
        "spec_key_1f": spec_key(base_p),
        "spec_2f": None if spec_2f_p is None else str(Path(args.spec_2f).resolve()),
        "spec_key_2f": None if spec_2f_p is None else spec_key(spec_2f_p),
        "calibration_cache_keys": cache_keys,
        "cache": str(Path(args.cache).resolve()),
        "models": model_names,
        "products": result.product_names,
        "reference": LV_NAME,
        "sim": dataclasses.asdict(sim),
        "wall_s": wall_s,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-paths", type=int, default=400_000)
    ap.add_argument(
        "--n-particles",
        type=int,
        default=800_000,
        help="particles per calibration (production convention 8e5, owner decision at M4b)",
    )
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument(
        "--price-seeds",
        type=int,
        nargs="*",
        default=None,
        help="average the table over these pricing seeds (owner decision, M6 review: the\n"
        "regression baseline uses six); the default prices the single --seed",
    )
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "m6" / "headline"))
    ap.add_argument("--spec-1f", default=str(ROOT / "configs/studies/lsv_reference_1f.yaml"))
    ap.add_argument("--spec-2f", default=str(ROOT / "configs/studies/lsv_reference_2f.yaml"))
    ap.add_argument("--no-2f", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    base = load_yaml(args.spec_1f, CalibrationSpec)
    spec_2f = None if args.no_2f else load_yaml(args.spec_2f, CalibrationSpec)
    t0 = time.perf_counter()
    models, _ = headline_models(
        LeverageCache(args.cache), base, spec_2f, n_particles=args.n_particles
    )
    logging.info("models ready in %.0f s", time.perf_counter() - t0)
    sim = dataclasses.replace(base.sim, n_paths=args.n_paths, seed=args.seed)
    if args.price_seeds:
        result = run_m6_headline_seeds(models, sim, args.price_seeds, reference=LV_NAME)
    else:
        result = run_m6_headline(models, sim, reference=LV_NAME)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out / "headline.csv", index=False)
    md = result.to_markdown()
    (out / "headline.md").write_text(md + "\n")
    meta = run_metadata(args, base, spec_2f, sim, result, list(models), time.perf_counter() - t0)
    (out / "headline.meta.json").write_text(json.dumps(meta, indent=2, default=str) + "\n")
    print(md)
    logging.info("done in %.0f s", time.perf_counter() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
