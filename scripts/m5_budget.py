"""M5 budget run (SPEC v2 §7.13): a full RiskReport under the reference 1F LSV (ω = 3) with the
recalibration count and wall clock — the viewer precompute budget.

    .venv/bin/python scripts/m5_budget.py --n-particles 200000 --n-paths 200000
    .venv/bin/python scripts/m5_budget.py --n-particles 800000 --n-paths 800000

Products: a 1y ATM call (the full report, bearing every recalibration) and, on the same
states (cache hits), the study cliquet's forward-variance ladders (recalibrated and
sticky-leverage) — the original study's comparison.  Writes ``report_<product>.csv``, the
Excel report, ``cliquet_fwd_var.csv`` and ``budget.md`` under ``--out``.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from pathlib import Path

import pandas as pd

from volsto.calibration import LeverageCache
from volsto.config import CalibrationSpec, load_yaml
from volsto.products import AdditiveCliquet, EuropeanOption
from volsto.risk import LSVBuilder, RiskEngine, RiskState, fwd_var_ladder, risk_report
from volsto.risk.engine import surface_of

ROOT = Path(__file__).resolve().parents[1]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-particles", type=int, default=200_000)
    ap.add_argument("--n-paths", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--spec", default=str(ROOT / "configs/studies/lsv_reference_1f.yaml"))
    ap.add_argument("--skip-cliquet", action="store_true")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out or ROOT / "outputs" / "m5" / f"budget_{args.n_particles // 1000}k")
    out.mkdir(parents=True, exist_ok=True)

    spec = load_yaml(args.spec, CalibrationSpec)
    spec = dataclasses.replace(
        spec, particle=dataclasses.replace(spec.particle, n_particles=args.n_particles)
    )
    state = RiskState(spec, label=f"1F omega=3, {args.n_particles} particles")
    sim = dataclasses.replace(spec.sim, n_paths=args.n_paths, seed=args.seed)
    t0 = time.perf_counter()
    engine = RiskEngine(LSVBuilder(LeverageCache(args.cache), state), sim)
    logging.info("base model ready in %.0f s", time.perf_counter() - t0)
    disc = surface_of(state).discount
    call = EuropeanOption(100.0, 1.0, 1, disc)

    t1 = time.perf_counter()
    rep = risk_report(engine, call, state)
    wall_call = time.perf_counter() - t1
    rep.to_dataframe().to_csv(out / "report_call.csv", index=False)
    rep.to_excel(out / "report_call.xlsx")
    lines = [
        f"# M5 budget — {args.n_particles} particles, {args.n_paths} paths, seed {args.seed}",
        "",
        f"- base calibration (cache {'hit' if engine.n_cache_misses == 0 else 'miss'}): "
        f"{t1 - t0:.0f} s",
        f"- 1y ATM call full report: {rep.budget['report_recalibrations']:.0f} recalibrations "
        f"({rep.budget['cache_misses']:.0f} cache misses), {rep.budget['pricings']:.0f} pricings, "
        f"{wall_call:.0f} s wall clock ({wall_call / 60:.1f} min)",
        f"- {rep.summary()}",
    ]
    logging.info(lines[-2])
    if not args.skip_cliquet:
        cliquet = AdditiveCliquet.study(1.0, disc, notional=100.0)
        t2 = time.perf_counter()
        n_cal = engine.n_calibrations
        frames = []
        for variant in ("recalibrated", "sticky_leverage"):
            lad = fwd_var_ladder(engine, cliquet, state, variant=variant)
            f = lad.as_frame()
            f.insert(0, "variant", variant)
            assert lad.total is not None and lad.parallel is not None
            f.attrs[variant] = {"total": lad.total.value, "parallel": lad.parallel.value}
            frames.append(f)
            lines.append(
                f"- cliquet fwd-var ladder [{variant}]: sum {lad.total.value:.4f} ± "
                f"{lad.total.stderr:.4f}, parallel {lad.parallel.value:.4f} ± "
                f"{lad.parallel.stderr:.4f} (% of notional per vol point)"
            )
        table = pd.concat(frames, ignore_index=True)
        table.to_csv(out / "cliquet_fwd_var.csv", index=False)
        lines.append(
            f"- cliquet ladders: {engine.n_calibrations - n_cal} new recalibrations, "
            f"{time.perf_counter() - t2:.0f} s"
        )
        wide = table.pivot(index="label", columns="variant", values="value").reindex(
            table["label"].unique()
        )
        se = table.pivot(index="label", columns="variant", values="stderr").reindex(wide.index)
        lines += [
            "",
            "| bucket | recalibrated | sticky leverage | net (recal - sticky) |",
            "|---|---|---|---|",
        ]
        for lab in wide.index:
            r, s_ = wide.loc[lab, "recalibrated"], wide.loc[lab, "sticky_leverage"]
            lines.append(
                f"| {lab} | {r:.4f} ± {se.loc[lab, 'recalibrated']:.4f} | "
                f"{s_:.4f} ± {se.loc[lab, 'sticky_leverage']:.4f} | {r - s_:+.4f} |"
            )
    lines.append("")
    lines.append(f"- total wall clock {time.perf_counter() - t0:.0f} s; budget {engine.budget()}")
    (out / "budget.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
