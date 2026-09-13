"""M6 Part 0 — calibration-side pass (SPEC §4.2 tagged diagnostic; owner's prescription at the
M4b acceptance).

    .venv/bin/python scripts/m6_calibration_pass.py --n-particles 800000 --seeds 12345 777 4242

For each particle seed and each spec (1F reference, optionally the 2F Table 8.2 set):

* **A. coarse / coarse** — calibrate ``L`` on the coarse schedule (1/1460 – 1/365 – 1/250: the
  "daily slice grid" beyond 3m) and reprice the §4.2 pillars on the same schedule (the
  production configuration);
* **B. coarse / fine** — the *diagnostic*: the same coarse-calibrated ``L`` priced on the fine
  schedule (1/2920 – 1/730 – 1/500); the kernel evaluates ``L`` at every step start by linear
  interpolation in ``t`` between the coarse slices (:meth:`LeverageFunction.rows`), so this is
  "simulate on the fine schedule with ``L`` interpolated linearly in ``t``" without any change to
  the code;
* **C. fine / fine** — calibrate on the fine schedule and price on it (the configuration that
  showed the +0.05 to +0.08 vp ATM bias at 1y–3y in M4b);
* **B2. coarse / fine, frozen** — the coarse ``L`` held per coarse slice (step function in
  ``t``) on the fine steps: separates the kernel step size from the time interpolation of ``L``;
* **D. fine / coarse** — NOT realisable: the engine grid always contains the leverage slices
  (SPEC §5), so a fine ``L`` is always priced on at least its own fine steps and D reproduces C
  exactly; kept only as a check of that rule.

If B shows no long-end bias while C does, the bias is the per-slice regression bias compounding
with the slice count (twice the slices on the fine schedule), not the time step.  Reports the
seed-averaged implied-vol errors (vol points; mean over seeds ± the standard error of that mean
and the mean MC stderr) at 1y, 18m, 2y, 3y for k = 0, −0.2, −0.3 and the variance-swap errors,
per variant, plus every per-seed table, under ``--out``.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from volsto.calibration import LeverageCache
from volsto.calibration.cache import build_market
from volsto.calibration.diagnostics import CalibrationReport, reprice_surface_seeds
from volsto.config import CalibrationSpec, StepSchedule, load_yaml
from volsto.models import LSV
from volsto.models.leverage import LeverageFunction

ROOT = Path(__file__).resolve().parents[1]
COARSE = StepSchedule(breaks=(0.25, 2.0), dts=(1.0 / 1460.0, 1.0 / 365.0, 1.0 / 250.0))
FINE = StepSchedule(breaks=(0.25, 2.0), dts=(1.0 / 2920.0, 1.0 / 730.0, 1.0 / 500.0))
VARIANTS: dict[str, tuple[StepSchedule, StepSchedule, str]] = {
    "A_coarse_coarse": (COARSE, COARSE, "linear"),
    "B_coarse_fine": (COARSE, FINE, "linear"),
    "B2_coarse_fine_frozen": (COARSE, FINE, "frozen"),
    "C_fine_fine": (FINE, FINE, "linear"),
    "D_fine_coarse": (FINE, COARSE, "linear"),
}
REPORT_T = (1.0, 1.5, 2.0, 3.0)
REPORT_K = (0.0, -0.2, -0.3)


def run_variant(
    cache: LeverageCache,
    base: CalibrationSpec,
    n_particles: int,
    seed: int,
    cal_schedule: StepSchedule,
    price_schedule: StepSchedule,
    n_paths: int,
    price_seeds: Sequence[int],
    time_interp: str = "linear",
) -> tuple[CalibrationReport, float, bool]:
    spec = dataclasses.replace(
        base,
        particle=dataclasses.replace(base.particle, n_particles=n_particles, seed=seed),
        sim=dataclasses.replace(base.sim, dt_max=cal_schedule),
    )
    hit = cache.has(spec)
    t0 = time.perf_counter()
    model, _ = cache.get_or_calibrate(spec)
    t_cal = time.perf_counter() - t0
    if time_interp == "frozen":
        model = LSV(model.kernel, step_function_leverage(model.leverage))
    _, surface, _ = build_market(spec)
    sim = dataclasses.replace(spec.sim, dt_max=price_schedule, n_paths=n_paths)
    report = reprice_surface_seeds(model, surface, sim, price_seeds)
    return report, t_cal, hit


def step_function_leverage(lev: LeverageFunction) -> LeverageFunction:
    """``L`` held at the slice value over each slice interval (the frozen-L rule of the
    calibration) instead of interpolated linearly in ``t``: every slice is duplicated just
    before the next one, so :meth:`LeverageFunction.rows` returns ``L_j`` on ``[t_j, t_{j+1})``.
    Separates the kernel's step-size effect from the time-interpolation of ``L`` (variant B2)."""
    t = lev.times
    eps = 1e-9
    times = np.empty(2 * t.size - 1)
    values = np.empty((2 * t.size - 1, lev.values.shape[1]))
    times[0::2] = t
    values[0::2] = lev.values
    times[1::2] = t[1:] - eps
    values[1::2] = lev.values[:-1]
    return LeverageFunction(
        times, lev.k_grid, values, lev.forward_curve, {**lev.metadata, "time_interp": "frozen"}
    )


def summarise(reports: list[CalibrationReport]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seed-averaged vanilla errors on the report cells and variance-swap errors."""
    van = pd.concat([r.vanillas.assign(seed=i) for i, r in enumerate(reports)])
    sel = van[van["T"].isin(REPORT_T) & van["k"].isin(REPORT_K)]
    g = sel.groupby(["T", "k"])
    out = pd.DataFrame(
        {
            "mean_error_vp": g["error_vp"].mean(),
            "seed_se_vp": g["error_vp"].std(ddof=1) / np.sqrt(g["error_vp"].count()),
            "mc_se_vp": g["stderr_vp"].mean(),
            "n_seeds": g["error_vp"].count(),
        }
    ).reset_index()
    vs = pd.concat([r.varswaps.assign(seed=i) for i, r in enumerate(reports)])
    err_col = next((c for c in vs.columns if c in ("diff_vp", "error_vp")), None)
    se_col = next((c for c in vs.columns if "stderr" in c or c == "se"), None)
    t_col = "T" if "T" in vs.columns else vs.columns[0]
    gv = vs.groupby(t_col)
    vs_out = pd.DataFrame(
        {
            "mean_error_vp": gv[err_col].mean() if err_col else np.nan,
            "seed_se_vp": (
                (gv[err_col].std(ddof=1) / np.sqrt(gv[err_col].count())) if err_col else np.nan
            ),
            "mc_se_vp": gv[se_col].mean() if se_col else np.nan,
        }
    ).reset_index()
    return out, vs_out


def markdown(name: str, table: pd.DataFrame, vs: pd.DataFrame) -> list[str]:
    lines = [
        f"### {name}",
        "",
        "| T | k | mean error (vp) | seed SE | MC SE |",
        "|---|---|---|---|---|",
    ]
    for _, r in table.iterrows():
        cells = (
            f"{r['T']:g} | {r['k']:+.1f} | {r['mean_error_vp']:+.3f} | "
            f"{r['seed_se_vp']:.3f} | {r['mc_se_vp']:.3f}"
        )
        lines.append(f"| {cells} |")
    lines += ["", "| VS T | mean error (vp) | seed SE | MC SE |", "|---|---|---|---|"]
    for _, r in vs.iterrows():
        cells = (
            f"{r.iloc[0]:g} | {r['mean_error_vp']:+.3f} | {r['seed_se_vp']:.3f} | "
            f"{r['mc_se_vp']:.3f}"
        )
        lines.append(f"| {cells} |")
    lines.append("")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-particles", type=int, default=800_000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[12345, 777, 4242])
    ap.add_argument("--n-paths", type=int, default=400_000)
    ap.add_argument(
        "--price-seeds",
        type=int,
        nargs="+",
        default=[2, 3, 4, 5, 6, 7],
        help="pricing seeds averaged per report (owner decision at M6 Part 0: never one seed)",
    )
    ap.add_argument("--specs", nargs="+", default=["1f"])
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    ap.add_argument(
        "--tail",
        default=None,
        help="override ParticleConfig.tail_extrapolation (e.g. sv_slope, M6 Part 0 item 4)",
    )
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "m6" / "calibration_pass"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = LeverageCache(args.cache)
    lines = [
        f"# M6 Part 0 — calibration-side pass ({args.n_particles} particles, "
        f"seeds {args.seeds}, {args.n_paths} pricing paths x pricing seeds {args.price_seeds})",
        "",
    ]
    for tag in args.specs:
        base = load_yaml(ROOT / f"configs/studies/lsv_reference_{tag}.yaml", CalibrationSpec)
        if args.tail is not None:
            base = dataclasses.replace(
                base, particle=dataclasses.replace(base.particle, tail_extrapolation=args.tail)
            )
        lines.append(
            f"## {tag.upper()} reference set" + (f" (tail {args.tail})" if args.tail else "")
        )
        lines.append("")
        for name in args.variants:
            cal_s, price_s, interp = VARIANTS[name]
            reports = []
            for seed in args.seeds:
                t0 = time.perf_counter()
                rep, t_cal, hit = run_variant(
                    cache,
                    base,
                    args.n_particles,
                    seed,
                    cal_s,
                    price_s,
                    args.n_paths,
                    args.price_seeds,
                    interp,
                )
                reports.append(rep)
                rep.save(out / f"report_{tag}_{name}_seed{seed}.json")
                logging.info(
                    "%s %s seed %d: calibration %s %.0f s, repricing %.0f s | %s",
                    tag,
                    name,
                    seed,
                    "hit" if hit else "miss",
                    t_cal,
                    rep.wall_time,
                    " ".join(
                        f"{k:+.1f}@{T:g}y {e:+.3f}"
                        for T, k, e in rep.vanillas[
                            rep.vanillas["T"].isin(REPORT_T) & rep.vanillas["k"].isin(REPORT_K)
                        ][["T", "k", "error_vp"]].itertuples(index=False)
                    ),
                )
                print(f"[{tag} {name} seed {seed}] {time.perf_counter() - t0:.0f} s", flush=True)
            table, vs = summarise(reports)
            table.to_csv(out / f"summary_{tag}_{name}.csv", index=False)
            van = pd.concat([r.vanillas for r in reports])
            prof = van.groupby(["T", "k"])["error_vp"].mean().unstack("k")
            prof.to_csv(out / f"profile_{tag}_{name}.csv")
            lines += [
                f"Seed-averaged error profile (vp), {name}:",
                "",
                "```",
                prof.round(3).to_string(),
                "```",
                "",
            ]
            vs.to_csv(out / f"varswaps_{tag}_{name}.csv", index=False)
            lines += markdown(name, table, vs)
            (out / "calibration_pass.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
