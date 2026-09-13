"""M7 real-data run: the 2022 H2 SPX history (SPEC §15 Parts 2–4 end to end, no fixed numbers).

Input: ``outputs/m7/hdn_history_ssvi.csv`` (``scripts/m7_hdn_history.py --no-essvi``) and the
snapshot of the pricing date in ``configs/surfaces/snapshots/hdn_2022H2_ssvi``.  Steps: the
historical estimates at the last date (windows 100 / 60 — the sample has 127 trading days, so
the SPEC's 250-day vol window is not available and every number is reported with the window it
was measured on); ``fit_2f`` stages 1–2 (ρ12 from the 3m/2y correlation, mixing refinement);
stage 3 on the pricing date's surface (2·10⁵ particles — the development count — and 2·10⁵
pricing paths); the rolling stability run (stage 1 every 21 days, stage 2 every 5 days, from the
first date with 100 increments behind it) with the identification flags; the ν ⟷ correlations
degeneracy profile.  Outputs in ``outputs/m7/``: ``fit_2f_hdn.md``, ``fit_2f_hdn.yaml``,
``hdn_estimates.csv``, ``hdn_stability.csv``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from volsto.calibration.fit_2f import Fit2FConfig, Stage3Inputs, fit_2f
from volsto.calibration.history import SurfaceHistory, estimate_history
from volsto.calibration.stability import flag_unidentified, rolling_fit
from volsto.config import ParticleConfig, SimConfig
from volsto.market.loaders import load_ssvi_surface


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="outputs/m7/hdn_history_ssvi.csv")
    ap.add_argument("--snapshots", default="configs/surfaces/snapshots/hdn_2022H2_ssvi")
    ap.add_argument("--out", default="outputs/m7")
    ap.add_argument("--window-vol", type=int, default=100)
    ap.add_argument("--window-ssr", type=int, default=60)
    ap.add_argument("--n-particles", type=int, default=200_000)
    ap.add_argument(
        "--max-pillar",
        type=float,
        default=1.0,
        help="longest pillar used by the fit (the sample quotes ATM maturities to 1.5-3y only)",
    )
    ap.add_argument("--n-paths", type=int, default=200_000)
    ap.add_argument("--no-stage3", action="store_true")
    ap.add_argument("--no-stability", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# M7 real-data run — SPX 2022 H2 (HDN sample, plain SSVI snapshots)", ""]
    t_all = time.perf_counter()
    hist = SurfaceHistory(pd.read_csv(args.history))
    lines.append(
        f"History: {hist!r}; windows vol {args.window_vol} / SSR {args.window_ssr} increments."
    )
    est = estimate_history(hist, window_vol=args.window_vol, window_ssr=args.window_ssr)
    est.to_frame().to_csv(out / "hdn_estimates.csv", index=False)
    lines += ["", "## Historical estimates at the last date", "", "```", repr(est), "```"]
    pillars = tuple(float(T) for T in hist.pillars if args.max_pillar + 1e-9 >= T)
    lines += [
        "",
        f"Pillars used by the fit: {list(pillars)} (the snapshots quote ATM maturities to 1.5-3y "
        "only, median 2y, so the 2y/3y pillars sit on the SSVI extrapolation and their daily "
        "changes are an artefact: annualised std of d ln vs_vol 0.46 / 0.62 against 0.24 at 1y, "
        "cross-pillar correlation with the 1y pillar 0.07 / -0.01).",
    ]
    cfg = Fit2FConfig(
        pillars=pillars,
        corr_pillars=(0.25, min(1.0, pillars[-1])),
        window_vol=args.window_vol,
        window_ssr=args.window_ssr,
        rho12_mode="from_correlation",
    )
    cfg_skew = Fit2FConfig(**{**cfg.__dict__, "ssr_scale": 1e6})
    stage3 = None
    if not args.no_stage3:
        date = str(hist.dates[-1].date())
        surface = load_ssvi_surface(Path(args.snapshots) / f"spx_{date}.yaml")
        stage3 = Stage3Inputs(
            surface=surface,
            particle=ParticleConfig(n_particles=args.n_particles, horizon=3.0),
            sim=SimConfig(),
            pricing_sim=SimConfig(n_paths=args.n_paths, chunk_size=100_000, seed=7),
            headline=False,
        )
    t0 = time.perf_counter()
    result = fit_2f(hist, None, cfg, stage3=stage3)
    lines += [
        "",
        f"## fit_2f as specified: skew term structure + SSR (stages 1-3, "
        f"{time.perf_counter() - t0:.0f} s)",
        "",
        "```",
        result.summary(),
        "```",
    ]
    t0 = time.perf_counter()
    result_skew = fit_2f(hist, None, cfg_skew, stage3=stage3)
    lines += [
        "",
        f"## fit_2f variant: skew term structure only (SSR reported, not fitted; "
        f"{time.perf_counter() - t0:.0f} s)",
        "",
        "```",
        result_skew.summary(),
        "```",
    ]
    result_skew.write_yaml(out / "fit_2f_hdn_skew_only.yaml")
    lines += [
        "",
        "### nu <-> correlations degeneracy profile",
        "",
        "```",
        result.degeneracy.round(4).to_string(index=False),
        "```",
    ]
    result.write_yaml(out / "fit_2f_hdn.yaml")
    if not args.no_stability:
        t0 = time.perf_counter()
        frame = rolling_fit(hist, cfg, stage1_every=21, stage2_every=5, refine_mixing=False)
        frame.to_csv(out / "hdn_stability.csv", index=False)
        flags = flag_unidentified(frame)
        n_refit = int(frame["stage1_refit"].sum())
        cols = [
            "date",
            "nu",
            "theta",
            "k1",
            "k2",
            "rho12",
            "rho_SX1",
            "rho_SX2",
            "chi",
            "stage1_objective",
            "stage2_objective",
            "stage1_refit",
        ]
        lines += [
            "",
            f"## Rolling stability ({len(frame)} stage-2 dates, {n_refit} stage-1 refits, "
            f"{time.perf_counter() - t0:.0f} s; order-one skew, no mixing refinement)",
            "",
            "```",
            frame[cols].round(4).to_string(index=False),
            "",
            flags.round(4).to_string(index=False),
            "```",
        ]
    lines += ["", f"Total wall clock {time.perf_counter() - t_all:.0f} s."]
    (out / "fit_2f_hdn.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
