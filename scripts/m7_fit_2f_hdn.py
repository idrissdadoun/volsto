"""M7 real-data run: the 2022 H2 SPX history with the break-even fitter (SPEC §15 Parts 2–4,
M7 addendum; no fixed numbers).

Input: ``outputs/m7/hdn_history_ssvi.csv`` (``scripts/m7_hdn_history.py --no-essvi``) and the
snapshot of the pricing date in ``configs/surfaces/snapshots/hdn_2022H2_ssvi``.  Steps: the
historical estimates at the last date (windows 100 / 60 — the sample has 127 trading days, so
the SPEC's 250-day vol window is not available and every number is reported with the window it
was measured on); the **historical-mode** fit (pillars 1m, 3m, 6m, 1y; ``k2`` fixed); the
**marking-mode** fit on the last snapshot (same pillars, ``ssr_target`` 1.0, the soft-skew fitter
at ``--skew-weight`` / ``--ssr-measure``, with the floor / ceiling message when it fires) with the
SSR dial table (0.8 / 1.0 / 1.2: fitted parameters, achieved SSR under both measures, the
message and the naked kernel's numerical SSR by ``ssr_numerical_many`` with 4·10⁴ paths); the
skew-weight trade-off at ``ssr_target`` over ``--tradeoff-weights`` (achieved SSR, naked-skew
gap, leverage proxy; with ``--tradeoff-stage3`` a leverage is calibrated per weight for the
actual mean ``|L − 1|`` and the numerical LSV SSR); stage 3 on the marking fit (2·10⁵ particles —
the development count — and 2·10⁵ pricing paths; **this script calibrates leverages** and says
so); the rolling stability run in historical mode every 5 dates with the identification flags.
Outputs in ``outputs/m7/``: ``fit_2f_hdn.md``, ``fit_2f_hdn.yaml`` (marking),
``fit_2f_hdn_historical.yaml``, ``hdn_estimates.csv``, ``hdn_stability.csv``,
``hdn_skew_tradeoff.csv``.

Data notes carried from the first M7 run: the snapshots quote ATM maturities to 1.5–3y only
(median 2y), so the 2y / 3y pillars sit on the SSVI extrapolation and their daily changes are
an artefact — the fit uses the 1m–1y pillars; over H2 2022 the historical SSR read 0.81–0.84
(60 days) at every pillar to 1y, below the floor ``R ≥ 1`` of a naked or local-vol-type model,
so the historical-mode ``SpotVolCovar`` target asks for less spot/vol covariance than the
kernel's own skew implies (the fit's tracking detail reports the miss; the floor / ceiling message
fires only when the request lies outside the attainable band of ``attainable_ssr``).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from volsto.analytics.smile_dynamics import ssr_numerical_many
from volsto.calibration.fit_2f import (
    DEFAULT_SKEW_WEIGHT,
    SSR_MEASURE_CHOICES,
    BreakEvenFitConfig,
    Stage3Inputs,
    fit_2f_historical,
    fit_2f_marking,
    naked_kernel,
    skew_weight_tradeoff,
)
from volsto.calibration.history import SurfaceHistory, estimate_history
from volsto.calibration.stability import flag_unidentified, rolling_fit
from volsto.config import ParticleConfig, SimConfig
from volsto.market.loaders import load_ssvi_surface

DIALS = (0.8, 1.0, 1.2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", default="outputs/m7/hdn_history_ssvi.csv")
    ap.add_argument("--snapshots", default="configs/surfaces/snapshots/hdn_2022H2_ssvi")
    ap.add_argument("--out", default="outputs/m7")
    ap.add_argument("--window-vol", type=int, default=100)
    ap.add_argument("--window-ssr", type=int, default=60)
    ap.add_argument("--k2", type=float, default=0.2)
    ap.add_argument("--ssr-target", type=float, default=1.0)
    ap.add_argument(
        "--prefactor",
        choices=("market", "model"),
        default="market",
        help="sigma_hat prefactor convention of the fit (fit_2f module docstring)",
    )
    ap.add_argument("--skew-weight", type=float, default=DEFAULT_SKEW_WEIGHT)
    ap.add_argument("--ssr-measure", choices=SSR_MEASURE_CHOICES, default="auto")
    ap.add_argument(
        "--tradeoff-weights",
        default="1,0.1",
        help="comma-separated skew weights of the marking trade-off table ('' to skip)",
    )
    ap.add_argument(
        "--tradeoff-stage3",
        action="store_true",
        help="calibrate a leverage per trade-off weight (actual mean |L - 1|, numerical SSR)",
    )
    ap.add_argument("--n-particles", type=int, default=200_000)
    ap.add_argument("--n-paths", type=int, default=200_000)
    ap.add_argument(
        "--max-pillar",
        type=float,
        default=1.0,
        help="longest pillar used by the fits (the sample quotes ATM maturities to 1.5-3y only)",
    )
    ap.add_argument("--every", type=int, default=5, help="rolling-fit step in dates")
    ap.add_argument("--no-stage3", action="store_true")
    ap.add_argument("--no-stability", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# M7 real-data run - SPX 2022 H2 (HDN sample, break-even fitter)", ""]
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
        f"Pillars used by the fits: {list(pillars)} (the snapshots quote ATM maturities to "
        "1.5-3y only, median 2y, so the 2y/3y pillars sit on the SSVI extrapolation and their "
        "daily changes are an artefact: annualised std of d ln vs_vol 0.46 / 0.62 against 0.24 "
        "at 1y, cross-pillar correlation with the 1y pillar 0.07 / -0.01).  SSR_hist < 1 over "
        "H2 2022 (0.81-0.84 on 60 days): below the R >= 1 of a naked kernel, so the historical "
        "SpotVolCovar target is missed (see the tracking detail and the attainable band).",
        f"k2 fixed at {args.k2:g}; sigma_hat prefactor convention: {args.prefactor}; skew weight "
        f"{args.skew_weight:g}; SSR measure {args.ssr_measure}.",
    ]
    cfg = BreakEvenFitConfig(
        pillars=pillars,
        k2=args.k2,
        sigma_hat_prefactor=args.prefactor,
        skew_weight=args.skew_weight,
        ssr_measure=args.ssr_measure,
    )
    # historical mode
    t0 = time.perf_counter()
    r_hist = fit_2f_historical(hist, cfg, window_vol=args.window_vol, window_ssr=args.window_ssr)
    r_hist.write_yaml(out / "fit_2f_hdn_historical.yaml")
    lines += [
        "",
        f"## Historical-mode fit at {hist.dates[-1].date()} ({time.perf_counter() - t0:.1f} s, "
        f"recalibrated: no)",
        "",
        "```",
        r_hist.summary(),
        "",
        "k1 profile:",
        r_hist.first.profile.round(5).to_string(index=False),
        "```",
    ]
    # marking mode on the last snapshot, with stage 3
    date = str(hist.dates[-1].date())
    surface = load_ssvi_surface(Path(args.snapshots) / f"spx_{date}.yaml")
    stage3 = None
    if not args.no_stage3:
        stage3 = Stage3Inputs(
            surface=surface,
            particle=ParticleConfig(n_particles=args.n_particles, horizon=3.0),
            sim=SimConfig(),
            pricing_sim=SimConfig(n_paths=args.n_paths, chunk_size=100_000, seed=7),
            ssr_pillars=pillars,
            breakeven_pillars=(0.25, 1.0),
            headline=False,
        )
    t0 = time.perf_counter()
    r_mark = fit_2f_marking(surface, cfg, ssr_target=args.ssr_target, stage3=stage3)
    r_mark.write_yaml(out / "fit_2f_hdn.yaml")
    lines += [
        "",
        f"## Marking-mode fit on the {date} snapshot, ssr_target {args.ssr_target:g} "
        f"({time.perf_counter() - t0:.0f} s incl. stage 3; recalibrated: "
        f"{'yes' if r_mark.recalibrated else 'no'})",
        "",
        "```",
        r_mark.summary(),
        "",
        "policy check:",
        r_mark.targets.policy_check().round(4).to_string(index=False),
        "```",
    ]
    if r_mark.stage3 is not None:
        s3 = r_mark.stage3
        lines += [
            "",
            f"Stage 3: calibration {s3.calibration_seconds:.0f} s at {s3.n_particles} "
            f"particles, wall clock {s3.wall_seconds:.0f} s, recalibrated: "
            f"{'yes' if s3.recalibrated else 'no'}.",
        ]
    # the SSR dial
    t0 = time.perf_counter()
    sim = SimConfig(n_paths=40_000, dt_max=1.0 / 100.0, chunk_size=20_000, seed=7)
    rows = []
    for s in DIALS:
        r = fit_2f_marking(surface, cfg, ssr_target=s)
        num = ssr_numerical_many(naked_kernel(r, surface.forward_curve), pillars, eps=0.05, sim=sim)
        row: dict[str, object] = {
            "ssr_target": s,
            "nu": r.params.nu,
            "theta": r.params.theta,
            "k1": r.params.k1,
            "rho_SX1": r.params.rho_SX1,
            "rho_SX2": r.params.rho_SX2,
            "rho12": r.params.rho12,
            "chi": r.breakeven.chi,
            "nu_limit": r.first.nu_limit_binding,
            "ssr_fitted": float(r.targets.ssr_target.mean()),
            "ssr_achieved_lsv": r.ssr_achieved_lsv_mean,
            "ssr_achieved_naked": r.ssr_achieved_naked_mean,
            "skew_gap": r.mean_skew_gap,
            "message": "yes" if r.message else "no",
        }
        for x in num:
            row[f"ssr_num_{x.T:g}"] = x.R
            row[f"se_{x.T:g}"] = x.R_stderr
        rows.append(row)
    lines += [
        "",
        f"## SSR dial (marking fits; naked kernel numerical SSR, {sim.n_paths} paths, "
        f"dt {sim.dt_max}; {time.perf_counter() - t0:.0f} s, recalibrated: no)",
        "",
        "```",
        pd.DataFrame(rows).round(4).to_string(index=False),
        "```",
    ]
    weights = [float(x) for x in args.tradeoff_weights.split(",") if x.strip()]
    if weights:
        t0 = time.perf_counter()
        s3_tr = None
        if args.tradeoff_stage3:
            s3_tr = Stage3Inputs(
                surface=surface,
                particle=ParticleConfig(n_particles=args.n_particles, horizon=3.0),
                sim=SimConfig(),
                pricing_sim=SimConfig(n_paths=args.n_paths, chunk_size=100_000, seed=7),
                ssr_pillars=pillars,
                breakeven_pillars=(0.25, 1.0),
                forward_starts=(),
            )
        trade = skew_weight_tradeoff(
            surface, cfg, weights=weights, ssr_target=args.ssr_target, stage3=s3_tr
        )
        trade.to_csv(out / "hdn_skew_tradeoff.csv", index=False)
        lines += [
            "",
            f"## Skew-weight trade-off at ssr_target {args.ssr_target:g} ({len(weights)} weights, "
            f"{time.perf_counter() - t0:.0f} s, recalibrated: "
            f"{'yes' if bool(trade['recalibrated'].any()) else 'no'})",
            "",
            "```",
            trade.drop(columns=["message"]).round(4).T.to_string(),
            "```",
        ]
    if not args.no_stability:
        t0 = time.perf_counter()
        frame = rolling_fit(
            hist, cfg, every=args.every, window_vol=args.window_vol, window_ssr=args.window_ssr
        )
        frame.to_csv(out / "hdn_stability.csv", index=False)
        flags = flag_unidentified(frame)
        cols = [
            "date",
            "k1",
            "lambda1",
            "lambda2",
            "omega1",
            "omega2",
            "chi",
            "nu",
            "theta",
            "rho_SX1",
            "rho_SX2",
            "rho12",
            "first_objective",
            "second_objective",
            "n_active",
            "bound_flags",
        ]
        lines += [
            "",
            f"## Rolling stability (historical mode every {args.every} dates, {len(frame)} fits, "
            f"{time.perf_counter() - t0:.0f} s, recalibrated: no)",
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
