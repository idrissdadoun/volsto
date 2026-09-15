"""M7 real-data run: the 2022 H2 SPX history with the P1 break-even fitter (SPEC §15 Parts 2–4;
owner's "M7 Part 3 FINAL" methodology; no fixed numbers).

Input: ``outputs/m7/hdn_history_ssvi.csv`` (``scripts/m7_hdn_history.py --no-essvi``) and the
snapshot of the pricing date in ``configs/surfaces/snapshots/hdn_2022H2_ssvi``.  Steps: the
historical estimates at the last date (windows 100 / 60 — the sample has 127 trading days, so
the SPEC's 250-day vol window is not available); the **historical-mode** fit (pillars 3M, 6M, 1Y
— MatMin 3M, and the snapshots quote ATM maturities to 1.5–3y only, so the 2y / 3y pillars sit on
the SSVI extrapolation; ``--skew-mode`` / ``--skew-weight`` / ``--skew-eps``); the
**marking-mode** fit on the last snapshot at the ``ssr_target`` dial 0.8 / 1.0 / 1.2 (status,
messages, parameters, constraint gaps); stage 3 on the dial-1.0 fit (**this script calibrates a
leverage** unless ``--no-stage3``, 2·10⁵ particles, 10⁵ pricing paths, and says so); the rolling
stability run in historical mode every ``--every`` dates with the identification flags.  Outputs
in ``outputs/m7/``: ``fit_2f_hdn.md``, ``fit_2f_hdn.yaml`` (marking),
``fit_2f_hdn_historical.yaml``, ``hdn_estimates.csv``, ``hdn_stability.csv``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd

from volsto.calibration.fit_2f import (
    DEFAULT_SKEW_EPS,
    SKEW_MODES,
    BreakEvenFitConfig,
    Stage3Inputs,
    fit_2f_historical,
    fit_2f_marking,
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
    ap.add_argument("--skew-mode", choices=SKEW_MODES, default="twopoint")
    ap.add_argument("--skew-weight", type=float, default=1.0, help="soft mode only")
    ap.add_argument("--skew-eps", type=float, default=DEFAULT_SKEW_EPS)
    ap.add_argument("--max-pillar", type=float, default=1.0)
    ap.add_argument("--n-particles", type=int, default=200_000)
    ap.add_argument("--n-paths", type=int, default=100_000)
    ap.add_argument("--every", type=int, default=5, help="rolling-fit step in dates")
    ap.add_argument("--no-stage3", action="store_true")
    ap.add_argument("--no-stability", action="store_true")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines: list[str] = ["# M7 real-data run - SPX 2022 H2 (HDN sample, P1 break-even fitter)", ""]
    t_all = time.perf_counter()
    hist = SurfaceHistory(pd.read_csv(args.history))
    lines.append(
        f"History: {hist!r}; windows vol {args.window_vol} / SSR {args.window_ssr} increments."
    )
    est = estimate_history(hist, window_vol=args.window_vol, window_ssr=args.window_ssr)
    est.to_frame().to_csv(out / "hdn_estimates.csv", index=False)
    lines += ["", "## Historical estimates at the last date", "", "```", repr(est), "```"]
    pillars = tuple(float(T) for T in hist.pillars if 0.25 - 1e-9 <= T <= args.max_pillar + 1e-9)
    cfg = BreakEvenFitConfig(
        pillars=pillars,
        k2=args.k2,
        skew_mode=args.skew_mode,
        skew_weight=args.skew_weight,
        skew_eps=args.skew_eps,
    )
    lines += ["", f"Fit config: {cfg}"]
    t0 = time.perf_counter()
    r_hist = fit_2f_historical(hist, cfg, window_vol=args.window_vol, window_ssr=args.window_ssr)
    r_hist.write_yaml(out / "fit_2f_hdn_historical.yaml")
    lines += [
        "",
        f"## Historical-mode fit at {hist.dates[-1].date()} ({time.perf_counter() - t0:.1f} s, "
        "recalibrated: no)",
        "",
        "```",
        r_hist.summary(),
        "",
        "k1 profile:",
        r_hist.first.profile.round(5).to_string(index=False),
        "```",
    ]
    date = str(hist.dates[-1].date())
    surface = load_ssvi_surface(Path(args.snapshots) / f"spx_{date}.yaml")
    rows = []
    for s in DIALS:
        stage3 = None
        if s == 1.0 and not args.no_stage3:
            stage3 = Stage3Inputs(
                surface=surface,
                particle=ParticleConfig(n_particles=args.n_particles, horizon=3.0),
                sim=SimConfig(),
                pricing_sim=SimConfig(n_paths=args.n_paths, chunk_size=100_000, seed=7),
                ssr_pillars=pillars,
                breakeven_pillars=(0.25, 1.0),
                forward_starts=((1.0, 2.0),),
            )
        t0 = time.perf_counter()
        r = fit_2f_marking(surface, cfg, ssr_target=s, stage3=stage3)
        p = r.params
        rows.append(
            {
                "ssr_target": s,
                "status": r.status,
                "nu": p.nu,
                "theta": p.theta,
                "k1": p.k1,
                "rho_SX1": p.rho_SX1,
                "rho_SX2": p.rho_SX2,
                "rho12": p.rho12,
                "active": ";".join(r.first.active),
                "svc_err_max": float(abs(r.svc_rel_error).max()),
                "mean_skew_gap": r.mean_skew_gap,
                "wall_s": time.perf_counter() - t0,
                "recalibrated": r.recalibrated,
            }
        )
        if s == 1.0:
            r.write_yaml(out / "fit_2f_hdn.yaml")
            lines += [
                "",
                f"## Marking-mode fit on the {date} snapshot, ssr_target 1 "
                f"({time.perf_counter() - t0:.0f} s; recalibrated: "
                f"{'yes' if r.recalibrated else 'no'})",
                "",
                "```",
                r.summary(),
                "",
                "policy check:",
                r.targets.policy_check().round(4).to_string(index=False),
                "```",
            ]
    lines += [
        "",
        "## ssr_target dial (marking fits, first order)",
        "",
        "```",
        pd.DataFrame(rows).round(4).to_string(index=False),
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
            "date", "k1", "lambda1", "lambda2", "omega1", "omega2", "chi", "nu", "theta",
            "rho_SX1", "rho_SX2", "rho12", "status", "svc_rel_error_max", "active", "bound_flags",
        ]  # fmt: skip
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
