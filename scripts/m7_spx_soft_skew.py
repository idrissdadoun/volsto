"""M7 Part 3 real data: the soft-skew break-even fitter on the SPX 2022-12-30 snapshot (owner's
M7 Part 3 redesign, "REAL DATA"; fitter, measures and messages in the
:mod:`volsto.calibration.fit_2f` module docstring; Bergomi ch. 12 eq. 12.52).

Setting: ``configs/surfaces/snapshots/hdn_2022H2_ssvi/spx_2022-12-30.yaml``, marking mode,
pillars 1M / 3M / 6M / 1Y (the snapshots quote ATM maturities to 1.5–3y only, so the 2y / 3y
pillars are left out as in ``scripts/m7_fit_2f_hdn.py``), ``k2 = 0.2`` fixed, ``ssr_target =
1.0`` (``--ssr-target``), the fitter's default ``ssr_measure`` unless ``--ssr-measure``.  For each
skew weight of ``--weights`` (default 10 / 1 / 0.1, skew-tight to skew-loose):

* the **unclamped** fit at the requested target (``clamp_to_attainable=False``) and the
  **clamped** fit (the fitter's default: a request below the attainable floor — the lowest
  first-order mean SSR over the scan — is refit at the scan target attaining it, with the owner's
  message) — ``--fits`` selects ``unclamped``, ``clamped`` or ``both`` (default); a clamped fit
  without a message is the unclamped fit and is not run through stage 3 twice;
* the attainable SSR floor / ceiling at that weight and the owner's message (first-order
  estimates: the numerical LSV SSR of stage 3 next to them is the truth);
* stage 3 (unless ``--no-stage3``) on every selected fit: the leverage is calibrated by
  :func:`~volsto.calibration.fit_2f.stage3_validation` (not cached; ``ParticleConfig(
  n_particles=200000, horizon=3.0)``, ``SimConfig()`` schedule — the development count), pricing
  ``SimConfig(n_paths=100000, chunk_size=100000, seed=7)``: the numerical LSV SSR per pillar with
  standard errors next to the three first-order achieved SSRs (naked vs market, naked own, lsv),
  the naked mixing skew, the simulated ``SpotVolCovar`` / ``VolVar`` against targets, the actual
  mean ``|L − 1|`` against the first-order proxy.

A historical-mode section (``--history``, default ``outputs/m7/hdn_history_ssvi.csv``; windows
vol 100 / SSR 60 as in the HDN run; first order only, seconds) fits the same weights at the last
date so the SSR_hist < 1 targets can be read next to the marking fits (``--no-historical`` skips
it; skipped with a note when the CSV is absent).

Output: ``<out>/spx_soft_skew.md`` and ``<out>/spx_soft_skew.csv``.  Every report states its wall
clock and whether it recalibrated (stage 3 always recalibrates here).  Full run (under
``caffeinate -i``; about 6 calibrations of 30–60 s plus pricing)::

    caffeinate -i .venv/bin/python scripts/m7_spx_soft_skew.py

Smoke test::

    .venv/bin/python scripts/m7_spx_soft_skew.py --no-stage3 --weights 1 --out /tmp/x
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volsto.calibration.fit_2f import (
    SSR_MEASURES,
    BreakEvenFitConfig,
    FitResult,
    Stage3Inputs,
    attainable_ssr,
    fit_2f_historical,
    fit_2f_marking,
    stage3_validation,
)
from volsto.calibration.history import SurfaceHistory
from volsto.config import ParticleConfig, SimConfig
from volsto.market.loaders import load_ssvi_surface

sys.path.insert(0, str(Path(__file__).resolve().parent))
from m7_skew_tradeoff import (
    fit_row,
    markdown_table,
    pillar_frame,
    stage3_row,
    summary_frame,
)

ROOT = Path(__file__).resolve().parents[1]
DATE = "2022-12-30"
SNAPSHOT = ROOT / "configs" / "surfaces" / "snapshots" / "hdn_2022H2_ssvi" / f"spx_{DATE}.yaml"
PILLARS = (1.0 / 12.0, 0.25, 0.5, 1.0)
K2 = 0.2
WINDOW_VOL = 100
WINDOW_SSR = 60


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--snapshot", default=str(SNAPSHOT))
    ap.add_argument("--ssr-target", type=float, default=1.0)
    ap.add_argument("--weights", default="10,1,0.1", help="skew weights, tight to loose")
    ap.add_argument("--fits", choices=("unclamped", "clamped", "both"), default="both")
    ap.add_argument("--ssr-measure", choices=SSR_MEASURES, default=None, help="default: fitter's")
    ap.add_argument("--no-stage3", action="store_true")
    ap.add_argument("--n-particles", type=int, default=200_000)
    ap.add_argument("--n-paths", type=int, default=100_000)
    ap.add_argument("--mixing-paths", type=int, default=100_000)
    ap.add_argument("--history", default=str(ROOT / "outputs" / "m7" / "hdn_history_ssvi.csv"))
    ap.add_argument("--no-historical", action="store_true")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "m7"))
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    t_all = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = _floats(args.weights)
    surface = load_ssvi_surface(args.snapshot)
    cfg_kw: dict[str, Any] = {"pillars": PILLARS, "k2": K2}
    if args.ssr_measure is not None:
        cfg_kw["ssr_measure"] = args.ssr_measure
    particle = ParticleConfig(n_particles=int(args.n_particles), horizon=3.0)
    pricing = SimConfig(n_paths=int(args.n_paths), chunk_size=100_000, seed=7)
    s3_inputs = Stage3Inputs(
        surface=surface,
        particle=particle,
        sim=SimConfig(),
        pricing_sim=pricing,
        ssr_pillars=PILLARS,
        breakeven_pillars=PILLARS,
        forward_starts=(),
        mixing_paths=int(args.mixing_paths),
    )
    kinds = ("unclamped", "clamped") if args.fits == "both" else (args.fits,)

    rows: list[dict[str, Any]] = []
    details: list[str] = []
    for w in weights:
        cfg = BreakEvenFitConfig(skew_weight=float(w), clamp_to_attainable=False, **cfg_kw)
        t0 = time.perf_counter()
        fits: dict[str, FitResult] = {
            "unclamped": fit_2f_marking(surface, cfg, ssr_target=args.ssr_target, proxy=True),
            "clamped": fit_2f_marking(
                surface,
                dataclasses.replace(cfg, clamp_to_attainable=True),
                ssr_target=args.ssr_target,
                proxy=True,
            ),
        }
        att = fits["clamped"].attainable or attainable_ssr(
            fits["unclamped"].targets, cfg, fits["unclamped"].xi0
        )
        fo_s = time.perf_counter() - t0
        msg = fits["clamped"].message
        print(f"[weight {w:g}] first order {fo_s:.1f} s; message: {msg or 'none'}", flush=True)
        for kind in kinds:
            clamped_differs = bool(
                np.any(
                    np.abs(
                        fits["clamped"].table["ssr_target"].to_numpy()
                        - fits["unclamped"].table["ssr_target"].to_numpy()
                    )
                    > 1e-12
                )
            )
            if kind == "clamped" and not clamped_differs and "unclamped" in kinds:
                details += [
                    "",
                    f"### skew_weight {w:g}, clamped: target tracked (same as the unclamped fit)",
                ]
                continue
            t1 = time.perf_counter()
            r = fits[kind]
            row: dict[str, Any] = {"fit": kind, "ssr_target": float(args.ssr_target)}
            row.update(fit_row(r, att, fits["clamped"]))
            row["recalibrated"] = False
            row["calibration_seconds"] = 0.0
            block = [
                "",
                f"### skew_weight {w:g}, {kind} fit",
                "",
                "```",
                r.summary(),
                "",
                att.summary(),
                "",
                "owner's message (clamped fit): " + (msg or "none (the target is tracked)"),
                *(f"  - {d}" for d in fits["clamped"].message_details),
                "```",
            ]
            if not args.no_stage3:
                s3 = stage3_validation(r.params, s3_inputs, r.targets, fit_table=r.table)
                row.update(stage3_row(r, s3))
                row["recalibrated"] = s3.recalibrated
                row["calibration_seconds"] = s3.calibration_seconds
                block += ["", "```", s3.summary(), "```"]
                print(
                    f"  {kind}: calibration {s3.calibration_seconds:.0f} s, stage 3 "
                    f"{s3.wall_seconds:.0f} s; mean |L-1| {s3.mean_abs_l_minus_1:.3f}; numerical "
                    "SSR "
                    + ", ".join(
                        f"{x.T:g}: {x.ssr_model:.3f}±{x.ssr_model_se:.3f}"
                        for x in s3.ssr_table.itertuples()
                    ),
                    flush=True,
                )
            row["wall_seconds"] = time.perf_counter() - t1 + (fo_s if kind == kinds[0] else 0.0)
            rows.append(row)
            details += block

    df = pd.DataFrame(rows)
    df.to_csv(out / "spx_soft_skew.csv", index=False)

    hist_lines: list[str] = []
    if not args.no_historical:
        hp = Path(args.history)
        if not hp.exists():
            hist_lines = ["", "## Historical mode", "", f"Skipped: {hp} not found."]
        else:
            th = time.perf_counter()
            hist = SurfaceHistory(pd.read_csv(hp))
            hpillars = tuple(float(T) for T in hist.pillars if float(T) <= 1.0 + 1e-9)
            hrows = []
            hblocks: list[str] = []
            for w in weights:
                hc = BreakEvenFitConfig(skew_weight=float(w), **{**cfg_kw, "pillars": hpillars})
                rh = fit_2f_historical(hist, hc, window_vol=WINDOW_VOL, window_ssr=WINDOW_SSR)
                hrows.append(
                    {
                        "w_skew": float(w),
                        "ssr_requested_mean": float(rh.ssr_requested.mean()),
                        "ssr_fitted_mean": float(rh.targets.ssr_target.mean()),
                        "SSR lsv": rh.ssr_achieved_lsv_mean,
                        "SSR naked/mkt": rh.ssr_achieved_naked_mean,
                        "SSR naked own": float(rh.table["ssr_naked_own"].mean()),
                        "skew gap": rh.mean_skew_gap,
                        "nu": rh.params.nu,
                        "k1": rh.params.k1,
                        "rho_SX1": rh.params.rho_SX1,
                        "rho_SX2": rh.params.rho_SX2,
                        "message": "yes" if rh.message else "no",
                    }
                )
                hblocks += [
                    "",
                    f"### historical, skew_weight {w:g}",
                    "",
                    "```",
                    rh.summary(),
                    "```",
                ]
            hist_lines = [
                "",
                f"## Historical mode at {hist.dates[-1].date()} (windows vol {WINDOW_VOL} / SSR "
                f"{WINDOW_SSR}; pillars {[round(t, 4) for t in hpillars]}; clamped fitter default;"
                f" {time.perf_counter() - th:.1f} s; recalibrated: no)",
                "",
                markdown_table(pd.DataFrame(hrows)),
                *hblocks,
            ]

    total = time.perf_counter() - t_all
    any_recal = bool(df["recalibrated"].any()) if len(df) else False
    measure = rows[0]["ssr_measure"] if rows else "?"
    head = summary_frame(df, PILLARS) if len(df) else pd.DataFrame()
    if len(df):
        head.insert(0, "fit", df["fit"])
    lines = [
        f"# M7 Part 3 - soft-skew fitter on SPX {DATE} (marking mode)",
        "",
        f"Snapshot `{args.snapshot}`, pillars {[round(t, 4) for t in PILLARS]}, k2 fixed "
        f"{K2:g}, ssr_target {args.ssr_target:g}, ssr_measure `{measure}`, skew weights "
        f"{weights}, fits {list(kinds)}.",
        (
            "Stage 3 skipped (`--no-stage3`): first order only, nothing calibrated."
            if args.no_stage3
            else f"Stage 3: leverage calibrated per fit (not cached) at {args.n_particles} "
            f"particles (ParticleConfig(horizon=3.0), SimConfig() schedule; the development "
            f"count), pricing {args.n_paths} paths (seed 7), mixing skew {args.mixing_paths} "
            "paths."
        ),
        f"Total wall clock {total:.0f} s; recalibrated: {'yes' if any_recal else 'no'}.",
        "",
        "## Summary (SSR columns: first-order achieved means; LSV num = stage 3 with se)",
        "",
        markdown_table(head) if len(df) else "(no rows)",
        "",
        "## Per pillar (fo = first order; num = LSV numerical with standard error)",
        "",
        (
            markdown_table(
                pd.concat(
                    [
                        pd.DataFrame({"fit": df["fit"].repeat(len(PILLARS)).to_numpy()}),
                        pillar_frame(df, PILLARS),
                    ],
                    axis=1,
                )
            )
            if len(df)
            else ""
        ),
        "",
        "## Floor messages",
        "",
        *(
            f"- weight {x['skew_weight']:g}: floor {x['ssr_floor']:.3f}, ceiling "
            f"{x['ssr_ceiling']:.3f}; " + (x["message"] or "none (tracked)")
            for _, x in df.drop_duplicates("skew_weight").iterrows()
        ),
        *hist_lines,
        "",
        "## Details",
        *details,
    ]
    (out / "spx_soft_skew.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'spx_soft_skew.md'} and .csv; total wall clock {total:.0f} s")


if __name__ == "__main__":
    main()
