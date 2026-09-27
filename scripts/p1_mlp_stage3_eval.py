"""The desk's most-likely-path break-evens (``volsto.analytics.p1_mlp``) against the stage-3
simulations already on disk (SPEC §15 Part 3, *The note's engine*): no fit, no Monte Carlo, no
calibration.  For each simulated parameter set of ``outputs/step0_stage3`` and
``outputs/step0_stage3_caps`` and ``outputs/step0_stage3_atmf`` (five dates of 2022 H2;
today's and the new marking default, the new one also at ν caps 4.0 and 4.5; the refits with
the ATMF kernels), SpotVolCovar and VolVar at the stored parameters, in
volsto's absolute units and with volsto's ``σ_0`` (the simulations' normalisation), by:

* ``volsto`` — volsto's first-order engine (the naked kernels on the variance-swap forward
  variance) with the pricing surface's skew, the one the calibrated LSV reproduces (the stored
  first-order values of the SABRW-step-0 fits read the SABRW skew in the leverage term);
* ``note_fo`` — the note's first-order terms (the naked kernels on the ATMF forward variance,
  SensiX / SensiY = ``½ ω_i A_i``);
* ``note`` — the note's closed forms (eqs. 33–35, 43, 57, appendix D).

Outputs in ``--out``: ``compare.csv`` and ``summary.md``."""

from __future__ import annotations

import argparse
import importlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from volsto.analytics.p1_mlp import MlpBreakEvens, mlp_breakevens, mlp_setup
from volsto.analytics.reparam import to_breakeven
from volsto.calibration.targets import SIGMA0_MATURITY
from volsto.config import BergomiParams
from volsto.market.loaders import load_ssvi_surface
from volsto.market.varswap import xi0_curve

f2 = importlib.import_module("volsto.calibration.fit_2f")

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "configs" / "surfaces" / "snapshots" / "hdn_2022H2"
RUNS = ("outputs/step0_stage3", "outputs/step0_stage3_caps", "outputs/step0_stage3_atmf")
ENGINES = ("volsto", "note_fo", "note")


def volsto_first_order(surf, p, k2: float, s0m: float) -> dict[float, tuple[float, float]]:  # type: ignore[no-untyped-def]
    """volsto's first-order SpotVolCovar and VolVar per pillar at ``p`` (surface skew), ``σ_0``
    the ATMF vol at ``s0m``."""
    cfg = f2.BreakEvenFitConfig(
        skew_eps=0.10,
        k2=k2,
        k1_bounds=(k2 + 0.1, 100.0),
        sigma0_maturity=None if s0m == SIGMA0_MATURITY else s0m,
    )
    targets = f2.marking_targets_for(surf, cfg, ssr_target=1.0)
    xi0 = xi0_curve(surf, float(min(surf.max_maturity, max(targets.pillars))))
    prob, _ = f2._first_problem(targets, cfg, xi0)
    mm = prob.maps(p.k1)
    lam = np.array([p.lambda1, p.lambda2])
    svc = mm.svc(lam)
    vv, _ = f2.volvar_p1(
        np.array([p.omega1, p.omega2, p.chi]), lam, mm.naked.A, prob.atf, mm.sensi_spot(lam)
    )
    return {float(T): (float(svc[i]), float(vv[i])) for i, T in enumerate(prob.T)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "outputs" / "p1_mlp_eval"))
    ap.add_argument("--runs", nargs="*", default=list(RUNS), help="stage-3 output directories")
    ap.add_argument(
        "--sigma0-maturity",
        type=float,
        default=SIGMA0_MATURITY,
        help="the ATMF maturity of the simulations' sigma_0 (the runs' sigma0_maturity)",
    )
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    fits = pd.concat([pd.read_csv(ROOT / r / "fits.csv") for r in a.runs], ignore_index=True)
    checks = pd.concat([pd.read_csv(ROOT / r / "check.csv") for r in a.runs], ignore_index=True)
    fits = fits.drop_duplicates(["date", "variant"])
    checks = checks.drop_duplicates(["date", "variant", "T", "quantity"])
    rows = []
    for f in fits.itertuples():
        surf = load_ssvi_surface(SNAPSHOTS / f"spx_{f.date}.yaml")
        sigma_0 = float(surf.atm_vol(a.sigma0_maturity))
        p = to_breakeven(
            BergomiParams(
                nu=f.nu,
                theta=f.theta,
                k1=f.k1,
                k2=f.k2,
                rho12=f.rho12,
                rho_SX1=f.rho_sx1,
                rho_SX2=f.rho_sx2,
            )
        )
        vfo = volsto_first_order(surf, p, float(f.k2), a.sigma0_maturity)
        mine = checks[(checks["date"] == f.date) & (checks["variant"] == f.variant)]
        for T in sorted(mine["T"].unique()):
            be = mlp_breakevens(surf, p, float(T), sigma_0=sigma_0)
            s = mlp_setup(surf, p, float(T))
            lead = MlpBreakEvens(
                be.T,
                be.atmf_vol,
                be.implied_vol,
                be.sensi_spot,
                s.leading_sensi("X"),
                s.leading_sensi("Y"),
                be.rho_sx,
                be.rho_sy,
                be.rho_xy,
            )
            vals = {"volsto": vfo[float(T)], "note_fo": lead.absolute(), "note": be.absolute()}
            for iq, q in enumerate(("SpotVolCovar", "VolVar")):
                c = mine[(mine["T"] == T) & (mine["quantity"] == q)].iloc[0]
                row = dict(
                    date=f.date,
                    variant=f.variant,
                    nu=f.nu,
                    T=T,
                    quantity=q,
                    sim=c["sim"],
                    se=c["se"],
                    target=c["target"],
                    stored_first_order=c["first_order"],
                    sensi_x_ratio=be.sensi_x / lead.sensi_x,
                    sensi_y_ratio=be.sensi_y / lead.sensi_y,
                )
                for e in ENGINES:
                    row[e] = vals[e][iq]
                    row[f"bias_{e}"] = c["sim"] / vals[e][iq] - 1.0
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out / "compare.csv", index=False)
    wall = time.perf_counter() - t0
    today = df[df["variant"] == "today"]
    chk = (today["volsto"] / today["stored_first_order"] - 1.0).abs().max()
    lines = [
        "# The note's most-likely-path engine against the stage-3 simulations",
        "",
        f"{df[['date', 'variant']].drop_duplicates().shape[0]} parameter sets x "
        f"{df['T'].nunique()} pillars; wall {wall:.0f} s; recalibrated: no (the simulations are "
        f"those on disk). volsto's stored first-order values of today's fits reproduced to "
        f"{chk:.1e} relative.",
        "",
        "Bias = simulated / engine - 1 (simulation standard errors "
        f"{100 * (df['se'] / df['sim']).abs().min():.1f} to "
        f"{100 * (df['se'] / df['sim']).abs().max():.1f} % of the values).",
        "",
        "| quantity | engine | mean abs bias | range | cells within 10 % |",
        "|---|---|---|---|---|",
    ]
    names = {
        "volsto": "volsto first order (VS kernels)",
        "note_fo": "note, first-order terms (ATMF kernels)",
        "note": "note, closed forms",
    }
    for q in ("SpotVolCovar", "VolVar"):
        g = df[df["quantity"] == q]
        for e in ENGINES:
            b = 100 * g[f"bias_{e}"]
            lines.append(
                f"| {q} | {names[e]} | {b.abs().mean():.1f} % | {b.min():+.1f} to {b.max():+.1f} % "
                f"| {int((b.abs() <= 10).sum())} of {len(b)} |"
            )
    lines += [
        "",
        "| date | variant | nu | T | quantity | sim | volsto | note FO | note | bias volsto | "
        "bias note FO | bias note | SensiX ratio | SensiY ratio |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in df.itertuples():
        lines.append(
            f"| {r.date} | {r.variant} | {r.nu:.2f} | {r.T:g} | {r.quantity} | {r.sim:.5f} | "
            f"{r.volsto:.5f} | {r.note_fo:.5f} | {r.note:.5f} | {100 * r.bias_volsto:+.1f} % | "
            f"{100 * r.bias_note_fo:+.1f} % | {100 * r.bias_note:+.1f} % | {r.sensi_x_ratio:.3f} | "
            f"{r.sensi_y_ratio:.3f} |"
        )
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
