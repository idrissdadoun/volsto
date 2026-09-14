"""Analytic-versus-simulation break-even table (SPEC §15 Part 3 acceptance gate of the
break-even engine): for a grid of vol-of-vol levels, correlation sets and curves, the
first-order SpotVolCovar / VolVar / skew against the simulation estimate (finite differences of
the ATMF vol under spot and factor shocks on common paths), with the discrepancy versus ν; then
an LSV block on the cached reference leverages (1F ω = 1, 2, 3 and 2F Table 8.2 at 800000
particles, read with ``allow_calibrate=False`` — a cache miss is reported, never calibrated)
with the local branch of the engine (market skew attached) against the same simulation.
Writes ``outputs/m7/breakeven_gate.csv``, ``breakeven_gate_lsv.csv`` and ``.md``.  Both
estimates are first order in the vol of vol in the sense of the closed forms; the simulation is
the truth check.  Every table states its wall clock and that nothing was recalibrated.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from volsto.analytics.breakeven import (
    first_order_breakevens,
    local_slope_from_leverage,
    simulated_breakevens,
)
from volsto.calibration.cache import CacheMissError, LeverageCache, build_market
from volsto.config import BergomiParams, CalibrationSpec, SimConfig, load_yaml
from volsto.market import ForwardCurve, ForwardVarianceCurve
from volsto.models import BergomiSV

ROOT = Path(__file__).resolve().parents[1]
LSV_SPECS = (
    ("1F omega=1", "lsv_reference_1f.yaml", 0.5),
    ("1F omega=2", "lsv_reference_1f.yaml", 1.0),
    ("1F omega=3", "lsv_reference_1f.yaml", 1.5),
    ("2F Table 8.2", "lsv_reference_2f.yaml", None),
)


def lsv_block(n_paths: int, Ts: tuple[float, ...]) -> tuple[pd.DataFrame, list[str]]:
    """The cached reference LSVs (never calibrated here): simulation against the first order
    with the leverage slope and the market skew attached, and against the unrescaled slope."""
    import dataclasses

    sim = SimConfig(n_paths=n_paths, dt_max=1.0 / 100.0, chunk_size=min(n_paths, 100_000), seed=3)
    cache = LeverageCache(ROOT / "cache")
    rows = []
    notes = []
    for label, fname, nu in LSV_SPECS:
        spec = load_yaml(ROOT / "configs" / "studies" / fname, CalibrationSpec)
        if nu is not None:
            spec = dataclasses.replace(spec, model=dataclasses.replace(spec.model, nu=nu))
        spec = dataclasses.replace(
            spec, particle=dataclasses.replace(spec.particle, n_particles=800_000)
        )
        try:
            lsv, _ = cache.get_or_calibrate(spec, allow_calibrate=False)
        except CacheMissError:
            notes.append(f"{label}: not in the cache (800000 particles, code tag m6) - skipped")
            continue
        _, surface, kernel = build_market(spec)
        p = kernel.params
        xi = kernel.xi0
        ls = local_slope_from_leverage(lsv, surface=surface)
        ls0 = local_slope_from_leverage(lsv)
        for T in Ts:
            t0 = time.perf_counter()
            s = simulated_breakevens(lsv, T, sim=sim)
            a = first_order_breakevens(p, xi, T, local=ls, sigma_hat=s.sigma_hat)
            a0 = first_order_breakevens(p, xi, T, local=ls0, sigma_hat=s.sigma_hat)
            rows.append(
                {
                    "model": label,
                    "T": T,
                    "sigma_0": s.sigma_0,
                    "sigma_hat_sim": s.sigma_hat,
                    "sensi_spot_sim": s.sensi_spot,
                    "sensi_spot_se": s.sensi_spot_se,
                    "sensi_spot_an": a.sensi_spot,
                    "sensi_spot_an_unrescaled": a0.sensi_spot,
                    "sensi_spot_rel": a.sensi_spot / s.sensi_spot - 1.0,
                    "svc_sim": s.spot_vol_covar,
                    "svc_se": s.spot_vol_covar_se,
                    "svc_an": a.spot_vol_covar,
                    "svc_rel": a.spot_vol_covar / s.spot_vol_covar - 1.0,
                    "volvar_sim": s.vol_var,
                    "volvar_se": s.vol_var_se,
                    "volvar_an": a.vol_var,
                    "volvar_rel": a.vol_var / s.vol_var - 1.0,
                    "skew_sim": s.skew,
                    "skew_se": s.skew_se,
                    "skew_market": a.skew,
                    "skew_an_unrescaled": a0.skew,
                    "ssr_sim": s.ssr,
                    "ssr_se": s.ssr_se,
                    "ssr_an": a.ssr,
                    "secs": time.perf_counter() - t0,
                }
            )
            print(
                f"LSV {label} T {T}: sensi_spot rel {rows[-1]['sensi_spot_rel']:+.3f}; svc rel "
                f"{rows[-1]['svc_rel']:+.3f}; volvar rel {rows[-1]['volvar_rel']:+.3f}; "
                f"ssr sim {s.ssr:.3f} +- {s.ssr_se:.3f} an {a.ssr:.3f} [{rows[-1]['secs']:.0f}s]",
                flush=True,
            )
    return pd.DataFrame(rows), notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-paths", type=int, default=200_000)
    ap.add_argument("--out", default="outputs/m7")
    ap.add_argument("--lsv-paths", type=int, default=40_000)
    ap.add_argument("--no-lsv", action="store_true")
    args = ap.parse_args()
    fc = ForwardCurve.flat(100.0, 0.0, 0.0)
    curves = {
        "flat 20%": ForwardVarianceCurve.flat(0.04, 10.0),
        "sloping 20%->24%": ForwardVarianceCurve(
            np.array([0.05, 0.25, 1.0, 3.0, 10.0]),
            np.array([0.05, 0.25, 1.0, 3.0, 10.0]) * np.array([0.04, 0.045, 0.05, 0.055, 0.06]),
        ),
    }
    corr_sets = {"Table 8.2": (-0.759, -0.487, 0.0), "rho12 = +0.5": (-0.6, -0.4, 0.5)}
    sim = SimConfig(n_paths=args.n_paths, dt_max=1.0 / 100.0, chunk_size=100_000, seed=3)
    rows = []
    t_all = time.perf_counter()
    for cname, xi in curves.items():
        for sname, (r1, r2, r12) in corr_sets.items():
            for nu in (0.5, 1.0, 1.74, 2.5):
                p = BergomiParams(nu, 0.245, 5.35, 0.28, r12, r1, r2)
                model = BergomiSV(p, xi, fc)
                for T in (0.25, 0.5, 1.0, 2.0):
                    t0 = time.perf_counter()
                    s = simulated_breakevens(model, T, sim=sim)
                    a = first_order_breakevens(p, xi, T, sigma_hat=s.sigma_hat)
                    a0 = first_order_breakevens(p, xi, T)
                    rows.append(
                        {
                            "curve": cname,
                            "correlations": sname,
                            "nu": nu,
                            "T": T,
                            "sigma_hat_sim": s.sigma_hat,
                            "sigma_hat_vs": a0.sigma_hat,
                            "svc_sim": s.spot_vol_covar,
                            "svc_se": s.spot_vol_covar_se,
                            "svc_an": a.spot_vol_covar,
                            "svc_an_vs_prefactor": a0.spot_vol_covar,
                            "svc_z": (a.spot_vol_covar - s.spot_vol_covar) / s.spot_vol_covar_se,
                            "svc_rel": a.spot_vol_covar / s.spot_vol_covar - 1.0,
                            "volvar_sim": s.vol_var,
                            "volvar_se": s.vol_var_se,
                            "volvar_an": a.vol_var,
                            "volvar_z": (a.vol_var - s.vol_var) / s.vol_var_se,
                            "volvar_rel": a.vol_var / s.vol_var - 1.0,
                            "skew_sim": s.skew,
                            "skew_se": s.skew_se,
                            "skew_an": a.skew,
                            "skew_rel": a.skew / s.skew - 1.0,
                            "ssr_sim": s.ssr,
                            "ssr_an": a.ssr,
                            "secs": time.perf_counter() - t0,
                        }
                    )
                    print(
                        f"{cname} {sname} nu {nu} T {T}: svc z {rows[-1]['svc_z']:+.1f} rel "
                        f"{rows[-1]['svc_rel']:+.3f}; volvar z {rows[-1]['volvar_z']:+.1f} rel "
                        f"{rows[-1]['volvar_rel']:+.3f}; skew rel {rows[-1]['skew_rel']:+.3f} "
                        f"[{rows[-1]['secs']:.0f}s]",
                        flush=True,
                    )
    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "breakeven_gate.csv", index=False)
    summary = (
        df.groupby("nu")[["svc_rel", "volvar_rel", "skew_rel"]].agg(["mean", "min", "max"]).round(4)
    )
    lines = [
        "# Break-even engine gate: first-order closed forms vs simulation",
        "",
        f"{args.n_paths} paths, eps 0.05, dt 1/100; prefactor sigma_hat = simulated ATMF vol; "
        f"{time.perf_counter() - t_all:.0f} s wall clock; no calibration involved.",
        "",
        "## Relative discrepancy (analytic / simulation - 1) by nu",
        "",
        summary.to_string(),
        "",
        "## Full table",
        "",
        df.round(5).to_string(index=False),
    ]
    if not args.no_lsv:
        t_lsv = time.perf_counter()
        df_lsv, notes = lsv_block(args.lsv_paths, (0.25, 1.0))
        df_lsv.to_csv(out / "breakeven_gate_lsv.csv", index=False)
        lines += [
            "",
            "## LSV block: cached reference leverages (read from the cache, recalibrated: no)",
            "",
            f"{args.lsv_paths} paths, eps 0.05, dt 1/100; first order with the leverage slope, "
            f"the market skew attached and the simulated ATMF vol as prefactor; "
            f"{time.perf_counter() - t_lsv:.0f} s wall clock.",
            "",
            *notes,
            "",
            df_lsv.round(5).to_string(index=False) if len(df_lsv) else "(no cached model)",
        ]
    (out / "breakeven_gate.md").write_text("\n".join(lines), encoding="utf-8")
    print(summary.to_string())


if __name__ == "__main__":
    main()
