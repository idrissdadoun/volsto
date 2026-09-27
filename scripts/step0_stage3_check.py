"""Stage-3 check of the marking default the owner chose on 2026-09-27 (SPEC §15 Part 3, *Owner's
decisions on the measurement*): the first-order engine's bias at the fitted parameters, by
simulation, for today's default (step 0 from the surface, ``k2`` 0.2) and the new one (step 0 from
the SABRW fits with the quote rule, ``k2`` fitted), on a few dates of the 2022 H2 SPX sample.

**This script calibrates** a leverage per fit not already in the cache (``cache/``, reference
particle settings, ``--n-particles``) and says so per row.  For each date: the stored snapshot
(``configs/surfaces/snapshots/hdn_2022H2``), the SABRW fits of the day's quotes
(:func:`volsto.market.import_hdn.sabrw_fits`), the marking fit at SSR 1 and ``ε`` 0.10 under both
variants, then :func:`~volsto.calibration.fit_2f.stage3_validation` at the fitted parameters —
the protocol of the M7 Part 3 report (break-even pillars 3M and 1Y, the engine bias
``simulated / first-order − 1`` against the 10 % tolerance, the target gaps reported).  Outputs
in ``--out`` (git-ignored): ``check.csv`` (date × variant × pillar × quantity), ``fits.csv``
(date × variant) and ``summary.md``.
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volsto.calibration.cache import LeverageCache, build_market
from volsto.calibration.fit_2f import (
    DEFAULT_NU_CAP,
    BreakEvenFitConfig,
    Stage3Inputs,
    fit_2f_marking,
    stage3_validation,
)
from volsto.config import CalibrationSpec, SimConfig, load_yaml
from volsto.market import import_hdn as ih
from volsto.market.loaders import snapshot_spec
from volsto.market.sabrw import SabrwTermStructure

ROOT = Path(__file__).resolve().parents[1]
REF_SPEC = ROOT / "configs" / "studies" / "lsv_reference_2f.yaml"
SNAPSHOTS = ROOT / "configs" / "surfaces" / "snapshots" / "hdn_2022H2"
DATES = ("2022-07-15", "2022-09-15", "2022-10-27", "2022-11-25", "2022-12-30")
BE_PILLARS = (0.25, 1.0)
SSR_PILLARS = (0.25, 1.0)
K2_BOUNDS = (0.05, 5.0)


def _atm_of(surface: Any) -> Any:
    def atm(T: float) -> float:
        return float(surface.atm_vol(T))

    return atm


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*", default=list(DATES))
    ap.add_argument("--root", default="data/hdn_sample/options_sample_2022H2")
    ap.add_argument("--n-particles", type=int, default=200_000)
    ap.add_argument("--n-paths", type=int, default=100_000)
    ap.add_argument("--mixing-paths", type=int, default=100_000)
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "step0_stage3"))
    ap.add_argument(
        "--variants",
        nargs="*",
        default=["today", "new"],
        choices=["today", "surface_k2", "new"],
    )
    ap.add_argument(
        "--sigma0-maturity",
        type=float,
        default=None,
        help="the fit's sigma0_maturity (default: 1M); variants named <v>..._s<m>",
    )
    ap.add_argument(
        "--kernel-curve",
        default=None,
        choices=["atmf"],
        help="the fit's kernel_curve (default: the variance-swap curve); variants named <v>_atmf",
    )
    ap.add_argument(
        "--nu-caps",
        nargs="*",
        type=float,
        default=[DEFAULT_NU_CAP],
        help="the fit's nu cap; a variant is run at each (named <variant>_cap<c> off the default)",
    )
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = LeverageCache(a.cache)
    pricing = SimConfig(n_paths=int(a.n_paths), chunk_size=100_000, seed=7)
    ref = load_yaml(REF_SPEC, CalibrationSpec)
    t_all = time.perf_counter()
    checks: list[pd.DataFrame] = []
    fits: list[dict[str, Any]] = []
    for date in a.dates:
        spec = snapshot_spec(ref, SNAPSHOTS / f"spx_{date}.yaml")
        spec = dataclasses.replace(
            spec, particle=dataclasses.replace(spec.particle, n_particles=int(a.n_particles))
        )
        _, surf, _ = build_market(spec)
        _cfg, _fit, points, _chain = ih.import_day(a.root, date)
        ts = SabrwTermStructure.from_fits(
            ih.sabrw_fits(points, t_min=0.05, t_max=3.1), _atm_of(surf)
        )
        base = {
            "today": ({"skew_eps": 0.10}, None),
            "surface_k2": ({"skew_eps": 0.10, "k2_bounds": K2_BOUNDS}, None),
            "new": ({"skew_eps": 0.10, "k2_bounds": K2_BOUNDS}, ts),
        }
        kc = {} if a.kernel_curve is None else {"kernel_curve": a.kernel_curve}
        sfx = "" if a.kernel_curve is None else f"_{a.kernel_curve}"
        if a.sigma0_maturity is not None:
            kc["sigma0_maturity"] = a.sigma0_maturity
            sfx += f"_s{a.sigma0_maturity:g}"
        variants = {
            (v if cap == DEFAULT_NU_CAP else f"{v}_cap{cap:g}")
            + sfx: (
                BreakEvenFitConfig(**base[v][0], **kc, nu_cap=cap),
                base[v][1],
            )
            for v in a.variants
            for cap in a.nu_caps
        }
        for name, (cfg, src) in variants.items():
            t0 = time.perf_counter()
            r = fit_2f_marking(surf, cfg, ssr_target=1.0, step0=src)
            fspec = dataclasses.replace(spec, model=r.params)
            miss = not cache.has(fspec)
            t1 = time.perf_counter()
            lsv, _ = cache.get_or_calibrate(fspec)
            cal_s = time.perf_counter() - t1
            s3 = stage3_validation(
                r.params,
                Stage3Inputs(
                    surface=surf,
                    particle=fspec.particle,
                    sim=fspec.sim,
                    pricing_sim=pricing,
                    ssr_pillars=tuple(T for T in SSR_PILLARS if surf.max_maturity + 1e-9 >= T),
                    breakeven_pillars=BE_PILLARS,
                    forward_starts=(),
                    model=lsv,
                    mixing_paths=int(a.mixing_paths),
                ),
                r.targets,
                fit_table=r.table,
                tolerance=cfg.stage3_tolerance,
            )
            chk = s3.check.copy()
            chk.insert(0, "variant", name)
            chk.insert(0, "date", date)
            checks.append(chk)
            p = r.params
            row: dict[str, Any] = dict(
                date=date,
                variant=name,
                status=r.status,
                nu=p.nu,
                theta=p.theta,
                k1=p.k1,
                k2=p.k2,
                rho12=p.rho12,
                rho_sx1=p.rho_SX1,
                rho_sx2=p.rho_SX2,
                nu_cap=cfg.nu_cap,
                max_first_order_miss=float(
                    np.max(np.abs(np.asarray(r.svc_rel_error, dtype=float)))
                ),
                verdict="pass" if s3.within_tolerance else "FAIL",
                max_bias_svc=float(
                    chk.query("quantity == 'SpotVolCovar'")["engine_bias"].abs().max()
                ),
                max_bias_volvar=float(chk.query("quantity == 'VolVar'")["engine_bias"].abs().max()),
                mean_abs_l_minus_1=s3.mean_abs_l_minus_1,
                recalibrated=miss,
                calibration_s=cal_s if miss else 0.0,
                stage3_s=s3.wall_seconds,
                fit_s=time.perf_counter() - t0,
            )
            for rec in s3.ssr_table.to_dict(orient="records"):
                row[f"ssr_lsv@{rec['T']:g}y"] = rec["ssr_lsv"]
                row[f"ssr_lsv_se@{rec['T']:g}y"] = rec["ssr_lsv_se"]
            fits.append(row)
            print(
                f"{date} {name:5s} nu {p.nu:.2f} k2 {p.k2:.2f}: {row['verdict']}, engine bias "
                f"SVC {row['max_bias_svc']:.1%} VolVar {row['max_bias_volvar']:.1%}, "
                f"recalibrated {miss} ({time.perf_counter() - t0:.0f} s)",
                flush=True,
            )
    wall = time.perf_counter() - t_all
    c = pd.concat(checks, ignore_index=True)
    f = pd.DataFrame(fits)
    c.to_csv(out / "check.csv", index=False)
    f.to_csv(out / "fits.csv", index=False)
    lines = [
        "# Stage-3 check of the marking default (SPEC §15 Part 3)",
        "",
        f"{len(a.dates)} dates x 2 variants; wall {wall:.0f} s; recalibrated: "
        f"{int(f['recalibrated'].sum())} of {len(f)} fits (the others from the cache).",
        "",
        "| date | variant | nu | k2 | verdict | quantity | T | simulated +/- se | first order | "
        "engine bias | target | simulated vs target |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in c.to_dict(orient="records"):
        fr = f[(f["date"] == r["date"]) & (f["variant"] == r["variant"])].iloc[0]
        lines.append(
            f"| {r['date']} | {r['variant']} | {fr['nu']:.2f} | {fr['k2']:.2f} | {fr['verdict']} "
            f"| {r['quantity']} | {r['T']:g} | {r['sim']:.5f} +/- {r['se']:.5f} | "
            f"{r['first_order']:.5f} | {r['engine_bias']:+.1%} | {r['target']:.5f} | "
            f"{r['gap_vs_target']:+.1%} |"
        )
    text = "\n".join(lines) + "\n"
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
