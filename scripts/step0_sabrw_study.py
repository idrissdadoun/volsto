"""Step 0 from the desk's SABRW fits against the surface's ATM derivatives over the 2022 H2 SPX
sample (SPEC §15 Part 3, *Step 0 from the desk's SABRW fits*; the 127-day first-order
measurement: marking fits only — no Monte Carlo, no calibration).

For every trading day of the HDN sample: the current importer's surface and quotes
(:func:`volsto.market.import_hdn.import_day`), the SABRW fits of every expiry from
``--t-min`` to ``--t-max`` (:func:`~volsto.market.import_hdn.sabrw_fits`) and the marking fit at
SSR 1 under each variant of :data:`VARIANTS`:

* ``surface`` — today's default (step 0 from the surface, ``SabrW_Power`` 1, ``k2`` 0.2, the
  two-point band at 0.10); ``surface_p0`` the same at ``SabrW_Power`` 0 (the physical curvature);
  ``surface_k2`` with ``k2`` fitted in :data:`K2_BOUNDS`;
* ``sabrw`` — step 0 from the SABRW fits (the round trip at ``p = 0``); ``sabrw_p1`` the other
  reading of the desk's stored convexity (the fits' curvature taken as physical, then scaled by
  ``(atf_ref/atf)^1`` as the surface path does); ``sabrw_k2`` with ``k2`` fitted;
  ``sabrw_k2_band4`` with ``k2`` fitted and the band on 3M / 6M / 1Y / 3Y at 0.3 / 0.2 / 0.1 / 0.1;
* ``desk`` — the named fit :data:`~volsto.calibration.fit_2f.DESK_FIT` as it stands (step 0 from
  the SABRW fits, ``k2`` fitted, ATMF kernels, 3M ``σ_0``, the desk note's bounds) at ``ε`` 0.10.

``--variants`` runs a subset.

Outputs in ``--out`` (git-ignored): ``variants.csv`` (date × variant: status, parameters,
smallest correlation eigenvalue, largest covariance miss, clipped pillars, binding edges, wall
time, error), ``pillars.csv`` (date × variant × pillar: targets and misses), ``sabrw.csv``
(date × expiry: the SABRW parameters, fit error, held / at-bound parameters) and ``summary.md``.
"""

from __future__ import annotations

import argparse
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volsto.calibration.fit_2f import BreakEvenFitConfig, fit_2f_marking, fit_preset
from volsto.calibration.targets import DEFAULT_ATF_REF
from volsto.market.loaders import sabrw_fits_from_config
from volsto.market.sabrw import SabrwTermStructure
from volsto.market.vendor import HdnSource

K2_BOUNDS = (0.05, 5.0)
BAND4 = (0.25, 0.5, 1.0, 3.0)
EPS4 = (0.3, 0.2, 0.1, 0.1)
#: variant -> (config overrides or a named fit, step-0 source: "surface" | "sabrw" | "sabrw_p1")
VARIANTS: dict[str, tuple[dict[str, Any] | str, str]] = {
    "surface": ({}, "surface"),
    "surface_p0": ({"sabrw_power": 0.0}, "surface"),
    "surface_k2": ({"k2_bounds": K2_BOUNDS}, "surface"),
    "sabrw": ({}, "sabrw"),
    "sabrw_p1": ({}, "sabrw_p1"),
    "sabrw_k2": ({"k2_bounds": K2_BOUNDS}, "sabrw"),
    "sabrw_k2_band4": (
        {"k2_bounds": K2_BOUNDS, "skew_pillars": BAND4, "skew_eps": EPS4},
        "sabrw",
    ),
    "desk": ("desk", "sabrw"),
}


@dataclass(frozen=True)
class PowerScaled:
    """The ``sabrw_p1`` reading: the fits' curvature scaled by ``(atf_ref/atf)^power``."""

    base: SabrwTermStructure
    power: float
    label: str

    def triplet(self, T: float) -> tuple[float, float, float]:
        atf, skw, cvx = self.base.triplet(T)
        return atf, skw, cvx * (DEFAULT_ATF_REF / atf) ** self.power


def run_date(
    args: tuple[str, str, float, float, tuple[str, ...]],
) -> tuple[list[dict], list[dict], list[dict]]:
    root, date, t_min, t_max, names = args
    rows: list[dict] = []
    pillars: list[dict] = []
    sab: list[dict] = []
    t0 = time.time()
    try:
        cfg, fit, _points, _chain = HdnSource(root).import_day(date)
        surface = fit.surface
        # the fits the import itself stored (data: read back, never fitted a second time)
        fits = tuple(f for f in sabrw_fits_from_config(cfg) or () if t_min <= f.T <= t_max)
        ts = SabrwTermStructure.from_fits(fits, lambda T: float(surface.atm_vol(T)))
    except Exception as exc:  # the date is recorded as failed for every variant
        traceback.print_exc()
        err = f"import or SABRW: {type(exc).__name__}: {exc}"
        return [dict(date=date, variant=n, status="error", error=err) for n in names], [], []
    t_import = time.time() - t0
    for f in fits:
        p = f.params
        sab.append(
            dict(
                date=date,
                T=f.T,
                n=f.n,
                sigma=p.sigma,
                rho=p.rho,
                nu=p.nu,
                t_d=p.t_d,
                t_u=p.t_u,
                ex_d=p.ex_d,
                ex_u=p.ex_u,
                rms_vp=f.rms_vp,
                held=",".join(f.held),
                at_bound=",".join(f.at_bound),
            )
        )
    sources: dict[str, Any] = {
        "surface": None,
        "sabrw": ts,
        "sabrw_p1": PowerScaled(ts, 1.0, ts.label + " (curvature scaled, SabrW_Power 1)"),
    }
    for name in names:
        over, src = VARIANTS[name]
        t1 = time.time()
        row: dict[str, Any] = dict(date=date, variant=name, import_s=t_import)
        try:
            cfg = (
                fit_preset(over, skew_eps=0.10)
                if isinstance(over, str)
                else BreakEvenFitConfig(**over)
            )
            r = fit_2f_marking(surface, cfg, ssr_target=1.0, step0=sources[src])
            prm = r.params
            miss = np.asarray(r.svc_rel_error, dtype=float)
            tg = r.targets
            row.update(
                status=r.status,
                nu=prm.nu,
                theta=prm.theta,
                k1=prm.k1,
                k2=prm.k2,
                rho12=prm.rho12,
                rho_sx1=prm.rho_SX1,
                rho_sx2=prm.rho_SX2,
                chi=r.second.chi,
                chi_at_bound=any("chi" in b for b in r.second.bound_flags),
                nu_at_cap=bool(r.second.nu_at_cap),
                min_eig=r.min_correlation_eigenvalue,
                max_abs_miss=float(np.max(np.abs(miss))),
                n_clipped=sum("clipped" in fl for fl in tg.flags),
                binding=";".join(str(b) for b in r.constraints["binding_edge"] if b),
                error="",
            )
            for i, T in enumerate(np.asarray(tg.pillars, dtype=float)):
                pillars.append(
                    dict(
                        date=date,
                        variant=name,
                        T=T,
                        corr_target=float(tg.correl_target[i]),
                        vov_target=float(tg.vovol[i]),
                        skew_target=float(tg.skew_target[i]),
                        nu_sabr=float(tg.sabr[i].nu_sabr),
                        svc_miss=float(miss[i]),
                    )
                )
        except Exception as exc:  # recorded, counted in the summary
            row.update(status="error", error=f"{type(exc).__name__}: {exc}")
            traceback.print_exc()
        row["fit_s"] = time.time() - t1
        rows.append(row)
    return rows, pillars, sab


def _q(s: pd.Series, q: float) -> float:
    return float(s.quantile(q)) if len(s) else float("nan")


def summary(v: pd.DataFrame, p: pd.DataFrame, s: pd.DataFrame, wall: float, n_dates: int) -> str:
    ok = v[v["status"] != "error"]
    names = [n for n in VARIANTS if n in set(v["variant"])]
    lines = [
        "# Step 0 from SABRW fits: the 2022 H2 SPX measurement",
        "",
        f"{n_dates} dates, {len(names)} variants; wall {wall:.0f} s; recalibrated: no "
        "(marking fits only, no Monte Carlo).",
        f"Errors: {int((v['status'] == 'error').sum())} of {len(v)} fits.",
        "",
        "| variant | infeasible | max miss median / p90 / max | dates with max miss <= 5 % | "
        "min eigenvalue median / p10 | dates with a clipped pillar | chi at bound | nu at cap | "
        "rho12 median [p10, p90] | nu median | k2 median [p10, p90] | daily |dk2| median | "
        "daily |dnu| median | daily |drho12| median |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        g = ok[ok["variant"] == name].sort_values("date")
        if g.empty:
            lines.append(f"| {name} | all failed |" + " |" * 13)
            continue
        miss = 100 * g["max_abs_miss"]
        lines.append(
            f"| {name} | {int((g['status'] == 'infeasible').sum())} | "
            f"{miss.median():.1f} / {_q(miss, 0.9):.1f} / {miss.max():.1f} % | "
            f"{int((miss <= 5.0).sum())} of {len(g)} | "
            f"{g['min_eig'].median():.4f} / {_q(g['min_eig'], 0.1):.4f} | "
            f"{int((g['n_clipped'] > 0).sum())} | {int(g['chi_at_bound'].sum())} | "
            f"{int(g['nu_at_cap'].sum())} | "
            f"{g['rho12'].median():+.2f} "
            f"[{_q(g['rho12'], 0.1):+.2f}, {_q(g['rho12'], 0.9):+.2f}] | "
            f"{g['nu'].median():.2f} | "
            f"{g['k2'].median():.2f} [{_q(g['k2'], 0.1):.2f}, {_q(g['k2'], 0.9):.2f}] | "
            f"{g['k2'].diff().abs().median():.3f} | {g['nu'].diff().abs().median():.3f} | "
            f"{g['rho12'].diff().abs().median():.3f} |"
        )
    lines += [
        "",
        "Correlation targets (median over dates) and covariance misses (median |miss|):",
        "",
    ]
    ts = sorted(p["T"].unique())
    lines.append(
        "| variant | "
        + " | ".join(f"Corr_BE {t:g}y" for t in ts)
        + " | "
        + " | ".join(f"|miss| {t:g}y" for t in ts)
        + " |"
    )
    lines.append("|---|" + "---|" * (2 * len(ts)))
    for name in names:
        g = p[p["variant"] == name]
        c = [g.loc[g["T"] == t, "corr_target"].median() for t in ts]
        m = [100 * g.loc[g["T"] == t, "svc_miss"].abs().median() for t in ts]
        lines.append(
            f"| {name} | "
            + " | ".join(f"{x:+.3f}" for x in c)
            + " | "
            + " | ".join(f"{x:.1f} %" for x in m)
            + " |"
        )
    at_b = s["at_bound"].fillna("")
    lines += [
        "",
        f"SABRW fits: {len(s)} over {s['date'].nunique()} dates; rho or nu at a bound in "
        f"{int(at_b.str.contains('rho|nu').sum())}; fit error median {s['rms_vp'].median():.2f} vp "
        f"(p90 {_q(s['rms_vp'], 0.9):.2f}); held slopes: "
        + ", ".join(
            f"{k} {int(s['held'].fillna('').str.contains(k).sum())}"
            for k in ("ex_d", "t_d", "t_u", "ex_u")
        )
        + ".",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/hdn_sample/options_sample_2022H2")
    ap.add_argument("--out", default="outputs/step0_sabrw")
    ap.add_argument("--t-min", type=float, default=0.05)
    ap.add_argument("--t-max", type=float, default=3.1)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--dates", nargs="*", help="default: every day of the sample")
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS), choices=list(VARIANTS))
    a = ap.parse_args()
    root = Path(a.root)
    dates = a.dates or HdnSource(root).available_dates()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    jobs = [(str(root), d, a.t_min, a.t_max, tuple(a.variants)) for d in dates]
    rows: list[dict] = []
    pil: list[dict] = []
    sab: list[dict] = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, (r, p, s) in enumerate(ex.map(run_date, jobs), 1):
            rows += r
            pil += p
            sab += s
            print(f"{i}/{len(jobs)} {r[0]['date']} ({time.time() - t0:.0f} s)", flush=True)
    wall = time.time() - t0
    v = pd.DataFrame(rows)
    p = pd.DataFrame(pil)
    s = pd.DataFrame(sab)
    v.to_csv(out / "variants.csv", index=False)
    p.to_csv(out / "pillars.csv", index=False)
    s.to_csv(out / "sabrw.csv", index=False)
    text = summary(v, p, s, wall, len(dates))
    (out / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
