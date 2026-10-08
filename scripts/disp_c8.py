"""Dispersion study, check C8 (spec §10): the copula against the library's multi-asset local vol.

    python scripts/disp_c8.py [--dates 12] [--paths 100000]

On dates spread over the sample: the three-month Palladium forward, the call at the forward's
price, the basket straddle and the single-name strip from the study's Gaussian copula against
``volsto.multi`` — one Dupire local-vol model per name, built from the same smiles, driven by
Brownians of constant correlation ``ρ_mark``.  The library's Dupire needs a smooth implied
surface: each listed expiry of each name is fitted with an SVI slice (least squares on the
study's vols within three standard deviations of the money), total variance linear in time
between the slices.  Writes ``c8.json`` and ``c8.csv``; the differences are reported, there is
no pass or fail (a terminal copula and a diffusion with the same marginals and the same
correlation parameter are different joint laws).

The SVI code lives in the library since M12 (``volsto/market/svi_slices.py``, SPEC §8.7): ``svi``,
``fit_svi`` and ``SviSlices`` below are its aliases, and the numbers are unchanged
(``tests/test_svi_slices.py::test_svi_matches_c8``, against the frozen copy of the code that was
here).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_entries as de  # noqa: E402

from volsto.config import LocalVolConfig, SimConfig  # noqa: E402
from volsto.market.curves import DiscountCurve, ForwardCurve  # noqa: E402
from volsto.market.dupire import LocalVolSurface  # noqa: E402
from volsto.market.svi_slices import (  # noqa: E402
    SviSlices,
    fit_svi_slice,
    fit_svi_surface,
    svi_total_variance,
)
from volsto.models.localvol import LocalVol  # noqa: E402
from volsto.multi.draws import constant_correlation  # noqa: E402
from volsto.multi.mc import MultiAssetMonteCarlo  # noqa: E402
from volsto.multi.model import MultiAssetModel  # noqa: E402
from volsto.multi.products import BasketStraddle, Palladium, SingleNameStraddles  # noqa: E402
from volsto.studies import disp_data as dd  # noqa: E402
from volsto.studies import disp_smile as ds  # noqa: E402

#: C8's three SVI names, kept as aliases of the library's (``SviSlices`` is the library's class).
__all__ = ["SviSlices", "fit_svi", "run_date", "svi"]

#: Raw SVI total variance: the library's (C8's name kept).
svi = svi_total_variance


def fit_svi(e: ds.ExpirySmile) -> tuple[np.ndarray, float]:
    """Raw-SVI parameters of one expiry's total variance and the root-mean-square error in vol
    points (:func:`volsto.market.svi_slices.fit_svi_slice`; C8's signature kept)."""
    f = fit_svi_slice(e.k, e.vol, e.T)
    return np.array(f.params), f.rms_vp


def run_date(date: str, n_paths: int) -> dict[str, Any]:
    t0 = time.time()
    with (dd.OUT / "entries" / "3m" / f"{date}.pkl").open("rb") as fh:
        r = pickle.load(fh)
    names, T, w = r["names"], r["T"], r["w_B1"]
    got = de.marginals_for(date, names, T)
    assert got is not None
    models = []
    rms = []
    for t in names:
        m, _, smiles = got[t]
        rate = m.rate
        q = rate - float(np.log(m.f)) / T
        fc = ForwardCurve.flat(1.0, rate, q)
        # expiries from ten days, those up to T and the next two; no fit records (a report)
        surf, fits = fit_svi_surface(smiles, fc, horizon=T, origin="disp_c8")
        rms += [f.rms_vp for f in fits]
        cfg = LocalVolConfig(
            t_min=1.0 / 365.0, t_max=T + 0.02, n_t=120, k_min=-2.0, k_max=2.0, n_k=1601
        )
        models.append(LocalVol(LocalVolSurface.from_implied(surf, cfg), fc))
    rho = float(r["rho_cop"])
    mm = MultiAssetModel(models, constant_correlation(len(names), rho), names=names)
    zero = DiscountCurve.flat(0.0)
    base = r["B1"]
    prods = [
        Palladium(w, 0.0, T, zero),
        Palladium(w, float(base["strikes"][2]), T, zero),
        BasketStraddle(w, T, zero),
        SingleNameStraddles(w, T, zero),
    ]
    mc = MultiAssetMonteCarlo(
        SimConfig(
            n_paths=n_paths,
            chunk_size=25_000,
            seed=int(date.replace("-", "")) % 100_000,
            dt_max=1.0 / 365.0,
        )
    )
    res = mc.price_many(prods, mm)
    out = {
        "date": date, "rho_mark": rho, "svi_rms_vp_median": float(np.median(rms)), "svi_rms_vp_max": float(np.max(rms)),
        "copula_P_D": base["P_D"], "lv_P_D": res[0].mean, "lv_P_D_se": res[0].stderr,
        "copula_C1": float(base["calls"][2]), "lv_C1": res[1].mean, "lv_C1_se": res[1].stderr,
        "copula_Str_B": base["Str_B"], "lv_Str_B": res[2].mean, "lv_Str_B_se": res[2].stderr,
        "smile_SS": base["SS_mkt"], "lv_SS": res[3].mean, "lv_SS_se": res[3].stderr,
        "seconds": time.time() - t0,
    }  # fmt: skip
    out["P_D_lv_over_copula"] = out["lv_P_D"] / out["copula_P_D"]
    out["C1_lv_over_copula"] = out["lv_C1"] / out["copula_C1"]
    out["Str_B_lv_over_copula"] = out["lv_Str_B"] / out["copula_Str_B"]
    out["SS_lv_over_smile"] = out["lv_SS"] / out["smile_SS"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--dates", type=int, default=12)
    ap.add_argument("--paths", type=int, default=100_000)
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--redo", nargs="*", default=[], help="run these dates again and replace their rows"
    )
    args = ap.parse_args()
    lock = dd.OUT / "c8.started"
    rows: list[dict[str, Any]] = []
    if args.redo:
        old = pd.read_csv(dd.OUT / "c8.csv")
        rows = old[~old["date"].isin(args.redo)].to_dict("records")
        pick = list(args.redo)
    else:
        if (lock.exists() or (dd.OUT / "c8.json").exists()) and not args.force:
            print(
                "C8 has run or is running (c8.started / c8.json): nothing to do; --force reruns it"
            )
            return
        lock.write_text(time.strftime("%Y-%m-%d %H:%M"))
        files = sorted(p.stem for p in (dd.OUT / "entries" / "3m").glob("*.pkl"))
        pick = [files[i] for i in np.linspace(0, len(files) - 2, args.dates).round().astype(int)]
    for d in pick:
        try:
            rows.append(run_date(d, args.paths))
            r = rows[-1]
            print(
                f"{d}: P_D LV/copula {r['P_D_lv_over_copula']:.4f}, call {r['C1_lv_over_copula']:.3f}, basket straddle {r['Str_B_lv_over_copula']:.4f}, strip LV/smile {r['SS_lv_over_smile']:.4f}, SVI rms {r['svi_rms_vp_median']:.2f} vp ({r['seconds']:.0f} s)",
                flush=True,
            )
        except Exception as exc:
            print(f"{d}: failed ({type(exc).__name__}: {exc})", flush=True)
    out = pd.DataFrame(rows)
    if len(out):
        out = out.sort_values("date").reset_index(drop=True)
    out.to_csv(dd.OUT / "c8.csv", index=False)
    if len(out):
        summary = {
            "dates": len(out), "paths": args.paths,
            "P_D LV / copula: mean, min, max": [float(out["P_D_lv_over_copula"].mean()), float(out["P_D_lv_over_copula"].min()), float(out["P_D_lv_over_copula"].max())],
            "call at the forward LV / copula: mean, min, max": [float(out["C1_lv_over_copula"].mean()), float(out["C1_lv_over_copula"].min()), float(out["C1_lv_over_copula"].max())],
            "basket straddle LV / copula: mean, min, max": [float(out["Str_B_lv_over_copula"].mean()), float(out["Str_B_lv_over_copula"].min()), float(out["Str_B_lv_over_copula"].max())],
            "single-name strip LV / smile: mean, min, max": [float(out["SS_lv_over_smile"].mean()), float(out["SS_lv_over_smile"].min()), float(out["SS_lv_over_smile"].max())],
            "SVI fit, rms in vol points: median, max": [float(out["svi_rms_vp_median"].median()), float(out["svi_rms_vp_max"].max())],
        }  # fmt: skip
        (dd.OUT / "c8.json").write_text(json.dumps(summary, indent=1))
        print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
