"""Dispersion study: the skew-consistent copula on the monthly subset (spec §3.7).

    python scripts/disp_models_run.py [--tenor 3m] [--workers 5]

For the first entry of each month: model S calibrated to the listed basket's at-the-money
straddle and 90 % put on the date's random numbers; the Palladium forward and the calls (the
cash strikes of the base run) repriced; the conditional profile of relative dispersion; check
C3 under model S.  Writes ``model_s_<tenor>.parquet``.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_entries as de  # noqa: E402

from volsto.studies import disp_copula as dc  # noqa: E402
from volsto.studies import disp_data as dd  # noqa: E402
from volsto.studies import disp_models as dm  # noqa: E402

MULT = ("050", "075", "100", "125", "150", "200")


def job(args: tuple[str, str]) -> dict[str, Any]:
    path, tenor = args
    t0 = time.time()
    with open(path, "rb") as fh:
        r = pickle.load(fh)
    date = r["date"]
    try:
        got = de.marginals_for(date, r["names"], r["T"])
        if got is None:
            return {"date": date, "error": "no smile"}
        margs = [got[t][0] for t in r["names"]]
        tab = dc.tables(margs)
        w = r["w_B1"]
        draws = dc.sobol_draws(len(margs), seed=int(date.replace("-", "")) * 1000 + r["n_days"])
        bk = r["baskets"][r["primary"]]
        base = r["B1"]
        c, s, flag = dm.calibrate_s(tab, draws, w, bk["straddle"], bk["put_90"])
        X = dm.simulate_s(tab, draws, c, s)
        p = dc.payoffs(X, w)
        D = p["D"]
        row: dict[str, Any] = {
            "date": date, "tenor": tenor, "c": c, "s": s, "slope_flag": flag,
            "straddle_error": float(np.abs(p["Rb"]).mean() - bk["straddle"]),
            "put90_error": float(np.maximum(0.9 - p["B"], 0.0).mean() - bk["put_90"]),
            "put90_mkt": bk["put_90"],
            "put90_gaussian": float(np.maximum(0.9 - dc.simulate(tab, draws, r["rho_cop"]) @ w, 0.0).mean()),
            "P_D_S": float(D.mean()), "EV_S": float(p["V"].mean()), "M_B_S": float((p["Rb"] ** 2).mean()),
            "c3_S": float(np.max(np.abs(np.abs(X - 1.0).mean(axis=0) / r["legs"]["straddle"] - 1.0))),
            "seconds": time.time() - t0,
        }  # fmt: skip
        row["converged"] = bool(
            flag == "" and abs(row["straddle_error"]) < 1e-4 and abs(row["put90_error"]) < 1e-4
        )
        for j, m in enumerate(MULT):
            row[f"C_S_{m}"] = float(np.maximum(D - base["strikes"][j], 0.0).mean())
        rel = D / p["B"]
        row["E_DB_S"] = float(rel.mean())
        bucket = np.digitize(p["Rb"], dc.PROFILE_EDGES)
        for j in range(5):
            row[f"profile_S_{j}"] = (
                float(rel[bucket == j].mean() / rel.mean()) if (bucket == j).any() else np.nan
            )
            row[f"profile_share_S_{j}"] = float((bucket == j).mean())
        return row
    except Exception as exc:
        return {
            "date": date,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-600:],
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--tenor", default="3m")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    files = sorted((dd.OUT / "entries" / args.tenor).glob("*.pkl"))
    monthly = dd.monthly_subset([p.stem for p in files])
    jobs = [(str(p), args.tenor) for p in files if p.stem in monthly]
    t0 = time.time()
    with Pool(max(1, min(args.workers, dd.cpu_budget()))) as pool:
        rows = list(pool.imap_unordered(job, jobs))
    bad = [r for r in rows if "error" in r]
    out = pd.DataFrame([r for r in rows if "error" not in r]).sort_values("date")
    out.to_parquet(dd.OUT / f"model_s_{args.tenor}.parquet", index=False)
    print(f"model S {args.tenor}: {len(jobs)} monthly dates, {len(bad)} errors, {int(out['converged'].sum())} converged, "
          f"{int((out['slope_flag'] != '').sum())} with the slope at a bound; C3 under S max {out['c3_S'].max():.4f}, median {out['c3_S'].median():.5f}; {time.time() - t0:.0f} s")  # fmt: skip
    for r in bad[:5]:
        print(r)


if __name__ == "__main__":
    main()
