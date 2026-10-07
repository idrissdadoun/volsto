"""Dispersion study: marks during the life (spec §5.5; the PM can unwind).

    python scripts/disp_marks.py [--tenor 3m] [--workers 5]

Monthly subset: at one third and two thirds of the life every structure is re-priced with
that day's smiles and that day's correlation.  The Palladium is priced conditional on the
performances to date, ``X_{i,T} = X_{i,t}·(S_{i,T}/S_{i,t})``, the remaining returns from that
day's smiles in the same copula; the day's correlation is the one at which the copula on the
frozen basket (weights drifted to that day) reprices the DJX at-the-money straddle of the
remaining maturity.  A holding changed by a corporate action takes the smile of its largest
component (flagged).  Writes ``marks_<tenor>.parquet``: the marks and the unwind P&L of each
structure, held unhedged or with the hedges accrued to that day.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import os
import pickle
import sys
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
import disp_outcomes as do  # noqa: E402

from volsto.studies import disp_copula as dc  # noqa: E402
from volsto.studies import disp_data as dd  # noqa: E402
from volsto.studies import disp_payoff as dp  # noqa: E402
from volsto.studies import disp_universe as du  # noqa: E402


def job(args: tuple[str, str]) -> list[dict[str, Any]]:
    path, tenor = args
    with open(path, "rb") as fh:
        r = pickle.load(fh)
    date, n_days, names, w = r["date"], r["n_days"], r["names"], r["w_B1"]
    days_all = dd.trading_days()
    i0 = days_all.index(date)
    if i0 + n_days >= len(days_all):
        return []
    panel = dd.prices()
    filled = panel.ffill()
    days = days_all[i0 : i0 + n_days + 1]
    full, _, _ = do.window_path(panel, filled, names, days)
    base = r["B1"]
    out: list[dict[str, Any]] = []
    for stage, k in (("1/3", n_days // 3), ("2/3", 2 * n_days // 3)):
        t_day = days[k]
        row: dict[str, Any] = {"date": date, "tenor": tenor, "stage": stage, "day": t_day}
        try:
            T_left = (pd.Timestamp(days[-1]) - pd.Timestamp(t_day)).days / 365.0
            x_t = full[k]
            # the ticker whose smile stands for each holding on that day
            tick, proxy = [], 0
            for t in names:
                comp = du.holding_path(t, date, t_day)[-1][1]
                best = max(
                    comp,
                    key=lambda c: (
                        comp[c] * float(filled.at[t_day, c])
                        if c in filled.columns and np.isfinite(filled.at[t_day, c])
                        else -1.0
                    ),
                )
                proxy += int(len(comp) > 1 or best != t)
                tick.append(best)
            got = de.marginals_for(t_day, sorted(set(tick)), T_left)
            bk = de.marginals_for(t_day, [r["primary"]], T_left)
            if got is None or bk is None:
                row["error"] = "no smile on that day"
                out.append(row)
                continue
            margs = [got[t][0] for t in tick]
            m_b = bk[r["primary"]][0]
            tab = dc.tables(margs)
            draws = dc.sobol_draws(len(names), seed=int(t_day.replace("-", "")) * 1000 + n_days + k)
            b_t = float(x_t @ w)
            w_t = w * x_t / b_t
            rho_t, flag = dc.calibrate_rho(tab, draws, w_t, m_b.straddle, xtol=1e-6)
            X = dc.simulate(tab, draws, rho_t) * x_t[None, :]
            t = dp.terminal(X, w)
            D = t["D"]
            marks = {"PF": float(D.mean()), "SS": float(t["absR"].mean())}
            # the listed basket straddle struck at the entry level, from DJX's own marginal that day
            b_idx = float(filled.at[t_day, r["primary"]] / filled.at[date, r["primary"]])
            marks["BS"] = float(np.trapezoid(np.abs(b_idx * m_b.table() - 1.0) * de._PDF, de._Z))
            marks["PC_100"] = float(np.maximum(D - base["strikes"][2], 0.0).mean())
            price = {
                "PF": base["P_D"],
                "SS": base["SS_mkt"],
                "BS": base["Str_B_mkt"],
                "PC_100": float(base["calls"][2]),
            }
            hd = dp.hedge_legs(
                full[: k + 1],
                w,
                r["T"],
                r["legs"]["atm_vol"],
                float(base["sig_B"]),
                base["sig_rel"],
                n_total=n_days,
            )
            u = {s: marks[s] - price[s] for s in marks}
            h = {
                "PF": u["PF"] + hd["hedge_PF"],
                "SS": u["SS"] + hd["hedge_SS"],
                "BS": u["BS"] + hd["hedge_BS"],
            }
            row.update(rho_t=rho_t, rho_flag=flag, rho_entry=r["rho_cop"], basket_t=b_t - 1.0, proxy_smiles=proxy, T_left=T_left,
                       **{f"mark_{s}": v for s, v in marks.items()})  # fmt: skip
            for tag, x in (("U", u), ("H", h)):
                row[f"PF_{tag}"], row[f"SS_{tag}"], row[f"BS_{tag}"] = x["PF"], x["SS"], x["BS"]
                row[f"PKG_v_{tag}"] = x["SS"] - x["BS"]
                row[f"PKG_theta_{tag}"] = x["SS"] - base["lam_theta"] * x["BS"]
                row[f"GAP_{tag}"] = x["PF"] - (x["SS"] - x["BS"])
                row[f"PF_v_{tag}"] = x["PF"] - base["h_v"] * x["SS"]
            row["PC_100_U"] = u["PC_100"]
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            row["trace"] = traceback.format_exc()[-500:]
        out.append(row)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--tenor", default="3m")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    files = sorted((dd.OUT / "entries" / args.tenor).glob("*.pkl"))
    monthly = dd.monthly_subset([p.stem for p in files])
    jobs = [(str(p), args.tenor) for p in files if p.stem in monthly]
    if args.limit:
        jobs = jobs[:: max(1, len(jobs) // args.limit)][: args.limit]
    with Pool(max(1, min(args.workers, dd.cpu_budget()))) as pool:
        rows = [x for part in pool.imap_unordered(job, jobs) for x in part]
    out = pd.DataFrame(rows).sort_values(["date", "stage"])
    bad = out["error"].notna().sum() if "error" in out else 0
    if not args.limit:
        out.drop(columns=[c for c in ("trace",) if c in out]).to_parquet(
            dd.OUT / f"marks_{args.tenor}.parquet", index=False
        )
    print(
        f"marks {args.tenor}: {len(jobs)} monthly entries, {len(out)} stage rows, {bad} without a mark"
    )
    if bad:
        print(
            out.loc[out["error"].notna(), ["date", "stage", "error"]].head(8).to_string(index=False)
        )
    if args.limit:
        print(out.drop(columns=[c for c in ("trace", "error") if c in out]).round(4).T.to_string())


if __name__ == "__main__":
    main()
