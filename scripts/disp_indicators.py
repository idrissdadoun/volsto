"""Dispersion study: indicators at entry and Gaussian forecast edges (spec §6, §7.1).

    python scripts/disp_indicators.py --tenor 3m

Builds (once) ``outputs/dispersion/returns.parquet`` — the daily adjusted returns of every
member, from the ORATS prices with the corporate actions applied on their ex-dates, continued
before 2007 by the yfinance adjusted closes — then, for every entry of the tenor and each
basket, the realised-side indicators from the returns strictly before the entry date, the
premia against the prices of ``entries_<tenor>.parquet`` and the Gaussian forecast edges of
every structure.  Writes ``indicators_<tenor>.parquet``.  Single process.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from volsto.studies import disp_data as dd
from volsto.studies import disp_indicators as di
from volsto.studies import disp_universe as du

#: yfinance files usable for the history before 2007 (the file of the continuing company).
PRE2007 = {**{t: t for t in du.all_members()}, "UTX": "RTX", "KFT": "MDLZ"}
NO_PRE2007 = {"DD", "DWDP", "GM", "WBA", "DOW", "V", "RTX", "CRM"}


def build_returns() -> pd.DataFrame:
    prices = dd.prices()
    cols = {}
    for t in du.all_members():
        r = du.history_returns(prices, t)
        src = dd.OUT / "history" / f"{PRE2007.get(t, t)}.csv"
        if t not in NO_PRE2007 and src.exists():
            h = pd.read_csv(src, index_col=0)["Close"]
            old = (h / h.shift(1) - 1.0).dropna()
            old = old[old.index <= prices.index[0]]
            r = pd.concat([old, r[r.index > prices.index[0]]])
        cols[t] = r
    out = pd.DataFrame(cols).sort_index()
    out.to_parquet(dd.OUT / "returns.parquet")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--tenor", default="3m", choices=list(dd.TENORS))
    ap.add_argument("--suffix", default="")
    ap.add_argument("--rebuild-returns", action="store_true")
    args = ap.parse_args()
    path = dd.OUT / "returns.parquet"
    returns = (
        build_returns() if args.rebuild_returns or not path.exists() else pd.read_parquet(path)
    )
    h = dd.TENORS[args.tenor]
    entries = pd.read_parquet(dd.OUT / f"entries_{args.tenor}{args.suffix}.parquet")
    legs = pd.read_parquet(dd.OUT / f"legs_{args.tenor}{args.suffix}.parquet")
    by_date = {d: g for d, g in legs.groupby("date")}
    rows = []
    for e in entries.itertuples():
        g = by_date[e.date]
        names = list(g["ticker"])
        w = g["w_B1"].to_numpy(float) if e.basket == "B1" else np.full(len(names), 1.0 / len(names))
        hist = di.trailing(returns, e.date, names, 1260)
        ind = di.indicators(hist, w, h, float(e.T))
        row = {"date": e.date, "basket": e.basket, "tenor": e.tenor, **ind}
        expiry = (
            e.expiry
            if isinstance(e.expiry, str)
            else (pd.Timestamp(e.date) + pd.Timedelta(days=round(365 * e.T))).strftime("%Y-%m-%d")
        )
        row["earnings_share"] = di.earnings_share(e.date, expiry)
        if ind:
            # premia: implied against forecast
            row["CRP"] = e.rho_atm - ind["rho_fc"]
            row["CRP_cop"] = e.rho_cop - ind["rho_fc"]
            row["vol_premium"] = e.sig_bar / ind["sig_fc_bar"]
            row["basket_vol_premium"] = e.sig_B / ind["sig_B_fc"]
            row["GP_G"] = e.P_G / ind["EG_G1"] if ind["EG_G1"] > 0 else np.nan
            row["GP_G0"] = e.P_G / ind["EG_G0"] if ind["EG_G0"] > 0 else np.nan
            row["disp_var_premium"] = e.EQV / ind["EV_G1"] if ind["EV_G1"] > 0 else np.nan
            row["kappa_fc_G"] = ind["ED_G1"] / np.sqrt(ind["EV_G1"]) if ind["EV_G1"] > 0 else np.nan
            for tag in ("G0", "G1"):
                pf = ind[f"ED_{tag}"] - e.P_D
                ss = ind[f"EabsR_{tag}"] - e.SS_mkt
                bs = ind[f"EabsRb_{tag}"] - e.Str_B_mkt
                row[f"edge_PF_{tag}"] = pf
                row[f"edge_SS_{tag}"] = ss
                row[f"edge_BS_{tag}"] = bs
                row[f"edge_PKG_v_{tag}"] = ss - bs
                row[f"edge_PKG_theta_{tag}"] = ss - e.lam_theta * bs
                row[f"edge_PKG_rho_{tag}"] = ss - e.lambda_rho * bs
                row[f"edge_GAP_{tag}"] = ind[f"EG_{tag}"] - e.P_G
                row[f"edge_GAP_rho_{tag}"] = pf - (ss - e.lambda_rho * bs)
                row[f"edge_PF_v_{tag}"] = pf - e.h_v * ss
        row["corr_skew"] = e.rho_90 - e.rho_atm
        row["corr_term"] = e.rho_atm_1m - e.rho_atm_12m
        row["rho_mark"] = e.rho_cop
        row["front_premium"] = e.sig_bar_1m - e.sig_bar_3m
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_parquet(dd.OUT / f"indicators_{args.tenor}{args.suffix}.parquet", index=False)
    b1 = out[out["basket"] == "B1"]
    first = {
        c: b1.loc[b1[c].notna(), "date"].min()
        for c in ("rho_fc", "VR_cs", "variability", "GP_G")
        if c in b1
    }
    print(f"indicators {args.tenor}{args.suffix}: {len(b1)} dates; first date of each: {first}")


if __name__ == "__main__":
    main()
