"""Dispersion study, setup (spec §1): the daily price panel, the membership and corporate-action
tables with their checks, and the pre-2007 histories.

    python scripts/disp_setup.py prices   [--workers 4]   # outputs/dispersion/prices.parquet, presence.csv
    python scripts/disp_setup.py members                  # djia_members.csv, corp_actions.csv, checks
    python scripts/disp_setup.py history                  # yfinance: closes before 2007, dividends, splits

``prices`` reads ``ticker, stkPx`` of the study's tickers from every daily file of the ORATS
store (read only; ``VOLSTO_DATA_STORE``).  ``members`` writes the two tables and checks each
corporate action against the prices (the holding's value across the ex-date) and lists every
overnight move above 15 % of a member that no action explains.  ``history`` needs the network;
what it cannot fetch is logged and left out.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from volsto.studies import disp_universe as du  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "dispersion"
STORE = Path(os.environ.get("VOLSTO_DATA_STORE", "/Users/idrissdadoun/Code/volsto/data/store"))
STRIKES = STORE / "orats" / "strikes"
#: Predecessor spellings read beside the study's tickers (renames in ORATS).
EXTRA = (
    "STA",
    "WAG",
    "GOOG",
    "HONA",
    "KHC",
    "MYL",
    "FRP",
    "Q",
    "GEHC",
    "GEV",
    "ARNC",
    "HWM",
    "HPE",
)


def day_files() -> list[Path]:
    return sorted(STRIKES.glob("year=*/*.parquet"))


def read_day(path: str) -> pd.DataFrame:
    wanted = du.price_tickers() + list(EXTRA)
    t = pq.read_table(path, columns=["ticker", "stkPx"], filters=[("ticker", "in", wanted)])
    df = t.to_pandas().drop_duplicates("ticker")
    df["date"] = Path(path).stem
    return df


def cmd_prices(workers: int) -> None:
    files = [str(p) for p in day_files()]
    with Pool(workers) as pool:
        parts = pool.map(read_day, files, chunksize=64)
    long = pd.concat(parts, ignore_index=True)
    wide = long.pivot(index="date", columns="ticker", values="stkPx").sort_index()
    wide = wide.where(wide > 0)
    OUT.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(OUT / "prices.parquet")
    rows = []
    for t in wide.columns:
        s = wide[t].dropna()
        idx = np.flatnonzero(wide[t].notna().to_numpy())
        gaps = int((np.diff(idx) > 1).sum()) if idx.size > 1 else 0
        longest = int(np.diff(idx).max() - 1) if idx.size > 1 else 0
        rows.append(
            {
                "ticker": t,
                "first": s.index[0],
                "last": s.index[-1],
                "days": len(s),
                "gaps": gaps,
                "longest_gap_days": longest,
            }
        )
    pd.DataFrame(rows).to_csv(OUT / "presence.csv", index=False)
    print(f"prices: {wide.shape[0]} days x {wide.shape[1]} tickers; presence.csv written")
    missing = [t for t in du.price_tickers() if t not in wide.columns]
    print("no row at all:", missing)


def load_prices() -> pd.DataFrame:
    """The cleaned price panel (ORATS, with the cuts and the yfinance fills of the universe)."""
    raw = pd.read_parquet(OUT / "prices.parquet")
    fills = OUT / "history" / "fills.parquet"
    return du.clean_prices(raw, pd.read_parquet(fills) if fills.exists() else None)


def cmd_members() -> None:
    prices = load_prices()
    days = list(prices.index)
    members = du.membership_table()
    pres = pd.read_csv(OUT / "presence.csv").set_index("ticker")
    members["orats_first"] = members["ticker"].map(pres["first"])
    members["orats_last"] = members["ticker"].map(pres["last"])
    gaps = []
    for r in members.itertuples():
        end = r.end or days[-1]
        span = (
            prices.loc[(prices.index >= r.start) & (prices.index <= end), r.ticker]
            if r.ticker in prices
            else pd.Series(dtype=float)
        )
        gaps.append(int(span.isna().sum()) if len(span) else -1)
    members["days_without_price_while_member"] = gaps
    members.to_csv(OUT / "djia_members.csv", index=False)
    print(members.to_string(index=False))
    # every action against the prices: the value of the holding across the ex-date
    acts = du.actions_frame()
    filled = prices.ffill()
    rows = []
    for a in du.ACTIONS:
        if a.ex_date not in prices.index:
            later = [d for d in days if d >= a.ex_date]
            rows.append(
                {
                    "ex_date": a.ex_date,
                    "parent": a.parent,
                    "check": f"no store file on the ex-date (next: {later[0] if later else '-'})",
                }
            )
            continue
        i = days.index(a.ex_date)
        prev = filled[a.parent].iloc[i - 1] if a.parent in filled else np.nan
        parts, value, first_print = [], 0.0, {}
        for child, ratio in a.becomes:
            px = prices[child].iloc[i] if child in prices else np.nan
            if not np.isfinite(px) and child in prices:
                nxt = prices[child].iloc[i : i + 8].dropna()
                px = float(nxt.iloc[0]) if len(nxt) else np.nan
                first_print[child] = nxt.index[0] if len(nxt) else "never"
            parts.append(f"{ratio:.6g} {child} @ {px:.2f}")
            value += ratio * px
        raw = prices[a.parent].iloc[i] / prev - 1 if a.parent in prices else np.nan
        rows.append(
            {
                "ex_date": a.ex_date, "parent": a.parent, "prev_close": round(float(prev), 2),
                "becomes": "; ".join(parts), "holding_return": round(float(value / prev - 1), 4),
                "parent_price_return": round(float(raw), 4) if np.isfinite(raw) else np.nan,
                "late_first_print": "; ".join(f"{k}: {v}" for k, v in first_print.items()),
                "check": "ok" if abs(value / prev - 1) < 0.08 else "LOOK",
            }
        )  # fmt: skip
    chk = pd.DataFrame(rows)
    acts.to_csv(OUT / "corp_actions.csv", index=False)
    chk.to_csv(OUT / "corp_actions_check.csv", index=False)
    print(chk.to_string(index=False))
    # overnight moves above 15 % of a member (while a member, or within two years after) that no action explains
    rows = []
    explained = {(a.parent, a.ex_date) for a in du.ACTIONS}
    for r in members.itertuples():
        if r.ticker not in prices:
            continue
        end = r.end or days[-1]
        horizon = (pd.Timestamp(end) + pd.Timedelta(days=740)).strftime("%Y-%m-%d")
        p = prices.loc[(prices.index >= r.start) & (prices.index <= horizon), r.ticker].dropna()
        lr = np.log(p / p.shift(1)).dropna()
        for d, x in lr[lr.abs() > 0.15].items():
            rows.append(
                {
                    "ticker": r.ticker,
                    "date": d,
                    "log_return": round(float(x), 4),
                    "explained_by_action": (r.ticker, d) in explained,
                }
            )
    jumps = pd.DataFrame(rows).drop_duplicates()
    jumps.to_csv(OUT / "price_jumps.csv", index=False)
    print(f"{len(jumps)} overnight moves above 15 %; not explained by an action:")
    print(jumps[~jumps["explained_by_action"]].to_string(index=False))


def cmd_history() -> None:
    import yfinance as yf

    dest = OUT / "history"
    dest.mkdir(parents=True, exist_ok=True)
    names = [t for t in du.all_members() if t not in ("DWDP", "KFT", "UTX", "GM")] + [
        "MDLZ",
        "^DJI",
    ]
    alias: dict[str, str] = {}
    log = []
    for t in names:
        try:
            tk = yf.Ticker(alias.get(t, t))
            h = tk.history(start="1999-01-01", end="2026-10-03", auto_adjust=True, actions=True)
            if h.empty:
                log.append(f"{t}: empty")
                continue
            h.index = h.index.strftime("%Y-%m-%d")
            h[["Close", "Dividends", "Stock Splits"]].to_csv(
                dest / f"{t.replace(chr(94), chr(95))}.csv"
            )
            log.append(f"{t}: {len(h)} days from {h.index[0]}")
        except Exception as exc:
            log.append(f"{t}: failed ({type(exc).__name__}: {exc})")
    # closing prices of the spun-off shares on the days ORATS has no row yet (unadjusted closes:
    # none of these tickers has split since)
    cols = {}
    for t, (lo, hi) in du.PRICE_FILLS.items():
        try:
            h = yf.Ticker(t).history(
                start=lo,
                end=(pd.Timestamp(hi) + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                auto_adjust=False,
            )
            h.index = h.index.strftime("%Y-%m-%d")
            cols[t] = h["Close"]
            log.append(f"fill {t}: {len(h)} closes {lo} to {hi}")
        except Exception as exc:
            log.append(f"fill {t}: failed ({type(exc).__name__}: {exc})")
    if cols:
        pd.DataFrame(cols).to_parquet(dest / "fills.parquet")
    (dest / "fetch_log.txt").write_text("\n".join(log) + "\n")
    print("\n".join(log))


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("what", choices=["prices", "members", "history"])
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.what == "prices":
        cmd_prices(min(args.workers, 5))
    elif args.what == "members":
        cmd_members()
    else:
        cmd_history()


if __name__ == "__main__":
    sys.exit(main())
