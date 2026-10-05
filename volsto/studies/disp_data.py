"""Dispersion study: where the data are and how the study reads them (spec §1).

The ORATS store of the main checkout is read only, one daily file at a time with a ticker
filter (:func:`read_chains`).  Outputs live under ``outputs/dispersion/`` of this worktree
(git-ignored): the price panel, the membership tables, the entry files, the outcomes.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from numpy.typing import NDArray

from volsto.studies import disp_universe as du

FloatArray = NDArray[np.float64]

ROOT: Final = Path(__file__).resolve().parents[2]
OUT: Final = ROOT / "outputs" / "dispersion"
MAIN: Final = Path("/Users/idrissdadoun/Code/volsto")
STORE: Final = Path(os.environ.get("VOLSTO_DATA_STORE", str(MAIN / "data" / "store")))
STRIKES: Final = STORE / "orats" / "strikes"
HISTORY: Final = MAIN / "data" / "history"
NIGHT_LOG: Final = MAIN / "outputs" / "interview" / "logs" / "night2.log"

CHAIN_COLUMNS: Final = (
    "ticker", "expirDate", "strike", "stkPx", "cValue", "pValue", "smoothSmvVol", "iRate",
    "residualRateData", "cBidPx", "cAskPx", "pBidPx", "pAskPx",
)  # fmt: skip
#: Tenors in trading days (spec §1.4): 3m is the core.
TENORS: Final = {"1m": 21, "3m": 63, "6m": 126, "12m": 252, "24m": 504}
FIRST_ENTRY: Final = "2007-01-08"
IS_END: Final = "2016-12-31"
LAST_DAY: Final = "2026-10-02"


def cpu_budget() -> int:
    """Processes the study may use now (spec §0.3): 6 until the barrier study's last LSV
    block has ended, then 12."""
    import re

    try:
        done = re.search(r"done +H_lsv_daily_2012-01-03", NIGHT_LOG.read_text()) is not None
    except OSError:
        done = False
    return 12 if done else 6


def day_path(date: str) -> Path:
    return STRIKES / f"year={date[:4]}" / f"{date}.parquet"


def read_chains(date: str, tickers: list[str]) -> dict[str, pd.DataFrame]:
    """The vendor's rows of ``tickers`` on ``date`` (the columns of :data:`CHAIN_COLUMNS`),
    one frame per ticker present."""
    path = day_path(date)
    if not path.exists():
        return {}
    t = pq.read_table(path, columns=list(CHAIN_COLUMNS), filters=[("ticker", "in", tickers)])
    df = t.to_pandas()
    df["expirDate"] = df["expirDate"].astype(str).str[:10]
    return {k: g.drop(columns="ticker").reset_index(drop=True) for k, g in df.groupby("ticker")}


@lru_cache(maxsize=1)
def prices() -> pd.DataFrame:
    """The cleaned daily price panel (index ISO dates, one column per ticker)."""
    raw = pd.read_parquet(OUT / "prices.parquet")
    fills = OUT / "history" / "fills.parquet"
    return du.clean_prices(raw, pd.read_parquet(fills) if fills.exists() else None)


@lru_cache(maxsize=1)
def trading_days() -> list[str]:
    return list(prices().index)


def entry_dates(n_days: int, last: str = LAST_DAY) -> list[str]:
    """Every Monday (or the next trading day) from :data:`FIRST_ENTRY` whose window of
    ``n_days`` trading days ends by ``last``."""
    days = trading_days()
    pos = {d: i for i, d in enumerate(days)}
    out: list[str] = []
    week = pd.Timestamp(FIRST_ENTRY)
    end = pd.Timestamp(last)
    while week <= end:
        iso = week.strftime("%Y-%m-%d")
        j = int(np.searchsorted(days, iso))
        if j < len(days) and pd.Timestamp(days[j]) < week + pd.Timedelta(days=7):
            d = days[j]
            if pos[d] + n_days < len(days) and days[pos[d] + n_days] <= last and d not in out:
                out.append(d)
        week += pd.Timedelta(days=7)
    return out


def expiry_of(entry: str, n_days: int) -> str | None:
    """The trading day ``entry + n_days`` (``None`` beyond the store)."""
    days = trading_days()
    i = days.index(entry) + n_days
    return days[i] if i < len(days) else None


def maturity(entry: str, n_days: int) -> float:
    """``T`` in years: calendar days to the expiry over 365 (for a window beyond the store,
    ``n_days`` trading days counted as ``n_days × 365/252`` calendar days)."""
    end = expiry_of(entry, n_days)
    if end is None:
        return n_days / 252.0
    return (pd.Timestamp(end) - pd.Timestamp(entry)).days / 365.0


def monthly_subset(entries: list[str]) -> set[str]:
    """The first entry of each month."""
    seen: dict[str, str] = {}
    for d in entries:
        seen.setdefault(d[:7], d)
    return set(seen.values())


#: yfinance file holding a ticker's dividend history when it is not its own (the continuing
#: company), and the tickers whose file is another company's on the study's dates (the "DD"
#: file before June 2019 is Dow Chemical's lineage, not E.I. du Pont's).
DIVIDEND_FILE: Final = {"UTX": "RTX", "KFT": "MDLZ"}
DIVIDEND_FROM: Final = {"DD": "2020-06-04"}


_SPLITS: dict[str, tuple[NDArray[np.str_], FloatArray]] = {}


@lru_cache(maxsize=128)
def dividends(ticker: str) -> tuple[NDArray[np.str_], FloatArray, str] | None:
    """``(ex-dates, split-adjusted amounts per share, first day of the file)`` of ``ticker``
    from the yfinance file of ``scripts/disp_setup.py history``, or ``None`` when there is no
    file."""
    path = OUT / "history" / f"{DIVIDEND_FILE.get(ticker, ticker)}.csv"
    if not path.exists():
        return None
    h = pd.read_csv(path, index_col=0)
    splits = h.loc[h["Stock Splits"] > 0, "Stock Splits"]
    div = h.loc[h["Dividends"] > 0, "Dividends"]
    _SPLITS[ticker] = (splits.index.to_numpy(str), splits.to_numpy(float))
    return div.index.to_numpy(str), div.to_numpy(float), str(h.index[0])


def projected_dividends(ticker: str, entry: str) -> tuple[FloatArray, FloatArray] | None:
    """The dividends of the twelve months before ``entry`` rolled forward one year, as
    ``(times in years from entry, amounts)``; repeated for a second year.  ``None`` when the
    ticker has no dividend file covering the twelve months before ``entry`` (the forward then
    falls back to put-call parity)."""
    d = dividends(ticker)
    if d is None:
        return None
    dates, amounts, first = d
    lo = (pd.Timestamp(entry) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    if first > lo or entry < DIVIDEND_FROM.get(ticker, "0000"):
        return None
    sel = (dates > lo) & (dates <= entry)
    # yfinance amounts are on today's share basis: put them on the basis of the entry date
    split_dates, ratios = _SPLITS[ticker]
    amounts = amounts * float(np.prod(ratios[split_dates > entry]))
    base = np.array(
        [(pd.Timestamp(str(x)) - pd.Timestamp(entry)).days / 365.0 + 1.0 for x in dates[sel]]
    )
    return np.concatenate([base, base + 1.0]), np.concatenate([amounts[sel], amounts[sel]])
