"""As-traded closes, dividends and splits for the single names of ``data/history``
(owner's decision H, 2026-10-04; SPEC §18.8).

``data/history/<TICKER>.csv`` holds dividend-adjusted closes (yfinance's adjusted close): they
are not fixing references.  This script writes, beside them and without touching them,
``data/history_unadjusted/<TICKER>.csv`` (git-ignored) with the columns

``date``                  trade date
``close``                 the close **as traded** that day
``close_split_adjusted``  yfinance ``Close`` (adjusted for later splits, not for dividends)
``adj_close``             yfinance ``Adj Close`` (splits and dividends)
``dividend``              the dividend **as paid**, on its ex-date (0 when none)
``split``                 the split ratio effective that day (1 when none)
``split_factor_after``    the product of the split ratios after that day

from ``yfinance.Ticker.history(auto_adjust=False, actions=True)``: its ``Close`` and
``Dividends`` are adjusted for later splits, so
``close = Close x split_factor_after`` and ``dividend = Dividends x split_factor_after``.
A ``manifest.json`` records the fetch time, row counts, splits and checksums.

Usage: ``python scripts/fetch_unadjusted_history.py`` (needs the ``data`` extra and network).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf  # type: ignore[import-untyped]

OUT = Path("data/history_unadjusted")
NAMES = ["AAPL", "AMZN", "HD", "JNJ", "JPM", "MSFT", "NVDA", "PG", "UNH", "XOM"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    series: dict[str, dict[str, Any]] = {}
    man = {
        "fetched": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "source": "yfinance Ticker.history(auto_adjust=False, actions=True)",
        "series": series,
    }
    for t in NAMES:
        h = yf.Ticker(t).history(start="1990-01-01", auto_adjust=False, actions=True)
        h.index = pd.to_datetime(h.index).tz_localize(None).normalize()
        split = h["Stock Splits"].replace(0.0, 1.0)
        after = (
            split[::-1].cumprod()[::-1].shift(-1).fillna(1.0)
        )  # product of the splits strictly after the date
        out = pd.DataFrame(
            {
                "date": h.index.strftime("%Y-%m-%d"),
                "close": (h["Close"] * after).to_numpy(),
                "close_split_adjusted": h["Close"].to_numpy(),
                "adj_close": h["Adj Close"].to_numpy(),
                "dividend": (h["Dividends"] * after).to_numpy(),
                "split": split.to_numpy(),
                "split_factor_after": after.to_numpy(),
            }
        )
        p = OUT / f"{t}.csv"
        out.to_csv(p, index=False)
        series[t] = {
            "rows": len(out),
            "first": out.date.iloc[0],
            "last": out.date.iloc[-1],
            "n_dividends": int((out.dividend > 0).sum()),
            "splits": {
                d: float(s) for d, s in zip(out.date[out.split != 1], out.split[out.split != 1])
            },
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        print(
            t,
            len(out),
            "rows",
            out.date.iloc[0],
            "..",
            out.date.iloc[-1],
            "dividends",
            series[t]["n_dividends"],
            "splits",
            series[t]["splits"],
        )
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")


if __name__ == "__main__":
    main()
