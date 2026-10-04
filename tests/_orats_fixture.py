"""A synthetic ORATS-format day (SPEC §18, M11): the 39-column strikes CSV in a zip, built in a
temporary directory so the data layer's plumbing tests run without the vendor sample.

The chain is internally consistent — Black-76 prices on a forward ``S·exp((r − q)T)`` with a
skewed smile, a bid/ask around the mid — so the importer can read it too.  Ticker ``SPX``
carries the two OPRA roots: ``SPXW`` (PM) on every expiry and ``SPX`` (AM, priced one day
shorter) on the third Fridays, **so the third Fridays 2024-01-19 and 2024-02-16 hold both
roots at the same (ticker, expirDate, strike)**, as the real file does.  ``XSP`` and ``AAPL``
(no ``spot_px``, as every single stock in the sample) fill the universe.  Nothing here is
vendor data.
"""

from __future__ import annotations

import datetime as dt
import zipfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from volsto.data import orats
from volsto.market.bs import black_price

RATE = 0.0555
CARRY = 0.015
SPOT = {"SPX": 4700.0, "XSP": 470.0, "AAPL": 184.0}
THIRD_FRIDAYS = (dt.date(2024, 1, 19), dt.date(2024, 2, 16))
#: (ticker, root, expiry, AM-settled)
SLICES: tuple[tuple[str, str, dt.date, bool], ...] = (
    ("SPX", "SPXW", dt.date(2024, 1, 12), False),
    ("SPX", "SPXW", dt.date(2024, 1, 19), False),
    ("SPX", "SPX", dt.date(2024, 1, 19), True),
    ("SPX", "SPXW", dt.date(2024, 2, 16), False),
    ("SPX", "SPX", dt.date(2024, 2, 16), True),
    ("SPX", "SPXW", dt.date(2024, 3, 28), False),
    ("SPX", "SPX", dt.date(2024, 6, 21), True),
    ("SPX", "SPX", dt.date(2024, 12, 20), True),
    ("XSP", "XSP", dt.date(2024, 1, 19), False),
    ("XSP", "XSP", dt.date(2024, 3, 15), False),
    ("AAPL", "AAPL", dt.date(2024, 1, 19), False),
    ("AAPL", "AAPL", dt.date(2024, 6, 21), False),
)
FILE_NAME = "ORATS_SMV_Strikes_{:%Y%m%d}.zip"


def us_date(d: dt.date) -> str:
    """``M/D/YYYY`` without zero padding, as the vendor writes it."""
    return f"{d.month}/{d.day}/{d.year}"


def osi(root: str, expiry: dt.date, cp: str, strike: float) -> str:
    return f"{root}{expiry:%y%m%d}{cp}{round(strike * 1000):08d}"


def smile(k: np.ndarray, T: float) -> np.ndarray:
    """A skewed, convex implied vol in log-moneyness."""
    return np.asarray(0.14 + 0.02 * np.sqrt(T) - 0.35 * k / (1.0 + 2.0 * T) + 0.6 * k * k)


def day_frame(trade_date: dt.date) -> pd.DataFrame:
    """The synthetic chain of ``trade_date`` with the 39 schema-v1 columns, in file order."""
    rows: list[pd.DataFrame] = []
    for ticker, root, expiry, am in SLICES:
        days = (expiry - trade_date).days
        if days < 0:
            continue
        S = SPOT[ticker]
        yte = round(days / 365.0, 5)
        T = max(days - (1 if am else 0), 0.25) / 365.0
        F = S * float(np.exp((RATE - CARRY) * T))
        step = S * 0.005
        K = np.round(np.arange(0.7, 1.3001, 0.0125) * S / step) * step
        k = np.log(K / F)
        vol = smile(k, T)
        disc = float(np.exp(-RATE * T))
        call = black_price(F, K, T, vol, 1, disc)
        put = black_price(F, K, T, vol, -1, disc)
        half = np.maximum(0.004 * np.maximum(call, put), 0.05 * S / 4700.0)
        c_bid, p_bid = np.maximum(call - half, 0.0), np.maximum(put - half, 0.0)
        n = len(K)
        rows.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "cOpra": [osi(root, expiry, "C", x) for x in K],
                    "pOpra": [osi(root, expiry, "P", x) for x in K],
                    "stkPx": np.round(F * np.exp(-RATE * yte), 2),
                    "expirDate": us_date(expiry),
                    "yte": yte,
                    "strike": K,
                    "cVolu": np.arange(n) % 7,
                    "cOi": 10 * (np.arange(n) % 11),
                    "pVolu": np.arange(n) % 5,
                    "pOi": 10 * (np.arange(n) % 13),
                    "cBidPx": np.round(c_bid, 2),
                    "cValue": np.round(call, 2),
                    "cAskPx": np.round(call + half, 2),
                    "pBidPx": np.round(p_bid, 2),
                    "pValue": np.round(put, 2),
                    "pAskPx": np.round(put + half, 2),
                    "cBidIv": np.round(vol - 0.002, 6),
                    "cMidIv": np.round(vol, 6),
                    "cAskIv": np.round(vol + 0.002, 6),
                    "smoothSmvVol": np.round(vol, 6),
                    "pBidIv": np.round(vol - 0.002, 6),
                    "pMidIv": np.round(vol, 6),
                    "pAskIv": np.round(vol + 0.002, 6),
                    "iRate": RATE,
                    "divRate": 0,
                    "residualRateData": -0.0108,
                    "delta": np.round(np.linspace(0.99, 0.01, n), 8),
                    "gamma": 0.0001,
                    "theta": -0.5,
                    "vega": 1.25,
                    "rho": 0.5,
                    "phi": -0.5,
                    "driftlessTheta": -0.4,
                    "extVol": np.round(vol, 6),
                    "extCTheo": np.round(call, 4),
                    "extPTheo": np.round(put, 4),
                    "spot_px": np.nan if ticker == "AAPL" else round(S * 0.993, 2),
                    "trade_date": us_date(trade_date),
                }
            )
        )
    df = pd.concat(rows, ignore_index=True)
    assert tuple(df.columns) == orats.schema_columns(1)
    return df


def write_day(
    directory: Path,
    trade_date: dt.date,
    *,
    mutate: Callable[[pd.DataFrame], pd.DataFrame] | None = None,
    name: str | None = None,
) -> Path:
    """Write the day's zip (one CSV member, comma-delimited, LF, no quoting) and return its
    path.  ``mutate`` edits the frame first (drop a column, blank the OPRA symbols, …)."""
    df = day_frame(trade_date)
    if mutate is not None:
        df = mutate(df)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (name or FILE_NAME.format(trade_date))
    csv = df.to_csv(index=False, lineterminator="\n", float_format="%.8g")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(path.with_suffix(".csv").name, csv)
    return path


def write_calendar(path: Path, dates: list[dt.date]) -> Path:
    """A ``date,close`` CSV like ``data/history/SPX.csv``."""
    path.write_text("date,close\n" + "".join(f"{d.isoformat()},4700.0\n" for d in dates))
    return path
