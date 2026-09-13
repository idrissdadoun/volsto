#!/usr/bin/env python
"""Capture today's SPX / SPY option chain from Yahoo Finance in the HistoricalData.net 34-column
layout (SPEC §13), so that a daily cron accumulates history the importer can read.

Usage (needs ``pip install yfinance``, i.e. the ``data`` extra)::

    python scripts/capture_yfinance.py --underlying SPX --out data/yfinance \\
        --rates 5.3,5.3,4.8,4.4,4.1,4.2,4.4

Writes ``<out>/day_by_date/YYYY-MM-DD_options.csv`` and updates ``<out>/day_by_date/manifest.json``
(row counts, SHA-256, and the seven-tenor Treasury curve you pass in percent — Yahoo has no rate
curve; blanks are left where a field is unavailable).  Calculated columns (``iv``, Greeks) are left
blank with ``iv_flag = 7`` (inputs not computed here); ``iv_bid``/``iv_ask`` are blank too.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from volsto.market.import_hdn import HDN_COLUMNS, RATE_TENORS, contract_root

TICKERS = {"SPX": "^SPX", "SPY": "SPY"}


def third_friday(year: int, month: int) -> _dt.date:
    d = _dt.date(year, month, 15)
    return d + _dt.timedelta(days=(4 - d.weekday()) % 7)


def build_rows(
    chain: pd.DataFrame,
    *,
    underlying: str,
    expiration: str,
    quote_date: str,
    spot: float,
    opt_type: str,
) -> pd.DataFrame:
    """Map a yfinance chain frame (calls or puts of one expiry) onto the 34 HDN columns."""
    exp = _dt.date.fromisoformat(expiration)
    out = pd.DataFrame(index=range(len(chain)), columns=list(HDN_COLUMNS), dtype=object)
    sym = chain["contractSymbol"].astype(str).to_numpy()
    roots = np.array([contract_root(s) for s in sym])
    out["contract"] = sym
    out["underlying"] = roots if underlying == "SPX" else underlying
    out["expiration"] = expiration
    out["type"] = opt_type
    out["strike"] = chain["strike"].to_numpy(float)
    out["style"] = "E" if underlying == "SPX" else "A"
    out["quote_date"] = quote_date
    for col in ("bid", "ask", "volume", "openInterest", "lastPrice"):
        if col not in chain:
            chain[col] = np.nan
    out["bid"] = chain["bid"].to_numpy(float)
    out["ask"] = chain["ask"].to_numpy(float)
    out["volume"] = chain["volume"].to_numpy(float)
    out["open_interest"] = chain["openInterest"].to_numpy(float)
    out["close"] = chain["lastPrice"].to_numpy(float)
    out["underlying_close"] = spot
    if underlying == "SPX":
        is_monthly = exp == third_friday(exp.year, exp.month)
        out["settlement_time"] = np.where((roots == "SPX") & is_monthly, "AM", "PM")
    else:
        out["settlement_time"] = "PM"
    out["iv_flag"] = 7  # calculated columns are not produced by this capture
    return out


def write_day(
    rows: pd.DataFrame, out_dir: Path, quote_date: str, rates_percent: list[float | None] | None
) -> Path:
    day_dir = out_dir / "day_by_date"
    day_dir.mkdir(parents=True, exist_ok=True)
    path = day_dir / f"{quote_date}_options.csv"
    rows = rows[list(HDN_COLUMNS)]
    rows.to_csv(path, index=False)
    man_path = day_dir / "manifest.json"
    manifest: dict[str, Any] = (
        json.loads(man_path.read_text())
        if man_path.exists()
        else {"product": "yfinance_capture", "files": [], "rates": {}, "rates_carried_forward": {}}
    )
    manifest["files"] = [f for f in manifest["files"] if f["name"] != path.name]
    manifest["files"].append(
        {
            "name": path.name,
            "rows": len(rows),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
    if rates_percent is not None:
        manifest["rates"][quote_date] = rates_percent
    man_path.write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return path


def capture(underlying: str, out_dir: Path, rates_percent: list[float | None] | None) -> Path:
    import yfinance as yf  # lazy: optional dependency

    tk = yf.Ticker(TICKERS[underlying])
    hist = tk.history(period="1d")
    spot = float(hist["Close"].iloc[-1])
    quote_date = _dt.date.today().isoformat()
    frames = []
    for expiration in tk.options:
        oc = tk.option_chain(expiration)
        frames.append(
            build_rows(
                oc.calls.copy(),
                underlying=underlying,
                expiration=expiration,
                quote_date=quote_date,
                spot=spot,
                opt_type="call",
            )
        )
        frames.append(
            build_rows(
                oc.puts.copy(),
                underlying=underlying,
                expiration=expiration,
                quote_date=quote_date,
                spot=spot,
                opt_type="put",
            )
        )
    rows = pd.concat(frames, ignore_index=True)
    return write_day(rows, out_dir, quote_date, rates_percent)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--underlying", default="SPX", choices=sorted(TICKERS))
    ap.add_argument("--out", default="data/yfinance")
    ap.add_argument(
        "--rates",
        default=None,
        help="Treasury par yields in percent for tenors "
        + ",".join(f"{t:g}y" for t in RATE_TENORS)
        + " (comma separated, blank allowed)",
    )
    args = ap.parse_args(argv)
    rates = None
    if args.rates:
        rates = [float(x) if x.strip() else None for x in args.rates.split(",")]
        if len(rates) != len(RATE_TENORS):
            raise SystemExit(f"--rates needs {len(RATE_TENORS)} values")
    path = capture(args.underlying, Path(args.out), rates)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
