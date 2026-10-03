#!/usr/bin/env python
"""Fetch the daily price histories the barrier-vs-vanilla and dispersion studies read
(``data/history/``): the SPX, the Cboe volatility indices and a ten-name large-cap basket, from
Yahoo Finance (``yfinance``, the ``data`` extra), plus the Cboe implied-correlation indices
(COR1M / COR3M) from Cboe's public CSV when reachable.

    .venv/bin/python scripts/fetch_history.py [--start 1990-01-01] [--out data/history]

Writes one CSV per series (``date,close``; adjusted closes for the stocks, index levels for the
indices) and ``manifest.json`` (source, fetch date, rows, first / last date, SHA-256).  The
studies read the files, never the network: re-run this script to refresh them.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import sys
from pathlib import Path

import pandas as pd

#: the SPX and the Cboe indices (levels)
INDICES: dict[str, str] = {
    "SPX": "^GSPC",
    "VIX": "^VIX",
    "VIX3M": "^VIX3M",
    "VVIX": "^VVIX",
    "SKEW": "^SKEW",
}
#: the dispersion basket: ten large caps listed over the whole window (equal weights in the study)
BASKET: tuple[str, ...] = ("AAPL", "MSFT", "AMZN", "NVDA", "JPM", "XOM", "JNJ", "PG", "HD", "UNH")
CBOE_CSV = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{sym}_History.csv"
CORRELATION_INDICES: tuple[str, ...] = ("COR1M", "COR3M")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(series: pd.Series, path: Path, source: str, manifest: dict[str, object]) -> None:
    s = series.dropna().astype(float)
    s.index = pd.to_datetime(s.index).tz_localize(None).normalize()
    s = s[~s.index.duplicated(keep="last")].sort_index()
    frame = pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "close": s.to_numpy()})
    frame.to_csv(path, index=False)
    manifest[path.stem] = {
        "source": source,
        "rows": int(len(frame)),
        "first": frame["date"].iloc[0],
        "last": frame["date"].iloc[-1],
        "sha256": _sha(path),
    }
    print(f"{path.name}: {len(frame)} rows {frame['date'].iloc[0]}..{frame['date'].iloc[-1]}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--start", default="1990-01-01")
    ap.add_argument("--out", type=Path, default=Path("data/history"))
    args = ap.parse_args(argv)
    import yfinance as yf

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "fetched": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "start": args.start,
        "series": {},
    }
    series: dict[str, object] = manifest["series"]  # type: ignore[assignment]
    tickers = list(INDICES.values()) + list(BASKET)
    data = yf.download(
        tickers, start=args.start, auto_adjust=True, progress=False, group_by="column"
    )
    close = data["Close"]
    for name, tk in INDICES.items():
        _write(close[tk], out / f"{name}.csv", f"yfinance {tk} (Close)", series)
    for tk in BASKET:
        _write(close[tk], out / f"{tk}.csv", f"yfinance {tk} (adjusted Close)", series)
    import requests

    for sym in CORRELATION_INDICES:
        try:
            r = requests.get(CBOE_CSV.format(sym=sym), timeout=20)
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            col = [c for c in df.columns if c.upper() in ("CLOSE", sym.upper())]
            df.index = pd.to_datetime(df.iloc[:, 0])
            _write(df[col[0]] if col else df.iloc[:, -1], out / f"{sym}.csv", "Cboe CSV", series)
        except Exception as exc:  # noqa: BLE001 — a missing optional series is reported, not fatal
            print(f"{sym}: not fetched ({exc})", file=sys.stderr)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
