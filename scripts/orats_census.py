"""Census of the ORATS store, one row per trade date (SPEC §18.10).

For every date of the store: the raw layout (schema version, whether the OPRA columns exist
and are populated on the SPX rows), the tickers of the SPX family that day (every ticker
starting with ``SPX`` whose ``stkPx`` is within 5% of ticker SPX's — the leveraged ETFs are
listed apart), the OPRA roots under ticker SPX, the duplicates ((expirDate, strike) repeated
under ticker SPX; the store's own uniqueness counts), the number of expiries and the longest
one, the rows, and what the importer's settlement rule (:func:`volsto.market.import_orats.settle`)
makes of the day: settled (rows by source, AM and PM slices) or excluded, with the reason.

Reads the store only.  Writes ``<out>/orats_census.csv`` and prints the period boundaries the
data shows.  The CSV holds counts, no quote: it is derived statistics, but it stays out of git
with the rest of ``outputs/``.

    python scripts/orats_census.py [--store DIR] [--out outputs/interview] [--workers 12]
"""

from __future__ import annotations

import argparse
import datetime as dt
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as pads

from volsto.data import orats
from volsto.data import store as storemod
from volsto.data.roots import DataError, DataRoots
from volsto.market import import_orats as io

COLUMNS = ["ticker", "cOpra", "pOpra", "stkPx", "expirDate", "yte", "strike", "trade_date"]


def census_day(job: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    store_dir, date, entry = job
    path = storemod.day_path(Path(store_dir), date)
    name = pads.field("ticker")
    frame = (
        pads.dataset(path, format="parquet")
        .to_table(columns=COLUMNS, filter=(name >= "SPX") & (name < "SPY"))
        .to_pandas()
    )
    row: dict[str, Any] = {
        "date": date,
        "weekday": dt.date.fromisoformat(date).strftime("%a"),
        "schema_version": entry["schema_version"],
        "rows": entry["rows"],
        "copra_empty": entry.get("copra_empty"),
        "copra_duplicates": entry.get("copra_duplicates"),
        "key_duplicates": entry.get("key_duplicates"),
    }
    spx = frame[frame["ticker"] == "SPX"]
    if spx.empty:
        return {**row, "outcome": "no SPX row"}
    level = float(spx["stkPx"].median())
    family: list[str] = []
    others: list[str] = []
    for ticker, g in frame.groupby("ticker"):
        px = float(g["stkPx"].median())
        (family if abs(px / level - 1.0) < 0.05 else others).append(f"{ticker}:{len(g)}")
    c = spx["cOpra"].fillna("").str.strip()
    p = spx["pOpra"].fillna("").str.strip()
    with_symbol = (c != "") & (p != "")
    roots = sorted(set(c[c != ""].map(orats.opra_root)) | set(p[p != ""].map(orats.opra_root)))
    fam = frame[frame["ticker"].isin(io.FAMILY["SPX"])]
    canon = fam["expirDate"].map(lambda d: orats.canonical_expiry(d)[0])
    row.update(
        {
            "spx_family": " ".join(family),
            "spx_other_tickers": " ".join(others),
            "spx_rows": len(spx),
            "spxpm_rows": int((frame["ticker"] == orats.SPX_PM_TICKER).sum()),
            "opra_columns": entry["schema_version"] == 1,
            "spx_rows_with_symbols": int(with_symbol.sum()),
            "spx_opra_roots": " ".join(roots),
            "spx_dup_expiry_strike": int(spx.duplicated(["expirDate", "strike"]).sum()),
            "n_expiries": int(canon.nunique()),
            "longest_expiry": max(canon).isoformat(),
            "longest_years": round(float(fam["yte"].max()), 3),
            "saturday_expiries": int(
                fam.loc[fam["expirDate"].map(lambda d: d.weekday() == 5), "expirDate"].nunique()
            ),
        }
    )
    fam = fam.copy()
    fam.attrs["trade_date"] = date
    try:
        settled = io.settle(fam, "SPX")
    except io.AmbiguousSettlement as exc:
        return {**row, "outcome": "excluded: ambiguous", "reason": str(exc)[:200]}
    except DataError as exc:
        return {**row, "outcome": "excluded: error", "reason": str(exc)[:200]}
    slices = settled.drop_duplicates(["root", "expiration"])
    by = settled["settled_by"].value_counts()
    monthly = settled["expirDate"].map(lambda d: orats.canonical_expiry(d)[1])
    row.update(
        {
            "outcome": "settled",
            "rows_by_opra": int(by.get("opra", 0)),
            "rows_by_ticker": int(by.get("ticker", 0)),
            "rows_by_calendar": int(by.get("calendar", 0)),
            "am_slices": int((slices["settlement"] == "AM").sum()),
            "pm_slices": int((slices["settlement"] == "PM").sum()),
            "am_monthly_rows": int(((settled["settlement"] == "AM") & monthly).sum()),
            "pm_monthly_rows": int(((settled["settlement"] == "PM") & monthly).sum()),
            "am_non_monthly_rows": int(((settled["settlement"] == "AM") & ~monthly).sum()),
        }
    )
    return row


def spans(frame: pd.DataFrame, key: str) -> list[str]:
    """Runs of consecutive store dates sharing ``key``'s value: ``value: first .. last (n)``."""
    out: list[str] = []
    values = frame[key].astype(str).to_numpy()
    dates = frame["date"].to_numpy()
    start = 0
    for i in range(1, len(frame) + 1):
        if i == len(frame) or values[i] != values[start]:
            out.append(f"  {values[start]}: {dates[start]} .. {dates[i - 1]} ({i - start} days)")
            start = i
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--out", default="outputs/interview")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    store_dir = DataRoots.resolve(store=args.store).store_dir(orats.VENDOR)
    files = storemod.read_manifest(store_dir)["files"]
    jobs = [(str(store_dir), d, e) for d, e in sorted(files.items())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(census_day, jobs, chunksize=8))
    frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out / "orats_census.csv", index=False)
    print(f"{len(frame)} dates {frame['date'].iloc[0]} .. {frame['date'].iloc[-1]}")
    frame["layout"] = (
        "v"
        + frame["schema_version"].astype(str)
        + frame["spx_rows_with_symbols"].gt(0).map({True: " symbols", False: " no symbols"})
    )
    frame["family"] = (
        frame["spx_family"].fillna("").map(lambda s: " ".join(x.split(":")[0] for x in s.split()))
    )
    unmarked = frame["spx_dup_expiry_strike"].fillna(0).gt(0) & ~frame[
        "spx_rows_with_symbols"
    ].fillna(0).gt(0)
    frame["dups"] = unmarked.map({True: "duplicates without symbols", False: "-"})
    frame["am"] = frame["am_slices"].fillna(0).gt(0).map({True: "AM listed", False: "no AM"})
    for key in ("layout", "family", "spx_opra_roots", "dups", "am", "outcome"):
        runs = spans(frame, key)
        print(f"{key} ({len(runs)} runs)")
        print("\n".join(runs[:60]))
        if len(runs) > 60:
            print(f"  … and {len(runs) - 60} more runs")
    print(frame.groupby(frame["date"].str[:4])[["rows", "n_expiries", "longest_years"]].agg(
        ["min", "median", "max"]
    ).to_string())  # fmt: skip


if __name__ == "__main__":
    main()
