"""Build the SPX settlement table from the open interest (BARRIER_STUDY_SPEC §2.2; SPEC §18.10).

For every store date on which ticker SPX has rows without OPRA symbols and no ``SPXPM`` ticker
is listed, on or after the first ``SPXPM`` date: the call + put open interest of each monthly
expiry, then :func:`volsto.data.spx_settlement.classify`.  Writes
``<store>/orats/spx_settlement.csv`` (derived from vendor data: it stays under the store) and
prints the switches, the relapse days and the days with a dropped expiry.

    python scripts/orats_spx_settlement.py [--store DIR] [--workers 12]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.dataset as pads

from volsto.data import orats, spx_settlement
from volsto.data import store as storemod
from volsto.data.roots import DataRoots


def one(job: tuple[str, str]) -> pd.DataFrame | None:
    """Open interest per monthly expiry of one date; ``None`` when the day is settled without
    the table (symbols, an ``SPXPM`` ticker, or no SPX row)."""
    store_dir, date = job
    name = pads.field("ticker")
    frame = (
        pads.dataset(storemod.day_path(Path(store_dir), date), format="parquet")
        .to_table(
            columns=["ticker", "expirDate", "cOi", "pOi", "cOpra"],
            filter=name.isin([orats.OPRA_REQUIRED_TICKER, orats.SPX_PM_TICKER]),
        )
        .to_pandas()
    )
    spx = frame[frame["ticker"] == orats.OPRA_REQUIRED_TICKER]
    if spx.empty or len(spx) != len(frame) or spx["cOpra"].fillna("").str.strip().ne("").any():
        return None
    canon = spx["expirDate"].map(orats.canonical_expiry)
    spx = spx.assign(expiry=canon.map(lambda c: c[0].isoformat()), oi=spx["cOi"] + spx["pOi"])
    spx = spx[canon.map(lambda c: c[1]) & (spx["expiry"] > date)]
    out = spx.groupby("expiry")["oi"].sum().reset_index()
    out.insert(0, "date", date)
    result: pd.DataFrame = out
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    store_dir = DataRoots.resolve(store=args.store).store_dir(orats.VENDOR)
    first = orats.SPXPM_FIRST_DATE.isoformat()
    dates = [d for d in storemod.available_dates(store_dir) if d >= first]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        parts = [p for p in pool.map(one, [(str(store_dir), d) for d in dates], chunksize=8)]
    kept = [p for p in parts if p is not None and not p.empty]
    long = pd.concat(kept, ignore_index=True)
    wide = long.pivot(index="date", columns="expiry", values="oi").sort_index()
    table = spx_settlement.classify(wide)
    path = spx_settlement.table_path(store_dir)
    table.to_csv(path, index=False)
    info = spx_settlement.summary(table)
    counts = table["settlement"].value_counts().to_dict()
    print(f"{path}: {len(table)} rows on {wide.shape[0]} dates {wide.index[0]} .. {wide.index[-1]}")
    print(f"  settlements {counts}")
    print(f"  {len(info['switches'])} switches:")
    for line in info["switches"]:
        print(f"    {line}")
    print(f"  relapse days: {info['relapse_days']}")
    print(f"  days with a dropped expiry: {info['drop_days']}")


if __name__ == "__main__":
    main()
