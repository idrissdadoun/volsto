"""Open interest of each SPX monthly expiry under ticker SPX, day by day (SPEC §18.10).

Where a file has no OPRA symbol, nothing in a row says whether a third-Friday series under
ticker SPX is the AM-settled monthly or the PM-settled one.  Their open interest differs by an
order of magnitude, so a change of series shows as a jump of an expiry's total open interest
from one trade date to the next.  This script writes, per (trade date, monthly expiry), the
rows and the call + put open interest (``<out>/orats_spx_monthly_oi.csv``), and prints every
jump by more than ``--factor`` (default 3) between consecutive store dates.  Counts only, no
quote; the file stays out of git with the rest of ``outputs/``.

    python scripts/orats_spx_monthly_oi.py [--start 2017-01-01] [--end 2021-12-31]
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.dataset as pads

from volsto.data import orats
from volsto.data import store as storemod
from volsto.data.roots import DataRoots


def one(job: tuple[str, str]) -> pd.DataFrame:
    store_dir, date = job
    frame = (
        pads.dataset(storemod.day_path(Path(store_dir), date), format="parquet")
        .to_table(
            columns=["expirDate", "cOi", "pOi", "cOpra"], filter=pads.field("ticker") == "SPX"
        )
        .to_pandas()
    )
    if frame.empty:
        return pd.DataFrame()
    canon = frame["expirDate"].map(orats.canonical_expiry)
    frame["expiry"] = canon.map(lambda c: c[0].isoformat())
    frame = frame[canon.map(lambda c: c[1])]
    frame["root"] = frame["cOpra"].fillna("").map(lambda s: orats.opra_root(s) if s else "")
    out = (
        frame.assign(oi=frame["cOi"] + frame["pOi"])
        .groupby(["expiry", "root"])
        .agg(rows=("oi", "size"), oi=("oi", "sum"))
        .reset_index()
    )
    out.insert(0, "date", date)
    return out[out["expiry"] > date]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--store", default=None)
    ap.add_argument("--out", default="outputs/interview")
    ap.add_argument("--start", default="2017-01-01")
    ap.add_argument("--end", default="2021-12-31")
    ap.add_argument("--factor", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    store_dir = DataRoots.resolve(store=args.store).store_dir(orats.VENDOR)
    dates = [d for d in storemod.available_dates(store_dir) if args.start <= d <= args.end]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        parts = list(pool.map(one, [(str(store_dir), d) for d in dates], chunksize=4))
    frame = pd.concat(parts, ignore_index=True)
    frame.to_csv(Path(args.out) / "orats_spx_monthly_oi.csv", index=False)
    bare = frame[frame["root"] == ""]
    wide = bare.pivot(index="date", columns="expiry", values="oi").reindex(dates)
    ratio = wide / wide.shift(1)
    jumps = ratio.stack()
    jumps = jumps[(jumps > args.factor) | (jumps < 1.0 / args.factor)]
    print(f"{len(dates)} dates {dates[0]} .. {dates[-1]}; rows without a symbol: {len(bare)}")
    print(f"jumps of an expiry's open interest by more than x{args.factor:g}: {len(jumps)}")
    for (date, expiry), r in jumps.items():
        prev, now = wide.shift(1).loc[date, expiry], wide.loc[date, expiry]
        print(f"  {date}  expiry {expiry}  {prev:>12,.0f} -> {now:>12,.0f}  x{r:.3f}")
    gone = wide.notna().sum(axis=1)
    print("dates with no monthly row under SPX:", list(gone[gone == 0].index))


if __name__ == "__main__":
    main()
