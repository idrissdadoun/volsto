"""Local-vol bucket statistics at entry (BARRIER_STUDY_ADDENDUM §5.2).

One extra pass per entry date with the entry pass's own path set (same model, grid and seed):
per cell and third of the life, under the continuous and the daily rule, the first-touch
probability, the probability of finishing beyond the barrier, the mean time left and the mean
payoff of the tight limit, the fly and C8 given a touch there
(:class:`volsto.studies.barrier_theory.Buckets`), with the knock-out values of the pass so that
check 11.1 can compare them with the entry pass's.

    python scripts/barrier_buckets.py [--workers 4] [--limit 5]

Writes ``barrier_results/buckets_lv/<entry>.parquet`` and, at the end,
``barrier_results/buckets_lv.parquet``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barrier_history as run  # noqa: E402

from volsto.studies import barrier_history as bh  # noqa: E402
from volsto.studies import barrier_theory as bt  # noqa: E402

OUT = bh.RESULTS / "buckets_lv"


def bucket_job(entry: str) -> dict[str, Any]:
    out = OUT / f"{entry}.parquet"
    t0 = time.perf_counter()
    try:
        ctx = run.context()
        ent = pd.read_parquet(run.ENTRIES / f"{entry}.parquet")
        market = bh.load_day(entry, ctx["calendar"], ctx["ohlc"])
        days, times = bh.future_times(entry, ctx["calendar"])
        buckets = bt.Buckets(ent, days, times)
        ko = run._ko_table(
            bh.local_vol_model(market.surface),
            times,
            bh.seed_of(entry),
            ent,
            days,
            ent["P_B6"].to_numpy(float),
            ent["DF0"].to_numpy(float),
            on_chunk=buckets,
        )
        res = ent[["entry", "months", "side", "barrier"]].copy()
        for key, col in buckets.columns("lv").items():
            res[key] = col
        # the pass's own knock-out values: they must be the entry pass's (same paths)
        res["lv_a1_again"] = ko["a1"].to_numpy()
        res["lv_a1_raw_again"] = ko["a1_raw"].to_numpy()
        res["lv_p1_again"] = ko["p1"].to_numpy()
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        res.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        return {"date": entry, "seconds": time.perf_counter() - t0}
    except Exception as exc:
        return {"date": entry, "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-700:]}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--entries", nargs="*", default=[])
    args = ap.parse_args()
    entries = args.entries or sorted(p.stem for p in run.ENTRIES.glob("*.parquet"))
    jobs = [e for e in entries if not (OUT / f"{e}.parquet").exists()]
    if args.limit:
        jobs = jobs[: args.limit]
    t0 = time.perf_counter()
    if args.workers == 1:
        res = [bucket_job(e) for e in jobs]
    else:
        with Pool(args.workers) as pool:
            res = list(pool.imap_unordered(bucket_job, jobs))
    bad = [r for r in res if "error" in r]
    print(f"buckets: {len(jobs)} run, {len(bad)} errors, {time.perf_counter() - t0:.0f} s")
    for r in sorted(bad, key=lambda r: r["date"])[:10]:
        print(f"  {r['date']}: {r['error']}\n{r.get('trace', '')}")
    files = sorted(OUT.glob("*.parquet"))
    if files and not args.limit and not args.entries:
        pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).to_parquet(
            bh.RESULTS / "buckets_lv.parquet", index=False
        )


if __name__ == "__main__":
    main()
