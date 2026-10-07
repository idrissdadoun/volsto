"""Addendum 2 of the barrier study: the attribution table and the turnover of the hedges.

    python scripts/barrier_attrib.py attrib
    python scripts/barrier_attrib.py turnover [--workers 2] [--entries D1 D2 …] [--out DIR]

``attrib`` reads ``positions.parquet`` and ``cells.parquet`` and writes
``positions_attrib.parquet`` (key ``cell, position, model``): the realised vol of the trade's
life, the vol premium ``dsig``, the entry vega, the vol carry and the hedged P&L ex carry
(:func:`volsto.studies.barrier_attrib.attribute`).  It is rebuilt, never edited, whenever
``positions.parquet`` has been reassembled.

``turnover`` reads the daily marks of each entry's trades and writes, per finished trade with
complete marks and per position, the trading its local-vol sticky-strike delta hedge did
(``turnover/<entry>.parquet``, then ``turnover.parquet``): ``turnover`` in units of delta and
``turnover_notional`` with each trade weighted by the forward over the entry spot.  The hedge
is held over the same intervals as in ``barrier_assemble.py``: to the knock for the knock-outs,
to the expiry for the vanilla structures.  No pricing in either pass.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import os
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barrier_history as run  # noqa: E402

from volsto.studies import barrier_attrib as ba  # noqa: E402
from volsto.studies import barrier_history as bh  # noqa: E402

ATTRIB = bh.RESULTS / "positions_attrib.parquet"
TURNOVER = bh.RESULTS / "turnover"
#: position → the daily table's mark column (bumped marks are ``<col>_up`` / ``<col>_dn``).
HEDGED = {"A1": "a1", "A2": "a2", "B3": "B3", "B4": "B4", "B5_2": "B5_2", "B5_6": "B5_6", "B6": "B6"}  # fmt: skip

_cells: dict[str, pd.DataFrame] = {}


def build_attrib(entries: list[str] | None = None) -> pd.DataFrame:
    """The attribution table, of every position or of the positions of ``entries``."""
    cells = pd.read_parquet(bh.RESULTS / "cells.parquet")
    pos = pd.read_parquet(
        bh.RESULTS / "positions.parquet", columns=["cell", "position", "model", "pnl_h"]
    )
    if entries is not None:
        cells = cells[cells["entry"].isin(entries)]
        pos = pos[pos["cell"].isin(set(cells["cell"]))]
    return ba.attribute(pos, cells, bh.load_ohlc()["close"]).reset_index(drop=True)


def cells_of(entry: str) -> pd.DataFrame:
    if "all" not in _cells:
        cols = ["cell", "entry", "K", "expiry", "tau1", "tau2", "close_T", "finished", "marks_complete"]  # fmt: skip
        c = pd.read_parquet(bh.RESULTS / "cells.parquet", columns=cols)
        ok = c["finished"].fillna(False).astype(bool) & c["marks_complete"].fillna(False).astype(bool)  # fmt: skip
        _cells["all"] = c[ok]
    c = _cells["all"]
    return c[c["entry"] == entry]


def turnover_job(args: tuple[str, str]) -> dict[str, Any]:
    entry, out_dir = args
    try:
        cal = run.context()["calendar"]
        cells = cells_of(entry)
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if cells.empty:
            pd.DataFrame().to_parquet(out / f"{entry}.parquet")
            return {"date": entry, "rows": 0}
        horizon = min(cells["expiry"].max(), run.LAST_DATA_DAY)
        dates = [d for d in cal[cal.index(entry) :] if d <= horizon]
        cols = ["date", "cell", "F", "DF"]
        for col in HEDGED.values():
            cols += [f"{col}_up", f"{col}_dn"]
        wanted = set(cells["cell"])
        parts = []
        for d in dates:
            p = run.DAILY / f"{d}.parquet"
            if p.exists():
                day = pd.read_parquet(p, columns=cols)
                parts.append(day[day["cell"].isin(wanted)])
        daily = pd.concat(parts, ignore_index=True)
        by_cell = {c: g.sort_values("date") for c, g in daily.groupby("cell")}
        e2 = 2.0 * bh.BUMP
        rows = []
        for r in cells.itertuples():
            g = by_cell.get(r.cell)
            if g is None:
                continue
            live = list(g["date"])
            n = len(live)
            F, DF = g["F"].to_numpy(float), g["DF"].to_numpy(float)
            K = float(r.K)
            for name, col in HEDGED.items():
                tau = r.tau1 if name == "A1" else (r.tau2 if name == "A2" else None)
                m = n
                if isinstance(tau, str) and tau != r.expiry:
                    m = live.index(tau)
                delta = (g[f"{col}_up"].to_numpy(float) - g[f"{col}_dn"].to_numpy(float)) / (
                    e2 * F * DF
                )
                held = delta[:m]
                close_w = F[m] / K if m < n else float(r.close_T)
                weights = np.append(F[:m] / K, close_w) if m > 0 else None
                rows.append(
                    {
                        "cell": r.cell,
                        "position": name,
                        "n_hedge": m,
                        "turnover": ba.turnover(held),
                        "turnover_notional": ba.turnover(held, weights),
                        "delta_first": float(held[0]) if m > 0 else 0.0,
                        "delta_last": float(held[-1]) if m > 0 else 0.0,
                        "delta_constant": bool(m > 0 and np.ptp(held) == 0.0),
                    }
                )
        tmp = out / f"{entry}.tmp"
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        os.replace(tmp, out / f"{entry}.parquet")
        return {"date": entry, "rows": len(rows)}
    except Exception as exc:
        return {"date": entry, "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-700:]}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("what", choices=["attrib", "turnover"])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--entries", nargs="*", default=[])
    ap.add_argument("--out", default=str(TURNOVER))
    args = ap.parse_args()
    if args.what == "attrib":
        frame = build_attrib()
        tmp = ATTRIB.with_suffix(".tmp")
        frame.to_parquet(tmp, index=False)
        os.replace(tmp, ATTRIB)
        done = frame["pnl_hx"].notna().sum()
        print(f"positions_attrib: {len(frame)} rows, {done} with an ex-vol hedged P&L")
        return
    if args.workers > 2:
        raise SystemExit("addendum 2: at most 2 processes for post-processing")
    out = Path(args.out)
    entries = args.entries or sorted(p.stem for p in run.ENTRIES.glob("*.parquet"))
    jobs = [(e, str(out)) for e in entries if not (out / f"{e}.parquet").exists()]
    if args.workers == 1:
        res = [turnover_job(j) for j in jobs]
    else:
        with Pool(args.workers) as pool:
            res = list(pool.imap_unordered(turnover_job, jobs, chunksize=4))
    bad = [r for r in res if "error" in r]
    print(f"turnover: {len(jobs)} entries, {len(bad)} errors")
    for r in sorted(bad, key=lambda r: r["date"])[:10]:
        print(f"  {r['date']}: {r['error']}\n{r.get('trace', '')}")
    if not args.entries and out == TURNOVER:
        parts = [pd.read_parquet(p) for p in sorted(out.glob("*.parquet")) if p.stat().st_size > 0]
        parts = [p for p in parts if len(p)]
        pd.concat(parts, ignore_index=True).to_parquet(bh.RESULTS / "turnover.parquet", index=False)
        print(f"turnover.parquet: {sum(len(p) for p in parts)} rows")


if __name__ == "__main__":
    main()
