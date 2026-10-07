"""Vanilla prices from the quotes (BARRIER_STUDY_ADDENDUM §8).

For every entry and cell, the implied vol of the strikes ``K``, ``B``, ``2B − K``, ``B²/K`` and
the two strikes of the 1 % digital spread is read from the day's quotes as the importer filters
them (our own inversion of the mid): linear in strike between the two neighbouring listed
strikes of an expiry, then linear in total variance at fixed log-moneyness between the two
expiries around the trade's expiry.  From these, the quote-based premiums ``Pq_B3``, ``Pq_B4``,
``Pq_B5_2``, ``Pq_C9_7_1``, ``Pq_C9_8_1`` (fractions of the entry spot), with the surface's vol
at each strike beside the quote-based one.  A strike beyond the listed range of either expiry
is NaN and counted.  Aggregates only: no vendor quote is written.

    python scripts/barrier_quotes.py [--workers 4] [--entries D1 D2 …]

Writes ``barrier_results/quotes/<entry>.parquet`` and ``barrier_results/quotes.parquet``.
"""

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

from volsto.studies import barrier_history as bh  # noqa: E402

OUT = bh.RESULTS / "quotes"
STRUCTURES = ("B3", "B4", "B5_2", "C9_7_1", "C9_8_1")


def slice_vol(strikes: np.ndarray, vols: np.ndarray, x: float) -> float:
    """Implied vol at ``x``, linear between the two neighbouring listed strikes; NaN beyond the
    listed range."""
    if strikes.size < 2 or x < strikes[0] or x > strikes[-1]:
        return float("nan")
    return float(np.interp(x, strikes, vols))


def quote_job(entry: str) -> dict[str, Any]:
    out_path = OUT / f"{entry}.parquet"
    try:
        from volsto.market import import_hdn as ih
        from volsto.market import import_orats as io

        ctx = run.context()
        ent = pd.read_parquet(run.ENTRIES / f"{entry}.parquet")
        market = bh.load_day(entry, ctx["calendar"], ctx["ohlc"])
        if market.carried:
            return {"date": entry, "error": "carried surface"}
        surf = market.surface
        f = ih.HdnFilters()
        chain = io.load_day(entry, "SPX")
        fwds = ih.implied_forwards(chain, max_years=f.max_years, band=f.near_atm_band)
        _, pts = ih.to_grid_surface(chain, fwds, f)
        points = pts.table
        slices = []
        for name, g in points.groupby("expiry"):
            g = g.groupby("strike", as_index=False)["iv_mid"].mean().sort_values("strike")
            first = points[points["expiry"] == name].iloc[0]
            slices.append(
                (
                    float(first["T"]),
                    float(first["forward"]),
                    g["strike"].to_numpy(float),
                    g["iv_mid"].to_numpy(float),
                )
            )
        slices.sort(key=lambda s: s[0])
        times = np.array([s[0] for s in slices])
        cache: dict[tuple[float, float], float] = {}

        def qvol(T: float, x: float) -> float:
            key = (round(T, 6), round(x, 4))
            if key in cache:
                return cache[key]
            k = float(np.log(x / market.forward(T)))
            lo = int(np.searchsorted(times, T, side="right")) - 1
            picks = [i for i in (lo, lo + 1) if 0 <= i < len(slices)]
            if len(picks) < 2 or times[picks[0]] > T or times[picks[1]] < T:
                picks = picks[:1] if picks and abs(times[picks[0]] - T) < 1e-9 else []
            ws = []
            for i in picks:
                t_i, f_i, ks, vs = slices[i]
                v = slice_vol(ks, vs, f_i * np.exp(k))
                ws.append((t_i, v * v * t_i))
            if not ws or any(not np.isfinite(w) for _, w in ws):
                out = float("nan")
            elif len(ws) == 1:
                out = float(np.sqrt(ws[0][1] / ws[0][0]))
            else:
                a = (T - ws[0][0]) / (ws[1][0] - ws[0][0])
                out = float(np.sqrt(((1 - a) * ws[0][1] + a * ws[1][1]) / T))
            cache[key] = out
            return out

        rows = []
        for r in ent.itertuples():
            side, K, B, T = int(r.side), float(r.K), float(r.B), float(r.T)
            legs = bh.legs_from_json(r.legs)
            F, df = np.asarray(market.forward(T)), np.asarray(market.df(T))
            rec: dict[str, Any] = {
                "entry": entry,
                "months": r.months,
                "side": side,
                "barrier": r.barrier,
            }
            for name, x in (("K", K), ("B", B), ("far", 2 * B - K), ("sym", B * B / K)):
                vq = qvol(T, x) if x > 0 else float("nan")
                vs = (
                    float(np.asarray(surf.implied_vol(x, T)).reshape(())) if x > 0 else float("nan")
                )
                rec[f"vq_{name}"], rec[f"vs_{name}"] = vq, vs
                if np.isfinite(vq):  # the vanilla at this strike, both ways (check 11.6)
                    xa, Ta = np.asarray(x), np.asarray(T)
                    rec[f"cq_{name}"] = float(bh.black(F, xa, Ta, np.asarray(vq), side, df)) / K
                    rec[f"cs_{name}"] = float(bh.black(F, xa, Ta, np.asarray(vs), side, df)) / K
            for name in STRUCTURES:
                total = 0.0
                for leg in legs[name]:
                    if leg.strike <= 0:  # a put struck at or below zero is worth nothing
                        continue
                    v = qvol(T, leg.strike)
                    if not np.isfinite(v):
                        total = float("nan")
                        break
                    total += leg.qty * float(
                        bh.black(F, np.asarray(leg.strike), np.asarray(T), np.asarray(v), side, df)
                    )
                rec[f"Pq_{name}"] = total / K
            rows.append(rec)
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        os.replace(tmp, out_path)
        return {"date": entry}
    except Exception as exc:
        return {"date": entry, "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-700:]}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--entries", nargs="*", default=[])
    args = ap.parse_args()
    entries = args.entries or sorted(p.stem for p in run.ENTRIES.glob("*.parquet"))
    jobs = [e for e in entries if not (OUT / f"{e}.parquet").exists()]
    if args.workers == 1:
        res = [quote_job(e) for e in jobs]
    else:
        with Pool(args.workers) as pool:
            res = list(pool.imap_unordered(quote_job, jobs, chunksize=2))
    bad = [r for r in res if "error" in r]
    print(f"quotes: {len(jobs)} entries, {len(bad)} errors")
    for r in sorted(bad, key=lambda r: r["date"])[:10]:
        print(f"  {r['date']}: {r['error']}\n{r.get('trace', '')}")
    files = sorted(OUT.glob("*.parquet"))
    if files and not args.entries:
        pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).to_parquet(
            bh.RESULTS / "quotes.parquet", index=False
        )


if __name__ == "__main__":
    main()
