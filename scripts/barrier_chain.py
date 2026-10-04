"""Quote-based measures at entry for the barrier study (BARRIER_STUDY_SPEC §6.6, §7.2, §10).

Per entry date, from the day's ORATS chain as the importer filters it (our own inversion of
each bid, mid and ask with our forward and discount factor) and from the vendor's rows:

* the **cost** of each vanilla structure: Σ over legs of |quantity| × half the quoted bid-ask in
  vol at the listed strike nearest the leg's strike, on the two expiries around the trade's
  expiry (interpolated in maturity), × the leg's vega (§6.6);
* the **fit quality** at the strikes the study uses — surface minus quote mid, in vol points,
  at K, B, the far fly strike 2B − K and B²/K — and the half-spread there (§10);
* the bid-ask width at K and B; the open interest within 1 % of the barrier over the total; a
  gamma-exposure proxy Σ γ·OI with calls positive and puts negative (§7.2).

    python scripts/barrier_chain.py [--workers 12]

Writes ``barrier_results/chain/<entry>.parquet`` (one row per cell; aggregates, no quote).
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

CHAIN = bh.RESULTS / "chain"
COSTED = [s for s in bh.STRUCTURES if s not in ("FWD",)]


def quote_at(
    points: pd.DataFrame, expiries: list[tuple[str, float]], T: float, x: float
) -> tuple[float, float]:
    """``(half-spread in vol, surface minus mid in vol)`` at the listed strike nearest ``x`` on
    the two listed expiries around ``T`` (linear in maturity; the nearest one outside)."""
    below = [e for e in expiries if e[1] <= T]
    above = [e for e in expiries if e[1] > T]
    picks = ([below[-1]] if below else []) + ([above[0]] if above else [])
    vals = []
    for name, t in picks:
        g = points[points["expiry"] == name]
        i = int(np.argmin(np.abs(g["strike"].to_numpy() - x)))
        row = g.iloc[i]
        if abs(row["strike"] / x - 1.0) > 0.03:  # nothing listed within 3 % of the strike
            vals.append((t, np.nan, np.nan))
            continue
        hs = 0.5 * (row["iv_ask_used"] - row["iv_bid_used"])
        vals.append((t, float(hs), float(row["model_minus_mid"])))
    if not vals:
        return np.nan, np.nan
    if len(vals) == 1 or vals[0][0] == vals[1][0]:
        return vals[0][1], vals[0][2]
    wgt = (T - vals[0][0]) / (vals[1][0] - vals[0][0])
    return (
        float((1 - wgt) * vals[0][1] + wgt * vals[1][1]),
        float((1 - wgt) * vals[0][2] + wgt * vals[1][2]),
    )


def chain_job(entry: str) -> dict[str, Any]:
    out = CHAIN / f"{entry}.parquet"
    try:
        from volsto.market import import_hdn as ih
        from volsto.market import import_orats as io
        from volsto.market import store as vendor_store

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
        points = pts.table.copy()
        points["model_minus_mid"] = (
            np.asarray(surf.implied_vol_k(points["k"].to_numpy(), points["T"].to_numpy()))
            - points["iv_mid"].to_numpy()
        )
        exp = points.groupby("expiry")["T"].first().sort_values()
        expiries = [(str(k), float(v)) for k, v in exp.items()]
        raw = vendor_store.load_chain(
            "orats", entry, "SPX", columns=["strike", "gamma", "cOi", "pOi"]
        )
        oi = (raw["cOi"] + raw["pOi"]).to_numpy(float)
        strikes = raw["strike"].to_numpy(float)
        total_oi = float(oi.sum())
        # dollar gamma per 1 % move, calls positive and puts negative (the convention that
        # dealers are long the calls and short the puts the public holds); rows whose vendor
        # gamma is not a per-dollar gamma below one are left out
        gamma = raw["gamma"].where((raw["gamma"] > 0) & (raw["gamma"] < 1.0), 0.0)
        gex = float((gamma * (raw["cOi"] - raw["pOi"])).sum() * market.spot**2 * 0.01 * 100.0)
        rows = []
        cache: dict[tuple[float, float], tuple[float, float]] = {}

        def q(T: float, x: float) -> tuple[float, float]:
            key = (round(T, 6), round(x, 4))
            if key not in cache:
                cache[key] = quote_at(points, expiries, T, x)
            return cache[key]

        for r in ent.itertuples():
            side, K, B, T = int(r.side), float(r.K), float(r.B), float(r.T)
            legs = bh.legs_from_json(r.legs)
            F, df = market.forward(T), market.df(T)
            rec: dict[str, Any] = {
                "entry": entry,
                "months": r.months,
                "side": side,
                "barrier": r.barrier,
            }
            for name, x in (("K", K), ("B", B), ("far", 2 * B - K), ("sym", B * B / K)):
                hs, err = q(T, x)
                rec[f"hs_{name}"] = 100.0 * hs
                rec[f"fit_{name}"] = 100.0 * err
            for name in COSTED:
                if name not in legs:
                    continue
                pieces = bh.expand_legs([(side, {name: legs[name]})])
                cost = 0.0
                for x, qty in zip(pieces["x"], pieces["q"], strict=True):
                    hs, _ = q(T, float(x))
                    vol = float(np.asarray(surf.implied_vol(float(x), T)).reshape(()))
                    s_ = vol * np.sqrt(T)
                    d1 = (np.log(F / x) + 0.5 * s_ * s_) / s_
                    vega = df * F * np.sqrt(T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi)
                    cost += abs(float(qty)) * hs * vega
                rec[f"cost_{name}"] = cost / K
            near = np.abs(strikes / B - 1.0) <= 0.01
            rec["oi_share_barrier"] = float(oi[near].sum() / total_oi) if total_oi > 0 else np.nan
            rec["gex_usd_per_pct"] = gex
            rows.append(rec)
        CHAIN.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(out, index=False)
        return {"date": entry}
    except Exception as exc:
        return {
            "date": entry,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-700:],
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--start", default="2007-01-01")
    ap.add_argument("--end", default="2026-12-31")
    args = ap.parse_args()
    entries = sorted(
        p.stem
        for p in run.ENTRIES.glob("*.parquet")
        if args.start <= p.stem <= args.end and not (CHAIN / f"{p.stem}.parquet").exists()
    )
    with Pool(args.workers) as pool:
        res = list(pool.imap_unordered(chain_job, entries))
    bad = [r for r in res if "error" in r]
    print(f"chain: {len(entries) - len(bad)} entries, {len(bad)} errors")
    for r in sorted(bad, key=lambda r: r["date"])[:15]:
        print(f"  {r['date']}: {r['error']}\n{r.get('trace', '')}")


if __name__ == "__main__":
    main()
