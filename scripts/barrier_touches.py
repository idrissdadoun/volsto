"""The touch test of addendum 1 (BARRIER_STUDY_ADDENDUM §6).

For every trade knocked under the daily rule (standard-deviation barriers, both sides), on the
knock day ``tau`` with time left ``theta``: what the touch-day surface shows (at-the-money vol,
skew, normalised skew ``n = −skew·√θ``; the day's marks of the tight limit, the fly and C8),
against three predictions made from the entry day — the stationary smile (entry-day smile by
moneyness at tenor ``theta``), the sticky-strike smile (entry-day vols by absolute strike, read
at the touch-day forward) and local vol (the entry day's Dupire surface restarted at the touch
day's time and spot, 20,000 antithetic paths) — and the derived quantities of §6.4.

All skews use one centred difference of half-width ``h = max(0.5·σ_ATM(θ)·√θ, 0.01)`` in
log-strike around the forward of tenor ``theta`` (σ_ATM the touch-day one).  Skipped and
counted: fewer than 3 trading days left; a carried touch-day surface.

    python scripts/barrier_touches.py [--workers 4] [--entries D1 D2 …]

Writes ``barrier_results/touches/<entry>.parquet`` and ``barrier_results/touches.parquet``.
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
from scipy.optimize import brentq  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barrier_history as run  # noqa: E402

from volsto.market.dupire import LocalVolSurface  # noqa: E402
from volsto.studies import barrier_history as bh  # noqa: E402
from volsto.studies import barrier_theory as bt  # noqa: E402

OUT = bh.RESULTS / "touches"
RESTART_PATHS = 20_000
MARKS = ("B6", "B5_2", "C8")

_cells: dict[str, pd.DataFrame] = {}


def knocked_cells(entry: str) -> pd.DataFrame:
    """The entry's finished sd-barrier cells knocked under the daily rule before expiry."""
    if "all" not in _cells:
        c = pd.read_parquet(
            bh.RESULTS / "cells.parquet", columns=["cell", "entry", "barrier", "tau1", "finished"]
        )
        c = c[c["finished"].fillna(False).astype(bool) & c["tau1"].notna() & c["barrier"].str.startswith("s")]
        _cells["all"] = c
    c = _cells["all"]
    return c[c["entry"] == entry]


def vol_at(surface: Any, k: float, theta: float) -> float:
    return float(np.asarray(surface.implied_vol_k(k, theta)).reshape(()))


def smile(surface: Any, k0: float, theta: float, h: float) -> tuple[float, float]:
    """``(vol at k0, centred slope over ±h)`` in log-moneyness against the surface's forward."""
    return vol_at(surface, k0, theta), (vol_at(surface, k0 + h, theta) - vol_at(surface, k0 - h, theta)) / (2 * h)


def implied(price: float, F: float, K: float, theta: float, cp: int) -> float:
    """Black implied vol of an undiscounted price (NaN outside the no-arbitrage range)."""
    one = np.asarray(1.0)

    def gap(v: float) -> float:
        return float(bh.black(np.asarray(F), np.asarray(K), np.asarray(theta), np.asarray(v), cp, one)) - price

    try:
        return float(brentq(gap, 1e-4, 5.0, xtol=1e-10))
    except ValueError:
        return float("nan")


def touch_job(entry: str) -> dict[str, Any]:
    out_path = OUT / f"{entry}.parquet"
    try:
        ctx = run.context()
        cal, ohlc = ctx["calendar"], ctx["ohlc"]
        knocked = knocked_cells(entry)
        ent = pd.read_parquet(run.ENTRIES / f"{entry}.parquet")
        ent["cell"] = (
            ent["entry"] + "|" + ent["months"].astype(str) + "|" + ent["side"].astype(str) + "|" + ent["barrier"]
        )
        ent = ent.merge(knocked[["cell", "tau1"]], on="cell")
        counts = {"knocked": len(ent), "short": 0, "carried": 0, "no_mark": 0}
        if ent.empty:
            OUT.mkdir(parents=True, exist_ok=True)
            pd.DataFrame().to_parquet(out_path)
            return {"date": entry, **counts}
        m0 = bh.load_day(entry, cal, ohlc)
        surf0 = m0.surface
        lv0 = LocalVolSurface.from_implied(surf0, bh.LV_CONFIG)
        S0 = m0.spot
        markets: dict[str, bh.DayMarket] = {}
        dailies: dict[str, pd.DataFrame] = {}
        restarts: dict[tuple[str, str], dict[str, Any]] = {}
        rows = []
        for r in ent.itertuples():
            tau, expiry = str(r.tau1), str(r.expiry)
            side, K, B = int(r.side), float(r.K), float(r.B)
            left = cal.index(expiry) - cal.index(tau)
            if left < 3:
                counts["short"] += 1
                continue
            if tau not in markets:
                markets[tau] = bh.load_day(tau, cal, ohlc)
                p = run.DAILY / f"{tau}.parquet"
                dailies[tau] = pd.read_parquet(p, columns=["cell", "DF", *MARKS]) if p.exists() else pd.DataFrame()
            mt = markets[tau]
            if mt.carried:
                counts["carried"] += 1
                continue
            theta = bh.year_fraction(tau, expiry)
            t_tau = bh.year_fraction(entry, tau)
            st = mt.spot
            sig_real = vol_at(mt.surface, 0.0, theta)
            h = max(0.5 * sig_real * np.sqrt(theta), 0.01)
            _, skew_real = smile(mt.surface, 0.0, theta, h)
            sig_stat, skew_stat = smile(surf0, 0.0, theta, h)
            # sticky strike: the entry-day smile by absolute strike, read at the touch-day forward
            f_tau = mt.forward(theta)
            k_ss = float(np.log(f_tau / m0.forward(theta)))
            sig_ss, skew_ss = smile(surf0, k_ss, theta, h)
            rec: dict[str, Any] = {
                "cell": r.cell,
                "entry": entry,
                "months": r.months,
                "side": side,
                "barrier": r.barrier,
                "tau": tau,
                "theta": theta,
                "days_left": left,
                "S_tau": st / S0,
                "h": h,
                "sig_real": sig_real,
                "skew_real": skew_real,
                "n_real": -skew_real * np.sqrt(theta),
                "sig_stat": sig_stat,
                "skew_stat": skew_stat,
                "n_stat": -skew_stat * np.sqrt(theta),
                "sig_ss": sig_ss,
                "n_ss": -skew_ss * np.sqrt(theta),
            }
            # the day's marks, valued at expiry (over the discount factor), fractions of S0
            day = dailies[tau]
            row = day[day["cell"] == r.cell] if len(day) else day
            if len(row):
                dft = float(row["DF"].iloc[0])
                for name in MARKS:
                    rec[f"mark_{name}"] = float(row[name].iloc[0]) / dft / K
                c8_cash = float(row["C8"].iloc[0]) / K
                rec["resid"] = -side * c8_cash
                rec["resid_pv"] = -side * float(row["C8"].iloc[0]) / dft * float(r.DF0) / K
                rr, qq = bt.carry_rates(mt.df(theta), f_tau, st, theta)
                rec["resid_skew"] = rec["resid"] + side * bt.c8_bs(side, st, K, B, theta, sig_real, rr, qq) / K
                rec["kap_touch"] = float(bt.kap(K, B, sig_real, theta))
            else:
                counts["no_mark"] += 1
            # local-vol restart at (tau, S_tau): once per (maturity, knock day)
            key = (expiry, tau)
            if key not in restarts:
                try:
                    model = bt.restart_model(lv0, t_tau, st)
                    paths = bh.simulate_day(
                        model, np.array([theta]), n_paths=RESTART_PATHS, seed=bh.seed_of(tau)
                    )
                    s_T = paths.close[:, 0]
                    fwd = float(s_T.mean())
                    vols = []
                    for kk in (-h, 0.0, h):
                        strike = fwd * np.exp(kk)
                        cp = 1 if kk >= 0 else -1
                        price = float(np.maximum(cp * (s_T - strike), 0.0).mean())
                        vols.append(implied(price, fwd, strike, theta, cp))
                    restarts[key] = {"s_T": s_T, "sig": vols[1], "skew": (vols[2] - vols[0]) / (2 * h)}
                except Exception as exc:
                    restarts[key] = {"error": f"{type(exc).__name__}: {exc}"}
            rs = restarts[key]
            if "error" not in rs:
                rec["sig_lv"] = rs["sig"]
                rec["n_lv"] = -rs["skew"] * np.sqrt(theta)
                legs = bh.legs_from_json(r.legs)
                for name in MARKS:
                    rec[f"lv_{name}"] = float(bh.legs_payoff(legs[name], side, rs["s_T"]).mean()) / K
            else:
                rec["restart_error"] = rs["error"]
            move = float(np.log(st / S0))
            rec["ssr_at_touch"] = (
                (sig_real - sig_stat) / (skew_stat * move) if abs(move) >= 0.01 and skew_stat != 0 else np.nan
            )
            rec["skew_ratio"] = rec["n_real"] / rec["n_stat"] if rec["n_stat"] != 0 else np.nan
            rows.append(rec)
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        pd.DataFrame(rows).to_parquet(tmp, index=False)
        os.replace(tmp, out_path)
        return {"date": entry, **counts}
    except Exception as exc:
        return {"date": entry, "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-800:]}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--entries", nargs="*", default=[])
    args = ap.parse_args()
    entries = args.entries or sorted(p.stem for p in run.ENTRIES.glob("*.parquet"))
    jobs = [e for e in entries if not (OUT / f"{e}.parquet").exists()]
    if args.workers == 1:
        res = [touch_job(e) for e in jobs]
    else:
        with Pool(args.workers) as pool:
            res = list(pool.imap_unordered(touch_job, jobs, chunksize=2))
    bad = [r for r in res if "error" in r]
    tot = {k: sum(r.get(k, 0) for r in res) for k in ("knocked", "short", "carried", "no_mark")}
    print(f"touches: {len(jobs)} entries, {len(bad)} errors; {tot}")
    for r in sorted(bad, key=lambda r: r["date"])[:10]:
        print(f"  {r['date']}: {r['error']}\n{r.get('trace', '')}")
    files = [p for p in sorted(OUT.glob("*.parquet")) if p.stat().st_size > 0]
    if files and not args.entries:
        parts = [pd.read_parquet(p) for p in files]
        parts = [p for p in parts if len(p)]
        pd.concat(parts, ignore_index=True).to_parquet(bh.RESULTS / "touches.parquet", index=False)
        print(f"touches.parquet: {sum(len(p) for p in parts)} knocked trades")


if __name__ == "__main__":
    main()
