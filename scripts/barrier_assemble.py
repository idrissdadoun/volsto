"""Assemble the outcomes of the barrier study (:mod:`volsto.studies.barrier_history`).

For each entry date: the realised knocks on the official closes / highs / lows, the payoffs,
and for every position of every cell (the two knock-outs under local vol, every vanilla
structure) the unhedged and delta-hedged P&L of spec §5.3 from the daily marks.

    python scripts/barrier_assemble.py [--start ...] [--end ...] [--workers 12]

Writes ``barrier_results/outcomes/<entry>.parquet`` (one row per cell and position, amounts in
fractions of the entry spot) and, at the end, ``barrier_results/positions.parquet`` (all of
them) and ``barrier_results/cells.parquet`` (one row per cell: the entry table plus the
realised path).  Derived per-trade tables: they stay under ``outputs/``.
"""

# ruff: noqa: B023 — the helpers below close over the cell of the loop and are called within it
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

import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barrier_history as run  # noqa: E402

from volsto.studies import barrier_history as bh  # noqa: E402

OUTCOMES = bh.RESULTS / "outcomes"
SOLD_AT_TOUCH = ("C7", "C8", "C9_7_1", "C9_7_2", "C9_8_1", "C9_8_2")


def assemble_entry(entry: str) -> dict[str, Any]:
    try:
        ctx = run.context()
        cal, ohlc = ctx["calendar"], ctx["ohlc"]
        ent = pd.read_parquet(run.ENTRIES / f"{entry}.parquet")
        ent["cell"] = (
            ent["entry"]
            + "|"
            + ent["months"].astype(str)
            + "|"
            + ent["side"].astype(str)
            + "|"
            + ent["barrier"]
        )
        last = run.LAST_DATA_DAY
        horizon = min(ent["expiry"].max(), last)
        i0 = cal.index(entry)
        dates = [d for d in cal[i0:] if d <= horizon]
        parts = []
        for d in dates:
            p = run.DAILY / f"{d}.parquet"
            if p.exists():
                day = pd.read_parquet(p)
                parts.append(day[day["cell"].str.startswith(entry + "|")])
        daily = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["cell"])
        by_cell = {c: g.sort_values("date") for c, g in daily.groupby("cell")}
        rows: list[dict[str, Any]] = []
        cells: list[dict[str, Any]] = []
        e2 = 2.0 * bh.BUMP
        for r in ent.itertuples():
            side, K, B, expiry = int(r.side), float(r.K), float(r.B), str(r.expiry)
            finished = expiry <= last
            live_dates = [d for d in cal[i0:] if d < expiry]
            info: dict[str, Any] = {"cell": r.cell, "finished": finished, "n_life": len(live_dates)}
            if not finished:
                cells.append(info)
                continue
            obs = [d for d in cal[i0 + 1 :] if d <= expiry]
            tau1, tau2 = bh.knock_dates(side, B, ohlc, obs)
            c_T = float(ohlc.loc[expiry, "close"])
            itm = side * (c_T - K) > 0
            info.update(
                tau1=tau1,
                tau2=tau2,
                close_T=c_T / K,
                itm=itm,
                beyond=side * (c_T - B) > 0,
                regret=bool(tau1 is not None and itm and side * (c_T - B) <= 0),
            )
            g = by_cell.get(r.cell)
            complete = g is not None and list(g["date"]) == live_dates
            info["marks_complete"] = bool(complete)
            info["carried_days"] = int(g["carried"].sum()) if g is not None else -1
            legs = bh.legs_from_json(r.legs)
            n = len(live_dates)
            if complete:
                assert g is not None
                F = g["F"].to_numpy(float)
                DF = g["DF"].to_numpy(float)
                S = g["S"].to_numpy(float)
                scale = e2 * F * DF

            def account(
                name: str,
                model: str,
                premium: float,
                terminal: float,
                closing: float,
                n_hedge: int,
                col: str,
                extra: dict[str, Any] | None = None,
            ) -> None:
                rec: dict[str, Any] = {
                    "cell": r.cell,
                    "position": name,
                    "model": model,
                    "premium": premium / K,
                    "terminal": terminal / K,
                    "pnl_u": (terminal - premium / r.DF0) / K,
                }
                if complete:
                    assert g is not None
                    v = g[col].to_numpy(float)
                    delta = (
                        g[f"{col}_up"].to_numpy(float) - g[f"{col}_dn"].to_numpy(float)
                    ) / scale
                    out = bh.position_outcome(
                        v,
                        delta,
                        F,
                        DF,
                        premium=premium,
                        df0=float(r.DF0),
                        terminal=terminal,
                        closing_forward=closing,
                        n_hedge=n_hedge,
                        n_life=n,
                    )
                    rec.update({k: x / K for k, x in out.items()})
                    rec["delta0"] = float(delta[0] * F[0] * DF[0] / S[0])  # spot delta at entry
                    rec["delta0_f"] = float(delta[0])
                if extra:
                    rec.update(extra)
                rows.append(rec)

            def end_of(tau: str | None) -> tuple[int, bool]:
                """(hedge intervals, the knock is before expiry)."""
                if tau is None or tau == expiry:
                    return n, False
                return live_dates.index(tau), True

            # the knock-outs under local vol
            for name, tau, col, prem in (("A1", tau1, "a1", r.lv_a1), ("A2", tau2, "a2", r.lv_a2)):
                m, early = end_of(tau)
                payoff = bh.knock_out_payoff(side, K, c_T, tau is not None)
                closing = c_T
                if tau is not None:
                    level = float(ohlc.loc[tau, "close"])
                    if name == "A2":
                        o = float(ohlc.loc[tau, "open"])
                        level = o if side * (o - B) >= 0 else B
                    closing = level
                    if early and complete:
                        closing = float(F[m] * level / S[m])
                account(name, "lv", float(prem), payoff, closing, m, col)
            # the vanilla structures
            m1, early1 = end_of(tau1)
            for name in bh.STRUCTURES:
                if name not in legs:
                    continue
                payoff = float(bh.legs_payoff(legs[name], side, c_T))
                premium = float(getattr(r, f"P_{name}"))
                if name in SOLD_AT_TOUCH and tau1 is not None:
                    if early1 and complete:
                        assert g is not None
                        sale = float(g[name].to_numpy(float)[m1])
                        account(
                            name,
                            "surface",
                            premium,
                            sale / float(DF[m1]),
                            float(F[m1]),
                            m1,
                            name,
                            {"sale": sale / K, "sold": True},
                        )
                    elif early1:
                        rows.append({"cell": r.cell, "position": name, "model": "surface",
                                     "premium": premium / K, "sold": True})  # fmt: skip
                    else:  # knocked on the expiry date: sold for its payoff
                        account(name, "surface", premium, payoff, c_T, n, name,
                                {"sale": payoff / K, "sold": True})  # fmt: skip
                    continue
                account(name, "surface", premium, payoff, c_T, n, name)
            cells.append(info)
        OUTCOMES.mkdir(parents=True, exist_ok=True)
        pos = pd.DataFrame(rows)
        pos.to_parquet(OUTCOMES / f"{entry}.parquet", index=False)
        cell_frame = ent.drop(columns=["legs"]).merge(pd.DataFrame(cells), on="cell", how="left")
        cell_frame.to_parquet(OUTCOMES / f"{entry}.cells.parquet", index=False)
        return {"date": entry, "positions": len(pos)}
    except Exception as exc:
        return {
            "date": entry,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-900:],
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--start", default="2007-01-03")
    ap.add_argument("--end", default=run.LAST_DATA_DAY)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    entries = sorted(
        p.stem for p in run.ENTRIES.glob("*.parquet") if args.start <= p.stem <= args.end
    )
    with Pool(args.workers) as pool:
        res = list(pool.imap_unordered(assemble_entry, entries))
    bad = [r for r in res if "error" in r]
    print(f"assembled {len(entries) - len(bad)} entries, {len(bad)} errors")
    for r in sorted(bad, key=lambda r: r["date"])[:20]:
        print(f"  {r['date']}: {r['error']}\n{r['trace']}")
    pos = pd.concat(
        [pd.read_parquet(p) for p in sorted(OUTCOMES.glob("*.parquet")) if ".cells" not in p.name],
        ignore_index=True,
    )
    cells = pd.concat(
        [pd.read_parquet(p) for p in sorted(OUTCOMES.glob("*.cells.parquet"))], ignore_index=True
    )
    pos.to_parquet(bh.RESULTS / "positions.parquet", index=False)
    cells.to_parquet(bh.RESULTS / "cells.parquet", index=False)
    print(f"positions {len(pos)}, cells {len(cells)} ({int(cells['finished'].sum())} finished)")


if __name__ == "__main__":
    main()
