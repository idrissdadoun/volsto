"""Check 11.5 of addendum 1: the skew local vol implies at the barrier, from restarts.

Monthly subset, standard-deviation barriers: for each cell and each third of the life, the
entry day's local vol is restarted at the middle of the third with the spot at the barrier
(:func:`volsto.studies.barrier_theory.restart_model`, 20,000 antithetic paths) and its
normalised skew ``n_lv`` at the time left is read with the touch test's centred difference.
The ``om``-weighted average is compared with ``n_eff[lv]`` (the skew the local-vol knock-out
price implies, features2).  Reported, no pass or fail.

    python scripts/barrier_restarts.py [--workers 4] [--limit 0]

Writes ``barrier_results/restarts/<entry>.parquet`` and ``barrier_results/restarts.parquet``,
and appends the regression to ``outputs/interview/checks2.md``.
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
import barrier_touches as tc  # noqa: E402

from volsto.market.dupire import LocalVolSurface  # noqa: E402
from volsto.studies import barrier_history as bh  # noqa: E402
from volsto.studies import barrier_theory as bt  # noqa: E402

OUT = bh.RESULTS / "restarts"


def restart_job(entry: str) -> dict[str, Any]:
    try:
        ctx = run.context()
        ent = pd.read_parquet(run.ENTRIES / f"{entry}.parquet")
        ent = ent[ent["barrier"].str.startswith("s")]
        f2 = pd.read_parquet(bh.RESULTS / "features2" / f"{entry}.parquet")
        ent = ent.merge(
            f2[["months", "side", "barrier", "sig_B", "n_eff_lv", "om_1", "om_2", "om_3"]],
            on=["months", "side", "barrier"],
        )
        market = bh.load_day(entry, ctx["calendar"], ctx["ohlc"])
        lv0 = LocalVolSurface.from_implied(market.surface, bh.LV_CONFIG)
        rows = []
        for r in ent.itertuples():
            T, B = float(r.T), float(r.B)
            n_lv = []
            for frac in (
                1 / 6,
                1 / 2,
                5 / 6,
            ):  # the middle of each third: time left 5/6, 1/2, 1/6 of T
                tau, theta = frac * T, (1 - frac) * T
                try:
                    model = bt.restart_model(lv0, tau, B)
                    paths = bh.simulate_day(
                        model, np.array([theta]), n_paths=20_000, seed=bh.seed_of(entry)
                    )
                    s_T = paths.close[:, 0]
                    fwd = float(s_T.mean())
                    atm_price = float(np.maximum(s_T - fwd, 0.0).mean())
                    sig = tc.implied(atm_price, fwd, fwd, theta, 1)
                    h = max(0.5 * sig * np.sqrt(theta), 0.01)
                    up = tc.implied(
                        float(np.maximum(s_T - fwd * np.exp(h), 0.0).mean()),
                        fwd,
                        fwd * np.exp(h),
                        theta,
                        1,
                    )
                    dn = tc.implied(
                        float(np.maximum(fwd * np.exp(-h) - s_T, 0.0).mean()),
                        fwd,
                        fwd * np.exp(-h),
                        theta,
                        -1,
                    )
                    n_lv.append(-(up - dn) / (2 * h) * np.sqrt(theta))
                except Exception:
                    n_lv.append(np.nan)
            om = np.array([r.om_1, r.om_2, r.om_3])
            rows.append(
                {
                    "entry": entry,
                    "months": r.months,
                    "side": r.side,
                    "barrier": r.barrier,
                    "n_eff_lv": r.n_eff_lv,
                    "n_lv_1": n_lv[0],
                    "n_lv_2": n_lv[1],
                    "n_lv_3": n_lv[2],
                    "n_lv_restart": float(np.sum(om * np.array(n_lv))),
                }
            )
        OUT.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(OUT / f"{entry}.parquet", index=False)
        return {"date": entry}
    except Exception as exc:
        return {
            "date": entry,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-600:],
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    entries = bh.monthly_subset(
        sorted(p.stem for p in (bh.RESULTS / "features2").glob("*.parquet"))
    )
    jobs = [e for e in entries if not (OUT / f"{e}.parquet").exists()]
    if args.limit:
        jobs = jobs[: args.limit]
    if args.workers == 1:
        res = [restart_job(e) for e in jobs]
    else:
        with Pool(args.workers) as pool:
            res = list(pool.imap_unordered(restart_job, jobs))
    bad = [r for r in res if "error" in r]
    print(f"restarts: {len(jobs)} entries, {len(bad)} errors")
    for r in bad[:5]:
        print(f"  {r['date']}: {r['error']}\n{r.get('trace', '')}")
    files = sorted(OUT.glob("*.parquet"))
    if not files or args.limit:
        return
    x = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    x.to_parquet(bh.RESULTS / "restarts.parquet", index=False)
    out = [
        f"- **11.5 n_eff[lv] against the local-vol restarts** (monthly subset, {x['entry'].nunique()} entries, sd barriers; reported, no pass or fail):"
    ]
    for side in (1, -1):
        g = x[(x["side"] == side) & x["n_eff_lv"].notna() & x["n_lv_restart"].notna()]
        if len(g) < 10:
            continue
        slope, const = np.polyfit(g["n_lv_restart"], g["n_eff_lv"], 1)
        corr = float(np.corrcoef(g["n_lv_restart"], g["n_eff_lv"])[0, 1])
        out.append(
            f"  {'call' if side > 0 else 'put'} side, {len(g)} cells: n_eff[lv] = {const:.3f} + {slope:.2f} x n_lv(restart), "
            f"correlation {corr:.2f}; medians {g['n_eff_lv'].median():.3f} against {g['n_lv_restart'].median():.3f}"
        )
    print("\n".join(out))
    with (bh.OUT / "checks2.md").open("a") as fh:
        fh.write("\n".join(out) + "\n")


if __name__ == "__main__":
    main()
