"""Pilot checks of the barrier study (BARRIER_STUDY_SPEC §10) that need a computation beyond
the pilot's own outputs: check 3 (sticky-strike bump of the 3m at-the-money call against the
Black delta) and check 5 (local-vol Monte Carlo against the PDE on the continuous up-and-out
call — the only knock-out the library's PDE prices).

    python scripts/barrier_pilot_checks.py --dates 2024-01-02 2024-02-12 2024-03-25
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barrier_history as run

from volsto.pde.lsv1f import LSV1FPDE
from volsto.studies import barrier_history as bh


def check3(dates: list[str]) -> pd.DataFrame:
    ctx = run.context()
    rows = []
    for d in dates:
        m = bh.load_day(d, ctx["calendar"], ctx["ohlc"])
        T = bh.year_fraction(d, bh.expiry_date(d, 3, ctx["calendar"]))
        K = m.spot
        legs = (bh.Leg("opt", K, 1.0),)
        up, dn = (bh.legs_value(legs, 1, m.surface, T, r) for r in (1 + bh.BUMP, 1 - bh.BUMP))
        F, df = m.forward(T), m.df(T)
        bump = (up - dn) / (2 * bh.BUMP * F * df)
        vol = float(np.asarray(m.surface.implied_vol(K, T)).reshape(()))
        d1 = (np.log(F / K) + 0.5 * vol * vol * T) / (vol * np.sqrt(T))
        rows.append(
            {
                "date": d,
                "bump_delta": bump,
                "black_delta": float(ndtr(d1)),
                "diff": bump - float(ndtr(d1)),
            }
        )
    return pd.DataFrame(rows)


def check5(dates: list[str], n_x: int) -> pd.DataFrame:
    ctx = run.context()
    rows = []
    for d in dates:
        ent = pd.read_parquet(run.ENTRIES / f"{d}.parquet")
        m = bh.load_day(d, ctx["calendar"], ctx["ohlc"])
        pde = LSV1FPDE(bh.local_vol_model(m.surface), n_x=n_x)
        calls = ent[ent["side"] == 1]
        for r in calls.itertuples():
            if r.B <= r.K * 1.002:
                continue
            value = pde.knock_out_call(float(r.K), float(r.T), float(r.B), "up").price_at(m.spot)
            rows.append(
                {
                    "date": d,
                    "months": r.months,
                    "barrier": r.barrier,
                    "mc": r.lv_a2_raw / r.K,
                    "se": r.lv_a2_raw_se / r.K,
                    "mc_controlled": r.lv_a2 / r.K,
                    "se_controlled": r.lv_a2_se / r.K,
                    "pde": value / r.K,
                }
            )
    out = pd.DataFrame(rows)
    out["z"] = (out["mc"] - out["pde"]) / out["se"].where(out["se"] > 0)
    out["diff_bp"] = 1e4 * (out["mc"] - out["pde"])
    out["z_controlled"] = (out["mc_controlled"] - out["pde"]) / out["se_controlled"].where(
        out["se_controlled"] > 0
    )
    out["diff_controlled_bp"] = 1e4 * (out["mc_controlled"] - out["pde"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--n-x", type=int, default=800)
    args = ap.parse_args()
    c3 = check3(args.dates)
    print("check 3 (pass: |diff| <= 2e-3)")
    print(c3.round(5).to_string(index=False))
    c5 = check5(args.dates, args.n_x)
    c5.to_csv(bh.OUT / "pilot_check5.csv", index=False)
    print(f"check 5: {len(c5)} continuous up-and-out calls, PDE n_x={args.n_x}")
    cols = ["diff_bp", "z", "diff_controlled_bp", "z_controlled"]
    print(c5.groupby("months")[cols].agg(["mean", "min", "max"]).round(2).to_string())
    bad = c5[c5["z"].abs() > 2]
    print(
        f"  raw Monte Carlo beyond 2 standard errors: {len(bad)} of {len(c5)}; "
        f"max |z| {c5['z'].abs().max():.2f}"
    )
    print(bad.round(5).to_string(index=False))
    n = int((c5["z_controlled"].abs() > 2).sum())
    print(
        f"  controlled estimate beyond 2 of its standard errors: {n} of {len(c5)}; "
        f"max |difference| {c5['diff_controlled_bp'].abs().max():.2f} bp of spot"
    )


if __name__ == "__main__":
    main()
