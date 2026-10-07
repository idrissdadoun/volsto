"""Dispersion study: every expiry smile of every entry date, and what the expiry guards remove.

    python scripts/disp_scan_smiles.py [--workers 5] [--summary]

For each weekly entry date (the one-month set, which contains the others, and the last day)
and each ticker the study prices on it (the Dow's members, DJX, DIA and the names of B3): the
expiry smiles **without** the guards of ``disp_smile`` (``EXPIRY_GUARDS`` off), each with its
maturity, number of strikes, the log-moneyness of the nearest strike on each side of the
forward, its at-the-money vol (linear between strikes, flat outside) and the vendor's smoothed
vol at the strike nearest the forward.  Writes ``smile_scan.parquet``.  The summary applies
the two guards to the scan — (1) one-sided expiries dropped when the ticker has a two-sided
one that day, (2) at-the-money vol within a factor ``TERM_BAND`` of the median of the four
nearest expiries — and writes ``smile_scan_removed.csv`` (the expiries removed) and
``smile_scan_dates.txt`` (the entry dates on which a ticker's list of expiries changes: the
dates to price again).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_entries as de  # noqa: E402

from volsto.studies import disp_data as dd  # noqa: E402
from volsto.studies import disp_smile as ds  # noqa: E402
from volsto.studies import disp_universe as du  # noqa: E402


def job(date: str) -> list[dict[str, Any]]:
    ds.EXPIRY_GUARDS = False
    names = [*du.members_on(date), "DJX", "DIA", *du.B3_NAMES]
    names = list(dict.fromkeys(names))
    chains = dd.read_chains(date, names)
    panel = dd.prices()
    rows: list[dict[str, Any]] = []
    for t in names:
        ch = chains.get(t)
        if ch is None or ch.empty:
            continue
        px = float(panel.at[date, t]) if t in panel.columns and date in panel.index else np.nan
        sm = ds.expiry_smiles(ch, date, "parity", spot=px if np.isfinite(px) else None)
        for e in sm:
            below, above = ds.money_reach(e.k)
            j = int(np.argmin(np.abs(e.k)))
            rows.append(
                {
                    "date": date,
                    "ticker": t,
                    "expiry": e.expiry,
                    "T": e.T,
                    "n": int(e.k.size),
                    "k_below": below,
                    "k_above": above,
                    "atm_vol": float(np.interp(0.0, e.k, e.vol)),
                    "vendor_near": float(e.vendor_vol[j]),
                    "k_near": float(e.k[j]),
                }
            )
    return rows


def job_exact(date: str) -> list[dict[str, Any]]:
    """The tickers of ``date`` whose list of expiry smiles differs with and without the guards,
    built as the entry script builds them (parity forwards, the previous days tried when the
    day has no usable smile, the forward check in between)."""
    names = [*du.members_on(date), "DJX", "DIA", *du.B3_NAMES]
    names = list(dict.fromkeys(names))
    chains = dd.read_chains(date, names)
    rows: list[dict[str, Any]] = []
    for t in names:
        ds.EXPIRY_GUARDS = False
        off, day_off, _ = de.smiles_of(t, date, chains.get(t))
        ds.EXPIRY_GUARDS = True
        on, day_on, _ = de.smiles_of(t, date, chains.get(t))
        a = [f"{day_off}:{e.expiry}" for e in off]
        b = [f"{day_on}:{e.expiry}" for e in on]
        if a != b:
            rows.append(
                {
                    "date": date,
                    "ticker": t,
                    "only without the guards": " ".join(sorted(set(a) - set(b))),
                    "only with the guards": " ".join(sorted(set(b) - set(a))),
                }
            )
    return rows


def removed() -> pd.DataFrame:
    """The scan with the two guards applied in the library's order: the expiries removed, with
    the guard that removes each."""
    s = pd.read_parquet(dd.OUT / "smile_scan.parquet")
    s = s.sort_values(["date", "ticker", "T"]).reset_index(drop=True)
    s["two"] = np.isfinite(s["k_below"]) & np.isfinite(s["k_above"])
    any_two = s.groupby(["date", "ticker"])["two"].transform("any")
    s["guard"] = np.where(~s["two"] & any_two, "one-sided", "")
    kept = s[s["guard"] == ""].copy()
    kept["ratio"] = kept.groupby(["date", "ticker"])["atm_vol"].transform(
        lambda v: ds.term_ratio(v.to_numpy(float))
    )
    off = (
        (kept["T"] >= ds.TERM_MIN_T)
        & kept["ratio"].notna()
        & ((kept["ratio"] > ds.TERM_BAND) | (kept["ratio"] < 1.0 / ds.TERM_BAND))
    )
    s.loc[kept.index[off], "guard"] = "term structure"
    s["ratio"] = kept["ratio"]
    s["one_sided_only"] = ~s["two"] & ~any_two
    return s


def summary() -> None:
    s = removed()
    out = s[s["guard"] != ""]
    print(f"{len(s)} expiry smiles on {s['date'].nunique()} dates, {s['ticker'].nunique()} tickers")
    for g, x in out.groupby("guard"):
        print(
            f"  removed, {g}: {len(x)} smiles on {x['date'].nunique()} dates, {x['ticker'].nunique()} tickers: {dict(x['ticker'].value_counts().head(12))}"
        )
    only = s[s["one_sided_only"]]
    pairs = only.groupby(["date", "ticker"]).size().reset_index()[["date", "ticker"]]
    print(
        f"  one-sided and kept (the ticker has no two-sided expiry that day): {len(only)} smiles; name-dates {[tuple(r) for r in pairs.to_numpy()]}"
    )
    out.drop(columns=["two", "one_sided_only"]).to_csv(
        dd.OUT / "smile_scan_removed.csv", index=False
    )
    dates = sorted(out["date"].unique())
    (dd.OUT / "smile_scan_dates.txt").write_text("\n".join(dates) + "\n")
    print(f"  dates to price again: {len(dates)} (smile_scan_dates.txt)")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--summary", action="store_true")
    ap.add_argument(
        "--exact",
        action="store_true",
        help="compare each ticker's final list of expiries with and without the guards",
    )
    args = ap.parse_args()
    if args.exact:
        dates = sorted({*dd.entry_dates(dd.TENORS["1m"]), dd.LAST_DAY})
        with Pool(max(1, min(args.workers, dd.cpu_budget()))) as pool:
            res = pool.map(job_exact, dates, chunksize=4)
        out = pd.DataFrame([r for rows in res for r in rows])
        out.to_csv(dd.OUT / "smile_scan_exact.csv", index=False)
        changed = sorted(out["date"].unique()) if len(out) else []
        (dd.OUT / "smile_scan_exact_dates.txt").write_text("\n".join(changed) + "\n")
        print(
            f"exact scan: {len(dates)} dates; a ticker's expiries differ with the guards on {len(changed)} dates, {len(out)} ticker-dates: {dict(out['ticker'].value_counts().head(15)) if len(out) else {}}"
        )
        return
    if not args.summary:
        dates = dd.entry_dates(dd.TENORS["1m"])
        dates = sorted({*dates, dd.LAST_DAY})
        with Pool(max(1, min(args.workers, dd.cpu_budget()))) as pool:
            res = pool.map(job, dates, chunksize=4)
        pd.DataFrame([r for rows in res for r in rows]).to_parquet(
            dd.OUT / "smile_scan.parquet", index=False
        )
    summary()


if __name__ == "__main__":
    main()
