"""Dispersion study, check C9 (spec §10): three dates recomputed bit-identically.

    python scripts/disp_c9.py [--tenor 3m]

Re-runs the copula pricing of three entry dates into a scratch folder and compares every
number with the stored entry file; re-runs the FHS of the same dates when ``fhs_<tenor>.parquet``
exists.  Writes ``outputs/dispersion/c9.json``.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_entries as de
import disp_fhs_run as fr

from volsto.studies import disp_data as dd

DATES = ("2009-03-09", "2016-06-06", "2024-08-05")


def same(a: Any, b: Any) -> bool:
    if isinstance(a, dict):
        return (
            isinstance(b, dict)
            and a.keys() == b.keys()
            and all(same(a[k], b[k]) for k in a if k != "seconds")
        )
    if isinstance(a, np.ndarray):
        return (
            isinstance(b, np.ndarray)
            and a.shape == b.shape
            and bool(np.array_equal(a, b, equal_nan=a.dtype.kind == "f"))
        )
    if isinstance(a, float):
        return isinstance(b, float) and (a == b or (np.isnan(a) and np.isnan(b)))
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b, strict=True))
    return bool(a == b)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--tenor", default="3m")
    args = ap.parse_args()
    n_days = dd.TENORS[args.tenor]
    scratch = dd.OUT / "scratch" / "c9"
    shutil.rmtree(scratch, ignore_errors=True)
    monthly = dd.monthly_subset(dd.entry_dates(n_days))
    ok_cop = []
    for d in DATES:
        de.entry_job((d, args.tenor, n_days, False, str(scratch), d in monthly, "parity"))
        with (scratch / f"{d}.pkl").open("rb") as fh:
            again = pickle.load(fh)
        with (dd.OUT / "entries" / args.tenor / f"{d}.pkl").open("rb") as fh:
            stored = pickle.load(fh)
        ok_cop.append(same(stored, again))
    ok_fhs: list[bool] = []
    fpath = dd.OUT / f"fhs_{args.tenor}.parquet"
    if fpath.exists():
        stored_f = pd.read_parquet(fpath)
        entries = pd.read_parquet(dd.OUT / f"entries_{args.tenor}.parquet")
        fr.init(args.tenor, 10, True)
        for d in DATES:
            e = entries[(entries["date"] == d) & (entries["basket"] == "B1")].to_dict("records")[0]
            again_f = fr.job(e)
            s = stored_f[(stored_f["date"] == d) & (stored_f["basket"] == "B1")].iloc[0]
            ok_fhs.append(
                all(
                    (s[k] == v) or (pd.isna(s[k]) and pd.isna(v))
                    for k, v in again_f.items()
                    if k in s.index
                )
            )
    res = {
        "dates": list(DATES),
        "copula identical": ok_cop,
        "fhs identical": ok_fhs,
        "pass": bool(all(ok_cop) and all(ok_fhs)),
        "result": f"copula: {sum(ok_cop)} of {len(ok_cop)} dates bit-identical"
        + (f"; FHS: {sum(ok_fhs)} of {len(ok_fhs)}" if ok_fhs else "; FHS not run yet"),
    }
    (dd.OUT / "c9.json").write_text(json.dumps(res, indent=1))
    print(res)


if __name__ == "__main__":
    main()
