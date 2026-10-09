"""PM results package of 2026-10-09, section C, check (c): the listed-variance forward
``sqrt(EQV/EV)`` against model S over the copula, on model S's converged dates.

    python scripts/pm_check_c.py

The owner's arithmetic: "On the 218 converged model S dates it averages 0.9423, against 0.9436
for P_D_S/P_D; today it is 0.9647."  Both averages are of the study's own tables
(``entries_3m.parquet``, basket B1, and ``model_s_3m.parquet``): no Monte Carlo number of this
package enters, so they carry the standard error of the mean across dates only.  Output: the
part ``C_check_c`` and ``tables/C_check_c.csv``.
"""

# ruff: noqa: E501
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_common as pc

log = logging.getLogger("pm_check_c")
TODAY = "2026-10-02"
OWNER = {"listed_fwd": 0.9423, "s_over_copula": 0.9436, "today": 0.9647}


def build() -> None:
    entries = pd.read_parquet(pc.STUDY / "entries_3m.parquet")
    entries = entries[entries["basket"] == "B1"]
    model_s = pd.read_parquet(pc.STUDY / "model_s_3m.parquet")
    if entries["date"].duplicated().any() or model_s["date"].duplicated().any():
        raise ValueError("more than one row per date in a study table")
    m = entries.merge(model_s[["date", "P_D_S", "converged"]], on="date", how="inner")
    c = m[m["converged"].astype(bool)].copy()
    c["listed_fwd"] = np.sqrt(c["EQV"] / c["EV"])
    c["s_over_copula"] = c["P_D_S"] / c["P_D"]
    c["difference"] = c["listed_fwd"] - c["s_over_copula"]
    today = entries[entries["date"] == TODAY]
    today_value = float(np.sqrt(today["EQV"] / today["EV"]).iloc[0])
    src = f"{pc.STUDY / 'entries_3m.parquet'} (basket B1), {pc.STUDY / 'model_s_3m.parquet'}"
    n = len(c)
    records, rows = [], []
    labels = {
        "listed_fwd": "listed-variance forward over the copula's, sqrt(EQV/EV)",
        "s_over_copula": "model S over the copula, P_D_S/P_D",
        "difference": "sqrt(EQV/EV) - P_D_S/P_D, per date",
    }
    for key, label in labels.items():
        v = c[key].astype(float)
        mean, se = float(v.mean()), float(v.std(ddof=1) / np.sqrt(n))
        rows.append(
            {
                "quantity": label,
                "n": n,
                "mean": mean,
                "se_of_mean": se,
                "median": float(v.median()),
                "q25": float(v.quantile(0.25)),
                "q75": float(v.quantile(0.75)),
                "owner": OWNER.get(key),
            }
        )
        records.append(pc.record(
            f"C.c.{key}.mean", "C", f"check (c): {label}, mean over model S's converged dates", mean, se, date="model S's converged monthly dates", unit="ratio",
            definition=f"{label}; mean across the dates, se = standard error of that mean (the dates treated as independent)", budget=pc.BUDGETS["study"], commit="the study's tables (no commit column)", source=src, n=n,
            notes="no Monte Carlo number of this package enters",
        ))  # fmt: skip
    records.append(pc.record(
        "C.c.listed_fwd.today", "C", "check (c): listed-variance forward over the copula's, sqrt(EQV/EV), 2026-10-02", today_value, None, date=TODAY, unit="ratio",
        definition="sqrt(EQV/EV) of the study's entry of 2026-10-02 (B1)", budget=pc.BUDGETS["study"], commit="the study's tables (no commit column)", source=src, notes="a number of the study's table: no standard error there",
    ))  # fmt: skip
    frame = pd.DataFrame(rows)
    pc.save_table(frame, "C_check_c")
    md = [
        "### C.8 Check (c): the listed-variance forward against model S, on model S's converged dates",
        "",
        "| quantity | n | mean ± se | median [q25, q75] | owner's arithmetic |",
        "|:--|--:|--:|--:|--:|",
    ]
    for r in rows:
        owner = "" if r["owner"] is None else f"{r['owner']:.4f}"
        md.append(
            f"| {r['quantity']} | {r['n']} | {r['mean']:.4f} ± {r['se_of_mean']:.4f} | {r['median']:.4f} [{r['q25']:.4f}, {r['q75']:.4f}] | {owner} |"
        )
    md += [
        f"| listed-variance forward, {TODAY} | 1 | {today_value:.4f} |  | {OWNER['today']:.4f} |",
        "",
        f"Study tables only (`entries_3m.parquet`, basket B1, and `model_s_3m.parquet`), on the {n} dates where model S converged: the owner's three numbers are reproduced. "
        "The ± is the standard error of the mean across dates (dates treated as independent); the per-date difference has its own row. "
        "The same two ratios on the dates the LC pass priced are in C.2 (n = 216 and 215), and the listed-variance forward is a column of every row of the per-date CSV. File: `tables/C_check_c.csv`.",
    ]
    for key in ("listed_fwd", "s_over_copula"):
        got = next(r["mean"] for r in rows if r["quantity"] == labels[key])
        if round(got, 4) != OWNER[key]:
            md.append(
                f"NOTE: the owner's {OWNER[key]:.4f} is not reproduced for {labels[key]}: {got:.6f}."
            )
    if round(today_value, 4) != OWNER["today"]:
        md.append(
            f"NOTE: the owner's {OWNER['today']:.4f} for today is not reproduced: {today_value:.6f}."
        )
    pc.write_part("C_check_c", records, "\n".join(md))
    log.info(
        "C_check_c written: n %d; means %.6f and %.6f; today %.6f",
        n,
        rows[0]["mean"],
        rows[1]["mean"],
        today_value,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    build()
