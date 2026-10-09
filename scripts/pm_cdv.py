"""PM results package of 2026-10-09, section D and figure F3: the stratified test of the
cross-dependent volatility prototype (owner's check f).

    python scripts/pm_cdv.py [--table <cdv_stratified_3m.parquet>] [--paragraph <text file>]

Input: the table of ``scripts/cdv_stratified.py`` (branch ``cross-dependent-vol``) run with
``--fixed 3,6``: for each of the 40 dates — the 20 with the largest and the 20 with the smallest
clipped mass of M12 inside ±2.5 sd in the variant development pass — the constant-correlation
companion (CC), M12 (LC = the prototype at ``β = 0``) and the prototype at ``β = 3`` and
``β = 6``, all four on the same pricing paths, at the development budget.

Outputs (``outputs/dispersion_lc/pm_update``): ``tables/D_stratified_by_date.csv`` and
``D_stratified_by_group.csv``, ``figures/F3_stratified_cdv_by_group.pdf`` with its CSV, and the
part ``D_stratified`` (records and Markdown).  Per date: CDV/CC, LC/CC (paired), model S over the
copula, CDV, LC and CC over the copula, the listed-variance forward ``√(EQV/EV)``; ``κ``,
``E[V]/EQV`` and ``E[R̄²]/M_B^listed`` under each model; the clipped mass.  By group: the mean
across the group's priced dates with the standard error of that mean, and the median.  A failed
date stays in the per-date table with its reason.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_common as pc

log = logging.getLogger("pm_cdv")
TABLE = pc.LC_OUT / "cdv" / "stratified" / "cdv_stratified_3m.parquet"
GROUPS = (
    ("high", "the 20 dates with the largest M12 clipped mass"),
    ("low", "the 20 dates with the smallest M12 clipped mass"),
)

#: (column of the by-date table, label, definition, has a Monte Carlo error column)
QUANTITIES: tuple[tuple[str, str, str], ...] = (
    (
        "lc_over_cc",
        "LC / CC",
        "E_LC[D] / E_CC[D], paired on the pricing paths (M12 = the prototype at beta = 0)",
    ),
    ("cdv3_over_cc", "CDV(beta=3) / CC", "E_CDV[D] at beta = 3 over E_CC[D], paired"),
    ("cdv6_over_cc", "CDV(beta=6) / CC", "E_CDV[D] at beta = 6 over E_CC[D], paired"),
    ("s_over_copula", "model S / copula", "P_D_S / P_D of the study's tables"),
    ("listed_fwd", "listed-variance forward", "sqrt(EQV / EV), EV the copula's E[V] of the entry"),
    ("lc_over_copula", "LC / copula", "E_LC[D] / P_D"),
    ("cdv3_over_copula", "CDV(beta=3) / copula", "E_CDV[D] at beta = 3 over P_D"),
    ("cdv6_over_copula", "CDV(beta=6) / copula", "E_CDV[D] at beta = 6 over P_D"),
    ("cc_over_copula", "CC / copula", "E_CC[D] / P_D"),
    ("kappa_cc", "kappa CC", "E[D]/sqrt(E[V]) under CC"),
    ("kappa_lc", "kappa LC", "E[D]/sqrt(E[V]) under LC"),
    ("kappa_cdv3", "kappa CDV(beta=3)", "E[D]/sqrt(E[V]) under the prototype at beta = 3"),
    ("kappa_cdv6", "kappa CDV(beta=6)", "E[D]/sqrt(E[V]) under the prototype at beta = 6"),
    ("kappa_copula", "kappa copula", "P_D / sqrt(EV) of the entry"),
    ("kappa_s", "kappa model S", "P_D_S / sqrt(EV_S) of the study's model S table"),
    ("ev_over_eqv_cc", "E[V]/EQV CC", "E_CC[V] / EQV"),
    ("ev_over_eqv_lc", "E[V]/EQV LC", "E_LC[V] / EQV"),
    ("ev_over_eqv_cdv3", "E[V]/EQV CDV(beta=3)", "E_CDV[V] / EQV at beta = 3"),
    ("ev_over_eqv_cdv6", "E[V]/EQV CDV(beta=6)", "E_CDV[V] / EQV at beta = 6"),
    (
        "rbar2_cc",
        "E[Rbar^2]/M_B^listed CC",
        "E_CC[Rbar^2] over the listed index second moment (sum_w M_i - EQV)",
    ),
    ("rbar2_lc", "E[Rbar^2]/M_B^listed LC", "E_LC[Rbar^2] over the listed index second moment"),
    ("rbar2_cdv3", "E[Rbar^2]/M_B^listed CDV(beta=3)", "the same under the prototype at beta = 3"),
    ("rbar2_cdv6", "E[Rbar^2]/M_B^listed CDV(beta=6)", "the same under the prototype at beta = 6"),
    (
        "clip_lc",
        "clipped mass inside ±2.5 sd, M12",
        "largest over the slices of the calibration; lambda at 0 or at its cap",
    ),
    ("clip_cdv3", "clipped mass inside ±2.5 sd, CDV(beta=3)", "the same at beta = 3"),
    ("clip_cdv6", "clipped mass inside ±2.5 sd, CDV(beta=6)", "the same at beta = 6"),
)
SOURCE = {
    "lc_over_cc": ("ratio_lc_cc", "ratio_lc_cc_se"), "cdv3_over_cc": ("ratio_cdv_a_cc", "ratio_cdv_a_cc_se"),
    "cdv6_over_cc": ("ratio_cdv_b_cc", "ratio_cdv_b_cc_se"), "s_over_copula": ("P_D_S_over_P_D", None),
    "listed_fwd": ("listed_variance_forward", None), "lc_over_copula": ("over_copula_lc", "over_copula_lc_se"),
    "cdv3_over_copula": ("over_copula_cdv_a", "over_copula_cdv_a_se"), "cdv6_over_copula": ("over_copula_cdv_b", "over_copula_cdv_b_se"),
    "cc_over_copula": ("over_copula_cc", "over_copula_cc_se"), "kappa_cc": ("kappa_cc", "kappa_cc_se"),
    "kappa_lc": ("kappa_lc", "kappa_lc_se"), "kappa_cdv3": ("kappa_cdv_a", "kappa_cdv_a_se"), "kappa_cdv6": ("kappa_cdv_b", "kappa_cdv_b_se"),
    "kappa_copula": ("kappa_cop", None), "ev_over_eqv_cc": ("EV_over_EQV_cc", "EV_over_EQV_cc_se"),
    "ev_over_eqv_lc": ("EV_over_EQV_lc", "EV_over_EQV_lc_se"), "ev_over_eqv_cdv3": ("EV_over_EQV_cdv_a", "EV_over_EQV_cdv_a_se"),
    "ev_over_eqv_cdv6": ("EV_over_EQV_cdv_b", "EV_over_EQV_cdv_b_se"), "rbar2_cc": ("E_Rbar2_over_listed_cc", "E_Rbar2_over_listed_cc_se"),
    "rbar2_lc": ("E_Rbar2_over_listed_lc", "E_Rbar2_over_listed_lc_se"), "rbar2_cdv3": ("E_Rbar2_over_listed_cdv_a", "E_Rbar2_over_listed_cdv_a_se"),
    "rbar2_cdv6": ("E_Rbar2_over_listed_cdv_b", "E_Rbar2_over_listed_cdv_b_se"), "clip_lc": ("clip_inner_lc", None),
    "clip_cdv3": ("clip_inner_cdv_a", None), "clip_cdv6": ("clip_inner_cdv_b", None),
}  # fmt: skip


def by_date(table: pd.DataFrame) -> pd.DataFrame:
    """The per-date table: the owner's quantities with their Monte Carlo errors, the flags."""
    model_s = pd.read_parquet(pc.STUDY / "model_s_3m.parquet").set_index("date")
    out = pd.DataFrame(
        {
            "date": table["date"],
            "group": table["group"],
            "status": table["status"],
            "reason": table["reason"],
        }
    )
    ok = table["status"] != "failed"
    if not bool(np.all(table.loc[ok, "beta_a"] == 3.0) and np.all(table.loc[ok, "beta_b"] == 6.0)):
        raise ValueError("the table is not a --fixed 3,6 run")
    out["flag_unscreened"] = table["n_names_unscreened"].fillna(0).astype(float) > 0
    out["model_s_converged"] = table["model_s_converged"]
    out["clip_inner_variant_pass"] = table.get("table_clip_inner_max", np.nan)
    for name, (col, se) in SOURCE.items():
        out[name] = table[col]
        if se is not None:
            out[f"{name}_se"] = table[se]
    ev_s = table["date"].map(model_s["EV_S"])
    out["kappa_s"] = table["P_D_S"] / np.sqrt(ev_s)
    out["ev_over_eqv_copula"] = table["EV_copula"] / table["EQV"]
    out["ev_over_eqv_s"] = ev_s / table["EQV"]
    out["idx_err_m25_lc"], out["idx_err_m25_cdv3"], out["idx_err_m25_cdv6"] = (
        table["idx_m25_lc"],
        table["idx_m25_cdv_a"],
        table["idx_m25_cdv_b"],
    )
    out["git_commit"], out["n_particles"], out["n_paths"] = (
        table["git_commit"],
        table["n_particles"],
        table["n_paths"],
    )
    return out.sort_values(["group", "date"]).reset_index(drop=True)


def by_group(dates: pd.DataFrame) -> pd.DataFrame:
    """Mean (with the standard error of the mean across dates), median and n by group, on the
    priced dates; also without the dates where a name is kept unscreened."""
    rows = []
    ok = dates[dates["status"] != "failed"]
    names = [q[0] for q in QUANTITIES] + ["ev_over_eqv_copula", "ev_over_eqv_s"]
    for group, _ in GROUPS:
        for sample, frame in (
            ("all priced", ok[ok["group"] == group]),
            ("without flagged dates", ok[(ok["group"] == group) & ~ok["flag_unscreened"]]),
        ):
            for name in names:
                v = frame[name].dropna().astype(float)
                rows.append({
                    "group": group, "sample": sample, "quantity": name, "n": len(v),
                    "mean": v.mean() if len(v) else np.nan,
                    "se_of_mean": v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                    "median": v.median() if len(v) else np.nan,
                    "median_mc_se": frame[f"{name}_se"].median() if f"{name}_se" in frame and len(frame) else np.nan,
                })  # fmt: skip
    return pd.DataFrame(rows)


def figure(groups: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    """F3: by group, the forward over the copula under M12, the prototype at beta 3 and 6, model
    S and the listed-variance forward (means across the group's dates, ± the standard error of
    the mean)."""
    series = (("lc_over_copula", "M12 (LC)"), ("cdv3_over_copula", "CDV, β = 3"), ("cdv6_over_copula", "CDV, β = 6"),
              ("s_over_copula", "model S"), ("listed_fwd", "listed-variance forward"))  # fmt: skip
    g = groups[groups["sample"] == "all priced"]
    data = []
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    width = 0.16
    for j, (name, label) in enumerate(series):
        means, errs = [], []
        for group, _ in GROUPS:
            r = g[(g["group"] == group) & (g["quantity"] == name)].iloc[0]
            means.append(r["mean"])
            errs.append(r["se_of_mean"])
            data.append(
                {
                    "group": group,
                    "series": label,
                    "quantity": name,
                    "mean": r["mean"],
                    "se_of_mean": r["se_of_mean"],
                    "n": r["n"],
                }
            )
        ax.bar(
            np.arange(len(GROUPS)) + (j - 2) * width,
            means,
            width,
            yerr=errs,
            capsize=2,
            label=label,
        )
    ax.set_xticks(np.arange(len(GROUPS)))
    ax.set_xticklabels(
        ["largest M12 clipped mass (20 dates)", "smallest M12 clipped mass (20 dates)"]
    )
    ax.set_ylabel("forward over the copula's")
    lo = min(d["mean"] for d in data) - 0.02
    ax.set_ylim(max(lo, 0.0), 1.01)
    ax.axhline(1.0, color="k", lw=0.5)
    ax.legend(fontsize=7, ncol=3, loc="lower center")
    fig.tight_layout()
    return fig, pd.DataFrame(data)


def cell(v: Any, se: Any = None, digits: int = 4) -> str:
    return pc.pm(
        None if v is None or pd.isna(v) else float(v),
        None if se is None or pd.isna(se) else float(se),
        digits,
    )


def build(table_path: Path, paragraph: Path | None) -> None:
    table = pd.read_parquet(table_path)
    dates = by_date(table)
    groups = by_group(dates)
    pc.save_table(dates, "D_stratified_by_date")
    pc.save_table(groups, "D_stratified_by_group")
    fig, data = figure(groups)
    pc.save_figure(fig, "F3_stratified_cdv_by_group", data)
    plt.close(fig)
    ok = dates[dates["status"] != "failed"]
    commit = ", ".join(sorted({str(c) for c in ok["git_commit"].dropna()}))
    budget = pc.BUDGETS["development"]
    src = str(table_path)
    records = []
    md: list[str] = []
    add = md.append
    n_by = {g: int((ok["group"] == g).sum()) for g, _ in GROUPS}
    failed = dates[dates["status"] == "failed"]
    add(
        f"Source: `{table_path.relative_to(pc.LC_OUT.parent.parent)}` — `scripts/cdv_stratified.py --fixed 3,6 --g-max 3` on branch `cross-dependent-vol` "
        f"(commit {commit}: M12's defaults of 2026-10-09 merged in), development budget ({budget}). {len(dates)} dates selected from the variant development pass "
        f"(`lcm_3m_dev_repair.parquet`) by M12's clipped mass inside ±2.5 sd; priced: {n_by['high']} of the high group, {n_by['low']} of the low group"
        + (
            f"; failed: {', '.join(f'{r.date} ({r.reason})' for r in failed.itertuples())}"
            if len(failed)
            else ""
        )
        + ". On each date CC, LC (M12, the prototype at β = 0) and the prototype at β = 3 and β = 6 are priced on the same paths; no search for the smallest β. "
        "Files: `tables/D_stratified_by_date.csv`, `tables/D_stratified_by_group.csv`, `figures/F3_stratified_cdv_by_group.pdf/.csv`."
    )
    add("")
    add(
        "### D.1 By group (mean across the group's dates ± the standard error of that mean; median in brackets)"
    )
    add("")
    add("| quantity | largest M12 clipped mass | smallest M12 clipped mass |")
    add("|:--|--:|--:|")
    g_all = groups[groups["sample"] == "all priced"]
    labels = {q[0]: q[1] for q in QUANTITIES} | {
        "ev_over_eqv_copula": "E[V]/EQV copula",
        "ev_over_eqv_s": "E[V]/EQV model S",
    }
    definitions = {q[0]: q[2] for q in QUANTITIES} | {
        "ev_over_eqv_copula": "EV / EQV of the entry",
        "ev_over_eqv_s": "EV_S / EQV",
    }
    for name in labels:
        cells = []
        for group, _ in GROUPS:
            r = g_all[(g_all["group"] == group) & (g_all["quantity"] == name)].iloc[0]
            cells.append(
                f"{cell(r['mean'], r['se_of_mean'])} ({cell(r['median'])}; n {int(r['n'])})"
            )
            records.append(pc.record(
                f"D.{group}.{name}.mean", "D", f"{labels[name]}: mean over {group} group", r["mean"], r["se_of_mean"], date=f"{group} group",
                unit="ratio" if "clip" not in name else "fraction of the cloud", definition=definitions[name] + "; mean across the group's priced dates, se = standard error of that mean",
                budget=budget, commit=commit, source=src, n=int(r["n"]), notes=f"median {r['median']:.6g}; median Monte Carlo se of a date {r['median_mc_se']:.3g}" if pd.notna(r["median_mc_se"]) else f"median {r['median']:.6g}",
            ))  # fmt: skip
        add(f"| {labels[name]} | {cells[0]} | {cells[1]} |")
    add("")
    add("Discounts (1 - ratio), from the rows above: " + "; ".join(
        f"{label} {', '.join(f'{100 * (1 - float(g_all[(g_all.group == g) & (g_all.quantity == name)].iloc[0]['mean'])):.2f} % ({g})' for g, _ in GROUPS)}"
        for name, label in (("lc_over_cc", "LC against CC"), ("cdv3_over_cc", "CDV β=3 against CC"), ("cdv6_over_cc", "CDV β=6 against CC"),
                            ("s_over_copula", "model S against the copula"), ("listed_fwd", "listed-variance forward against the copula"),
                            ("lc_over_copula", "LC against the copula"), ("cdv3_over_copula", "CDV β=3 against the copula"), ("cdv6_over_copula", "CDV β=6 against the copula"))
    ) + ".")  # fmt: skip
    flagged = ok[ok["flag_unscreened"]]
    if len(flagged):
        add("")
        add(
            f"Flagged dates (a name kept unscreened) among the 40: {', '.join(f'{r.date} ({r.group})' for r in flagged.itertuples())}; the by-group table without them is in `tables/D_stratified_by_group.csv` (sample \"without flagged dates\")."
        )
    not_conv = ok[ok["model_s_converged"] == False]  # noqa: E712
    if len(not_conv):
        add("")
        add(
            f"Model S did not converge on: {', '.join(not_conv['date'])}; its ratio is in the table as the study's file has it."
        )
    if paragraph is not None and paragraph.exists():
        add("")
        add("### D.2 Reading")
        add("")
        add(paragraph.read_text().strip())
    add("")
    add("### D.3 Per date (± the date's Monte Carlo standard error; ratios to CC are paired)")
    add("")
    add(
        "| date | group | M12 clipped mass | clipped mass β=3 / β=6 | LC/CC | CDV(3)/CC | CDV(6)/CC | S/copula | listed fwd | LC/copula | CDV(3)/copula | CDV(6)/copula | κ CC / LC / CDV(3) / CDV(6) | E[R̄²]/M_B LC / CDV(3) / CDV(6) |"
    )
    add("|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|:--|")
    for d in dates.itertuples():
        if d.status == "failed":
            add(f"| {d.date} | {d.group} | failed: {d.reason} | | | | | | | | | | | |")
            continue
        flag = " †" if d.flag_unscreened else ""
        add(
            f"| {d.date}{flag} | {d.group} | {d.clip_lc:.3f} | {d.clip_cdv3:.3f} / {d.clip_cdv6:.3f} | {cell(d.lc_over_cc, d.lc_over_cc_se)} | {cell(d.cdv3_over_cc, d.cdv3_over_cc_se)} | "
            f"{cell(d.cdv6_over_cc, d.cdv6_over_cc_se)} | {cell(d.s_over_copula)} | {cell(d.listed_fwd)} | {cell(d.lc_over_copula, d.lc_over_copula_se)} | "
            f"{cell(d.cdv3_over_copula, d.cdv3_over_copula_se)} | {cell(d.cdv6_over_copula, d.cdv6_over_copula_se)} | "
            f"{d.kappa_cc:.4f} / {d.kappa_lc:.4f} / {d.kappa_cdv3:.4f} / {d.kappa_cdv6:.4f} | {d.rbar2_lc:.3f} / {d.rbar2_cdv3:.3f} / {d.rbar2_cdv6:.3f} |"
        )
        for name in (
            "lc_over_cc",
            "cdv3_over_cc",
            "cdv6_over_cc",
            "lc_over_copula",
            "cdv3_over_copula",
            "cdv6_over_copula",
            "s_over_copula",
            "listed_fwd",
            "clip_lc",
            "clip_cdv3",
            "clip_cdv6",
        ):
            records.append(pc.record(
                f"D.{d.date}.{name}", "D", labels[name], getattr(d, name), getattr(d, f"{name}_se", None), date=str(d.date), unit="ratio" if "clip" not in name else "fraction of the cloud",
                definition=definitions[name], budget=budget if name not in ("s_over_copula", "listed_fwd") else pc.BUDGETS["study"], commit=commit, source=src, notes=f"group {d.group}" + ("; a name kept unscreened" if d.flag_unscreened else ""),
            ))  # fmt: skip
    add("")
    add(
        "† a name is kept unscreened on that date (decision 2). κ under the copula and under model S, `E[V]/EQV` under every model and the index errors at -2.5 sd are in the per-date CSV."
    )
    pc.write_part("D_stratified", records, "\n".join(md))
    figs = (
        "| figure | file | CSV | what it shows |\n|:--|:--|:--|:--|\n"
        "| F1 | `figures/F1_forward_over_copula.pdf` | `figures/F1_forward_over_copula.csv` | the forward over the copula's on the monthly dates 2007-2026: LC, model S, the listed-variance forward (3m, development budget, decisions 1-2 on) |\n"
        "| F2 | `figures/F2_calls_over_copula_by_strike.pdf` | `figures/F2_calls_over_copula_by_strike.csv` | calls over the copula's by strike (0.75, 1, 1.25, 1.5 x the copula's forward), LC and model S: mean and interquartile range across dates |\n"
        "| F3 | `figures/F3_stratified_cdv_by_group.pdf` | `figures/F3_stratified_cdv_by_group.csv` | the stratified CDV test by group: forward over the copula's under M12, CDV at β = 3 and β = 6, model S, and the listed-variance forward (means ± standard error of the mean) |\n\n"
        "Vector PDF, 6.5 x 3 inches, plain matplotlib, no titles."
    )
    pc.write_part("F_figures", [], figs)
    pc.status(
        f"D: part D_stratified written from {len(ok)} priced dates of {len(dates)} ({n_by['high']} high, {n_by['low']} low); F3 written."
    )
    log.info("D written: %d priced of %d dates; by group n %s", len(ok), len(dates), n_by)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--table", default=str(TABLE))
    ap.add_argument("--paragraph", default=str(pc.PM / "parts" / "D_reading.txt"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    build(Path(args.table), Path(args.paragraph))
    return 0


if __name__ == "__main__":
    sys.exit(main())
