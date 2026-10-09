"""PM results package of 2026-10-09, section D and figure F3: the stratified test of the
cross-dependent volatility prototype (owner's check f).

    python scripts/pm_cdv.py [--table <cdv_stratified_3m.parquet>] [--selection <history table>]
                             [--paragraph <text file>] [--no-status]

Input: the table of ``scripts/cdv_stratified.py`` (branch ``cross-dependent-vol``) run with
``--fixed 3,6``: for each of the 40 dates — the 20 with the largest and the 20 with the smallest
clipped mass of M12 inside ±2.5 sd in the variant development pass — the constant-correlation
companion (CC), M12 (LC = the prototype at ``β = 0``) and the prototype at ``β = 3`` and
``β = 6``, all four on the same pricing paths, at the development budget.

Outputs (``outputs/dispersion_lc/pm_update``): ``tables/D_stratified_by_date.csv`` and
``D_stratified_by_group.csv``, ``figures/F3_stratified_cdv_by_group.pdf`` with its CSV, and the
part ``D_stratified`` (records and Markdown; D.1 by group in both samples, D.1b the move from LC
to the prototype by channel, D.2 the reading, D.3 per date with the history table's status and
flags), and ``parts/F_figures.md``.  Per date: CDV/CC, LC/CC (paired), model S over the
copula, CDV, LC and CC over the copula, the listed-variance forward ``√(EQV/EV)``; ``κ``,
``E[V]/EQV`` and ``E[R̄²]/M_B^listed`` under each model; the clipped mass.  By group: the mean
across the group's priced dates with the standard error of that mean, and the median.  A failed
date stays in the per-date table with its reason.
"""

# ruff: noqa: E501, RUF001
from __future__ import annotations

import argparse
import logging
import math
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
#: the history table the 40 dates were selected from (its ``clip_inner_max``)
SELECTION = pc.LC_OUT / "lcm_3m_dev_repair.parquet"
GROUPS = (
    ("high", "the 20 dates with the largest M12 clipped mass in the selection table"),
    ("low", "the 20 dates with the smallest M12 clipped mass in the selection table"),
)
HEAD = {"high": "largest M12 clipped mass", "low": "smallest M12 clipped mass"}
#: the gates of ``scripts/lcm_price.py`` ``row_status`` (``check_names`` is a diagnostic, not a gate)
GATING_CHECKS = ("check_no_nan", "check_forward", "check_index")
CLIP_LINE = 0.01  # the wing binds above this clipped mass (decision 5)
INDEX_MOMENT_LINE = (
    0.15  # |basket part / M_B^listed| above this: the index target is not the study's
)
KAPPA_SE_LINE = 0.005  # a kappa with a Monte Carlo error at or above this is listed as not resolved
PROTOTYPE_CLIP_LINE = 0.05  # the low group is split by the prototype's clipped mass at beta = 6
ALL, UNFLAGGED = "all priced", "without flagged dates"
NO_LOW_CLIP = (
    "without the dates whose M12 clipped mass in this run is below the selection's 20th largest"
)
OUTSIDE_2020, IN_2020 = "outside 2020", "dates of 2020"
CLIP6_LE, CLIP6_GT = "CDV(beta=6) clips at most 5 %", "CDV(beta=6) clips more than 5 %"
SUFFIX = {
    ALL: "mean", UNFLAGGED: "mean_unflagged", NO_LOW_CLIP: "mean_without_low_clip", OUTSIDE_2020: "mean_outside_2020",
    IN_2020: "mean_2020", CLIP6_LE: "mean_cdv6_clip_le_5pct", CLIP6_GT: "mean_cdv6_clip_gt_5pct",
}  # fmt: skip
STUDY_ONLY = ("s_over_copula", "listed_fwd", "kappa_copula", "kappa_s", "ev_over_eqv_copula", "ev_over_eqv_s")  # fmt: skip
MODELS = (("cc", "CC"), ("lc", "LC"), ("cdv3", "CDV(beta=3)"), ("cdv6", "CDV(beta=6)"))

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
        "clipped mass inside ±2.5 sd, M12, this run (fraction of the particles)",
        "the larger of the two one-sided masses (lambda clipped at 0; lambda clipped at its cap), each at its worst calibration slice, inside +-2.5 sd, as a fraction of the particles",
    ),
    (
        "clip_cdv3",
        "clipped mass inside ±2.5 sd, CDV(beta=3) (fraction of the particles)",
        "the same at beta = 3",
    ),
    (
        "clip_cdv6",
        "clipped mass inside ±2.5 sd, CDV(beta=6) (fraction of the particles)",
        "the same at beta = 6",
    ),
)
#: rows added to D.1 after the owner's quantities: name -> (label, definition)
EXTRA = {
    "idx_err_m25_lc": ("index error at -2.5 sd, LC (vol points)", "the model's index implied volatility minus the index target (the model's own SVI fit of the DJX smile) at -2.5 sd, in vol points, under LC"),
    "idx_err_m25_cdv3": ("index error at -2.5 sd, CDV(beta=3) (vol points)", "the same under the prototype at beta = 3"),
    "idx_err_m25_cdv6": ("index error at -2.5 sd, CDV(beta=6) (vol points)", "the same under the prototype at beta = 6"),
    "kappa_cdv6_over_copula": ("kappa CDV(beta=6) / kappa copula", "per date, kappa under the prototype at beta = 6 over P_D / sqrt(EV) of the entry"),
    "clip_lc_selection": ("clipped mass inside ±2.5 sd, M12, selection table (fraction of the particles)", "clip_inner_max of the selection table, by which the 40 dates were chosen"),
    "lc_over_cc_selection": ("LC / CC in the selection table", "the paired ratio E_LC[D] / E_CC[D] of the selection table on the same dates (another estimate of LC / CC: its own run)"),
}  # fmt: skip
#: the move from LC to the prototype by channel: name -> (label, definition)
CHANNELS = {
    f"{kind}_cdv{b}": (f"{lab}, LC to beta = {b}", f"per date, {what} from LC to the prototype at beta = {b}; d log E[D] = d log kappa + 0.5 d log E[V]")
    for b in ("3", "6")
    for kind, lab, what in (
        ("dlog_ed", "d log E[D]", "the log of the paired ratio E_CDV[D] / E_LC[D]"),
        ("dlog_kappa", "d log kappa", "the log of kappa_CDV / kappa_LC"),
        ("half_dlog_ev", "0.5 d log E[V]", "half the log of E_CDV[V] / E_LC[V] (the log-change of sqrt(E[V]))"),
    )
}  # fmt: skip
#: (group, quantity) whose standard error clustered by calendar year is printed under D.1
CLUSTERED = (("high", "listed_fwd"), ("high", "cdv6_minus_listed_fwd"), ("low", "cdv6_minus_lc_over_cc"), ("high", "kappa_cdv6"))  # fmt: skip
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


#: per-date differences summarised by group: name -> (minuend, subtrahend), label
DIFFERENCES = {
    "cdv3_minus_lc_over_cc": ("cdv3_over_cc", "lc_over_cc"),
    "cdv6_minus_lc_over_cc": ("cdv6_over_cc", "lc_over_cc"),
    "lc_minus_s_over_copula": ("lc_over_copula", "s_over_copula"),
    "cdv3_minus_s_over_copula": ("cdv3_over_copula", "s_over_copula"),
    "cdv6_minus_s_over_copula": ("cdv6_over_copula", "s_over_copula"),
    "lc_minus_listed_fwd": ("lc_over_copula", "listed_fwd"),
    "cdv3_minus_listed_fwd": ("cdv3_over_copula", "listed_fwd"),
    "cdv6_minus_listed_fwd": ("cdv6_over_copula", "listed_fwd"),
}
DIFFERENCE_LABELS = {
    "cdv3_minus_lc_over_cc": "CDV(beta=3)/CC - LC/CC",
    "cdv6_minus_lc_over_cc": "CDV(beta=6)/CC - LC/CC",
    "lc_minus_s_over_copula": "LC/copula - model S/copula",
    "cdv3_minus_s_over_copula": "CDV(beta=3)/copula - model S/copula",
    "cdv6_minus_s_over_copula": "CDV(beta=6)/copula - model S/copula",
    "lc_minus_listed_fwd": "LC/copula - listed-variance forward",
    "cdv3_minus_listed_fwd": "CDV(beta=3)/copula - listed-variance forward",
    "cdv6_minus_listed_fwd": "CDV(beta=6)/copula - listed-variance forward",
}


def by_date(table: pd.DataFrame, selection: pd.DataFrame) -> pd.DataFrame:
    """The per-date table: the owner's quantities with their Monte Carlo errors, the flags, the
    history table's status under the current rule, and the move from LC to the prototype by
    channel."""
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
    # model S's numbers count on its converged dates only
    not_converged = ~table["model_s_converged"].fillna(False).astype(bool)
    out.loc[not_converged.to_numpy(), ["s_over_copula", "kappa_s", "ev_over_eqv_s"]] = np.nan
    for name, (a, b) in DIFFERENCES.items():
        out[name] = out[a] - out[b]
    for tag, name in (("lc", "lc"), ("cdv_a", "cdv3"), ("cdv_b", "cdv6")):
        out[f"clip_low_{name}"] = table[f"clip_low_inner_{tag}"]
        out[f"clip_high_{name}"] = table[f"clip_high_inner_{tag}"]
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
    for tag, name in (("lc", "lc"), ("cdv_a", "cdv3"), ("cdv_b", "cdv6")):
        out[f"idx_err_m25_{name}_se"] = table[f"idx_m25_{tag}_se"]
    # the selection (history) table: clipped mass, specification, status under the current rule
    sel = selection.set_index("date").loc[table["date"].to_numpy()]
    if bool((sel["status"] == "failed").any()):
        raise ValueError("a selected date is a failed row of the selection table")
    out["clip_lc_selection"] = sel["clip_inner_max"].to_numpy()
    if not np.allclose(out["clip_lc_selection"], out["clip_inner_variant_pass"], atol=1e-12):
        raise ValueError(
            "the run's table_clip_inner_max is not the selection table's clip_inner_max"
        )
    out["spec_differs"] = table["spec_key"].to_numpy() != sel["spec_key"].to_numpy()
    gates = [", ".join(k for k in GATING_CHECKS if not bool(r[k])) for _, r in sel.iterrows()]
    out["history_status"] = ["check" if g else "ok" for g in gates]
    out["history_failed_gates"] = gates
    out["flag_index_gate"] = ~sel["check_index"].to_numpy().astype(bool)
    out["index_moment_rel"] = (sel["EV_basket_part"] / sel["M_B_listed"]).to_numpy()
    out["flag_index_moment"] = out["index_moment_rel"].abs() > INDEX_MOMENT_LINE
    out["lc_over_cc_selection"] = sel["ratio"].to_numpy()
    out["lc_over_cc_selection_se"] = sel["ratio_se"].to_numpy()
    out["selection_commit"] = sel["git_commit"].to_numpy()
    # the move from LC to the prototype: d log E[D] = d log kappa + 0.5 d log E[V], per date
    for tag, b in (("cdv_a", "3"), ("cdv_b", "6")):
        out[f"dlog_ed_cdv{b}"] = np.log(table[f"ratio_{tag}_lc"])
        out[f"dlog_ed_cdv{b}_se"] = table[f"ratio_{tag}_lc_se"] / table[f"ratio_{tag}_lc"]
        out[f"dlog_kappa_cdv{b}"] = np.log(table[f"kappa_{tag}"] / table["kappa_lc"])
        out[f"half_dlog_ev_cdv{b}"] = 0.5 * np.log(table[f"EV_{tag}"] / table["EV_lc"])
        gap = (out[f"dlog_ed_cdv{b}"] - out[f"dlog_kappa_cdv{b}"] - out[f"half_dlog_ev_cdv{b}"])[ok]
        if not bool((gap.abs() < 1e-6).all()):
            raise ValueError(
                f"d log E[D] = d log kappa + 0.5 d log E[V] fails at beta = {b}: {gap.abs().max():.3g}"
            )
        out[f"frac_to_s_cdv{b}"] = (out["lc_over_copula"] - out[f"cdv{b}_over_copula"]) / (out["lc_over_copula"] - out["s_over_copula"])  # fmt: skip
    out["kappa_cdv6_over_copula"] = out["kappa_cdv6"] / out["kappa_copula"]
    out["kappa_cdv6_over_copula_se"] = out["kappa_cdv6_se"] / out["kappa_copula"]
    out["kappa_se_max"] = out[[f"kappa_{m}_se" for m, _ in MODELS]].max(axis=1)
    out["year"] = out["date"].astype(str).str[:4]
    return out.sort_values(["group", "date"]).reset_index(drop=True)


def selection_line(selection: pd.DataFrame, n: int = 20) -> float:
    """The n-th largest clipped mass among the priced rows of the selection table."""
    priced = selection[selection["status"] != "failed"]
    return float(priced["clip_inner_max"].sort_values(ascending=False).iloc[n - 1])


def cluster_se(values: pd.Series, years: pd.Series) -> float:
    """Standard error of the mean with the dates clustered by calendar year: the cluster-robust
    estimator sqrt(G/(G-1) * sum_g (sum_{i in g} (x_i - mean))^2) / n."""
    v = values.astype(float)
    sums = (v - v.mean()).groupby(years).sum()
    if len(sums) < 2:
        return float("nan")
    return float(np.sqrt(len(sums) / (len(sums) - 1) * float((sums**2).sum())) / len(v))


def by_group(dates: pd.DataFrame, line: float) -> pd.DataFrame:
    """Mean (with the standard error of the mean across dates, dates taken as independent; and
    the same clustered by calendar year), median and n by group, on the priced dates.  Samples:
    all priced; without the dates where a name is kept unscreened; the high group without the
    dates whose M12 clipped mass in this run is below ``line`` (the selection table's 20th
    largest), outside 2020 and in 2020; the low group by whether the prototype at beta = 6 clips
    more than 5 % of the particles."""
    rows = []
    ok = dates[dates["status"] != "failed"]
    names = [q[0] for q in QUANTITIES] + ["ev_over_eqv_copula", "ev_over_eqv_s", *DIFFERENCES, *EXTRA, *CHANNELS, "frac_to_s_cdv3", "frac_to_s_cdv6"]  # fmt: skip
    for group, _ in GROUPS:
        f = ok[ok["group"] == group]
        samples = [(ALL, f), (UNFLAGGED, f[~f["flag_unscreened"]])]
        if group == "high":
            samples += [(NO_LOW_CLIP, f[f["clip_lc"] >= line]), (OUTSIDE_2020, f[f["year"] != "2020"]), (IN_2020, f[f["year"] == "2020"])]  # fmt: skip
        else:
            samples += [(CLIP6_LE, f[f["clip_cdv6"] <= PROTOTYPE_CLIP_LINE]), (CLIP6_GT, f[f["clip_cdv6"] > PROTOTYPE_CLIP_LINE])]  # fmt: skip
        for sample, frame in samples:
            for name in names:
                if name.startswith("frac_to_s") and group != "high":
                    continue  # the distance LC - model S is about zero in the low group
                v = frame[name].dropna().astype(float)
                mc = (
                    frame[f"{name}_se"].astype(float)
                    if f"{name}_se" in frame and len(frame)
                    else None
                )
                rows.append({
                    "group": group, "sample": sample, "quantity": name, "n": len(v),
                    "mean": v.mean() if len(v) else np.nan,
                    "se_of_mean": v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan,
                    "median": v.median() if len(v) else np.nan,
                    "median_mc_se": mc.median() if mc is not None else np.nan,
                    "max_mc_se": mc.max() if mc is not None else np.nan,
                    "max_mc_se_date": frame.loc[mc.idxmax(), "date"] if mc is not None and mc.notna().any() else "",
                    "se_clustered_by_year": cluster_se(v, frame.loc[v.index, "year"]) if len(v) > 1 else np.nan,
                    "n_years": frame.loc[v.index, "year"].nunique(),
                })  # fmt: skip
    return pd.DataFrame(rows)


def figure(groups: pd.DataFrame) -> tuple[Any, pd.DataFrame]:
    """F3: by group, the forward over the copula under M12, the prototype at beta 3 and 6, model
    S and the listed-variance forward (means across the group's dates, ± the standard error of
    the mean)."""
    series = (("lc_over_copula", "M12 (LC)"), ("cdv3_over_copula", "CDV, β = 3"), ("cdv6_over_copula", "CDV, β = 6"),
              ("s_over_copula", "model S"), ("listed_fwd", "listed-variance forward"))  # fmt: skip
    g = groups[groups["sample"] == ALL]
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
        [
            "largest M12 clipped mass\nin the selection table (20 dates)",
            "smallest M12 clipped mass\nin the selection table (20 dates)",
        ],
        fontsize=8,
    )
    ax.set_ylabel("forward over the copula's")
    lo = min(d["mean"] for d in data) - 0.02
    ax.set_ylim(max(lo, 0.0), 1.01)
    ax.axhline(1.0, color="k", lw=0.5)
    ax.legend(
        fontsize=7,
        ncol=5,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.0),
        frameon=False,
        columnspacing=1.0,
        handlelength=1.2,
    )
    fig.tight_layout()
    return fig, pd.DataFrame(data)


def reading(
    groups: pd.DataFrame, dates: pd.DataFrame, line: float
) -> tuple[str, dict[tuple[str, str], tuple[float, float]], dict[str, float]]:
    """The paragraph of check (f), from the by-group and by-date tables (every number is a record
    of section D).  The sentences' claims are checked on the numbers before they are written.
    Returns the text, the high-minus-low differences (beta, sample) -> (value, se), and the
    counts and fractions the text quotes."""

    def row(group: str, name: str, sample: str = ALL) -> pd.Series:
        return groups[(groups["group"] == group) & (groups["quantity"] == name) & (groups["sample"] == sample)].iloc[0]  # fmt: skip

    def m(group: str, name: str, sample: str = ALL) -> float:
        return float(row(group, name, sample)["mean"])

    def se(group: str, name: str, sample: str = ALL) -> float:
        return float(row(group, name, sample)["se_of_mean"])

    def med(group: str, name: str, sample: str = ALL) -> float:
        return float(row(group, name, sample)["median"])

    def n(group: str, name: str = "lc_over_cc", sample: str = ALL) -> int:
        return int(row(group, name, sample)["n"])

    def pm(group: str, name: str, digits: int = 4, sign: bool = False, sample: str = ALL, scale: float = 1.0) -> str:  # fmt: skip
        r = row(group, name, sample)
        return f"{scale * r['mean']:{'+' if sign else ''}.{digits}f} ± {scale * r['se_of_mean']:.{digits}f}"

    def far(group: str, name: str, ref: float = 0.0, sample: str = ALL) -> float:
        """Distance of the group mean from ``ref`` in standard errors of the mean."""
        return (m(group, name, sample) - ref) / se(group, name, sample)

    def pc_(group: str, name: str, sample: str = ALL) -> str:
        return pm(group, name, 2, True, sample, 100.0)

    h, lo = "high", "low"
    ok = dates[dates["status"] != "failed"]
    hi_d, lo_d = ok[ok["group"] == h], ok[ok["group"] == lo]
    c0, c3, c6, s_h = m(h, "lc_over_copula"), m(h, "cdv3_over_copula"), m(h, "cdv6_over_copula"), m(h, "s_over_copula")  # fmt: skip
    way3, way6 = (c0 - c3) / (c0 - s_h), (c0 - c6) / (c0 - s_h)
    d0, d3, d6, ds = (100 * (1 - x) for x in (c0, c3, c6, s_h))
    k3, k6 = int((hi_d["clip_cdv3"] <= CLIP_LINE).sum()), int(
        (hi_d["clip_cdv6"] <= CLIP_LINE).sum()
    )
    min3, min6 = float(hi_d["clip_cdv3"].min()), float(hi_d["clip_cdv6"].min())
    low_max = float(lo_d["clip_lc_selection"].max())
    low_under = int((lo_d["clip_lc_selection"] <= CLIP_LINE).sum())
    flagged = ok[ok["flag_unscreened"]]
    gate, moment = lo_d[lo_d["flag_index_gate"]], lo_d[lo_d["flag_index_moment"]]
    moved = hi_d[hi_d["clip_lc"] < line]
    unres = ok[ok["kappa_se_max"] >= KAPPA_SE_LINE]
    fwd_se_max = float(ok[[f"{q}_se" for q in ("lc_over_cc", "cdv3_over_cc", "cdv6_over_cc", "cc_over_copula", "lc_over_copula", "cdv3_over_copula", "cdv6_over_copula")]].max().max())  # fmt: skip
    # what is specific to the binding dates: the difference of the two groups' per-date moves
    did: dict[tuple[str, str], tuple[float, float]] = {}
    for b in ("3", "6"):
        name = f"cdv{b}_minus_lc_over_cc"
        for key, sample in (("all", ALL), ("unflagged", UNFLAGGED)):
            did[b, key] = (m(h, name, sample) - m(lo, name, sample), math.hypot(se(h, name, sample), se(lo, name, sample)))  # fmt: skip
    t = {k: abs(v) / e for k, (v, e) in did.items()}
    claims = {
        "high group: the forward falls with beta (over the copula and over CC)": c6 < c3 < c0 and m(h, "cdv6_over_cc") < m(h, "cdv3_over_cc") < m(h, "lc_over_cc"),
        "high group: model S is below CDV at beta = 6 by more than two standard errors": far(h, "cdv6_minus_s_over_copula") > 2,
        "high group: about a quarter and about a half of the distance, on the means and on the per-date medians": all(0.20 < x < 0.30 for x in (way3, med(h, "frac_to_s_cdv3"))) and all(0.45 < x < 0.60 for x in (way6, med(h, "frac_to_s_cdv6"))),
        "high group: the clipped mass goes under 1 % on no date at beta = 3 or 6": k3 == 0 and k6 == 0 and min(min3, min6) > CLIP_LINE,
        "high group: the basket's second moment is within two standard errors of the listed one at beta = 3 and 6, and is not under LC": abs(far(h, "rbar2_cdv3", 1.0)) < 2 and abs(far(h, "rbar2_cdv6", 1.0)) < 2 and abs(far(h, "rbar2_lc", 1.0)) > 2,
        "high group: the index error at -2.5 sd falls in size with beta": abs(m(h, "idx_err_m25_cdv6")) < abs(m(h, "idx_err_m25_cdv3")) < abs(m(h, "idx_err_m25_lc")),
        "high group: CDV at beta = 6 within two standard errors of the listed-variance forward on the group mean": abs(far(h, "cdv6_minus_listed_fwd")) < 2,
        "high group: above the listed-variance forward outside 2020 and below it in 2020, each by more than two standard errors": far(h, "cdv6_minus_listed_fwd", 0.0, OUTSIDE_2020) > 2 and far(h, "cdv6_minus_listed_fwd", 0.0, IN_2020) < -2,
        "high group: E[V] above EQV at beta = 6 and kappa below the copula's, each by more than two standard errors": far(h, "ev_over_eqv_cdv6", 1.0) > 2 and far(h, "kappa_cdv6_over_copula", 1.0) < -2,
        "both groups: the prototype lowers the forward over CC at both betas": all(m(g, f"cdv{b}_minus_lc_over_cc") < 0 for g in (h, lo) for b in ("3", "6")),
        "beta = 3: the move is larger on the binding dates in both samples": did["3", "all"][0] < 0 and did["3", "unflagged"][0] < 0,
        "beta = 6: the two groups' moves are within one standard error of each other in both samples": t["6", "all"] < 1 and t["6", "unflagged"] < 1,
        "the flagged dates are all in the low group": bool((flagged["group"] == lo).all()) and len(flagged) > 0,
        "the channels add up on the group means": all(abs(m(g, f"dlog_ed_cdv{b}") - m(g, f"dlog_kappa_cdv{b}") - m(g, f"half_dlog_ev_cdv{b}")) < 1e-9 for g in (h, lo) for b in ("3", "6")),
        "high group: the fall comes through E[V] at both betas (beyond two standard errors, and larger than kappa's part)": all(far(h, f"half_dlog_ev_cdv{b}") < -2 and abs(m(h, f"half_dlog_ev_cdv{b}")) > abs(m(h, f"dlog_kappa_cdv{b}")) for b in ("3", "6")),
        "high group: kappa up at beta = 3 and within two standard errors of unchanged at beta = 6": m(h, "dlog_kappa_cdv3") > 0 and abs(far(h, "dlog_kappa_cdv6")) < 2,
        "low group: the fall comes through kappa (larger than E[V]'s part at both betas; beyond two standard errors at beta = 6)": all(m(lo, f"dlog_kappa_cdv{b}") < 0 and abs(m(lo, f"dlog_kappa_cdv{b}")) > abs(m(lo, f"half_dlog_ev_cdv{b}")) for b in ("3", "6")) and far(lo, "dlog_kappa_cdv6") < -2,
        "low group: E[V] unchanged at beta = 3 and rising at beta = 6": abs(far(lo, "half_dlog_ev_cdv3")) < 2 and far(lo, "half_dlog_ev_cdv6") > 2,
        "low group: the prototype's clipped mass exceeds M12's": m(lo, "clip_cdv3") > m(lo, "clip_lc") and m(lo, "clip_cdv6") > m(lo, "clip_lc"),
        "low group: the move at beta = 6 is larger where the prototype clips more than 5 %": m(lo, "cdv6_minus_lc_over_cc", CLIP6_GT) < m(lo, "cdv6_minus_lc_over_cc", CLIP6_LE),
        "high group without the dates below the selection line: the forward still falls with beta": m(h, "cdv6_over_copula", NO_LOW_CLIP) < m(h, "cdv3_over_copula", NO_LOW_CLIP) < m(h, "lc_over_copula", NO_LOW_CLIP),
        "low group: not all dates under the 1 % line": 0 < low_under < n(lo),
        "low group: LC within two standard errors of model S on the mean, and within 0.002 on the median": abs(far(lo, "lc_minus_s_over_copula")) < 2 and abs(med(lo, "lc_minus_s_over_copula")) < 0.002,
        "low group: CDV at beta = 6 below model S by more than two standard errors": far(lo, "cdv6_minus_s_over_copula") < -2,
        "the forward ratios' Monte Carlo errors are below the kappa line": fwd_se_max < KAPPA_SE_LINE,
    }  # fmt: skip
    failed = [k for k, good in claims.items() if not good]
    if failed:
        raise ValueError(f"the reading's sentences do not hold on the tables: {failed}")
    cdv6_listed_cl = float(row(h, "cdv6_minus_listed_fwd")["se_clustered_by_year"])
    moved_txt = ", ".join(
        f"{r.date} {r.clip_lc_selection:.3f} → {r.clip_lc:.3f}" for r in moved.itertuples()
    )
    flagged_txt = ", ".join(f"{r.date} ({r.group})" for r in flagged.itertuples())
    high_check = hi_d[hi_d["history_status"] == "check"]
    high_check_txt = (
        f"; {len(high_check)} high date{'s have' if len(high_check) > 1 else ' has'} status check in the selection table ({', '.join(f'{r.date}: {r.history_failed_gates}' for r in high_check.itertuples())})"
        if len(high_check)
        else ""
    )
    text = (
        f"**Level.** In level the discount moves towards model S's at β = 3 and at β = 6, without the clipped mass going under the 1 % line on any date. On the {n(h)} dates with the largest M12 clipped mass in the selection table ({pm(h, 'clip_lc', 3)} of the particles in this run), β imposed at 3 and at 6 takes the forward over the copula's from {pm(h, 'lc_over_copula')} under LC to "
        f"{pm(h, 'cdv3_over_copula')} and {pm(h, 'cdv6_over_copula')}, against {pm(h, 's_over_copula')} for model S and {pm(h, 'listed_fwd')} for the listed-variance forward (over CC: {pm(h, 'lc_over_cc')}, {pm(h, 'cdv3_over_cc')}, {pm(h, 'cdv6_over_cc')}). "
        f"M12's discount to the copula goes from {d0:.1f} % to {d3:.1f} % at β = 3 and {d6:.1f} % at β = 6, against model S's {ds:.1f} %: about a quarter and about a half of the distance ({100 * way3:.0f} % and {100 * way6:.0f} % on the group means; "
        f"{100 * med(h, 'frac_to_s_cdv3'):.0f} % and {100 * med(h, 'frac_to_s_cdv6'):.0f} % on the per-date medians; no standard error computed for these fractions); at β = 6 the forward over the copula's is still {pm(h, 'cdv6_minus_s_over_copula', 4, True)} above model S's. "
        f"The clipped mass is under 1 % on {k3} of the {n(h)} dates at β = 3 and on {k6} at β = 6 (smallest values {min3:.3f} and {min6:.3f}; group means {pm(h, 'clip_lc', 3)} → {pm(h, 'clip_cdv3', 3)} → {pm(h, 'clip_cdv6', 3)}), "
        f"while the basket's second moment is matched (E[R̄²]/M_B^listed {pm(h, 'rbar2_lc', 3)} under LC, {pm(h, 'rbar2_cdv3', 3)} at β = 3, {pm(h, 'rbar2_cdv6', 3)} at β = 6) and the index error at −2.5 sd falls from {pm(h, 'idx_err_m25_lc', 2, True)} to "
        f"{pm(h, 'idx_err_m25_cdv3', 2, True)} and {pm(h, 'idx_err_m25_cdv6', 2, True)} vol points. The group mean of CDV(6)/copula minus the listed-variance forward is {pm(h, 'cdv6_minus_listed_fwd', 4, True)} (± {cdv6_listed_cl:.4f} with the dates clustered by calendar year): "
        f"{pm(h, 'cdv6_minus_listed_fwd', 4, True, OUTSIDE_2020)} on the {n(h, 'cdv6_minus_listed_fwd', OUTSIDE_2020)} dates outside 2020 and {pm(h, 'cdv6_minus_listed_fwd', 4, True, IN_2020)} on the {n(h, 'cdv6_minus_listed_fwd', IN_2020)} dates of 2020; "
        f"E[V]/EQV is still {pm(h, 'ev_over_eqv_cdv6', 3)} at β = 6 ({pm(h, 'ev_over_eqv_lc', 3)} under LC) and κ is {pm(h, 'kappa_cdv6_over_copula', 3)} of the copula's: the level of the listed-variance forward is reached on the group mean without E[V] reaching EQV. "
        f"**Not specific to the binding dates?** Per date, CDV/CC − LC/CC is {pm(h, 'cdv3_minus_lc_over_cc', 4, True)} at β = 3 and {pm(h, 'cdv6_minus_lc_over_cc', 4, True)} at β = 6 on these dates, against {pm(lo, 'cdv3_minus_lc_over_cc', 4, True)} and "
        f"{pm(lo, 'cdv6_minus_lc_over_cc', 4, True)} on the {n(lo)} dates with the smallest clipped mass in the selection table (at most {low_max:.4f}; {low_under} of them under the 1 % line). High group minus low group, at β = 3: "
        f"{did['3', 'all'][0]:+.4f} ± {did['3', 'all'][1]:.4f} on all dates ({t['3', 'all']:.1f} standard errors) and {did['3', 'unflagged'][0]:+.4f} ± {did['3', 'unflagged'][1]:.4f} without the {len(flagged)} flagged dates "
        f"(a name kept unscreened: {flagged_txt}; {t['3', 'unflagged']:.1f} standard errors); at β = 6: {did['6', 'all'][0]:+.4f} ± {did['6', 'all'][1]:.4f} ({t['6', 'all']:.1f} standard errors) and "
        f"{did['6', 'unflagged'][0]:+.4f} ± {did['6', 'unflagged'][1]:.4f} ({t['6', 'unflagged']:.1f} standard errors). "
        f"**Channels.** Per date d log E[D] = d log κ + ½ d log E[V] (table D.1b, in %). On the binding dates, from LC to β = 3 the forward moves by {pc_(h, 'dlog_ed_cdv3')} % = κ {pc_(h, 'dlog_kappa_cdv3')} % and √E[V] {pc_(h, 'half_dlog_ev_cdv3')} %, "
        f"from LC to β = 6 by {pc_(h, 'dlog_ed_cdv6')} % = κ {pc_(h, 'dlog_kappa_cdv6')} % and √E[V] {pc_(h, 'half_dlog_ev_cdv6')} %: the fall comes through E[V], the second-moment channel (E[V]/EQV {m(h, 'ev_over_eqv_lc'):.3f} → {m(h, 'ev_over_eqv_cdv3'):.3f} → {m(h, 'ev_over_eqv_cdv6'):.3f}), "
        f"with κ up at β = 3 and {abs(far(h, 'dlog_kappa_cdv6')):.1f} standard errors from unchanged at β = 6. On the dates with the smallest clipped mass, β = 3: {pc_(lo, 'dlog_ed_cdv3')} % = κ {pc_(lo, 'dlog_kappa_cdv3')} % and √E[V] {pc_(lo, 'half_dlog_ev_cdv3')} %; "
        f"β = 6: {pc_(lo, 'dlog_ed_cdv6')} % = κ {pc_(lo, 'dlog_kappa_cdv6')} % and √E[V] {pc_(lo, 'half_dlog_ev_cdv6')} %: the fall comes through κ, with E[V] rising at β = 6 (E[V]/EQV {m(lo, 'ev_over_eqv_lc'):.3f} → {m(lo, 'ev_over_eqv_cdv3'):.3f} → {m(lo, 'ev_over_eqv_cdv6'):.3f}), "
        f"and there the prototype itself clips λ ({pm(lo, 'clip_cdv3', 3)} and {pm(lo, 'clip_cdv6', 3)} of the particles, against {pm(lo, 'clip_lc', 3)} under M12): CDV(6)/CC − LC/CC is {pm(lo, 'cdv6_minus_lc_over_cc', 4, True, CLIP6_LE)} on the "
        f"{n(lo, 'cdv6_minus_lc_over_cc', CLIP6_LE)} dates where the prototype clips at most 5 % of the particles and {pm(lo, 'cdv6_minus_lc_over_cc', 4, True, CLIP6_GT)} on the {n(lo, 'cdv6_minus_lc_over_cc', CLIP6_GT)} where it clips more. "
        f"**Caveats on the groups.** (i) The groups are those of the selection table; in this run M12's clipped mass is below that table's 20th largest ({line:.3f}) on {len(moved)} of the high dates ({moved_txt}); without them (n = {n(h, 'lc_over_copula', NO_LOW_CLIP)}) "
        f"LC/copula, CDV(3)/copula and CDV(6)/copula are {pm(h, 'lc_over_copula', 4, False, NO_LOW_CLIP)}, {pm(h, 'cdv3_over_copula', 4, False, NO_LOW_CLIP)} and {pm(h, 'cdv6_over_copula', 4, False, NO_LOW_CLIP)}, and CDV/CC − LC/CC "
        f"{pm(h, 'cdv3_minus_lc_over_cc', 4, True, NO_LOW_CLIP)} and {pm(h, 'cdv6_minus_lc_over_cc', 4, True, NO_LOW_CLIP)}{high_check_txt}. (ii) The low group is not a group where nothing binds: {n(lo) - low_under} of its dates are above the 1 % line, "
        f"{len(gate)} fail the index gate in the selection table ({', '.join(gate['date'])}), and on {len(moment)} ({', '.join(moment['date'])}) the model's index second moment is more than {100 * INDEX_MOMENT_LINE:.0f} % from the listed one "
        f"(the index target is not the study's); the low group's ratios to the copula include these {len(moment)} dates: E[R̄²]/M_B^listed under LC has mean {pm(lo, 'rbar2_lc', 4)} and median {med(lo, 'rbar2_lc'):.4f}, LC/copula − S/copula has mean "
        f"{pm(lo, 'lc_minus_s_over_copula', 4, True)} and median {med(lo, 'lc_minus_s_over_copula'):+.4f} (binding dates: {pm(h, 'lc_minus_s_over_copula', 4, True)}, median {med(h, 'lc_minus_s_over_copula'):+.4f}), and CDV(6)/copula − S/copula is "
        f"{pm(lo, 'cdv6_minus_s_over_copula', 4, True)}. (iii) κ and E[V]/EQV are not resolved on {len(unres)} dates ({int((unres['group'] == h).sum())} high, {int((unres['group'] == lo).sum())} low: a κ with a Monte Carlo error of {KAPPA_SE_LINE} or more, "
        f"listed under D.3); the forward ratios are not affected (largest Monte Carlo error of a ratio to CC or to the copula {fwd_se_max:.4f}). (iv) The ± are standard errors of means across dates taken as independent; the clustering by calendar year is under D.1. "
        f"**Conclusion.** In level, on the binding dates, β = 3 and β = 6 take M12's discount to the copula from {d0:.1f} % to {d3:.1f} % and {d6:.1f} %, against model S's {ds:.1f} %. The move at β = 3 is through E[V] on the binding dates and is larger there "
        f"than on the other dates by {-did['3', 'all'][0]:.4f} ± {did['3', 'all'][1]:.4f} ({-did['3', 'unflagged'][0]:.4f} ± {did['3', 'unflagged'][1]:.4f} without the {len(flagged)} flagged dates). At β = 6 the total move is the same in both groups to within one "
        f"standard error ({did['6', 'all'][0]:+.4f} ± {did['6', 'all'][1]:.4f}; {did['6', 'unflagged'][0]:+.4f} ± {did['6', 'unflagged'][1]:.4f} without the flagged dates) but goes through different channels: E[V] on the binding dates, κ on the others, "
        f"where the prototype itself clips λ. β was imposed and the clipped mass does not go under 1 % on any binding date, so the test does not measure what a β calibrated date by date would give."
    )
    facts = {
        "way3": way3, "way6": way6, "k3": k3, "k6": k6, "min3": min3, "min6": min6, "low_max": low_max, "low_under": low_under,
        "n_unresolved": len(unres), "fwd_se_max": fwd_se_max, "n_moved": len(moved),
    }  # fmt: skip
    return text, did, facts


def cell(v: Any, se: Any = None, digits: int = 4) -> str:
    return pc.pm(
        None if v is None or pd.isna(v) else float(v),
        None if se is None or pd.isna(se) else float(se),
        digits,
    )


def build(table_path: Path, selection_path: Path, paragraph: Path | None, status: bool = True) -> None:  # fmt: skip
    table = pd.read_parquet(table_path)
    selection = pd.read_parquet(selection_path)
    line = selection_line(selection)
    dates = by_date(table, selection)
    groups = by_group(dates, line)
    pc.save_table(dates, "D_stratified_by_date")
    pc.save_table(groups, "D_stratified_by_group")
    fig, data = figure(groups)
    pc.save_figure(fig, "F3_stratified_cdv_by_group", data)
    plt.close(fig)
    ok = dates[dates["status"] != "failed"]
    commit = ", ".join(sorted({str(c) for c in ok["git_commit"].dropna()}))
    sel_commit = ", ".join(sorted({str(c) for c in ok["selection_commit"].dropna()}))
    budget = pc.BUDGETS["development"]
    src, sel_src = str(table_path), str(selection_path)
    records: list[dict[str, Any]] = []
    md: list[str] = []
    add = md.append
    n_by = {g: int((ok["group"] == g).sum()) for g, _ in GROUPS}
    failed = dates[dates["status"] == "failed"]
    hi_d, lo_d = ok[ok["group"] == "high"], ok[ok["group"] == "low"]
    flagged = ok[ok["flag_unscreened"]]

    def row(group: str, name: str, sample: str = ALL) -> pd.Series:
        return groups[(groups["group"] == group) & (groups["quantity"] == name) & (groups["sample"] == sample)].iloc[0]  # fmt: skip

    def rec(id: str, quantity: str, value: Any, se: Any = None, **kw: Any) -> None:
        kw.setdefault("budget", budget)
        kw.setdefault("commit", commit)
        kw.setdefault("source", src)
        records.append(pc.record(id, "D", quantity, value, se, **kw))

    labels = {q[0]: q[1] for q in QUANTITIES} | {
        "ev_over_eqv_copula": "E[V]/EQV copula",
        "ev_over_eqv_s": "E[V]/EQV model S",
        **DIFFERENCE_LABELS,
        **{k: v[0] for k, v in EXTRA.items()},
    }
    definitions = {q[0]: q[2] for q in QUANTITIES} | {
        "ev_over_eqv_copula": "EV / EQV of the entry",
        "ev_over_eqv_s": "EV_S / EQV",
        **{
            k: f"per date, {a} minus {b}; then the mean across the group's dates"
            for k, (a, b) in DIFFERENCES.items()
        },
        **{k: v[1] for k, v in EXTRA.items()},
    }
    all_labels = labels | {k: v[0] for k, v in CHANNELS.items()}
    all_definitions = definitions | {k: v[1] for k, v in CHANNELS.items()}

    def unit_of(name: str) -> str:
        if "clip" in name:
            return "fraction of the particles"
        if name.startswith("idx_err"):
            return "vol points"
        return "log change" if name in CHANNELS else "ratio"

    def no_se_reason(name: str) -> str:
        if "clip" in name:
            return "a calibration diagnostic of the particle cloud: no standard error computed"
        if name in STUDY_ONLY:
            return "study table: no Monte Carlo error here"
        return ""

    def group_record(group: str, name: str, sample: str = ALL) -> None:
        r = row(group, name, sample)
        notes = f"median {r['median']:.6g}"
        if pd.notna(r["median_mc_se"]):
            notes += f"; Monte Carlo se of a date: median {r['median_mc_se']:.3g}, largest {r['max_mc_se']:.3g} ({r['max_mc_se_date']})"
        elif no_se_reason(name):
            notes += f"; per date: {no_se_reason(name)}"
        if pd.notna(r["se_clustered_by_year"]):
            notes += f"; se with the dates clustered by calendar year {r['se_clustered_by_year']:.3g} ({int(r['n_years'])} years)"
        from_selection = name.endswith("_selection")
        rec(
            f"D.{group}.{name}.{SUFFIX[sample]}", f"{all_labels[name]}: mean over {group} group" + ("" if sample == ALL else f", {sample}"), r["mean"], r["se_of_mean"],
            date=f"{group} group" + ("" if sample == ALL else f", {sample}"), unit=unit_of(name),
            definition=all_definitions[name] + "; mean across the group's priced dates" + ("" if sample == ALL else f" ({sample})") + ", se = standard error of that mean",
            budget=pc.BUDGETS["study"] if name in STUDY_ONLY else budget, commit=sel_commit if from_selection else commit, source=sel_src if from_selection else src, n=int(r["n"]), notes=notes,
        )  # fmt: skip

    moved = hi_d[hi_d["clip_lc"] < line]
    spec_diff = ok[ok["spec_differs"]]
    same_gap = float((ok.loc[~ok["spec_differs"], "clip_lc"] - ok.loc[~ok["spec_differs"], "clip_lc_selection"]).abs().max())  # fmt: skip
    same_txt = "are equal" if same_gap < 5e-7 else f"differ by at most {same_gap:.1e}"
    add(
        f"Source: `{table_path.relative_to(pc.LC_OUT.parent.parent)}` — `scripts/cdv_stratified.py --fixed 3,6 --g-max 3` on branch `cross-dependent-vol` "
        f"(commit {commit}: M12's defaults of 2026-10-09 merged in), development budget ({budget}). {len(dates)} dates selected from the variant development pass "
        f"(`{selection_path.name}`, commit {sel_commit}) by M12's clipped mass inside ±2.5 sd; priced: {n_by['high']} of the high group, {n_by['low']} of the low group"
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
        f"The selection. The groups are those of the selection table (commit {sel_commit}, before the DJX repair of decision 5): its {n_by['high']} largest and {n_by['low']} smallest clipped masses among its priced rows "
        f"(20th largest {line:.4f}; the low group's are at most {lo_d['clip_lc_selection'].max():.4f}, {int((lo_d['clip_lc_selection'] <= CLIP_LINE).sum())} of them under the 1 % line). This run (commit {commit}) recomputes M12 at the defaults of 9 Oct: "
        f"the specification key differs from the selection table's on {len(spec_diff)} dates ({', '.join(f'{r.date} ({r.group})' for r in spec_diff.itertuples())}), and M12's clipped mass recomputed here is below the selection table's 20th largest on {len(moved)} of them "
        f"({', '.join(f'{r.date} {r.clip_lc_selection:.3f} → {r.clip_lc:.3f}' for r in moved.itertuples())}); on the other {len(ok) - len(spec_diff)} dates the two clipped masses {same_txt}. "
        f"Both clipped masses are in D.1 and D.3. LC/CC of this run and of the selection table are two estimates: group means {cell(row('high', 'lc_over_cc')['mean'])} here and {cell(row('high', 'lc_over_cc_selection')['mean'])} in the selection table on the same {n_by['high']} high dates "
        f"({cell(row('low', 'lc_over_cc')['mean'])} and {cell(row('low', 'lc_over_cc_selection')['mean'])} on the low dates)."
    )
    add("")
    add(
        "### D.1 By group (mean across the group's dates ± the standard error of that mean, the dates taken as independent; median in brackets)"
    )
    add("")
    cols = []
    for group, _ in GROUPS:
        k = int((flagged["group"] == group).sum())
        cols.append((group, ALL, f"{HEAD[group]}" + (", all dates" if k else "")))
        if k:
            cols.append((group, UNFLAGGED, f"{HEAD[group]}, without the {k} flagged dates"))
    add("| quantity | " + " | ".join(c[2] for c in cols) + " |")
    add("|:--|" + "--:|" * len(cols))
    g_all = groups[groups["sample"] == ALL]
    for name in labels:
        digits = 2 if name.startswith("idx_err") else 4
        cells = []
        for group, sample, _ in cols:
            r = row(group, name, sample)
            cells.append(f"{cell(r['mean'], r['se_of_mean'], digits)} ({cell(r['median'], None, digits)}; n {int(r['n'])})")  # fmt: skip
            group_record(group, name, sample)
        add(f"| {labels[name]} | " + " | ".join(cells) + " |")
    add("")
    add("Discounts (1 - ratio), from the rows above: " + "; ".join(
        f"{label} {', '.join(f'{100 * (1 - float(g_all[(g_all.group == g) & (g_all.quantity == name)].iloc[0]['mean'])):.2f} % ({g})' for g, _ in GROUPS)}"
        for name, label in (("lc_over_cc", "LC against CC"), ("cdv3_over_cc", "CDV β=3 against CC"), ("cdv6_over_cc", "CDV β=6 against CC"),
                            ("s_over_copula", "model S against the copula"), ("listed_fwd", "listed-variance forward against the copula"),
                            ("lc_over_copula", "LC against the copula"), ("cdv3_over_copula", "CDV β=3 against the copula"), ("cdv6_over_copula", "CDV β=6 against the copula"))
    ) + ".")  # fmt: skip
    add("")
    add(
        "The clipped mass rows are fractions of the particles (sections A and B print the clipped mass in %). The index error at −2.5 sd is against the model's own index target, in vol points."
    )
    if len(flagged):
        no_flag = [g for g, _ in GROUPS if not int((flagged["group"] == g).sum())]
        add("")
        add(
            f"Flagged dates (a name kept unscreened, decision 2) among the {len(ok)}: {', '.join(f'{r.date} ({r.group})' for r in flagged.itertuples())}; the columns \"without the flagged dates\" leave them out"
            + (
                f" (no date of the {' or '.join(no_flag)} group is flagged: its two samples are the same)"
                if no_flag
                else ""
            )
            + "."
        )
    years = {g: ok[ok["group"] == g]["year"].value_counts().sort_index() for g, _ in GROUPS}
    ratio_of = {(g, name): float(row(g, name)["se_clustered_by_year"] / row(g, name)["se_of_mean"]) for g, _ in GROUPS for name in labels if row(g, name)["se_of_mean"] > 0}  # fmt: skip
    large = sorted(((v, g, name) for (g, name), v in ratio_of.items() if v >= 1.5), reverse=True)
    rest = [v for v in ratio_of.values() if v < 1.5]
    clustered = []
    for group, name in CLUSTERED:
        r = row(group, name)
        if not r["se_clustered_by_year"] > 1.4 * r["se_of_mean"]:
            raise ValueError(
                f"{group} {name}: the clustered standard error is not 1.4 times the printed one"
            )
        clustered.append(
            f"{labels[name]}, {group} group: {r['se_clustered_by_year']:.4f} against {r['se_of_mean']:.4f}"
        )
        rec(
            f"D.{group}.{name}.se_clustered_by_year", f"{labels[name]}: standard error of the {group} group's mean with the dates clustered by calendar year", r["se_clustered_by_year"], None, date=f"{group} group", unit=unit_of(name),
            definition="cluster-robust standard error of the mean across the group's dates, clusters = calendar years: sqrt(G/(G-1) * sum over years of (sum of the year's deviations from the mean)^2) / n",
            budget=pc.BUDGETS["study"] if name in STUDY_ONLY else budget, n=int(r["n"]), notes=f"a standard error itself: no error on it computed; {int(r['n_years'])} years; the printed standard error (dates independent) is {r['se_of_mean']:.6g}",
        )  # fmt: skip
    did_cl = {}
    for b in ("3", "6"):
        name = f"cdv{b}_minus_lc_over_cc"
        did_cl[b] = math.hypot(
            float(row("high", name)["se_clustered_by_year"]),
            float(row("low", name)["se_clustered_by_year"]),
        )
    add("")
    add(
        "Clustering. The ± treat a group's dates as independent. By calendar year the dates are: "
        + "; ".join(
            f"{g} group " + ", ".join(f"{y} {c}" for y, c in years[g].items()) for g, _ in GROUPS
        )
        + " (consecutive monthly dates have overlapping 3-month horizons). With the dates clustered by calendar year (cluster-robust standard error of the mean, G/(G−1) correction; column `se_clustered_by_year` of the by-group CSV) "
        "the standard errors of the four rows the reading leans on are: "
        + "; ".join(clustered)
        + f". Over all rows of D.1 the clustered standard error is 1.5 times the printed one or more on {len(large)} (group, row) pairs — "
        + "; ".join(f"{labels[name]}, {g}: {v:.1f}" for v, g, name in large)
        + f" — and {min(rest):.1f} to {max(rest):.1f} times on the other {len(rest)} (with {years['high'].size} and {years['low'].size} years the clustered errors are themselves rough). "
        f"High group minus low group of CDV/CC − LC/CC has a clustered standard error of {did_cl['3']:.4f} at β = 3 and {did_cl['6']:.4f} at β = 6."
    )
    not_conv = ok[ok["model_s_converged"] == False]  # noqa: E712
    if len(not_conv):
        add("")
        add(
            f"Model S did not converge on: {', '.join(not_conv['date'])}; its ratio is in the table as the study's file has it."
        )
    add("")
    add(
        "### D.1b The move from LC to the prototype, by channel (per date, in %: d log E[D] = d log κ + ½ d log E[V]; mean ± standard error of the mean, median in brackets)"
    )
    add("")
    add("| LC to | channel | " + " | ".join(f"{c[2]} (n {int(row(c[0], 'dlog_ed_cdv3', c[1])['n'])})" for c in cols) + " |")  # fmt: skip
    add("|:--|:--|" + "--:|" * len(cols))
    for b in ("3", "6"):
        for kind, lab in (("dlog_ed", "forward: d log E[D]"), ("dlog_kappa", "κ: d log κ"), ("half_dlog_ev", "√E[V]: ½ d log E[V]")):  # fmt: skip
            name = f"{kind}_cdv{b}"
            cells = []
            for group, sample, _ in cols:
                r = row(group, name, sample)
                cells.append(
                    f"{100 * r['mean']:+.2f} ± {100 * r['se_of_mean']:.2f} ({100 * r['median']:+.2f})"
                )
                group_record(group, name, sample)
            add(f"| β = {b} | {lab} | " + " | ".join(cells) + " |")
    add("")
    robust = ("lc_over_copula", "cdv3_over_copula", "cdv6_over_copula", "cdv3_minus_lc_over_cc", "cdv6_minus_lc_over_cc")  # fmt: skip
    for name in robust:
        group_record("high", name, NO_LOW_CLIP)
    for sample in (OUTSIDE_2020, IN_2020):
        group_record("high", "cdv6_minus_listed_fwd", sample)
    for sample in (CLIP6_LE, CLIP6_GT):
        group_record("low", "cdv6_minus_lc_over_cc", sample)
    add(
        "The identity holds per date to 1e-6 (checked on every priced date: d log E[D] is the log of the paired ratio CDV/LC, κ = E[D]/√E[V]); the records hold the log-changes, the table prints them in %. "
        f"Robustness of the high group to the selection: without the {len(moved)} dates whose M12 clipped mass in this run is below the selection table's 20th largest ({', '.join(moved['date'])}; n = {int(row('high', 'lc_over_copula', NO_LOW_CLIP)['n'])}), "
        + "; ".join(
            f"{labels[name]} {cell(row('high', name, NO_LOW_CLIP)['mean'], row('high', name, NO_LOW_CLIP)['se_of_mean'])}"
            for name in robust
        )
        + "."
    )
    add("")
    add(
        "### D.2 Reading: on the dates where M12 binds, does closing the wing move its discount towards model S's?"
    )
    add("")
    text, did, facts = reading(groups, dates, line)
    add(text)
    for (b, key), (value, err) in did.items():
        unflagged = key == "unflagged"
        rec(
            f"D.high_minus_low{'_unflagged' if unflagged else ''}.cdv{b}_minus_lc_over_cc", f"CDV(beta={b})/CC - LC/CC: mean over the high group minus mean over the low group" + (", without the flagged dates" if unflagged else ""), value, err,
            date="high group minus low group" + (", without the dates where a name is kept unscreened" if unflagged else ""), unit="ratio difference",
            definition="difference of the two groups' means of the per-date CDV/CC - LC/CC; se = root of the sum of the two squared standard errors of the means (the groups have no date in common)",
            n=len(ok) - (len(flagged) if unflagged else 0),
            notes="the part of the prototype's move that is specific to the dates where M12 binds" + ("" if unflagged else f"; se with the dates clustered by calendar year {did_cl[b]:.3g}"),
        )  # fmt: skip
    for b in ("3", "6"):
        rec(f"D.high_minus_low.cdv{b}_minus_lc_over_cc.se_clustered_by_year", f"CDV(beta={b})/CC - LC/CC, high group minus low group: standard error with the dates clustered by calendar year", did_cl[b], None, date="high group minus low group", unit="ratio difference",
            definition="root of the sum of the two groups' squared cluster-robust standard errors of the mean (clusters = calendar years)", n=len(ok), notes="a standard error itself: no error on it computed")  # fmt: skip
        rec(f"D.high.frac_to_s_cdv{b}.of_group_means", f"share of the distance from LC to model S covered at beta = {b}, on the high group's means", facts[f"way{b}"], None, date="high group", unit="fraction",
            definition=f"(mean LC/copula - mean CDV(beta={b})/copula) / (mean LC/copula - mean model S/copula), means across the high group's dates", n=n_by["high"], notes="a ratio of differences of group means: no standard error computed")  # fmt: skip
        r = row("high", f"frac_to_s_cdv{b}")
        rec(f"D.high.frac_to_s_cdv{b}.median", f"share of the distance from LC to model S covered at beta = {b}: median of the per-date shares, high group", r["median"], None, date="high group", unit="fraction",
            definition=f"per date, (LC/copula - CDV(beta={b})/copula) / (LC/copula - model S/copula); median across the high group's dates", n=int(r["n"]), notes="a median across dates: no standard error computed")  # fmt: skip
        rec(f"D.high.clip_cdv{b}.n_under_1pct", f"number of high-group dates with the clipped mass under 1 % at beta = {b}", facts[f"k{b}"], None, date="high group", unit="count of dates",
            definition=f"dates of the high group with clip_cdv{b} <= {CLIP_LINE}", n=n_by["high"], notes="a count: no standard error")  # fmt: skip
        rec(f"D.high.clip_cdv{b}.min", f"smallest clipped mass at beta = {b} among the high-group dates (fraction of the particles)", facts[f"min{b}"], None, date=str(hi_d.loc[hi_d[f"clip_cdv{b}"].idxmin(), "date"]), unit="fraction of the particles",
            definition=f"minimum of clip_cdv{b} over the high group's dates", n=n_by["high"], notes="a calibration diagnostic of the particle cloud: no standard error computed")  # fmt: skip
    for group, name in (
        ("low", "lc_minus_s_over_copula"),
        ("high", "lc_minus_s_over_copula"),
        ("low", "rbar2_lc"),
    ):
        r = row(group, name)
        rec(f"D.{group}.{name}.median", f"{labels[name]}: median over {group} group", r["median"], None, date=f"{group} group", unit="ratio", definition=definitions[name] + "; median across the group's priced dates", n=int(r["n"]), notes="a median across dates: no standard error computed")  # fmt: skip
    sel_kw = {"commit": sel_commit, "source": sel_src, "unit": "fraction of the particles"}
    rec("D.selection.clip_20th_largest", "20th largest M12 clipped mass among the priced rows of the selection table (fraction of the particles)", line, None, date="selection table", definition="20th largest clip_inner_max among the rows of the selection table that are not failed", notes="a calibration diagnostic of the particle cloud: no standard error computed", **sel_kw)  # fmt: skip
    rec("D.low.clip_lc_selection.max", "largest M12 clipped mass of the low group in the selection table (fraction of the particles)", facts["low_max"], None, date="low group", definition="maximum of clip_inner_max of the selection table over the low group's dates", n=n_by["low"], notes="a calibration diagnostic of the particle cloud: no standard error computed", **sel_kw)  # fmt: skip
    rec("D.low.clip_lc_selection.n_under_1pct", "number of low-group dates with M12's clipped mass under the 1 % line in the selection table", facts["low_under"], None, date="low group", definition=f"dates of the low group with clip_inner_max <= {CLIP_LINE} in the selection table", n=n_by["low"], notes="a count: no standard error", **(sel_kw | {"unit": "count of dates"}))  # fmt: skip
    rec("D.high.clip_lc.n_below_selection_line", "number of high-group dates whose M12 clipped mass in this run is below the selection table's 20th largest", facts["n_moved"], None, date="high group", unit="count of dates", definition="dates of the high group with clip_lc of this run < the 20th largest clip_inner_max of the selection table", n=n_by["high"], notes="a count: no standard error")  # fmt: skip
    rec("D.kappa.n_dates_not_resolved", f"number of dates where a kappa (CC, LC, CDV at beta 3 or 6) has a Monte Carlo error of {KAPPA_SE_LINE} or more", facts["n_unresolved"], None, date="the 40 dates", unit="count of dates", definition=f"dates with max(kappa_cc_se, kappa_lc_se, kappa_cdv3_se, kappa_cdv6_se) >= {KAPPA_SE_LINE}", n=len(ok), notes="a count: no standard error")  # fmt: skip
    rec("D.forward_ratio.max_mc_se", "largest Monte Carlo error of a forward ratio (to CC or to the copula; CC, LC, CDV at beta 3 and 6) among the dates", facts["fwd_se_max"], None, date="the 40 dates", unit="ratio", definition="maximum over the dates of the standard errors of LC/CC, CDV/CC (paired) and CC, LC, CDV over the copula", n=len(ok), notes="a standard error itself: no error on it computed")  # fmt: skip
    if paragraph is not None and paragraph.exists():
        add("")
        add(paragraph.read_text().strip())
    add("")
    add("### D.3 Per date (± the date's Monte Carlo standard error; ratios to CC are paired)")
    add("")
    add(
        "| date | group | history status | flags | M12 clipped mass: selection table / this run | clipped mass β=3 / β=6 | LC/CC | CDV(3)/CC | CDV(6)/CC | S/copula | listed fwd | LC/copula | CDV(3)/copula | CDV(6)/copula | κ CC | κ LC | κ CDV(3) | κ CDV(6) | E[R̄²]/M_B LC / CDV(3) / CDV(6) |"
    )
    add("|:--|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--|")
    per_date = (
        "lc_over_cc", "cdv3_over_cc", "cdv6_over_cc", "lc_over_copula", "cdv3_over_copula", "cdv6_over_copula", "s_over_copula", "listed_fwd", "clip_lc", "clip_cdv3", "clip_cdv6",
        *(f"{q}_{m}" for q in ("kappa", "ev_over_eqv", "rbar2") for m, _ in MODELS), "clip_lc_selection",
    )  # fmt: skip
    for d in dates.itertuples():
        if d.status == "failed":
            add(f"| {d.date} | {d.group} | failed: {d.reason} |" + " |" * 16)
            continue
        flags = [
            "unscreened" if d.flag_unscreened else "",
            f"index moment {100 * d.index_moment_rel:+.1f} %" if d.flag_index_moment else "",
            "spec" if d.spec_differs else "",
            "κ" if d.kappa_se_max >= KAPPA_SE_LINE else "",
        ]
        add(
            f"| {d.date}{' †' if d.flag_unscreened else ''} | {d.group} | {'check: ' + d.history_failed_gates if d.history_failed_gates else 'ok'} | {'; '.join(f for f in flags if f)} | {d.clip_lc_selection:.3f} / {d.clip_lc:.3f} | {d.clip_cdv3:.3f} / {d.clip_cdv6:.3f} | "
            f"{cell(d.lc_over_cc, d.lc_over_cc_se)} | {cell(d.cdv3_over_cc, d.cdv3_over_cc_se)} | "
            f"{cell(d.cdv6_over_cc, d.cdv6_over_cc_se)} | {cell(d.s_over_copula)} | {cell(d.listed_fwd)} | {cell(d.lc_over_copula, d.lc_over_copula_se)} | "
            f"{cell(d.cdv3_over_copula, d.cdv3_over_copula_se)} | {cell(d.cdv6_over_copula, d.cdv6_over_copula_se)} | "
            f"{cell(d.kappa_cc, d.kappa_cc_se)} | {cell(d.kappa_lc, d.kappa_lc_se)} | {cell(d.kappa_cdv3, d.kappa_cdv3_se)} | {cell(d.kappa_cdv6, d.kappa_cdv6_se)} | {d.rbar2_lc:.3f} / {d.rbar2_cdv3:.3f} / {d.rbar2_cdv6:.3f} |"
        )
        base_note = f"group {d.group}" + ("; a name kept unscreened" if d.flag_unscreened else "")
        for name in per_date:
            from_selection = name.endswith("_selection")
            rec(
                f"D.{d.date}.{name}", labels[name], getattr(d, name), getattr(d, f"{name}_se", None), date=str(d.date), unit=unit_of(name),
                definition=definitions[name], budget=budget if name not in ("s_over_copula", "listed_fwd") else pc.BUDGETS["study"], commit=sel_commit if from_selection else commit,
                source=sel_src if from_selection else src, notes=base_note + (f"; {no_se_reason(name)}" if no_se_reason(name) else ""),
            )  # fmt: skip
    add("")
    unres = ok[ok["kappa_se_max"] >= KAPPA_SE_LINE]
    worst = []
    for d in unres.itertuples():
        m_, lab = max(MODELS, key=lambda ml: getattr(d, f"kappa_{ml[0]}_se"))
        worst.append(f"{d.date} ({d.group}; κ {lab} {cell(getattr(d, f'kappa_{m_}'), getattr(d, f'kappa_{m_}_se'))}, E[V]/EQV {cell(getattr(d, f'ev_over_eqv_{m_}'), getattr(d, f'ev_over_eqv_{m_}_se'), 3)})")  # fmt: skip
    top = unres.loc[unres["kappa_se_max"].idxmax()]
    add(
        "Columns. History status: the row of the selection table under the current rule of `scripts/lcm_price.py` (`check` when `check_no_nan`, `check_forward` or `check_index` fails — the failing gate is named; `check_names` is a diagnostic, not a gate); "
        f"`check: check_index` is a date that fails the index gate ({', '.join(ok.loc[ok['flag_index_gate'], 'date']) or 'none'}). Flags: `unscreened` (†) a name is kept unscreened on that date (decision 2); `index moment x %` the basket part over M_B^listed of the selection table "
        f"is beyond ±{100 * INDEX_MOMENT_LINE:.0f} % (the index target is not the study's: ratios to the copula are not like for like on {', '.join(ok.loc[ok['flag_index_moment'], 'date']) or 'no date'}); `spec` the specification key of this run differs from the selection table's; "
        f"`κ` a κ has a Monte Carlo error of {KAPPA_SE_LINE} or more. The clipped masses are fractions of the particles."
    )
    add("")
    add(
        f"κ and E[V]/EQV are not resolved on {len(unres)} of the {len(ok)} dates ({int((unres['group'] == 'high').sum())} high, {int((unres['group'] == 'low').sum())} low): a κ under CC, LC or the prototype has a Monte Carlo error of {KAPPA_SE_LINE} or more "
        f"(the model with the largest error is shown): {'; '.join(worst)}. The largest is on {top['date']}. The forward ratios are not affected: the largest Monte Carlo error of a ratio to CC or to the copula among the {len(ok)} dates is {facts['fwd_se_max']:.4f}. "
        "κ under the copula and under model S, `E[V]/EQV` and `E[R̄²]/M_B^listed` under every model with their errors, the index errors at -2.5 sd and the per-date channels of D.1b are in the per-date CSV; κ, `E[V]/EQV` and `E[R̄²]/M_B^listed` per date and model are records with their errors."
    )
    pc.write_part("D_stratified", records, "\n".join(md))
    figs = (
        "| figure | file | CSV | what it shows |\n|:--|:--|:--|:--|\n"
        "| F1 | `figures/F1_forward_over_copula.pdf` | `figures/F1_forward_over_copula.csv` | the forward over the copula's on the dates of the history table, 2007-2026: LC, model S, the listed-variance forward (3m, development budget, decisions 1-2 on); the lines break at the 16 monthly dates of the study that have no row and at the 3 failed dates; today's point is the development-budget row |\n"
        "| F2 | `figures/F2_calls_over_copula_by_strike.pdf` | `figures/F2_calls_over_copula_by_strike.csv` | calls over the copula's by strike (0.75, 1, 1.25, 1.5 x the copula's forward), LC and model S: mean and interquartile range across dates (the 215 dates where model S converged; at 1.25 x and 1.5 x see the caveat of section V3) |\n"
        "| F3 | `figures/F3_stratified_cdv_by_group.pdf` | `figures/F3_stratified_cdv_by_group.csv` | the stratified CDV test by group (the 20 dates with the largest and the 20 with the smallest M12 clipped mass in the selection table): forward over the copula's under M12, CDV at β = 3 and β = 6, model S, and the listed-variance forward; the bars are means across the group's dates and the whiskers are the standard error of that mean across the group's dates (dates taken as independent; see D.1 for the clustering by calendar year); the vertical axis does not start at 0 |\n\n"
        "Vector PDF, 6.5 x 3 inches, plain matplotlib, no titles."
    )
    pc.write_part("F_figures", [], figs)
    if status:
        pc.status(
            f"D: part D_stratified revised after the verification (both samples, channels D.1b, flags and κ errors in D.3, D.2 rewritten; no existing value changed) from {len(ok)} priced dates of {len(dates)} ({n_by['high']} high, {n_by['low']} low); F3 redrawn."
        )
    log.info("D written: %d priced of %d dates; by group n %s; %d records", len(ok), len(dates), n_by, len(records))  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--table", default=str(TABLE))
    ap.add_argument("--selection", default=str(SELECTION), help="the history table the dates were selected from")  # fmt: skip
    ap.add_argument("--paragraph", default=str(pc.PM / "parts" / "D_reading.txt"))
    ap.add_argument("--no-status", action="store_true", help="do not append the line to STATUS.md (a rerun)")  # fmt: skip
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    build(Path(args.table), Path(args.selection), Path(args.paragraph), status=not args.no_status)
    return 0


if __name__ == "__main__":
    sys.exit(main())
