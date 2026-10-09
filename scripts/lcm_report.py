"""Local correlation model (SPEC §8.7, M12 part LC7): the comparison report of a sweep.

    python scripts/lcm_report.py --tenor 3m|12m|24m [--budget production|development]
        [--root <dir>] [--config <yaml>]

Reads the sweep's table (``outputs/dispersion_lc/lcm_<tenor>[_dev].parquet``, one row per date,
``scripts/disp_lcm.py``) and, read-only, the study's own tables (``entries_<tenor>.parquet``:
the copula's prices; ``model_s_3m.parquet``: model S at 3m), and writes
``outputs/dispersion_lc/report_<tenor>[_dev].md`` with its figures in
``outputs/dispersion_lc/figures/``.  It runs on a partial table.

What is compared (no pass or fail beyond the sanity checks):

* the forward: ``E_LC[D]/E_CC[D]`` (paired) against model S's ``P_D_S/P_D`` (the study's copula
  is the denominator there), and both models against the copula's ``P_D``;
* the calls at the study's cash strikes ``K_050 … K_200``: ``C_LC/C_CC`` against ``C_S/C``;
* per quantity: mean, quartiles, the median Monte Carlo standard error of a date, the
  correlation across dates of the two ratios, and the dates where LC and model S disagree in
  direction;
* the wing: ``ED_wing/ED_cc`` and ``ED_eqv/ED_cc`` beside ``E_LC[D]/E_CC[D]``, and how often the
  clipped mass inside ±2.5 sd exceeds 1 % (the index downside wing the model cannot reach);
* ``κ = E[D]/√E[V]`` under LC, CC and the copula; the sanity checks; the timings.

A FAIL or a failed date stays in every table.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lcm_price as lp

from volsto.studies import disp_data as dd

log = logging.getLogger("lcm_report")

TAGS = lp.MULT_TAGS
CAVEAT = (
    "Data caveat of the study (outputs/dispersion/PROGRESS_Q2.md) [measured there]: a name is priced "
    "beyond its last listed expiry on 226 of 972 dates at 12m and on 804 of 919 at 24m (28 of 30 names "
    "in the median at 24m), so the 24m numbers extrapolate the 12-18m smiles; DJX long-dated expiries "
    "at 1.5-1.8 times their neighbours on six dates (four in 2023, two in 2007) touch 12m and 24m."
)


def summary(x: pd.Series, se: pd.Series | None = None) -> dict[str, Any]:
    v = x.dropna()
    out: dict[str, Any] = {"n": len(v)}
    if len(v) == 0:
        return {
            **out,
            "mean": np.nan,
            "q25": np.nan,
            "median": np.nan,
            "q75": np.nan,
            "min": np.nan,
            "max": np.nan,
            "se of the mean": np.nan,
            "median MC se": np.nan,
        }
    out.update(
        mean=v.mean(), q25=v.quantile(0.25), median=v.median(), q75=v.quantile(0.75), min=v.min(), max=v.max(),
    )  # fmt: skip
    out["se of the mean"] = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else np.nan
    out["median MC se"] = se.dropna().median() if se is not None and se.notna().any() else np.nan
    return out


def table(rows: dict[str, dict[str, Any]], digits: int = 4) -> str:
    frame = pd.DataFrame(rows).T
    cols = list(frame.columns)
    lines = ["| quantity | " + " | ".join(cols) + " |", "|---|" + "---:|" * len(cols)]
    for name, r in frame.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if c == "n" or isinstance(v, (int, np.integer)):
                cells.append(f"{int(v)}")
            elif isinstance(v, str):
                cells.append(v)
            elif v is None or not np.isfinite(v):
                cells.append("n/a")
            else:
                cells.append(f"{v:.{digits + 2 if 'se' in c else digits}f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def agreement(a: pd.Series, b: pd.Series) -> dict[str, Any]:
    """Two ratios across dates: correlations of ``ratio − 1`` and the dates of opposite sign."""
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
    if len(both) < 3:
        return {"n": len(both), "pearson": np.nan, "spearman": np.nan, "n_opposite": 0, "dates": []}
    x, y = both["a"] - 1.0, both["b"] - 1.0
    opposite = both.index[(np.sign(x) * np.sign(y)) < 0]
    return {
        "n": len(both), "pearson": float(x.corr(y)), "spearman": float(x.corr(y, method="spearman")),
        "n_opposite": len(opposite), "dates": [str(d) for d in opposite],
    }  # fmt: skip


def load(tenor: str, budget: str, out: Path, label: str = "") -> tuple[pd.DataFrame, Path]:
    stem = (f"lcm_{tenor}" if budget == "production" else f"lcm_{tenor}_dev") + (
        f"_{label}" if label else ""
    )
    path = out / f"{stem}.parquet"
    rows = pd.read_parquet(path)
    entries = pd.read_parquet(dd.OUT / f"entries_{tenor}.parquet")
    entries = entries[entries["basket"] == "B1"].drop_duplicates("date").set_index("date")
    keep = ["P_D", "P_D_se", "EV", "kappa_cop", "sd_D", "SS", "Str_B", *[f"C_{t}" for t in TAGS]]
    frame = rows.set_index("date").join(entries[keep].add_suffix("_cop"), how="left")
    if tenor == "3m":
        s = pd.read_parquet(dd.OUT / "model_s_3m.parquet").set_index("date")
        frame = frame.join(
            s[["converged", "P_D_S", "EV_S", *[f"C_S_{t}" for t in TAGS]]], how="left"
        )
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index(), path


def build(tenor: str, budget: str, out: Path, label: str = "") -> Path:
    frame, source = load(tenor, budget, out, label)
    tag = (tenor if budget == "production" else f"{tenor}_dev") + (f"_{label}" if label else "")
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ok = frame[frame["status"] != "failed"].copy()
    failed = frame[frame["status"] == "failed"]
    has_s = "P_D_S" in ok.columns
    if has_s:
        conv = ok["converged"].fillna(False).astype(bool)
        ok["S_ratio"] = (ok["P_D_S"] / ok["P_D_cop"]).where(conv)
        for t in TAGS:
            ok[f"S_C_{t}"] = (ok[f"C_S_{t}"] / ok[f"C_{t}_cop"].where(ok[f"C_{t}_cop"] > 0)).where(
                conv
            )
    ok["lc_over_cop"] = ok["ED_lc"] / ok["P_D_cop"]
    ok["cc_over_cop"] = ok["ED_cc"] / ok["P_D_cop"]
    md: list[str] = []
    add = md.append
    commits = ", ".join(sorted({str(c) for c in frame["git_commit"].dropna()}))
    add(f"# Local correlation model against the study's models: {tenor}, {budget} budget")
    add("")
    add(
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{source.name}`: {len(frame)} dates "
        f"({ok.index.min():%Y-%m-%d} to {ok.index.max():%Y-%m-%d}), statuses "
        f"{frame['status'].value_counts().to_dict()}; commit(s) {commits}; "
        f"particles {sorted({int(x) for x in ok['n_particles'].dropna()})}, paths {sorted({int(x) for x in ok['n_paths'].dropna()})}."
    )
    add("")
    add(
        "LC: the calibrated local correlation model. CC: its constant-correlation companion (the same "
        "local vols, the constant correlation that reprices the index at-the-money straddle at the horizon), "
        "priced on the same paths. Copula: the study's model (`P_D`, `C_<m>` of `entries`). Model S: the "
        "study's skewed model (`model_s_3m.parquet`, its converged dates). Every number below is [measured] "
        "from these tables unless it says otherwise; standard errors are Monte Carlo errors of one date, "
        "or the standard error of a mean across dates where the column says so."
    )
    if tenor != "3m":
        add("")
        add(CAVEAT)
    if len(ok) < len(frame):
        add("")
        add(
            f"**{len(failed)} date(s) failed** and are in no statistic below: "
            + "; ".join(f"{d:%Y-%m-%d} ({r})" for d, r in failed["reason"].items())
        )
    # --- 1. the forward
    add("")
    add("## 1. The Palladium forward")
    add("")
    rows = {"E_LC[D] / E_CC[D] (paired)": summary(ok["ratio"], ok["ratio_se"])}
    if has_s:
        rows["model S / copula (P_D_S / P_D)"] = summary(ok["S_ratio"])
    rows["E_LC[D] / copula P_D"] = summary(ok["lc_over_cop"], ok["ED_lc_se"] / ok["P_D_cop"])
    rows["E_CC[D] / copula P_D"] = summary(ok["cc_over_cop"], ok["ED_cc_se"] / ok["P_D_cop"])
    add(table(rows))
    if has_s:
        a = agreement(ok["ratio"], ok["S_ratio"])
        add("")
        add(
            f"Across the {a['n']} dates with both: correlation of (LC/CC - 1) with (S/copula - 1) "
            f"{a['pearson']:+.3f} (Pearson), {a['spearman']:+.3f} (Spearman); opposite direction on "
            f"{a['n_opposite']} dates"
            + (
                ": "
                + ", ".join(d[:10] for d in a["dates"][:20])
                + (" ..." if a["n_opposite"] > 20 else "")
                if a["n_opposite"]
                else ""
            )
            + "."
        )
    # --- 2. the calls
    add("")
    add("## 2. Calls on the dispersion at the study's strikes")
    add("")
    rows = {}
    notes = []
    for t in TAGS:
        rows[f"K_{t}: C_LC / C_CC"] = summary(ok[f"C_{t}_ratio"], ok[f"C_{t}_ratio_se"])
        if has_s:
            rows[f"K_{t}: C_S / C_copula"] = summary(ok[f"S_C_{t}"])
            a = agreement(ok[f"C_{t}_ratio"], ok[f"S_C_{t}"])
            notes.append(
                f"- K_{t}: n {a['n']}, Pearson {a['pearson']:+.3f}, Spearman {a['spearman']:+.3f}, opposite direction on {a['n_opposite']} dates"
                + (
                    " ("
                    + ", ".join(d[:10] for d in a["dates"][:8])
                    + (" ..." if a["n_opposite"] > 8 else "")
                    + ")"
                    if a["n_opposite"]
                    else ""
                )
            )
    add(table(rows))
    if notes:
        add("")
        add("Agreement of the two ratios across dates:")
        add("")
        md.extend(notes)
    add("")
    add(
        "The strikes are multiples of the copula's forward `P_D`; at K_200 the copula's price is zero on many dates and the ratio is then undefined."
    )
    # --- 3. the wing
    add("")
    add("## 3. The index downside wing")
    add("")
    binds = ok["clip_inner_max"] > 0.01
    high, low = ok["clip_high_inner_max"] > 0.01, ok["clip_low_inner_max"] > 0.01
    rows = {
        "clipped mass inside ±2.5 sd, max over slices": summary(ok["clip_inner_max"]),
        "of which on the high side (λ at its cap)": summary(ok["clip_high_inner_max"]),
        "of which on the low side (λ at 0)": summary(ok["clip_low_inner_max"]),
        "clipped mass, whole cloud, high side, max": summary(ok["clip_high_max"]),
        "E_LC[V] / listed E_Q[V]": summary(ok["EV_over_EQV"], ok["EV_over_EQV_se"]),
        "ED_wing / E_CC[D]": summary(ok["ED_wing_ratio"], ok["ED_wing_ratio_se"]),
        "ED_eqv / E_CC[D]": summary(ok["ED_eqv_ratio"], ok["ED_eqv_ratio_se"]),
        "E_LC[D] / E_CC[D]": summary(ok["ratio"], ok["ratio_se"]),
        "index error at the 90 % strike (vp)": summary(ok["idx_err_90"], ok["idx_err_90_se"]),
        "index error at -1.5 sd (vp)": summary(ok["idx_err_m15"], ok["idx_err_m15_se"]),
        "index error at the money (vp)": summary(ok["idx_err_atm"], ok["idx_err_atm_se"]),
    }
    add(table(rows))
    add("")
    add(
        f"The clipped mass inside ±2.5 sd exceeds 1 % on {int(binds.sum())} of {len(ok)} dates ({binds.mean():.1%}) — on the high side "
        f"(the index target asks for more correlation than `rho_max` allows, the downside wing) on {int(high.sum())}, on the low side "
        f"(less than `rho_min` allows) on {int(low.sum())}: on those dates the smile is reported, not gated. "
        "`ED_wing = κ_LC·√(Σ w E_LC[R_i²] - M_B^listed)` replaces the model's basket second moment by the listed one; "
        "`ED_eqv = κ_LC·√EQV` replaces both. Where the wing binds: "
        f"mean LC/CC {ok.loc[binds, 'ratio'].mean():.4f}, ED_wing/CC {ok.loc[binds, 'ED_wing_ratio'].mean():.4f}, ED_eqv/CC {ok.loc[binds, 'ED_eqv_ratio'].mean():.4f} (n {int(binds.sum())}); "
        f"where it does not: {ok.loc[~binds, 'ratio'].mean():.4f}, {ok.loc[~binds, 'ED_wing_ratio'].mean():.4f}, {ok.loc[~binds, 'ED_eqv_ratio'].mean():.4f} (n {int((~binds).sum())})."
    )
    if has_s:
        add("")
        add(
            "Against model S across dates (the effect is the ratio minus 1; model S over its copula):"
        )
        add("")
        add("| LC quantity | dates | mean effect | model S, same dates | Pearson | Spearman |")
        add("|---|---:|---:|---:|---:|---:|")
        free = ok["clip_inner_max"] < 0.05
        for label, col, mask in (
            ("E_LC[D] / E_CC[D], all dates", "ratio", ok["ratio"].notna()),
            ("E_LC[D] / E_CC[D], clipped mass inside ±2.5 sd below 5 %", "ratio", free),
            ("E_LC[D] / E_CC[D], clipped mass 5 % or more", "ratio", ~free),
            ("ED_wing / E_CC[D], all dates", "ED_wing_ratio", ok["ratio"].notna()),
            ("ED_eqv / E_CC[D], all dates", "ED_eqv_ratio", ok["ratio"].notna()),
        ):
            both = pd.concat(
                [ok.loc[mask, col] - 1.0, ok.loc[mask, "S_ratio"] - 1.0], axis=1, keys=["a", "b"]
            ).dropna()
            if len(both) < 3:
                add(f"| {label} | {len(both)} | n/a | n/a | n/a | n/a |")
                continue
            add(
                f"| {label} | {len(both)} | {both['a'].mean():+.4f} | {both['b'].mean():+.4f} | "
                f"{both['a'].corr(both['b']):+.3f} | {both['a'].corr(both['b'], method='spearman'):+.3f} |"
            )
        both = pd.concat(
            [ok["clip_inner_max"], ok["S_ratio"] - 1.0], axis=1, keys=["a", "b"]
        ).dropna()
        if len(both) >= 3:
            add("")
            add(
                f"Correlation across dates of the clipped mass inside ±2.5 sd with model S's effect: {both['a'].corr(both['b']):+.3f} "
                f"(n {len(both)}): the dates on which model S moves the forward most are those on which the model is clipped most."
            )
    # --- 4. kappa, deltas
    add("")
    add("## 4. κ = E[D]/√E[V]" + (", and the deltas" if "delta_fwd_lc" in ok.columns else ""))
    add("")
    rows = {
        "κ LC": summary(ok["kappa_lc"], ok["kappa_lc_se"]),
        "κ CC": summary(ok["kappa_cc"], ok["kappa_cc_se"]),
        "κ copula": summary(ok["kappa_cop_cop"]),
        "λ_c": summary(ok["lambda_c"]),
        "rho_CC": summary(ok["rho_cc"]),
        "copula rho": summary(ok["rho_cop"]),
    }
    if "delta_fwd_lc" in ok.columns:
        rows.update({
            "forward: sticky-strike delta LC (% per +1 %)": summary(ok["delta_fwd_lc"], ok["delta_fwd_lc_se"]),
            "forward: sticky-strike delta CC": summary(ok["delta_fwd_cc"], ok["delta_fwd_cc_se"]),
            "forward: skew channel (CC_ss - 1)": summary(ok["delta_skew_channel"], ok["delta_skew_channel_se"]),
            "forward: correlation channel (LC_ss - CC_ss)": summary(ok["delta_correlation_channel"], ok["delta_correlation_channel_se"]),
            "call K_100: sticky-strike delta LC": summary(ok["delta_C100_lc"], ok["delta_C100_lc_se"]),
            "call K_100: sticky-strike delta CC": summary(ok["delta_C100_cc"], ok["delta_C100_cc_se"]),
        })  # fmt: skip
    add(table(rows))
    # --- 5. sanity checks
    add("")
    add("## 5. Sanity checks")
    add("")
    add("| check | dates that pass | dates that do not |")
    add("|---|---:|---|")
    for name, label in (
        ("check_no_nan", "no NaN"),
        ("check_forward", "E[B_T] within 3 standard errors of its forward"),
        (
            "check_index",
            "index ATM and 90 % vols within 0.15 vp, unless the clipped mass inside ±2.5 sd exceeds 1 %",
        ),
        ("check_names", "Σ w E[R_i²] within 2 % of the listed strips"),
    ):
        bad = ok.index[~ok[name].astype(bool)]
        shown = ", ".join(f"{d:%Y-%m-%d}" for d in bad[:12]) + (" ..." if len(bad) > 12 else "")
        add(f"| {label} | {len(ok) - len(bad)} of {len(ok)} | {shown or 'none'} |")
    add("")
    add(
        f"Index extrapolated beyond its last kept expiry on {int(ok['index_extrapolated'].sum())} dates; a name priced beyond its last "
        f"listed expiry on {int((ok['n_names_extrapolated'] > 0).sum())} dates; expiries dropped by the screen: median {ok['n_dropped'].median():.0f}, max {ok['n_dropped'].max():.0f} per date; "
        f"alignment flagged on {int(ok['align_flagged'].sum())} dates."
    )
    if "names_mc_over_listed" in ok.columns:
        add("")
        add(
            "The names' second moment `Σ w E[R_i²]` three ways — Monte Carlo under the model, the names' own SVI strips "
            "(no Monte Carlo; sensitive to the slices' wings beyond the quotes), the study's listed strips:"
        )
        add("")
        rows = {
            "Monte Carlo / listed strips - 1 (the check, 2 %)": summary(
                ok["names_mc_over_listed"], ok["sum_w_ER2_lc_se"] / ok["sum_w_M"]
            ),
            "SVI strips / listed strips - 1": summary(ok["names_svi_over_listed"]),
            "Monte Carlo / SVI strips - 1": summary(
                ok["names_mc_over_svi"], ok["sum_w_ER2_lc_se"] / ok["sum_w_M_svi"]
            ),
            "z of Monte Carlo against the SVI strips": summary(ok["names_mc_z"]),
        }
        add(table(rows))
    # --- 6. timings
    add("")
    add("## 6. Timings (seconds per date)")
    add("")
    cols = [
        "seconds_calibration",
        "seconds_companion",
        "seconds_pricing_lc",
        "seconds_pricing_cc",
        "seconds_deltas",
        "seconds_total",
    ]
    rows = {c.replace("seconds_", ""): summary(ok[c]) for c in cols if c in ok.columns}
    add(table(rows, digits=1))
    add("")
    add(
        f"Threads per process: {sorted({int(x) for x in ok['threads'].dropna()})}; cache hits {int(ok['cache_hit'].sum())} of {len(ok)}. The times are those of processes that shared the machine with the pass's other worker and with other runs."
    )
    # --- figures
    add("")
    add("## Figures")
    add("")
    for name, caption in figures_of(ok, tag, figures, has_s):
        add(f"![{caption}](figures/{name})")
        add("")
        add(f"*{caption}*")
        add("")
    path = out / f"report_{tag}.md"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(md) + "\n")
    tmp.replace(path)
    return path


def compare(tenor: str, budget: str, out: Path, label: str) -> Path:
    """A variant pass (``disp_lcm.py --tag <label>``) against the main pass of the same tenor and
    budget, on the dates both priced: the differences variant minus main of the forward, of its
    ratio, of the calls and their ratios, of the names' second moment, and the dates only one of
    them priced.  Written to ``report_<tenor>[_dev]_<label>_vs_main.md``."""
    base, _ = load(tenor, budget, out)
    var, source = load(tenor, budget, out, label)
    b, v = base[base["status"] != "failed"], var[var["status"] != "failed"]
    common = b.index.intersection(v.index)
    b, v = b.loc[common], v.loc[common]
    md: list[str] = []
    add = md.append
    stem = (tenor if budget == "production" else f"{tenor}_dev") + f"_{label}"
    add(f"# The variant pass `{label}` against the main pass: {tenor}, {budget} budget")
    add("")
    add(
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{source.name}` ({len(var)} dates, statuses "
        f"{var['status'].value_counts().to_dict()}; commit(s) {', '.join(sorted({str(c) for c in var['git_commit'].dropna()}))}) and the main "
        f"table ({len(base)} dates, statuses {base['status'].value_counts().to_dict()}). {len(common)} dates are priced by both. "
        "Differences are variant minus main; relative differences are of the variant over the main, minus 1. [measured]"
    )
    add("")
    rows: dict[str, dict[str, Any]] = {
        "E_LC[D], relative": summary(v["ED_lc"] / b["ED_lc"] - 1.0, b["ED_lc_se"] / b["ED_lc"]),
        "E_CC[D], relative": summary(v["ED_cc"] / b["ED_cc"] - 1.0, b["ED_cc_se"] / b["ED_cc"]),
        "LC/CC of the forward, difference": summary(v["ratio"] - b["ratio"], b["ratio_se"]),
    }
    for t in TAGS:
        ok_b = b[f"C_{t}_lc"] > 0
        rows[f"K_{t}: C_LC, relative"] = summary(
            (v[f"C_{t}_lc"] / b[f"C_{t}_lc"].where(ok_b) - 1.0),
            b[f"C_{t}_lc_se"] / b[f"C_{t}_lc"].where(ok_b),
        )
        rows[f"K_{t}: LC/CC, difference"] = summary(
            v[f"C_{t}_ratio"] - b[f"C_{t}_ratio"], b[f"C_{t}_ratio_se"]
        )
    rows.update({
        "names' second moment over listed strips: main": summary(b["names_mc_over_listed"]),
        "names' second moment over listed strips: variant": summary(v["names_mc_over_listed"]),
        "clipped mass inside ±2.5 sd, difference": summary(v["clip_inner_max"] - b["clip_inner_max"]),
        "ED_wing / E_CC[D], difference": summary(v["ED_wing_ratio"] - b["ED_wing_ratio"], b["ED_wing_ratio_se"]),
    })  # fmt: skip
    if "n_dropped_calendar" in v.columns:
        rows["slices dropped by the calendar repair, per date"] = summary(
            v["n_dropped_calendar"].astype(float)
        )
    add(table(rows))
    add("")
    by_year = pd.DataFrame({
        "year": common.year,
        "E_LC[D] rel": (v["ED_lc"] / b["ED_lc"] - 1.0).to_numpy(),
        "LC/CC diff": (v["ratio"] - b["ratio"]).to_numpy(),
        "C_125 rel": (v["C_125_lc"] / b["C_125_lc"].where(b["C_125_lc"] > 0) - 1.0).to_numpy(),
        "slices dropped": v["n_dropped_calendar"].to_numpy() if "n_dropped_calendar" in v.columns else np.nan,
    })  # fmt: skip
    by_year["period"] = pd.cut(
        by_year["year"],
        [2006, 2010, 2014, 2018, 2022, 2027],
        labels=["2007-10", "2011-14", "2015-18", "2019-22", "2023-26"],
    )
    add("By period (means over the dates of the period):")
    add("")
    add(
        "| period | dates | E_LC[D], relative | LC/CC, difference | call at K_125, relative | slices dropped per date |"
    )
    add("|---|---:|---:|---:|---:|---:|")
    for period, g in by_year.groupby("period", observed=True):
        add(
            f"| {period} | {len(g)} | {g['E_LC[D] rel'].mean():+.4f} | {g['LC/CC diff'].mean():+.4f} | {g['C_125 rel'].mean():+.3f} | {g['slices dropped'].mean():.1f} |"
        )
    only_v = sorted(
        set(var.index[var["status"] != "failed"]) - set(base.index[base["status"] != "failed"])
    )
    only_b = sorted(
        set(base.index[base["status"] != "failed"]) - set(var.index[var["status"] != "failed"])
    )
    add("")
    add(
        f"Priced by the variant only ({len(only_v)}): "
        + (", ".join(f"{d:%Y-%m-%d}" for d in only_v[:30]) or "none")
        + ("" if len(only_v) <= 30 else " ...")
    )
    if only_v and "names_unscreened" in var.columns:
        kept = var.loc[only_v, "names_unscreened"].fillna("").value_counts().to_dict()
        add("")
        add(
            f"On those dates the names kept unscreened are: {kept}; LC/CC of the forward there: mean {var.loc[only_v, 'ratio'].mean():.4f}, min {var.loc[only_v, 'ratio'].min():.4f}, max {var.loc[only_v, 'ratio'].max():.4f}."
        )
    add("")
    add(
        f"Priced by the main pass only ({len(only_b)}): "
        + (", ".join(f"{d:%Y-%m-%d}" for d in only_b[:30]) or "none")
        + ("" if len(only_b) <= 30 else " ...")
    )
    path = out / f"report_{stem}_vs_main.md"
    tmp = path.with_suffix(".md.tmp")
    tmp.write_text("\n".join(md) + "\n")
    tmp.replace(path)
    return path


def figures_of(ok: pd.DataFrame, tag: str, folder: Path, has_s: bool) -> list[tuple[str, str]]:
    made = []
    x = ok.index
    # 1. the forward over time
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.errorbar(
        x,
        ok["ratio"],
        yerr=2 * ok["ratio_se"],
        fmt="o",
        ms=3,
        lw=0.8,
        label="E_LC[D] / E_CC[D] (±2 se)",
    )
    if has_s:
        ax.plot(x, ok["S_ratio"], "s", ms=3, alpha=0.7, label="model S / copula")
    ax.plot(x, ok["lc_over_cop"], "^", ms=3, alpha=0.5, label="E_LC[D] / copula P_D")
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_ylabel("ratio")
    ax.legend(fontsize=8)
    ax.set_title(f"Palladium forward, {tag}: LC against CC, model S against the copula")
    made.append(
        (
            f"lcm_{tag}_forward.png",
            "The forward: LC over its constant-correlation companion, model S over the copula, LC over the copula, by date.",
        )
    )
    fig.tight_layout()
    fig.savefig(folder / made[-1][0], dpi=130)
    plt.close(fig)
    # 2. the calls by strike
    fig, ax = plt.subplots(figsize=(8, 4.2))
    pos = np.arange(len(TAGS))
    q = np.array([[ok[f"C_{t}_ratio"].quantile(p) for p in (0.25, 0.5, 0.75)] for t in TAGS])
    ax.errorbar(
        pos - 0.08,
        q[:, 1],
        yerr=[q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]],
        fmt="o",
        capsize=3,
        label="C_LC / C_CC",
    )
    if has_s:
        qs = np.array([[ok[f"S_C_{t}"].quantile(p) for p in (0.25, 0.5, 0.75)] for t in TAGS])
        ax.errorbar(
            pos + 0.08,
            qs[:, 1],
            yerr=[qs[:, 1] - qs[:, 0], qs[:, 2] - qs[:, 1]],
            fmt="s",
            capsize=3,
            label="C_S / C_copula",
        )
    ax.axhline(1.0, color="k", lw=0.6)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{int(t) / 100:g} x P_D" for t in TAGS])
    ax.set_ylabel("ratio: median and quartiles across dates")
    ax.set_yscale("log")
    ax.legend(fontsize=8)
    ax.set_title(f"Calls on the dispersion, {tag}")
    made.append(
        (
            f"lcm_{tag}_calls.png",
            "Calls at the study's strikes: median and quartiles across dates of LC/CC and of model S/copula (log scale).",
        )
    )
    fig.tight_layout()
    fig.savefig(folder / made[-1][0], dpi=130)
    plt.close(fig)
    # 3. the wing
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    a1.plot(x, ok["ratio"], "o", ms=3, label="E_LC[D] / E_CC[D]")
    a1.plot(x, ok["ED_wing_ratio"], "^", ms=3, alpha=0.7, label="ED_wing / E_CC[D]")
    a1.plot(x, ok["ED_eqv_ratio"], "s", ms=3, alpha=0.7, label="ED_eqv / E_CC[D]")
    a1.axhline(1.0, color="k", lw=0.6)
    a1.legend(fontsize=8)
    a1.set_ylabel("ratio")
    a2.semilogy(
        x,
        np.maximum(ok["clip_inner_max"], 1e-5),
        "o",
        ms=3,
        label="clipped mass inside ±2.5 sd (max over slices)",
    )
    a2.semilogy(
        x,
        np.maximum(ok["clip_high_max"], 1e-5),
        "x",
        ms=3,
        alpha=0.6,
        label="whole cloud, high side",
    )
    a2.axhline(0.01, color="r", lw=0.8, label="1 %")
    a2.legend(fontsize=8)
    a2.set_ylabel("mass")
    a1.set_title(
        f"The index downside wing, {tag}: wing-corrected forwards and how often the wing binds"
    )
    made.append(
        (
            f"lcm_{tag}_wing.png",
            "Top: the LC forward and its two wing-corrected versions over the CC forward. Bottom: the clipped mass (floored at 1e-5 for the log scale); above the red line the wing binds.",
        )
    )
    fig.tight_layout()
    fig.savefig(folder / made[-1][0], dpi=130)
    plt.close(fig)
    # 4. kappa and the index errors
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    a1.plot(x, ok["kappa_lc"], "o", ms=3, label="κ LC")
    a1.plot(x, ok["kappa_cc"], "s", ms=3, alpha=0.7, label="κ CC")
    a1.plot(x, ok["kappa_cop_cop"], "^", ms=3, alpha=0.6, label="κ copula")
    a1.legend(fontsize=8)
    a1.set_ylabel("κ = E[D]/√E[V]")
    a2.errorbar(
        x,
        ok["idx_err_atm"],
        yerr=2 * ok["idx_err_atm_se"],
        fmt="o",
        ms=3,
        lw=0.8,
        label="index ATM error (vp, ±2 se)",
    )
    a2.errorbar(
        x,
        ok["idx_err_90"],
        yerr=2 * ok["idx_err_90_se"],
        fmt="s",
        ms=3,
        lw=0.8,
        alpha=0.7,
        label="index 90 % error",
    )
    for level in (-0.15, 0.15):
        a2.axhline(level, color="r", lw=0.6)
    a2.set_ylabel("vol points")
    a2.legend(fontsize=8)
    a1.set_title(f"κ and the index repricing at the horizon, {tag}")
    made.append(
        (
            f"lcm_{tag}_kappa_index.png",
            "Top: κ under the three models. Bottom: the model's index implied vol minus the target at the horizon, at the money and at the 90 % strike (red: ±0.15 vp).",
        )
    )
    fig.tight_layout()
    fig.savefig(folder / made[-1][0], dpi=130)
    plt.close(fig)
    return made


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--tenor", default="3m", choices=("3m", "12m", "24m"))
    ap.add_argument("--budget", default="production", choices=("production", "development"))
    ap.add_argument("--config", default=str(lp.CONFIG))
    ap.add_argument("--root", default=None)
    ap.add_argument("--tag", default="", help="the suffix of a variant pass (disp_lcm.py --tag)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    cfg = lp.load_config(args.config)
    path = build(args.tenor, args.budget, lp.out_root(cfg, args.root), args.tag)
    log.info("written %s", path)
    if args.tag:
        try:
            log.info(
                "written %s",
                compare(args.tenor, args.budget, lp.out_root(cfg, args.root), args.tag),
            )
        except FileNotFoundError as exc:
            log.info("no comparison with the main pass: %s", exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
