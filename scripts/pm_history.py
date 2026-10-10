"""Section C of the PM results package of 2026-10-09 (``outputs/dispersion_lc/pm_update``): the
3m history of the local correlation model at the development budget with the owner's decisions
1-2 on, the answers to checks (b) and (d), and figures F1 and F2 (``--budget production``: the
same part on the production-budget pass, written elsewhere; see the end of this text).

Reads (read-only) ``outputs/dispersion_lc/lcm_3m_dev_repair.parquet`` (one row per date),
``outputs/dispersion/entries_3m.parquet`` (basket B1, one row per date: checked) and
``outputs/dispersion/model_s_3m.parquet``.  Writes, through ``scripts/pm_common.py`` only:

- ``tables/C_history_by_date.csv``: one row per date (failed dates stay, their numbers empty);
- ``tables/C_history_summaries.csv``: long format ``quantity, label, sample, statistic, value,
  se, se_kind, mc_se, nw_se, lag1_autocorr, n, nw_se_12, nw_se_24``;
- ``tables/C_history_samples.csv`` (the samples and the tercile bounds) and
  ``tables/C_history_regression.csv`` (check (d));
- ``figures/F1_forward_over_copula.pdf/.csv`` and ``figures/F2_calls_over_copula_by_strike.pdf/.csv``;
- ``parts/C_history.json`` and ``parts/C_history.md``.

Conventions.  A date is *priced* when its status is not ``failed``; every sample is a subset of
the priced dates.  The status printed and counted is recomputed here with the current rule of
``scripts/lcm_price.py`` (``GATING_CHECKS`` and ``row_status``: the owner's decision 3 of
2026-10-09 makes the names' 2 % check a reported diagnostic, not a gate); the status stored in
the rows, which predates the decision, stays in the per-date CSV as ``status_stored``.  A date is *flagged* when a name is kept unscreened (``n_names_unscreened >
0``, decision 2); every summary is given with and without the flagged dates.  Model S numbers
are used on its converged dates only, as ``scripts/disp_tables2.py::model_s_tables`` does
(``x = x[x["converged"]]``), and the pooled ratios mirror that function: the sum over the dates
of the numerator over the sum of the denominator (``gx["P_D_S"].sum() / gx["P_D"].sum()``,
``gx[f"C_S_{m}"].sum() / gx[f"C_{m}"].sum()``).

Restricted samples (added after the verification of 2026-10-09): ``all_index10`` and
``all_index15`` leave out the dates on which the model's basket second moment is more than 10 %
(15 %) from the listed index strip (``flag_index_moment``); ``all_names2pct`` keeps the dates on
which the names' diagnostic is inside 2 %; ``S_dropidx_lt4`` and ``S_dropidx_0`` are the dates of
check (d) with fewer than 4, and with no, index slices dropped by the screen.

Standard errors.  Per date: the Monte Carlo error (the row's for LC/CC, which is paired; the
delta method with the copula's own ``P_D_se`` / ``C_se_<m>`` for a ratio to the copula, the two
errors independent).  A mean over dates: ``se`` is the standard error across dates (sd/√n, no
serial-correlation adjustment); ``mc_se`` is an upper bound of its Monte Carlo error with the
dates fixed — the dates share the particle and pricing seeds, so the per-date errors are added
linearly (the mean of the per-date errors), not in quadrature; ``nw_se`` is the Newey-West
standard error of the mean (Bartlett kernel, 6 lags, the lags counted in consecutive dates of the
sample); ``nw_se_12`` and ``nw_se_24`` are the same at 12 and 24 lags (the 6-lag error is a lower
value: the error still grows with the lag, and the sample has gaps).  A pooled ratio ``R = Σa/Σb``:
``se`` is the across-dates linearisation ``√(n/(n−1)·Σ(a_i − R·b_i)²)/Σb``; ``mc_se`` is the
same upper bound, ``Σ sd(a_i − R·b_i)/Σb``.  The study's own numbers (model S, the copula's κ
and E[V], the listed-variance forward) carry no Monte Carlo error here.

Check (d) is an ordinary least squares fit with an intercept, classical, HC1 and Newey-West
(Bartlett, 6 lags, the factor n/(n − k) of HC1) standard errors (numpy only, QR); each leg of its
regressand (LC/CC, S/copula) is also fitted on the clipped mass alone.  When the results file of
the independent check is on disk (its copy under ``diagnostics/check_d`` of the package first) the
fits are compared with it to the digit.  A failed date's step and leg are read from the traceback
in the run's log of the date (``outputs/dispersion_lc/logs/3m_development_repair``).

Run: ``.venv/bin/python scripts/pm_history.py`` (idempotent; rerun when the table changes).

Another pass (added after the freeze of the package; the defaults are the frozen part's and
reproduce it byte for byte): ``--budget production`` reads the production-budget pass
``lcm_3m.parquet`` (commit d4faa74: decisions 1, 2 and 5 on) with the logs of that run; ``--rows``
and ``--logs`` name another table or log folder; ``--base <folder>`` writes ``parts/``, ``tables/``
and ``figures/`` under that folder instead of the package (the package is frozen: its own files
are refused by ``pm_common.guard_frozen``) and appends no line to the package's ``STATUS.md``.  The budget of the priced rows must be the one asked
for, and every label that names the budget, the commit, the decisions or the source follows the
table read (:class:`Pass`); no statistic depends on the option.  What differs by construction on
the production pass: the status stored in the rows is already under decision 3 and is printed as
the row's own; the row of the reference date in the table is section A's production row, and C.6
shows the frozen development row beside it; ``n_dropped_index`` of the rows counts the index
slices dropped under any rule, decision 5's calendar repair included, so the slices dropped by the
quote screen — what this part tabulates and conditions on — are ``n_dropped_index −
n_dropped_calendar_index`` (column ``n_dropped_index_screen`` of the per-date CSV).

    .venv/bin/python scripts/pm_history.py --budget production --no-status \\
        --base outputs/dispersion_lc/pm_update/later/production_3m/history
"""

# ruff: noqa: RUF001, E501 — report prose: typographic signs and long table lines

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pm_common as pc

LOG = logging.getLogger("pm_history")

SECTION = "C"
PART = "C_history"


@dataclass(frozen=True)
class Pass:
    """The pass behind the table read (``--budget``): what the part's labels say about it.  The
    statistics do not depend on it."""

    budget: str  #: key of ``pm_common.BUDGETS``
    sizes: tuple[float, float, float]  #: particles, paths, paths of the constant-correlation fit
    rows: Path  #: the table read by default
    logs: Path  #: the logs of the run behind it, one per date
    title: str  #: the heading after "C. History, 3m, " (``{commit}``: the commit of the rows)
    decisions: str  #: the owner's decisions that are on, as the labels write them
    repairs: str  #: what those decisions switch on


PASSES = {
    "development": Pass(
        "development",
        (2e5, 2e5, 1e5),
        pc.LC_OUT / "lcm_3m_dev_repair.parquet",
        pc.LC_OUT / "logs" / "3m_development_repair",
        "development budget, decisions 1–2 on",
        "decisions 1–2 on",
        "calendar repair of the names' slices and the unscreened fallback on (decisions 1–2)",
    ),
    "production": Pass(
        "production",
        (8e5, 8e5, 4e5),
        pc.LC_OUT / "lcm_3m.parquet",
        pc.LC_OUT / "logs" / "3m_production",
        "after the freeze: production budget, commit {commit}, decisions 1, 2 and 5 on",
        "decisions 1, 2 and 5 on",
        "calendar repair of the names' slices, unscreened fallback and calendar repair of the DJX target on (decisions 1, 2 and 5)",
    ),
}
#: The frozen part's pass: the default, and the development row shown in C.6 of another pass.
DEVELOPMENT = PASSES["development"]
ROWS = DEVELOPMENT.rows
ENTRIES = pc.STUDY / "entries_3m.parquet"
MODEL_S = pc.STUDY / "model_s_3m.parquet"
#: The independent check of (d) (another agent's results; compared when present): the copy kept in
#: the package, and the session scratch file it was copied from (read when the copy is missing).
CHECK_D_PACKAGE = pc.PM / "diagnostics" / "check_d" / "check_d_results.json"
CHECK_D_SCRATCH = Path(
    "/private/tmp/claude-501/-Users-idrissdadoun-Code-volsto/75dd7f23-d74c-43a9-b99e-61c23722b22c/scratchpad/r4/check-d/check_d_results.json"
)
#: The logs of the run behind the default table (one per date; a failed date's log has its
#: traceback); another pass has its own (:class:`Pass`, ``--logs``).
RUN_LOGS = DEVELOPMENT.logs
#: The stopped production pass at the old defaults: today's row is shown beside check (b).
PRODUCTION_OLD = pc.LC_OUT / "lcm_3m_norepair.parquet"
#: Today's production row at the new defaults (decisions 1-2 and 5 on): section A's row.
PRODUCTION_ROW = pc.LC_OUT / "rows" / "3m_production" / f"{pc.REFERENCE_DATES[0]}.json"
CALL_TAGS = ("075", "100", "125", "150")
#: ``scripts/lcm_price.py::CLIP_FLAG_MASS`` (copied: that file's neighbours are being edited).
CLIP_FLAG_MASS = 0.01
#: ``scripts/lcm_price.py::GATING_CHECKS`` (copied likewise): the sanity checks that set the status
#: of a priced row under the owner's decision 3 of 2026-10-09; ``check_names`` is not one of them.
GATING_CHECKS = ("check_no_nan", "check_forward", "check_index")
#: The tolerance of the index gate, in vol points, and the clipped mass above which it is waived
#: (``scripts/lcm_price.py``: ``check_index`` and ``wing_binds``).
INDEX_GATE_VP = 0.15
#: A date is marked ``flag_index_moment`` when |basket part / M_B^listed| exceeds the first; the
#: dates beyond the second are listed one by one.
INDEX_MOMENT_FLAG = 0.10
INDEX_MOMENT_LIST = 0.15
#: The names' diagnostic (``check_names``): Σw E_LC[R_i²] within this of the listed strips.
NAMES_TOL = 0.02
#: Newey-West: Bartlett kernel, this many lags.
NW_LAGS = 6
#: The longer lags at which the error of the mean is also given (the error still grows with the lag).
NW_MORE_LAGS = (12, 24)
#: Two consecutive dates of a sample more than this many calendar days apart are counted as a gap.
GAP_DAYS = 45
#: The commit field of a record read from the study's tables.
STUDY_COMMIT = "the study's tables (no commit column)"
#: The unit of the clipped-mass quantities (sections A and B print the same masses in %).
CLIP_UNIT = "fraction of the particles"
#: The rows of C.2 that carry the Newey-West error in the Markdown.
HEADLINE = ("lc_over_cc", "lc_over_copula", "cc_over_copula", "s_over_copula", "listed_fwd_ratio", "y_check_d")  # fmt: skip
#: The samples added to the base ones (no version without the flagged dates).
EXTRA_SAMPLES = ("all_index10", "all_index15", "all_names2pct", "S_dropidx_lt4", "S_dropidx_0")
CLIP_SHORT = "`clip_inner_max` = max(`clip_low_inner_max`, `clip_high_inner_max`): the larger of the two one-sided clipped masses, not their sum"
CLIP_DEF = (
    "the larger of the two one-sided clipped masses - the mass on which the correlation multiplier λ is clipped at 0, and the mass on which it is clipped at its cap - "
    "each taken at its own worst calibration slice, inside ±2.5 sd, as a fraction of the particles; it is not the sum of the two sides"
)
TODAY = pc.REFERENCE_DATES[0]
#: The index slices the calendar repair of the DJX target drops (decision 5): a column of the rows
#: of a pass that has the decision in its code; ``n_dropped_index`` of those rows includes them.
DROPPED_CALENDAR = "n_dropped_calendar_index"
#: The index slices dropped by the quote screen, in the per-date table of such a pass.
DROPPED_SCREEN = "n_dropped_index_screen"
#: The owner's arithmetic for today in check (b).
OWNER_B = {"lc_over_copula": 0.9693, "cc_over_copula": 0.9992}
BASE_SAMPLES = (
    ("all", "all priced dates"),
    ("h1", "2007–2016"),
    ("h2", "2017–2026"),
    ("clip_low", "clipped mass: low tercile"),
    ("clip_mid", "clipped mass: middle tercile"),
    ("clip_high", "clipped mass: high tercile"),
    ("S", "∩ model S converged"),
)
STUDY_NOTE = "the study's numbers on this table's dates: no Monte Carlo standard error here"


@dataclass(frozen=True)
class Quantity:
    """One per-date quantity: its column, label and definition, the column of its per-date
    standard error, and, for a ratio, the numerator and denominator of its pooled form."""

    key: str
    label: str
    definition: str
    se: str | None = None
    num: str | None = None
    den: str | None = None
    num_se: str | None = None
    den_se: str | None = None
    paired: bool = False
    study: bool = False
    digits: int = 4
    in_md: bool = True


def truth(flag: pd.Series) -> pd.Series:
    """A flag column as booleans (empty on a failed date: False)."""
    return flag.map(lambda x: x is True or x == 1.0).astype(bool)


def mult(tag: str) -> str:
    return f"{int(tag) / 100:g}"


def pct(x: float) -> str:
    """A threshold as a percentage, written as the part writes them ("10 %")."""
    return f"{100.0 * x:g} %"


def quantities() -> list[Quantity]:
    cop = "P_D the copula's forward of `entries_3m.parquet` (basket B1)"
    out = [
        Quantity("lc_over_cc", "LC/CC", "E_LC[D] / E_CC[D], paired on common paths (the row's `ratio` and `ratio_se`)", "lc_over_cc_se", "ED_lc", "ED_cc", "ED_lc_se", "ED_cc_se", paired=True),
        Quantity("lc_over_copula", "LC/copula", f"E_LC[D] / P_D, {cop}; se: delta method with `ED_lc_se` and the copula's `P_D_se`, independent", "lc_over_copula_se", "ED_lc", "P_D", "ED_lc_se", "P_D_se"),
        Quantity("cc_over_copula", "CC/copula", f"E_CC[D] / P_D, {cop}; se: delta method with `ED_cc_se` and the copula's `P_D_se`, independent", "cc_over_copula_se", "ED_cc", "P_D", "ED_cc_se", "P_D_se"),
        Quantity("s_over_copula", "S/copula", "P_D_S / P_D on the dates where model S converged (`model_s_3m.parquet`)", num="P_D_S", den="P_D", study=True),
        Quantity("listed_fwd_ratio", "listed-variance forward / copula", "√(EQV / EV), EV the copula's E[V] of the entry: κ_cop·√EQV over P_D", num="listed_fwd", den="P_D", study=True),
        Quantity("ED_wing_over_copula", "ED_wing / copula", "κ_LC·√(Σw E_LC[R_i²] − M_B^listed) / P_D; se: delta method with `ED_wing_se` and `P_D_se`", "ED_wing_over_copula_se", "ED_wing", "P_D", "ED_wing_se", "P_D_se"),
        Quantity("ED_eqv_over_copula", "ED_eqv / copula", "κ_LC·√EQV / P_D; se: delta method with `ED_eqv_se` and `P_D_se`", "ED_eqv_over_copula_se", "ED_eqv", "P_D", "ED_eqv_se", "P_D_se"),
        Quantity("kappa_lc", "κ, LC", "E_LC[D] / √E_LC[V]", "kappa_lc_se"),
        Quantity("kappa_cc", "κ, CC", "E_CC[D] / √E_CC[V]", "kappa_cc_se"),
        Quantity("kappa_cop", "κ, copula", "P_D / √EV (`kappa_cop` of the entry)", study=True),
        Quantity("kappa_S", "κ, model S", "P_D_S / √EV_S on the dates where model S converged", study=True),
        Quantity("EV_over_EQV_lc", "E[V]/EQV, LC", "EV_lc / EQV, EQV the listed-option value of E[V]; se: `EV_lc_se` / EQV", "EV_over_EQV_lc_se", "EV_lc", "EQV", "EV_lc_se"),
        Quantity("EV_over_EQV_cc", "E[V]/EQV, CC", "EV_cc / EQV; se: `EV_cc_se` / EQV", "EV_over_EQV_cc_se", "EV_cc", "EQV", "EV_cc_se"),
        Quantity("EV_over_EQV_cop", "E[V]/EQV, copula", "EV / EQV of the entry", num="EV_cop", den="EQV", study=True),
        Quantity("EV_over_EQV_S", "E[V]/EQV, model S", "EV_S / EQV on the dates where model S converged", num="EV_S", den="EQV", study=True),
        Quantity("EV_single_part", "E[V] split, single-name part", "Σw E_LC[R_i²] − Σw M_i^listed (EV_lc − EQV = single-name part − basket part)", "EV_single_part_se", digits=6),
        Quantity("EV_basket_part", "E[V] split, basket part", "E_LC[R̄²] − M_B^listed", "EV_basket_part_se", digits=6),
        Quantity("EV_basket_part_rel_MB", "basket part / M_B^listed", "(E_LC[R̄²] − M_B^listed) / M_B^listed; se: `EV_basket_part_se` / M_B^listed", "EV_basket_part_rel_MB_se"),
        Quantity("clip_inner_max", f"clipped mass inside ±2.5 sd ({CLIP_UNIT})", f"`clip_inner_max` = max(`clip_low_inner_max`, `clip_high_inner_max`): {CLIP_DEF}"),
        Quantity("clip_low_inner_max", f"clipped mass at λ = 0 ({CLIP_UNIT})", "the fraction of the particles inside ±2.5 sd on which λ is clipped at its lower bound 0, at the calibration slice where that fraction is largest (`clip_low_inner_max`)"),
        Quantity("clip_high_inner_max", f"clipped mass at the cap ({CLIP_UNIT})", "the fraction of the particles inside ±2.5 sd on which λ is clipped at its cap, at the calibration slice where that fraction is largest (`clip_high_inner_max`)"),
    ]  # fmt: skip
    for m in CALL_TAGS:
        k = f"K = {mult(m)} × P_D (the study's cash strike `K_{m}`, the same under every model)"
        out.append(Quantity(f"C_lc_over_copula_{m}", f"call {mult(m)}×: LC/copula", f"E_LC[(D − K)⁺] over the copula's `C_{m}`, {k}; se: delta method with `C_{m}_lc_se` and the copula's `C_se_{m}`", f"C_lc_over_copula_{m}_se", f"C_{m}_lc", f"C_{m}_cop", f"C_{m}_lc_se", f"C_{m}_cop_se"))  # fmt: skip
    for m in CALL_TAGS:
        out.append(Quantity(f"C_S_over_copula_{m}", f"call {mult(m)}×: S/copula", f"model S's `C_S_{m}` over the copula's `C_{m}`, same cash strike, on the dates where model S converged", num=f"C_{m}_S", den=f"C_{m}_cop", study=True))  # fmt: skip
    for m in CALL_TAGS:
        out.append(Quantity(f"C_cc_over_copula_{m}", f"call {mult(m)}×: CC/copula", f"E_CC[(D − K)⁺] over the copula's `C_{m}`, same cash strike; se: delta method", f"C_cc_over_copula_{m}_se", f"C_{m}_cc", f"C_{m}_cop", f"C_{m}_cc_se", f"C_{m}_cop_se", in_md=False))  # fmt: skip
    out.append(Quantity("y_check_d", "LC/CC − S/copula", "the regressand of check (d): `lc_over_cc` − `s_over_copula`, on the dates where model S converged; no standard error per date here (model S has none)"))  # fmt: skip
    return out


# ----------------------------------------------------------------------------- the per-date table
def ratio_and_se(
    a: pd.Series, a_se: pd.Series | None, b: pd.Series, b_se: pd.Series | None
) -> tuple[pd.Series, pd.Series]:
    """``a / b`` per date (empty where the denominator is not positive) with the delta-method
    standard error of two independent estimates (``pm_common.ratio_se``)."""
    safe = b.where(b > 0.0)
    ratio = a / safe
    zeros = pd.Series(0.0, index=a.index)
    sa = zeros if a_se is None else a_se
    sb = zeros if b_se is None else b_se
    se = []
    for x, xs, y, ys in zip(a, sa, safe, sb, strict=True):
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(xs) and np.isfinite(ys)):
            se.append(np.nan)
        elif x == 0.0:
            se.append(xs / y)
        else:
            se.append(pc.ratio_se(float(x), float(xs), float(y), float(ys)))
    return ratio, pd.Series(se, index=a.index)


def load(rows_path: Path) -> pd.DataFrame:
    """The sweep's rows joined on the date with the study's entry (basket B1) and model S."""
    rows = pd.read_parquet(rows_path)
    if not rows["date"].is_unique:
        raise ValueError(f"{rows_path.name}: several rows on a date")
    entries = pd.read_parquet(ENTRIES)
    b1 = entries[entries["basket"] == "B1"]
    if not b1["date"].is_unique:
        raise ValueError("entries_3m: several B1 rows on a date")
    keep = {"P_D": "P_D", "P_D_se": "P_D_se", "EV": "EV_cop", "EQV": "EQV_cop", "kappa_cop": "kappa_cop", "monthly": "monthly"}  # fmt: skip
    for m in CALL_TAGS:
        keep |= {f"K_{m}": f"K_{m}_cop", f"C_{m}": f"C_{m}_cop", f"C_se_{m}": f"C_{m}_cop_se"}
    b1 = b1[["date", *keep]].rename(columns=keep)
    s = pd.read_parquet(MODEL_S)
    if not s["date"].is_unique:
        raise ValueError("model_s_3m: several rows on a date")
    s_keep = {"converged": "model_s_converged", "P_D_S": "P_D_S", "EV_S": "EV_S"}
    s_keep |= {f"C_S_{m}": f"C_{m}_S" for m in CALL_TAGS}
    s = s[["date", *s_keep]].rename(columns=s_keep)
    frame = rows.merge(b1, on="date", how="left", validate="one_to_one")
    frame = frame.merge(s, on="date", how="left", validate="one_to_one", indicator="model_s_row")
    frame["model_s_row"] = frame["model_s_row"] == "both"
    if frame["P_D"].isna().any():
        raise ValueError(
            f"dates without a B1 entry: {frame.loc[frame['P_D'].isna(), 'date'].tolist()}"
        )
    frame = frame.sort_values("date").reset_index(drop=True)
    priced = (frame["status"] != "failed") & frame["ED_lc"].notna()
    # the copula's numbers of the row are the entry's: a row priced against another entry is refused
    for ours, theirs in (("P_D_copula", "P_D"), ("EV_copula", "EV_cop"), ("EQV", "EQV_cop"), *((f"K_{m}", f"K_{m}_cop") for m in CALL_TAGS)):  # fmt: skip
        gap = (frame.loc[priced, ours] - frame.loc[priced, theirs]).abs() / frame.loc[
            priced, theirs
        ].abs()
        if not (gap <= 1e-12).all():
            raise ValueError(
                f"{ours} of the rows differs from the entry's {theirs} (max relative {gap.max():.3g})"
            )
    frame["priced"] = priced
    # the status under the current rule (decision 3): only GATING_CHECKS set "check"; a row that is
    # not priced keeps its stored status and reason
    failing = pd.Series("", index=frame.index)
    for k in GATING_CHECKS:
        bad = priced & ~truth(frame[k])
        failing = failing.where(~bad, failing + np.where(failing != "", ", ", "") + k)
    frame["status_stored"] = frame["status"]
    frame["reason_stored"] = frame["reason"].fillna("")
    frame["status"] = np.where(priced, np.where(failing != "", "check", "ok"), frame["status"])
    frame["reason"] = np.where(
        priced,
        np.where(failing != "", "sanity checks: " + failing, ""),
        frame["reason_stored"],
    )
    gap = frame["sum_w_ER2_lc"] / frame["sum_w_M"] - 1.0
    if not ((gap.abs() <= NAMES_TOL)[priced] == truth(frame["check_names"])[priced]).all():
        raise ValueError("check_names of the rows is not |Σw E[R_i²] / Σw M_i - 1| <= 2 %")
    frame["names_gap"] = gap
    frame["names_gap_se"] = frame["sum_w_ER2_lc_se"] / frame["sum_w_M"]
    # what check_no_nan read (scripts/lcm_price.py): the floats of the row but these prefixes.  A
    # column that is empty on a priced date that passes the check is not a float of that row (a
    # key the row does not carry: the columns a later commit writes on the reference dates only),
    # so it is not what makes the check fail elsewhere: left out
    floats = [c for c in rows.columns if rows[c].dtype == float and not c.startswith(("align_", "C_200", "Cfwd", "profile_"))]  # fmt: skip
    passing = priced & truth(frame["check_no_nan"])
    floats = [c for c in floats if np.isfinite(frame.loc[passing, c]).all()]
    frame["non_finite_columns"] = ""
    for i in frame.index[priced & ~truth(frame["check_no_nan"])]:
        frame.loc[i, "non_finite_columns"] = ", ".join(c for c in floats if not np.isfinite(frame.loc[i, c]))  # fmt: skip
    sides = np.maximum(frame["clip_low_inner_max"], frame["clip_high_inner_max"])
    if not ((frame["clip_inner_max"] - sides).abs()[priced] <= 1e-12).all():
        raise ValueError("clip_inner_max is not the larger of the two one-sided masses")
    if not ((frame["clip_inner_max"] > CLIP_FLAG_MASS)[priced] == truth(frame["wing_binds"])[priced]).all():  # fmt: skip
        raise ValueError("wing_binds of the rows is not clip_inner_max > 0.01")
    return frame


def study_dates() -> tuple[list[str], set[str]]:
    """The study's monthly B1 dates (``entries_3m``, ``monthly``) and the dates on which model S
    converged (``model_s_3m``, ``converged``)."""
    entries = pd.read_parquet(ENTRIES, columns=["date", "basket", "monthly"])
    b1 = entries[entries["basket"] == "B1"]
    s = pd.read_parquet(MODEL_S, columns=["date", "converged"])
    return sorted(b1.loc[truth(b1["monthly"]), "date"]), set(s.loc[truth(s["converged"]), "date"])


def by_date(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per date with the flags, every quantity of section C and its standard error."""
    f = frame
    priced = f["priced"]
    cols: dict[str, Any] = {
        "date": f["date"],
        "status": f["status"],
        "status_stored": f["status_stored"],
        "reason": f["reason"].fillna(""),
        "reason_stored": f["reason_stored"],
        "T": f["T"],
        "monthly": truth(f["monthly"]),
    }
    d: Any = cols  # filled as a dict, then one frame (no fragmentation)
    year = pd.to_datetime(f["date"]).dt.year
    d["half"] = np.where(year <= 2016, "2007-2016", "2017-2026")
    converged = f["model_s_converged"].fillna(False).astype(bool)
    d["flag_unscreened"] = f["n_names_unscreened"] > 0
    d["names_unscreened"] = f["names_unscreened"].fillna("")
    d["n_names_unscreened"] = f["n_names_unscreened"]
    d["flag_clip_low"] = f["clip_low_inner_max"] > CLIP_FLAG_MASS
    d["flag_clip_high"] = f["clip_high_inner_max"] > CLIP_FLAG_MASS
    d["flag_clip"] = d["flag_clip_low"] | d["flag_clip_high"]
    d["model_s_converged"] = converged
    for k in (*GATING_CHECKS, "check_names"):
        d[k] = truth(f[k])
    d["names_within_2pct"] = truth(f["check_names"])
    d["names_gap"], d["names_gap_se"] = f["names_gap"], f["names_gap_se"]
    d["non_finite_columns"] = f["non_finite_columns"]
    d["wing_binds"] = truth(f["wing_binds"])
    d["idx_err_atm"], d["idx_err_90"] = f["idx_err_atm"], f["idx_err_90"]
    d["index_error_above_gate"] = (f["idx_err_atm"].abs() > INDEX_GATE_VP) | (
        f["idx_err_90"].abs() > INDEX_GATE_VP
    )
    rel_mb = f["EV_basket_part"] / f["M_B_listed"].where(f["M_B_listed"] > 0.0)
    d["flag_index_moment"] = rel_mb.abs() > INDEX_MOMENT_FLAG
    d["n_dropped_index"] = f["n_dropped_index"]
    if DROPPED_CALENDAR in f.columns:
        # the rows of a pass with decision 5: n_dropped_index counts the index slices dropped
        # under any rule; those dropped by the quote screen are the rest (module docstring)
        d[DROPPED_CALENDAR] = f[DROPPED_CALENDAR]
        d[DROPPED_SCREEN] = f["n_dropped_index"] - f[DROPPED_CALENDAR]
    d["rho_cc"], d["rho_cop"] = f["rho_cc"], f["rho_cop"]
    d["clip_larger_side"] = np.where(
        f["clip_low_inner_max"] > f["clip_high_inner_max"],
        "low",
        np.where(f["clip_low_inner_max"] < f["clip_high_inner_max"], "high", "equal"),
    )
    d["n_names_extrapolated"] = f["n_names_extrapolated"]
    d["index_extrapolated"] = f["index_extrapolated"]
    for c in ("n_particles", "n_paths", "companion_paths", "git_commit"):
        d[c] = f[c]
    for c in ("ED_lc", "ED_lc_se", "ED_cc", "ED_cc_se", "P_D", "P_D_se"):
        d[c] = f[c]
    use_s = priced & converged
    d["P_D_S"] = f["P_D_S"].where(use_s)
    d["lc_over_cc"], d["lc_over_cc_se"] = f["ratio"], f["ratio_se"]
    d["lc_over_copula"], d["lc_over_copula_se"] = ratio_and_se(
        f["ED_lc"], f["ED_lc_se"], f["P_D"], f["P_D_se"]
    )
    d["cc_over_copula"], d["cc_over_copula_se"] = ratio_and_se(
        f["ED_cc"], f["ED_cc_se"], f["P_D"], f["P_D_se"]
    )
    d["s_over_copula"] = d["P_D_S"] / f["P_D"].where(f["P_D"] > 0.0)
    ok_var = (f["EQV_cop"] > 0.0) & (f["EV_cop"] > 0.0)
    d["listed_fwd_ratio"] = np.sqrt((f["EQV_cop"] / f["EV_cop"]).where(ok_var))
    d["listed_fwd"] = f["P_D"] * d["listed_fwd_ratio"]
    for name in ("ED_wing", "ED_eqv"):
        d[name], d[f"{name}_se"] = f[name], f[f"{name}_se"]
        d[f"{name}_over_copula"], d[f"{name}_over_copula_se"] = ratio_and_se(
            f[name], f[f"{name}_se"], f["P_D"], f["P_D_se"]
        )
        d[f"{name}_over_cc"], d[f"{name}_over_cc_se"] = f[f"{name}_ratio"], f[f"{name}_ratio_se"]
    for c in ("kappa_lc", "kappa_lc_se", "kappa_cc", "kappa_cc_se", "kappa_cop"):
        d[c] = f[c]
    d["EV_S"] = f["EV_S"].where(use_s)
    d["kappa_S"] = d["P_D_S"] / np.sqrt(d["EV_S"].where(d["EV_S"] > 0.0))
    for c in ("EV_lc", "EV_lc_se", "EV_cc", "EV_cc_se", "EV_cop"):
        d[c] = f[c]
    d["EQV"], d["M_B_listed"] = f["EQV"], f["M_B_listed"]
    eqv = f["EQV"].where(f["EQV"] > 0.0)
    d["EV_over_EQV_lc"], d["EV_over_EQV_lc_se"] = f["EV_lc"] / eqv, f["EV_lc_se"] / eqv
    d["EV_over_EQV_cc"], d["EV_over_EQV_cc_se"] = f["EV_cc"] / eqv, f["EV_cc_se"] / eqv
    d["EV_over_EQV_cop"] = f["EV_cop"] / eqv
    d["EV_over_EQV_S"] = d["EV_S"] / eqv
    for c in ("EV_single_part", "EV_single_part_se", "EV_basket_part", "EV_basket_part_se"):
        d[c] = f[c]
    m_b = f["M_B_listed"].where(f["M_B_listed"] > 0.0)
    d["EV_basket_part_rel_MB"], d["EV_basket_part_rel_MB_se"] = (
        f["EV_basket_part"] / m_b,
        f["EV_basket_part_se"] / m_b,
    )
    for c in ("clip_inner_max", "clip_low_inner_max", "clip_high_inner_max"):
        d[c] = f[c]
    for m in CALL_TAGS:
        d[f"K_{m}"] = f[f"K_{m}_cop"]
        for c in (
            f"C_{m}_cop",
            f"C_{m}_cop_se",
            f"C_{m}_lc",
            f"C_{m}_lc_se",
            f"C_{m}_cc",
            f"C_{m}_cc_se",
        ):
            d[c] = f[c]
        d[f"C_{m}_S"] = f[f"C_{m}_S"].where(use_s)
        for tag in ("lc", "cc"):
            d[f"C_{tag}_over_copula_{m}"], d[f"C_{tag}_over_copula_{m}_se"] = ratio_and_se(f[f"C_{m}_{tag}"], f[f"C_{m}_{tag}_se"], f[f"C_{m}_cop"], f[f"C_{m}_cop_se"])  # fmt: skip
        d[f"C_S_over_copula_{m}"] = d[f"C_{m}_S"] / f[f"C_{m}_cop"].where(f[f"C_{m}_cop"] > 0.0)
    d["y_check_d"] = d["lc_over_cc"] - d["s_over_copula"]
    d = pd.DataFrame(cols)
    # the identities the definitions rest on, on the priced dates
    checks = {
        "EV_over_EQV of the row = EV_lc/EQV": (f["EV_over_EQV"] - d["EV_over_EQV_lc"]).abs(),
        "kappa_cop = P_D/sqrt(EV)": (f["kappa_cop"] - f["P_D"] / np.sqrt(f["EV_cop"])).abs(),
        "EV_lc - EQV = single part - basket part": (f["EV_lc"] - f["EQV"] - f["EV_single_part"] + f["EV_basket_part"]).abs(),
        "ratio = ED_lc/ED_cc": (f["ratio"] - f["ED_lc"] / f["ED_cc"]).abs(),
    }  # fmt: skip
    for name, gap in checks.items():
        worst = float(gap[priced].max())
        if not worst <= 1e-10:
            raise ValueError(f"identity broken: {name} (max gap {worst:.3g})")
    # failed dates stay with their reason; every number is empty there
    numeric = [c for c in d.columns if c not in ("date", "status", "status_stored", "reason", "reason_stored", "monthly", "half", "names_unscreened", "non_finite_columns", "git_commit", "model_s_converged")]  # fmt: skip
    flags = ["flag_unscreened", "flag_clip_low", "flag_clip_high", "flag_clip", "model_s_converged", "index_extrapolated"]  # fmt: skip
    flags += [*GATING_CHECKS, "check_names", "names_within_2pct", "wing_binds", "index_error_above_gate", "flag_index_moment", "clip_larger_side"]  # fmt: skip
    d[flags] = d[flags].astype(object)
    d.loc[~priced, numeric] = np.nan
    d["priced"] = priced.to_numpy()
    return d


def production_old_defaults(p_d: float, p_d_se: float) -> dict[str, Any] | None:
    """Today's row of the stopped production pass at the old defaults (``lcm_3m_norepair.parquet``),
    for check (b): LC/CC, LC/copula and CC/copula with their Monte Carlo errors, the budget read
    from the row.  ``None`` when the file or the row is missing, or priced against another entry."""
    if not PRODUCTION_OLD.exists():
        return None
    rows = pd.read_parquet(PRODUCTION_OLD)
    rows = rows[(rows["date"] == TODAY) & (rows["status"] != "failed") & rows["ED_lc"].notna()]
    if len(rows) != 1:
        return None
    r = rows.iloc[0]
    if abs(float(r["P_D_copula"]) - p_d) > 1e-12 * p_d:
        LOG.warning(
            "%s: the row of %s is priced against another entry; left out",
            PRODUCTION_OLD.name,
            TODAY,
        )
        return None
    sizes = (float(r["n_particles"]), float(r["n_paths"]), float(r["companion_paths"]))
    budget = pc.BUDGETS["production"] if sizes == (8e5, 8e5, 4e5) else f"{sizes[0]:g} particles / {sizes[1]:g} paths (constant-correlation fit on {sizes[2]:g} paths)"  # fmt: skip
    lc, lc_se, cc, cc_se = (float(r[c]) for c in ("ED_lc", "ED_lc_se", "ED_cc", "ED_cc_se"))
    values = {
        "lc_over_cc": (float(r["ratio"]), float(r["ratio_se"])),
        "lc_over_copula": (lc / p_d, pc.ratio_se(lc, lc_se, p_d, p_d_se)),
        "cc_over_copula": (cc / p_d, pc.ratio_se(cc, cc_se, p_d, p_d_se)),
    }
    return {
        "values": values,
        "budget": budget,
        "commit": str(r["git_commit"]),
        "status": str(r["status"]),
    }


def production_new_defaults(p_d: float, p_d_se: float) -> dict[str, Any] | None:
    """Today's production row at the new defaults (``rows/3m_production/<today>.json``, section
    A's row), for check (b), in the format of :func:`production_old_defaults`.  ``None`` when the
    file is missing, the row is not priced, or it is priced against another entry."""
    if not PRODUCTION_ROW.exists():
        return None
    r = json.loads(PRODUCTION_ROW.read_text())
    if r.get("date") != TODAY or r.get("status") == "failed" or r.get("ED_lc") is None:
        return None
    if abs(float(r["P_D_copula"]) - p_d) > 1e-12 * p_d:
        LOG.warning("%s: priced against another entry; left out", PRODUCTION_ROW.name)
        return None
    sizes = (float(r["n_particles"]), float(r["n_paths"]), float(r["companion_paths"]))
    budget = pc.BUDGETS["production"] if sizes == (8e5, 8e5, 4e5) else f"{sizes[0]:g} particles / {sizes[1]:g} paths (constant-correlation fit on {sizes[2]:g} paths)"  # fmt: skip
    lc, lc_se, cc, cc_se = (float(r[c]) for c in ("ED_lc", "ED_lc_se", "ED_cc", "ED_cc_se"))
    values = {
        "lc_over_cc": (float(r["ratio"]), float(r["ratio_se"])),
        "lc_over_copula": (lc / p_d, pc.ratio_se(lc, lc_se, p_d, p_d_se)),
        "cc_over_copula": (cc / p_d, pc.ratio_se(cc, cc_se, p_d, p_d_se)),
    }
    failing = [k for k in GATING_CHECKS if not r[k]]
    return {
        "values": values,
        "budget": budget,
        "commit": str(r["git_commit"]),
        "status": "check" if failing else "ok",
        "status_stored": str(r["status"]),
    }


def development_row(p_d: float, p_d_se: float) -> dict[str, Any] | None:
    """Today's row of the frozen part's development pass (``DEVELOPMENT.rows``), for check (b) of
    another pass, in the format of :func:`production_new_defaults` (the status under decision 3
    recomputed from the gates, as the frozen part prints it).  ``None`` when the file or the row
    is missing, not priced, or priced against another entry."""
    if not DEVELOPMENT.rows.exists():
        return None
    rows = pd.read_parquet(DEVELOPMENT.rows)
    rows = rows[(rows["date"] == TODAY) & (rows["status"] != "failed") & rows["ED_lc"].notna()]
    if len(rows) != 1:
        return None
    r = rows.iloc[0]
    if abs(float(r["P_D_copula"]) - p_d) > 1e-12 * p_d:
        LOG.warning("%s: the row of %s is priced against another entry; left out", DEVELOPMENT.rows.name, TODAY)  # fmt: skip
        return None
    sizes = (float(r["n_particles"]), float(r["n_paths"]), float(r["companion_paths"]))
    budget = pc.BUDGETS["development"] if sizes == DEVELOPMENT.sizes else f"{sizes[0]:g} particles / {sizes[1]:g} paths (constant-correlation fit on {sizes[2]:g} paths)"  # fmt: skip
    lc, lc_se, cc, cc_se = (float(r[c]) for c in ("ED_lc", "ED_lc_se", "ED_cc", "ED_cc_se"))
    values = {
        "lc_over_cc": (float(r["ratio"]), float(r["ratio_se"])),
        "lc_over_copula": (lc / p_d, pc.ratio_se(lc, lc_se, p_d, p_d_se)),
        "cc_over_copula": (cc / p_d, pc.ratio_se(cc, cc_se, p_d, p_d_se)),
    }
    failing = [k for k in GATING_CHECKS if not r[k]]
    return {
        "values": values,
        "budget": budget,
        "commit": str(r["git_commit"]),
        "status": "check" if failing else "ok",
        "status_stored": str(r["status"]),
        "reason_stored": str(r["reason"] or ""),
    }


def same_as_section_a(t: Any) -> bool | None:
    """Whether today's row of the table read is section A's production row
    (``PRODUCTION_ROW``): the same ``ED_lc``, ``ED_cc``, their standard errors, the paired ratio
    and its error, and the commit.  ``None`` when the row file is not on disk."""
    if not PRODUCTION_ROW.exists():
        return None
    r = json.loads(PRODUCTION_ROW.read_text())
    pairs = (("ED_lc", "ED_lc"), ("ED_lc_se", "ED_lc_se"), ("ED_cc", "ED_cc"), ("ED_cc_se", "ED_cc_se"), ("lc_over_cc", "ratio"), ("lc_over_cc_se", "ratio_se"))  # fmt: skip
    same = all(r.get(theirs) is not None and float(t[ours]) == float(r[theirs]) for ours, theirs in pairs)  # fmt: skip
    return same and str(t["git_commit"]) == str(r.get("git_commit"))


# ----------------------------------------------------------------------------- samples and summaries
def screen_drops(d: pd.DataFrame) -> tuple[pd.Series, str, str]:
    """The index slices dropped by the quote screen on each date, the column of the per-date
    table that holds them, as the part writes it, and the note a definition adds to it.  In the
    rows of a pass without decision 5 that is ``n_dropped_index``; with decision 5 the rows'
    ``n_dropped_index`` also counts the slices its calendar repair drops, and the screen's are
    ``DROPPED_SCREEN`` (the same quantity as before: no statistic changes its definition)."""
    if DROPPED_SCREEN not in d.columns:
        return d["n_dropped_index"].astype(float), "`n_dropped_index`", ""
    note = f"; `{DROPPED_SCREEN}` = `n_dropped_index` − `{DROPPED_CALENDAR}`: the row's count without the slices dropped by the calendar repair of the DJX target (decision 5)"
    return d[DROPPED_SCREEN].astype(float), f"`{DROPPED_SCREEN}`", note


def samples(d: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """The samples (boolean masks over the per-date table) and the table that describes them.
    The terciles of the clipped mass are those of ``clip_inner_max`` on all priced dates (low:
    at or below the 1/3 quantile; high: above the 2/3 quantile); the samples without the
    flagged dates keep the same bounds."""
    priced = d["priced"].astype(bool)
    clip = d["clip_inner_max"].astype(float)
    lo, hi = (float(x) for x in clip[priced].quantile([1.0 / 3.0, 2.0 / 3.0]))
    year = pd.to_datetime(d["date"]).dt.year
    flagged = priced & truth(d["flag_unscreened"])
    converged = priced & truth(d["model_s_converged"])
    base = {
        "all": (priced, "status not failed", np.nan, np.nan),
        "h1": (priced & (year <= 2016), "entry date in 2007–2016", np.nan, np.nan),
        "h2": (priced & (year >= 2017), "entry date in 2017–2026", np.nan, np.nan),
        "clip_low": (priced & (clip <= lo), f"clip_inner_max at or below its 1/3 quantile on all priced dates ({CLIP_SHORT})", float(clip[priced].min()), lo),
        "clip_mid": (priced & (clip > lo) & (clip <= hi), f"clip_inner_max above the 1/3 and at or below the 2/3 quantile ({CLIP_SHORT})", lo, hi),
        "clip_high": (priced & (clip > hi), f"clip_inner_max above its 2/3 quantile ({CLIP_SHORT})", hi, float(clip[priced].max())),
        "S": (converged, "priced and model S converged on the date", np.nan, np.nan),
    }  # fmt: skip
    masks: dict[str, pd.Series] = {}
    rows = []
    for key, (mask, text, low, high) in base.items():
        masks[key] = mask
        masks[f"{key}_unflagged"] = mask & ~flagged
        for name, suffix in (
            (key, ""),
            (f"{key}_unflagged", "; without the dates on which a name is kept unscreened"),
        ):
            rows.append({"sample": name, "n": int(masks[name].sum()), "definition": text + suffix, "clip_inner_max_from": low, "clip_inner_max_to": high})  # fmt: skip
    rel = d["EV_basket_part_rel_MB"].astype(float).abs()
    dropped, col, col_note = screen_drops(d)
    extra = {
        "all_index10": (priced & ~(rel > INDEX_MOMENT_FLAG), f"priced and |basket part / M_B^listed| at or below {INDEX_MOMENT_FLAG:g} (not `flag_index_moment`)"),
        "all_index15": (priced & ~(rel > INDEX_MOMENT_LIST), f"priced and |basket part / M_B^listed| at or below {INDEX_MOMENT_LIST:g}"),
        "all_names2pct": (priced & truth(d["names_within_2pct"]), "priced and the names' diagnostic inside 2 % (`names_within_2pct`)"),
        "S_dropidx_lt4": (converged & (dropped < 4), f"priced, model S converged and fewer than 4 index slices dropped by the screen ({col} < 4{col_note})"),
        "S_dropidx_0": (converged & (dropped == 0), f"priced, model S converged and no index slice dropped by the screen ({col} = 0{col_note})"),
    }  # fmt: skip
    if tuple(extra) != EXTRA_SAMPLES:
        raise ValueError("the extra samples are not the declared ones")
    for name, (mask, text) in extra.items():
        masks[name] = mask
        rows.append({"sample": name, "n": int(mask.sum()), "definition": text, "clip_inner_max_from": np.nan, "clip_inner_max_to": np.nan})  # fmt: skip
    tercile = np.select(
        [masks["clip_low"], masks["clip_mid"], masks["clip_high"]],
        ["low", "mid", "high"],
        default="",
    )
    d["clip_tercile"] = tercile
    return masks, pd.DataFrame(rows)


def bartlett_long_run(u: np.ndarray, lags: int = NW_LAGS) -> np.ndarray:
    """``Σ_t u_t u_t' + Σ_{j=1..L} (1 − j/(L+1))·Σ_t (u_t u_{t−j}' + u_{t−j} u_t')`` for the rows
    ``u_t`` of ``u`` in their order (Newey-West with the Bartlett kernel; ``L`` = ``lags``, cut at
    ``n − 1``)."""
    u = np.asarray(u, dtype=float).reshape(len(u), -1)
    out = u.T @ u
    for j in range(1, min(lags, len(u) - 1) + 1):
        g = u[j:].T @ u[:-j]
        out = out + (1.0 - j / (lags + 1.0)) * (g + g.T)
    return out


def newey_west_se(v: Any, lags: int = NW_LAGS) -> float:
    """Newey-West standard error of the mean of ``v`` (in its order): ``√(Ω/n²)``, ``Ω`` the
    Bartlett long-run sum of the deviations from the mean (no small-sample factor)."""
    x = np.asarray(v, dtype=float)
    if len(x) < 2:
        return float("nan")
    return float(math.sqrt(max(float(bartlett_long_run(x - x.mean(), lags)[0, 0]), 0.0)) / len(x))


def lag1_autocorr(v: Any) -> float:
    """``Σ e_t e_{t−1} / Σ e_t²`` of the deviations from the mean, in the order of ``v``."""
    x = np.asarray(v, dtype=float)
    e = x - x.mean() if len(x) else x
    total = float(e @ e)
    return float(e[1:] @ e[:-1]) / total if len(x) > 2 and total > 0.0 else float("nan")


def summarise(d: pd.DataFrame, mask: pd.Series, q: Quantity) -> dict[str, dict[str, float]]:
    """``statistic -> {value, se, mc_se, n}`` of one quantity on one sample (module docstring);
    the mean also carries ``nw_se`` and ``ac1`` (the dates of the sample in their order)."""
    sub = d.loc[mask]
    v = sub[q.key].astype(float)
    ok = v.notna()
    v = v[ok]
    n = len(v)
    out: dict[str, dict[str, float]] = {}
    if n == 0:
        return out
    mc = np.nan
    if q.se is not None:
        s = sub.loc[ok, q.se].astype(float)
        mc = float(s.mean()) if s.notna().all() else np.nan
    sem = float(v.std(ddof=1) / math.sqrt(n)) if n > 1 else np.nan
    out["mean"] = {"value": float(v.mean()), "se": sem, "mc_se": mc, "n": n, "nw_se": newey_west_se(v.to_numpy()), "ac1": lag1_autocorr(v.to_numpy())}  # fmt: skip
    for lags in NW_MORE_LAGS:
        out["mean"][f"nw_se_{lags}"] = newey_west_se(v.to_numpy(), lags)
    quart = v.quantile([0.25, 0.5, 0.75]).to_numpy()
    for name, value in (("q25", quart[0]), ("median", quart[1]), ("q75", quart[2]), ("min", v.min()), ("max", v.max())):  # fmt: skip
        out[name] = {"value": float(value), "se": np.nan, "mc_se": np.nan, "n": n}
    if q.num is None or q.den is None:
        return out
    a, b = sub.loc[ok, q.num].astype(float), sub.loc[ok, q.den].astype(float)
    if a.isna().any() or b.isna().any():
        raise ValueError(f"{q.key}: a pooled term is missing on a date where the ratio is not")
    total = float(b.sum())
    pooled = float(a.sum()) / total
    resid = a - pooled * b
    se_dates = math.sqrt(n / (n - 1) * float((resid**2).sum())) / abs(total) if n > 1 else np.nan
    mc = np.nan
    if q.num_se is not None:
        sa = sub.loc[ok, q.num_se].astype(float)
        sb = sub.loc[ok, q.den_se].astype(float) if q.den_se is not None else 0.0 * sa
        if q.paired:
            # the covariance of the paired estimates, from the row's standard error of their ratio
            r = a / b
            cov = (sa**2 + r**2 * sb**2 - (sub.loc[ok, q.se].astype(float) * b) ** 2) / (2.0 * r)
            var = sa**2 + pooled**2 * sb**2 - 2.0 * pooled * cov
        else:
            var = sa**2 + pooled**2 * sb**2
        mc = float(np.sqrt(var.clip(lower=0.0)).sum()) / abs(total) if var.notna().all() else np.nan
    out["pooled"] = {"value": pooled, "se": se_dates, "mc_se": mc, "n": n}
    return out


SE_KIND = {
    "mean": "across dates: sd/√n",
    "pooled": "across dates: √(n/(n−1)·Σ(a_i − R·b_i)²)/Σb",
}


def all_summaries(
    d: pd.DataFrame, masks: dict[str, pd.Series], qs: list[Quantity]
) -> tuple[dict[tuple[str, str], dict[str, dict[str, float]]], pd.DataFrame]:
    table: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    rows = []
    for q in qs:
        for name, mask in masks.items():
            stats = summarise(d, mask, q)
            table[(q.key, name)] = stats
            for stat, cell in stats.items():
                rows.append({"quantity": q.key, "label": q.label, "sample": name, "statistic": stat, "value": cell["value"], "se": cell["se"], "se_kind": SE_KIND.get(stat, ""), "mc_se": cell["mc_se"], "nw_se": cell.get("nw_se", np.nan), "lag1_autocorr": cell.get("ac1", np.nan), "n": cell["n"], **{f"nw_se_{lags}": cell.get(f"nw_se_{lags}", np.nan) for lags in NW_MORE_LAGS}})  # fmt: skip
    return table, pd.DataFrame(rows)


# ----------------------------------------------------------------------------- check (d)
def ols(y: Any, x: Any, names: list[str]) -> dict[str, Any]:
    """Ordinary least squares of ``y`` on an intercept and the columns of ``x`` (QR): the
    coefficients, the classical standard errors ``s²(X'X)⁻¹`` with ``s² = e'e/(n − k)``, the
    HC1 ones ``n/(n − k)·(X'X)⁻¹[Σ e_i² x_i x_i'](X'X)⁻¹``, the Newey-West ones
    ``n/(n − k)·(X'X)⁻¹ Ω (X'X)⁻¹`` (``Ω`` the Bartlett long-run sum of ``e_i x_i`` over ``NW_LAGS``
    lags, the rows in their order; the same factor as HC1, to which it reduces with no lag), their
    t ratios, the centred R² and the lag-1 autocorrelation of the residuals."""
    yv = np.asarray(y, dtype=float).ravel()
    xv = np.asarray(x, dtype=float).reshape(len(yv), -1)
    if not (np.isfinite(yv).all() and np.isfinite(xv).all()):
        raise ValueError("non-finite value in the regression sample")
    n = len(yv)
    design = np.column_stack([np.ones(n), xv])
    k = design.shape[1]
    if n <= k:
        raise ValueError("not enough dates for the regression")
    qm, rm = np.linalg.qr(design)
    coef = np.linalg.solve(rm, qm.T @ yv)
    r_inv = np.linalg.inv(rm)
    xtx_inv = r_inv @ r_inv.T
    resid = yv - design @ coef
    rss = float(resid @ resid)
    tss = float(((yv - yv.mean()) ** 2).sum())
    cov = rss / (n - k) * xtx_inv
    xe = design * resid[:, None]
    cov_hc1 = n / (n - k) * xtx_inv @ (xe.T @ xe) @ xtx_inv
    se, se_hc1 = np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov_hc1))
    se_nw = np.sqrt(np.diag(n / (n - k) * xtx_inv @ bartlett_long_run(xe) @ xtx_inv))
    r2 = 1.0 - rss / tss
    return {
        "names": ["const", *names], "coef": coef, "se": se, "se_hc1": se_hc1, "t": coef / se, "t_hc1": coef / se_hc1,
        "se_nw": se_nw, "t_nw": coef / se_nw, "resid_ac1": float(resid[1:] @ resid[:-1]) / rss,
        "r2": r2, "r2_adj": 1.0 - (1.0 - r2) * (n - 1) / (n - k), "n": n,
    }  # fmt: skip


FITS = (
    ("rel", "clipped mass + basket part / M_B^listed", ["clip_inner_max", "EV_basket_part_rel_MB"]),
    ("raw", "clipped mass + basket part (raw)", ["clip_inner_max", "EV_basket_part"]),
    ("clip", "clipped mass alone", ["clip_inner_max"]),
    ("bp_rel", "basket part / M_B^listed alone", ["EV_basket_part_rel_MB"]),
    ("bp_raw", "basket part (raw) alone", ["EV_basket_part"]),
)


#: The two legs of the regressand of check (d), each on the clipped mass alone (sample ``S``): the
#: key of the fit, the column regressed and the label of the fit.
LEG_FITS = (
    ("leg_lc_over_cc", "lc_over_cc", "regressand LC/CC: clipped mass alone"),
    ("leg_s_over_copula", "s_over_copula", "regressand S/copula: clipped mass alone"),
)


def regressions(
    d: pd.DataFrame, masks: dict[str, pd.Series]
) -> tuple[dict[tuple[str, str], dict[str, Any]], pd.DataFrame]:
    """The fits of check (d) (regressand ``y_check_d``; ``FITS``) and, on the sample ``S``, the
    fit of each leg of the regressand on the clipped mass alone (``LEG_FITS``)."""
    fits: dict[tuple[str, str], dict[str, Any]] = {}
    rows = []

    def keep(sample: str, key: str, label: str, fit: dict[str, Any]) -> None:
        fits[(sample, key)] = fit
        for j, term in enumerate(fit["names"]):
            rows.append({
                "sample": sample, "fit": key, "fit_label": label, "term": term, "coef": fit["coef"][j], "se": fit["se"][j], "t": fit["t"][j],
                "se_hc1": fit["se_hc1"][j], "t_hc1": fit["t_hc1"][j], "r2": fit["r2"], "r2_adj": fit["r2_adj"], "n": fit["n"],
                "se_nw": fit["se_nw"][j], "t_nw": fit["t_nw"][j], "resid_lag1_autocorr": fit["resid_ac1"],
            })  # fmt: skip

    for sample in ("S", "S_unflagged", "S_dropidx_lt4", "S_dropidx_0"):
        sub = d.loc[masks[sample]]
        for key, label, regs in FITS:
            if sample.startswith("S_dropidx") and key != "rel":
                continue
            keep(sample, key, label, ols(sub["y_check_d"], sub[regs], regs))
    sub = d.loc[masks["S"]]
    for key, column, label in LEG_FITS:
        keep("S", key, label, ols(sub[column], sub[["clip_inner_max"]], ["clip_inner_max"]))
    return fits, pd.DataFrame(rows)


def compare_check_d(
    fits: dict[tuple[str, str], dict[str, Any]], rows_name: str
) -> tuple[str, dict[str, float]]:
    """One sentence, and its counts: the fits against the independent check's JSON, to its 6
    significant digits."""
    path = check_d_path()
    if path is None:
        return "The independent check's results file is not on disk: not compared in this run.", {}
    results = json.loads(path.read_text())
    block = next(
        (b for b in results.values() if isinstance(b, dict) and b.get("table") == rows_name), None
    )
    if block is None:
        return (
            f"The independent check has no block for `{rows_name}`: not compared in this run.",
            {},
        )
    pairs = [
        (("S", "raw"), block["main"]["raw"], ("coef", "se", "se_hc1", "t", "t_hc1")),
        (("S", "rel"), block["main"]["/ M_B_listed"], ("coef", "se", "se_hc1", "t", "t_hc1")),
        (
            ("S_unflagged", "raw"),
            block["robust"]["raw | without unscreened names"],
            ("coef", "se_hc1"),
        ),
    ]
    total, agree, worst, off = 0, 0, 0.0, []
    for key, theirs, fields in pairs:
        mine = fits[key]
        cells = [
            (f"{key[0]}.{key[1]}.{f}[{j}]", mine[f][j], theirs[f][j])
            for f in fields
            for j in range(len(mine["coef"]))
        ]
        cells += [
            (f"{key[0]}.{key[1]}.r2", mine["r2"], theirs["r2"]),
            (f"{key[0]}.{key[1]}.n", mine["n"], theirs["n"]),
        ]
        for name, a, b in cells:
            total += 1
            same = float(f"{float(a):.6g}") == float(b)
            agree += same
            worst = max(worst, abs(float(a) - float(b)) / max(abs(float(b)), 1e-300))
            if not same:
                off.append(name)
    where = (
        f"`{check_d_source(path)}`, the copy kept in the package of the checking session's results file"
        if path == CHECK_D_PACKAGE
        else f"`{path}`, a session scratch file that is not part of the package and cannot be audited from it later"
    )
    text = (
        f"Against the independent check ({where}; its own `ols`; the two-regressor fits raw and relative on all dates and the raw one without the flagged dates): "
        f"{agree} of {total} numbers (coefficients, classical and HC1 standard errors, t, R², n) agree to the 6 significant digits it prints; largest relative difference {worst:.1e}."
    )
    if off:
        text += f" Not to the digit: {', '.join(off)}."
    return text, {"compared": total, "agree": agree, "worst_relative_difference": worst}


def check_d_path() -> Path | None:
    """The independent check's results file: the copy in the package when it exists, else the
    session scratch file, else ``None``."""
    for path in (CHECK_D_PACKAGE, CHECK_D_SCRATCH):
        if path.exists():
            return path
    return None


def check_d_source(path: Path) -> str:
    """The path of the independent check's file as a record cites it."""
    if path == CHECK_D_PACKAGE:
        return f"outputs/dispersion_lc/pm_update/{path.relative_to(pc.PM)}"
    return str(path)


def failure_clause(date: str, reason: str, logs: Path = RUN_LOGS) -> str:
    """One clause for a failed date: the step and the leg that failed, read from the reason and
    from the traceback in the run's log of the date under ``logs`` (the call of
    ``lc_spec_from_smiles`` in which the date raised: the names are fitted one by one first, the
    index after them)."""
    log = logs / f"{date}.log"
    lines = log.read_text().splitlines() if log.exists() else []
    call = ""
    for i, line in enumerate(lines[:-1]):
        if "in lc_spec_from_smiles" in line:
            call = lines[i + 1].strip()
    if "no listed expiry passes the screen" in reason:
        who = reason.split(":")[1].strip() if reason.count(":") > 1 else "the index"
        leg = "the index leg" if who == "the index" else f"the names' leg ({who})"
        what = "the DJX" if who == "the index" else who
        return f"{leg}, at the quote screen of the expiries: no listed expiry of {what} passes it, so there is no smile to fit; nothing is calibrated or priced"
    if "Initial guess is outside of provided bounds" in reason:
        if not call:
            return "an SVI slice fit (`fit_svi_slice`): the start value of the bounded least squares is outside its bounds; the run's log of the date is not on disk, so the leg is not identified"
        leg = "the index leg" if call.startswith("index_") else "the names' leg"
        whose = "the index's" if call.startswith("index_") else "one name's"
        if "repair_calendar" in call:
            step = f"in the calendar repair of {whose} slices (decision 1)"
        elif "_surface_config" in call:
            step = f"in the fit of {whose} SVI surface"
        else:
            step = f"in `{call}`"
        tail = (
            ""
            if call.startswith("index_")
            else "; the log does not say which name; the index leg is not reached"
        )
        return f"{leg}, {step}: the SVI fit of a slice (`fit_svi_slice`) stops because the start value of its bounded least squares is outside the bounds{tail}; nothing is calibrated or priced"
    return "step and leg not identified from the reason"


# ----------------------------------------------------------------------------- the part
class Book:
    """Formats a number for the Markdown and files its record at the same time, so that every
    number of the part has one (an id asked twice must carry the same value)."""

    def __init__(self, commit: str, source: str, budget: str = DEVELOPMENT.budget) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.commit = commit
        self.source = source
        self.budget = pc.BUDGETS[budget]

    def num(
        self, id: str, quantity: str, value: float | None, se: float | None = None, *, digits: int = 4, spec: str | None = None,
        show_se: bool = True, study: bool = False, **kwargs: Any,
    ) -> str:  # fmt: skip
        kwargs.setdefault("budget", pc.BUDGETS["study"] if study else self.budget)
        kwargs.setdefault("commit", STUDY_COMMIT if study else self.commit)
        kwargs.setdefault("source", self.source)
        if study and not kwargs.get("notes"):
            kwargs["notes"] = STUDY_NOTE
        if se is None and not kwargs.get("notes"):
            kwargs["notes"] = (
                "no standard error: a count, a statistic of a fit or a stated value, not a Monte Carlo estimate"
            )
        rec = pc.record(id, SECTION, quantity, value, se, **kwargs)
        old = self.records.get(id)
        if old is not None and (old["value"], old["se"]) != (rec["value"], rec["se"]):
            raise ValueError(f"record {id} asked twice with different values")
        self.records[id] = rec
        if rec["value"] is None:
            return "n/a"
        if spec is not None:
            return format(rec["value"], spec)
        return pc.pm(rec["value"], rec["se"] if show_se else None, digits)


def md_table(headers: list[str], rows: list[list[str]], align: str = "") -> str:
    """A Markdown table; ``align`` gives ``l`` or ``r`` per column (default: first left, rest right)."""
    align = align or "l" + "r" * (len(headers) - 1)
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(":--" if a == "l" else "--:" for a in align) + "|",
    ]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def build_markdown(
    d: pd.DataFrame, masks: dict[str, pd.Series], sample_table: pd.DataFrame, qs: list[Quantity],
    table: dict[tuple[str, str], dict[str, dict[str, float]]], fits: dict[tuple[str, str], dict[str, Any]], agreement: tuple[str, dict[str, float]],
    book: Book, rows_path: Path, run: Pass = DEVELOPMENT, logs: Path | None = None,
) -> str:  # fmt: skip
    """The Markdown of the part; every number it prints is filed in ``book``.  ``run`` is the pass
    behind the table (its labels) and ``logs`` the folder of its logs (default: the pass's)."""
    logs = run.logs if logs is None else logs
    priced = d["priced"].astype(bool)
    md_qs = [q for q in qs if q.in_md]
    n_of = dict(zip(sample_table["sample"], sample_table["n"], strict=True))
    sample_def = dict(zip(sample_table["sample"], sample_table["definition"], strict=True))

    def in_sample(sample: str) -> str:
        """What a sample is, for the definition of a record (the records name it by its key)."""
        return f" Sample `{sample}`: {sample_def[sample]}."

    def unit_of(q: Quantity) -> str:
        return CLIP_UNIT if q.key.startswith("clip_") else ""

    src = f"outputs/dispersion_lc/{rows_path.name} + outputs/dispersion/entries_3m.parquet (B1) + outputs/dispersion/model_s_3m.parquet → tables/C_history_summaries.csv"

    def stat_cell(q: Quantity, sample: str, stat: str, with_se: bool = True) -> str:
        cell = table[(q.key, sample)].get(stat)
        if cell is None:
            return "n/a"
        notes = STUDY_NOTE if q.study else ""
        if stat in ("mean", "pooled"):
            kind = (
                "mean over the dates of the sample; se across dates (sd/√n)"
                if stat == "mean"
                else "pooled: Σ numerator / Σ denominator over the dates of the sample; se across dates (linearised)"
            )
            if np.isfinite(cell["mc_se"]):
                notes = f"Monte Carlo standard error with the dates fixed, upper bound (seeds shared across dates): {cell['mc_se']:.2g}"
            definition = f"{q.definition}. {kind}.{in_sample(sample)}"
        else:
            definition = f"{q.definition}. {stat} over the dates of the sample.{in_sample(sample)}"
            notes = (
                notes + "; " if notes else ""
            ) + "an order statistic across dates: no standard error given"
        return book.num(
            f"C.hist.{q.key}.{stat}.{sample}", f"{q.label}: {stat}, {sample}", cell["value"], cell["se"] if stat in ("mean", "pooled") else None,
            digits=q.digits, show_se=with_se, study=q.study, definition=definition, n=cell["n"], notes=notes, source=src, unit=unit_of(q),
        )  # fmt: skip

    def n_cell(sample: str) -> str:
        return book.num(f"C.hist.n_dates.{sample}", f"dates in the sample {sample}", n_of[sample], spec=".0f", definition=str(sample_table.set_index("sample").loc[sample, "definition"]), n=int(n_of[sample]), unit="dates", source=src)  # fmt: skip

    def nw_cell(q: Quantity, sample: str, more: bool = False) -> str:
        """The Newey-West error of the mean at ``NW_LAGS`` lags (with ``more``: also at
        ``NW_MORE_LAGS``, after a slash each) and the lag-1 autocorrelation in brackets."""
        cell = table[(q.key, sample)].get("mean")
        if cell is None or not np.isfinite(cell["nw_se"]):
            return ""
        base = f"{q.definition}. Over the dates of the sample in their order"
        longer = ", ".join(f"C.hist.{q.key}.mean_nw_se_{lags}.{sample}" for lags in NW_MORE_LAGS)
        nw = book.num(
            f"C.hist.{q.key}.mean_nw_se.{sample}", f"{q.label}: Newey-West standard error of the mean, {sample}", cell["nw_se"], spec=f".{q.digits}f", study=q.study,
            definition=f"{base}: the Newey-West standard error of the mean (Bartlett kernel, {NW_LAGS} lags counted in consecutive dates of the sample, no small-sample factor).{in_sample(sample)}",
            n=cell["n"], source=src,
            notes=f"a standard error of the mean across dates that allows for serial correlation up to {NW_LAGS} rows of the sample: a lower value, the error still grows with the lag ({' and '.join(str(x) for x in NW_MORE_LAGS)} lags: {longer} where printed, and the summaries CSV); the lags are counted in rows of the sample, which has gaps; it has no standard error of its own",
        )  # fmt: skip
        if more:
            for lags in NW_MORE_LAGS:
                nw += " / " + book.num(
                    f"C.hist.{q.key}.mean_nw_se_{lags}.{sample}", f"{q.label}: Newey-West standard error of the mean at {lags} lags, {sample}", cell[f"nw_se_{lags}"], spec=f".{q.digits}f", study=q.study,
                    definition=f"{base}: the Newey-West standard error of the mean (Bartlett kernel, {lags} lags counted in consecutive rows of the sample, which has gaps; no small-sample factor).{in_sample(sample)}",
                    n=cell["n"], source=src, notes="a standard error of the mean across dates that allows for serial correlation: it has no standard error of its own",
                )  # fmt: skip
        ac = book.num(
            f"C.hist.{q.key}.mean_lag1_autocorr.{sample}", f"{q.label}: lag-1 autocorrelation across dates, {sample}", cell["ac1"], spec="+.2f", study=q.study,
            definition=f"{base}: Σ e_t e_(t−1) / Σ e_t², e the deviation from the mean of the sample.{in_sample(sample)}", n=cell["n"], source=src, notes="a statistic of the series of dates: no standard error given",
        )  # fmt: skip
        return f"± {nw} ({ac})"

    def mc_cell(q: Quantity, sample: str, stat: str) -> str:
        cell = table[(q.key, sample)].get(stat)
        if cell is None or not np.isfinite(cell["mc_se"]):
            return ""
        how = (
            "the mean of the per-date Monte Carlo errors"
            if stat == "mean"
            else "Σ sd(a_i − R·b_i) / Σb"
        )
        return book.num(
            f"C.hist.{q.key}.{stat}_mc_bound.{sample}", f"{q.label}: Monte Carlo bound of the {stat}, {sample}", cell["mc_se"], spec=f".{q.digits}f",
            definition=f"{q.definition}. The Monte Carlo standard error of the {stat} with the dates fixed, an upper bound: the dates share the particle and pricing seeds, so the per-date errors are added linearly ({how}).{in_sample(sample)}",
            n=cell["n"], source=src, notes="a bound on a standard error: it has no standard error of its own",
        )  # fmt: skip

    def n_s_cell(sample: str) -> str:
        n_s = int(table[("s_over_copula", sample)].get("mean", {}).get("n", 0))
        return book.num(
            f"C.hist.n_dates_model_s.{sample}", f"dates of the model S rows in the sample {sample}", n_s, spec=".0f", unit="dates",
            definition="the dates of the sample on which model S converged: the n of the rows S/copula, κ model S, E[V]/EQV model S, the calls S/copula and LC/CC − S/copula",
            n=n_s, source=src, notes="a count of dates",
        )  # fmt: skip

    def rel_cell(r: Any) -> str:
        cell = book.num(
            f"C.hist.index_moment.{r['date']}", "basket part / M_B^listed on the date", r["EV_basket_part_rel_MB"], r["EV_basket_part_rel_MB_se"], spec="+.1%", date=r["date"],
            definition=by_key["EV_basket_part_rel_MB"].definition, source="tables/C_history_by_date.csv",
        )  # fmt: skip
        return cell.replace("%", " %")

    def gap_cell(r: Any) -> str:
        cell = book.num(
            f"C.hist.names_gap.{r['date']}", "names' diagnostic on the date: Σw E_LC[R_i²] / Σw M_i^listed − 1", r["names_gap"], r["names_gap_se"], spec="+.2%", date=r["date"],
            definition="`sum_w_ER2_lc` / `sum_w_M` − 1 of the row (the quantity `check_names` compares with 2 %); se: `sum_w_ER2_lc_se` / `sum_w_M`", source="tables/C_history_by_date.csv",
        )  # fmt: skip
        return cell.replace("%", " %")

    def marks(r: Any) -> str:
        """Status and flags of a date, for the tables of extremes."""
        out = [f"`{r['status']}`"]
        if bool(r["flag_unscreened"]):
            out.append("flagged")
        if abs(float(r["EV_basket_part_rel_MB"])) > INDEX_MOMENT_FLAG:
            out.append(f"index second moment {rel_cell(r)} from the listed strip")
        if abs(float(r["names_gap"])) > NAMES_TOL:
            out.append(f"names' second moment {gap_cell(r)} from the listed strips")
        return ", ".join(out)

    commit = book.commit
    by_key = {q.key: q for q in qs}
    statuses = d["status"].value_counts().to_dict()
    stored = d["status_stored"].value_counts().to_dict()
    monthly_dates, s_converged = study_dates()
    n_model_s_rows = len(pd.read_parquet(MODEL_S, columns=["date"]))
    table_dates = set(d["date"])
    absent = [x for x in monthly_dates if x not in table_dates]
    failed_dates = d.loc[~priced, "date"].tolist()
    not_monthly = sorted(table_dates - set(monthly_dates))
    not_in_s = sorted(table_dates - s_converged)
    gate_fail = {k: d.loc[priced & ~truth(d[k]), "date"].tolist() for k in GATING_CHECKS}
    nan_only = [x for x in gate_fail["check_no_nan"] if x not in gate_fail["check_forward"] + gate_fail["check_index"]]  # fmt: skip
    nan_cols = sorted({c for x in d.loc[d["date"].isin(gate_fail["check_no_nan"]), "non_finite_columns"] for c in x.split(", ") if c})  # fmt: skip
    now_ok = int((priced & (d["status_stored"] == "check") & (d["status"] == "ok")).sum())
    names_out = priced & ~truth(d["names_within_2pct"])
    gap_abs = d["names_gap"].astype(float).abs()
    rel_abs = d["EV_basket_part_rel_MB"].astype(float).abs()
    idx10, idx15 = priced & (rel_abs > INDEX_MOMENT_FLAG), priced & (rel_abs > INDEX_MOMENT_LIST)
    # the status of a table written under decision 3 is the row's own: the recomputation agrees on
    # every date, and the labels say so; a table written before it keeps the recomputed status
    rows_own = bool((d["status_stored"] == d["status"]).all())
    how_status = "the row's own status, stored under decision 3" if rows_own else "status recomputed under decision 3"  # fmt: skip
    how_stored = "the current rule, decision 3" if rows_own else "the rule before decision 3"
    status_note = "the row's own, under decision 3" if rows_own else "recomputed under decision 3"
    drops, drop_col, _drop_note = screen_drops(d)
    with_d5 = DROPPED_CALENDAR in d.columns
    lines: list[str] = []
    add = lines.append
    add(f"## C. History, 3m, {run.title.format(commit=commit)}")
    add("")

    def count(tag: str, what: str, value: int) -> str:
        return book.num(f"C.hist.count.{tag}", what, value, spec=".0f", unit="dates", definition=what, n=int(value), source=f"{book.source} → tables/C_history_by_date.csv", notes="a count of dates")  # fmt: skip

    if run is not DEVELOPMENT:
        add(
            f"After the freeze of the package: this part is not in the frozen package. It is the frozen part `parts/{PART}.md` rebuilt by the same builder (`scripts/pm_history.py --budget {run.budget}`) on the {run.budget}-budget pass "
            f"`outputs/dispersion_lc/{rows_path.name}` (commit {commit}, {run.decisions}); the frozen part is at the {DEVELOPMENT.budget} budget (`{DEVELOPMENT.rows.name}`, {DEVELOPMENT.decisions}) and is not changed. "
            "Every number below is of this pass unless its row says otherwise; the tables and figures named below are those of this folder, not the frozen package's."
        )
        add("")
    add(
        f"Source: `outputs/dispersion_lc/{rows_path.name}` (one row per date; {count('rows', 'dates of the table', len(d))} dates {d['date'].min()} to {d['date'].max()}: "
        f"{count('priced', 'priced dates (status not failed)', int(priced.sum()))} priced — {count('ok', f'dates with status ok ({how_status})', statuses.get('ok', 0))} `ok`, "
        f"{count('check', f'dates with status check (priced, a gating check not passed; {how_status})', statuses.get('check', 0))} `check` (priced, a gating check not passed: the reason is in the per-date CSV) — "
        f"and {count('failed', 'dates with status failed', statuses.get('failed', 0))} failed), "
        f"joined on the date with `outputs/dispersion/entries_3m.parquet` (basket B1: one row per date, checked) and `outputs/dispersion/model_s_3m.parquet`. "
        f"Budget: {pc.BUDGETS[run.budget]}; commit of the rows {commit}; {run.repairs}. "
        f"The row of {TODAY} in this table is at the {run.budget} budget."
    )
    add("")
    if with_d5:
        cal = d[DROPPED_CALENDAR].astype(float)
        add(
            f"Decision 5 (the calendar repair of the DJX target: the names' rule on the DJX slices). `{DROPPED_CALENDAR}` of the rows is positive on "
            f"{count('dropped_calendar_index', f'priced dates on which the calendar repair of the DJX target drops an index slice ({DROPPED_CALENDAR} > 0)', int((priced & (cal > 0)).sum()))} of the {int(priced.sum())} priced dates "
            f"(one slice on {count('dropped_calendar_index_1', f'priced dates with {DROPPED_CALENDAR} = 1', int((priced & (cal == 1)).sum()))}, "
            f"more than one on {count('dropped_calendar_index_2plus', f'priced dates with {DROPPED_CALENDAR} > 1', int((priced & (cal > 1)).sum()))}); on the other priced dates the repair drops no DJX slice. "
            f"`n_dropped_index` of these rows counts the index slices dropped under any rule, these included. The slices dropped by the quote screen, which this part tabulates and conditions on (C.0, C.7c), "
            f"are {drop_col} = `n_dropped_index` − `{DROPPED_CALENDAR}` of the per-date CSV: the same quantity as `n_dropped_index` of the frozen part's table, which has no calendar repair of the index."
        )
        add("")
    today_row = d[d["date"] == TODAY]
    today_stored = "" if rows_own else f"stored: `{today_row['status_stored'].iloc[0]}`; "
    today_status = (
        f" The row of {TODAY} is `{today_row['status'].iloc[0]}` ({today_stored}names' gap {gap_cell(today_row.iloc[0])} against the 2 % threshold)."
        if len(today_row) == 1 and bool(today_row["priced"].iloc[0])
        else ""
    )
    stored_counts = f"({count('ok_stored', f'dates with stored status ok ({how_stored})', stored.get('ok', 0))} `ok`, {count('check_stored', f'dates with stored status check ({how_stored})', stored.get('check', 0))} `check`)"
    rule = "the current rule of `scripts/lcm_price.py` (`GATING_CHECKS`, `row_status`): a priced date is `check` when `check_no_nan`, `check_forward` or `check_index` fails and `ok` otherwise; a failed date stays failed. "

    def now_ok_s() -> str:  # filed where it is printed: the records keep the order of the text
        return count("check_stored_now_ok", "priced dates stored as check and ok under decision 3", now_ok)  # fmt: skip

    forward_dates = f" ({', '.join(gate_fail['check_forward'])})" if gate_fail["check_forward"] else ""  # fmt: skip
    add(
        (
            f"Status. The table was written at commit {commit}, under the owner's decision 3 of 2026-10-09 (the names' 2 % check is a reported diagnostic, not a gate): the status stored in its rows {stored_counts} follows {rule}"
            f"The status printed and counted in this part is the row's own: recomputed here from the three gates it is the same on every date (the column `status_stored` of the per-date CSV equals `status`; {now_ok_s()} of the dates stored as `check` are `ok` in the recomputation). "
            if rows_own
            else f"Status. The table was written at commit {commit}, before the owner's decision 3 of 2026-10-09 (the names' 2 % check is a reported diagnostic, not a gate): the status stored in its rows {stored_counts} follows the earlier rule. "
            f"The status printed and counted in this part is recomputed with {rule}"
            "The stored status is the column `status_stored` of the per-date CSV. "
        )
        + f"The {int(statuses.get('check', 0))} `check` dates: {count('check_no_nan_only', 'priced dates failing check_no_nan and no other gate', len(nan_only))} fail `check_no_nan` only ({', '.join(nan_only) or 'none'}): "
        f"the non-finite number of these rows is the diagnostic column {', '.join(f'`{c}`' for c in nan_cols) or 'none'}, which is not a price and is not used in this part; "
        f"{count('check_index_fail', 'priced dates failing the index gate check_index', len(gate_fail['check_index']))} fail the index gate `check_index` ({', '.join(gate_fail['check_index']) or 'none'}); "
        f"`check_forward` fails on {count('check_forward_fail', 'priced dates failing check_forward', len(gate_fail['check_forward']))}{forward_dates}. "
        + (
            ""
            if rows_own
            else f"{now_ok_s()} of the dates stored as `check` are `ok` under the current rule. "
        )
        + f"Separately, the names' diagnostic (Σw E_LC[R_i²] within 2 % of the listed strips, `check_names`) is outside 2 % on {count('names_outside_2pct', 'priced dates with the names diagnostic outside 2 %', int(names_out.sum()))} priced dates "
        "(column `names_within_2pct` of the per-date CSV; the gap is `names_gap`)." + today_status
    )
    add("")
    # what F1 leaves empty although the study has it: the listed-variance forward needs the entry
    # alone; model S has a number on its converged dates only
    entries = pd.read_parquet(ENTRIES, columns=["date", "basket", "EQV", "EV"])
    b1 = entries[entries["basket"] == "B1"].set_index("date")
    no_line = [*absent, *failed_dates]
    has_fwd = int(((b1.loc[no_line, "EQV"] > 0.0) & (b1.loc[no_line, "EV"] > 0.0)).sum())
    f1_empty = (
        f"the listed-variance forward is left empty although the study has it on {count('f1_empty_listed_fwd_in_study', 'failed dates and absent monthly dates on which the study has the listed-variance forward (EQV and EV of the B1 entry)', has_fwd)} of these {len(no_line)} dates (it needs the entry alone, not the LC run); "
        f"model S is left empty on the {len(failed_dates)} failed dates although it converged on "
        f"{count('failed_s_converged', 'failed dates of the table on which model S converged', len(set(failed_dates) & s_converged))} of them, "
        f"and has no converged number on the {len(absent)} absent dates (it converged on {len(set(absent) & s_converged)} of them)."
    )
    add(
        f"Dates. The table's {len(d)} dates are the {count('table_in_model_s', 'dates of the table on which model S converged (model_s_3m, converged)', len(table_dates & s_converged))} dates on which model S converged (all {count('model_s_converged_study', 'dates on which model S converged in model_s_3m', len(s_converged))} converged dates of `model_s_3m.parquet`, which has {count('model_s_rows', 'rows of model_s_3m.parquet (one per date, converged or not)', n_model_s_rows)} rows) plus {', '.join(not_in_s) or 'none'}; "
        f"they are not all of the study's monthly dates. Of the study's {count('study_monthly', 'monthly B1 dates of the study (entries_3m, monthly)', len(monthly_dates))} monthly B1 dates (`entries_3m.parquet`, `monthly`), "
        f"{count('study_monthly_in_table', 'monthly B1 dates of the study that have a row in the table', len(monthly_dates) - len(absent))} have a row in the table and "
        f"{count('study_monthly_absent', 'monthly B1 dates of the study with no row in the table', len(absent))} have none: {', '.join(absent) or 'none'} "
        f"(model S converged on {count('study_monthly_absent_s_converged', 'monthly B1 dates with no row in the table on which model S converged', len(set(absent) & s_converged))} of them). "
        f'"All priced dates" is therefore conditional on model S converging, except for {TODAY}, and the sample "∩ model S converged" differs from it by that date only. '
        f"In the table and not flagged monthly in the entries: {', '.join(not_monthly) or 'none'}. "
        f"Figure F1 draws nothing on the {len(absent)} monthly dates with no row and on the {int((~priced).sum())} failed dates: {f1_empty}"
    )
    add("")
    add(
        "Files: `tables/C_history_by_date.csv` (per date), `tables/C_history_summaries.csv` (long: quantity, sample, statistic, value, se, mc_se, nw_se, lag1_autocorr, n, nw_se_12, nw_se_24), "
        "`tables/C_history_samples.csv`, `tables/C_history_regression.csv`, `figures/F1_forward_over_copula.pdf/.csv`, `figures/F2_calls_over_copula_by_strike.pdf/.csv`."
    )
    add("")
    # ---------------------------------------------------------------- C.0 samples
    add("### C.0 Samples")
    add("")
    rows = []
    desc = sample_table.set_index("sample")
    for key, label in BASE_SAMPLES:
        rows.append(
            [label, n_cell(key), n_cell(f"{key}_unflagged"), str(desc.loc[key, "definition"])]
        )
    add(md_table(["sample", "n", "n without flagged dates", "definition"], rows, "lrrl"))
    add("")
    lo, hi = float(desc.loc["clip_low", "clip_inner_max_to"]), float(
        desc.loc["clip_high", "clip_inner_max_from"]
    )
    lo_s = book.num(
        "C.hist.clip_tercile_bound.low",
        "clipped mass inside ±2.5 sd: 1/3 quantile on all priced dates",
        lo,
        unit=CLIP_UNIT,
        definition=f"the 1/3 quantile of clip_inner_max over all priced dates (linear interpolation); clip_inner_max: {CLIP_DEF}",
        n=int(n_of["all"]),
        source=src,
    )
    hi_s = book.num(
        "C.hist.clip_tercile_bound.high",
        "clipped mass inside ±2.5 sd: 2/3 quantile on all priced dates",
        hi,
        unit=CLIP_UNIT,
        definition=f"the 2/3 quantile of clip_inner_max over all priced dates (linear interpolation); clip_inner_max: {CLIP_DEF}",
        n=int(n_of["all"]),
        source=src,
    )
    add(
        f"A date is priced when its status is not `failed`; flagged when a name is kept unscreened (`n_names_unscreened > 0`). Terciles of the clipped mass inside ±2.5 sd (`clip_inner_max`: the larger of the two one-sided clipped masses, not their sum; C.1) on all priced dates: "
        f"low ≤ {lo_s} < middle ≤ {hi_s} < high (fractions of the particles: {100 * lo:.2f} % and {100 * hi:.2f} %; sections A and B print the clipped mass in %); the samples without flagged dates keep these bounds. Model S numbers are used on its converged dates only (the study's convention)."
    )
    add("")
    failed = d[d["status"] == "failed"]
    add(
        md_table(
            ["failed date", "what failed (step, leg)", "exception text (`reason` of the per-date CSV)"],
            [[r["date"], failure_clause(r["date"], r["reason"], logs), r["reason"]] for _, r in failed.iterrows()],
            "lll",
        )
        if len(failed)
        else "No failed date."
    )  # fmt: skip
    add("")
    add(
        "Failed dates: no number of the run; they stay in the per-date CSV with the exception text (`reason`). "
        f"The step and the leg are read from the traceback of the run's log of the date (`outputs/dispersion_lc/logs/{logs.name}/<date>.log`) and the code path it names "
        f"(`scripts/lcm_price.py` → `volsto/studies/disp_lc.py::lc_spec_from_smiles`: each name's smile is screened, repaired and fitted in turn, then the index's is screened{', repaired (decision 5)' if with_d5 else ''} and fitted): all three dates stop while the model's inputs (the SVI surfaces) are built, before the local correlation is calibrated."
    )
    add("")
    flagged = d[priced & truth(d["flag_unscreened"])]
    if len(flagged):
        add(md_table(["flagged date", "name kept unscreened", "status", "LC/CC", "LC/copula"], [
            [r["date"], r["names_unscreened"], r["status"],
             book.num(f"C.hist.flagged.{r['date']}.lc_over_cc", "LC/CC on a flagged date", r["lc_over_cc"], r["lc_over_cc_se"], date=r["date"], definition=qs[0].definition, source="tables/C_history_by_date.csv"),
             book.num(f"C.hist.flagged.{r['date']}.lc_over_copula", "LC/copula on a flagged date", r["lc_over_copula"], r["lc_over_copula_se"], date=r["date"], definition=qs[1].definition, source="tables/C_history_by_date.csv")]
            for _, r in flagged.iterrows()
        ], "lllrr"))  # fmt: skip
        add("")
        add(
            "Flagged dates (decision 2): a name with no expiry passing the quote screen is kept unscreened; ± is the Monte Carlo standard error of the date."
        )
    else:
        add("No flagged date.")
    add("")
    extra_names = d[priced & (d["n_names_extrapolated"].astype(float) > 0)]
    extra_index = d[priced & truth(d["index_extrapolated"])]
    s_not = d[priced & ~truth(d["model_s_converged"])]
    names_list = ", ".join(
        f"{r.date} ({book.num(f'C.hist.n_names_extrapolated.{r.date}', 'names priced beyond their last listed expiry', r.n_names_extrapolated, spec='.0f', date=r.date, unit='names', definition='`n_names_extrapolated` of the row', source=book.source, notes='a count of names')})"
        for r in extra_names.itertuples()
    )  # fmt: skip
    low_n, high_n = (
        int(truth(d.loc[priced, c]).sum()) for c in ("flag_clip_low", "flag_clip_high")
    )
    add(
        f"Other flags of the per-date CSV: `n_names_extrapolated > 0` on {count('names_extrapolated', 'priced dates with n_names_extrapolated > 0', len(extra_names))} priced dates ({names_list or 'none'}); "
        f"`index_extrapolated` on {count('index_extrapolated', 'priced dates with index_extrapolated', len(extra_index))} ({', '.join(extra_index['date']) or 'none'}); "
        f"`flag_clip_low` (clipped mass at λ = 0 above {CLIP_FLAG_MASS:g}) on {count('flag_clip_low', f'priced dates with clip_low_inner_max > {CLIP_FLAG_MASS:g}', low_n)}, "
        f"`flag_clip_high` (at the cap, above {CLIP_FLAG_MASS:g}) on {count('flag_clip_high', f'priced dates with clip_high_inner_max > {CLIP_FLAG_MASS:g}', high_n)}. "
        f"Model S converged on {count('model_s_converged', 'dates of the table on which model S converged', int(truth(d['model_s_converged']).sum()))} of the table's {len(d)} dates "
        f"and on {int(n_of['S'])} of the {int(n_of['all'])} priced ones; not on {', '.join(s_not['date']) or 'none'} (no model S number there)."
    )
    add("")
    # what "ok" means for the index (the gate is waived when the wing binds)
    wing = priced & truth(d["wing_binds"])
    wing_err = wing & truth(d["index_error_above_gate"])
    ok_dates = priced & (d["status"] == "ok")
    e90 = d.loc[ok_dates, "idx_err_90"].astype(float)
    eatm = d.loc[ok_dates, "idx_err_atm"].astype(float)
    worst_ok = d.loc[e90.idxmin()]
    worst_atm = d.loc[eatm.abs().idxmax()]
    target = "its target (the model's own SVI fit of the DJX smile, not the study's listed vols)"
    err_def = f"the model's index implied volatility minus {target}, in vol points"
    by_csv = "tables/C_history_by_date.csv"

    def ok_count(column: str, level: float) -> str:
        v = e90 if column == "idx_err_90" else eatm
        tag = f"ok_{column}_above_{level:g}".replace(".", "p")
        return count(tag, f"dates with status ok on which |{column}| exceeds {level:g} vol points ({column}: {err_def})", int((v.abs() > level).sum()))  # fmt: skip

    add(
        f"What `ok` means for the index. The index gate `check_index` asks the model's index smile within {INDEX_GATE_VP:g} vol points of {target} at the money and at 90 % of the forward "
        "(`idx_err_atm`, `idx_err_90`: the model's index implied volatility minus that target, in vol points), "
        f"and is waived when the wing binds (`wing_binds`: `clip_inner_max` above {CLIP_FLAG_MASS:g}, i.e. {pct(CLIP_FLAG_MASS)} of the particles). The wing binds on {count('wing_binds', 'priced dates on which the wing binds (clip_inner_max > 0.01): the index gate is waived', int(wing.sum()))} of the {int(priced.sum())} priced dates; "
        f"on {count('wing_binds_index_error_above_gate', 'priced dates on which the wing binds and |idx_err_atm| or |idx_err_90| exceeds 0.15 vol points', int(wing_err.sum()))} of those |`idx_err_atm`| or |`idx_err_90`| exceeds {INDEX_GATE_VP:g} vol points; "
        f"`check_index` fails on {len(gate_fail['check_index'])}. "
        f"On the {int(ok_dates.sum())} `ok` dates: |`idx_err_90`| exceeds {INDEX_GATE_VP:g} vol points on {ok_count('idx_err_90', INDEX_GATE_VP)}, 0.5 on {ok_count('idx_err_90', 0.5)} and 1 on {ok_count('idx_err_90', 1.0)} "
        f"(median `idx_err_90` {book.num('C.hist.idx_err_90.median_ok', 'median idx_err_90 over the dates with status ok', float(e90.median()), spec='+.2f', unit='vol points', definition=f'`idx_err_90` of the row: {err_def}, at 90 % of the forward; the median over the dates with status ok', n=int(ok_dates.sum()), source=by_csv, notes='an order statistic across dates: no standard error given')} vol points; "
        f"the lowest: {book.num('C.hist.idx_err_90.min_ok', 'lowest idx_err_90 on a date with status ok', float(worst_ok['idx_err_90']), spec='+.2f', unit='vol points', date=str(worst_ok['date']), definition=f'`idx_err_90` of the row: {err_def}, at 90 % of the forward; the minimum over the dates with status ok', source=by_csv, notes='the Monte Carlo error of the date is the column `idx_err_90_se` of the source table; not read here')} vol points, on {worst_ok['date']}); "
        f"|`idx_err_atm`| exceeds {INDEX_GATE_VP:g} vol points on {ok_count('idx_err_atm', INDEX_GATE_VP)} "
        f"(the largest in size: {book.num('C.hist.idx_err_atm.max_abs_ok', 'idx_err_atm of largest size on a date with status ok', float(worst_atm['idx_err_atm']), spec='+.2f', unit='vol points', date=str(worst_atm['date']), definition=f'`idx_err_atm` of the row: {err_def}, at the money; the value of largest absolute size over the dates with status ok', source=by_csv, notes='the Monte Carlo error of the date is the column `idx_err_atm_se` of the source table; not read here')} vol points, on {worst_atm['date']}). "
        f"Status `ok` therefore does not mean that the model's index smile is within {INDEX_GATE_VP:g} vol points of that target, at the money or at 90 % of the forward."
    )
    add("")
    # the index second moment against the listed strip
    rows = []
    listed = d[idx15]
    drop_def = f"`n_dropped_index` − `{DROPPED_CALENDAR}` of the row: the index slices dropped by the quote screen, without those dropped by the calendar repair of the DJX target (decision 5)" if with_d5 else "`n_dropped_index` of the row"  # fmt: skip
    for at, r in listed.iterrows():
        rows.append([
            r["date"], rel_cell(r),
            book.num(f"C.hist.n_dropped_index.{r['date']}", "index slices dropped by the screen on the date", drops[at], spec=".0f", date=r["date"], unit="slices", definition=drop_def, source=book.source, notes="a count of slices"),
            *([book.num(f"C.hist.{DROPPED_CALENDAR}.{r['date']}", "index slices dropped by the calendar repair of the DJX target (decision 5) on the date", r[DROPPED_CALENDAR], spec=".0f", date=r["date"], unit="slices", definition=f"`{DROPPED_CALENDAR}` of the row", source=book.source, notes="a count of slices")] if with_d5 else []),
            book.num(f"C.hist.rho_cc.{r['date']}", "constant correlation of the companion on the date", r["rho_cc"], spec=".3f", date=r["date"], definition="`rho_cc` of the row: the constant correlation fitted for the companion model", source=book.source, notes="a fitted parameter: no standard error in the row"),
            book.num(f"C.hist.rho_cop.{r['date']}", "the copula's correlation on the date", r["rho_cop"], spec=".3f", date=r["date"], study=True, definition="`rho_cop` of the row (the study's copula correlation of the entry)", source=book.source),
            *(book.num(f"C.hist.index_moment_dates.{r['date']}.{k}", f"{by_key[k].label} on a date with the index second moment more than 15 % from the listed strip", r[k], r[f"{k}_se"], date=r["date"], definition=by_key[k].definition, source="tables/C_history_by_date.csv") for k in ("lc_over_cc", "lc_over_copula", "cc_over_copula")),
            f"`{r['status']}`",
        ])  # fmt: skip
    other = d[idx10 & ~idx15]
    extremes_here = []
    for k in ("lc_over_cc", "lc_over_copula", "cc_over_copula", "EV_over_EQV_lc"):
        v = d.loc[priced, k].astype(float)
        for stat, at in (("minimum", v.idxmin()), ("maximum", v.idxmax())):
            if bool(idx15[at]):
                extremes_here.append(f"the {stat} of {by_key[k].label} ({d.loc[at, 'date']})")
    add(
        f"Index second moment (`flag_index_moment`). On {count('flag_index_moment', 'priced dates with |basket part / M_B^listed| > 0.10 (flag_index_moment)', int(idx10.sum()))} priced dates the model's basket second moment E_LC[R̄²] is more than {pct(INDEX_MOMENT_FLAG)} from the listed index strip M_B^listed "
        f"(|basket part / M_B^listed| > {INDEX_MOMENT_FLAG:g}; column `flag_index_moment` of the per-date CSV). On {count('index_moment_beyond_15pct', 'priced dates with |basket part / M_B^listed| > 0.15', int(idx15.sum()))} of them it is more than {pct(INDEX_MOMENT_LIST)} away:"
    )
    add("")
    drop_heads = [drop_col.strip("`"), *([DROPPED_CALENDAR] if with_d5 else [])]
    add(md_table(["date", "basket part / M_B^listed", *drop_heads, "rho_cc", "rho_cop", "LC/CC", "LC/copula", "CC/copula", "status"], rows, "lr" + "r" * len(drop_heads) + "rrrrrl"))  # fmt: skip
    add("")
    if len(listed):
        dropped_lo, dropped_hi = int(drops[idx15].min()), int(drops[idx15].max())
        also_d5 = f"; the calendar repair of the DJX target (decision 5) drops a slice on {int((listed[DROPPED_CALENDAR].astype(float) > 0).sum())} of them (`{DROPPED_CALENDAR}`)" if with_d5 else ""  # fmt: skip
        by_screen = " by the quote screen" if with_d5 else ""
        add(
            f"± is the Monte Carlo standard error of the date. On these {len(listed)} dates the quote screen dropped {dropped_lo} to {dropped_hi} of the index's own slices ({drop_col}){also_d5}: LC and CC are fitted to an index variance that is not the listed strip's, "
            "so the comparison with the copula is not like for like on these dates (`rho_cc` against `rho_cop` above). "
            f"Of the extremes over all priced dates (C.2a), these dates supply {'; '.join(extremes_here) or 'none'}. "
            + (
                f"On the other {len(other)} dates of the flag the basket part / M_B^listed runs from {100 * float(other['EV_basket_part_rel_MB'].min()):+.1f} % to {100 * float(other['EV_basket_part_rel_MB'].max()):+.1f} % "
                f"and no index slice was dropped{by_screen} on {int((drops[other.index] == 0).sum())} of them (per-date CSV). "
                if len(other)
                else ""
            )
            + "The dates of the flag are marked in the tables of extremes under C.2a and C.2b; C.2c gives the headline rows without them."
        )
        add("")
    # the names' diagnostic
    largest = d[priced & (gap_abs > 0.10)].assign(_a=gap_abs).sort_values("_a", ascending=False)
    unflagged_out = int((names_out & ~truth(d["flag_unscreened"])).sum())
    add(
        f"Names' diagnostic. On {int(names_out.sum())} priced dates Σw E_LC[R_i²] is more than 2 % from the listed strips Σw M_i^listed (`names_gap`); more than 5 % on "
        f"{count('names_outside_5pct', 'priced dates with the names gap beyond 5 %', int((priced & (gap_abs > 0.05)).sum()))}, more than 10 % on {count('names_outside_10pct', 'priced dates with the names gap beyond 10 %', len(largest))}: "
        + "; ".join(
            f"{r['date']} {gap_cell(r)}" + (" (flagged)" if bool(r["flag_unscreened"]) else "")
            for _, r in largest.iterrows()
        )
        + f". \"Without the flagged dates\" removes only the {int(n_of['all']) - int(n_of['all_unflagged'])} dates on which a name is kept unscreened: "
        f"{count('names_outside_2pct_unflagged', 'priced dates with the names diagnostic outside 2 % that are not flagged (they stay in the summaries without the flagged dates)', unflagged_out)} of the {int(names_out.sum())} dates stay in every summary without the flagged dates. "
        "C.2c gives the rows this moves on the dates inside 2 %. "
        "Section V4 measures in which region of strikes the names' second moment under LC exceeds the listed strips; section V3 is the caveat on the high-strike calls (rows call 1.25× and 1.5× below)."
    )
    add("")
    # ---------------------------------------------------------------- C.1 definitions
    add("### C.1 Definitions")
    add("")
    add(
        md_table(
            ["quantity", "definition"],
            [
                [
                    q.label,
                    q.definition + ("; no standard error (the study's numbers)" if q.study else ""),
                ]
                for q in md_qs
            ],
            "ll",
        )
    )
    add("")
    add(
        "LC = the calibrated local correlation model, CC = its constant-correlation companion, copula = the study's model (P_D), model S = the study's skewed model; "
        "D = Σ w_i |R_i − R̄|, V = Σ w_i (R_i − R̄)². In the tables below a mean's ± is the standard error across dates (sd/√n, no serial-correlation adjustment) and a pooled ratio's ± is its across-dates linearisation; "
        'the Monte Carlo error with the dates fixed is the column `mc_se` of the summaries CSV (an upper bound: the dates share the seeds), printed in C.2 as "MC bound".'
    )
    add("")
    last = NW_MORE_LAGS[-1]
    factors = {
        (smp, lags): [table[(k, smp)]["mean"]["nw_se" if lags == NW_LAGS else f"nw_se_{lags}"] / table[(k, smp)]["mean"]["se"] for k in HEADLINE]
        for smp in ("all", "all_unflagged")
        for lags in (NW_LAGS, last)
    }  # fmt: skip

    def factor(which: str, smp: str, lags: int = NW_LAGS) -> str:
        value = min(factors[(smp, lags)]) if which == "min" else max(factors[(smp, lags)])
        stem = "nw_over_se" if lags == NW_LAGS else f"nw{lags}_over_se"
        return book.num(
            f"C.hist.{stem}.{which}.{smp}", f"Newey-West error at {lags} lags over the printed across-dates error of the mean: {which} over the headline rows, {smp}", value, spec=".1f",
            definition=f"the Newey-West standard error of the mean at {lags} lags over its across-dates standard error sd/√n, the {which} over the rows {', '.join(by_key[k].label for k in HEADLINE)}.{in_sample(smp)}", n=int(n_of[smp]), source=src,
        )  # fmt: skip

    # the gaps of the sample in which the lags are counted (rows, not months)
    days = pd.to_datetime(d.loc[priced, "date"]).diff().dt.days.dropna()
    n_gaps = count("gaps_all_priced", f"pairs of consecutive priced dates more than {GAP_DAYS} calendar days apart", int((days > GAP_DAYS).sum()))  # fmt: skip
    longest = book.num("C.hist.longest_gap_days.all", "longest gap between two consecutive priced dates", float(days.max()), spec=".0f", unit="calendar days", definition="the largest difference in calendar days between two consecutive dates of the sample of all priced dates", n=int(n_of["all"]), source=src, notes="a count of days")  # fmt: skip
    lag_list = f"{NW_LAGS}, {' and '.join(str(x) for x in NW_MORE_LAGS)}"
    # the headline rows on which the error of the mean still grows from the shortest to the longest lag
    growth = {k: table[(k, "all")]["mean"][f"nw_se_{last}"] / table[(k, "all")]["mean"]["nw_se"] for k in HEADLINE}  # fmt: skip
    grows = [by_key[k].label for k in HEADLINE if growth[k] > 1.10]
    flat = [by_key[k].label for k in HEADLINE if growth[k] <= 1.10]
    grows_s = f"On all priced dates the {last}-lag error is more than 10 % above the {NW_LAGS}-lag one on the rows {', '.join(grows) or 'none'}" + (f"; not on {', '.join(flat)}" if flat else "") + "."  # fmt: skip
    add(
        "Serial correlation. The printed ± of a mean treats the dates as independent; the entries are monthly, the trade lasts three months, and the series are serially correlated. "
        f"The Newey-West standard error of the mean (Bartlett kernel) is given in C.2a and C.2b at {lag_list} lags for the rows {', '.join(by_key[k].label for k in HEADLINE)}, with the lag-1 autocorrelation. "
        f"Newey-West at {NW_LAGS} lags is a lower value: the error still grows with the lag ({' and '.join(str(x) for x in NW_MORE_LAGS)} lags shown); lags are counted in rows of the sample, which has gaps. "
        f"{grows_s} "
        f"The gaps: on all priced dates {n_gaps} pairs of consecutive rows are more than {GAP_DAYS} calendar days apart, the longest {longest} days. "
        f"Over the {len(HEADLINE)} rows the Newey-West error on all priced dates is {factor('min', 'all')} to {factor('max', 'all')} times the printed ± at {NW_LAGS} lags and {factor('min', 'all', last)} to {factor('max', 'all', last)} times at {last} lags "
        f"({factor('min', 'all_unflagged')} to {factor('max', 'all_unflagged')} times and {factor('min', 'all_unflagged', last)} to {factor('max', 'all_unflagged', last)} times without the flagged dates). "
        f"The tables by sample (C.3–C.5), the pooled ratios and C.6 print the independent-dates ± only; C.2c prints the {NW_LAGS}-lag error only; the Newey-West error of every mean is in the summaries CSV "
        f"(columns `nw_se` at {NW_LAGS} lags, {', '.join(f'`nw_se_{x}`' for x in NW_MORE_LAGS)}; for a half, a tercile or a restricted sample the lags are counted in consecutive rows of that sample)."
    )
    add("")
    low_larger = int((priced & (d["clip_larger_side"] == "low")).sum())
    high_larger = int((priced & (d["clip_larger_side"] == "high")).sum())
    add(
        f"Clipped mass. `clip_inner_max` is {CLIP_DEF} (the two maxima need not be at the same slice). "
        "Unit: in this part and in section D the clipped masses are fractions of the particles (0.10 = 10 % of the particles); sections A and B print the same masses in %. "
        f"The lower bound (λ = 0) is the larger side on {count('clip_low_larger', 'priced dates on which clip_low_inner_max > clip_high_inner_max', low_larger)} of the {int(priced.sum())} priced dates, "
        f"the cap on {count('clip_high_larger', 'priced dates on which clip_high_inner_max > clip_low_inner_max', high_larger)}, and the two are equal on {count('clip_sides_equal', 'priced dates on which the two one-sided clipped masses are equal', int(priced.sum()) - low_larger - high_larger)} (column `clip_larger_side`). "
        "The terciles of C.0, the regressor of C.7 and the flags `flag_clip_low` / `flag_clip_high` use these columns as defined here."
    )
    add("")
    # ---------------------------------------------------------------- C.2 full statistics
    for tag, sample, title in (
        ("a", "all", "all priced dates"),
        ("b", "all_unflagged", "without the flagged dates"),
    ):
        add(f"### C.2{tag} Full statistics, {title} (n = {n_cell(sample)})")
        add("")
        rows = []
        for q in md_qs:
            n_q = table[(q.key, sample)].get("mean", {}).get("n", 0)
            rows.append([
                q.label, f"{n_q:d}", stat_cell(q, sample, "mean"), nw_cell(q, sample, more=True) if q.key in HEADLINE else "", mc_cell(q, sample, "mean"),
                *(stat_cell(q, sample, s) for s in ("q25", "median", "q75", "min", "max")), stat_cell(q, sample, "pooled") if q.num else "", mc_cell(q, sample, "pooled") if q.num else "",
            ])  # fmt: skip
        add(
            md_table(
                ["quantity", "n", "mean ± se", f"± Newey-West at {lag_list.replace(', ', ' / ').replace(' and ', ' / ')} lags (lag-1 autocorrelation)", "MC bound, mean", "q25", "median", "q75", "min", "max", "pooled ± se", "MC bound, pooled"],
                rows,
            )
        )  # fmt: skip
        add("")
        above = []
        for q in md_qs:
            for stat in ("mean", "pooled"):
                cell = table[(q.key, sample)].get(stat)
                if cell is not None and np.isfinite(cell["mc_se"]) and cell["mc_se"] > cell["se"]:
                    dg = q.digits  # one more digit while the two print alike
                    while dg < q.digits + 4 and f"{cell['mc_se']:.{dg}f}" == f"{cell['se']:.{dg}f}":
                        dg += 1
                    above.append(
                        f"{q.label} ({stat}: {cell['mc_se']:.{dg}f} against {cell['se']:.{dg}f})"
                    )
        add(
            "Per-date quantities over the dates of the sample; pooled = Σ numerator / Σ denominator over the same dates; n is smaller for model S (converged dates only). "
            f"± Newey-West: the standard error of the mean with the Bartlett kernel at {lag_list} lags, for the headline rows, with the lag-1 autocorrelation of the series in brackets. "
            f"Newey-West at {NW_LAGS} lags is a lower value: the error still grows with the lag ({' and '.join(str(x) for x in NW_MORE_LAGS)} lags shown; C.1 names the rows on which it does); lags are counted in rows of the sample, which has gaps (C.1). "
            "MC bound: the Monte Carlo standard error with the dates fixed, an upper bound; the dates share their seeds, so this error does not average out over the dates and is not contained in the across-dates ±; "
            "empty for the study's numbers and for the clipped masses, which carry no Monte Carlo error here. "
            f"The MC bound is larger than the across-dates ± for: {'; '.join(above) or 'no row'}."
        )
        add("")
        rows = []
        sub = d.loc[masks[sample]].set_index("date")
        for key in (
            "lc_over_cc",
            "lc_over_copula",
            "cc_over_copula",
            "s_over_copula",
            "kappa_lc",
            "EV_over_EQV_lc",
        ):
            q = by_key[key]
            v = sub[key].astype(float).dropna()
            cells = []
            for stat, at in (("min", v.idxmin()), ("max", v.idxmax())):
                r_at = sub.loc[at].copy()
                r_at["date"] = at
                cells.append(f"{stat_cell(q, sample, stat)} on {at} ({marks(r_at)})")
            rows.append([q.label, *cells])
        add(
            md_table(
                ["quantity", "min: date (status, flags)", "max: date (status, flags)"], rows, "lll"
            )
        )
        add("")
        add(
            f"The dates of the extremes of the table above, with the status ({status_note}) and, where they apply: flagged (a name kept unscreened), "
            f"the index second moment more than {pct(INDEX_MOMENT_FLAG)} from the listed strip (`flag_index_moment`, with basket part / M_B^listed), the names' second moment more than 2 % from the listed strips (with `names_gap`)."
        )
        add("")
    # ---------------------------------------------------------------- C.2c restricted samples
    add("### C.2c Restricted samples: the index second moment and the names' diagnostic")
    add("")
    n10, n15 = int(idx10.sum()), int(idx15.sum())
    blocks = (
        (
            ("lc_over_cc", "lc_over_copula", "cc_over_copula", "s_over_copula", "listed_fwd_ratio"),
            (("all", "all priced dates"), ("all_index10", f"without the {n10} dates more than {pct(INDEX_MOMENT_FLAG)} from the listed strip"), ("all_index15", f"without the {n15} dates more than {pct(INDEX_MOMENT_LIST)} from the listed strip")),
            f"Without the dates where the model's index second moment is more than {pct(INDEX_MOMENT_FLAG)} ({pct(INDEX_MOMENT_LIST)}) from the listed strip: |basket part / M_B^listed| > {INDEX_MOMENT_FLAG:g} ({INDEX_MOMENT_LIST:g}), C.0. "
            f"± is the standard error across dates (independent dates); the Newey-West column is at {NW_LAGS} lags, a lower value as in C.2 (the error still grows with the lag on the rows named in C.1; {', '.join(f'`nw_se_{x}`' for x in NW_MORE_LAGS)} of the summaries CSV; the rows of a restricted sample are further apart in time). Model S rows: its converged dates within the sample.",
        ),
        (
            ("EV_over_EQV_lc", "kappa_lc", "EV_single_part", "EV_basket_part", "C_lc_over_copula_150", "lc_over_cc", "lc_over_copula"),
            (("all", "all priced dates"), ("all_unflagged", "without the flagged dates"), ("all_names2pct", "names' diagnostic inside 2 %")),
            "On the dates where the names' diagnostic is inside 2 % (`names_within_2pct`), beside all priced dates and the sample without the flagged dates, for the rows the diagnostic moves and, for comparison, LC/CC and LC/copula. "
            "± is the standard error across dates (independent dates).",
        ),
    )  # fmt: skip
    for keys, sample_labels, sentence in blocks:
        rows = []
        for key in keys:
            q = by_key[key]
            for sample, label in sample_labels:
                n_q = table[(q.key, sample)].get("mean", {}).get("n", 0)
                rows.append([q.label, label, f"{n_q:d}", stat_cell(q, sample, "mean"), nw_cell(q, sample) if q.key in HEADLINE else "", *(stat_cell(q, sample, s) for s in ("q25", "median", "q75", "min", "max"))])  # fmt: skip
        add(md_table(["quantity", "sample", "n", "mean ± se", f"± Newey-West at {NW_LAGS} lags (lag-1 autocorrelation)", "q25", "median", "q75", "min", "max"], rows, "llrrrrrrrr"))  # fmt: skip
        add("")
        add(sentence)
        add("")
    # ---------------------------------------------------------------- C.3-C.5 by sample
    for num, stat_title, maker, sentence, only_ratios in (
        ("C.3", "Mean ± se by sample", lambda q, s: stat_cell(q, s, "mean"), "Mean over the dates of each sample ± the standard error across dates.", False),
        ("C.4", "Pooled ratio by sample", lambda q, s: stat_cell(q, s, "pooled"), "Σ numerator / Σ denominator over the dates of each sample (the convention of the study's model S table, `scripts/disp_tables2.py::model_s_tables`) ± the across-dates linearised standard error.", True),
        ("C.5", "Median [q25, q75] by sample", lambda q, s: f"{stat_cell(q, s, 'median')} [{stat_cell(q, s, 'q25')}, {stat_cell(q, s, 'q75')}]", "Median and quartiles over the dates of each sample.", False),
    ):  # fmt: skip
        for tag, suffix, title in (
            ("a", "", "with all priced dates"),
            ("b", "_unflagged", "without the flagged dates"),
        ):
            add(f"### {num}{tag} {stat_title}, {title}")
            add("")
            keys = [f"{key}{suffix}" for key, _ in BASE_SAMPLES]
            rows = [
                ["n dates", *(n_cell(k) for k in keys)],
                ["n dates, model S rows and LC/CC − S/copula", *(n_s_cell(k) for k in keys)],
            ]
            for q in md_qs:
                if only_ratios and not q.num:
                    continue
                rows.append([q.label, *(maker(q, k) for k in keys)])
            add(md_table(["quantity", *(label for _, label in BASE_SAMPLES)], rows))
            add("")
            add(
                sentence
                + f" Model S rows (S/copula, κ model S, E[V]/EQV model S, calls S/copula) and LC/CC − S/copula: model S's converged dates within the sample, the second line of the table; they differ from the first line where {TODAY}, which has no model S number, is in the sample."
            )
            add("")
    # ---------------------------------------------------------------- check (b)
    add("### C.6 Check (b): LC/copula and CC/copula")
    add("")
    today = d[d["date"] == TODAY]
    keys_b = ("lc_over_cc", "lc_over_copula", "cc_over_copula")
    if len(today) == 1 and bool(today["priced"].iloc[0]):
        t = today.iloc[0]
        stored_s = f"stored in the table, before decision 3: `{t['status_stored']}`" + (f", {t['reason_stored']}" if t["reason_stored"] else "")  # fmt: skip
        status_s = f"`{t['status']}`" + (f" ({t['reason']})" if t["reason"] else "") + (" (the row's own status, stored under decision 3)" if rows_own else f" under decision 3; {stored_s}")  # fmt: skip
        own_values = {k: (float(t[k]), float(t[f"{k}_se"])) for k in keys_b}
        old = production_old_defaults(float(t["P_D"]), float(t["P_D_se"]))
        old_run = None if old is None else ("today_production_old_defaults", f"production budget, old defaults: no repair, no fallback (`{PRODUCTION_OLD.name}`, commit {old['commit']}, status `{old['status']}`)", old["values"], {"budget": old["budget"], "commit": old["commit"], "source": f"outputs/dispersion_lc/{PRODUCTION_OLD.name}", "notes": "the stopped production pass at the old defaults (no calendar repair, no unscreened fallback)"})  # fmt: skip
        # the frozen part's pass: its own row first, then the two production rows; another pass:
        # the frozen part's development row, the old-defaults production row, then its own row
        dev: dict[str, Any] | None = None
        new: dict[str, Any] | None = None
        if run is DEVELOPMENT:
            runs = [("today_dev", f"development budget, decisions 1–2 on (`{rows_path.name}`, commit {commit}, status `{t['status']}`)", own_values, {"source": f"{book.source} (row of {TODAY})", "notes": f"status of the row: {status_s}"})]  # fmt: skip
            if old_run is not None:
                runs.append(old_run)
            new = production_new_defaults(float(t["P_D"]), float(t["P_D_se"]))
        else:
            runs = []
            dev = development_row(float(t["P_D"]), float(t["P_D_se"]))
            if dev is not None:
                dev_stored = f"stored in the table, before decision 3: `{dev['status_stored']}`" + (f", {dev['reason_stored']}" if dev["reason_stored"] else "")  # fmt: skip
                runs.append(("today_dev", f"development budget, {DEVELOPMENT.decisions} (the frozen part's row; `{DEVELOPMENT.rows.name}`, commit {dev['commit']}, status `{dev['status']}`)", dev["values"], {"budget": dev["budget"], "commit": dev["commit"], "source": f"outputs/dispersion_lc/{DEVELOPMENT.rows.name} (row of {TODAY})", "notes": f"the row of the frozen part's development pass; status of the row: `{dev['status']}` under decision 3; {dev_stored}"}))  # fmt: skip
            if old_run is not None:
                runs.append(old_run)
            in_a = same_as_section_a(t)
            a_file = f"`{PRODUCTION_ROW.relative_to(pc.LC_OUT)}`"
            a_note = "" if in_a is None else (f"; the same row as section A's row file {a_file} (E_LC[D], E_CC[D], their standard errors, the paired ratio and the commit are equal)" if in_a else f"; not the row of section A's row file {a_file}: a number or the commit differs")  # fmt: skip
            runs.append(("today_production", f"{run.budget} budget, {run.decisions} (this table's row; `{rows_path.name}`, commit {commit}, status `{t['status']}`)", own_values, {"source": f"{book.source} (row of {TODAY})", "notes": f"today's row of this table; status of the row: {status_s}{a_note}"}))  # fmt: skip
        if new is not None:
            new_src = f"outputs/dispersion_lc/{PRODUCTION_ROW.relative_to(pc.LC_OUT)}"
            runs.append(("today_production", f"production budget, decisions 1–2 and 5 on (section A's row; `{PRODUCTION_ROW.relative_to(pc.LC_OUT)}`, commit {new['commit']}, status `{new['status']}`)", new["values"], {"budget": new["budget"], "commit": new["commit"], "source": new_src, "notes": f"today's production row at the new defaults (calendar repair, unscreened fallback, decision 5): section A's row; ratios to the copula with the copula's P_D and P_D_se of entries_3m.parquet (B1); status stored in the row: {new['status_stored']}"}))  # fmt: skip
        rows = []
        for tag, label, values, extra in runs:
            rows.append([label, *(book.num(f"C.b.{tag}.{k}", f"{by_key[k].label}, {TODAY}: {label}", values[k][0], values[k][1], digits=6, date=TODAY, definition=by_key[k].definition, **{"source": "tables/C_history_by_date.csv", **extra}) for k in keys_b)])  # fmt: skip
        rows.append(["the owner's arithmetic", "", *(book.num(f"C.b.owner.{k}", f"{by_key[k].label}, {TODAY}: the owner's arithmetic", OWNER_B[k], date=TODAY, definition="the owner's arithmetic of check (b), as written in the request (4 decimals)", budget="the owner's arithmetic (not computed here)", commit="none: the owner's arithmetic, not computed by code", source="the owner's request, check (b)") for k in keys_b[1:])])  # fmt: skip
        for tag, label, values, extra in runs:
            cells = []
            for k in keys_b[1:]:
                diff, se = values[k][0] - OWNER_B[k], values[k][1]
                kw = {"source": "tables/C_history_by_date.csv", **extra, "notes": ""}
                diff_s = book.num(f"C.b.{tag}.{k}.minus_owner", f"{by_key[k].label}, {TODAY}: {label}, minus the owner's arithmetic", diff, se, digits=6, date=TODAY, definition=f"{by_key[k].label} of the run minus the owner's arithmetic; se: the Monte Carlo error of the run's value", **kw)  # fmt: skip
                z = book.num(f"C.b.{tag}.{k}.minus_owner_in_se", f"{by_key[k].label}, {TODAY}: {label}, that difference in standard errors", diff / se, spec="+.1f", date=TODAY, definition="the difference to the owner's arithmetic over the Monte Carlo standard error of the run's value", **kw)  # fmt: skip
                cells.append(f"{diff_s} ({z} se)")
            rows.append([f"minus the owner's arithmetic: {label.split(' (')[0]}", "", *cells])
        add(md_table([f"{TODAY}", "LC/CC", "LC/copula", "CC/copula"], rows))
        add("")
        tags = [tag for tag, *_ in runs]
        at = tags.index("today_production") if "today_production" in tags else 0
        if run is DEVELOPMENT:
            today_sentence = f"on {TODAY} at the development budget (decisions 1–2 on) LC/copula = {rows[0][2]} and CC/copula = {rows[0][3]}"
        else:
            today_sentence = f"on {TODAY} at the {run.budget} budget ({run.decisions}; this table's row) LC/copula = {rows[at][2]} and CC/copula = {rows[at][3]}"
            if dev is not None:
                today_sentence += f"; the development row of the frozen part ({DEVELOPMENT.decisions}) gives {rows[0][2]} and {rows[0][3]}"
        if old is not None:
            at_old = tags.index("today_production_old_defaults")
            today_sentence += f"; the production row at the old defaults gives {rows[at_old][2]} and {rows[at_old][3]}"
        if new is not None:
            today_sentence += f"; the production row at the new defaults (decisions 1–2 and 5 on, section A's row) gives {rows[at][2]} and {rows[at][3]}"
        today_sentence += f"; the owner's arithmetic is {OWNER_B['lc_over_copula']:.4f} and {OWNER_B['cc_over_copula']:.4f}"
        # which of today's rows gives the owner's two numbers at the 4 decimals they are written to
        same = [label.split(" (")[0] for _tag, label, values, _extra in runs if all(f"{values[k][0]:.4f}" == f"{OWNER_B[k]:.4f}" for k in keys_b[1:])]  # fmt: skip
        owner_s = (
            f" Of the {len(runs)} rows of {TODAY} above, the owner's {OWNER_B['lc_over_copula']:.4f} and {OWNER_B['cc_over_copula']:.4f} are reproduced at their 4 decimals by "
            + (" and ".join(f"the row «{x}»" for x in same) if same else "none of the rows")
            + "; the other rows differ from them by the amounts of the table (old against new defaults, development against production budget)."
        )
        if new is not None:
            owner_s += f" The row to quote for {TODAY} is section A's row: the production budget with decisions 1–2 and 5 on (LC/copula {rows[at][2]}, CC/copula {rows[at][3]}); the two other rows are shown for the comparison only."
        if run is not DEVELOPMENT:
            is_a = {True: f", which is section A's row ({a_file}: the same E_LC[D], E_CC[D], standard errors, paired ratio and commit)", False: f", which is not the row of section A's row file {a_file} (a number or the commit differs)", None: ""}[in_a]  # fmt: skip
            others = f"; the {len(runs) - 1} other rows are shown for the comparison only" if len(runs) > 1 else ""  # fmt: skip
            owner_s += f" The row to quote for {TODAY} is this table's row{is_a}: the {run.budget} budget with {run.decisions} (LC/copula {rows[at][2]}, CC/copula {rows[at][3]}){others}."
        s_today = (
            ""
            if bool(t["model_s_converged"])
            else f" Model S did not converge on {TODAY}: no S/copula for today."
        )
        beside = ""
        if dev is not None:
            beside += f" The development row is the frozen part's row of {TODAY} ({dev['budget']}; commit {dev['commit']}, {DEVELOPMENT.decisions}), read from `{DEVELOPMENT.rows.name}`: status `{dev['status']}` under decision 3, {dev_stored}."
        if run is not DEVELOPMENT and old is not None:
            beside += " The production row at the old defaults is the stopped pass without repair or fallback."
        add(
            f"± is the Monte Carlo standard error of the date (delta method with the copula's `P_D_se` for the ratios to the copula). Today's row of this section's table is at the {run.budget} budget ({pc.BUDGETS[run.budget]}), "
            f"status {status_s}; the owner's arithmetic is given to 4 decimals."
            + (
                f" The production row at the new defaults is section A's row of {TODAY} ({new['budget']}; commit {new['commit']}), read from its row file; the production row at the old defaults is the stopped pass without repair or fallback."
                if new is not None
                else ""
            )
            + beside
            + owner_s
            + s_today
        )
    else:
        add(f"Pending: the row of {TODAY} is not priced in `{rows_path.name}`; waits for that row.")
        today_sentence = f"the row of {TODAY} is pending"
    add("")
    rows = []
    for sample, label in (
        ("all", "all priced dates"),
        ("all_unflagged", "without flagged dates"),
        ("S", "∩ model S converged"),
        ("S_unflagged", "∩ model S converged, without flagged dates"),
    ):
        for key in ("lc_over_cc", "lc_over_copula", "cc_over_copula", "s_over_copula"):
            q = by_key[key]
            n_q = table[(q.key, sample)].get("mean", {}).get("n", 0)
            rows.append([label, q.label, f"{n_q:d}", stat_cell(q, sample, "mean"), stat_cell(q, sample, "q25"), stat_cell(q, sample, "median"), stat_cell(q, sample, "q75"), stat_cell(q, sample, "pooled")])  # fmt: skip
    add(
        md_table(
            ["sample", "quantity", "n", "mean ± se", "q25", "median", "q75", "pooled ± se"],
            rows,
            "llrrrrrr",
        )
    )
    add("")
    add(
        "Like for like across dates (the by-date values are the columns `lc_over_copula`, `cc_over_copula`, `s_over_copula` and `lc_over_cc` of `tables/C_history_by_date.csv`); ± across dates."
    )
    add("")

    def spread(key: str, sample: str) -> str:
        q = by_key[key]
        return f"{stat_cell(q, sample, 'mean')} (quartiles {stat_cell(q, sample, 'q25')}, {stat_cell(q, sample, 'median')}, {stat_cell(q, sample, 'q75')})"

    add(
        f"(b) as measured: {today_sentence}. Across the {int(n_of['all'])} priced dates the mean of LC/copula is {spread('lc_over_copula', 'all')} and of CC/copula {spread('cc_over_copula', 'all')}; "
        f"without the flagged dates (n = {int(n_of['all_unflagged'])}) {spread('lc_over_copula', 'all_unflagged')} and {spread('cc_over_copula', 'all_unflagged')}. "
        f"On the {int(n_of['S'])} dates where model S converged: S/copula {spread('s_over_copula', 'S')}, LC/CC {spread('lc_over_cc', 'S')}, LC/copula {spread('lc_over_copula', 'S')}, CC/copula {spread('cc_over_copula', 'S')}."
    )
    add("")
    # ---------------------------------------------------------------- check (d)
    add(
        "### C.7 Check (d): LC/CC − S/copula on the clipped mass and the basket part of the E[V] split"
    )
    add("")
    reg_src = "tables/C_history_regression.csv"

    def fit_cells(sample: str, key: str, label: str, fit: dict[str, Any], j: int, term: str, with_nw: bool = False, y_text: str = "y = LC/CC − P_D_S/P_D", on: str = "") -> list[str]:  # fmt: skip
        """The cells of one term of a fit (coefficient, classical se and t, HC1 se and t, R² and
        n on the first term; the Newey-West se and t when asked), each filed as a record.
        ``y_text`` names the regressand and ``on`` the regressors when the label does not."""
        base = f"C.d.{sample}.{key}.{term}"
        what = f"check (d), {label}, {sample}: {term}"
        defn = f"OLS with an intercept of {y_text} on {on or label}; n = {fit['n']} dates.{in_sample(sample)} The clipped mass is a fraction of the particles: a coefficient on it is per unit of that fraction"
        coef = book.num(f"{base}.coef", f"{what}, coefficient", fit["coef"][j], fit["se"][j], spec="+.5g", definition=defn + "; se: classical", n=fit["n"], source=reg_src, notes=f"HC1 standard error {fit['se_hc1'][j]:.4g}")  # fmt: skip
        se = f"{fit['se'][j]:.4g}"
        t = book.num(f"{base}.t", f"{what}, t (classical)", fit["t"][j], spec="+.2f", definition=defn + "; coefficient over its classical standard error", n=fit["n"], source=reg_src)  # fmt: skip
        hc = book.num(f"{base}.se_hc1", f"{what}, HC1 standard error", fit["se_hc1"][j], spec=".4g", definition=defn + "; HC1 (heteroskedasticity-consistent, n/(n − k)) standard error of the coefficient", n=fit["n"], source=reg_src)  # fmt: skip
        t_hc = book.num(f"{base}.t_hc1", f"{what}, t (HC1)", fit["t_hc1"][j], spec="+.2f", definition=defn + "; coefficient over its HC1 standard error", n=fit["n"], source=reg_src)  # fmt: skip
        cells = [coef, se, t, hc, t_hc]
        if with_nw:
            nw_def = f"; Newey-West standard error of the coefficient (Bartlett kernel, {NW_LAGS} lags counted in consecutive dates of the sample, the factor n/(n − k) of HC1)"
            cells.append(book.num(f"{base}.se_nw", f"{what}, Newey-West standard error", fit["se_nw"][j], spec=".4g", definition=defn + nw_def, n=fit["n"], source=reg_src))  # fmt: skip
            cells.append(book.num(f"{base}.t_nw", f"{what}, t (Newey-West)", fit["t_nw"][j], spec="+.2f", definition=defn + "; coefficient over its Newey-West standard error", n=fit["n"], source=reg_src))  # fmt: skip
        if j == 0:
            cells.append(book.num(f"C.d.{sample}.{key}.r2", f"check (d), {label}, {sample}: R²", fit["r2"], spec=".4f", definition=defn + "; centred R²", n=fit["n"], source=reg_src))  # fmt: skip
            cells.append(book.num(f"C.d.{sample}.{key}.n", f"check (d), {label}, {sample}: n", fit["n"], spec=".0f", definition=defn, n=fit["n"], unit="dates", source=reg_src))  # fmt: skip
            if with_nw:
                cells.append(book.num(f"C.d.{sample}.{key}.resid_lag1_autocorr", f"check (d), {label}, {sample}: lag-1 autocorrelation of the residuals", fit["resid_ac1"], spec="+.2f", definition=defn + "; Σ e_t e_(t−1) / Σ e_t² of the residuals, the dates of the sample in their order", n=fit["n"], source=reg_src))  # fmt: skip
        else:
            cells += ["", "", ""] if with_nw else ["", ""]
        return cells

    for tag, sample, title in (
        ("a", "S", "all dates of the intersection"),
        ("b", "S_unflagged", "without the flagged dates"),
    ):
        add(f"**C.7{tag} {title}**")
        add("")
        rows = []
        for key, label, _regs in FITS:
            fit = fits[(sample, key)]
            for j, term in enumerate(fit["names"]):
                rows.append([label if j == 0 else "", term, *fit_cells(sample, key, label, fit, j, term)])  # fmt: skip
        add(
            md_table(
                ["fit", "term", "coefficient", "se classical", "t", "se HC1", "t HC1", "R²", "n"],
                rows,
                "llrrrrrrr",
            )
        )
        add("")
        add(
            "OLS with an intercept of y = LC/CC − P_D_S/P_D across the dates where model S converged; `clip_inner_max` = clipped mass inside ±2.5 sd, a fraction of the particles (the larger of the two one-sided masses, not their sum; C.1): its coefficient is per unit of that fraction, a tenth of it per 0.10 of clipped mass (10 % of the particles); "
            "`EV_basket_part` = E_LC[R̄²] − M_B^listed, `EV_basket_part_rel_MB` = that over M_B^listed; the regressors are the dates' Monte Carlo estimates."
        )
        add("")
    # the two-regressor fit by the index slices dropped by the screen, with Newey-West errors
    add("**C.7c The fit on the clipped mass and the basket part / M_B^listed, by the index slices dropped by the screen**")  # fmt: skip
    add("")
    by_drop = (
        ("S", "all dates of the intersection"),
        ("S_dropidx_lt4", f"without the dates with {drop_col} ≥ 4"),
        ("S_dropidx_0", f"dates with no dropped index slice ({drop_col} = 0)"),
    )
    rows = []
    rel_label = FITS[0][1]
    for sample, title in by_drop:
        fit = fits[(sample, "rel")]
        for j, term in enumerate(fit["names"]):
            rows.append([title if j == 0 else "", term, *fit_cells(sample, "rel", rel_label, fit, j, term, with_nw=True)])  # fmt: skip
    add(
        md_table(
            ["sample", "term", "coefficient", "se classical", "t", "se HC1", "t HC1", "se Newey-West", "t Newey-West", "R²", "n", "residual lag-1 autocorrelation"],
            rows,
            "llrrrrrrrrrr",
        )
    )  # fmt: skip
    add("")
    add(
        f"The same OLS as the first fit of C.7a on three samples: the {fits[('S', 'rel')]['n']} dates of the intersection; those with fewer than 4 index slices dropped by the quote screen ({drop_col} < 4); those with none. "
        + (
            f"On this pass `n_dropped_index` of the rows also counts the slices dropped by the calendar repair of the DJX target (decision 5); {drop_col} leaves them out, so the three samples are defined as in the frozen part. "
            if with_d5
            else ""
        )
        + f"Newey-West: Bartlett kernel, {NW_LAGS} lags counted in consecutive dates of the sample, with the factor n/(n − k) of HC1 (with no lag it is HC1). The classical and HC1 columns of the first sample are those of C.7a."
    )
    add("")
    # the two legs of the regressand on the clipped mass alone, on the same dates
    add("**C.7d The two legs of y, each on the clipped mass alone, and their means by tercile of the clipped mass**")  # fmt: skip
    add("")
    rows = []
    n_legs = fits[("S", LEG_FITS[0][0])]["n"]
    for key, column, label in LEG_FITS:
        fit = fits[("S", key)]
        y_text = f"{by_key[column].label} ({by_key[column].definition})"
        for j, term in enumerate(fit["names"]):
            rows.append([label if j == 0 else "", term, *fit_cells("S", key, label, fit, j, term, y_text=y_text, on="the clipped mass alone (`clip_inner_max`)")])  # fmt: skip
    add(
        md_table(
            ["fit", "term", "coefficient", "se classical", "t", "se HC1", "t HC1", "R²", "n"],
            rows,
            "llrrrrrrr",
        )
    )
    add("")
    terciles = (("clip_low", "low"), ("clip_mid", "middle"), ("clip_high", "high"))
    leg_means: dict[str, list[str]] = {}
    rows = []
    for _key, column, _label in LEG_FITS:
        q = by_key[column]
        cells = []
        for terc, terc_label in terciles:
            v = d.loc[masks["S"] & masks[terc], column].astype(float)
            cells.append(book.num(
                f"C.d.legs.S.{column}.mean.{terc}", f"{q.label}: mean over the dates of check (d) in the {terc_label} tercile of the clipped mass", float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v))), study=q.study,
                definition=f"{q.definition}. Mean over the dates of the sample `S` ({sample_def['S']}) that are in the sample `{terc}` ({sample_def[terc]}); se across dates (sd/√n, independent dates)",
                n=len(v), source="tables/C_history_by_date.csv", notes=STUDY_NOTE if q.study else "the ± is the standard error across dates; the Monte Carlo error of the dates is not in it",
            ))  # fmt: skip
        leg_means[column] = cells
        rows.append([q.label, *cells])
    n_terc = [int((masks["S"] & masks[terc]).sum()) for terc, _ in terciles]
    # the priced dates of a tercile that are not in the sample of check (d) (no model S number)
    not_s = [
        f"its {x} tercile also has {', '.join(d.loc[masks[terc] & ~masks['S'], 'date'])}, which has no model S number"
        for terc, x in terciles
        if (masks[terc] & ~masks["S"]).any()
    ]
    rows.insert(0, ["n dates", *(f"{x:d}" for x in n_terc)])
    add(md_table(["mean ± se across dates", *(f"clipped mass: {x} tercile" for _, x in terciles)], rows))  # fmt: skip
    add("")
    add(
        f"The same {n_legs} dates as C.7a. First table: OLS with an intercept of each leg of y on the clipped mass alone (y = LC/CC − S/copula, so the two slopes differ by the slope of y on the clipped mass alone, C.7a). "
        "Second table: the mean of each leg over the dates of the intersection in each tercile of the clipped mass (the bounds of C.0, set on all priced dates); ± is the standard error across dates (independent dates). "
        + (
            f"C.3a's tercile columns are on all priced dates: {'; '.join(not_s)}, so the LC/CC mean of that column is not the one of this table. "
            if not_s
            else ""
        )
        + "The clipped mass is a fraction of the particles."
    )
    add("")
    add(agreement[0])
    check_d_file = check_d_path()
    check_d_cited = "" if check_d_file is None else check_d_source(check_d_file)
    check_d_notes = "a count or a difference of the comparison, not a Monte Carlo estimate; " + (
        "the source is the copy kept in the package of the checking session's results file"
        if check_d_file == CHECK_D_PACKAGE
        else "the source is a session scratch file, not part of the package"
    )
    for tag, value in agreement[1].items():
        book.num(f"C.d.independent_check.{tag}", f"check (d) against the independent check: {tag.replace('_', ' ')}", value, definition="the fits of this section compared with the independent check's results file, number by number, at its 6 significant digits", source=check_d_cited, notes=check_d_notes)  # fmt: skip
    add("")
    # measured statements
    main, clip_only, bp_only = fits[("S", "rel")], fits[("S", "clip")], fits[("S", "bp_rel")]
    y_q = by_key["y_check_d"]
    lt4, none = fits[("S_dropidx_lt4", "rel")], fits[("S_dropidx_0", "rel")]

    def term_s(fit: dict[str, Any], j: int, nw: bool = False) -> str:
        out = f"{fit['coef'][j]:+.4f} (classical se {fit['se'][j]:.4f}, t {fit['t'][j]:+.2f}; HC1 se {fit['se_hc1'][j]:.4f}, t {fit['t_hc1'][j]:+.2f}"
        if nw:
            out += f"; Newey-West se {fit['se_nw'][j]:.4f}, t {fit['t_nw'][j]:+.2f}"
        return out + ")"

    add(
        f"(d) as measured, n = {main['n']}: mean y = {stat_cell(y_q, 'S', 'mean')} (median {stat_cell(y_q, 'S', 'median')}). "
        f"With both regressors the coefficient on the clipped mass is {term_s(main, 1, nw=True)} "
        f"and on the basket part over M_B^listed {term_s(main, 2, nw=True)}, "
        f"intercept {main['coef'][0]:+.4f} (HC1 se {main['se_hc1'][0]:.4f}), R² {main['r2']:.4f}. "
        f"The clipped mass alone: R² {clip_only['r2']:.4f}; the basket part over M_B^listed alone: R² {bp_only['r2']:.4f}. "
        f"Without the flagged dates (n = {fits[('S_unflagged', 'rel')]['n']}): {fits[('S_unflagged', 'rel')]['coef'][1]:+.4f} and {fits[('S_unflagged', 'rel')]['coef'][2]:+.4f}, R² {fits[('S_unflagged', 'rel')]['r2']:.4f}. "
        f"Unit: the clipped mass is a fraction of the particles, so {main['coef'][1]:+.4f} is per unit of that fraction: "
        f"{book.num('C.d.S.rel.clip_inner_max.coef_per_10_points', 'check (d), clipped mass + basket part / M_B^listed, S: coefficient on the clipped mass per 10 points of clipped mass', 0.1 * main['coef'][1], 0.1 * main['se'][1], spec='+.4f', definition='0.1 × the coefficient on `clip_inner_max` (a fraction of the particles) of the two-regressor fit of y = LC/CC − P_D_S/P_D: the change of y per 10 % of the particles clipped; se: 0.1 × the classical standard error', n=main['n'], source=reg_src, notes=f'0.1 × the HC1 standard error: {0.1 * main["se_hc1"][1]:.4g}')} per 0.10 of clipped mass (10 % of the particles)."
    )
    add("")
    leg_lc, leg_s = fits[("S", "leg_lc_over_cc")], fits[("S", "leg_s_over_copula")]
    add(
        f"By leg (C.7d), on the same {main['n']} dates, each leg on the clipped mass alone: LC/CC has slope {leg_lc['coef'][1]:+.3f} (HC1 se {leg_lc['se_hc1'][1]:.3f}), R² {leg_lc['r2']:.3f}; "
        f"S/copula has slope {leg_s['coef'][1]:+.3f} (HC1 se {leg_s['se_hc1'][1]:.3f}), R² {leg_s['r2']:.2f}. "
        f"Means by tercile of the clipped mass (low, middle, high): LC/CC {', '.join(leg_means['lc_over_cc'])}; S/copula {', '.join(leg_means['s_over_copula'])}. "
        "The relation of the difference y with the clipped mass is therefore model S's discount to the copula growing with the clipped mass; LC/CC does not move with it in the fit (its slope is within one HC1 standard error of zero), and its tercile means differ without being ordered. "
        "It is an association across dates: the fit does not show that the clipping causes the difference, and the clipped mass may stand for another property of the dates."
    )
    add("")
    add(
        f"The clipped-mass coefficient is stable across the three samples of C.7c: {main['coef'][1]:+.4f} on the {main['n']} dates, {lt4['coef'][1]:+.4f} without the dates with {drop_col} ≥ 4 (n = {lt4['n']}) and {none['coef'][1]:+.4f} on the dates with no dropped index slice (n = {none['n']}); "
        f"HC1 t {main['t_hc1'][1]:+.2f}, {lt4['t_hc1'][1]:+.2f} and {none['t_hc1'][1]:+.2f}. "
        f"The basket-part coefficient is not: on the {main['n']} dates it is {term_s(main, 2, nw=True)}, not distinguishable from zero under HC1 or Newey-West; "
        f"without the dates with {drop_col} ≥ 4 it is {term_s(lt4, 2, nw=True)}, R² {lt4['r2']:.4f}, {lt4['coef'][2] / main['coef'][2]:.1f} times the full-sample coefficient; "
        f"on the dates with no dropped index slice it is {term_s(none, 2, nw=True)}, R² {none['r2']:.4f}, {none['coef'][2] / main['coef'][2]:.1f} times. "
        f"The first two samples differ by the {main['n'] - lt4['n']} dates with {drop_col} ≥ 4, which include {int((masks['S'] & ~masks['S_dropidx_lt4'] & idx15).sum())} of the {int(idx15.sum())} dates of C.0 where the model's index second moment is more than {pct(INDEX_MOMENT_LIST)} from the listed strip. "
        f"The residuals are serially correlated: lag-1 autocorrelation {main['resid_ac1']:+.2f} on the {main['n']} dates ({lt4['resid_ac1']:+.2f} and {none['resid_ac1']:+.2f} on the two other samples); the classical and HC1 errors do not allow for it; the Newey-West ones do up to {NW_LAGS} rows of the sample, a lower value (C.1)."
    )
    add("")
    # ---------------------------------------------------------------- figures
    add("### Figures F1 and F2")
    add("")
    before_today = [x for x in monthly_dates if x < TODAY]
    add(
        "- `figures/F1_forward_over_copula.pdf` (data: `figures/F1_forward_over_copula.csv`): by entry date, LC/copula, S/copula (converged dates) and the listed-variance forward over the copula √(EQV/EV); priced dates only. "
        f"The three lines are broken (a gap) at the {int((~priced).sum())} failed dates and at the {len(absent)} monthly dates of the study that have no row in the LC table (C, Dates): no line joins across a failed or an absent month; a date with a gap on both sides is drawn as a dot. "
        f"On these {len(no_line)} dates {f1_empty} "
        f"The CSV has one line per date of the table (as `tables/C_history_by_date.csv`) plus one empty line per absent monthly date (`in_lc_table` False). The last LC point is the {run.budget}-budget row of {TODAY}"
        + (
            f"; the study's last monthly date before it is {before_today[-1]}, which has no row in the table, so that point stands alone and is drawn as a dot. "
            if before_today and before_today[-1] in absent
            else ". "
        )
        + f"The lines include the {int(idx15.sum())} dates of C.0 on which the model's index second moment is more than {pct(INDEX_MOMENT_LIST)} from the listed strip ({', '.join(d.loc[idx15, 'date'])})."
    )
    add(
        f"- `figures/F2_calls_over_copula_by_strike.pdf` (data: `figures/F2_calls_over_copula_by_strike.csv`): calls over the copula's at 0.75, 1, 1.25 and 1.5 × the forward, LC and model S, mean and interquartile range over the same {int(n_of['S'])} dates (∩ model S converged); "
        "the CSV also carries LC on all priced dates, the medians and the pooled ratios. "
        "At 1.25 × and 1.5 × the forward the LC and CC calls carry the caveat of section V3: on the dates measured there a few paths on which one name ends above 3 times its spot carry a large share of these calls, "
        "and section V3 says the ratios to the copula at 1.5 × should not be quoted as model results. The same holds for the rows call 1.25× and call 1.5× of C.2 to C.5."
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------------- figures
def figure_f1(d: pd.DataFrame, base: Path = pc.PM) -> None:
    """LC, model S and the listed-variance forward over the copula's forward by entry date.  The
    study's monthly dates that have no row in the table are added as empty lines, so that a line
    is broken there as it is at a failed date; a date with a gap on both sides is drawn as a dot."""
    cols = ["date", "status", "flag_unscreened", "model_s_converged", "lc_over_copula", "lc_over_copula_se", "s_over_copula", "listed_fwd_ratio"]  # fmt: skip
    frame = d[cols].copy()
    frame["in_lc_table"] = True
    monthly_dates, _ = study_dates()
    have = set(frame["date"])
    gaps = pd.DataFrame({"date": [x for x in monthly_dates if x not in have], "in_lc_table": False})
    if len(gaps):
        frame = pd.concat([frame, gaps.astype(object)], ignore_index=True)
        frame = frame.sort_values("date").reset_index(drop=True)
    x = pd.to_datetime(frame["date"])
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.axhline(1.0, color="0.7", linewidth=0.6)
    for col, color, style, label in (
        ("lc_over_copula", "C0", "-", "LC"),
        ("s_over_copula", "C1", "-", "model S"),
        ("listed_fwd_ratio", "C2", "--", "listed-variance forward"),
    ):
        y = frame[col].astype(float)
        ax.plot(x, y, color=color, linewidth=1.0, linestyle=style, label=label)
        alone = y.notna() & y.shift(1).isna() & y.shift(-1).isna()
        ax.plot(x[alone], y[alone], color=color, linestyle="none", marker=".", markersize=2.5)
    ax.set_xlabel("entry date")
    ax.set_ylabel("forward / copula forward")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    pc.save_figure(fig, "F1_forward_over_copula", frame, base)
    plt.close(fig)


def figure_f2(
    table: dict[tuple[str, str], dict[str, dict[str, float]]], base: Path = pc.PM
) -> None:
    rows = []
    for model, stem in (
        ("LC", "C_lc_over_copula"),
        ("model S", "C_S_over_copula"),
        ("CC", "C_cc_over_copula"),
    ):
        for sample in ("S", "all", "S_unflagged", "all_unflagged"):
            for m in CALL_TAGS:
                stats = table[(f"{stem}_{m}", sample)]
                if not stats:
                    continue
                rows.append({
                    "strike_multiple": int(m) / 100.0, "model": model, "sample": sample, "n": stats["mean"]["n"], "mean": stats["mean"]["value"],
                    "mean_se_across_dates": stats["mean"]["se"], "q25": stats["q25"]["value"], "median": stats["median"]["value"], "q75": stats["q75"]["value"],
                    "pooled": stats["pooled"]["value"], "plotted": sample == "S" and model != "CC",
                })  # fmt: skip
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.axhline(1.0, color="0.7", linewidth=0.6)
    for model, color, marker, shift in (("LC", "C0", "o", -0.012), ("model S", "C1", "s", 0.012)):
        g = frame[frame["plotted"] & (frame["model"] == model)].sort_values("strike_multiple")
        at = g["strike_multiple"] + shift  # the two models side by side at each strike
        ax.vlines(at, g["q25"], g["q75"], color=color, linewidth=4.0, alpha=0.35)
        ax.plot(at, g["mean"], color=color, marker=marker, markersize=4, linewidth=1.0, label=f"{model}: mean (bar: interquartile range)")  # fmt: skip
    ax.set_xticks([int(m) / 100.0 for m in CALL_TAGS])
    ax.set_xlabel("strike / forward")
    ax.set_ylabel("call / copula call")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    pc.save_figure(fig, "F2_calls_over_copula_by_strike", frame, base)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--budget",
        choices=tuple(PASSES),
        default=DEVELOPMENT.budget,
        help="the pass behind the table: the budget its priced rows must have and the labels of the part (default: development, the frozen part's)",
    )
    parser.add_argument(
        "--rows",
        type=Path,
        default=None,
        help="the 3m table (default: the pass's, lcm_3m_dev_repair.parquet for the development budget and lcm_3m.parquet for the production one)",
    )
    parser.add_argument(
        "--logs",
        type=Path,
        default=None,
        help="the folder of the logs of the run behind the table, one per date (default: the pass's)",
    )
    parser.add_argument(
        "--base",
        type=Path,
        default=pc.PM,
        help="the folder written to: parts/, tables/ and figures/ under it (default: the package, which refuses its frozen files)",
    )
    parser.add_argument(
        "--no-status",
        action="store_true",
        help="do not append a line to STATUS.md (a rerun that changes nothing)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    run = PASSES[args.budget]
    rows_path = run.rows if args.rows is None else args.rows
    logs = run.logs if args.logs is None else args.logs
    base = args.base.resolve()
    frame = load(rows_path)
    priced = frame["priced"]
    budget = frame.loc[priced, ["n_particles", "n_paths", "companion_paths"]].drop_duplicates()
    if budget.to_numpy().tolist() != [list(run.sizes)]:
        raise ValueError(
            f"the priced rows are not all at the {run.budget} budget: {budget.to_dict('records')}"
        )
    if run is not DEVELOPMENT and DROPPED_CALENDAR not in frame.columns:
        raise ValueError(
            f"{rows_path.name} has no column {DROPPED_CALENDAR}: not a pass with decision 5 in its code"
        )
    commits = sorted(frame.loc[priced, "git_commit"].dropna().unique())
    commit = ", ".join(commits)
    d = by_date(frame)
    masks, sample_table = samples(d)
    qs = quantities()
    table, summary = all_summaries(d, masks, qs)
    fits, regression = regressions(d, masks)
    agreement = compare_check_d(fits, rows_path.name)
    LOG.info("%s", agreement[0])

    first = ["date", "status", "status_stored", "reason", "reason_stored", "T", "half", "clip_tercile"]  # fmt: skip
    out = d[[*first, *(c for c in d.columns if c not in first and c != "priced")]]
    pc.save_table(out, "C_history_by_date", base)
    pc.save_table(summary, "C_history_summaries", base)
    pc.save_table(sample_table, "C_history_samples", base)
    pc.save_table(regression, "C_history_regression", base)
    figure_f1(d, base)
    figure_f2(table, base)
    book = Book(commit, f"outputs/dispersion_lc/{rows_path.name}", run.budget)
    markdown = build_markdown(
        d, masks, sample_table, qs, table, fits, agreement, book, rows_path, run, logs
    )
    pc.write_part(PART, list(book.records.values()), markdown, base)
    n_priced, n_failed = int(priced.sum()), int((~priced).sum())
    line = (
        f"C_history: parts/{PART}.md/.json ({len(book.records)} records), tables/C_history_by_date.csv, C_history_summaries.csv, C_history_samples.csv, "
        f"C_history_regression.csv, figures F1_forward_over_copula and F2_calls_over_copula_by_strike (pdf + csv) from {rows_path.name} "
        f"({n_priced} priced, {n_failed} failed, commit {commit}, {run.budget} budget)"
    )
    # another folder than the package is not the package's part: no line in its STATUS.md
    if not args.no_status and base == pc.PM.resolve():
        pc.status(line)
    LOG.info(
        "section C written under %s: %d dates, %d priced, %d records",
        base,
        len(d),
        n_priced,
        len(book.records),
    )


if __name__ == "__main__":
    main()
