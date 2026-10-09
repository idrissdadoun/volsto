"""Local correlation model (SPEC §8.7, M12 part LC7): the comparison report of a sweep.

    python scripts/lcm_report.py --tenor 3m|12m|24m [--budget production|development]
        [--root <dir>] [--config <yaml>] [--tag <name>] [--against <name>]

Reads the sweep's table (``outputs/dispersion_lc/lcm_<tenor>[_dev][_<tag>].parquet``, one row
per date, ``scripts/disp_lcm.py``) and, read-only, the study's own tables
(``entries_<tenor>.parquet``: the copula's prices; ``model_s_3m.parquet``: model S at 3m), and
writes ``outputs/dispersion_lc/report_<stem>.md`` (``<stem> = <tenor>[_dev][_<tag>]``), its
by-date companion ``report_<stem>_by_date.csv`` and its figures in
``outputs/dispersion_lc/figures/``.  It runs on a partial table.

What is compared (no pass or fail beyond the gates of section 6):

* the forward, like for like (owner's check (b) of 2026-10-09): model S is measured over the
  study's copula (``P_D_S/P_D``), so ``E_LC[D]/P_D`` and ``E_CC[D]/P_D`` stand beside it, with
  the listed-variance forward ``√(EQV/EV)`` (check (c)); ``E_LC[D]/E_CC[D]`` (paired) is M12
  against its own companion;
* the calls at the study's cash strikes ``K_050 … K_200``: ``C_LC/C_CC`` against ``C_S/C``;
* per quantity: mean, quartiles, the median Monte Carlo standard error of a date, the
  correlation across dates of the two ratios, and the dates where LC and model S disagree in
  direction;
* the wing: ``ED_wing/ED_cc`` and ``ED_eqv/ED_cc`` beside ``E_LC[D]/E_CC[D]``, and how often the
  clipped mass inside ±2.5 sd exceeds 1 % (the index downside wing the model cannot reach);
* check (d): the regression across dates of ``LC/CC − P_D_S/P_D`` on the clipped mass inside
  ±2.5 sd and on the basket part of the ``E[V]`` split (``ols``);
* ``κ = E[D]/√E[V]`` under LC, CC and the copula; the gates and the names' diagnostic; the
  timings.

Which rows enter a summary (owner's decisions 2 and 5 of 2026-10-09, ``subsets``): every summary
is given with and without the dates on which a name is kept unscreened (``flag_unscreened``);
at 12m and 24m the summaries are on the rows without a clip flag (``flag_clip``), and the same
summary on all priced rows and on the rows without the low-side flag only is printed for
information.  A flagged row, a FAIL or a failed date stays in the table and in the by-date file;
the failed dates and their reasons are listed in full.

Rows written before 2026-10-09 do not carry the columns of the checks: ``derive`` computes them
from the base columns.

``--against <name>`` (``''`` = the main table) also writes ``report_<stem>_vs_<other>.md``, the
pass against another pass of the same tenor and budget (``compare``); a tagged pass is compared
with the main table when ``--against`` is not given.

Tests: ``tests/test_lcm_report.py``.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

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
# a date is flagged when the clipped mass inside ±2.5 sd at λ = 0, or at the cap, exceeds this
# (owner's decision 5 of 2026-10-09; the same constant as scripts/lcm_price.py)
CLIP_FLAG_MASS = 0.01
# the tenors whose summaries leave the clip-flagged rows out (decision 5); at 3m the cap binds on
# almost every date and the flags are reported in the wing section only
CLIP_SUMMARY_TENORS = ("12m", "24m")
# the tenors run on the reference dates only (decision 5)
INDICATIVE_TENORS = ("24m",)
# the checks that gate a date, and the names' check: a reported diagnostic (decision 3)
GATES = (
    ("check_no_nan", "no NaN"),
    ("check_forward", "E[B_T] within 3 standard errors of its forward"),
    ("check_index", "index ATM and 90 % vols within 0.15 vp, unless the clipped mass inside ±2.5 sd exceeds 1 %"),
)  # fmt: skip
DIAGNOSTIC = ("check_names", "Σ w E[R_i²] within 2 % of the listed strips")
# the study's columns ``load`` joins to the table (the entry of basket B1)
JOINED = ("P_D_cop", "P_D_se_cop", "EV_cop", "EQV_cop", "kappa_cop_cop")
# the base columns the report reads: one a table does not carry is reported as n/a, not an error
BASE = (
    "status", "reason", "git_commit", "n_particles", "n_paths", "threads", "cache_hit",
    "ED_lc", "ED_lc_se", "ED_cc", "ED_cc_se", "ratio", "ratio_se",
    *[f"C_{t}_{s}" for t in lp.MULT_TAGS for s in ("lc", "lc_se", "ratio", "ratio_se")],
    "clip_inner_max", "clip_high_inner_max", "clip_low_inner_max", "clip_high_max",
    "EV_over_EQV", "EV_over_EQV_se", "ED_wing_ratio", "ED_wing_ratio_se", "ED_eqv_ratio", "ED_eqv_ratio_se",
    "idx_err_90", "idx_err_90_se", "idx_err_m15", "idx_err_m15_se", "idx_err_atm", "idx_err_atm_se",
    "kappa_lc", "kappa_lc_se", "kappa_cc", "kappa_cc_se", "lambda_c", "rho_cc", "rho_cop",
    "check_no_nan", "check_forward", "check_index", "check_names",
    "index_extrapolated", "n_names_extrapolated", "n_dropped", "align_flagged",
    "EQV", "EV_copula", "P_D_copula", "M_B_listed", "E_Rbar2_lc", "E_Rbar2_lc_se",
    *JOINED,
)  # fmt: skip
# the columns of the checks and decisions of 2026-10-09 (those of the pricer's DERIVED_COLUMNS):
# ``derive`` computes the ones a table does not carry
DERIVED = (
    "listed_fwd_ratio", "listed_fwd", "lc_over_copula", "lc_over_copula_se", "cc_over_copula",
    "cc_over_copula_se", "flag_unscreened", "flag_clip_low", "flag_clip_high", "flag_clip",
    "indicative",
)  # fmt: skip
WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")
# LC, model S, CC: the first three slots of the categorical palette (distinct pairwise, also for
# colour-blind readers); the listed-variance forward is a reference line in grey
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "0.45"


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


# ---------------------------------------------------------------------------------------------
# the columns of the checks, the flags and the rows of a summary
# ---------------------------------------------------------------------------------------------


def truth(x: pd.Series) -> pd.Series:
    """A column of flags as booleans; an absent value (a failed row) is ``False``."""
    return pd.Series([bool(v) if pd.notna(v) else False for v in x], index=x.index, dtype=bool)


def column(frame: pd.DataFrame, *names: str) -> pd.Series:
    """The first of the named columns the frame carries, as numbers, the later ones filling its
    gaps; NaN where none of them has a value."""
    out = pd.Series(np.nan, index=frame.index, dtype=float)
    for name in names:
        if name in frame.columns:
            out = out.where(out.notna(), pd.to_numeric(frame[name], errors="coerce").astype(float))
    return out


def complete(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The frame with every column of ``BASE``, and the names of those it did not carry (added
    as NaN: the report prints n/a for them instead of failing)."""
    absent = [c for c in BASE if c not in frame.columns]
    out = frame.copy()
    for c in absent:
        out[c] = np.nan
    return out, absent


def derive(frame: pd.DataFrame) -> pd.DataFrame:
    """The columns of the owner's checks and decisions of 2026-10-09 (SPEC §8.7), computed from
    the base columns wherever the table does not carry them (rows written before that date); a
    value the pricer wrote is kept.  The definitions are those of
    ``scripts/lcm_price.py::derived_columns``::

        lc_over_copula   = ED_lc / P_D_copula          lc_over_copula_se = ED_lc_se / P_D_copula
        cc_over_copula   = ED_cc / P_D_copula          cc_over_copula_se = ED_cc_se / P_D_copula
        listed_fwd_ratio = √(EQV / EV_copula)          listed_fwd = P_D_copula · listed_fwd_ratio
        flag_unscreened  = n_names_unscreened > 0      (False without that column)
        flag_clip_low    = clip_low_inner_max  > 0.01  flag_clip_high = clip_high_inner_max > 0.01
        flag_clip        = flag_clip_low or flag_clip_high
        indicative       = tenor == "24m"

    ``listed_fwd`` is the copula's ``κ_cop = P_D/√EV`` times ``√EQV`` (check (c)); the standard
    errors of check (b) are the Monte Carlo errors of the numerators, the copula's own error
    (``P_D_se`` of the entry) not included.  The copula's ``P_D``, ``EV`` and ``EQV`` are the
    row's, or the study's entry joined by ``load`` (``*_cop``) where the row has none (a failed
    date).  Added for check (d)::

        EV_basket_part = E_Rbar2_lc − M_B_listed       (the row's, when it carries it)
        EV_basket_rel  = EV_basket_part / M_B_listed   (se: E_Rbar2_lc_se / M_B_listed)

    and, with model S joined (``P_D_S``, ``converged``), ``S_ratio = P_D_S/P_D`` and
    ``S_C_<m> = C_S_<m>/C_<m>`` on its converged dates and ``gap_to_S = ratio − S_ratio``.
    Test: ``tests/test_lcm_report.py::test_derived_columns`` and
    ``::test_derived_columns_agree_with_the_pricer``."""
    out = frame.copy()
    p_d = column(out, "P_D_copula", "P_D_cop")
    p_d = p_d.where(p_d > 0.0)
    ev = column(out, "EV_copula", "EV_cop")
    eqv = column(out, "EQV", "EQV_cop")
    ratio = np.sqrt(eqv.where(eqv >= 0.0) / ev.where(ev > 0.0))
    m_b = column(out, "M_B_listed")
    computed = {
        "listed_fwd_ratio": ratio,
        "listed_fwd": p_d * ratio,
        "lc_over_copula": column(out, "ED_lc") / p_d,
        "lc_over_copula_se": column(out, "ED_lc_se") / p_d,
        "cc_over_copula": column(out, "ED_cc") / p_d,
        "cc_over_copula_se": column(out, "ED_cc_se") / p_d,
        "EV_basket_part": column(out, "E_Rbar2_lc") - m_b,
        "EV_basket_part_se": column(out, "E_Rbar2_lc_se"),
    }
    for name, values in computed.items():
        given = column(out, name)
        out[name] = given.where(given.notna(), values)
    out["EV_basket_rel"] = out["EV_basket_part"] / m_b.where(m_b > 0.0)
    out["EV_basket_rel_se"] = out["EV_basket_part_se"] / m_b.where(m_b > 0.0)
    flags = {
        "flag_unscreened": column(out, "n_names_unscreened") > 0,
        "flag_clip_low": column(out, "clip_low_inner_max") > CLIP_FLAG_MASS,
        "flag_clip_high": column(out, "clip_high_inner_max") > CLIP_FLAG_MASS,
    }
    for name, values in flags.items():
        out[name] = _flag(out, name, values)
    out["flag_clip"] = _flag(out, "flag_clip", out["flag_clip_low"] | out["flag_clip_high"])
    tenor = out["tenor"] if "tenor" in out.columns else pd.Series("", index=out.index)
    out["indicative"] = _flag(out, "indicative", tenor.isin(INDICATIVE_TENORS))
    if "P_D_S" in out.columns:
        conv = truth(out["converged"])
        out["S_ratio"] = (column(out, "P_D_S") / p_d).where(conv)
        for t in TAGS:
            c_cop = column(out, f"C_{t}_cop")
            out[f"S_C_{t}"] = (column(out, f"C_S_{t}") / c_cop.where(c_cop > 0.0)).where(conv)
        out["gap_to_S"] = column(out, "ratio") - out["S_ratio"]
    return out


def _flag(frame: pd.DataFrame, name: str, computed: pd.Series) -> pd.Series:
    """The flag the table carries, the computed one where it carries none."""
    if name not in frame.columns:
        return truth(computed)
    given = frame[name].astype(object)
    return truth(given.where(given.notna(), computed.astype(object)))


class Subset(NamedTuple):
    """The rows of one summary: a caption, a mask over the priced rows, and whether the summary
    is printed for information only."""

    label: str
    mask: pd.Series
    information: bool


def subsets(ok: pd.DataFrame, tenor: str) -> list[Subset]:
    """The rows every summary of the report is computed on, among the priced rows ``ok`` (owner's
    decisions 2 and 5 of 2026-10-09).

    * 3m: all priced dates, and the same without the dates on which a name is kept unscreened
      (``flag_unscreened``).  The clip flags remove nothing at 3m.
    * 12m and 24m: the reported summary is on the rows without ``flag_clip`` (the clipped mass
      inside ±2.5 sd at ``λ = 0`` or at the cap above 1 %), with and without the unscreened
      dates; then, for information only, the same summary on all priced rows and on the rows
      without ``flag_clip_low`` only.

    The "without the flagged dates" subset is listed only when a date is flagged: without one
    the two summaries are the same rows.  Test:
    ``tests/test_lcm_report.py::test_subsets_drop_exactly_the_flagged_rows``."""
    unscreened = truth(ok["flag_unscreened"])
    everything = pd.Series(True, index=ok.index, dtype=bool)
    if tenor not in CLIP_SUMMARY_TENORS:
        out = [Subset("all priced dates", everything, False)]
        if unscreened.any():
            out.append(
                Subset(
                    f"without the {count(int(unscreened.sum()))} on which a name is kept unscreened",
                    ~unscreened,
                    False,
                )
            )
        return out
    clip, low = truth(ok["flag_clip"]), truth(ok["flag_clip_low"])
    out = [Subset("priced dates without a clip flag (the reported summary)", ~clip, False)]
    if (unscreened & ~clip).any():
        out.append(
            Subset(
                f"priced dates without a clip flag and without the {count(int((unscreened & ~clip).sum()))} "
                "on which a name is kept unscreened",
                ~clip & ~unscreened,
                False,
            )
        )
    out.append(
        Subset(
            f"for information only: all priced dates, the {int(clip.sum())} clip-flagged included",
            everything,
            True,
        )
    )
    out.append(
        Subset(
            f"for information only: priced dates without the low-side clip flag ({int(low.sum())} removed; "
            "the cap-side flags stay in)",
            ~low,
            True,
        )
    )
    return out


# ---------------------------------------------------------------------------------------------
# the regression of check (d)
# ---------------------------------------------------------------------------------------------


def ols(y: np.ndarray, x: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    """Ordinary least squares of ``y`` (n values) on an intercept and the columns of ``x``
    (n × p), with classical and heteroskedasticity-robust standard errors.

    With ``Z = [1, x]`` (n × k, k = p + 1), ``b = (Z'Z)⁻¹ Z'y`` and ``e = y − Z b``:

    * classical covariance ``s² (Z'Z)⁻¹`` with ``s² = e'e / (n − k)``;
    * HC1 covariance (White 1980, with the degrees-of-freedom factor of MacKinnon and White
      1985): ``n/(n − k) · (Z'Z)⁻¹ Z' diag(e²) Z (Z'Z)⁻¹``;
    * ``t = b / se`` under each; ``R² = 1 − e'e / Σ (y − ȳ)²``.

    ``Z = QR`` (thin QR): ``b = R⁻¹ Q'y`` and ``(Z'Z)⁻¹ = R⁻¹ R⁻ᵀ``, so a regressor of small
    scale (a variance of order 10⁻⁴) does not square the conditioning.  Raises ``ValueError``
    when ``n ≤ k`` or the design is rank deficient.  Returns ``names`` (the intercept first),
    ``coef``, ``se``, ``t``, ``se_hc1``, ``t_hc1`` (arrays of length k), ``r2``, ``n``, ``k``,
    ``s2``.  Test: ``tests/test_lcm_report.py::test_ols_recovers_known_coefficients``,
    ``::test_ols_hand_example`` and ``::test_ols_hc1_matches_the_direct_formula``."""
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).reshape(len(y), -1)
    n, k = len(y), x.shape[1] + 1
    if len(names) != k - 1:
        raise ValueError(f"{len(names)} names for {k - 1} regressors")
    if n <= k:
        raise ValueError(f"{n} observations for {k} coefficients")
    z = np.column_stack([np.ones(n), x])
    if np.linalg.matrix_rank(z) < k:
        raise ValueError("the design is rank deficient")
    q, r = np.linalg.qr(z)
    r_inv = np.linalg.inv(r)
    coef = r_inv @ (q.T @ y)
    e = y - z @ coef
    zz_inv = r_inv @ r_inv.T
    s2 = float(e @ e) / (n - k)
    meat = (z * (e**2)[:, None]).T @ z
    se = np.sqrt(s2 * np.diag(zz_inv))
    se_hc1 = np.sqrt(n / (n - k) * np.diag(zz_inv @ meat @ zz_inv))
    total = float(np.sum((y - y.mean()) ** 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        t, t_hc1 = coef / se, coef / se_hc1
    return {
        "names": ["intercept", *names], "coef": coef, "se": se, "t": t, "se_hc1": se_hc1, "t_hc1": t_hc1,
        "r2": 1.0 - float(e @ e) / total if total > 0.0 else float("nan"), "n": n, "k": k, "s2": s2,
    }  # fmt: skip


def regress(frame: pd.DataFrame, y: str, xs: Sequence[str]) -> dict[str, Any] | None:
    """``ols`` of the column ``y`` on the columns ``xs`` over the rows where all are finite;
    ``None`` when there are too few of them or a regressor does not vary (a partial table)."""
    data = pd.concat([column(frame, c) for c in (y, *xs)], axis=1, keys=[y, *xs])
    data = data.replace([np.inf, -np.inf], np.nan).dropna()
    try:
        return ols(data[y].to_numpy(), data[list(xs)].to_numpy(), list(xs))
    except ValueError:
        return None


def regression_table(fits: Sequence[tuple[str, dict[str, Any] | None]], labels: dict[str, str]) -> str:  # fmt: skip
    """The fits as one markdown table: a line per coefficient, ``R²`` and ``n`` on the first."""
    lines = [
        "| fit | regressor | coefficient | se (classical) | t (classical) | se (HC1) | t (HC1) | R² | n |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, fit in fits:
        if fit is None:
            lines.append(f"| {name} | n/a: too few dates, or a regressor that does not vary | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")  # fmt: skip
            continue
        for j, reg in enumerate(fit["names"]):
            lines.append(
                f"| {name if j == 0 else ''} | {labels.get(reg, reg)} | {fit['coef'][j]:+.4f} | {fit['se'][j]:.4f} | "
                f"{fit['t'][j]:+.2f} | {fit['se_hc1'][j]:.4f} | {fit['t_hc1'][j]:+.2f} | "
                + (f"{fit['r2']:.3f} | {fit['n']} |" if j == 0 else "  |  |")
            )
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# the tables read
# ---------------------------------------------------------------------------------------------


def stem_of(tenor: str, budget: str, label: str = "") -> str:
    """``<tenor>[_dev][_<label>]``: the suffix of a pass's table, report and figures."""
    return (tenor if budget == "production" else f"{tenor}_dev") + (f"_{label}" if label else "")


def load(
    tenor: str, budget: str, out: Path, label: str = "", study: Path = dd.OUT
) -> tuple[pd.DataFrame, Path]:
    """The pass's table by date, with the study's entry of basket B1 joined (``*_cop``) and, at
    3m, model S; ``study`` is the study's output folder (read-only)."""
    path = out / f"lcm_{stem_of(tenor, budget, label)}.parquet"
    rows = pd.read_parquet(path)
    entries = pd.read_parquet(study / f"entries_{tenor}.parquet")
    entries = entries[entries["basket"] == "B1"].drop_duplicates("date").set_index("date")
    keep = [
        c
        for c in ("P_D", "P_D_se", "EV", "EQV", "kappa_cop", *[f"C_{t}" for t in TAGS])
        if c in entries.columns
    ]
    frame = rows.set_index("date").join(entries[keep].add_suffix("_cop"), how="left")
    if tenor == "3m":
        s = pd.read_parquet(study / "model_s_3m.parquet").set_index("date")
        frame = frame.join(s[["converged", "P_D_S", *[f"C_S_{t}" for t in TAGS]]], how="left")
    frame.index = pd.to_datetime(frame.index)
    return frame.sort_index(), path


def study_line(study: Path, tenor: str) -> dict[str, Any] | None:
    """Owner's check (c) from the study's own tables alone (no number of this model): on the
    dates with a converged model S, the mean across dates of the listed-variance forward over
    the copula's, ``√(EQV/EV)`` (``entries_<tenor>.parquet``, basket B1), and of model S over the
    copula, ``P_D_S/P_D`` (``model_s_<tenor>.parquet``), each with the standard error of the
    mean; and both on the study's last date.  ``None`` without a model S table."""
    path = study / f"model_s_{tenor}.parquet"
    if not path.exists():
        return None
    entries = pd.read_parquet(study / f"entries_{tenor}.parquet")
    entries = entries[entries["basket"] == "B1"].drop_duplicates("date").set_index("date")
    s = pd.read_parquet(path).set_index("date")
    both = s[["converged", "P_D_S"]].join(entries[["P_D", "EV", "EQV"]], how="left")
    conv = truth(both["converged"])
    listed = np.sqrt(both["EQV"] / both["EV"])
    ratio = (both["P_D_S"] / both["P_D"]).where(conv)
    used = pd.concat([listed, ratio], axis=1, keys=["listed", "s"])[conv].dropna()
    last = entries.index.max()
    n = len(used)
    return {
        "n": n, "n_converged": int(conv.sum()), "n_model_s": len(s),
        "mean_listed": float(used["listed"].mean()) if n else np.nan,
        "se_listed": float(used["listed"].std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
        "mean_s": float(used["s"].mean()) if n else np.nan,
        "se_s": float(used["s"].std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
        "last": str(last)[:10],
        "listed_last": float(np.sqrt(entries.loc[last, "EQV"] / entries.loc[last, "EV"])),
        "s_last": float(ratio.loc[last]) if last in ratio.index else np.nan,
    }  # fmt: skip


def by_date(frame: pd.DataFrame, tenor: str) -> pd.DataFrame:
    """One line per date of the table (derived), the failed ones included: status, reason and
    flags, the forward's ratios with their Monte Carlo errors, the clipped mass inside ±2.5 sd
    by side, the basket part of the ``E[V]`` split (raw and over ``M_B^listed``), ``κ`` under the
    three models, and the checks.  ``in_summary`` says whether the row enters the reported
    summary (priced, and at 12m and 24m without a clip flag)."""
    priced = frame["status"] != "failed"
    names = frame["names_unscreened"] if "names_unscreened" in frame.columns else None
    flags = []
    for i in range(len(frame)):
        row = []
        if frame["flag_unscreened"].iloc[i]:
            kept = str(names.iloc[i]) if names is not None and pd.notna(names.iloc[i]) else ""
            row.append(f"unscreened({kept})" if kept else "unscreened")
        row += [k for k in ("clip_low", "clip_high") if frame[f"flag_{k}"].iloc[i]]
        if frame["indicative"].iloc[i]:
            row.append("indicative")
        flags.append(";".join(row))
    gates = pd.DataFrame({name: truth(frame[name]) for name, _ in GATES})
    out = pd.DataFrame({
        "date": [f"{d:%Y-%m-%d}" for d in frame.index],
        "status": frame["status"].to_numpy(),
        "reason": frame["reason"].fillna("").to_numpy(),
        "flags": flags,
        "in_summary": (priced & ~(frame["flag_clip"] & (tenor in CLIP_SUMMARY_TENORS))).to_numpy(),
        "lc_over_cc": frame["ratio"].to_numpy(),
        "lc_over_cc_se": frame["ratio_se"].to_numpy(),
        "lc_over_copula": frame["lc_over_copula"].to_numpy(),
        "lc_over_copula_se": frame["lc_over_copula_se"].to_numpy(),
        "cc_over_copula": frame["cc_over_copula"].to_numpy(),
        "cc_over_copula_se": frame["cc_over_copula_se"].to_numpy(),
        "model_s_over_copula": frame["S_ratio"].to_numpy() if "S_ratio" in frame.columns else np.nan,
        "listed_fwd_ratio": frame["listed_fwd_ratio"].to_numpy(),
        "clip_inner_max": frame["clip_inner_max"].to_numpy(),
        "clip_low_inner_max": frame["clip_low_inner_max"].to_numpy(),
        "clip_high_inner_max": frame["clip_high_inner_max"].to_numpy(),
        "EV_basket_part": frame["EV_basket_part"].to_numpy(),
        "EV_basket_part_se": frame["EV_basket_part_se"].to_numpy(),
        "EV_basket_part_over_M_B_listed": frame["EV_basket_rel"].to_numpy(),
        "EV_basket_part_over_M_B_listed_se": frame["EV_basket_rel_se"].to_numpy(),
        "kappa_lc": frame["kappa_lc"].to_numpy(),
        "kappa_lc_se": frame["kappa_lc_se"].to_numpy(),
        "kappa_cc": frame["kappa_cc"].to_numpy(),
        "kappa_cc_se": frame["kappa_cc_se"].to_numpy(),
        "kappa_copula": frame["kappa_cop_cop"].to_numpy(),
        "gates_failed": [
            ",".join(name for name, _ in GATES if not gates[name].iloc[i]) if priced.iloc[i] else ""
            for i in range(len(frame))
        ],
        "names_within_2pct": [
            bool(v) if p else "" for v, p in zip(truth(frame[DIAGNOSTIC[0]]), priced, strict=True)
        ],
    })  # fmt: skip
    return out


# ---------------------------------------------------------------------------------------------
# the sections: each is the summary of one set of rows
# ---------------------------------------------------------------------------------------------

Caption = Callable[[str], str]


def count(n: int, noun: str = "date", words: bool = False) -> str:
    """``1 date``, ``4 dates``; with ``words`` the number is spelled out up to ten."""
    number = WORDS[n] if words and 0 <= n < len(WORDS) else str(n)
    return f"{number} {noun}{'' if n == 1 else 's'}"


def dates_of(index: Sequence[Any]) -> str:
    """Every date of the index, never truncated."""
    return ", ".join(f"{d:%Y-%m-%d}" for d in index) or "none"


def forward_lines(sub: pd.DataFrame, cap: Caption, has_s: bool) -> list[str]:
    """Section 1: the forward over the copula (model S, LC, CC, the listed-variance forward) and
    LC over its CC companion."""
    rows: dict[str, dict[str, Any]] = {}
    if has_s:
        rows["model S / copula (P_D_S / P_D)"] = summary(sub["S_ratio"])
    rows["**E_LC[D] / copula P_D**"] = summary(sub["lc_over_copula"], sub["lc_over_copula_se"])
    rows["**E_CC[D] / copula P_D**"] = summary(sub["cc_over_copula"], sub["cc_over_copula_se"])
    rows["listed-variance forward / copula P_D, √(EQV/EV)"] = summary(sub["listed_fwd_ratio"])
    rows["E_LC[D] / E_CC[D] (paired)"] = summary(sub["ratio"], sub["ratio_se"])
    lines = [cap("The forward"), "", table(rows)]
    if not has_s:
        return lines
    same = sub[sub["S_ratio"].notna() & sub["ratio"].notna()]
    if len(same):
        lines += [
            "",
            f"On the {len(same)} of these dates with a converged model S, the means are: model S/copula {same['S_ratio'].mean():.4f}, "
            f"LC/copula {same['lc_over_copula'].mean():.4f}, CC/copula {same['cc_over_copula'].mean():.4f}, listed-variance forward/copula "
            f"{same['listed_fwd_ratio'].mean():.4f}, LC/CC {same['ratio'].mean():.4f}.",
        ]
    for what, col in (("LC/CC", "ratio"), ("LC/copula", "lc_over_copula")):
        a = agreement(sub[col], sub["S_ratio"])
        lines += [
            "",
            f"Across the {a['n']} dates with both: correlation of ({what} - 1) with (S/copula - 1) "
            f"{a['pearson']:+.3f} (Pearson), {a['spearman']:+.3f} (Spearman); opposite direction on "
            f"{a['n_opposite']} dates"
            + (
                ": "
                + ", ".join(d[:10] for d in a["dates"][:20])
                + (" ..." if a["n_opposite"] > 20 else "")
                if a["n_opposite"]
                else ""
            )
            + ".",
        ]
    return lines


def calls_lines(sub: pd.DataFrame, cap: Caption, has_s: bool) -> list[str]:
    """Section 2: the calls at the study's strikes."""
    rows: dict[str, dict[str, Any]] = {}
    notes = []
    for t in TAGS:
        rows[f"K_{t}: C_LC / C_CC"] = summary(sub[f"C_{t}_ratio"], sub[f"C_{t}_ratio_se"])
        if has_s:
            rows[f"K_{t}: C_S / C_copula"] = summary(sub[f"S_C_{t}"])
            a = agreement(sub[f"C_{t}_ratio"], sub[f"S_C_{t}"])
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
    lines = [cap("The calls"), "", table(rows)]
    if notes:
        lines += ["", "Agreement of the two ratios across dates:", "", *notes]
    return lines


def wing_lines(sub: pd.DataFrame, cap: Caption, has_s: bool) -> list[str]:
    """Section 3: the clipped mass, the wing-corrected forwards and the index errors."""
    binds = sub["clip_inner_max"] > CLIP_FLAG_MASS
    high, low = truth(sub["flag_clip_high"]), truth(sub["flag_clip_low"])
    rows = {
        "clipped mass inside ±2.5 sd, max over slices": summary(sub["clip_inner_max"]),
        "of which on the high side (λ at its cap)": summary(sub["clip_high_inner_max"]),
        "of which on the low side (λ at 0)": summary(sub["clip_low_inner_max"]),
        "clipped mass, whole cloud, high side, max": summary(sub["clip_high_max"]),
        "E_LC[V] / listed E_Q[V]": summary(sub["EV_over_EQV"], sub["EV_over_EQV_se"]),
        "basket part of the E[V] split over M_B^listed, E_LC[R̄²]/M_B^listed - 1": summary(sub["EV_basket_rel"], sub["EV_basket_rel_se"]),
        "ED_wing / E_CC[D]": summary(sub["ED_wing_ratio"], sub["ED_wing_ratio_se"]),
        "ED_eqv / E_CC[D]": summary(sub["ED_eqv_ratio"], sub["ED_eqv_ratio_se"]),
        "E_LC[D] / E_CC[D]": summary(sub["ratio"], sub["ratio_se"]),
        "index error at the 90 % strike (vp)": summary(sub["idx_err_90"], sub["idx_err_90_se"]),
        "index error at -1.5 sd (vp)": summary(sub["idx_err_m15"], sub["idx_err_m15_se"]),
        "index error at the money (vp)": summary(sub["idx_err_atm"], sub["idx_err_atm_se"]),
    }  # fmt: skip
    lines = [cap("The wing"), "", table(rows), ""]
    lines.append(
        f"The clipped mass inside ±2.5 sd exceeds 1 % on {int(binds.sum())} of {len(sub)} dates ({binds.mean():.1%}) — on the high side "
        f"(the index target asks for more correlation than `rho_max` allows, the downside wing; `flag_clip_high`) on {int(high.sum())}, on the low side "
        f"(less than `rho_min` allows; `flag_clip_low`) on {int(low.sum())}: on those dates the smile is reported, not gated. "
        "`ED_wing = κ_LC·√(Σ w E_LC[R_i²] - M_B^listed)` replaces the model's basket second moment by the listed one; "
        "`ED_eqv = κ_LC·√EQV` replaces both. Where the wing binds: "
        f"mean LC/CC {sub.loc[binds, 'ratio'].mean():.4f}, ED_wing/CC {sub.loc[binds, 'ED_wing_ratio'].mean():.4f}, ED_eqv/CC {sub.loc[binds, 'ED_eqv_ratio'].mean():.4f} (n {int(binds.sum())}); "
        f"where it does not: {sub.loc[~binds, 'ratio'].mean():.4f}, {sub.loc[~binds, 'ED_wing_ratio'].mean():.4f}, {sub.loc[~binds, 'ED_eqv_ratio'].mean():.4f} (n {int((~binds).sum())})."
    )
    if not has_s:
        return lines
    lines += [
        "",
        cap("Against model S across dates (the effect is the ratio minus 1; model S over its copula)"),
        "",
        "| LC quantity | dates | mean effect | model S, same dates | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|---:|",
    ]  # fmt: skip
    free = sub["clip_inner_max"] < 0.05
    for label, col, mask in (
        ("E_LC[D] / E_CC[D], all dates", "ratio", sub["ratio"].notna()),
        ("E_LC[D] / E_CC[D], clipped mass inside ±2.5 sd below 5 %", "ratio", free),
        ("E_LC[D] / E_CC[D], clipped mass 5 % or more", "ratio", ~free),
        ("E_LC[D] / copula P_D, all dates", "lc_over_copula", sub["ratio"].notna()),
        ("ED_wing / E_CC[D], all dates", "ED_wing_ratio", sub["ratio"].notna()),
        ("ED_eqv / E_CC[D], all dates", "ED_eqv_ratio", sub["ratio"].notna()),
        ("listed-variance forward / copula P_D, all dates", "listed_fwd_ratio", sub["ratio"].notna()),
    ):  # fmt: skip
        both = pd.concat(
            [sub.loc[mask, col] - 1.0, sub.loc[mask, "S_ratio"] - 1.0], axis=1, keys=["a", "b"]
        ).dropna()
        if len(both) < 3:
            lines.append(f"| {label} | {len(both)} | n/a | n/a | n/a | n/a |")
            continue
        lines.append(
            f"| {label} | {len(both)} | {both['a'].mean():+.4f} | {both['b'].mean():+.4f} | "
            f"{both['a'].corr(both['b']):+.3f} | {both['a'].corr(both['b'], method='spearman'):+.3f} |"
        )
    both = pd.concat([sub["clip_inner_max"], sub["S_ratio"] - 1.0], axis=1, keys=["a", "b"]).dropna()  # fmt: skip
    if len(both) >= 3:
        lines += [
            "",
            f"Correlation across dates of the clipped mass inside ±2.5 sd with model S's effect: {both['a'].corr(both['b']):+.3f} "
            f"(n {len(both)}): the dates on which model S moves the forward most are those on which the model is clipped most.",
        ]
    return lines


def gap_lines(sub: pd.DataFrame, cap: Caption) -> list[str]:
    """Section 4, owner's check (d): ``LC/CC − P_D_S/P_D`` regressed across dates on the clipped
    mass inside ±2.5 sd and on the basket part of the ``E[V]`` split (``ols``), the basket part
    over ``M_B^listed`` in the first table and raw in the second."""
    y, clip, rel, raw = "gap_to_S", "clip_inner_max", "EV_basket_rel", "EV_basket_part"
    labels = {
        "intercept": "intercept",
        clip: "clipped mass inside ±2.5 sd",
        rel: "basket part / M_B^listed",
        raw: "basket part, raw",
    }
    used = sub[[y, clip, rel, raw]].replace([np.inf, -np.inf], np.nan).dropna()
    lines = [
        cap(
            "Regression of LC/CC - P_D_S/P_D, the basket part over M_B^listed (the one of the table)"
        ),
        "",
        regression_table(
            [
                ("both regressors", regress(sub, y, [clip, rel])),
                ("clipped mass alone", regress(sub, y, [clip])),
                ("basket part alone", regress(sub, y, [rel])),
            ],
            labels,
        ),
        "",
        cap("The same with the basket part raw, E_LC[R̄²] - M_B^listed (a variance, not a ratio)"),
        "",
        regression_table(
            [
                ("both regressors", regress(sub, y, [clip, raw])),
                ("basket part alone", regress(sub, y, [raw])),
            ],
            labels,
        ),
    ]
    if len(used) >= 3:
        noise = float(sub.loc[used.index, "EV_basket_rel_se"].median())
        spread = float(used[rel].std(ddof=1))
        lines += [
            "",
            f"On these {len(used)} dates (mean, standard deviation across dates): LC/CC - P_D_S/P_D {used[y].mean():+.4f}, {used[y].std(ddof=1):.4f}; "
            f"clipped mass {used[clip].mean():.4f}, {used[clip].std(ddof=1):.4f}; basket part over M_B^listed {used[rel].mean():+.4f}, {spread:.4f}; "
            f"basket part raw {used[raw].mean():+.3e}, {used[raw].std(ddof=1):.3e}. Correlation of the two regressors: {used[clip].corr(used[rel]):+.3f} "
            f"(basket part over M_B^listed), {used[clip].corr(used[raw]):+.3f} (raw). The basket part is a Monte Carlo number: its median standard error "
            f"on one date is {noise:.4f} (over M_B^listed), the square of which is {(noise / spread) ** 2 if spread > 0 else float('nan'):.1%} of its variance across dates; "
            f"the median Monte Carlo error of LC/CC is {float(sub.loc[used.index, 'ratio_se'].median()):.5f} (model S's own error is not in the study's table).",
        ]
    return lines


def kappa_lines(sub: pd.DataFrame, cap: Caption) -> list[str]:
    """Section 5: κ under the three models, the correlations, and the deltas when priced."""
    rows = {
        "κ LC": summary(sub["kappa_lc"], sub["kappa_lc_se"]),
        "κ CC": summary(sub["kappa_cc"], sub["kappa_cc_se"]),
        "κ copula": summary(sub["kappa_cop_cop"]),
        "λ_c": summary(sub["lambda_c"]),
        "rho_CC": summary(sub["rho_cc"]),
        "copula rho": summary(sub["rho_cop"]),
    }
    if "delta_fwd_lc" in sub.columns:
        rows.update({
            "forward: sticky-strike delta LC (% per +1 %)": summary(sub["delta_fwd_lc"], sub["delta_fwd_lc_se"]),
            "forward: sticky-strike delta CC": summary(sub["delta_fwd_cc"], sub["delta_fwd_cc_se"]),
            "forward: skew channel (CC_ss - 1)": summary(sub["delta_skew_channel"], sub["delta_skew_channel_se"]),
            "forward: correlation channel (LC_ss - CC_ss)": summary(sub["delta_correlation_channel"], sub["delta_correlation_channel_se"]),
            "call K_100: sticky-strike delta LC": summary(sub["delta_C100_lc"], sub["delta_C100_lc_se"]),
            "call K_100: sticky-strike delta CC": summary(sub["delta_C100_cc"], sub["delta_C100_cc_se"]),
        })  # fmt: skip
    return [cap("κ"), "", table(rows)]


def checks_lines(sub: pd.DataFrame, cap: Caption, absent: Sequence[str]) -> list[str]:
    """Section 6: the gates (every date that does not pass is named), the names' 2 % check as a
    reported diagnostic (owner's decision 3: its count stays, it gates nothing), the screen."""
    lines = [cap("Gates"), "", "| gate | dates that pass | dates that do not |", "|---|---:|---|"]
    passes = pd.Series(True, index=sub.index, dtype=bool)
    for name, text in GATES:
        if name in absent:
            lines.append(f"| {text} | n/a (the table has no `{name}`) | n/a |")
            continue
        good = truth(sub[name])
        passes &= good
        lines.append(f"| {text} | {int(good.sum())} of {len(sub)} | {dates_of(sub.index[~good])} |")
    name, text = DIAGNOSTIC
    if name in absent:
        lines += ["", f"The names' diagnostic: the table has no `{name}`."]
    else:
        within = truth(sub[name])
        checked = sub["status"] == "check"
        lines += [
            "",
            f"**Reported diagnostic, not a gate (owner's decision 3 of 2026-10-09):** {text} on {int(within.sum())} of {len(sub)} dates; "
            f"of the {int((~within).sum())} dates outside 2 %, {int((~within & passes).sum())} pass every gate. The dates are in the by-date file "
            "(column `names_within_2pct`)."
            + (
                f" Status `check` is carried by {int(checked.sum())} of these dates in the table; {int((checked & passes).sum())} of them pass every gate "
                "and owe their status to the names' diagnostic alone (rows written when it still set the status)."
                if checked.any()
                else ""
            ),
        ]
    extra = ""
    if "n_dropped_calendar" in sub.columns:
        dropped = column(sub, "n_dropped_calendar")
        extra += f" Names' slices dropped by the calendar repair: median {dropped.median():.0f}, max {dropped.max():.0f} per date."
    if "n_dropped_calendar_index" in sub.columns:
        dropped = column(sub, "n_dropped_calendar_index")
        extra += f" DJX slices dropped by the calendar repair: on {int((dropped > 0).sum())} dates, max {dropped.max():.0f}."
    lines += [
        "",
        f"Index extrapolated beyond its last kept expiry on {int(truth(sub['index_extrapolated']).sum())} dates; a name priced beyond its last "
        f"listed expiry on {int((sub['n_names_extrapolated'] > 0).sum())} dates; expiries dropped by the screen: median {sub['n_dropped'].median():.0f}, max {sub['n_dropped'].max():.0f} per date; "
        f"alignment flagged on {int(truth(sub['align_flagged']).sum())} dates." + extra,
    ]
    if "names_mc_over_listed" in sub.columns:
        rows = {
            "Monte Carlo / listed strips - 1 (the diagnostic, 2 %)": summary(
                sub["names_mc_over_listed"], column(sub, "sum_w_ER2_lc_se") / column(sub, "sum_w_M")
            ),
            "SVI strips / listed strips - 1": summary(column(sub, "names_svi_over_listed")),
            "Monte Carlo / SVI strips - 1": summary(
                column(sub, "names_mc_over_svi"),
                column(sub, "sum_w_ER2_lc_se") / column(sub, "sum_w_M_svi"),
            ),
            "z of Monte Carlo against the SVI strips": summary(column(sub, "names_mc_z")),
        }
        lines += [
            "",
            cap(
                "The names' second moment `Σ w E[R_i²]` three ways: Monte Carlo under the model, the names' own SVI strips "
                "(no Monte Carlo; sensitive to the slices' wings beyond the quotes), the study's listed strips"
            ),
            "",
            table(rows),
        ]
    return lines


def timing_lines(sub: pd.DataFrame, cap: Caption) -> list[str]:
    """Section 7: the timings of the pass."""
    cols = [
        "seconds_calibration",
        "seconds_companion",
        "seconds_pricing_lc",
        "seconds_pricing_cc",
        "seconds_deltas",
        "seconds_total",
    ]
    rows = {c.replace("seconds_", ""): summary(sub[c]) for c in cols if c in sub.columns}
    lines = [cap("Timings")]
    if rows:
        lines += ["", table(rows, digits=1)]
    lines += [
        "",
        f"Threads per process: {sorted({int(x) for x in sub['threads'].dropna()})}; cache hits {int(truth(sub['cache_hit']).sum())} of {len(sub)}. "
        "The times are those of processes that shared the machine with the pass's other workers and with other runs.",
    ]
    return lines


def flag_lines(frame: pd.DataFrame, ok: pd.DataFrame, tenor: str, columns: Sequence[str]) -> list[str]:  # fmt: skip
    """The failed dates with their reasons, the dates flagged for a name kept unscreened
    (decision 2) and, at 12m and 24m, the dates flagged by the clipped mass (decision 5): every
    one is listed, none is truncated."""
    failed = frame[frame["status"] == "failed"]
    lines = ["## Failed and flagged dates", ""]
    if len(failed):
        lines += [
            f"**{len(failed)} date(s) failed** and are in no statistic below (they stay in the table and in the by-date file):",
            "",
            "| date | reason |",
            "|---|---|",
            *[f"| {d:%Y-%m-%d} | {r if pd.notna(r) and r else 'no reason recorded'} |" for d, r in failed["reason"].items()],
        ]  # fmt: skip
    else:
        lines.append("No date failed.")
    lines.append("")
    unscreened = ok[truth(ok["flag_unscreened"])]
    if "n_names_unscreened" not in columns and "flag_unscreened" not in columns:
        lines.append(
            "A name kept unscreened (decision 2): the table carries neither `n_names_unscreened` nor `flag_unscreened` "
            "(a pass without the fallback): no date is flagged, and the summaries with and without the flagged dates are the same."
        )
    elif len(unscreened) == 0:
        lines.append(
            f"A name kept unscreened (decision 2, `flag_unscreened`): on none of the {len(ok)} priced dates; "
            "the summaries with and without the flagged dates are the same."
        )
    else:
        kept = unscreened["names_unscreened"] if "names_unscreened" in unscreened.columns else pd.Series("", index=unscreened.index)  # fmt: skip
        lines.append(
            f"**A name kept unscreened (decision 2, `flag_unscreened`) on {len(unscreened)} of {len(ok)} priced dates**: "
            + ", ".join(f"{d:%Y-%m-%d} ({n})" if pd.notna(n) and n else f"{d:%Y-%m-%d}" for d, n in kept.items())
            + ". Every summary below is given with and without them."
        )  # fmt: skip
    clip, low, high = truth(ok["flag_clip"]), truth(ok["flag_clip_low"]), truth(ok["flag_clip_high"])  # fmt: skip
    lines.append("")
    if tenor in CLIP_SUMMARY_TENORS:
        lines.append(
            f"**Clip flag (decision 5, `flag_clip`): {int(clip.sum())} of {len(ok)} priced dates** have a clipped mass inside ±2.5 sd above 1 % at "
            f"`λ = 0` (`flag_clip_low`, {count(int(low.sum()))}) or at the cap (`flag_clip_high`, {count(int(high.sum()))}). They stay in the table and in "
            f"the by-date file and are out of the summaries, which are on the other {int((~clip).sum())}; each is listed with its clipped masses "
            "at the end of the report."
        )
    else:
        lines.append(
            f"Clip flags: at {tenor} they remove no date from the summaries; `flag_clip_low` on {int(low.sum())} and `flag_clip_high` on "
            f"{int(high.sum())} of the {len(ok)} priced dates, either on {int(clip.sum())} (section 3)."
        )
    return lines


def clip_flag_lines(ok: pd.DataFrame) -> list[str]:
    """Every clip-flagged date (decision 5) with its clipped mass inside ±2.5 sd at ``λ = 0`` and
    at the cap: the list is complete."""
    clip = truth(ok["flag_clip"])
    lines = [f"## Clip-flagged dates (decision 5): {int(clip.sum())} of {len(ok)} priced dates", ""]
    if not clip.any():
        return [*lines, "None."]
    return [
        *lines,
        "| flagged date | clipped mass at λ = 0 | clipped mass at the cap | flag |",
        "|---|---:|---:|---|",
        *[
            f"| {d:%Y-%m-%d} | {r['clip_low_inner_max']:.4f} | {r['clip_high_inner_max']:.4f} | "
            f"{' and '.join(k for k, v in (('low', r['flag_clip_low']), ('cap', r['flag_clip_high'])) if v)} |"
            for d, r in ok[clip].iterrows()
        ],
    ]


# ---------------------------------------------------------------------------------------------
# the report of a pass
# ---------------------------------------------------------------------------------------------


def build(tenor: str, budget: str, out: Path, label: str = "", study: Path = dd.OUT) -> Path:
    """Write ``report_<stem>.md``, ``report_<stem>_by_date.csv`` and the figures of the pass."""
    raw, source = load(tenor, budget, out, label, study)
    columns = list(raw.columns)
    frame, absent = complete(raw)
    frame = derive(frame)
    stem = stem_of(tenor, budget, label)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ok = frame[frame["status"] != "failed"]
    has_s = "S_ratio" in ok.columns
    indicative = tenor in INDICATIVE_TENORS or (len(ok) > 0 and bool(ok["indicative"].all()))
    n_all = len(frame)
    note = f"indicative ({count(n_all, words=True)})" if indicative else ""
    md: list[str] = []
    add = md.append
    commits = ", ".join(sorted({str(c) for c in frame["git_commit"].dropna()}))
    add(
        f"# Local correlation model against the study's models: {tenor}, {budget} budget"
        + (f", pass `{label}`" if label else "")
        + (f" — {note}" if note else "")
    )
    add("")
    span = f"{ok.index.min():%Y-%m-%d} to {ok.index.max():%Y-%m-%d}" if len(ok) else "no priced date"  # fmt: skip
    add(
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{source.name}`: {n_all} dates "
        f"({span}), statuses {frame['status'].value_counts().to_dict()}; commit(s) {commits}; "
        f"particles {sorted({int(x) for x in ok['n_particles'].dropna()})}, paths {sorted({int(x) for x in ok['n_paths'].dropna()})}."
    )
    recorded = []
    if "calendar_repair" in columns:
        recorded.append(f"calendar repair of the names {truth(ok['calendar_repair']).map({True: 'on', False: 'off'}).value_counts().to_dict()}")  # fmt: skip
    else:
        recorded.append("no `calendar_repair` column (rows written before the repair existed: off)")
    recorded.append(
        "the unscreened fallback is recorded (`n_names_unscreened`)"
        if "n_names_unscreened" in columns
        else "no `n_names_unscreened` column (no fallback)"
    )
    add("")
    add("The pass as its rows record it: " + "; ".join(recorded) + ".")
    add("")
    add(
        "LC: the calibrated local correlation model. CC: its constant-correlation companion (the same "
        "local vols, the constant correlation that reprices the index at-the-money straddle at the horizon), "
        "priced on the same paths. Copula: the study's model (`P_D`, `C_<m>` of `entries`, basket B1). "
        + (
            "Model S: the study's skewed model (`model_s_3m.parquet`, its converged dates). "
            if has_s
            else f"Model S does not exist at {tenor}. "
        )
        + "Every number below is [measured] "
        "from these tables unless it says otherwise; standard errors are Monte Carlo errors of one date, "
        "or the standard error of a mean across dates where the column says so."
    )
    if absent:
        add("")
        add("Columns the table does not carry, reported as n/a: " + ", ".join(f"`{c}`" for c in absent) + ".")  # fmt: skip
    missing = [c for c in DERIVED if c not in columns]
    if missing:
        add("")
        add(
            "Computed here from the base columns, the table not carrying them (rows written before 2026-10-09): "
            + ", ".join(f"`{c}`" for c in missing)
            + "."
        )
    if tenor != "3m":
        add("")
        add(CAVEAT)
    add("")
    md.extend(flag_lines(frame, ok, tenor, columns))
    table_by_date = by_date(frame, tenor)
    csv = out / f"report_{stem}_by_date.csv"
    tmp = csv.with_suffix(".csv.tmp")
    table_by_date.to_csv(tmp, index=False, float_format="%.8g")
    tmp.replace(csv)
    add("")
    add(
        f"By date: `{csv.name}` — one line per date of the table, the failed ones included: status, reason, flags, LC/CC, LC/copula, "
        "CC/copula, model S/copula, the listed-variance forward over the copula's, the clipped mass inside ±2.5 sd by side, the basket "
        "part of the `E[V]` split (raw and over `M_B^listed`), κ under LC, CC and the copula, the gates not passed and the names' diagnostic."
    )
    if len(ok) == 0:
        add("")
        add("No date of the table is priced: there is nothing to summarise.")
        return _write(out / f"report_{stem}.md", md)
    subs = subsets(ok, tenor)
    add("")
    add(
        "Rows of the summaries: "
        + "; ".join(f"{s.label} ({int(s.mask.sum())})" for s in subs)
        + ". Each table says which."
    )

    def section(lines_of: Callable[[pd.DataFrame, Caption], list[str]]) -> None:
        for s in subs:
            sub = ok[s.mask]

            def cap(what: str, s: Subset = s, sub: pd.DataFrame = sub) -> str:
                return f"**{what} — {s.label}: {count(len(sub))}{'; ' + note if note else ''}.**"

            add("")
            if len(sub) == 0:
                add(cap("No date in this set"))
                continue
            md.extend(lines_of(sub, cap))

    # --- 1. the forward
    add("")
    add("## 1. The Palladium forward, like for like")
    add("")
    add(
        "**Like for like** (owner's check (b)): model S is measured over the study's copula, so LC and CC are put over the same copula "
        "`P_D` beside it; the listed-variance forward `κ_cop·√EQV` over `P_D`, i.e. `√(EQV/EV)` (check (c)), is the copula's own forward on the "
        "listed variance; LC over CC (paired, the same paths) is M12 against its own companion. The Monte Carlo error of LC/copula and "
        f"CC/copula is the numerator's; the copula's own error (median `P_D_se/P_D` {(ok['P_D_se_cop'] / ok['P_D_cop']).median():.5f} on these dates) is not included."
    )
    last = ok.iloc[-1]
    add("")
    add(
        f"Latest priced date, {ok.index[-1]:%Y-%m-%d}: LC/copula {last['lc_over_copula']:.4f} ({last['lc_over_copula_se']:.4f}), "
        f"CC/copula {last['cc_over_copula']:.4f} ({last['cc_over_copula_se']:.4f}), "
        + (
            (
                f"model S/copula {last['S_ratio']:.4f}, "
                if pd.notna(last["S_ratio"])
                else "model S did not converge on that date, "
            )
            if has_s
            else ""
        )
        + f"listed-variance forward/copula {last['listed_fwd_ratio']:.4f}, LC/CC {last['ratio']:.4f} ({last['ratio_se']:.4f}); Monte Carlo standard errors in brackets."
    )
    line = study_line(study, tenor)
    if line is not None:
        add("")
        add(
            f"From the study's own tables alone (`entries_{tenor}.parquet`, basket B1, and `model_s_{tenor}.parquet`; no number of this model) "
            f"[measured]: on the {line['n']} dates with a converged model S (of {line['n_model_s']}), the mean of `√(EQV/EV)` is "
            f"{line['mean_listed']:.4f} (standard error of the mean {line['se_listed']:.4f}) and the mean of `P_D_S/P_D` is {line['mean_s']:.4f} "
            f"({line['se_s']:.4f}); on the study's last date, {line['last']}, `√(EQV/EV)` is {line['listed_last']:.4f}"
            + (f" and `P_D_S/P_D` {line['s_last']:.4f}" if np.isfinite(line["s_last"]) else "")
            + "."
        )
    section(lambda sub, cap: forward_lines(sub, cap, has_s))
    # --- 2. the calls
    add("")
    add("## 2. Calls on the dispersion at the study's strikes")
    section(lambda sub, cap: calls_lines(sub, cap, has_s))
    add("")
    add(
        "The strikes are multiples of the copula's forward `P_D`; at K_200 the copula's price is zero on many dates and the ratio is then undefined."
    )
    # --- 3. the wing
    add("")
    add("## 3. The index downside wing")
    section(lambda sub, cap: wing_lines(sub, cap, has_s))
    # --- 4. the gap to model S
    add("")
    add("## 4. The gap to model S across dates: a regression")
    add("")
    if has_s:
        add(
            "Owner's check (d): across the dates with a converged model S, `LC/CC - P_D_S/P_D` (M12 against its companion, minus model S "
            "against the copula) regressed by ordinary least squares, with an intercept, on the clipped mass inside ±2.5 sd (`clip_inner_max`, "
            "the maximum over slices) and on the basket part of the `E[V]` split, `EV_basket_part = E_LC[R̄²] - M_B^listed` (negative when the "
            "model's basket variance is short of the listed one, which raises `E[V]`). The first table takes the basket part over `M_B^listed` "
            "(`E_LC[R̄²]/M_B^listed - 1`, a ratio comparable across dates): it is the one used; the second gives the raw variance, which scales "
            "with the level of volatility of the date. Standard errors: classical `s²(Z'Z)⁻¹`, and heteroskedasticity-robust HC1; `t` is the "
            "coefficient over its standard error. [measured]"
        )
        section(gap_lines)
    else:
        add(f"No model S at {tenor}: the regression of check (d) is not run.")
    # --- 5. kappa, deltas
    add("")
    add("## 5. κ = E[D]/√E[V]" + (", and the deltas" if "delta_fwd_lc" in ok.columns else ""))
    section(kappa_lines)
    # --- 6. gates and the diagnostic
    add("")
    add("## 6. Gates, and the names' diagnostic")
    section(lambda sub, cap: checks_lines(sub, cap, absent))
    # --- 7. timings
    add("")
    add("## 7. Timings (seconds per date)")
    section(timing_lines)
    if tenor in CLIP_SUMMARY_TENORS:
        add("")
        md.extend(clip_flag_lines(ok))
    # --- figures
    add("")
    add("## Figures")
    add("")
    add(
        "All priced dates, the flagged ones included"
        + (f"; {note}" if note else "")
        + "."
    )  # fmt: skip
    add("")
    for name, caption in figures_of(ok, stem, figures, has_s):
        add(f"![{caption}](figures/{name})")
        add("")
        add(f"*{caption}*")
        add("")
    return _write(out / f"report_{stem}.md", md)


def _write(path: Path, md: list[str]) -> Path:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(md) + "\n")
    tmp.replace(path)
    return path


# ---------------------------------------------------------------------------------------------
# one pass against another
# ---------------------------------------------------------------------------------------------


def compare(
    tenor: str, budget: str, out: Path, label: str, against: str = "", study: Path = dd.OUT
) -> Path:
    """The pass ``label`` against the pass ``against`` of the same tenor and budget (``''`` is the
    main table on either side), on the dates both priced: the differences ``label`` minus
    ``against`` of the forward, of its ratios to the CC companion and to the copula, of the calls
    and their ratios, of the names' second moment and of the clipped mass, by period, and the
    dates only one of them priced with what the other says of them.  With and without the dates
    flagged in either pass (``subsets``).  A column one table lacks is reported as n/a; the
    commits may differ and are printed.  Written to ``report_<stem>_vs_<other>.md``
    (``<other>`` = ``against``, or ``main``).  Test:
    ``tests/test_lcm_report.py::test_build_and_compare_on_synthetic_tables``."""
    var_raw, source = load(tenor, budget, out, label, study)
    base_raw, base_source = load(tenor, budget, out, against, study)
    var, base = derive(complete(var_raw)[0]), derive(complete(base_raw)[0])
    name_v = f"the pass `{label}`" if label else "the main pass"
    name_b = f"the pass `{against}`" if against else "the main pass"
    priced_v, priced_b = var.index[var["status"] != "failed"], base.index[base["status"] != "failed"]  # fmt: skip
    common = priced_v.intersection(priced_b)
    b, v = base.loc[common], var.loc[common]
    md: list[str] = []
    add = md.append

    def commits(frame: pd.DataFrame) -> str:
        return ", ".join(sorted({str(c) for c in frame["git_commit"].dropna()})) or "not recorded"

    add(f"# {name_v[0].upper()}{name_v[1:]} against {name_b}: {tenor}, {budget} budget")
    add("")
    add(
        f"Generated {time.strftime('%Y-%m-%d %H:%M')} from `{source.name}` ({len(var)} dates, statuses "
        f"{var['status'].value_counts().to_dict()}; commit(s) {commits(var)}) and `{base_source.name}` ({len(base)} dates, statuses "
        f"{base['status'].value_counts().to_dict()}; commit(s) {commits(base)}). {len(common)} dates are priced by both. "
        f"Differences are {name_v} minus {name_b}; relative differences are of the first over the second, minus 1. [measured]"
    )
    lacking = {
        name: sorted(c for c in set(other.columns) - set(this.columns) if not c.endswith("_cop"))
        for name, this, other in ((name_v, var_raw, base_raw), (name_b, base_raw, var_raw))
    }
    for name, cols in lacking.items():
        if cols:
            add("")
            add(f"Columns of the other table that {name} does not carry (n/a where a row below needs one): " + ", ".join(f"`{c}`" for c in cols) + ".")  # fmt: skip

    def rows_of(index: pd.Index) -> dict[str, dict[str, Any]]:
        b, v = base.loc[index], var.loc[index]
        rows: dict[str, dict[str, Any]] = {
            "E_LC[D], relative": summary(v["ED_lc"] / b["ED_lc"] - 1.0, b["ED_lc_se"] / b["ED_lc"]),
            "E_CC[D], relative": summary(v["ED_cc"] / b["ED_cc"] - 1.0, b["ED_cc_se"] / b["ED_cc"]),
            "LC/CC of the forward, difference": summary(v["ratio"] - b["ratio"], b["ratio_se"]),
            "LC/copula, difference": summary(v["lc_over_copula"] - b["lc_over_copula"], b["lc_over_copula_se"]),
            "CC/copula, difference": summary(v["cc_over_copula"] - b["cc_over_copula"], b["cc_over_copula_se"]),
        }  # fmt: skip
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
            f"names' second moment over listed strips - 1: {name_b}": summary(column(b, "names_mc_over_listed")),
            f"names' second moment over listed strips - 1: {name_v}": summary(column(v, "names_mc_over_listed")),
            "clipped mass inside ±2.5 sd, difference": summary(v["clip_inner_max"] - b["clip_inner_max"]),
            "ED_wing / E_CC[D], difference": summary(v["ED_wing_ratio"] - b["ED_wing_ratio"], b["ED_wing_ratio_se"]),
        })  # fmt: skip
        for name, frame in ((name_b, b), (name_v, v)):
            if "n_dropped_calendar" in frame.columns:
                rows[f"slices dropped by the calendar repair, per date: {name}"] = summary(
                    column(frame, "n_dropped_calendar")
                )
        return rows

    flags = pd.DataFrame(
        {k: truth(v[k]) | truth(b[k]) for k in ("flag_unscreened", "flag_clip", "flag_clip_low")}
    )
    subs = subsets(flags, tenor)
    if len(subs) == 1 and tenor not in CLIP_SUMMARY_TENORS:
        add("")
        add(
            "No date priced by both carries a name kept unscreened in either pass: the summaries with and without the flagged dates are the same."
        )
    for s in subs:
        index = common[s.mask.to_numpy()]
        add("")
        what = s.label.replace("all priced dates", "all dates priced by both").replace(
            "priced dates", "dates priced by both"
        )
        add(f"**{what[0].upper()}{what[1:]}: {count(len(index))} (a flag in either pass counts).**")
        add("")
        add(table(rows_of(index)))
    has_repair = [f for f in (v, b) if "n_dropped_calendar" in f.columns]
    dropped = pd.Series(np.nan, index=common, dtype=float)
    if has_repair:  # a table without the column is a pass without the repair: no slice dropped
        d_v = column(v, "n_dropped_calendar") if "n_dropped_calendar" in v.columns else 0.0
        d_b = column(b, "n_dropped_calendar") if "n_dropped_calendar" in b.columns else 0.0
        dropped = d_v - d_b + pd.Series(0.0, index=common)
    by_year = pd.DataFrame({
        "year": common.year,
        "E_LC[D] rel": (v["ED_lc"] / b["ED_lc"] - 1.0).to_numpy(),
        "LC/CC diff": (v["ratio"] - b["ratio"]).to_numpy(),
        "C_125 rel": (v["C_125_lc"] / b["C_125_lc"].where(b["C_125_lc"] > 0) - 1.0).to_numpy(),
        "slices dropped": dropped.to_numpy(),
    })  # fmt: skip
    by_year["period"] = pd.cut(
        by_year["year"],
        [2006, 2010, 2014, 2018, 2022, 2027],
        labels=["2007-10", "2011-14", "2015-18", "2019-22", "2023-26"],
    )
    add("")
    add("By period (means over the dates of the period priced by both, flagged or not):")
    add("")
    add(
        "| period | dates | E_LC[D], relative | LC/CC, difference | call at K_125, relative | slices dropped by the calendar repair per date, first minus second (none in a pass without the column) |"
    )
    add("|---|---:|---:|---:|---:|---:|")
    for period, g in by_year.groupby("period", observed=True):
        slices = g["slices dropped"].mean()
        add(
            f"| {period} | {len(g)} | {g['E_LC[D] rel'].mean():+.4f} | {g['LC/CC diff'].mean():+.4f} | {g['C_125 rel'].mean():+.3f} | "
            f"{f'{slices:+.1f}' if np.isfinite(slices) else 'n/a'} |"
        )
    for name, this, other, other_name in ((name_v, var, base, name_b), (name_b, base, var, name_v)):
        mine = this.index[this["status"] != "failed"]
        theirs = other.index[other["status"] != "failed"]
        only = sorted(set(mine) - set(theirs))
        add("")
        add(f"Priced by {name} only ({len(only)}): {dates_of(only)}.")
        if not only:
            continue
        said = []
        for d in only:
            if d not in other.index:
                said.append(f"{d:%Y-%m-%d}: not in the table")
            else:
                reason = other.loc[d, "reason"]
                said.append(f"{d:%Y-%m-%d}: {reason if pd.notna(reason) and reason else 'failed, no reason recorded'}")  # fmt: skip
        add("")
        add(f"What {other_name} says of them: " + "; ".join(said) + ".")
        kept = this.loc[only, "names_unscreened"].fillna("").value_counts().to_dict() if "names_unscreened" in this.columns else {}  # fmt: skip
        add("")
        add(
            (f"On those dates the names kept unscreened by {name} are: {kept}; " if kept else "On those dates, ")
            + f"LC/CC of the forward: mean {this.loc[only, 'ratio'].mean():.4f}, min {this.loc[only, 'ratio'].min():.4f}, max {this.loc[only, 'ratio'].max():.4f}."
        )  # fmt: skip
    return _write(out / f"report_{stem_of(tenor, budget, label)}_vs_{against or 'main'}.md", md)


# ---------------------------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------------------------


def figures_of(ok: pd.DataFrame, tag: str, folder: Path, has_s: bool) -> list[tuple[str, str]]:
    made = []
    x = ok.index
    # 1. the forward over time: over the CC companion (top), over the copula (bottom)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True, height_ratios=(1.0, 1.5))
    a1.errorbar(
        x,
        ok["ratio"],
        yerr=2 * ok["ratio_se"],
        fmt="o",
        ms=3.5,
        lw=0.8,
        color=BLUE,
        label="E_LC[D] / E_CC[D] (±2 se)",
    )
    a1.axhline(1.0, color="k", lw=0.6)
    a1.set_ylabel("over the CC companion")
    a1.legend(fontsize=8)
    for axis in (a1, a2):
        axis.grid(axis="y", color="0.9", lw=0.6)
        axis.set_axisbelow(True)
    if has_s:
        a2.plot(x, ok["S_ratio"], "s", ms=3.5, color=ORANGE, alpha=0.85, label="model S / copula P_D")  # fmt: skip
    a2.plot(x, ok["lc_over_copula"], "o", ms=3.5, color=BLUE, alpha=0.85, label="E_LC[D] / copula P_D")  # fmt: skip
    a2.plot(x, ok["cc_over_copula"], "^", ms=3.5, color=AQUA, alpha=0.85, label="E_CC[D] / copula P_D")  # fmt: skip
    a2.plot(x, ok["listed_fwd_ratio"], "-", lw=0.9, color=GREY, label="listed-variance forward / copula P_D, √(EQV/EV)")  # fmt: skip
    a2.axhline(1.0, color="k", lw=0.6)
    a2.set_ylabel("over the copula")
    a2.legend(fontsize=8, ncol=2)
    a1.set_title(f"Palladium forward, {tag}: LC over its CC companion; model S, LC and CC over the copula")  # fmt: skip
    made.append(
        (
            f"lcm_{tag}_forward.png",
            "The forward by date. Top: LC over its constant-correlation companion (paired, ±2 Monte Carlo standard errors). "
            "Bottom, like for like over the copula's P_D: model S, LC, CC and the listed-variance forward √(EQV/EV).",
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
    a2.axhline(CLIP_FLAG_MASS, color="r", lw=0.8, label="1 %")
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
    ap.add_argument(
        "--tag", default="", help="the suffix of the pass reported (disp_lcm.py --tag); '' is the main table"
    )  # fmt: skip
    ap.add_argument(
        "--against",
        default=None,
        help="also compare the pass with the pass of this suffix, '' meaning the main table "
        "(report_<stem>_vs_<other>.md); not given, a tagged pass is compared with the main table",
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    cfg = lp.load_config(args.config)
    out = lp.out_root(cfg, args.root)
    path = build(args.tenor, args.budget, out, args.tag)
    log.info("written %s", path)
    against = args.against if args.against is not None else ("" if args.tag else None)
    if against is None:
        return 0
    if against == args.tag:
        log.info("no comparison: --against names the pass itself")
        return 0
    try:
        log.info("written %s", compare(args.tenor, args.budget, out, args.tag, against))
    except FileNotFoundError as exc:
        log.info("no comparison with %s: %s", f"the pass {against}" if against else "the main pass", exc)  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
