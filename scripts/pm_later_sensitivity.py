"""After the freeze of the PM results package of 2026-10-09: two sensitivities of the 3m history
of the local correlation model at the PRODUCTION budget.  Nothing of the frozen package is
written: the output goes to ``outputs/dispersion_lc/pm_update/later/production_3m/sensitivity``
(``SENSITIVITY.md``, ``sensitivity.json``, ``tables/*.csv``, and the same page and records in the
package's part format under ``parts/``), and no line is added to ``STATUS.md``.

What it computes.

1. *The owner's decisions 1, 2 and 5* ("keep the no-repair results as a reported sensitivity").
   The production pass at the defaults of 9 Oct (``lcm_3m.parquet``: decisions 1, 2 and 5 on)
   against the production pass at the old defaults that was stopped (``lcm_3m_norepair.parquet``:
   no calendar repair, no unscreened fallback), on the dates priced in both: new minus old, per
   date and in summary (mean, median, quartiles, minimum and maximum with their dates, pooled
   ratios), for E_LC[D], E_CC[D], LC/CC, LC/copula, CC/copula, the calls at 0.75, 1 and 1.25 times
   the copula's forward (LC over copula), κ, E[V]/EQV, the clipped mass, the index errors at the
   money, at 90 % of the forward and at −2.5 sd, and the sticky-strike deltas of the forward.
   The dates are split by what changed on them, read from the new row: the names' slices dropped
   by the calendar repair (``n_dropped_calendar − n_dropped_calendar_index``: none, 1-9, 10 or
   more), whether a DJX slice is dropped (``n_dropped_calendar_index > 0``, decision 5), and
   whether a name is kept unscreened (``n_names_unscreened > 0``, decision 2: priced at the new
   defaults only, listed with their values and with the names' 2 % diagnostic of the row).  The
   differences of the frozen table B4 (``tables/B_sensitivity.csv``) are recomputed from the two
   tables and compared.
   *The sample.*  The stopped pass did not run a cross-section of the history, so the page says
   which dates it ran: the dates run per year, two periods (split at ``PERIOD_SPLIT``) and the
   partition ``MIX`` of the dates by what the repairs drop, each on the dates priced in both
   tables and on the priced dates of the production table.  The means of the common dates by
   group and by period are reweighted to the priced dates of the production table (two
   reweightings of the same differences; no error is computed for them).
   *Concentration.*  For E_LC[D], E_CC[D] and the three ratios: the share of the sum of the
   differences supplied by the ``TOP_COUNTS`` dates with the largest absolute difference, the
   mean without the ``N_EXTREMES`` largest and the ``TRIM`` trimmed mean (``concentration``).
2. *The budget.*  Production (8e5 particles and paths) minus development
   (``lcm_3m_dev_repair.parquet``, 2e5) on the dates where the two rows have the same
   specification — equal ``spec_key`` when the key is equal on some date; otherwise (the key
   hashes the particle count, so it differs between budgets) ``n_dropped_calendar_index == 0`` in
   the production row and equal ``n_dropped`` — for LC/CC, LC/copula, CC/copula, E_LC[D],
   E_CC[D], the companion's fitted correlation ``rho_cc``, ``lambda_c``, κ and the clipped mass:
   mean, median, quartiles, extremes, the share of dates on the more frequent side of zero,
   the sign test, and the difference in units of the development row's Monte Carlo error.  The
   LC/CC difference is split exactly into its two legs, ``ΔE_LC[D]/E_CC[D]_dev`` and
   ``−(LC/CC)_prod·ΔE_CC[D]/E_CC[D]_dev``.  On the dates of the rule the columns of a row that
   describe its inputs and settings (``INPUT_COLUMNS``) are compared between the two tables.
   The frozen sentence of reader note 1 on the budget (four dates) is recomputed, with the
   number of dates on which the size of the difference is outside its range.  The basket
   forward error (pricing noise alone) is used to check the bound of a difference and the
   independence of the dates.  The size of the companion's fitting error is given as an order of
   magnitude from the rows' own straddle error (``straddle_B_cc_se``), not as a measurement.
3. *Other frozen statements* (the errata section): the frozen records of the rows of the
   reference dates (``numbers.json``) against the three tables, the frozen specification-only
   check of the index repair (``diagnostics/index_repair_3m.json``) against
   ``n_dropped_calendar_index`` of the production table, the row files of the reference dates
   (``outputs/dispersion_lc/rows``) against the rows of the three tables, and three sentences of
   the frozen findings and reader notes; a quoted frozen sentence is looked up word for word in
   its file (``quoted``) before it is printed.

Reads (read-only): ``outputs/dispersion_lc/lcm_3m.parquet``, ``lcm_3m_norepair.parquet`` and
``lcm_3m_dev_repair.parquet`` (one row per date), ``outputs/dispersion/entries_3m.parquet``
(basket B1: the copula's ``P_D``, ``P_D_se``, calls ``C_<m>``, ``C_se_<m>`` and ``EQV``), the
row files ``outputs/dispersion_lc/rows/3m_*/<date>.json`` of the four reference dates, and of
the frozen package ``tables/B_sensitivity.csv``, ``numbers.json``,
``diagnostics/index_repair_3m.json`` and the parts ``R_reader_notes.md``, ``Z_findings.md`` and
``V_validation.md``.  It also asks git which files under ``volsto/`` differ between the commits
of the tables (``git diff --name-only``, read-only); the sentence is left out when git cannot
answer.

What each error is.

- *Per date, one row*: the pricing Monte Carlo standard error of the row given its calibrated
  model (``ratio_se`` for LC/CC, paired on common paths; the delta method of
  ``pm_common.ratio_se`` with the copula's own error for a ratio to the copula, the two
  independent; ``EV_se / EQV`` for E[V]/EQV).  The calibration's own noise is not in it.  The
  clipped mass, ``rho_cc`` and ``lambda_c`` are calibration outputs without a standard error.
- *Per date, a difference of two runs*: ``√(se_a² + se_b²)``.  The two runs share their particle
  and pricing seeds, so their pricing noise is in part common, but they are two calibrations and
  are not paired path by path: this is an upper bound of the pricing error of the difference, not
  its error (for a ratio to the copula the copula's error, the same number in the two rows, is
  in it twice).  ``over_bound`` is the difference over this bound: it is not a test.
- *A mean of differences across dates*: the standard error across dates, sd/√n, the dates
  treated as independent (no serial-correlation adjustment).  ``mc_bound_mean`` is the mean of
  the per-date bounds: the dates share their seeds, so the per-date errors are added linearly.
- *A pooled ratio* ``Σa/Σb`` and a difference of two pooled ratios: the across-dates
  linearisation ``√(n/(n−1)·Σu_i²)``, ``u_i = (a_i − R·b_i)/Σb`` (for a difference, the
  difference of the two runs' ``u_i``).
- *A share of a sum, a mean without the largest moves, a trimmed mean, a reweighted mean*: no
  error is computed (the dates left out are selected on the outcome; a reweighting is not an
  estimate on the other dates).
- *Order statistics, counts, shares*: no standard error.  The sign test is the exact two-sided
  binomial test of "as many dates above as below zero" (zeros left out); it treats the dates as
  independent, which they are not: every date uses the same particle and pricing seeds, so an
  error that is the same draw on every date has one sign on every date.

A date is *priced* in a table when its status there is not ``failed``.  The status printed is
the one under the current rule (the owner's decision 3: ``check_no_nan``, ``check_forward`` and
``check_index`` are the gates; the names' 2 % check is a diagnostic); the status stored in a
table is kept beside it in the CSV.  A date is *flagged* when a name is kept unscreened.  In the
page a date printed with a statistic carries † when its status is ``check`` (at the new
defaults; in the production row) and ‡ when it is flagged.

Tables (``tables/``): ``decisions_dates`` (the dates of the old-defaults table: where each is
priced, statuses, what changed), ``decisions_by_date`` and ``budget_by_date`` (long: one row per
date and quantity), ``decisions_summary`` and ``budget_summary`` (long: one row per quantity,
sample and statistic), ``decisions_sample_by_year`` and ``decisions_sample_mix`` (which dates
the stopped pass ran, against the production table), ``decisions_reweighted``,
``decisions_concentration``, ``decisions_only_new_defaults``, ``decisions_b4_check``,
``budget_lc_over_cc_legs``, ``budget_reference_dates``, ``frozen_rows_check``.  Records: the
statistics of the quantities of the page (``DECISION_KEYS``, ``BUDGET_KEYS``) for the samples
the page prints, and the other numbers of its text; the CSVs also hold the other quantities and
samples.

Run: ``.venv/bin/python scripts/pm_later_sensitivity.py`` (idempotent; ``--out`` must be a
folder under ``pm_update/later``).
"""

# ruff: noqa: RUF001, E501 — report prose: typographic signs and long table lines

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import pandas as pd
import pm_common as pc

SECTION = "later/production_3m/sensitivity"
PART = "sensitivity"
LATER = pc.PM / "later"
OUT = LATER / "production_3m" / "sensitivity"
NEW = pc.LC_OUT / "lcm_3m.parquet"
OLD = pc.LC_OUT / "lcm_3m_norepair.parquet"
DEV = pc.LC_OUT / "lcm_3m_dev_repair.parquet"
ENTRIES = pc.STUDY / "entries_3m.parquet"
B4 = pc.PM / "tables" / "B_sensitivity.csv"
CALL_TAGS = ("075", "100", "125")
#: ``scripts/lcm_price.py::GATING_CHECKS``: the checks that set the status under decision 3.
GATING_CHECKS = ("check_no_nan", "check_forward", "check_index")
NAN = float("nan")
STUDY_SOURCE = "outputs/dispersion/entries_3m.parquet (basket B1)"
SE_MEAN = "across dates: sd/√n, the dates treated as independent"
SE_POOLED = "across dates, linearised: √(n/(n−1)·Σu_i²), u_i the date's term of the pooled ratio (of the difference of the two pooled ratios)"
BOUND = (
    "√(se_a² + se_b²) of the two rows: an upper bound of the pricing Monte Carlo error of the difference, not its error "
    "(the two runs share their particle and pricing seeds but are two calibrations, not paired path by path)"
)
#: The frozen table B4: its keys, the quantity here and the factor to its unit.
B4_KEYS = (
    ("ED.lc", "ED_lc", 1.0),
    ("ED.cc", "ED_cc", 1.0),
    ("ratio.lc_over_cc", "lc_over_cc", 1.0),
    ("ratio.lc_over_copula", "lc_over_copula", 1.0),
    ("ratio.cc_over_copula", "cc_over_copula", 1.0),
    ("delta.lc_ss", "delta_fwd_lc", 1.0),
    ("delta.cc_ss", "delta_fwd_cc", 1.0),
    ("clip.inner_total", "clip_inner_max", 100.0),
    ("fwd_err", "forward_error", 1.0),
)
#: Columns of a row that describe its inputs and settings and do not depend on the budget.
INPUT_COLUMNS = (
    "T", "n_names", "n_dropped", "n_dropped_third_friday", "n_dropped_strikes", "n_dropped_spread", "n_dropped_calendar",
    "n_dropped_index", "svi_rms_vp_median", "svi_rms_vp_max", "svi_rms_vp_index", "n_slices", "sum_w_M_svi", "index_last_slice",
    "align_below", "align_above", "floored_names_max", "floored_index", "n_names_extrapolated", "n_names_unscreened",
    "lc_code_tag", "schedule", "family", "r_low", "particle_seed", "pricing_seed",
)  # fmt: skip
#: The frozen sentence of reader note 1 on the budget (the four dates of sections A and B).
FROZEN_BUDGET = "between the development and the production budget, same specification, LC/CC differs by 0.0005 to 0.0019 on the four dates of A and B (production lower on all four)"
FROZEN_BUDGET_RANGE = (0.0005, 0.0019)
#: Frozen files read for the other statements tested in the errata section (read-only).
NUMBERS = pc.PM / "numbers.json"
INDEX_REPAIR = pc.PM / "diagnostics" / "index_repair_3m.json"
READER_NOTES = pc.PM / "parts" / "R_reader_notes.md"
FINDINGS = pc.PM / "parts" / "Z_findings.md"
VALIDATION = pc.PM / "parts" / "V_validation.md"
ROWS = pc.LC_OUT / "rows"
#: The row folders the frozen sections A and B were built from, and the table of each.
ROW_FOLDERS = (("3m_production", "new"), ("3m_production_norepair", "old"), ("3m_development_repair", "dev"))  # fmt: skip
#: Frozen sentences tested beside table B4 and the budget sentence (each is looked up word for
#: word in its file before it is printed).
FROZEN_D1 = "the calendar repair of the names' slices (decision 1) moved LC/CC by -0.0041 and -0.0056 on the first two dates, about forty times the printed Monte Carlo error, and by -0.0001 today"
FROZEN_D1_MOVES = (("2019-09-03", -0.0041), ("2017-04-03", -0.0056), ("2026-10-02", -0.0001))
FROZEN_D1_MULTIPLE = 40.0
FROZEN_CLIP = "on 2015-11-02 M12's clipped mass goes from 0.320 to 0.092"
FROZEN_CLIP_VALUES = ("2015-11-02", 0.320, 0.092)
FROZEN_V2 = "32 of the 218 dates built in a specification-only check"
FROZEN_NOTE3 = "On 32 of the 218 dates the defaults of 9 Oct give another specification"
FROZEN_V2_COUNTS = (32, 218)
#: The frozen records of the rows of the reference dates: key of the record, quantity here.
FROZEN_ROW_KEYS = (
    ("ED.lc", "ED_lc"),
    ("ED.cc", "ED_cc"),
    ("ratio.lc_over_cc", "lc_over_cc"),
    ("ratio.lc_over_copula", "lc_over_copula"),
    ("ratio.cc_over_copula", "cc_over_copula"),
)
#: First year of the second period of 1.0 (the stopped old-defaults pass ran fewer dates per
#: year from this year on).
PERIOD_SPLIT = 2014
#: The concentration of 1.3: the numbers of largest absolute moves whose share of the sum is
#: given, the number of dates of the table of extremes, the share cut at each end of the
#: trimmed mean, and the quantities.
TOP_COUNTS = (3, 5, 8, 10)
N_EXTREMES = 8
TRIM = 0.10
CONCENTRATION_KEYS = ("lc_over_cc", "lc_over_copula", "cc_over_copula", "ED_lc", "ED_cc")
#: The partition of the priced dates by what the calendar repairs drop (samples of part 1).
MIX = (
    ("names_0", "no slice dropped"),
    ("d1_only_1_9", "decision 1 alone: 1–9 names' slices"),
    ("d1_only_10plus", "decision 1 alone: 10 or more names' slices"),
    ("djx_yes", "a DJX slice dropped (decision 5)"),
)
NO_ERROR = "no error computed"


@dataclass(frozen=True)
class Quantity:
    """One per-date quantity: its column in :func:`measures`, label, unit, definition, the digits
    it is printed with, and, for a ratio, the numerator and denominator of its pooled form."""

    key: str
    label: str
    unit: str
    definition: str
    digits: int = 5
    num: str | None = None
    den: str | None = None
    has_se: bool = True


def quantities() -> dict[str, Quantity]:
    cop = "P_D the copula's forward of the study's entry (basket B1)"
    vp = "vol points"
    idx = "the model's index implied volatility at the horizon minus the model's own target (its SVI fit of the DJX smile, after the screen and the repairs of the run)"
    dlt = "% of E[D] per +1 %"
    out = [
        Quantity("ED_lc", "E_LC[D]", "notional", "E[D], D = Σ w_i |R_i − R̄| at the horizon, under the calibrated local correlation (`ED_lc`, `ED_lc_se`)", 6),
        Quantity("ED_cc", "E_CC[D]", "notional", "E[D] under the constant-correlation companion (`ED_cc`, `ED_cc_se`)", 6),
        Quantity("lc_over_cc", "LC/CC", "", "E_LC[D] / E_CC[D], paired on common paths (the row's `ratio`, `ratio_se`)", 5, "_ED_lc", "_ED_cc"),
        Quantity("lc_over_copula", "LC/copula", "", f"E_LC[D] / P_D, {cop}; se: delta method with `ED_lc_se` and `P_D_se`, independent", 5, "_ED_lc", "_P_D"),
        Quantity("cc_over_copula", "CC/copula", "", f"E_CC[D] / P_D, {cop}; se: delta method with `ED_cc_se` and `P_D_se`, independent", 5, "_ED_cc", "_P_D"),
    ]  # fmt: skip
    for m in CALL_TAGS:
        out.append(Quantity(f"C_lc_over_copula_{m}", f"call {int(m) / 100:g}×: LC/copula", "", f"E_LC[(D − K)⁺] over the copula's `C_{m}`, K = {int(m) / 100:g} × P_D (the study's cash strike `K_{m}`); se: delta method with `C_{m}_lc_se` and the copula's `C_se_{m}`", 4, f"_C_{m}_lc", f"_C_{m}_cop"))  # fmt: skip
    out += [
        Quantity("kappa_lc", "κ, LC", "", "E_LC[D] / √E_LC[V] (`kappa_lc`, `kappa_lc_se`)", 5),
        Quantity("kappa_cc", "κ, CC", "", "E_CC[D] / √E_CC[V] (`kappa_cc`, `kappa_cc_se`)", 5),
        Quantity("EV_over_EQV_lc", "E[V]/EQV, LC", "", "`EV_lc` / EQV, EQV the listed-option value of E[V] of the study's entry; se: `EV_lc_se` / EQV", 4, "_EV_lc", "_EQV"),
        Quantity("EV_over_EQV_cc", "E[V]/EQV, CC", "", "`EV_cc` / EQV; se: `EV_cc_se` / EQV", 4, "_EV_cc", "_EQV"),
        Quantity("clip_inner_max", "clipped mass inside ±2.5 sd (fraction of the particles)", "fraction of the particles", "`clip_inner_max` = max(`clip_low_inner_max`, `clip_high_inner_max`): the larger of the two one-sided clipped masses (λ clipped at 0; λ clipped at its cap), each at its worst calibration slice, inside ±2.5 sd; not their sum", 5, has_se=False),
        Quantity("idx_err_atm", "index error at the money (vol points)", vp, f"`idx_err_atm`: {idx}, at the money", 3),
        Quantity("idx_err_90", "index error at 90 % of the forward (vol points)", vp, f"`idx_err_90`: {idx}, at the strike 90 % of the forward", 3),
        Quantity("idx_err_m25", "index error at −2.5 sd (vol points)", vp, f"`idx_err_m25`: {idx}, at −2.5 standard deviations", 3),
        Quantity("delta_fwd_lc", "Δ sticky strike of the forward, LC", dlt, "`delta_fwd_lc`: the common delta of the forward under LC, the model's own sticky strike (local vols in absolute spot, λ in absolute basket level, no recalibration)", 3),
        Quantity("delta_fwd_cc", "Δ sticky strike of the forward, CC", dlt, "`delta_fwd_cc`: the same under the constant-correlation companion", 3),
        Quantity("rho_cc", "ρ_CC, the companion's fitted correlation", "", "`rho_cc` = ρ_low + λ_c·(1 − ρ_low): the constant correlation that reprices the index at-the-money straddle at the horizon (fitted on the companion's paths)", 5, has_se=False),
        Quantity("lambda_c", "λ_c", "", "`lambda_c`: the constant λ of the companion", 5, has_se=False),
        Quantity("forward_error", "basket forward error", "forward", "`forward_error`: the Monte Carlo basket forward against its exact value (pricing noise: its expectation is zero)", 7),
    ]  # fmt: skip
    return {q.key: q for q in out}


Q = quantities()
#: The quantities of part 1 (the task's list, with κ and E[V]/EQV under both models).
DECISION_KEYS = (
    "ED_lc", "ED_cc", "lc_over_cc", "lc_over_copula", "cc_over_copula", "C_lc_over_copula_075", "C_lc_over_copula_100",
    "C_lc_over_copula_125", "kappa_lc", "kappa_cc", "EV_over_EQV_lc", "EV_over_EQV_cc", "clip_inner_max", "idx_err_atm",
    "idx_err_90", "idx_err_m25", "delta_fwd_lc", "delta_fwd_cc",
)  # fmt: skip
#: The quantities of part 2.
BUDGET_KEYS = ("lc_over_cc", "lc_over_copula", "cc_over_copula", "ED_lc", "ED_cc", "rho_cc", "lambda_c", "kappa_lc", "clip_inner_max")  # fmt: skip


# ------------------------------------------------------------------------------------- small tools
def rel(path: Path) -> str:
    """A path as a record cites it: relative to the worktree."""
    try:
        return str(Path(path).resolve().relative_to(pc.ROOT.resolve()))
    except ValueError:
        return str(path)


def finite(x: Any) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def truth(flag: pd.Series) -> pd.Series:
    """A flag column as booleans (empty on a failed date: False)."""
    return flag.map(lambda x: x is True or x == 1.0).astype(bool)


def cell(value: Any, se: Any = None, digits: int = 4, signed: bool = False) -> str:
    """``value ± se`` for a table cell (``n/a`` when missing).  An exact zero is printed without
    a sign; a value that is not zero keeps its sign when it rounds to zero."""
    if not finite(value):
        return "n/a"
    sign = "+" if signed else ""
    text = f"{0.0:.{digits}f}" if value == 0 else f"{value:{sign}.{digits}f}"
    return f"{text} ± {se:.{digits}f}" if finite(se) else text


def pct(x: float, digits: int = 0) -> str:
    """A share as a percentage."""
    return f"{100.0 * x:.{digits}f} %"


def sha12(path: Path) -> str:
    """The first 12 hex digits of the SHA-256 of a file."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]


def quoted(path: Path, text: str) -> str:
    """``text``, once it is found word for word in a frozen file (refused otherwise)."""
    if text not in Path(path).read_text():
        raise ValueError(f"{Path(path).name}: the quoted sentence is not in the file: {text[:70]}")
    return text


def changed_files(a: str, b: str, under: str = "volsto") -> list[str] | None:
    """The files under ``under`` that differ between two commits (read-only: ``git diff
    --name-only``); ``None`` when git cannot answer (no git, a commit that is not known)."""
    cmd = ["git", "-C", str(pc.ROOT), "diff", "--name-only", a, b, "--", under]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return sorted(out.stdout.split()) if out.returncode == 0 else None


def largest(d: pd.Series, k: int) -> list[str]:
    """The ``k`` dates with the largest ``|d|``, from the largest."""
    return list(d.abs().sort_values(ascending=False, kind="mergesort").index[:k])


def concentration(d: pd.Series) -> dict[str, Any]:
    """How much of the sum of the per-date differences ``d`` the dates with the largest ``|d|``
    supply (``share_top_<k>``: the sum over those ``k`` dates over the sum over all dates, signs
    kept), the mean without the ``N_EXTREMES`` largest and the trimmed mean (the ``TRIM`` share
    of the dates, rounded down, left out at each end of the sorted differences)."""
    total = float(d.sum())
    out: dict[str, Any] = {"n": len(d), "mean": float(d.mean()), "median": float(d.median())}
    for k in TOP_COUNTS:
        out[f"share_top_{k}"] = float(d[largest(d, k)].sum()) / total if total else NAN
    top = largest(d, N_EXTREMES)
    rest = d.drop(top)
    cut = int(TRIM * len(d))
    middle = d.sort_values(kind="mergesort").iloc[cut : len(d) - cut]
    out |= {
        "n_positive_in_top": int((d[top] > 0).sum()),
        "n_negative_in_top": int((d[top] < 0).sum()),
        "mean_without_top": float(rest.mean()),
        "n_without_top": len(rest),
        "trimmed_mean": float(middle.mean()),
        "n_trimmed": len(middle),
        "n_cut_each_end": cut,
    }
    return out


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def write_text(path: Path, text: str) -> None:
    """One file of the output folder, atomically; a frozen file is refused."""
    pc.guard_frozen(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def se_mean(v: pd.Series) -> float:
    return float(v.std(ddof=1) / math.sqrt(len(v))) if len(v) > 1 else NAN


def sign_test(n_neg: int, n_pos: int) -> float:
    """Exact two-sided binomial test of p = 1/2 on the dates that are not zero."""
    n = n_neg + n_pos
    if n == 0:
        return NAN
    tail = sum(math.comb(n, j) for j in range(min(n_neg, n_pos) + 1))
    return min(1.0, float(Fraction(2 * tail, 2**n)))


# ------------------------------------------------------------------------------------ the inputs
def over(a: pd.Series, a_se: pd.Series, b: pd.Series, b_se: pd.Series) -> tuple[pd.Series, pd.Series]:  # fmt: skip
    """``a / b`` per date (empty where the denominator is not positive) with the delta-method
    standard error of two independent estimates (``pm_common.ratio_se``)."""
    safe = b.where(b > 0.0)
    se = []
    for x, xs, y, ys in zip(a, a_se, safe, b_se, strict=True):
        if not all(finite(float(v)) for v in (x, xs, y, ys)):
            se.append(NAN)
        elif x == 0.0:
            se.append(float(xs) / float(y))
        else:
            se.append(pc.ratio_se(float(x), float(xs), float(y), float(ys)))
    return a / safe, pd.Series(se, index=a.index)


def load_entries() -> pd.DataFrame:
    """The study's entries of basket B1: one row per date."""
    entries = pd.read_parquet(ENTRIES)
    b1 = entries[entries["basket"] == "B1"]
    if not b1["date"].is_unique:
        raise ValueError("entries_3m: several B1 rows on a date")
    keep = {"P_D": "P_D", "P_D_se": "P_D_se", "EQV": "EQV_cop"}
    for m in CALL_TAGS:
        keep |= {f"K_{m}": f"K_{m}_cop", f"C_{m}": f"C_{m}_cop", f"C_se_{m}": f"C_{m}_cop_se"}
    return b1[["date", *keep]].rename(columns=keep)


def load(path: Path, entries: pd.DataFrame) -> pd.DataFrame:
    """One table of rows joined with the study's entry, indexed by the date; the status under
    the current rule in ``status_rule`` (``failing``: the gating checks not passed)."""
    rows = pd.read_parquet(path)
    if not rows["date"].is_unique:
        raise ValueError(f"{path.name}: several rows on a date")
    frame = rows.merge(entries, on="date", how="left", validate="one_to_one")
    frame = frame.sort_values("date").set_index("date")
    frame["priced"] = (frame["status"] != "failed") & frame["ED_lc"].notna()
    priced = frame["priced"]
    if frame.loc[priced, "P_D"].isna().any():
        raise ValueError(f"{path.name}: a priced date without a B1 entry")
    # the copula's numbers of the row are the entry's
    for ours, theirs in (("P_D_copula", "P_D"), ("EQV", "EQV_cop"), *((f"K_{m}", f"K_{m}_cop") for m in CALL_TAGS)):  # fmt: skip
        gap = (frame.loc[priced, ours] - frame.loc[priced, theirs]).abs() / frame.loc[priced, theirs].abs()  # fmt: skip
        if not (gap <= 1e-12).all():
            raise ValueError(f"{path.name}: {ours} differs from the entry's {theirs}")
    gap = (frame["ED_lc"] / frame["ED_cc"] - frame["ratio"]).abs()[priced]
    if not (gap <= 1e-12).all():
        raise ValueError(f"{path.name}: ratio is not ED_lc / ED_cc")
    sides = frame[["clip_low_inner_max", "clip_high_inner_max"]].max(axis=1)
    if not ((frame["clip_inner_max"] - sides).abs()[priced] <= 1e-12).all():
        raise ValueError(f"{path.name}: clip_inner_max is not the larger of the two sides")
    status, failing = [], []
    for _, r in frame.iterrows():
        bad = [k for k in GATING_CHECKS if not (r[k] is True or r[k] == 1.0)] if r["priced"] else []
        failing.append(", ".join(bad))
        status.append(("check" if bad else "ok") if r["priced"] else str(r["status"]))
    frame["status_rule"], frame["failing"] = status, failing
    return frame


def one_value(frame: pd.DataFrame, column: str) -> Any:
    """The single value of ``column`` over the priced rows (refused when there are several)."""
    values = frame.loc[frame["priced"], column].dropna().unique().tolist()
    if len(values) != 1:
        raise ValueError(f"{column}: {values}")
    return values[0]


def short(n: float) -> str:
    """A path count as the package writes it (``8e5``)."""
    mantissa, exponent = f"{n:.0e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def budget_of(frame: pd.DataFrame) -> str:
    n, p, c = (one_value(frame, k) for k in ("n_particles", "n_paths", "companion_paths"))
    return f"{short(n)} particles / {short(p)} paths (constant-correlation fit on {short(c)} paths)"


def measures(f: pd.DataFrame) -> pd.DataFrame:
    """The quantities of :func:`quantities` per date with ``<key>_se``, and the terms of the
    pooled ratios (columns with a leading underscore)."""
    empty = pd.Series(NAN, index=f.index)

    def col(name: str) -> pd.Series:
        return f[name].astype(float) if name in f.columns else empty

    m: dict[str, pd.Series] = {}

    def put(key: str, value: pd.Series, se: pd.Series | None = None) -> None:
        m[key] = value
        m[key + "_se"] = empty if se is None else se

    put("ED_lc", col("ED_lc"), col("ED_lc_se"))
    put("ED_cc", col("ED_cc"), col("ED_cc_se"))
    put("lc_over_cc", col("ratio"), col("ratio_se"))
    put("lc_over_copula", *over(col("ED_lc"), col("ED_lc_se"), f["P_D"], f["P_D_se"]))
    put("cc_over_copula", *over(col("ED_cc"), col("ED_cc_se"), f["P_D"], f["P_D_se"]))
    for t in CALL_TAGS:
        put(f"C_lc_over_copula_{t}", *over(col(f"C_{t}_lc"), col(f"C_{t}_lc_se"), f[f"C_{t}_cop"], f[f"C_{t}_cop_se"]))  # fmt: skip
    put("kappa_lc", col("kappa_lc"), col("kappa_lc_se"))
    put("kappa_cc", col("kappa_cc"), col("kappa_cc_se"))
    put("EV_over_EQV_lc", col("EV_lc") / f["EQV_cop"], col("EV_lc_se") / f["EQV_cop"])
    put("EV_over_EQV_cc", col("EV_cc") / f["EQV_cop"], col("EV_cc_se") / f["EQV_cop"])
    put("clip_inner_max", col("clip_inner_max"))
    for key in ("idx_err_atm", "idx_err_90", "idx_err_m25", "delta_fwd_lc", "delta_fwd_cc", "forward_error"):  # fmt: skip
        put(key, col(key), col(key + "_se"))
    put("rho_cc", col("rho_cc"))
    put("lambda_c", col("lambda_c"))
    for name in ("ED_lc", "ED_cc", "EV_lc", "EV_cc"):
        m["_" + name] = col(name)
    m["_P_D"], m["_EQV"] = f["P_D"], f["EQV_cop"]
    for t in CALL_TAGS:
        m[f"_C_{t}_lc"], m[f"_C_{t}_cop"] = col(f"C_{t}_lc"), f[f"C_{t}_cop"]
    return pd.DataFrame(m, index=f.index)


# --------------------------------------------------------------------------------- the statistics
def by_date(a: pd.DataFrame, b: pd.DataFrame, dates: list[str], keys: tuple[str, ...], names: tuple[str, str]) -> pd.DataFrame:  # fmt: skip
    """Long table, one row per date and quantity: the two runs, ``b − a``, the bound of its
    error, the difference over the bound and over run ``a``'s own error."""
    na, nb = names
    out = []
    for key in keys:
        q = Q[key]
        x, xs, y, ys = (t.loc[dates, c].astype(float) for t, c in ((a, key), (a, key + "_se"), (b, key), (b, key + "_se")))  # fmt: skip
        bound = (xs**2 + ys**2) ** 0.5
        out.append(pd.DataFrame({
            "date": dates, "quantity": key, "label": q.label, "unit": q.unit, na: x.to_numpy(), f"{na}_se": xs.to_numpy(),
            nb: y.to_numpy(), f"{nb}_se": ys.to_numpy(), "difference": (y - x).to_numpy(), "bound": bound.to_numpy(),
            "over_bound": ((y - x) / bound.where(bound > 0.0)).to_numpy(), f"in_{na}_se": ((y - x) / xs.where(xs > 0.0)).to_numpy(),
        }))  # fmt: skip
    return pd.concat(out, ignore_index=True)


def pooled(a: pd.DataFrame, b: pd.DataFrame, dates: list[str], q: Quantity) -> dict[str, tuple[float, float]]:  # fmt: skip
    """The pooled ratio ``Σnum/Σden`` of each run over ``dates`` and their difference, each with
    its across-dates linearised standard error."""
    if q.num is None or q.den is None or len(dates) < 2:
        return {}
    n, terms, out = len(dates), {}, {}
    for name, t in (("a", a), ("b", b)):
        num, den = t.loc[dates, q.num].astype(float), t.loc[dates, q.den].astype(float)
        if num.isna().any() or den.isna().any():
            return {}
        ratio = float(num.sum() / den.sum())
        terms[name] = (num - ratio * den) / float(den.sum())
        out[f"pooled_{name}"] = (ratio, math.sqrt(n / (n - 1) * float((terms[name] ** 2).sum())))
    u = terms["b"] - terms["a"]
    out["pooled"] = (out["pooled_b"][0] - out["pooled_a"][0], math.sqrt(n / (n - 1) * float((u**2).sum())))  # fmt: skip
    return out


def summarise(a: pd.DataFrame, b: pd.DataFrame, dates: list[str], q: Quantity, with_z: bool) -> list[dict[str, Any]]:  # fmt: skip
    """The statistics of ``b − a`` for one quantity over ``dates``: rows ``statistic, value, se,
    se_kind, date, note, n``; ``with_z`` adds the difference in units of run ``a``'s error."""
    x, y = a.loc[dates, q.key].astype(float), b.loc[dates, q.key].astype(float)
    keep = x.notna() & y.notna()
    x, y = x[keep], y[keep]
    xs, ys = a.loc[x.index, q.key + "_se"].astype(float), b.loc[x.index, q.key + "_se"].astype(float)  # fmt: skip
    d = y - x
    n = len(d)
    rows: list[dict[str, Any]] = []

    def add(stat: str, value: Any, se: Any = None, kind: str = "", date: str = "", note: str = "") -> None:  # fmt: skip
        rows.append({"statistic": stat, "value": value, "se": se, "se_kind": kind, "date": date, "note": note, "n": n})  # fmt: skip

    if n == 0:
        return rows
    add("mean_a", float(x.mean()), se_mean(x), SE_MEAN)
    add("mean_b", float(y.mean()), se_mean(y), SE_MEAN)
    add("mean", float(d.mean()), se_mean(d), SE_MEAN)
    add("median", float(d.median()))
    add("q25", float(d.quantile(0.25)))
    add("q75", float(d.quantile(0.75)))
    add("min", float(d.min()), date=str(d.idxmin()))
    add("max", float(d.max()), date=str(d.idxmax()))
    n_neg, n_pos = int((d < 0).sum()), int((d > 0).sum())
    add("n_neg", n_neg)
    add("n_pos", n_pos)
    add("n_zero", int((d == 0).sum()))
    add("share_same_sign", max(n_neg, n_pos) / n, note="below zero" if n_neg >= n_pos else "above zero")  # fmt: skip
    add("sign_test_p", sign_test(n_neg, n_pos))
    if q.has_se:
        bound = (xs**2 + ys**2) ** 0.5
        ok = bound.notna() & (bound > 0.0)
        if ok.any():
            add("mc_bound_mean", float(bound[ok].mean()))
            add("n_beyond_2_bounds", int((d[ok].abs() > 2.0 * bound[ok]).sum()))
        ok = xs.notna() & (xs > 0.0)
        if with_z and ok.any():
            z = d[ok] / xs[ok]
            add("z_mean", float(z.mean()), se_mean(z), SE_MEAN)
            add("z_sd", float(z.std(ddof=1)) if len(z) > 1 else NAN)
            add("z_median", float(z.median()))
            add("z_q25", float(z.quantile(0.25)))
            add("z_q75", float(z.quantile(0.75)))
            add("z_min", float(z.min()), date=str(z.idxmin()))
            add("z_max", float(z.max()), date=str(z.idxmax()))
            add("z_share_beyond_2", float((z.abs() > 2.0).mean()))
            both = ok & ys.notna()
            add("z_sd_if_independent", math.sqrt(float((1.0 + (ys[both] / xs[both]) ** 2).mean())))
    for stat, (value, se) in pooled(a, b, list(x.index), q).items():
        add(stat, value, se, SE_POOLED)
    return rows


STAT_TEXT = {
    "mean_a": "mean across the dates of the sample, run a",
    "mean_b": "mean across the dates of the sample, run b",
    "mean": "mean across the dates of b − a",
    "median": "median across the dates of b − a",
    "q25": "first quartile across the dates of b − a (linear interpolation)",
    "q75": "third quartile across the dates of b − a (linear interpolation)",
    "min": "smallest b − a over the dates",
    "max": "largest b − a over the dates",
    "n_neg": "number of dates with b − a below zero",
    "n_pos": "number of dates with b − a above zero",
    "n_zero": "number of dates with b − a exactly zero",
    "share_same_sign": "share of the dates on which b − a has the more frequent of its two signs",
    "sign_test_p": "exact two-sided binomial test of as many dates above as below zero (zeros left out; the dates treated as independent)",
    "mc_bound_mean": "mean over the dates of √(se_a² + se_b²): an upper bound of the pricing Monte Carlo error of the mean difference (the dates share their seeds: errors added linearly)",
    "n_beyond_2_bounds": "number of dates on which |b − a| exceeds twice √(se_a² + se_b²)",
    "z_mean": "mean across the dates of (b − a) / se_a",
    "z_sd": "standard deviation across the dates of (b − a) / se_a",
    "z_median": "median across the dates of (b − a) / se_a",
    "z_q25": "first quartile across the dates of (b − a) / se_a",
    "z_q75": "third quartile across the dates of (b − a) / se_a",
    "z_min": "smallest (b − a) / se_a over the dates",
    "z_max": "largest (b − a) / se_a over the dates",
    "z_share_beyond_2": "share of the dates on which |b − a| exceeds twice se_a",
    "z_sd_if_independent": "√(mean of 1 + se_b²/se_a²): the standard deviation (b − a) / se_a would have if the two rows were independent draws of the same quantity",
    "pooled_a": "pooled ratio Σ numerator / Σ denominator over the dates, run a",
    "pooled_b": "pooled ratio Σ numerator / Σ denominator over the dates, run b",
    "pooled": "difference of the two pooled ratios, b − a",
}
NO_SE = "an order statistic, a count or a share across dates: no standard error"


class Summary:
    """The statistics of one part: ``rows`` (the long CSV) and a lookup for the page.  ``labels``
    names run a, run b and their difference in the text of a statistic; ``marks`` maps a date to
    the sign printed after it (a status ``check``, a flagged date)."""

    def __init__(self, part: str, a: pd.DataFrame, b: pd.DataFrame, labels: tuple[str, str, str], with_z: bool, marks: dict[str, str]) -> None:  # fmt: skip
        self.part, self.a, self.b, self.labels, self.with_z, self.marks = part, a, b, labels, with_z, marks  # fmt: skip
        self.rows: list[dict[str, Any]] = []
        self.samples: dict[str, tuple[str, list[str]]] = {}
        self._at: dict[tuple[str, str, str], dict[str, Any]] = {}

    def meaning(self, stat: str) -> str:
        """What a statistic is, with the runs named."""
        la, lb, ld = self.labels
        text = STAT_TEXT[stat].replace("run a", la).replace("run b", lb).replace("b − a", ld)
        return text.replace("se_a", f"[the Monte Carlo standard error of the row at {la}]").replace("se_b", f"[the Monte Carlo standard error of the row at {lb}]")  # fmt: skip

    def add(self, sample: str, label: str, dates: list[str], keys: tuple[str, ...]) -> None:
        self.samples[sample] = (label, dates)
        for key in keys:
            for r in summarise(self.a, self.b, dates, Q[key], self.with_z):
                row = {"part": self.part, "quantity": key, "label": Q[key].label, "unit": Q[key].unit, "sample": sample, "sample_label": label, **r, "meaning": self.meaning(r["statistic"])}  # fmt: skip
                self.rows.append(row)
                self._at[(key, sample, r["statistic"])] = row

    def get(self, key: str, sample: str, stat: str) -> dict[str, Any]:
        return self._at.get((key, sample, stat), {"value": NAN, "se": None, "date": "", "note": "", "n": 0})  # fmt: skip

    def value(self, key: str, sample: str, stat: str) -> float:
        return float(self.get(key, sample, stat)["value"])

    def n(self, sample: str) -> int:
        return len(self.samples[sample][1])

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def dated(self, date: str) -> str:
        return f"{date}{self.marks.get(date, '')}"

    def cell(self, key: str, sample: str, stat: str, signed: bool = True, digits: int | None = None, with_date: bool = False) -> str:  # fmt: skip
        r = self.get(key, sample, stat)
        text = cell(r["value"], r["se"], Q[key].digits if digits is None else digits, signed)
        return f"{text} ({self.dated(r['date'])})" if with_date and r["date"] else text

    def signs(self, key: str, sample: str) -> str:
        return " / ".join(
            f"{int(self.value(key, sample, k))}" for k in ("n_neg", "n_pos", "n_zero")
        )

    def records(self, keys: tuple[str, ...], samples: tuple[str, ...], runs: dict[str, dict[str, str]]) -> list[dict[str, Any]]:  # fmt: skip
        """One record per statistic.  ``runs`` gives, for ``"a"``, ``"b"`` and ``"diff"``, the
        ``budget``, ``commit`` and ``source`` a record of that run (of the difference) carries."""
        out = []
        for r in self.rows:
            if r["quantity"] not in keys or r["sample"] not in samples:
                continue
            q = Q[r["quantity"]]
            stat = r["statistic"]
            run = runs["a" if stat.endswith("_a") else "b" if stat.endswith("_b") else "diff"]
            counts = stat.startswith(("n_", "share", "sign_test", "z_"))
            notes = r["se_kind"] or NO_SE
            if r["note"]:
                notes += f"; {r['note']}"
            if r["date"] and self.marks.get(r["date"]):
                notes += f"; the date is marked{self.marks[r['date']]} in the page"
            out.append(pc.record(
                f"L.sens.{self.part}.{q.key}.{r['sample']}.{stat}", SECTION, f"{q.label}: {r['meaning']}; sample: {r['sample_label']}", r["value"], r["se"],
                date=r["date"], unit="" if counts else q.unit, definition=q.definition, budget=run["budget"], commit=run["commit"], source=run["source"], n=r["n"], notes=notes,
            ))  # fmt: skip
        return out


# ------------------------------------------------------------------------------------------ part 1
def decision_dates(new: pd.DataFrame, old: pd.DataFrame) -> pd.DataFrame:
    """The dates of the old-defaults table: where each is priced and what changed on it."""
    rows = []
    for date in old.index:
        o = old.loc[date]
        if date not in new.index:
            raise ValueError(f"{date}: in the old-defaults table, not in the new one")
        n = new.loc[date]
        po, pn = bool(o["priced"]), bool(n["priced"])
        where = {(True, True): "both", (False, True): "new defaults only", (True, False): "old defaults only", (False, False): "neither"}[(po, pn)]  # fmt: skip
        names = n["n_dropped_calendar"] - n["n_dropped_calendar_index"] if pn else NAN
        djx = n["n_dropped_calendar_index"] if pn else NAN
        rows.append({
            "date": date, "priced": where,
            "status_old": o["status_rule"], "failing_old": o["failing"], "status_old_stored": o["status"], "reason_old_stored": o["reason"] if isinstance(o["reason"], str) else "",
            "status_new": n["status_rule"], "failing_new": n["failing"], "status_new_stored": n["status"], "reason_new_stored": n["reason"] if isinstance(n["reason"], str) else "",
            "n_dropped_old": o["n_dropped"], "n_dropped_new": n["n_dropped"], "n_dropped_index_old": o["n_dropped_index"], "n_dropped_index_new": n["n_dropped_index"],
            "names_slices_dropped_by_calendar_repair": names, "djx_slices_dropped_by_calendar_repair": djx,
            "n_names_unscreened": n["n_names_unscreened"], "names_unscreened": n["names_unscreened"] if isinstance(n["names_unscreened"], str) else "",
            "same_spec_key": bool(po and pn and o["spec_key"] == n["spec_key"]), "cache_hit_new": n["cache_hit"],
            "names_group": "" if not pn else "none" if names == 0 else "1-9" if names < 10 else "10 or more",
            "djx_group": "" if not pn else "a DJX slice dropped" if djx > 0 else "no DJX slice dropped",
        })  # fmt: skip
    return pd.DataFrame(rows).set_index("date", drop=False)


def check_decisions(new: pd.DataFrame, old: pd.DataFrame, dates: pd.DataFrame) -> dict[str, Any]:
    """What the split rests on, on the dates priced in both tables: the two runs differ by the
    calendar repairs alone (and by nothing where no slice is dropped)."""
    both = dates.index[dates["priced"] == "both"]
    o, n = old.loc[both], new.loc[both]
    names, djx = dates.loc[both, "names_slices_dropped_by_calendar_repair"], dates.loc[both, "djx_slices_dropped_by_calendar_repair"]  # fmt: skip
    for k in ("particle_seed", "pricing_seed", "n_particles", "n_paths", "companion_paths"):
        if not (o[k] == n[k]).all():
            raise ValueError(f"{k} differs between the two production tables")
    if (o["n_dropped_calendar"] != 0).any():
        raise ValueError("a slice dropped by a calendar repair in the old-defaults table")
    unchanged = (names == 0) & (djx == 0)
    return {
        "n_dropped_new_minus_old_is_calendar": bool(((n["n_dropped"] - o["n_dropped"]) == n["n_dropped_calendar"]).all()),
        "n_dropped_index_new_minus_old_is_calendar_index": bool(((n["n_dropped_index"] - o["n_dropped_index"]) == djx).all()),
        "no_name_unscreened_on_common_dates": bool((n["n_names_unscreened"] == 0).all()),
        "same_spec_key_iff_no_slice_dropped": bool((dates.loc[both, "same_spec_key"] == unchanged).all()),
        "n_unchanged": int(unchanged.sum()),
        "unchanged_dates": list(both[unchanged]),
        "unchanged_cache_hit": bool(truth(n.loc[unchanged, "cache_hit"]).all()),
    }  # fmt: skip


def b4_check(new_m: pd.DataFrame, old_m: pd.DataFrame, common: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:  # fmt: skip
    """The differences of the frozen table B4 recomputed from the two parquet tables."""
    frozen = pd.read_csv(B4).set_index("id")
    rows = []
    for date in pc.REFERENCE_DATES:
        for key, mine, factor in B4_KEYS:
            q = Q[mine]
            for kind in ("new_minus_old", "old_defaults"):
                rid = f"B.{date}.{kind}.{key}"
                if rid not in frozen.index:
                    continue
                f_value, f_se = float(frozen.loc[rid, "value"]), float(frozen.loc[rid, "se"])
                row = {"date": date, "frozen_id": rid, "quantity": q.label, "in_both_tables": date in common, "frozen_value": f_value, "frozen_se": f_se}  # fmt: skip
                if date in common:
                    o, n = float(old_m.loc[date, mine]), float(new_m.loc[date, mine])
                    os_, ns = float(old_m.loc[date, mine + "_se"]), float(new_m.loc[date, mine + "_se"])  # fmt: skip
                    value = factor * (n - o if kind == "new_minus_old" else o)
                    se = factor * (math.hypot(ns, os_) if kind == "new_minus_old" else os_)
                    row |= {"here_value": value, "here_se": se if q.has_se else NAN}
                    row["gap_value"] = abs(value - f_value) / max(abs(f_value), 1e-300)
                    row["gap_se"] = abs(se - f_se) / max(abs(f_se), 1e-300) if q.has_se and finite(f_se) else NAN  # fmt: skip
                rows.append(row)
    table = pd.DataFrame(rows)
    done = table[table["in_both_tables"]]
    gaps = pd.concat([done["gap_value"], done["gap_se"].dropna()])
    info = {
        "dates_in_both": [d for d in pc.REFERENCE_DATES if d in common],
        "dates_missing": [d for d in pc.REFERENCE_DATES if d not in common],
        "compared": len(gaps),
        "agree": int((gaps <= 1e-8).sum()),
        "worst": float(gaps.max()) if len(gaps) else NAN,
    }
    return table, info


# ------------------------------------------------------------------------------------------ part 2
def same_specification(prod: pd.DataFrame, dev: pd.DataFrame) -> tuple[pd.Series, str, int]:
    """The dates priced in both passes on which the two rows have the same specification, the
    rule used and the number of dates with an equal ``spec_key``."""
    if list(prod.index) != list(dev.index):
        raise ValueError("the development and the production tables do not hold the same dates")
    both = prod["priced"] & dev["priced"]
    n_key = 0
    if "spec_key" in prod.columns and "spec_key" in dev.columns:
        equal = both & (prod["spec_key"] == dev["spec_key"])
        n_key = int(equal.sum())
        if n_key:
            return equal, "equal spec_key", n_key
    same = both & (prod["n_dropped_calendar_index"] == 0) & (prod["n_dropped"] == dev["n_dropped"])
    return same, "n_dropped_calendar_index == 0 in the production row and equal n_dropped", n_key


def legs(prod_m: pd.DataFrame, dev_m: pd.DataFrame, dates: list[str]) -> pd.DataFrame:
    """The LC/CC difference per date and its two legs (exact): ``R_p − R_d = Δa/b_d −
    R_p·Δb/b_d`` with ``a = E_LC[D]``, ``b = E_CC[D]``."""
    a_p, a_d = prod_m.loc[dates, "ED_lc"], dev_m.loc[dates, "ED_lc"]
    b_p, b_d = prod_m.loc[dates, "ED_cc"], dev_m.loc[dates, "ED_cc"]
    r_p, r_d = prod_m.loc[dates, "lc_over_cc"], dev_m.loc[dates, "lc_over_cc"]
    out = pd.DataFrame({"date": dates, "difference": (r_p - r_d).to_numpy(), "leg_E_LC": ((a_p - a_d) / b_d).to_numpy(), "leg_E_CC": (-r_p * (b_p - b_d) / b_d).to_numpy()})  # fmt: skip
    out["d_rho_cc"] = (prod_m.loc[dates, "rho_cc"] - dev_m.loc[dates, "rho_cc"]).to_numpy()
    out["d_ln_E_CC"] = [math.log(p / d) for p, d in zip(b_p, b_d, strict=True)]
    out["equicorrelation_estimate_d_ln_E_CC"] = (-0.5 * out["d_rho_cc"].to_numpy() / (1.0 - dev_m.loc[dates, "rho_cc"]).to_numpy())  # fmt: skip
    if not ((out["leg_E_LC"] + out["leg_E_CC"] - out["difference"]).abs() <= 1e-12).all():
        raise ValueError("the two legs do not add up to the LC/CC difference")
    return out


# ------------------------------------------------------------------- other frozen statements
def frozen_rows_check(
    measured: dict[str, pd.DataFrame], frozen: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """The frozen records of the rows of the reference dates (sections A and B: the production
    row; C.6: today's row in the three tables) against the same numbers of the three tables:
    value and standard error, relative gap at most 1e-9.  ``frozen``: the records of the frozen
    ``numbers.json`` by id."""
    today = pc.REFERENCE_DATES[0]
    wanted = [(f"A.{f}", "new", today, k) for f, k in FROZEN_ROW_KEYS]
    wanted += [(f"B.{d}.{f}", "new", d, k) for d in pc.REFERENCE_DATES for f, k in FROZEN_ROW_KEYS]
    for tag, table in (("today_dev", "dev"), ("today_production_old_defaults", "old"), ("today_production", "new")):  # fmt: skip
        wanted += [(f"C.b.{tag}.{k}", table, today, k) for _, k in FROZEN_ROW_KEYS[2:]]
    rows = []
    for rid, table, date, key in wanted:
        if rid not in frozen:
            rows.append({"frozen_id": rid, "table": table, "date": date, "quantity": Q[key].label, "in_frozen": False})  # fmt: skip
            continue
        f_value, f_se = float(frozen[rid]["value"]), float(frozen[rid]["se"])
        value, se = float(measured[table].loc[date, key]), float(measured[table].loc[date, key + "_se"])  # fmt: skip
        rows.append({
            "frozen_id": rid, "table": table, "date": date, "quantity": Q[key].label, "in_frozen": True, "frozen_value": f_value, "frozen_se": f_se, "here_value": value, "here_se": se,
            "gap_value": abs(value - f_value) / abs(f_value), "gap_se": abs(se - f_se) / abs(f_se),
        })  # fmt: skip
    out = pd.DataFrame(rows)
    found = out[out["in_frozen"]]
    gaps = pd.concat([found["gap_value"], found["gap_se"]])
    info = {"records": len(found), "missing": int((~out["in_frozen"]).sum()), "compared": len(gaps), "agree": int((gaps <= 1e-9).sum()), "worst": float(gaps.max()) if len(gaps) else NAN}  # fmt: skip
    return out, info


def index_repair_check(new: pd.DataFrame) -> dict[str, Any]:
    """The frozen specification-only check of the index repair (``diagnostics/
    index_repair_3m.json``: the DJX slices the repair drops per date) against
    ``n_dropped_calendar_index`` of the production table."""
    frozen = json.loads(INDEX_REPAIR.read_text())
    built = [d for d, v in frozen.items() if v.get("status") == "ok"]
    n_frozen = {d: len(frozen[d].get("dropped", [])) for d in built}
    frozen_dates = sorted(d for d, n in n_frozen.items() if n > 0)
    here = new.loc[new["priced"], "n_dropped_calendar_index"]
    here_dates = sorted(here.index[here > 0])
    same = frozen_dates == here_dates and all(int(here[d]) == n_frozen[d] for d in here_dates)
    return {"n_built": len(built), "n_frozen": len(frozen_dates), "n_here": len(here_dates), "same_dates_and_counts": bool(same)}  # fmt: skip


def row_files_check(tables: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """The row files of the reference dates (``rows/<folder>/<date>.json``, the source of the
    frozen sections A and B) against the rows of the three tables: every number of a row file
    whose key is a column of the table."""
    compared, equal, worst, missing = 0, 0, 0.0, []
    for folder, name in ROW_FOLDERS:
        table = tables[name]
        for date in pc.REFERENCE_DATES:
            path = ROWS / folder / f"{date}.json"
            if not path.exists() or date not in table.index:
                missing.append(f"{folder}/{date}")
                continue
            for key, value in json.loads(path.read_text()).items():
                if key not in table.columns or isinstance(value, bool) or not isinstance(value, int | float):  # fmt: skip
                    continue
                there = table.loc[date, key]
                compared += 1
                if finite(value) and finite(there):
                    gap = abs(float(there) - value) / max(abs(value), 1e-300)
                else:
                    gap = 0.0 if not finite(value) and not finite(there) else math.inf
                equal += int(gap == 0.0)
                worst = max(worst, gap)
    return {"compared": compared, "equal": equal, "worst": worst, "missing": missing}


# ----------------------------------------------------------------------------------------- the page
def inputs_equal(
    prod: pd.DataFrame,
    dev: pd.DataFrame,
    dates: list[str],
    columns: tuple[str, ...] = INPUT_COLUMNS,
) -> tuple[int, list[str]]:
    """The number of ``dates`` on which every column of ``columns`` is identical in the two
    tables, and the columns that differ on some date."""
    a, b = prod.loc[dates, list(columns)], dev.loc[dates, list(columns)]
    equal = (a == b) | (a.isna() & b.isna())
    return int(equal.all(axis=1).sum()), [c for c in columns if not equal[c].all()]


def date_cell(per: pd.DataFrame, date: str, key: str) -> str:
    """``b − a ± bound`` of one quantity on one date, as a cell."""
    r = per[(per["quantity"] == key) & (per["date"] == date)].iloc[0]
    return cell(r["difference"], r["bound"], Q[key].digits, signed=True)


def build(out_dir: Path) -> tuple[str, list[dict[str, Any]], dict[str, pd.DataFrame]]:
    entries = load_entries()
    new, old, dev = load(NEW, entries), load(OLD, entries), load(DEV, entries)
    new_m, old_m, dev_m = measures(new), measures(old), measures(dev)
    c_new, c_old, c_dev = (str(one_value(t, "git_commit")) for t in (new, old, dev))
    b_new, b_old, b_dev = budget_of(new), budget_of(old), budget_of(dev)
    if b_new != b_old:
        raise ValueError("the two production tables are not at the same budget")
    seeds = {(one_value(t, "particle_seed"), one_value(t, "pricing_seed")) for t in (new, old, dev)}
    if len(seeds) != 1:
        raise ValueError(f"the three tables do not share their seeds: {seeds}")
    particle_seed, pricing_seed = (int(v) for v in next(iter(seeds)))
    src_new, src_old, src_dev = rel(NEW), rel(OLD), rel(DEV)
    tables: dict[str, pd.DataFrame] = {}
    records: list[dict[str, Any]] = []
    md: list[str] = []
    all_keys = tuple(Q)

    def count(rid: str, what: str, value: int, definition: str, source: str, commit: str, budget: str) -> str:  # fmt: skip
        records.append(pc.record(f"L.sens.{rid}", SECTION, what, value, unit="dates", definition=definition, source=source, commit=commit, budget=budget, notes="a count: no standard error"))  # fmt: skip
        return f"{value:d}"

    # ============================== part 1: the decisions
    dates = decision_dates(new, old)
    checks = check_decisions(new, old, dates)
    common = list(dates.index[dates["priced"] == "both"])
    only_new = list(dates.index[dates["priced"] == "new defaults only"])
    only_old = list(dates.index[dates["priced"] == "old defaults only"])
    neither = list(dates.index[dates["priced"] == "neither"])
    dd = dates.loc[common]
    names_n, djx_n = dd["names_slices_dropped_by_calendar_repair"], dd["djx_slices_dropped_by_calendar_repair"]  # fmt: skip
    check_new, check_old = list(dd.index[dd["status_new"] == "check"]), list(dd.index[dd["status_old"] == "check"])  # fmt: skip
    marks1 = {d: " †" for d in check_new} | {d: " ‡" for d in check_old if d not in check_new}
    s1 = Summary("decisions", old_m, new_m, ("the old defaults", "the new defaults", "new defaults minus old defaults"), False, marks1)  # fmt: skip
    alone = djx_n == 0
    groups = [
        ("all", "all dates priced in both tables", dd.index),
        ("changed", "a slice dropped by a calendar repair (the specification differs)", dd.index[(names_n > 0) | (djx_n > 0)]),
        ("names_0", "no slice of a name dropped by the calendar repair", dd.index[names_n == 0]),
        ("names_1_9", "1 to 9 slices of the names dropped by the calendar repair", dd.index[(names_n >= 1) & (names_n <= 9)]),
        ("names_10plus", "10 or more slices of the names dropped by the calendar repair", dd.index[names_n >= 10]),
        ("djx_no", "no DJX slice dropped (decision 5 does not act)", dd.index[alone]),
        ("djx_yes", "a DJX slice dropped (decision 5 acts)", dd.index[~alone]),
        ("d1_only_1_9", "decision 1 alone: 1 to 9 slices of the names, no DJX slice", dd.index[(names_n >= 1) & (names_n <= 9) & alone]),
        ("d1_only_10plus", "decision 1 alone: 10 or more slices of the names, no DJX slice", dd.index[(names_n >= 10) & alone]),
    ]  # fmt: skip
    # the two periods of 1.0, and the partition of the priced dates by what the repairs drop
    years = pd.Series([int(d[:4]) for d in new.index], index=new.index)
    early = years < PERIOD_SPLIT
    period_labels = {"period_1": f"{int(years.min())}–{PERIOD_SPLIT - 1}", "period_2": f"{PERIOD_SPLIT}–{int(years.max())}"}  # fmt: skip
    groups += [
        ("period_1", f"entry dates of {period_labels['period_1']}", dd.index[early[dd.index].to_numpy()]),
        ("period_2", f"entry dates of {period_labels['period_2']}", dd.index[~early[dd.index].to_numpy()]),
    ]  # fmt: skip
    for sample, label, idx in groups:
        s1.add(sample, label, list(idx), all_keys)
    priced_new = new.index[new["priced"]]
    names_all = (new["n_dropped_calendar"] - new["n_dropped_calendar_index"])[priced_new]
    djx_all = new.loc[priced_new, "n_dropped_calendar_index"]
    mix_all = pd.Series(
        ["djx_yes" if j > 0 else "names_0" if n == 0 else "d1_only_1_9" if n < 10 else "d1_only_10plus" for n, j in zip(names_all, djx_all, strict=True)],
        index=priced_new,
    )  # fmt: skip
    if sorted(d for g, _ in MIX for d in s1.samples[g][1]) != sorted(common) or any(mix_all[d] != g for g, _ in MIX for d in s1.samples[g][1]):  # fmt: skip
        raise ValueError("the four groups of MIX do not partition the common dates")
    per1 = by_date(old_m, new_m, common, all_keys, ("old", "new"))
    attrs1 = dd[["date", "names_slices_dropped_by_calendar_repair", "djx_slices_dropped_by_calendar_repair", "names_group", "djx_group", "status_old", "failing_old", "status_new", "failing_new"]]  # fmt: skip
    per1 = per1.drop(columns=["in_old_se"]).merge(attrs1.reset_index(drop=True), on="date", how="left")  # fmt: skip
    tables["decisions_dates"] = dates.reset_index(drop=True)
    tables["decisions_by_date"] = per1
    tables["decisions_summary"] = s1.frame()
    src1 = f"{src_new}; {src_old}; {STUDY_SOURCE}"
    commit1 = f"{c_new} minus {c_old}"
    runs1 = {
        "a": {"budget": b_old, "commit": c_old, "source": f"{src_old}; {STUDY_SOURCE}"},
        "b": {"budget": b_new, "commit": c_new, "source": f"{src_new}; {STUDY_SOURCE}"},
        "diff": {"budget": b_new, "commit": commit1, "source": src1},
    }  # fmt: skip
    records += s1.records(DECISION_KEYS, ("all", "names_1_9", "names_10plus", "djx_no", "djx_yes", "d1_only_1_9", "d1_only_10plus", "period_1", "period_2"), runs1)  # fmt: skip
    unchanged = checks["unchanged_dates"]
    unchanged_identical = bool((per1.loc[per1["date"].isin(unchanged), "difference"].dropna() == 0.0).all())  # fmt: skip
    compared_finite = bool(per1[per1["quantity"].isin(DECISION_KEYS)]["difference"].notna().all())
    nan_new = [d for d in dd.index if "check_no_nan" in dd.loc[d, "failing_new"]]
    nan_is_p25 = bool(nan_new) and all(not finite(new.loc[d, "idx_err_p25_se"]) for d in nan_new)
    # decision 1 alone: the rank correlation of the move of LC/CC with the number of slices dropped
    lc1 = per1[per1["quantity"] == "lc_over_cc"].set_index("date")["difference"]
    d1_dates = list(dd.index[(names_n >= 1) & alone])
    rank_corr = float(names_n[d1_dates].rank().corr(lc1[d1_dates].rank()))
    records.append(pc.record("L.sens.decisions.lc_over_cc.d1_only.rank_correlation_with_slices", SECTION, "rank correlation across dates of LC/CC (new minus old) with the number of names' slices dropped, decision 1 alone", rank_corr, definition="Pearson correlation of the ranks (Spearman) of `n_dropped_calendar − n_dropped_calendar_index` and of the LC/CC difference, on the common dates with a slice of a name dropped and no DJX slice dropped",
                             budget=b_new, commit=commit1, source=src1, n=len(d1_dates), notes="a correlation across dates: no standard error given"))  # fmt: skip
    # the sample: which dates the stopped pass ran, against the dates of the production table
    all_years = sorted(int(y) for y in years.unique())

    def per_year(idx: Any) -> pd.Series:
        return years[list(idx)].value_counts().reindex(all_years, fill_value=0)

    by_year = pd.DataFrame({
        "dates_of_the_production_table": per_year(new.index), "priced_in_the_production_table": per_year(priced_new),
        "run_by_the_old_defaults_pass": per_year(old.index), "priced_in_both_production_tables": per_year(common),
    })  # fmt: skip
    by_year.index.name = "year"
    tables["decisions_sample_by_year"] = by_year.reset_index()
    for y in all_years:
        for column, rid, what in (("run_by_the_old_defaults_pass", "run", "dates run by the stopped old-defaults pass"), ("dates_of_the_production_table", "table", "dates of the production table")):  # fmt: skip
            records.append(pc.record(f"L.sens.decisions.sample.year.{y}.{rid}", SECTION, f"{what} with an entry date in {y}", int(by_year.loc[y, column]), unit="dates", definition="rows of the table whose date is in the year, priced or not",
                                     budget=b_new, commit=c_old if rid == "run" else c_new, source=src_old if rid == "run" else src_new, notes="a count: no standard error"))  # fmt: skip
    mix_rows = []

    def mix_row(split: str, key: str, label: str, here: list[str], there: list[str]) -> None:
        mix_rows.append({
            "split": split, "sample": key, "label": label, "n_common": len(here), "share_common": len(here) / len(common), "n_priced": len(there), "share_priced": len(there) / len(priced_new),
            "names_slices_per_date_common": float(names_n[here].mean()) if here else NAN, "names_slices_per_date_priced": float(names_all[there].mean()) if there else NAN,
            "lc_over_cc_mean": s1.value("lc_over_cc", key, "mean"), "lc_over_cc_se": s1.get("lc_over_cc", key, "mean")["se"], "lc_over_cc_median": s1.value("lc_over_cc", key, "median"),
        })  # fmt: skip

    for key in ("period_1", "period_2"):
        mix_row("period", key, period_labels[key], s1.samples[key][1], [d for d in priced_new if bool(early[d]) == (key == "period_1")])  # fmt: skip
    for key, label in MIX:
        mix_row("group", key, label, s1.samples[key][1], list(mix_all.index[mix_all == key]))
    mix = pd.DataFrame(mix_rows).set_index("sample", drop=False)
    tables["decisions_sample_mix"] = mix.reset_index(drop=True)
    n_priced_new = len(priced_new)
    priced_def = "status not `failed` in `lcm_3m.parquet`; groups by `n_dropped_calendar − n_dropped_calendar_index` (names' slices) and `n_dropped_calendar_index` (DJX slices) of the production row; periods by the year of the entry date"
    for key in mix.index:
        r = mix.loc[key]
        records.append(pc.record(f"L.sens.decisions.sample.{key}.n_common", SECTION, f"dates priced in both production tables: {r['label']}", int(r["n_common"]), unit="dates", definition=priced_def, budget=b_new, commit=commit1, source=f"{src_new}; {src_old}", n=len(common), notes="a count: no standard error"))  # fmt: skip
        records.append(pc.record(f"L.sens.decisions.sample.{key}.n_priced", SECTION, f"priced dates of the production table: {r['label']}", int(r["n_priced"]), unit="dates", definition=priced_def, budget=b_new, commit=c_new, source=src_new, n=n_priced_new, notes="a count: no standard error"))  # fmt: skip
        records.append(pc.record(f"L.sens.decisions.sample.{key}.names_slices_per_date_common", SECTION, f"names' slices dropped by the calendar repair per date, mean over the dates priced in both production tables: {r['label']}", r["names_slices_per_date_common"], unit="slices per date", definition=priced_def, budget=b_new, commit=c_new, source=src_new, n=int(r["n_common"]), notes="a mean of counts across dates: no standard error given"))  # fmt: skip
        records.append(pc.record(f"L.sens.decisions.sample.{key}.names_slices_per_date_priced", SECTION, f"names' slices dropped by the calendar repair per date, mean over the priced dates of the production table: {r['label']}", r["names_slices_per_date_priced"], unit="slices per date", definition=priced_def, budget=b_new, commit=c_new, source=src_new, n=int(r["n_priced"]), notes="a mean of counts across dates: no standard error given"))  # fmt: skip

    def reweighted(key: str, split: str) -> float:
        """The means of ``key`` (new minus old) over the samples of ``split``, weighted by the
        samples' numbers of priced dates of the production table."""
        part = mix[mix["split"] == split]
        total = sum(n * s1.value(key, s, "mean") for s, n in zip(part["sample"], part["n_priced"], strict=True))  # fmt: skip
        return float(total / part["n_priced"].sum())

    tables["decisions_reweighted"] = pd.DataFrame([
        {"quantity": k, "label": Q[k].label, "n_common": len(common), "mean_common": s1.value(k, "all", "mean"), "n_priced": n_priced_new,
         "reweighted_to_the_groups_of_the_priced_dates": reweighted(k, "group"), "reweighted_to_the_periods_of_the_priced_dates": reweighted(k, "period")}
        for k in DECISION_KEYS
    ])  # fmt: skip
    rew_group, rew_period = reweighted("lc_over_cc", "group"), reweighted("lc_over_cc", "period")
    for rid, what, value, split in (("groups", f"the four groups by what the repairs drop ({'; '.join(label for _, label in MIX)})", rew_group, "group"), ("periods", f"the two periods ({period_labels['period_1']}, {period_labels['period_2']})", rew_period, "period")):  # fmt: skip
        records.append(pc.record(f"L.sens.decisions.lc_over_cc.reweighted_to_the_{rid}_of_the_priced_dates", SECTION, f"LC/CC, new defaults minus old defaults: the means of the common dates by sample, reweighted to the shares of the {n_priced_new} priced dates of the production table in {what}", value,
                                 definition=f"Σ_s n_s·m_s / Σ_s n_s: m_s the mean of the difference over the common dates of sample s ({split}), n_s the number of priced dates of the production table in s; {priced_def}", budget=b_new, commit=commit1, source=src1, n=len(common),
                                 notes=f"a reweighting of the differences of the {len(common)} common dates, not an estimate on the {n_priced_new} dates: {NO_ERROR}"))  # fmt: skip
    # how much of each mean a few dates supply
    conc_rows = []
    for key in CONCENTRATION_KEYS:
        moves = per1[per1["quantity"] == key].set_index("date")["difference"].astype(float)
        conc_rows.append({"quantity": key, "label": Q[key].label, **concentration(moves), "dates_from_the_largest_move": ", ".join(largest(moves, max(TOP_COUNTS)))})  # fmt: skip
    conc = pd.DataFrame(conc_rows).set_index("quantity", drop=False)
    tables["decisions_concentration"] = conc.reset_index(drop=True)
    for key in CONCENTRATION_KEYS:
        r, q = conc.loc[key], Q[key]
        for k in TOP_COUNTS:
            records.append(pc.record(f"L.sens.decisions.{key}.all.share_top_{k}", SECTION, f"{q.label}, new defaults minus old defaults: share of the sum over the common dates supplied by the {k} dates with the largest absolute difference", r[f"share_top_{k}"], unit="fraction of the sum",
                                     definition="Σ of the difference over the k dates with the largest |difference| / Σ over all common dates (signs kept)", budget=b_new, commit=commit1, source=src1, n=int(r["n"]), notes=f"a share of a sum, the dates selected on the outcome: {NO_ERROR}"))  # fmt: skip
        records.append(pc.record(f"L.sens.decisions.{key}.all.mean_without_top_{N_EXTREMES}", SECTION, f"{q.label}, new defaults minus old defaults: mean over the common dates without the {N_EXTREMES} dates with the largest absolute difference of the quantity", r["mean_without_top"], unit=q.unit,
                                 definition=q.definition, budget=b_new, commit=commit1, source=src1, n=int(r["n_without_top"]), notes=f"the dates left out are selected on the outcome: {NO_ERROR}; {int(r['n_positive_in_top'])} of the {N_EXTREMES} dates left out have a positive difference"))  # fmt: skip
        records.append(pc.record(f"L.sens.decisions.{key}.all.trimmed_mean", SECTION, f"{q.label}, new defaults minus old defaults: {100 * TRIM:.0f} % trimmed mean over the common dates", r["trimmed_mean"], unit=q.unit,
                                 definition=f"{q.definition}; mean of the sorted differences without the {int(r['n_cut_each_end'])} lowest and the {int(r['n_cut_each_end'])} highest", budget=b_new, commit=commit1, source=src1, n=int(r["n_trimmed"]), notes=f"a trimmed mean: {NO_ERROR}"))  # fmt: skip
    # the columns of a row that describe its inputs, old defaults against new defaults
    inputs_absent_1 = [c for c in INPUT_COLUMNS if c not in old.columns]
    inputs_in_both_1 = tuple(c for c in INPUT_COLUMNS if c in old.columns)
    _, inputs_differ_1 = inputs_equal(new, old, common, inputs_in_both_1)
    code_1 = changed_files(c_old, c_new)
    # the dates priced at the new defaults only
    only_rows = []
    for date in only_new:
        r = {"date": date, "name_unscreened": dates.loc[date, "names_unscreened"], "status_new": dates.loc[date, "status_new"], "failing_new": dates.loc[date, "failing_new"], "reason_old": dates.loc[date, "reason_old_stored"],
             "names_slices_dropped_by_calendar_repair": dates.loc[date, "names_slices_dropped_by_calendar_repair"], "djx_slices_dropped_by_calendar_repair": dates.loc[date, "djx_slices_dropped_by_calendar_repair"]}  # fmt: skip
        for key in DECISION_KEYS:
            r[key], r[key + "_se"] = float(new_m.loc[date, key]), float(
                new_m.loc[date, key + "_se"]
            )
            records.append(pc.record(
                f"L.sens.decisions.only_new.{date}.{key}", SECTION, f"{Q[key].label}, new defaults, a date priced at the new defaults only (a name kept unscreened: {r['name_unscreened']})", r[key], r[key + "_se"] if Q[key].has_se else None,
                date=date, unit=Q[key].unit, definition=Q[key].definition, budget=b_new, commit=c_new, source=f"{src_new}; {STUDY_SOURCE}",
                notes="flagged date (decision 2); pricing Monte Carlo standard error of the row" if Q[key].has_se else "flagged date (decision 2); a calibration diagnostic: no standard error",
            ))  # fmt: skip
        # the names' 2 % diagnostic of the row (decision 3: not a gate, so the status can be `ok`)
        r["names_second_moment_gap"] = float(new.loc[date, "sum_w_ER2_lc"] / new.loc[date, "sum_w_M"] - 1.0)  # fmt: skip
        r["names_second_moment_gap_se"] = float(new.loc[date, "sum_w_ER2_lc_se"] / new.loc[date, "sum_w_M"])  # fmt: skip
        records.append(pc.record(f"L.sens.decisions.only_new.{date}.names_second_moment_gap", SECTION, f"Σw E_LC[R_i²] over the listed strips minus 1, new defaults, a date priced at the new defaults only (a name kept unscreened: {r['name_unscreened']})", r["names_second_moment_gap"], r["names_second_moment_gap_se"],
                                 date=date, definition="`sum_w_ER2_lc` / `sum_w_M` − 1: the names' second moment under the model's Monte Carlo against the study's listed strips (the names' 2 % diagnostic); se: `sum_w_ER2_lc_se` / `sum_w_M`", budget=b_new, commit=c_new, source=src_new,
                                 notes="flagged date (decision 2); pricing Monte Carlo standard error of the row; the frozen package's reader note 8: this moment is not pinned down in the call wing"))  # fmt: skip
        only_rows.append(r)
    tables["decisions_only_new_defaults"] = pd.DataFrame(only_rows)
    # B4
    b4_table, b4 = b4_check(new_m, old_m, common)
    tables["decisions_b4_check"] = b4_table
    b4_def = "values and standard errors of the records `B.<date>.new_minus_old.<key>` and `B.<date>.old_defaults.<key>` of the frozen `tables/B_sensitivity.csv` against the same recomputed from `lcm_3m.parquet` and `lcm_3m_norepair.parquet`; agreement: relative gap at most 1e-8 (the CSV holds 10 significant digits)"
    for name, value in (("compared", b4["compared"]), ("agree", b4["agree"])):
        records.append(pc.record(f"L.sens.decisions.b4.{name}", SECTION, f"table B4 of the frozen package recomputed from the two tables: numbers {name}", value, unit="numbers", definition=b4_def, budget=b_new, commit=commit1, source=f"{rel(B4)}; {src1}", notes="a count: no standard error"))  # fmt: skip
    records.append(pc.record("L.sens.decisions.b4.worst_relative_gap", SECTION, "table B4 of the frozen package recomputed from the two tables: largest relative gap", b4["worst"], definition="largest |recomputed − frozen| / |frozen| over the compared values and standard errors", budget=b_new, commit=commit1, source=f"{rel(B4)}; {src1}", notes="a maximum: no standard error"))  # fmt: skip

    # ============================== part 2: the budget
    same, rule, n_key = same_specification(new, dev)
    priced_both = new["priced"] & dev["priced"]
    flagged = new["n_names_unscreened"] > 0
    djx_any = new["n_dropped_calendar_index"] > 0
    marks2 = {d: (" †" if new.loc[d, "status_rule"] == "check" else "") + (" ‡" if flagged[d] else "") for d in new.index[priced_both]}  # fmt: skip
    marks2 = {d: m for d, m in marks2.items() if m}
    s2 = Summary("budget", dev_m, new_m, ("the development budget", "the production budget", "production minus development"), True, marks2)  # fmt: skip
    samples2 = [
        ("same", "same specification in the two passes", new.index[same]),
        ("same_unflagged", "same specification, without the flagged dates (a name kept unscreened)", new.index[same & ~flagged]),
        ("same_ok", "same specification, status `ok` in the production row", new.index[same & (new["status_rule"] == "ok")]),
        ("other", "priced in both passes, another specification (decision 5 drops a DJX slice in the production pass): budget and specification both differ", new.index[priced_both & ~same]),
        ("all", "all dates priced in both passes (the specification differs on some)", new.index[priced_both]),
        ("all_unflagged", "all dates priced in both passes, without the flagged dates (the specification differs on some)", new.index[priced_both & ~flagged]),
    ]  # fmt: skip
    for sample, label, idx in samples2:
        s2.add(sample, label, list(idx), all_keys)
    same_dates, other_dates = list(new.index[same]), list(new.index[priced_both & ~same])
    other_not_djx = int((priced_both & ~same & ~djx_any).sum())
    n_inputs_equal, inputs_differ = inputs_equal(new, dev, same_dates)
    _, inputs_differ_other = inputs_equal(new, dev, other_dates)
    per2 = by_date(dev_m, new_m, list(new.index[priced_both]), all_keys, ("development", "production"))  # fmt: skip
    attrs2 = pd.DataFrame({"date": new.index, "same_specification": same.to_numpy(), "flagged": flagged.to_numpy(), "djx_slices_dropped_by_calendar_repair": new["n_dropped_calendar_index"].to_numpy(),
                           "status_development": dev["status_rule"].to_numpy(), "status_production": new["status_rule"].to_numpy(), "failing_production": new["failing"].to_numpy()})  # fmt: skip
    tables["budget_by_date"] = per2.merge(attrs2, on="date", how="left")
    tables["budget_summary"] = s2.frame()
    leg = legs(new_m, dev_m, same_dates)
    tables["budget_lc_over_cc_legs"] = leg
    src2 = f"{src_new}; {src_dev}; {STUDY_SOURCE}"
    commit2 = f"{c_new} minus {c_dev}"
    budget2 = f"production ({b_new}) minus development ({b_dev})"
    runs2 = {
        "a": {"budget": b_dev, "commit": c_dev, "source": f"{src_dev}; {STUDY_SOURCE}"},
        "b": {"budget": b_new, "commit": c_new, "source": f"{src_new}; {STUDY_SOURCE}"},
        "diff": {"budget": budget2, "commit": commit2, "source": src2},
    }  # fmt: skip
    records += s2.records(BUDGET_KEYS, ("same", "same_unflagged", "same_ok", "other", "all", "all_unflagged"), runs2)  # fmt: skip
    n_leg = len(leg)
    leg_stats = {}
    for column, what in (("leg_E_LC", "the leg of E_LC[D]: ΔE_LC[D] / E_CC[D] of the development row"), ("leg_E_CC", "the leg of E_CC[D]: −(LC/CC of the production row)·ΔE_CC[D] / E_CC[D] of the development row"),
                         ("d_ln_E_CC", "ln E_CC[D], production minus development"), ("equicorrelation_estimate_d_ln_E_CC", "−½·Δρ_CC/(1 − ρ_CC): the first-order change of ln E_CC[D] for the change of ρ_CC if D scaled as √(1 − ρ) (an equicorrelated basket: an approximation)")):  # fmt: skip
        v = leg[column]
        leg_stats[column] = (float(v.mean()), se_mean(v), int((v < 0).sum()), int((v > 0).sum()))
        records.append(pc.record(f"L.sens.budget.legs.{column}.mean", SECTION, f"LC/CC, production minus development, same specification: {what}; mean across dates", leg_stats[column][0], leg_stats[column][1],
                                 definition="LC/CC_prod − LC/CC_dev = ΔE_LC[D]/E_CC[D]_dev − LC/CC_prod·ΔE_CC[D]/E_CC[D]_dev (exact); mean over the dates with the same specification", budget=budget2, commit=commit2, source=src2, n=n_leg,
                                 notes=f"{SE_MEAN}; below zero on {leg_stats[column][2]} dates, above on {leg_stats[column][3]}"))  # fmt: skip
    corr_rho = float(leg["d_rho_cc"].corr(leg["d_ln_E_CC"]))
    corr_diff = float(leg["d_rho_cc"].corr(leg["difference"]))
    for rid, what, value in (("corr_d_rho_cc_d_ln_E_CC", "Δρ_CC and Δ ln E_CC[D]", corr_rho), ("corr_d_rho_cc_d_lc_over_cc", "Δρ_CC and Δ(LC/CC)", corr_diff)):  # fmt: skip
        records.append(pc.record(f"L.sens.budget.legs.{rid}", SECTION, f"correlation across dates of {what}, production minus development, same specification", value, definition="Pearson correlation across the dates with the same specification", budget=budget2, commit=commit2, source=src2, n=n_leg, notes="a correlation across dates: no standard error given"))  # fmt: skip
    code_2 = changed_files(c_dev, c_new)
    # the share of the mean LC/CC difference that each leg supplies
    leg_total = float(leg["difference"].mean())
    leg_share = {c: leg_stats[c][0] / leg_total for c in ("leg_E_LC", "leg_E_CC")}
    n_lc_leg_larger = int((leg["leg_E_LC"].abs() > leg["leg_E_CC"].abs()).sum())
    for column, what in (("leg_E_LC", "E_LC[D]"), ("leg_E_CC", "E_CC[D]")):
        records.append(pc.record(f"L.sens.budget.legs.{column}.share_of_mean", SECTION, f"LC/CC, production minus development, same specification: share of the mean difference supplied by the leg of {what}", leg_share[column], unit="fraction of the mean difference",
                                 definition="mean of the leg across the dates / mean of the LC/CC difference across the dates; the two legs add up to the difference on every date", budget=budget2, commit=commit2, source=src2, n=n_leg, notes=f"a ratio of two means across the same dates: {NO_ERROR}"))  # fmt: skip
    records.append(pc.record("L.sens.budget.legs.n_leg_E_LC_larger", SECTION, "LC/CC, production minus development, same specification: dates on which the leg of E_LC[D] is larger in size than the leg of E_CC[D]", n_lc_leg_larger, unit="dates",
                             definition="|ΔE_LC[D]/E_CC[D]_dev| > |LC/CC_prod·ΔE_CC[D]/E_CC[D]_dev|", budget=budget2, commit=commit2, source=src2, n=n_leg, notes="a count: no standard error"))  # fmt: skip
    # the sizes against the two ends of the frozen sentence
    lc2 = per2[per2["quantity"] == "lc_over_cc"].set_index("date")["difference"].astype(float)
    size = lc2[same_dates].abs()
    above_dates, below_dates = list(size.index[size > FROZEN_BUDGET_RANGE[1]]), list(size.index[size < FROZEN_BUDGET_RANGE[0]])  # fmt: skip
    for rid, what, value in (("n_above_frozen_upper_end", f"above {FROZEN_BUDGET_RANGE[1]}", len(above_dates)), ("n_below_frozen_lower_end", f"below {FROZEN_BUDGET_RANGE[0]}", len(below_dates))):  # fmt: skip
        records.append(pc.record(f"L.sens.budget.lc_over_cc.same.{rid}", SECTION, f"LC/CC, production minus development, same specification: dates on which the size of the difference is {what}, an end of the range of the frozen reader note 1", value, unit="dates",
                                 definition="|LC/CC_prod − LC/CC_dev|, not rounded, against the printed end of the frozen range", budget=budget2, commit=commit2, source=src2, n=n_leg, notes="a count: no standard error"))  # fmt: skip
    # the dates with another specification: decision 5 together with the budget
    other_sd = float(lc2[other_dates].std(ddof=1)) if len(other_dates) > 1 else NAN
    m_other, m_same = s2.get("lc_over_cc", "other", "mean"), s2.get("lc_over_cc", "same", "mean")
    net5, net5_se = float(m_other["value"]) - float(m_same["value"]), math.hypot(float(m_other["se"] or NAN), float(m_same["se"] or NAN))  # fmt: skip
    records.append(pc.record("L.sens.budget.lc_over_cc.other_minus_same.mean", SECTION, "LC/CC, production minus development: mean over the dates with another specification (decision 5 drops a DJX slice in the production pass) minus mean over the dates with the same specification", net5, net5_se,
                             definition="difference of the two means across dates; se: √(se_other² + se_same²), the two samples of dates treated as independent", budget=budget2, commit=commit2, source=src2, n=len(other_dates) + n_leg,
                             notes="it is the effect of decision 5 only if the budget effect on the dates with another specification is that of the dates with the same specification"))  # fmt: skip
    records.append(pc.record("L.sens.budget.lc_over_cc.other.sd", SECTION, "LC/CC, production minus development: standard deviation across the dates with another specification", other_sd, definition="sample standard deviation (n − 1) of the per-date differences", budget=budget2, commit=commit2, source=src2, n=len(other_dates), notes="a standard deviation across dates: no standard error given"))  # fmt: skip

    # the basket forward error (pricing noise alone): the bound of a difference, and the dates
    def over_bound(per: pd.DataFrame, idx: list[str]) -> dict[str, Any]:
        r = per[(per["quantity"] == "forward_error") & per["date"].isin(idx)].set_index("date")["over_bound"].astype(float)  # fmt: skip
        return {"n": len(r), "rms": math.sqrt(float((r**2).mean())), "max": float(r.abs().max()), "date": str(r.abs().idxmax()), "n_above_1": int((r.abs() > 1.0).sum())}  # fmt: skip

    fwd_bound = {"decisions": over_bound(per1, s1.samples["changed"][1]), "budget": over_bound(per2, same_dates)}  # fmt: skip
    fwd_runs = {"decisions": (b_new, commit1, src1, "new defaults minus old defaults, the common dates with a dropped slice"), "budget": (budget2, commit2, src2, "production minus development, same specification")}  # fmt: skip
    for tag, st in fwd_bound.items():
        bud, com, src, what = fwd_runs[tag]
        records.append(pc.record(f"L.sens.{tag}.forward_error.over_bound.rms", SECTION, f"basket forward error, {what}: root mean square across dates of the difference over its bound", st["rms"], definition=f"difference of `forward_error` over √(se_a² + se_b²) per date; {Q['forward_error'].definition}",
                                 budget=bud, commit=com, source=src, n=st["n"], notes="two independent runs would give a root mean square of 1; no standard error given"))  # fmt: skip
        records.append(pc.record(f"L.sens.{tag}.forward_error.over_bound.max", SECTION, f"basket forward error, {what}: largest size across dates of the difference over its bound", st["max"], date=st["date"], definition=f"difference of `forward_error` over √(se_a² + se_b²) per date; {Q['forward_error'].definition}",
                                 budget=bud, commit=com, source=src, n=st["n"], notes=f"a maximum: no standard error; above 1 on {st['n_above_1']} dates"))  # fmt: skip
    fwd_z = {}
    for tag, t, run in (("development", dev_m, runs2["a"]), ("production", new_m, runs2["b"])):
        z = (t.loc[same_dates, "forward_error"] / t.loc[same_dates, "forward_error_se"]).astype(float)  # fmt: skip
        fwd_z[tag] = {"mean": float(z.mean()), "sd": float(z.std(ddof=1)), "n_neg": int((z < 0).sum())}  # fmt: skip
        records.append(pc.record(f"L.sens.budget.forward_error.in_own_se.{tag}.mean", SECTION, f"basket forward error over its own standard error, {tag} pass, the dates with the same specification: mean across dates", fwd_z[tag]["mean"], se_mean(z), definition=f"`forward_error` / `forward_error_se` per date; {Q['forward_error'].definition}",
                                 budget=run["budget"], commit=run["commit"], source=run["source"], n=len(z), notes=f"{SE_MEAN}; below zero on {fwd_z[tag]['n_neg']} dates; independent draws date by date would give a mean of 0 and a standard deviation of 1"))  # fmt: skip
        records.append(pc.record(f"L.sens.budget.forward_error.in_own_se.{tag}.sd", SECTION, f"basket forward error over its own standard error, {tag} pass, the dates with the same specification: standard deviation across dates", fwd_z[tag]["sd"], definition=f"`forward_error` / `forward_error_se` per date; {Q['forward_error'].definition}",
                                 budget=run["budget"], commit=run["commit"], source=run["source"], n=len(z), notes="a standard deviation across dates: no standard error given"))  # fmt: skip
    # the size of the companion's fitting error: an order of magnitude from the rows
    cc_slope = leg_stats["leg_E_CC"][0] / float(leg["d_rho_cc"].mean())
    fit = {}
    for tag, t, run in (("development", dev, runs2["a"]), ("production", new, runs2["b"])):
        rel_se = (t.loc[same_dates, "straddle_B_cc_se"] / t.loc[same_dates, "straddle_B_cc"]).astype(float)  # fmt: skip
        scale = math.sqrt(float(one_value(t, "n_paths")) / float(one_value(t, "companion_paths")))
        on_rho = 2.0 * t.loc[same_dates, "rho_cc"].astype(float) * rel_se * scale
        fit[tag] = {"pricing": float(rel_se.mean()), "companion": float((rel_se * scale).mean()), "rho": float(on_rho.mean()), "lc_over_cc": float(on_rho.mean()) * abs(cc_slope)}  # fmt: skip
        fit_def = "r = `straddle_B_cc_se` / `straddle_B_cc` of the row (the pricing Monte Carlo relative error of E|R̄| under CC), scaled by √(`n_paths` / `companion_paths`) to the paths of the companion's fit"
        fit_note = "an order of magnitude, not a measurement: it takes the fit's straddle to have the relative error per path of the pricing straddle, and the straddle to be proportional to √ρ_CC; no pass was run with another seed"
        records.append(pc.record(f"L.sens.budget.companion_fit.{tag}.straddle_relative_error", SECTION, f"relative Monte Carlo error of the basket straddle under CC at the number of paths of the companion's fit, {tag} budget: mean across the dates with the same specification", fit[tag]["companion"], unit="fraction of the straddle",
                                 definition=fit_def, budget=run["budget"], commit=run["commit"], source=run["source"], n=n_leg, notes=fit_note))  # fmt: skip
        records.append(pc.record(f"L.sens.budget.companion_fit.{tag}.on_rho_cc", SECTION, f"order of magnitude of the companion's fitting error on ρ_CC, {tag} budget: mean across the dates with the same specification", fit[tag]["rho"],
                                 definition=f"2·ρ_CC·r per date; {fit_def}", budget=run["budget"], commit=run["commit"], source=run["source"], n=n_leg, notes=fit_note))  # fmt: skip
        records.append(pc.record(f"L.sens.budget.companion_fit.{tag}.on_lc_over_cc", SECTION, f"order of magnitude of the companion's fitting error on LC/CC, {tag} budget", fit[tag]["lc_over_cc"],
                                 definition=f"the value on ρ_CC times |mean leg of E_CC[D] / mean Δρ_CC| of the budget comparison; {fit_def}", budget=run["budget"], commit=run["commit"], source=run["source"], n=n_leg, notes=fit_note))  # fmt: skip
    records.append(pc.record("L.sens.budget.legs.leg_E_CC_over_d_rho_cc", SECTION, "LC/CC, production minus development, same specification: mean leg of E_CC[D] over mean Δρ_CC", cc_slope, definition="ratio of the two means across the dates with the same specification",
                             budget=budget2, commit=commit2, source=src2, n=n_leg, notes=f"a ratio of two means across the same dates: {NO_ERROR}"))  # fmt: skip
    # the frozen sentence on the four dates
    ref_rows = []
    for date in pc.REFERENCE_DATES:
        row = {"date": date, "same_specification": bool(same.get(date, False))}
        for key in BUDGET_KEYS:
            d_, p_ = float(dev_m.loc[date, key]), float(new_m.loc[date, key])
            ds, ps = float(dev_m.loc[date, key + "_se"]), float(new_m.loc[date, key + "_se"])
            row |= {f"{key}_development": d_, f"{key}_development_se": ds, f"{key}_production": p_, f"{key}_production_se": ps, f"{key}_difference": p_ - d_, f"{key}_bound": math.hypot(ds, ps)}  # fmt: skip
            records.append(pc.record(f"L.sens.budget.reference.{date}.{key}", SECTION, f"{Q[key].label}: production minus development, {date}", p_ - d_, math.hypot(ds, ps) if Q[key].has_se else None, date=date, unit=Q[key].unit, definition=Q[key].definition,
                                     budget=budget2, commit=commit2, source=src2, notes=f"the ± is an upper bound: {BOUND}" if Q[key].has_se else "a calibration output: no standard error"))  # fmt: skip
        ref_rows.append(row)
    ref = pd.DataFrame(ref_rows).set_index("date", drop=False)
    tables["budget_reference_dates"] = ref.reset_index(drop=True)
    ref_diff = ref["lc_over_cc_difference"]
    frozen_range_ok = (round(float(ref_diff.abs().min()), 4), round(float(ref_diff.abs().max()), 4)) == FROZEN_BUDGET_RANGE  # fmt: skip
    frozen_sign_ok = bool((ref_diff < 0).all()) and bool(ref["same_specification"].all())

    # ============================== other frozen statements (the errata section)
    frozen_records = {r["id"]: r for r in json.loads(NUMBERS.read_text())["records"]}
    frozen_rows, fr = frozen_rows_check({"new": new_m, "old": old_m, "dev": dev_m}, frozen_records)
    tables["frozen_rows_check"] = frozen_rows
    ir = index_repair_check(new)
    rf = row_files_check({"new": new, "old": old, "dev": dev})
    src_all = f"{src_new}; {src_old}; {src_dev}"
    for rid, what, value, definition, source in (
        ("rows.compared", "frozen records of the rows of the reference dates recomputed from the three tables: numbers compared", fr["compared"], "values and standard errors of the frozen records `A.<key>`, `B.<date>.<key>` (E_LC[D], E_CC[D], LC/CC, LC/copula, CC/copula) and `C.b.today_*.<key>` against the same numbers of the three tables", f"{rel(NUMBERS)}; {src_all}; {STUDY_SOURCE}"),
        ("rows.agree", "frozen records of the rows of the reference dates recomputed from the three tables: numbers that agree", fr["agree"], "relative gap at most 1e-9", f"{rel(NUMBERS)}; {src_all}; {STUDY_SOURCE}"),
        ("index_repair.n_frozen", "dates with a dropped DJX slice in the frozen specification-only check of the index repair", ir["n_frozen"], f"dates of `diagnostics/index_repair_3m.json` with status `ok` ({ir['n_built']}) and a non-empty `dropped`", rel(INDEX_REPAIR)),
        ("index_repair.n_here", "priced dates of the production table on which the calendar repair drops a DJX slice", ir["n_here"], "`n_dropped_calendar_index > 0`; the dates and the numbers of slices are " + ("those of the frozen check" if ir["same_dates_and_counts"] else "NOT those of the frozen check"), src_new),
        ("row_files.compared", "numbers of the row files of the reference dates compared with the rows of the three tables", rf["compared"], "every number of `rows/<folder>/<date>.json` whose key is a column of the table, for the four reference dates and the three row folders", f"{rel(ROWS)}; {src_all}"),
        ("row_files.equal", "numbers of the row files of the reference dates equal to those of the rows of the three tables", rf["equal"], "relative gap 0", f"{rel(ROWS)}; {src_all}"),
    ):  # fmt: skip
        records.append(pc.record(f"L.sens.frozen.{rid}", SECTION, what, value, unit="dates" if rid.startswith("index_repair") else "numbers", definition=definition, source=source, commit=f"{c_new}; {c_old}; {c_dev}", budget=f"{b_new}; {b_dev}", notes="a count: no standard error"))  # fmt: skip

    # ============================== the page
    n81, n_new = len(old), len(new)
    st_new, st_old = dd["status_new"].value_counts(), dd["status_old"].value_counts()
    st_old_stored = dd["status_old_stored"].value_counts()

    def grouped(idx: Any, column: str) -> str:
        by: dict[str, list[str]] = {}
        for d in idx:
            by.setdefault(str(dates.loc[d, column]), []).append(d)
        return "; ".join(f"{what} ({', '.join(ds)})" for what, ds in by.items()) or "none"

    md.append(
        "# Sensitivities of the 3m production pass: the owner's decisions 1, 2 and 5, and the budget"
    )
    md.append("")
    md.append(
        "Produced after the freeze of 12:30 (2026-10-09) by `scripts/pm_later_sensitivity.py`; nothing of the frozen package is changed. Local correlation model (LC), its constant-correlation companion (CC) and the study's copula; 3-month Palladium forward on the Dow; one row per entry date. "
        "Tables: `tables/*.csv` of this folder; records: `sensitivity.json` (the same page and records are under `parts/` in the package's part format)."
    )
    md.append("")
    md.append("**What was compared.**")
    md.append("")
    md.append(
        f"1. *Decisions.* The production pass at the defaults of 9 Oct (`{src_new}`, commit {c_new}: decisions 1, 2 and 5 on; {n_new} dates) against the production pass at the old defaults, stopped this morning "
        f"(`{src_old}`, commit {c_old}: no calendar repair, no unscreened fallback; {n81} dates run). Budget of both: {b_new}. Differences are new minus old. "
        f"The {n81} dates are not a cross-section of the history: 1.0 says which they are. Every statistic of new minus old is a statistic of the {len(common)} of them that are priced in both tables, not of the history."
    )
    md.append(
        f"2. *Budget.* The production pass (the same table) against the development pass (`{src_dev}`, commit {c_dev}: decisions 1 and 2 on, decision 5 not in the code; {b_dev}), "
        "on the dates where the two rows have the same specification. Differences are production minus development."
    )
    md.append("")
    md.append(
        f"The three passes use the same seeds on every date (particle seed {particle_seed}, pricing seed {pricing_seed})."
    )
    md.append("")
    md.append(
        "**What each ± is.** A value of one row: its pricing Monte Carlo standard error given the calibrated model (LC/CC paired on common paths; a ratio to the copula: delta method with the copula's own error); the calibration's own noise, the companion's fit included, is not in it. "
        "A per-date difference of two runs: √(se_a² + se_b²), which is an upper bound of the pricing error of the difference and not its error — the two runs share their particle and pricing seeds, so part of their pricing noise is common, but they are two calibrations and are not paired path by path "
        "(for a ratio to the copula the copula's error, the same number in the two rows, is in the bound twice; 2.3 checks the bound on the basket forward error). A mean of differences: the standard error across dates (sd/√n, the dates treated as independent). "
        "A pooled ratio (Σ numerator / Σ denominator over the dates) and a difference of two: the across-dates linearisation. Medians, quartiles (linear interpolation), extremes, counts: no error. "
        f"A share of a sum, a mean without the largest moves, a trimmed mean and a reweighted mean: {NO_ERROR}. "
        "The clipped mass, ρ_CC and λ_c are calibration outputs without a standard error. Status is the one under the current rule (decision 3: the names' 2 % check is not a gate); `check` = a gating check not passed; flagged = a name kept unscreened. "
        "A difference printed as 0 without a sign is exactly zero; one printed with a sign and zeros only (+0.000, -0.000) is not zero and rounds to zero at the digits shown."
    )
    md.append("")

    # ---- 1
    md.append("## 1. New defaults against old defaults, production budget")
    md.append("")
    md.append("### 1.0 Dates")
    md.append("")
    n_common = count("decisions.n_common", "dates priced in both production tables (new defaults and old defaults)", len(common), "status not `failed` in `lcm_3m.parquet` and in `lcm_3m_norepair.parquet`", f"{src_new}; {src_old}", commit1, b_new)  # fmt: skip
    n_only_new = count("decisions.n_only_new", "dates of the old-defaults table priced at the new defaults only", len(only_new), "status `failed` in `lcm_3m_norepair.parquet`, not `failed` in `lcm_3m.parquet`", f"{src_new}; {src_old}", commit1, b_new)  # fmt: skip
    n_neither = count("decisions.n_neither", "dates of the old-defaults table priced in neither table", len(neither), "status `failed` in both tables", f"{src_new}; {src_old}", commit1, b_new)  # fmt: skip
    n_only_old = count("decisions.n_only_old", "dates of the old-defaults table priced at the old defaults only", len(only_old), "status not `failed` in `lcm_3m_norepair.parquet`, `failed` in `lcm_3m.parquet`", f"{src_new}; {src_old}", commit1, b_new)  # fmt: skip
    md.append(
        f"The old-defaults pass was stopped after {n81} of the {n_new} dates; the other {n_new - n81} dates have no old-defaults row. Of the {n81}: {n_common} are priced in both tables; "
        f"{n_only_new} are priced at the new defaults only; {n_neither} in neither; {n_only_old} at the old defaults only."
    )
    md.append("")
    # which dates the stopped pass ran
    run_y, table_y = by_year["run_by_the_old_defaults_pass"], by_year["dates_of_the_production_table"]  # fmt: skip
    y_early, y_late = [y for y in all_years if y < PERIOD_SPLIT], [y for y in all_years if y >= PERIOD_SPLIT]  # fmt: skip
    p1, p2 = period_labels["period_1"], period_labels["period_2"]
    m1, m2, m10 = mix.loc["period_1"], mix.loc["period_2"], mix.loc["d1_only_10plus"]
    n_run_early, n_table_early = int(early[old.index].sum()), int(early.sum())
    n_flagged_priced = int((new.loc[priced_new, "n_names_unscreened"] > 0).sum())

    def of(n: int, total: int, what: str) -> str:
        return f"{n} of the {total} {what} ({pct(n / total)})"

    md.append(
        f"**Which dates the {n81} are.** The stopped pass did not run a cross-section of the history. Dates run / dates of the production table, by year: {', '.join(f'{y} {int(run_y[y])}/{int(table_y[y])}' for y in all_years)}: "
        f"{int(run_y[y_early].min())} to {int(run_y[y_early].max())} dates of each year of {p1} and {int(run_y[y_late].min())} to {int(run_y[y_late].max())} of each year of {p2}, "
        f"that is {of(n_run_early, n_table_early, f'dates of {p1}')} and {of(n81 - n_run_early, n_new - n_table_early, f'dates of {p2}')}. "
        f"The years {p1} hold {of(n_run_early, n81, 'dates run')}, against {of(n_table_early, n_new, 'dates of the production table')}, and {of(int(m1['n_common']), len(common), 'dates priced in both tables')}, against {of(int(m1['n_priced']), n_priced_new, 'priced dates of the production table')}. "
        "The dates, with what changed on each: `tables/decisions_dates.csv`."
    )
    md.append("")
    rows = []
    for key in mix.index:
        r = mix.loc[key]
        rows.append([
            f"entry dates of {r['label']}" if r["split"] == "period" else r["label"], f"{int(r['n_common'])} ({pct(r['share_common'])})", f"{int(r['n_priced'])} ({pct(r['share_priced'])})",
            f"{r['names_slices_per_date_common']:.1f} / {r['names_slices_per_date_priced']:.1f}", f"{s1.cell('lc_over_cc', key, 'mean')} ({s1.cell('lc_over_cc', key, 'median')})",
        ])  # fmt: skip
    md.append(md_table(
        ["sample", f"dates priced in both tables (of {len(common)})", f"priced dates of the production table (of {n_priced_new})", f"names' slices dropped by the calendar repair per date: the {len(common)} / the {n_priced_new}", "LC/CC, new − old, on the dates priced in both: mean ± se across dates (median)"],
        rows,
    ))  # fmt: skip
    md.append("")
    md.append(
        f"Every mean, median, quartile, extreme and count of new minus old in part 1 is a statistic of these {len(common)} dates; none is a statistic of the {n_priced_new} priced dates of the history. The sample and the history differ in two ways that the table shows. "
        f"By period: on the priced dates of the production table the calendar repair drops {m1['names_slices_per_date_priced']:.1f} names' slices per date in {p1} and {m2['names_slices_per_date_priced']:.1f} in {p2}, "
        f"and on the dates priced in both tables LC/CC moves by {s1.cell('lc_over_cc', 'period_1', 'mean')} in {p1} (n = {s1.n('period_1')}) and by {s1.cell('lc_over_cc', 'period_2', 'mean')} in {p2} (n = {s1.n('period_2')}). "
        f"By group: the dates with 10 or more names' slices dropped and no DJX slice are {of(int(m10['n_common']), len(common), 'dates priced in both tables')} and {of(int(m10['n_priced']), n_priced_new, 'priced dates of the production table')}. "
        f"Two reweightings of the {len(common)} differences of LC/CC (mean {s1.cell('lc_over_cc', 'all', 'mean')}), {NO_ERROR} for either: the four group means weighted by the groups' numbers of priced dates of the production table give {cell(rew_group, None, 5, True)}; "
        f"the two period means weighted by the periods' numbers of priced dates give {cell(rew_period, None, 5, True)}. They are reweightings of this sample, not estimates on the {n_priced_new} dates. "
        f"The {n_flagged_priced} flagged dates among the {n_priced_new} are counted in their group and period; on a flagged date no old-defaults price exists (1.4). "
        "By year: `tables/decisions_sample_by_year.csv`; the table above: `tables/decisions_sample_mix.csv`; the two reweightings for the other quantities: `tables/decisions_reweighted.csv`."
    )
    md.append("")
    md.append(
        f"- *New defaults only* ({len(only_new)}). At the old defaults each FAILED before pricing: {grouped(only_new, 'reason_old_stored')}. At the new defaults the name is kept on its unscreened expiries (decision 2): these are flagged dates; their values are in 1.4."
    )
    md.append(f"- *Neither* ({len(neither)}): " + ("; ".join(f"{d} — FAILED at the old defaults ({dates.loc[d, 'reason_old_stored']}) and FAILED at the new defaults ({dates.loc[d, 'reason_new_stored']})" for d in neither) or "none") + ".")  # fmt: skip
    md.append(f"- *Old defaults only* ({len(only_old)}): {', '.join(only_old) or 'none'}.")
    md.append(
        f"- *Status of the {len(common)} common dates.* New defaults: {int(st_new.get('ok', 0))} `ok`, {int(st_new.get('check', 0))} `check` ({grouped(check_new, 'failing_new')}). "
        f"Old defaults, same rule: {int(st_old.get('ok', 0))} `ok`, {int(st_old.get('check', 0))} `check` ({grouped(check_old, 'failing_old')}); as stored at commit {c_old}, where the names' 2 % check was still a gate: {int(st_old_stored.get('ok', 0))} `ok`, {int(st_old_stored.get('check', 0))} `check`. "
        + (
            "On every date that fails `check_no_nan` at the new defaults `idx_err_p25_se` (the error of the index error at +2.5 sd) is not finite; it is not a quantity of this page. "
            if nan_is_p25
            else ""
        )
        + (
            "Every quantity compared below is finite on every common date in both tables. "
            if compared_finite
            else "Some compared quantities are not finite on some dates. "
        )
        + "In the tables † marks a date with status `check` at the new defaults"
        + (
            " and ‡ one with status `check` at the old defaults only."
            if any(m == " ‡" for m in marks1.values())
            else " (the dates with status `check` at the old defaults are among them)."
        )
    )
    md.append(
        f"- *What changed on the common dates* (read from the new row). On each of them the new row's `n_dropped` exceeds the old row's by `n_dropped_calendar` ({'checked' if checks['n_dropped_new_minus_old_is_calendar'] else 'NOT the case on some dates'}), "
        f"the index's by `n_dropped_calendar_index` ({'checked' if checks['n_dropped_index_new_minus_old_is_calendar_index'] else 'NOT the case on some dates'}), and no name is kept unscreened ({'checked' if checks['no_name_unscreened_on_common_dates'] else 'NOT the case'}): "
        f"the two runs differ by the calendar repairs alone. Names' slices dropped by the calendar repair (decision 1): none on {s1.n('names_0')} dates, 1–9 on {s1.n('names_1_9')}, 10 or more on {s1.n('names_10plus')} (largest: {int(names_n.max())}, on {names_n.idxmax()}). "
        f"A DJX slice is dropped (decision 5) on {s1.n('djx_yes')} dates ({', '.join(f'{s1.dated(d)}: {int(djx_n[d])}' for d in dd.index[~alone])}); on "
        + (
            "each of them slices of the names are dropped too, so decision 5 is never seen alone in part 1 (in 2.4 it is seen with the budget)."
            if bool((names_n[~alone] > 0).all())
            else "some of them no slice of a name is dropped."
        )
        + f" Of the {len(INPUT_COLUMNS)} columns of a row that describe its inputs and settings (`INPUT_COLUMNS` of the script, listed in 2.0), {len(inputs_in_both_1)} are in both tables"
        + (
            f" (not in the old-defaults table: {', '.join(inputs_absent_1)})"
            if inputs_absent_1
            else ""
        )
        + f"; those that differ between the two rows of some common date are: {', '.join(inputs_differ_1) or 'none'}; "
        "the others are identical on every common date"
        + (
            ", the expiries dropped by the third-Friday, strikes and spread rules, the code tag, the step schedule and the seeds among them."
            if not {
                "n_dropped_third_friday",
                "n_dropped_strikes",
                "n_dropped_spread",
                "lc_code_tag",
                "schedule",
                "particle_seed",
                "pricing_seed",
            }
            & set(inputs_differ_1)
            else "."
        )
        + (
            ""
            if code_1 is None
            else f" Between the two commits ({c_old}, {c_new}) the files under `volsto/` that differ are: {', '.join(f'`{f}`' for f in code_1) or 'none'} (`git diff --name-only`)"
            + (
                ": no file of `volsto/calibration` or `volsto/models`."
                if not any(f.startswith(("volsto/calibration/", "volsto/models/")) for f in code_1)
                else "."
            )
        )
    )
    md.append(
        f"- *The {checks['n_unchanged']} dates with no slice dropped* ({', '.join(unchanged)}): the specification key is the same in the two tables ({'checked: equal key if and only if no slice is dropped' if checks['same_spec_key_iff_no_slice_dropped'] else 'NOT an equivalence'}), "
        f"the new run read the calibration of the old run from the cache ({'`cache_hit` true on each' if checks['unchanged_cache_hit'] else '`cache_hit` not true on each'}), and every compared number is {'identical: the difference is exactly 0' if unchanged_identical else 'NOT identical'}. They are in the summary of 1.1 with a difference of 0."
    )
    md.append("")

    # ---- 1.1 summary
    md.append(f"### 1.1 Summary: new minus old on the {len(common)} common dates")
    md.append("")
    rows = []
    for key in DECISION_KEYS:
        q = Q[key]
        rows.append([
            q.label, s1.cell(key, "all", "mean_a", signed=False), s1.cell(key, "all", "mean_b", signed=False), s1.cell(key, "all", "mean"),
            s1.cell(key, "all", "mc_bound_mean", signed=False) if q.has_se else "—",
            f"{s1.cell(key, 'all', 'median')} [{s1.cell(key, 'all', 'q25')}, {s1.cell(key, 'all', 'q75')}]",
            s1.cell(key, "all", "min", with_date=True), s1.cell(key, "all", "max", with_date=True), s1.signs(key, "all"),
            f"{int(s1.value(key, 'all', 'n_beyond_2_bounds'))}" if q.has_se else "—", s1.cell(key, "all", "pooled") if q.num else "",
        ])  # fmt: skip
    md.append(md_table(
        ["quantity", "old defaults: mean ± se", "new defaults: mean ± se", "new − old: mean ± se across dates", "MC bound of the mean", "median [q25, q75]", "min (date)", "max (date)", "dates below / above / at 0", "dates beyond 2 × their bound", "pooled: new − old ± se"],
        rows,
    ))  # fmt: skip
    md.append("")
    md.append(
        f"n = {len(common)} for every row. A level's ± (columns 2 and 3) and a mean difference's ± are standard errors across dates. \"MC bound of the mean\": the mean of the per-date bounds √(se_new² + se_old²), an upper bound of the pricing Monte Carlo error of the mean difference with the dates fixed (the dates share their seeds). "
        '"Dates beyond 2 × their bound": dates on which |new − old| exceeds twice its bound; the bound overstates the error, so this count understates how many dates moved by more than two standard errors. '
        "Pooled: Σ numerator / Σ denominator over the dates (E_LC[D]/E_CC[D], E[D]/P_D, calls/copula's calls, E[V]/EQV). The index errors are against each run's own index target; on the dates where decision 5 drops a DJX slice the target itself differs between the two runs. "
        "Two cautions of the frozen package apply to these rows as well: the calls at 1.25 × the forward are carried in part by a few runaway paths (its reader note 7), and the errors of κ and of E[V]/EQV are not their uncertainty, the names' second moment not being pinned down in the call wing (its reader note 8). "
        "† status `check` at the new defaults. Per date: `tables/decisions_by_date.csv`; all statistics, by sample: `tables/decisions_summary.csv`."
    )
    md.append("")

    # ---- 1.2 splits
    md.append("### 1.2 Splits by what changed: new − old, mean ± se across dates (median)")
    md.append("")
    split_cols = [("names_1_9", "names' slices dropped: 1–9"), ("names_10plus", "names' slices dropped: 10 or more"), ("djx_no", "no DJX slice dropped"), ("djx_yes", "a DJX slice dropped (decision 5)"), ("d1_only_1_9", "decision 1 alone: 1–9"), ("d1_only_10plus", "decision 1 alone: 10 or more")]  # fmt: skip
    rows = [[Q[key].label, *(f"{s1.cell(key, s, 'mean')} ({s1.cell(key, s, 'median')})" for s, _ in split_cols)] for key in DECISION_KEYS]  # fmt: skip
    md.append(
        md_table(["quantity", *(f"{label} (n = {s1.n(s)})" for s, label in split_cols)], rows)
    )
    md.append("")
    md.append(
        f"Columns 1–2 split the {s1.n('changed')} dates with a dropped slice by the number of names' slices, columns 3–4 split the {len(common)} common dates by decision 5 (the {s1.n('names_0')} dates with no slice dropped, difference exactly 0, are in column 3 and not in columns 1–2); "
        "columns 5–6 are the dates on which decision 1 acts alone (a slice of a name dropped, no DJX slice). Decision 2 has no column: a date on which a name is kept unscreened is priced at the new defaults only (1.4). "
        f"With decision 1 alone, LC/CC is lower at the new defaults on {int(s1.value('lc_over_cc', 'd1_only_1_9', 'n_neg'))} of the {s1.n('d1_only_1_9')} dates with 1–9 slices and on {int(s1.value('lc_over_cc', 'd1_only_10plus', 'n_neg'))} of the {s1.n('d1_only_10plus')} with 10 or more; "
        f"the rank correlation between the number of slices dropped and the LC/CC difference on these {len(d1_dates)} dates is {rank_corr:+.2f}."
    )
    md.append("")

    # ---- 1.3 extremes
    md.append("### 1.3 Extremes")
    md.append("")
    order = largest(lc1, N_EXTREMES)
    ext_rows = []
    for d in order:
        cells_ = [d, f"{int(names_n[d])}", f"{int(djx_n[d])}", f"{dates.loc[d, 'status_old']} / {dates.loc[d, 'status_new']}"]  # fmt: skip
        for key in ("lc_over_cc", "lc_over_copula", "cc_over_copula"):
            r = per1[(per1["quantity"] == key) & (per1["date"] == d)].iloc[0]
            cells_.append(f"{cell(r['old'], r['old_se'], 5)} → {cell(r['new'], r['new_se'], 5)}")
            cells_.append(cell(r["difference"], r["bound"], 5, signed=True))
            records.append(pc.record(f"L.sens.decisions.extreme.{d}.{key}", SECTION, f"{Q[key].label}: new defaults minus old defaults, {d} (one of the 8 largest moves of LC/CC)", r["difference"], r["bound"], date=d, definition=Q[key].definition, budget=b_new, commit=commit1, source=src1,
                                     notes=f"the ± is an upper bound: {BOUND}; old {r['old']:.6f} ± {r['old_se']:.6f}, new {r['new']:.6f} ± {r['new_se']:.6f}; names' slices dropped {int(names_n[d])}, DJX slices dropped {int(djx_n[d])}; status old / new: {dates.loc[d, 'status_old']} / {dates.loc[d, 'status_new']}"))  # fmt: skip
        ext_rows.append(cells_)
    md.append(
        f"The {N_EXTREMES} dates with the largest |new − old| of LC/CC (old → new; the ± of a difference is the upper bound):"
    )
    md.append("")
    md.append(md_table(["date", "names' slices dropped", "DJX slices dropped", "status old / new", "LC/CC old → new", "LC/CC new − old", "LC/copula old → new", "LC/copula new − old", "CC/copula old → new", "CC/copula new − old"], ext_rows))  # fmt: skip
    md.append("")
    n_ext_djx = sum(1 for d in order if djx_n[d] > 0)
    # the date that supplies most extremes of table 1.1
    ext_dates = pd.Series(
        [s1.get(k, "all", st)["date"] for k in DECISION_KEYS for st in ("min", "max")]
    )
    top, top_n = str(ext_dates.value_counts().idxmax()), int(ext_dates.value_counts().max())
    gap = {}
    for tag, t, c_, s_ in (("old", old, c_old, src_old), ("new", new, c_new, src_new)):
        gap[tag] = float(t.loc[top, "sum_w_ER2_lc"] / t.loc[top, "sum_w_M"] - 1.0)
        records.append(pc.record(f"L.sens.decisions.extreme.{top}.names_second_moment_gap.{tag}", SECTION, f"Σw E_LC[R_i²] over the listed strips minus 1, {top}, {tag} defaults", gap[tag], date=top, definition="`sum_w_ER2_lc` / `sum_w_M` − 1: the names' second moment under the model's Monte Carlo against the study's listed strips (the names' 2 % diagnostic)", budget=b_new, commit=c_, source=s_, notes="the Monte Carlo error of the date is `sum_w_ER2_lc_se` / `sum_w_M` of the source table; not read here"))  # fmt: skip

    def arrow(key: str, digits: int) -> str:
        return f"{cell(old_m.loc[top, key], old_m.loc[top, key + '_se'], digits)} → {cell(new_m.loc[top, key], new_m.loc[top, key + '_se'], digits)}"

    md.append(
        f"{n_ext_djx} of these {N_EXTREMES} dates have a DJX slice dropped. {s1.dated(top)} supplies {top_n} of the {2 * len(DECISION_KEYS)} extremes of table 1.1 ({int(names_n[top])} slice{'' if int(names_n[top]) == 1 else 's'} of the names dropped, {int(djx_n[top])} of the DJX; status old / new: {dates.loc[top, 'status_old']} / {dates.loc[top, 'status_new']}). "
        f"On that date the names' second moment under the model is {100 * gap['old']:+.1f} % from the listed strips at the old defaults and {100 * gap['new']:+.1f} % at the new defaults (the names' 2 % diagnostic). Old → new: "
        f"E_LC[V]/EQV {arrow('EV_over_EQV_lc', 3)}, κ_LC {arrow('kappa_lc', 4)}, E_LC[D] {arrow('ED_lc', 6)}, E_CC[D] {arrow('ED_cc', 6)}; LC/CC moves by {date_cell(per1, top, 'lc_over_cc')}."
    )
    md.append("")
    # how much of each mean a few dates supply
    c_lc = conc.loc["lc_over_cc"]
    md.append(
        f"How much of each mean a few dates supply. Share of the sum of the {len(common)} differences supplied by the dates with the largest |new − old| of the quantity itself (the sum over those dates over the sum over all {len(common)}, signs kept):"
    )
    md.append("")
    rows = []
    for key in CONCENTRATION_KEYS:
        r = conc.loc[key]
        rows.append([
            Q[key].label, cell(r["mean"], None, Q[key].digits, True), cell(r["median"], None, Q[key].digits, True), *(pct(r[f"share_top_{k}"]) for k in TOP_COUNTS),
            f"{int(r['n_negative_in_top'])} / {int(r['n_positive_in_top'])}", cell(r["mean_without_top"], None, Q[key].digits, True), cell(r["trimmed_mean"], None, Q[key].digits, True),
        ])  # fmt: skip
    md.append(md_table(
        ["quantity", f"mean (n = {len(common)})", "median", *(f"{k} largest" for k in TOP_COUNTS), f"negative / positive among its {N_EXTREMES} largest",
         f"mean without its {N_EXTREMES} largest (n = {int(c_lc['n_without_top'])})", f"{100 * TRIM:.0f} % trimmed mean (n = {int(c_lc['n_trimmed'])})"],
        rows,
    ))  # fmt: skip
    md.append("")
    md.append(
        f"The largest moves are each quantity's own; for LC/CC the {N_EXTREMES} largest are the dates of the table of extremes above. Trimmed mean: the mean without the {int(c_lc['n_cut_each_end'])} lowest and the {int(c_lc['n_cut_each_end'])} highest of the {len(common)} differences. "
        f"The dates left out are selected on the outcome: {NO_ERROR} for these two means. The dates, from the largest move: `tables/decisions_concentration.csv`."
    )
    md.append("")

    # ---- 1.4 only new
    md.append(
        "### 1.4 Dates priced at the new defaults only (decision 2: a name kept unscreened; flagged dates)"
    )
    md.append("")
    show = ("ED_lc", "ED_cc", "lc_over_cc", "lc_over_copula", "cc_over_copula", "kappa_lc", "EV_over_EQV_lc", "clip_inner_max", "idx_err_90", "delta_fwd_lc")  # fmt: skip
    on_moment = ("kappa_lc", "EV_over_EQV_lc")  # the cells that rest on the names' second moment

    def outside(r: dict[str, Any]) -> bool:
        return abs(r["names_second_moment_gap"]) > 0.02

    rows = [
        [r["date"], r["name_unscreened"], r["status_new"] + (f" ({r['failing_new']})" if r["failing_new"] else ""),
         f"{100 * r['names_second_moment_gap']:+.1f} % ± {100 * r['names_second_moment_gap_se']:.1f} %" + (" §" if outside(r) else ""),
         *(cell(r[k], r[k + "_se"], Q[k].digits) + (" §" if outside(r) and k in on_moment else "") for k in show)]
        for r in only_rows
    ]  # fmt: skip
    md.append(md_table(["date (flagged)", "name kept unscreened", "status", "names' second moment against the listed strips", *(Q[k].label for k in show)], rows))  # fmt: skip
    md.append("")
    out_rows = [r for r in only_rows if outside(r)]
    worst = max(only_rows, key=lambda r: abs(r["names_second_moment_gap"])) if only_rows else None
    md.append(
        "Values at the new defaults, ± the row's pricing Monte Carlo standard error; no old-defaults value exists for these dates. The other quantities of table 1.1 for these dates are in `tables/decisions_only_new_defaults.csv`. "
        "Names' second moment against the listed strips: Σw E_LC[R_i²] of the row over the study's listed strips, minus 1 (the names' 2 % diagnostic; under decision 3 it is not a gate, so a row outside 2 % keeps the status `ok`). "
        + (
            f"§ the diagnostic is outside 2 % on {len(out_rows)} of the {len(only_rows)} dates ({', '.join(r['date'] for r in out_rows)}); κ and E[V]/EQV of such a row rest on that moment, and the frozen package's reader note 8 applies to them. "
            f"On {worst['date']} the moment is {100 * worst['names_second_moment_gap']:+.1f} % ± {100 * worst['names_second_moment_gap_se']:.1f} % from the listed strips, with κ_LC {cell(worst['kappa_lc'], None, 3)} and E[V]/EQV {cell(worst['EV_over_EQV_lc'], None, 2)}."
            + (
                f' The frozen package\'s arithmetic does not use that date (its reader note 12: "{quoted(READER_NOTES, "the study drops 2009-06-01")}").'
                if worst["date"] == "2009-06-01"
                else ""
            )
            if out_rows and worst is not None
            else "The diagnostic is inside 2 % on each of these dates."
        )
    )
    md.append("")

    # ---- 1.5 B4
    md.append("### 1.5 Table B4 of the frozen package")
    md.append("")
    in_both = [d for d in pc.REFERENCE_DATES if d in common]
    b4_moves = []
    for d in in_both:
        sample = next(k for k in ("names_0", "d1_only_1_9", "d1_only_10plus", "djx_yes") if d in s1.samples[k][1])  # fmt: skip
        peers = s1.samples[sample][1]
        rank = int((lc1[peers] < lc1[d]).sum()) + 1
        b4_moves.append(f"{d}: {int(names_n[d])} names' slices and {int(djx_n[d])} DJX slices dropped, LC/CC {date_cell(per1, d, 'lc_over_cc')}, rank {rank} of the {len(peers)} dates of its group ({s1.samples[sample][0]})")  # fmt: skip
    md.append(
        f"B4 gives new minus old on the four reference dates ({', '.join(pc.REFERENCE_DATES)}) from the row files of the same two runs. "
        + (
            "All four dates are in both tables here. "
            if not b4["dates_missing"]
            else f"In both tables here: {', '.join(b4['dates_in_both'])}; not in both: {', '.join(b4['dates_missing'])}. "
        )
        + f"Recomputed from the two parquet tables, {b4['agree']} of the {b4['compared']} numbers of the frozen `tables/B_sensitivity.csv` that this page can form (the old-defaults values, the differences and their errors, for E_LC[D], E_CC[D], LC/CC, LC/copula, CC/copula, the two deltas, the clipped mass and the basket forward error) "
        f"agree to the 10 significant digits of the CSV (largest relative gap {b4['worst']:.1e}): "
        + (
            "B4 is reproduced."
            if b4["agree"] == b4["compared"]
            else "B4 is NOT reproduced on every number: `tables/decisions_b4_check.csv`."
        )
        + f" The four dates, with new minus old of LC/CC (± the upper bound) and its rank inside the date's group of 1.2, from the most negative: {'; '.join(b4_moves)}. "
        f"On the {len(common)} common dates the mean is {s1.cell('lc_over_cc', 'all', 'mean')}, the median {s1.cell('lc_over_cc', 'all', 'median')} and the quartiles {s1.cell('lc_over_cc', 'all', 'q25')} and {s1.cell('lc_over_cc', 'all', 'q75')}."
    )
    md.append("")

    # ---- 2
    md.append("## 2. Production budget against development budget, same specification")
    md.append("")
    md.append("### 2.0 Dates")
    md.append("")
    n_same = count("budget.n_same", "dates priced in both passes with the same specification", len(same_dates), rule, f"{src_new}; {src_dev}", commit2, budget2)  # fmt: skip
    n_both = count("budget.n_priced_both", "dates priced in the development and in the production pass", int(priced_both.sum()), "status not `failed` in both tables", f"{src_new}; {src_dev}", commit2, budget2)  # fmt: skip
    n_other = count("budget.n_other", "dates priced in both passes with another specification", len(other_dates), f"priced in both and not: {rule}", f"{src_new}; {src_dev}", commit2, budget2)  # fmt: skip
    n_flag = count("budget.n_same_flagged", "flagged dates (a name kept unscreened) among the dates with the same specification", int((same & flagged).sum()), "`n_names_unscreened > 0` in the production row", f"{src_new}; {src_dev}", commit2, budget2)  # fmt: skip
    n_check = count("budget.n_same_check", "dates with status `check` in the production row among the dates with the same specification", int((same & (new["status_rule"] == "check")).sum()), "a gating check (`check_no_nan`, `check_forward`, `check_index`) not passed in the production row", f"{src_new}; {src_dev}", commit2, budget2)  # fmt: skip
    failed = {name: "; ".join(f"{d} ({t.loc[d, 'reason']})" for d in t.index[~t["priced"]]) for name, t in (("production", new), ("development", dev))}  # fmt: skip
    md.append(
        f"The two tables hold the same {n_new} dates; {n_both} are priced in both. "
        + (
            f"FAILED in both passes, with the same reason: {failed['production']}. "
            if failed["production"] == failed["development"]
            else f"FAILED in the production pass: {failed['production']}. FAILED in the development pass: {failed['development']}. "
        )
        + f"The companion is fitted on {short(one_value(dev, 'companion_paths'))} paths at the development budget and {short(one_value(new, 'companion_paths'))} at the production budget. "
        + f"`spec_key` is in both tables and is equal on {n_key} dates: the key hashes the calibration's settings, the particle count among them, so it differs between the two budgets by construction. "
        f"The rule used is therefore: {rule}. It holds on {n_same} dates; on the other {n_other} "
        + (
            "decision 5 drops a DJX slice in the production pass"
            if other_not_djx == 0
            else f"the specification differs (on {other_not_djx} of them for another reason than a dropped DJX slice)"
        )
        + f", so budget and specification both differ there (2.4). On {n_inputs_equal} of the {n_same} dates the {len(INPUT_COLUMNS)} columns of a row that describe its inputs and settings and do not depend on the budget "
        f"(the dropped expiries by rule, the SVI fit errors of the names and of the index, the number of calibration slices, the SVI strips, the alignment of the index forward, the floored fractions of the local-volatility surfaces, the code tag, the step schedule, the seeds: `INPUT_COLUMNS` of the script) are identical in the two tables"
        + (
            ""
            if not inputs_differ
            else f" (they differ on some date for: {', '.join(inputs_differ)})"
        )
        + f"; on the other {n_other} dates these differ: {', '.join(inputs_differ_other) or 'none'}. "
        f"Of the {n_same} dates, {n_flag} are flagged (a name kept unscreened, ‡ in the tables) and {n_check} have status `check` in the production row (†)."
        + (
            ""
            if code_2 is None
            else f" Between the two commits ({c_dev}, {c_new}) the files under `volsto/` that differ are: {', '.join(f'`{f}`' for f in code_2) or 'none'} (`git diff --name-only`)"
            + (
                ": no file of `volsto/calibration` or `volsto/models`."
                if not any(f.startswith(("volsto/calibration/", "volsto/models/")) for f in code_2)
                else "."
            )
        )
    )
    md.append("")

    # ---- 2.1 summary
    kappa_notes = []
    for st in ("min", "max"):
        d = s2.get("kappa_lc", "same", st)["date"]
        kappa_notes.append(f"{s2.dated(d)}: {cell(dev.loc[d, 'sum_w_ER2_lc'], dev.loc[d, 'sum_w_ER2_lc_se'], 6)} → {cell(new.loc[d, 'sum_w_ER2_lc'], new.loc[d, 'sum_w_ER2_lc_se'], 6)}")  # fmt: skip
        for tag, t, run in (("development", dev, runs2["a"]), ("production", new, runs2["b"])):
            records.append(pc.record(f"L.sens.budget.kappa_extreme.{d}.sum_w_ER2_lc.{tag}", SECTION, f"Σw E_LC[R_i²], {d} ({tag} budget): the date of the {'smallest' if st == 'min' else 'largest'} production minus development of κ_LC", t.loc[d, "sum_w_ER2_lc"], t.loc[d, "sum_w_ER2_lc_se"], date=d,
                                     definition="`sum_w_ER2_lc`, `sum_w_ER2_lc_se`: the names' second moment under the model's Monte Carlo", budget=run["budget"], commit=run["commit"], source=run["source"], notes="pricing Monte Carlo standard error of the row"))  # fmt: skip
    md.append(
        f"### 2.1 Summary: production minus development on the {len(same_dates)} dates with the same specification"
    )
    md.append("")
    rows = []
    for key in BUDGET_KEYS:
        share = s2.get(key, "same", "share_same_sign")
        p = s2.value(key, "same", "sign_test_p")
        rows.append([
            Q[key].label, s2.cell(key, "same", "mean_a", signed=False), s2.cell(key, "same", "mean_b", signed=False), s2.cell(key, "same", "mean"),
            f"{s2.cell(key, 'same', 'median')} [{s2.cell(key, 'same', 'q25')}, {s2.cell(key, 'same', 'q75')}]",
            s2.cell(key, "same", "min", with_date=True), s2.cell(key, "same", "max", with_date=True), s2.signs(key, "same"),
            f"{100 * share['value']:.1f} % {share['note']}", f"{p:.1e}" if finite(p) else "n/a",
        ])  # fmt: skip
    md.append(md_table(
        ["quantity", "development: mean ± se", "production: mean ± se", "production − development: mean ± se across dates", "median [q25, q75]", "min (date)", "max (date)", "dates below / above / at 0", "share of dates with the same sign", "sign test, p"],
        rows,
    ))  # fmt: skip
    md.append("")
    md.append(
        f'n = {len(same_dates)} for every row; every ± is a standard error across dates. "Share of dates with the same sign": the share on the more frequent side of zero. Sign test: exact two-sided binomial test of as many dates above as below zero; it treats the dates as independent, '
        "and they are not: every date uses the same seeds (2.3). "
        f"κ: its two extremes are dates on which the names' second moment Σw E_LC[R_i²] differs between the two passes (development → production: {'; '.join(kappa_notes)}); the frozen package's reader note 8 says that this moment is not pinned down in the call wing. "
        "† status `check` in the production row; ‡ flagged date. Per date: `tables/budget_by_date.csv`; all statistics and the other samples: `tables/budget_summary.csv`."
    )
    md.append("")

    # ---- 2.2 z
    md.append("### 2.2 The same differences in units of the development row's Monte Carlo error")
    md.append("")
    rows = []
    for key in BUDGET_KEYS:
        if not Q[key].has_se:
            rows.append([Q[key].label, "no Monte Carlo error in the row", *[""] * 6])
            continue
        rows.append([
            Q[key].label, s2.cell(key, "same", "z_mean", digits=2), s2.cell(key, "same", "z_sd", signed=False, digits=2),
            f"{s2.cell(key, 'same', 'z_median', digits=2)} [{s2.cell(key, 'same', 'z_q25', digits=2)}, {s2.cell(key, 'same', 'z_q75', digits=2)}]",
            s2.cell(key, "same", "z_min", digits=2, with_date=True), s2.cell(key, "same", "z_max", digits=2, with_date=True),
            f"{100 * s2.value(key, 'same', 'z_share_beyond_2'):.1f} %", s2.cell(key, "same", "z_sd_if_independent", signed=False, digits=2),
        ])  # fmt: skip
    md.append(md_table(["quantity", "mean ± se across dates", "sd", "median [q25, q75]", "min (date)", "max (date)", "share of dates beyond ±2", "sd if the two rows were independent draws"], rows))  # fmt: skip
    md.append("")
    md.append(
        "(production − development) / se_dev per date, se_dev the pricing Monte Carlo standard error printed for the development row (for a ratio to the copula it contains the copula's own error, which is the same number in the two rows). "
        "Last column: √(mean of 1 + se_prod²/se_dev²), the standard deviation this ratio would have, around a mean of 0, if the two rows were independent draws of the same quantity."
    )
    md.append("")

    # ---- 2.3 legs and the frozen sentence
    md.append("### 2.3 LC/CC: the two legs, the seeds, and the frozen sentence on the four dates")
    md.append("")
    lcc = {k: s2.get("lc_over_cc", "same", k) for k in ("mean", "median", "min", "max")}
    lcc_neg, lcc_pos = int(s2.value("lc_over_cc", "same", "n_neg")), int(s2.value("lc_over_cc", "same", "n_pos"))  # fmt: skip

    def side(key: str) -> str:
        neg, pos = int(s2.value(key, "same", "n_neg")), int(s2.value(key, "same", "n_pos"))
        return f"{'lower' if neg >= pos else 'higher'} at the production budget on {max(neg, pos)} of the {n_leg} dates"

    md.append(
        f"Exactly, per date: Δ(LC/CC) = ΔE_LC[D]/E_CC[D]_dev − (LC/CC)_prod·ΔE_CC[D]/E_CC[D]_dev. Means on the {n_leg} dates: Δ(LC/CC) {s2.cell('lc_over_cc', 'same', 'mean')} = "
        f"leg of E_LC[D] {cell(leg_stats['leg_E_LC'][0], leg_stats['leg_E_LC'][1], 5, True)} (below zero on {leg_stats['leg_E_LC'][2]} dates, above on {leg_stats['leg_E_LC'][3]}) "
        f"+ leg of E_CC[D] {cell(leg_stats['leg_E_CC'][0], leg_stats['leg_E_CC'][1], 5, True)} (below zero on {leg_stats['leg_E_CC'][2]} dates, above on {leg_stats['leg_E_CC'][3]}). "
        f"Of the mean difference, {pct(leg_share['leg_E_CC'], 1)} is the leg of E_CC[D] and {pct(leg_share['leg_E_LC'], 1)} the leg of E_LC[D]; the leg of E_LC[D] is the larger of the two in size on {n_lc_leg_larger} of the {n_leg} dates. "
        f"E_CC[D] is {side('ED_cc')}; the companion's fitted correlation ρ_CC is {side('rho_cc')} (mean {s2.cell('rho_cc', 'same', 'mean')}); "
        f"across the dates the correlation of Δρ_CC with Δ ln E_CC[D] is {corr_rho:+.2f} and with Δ(LC/CC) {corr_diff:+.2f}. "
        f"Mean Δ ln E_CC[D] is {cell(leg_stats['d_ln_E_CC'][0], leg_stats['d_ln_E_CC'][1], 5, True)}; the first-order value −½·Δρ_CC/(1 − ρ_CC) of an equicorrelated basket (an approximation) is {cell(leg_stats['equicorrelation_estimate_d_ln_E_CC'][0], leg_stats['equicorrelation_estimate_d_ln_E_CC'][1], 5, True)}. "
        "Per date: `tables/budget_lc_over_cc_legs.csv`."
    )
    md.append("")
    md.append(
        "The seeds. The companion's ρ_CC is fitted on its own paths with the particle seed, the same on every date, and its fitting error is in no printed ±. "
        "A fitting error of the development companion that is the same draw on every date gives a difference of one sign on every date, as a bias of the smaller budget would: these tables do not separate the two, and they do not say what another seed would give. "
        'The sign test above rejects "independent noise date by date"; it does not show that the smaller budget is biased.'
    )
    md.append("")
    fb, fd, zd, zp = fwd_bound["budget"], fwd_bound["decisions"], fwd_z["development"], fwd_z["production"]  # fmt: skip
    md.append(
        "The basket forward error is pricing noise alone (its expectation is zero) and gives two checks. "
        f"The bound of a difference: production minus development of the forward error, over √(se_dev² + se_prod²), has a root mean square of {fb['rms']:.2f} across the {fb['n']} dates; its largest size is {fb['max']:.3f} ({s2.dated(fb['date'])}) and it is above 1 on {fb['n_above_1']} of them. "
        f"New minus old, over √(se_old² + se_new²), on the {fd['n']} common dates with a dropped slice: root mean square {fd['rms']:.2f}, largest size {fd['max']:.3f} ({s1.dated(fd['date'])}), above 1 on {fd['n_above_1']}. "
        "Two independent runs would give a root mean square of 1: part of the pricing noise is common to the two runs, and the bound overstates the error of a difference. "
        f"The dates: the forward error over its own standard error has a mean of {zd['mean']:+.2f} and a standard deviation of {zd['sd']:.2f} across the {n_leg} dates in the development pass (below zero on {zd['n_neg']}), and {zp['mean']:+.2f} and {zp['sd']:.2f} in the production pass (below zero on {zp['n_neg']}); "
        f"draws independent from date to date would give a mean within ±{2.0 / math.sqrt(n_leg):.2f} of zero (two standard errors) and a standard deviation of 1. The pricing noise is common to the dates: the dates are not the independent draws the sign test takes them to be."
    )
    md.append("")
    fdv, fpr = fit["development"], fit["production"]
    md.append(
        "Size of the companion's fitting error: an order of magnitude from the rows, not a measurement (no pass was run with another seed). "
        f"A row holds the pricing Monte Carlo error of the basket straddle E|R̄| under CC (`straddle_B_cc_se`): {100 * fdv['pricing']:.3f} % of the straddle on the {short(one_value(dev, 'n_paths'))} pricing paths of a development row and {100 * fpr['pricing']:.3f} % on the {short(one_value(new, 'n_paths'))} of a production row (means over the {n_leg} dates). "
        f"Scaled by √(pricing paths / companion paths) to the paths of the companion's fit, that is {100 * fdv['companion']:.2f} % at the development budget ({short(one_value(dev, 'companion_paths'))} paths) and {100 * fpr['companion']:.2f} % at the production budget ({short(one_value(new, 'companion_paths'))} paths). "
        f"If the straddle that the fit reprices has that relative error r and is proportional to √ρ_CC, the error on ρ_CC is 2·ρ_CC·r: {fdv['rho']:.4f} and {fpr['rho']:.4f} (means over the dates). "
        f"Times the size of the ratio of this section's mean leg of E_CC[D] to its mean Δρ_CC ({abs(cc_slope):.2f}), that is {fdv['lc_over_cc']:.4f} and {fpr['lc_over_cc']:.4f} on LC/CC. "
        f"The observed mean Δρ_CC is {s2.cell('rho_cc', 'same', 'mean')}. The fitting error of the production companion is in no printed ± either."
    )
    md.append("")
    ref_cells = "; ".join(f"{d} {cell(ref.loc[d, 'lc_over_cc_difference'], ref.loc[d, 'lc_over_cc_bound'], 5, True)}" for d in pc.REFERENCE_DATES)  # fmt: skip
    md.append(
        f'The frozen package (reader note 1) states: "{FROZEN_BUDGET}". From the two tables, production minus development on those four dates (± the upper bound): {ref_cells}; '
        f"the four have the same specification in the two passes: {'yes' if bool(ref['same_specification'].all()) else 'NO'}. "
        + (
            "The sentence is reproduced: the sizes round to 0.0005 and 0.0019 at the two ends and production is lower on all four. "
            if frozen_range_ok and frozen_sign_ok
            else "The sentence is NOT reproduced from these two tables. "
        )
        + f"On the full set of {n_leg} dates with the same specification the difference runs from {cell(lcc['max']['value'], None, 5, True)} ({s2.dated(lcc['max']['date'])}) to {cell(lcc['min']['value'], None, 5, True)} ({s2.dated(lcc['min']['date'])}), "
        f"mean {s2.cell('lc_over_cc', 'same', 'mean')}, median {s2.cell('lc_over_cc', 'same', 'median')}, quartiles {s2.cell('lc_over_cc', 'same', 'q25')} and {s2.cell('lc_over_cc', 'same', 'q75')}; production is lower on {lcc_neg} dates and higher on {lcc_pos} (sign test p = {s2.value('lc_over_cc', 'same', 'sign_test_p'):.1e}). "
        f"The size of the difference (not rounded) is above {FROZEN_BUDGET_RANGE[1]}, the upper end of the frozen sentence, on {len(above_dates)} of the {n_leg} dates ({pct(len(above_dates) / n_leg, 1)}) and below {FROZEN_BUDGET_RANGE[0]}, its lower end, on {len(below_dates)} ({', '.join(f'{s2.dated(d)} {size[d]:.6f}' for d in below_dates) or 'none'}). "
        + (
            f"The largest is on a date with status `check` in the production row ({new.loc[lcc['min']['date'], 'failing']}). "
            if new.loc[lcc["min"]["date"], "status_rule"] == "check"
            else ""
        )
        + f"On the {s2.n('same_ok')} dates with status `ok` in the production row the mean is {s2.cell('lc_over_cc', 'same_ok', 'mean')}, the difference is below zero on {int(s2.value('lc_over_cc', 'same_ok', 'n_neg'))} of them and runs from {s2.cell('lc_over_cc', 'same_ok', 'max', with_date=True)} to {s2.cell('lc_over_cc', 'same_ok', 'min', with_date=True)}."
    )
    md.append("")

    # ---- 2.4 other samples
    md.append("### 2.4 Other samples: production − development, mean ± se across dates (median)")
    md.append("")
    cols2 = [("same", "same specification"), ("same_unflagged", "same specification, without the flagged dates"), ("same_ok", "same specification, status `ok` in production"), ("other", "another specification (a DJX slice dropped in production)"), ("all", "all dates priced in both")]  # fmt: skip
    rows = [[Q[key].label, *(f"{s2.cell(key, s, 'mean')} ({s2.cell(key, s, 'median')})" for s, _ in cols2)] for key in BUDGET_KEYS]  # fmt: skip
    md.append(md_table(["quantity", *(f"{label} (n = {s2.n(s)})" for s, label in cols2)], rows))
    md.append("")
    n_oth = s2.n("other")

    def levels(sample: str) -> str:
        return ", ".join(f"{Q[k].label} {s2.cell(k, sample, 'mean_a', signed=False)} → {s2.cell(k, sample, 'mean_b', signed=False)}" for k in ("lc_over_cc", "lc_over_copula", "cc_over_copula"))  # fmt: skip

    md.append(
        f"In the column of the {n_oth} dates with another specification the budget and the specification both differ: the development pass has decisions 1 and 2 and the production pass decisions 1, 2 and 5. The column is decision 5 together with the budget; it is not a budget effect. "
        f"There LC/CC moves by {s2.cell('lc_over_cc', 'other', 'min', with_date=True)} to {s2.cell('lc_over_cc', 'other', 'max', with_date=True)}, lower in production on {int(s2.value('lc_over_cc', 'other', 'n_neg'))} dates and higher on {int(s2.value('lc_over_cc', 'other', 'n_pos'))}, "
        f"with a standard deviation of {other_sd:.4f} across the dates; LC/copula moves by {s2.cell('lc_over_copula', 'other', 'min', with_date=True)} to {s2.cell('lc_over_copula', 'other', 'max', with_date=True)}. "
        f"The mean of LC/CC on these {n_oth} dates minus the mean on the {n_leg} dates with the same specification is {cell(net5, net5_se, 5, True)} (± = √ of the sum of the squares of the two standard errors across dates, the two samples of dates treated as independent). "
        f"It is the effect of decision 5 only if the budget effect on these {n_oth} dates is that of the {n_leg}. "
        f"Levels, development → production, mean ± se across dates: on the {s2.n('all')} dates priced in both passes, {levels('all')}; on the {s2.n('all_unflagged')} of them that are not flagged, {levels('all_unflagged')}."
    )
    md.append("")

    # ---- reading
    med_over_mean = [s1.value(k, "all", "median") / s1.value(k, "all", "mean") for k in ("ED_lc", "ED_cc", "lc_over_copula", "cc_over_copula", "lc_over_cc")]  # fmt: skip
    tops = ", ".join(str(k) for k in TOP_COUNTS[:-1]) + f" and {TOP_COUNTS[-1]}"
    sample_words = f"the {len(common)} common dates of the stopped pass"

    def shares(key: str) -> str:
        return ", ".join(pct(conc.loc[key, f"share_top_{k}"]) for k in TOP_COUNTS)

    md.append("## Reading")
    md.append("")
    md.append(
        f"1. *Decisions 1 and 5 together, {sample_words}, new minus old.* LC/CC {s1.cell('lc_over_cc', 'all', 'mean')} (median {s1.cell('lc_over_cc', 'all', 'median')}; lower on {int(s1.value('lc_over_cc', 'all', 'n_neg'))} dates, higher on {int(s1.value('lc_over_cc', 'all', 'n_pos'))}, identical on {int(s1.value('lc_over_cc', 'all', 'n_zero'))}), "
        f"LC/copula {s1.cell('lc_over_copula', 'all', 'mean')} (median {s1.cell('lc_over_copula', 'all', 'median')}), CC/copula {s1.cell('cc_over_copula', 'all', 'mean')} (median {s1.cell('cc_over_copula', 'all', 'median')}). "
        f"These are statistics of that sample and not of the history (1.0): {of(int(m1['n_common']), len(common), 'dates')} are in {p1}, against {of(int(m1['n_priced']), n_priced_new, 'priced dates of the production table')}, "
        f"and LC/CC moves by {s1.cell('lc_over_cc', 'period_1', 'mean')} in {p1} (n = {s1.n('period_1')}) and by {s1.cell('lc_over_cc', 'period_2', 'mean')} in {p2} (n = {s1.n('period_2')}). "
        f"Two reweightings of the {len(common)} differences of LC/CC to the {n_priced_new} priced dates, {NO_ERROR}: {cell(rew_group, None, 5, True)} by the four groups of what the repairs drop and {cell(rew_period, None, 5, True)} by the two periods. "
        f"For E_LC[D], E_CC[D] and the three ratios the median is {100 * min(med_over_mean):.0f} % to {100 * max(med_over_mean):.0f} % of the mean. How much of the sum of the {len(common)} differences the dates with the largest absolute moves supply differs by quantity (1.3): "
        f"the {tops} largest supply {shares('lc_over_cc')} of the sum for LC/CC, {shares('lc_over_copula')} for LC/copula, {shares('cc_over_copula')} for CC/copula, {shares('ED_lc')} for E_LC[D] and {shares('ED_cc')} for E_CC[D]. "
        f"For LC/CC, {int(c_lc['n_positive_in_top'])} of the {N_EXTREMES} largest moves are positive; without the {N_EXTREMES} dates of the table of extremes its mean is {cell(c_lc['mean_without_top'], None, 5, True)} (n = {int(c_lc['n_without_top'])}), "
        f"and its {100 * TRIM:.0f} % trimmed mean is {cell(c_lc['trimmed_mean'], None, 5, True)} (n = {int(c_lc['n_trimmed'])}); {NO_ERROR} for these two."
    )
    md.append(
        f"2. *By what changed.* Decision 1 alone: LC/CC {s1.cell('lc_over_cc', 'd1_only_1_9', 'mean')} (median {s1.cell('lc_over_cc', 'd1_only_1_9', 'median')}) on the {s1.n('d1_only_1_9')} dates with 1–9 names' slices dropped and {s1.cell('lc_over_cc', 'd1_only_10plus', 'mean')} (median {s1.cell('lc_over_cc', 'd1_only_10plus', 'median')}) on the {s1.n('d1_only_10plus')} dates with 10 or more, "
        f"lower on {int(s1.value('lc_over_cc', 'd1_only_10plus', 'n_neg'))} of these {s1.n('d1_only_10plus')}. Where a DJX slice is also dropped ({s1.n('djx_yes')} dates) LC/CC moves by {s1.cell('lc_over_cc', 'djx_yes', 'min', with_date=True)} to {s1.cell('lc_over_cc', 'djx_yes', 'max', with_date=True)}, "
        f"lower on {int(s1.value('lc_over_cc', 'djx_yes', 'n_neg'))} dates and higher on {int(s1.value('lc_over_cc', 'djx_yes', 'n_pos'))}; {sum(1 for d in dd.index[~alone] if d in check_new)} of these {s1.n('djx_yes')} dates have status `check` at the new defaults. "
        f"Decision 2 turns {len(only_new)} FAILED dates of the {n81} into priced, flagged dates (1.4)"
        + (
            f"; on {len(out_rows)} of them the names' second moment is more than 2 % from the listed strips ({100 * worst['names_second_moment_gap']:+.1f} % on {worst['date']})."
            if out_rows and worst is not None
            else "."
        )
    )
    md.append(
        f"3. *Budget, {n_leg} dates with the same specification.* LC/CC is lower at the production budget on {lcc_neg} dates and higher on {lcc_pos}: {s2.cell('lc_over_cc', 'same', 'mean')} on average, {s2.cell('lc_over_cc', 'same', 'z_mean', digits=1)} development standard errors "
        f"(independent draws would give a mean of 0 and a standard deviation of {s2.cell('lc_over_cc', 'same', 'z_sd_if_independent', signed=False, digits=2)}). "
        f"Of the mean difference, {pct(leg_share['leg_E_CC'], 1)} comes through E_CC[D] (leg {cell(leg_stats['leg_E_CC'][0], leg_stats['leg_E_CC'][1], 5, True)}, below zero on {leg_stats['leg_E_CC'][2]} of the {n_leg} dates; CC/copula {s2.cell('cc_over_copula', 'same', 'mean')}; ρ_CC {s2.cell('rho_cc', 'same', 'mean')}) "
        f"and {pct(leg_share['leg_E_LC'], 1)} through E_LC[D] (leg {cell(leg_stats['leg_E_LC'][0], leg_stats['leg_E_LC'][1], 5, True)}, below zero on {leg_stats['leg_E_LC'][2]} of the {n_leg}; LC/copula {s2.cell('lc_over_copula', 'same', 'mean')}, {s2.cell('lc_over_copula', 'same', 'z_mean', digits=2)} development standard errors). "
        "Every date uses the same seeds, so one sign on every date is also what a single draw of the development companion's fitting error would give (2.3)."
    )
    md.append(
        f"4. *Sizes side by side, for LC/CC; the two comparisons are on different samples of dates.* Printed pricing error of a production row: {float(new_m.loc[common, 'lc_over_cc_se'].median()):.5f} (median over the {len(common)} common dates). "
        f"Budget, same specification, {n_leg} dates: {s2.cell('lc_over_cc', 'same', 'mean')} (from {cell(lcc['max']['value'], None, 4, True)} to {cell(lcc['min']['value'], None, 4, True)}; above {FROZEN_BUDGET_RANGE[1]} in size on {len(above_dates)} dates). "
        f"Decisions 1 and 5, {sample_words}, which are not a cross-section of the history (1.0): median {s1.cell('lc_over_cc', 'all', 'median')}, quartiles {s1.cell('lc_over_cc', 'all', 'q25')} and {s1.cell('lc_over_cc', 'all', 'q75')}, from {s1.cell('lc_over_cc', 'all', 'min', digits=4)} to {s1.cell('lc_over_cc', 'all', 'max', digits=4)}; "
        f"mean {s1.cell('lc_over_cc', 'all', 'mean')}, and {cell(rew_group, None, 5, True)} and {cell(rew_period, None, 5, True)} for its two reweightings to the {n_priced_new} priced dates ({NO_ERROR}). "
        "The frozen section C is at the development budget and says so; a development row and a production row of the same specification differ by the budget amounts above on average."
    )
    md.append("")

    # ---- errata
    md.append("## Candidates for ERRATA.md")
    md.append("")
    today = pc.REFERENCE_DATES[0]
    notes_text, findings_text = READER_NOTES.read_text(), FINDINGS.read_text()
    tested: list[tuple[bool, str]] = []
    tested.append((
        b4["agree"] == b4["compared"] and not b4["dates_missing"],
        f"table B4 (1.5): {b4['agree']} of {b4['compared']} numbers agree (`tables/decisions_b4_check.csv`)",
    ))  # fmt: skip
    tested.append((
        frozen_range_ok and frozen_sign_ok,
        f'reader note 1, "{quoted(READER_NOTES, FROZEN_BUDGET)}" (2.3): {ref_cells}',
    ))  # fmt: skip
    tested.append((
        ir["same_dates_and_counts"] and (ir["n_frozen"], ir["n_built"]) == FROZEN_V2_COUNTS and ir["n_here"] == ir["n_frozen"],
        f'V2, "{quoted(VALIDATION, FROZEN_V2)}", and reader note 3, "{quoted(READER_NOTES, FROZEN_NOTE3)}": the frozen `diagnostics/index_repair_3m.json` has a dropped DJX slice on {ir["n_frozen"]} of its {ir["n_built"]} built dates; '
        f"the production table has one on {ir['n_here']} dates, {'the same dates with the same numbers of slices' if ir['same_dates_and_counts'] else 'NOT the same dates or numbers of slices'}",
    ))  # fmt: skip
    today_cop = {name: cell(m.loc[today, "lc_over_copula"], m.loc[today, "lc_over_copula_se"], 5) for name, m in (("new", new_m), ("old", old_m), ("dev", dev_m))}  # fmt: skip
    today_dev_lcc = cell(dev_m.loc[today, "lc_over_cc"], dev_m.loc[today, "lc_over_cc_se"], 4)
    printed_ok = all(text.split(" ± ")[0] in notes_text for text in today_cop.values()) and today_dev_lcc in findings_text  # fmt: skip
    tested.append((
        fr["agree"] == fr["compared"] and fr["missing"] == 0 and printed_ok,
        f"the frozen records of the rows of the four reference dates (E_LC[D], E_CC[D], LC/CC, LC/copula and CC/copula with their errors: sections A and B against the production table, and the three rows of {today} of C.6 against the three tables): "
        f"{fr['agree']} of {fr['compared']} numbers agree (largest relative gap {fr['worst']:.1e}; `tables/frozen_rows_check.csv`). LC/copula of {today} is {today_cop['new']} in the production table, {today_cop['old']} in the old-defaults table and {today_cop['dev']} in the development table, "
        f"the three values of reader note 2 that come from these tables, and the development LC/CC of finding 1 is {today_dev_lcc}",
    ))  # fmt: skip
    d1_dates_ok = all(d in common for d, _ in FROZEN_D1_MOVES)
    d1_moves = {d: float(lc1[d]) for d, _ in FROZEN_D1_MOVES} if d1_dates_ok else {}
    d1_mult = {d: abs(d1_moves[d]) / float(new_m.loc[d, "lc_over_cc_se"]) for d, _ in FROZEN_D1_MOVES[:2]} if d1_dates_ok else {}  # fmt: skip
    tested.append((
        d1_dates_ok and all(round(d1_moves[d], 4) == f for d, f in FROZEN_D1_MOVES) and all(round(m, -1) == FROZEN_D1_MULTIPLE for m in d1_mult.values()),
        f'finding 2 and reader note 1, "{quoted(FINDINGS, FROZEN_D1)}": '
        + (
            f"{'; '.join(f'{cell(d1_moves[d], None, 5, True)} on {d}' for d, _ in FROZEN_D1_MOVES)}; on the first two dates that is {' and '.join(f'{m:.1f}' for m in d1_mult.values())} times the printed error of the production row at the new defaults"
            if d1_dates_ok
            else "the dates are not all priced in both production tables"
        ),
    ))  # fmt: skip
    clip_date, clip_from, clip_to = FROZEN_CLIP_VALUES
    clip_dev, clip_new = float(dev.loc[clip_date, "clip_inner_max"]), float(new.loc[clip_date, "clip_inner_max"])  # fmt: skip
    clip_d3 = frozen_records.get(f"D.{clip_date}.clip_lc")
    tested.append((
        round(clip_dev, 3) == clip_from and round(clip_new, 3) == clip_to,
        f'reader note 3, "{quoted(READER_NOTES, FROZEN_CLIP)}": {clip_dev:.5f} in the development table and {clip_new:.5f} in the production table'
        + (f" (the frozen D.3 has {float(clip_d3['value']):.5f} from its own rerun at {clip_d3['budget']}, commit {clip_d3['commit']}: another run than the production row)" if clip_d3 else ""),
    ))  # fmt: skip
    tested.append((
        rf["equal"] == rf["compared"] and not rf["missing"],
        f"the row files of the four reference dates from which the frozen sections A and B were built ({', '.join(f'`rows/{folder}`' for folder, _ in ROW_FOLDERS)}): {rf['equal']} of {rf['compared']} numbers equal those of the rows of the three tables",
    ))  # fmt: skip
    candidates = [text for ok, text in tested if not ok]
    reproduced = [text for ok, text in tested if ok]
    if candidates:
        md.append(f"{len(candidates)} of the {len(tested)} frozen statements tested from the three tables are NOT reproduced:")  # fmt: skip
        md.append("")
        md += [f"- NOT reproduced: {c}." for c in candidates]
        md.append("")
        md.append(f"Reproduced ({len(reproduced)}): " + "; ".join(f"({i}) {t}" for i, t in enumerate(reproduced, 1)) + ".")  # fmt: skip
    else:
        md.append(
            f"None. {len(tested)} frozen statements are tested from the three tables and each is reproduced; the frozen package's other statements are not tested on this page. "
            + " ".join(f"({i}) {t[0].upper()}{t[1:]}." for i, t in enumerate(reproduced, 1))
        )
    md.append("")
    md.append(
        f"Not errata, for information: (a) reader note 1's sentence is about four dates; on the {n_leg} dates with the same specification the size of the LC/CC difference runs from {abs(lcc['max']['value']):.4f} to {abs(lcc['min']['value']):.4f}, production is lower on {lcc_neg} of them, "
        f"and the size is above {FROZEN_BUDGET_RANGE[1]} on {len(above_dates)} dates and below {FROZEN_BUDGET_RANGE[0]} on {len(below_dates)}: for the budget effect over the history the figures of the {n_leg} dates (2.1, 2.3) are the ones to use, not the range of the four dates; "
        "(b) all four reference dates of B4 are in both production tables."
        if not b4["dates_missing"]
        else f"Not errata, for information: reader note 1's sentence is about four dates; on the {n_leg} dates with the same specification the size of the LC/CC difference runs from {abs(lcc['max']['value']):.4f} to {abs(lcc['min']['value']):.4f} and production is lower on {lcc_neg} of them."
    )
    md.append("")
    pinned = (NEW, OLD, DEV, ENTRIES, B4, NUMBERS, INDEX_REPAIR)
    md.append(
        f"Built by `scripts/pm_later_sensitivity.py` into `{rel(out_dir)}`; inputs read-only; {len(records)} records. "
        f"Inputs, SHA-256 (first 12 hex digits): {'; '.join(f'`{p.relative_to(pc.ROOT) if p.is_relative_to(pc.ROOT) else rel(p)}` {sha12(p)}' for p in pinned)}."
    )
    return "\n".join(md), records, tables


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", type=Path, default=OUT, help="output folder, under pm_update/later")
    args = ap.parse_args()
    out_dir = args.out.resolve()
    if LATER.resolve() not in out_dir.parents:
        raise SystemExit(f"--out must be a folder under {LATER}")
    markdown, records, tables = build(out_dir)
    for name, frame in tables.items():
        pc.save_table(frame, name, base=out_dir)
    pc.write_part(PART, records, markdown, base=out_dir)
    write_text(out_dir / "SENSITIVITY.md", markdown.rstrip() + "\n")
    write_text(out_dir / "sensitivity.json", json.dumps(records, indent=1))
    print(
        f"{out_dir}: SENSITIVITY.md, sensitivity.json ({len(records)} records), {len(tables)} tables"
    )


if __name__ == "__main__":
    main()
