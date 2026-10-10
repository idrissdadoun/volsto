# ruff: noqa: E501, RUF001
"""After the freeze of the PM results package of 2026-10-09: a note on the finished one-year
pass of the local correlation model on the monthly dates (development budget), compared with
the frozen one-year addendum.  Nothing is priced here and no frozen file is written: the output
goes to ``outputs/dispersion_lc/pm_update/later/12m_monthly`` (``NOTE_12M_MONTHLY.md``,
``tables/*.csv``, ``parts/12m_monthly.{json,md}``).

    .venv/bin/python scripts/pm_later_12m_monthly.py [--spec-build]

What it computes.

1. *The state of the pass*: rows, statuses, failed dates with the step that raised (from the
   run's log of the date), which gating check FAILS on each ``check`` date and which column is
   not finite, what the owner's decisions 1, 2 and 5 do on how many dates, where the row outside
   the monthly list comes from.
2. *The forward at one year*: E_LC[D]/E_CC[D], E_LC[D]/copula, E_CC[D]/copula and the
   listed-variance forward √(EQV/EV) (model S has no one-year table), each summarised on eight
   samples — (a) the dates without decision 5's clip flag (the owner's rule), (a2) the same
   without decision 2's flagged date, (a-ok) and (a2-ok) the same two on their dates with status
   ``ok``, (b) all priced dates, (c) all priced dates without decision 2's flagged dates,
   (d) status ``ok`` only, (d2) the same without decision 2's flagged dates.
3. *The clipped mass*: the two one-sided masses; the unflagged dates with their checks, what
   fails, the index gate's two numbers and the DJX slices behind the one-year index target (the
   slices the screen and the repair dropped, the kept slices nearest the horizon, whether the
   target is extrapolated or interpolated across more than ``BRACKET_YEAR``: from the
   specification-only build; and the DJX expiries the inputs list on those dates, read with
   the pricer's own loader, each kept, dropped by the screen, dropped by the repair or left
   out by the surface's selection: :func:`index_expiry_account`); the ratios by tercile (cut on each sample's own dates) and by bin
   of the larger mass with the years of each bin; a least-squares fit on the two masses over all
   dates and fits on the larger mass over all dates, at or below ``CLIP_RESTRICT`` (also among
   the flagged dates alone, and with a dummy for each year) and above ``CLIP_TURN`` (with a rank
   correlation); the rank of the unflagged dates and their mean beside the flagged dates of the
   same year.
4. *The extremes*: the gap between the model's index target (read at the forward) and the
   study's index level (read at the spot), what it is made of (its correlation with the study's
   forward, the same gap read at the spot strike, the effect of the calendar repair on the
   target) and E_CC[D]/copula against it; the date with a clipped mass of 1; the date with κ
   near zero; the diagnostic "SVI strips / listed strips − 1" (which dates, why, and a check in
   the pricer's source, at the rows' commit, that no other column reads it).
5. *The calls* at the study's strikes: what can be quoted and what cannot (reader note 7).
6. *Against the frozen one-year addendum* (``pm_update/1y``): its per-date records recomputed
   from the rows with the addendum's own quantity definitions (``pm_1y.ALL``), its summaries
   beside the monthly ones (sample (c) also against the frozen per-date records without the
   dates decision 2 flags: its dates out on both sides), its production rows beside the monthly development rows of the same
   dates with the difference of LC/CC split into the part from E_LC[D] and the part from
   E_CC[D], and its sentences that read differently on the monthly dates (each quoted sentence
   is looked up word for word in the frozen page before it is printed).
7. *Two years*: the four rows of ``lcm_24m_dev.parquet``, every number labelled indicative.
8. *What can be said at one year*: at most six sentences, each built from records by a rule,
   and the comparison with three months sample by sample (every common date, without decision
   2's flagged dates, the dates decision 5's condition flags at neither horizon), against the
   three-month development pass of the frozen history and against the production pass run
   after the freeze, with the verdict at every number of lags.

Reads (read-only): ``outputs/dispersion_lc/lcm_12m_dev.parquet`` with its row files
``rows/12m_development/<date>.json`` (the list ``index_errors`` is only there) and logs
``logs/12m_development``, ``lcm_12m.parquet`` (production, four dates), ``lcm_24m_dev.parquet``
with ``rows/24m_development``, ``lcm_3m_dev_repair.parquet`` and ``lcm_3m.parquet`` (the 3m
passes), ``lcm_12m_dev_norepair.parquet`` with ``rows/12m_development_norepair`` (the one
old-defaults row), the driver's log ``logs/driver.log``, the automatic reports
``report_12m_dev.md`` and ``report_24m_dev.md``; of the study
(``outputs/dispersion``, a read-only link) ``entries_12m.parquet``, ``entries_24m.parquet``,
``legs_12m.parquet``, ``model_s_3m.parquet``, the entry pickles' names and ``PROGRESS_Q2.md``,
and on the unflagged dates the inputs as ``lcm_diagnostics.load_inputs`` reads them (the
listed DJX expiries: :func:`listed_index_expiries`, when the specification-only tables are
there); of the frozen package ``1y/NUMBERS_1Y.md``, ``1y/numbers_1y.json``, ``parts/C_history.json``,
``parts/R_reader_notes.md``, ``parts/V_validation.json``; and ``git show
<rows' commit>:scripts/lcm_price.py`` (read-only).

``--spec-build`` adds a *specification-only* build of every date of the table under the
worktree's ``configs/studies/dispersion/lcm.yaml`` (``lcm_price.spec_for``: no calibration, no
simulation; the SVI fit records of the runs are read through a store that never writes): the
kept and dropped DJX slices around the horizon, the names behind the counts, per name the SVI
strip second moment on the dates where the diagnostic is far from the listed strips, on the
failed dates the name whose slice fit raised, and on every priced date the index target itself
(:func:`index_target`: the SVI slices of the specification's index surface, compared with the
row's ``target_vol`` at every strike of the horizon) read at the spot strike of the study's
forward and rebuilt with ``screen.calendar_repair`` off (the target with and without the
repair; compared on 2026-10-02 with the old-defaults row), and the forward and at-the-money
vol of each name kept unscreened.  It writes ``tables/spec_*.csv``; a run without the option
uses those tables when they are there and says so when they are not.

What each error is.  *Per date*: the pricing Monte Carlo standard error of the row given its
calibrated model (the row's own for LC/CC, paired on common paths; the delta method of
``pm_common.ratio_se`` with the copula's own error for a ratio to the copula).  *A mean across
dates*: sd/√n with the dates treated as independent, printed when n ≥ ``MIN_N_SE``; beside it
the Newey–West error of the mean (Bartlett kernel, ``NW_LAGS`` lags counted in consecutive dates
of the sample: ``pm_history.newey_west_se``), printed when n ≥ ``MIN_N_NW``.  Order statistics,
counts and shares carry none.  A regression coefficient: an HC1 error and, when the fit has
``MIN_N_NW`` dates or more, a Newey–West one.

Rules behind the sentences (``Rules``): a mean is said to be below or above a level when it is
more than ``K_SIGMA`` times the larger of its two errors away from it, and "not distinguishable"
otherwise; the number of errors is printed, and the sentence says so when the verdict is not
the same at the other lags of ``NW_OTHER``; with fewer than ``MIN_N_SE`` dates nothing is said
about a sign.  A frozen mean "reads differently" when the monthly mean is more than ``K_SIGMA``
times √(se_f² + se_m²) away; a mean is "carried by one date" when that date accounts for more
than ``ONE_DATE_SHARE`` of the sum of the squared deviations from the sample's median, and
"not resolved" when it is also within ``K_SIGMA`` of its errors of zero.  Sample (c) is also
set against the frozen dates without those decision 2 flags (mean and sd/√n of the frozen
per-date records by ``pm_1y.summarise``, which carries an error from two dates on: the page
says so when they are fewer than ``MIN_N_SE``), and a row whose verdict changes between the two
frozen sides "sits at the threshold" when each |z| is within ``Z_AT_THRESHOLD`` of
``K_SIGMA``.  The source of a
difference between two budgets is the larger, in absolute value, of its part from E_LC[D] and
its part from E_CC[D].  Means are said to fall, rise or be unordered from one group to the next
by ``Rules.ordering``; the direction a fit gives is the sign of its slope, stated when the
slope is distinguishable from zero.

How it is checked (every check raises or is printed in the page's last section): the table
against its row files, column by column; the rows' copula columns against the study's entries;
the flags and the status recomputed from their definitions; the means of sample (b) against
the automatic report's printed ones; the least-squares routine against ``pm_history.ols``; the
frozen 3m mean recomputed from its table; the frozen per-date development records recomputed
from the row files, and the frozen summary means from the frozen per-date records; decision
5's condition at three months against the production rows' own flag and reader note 4's count;
the listed DJX expiries of the unflagged dates against the build's kept and dropped slices;
the rebuilt index target against the rows' ``target_vol`` and, with the
repair off, against the old-defaults row; every quoted sentence found word for word in its
file.
"""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import logging
import math
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_1y
import pm_common as pc
import pm_history as ph

LOG = logging.getLogger("pm_later_12m_monthly")

SECTION = "later/12m_monthly"
PART = "12m_monthly"
OUT = pc.PM / "later" / "12m_monthly"
TENOR = "12m"
TABLE = pc.LC_OUT / "lcm_12m_dev.parquet"
ROWS = pc.LC_OUT / "rows" / "12m_development"
LOGS = pc.LC_OUT / "logs" / "12m_development"
REPORT = pc.LC_OUT / "report_12m_dev.md"
DRIVER_LOG = pc.LC_OUT / "logs" / "driver.log"
README_3M = pc.PM / "later" / "production_3m" / "README.md"
TABLE_PROD = pc.LC_OUT / "lcm_12m.parquet"
ROWS_PROD = pc.LC_OUT / "rows" / "12m_production"
TABLE_24 = pc.LC_OUT / "lcm_24m_dev.parquet"
ROWS_24 = pc.LC_OUT / "rows" / "24m_development"
REPORT_24 = pc.LC_OUT / "report_24m_dev.md"
TABLE_NOREPAIR = pc.LC_OUT / "lcm_12m_dev_norepair.parquet"
ROWS_NOREPAIR = pc.LC_OUT / "rows" / "12m_development_norepair"
TABLE_3M_FROZEN = pc.LC_OUT / "lcm_3m_dev_repair.parquet"
TABLE_3M_PROD = pc.LC_OUT / "lcm_3m.parquet"
FROZEN_1Y = pc.PM / "1y"
HIST_FROZEN = pc.PM / "parts" / "C_history.json"
HIST_PROD = pc.PM / "later" / "production_3m" / "history" / "parts" / "C_history.json"
READER_NOTES = pc.PM / "parts" / "R_reader_notes.md"
VALIDATION = pc.PM / "parts" / "V_validation.json"
TODAY = pc.REFERENCE_DATES[0]
CALL_TAGS = pc.MULT_TAGS
GATING = ("check_no_nan", "check_forward", "check_index")
NAN_EXEMPT = pm_1y.NAN_EXEMPT
CLIP_FLAG = 0.01
GATE_VP = 0.15
#: an across-dates standard error is printed from this many dates on
MIN_N_SE = 20
#: a Newey–West error from this many dates on (five times the lags)
MIN_N_NW = 60
NW_LAGS = 12
NW_OTHER = (6, 24)
K_SIGMA = 2.0
#: a difference between two runs of a date is listed above this many times its bound
Z_LIST = 3.0
#: the model's index target is called far from the study's level beyond this (vol points)
GAP_FAR = 1.5
#: a clipped mass is called near 1 from this on
CLIP_NEAR_ONE = 0.5
#: the SVI strips are called far from the listed strips beyond this (ratio − 1)
STRIP_FAR = 1.0
#: a call ratio is called statistically unusable when its median relative Monte Carlo error exceeds this
CALL_REL_SE = 0.10
#: the bins of the larger clipped mass (upper edges; one more bin above the last)
CLIP_BINS = (0.01, 0.02, 0.03, 0.05, 0.10, 0.20)
#: the mass above which the relation between LC/CC and the clipped mass is looked at apart
CLIP_TURN = 0.20
#: the fits of LC/CC on the larger clipped mass are restricted to the dates at or below these
CLIP_RESTRICT = (0.20, 0.10)
#: the calendar repair is said to move the index target beyond these (vol points)
REPAIR_MOVES = (0.1, 0.5)
#: two target smiles are called the same below this largest difference (vol points)
SAME_SMILE_VP = 0.005
#: the one-year index target is called interpolated across more than one year when the two kept
#: DJX slices nearest the horizon are further apart than this (years)
BRACKET_YEAR = 1.0
#: a mean is called carried by one date when that date accounts for more than this share of the
#: sum, over the dates, of the squared deviations from the sample's median
ONE_DATE_SHARE = 0.95
#: a z is said to sit at the threshold when |z| is within this of ``K_SIGMA``
Z_AT_THRESHOLD = 0.25
#: the strikes (at-the-money sd) of the target smile printed for a date
SMILE_STRIKES = ("-2.5", "-1.5", "+0.0", "+1.5", "+2.0", "+2.5")
NAN = float("nan")
BUDGET = pc.BUDGETS["development"]
STUDY = f"{pc.BUDGETS['study']}: outputs/dispersion/entries_12m.parquet, basket B1"
INCL = "flagged rows included, not the rule of decision 5"
SAMPLES = (
    ("a", "(a) unflagged dates — the rule of decision 5"),
    (
        "a2",
        "(a2) unflagged dates without decision 2's flagged date — the rules of decisions 5 and 2",
    ),
    ("a_ok", "(a-ok) unflagged dates with status ok — the rule of decision 5, status ok only"),
    (
        "a2_ok",
        "(a2-ok) unflagged dates with status ok, without decision 2's flagged date — the rules of decisions 5 and 2, status ok only",
    ),
    ("b", f"(b) all priced dates [{INCL}]"),
    ("c", f"(c) all priced dates without decision 2's flagged dates [{INCL}]"),
    ("d", f"(d) status ok only [{INCL}]"),
    ("d2", f"(d2) status ok only, without decision 2's flagged dates [{INCL}]"),
)
#: the samples of the calls' table (section 5)
CALL_SAMPLES = ("a", "a2", "b", "c")
#: the units of the frozen addendum's quantities (``pm_1y``), as the page writes them
UNIT_WORDS = {
    "notional": "fraction of the notional",
    "fraction": "fraction of the particles",
    "squared return": "squared return",
    "ratio": "ratio",
    "vol points": "vol points",
}
FOUR_RATIOS = (
    ("lc_over_cc", "E_LC[D] / E_CC[D]", "the row's `ratio`, paired on common paths (`ratio_se`)"),
    (
        "lc_over_copula",
        "E_LC[D] / copula P_D",
        "`ED_lc` over the entry's `P_D`; error: delta method with `ED_lc_se` and the copula's `P_D_se`, independent",
    ),
    (
        "cc_over_copula",
        "E_CC[D] / copula P_D",
        "`ED_cc` over the entry's `P_D`; error: delta method with `ED_cc_se` and the copula's `P_D_se`, independent",
    ),
    (
        "listed_fwd_ratio",
        "listed-variance forward / copula P_D, √(EQV/EV)",
        "√(EQV/EV) of the study's entry: the study's number, no Monte Carlo error here",
    ),
)
INPUT_COLUMNS = ("n_dropped", "n_dropped_calendar", "n_dropped_calendar_index", "n_dropped_index", "n_names_extrapolated", "n_names_unscreened", "index_last_slice", "n_slices", "sum_w_M_svi", "svi_rms_vp_median", "svi_rms_vp_max")  # fmt: skip
STRIP_COLUMNS = ("sum_w_M_svi", "names_mc_over_svi", "names_mc_z", "names_svi_over_listed")


# ----------------------------------------------------------------------------- small helpers
def finite(x: Any) -> bool:
    return isinstance(x, (int, float, np.floating, np.integer)) and math.isfinite(float(x))


def fmt(x: Any, digits: int = 4, signed: bool = False) -> str:
    if not finite(x):
        return "n/a"
    return f"{float(x):+.{digits}f}" if signed else f"{float(x):.{digits}f}"


def pm(x: Any, se: Any, digits: int = 4, signed: bool = False) -> str:
    return f"{fmt(x, digits, signed)} ± {fmt(se, digits)}" if finite(se) else fmt(x, digits, signed)


def chk(value: Any) -> str:
    """A check as printed: a FAIL stays a FAIL."""
    return "pass" if value is True or value == 1.0 else "FAIL"


def yes(value: Any) -> str:
    return "yes" if bool(value) else "no"


def count(n: int, one: str, many: str) -> str:
    """``n`` with its noun in the right number."""
    return f"{n} {one if n == 1 else many}"


def years(x: Any) -> str:
    return f"{float(x):.2f}y" if finite(x) else "none"


def errs(x: float) -> str:
    """A number of errors: one decimal far from the threshold ``K_SIGMA``, two near it, and as
    many more (up to six) as it takes not to print the threshold itself."""
    if x >= 5:
        return f"{x:.1f}"
    for digits in range(2, 7):
        if f"{x:.{digits}f}" != f"{K_SIGMA:.{digits}f}":
            break
    return f"{x:.{digits}f}"


def z_cell(z: Any) -> str:
    """A z as a table cell: one signed decimal, and as many more (up to six) as it takes not to
    print the threshold ``K_SIGMA`` itself for a z that is not on it."""
    if not finite(z):
        return "n/a"
    for digits in range(1, 7):
        if f"{abs(float(z)):.{digits}f}" != f"{K_SIGMA:.{digits}f}":
            break
    return f"{float(z):+.{digits}f}"


def floor_to(x: float, digits: int) -> float:
    """``x`` rounded down at ``digits`` decimals (a bound printed as "x or more")."""
    scale = 10.0**digits
    return math.floor(float(x) * scale + 1e-9) / scale


def few(k: int, n: int, dates: list[str], limit: int = 3) -> str:
    """``k of n dates``, with the dates when there are at most ``limit`` of them."""
    named = f" ({', '.join(dates)})" if 0 < k <= limit else ""
    return f"{k} of {n} dates{named}"


def ordinal(k: int) -> str:
    tail = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{tail}"


def truth(flag: pd.Series) -> pd.Series:
    return flag.map(lambda x: x is True or x == 1.0).astype(bool)


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(" --- " for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(pc.PM.parents[2].resolve()))
    except ValueError:
        return str(path)


def quoted(path: Path, text: str) -> str:
    """``text`` when it stands word for word in ``path``; raises otherwise."""
    if text not in path.read_text():
        raise ValueError(f"not found word for word in {path}: {text[:80]!r}")
    return text


def dates_text(dates: list[str], limit: int = 40) -> str:
    if not dates:
        return "none"
    return (
        ", ".join(dates)
        if len(dates) <= limit
        else ", ".join(dates[:limit]) + f" … ({len(dates)} dates: the CSV has them all)"
    )


def write_text(path: Path, text: str) -> None:
    pc.guard_frozen(path)
    if OUT.resolve() not in path.resolve().parents:
        raise RuntimeError(f"{path} is outside {OUT}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def git(*args: str) -> str:
    done = subprocess.run(["git", *args], cwd=pc.ROOT, capture_output=True, text=True, check=False)
    return done.stdout if done.returncode == 0 else ""


def ols_fit(y: Any, x: Any, names: list[str], lags: int = NW_LAGS) -> dict[str, Any]:
    """Least squares of ``y`` on an intercept and the columns of ``x`` with HC1 and Newey–West
    (Bartlett, ``lags`` lags, the factor n/(n − k) of HC1) errors: ``pm_history.ols`` with the
    number of lags as an argument (compared with it at its own lags in :func:`self_checks`).
    As for a mean, the Newey–West error is carried from ``MIN_N_NW`` dates on (NaN below)."""
    yv = np.asarray(y, dtype=float).ravel()
    xv = np.asarray(x, dtype=float).reshape(len(yv), -1)
    n = len(yv)
    design = np.column_stack([np.ones(n), xv])
    k = design.shape[1]
    if np.linalg.matrix_rank(design) < k:
        raise ValueError(f"ols_fit: the design of {names} is rank deficient on {n} dates")
    qm, rm = np.linalg.qr(design)
    coef = np.linalg.solve(rm, qm.T @ yv)
    r_inv = np.linalg.inv(rm)
    xtx_inv = r_inv @ r_inv.T
    resid = yv - design @ coef
    xe = design * resid[:, None]
    hc1 = np.sqrt(np.diag(n / (n - k) * xtx_inv @ (xe.T @ xe) @ xtx_inv))
    nw = (
        np.sqrt(np.diag(n / (n - k) * xtx_inv @ ph.bartlett_long_run(xe, lags) @ xtx_inv))
        if n >= MIN_N_NW
        else np.full(k, NAN)
    )
    tss = float(((yv - yv.mean()) ** 2).sum())
    return {"names": ["const", *names], "coef": coef, "se_hc1": hc1, "se_nw": nw, "r2": 1.0 - float(resid @ resid) / tss, "n": n}  # fmt: skip


def fit_errs(f: dict[str, Any], j: int, digits: int = 4, with_n: bool = False) -> str:
    """The two errors of coefficient ``j`` of a fit as printed, with the number of dates when
    asked: the Newey–West one only when the fit carries it (``MIN_N_NW`` dates or more)."""
    hc1 = f"HC1 {fmt(f['se_hc1'][j], digits)}"
    if finite(f["se_nw"][j]):
        return f"{hc1}, Newey–West {fmt(f['se_nw'][j], digits)}" + (f"; n {f['n']}" if with_n else "")  # fmt: skip
    return f"{hc1}; n {f['n']}: no Newey–West error on fewer than {MIN_N_NW} dates"


def nw_note(f: dict[str, Any], j: int) -> str:
    """The Newey–West error of coefficient ``j`` of a fit as a record's note."""
    if finite(f["se_nw"][j]):
        return f"Newey–West ({NW_LAGS} lags) {f['se_nw'][j]:.4f}"
    return f"no Newey–West error on {f['n']} dates (fewer than {MIN_N_NW})"


def year_spans(dates: Any) -> str:
    """The calendar years of ``dates`` as ranges of consecutive years: ``2008, 2022–2026``."""
    spans: list[list[int]] = []
    for y in sorted({int(str(x)[:4]) for x in dates}):
        if spans and y == spans[-1][1] + 1:
            spans[-1][1] = y
        else:
            spans.append([y, y])
    return ", ".join(str(a) if a == b else f"{a}–{b}" for a, b in spans) if spans else "none"


def one_date_share(x: pd.Series) -> tuple[str, float]:
    """The date (the index of ``x``) that accounts for the largest share of the sum of the squared
    deviations from the sample's median, and that share (see ``ONE_DATE_SHARE``)."""
    v = x.dropna().astype(float)
    dev = (v - v.median()) ** 2
    total = float(dev.sum())
    if len(v) == 0 or not total > 0:
        return "", NAN
    return str(dev.idxmax()), float(dev.max() / total)


def listing(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


class Rules:
    """The explicit rules behind every sentence that states a sign, an ordering or "no change"."""

    @staticmethod
    def larger(se: Any, nw: Any) -> float:
        xs = [float(x) for x in (se, nw) if finite(x)]
        return max(xs) if xs else NAN

    @staticmethod
    def against(mean: float, level: float, se: Any, nw: Any, n: int) -> str:
        """Where a mean stands against a level: more than ``K_SIGMA`` times the larger of its
        two errors away, or not; nothing is said with fewer than ``MIN_N_SE`` dates."""
        err = Rules.larger(se, nw)
        if n < MIN_N_SE or not finite(err):
            return f"on {n} dates no standard error is carried and nothing is said about its sign against {level:g}"
        side = "below" if mean < level else "above"
        if abs(mean - level) > K_SIGMA * err:
            return f"{side} {level:g} by more than {K_SIGMA:g} times the larger of its two errors"
        return (
            f"not distinguishable from {level:g} at {K_SIGMA:g} times the larger of its two errors"
        )

    @staticmethod
    def errors(mean: float, level: float, se: Any, nw: Any) -> float:
        """The distance of a mean from a level in units of the larger of its two errors."""
        err = Rules.larger(se, nw)
        return abs(mean - level) / err if finite(err) and err > 0 else NAN

    @staticmethod
    def against_lags(mean: float, level: float, s: dict[str, Any]) -> str:
        """:meth:`against` at ``NW_LAGS`` lags with the number of errors, and what the same rule
        gives at the other lags of ``NW_OTHER`` when the statistics carry them (``s`` is a
        result of :func:`summarise`): the sentence says so when the verdict is not the same."""
        n, se, nw = int(s["n"]), s.get("se"), s.get("nw")
        base = Rules.against(mean, level, se, nw, n)
        here = Rules.errors(mean, level, se, nw)
        if n < MIN_N_SE or not finite(here):
            return base
        if not finite(nw):
            return f"{base} ({errs(here)} errors; no Newey–West error on {n} dates)"
        others = [
            (lags, Rules.errors(mean, level, se, s.get(f"nw_{lags}")))
            for lags in NW_OTHER
            if finite(s.get(f"nw_{lags}"))
        ]
        changed = [(lags, x) for lags, x in others if (x > K_SIGMA) != (here > K_SIGMA)]
        if not others:
            return f"{base} ({errs(here)} errors at {NW_LAGS} lags)"
        if not changed:
            return f"{base} ({errs(here)} errors at {NW_LAGS} lags; the same verdict at {' and '.join(str(lags) for lags, _ in others)} lags)"
        return f"{base} ({errs(here)} errors at {NW_LAGS} lags); " + " and ".join(
            f"at {lags} lags it is {'beyond' if x > K_SIGMA else 'within'} {K_SIGMA:g} errors ({errs(x)})"
            for lags, x in changed
        )

    @staticmethod
    def side(mean: float, level: float, s: dict[str, Any], lags: int = NW_LAGS) -> str:
        """``above``, ``below`` or ``not distinguishable`` by the rule of :meth:`against` with the
        Newey–West error at ``lags`` lags (``s`` is a result of :func:`summarise`); an empty
        string when the sample carries no error."""
        nw = s.get("nw") if lags == NW_LAGS else s.get(f"nw_{lags}")
        x = Rules.errors(mean, level, s.get("se"), nw)
        if int(s["n"]) < MIN_N_SE or not finite(x):
            return ""
        if x > K_SIGMA:
            return "above" if mean > level else "below"
        return "not distinguishable"

    @staticmethod
    def at_each_lag(mean: float, level: float, s: dict[str, Any]) -> str:
        """The verdict of :meth:`against` stated at every number of lags, ``NW_LAGS`` and
        ``NW_OTHER``, with the number of errors at each."""
        n = int(s["n"])
        if n < MIN_N_SE or not finite(s.get("se")):
            return Rules.against(mean, level, s.get("se"), s.get("nw"), n)
        if not finite(s.get("nw")):
            return f"{Rules.against(mean, level, s['se'], None, n)} ({errs(Rules.errors(mean, level, s['se'], None))} errors; no Newey–West error on {n} dates)"
        lags = sorted({NW_LAGS, *NW_OTHER})
        at = [(k, Rules.errors(mean, level, s["se"], s.get("nw") if k == NW_LAGS else s.get(f"nw_{k}"))) for k in lags]  # fmt: skip
        beyond = [(k, x) for k, x in at if x > K_SIGMA]
        within = [(k, x) for k, x in at if not x > K_SIGMA]
        side = "below" if mean < level else "above"

        def lag_list(xs: list[tuple[int, float]]) -> str:
            return f"at {listing([str(k) for k, _ in xs])} lags ({listing([errs(x) for _, x in xs])} errors)"

        far = f"{side} {level:g} by more than {K_SIGMA:g} times the larger of its two errors"
        near = (
            f"not distinguishable from {level:g} at {K_SIGMA:g} times the larger of its two errors"
        )
        if not within:
            return f"{far} {lag_list(beyond)}"
        if not beyond:
            return f"{near} {lag_list(within)}"
        return f"{far} {lag_list(beyond)} and within {K_SIGMA:g} errors {lag_list(within)}"

    @staticmethod
    def ordering(values: list[float], what: str) -> str:
        """Whether a list of means falls, rises or is not ordered from one ``what`` to the next."""
        pairs = list(itertools.pairwise(values))
        if all(y < x for x, y in pairs):
            return f"fall from {what} to {what}"
        if all(y > x for x, y in pairs):
            return f"rise from {what} to {what}"
        return "are not ordered"

    @staticmethod
    def against_small(level: float, *samples: dict[str, Any]) -> str:
        """What is said of samples that are each too small to carry an error (fewer than
        ``MIN_N_SE`` dates); :meth:`against_lags` of each otherwise."""
        if all(int(s["n"]) < MIN_N_SE for s in samples):
            ns = [str(int(s["n"])) for s in samples]
            return f"on {' and on '.join(ns) if len(ns) <= 2 else listing(ns)} dates no standard error is carried and nothing is said about a sign against {level:g}"
        return "; ".join(Rules.against_lags(s["mean"], level, s) for s in samples)

    @staticmethod
    def differs(a: float, a_se: Any, b: float, b_se: Any) -> bool | None:
        if not (finite(a_se) and finite(b_se)):
            return None
        return abs(a - b) > K_SIGMA * math.hypot(float(a_se), float(b_se))

    @staticmethod
    def slope(coef: float, hc1: float, nw: float) -> str:
        """Where a slope stands against zero: more than ``K_SIGMA`` times the larger of its HC1
        and Newey–West errors away, or not; with its HC1 error alone when the fit carries no
        Newey–West one (fewer than ``MIN_N_NW`` dates)."""
        err = Rules.larger(hc1, nw)
        which = "the larger of its two errors" if finite(nw) else f"its HC1 error (no Newey–West error on fewer than {MIN_N_NW} dates)"  # fmt: skip
        if abs(coef) > K_SIGMA * err:
            return f"{'positive' if coef > 0 else 'negative'}, more than {K_SIGMA:g} times {which} from zero"
        return f"not distinguishable from zero at {K_SIGMA:g} times {which}"

    @staticmethod
    def slope_errors(coef: float, hc1: float, nw: float) -> float:
        """The distance of a slope from zero in units of the larger of its errors."""
        err = Rules.larger(hc1, nw)
        return abs(coef) / err if finite(err) and err > 0 else NAN


@dataclass
class Book:
    """The records of the note, by id, and the page's blocks."""

    commit: str
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    blocks: list[str] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    checks: list[str] = field(default_factory=list)

    def add(self, key: str, quantity: str, value: Any, se: Any = None, **kw: Any) -> float:
        rid = f"L12M.{key}"
        if rid in self.records:
            raise ValueError(f"duplicate record {rid}")
        kw.setdefault("tenor", TENOR)
        kw.setdefault("budget", BUDGET)
        kw.setdefault("commit", self.commit)
        kw.setdefault("source", rel(TABLE))
        v = float(value) if finite(value) else None
        self.records[rid] = pc.record(
            rid, SECTION, quantity, v, float(se) if finite(se) else None, **kw
        )
        return NAN if v is None else v

    def get(self, key: str) -> float:
        v = self.records[f"L12M.{key}"]["value"]
        return NAN if v is None else float(v)

    def se(self, key: str) -> float:
        v = self.records[f"L12M.{key}"]["se"]
        return NAN if v is None else float(v)

    def text(self, *lines: str) -> None:
        self.blocks.append("\n\n".join(lines))

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        self.blocks.append(md_table(headers, rows))

    def save(self, name: str, frame: pd.DataFrame) -> str:
        self.tables[name] = frame
        return f"`tables/{name}.csv`"


# ----------------------------------------------------------------------------- the data
@dataclass
class Data:
    d: pd.DataFrame  #: one row per date of the table, in time order, with the derived columns
    rows: dict[str, dict[str, Any]]  #: the row files
    entries: pd.DataFrame  #: the study's entries of basket B1, by date
    monthly: list[str]  #: the pass's list of monthly dates
    yearly: list[str]  #: the frozen addendum's twenty dates
    commit: str

    @property
    def priced(self) -> pd.DataFrame:
        return self.d[self.d["priced"]].reset_index(drop=True)


def monthly_list(tenor: str) -> list[str]:
    """The list of ``scripts/disp_lcm.py::study_dates`` (not imported: it loads the pricer): the
    converged dates of the study's model S table and today, with an entry pickle at the tenor."""
    table = pd.read_parquet(pc.STUDY / "model_s_3m.parquet")
    dates = {str(x)[:10] for x in table.loc[table["converged"], "date"]} | {TODAY}
    have = {p.stem for p in (pc.STUDY / "entries" / tenor).glob("*.pkl")}
    return sorted(dates & have)


def nonfinite_columns(row: dict[str, Any]) -> list[str]:
    """The floats ``check_no_nan`` of ``scripts/lcm_price.py`` reads that are not finite."""
    return [k for k, v in row.items() if isinstance(v, float) and not k.startswith(NAN_EXEMPT) and not math.isfinite(v)]  # fmt: skip


def by_key(quantities: list[Any]) -> dict[str, Any]:
    return {q.key: q for q in quantities}


Q1Y = by_key(pm_1y.ALL)


def value_of(key: str, row: dict[str, Any], entry: dict[str, Any]) -> tuple[float, float]:
    """A quantity of the frozen addendum (``pm_1y.ALL``) on a row and its entry: value and
    standard error (NaN when it has none or the row does not carry the column)."""
    try:
        v, s = Q1Y[key].get(row, entry)
    except KeyError:
        return NAN, NAN
    return (float(v) if finite(v) else NAN), (float(s) if finite(s) else NAN)


def derive(
    table: pd.DataFrame, rows: dict[str, dict[str, Any]], entries: pd.DataFrame, tenor: str = TENOR
) -> pd.DataFrame:
    """The per-date frame: the table's columns and, on the priced dates, the ratios with the
    addendum's definitions, the index level, the kept slices below the horizon and the checks."""
    d = table.sort_values("date").reset_index(drop=True).copy()
    # the table's own ratios to the copula (their error leaves the copula's out) keep another name
    d = d.rename(
        columns={
            f"{k}{s}": f"{k}{s}_table_own"
            for k in ("lc_over_copula", "cc_over_copula")
            for s in ("", "_se")
        }
    )
    d["priced"] = d["status"] != "failed"
    extra: list[dict[str, Any]] = []
    for date, priced in zip(d["date"], d["priced"], strict=True):
        out: dict[str, Any] = {}
        if priced:
            r, e = rows[date], entries.loc[date].to_dict()
            for key in ("lc_over_cc", "lc_over_copula", "cc_over_copula"):
                out[key], out[f"{key}_se"] = value_of(key, r, e)
            out["listed_ratio"] = value_of("listed_fwd_ratio", r, e)[0]
            out["target_atm"], out["study_atm"] = (
                value_of("target_atm_vol", r, e)[0],
                value_of("study_atm_vol", r, e)[0],
            )
            out["atm_gap"] = out["target_atm"] - out["study_atm"]
            out["study_fwd"] = float(e["f_B"])
            horizon = float(r["T"])
            ts = sorted({float(c["T"]) for c in pm_1y.index_cells(r)})
            below = [t for t in ts if t < horizon - 1e-9]
            out["n_kept_below"], out["nearest_kept_below"] = len(below), (
                max(below) if below else NAN
            )
            out["nonfinite"] = ", ".join(nonfinite_columns(r))
            out["gate_outside"] = bool(
                abs(r["idx_err_atm"]) > GATE_VP or abs(r["idx_err_90"]) > GATE_VP
            )
            out["forward_z"] = r["forward_error"] / r["forward_error_se"]
            for tag in CALL_TAGS:
                c, c_se = float(e[f"C_{tag}"]), float(e[f"C_se_{tag}"])
                for model in ("lc", "cc"):
                    a, a_se = float(r[f"C_{tag}_{model}"]), float(r[f"C_{tag}_{model}_se"])
                    ok = c > 0.0 and a > 0.0
                    out[f"C_{tag}_{model}_over_copula"] = a / c if ok else NAN
                    out[f"C_{tag}_{model}_over_copula_se"] = (
                        pc.ratio_se(a, a_se, c, c_se) if ok else NAN
                    )
        extra.append(out)
    d = pd.concat([d, pd.DataFrame(extra)], axis=1)
    if d.columns.duplicated().any():
        raise ValueError(f"duplicate columns: {sorted(d.columns[d.columns.duplicated()])}")
    for col in ("flag_clip", "flag_clip_low", "flag_clip_high", "flag_unscreened"):
        d[col] = truth(d[col]) & d["priced"]
    d["clip_larger"] = d[["clip_low_inner_max", "clip_high_inner_max"]].max(axis=1)
    d["n_dropped_calendar_names"] = d["n_dropped_calendar"] - d["n_dropped_calendar_index"]
    d["n_dropped_index_screen"] = d["n_dropped_index"] - d["n_dropped_calendar_index"]
    d["sample_a"] = d["priced"] & ~d["flag_clip"]
    d["sample_a2"] = d["sample_a"] & ~d["flag_unscreened"]
    d["sample_a_ok"] = d["sample_a"] & (d["status"] == "ok")
    d["sample_a2_ok"] = d["sample_a2"] & (d["status"] == "ok")
    d["sample_b"] = d["priced"]
    d["sample_c"] = d["priced"] & ~d["flag_unscreened"]
    d["sample_d"] = d["status"] == "ok"
    d["sample_d2"] = d["sample_d"] & ~d["flag_unscreened"]
    d["tenor_label"] = tenor
    return d


def load() -> Data:
    table = pd.read_parquet(TABLE)
    rows = {p.stem: json.loads(p.read_text()) for p in sorted(ROWS.glob("*.json"))}
    if set(rows) != set(table["date"]):
        raise ValueError("the table and the row files do not hold the same dates")
    entries = pm_1y.study_entries(TENOR)
    commits = sorted(set(table["git_commit"]))
    if len(commits) != 1:
        raise ValueError(f"several commits in the table: {commits}")
    d = derive(table, rows, entries)
    yearly = [str(x) for x in pd.read_csv(FROZEN_1Y / "tables" / "1y_dates.csv")["date"]]
    monthly = monthly_list(TENOR)
    d["in_monthly_list"] = d["date"].isin(monthly)
    d["yearly_date"] = d["date"].isin(yearly)
    return Data(d, rows, entries, monthly, yearly, commits[0])


# ----------------------------------------------------------------------------- statistics
def summarise(v: pd.Series, mc: pd.Series | None = None) -> dict[str, float]:
    """The statistics of a quantity on a sample, the dates in time order (module docstring)."""
    x = v.dropna().astype(float).to_numpy()
    n = len(x)
    out: dict[str, float] = {"n": n}
    if n == 0:
        return out
    q = np.quantile(x, [0.25, 0.5, 0.75])
    out.update(
        mean=float(x.mean()),
        q25=float(q[0]),
        median=float(q[1]),
        q75=float(q[2]),
        min=float(x.min()),
        max=float(x.max()),
    )
    out["se"] = float(x.std(ddof=1) / math.sqrt(n)) if n >= MIN_N_SE else NAN
    out["nw"] = ph.newey_west_se(x, NW_LAGS) if n >= MIN_N_NW else NAN
    for lags in NW_OTHER:
        out[f"nw_{lags}"] = ph.newey_west_se(x, lags) if n >= MIN_N_NW else NAN
    out["ac1"] = ph.lag1_autocorr(x) if n >= MIN_N_SE else NAN
    if mc is not None:
        s = mc[v.notna()].astype(float)
        out["mc_median"] = float(s.median()) if s.notna().any() else NAN
        out["mc_max"] = float(s.max()) if s.notna().any() else NAN
    return out


def no_se_reason(n: int) -> str:
    return f"n = {n}: fewer than {MIN_N_SE} dates, no standard error is carried"


def summary_records(
    b: Book,
    key: str,
    label: str,
    definition: str,
    sample: str,
    sample_label: str,
    s: dict[str, float],
    study: bool = False,
) -> None:
    n = int(s["n"])
    base = {"unit": "ratio", "definition": definition, "n": n}
    if study:
        base.update(budget=pc.BUDGETS["study"], commit=pm_1y.STUDY_NOTE, source=STUDY)
    why = (
        "standard error across dates, sd/√n, the dates treated as independent"
        if finite(s.get("se"))
        else no_se_reason(n)
    )
    b.add(
        f"fwd.{key}.mean.{sample}",
        f"{label}: mean, {sample_label}",
        s.get("mean"),
        s.get("se"),
        notes=why,
        **base,
    )
    for stat in ("median", "q25", "q75", "min", "max"):
        b.add(
            f"fwd.{key}.{stat}.{sample}",
            f"{label}: {stat}, {sample_label}",
            s.get(stat),
            notes="an order statistic across dates: no standard error",
            **base,
        )
    b.add(
        f"fwd.{key}.mean_nw_se.{sample}",
        f"{label}: Newey–West standard error of the mean ({NW_LAGS} lags), {sample_label}",
        s.get("nw"),
        notes=f"an error, with no error of its own; printed from {MIN_N_NW} dates on",
        **base,
    )
    if not study:
        b.add(
            f"fwd.{key}.mc_median.{sample}",
            f"{label}: median Monte Carlo standard error of a date, {sample_label}",
            s.get("mc_median"),
            notes="an error, with no error of its own",
            **base,
        )


# ----------------------------------------------------------------------------- self checks
def self_checks(b: Book, data: Data) -> None:
    d, p = data.d, data.priced
    table = pd.read_parquet(TABLE).set_index("date")
    # 1. the table against its row files
    n_cells = 0
    for date, row in data.rows.items():
        for k, v in row.items():
            if isinstance(v, (list, dict)) or k not in table.columns:
                continue
            t = table.loc[date, k]
            n_cells += 1
            both_missing = (v is None or (isinstance(v, float) and math.isnan(v))) and (
                t is None or (isinstance(t, float) and math.isnan(t))
            )
            if not (both_missing or t == v):
                raise ValueError(f"table and row file differ: {date} {k}: {t!r} against {v!r}")
    b.checks.append(
        f"The table `{rel(TABLE)}` equals its {len(data.rows)} row files on every scalar column ({n_cells} cells compared; the list `index_errors` is in the row files only)."
    )
    # 2. the copula columns against the study's entries
    worst = 0.0
    for date in p["date"]:
        r, e = data.rows[date], data.entries.loc[date]
        for rk, ek in pm_1y.ENTRY_COLUMNS:
            worst = max(worst, abs(float(r[rk]) - float(e[ek])))
    if worst > pm_1y.ENTRY_TOLERANCE:
        raise ValueError(f"a row's copula column differs from the study's entry by {worst}")
    b.checks.append(
        f"The copula columns the {len(p)} priced rows repeat (P_D, EV, EQV, T and the cash strikes K_075 to K_150) equal basket B1 of the study's `entries_12m.parquet` (largest absolute difference {worst:.1g})."
    )
    # 3. flags and status from their definitions
    raw = table.loc[p["date"]]
    low, high = raw["clip_low_inner_max"] > CLIP_FLAG, raw["clip_high_inner_max"] > CLIP_FLAG
    same = (
        (truth(raw["flag_clip_low"]) == low).all()
        and (truth(raw["flag_clip_high"]) == high).all()
        and (truth(raw["flag_clip"]) == (low | high)).all()
    )
    same = (
        same
        and (truth(raw["flag_unscreened"]) == (raw["n_names_unscreened"] > 0)).all()
        and (truth(raw["wing_binds"]) == (low | high)).all()
    )
    status = np.where(np.column_stack([truth(raw[c]) for c in GATING]).all(axis=1), "ok", "check")
    nan_ok = all(
        (not nonfinite_columns(data.rows[x])) == bool(data.rows[x]["check_no_nan"])
        for x in p["date"]
    )
    if not (same and (status == raw["status"].to_numpy()).all() and nan_ok):
        raise ValueError("a flag or a status differs from its definition")
    b.checks.append(
        "Recomputed from their definitions and equal to the stored columns on every priced date: `flag_clip_low`, `flag_clip_high`, `flag_clip` (each mass strictly above 0.01), `wing_binds` (the same condition), `flag_unscreened` (`n_names_unscreened > 0`), the status (`check` when one of `check_no_nan`, `check_forward`, `check_index` fails) and `check_no_nan` (a float of the row is not finite)."
    )
    # 4. the automatic report's section 1
    text = REPORT.read_text()
    for label, key in (
        ("E_LC[D] / E_CC[D] (paired)", "ratio"),
        ("E_LC[D] / copula P_D", "lc_over_copula"),
        ("E_CC[D] / copula P_D", "cc_over_copula"),
    ):
        m = re.search(re.escape(f"| {label} | ") + r"(\d+) \| ([\d.]+) \|", text)
        if (
            m is None
            or int(m.group(1)) != len(p)
            or abs(float(m.group(2)) - float(raw[key].mean())) > 5.1e-5
        ):
            raise ValueError(f"the automatic report's line '{label}' is not reproduced")
    b.checks.append(
        f"The means of sample (b) reproduce section 1 of the automatic report `{rel(REPORT)}` to its four printed decimals (n {len(p)}); that report's summaries are all on sample (b), flagged rows included."
    )
    # 5. the least-squares routine
    mine = ols_fit(p["lc_over_cc"], p[["clip_larger"]], ["clip"], lags=ph.NW_LAGS)
    theirs = ph.ols(p["lc_over_cc"], p[["clip_larger"]], ["clip"])
    if not (
        np.allclose(mine["coef"], theirs["coef"])
        and np.allclose(mine["se_hc1"], theirs["se_hc1"])
        and np.allclose(mine["se_nw"], theirs["se_nw"])
    ):
        raise ValueError("ols_fit differs from pm_history.ols")
    b.checks.append(
        "The least-squares routine equals `pm_history.ols` (coefficients, HC1 and Newey–West errors) at that routine's own 6 lags."
    )
    if len(d) != len(table):
        raise ValueError("rows lost")


# ----------------------------------------------------------------------------- the specification-only build
def index_target(spec: Any) -> Any:
    """The index target of a specification: the SVI slices of ``spec.index_surface`` on the curve
    F_B ≡ 1, which is what ``volsto.calibration.lc_cache.build_lc_market`` builds in performance
    mode without a perturbation, here without the names' Dupire surfaces.  :func:`spec_build`
    compares it with the row's ``target_vol`` at every strike of the horizon."""
    from volsto.market.curves import DiscountCurve, ForwardCurve
    from volsto.market.svi_slices import SviSlices

    if spec.lc.mode != "performance" or spec.index_perturbation is not None:
        raise ValueError("index_target: performance mode without a perturbation is assumed")
    c = spec.index_surface
    zero = DiscountCurve.flat(0.0)
    return SviSlices(np.array(c.times), np.array(c.params), ForwardCurve(1.0, zero, zero), c.max_maturity)  # fmt: skip


def same_index_surface(a: Any, b: Any) -> bool:
    """Whether two specifications hold the same index surface (slices and SVI parameters)."""
    x, y = a.index_surface, b.index_surface
    return (
        np.array_equal(np.array(x.times), np.array(y.times))
        and np.array_equal(np.array(x.params), np.array(y.params))
        and x.max_maturity == y.max_maturity
    )


def spec_build(data: Data) -> None:
    """The specification of every date of the table, as ``scripts/lcm_price.py`` builds it — no
    calibration, no simulation (module docstring).  Writes ``tables/spec_*.csv``."""
    import os

    os.environ.setdefault("NUMBA_NUM_THREADS", "3")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import lcm_diagnostics as lcd
    import lcm_price as lp
    import pm_1y_slices as sl

    import volsto.market.svi_slices as svi
    from volsto.calibration.lc_cache import lc_spec_key
    from volsto.market.curves import ForwardCurve
    from volsto.multi.analytics import strip_second_moment

    for name in ("volsto.studies.disp_lc", "lcm_diagnostics", "lcm_price"):
        logging.getLogger(name).setLevel(logging.WARNING)
    cfg = lp.load_config(str(lp.CONFIG))
    cfg_off = copy.deepcopy(cfg)
    cfg_off["screen"]["calendar_repair"] = False
    records = sl.ReadOnlyRecords(pc.LC_OUT.parents[1] / cfg["cache"] / "svi_fits")
    head = git("rev-parse", "--short", "HEAD").strip()
    built = time.strftime("%Y-%m-%d %H:%M")
    asked: dict[str, Any] = {}
    original = svi.recorded_svi_fit

    def watching(*args: Any, **kwargs: Any) -> Any:
        asked.update(origin=kwargs.get("origin", ""), T=float(args[2]) if len(args) > 2 else NAN)
        return original(*args, **kwargs)

    by_date, strips, failed, unscreened = [], [], [], []
    svi.recorded_svi_fit = watching  # type: ignore[assignment]
    try:
        for date, priced, far in zip(
            data.d["date"], data.d["priced"], data.d["names_svi_over_listed"], strict=True
        ):
            t0 = time.perf_counter()
            asked.clear()
            inp = lcd.load_inputs(date, TENOR, cfg["index"])
            try:
                spec, info = lp.spec_for(inp, cfg, "development", records)
            except Exception as exc:
                origin = str(asked.get("origin", ""))
                failed.append(
                    {
                        "date": date,
                        "raised": f"{type(exc).__name__}: {exc}",
                        "last_fit_asked": origin,
                        "name": origin.split(":")[-1],
                        "slice_T": asked.get("T", NAN),
                        "row_reason": data.rows[date].get("reason", ""),
                        "row_priced": bool(priced),
                    }
                )
                LOG.info(
                    "%s: the specification build raised %s (last fit asked: %s)", date, exc, origin
                )
                continue
            horizon = float(inp.T)
            kept = sorted(float(t) for t in info["index_slices"])
            below = [t for t in kept if t <= horizon + 1e-9]
            above = [t for t in kept if t > horizon + 1e-9]
            gone = [g for g in info["dropped"] if g["leg"] == "index"]
            cal = sorted(float(g["T"]) for g in gone if g["rule"] == "calendar")
            scr = sorted(float(g["T"]) for g in gone if g["rule"] != "calendar")
            line: dict[str, Any] = {
                "date": date, "horizon": horizon, "spec_key_is_the_rows": lc_spec_key(spec) == data.rows[date].get("spec_key"), "row_priced": bool(priced),
                "kept_djx_slices": " ".join(f"{t:.3f}" for t in kept), "nearest_kept_below": max(below) if below else NAN, "nearest_kept_above": min(above) if above else NAN,
                "dropped_by_calendar_repair": " ".join(f"{t:.3f}" for t in cal), "n_dropped_calendar_index": len(cal), "dropped_by_screen": " ".join(f"{t:.3f}" for t in scr),
                "names_unscreened": ",".join(info["names_unscreened"]), "names_beyond_last_kept_expiry": ",".join(info["names_extrapolated"]), "index_extrapolated": bool(info["index_extrapolated"]),
            }  # fmt: skip
            if priced:
                # the index target: rebuilt, read at the spot strike, and with the repair off
                cells = [c for c in pm_1y.index_cells(data.rows[date]) if abs(float(c["T"]) - horizon) < 1e-9]  # fmt: skip
                target = index_target(spec)
                on = {c["strike"]: float(target.implied_vol_k(float(c["k"]), horizon)) for c in cells}  # fmt: skip
                f_b = float(data.entries.loc[date, "f_B"])
                line.update(
                    study_forward_over_spot=f_b,
                    target_atm_built=100 * on["+0.0"],
                    target_rebuilt_max_abs_diff_vp=100 * max(abs(on[c["strike"]] - float(c["target_vol"])) for c in cells),
                    target_at_spot_strike=100 * float(target.implied_vol_k(-math.log(f_b), horizon)),
                    target_smile_repair_on=" ".join(f"{100 * on[s]:.2f}" for s in SMILE_STRIKES),
                )  # fmt: skip
                try:
                    spec_off, info_off = lp.spec_for(inp, cfg_off, "development", records)
                    target_off = index_target(spec_off)
                    off = {c["strike"]: float(target_off.implied_vol_k(float(c["k"]), horizon)) for c in cells}  # fmt: skip
                    line.update(
                        norepair_raised="",
                        norepair_index_surface_equal=same_index_surface(spec, spec_off),
                        kept_djx_slices_norepair=" ".join(f"{float(t):.3f}" for t in sorted(info_off["index_slices"])),
                        target_atm_norepair=100 * off["+0.0"],
                        repair_effect_atm_vp=100 * (on["+0.0"] - off["+0.0"]),
                        repair_effect_max_abs_vp=100 * max(abs(on[k] - off[k]) for k in on),
                        target_smile_repair_off=" ".join(f"{100 * off[s]:.2f}" for s in SMILE_STRIKES),
                    )  # fmt: skip
                except Exception as exc:
                    line["norepair_raised"] = f"{type(exc).__name__}: {exc}"
                    LOG.info("%s: the build with the repair off raised %s", date, exc)
                # the names decision 2 keeps unscreened: their own surface in the specification
                w_all = np.asarray(inp.weights, dtype=float) / float(np.sum(inp.weights))
                for name in info["names_unscreened"]:
                    i = list(spec.names).index(name)
                    curve, c = ForwardCurve.from_config(spec.markets[i]), spec.surfaces[i]
                    surface = svi.SviSlices(np.array(c.times), np.array(c.params), curve, c.max_maturity)  # fmt: skip
                    unscreened.append({
                        "date": date, "name": str(name), "weight": float(w_all[i]), "listed_spot": float(inp.spots[i]), "kept_slices": " ".join(f"{float(t):.3f}" for t in c.times), "n_kept_slices": len(c.times),
                        "forward_over_spot": float(curve.forward(horizon)) / float(curve.spot), "atm_vol": float(surface.atm_vol(horizon)),
                    })  # fmt: skip
            line["seconds"] = round(time.perf_counter() - t0, 2)
            by_date.append(line)
            if priced and finite(far) and far > STRIP_FAR:
                market = lp.build_lc_market(spec)
                w = np.asarray(inp.weights, dtype=float) / float(np.sum(inp.weights))
                listed = np.asarray(inp.entry["legs"]["M"], dtype=float)
                per = []
                for name, wi, s, ml in zip(inp.names, w, market.surfaces, listed, strict=True):
                    total, parts = strip_second_moment(s, horizon, splits=[0.0])  # type: ignore[misc]
                    f_ratio = float(s.forward_curve.forward(horizon)) / float(s.forward_curve.spot)
                    a = max(float(s.atm_vol(horizon)) * math.sqrt(horizon), 0.1)
                    k_hi = max(8.0 * a, math.log(3.0 / f_ratio))
                    per.append({
                        "date": date, "name": str(name), "weight": float(wi), "M_svi": float(total), "M_listed": float(ml), "weighted_M_svi": float(wi * total),
                        "put_side": float(parts[0]), "call_side": float(parts[1]), "k_upper_edge": k_hi, "strike_over_forward_at_edge": math.exp(k_hi),
                        "total_variance_at_upper_edge": float(np.asarray(s.total_variance(np.array([k_hi]), horizon))[0]), "atm_vol": float(s.atm_vol(horizon)),
                    })  # fmt: skip
                total_w = sum(x["weighted_M_svi"] for x in per)
                per.sort(key=lambda x: -x["weighted_M_svi"])
                for rank, x in enumerate(per[:3], 1):
                    strips.append(
                        {
                            **x,
                            "rank": rank,
                            "share_of_sum": x["weighted_M_svi"] / total_w,
                            "sum_w_M_svi_built": total_w,
                            "sum_w_M_svi_row": float(data.rows[date]["sum_w_M_svi"]),
                        }
                    )
            LOG.info("%s: specification built in %.1f s", date, time.perf_counter() - t0)
    finally:
        svi.recorded_svi_fit = original  # type: ignore[assignment]
    for name, frame in (
        ("spec_by_date", by_date),
        ("spec_strips_by_name", strips),
        ("spec_failed_dates", failed),
        ("spec_unscreened_names", unscreened),
    ):
        out = pd.DataFrame(frame)
        out["config"], out["worktree_commit"], out["built"] = rel(lp.CONFIG), head, built
        pc.save_table(out, name, base=OUT)
    LOG.info(
        "specification-only build: %d dates built, %d raised, %d strip rows, %d unscreened names",
        len(by_date),
        len(failed),
        len(strips),
        len(unscreened),
    )


def read_spec(name: str) -> pd.DataFrame | None:
    path = OUT / "tables" / f"{name}.csv"
    return pd.read_csv(path) if path.exists() else None


def slices_of(cell: Any) -> list[float]:
    """The maturities (years) of a cell of ``tables/spec_by_date.csv`` that lists DJX slices."""
    return [float(x) for x in cell.split()] if isinstance(cell, str) else []


def slice_list(ts: list[float]) -> str:
    return ", ".join(f"{t:.3f}" for t in ts) + "y" if ts else "none"


def listed_index_expiries(dates: list[str]) -> dict[str, list[float]]:
    """The maturities (years, in time order) of the DJX expiries the inputs list on each of
    ``dates``: the list ``index_smiles`` of the pricer's own loader
    (``lcm_diagnostics.load_inputs``, what ``lcm_price.spec_for`` builds the specification
    from), under the worktree's configuration.  Nothing is fitted, built or written."""
    import os

    os.environ.setdefault("NUMBA_NUM_THREADS", "3")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import lcm_diagnostics as lcd
    import lcm_price as lp

    for name in ("volsto.studies.disp_lc", "lcm_diagnostics", "lcm_price"):
        logging.getLogger(name).setLevel(logging.WARNING)
    cfg = lp.load_config(str(lp.CONFIG))
    return {
        x: sorted(float(e.T) for e in lcd.load_inputs(x, TENOR, cfg["index"]).index_smiles)
        for x in dates
    }


def index_expiry_account(
    date: str,
    listed: list[float],
    kept: list[float],
    screen: list[float],
    repair: list[float],
    horizon: float,
) -> dict[str, Any]:
    """What became of each DJX expiry the inputs list on a date: kept, dropped by the quote
    screen, dropped by the calendar repair, or left out by the surface's selection
    (``volsto.market.svi_slices.fit_svi_surface``: the expiries of at least ``MIN_SLICE_T`` up to
    the horizon and the next ``N_BEYOND`` after it; an expiry it leaves out is neither kept nor
    reported as dropped).  ``kept``, ``screen`` and ``repair`` are the specification-only
    build's lists, at the three decimals of its table.  Raises unless each of them is a listed
    expiry, once, and each expiry left out is left out by one of the selection's two rules, so
    that kept + dropped + left out = listed."""
    from volsto.market.svi_slices import MIN_SLICE_T, N_BEYOND

    names = [f"{t:.3f}" for t in listed]
    full = dict(zip(names, listed, strict=True))
    used = [f"{t:.3f}" for t in (*kept, *screen, *repair)]
    if len(set(names)) != len(names) or len(set(used)) != len(used) or not set(used) <= set(names):
        raise ValueError(
            f"{date}: the build's kept and dropped DJX slices are not listed expiries, each once"
        )
    after = sorted(full[f"{t:.3f}"] for t in kept if full[f"{t:.3f}"] > horizon)
    left_out = [t for n, t in full.items() if n not in set(used)]
    beyond = [t for t in left_out if t > horizon and sum(1 for u in after if u < t) >= N_BEYOND]
    short = [t for t in left_out if t < MIN_SLICE_T]
    if len(beyond) + len(short) != len(left_out) or set(beyond) & set(short):
        raise ValueError(f"{date}: a listed DJX expiry is left out by no rule of the selection")
    if len(kept) + len(screen) + len(repair) + len(left_out) != len(listed):
        raise ValueError(f"{date}: kept + dropped + left out is not the listed DJX expiries")
    return {
        "listed": listed,
        "left_out": left_out,
        "beyond": beyond,
        "short": short,
        "kept_after": after,
        "n_beyond": int(N_BEYOND),
        "min_slice_t": float(MIN_SLICE_T),
    }


SPEC_NOTE = "a specification-only build by this script (`--spec-build`: the inputs loaded and the specification built as `scripts/lcm_price.py` builds it, under `configs/studies/dispersion/lcm.yaml`; no calibration, no simulation; the runs' SVI fit records read through a store that never writes)"


SPEC_TARGET_COLUMNS = ("target_atm_built", "target_rebuilt_max_abs_diff_vp", "target_at_spot_strike", "norepair_index_surface_equal", "target_atm_norepair", "repair_effect_atm_vp", "repair_effect_max_abs_vp")  # fmt: skip
GAP_COLUMNS = ("gap_same_strike", "gap_strike_part")


def attach_spec(b: Book, data: Data) -> bool:
    """The index-target columns of the specification-only build on the per-date frame: the
    target read at the spot strike of the study's forward, the two parts of the gap, and the
    target with the calendar repair off.  Checked here: the rebuilt target against the row's
    ``target_vol`` at every strike of the horizon and against the row's at-the-money target, the
    forward against the entry's, and the target with the repair off against the old-defaults
    rows.  Returns whether the columns are there (the build was run with them)."""
    spec = read_spec("spec_by_date")
    if spec is None or "target_at_spot_strike" not in spec.columns:
        for c in (*SPEC_TARGET_COLUMNS, *GAP_COLUMNS):
            data.d[c] = NAN
        return False
    d = data.d.merge(
        spec[["date", "study_forward_over_spot", "norepair_raised", *SPEC_TARGET_COLUMNS]],
        on="date",
        how="left",
    )
    priced = d["priced"]
    raised = d.loc[priced & d["norepair_raised"].notna(), "date"]
    if d.loc[priced, "target_at_spot_strike"].isna().any() or len(raised):
        raise ValueError(
            f"a priced date has no rebuilt target (repair off raised on {list(raised)})"
        )
    worst = float(d.loc[priced, "target_rebuilt_max_abs_diff_vp"].abs().max())
    worst_atm = float((d.loc[priced, "target_atm_built"] - d.loc[priced, "target_atm"]).abs().max())
    worst_f = float((d.loc[priced, "study_forward_over_spot"] - d.loc[priced, "study_fwd"]).abs().max())  # fmt: skip
    # the CSV holds ten significant digits
    if worst > 1e-7 or worst_atm > 1e-7 or worst_f > 1e-8:
        raise ValueError(
            f"the rebuilt index target is not the rows': {worst}, {worst_atm}, {worst_f}"
        )
    d["gap_same_strike"] = d["target_at_spot_strike"] - d["study_atm"]
    d["gap_strike_part"] = d["target_atm"] - d["target_at_spot_strike"]
    data.d = d.drop(columns=["study_forward_over_spot", "norepair_raised"])
    # the DJX slices the build lists as dropped, against the rows' counts; the bracket of the horizon
    by = spec.set_index("date")
    for _, r in d[priced].iterrows():
        s = by.loc[r["date"]]
        counts = (len(slices_of(s["dropped_by_screen"])), len(slices_of(s["dropped_by_calendar_repair"])))  # fmt: skip
        if counts != (int(r["n_dropped_index_screen"]), int(r["n_dropped_calendar_index"])):
            raise ValueError(
                f"{r['date']}: the build drops {counts} DJX slices, the row counts otherwise"
            )
        below = s["nearest_kept_below"] if finite(s["nearest_kept_below"]) else NAN
        row_below = r["nearest_kept_below"] if finite(r["nearest_kept_below"]) else NAN
        if finite(below) != finite(row_below) or (finite(below) and abs(below - row_below) > 1e-6):
            raise ValueError(
                f"{r['date']}: the nearest kept DJX slice below the horizon is not the row's"
            )
    b.checks.append(
        f"On the {int(priced.sum())} priced dates the DJX slices the specification-only build lists as dropped by the screen and by the calendar repair are as many as the row counts (`n_dropped_index − n_dropped_calendar_index`, `n_dropped_calendar_index`), and its nearest kept slice below the horizon is the last slice of the row's `index_errors` before the horizon."
    )
    b.checks.append(
        f"The index target rebuilt by the specification-only build (`index_target`) equals the row's `target_vol` at every strike of the horizon on the {int(priced.sum())} priced dates (largest difference {worst:.1g} vp), its at-the-money value equals the row's to {worst_atm:.1g} vp (the CSV holds ten significant digits), and the forward it reads at the spot strike is the entry's `f_B`."
    )
    # the target with the repair off against the old-defaults rows (decision 2 inert on them)
    if TABLE_NOREPAIR.exists():
        by = d.set_index("date")
        compared = []
        for path in sorted(ROWS_NOREPAIR.glob("*.json")):
            old = json.loads(path.read_text())
            if path.stem not in by.index or old.get("status") == "failed":
                continue
            if int(by.loc[path.stem, "n_names_unscreened"]) != 0:
                continue  # the old defaults also switch the fallback off: not the same build
            cell = [
                c
                for c in pm_1y.index_cells(old)
                if c["strike"] == "+0.0" and abs(float(c["T"]) - float(old["T"])) < 1e-9
            ]
            off, on = float(by.loc[path.stem, "target_atm_norepair"]), float(by.loc[path.stem, "target_atm"])  # fmt: skip
            gap = abs(100 * float(cell[0]["target_vol"]) - off)
            if gap > 1e-6:
                raise ValueError(
                    f"{path.stem}: the target with the repair off is not the old-defaults row's ({gap} vp)"
                )
            compared.append((path.stem, off, on, gap))
        if compared:
            b.checks.append(
                f"With `screen.calendar_repair` off the rebuilt target at the money equals that of the old-defaults row (`{rel(ROWS_NOREPAIR)}`) on "
                + "; ".join(
                    f"{x} ({off:.2f} % against {on:.2f} % with the repair; difference from the row {g:.1g} vp)"
                    for x, off, on, g in compared
                )
                + f": {count(len(compared), 'old-defaults row', 'old-defaults rows')} at one year."
            )
    return True


# ----------------------------------------------------------------------------- section 0 and 1
def what_fails(r: Any, diagnostic: bool = False) -> list[str]:
    """What fails on a priced date, check by check, with its numbers: the gating checks, and with
    ``diagnostic`` the names' diagnostic ``check_names`` (not a gate: decision 3)."""
    detail = []
    if not r["check_no_nan"]:
        detail.append(f"not finite: {r['nonfinite']}")
    if not r["check_forward"]:
        detail.append(
            f"basket forward error {r['forward_error']:+.5f} ± {r['forward_error_se']:.5f} ({r['forward_z']:+.1f} standard errors; the gate is 3)"
        )
    if not r["check_index"]:
        detail.append(
            f"index error {r['idx_err_atm']:+.3f} vp at the money and {r['idx_err_90']:+.3f} vp at the 90 % strike (the gate is {GATE_VP})"
        )
    if diagnostic and not r["check_names"]:
        detail.append(
            f"`check_names` (a diagnostic): Σ w E_LC[R_i²] is {100 * r['names_mc_over_listed']:+.1f} % from the listed strips (its bound is 2 %)"
        )
    return detail


def section_state(b: Book, data: Data) -> None:
    d, p = data.d, data.priced
    counts = d["status"].value_counts().to_dict()
    n, n_p = len(d), len(p)
    for s in ("ok", "check", "failed"):
        b.add(
            f"state.n_{s}",
            f"rows with status {s}",
            counts.get(s, 0),
            unit="dates",
            n=n,
            notes="a count",
        )
    b.add("state.n_rows", "rows of the table", n, unit="dates", n=n, notes="a count")
    b.add(
        "state.n_priced",
        "priced dates (status ok or check)",
        n_p,
        unit="dates",
        n=n,
        notes="a count",
    )
    outside = sorted(d.loc[~d["in_monthly_list"], "date"])
    missing = sorted(set(data.monthly) - set(d["date"]))
    yearly_in = [x for x in data.yearly if x in data.monthly]
    yearly_out = [x for x in data.yearly if x not in data.monthly]
    risk = d["risk"].value_counts().to_dict()
    gaps = pd.to_datetime(p["date"]).diff().dt.days.dropna()
    started = quoted(DRIVER_LOG, "pass 12m (development budget, risk none): 212 dates, 193 to run")
    b.add(
        "state.n_monthly_list",
        "dates of the pass's monthly list",
        len(data.monthly),
        unit="dates",
        notes="a count",
        source="outputs/dispersion/model_s_3m.parquet (converged dates) and the entry pickles of outputs/dispersion/entries/12m",
    )
    b.add(
        "state.n_gaps",
        "consecutive priced dates more than 45 days apart",
        int((gaps > 45).sum()),
        unit="pairs of dates",
        n=n_p,
        notes="a count",
    )
    b.text(
        "## 1. State of the pass",
        f"**Rows.** `{rel(TABLE)}` has {n} rows, {d['date'].iloc[0]} to {d['date'].iloc[-1]}: {counts.get('ok', 0)} with status ok, {counts.get('check', 0)} with status check, {counts.get('failed', 0)} failed; {n_p} are priced (ok or check). "
        f"All rows are at the development budget ({BUDGET}), commit {data.commit}, one configuration digest ({', '.join(sorted(set(d['config_digest'])))}), particle seed 12345 and pricing seed 2024 on every priced date.",
        f"**Which dates.** The pass's list (`scripts/disp_lcm.py --dates monthly`) is the dates on which the study's model S converged at three months (`model_s_3m.parquet`) plus {TODAY}, kept when the study has a one-year entry: {len(data.monthly)} dates, all in the table ({'none missing' if not missing else 'missing: ' + ', '.join(missing)}). "
        f"The {n}th row is {dates_text(outside)}: a date of the frozen addendum's yearly list (the first monthly one-year entry of 2024) that is not in the monthly list (model S has no converged row on it); its row was written on 9 Oct for the addendum and the table is rebuilt from every row file of the folder. "
        f"It is priced and is counted in every sample below; section 2 gives the mean of LC/CC on sample (b) without it. "
        f"The frozen addendum's twenty yearly dates: {len(yearly_in)} are in the monthly list, {len(yearly_out)} is not ({dates_text(yearly_out)}); all twenty are rows of the table, and they are the same row files the addendum read (the pass did not run again the {len(yearly_in)} of them that are in its list: the driver's log says \"{started}\", and counts {212 - 193} rows as already there; they carry the sticky-strike deltas, `risk = deltas`, on {risk.get('deltas', 0)} rows, the other {risk.get('none', 0)} rows have `risk = none`). "
        f"The priced dates are not evenly spaced: {int((gaps > 45).sum())} of the {n_p - 1} steps between consecutive priced dates are longer than 45 days (the longest {int(gaps.max())} days, into {TODAY}).",
    )
    # failed dates
    spec_failed = read_spec("spec_failed_dates")
    rows = []
    for _, r in d[~d["priced"]].iterrows():
        clause = ph.failure_clause(r["date"], r["reason"], LOGS)
        who = ""
        if spec_failed is not None and r["date"] in set(spec_failed["date"]):
            f = spec_failed.set_index("date").loc[r["date"]]
            if "Initial guess" in str(f["raised"]):
                who = f"{f['name']}, slice at {float(f['slice_T']):.3f}y (the last slice fit asked when the build raised)"
        decision = (
            'left failed by the owner\'s decision 2 ("Leave 2023-07-03 failed")'
            if r["date"] == "2023-07-03"
            else "not covered by a decision: failed, in no statistic"
        )
        rows.append([r["date"], "failed", f"`{r['reason']}`", clause, who or "—", decision])
    b.text(
        "**The failed dates** (in no statistic of this page; each was run twice with the same seeds):"
    )
    b.table(
        [
            "date",
            "status",
            "reason in the row",
            "step that raised (the run's log of the date)",
            "name and slice (specification-only build)",
            "standing",
        ],
        rows,
    )
    b.save(
        "failed_dates",
        pd.DataFrame(
            rows, columns=["date", "status", "reason", "step", "name_and_slice", "standing"]
        ),
    )
    if spec_failed is not None:
        b.text(
            f"The name and slice are from {SPEC_NOTE}; the build raises on {len(spec_failed)} dates, {int((~spec_failed['row_priced']).sum())} of them the failed dates of the pass. The three dates that are not 2023-07-03 fail in the SVI fit of one slice of one name between December 2008 and April 2009; the neighbouring priced dates carry decision 2's flag (below)."
        )
    # check dates
    c = p[p["status"] == "check"]
    fails = {g: sorted(c.loc[~truth(c[g]), "date"]) for g in GATING}
    for g in GATING:
        b.add(
            f"state.fail.{g}",
            f"priced dates on which {g} FAILS",
            int((~truth(p[g])).sum()),
            unit="dates",
            n=n_p,
            notes="a count",
        )
    b.add(
        "state.fail.check_names",
        "priced dates on which check_names FAILS (a reported diagnostic, not a gate: decision 3)",
        int((~truth(p["check_names"])).sum()),
        unit="dates",
        n=n_p,
        notes="a count",
    )
    kinds = c["nonfinite"].value_counts().to_dict()

    def model_vol_zero(date: str, strike: str) -> bool:
        row = data.rows[date]
        cells = [
            x
            for x in pm_1y.index_cells(row)
            if x["strike"] == strike and abs(float(x["T"]) - float(row["T"])) < 1e-9
        ]
        return bool(cells) and not float(cells[0]["model_vol"]) > 0

    zero_vol = {
        s: sum(model_vol_zero(x, s) for x in fails["check_no_nan"]) for s in ("+2.0", "+2.5")
    }
    for s, k in zero_vol.items():
        b.add(
            f"state.model_vol_zero.{s}",
            f"dates on which check_no_nan FAILS and the model vol of the row's index_errors is 0 at {s} sd at the horizon",
            k,
            unit="dates",
            n=len(fails["check_no_nan"]),
            notes="a count",
            source=rel(ROWS),
        )
    rows = []
    for _, r in c.iterrows():
        detail = what_fails(r)
        rows.append(
            [
                r["date"],
                chk(r["check_no_nan"]),
                chk(r["check_forward"]),
                chk(r["check_index"]),
                chk(r["check_names"]),
                yes(r["flag_clip"]),
                yes(r["flag_unscreened"]),
                "; ".join(detail),
            ]
        )
    b.text(
        f"**The {len(c)} dates with status check.** A date reads check when one of the three gating checks fails: `check_no_nan` (a float of the row is not finite) FAILS on {len(fails['check_no_nan'])} dates, `check_forward` (the basket's forward within 3 standard errors) FAILS on {len(fails['check_forward'])} ({dates_text(fails['check_forward'])}), `check_index` (index errors at the money and at the 90 % strike within {GATE_VP} vol points, waived when the wing binds) FAILS on {len(fails['check_index'])}. "
        f"On the `check_no_nan` dates the columns that are not finite are: "
        + "; ".join(f"`{k}` on {v} dates" for k, v in kinds.items() if k)
        + f". These are the standard errors of the index error at +2.0 and +2.5 at-the-money standard deviations at the horizon, where the model's implied vol is not invertible: the model vol of the row's `index_errors` is 0 at +2.5 sd at the horizon on {zero_vol['+2.5']} of these {len(fails['check_no_nan'])} dates and at +2.0 sd on {zero_vol['+2.0']} (no vega, no standard error); no price column reads them. "
        f"`check_names` (Σ w E_LC[R_i²] within 2 % of the listed strips) is a reported diagnostic, not a gate (decision 3): it FAILS on {int((~truth(p['check_names'])).sum())} of the {n_p} priced dates, {int((~truth(p.loc[p['status'] == 'ok', 'check_names'])).sum())} of them with status ok.",
    )
    b.table(
        [
            "date",
            "check_no_nan",
            "check_forward",
            "check_index",
            "check_names (diagnostic)",
            "clip flag (decision 5)",
            "unscreened flag (decision 2)",
            "what fails",
        ],
        rows,
    )
    b.save(
        "check_dates",
        pd.DataFrame(
            rows,
            columns=[
                "date",
                "check_no_nan",
                "check_forward",
                "check_index",
                "check_names",
                "flag_clip",
                "flag_unscreened",
                "what_fails",
            ],
        ),
    )
    outside_gate = p[p["gate_outside"]]
    b.add(
        "state.gate_outside",
        f"priced dates whose index error at the money or at the 90 % strike is beyond {GATE_VP} vol points",
        len(outside_gate),
        unit="dates",
        n=n_p,
        notes="a count; the gate is waived when the wing binds",
    )
    b.add(
        "state.gate_outside_ok",
        "of which status ok",
        int((outside_gate["status"] == "ok").sum()),
        unit="dates",
        n=len(outside_gate),
        notes="a count",
    )
    waived, kept = p[truth(p["wing_binds"])], p[~truth(p["wing_binds"])]
    if set(kept["date"]) != set(p.loc[p["sample_a"], "date"]):
        raise ValueError("the dates on which the index gate is not waived are not sample (a)")
    out_waived, out_kept = int(waived["gate_outside"].sum()), int(kept["gate_outside"].sum())
    n_index_pass = int(truth(p["check_index"]).sum())
    if (out_kept == 0) != bool(truth(kept["check_index"]).all()):
        raise ValueError(
            "check_index on the dates where the gate is not waived is not its two numbers"
        )
    b.add(
        "state.gate_not_waived",
        "priced dates on which the index gate is not waived (the wing does not bind)",
        len(kept),
        unit="dates",
        n=n_p,
        notes="a count; these are the dates of sample (a)",
    )
    b.add(
        "state.gate_not_waived.outside",
        f"of which dates whose index error at the money or at the 90 % strike is beyond {GATE_VP} vol points",
        out_kept,
        unit="dates",
        n=len(kept),
        notes="a count",
    )
    for key, label in (("idx_err_atm", "at the money"), ("idx_err_90", "at the 90 % strike")):
        b.add(
            f"state.gate_not_waived.max_abs_{key}",
            f"largest absolute index error {label} at the horizon over the dates on which the gate is not waived",
            kept[key].abs().max(),
            date=str(kept.loc[kept[key].abs().idxmax(), "date"]),
            unit="vol points",
            n=len(kept),
            notes=f"an order statistic across dates: no standard error; the Monte Carlo error of that date's index error is {kept.loc[kept[key].abs().idxmax(), key + '_se']:.3f} vp",
        )
    b.text(
        f"`check_index` passes on {n_index_pass} of the {n_p} priced dates. On the {len(waived)} dates where the wing binds the gate is waived, and that is why it passes there: the two gate numbers are beyond {GATE_VP} vol points on {out_waived} of those {len(waived)} dates{'' if out_waived == len(outside_gate) else f' ({len(outside_gate)} of the {n_p} priced dates)'}, {int((waived.loc[waived['gate_outside'], 'status'] == 'ok').sum())} of which read ok. "
        f"On the {len(kept)} dates where the wing does not bind — the dates of sample (a) — the gate is not waived and "
        + (
            f"passes on its numbers: the largest absolute index error over the {len(kept)} dates is {fmt(b.get('state.gate_not_waived.max_abs_idx_err_atm'), 3)} vp at the money and {fmt(b.get('state.gate_not_waived.max_abs_idx_err_90'), 3)} vp at the 90 % strike, against the gate's {GATE_VP} (the two numbers of each date, with their Monte Carlo errors, are in section 3). "
            if out_kept == 0
            else f"FAILS on {out_kept} of them. "
        )
        + "Status ok does not mean that the index smile is repriced on a date where the gate is waived."
    )
    # decisions
    counts5 = {
        "d5_index": int((p["n_dropped_calendar_index"] > 0).sum()), "d1_names": int((p["n_dropped_calendar_names"] > 0).sum()), "d2": int(p["flag_unscreened"].sum()),
        "beyond": int((p["n_names_extrapolated"] > 0).sum()), "index_extrapolated": int(truth(p["index_extrapolated"]).sum()), "clip": int(p["flag_clip"].sum()),
        "clip_low": int(p["flag_clip_low"].sum()), "clip_high": int(p["flag_clip_high"].sum()), "clip_both": int((p["flag_clip_low"] & p["flag_clip_high"]).sum()),
    }  # fmt: skip
    labels = {
        "d5_index": "priced dates with a DJX slice dropped by the calendar repair (decision 5)", "d1_names": "priced dates with a slice of a name dropped by the calendar repair (decision 1)",
        "d2": "priced dates with a name kept unscreened (decision 2's flag)", "beyond": "priced dates with a name priced beyond its last kept expiry", "index_extrapolated": "priced dates with the index target extrapolated beyond its last kept DJX slice",
        "clip": "priced dates with decision 5's clip flag", "clip_low": "priced dates with the mass clipped at λ = 0 above 1 %", "clip_high": "priced dates with the mass clipped at the cap above 1 %", "clip_both": "priced dates with both masses above 1 %",
    }  # fmt: skip
    for k, v in counts5.items():
        b.add(f"state.{k}", labels[k], v, unit="dates", n=n_p, notes="a count")
    unscreened = p[p["flag_unscreened"]]
    extrap = p[truth(p["index_extrapolated"])]
    b.text(
        f"**What the decisions do on the priced dates** (counts from the rows, over all priced dates [{INCL}]).",
        f"- Decision 5, the calendar repair of the DJX target: a DJX slice is dropped on {counts5['d5_index']} of {n_p} dates (`n_dropped_calendar_index > 0`; {int(p['n_dropped_calendar_index'].sum())} slices in all, at most {int(p['n_dropped_calendar_index'].max())} on a date).\n"
        f"- Decision 1, the calendar repair of the names' slices: a slice of a name is dropped on {counts5['d1_names']} of {n_p} dates (`n_dropped_calendar − n_dropped_calendar_index > 0`; median {p['n_dropped_calendar_names'].median():.0f} slices per date, at most {int(p['n_dropped_calendar_names'].max())}).\n"
        f"- Decision 2, a name kept unscreened and the date flagged: {counts5['d2']} dates — "
        + "; ".join(f"{r['date']} ({r['names_unscreened']})" for _, r in unscreened.iterrows())
        + ".\n"
        f"- A name priced beyond its last kept expiry (`n_names_extrapolated > 0`): {counts5['beyond']} dates ({int(p['n_names_extrapolated'].sum())} names in all, at most {int(p['n_names_extrapolated'].max())} on a date). This is a count of the row, not a flag of a decision.\n"
        f"- The index target extrapolated beyond its last kept DJX slice (`index_extrapolated`): {counts5['index_extrapolated']} dates — "
        + "; ".join(
            f"{r['date']} (last kept slice {r['index_last_slice']:.2f}y for a horizon of {r['T']:.2f}y)"
            for _, r in extrap.iterrows()
        )
        + ".\n"
        f"- Decision 5's clip flag (the mass clipped at λ = 0 or at the cap above 1 % inside ±2.5 sd): {counts5['clip']} of {n_p} dates — at the cap on {counts5['clip_high']}, at λ = 0 on {counts5['clip_low']}, both on {counts5['clip_both']}. {n_p - counts5['clip']} dates are not flagged.",
    )


def section_read_first(b: Book, data: Data) -> str:
    d, p = data.d, data.priced
    n_p, n_flag = len(p), int(p["flag_clip"].sum())
    counts = d["status"].value_counts().to_dict()
    finished = quoted(
        DRIVER_LOG, "=== 2026-10-10 02:05:59 pass 12m (development) finished in 4h42m31s: 213 rows"
    )
    # the log's own status count is over the pass's list; the table has one more row
    m = re.search(re.escape(finished) + r"[^\n]*statuses (\{[^}\n]*\})", DRIVER_LOG.read_text())
    if m is None:
        raise ValueError("the driver's log line of the finished pass carries no status count")
    logged = {k: int(v) for k, v in re.findall(r"'(\w+)': (\d+)", m.group(1))}
    listed = d[d["in_monthly_list"]]["status"].value_counts().to_dict()
    if logged != {k: int(v) for k, v in listed.items()}:
        raise ValueError(f"the log's status count {logged} is not the table's on the pass's list")
    outside = d[~d["in_monthly_list"]]
    a, a2 = p[p["sample_a"]], p[p["sample_a2"]]
    a_flagged = sorted(a.loc[a["flag_unscreened"], "date"])
    a_check = sorted(a.loc[a["status"] != "ok", "date"])
    single, basket = (Q1Y[k].definition.split(":")[0] for k in ("EV_single_part", "EV_basket_part"))  # fmt: skip
    # the row's M_B_listed is Σ w M − EQV (scripts/lcm_price.py): the identity is the row's own
    worst = max(abs(float(r["sum_w_M"]) - float(r["M_B_listed"]) - float(r["EQV"])) / max(1.0, abs(float(r["EQV"]))) for r in (data.rows[x] for x in p["date"]))  # fmt: skip
    if worst > 1e-12:
        raise ValueError(f"EQV is not Σ w M − M_B_listed in a row ({worst})")
    text = "\n\n".join([
        "## 0. Read this first",
        f"**What this is.** A note on a finished run, written after the freeze of the results package; nothing is priced here and no frozen number is changed. The run is the one-year (12m) pass of the calibrated local correlation model (LC) on the monthly dates: {len(d)} rows, {d['date'].iloc[0]} to {d['date'].iloc[-1]} ({counts.get('ok', 0)} ok, {counts.get('check', 0)} check, {counts.get('failed', 0)} failed), finished on 10 Oct 2026 at 02:06 New York time (the driver's log: \"{finished}\"; the log's own status count, {', '.join(f'{k} {v}' for k, v in logged.items())}, is over the {int(sum(listed.values()))} dates of the pass's list: the table's {counts.get('ok', 0)} ok include the {len(d)}th row, {dates_text(list(outside['date']))}, whose status is {', '.join(sorted(set(outside['status'])))}). "
        f"Budget: development, {BUDGET}. Commit {data.commit}: the defaults of 9 Oct, with the owner's decisions 1 (calendar repair of the names' slices), 2 (a name with no expiry passing the quote screen is kept unscreened and its date flagged) and 5 (the same calendar repair on the DJX target, and a flag on the clipped mass) on.",
        "**Terms.** D = Σ w_i |R_i − R̄| is the Palladium payoff on the 30 Dow names (R_i the name's return over the year, R̄ the basket's), in fractions of the notional. E_LC[D] is its forward under LC; E_CC[D] under the constant-correlation companion (CC: the same local vols, the one correlation that reprices the index at-the-money straddle at the horizon), priced on the same paths; P_D is the price of the dispersion study's copula model in the study's entry (basket B1). "
        "LC/CC = E_LC[D]/E_CC[D] is the only one of the three ratios that isolates local correlation; LC/copula and CC/copula also carry the gap between the model's index target and the study's index level — \"the gap\" of this page, which is the target's at-the-money vol read at the forward minus the study's index vol read at the spot (section 4 (i) says what it is made of). "
        "DJX is the listed index option on the Dow; SVI the parametric smile fitted to each listed expiry; sd an at-the-money standard deviation of the basket at the slice; vp a vol point (0.01 of implied vol); EQV the listed-option value of E[V] with V = Σ w_i R_i² − R̄², and EV the copula's E[V]; κ = E[D]/√E[V]; λ the local correlation parameter, clipped at 0 from below and at its cap from above. "
        "The clipped mass at λ = 0 (resp. at the cap) is the share of the particles inside ±2.5 sd whose λ is clipped there, at the calibration slice where that share is largest; \"the wing binds\" means that one of the two exceeds 1 %.",
        "**Terms of the frozen addendum used from section 4 on** (its own definitions, read from `scripts/pm_1y.py`). M_i^listed is the second moment E[R_i²] of name i read from its strip of listed options (`M` of the study's one-year legs) and M_B^listed the same for the basket's return R̄, from the study's DJX marginal: EQV = Σ w M_i^listed − M_B^listed (the row's `M_B_listed` is its Σ w M_i^listed minus the entry's EQV). "
        "E_cop[V] is the copula's E[V], the EV above, written with its model like E_LC[V] and E_CC[V]. "
        f"ED_wing = {Q1Y['ED_wing'].definition} and ED_eqv = {Q1Y['ED_eqv'].definition}, in fractions of the notional: ED_eqv is κ_LC applied to the listed EQV; ED_wing is κ_LC applied to the model's single-name second moment (Σ w E_LC[R_i²], a Monte Carlo estimate) minus the listed basket moment M_B^listed. "
        f"The single-name part is {single} and the basket part {basket}, in squared returns, so that E_LC[V] − EQV = single-name part − basket part. "
        "ATM is at the money: the forward (log-forward-moneyness 0) for the model's index target and for the index errors, the spot for the study's index level.",
        "**What each ± is.** There are three, and every table names the one it prints.\n"
        "1. *Per date*: the pricing Monte Carlo standard error of that date's row, given its calibrated model. It contains neither the calibration's own noise nor the budget: on the four dates priced at both budgets, LC/CC at the development budget differs from the production one by several of these errors (section 6).\n"
        f"2. *Across dates*: sd/√n of a mean over n dates, the dates treated as independent. It is printed when n ≥ {MIN_N_SE}.\n"
        f"3. *Newey–West*: the standard error of the same mean allowing for serial correlation (Bartlett kernel, {NW_LAGS} lags; a one-year price on a monthly date shares eleven months of horizon with the next one). The lags are counted in consecutive dates of the sample, which has gaps; it is printed when n ≥ {MIN_N_NW}, with the 6-lag and 24-lag values in the CSV.\n"
        f"Order statistics, counts and shares carry no error. A regression coefficient carries an HC1 (heteroskedasticity-robust) error and, as a mean does, a Newey–West one when the fit has {MIN_N_NW} dates or more; on a subsample selected on the clipped mass the consecutive dates the lags count can be months or years apart.",
        f"**How the sentences are written.** A mean is said to be below or above a level only when it is more than {K_SIGMA:g} times the larger of its across-dates and Newey–West errors away from it; otherwise it is said to be not distinguishable from it; on fewer than {MIN_N_SE} dates nothing is said about a sign. The distance is printed in errors, and when the verdict is not the same with the Newey–West error at {' and '.join(str(x) for x in NW_OTHER)} lags the sentence says so. The same rule with HC1 and Newey–West errors is applied to a regression slope (with the HC1 error alone when the fit has fewer than {MIN_N_NW} dates). "
        f"The source of a difference of LC/CC between two budgets is the larger, in absolute value, of its part from E_LC[D] and its part from E_CC[D] (section 6 (2)). These sentences are generated from the records of `parts/{PART}.json`.",
        "**The flags.** Two, both the owner's (9 Oct). *Decision 5's clip flag*: a date is flagged when the mass clipped at λ = 0 or at the cap exceeds 1 % inside ±2.5 sd; \"Flagged rows stay in the table and out of the summaries.\" *Decision 2's flag*: a name with no expiry passing the quote screen is kept unscreened and the date flagged; \"Report every summary with and without the flagged dates.\" A FAIL of a check is printed as FAIL in every table.",
        f"**The open decision.** At one year decision 5's flag fires on {n_flag} of the {n_p} priced dates. \"Out of the summaries\" therefore leaves {len(a)} dates: the summary by the owner's rule exists and is given with its n and without a standard error — as sample (a), first where it stands in one table or sentence with the others (sections 2, 3, 5, 6 (3) and 8), and, because {count(len(a_flagged), 'of its dates carries', 'of its dates carry')} decision 2's flag ({dates_text(a_flagged)}), also without {'it' if len(a_flagged) == 1 else 'them'} as sample (a2), {len(a2)} dates"
        + (f"; and, because {count(len(a_check), 'of its dates has', 'of its dates have')} status check ({dates_text(a_check)}), section 2 also gives both on the dates with status ok, as samples (a-ok) and (a2-ok), {int(p['sample_a_ok'].sum())} and {int(p['sample_a2_ok'].sum())} dates. " if a_check else ". ")
        +         f"The summary on all priced dates is sample (b) and is labelled every time \"{INCL}\"; so is every other statistic, fit or correlation of this page that contains flagged rows. Which of the two to quote at one year is the owner's decision and is not taken here.",
    ])  # fmt: skip
    return text


# ----------------------------------------------------------------------------- section 2
def section_forward(b: Book, data: Data) -> dict[tuple[str, str], dict[str, float]]:
    p = data.priced
    stats: dict[tuple[str, str], dict[str, float]] = {}
    long_rows = []
    for key, label, definition in FOUR_RATIOS:
        column = "listed_ratio" if key == "listed_fwd_ratio" else key
        se_col = None if key == "listed_fwd_ratio" else f"{key}_se"
        for sample, sample_label in SAMPLES:
            sub = p[p[f"sample_{sample}"]]
            s = summarise(sub[column], sub[se_col] if se_col else None)
            stats[(key, sample)] = s
            summary_records(
                b, key, label, definition, sample, sample_label, s, study=se_col is None
            )
            long_rows.append(
                {
                    "quantity": key,
                    "label": label,
                    "sample": sample,
                    "sample_label": sample_label,
                    **s,
                }
            )
    b.save("forward_summary", pd.DataFrame(long_rows))
    s3m_exists = (pc.STUDY / f"model_s_{TENOR}.parquet").exists()
    b.text(
        "## 2. The forward at one year",
        "Four ratios. LC/CC, LC/copula and CC/copula are defined in section 0; the fourth is the listed-variance forward over the copula's price, √(EQV/EV) = κ_cop·√EQV / P_D, a number of the study's entry alone (the row's `listed_fwd_ratio`). "
        f"**Model S has no one-year number**: the study's skewed model exists at three months only (`outputs/dispersion/model_s_3m.parquet`; there is {'a' if s3m_exists else 'no'} `model_s_12m.parquet`), and `scripts/pm_1y.py` reads no model S column. S/copula is therefore absent from every table of this page. "
        "The ratios to the copula carry the delta-method error of two independent estimates (the row's and the copula's own `P_D_se`), as in the frozen addendum; the table's own `lc_over_copula_se` leaves the copula's error out and is kept in the by-date CSV.",
    )
    for key, label, _ in FOUR_RATIOS:
        rows = []
        for sample, sample_label in SAMPLES:
            s = stats[(key, sample)]
            n = int(s["n"])
            rows.append([
                sample_label, str(n), fmt(s["mean"]), fmt(s["median"]), fmt(s["q25"]), fmt(s["q75"]), fmt(s["min"]), fmt(s["max"]),
                fmt(s["se"], 5) if finite(s["se"]) else f"none ({n} dates)", fmt(s["nw"], 5) if finite(s["nw"]) else f"none ({n} dates)",
                fmt(s.get("mc_median"), 5) if key != "listed_fwd_ratio" else "none (the study's number)",
            ])  # fmt: skip
        b.text(f"**{label}**")
        b.table(
            [
                "sample",
                "n",
                "mean",
                "median",
                "q25",
                "q75",
                "min",
                "max",
                "± across dates (sd/√n)",
                f"± Newey–West ({NW_LAGS} lags)",
                "median Monte Carlo ± of a date",
            ],
            rows,
        )
    sa, sb = stats[("lc_over_cc", "a")], stats[("lc_over_cc", "b")]
    sa2, sc = stats[("lc_over_cc", "a2")], stats[("lc_over_cc", "c")]
    unflagged = p[p["sample_a"]]
    a_flagged = sorted(unflagged.loc[unflagged["flag_unscreened"], "date"])
    wo = p[p["in_monthly_list"]]
    cop = [stats[(k, "b")] for k in ("lc_over_copula", "cc_over_copula", "listed_fwd_ratio")]
    nw_ratio = [x["nw"] / x["se"] for x in cop]
    grows = all(x["nw_6"] < x["nw"] < x["nw_24"] for x in cop)
    b.add(
        "fwd.lc_over_cc.mean.b_monthly_list_only",
        f"E_LC[D] / E_CC[D]: mean on the priced dates of the monthly list (without the row outside the list) [{INCL}]",
        wo["lc_over_cc"].mean(),
        wo["lc_over_cc"].std(ddof=1) / math.sqrt(len(wo)),
        unit="ratio",
        n=len(wo),
        notes="standard error across dates, sd/√n",
    )
    sa_ok, sa2_ok = stats[("lc_over_cc", "a_ok")], stats[("lc_over_cc", "a2_ok")]
    a_check = unflagged[unflagged["status"] != "ok"]
    small = [("a", sa), ("a2", sa2), ("a-ok", sa_ok), ("a2-ok", sa2_ok)]

    def failing(r: Any) -> str:
        """A date with status check: the gating checks that fail, with their numbers."""
        gates = ", ".join(f"`{g}`" for g in GATING if not (r[g] is True or r[g] == 1.0))
        return f"{r['date']}: {gates} FAIL — {'; '.join(what_fails(r))}"

    def copula_reading(key: str, label: str) -> str:
        """One ratio to the copula: sample (a) and (a2) beside (b) and (c), each with its verdict."""
        x, x2, y, z = (stats[(key, k)] for k in ("a", "a2", "b", "c"))
        col = "listed_ratio" if key == "listed_fwd_ratio" else key
        above = int((unflagged[col] > 1.0).sum())
        return (
            f"{label}: on sample (a) {fmt(x['mean'])} (n {int(x['n'])}; above 1 on {above} of the {int(x['n'])} dates) and on sample (a2) {fmt(x2['mean'])} (n {int(x2['n'])}) — {Rules.against_small(1.0, x, x2)}; "
            f"on sample (b) [{INCL}] {pm(y['mean'], y['se'])} (Newey–West {fmt(y['nw'])}): {Rules.against_lags(y['mean'], 1.0, y)}; "
            f"on sample (c) [{INCL}] {pm(z['mean'], z['se'])} (Newey–West {fmt(z['nw'])}): {Rules.against_lags(z['mean'], 1.0, z)}."
        )

    b.text(
        f"Samples: (a) is the owner's rule for decision 5 — the {int(sa['n'])} priced dates without the clip flag ({dates_text(list(unflagged['date']))}); on {int(sa['n'])} dates no standard error and no interval is printed. "
        f"{count(len(a_flagged), 'of them carries', 'of them carry')} decision 2's flag ({dates_text(a_flagged)}), and decision 2 asks for every summary with and without the flagged dates: (a2) is (a) without {'it' if len(a_flagged) == 1 else 'them'}, {int(sa2['n'])} dates ({dates_text(list(p.loc[p['sample_a2'], 'date']))}), again without a standard error. "
        + (
            f"{count(len(a_check), 'of the dates of (a) has', 'of the dates of (a) have')} status check ({'; '.join(failing(r) for _, r in a_check.iterrows())}): (a-ok) is (a) on its {int(sa_ok['n'])} dates with status ok and (a2-ok) is (a2) on its {int(sa2_ok['n'])}, both without a standard error. "
            if len(a_check)
            else "Every date of (a) has status ok: (a-ok) and (a2-ok) are (a) and (a2). "
        )
        + f"(b), (c), (d) and (d2) contain flagged rows: they are not the rule of decision 5. (c) leaves out the {int(p['flag_unscreened'].sum())} dates of decision 2's flag; (d) keeps the {int(p['sample_d'].sum())} dates with status ok, {int((p['sample_d'] & p['flag_unscreened']).sum())} of which carry decision 2's flag, and (d2) is (d) without them, {int(p['sample_d2'].sum())} dates. "
        f"The quartiles of samples (a), (a2), (a-ok) and (a2-ok) are interpolated between {listing([str(int(s['n'])) for _, s in small])} values. Without the row outside the monthly list ({dates_text(sorted(p.loc[~p['in_monthly_list'], 'date']))}), the mean of LC/CC on sample (b) [{INCL}] is {pm(b.get('fwd.lc_over_cc.mean.b_monthly_list_only'), b.se('fwd.lc_over_cc.mean.b_monthly_list_only'))} (n {len(wo)}) against {pm(sb['mean'], sb['se'])} (n {int(sb['n'])}).",
        f"Reading, by the rule of section 0. On sample (a), the rule of decision 5, the mean of LC/CC is {fmt(sa['mean'])} with a median of {fmt(sa['median'])} (n {int(sa['n'])}; from {fmt(sa['min'])} to {fmt(sa['max'])}), and on sample (a2), without decision 2's flagged date, {fmt(sa2['mean'])} with a median of {fmt(sa2['median'])} (n {int(sa2['n'])}; from {fmt(sa2['min'])} to {fmt(sa2['max'])}); "
        f"on their dates with status ok it is {fmt(sa_ok['mean'])} with a median of {fmt(sa_ok['median'])} on sample (a-ok) (n {int(sa_ok['n'])}) and {fmt(sa2_ok['mean'])} with a median of {fmt(sa2_ok['median'])} on sample (a2-ok) (n {int(sa2_ok['n'])}): {Rules.against_small(1.0, *(s for _, s in small))}. "
        f"On sample (b) [{INCL}] it is {pm(sb['mean'], sb['se'])} across dates (Newey–West {fmt(sb['nw'])}; n {int(sb['n'])}): {Rules.against_lags(sb['mean'], 1.0, sb)}; on sample (c) [{INCL}] {pm(sc['mean'], sc['se'])} (Newey–West {fmt(sc['nw'])}; n {int(sc['n'])}): {Rules.against_lags(sc['mean'], 1.0, sc)}. "
        + " ".join(copula_reading(key, label) for key, label, _ in FOUR_RATIOS[1:])
        + f" For the three ratios to the copula the Newey–West error is {fmt(min(nw_ratio), 1)} to {fmt(max(nw_ratio), 1)} times the across-dates one on sample (b), and from 6 to 12 to 24 lags it {'grows for each of the three' if grows else 'does not grow for each of the three'} (the CSV has the three values): quote the Newey–West one. For LC/CC the Newey–West error is {fmt(sb['nw'] / sb['se'], 2)} times the across-dates one on sample (b) [{INCL}] ({fmt(sb['nw'], 5)} against {fmt(sb['se'], 5)}).",
        f"Tables: `tables/forward_summary.csv` (every statistic, with the Newey–West error at 6, 12 and 24 lags and the lag-1 autocorrelation) and `tables/by_date.csv` (one line per date of the table, failed dates included, with every flag and check column and the sample each date belongs to; on the {int((~data.d['priced']).sum())} failed dates the flag columns are empty and the sample columns read False, so that the unflagged sample is `sample_a`, not `flag_clip == False`).",
    )
    return stats


# ----------------------------------------------------------------------------- section 3
def section_clip(b: Book, data: Data) -> dict[str, Any]:
    p = data.priced
    n_p = len(p)
    rows, dist = [], []
    for col, label in (
        ("clip_low_inner_max", "clipped at λ = 0"),
        ("clip_high_inner_max", "clipped at the cap"),
        ("clip_larger", "the larger of the two"),
    ):
        v = p[col]
        q = v.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        cells = {
            "min": v.min(),
            "q05": q[0.05],
            "q25": q[0.25],
            "median": q[0.5],
            "q75": q[0.75],
            "q95": q[0.95],
            "max": v.max(),
            "mean": v.mean(),
        }
        over = {t: int((v > t).sum()) for t in (0.01, 0.05, 0.10, 0.20)}
        rows.append(
            [label, str(n_p), *(fmt(x) for x in cells.values()), *(str(x) for x in over.values())]
        )
        dist.append(
            {"mass": col, "n": n_p, **cells, **{f"dates_above_{t:g}": k for t, k in over.items()}}
        )
        for stat, x in cells.items():
            b.add(
                f"clip.{col}.{stat}",
                f"{label}: {stat} over the priced dates [{INCL}]",
                x,
                unit="fraction of the particles",
                n=n_p,
                notes="a statistic of a calibration diagnostic across dates: no standard error",
            )
    b.save("clip_distribution", pd.DataFrame(dist))
    b.text(
        "## 3. The clipped mass",
        f"The two one-sided masses on the {n_p} priced dates, sample (b) [{INCL}] (fractions of the particles inside ±2.5 sd; a calibration diagnostic, without a standard error). Decision 5 flags a date when either exceeds 0.01.",
    )
    b.table(
        [
            "mass",
            "n",
            "min",
            "5 %",
            "q25",
            "median",
            "q75",
            "95 %",
            "max",
            "mean",
            "dates above 0.01",
            "above 0.05",
            "above 0.10",
            "above 0.20",
        ],
        rows,
    )
    low_larger = int((p["clip_low_inner_max"] > p["clip_high_inner_max"]).sum())
    b.add(
        "clip.low_larger",
        "priced dates on which the mass at λ = 0 is the larger of the two",
        low_larger,
        unit="dates",
        n=n_p,
        notes="a count",
    )
    b.text(
        f"The flag fires at the cap on {int(p['flag_clip_high'].sum())} dates, at λ = 0 on {int(p['flag_clip_low'].sum())}, on both on {int((p['flag_clip_low'] & p['flag_clip_high']).sum())}, on either on {int(p['flag_clip'].sum())}. The mass at λ = 0 is the larger of the two on {low_larger} dates and the mass at the cap on the other {n_p - low_larger} (the automatic report's reading of the cap side: the index target asks for more correlation than the cap allows, on the downside wing).",
        "**The dates that are not flagged** (sample (a), the rule of decision 5), with their rows. Each ± is the Monte Carlo error of the date.",
    )
    a = p[p["sample_a"]]
    spec = read_spec("spec_by_date")
    spec_i = spec.set_index("date") if spec is not None else None
    # the DJX expiries the inputs list on these dates (read from the inputs, not from the build)
    listed_ts = listed_index_expiries(list(a["date"])) if spec_i is not None else {}
    rows, rows2, un_rows = [], [], []
    sides: dict[str, dict[str, Any]] = {}
    for _, r in a.iterrows():
        rows.append([
            r["date"], r["status"], fmt(r["clip_low_inner_max"]), fmt(r["clip_high_inner_max"]), pm(r["lc_over_cc"], r["lc_over_cc_se"], 5), pm(r["lc_over_copula"], r["lc_over_copula_se"], 5), pm(r["cc_over_copula"], r["cc_over_copula_se"], 5),
            fmt(r["atm_gap"], 2, True),
        ])  # fmt: skip
        side: dict[str, Any] = {"built": spec_i is not None and r["date"] in spec_i.index}
        if side["built"]:
            s = spec_i.loc[r["date"]]
            below = float(s["nearest_kept_below"]) if finite(s["nearest_kept_below"]) else NAN
            above = float(s["nearest_kept_above"]) if finite(s["nearest_kept_above"]) else NAN
            width = above - below if finite(below) and finite(above) else NAN
            side.update(
                horizon=float(s["horizon"]), kept=slices_of(s["kept_djx_slices"]), screen=slices_of(s["dropped_by_screen"]), repair=slices_of(s["dropped_by_calendar_repair"]),
                below=below, above=above, width=width, extrapolated=not finite(width), wide=finite(width) and width > BRACKET_YEAR, repair_effect=float(r["repair_effect_atm_vp"]) if finite(r["repair_effect_atm_vp"]) else NAN,
            )  # fmt: skip
            if not finite(below) and not finite(above):
                raise ValueError(f"{r['date']}: no kept DJX slice")
            side["account"] = index_expiry_account(
                r["date"], listed_ts[r["date"]], side["kept"], side["screen"], side["repair"], side["horizon"]
            )  # fmt: skip
            side["read"] = (
                f"extrapolated back from the first kept slice, {above:.2f}y ({above - side['horizon']:.2f}y after the horizon)"
                if not finite(below)
                else (
                    f"extrapolated beyond the last kept slice, {below:.2f}y ({side['horizon'] - below:.2f}y before the horizon)"
                    if not finite(above)
                    else f"interpolated across {width:.2f}y"
                )
            )
        sides[r["date"]] = side
        fails = what_fails(r, diagnostic=True)
        rows2.append([
            r["date"], r["status"], pm(r["lc_over_cc"], r["lc_over_cc_se"], 5), chk(r["check_no_nan"]), chk(r["check_forward"]), chk(r["check_index"]) + (" (not waived)" if not (r["wing_binds"] is True or r["wing_binds"] == 1.0) else " (waived)"),
            pm(r["idx_err_atm"], r["idx_err_atm_se"], 3, True), pm(r["idx_err_90"], r["idx_err_90_se"], 3, True), chk(r["check_names"]), "; ".join(fails) if fails else "nothing",
            yes(r["flag_unscreened"]) + (f" ({r['names_unscreened']})" if r["flag_unscreened"] else ""), f"{int(r['n_names_extrapolated'])}",
            *(
                [f"{len(side['repair'])}" + (f" ({slice_list(side['repair'])})" if side["repair"] else ""), f"{len(side['screen'])}" + (f" ({slice_list(side['screen'])})" if side["screen"] else ""), slice_list(side["kept"]),
                 f"{years(side['below'])} / {years(side['above'])}", side["read"], fmt(side["repair_effect"], 2, True)]
                if side["built"]
                else [f"{int(r['n_dropped_calendar_index'])}", f"{int(r['n_dropped_index_screen'])}", "not built", "not built", "not built", "not built"]
            ),
        ])  # fmt: skip
        un_rows.append({
            "date": r["date"], "status": r["status"], "clip_low_inner_max": r["clip_low_inner_max"], "clip_high_inner_max": r["clip_high_inner_max"], "lc_over_cc": r["lc_over_cc"], "lc_over_cc_se": r["lc_over_cc_se"],
            "lc_over_copula": r["lc_over_copula"], "lc_over_copula_se": r["lc_over_copula_se"], "cc_over_copula": r["cc_over_copula"], "cc_over_copula_se": r["cc_over_copula_se"], "atm_gap": r["atm_gap"],
            "check_no_nan": chk(r["check_no_nan"]), "check_forward": chk(r["check_forward"]), "forward_z": r["forward_z"], "check_index": chk(r["check_index"]), "index_gate_waived": bool(r["wing_binds"] is True or r["wing_binds"] == 1.0),
            "idx_err_atm": r["idx_err_atm"], "idx_err_atm_se": r["idx_err_atm_se"], "idx_err_90": r["idx_err_90"], "idx_err_90_se": r["idx_err_90_se"], "check_names": chk(r["check_names"]), "names_mc_over_listed": r["names_mc_over_listed"],
            "what_fails": "; ".join(fails), "flag_unscreened": bool(r["flag_unscreened"]), "names_unscreened": r["names_unscreened"], "n_names_extrapolated": int(r["n_names_extrapolated"]),
            "n_djx_dropped_by_repair": int(r["n_dropped_calendar_index"]), "djx_dropped_by_repair": " ".join(f"{t:.3f}" for t in side.get("repair", [])), "n_djx_dropped_by_screen": int(r["n_dropped_index_screen"]), "djx_dropped_by_screen": " ".join(f"{t:.3f}" for t in side.get("screen", [])),
            "djx_kept": " ".join(f"{t:.3f}" for t in side.get("kept", [])), "horizon": side.get("horizon", NAN), "nearest_kept_below": side.get("below", NAN), "nearest_kept_above": side.get("above", NAN), "bracket_width": side.get("width", NAN),
            "target_extrapolated": side.get("extrapolated"), "target_interpolated_across_more_than_a_year": side.get("wide"), "one_year_target": side.get("read", "not built"), "repair_effect_atm_vp": side.get("repair_effect", NAN),
        })  # fmt: skip
        acc = side.get("account")
        un_rows[-1].update({
            "n_djx_listed": len(acc["listed"]) if acc else NAN, "djx_listed": " ".join(f"{t:.3f}" for t in acc["listed"]) if acc else "", "n_djx_kept": len(side["kept"]) if acc else NAN,
            "n_djx_left_out_by_selection": len(acc["left_out"]) if acc else NAN, "djx_left_out_by_selection": " ".join(f"{t:.3f}" for t in acc["left_out"]) if acc else "",
        })  # fmt: skip
    b.table(
        [
            "date",
            "status",
            "clipped at λ = 0",
            "clipped at the cap",
            "LC/CC",
            "LC/copula",
            "CC/copula",
            "the gap: index target at the forward minus study at the spot (vp)",
        ],
        rows,
    )
    b.text(
        f"The same {len(a)} dates: the checks with what fails, and the DJX slices behind the one-year index target. `check_index` is not waived on these dates (the wing does not bind): its two numbers are printed, each with the Monte Carlo error of the date, against the gate's {GATE_VP} vp (`tables/unflagged_dates.csv` has both tables). "
        + (
            f"The slices are from {SPEC_NOTE}: the listed DJX expiries the quote screen dropped, those the calendar repair dropped, those kept, and the kept slices nearest the horizon; the last column is the target with the repair minus the target with `screen.calendar_repair` off."
            if spec_i is not None
            else "The slices need the specification-only build (`--spec-build`), which was not run: the counts are the rows'."
        )
    )
    b.table(
        [
            "date",
            "status",
            "LC/CC",
            "check_no_nan",
            "check_forward",
            "check_index",
            "index error at the money (vp)",
            "index error at the 90 % strike (vp)",
            "check_names (diagnostic)",
            "what fails",
            "unscreened (decision 2)",
            "names beyond last kept expiry",
            "DJX slices dropped by the repair",
            "DJX slices dropped by the screen",
            "DJX slices kept",
            "nearest kept DJX slice below / above the horizon",
            "the one-year index target is",
            "target with the repair minus without, at the money (vp)",
        ],
        rows2,
    )
    b.save("unflagged_dates", pd.DataFrame(un_rows))
    b.text(
        f"Of these {len(a)} dates, {int((a['date'] < '2009').sum())} are in 2008, {int(a['flag_unscreened'].sum())} carries decision 2's flag, {int((a['n_names_extrapolated'] > 0).sum())} have a name priced beyond its last kept expiry, {int((~truth(a['check_forward'])).sum())} FAILS `check_forward` and {int((~truth(a['check_names'])).sum())} FAIL the names' diagnostic. The largest of their masses is {fmt(a['clip_larger'].max())} on {a.loc[a['clip_larger'].idxmax(), 'date']}, against the flag's 0.01. {len(a)} dates, {int((a['date'] < '2009').sum())} of them in 2008, are not a cross-section of the history."
    )
    if spec_i is not None:
        built = {x: s for x, s in sides.items() if s["built"]}
        if len(built) != len(a):
            raise ValueError("an unflagged date has no specification-only build")
        extr = [x for x, s in built.items() if s["extrapolated"]]
        wide = [x for x, s in built.items() if s["wide"]]
        rest = [x for x in built if x not in extr and x not in wide]
        screened = [x for x, s in built.items() if s["screen"]]
        repaired = [x for x, s in built.items() if s["repair"]]

        left = [x for x, s in built.items() if s["account"]["left_out"]]
        n_beyond = next(iter(built.values()))["account"]["n_beyond"]
        b.checks.append(
            f"On the {len(built)} unflagged dates every DJX slice the specification-only build keeps or drops is an expiry the inputs list (the pricer's own loader, `lcm_diagnostics.load_inputs`), once, and every listed expiry that is neither is left out by a rule of the surface's selection (beyond the {n_beyond} kept after the horizon, or shorter than its minimum maturity): kept + dropped + left out = listed on each date."
        )

        def listed(x: str) -> int:
            """The DJX expiries the inputs list on a date (:func:`listed_index_expiries`)."""
            return len(built[x]["account"]["listed"])

        def left_text(x: str) -> str:
            acc = built[x]["account"]
            why = [
                *([f"{slice_list(acc['beyond'])}, beyond the {count(len(acc['kept_after']), 'slice', 'slices')} kept after the horizon ({slice_list(acc['kept_after'])})"] if acc["beyond"] else []),
                *([f"{slice_list(acc['short'])}, shorter than the selection's minimum of {acc['min_slice_t']:.3f}y"] if acc["short"] else []),
            ]  # fmt: skip
            return f"{x}: {len(acc['left_out'])} of the {listed(x)} listed DJX expiries, at {'; at '.join(why)}"

        def account_text(x: str) -> str:
            s = built[x]
            return f"{x}: {len(s['kept'])} + {len(s['screen'])} + {len(s['repair'])} + {len(s['account']['left_out'])} = {listed(x)}"

        def extr_text(x: str) -> str:
            s = built[x]
            if not finite(s["below"]):
                return f"{x}: no kept DJX slice at or below the horizon of {s['horizon']:.2f}y, the first kept slice is at {s['above']:.2f}y, and the target is that slice's smile carried back to the horizon, the same implied vol at each log-forward-moneyness (total variance proportional to time: how the model's surface reads a maturity before its first slice; the row's `index_extrapolated` marks a horizon beyond the last kept slice only and reads {yes(a.set_index('date').loc[x, 'index_extrapolated'])} here)"
            return f"{x}: no kept DJX slice above the horizon of {s['horizon']:.2f}y, the last kept slice is at {s['below']:.2f}y, and the target is that slice's smile carried forward to the horizon, the same implied vol at each log-forward-moneyness (total variance proportional to time)"

        def wide_text(x: str) -> str:
            s = built[x]
            moved = (
                f", and the repair moves the target at the money by {fmt(s['repair_effect'], 2, True)} vp"
                if s["repair"] and finite(s["repair_effect"])
                else ""
            )
            return f"{x}: between the kept slices at {s['below']:.2f}y and {s['above']:.2f}y, {s['width']:.2f}y apart{moved}"

        for key, label, xs in (
            (
                "extrapolated",
                "the one-year index target is extrapolated (no kept DJX slice on one side of the horizon)",
                extr,
            ),
            (
                "wide",
                f"the one-year index target is interpolated between kept DJX slices more than {BRACKET_YEAR:g} year apart",
                wide,
            ),
            ("screened", "the quote screen drops a DJX slice", screened),
        ):
            b.add(
                f"clip.unflagged.n_{key}",
                f"unflagged dates on which {label}",
                len(xs),
                unit="dates",
                n=len(a),
                notes="a count",
                source="tables/spec_by_date.csv",
            )
        widths = [built[x]["width"] for x in rest]
        # the caveats of each date that bear on its inputs or its gates (the names' diagnostic is not a gate)
        caveats = {
            r["date"]: [
                *(["decision 2's flag"] if r["flag_unscreened"] else []),
                *([f"status check ({', '.join(g for g in GATING if not (r[g] is True or r[g] == 1.0))} FAIL)"] if r["status"] != "ok" else []),
                *([count(int(r["n_names_extrapolated"]), "name", "names") + " beyond the last kept expiry"] if r["n_names_extrapolated"] > 0 else []),
                *(["target extrapolated"] if r["date"] in extr else []),
                *([f"target interpolated across {built[r['date']]['width']:.2f}y"] if r["date"] in wide else []),
            ]
            for _, r in a.iterrows()
        }  # fmt: skip
        ratio = a.set_index("date")["lc_over_cc"]
        names_fail = set(a.loc[~truth(a["check_names"]), "date"])
        clean = [x for x in ratio.index if not caveats[x]]
        marked = [x for x in ratio.index if caveats[x]]

        def screen_text(x: str) -> str:
            return f"{x}: {len(built[x]['screen'])} of the {listed(x)} listed DJX expiries, at {slice_list(built[x]['screen'])}"

        def repair_text(x: str) -> str:
            return f"{x}: {slice_list(built[x]['repair'])}"

        def clean_text(x: str) -> str:
            return f"{fmt(ratio[x], 5)} ({x}{', which FAILS `check_names`, a diagnostic' if x in names_fail else ''})"

        def marked_text(x: str) -> str:
            return f"{fmt(ratio[x], 5)} ({x}: {', '.join(caveats[x])})"

        b.text(
            f"The index side of these {len(a)} dates. The quote screen drops DJX slices on {len(screened)} of them"
            + (f" ({'; '.join(screen_text(x) for x in screened)})" if screened else "")
            + f" and the calendar repair on {len(repaired)}"
            + (f" ({'; '.join(repair_text(x) for x in repaired)})" if repaired else "")
            + f"; the surface's selection, which takes, of the expiries the screen and the repair leave, those up to the horizon and the next {n_beyond} after it, leaves a listed expiry out on {len(left)}"
            + (
                f" ({'; '.join(left_text(x) for x in left)}), and an expiry it leaves out is neither kept nor reported as dropped"
                if left
                else ""
            )
            + f". On each of the {len(a)} dates kept + dropped by the screen + dropped by the repair + left out by the selection is the number of DJX expiries the inputs list, read from the inputs with the pricer's own loader ({'; '.join(account_text(x) for x in built)})"
            + f". The one-year index target is extrapolated on {len(extr)} of the {len(a)}"
            + (f" — {'; '.join(extr_text(x) for x in extr)} —" if extr else "")
            + f" and interpolated across more than {BRACKET_YEAR:g} year on {len(wide)}"
            + (f" — {'; '.join(wide_text(x) for x in wide)}" if wide else "")
            + (
                f"; on the other {len(rest)} the two kept slices nearest the horizon are {min(widths):.2f} to {max(widths):.2f}y apart."
                if rest
                else "."
            ),
            "LC/CC beside the caveats of each date that bear on its inputs or its gates — decision 2's flag, status check, a name priced beyond its last kept expiry, a one-year target extrapolated or interpolated across more than one year: "
            + (
                f"the {count(len(clean), 'date that carries', 'dates that carry')} none of them {'has' if len(clean) == 1 else 'have'} {listing([clean_text(x) for x in clean])}; "
                if clean
                else "every date carries at least one; "
            )
            + (
                f"the {count(len(marked), 'date that carries', 'dates that carry')} at least one {'has' if len(marked) == 1 else 'have'} {listing([marked_text(x) for x in marked])}. "
                if marked
                else "no date carries one. "
            )
            + f"This is a listing of {len(a)} rows: on {len(a)} dates nothing is said about a sign, an ordering or a cause.",
        )
    # terciles: cut at the 1/3 and 2/3 quantiles of the sample's own larger clipped mass
    rows, long_rows = [], []
    ter_means: dict[str, list[float]] = {}
    ter_cut: dict[str, dict[str, Any]] = {}
    ter = pd.Series(dtype=object)
    for skey, slabel in (
        ("b", f"(b) [{INCL}]"),
        ("c", f"(c), without decision 2's flagged dates [{INCL}]"),
    ):
        inside = p[p[f"sample_{skey}"]]
        which, edges = pd.qcut(
            inside["clip_larger"], 3, labels=["low", "middle", "high"], retbins=True
        )
        ter_cut[skey] = {
            "edges": [float(edges[1]), float(edges[2])],
            "n": [int((which == x).sum()) for x in ("low", "middle", "high")],
            "n_all": len(inside),
        }
        ter_means[skey] = []
        if skey == "b":
            ter = which
        for name in ("low", "middle", "high"):
            sub = inside[which == name]
            cells = [
                slabel,
                name,
                str(len(sub)),
                f"{sub['clip_larger'].min():.4f} to {sub['clip_larger'].max():.4f}",
            ]
            for key in ("lc_over_cc", "lc_over_copula", "cc_over_copula"):
                s = summarise(sub[key], sub[f"{key}_se"])
                cells.append(pm(s["mean"], s["se"]))
                if key == "lc_over_cc":
                    ter_means[skey].append(s["mean"])
                b.add(
                    f"clip.tercile{'' if skey == 'b' else '_c'}.{name}.{key}",
                    f"{key}: mean over the {name} tercile of the larger clipped mass"
                    + (
                        ""
                        if skey == "b"
                        else ", the terciles cut on sample (c), without decision 2's flagged dates"
                    )
                    + f" [{INCL}]",
                    s["mean"],
                    s["se"],
                    unit="ratio",
                    n=int(s["n"]),
                    notes="standard error across dates, sd/√n",
                )
                long_rows.append(
                    {
                        "tercile": name,
                        "quantity": key,
                        "clip_from": sub["clip_larger"].min(),
                        "clip_to": sub["clip_larger"].max(),
                        **s,
                        "sample": skey,
                    }
                )
            rows.append(cells)
    b.save("clip_terciles", pd.DataFrame(long_rows))
    cut_b, cut_c = ter_cut["b"], ter_cut["c"]
    b.text(
        f"**The ratios by tercile of the larger clipped mass** (mean ± across dates; sample (b) [{INCL}] and, cut again on its own dates, sample (c), without decision 2's flagged dates [{INCL}]). "
        f"The terciles are cut at the 1/3 and 2/3 quantiles of the larger clipped mass over the dates of the sample, a date whose mass equals an edge going to the lower tercile (`pandas.qcut`): the edges are {cut_b['edges'][0]:.4f} and {cut_b['edges'][1]:.4f} on the {cut_b['n_all']} dates of sample (b), which gives {listing([str(x) for x in cut_b['n']])} dates, and {cut_c['edges'][0]:.4f} and {cut_c['edges'][1]:.4f} on the {cut_c['n_all']} dates of sample (c), which gives {listing([str(x) for x in cut_c['n']])}."
    )
    b.table(
        ["sample", "tercile", "n", "larger clipped mass", "LC/CC", "LC/copula", "CC/copula"], rows
    )
    # regressions
    near_one = sorted(p.loc[p["clip_larger"] >= CLIP_NEAR_ONE, "date"])
    fits: dict[str, Any] = {}
    reg_rows, md_rows = [], []
    subsets = (
        ("b", f"(b) all priced dates [{INCL}]", p),
        ("c", f"(c) without decision 2's flagged dates [{INCL}]", p[p["sample_c"]]),
        (
            "c_trim",
            f"(c) without the dates with a clipped mass of {CLIP_NEAR_ONE:g} or more ({dates_text(near_one)}) [{INCL}]",
            p[p["sample_c"] & (p["clip_larger"] < CLIP_NEAR_ONE)],
        ),
    )
    for skey, slabel, sub in subsets:
        for key in ("lc_over_cc", "lc_over_copula", "cc_over_copula"):
            f = ols_fit(
                sub[key],
                sub[["clip_low_inner_max", "clip_high_inner_max"]],
                ["clipped at λ = 0", "clipped at the cap"],
            )
            fits[(skey, key)] = f
            for j, term in enumerate(f["names"]):
                reg_rows.append(
                    {
                        "sample": skey,
                        "sample_label": slabel,
                        "regressand": key,
                        "term": term,
                        "coef": f["coef"][j],
                        "se_hc1": f["se_hc1"][j],
                        f"se_nw_{NW_LAGS}": f["se_nw"][j],
                        "r2": f["r2"],
                        "n": f["n"],
                    }
                )
                if j:
                    b.add(
                        f"clip.reg.{skey}.{key}.{'low' if j == 1 else 'cap'}",
                        f"{key} on the two clipped masses, coefficient of the mass {term}, {slabel}",
                        f["coef"][j],
                        f["se_hc1"][j],
                        unit="ratio per unit of mass",
                        n=f["n"],
                        notes=f"the ± is the HC1 error; Newey–West ({NW_LAGS} lags) {f['se_nw'][j]:.4f}; R² {f['r2']:.3f}",
                    )
            md_rows.append(
                [
                    slabel,
                    key.replace("_over_", "/").replace("lc", "LC").replace("cc", "CC"),
                    str(f["n"]),
                    f"{fmt(f['coef'][0])} (HC1 {fmt(f['se_hc1'][0])}, Newey–West {fmt(f['se_nw'][0])})",
                    f"{fmt(f['coef'][1], 4, True)} (HC1 {fmt(f['se_hc1'][1])}, Newey–West {fmt(f['se_nw'][1])})",
                    f"{fmt(f['coef'][2], 4, True)} (HC1 {fmt(f['se_hc1'][2])}, Newey–West {fmt(f['se_nw'][2])})",
                    fmt(f["r2"], 3),
                ]
            )
    b.save("clip_regression", pd.DataFrame(reg_rows))
    b.text(
        f"**A least-squares fit of each ratio on the two clipped masses** (an intercept and the two masses; HC1 and Newey–West errors) [{INCL}]:"
    )
    b.table(
        [
            "sample",
            "regressand",
            "n",
            "intercept",
            "coefficient of the mass at λ = 0",
            "coefficient of the mass at the cap",
            "R²",
        ],
        md_rows,
    )
    f_b, f_t = fits[("b", "lc_over_cc")], fits[("c_trim", "lc_over_cc")]
    f_cc = fits[("c_trim", "cc_over_copula")]
    trim = p[p["sample_c"] & (p["clip_larger"] < CLIP_NEAR_ONE)]
    f_gap = ols_fit(
        trim["cc_over_copula"],
        trim[["clip_low_inner_max", "clip_high_inner_max", "atm_gap"]],
        ["clipped at λ = 0", "clipped at the cap", "gap"],
    )
    b.add(
        "clip.reg.c_trim.cc_over_copula.cap_with_gap",
        f"cc_over_copula on the two clipped masses and the gap of section 4 (i), coefficient of the mass at the cap, sample (c) without the dates with a clipped mass of {CLIP_NEAR_ONE:g} or more [{INCL}]",
        f_gap["coef"][2],
        f_gap["se_hc1"][2],
        unit="ratio per unit of mass",
        n=f_gap["n"],
        notes=f"the ± is the HC1 error; Newey–West ({NW_LAGS} lags) {f_gap['se_nw'][2]:.4f}; R² {f_gap['r2']:.3f}",
    )
    cap_alone = Rules.slope(f_cc["coef"][2], f_cc["se_hc1"][2], f_cc["se_nw"][2])
    cap_with_gap = Rules.slope(f_gap["coef"][2], f_gap["se_hc1"][2], f_gap["se_nw"][2])
    through_target = not cap_alone.startswith("not") and cap_with_gap.startswith("not")
    ter_low = p[ter == "low"]["clip_larger"]
    b.text(
        f"What the straight line over all dates shows [{INCL}]. For LC/CC the coefficient of the mass at the cap is {fmt(f_b['coef'][2], 4, True)} on sample (b) — {Rules.slope(f_b['coef'][2], f_b['se_hc1'][2], f_b['se_nw'][2])} — and {fmt(f_t['coef'][2], 4, True)} without decision 2's dates and without the {count(len(near_one), 'date', 'dates')} with a clipped mass of {CLIP_NEAR_ONE:g} or more — {Rules.slope(f_t['coef'][2], f_t['se_hc1'][2], f_t['se_nw'][2])}; "
        f"the coefficient of the mass at λ = 0 is {fmt(f_t['coef'][1], 4, True)} on that last sample — {Rules.slope(f_t['coef'][1], f_t['se_hc1'][1], f_t['se_nw'][1])}. R² is {fmt(f_b['r2'], 3)} and {fmt(f_t['r2'], 3)}, and from the low to the high tercile the means of LC/CC {Rules.ordering(ter_means['b'], 'tercile')} on sample (b) ({', '.join(fmt(x) for x in ter_means['b'])}) and {Rules.ordering(ter_means['c'], 'tercile')} on sample (c) ({', '.join(fmt(x) for x in ter_means['c'])}). "
        f"That is a statement about a straight line over the whole range and about terciles whose lowest one runs from {fmt(ter_low.min())} to {fmt(ter_low.max())}, {ter_low.max() / CLIP_FLAG:.0f} times the flag's 0.01: the next table looks below a mass of {CLIP_TURN:.2f}. "
        f"For CC/copula the coefficient of the mass at the cap is {fmt(f_cc['coef'][2], 4, True)} on the same sample — {cap_alone}. CC has no λ to clip; with the gap of section 4 (i) added as a third regressor that coefficient is {fmt(f_gap['coef'][2], 4, True)} (HC1 {fmt(f_gap['se_hc1'][2])}, Newey–West {fmt(f_gap['se_nw'][2])}; n {f_gap['n']}) — {cap_with_gap}: "
        + (
            "the relation of CC/copula with the clipped mass runs through the index target (section 4 (i))."
            if through_target
            else "adding the gap does not settle whether the relation of CC/copula with the clipped mass runs through the index target (section 4 (i))."
        )
    )
    # --- LC/CC by bin of the larger clipped mass, and the fits restricted below the turn
    cuts = [-math.inf, *CLIP_BINS, math.inf]
    bin_labels = [
        f"up to {CLIP_BINS[0]:.2f}",
        *(f"{lo:.2f} to {hi:.2f}" for lo, hi in itertools.pairwise(CLIP_BINS)),
        f"above {CLIP_BINS[-1]:.2f}",
    ]
    which = pd.cut(p["clip_larger"], cuts, labels=bin_labels)
    if set(p.loc[which == bin_labels[0], "date"]) != set(p.loc[p["sample_a"], "date"]):
        raise ValueError("the first bin of the larger clipped mass is not sample (a)")
    bins: list[dict[str, Any]] = []
    for i, lab in enumerate(bin_labels):
        sub = p[which == lab]
        s_b, s_c = summarise(sub["lc_over_cc"]), summarise(sub.loc[sub["sample_c"], "lc_over_cc"])
        bins.append({
            "bin": lab, "n_b": int(s_b["n"]), "mean_b": s_b.get("mean", NAN), "se_b": s_b.get("se", NAN), "median_b": s_b.get("median", NAN),
            "n_c": int(s_c["n"]), "mean_c": s_c.get("mean", NAN), "se_c": s_c.get("se", NAN), "median_c": s_c.get("median", NAN), "n_flagged_decision_5": int(sub["flag_clip"].sum()),
            "years": year_spans(sub["date"]), "n_years": int(sub["date"].str[:4].nunique()),
        })  # fmt: skip
        for skey, s in (("b", s_b), ("c", s_c)):
            b.add(
                f"clip.bin.{i}.lc_over_cc.{skey}",
                f"E_LC[D] / E_CC[D]: mean over the priced dates with a larger clipped mass {lab}, sample ({skey})"
                + ("" if i == 0 else f" [{INCL}]"),
                s.get("mean"),
                s.get("se"),
                unit="ratio",
                n=int(s["n"]),
                notes=(
                    "standard error across dates, sd/√n"
                    if finite(s.get("se"))
                    else no_se_reason(int(s["n"]))
                ),
            )
    b.save("clip_bins", pd.DataFrame(bins))
    b.text(
        f"**LC/CC by bin of the larger clipped mass.** The first bin is sample (a), the rule of decision 5, and beside it sample (a2); every other bin is made of flagged rows [{INCL}]. Mean, with its ± across dates when the bin has {MIN_N_SE} dates or more; the last column gives the calendar years of the bin's dates, which {'are' if len({x['years'] for x in bins}) == 1 else 'are not'} the same from bin to bin:"
    )
    b.table(
        [
            "larger clipped mass",
            "dates flagged by decision 5",
            "n",
            "LC/CC, mean",
            "LC/CC, median",
            "n without decision 2's flagged dates",
            "LC/CC, mean without them",
            "years of the bin's dates",
        ],
        [
            [
                x["bin"],
                f"{x['n_flagged_decision_5']} of {x['n_b']}",
                str(x["n_b"]),
                pm(x["mean_b"], x["se_b"]) if finite(x["se_b"]) else f"{fmt(x['mean_b'])} (no error: {x['n_b']} dates)",
                fmt(x["median_b"]),
                str(x["n_c"]),
                pm(x["mean_c"], x["se_c"]) if finite(x["se_c"]) else f"{fmt(x['mean_c'])} (no error: {x['n_c']} dates)",
                f"{x['years']} ({count(x['n_years'], 'year', 'years')})",
            ]
            for x in bins
        ],
    )  # fmt: skip
    restricted: dict[tuple[str, str], dict[str, Any]] = {}
    res_rows, res_md = [], []
    both_samples = (("b", "sample (b)", p["sample_b"]), ("c", "sample (c)", p["sample_c"]))
    trimmed = (
        "c_trim",
        f"sample (c) without the dates with a clipped mass of {CLIP_NEAR_ONE:g} or more",
        p["sample_c"] & (p["clip_larger"] < CLIP_NEAR_ONE),
    )
    cap0 = CLIP_RESTRICT[0]
    # (key, label, dates, samples, a dummy for each year but the first)
    groups: list[tuple[str, str, pd.Series, tuple[Any, ...], bool]] = [
        (
            f"le_{cap0:g}",
            f"larger clipped mass of at most {cap0:.2f}",
            p["clip_larger"] <= cap0,
            both_samples,
            False,
        ),
        (
            f"le_{cap0:g}_flagged",
            f"larger clipped mass of at most {cap0:.2f}, flagged dates only",
            (p["clip_larger"] <= cap0) & p["flag_clip"],
            both_samples,
            False,
        ),
        (
            f"le_{cap0:g}_years",
            f"larger clipped mass of at most {cap0:.2f}, with a dummy for each year",
            p["clip_larger"] <= cap0,
            both_samples,
            True,
        ),
        *(
            (
                f"le_{cap:g}",
                f"larger clipped mass of at most {cap:.2f}",
                p["clip_larger"] <= cap,
                both_samples,
                False,
            )
            for cap in CLIP_RESTRICT[1:]
        ),
        (
            f"gt_{CLIP_TURN:g}",
            f"larger clipped mass above {CLIP_TURN:.2f}",
            p["clip_larger"] > CLIP_TURN,
            (*both_samples, trimmed),
            False,
        ),
        ("all", "every larger clipped mass", p["priced"], (*both_samples, trimmed), False),
    ]
    for gkey, glabel, mask, samples, dummies in groups:
        for skey, slabel, in_sample in samples:
            sub = p[mask & in_sample]
            regressors, names = sub[["clip_larger"]].to_numpy(dtype=float), ["larger clipped mass"]
            sub_years = sorted(set(sub["date"].str[:4]))
            if dummies:
                regressors = np.column_stack(
                    [
                        regressors,
                        *((sub["date"].str[:4] == y).to_numpy(dtype=float) for y in sub_years[1:]),
                    ]
                )
                names += [f"year {y}" for y in sub_years[1:]]
            f = ols_fit(sub["lc_over_cc"], regressors, names)
            # with the dummies the intercept is the first year's level and a rank correlation
            # of the raw values is not the fit's: neither is carried
            rank_corr = NAN if dummies else float(sub["lc_over_cc"].corr(sub["clip_larger"], method="spearman"))  # fmt: skip
            verdict = Rules.slope(f["coef"][1], f["se_hc1"][1], f["se_nw"][1])
            restricted[(gkey, skey)] = {**f, "spearman": rank_corr, "verdict": verdict, "n_years": len(sub_years)}  # fmt: skip
            b.add(
                f"clip.restricted.{gkey}.{skey}.slope",
                f"E_LC[D] / E_CC[D] on the larger clipped mass, slope, {slabel}, {glabel} [{INCL}]",
                f["coef"][1],
                f["se_hc1"][1],
                unit="ratio per unit of mass",
                n=f["n"],
                notes=f"the ± is the HC1 error; {nw_note(f, 1)}; R² {f['r2']:.3f}"
                + (
                    f"; {len(sub_years) - 1} year dummies, {len(sub_years)} years"
                    if dummies
                    else f"; Spearman rank correlation {rank_corr:+.2f}"
                ),
            )
            res_rows.append({
                "restriction": glabel, "sample": skey, "n": f["n"], "intercept": NAN if dummies else f["coef"][0], "intercept_se_hc1": NAN if dummies else f["se_hc1"][0], f"intercept_se_nw_{NW_LAGS}": NAN if dummies else f["se_nw"][0],
                "slope": f["coef"][1], "slope_se_hc1": f["se_hc1"][1], f"slope_se_nw_{NW_LAGS}": f["se_nw"][1], "r2": f["r2"], "spearman": rank_corr, "verdict": verdict,
                "slope_over_larger_error": Rules.slope_errors(f["coef"][1], f["se_hc1"][1], f["se_nw"][1]), "year_dummies": dummies, "n_years": len(sub_years),
            })  # fmt: skip
            res_md.append([
                f"{glabel}, {slabel}", str(f["n"]), f"a level per year ({len(sub_years)} years)" if dummies else f"{fmt(f['coef'][0])} ({fit_errs(f, 0)})",
                f"{fmt(f['coef'][1], 4, True)} ({fit_errs(f, 1)})", fmt(f["r2"], 3), "none (not the fit's)" if dummies else fmt(rank_corr, 2, True), verdict,
            ])  # fmt: skip
    b.save("clip_restricted_fits", pd.DataFrame(res_rows))
    b.text(
        f"**A least-squares fit of LC/CC on the larger clipped mass, restricted** (an intercept and the larger of the two masses; HC1 errors, and Newey–West errors when the fit has {MIN_N_NW} dates or more, the lags counted in consecutive dates of the subsample, which a selection on the mass can leave months or years apart; the rank correlation is Spearman's and carries no error) [{INCL}]. "
        f"Two rows look at what the fit at or below {cap0:.2f} is made of: \"flagged dates only\" leaves out the {int(p['sample_a'].sum())} unflagged dates, and \"with a dummy for each year\" adds an indicator of each calendar year but the first to the regressors, so that the slope is read within the year (`tables/clip_restricted_fits.csv`):"
    )
    b.table(
        [
            "dates",
            "n",
            "intercept",
            "slope",
            "R²",
            "rank correlation",
            "the slope, by the rule of section 0",
        ],
        res_md,
    )
    # the unflagged dates among all priced dates
    rank = p["lc_over_cc"].rank(ascending=False, method="min").astype(int)
    a_rank = sorted(int(x) for x in rank[p["sample_a"]])
    year = p["date"].str[:4]
    within, within_rows = [], []
    for y in sorted(set(year[p["sample_a"]])):
        cells = {}
        for skey in ("b", "c"):
            inside = p[(year == y) & p[f"sample_{skey}"]]
            un, fl = (
                inside.loc[inside["sample_a"], "lc_over_cc"],
                inside.loc[~inside["sample_a"], "lc_over_cc"],
            )
            cells[skey] = (len(un), float(un.mean()) if len(un) else NAN, len(fl), float(fl.mean()) if len(fl) else NAN)  # fmt: skip
        within_rows.append({
            "year": y, "n_unflagged": cells["b"][0], "mean_unflagged": cells["b"][1], "n_flagged": cells["b"][2], "mean_flagged": cells["b"][3],
            "n_unflagged_without_decision_2": cells["c"][0], "mean_unflagged_without_decision_2": cells["c"][1], "n_flagged_without_decision_2": cells["c"][2], "mean_flagged_without_decision_2": cells["c"][3],
        })  # fmt: skip
        if cells["b"][2]:
            both = f"{fmt(cells['b'][1])} (n {cells['b'][0]}) against {fmt(cells['b'][3])} (n {cells['b'][2]}) in {y}"
            if cells["c"] != cells["b"] and cells["c"][0] and cells["c"][2]:
                both += f", and without decision 2's flagged dates {fmt(cells['c'][1])} (n {cells['c'][0]}) against {fmt(cells['c'][3])} (n {cells['c'][2]})"
            within.append((cells["b"][1] > cells["b"][3], both))
    b.save("clip_unflagged_within_year", pd.DataFrame(within_rows))
    for i, x in enumerate(a_rank):
        b.add(
            f"clip.unflagged_rank.{i}",
            "rank by E_LC[D] / E_CC[D], the highest first, of an unflagged date among the priced dates",
            x,
            unit="rank",
            n=n_p,
            notes="a rank: no standard error",
        )
    r20b, r20c = restricted[(f"le_{CLIP_RESTRICT[0]:g}", "b")], restricted[(f"le_{CLIP_RESTRICT[0]:g}", "c")]  # fmt: skip
    r10b = restricted[(f"le_{CLIP_RESTRICT[1]:g}", "b")]
    below, last = bins[:-1], bins[-1]
    means = [x["mean_b"] for x in below]
    falls = all(y < x for x, y in itertools.pairwise(means))
    falls = falls and all(y < x for x, y in itertools.pairwise([x["mean_c"] for x in below]))
    step = last["mean_b"] - below[-1]["mean_b"]
    step_err = math.hypot(last["se_b"], below[-1]["se_b"])
    turns = finite(step_err) and step > K_SIGMA * step_err
    whole, whole_trim = restricted[("all", "b")], restricted[("all", "c_trim")]
    above = restricted[(f"gt_{CLIP_TURN:g}", "b")]
    above_trim = restricted[(f"gt_{CLIP_TURN:g}", "c_trim")]
    negative = r20b["verdict"].startswith("negative") and r20c["verdict"].startswith("negative")
    closer = "closer to 1" if all(x["mean_b"] < 1.0 for x in bins) else "higher"
    cap20, cap10 = f"{CLIP_RESTRICT[0]:.2f}", f"{CLIP_RESTRICT[1]:.2f}"
    turn_at = f"{CLIP_TURN:.2f}"
    if negative and falls:
        relation = f"LC/CC is {closer} where the clipped mass is smaller, up to a mass of about {turn_at}: on sample (b) and on sample (c) its mean falls from bin to bin, on sample (b) from {fmt(means[0])} on the {below[0]['n_b']} unflagged dates to {fmt(means[-1])} on the {below[-1]['n_b']} dates with a mass of {below[-1]['bin']}, and on the {r20b['n']} dates with a mass of at most {cap20} the slope is {fmt(r20b['coef'][1], 3, True)} ({fit_errs(r20b, 1, 3)}; rank correlation {fmt(r20b['spearman'], 2, True)}) — {r20b['verdict']} — and {fmt(r20c['coef'][1], 3, True)} ({fit_errs(r20c, 1, 3, True)}) without decision 2's flagged dates — {r20c['verdict']}"
    else:
        relation = f"below a mass of {turn_at} a relation between LC/CC and the clipped mass is not established by the rule: the slope on the {r20b['n']} dates with a mass of at most {cap20} is {fmt(r20b['coef'][1], 3, True)} ({fit_errs(r20b, 1, 3)}) — {r20b['verdict']} — and {fmt(r20c['coef'][1], 3, True)} without decision 2's flagged dates — {r20c['verdict']}; the bin means {'fall from bin to bin' if falls else 'are not ordered'}"
    step_text = f"the mean is {pm(last['mean_b'], last['se_b'])} on the {last['n_b']} dates above {turn_at} against {pm(below[-1]['mean_b'], below[-1]['se_b'])} on the {below[-1]['n_b']} dates just below, a step of {fmt(step, 4, True)} for a combined across-dates error of {fmt(step_err)}, and on the dates above {turn_at} the slope is {fmt(above['coef'][1], 3, True)} ({fit_errs(above, 1, 3, True)}) — {above['verdict']} — and {fmt(above_trim['coef'][1], 3, True)} ({fit_errs(above_trim, 1, 3, True)}) without decision 2's flagged dates and without the {count(len(near_one), 'date', 'dates')} with a mass of {CLIP_NEAR_ONE:g} or more — {above_trim['verdict']}"
    turn = (
        f"A straight line in the larger mass over all dates does not show it — its slope is {fmt(whole['coef'][1], 4, True)} ({fit_errs(whole, 1, 4, True)}), {whole['verdict']}, and {fmt(whole_trim['coef'][1], 4, True)} ({fit_errs(whole_trim, 1, 4, True)}) without decision 2's flagged dates and without the {count(len(near_one), 'date', 'dates')} with a mass of {CLIP_NEAR_ONE:g} or more, {whole_trim['verdict']} — and the relation turns above {turn_at}: {step_text}."
        if turns and negative and falls
        else f"Above {turn_at} a turn is not established by the rule: {step_text}."
    )
    n_higher = sum(1 for hi, _ in within if hi)
    sa_mean, sb_mean = float(p.loc[p["sample_a"], "lc_over_cc"].mean()), float(p["lc_over_cc"].mean())  # fmt: skip
    # what the fit at or below the first restriction is made of: the unflagged dates, the years
    fl_b, fl_c = restricted[(f"le_{cap0:g}_flagged", "b")], restricted[(f"le_{cap0:g}_flagged", "c")]  # fmt: skip
    yr_b, yr_c = restricted[(f"le_{cap0:g}_years", "b")], restricted[(f"le_{cap0:g}_years", "c")]
    n_un = int(p["sample_a"].sum())

    def holds(x: dict[str, Any], y: dict[str, Any]) -> str:
        """Whether the two slopes (with and without decision 2's flagged dates) are negative by the rule."""
        neg = [v["verdict"].startswith("negative") for v in (x, y)]
        return "both" if all(neg) else "neither" if not any(neg) else "one"

    made_of = (
        f"Among the flagged dates alone the slope at or below {cap20} is {fmt(fl_b['coef'][1], 3, True)} ({fit_errs(fl_b, 1, 3, True)}; {errs(Rules.slope_errors(fl_b['coef'][1], fl_b['se_hc1'][1], fl_b['se_nw'][1]))} errors) — {fl_b['verdict']} — and {fmt(fl_c['coef'][1], 3, True)} ({fit_errs(fl_c, 1, 3, True)}; {errs(Rules.slope_errors(fl_c['coef'][1], fl_c['se_hc1'][1], fl_c['se_nw'][1]))} errors) without decision 2's flagged dates — {fl_c['verdict']}: "
        + {
            "both": f"the relation below {cap20} is not made by the {n_un} unflagged dates",
            "neither": f"without the {n_un} unflagged dates a relation below {cap20} is not established by the rule",
            "one": f"without the {n_un} unflagged dates the relation below {cap20} holds by the rule on one of the two samples only",
        }[holds(fl_b, fl_c)]
        + f". The bins {'are' if len({x['years'] for x in bins}) == 1 else 'are not'} the same years (the last column of the bins' table: the dates with a mass of at most {CLIP_BINS[2]:.2f} are in {year_spans(p.loc[p['clip_larger'] <= CLIP_BINS[2], 'date'])}, the {below[-1]['n_b']} with a mass of {below[-1]['bin']} in {below[-1]['years']}), and with a dummy for each year the slope at or below {cap20} is {fmt(yr_b['coef'][1], 3, True)} ({fit_errs(yr_b, 1, 3, True)}, {yr_b['n_years']} years; {errs(Rules.slope_errors(yr_b['coef'][1], yr_b['se_hc1'][1], yr_b['se_nw'][1]))} errors) — {yr_b['verdict']} — and {fmt(yr_c['coef'][1], 3, True)} ({fit_errs(yr_c, 1, 3, True)}; {errs(Rules.slope_errors(yr_c['coef'][1], yr_c['se_hc1'][1], yr_c['se_nw'][1]))} errors) without decision 2's flagged dates — {yr_c['verdict']}: "
        + {
            "both": "the relation holds within the year as well",
            "neither": "within the year the relation is not established by the rule, so that a difference between years is not told apart from a relation with the clipped mass",
            "one": "within the year the relation holds by the rule on one of the two samples only",
        }[holds(yr_b, yr_c)]
        + "."
    )
    b.text(
        f"What the bins and the restricted fit show [{INCL}]. By the rule of section 0, {relation}. On the {r10b['n']} dates with a mass of at most {cap10} the slope is {fmt(r10b['coef'][1], 3, True)} ({fit_errs(r10b, 1, 3)}) — {r10b['verdict']}. {made_of} {turn} "
        f"The {len(a_rank)} unflagged dates rank {', '.join(ordinal(x) for x in a_rank[:-1])} and {ordinal(a_rank[-1])} of {n_p} by LC/CC, the highest first; within the year the unflagged dates stand above the flagged ones in {n_higher} of the {len(within)} years that have both: {'; '.join(t for _, t in within)}.",
        f"What none of this shows. The fits and the bins are across dates, on {n_p - int(p['flag_clip'].sum())} unflagged dates against {int(p['flag_clip'].sum())} flagged ones: an association, not a correction. It says nothing about what LC/CC would be on a flagged date if its wing did not bind, and it cannot move a flagged date into the summary of decision 5. The masses have no standard error and the regressors are calibration outputs of the same run as the regressand. "
        f"The two summaries of section 2 differ ({fmt(sa_mean)} on sample (a), {fmt(sb_mean)} on sample (b) [{INCL}]) {'in the direction of this association' if (sa_mean > sb_mean) == (r20b['coef'][1] < 0) else 'against the direction of this association'}; what that means for the open decision is the owner's to judge.",
    )
    return {"near_one": near_one, "fits": fits, "bins": bins, "restricted": restricted, "terciles": ter_means, "made_of": (holds(fl_b, fl_c), holds(yr_b, yr_c))}  # fmt: skip


# ----------------------------------------------------------------------------- section 4
def strip_source_check(commit: str) -> tuple[str, list[str]]:
    """Where the pricer's source at the rows' commit reads the SVI strip sum and its columns:
    every line of ``git show <commit>:scripts/lcm_price.py`` that names one of them."""
    source = git("show", f"{commit}:scripts/lcm_price.py")
    origin = f"`git show {commit}:scripts/lcm_price.py`"
    if not source:
        source, origin = (
            pc.ROOT / "scripts" / "lcm_price.py"
        ).read_text(), "`scripts/lcm_price.py` of the worktree (git could not show the rows' commit)"
    hits = [
        f"{i}: {line.strip()}"
        for i, line in enumerate(source.splitlines(), 1)
        if re.search(r"\bsum_svi\b|" + "|".join(STRIP_COLUMNS), line)
    ]
    return origin, hits


def section_extremes(b: Book, data: Data, clip: dict[str, Any]) -> None:
    p = data.priced
    n_p = len(p)
    spec = read_spec("spec_by_date")
    spec_i = spec.set_index("date") if spec is not None else None
    has_target = bool(p["gap_same_strike"].notna().any())
    norepair_dates = (
        sorted(str(x)[:10] for x in pd.read_parquet(TABLE_NOREPAIR)["date"])
        if TABLE_NOREPAIR.exists()
        else []
    )
    b.text("## 4. Where the extremes come from")
    # --- (i) the index level
    corr = {}
    for skey, sub in (("b", p), ("c", p[p["sample_c"]])):
        r = float(np.corrcoef(sub["atm_gap"], sub["cc_over_copula"])[0, 1])
        f = ols_fit(sub["cc_over_copula"], sub[["atm_gap"]], ["gap"])
        corr[skey] = (r, f, len(sub))
        b.add(
            f"level.corr.{skey}",
            f"correlation across dates between E_CC[D]/copula and (index target at-the-money vol minus the study's), sample ({skey})",
            r,
            unit="correlation",
            n=len(sub),
            notes="a Pearson correlation across dates: no standard error",
        )
        b.add(
            f"level.slope.{skey}",
            f"E_CC[D]/copula on the gap, slope, sample ({skey})",
            f["coef"][1],
            f["se_hc1"][1],
            unit="ratio per vol point",
            n=len(sub),
            notes=f"the ± is the HC1 error; Newey–West ({NW_LAGS} lags) {f['se_nw'][1]:.4f}; R² {f['r2']:.3f}; intercept {f['coef'][0]:.4f}",
        )
    gap = p["atm_gap"]
    for stat, x in (
        ("mean", gap.mean()),
        ("median", gap.median()),
        ("min", gap.min()),
        ("max", gap.max()),
    ):
        b.add(
            f"level.gap.{stat}",
            f"index target at-the-money vol at the horizon minus the study's: {stat} over the priced dates",
            x,
            unit="vol points",
            n=n_p,
            notes="a statistic of two inputs across dates: no standard error",
            source=f"{rel(ROWS)} (index_errors) and {STUDY}",
        )
    p_c = p[p["sample_c"]]
    gap_c = p_c["atm_gap"]
    for stat, x in (
        ("mean", gap_c.mean()),
        ("median", gap_c.median()),
        ("min", gap_c.min()),
        ("max", gap_c.max()),
    ):
        b.add(
            f"level.gap.{stat}.c",
            f"index target at-the-money vol at the horizon minus the study's: {stat} over the priced dates without decision 2's flagged dates [{INCL}]",
            x,
            unit="vol points",
            n=len(p_c),
            notes="a statistic of two inputs across dates: no standard error",
            source=f"{rel(ROWS)} (index_errors) and {STUDY}",
        )
    far = p[gap.abs() > GAP_FAR].sort_values("atm_gap")
    b.add(
        "level.n_far",
        f"priced dates whose index target is more than {GAP_FAR} vol points from the study's level at the money",
        len(far),
        unit="dates",
        n=n_p,
        notes="a count",
    )
    rb, fb, nb = corr["b"]
    rc, fc, nc = corr["c"]
    b.text(
        "**(i) The model's index target against the study's index level.** The model's target is its own SVI surface of the kept DJX slices, after the screen and the calendar repair, read at the horizon at the money, that is at the forward (log-forward-moneyness 0: the `target_vol` of the row's `index_errors`). The study's level is `sig_B_DJX` of its entry: the implied vol at the spot (K = S_0) of the study's DJX marginal at the entry's maturity. "
        f"The gap is the first minus the second, in vol points — a vol at the forward minus a vol at the spot; CC is fitted to the first and the copula to the second. Every statistic, correlation and fit of this subsection is over the priced dates, {int(p['flag_clip'].sum())} of {n_p} of them flagged [{INCL}]; none is made on the {int(p['sample_a'].sum())} dates of sample (a) alone.",
        f"Over the {n_p} priced dates, sample (b) [{INCL}], the gap has a mean of {fmt(gap.mean(), 2, True)} vp and a median of {fmt(gap.median(), 2, True)} vp, from {fmt(gap.min(), 2, True)} to {fmt(gap.max(), 2, True)}, and over the {len(p_c)} dates of sample (c), without decision 2's flagged dates [{INCL}], a mean of {fmt(gap_c.mean(), 2, True)} vp and a median of {fmt(gap_c.median(), 2, True)} vp, from {fmt(gap_c.min(), 2, True)} to {fmt(gap_c.max(), 2, True)} (statistics of two inputs across dates, like the correlations below: no standard error). The correlation across dates between CC/copula and the gap is {fmt(rb, 2, True)} on sample (b) (n {nb}) and {fmt(rc, 2, True)} on sample (c), without decision 2's flagged dates (n {nc}); a least-squares fit on sample (b) gives {fmt(fb['coef'][1], 4, True)} of CC/copula per vol point of gap (HC1 {fmt(fb['se_hc1'][1])}, Newey–West {fmt(fb['se_nw'][1])}; R² {fmt(fb['r2'], 2)}; n {nb}) with an intercept of {fmt(fb['coef'][0])} (HC1 {fmt(fb['se_hc1'][0])}, Newey–West {fmt(fb['se_nw'][0])}): {Rules.slope(fb['coef'][1], fb['se_hc1'][1], fb['se_nw'][1])}; "
        f"on sample (c) {fmt(fc['coef'][1], 4, True)} (HC1 {fmt(fc['se_hc1'][1])}, Newey–West {fmt(fc['se_nw'][1])}; n {nc}): {Rules.slope(fc['coef'][1], fc['se_hc1'][1], fc['se_nw'][1])}. "
        + (
            "By the fit on sample (b) the direction of the relation is not established by the rule. "
            if Rules.slope(fb["coef"][1], fb["se_hc1"][1], fb["se_nw"][1]).startswith("not")
            else f"By the fit on sample (b) a target one vol point above the study's level goes with a CC/copula {'lower' if fb['coef'][1] < 0 else 'higher'} by {abs(fb['coef'][1]):.3f}. "
        )
        + "This is why only LC/CC is read as a local-correlation effect.",
    )
    # what the gap is made of: the strike, and the repair
    fwd = p["study_fwd"]
    r_fwd = float(np.corrcoef(gap, np.log(fwd))[0, 1])
    b.add(
        "level.corr_gap_log_forward",
        f"correlation across dates between the gap and ln(F/S), the study's index forward over the spot (f_B of the entry), sample (b) [{INCL}]",
        r_fwd,
        unit="correlation",
        n=n_p,
        notes="a Pearson correlation across dates: no standard error",
        source=f"{rel(ROWS)} (index_errors) and {STUDY}",
    )
    r_fwd_c = float(np.corrcoef(gap_c, np.log(p_c["study_fwd"]))[0, 1])
    b.add(
        "level.corr_gap_log_forward.c",
        f"correlation across dates between the gap and ln(F/S), the study's index forward over the spot (f_B of the entry), sample (c), without decision 2's flagged dates [{INCL}]",
        r_fwd_c,
        unit="correlation",
        n=len(p_c),
        notes="a Pearson correlation across dates: no standard error",
        source=f"{rel(ROWS)} (index_errors) and {STUDY}",
    )
    year = p["date"].str[:4]
    by_year = (
        pd.DataFrame({"year": year, "gap": gap, "forward_over_spot": fwd})
        .groupby("year")
        .agg(n=("gap", "size"), mean_gap_vp=("gap", "mean"), mean_forward_over_spot=("forward_over_spot", "mean"))
        .reset_index()
    )  # fmt: skip
    b.save("level_gap_by_year", by_year)
    y_lo, y_hi = by_year.loc[by_year["mean_gap_vp"].idxmin()], by_year.loc[by_year["mean_gap_vp"].idxmax()]  # fmt: skip
    made_of = f"**What the gap is made of** [{INCL}]. The two vols are not read at the same strike. Across the {n_p} priced dates the correlation of the gap with ln(F/S), the study's index forward over the spot (`f_B` of the entry), is {fmt(r_fwd, 2, True)} ({fmt(r_fwd_c, 2, True)} on the {len(p_c)} dates of sample (c), without decision 2's flagged dates); by year the mean gap is lowest in {y_lo['year']} ({fmt(y_lo['mean_gap_vp'], 2, True)} vp, mean F/S {fmt(y_lo['mean_forward_over_spot'], 3)}, n {int(y_lo['n'])}) and highest in {y_hi['year']} ({fmt(y_hi['mean_gap_vp'], 2, True)} vp, mean F/S {fmt(y_hi['mean_forward_over_spot'], 3)}, n {int(y_hi['n'])}) (`tables/level_gap_by_year.csv`; {'a yearly mean is over at most ' + str(int(by_year['n'].max())) + ' dates and carries no error' if int(by_year['n'].max()) < MIN_N_SE else 'yearly means without their errors'}). "
    if has_target:
        same = p["gap_same_strike"]
        far_same = p[same.abs() > GAP_FAR].sort_values("gap_same_strike")
        var_left = float(same.var(ddof=1) / gap.var(ddof=1))
        f2 = ols_fit(p["cc_over_copula"], p[["gap_strike_part", "gap_same_strike"]], ["strike part", "same-strike part"])  # fmt: skip
        same_c = p_c["gap_same_strike"]
        var_left_c = float(same_c.var(ddof=1) / gap_c.var(ddof=1))
        for stat, x in (
            ("mean", same_c.mean()),
            ("median", same_c.median()),
            ("min", same_c.min()),
            ("max", same_c.max()),
            ("variance_share", var_left_c),
        ):
            b.add(
                f"level.gap_same_strike.{stat}.c",
                f"index target at the horizon read at the spot strike minus the study's index vol at the spot: {stat.replace('_', ' ')} over the priced dates without decision 2's flagged dates [{INCL}]",
                x,
                unit="ratio" if stat == "variance_share" else "vol points",
                n=len(p_c),
                notes=(
                    "a ratio of two sample variances (the gap read at the same strike over the gap): no standard error"
                    if stat == "variance_share"
                    else "a statistic of two inputs across dates: no standard error"
                ),
                source=f"tables/spec_by_date.csv (target_at_spot_strike) and {STUDY}",
            )
        f2c = ols_fit(p_c["cc_over_copula"], p_c[["gap_strike_part", "gap_same_strike"]], ["strike part", "same-strike part"])  # fmt: skip
        for stat, x in (
            ("mean", same.mean()),
            ("median", same.median()),
            ("min", same.min()),
            ("max", same.max()),
        ):
            b.add(
                f"level.gap_same_strike.{stat}",
                f"index target at the horizon read at the spot strike (log-forward-moneyness −ln(F/S)) minus the study's index vol at the spot: {stat} over the priced dates [{INCL}]",
                x,
                unit="vol points",
                n=n_p,
                notes="a statistic of two inputs across dates: no standard error",
                source=f"tables/spec_by_date.csv (target_at_spot_strike) and {STUDY}",
            )
        b.add(
            "level.gap_same_strike.n_far",
            f"priced dates whose target read at the spot strike is more than {GAP_FAR} vol points from the study's level",
            len(far_same),
            unit="dates",
            n=n_p,
            notes="a count",
            source="tables/spec_by_date.csv",
        )
        b.add(
            "level.gap_same_strike.variance_share",
            f"variance across dates of the gap read at the same strike over the variance of the gap [{INCL}]",
            var_left,
            unit="ratio",
            n=n_p,
            notes="a ratio of two sample variances: no standard error",
            source="tables/spec_by_date.csv",
        )
        for j, term in ((1, "strike_part"), (2, "same_strike_part")):
            for skey, f in (("b", f2), ("c", f2c)):
                b.add(
                    f"level.slope2.{term}.{skey}",
                    f"E_CC[D]/copula on the two parts of the gap, coefficient of the {term.replace('_', ' ')}, sample ({skey}) [{INCL}]",
                    f["coef"][j],
                    f["se_hc1"][j],
                    unit="ratio per vol point",
                    n=f["n"],
                    notes=f"the ± is the HC1 error; Newey–West ({NW_LAGS} lags) {f['se_nw'][j]:.4f}; R² {f['r2']:.3f}",
                    source="tables/spec_by_date.csv",
                )
        v1, v2 = (Rules.slope(f2["coef"][j], f2["se_hc1"][j], f2["se_nw"][j]) for j in (1, 2))
        both = v1.startswith("negative") and v2.startswith("negative")
        made_of += (
            f"Read at the spot strike on the same target surface (log-forward-moneyness −ln(F/S); from {SPEC_NOTE}, whose rebuilt target equals the row's `target_vol` at every strike of the horizon on every priced date), the target minus the study's level has a mean of {fmt(same.mean(), 2, True)} vp and a median of {fmt(same.median(), 2, True)} vp, from {fmt(same.min(), 2, True)} to {fmt(same.max(), 2, True)}; "
            f"it is beyond {GAP_FAR} vp on {count(len(far_same), 'date', 'dates')} ({dates_text(list(far_same['date']))}) against {len(far)} for the gap, and its variance across dates is {100 * var_left:.0f} % of the gap's (on sample (c), without decision 2's flagged dates: mean {fmt(same_c.mean(), 2, True)} vp, median {fmt(same_c.median(), 2, True)} vp, from {fmt(same_c.min(), 2, True)} to {fmt(same_c.max(), 2, True)}, variance {100 * var_left_c:.0f} % of the gap's). "
            f"A least-squares fit of CC/copula on the two parts of the gap — the strike part (target at the forward minus target at the spot strike) and the same-strike part (target at the spot strike minus the study's level) — gives {fmt(f2['coef'][1], 4, True)} per vp of the first (HC1 {fmt(f2['se_hc1'][1])}, Newey–West {fmt(f2['se_nw'][1])}) and {fmt(f2['coef'][2], 4, True)} per vp of the second (HC1 {fmt(f2['se_hc1'][2])}, Newey–West {fmt(f2['se_nw'][2])}), R² {fmt(f2['r2'], 2)}, n {f2['n']}: the first is {v1}, the second {v2}; "
            f"on sample (c) {fmt(f2c['coef'][1], 4, True)} (HC1 {fmt(f2c['se_hc1'][1])}, Newey–West {fmt(f2c['se_nw'][1])}) and {fmt(f2c['coef'][2], 4, True)} (HC1 {fmt(f2c['se_hc1'][2])}, Newey–West {fmt(f2c['se_nw'][2])}), n {f2c['n']}. "
            + (
                "CC/copula follows both parts, so the fit of CC/copula on the gap stands as a description; "
                if both
                else "CC/copula does not follow both parts by the rule; "
            )
            + (
                f"but the gap is for the larger part the distance between the forward and the spot ({100 * (1 - var_left):.0f} % of its variance goes when both vols are read at the spot), not a difference between two readings of the DJX quotes at one strike."
                if var_left < 0.5
                else f"and {100 * var_left:.0f} % of the gap's variance remains when both vols are read at the spot."
            )
        )
    else:
        made_of += "The gap read at one strike and the effect of the calendar repair on the target need the specification-only build (`--spec-build`), which was not run."
    b.text(
        made_of,
        f"The {len(far)} dates whose target is more than {GAP_FAR} vp from the study's level:",
    )
    rows = []
    for _, r in far.iterrows():
        bracket = "not built"
        if spec_i is not None and r["date"] in spec_i.index:
            s = spec_i.loc[r["date"]]
            bracket = f"{years(s['nearest_kept_below'])} / {years(s['nearest_kept_above'])}"
        rows.append([
            r["date"], r["status"], fmt(r["target_atm"], 2), fmt(r["study_atm"], 2), fmt(r["atm_gap"], 2, True), fmt(r["study_fwd"], 3), fmt(r["gap_same_strike"], 2, True), fmt(r["repair_effect_atm_vp"], 2, True),
            pm(r["cc_over_copula"], r["cc_over_copula_se"]), pm(r["lc_over_copula"], r["lc_over_copula_se"]), pm(r["lc_over_cc"], r["lc_over_cc_se"]),
            fmt(r["rho_cc"], 3), fmt(r["rho_cop"], 3), f"{int(r['n_dropped_calendar_index'])} / {int(r['n_dropped_index_screen'])}", bracket, yes(r["index_extrapolated"]), yes(r["flag_clip"]), yes(r["flag_unscreened"]), chk(r["check_no_nan"]),
        ])  # fmt: skip
    b.table(
        [
            "date",
            "status",
            "target at the money, at the forward (%)",
            "study, at the spot (%)",
            "gap (vp)",
            "study's forward over spot",
            "gap read at the spot strike (vp)",
            "target with the repair minus without, at the money (vp)",
            "CC/copula",
            "LC/copula",
            "LC/CC",
            "ρ_CC (no error in the row)",
            "ρ_cop (the study's)",
            "DJX slices dropped: repair / screen",
            "nearest kept DJX slice below / above the horizon",
            "index extrapolated",
            "clip flag (decision 5)",
            "unscreened (decision 2)",
            "check_no_nan",
        ],
        rows,
    )
    b.save(
        "level_gap_far",
        far[
            [
                "date",
                "status",
                "target_atm",
                "study_atm",
                "atm_gap",
                "study_fwd",
                "target_at_spot_strike",
                "gap_same_strike",
                "gap_strike_part",
                "target_atm_norepair",
                "repair_effect_atm_vp",
                "cc_over_copula",
                "cc_over_copula_se",
                "lc_over_copula",
                "lc_over_copula_se",
                "lc_over_cc",
                "lc_over_cc_se",
                "rho_cc",
                "rho_cop",
                "n_dropped_calendar_index",
                "n_dropped_index_screen",
                "index_extrapolated",
                "flag_clip",
                "flag_unscreened",
                "check_no_nan",
                "check_forward",
            ]
        ],
    )
    y2007 = far[far["date"].str.startswith("2007")]
    extrap = p[truth(p["index_extrapolated"])]
    hi, lo = p.loc[p["cc_over_copula"].idxmax()], p.loc[p["cc_over_copula"].idxmin()]
    repair_2007 = ""
    repair_pass = ""
    if has_target:
        eff = y2007["repair_effect_atm_vp"]
        big = y2007[eff.abs() > REPAIR_MOVES[1]]
        rest = eff[eff.abs() <= REPAIR_MOVES[1]]
        repair_2007 = (
            f"the calendar repair, which drops {int(y2007['n_dropped_calendar_index'].min())} to {int(y2007['n_dropped_calendar_index'].max())} DJX slices on each, moves the target at the money by more than {REPAIR_MOVES[1]:g} vp on {len(big)} of them"
            + (f" ({'; '.join(f'{r.date}: {r.repair_effect_atm_vp:+.2f} vp' for r in big.itertuples())})" if len(big) else "")
            + (f" and by {fmt(rest.abs().min(), 2)} to {fmt(rest.abs().max(), 2)} vp in absolute value on the other {len(rest)}" if len(rest) else "")
            + " (target with the repair minus target without it, the specification rebuilt with `screen.calendar_repair` off: a difference of two fitted surfaces, without a Monte Carlo error)"
        )  # fmt: skip
        dropped = p[p["n_dropped_calendar_index"] > 0]
        kept_all = p[p["n_dropped_calendar_index"] == 0]
        e_all = dropped["repair_effect_atm_vp"]
        moved = {t: dropped[e_all.abs() > t] for t in REPAIR_MOVES}
        for t, sub in moved.items():
            b.add(
                f"level.repair.n_moved_{t:g}",
                f"priced dates with a DJX slice dropped by the calendar repair on which the repair moves the index target at the money at the horizon by more than {t:g} vol points [{INCL}]",
                len(sub),
                unit="dates",
                n=len(dropped),
                notes="a count",
                source="tables/spec_by_date.csv (repair_effect_atm_vp)",
            )
        top = moved[REPAIR_MOVES[1]].sort_values("repair_effect_atm_vp", key=lambda x: -x.abs())
        in_a = top[top["sample_a"]]
        in_ref = top[top["date"].isin(pc.REFERENCE_DATES)]
        same_spec = int(truth(kept_all["norepair_index_surface_equal"]).sum())
        repair_pass = (
            f" Over the pass [{INCL}] the repair moves the target at the money by more than {REPAIR_MOVES[0]:g} vp on {len(moved[REPAIR_MOVES[0]])} of the {len(dropped)} dates with a dropped DJX slice and by more than {REPAIR_MOVES[1]:g} vp on {len(top)} ({'; '.join(f'{r.date} {r.repair_effect_atm_vp:+.2f}' for r in top.itertuples())})"
            + (f", among them the unflagged {'date' if len(in_a) == 1 else 'dates'} {dates_text(list(in_a['date']))}" if len(in_a) else "")
            + (f"{' and' if len(in_a) else ', among them'} the reference {'date' if len(in_ref) == 1 else 'dates'} {dates_text(list(in_ref['date']))}" if len(in_ref) else "")
            + f"; on {same_spec} of the other {len(kept_all)} dates the index surface of the specification is the same with and without the repair."
        )  # fmt: skip
    b.text(
        f"The ± are Monte Carlo errors of the date; the rows carry no error for ρ_CC, and ρ_cop is the study's number. {len(y2007)} of the {len(far)} dates are in 2007, with a target {fmt(-y2007['atm_gap'].max(), 1)} to {fmt(-y2007['atm_gap'].min(), 1)} vp below the study's: on these {len(y2007)} dates the study's forward is {100 * (y2007['study_fwd'].min() - 1):.1f} to {100 * (y2007['study_fwd'].max() - 1):.1f} % above the spot"
        + (f", and {repair_2007}" if repair_2007 else "")
        + f"; `check_no_nan` FAILS on {int((~truth(y2007['check_no_nan'])).sum())} of them (the standard error of the index error on the upside at the horizon is not finite: the model vol there is not invertible).{repair_pass} "
        f"Of the {count(len(extrap), 'date', 'dates')} on which the index target is extrapolated ({dates_text(list(extrap['date']))}), {int(extrap['date'].isin(far['date']).sum())} are among the {len(far)}: the last kept DJX slice is at {' and '.join(f'{x:.2f}y' for x in extrap['index_last_slice'])} for a horizon of one year, the screen having dropped {' and '.join(str(int(x)) for x in extrap['n_dropped_index_screen'])} slices. "
        f"The highest CC/copula of the pass is {fmt(hi['cc_over_copula'])} on {hi['date']} (gap {fmt(hi['atm_gap'], 2, True)} vp) and the lowest {fmt(lo['cc_over_copula'])} on {lo['date']} (gap {fmt(lo['atm_gap'], 2, True)} vp; see (iii))."
        + (f" The brackets are from {SPEC_NOTE}." if spec_i is not None else "")
    )
    if spec_i is not None:
        sp = spec_i[spec_i["row_priced"]]
        wide = sp[(sp["nearest_kept_above"] - sp["nearest_kept_below"]) > 0.75]
        none_below = sp[sp["nearest_kept_below"].isna()]
        none_above = sp[sp["nearest_kept_above"].isna()]
        b.add(
            "level.spec.n_built",
            "priced dates whose specification was rebuilt here",
            len(sp),
            unit="dates",
            n=n_p,
            notes="a count",
            source="tables/spec_by_date.csv",
        )
        b.add(
            "level.spec.n_key",
            "of which the specification key equals the row's",
            int(sp["spec_key_is_the_rows"].sum()),
            unit="dates",
            n=len(sp),
            notes="a count",
            source="tables/spec_by_date.csv",
        )
        b.add(
            "level.spec.n_wide",
            "priced dates whose two nearest kept DJX slices around the horizon are more than 0.75y apart",
            len(wide),
            unit="dates",
            n=len(sp),
            notes="a count",
            source="tables/spec_by_date.csv",
        )
        b.add(
            "level.spec.n_none_above",
            "priced dates with no kept DJX slice above the horizon",
            len(none_above),
            unit="dates",
            n=len(sp),
            notes="a count",
            source="tables/spec_by_date.csv",
        )
        b.add(
            "level.spec.n_none_below",
            "priced dates with no kept DJX slice at or below the horizon",
            len(none_below),
            unit="dates",
            n=len(sp),
            notes="a count",
            source="tables/spec_by_date.csv",
        )
        sub = p.set_index("date")
        g_wide = sub.loc[[x for x in wide.index if x in sub.index], "atm_gap"].abs()
        g_rest = sub.loc[
            [x for x in sp.index if x not in wide.index and x in sub.index], "atm_gap"
        ].abs()
        b.text(
            f"The specification was rebuilt for {len(sp)} of the {n_p} priced dates; its key equals the row's on {int(sp['spec_key_is_the_rows'].sum())} of them, so the slices listed are the rows' own (`tables/spec_by_date.csv`: per date the kept DJX slices, the slices dropped by the repair and by the screen, the names kept unscreened and the names beyond their last kept expiry). "
            f"On {len(wide)} priced dates the two kept DJX slices nearest the horizon are more than 0.75y apart, so that the one-year target is an interpolation over that distance; on {len(none_above)} no kept slice lies above the horizon and on {len(none_below)} none at or below it. "
            f"The median absolute gap to the study's level is {fmt(g_wide.median(), 2)} vp on the {len(g_wide)} dates with the wide bracket against {fmt(g_rest.median(), 2)} vp on the other {len(g_rest)} [{INCL}]."
        )
    # --- (ii) a clipped mass near 1
    near = p[p["clip_larger"] >= CLIP_NEAR_ONE]
    second = p["clip_larger"].nlargest(len(near) + 1).iloc[-1]
    b.add(
        "extreme.n_clip_near_one",
        f"priced dates with a clipped mass of {CLIP_NEAR_ONE:g} or more",
        len(near),
        unit="dates",
        n=n_p,
        notes="a count",
    )
    b.text(
        f"**(ii) A clipped mass near 1.** {count(len(near), 'priced date has', 'priced dates have')} a one-sided clipped mass of {CLIP_NEAR_ONE:g} or more; the next largest value of the pass is {fmt(second)}."
    )
    for _, r in near.iterrows():
        cells = [
            c
            for c in pm_1y.index_cells(data.rows[r["date"]])
            if c["strike"] in ("-2.5", "+0.0", "+1.5", "+2.5")
            and abs(float(c["T"]) - float(r["T"])) < 1e-9
        ]
        target = ", ".join(
            f"{c['strike']} sd {100 * float(c['target_vol']):.2f} %"
            + (f" ({m})" if (m := ", ".join(pm_1y.cell_flags(data.rows[r["date"]], c))) else "")
            for c in cells
        )
        slices, cause = "", ""
        if spec_i is not None and r["date"] in spec_i.index:
            s = spec_i.loc[r["date"]]
            gone = [float(x) for x in str(s["dropped_by_calendar_repair"]).split()] if isinstance(s["dropped_by_calendar_repair"], str) else []  # fmt: skip
            kept = [float(x) for x in str(s["kept_djx_slices"]).split()]
            slices = f" The repair drops the DJX slices at {', '.join(f'{x:.3f}' for x in gone)}y and keeps {', '.join(f'{x:.3f}' for x in kept)}y"
            slices += (
                f": it leaves no slice before {min(kept):.3f}y."
                if gone and kept and max(gone) < min(kept)
                else "."
            )
            if has_target and finite(s.get("repair_effect_max_abs_vp")):
                worst = float(s["repair_effect_max_abs_vp"])
                cause = (
                    f" With the repair switched off (the specification rebuilt with `screen.calendar_repair` off) the target at the horizon is the same at every strike of the row (largest difference {worst:.3f} vp; at {', '.join(SMILE_STRIKES)} sd: {', '.join(str(s['target_smile_repair_off']).split())} % against {', '.join(str(s['target_smile_repair_on']).split())} % with it): in both builds it is read between the kept slices at {years(s['nearest_kept_below'])} and {years(s['nearest_kept_above'])}, and the repair does not change the target at the horizon on that date."
                    if worst < SAME_SMILE_VP
                    else f" With the repair switched off (the specification rebuilt with `screen.calendar_repair` off) the target at the horizon differs by up to {worst:.2f} vp over the strikes of the row (at {', '.join(SMILE_STRIKES)} sd: {', '.join(str(s['target_smile_repair_off']).split())} % against {', '.join(str(s['target_smile_repair_on']).split())} % with it)."
                )
        cause += (
            f" Whether the mass of {fmt(r['clip_high_inner_max'])} is the repair's doing is not established: there is no old-defaults row for that date (`{rel(TABLE_NOREPAIR)}` has {dates_text(norepair_dates)} only), and nothing is priced here."
            if r["date"] not in norepair_dates
            else f" The old-defaults row of that date (`{rel(TABLE_NOREPAIR)}`) is the comparison to read."
        )
        b.text(
            f"- {r['date']} (status {r['status']}; `check_no_nan` {chk(r['check_no_nan'])}; decision 5's clip flag: {yes(r['flag_clip'])}; decision 2's flag: {yes(r['flag_unscreened'])}): the mass at the cap is {fmt(r['clip_high_inner_max'])} inside ±2.5 sd — at one calibration slice every particle has λ at its cap (over the whole cloud the largest slice value is {fmt(r['clip_high_max'])} and the mean over the slices {fmt(r['clip_high_mean'])}); the mass at λ = 0 is {fmt(r['clip_low_inner_max'])}. "
            f"{'' if slices else 'The calendar repair drops ' + str(int(r['n_dropped_calendar_index'])) + ' DJX slices on that date.'}{slices.lstrip() if slices else ''} The target vols at the horizon are {target} (the marks are the frozen addendum's: a target vol below a quarter of the at-the-money one is called degenerate, below a half thin).{cause} "
            f"LC/CC is {pm(r['lc_over_cc'], r['lc_over_cc_se'], 5)}, the {'highest' if r['lc_over_cc'] >= p['lc_over_cc'].max() else 'not the highest'} of the pass; LC/copula {pm(r['lc_over_copula'], r['lc_over_copula_se'])}, CC/copula {pm(r['cc_over_copula'], r['cc_over_copula_se'])}; ρ_CC {fmt(r['rho_cc'], 3)} (no error in the row) against ρ_cop {fmt(r['rho_cop'], 3)} (the study's); index error at the money {pm(r['idx_err_atm'], r['idx_err_atm_se'], 3, True)} vp and at −2.5 sd {pm(r['idx_err_m25'], r['idx_err_m25_se'], 3, True)} vp."
        )
    # --- (iii) kappa near zero
    k = p.sort_values("kappa_lc").head(2)
    lowk = k.iloc[0]
    kappa_cop = data.entries.loc[p["date"], "kappa_cop"].astype(float)
    if not (p.loc[p["kappa_cc"].idxmin(), "date"] == lowk["date"] == kappa_cop.idxmin()):
        raise ValueError("the three minima of κ are not on one date")
    quoted(REPORT, f"| {fmt(p['kappa_lc'].min())} | {fmt(p['kappa_lc'].max())} |")
    e = data.entries.loc[lowk["date"]]
    legs = pd.read_parquet(pc.STUDY / f"legs_{TENOR}.parquet")
    leg = legs[legs["date"].astype(str).str[:10] == lowk["date"]].sort_values("M").iloc[-1]
    b.add(
        "extreme.kappa_min",
        "κ_LC: the lowest value of the pass",
        lowk["kappa_lc"],
        lowk["kappa_lc_se"],
        date=lowk["date"],
        unit="ratio",
        notes="the ± is the Monte Carlo error of the date; it is not a usable error (reader note 8)",
    )
    b.add(
        "extreme.kappa_min.eqv",
        "EQV of the study's entry on that date",
        e["EQV"],
        date=lowk["date"],
        unit="squared return",
        budget=pc.BUDGETS["study"],
        commit=pm_1y.STUDY_NOTE,
        source=STUDY,
        notes="the study's number: no standard error",
    )
    b.add(
        "extreme.kappa_min.leg_M",
        f"the study's listed second moment of {leg['ticker']} on that date",
        leg["M"],
        date=lowk["date"],
        unit="squared return",
        budget=pc.BUDGETS["study"],
        commit=pm_1y.STUDY_NOTE,
        source="outputs/dispersion/legs_12m.parquet",
        notes="the study's number: no standard error",
    )
    b.add(
        "extreme.kappa_min.leg_f",
        f"the study's forward of {leg['ticker']} over its spot on that date",
        leg["f"],
        date=lowk["date"],
        unit="ratio",
        budget=pc.BUDGETS["study"],
        commit=pm_1y.STUDY_NOTE,
        source="outputs/dispersion/legs_12m.parquet",
        notes="the study's number: no standard error",
    )
    med_eqv = float(data.entries.loc[p["date"], "EQV"].median())
    med_eqv_c = float(data.entries.loc[p.loc[p["sample_c"], "date"], "EQV"].median())
    med_mc, med_mc_c = float(p["lc_over_cc_se"].median()), float(p.loc[p["sample_c"], "lc_over_cc_se"].median())  # fmt: skip
    for skey, n_s, eqv_s, mc_s in (("b", n_p, med_eqv, med_mc), ("c", int(p["sample_c"].sum()), med_eqv_c, med_mc_c)):  # fmt: skip
        b.add(
            f"extreme.eqv_median.{skey}",
            f"EQV of the study's entries: median over the priced dates, sample ({skey}) [{INCL}]",
            eqv_s,
            unit="squared return",
            n=n_s,
            budget=pc.BUDGETS["study"],
            commit=pm_1y.STUDY_NOTE,
            source=STUDY,
            notes="an order statistic of the study's numbers: no standard error",
        )
        b.add(
            f"extreme.kappa_min.lc_over_cc_se_over_median.{skey}",
            f"the Monte Carlo error of LC/CC on the date of the lowest κ over the median Monte Carlo error of LC/CC of the priced dates, sample ({skey}) [{INCL}]",
            lowk["lc_over_cc_se"] / mc_s,
            date=lowk["date"],
            unit="ratio",
            n=n_s,
            notes="a ratio of an error to an order statistic of errors: no error of its own",
        )
    note12 = quoted(READER_NOTES, f"the study drops {lowk['date']}")
    nxt = k.iloc[1]
    for key, label in (("EV_lc", "E_LC[V]"), ("EV_cc", "E_CC[V]")):
        b.add(
            f"extreme.kappa_min.{key}",
            f"{label} on the date of the lowest κ",
            lowk[key],
            lowk[f"{key}_se"],
            date=lowk["date"],
            unit="squared return",
            notes="the ± is the Monte Carlo error of the date; one name's second moment carries it",
        )
    own = ""
    un = read_spec("spec_unscreened_names")
    if un is not None:
        g = un[(un["date"] == lowk["date"]) & (un["name"] == leg["ticker"])]
        if len(g) == 1:
            g = g.iloc[0]
            own = f" and its own surface in the specification, built before any calibration, has a forward of {g['forward_over_spot']:.2f} times the spot and an at-the-money vol of {100 * g['atm_vol']:.1f} % at the horizon, from {count(int(g['n_kept_slices']), 'kept slice', 'kept slices')} at {', '.join(str(g['kept_slices']).split())}y (`tables/spec_unscreened_names.csv`)"
    b.text(
        f"**(iii) κ near zero.** The automatic report prints a minimum of κ of {fmt(p['kappa_lc'].min())} under LC, {fmt(p['kappa_cc'].min())} under CC and {fmt(kappa_cop.min())} under the copula. All three are {lowk['date']} (status {lowk['status']}; `check_forward` {chk(lowk['check_forward'])}, {lowk['forward_z']:+.1f} standard errors; decision 5's clip flag: {yes(lowk['flag_clip'])}; decision 2's flag: {lowk['names_unscreened']} kept unscreened).",
        f"What happened is in the input of one name, which the study's entry and the model's specification both read: an input, before any calibration. On that date the study's one-year leg of {leg['ticker']} (weight {leg['w_B1']:.6f}, spot {leg['spot']:.2f}) has a forward of {leg['f']:.1f} times the spot, an at-the-money vol of {100 * leg['atm_vol']:.0f} % and a listed second moment M = {leg['M']:.3g} (`outputs/dispersion/legs_12m.parquet`; its smile is bracketed by the single expiry {leg['bracket_lo']}). "
        f"The model reads the same quotes: {leg['ticker']} has no expiry passing the quote screen and is kept unscreened by decision 2 — with the fallback off the build is refused and the date is not priced —{own if own else ' (its surface in the specification was not rebuilt: `--spec-build`)'}. "
        f"Its weight times the listed moment is {leg['w_B1'] * leg['M']:.0f}, which is the whole of the entry's Σ w M = {lowk['sum_w_M']:.0f} and of its EQV = {e['EQV']:.0f} (the median EQV of the priced dates [{INCL}] is {med_eqv:.4f}, and {med_eqv_c:.4f} without decision 2's flagged dates). The copula's own numbers on that date follow: E[V] = {e['EV']:.0f}, κ_cop = {e['kappa_cop']:.4f}, P_D = {e['P_D']:.4f}, ρ_cop = {e['rho_cop']:.3f} (the study's numbers). "
        f"Under the model E_LC[V] is {lowk['EV_lc']:.0f} ± {lowk['EV_lc_se']:.0f} and E_CC[V] {lowk['EV_cc']:.0f} ± {lowk['EV_cc_se']:.0f} (Monte Carlo errors of the date; the second error is {100 * lowk['EV_cc_se'] / lowk['EV_cc']:.0f} % of its value), and E_LC[V] over the name's weight is {lowk['EV_lc'] / leg['w_B1']:.2g}: κ = E[D]/√E[V] is near zero under every model because one name's second moment is of the order of 10⁷. "
        f"The row's ratios are not resolved: LC/CC {pm(lowk['lc_over_cc'], lowk['lc_over_cc_se'])} (a Monte Carlo error {lowk['lc_over_cc_se'] / med_mc:.0f} times the median one of the priced dates [{INCL}], and {lowk['lc_over_cc_se'] / med_mc_c:.0f} times that of sample (c)), LC/copula {pm(lowk['lc_over_copula'], lowk['lc_over_copula_se'])}, CC/copula {pm(lowk['cc_over_copula'], lowk['cc_over_copula_se'])}. "
        f'They are the minima of the three ratios on sample (b) and they are out of sample (c). The frozen package\'s reader note 12 records it: "{note12}".',
        f"The next lowest κ_LC is {pm(nxt['kappa_lc'], nxt['kappa_lc_se'])} on {nxt['date']} (decision 5's clip flag: {yes(nxt['flag_clip'])}): there E_LC[V]/EQV is {pm(nxt['EV_over_EQV'], nxt['EV_over_EQV_se'], 2)} and the names' Monte Carlo second moment is {100 * nxt['names_mc_over_listed']:+.0f} % from the listed strips — a second moment carried by a few paths (reader note 8); the row's LC/CC is {pm(nxt['lc_over_cc'], nxt['lc_over_cc_se'])}. κ_LC is below 0.70 on {int((p['kappa_lc'] < 0.70).sum())} of the {n_p} priced dates [{INCL}]. κ, E[V]/EQV and ED_eqv (= κ_LC·√EQV, section 0) are not among the four ratios of section 2 for that reason; section 6 sets them beside the frozen ones, with and without decision 2's dates.",
    )
    # --- (iv) the SVI-strip diagnostic
    v = p["names_svi_over_listed"]
    broken = p[v > STRIP_FAR].sort_values("names_svi_over_listed", ascending=False)
    origin, hits = strip_source_check(data.commit)
    report_mean = f"{v.mean():.0f}"
    quoted(REPORT, f"| SVI strips / listed strips - 1 | {n_p} | {report_mean}.0000 |")
    n_other = sum(
        1
        for h in hits
        if not re.search(
            r"sum_svi =|row\.update\(|sum_w_M_svi=|names_mc_z=|names_mc_over_listed=|r\[\"", h
        )
    )
    finite_all = all(np.isfinite(p[c]).all() for c in STRIP_COLUMNS)
    b.add(
        "strip.n_far",
        f"priced dates on which the SVI strips exceed the listed strips by more than {STRIP_FAR:g} times the listed strips",
        len(broken),
        unit="dates",
        n=n_p,
        notes="a count",
    )
    b.add(
        "strip.median",
        "SVI strips / listed strips − 1: median over the priced dates",
        v.median(),
        unit="ratio",
        n=n_p,
        notes="an order statistic of a quadrature (no Monte Carlo): no standard error",
    )
    b.add(
        "strip.max",
        "SVI strips / listed strips − 1: largest value",
        v.max(),
        date=str(p.loc[v.idxmax(), "date"]),
        unit="ratio",
        n=n_p,
        notes="a quadrature, no Monte Carlo: no standard error",
    )
    b.add(
        "strip.mean_without_far",
        f"SVI strips / listed strips − 1: mean over the priced dates on which it is at most {STRIP_FAR:g}",
        v[v <= STRIP_FAR].mean(),
        v[v <= STRIP_FAR].std(ddof=1) / math.sqrt(int((v <= STRIP_FAR).sum())),
        unit="ratio",
        n=int((v <= STRIP_FAR).sum()),
        notes="standard error across dates, sd/√n; the dates left out are selected on the value itself",
    )
    b.add(
        "strip.source_lines",
        "lines of the pricer's source at the rows' commit that name the SVI strip sum or one of its four columns",
        len(hits),
        unit="lines",
        notes="a count",
        source=origin.strip("`"),
    )
    b.add(
        "strip.source_lines_other",
        "of which lines that are neither the computation, the row's assignment nor the date's printed summary",
        n_other,
        unit="lines",
        notes="a count",
        source=origin.strip("`"),
    )
    b.text(
        f"**(iv) The diagnostic \"SVI strips / listed strips − 1\".** The automatic report prints its mean as {report_mean} ({v.mean():.1e}) and the mean of the companion z-score as {p['names_mc_z'].mean():.1e}. Neither is a result. The diagnostic is the names' second moment Σ w E[R_i²] computed from each name's own SVI surface by quadrature (`strip_second_moment`: the strip of vanillas over ±8 at-the-money sd in log-strike, plus lognormal tails at the edge vols), over the study's listed strips, minus 1. "
        f"Its median over the {n_p} priced dates [{INCL}] is {fmt(v.median(), 4, True)} and its quartiles {fmt(v.quantile(0.25), 4, True)} and {fmt(v.quantile(0.75), 4, True)}; it exceeds {STRIP_FAR:g} (the SVI strips more than twice the listed ones) on {len(broken)} dates and 100 on {int((v > 100).sum())}; the largest value, {v.max():.3g}, is {p.loc[v.idxmax(), 'date']} (the date of (iii)) and alone makes the printed mean. On the other {int((v <= STRIP_FAR).sum())} dates its mean is {pm(b.get('strip.mean_without_far'), b.se('strip.mean_without_far'), 4, True)} (± across dates; those dates are selected on the value itself) [{INCL}].",
    )
    strips = read_spec("spec_strips_by_name")
    rows = []
    for _, r in broken.iterrows():
        who = "not built"
        if strips is not None and r["date"] in set(strips["date"]):
            s = strips[(strips["date"] == r["date"]) & (strips["rank"] == 1)].iloc[0]
            who = f"{s['name']} (weight {s['weight']:.4f}): call side of its strip {s['call_side']:.3g} against a listed M of {s['M_listed']:.3g}; SVI total variance {s['total_variance_at_upper_edge']:.1f} at the grid's upper edge, a strike {s['strike_over_forward_at_edge']:.3g} times the forward; {100 * s['share_of_sum']:.1f} % of the sum"
        rows.append(
            [
                r["date"],
                r["status"],
                f"{r['names_svi_over_listed']:.3g}",
                f"{100 * r['names_mc_over_listed']:+.2f} %",
                chk(r["check_names"]),
                yes(r["flag_clip"]),
                yes(r["flag_unscreened"]),
                who,
            ]
        )
    b.table(
        [
            "date",
            "status",
            "SVI strips / listed strips − 1",
            "Monte Carlo / listed strips − 1",
            "check_names (diagnostic)",
            "clip flag (decision 5)",
            "unscreened (decision 2)",
            "the name behind it (specification-only build)",
        ],
        rows,
    )
    b.save(
        "strip_diagnostic_dates",
        broken[
            [
                "date",
                "status",
                "names_svi_over_listed",
                "names_mc_over_listed",
                "names_mc_over_svi",
                "names_mc_z",
                "sum_w_M_svi",
                "sum_w_M",
                "sum_w_ER2_lc",
                "sum_w_ER2_lc_se",
                "check_names",
                "flag_clip",
                "flag_unscreened",
                "n_names_extrapolated",
            ]
        ],
    )
    first = strips[strips["rank"] == 1] if strips is not None else None
    edge = first.loc[first["strike_over_forward_at_edge"].idxmin()] if first is not None else None
    strip_gap = (
        float((strips["sum_w_M_svi_built"] / strips["sum_w_M_svi_row"] - 1).abs().max())
        if strips is not None
        else NAN
    )
    strip_same = (
        "equals the row's `sum_w_M_svi` on every date"
        if strip_gap == 0.0
        else f"differs from the row's `sum_w_M_svi` by at most {strip_gap:.1g} in relative terms"
    )
    why = (
        f"Why. On {int((first['share_of_sum'] > 0.9).sum())} of these {len(first)} dates one name carries more than 90 % of the sum, and on every one the first name carries at least {math.floor(100 * first['share_of_sum'].min())} % (table: `tables/spec_strips_by_name.csv`, from {SPEC_NOTE}; the sum rebuilt there {strip_same}). The name's SVI slice at the horizon has a call wing whose total variance, extrapolated beyond the listed strikes to the upper edge of the quadrature grid (a strike {floor_to(edge['strike_over_forward_at_edge'], 1):.1f} or more times the forward: the lowest is {edge['strike_over_forward_at_edge']:.2f}, {edge['name']} on {edge['date']}), is between {first['total_variance_at_upper_edge'].min():.1f} and {first['total_variance_at_upper_edge'].max():.1f}; the lognormal tail added at that edge grows as e to that power. "
        if strips is not None
        else "Why: the specification-only build that attributes the sum to a name was not run (`--spec-build`). "
    )
    b.text(
        why
        + f"The Monte Carlo second moment of the same names is within 5 % of the listed strips on {int((broken['names_mc_over_listed'].abs() <= 0.05).sum())} of these {len(broken)} dates (the column \"Monte Carlo / listed strips − 1\"): the diagnostic measures the extrapolation of an SVI wing beyond the quotes, which the simulated second moment does not follow.",
        f"**No price depends on it** — checked in the source at the rows' commit ({origin}). The sum `sum_svi` and the four columns it feeds (`sum_w_M_svi`, `names_mc_over_svi`, `names_mc_z`, `names_svi_over_listed`) are named on {len(hits)} lines: the line that computes the sum, the lines that assign the four columns, and the lines that print them in the date's summary (`log.info`); {'no other line reads them' if not n_other else str(n_other) + ' other line(s) read them'}. "
        f"E_LC[D], E_CC[D], the calls, κ, ED_wing and ED_eqv are computed before or without them (ED_wing and ED_eqv use the study's listed strips, `sum_wm` and `eqv`), and `check_names` compares the Monte Carlo moment with the listed strips. The one dependence is the status: `check_no_nan` reads every float of the row, so a non-finite value of one of the four columns would turn a date to check; on the {n_p} priced dates the four columns are {'finite on every date, so no status depends on them either' if finite_all else 'NOT finite on some dates'}. The lines:",
        "```\n" + "\n".join(hits) + "\n```",
    )
    if n_other:
        raise ValueError(
            "the pricer's source reads the SVI strip columns elsewhere: the claim of section 4 (iv) does not hold"
        )


# ----------------------------------------------------------------------------- section 5
def section_calls(b: Book, data: Data) -> None:
    p = data.priced
    validation = {r["id"]: r for r in json.loads(VALIDATION.read_text())}
    note7 = quoted(
        READER_NOTES,
        "**Calls at 1.5 × and 2 × the forward under LC and CC are carried by a few runaway paths** (one name ending above 3 times its spot, beyond the strikes listed around the horizon): do not quote ratios or payouts at those strikes as model results, and treat 1.25 × with care (section V3; measured at the development budget on two dates).",
    )
    quoted(READER_NOTES, "The K_200 ratios are in addition statistically unusable (A4, B2).")
    usable, long_rows, verdicts = [], [], {}
    runaway_3m: dict[str, str] = {}
    for tag in CALL_TAGS:
        mult = int(tag) / 100
        ratio, ratio_se = p[f"C_{tag}_ratio"], p[f"C_{tag}_ratio_se"]
        rel_se = (ratio_se / ratio).dropna()
        shares = [
            validation[f"V.runaway.{d}.lc.{tag}.share_above_3x"]["value"]
            for d in ("2026-10-02", "2023-02-06")
        ]
        runaway_3m[tag] = f"{100 * min(shares):.0f} and {100 * max(shares):.0f} %"
        unusable = float(rel_se.median()) > CALL_REL_SE
        runaway = tag in ("150", "200")
        not_measured = "runaway share at one year not measured"
        verdict = (
            "not a result: statistically unusable, and carried by runaway paths at three months"
            if unusable
            else (
                "not a result: carried by a few runaway paths at three months (reader note 7)"
                if runaway
                else (
                    f"quoted with care (reader note 7); {not_measured}"
                    if tag == "125"
                    else f"quoted; {not_measured}"
                )
            )
        )
        verdicts[tag] = verdict
        above = sorted(p.loc[rel_se[rel_se > CALL_REL_SE].index, "date"])
        b.add(
            f"calls.{tag}.rel_se_median",
            f"call at {mult:g} × P_D: median over the dates of the relative Monte Carlo error of C_LC/C_CC",
            rel_se.median(),
            unit="ratio",
            n=len(rel_se),
            notes="an order statistic of errors: no error of its own",
        )
        b.add(
            f"calls.{tag}.share_rel_se_above",
            f"call at {mult:g} × P_D: share of the dates on which that relative error exceeds {CALL_REL_SE:g}",
            float((rel_se > CALL_REL_SE).mean()),
            unit="share of dates",
            n=len(rel_se),
            notes="a share: no standard error",
        )
        rel_c = rel_se[p.loc[rel_se.index, "sample_c"]]
        above_c = sorted(p.loc[rel_c[rel_c > CALL_REL_SE].index, "date"])
        same_verdict = (float(rel_c.median()) > CALL_REL_SE) == unusable
        b.add(
            f"calls.{tag}.rel_se_median.c",
            f"call at {mult:g} × P_D: median over the dates without decision 2's flagged dates of the relative Monte Carlo error of C_LC/C_CC [{INCL}]",
            rel_c.median(),
            unit="ratio",
            n=len(rel_c),
            notes="an order statistic of errors: no error of its own",
        )
        b.add(
            f"calls.{tag}.share_rel_se_above.c",
            f"call at {mult:g} × P_D: share of the dates without decision 2's flagged dates on which that relative error exceeds {CALL_REL_SE:g} [{INCL}]",
            float((rel_c > CALL_REL_SE).mean()),
            unit="share of dates",
            n=len(rel_c),
            notes="a share: no standard error",
        )
        usable.append(
            [
                f"{mult:g} × P_D",
                str(int(ratio.notna().sum())),
                f"{100 * rel_se.median():.1f} %",
                few(len(above), len(rel_se), above),
                f"{100 * rel_c.median():.1f} %; {few(len(above_c), len(rel_c), above_c)}"
                + ("" if same_verdict else f" — on the other side of {100 * CALL_REL_SE:.0f} %"),
                f"{100 * min(shares):.1f} % and {100 * max(shares):.1f} %",
                verdict,
            ]
        )
        for key, label, col, se_col in (
            ("lc_over_cc", "C_LC / C_CC", f"C_{tag}_ratio", f"C_{tag}_ratio_se"),
            (
                "lc_over_copula",
                "C_LC / copula",
                f"C_{tag}_lc_over_copula",
                f"C_{tag}_lc_over_copula_se",
            ),
            (
                "cc_over_copula",
                "C_CC / copula",
                f"C_{tag}_cc_over_copula",
                f"C_{tag}_cc_over_copula_se",
            ),
        ):
            for sample, sample_label in (x for x in SAMPLES if x[0] in CALL_SAMPLES):
                sub = p[p[f"sample_{sample}"]]
                s = summarise(sub[col], sub[se_col])
                long_rows.append(
                    {
                        "strike": f"K_{tag}",
                        "multiple_of_P_D": mult,
                        "quantity": key,
                        "label": label,
                        "sample": sample,
                        "verdict": verdict,
                        **s,
                    }
                )
                if tag in ("050", "075", "100", "125"):
                    b.add(
                        f"calls.{tag}.{key}.mean.{sample}",
                        f"call at {mult:g} × P_D, {label}: mean, {sample_label}",
                        s.get("mean"),
                        s.get("se"),
                        unit="ratio",
                        n=int(s["n"]),
                        notes=(
                            "standard error across dates, sd/√n"
                            if finite(s.get("se"))
                            else no_se_reason(int(s["n"]))
                        )
                        + ("; " + verdict if tag == "125" else ""),
                    )
                    b.add(
                        f"calls.{tag}.{key}.median.{sample}",
                        f"call at {mult:g} × P_D, {label}: median, {sample_label}",
                        s.get("median"),
                        unit="ratio",
                        n=int(s["n"]),
                        notes="an order statistic across dates: no standard error"
                        + ("; " + verdict if tag == "125" else ""),
                    )
    calls = pd.DataFrame(long_rows)
    b.save("calls_summary", calls)
    b.text(
        "## 5. The calls on the dispersion at the study's strikes",
        "The strikes are the study's cash strikes, multiples of the copula's forward P_D, the same under every model: K_050 to K_200 are 0.5, 0.75, 1, 1.25, 1.5 and 2 times P_D. C_LC and C_CC are E[(D − K)⁺] under LC and CC on common paths; the copula's call is `C_<m>` of the study's entry.",
        f'**Which strikes can be quoted.** The frozen package\'s reader note 7 says: "{note7}" It adds that the K_200 ratios are statistically unusable. That measurement is at three months (records `V.runaway.*` of `parts/V_validation.json`: the share of the LC call carried by the paths on which one name ends above 3 times its spot, on 2026-10-02 and 2023-02-06). **It has not been made at one year**: the rows hold no paths, and nothing is priced here. '
        f'The table gives, per strike, what the one-year rows themselves say about the Monte Carlo error on the {len(p)} priced dates, sample (b) [{INCL}], the same on sample (c), without the dates of decision 2\'s flag, and the three-month shares for reference. "Quoted" below is therefore conditional: it rests on the Monte Carlo error of the one-year rows and on runaway shares measured at three months on two dates, not on a one-year measurement.',
    )
    b.table(
        [
            "strike",
            "dates with a ratio C_LC/C_CC",
            "median relative Monte Carlo ± of C_LC/C_CC (one year)",
            f"dates with that ± above {100 * CALL_REL_SE:.0f} %",
            f"the same two without decision 2's flagged dates (sample (c)) [{INCL}]: median ±; dates above {100 * CALL_REL_SE:.0f} %",
            "share of the LC call on runaway paths at three months (two dates)",
            "in this page",
        ],
        usable,
    )
    rows = []
    for tag in ("050", "075", "100", "125"):
        for key, label in (
            ("lc_over_cc", "C_LC / C_CC"),
            ("lc_over_copula", "C_LC / copula"),
            ("cc_over_copula", "C_CC / copula"),
        ):
            cells = [f"{int(tag) / 100:g} × P_D" + (" †" if tag == "125" else ""), label]
            for sample in ("a", "a2", "b", "c"):
                s = calls[
                    (calls["strike"] == f"K_{tag}")
                    & (calls["quantity"] == key)
                    & (calls["sample"] == sample)
                ].iloc[0]
                if sample == "a":
                    cells += [str(int(s["n"])), fmt(s["mean"]), fmt(s["median"]), f"{fmt(s['q25'])} to {fmt(s['q75'])}"]  # fmt: skip
                elif sample == "a2":
                    cells += [str(int(s["n"])), fmt(s["mean"]), fmt(s["median"])]
                elif sample == "b":
                    cells += [str(int(s["n"])), pm(s["mean"], s["se"]), fmt(s["median"]), f"{fmt(s['q25'])} to {fmt(s['q75'])}", fmt(s["nw"])]  # fmt: skip
                else:
                    cells += [str(int(s["n"])), pm(s["mean"], s["se"]), fmt(s["nw"])]
            rows.append(cells)
    b.text(
        "**The strikes quoted on that condition** (0.5, 0.75 and 1 times P_D; 1.25 † with care). Sample (a) is the rule of decision 5 and (a2) the same without decision 2's flagged date: neither carries a standard error. Samples (b), all priced dates, and (c), without decision 2's flagged dates, contain flagged rows ["
        + INCL
        + "]: mean ± across dates."
    )
    b.table(
        [
            "strike",
            "ratio",
            "(a) n",
            "(a) mean",
            "(a) median",
            "(a) quartiles",
            "(a2) n",
            "(a2) mean",
            "(a2) median",
            "(b) n",
            "(b) mean ± across dates",
            "(b) median",
            "(b) quartiles",
            f"(b) ± Newey–West ({NW_LAGS} lags)",
            "(c) n",
            "(c) mean ± across dates",
            f"(c) ± Newey–West ({NW_LAGS} lags)",
        ],
        rows,
    )

    def call_stats(tag: str, sample: str = "b") -> dict[str, Any]:
        row = calls[
            (calls["strike"] == f"K_{tag}")
            & (calls["quantity"] == "lc_over_cc")
            & (calls["sample"] == sample)
        ].iloc[0]
        return dict(row.to_dict())

    readings = []
    for tag in ("050", "075", "100"):
        s, s_c = call_stats(tag), call_stats(tag, "c")
        readings.append(
            f"at {int(tag) / 100:g} × P_D it is {pm(s['mean'], s['se'])} (Newey–West {fmt(s['nw'])}; n {int(s['n'])}; quartiles {fmt(s['q25'])} to {fmt(s['q75'])}) — {Rules.against_lags(s['mean'], 1.0, s)} — and on sample (c) {pm(s_c['mean'], s_c['se'])} (Newey–West {fmt(s_c['nw'])}; n {int(s_c['n'])}) — {Rules.against_lags(s_c['mean'], 1.0, s_c)}"
        )
    und = int(p["C_200_ratio"].isna().sum())
    n_200 = int(p["C_200_ratio"].notna().sum())
    read_tags = ("050", "075", "100")
    small_a, small_a2 = ([call_stats(tag, k) for tag in read_tags] for k in ("a", "a2"))
    strikes = listing([f"{int(tag) / 100:g}" for tag in read_tags])
    b.text(
        f"† 1.25 × P_D: at three months {runaway_3m['125']} of the LC call at this strike sat on runaway paths on the two dates measured (against {runaway_3m['100']} at 1 × P_D); its mean and its ± are to be read with that in mind. "
        f"Reading of C_LC/C_CC, by the rule of section 0. On sample (a), the rule of decision 5 (n {int(small_a[0]['n'])}), its mean at {strikes} times P_D is {listing([fmt(s['mean']) for s in small_a])}, and on sample (a2), without decision 2's flagged date (n {int(small_a2[0]['n'])}), {listing([fmt(s['mean']) for s in small_a2])}: {Rules.against_small(1.0, small_a[0], small_a2[0])}. "
        f"On sample (b) [{INCL}]: " + "; ".join(readings) + ".",
        f"**The strikes that are not quoted.** At 1.5 × P_D the one-year rows' own Monte Carlo error is {100 * b.get('calls.150.rel_se_median'):.1f} % of the ratio in the median on sample (b) [{INCL}], but that error is not a usable one when a few paths carry the price, which is the case at three months ({runaway_3m['150']} of the LC call on the two dates measured); no mean ratio is printed for it. At 2 × P_D, on sample (b) as well [{INCL}], the ratio C_LC/C_CC is undefined on {count(und, 'date', 'dates')} and its median relative Monte Carlo error is {100 * b.get('calls.200.rel_se_median'):.0f} %, above {100 * CALL_REL_SE:.0f} % on {round(b.get('calls.200.share_rel_se_above') * n_200)} of the {n_200} dates with a ratio: it is statistically unusable, and {runaway_3m['200']} of the LC call sat on runaway paths at three months. "
        f"The automatic report's section 2 prints means of C_LC/C_CC for both strikes ({fmt(p['C_150_ratio'].mean())} and {fmt(p['C_200_ratio'].mean())}); they are not results. `tables/calls_summary.csv` keeps the statistics of every strike with the verdict of the table above in each line.",
    )


# ----------------------------------------------------------------------------- section 6
def section_frozen(
    b: Book, data: Data, stats: dict[tuple[str, str], dict[str, float]]
) -> list[str]:
    p = data.priced
    n_p = len(p)
    page = FROZEN_1Y / "NUMBERS_1Y.md"
    frozen = {r["id"]: r for r in json.loads((FROZEN_1Y / "numbers_1y.json").read_text())}
    b.text(
        "## 6. Against the frozen one-year addendum",
        f"The addendum (`{rel(page)}`, closed 9 Oct 13:00, {len(frozen)} records in `numbers_1y.json`) has two samples: the four reference dates at the production budget and twenty yearly dates at the development budget (the first monthly one-year entry of each year 2007–2026, with {TODAY} for 2026). Nothing of it is changed here.",
    )
    # --- (1) the yearly development rows, record by record
    n_cmp, n_same, diffs, not_recomputed = 0, 0, [], 0
    keys_seen: set[str] = set()
    other_keys: set[str] = set()
    for rid, rec in frozen.items():
        m = re.match(r"1Y\.dev\.(\d{4}-\d{2}-\d{2})\.(.+)$", rid)
        if not m:
            continue
        date, key = m.group(1), m.group(2)
        if key not in Q1Y or date not in data.rows or rec["value"] is None:
            not_recomputed += 1
            other_keys.add(key)
            continue
        v, s = value_of(key, data.rows[date], data.entries.loc[date].to_dict())
        n_cmp += 1
        keys_seen.add(key)
        same_v = finite(v) and abs(v - rec["value"]) <= 1e-12 * max(1.0, abs(rec["value"]))
        same_s = (rec["se"] is None and not finite(s)) or (
            rec["se"] is not None
            and finite(s)
            and abs(s - rec["se"]) <= 1e-12 * max(1.0, abs(rec["se"]))
        )
        if same_v and same_s:
            n_same += 1
        else:
            diffs.append(
                {
                    "record": rid,
                    "frozen_value": rec["value"],
                    "frozen_se": rec["se"],
                    "recomputed_value": v,
                    "recomputed_se": s,
                }
            )
    b.add(
        "frozen.dev_records_compared",
        "frozen per-date development records recomputed from the row files",
        n_cmp,
        unit="records",
        notes="a count",
        source=rel(FROZEN_1Y / "numbers_1y.json"),
    )
    b.add(
        "frozen.dev_records_equal",
        "of which equal to the frozen value and standard error to 12 significant digits",
        n_same,
        unit="records",
        n=n_cmp,
        notes="a count",
        source=rel(FROZEN_1Y / "numbers_1y.json"),
    )
    b.save(
        "frozen_dev_record_differences",
        pd.DataFrame(
            diffs,
            columns=["record", "frozen_value", "frozen_se", "recomputed_value", "recomputed_se"],
        ),
    )
    b.checks.append(
        f"The frozen addendum's per-date development records recomputed from the row files with the addendum's own definitions (`pm_1y.ALL`): {n_same} of {n_cmp} equal to 12 significant digits ({len(keys_seen)} quantities on the twenty yearly dates)."
    )
    in_list = [x for x in data.yearly if x in data.monthly]
    b.text(
        f"**(1) The twenty yearly dates, date by date.** Are the yearly dates in the monthly list? {len(in_list)} of {len(data.yearly)} are; {dates_text([x for x in data.yearly if x not in data.monthly])} is not, and is in the table all the same (section 1). On all twenty the monthly table holds the very row file the addendum read: the pass found the rows written by the same commit with the same configuration and did not run them again. "
        f"There is therefore no second Monte Carlo draw to compare on these dates. Recomputed from the row files with the addendum's own quantity definitions, {n_same} of its {n_cmp} per-date development records ({len(keys_seen)} quantities × 20 dates, value and standard error) are equal to 12 significant digits"
        + (
            "; no record differs."
            if not diffs
            else f"; {len(diffs)} differ (`tables/frozen_dev_record_differences.csv`)."
        )
        + f" {not_recomputed} further per-date development records are not recomputed here ({', '.join(f'`{k}`' for k in sorted(other_keys))}: the nearest kept DJX slices, which the addendum read from its own specification build, and cells that are empty in the frozen file)."
    )
    # --- (2) the production rows against the monthly development rows
    rows, long_rows = [], []
    keys2 = (
        "ED_lc",
        "ED_cc",
        "lc_over_cc",
        "lc_over_copula",
        "cc_over_copula",
        "kappa_lc",
        "kappa_cc",
        "C_075_lc_over_cc",
        "C_100_lc_over_cc",
        "ED_wing_over_cc",
        "idx_err_atm",
        "idx_err_90",
        "idx_err_m25",
        "clip_low_inner_max",
        "clip_high_inner_max",
        "rho_cc",
    )
    prod_table = pd.read_parquet(TABLE_PROD).set_index("date")
    dev_table = p.set_index("date")
    inputs_equal = all(
        float(prod_table.loc[x, c]) == float(dev_table.loc[x, c])
        for x in prod_table.index
        for c in INPUT_COLUMNS
    )

    def clip_words(low: Any, high: Any) -> str:
        sides = [k for k, v in (("λ = 0", low), ("the cap", high)) if v is True or v == 1.0]
        return "yes: " + " and ".join(sides) if sides else "no"

    over = []
    for date in pc.REFERENCE_DATES:
        for key in keys2:
            rec = frozen.get(f"1Y.prod.{date}.{key}")
            if rec is None or rec["value"] is None:
                continue
            v, s = value_of(key, data.rows[date], data.entries.loc[date].to_dict())
            bound = math.hypot(s, rec["se"]) if finite(s) and rec["se"] is not None else NAN
            z = (v - rec["value"]) / bound if finite(bound) and bound > 0 else NAN
            long_rows.append(
                {
                    "date": date,
                    "quantity": key,
                    "frozen_production": rec["value"],
                    "frozen_production_se": rec["se"],
                    "monthly_development": v,
                    "monthly_development_se": s,
                    "development_minus_production": v - rec["value"],
                    "bound": bound,
                    "difference_over_bound": z,
                    "listed": bool(finite(z) and abs(z) > Z_LIST),
                }
            )
            if finite(z) and abs(z) > Z_LIST:
                over.append((date, key, v - rec["value"], bound, z))
    cmp2 = pd.DataFrame(long_rows)
    b.save("frozen_production_vs_monthly_development", cmp2)
    for date in pc.REFERENCE_DATES:
        cells = [date]
        for key in ("lc_over_cc", "ED_lc", "ED_cc", "lc_over_copula", "cc_over_copula", "rho_cc"):
            r = cmp2[(cmp2["date"] == date) & (cmp2["quantity"] == key)].iloc[0]
            digits = (
                5
                if key in ("lc_over_cc", "lc_over_copula", "cc_over_copula")
                else 6 if key.startswith("ED") else 4
            )
            cells.append(
                f"{fmt(r['frozen_production'], digits)} → {fmt(r['monthly_development'], digits)} ({fmt(r['development_minus_production'], digits, True)}"
                + (
                    f"; {r['difference_over_bound']:+.1f} bounds)"
                    if finite(r["difference_over_bound"])
                    else "; no error)"
                )
            )
        cells.append(
            f"{clip_words(prod_table.loc[date, 'flag_clip_low'], prod_table.loc[date, 'flag_clip_high'])} / {clip_words(dev_table.loc[date, 'flag_clip_low'], dev_table.loc[date, 'flag_clip_high'])}"
        )
        rows.append(cells)
        r = cmp2[(cmp2["date"] == date) & (cmp2["quantity"] == "lc_over_cc")].iloc[0]
        b.add(
            f"frozen.budget.{date}.lc_over_cc_dev_minus_prod",
            "LC/CC at the development budget (monthly pass) minus LC/CC at the production budget (frozen addendum), one year",
            r["development_minus_production"],
            r["bound"],
            date=date,
            unit="ratio",
            budget=f"{BUDGET} minus {pc.BUDGETS['production']}",
            source=f"{rel(ROWS)}; {rel(FROZEN_1Y / 'numbers_1y.json')}",
            notes="the ± is √(se_dev² + se_prod²), an upper bound of the pricing error of the difference (the two runs are two calibrations with the same seeds, not paired path by path); it contains no calibration noise",
        )
    dd = cmp2[cmp2["quantity"] == "lc_over_cc"]["development_minus_production"]
    note1 = quoted(
        READER_NOTES,
        "LC/CC differs by 0.0005 to 0.0019 on the four dates of A and B (production lower on all four)",
    )
    b.text(
        f"**(2) The four reference dates: the frozen production rows beside the monthly development rows.** Same commit ({data.commit}), same defaults, same seeds; the inputs of the two rows are {'the same' if inputs_equal else 'NOT the same'} on the four dates (equal `{'`, `'.join(INPUT_COLUMNS)}`), and the budget differs: {pc.BUDGETS['production']} against {BUDGET}. Each cell: frozen production → monthly development (development minus production; the difference in units of its bound √(se_dev² + se_prod²), the two Monte Carlo errors).",
    )
    b.table(
        [
            "date",
            "LC/CC",
            "E_LC[D]",
            "E_CC[D]",
            "LC/copula",
            "CC/copula",
            "ρ_CC",
            "clip flag (decision 5): production row / development row",
        ],
        rows,
    )
    by_q = {}
    for _date, key, _diff, _bound, _z in over:
        by_q.setdefault(key, []).append(f"{_date} ({_z:+.1f})")
    n_lower, n_higher = int((dd < 0).sum()), int((dd > 0).sum())
    side = "lower" if n_lower == len(dd) else "higher" if n_higher == len(dd) else "lower or higher"
    b.text(
        f"Of the {len(cmp2)} cells compared ({len(keys2)} quantities on four dates; `tables/frozen_production_vs_monthly_development.csv`), {len(over)} differ by more than {Z_LIST:g} bounds: "
        + "; ".join(f"{k}: {', '.join(v)}" for k, v in by_q.items())
        + f". The two rows of a date differ by the budget alone (same commit, defaults, seeds and inputs). What the table shows is that these differences are larger than the pricing errors of the two rows, and that LC/CC at the development budget is {side} than at the production budget on {max(n_lower, n_higher)} of the {len(dd)} dates, by {fmt(dd.abs().min(), 5)} to {fmt(dd.abs().max(), 5)}. "
        "It does not show which part of the budget produces them: the particles of the calibration, the paths of the companion's fit or the pricing paths all differ between the two rows. The bound is the unpaired error of two runs, while the ± of LC/CC in a row is paired on common paths: a move of E_LC[D] or of E_CC[D] that is inside one bound can be several of LC/CC's own errors, so \"within one bound\" does not clear either of them. The printed ± of a date contains none of this."
    )
    # the difference of LC/CC split into its part from E_LC[D] and its part from E_CC[D]
    decomp = []
    for date in pc.REFERENCE_DATES:
        pr, dv = prod_table.loc[date], dev_table.loc[date]
        d_ratio = float(dv["ratio"]) - float(pr["ratio"])
        part_lc = (float(dv["ED_lc"]) - float(pr["ED_lc"])) / float(pr["ED_cc"])
        part_cc = float(dv["ED_lc"]) * (1.0 / float(dv["ED_cc"]) - 1.0 / float(pr["ED_cc"]))
        listed = cmp2[(cmp2["date"] == date) & (cmp2["quantity"] == "lc_over_cc")].iloc[0]
        if abs(part_lc + part_cc - d_ratio) > 1e-12 or abs(d_ratio - listed["development_minus_production"]) > 1e-12:  # fmt: skip
            raise ValueError(f"{date}: the two parts do not sum to the difference of LC/CC")
        z = {
            key: float(cmp2[(cmp2["date"] == date) & (cmp2["quantity"] == key)].iloc[0]["difference_over_bound"])
            for key in ("ED_lc", "ED_cc")
        }  # fmt: skip
        d_rho = float(cmp2[(cmp2["date"] == date) & (cmp2["quantity"] == "rho_cc")].iloc[0]["development_minus_production"])  # fmt: skip
        decomp.append({
            "date": date, "lc_over_cc_dev_minus_prod": d_ratio, "part_from_ED_lc": part_lc, "part_from_ED_cc": part_cc, "source": "E_CC[D]" if abs(part_cc) > abs(part_lc) else "E_LC[D]",
            "ED_lc_difference_over_bound": z["ED_lc"], "ED_cc_difference_over_bound": z["ED_cc"], "rho_cc_dev_minus_prod": d_rho,
        })  # fmt: skip
        for key, value in (("part_lc", part_lc), ("part_cc", part_cc)):
            b.add(
                f"frozen.budget.{date}.lc_over_cc_dev_minus_prod.{key}",
                f"LC/CC, development minus production, one year: the part from {'E_LC[D]' if key == 'part_lc' else 'E_CC[D]'}",
                value,
                date=date,
                unit="ratio",
                budget=f"{BUDGET} minus {pc.BUDGETS['production']}",
                source=f"{rel(TABLE)}; {rel(TABLE_PROD)}",
                notes="an exact split of the difference of two rows (ΔE_LC[D]/E_CC[D] at the production budget; the rest): no error of its own; the bound of the whole difference is that of the record without the suffix",
            )
    b.save("budget_decomposition", pd.DataFrame(decomp))
    b.text(
        "**Where the difference of LC/CC comes from, date by date.** The difference splits exactly into a part from E_LC[D] (the difference of E_LC[D] over E_CC[D] at the production budget) and a part from the companion (E_LC[D] at the development budget times the difference of 1/E_CC[D]). The source named is the larger part in absolute value."
    )
    b.table(
        [
            "date",
            "LC/CC, development minus production",
            "part from E_LC[D]",
            "part from E_CC[D]",
            "source: the larger part",
            "E_LC[D], difference in bounds",
            "E_CC[D], difference in bounds",
            "ρ_CC, development minus production (no error in the row)",
        ],
        [
            [
                x["date"], fmt(x["lc_over_cc_dev_minus_prod"], 5, True), fmt(x["part_from_ED_lc"], 5, True), fmt(x["part_from_ED_cc"], 5, True), x["source"],
                fmt(x["ED_lc_difference_over_bound"], 1, True), fmt(x["ED_cc_difference_over_bound"], 1, True), fmt(x["rho_cc_dev_minus_prod"], 4, True),
            ]
            for x in decomp
        ],
    )  # fmt: skip

    def span(values: list[float], digits: int) -> str:
        lo_, hi_ = min(values), max(values)
        return fmt(lo_, digits) if len(values) == 1 else f"{fmt(lo_, digits)} to {fmt(hi_, digits)}"

    def through(source: str, rest: str, label: str) -> str:
        """The dates whose larger part is that of ``source``, with the bounds of both sides."""
        key = {"E_LC[D]": "ED_lc", "E_CC[D]": "ED_cc"}
        xs = [x for x in decomp if x["source"] == source]
        if not xs:
            return f"through {label} on none of the {len(decomp)} dates"
        zs = [x[f"{key[source]}_difference_over_bound"] for x in xs]
        way = "higher" if all(v > 0 for v in zs) else "lower" if all(v < 0 for v in zs) else "higher or lower"  # fmt: skip
        return (
            f"through {label} on {len(xs)} of the {len(decomp)} dates ({', '.join(x['date'] for x in xs)}: {source} is {way} at the development budget by {span([abs(v) for v in zs], 1)} bounds; "
            f"its part is {', '.join(fmt(x[f'part_from_{key[source]}'], 5, True) for x in xs)} against {', '.join(fmt(x[f'part_from_{key[rest]}'], 5, True) for x in xs)} from {rest}, which moves by {', '.join(fmt(x[f'{key[rest]}_difference_over_bound'], 1, True) for x in xs)} bounds)"
        )

    n_rho = int((cmp2[cmp2["quantity"] == "rho_cc"]["development_minus_production"] < 0).sum())
    sign_3m = "the opposite sign" if n_lower == len(dd) else "the same sign" if n_higher == len(dd) else "no single sign at one year"  # fmt: skip
    b.text(
        f"By that rule the difference of LC/CC runs {through('E_CC[D]', 'E_LC[D]', 'the companion')} and {through('E_LC[D]', 'E_CC[D]', 'E_LC[D]')}. "
        f"ρ_CC is lower at the development budget on {n_rho} of the {len(dd)} dates (the rows carry no error for ρ_CC). "
        f'At three months the frozen package\'s reader note 1 has "{note1}": at one year the differences are {fmt(dd.abs().min(), 4)} to {fmt(dd.abs().max(), 4)} in absolute value, with {sign_3m} (development {side} on {max(n_lower, n_higher)} of the {len(dd)}).'
        + (
            " LC/CC at one year is known to three decimals at most."
            if dd.abs().max() >= 0.0005
            else ""
        )
    )
    # --- (3) the summaries
    sm_rows, md_rows = [], []
    digits_of: dict[str, int] = {}
    deltas_skipped = 0
    care = {
        f"C_{t}_{s}"
        for t in ("125", "150")
        for s in ("lc", "cc", "cop", "lc_over_copula", "lc_over_cc")
    }
    by_date = p.set_index("date")
    # the frozen side like for like with sample (c): the frozen per-date records of the yearly
    # dates without those decision 2 flags — the flag read from the row files, and checked
    # against the frozen records' own count of the names kept unscreened
    frozen_flag2 = [x for x in data.yearly if data.rows[x].get("flag_unscreened") in (True, 1.0)]
    if frozen_flag2 != [
        x
        for x in data.yearly
        if ((frozen.get(f"1Y.dev.{x}.n_names_unscreened") or {}).get("value") or 0) > 0
    ]:
        raise ValueError("decision 2's flag in the row files is not the frozen records' count")
    like_dates = [x for x in data.yearly if x not in frozen_flag2]
    n_twenty, n_like = len(data.yearly), len(like_dates)

    def same_number(x: Any, y: Any) -> bool:
        if x is None or y is None:
            return x is None and y is None
        return abs(float(x) - float(y)) <= 1e-12 * max(1.0, abs(float(y)))

    for q in pm_1y.ALL:
        fa = frozen.get(f"1Y.dev.summary.{q.key}.mean.all")
        if fa is None or not q.summarise:
            continue
        if q.key.startswith("delta_"):
            deltas_skipped += 1
            continue
        fvals = pd.Series(
            {x: (frozen.get(f"1Y.dev.{x}.{q.key}") or {}).get("value") for x in data.yearly},
            dtype=float,
        )
        f_all = pm_1y.summarise(fvals)
        if not (
            f_all["n"] == fa["n"]
            and same_number(f_all["mean"], fa["value"])
            and same_number(f_all["se"], fa["se"])
        ):
            raise ValueError(
                f"{q.key}: the frozen summary mean is not reproduced from the frozen per-date records"
            )
        fl = pm_1y.summarise(fvals[like_dates])
        vals = pd.Series(
            [value_of(q.key, data.rows[x], data.entries.loc[x].to_dict())[0] for x in p["date"]],
            index=p["date"],
        )
        mb = pm_1y.summarise(vals)
        mc = pm_1y.summarise(vals[by_date["sample_c"]])
        ma = pm_1y.summarise(vals[by_date["sample_a"]])
        ma2 = pm_1y.summarise(vals[by_date["sample_a2"]])
        mo = pm_1y.summarise(vals[~by_date["yearly_date"]])
        fu = frozen.get(f"1Y.dev.summary.{q.key}.mean.unflagged")
        fmed = frozen.get(f"1Y.dev.summary.{q.key}.median.all")
        diff = Rules.differs(mb["mean"], mb["se"], fa["value"], fa["se"])
        z = (
            (mb["mean"] - fa["value"]) / math.hypot(mb["se"], fa["se"])
            if fa["se"] and mb["se"]
            else NAN
        )
        diff_c = Rules.differs(mc["mean"], mc["se"], fa["value"], fa["se"])
        z_c = (
            (mc["mean"] - fa["value"]) / math.hypot(mc["se"], fa["se"])
            if fa["se"] and mc["se"]
            else NAN
        )
        # sample (c) against the frozen dates without decision 2's: its dates out on both sides
        diff_l = Rules.differs(mc["mean"], mc["se"], fl["mean"], fl["se"])
        z_l = (
            (mc["mean"] - fl["mean"]) / math.hypot(mc["se"], fl["se"])
            if fl["se"] and mc["se"]
            else NAN
        )
        # the date that carries the mean of sample (b), if one does (ONE_DATE_SHARE)
        one_date, one_share = one_date_share(vals)
        carried = bool(finite(one_share) and one_share > ONE_DATE_SHARE)
        unresolved = bool(carried and mb["se"] and abs(mb["mean"]) <= K_SIGMA * mb["se"])
        digits_of[q.key] = q.digits
        # the same against the monthly dates that are not among the twenty (independent of them)
        diff_o = Rules.differs(mo["mean"], mo["se"], fa["value"], fa["se"])
        z_o = (
            (mo["mean"] - fa["value"]) / math.hypot(mo["se"], fa["se"])
            if fa["se"] and mo["se"]
            else NAN
        )
        mark = "‡" if q.key in care and "150" in q.key else "†" if q.key in care else ""
        sm_rows.append({
            "quantity": q.key, "label": q.label, "unit": q.unit, "frozen_mean_all": fa["value"], "frozen_se_all": fa["se"], "frozen_n_all": fa["n"], "monthly_mean_all": mb["mean"], "monthly_se_all": mb["se"], "monthly_n_all": mb["n"],
            "z": z, "reads_differently": diff, "monthly_mean_c": mc["mean"], "monthly_se_c": mc["se"], "monthly_n_c": mc["n"], "reads_differently_c": diff_c, "frozen_median_all": fmed["value"] if fmed else NAN, "monthly_median_all": mb["median"],
            "frozen_q25_all": (frozen.get(f"1Y.dev.summary.{q.key}.q25.all") or {}).get("value"), "monthly_q25_all": mb["q25"], "frozen_q75_all": (frozen.get(f"1Y.dev.summary.{q.key}.q75.all") or {}).get("value"), "monthly_q75_all": mb["q75"],
            "frozen_unflagged_value": fu["value"] if fu else NAN, "frozen_unflagged_n": fu["n"] if fu else 0, "monthly_unflagged_mean": ma["mean"], "monthly_unflagged_median": ma["median"], "monthly_unflagged_n": ma["n"],
            "monthly_unflagged_without_decision_2_mean": ma2["mean"], "monthly_unflagged_without_decision_2_median": ma2["median"], "monthly_unflagged_without_decision_2_n": ma2["n"],
            "monthly_mean_other_dates": mo["mean"], "monthly_se_other_dates": mo["se"], "monthly_n_other_dates": mo["n"], "z_other_dates": z_o, "reads_differently_other_dates": diff_o, "mark": mark,
        })  # fmt: skip
        sm_rows[-1].update({
            "z_c": z_c, "largest_one_date_b": one_date, "largest_one_date_share_b": one_share, "value_on_that_date": float(vals[one_date]) if one_date else NAN,
            "carried_by_one_date_b": carried, "unresolved_b": unresolved, "in_the_page_table": "150" not in q.key,
        })  # fmt: skip
        sm_rows[-1].update({
            "frozen_mean_without_decision_2": fl["mean"], "frozen_se_without_decision_2": fl["se"], "frozen_n_without_decision_2": fl["n"],
            "z_c_without_decision_2": z_l, "reads_differently_c_without_decision_2": diff_l,
        })  # fmt: skip
        if "150" in q.key:
            continue
        dg = q.digits
        md_rows.append([
            q.label + (" †" if mark else ""), UNIT_WORDS.get(q.unit, q.unit), fmt(fu["value"] if fu else NAN, dg), fmt(ma["mean"], dg), fmt(ma2["mean"], dg),
            pm(fa["value"], fa["se"], dg), pm(mb["mean"], mb["se"], dg), fmt(z, 1, True), "yes" if diff else "no", pm(mc["mean"], mc["se"], dg), fmt(z_c, 1, True), "yes" if diff_c else "no", z_cell(z_l), "yes" if diff_l else "no",
            fmt(fmed["value"] if fmed else NAN, dg), fmt(mb["median"], dg),
        ])  # fmt: skip
    sm = pd.DataFrame(sm_rows)
    b.save("frozen_vs_monthly_summary", sm)
    b.add(
        "frozen.summary.n_compared",
        "frozen summary means over the twenty yearly dates set beside the monthly pass's",
        len(sm),
        unit="quantities",
        notes="a count",
    )
    b.add(
        "frozen.summary.n_differ",
        f"of which the monthly mean is more than {K_SIGMA:g} times √(se_f² + se_m²) from the frozen one",
        int(sm["reads_differently"].fillna(False).sum()),
        unit="quantities",
        n=len(sm),
        notes="a count; the yearly dates are a subset of the monthly table, so the yardstick is conservative",
    )
    d_b = sm["reads_differently"].fillna(False).astype(bool)
    d_c = sm["reads_differently_c"].fillna(False).astype(bool)
    shown = sm["in_the_page_table"].astype(bool)
    b.add(
        "frozen.summary.n_differ_c",
        f"of which the monthly mean without decision 2's flagged dates (sample (c)) is more than {K_SIGMA:g} times √(se_f² + se_m²) from the frozen one",
        int(d_c.sum()),
        unit="quantities",
        n=len(sm),
        notes="a count; the frozen mean is over its twenty dates, one of which carries decision 2's flag",
    )
    b.add(
        "frozen.summary.n_verdicts_differ_b_c",
        "frozen summary means whose verdict is not the same on sample (b) and on sample (c)",
        int((d_b != d_c).sum()),
        unit="quantities",
        n=len(sm),
        notes="a count",
    )
    # the like-for-like comparison of sample (c): decision 2's dates out on the frozen side too
    d_l = sm["reads_differently_c_without_decision_2"].fillna(False).astype(bool)
    like_source = f"{rel(TABLE)}; {rel(FROZEN_1Y / 'numbers_1y.json')}"
    b.add(
        "frozen.summary.n_frozen_dates_decision_2",
        "the frozen addendum's yearly development dates that decision 2 flags (a name kept unscreened), the flag read from the row files",
        len(frozen_flag2),
        unit="dates",
        n=n_twenty,
        notes=f"a count: {dates_text(frozen_flag2)}; equal to the dates whose frozen record n_names_unscreened is positive",
        source=like_source,
    )
    b.add(
        "frozen.summary.n_differ_c_without_decision_2",
        f"frozen summary quantities whose monthly mean without decision 2's flagged dates (sample (c)) is more than {K_SIGMA:g} times √(se_f² + se_m²) from the frozen mean without decision 2's flagged dates",
        int(d_l.sum()),
        unit="quantities",
        n=len(sm),
        notes=f"a count; the frozen side is the mean and sd/√n of the frozen per-date records of its {n_like} dates that decision 2 does not flag (pm_1y.summarise), "
        + (
            f"fewer than the {MIN_N_SE} dates from which the page prints an across-dates error"
            if n_like < MIN_N_SE
            else f"at least the {MIN_N_SE} dates from which the page prints an across-dates error"
        ),
        source=like_source,
    )
    b.add(
        "frozen.summary.n_verdicts_differ_c_two_frozen_sides",
        "frozen summary quantities whose verdict on sample (c) is not the same against the frozen mean over all its dates and against the frozen mean without decision 2's flagged dates",
        int((d_c != d_l).sum()),
        unit="quantities",
        n=len(sm),
        notes="a count",
        source=like_source,
    )
    b.checks.append(
        f"The frozen addendum's {len(sm)} summary means set beside the monthly pass (value, standard error and n over its {n_twenty} dates) are reproduced to 12 significant digits from its own per-date records of `numbers_1y.json` with its own `pm_1y.summarise`, before the {count(len(frozen_flag2), 'date', 'dates')} decision 2 flags ({dates_text(frozen_flag2)}) {'is' if len(frozen_flag2) == 1 else 'are'} taken out for the like-for-like comparison of section 6 (3); the flag is read from the row files and falls on the dates whose frozen record `n_names_unscreened` is positive."
    )
    b.add(
        "frozen.summary.n_carried_by_one_date_b",
        f"frozen summary quantities whose mean on sample (b) is carried by one date (that date accounts for more than {100 * ONE_DATE_SHARE:.0f} % of the sum of the squared deviations from the median)",
        int(sm["carried_by_one_date_b"].sum()),
        unit="quantities",
        n=len(sm),
        notes="a count; the rows of the calls at 1.5 × P_D, which are in the CSV only, are counted",
    )
    carried_rows = sm[sm["carried_by_one_date_b"] & shown]
    carried_dates = sorted(set(carried_rows["largest_one_date_b"]))
    flagged_2 = set(p.loc[p["flag_unscreened"], "date"])
    b.text(
        f"**(3) The addendum's summaries (its table 3g) beside the monthly pass.** The frozen \"not flagged\" block is one date (2023-01-03); the monthly one is sample (a), the rule of decision 5, {int(p['sample_a'].sum())} dates, and (a2), the same without decision 2's flagged date, {int(p['sample_a2'].sum())} dates: these come first in the table and carry no error. "
        f"The frozen \"all priced dates\" block is 20 dates, 19 of them flagged; the monthly one is sample (b), {n_p} dates, {int(p['flag_clip'].sum())} of them flagged — both blocks contain flagged rows [{INCL}], and so do sample (c) and the medians. "
        f'Mean ± across dates (sd/√n) on both sides. z is the monthly mean minus the frozen one over √(se_f² + se_m²); "reads differently" is |z| > {K_SIGMA:g}. The frozen dates are a subset of the monthly ones, so z is not a test between independent samples: the CSV carries the same z against the {int((~p["yearly_date"]).sum())} monthly dates that are not among the twenty (columns `z_other_dates`, `reads_differently_other_dates`), and the sentence after the table says what changes. Medians and the unflagged blocks carry no error. The second column is the unit of the row. '
        f"Sample (c) is given beside (b), with its own z and its own verdict against the same frozen mean, because decision 2 asks for every summary with and without its flagged dates and because one of them (2009-06-01, section 4 (iii)) carries second moments of the order of 10⁴. "
        + (
            f"That frozen mean keeps {count(len(frozen_flag2), 'date that decision 2 flags', 'dates that decision 2 flags')} ({dates_text(frozen_flag2)}), so that against it decision 2's dates are out on the monthly side only. The two columns after those of sample (c) set the same monthly mean against the frozen dates without {'it' if len(frozen_flag2) == 1 else 'them'}, {n_like} dates, decision 2's dates out on both sides: their mean and sd/√n are recomputed from the frozen per-date records of `numbers_1y.json`, the flag being read from the row files (CSV columns `frozen_mean_without_decision_2`, `frozen_se_without_decision_2`, `z_c_without_decision_2`, `reads_differently_c_without_decision_2`). "
            + (
                f"{n_like} dates are fewer than the {MIN_N_SE} from which this page prints an across-dates error (section 0): that sd/√n is computed as the addendum computes its own (`pm_1y.summarise`), is used for this z and its verdict only, and is in the CSV, not printed here. "
                if n_like < MIN_N_SE
                else ""
            )
            if frozen_flag2
            else f"Decision 2 flags none of the frozen {n_twenty} dates: the two columns after those of sample (c) repeat its z and its verdict. "
        )
        + f"A mean of sample (b) is called carried by one date when that date accounts for more than {100 * ONE_DATE_SHARE:.0f} % of the sum, over the dates, of the squared deviations from the sample's median (`tables/frozen_vs_monthly_summary.csv`, columns `largest_one_date_b` and `largest_one_date_share_b`; `z_c` is the z of sample (c)): "
        + (
            f"of the {int(shown.sum())} rows of the table that is the case of {len(carried_rows)} — {', '.join(carried_rows['label'])} — and the date is {listing([x + (' (decision 2 flags it)' if x in flagged_2 else '') for x in carried_dates])}. "
            if len(carried_rows)
            else f"none of the {int(shown.sum())} rows of the table is. "
        )
        + f"† marks the calls at 1.25 × P_D (quoted with care: section 5); the rows of the calls at 1.5 × P_D are in the CSV only, each marked ‡: they are not results (section 5). The sticky-strike deltas ({deltas_skipped} rows of the frozen table) are not in the table: the monthly pass ran them on no other date than the twenty.",
    )
    b.table(
        [
            "quantity",
            "unit",
            "frozen not flagged (n 1): the value",
            f"monthly (a) (n {int(p['sample_a'].sum())}): mean, no error",
            f"monthly (a2) (n {int(p['sample_a2'].sum())}): mean, no error",
            f"frozen, all priced dates (n 20) [{INCL}]: mean ± across dates",
            f"monthly (b) (n {n_p}) [{INCL}]: mean ± across dates",
            "z, sample (b)",
            "reads differently, sample (b)",
            f"monthly (c) (n {int(p['sample_c'].sum())}) [{INCL}]: mean ± across dates",
            "z, sample (c)",
            "reads differently, sample (c)",
            f"z, sample (c) against the frozen dates without decision 2's (n {n_like})",
            f"reads differently, sample (c) against the frozen dates without decision 2's (n {n_like})",
            f"frozen median (n 20) [{INCL}]",
            f"monthly median, sample (b) (n {n_p}) [{INCL}]",
        ],
        md_rows,
    )
    flips = sm[sm["reads_differently"].fillna(False).astype(bool) != sm["reads_differently_other_dates"].fillna(False).astype(bool)]  # fmt: skip
    near = sm.assign(abs_z=sm["z_other_dates"].abs()).sort_values("abs_z", ascending=False).head(3)
    four = sm[sm["quantity"].isin([k for k, _, _ in FOUR_RATIOS])]
    if len(four) != len(FOUR_RATIOS):
        raise ValueError("the four ratios of section 2 are not all in the frozen summary table")
    n_four = int(four["reads_differently"].fillna(False).sum())
    n_other = int((~p["yearly_date"]).sum())
    b.add(
        "frozen.summary.n_verdicts_changed_other_dates",
        "frozen summary means whose verdict changes when the monthly mean is taken over the dates that are not among the twenty yearly ones",
        len(flips),
        unit="quantities",
        n=len(sm),
        notes=f"a count; the other dates are {n_other}",
    )

    def like_mean(r: Any) -> str:
        """The frozen mean without decision 2's dates as the page prints it: with its sd/√n only
        from ``MIN_N_SE`` dates on (the CSV has it either way)."""
        se = r["frozen_se_without_decision_2"] if n_like >= MIN_N_SE else None
        return pm(r["frozen_mean_without_decision_2"], se, digits_of[r["quantity"]])

    def moved(r: Any, skey: str) -> str:
        """A frozen mean and the monthly one of sample ``skey``, with its z (``l``: sample (c)
        against the frozen dates without decision 2's)."""
        dg = digits_of[r["quantity"]]
        mean, se, z = (r["monthly_mean_all"], r["monthly_se_all"], r["z"]) if skey == "b" else (r["monthly_mean_c"], r["monthly_se_c"], r["z_c_without_decision_2"] if skey == "l" else r["z_c"])  # fmt: skip
        was = like_mean(r) if skey == "l" else pm(r["frozen_mean_all"], r["frozen_se_all"], dg)
        return f"{r['label']} {was} → {pm(mean, se, dg)} (z {z:+.2f})"

    def sides_text(r: Any) -> str:
        """A mean whose verdict on sample (c) is not the same against the two frozen sides."""
        dg = digits_of[r["quantity"]]
        yes_on, no_on = (n_twenty, n_like) if d_c[r.name] else (n_like, n_twenty)
        return (
            f"{r['label']} reads differently against the frozen {yes_on} and not against the frozen {no_on} "
            f"(z {r['z_c']:+.2f} against the {n_twenty}, whose mean is {pm(r['frozen_mean_all'], r['frozen_se_all'], dg)}; z {r['z_c_without_decision_2']:+.2f} against the {n_like}, whose mean is {like_mean(r)})"
        )

    def at_threshold(rows: pd.DataFrame) -> str:
        """Whether the rows whose verdict changes between the two frozen sides sit at the
        threshold: each |z| within ``Z_AT_THRESHOLD`` of ``K_SIGMA``, on both readings."""
        zs = {r["label"]: [abs(float(r["z_c"])), abs(float(r["z_c_without_decision_2"]))] for _, r in rows.iterrows()}  # fmt: skip
        near_it = [k for k, v in zs.items() if all(abs(z - K_SIGMA) <= Z_AT_THRESHOLD for z in v)]
        far = [k for k in zs if k not in near_it]
        every = [z for v in zs.values() for z in v]
        one = len(zs) == 1
        band = f"within {Z_AT_THRESHOLD:g} of {K_SIGMA:g}"
        turns = f"the verdict of sample (c) on {'it' if one else 'them'} turns on whether the frozen side keeps {'the date' if len(frozen_flag2) == 1 else 'the dates'} decision 2 flags"
        if not far:
            return f"{'This row sits' if one else 'These rows sit'} at the threshold on both readings — {'its' if one else 'their'} |z| run from {min(every):.2f} to {max(every):.2f}, each {band} — and {turns}"
        return f"Of these, {len(near_it)} {'sits' if len(near_it) == 1 else 'sit'} at the threshold on both readings (each |z| {band}{': ' + listing(near_it) if near_it else ''}) and {len(far)} {'does' if len(far) == 1 else 'do'} not ({listing(far)}); {turns}"

    def one_sample(skey: str, mask: pd.Series) -> str:
        k = int(mask.sum())
        return f"{k} {'reads' if k == 1 else 'read'} differently by the rule" + (
            " (" + "; ".join(moved(r, skey) for _, r in sm[mask].iterrows()) + ")" if k else ""
        )

    def not_same(r: Any) -> str:
        yes_on, no_on = ("b", "c") if d_b[r.name] else ("c", "b")
        return f"{r['label']} reads differently on sample ({yes_on}) and not on sample ({no_on}) (z {r['z']:+.2f} on (b), {r['z_c']:+.2f} on (c))"

    def unresolved_text(r: Any) -> str:
        dg = digits_of[r["quantity"]]
        return f"{r['label']} ({pm(r['monthly_mean_all'], r['monthly_se_all'], dg)} on sample (b); {r['largest_one_date_b']} accounts for {100 * r['largest_one_date_share_b']:.1f} % of the squared deviations, its value there being {r['value_on_that_date']:.6g})"

    changed = sm[d_b != d_c]
    unresolved = sm[sm["unresolved_b"] & shown]
    both_ways = unresolved[unresolved.index.isin(changed.index)]
    # sample (c) against the two frozen sides: all the frozen dates, and those without decision 2's
    sides_changed = sm[d_c != d_l]
    open_rows = both_ways[both_ways.index.isin(sides_changed.index)]
    if len(open_rows) == 0:
        reads_others = "the verdict that reads the other dates is that of sample (c)" + (
            ", the same against the two frozen sides" if frozen_flag2 else ""
        )
    elif len(open_rows) == len(both_ways):
        reads_others = f"the verdict that reads the other dates is that of sample (c), which is not the same against the two frozen sides (above): neither sample settles {'it' if len(both_ways) == 1 else 'them'}"
    else:
        reads_others = f"the verdict that reads the other dates is that of sample (c), which is not the same against the two frozen sides for {listing(list(open_rows['label']))} (above): neither sample settles {'it' if len(open_rows) == 1 else 'them'}"
    b.text(
        f"Of the {len(sm)} means compared [{INCL}]: on sample (b), {one_sample('b', d_b)}; on sample (c), without decision 2's flagged dates, "
        + (
            f"against the frozen {n_twenty}, {count(len(frozen_flag2), 'of which decision 2 flags', 'of which decision 2 flags')} ({dates_text(frozen_flag2)}), {one_sample('c', d_c)}; on sample (c) against the frozen {n_like} without {'it' if len(frozen_flag2) == 1 else 'them'}, decision 2's dates out on both sides, {one_sample('l', d_l)}. "
            + (
                f"The verdict of sample (c) is not the same against the two frozen sides for {len(sides_changed)} of the {len(sm)} means: {'; '.join(sides_text(r) for _, r in sides_changed.iterrows())}. {at_threshold(sides_changed)}. "
                if len(sides_changed)
                else f"The verdict of sample (c) is the same against the two frozen sides for each of the {len(sm)} means. "
            )
            if frozen_flag2
            else f"{one_sample('c', d_c)} (decision 2 flags none of the frozen {n_twenty}: its dates are out on both sides). "
        )
        + (
            f"The verdict is not the same on samples (b) and (c), each against the frozen {n_twenty}, for {len(changed)} of the {len(sm)} means: {'; '.join(not_same(r) for _, r in changed.iterrows())}. "
            if len(changed)
            else f"The verdict is the same on samples (b) and (c), each against the frozen {n_twenty}, for each of the {len(sm)} means. "
        )
        + (
            f"On sample (b) {count(len(unresolved), 'mean of the table is', 'means of the table are')} not resolved — each is carried by one date by the rule above and is within {K_SIGMA:g} of its own errors of zero: {'; '.join(unresolved_text(r) for _, r in unresolved.iterrows())}. "
            + (
                f"For the {'row' if len(both_ways) == 1 else 'rows'} {listing(list(both_ways['label']))}, whose verdict is not the same on samples (b) and (c), the verdict of sample (b) therefore compares an unresolved mean with the frozen one, and {reads_others}."
                if len(both_ways)
                else "None of them changes its verdict between the two samples."
            )
            if len(unresolved)
            else "No mean of the table is unresolved on sample (b) by that rule."
        )
        + f" Against the {n_other} monthly dates that are not among the twenty, {'no verdict changes' if flips.empty else str(len(flips)) + ' of the ' + str(len(sm)) + ' verdicts change (' + ', '.join(flips['label']) + ')'}; the largest |z| there are {', '.join(f'{r.label} {r.z_other_dates:+.2f}' for r in near.itertuples())}."
        + f" Of the means of the four ratios of section 2 on sample (b), {n_four} read differently: LC/CC {pm(frozen['1Y.dev.summary.lc_over_cc.mean.all']['value'], frozen['1Y.dev.summary.lc_over_cc.mean.all']['se'])} (n 20) against {pm(stats[('lc_over_cc', 'b')]['mean'], stats[('lc_over_cc', 'b')]['se'])} (n {n_p}). The monthly errors are {fmt(frozen['1Y.dev.summary.lc_over_cc.mean.all']['se'] / stats[('lc_over_cc', 'b')]['se'], 1)} times smaller for LC/CC; for the ratios to the copula the Newey–West errors of section 2 are the ones to quote: {fmt(stats[('lc_over_copula', 'b')]['nw'])} for LC/copula and {fmt(stats[('cc_over_copula', 'b')]['nw'])} for CC/copula, against the frozen across-dates {fmt(frozen['1Y.dev.summary.lc_over_copula.mean.all']['se'])} and {fmt(frozen['1Y.dev.summary.cc_over_copula.mean.all']['se'])}."
    )
    # --- (4) the sentences
    statements: list[dict[str, Any]] = []

    def state(
        kind: str, quote: str, monthly: str, differs: bool, where: str = "Read this first"
    ) -> None:
        statements.append(
            {
                "where": where,
                "frozen_sentence": quoted(page, quote),
                "monthly_reading": monthly,
                "reads_differently": bool(differs),
                "kind": kind,
            }
        )

    f_corr = frozen["1Y.dev.level.corr_cc_over_copula_vs_atm_gap"]["value"]
    cc = p["cc_over_copula"]
    lo, hi = p.loc[cc.idxmin()], p.loc[cc.idxmax()]
    hi_lc = p.loc[p["lc_over_copula"].idxmax()]
    n_out, n_out_ok = int(p["gate_outside"].sum()), int(
        (p["gate_outside"] & (p["status"] == "ok")).sum()
    )
    n_flag = int(p["flag_clip"].sum())
    unflagged = list(p.loc[p["sample_a"], "date"])
    state(
        "the sample grows; the statement holds",
        "At this writing the wing binds, and the gate is waived, on 23 of the 24 priced rows; the gate's two numbers are outside 0.15 vp on 9 of the 24",
        f"The wing binds, and the gate is waived, on {n_flag} of the {n_p} priced dates; the gate's two numbers are outside {GATE_VP} vp on {n_out} of the {n_p} ({100 * n_out / n_p:.0f} % against {100 * 9 / 24:.0f} %), of which {n_out_ok} read ok.",
        n_out / n_p > 2 * 9 / 24 or n_out / n_p < 0.5 * 9 / 24,
    )
    state(
        "measured on one date; now on four",
        "At 12m, development minus production in LC/CC on the dates priced at both budgets — 2026-10-02: -0.00057",
        f"{len(dd)} dates are now priced at both budgets: development minus production in LC/CC is "
        + ", ".join(
            f"{fmt(b.get(f'frozen.budget.{x}.lc_over_cc_dev_minus_prod'), 5, True)} ({x})"
            for x in pc.REFERENCE_DATES
        )
        + f"; development is lower on {int((dd < 0).sum())} of the {len(dd)}, by up to {fmt(dd.abs().max(), 5)}.",
        len(dd) != 1,
    )
    state(
        "the statement changes",
        "The summary the owner defined (flagged rows out) has 1 date; 19 of the 20 priced dates are flagged for the clipped mass. It is one observation, not a summary",
        f"The summary the owner defined has {len(unflagged)} dates ({dates_text(unflagged)}); {n_flag} of the {n_p} priced dates are flagged. It is {len(unflagged)} observations: a mean and a median are printed for it, without a standard error (fewer than {MIN_N_SE} dates).",
        len(unflagged) != 1,
    )
    state(
        "the range widens; the relation holds",
        "CC/copula runs from 0.962 (2016-01-04: target minus study +1.04 vp) to 1.077 (2007-01-08: -1.65 vp) over the 20 dates",
        f"CC/copula runs from {fmt(lo['cc_over_copula'], 3)} ({lo['date']}: target minus study {fmt(lo['atm_gap'], 2, True)} vp; decision 2's flag) to {fmt(hi['cc_over_copula'], 3)} ({hi['date']}: {fmt(hi['atm_gap'], 2, True)} vp; index target extrapolated) over the {n_p} priced dates [{INCL}]; without decision 2's dates from {fmt(p.loc[p['sample_c'], 'cc_over_copula'].min(), 3)}.",
        cc.min() < 0.962 - 0.01 or cc.max() > 1.077 + 0.01,
        "Table 3a-2",
    )
    state(
        "the number changes; the relation holds",
        "the correlation across the 20 dates between CC/copula and (target ATM vol minus the study's) is -0.92",
        f"The correlation is {fmt(b.get('level.corr.b'), 2, True)} across the {n_p} priced dates and {fmt(b.get('level.corr.c'), 2, True)} without decision 2's flagged dates (n {int(p['sample_c'].sum())}) [{INCL}].",
        abs(b.get("level.corr.b") - f_corr) > 0.05,
        "Table 3a-2",
    )
    state(
        "the statement changes",
        "its CC/copula 1.077 and LC/copula 1.043 are the highest of the 20 development dates",
        f"On the monthly dates the highest CC/copula is {fmt(hi['cc_over_copula'], 3)} ({hi['date']}) and the highest LC/copula {fmt(hi_lc['lc_over_copula'], 3)} ({hi_lc['date']}); 2007-01-08 is {ordinal(int((cc > float(p.set_index('date').loc['2007-01-08', 'cc_over_copula'])).sum()) + 1)} of {n_p} for CC/copula [{INCL}].",
        hi["date"] != "2007-01-08",
        "Section 4a",
    )
    n_ext = int(truth(p["index_extrapolated"]).sum())
    n_chk = int((p["status"] == "check").sum())
    state(
        "the statement changes",
        "a name kept unscreened: 1 (2009-01-05); a name beyond its last kept expiry: 4 (2009-01-05, 2016-01-04, 2023-01-03, 2024-01-02); DJX target repaired: 8 (2007-01-08, 2015-01-05, 2016-01-04, 2020-01-06, 2021-01-04, 2023-01-03, 2025-01-06, 2026-10-02); DJX target extrapolated: none; status check: 2 (2007-01-08, 2016-01-04)",
        f"On the {n_p} priced dates — a name kept unscreened: {int(p['flag_unscreened'].sum())}; a name beyond its last kept expiry: {int((p['n_names_extrapolated'] > 0).sum())}; DJX target repaired: {int((p['n_dropped_calendar_index'] > 0).sum())}; DJX target extrapolated: {n_ext} ({dates_text(sorted(p.loc[truth(p['index_extrapolated']), 'date']))}), no longer none; status check: {n_chk}, of which {int((~truth(p['check_forward'])).sum())} FAIL `check_forward` — a check that fails on no yearly date.",
        n_ext > 0 or int((~truth(p["check_forward"])).sum()) > 0,
        "Section 3g",
    )
    se_ratio = {
        x: float(prod_table.loc[x, "kappa_lc_se"]) / float(dev_table.loc[x, "kappa_lc_se"])
        for x in pc.REFERENCE_DATES
    }
    others = [x for x in pc.REFERENCE_DATES if x != TODAY]
    for x, v in se_ratio.items():
        b.add(
            f"frozen.budget.{x}.kappa_lc_se_prod_over_dev",
            "κ_LC: the production row's Monte Carlo error over the development row's, one year",
            v,
            date=x,
            unit="ratio",
            budget=f"{pc.BUDGETS['production']} over {BUDGET}",
            source=f"{rel(TABLE_PROD)}; {rel(TABLE)}",
            notes="a ratio of two errors: no error of its own; 0.5 is what four times the paths give when the error falls as 1/√paths",
        )
    state(
        "measured on one date; holds there, not on the other three",
        "On these quantities, production against development: κ_LC ± 0.0028 against ± 0.00136 (ratio 2.06)",
        f"The same ratio of the two rows' errors of κ_LC, now on the four dates priced at both budgets: {', '.join(f'{v:.2f} ({x})' for x, v in se_ratio.items())}. On {TODAY} the error does not fall with the budget, as the addendum says; on {sum(1 for x in others if se_ratio[x] < 0.75)} of the other {len(others)} dates it falls as 1/√paths (0.50 for four times the paths), so the addendum's \"{quoted(page, 'Their ± is not a usable error.')}\" rests on {TODAY}: on the other dates priced at both budgets this test does not show it.",
        any(se_ratio[x] < 0.75 for x in others),
    )
    far_below = p[(p["T"] - p["nearest_kept_below"]) > 0.5]
    none_below = p[p["nearest_kept_below"].isna()]
    state(
        "the list grows",
        "Dates whose nearest kept DJX slice below the horizon is more than 0.5y before it (nearest kept slice below): 2023-01-03 (0.200y), 2024-01-02 (0.047y), 2025-01-06 (0.030y), 2026-10-02 (0.460y).",
        f"Read from the rows (the slices of `index_errors` before the horizon): {len(far_below)} of the {n_p} priced dates have their nearest kept DJX slice more than 0.5y before the horizon, and {len(none_below)} more have no kept slice before the horizon at all ({dates_text(sorted(none_below['date']))}); `tables/by_date.csv` has `nearest_kept_below` for every date.",
        len(far_below) / n_p > 2 * 4 / 20 or len(far_below) / n_p < 0.5 * 4 / 20,
    )
    spec = read_spec("spec_by_date")
    if spec is not None:
        sp = spec[spec["row_priced"]]
        wide = sp[(sp["nearest_kept_above"] - sp["nearest_kept_below"]) > 0.75]
        state(
            "the list grows",
            "Dates whose two nearest kept DJX slices around the horizon are more than 0.75y apart (below / above): 2020-01-06 (0.951y / 1.948y), 2023-01-03 (0.200y / 1.964y), 2024-01-02 (0.047y / 1.466y), 2025-01-06 (0.030y / 1.447y), 2017-04-03 (0.701y / 1.718y).",
            f"From this script's specification-only build: {len(wide)} of the {len(sp)} priced dates built ({100 * len(wide) / len(sp):.0f} % against 5 of the addendum's 23 distinct dates); `tables/spec_by_date.csv` lists them.",
            len(wide) / len(sp) > 2 * 5 / 23 or len(wide) / len(sp) < 0.5 * 5 / 23,
        )
    state(
        "still true",
        "It is not measured at 12m.",
        "The share of the high-strike calls carried by runaway paths is still not measured at one year: the monthly pass stored no paths (section 5).",
        False,
    )
    slope_holds = abs(b.get("level.slope.b")) > K_SIGMA * b.se("level.slope.b")
    state(
        "holds on the monthly dates" if slope_holds else "not confirmed on the monthly dates",
        "Only LC/CC is a local-correlation effect; LC/copula and CC/copula also carry the gap between the model's index target and the study's index level.",
        f"CC/copula moves by {fmt(b.get('level.slope.b'), 4, True)} per vol point of gap (HC1 {fmt(b.se('level.slope.b'))}; n {n_p}) [{INCL}]; the gap is the target at the forward minus the study's vol at the spot"
        + (
            f", and read at the same strike it is beyond {GAP_FAR} vp on {int(b.get('level.gap_same_strike.n_far'))} dates against {int(b.get('level.n_far'))} (section 4 (i))."
            if "L12M.level.gap_same_strike.n_far" in b.records
            else " (section 4 (i))."
        ),
        not slope_holds,
        "Tables 2a-2 and 3a-2",
    )
    st = pd.DataFrame(statements)
    b.save("frozen_statements", st)
    b.add(
        "frozen.statements.n",
        "sentences of the frozen addendum set beside the monthly pass",
        len(st),
        unit="sentences",
        notes="a count",
    )
    b.add(
        "frozen.statements.n_differ",
        "of which read differently on the monthly dates",
        int(st["reads_differently"].sum()),
        unit="sentences",
        n=len(st),
        notes="a count; the rule of each sentence is in scripts/pm_later_12m_monthly.py::section_frozen",
    )
    b.text(
        f'**(4) The addendum\'s sentences on the monthly dates.** Each frozen sentence below is quoted word for word (looked up in `{rel(page)}` before printing). "Reads differently" is decided by a rule per sentence: a count or a list that is no longer the same statement ("none", "one observation", "the highest"), a proportion that changes by more than a factor of two, a correlation that moves by more than 0.05, a range that widens by more than 0.01. None of these is an error in a frozen number: each frozen sentence is right for the twenty yearly dates or the four reference dates it names.',
    )
    b.table(
        [
            "where in the addendum",
            "the frozen sentence",
            "on the monthly dates",
            "reads differently",
            "kind",
        ],
        [
            [
                r["where"],
                f"\"{r['frozen_sentence']}\"",
                r["monthly_reading"],
                "**yes**" if r["reads_differently"] else "no",
                r["kind"],
            ]
            for r in statements
        ],
    )
    b.text(
        'The other parts of the addendum\'s "Read this first" are not re-read here because the monthly pass adds nothing to them: the before/after of decision 5 on 2026-10-02 (it needs the old-defaults row, which exists for that date only), the sticky-strike deltas (run on the twenty yearly dates only), and the definitions of the marks and of the units.',
        f"**An error in a frozen number?** None was found. The per-date development records are reproduced from the row files to 12 significant digits; the summaries, counts and sentences that change do so because the sample is {n_p} dates instead of 20, or four dates at two budgets instead of one, not because a frozen number is wrong for what it says it is. Nothing is to be written in `ERRATA.md` on account of this note.",
    )
    if diffs:
        raise ValueError(
            f"{len(diffs)} frozen per-date development records are not reproduced: report them before anything else"
        )
    return [
        f"{r['where']}: \"{r['frozen_sentence']}\" → {r['monthly_reading']}"
        for r in statements
        if r["reads_differently"]
    ]


# ----------------------------------------------------------------------------- section 7
def section_two_years(b: Book) -> None:
    table = pd.read_parquet(TABLE_24)
    rows24 = {x.stem: json.loads(x.read_text()) for x in sorted(ROWS_24.glob("*.json"))}
    entries = pm_1y.study_entries("24m")
    d = derive(table, rows24, entries, "24m")
    caveat = quoted(
        REPORT_24,
        "Data caveat of the study (outputs/dispersion/PROGRESS_Q2.md) [measured there]: a name is priced beyond its last listed expiry on 226 of 972 dates at 12m and on 804 of 919 at 24m (28 of 30 names in the median at 24m), so the 24m numbers extrapolate the 12-18m smiles; DJX long-dated expiries at 1.5-1.8 times their neighbours on six dates (four in 2023, two in 2007) touch 12m and 24m.",
    )
    progress = (pc.STUDY / "PROGRESS_Q2.md").read_text()
    in_source = "226 of 972" in progress and "804 of 919" in progress
    commit = ", ".join(sorted(set(d["git_commit"])))
    kw = {"tenor": "24m", "commit": commit, "source": rel(TABLE_24)}
    md = []
    for _, r in d.iterrows():
        date = r["date"]
        for key, label, unit in (
            ("ED_lc", "E_LC[D]", "notional"),
            ("ED_cc", "E_CC[D]", "notional"),
        ):
            b.add(
                f"24m.{date}.{key}",
                f"indicative: {label}, two years",
                r[key],
                r[f"{key}_se"],
                date=date,
                unit=unit,
                notes="indicative (owner's decision 5); Monte Carlo error of the date",
                **kw,
            )
        for key, label in (
            ("lc_over_cc", "LC/CC"),
            ("lc_over_copula", "LC/copula"),
            ("cc_over_copula", "CC/copula"),
        ):
            b.add(
                f"24m.{date}.{key}",
                f"indicative: {label}, two years",
                r[key],
                r[f"{key}_se"],
                date=date,
                unit="ratio",
                notes="indicative (owner's decision 5); Monte Carlo error of the date",
                **kw,
            )
        for key, label in (
            ("clip_low_inner_max", "mass clipped at λ = 0"),
            ("clip_high_inner_max", "mass clipped at the cap"),
        ):
            b.add(
                f"24m.{date}.{key}",
                f"indicative: {label}, two years",
                r[key],
                date=date,
                unit="fraction of the particles",
                notes="indicative; a calibration diagnostic: no standard error",
                **kw,
            )
        fails = [g for g in (*GATING, "check_names") if not (r[g] is True or r[g] == 1.0)]
        md.append([
            date, r["status"], pm(r["ED_lc"], r["ED_lc_se"], 6), pm(r["ED_cc"], r["ED_cc_se"], 6), pm(r["lc_over_cc"], r["lc_over_cc_se"], 5), pm(r["lc_over_copula"], r["lc_over_copula_se"], 5), pm(r["cc_over_copula"], r["cc_over_copula_se"], 5),
            fmt(r["clip_low_inner_max"]), fmt(r["clip_high_inner_max"]), ("yes: " + " and ".join(k for k, v in (("λ = 0", r["flag_clip_low"]), ("the cap", r["flag_clip_high"])) if v)) if r["flag_clip"] else "no", yes(r["flag_unscreened"]),
            f"{int(r['n_names_extrapolated'])} of 30", yes(r["index_extrapolated"]) + f" (last kept slice {r['index_last_slice']:.2f}y)", f"{int(r['n_dropped_calendar_index'])}", ("FAIL: " + ", ".join(f + (f" ({r['nonfinite']})" if f == "check_no_nan" else " (a diagnostic, not a gate)" if f == "check_names" else "") for f in fails)) if fails else "all pass",
        ])  # fmt: skip
    b.save(
        "two_years_indicative",
        d[
            [
                "date",
                "status",
                "reason",
                "indicative",
                "ED_lc",
                "ED_lc_se",
                "ED_cc",
                "ED_cc_se",
                "lc_over_cc",
                "lc_over_cc_se",
                "lc_over_copula",
                "lc_over_copula_se",
                "cc_over_copula",
                "cc_over_copula_se",
                "clip_low_inner_max",
                "clip_high_inner_max",
                "flag_clip_low",
                "flag_clip_high",
                "flag_clip",
                "flag_unscreened",
                "n_names_extrapolated",
                "index_extrapolated",
                "index_last_slice",
                "n_dropped_calendar_index",
                *GATING,
                "check_names",
                "nonfinite",
                "target_atm",
                "study_atm",
                "atm_gap",
            ]
        ],
    )
    n_flag = int(d["flag_clip"].sum())
    b.text(
        "## 7. Two years, indicative",
        f'The owner asked for two years "on today and the three reference dates only, labelled indicative". `{rel(TABLE_24)}` has {len(d)} rows, all with `indicative = true`, at the development budget ({BUDGET}), commit {commit}. **Every number of this section is indicative.** Each ± is the Monte Carlo error of the date; with four dates no mean and no error across dates is printed.',
    )
    b.table(
        [
            "date (all indicative)",
            "status",
            "E_LC[D]",
            "E_CC[D]",
            "LC/CC",
            "LC/copula",
            "CC/copula",
            "clipped at λ = 0",
            "clipped at the cap",
            "clip flag (decision 5)",
            "unscreened (decision 2)",
            "names priced beyond their last kept expiry",
            "index target extrapolated",
            "DJX slices dropped by the repair",
            "checks",
        ],
        md,
    )
    b.text(
        f"All {n_flag} of the {len(d)} rows carry decision 5's clip flag: by the owner's rule no two-year row is in a summary, and none is made. On {int((d['n_names_extrapolated'] == 30).sum())} of the {len(d)} dates all 30 names are priced beyond their last kept expiry; on {TODAY} {'none is' if int(d.set_index('date').loc[TODAY, 'n_names_extrapolated']) == 0 else str(int(d.set_index('date').loc[TODAY, 'n_names_extrapolated'])) + ' are'}. "
        f'The study\'s own caveat, as the automatic report `{rel(REPORT_24)}` quotes it: "{caveat}" '
        f"Its source is the study's `outputs/dispersion/PROGRESS_Q2.md` ({'both counts are found there' if in_source else 'the counts were NOT found there'}); it is the study's measurement on the study's dates and is not recomputed here. The automatic report's two-year means over the {len(d)} dates (for example LC/CC {fmt(d['lc_over_cc'].mean())}) are means of flagged, indicative rows and are not quoted.",
    )


# ----------------------------------------------------------------------------- section 8
def against_three_months(
    b: Book, p: pd.DataFrame, t3: pd.DataFrame, key: str, side: str, budget: str, source: Path
) -> dict[str, Any]:
    """LC/CC at one year minus LC/CC at three months, paired on the dates priced in both passes
    (``t3``: the priced rows of a three-month pass, by date), on four samples: every common date
    (``all``); without the dates decision 2 flags at either horizon (``c``); the dates decision
    5's condition flags at neither horizon (``a``: the three-month side recomputed from the row's
    two clipped masses, the development pass of the frozen history having no flag column); and
    those without decision 2's flagged dates (``a2``).  Adds the records of ``c``, ``a`` and
    ``a2`` (the record of ``all`` is the caller's) and returns the statistics."""
    one = p.set_index("date")
    common = [x for x in p["date"] if x in t3.index]
    diff = pd.Series(
        one.loc[common, "lc_over_cc"].to_numpy() - t3.loc[common, "ratio"].to_numpy(), index=common
    )
    flag2_one = set(one.index[one["flag_unscreened"]])
    flag2_three = set(t3.index[t3["n_names_unscreened"] > 0])
    flag2 = flag2_one | flag2_three
    if t3[["clip_low_inner_max", "clip_high_inner_max"]].isna().any().any():
        raise ValueError(f"{side}: a priced three-month row carries no clipped mass")
    clip3 = (t3["clip_low_inner_max"] > CLIP_FLAG) | (t3["clip_high_inner_max"] > CLIP_FLAG)
    # a pass that carries decision 5's flag column must have it equal to the recomputed condition
    if "flag_clip" in t3.columns and not (truth(t3["flag_clip"]) == clip3).all():
        raise ValueError(f"{side}: decision 5's recomputed condition is not the rows' flag_clip")
    neither = [x for x in common if one.loc[x, "sample_a"] and not clip3[x]]
    dates = {
        "all": common,
        "c": [x for x in common if x not in flag2],
        "a": neither,
        "a2": [x for x in neither if x not in flag2],
    }
    out: dict[str, Any] = {
        "side": side,
        "short": side.split(" budget")[0],
        "dates": dates,
        "diff": diff,
        "n_flag2_common": len([x for x in common if x in flag2]),
        "flag2_same": {x for x in common if x in flag2_one}
        == {x for x in common if x in flag2_three},
        "a_is_sample_a": set(neither) == set(one.index[one["sample_a"]]),
        "n_three": len(t3),
        "n_flagged_three": int(clip3.sum()),
        "n_unflagged_three": int((~clip3).sum()),
    }
    labels = {
        "c": "without the dates decision 2 flags at either horizon, flagged rows included",
        "a": "the dates decision 5's condition flags at neither horizon (the rule of decision 5)",
        "a2": "the dates decision 5's condition flags at neither horizon, without decision 2's flagged dates",
    }
    for skey, xs in dates.items():
        s = summarise(diff[xs])
        out[skey] = s
        if skey == "all":
            continue
        b.add(
            f"{key}.{skey}",
            f"LC/CC at one year minus LC/CC at three months ({side}), mean over the dates priced in both passes: {labels[skey]}",
            s.get("mean"),
            s.get("se"),
            unit="ratio",
            n=int(s["n"]),
            budget=budget,
            source=f"{rel(TABLE)}; {rel(source)}",
            notes=(
                f"standard error across dates, sd/√n; Newey–West ({NW_LAGS} lags) {s['nw']:.5f}"
                if finite(s.get("nw"))
                else no_se_reason(int(s["n"]))
            ),
        )
    return out


def section_said(
    b: Book, data: Data, stats: dict[tuple[str, str], dict[str, float]], clip: dict[str, Any]
) -> str:
    p = data.priced
    n_p = len(p)
    sb, sa = stats[("lc_over_cc", "b")], stats[("lc_over_cc", "a")]
    sa2, sc = stats[("lc_over_cc", "a2")], stats[("lc_over_cc", "c")]
    sa_ok, sa2_ok = stats[("lc_over_cc", "a_ok")], stats[("lc_over_cc", "a2_ok")]
    # three months: the frozen record, and the paired difference on the common dates
    hist = {r["id"]: r for r in json.loads(HIST_FROZEN.read_text())}
    f3 = hist["C.hist.lc_over_cc.mean.all"]
    f3_nw = hist["C.hist.lc_over_cc.mean_nw_se_12.all"]["value"]
    t3 = pd.read_parquet(TABLE_3M_FROZEN)
    t3 = t3[t3["status"] != "failed"].sort_values("date").set_index("date")
    if abs(float(t3["ratio"].mean()) - f3["value"]) > 1e-12 or len(t3) != f3["n"]:
        raise ValueError("the frozen 3m mean of LC/CC is not reproduced from its table")
    b.checks.append(
        f"The frozen three-month mean of LC/CC (record `C.hist.lc_over_cc.mean.all`: {f3['value']:.5f}, n {f3['n']}) is reproduced from `{rel(TABLE_3M_FROZEN)}`."
    )
    dev_budget = f"{BUDGET}, both sides"
    dev = against_three_months(
        b, p, t3, "said.lc_over_cc_12m_minus_3m.mean", "development budget, the frozen history's table", dev_budget, TABLE_3M_FROZEN
    )  # fmt: skip
    sd = dev["all"]
    b.add(
        "said.lc_over_cc_12m_minus_3m.mean",
        "LC/CC at one year minus LC/CC at three months, mean over the dates priced in both passes (the three-month side: the frozen history's table)",
        sd["mean"],
        sd["se"],
        unit="ratio",
        n=int(sd["n"]),
        budget=dev_budget,
        commit=f"{data.commit} (one year); {f3['commit']} (three months: decisions 1–2 on, decision 5 not yet in the code)",
        source=f"{rel(TABLE)}; {rel(TABLE_3M_FROZEN)}",
        notes=f"standard error across dates, sd/√n; Newey–West ({NW_LAGS} lags) {sd['nw']:.5f}; flagged rows included on both sides",
    )
    budget_3m = quoted(
        README_3M, "LC/CC is lower at the production budget on all 184, by 0.0018 on average"
    )
    note4 = quoted(READER_NOTES, "which is the case on 201 of the 216 priced dates of the history")
    # reader note 4's count of the three-month development pass against the recomputed condition
    if [int(x) for x in re.findall(r"\d+", note4)] != [dev["n_flagged_three"], dev["n_three"]]:
        raise ValueError(
            f"reader note 4 ({note4!r}) is not decision 5's recomputed condition: {dev['n_flagged_three']} of {dev['n_three']}"
        )
    prod: dict[str, Any] | None = None
    s3: dict[str, float] = {}
    s_bud: dict[str, float] = {}
    n_bud_above = 0
    if TABLE_3M_PROD.exists() and HIST_PROD.exists():
        p3 = pd.read_parquet(TABLE_3M_PROD)
        p3 = p3[p3["status"] != "failed"].sort_values("date").set_index("date")
        s3 = summarise(p3["ratio"])
        b.add(
            "said.lc_over_cc_3m_production.mean",
            "LC/CC at three months, production budget (the pass run after the freeze): mean over its priced dates, flagged rows included",
            s3["mean"],
            s3["se"],
            tenor="3m",
            unit="ratio",
            n=int(s3["n"]),
            budget=pc.BUDGETS["production"],
            source=rel(TABLE_3M_PROD),
            notes=f"standard error across dates, sd/√n; Newey–West ({NW_LAGS} lags) {s3['nw']:.5f}",
        )
        two_budgets = f"{BUDGET} minus {pc.BUDGETS['production']}"
        prod = against_three_months(
            b, p, p3, "said.lc_over_cc_12m_minus_3m_production.mean", "production budget, the pass run after the freeze", two_budgets, TABLE_3M_PROD
        )  # fmt: skip
        if int(s3["n"]) != prod["n_three"] or int(f3["n"]) != dev["n_three"]:
            raise ValueError("a three-month mean of LC/CC is not over the pass's priced dates")
        sd2 = prod["all"]
        b.add(
            "said.lc_over_cc_12m_minus_3m_production.mean",
            "LC/CC at one year (development budget) minus LC/CC at three months (production budget, after the freeze), mean over the dates priced in both",
            sd2["mean"],
            sd2["se"],
            unit="ratio",
            n=int(sd2["n"]),
            budget=two_budgets,
            source=f"{rel(TABLE)}; {rel(TABLE_3M_PROD)}",
            notes=f"standard error across dates; Newey–West ({NW_LAGS} lags) {sd2['nw']:.5f}; the two sides are at different budgets: at three months LC/CC is lower at the production budget by 0.0018 on average (later/production_3m/README.md)",
        )
        # the two three-month passes on the dates of the comparison: development minus production
        if prod["dates"]["all"] != dev["dates"]["all"]:
            raise ValueError("the two three-month passes do not share the one-year pass's dates")
        both3 = dev["dates"]["all"]
        bud = pd.Series(t3.loc[both3, "ratio"].to_numpy() - p3.loc[both3, "ratio"].to_numpy(), index=both3)  # fmt: skip
        s_bud, n_bud_above = summarise(bud), int((bud > 0).sum())
        if abs(s_bud["mean"] - (sd2["mean"] - sd["mean"])) > 1e-12:
            raise ValueError("the two comparisons do not differ by the three-month difference")
        b.add(
            "said.budget_effect_3m.mean",
            "LC/CC at three months, the development pass of the frozen history minus the production pass run after the freeze: mean over the dates of the comparison with one year",
            s_bud["mean"],
            s_bud["se"],
            tenor="3m",
            unit="ratio",
            n=int(s_bud["n"]),
            budget=f"{BUDGET} minus {pc.BUDGETS['production']}",
            commit=f"{f3['commit']} (development); {data.commit} (production)",
            source=f"{rel(TABLE_3M_FROZEN)}; {rel(TABLE_3M_PROD)}",
            notes=f"standard error across dates, sd/√n; Newey–West ({NW_LAGS} lags) {s_bud['nw']:.5f}; the two passes differ by their budget and by their commit; it is the difference between the two comparisons with one year",
        )
    fb = clip["fits"][("c_trim", "lc_over_cc")]
    bins = clip["bins"]
    cap0 = CLIP_RESTRICT[0]
    r20 = clip["restricted"][(f"le_{cap0:g}", "b")]
    r20c = clip["restricted"][(f"le_{cap0:g}", "c")]
    fl, fl_c = clip["restricted"][(f"le_{cap0:g}_flagged", "b")], clip["restricted"][(f"le_{cap0:g}_flagged", "c")]  # fmt: skip
    yr, yr_c = clip["restricted"][(f"le_{cap0:g}_years", "b")], clip["restricted"][(f"le_{cap0:g}_years", "c")]  # fmt: skip
    n_flag = int(p["flag_clip"].sum())
    dd = [b.get(f"frozen.budget.{x}.lc_over_cc_dev_minus_prod") for x in pc.REFERENCE_DATES]
    mean_dd = float(np.mean(dd))
    n_dev_lower = sum(1 for x in dd if x < 0)
    b.add(
        "said.budget_effect_12m.mean",
        "LC/CC at the development budget minus LC/CC at the production budget, one year: mean over the dates priced at both",
        mean_dd,
        unit="ratio",
        n=len(dd),
        notes=f"a mean of {len(dd)} differences: no standard error on {len(dd)} dates",
    )

    words = {"above": "above zero", "below": "below zero", "not distinguishable": "not distinguishable from zero", "": "without an error"}  # fmt: skip

    def with_and_without(c: dict[str, Any]) -> str:
        """The verdict of a comparison at ``NW_LAGS`` lags with and without decision 2's flagged
        dates, and where it is not the same at the other lags."""
        sides = {k: Rules.side(c[k]["mean"], 0.0, c[k]) for k in ("all", "c")}
        base = (
            f"{words[sides['all']]}, with and without decision 2's flagged dates"
            if sides["all"] == sides["c"]
            else f"{words[sides['all']]} with decision 2's flagged dates and {words[sides['c']]} without them"
        )
        notes = []
        for k, label in (("all", "with decision 2's flagged dates"), ("c", "without them")):
            other = [lags for lags in NW_OTHER if Rules.side(c[k]["mean"], 0.0, c[k], lags) != sides[k]]  # fmt: skip
            if other:
                notes.append(
                    f"{label} the verdict is not the same at {listing([str(x) for x in other])} lags"
                )
        return base + (f" ({'; '.join(notes)})" if notes else f" (the same verdict at {listing([str(x) for x in sorted({NW_LAGS, *NW_OTHER})])} lags)")  # fmt: skip

    def two_samples(c: dict[str, Any]) -> str:
        """A comparison on every common date and without decision 2's flagged dates, each with
        its verdict at every number of lags."""
        x, y = c["all"], c["c"]
        return (
            f"{pm(x['mean'], x['se'], 5, True)} across dates (Newey–West {fmt(x['nw'], 5)}) on the {int(x['n'])} common dates — {Rules.at_each_lag(x['mean'], 0.0, x)} — "
            f"and {pm(y['mean'], y['se'], 5, True)} (Newey–West {fmt(y['nw'], 5)}) on the {int(y['n'])} without the {c['n_flag2_common']} that decision 2 flags — {Rules.at_each_lag(y['mean'], 0.0, y)}"
        )

    def owner_rule(*cs: dict[str, Any]) -> str:
        """The same differences on the dates decision 5's condition flags at neither horizon."""
        first = cs[0]
        who = (
            f"on the {int(first['a']['n'])} dates of sample (a), which decision 5's condition (recomputed from the three-month rows' two clipped masses) leaves unflagged at three months as well,"
            if all(c["a_is_sample_a"] for c in cs)
            else f"on the {listing([str(int(c['a']['n'])) for c in cs])} dates that decision 5's condition (recomputed from the three-month rows' two clipped masses) flags at neither horizon"
        )
        vals = " and ".join(
            f"{fmt(c['a']['mean'], 4, True)} against the three-month {c['short']} pass ({', '.join(fmt(v, 4, True) for v in c['diff'][c['dates']['a']])}; {fmt(c['a2']['mean'], 4, True)} on the {int(c['a2']['n'])} without decision 2's flagged date)"
            for c in cs
        )
        sizes = {int(c["a"]["n"]): c["a"] for c in cs}
        return f"{who} the differences average {vals}, where {Rules.against_small(0.0, *sizes.values())}"

    if prod is not None:
        side3 = Rules.side(s_bud["mean"], 0.0, s_bud)
        three = (
            f"{side3} the production one"
            if side3 in ("above", "below")
            else "not distinguishable from the production one"
        )
        one = (
            "below it on" if n_dev_lower == len(dd) else "above it on" if n_dev_lower == 0 else f"below it on {n_dev_lower} of"
        )  # fmt: skip
        versus = (
            f"with the development budget on both sides it averages {two_samples(dev)}; "
            f"against the three-month production pass run after the freeze (mean {pm(s3['mean'], s3['se'])}, Newey–West {fmt(s3['nw'])}, n {int(s3['n'])} priced dates, the wing binding on {prod['n_flagged_three']} of them by decision 5's condition recomputed from the rows' two clipped masses, same commit as the one-year pass) it averages {two_samples(prod)}; "
            f"{owner_rule(dev, prod)}; "
            f"the second comparison is across two budgets, and one at the production budget on both sides is not available, the one-year pass being at the development budget: at three months the development LC/CC is {three} on the {int(s_bud['n'])} common dates ({pm(s_bud['mean'], s_bud['se'], 5, True)} across dates, Newey–West {fmt(s_bud['nw'], 5)}, higher on {n_bud_above} of the {int(s_bud['n'])}; the two three-month passes also differ by their commit, and on the dates with the same specification the three-month note has \"{budget_3m}\"), "
            f"and at one year it is {one} the {len(dd)} dates priced at both budgets (by {abs(mean_dd):.4f} on average, a mean of {len(dd)} differences that carries no error), so that neither comparison is that one; "
            f"by the rule at {NW_LAGS} lags, then, at the development budget on both sides the difference is {with_and_without(dev)}, and against the three-month production pass it is {with_and_without(prod)}, a comparison across two budgets"
        )
    else:
        versus = (
            f"with the development budget on both sides it averages {two_samples(dev)}; {owner_rule(dev)}; "
            f"by the rule at {NW_LAGS} lags, then, the difference is {with_and_without(dev)}"
        )
    zz = [
        b.get(f"frozen.budget.{x}.lc_over_cc_dev_minus_prod")
        / b.se(f"frozen.budget.{x}.lc_over_cc_dev_minus_prod")
        for x in pc.REFERENCE_DATES
    ]
    lcc, ccc = stats[("lc_over_copula", "b")], stats[("cc_over_copula", "b")]
    lcc_c, ccc_c = stats[("lc_over_copula", "c")], stats[("cc_over_copula", "c")]
    lcc_a, ccc_a = stats[("lc_over_copula", "a")], stats[("cc_over_copula", "a")]
    below_2008 = int((p.loc[p["sample_a"], "date"] < "2009").sum())
    med_c = float(p.loc[p["sample_c"], "clip_larger"].median())
    b.add(
        "said.clip_larger_median.c",
        f"the larger of the two clipped masses: median over the priced dates without decision 2's flagged dates [{INCL}]",
        med_c,
        unit="fraction of the particles",
        n=int(p["sample_c"].sum()),
        notes="a statistic of a calibration diagnostic across dates: no standard error",
    )
    n_kept = int(b.get("state.gate_not_waived"))
    gate = (
        f"passes on its numbers (the largest absolute index error there is {fmt(b.get('state.gate_not_waived.max_abs_idx_err_atm'), 3)} vp at the money and {fmt(b.get('state.gate_not_waived.max_abs_idx_err_90'), 3)} vp at the 90 % strike, against the gate's {GATE_VP})"
        if int(b.get("state.gate_not_waived.outside")) == 0
        else f"FAILS on {int(b.get('state.gate_not_waived.outside'))} of them"
    )
    within_year = {
        "both": "holds within the year as well",
        "neither": "is not established within the year",
        "one": "holds within the year on one of the two samples only",
    }[clip["made_of"][1]]
    sentences = [
        f"**The size and sign of the local-correlation effect at one year.** By the rule of decision 5 (sample (a), the {int(sa['n'])} unflagged dates) LC/CC − 1 averages {100 * (sa['mean'] - 1):+.2f} % with a median of {100 * (sa['median'] - 1):+.2f} %, and without decision 2's flagged date (sample (a2), {int(sa2['n'])} dates) {100 * (sa2['mean'] - 1):+.2f} % with a median of {100 * (sa2['median'] - 1):+.2f} %, "
        f"and on their dates with status ok (samples (a-ok) and (a2-ok), {int(sa_ok['n'])} and {int(sa2_ok['n'])} dates) {100 * (sa_ok['mean'] - 1):+.2f} % and {100 * (sa2_ok['mean'] - 1):+.2f} %: {Rules.against_small(1.0, sa, sa2, sa_ok, sa2_ok)}; "
        f"on all {n_p} priced dates (sample (b)) [{INCL}] LC/CC − 1 averages {100 * (sb['mean'] - 1):+.2f} % ± {100 * sb['se']:.2f} across dates (Newey–West, {NW_LAGS} lags: ± {100 * sb['nw']:.2f}; median {100 * (sb['median'] - 1):+.2f} %, quartiles {100 * (sb['q25'] - 1):+.2f} % and {100 * (sb['q75'] - 1):+.2f} %), so that LC/CC is {Rules.against_lags(sb['mean'], 1.0, sb)}, and without decision 2's flagged dates (sample (c), n {int(sc['n'])}) [{INCL}] {100 * (sc['mean'] - 1):+.2f} % ± {100 * sc['se']:.2f} (Newey–West ± {100 * sc['nw']:.2f}), {Rules.against_lags(sc['mean'], 1.0, sc)}.",
        f"**Against three months.** The frozen three-month value of LC/CC is {pm(f3['value'], f3['se'])} across dates (Newey–West, 12 lags: {fmt(f3_nw)}; n {f3['n']} priced dates, the wing binding on 201 of them by reader note 4, so flagged rows are included as in sample (b) here [{INCL}]; development budget, commit {f3['commit']}), and LC/CC at one year minus LC/CC at three months is paired on the dates priced in both passes [{INCL}]: {versus}.",
        f"**Where the model's wing binds.** Decision 5's flag fires on {n_flag} of the {n_p} priced dates ({100 * n_flag / n_p:.1f} %) — at the cap on {int(p['flag_clip_high'].sum())}, at λ = 0 on {int(p['flag_clip_low'].sum())} — the median of the larger clipped mass over the priced dates [{INCL}] is {fmt(b.get('clip.clip_larger.median'))} of the particles inside ±2.5 sd ({fmt(med_c)} without decision 2's flagged dates) against the flag's 0.01, and the index gate is waived on those {n_flag} dates, while on the other {n_kept} it is not waived and {gate}.",
        f"**The ratios to the copula are not local-correlation effects.** On sample (a) CC/copula averages {fmt(ccc_a['mean'])} and LC/copula {fmt(lcc_a['mean'])} ({Rules.against_small(1.0, ccc_a)}); on the {n_p} priced dates [{INCL}] CC/copula averages {pm(ccc['mean'], ccc['se'])} (Newey–West {fmt(ccc['nw'])}) and is {Rules.against_lags(ccc['mean'], 1.0, ccc)}, LC/copula averages {pm(lcc['mean'], lcc['se'])} (Newey–West {fmt(lcc['nw'])}) and is {Rules.against_lags(lcc['mean'], 1.0, lcc)}; "
        f"without decision 2's flagged dates (sample (c), n {int(ccc_c['n'])}) [{INCL}] CC/copula averages {pm(ccc_c['mean'], ccc_c['se'])} (Newey–West {fmt(ccc_c['nw'])}) and is {Rules.against_lags(ccc_c['mean'], 1.0, ccc_c)}, LC/copula averages {pm(lcc_c['mean'], lcc_c['se'])} (Newey–West {fmt(lcc_c['nw'])}) and is {Rules.against_lags(lcc_c['mean'], 1.0, lcc_c)}; "
        f"and CC/copula moves by {fmt(b.get('level.slope.b'), 4, True)} ± {fmt(b.se('level.slope.b'))} (HC1) per vol point of gap between the model's index target, read at the forward, and the study's index level, read at the spot (correlation {fmt(b.get('level.corr.b'), 2, True)}, n {n_p}; {fmt(b.get('level.slope.c'), 4, True)} ± {fmt(b.se('level.slope.c'))} and {fmt(b.get('level.corr.c'), 2, True)} on sample (c)) [{INCL}]"
        + (
            f"; that gap follows the study's forward (correlation {fmt(b.get('level.corr_gap_log_forward'), 2, True)} with ln(F/S)), and read at the same strike it has a mean of {fmt(b.get('level.gap_same_strike.mean'), 2, True)} vp and is beyond {GAP_FAR} vp on {int(b.get('level.gap_same_strike.n_far'))} dates against {int(b.get('level.n_far'))}."
            if "L12M.level.gap_same_strike.n_far" in b.records
            else "."
        ),
        f"**The effect and the clipped mass.** On the {r20['n']} priced dates with a larger clipped mass of at most {cap0:.2f} [{INCL}] the slope of LC/CC on that mass is {fmt(r20['coef'][1], 3, True)} ({fit_errs(r20, 1, 3)}), {r20['verdict']}, and {fmt(r20c['coef'][1], 3, True)} ({fit_errs(r20c, 1, 3, True)}) without decision 2's flagged dates, {r20c['verdict']}; "
        f"among the flagged dates alone it is {fmt(fl['coef'][1], 3, True)} ({fit_errs(fl, 1, 3, True)}), {fl['verdict']}, and {fmt(fl_c['coef'][1], 3, True)} ({fit_errs(fl_c, 1, 3, True)}) without decision 2's flagged dates, {fl_c['verdict']}; "
        f"with a dummy for each year it is {fmt(yr['coef'][1], 3, True)} ({fit_errs(yr, 1, 3, True)}), {yr['verdict']}, and {fmt(yr_c['coef'][1], 3, True)} ({fit_errs(yr_c, 1, 3, True)}) without decision 2's flagged dates, {yr_c['verdict']}, so that by the rule the relation {within_year}; "
        f"the mean of LC/CC goes from {fmt(bins[0]['mean_b'])} on the {bins[0]['n_b']} unflagged dates to {fmt(bins[-2]['mean_b'])} on the {bins[-2]['n_b']} dates with a mass of {bins[-2]['bin']} and {fmt(bins[-1]['mean_b'])} on the {bins[-1]['n_b']} dates {bins[-1]['bin']}, while over the {fb['n']} dates of sample (c) without the {count(len(clip['near_one']), 'date', 'dates')} at a clipped mass of {CLIP_NEAR_ONE:g} or more a straight line has a coefficient on the mass clipped at the cap of {fmt(fb['coef'][2], 4, True)} ({fit_errs(fb, 2)}; R² {fmt(fb['r2'], 3)}), {Rules.slope(fb['coef'][2], fb['se_hc1'][2], fb['se_nw'][2])}; this is an association across dates, flagged ones included, and does not say what an unclipped model would give.",
        f"**What is not established.** (i) A one-year LC/CC by the owner's rule with an error ({int(sa['n'])} unflagged dates, {below_2008} of them in 2008, carry none); (ii) LC/CC beyond three decimals (on the {len(dd)} dates priced at both budgets the development value is {fmt(min(abs(x) for x in dd), 4)} to {fmt(max(abs(x) for x in dd), 4)} {'below' if all(x < 0 for x in dd) else 'away from'} the production one, which is {fmt(min(abs(z) for z in zz), 1)} to {fmt(max(abs(z) for z in zz), 1)} times the two rows' combined Monte Carlo error); (iii) a comparison of one year with three months at the production budget on both sides, the one-year pass being priced at it on {len(dd)} dates only; (iv) any call ratio at 1.5 or 2 times P_D, and the runaway-path share at one year, which is not measured; (v) a model S comparison at one year, since the study has no one-year model S; (vi) anything at two years beyond four indicative, flagged rows.",
    ]
    for one_sentence in sentences:
        if re.search(r"[.!?] [A-Z(]", one_sentence.split(".** ", 1)[1]):
            raise ValueError(
                f"a statement of section 8 is more than one sentence: {one_sentence[:60]}"
            )
    if len(sentences) > 6:
        raise ValueError("section 8 has more than six sentences")
    # the numbers of the second sentence, sample by sample
    lags_all = sorted({NW_LAGS, *NW_OTHER})
    sample_rows = (
        ("all", f"every date priced in both passes [{INCL}]"),
        ("c", f"without the dates decision 2 flags at either horizon [{INCL}]"),
        ("a", "the dates decision 5's condition flags at neither horizon — the rule of decision 5"),
        ("a2", "the same without decision 2's flagged dates — the rules of decisions 5 and 2"),
    )
    cmp_rows, cmp_md = [], []
    for c in (dev, prod):
        if c is None:
            continue
        for skey, slabel in sample_rows:
            s = c[skey]
            n = int(s["n"])
            nws = {k: (s.get("nw") if k == NW_LAGS else s.get(f"nw_{k}")) for k in lags_all}
            xs = {k: Rules.errors(s["mean"], 0.0, s.get("se"), nws[k]) if n >= MIN_N_SE else NAN for k in lags_all}  # fmt: skip
            sides = {k: Rules.side(s["mean"], 0.0, s, k) for k in lags_all}
            cmp_rows.append({
                "three_month_side": c["side"], "sample": skey, "sample_label": slabel, "n": n, "mean": s["mean"], "median": s["median"], "min": s["min"], "max": s["max"], "se": s.get("se", NAN),
                **{f"nw_{k}": nws[k] for k in lags_all}, **{f"errors_{k}": xs[k] for k in lags_all}, **{f"verdict_{k}": sides[k] or "nothing said" for k in lags_all},
            })  # fmt: skip
            cmp_md.append([
                c["side"], slabel, str(n), fmt(s["mean"], 5, True), fmt(s["median"], 5, True), fmt(s.get("se"), 5) if finite(s.get("se")) else f"none ({n} dates)",
                " / ".join(fmt(nws[k], 5) for k in lags_all) if finite(s.get("nw")) else f"none ({n} dates)",
                " / ".join(errs(xs[k]) for k in lags_all) if finite(s.get("nw")) else "none",
                " / ".join(f"{sides[k]} 0" if sides[k] in ("above", "below") else f"{sides[k]} from 0" for k in lags_all) if finite(s.get("nw")) else f"nothing said ({n} dates)",
            ])  # fmt: skip
    b.save("one_year_vs_three_months", pd.DataFrame(cmp_rows))
    table = md_table(
        [
            "three-month side",
            "sample",
            "n",
            "mean of LC/CC at one year minus LC/CC at three months",
            "median",
            "± across dates (sd/√n)",
            f"± Newey–West at {' / '.join(str(k) for k in lags_all)} lags",
            f"errors from 0 at {' / '.join(str(k) for k in lags_all)} lags",
            f"by the rule of section 0 at {' / '.join(str(k) for k in lags_all)} lags",
        ],
        cmp_md,
    )
    same_flags = all(c["flag2_same"] for c in (dev, prod) if c is not None)
    text = "\n\n".join(
        [
            "## 8. What can be said at one year",
            "Six sentences, each built from the records of this note by the rules of section 0, with its sample and its errors. The first gives the summary by the owner's rule and the summary on all priced dates side by side: the choice between them is the owner's (section 0) and is not taken here. The table after the sentences gives the numbers of the second one, sample by sample.",
            *[f"{i}. {s}" for i, s in enumerate(sentences, 1)],
            f"**The comparison of sentence 2, sample by sample.** The one-year side is this pass (development budget). A date is out of the second and fourth samples when decision 2 flags it at one year or at three months; on the dates priced in both passes the two flags {'fall on the same dates' if same_flags else 'do not fall on the same dates'}. Decision 5's condition at three months is recomputed from the row's two clipped masses (a date is flagged when either is above {CLIP_FLAG:g}): of the {dev['n_three']} priced dates of the three-month development pass it flags {dev['n_flagged_three']} and does not flag {dev['n_unflagged_three']}"
            + (
                f", and of the {prod['n_three']} priced dates of the production pass it flags {prod['n_flagged_three']} and does not flag {prod['n_unflagged_three']}"
                if prod is not None
                else ""
            )
            + f". No error is carried below {MIN_N_SE} dates (`tables/one_year_vs_three_months.csv`).",
            table,
        ]
    )
    b.save(
        "what_can_be_said",
        pd.DataFrame({"n": range(1, len(sentences) + 1), "statement": sentences}),
    )
    return text


# ----------------------------------------------------------------------------- the page
def by_date_table(data: Data) -> pd.DataFrame:
    """One line per date of the table.  On a failed date the flags of the two decisions are left
    empty (the date has no clipped mass and no specification); the sample columns read False,
    a failed date being in no sample."""
    d = data.d.copy()
    for g in (*GATING, "check_names"):
        d[f"{g}_printed"] = [chk(x) if pr else "" for x, pr in zip(d[g], d["priced"], strict=True)]
    for col in ("flag_clip", "flag_clip_low", "flag_clip_high", "flag_unscreened"):
        d[col] = d[col].astype(object).where(d["priced"], None)
    front = [
        "date", "in_monthly_list", "yearly_date", "status", "reason", "priced", "sample_a", "sample_a2", "sample_a_ok", "sample_a2_ok", "sample_b", "sample_c", "sample_d", "sample_d2", "flag_clip", "flag_clip_low", "flag_clip_high", "flag_unscreened", "names_unscreened",
        *(f"{g}_printed" for g in (*GATING, "check_names")), "nonfinite", "gate_outside", "wing_binds", "index_extrapolated", "n_names_extrapolated", "n_dropped_calendar_index", "n_dropped_calendar_names", "n_dropped_index_screen",
        "clip_low_inner_max", "clip_high_inner_max", "clip_larger", "lc_over_cc", "lc_over_cc_se", "lc_over_copula", "lc_over_copula_se", "cc_over_copula", "cc_over_copula_se", "listed_ratio",
        "ED_lc", "ED_lc_se", "ED_cc", "ED_cc_se", "P_D_copula", "target_atm", "study_atm", "atm_gap", "study_fwd", "target_at_spot_strike", "gap_same_strike", "gap_strike_part", "target_atm_norepair", "repair_effect_atm_vp", "repair_effect_max_abs_vp", "norepair_index_surface_equal",
        "rho_cc", "rho_cop", "n_kept_below", "nearest_kept_below", "index_last_slice", "T",
        "forward_error", "forward_error_se", "forward_z", "idx_err_atm", "idx_err_atm_se", "idx_err_90", "idx_err_90_se", "idx_err_m25", "idx_err_m25_se", "kappa_lc", "kappa_lc_se", "kappa_cc", "kappa_cc_se", "EV_over_EQV", "EV_over_EQV_se",
        "names_mc_over_listed", "names_svi_over_listed", "names_mc_over_svi", "names_mc_z", "ED_wing_ratio", "ED_wing_ratio_se", "ED_eqv_ratio", "ED_eqv_ratio_se",
    ]  # fmt: skip
    calls = [
        c
        for tag in CALL_TAGS
        for c in (
            f"K_{tag}",
            f"C_{tag}_lc",
            f"C_{tag}_lc_se",
            f"C_{tag}_cc",
            f"C_{tag}_cc_se",
            f"C_{tag}_ratio",
            f"C_{tag}_ratio_se",
            f"C_{tag}_lc_over_copula",
            f"C_{tag}_lc_over_copula_se",
            f"C_{tag}_cc_over_copula",
            f"C_{tag}_cc_over_copula_se",
        )
    ]
    own = ["lc_over_copula_se_table_own", "cc_over_copula_se_table_own"]
    return d[
        [
            *front,
            *calls,
            *own,
            "risk",
            "git_commit",
            "config_digest",
            "spec_key",
            "budget",
            "n_particles",
            "n_paths",
            "companion_paths",
        ]
    ].copy()


def build(run_spec: bool) -> tuple[str, Book, str, str, list[str]]:
    data = load()
    if run_spec:
        spec_build(data)
    b = Book(data.commit)
    self_checks(b, data)
    attach_spec(b, data)
    read_first = section_read_first(b, data)
    b.blocks.append(read_first)
    section_state(b, data)
    stats = section_forward(b, data)
    clip = section_clip(b, data)
    section_extremes(b, data, clip)
    section_calls(b, data)
    differently = section_frozen(b, data, stats)
    section_two_years(b)
    said = section_said(b, data, stats, clip)
    b.blocks.append(said)
    b.save("by_date", by_date_table(data))
    spec = read_spec("spec_by_date")
    head = git("rev-parse", "--short", "HEAD").strip()
    spec_line = (
        f"- `tables/spec_by_date.csv`, `tables/spec_strips_by_name.csv`, `tables/spec_failed_dates.csv`, `tables/spec_unscreened_names.csv`: {SPEC_NOTE}; built {spec['built'].iloc[0]} New York time from the worktree at commit {spec['worktree_commit'].iloc[0]}; {int(spec['spec_key_is_the_rows'].sum())} of {len(spec)} keys equal the rows'. The same build rebuilds each priced date with `screen.calendar_repair` off for the index target without the repair (columns `target_atm_norepair`, `repair_effect_atm_vp`): that second specification is not a row of any run."
        if spec is not None
        else "- The specification-only tables (`tables/spec_*.csv`) are not built: run the script with `--spec-build`."
    )
    b.text(
        "## Scripts, inputs and checks",
        f"Generated by `scripts/pm_later_12m_monthly.py` (worktree at commit {head}; the script reads finished runs and prices nothing). Every number of the page is a record of `parts/{PART}.json` ({len(b.records)} records, in the package's record format) or a cell of a CSV under `tables/`.",
        "Inputs (read only):\n"
        f"- `{rel(TABLE)}`, its row files `{rel(ROWS)}/<date>.json` and the runs' logs `{rel(LOGS)}`; the automatic report `{rel(REPORT)}`; the driver's log `outputs/dispersion_lc/logs/driver.log` (quoted once).\n"
        f"- `{rel(TABLE_PROD)}` and `{rel(TABLE_24)}` with `{rel(ROWS_24)}`; `{rel(REPORT_24)}`; `{rel(TABLE_3M_FROZEN)}` and `{rel(TABLE_3M_PROD)}`; the old-defaults row `{rel(TABLE_NOREPAIR)}` with `{rel(ROWS_NOREPAIR)}`.\n"
        "- The study (a read-only link): `outputs/dispersion/entries_12m.parquet`, `entries_24m.parquet`, `legs_12m.parquet`, `model_s_3m.parquet`, the names of the entry pickles, `PROGRESS_Q2.md`"
        + (
            f"; and, on the {int(data.priced['sample_a'].sum())} unflagged dates, the inputs as the pricer's own loader reads them (`lcm_diagnostics.load_inputs`: the entry, the chains and the prices of the date), for the DJX expiries they list (section 3).\n"
            if spec is not None
            else ".\n"
        )
        + f"- The frozen package: `{rel(FROZEN_1Y)}/NUMBERS_1Y.md`, `numbers_1y.json`, `tables/1y_dates.csv`; `{rel(HIST_FROZEN)}`; `{rel(READER_NOTES)}`; `{rel(VALIDATION)}`. After the freeze: `{rel(HIST_PROD)}`.\n"
        f"- The pricer's source at the rows' commit: `git show {data.commit}:scripts/lcm_price.py`.\n"
        + spec_line,
        "Checks made by the script on this run (each raises when it does not hold):\n"
        + "\n".join(f"- {c}" for c in b.checks),
        "The freeze check (`scripts/pm_freeze.py --check`) is run after the script and its result is in the note's return, not here: this page is under `later/`, outside the manifest.",
    )
    title = "\n\n".join([
        "# The one-year pass on the monthly dates (12m, development budget): a note after the freeze",
        f"Written {time.strftime('%Y-%m-%d %H:%M')} New York time by `scripts/pm_later_12m_monthly.py` into `{rel(OUT)}`. Not part of the frozen package: no frozen file is changed (`scripts/pm_freeze.py --check`), and no number of the frozen one-year addendum is corrected or replaced.",
    ])  # fmt: skip
    return "\n\n".join([title, *b.blocks]) + "\n", b, read_first, said, differently


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument(
        "--spec-build",
        action="store_true",
        help="rebuild the specification-only tables first (a few minutes)",
    )
    args = ap.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S"
    )
    markdown, b, _, _, differently = build(args.spec_build)
    for name, frame in b.tables.items():
        pc.save_table(frame, name, base=OUT)
    pc.write_part(PART, list(b.records.values()), markdown, base=OUT)
    write_text(OUT / "NOTE_12M_MONTHLY.md", markdown)
    LOG.info(
        "%d records, %d tables, %d characters -> %s",
        len(b.records),
        len(b.tables),
        len(markdown),
        OUT / "NOTE_12M_MONTHLY.md",
    )
    LOG.info("%d frozen sentences read differently on the monthly dates", len(differently))


if __name__ == "__main__":
    main()
