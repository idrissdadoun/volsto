"""After the freeze of the PM results package of 2026-10-09: the gap of the ONE-YEAR trades of the
dispersion study, held to expiry and delta-hedged, at three entry prices of the Palladium forward:
the study's copula, the local-correlation model (LC) and LC's constant-correlation companion (CC).

    python scripts/pm_later_1y_gap.py [--rows-dir DIR]

The owner's question of 2026-10-10: is the delta-hedged gap of the 12m trades positive or
negative?  The script can be rerun while the LC pass at 12m is running: it uses the row files
that exist, and says how many it used and which dates are missing.  Every sentence of the answer
is generated from the computed numbers by an explicit rule (:func:`answer`), so the page stays
right when the script is rerun on more trades.

The rules of the answer.

* *Sign of a mean* (:func:`sign_rule`): "negative" or "positive" only when the 95 % block-bootstrap
  interval excludes zero AND ``|t| >= 1.96``; "about zero, sign not determined" when the mean is
  less than ``ABOUT_ZERO`` standard errors from zero; otherwise the sample mean, its error and
  "the sign of the mean is not established".
* *The windows* (:func:`year_stats`, :func:`windows_stats`, :func:`windows_rule`): a mean that
  passes the rule is graded on three tests that count its one-year windows, each on as many
  degrees of freedom as entry years less one: the plain t of its entry-year means, the same t
  with one lag between adjacent years (Hansen-Hodrick on the year means), and its own
  Hansen-Hodrick t.  Adjacent year means are positively correlated, so the plain t is the laxest;
  the grade is on the largest of the three two-sided p-values: "at the 5 % limit" between 2.5 %
  and 5 %, "does not pass at 5 %" above.
* *Share of negative trades* (:func:`describe`): Hansen-Hodrick standard error on the indicator
  (same lags as the mean) and the block-bootstrap interval; "distinguishable from one half" only
  when that interval excludes 0.5.  The median has a block-bootstrap interval, and its sign is
  named only when that interval excludes zero.  Both are judged on the interval alone.
* *Sub-samples* (:func:`subs_words`): the answer says what the rule calls the mean on each, not
  the sign of its sample mean; the sub-samples share their trades and are not separate evidence.
* *Other bootstrap seeds*: the intervals of the three main samples are redrawn on seeds 1 to 40
  without the study's own (39 seeds).
* *Chains of non-overlapping trades* (:func:`chain_stats`): from each trade entered in the first
  year, the next trade entered on or after the expiry of the current one, and so on; the count
  of negative trades on the median chain with its two-sided binomial p.  The median chain is
  the middle one when the chains are ordered by their count of negative trades and, at equal
  counts, by their number of trades.  The chains that have the median count can differ in
  length, so the count is judged on all of them (:func:`chain_rule`): "distinguishable from one
  half at 5 %" when its p is below 5 % on each, "at the 5 % limit" when it is below on some and
  not on others (the verdict then turns on the tie-break), "not distinguishable" when on none.
  The page also counts the chains whose own count has a p below 5 %.
* *The months without an LC price* (:func:`combined_rows`): the copula gap of the monthly trades
  with and without a priced row, and the combined estimator "copula gap on all the study's
  monthly trades + mean price shift on the priced trades", plain and with the year-weighted
  shift, with a Hansen-Hodrick error computed on the combined estimator.  The combined estimate
  and the difference of two means are not called "the sample mean" (:func:`noun_of`).  At the
  copula's price the mean of the priced trades unweighted, the same year-weighted and the mean
  of all the monthly trades are printed side by side, with whether the year weights bring the
  priced trades closer to all of them (:func:`three_means`).
* *The two errors of a mean* (:func:`se_gap_words`): in Tables 5, 6 and 7 the standard
  deviation of the block-bootstrap means is printed beside the Hansen-Hodrick error; where the
  larger of the two is more than ``SE_RATIO`` times the smaller, one sentence names the cells,
  which of the two is the larger and what the t on the larger is.

What it computes.

* *The gap* is the Palladium forward minus the vega-neutral package (the single-name straddles
  minus the basket straddle, one unit each), per trade at unit notional, in % of notional: held to
  expiry it is the study's column ``GAP_U`` (= ``PF_U - PKG_v_U`` = ``G - P_G``), delta-hedged
  ``GAP_H`` (= ``PF_H - PKG_v_H``, with ``PF_H = PF_U + hedge_PF``); the identities are checked.
  A *trade* is one entry date of ``scripts/disp_tables.py::load("12m", "B1")`` with its outcome
  (the entry with a member stuck over the window, 2009-06-01, is left out by the study).
* *At another price of the forward* only the entry price changes (the hedge and the package stay
  the study's):  P&L at a model's price = P&L at the copula's price + (copula price - model
  price), per trade; copula price ``P_D``, LC price ``ED_lc``, CC price ``ED_cc``.
* *Validation first.*  The study's printed one-year numbers (``T17_other_runs.csv``,
  ``T13_other_runs.csv``, ``T4_T6_other_tenors.csv``, ``T2_other_runs.csv`` of
  ``outputs/dispersion/report/tables``, read as text, and the table of section 5.7 of the report
  draft, ``f5c_1y.tex``, whose lines are looked up word for word) are recomputed on the study's
  weekly trades.  A number at the LC or CC price is given only when the printed numbers it rests
  on (the gap's own numbers, the means of its two legs, the prices and the payoffs) are
  reproduced to the printed digit; a printed number that is not reproduced is named on the page.
* *The monthly trades.*  One trade per calendar month: the study's one-year trades entered on a
  date its table flags ``monthly`` or on a date of the LC pass (``scripts/disp_lcm.py``'s rule:
  the converged dates of ``model_s_3m.parquet`` and today, with an entry at 12m).  The LC sample
  is those of them with a priced row (status ``ok`` or ``check``, finite prices) in
  ``outputs/dispersion_lc/rows/12m_development`` (one JSON per date).
* *Sub-samples* of the LC sample: by half, without the dates where a name is kept unscreened or
  priced beyond its last kept expiry, the dates flagged and not flagged for the clipped mass,
  without the dates whose DJX target was repaired, without the entries of 2008-2009, status
  ``ok`` only, and the whole sample weighted to the study's monthly trades per entry year; and,
  at the copula's price, the two halves of the study's weekly and monthly trades.
* *Payout per 1 of premium* of the forward: ``sum(D) / sum(price)`` at the three prices.
* *The study's other gap*, against the correlation-neutral package (``GAP_rho_H``), with the same
  price shift: given for completeness; the report's gap is the vega-neutral one.
* *The state of the row folder* at run time (files, time of the newest, priced, pending,
  failed) and whether the pass is complete, by rule (:func:`pass_state`).

The errors.

* A one-year trade overlaps the eleven monthly entries after it.  The standard error of a mean is
  Hansen-Hodrick (uniform kernel; the study's ``volsto.studies.disp_stats.mean_se``): 50 lags on
  the weekly trades (``disp_tables.LAG["12m"]``), 11 lags on the monthly ones (the window in
  entries less one, as the study's 12 weekly and the package's 2 monthly lags at 3m).  On the
  monthly trades the lag is counted in calendar months of entry, so a missing month stays at its
  place (:func:`hh_se`; equal to ``mean_se`` on a series without a hole and to
  ``disp_stats.subset_mean_se`` on a subset of a series: both checked on every run).  The 0-lag
  value (trades taken as independent: not an error to quote) and the 12- and 17-lag values are
  printed beside it.  Where the Hansen-Hodrick variance is not positive the script falls back to
  Newey-West at twice the lag, as the study does: every such cell of the page carries a mark and
  every CSV that prints such an error has a fallback column.
* The 95 % interval is the study's circular block bootstrap (the resampling of
  ``disp_stats.bootstrap_ratio``, checked against it): 2,000 resamples, seed 11, blocks of 51
  weekly entries (``disp_tables.STEP["12m"]``) or 12 monthly entries (``STEP // 4``, the rule of
  ``disp_tables2.model_s_tables``).  A block is 12 consecutive trades of the sample: where months
  are missing it spans more than 12 months.
* The Monte Carlo error of a mean LC or CC price is given as a bound, the mean of the per-date
  standard errors (the dates are priced on the same seeds): it is the pricing noise given the
  calibrated model, not the calibration's noise nor the budget.
* A sample of fewer than 10 trades gets no standard error and no interval; its trades are listed.
  A sample shorter than three blocks (36 monthly trades) gets a standard error and no interval,
  hence no sign by the rule: with a single block the circular resample is the sample itself.

Reads (read-only): ``outputs/dispersion/entries_12m.parquet``, ``outcomes_12m.parquet``,
``indicators_12m.parquet``, ``model_s_3m.parquet``, ``entries/12m/*.pkl`` (names only),
``report/tables/*.csv``; ``outputs/interview/pm_package_lc_draft/src/f5c_1y.tex``; the row files.
Writes only under ``outputs/dispersion_lc/pm_update/later/1y_gap``: ``GAP_12M.md``,
``parts/gap_12m.json`` and ``.md``, ``tables/gap_12m_*.csv``.  No Monte Carlo, no calibration.
The CSV files are in % of notional with the suffix ``_pct``; in ``gap_12m_by_trade.csv`` the
study's and the rows' columns of a price or of a P&L are times 100 and renamed with that suffix
(``BY_TRADE_PCT``).
"""

# ruff: noqa: E501, RUF001
from __future__ import annotations

import argparse
import functools
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_tables as tb  # load, LAG, STEP: the study's frame, its lags and its blocks
import pm_common as pc

from volsto.studies import disp_data as dd
from volsto.studies import disp_stats as st

PART = "gap_12m"
SECTION = "later/1y_gap"
TENOR = "12m"
BASKET = "B1"
TODAY = "2026-10-02"
LATER = pc.PM / "later"
OUT = LATER / "1y_gap"
ROWS = pc.LC_OUT / "rows" / "12m_development"
TABLES = pc.STUDY / "report" / "tables"
DRAFT = pc.ROOT / "outputs" / "interview" / "pm_package_lc_draft" / "src" / "f5c_1y.tex"
#: The study's conventions for weekly entries of one-year trades.
LAG_WEEKLY = tb.LAG[TENOR]
BLOCK_WEEKLY = tb.STEP[TENOR]
#: The sidenote of the draft's table says 51 lags; the study's tables are computed with 50.
LAG_DRAFT = 51
#: Monthly entries of one-year trades: the window in entries less one, and one year of entries.
LAG_MONTHLY = 11
BLOCK_MONTHLY = BLOCK_WEEKLY // 4
#: The lags printed for the sensitivity: (independent trades, the convention, two longer ones).
LAGS_WEEKLY = (0, LAG_WEEKLY, LAG_DRAFT, 75)
LAGS_MONTHLY = (0, LAG_MONTHLY, 12, 17)
N_RESAMPLES = 2000
SEED = 11  # disp_stats.bootstrap_ratio
#: The other bootstrap seeds of the sensitivity: 1 to 40 without the study's own (39 seeds).
SEEDS_RANGE = tuple(s for s in range(1, 41) if s != SEED)
SEEDS_WORDS = f"{len(SEEDS_RANGE)} other bootstrap seeds (1 to 40 without the study's seed {SEED})"
MIN_N = 10  # fewer trades: no standard error, no interval (disp_stats.bootstrap_ratio's rule)
#: A sample shorter than this many blocks has no bootstrap interval, hence no sign by the rule:
#: with one block the circular resample is the sample itself and the interval has no width.
MIN_BLOCKS = 3
T_CRIT = 1.96
#: A mean less than this many standard errors from zero is called "about zero".
ABOUT_ZERO = 0.5
#: "At the 5 % limit": the most cautious two-sided p of the tests of the windows is between this
#: and 0.05.
LIMIT_P = 0.025
#: A year mean closer to zero than this (% of notional) is named where the years are counted.
NEAR_ZERO = 0.01
#: The level of the binomial test of a count of negative trades on a chain.
CHAIN_P = 0.05
#: What :func:`chain_rule` says of the count of negative trades on the median chain.
CHAIN_YES = "distinguishable from one half at 5 %"
CHAIN_LIMIT = "at the 5 % limit"
CHAIN_NO = "not distinguishable from one half at 5 %"
#: The Hansen-Hodrick error of a mean and the standard deviation of its block-bootstrap means
#: "differ by more than a quarter" when the larger is more than this many times the smaller.
SE_RATIO = 1.25
#: A pass with pending dates and no row file written for this many minutes is called stopped.
STALE_MINUTES = 60
#: The mark of a Newey-West fallback in a table cell.
MARK = "†"
CLIP_FLAG = 0.01  # scripts/lcm_price.py, CLIP_FLAG_MASS
GATES = ("check_no_nan", "check_forward", "check_index")  # scripts/lcm_price.py, GATING_CHECKS
BUDGET = (2e5, 2e5, 1e5)  # particles, paths, paths of the constant-correlation fit
PRICES = ("copula", "lc", "cc")
PRICE_LABEL = {"copula": "copula", "lc": "LC", "cc": "CC"}
PRICE_COL = {"copula": "P_D", "lc": "ED_lc", "cc": "ED_cc"}
PRICE_SE = {"copula": "P_D_se", "lc": "ED_lc_se", "cc": "ED_cc_se"}
STRUCTURES = (("held", "GAP_U", "held to expiry"), ("hedged", "GAP_H", "delta-hedged"))
#: The study's other gap, against the correlation-neutral package: given for completeness.
OTHER = (("hedged_rho", "GAP_rho_H", "delta-hedged"),)
HEADLINE = (("weekly", "copula"), ("monthly", "copula"), ("priced", "copula"), ("priced", "lc"), ("priced", "cc"))  # fmt: skip
#: The columns of ``tables/gap_12m_by_trade.csv`` that are a price or a P&L: fractions of the
#: notional in the study's table and in the row files, written times 100 with the suffix ``_pct``.
BY_TRADE_PCT = (
    "P_D", "P_D_se", "D", "G", "P_G", "PF_U", "PF_H", "PKG_v_U", "PKG_v_H", "GAP_U", "GAP_H", "GAP_rho_U", "GAP_rho_H",
    "ED_lc", "ED_lc_se", "ED_cc", "ED_cc_se", *(f"{s}_{p}" for s, _, _ in (*STRUCTURES, *OTHER) for p in PRICES),
)  # fmt: skip
SAMPLE_LABEL = {"weekly": "the study's weekly trades", "monthly": "the study's monthly trades", "priced": "the monthly trades with a priced LC row"}  # fmt: skip
ROW_KEYS = (
    "date", "tenor", "status", "reason", "ED_lc", "ED_lc_se", "ED_cc", "ED_cc_se", "P_D_copula",
    "clip_low_inner_max", "clip_high_inner_max", "n_names_unscreened", "names_unscreened",
    "n_dropped_calendar_index", "n_names_extrapolated", "n_particles", "n_paths",
    "companion_paths", "git_commit", "config_digest", "carries_decisions", "forward_error", "forward_error_se", *GATES,
)  # fmt: skip
#: The owner's decisions of 2026-10-09 that a row of the pass carries, by the rule of
#: ``scripts/pm_1y.py``: ``calendar_repair`` true and the two columns present.
DECISIONS = (
    "the owner's decisions 1, 2 and 5 of 2026-10-09 (1: calendar repair of the names' slices; 2: a name with no screened expiry is kept unscreened; "
    "5: calendar repair of the target of the Dow index options, DJX)"
)
DECISION_COLUMNS = ("n_names_unscreened", "n_dropped_calendar_index")
NEED = ("ED_lc", "ED_lc_se", "ED_cc", "ED_cc_se")

SRC_STUDY = "outputs/dispersion/entries_12m.parquet + outcomes_12m.parquet (basket B1)"
SRC_ROWS = "outputs/dispersion_lc/rows/12m_development/<date>.json"
SRC_T17 = "outputs/dispersion/report/tables/T17_other_runs.csv, run 'B1 12m', structure GAP"
SRC_T17_RHO = "outputs/dispersion/report/tables/T17_other_runs.csv, run 'B1 12m', structure GAP_ρ"
SRC_T13 = "outputs/dispersion/report/tables/T13_other_runs.csv, run 'B1 12m'"
SRC_T4 = "outputs/dispersion/report/tables/T4_T6_other_tenors.csv, run 'B1 12m'"
SRC_T2 = "outputs/dispersion/report/tables/T2_other_runs.csv, run 'B1 12m', row 'all'"
SRC_TEX = "report draft sec. 5.7, outputs/interview/pm_package_lc_draft/src/f5c_1y.tex"
#: Lines of the draft looked up word for word, and the numbers read on each: (key, printed).
TEX_LINES: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    (
        "the table's row of the gap",
        r"gap = Palladium $-$ vega-neutral package & 8.90 & +0.91 \ (1.0) & \m 0.51 \ (\m 1.3) & \m 0.85 / \m 0.12\\",
        (("P_G.all", "8.90"), ("GAP_U.all.mean", "+0.91"), ("GAP_U.all.t", "1.0"), ("GAP_H.all.mean", "-0.51"),
         ("GAP_H.all.t", "-1.3"), ("GAP_H.IS.mean", "-0.85"), ("GAP_H.OOS.mean", "-0.12")),
    ),
    (
        "the table's row of the Palladium forward",
        r"Palladium forward & 14.62 & +1.49 \ (1.8) & \m 0.08 \ (\m 0.1) & \m 0.70 / +0.62\\",
        (("P_D.all", "14.62"), ("PF_U.all.mean", "+1.49"), ("PF_U.all.t", "1.8"), ("PF_H.all.mean", "-0.08"),
         ("PF_H.all.t", "-0.1"), ("PF_H.IS.mean", "-0.70"), ("PF_H.OOS.mean", "+0.62")),
    ),
    (
        "the table's row of the vega-neutral package",
        r"vega-neutral package & 5.72 & +0.58 \ (0.7) & +0.43 \ (1.1) & +0.14 / +0.75\\",
        (("P_PKG.all", "5.72"), ("PKG_v_U.all.mean", "+0.58"), ("PKG_v_U.all.t", "0.7"), ("PKG_v_H.all.mean", "+0.43"),
         ("PKG_v_H.all.t", "1.1"), ("PKG_v_H.IS.mean", "+0.14"), ("PKG_v_H.OOS.mean", "+0.75")),
    ),
    (
        "the sentence under the table",
        "The forward cost 14.62 and paid 16.11: price over payoff 0.91 (0.95 for entries to 2016, 0.87 from 2017).",
        (("P_D.all", "14.62"), ("D.all", "16.11"), ("P_D_over_D.all", "0.91"), ("P_D_over_D.IS", "0.95"), ("P_D_over_D.OOS", "0.87")),
    ),
)  # fmt: skip
WHAT = {
    "GAP_U": "held gap", "GAP_H": "hedged gap", "PF_U": "held Palladium forward", "PF_H": "hedged Palladium forward",
    "PKG_v_U": "held vega-neutral package", "PKG_v_H": "hedged vega-neutral package", "P_G": "price of the gap, mean",
    "G": "payoff of the gap, mean", "P_D": "copula price of the forward, mean", "D": "payoff of the forward, mean",
    "P_PKG": "price of the vega-neutral package, mean", "P_D_over_D": "price over payoff of the forward (sum over sum)",
    "GAP_rho_U": "held gap against the correlation-neutral package",
    "P_D_over_D_strip": "price over payoff of the forward on the windows with a usable strip",
}  # fmt: skip
STAT = {"mean": "mean", "t": "t (Hansen-Hodrick)", "sd": "sd over trades", "hit": "share of trades with a positive P&L",
        "q05": "5 % quantile", "n": "trades"}  # fmt: skip
SAMPLE_WORDS = {"all": "all weekly trades", "IS": "entries 2007-2016", "OOS": "entries 2017-2025"}
#: Which numbers at the LC or CC price rest on a printed number, by the column it is about: the
#: gap's own numbers, the means of its two legs, the prices and the payoffs.  The t-ratio of a leg
#: is compared and reported, and nothing at the LC or CC price rests on it (:func:`rests_on`).
#: ``other gap``: the study's other gap (against the correlation-neutral package), of which
#: the study prints the held numbers only at one year.
GROUPS = {
    "GAP_rho_U": ("other gap",),
    "GAP_U": ("held",), "PF_U": ("held",), "PKG_v_U": ("held",), "GAP_H": ("hedged",), "PF_H": ("hedged",),
    "PKG_v_H": ("hedged",), "P_G": ("held", "hedged"), "G": ("held", "hedged"), "P_PKG": ("held", "hedged"),
    "P_D": ("held", "hedged", "payout", "other gap"), "D": ("payout",), "P_D_over_D": ("payout",), "P_D_over_D_strip": ("payout",),
}  # fmt: skip
LEGS = ("PF_U", "PF_H", "PKG_v_U", "PKG_v_H")


def rests_on(key: str) -> tuple[str, ...]:
    """The numbers at the LC or CC price (``held``, ``hedged``, ``payout``) that rest on the
    printed number ``key``."""
    col, _, *stat = key.split(".")
    return () if col in LEGS and stat and stat[0] != "mean" else GROUPS[col]


# ------------------------------------------------------------------------------------ the data
def study_trades() -> pd.DataFrame:
    """The study's one-year entries on basket B1 as ``scripts/disp_tables.py::load`` builds them
    (one row per entry date, the entry with a stuck member left out), with the calendar month of
    entry as an integer and the identities the arithmetic rests on checked."""
    if dd.OUT.resolve() != pc.STUDY.resolve():
        raise ValueError(f"the study's folder: {dd.OUT} is not {pc.STUDY}")
    d = tb.load(TENOR, BASKET)
    if d["date"].duplicated().any() or not d["date"].is_monotonic_increasing:
        raise ValueError("the study's frame: dates not unique or not in order")
    d["month_index"] = d["date"].str[:4].astype(int) * 12 + d["date"].str[5:7].astype(int)
    d["monthly"] = d["monthly"].astype(bool)
    h = d[d["has_outcome"]]
    checks = {
        "PF_U = D - P_D": h["PF_U"] - (h["D"] - h["P_D"]),
        "PF_H = PF_U + hedge_PF": h["PF_H"] - (h["PF_U"] + h["hedge_PF"]),
        "GAP_U = G - P_G": h["GAP_U"] - (h["G"] - h["P_G"]),
        "GAP_U = PF_U - PKG_v_U": h["GAP_U"] - (h["PF_U"] - h["PKG_v_U"]),
        "GAP_H = PF_H - PKG_v_H": h["GAP_H"] - (h["PF_H"] - h["PKG_v_H"]),
        "P_G = P_D - (SS_mkt - Str_B_mkt)": h["P_G"] - (h["P_D"] - (h["SS_mkt"] - h["Str_B_mkt"])),
        "GAP_rho_U = PF_U - PKG_rho_U": h["GAP_rho_U"] - (h["PF_U"] - h["PKG_rho_U"]),
        "GAP_rho_H = PF_H - PKG_rho_H": h["GAP_rho_H"] - (h["PF_H"] - h["PKG_rho_H"]),
    }
    bad = {k: float(v.abs().max()) for k, v in checks.items() if not v.abs().max() < 1e-12}
    if bad:
        raise ValueError(f"the study's columns do not satisfy {bad}")
    need = h[["GAP_U", "GAP_H", "GAP_rho_U", "GAP_rho_H", "P_D", "P_D_se", "D", "G", "P_G"]]
    if not np.isfinite(need.to_numpy(float)).all():
        raise ValueError("a non-finite value in the study's columns of a trade with an outcome")
    return d


def pass_dates() -> list[str]:
    """The dates of the LC pass at 12m, by the rule of ``scripts/disp_lcm.py::study_dates`` (not
    imported: that module loads the pricing code): the converged dates of the study's model S
    table and today, with an entry at 12m."""
    table = pd.read_parquet(pc.STUDY / "model_s_3m.parquet")
    dates = {str(x)[:10] for x in table.loc[table["converged"].astype(bool), "date"]} | {TODAY}
    have = {p.stem for p in (pc.STUDY / "entries" / TENOR).glob("*.pkl")}
    return sorted(dates & have)


def lc_rows(rows_dir: Path) -> tuple[pd.DataFrame, list[str], dict[str, list[str]], dict[str, Any]]:  # fmt: skip
    """The row files of the pass: one line per readable file with the keys of ``ROW_KEYS``; the
    dates whose file cannot be parsed (a file being written); for each row whose
    ``check_no_nan`` fails, its non-finite numbers; and the state of the folder at this reading
    (the number of row files, the newest one, its time and its age in minutes)."""
    out, unreadable, non_finite, stamps = [], [], {}, {}
    now = time.time()
    for path in sorted(rows_dir.glob("*.json")):
        try:
            stamps[path.name] = path.stat().st_mtime
            row = json.loads(path.read_text())
        except (json.JSONDecodeError, FileNotFoundError):
            unreadable.append(path.stem)
            continue
        if row.get("date") != path.stem or row.get("tenor") != TENOR:
            raise ValueError(f"{path}: the row is not that of {path.stem} at {TENOR}")
        carries = row.get("calendar_repair") is True and all(c in row for c in DECISION_COLUMNS)
        out.append({**{k: row.get(k) for k in ROW_KEYS}, "carries_decisions": carries})
        if row.get("status") != "failed" and not row.get("check_no_nan", True):
            non_finite[path.stem] = sorted(k for k, v in row.items() if isinstance(v, float) and not np.isfinite(v))  # fmt: skip
    frame = pd.DataFrame(out, columns=list(ROW_KEYS))
    if frame["date"].duplicated().any():
        raise ValueError("the row files: more than one row on a date")
    newest = max(stamps, key=lambda k: stamps[k]) if stamps else ""
    folder = {
        "run_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)), "files": len(stamps), "newest_file": newest,
        "newest_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(stamps[newest])) if stamps else "n/a",
        "newest_age_minutes": (now - stamps[newest]) / 60.0 if stamps else float("nan"),
    }  # fmt: skip
    return frame, unreadable, non_finite, folder


def samples(rows_dir: Path) -> dict[str, Any]:
    """The trades and how they are counted."""
    d = study_trades()
    weekly = d[d["has_outcome"]].reset_index(drop=True)
    lc, unreadable, non_finite, folder = lc_rows(rows_dir)
    in_pass = pass_dates()
    is_priced = lc["status"].isin(["ok", "check"]) & np.isfinite(lc[list(NEED)].astype(float)).all(axis=1)  # fmt: skip
    priced, failed = lc[is_priced], lc[lc["status"] == "failed"]
    other = lc[~is_priced & (lc["status"] != "failed")]
    # one monthly trade per calendar month: the study's monthly flag, or a date of the pass
    monthly = weekly[weekly["monthly"] | weekly["date"].isin(in_pass)].reset_index(drop=True)
    if monthly["month_index"].duplicated().any():
        raise ValueError("two monthly trades entered in the same calendar month")
    cols = [c for c in ROW_KEYS if c != "tenor"]
    if set(cols) & set(monthly.columns) != {"date"}:
        raise ValueError("a key of the rows is also a column of the study's frame")
    x = monthly.merge(priced[cols], on="date", how="inner").sort_values("date").reset_index(drop=True)  # fmt: skip
    gap = float((x["P_D"] - x["P_D_copula"]).abs().max()) if len(x) else 0.0
    if not gap < 1e-12:
        raise ValueError(f"LC rows: P_D_copula differs from the study's P_D by up to {gap:.3g}")
    budget = priced[["n_particles", "n_paths", "companion_paths"]].drop_duplicates()
    if len(priced) and (len(budget) != 1 or tuple(budget.iloc[0]) != BUDGET):
        raise ValueError(f"the priced rows are not all at the development budget: {budget.to_dict('records')}")  # fmt: skip
    # the stored status is the current rule's (scripts/lcm_price.py::row_status)
    rule = np.where(x[list(GATES)].astype(bool).all(axis=1), "ok", "check")
    if not (rule == x["status"]).all():
        raise ValueError("a stored status is not the one of the gating checks of its row")
    x["flag_unscreened"] = x["n_names_unscreened"].fillna(0) > 0
    x["flag_extrapolated"] = x["n_names_extrapolated"].fillna(0) > 0
    x["flag_clip"] = (x["clip_low_inner_max"] > CLIP_FLAG) | (x["clip_high_inner_max"] > CLIP_FLAG)
    x["flag_djx_repaired"] = x["n_dropped_calendar_index"].fillna(0) > 0
    for p in PRICES:
        shift = x["P_D"] - x[PRICE_COL[p]]  # copula price - this price, per trade
        for s, col, _ in (*STRUCTURES, *OTHER):
            x[f"{s}_{p}"] = x[col] + shift
    stuck = list(d.attrs["stuck_dates"])
    no_outcome = sorted(d.loc[~d["has_outcome"], "date"])
    not_trade = sorted(set(priced["date"]) - set(x["date"]))
    why = {"stuck": [t for t in not_trade if t in stuck], "no_outcome": [t for t in not_trade if t in no_outcome]}  # fmt: skip
    why["other"] = [t for t in not_trade if t not in stuck and t not in no_outcome]
    # the trades the row folder can reach: the pass's dates and the rows already there
    reach = monthly[monthly["date"].isin(set(in_pass) | set(lc["date"]))]
    years = monthly["date"].str[:4]
    by_year = pd.DataFrame({
        "entry_year": sorted(years.unique()),
        "monthly_trades": [int((years == y).sum()) for y in sorted(years.unique())],
        "on_a_date_of_the_pass_or_with_a_row": [int((reach["date"].str[:4] == y).sum()) for y in sorted(years.unique())],
        "priced": [int((x["date"].str[:4] == y).sum()) for y in sorted(years.unique())],
    })  # fmt: skip
    per_year = by_year.set_index("entry_year")
    x["year_weight"] = [per_year.loc[y, "monthly_trades"] / per_year.loc[y, "priced"] for y in x["date"].str[:4]]  # fmt: skip
    # CC over copula over every priced row, the dates that are not trades included
    ratio = (priced["ED_cc"].astype(float) / priced["P_D_copula"].astype(float)).reset_index(
        drop=True
    )
    low = priced["date"].iloc[int(ratio.idxmin())] if len(priced) else ""
    all_ratio = {
        "n": len(priced), "min": float(ratio.min()) if len(priced) else float("nan"), "min_date": low,
        "min_what": "a monthly trade of the sample" if low in set(x["date"]) else ("not a trade of the study: stuck member" if low in stuck else ("no outcome yet" if low in no_outcome else "not a monthly trade of the study")),
    }  # fmt: skip
    # the monthly trades outside the pass's list: without a row, or with a row from elsewhere
    outside = sorted(set(monthly["date"]) - set(in_pass))
    first_of_year = set(d.groupby(d["date"].str[:4])["date"].min())
    outside_rows = [
        {"date": t, "state": "priced" if t in set(x["date"]) else ("failed" if t in set(failed["date"]) else "not priced"), "first_entry_of_its_year": t in first_of_year}
        for t in outside if t in set(lc["date"])
    ]  # fmt: skip
    meta = {
        "rows": len(lc), "unreadable": unreadable, "priced": len(priced), "failed": sorted(failed["date"]),
        "failed_reasons": failed[["date", "reason"]].to_dict("records"), "other_rows": other[["date", "status"]].to_dict("records"),
        "pass_dates": in_pass, "pass_with_row": sorted(set(in_pass) & set(lc["date"])), "pass_pending": sorted(set(in_pass) - set(lc["date"]) - set(unreadable)),
        "rows_outside_pass": sorted(set(lc["date"]) - set(in_pass)), "not_trade": why, "stuck": stuck, "no_outcome": no_outcome,
        "monthly_flag": int(weekly["monthly"].sum()), "monthly_added": sorted(monthly.loc[~monthly["monthly"], "date"]),
        "monthly_not_reachable": sorted(set(monthly["date"]) - set(reach["date"])), "reach": len(reach),
        "monthly_outside_pass": outside, "monthly_outside_with_row": outside_rows, "folder": folder, "all_priced_ratio": all_ratio,
        "pending_trades": sorted(set(reach["date"]) - set(lc["date"])), "failed_trades": sorted(set(reach["date"]) & set(failed["date"])),
        "months_without_trade": [m for m in range(monthly["month_index"].min(), monthly["month_index"].max() + 1) if m not in set(monthly["month_index"])],
        "commits": sorted(set(priced["git_commit"].dropna().astype(str))), "digests": sorted(set(priced["config_digest"].dropna().astype(str))),
        "status_counts": x["status"].value_counts().to_dict(), "check_reasons": x.loc[x["status"] == "check", "reason"].value_counts().to_dict(),
        "non_finite": {k: v for k, v in non_finite.items() if k in set(x["date"])}, "by_year": by_year,
        "years_not_priced": [y for y in by_year["entry_year"] if per_year.loc[y, "priced"] == 0],
        "carry_decisions": int(x["carries_decisions"].astype(bool).sum()),
    }  # fmt: skip
    return {"weekly": weekly, "monthly": monthly, "priced": x, "meta": meta}


# ------------------------------------------------------------------------------ the statistics
def hh_sum(u: Any, pos: Any, lag: int) -> tuple[float, bool]:
    """Hansen-Hodrick standard error of an estimator whose error is the sum of one term per
    trade, ``u_i`` (already divided by the sample sizes), observed at the integer positions
    ``pos``: the square root of ``sum over the pairs with |pos_i - pos_j| <= lag of u_i u_j``.
    When that sum is not positive (to rounding): Newey-West (Bartlett) at ``2 * lag``, as the
    study does; the second value says so.  NaN below three terms; 0 when every term is 0."""
    v, p = np.asarray(u, dtype=np.float64), np.asarray(pos, dtype=np.int64)
    if len(v) < 3:
        return float("nan"), False
    total = float((v * v).sum())
    if total == 0.0:
        return 0.0, False
    dist = np.abs(p[:, None] - p[None, :])
    prod = v[:, None] * v[None, :]
    var, fallback = float(prod[dist <= lag].sum()), False
    # not positive, to rounding: when every pair is within the lag the sum is (sum u)^2 = 0
    if var <= 1e-12 * total:
        kernel = np.where(dist <= 2 * lag, 1.0 - dist / (2 * lag + 1.0), 0.0)
        var, fallback = float((prod * kernel).sum()), True
    return float(np.sqrt(max(var, 0.0))), fallback


def hh_se(x: Any, pos: Any, lag: int, weights: Any = None) -> tuple[float, bool]:
    """Hansen-Hodrick standard error of the (weighted) mean of ``x`` observed at the integer
    positions ``pos``: :func:`hh_sum` of the terms ``w_i (x_i - mean) / sum w``.  With ``pos =
    0, 1, 2, ...`` it is the study's ``disp_stats.mean_se``; with holes in ``pos`` the trades
    stay at their place in time, as in ``disp_stats.subset_mean_se``.  The second value says
    whether the Newey-West fallback was used.  NaN below three observations."""
    v = np.asarray(x, dtype=np.float64)
    w = np.ones(len(v)) if weights is None else np.asarray(weights, dtype=np.float64)
    if len(v) < 3:
        return float("nan"), False
    return hh_sum(w * (v - (w * v).sum() / w.sum()) / w.sum(), pos, lag)


def block_index(n: int, block: int, seed: int) -> np.ndarray:
    """The resampled row indices of the study's circular block bootstrap (the lines of
    ``disp_stats.bootstrap_ratio``); read-only.  At the study's seed the indices are kept for
    the next statistic on a sample of the same size."""
    return _block_index(n, block) if seed == SEED else _draw_index(n, block, seed)


def _draw_index(n: int, block: int, seed: int) -> np.ndarray:
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(N_RESAMPLES, int(np.ceil(n / block))))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(N_RESAMPLES, -1)[:, :n]  # fmt: skip
    idx = idx.astype(np.int32)
    idx.setflags(write=False)
    return idx


@functools.lru_cache(maxsize=48)
def _block_index(n: int, block: int) -> np.ndarray:
    return _draw_index(n, block, SEED)


def boot_ratio(num: Any, den: Any, block: int, seed: int = SEED) -> dict[str, float]:
    """``sum(num) / sum(den)`` with the 95 % interval and the standard deviation of the study's
    block bootstrap; at the study's seed it is checked against the study's own function.  A
    mean is the ratio with ``den = 1`` and a weighted mean the one with ``num = w x, den = w``."""
    a = np.column_stack([np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)])
    if not np.isfinite(a).all():
        raise ValueError("a non-finite value in a sample (the study's bootstrap would drop it)")
    s = a[block_index(len(a), block, seed)].sum(axis=1)
    draws = s[:, 0] / s[:, 1]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    out = {"value": float(a[:, 0].sum() / a[:, 1].sum()), "lo": float(lo), "hi": float(hi), "se": float(draws.std(ddof=1))}  # fmt: skip
    if seed == SEED:
        ref = st.bootstrap_ratio(a[:, 0], a[:, 1], block)
        if not np.allclose([out["value"], out["lo"], out["hi"]], ref, rtol=0, atol=1e-12):
            raise ValueError("boot_ratio does not reproduce disp_stats.bootstrap_ratio")
    return out


def boot_interval(draws: np.ndarray) -> tuple[float, float]:
    """Percentiles 2.5 and 97.5 of the bootstrap draws of a statistic; no interval when more
    than 1 % of the draws are undefined (a resample without a trade of one of the groups)."""
    ok = np.isfinite(draws)
    if ok.mean() < 0.99:
        return float("nan"), float("nan")
    lo, hi = np.percentile(draws[ok], [2.5, 97.5])
    return float(lo), float(hi)


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(x)
    c = np.cumsum(w[order])
    return float(x[order][np.searchsorted(c, 0.5 * c[-1])])


def describe(x: Any, pos: Any, lags: tuple[int, ...], block: int, weights: Any = None, seeds: bool = False) -> dict[str, Any]:  # fmt: skip
    """Mean of ``x`` (weighted when ``weights`` is given), in the units of ``x``: the
    Hansen-Hodrick standard error at ``lags[1]`` (the convention) and at the other lags, each
    with its fallback flag, the t-ratio and the block-bootstrap interval (none on a sample
    shorter than ``MIN_BLOCKS`` blocks); the share of negative
    values with its Hansen-Hodrick standard error (on the indicator, at ``lags[1]``) and its
    block-bootstrap interval; the median with its block-bootstrap interval (unweighted samples
    only); the share of positive values; with ``seeds`` the range of the bounds of the three
    intervals over ``SEEDS_RANGE`` and, per interval, on how many of those seeds it answers the
    page's question as it does at the study's seed (the mean's interval excludes zero or not, the
    share's excludes one half or not, the median's excludes zero or not)."""
    v = np.asarray(x, dtype=np.float64)
    w = np.ones(len(v)) if weights is None else np.asarray(weights, dtype=np.float64)
    nan = float("nan")
    out: dict[str, Any] = {
        "n": len(v), "mean": float((w * v).sum() / w.sum()) if len(v) else nan, "hit": float((w * (v > 0)).sum() / w.sum()) if len(v) else nan,
        "neg": float((w * (v < 0)).sum() / w.sum()) if len(v) else nan, "neg_se": nan, "neg_fallback": False, "neg_lo": nan, "neg_hi": nan,
        "median": (float(np.median(v)) if weights is None else weighted_median(v, w)) if len(v) else nan, "median_lo": nan, "median_hi": nan,
        "se": nan, "fallback": False, "t": nan, "lo": nan, "hi": nan, "se_boot": nan, "lags": lags, "block": block,
        **{f"se_lag{k}": nan for k in lags}, **{f"fallback_lag{k}": False for k in lags},
        "lo_seed_min": nan, "lo_seed_max": nan, "hi_seed_min": nan, "hi_seed_max": nan,
        **{f"{k}_{b}_seed_{e}": nan for k in ("neg", "median") for b in ("lo", "hi") for e in ("min", "max")},
        "seeds": 0, "seeds_same_mean": nan, "seeds_same_neg": nan, "seeds_same_median": nan,
    }  # fmt: skip
    if len(v) < MIN_N:
        return out
    for k in lags:
        out[f"se_lag{k}"], out[f"fallback_lag{k}"] = hh_se(v, pos, k, weights)
    out["se"], out["fallback"] = out[f"se_lag{lags[1]}"], out[f"fallback_lag{lags[1]}"]
    out["t"] = out["mean"] / out["se"] if out["se"] > 0 else nan
    negative = (v < 0).astype(np.float64)
    out["neg_se"], out["neg_fallback"] = hh_se(negative, pos, lags[1], weights)
    if len(v) < MIN_BLOCKS * block:
        return out
    b = boot_ratio(w * v, w, block)
    out["lo"], out["hi"], out["se_boot"] = b["lo"], b["hi"], b["se"]
    b = boot_ratio(w * negative, w, block)
    out["neg_lo"], out["neg_hi"] = b["lo"], b["hi"]
    if weights is None:
        out["median_lo"], out["median_hi"] = boot_interval(np.median(v[block_index(len(v), block, SEED)], axis=1))  # fmt: skip
    if seeds:
        bounds: dict[str, list[Any]] = {"mean": [], "neg": [], "median": []}
        for seed in SEEDS_RANGE:
            idx = block_index(len(v), block, seed)
            total = w[idx].sum(axis=1)
            bounds["mean"].append(np.percentile((w * v)[idx].sum(axis=1) / total, [2.5, 97.5]))
            bounds["neg"].append(np.percentile((w * negative)[idx].sum(axis=1) / total, [2.5, 97.5]))  # fmt: skip
            if weights is None:
                bounds["median"].append(np.percentile(np.median(v[idx], axis=1), [2.5, 97.5]))
        out["seeds"] = len(SEEDS_RANGE)
        for key, prefix, ref, level in (("mean", "", (out["lo"], out["hi"]), 0.0), ("neg", "neg_", (out["neg_lo"], out["neg_hi"]), 0.5),
                                        ("median", "median_", (out["median_lo"], out["median_hi"]), 0.0)):  # fmt: skip
            if not bounds[key]:
                continue
            lo, hi = np.array(bounds[key]).T
            out[f"{prefix}lo_seed_min"], out[f"{prefix}lo_seed_max"] = float(lo.min()), float(lo.max())  # fmt: skip
            out[f"{prefix}hi_seed_min"], out[f"{prefix}hi_seed_max"] = float(hi.min()), float(hi.max())  # fmt: skip
            out[f"seeds_same_{key}"] = int((((lo > level) | (hi < level)) == (ref[0] > level or ref[1] < level)).sum())  # fmt: skip
    return out


def year_stats(x: Any, years: Any) -> dict[str, Any]:
    """The means of ``x`` per entry year: how many are negative, which are positive, which are
    closer to zero than ``NEAR_ZERO``, the two-sided binomial p of the count of negative years
    against one half (as if the years were independent, which they are not), and two t-ratios of
    the mean of the year means, each with as many degrees of freedom as years less one.

    * The plain one: standard deviation of the year means over the square root of their number,
      with its 5 % and 1 % two-sided critical values and its two-sided p-value.
    * The one with one lag between adjacent years: Hansen-Hodrick with one lag on the year means
      at their calendar year (:func:`hh_se`), with its p-value and its fallback flag.

    The windows of adjacent entry years overlap, so adjacent year means are positively
    correlated (their lag-1 autocorrelation is returned) and the plain t overstates ``|t|``."""
    nan = float("nan")
    means = pd.Series(np.asarray(x, dtype=np.float64)).groupby(np.asarray(years)).mean()
    k = len(means)
    near = [(str(y), float(v)) for y, v in means.items() if abs(v) < NEAR_ZERO]
    out: dict[str, Any] = {
        "entry_years": k, "entry_years_negative": int((means < 0).sum()), "entry_years_positive": ", ".join(str(y) for y in means.index[means > 0]),
        "entry_years_near_zero": "; ".join(f"{y} at {v:+.3f}, counted {'negative' if v < 0 else 'positive'}" for y, v in near),
        "entry_years_binomial_p": float(sps.binomtest(int((means < 0).sum()), k, 0.5).pvalue) if k else nan,
        "year_means_mean_pct": nan, "year_means_se_pct": nan, "year_means_t": nan, "year_means_df": max(k - 1, 0), "year_means_t_crit5": nan, "year_means_t_crit1": nan,
        "year_means_p": nan, "year_means_lag1_autocorr": nan, "year_means_se_lag1_pct": nan, "year_means_t_lag1": nan, "year_means_p_lag1": nan, "year_means_lag1_fallback": False,
    }  # fmt: skip
    if k >= 3:
        se = float(means.std(ddof=1)) / np.sqrt(k)
        out["year_means_mean_pct"], out["year_means_se_pct"] = float(means.mean()), se
        out["year_means_t"] = float(means.mean()) / se if se > 0 else nan
        out["year_means_t_crit5"], out["year_means_t_crit1"] = float(sps.t.ppf(0.975, k - 1)), float(sps.t.ppf(0.995, k - 1))  # fmt: skip
        out["year_means_p"] = float(2.0 * sps.t.sf(abs(out["year_means_t"]), k - 1)) if se > 0 else nan  # fmt: skip
        pos, u = means.index.astype(int).to_numpy(), (means - means.mean()).to_numpy()
        adjacent = np.abs(pos[:, None] - pos[None, :]) == 1
        if (u * u).sum() > 0:
            out["year_means_lag1_autocorr"] = float(0.5 * (u[:, None] * u[None, :])[adjacent].sum() / (u * u).sum())  # fmt: skip
        se1, out["year_means_lag1_fallback"] = hh_se(means.to_numpy(), pos, 1)
        out["year_means_se_lag1_pct"] = se1
        if se1 > 0:
            out["year_means_t_lag1"] = float(means.mean()) / se1
            out["year_means_p_lag1"] = float(2.0 * sps.t.sf(abs(out["year_means_t_lag1"]), k - 1))
    return out


def windows_stats(sign: str, t_hh: float, yr: dict[str, Any]) -> dict[str, Any]:
    """The three tests of a mean once its one-year windows are counted, from its Hansen-Hodrick
    t and the statistics of its entry-year means (:func:`year_stats`): the two-sided p of the
    Hansen-Hodrick t read on the degrees of freedom of the year means (years less one) and not
    as a normal variable; the largest of the three p-values (that one, the plain t of the year
    means and the t with one lag), which is the most cautious; and the grade it gives a mean
    that the rule calls negative or positive (:func:`windows_rule`)."""
    nan, df = float("nan"), yr["year_means_df"]
    p_hh = float(2.0 * sps.t.sf(abs(t_hh), df)) if np.isfinite(t_hh) and df >= 2 else nan
    ps = [yr["year_means_p"], yr["year_means_p_lag1"], p_hh]
    cautious = float(max(ps)) if all(np.isfinite(p) for p in ps) else nan
    return {"hh_t_p_on_year_df": p_hh, "windows_p_most_cautious": cautious, "sign_once_windows_counted": windows_rule(sign, cautious)}  # fmt: skip


def windows_of(x: Any, years: Any, sign: str, t_hh: float) -> dict[str, Any]:
    """:func:`year_stats` of ``x`` and the :func:`windows_stats` of its mean, as one record."""
    yr = year_stats(x, years)
    return {**yr, **windows_stats(sign, t_hh, yr)}


def next_year_share(entry: Any, expiry: Any) -> float:
    """The mean share of a trade's window (entry to expiry) that lies in the calendar year after
    its entry year."""
    s, e = np.asarray(entry, dtype="datetime64[D]"), np.asarray(expiry, dtype="datetime64[D]")
    year = s.astype("datetime64[Y]")
    jan1 = (year + np.timedelta64(1, "Y")).astype("datetime64[D]")
    end = (year + np.timedelta64(2, "Y")).astype("datetime64[D]")
    inside = (np.minimum(e, end) - jan1).astype(np.float64)
    return float(np.mean(np.clip(inside, 0.0, None) / (e - s).astype(np.float64)))


def binomial_p(negative: int, trades: int) -> float:
    """Two-sided binomial p of a count of negative trades against one half."""
    return float(sps.binomtest(int(negative), int(trades), 0.5).pvalue)


def chain_rule(p_min: float, p_max: float) -> str:
    """The count of negative trades on the median chain against one half, from the smallest and
    the largest two-sided binomial p over the chains that have the median count (they can differ
    in length): ``CHAIN_YES`` when every p is below ``CHAIN_P``, ``CHAIN_NO`` when none is, and
    ``CHAIN_LIMIT`` when some are and some are not: the verdict then turns on which of these
    chains is taken as the median one.  Empty when there is no chain."""
    if not (np.isfinite(p_min) and np.isfinite(p_max)):
        return ""
    if p_max < CHAIN_P:
        return CHAIN_YES
    return CHAIN_NO if p_min >= CHAIN_P else CHAIN_LIMIT


def chain_needs(trades: int) -> int:
    """The smallest count of trades of one sign, above one half, whose two-sided binomial p on a
    chain of ``trades`` trades is below ``CHAIN_P``; ``trades + 1`` when no count passes."""
    return next((k for k in range(trades // 2 + 1, trades + 1) if binomial_p(k, trades) < CHAIN_P), trades + 1)  # fmt: skip


def chain_stats(x: Any, entry: Any, expiry: Any) -> dict[str, Any]:
    """Chains of non-overlapping trades of a sample in date order: from each trade entered in
    the first 365 days, the next trade entered on or after the expiry of the current one, and so
    on to the end of the sample.  Per chain: its number of trades, of negative trades, its mean
    and the plain t-ratio of that mean.  Returned: the number of chains, the range of their
    lengths, the median chain (the middle one when the chains are ordered by their count of
    negative trades, then by their length; the lower of the two middle ones when the number of
    chains is even) with its count and its two-sided binomial p against one half, the range of
    the share of negative trades, and the t-ratios (median, range, how many pass their own 5 %
    critical value).

    The chains that have the median count need not have the same length, and the p of the count
    depends on the length.  So are also returned: how many chains have the median count, the
    shortest and the longest of them with the p of the count on each, the smallest and the
    largest p over all of them, the verdict of :func:`chain_rule` on these two, and the number
    of chains (all of them, not only the middle ones) whose own count has a p below
    ``CHAIN_P``."""
    nan = float("nan")
    keys = ("chain_trades_min", "chain_trades_max", "chain_median_negative", "chain_median_trades", "chain_median_binomial_p", "chain_share_negative_min",
            "chain_share_negative_max", "chain_means_mean_pct", "chain_means_negative", "chain_t_median", "chain_t_min", "chain_t_max", "chain_t_beyond_crit5",
            "chain_median_tied", "chain_median_trades_shortest", "chain_median_trades_longest", "chain_median_binomial_p_shortest", "chain_median_binomial_p_longest",
            "chain_median_binomial_p_min", "chain_median_binomial_p_max", "chains_count_below_5pct")  # fmt: skip
    out: dict[str, Any] = {"chains": 0, **dict.fromkeys(keys, nan), "chain_median_at_5pct": ""}
    v = np.asarray(x, dtype=np.float64)
    if len(v) < MIN_N:
        return out
    s, e = np.asarray(entry, dtype="datetime64[D]"), np.asarray(expiry, dtype="datetime64[D]")
    if not (np.diff(s.astype(np.int64)) > 0).all():
        raise ValueError("chain_stats: the trades are not in date order")
    rows = []
    for first in np.flatnonzero(s < s[0] + np.timedelta64(365, "D")):
        chain = [int(first)]
        while True:
            later = np.flatnonzero(s >= e[chain[-1]])
            if not len(later):
                break
            chain.append(int(later[0]))
        c = v[chain]
        if len(c) < 3:
            continue
        se = float(c.std(ddof=1)) / np.sqrt(len(c))
        t = float(c.mean()) / se if se > 0 else nan
        rows.append((len(c), int((c < 0).sum()), float(c.mean()), t, float(sps.t.ppf(0.975, len(c) - 1))))  # fmt: skip
    if not rows:
        return out
    n, neg, mean, t, crit = (np.array(col) for col in zip(*rows, strict=True))
    mid = sorted(range(len(rows)), key=lambda i: (neg[i], n[i]))[(len(rows) - 1) // 2]
    p = np.array([binomial_p(k, m) for k, m in zip(neg, n, strict=True)])
    tied = np.flatnonzero(neg == neg[mid])  # the chains that have the median count
    short, long_ = tied[np.argmin(n[tied])], tied[np.argmax(n[tied])]
    out.update({
        "chains": len(rows), "chain_trades_min": int(n.min()), "chain_trades_max": int(n.max()), "chain_median_negative": int(neg[mid]), "chain_median_trades": int(n[mid]),
        "chain_median_binomial_p": float(p[mid]), "chain_share_negative_min": float((neg / n).min()),
        "chain_share_negative_max": float((neg / n).max()), "chain_means_mean_pct": float(mean.mean()), "chain_means_negative": int((mean < 0).sum()),
        "chain_t_median": float(np.nanmedian(t)), "chain_t_min": float(np.nanmin(t)), "chain_t_max": float(np.nanmax(t)), "chain_t_beyond_crit5": int((np.abs(t) >= crit).sum()),
        "chain_median_tied": len(tied), "chain_median_trades_shortest": int(n[short]), "chain_median_trades_longest": int(n[long_]),
        "chain_median_binomial_p_shortest": float(p[short]), "chain_median_binomial_p_longest": float(p[long_]),
        "chain_median_binomial_p_min": float(p[tied].min()), "chain_median_binomial_p_max": float(p[tied].max()),
        "chains_count_below_5pct": int((p < CHAIN_P).sum()), "chain_median_at_5pct": chain_rule(float(p[tied].min()), float(p[tied].max())),
    })  # fmt: skip
    return out


def sign_rule(mean: float, t: float, lo: float, hi: float) -> str:
    """The rule of the page for the sign of a mean.  ``negative`` or ``positive``: the 95 %
    block-bootstrap interval excludes zero on the side of the mean AND ``|t| >= T_CRIT``.
    ``about zero``: not so, and the mean is less than ``ABOUT_ZERO`` standard errors from zero.
    ``not established``: neither.  ``no error``: no standard error or no interval."""
    if not (np.isfinite(mean) and np.isfinite(t) and np.isfinite(lo) and np.isfinite(hi)):
        return "no error"
    if abs(t) >= T_CRIT and ((mean < 0 and hi < 0) or (mean > 0 and lo > 0)):
        return "negative" if mean < 0 else "positive"
    return "about zero" if abs(t) < ABOUT_ZERO else "not established"


def windows_rule(sign: str, p: float) -> str:
    """For a mean the rule calls negative or positive, its grade once the windows are counted,
    from the most cautious (largest) two-sided p of the three tests of :func:`windows_stats`:
    ``passes at 1 %`` (p below 0.01), ``passes at 5 %, not at 1 %`` (0.01 to 0.025), ``at the
    5 % limit`` (0.025 to 0.05) or ``does not pass at 5 %``; ``not tested`` when a test has no
    value (fewer than three entry years); empty for any other mean."""
    if sign not in ("negative", "positive"):
        return ""
    if not np.isfinite(p):
        return "not tested"
    if p < 0.01:
        return "passes at 1 %"
    if p < LIMIT_P:
        return "passes at 5 %, not at 1 %"
    return "at the 5 % limit" if p < 0.05 else "does not pass at 5 %"


def self_checks(S: dict[str, Any]) -> dict[str, float]:
    """:func:`hh_se` against the study's two functions, on the study's own columns: on the
    weekly trades at their row (``mean_se``, 50 lags) and on the priced trades at their row
    among the monthly trades (``subset_mean_se``, 11 lags; compared when there are at least
    ``MIN_N`` priced trades and the Hansen-Hodrick variance is positive).  Raises when they
    differ."""
    out = {}
    w, m, x = S["weekly"], S["monthly"], S["priced"]
    for col in ("GAP_U", "GAP_H"):
        mine = hh_se(w[col], np.arange(len(w)), LAG_WEEKLY)[0]
        out[f"weekly.{col}"] = abs(mine - st.mean_se(w[col], LAG_WEEKLY)[1])
        mask = m["date"].isin(x["date"]).to_numpy()
        mine, fallback = hh_se(m.loc[mask, col], np.flatnonzero(mask), LAG_MONTHLY)
        if MIN_N <= len(x) < len(m) and not fallback:
            out[f"subset.{col}"] = abs(mine - st.subset_mean_se(m[col], mask, LAG_MONTHLY)[1])
    bad = {k: v for k, v in out.items() if not v < 1e-12}
    if bad:
        raise ValueError(f"hh_se does not reproduce the study's functions: {bad}")
    return out


# ------------------------------------------------------------------------------ the validation
def study_values(weekly: pd.DataFrame) -> dict[str, float]:
    """This script's values of the study's printed one-year numbers, on the weekly trades; and,
    under ``n_sample.<sample>`` and ``n_sample.strip.<sample>``, the size of the sample each is
    computed on (the trades of the sample, or its windows with a usable strip)."""
    mine: dict[str, float] = {}
    for s, g in (("all", weekly), ("IS", weekly[weekly["IS"]]), ("OOS", weekly[~weekly["IS"]])):
        for col in ("GAP_U", "GAP_H", "PF_U", "PF_H", "PKG_v_U", "PKG_v_H", "GAP_rho_U"):
            q = st.describe(g[col], LAG_WEEKLY)
            if int(q["n"]) != len(g):
                raise ValueError(f"{col}, {s}: the study's describe uses {q['n']} of the {len(g)} trades")  # fmt: skip
            mine[f"{col}.{s}.mean"], mine[f"{col}.{s}.t"] = 100 * q["mean"], q["t"]
            mine[f"{col}.{s}.sd"], mine[f"{col}.{s}.hit"] = 100 * q["sd"], q["hit"]
            mine[f"{col}.{s}.q05"], mine[f"{col}.{s}.n"] = 100 * q["q05"], q["n"]
            mine[f"{col}.{s}.t@{LAG_DRAFT}"] = st.describe(g[col], LAG_DRAFT)["t"]
            mine[f"{col}.{s}.t.fallback"] = float(hh_se(g[col], np.arange(len(g)), LAG_WEEKLY)[1])
        for col in ("P_G", "G", "P_D", "D"):
            mine[f"{col}.{s}"] = 100 * g[col].mean()
        mine[f"P_PKG.{s}"] = 100 * (g["P_D"] - g["P_G"]).mean()
        mine[f"P_D_over_D.{s}"] = g["P_D"].sum() / g["D"].sum()
        ok = g[g["strip_ok"]]
        mine[f"P_D_over_D_strip.{s}"] = ok["P_D"].sum() / ok["D"].sum()
        mine[f"P_D_over_D_strip.{s}.n"] = len(ok)
        mine[f"n_sample.{s}"], mine[f"n_sample.strip.{s}"] = len(g), len(ok)
    return mine


def digits_of(text: str) -> int:
    return len(text.split(".")[1]) if "." in text else 0


def validation(weekly: pd.DataFrame) -> pd.DataFrame:
    """The study's printed one-year numbers against this script's: one row per printed number,
    with the size of the sample of the statistic (``n_trades``: the weekly trades of the sample
    named by the key, or its windows with a usable strip for the numbers on the strip)."""
    mine = study_values(weekly)
    rows: list[dict[str, Any]] = []

    def add(key: str, what: str, source: str, printed: str, tol: float = 0.0, found: bool = True) -> None:  # fmt: skip
        digits, value = digits_of(printed), mine[key]
        match = found and (abs(value - float(printed)) <= tol if tol else round(value, digits) == round(float(printed), digits))  # fmt: skip
        alt = mine.get(f"{key}@{LAG_DRAFT}", float("nan"))
        col, s = key.split(".")[:2]
        size = int(mine[f"n_sample.strip.{s}" if col == "P_D_over_D_strip" else f"n_sample.{s}"])
        rows.append({"key": key, "quantity": what, "source": source, "printed": printed, "this_script": value, "digits": digits,
                     "match": bool(match), "line_found": found, "this_script_51_lags": alt, "rests_on": ", ".join(rests_on(key)),
                     "newey_west_fallback": bool(mine.get(f"{key}.fallback", 0.0)), "n_trades": size})  # fmt: skip

    def label(key: str) -> str:
        col, s, *stat = key.split(".")
        return f"{WHAT[col]}, {SAMPLE_WORDS[s]}" + (f": {STAT[stat[0]]}" if stat else "")

    t17 = pd.read_csv(TABLES / "T17_other_runs.csv", dtype=str)
    t17 = t17[(t17["run"] == f"{BASKET} {TENOR}") & (t17["structure"] == "GAP")].set_index("sample")
    for s in ("all", "IS", "OOS"):
        for column, stat in (("mean", "mean"), ("t", "t"), ("sd", "sd"), ("hit", "hit"), ("5 %", "q05"), ("n", "n")):  # fmt: skip
            add(f"GAP_U.{s}.{stat}", label(f"GAP_U.{s}.{stat}"), SRC_T17, t17.loc[s, column])
        add(f"P_G.{s}", label(f"P_G.{s}"), SRC_T17, t17.loc[s, "mean P_G"])
        add(f"G.{s}", label(f"G.{s}"), SRC_T17, t17.loc[s, "mean G"])
    t13 = pd.read_csv(TABLES / "T13_other_runs.csv", dtype=str)
    t13 = t13[t13["run"] == f"{BASKET} {TENOR}"].set_index("structure")
    for structure, col in (("PF U", "PF_U"), ("PF H", "PF_H"), ("PKG_v U", "PKG_v_U"), ("PKG_v H", "PKG_v_H"), ("GAP U", "GAP_U")):  # fmt: skip
        for s in ("all", "IS", "OOS"):
            add(f"{col}.{s}.mean", label(f"{col}.{s}.mean"), SRC_T13, t13.loc[structure, f"{s} mean"])  # fmt: skip
            add(f"{col}.{s}.t", label(f"{col}.{s}.t"), SRC_T13, t13.loc[structure, f"{s} t"])
    # the hedged gap is not a row of the study's one-year tables: it is the difference of two
    for s in ("all", "IS", "OOS"):
        diff = float(t13.loc["PF H", f"{s} mean"]) - float(t13.loc["PKG_v H", f"{s} mean"])
        add(f"GAP_H.{s}.mean", f"hedged gap, {SAMPLE_WORDS[s]}: the printed hedged forward minus the printed hedged package (equal within the rounding of the two, 0.001)",
            SRC_T13, f"{diff:.3f}", tol=0.001)  # fmt: skip
    t4 = pd.read_csv(TABLES / "T4_T6_other_tenors.csv", dtype=str)
    t4 = t4[t4["run"] == f"{BASKET} {TENOR}"].set_index("sample")
    for s in ("all", "IS", "OOS"):
        add(f"P_D_over_D_strip.{s}", label(f"P_D_over_D_strip.{s}"), SRC_T4, t4.loc[s, "P_D / D"])
        add(f"P_D_over_D_strip.{s}.n", label(f"P_D_over_D_strip.{s}.n"), SRC_T4, t4.loc[s, "windows"])  # fmt: skip
    t2 = pd.read_csv(TABLES / "T2_other_runs.csv", dtype=str)
    t2 = t2[(t2["run"] == f"{BASKET} {TENOR}") & (t2.iloc[:, 1] == "all")].iloc[0]
    add("D.all", label("D.all"), SRC_T2, t2["mean D"])
    add("G.all", label("G.all"), SRC_T2, t2["mean G"])
    tex = DRAFT.read_text()
    for where, line, items in TEX_LINES:
        found = line in tex
        number = 1 + tex[: tex.index(line)].count("\n") if found else 0
        source = f"{SRC_TEX}, line {number} ({where})" if found else f"{SRC_TEX}: {where} NOT FOUND as quoted"  # fmt: skip
        for key, printed in items:
            add(key, label(key), source, printed, found=found)
    # the study's other gap, against the correlation-neutral package: the study prints its held
    # numbers at one year (added after the rows above, whose order is the first page's)
    rho = pd.read_csv(TABLES / "T17_other_runs.csv", dtype=str)
    rho = rho[(rho["run"] == f"{BASKET} {TENOR}") & (rho["structure"] == "GAP_ρ")].set_index("sample")  # fmt: skip
    for s in ("all", "IS", "OOS"):
        for column, stat in (("mean", "mean"), ("t", "t"), ("sd", "sd"), ("hit", "hit"), ("5 %", "q05"), ("n", "n")):  # fmt: skip
            add(f"GAP_rho_U.{s}.{stat}", label(f"GAP_rho_U.{s}.{stat}"), SRC_T17_RHO, rho.loc[s, column])  # fmt: skip
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ the rows
def subsamples(x: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame, bool]]:
    """``(key, label, trades, weighted)``: the LC sample and its sub-samples."""
    year = x["date"].str[:4]
    ok = x[x["status"] == "ok"]
    ok_label = f"status ok only (not a filter on the index fit: {int(ok['flag_clip'].sum())} of the {len(ok)} are flagged for the clipped mass)"  # fmt: skip
    return [
        ("priced", "all priced trades", x, False),
        ("entries_2007_2016", "entries 2007-2016", x[x["IS"]], False),
        ("entries_2017_2025", "entries 2017-2025", x[~x["IS"]], False),
        ("no_unscreened", "without the dates where a name is kept unscreened", x[~x["flag_unscreened"]], False),
        ("clip_not_flagged", "dates NOT flagged for the clipped mass", x[~x["flag_clip"]], False),
        ("clip_flagged", "dates flagged for the clipped mass", x[x["flag_clip"]], False),
        ("no_djx_repair", "without the dates whose DJX target was repaired", x[~x["flag_djx_repaired"]], False),
        ("no_2008_2009", "without the entries of 2008 and 2009", x[~year.isin(["2008", "2009"])], False),
        ("status_ok", ok_label, ok, False),
        ("no_extrapolated", "without the dates where a name is priced beyond its last kept expiry", x[~x["flag_extrapolated"]], False),
        ("year_weighted", "all priced trades, weighted to the study's monthly trades per entry year", x, True),
    ]  # fmt: skip


def gap_rows(S: dict[str, Any], valid: dict[str, bool], structures: tuple[tuple[str, str, str], ...] = STRUCTURES, headline: bool = False) -> pd.DataFrame:  # fmt: skip
    """One row per sample, structure and price: the mean P&L of the gap with its errors, the
    sign the rule gives it, the share of negative trades, the median, the entry-year means and
    the chains of non-overlapping trades.  ``headline``: the three main samples only."""
    rows = []
    w, m = S["weekly"], S["monthly"]
    specs: list[tuple[Any, ...]] = [
        ("weekly", SAMPLE_LABEL["weekly"], w, np.arange(len(w)), LAGS_WEEKLY, BLOCK_WEEKLY, False, ("copula",)),
        ("monthly", SAMPLE_LABEL["monthly"], m, m["month_index"], LAGS_MONTHLY, BLOCK_MONTHLY, False, ("copula",)),
    ]  # fmt: skip
    for half, mask_w, mask_m in (("2007_2016", w["IS"], m["IS"]), ("2017_2025", ~w["IS"], ~m["IS"])):  # fmt: skip
        years = half.replace("_", "-")
        specs.append((f"weekly_{half}", f"the study's weekly trades, entries {years}", w[mask_w], np.arange(int(mask_w.sum())), LAGS_WEEKLY, BLOCK_WEEKLY, False, ("copula",)))  # fmt: skip
        specs.append((f"monthly_{half}", f"the study's monthly trades, entries {years}", m[mask_m], m.loc[mask_m, "month_index"], LAGS_MONTHLY, BLOCK_MONTHLY, False, ("copula",)))  # fmt: skip
    specs += [(k, label, g, g["month_index"], LAGS_MONTHLY, BLOCK_MONTHLY, wt, PRICES) for k, label, g, wt in subsamples(S["priced"])]  # fmt: skip
    if headline:
        specs = [q for q in specs if q[0] in ("weekly", "monthly", "priced")]
    for key, label, g, pos, lags, block, weighted, prices in specs:
        weights = g["year_weight"].to_numpy(float) if weighted else None
        for structure, col, _ in structures:
            for price in prices:
                if price != "copula" and not valid[structure]:
                    continue
                shift = g["P_D"] - g[PRICE_COL[price]]
                pnl = 100 * (g[col] + shift)
                q = describe(pnl, pos, lags, block, weights, seeds=key in SAMPLE_LABEL)
                sh = describe(100 * shift, pos, lags, block, weights)
                yr = year_stats(pnl, g["date"].str[:4])
                sign = sign_rule(q["mean"], q["t"], q["lo"], q["hi"])
                w_ = np.ones(len(g)) if weights is None else weights
                rows.append({
                    "sample": key, "sample_label": label, "n_trades": q["n"], "first_entry": g["date"].min() if len(g) else "", "last_entry": g["date"].max() if len(g) else "",
                    "structure": structure, "price": price, "mean_pnl_pct": q["mean"], "se_hh_pct": q["se"], "hh_lags": lags[1], "newey_west_fallback": q["fallback"],
                    "se_method": se_method(lags[1], q["fallback"]) if np.isfinite(q["se"]) else "none: fewer than 10 trades",
                    "hh_position": "row of the weekly series" if key.startswith("weekly") else "calendar month of entry", "t_ratio": q["t"],
                    "ci95_lo_pct": q["lo"], "ci95_hi_pct": q["hi"], "se_boot_pct": q["se_boot"], "boot_block": block, "boot_seed": SEED,
                    "sign_by_rule": sign, **windows_stats(sign, q["t"], yr),
                    "share_negative": q["neg"], "share_negative_se_hh": q["neg_se"], "share_negative_se_fallback": q["neg_fallback"],
                    "share_negative_z_against_half": (q["neg"] - 0.5) / q["neg_se"] if np.isfinite(q["neg_se"]) and q["neg_se"] > 0 else float("nan"),
                    "share_negative_ci95_lo": q["neg_lo"], "share_negative_ci95_hi": q["neg_hi"],
                    "share_differs_from_half": bool(q["neg_lo"] > 0.5 or q["neg_hi"] < 0.5) if np.isfinite(q["neg_lo"]) else False,
                    "hit_rate": q["hit"], "median_pnl_pct": q["median"], "median_ci95_lo_pct": q["median_lo"], "median_ci95_hi_pct": q["median_hi"],
                    **yr, **chain_stats(pnl, g["date"], g["expiry"]),
                    "se_lag0_pct": q[f"se_lag{lags[0]}"], "lag_more_1": lags[2], "se_lag_more_1_pct": q[f"se_lag{lags[2]}"], "se_lag_more_1_fallback": q[f"fallback_lag{lags[2]}"],
                    "lag_more_2": lags[3], "se_lag_more_2_pct": q[f"se_lag{lags[3]}"], "se_lag_more_2_fallback": q[f"fallback_lag{lags[3]}"],
                    "mean_price_pct": 100 * float((w_ * g[PRICE_COL[price]]).sum() / w_.sum()) if len(g) else float("nan"),
                    "price_mc_bound_pct": 100 * float((w_ * g[PRICE_SE[price]]).sum() / w_.sum()) if len(g) else float("nan"),
                    "mean_copula_minus_price_pct": sh["mean"], "copula_minus_price_se_hh_pct": sh["se"] if price != "copula" else 0.0,
                    "copula_minus_price_se_fallback": sh["fallback"] if price != "copula" else False,
                    "mean_gap_payoff_pct": 100 * float((w_ * g["G"]).sum() / w_.sum()) if len(g) else float("nan"),
                    "ci95_lo_pct_seed_min": q["lo_seed_min"], "ci95_lo_pct_seed_max": q["lo_seed_max"], "ci95_hi_pct_seed_min": q["hi_seed_min"], "ci95_hi_pct_seed_max": q["hi_seed_max"],
                    "share_negative_ci95_lo_seed_min": q["neg_lo_seed_min"], "share_negative_ci95_lo_seed_max": q["neg_lo_seed_max"],
                    "share_negative_ci95_hi_seed_min": q["neg_hi_seed_min"], "share_negative_ci95_hi_seed_max": q["neg_hi_seed_max"],
                    "median_ci95_lo_pct_seed_min": q["median_lo_seed_min"], "median_ci95_lo_pct_seed_max": q["median_lo_seed_max"],
                    "median_ci95_hi_pct_seed_min": q["median_hi_seed_min"], "median_ci95_hi_pct_seed_max": q["median_hi_seed_max"],
                    "other_seeds": q["seeds"], "other_seeds_mean_interval_same_side_of_zero": q["seeds_same_mean"],
                    "other_seeds_share_interval_same_side_of_half": q["seeds_same_neg"], "other_seeds_median_interval_same_side_of_zero": q["seeds_same_median"],
                    "weighted": weighted, "n_trades_clip_flagged": clip_flagged(g),
                })  # fmt: skip
    return pd.DataFrame(rows)


def clip_flagged(g: pd.DataFrame) -> float:
    """How many trades of a sample are entered on a date flagged for the clipped mass; NaN for a
    sample of the study's trades, which has no row of the LC pass."""
    return float(g["flag_clip"].sum()) if "flag_clip" in g.columns else float("nan")


def se_method(lag: int, fallback: bool) -> str:
    """The words of the CSV column that names the error of a row."""
    if fallback:
        return f"Newey-West (Bartlett), {2 * lag} lags: the Hansen-Hodrick variance at {lag} lags is not positive"
    return f"Hansen-Hodrick (uniform kernel), {lag} lags"


def shift_rows(S: dict[str, Any]) -> pd.DataFrame:
    """The entry prices and their differences on the LC sample and its sub-samples (the same for
    the held and the hedged gap): mean across trades with its errors, and the pooled ratios."""
    rows = []
    for key, label, g, weighted in subsamples(S["priced"]):
        weights = g["year_weight"].to_numpy(float) if weighted else None
        w_ = np.ones(len(g)) if weights is None else weights
        pairs = (
            ("copula_minus_lc", "copula price minus LC price (P&L at the LC price minus P&L at the copula's)", g["P_D"] - g["ED_lc"], g["P_D_se"] + g["ED_lc_se"]),
            ("copula_minus_cc", "copula price minus CC price (P&L at the CC price minus P&L at the copula's)", g["P_D"] - g["ED_cc"], g["P_D_se"] + g["ED_cc_se"]),
            ("cc_minus_lc", "CC price minus LC price (P&L at the LC price minus P&L at the CC price: the LC model against its constant-correlation companion on the same index target)", g["ED_cc"] - g["ED_lc"], g["ED_cc_se"] + g["ED_lc_se"]),
        )  # fmt: skip
        for name, what, diff, se in pairs:
            q = describe(100 * diff, g["month_index"], LAGS_MONTHLY, BLOCK_MONTHLY, weights)
            sign = sign_rule(q["mean"], q["t"], q["lo"], q["hi"])
            yr = year_stats(100 * diff, g["date"].str[:4])
            a, b = name.split("_minus_")
            col_a, col_b = PRICE_COL[a], PRICE_COL[b]
            ratio = g[col_b] / g[col_a]
            rows.append({
                "sample": key, "sample_label": label, "n_trades": q["n"], "difference": name, "quantity": what, "mean_pct": q["mean"], "se_hh_pct": q["se"], "hh_lags": LAG_MONTHLY,
                "newey_west_fallback": q["fallback"], "se_method": se_method(LAG_MONTHLY, q["fallback"]) if np.isfinite(q["se"]) else "none: fewer than 10 trades",
                "t_ratio": q["t"], "ci95_lo_pct": q["lo"], "ci95_hi_pct": q["hi"], "se_boot_pct": q["se_boot"], "sign_by_rule": sign, **yr, **windows_stats(sign, q["t"], yr),
                "mc_bound_pct": 100 * float((w_ * se).sum() / w_.sum()) if len(g) else float("nan"),
                "share_positive": q["hit"], "median_pct": q["median"], "min_pct": 100 * float(diff.min()) if len(g) else float("nan"), "min_date": g.loc[diff.idxmin(), "date"] if len(g) else "",
                "max_pct": 100 * float(diff.max()) if len(g) else float("nan"), "max_date": g.loc[diff.idxmax(), "date"] if len(g) else "",
                "pooled_ratio": float((w_ * g[col_b]).sum() / (w_ * g[col_a]).sum()) if len(g) else float("nan"), "pooled_ratio_label": f"{PRICE_LABEL[b]} over {PRICE_LABEL[a]}, sum over sum",
                "ratio_min": float(ratio.min()) if len(g) else float("nan"), "ratio_min_date": g.loc[ratio.idxmin(), "date"] if len(g) else "",
                "ratio_max": float(ratio.max()) if len(g) else float("nan"), "ratio_max_date": g.loc[ratio.idxmax(), "date"] if len(g) else "", "weighted": weighted,
                "n_trades_clip_flagged": clip_flagged(g),
            })  # fmt: skip
    return pd.DataFrame(rows)


def combined_rows(S: dict[str, Any], gap: pd.DataFrame, valid: dict[str, bool]) -> pd.DataFrame:
    """The months without a priced row.  Per structure: the gap at the copula's price on the
    monthly trades with and without a priced row (and by reason: pending, failed, outside the
    pass's list), their difference, and the combined estimator of the gap at the LC and CC price
    on all the study's monthly trades:

        mean over ALL monthly trades of the P&L at the copula's price
        + mean over the PRICED trades of (copula price - model price),

    plain and with the year weights on the price shift.  Its error is Hansen-Hodrick on the
    combined estimator: the term of a monthly trade is ``(g_i - mean g) / N`` plus, when it is
    priced, ``w_i (s_i - mean_w s) / sum w`` (:func:`hh_sum`, 11 lags in calendar months); its
    interval resamples the monthly trades in blocks and recomputes both means.  It assumes that
    the months without a price have the mean price shift of the priced ones."""
    m, x, meta = S["monthly"], S["priced"], S["meta"]
    pos, n = m["month_index"].to_numpy(), len(m)
    priced = m["date"].isin(x["date"]).to_numpy()
    idx = block_index(n, BLOCK_MONTHLY, SEED)
    weight = m["date"].map(x.set_index("date")["year_weight"]).fillna(0.0).to_numpy(float)
    # why a monthly trade has no priced row: the first reason that applies
    reasons = (
        ("failed", "of them failed", set(meta["failed"])),
        ("non_finite", "of them with a row whose price is not finite", {r["date"] for r in meta["other_rows"]}),
        ("unreadable", "of them with a row file that could not be read (being written)", set(meta["unreadable"])),
        ("pending", "of them pending (a date of the pass, no row file yet)", set(meta["pass_pending"])),
        ("outside", "of them outside the pass's list, without a row file", set(meta["monthly_not_reachable"])),
    )  # fmt: skip
    groups: list[tuple[str, str, np.ndarray]] = [("not_priced", "the monthly trades without a priced row", ~priced)]  # fmt: skip
    left = ~priced
    for key, label, dates in reasons:
        mask = left & m["date"].isin(dates).to_numpy()
        groups.append((key, label, mask))
        left = left & ~mask
    if left.any():
        raise ValueError(f"monthly trades without a priced row and without a reason: {sorted(m.loc[left, 'date'])}")  # fmt: skip
    rows: list[dict[str, Any]] = []
    years = m["date"].str[:4].to_numpy()
    window_keys = list(windows_of([], [], "", float("nan")))

    def add(structure: str, key: str, price: str, label: str, n_gap: int, n_shift: int, value: float, se: float, fallback: bool, lo: float, hi: float, note: str = "",
            win: dict[str, Any] | None = None) -> None:  # fmt: skip
        """One row; ``win``: the tests of the windows of a mean of one sample of trades (an
        estimate that is not such a mean has none, and a named sign reads ``not tested``)."""
        t = value / se if np.isfinite(se) and se > 0 else float("nan")
        sign = sign_rule(value, t, lo, hi)
        rows.append({
            "structure": structure, "estimator": key, "price": price, "label": label, "n_trades_gap": n_gap, "n_trades_price_shift": n_shift, "value_pct": value, "se_hh_pct": se,
            "hh_lags": LAG_MONTHLY, "newey_west_fallback": fallback, "se_method": se_method(LAG_MONTHLY, fallback) if np.isfinite(se) else "none: fewer than 10 trades",
            "t_ratio": t, "ci95_lo_pct": lo, "ci95_hi_pct": hi, "boot_block": BLOCK_MONTHLY, "boot_seed": SEED, "sign_by_rule": sign, "note": note,
            **(win if win is not None else windows_of([], [], sign, float("nan"))),
        })  # fmt: skip

    def copy(structure: str, key: str, sample: str, price: str, label: str, n_shift: int) -> None:
        r = pick(gap, sample, structure, price)
        if r is not None:
            add(structure, key, price, label, int(r["n_trades"]), n_shift, r["mean_pnl_pct"], r["se_hh_pct"], bool(r["newey_west_fallback"]), r["ci95_lo_pct"], r["ci95_hi_pct"],
                win={k: r[k] for k in window_keys})  # fmt: skip

    for structure, col, _ in STRUCTURES:
        g = 100 * m[col].to_numpy(float)
        copy(structure, "copula_all_monthly", "monthly", "copula", "copula price, all the study's monthly trades", 0)  # fmt: skip
        copy(structure, "copula_priced", "priced", "copula", "copula price, the monthly trades with a priced row", 0)  # fmt: skip
        for key, label, mask in groups:
            k = int(mask.sum())
            if key != "not_priced" and k == 0:
                continue
            q = describe(g[mask], pos[mask], LAGS_MONTHLY, BLOCK_MONTHLY, seeds=key == "not_priced")
            dates = ", ".join(f"{t} ({num(v, 2, True)})" for t, v in zip(m.loc[mask, "date"], g[mask], strict=True)) if 0 < k <= 12 else ""  # fmt: skip
            redrawn = f"the interval is on the same side of zero as here on {int(q['seeds_same_mean'])} of the {SEEDS_WORDS}" if q["seeds"] else ""  # fmt: skip
            win = windows_of(g[mask], years[mask], sign_rule(q["mean"], q["t"], q["lo"], q["hi"]), q["t"])  # fmt: skip
            add(structure, f"copula_{key}", "copula", f"copula price, {label}", k, 0, q["mean"], q["se"], q["fallback"], q["lo"], q["hi"], "; ".join(w for w in (dates, redrawn) if w), win)  # fmt: skip
        if min(int(priced.sum()), int((~priced).sum())) >= MIN_N:
            a, b = priced.astype(float), (~priced).astype(float)
            u = a * (g - g[priced].mean()) / a.sum() - b * (g - g[~priced].mean()) / b.sum()
            se, fallback = hh_sum(u, pos, LAG_MONTHLY)
            with np.errstate(invalid="ignore", divide="ignore"):
                draws = (a * g)[idx].sum(axis=1) / a[idx].sum(axis=1) - (b * g)[idx].sum(axis=1) / b[idx].sum(axis=1)  # fmt: skip
            add(structure, "copula_priced_minus_not_priced", "copula", "copula price, priced minus not priced", n, 0, float(g[priced].mean() - g[~priced].mean()), se, fallback, *boot_interval(draws))  # fmt: skip
        copy(structure, "priced_year_weighted_copula", "year_weighted", "copula", "copula price, the priced trades only, each entry year weighted as in the study's monthly trades", 0)  # fmt: skip
        if not valid[structure] or len(x) < MIN_N:
            continue
        for price in ("lc", "cc"):
            s = np.where(priced, 100 * m["date"].map(x.set_index("date")["P_D"] - x.set_index("date")[PRICE_COL[price]]).to_numpy(float), 0.0)  # fmt: skip
            copy(structure, f"priced_{price}", "priced", price, f"{PRICE_LABEL[price]} price, the priced trades only (Table 1)", len(x))  # fmt: skip
            copy(structure, f"priced_year_weighted_{price}", "year_weighted", price, f"{PRICE_LABEL[price]} price, the priced trades only, each entry year weighted as in the study's monthly trades", len(x))  # fmt: skip
            for key, label, w_ in (("combined", "copula gap of all the monthly trades + mean price shift of the priced trades", priced.astype(float)),
                                   ("combined_year_weighted", "copula gap of all the monthly trades + year-weighted mean price shift of the priced trades", weight)):  # fmt: skip
                shift = float((w_ * s).sum() / w_.sum())
                u = (g - g.mean()) / n + w_ * (s - shift) / w_.sum()
                se, fallback = hh_sum(u, pos, LAG_MONTHLY)
                with np.errstate(invalid="ignore", divide="ignore"):
                    draws = g[idx].mean(axis=1) + (w_ * s)[idx].sum(axis=1) / w_[idx].sum(axis=1)
                add(structure, f"{key}_{price}", price, f"{PRICE_LABEL[price]} price, {label}", n, len(x), float(g.mean()) + shift, se, fallback, *boot_interval(draws),
                    f"mean price shift {num(shift, 4, True)} % of notional")  # fmt: skip
    out = pd.DataFrame(rows)
    # the rank of each failed date among the monthly trades, from the lowest
    out.attrs["failed_ranks"] = {
        structure: [(t, float(100 * m.loc[m["date"] == t, col].iloc[0]), int((m[col] < m.loc[m["date"] == t, col].iloc[0]).sum()) + 1) for t in meta["failed_trades"]]
        for structure, col, _ in STRUCTURES
    }  # fmt: skip
    out.attrs["not_priced"] = {key: int(mask.sum()) for key, _, mask in groups}
    return out


def payout_rows(S: dict[str, Any], valid: dict[str, bool]) -> pd.DataFrame:
    """Payout per 1 of premium of the forward, ``sum(D) / sum(price)``, with the study's
    interval (``disp_stats.bootstrap_ratio``)."""
    rows = []
    specs = [("weekly", "the study's weekly trades", S["weekly"], BLOCK_WEEKLY, ("copula",)),
             ("monthly", "the study's monthly trades", S["monthly"], BLOCK_MONTHLY, ("copula",))]  # fmt: skip
    specs += [(k, label, g, BLOCK_MONTHLY, PRICES) for k, label, g, wt in subsamples(S["priced"]) if not wt]  # fmt: skip
    for key, label, g, block, prices in specs:
        for price in prices:
            if price != "copula" and not valid["payout"]:
                continue
            nan = float("nan")
            b = boot_ratio(g["D"], g[PRICE_COL[price]], block) if len(g) >= MIN_N else {"value": float(g["D"].sum() / g[PRICE_COL[price]].sum()) if len(g) else nan, "lo": nan, "hi": nan, "se": nan}  # fmt: skip
            rows.append({
                "sample": key, "sample_label": label, "n_trades": len(g), "price": price, "payout_per_1_of_premium": b["value"], "ci95_lo": b["lo"], "ci95_hi": b["hi"],
                "se_boot": b["se"], "se_method": "standard deviation of the block-bootstrap ratios (no Hansen-Hodrick error)", "newey_west_fallback": False,
                "boot_block": block, "boot_seed": SEED, "mean_payoff_pct": 100 * g["D"].mean(), "mean_premium_pct": 100 * g[PRICE_COL[price]].mean(),
                "mc_bound": b["value"] * float(g[PRICE_SE[price]].sum() / g[PRICE_COL[price]].sum()) if len(g) else nan,
                "n_trades_clip_flagged": clip_flagged(g),
            })  # fmt: skip
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------------ the page
FALLBACK_NOTE = (
    f"{MARK} Newey-West standard error (Bartlett kernel) at twice the lag ({2 * LAG_WEEKLY} lags on weekly entries, {2 * LAG_MONTHLY} on monthly ones): on this sample the "
    "Hansen-Hodrick variance at the stated lag is not positive and the script falls back, as the study's `disp_stats.mean_se` does. The CSV files name it in a fallback column."
)


def num(x: float, digits: int = 3, sign: bool = False) -> str:
    """A number at ``digits`` decimals; one that rounds to zero is printed without a sign."""
    if x is None or not np.isfinite(x):
        return "n/a"
    if round(float(x), digits) == 0:
        return f"{0.0:.{digits}f}"
    return f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"


def pm(mean: float, se: float, digits: int = 3, fallback: bool = False) -> str:
    """``mean ± standard error``, the error marked when it is a Newey-West fallback."""
    if not np.isfinite(se):
        return num(mean, digits, True)
    return f"{num(mean, digits, True)} ± {num(se, digits)}{MARK if fallback else ''}"


def t_text(t: float) -> str:
    """A t-ratio at two decimals; at three when two decimals would print it on the other side of
    a threshold of the rule (``T_CRIT`` or ``ABOUT_ZERO``) than it is."""
    if t is None or not np.isfinite(t):
        return "n/a"
    crossed = any((round(abs(t), 2) >= c) != (abs(t) >= c) for c in (T_CRIT, ABOUT_ZERO))
    return num(t, 3 if crossed else 2, True)


def p_text(p: float) -> str:
    """``= 0.039`` or ``below 0.001``, to follow the letter p."""
    if not np.isfinite(p):
        return "n/a"
    return "below 0.001" if p < 0.0005 else f"= {p:.3f}"


def year_mean_text(v: float) -> str:
    """A year mean at two decimals; at three when it is closer to zero than ``NEAR_ZERO``."""
    return f"{v:+.3f}" if abs(v) < NEAR_ZERO else num(v, 2, True)


def interval(lo: float, hi: float, digits: int = 3, sign: bool = True) -> str:
    """``lo to hi``; a signed bound keeps its sign when it rounds to zero (the side matters)."""
    if not np.isfinite(lo):
        return "n/a"
    return f"{lo:+.{digits}f} to {hi:+.{digits}f}" if sign else f"{num(lo, digits)} to {num(hi, digits)}"  # fmt: skip


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(lines)


def with_note(text: str) -> list[str]:
    """A table, followed by the note on the fallback when one of its cells carries the mark."""
    return [text, FALLBACK_NOTE] if MARK in text else [text]


def seed_words(rows: pd.DataFrame, column: str, what: str) -> str:
    """Whether the intervals of the rows that were redrawn on other seeds answer the page's
    question (``what``) on every seed as they do at the study's seed; the rows where not."""
    redrawn = rows[rows["other_seeds"] > 0]
    if not len(redrawn):
        return ""
    count = int(redrawn["other_seeds"].iloc[0])
    moved = redrawn[redrawn[column] < redrawn["other_seeds"]]
    if not len(moved):
        return f" On each of the {count} other seeds every one of these intervals {what} as it does in the table."  # fmt: skip
    words = {s: w for s, _, w in (*STRUCTURES, *OTHER)}
    return (f" On some of the {count} other seeds an interval does not answer as in the table (it {what}): "
            + "; ".join(f"{words[r['structure']]}, {SAMPLE_LABEL.get(r['sample'], r['sample_label'])}, {PRICE_LABEL[r['price']]} price: as in the table on {int(r[column])} of the {count} seeds" for r in moved.to_dict("records")) + ".")  # fmt: skip


def month_text(month_index: int) -> str:
    return f"{(month_index - 1) // 12}-{(month_index - 1) % 12 + 1:02d}"


def dates_text(dates: list[str], limit: int = 400) -> str:
    return ", ".join(dates) if 0 < len(dates) <= limit else ("none" if not dates else f"{len(dates)} dates")  # fmt: skip


def listing(items: list[str]) -> str:
    """``a``, ``a and b``, ``a, b and c``."""
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def span(lo: Any, hi: Any) -> str:
    return f"{lo}" if lo == hi else f"{lo} to {hi}"


def pick(gap: pd.DataFrame, sample: str, structure: str, price: str) -> pd.Series | None:
    q = gap[(gap["sample"] == sample) & (gap["structure"] == structure) & (gap["price"] == price)]
    return q.iloc[0] if len(q) else None


def at_price(price: str) -> str:
    return "the copula's price" if price == "copula" else f"the {PRICE_LABEL[price]} price"


def disagree(t: float, lo: float, hi: float) -> str:
    """When the two conditions of the rule disagree, which one fails."""
    if not (np.isfinite(t) and np.isfinite(lo)):
        return ""
    by_t, by_ci = abs(t) >= T_CRIT, not (lo <= 0.0 <= hi)
    if by_ci and not by_t:
        return f" (the interval excludes zero but |t| is below {T_CRIT})"
    return f" (|t| is at least {T_CRIT} but the interval contains zero)" if by_t and not by_ci else ""  # fmt: skip


def noun_of(estimator: str) -> str:
    """What a row of Table 1e is called where its sign is not established: the mean of one
    sample of trades is a sample mean; the combined estimate (a mean over all the monthly
    trades plus a mean over the priced ones) and the difference of two means are not."""
    if estimator.startswith("combined"):
        return "estimate"
    return "difference" if estimator == "copula_priced_minus_not_priced" else "sample mean"


def sign_words(sign: str, mean: float, why: str = "", noun: str = "sample mean") -> str:
    """The words of the answer for a mean, from the sign the rule gives it (``why``: a
    bracketed note, on a disagreement of the two conditions or on the bootstrap seed; ``noun``:
    what the number is called when it is not the mean of one sample, :func:`noun_of`)."""
    if sign in ("negative", "positive"):
        return f"{sign}{why}"
    if sign == "about zero":
        return f"about zero, sign not determined{why}"
    if sign == "not established":
        side = "negative" if mean < 0 else "positive"
        if noun != "sample mean":
            return f"the {noun} is {side} and its sign is not established{why}"
        return f"the sample mean is {side} and the sign of the mean is not established{why}"
    return "no sign by the rule (the sample is too short for a standard error or for an interval)"


def sign_cell(sign: str, mean: float, why: str = "", noun: str = "sample mean") -> str:
    """The same in a table cell (``why``: the words of :func:`disagree`)."""
    if sign == "not established":
        return f"not established ({noun} {'negative' if mean < 0 else 'positive'}{why.replace(' (', '; ').rstrip(')')})"  # fmt: skip
    return {"about zero": "about zero, sign not determined", "no error": "n/a"}.get(sign, sign)


def seed_caveat(r: pd.Series, column: str, level: str, excludes: bool) -> str:
    """When the interval of a row answers differently on some of the other bootstrap seeds, the
    words that say so (``excludes``: whether it excludes ``level`` at the study's seed); empty
    when it answers the same on all of them, or was not redrawn."""
    count, same = r.get("other_seeds", 0), r.get(column, float("nan"))
    if not count or not np.isfinite(same) or same >= count:
        return ""
    here, there = ("excludes", "contains") if excludes else ("contains", "excludes")
    return f" (with the study's bootstrap seed the interval {here} {level}; on {int(count - same)} of the {int(count)} other seeds it {there} it)"  # fmt: skip


def mean_caveat(r: pd.Series) -> str:
    return seed_caveat(r, "other_seeds_mean_interval_same_side_of_zero", "zero", not (r["ci95_lo_pct"] <= 0.0 <= r["ci95_hi_pct"]))  # fmt: skip


def one_bracket(*notes: str) -> str:
    """Several bracketed notes (each `` (words)`` or empty) as one bracket."""
    words = [q.strip()[1:-1] for q in notes if q]
    return f" ({'; '.join(words)})" if words else ""


def gap_words(r: pd.Series) -> str:
    return sign_words(r["sign_by_rule"], r["mean_pnl_pct"], one_bracket(disagree(r["t_ratio"], r["ci95_lo_pct"], r["ci95_hi_pct"]), mean_caveat(r)))  # fmt: skip


def gap_cell(r: pd.Series, digits: int = 2) -> str:
    """``mean ± s.e. (t, interval)`` of a row of the gap."""
    return (f"{pm(r['mean_pnl_pct'], r['se_hh_pct'], digits, bool(r['newey_west_fallback']))} "
            f"(t = {t_text(r['t_ratio'])}, interval {interval(r['ci95_lo_pct'], r['ci95_hi_pct'], 2)})")  # fmt: skip


def share_digits(*bounds: float) -> int:
    """Whole percents, unless a bound of the interval would then print as 50 %."""
    return 1 if any(np.isfinite(b) and round(100 * b) == 50 for b in bounds) else 0


def share_cell(r: pd.Series, digits: int | None = None) -> str:
    """``share ± s.e. (interval)`` of the trades with a negative gap, in % of the trades."""
    lo, hi = r["share_negative_ci95_lo"], r["share_negative_ci95_hi"]
    d = share_digits(lo, hi) if digits is None else digits
    if not np.isfinite(r["share_negative_se_hh"]):
        return f"{100 * r['share_negative']:.{d}f} %"
    return (f"{100 * r['share_negative']:.{d}f} % ± {100 * r['share_negative_se_hh']:.{d}f} points{MARK if r['share_negative_se_fallback'] else ''} "
            f"(interval {100 * lo:.{d}f} % to {100 * hi:.{d}f} %)")  # fmt: skip


def half_words(r: pd.Series) -> str:
    if not np.isfinite(r["share_negative_ci95_lo"]):
        return "no interval"
    return (("distinguishable from one half" if r["share_differs_from_half"] else "not distinguishable from one half")
            + seed_caveat(r, "other_seeds_share_interval_same_side_of_half", "one half", bool(r["share_differs_from_half"])))  # fmt: skip


def median_words(r: pd.Series) -> str:
    """The sign of a median: named only when its block-bootstrap interval excludes zero."""
    lo, hi = r["median_ci95_lo_pct"], r["median_ci95_hi_pct"]
    if not np.isfinite(lo):
        return "no interval"
    return ("negative" if hi < 0 else ("positive" if lo > 0 else "its sign is not established")) + seed_caveat(r, "other_seeds_median_interval_same_side_of_zero", "zero", bool(hi < 0 or lo > 0))  # fmt: skip


def median_cell(r: pd.Series, digits: int = 2) -> str:
    return f"{num(r['median_pnl_pct'], digits, True)} % of notional (interval {interval(r['median_ci95_lo_pct'], r['median_ci95_hi_pct'], digits)})"  # fmt: skip


def years_cell(r: pd.Series) -> str:
    """The count of entry years with a negative mean; a year mean closer to zero than
    ``NEAR_ZERO`` is named with its value and the side it is counted on."""
    near = f" ({r['entry_years_near_zero']})" if r["entry_years_near_zero"] else ""
    return f"{int(r['entry_years_negative'])} of {int(r['entry_years'])}{near}"


def chain_p(p: float) -> str:
    """A binomial p at three decimals; at four when three would print it on the other side of
    ``CHAIN_P`` than it is."""
    return f"{p:.4f}" if (round(p, 3) >= CHAIN_P) != (p >= CHAIN_P) else f"{p:.3f}"


def chain_of(r: pd.Series) -> tuple[str, str]:
    """The lengths of the chains that have the median count and the p of the count on them:
    (``19``, ``0.167``) when they all have the same length, (``17 or of 18``, ``0.049 or 0.096``)
    when they do not (the shortest and the longest; ``to`` when more than one trade apart)."""
    lo, hi = int(r["chain_median_trades_shortest"]), int(r["chain_median_trades_longest"])
    if lo == hi:
        return str(hi), chain_p(r["chain_median_binomial_p"])
    p_lo, p_hi = chain_p(r["chain_median_binomial_p_shortest"]), chain_p(r["chain_median_binomial_p_longest"])  # fmt: skip
    return (f"{lo} or of {hi}", f"{p_lo} or {p_hi}") if hi == lo + 1 else (f"{lo} to {hi}", f"{p_lo} to {p_hi}")  # fmt: skip


def chain_cell(r: pd.Series) -> str:
    """The median chain: its count of negative trades, with the binomial p and what the rule
    says of it (:func:`chain_rule`); where the chains that have the median count differ in
    length, the count on the shortest and on the longest of them, each with its p."""
    of, p = chain_of(r)
    return f"{int(r['chain_median_negative'])} negative trades of {of} (two-sided binomial p = {p}: {r['chain_median_at_5pct']})"  # fmt: skip


def chains_closing(shown: list[tuple[str, pd.Series]]) -> str:
    """The closing sentences on the median chains of ``shown`` (the trades and the price in
    words, row), from the verdict of :func:`chain_rule` on each: where the count is
    distinguishable from one half at 5 %, where it is at the 5 % limit and where it is not
    distinguishable."""
    at = {v: [f"on {words}" for words, q in shown if q["chain_median_at_5pct"] == v] for v in (CHAIN_YES, CHAIN_LIMIT, CHAIN_NO)}  # fmt: skip
    if len(at[CHAIN_NO]) == len(shown):
        return "On none of these median chains is the count distinguishable from one half at 5 %: on non-overlapping trades neither the share nor the sign of the median is distinguishable at 5 %."  # fmt: skip
    if len(at[CHAIN_YES]) == len(shown):
        return "On each of these median chains the count is distinguishable from one half at 5 %."
    said = []
    if at[CHAIN_YES]:
        said.append(f"is distinguishable from one half at 5 % {listing(at[CHAIN_YES])}.")
    if at[CHAIN_LIMIT]:
        said.append(f"is at the 5 % limit {listing(at[CHAIN_LIMIT])}: the chains that have the median count give there a p below 5 % or not according to their number of trades, "
                    "so the verdict turns on which of them is taken as the median chain (the tie-break is under Table 1d).")  # fmt: skip
    if at[CHAIN_NO]:
        said.append(f"is not distinguishable from one half at 5 % {listing(at[CHAIN_NO])}: there, on non-overlapping trades, neither the share nor the sign of the median is distinguishable at 5 %.")  # fmt: skip
    return " ".join(("The count of the median chain " if i == 0 else "It ") + words for i, words in enumerate(said))  # fmt: skip


def chains_passing(weekly: pd.Series, priced: dict[str, pd.Series]) -> str:
    """Over all the chains of the weekly trades at the copula's price and of the priced trades
    at each price, on how many the count of negative trades has a two-sided binomial p below
    ``CHAIN_P``."""

    def of(q: pd.Series) -> str:
        return f"{int(q['chains_count_below_5pct'])} of the {int(q['chains'])} chains"

    return (f"Counting every chain and not only the median one, the count of negative trades has a two-sided binomial p below 5 % on {of(weekly)} of weekly trades at the copula's price and, on the priced trades, on "
            + listing([f"{of(q)} at {at_price(p)}" for p, q in priced.items()])
            + "; the chains of a sample share most of their trades, so these are not that many separate tests.")  # fmt: skip


def chains_lengths(rows: list[pd.Series]) -> str:
    """The sentence that says why a count is given on two numbers of trades, when it is on one
    of ``rows``; empty when the chains that have the median count all have the same length."""
    if all(q["chain_median_trades_shortest"] == q["chain_median_trades_longest"] for q in rows):
        return ""
    return (" The chains that have the median count of negative trades do not all have the same number of trades, and the p of a count depends on that number: "
            "where they differ the count is given on the shortest and on the longest of them, each with its p, and it is judged on all of them.")  # fmt: skip


def year_test(r: pd.Series) -> str:
    """The three tests of the windows of a mean, written out: the plain t of its entry-year
    means (degrees of freedom, 5 % critical value, p), the t with one lag between adjacent
    years, and the Hansen-Hodrick t read on the same degrees of freedom."""
    k, df, neg = int(r["entry_years"]), int(r["year_means_df"]), int(r["entry_years_negative"])
    side = f"{neg} of the {k} years negative" if r["year_means_mean_pct"] < 0 else f"{k - neg} of the {k} years positive"  # fmt: skip
    return (f"its {k} entry-year means give {pm(r['year_means_mean_pct'], r['year_means_se_pct'], 2)} ({side}), a plain t of {num(r['year_means_t'], 2, True)} on {df} degrees of freedom "
            f"(5 % critical value {num(r['year_means_t_crit5'], 2)}; two-sided p {p_text(r['year_means_p'])}); with one lag between adjacent years (lag-1 autocorrelation of the year means "
            f"{num(r['year_means_lag1_autocorr'], 2, True)}), t = {num(r['year_means_t_lag1'], 2, True)}{MARK if r['year_means_lag1_fallback'] else ''} (p {p_text(r['year_means_p_lag1'])}); "
            f"the Hansen-Hodrick t read on the same {df} degrees of freedom has p {p_text(r['hh_t_p_on_year_df'])}")  # fmt: skip


def tests_cell(r: pd.Series) -> str:
    """The same three tests in a table cell, with the grade of a named sign."""
    grade = (
        f". Grade of the named sign: {r['sign_once_windows_counted']}"
        if r["sign_once_windows_counted"]
        else ""
    )
    return (f"plain {num(r['year_means_t'], 2, True)} ({int(r['year_means_df'])}; {num(r['year_means_t_crit5'], 2)}; p {p_text(r['year_means_p'])}); "
            f"one lag {num(r['year_means_t_lag1'], 2, True)}{MARK if r['year_means_lag1_fallback'] else ''} (p {p_text(r['year_means_p_lag1'])}); Hansen-Hodrick t on {int(r['year_means_df'])} degrees of freedom (p {p_text(r['hh_t_p_on_year_df'])}){grade}")  # fmt: skip


def graded(r: pd.Series, detail: bool = True) -> str:
    """For a mean that the rule calls negative or positive: the sign and its grade once the
    windows are counted (:func:`windows_rule`, on the most cautious of the three tests); with
    ``detail`` the three tests are written out after it."""
    sign, grade = r["sign_by_rule"], r["sign_once_windows_counted"]
    if grade == "not tested":
        return f"{sign} by the rule of this page; it is not tested on its windows (it is not the mean of one sample of trades over at least three entry years)"  # fmt: skip
    most = f"the most cautious of the three tests of its windows (two-sided p {p_text(r['windows_p_most_cautious'])})"  # fmt: skip
    test = f": {year_test(r)}" if detail else ""
    if grade == "passes at 1 %":
        return f"{sign}, and it passes at 1 % on {most}{test}"
    if grade == "passes at 5 %, not at 1 %":
        return f"{sign} by the rule of this page; once the windows are counted it passes at 5 % and not at 1 % on {most}{test}"  # fmt: skip
    if grade == "at the 5 % limit":
        return f"{sign} by the rule of this page, and at the 5 % limit once the windows are counted, on {most}{test}"  # fmt: skip
    return f"{sign} by the rule of this page, but it does not pass at 5 % on {most}, so the sign is not established once the windows are counted{test}"  # fmt: skip


def rule_words(q: pd.Series, mean: float, detail: bool = True, noun: str = "sample mean") -> str:
    """What the rule says of a row of the combined or of the price table (``mean``: its
    value; ``noun``: what it is called when it is not the mean of one sample): graded on its
    windows when the rule names a sign."""
    if q["sign_once_windows_counted"]:
        return graded(q, detail)
    if q["sign_by_rule"] == "no error" and np.isfinite(q["t_ratio"]):
        return f"no sign by the rule (a sample of fewer than {MIN_BLOCKS * BLOCK_MONTHLY} monthly trades has a standard error and no interval)"  # fmt: skip
    return sign_words(q["sign_by_rule"], mean, disagree(q["t_ratio"], q["ci95_lo_pct"], q["ci95_hi_pct"]), noun)  # fmt: skip


def se_gap(se: float, boot: float) -> float:
    """The larger of the Hansen-Hodrick error of a mean and the standard deviation of its
    block-bootstrap means, over the smaller; NaN when one of the two is missing or zero."""
    if not (np.isfinite(se) and np.isfinite(boot)) or min(se, boot) <= 0:
        return float("nan")
    return float(max(se, boot) / min(se, boot))


def pm_boot(mean: float, se: float, boot: float, digits: int = 3, fallback: bool = False) -> str:
    """``mean ± Hansen-Hodrick error [standard deviation of the block-bootstrap means]``; the
    square bracket is left out where the sample is too short for a bootstrap."""
    return pm(mean, se, digits, fallback) + (f" [{num(boot, digits)}]" if np.isfinite(se) and np.isfinite(boot) else "")  # fmt: skip


def se_gap_words(cells: list[dict[str, Any]], where: str) -> str:
    """One paragraph, by rule, on the two errors of the means of a table (``cells``: label,
    mean, Hansen-Hodrick error ``se``, bootstrap standard deviation ``boot``, sign by the rule
    of the page).  The two "differ by more than a quarter" when the larger is more than
    ``SE_RATIO`` times the smaller.  Where the Hansen-Hodrick error is the larger, the ± and
    the t of the page are the more cautious; where it is the smaller, the cell is written out
    with the t on the bootstrap standard deviation and whether |t| stays at least ``T_CRIT``."""
    both = [{**c, "ratio": se_gap(c["se"], c["boot"])} for c in cells]
    both = [c for c in both if np.isfinite(c["ratio"])]
    if not both:
        return ""
    lead = (f"The number in square brackets is the standard deviation of the block-bootstrap means (the bootstrap of the interval), to set beside the Hansen-Hodrick error before it; "
            f"the ± and the t of the page use the Hansen-Hodrick error. Of the {len(both)} {where} that have both, ")  # fmt: skip
    far = [c for c in both if c["ratio"] > SE_RATIO]
    if not far:
        return lead + f"none has the two apart by more than a quarter (the larger is at most {max(c['ratio'] for c in both):.2f} times the smaller; the threshold is {SE_RATIO})."  # fmt: skip
    out = lead + f"{len(far)} {'has' if len(far) == 1 else 'have'} the two apart by more than a quarter (the larger more than {SE_RATIO} times the smaller)."  # fmt: skip
    larger, smaller = [c for c in far if c["se"] > c["boot"]], [c for c in far if c["se"] < c["boot"]]  # fmt: skip
    if larger:
        out += (f" On {len(larger)} the Hansen-Hodrick error is the larger, so the page's ± is the more cautious of the two: "
                + "; ".join(f"{c['label']} ({num(c['se'])} against {num(c['boot'])})" for c in larger) + ".")  # fmt: skip
    if smaller:
        said = []
        for c in smaller:
            t_hh, t_boot = c["mean"] / c["se"], c["mean"] / c["boot"]
            if c["sign"] in ("negative", "positive"):
                tail = f"|t| stays above {T_CRIT}" if abs(t_boot) >= T_CRIT else f"|t| falls below {T_CRIT}: the sign that the rule names rests on the Hansen-Hodrick error"  # fmt: skip
            else:
                tail = "the rule names no sign with either"
            said.append(f"{c['label']}, {pm(c['mean'], c['se'])} against a bootstrap standard deviation of {num(c['boot'])} (on it t = {t_text(t_boot)} in place of {t_text(t_hh)}; {tail})")  # fmt: skip
        out += f" On {len(smaller)} it is the smaller, so the page's ± may be too small there: " + "; ".join(said) + "."  # fmt: skip
    return out


def windows_words(r: pd.Series, detail: bool = False) -> str:
    """The conclusion on a mean of the gap once its windows are counted: the rule's sign,
    graded when the rule names one (:func:`graded`); the words of the rule otherwise."""
    if r["sign_once_windows_counted"]:
        return graded(r, detail) + mean_caveat(r)
    return gap_words(r)


def subs_words(gap: pd.DataFrame, n: int) -> str:
    """What the rule of the page says of the hedged gap on the sub-samples of the priced trades
    that have a standard error (Table 6), per price: on how many it names a sign, calls the mean
    about zero or leaves the sign not established (at most two rows are written out, more are
    given as a range), and how far the sub-samples are the same trades.  It does not count the
    signs of sample means: the sub-samples share their trades and are not separate evidence."""
    rows = gap[(gap["structure"] == "hedged") & ~gap["sample"].str.startswith(("weekly", "monthly")) & gap["t_ratio"].notna()]  # fmt: skip
    one = rows[rows["price"] == "copula"]
    is_half = one["sample"].str.startswith("entries_")
    halves, rest = one[is_half], one[~is_half]
    out = f"Table 6 gives the hedged gap on {len(one)} sub-samples of the {n} priced trades that have a standard error. They are not separate samples: "
    if len(rest):
        out += (f"only the {len(halves)} halves share no trade ({' and '.join(str(k) for k in halves['n_trades'])} trades), and the other {len(rest)} keep "
                f"{span(int(rest['n_trades'].min()), int(rest['n_trades'].max()))} of the same {n} trades, so the signs of their sample means are not separate evidence. ")  # fmt: skip

    def one_row(q: dict[str, Any]) -> str:
        return f"{q['sample_label']}, {pm(q['mean_pnl_pct'], q['se_hh_pct'], 2, bool(q['newey_west_fallback']))}, t {t_text(q['t_ratio'])}, {int(q['n_trades'])} trades"  # fmt: skip

    def group(q: pd.DataFrame, about_zero: bool) -> str:
        if len(q) <= 2:
            return "; ".join(one_row(k) for k in q.to_dict("records"))
        means = f"sample means {num(q['mean_pnl_pct'].min(), 2, True)} to {num(q['mean_pnl_pct'].max(), 2, True)}"
        if about_zero:
            return f"{means}, |t| at most {num(q['t_ratio'].abs().max(), 2)}"
        return f"{means}, t {t_text(q['t_ratio'].min())} to {t_text(q['t_ratio'].max())}"

    parts = []
    for p in PRICES:
        q = rows[rows["price"] == p]
        named = q[q["sign_by_rule"].isin(["negative", "positive"])]
        says = [f"{k['sign_by_rule']} on {one_row(k)} (once its windows are counted: {k['sign_once_windows_counted']}, most cautious p {p_text(k['windows_p_most_cautious'])})" for k in named.to_dict("records")]  # fmt: skip
        for key, words in (("about zero", "about zero on"), ("not established", "sign not established on"), ("no error", "no sign for want of an interval on")):  # fmt: skip
            k = q[q["sign_by_rule"] == key]
            if len(k):
                says.append(f"{words} {len(k)} ({group(k, key == 'about zero')})")
        lead = f"it names a sign on {len(named)} of the {len(q)}" if len(named) else f"it names no sign on any of the {len(q)}"  # fmt: skip
        parts.append(f"at {at_price(p)} {lead}: " + "; ".join(says))
    return out + "By the rule of this page, " + ". At ".join(k[3:] if i else k for i, k in enumerate(parts)) + ". The halves at the three prices are rows of Table 6."  # fmt: skip


def three_means(plain: float, weighted: float, whole: float, se_whole: float, n: int, n_all: int) -> str:  # fmt: skip
    """The hedged gap at the copula's price three ways, side by side: on the priced trades
    unweighted, on the same trades year-weighted, and on all the study's monthly trades; how far
    each of the first two is from the third (in % of notional and in standard errors of the
    third); and whether the year weights bring the priced trades closer to all of them.  It ends
    on the caution about a year-weighted mean, stated as a principle and not as a consequence of
    these three numbers."""
    d_plain, d_weighted = abs(plain - whole), abs(weighted - whole)
    printed = (abs(round(plain, 3) - round(whole, 3)), abs(round(weighted, 3) - round(whole, 3)))
    exact = (round(d_plain, 3), round(d_weighted, 3)) == (round(printed[0], 3), round(printed[1], 3))  # fmt: skip
    if round(d_weighted, 3) < round(d_plain, 3):
        moved = "here the year weights bring the mean of the priced trades closer to that of all the monthly trades"  # fmt: skip
    elif round(d_weighted, 3) > round(d_plain, 3):
        moved = "here the year weights take the mean of the priced trades further from that of all the monthly trades"  # fmt: skip
    else:
        moved = "here the year weights leave the mean of the priced trades as far from that of all the monthly trades"  # fmt: skip
    return (
        f"At the copula's price, the one price that all {n_all} monthly trades have, the three means side by side are {num(plain, 3, True)} on the {n} priced trades unweighted, "
        f"{num(weighted, 3, True)} on the same trades year-weighted and {num(whole, 3, True)} on all {n_all} monthly trades. The unweighted mean is {num(d_plain, 3)} from that of all {n_all}, "
        f"which is {num(d_plain / se_whole, 2)} of its standard error ({num(se_whole, 3)}), and the year-weighted one {num(d_weighted, 3)}, which is {num(d_weighted / se_whole, 2)} of it"
        f"{'' if exact else ' (distances between the unrounded means)'}: {moved}. "
        "In principle the year weights correct how many trades of each entry year are priced, not which months are priced inside a year: a year-weighted mean is to be read with the same caution "
        "as the unweighted one, and neither is evidence of a sign that the rule of this page does not name."
    )  # fmt: skip


def cover_words(by_year: pd.DataFrame) -> str:
    """The lowest and the highest priced share of the study's monthly trades over the entry
    years, each with every entry year that has it (``9 of 12 (entries of 2014 and 2017)``)."""
    share = (by_year["priced"] / by_year["monthly_trades"]).to_numpy(float)

    def at(level: float) -> str:
        q = by_year[np.isclose(share, level)]
        if np.isclose(level, 1.0):
            return f"every trade of the year ({len(q)} of the {len(by_year)} entry years: {listing(list(q['entry_year']))})"  # fmt: skip
        groups = q.groupby(["priced", "monthly_trades"], sort=True)["entry_year"].apply(list)
        return " and ".join(f"{int(a)} of {int(b)} (entries of {listing(years)})" for (a, b), years in groups.items())  # fmt: skip

    lo, hi = float(share.min()), float(share.max())
    return f"is {at(lo)}" if np.isclose(lo, hi) else f"runs from {at(lo)} to {at(hi)}"


def extrapolated_words(count: Any) -> str:
    """``a name priced beyond its last kept expiry``, or the number of names when more than one."""
    k = int(count)
    return "a name priced beyond its last kept expiry" if k == 1 else f"{k} names priced beyond their last kept expiry"  # fmt: skip


def other_marks(g: pd.DataFrame) -> tuple[list[tuple[str, np.ndarray]], np.ndarray]:
    """What a date can carry besides the flag for the clipped mass: the three other flags of the
    page and the status check.  Returned: (words, mask over the trades of ``g``) for each, and
    the mask of the trades that carry at least one."""
    marks = [
        ("a name priced beyond its last kept expiry", g["flag_extrapolated"].to_numpy(bool)),
        ("a name kept unscreened", g["flag_unscreened"].to_numpy(bool)),
        ("the DJX target repaired", g["flag_djx_repaired"].to_numpy(bool)),
        ("status check", (g["status"] == "check").to_numpy(bool)),
    ]
    some = np.logical_or.reduce([mask for _, mask in marks]) if len(g) else np.zeros(0, dtype=bool)
    return marks, some


def not_clean_words(few: pd.DataFrame) -> str:
    """The dates that are not flagged for the clipped mass are not clean dates for that reason
    alone: how many of them carry each of the other flags of the page or the status check, with
    their dates, and which carry none."""
    if not len(few):
        return ""
    marks, some = other_marks(few)
    if not some.any():
        return f" None of these {len(few)} dates carries another flag of the page (a name priced beyond its last kept expiry, a name kept unscreened, the DJX target repaired) nor the status check."  # fmt: skip
    said = [f"{words} on {int(mask.sum())} ({dates_text(list(few.loc[mask, 'date']))})" for words, mask in marks if mask.any()]  # fmt: skip
    clean = list(few.loc[~some, "date"])
    return (f" Not being flagged for the clipped mass does not make a date clean. Of these {len(few)} dates, {int(some.sum())} carry another flag of the page or the status check: {listing(said)}. "
            + (f"The other {len(clean)} carry none of these ({dates_text(clean)})." if clean else "None of them is without one."))  # fmt: skip


def gate_words(x: pd.DataFrame, meta: dict[str, Any]) -> str:
    """The priced trades whose status is ``check``, by the gating check of the row that fails
    (``scripts/lcm_price.py``): no non-finite number, the basket's forward within three standard
    errors, the index gate.  A failed forward or index check bears on the prices of the date and
    is said so; the non-finite numbers of the rows that fail the first are named."""
    check = x[x["status"] == "check"]
    if not len(check):
        return ""
    out = f"{len(check)} of the {len(x)} trades have status check. "
    bad = x[~x["check_no_nan"].astype(bool)]
    if len(bad):
        fields = sorted({c for v in meta["non_finite"].values() for c in v})
        out += (f"On {len(bad)} the row fails `check_no_nan`: the non-finite numbers of these rows are {', '.join(f'`{c}`' for c in fields) or 'none'}, none of them a price used here "
                "(`ED_lc`, `ED_lc_se`, `ED_cc`, `ED_cc_se` are finite on every priced trade, checked). ")  # fmt: skip
    bad = x[~x["check_forward"].astype(bool)]
    if len(bad):
        out += (f"On {len(bad)} the row fails `check_forward` (the mean terminal level of the basket over its forward, less one, is more than three standard errors from zero): "
                + "; ".join(f"{r['date']}, {100 * r['forward_error']:+.3f} % ± {100 * r['forward_error_se']:.3f} %" for r in bad.to_dict("records"))
                + ". This check bears on the prices of the date: its LC and CC prices are used here with this flag. ")  # fmt: skip
    bad = x[~x["check_index"].astype(bool)]
    if len(bad):
        out += (f"On {len(bad)} the row fails `check_index`, the index gate at the horizon ({dates_text(list(bad['date']), 12)}): "
                "the LC and CC prices of these dates are used here with this flag. ")  # fmt: skip
    return out


def pass_state(meta: dict[str, Any]) -> tuple[str, str]:
    """``complete``, ``running`` or ``stopped``, with its sentence.  Complete: every date of the
    pass's list has a readable row file.  Running: not complete, and the newest row file is at
    most ``STALE_MINUTES`` old.  Stopped: not complete, and no row file since."""
    total, have, folder = len(meta["pass_dates"]), len(meta["pass_with_row"]), meta["folder"]
    failed = sorted(set(meta["failed"]) & set(meta["pass_dates"]))
    if have == total:
        return "complete", f"the pass is complete ({have} of {total} dates have a row" + (f", of which {len(failed)} failed)" if failed else ")")  # fmt: skip
    missing = f"{have} of its {total} dates have a row, {total - have} do not"
    if folder["newest_age_minutes"] <= STALE_MINUTES:
        return "running", f"the pass was still running ({missing}; the newest row file is less than {STALE_MINUTES} minutes old)"  # fmt: skip
    return "stopped", f"the pass is not complete and has stopped or is stalled ({missing}; no row file was written in the {STALE_MINUTES} minutes before this page)"  # fmt: skip


def answer(S: dict[str, Any], gap: pd.DataFrame, shifts: pd.DataFrame, comb: pd.DataFrame, other: pd.DataFrame, valid: dict[str, bool]) -> list[str]:  # fmt: skip
    """The answer to the owner's question: paragraphs whose every sentence is generated from the
    rows by a rule (the sign of a mean: :func:`sign_rule`; the share against one half: its
    interval; the half-samples: :func:`windows_rule`; the state of the pass: :func:`pass_state`)."""
    meta, x, m = S["meta"], S["priced"], S["monthly"]
    n = len(x)
    state_key, state = pass_state(meta)
    wk, mo = pick(gap, "weekly", "hedged", "copula"), pick(gap, "monthly", "hedged", "copula")
    models = valid["hedged"] and n >= MIN_N
    r = {p: pick(gap, "priced", "hedged", p) for p in PRICES} if models else {}
    heads = [("at the copula's price on the study's weekly trades", wk), ("at the copula's price on the study's monthly trades", mo)]  # fmt: skip
    heads += [(f"at {at_price(p)} on the {n} priced trades", r[p]) for p in r]
    named = [(k, h["sign_by_rule"], h["sign_once_windows_counted"]) for k, h in heads if h["sign_by_rule"] in ("negative", "positive")]  # fmt: skip
    where = "at any of the three prices (the copula's, LC and CC)" if models else "at the copula's price, the only one with numbers here"  # fmt: skip
    if not named:
        lead = f"Neither sign is established: delta-hedged, the mean of the gap of the one-year trades is not distinguishable from zero {where}."  # fmt: skip
    elif len(named) == len(heads) and len({s for _, s, _ in named}) == 1 and all(g.startswith("passes") for _, _, g in named):  # fmt: skip
        lead = f"{named[0][1].capitalize()}: delta-hedged, the mean of the gap of the one-year trades is {named[0][1]} on every sample and at every price below, and passes at 5 % once its windows are counted."  # fmt: skip
    else:
        lead = ("Delta-hedged, the rule of this page names a sign of the mean of the gap of the one-year trades only " + "; ".join(f"{k} ({s}; once its windows are counted: {g})" for k, s, g in named)
                + f", and on {len(heads) - len(named)} of the {len(heads)} samples and prices below it names none.")  # fmt: skip
    first = (
        f"**Answer.** {lead} The trades are the study's one-year trades on basket {BASKET} (the 30 Dow names). The Palladium forward pays at expiry the dispersion of the basket, which is the weighted sum over the names of "
        "the absolute difference between the name's performance and the basket's over the year, against a price paid at entry. "
        "The gap is the Palladium forward minus the vega-neutral package (the single-name straddles minus the basket straddle, one unit each), each leg delta-hedged daily with the study's hedge; "
        f"the numbers are the mean profit and loss (P&L) per trade, in % of notional. Each ± is a Hansen-Hodrick standard error of the mean ({LAG_WEEKLY} lags on weekly entries, {LAG_MONTHLY} on monthly ones, "
        f"because the one-year windows overlap), t is the mean over it, and each interval is the 95 % block bootstrap of the mean. "
        f"At the copula's price (the study's own price of the forward) the mean is {gap_cell(wk)} on the study's {wk['n_trades']} weekly trades (entries {wk['first_entry']} to {wk['last_entry']}): {windows_words(wk)}; "
        f"it is {gap_cell(mo)} on its {mo['n_trades']} monthly trades (one per calendar month, entries {mo['first_entry']} to {mo['last_entry']}): {windows_words(mo)}. "
    )
    rule = (f"The rule of this page: a mean is called negative or positive only when its 95 % interval excludes zero and |t| is at least {T_CRIT}, and a mean so called is then graded on three tests that count its one-year windows (Table 1c); "
            f"it is called about zero when it is less than {ABOUT_ZERO} standard errors from zero; otherwise the sample mean and its error are given and the sign of the mean is not established.")  # fmt: skip
    ok, plain = x[x["status"] == "ok"], x[~x["flag_clip"]]
    marked = int(other_marks(plain)[1].sum())
    limits = (
        f"**Limits.** The LC and CC prices are at the development budget ({pc.BUDGETS['development']}). {int(x['flag_clip'].sum())} of the {n} priced trades are entered on dates flagged because the model does not "
        f"reach its index target (clipped mass above {100 * CLIP_FLAG:.0f} % inside ±2.5 standard deviations); the {len(plain)} not flagged: {dates_text(list(plain['date']), 12)}"
        + (f" ({marked} of them carry another flag of the page or the status check: the paragraph under Table 7)" if marked else "")
        + ". "
        f"Status ok does not mean that the index target is reached: {int(ok['flag_clip'].sum())} of the {len(ok)} trades with status ok are flagged (the index gate of the status is waived where the wing binds: "
        "`outputs/dispersion_lc/pm_update/1y/NUMBERS_1Y.md`, \"Read this first\"), so the row 'status ok only' of Table 6 is not a clean sub-sample. "
        + gate_words(x, meta)
        + f"A name is priced beyond its last kept expiry on {int(x['flag_extrapolated'].sum())} of the {n} trades and the target of the Dow index options (DJX) is repaired on {int(x['flag_djx_repaired'].sum())}. "
        f"At {meta['folder']['run_time'][-8:-3]} {state}"
    )  # fmt: skip
    limits += f"; the priced share of the study's monthly trades {cover_words(meta['by_year'])}."
    if not models:
        why = "the study's printed hedged numbers are not reproduced (Table 3)" if not valid["hedged"] else f"only {n} monthly trades have a priced LC row"  # fmt: skip
        return [f"{first}**No number is given at the price of the local-correlation model (LC) or of its constant-correlation companion (CC): {why}.** {rule}", limits]  # fmt: skip
    first += (
        f"On the {n} monthly trades that have a price from the local-correlation model (LC) and from its constant-correlation companion (CC){'' if state_key == 'complete' else ' so far'} "
        f"(entries {x['date'].min()} to {x['date'].max()}) the mean is {gap_cell(r['copula'])} at the copula's price: {windows_words(r['copula'])}; "
        f"{gap_cell(r['lc'])} at the LC price: {windows_words(r['lc'])}; {gap_cell(r['cc'])} at the CC price: {windows_words(r['cc'])}. {rule}"
    )

    # ---- how often
    shown = [wk, *(r[p] for p in PRICES)]
    counts_ = sorted({(int(q["entry_years_negative"]), int(q["entry_years"]), float(q["entry_years_binomial_p"])) for q in shown}, reverse=True)  # fmt: skip
    chained = [("the weekly trades at the copula's price", wk), *((f"the priced trades at {at_price(p)}", r[p]) for p in PRICES)]  # fmt: skip
    chains_say = f"{chains_passing(wk, r)} {chains_closing(chained)}"
    often = (
        "**How often the gap is negative.** A share says how often the hedged gap is negative, not by how much. A share and a median are judged on their 95 % block-bootstrap interval alone "
        "(a share is called distinguishable from one half when its interval excludes 50 %, a median negative or positive when its interval excludes zero): a lighter rule than the one for a mean, "
        f"which also asks for |t| at least {T_CRIT} and is then graded on its windows. A median below zero and a share of negative trades above one half are the same statement. The share of trades with a negative hedged gap "
        "(± a Hansen-Hodrick standard error on the indicator, with the lags of the mean; 95 % block-bootstrap interval) is "
        f"{share_cell(wk)} on the {wk['n_trades']} weekly trades at the copula's price: {half_words(wk)}. On the {n} priced trades it is "
        + "; ".join(f"{share_cell(r[p])} at {at_price(p)}: {half_words(r[p])}" for p in PRICES)
        + f". The median hedged gap is {median_cell(wk)} on the weekly trades at the copula's price: {median_words(wk)}; on the priced trades it is "
        + "; ".join(f"{median_cell(r[p])} at {at_price(p)}: {median_words(r[p])}" for p in PRICES)
        + f". The mean of an entry year is negative in {years_cell(wk)} years on the weekly trades at the copula's price"
        + (f" (positive: {wk['entry_years_positive']})" if wk["entry_years_positive"] else "")
        + "; on the priced trades in " + ", ".join(f"{years_cell(r[p])} at {at_price(p)}" for p in PRICES)
        + f". A year mean within {NEAR_ZERO} % of notional of zero is named with its value. A count of years is not a test: if the years were independent, which they are not, the two-sided binomial p of these counts would be "
        + ", ".join(f"{p_:.3f} for {a} of {b}" for a, b, p_ in counts_)
        + f". These shares, medians and counts are on overlapping trades. Chains of non-overlapping one-year trades count each window once.{chains_lengths([q for _, q in chained])} "
        f"Of the {int(wk['chains'])} chains of {span(int(wk['chain_trades_min']), int(wk['chain_trades_max']))} "
        f"weekly trades (one chain from each trade of the first year), the median chain has at the copula's price {chain_cell(wk)}; on the priced trades the median chain has "
        + "; ".join(f"at {at_price(p)} {chain_cell(r[p])}" for p in PRICES) + f". {chains_say}"
    )  # fmt: skip

    # ---- the months without an LC price
    c = comb[comb["structure"] == "hedged"].set_index("estimator")
    counts = comb.attrs["not_priced"]
    months = ""
    if counts["not_priced"]:
        parts = [f"{counts[k]} {words}" for k, words in (("pending", "pending"), ("failed", "failed"), ("outside", "outside the pass's list"), ("unreadable", "with a row file being written"),
                                                         ("non_finite", "with a non-finite price")) if counts[k]]  # fmt: skip
        u, pr_ = c.loc["copula_not_priced"], c.loc["copula_priced"]

        def est(q: pd.Series, unit: str = "") -> str:
            """``mean ± s.e. (t, interval): what the rule says`` of a row of Table 1e."""
            inside = [f"t = {t_text(q['t_ratio'])}"] if np.isfinite(q["t_ratio"]) else []
            inside += [f"interval {interval(q['ci95_lo_pct'], q['ci95_hi_pct'], 2)}"] if np.isfinite(q["ci95_lo_pct"]) else []  # fmt: skip
            return (f"{pm(q['value_pct'], q['se_hh_pct'], 2, bool(q['newey_west_fallback']))}{unit}" + (f" ({', '.join(inside)})" if inside else "")
                    + f": {rule_words(q, q['value_pct'], detail=False, noun=noun_of(str(q.name)))}"
                    + (" (the three tests are written out under Table 1e)" if q["sign_once_windows_counted"] not in ("", "not tested") else ""))  # fmt: skip

        months = (
            f"**The months without an LC price.** {counts['not_priced']} of the study's {len(m)} monthly trades have no priced row ({', '.join(parts)}). Which months are priced is not drawn at random: it follows "
            + listing([words for k, words in (("pending", "the order of the pass"), ("failed", "the failures of the pass"), ("outside", "the dates outside the pass's list")) if counts[k]])
            + f". At the copula's price the hedged gap of the {int(u['n_trades_gap'])} trades without a priced row averages {est(u, ' % of notional')}. "
            f"The {n} priced ones average {pm(pr_['value_pct'], pr_['se_hh_pct'], 2, bool(pr_['newey_west_fallback']))}"
        )  # fmt: skip
        if "copula_priced_minus_not_priced" in c.index:
            months += f". Priced minus not priced is {est(c.loc['copula_priced_minus_not_priced'])}"
        months += "."
        ranks = comb.attrs["failed_ranks"]["hedged"]
        if ranks:
            months += (f" The hedged gap at the copula's price of the {len(ranks)} failed date{'s' if len(ranks) > 1 else ''} is "
                       + ", ".join(f"{num(v, 2, True)} ({t})" for t, v, _ in ranks) + f": from the lowest of the {len(m)} monthly trades, rank "
                       + ", ".join(str(k) for _, _, k in ranks) + f" (the median trade, rank {(len(m) + 1) // 2}, is at {num(mo['median_pnl_pct'], 2, True)}).")  # fmt: skip
        both = {p: c.loc[f"combined_{p}"] for p in ("lc", "cc")}
        yw = {p: c.loc[f"combined_year_weighted_{p}"] for p in ("lc", "cc")}
        py = {p: c.loc[f"priced_year_weighted_{p}"] for p in PRICES}
        months += (
            f" An estimate for all {len(m)} monthly trades adds the mean price shift of the {n} priced trades to the copula gap of all {len(m)} (Hansen-Hodrick error on the combined estimator, {LAG_MONTHLY} lags): "
            f"at the LC price {est(both['lc'])}; at the CC price {est(both['cc'])}. It assumes that the months without a price have the mean price shift of the priced ones. "
            f"With each entry year weighted as in the study's monthly trades, the means on the priced trades are "
            + ", ".join(f"{pm(py[p]['value_pct'], py[p]['se_hh_pct'], 2, bool(py[p]['newey_west_fallback']))} ({PRICE_LABEL[p]})" for p in PRICES)
            + f", and the combined estimate with the year-weighted shift is {pm(yw['lc']['value_pct'], yw['lc']['se_hh_pct'], 2, bool(yw['lc']['newey_west_fallback']))} (LC) and "
            f"{pm(yw['cc']['value_pct'], yw['cc']['se_hh_pct'], 2, bool(yw['cc']['newey_west_fallback']))} (CC). "
            + three_means(pr_["value_pct"], py["copula"]["value_pct"], mo["mean_pnl_pct"], mo["se_hh_pct"], n, len(m))
        )  # fmt: skip

    # ---- what changes between the prices
    d = shifts[shifts["sample"] == "priced"].set_index("difference")

    def diff(key: str) -> str:
        q = d.loc[key]
        return (f"{pm(q['mean_pct'], q['se_hh_pct'], 2, bool(q['newey_west_fallback']))} % of notional (t = {t_text(q['t_ratio'])}, interval {interval(q['ci95_lo_pct'], q['ci95_hi_pct'], 2)}): "
                f"{rule_words(q, q['mean_pct'], detail=False)}")  # fmt: skip

    prices = (
        "**What changes between the prices.** Only the entry price of the forward changes: P&L at the LC or CC price = P&L at the copula's price + (copula price - model price), per trade. "
        "The delta hedge of the forward and the package are the study's: the forward is not re-hedged with LC or CC deltas. "
        f"On the {n} priced trades the copula price minus the LC price is {diff('copula_minus_lc')}. It is the sum of two parts. The first is copula minus CC, "
        f"which holds the gap between the model's index target and the study's index level and the constant-correlation fit: {diff('copula_minus_cc')}. The second is CC minus LC, "
        f"which is the LC model against its constant-correlation companion on the same index target, a target that the LC model does not reach on {int(x['flag_clip'].sum())} of the {n} dates (the flag for the clipped mass): {diff('cc_minus_lc')}. "
        f"Each ± of this paragraph is the sampling error across the {n} entry dates (Hansen-Hodrick, {LAG_MONTHLY} lags in calendar months). The Monte Carlo error of the prices is not in it: its bound on the mean difference is "
        + ", ".join(f"{num(d.loc[k, 'mc_bound_pct'], 3)} ({w_})" for k, w_ in (("copula_minus_lc", "copula minus LC"), ("copula_minus_cc", "copula minus CC"), ("cc_minus_lc", "CC minus LC")))
        + " % of notional, and the dates are priced on the same seeds, so these errors may not average out. Nor is the budget in it (development)."
        + (" The three tests of the windows of each difference that the rule names are written out under Table 5." if d["sign_once_windows_counted"].astype(bool).any() else "")
    )  # fmt: skip

    # ---- by half, and the sub-samples
    halves = []
    for key, years in (("weekly_2007_2016", "2007-2016"), ("weekly_2017_2025", "2017-2025")):
        h = pick(gap, key, "hedged", "copula")
        halves.append(f"for the {h['n_trades']} entries of {years}, {gap_cell(h)}; this half holds {span(int(h['chain_trades_min']), int(h['chain_trades_max']))} non-overlapping one-year windows and "
                      f"{year_test(h)}: {windows_words(h)}")  # fmt: skip
    by_half = f"**By half.** At the copula's price the study's weekly trades give, {halves[0]}. They give, {halves[1]}. {subs_words(gap, n)}"  # fmt: skip
    rho = {k: pick(other, sample, "hedged_rho", price) for k, sample, price in (("weekly", "weekly", "copula"), *((p, "priced", p) for p in PRICES))}  # fmt: skip
    second = (
        "**The study's other gap, for completeness.** The gap of the report, and of everything above, is the vega-neutral one. The study also computes a gap against the correlation-neutral package "
        f"(the basket straddle scaled so that the package has no sensitivity to correlation; Table 9). Delta-hedged, at the copula's price, its mean is {gap_cell(rho['weekly'])} on the {rho['weekly']['n_trades']} weekly trades: "
        f"{windows_words(rho['weekly'], detail=True)}"
        + (f". On the {n} priced trades, with the same price shift, it is " + "; ".join(f"{gap_cell(rho[p])} at {at_price(p)}: {windows_words(rho[p])}" for p in PRICES if rho[p] is not None) + "." if rho["lc"] is not None else ".")
    )  # fmt: skip
    out = [first, often, *([months] if months else []), prices, by_half, second, limits]
    if any(MARK in t for t in out):
        out.append(FALLBACK_NOTE)
    return out


def build(rows_dir: Path) -> tuple[str, list[dict[str, Any]], dict[str, pd.DataFrame], list[str]]:
    S = samples(rows_dir)
    if not len(S["priced"]):
        raise SystemExit(f"no monthly trade of the study has a priced row in {rows_dir}")
    checks = self_checks(S)
    val = validation(S["weekly"])
    valid = {g: bool(val.loc[val["rests_on"].str.contains(g), "match"].all()) for g in ("held", "hedged", "payout", "other gap")}  # fmt: skip
    valid["hedged_rho"] = valid["other gap"] and valid["hedged"]
    gap, shifts, payout = gap_rows(S, valid), shift_rows(S), payout_rows(S, valid)
    comb = combined_rows(S, gap, valid)
    other = gap_rows(S, valid, OTHER, headline=True)
    meta, x, w, m = S["meta"], S["priced"], S["weekly"], S["monthly"]
    commit = ", ".join(meta["commits"]) or "n/a"
    budget = pc.BUDGETS["development"]
    rel_rows = str(rows_dir).replace(str(pc.ROOT) + "/", "")
    n = len(x)
    folder = meta["folder"]
    _, state = pass_state(meta)
    md: list[str] = []
    md.append(
        "# The gap of the one-year trades, held and delta-hedged, at the copula's, the LC and the CC price"
    )
    md.append(
        f"Written {folder['run_time']} (machine time) by `scripts/pm_later_1y_gap.py`. State of the row folder `{rel_rows}` when it was listed at that time (a row file written after the listing is not read): {folder['files']} row files, the newest "
        f"(`{folder['newest_file']}`) written at {folder['newest_time'][-8:]}; {n} monthly trades priced, {len(meta['pending_trades'])} pending, "
        f"{len(meta['failed'])} dates failed ({dates_text(meta['failed'])}); {state}. LC is the local-correlation model and CC its constant-correlation companion: {budget}; commit {commit}; "
        f"configuration {', '.join(meta['digests']) or 'n/a'}. The rows of {meta['carry_decisions']} of the {n} priced trades carry {DECISIONS}: this is read from each row "
        f"(`calendar_repair` true and the columns {' and '.join(f'`{c}`' for c in DECISION_COLUMNS)} present, the rule of `scripts/pm_1y.py`); what the decisions change is in `outputs/dispersion_lc/pm_update/1y/NUMBERS_1Y.md`, section 1. "
        "Every number below, and every sentence of the answer, is recomputed from the row files at each run."
    )
    answer_text = answer(S, gap, shifts, comb, other, valid)
    md += answer_text

    # ---- table 1: the headline
    md.append("## 1. The gap at the three prices")
    head = ["structure", "sample", "price", "trades", "mean P&L ± s.e.", "t", "95 % interval", "sign of the mean by the rule of this page"]  # fmt: skip
    body = []
    for structure, _, words in STRUCTURES:
        for sample, price in HEADLINE:
            r = pick(gap, sample, structure, price)
            if r is None:
                body.append([words, SAMPLE_LABEL["priced"], PRICE_LABEL[price], str(n), "not given: the study's printed numbers are not reproduced (Table 3)", "", "", ""])  # fmt: skip
                continue
            body.append([words, SAMPLE_LABEL[sample], PRICE_LABEL[price], str(r["n_trades"]), pm(r["mean_pnl_pct"], r["se_hh_pct"], 3, bool(r["newey_west_fallback"])),
                         t_text(r["t_ratio"]), interval(r["ci95_lo_pct"], r["ci95_hi_pct"]), sign_cell(r["sign_by_rule"], r["mean_pnl_pct"], disagree(r["t_ratio"], r["ci95_lo_pct"], r["ci95_hi_pct"]))
                         + (f"; once its windows are counted: {r['sign_once_windows_counted']}" if r["sign_once_windows_counted"] else "")])  # fmt: skip
    md.append(f"**Table 1.** Mean P&L per trade of the gap (Palladium forward minus the vega-neutral package), % of notional; s.e. is the standard error; the three rows of the {n} trades are on the same trades.")  # fmt: skip
    md += with_note(table(head, body))
    md.append(
        f"± is the Hansen-Hodrick standard error of the mean: {LAG_WEEKLY} lags on the weekly trades, {LAG_MONTHLY} lags in calendar months of entry on the monthly ones; t is the mean over it. "
        f"The interval is the 95 % circular block bootstrap of the mean ({N_RESAMPLES} resamples, seed {SEED}; blocks of {BLOCK_WEEKLY} weekly or {BLOCK_MONTHLY} monthly entries). "
        f"The last column is the rule of the answer: negative or positive only when the interval excludes zero and |t| is at least {T_CRIT} (a mean so called is then graded on its windows, Table 1c); about zero when |t| is below {ABOUT_ZERO}; not established otherwise. "
        f"A t that two decimals would print on the other side of {T_CRIT} or of {ABOUT_ZERO} is printed with three. "
        f"Weekly trades: entries {w['date'].min()} to {w['date'].max()}; monthly trades: {m['date'].min()} to {m['date'].max()}; trades with a priced LC row: {x['date'].min() if n else 'n/a'} to {x['date'].max() if n else 'n/a'}. "
        f"Copula rows: the study's own table (source {SRC_STUDY}). LC and CC rows: {SRC_ROWS}, {budget}, commit {commit}."
    )
    pr = gap[gap["other_seeds"] > 0]

    def moves(lo: str, hi: str) -> pd.Series:
        """The largest move of a bound over the other seeds, per structure."""
        return pr.assign(move=np.maximum(pr[f"{lo}_seed_max"] - pr[f"{lo}_seed_min"], pr[f"{hi}_seed_max"] - pr[f"{hi}_seed_min"])).groupby("structure")["move"].max()  # fmt: skip

    spread = moves("ci95_lo_pct", "ci95_hi_pct")
    md.append(
        f"Over {SEEDS_WORDS} a bound of an interval of Table 1 moves by at most "
        + " and ".join(
            f"{num(spread[s], 2)} ({words})" for s, _, words in STRUCTURES if s in spread.index
        )
        + ": the second decimal of a bound is not determined (the range of each bound is in `tables/gap_12m_summary.csv`)."
        + seed_words(
            pr, "other_seeds_mean_interval_same_side_of_zero", "contains zero or excludes it"
        )
    )

    # ---- table 1b: how often
    body = []
    for structure, _, words in STRUCTURES:
        for sample, price in HEADLINE:
            r = pick(gap, sample, structure, price)
            if r is None:
                continue
            body.append([words, SAMPLE_LABEL[sample], PRICE_LABEL[price], str(r["n_trades"]),
                         f"{100 * r['share_negative']:.1f} % ± {100 * r['share_negative_se_hh']:.1f}{MARK if r['share_negative_se_fallback'] else ''}",
                         f"{100 * r['share_negative_ci95_lo']:.1f} % to {100 * r['share_negative_ci95_hi']:.1f} %", num(r["share_negative_z_against_half"], 2, True), half_words(r),
                         num(r["median_pnl_pct"], 3, True), interval(r["median_ci95_lo_pct"], r["median_ci95_hi_pct"]), median_words(r)])  # fmt: skip
    md.append("**Table 1b.** How often the gap is negative, and its median: the share of trades with a negative P&L (% of the trades, ± standard error in points) and the median P&L per trade (% of notional).")  # fmt: skip
    md += with_note(table(["structure", "sample", "price", "trades", "share of trades with a negative gap ± s.e.", "95 % interval of the share", "z = (share - 50 %) / s.e.", "the share against one half",
                           "median P&L", "95 % interval of the median", "sign of the median"], body))  # fmt: skip
    md.append(
        f'± of a share is the Hansen-Hodrick standard error of the mean of the indicator "the P&L of the trade is negative" ({LAG_WEEKLY} lags weekly, {LAG_MONTHLY} monthly, as for the mean). '
        f"The intervals are the same block bootstrap as in Table 1 (seed {SEED}), of the share and of the median. A share is called distinguishable from one half only when its interval excludes 50 %; "
        "the sign of a median is named only when its interval excludes zero. That is the interval alone, a lighter rule than the one for a mean: the column z (the share less 50 %, over its standard error) is printed for comparison and is not part of the rule. "
        "A median below zero and a share above one half are the same statement; on chains of non-overlapping trades (Table 1d) the count of negative trades is the test of both. "
        "A share is a count of trades: it says nothing of the size of the gains against the size of the losses, which is what the mean of Table 1 measures."
    )
    share_move, median_move = moves("share_negative_ci95_lo", "share_negative_ci95_hi"), moves("median_ci95_lo_pct", "median_ci95_hi_pct")  # fmt: skip
    md.append(
        f"Over {SEEDS_WORDS} a bound of the interval of a share moves by at most "
        + " and ".join(f"{100 * share_move[s]:.1f} points ({words})" for s, _, words in STRUCTURES if s in share_move.index)
        + ", and a bound of the interval of a median by at most "
        + " and ".join(f"{num(median_move[s], 2)} % of notional ({words})" for s, _, words in STRUCTURES if s in median_move.index)
        + "." + seed_words(pr, "other_seeds_share_interval_same_side_of_half", "contains one half or excludes it")
        + seed_words(pr, "other_seeds_median_interval_same_side_of_zero", "contains zero or excludes it").replace("every one of these intervals", "every interval of a median")
    )  # fmt: skip

    # ---- table 1c: the entry years
    cols = [(sample, price, pick(gap, sample, "hedged", price)) for sample, price in HEADLINE]
    cols = [(sample, price, r) for sample, price, r in cols if r is not None]
    frames = {"weekly": w, "monthly": m, "priced": x}
    year_rows = []
    for structure, col, _ in STRUCTURES:
        for sample, price in HEADLINE:
            if pick(gap, sample, structure, price) is None:
                continue
            g = frames[sample]
            pnl = 100 * (g[col] + g["P_D"] - g[PRICE_COL[price]]) if price != "copula" else 100 * g[col]  # fmt: skip
            for year, v in pnl.groupby(g["date"].str[:4]):
                year_rows.append({"structure": structure, "sample": sample, "price": price, "entry_year": year, "n_trades": len(v), "mean_pnl_pct": float(v.mean())})  # fmt: skip
    entry_years = pd.DataFrame(year_rows)
    ey = entry_years[entry_years["structure"] == "hedged"].set_index(["sample", "price", "entry_year"])  # fmt: skip
    body = []
    for year in sorted(entry_years["entry_year"].unique()):
        line = [year]
        for sample, price, _ in cols:
            key = (sample, price, year)
            line.append(f"{year_mean_text(ey.loc[key, 'mean_pnl_pct'])} ({int(ey.loc[key, 'n_trades'])})" if key in ey.index else "no trade")  # fmt: skip
        body.append(line)
    body.append(["**years with a negative mean**", *(f"{int(r['entry_years_negative'])} of {int(r['entry_years'])}" for _, _, r in cols)])  # fmt: skip
    body.append(["**two-sided binomial p of that count if the years were independent (they are not)**", *(f"{r['entry_years_binomial_p']:.3f}" for _, _, r in cols)])  # fmt: skip
    body.append(["**mean of the year means ± s.e.**", *(pm(r["year_means_mean_pct"], r["year_means_se_pct"], 2) for _, _, r in cols)])  # fmt: skip
    body.append(["**plain t of the year means (degrees of freedom; 5 % critical value; two-sided p)**", *(f"{num(r['year_means_t'], 2, True)} ({int(r['year_means_df'])}; {num(r['year_means_t_crit5'], 2)}; p {p_text(r['year_means_p'])})" for _, _, r in cols)])  # fmt: skip
    body.append(["**lag-1 autocorrelation of the year means**", *(num(r["year_means_lag1_autocorr"], 2, True) for _, _, r in cols)])  # fmt: skip
    body.append(["**t of the year means with one lag between adjacent years (two-sided p, same degrees of freedom)**", *(f"{num(r['year_means_t_lag1'], 2, True)}{MARK if r['year_means_lag1_fallback'] else ''} (p {p_text(r['year_means_p_lag1'])})" for _, _, r in cols)])  # fmt: skip
    body.append(["**Hansen-Hodrick t of the mean (Table 1), read on the same degrees of freedom (two-sided p)**", *(f"{t_text(r['t_ratio'])} (p {p_text(r['hh_t_p_on_year_df'])})" for _, _, r in cols)])  # fmt: skip
    md.append("**Table 1c.** The delta-hedged gap by entry year: mean P&L per trade, % of notional, with the number of trades of the year in brackets; and the three tests that count the one-year windows.")  # fmt: skip
    md += with_note(table(["entry year", *(f"{'weekly' if s == 'weekly' else ('monthly' if s == 'monthly' else 'priced')} trades, {PRICE_LABEL[p]} price" for s, p, _ in cols)], body))  # fmt: skip
    both_t = pd.concat([gap[gap["structure"] == "hedged"], other])
    both_t = both_t[both_t["t_ratio"].notna() & both_t["year_means_t"].notna()]
    md.append(
        f"A year mean closer to zero than {NEAR_ZERO} % of notional is printed with three decimals: the count of negative years turns on its sign. "
        "The standard error of the mean of the year means is their standard deviation over the square root of their number, and the plain t is that mean over it, with as many degrees of freedom as years less one. "
        f"On this page the plain t is the laxer test, not the stricter one: a one-year window entered in a year runs into the next (on average {100 * next_year_share(w['date'], w['expiry']):.0f} % of the window of a weekly trade "
        "lies in the calendar year after its entry year), so the means of adjacent entry years share a part of their exposure: their lag-1 autocorrelation is positive in "
        f"{sum(int(r['year_means_lag1_autocorr'] > 0) for _, _, r in cols)} of the {len(cols)} columns of the table. A plain t takes the year means as independent and, where they are positively correlated, overstates |t|. "
        f"Its |t| is larger than the Hansen-Hodrick |t| of the same mean on {int((both_t['year_means_t'].abs() > both_t['t_ratio'].abs()).sum())} of the {len(both_t)} delta-hedged means of "
        "`tables/gap_12m_summary.csv` and `tables/gap_12m_other_gap.csv` that have both. Two more tests are therefore printed. The t with one lag is the mean of the year means over its Hansen-Hodrick standard error with one lag "
        "(each year paired with the year before and the year after), on the same degrees of freedom. The last row reads the Hansen-Hodrick t of the mean of Table 1 on those degrees of freedom and not as a normal variable. "
        "None of the three is an exact test. They are used for one thing: a mean that the rule of the page calls negative or positive is graded on the most cautious of the three, the largest two-sided p: "
        f"below 1 %, it passes at 1 %; from 1 % to {100 * LIMIT_P:.1f} %, it passes at 5 % and not at 1 %; from {100 * LIMIT_P:.1f} % to 5 %, it is at the 5 % limit; above 5 %, it does not pass at 5 % and the sign is not established "
        "(columns `windows_p_most_cautious` and `sign_once_windows_counted` of the CSV files). An estimate that is not the mean of one sample of trades (the combined estimates and the difference of Table 1e) has no such test. "
        "The entry-year means of the held gap are in `tables/gap_12m_entry_years.csv`."
    )

    # ---- table 1d: chains of non-overlapping trades
    body = []
    chain_keys = [("weekly", "copula"), ("weekly_2007_2016", "copula"), ("weekly_2017_2025", "copula"), ("monthly", "copula"), ("priced", "copula"), ("priced", "lc"), ("priced", "cc")]  # fmt: skip
    lengths: set[int] = set()
    for sample, price in chain_keys:
        r = pick(gap, sample, "hedged", price)
        if r is None or not r["chains"]:
            continue
        lengths |= set(range(int(r["chain_trades_min"]), int(r["chain_trades_max"]) + 1))
        of, p_count = chain_of(r)
        body.append([SAMPLE_LABEL.get(sample, r["sample_label"]), PRICE_LABEL[price], str(r["n_trades"]), str(int(r["chains"])), span(int(r["chain_trades_min"]), int(r["chain_trades_max"])),
                     f"{int(r['chain_median_negative'])} of {of}", p_count, r["chain_median_at_5pct"], f"{int(r['chains_count_below_5pct'])} of {int(r['chains'])}",
                     f"{100 * r['chain_share_negative_min']:.0f} % to {100 * r['chain_share_negative_max']:.0f} %", f"{num(r['chain_means_mean_pct'], 2, True)} ({int(r['chain_means_negative'])} of {int(r['chains'])})",
                     f"{num(r['chain_t_median'], 2, True)} ({num(r['chain_t_min'], 2, True)} to {num(r['chain_t_max'], 2, True)})", f"{int(r['chain_t_beyond_crit5'])} of {int(r['chains'])}"])  # fmt: skip
    md.append("**Table 1d.** The delta-hedged gap on chains of non-overlapping one-year trades.")
    md.append(table(["sample", "price", "trades", "chains", "trades per chain", "negative trades on the median chain", "two-sided binomial p of that count", "the count against one half",
                     "chains whose count of negative trades has a p below 5 %", "share of negative trades, lowest to highest chain",
                     "mean of the chain means, % of notional (chains with a negative mean)", "t of a chain mean: median (lowest to highest)", "chains whose |t| passes its 5 % critical value"], body))  # fmt: skip
    needs = [(k, chain_needs(k)) for k in sorted(lengths)]
    md.append(
        "A chain starts at one trade entered in the first 365 days of the sample and takes, each time, the first trade entered on or after the expiry of the current one: its trades do not overlap. "
        "There is one chain per trade of the first year; the chains share most of their years, so they are not independent of one another and their count is not a sample size. "
        "The median chain is the middle one when the chains are ordered by their number of negative trades and, among chains with the same number, by their number of trades, the shorter first "
        "(the lower of the two middle ones when the number of chains is even); the binomial p is the two-sided probability of a count at least as far from one half under a fair coin. "
        "Several chains can have the median count on different numbers of trades, and the p of a count depends on that number. Where they do, the table gives the count on the shortest and on the longest of them, "
        "each with its p, and the count is judged on all of them: it is called distinguishable from one half at 5 % when its p is below 5 % on each of these chains, not distinguishable when on none, "
        "and at the 5 % limit when it is below 5 % on some and not on others, for the verdict then turns on the tie-break "
        "(columns `chain_median_trades_shortest`, `chain_median_trades_longest`, their two p and `chain_median_at_5pct` of `tables/gap_12m_summary.csv`; `chain_median_trades` and `chain_median_binomial_p` are those of the tie-break). "
        "The next column counts, over all the chains of the sample, those whose own count of negative trades has a p below 5 %; the chains share most of their trades, so it is not a number of separate tests. "
        + ("A count on so few trades passes at 5 % only far from one half: it takes at least "
           + listing([(f"{k} out of {n_}" if k <= n_ else f"more than the {n_} of a chain of {n_}") + (" trades of the same sign" if i == 0 else "") for i, (n_, k) in enumerate(needs)]) + ". " if needs else "")
        + "The t of a chain mean is the plain one (standard deviation of its trades over the square root of their number), with its own critical value (trades of the chain less one degrees of freedom)."
    )  # fmt: skip

    # ---- table 1e: the months without an LC price
    c = comb[comb["structure"] == "hedged"]
    body = [[r["label"], str(r["n_trades_gap"]), str(r["n_trades_price_shift"]) if r["n_trades_price_shift"] else "", pm(r["value_pct"], r["se_hh_pct"], 3, bool(r["newey_west_fallback"])),
             t_text(r["t_ratio"]) if np.isfinite(r["t_ratio"]) else "", interval(r["ci95_lo_pct"], r["ci95_hi_pct"]) if np.isfinite(r["ci95_lo_pct"]) else "",
             (sign_cell(r["sign_by_rule"], r["value_pct"], disagree(r["t_ratio"], r["ci95_lo_pct"], r["ci95_hi_pct"]), noun_of(r["estimator"])) if r["sign_by_rule"] != "no error" else "")
             + (f"; once its windows are counted: {r['sign_once_windows_counted']}" if r["sign_once_windows_counted"] else ""), r["note"]] for r in c.to_dict("records")]  # fmt: skip
    tested = [r for r in c.to_dict("records") if r["sign_once_windows_counted"] not in ("", "not tested")]  # fmt: skip
    md.append(f"**Table 1e.** The months without an LC price: the delta-hedged gap at the copula's price with and without a priced row, and the estimate for all {len(m)} monthly trades; mean P&L per trade, % of notional.")  # fmt: skip
    md += with_note(table(["quantity", "trades in the mean of the gap", "trades in the mean price shift", "mean ± s.e.", "t", "95 % interval", "sign by the rule of this page", "note"], body))  # fmt: skip
    md.append(
        f"± is a Hansen-Hodrick standard error with {LAG_MONTHLY} lags in calendar months of entry; a group of fewer than {MIN_N} trades has none, and a group of fewer than {MIN_BLOCKS * BLOCK_MONTHLY} trades has no interval and so no sign by the rule. The combined estimate is the mean of the gap at the copula's price over all "
        f"{len(m)} monthly trades plus the mean of (copula price - model price) over the priced trades; its error is computed on the sum (the term of a monthly trade is its gap less the mean, over {len(m)}, plus, when it is priced, "
        "its price shift less the mean shift, over the number of priced trades), and its interval resamples the monthly trades in blocks of 12 and recomputes both means. It is an estimate, not a measurement: it assumes that the months "
        "without a price have the mean price shift of the priced ones. The year-weighted rows give each priced trade the weight (the study's monthly trades of its entry year) / (priced trades of that year): they make every year count as in "
        f"the study and leave unchanged which months are priced inside a year. The same rows for the held gap are in `tables/gap_12m_combined.csv`."
        + "".join(f" The mean that the rule names, {r['label']} ({int(r['n_trades_gap'])} trades), once its windows are counted (Table 1c): {graded(pd.Series(r))}." for r in tested)
    )  # fmt: skip

    # ---- table 2: the trades
    md.append("## 2. The trades")
    nt = meta["not_trade"]
    outside, with_row = meta["monthly_outside_pass"], meta["monthly_outside_with_row"]
    outside_words = f"{len(outside)}, of which {len(meta['monthly_not_reachable'])} without a row ({dates_text(meta['monthly_not_reachable'])})"  # fmt: skip
    if with_row:
        outside_words += f" and {len(with_row)} with a row: " + "; ".join(
            f"{q['date']}, {q['state']} from " + ("the earlier pass over the first entry of each year (it is the first entry of its year)" if q["first_entry_of_its_year"] else "a row that is not of this pass")
            for q in with_row)  # fmt: skip
    body = [
        ["the study's weekly one-year trades (basket B1, the 30 Dow names; with an outcome, the entry with a stuck member left out)", str(len(w)), f"{w['date'].min()} to {w['date'].max()}"],
        [f"the study's monthly one-year trades: one per calendar month ({meta['monthly_flag']} flagged `monthly` by the study's table" + (f", and {dates_text(meta['monthly_added'])}, a date of the LC pass in a month without a flagged entry)" if meta["monthly_added"] else ")"), str(len(m)), f"{m['date'].min()} to {m['date'].max()}"],
        [f"dates of the LC pass at 12m (the dates on which the study's model S converged, and today, {TODAY}, with an entry at 12m; model S is the study's skewed one-factor model, table `model_s_3m.parquet`)", str(len(meta["pass_dates"])), f"{meta['pass_dates'][0]} to {meta['pass_dates'][-1]}"],
        ["of them with a row file now", str(len(meta["pass_with_row"])), ""],
        ["of them pending (no row file yet)", str(len(meta["pass_pending"])), ""],
        [f"row files outside the pass's list ({dates_text(meta['rows_outside_pass'])})", str(len(meta["rows_outside_pass"])), ""],
        ["row files read", str(meta["rows"]), ""],
        ["of them priced (status ok or check, finite prices)", str(meta["priced"]), ""],
        [f"of them failed ({dates_text(meta['failed'])})", str(len(meta["failed"])), ""],
        [f"of them with a status ok or check and a non-finite price, left out ({dates_text([r['date'] for r in meta['other_rows']])})", str(len(meta["other_rows"])), ""],
        [f"priced dates with no outcome: entered after {w['date'].max()} ({dates_text(nt['no_outcome'])})", str(len(nt["no_outcome"])), ""],
        [f"priced dates that are not a trade of the study: a member with no price move over the window ({dates_text(nt['stuck'])})", str(len(nt["stuck"])), ""],
        [f"priced dates that are neither a monthly entry of the study nor a date of the pass, left out ({dates_text(nt['other'])})", str(len(nt["other"])), ""],
        ["**monthly trades with a priced LC row (the sample of the LC and CC numbers)**", f"**{n}**", f"{x['date'].min() if n else 'n/a'} to {x['date'].max() if n else 'n/a'}"],
        ["monthly trades the row folder can reach (a date of the pass, or a row already there)", str(meta["reach"]), ""],
        ["of them not priced yet (pending)", str(len(meta["pending_trades"])), ""],
        ["of them failed", str(len(meta["failed_trades"])), ""],
        [f"monthly trades of the study outside the pass's list (model S did not converge on the date): {outside_words}", str(len(outside)), ""],
    ]  # fmt: skip
    md.append("**Table 2a.** The counts.")
    md.append(table(["", "count", "entries"], body))
    by_year = meta["by_year"]
    md.append(
        "**Table 2b.** Monthly trades per entry year: the study's, those the row folder can reach, and those priced now."
    )
    md.append(table(["entry year", *by_year["entry_year"]], [
        ["the study's monthly trades", *map(str, by_year["monthly_trades"])],
        ["on a date of the pass or with a row", *map(str, by_year["on_a_date_of_the_pass_or_with_a_row"])],
        ["priced now", *map(str, by_year["priced"])],
    ]))  # fmt: skip
    md.append(
        f"Pending dates of the pass ({len(meta['pass_pending'])}): {dates_text(meta['pass_pending'])}. "
        f"Failed rows: {'; '.join(r['date'] + ' (' + str(r['reason']) + ')' for r in meta['failed_reasons']) or 'none'}. "
        f"Files that could not be parsed (being written): {dates_text(meta['unreadable'])}. "
        f"Status of the {n} priced trades: {', '.join(f'{k} {v}' for k, v in sorted(meta['status_counts'].items())) or 'n/a'}"
        f"{' (' + '; '.join(f'{k!r} on {v}' for k, v in meta['check_reasons'].items()) + ')' if meta['check_reasons'] else ''}; the stored status is the one of the row's gating checks "
        f"({', '.join(GATES)}), recomputed here. "
        + gate_words(x, meta)
        + f"Flags on the {n} trades: clipped mass above {100 * CLIP_FLAG:.0f} % at λ = 0 or at the cap inside ±2.5 standard deviations (λ is the model's correlation multiplier) on {int(x['flag_clip'].sum())}; a name kept unscreened on {int(x['flag_unscreened'].sum())}; "
        f"a name priced beyond its last kept expiry on {int(x['flag_extrapolated'].sum())}; target of the Dow index options (DJX) repaired on {int(x['flag_djx_repaired'].sum())}."
    )  # fmt: skip

    # ---- table 3: the validation
    md.append("## 3. Validation: the study's printed one-year numbers recomputed")
    n_ok = int(val["match"].sum())
    first_page = val[~val["rests_on"].str.contains("other gap") | val["key"].str.startswith("P_D")]
    added = val[val["key"].str.startswith("GAP_rho")]
    # the header of Table 3; the sentence above the table names its fifth column by this list
    val_head = ["quantity", "printed", "this script", "match to the printed digits", f"t at {LAG_DRAFT} lags", "source of the printed number", "what rests on it"]  # fmt: skip
    md.append(
        f"**Table 3.** {n_ok} of the {len(val)} printed numbers are reproduced to the printed digit on the study's {len(w)} weekly trades (copula price): {int(first_page['match'].sum())} of the {len(first_page)} of the gap, its legs, "
        f"the prices and the payoffs, and {int(added['match'].sum())} of the {len(added)} that the study prints for its other gap, held (the last rows; section 8). "
        f"The study prints no number on a monthly subset at one year. The hedged gap is not a row of the study's one-year tables: it is the hedged forward minus the hedged package of `T13_other_runs.csv` "
        f"(the identity `GAP_H = PF_H - PKG_v_H` holds on every trade) and the row of the gap in the draft's table. "
        f"t-ratios use {LAG_WEEKLY} lags, the lag of the study's code (`disp_tables.LAG['12m']`); the sidenote of the draft's table says {LAG_DRAFT} lags: the value at {LAG_DRAFT} lags is in the column '{val_head[4]}'. "
        f"The last column names what rests on the printed number at the LC or CC price (the gap's own numbers, the means of its two legs, the prices and the payoffs; nothing rests on the t-ratio of a leg)."
        + (" No t-ratio of this table uses the Newey-West fallback." if not val["newey_west_fallback"].any() else f" {int(val['newey_west_fallback'].sum())} t-ratios of this table use the Newey-West fallback (marked {MARK}).")
    )  # fmt: skip
    body = [[r["quantity"], r["printed"], num(r["this_script"], r["digits"] + 1 if r["digits"] else 0, r["printed"][0] in "+-") + (MARK if r["newey_west_fallback"] else ""), "yes" if r["match"] else "NO",
             num(r["this_script_51_lags"], r["digits"] + 1, True) if np.isfinite(r["this_script_51_lags"]) else "", r["source"], r["rests_on"] or "nothing"] for r in val.to_dict("records")]  # fmt: skip
    md.append(table(val_head, body))
    for g, words in (("held", "the held gap"), ("hedged", "the hedged gap"), ("payout", "the payout per 1 of premium"), ("other gap", "the study's other gap")):  # fmt: skip
        if not valid[g]:
            bad = val[val["rests_on"].str.contains(g) & ~val["match"]]
            md.append(f"**Not reproduced: {'; '.join(bad['quantity'])}. No number of {words} is given at the LC or CC price.**")  # fmt: skip
    for r in val[~val["match"] & (val["rests_on"] == "")].to_dict("records"):
        same = val[(val["key"] == r["key"]) & val["match"]]
        at51 = f" and {num(r['this_script_51_lags'], 4)} at {LAG_DRAFT} lags" if np.isfinite(r["this_script_51_lags"]) else ""  # fmt: skip
        md.append(
            f"Not reproduced: {r['quantity']}, printed {r['printed']} in {r['source']}; this script {num(r['this_script'], 4)} at {LAG_WEEKLY} lags{at51}, which rounds to {num(r['this_script'], r['digits'])}. "
            + (f"The same quantity is printed {same.iloc[0]['printed']} in {same.iloc[0]['source']} and is reproduced there. " if len(same) else "")
            + "It is the t-ratio of one leg of the gap: no number at the LC or CC price rests on it (the gap's own numbers and the means of its two legs are reproduced)."
        )  # fmt: skip
    md.append(
        f"Checks of the standard error on every run: `hh_se` of this script equals the study's `disp_stats.mean_se` on the weekly trades ({LAG_WEEKLY} lags)"
        + (
            f" and the study's `disp_stats.subset_mean_se` on the priced trades placed at their row among the monthly trades ({LAG_MONTHLY} lags)"
            if any(k.startswith("subset") for k in checks)
            else ""
        )
        + f"; largest difference {max(checks.values()):.1e}."
    )

    # ---- table 4: the lags
    md.append("## 4. The standard error against the lag")
    body = []
    for structure, _, words in STRUCTURES:
        for sample, price in HEADLINE:
            r = pick(gap, sample, structure, price)
            if r is None:
                continue
            body.append([words, f"{sample} ({r['n_trades']})", PRICE_LABEL[price], num(r["mean_pnl_pct"], 3, True), num(r["se_lag0_pct"]), f"{num(r['se_hh_pct'])}{MARK if r['newey_west_fallback'] else ''} ({r['hh_lags']})",
                         f"{num(r['se_lag_more_1_pct'])}{MARK if r['se_lag_more_1_fallback'] else ''} ({r['lag_more_1']})", f"{num(r['se_lag_more_2_pct'])}{MARK if r['se_lag_more_2_fallback'] else ''} ({r['lag_more_2']})",
                         num(r["se_boot_pct"])])  # fmt: skip
    md.append(
        "**Table 4.** Hansen-Hodrick standard error of the mean P&L at 0 lags, at the convention and at two longer lags (lags in brackets), and the standard deviation of the bootstrap means; % of notional."
    )
    md += with_note(table(["structure", "sample (trades)", "price", "mean", "0 lags (trades taken as independent)", "the convention", "longer", "longer still", "block bootstrap"], body))  # fmt: skip
    n_fb = int(gap[["newey_west_fallback", "se_lag_more_1_fallback", "se_lag_more_2_fallback", "share_negative_se_fallback"]].to_numpy(bool).sum())  # fmt: skip
    md.append(
        "The 0-lag column takes the trades as independent: it is not an error to quote and is printed only to show what the overlap adds. "
        "The Hansen-Hodrick estimator weights every lag equally, so its value can fall when lags are added. "
        + (f"Of the standard errors of the means and shares of `tables/gap_12m_summary.csv`, {n_fb} use the Newey-West fallback (its fallback columns name them)." if n_fb
           else "No standard error of a mean or of a share of `tables/gap_12m_summary.csv` uses the Newey-West fallback (its fallback columns are all false).")
    )  # fmt: skip

    # ---- table 5: the prices
    md.append("## 5. The entry prices and their differences")
    if n:
        body, cells = [], []
        diffs = (("copula_minus_lc", "copula minus LC"), ("copula_minus_cc", "copula minus CC"), ("cc_minus_lc", "CC minus LC"))  # fmt: skip
        for key, label, g, weighted in subsamples(x):
            if key not in ("priced", "entries_2007_2016", "entries_2017_2025", "year_weighted") or len(g) == 0:  # fmt: skip
                continue
            q = shifts[shifts["sample"] == key].set_index("difference")
            wts = g["year_weight"] if weighted else pd.Series(1.0, index=g.index)
            mean = {p: 100 * float((wts * g[PRICE_COL[p]]).sum() / wts.sum()) for p in PRICES}
            bound = {p: 100 * float((wts * g[PRICE_SE[p]]).sum() / wts.sum()) for p in PRICES}
            body.append([label, str(len(g)), *(f"{num(mean[p])} (± {num(bound[p], 4)})" for p in PRICES),
                         *(f"{pm_boot(q.loc[k, 'mean_pct'], q.loc[k, 'se_hh_pct'], q.loc[k, 'se_boot_pct'], 3, bool(q.loc[k, 'newey_west_fallback']))} (± {num(q.loc[k, 'mc_bound_pct'], 4)})" for k, _ in diffs)])  # fmt: skip
            cells += [{"label": f"{words}, {label}", "mean": q.loc[k, "mean_pct"], "se": q.loc[k, "se_hh_pct"], "boot": q.loc[k, "se_boot_pct"], "sign": q.loc[k, "sign_by_rule"]} for k, words in diffs]  # fmt: skip
        md.append("**Table 5.** Mean entry price of the forward at the three prices and the mean differences, % of notional. A difference is the same for the held and the hedged gap. "
                  "Copula minus LC = (copula minus CC) + (CC minus LC), row by row.")  # fmt: skip
        md += with_note(table(["sample", "trades", "copula price (Monte Carlo bound)", "LC price (Monte Carlo bound)", "CC price (Monte Carlo bound)",
                               *(f"{words} ± s.e. [bootstrap s.d.] (Monte Carlo bound)" for _, words in diffs)], body))  # fmt: skip
        q = shifts[shifts["sample"] == "priced"].set_index("difference")
        every = S["meta"]["all_priced_ratio"]
        md.append(
            f"± s.e. is the Hansen-Hodrick standard error across entry dates of the mean difference ({LAG_MONTHLY} lags in calendar months). The Monte Carlo bound is the mean of the per-date standard errors "
            f"(for a difference, of the sum of the two): the dates are priced on the same seeds, so their errors may not average out; it is the pricing noise given the calibrated model and holds neither the calibration's noise nor the budget. "
            f"On the {n} trades the pooled ratios (sum over sum) are LC over copula {num(q.loc['copula_minus_lc', 'pooled_ratio'], 4)}, CC over copula {num(q.loc['copula_minus_cc', 'pooled_ratio'], 4)} and LC over CC {num(q.loc['cc_minus_lc', 'pooled_ratio'], 4)}. "
            f"Per trade, CC over copula runs from {num(q.loc['copula_minus_cc', 'ratio_min'], 2)} ({q.loc['copula_minus_cc', 'ratio_min_date']}) to {num(q.loc['copula_minus_cc', 'ratio_max'], 2)} ({q.loc['copula_minus_cc', 'ratio_max_date']}) "
            f"and LC over copula from {num(q.loc['copula_minus_lc', 'ratio_min'], 2)} ({q.loc['copula_minus_lc', 'ratio_min_date']}) to {num(q.loc['copula_minus_lc', 'ratio_max'], 2)} ({q.loc['copula_minus_lc', 'ratio_max_date']}): "
            f"the copula is priced on the study's index level and CC and LC on the model's index target, so a difference from the copula carries the gap between the two; LC against CC is on the same target, "
            f"a target that the LC model does not reach on {int(x['flag_clip'].sum())} of the {n} dates (the flag for the clipped mass): it is the LC model against its constant-correlation companion "
            f"(LC over CC from {num(q.loc['cc_minus_lc', 'ratio_min'], 2)} to {num(q.loc['cc_minus_lc', 'ratio_max'], 2)})."
            + (f" Over all {every['n']} priced rows, the dates that are not trades included, the lowest CC over copula is {num(every['min'], 2)} ({every['min_date']}, {every['min_what']})." if every["n"] else "")
            + "".join(f" {words} on the {n} trades, once its windows are counted (Table 1c): {graded(q.loc[k])}."
                      for k, words in (("copula_minus_lc", "Copula minus LC"), ("copula_minus_cc", "Copula minus CC"), ("cc_minus_lc", "CC minus LC")) if q.loc[k, "sign_once_windows_counted"] not in ("", "not tested"))
        )  # fmt: skip
        md.append(se_gap_words(cells, "mean differences of Table 5"))

    # ---- tables 6 and 7: the sub-samples
    md.append("## 6. Sub-samples")
    for number, (structure, _, words) in ((6, STRUCTURES[1]), (7, STRUCTURES[0])):
        if not valid[structure]:
            md.append(f"**Table {number}.** The {words} gap on sub-samples: not given, the study's printed numbers are not reproduced (Table 3).")  # fmt: skip
            continue
        body, named, split, judged = [], [], [], 0
        cells: list[dict[str, Any]] = []

        def cell(r: pd.Series) -> list[str]:
            t = f" (t {t_text(r['t_ratio'])})" if np.isfinite(r["t_ratio"]) else ""
            return [pm_boot(r["mean_pnl_pct"], r["se_hh_pct"], r["se_boot_pct"], 3, bool(r["newey_west_fallback"])) + t, interval(r["ci95_lo_pct"], r["ci95_hi_pct"], 2)]  # fmt: skip

        def both(r: pd.Series, label: str, mean: str = "mean_pnl_pct") -> dict[str, Any]:
            return {"label": label, "mean": r[mean], "se": r["se_hh_pct"], "boot": r["se_boot_pct"], "sign": r["sign_by_rule"]}  # fmt: skip

        def judge(r: pd.Series, label: str, price: str, named: list[str], split: list[str]) -> int:  # fmt: skip
            """Whether the mean has a standard error (1 or 0); the words of a mean that the rule
            names go to ``named`` and those of a mean on which the two conditions of the rule
            disagree to ``split``."""
            why = disagree(r["t_ratio"], r["ci95_lo_pct"], r["ci95_hi_pct"])
            if r["sign_by_rule"] in ("negative", "positive"):
                named.append(f"{label} at {at_price(price)} ({pm(r['mean_pnl_pct'], r['se_hh_pct'], 3, bool(r['newey_west_fallback']))}, t {t_text(r['t_ratio'])}; {int(r['n_trades'])} trades, "
                             f"{span(int(r['chain_trades_min']), int(r['chain_trades_max']))} non-overlapping windows): {graded(r)}")  # fmt: skip
            elif why:
                split.append(f"{label} at {at_price(price)}, t {t_text(r['t_ratio'])}, interval {interval(r['ci95_lo_pct'], r['ci95_hi_pct'], 2)}{why}")  # fmt: skip
            return int(np.isfinite(r["t_ratio"]))

        for key in ("weekly", "weekly_2007_2016", "weekly_2017_2025", "monthly", "monthly_2007_2016", "monthly_2017_2025"):  # fmt: skip
            q = pick(gap, key, structure, "copula")
            body.append([q["sample_label"], str(q["n_trades"]), *cell(q), "", "", "", "", ""])
            judged += judge(q, q["sample_label"], "copula", named, split)
            cells.append(both(q, f"{q['sample_label']} at {at_price('copula')}"))
        for key, label, g, _ in subsamples(x):
            r = {p: pick(gap, key, structure, p) for p in PRICES}
            d = shifts[(shifts["sample"] == key) & (shifts["difference"] == "cc_minus_lc")].iloc[0]
            body.append([f"priced: {label}", str(len(g)), *(c for p in PRICES for c in cell(r[p])),
                         pm_boot(d["mean_pct"], d["se_hh_pct"], d["se_boot_pct"], 3, bool(d["newey_west_fallback"]))])  # fmt: skip
            for p in PRICES:
                judged += judge(r[p], f"priced: {label}", p, named, split)
                cells.append(both(r[p], f"priced: {label} at {at_price(p)}"))
            cells.append(both(d, f"the last column (LC minus CC), priced: {label}", "mean_pct"))
        md.append(f"**Table {number}.** The {words} gap by half on the study's trades (copula price) and on sub-samples of the {n} trades with a priced LC row: mean P&L per trade ± Hansen-Hodrick s.e. "
                  f"({LAG_WEEKLY} lags on weekly trades, {LAG_MONTHLY} lags in calendar months of entry on monthly ones), in square brackets the standard deviation of its block-bootstrap means, its t (the mean over the Hansen-Hodrick s.e.), "
                  f"and the 95 % block-bootstrap interval (blocks of {BLOCK_WEEKLY} weekly or {BLOCK_MONTHLY} monthly consecutive trades of the sub-sample), % of notional.")  # fmt: skip
        md += with_note(table(["sub-sample", "trades", "copula: mean ± s.e. [bootstrap s.d.] (t)", "interval", "LC: mean ± s.e. [bootstrap s.d.] (t)", "interval", "CC: mean ± s.e. [bootstrap s.d.] (t)", "interval",
                               "LC minus CC ± s.e. [bootstrap s.d.]"], body))  # fmt: skip
        md.append(
            f"Of the {judged} means of Table {number} that have a standard error, the rule of the page (|t| at least {T_CRIT} and an interval that excludes zero) names a sign for "
            + (f"{len(named)}. " + " ".join(f"({i}) {words[0].upper()}{words[1:]}." for i, words in enumerate(named, 1)) + " Each is a half or a sub-sample, graded on the most cautious of the three tests of its windows (Table 1c explains them)." if named else "none.")
            + (f" On {len(split)} of the means that the rule leaves without a sign its two conditions disagree: " + "; ".join(split) + "." if split else "")
            + " The sub-samples of the priced trades share their trades (only the two halves share none), so their rows are not separate evidence."
        )  # fmt: skip
        md.append(se_gap_words(cells, f"means of Table {number} (its last column included)"))
    few = x[~x["flag_clip"]]
    md.append(
        f"The dates NOT flagged for the clipped mass are {len(few)} of the {n}: "
        + ("; ".join(f"{r['date']} (clipped mass {r['clip_low_inner_max']:.4f} at λ = 0, {r['clip_high_inner_max']:.4f} at the cap; status {r['status']}"
                     f"{'; a name kept unscreened: ' + str(r['names_unscreened']) if r['flag_unscreened'] else ''}{'; ' + extrapolated_words(r['n_names_extrapolated']) if r['flag_extrapolated'] else ''}"
                     f"{'; DJX target repaired' if r['flag_djx_repaired'] else ''}; "
                     f"hedged gap {num(100 * r['hedged_copula'], 2, True)} copula, {num(100 * r['hedged_lc'], 2, True)} LC, {num(100 * r['hedged_cc'], 2, True)} CC; "
                     f"held gap {num(100 * r['held_copula'], 2, True)}, {num(100 * r['held_lc'], 2, True)}, {num(100 * r['held_cc'], 2, True)})" for r in few.to_dict("records")) or "none")
        + "." + not_clean_words(few)
        + f" A sub-sample of fewer than {MIN_N} trades has no standard error and no interval; one of fewer than {MIN_BLOCKS * BLOCK_MONTHLY} monthly trades ({MIN_BLOCKS} blocks) has a standard error and no interval, hence no sign by the rule. "
        f"The weighted row gives each priced trade of an entry year the weight (the study's monthly trades of that year) / (priced trades of that year), so that every year counts as in the study's {len(m)} monthly trades"
        + (f"; no trade is priced in {', '.join(meta['years_not_priced'])}, which the weighted row leaves out" if meta["years_not_priced"] else "")
        + "; it does not correct which months are priced inside a year (Table 1e). 'LC minus CC' is the mean CC price minus LC price (Table 5), the same for the held and the hedged gap."
    )  # fmt: skip

    # ---- table 8: the payout
    md.append("## 7. Payout per 1 of premium of the forward")
    body = []
    for key in ("weekly", "monthly", "priced", "entries_2007_2016", "entries_2017_2025"):
        for r in payout[payout["sample"] == key].to_dict("records"):
            body.append([r["sample_label"], str(r["n_trades"]), PRICE_LABEL[r["price"]], num(r["mean_premium_pct"]), num(r["mean_payoff_pct"]),
                         num(r["payout_per_1_of_premium"]), interval(r["ci95_lo"], r["ci95_hi"], 3, False), num(r["mc_bound"], 4) if r["price"] != "copula" else "study table"])  # fmt: skip
    md.append(
        "**Table 8.** Payout per 1 of premium of the Palladium forward held to expiry: sum of the payoffs `D` over the sum of the prices."
    )
    md.append(table(["sample", "trades", "price", "mean premium (% of notional)", "mean payoff (% of notional)", "payout per 1 of premium", "95 % interval", "Monte Carlo bound"], body))  # fmt: skip
    md.append(
        f"The interval is the study's `disp_stats.bootstrap_ratio` (circular blocks, {N_RESAMPLES} resamples, seed {SEED}), with the block the study uses at one year for weekly entries ({BLOCK_WEEKLY}, `disp_tables.STEP['12m']`) "
        f"and its monthly counterpart ({BLOCK_MONTHLY} = {BLOCK_WEEKLY} // 4, the rule of `disp_tables2.model_s_tables`): one year of entries per block. The study prints no interval for the one-year forward; "
        f"its printed price over payoff (Table 3) is the inverse of the weekly row. The Monte Carlo bound is the ratio times the sum of the per-date standard errors of the prices over the sum of the prices."
        + (
            ""
            if valid["payout"]
            else " **No LC or CC row: the study's printed numbers are not reproduced (Table 3).**"
        )
    )

    # ---- definitions and sources, with the study's other gap
    md.append("## 8. Definitions, the study's other gap, errors and sources")
    body = []
    for sample, price in HEADLINE:
        r = pick(other, sample, "hedged_rho", price)
        if r is None:
            continue
        body.append([SAMPLE_LABEL[sample], PRICE_LABEL[price], str(r["n_trades"]), pm(r["mean_pnl_pct"], r["se_hh_pct"], 3, bool(r["newey_west_fallback"])), t_text(r["t_ratio"]),
                     interval(r["ci95_lo_pct"], r["ci95_hi_pct"]), sign_cell(r["sign_by_rule"], r["mean_pnl_pct"], disagree(r["t_ratio"], r["ci95_lo_pct"], r["ci95_hi_pct"])), tests_cell(r)])  # fmt: skip
    md.append(
        "**The gap of the report is the vega-neutral one**: Palladium forward minus the package of one unit of the single-name straddles against one unit of the basket straddle (the study's columns `GAP_U` and `GAP_H`); "
        "every table above is about it. The study also computes a second gap, against the correlation-neutral package, in which the basket straddle is scaled by the study's ratio `lambda_rho` so that the package has no sensitivity "
        "to correlation (columns `GAP_rho_U` and `GAP_rho_H`). It is not the gap of the report and it is given here for completeness, delta-hedged, with the same price shift at the LC and CC price."
    )
    md.append("**Table 9.** The study's other gap, delta-hedged (Palladium forward minus the correlation-neutral package, `GAP_rho_H`): mean P&L per trade, % of notional.")  # fmt: skip
    if valid["hedged_rho"] or len(body):
        md += with_note(table(["sample", "price", "trades", "mean P&L ± s.e.", "t", "95 % interval", "sign of the mean by the rule of this page",
                               "the three tests of the windows: plain t of the entry-year means (degrees of freedom; 5 % critical value; two-sided p); t with one lag between adjacent years (p); p of the Hansen-Hodrick t on those degrees of freedom; and the grade of a named sign on the largest p"], body))  # fmt: skip
    md.append(
        f"Errors, interval and rule as in Table 1; the three tests of the windows as in Table 1c: the plain t of the entry-year means overstates |t| where adjacent year means are positively correlated, and a named sign is graded on the largest of the three p-values."
        f"{seed_words(other, 'other_seeds_mean_interval_same_side_of_zero', 'contains zero or excludes it')} The study prints this gap held to expiry only at one year ({int(added['match'].sum())} of its {len(added)} printed numbers are reproduced, Table 3); "
        "the hedged one is the study's column (the identity `GAP_rho_H = PF_H - PKG_rho_H` holds on every trade)."
        + (
            ""
            if valid["hedged_rho"]
            else " **No LC or CC row: the study's printed numbers are not reproduced (Table 3).**"
        )
        + " The rows are in `tables/gap_12m_other_gap.csv`."
    )
    md.append("\n".join([
        "- **Trade.** One entry date of the study at one year on basket B1 (the 30 Dow names) at unit notional, with its outcome: a row of `scripts/disp_tables.py::load('12m', 'B1')` "
        f"(`entries_12m.parquet` merged with `outcomes_12m.parquet`; the entry with a member stuck over the window, {dates_text(meta['stuck'])}, is left out by the study). The study has no one-year entry after {w['date'].max()} but today's ({TODAY}), which has no outcome.",
        "- **Palladium forward.** The forward on the dispersion of the basket: it pays at expiry `D`, the weighted sum over the names of the absolute difference between the name's performance and the basket's over the window, "
        "against its price paid at entry (the study's `PF_U = D - P_D`: the forward is bought). The copula price `P_D` is the study's own price of it.",
        "- **Gap.** Palladium forward minus the vega-neutral package (long the single-name straddles, short the basket straddle, one unit each; it pays G). Held to expiry: the study's column `GAP_U` = `G - P_G`. "
        "Delta-hedged: `GAP_H` = `PF_H - PKG_v_H`, the forward and both straddle legs hedged daily by the study (`PF_H = PF_U + hedge_PF`). Both are fractions of the notional, printed here times 100.",
        "- **At the LC or CC price.** P&L at a model's price = P&L at the copula's price + (copula price `P_D` - model price), per trade; LC price `ED_lc`, CC price `ED_cc` of the date's row. Only the entry price of the forward changes: "
        "the delta hedge and the package are the study's, and the forward is not re-hedged with the deltas of LC or CC.",
        f"- **Monthly trades.** One per calendar month of entry: the study's trades flagged `monthly` in `entries_12m.parquet`, and the trades entered on a date of the LC pass (`scripts/disp_lcm.py::study_dates`). "
        f"Calendar months between the first and the last without a monthly trade: {len(meta['months_without_trade'])} ({', '.join(month_text(k) for k in meta['months_without_trade']) or 'none'}).",
        f"- **± of a mean P&L.** Hansen-Hodrick standard error (uniform kernel; the study's `disp_stats.mean_se`): {LAG_WEEKLY} lags on weekly entries (the study's `disp_tables.LAG['12m']`), {LAG_MONTHLY} lags on monthly entries "
        "(a one-year trade overlaps the eleven monthly entries after it). On the monthly trades a lag is a calendar month of entry: two trades are paired when they are entered at most 11 months apart, whatever number of months is missing between them. "
        f"{MARK} marks a cell where the Hansen-Hodrick variance is not positive and the Newey-West value at twice the lag is printed in its place (columns `newey_west_fallback`, `se_method` and the other fallback columns of the CSV files).",
        f"- **t, and the sign of a mean.** t is the mean over its Hansen-Hodrick standard error. A mean is called **negative** or **positive** only when |t| is at least {T_CRIT} and its 95 % interval excludes zero; "
        f"**about zero, sign not determined** when that fails and |t| is below {ABOUT_ZERO}; otherwise the page gives the sample mean and its error and says that **the sign of the mean is not established**. "
        f"A t that two decimals would print on the other side of {T_CRIT} or of {ABOUT_ZERO} is printed with three. "
        f"The entries span {(pd.Timestamp(w['date'].max()) - pd.Timestamp(w['date'].min())).days / 365.25:.1f} years: about {len(w) // BLOCK_WEEKLY} one-year windows laid end to end ({len(w)} weekly trades / {BLOCK_WEEKLY}), so {T_CRIT} is a low bar. "
        "That is why a mean that passes the rule is also graded on three tests that count its windows (Table 1c): the plain t of its entry-year means, the same t with one lag between adjacent years, and its Hansen-Hodrick t read on "
        "the degrees of freedom of the year means (years less one). The plain t of the year means overstates |t| where adjacent year means are positively correlated, so the grade is on the most cautious of the three, the largest two-sided p: "
        f"the mean passes at 1 % (p below 1 %), passes at 5 % and not at 1 % (p from 1 % to {100 * LIMIT_P:.1f} %), is **at the 5 % limit** (p from {100 * LIMIT_P:.1f} % to 5 %) or does not pass at 5 %, and then its sign is not established.",
        f"- **95 % interval.** Percentiles 2.5 and 97.5 of the means of {N_RESAMPLES} circular-block resamples of the trades in date order (seed {SEED}, the study's; blocks of {BLOCK_WEEKLY} weekly or {BLOCK_MONTHLY} monthly entries). "
        f"A block is consecutive trades of the sample: where months are missing it spans more than a year. The intervals of a share and of a median are the same percentiles of the share and of the median of each resample. "
        f"A sample shorter than {MIN_BLOCKS} blocks ({MIN_BLOCKS * BLOCK_MONTHLY} monthly or {MIN_BLOCKS * BLOCK_WEEKLY} weekly trades) gets no interval: with a single block the resample is the sample itself.",
        "- **Share of negative trades, and the sign of a median.** The share of the trades of a sample whose P&L is below zero is **distinguishable from one half** only when its 95 % interval excludes 50 %; a median is called negative or positive "
        "only when its 95 % interval excludes zero. Both are judged on the interval alone, a lighter rule than the one for a mean, and they are the same statement (a median below zero is a share of negative trades above one half).",
        f"- **Other bootstrap seeds.** The intervals of the three main samples are redrawn on {SEEDS_WORDS}; the page says on how many of them an interval answers as it does at the study's seed.",
        "- **Chain of non-overlapping trades.** From one trade entered in the first 365 days of a sample, the first trade entered on or after the expiry of the current one, repeated to the end (Table 1d). "
        "The median chain is the middle one when the chains are ordered by their number of negative trades, then by their number of trades. Its count of negative trades is judged on every chain that has that count: "
        "distinguishable from one half at 5 % when the two-sided binomial p is below 5 % on each, not distinguishable when on none, and **at the 5 % limit** when on some and not on others "
        "(for a count this is not the grade of a mean of the bullet above: it says that the verdict turns on the tie-break).",
        "- **Combined estimate.** Mean of the gap at the copula's price over all the study's monthly trades + mean of (copula price - model price) over the priced trades (Table 1e).",
        "- **Monte Carlo bound.** Mean over the trades of the per-date Monte Carlo standard error of the price (`ED_lc_se`, `ED_cc_se`; the copula's `P_D_se`). It is not added to the ±: the ± is the sampling error across trades.",
        f"- **Flags of a date** (from its row): clipped mass, `clip_low_inner_max` or `clip_high_inner_max` above {CLIP_FLAG} (the share of particles inside ±2.5 standard deviations whose correlation multiplier is clipped at 0 or at its cap: the model does not reach its index target); "
        "a name kept unscreened, `n_names_unscreened > 0`; a name priced beyond its last kept expiry, `n_names_extrapolated > 0`; DJX target repaired, `n_dropped_calendar_index > 0`.",
        f"- **State of the pass.** The list of the pass is rebuilt at each run by the rule of `scripts/disp_lcm.py::study_dates` ({len(meta['pass_dates'])} dates). The pass is called complete when every date of the list has a readable row file (a failed row counts); "
        f"still running when it is not complete and the newest row file is at most {STALE_MINUTES} minutes old; stopped or stalled otherwise.",
        f"- **Budget and code of the LC and CC prices.** {budget}; commit {commit}; one row per date in `{rel_rows}`. The caveats of the 12m rows are in `outputs/dispersion_lc/pm_update/1y/NUMBERS_1Y.md` (\"Read this first\").",
        "- **Files.** `tables/gap_12m_summary.csv` (every row of Tables 1, 1b, 1d, 4, 6 and 7 with all its errors, its sign by the rule, the three tests of its windows and its fallback columns), `gap_12m_entry_years.csv` (Table 1c, held and hedged), `gap_12m_combined.csv` (Table 1e, held and hedged), "
        "`gap_12m_other_gap.csv` (Table 9), `gap_12m_by_trade.csv` (one row per monthly trade of the study, with its LC row when there is one; its prices and P&Ls are in % of notional and carry the suffix `_pct`, as in the other files: "
        "`GAP_H_pct` is 100 times the study's column `GAP_H` and `ED_lc_pct` 100 times `ED_lc` of the date's row, both fractions of the notional at their source. The columns were renamed on 2026-10-10: until then this file held them "
        "as fractions of the notional under the source's names, without a unit. Its clipped masses are fractions of the particles and its `_se_pct` columns are per-date Monte Carlo errors of a price, not errors across trades), "
        "`gap_12m_validation.csv` (Table 3; `n_trades` is the size of the sample of each statistic), "
        "`gap_12m_shifts.csv`, `gap_12m_payout.csv`, `gap_12m_dates.csv` (every date of the pass and every row file, with its state); `parts/gap_12m.json` (the records; `n` is the size of the sample of the statistic).",
        "- **Rerun.** `cd /Users/idrissdadoun/Code/volsto-lc && OMP_NUM_THREADS=1 .venv/bin/python scripts/pm_later_1y_gap.py` (under a minute, one process; reads the row files that exist; writes only under `outputs/dispersion_lc/pm_update/later/1y_gap`).",
    ]))  # fmt: skip

    # ---- the tables behind the page
    by_trade = m[["date", "expiry", "monthly", "IS", "P_D", "P_D_se", "D", "G", "P_G", "PF_U", "PF_H", "PKG_v_U", "PKG_v_H", "GAP_U", "GAP_H", "GAP_rho_U", "GAP_rho_H"]].merge(
        x[["date", "status", "reason", "ED_lc", "ED_lc_se", "ED_cc", "ED_cc_se", "clip_low_inner_max", "clip_high_inner_max", "n_names_unscreened", "names_unscreened", "n_names_extrapolated",
           "n_dropped_calendar_index", "flag_clip", "flag_unscreened", "flag_extrapolated", "flag_djx_repaired", "year_weight", *(f"{s}_{p}" for s, _, _ in (*STRUCTURES, *OTHER) for p in PRICES)]],
        on="date", how="left")  # fmt: skip
    state_ = np.where(by_trade["date"].isin(x["date"]), "priced", np.where(by_trade["date"].isin(meta["failed"]), "failed",
                      np.where(by_trade["date"].isin(meta["pass_pending"]), "pending", np.where(by_trade["date"].isin(meta["unreadable"]), "file being written", "not a date of the pass"))))  # fmt: skip
    by_trade.insert(1, "lc_state", state_)
    # prices and P&Ls in % of notional with the suffix of the other files (the source: fractions)
    by_trade[list(BY_TRADE_PCT)] = 100 * by_trade[list(BY_TRADE_PCT)]
    by_trade = by_trade.rename(columns={c: f"{c}_pct" for c in BY_TRADE_PCT})
    every_date = sorted(
        set(meta["pass_dates"]) | set(meta["rows_outside_pass"]) | set(meta["unreadable"])
    )
    dates = pd.DataFrame({"date": every_date})
    dates["in_pass_list"] = dates["date"].isin(meta["pass_dates"])
    dates["row"] = np.where(dates["date"].isin(x["date"]) | dates["date"].isin(nt["stuck"] + nt["no_outcome"] + nt["other"]), "priced", np.where(dates["date"].isin(meta["failed"]), "failed",
                            np.where(dates["date"].isin(meta["unreadable"]), "file being written", np.where(dates["date"].isin(meta["pass_pending"]), "pending", "other"))))  # fmt: skip
    dates["trade"] = np.where(dates["date"].isin(m["date"]), "monthly trade of the study", np.where(dates["date"].isin(meta["stuck"]), "not a trade: stuck member",
                              np.where(dates["date"].isin(meta["no_outcome"]), "no outcome yet", "not an entry of the study")))  # fmt: skip
    tables = {"gap_12m_summary": gap, "gap_12m_by_trade": by_trade, "gap_12m_validation": val, "gap_12m_shifts": shifts, "gap_12m_payout": payout, "gap_12m_dates": dates,
              "gap_12m_entry_years": entry_years, "gap_12m_combined": comb, "gap_12m_other_gap": other}  # fmt: skip
    records = make_records(S, gap, shifts, payout, val, comb, other, budget, commit)
    return "\n\n".join(md), records, tables, answer_text


def model_note(price: str, flagged: float, n: int, of: str = "trades of the sample") -> str:
    """The words added to the notes of a record that rests on an LC or CC price: the name of
    the budget and how many of the trades it is computed on are entered on a date flagged for
    the clipped mass.  Empty for a record at the copula's price."""
    if price == "copula" or not np.isfinite(flagged):
        return ""
    return f"; LC and CC prices at the development budget; {int(flagged)} of the {n} {of} are on a date flagged for the clipped mass (the model does not reach its index target there)"  # fmt: skip


def boot_note(se_boot: float) -> str:
    """The standard deviation of the block-bootstrap means, for the notes of a mean that has one."""
    return f"; standard deviation of the block-bootstrap means {se_boot:.4f}" if np.isfinite(se_boot) else ""  # fmt: skip


def chain_note(r: dict[str, Any]) -> str:
    """The note of the record of the count of negative trades on the median chain."""
    if not r["chains"]:
        return "no chain"
    out = f"of {r['chain_median_trades']:.0f} trades; two-sided binomial p {r['chain_median_binomial_p']:.3f}; {int(r['chains'])} chains"  # fmt: skip
    if r["chain_median_trades_shortest"] != r["chain_median_trades_longest"]:
        out += (f"; the {int(r['chain_median_tied'])} chains that have this count have {int(r['chain_median_trades_shortest'])} to {int(r['chain_median_trades_longest'])} trades: "
                f"p {r['chain_median_binomial_p_shortest']:.4f} on the shortest and {r['chain_median_binomial_p_longest']:.4f} on the longest")  # fmt: skip
    return f"{out}; the count against one half, on every chain that has it: {r['chain_median_at_5pct']}"


def make_records(S: dict[str, Any], gap: pd.DataFrame, shifts: pd.DataFrame, payout: pd.DataFrame, val: pd.DataFrame, comb: pd.DataFrame, other: pd.DataFrame, budget: str, commit: str) -> list[dict[str, Any]]:  # fmt: skip
    """Every number of the page as a record of the package's format."""
    meta, out = S["meta"], []
    study = {"budget": pc.BUDGETS["study"], "commit": "the study's tables (no commit column)", "source": SRC_STUDY}  # fmt: skip
    model = {"budget": budget, "commit": commit, "source": f"{SRC_ROWS} + {SRC_STUDY}"}

    def rec(rid: str, quantity: str, value: float, se: float | None = None, **kw: Any) -> None:
        if value is None or not np.isfinite(value):
            kw["notes"] = f"not computed (fewer than {MIN_N} trades); " + kw.get("notes", "")
        out.append(pc.record(f"L.gap12m.{rid}", SECTION, quantity, value, se, tenor=TENOR, **kw))

    counts = (
        ("weekly_trades", "the study's weekly one-year trades", len(S["weekly"]), study),
        ("monthly_trades", "the study's monthly one-year trades (one per calendar month)", len(S["monthly"]), study),
        ("pass_dates", "dates of the LC pass at 12m", len(meta["pass_dates"]), model),
        ("pass_dates_with_row", "dates of the LC pass at 12m with a row file", len(meta["pass_with_row"]), model),
        ("pass_dates_pending", "dates of the LC pass at 12m without a row file", len(meta["pass_pending"]), model),
        ("rows_read", "row files read", meta["rows"], model),
        ("rows_priced", "row files priced (status ok or check, finite prices)", meta["priced"], model),
        ("rows_failed", "row files failed", len(meta["failed"]), model),
        ("priced_no_outcome", "priced dates with no outcome (entered after the study's last one-year trade)", len(meta["not_trade"]["no_outcome"]), model),
        ("priced_stuck", "priced dates that are not a trade of the study (stuck member)", len(meta["not_trade"]["stuck"]), model),
        ("priced_trades", "monthly one-year trades with a priced LC row", len(S["priced"]), model),
        ("priced_trades_clip_flagged", "of them on a date flagged for the clipped mass", int(S["priced"]["flag_clip"].sum()), model),
        ("monthly_trades_pending", "monthly one-year trades on a date of the pass without a row file", len(meta["pending_trades"]), model),
        ("monthly_trades_outside_pass", "monthly one-year trades outside the pass's list", len(meta["monthly_outside_pass"]), model),
        ("monthly_trades_outside_pass_without_row", "monthly one-year trades outside the pass's list and without a row", len(meta["monthly_not_reachable"]), model),
    )  # fmt: skip
    for rid, what, value, src in counts:
        rec(f"sample.{rid}", what, value, unit="count", definition="a count of entry dates (one trade per entry date)", n=int(value), notes="a count, no standard error", **src)  # fmt: skip
    for i, r in enumerate(val.to_dict("records")):
        rec(f"validation.{i:02d}.{r['key']}", f"{r['quantity']}: this script's value of the study's printed {r['printed']}", r["this_script"], unit="as printed", n=int(r["n_trades"]),
            definition="the study's printed one-year number recomputed on the study's weekly trades at the copula's price; n is the size of the sample of the statistic (all the weekly trades, a half, or the windows with a usable strip)",
            notes=f"printed {r['printed']} in {r['source']}; match to the printed digits: {'yes' if r['match'] else 'NO'}", **study)  # fmt: skip
    gaps = {"held": "the vega-neutral package; held to expiry", "hedged": "the vega-neutral package; delta-hedged", "hedged_rho": "the correlation-neutral package (the study's other gap, not the report's); delta-hedged"}  # fmt: skip
    for r in (*gap.to_dict("records"), *other.to_dict("records")):
        src = study if r["sample"].startswith(("weekly", "monthly")) else model
        base = f"{r['structure']}.{r['price']}.{r['sample']}"
        what = f"gap against {gaps[r['structure']]}, at the {PRICE_LABEL[r['price']]} price, {r['sample_label']}"
        definition = (f"mean over trades of the study's P&L of the gap (Palladium forward minus {gaps[r['structure']]}) at the copula's price + (copula forward price - {PRICE_LABEL[r['price']]} forward price) of the entry date, % of notional"
                      + ("; weighted mean, weight = the study's monthly trades of the entry year over the priced trades of that year" if r["weighted"] else ""))  # fmt: skip
        flag = model_note(r["price"], r["n_trades_clip_flagged"], int(r["n_trades"]))
        se_note = (f"se = {r['se_method']} ({r['hh_position']}){boot_note(r['se_boot_pct'])}; sign by the rule of the page: {r['sign_by_rule']}; "
                   f"the Monte Carlo bound of the mean entry price is {r['price_mc_bound_pct']:.4f} % of notional and is not in it{flag}")  # fmt: skip
        common = {"unit": "% of notional", "n": int(r["n_trades"]), **src}
        rec(base, f"{what}: mean P&L per trade", r["mean_pnl_pct"], r["se_hh_pct"], definition=definition, notes=se_note, **common)  # fmt: skip
        extra = (
            ("ci95_lo", "95 % block-bootstrap interval, lower", r["ci95_lo_pct"], "% of notional", f"percentile 2.5 of {N_RESAMPLES} circular-block resamples, blocks of {r['boot_block']} entries, seed {SEED}"),
            ("ci95_hi", "95 % block-bootstrap interval, upper", r["ci95_hi_pct"], "% of notional", f"percentile 97.5 of {N_RESAMPLES} circular-block resamples, blocks of {r['boot_block']} entries, seed {SEED}"),
            ("t_ratio", "t-ratio", r["t_ratio"], "ratio", f"mean over its standard error: {r['se_method']}"),
            ("se_lag0", "standard error at 0 lags (trades taken as independent: not an error to quote)", r["se_lag0_pct"], "% of notional", "Hansen-Hodrick with 0 lags"),
            (f"se_lag{r['lag_more_1']}", f"standard error with {r['lag_more_1']} lags", r["se_lag_more_1_pct"], "% of notional", f"a longer lag, for the sensitivity: {se_method(int(r['lag_more_1']), bool(r['se_lag_more_1_fallback']))}"),
            (f"se_lag{r['lag_more_2']}", f"standard error with {r['lag_more_2']} lags", r["se_lag_more_2_pct"], "% of notional", f"a longer lag, for the sensitivity: {se_method(int(r['lag_more_2']), bool(r['se_lag_more_2_fallback']))}"),
            ("hit_rate", "share of trades with a positive P&L", r["hit_rate"], "fraction of trades", "a share of trades"),
            ("median", "median P&L per trade", r["median_pnl_pct"], "% of notional", "a median over trades"),
            ("median_ci95_lo", "median P&L per trade: 95 % block-bootstrap interval, lower", r["median_ci95_lo_pct"], "% of notional", "percentile 2.5 of the medians of the resamples"),
            ("median_ci95_hi", "median P&L per trade: 95 % block-bootstrap interval, upper", r["median_ci95_hi_pct"], "% of notional", "percentile 97.5 of the medians of the resamples"),
            ("share_negative_ci95_lo", "share of trades with a negative P&L: 95 % block-bootstrap interval, lower", r["share_negative_ci95_lo"], "fraction of trades", "percentile 2.5 of the shares of the resamples"),
            ("share_negative_ci95_hi", "share of trades with a negative P&L: 95 % block-bootstrap interval, upper", r["share_negative_ci95_hi"], "fraction of trades", "percentile 97.5 of the shares of the resamples"),
            ("entry_years_negative", "entry years with a negative mean P&L", float(r["entry_years_negative"]), "count", f"of {int(r['entry_years'])} entry years"),
            ("entry_years_binomial_p", "two-sided binomial p of the count of entry years with a negative mean", r["entry_years_binomial_p"], "probability", "against one half, as if the entry years were independent, which they are not"),
            ("year_means_t", "t-ratio of the mean of the entry-year means", r["year_means_t"], "ratio", f"plain t, {int(r['year_means_df'])} degrees of freedom, 5 % critical value {r['year_means_t_crit5']:.3f}, two-sided p {r['year_means_p']:.4f}; adjacent entry years overlap, "
             f"so it overstates |t| where adjacent year means are positively correlated (lag-1 autocorrelation {r['year_means_lag1_autocorr']:.3f})"),
            ("year_means_t_lag1", "t-ratio of the mean of the entry-year means with one lag between adjacent years", r["year_means_t_lag1"], "ratio",
             f"Hansen-Hodrick with one lag on the year means{' (Newey-West fallback at two lags)' if r['year_means_lag1_fallback'] else ''}, {int(r['year_means_df'])} degrees of freedom, two-sided p {r['year_means_p_lag1']:.4f}"),
            ("windows_p_most_cautious", "largest two-sided p of the three tests that count the windows", r["windows_p_most_cautious"], "probability",
             f"plain t of the entry-year means p {r['year_means_p']:.4f}; with one lag p {r['year_means_p_lag1']:.4f}; Hansen-Hodrick t read on {int(r['year_means_df'])} degrees of freedom p {r['hh_t_p_on_year_df']:.4f}; "
             f"grade of the sign: {r['sign_once_windows_counted'] or 'none, the rule names no sign'}"),
            ("share_negative_z", "share of trades with a negative P&L less one half, over its standard error", r["share_negative_z_against_half"], "ratio", "printed for comparison; the rule for a share uses its interval"),
            ("chain_median_negative", "negative trades on the median chain of non-overlapping trades", float(r["chain_median_negative"]), "count", chain_note(r)),
            ("chains_count_below_5pct", "chains of non-overlapping trades whose count of negative trades has a two-sided binomial p below 5 %", float(r["chains_count_below_5pct"]), "count",
             f"of {int(r['chains'])} chains, which share most of their trades: not a number of separate tests" if r["chains"] else "no chain"),
        )  # fmt: skip
        for suffix, label, value, unit, note in extra:
            rec(f"{base}.{suffix}", f"{what}: {label}", value, unit=unit, definition=f"{label} of the P&L per trade; the mean is the {definition}", n=int(r["n_trades"]), notes=f"{note}; no standard error of its own{flag}", **src)  # fmt: skip
        rec(f"{base}.share_negative", f"{what}: share of trades with a negative P&L", r["share_negative"], r["share_negative_se_hh"], unit="fraction of trades", n=int(r["n_trades"]),
            definition="share of the trades whose P&L is below zero",
            notes=f"se = {se_method(int(r['hh_lags']), bool(r['share_negative_se_fallback']))} on the indicator; distinguishable from one half: {'yes' if r['share_differs_from_half'] else 'no'}{flag}", **src)  # fmt: skip

    def windows_rec(base: str, what: str, r: dict[str, Any], n: int, src: dict[str, str], flag: str = "") -> None:  # fmt: skip
        """The tests of the windows of a row of the combined or of the price table, when it has
        them (``flag``: the words of :func:`model_note` for a row that rests on a model price)."""
        if not np.isfinite(r["windows_p_most_cautious"]):
            return
        rec(f"{base}.windows_p_most_cautious", f"{what}: largest two-sided p of the three tests that count the windows", r["windows_p_most_cautious"], unit="probability", n=n,
            definition="the largest of three two-sided p-values: the plain t of the entry-year means, that t with one lag between adjacent years, and the Hansen-Hodrick t of the mean read on years less one degrees of freedom",
            notes=f"plain t {r['year_means_t']:.3f} on {int(r['year_means_df'])} degrees of freedom (p {r['year_means_p']:.4f}); with one lag t {r['year_means_t_lag1']:.3f} (p {r['year_means_p_lag1']:.4f}); Hansen-Hodrick t p {r['hh_t_p_on_year_df']:.4f}; "
                  f"sign by the rule of the page: {r['sign_by_rule']}; grade of the sign: {r['sign_once_windows_counted'] or 'none, the rule names no sign'}; no standard error of its own{flag}", **src)  # fmt: skip

    priced_flagged = float(S["priced"]["flag_clip"].sum())
    for r in comb.to_dict("records"):
        src = study if r["estimator"] in ("copula_all_monthly",) else model
        base = f"combined.{r['structure']}.{r['estimator']}"
        what = f"gap {r['structure']}: {r['label']}"
        flag = model_note(r["price"], priced_flagged, len(S["priced"]), "priced trades it uses")
        rec(base, what, r["value_pct"], r["se_hh_pct"], unit="% of notional", n=int(r["n_trades_gap"]), definition="see Table 1e of GAP_12M.md: the gap at the copula's price on a group of monthly trades, or the combined estimate (copula gap of all monthly trades + mean price shift of the priced trades)",
            notes=f"se = {r['se_method']}; sign by the rule of the page: {r['sign_by_rule']}; trades in the mean price shift: {int(r['n_trades_price_shift'])}{flag}", **src)  # fmt: skip
        for suffix, label, value in (("ci95_lo", "lower", r["ci95_lo_pct"]), ("ci95_hi", "upper", r["ci95_hi_pct"]), ("t_ratio", "t-ratio", r["t_ratio"])):  # fmt: skip
            rec(f"{base}.{suffix}", f"{what}: {'95 % block-bootstrap interval, ' + label if suffix != 't_ratio' else label}", value, unit="ratio" if suffix == "t_ratio" else "% of notional", n=int(r["n_trades_gap"]),
                definition="a percentile of the block bootstrap of the estimator, or the estimator over its standard error", notes=f"no standard error of its own{flag}", **src)  # fmt: skip
        if r["estimator"] not in ("copula_all_monthly", "copula_priced") and not r[
            "estimator"
        ].startswith("priced_"):
            # the copied rows have theirs
            windows_rec(base, what, r, int(r["n_trades_gap"]), src, flag)
    for r in shifts.to_dict("records"):
        # each difference holds a model price
        flag = model_note("lc", r["n_trades_clip_flagged"], int(r["n_trades"]))
        rec(f"price.{r['difference']}.{r['sample']}", f"{r['quantity']}, mean over trades, {r['sample_label']}", r["mean_pct"], r["se_hh_pct"], unit="% of notional", n=int(r["n_trades"]),
            definition="mean over trades of the difference of the two entry prices of the forward, % of notional",
            notes=f"se = {r['se_method']}, across entry dates, lags in calendar months{boot_note(r['se_boot_pct'])}; Monte Carlo bound of the mean {r['mc_bound_pct']:.4f} (mean of the sums of the two per-date standard errors); "
                  f"sign by the rule of the page: {r['sign_by_rule']}{flag}", **model)  # fmt: skip
        windows_rec(
            f"price.{r['difference']}.{r['sample']}",
            f"{r['quantity']}, mean over trades, {r['sample_label']}",
            r,
            int(r["n_trades"]),
            model,
            flag,
        )
        rec(f"price.{r['difference']}.{r['sample']}.pooled_ratio", f"{r['pooled_ratio_label']}, {r['sample_label']}", r["pooled_ratio"], unit="ratio", n=int(r["n_trades"]),
            definition="sum over trades of one price over the sum of the other", notes=f"no standard error; per date the ratio runs from {r['ratio_min']:.4f} ({r['ratio_min_date']}) to {r['ratio_max']:.4f} ({r['ratio_max_date']}){flag}", **model)  # fmt: skip
    for r in payout.to_dict("records"):
        src = study if r["price"] == "copula" and r["sample"] in ("weekly", "monthly") else model
        base = f"payout.forward.{r['price']}.{r['sample']}"
        what = f"payout per 1 of premium of the forward at the {PRICE_LABEL[r['price']]} price, {r['sample_label']}"
        flag = model_note(r["price"], r["n_trades_clip_flagged"], int(r["n_trades"]))
        rec(base, what, r["payout_per_1_of_premium"], r["se_boot"], unit="ratio", n=int(r["n_trades"]), definition="sum over trades of the payoff D over the sum of the entry prices",
            notes=f"se = standard deviation of the {N_RESAMPLES} block-bootstrap ratios (blocks of {r['boot_block']} entries, seed {SEED}); Monte Carlo bound {r['mc_bound']:.4f}{flag}", **src)  # fmt: skip
        for suffix, label, value in (("ci95_lo", "lower", r["ci95_lo"]), ("ci95_hi", "upper", r["ci95_hi"])):  # fmt: skip
            rec(f"{base}.{suffix}", f"{what}: 95 % interval, {label}", value, unit="ratio", n=int(r["n_trades"]), definition="percentile of the study's disp_stats.bootstrap_ratio",
                notes=f"a percentile of the block bootstrap; no standard error of its own{flag}", **src)  # fmt: skip
    return out


def write_text(path: Path, text: str) -> None:
    """One file of the output folder, atomically; a frozen file is refused."""
    pc.guard_frozen(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--rows-dir", type=Path, default=ROWS, help="the row files of the 12m pass")
    args = ap.parse_args()
    if LATER.resolve() not in OUT.resolve().parents:
        raise SystemExit(f"the output folder must be under {LATER}")
    markdown, records, tables, answer_text = build(args.rows_dir.resolve())
    for name, frame in tables.items():
        pc.save_table(frame, name, base=OUT)
    pc.write_part(PART, records, markdown, base=OUT)
    write_text(OUT / "GAP_12M.md", markdown.rstrip() + "\n")
    print(f"{OUT}: GAP_12M.md, parts/{PART}.json ({len(records)} records), {len(tables)} tables")
    print("\n\n".join(answer_text))


if __name__ == "__main__":
    main()
