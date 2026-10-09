"""The PM results package of 2026-10-09, the one-year addendum (``pm_update/1y``): the local
correlation model against its constant-correlation companion and the dispersion study's copula
at the 12-month tenor.

    python scripts/pm_1y.py [--tenor 12m] [--rows-dir <production rows>]
        [--development-rows-dir <development rows>] [--before-rows-dir <old defaults rows>]
        [--base <folder>] [--no-strict] [--no-status]

What it writes, under the base (``pm_update/1y`` until 13:00 New York time on 2026-10-09,
``pm_update/later/1y`` after: the owner's rule): ``NUMBERS_1Y.md``, ``numbers_1y.json`` (one
record per number, ``pm_common.record``), ``parts/oneyear.{json,md}`` and ``tables/1y_*.csv``.

Inputs.  The per-date rows of ``scripts/lcm_price.py`` at the tenor: the production budget on
the four reference dates (``rows/12m_production/<date>.json``, else the table
``lcm_12m.parquet``) and the development budget on the twenty dates
(``rows/12m_development/<date>.json``, else ``lcm_12m_dev.parquet``); the study's entries
``outputs/dispersion/entries_<tenor>.parquet``, basket B1 (read only).  The twenty dates are, for
each year 2007..2026, the first date of the year among the study's monthly entries of basket B1
that have their pickle (:func:`yearly_dates`).

A row is accepted when its date, tenor, budget and budget sizes are the ones asked, the columns
it repeats from the study's entry (``P_D_copula``, ``EV_copula``, ``EQV``, ``T``, the cash
strikes) are the entry's of that date and tenor, and, unless
``--no-strict`` (tests against rows of another specification, never into the package), it carries
the owner's decisions 1, 2 and 5: ``calendar_repair`` true and the columns
``n_names_unscreened`` and ``n_dropped_calendar_index``.  A row that is missing or rejected is
written as *pending* with the file it waits for; a row whose date failed as *failed* with its
reason; a column the row does not carry (the deltas of a row run with ``risk = none``) as *not
run*.  No number is estimated here: every value is a column of a row or of the study's table, or
a ratio of two of them.

Standard errors.  The row's own (antithetic pair means; its ratios paired on common paths) are
kept.  A ratio of a row's number to a number of the study's table (LC or CC over the copula)
gets the delta-method error of two independent estimates (``pm_common.ratio_se``).  The summaries
over dates carry the standard error of the mean across dates (sd/√n): the dispersion over the
dates, not a Monte Carlo error.  Quartiles are linearly interpolated (pandas' default).

Flags (owner's decision 5).  A date is flagged for the clipped mass when the mass clipped at
``λ = 0`` or at the cap exceeds 1 % inside ±2.5 sd (strictly); flagged rows stay in the tables
and are out of the first summary; the second summary is over every priced date.

Re-runnable: every output is rewritten atomically from the inputs present.
"""

# ruff: noqa: RUF001, E501 — report prose: typographic signs and long caption lines
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_common as pc

log = logging.getLogger("pm_1y")

NY = ZoneInfo("America/New_York")
#: the owner's deadline: what is written later goes to ``pm_update/later/1y``
CUTOFF = dt.datetime(2026, 10, 9, 13, 0, tzinfo=NY)
SECTION = "1Y"
PART = "oneyear"
YEARS = tuple(range(2007, 2027))
CALL_TAGS = ("075", "100", "125", "150")
CALL_MULT = {"075": "0.75", "100": "1", "125": "1.25", "150": "1.5"}
#: the clipped mass above which a date is flagged (owner's decision 5); the constant of
#: ``scripts/lcm_price.py`` (``CLIP_FLAG_MASS``), which is not imported here: it loads the pricer
CLIP_FLAG_MASS = 0.01
CONFIG = pc.ROOT / "configs" / "studies" / "dispersion" / "lcm.yaml"
SWEEP_3M = pc.LC_OUT / "lcm_3m_dev_repair.parquet"
DECISION_COLUMNS = ("n_names_unscreened", "n_dropped_calendar_index")
#: the columns a priced row repeats from the study's entry (row column, entry column)
ENTRY_COLUMNS = (
    ("P_D_copula", "P_D"),
    ("EV_copula", "EV"),
    ("EQV", "EQV"),
    ("T", "T"),
    *((f"K_{t}", f"K_{t}") for t in CALL_TAGS),
)
ENTRY_TOLERANCE = 1e-9
STUDY_NOTE = "the dispersion study's table: no commit column"


# ---------------------------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------------------------


def study_entries(tenor: str) -> pd.DataFrame:
    """The study's entries of basket B1 at ``tenor``, indexed by date.  One row per date is
    required (the table holds one row per date and basket): a duplicate raises."""
    path = pc.STUDY / f"entries_{tenor}.parquet"
    frame = pd.read_parquet(path)
    b1 = frame[frame["basket"] == "B1"].copy()
    if b1["date"].duplicated().any():
        raise ValueError(f"{path}: basket B1 has several rows on a date")
    return b1.set_index("date", drop=False).sort_index()


def yearly_dates(entries: pd.DataFrame, tenor: str) -> pd.DataFrame:
    """For each year of ``YEARS``, the first date among the monthly entries of basket B1 that
    have their pickle (``outputs/dispersion/entries/<tenor>/<date>.pkl``), with whether it is a
    date of the 3-month sweep (``lcm_3m_dev_repair.parquet``).  A year without one has an empty
    date."""
    monthly = entries[entries["monthly"].astype(bool)]
    sweep: set[str] = set()
    if SWEEP_3M.exists():
        sweep = set(pd.read_parquet(SWEEP_3M, columns=["date"])["date"].astype(str))
    rows = []
    for year in YEARS:
        dates = [d for d in monthly.index if d.startswith(f"{year}-")]
        with_pickle = [d for d in dates if (pc.STUDY / "entries" / tenor / f"{d}.pkl").exists()]
        date = min(with_pickle) if with_pickle else ""
        rows.append(
            {
                "year": year,
                "date": date,
                "monthly_entries_in_year": len(dates),
                "with_pickle": len(with_pickle),
                "in_3m_sweep": bool(date in sweep) if date else False,
                "T": float(entries.loc[date, "T"]) if date else float("nan"),
                "expiry": entries.loc[date, "expiry"] if date else None,
            }
        )
    return pd.DataFrame(rows)


def budget_sizes() -> dict[str, dict[str, int]]:
    """The sizes of the budgets, from the runs' configuration."""
    return {k: dict(v) for k, v in yaml.safe_load(CONFIG.read_text())["budgets"].items()}


@dataclass
class Slot:
    """One date at one budget: the row (when there is one that is accepted), its state
    (``ok`` — priced, status ok or check; ``failed``; ``pending``) and the study's entry."""

    group: str  # "prod" | "dev"
    date: str
    budget: str
    entry: dict[str, Any] | None
    row: dict[str, Any] | None = None
    state: str = "pending"
    reason: str = ""
    source: str = ""
    waits_for: str = ""
    written: str = ""
    problems: list[str] = field(default_factory=list)

    @property
    def priced(self) -> bool:
        return self.state == "ok"

    @property
    def commit(self) -> str:
        return str(self.row.get("git_commit", "")) if self.row else ""


def _number(d: dict[str, Any] | None, key: str) -> float:
    """``d[key]`` as a float (NaN for ``None``); ``KeyError`` when the column is absent."""
    if d is None:
        raise KeyError(key)
    value = d[key]
    if value is None:
        return float("nan")
    return float(value)


def read_rows_table(path: Path) -> dict[str, dict[str, Any]]:
    """The rows of a sweep's table by date (empty when the table is not there)."""
    if not path.exists():
        return {}
    frame = pd.read_parquet(path)
    out = {}
    for rec in frame.to_dict(orient="records"):
        # a table gives a failed row every column, as NaN: drop them so that it reads as the JSON
        out[str(rec["date"])] = {
            k: v for k, v in rec.items() if not (isinstance(v, float) and math.isnan(v))
        }
    return out


def validate(
    row: dict[str, Any],
    date: str,
    tenor: str,
    budget: str,
    sizes: dict[str, dict[str, int]],
    strict: bool,
    entry: dict[str, Any] | None = None,
) -> list[str]:
    """Why a row is not the one asked for (module docstring); empty when it is.  A priced row
    repeats columns of the study's entry (``ENTRY_COLUMNS``): one that differs from the entry of
    the date and tenor by more than ``ENTRY_TOLERANCE`` (relative) is not a row of that entry."""
    problems = []
    if str(row.get("date")) != date:
        problems.append(f"date {row.get('date')!r}")
    if row.get("tenor") != tenor:
        problems.append(f"tenor {row.get('tenor')!r}, not {tenor}")
    if row.get("budget") != budget:
        problems.append(f"budget {row.get('budget')!r}, not {budget}")
    if row.get("status") == "failed":
        return problems
    for key, want in sizes[budget].items():
        if row.get(key) is None or int(row[key]) != int(want):
            problems.append(f"{key} {row.get(key)!r}, not {want}")
    for row_key, entry_key in ENTRY_COLUMNS if entry is not None else ():
        if row.get(row_key) is None:
            problems.append(f"no column {row_key}")
            continue
        a, b = float(row[row_key]), float(entry[entry_key])
        if not abs(a - b) <= ENTRY_TOLERANCE * max(1.0, abs(b)):
            problems.append(f"{row_key} {a:.6g} is not the entry's {entry_key} {b:.6g}")
    if strict:
        if row.get("calendar_repair") is not True:
            problems.append("calendar_repair is not true (decisions 1 and 5)")
        for key in DECISION_COLUMNS:
            if key not in row:
                problems.append(f"no column {key} (decisions 2 and 5: a row before d4faa74)")
    return problems


def load_slot(
    group: str,
    date: str,
    budget: str,
    tenor: str,
    rows_dir: Path,
    table: dict[str, dict[str, Any]],
    table_path: Path,
    entries: pd.DataFrame,
    sizes: dict[str, dict[str, int]],
    strict: bool,
) -> Slot:
    entry = entries.loc[date].to_dict() if date in entries.index else None
    path = rows_dir / f"{date}.json"
    slot = Slot(group, date, budget, entry, waits_for=str(path))
    if path.exists():
        row = json.loads(path.read_text())
        slot.source = str(path)
        slot.written = dt.datetime.fromtimestamp(path.stat().st_mtime, NY).strftime("%m-%d %H:%M")
    elif date in table:
        row = table[date]
        slot.source = f"{table_path} (date {date})"
        slot.written = dt.datetime.fromtimestamp(table_path.stat().st_mtime, NY).strftime(
            "%m-%d %H:%M"
        )
    else:
        slot.reason = f"pending: waits for {path}"
        return slot
    slot.problems = validate(row, date, tenor, budget, sizes, strict, entry)
    if slot.problems:
        slot.reason = (
            f"pending: the row at {slot.source} is not the one asked for "
            f"({'; '.join(slot.problems)}); waits for {path}"
        )
        slot.source = ""
        log.warning("%s %s %s: row rejected — %s", group, date, budget, "; ".join(slot.problems))
        return slot
    slot.row = row
    if row.get("status") == "failed":
        slot.state = "failed"
        slot.reason = f"failed: {row.get('reason', '')}"
    else:
        slot.state = "ok"
        slot.reason = str(row.get("reason", "") or "")
    return slot


# ---------------------------------------------------------------------------------------------
# the quantities
# ---------------------------------------------------------------------------------------------

Getter = Callable[[dict[str, Any] | None, dict[str, Any] | None], tuple[float, float | None]]


@dataclass(frozen=True)
class Quantity:
    """One reported quantity: where it comes from (``origin``: ``row``, ``study`` or ``both``),
    how its value and standard error are read, and how it is shown."""

    key: str
    label: str
    definition: str
    get: Getter
    origin: str = "row"
    unit: str = ""
    digits: int = 4
    kind: str = "float"  # "float" | "int" | "bool"
    no_se: str = ""  # why the number has no standard error
    summarise: bool = True


def _row(key: str, se: str | None = None) -> Getter:
    return lambda r, e: (_number(r, key), _number(r, se) if se else None)


def _study(key: str, se: str | None = None) -> Getter:
    return lambda r, e: (_number(e, key), _number(e, se) if se else None)


def _over_study(num: str, num_se: str, den: str, den_se: str) -> Getter:
    """A row's number over a number of the study's entry: two independent estimates."""

    def get(r: dict[str, Any] | None, e: dict[str, Any] | None) -> tuple[float, float | None]:
        a, a_se, b, b_se = _number(r, num), _number(r, num_se), _number(e, den), _number(e, den_se)
        if not (b and a):
            return float("nan"), float("nan")
        return a / b, pc.ratio_se(a, a_se, b, b_se)

    return get


def _over(num: str, num_se: str, den: str) -> Getter:
    """A row's number over a listed value of the row (``EQV``, ``M_B_listed``: no Monte Carlo
    error in the denominator)."""

    def get(r: dict[str, Any] | None, e: dict[str, Any] | None) -> tuple[float, float | None]:
        d = _number(r, den)
        return _number(r, num) / d, _number(r, num_se) / abs(d)

    return get


def _listed_ratio(r: dict[str, Any] | None, e: dict[str, Any] | None) -> tuple[float, None]:
    return math.sqrt(_number(e, "EQV") / _number(e, "EV")), None


def _listed_level(r: dict[str, Any] | None, e: dict[str, Any] | None) -> tuple[float, None]:
    return _number(e, "P_D") * math.sqrt(_number(e, "EQV") / _number(e, "EV")), None


def _ev_over_eqv_study(r: dict[str, Any] | None, e: dict[str, Any] | None) -> tuple[float, None]:
    return _number(e, "EV") / _number(e, "EQV"), None


def _flag(key: str) -> Getter:
    return lambda r, e: (float(_number(r, key) > CLIP_FLAG_MASS), None)


def _truth(key: str) -> Getter:
    def get(r: dict[str, Any] | None, e: dict[str, Any] | None) -> tuple[float, None]:
        if r is None or key not in r:
            raise KeyError(key)
        return (float("nan") if r[key] is None else float(bool(r[key]))), None

    return get


NO_SE_STUDY = "read from the study's table, which carries no standard error for it"
NO_SE_COUNT = "a count of the inputs, not a Monte Carlo estimate"
NO_SE_CLIP = "a share of the calibration's particles (a diagnostic of the calibration): no standard error is computed for it"

FORWARD = [
    Quantity("ED_lc", "E_LC[D]", "the Palladium forward E[Σ w_i |R_i − R̄|] under the calibrated local correlation model", _row("ED_lc", "ED_lc_se"), digits=6),
    Quantity("ED_cc", "E_CC[D]", "the Palladium forward E[Σ w_i |R_i − R̄|] under the constant-correlation companion (same local vols, the constant correlation that reprices the index at-the-money straddle)", _row("ED_cc", "ED_cc_se"), digits=6),
    Quantity("P_D", "copula P_D", "the copula's price of the forward in the study's entry, basket B1", _study("P_D", "P_D_se"), origin="study", digits=6),
    Quantity("lc_over_cc", "LC/CC", "E_LC[D]/E_CC[D], paired on common paths (the row's ratio and its error)", _row("ratio", "ratio_se"), digits=5),
    Quantity("lc_over_copula", "LC/copula", "E_LC[D]/P_D; delta-method error of two independent estimates (ED_lc_se and the entry's P_D_se)", _over_study("ED_lc", "ED_lc_se", "P_D", "P_D_se"), origin="both", digits=5),
    Quantity("cc_over_copula", "CC/copula", "E_CC[D]/P_D; delta-method error of two independent estimates (ED_cc_se and the entry's P_D_se)", _over_study("ED_cc", "ED_cc_se", "P_D", "P_D_se"), origin="both", digits=5),
]  # fmt: skip
LISTED = [
    Quantity("listed_fwd_ratio", "√(EQV/EV)", "the listed-variance forward over P_D: √(EQV/EV), EQV the listed-option value of E[V] and EV the copula's E[V] of the entry (= κ_cop·√EQV/P_D)", _listed_ratio, origin="study", digits=5, no_se=NO_SE_STUDY),
    Quantity("listed_fwd", "P_D·√(EQV/EV)", "the listed-variance forward κ_cop·√EQV = P_D·√(EQV/EV) of the entry", _listed_level, origin="study", digits=6, no_se=NO_SE_STUDY),
    Quantity("ED_wing", "ED_wing", "κ_LC·√(Σ w E_LC[R_i²] − M_B^listed)", _row("ED_wing", "ED_wing_se"), digits=6),
    Quantity("ED_wing_over_cc", "ED_wing/E_CC[D]", "κ_LC·√(Σ w E_LC[R_i²] − M_B^listed) over E_CC[D] (the row's ED_wing_ratio and its delta-method error)", _row("ED_wing_ratio", "ED_wing_ratio_se"), digits=5),
    Quantity("ED_wing_over_copula", "ED_wing/copula", "κ_LC·√(Σ w E_LC[R_i²] − M_B^listed) over the copula's P_D; delta-method error of two independent estimates (ED_wing_se and the entry's P_D_se)", _over_study("ED_wing", "ED_wing_se", "P_D", "P_D_se"), origin="both", digits=5),
    Quantity("ED_eqv", "ED_eqv", "κ_LC·√EQV", _row("ED_eqv", "ED_eqv_se"), digits=6),
    Quantity("ED_eqv_over_cc", "ED_eqv/E_CC[D]", "κ_LC·√EQV over E_CC[D] (the row's ED_eqv_ratio and its delta-method error)", _row("ED_eqv_ratio", "ED_eqv_ratio_se"), digits=5),
    Quantity("ED_eqv_over_copula", "ED_eqv/copula", "κ_LC·√EQV over the copula's P_D; delta-method error of two independent estimates (ED_eqv_se and the entry's P_D_se)", _over_study("ED_eqv", "ED_eqv_se", "P_D", "P_D_se"), origin="both", digits=5),
]  # fmt: skip
KAPPA = [
    Quantity("kappa_lc", "κ_LC", "E_LC[D]/√E_LC[V] (the row's delta-method error)", _row("kappa_lc", "kappa_lc_se")),
    Quantity("kappa_cc", "κ_CC", "E_CC[D]/√E_CC[V] (the row's delta-method error)", _row("kappa_cc", "kappa_cc_se")),
    Quantity("kappa_cop", "κ_cop", "P_D/√EV of the study's entry", _study("kappa_cop"), origin="study", no_se=NO_SE_STUDY),
    Quantity("EV_over_EQV_lc", "E_LC[V]/EQV", "E_LC[V] over the listed-option value EQV (error: EV_lc_se/EQV)", _row("EV_over_EQV", "EV_over_EQV_se")),
    Quantity("EV_over_EQV_cc", "E_CC[V]/EQV", "E_CC[V] over EQV (error: EV_cc_se/EQV)", _over("EV_cc", "EV_cc_se", "EQV")),
    Quantity("EV_over_EQV_cop", "E_cop[V]/EQV", "the copula's EV over EQV, both of the study's entry", _ev_over_eqv_study, origin="study", no_se=NO_SE_STUDY),
]  # fmt: skip
SPLIT = [
    Quantity("EV_single_part", "single-name part", "Σ w E_LC[R_i²] − Σ w M_i^listed: the single-name part of E_LC[V] − EQV (E_LC[V] − EQV = single-name part − basket part)", _row("EV_single_part", "EV_single_part_se"), digits=6),
    Quantity("EV_basket_part", "basket part", "E_LC[R̄²] − M_B^listed: the basket part of E_LC[V] − EQV (E_LC[V] − EQV = single-name part − basket part)", _row("EV_basket_part", "EV_basket_part_se"), digits=6),
    Quantity("EV_single_part_over_EQV", "single-name part/EQV", "(Σ w E_LC[R_i²] − Σ w M_i^listed)/EQV (error: EV_single_part_se/EQV); E_LC[V]/EQV − 1 = single-name part/EQV − basket part/EQV", _over("EV_single_part", "EV_single_part_se", "EQV")),
    Quantity("EV_basket_part_over_EQV", "basket part/EQV", "(E_LC[R̄²] − M_B^listed)/EQV (error: EV_basket_part_se/EQV); E_LC[V]/EQV − 1 = single-name part/EQV − basket part/EQV", _over("EV_basket_part", "EV_basket_part_se", "EQV")),
    Quantity("EV_basket_part_over_MB", "basket part/M_B^listed", "(E_LC[R̄²] − M_B^listed)/M_B^listed (error: EV_basket_part_se/M_B^listed)", _over("EV_basket_part", "EV_basket_part_se", "M_B_listed")),
]  # fmt: skip


def call_quantities(tag: str) -> list[Quantity]:
    m = CALL_MULT[tag]
    call = f"the call E[(D − K)⁺] at the study's cash strike K_{tag} = {m} × P_D"
    return [
        Quantity(f"K_{tag}", f"K_{tag}", f"the study's cash strike {m} × P_D of the entry", _study(f"K_{tag}"), origin="study", digits=6, no_se="a strike, not an estimate", summarise=False),
        Quantity(f"C_{tag}_lc", f"C_{tag} LC", f"{call}, under LC", _row(f"C_{tag}_lc", f"C_{tag}_lc_se"), digits=6),
        Quantity(f"C_{tag}_cc", f"C_{tag} CC", f"{call}, under CC", _row(f"C_{tag}_cc", f"C_{tag}_cc_se"), digits=6),
        Quantity(f"C_{tag}_cop", f"C_{tag} copula", f"{call}, the copula's of the study's entry", _study(f"C_{tag}", f"C_se_{tag}"), origin="study", digits=6),
        Quantity(f"C_{tag}_lc_over_copula", f"C_{tag} LC/copula", f"{call}: LC over the copula's; delta-method error of two independent estimates", _over_study(f"C_{tag}_lc", f"C_{tag}_lc_se", f"C_{tag}", f"C_se_{tag}"), origin="both"),
        Quantity(f"C_{tag}_lc_over_cc", f"C_{tag} LC/CC", f"{call}: LC over CC, paired on common paths (the row's ratio and its error)", _row(f"C_{tag}_ratio", f"C_{tag}_ratio_se")),
    ]  # fmt: skip


CALLS = {tag: call_quantities(tag) for tag in CALL_TAGS}
CLIP = [
    Quantity("clip_low_inner_max", "clipped at λ = 0", "the share of the particles that are inside ±2.5 at-the-money sd of the basket and whose λ is clipped at 0, the largest over the calibration's slices (clip_low_inner_max)", _row("clip_low_inner_max"), unit="fraction", no_se=NO_SE_CLIP),
    Quantity("clip_high_inner_max", "clipped at the cap", "the share of the particles that are inside ±2.5 at-the-money sd of the basket and whose λ is clipped at the cap, the largest over the calibration's slices (clip_high_inner_max)", _row("clip_high_inner_max"), unit="fraction", no_se=NO_SE_CLIP),
    Quantity("clip_inner_max", "clipped, larger side", "the larger of the two shares of the particles inside ±2.5 at-the-money sd of the basket whose λ is clipped, at 0 or at the cap, each the largest over the calibration's slices (clip_inner_max)", _row("clip_inner_max"), unit="fraction", no_se=NO_SE_CLIP),
]  # fmt: skip
INDEX = [
    Quantity("idx_err_atm", "index error ATM", "the model's implied vol of the index at the horizon minus its target, at the money", _row("idx_err_atm", "idx_err_atm_se"), unit="vol points", digits=3),
    Quantity("idx_err_90", "90 %", "the model's implied vol of the index at the horizon minus its target, at the 90 % strike", _row("idx_err_90", "idx_err_90_se"), unit="vol points", digits=3),
    Quantity("idx_err_m15", "−1.5 sd", "the model's implied vol of the index at the horizon minus its target, at −1.5 sd", _row("idx_err_m15", "idx_err_m15_se"), unit="vol points", digits=3),
    Quantity("idx_err_m20", "−2 sd", "the model's implied vol of the index at the horizon minus its target, at −2 sd", _row("idx_err_m20", "idx_err_m20_se"), unit="vol points", digits=3),
    Quantity("idx_err_m25", "−2.5 sd", "the model's implied vol of the index at the horizon minus its target, at −2.5 sd", _row("idx_err_m25", "idx_err_m25_se"), unit="vol points", digits=3),
    Quantity("n_names_extrapolated", "names beyond last expiry", "the number of names priced beyond their last listed expiry (the horizon is after the name's last kept slice)", _row("n_names_extrapolated"), unit="names", kind="int", no_se=NO_SE_COUNT),
]  # fmt: skip
DELTAS = [
    Quantity("delta_fwd_lc", "Δ forward LC", "sticky-strike common delta of the forward under LC: percent of the model's price per +1 % of every spot", _row("delta_fwd_lc", "delta_fwd_lc_se"), unit="% per %"),
    Quantity("delta_fwd_cc", "Δ forward CC", "sticky-strike common delta of the forward under CC: percent of the model's price per +1 % of every spot", _row("delta_fwd_cc", "delta_fwd_cc_se"), unit="% per %"),
    Quantity("delta_C100_lc", "Δ call K_100 LC", "sticky-strike common delta of the call at K_100 under LC, percent of its price per +1 % of every spot", _row("delta_C100_lc", "delta_C100_lc_se"), unit="% per %", digits=3),
    Quantity("delta_C100_cc", "Δ call K_100 CC", "sticky-strike common delta of the call at K_100 under CC, percent of its price per +1 % of every spot", _row("delta_C100_cc", "delta_C100_cc_se"), unit="% per %", digits=3),
    Quantity("delta_homogeneity", "homogeneity", "the homogeneity term of the forward's sticky-strike common delta (a common move at sticky moneyness scales D): exactly 1", _row("delta_homogeneity"), unit="% per %", no_se="exact by homogeneity"),
    Quantity("delta_skew_channel", "skew channel", "the skew channel of the forward's sticky-strike common delta: Δ^CC_ss − 1", _row("delta_skew_channel", "delta_skew_channel_se"), unit="% per %"),
    Quantity("delta_correlation_channel", "correlation channel", "the correlation channel of the forward's sticky-strike common delta: Δ^LC_ss − Δ^CC_ss, paired on common paths", _row("delta_correlation_channel", "delta_correlation_channel_se"), unit="% per %"),
]  # fmt: skip
FLAGS = [
    Quantity("index_extrapolated", "DJX target extrapolated", "true when the last listed DJX expiry kept is before the horizon", _truth("index_extrapolated"), kind="bool", unit="1 = yes", no_se=NO_SE_COUNT, summarise=False),
    Quantity("index_last_slice", "last DJX slice", "the last DJX slice kept, in years", _row("index_last_slice"), unit="years", no_se=NO_SE_COUNT, summarise=False),
    Quantity("n_dropped_calendar_index", "DJX slices dropped by the repair", "the number of DJX expiries the calendar repair dropped (decision 5); the target is repaired when it is above 0", _row("n_dropped_calendar_index"), unit="expiries", kind="int", no_se=NO_SE_COUNT, summarise=False),
    Quantity("n_dropped_index", "DJX slices dropped, any rule", "the number of DJX expiries dropped by any rule of the screen", _row("n_dropped_index"), unit="expiries", kind="int", no_se=NO_SE_COUNT, summarise=False),
    Quantity("n_names_unscreened", "names kept unscreened", "the number of names the quote screen emptied, kept on their unscreened expiries (decision 2)", _row("n_names_unscreened"), unit="names", kind="int", no_se=NO_SE_COUNT, summarise=False),
    Quantity("flag_clip_low", "clip at λ = 0 > 1 %", "true when the share clipped at λ = 0 inside ±2.5 sd exceeds 1 %", _flag("clip_low_inner_max"), kind="bool", unit="1 = yes", no_se=NO_SE_COUNT, summarise=False),
    Quantity("flag_clip_high", "clip at the cap > 1 %", "true when the share clipped at the cap inside ±2.5 sd exceeds 1 %", _flag("clip_high_inner_max"), kind="bool", unit="1 = yes", no_se=NO_SE_COUNT, summarise=False),
]  # fmt: skip
GROUPS: dict[str, list[Quantity]] = {
    "forward": FORWARD,
    "listed": LISTED,
    "kappa": KAPPA,
    "split": SPLIT,
    **{f"calls_{tag}": CALLS[tag] for tag in CALL_TAGS},
    "clip": CLIP,
    "index": INDEX,
    "deltas": DELTAS,
    "flags": FLAGS,
}
ALL = [q for group in GROUPS.values() for q in group]


@dataclass(frozen=True)
class Cell:
    value: float | None
    se: float | None
    state: str  # "value" | "pending" | "failed" | "absent"
    note: str = ""


def evaluate(q: Quantity, slot: Slot) -> Cell:
    """The quantity on a date: its value and error, or why there is none."""
    if q.origin != "study":
        if slot.state == "pending":
            return Cell(None, None, "pending", slot.reason)
        if slot.state == "failed":
            return Cell(None, None, "failed", slot.reason)
    if q.origin != "row" and slot.entry is None:
        return Cell(None, None, "absent", "no entry of basket B1 on the date in the study's table")
    try:
        value, se = q.get(slot.row, slot.entry)
    except KeyError as exc:
        risk = slot.row.get("risk") if slot.row else None
        return Cell(
            None,
            None,
            "absent",
            f"not run: the row has no column {exc.args[0]} (risk = {risk})",
        )
    if value is None or not math.isfinite(value):
        return Cell(None, None, "absent", "not finite in the source")
    if se is not None and not math.isfinite(se):
        se = None
    return Cell(float(value), se, "value")


def show(q: Quantity, cell: Cell) -> str:
    if cell.state == "pending":
        return "pending"
    if cell.state == "failed":
        return "failed"
    if cell.state == "absent":
        return "not run" if cell.note.startswith("not run") else "n/a"
    assert cell.value is not None
    if q.kind == "int":
        return str(round(cell.value))
    if q.kind == "bool":
        return "yes" if cell.value else "no"
    return pc.pm(cell.value, cell.se, q.digits)


def clip_flagged(slot: Slot) -> bool | None:
    """Flagged for the clipped mass (``None`` when the date is not priced)."""
    if not slot.priced:
        return None
    low, high = _number(slot.row, "clip_low_inner_max"), _number(slot.row, "clip_high_inner_max")
    return bool(low > CLIP_FLAG_MASS or high > CLIP_FLAG_MASS)


# ---------------------------------------------------------------------------------------------
# records, tables, Markdown
# ---------------------------------------------------------------------------------------------


class Builder:
    def __init__(self, tenor: str, entries_path: Path) -> None:
        self.tenor = tenor
        self.entries_path = entries_path
        self.records: list[dict[str, Any]] = []
        self.md: list[str] = []

    # --- records
    def provenance(self, q: Quantity, slot: Slot) -> tuple[str, str, str]:
        """Budget, commit and source of a quantity on a date."""
        study = f"{self.entries_path} (basket B1)"
        if q.origin == "study":
            return pc.BUDGETS["study"], "", study
        budget = pc.BUDGETS.get(slot.budget, slot.budget)
        source = slot.source or slot.waits_for
        if q.origin == "both":
            return (
                f"{budget}; denominator: {pc.BUDGETS['study']}",
                slot.commit,
                f"{source}; {study}",
            )
        return budget, slot.commit, source

    def add(self, q: Quantity, slot: Slot) -> Cell:
        cell = evaluate(q, slot)
        budget, commit, source = self.provenance(q, slot)
        notes = [cell.note] if cell.note else []
        if cell.state == "value" and cell.se is None:
            notes.append(f"no standard error: {q.no_se or 'none in the source'}")
        if q.origin == "study":
            notes.append(STUDY_NOTE)
        if slot.priced and q.origin != "study" and slot.row and slot.row.get("status") != "ok":
            notes.append(f"row status {slot.row.get('status')}: {slot.row.get('reason', '')}")
        self.records.append(
            pc.record(
                f"{SECTION}.{slot.group}.{slot.date}.{q.key}",
                SECTION,
                q.label,
                cell.value,
                cell.se,
                date=slot.date,
                tenor=self.tenor,
                unit=q.unit,
                definition=q.definition,
                budget=budget,
                commit=commit,
                source=source,
                notes="; ".join(notes),
            )
        )
        return cell

    # --- Markdown
    def table(self, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
        self.md.append("| " + " | ".join(headers) + " |")
        self.md.append("|" + "|".join(" --- " for _ in headers) + "|")
        self.md.extend("| " + " | ".join(str(c) for c in r) + " |" for r in rows)
        self.md.append("")

    def text(self, *lines: str) -> None:
        self.md.extend(lines)
        self.md.append("")


def footer(slots: Sequence[Slot], study: bool, entries_path: Path, rows_dir: Path | None) -> str:
    """Budget, commit and source of a table, and what it waits for."""
    parts = []
    if rows_dir is not None and slots:
        budget = slots[0].budget
        commits = sorted({s.commit for s in slots if s.row and s.commit})
        parts.append(f"Budget: {pc.BUDGETS[budget]}.")
        parts.append(f"Commit: {', '.join(commits) if commits else 'pending (no row yet)'}.")
        sources = sorted({str(Path(s.source).parent) if s.source.endswith(".json") else s.source.split(" (")[0] for s in slots if s.source})  # fmt: skip
        parts.append(
            f"Source: {', '.join(sources) if sources else str(rows_dir)} (one row per date)."
        )
        waiting = [s.date for s in slots if s.state == "pending"]
        if waiting:
            parts.append(f"Pending ({len(waiting)} of {len(slots)}): {', '.join(waiting)} — each waits for `{rows_dir}/<date>.json`.")  # fmt: skip
        failed = [s.date for s in slots if s.state == "failed"]
        if failed:
            parts.append(f"Failed: {', '.join(failed)} (reasons in section 4).")
    if study:
        parts.append(f"Copula columns: {pc.BUDGETS['study']}, `{entries_path}`, basket B1.")
    return " ".join(parts)


DEFINITIONS = {
    "forward": "E_LC[D], E_CC[D]: the Palladium forward E[Σ w_i |R_i − R̄|] under the calibrated local correlation model and under its constant-correlation companion; P_D: the copula's price in the study's entry (basket B1); LC/CC is paired on common paths, LC/copula = E_LC[D]/P_D and CC/copula = E_CC[D]/P_D carry the delta-method error of two independent estimates.",
    "listed": "√(EQV/EV): the listed-variance forward over P_D, with EQV the listed-option value of E[V] and EV the copula's E[V] of the entry (no standard error in the study's table), and its level P_D·√(EQV/EV) = κ_cop·√EQV; ED_wing = κ_LC·√(Σ w E_LC[R_i²] − M_B^listed) and ED_eqv = κ_LC·√EQV, each also over E_CC[D] (the row's delta-method error) and over the copula's P_D (delta-method error of two independent estimates).",
    "kappa": "κ = E[D]/√E[V] under each model (κ_cop = P_D/√EV of the entry); E[V]/EQV under LC (EV_lc/EQV), CC (EV_cc/EQV) and the copula (EV/EQV of the entry), EQV the listed-option value of E[V].",
    "split": "The split of the LC figure: E_LC[V] − EQV = single-name part − basket part, with single-name part = Σ w E_LC[R_i²] − Σ w M_i^listed and basket part = E_LC[R̄²] − M_B^listed; each also over EQV (so that E_LC[V]/EQV − 1 = single-name part/EQV − basket part/EQV), and the basket part over M_B^listed.",
    "calls": "C = E[(D − K)⁺] at the study's cash strikes K_m = m × P_D under LC, CC and the copula (the entry's C_m ± C_se_m); LC/copula carries the delta-method error of two independent estimates, LC/CC is paired on common paths.",
    "calibration": "Clipped mass: the share of the particles that are inside ±2.5 at-the-money sd of the basket and whose λ is clipped at 0, resp. at the cap, the largest over the calibration's slices (fractions; a date is flagged above 0.01); index errors: the model's implied vol of the index at the horizon minus its target, in vol points; names beyond last expiry: the number of names priced beyond their last listed expiry.",
    "deltas": "Sticky-strike common deltas as elasticities (percent of each model's price per +1 % of every spot, central difference of ±1 %); the forward's LC delta = homogeneity (exactly 1) + skew channel (Δ^CC − 1) + correlation channel (Δ^LC − Δ^CC, paired).",
    "flags": "Row written: the time of the row's file, New York (month-day hour:minute); names beyond last expiry: n_names_extrapolated > 0; DJX target extrapolated: the last DJX slice kept is before the horizon; DJX target repaired: the calendar repair dropped at least one DJX expiry (n_dropped_calendar_index > 0); clip flags: the share clipped at λ = 0, resp. at the cap, exceeds 1 % inside ±2.5 sd (strictly).",
}


def by_quantity(b: Builder, slots: Sequence[Slot], quantities: Sequence[Quantity]) -> None:
    """A table with the quantities as rows and the dates as columns (the four dates)."""
    rows = []
    for q in quantities:
        rows.append([q.label, *(show(q, b.add(q, s)) for s in slots)])
    b.table(["quantity", *(s.date for s in slots)], rows)


def by_date(b: Builder, slots: Sequence[Slot], quantities: Sequence[Quantity]) -> None:
    """A table with the dates as rows and the quantities as columns (the twenty dates)."""
    rows = []
    for s in slots:
        rows.append([s.date, *(show(q, b.add(q, s)) for q in quantities)])
    b.table(["date", *(q.label for q in quantities)], rows)


def frame_of(slots: Sequence[Slot], tenor: str) -> pd.DataFrame:
    """The CSV behind the tables of a set of dates: one row per date, every quantity with its
    standard error, and the row's provenance."""
    out = []
    for s in slots:
        rec: dict[str, Any] = {
            "date": s.date, "tenor": tenor, "budget": s.budget, "state": s.state,
            "status": s.row.get("status") if s.row else None, "reason": s.reason,
            "git_commit": s.commit, "source": s.source or None, "row_written": s.written or None,
            "flag_clip": clip_flagged(s),
        }  # fmt: skip
        for q in ALL:
            cell = evaluate(q, s)
            rec[q.key] = cell.value
            if not q.no_se:
                rec[f"{q.key}_se"] = cell.se
        if s.row:
            for key in ("n_particles", "n_paths", "companion_paths", "T", "rho_cc", "lambda_c", "names_unscreened", "n_dropped", "n_dropped_calendar", "clip_low_max", "clip_high_max", "risk"):  # fmt: skip
                rec[key] = s.row.get(key)
        out.append(rec)
    return pd.DataFrame(out)


STATS: tuple[tuple[str, str], ...] = (
    ("mean", "mean"),
    ("q25", "first quartile"),
    ("median", "median"),
    ("q75", "third quartile"),
)


def summarise(values: pd.Series) -> dict[str, float | None]:
    """Mean with the standard error of the mean across dates, median and quartiles."""
    x = values.dropna().astype(float)
    n = len(x)
    if n == 0:
        return {"n": 0, "mean": None, "se": None, "q25": None, "median": None, "q75": None}
    return {
        "n": n,
        "mean": float(x.mean()),
        "se": float(x.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
        "q25": float(x.quantile(0.25)),
        "median": float(x.median()),
        "q75": float(x.quantile(0.75)),
    }


def summaries(
    b: Builder, slots: Sequence[Slot], frame: pd.DataFrame, rows_dir: Path, csv: Path
) -> pd.DataFrame:
    """Section 3's summaries over the dates not flagged for the clipped mass and over every
    priced date; returns the CSV behind the table."""
    priced = frame[frame["state"] == "ok"]
    samples = {
        "unflagged": ("the priced dates not flagged for the clipped mass", priced[priced["flag_clip"].eq(False)]),
        "all": ("every priced date", priced),
    }  # fmt: skip
    n_pending = sum(s.state == "pending" for s in slots)
    n_failed = sum(s.state == "failed" for s in slots)
    commits = ", ".join(sorted({s.commit for s in slots if s.row and s.commit}))
    partial = (
        f"partial: {len(priced)} of {len(slots)} dates priced so far ({n_pending} pending, {n_failed} failed)"
        if n_pending
        else ""
    )
    out, rows = [], []
    for q in ALL:
        if not q.summarise:
            continue
        line = [q.label]
        # the priced rows do not carry the column (the deltas of rows run with risk = none)
        not_run = any(evaluate(q, s).note.startswith("not run") for s in slots if s.priced)
        for sample, (label, sub) in samples.items():
            # a study-origin quantity is summarised on the same dates as the model's
            stats = summarise(sub[q.key]) if q.key in sub else summarise(pd.Series(dtype=float))
            n = int(stats["n"] or 0)
            out.append({"quantity": q.key, "label": q.label, "sample": sample, **stats})
            for stat, stat_label in STATS:
                notes = []
                if stats[stat] is None:
                    if not_run:
                        notes.append("not run: the priced rows do not carry the column")
                    elif n_pending:
                        notes.append(f"pending: waits for the development rows in {rows_dir}")
                    else:
                        notes.append("no date in the sample has this quantity")
                elif partial:
                    notes.append(partial)
                if stat != "mean":
                    notes.append(
                        "an order statistic over the dates (linear interpolation): no standard error"
                    )
                else:
                    notes.append("standard error of the mean across the dates (sd/√n), not a Monte Carlo error")  # fmt: skip
                b.records.append(
                    pc.record(
                        f"{SECTION}.dev.summary.{q.key}.{stat}.{sample}",
                        SECTION,
                        f"{q.label}: {stat_label} over {label}",
                        stats[stat],
                        stats["se"] if stat == "mean" else None,
                        tenor=b.tenor,
                        unit=q.unit,
                        definition=f"{stat_label} over {label} (yearly dates, first monthly entry of each year 2007–2026) of: {q.definition}",
                        budget=(
                            pc.BUDGETS["study"]
                            if q.origin == "study"
                            else pc.BUDGETS["development"]
                        ),
                        commit="" if q.origin == "study" else commits,
                        source=f"{csv} (from {rows_dir} and {b.entries_path})",
                        n=n,
                        notes="; ".join(notes),
                    )
                )
            if n == 0:
                line += ["0", "not run" if not_run else ("pending" if n_pending else "n/a"), "", "", ""]  # fmt: skip
            else:
                d = q.digits
                line += [
                    str(n),
                    pc.pm(stats["mean"], stats["se"], d),
                    *(pc.pm(stats[k], None, d) for k in ("q25", "median", "q75")),
                ]
        rows.append(line)
    heads = ["quantity"]
    for sample in ("not flagged", "all priced"):
        heads += [f"n ({sample})", "mean ± se", "q25", "median", "q75"]
    b.table(heads, rows)
    return pd.DataFrame(out)


def decision5(
    b: Builder, variants: Sequence[tuple[str, Slot | None, str]], date: str
) -> pd.DataFrame:
    """Section 1: the clipped mass by side and the DJX slices on the check's date, per variant
    (``(label, slot, text when there is no row)``)."""
    quantities = [*CLIP, FLAGS[5], FLAGS[6], FLAGS[2], FLAGS[3], FLAGS[1], FLAGS[0]]
    rows, out = [], []
    for q in quantities:
        line = [q.label]
        for label, slot, missing in variants:
            if slot is None:
                line.append(missing)
                continue
            cell = evaluate(q, slot)
            line.append(show(q, cell))
            out.append({"variant": label, "date": date, "quantity": q.key, "value": cell.value, "state": cell.state, "budget": slot.budget, "git_commit": slot.commit, "source": slot.source or slot.waits_for})  # fmt: skip
            if slot.group == "before":
                b.add(q, slot)
        rows.append(line)
    b.table(["quantity", *(v[0] for v in variants)], rows)
    return pd.DataFrame(out)


def flags_section(b: Builder, slots: Sequence[Slot]) -> pd.DataFrame:
    """Section 4: the flags of every date of both sets (a column the row does not carry reads
    ``n/a``: it can only happen under ``--no-strict``)."""

    def yes_no(flag: bool | None, detail: str = "") -> str:
        if flag is None:
            return "n/a"
        return (f"yes ({detail})" if detail else "yes") if flag else "no"

    rows, out = [], []
    for s in slots:
        r = s.row or {}
        rec: dict[str, Any] = {"set": s.group, "date": s.date, "budget": s.budget, "state": s.state, "status": r.get("status"), "reason": s.reason, "row_written": s.written or None, "git_commit": s.commit}  # fmt: skip
        if not s.priced:
            word = "pending" if s.state == "pending" else "failed"
            rows.append([s.budget, s.date, s.written or "—", word if s.state == "pending" else f"failed: {r.get('reason', '')}", *[word] * 7])  # fmt: skip
            out.append(rec)
            continue

        def val(key: str, r: dict[str, Any] = r) -> float | None:
            return float(r[key]) if r.get(key) is not None else None

        n_ext, n_rep, n_uns = val("n_names_extrapolated"), val("n_dropped_calendar_index"), val("n_names_unscreened")  # fmt: skip
        low, high = _number(r, "clip_low_inner_max"), _number(r, "clip_high_inner_max")
        idx_ext = None if r.get("index_extrapolated") is None else bool(r["index_extrapolated"])
        flagged = clip_flagged(s)
        rec.update(
            name_beyond_last_expiry=None if n_ext is None else n_ext > 0, n_names_extrapolated=n_ext,
            index_extrapolated=idx_ext, index_last_slice=r.get("index_last_slice"), T=r.get("T"),
            index_repaired=None if n_rep is None else n_rep > 0, n_dropped_calendar_index=n_rep,
            name_unscreened=None if n_uns is None else n_uns > 0, n_names_unscreened=n_uns,
            names_unscreened=r.get("names_unscreened", ""), clip_low_inner_max=low, clip_high_inner_max=high,
            flag_clip_low=low > CLIP_FLAG_MASS, flag_clip_high=high > CLIP_FLAG_MASS, flag_clip=flagged,
        )  # fmt: skip
        out.append(rec)
        status = str(r.get("status")) + (f" ({r.get('reason')})" if r.get("reason") else "")
        last = f"last slice {_number(r, 'index_last_slice'):.3f}y, T {_number(r, 'T'):.3f}y" if "index_last_slice" in r else ""  # fmt: skip
        rows.append(
            [
                s.budget,
                s.date,
                s.written,
                status,
                yes_no(rec["name_beyond_last_expiry"], f"{n_ext:.0f}" if n_ext else ""),
                yes_no(idx_ext, last),
                yes_no(rec["index_repaired"], f"{n_rep:.0f}" if n_rep else ""),
                yes_no(rec["name_unscreened"], str(r.get("names_unscreened", ""))),
                f"{'yes' if low > CLIP_FLAG_MASS else 'no'} ({low:.4f})",
                f"{'yes' if high > CLIP_FLAG_MASS else 'no'} ({high:.4f})",
                "yes" if flagged else "no",
            ]
        )
    b.table(
        ["budget", "date", "row written", "status", "a name beyond its last listed expiry", "DJX target extrapolated", "DJX target repaired", "a name kept unscreened", "clip at λ = 0 > 1 %", "clip at the cap > 1 %", "flagged (clip)"],
        rows,
    )  # fmt: skip
    return pd.DataFrame(out)


def consistency(b: Builder, slots: Sequence[Slot]) -> None:
    """The rows against the study's entries (the copula columns a row repeats) and against the
    flags a row carries: counts of disagreements, as records and one table."""
    pairs = ENTRY_COLUMNS
    priced = [s for s in slots if s.priced and s.entry is not None]
    rows = []
    for row_key, entry_key in pairs:
        diffs = [abs(_number(s.row, row_key) - _number(s.entry, entry_key)) for s in priced if s.row and row_key in s.row]  # fmt: skip
        worst = max(diffs) if diffs else None
        b.records.append(
            pc.record(
                f"{SECTION}.check.{row_key}.max_abs_diff", SECTION, f"rows against the study's entries: largest |row {row_key} − entry {entry_key}|",
                worst, None, tenor=b.tenor, definition=f"the largest absolute difference between the row's {row_key} and the entry's {entry_key} (basket B1) over the priced rows of both budgets",
                budget="no simulation: a comparison of stored columns", commit=", ".join(sorted({s.commit for s in priced if s.commit})), source=f"the rows of sections 2 and 3; {b.entries_path}",
                n=len(diffs), notes="no standard error: a difference of stored numbers" if diffs else "pending: no priced row yet",
            )
        )  # fmt: skip
        rows.append([f"row {row_key} − entry {entry_key}", str(len(diffs)), f"{worst:.3g}" if worst is not None else "pending"])  # fmt: skip
    bad = 0
    seen = 0
    for s in priced:
        assert s.row is not None
        if "flag_clip" in s.row:
            seen += 1
            bad += bool(s.row["flag_clip"]) != clip_flagged(s)
    b.records.append(
        pc.record(
            f"{SECTION}.check.flag_clip.disagreements", SECTION, "rows whose own flag_clip differs from the flag recomputed here", float(bad) if seen else None, None,
            tenor=b.tenor, unit="rows", definition="the number of priced rows whose flag_clip column differs from (clip_low_inner_max > 0.01 or clip_high_inner_max > 0.01)",
            budget="no simulation: a comparison of stored columns", commit=", ".join(sorted({s.commit for s in priced if s.commit})), source="the rows of sections 2 and 3", n=seen,
            notes="no standard error: a count" if seen else "pending: no priced row carries the flag yet",
        )
    )  # fmt: skip
    rows.append(["rows whose flag_clip differs from the recomputed flag", str(seen), str(bad) if seen else "pending"])  # fmt: skip
    b.table(["check", "rows compared", "largest absolute difference (count for the flag)"], rows)


# ---------------------------------------------------------------------------------------------
# the part
# ---------------------------------------------------------------------------------------------


def build(
    args: argparse.Namespace, base: Path, now: dt.datetime
) -> tuple[Builder, dict[str, pd.DataFrame], str]:
    tenor = args.tenor
    strict = not args.no_strict
    entries_path = pc.STUDY / f"entries_{tenor}.parquet"
    entries = study_entries(tenor)
    sizes = budget_sizes()
    dates = yearly_dates(entries, tenor)
    dev_dates = [d for d in dates["date"] if d]
    prod_dir, dev_dir, before_dir = (Path(p) for p in (args.rows_dir, args.development_rows_dir, args.before_rows_dir))  # fmt: skip
    prod_table_path, dev_table_path = Path(args.production_table), Path(args.development_table)
    prod_table, dev_table = read_rows_table(prod_table_path), read_rows_table(dev_table_path)
    prod = [load_slot("prod", d, "production", tenor, prod_dir, prod_table, prod_table_path, entries, sizes, strict) for d in pc.REFERENCE_DATES]  # fmt: skip
    dev = [load_slot("dev", d, "development", tenor, dev_dir, dev_table, dev_table_path, entries, sizes, strict) for d in dev_dates]  # fmt: skip
    b = Builder(tenor, entries_path)
    tables: dict[str, pd.DataFrame] = {"1y_dates": dates}
    check_date = pc.REFERENCE_DATES[0]

    def count(slots: Sequence[Slot]) -> str:
        n = {k: sum(s.state == k for s in slots) for k in ("ok", "failed", "pending")}
        return f"{n['ok']} priced, {n['failed']} failed, {n['pending']} pending of {len(slots)}"

    b.text(
        f"# One-year addendum ({tenor}): local correlation against the constant-correlation companion and the copula",
        "",
        f"Written {now.strftime('%Y-%m-%d %H:%M')} New York time into `{base}` by `scripts/pm_1y.py` (re-run as the rows arrive; the folder `1y` is frozen at 13:00, later runs write `later/1y`).",
        f"Rows at this writing — production budget, four reference dates: {count(prod)}; development budget, {len(dev)} yearly dates: {count(dev)}.",
        "A cell reads *pending* when its row is not there yet (the file it waits for is under the table), *failed* when the date's run failed, *not run* when the row does not carry the column. Every number is in `numbers_1y.json` with its standard error, definition, budget, commit and source.",
    )
    if not strict:
        b.text("**TEST RENDER (`--no-strict`): rows that do not carry the owner's decisions 1, 2 and 5 are accepted. Not for the package.**")  # fmt: skip
    for year in dates[dates["date"] == ""]["year"]:
        b.text(f"Year {year}: no monthly {tenor} entry of basket B1 with its pickle — no date for that year.")  # fmt: skip
    rejected = [s for s in (*prod, *dev) if s.problems]
    if rejected:
        b.text("Rows present but not of the specification asked for, written as pending: " + "; ".join(f"{s.budget} {s.date} ({', '.join(s.problems)})" for s in rejected) + ".")  # fmt: skip

    # --- 1. decision 5
    b.text(
        f"## 1. Decision 5 (the names' calendar repair on the DJX surface): the check on {check_date} at {tenor}"
    )
    after_prod = next(s for s in prod if s.date == check_date)
    after_dev = next((s for s in dev if s.date == check_date), None)
    before: Slot | None = None
    before_path = before_dir / f"{check_date}.json"
    if before_path.exists():
        row = json.loads(before_path.read_text())
        is_before = row.get("calendar_repair") is not True or "n_dropped_calendar_index" not in row
        if row.get("tenor") == tenor and str(row.get("date")) == check_date and is_before:
            before = Slot("before", check_date, str(row.get("budget")), None, row=row, source=str(before_path),
                          state="failed" if row.get("status") == "failed" else "ok", reason=str(row.get("reason", "")))  # fmt: skip
        else:
            log.warning("%s: not a %s row of the old defaults on %s — ignored", before_path, tenor, check_date)  # fmt: skip
    variants: list[tuple[str, Slot | None, str]] = [
        ("decisions 1, 2, 5 on — production budget", after_prod, "")
    ]
    if after_dev is not None:
        variants.append(("decisions 1, 2, 5 on — development budget", after_dev, ""))
    variants.append(("before the repair (old defaults)" + (f" — {before.budget} budget, commit {before.commit}" if before else ""), before, "not run"))  # fmt: skip
    tables["1y_decision5"] = decision5(b, variants, check_date)
    d5 = [f"Clipped mass: the share of the particles that are inside ±2.5 at-the-money sd of the basket and whose λ is clipped at 0, resp. at the cap, the largest over the calibration's slices (fractions; flagged above 0.01); DJX slices dropped by the repair: n_dropped_calendar_index; last DJX slice: index_last_slice, in years (horizon {float(entries.loc[check_date, 'T']):.4f}y)."]  # fmt: skip
    d5.append(footer([after_prod], False, entries_path, prod_dir))
    if after_dev is not None:
        d5.append("Development column — " + footer([after_dev], False, entries_path, dev_dir))
    d5.append(f"Before the repair: {'`' + before.source + '`' if before else f'not run at {tenor} (no row of the old defaults at `{before_path}`)'}.")  # fmt: skip
    b.text(" ".join(d5))

    # --- 2. the four dates at the production budget
    b.text(
        f"## 2. The four reference dates at the production budget ({tenor}, decisions 1, 2 and 5 on)"
    )
    plan = [
        ("2a. The forward", FORWARD, "forward", True),
        ("2b. The listed-variance forward, ED_wing and ED_eqv", LISTED, "listed", True),
        ("2c. κ and E[V]/EQV", KAPPA, "kappa", True),
        ("2c′. The split of E_LC[V] − EQV", SPLIT, "split", False),
        ("2d. The calls at K_075 … K_150 against the copula's", [q for t in CALL_TAGS for q in CALLS[t]], "calls", True),
        ("2e. Clipped mass, index errors, names priced beyond their last listed expiry", [*CLIP, *INDEX], "calibration", False),
        ("2f. Sticky-strike deltas", DELTAS, "deltas", False),
    ]  # fmt: skip
    for title, quantities, key, study in plan:
        b.text(f"### {title}")
        by_quantity(b, prod, quantities)
        b.text(f"{DEFINITIONS[key]} {footer(prod, study, entries_path, prod_dir)}")
    tables["1y_production"] = frame_of(prod, tenor)

    # --- 3. the twenty dates at the development budget
    b.text(
        f"## 3. The yearly dates at the development budget ({tenor}): the first monthly entry of each year 2007–2026"
    )
    in_sweep = dates[dates["date"] != ""]
    outside = [d for d, inside in zip(in_sweep["date"], in_sweep["in_3m_sweep"], strict=True) if not inside]  # fmt: skip
    short = [f"{y} ({n})" for y, n in zip(in_sweep["year"], in_sweep["monthly_entries_in_year"], strict=True) if n < 12]  # fmt: skip
    also_reference = [d for d in dev_dates if d in pc.REFERENCE_DATES]
    b.text(
        f"Dates ({len(dev_dates)}): {', '.join(dev_dates)}. "
        + (f"Not among the 219 dates of the 3-month sweep: {', '.join(outside)}. " if outside else "All are among the 219 dates of the 3-month sweep. ")
        + f"Selection: basket B1 of `{entries_path}` (one row per date and basket), `monthly` true, the first date of each year with its pickle under `outputs/dispersion/entries/{tenor}/`. "
        + (f"Years with fewer than 12 monthly entries at {tenor} (number in the year): {', '.join(short)}. " if short else "")
        + (f"Also a reference date of section 2: {', '.join(also_reference)} — its row here is the development-budget one, a run of its own." if also_reference else "")
    )  # fmt: skip
    plan3 = [
        ("3a. The forward", FORWARD, "forward", True),
        ("3b. The listed-variance forward, ED_wing and ED_eqv", LISTED, "listed", True),
        ("3c. κ and E[V]/EQV", KAPPA, "kappa", True),
        ("3c′. The split of E_LC[V] − EQV", SPLIT, "split", False),
        *[(f"3d. The call at K_{t} = {CALL_MULT[t]} × P_D against the copula's", CALLS[t], "calls", True) for t in CALL_TAGS],
        ("3e. Clipped mass, index errors, names priced beyond their last listed expiry", [*CLIP, *INDEX], "calibration", False),
        ("3f. Sticky-strike deltas", DELTAS, "deltas", False),
    ]  # fmt: skip
    for title, quantities, key, study in plan3:
        b.text(f"### {title}")
        by_date(b, dev, quantities)
        b.text(f"{DEFINITIONS[key]} {footer(dev, study, entries_path, dev_dir)}")
    dev_frame = frame_of(dev, tenor)
    tables["1y_development"] = dev_frame
    n_priced = int((dev_frame["state"] == "ok").sum())
    n_unflagged = int(((dev_frame["state"] == "ok") & (dev_frame["flag_clip"].eq(False))).sum())
    b.text("### 3g. Mean, median and quartiles over the dates")
    csv = base / "tables" / "1y_development_summary.csv"
    tables["1y_development_summary"] = summaries(b, dev, dev_frame, dev_dir, csv)
    b.text(
        f"Left block: the {n_unflagged} priced dates not flagged for the clipped mass (the owner's decision 5: flagged rows stay in the tables and out of the summaries); right block, labelled *all priced*: the {n_priced} priced dates, flagged or not. "
        f"n is the number of dates behind each figure ({len(dev)} dates asked); mean ± the standard error of the mean across dates (sd/√n, not a Monte Carlo error); quartiles linearly interpolated. "
        f"{footer(dev, True, entries_path, dev_dir)} The CSV behind: `{csv}`."
    )

    # --- 4. flags
    b.text("## 4. Flags per date")
    tables["1y_flags"] = flags_section(b, [*prod, *dev])
    for q in FLAGS:
        for s in [*prod, *dev]:
            b.add(q, s)
    b.text(f"{DEFINITIONS['flags']} Production rows — {footer(prod, False, entries_path, prod_dir)} Development rows — {footer(dev, False, entries_path, dev_dir)}")  # fmt: skip

    # --- 5. consistency
    b.text("## 5. The rows against the study's entries")
    consistency(b, [*prod, *dev])
    b.text(f"The copula columns a row repeats (P_D_copula, EV_copula, EQV, T, the cash strikes) against basket B1 of `{entries_path}`, and the row's own flag against the flag recomputed here; no simulation. Sources: the rows of sections 2 and 3.")  # fmt: skip
    summary = f"1y ({tenor}): production {count(prod)}; development {count(dev)}"
    return b, tables, summary


def default_base(now: dt.datetime) -> Path:
    """Where a run at ``now`` writes: ``pm_update/1y`` before the owner's cutoff,
    ``pm_update/later/1y`` from it on."""
    return (pc.PM / "1y" if now < CUTOFF else pc.PM / "later" / "1y").resolve()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--tenor", default="12m")
    ap.add_argument("--rows-dir", "--production-rows-dir", dest="rows_dir", default=None, help="the production rows (default: rows/<tenor>_production)")  # fmt: skip
    ap.add_argument("--development-rows-dir", default=None, help="the development rows (default: rows/<tenor>_development)")  # fmt: skip
    ap.add_argument("--before-rows-dir", default=None, help="rows of the old defaults, for section 1 (default: rows/<tenor>_production_norepair)")  # fmt: skip
    ap.add_argument("--production-table", default=None, help="fallback table (default: lcm_<tenor>.parquet)")  # fmt: skip
    ap.add_argument("--development-table", default=None, help="fallback table (default: lcm_<tenor>_dev.parquet)")  # fmt: skip
    ap.add_argument("--base", default=None, help="the output folder (default: pm_update/1y until the cutoff, pm_update/later/1y after)")  # fmt: skip
    ap.add_argument("--no-strict", action="store_true", help="accept rows without the decisions (tests; never into the package)")  # fmt: skip
    ap.add_argument("--no-status", action="store_true", help="do not append to STATUS.md")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    rows = pc.LC_OUT / "rows"
    args.rows_dir = args.rows_dir or str(rows / f"{args.tenor}_production")
    args.development_rows_dir = args.development_rows_dir or str(rows / f"{args.tenor}_development")  # fmt: skip
    args.before_rows_dir = args.before_rows_dir or str(rows / f"{args.tenor}_production_norepair")
    args.production_table = args.production_table or str(pc.LC_OUT / f"lcm_{args.tenor}.parquet")
    args.development_table = args.development_table or str(pc.LC_OUT / f"lcm_{args.tenor}_dev.parquet")  # fmt: skip
    now = dt.datetime.now(NY)
    base = Path(args.base).resolve() if args.base else default_base(now)
    in_package = pc.PM.resolve() in [base, *base.parents]
    if in_package and (args.no_strict or args.tenor != "12m"):
        raise SystemExit("--no-strict and tenors other than 12m are tests: give a --base outside pm_update")  # fmt: skip
    if in_package and now >= CUTOFF and base == (pc.PM / "1y").resolve():
        raise SystemExit(
            f"{base} is frozen since {CUTOFF:%H:%M}: write to {pc.PM / 'later' / '1y'}"
        )
    b, tables, summary = build(args, base, now)
    for name, frame in tables.items():
        pc.save_table(frame, name, base=base)
    markdown = "\n".join(b.md)
    pc.write_part(PART, b.records, markdown, base=base)
    for name, text in (("NUMBERS_1Y.md", markdown.rstrip() + "\n"), ("numbers_1y.json", json.dumps(b.records, indent=1))):  # fmt: skip
        tmp = base / (name + ".tmp")
        tmp.write_text(text)
        tmp.replace(base / name)
    n_values = sum(r["value"] is not None for r in b.records)
    log.info("%s: %d records (%d with a value) -> %s", summary, len(b.records), n_values, base / "NUMBERS_1Y.md")  # fmt: skip
    if not args.no_status:
        line = f"{summary}; {len(b.records)} records ({n_values} with a value) -> `{base / 'NUMBERS_1Y.md'}`"
        pc.status(line) if in_package else pc.status(line, base=base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
