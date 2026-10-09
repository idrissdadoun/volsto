"""The PM results package of 2026-10-09, the one-year addendum (``pm_update/1y``): the local
correlation model against its constant-correlation companion and the dispersion study's copula
at the 12-month tenor.

    python scripts/pm_1y.py [--tenor 12m] [--rows-dir <production rows>]
        [--development-rows-dir <development rows>] [--before-rows-dir <old defaults rows>]
        [--slices-table <1y_djx_slices.csv>] [--base <folder>] [--no-strict] [--no-status]

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
and are out of the first summary; the second summary is over every priced date.  No other flag
excludes a date.  A count is summarised by its total; an exact constant is not summarised.

The index.  Status ``ok`` means that no gating check of the row fails; the index gate (0.15
vol points at the money and at the 90 % strike) is waived whenever the wing binds, so the page
prints, per date, whether the wing binds, the gate's two numbers and whether either is outside
0.15, the errors at +1.5 and +2.5 sd at the horizon and the row's largest errors over the kept
slices with where they sit (from the row's ``index_errors`` list).  A cell whose model vol is 0
in the row is marked not invertible; a cell whose target vol is below a quarter of the slice's
at-the-money target vol is marked degenerate.

The DJX slices.  A row carries counts and, in ``index_errors``, the kept slices up to the
horizon.  Which slices the screen and the calendar repair drop, and which kept slices bracket
the horizon, are read from ``tables/1y_djx_slices.csv`` when it is there (the
specification-only build of ``scripts/pm_1y_slices.py``; its specification key is compared
with the row's); without it the page prints what the row gives and says that the list is not
available.

The index level.  Under 2a and 3a a second table gives, per date, the at-the-money vol of the
model's index target at the horizon (the row's ``index_errors``), ``sig_B_DJX`` of the study's
entry, their difference, ``rho_cc`` and ``rho_cop``: LC/copula and CC/copula carry that gap,
only LC/CC is a local-correlation effect.

Flagged rows.  A row flagged for the clipped mass stays in every table; its records say so in
their notes (``row flagged: …``), and so do the records of the summary over every priced date.
A cell of the index errors is marked when its model vol is not invertible, when its target vol
is below a quarter of the slice's at-the-money target vol (degenerate) or below half of it
(thin); the CSVs carry one ``<column>_mark`` per index-error cell.

Section 1's "before" row is the one of the old defaults (no calendar repair of the names or of
DJX, no unscreened fallback): ``rows/<tenor>_production_norepair/<date>.json``, else
``rows/<tenor>_development_norepair/<date>.json``; it is compared with the column of the same
budget.

Re-runnable: every output is rewritten atomically from the inputs present.
"""

# ruff: noqa: RUF001, E501 — report prose: typographic signs and long caption lines
from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import logging
import math
import re
import subprocess
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
#: the index gate of ``scripts/lcm_price.py`` (``check_index``): both numbers within this, in vol
#: points, unless the wing binds
GATE_VP = 0.15
#: the floats ``check_no_nan`` of ``scripts/lcm_price.py`` does not read (prefixes)
NAN_EXEMPT = ("align_", "C_200", "Cfwd", "profile_")
#: a target vol below this share of the slice's at-the-money target vol is called degenerate
DEGENERATE_TARGET = 0.25
#: a target vol below this share of the slice's at-the-money target vol (and not degenerate) is
#: called thin
THIN_TARGET = 0.5
#: the short marks of a cell of ``index_errors`` (the CSVs' mark columns)
MARK_NOT_INVERTIBLE = "model vol not invertible"
MARK_DEGENERATE = "target degenerate"
MARK_THIN = "thin target"
NOT_INVERTIBLE = (
    "model vol not invertible (no Monte Carlo payoff beyond the strike); error = −target"
)
#: the table of ``scripts/pm_1y_slices.py`` (the specification-only build of the DJX slices)
SLICES_TABLE = "1y_djx_slices"
ROWS_3M = {"production": pc.LC_OUT / "rows" / "3m_production", "development": pc.LC_OUT / "rows" / "3m_development_repair"}  # fmt: skip


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
    #: ``(row, value, se) -> (cell text, note, mark)``: what replaces the plain ``value ± se`` in a
    #: table cell and what the record's notes say of it (empty strings: nothing to say)
    #: and its short mark (``cell_flags`` joined; empty: a usable cell)
    annotate: Callable[[dict[str, Any], float, float | None], tuple[str, str, str]] | None = None


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


def _gate_outside(r: dict[str, Any] | None, e: dict[str, Any] | None) -> tuple[float, None]:
    atm, k90 = _number(r, "idx_err_atm"), _number(r, "idx_err_90")
    return float(abs(atm) > GATE_VP or abs(k90) > GATE_VP), None


def index_cells(row: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The row's ``index_errors`` list (model against target per kept slice up to the horizon
    and per strike); empty when the row does not carry it."""
    cells = (row or {}).get("index_errors")
    return list(cells) if isinstance(cells, list) else []


def cell_flags(row: dict[str, Any], cell: dict[str, Any]) -> list[str]:
    """The short marks of one cell of ``index_errors``: the model vol is not invertible (0 in
    the row); the target vol is below ``DEGENERATE_TARGET`` of the slice's at-the-money target
    vol (degenerate) or between that and ``THIN_TARGET`` of it (thin).  Empty: a usable cell."""
    marks = []
    if not _number(cell, "model_vol") > 0:
        marks.append(MARK_NOT_INVERTIBLE)
    atm = [c for c in index_cells(row) if c["T"] == cell["T"] and c["strike"] == "+0.0"]
    if atm and _number(atm[0], "target_vol") > 0:
        share = _number(cell, "target_vol") / _number(atm[0], "target_vol")
        if share < DEGENERATE_TARGET:
            marks.append(MARK_DEGENERATE)
        elif share < THIN_TARGET:
            marks.append(MARK_THIN)
    return marks


def cell_mark(row: dict[str, Any], cell: dict[str, Any]) -> str:
    """What a reader must know of one cell of ``index_errors`` beyond its number, in words
    (:func:`cell_flags`).  Empty when the cell is usable."""
    flags = cell_flags(row, cell)
    marks = []
    target = _number(cell, "target_vol")
    if MARK_NOT_INVERTIBLE in flags:
        marks.append(NOT_INVERTIBLE)
    atm = [c for c in index_cells(row) if c["T"] == cell["T"] and c["strike"] == "+0.0"]
    if MARK_DEGENERATE in flags:
        text = f"target vol {100 * target:.2f} % against {100 * _number(atm[0], 'target_vol'):.2f} % at the money: degenerate"
        at_horizon = math.isclose(float(cell["T"]), _number(row, "T"), abs_tol=1e-9)
        if at_horizon and str(cell["strike"]).startswith("+"):
            repaired = "repaired " if (row.get("n_dropped_calendar_index") or 0) > 0 else ""
            text += f"; the {repaired}DJX target is not usable on the upside on this date"
        marks.append(text)
    if MARK_THIN in flags:
        marks.append(f"thin target: target vol {100 * target:.2f} % against {100 * _number(atm[0], 'target_vol'):.2f} % at the money, below half of it")  # fmt: skip
    return "; ".join(marks)


def horizon_cell(row: dict[str, Any] | None, strike: str) -> dict[str, Any] | None:
    """The cell of ``index_errors`` at the horizon at ``strike`` (a label of the list)."""
    if row is None:
        return None
    T = _number(row, "T")
    cells = [c for c in index_cells(row) if c["strike"] == strike and math.isclose(float(c["T"]), T, abs_tol=1e-9)]  # fmt: skip
    return cells[0] if cells else None


def _where(row: dict[str, Any], cell: dict[str, Any]) -> str:
    at_horizon = math.isclose(float(cell["T"]), _number(row, "T"), abs_tol=1e-9)
    strike = cell["strike"] if cell["strike"] == "90%" else f"{cell['strike']} sd"
    return f"{float(cell['T']):.3f}y{' (the horizon)' if at_horizon else ''}, {strike}"


def _vols(cell: dict[str, Any]) -> str:
    return f"model vol {100 * _number(cell, 'model_vol'):.2f} %, target vol {100 * _number(cell, 'target_vol'):.2f} %"


Annotate = Callable[[dict[str, Any], float, float | None], tuple[str, str, str]]


def _mark_at_horizon(strike: str) -> Annotate:
    """The cell text of the index error at ``strike`` (a label of ``index_errors``) at the
    horizon when the cell is marked (:func:`cell_mark`)."""

    def annotate(row: dict[str, Any], value: float, se: float | None) -> tuple[str, str, str]:
        cell = horizon_cell(row, strike)
        if cell is None:
            return "", "", ""
        mark, short = cell_mark(row, cell), "; ".join(cell_flags(row, cell))
        if not mark:
            return "", _vols(cell), ""
        if _number(cell, "model_vol") > 0:
            return f"{pc.pm(value, se, 3)} ({mark})", f"{_vols(cell)}; {mark}", short
        return f"{mark} ({value:.3f})", f"{_vols(cell)}; {mark}", short

    return annotate


def sd_cells(
    row: dict[str, Any] | None, limit: float, usable: bool = False
) -> list[dict[str, Any]]:
    """The cells of ``index_errors`` at the strikes in at-the-money sd inside ±``limit``, over
    the slices; with ``usable``, only those that carry no mark (:func:`cell_flags`)."""
    cells = [c for c in index_cells(row) if c["strike"] != "90%" and abs(float(c["strike"])) <= limit + 1e-9]  # fmt: skip
    return [c for c in cells if not cell_flags(row or {}, c)] if usable else cells


def _usable_max(r: dict[str, Any] | None, e: dict[str, Any] | None) -> tuple[float, None]:
    if not index_cells(r):
        raise KeyError("index_errors")
    cells = sd_cells(r, 2.5, usable=True)
    return (max(abs(_number(c, "error_vp")) for c in cells) if cells else float("nan")), None


def _mark_max(limit: float, usable: bool = False) -> Annotate:
    """The cell text of the row's largest |index error| inside ±``limit`` sd over the slices
    (over the usable cells only with ``usable``): the value, the slice and the strike where it
    sits (found in ``index_errors``), the error of that cell, and its mark."""

    def annotate(row: dict[str, Any], value: float, se: float | None) -> tuple[str, str, str]:
        cells = sd_cells(row, limit, usable)
        if not cells:
            return (
                "",
                "the row does not carry the list of cells: where the maximum sits is not known",
                "",
            )
        worst = max(cells, key=lambda c: abs(_number(c, "error_vp")))
        if not math.isclose(abs(_number(worst, "error_vp")), value, rel_tol=1e-9, abs_tol=1e-12):
            return "", "the row's maximum is not a cell of its index_errors list", ""
        cell_se = _number(worst, "stderr_vp")
        text = f"{value:.3f} at {_where(row, worst)}"
        note = f"sits at {_where(row, worst)}: error {_number(worst, 'error_vp'):+.3f} vol points, {_vols(worst)}"
        if math.isfinite(cell_se):
            text += f" (that cell: {_number(worst, 'error_vp'):+.3f} ± {cell_se:.3f})"
            note += f", standard error of that cell {cell_se:.3f}"
        mark = cell_mark(row, worst)
        if mark:
            text += f" — {mark}"
            note += f"; {mark}"
        return text, note, "; ".join(cell_flags(row, worst))

    return annotate


def _target_atm(r: dict[str, Any] | None, e: dict[str, Any] | None) -> tuple[float, None]:
    cell = horizon_cell(r, "+0.0")
    if cell is None:
        raise KeyError("index_errors")
    return 100 * _number(cell, "target_vol"), None


def _study_atm(r: dict[str, Any] | None, e: dict[str, Any] | None) -> tuple[float, None]:
    return 100 * _number(e, "sig_B_DJX"), None


def _atm_gap(r: dict[str, Any] | None, e: dict[str, Any] | None) -> tuple[float, None]:
    return _target_atm(r, e)[0] - _study_atm(r, e)[0], None


NO_SE_STUDY = "read from the study's table, which carries no standard error for it"
NO_SE_COUNT = "a count of the inputs, not a Monte Carlo estimate"
NO_SE_FLAG = "a yes/no read from the row, not a Monte Carlo estimate"
NO_SE_MAX = "a maximum of |error| over cells has no standard error of its own (the error of the cell where it sits is in the notes)"
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
    Quantity("idx_err_atm", "index error ATM", "the model's implied vol of the index at the horizon minus its target, at the money", _row("idx_err_atm", "idx_err_atm_se"), unit="vol points", digits=3, annotate=_mark_at_horizon("+0.0")),
    Quantity("idx_err_90", "90 %", "the model's implied vol of the index at the horizon minus its target, at the 90 % strike", _row("idx_err_90", "idx_err_90_se"), unit="vol points", digits=3, annotate=_mark_at_horizon("90%")),
    Quantity("idx_err_m15", "−1.5 sd", "the model's implied vol of the index at the horizon minus its target, at −1.5 sd", _row("idx_err_m15", "idx_err_m15_se"), unit="vol points", digits=3, annotate=_mark_at_horizon("-1.5")),
    Quantity("idx_err_m20", "−2 sd", "the model's implied vol of the index at the horizon minus its target, at −2 sd", _row("idx_err_m20", "idx_err_m20_se"), unit="vol points", digits=3, annotate=_mark_at_horizon("-2.0")),
    Quantity("idx_err_m25", "−2.5 sd", "the model's implied vol of the index at the horizon minus its target, at −2.5 sd", _row("idx_err_m25", "idx_err_m25_se"), unit="vol points", digits=3, annotate=_mark_at_horizon("-2.5")),
    Quantity("n_names_extrapolated", "names beyond last kept expiry (count)", "the number of names priced beyond their last kept expiry: the horizon is after the name's last expiry kept after the screen and the calendar repair (a count: the row does not name the names)", _row("n_names_extrapolated"), unit="names", kind="int", no_se=NO_SE_COUNT),
]  # fmt: skip
_IDX = "the model's implied vol of the index minus its target"
GATE = [
    Quantity("wing_binds", "wing binds (clipped mass > 1 %)", "the row's wing_binds: the clipped mass inside ±2.5 sd (clip_inner_max) exceeds 1 %; the index gate is then waived", _truth("wing_binds"), kind="bool", unit="1 = yes", no_se=NO_SE_FLAG, summarise=False),
    Quantity("index_gate_outside", "outside 0.15 vp", "true when the index error at the money or at the 90 % strike at the horizon is beyond 0.15 vol points in absolute value (the two numbers of the index gate, whether or not the gate is waived)", _gate_outside, kind="bool", unit="1 = yes", no_se=NO_SE_FLAG, summarise=False),
    Quantity("check_index", "check_index (the row's)", "the row's check_index: true when the wing binds or both gate numbers are within 0.15 vol points", _truth("check_index"), kind="bool", unit="1 = yes", no_se=NO_SE_FLAG, summarise=False),
]  # fmt: skip
INDEX_MORE = [
    Quantity("idx_err_p15", "+1.5 sd", f"{_IDX}, at the horizon at +1.5 sd", _row("idx_err_p15", "idx_err_p15_se"), unit="vol points", digits=3, summarise=False, annotate=_mark_at_horizon("+1.5")),
    Quantity("idx_err_p25", "+2.5 sd", f"{_IDX}, at the horizon at +2.5 sd", _row("idx_err_p25", "idx_err_p25_se"), unit="vol points", digits=3, summarise=False, annotate=_mark_at_horizon("+2.5")),
    Quantity("idx_err_max_1p5", "largest absolute error inside ±1.5 sd, all slices", f"the largest absolute value of {_IDX} over the kept DJX slices up to the horizon and the strikes at 0, ±0.5, ±1 and ±1.5 sd (the row's idx_err_max_1p5)", _row("idx_err_max_1p5"), unit="vol points", digits=3, no_se=NO_SE_MAX, summarise=False, annotate=_mark_max(1.5)),
    Quantity("idx_err_max_2p5", "largest absolute error inside ±2.5 sd, all slices", f"the largest absolute value of {_IDX} over the kept DJX slices up to the horizon and the strikes at 0, ±0.5, …, ±2.5 sd (the row's idx_err_max_2p5)", _row("idx_err_max_2p5"), unit="vol points", digits=3, no_se=NO_SE_MAX, summarise=False, annotate=_mark_max(2.5)),
    Quantity("idx_err_max_2p5_usable", "largest absolute error inside ±2.5 sd over the usable cells, all slices", f"the largest absolute value of {_IDX} over the cells of the row's index_errors list (kept DJX slices up to the horizon, strikes at 0, ±0.5, …, ±2.5 sd) that carry no mark: model vol invertible and target vol at least half the slice's at-the-money target vol (computed here from the list)", _usable_max, unit="vol points", digits=3, no_se=NO_SE_MAX, summarise=False, annotate=_mark_max(2.5, usable=True)),
]  # fmt: skip
_NO_SE_INPUT = "the calibration's target, an input of the model: not a Monte Carlo estimate"
LEVEL = [
    Quantity("target_atm_vol", "index target ATM vol at the horizon (model)", "the at-the-money (log-forward-moneyness 0) vol of the model's index target — its SVI surface of the kept DJX slices — at the horizon, in percent (the target_vol of the row's index_errors list at the horizon, strike +0.0)", _target_atm, unit="vol in %", digits=2, no_se=_NO_SE_INPUT, summarise=False),
    Quantity("study_atm_vol", "index ATM vol of the study's entry", "sig_B_DJX of the study's entry, basket B1, in percent: the implied vol at K = S_0 (the spot) of the study's DJX marginal at the entry's maturity (scripts/disp_entries.py: the marginal's atm_vol; for B1 it is the study's sig_B)", _study_atm, origin="study", unit="vol in %", digits=2, no_se=NO_SE_STUDY, summarise=False),
    Quantity("target_minus_study_atm", "target minus study, ATM", "the model's index target at-the-money vol at the horizon minus sig_B_DJX of the study's entry, in vol points (the first at the forward, the second at the spot)", _atm_gap, origin="both", unit="vol points", digits=2, no_se="a difference of two inputs, not a Monte Carlo estimate", summarise=False),
    Quantity("rho_cc", "ρ_CC", "the constant correlation of the companion model, the one that reprices the index at-the-money straddle (the row's rho_cc)", _row("rho_cc"), unit="correlation", no_se="a fitted parameter: the row carries no standard error for it", summarise=False),
    Quantity("rho_cop", "ρ_cop", "the copula's correlation of the study's entry, basket B1 (rho_cop)", _study("rho_cop"), origin="study", unit="correlation", no_se=NO_SE_STUDY, summarise=False),
]  # fmt: skip
DELTAS = [
    Quantity("delta_fwd_lc", "Δ forward LC", "sticky-strike common delta of the forward under LC: percent of the model's price per +1 % of every spot", _row("delta_fwd_lc", "delta_fwd_lc_se"), unit="% per %"),
    Quantity("delta_fwd_cc", "Δ forward CC", "sticky-strike common delta of the forward under CC: percent of the model's price per +1 % of every spot", _row("delta_fwd_cc", "delta_fwd_cc_se"), unit="% per %"),
    Quantity("delta_C100_lc", "Δ call K_100 LC", "sticky-strike common delta of the call at K_100 under LC, percent of its price per +1 % of every spot", _row("delta_C100_lc", "delta_C100_lc_se"), unit="% per %", digits=3),
    Quantity("delta_C100_cc", "Δ call K_100 CC", "sticky-strike common delta of the call at K_100 under CC, percent of its price per +1 % of every spot", _row("delta_C100_cc", "delta_C100_cc_se"), unit="% per %", digits=3),
    Quantity("delta_homogeneity", "homogeneity", "the homogeneity term of the forward's sticky-strike common delta (a common move at sticky moneyness scales D): exactly 1", _row("delta_homogeneity"), unit="% per %", no_se="exact by homogeneity", summarise=False),
    Quantity("delta_skew_channel", "skew channel", "the skew channel of the forward's sticky-strike common delta: Δ^CC_ss − 1", _row("delta_skew_channel", "delta_skew_channel_se"), unit="% per %"),
    Quantity("delta_correlation_channel", "correlation channel", "the correlation channel of the forward's sticky-strike common delta: Δ^LC_ss − Δ^CC_ss, paired on common paths", _row("delta_correlation_channel", "delta_correlation_channel_se"), unit="% per %"),
]  # fmt: skip
FLAGS = [
    Quantity("index_extrapolated", "DJX target extrapolated", "true when the last listed DJX expiry kept is before the horizon", _truth("index_extrapolated"), kind="bool", unit="1 = yes", no_se=NO_SE_FLAG, summarise=False),
    Quantity("index_last_slice", "last DJX slice", "the last DJX slice kept, in years", _row("index_last_slice"), unit="years", no_se=NO_SE_COUNT, summarise=False),
    Quantity("n_dropped_calendar_index", "DJX slices dropped by the repair", "the number of DJX expiries the calendar repair dropped (decision 5); the target is repaired when it is above 0", _row("n_dropped_calendar_index"), unit="expiries", kind="int", no_se=NO_SE_COUNT, summarise=False),
    Quantity("n_dropped_index", "DJX slices dropped, any rule", "the number of DJX expiries dropped by any rule of the screen", _row("n_dropped_index"), unit="expiries", kind="int", no_se=NO_SE_COUNT, summarise=False),
    Quantity("n_names_unscreened", "names kept unscreened", "the number of names the quote screen emptied, kept on their unscreened expiries (decision 2)", _row("n_names_unscreened"), unit="names", kind="int", no_se=NO_SE_COUNT, summarise=False),
    Quantity("flag_clip_low", "clip at λ = 0 > 1 %", "true when the share clipped at λ = 0 inside ±2.5 sd exceeds 1 %", _flag("clip_low_inner_max"), kind="bool", unit="1 = yes", no_se=NO_SE_FLAG, summarise=False),
    Quantity("flag_clip_high", "clip at the cap > 1 %", "true when the share clipped at the cap inside ±2.5 sd exceeds 1 %", _flag("clip_high_inner_max"), kind="bool", unit="1 = yes", no_se=NO_SE_FLAG, summarise=False),
]  # fmt: skip
GROUPS: dict[str, list[Quantity]] = {
    "forward": FORWARD,
    "listed": LISTED,
    "kappa": KAPPA,
    "split": SPLIT,
    **{f"calls_{tag}": CALLS[tag] for tag in CALL_TAGS},
    "clip": CLIP,
    "index": INDEX,
    "gate": GATE,
    "index_more": INDEX_MORE,
    "level": LEVEL,
    "deltas": DELTAS,
    "flags": FLAGS,
}
#: units: fractions of the notional for the forward, the copula's price, the strikes, the calls,
#: ED_wing and ED_eqv; squared returns for the two parts of the split; the rest are ratios
NOTIONAL = {"ED_lc", "ED_cc", "P_D", "listed_fwd", "ED_wing", "ED_eqv", *(f"{k}_{t}{s}" for t in CALL_TAGS for k, s in (("K", ""), ("C", "_lc"), ("C", "_cc"), ("C", "_cop")))}  # fmt: skip
SQUARED_RETURN = {"EV_single_part", "EV_basket_part"}
for _group in GROUPS.values():
    for _i, _q in enumerate(_group):
        if not _q.unit:
            _unit = "notional" if _q.key in NOTIONAL else "squared return" if _q.key in SQUARED_RETURN else "ratio"  # fmt: skip
            _group[_i] = dataclasses.replace(_q, unit=_unit)
ALL = [q for group in GROUPS.values() for q in group]


@dataclass(frozen=True)
class Cell:
    value: float | None
    se: float | None
    state: str  # "value" | "pending" | "failed" | "absent"
    note: str = ""
    text: str = ""  # what a table cell shows instead of ``value ± se`` (``Quantity.annotate``)
    mark: str = ""  # the short mark of an index-error cell (empty: usable)


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
    if q.annotate is not None and slot.row is not None:
        text, note, mark = q.annotate(slot.row, float(value), se)
        return Cell(float(value), se, "value", note, text, mark)
    return Cell(float(value), se, "value")


def show(q: Quantity, cell: Cell) -> str:
    if cell.state == "pending":
        return "pending"
    if cell.state == "failed":
        return "failed"
    if cell.state == "absent":
        return "not run" if cell.note.startswith("not run") else "n/a"
    assert cell.value is not None
    if cell.text:
        return cell.text
    if q.kind == "int":
        return str(round(cell.value))
    if q.kind == "bool":
        return "yes" if cell.value else "no"
    return pc.pm(cell.value, cell.se, q.digits)


RUNAWAY_NOTE = "high-strike call: a few runaway paths carry part of it (section V3 of the main package, measured at 3m; not measured at 12m); its ± is not a usable error"
SECOND_MOMENT_NOTE = "a second-moment quantity carried by a few paths, like the high-strike calls: its ± is not a usable error (see 'What the ± is')"
SECOND_MOMENT_KEYS = ("kappa_lc", "kappa_cc", "EV_over_EQV_lc", "EV_over_EQV_cc", "ED_eqv", "ED_eqv_over_cc", "ED_eqv_over_copula", "EV_single_part", "EV_single_part_over_EQV")  # fmt: skip


def key_notes(q: Quantity) -> list[str]:
    """What the notes of every record of a quantity say, whatever the date: the runaway caveat
    of the calls at K_125 and K_150 (the model's, not the copula's), the caveat of the
    second-moment quantities."""
    if q.origin != "study" and q.key.startswith(("C_125", "C_150")):
        return [RUNAWAY_NOTE]
    return [SECOND_MOMENT_NOTE] if q.key in SECOND_MOMENT_KEYS else []


def flag_note(slot: Slot) -> str:
    """``row flagged: …`` for a priced row flagged for the clipped mass (owner's decision 5)."""
    if not clip_flagged(slot):
        return ""
    low, high = _number(slot.row, "clip_low_inner_max"), _number(slot.row, "clip_high_inner_max")
    return f"row flagged: clipped mass {low:.4f} at λ = 0 / {high:.4f} at the cap inside ±2.5 sd (flagged above {CLIP_FLAG_MASS})"


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
        if q.origin != "study":
            notes += [n for n in (flag_note(slot), *key_notes(q)) if n]
        elif flag_note(slot):
            # a number of the study's entry on a date whose model row is flagged
            notes.append(f"the model's row of this date and budget is flagged ({flag_note(slot)})")
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


PM_SE = "Every ± in the per-date tables is the pricing Monte Carlo standard error on antithetic pair means given the calibrated model; it does not contain the calibration's own noise, the budget, or the sensitivity to the screen."
STATUS_NOTE = "Status ok means that no gating check fails (check_no_nan, check_forward, check_index). The index gate (0.15 vol points at the money and at the 90 % strike, at the horizon) is waived whenever the wing binds (clipped mass above 1 % inside ±2.5 sd), so ok does not mean that the index smile is repriced."
STICKY = "Sticky strike: each name's local volatility is held as a function of absolute spot and λ as a function of the absolute basket level, with no recalibration. The ± of a delta takes the base price as a constant (the base price's own error is left out)."
DEFINITIONS = {
    "forward": "E_LC[D], E_CC[D]: the Palladium forward E[Σ w_i |R_i − R̄|] under the calibrated local correlation model and under its constant-correlation companion; P_D: the copula's price in the study's entry (basket B1); all three in fractions of the notional; LC/CC is paired on common paths, LC/copula = E_LC[D]/P_D and CC/copula = E_CC[D]/P_D carry the delta-method error of two independent estimates.",
    "listed": "√(EQV/EV): the listed-variance forward over P_D, with EQV the listed-option value of E[V] and EV the copula's E[V] of the entry (no standard error in the study's table), and its level P_D·√(EQV/EV) = κ_cop·√EQV; ED_wing = κ_LC·√(Σ w E_LC[R_i²] − M_B^listed) and ED_eqv = κ_LC·√EQV (fractions of the notional), each also over E_CC[D] (the row's delta-method error) and over the copula's P_D (delta-method error of two independent estimates).",
    "kappa": "κ = E[D]/√E[V] under each model (κ_cop = P_D/√EV of the entry); E[V]/EQV under LC (EV_lc/EQV), CC (EV_cc/EQV) and the copula (EV/EQV of the entry), EQV the listed-option value of E[V].",
    "split": "The split of the LC figure: E_LC[V] − EQV = single-name part − basket part, with single-name part = Σ w E_LC[R_i²] − Σ w M_i^listed and basket part = E_LC[R̄²] − M_B^listed (both in squared returns); each also over EQV (so that E_LC[V]/EQV − 1 = single-name part/EQV − basket part/EQV), and the basket part over M_B^listed.",
    "calls": "C = E[(D − K)⁺] at the study's cash strikes K_m = m × P_D under LC, CC and the copula (the entry's C_m ± C_se_m), in fractions of the notional; LC/copula carries the delta-method error of two independent estimates, LC/CC is paired on common paths.",
    "calibration": "Clipped mass: the share of the particles that are inside ±2.5 at-the-money sd of the basket and whose λ is clipped at 0, resp. at the cap, the largest over the calibration's slices (fractions; a date is flagged above 0.01); index errors: the model's implied vol of the index at the horizon minus its target, in vol points; names beyond last kept expiry: the number of names whose last expiry kept after the screen and the calendar repair is before the horizon (a count: the row does not name the names).",
    "gate": "Index gate (the row's check_index): the index errors at the money and at the 90 % strike at the horizon, each within 0.15 vol points in absolute value; it is waived whenever the wing binds (the row's wing_binds: clipped mass above 1 % inside ±2.5 sd), so check_index true and status ok do not mean that the index smile is repriced. 'Outside 0.15 vp' says whether either of the two numbers is beyond 0.15, waived or not. +1.5 sd and +2.5 sd: the index error at the horizon on the upside. Largest absolute error: the row's idx_err_max_1p5 and idx_err_max_2p5, over the kept DJX slices up to the horizon and the strikes in at-the-money sd, with the slice and the strike where it sits (found in the row's index_errors list) and that cell's own error ± its standard error; a maximum over cells has no standard error of its own. A cell marked 'model vol not invertible (no Monte Carlo payoff beyond the strike); error = −target' is a far out-of-the-money strike with a negligible price: the Monte Carlo option has no payoff beyond the strike, the row stores a model vol of 0, and the 'error' equals minus the target vol; it is not a mispriced smile. A cell marked 'target vol … degenerate' is one where the target vol itself is below a quarter of the slice's at-the-money target vol: the DJX target is not usable at that strike. A cell marked 'thin target' is one where the target vol is between a quarter and a half of the slice's at-the-money target vol. The largest absolute error over the usable cells is the maximum over the cells that carry none of the three marks (computed here from the row's index_errors list). The CSVs carry one mark column per index-error cell (`<column>_mark`).",
    "level": "Index target ATM vol at the horizon (model): the at-the-money vol (log-forward-moneyness 0) of the model's index target — its SVI surface of the kept DJX slices, after the screen and the repair — at the horizon, in percent (the target_vol of the row's index_errors list). Index ATM vol of the study's entry: sig_B_DJX of basket B1, in percent — the implied vol at K = S_0 (the spot) of the study's DJX marginal at the entry's maturity (`scripts/disp_entries.py`: the marginal's atm_vol; for B1 it is the study's sig_B). Their difference is in vol points (the first is at the forward, the second at the spot). ρ_CC: the companion's constant correlation, the one that reprices the index at-the-money straddle (the row's rho_cc; the row carries no standard error for it). ρ_cop: the copula's correlation of the study's entry.",
    "deltas": f"Sticky-strike common deltas as elasticities (percent of each model's price per +1 % of every spot, central difference of ±1 %); the forward's LC delta = homogeneity (exactly 1) + skew channel (Δ^CC − 1) + correlation channel (Δ^LC − Δ^CC, paired). {STICKY}",
    "flags": f"{STATUS_NOTE} Row written: the time of the row's file, New York (month-day hour:minute); a name beyond its last kept expiry: n_names_extrapolated > 0, a count of names whose last expiry kept after the screen and the calendar repair is before the horizon (the row does not name them; the names in brackets are the specification build's, when its table is there); DJX target extrapolated: the last DJX slice kept is before the horizon; DJX target repaired: the calendar repair dropped at least one DJX expiry (n_dropped_calendar_index > 0; which ones: table 4b); a name kept unscreened: the quote screen emptied the name and it is kept on its unscreened expiries (decision 2); wing binds: the row's wing_binds; index gate: the index errors at the money and at the 90 % strike at the horizon, in vol points; outside 0.15 vp: either is beyond 0.15 in absolute value, waived or not; clip flags: the share clipped at λ = 0, resp. at the cap, exceeds 1 % inside ±2.5 sd (strictly).",
}
PRICED = {"ok": "yes", "failed": "no (failed)", "pending": "no (pending)"}
#: units of the columns a row repeats from the study's entry (the others: notional)
CHECK_UNITS = {"EV_copula": "squared return", "EQV": "squared return", "T": "years"}
GATING = ("check_no_nan", "check_forward", "check_index")


def call_caveat() -> str:
    """The caveat on the calls at K_125 and K_150, with the shares measured at 3m in section V3
    of the main package (read from its records; the sentence without them when they are not
    there)."""
    path = pc.PM / "parts" / "V_validation.json"
    shares: dict[str, list[float]] = {}
    if path.exists():
        for rec in json.loads(path.read_text()):
            m = re.fullmatch(r"V\.runaway\.[\d-]+\.lc\.(125|150|200)\.share_above_3x", rec["id"])
            if m and rec.get("value") is not None:
                shares.setdefault(m.group(1), []).append(100 * float(rec["value"]))
    head = "Caveat on the calls at K_125 and K_150 (section V3 of the main package, `parts/V_validation.md`): a few runaway paths carry the high-strike calls."
    tail = "It is not measured at 12m. At these strikes the ± is not a usable error."
    if not all(shares.get(k) for k in ("125", "150", "200")):
        return f"{head} {tail}"

    def span(k: str) -> str:
        return f"{min(shares[k]):.0f} to {max(shares[k]):.0f} %"

    return (
        f"{head} Measured at 3m at the development budget on {len(shares['150'])} dates (records `V.runaway.<date>.lc.<strike>.share_above_3x` of `{path}`): "
        f"the paths on which one name ends above 3 times its spot carry {span('150')} of the LC call at 1.5 × P_D, {span('200')} at 2 × P_D and {span('125')} at 1.25 × P_D. {tail}"
    )


def nonfinite_columns(row: dict[str, Any]) -> list[str]:
    """The floats of the row that are not finite, among those ``check_no_nan`` reads."""
    return [k for k, v in row.items() if isinstance(v, float) and not math.isfinite(v) and not k.startswith(NAN_EXEMPT)]  # fmt: skip


def reads_nonfinite(q: Quantity, slot: Slot) -> bool:
    """True when the quantity's value or standard error is not finite in its source."""
    try:
        value, se = q.get(slot.row, slot.entry)
    except (KeyError, TypeError):
        return False
    return not math.isfinite(value) or (se is not None and not math.isfinite(se))


def unusable_upside(slot: Slot, peers: Sequence[Slot]) -> str:
    """For a date whose index target is degenerate on the upside at the horizon: the target
    vols there against the at-the-money one, and where the date's CC/copula and LC/copula stand
    among its set of dates.  Empty when the target is not degenerate there."""
    r = slot.row or {}
    atm = horizon_cell(r, "+0.0")
    up = [(k, horizon_cell(r, k)) for k in ("+1.0", "+1.5", "+2.0", "+2.5")]
    if atm is None or not any(c is not None and MARK_DEGENERATE in cell_flags(r, c) for _, c in up):  # fmt: skip
        return ""
    vols = ", ".join(f"{k} sd {100 * _number(c, 'target_vol'):.2f} %" for k, c in up[:3] if c is not None)  # fmt: skip
    repaired = "repaired " if (r.get("n_dropped_calendar_index") or 0) > 0 else ""
    text = f"on this date the {repaired}DJX target is not usable on the upside at the horizon (target vols: {vols}, against {100 * _number(atm, 'target_vol'):.2f} % at the money) and that target feeds every LC and CC number of the row"
    ranks = []
    for q in (FORWARD[5], FORWARD[4]):
        mine = evaluate(q, slot).value
        others = [v for v in (evaluate(q, s).value for s in peers if s.priced) if v is not None]
        if mine is not None and others:
            ranks.append((q.label, mine, 1 + sum(v > mine for v in others), len(others)))
    if ranks and all(rank == 1 for _, _, rank, _ in ranks):
        text += f" (its {' and '.join(f'{label} {value:.3f}' for label, value, _, _ in ranks)} are the highest of the {ranks[0][3]} {slot.budget} dates)"
    elif ranks:
        text += " (" + "; ".join(f"its {label} {value:.3f} is number {rank} of the {n} {slot.budget} dates from the top" for label, value, rank, n in ranks) + ")"  # fmt: skip
    return text


def check_detail(slot: Slot, peers: Sequence[Slot] = ()) -> str:
    """For a priced row whose status is not ok: which gating check fails and, for
    ``check_no_nan``, which columns are not finite and which printed cells read them — computed
    from the row; ``peers``: the dates of the row's set (for the standing of its ratios)."""
    r = slot.row or {}
    parts = []
    for check in (c for c in GATING if r.get(c) is False):
        if check == "check_no_nan":
            bad = nonfinite_columns(r)
            dead = []
            for key in bad:
                m = re.fullmatch(r"idx_err_([mp])(\d)(\d)_se", key)
                if m:
                    dead.append(f"{'−' if m.group(1) == 'm' else '+'}{m.group(2)}.{m.group(3)} sd")
            text = (
                f"check_no_nan fails: not finite in the row: {', '.join(bad) or 'no column found'}"
            )
            if dead and len(dead) == len(bad):
                text += f" — the standard error of the index error at {' and '.join(dead)} at the horizon, where the model's implied vol is not invertible (model vol 0 in the row: no vega, no standard error)"
            affected = [q.label for q in ALL if q.origin != "study" and reads_nonfinite(q, slot)]
            if affected:
                text += f". Printed cells that read these columns: {', '.join(repr(a) for a in affected)} of the index tables (marked there); no other printed cell reads these columns"
            else:
                text += ". No printed cell reads these columns"
            upside = unusable_upside(slot, peers)
            if upside:
                text += f"; {upside}"
            parts.append(text)
        elif check == "check_forward":
            parts.append(
                f"check_forward fails: the basket's forward error {_number(r, 'forward_error'):+.6f} is beyond three standard errors (± {_number(r, 'forward_error_se'):.6f})"
            )
        else:
            parts.append(
                "check_index fails: the wing does not bind and a gate number is beyond 0.15 vol points"
            )
    if not parts:
        parts.append(f"the row's reason: {r.get('reason', '')}")
    return "; ".join(parts) + "."


def gate_numbers(slot: Slot) -> str:
    """The two numbers of the index gate, with their errors."""
    return f"ATM {show(INDEX[0], evaluate(INDEX[0], slot))}; 90 % {show(INDEX[1], evaluate(INDEX[1], slot))}"


def gate_summary(slots: Sequence[Slot]) -> str:
    """How many priced rows have the gate waived and its numbers outside 0.15 vp."""
    priced = [s for s in slots if s.priced and s.row and "wing_binds" in s.row]
    if not priced:
        return "No row is priced yet."
    binds = [s for s in priced if s.row and s.row["wing_binds"]]
    outside = [s for s in priced if _gate_outside(s.row, None)[0]]
    ok = [s for s in outside if s.row and s.row.get("status") == "ok"]
    text = f"At this writing the wing binds, and the gate is waived, on {len(binds)} of the {len(priced)} priced rows; the gate's two numbers are outside 0.15 vp on {len(outside)} of the {len(priced)}"
    if outside:
        text += (
            f" ({', '.join(f'{s.date} {s.budget}' for s in outside)}), of which {len(ok)} read ok"
        )
    return text + "."


# ---------------------------------------------------------------------------------------------
# the DJX slices: from the row and from the specification build
# ---------------------------------------------------------------------------------------------


def read_slices(base: Path, arg: str | None) -> tuple[pd.DataFrame | None, Path]:
    """The table of ``scripts/pm_1y_slices.py``: the one given, else the base's, else the one of
    ``pm_update/1y``; ``None`` (and the first path looked at) when there is none."""
    name = f"{SLICES_TABLE}.csv"
    paths = [Path(arg)] if arg else [base / "tables" / name, pc.PM / "1y" / "tables" / name]
    for path in paths:
        if path.exists():
            frame = pd.read_csv(path, dtype={"date": str})
            for col in ("expiry", "rule", "reason", "rms_source", "names_extrapolated", "names_unscreened"):  # fmt: skip
                if col in frame:
                    frame[col] = frame[col].fillna("").astype(str)
            return frame, path
    return None, paths[0]


@dataclass
class Djx:
    """What the page knows of a date's DJX target slices: (a) from the row — the kept slices
    below the horizon are the T values of its ``index_errors`` list, the last kept slice is
    ``index_last_slice``; (b) from the specification build, when its table has the date."""

    horizon: float
    row_below: list[float] | None  # None: no priced row
    last: float | None
    build: pd.DataFrame | None  # the date's rows of the table; None: no table, or date skipped
    skipped: str = ""  # why the build has no list for the date
    key_match: bool | None = None  # the build's specification key is the row's

    def bracket(self) -> tuple[float | None, float | None, str]:
        """The kept slices nearest the horizon, below and above, and where they are from."""
        if self.build is not None:
            kept = self.build[self.build["status"] == "kept"]["T"].astype(float)
            below, above = kept[kept <= self.horizon + 1e-9], kept[kept > self.horizon + 1e-9]
            return (float(below.max()) if len(below) else None, float(above.min()) if len(above) else None, "specification build")  # fmt: skip
        if self.row_below is not None:
            return (max(self.row_below) if self.row_below else None), None, "row"
        return None, None, ""

    def bracket_text(self) -> str:
        lo, hi, origin = self.bracket()
        if not origin:
            return "pending" if self.last is None else "not known: the row does not carry the list"
        low = f"{lo:.3f}y" if lo is not None else "none"
        if origin == "specification build":
            return f"{low} / {f'{hi:.3f}y' if hi is not None else 'none'} (specification build)"
        if self.last is not None and self.last > self.horizon + 1e-9:
            return f"{low} / not known from the row (a kept slice lies above the horizon: the last kept is {self.last:.3f}y)"
        return f"{low} / none (the last kept slice is before the horizon)"

    def dropped(self, calendar: bool) -> pd.DataFrame:
        assert self.build is not None
        gone = self.build[self.build["status"] == "dropped"]
        return gone[(gone["rule"] == "calendar") == calendar]

    def dropped_text(self, n_calendar: float | None) -> tuple[str, str]:
        """``(by the calendar repair, by the screen)`` as cell texts."""
        if self.build is None:
            why = self.skipped or "no table of the specification build"
            cal = (
                "pending"
                if n_calendar is None
                else f"{n_calendar:.0f} (the list is not available: {why})"
            )
            return cal, f"the list is not available: {why}"
        cal = "; ".join(f"{float(g['T']):.3f}y (SVI rms {float(g['svi_rms_vp']):.2f} vp; {g['reason']})" for _, g in self.dropped(True).iterrows())  # fmt: skip
        screen = "; ".join(f"{len(g)} by the rule {rule} ({years(g['T'])})" for rule, g in self.dropped(False).groupby("rule"))  # fmt: skip
        return cal or "none", screen or "none"


def n_dates(n: int) -> str:
    return f"{n} date" if n == 1 else f"{n} dates"


def years(values: Any) -> str:
    ts = [float(t) for t in values]
    return (", ".join(f"{t:.3f}" for t in ts) + "y") if ts else "none"


def djx_of(slot: Slot, slices: pd.DataFrame | None) -> Djx:
    r = slot.row if slot.priced else None
    horizon = float(slot.entry["T"]) if slot.entry is not None else _number(r, "T")
    below = None
    if r is not None and index_cells(r):
        below = sorted({float(c["T"]) for c in index_cells(r) if float(c["T"]) < horizon - 1e-9})
    last = float(r["index_last_slice"]) if r is not None and r.get("index_last_slice") is not None else None  # fmt: skip
    out = Djx(horizon, below, last, None)
    if slices is None:
        return out
    sub = slices[slices["date"] == slot.date]
    if len(sub) and (sub["status"] == "skipped").any():
        out.skipped = f"the specification build skipped the date ({sub['reason'].iloc[0]})"
    elif len(sub):
        out.build = sub.sort_values("T")
        column = f"spec_key_{slot.budget}"
        if r is not None and r.get("spec_key") and column in sub:
            out.key_match = str(sub[column].iloc[0]) == str(r["spec_key"])
    else:
        out.skipped = "the specification build has no row for the date"
    return out


def djx_statement(d: Djx, date: str, tenor: str) -> list[str]:
    """Section 1's sentences on the DJX slices of the check's date, from the specification
    build."""
    assert d.build is not None
    build = d.build
    kept, cal, screen = build[build["status"] == "kept"], d.dropped(True), d.dropped(False)
    rest = build[build["status"] == "not_selected"]
    lo, hi, _ = d.bracket()
    out = [
        f"On {date} at {tenor} (horizon {d.horizon:.3f}y) the loader lists {len(build)} DJX expiries: {len(kept)} are slices of the index surface, "
        f"{len(screen)} are dropped by the screen ({', '.join(f'{len(g)} by the rule {rule}' for rule, g in screen.groupby('rule')) or 'none'}), {len(cal)} by the calendar repair, "
        f"and {len(rest)} {'passes both and is' if len(rest) == 1 else 'pass both and are'} outside the surface's selection ({build['selection'].iloc[0]})."
    ]
    if len(cal):
        out.append(f"The calendar repair drops the DJX slices at {years(cal['T'])} (SVI rms {', '.join(f'{float(x):.2f}' for x in cal['svi_rms_vp'])} vp; all {'are' if cal['third_friday'].astype(str).eq('True').all() else 'are not'} third-Friday expiries).")  # fmt: skip
    else:
        out.append("The calendar repair drops no DJX slice.")
    if lo is not None and hi is not None:
        rms = {
            round(float(t), 9): float(x) for t, x in zip(kept["T"], kept["svi_rms_vp"], strict=True)
        }
        between = cal[(cal["T"].astype(float) > lo) & (cal["T"].astype(float) < hi)]
        out.append(
            f"The kept slices nearest the horizon are at {lo:.3f}y below and {hi:.3f}y above (SVI rms {rms[round(lo, 9)]:.2f} and {rms[round(hi, 9)]:.2f} vp): no kept DJX slice lies between {lo:.2f}y and {hi:.2f}y, "
            f"so the {d.horizon:.0f}y target is an interpolation between the {lo:.2f}y and the {hi:.2f}y slices"
            + (
                f"; the {len(between)} slices the repair drops lie between them."
                if len(between)
                else "."
            )
        )
        above = kept[kept["T"].astype(float) > d.horizon + 1e-9]
        if (
            len(cal)
            and len(above)
            and float(cal["svi_rms_vp"].max()) < float(above["svi_rms_vp"].min())
        ):
            out.append(
                f"The slices the repair drops are the better fitted ones (SVI rms {float(cal['svi_rms_vp'].min()):.2f} to {float(cal['svi_rms_vp'].max()):.2f} vp) and the slices it keeps above the horizon the worse fitted ones "
                f"({years(above['T'])}: SVI rms {float(above['svi_rms_vp'].min()):.2f} to {float(above['svi_rms_vp'].max()):.2f} vp): of a crossing pair the rule drops the shorter slice when both are third-Friday expiries. "
                "This is the opposite of what the earlier review recommended (SPEC 8.7, [review] LC4G-g: drop the long, badly fitted slices)."
            )
    elif hi is None:
        out.append(f"No kept DJX slice lies above the horizon: the target is extrapolated beyond {lo:.3f}y." if lo is not None else "No DJX slice is kept on either side of the horizon.")  # fmt: skip
    return out


def slice_records(b: Builder, slices: pd.DataFrame, path: Path) -> None:
    """One record per listed DJX expiry of the specification build: its maturity and, when the
    slice was fitted, its SVI rms."""
    for _, g in slices[slices["status"] != "skipped"].iterrows():
        status = g["status"] + (
            f" by the rule {g['rule']}: {g['reason']}" if g["status"] == "dropped" else ""
        )
        common = {
            "date": g["date"], "tenor": b.tenor, "budget": "no simulation: a specification-only build (scripts/pm_1y_slices.py), the same slices at both budgets",
            "commit": str(g.get("git_commit", "")), "source": f"{path} (inputs: the study's loader; fits: the runs' fit records)",
        }  # fmt: skip
        b.records.append(
            pc.record(
                f"{SECTION}.djx.{g['date']}.{g['expiry']}.T", SECTION, f"DJX expiry {g['expiry']}: maturity", float(g["T"]), None, unit="years",
                definition="the maturity of a listed DJX expiry of the date and what the specification build does with it (kept: a slice of the index surface; dropped: by the screen or by the calendar repair; not_selected: passes both, outside the surface's selection)",
                notes=f"{status}; third-Friday expiry: {g['third_friday']}; no standard error: an input, not an estimate", **common,
            )
        )  # fmt: skip
        if pd.notna(g["svi_rms_vp"]):
            b.records.append(
                pc.record(
                    f"{SECTION}.djx.{g['date']}.{g['expiry']}.svi_rms_vp", SECTION, f"DJX expiry {g['expiry']}: SVI rms", float(g["svi_rms_vp"]), None, unit="vol points",
                    definition="the root-mean-square error of the slice's own SVI fit on its fitted strikes, in vol points",
                    notes=f"{status}; {g['rms_source']}; no standard error: a fit's residual, not a Monte Carlo estimate", **common,
                )
            )  # fmt: skip


def budget_gap_3m(b: Builder) -> str:
    """The 3m fact on the budget: development minus production in LC/CC on the reference dates,
    from the main package's rows (one record per date); empty when a row is not there."""
    gaps = []
    commits: dict[str, set[str]] = {"development": set(), "production": set()}
    for date in pc.REFERENCE_DATES:
        paths = {k: d / f"{date}.json" for k, d in ROWS_3M.items()}
        if not all(p.exists() for p in paths.values()):
            return ""
        rows = {k: json.loads(p.read_text()) for k, p in paths.items()}
        if any(r.get("status") == "failed" or r.get("ratio") is None for r in rows.values()):
            return ""
        gap = float(rows["development"]["ratio"]) - float(rows["production"]["ratio"])
        se = float(rows["development"]["ratio_se"])
        gaps.append((gap, se))
        commits["development"].add(str(rows["development"].get("git_commit", "")))
        commits["production"].add(str(rows["production"].get("git_commit", "")))
        b.records.append(
            pc.record(
                f"{SECTION}.budget_3m.{date}.lc_over_cc_dev_minus_prod", SECTION, "3m: LC/CC at the development budget minus LC/CC at the production budget", gap, None, date=date, tenor="3m", unit="ratio",
                definition="E_LC[D]/E_CC[D] of the 3m development row (calendar repair on) minus that of the 3m production row, same date and seeds",
                budget=f"{pc.BUDGETS['development']} minus {pc.BUDGETS['production']}", commit=f"{rows['development'].get('git_commit', '')} (development), {rows['production'].get('git_commit', '')} (production)",
                source=f"{paths['development']}; {paths['production']}", notes=f"no standard error: the two runs share their seeds and their covariance is not in the rows; the rows' own ± of LC/CC are {se:.5f} (development) and {float(rows['production']['ratio_se']):.5f} (production)",
            )
        )  # fmt: skip
    lo, hi = min(g for g, _ in gaps), max(g for g, _ in gaps)
    times = [abs(g) / s for g, s in gaps]
    return (
        f"At 3m, development minus production in LC/CC is {lo:.4f} to {hi:.4f} on the {len(gaps)} reference dates, several times the printed error "
        f"({min(times):.1f} to {max(times):.1f} times the ± of the development rows; rows of the main package: `{ROWS_3M['development']}`, written at commit {', '.join(sorted(commits['development']))}, and `{ROWS_3M['production']}`, written at commit {', '.join(sorted(commits['production']))})."
    )


def budget_gap_12m(b: Builder, prod: Sequence[Slot], dev: Sequence[Slot]) -> str:
    """The same difference at the tenor of the page, on each date priced at both budgets."""
    out = []
    for p in prod:
        d = next((s for s in dev if s.date == p.date), None)
        if d is None or not (p.priced and d.priced) or p.row is None or d.row is None:
            continue
        gap = _number(d.row, "ratio") - _number(p.row, "ratio")
        b.records.append(
            pc.record(
                f"{SECTION}.budget.{p.date}.lc_over_cc_dev_minus_prod", SECTION, "LC/CC at the development budget minus LC/CC at the production budget", gap, None, date=p.date, tenor=b.tenor, unit="ratio",
                definition="E_LC[D]/E_CC[D] of the development row minus that of the production row, same date and seeds",
                budget=f"{pc.BUDGETS['development']} minus {pc.BUDGETS['production']}", commit=f"{d.commit} (development), {p.commit} (production)", source=f"{d.source}; {p.source}",
                notes=f"no standard error: the two runs share their seeds and their covariance is not in the rows; the rows' own ± of LC/CC are {_number(d.row, 'ratio_se'):.5f} (development) and {_number(p.row, 'ratio_se'):.5f} (production)",
            )
        )  # fmt: skip
        out.append(
            f"{p.date}: {gap:+.5f} (the rows' ± of LC/CC: {_number(d.row, 'ratio_se'):.5f} development, {_number(p.row, 'ratio_se'):.5f} production)"
        )
    if not out:
        return f"At {b.tenor} no date is priced at both budgets yet: the difference is not measured at {b.tenor}."
    return f"At {b.tenor}, development minus production in LC/CC on the dates priced at both budgets — {'; '.join(out)}."


def flag_line(slots: Sequence[Slot], short: bool = False) -> str:
    """One line under a table of a set of dates: which of its priced dates are flagged for the
    clipped mass (with the two masses) and which have gate numbers outside 0.15 vp; ``short``:
    the count and the dates not flagged only (the twenty dates)."""
    priced = [s for s in slots if s.priced and s.row]
    if not priced:
        return ""
    budget = priced[0].budget
    flagged = [s for s in priced if clip_flagged(s)]
    if short:
        clear = [s.date for s in priced if not clip_flagged(s)]
        n_out = sum(bool(_gate_outside(s.row, None)[0]) for s in priced if "idx_err_atm" in (s.row or {}))  # fmt: skip
        return (
            f"Flagged rows stay in this table: {len(flagged)} of the {len(priced)} priced {budget} dates are flagged for the clipped mass "
            f"(not flagged: {', '.join(clear) if clear else 'none'}); {n_out} have gate numbers outside 0.15 vp. The two masses are in table 3e, the gate numbers in table 3e-2, the flags in table 4a."
        )
    head = f"all {len(priced)}" if len(flagged) == len(priced) else f"{len(flagged)} of the {len(priced)}"  # fmt: skip
    text = f"Flagged rows stay in this table: {head} priced {budget} dates are flagged for the clipped mass"
    if flagged:
        text += " (at λ = 0 / at the cap, flagged above 0.01: " + "; ".join(f"{s.date} {_number(s.row, 'clip_low_inner_max'):.4f} / {_number(s.row, 'clip_high_inner_max'):.4f}" for s in flagged) + ")"  # fmt: skip
    outside = [s for s in priced if "idx_err_atm" in (s.row or {}) and _gate_outside(s.row, None)[0]]  # fmt: skip
    if outside:
        text += ". Gate numbers outside 0.15 vp: " + "; ".join(f"{s.date} ({gate_numbers(s)}; {'waived, the wing binds' if (s.row or {}).get('wing_binds') else 'not waived'})" for s in outside)  # fmt: skip
    else:
        text += ". Gate numbers outside 0.15 vp: none"
    return text + "."


def level_table(b: Builder, slots: Sequence[Slot], by_date: bool) -> None:
    """The second table of sections 2a and 3a: the index level behind the ratios to the copula."""
    columns = [(q.label, q) for q in LEVEL]
    if by_date:
        b.table(["date", *(c[0] for c in columns)], [[s.date, *(show(q, b.add(q, s)) for _, q in columns)] for s in slots])  # fmt: skip
    else:
        b.table(["quantity", *(s.date for s in slots)], [[label, *(show(q, b.add(q, s)) for s in slots)] for label, q in columns])  # fmt: skip


def level_sentence(b: Builder, slot: Slot, d: Djx) -> str:
    """Under 2a, for one production date: LC/copula split into LC/CC and CC/copula, and what
    the second factor carries (the index level of the model's target against the study's)."""
    r, e = slot.row, slot.entry
    if not slot.priced or r is None or e is None or horizon_cell(r, "+0.0") is None:
        return ""
    lc_cop, cc_cop, lc_cc = _number(r, "ED_lc") / _number(e, "P_D"), _number(r, "ED_cc") / _number(e, "P_D"), _number(r, "ratio")  # fmt: skip
    share = math.log(cc_cop) / math.log(lc_cop)
    b.records.append(
        pc.record(
            f"{SECTION}.{slot.group}.{slot.date}.log_share_cc_over_copula", SECTION, "share of ln(LC/copula) that is ln(CC/copula)", share, None, date=slot.date, tenor=b.tenor, unit="ratio",
            definition="ln(E_CC[D]/P_D) / ln(E_LC[D]/P_D): the part of the log gap between the LC forward and the copula's price that the constant-correlation companion already has",
            budget=f"{pc.BUDGETS.get(slot.budget, slot.budget)}; denominator: {pc.BUDGETS['study']}", commit=slot.commit, source=f"{slot.source}; {b.entries_path} (basket B1)",
            notes="; ".join(filter(None, ["no standard error: a ratio of logs of two ratios, none computed", flag_note(slot)])),
        )
    )  # fmt: skip
    atm, sig = _target_atm(r, e)[0], _study_atm(r, e)[0]
    n_rep = r.get("n_dropped_calendar_index") or 0
    low, high = _number(r, "clip_low_inner_max"), _number(r, "clip_high_inner_max")
    text = (
        f"{slot.date}: LC/copula = LC/CC × CC/copula ({lc_cop:.3f} = {lc_cc:.3f} × {cc_cop:.3f}); in logs {100 * share:.0f} % of the gap between LC and the copula is CC against the copula (ln {cc_cop:.3f} / ln {lc_cop:.3f}). "
        f"CC sits on the listed EQV (E_CC[V]/EQV {_number(r, 'EV_cc') / _number(r, 'EQV'):.3f}) while the copula's E[V] is {100 * (_number(e, 'EV') / _number(e, 'EQV') - 1):.0f} % above it (EV/EQV {_number(e, 'EV') / _number(e, 'EQV'):.3f}). "
        f"The model's {'repaired ' if n_rep else ''}index target has an at-the-money vol of {atm:.2f} % at the horizon against the study's {sig:.2f} % ({atm - sig:+.2f} vp); ρ_CC is {_number(r, 'rho_cc'):.3f} against ρ_cop {_number(e, 'rho_cop'):.3f}. "
    )
    if d.build is not None:
        lo, hi, _ = d.bracket()
        passed = d.build[(d.build["status"] != "dropped") | (d.build["rule"] == "calendar")]
        below, above = passed[passed["T"] <= d.horizon + 1e-9], passed[passed["T"] > d.horizon + 1e-9]  # fmt: skip
        if len(below) and len(above) and lo is not None and hi is not None:
            near = [below.iloc[-1], above.iloc[0]]
            gone = [g for g in near if g["rule"] == "calendar"]
            kept = f"the kept slices nearest the horizon are at {lo:.3f}y and {hi:.3f}y"
            if len(gone) == 2:
                text += f"The two DJX expiries bracketing the horizon ({float(near[0]['T']):.3f}y and {float(near[1]['T']):.3f}y) are dropped by the calendar repair (`tables/{SLICES_TABLE}.csv`); {kept}. "
            elif gone:
                text += f"The DJX expiry nearest the horizon at {float(gone[0]['T']):.3f}y is dropped by the calendar repair (`tables/{SLICES_TABLE}.csv`); {kept}. "
            else:
                text += (
                    f"The calendar repair drops neither DJX expiry bracketing the horizon; {kept}. "
                )
    outside = bool(_gate_outside(r, None)[0])
    text += (
        f"The date is {'flagged' if clip_flagged(slot) else 'not flagged'} for the clipped mass ({low:.4f} at λ = 0, {high:.4f} at the cap) and its gate numbers ({gate_numbers(slot)}) are "
        + (
            f"outside 0.15 vp ({'waived: the wing binds' if r.get('wing_binds') else 'not waived'})."
            if outside
            else "within 0.15 vp."
        )
    )
    return text


LEVEL_CLOSE = "Only LC/CC is a local-correlation effect; LC/copula and CC/copula also carry the gap between the model's index target and the study's index level."


def level_dev_sentence(b: Builder, dev: Sequence[Slot]) -> str:
    """Under 3a: the range of CC/copula over the priced dates and its correlation with the gap
    between the model's index target and the study's index level, with the two extremes."""
    points = []
    for s in dev:
        gap, cc = evaluate(LEVEL[2], s), evaluate(FORWARD[5], s)
        if gap.value is not None and cc.value is not None:
            points.append((s.date, gap.value, cc.value))
    if len(points) < 3:
        return ""
    frame = pd.DataFrame(points, columns=["date", "gap", "cc"])
    corr = float(frame["gap"].corr(frame["cc"]))
    lo, hi = frame.loc[frame["cc"].idxmin()], frame.loc[frame["cc"].idxmax()]
    commits = ", ".join(sorted({s.commit for s in dev if s.row and s.commit}))
    b.records.append(
        pc.record(
            f"{SECTION}.dev.level.corr_cc_over_copula_vs_atm_gap", SECTION, "correlation across the dates between CC/copula and (index target ATM vol minus the study's)", corr, None, tenor=b.tenor, unit="correlation",
            definition="the Pearson correlation, across the priced yearly dates, between E_CC[D]/P_D and the model's index target at-the-money vol at the horizon minus sig_B_DJX of the study's entry",
            budget=f"{pc.BUDGETS['development']}; {pc.BUDGETS['study']}", commit=commits, source=f"the development rows of section 3; {b.entries_path} (basket B1)", n=len(frame),
            notes="no standard error: a sample correlation across the dates",
        )
    )  # fmt: skip
    return (
        f"CC/copula runs from {lo['cc']:.3f} ({lo['date']}: target minus study {lo['gap']:+.2f} vp) to {hi['cc']:.3f} ({hi['date']}: {hi['gap']:+.2f} vp) over the {len(frame)} dates "
        f"and moves with the gap between the model's index target and the study's index level ({'the higher the target against the study, the lower CC/copula' if corr < 0 else 'the higher the target against the study, the higher CC/copula'}): the correlation across the {len(frame)} dates between CC/copula and (target ATM vol minus the study's) is {corr:.2f}. {LEVEL_CLOSE}"
    )


SECOND_MOMENT = (("κ_LC", "kappa_lc"), ("κ_CC", "kappa_cc"), ("E_LC[V]/EQV", "EV_over_EQV_lc"), ("E_CC[V]/EQV", "EV_over_EQV_cc"), ("ED_eqv", "ED_eqv"), ("single-name part", "EV_single_part"))  # fmt: skip


def second_moment_sentence(b: Builder, prod: Sequence[Slot], dev: Sequence[Slot]) -> str:
    """The ± of the second-moment quantities on a date priced at both budgets: the production
    row's against the development row's (one record per ratio); empty when no date has both."""
    by_key = {q.key: q for q in ALL}
    for p in prod:
        d = next((s for s in dev if s.date == p.date), None)
        if d is None or not (p.priced and d.priced) or p.row is None or d.row is None:
            continue
        times = _number(p.row, "n_paths") / _number(d.row, "n_paths")
        ref_p, ref_d = evaluate(by_key["ED_lc"], p).se, evaluate(by_key["ED_lc"], d).se
        pairs = []
        for label, key in SECOND_MOMENT:
            q = by_key[key]
            a, c = evaluate(q, p).se, evaluate(q, d).se
            if not a or not c:
                continue
            pairs.append(f"{label} ± {a:.3g} against ± {c:.3g} (ratio {a / c:.2f})")
            b.records.append(
                pc.record(
                    f"{SECTION}.budget.{p.date}.{key}.se_prod_over_dev", SECTION, f"{label}: the production row's ± over the development row's", a / c, None, date=p.date, tenor=b.tenor, unit="ratio",
                    definition=f"the standard error of the production row over that of the development row, same date and seeds, of: {q.definition}",
                    budget=f"{pc.BUDGETS['production']} over {pc.BUDGETS['development']}", commit=f"{p.commit} (production), {d.commit} (development)", source=f"{p.source}; {d.source}",
                    notes=f"no standard error: a ratio of two standard errors; with {times:.0f} times the paths an error that falls as 1/√paths would give {1 / math.sqrt(times):.2f}",
                )
            )  # fmt: skip
        if not pairs or not ref_p or not ref_d:
            continue
        return (
            f"κ, E[V]/EQV, ED_eqv and the single-name part are carried by a few paths, like the high-strike calls. On {p.date} the production row has {times:.0f} times the paths of the development row: an error that falls as 1/√paths would be {1 / math.sqrt(times):.2f} times the development row's "
            f"(E_LC[D]: ± {ref_p:.6f} against ± {ref_d:.6f}, ratio {ref_p / ref_d:.2f}). On these quantities, production against development: {'; '.join(pairs)}. Their ± is not a usable error."
        )
    return ""


D5_STRIKES = (("+0.0", "atm", "at the money"), ("-1.5", "m15", "−1.5 sd"), ("-2.5", "m25", "−2.5 sd"), ("+1.5", "p15", "+1.5 sd"), ("+2.5", "p25", "+2.5 sd"))  # fmt: skip


def target_change(
    b: Builder, on: Slot, old: Slot, on_slots: Sequence[Slot], date: str
) -> list[str]:
    """Section 1: the index target at the horizon with the decisions on against the old
    defaults (a table, one record per number) and what it means for the before/after; returns
    the sentences (empty when a row does not carry the cells)."""
    if on.row is None or old.row is None or not (on.priced and old.priced):
        return []
    rows, parts, vals = [], [], {}
    common = {"date": date, "tenor": b.tenor, "budget": pc.BUDGETS.get(on.budget, on.budget)}
    for strike, tag, label in D5_STRIKES:
        a, c = horizon_cell(on.row, strike), horizon_cell(old.row, strike)
        if a is None or c is None:
            return []
        ta, tc = 100 * _number(a, "target_vol"), 100 * _number(c, "target_vol")
        vals[strike] = (ta, tc)
        rows.append([label, f"{ta:.2f}", f"{tc:.2f}", f"{ta - tc:+.2f}", f"{_number(a, 'k'):+.4f}", f"{_number(c, 'k'):+.4f}"])  # fmt: skip
        parts.append(f"{label} {ta:.2f} % against {tc:.2f} % ({ta - tc:+.2f} vp)")
        what = f"the vol of the index target at the horizon at the strike {label} (in the run's own at-the-money sd; the target_vol of the row's index_errors list)"
        for name, value, unit, commit, source in (
            ("on", ta, "vol in %", on.commit, on.source),
            ("old", tc, "vol in %", old.commit, old.source),
            ("on_minus_old", ta - tc, "vol points", f"{on.commit} (on), {old.commit} (old defaults)", f"{on.source}; {old.source}"),
        ):  # fmt: skip
            b.records.append(
                pc.record(
                    f"{SECTION}.d5.{date}.target_vol.{tag}.{name}", SECTION, f"index target vol at the horizon, {label}: {name.replace('_', ' ')}", value, None, unit=unit,
                    definition=f"{what}: {'the row with decisions 1, 2 and 5 on' if name == 'on' else 'the row of the old defaults' if name == 'old' else 'decisions on minus the old defaults (each at its own strike)'}",
                    commit=commit, source=source, notes="no standard error: the calibration's target, an input of the model", **common,
                )
            )  # fmt: skip
    b.text(f"Table 1a-2. The {b.tenor} index target itself at the horizon, with the decisions on and at the old defaults ({on.budget} rows; the production row with the decisions on has the same target):")  # fmt: skip
    b.table(["strike", "target vol, decisions on (%)", "target vol, old defaults (%)", "difference (vol points)", "log-moneyness k, decisions on", "log-moneyness k, old defaults"], rows)  # fmt: skip
    out = [
        f"The repair changes the {b.tenor} target itself. Target vol at the horizon with the decisions on against the old defaults: {'; '.join(parts)} (each run's strikes are in its own at-the-money sd: table 1a-2 gives their log-moneyness).",
        "The clipped mass and LC/CC before and after are measured against different targets: the model follows its own target in both runs.",
    ]
    skew_on, skew_old = vals["-2.5"][0] - vals["+0.0"][0], vals["-2.5"][1] - vals["+0.0"][1]
    low_on, low_old = _number(on.row, "clip_low_inner_max"), _number(old.row, "clip_low_inner_max")
    if skew_old:
        ratio = skew_on / skew_old
        out.append(
            f"The mass clipped at λ = 0 goes from {low_old:.3f} to {low_on:.3f} ({on.budget} budget) against a target whose downside skew is {'about half as steep: ' if 0.4 <= ratio <= 0.6 else ''}{ratio:.2f} times that of the old defaults "
            f"(target vol at −2.5 sd minus at the money: {skew_on:.2f} vp with the decisions on, {skew_old:.2f} vp at the old defaults). It is not a better fit to the same target."
        )
    still = []
    for s in on_slots:
        if s.priced and s.row:
            lo, hi = _number(s.row, "clip_low_inner_max"), _number(s.row, "clip_high_inner_max")
            sides = "both sides" if lo > CLIP_FLAG_MASS and hi > CLIP_FLAG_MASS else "λ = 0 only" if lo > CLIP_FLAG_MASS else "the cap only" if hi > CLIP_FLAG_MASS else "neither side"  # fmt: skip
            still.append((s.budget, lo, hi, sides))
    if still:
        same = {x[3] for x in still}
        head = f"With the decisions on the date is {'still flagged on ' + still[0][3] if same != {'neither side'} and len(same) == 1 else 'flagged as follows'}" if same != {"neither side"} else "With the decisions on the date is not flagged"  # fmt: skip
        out.append(head + ": " + "; ".join(f"{bud} {lo:.4f} at λ = 0 and {hi:.4f} at the cap" + (f" ({sides})" if len(same) > 1 else "") for bud, lo, hi, sides in still) + f", against {CLIP_FLAG_MASS}.")  # fmt: skip
    uns_on, uns_old = on.row.get("n_names_unscreened") or 0, old.row.get("n_names_unscreened") or 0
    n_idx = on.row.get("n_dropped_calendar_index") or 0
    n_names = (on.row.get("n_dropped_calendar") or 0) - n_idx
    two = "decision 2 is inert (no name is kept unscreened in either run)" if not (uns_on or uns_old) else f"decision 2 acts ({uns_on} names kept unscreened with the decisions on: {on.row.get('names_unscreened', '')})"  # fmt: skip
    out.append(f"Which decisions act on this date: {two}; the change is decision 1 (the calendar repair drops {n_names} slices of the names: the row's n_dropped_calendar − n_dropped_calendar_index) plus decision 5 ({n_idx} DJX slices).")  # fmt: skip
    rel = {}
    for key, label in (("ED_cc", "E_CC[D]"), ("ED_lc", "E_LC[D]")):
        rel[key] = 100 * (_number(on.row, key) / _number(old.row, key) - 1)
        b.records.append(
            pc.record(
                f"{SECTION}.d5.{date}.{key}.rel_change_pct", SECTION, f"{label}: relative change, decisions on against the old defaults", rel[key], None, unit="%",
                definition=f"100 × ({label} of the row with decisions 1, 2 and 5 on / {label} of the row of the old defaults − 1), same budget and seeds",
                commit=f"{on.commit} (on), {old.commit} (old defaults)", source=f"{on.source}; {old.source}",
                notes="; ".join(filter(None, ["no standard error: the two runs share their seeds and their covariance is not in the rows", "each number is against its own run's target", flag_note(on)])), **common,
            )
        )  # fmt: skip
    line = f"The move of LC/CC from {_number(old.row, 'ratio'):.5f} to {_number(on.row, 'ratio'):.5f} ({_number(on.row, 'ratio') - _number(old.row, 'ratio'):+.5f}): E_CC[D] changes by {rel['ED_cc']:+.2f} % and E_LC[D] by {rel['ED_lc']:+.2f} %"
    if on.row.get("rho_cc") is not None and old.row.get("rho_cc") is not None:
        line += (
            f"; ρ_CC goes from {_number(old.row, 'rho_cc'):.3f} to {_number(on.row, 'rho_cc'):.3f}"
        )
    out.append(line + ".")
    return out


def carries(slot: Slot, d: Djx, frame: pd.DataFrame) -> str:
    """What a date of the left block of the summaries carries: its other flags, the kept DJX
    slices around its horizon, its index error at +2.5 sd and its Monte Carlo errors on κ and
    E[V]/EQV against the medians of the priced dates."""
    r = slot.row or {}
    parts = []
    names = str(d.build["names_extrapolated"].iloc[0]).replace(",", ", ") if d.build is not None and d.key_match else ""  # fmt: skip
    if (r.get("n_names_extrapolated") or 0) > 0:
        parts.append(f"{names or r.get('n_names_extrapolated')} priced beyond {'its' if r.get('n_names_extrapolated') == 1 else 'their'} last kept expiry")  # fmt: skip
    if (r.get("n_names_unscreened") or 0) > 0:
        parts.append(f"{r.get('names_unscreened')} kept unscreened")
    if (r.get("n_dropped_calendar_index") or 0) > 0:
        parts.append(f"DJX target repaired ({r.get('n_dropped_calendar_index')} slice{'s' if r.get('n_dropped_calendar_index') != 1 else ''} dropped)")  # fmt: skip
    if r.get("index_extrapolated"):
        parts.append("DJX target extrapolated")
    if r.get("status") != "ok":
        parts.append(f"status {r.get('status')}")
    lo, hi, origin = d.bracket()
    if origin == "specification build" and lo is not None and hi is not None:
        parts.append(
            f"no kept DJX slice between {lo:.2f}y and {hi:.2f}y (horizon {d.horizon:.2f}y)"
        )
    parts.append(f"index error at +2.5 sd at the horizon {show(INDEX_MORE[1], evaluate(INDEX_MORE[1], slot))} vp")  # fmt: skip
    priced = frame[frame["priced"] == "yes"]
    for q in (KAPPA[0], KAPPA[3]):
        cell = evaluate(q, slot)
        if cell.se is not None and f"{q.key}_se" in priced:
            parts.append(f"{q.label} {show(q, cell)} (median ± over the {len(priced)} priced dates: {float(priced[f'{q.key}_se'].median()):.4f})")  # fmt: skip
    return f"{slot.date}: " + "; ".join(parts)


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
            "date": s.date, "tenor": tenor, "budget": s.budget, "priced": PRICED[s.state],
            "status": s.row.get("status") if s.row else None, "reason": s.reason,
            "git_commit": s.commit, "source": s.source or None, "row_written": s.written or None,
            "flag_clip": clip_flagged(s),
        }  # fmt: skip
        for q in ALL:
            cell = evaluate(q, s)
            rec[q.key] = cell.value
            if not q.no_se:
                rec[f"{q.key}_se"] = cell.se
            if q.annotate is not None:
                # the short mark of an index-error cell (empty: a usable cell)
                rec[f"{q.key}_mark"] = cell.mark or None
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


def clip_range(priced: pd.DataFrame) -> str:
    """The clipped masses of the priced dates as ranges in percent (at λ = 0; at the cap)."""
    low, high = 100 * priced["clip_low_inner_max"], 100 * priced["clip_high_inner_max"]
    return f"at the cap: {high.min():.1f} % to {high.max():.1f} %; at λ = 0: {low.min():.1f} % to {low.max():.1f} %"


def summaries(
    b: Builder, slots: Sequence[Slot], frame: pd.DataFrame, rows_dir: Path, csv: Path
) -> pd.DataFrame:
    """Section 3's summaries over the dates not flagged for the clipped mass and over every
    priced date; returns the CSV behind the table.  A count is given as its total over the
    dates; an exact constant is not summarised."""
    priced = frame[frame["priced"] == "yes"]
    flagged = priced[priced["flag_clip"].eq(True)]
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
    all_flagged = len(priced) > 0 and len(flagged) == len(priced)
    none = f"none: all {len(priced)} priced dates are flagged for the clipped mass ({clip_range(priced)})" if all_flagged else ""  # fmt: skip
    if none and n_pending:
        none += f"; {n_pending} of {len(slots)} dates pending"
    out, rows = [], []
    for q in ALL:
        if not q.summarise:
            continue
        line = [q.label]
        # the priced rows do not carry the column (the deltas of rows run with risk = none)
        not_run = any(evaluate(q, s).note.startswith("not run") for s in slots if s.priced)
        for sample, (label, sub) in samples.items():
            empty = "not run" if not_run else (f"none: all {len(priced)} priced dates are flagged" if none and sample == "unflagged" else ("pending" if n_pending else "n/a"))  # fmt: skip
            if q.kind == "int":
                # a count: its total over the dates and the number of dates that carry it
                x = sub[q.key].dropna().astype(float) if q.key in sub else pd.Series(dtype=float)
                total = float(x.sum()) if len(x) else None
                out.append({"quantity": q.key, "label": q.label, "sample": sample, "n": len(x), "total": total, "n_dates_positive": int((x > 0).sum())})  # fmt: skip
                b.records.append(
                    pc.record(
                        f"{SECTION}.dev.summary.{q.key}.total.{sample}", SECTION, f"{q.label}: total over {label}", total, None, tenor=b.tenor, unit=q.unit,
                        definition=f"the sum over {label} (yearly dates, first monthly entry of each year 2007–2026) of: {q.definition}",
                        budget=pc.BUDGETS["development"], commit=commits, source=f"{csv} (from {rows_dir})", n=len(x),
                        notes="; ".join(filter(None, [none if sample == "unflagged" and not len(x) else "", partial if len(x) else "", f"{len(flagged)} of {len(slots)} dates flagged for the clipped mass ({len(priced)} priced): flagged rows are in this total" if sample == "all" and len(x) else "", f"{int((x > 0).sum())} dates with at least one", "no standard error: a count"])),
                    )
                )  # fmt: skip
                line += [str(len(x)), f"total {total:.0f} (on {n_dates(int((x > 0).sum()))})" if len(x) else empty, "", "", ""]  # fmt: skip
                continue
            # a study-origin quantity is summarised on the same dates as the model's
            stats = summarise(sub[q.key]) if q.key in sub else summarise(pd.Series(dtype=float))
            n = int(stats["n"] or 0)
            out.append({"quantity": q.key, "label": q.label, "sample": sample, **stats})
            for stat, stat_label in STATS:
                notes = []
                if stats[stat] is None:
                    if not_run:
                        notes.append("not run: the priced rows do not carry the column")
                    elif none and sample == "unflagged":
                        notes.append(none)
                    elif n_pending:
                        notes.append(f"pending: waits for the development rows in {rows_dir}")
                    else:
                        notes.append("no date in the sample has this quantity")
                elif partial:
                    notes.append(partial)
                if sample == "all" and stats[stat] is not None:
                    notes.append(f"{len(flagged)} of {len(slots)} dates flagged for the clipped mass ({len(priced)} priced): flagged rows are in this summary")  # fmt: skip
                notes += key_notes(q)
                if stat != "mean":
                    notes.append(
                        "an order statistic over the dates (linear interpolation): no standard error"
                    )
                elif stats["mean"] is not None and stats["se"] is None:
                    notes.append("no standard error: one date in the sample")
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
                line += ["0", empty, "", "", ""]
            else:
                d = q.digits
                line += [
                    str(n),
                    pc.pm(stats["mean"], stats["se"], d),
                    *(pc.pm(stats[k], None, d) for k in ("q25", "median", "q75")),
                ]
        rows.append(line)
    heads = ["quantity"]
    for sample in ("not flagged", f"all priced dates: {len(priced)} of {len(slots)}; {len(flagged)} of them flagged"):  # fmt: skip
        heads += [f"n ({sample})", "mean ± se", "q25", "median", "q75"]
    b.table(heads, rows)
    return pd.DataFrame(out)


def other_flags(slots: Sequence[Slot]) -> str:
    """For information under the summaries: the priced dates that carry a flag other than the
    clip flag (they are not excluded from the left block)."""

    def dates(test: Callable[[dict[str, Any]], bool]) -> str:
        hit = [s.date for s in slots if s.priced and s.row and test(s.row)]
        return f"{len(hit)} ({', '.join(hit)})" if hit else "none"

    return (
        f"a name kept unscreened: {dates(lambda r: (r.get('n_names_unscreened') or 0) > 0)}; "
        f"a name beyond its last kept expiry: {dates(lambda r: (r.get('n_names_extrapolated') or 0) > 0)}; "
        f"DJX target repaired: {dates(lambda r: (r.get('n_dropped_calendar_index') or 0) > 0)}; "
        f"DJX target extrapolated: {dates(lambda r: bool(r.get('index_extrapolated')))}; "
        f"status check: {dates(lambda r: r.get('status') != 'ok')}"
    )


def decision5(
    b: Builder,
    variants: Sequence[tuple[str, Slot | None, str]],
    date: str,
    compare: tuple[Slot, Slot] | None,
) -> pd.DataFrame:
    """Section 1: the clipped mass by side, the forward, the index errors and the DJX slices on
    the check's date, per variant (``(label, slot, text when there is no row)``), and the
    difference ``compare[0] − compare[1]`` (decisions on minus the old defaults, same budget)."""
    quantities = [*CLIP, FLAGS[5], FLAGS[6], GATE[0], FORWARD[0], FORWARD[1], FORWARD[3], LEVEL[3], LEVEL[0], *INDEX[:5], *INDEX_MORE, FLAGS[2], FLAGS[3], FLAGS[1], FLAGS[0], FLAGS[4], INDEX[5]]  # fmt: skip
    rows, out = [], []
    for q in quantities:
        line = [q.label]
        for label, slot, missing in variants:
            if slot is None:
                line.append(missing)
                continue
            cell = evaluate(q, slot)
            line.append(show(q, cell))
            out.append({"variant": label, "date": date, "quantity": q.key, "value": cell.value, "se": cell.se, "cell": cell.state, "budget": slot.budget, "git_commit": slot.commit, "source": slot.source or slot.waits_for})  # fmt: skip
            if slot.group == "before":
                b.add(q, slot)
        if compare is None:
            line.append("not run")
        elif q.kind != "float":
            line.append("")
        else:
            on, old = compare
            a, c = evaluate(q, on), evaluate(q, old)
            over_slices = q.key.startswith("idx_err_max")
            against_target = q.key.startswith("idx_err") and not over_slices
            if over_slices:
                # the two maxima are over different sets of slices: no difference
                line.append("n/a (maxima over different sets of slices)")
                diff = None
            elif a.value is None or c.value is None:
                line.append("pending" if "pending" in (a.state, c.state) else "n/a")
                diff = None
            else:
                diff = a.value - c.value
                line.append(f"{diff:+.{q.digits}f}")
            out.append({"variant": "decisions on minus old defaults", "date": date, "quantity": q.key, "value": diff, "se": None, "cell": "value" if diff is not None else "pending", "budget": on.budget, "git_commit": f"{on.commit}; {old.commit}", "source": f"{on.source or on.waits_for}; {old.source}"})  # fmt: skip
            b.records.append(
                pc.record(
                    f"{SECTION}.d5.{date}.{q.key}.on_minus_old", SECTION, f"{q.label}: decisions 1, 2 and 5 on minus the old defaults", diff, None, date=date, tenor=b.tenor, unit=q.unit,
                    definition=f"the {on.budget}-budget row with decisions 1, 2 and 5 on minus the {old.budget}-budget row of the old defaults (no calendar repair of the names or of DJX, no unscreened fallback), of: {q.definition}",
                    budget=pc.BUDGETS.get(on.budget, on.budget), commit=f"{on.commit} (on), {old.commit} (old defaults)", source=f"{on.source or on.waits_for}; {old.source}",
                    notes="; ".join(filter(None, [
                        "n/a: the two maxima are over different sets of slices (the kept DJX slices of the two runs differ), so no difference is given" if over_slices
                        else "no standard error: the two runs share the particle and pricing seeds and their covariance is not in the rows; each row's own ± is in its column" if diff is not None
                        else (a.note if a.state != "value" else c.note if c.state != "value" else "") or "pending",
                        "each error is against its own run's target" if against_target and diff is not None else "",
                        flag_note(on),
                    ])),
                )
            )  # fmt: skip
        rows.append(line)
    b.table(["quantity", *(v[0] for v in variants), "decisions on minus old defaults, same budget"], rows)  # fmt: skip
    return pd.DataFrame(out)


def gate_table(b: Builder, slots: Sequence[Slot], by_date: bool) -> None:
    """The second table of sections 2e and 3e: the index gate and the index errors beyond the
    money and the downside at the horizon."""
    columns: list[tuple[str, Callable[[Slot], str]]] = [
        (GATE[0].label, lambda s: show(GATE[0], b.add(GATE[0], s))),
        ("index gate: error ATM", lambda s: show(INDEX[0], evaluate(INDEX[0], s))),
        ("index gate: error at 90 %", lambda s: show(INDEX[1], evaluate(INDEX[1], s))),
        (GATE[1].label, lambda s: show(GATE[1], b.add(GATE[1], s))),
        (GATE[2].label, lambda s: show(GATE[2], b.add(GATE[2], s))),
        *((q.label, lambda s, q=q: show(q, b.add(q, s))) for q in INDEX_MORE),
    ]
    if by_date:
        b.table(["date", *(c[0] for c in columns)], [[s.date, *(fn(s) for _, fn in columns)] for s in slots])  # fmt: skip
    else:
        b.table(["quantity", *(s.date for s in slots)], [[label, *(fn(s) for s in slots)] for label, fn in columns])  # fmt: skip


def flags_section(b: Builder, slots: Sequence[Slot], slices: pd.DataFrame | None) -> pd.DataFrame:
    """Section 4: the flags of every date of both sets (a column the row does not carry reads
    ``n/a``: it can only happen under ``--no-strict``)."""

    def yes_no(flag: bool | None, detail: str = "") -> str:
        if flag is None:
            return "n/a"
        return (f"yes ({detail})" if detail else "yes") if flag else "no"

    rows, out = [], []
    for s in slots:
        r = s.row or {}
        d = djx_of(s, slices)
        rec: dict[str, Any] = {"set": s.group, "date": s.date, "budget": s.budget, "priced": PRICED[s.state], "status": r.get("status"), "reason": s.reason, "row_written": s.written or None, "git_commit": s.commit}  # fmt: skip
        if not s.priced:
            word = "pending" if s.state == "pending" else "failed"
            rows.append([s.budget, s.date, s.written or "—", word if s.state == "pending" else f"failed: {r.get('reason', '')}", *[word] * 10])  # fmt: skip
            out.append(rec)
            continue

        def val(key: str, r: dict[str, Any] = r) -> float | None:
            return float(r[key]) if r.get(key) is not None else None

        n_ext, n_rep, n_uns = val("n_names_extrapolated"), val("n_dropped_calendar_index"), val("n_names_unscreened")  # fmt: skip
        low, high = _number(r, "clip_low_inner_max"), _number(r, "clip_high_inner_max")
        idx_ext = None if r.get("index_extrapolated") is None else bool(r["index_extrapolated"])
        flagged = clip_flagged(s)
        binds = None if r.get("wing_binds") is None else bool(r["wing_binds"])
        outside = bool(_gate_outside(r, None)[0]) if "idx_err_atm" in r and "idx_err_90" in r else None  # fmt: skip
        lo, hi, origin = d.bracket()
        names = str(d.build["names_extrapolated"].iloc[0]).replace(",", ", ") if d.build is not None and d.key_match else ""  # fmt: skip
        rec.update(
            name_beyond_last_kept_expiry=None if n_ext is None else n_ext > 0, n_names_extrapolated=n_ext,
            names_extrapolated_specification_build=names or None,
            index_extrapolated=idx_ext, index_last_slice=r.get("index_last_slice"), T=r.get("T"),
            index_repaired=None if n_rep is None else n_rep > 0, n_dropped_calendar_index=n_rep,
            name_unscreened=None if n_uns is None else n_uns > 0, n_names_unscreened=n_uns,
            names_unscreened=r.get("names_unscreened", ""), clip_low_inner_max=low, clip_high_inner_max=high,
            flag_clip_low=low > CLIP_FLAG_MASS, flag_clip_high=high > CLIP_FLAG_MASS, flag_clip=flagged,
            wing_binds=binds, idx_err_atm=val("idx_err_atm"), idx_err_atm_se=val("idx_err_atm_se"), idx_err_90=val("idx_err_90"),
            idx_err_90_se=val("idx_err_90_se"), index_gate_outside=outside, check_index=r.get("check_index"),
            checks_failing=",".join(c for c in GATING if r.get(c) is False), columns_not_finite=",".join(nonfinite_columns(r)),
            djx_kept_below_horizon_row=years(d.row_below) if d.row_below is not None else None,
            djx_nearest_kept_below=lo, djx_nearest_kept_above=hi, djx_nearest_source=origin or None,
            specification_build_key_is_the_rows=d.key_match,
        )  # fmt: skip
        out.append(rec)
        status = str(r.get("status")) + (f" ({r.get('reason')})" if r.get("reason") else "")
        last = f"last slice {_number(r, 'index_last_slice'):.3f}y, T {_number(r, 'T'):.3f}y" if "index_last_slice" in r else ""  # fmt: skip
        beyond = (f"{n_ext:.0f}" + (f": {names}" if names else "")) if n_ext else ""
        rows.append(
            [
                s.budget,
                s.date,
                s.written,
                status,
                yes_no(rec["name_beyond_last_kept_expiry"], beyond),
                yes_no(idx_ext, last),
                yes_no(rec["index_repaired"], f"{n_rep:.0f}" if n_rep else ""),
                yes_no(rec["name_unscreened"], str(r.get("names_unscreened", ""))),
                yes_no(binds),
                gate_numbers(s),
                yes_no(outside),
                f"{'yes' if low > CLIP_FLAG_MASS else 'no'} ({low:.4f})",
                f"{'yes' if high > CLIP_FLAG_MASS else 'no'} ({high:.4f})",
                "yes" if flagged else "no",
            ]
        )
    b.table(
        ["budget", "date", "row written", "status", "a name beyond its last kept expiry (count)", "DJX target extrapolated", "DJX target repaired", "a name kept unscreened", "wing binds", "index gate (vol points)", "outside 0.15 vp", "clip at λ = 0 > 1 %", "clip at the cap > 1 %", "flagged (clip)"],
        rows,
    )  # fmt: skip
    return pd.DataFrame(out)


def djx_section(
    b: Builder, slots: Sequence[Slot], slices: pd.DataFrame | None, slices_path: Path
) -> None:
    """Table 4b: per date, the kept DJX slices up to the horizon (the row), the kept slices on
    each side of the horizon and the slices dropped (the specification build, when its table is
    there), with one record per date for each side."""
    rows = []
    for s in slots:
        d = djx_of(s, slices)
        r = s.row if s.priced else None
        n_rep = float(r["n_dropped_calendar_index"]) if r and r.get("n_dropped_calendar_index") is not None else None  # fmt: skip
        cal, screen = d.dropped_text(n_rep)
        lo, hi, origin = d.bracket()
        for side, value in (("below", lo), ("above", hi)):
            from_build = origin == "specification build" and d.build is not None
            source = f"{slices_path} (scripts/pm_1y_slices.py)" if from_build else (s.source or s.waits_for)  # fmt: skip
            commit = str(d.build["git_commit"].iloc[0]) if from_build and d.build is not None else s.commit  # fmt: skip
            b.records.append(
                pc.record(
                    f"{SECTION}.{s.group}.{s.date}.djx_nearest_kept_{side}", SECTION, f"the kept DJX slice nearest the horizon, {side}", value, None, date=s.date, tenor=b.tenor, unit="years",
                    definition=f"the maturity of the kept DJX slice nearest the horizon {side} it (a slice at the horizon counts as below); from the specification build when its table has the date, else from the row (the T values of its index_errors list, below the horizon only)",
                    budget="no simulation: a specification-only build" if from_build else pc.BUDGETS.get(s.budget, s.budget), commit=commit, source=source,
                    notes="no standard error: an input, not an estimate" if value is not None else ("pending: no row and no specification build" if not origin else ("not known from the row: the specification build has no list for the date" if origin == "row" and side == "above" else f"none: no kept slice {side} the horizon")),
                )
            )  # fmt: skip
        match = "row pending" if r is None else ("n/a" if d.key_match is None else ("yes" if d.key_match else "no: the list is that of the current defaults, not of the row"))  # fmt: skip
        rows.append([s.budget, s.date, f"{d.horizon:.3f}y", years(d.row_below) if d.row_below is not None else ("pending" if r is None else "not carried by the row"), d.bracket_text(), cal, screen, match])  # fmt: skip
    b.table(
        ["budget", "date", "horizon", "kept DJX slices below the horizon (the row)", "nearest kept slice below / above the horizon", "dropped by the calendar repair: T (SVI rms; reason)", "dropped by the screen", "the build's specification key is the row's"],
        rows,
    )  # fmt: skip


def checks_section(b: Builder, slots: Sequence[Slot]) -> None:
    """Under the flags: for each priced row whose status is not ok, which check and why."""
    lines = [f"- {s.budget} {s.date}, status {s.row.get('status')}: {check_detail(s, [x for x in slots if x.group == s.group])}" for s in slots if s.priced and s.row and s.row.get("status") != "ok"]  # fmt: skip
    if lines:
        b.text("Rows whose status is check — which check, and what it touches:", *lines)
    elif any(s.priced for s in slots):
        b.text("No priced row has a status other than ok.")


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
                worst, None, tenor=b.tenor, unit=CHECK_UNITS.get(row_key, "notional"), definition=f"the largest absolute difference between the row's {row_key} and the entry's {entry_key} (basket B1) over the priced rows of both budgets",
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


def builder_commit() -> str:
    """The commit of the worktree the builder runs from (``git rev-parse --short HEAD``, read
    only) and whether the two scripts of the addendum differ from it."""
    here = Path(__file__).resolve().parent

    def git(*args: str) -> str:
        try:
            out = subprocess.run(["git", *args], cwd=here, capture_output=True, text=True, timeout=10, check=False)  # fmt: skip
        except (OSError, subprocess.SubprocessError):
            return ""
        return out.stdout.strip() if out.returncode == 0 else ""

    commit = git("rev-parse", "--short", "HEAD")
    if not commit:
        return "Builder: `scripts/pm_1y.py`; commit not available (git rev-parse failed)."
    dirty = git("status", "--porcelain", "--", "pm_1y.py", "pm_1y_slices.py")
    state = "the working copies of `scripts/pm_1y.py` / `scripts/pm_1y_slices.py` differ from that commit (not committed)" if dirty else "`scripts/pm_1y.py` and `scripts/pm_1y_slices.py` are as committed"  # fmt: skip
    return f"Builder: `scripts/pm_1y.py` in the worktree at commit {commit} (`git rev-parse --short HEAD`); {state}. The rows carry their own commit (the pricer's), given under each table."


def old_defaults_config(row: dict[str, Any], tenor: str) -> str:
    """Whether the 'before' row's configuration digest is that of ``lcm_norepair.yaml`` (the
    digest of ``scripts/lcm_price.py``: the YAML, the tenor and the budget)."""
    path = CONFIG.with_name("lcm_norepair.yaml")
    if not path.exists() or not row.get("config_digest"):
        return ""
    doc = {"config": yaml.safe_load(path.read_text()), "tenor": tenor, "budget": row.get("budget")}
    digest = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]
    same = "is" if digest == row["config_digest"] else "is not"
    return f"The row's configuration digest ({row['config_digest']}) {same} that of `{path}` at the {row.get('budget')} budget (lcm.yaml with calendar_repair and unscreened_fallback off)."


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
    prod_dir, dev_dir = Path(args.rows_dir), Path(args.development_rows_dir)
    prod_table_path, dev_table_path = Path(args.production_table), Path(args.development_table)
    prod_table, dev_table = read_rows_table(prod_table_path), read_rows_table(dev_table_path)
    prod = [load_slot("prod", d, "production", tenor, prod_dir, prod_table, prod_table_path, entries, sizes, strict) for d in pc.REFERENCE_DATES]  # fmt: skip
    dev = [load_slot("dev", d, "development", tenor, dev_dir, dev_table, dev_table_path, entries, sizes, strict) for d in dev_dates]  # fmt: skip
    b = Builder(tenor, entries_path)
    tables: dict[str, pd.DataFrame] = {"1y_dates": dates}
    check_date = pc.REFERENCE_DATES[0]
    slices, slices_path = read_slices(base, args.slices_table)
    n_dev_priced = sum(s.priced for s in dev)

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
    if pc.STUDY.is_symlink():
        b.text(f"`{pc.STUDY}` is a link to the dispersion study's own folder (`{pc.STUDY.resolve()}`), read only: the study's table `entries_{tenor}.parquet` and its entries are read from there.")  # fmt: skip
    if not strict:
        b.text("**TEST RENDER (`--no-strict`): rows that do not carry the owner's decisions 1, 2 and 5 are accepted. Not for the package.**")  # fmt: skip
    for year in dates[dates["date"] == ""]["year"]:
        b.text(f"Year {year}: no monthly {tenor} entry of basket B1 with its pickle — no date for that year.")  # fmt: skip
    rejected = [s for s in (*prod, *dev) if s.problems]
    if rejected:
        b.text("Rows present but not of the specification asked for, written as pending: " + "; ".join(f"{s.budget} {s.date} ({', '.join(s.problems)})" for s in rejected) + ".")  # fmt: skip
    first_at = len(b.md)  # where the block "Read this first" goes, once the page is built
    se_3m = budget_gap_3m(b)
    se_12m = budget_gap_12m(b, prod, dev)
    second = second_moment_sentence(b, prod, dev)
    se_note = " ".join(filter(None, [PM_SE, se_3m or "The 3m rows of the main package that measure the budget's effect are not all there: no figure is quoted.", se_12m, second]))  # fmt: skip
    caveat = call_caveat()

    # --- 1. decision 5
    b.text(
        f"## 1. Decision 5 (the names' calendar repair on the DJX surface): the check on {check_date} at {tenor}"
    )
    after_prod = next(s for s in prod if s.date == check_date)
    after_dev = next((s for s in dev if s.date == check_date), None)
    before: Slot | None = None
    # the rows of the old defaults: the folder given, else the production one, else the
    # development one (the first that holds the date)
    before_dirs = [Path(p) for p in args.before_rows_dirs]
    before_path = next((d / f"{check_date}.json" for d in before_dirs if (d / f"{check_date}.json").exists()), before_dirs[0] / f"{check_date}.json")  # fmt: skip
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
    old_label = "old defaults: no calendar repair of the names or of DJX, no unscreened fallback (decisions 1, 2 and 5 off)"
    variants.append((old_label + (f", {before.budget} budget, commit {before.commit}" if before else ""), before, "not run"))  # fmt: skip
    compare: tuple[Slot, Slot] | None = None
    if before is not None and before.priced:
        same = after_dev if before.budget == "development" else after_prod
        compare = (same, before) if same is not None else None
    b.text(f"### 1a. Before and after on {check_date}: the clipped mass, the forward, the index errors and the target")  # fmt: skip
    tables["1y_decision5"] = decision5(b, variants, check_date, compare)
    max_note = ""
    if compare is not None and compare[0].priced:
        worst = max(sd_cells(before.row if before else None, 2.5), key=lambda c: abs(_number(c, "error_vp")), default=None)  # fmt: skip
        if worst is not None:
            kept_on = {round(float(c["T"]), 9) for c in index_cells(compare[0].row)}
            gone = round(float(worst["T"]), 9) not in kept_on
            max_note = f"No difference is printed for the largest errors over the slices: the two maxima are over different sets of slices (at the old defaults the largest inside ±2.5 sd sits at {float(worst['T']):.3f}y, a slice the run with the decisions on {'no longer has' if gone else 'also has'}). "
    d5 = [f"Clipped mass: the share of the particles that are inside ±2.5 at-the-money sd of the basket and whose λ is clipped at 0, resp. at the cap, the largest over the calibration's slices (fractions; flagged above 0.01); E_LC[D], E_CC[D] in fractions of the notional; index errors in vol points (model minus target at the horizon; the largest over the kept slices up to the horizon); DJX slices dropped by the repair: n_dropped_calendar_index; last DJX slice: index_last_slice, in years (horizon {float(entries.loc[check_date, 'T']):.4f}y)."]  # fmt: skip
    if before is not None:
        d5.append(
            f"The last column is the {compare[0].budget if compare else before.budget}-budget column with the decisions on minus the old-defaults column (same budget, same seeds). "
            "The before/after is decisions 1, 2 and 5 together, not decision 5 alone: the old-defaults run has the calendar repair of the names, the calendar repair of DJX and the unscreened fallback all off. "
            "A difference has no standard error (the two runs share their seeds; their covariance is not in the rows): each row's own ± is in its column. "
            "The differences of the index errors at the horizon are differences of errors each measured against its own run's target. "
            + max_note
            + old_defaults_config(before.row or {}, tenor)
        )
    d5.append("Production column — " + footer([after_prod], False, entries_path, prod_dir))
    if after_dev is not None:
        d5.append("Development column — " + footer([after_dev], False, entries_path, dev_dir))
    d5.append(f"Old defaults: {'`' + before.source + '`' if before else f'not run at {tenor} (no row of the old defaults at ' + ' or '.join(f'`{d}/{check_date}.json`' for d in before_dirs) + ')'}.")  # fmt: skip
    b.text(" ".join(d5))
    target_lines = target_change(b, compare[0], compare[1], [s for s in (after_prod, after_dev) if s is not None], check_date) if compare is not None else []  # fmt: skip
    if target_lines:
        b.text(" ".join(target_lines) + f" Sources: the two {compare[0].budget if compare else ''}-budget rows of table 1a (their index_errors lists at the horizon, clip_low_inner_max, clip_high_inner_max, n_dropped_calendar, n_dropped_calendar_index, n_names_unscreened, ED_cc, ED_lc, ratio, rho_cc) and, for the production figures, the production row.")  # fmt: skip
    elif before is not None:
        b.text("The target with the decisions on against the old defaults: pending (it needs the row with the decisions on at the budget of the old-defaults row).")  # fmt: skip
    # the DJX slices on the check's date: the specification build, and what the rows give
    b.text(f"### 1b. The DJX slices on {check_date}: what the repair drops and what brackets the horizon")  # fmt: skip
    djx_check = djx_of(
        after_dev if after_dev is not None and after_dev.priced else after_prod, slices
    )
    headline = ""
    if djx_check.build is not None:
        statement = djx_statement(djx_check, check_date, tenor)
        headline = " ".join(statement[1:])
        b.text(" ".join(statement))
        b.table(
            ["DJX expiry", "T (years)", "third Friday", "in the specification", "rule", "reason", "SVI rms (vol points)"],
            [[g["expiry"], f"{float(g['T']):.3f}", "yes" if str(g["third_friday"]) == "True" else "no", g["status"].replace("_", " "), g["rule"] or "—", g["reason"] or "—",
              f"{float(g['svi_rms_vp']):.2f} ({g['rms_source']})" if pd.notna(g["svi_rms_vp"]) else "not fitted (dropped by the screen)"] for _, g in djx_check.build.iterrows()],
        )  # fmt: skip
        keys = [f"{s.budget}: {'yes' if djx_of(s, slices).key_match else 'no'}" for s in (after_prod, after_dev) if s is not None and s.priced]  # fmt: skip
        odd = [(str(g["expiry"]), dt.date.fromisoformat(str(g["expiry"])).strftime("%A")) for _, g in djx_check.build.iterrows() if str(g["third_friday"]) == "True"]  # fmt: skip
        odd = [(e, day) for e, day in odd if day != "Friday"]
        weekday_note = "".join(f"{e} is a {day}: the third-Friday rule counts it as the standard monthly expiry of its month (the Thursday before, or the Saturday after, a third Friday that is not itself among the listed DJX expiries). " for e, day in odd)  # fmt: skip
        b.text(
            f"Source: `{slices_path}`, a specification-only build under the current defaults (`{CONFIG}`) by `scripts/pm_1y_slices.py` run from the worktree at commit {djx_check.build['git_commit'].iloc[0]}, built {djx_check.build['built'].iloc[0]} New York: no calibration, no Monte Carlo; the same slices at both budgets. "
            "SVI rms: the root-mean-square error of the slice's own SVI fit in vol points (fit record: read from the runs' fit records; fitted here: the runs never fitted the slice). "
            + weekday_note
            + (
                f"The build's specification key is the row's — {'; '.join(keys)}. "
                if keys
                else f"No row with the decisions on is priced on {check_date} yet: the build's specification key is not compared with a row. "
            )
        )
    else:
        b.text(f"The list of the dropped DJX slices is not available: {djx_check.skipped or f'no table of the specification build at `{slices_path}` (run `scripts/pm_1y_slices.py`)'}. What the rows give is below.")  # fmt: skip
    known = []
    for label, slot, _ in variants:
        if slot is not None and slot.priced:
            d = djx_of(slot, None)
            name = f"old defaults, {slot.budget} budget" if slot.group == "before" else label
            known.append(f"{name}: kept below the horizon {years(d.row_below or [])}" + (f"; last kept slice {d.last:.3f}y" if d.last is not None else ""))  # fmt: skip
    b.text("From the rows (the T values of the row's index_errors list are the kept DJX slices below the horizon; index_last_slice is the last kept slice) — " + ("; ".join(known) if known else "no row yet") + ".")  # fmt: skip

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
        ("2e. Clipped mass, index errors, names priced beyond their last kept expiry", [*CLIP, *INDEX], "calibration", False),
        ("2f. Sticky-strike deltas", DELTAS, "deltas", False),
    ]  # fmt: skip
    prod_flags = flag_line(prod)
    for title, quantities, key, study in plan:
        b.text(f"### {title}")
        by_quantity(b, prod, quantities)
        extra = (caveat + " " if key == "calls" else "") + (second + " " if second and key in ("listed", "kappa", "split") else "")  # fmt: skip
        b.text(f"{DEFINITIONS[key]} {extra}{PM_SE} {footer(prod, study, entries_path, prod_dir)}")
        if key != "calibration" and prod_flags:
            b.text(prod_flags)
        if key == "forward":
            b.text("Table 2a-2. The index level behind the ratios to the copula:")
            level_table(b, prod, by_date=False)
            worst = min((s for s in prod if s.priced), key=lambda s: evaluate(FORWARD[4], s).value or float("inf"), default=None)  # fmt: skip
            sentence = level_sentence(b, worst, djx_of(worst, slices)) if worst is not None else ""
            b.text(" ".join(filter(None, [DEFINITIONS["level"], LEVEL_CLOSE, f"The date with the lowest LC/copula — {sentence}" if sentence else "", footer(prod, True, entries_path, prod_dir)])))  # fmt: skip
        if key == "calibration":
            b.text(
                "Table 2e-2. The index gate and the index errors on the upside and over the slices:"
            )
            gate_table(b, prod, by_date=False)
            b.text(f"{DEFINITIONS['gate']} {footer(prod, False, entries_path, prod_dir)}")
    tables["1y_production"] = frame_of(prod, tenor)

    # --- 3. the twenty dates at the development budget
    b.text(
        f"## 3. The yearly dates at the development budget ({tenor}): the first monthly entry of each year 2007–2026"
    )
    in_sweep = dates[dates["date"] != ""]
    outside = [d for d, inside in zip(in_sweep["date"], in_sweep["in_3m_sweep"], strict=True) if not inside]  # fmt: skip
    short = [f"{y} ({n})" for y, n in zip(in_sweep["year"], in_sweep["monthly_entries_in_year"], strict=True) if n < 12]  # fmt: skip
    also_reference = [d for d in dev_dates if d in pc.REFERENCE_DATES]
    partial_md = f"**Partial: {n_dev_priced} of {len(dev)} dates priced at this writing** ({sum(s.state == 'pending' for s in dev)} pending, {sum(s.state == 'failed' for s in dev)} failed). " if n_dev_priced < len(dev) else ""  # fmt: skip
    b.text(
        partial_md
        + f"Dates ({len(dev_dates)}): {', '.join(dev_dates)}. "
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
        *[(f"3d-{t}. The call at K_{t} = {CALL_MULT[t]} × P_D against the copula's", CALLS[t], "calls", True) for t in CALL_TAGS],
        ("3e. Clipped mass, index errors, names priced beyond their last kept expiry", [*CLIP, *INDEX], "calibration", False),
        ("3f. Sticky-strike deltas", DELTAS, "deltas", False),
    ]  # fmt: skip
    dev_flags = flag_line(dev, short=True)
    for title, quantities, key, study in plan3:
        b.text(f"### {title}")
        by_date(b, dev, quantities)
        high_strike = key == "calls" and title.startswith(("3d-125", "3d-150"))
        extra = (caveat + " " if high_strike else "") + (second + " " if second and key in ("listed", "kappa", "split") else "")  # fmt: skip
        b.text(f"{DEFINITIONS[key]} {extra}{PM_SE} {footer(dev, study, entries_path, dev_dir)}")
        if key != "calibration" and dev_flags:
            b.text(dev_flags)
        if key == "forward":
            b.text("Table 3a-2. The index level behind the ratios to the copula:")
            level_table(b, dev, by_date=True)
            b.text(" ".join(filter(None, [DEFINITIONS["level"], level_dev_sentence(b, dev) or LEVEL_CLOSE, footer(dev, True, entries_path, dev_dir)])))  # fmt: skip
        if key == "calibration":
            b.text(
                "Table 3e-2. The index gate and the index errors on the upside and over the slices:"
            )
            gate_table(b, dev, by_date=True)
            b.text(f"{DEFINITIONS['gate']} {footer(dev, False, entries_path, dev_dir)}")
    dev_frame = frame_of(dev, tenor)
    tables["1y_development"] = dev_frame
    is_priced = dev_frame["priced"] == "yes"
    n_priced = int(is_priced.sum())
    n_unflagged = int((is_priced & (dev_frame["flag_clip"].eq(False))).sum())
    n_flagged = n_priced - n_unflagged
    b.text("### 3g. Mean, median and quartiles over the dates")
    if n_priced < len(dev):
        b.text(f"**Partial: {n_priced} of {len(dev)} dates priced** — every figure below moves as the rows arrive.")  # fmt: skip
    csv = base / "tables" / "1y_development_summary.csv"
    tables["1y_development_summary"] = summaries(b, dev, dev_frame, dev_dir, csv)
    priced_frame = dev_frame[is_priced]
    if n_priced and not n_unflagged:
        left = f"Left block (the summary the owner defined: flagged rows out): none — all {n_priced} priced dates are flagged for the clipped mass ({clip_range(priced_frame)})."  # fmt: skip
    else:
        left = f"Left block (the summary the owner defined: flagged rows out): the {n_dates(n_unflagged)} priced and not flagged for the clipped mass."  # fmt: skip
    for kind, column in (("min", "clip_low_inner_max"), ("max", "clip_low_inner_max"), ("min", "clip_high_inner_max"), ("max", "clip_high_inner_max")):  # fmt: skip
        value = float(getattr(priced_frame[column], kind)()) if n_priced else None
        b.records.append(
            pc.record(
                f"{SECTION}.dev.summary.{column}.{kind}.all", SECTION, f"{column}: {kind} over every priced date", value, None, tenor=tenor, unit="fraction",
                definition=f"the {kind}imum over the priced yearly dates of the row's {column} (the share of the particles inside ±2.5 sd whose λ is clipped, the largest over the calibration's slices)",
                budget=pc.BUDGETS["development"], commit=", ".join(sorted({s.commit for s in dev if s.row and s.commit})), source=f"{base / 'tables' / '1y_development.csv'} (from {dev_dir})", n=n_priced,
                notes="no standard error: a share of the calibration's particles" if n_priced else "pending: no priced row yet",
            )
        )  # fmt: skip
    unflagged = [s for s in dev if s.priced and clip_flagged(s) is False]
    carried = "; ".join(carries(s, djx_of(s, slices), dev_frame) for s in unflagged)
    if len(unflagged) == 1:
        left += f" It is one observation, not a summary — {carried}."
    elif unflagged:
        left += f" What each of them carries — {carried}."
    b.text(
        f"{left} Right block: all priced dates ({n_priced} of {len(dev)}; {n_flagged} of them flagged), flagged or not; it is not the summary the owner defined. "
        "A date is out of the left block for one flag only: the clip flag of decision 5 (the mass clipped at λ = 0 or at the cap exceeds 1 % inside ±2.5 sd). No other flag excludes a date. "
        f"For information, the priced dates that carry the other flags — {other_flags(dev)}. "
        f"n is the number of dates behind each figure ({len(dev)} dates asked); mean ± the standard error of the mean across dates (sd/√n: the dispersion over the dates, not a Monte Carlo error); quartiles linearly interpolated. "
        "A count (names beyond last kept expiry) is given as its total over the dates; the homogeneity term (exactly 1 on every date) is not summarised; the index errors at +1.5 and +2.5 sd and the largest errors over the slices are not summarised (on some dates the cell is a strike where the model vol is not invertible: table 3e-2); the index level of table 3a-2 is not summarised. "
        f"The rows for K_125 and K_150: {caveat} "
        f"{footer(dev, True, entries_path, dev_dir)} The CSV behind: `{csv}`."
    )

    # --- 4. flags
    b.text("## 4. Flags per date")
    b.text("### 4a. The flags")
    everyone = [*prod, *dev]
    tables["1y_flags"] = flags_section(b, everyone, slices)
    for q in FLAGS:
        for s in everyone:
            b.add(q, s)
    b.text(f"{DEFINITIONS['flags']} {gate_summary(everyone)} Production rows — {footer(prod, False, entries_path, prod_dir)} Development rows — {footer(dev, False, entries_path, dev_dir)}")  # fmt: skip
    checks_section(b, everyone)
    b.text(
        "### 4b. The DJX target per date: the kept slices around the horizon and the slices dropped"
    )
    djx_section(b, everyone, slices, slices_path)
    if slices is not None:
        slice_records(b, slices, slices_path)
        skipped = sorted(set(slices[slices["status"] == "skipped"]["date"]))
        b.text(
            "Kept DJX slices below the horizon: the T values of the row's index_errors list (the model is repriced on the kept slices before the horizon, and at the horizon). "
            f"The other columns are from `{slices_path}`: a specification-only build under the current defaults (`{CONFIG}`) by `scripts/pm_1y_slices.py` run from the worktree at commit {slices['git_commit'].iloc[0]}, built {slices['built'].iloc[0]} New York — no calibration, no Monte Carlo; the slices do not depend on the budget. "
            "Nearest kept slice: the slice of the index surface nearest the horizon on each side; when no kept slice lies between a date's two, the target at the horizon is an interpolation between them. "
            "SVI rms: the root-mean-square error of the dropped slice's own SVI fit, in vol points. Dropped by the screen: by the third-Friday rule or the quote rules, before any fit. "
            "'The build's specification key is the row's': the key of the specification built here at the row's budget equals the row's spec_key, so the list is the row's own. "
            + (
                f"Dates the build skipped: {', '.join(skipped)}."
                if skipped
                else "The build skipped no date."
            )
        )
    else:
        b.text(f"Kept DJX slices below the horizon: the T values of the row's index_errors list. The list of the dropped DJX slices and the kept slice above the horizon are not available: no table of the specification build at `{slices_path}` (run `scripts/pm_1y_slices.py`); the rows carry counts only.")  # fmt: skip

    # --- 5. consistency
    b.text("## 5. The rows against the study's entries")
    consistency(b, everyone)
    b.text(f"The copula columns a row repeats (P_D_copula, EV_copula, EQV, T, the cash strikes) against basket B1 of `{entries_path}`, and the row's own flag against the flag recomputed here; no simulation. Sources: the rows of sections 2 and 3.")  # fmt: skip
    b.text(builder_commit())

    # --- read this first (placed under the header)
    d5_cells = []
    for label, slot, missing in variants:
        name = f"old defaults, {slot.budget} budget" if slot is not None and slot.group == "before" else label.replace(" — ", ", ")  # fmt: skip
        if slot is None:
            d5_cells.append(f"old defaults: {missing}")
        else:
            d5_cells.append(f"{name}: {show(CLIP[0], evaluate(CLIP[0], slot))} / {show(CLIP[1], evaluate(CLIP[1], slot))}")  # fmt: skip
    d5_line = "; ".join(d5_cells)
    far = []
    for s in [*dev, *(x for x in prod if x.date not in dev_dates)]:
        lo = djx_of(s, slices).bracket()[0]
        horizon = float(s.entry["T"]) if s.entry is not None else float("nan")
        if lo is not None and horizon - lo > 0.5:
            far.append(f"{s.date} ({lo:.3f}y)")
    far_line = f" Dates whose nearest kept DJX slice below the horizon is more than 0.5y before it (nearest kept slice below): {', '.join(far)}." if far else ""  # fmt: skip
    wide = []
    for s in [*dev, *(x for x in prod if x.date not in dev_dates)]:
        lo, hi, _ = djx_of(s, slices).bracket()
        if lo is not None and hi is not None and hi - lo > 0.75:
            wide.append(f"{s.date} ({lo:.3f}y / {hi:.3f}y)")
    if wide:
        far_line += f" Dates whose two nearest kept DJX slices around the horizon are more than 0.75y apart (below / above): {', '.join(wide)}."
    first = [
        "## Read this first",
        "",
        f"- **Status.** {STATUS_NOTE} {gate_summary(everyone)} Per date: the flags table (section 4) and tables 2e and 3e.",
        f"- **What the ± is.** {se_note}",
        "- **Index errors.** Tables 2e and 3e give, besides the money and the downside at the horizon, the errors at +1.5 and +2.5 sd and the largest error over the kept slices with where it sits. A cell marked *model vol not invertible* is a far out-of-the-money strike with a negligible price, not a mispriced smile; a cell marked *target vol … degenerate* (below a quarter of the at-the-money target vol) is one where the DJX target itself is not usable; a cell marked *thin target* is one where the target vol is between a quarter and a half of the at-the-money one. The largest error over the usable cells (no mark) is printed beside the largest over all cells.",
        f"- **DJX target repaired.** Table 4b lists, per date, the DJX slices the repair drops and the kept slices on each side of the horizon. {check_date}: {headline or 'the list is not available (no table of the specification build); section 1b gives what the rows carry.'}{far_line}",
        f"- **Decision 5 check (section 1).** {check_date} at {tenor}, clipped mass inside ±2.5 sd at λ = 0 / at the cap (flagged above 0.01) — {d5_line}. The before/after is decisions 1, 2 and 5 together, not decision 5 alone. {' '.join(target_lines)}",
        "- **Summaries (3g).** "
        + (f"Partial: {n_priced} of {len(dev)} dates priced. " if n_priced < len(dev) else f"All {len(dev)} dates priced. ")
        + (f"The summary the owner defined (flagged rows out) has no date: all {n_priced} priced dates are flagged for the clipped mass ({clip_range(priced_frame)}). The block *all priced dates* is made of flagged rows only." if n_priced and not n_unflagged else f"The summary the owner defined (flagged rows out) has {n_dates(n_unflagged)}; {n_flagged} of the {n_priced} priced dates are flagged for the clipped mass.{' It is one observation, not a summary — ' + carried + '.' if len(unflagged) == 1 else ''}" if n_priced else "No date is priced yet."),
        f"- **Calls at K_125 and K_150.** {caveat}",
        f"- **Sticky-strike deltas.** {STICKY}",
        "- **Units.** E[D], P_D, the strikes, the calls, ED_wing and ED_eqv are fractions of the notional; the two parts of the split are squared returns; the clipped masses are fractions of the particles; the index errors are vol points and the target vols percent; the deltas are percent of the price per +1 % of every spot; the times of the DJX slices and the horizon are years.",
        "",
    ]  # fmt: skip
    b.md[first_at:first_at] = first
    summary = f"1y ({tenor}): production {count(prod)}; development {count(dev)}"
    return b, tables, summary


def default_base(now: dt.datetime) -> Path:
    """Where a run at ``now`` writes: ``pm_update/1y`` before the owner's cutoff,
    ``pm_update/later/1y`` from it on."""
    return (pc.PM / "1y" if now < CUTOFF else pc.PM / "later" / "1y").resolve()


def output_base(arg: str | None, now: dt.datetime, test: bool = False) -> tuple[Path, bool]:
    """``(base, in_package)`` of a run at ``now``: ``arg`` when given, else
    :func:`default_base`.  Raises ``SystemExit`` for a test run (``test``) into the package and
    for a run into ``pm_update/1y`` from the cutoff on.  ``scripts/pm_1y_slices.py`` writes
    under the same rule."""
    base = Path(arg).resolve() if arg else default_base(now)
    in_package = pc.PM.resolve() in [base, *base.parents]
    if in_package and test:
        raise SystemExit("--no-strict and tenors other than 12m are tests: give a --base outside pm_update")  # fmt: skip
    if in_package and now >= CUTOFF and base == (pc.PM / "1y").resolve():
        raise SystemExit(
            f"{base} is frozen since {CUTOFF:%H:%M}: write to {pc.PM / 'later' / '1y'}"
        )
    return base, in_package


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--tenor", default="12m")
    ap.add_argument("--rows-dir", "--production-rows-dir", dest="rows_dir", default=None, help="the production rows (default: rows/<tenor>_production)")  # fmt: skip
    ap.add_argument("--development-rows-dir", default=None, help="the development rows (default: rows/<tenor>_development)")  # fmt: skip
    ap.add_argument("--before-rows-dir", default=None, help="rows of the old defaults, for section 1 (default: rows/<tenor>_production_norepair, else rows/<tenor>_development_norepair)")  # fmt: skip
    ap.add_argument("--slices-table", default=None, help="the table of scripts/pm_1y_slices.py (default: <base>/tables/1y_djx_slices.csv, else the one of pm_update/1y)")  # fmt: skip
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
    args.before_rows_dirs = [args.before_rows_dir] if args.before_rows_dir else [str(rows / f"{args.tenor}_{b}_norepair") for b in ("production", "development")]  # fmt: skip
    args.production_table = args.production_table or str(pc.LC_OUT / f"lcm_{args.tenor}.parquet")
    args.development_table = args.development_table or str(pc.LC_OUT / f"lcm_{args.tenor}_dev.parquet")  # fmt: skip
    now = dt.datetime.now(NY)
    base, in_package = output_base(args.base, now, test=args.no_strict or args.tenor != "12m")
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
