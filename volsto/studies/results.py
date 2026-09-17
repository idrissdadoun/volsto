"""Study results (SPEC §10.1, M10 Part 1): the long-format table every study writes to
``results.parquet``, and **the single place that enforces "no Monte Carlo number without its
standard error"** for the study runner.

One row per number (:data:`RESULT_COLUMNS`):

======== ===== ==================================================================================
column   type  meaning
======== ===== ==================================================================================
table    str   logical table name (``"headline"``)
row      str   row label (``"1F nu=1.0 rho=-0.7 kappa=1.5"``)
column   str   quantity key (``"fwd_atm_vol_1y2y"``)
value    float the number (NaN = not available, rendered ``--``)
stderr   float its Monte Carlo standard error; NaN exactly when ``exact`` is True (or the value
               itself is NaN)
exact    bool  closed-form / input / count values
unit     str   ``"vol pts"``, ``"% notional"``, :data:`DIMENSIONLESS` for a Monte Carlo ratio or
               probability, ``""`` only for exact values
source   str   ``store:<point_id>``, ``cache:<key>``, ``artefact:<path>``, ``computed``
               (several joined by ``;``)
note     str   free text
axes     str   JSON object of the row's coordinates (``{"nu": 1.0}``) or ``"{}"``
======== ===== ==================================================================================

The invariants (:func:`validate_frame`, run by :meth:`ResultsBuilder.build` and by every
:class:`Results` construction, including :meth:`Results.read`):

* a non-exact row with a finite value has a finite, non-negative stderr;
* an exact row has stderr NaN (``0`` or NaN are accepted from the caller and stored as NaN);
* a non-exact row with a NaN value must have a NaN stderr — a *missing* number, not a number
  without its error (reported, rendered ``--``);
* no infinite value; no duplicated ``(table, row, column)``; non-empty ``table / row / column``;
* a non-exact value carries a unit (``DIMENSIONLESS`` for a pure number);
* ``source`` is one of the four forms above; ``axes`` is a JSON object of finite scalars.

:func:`diff_results` is the rerun comparison: joined on ``(table, row, column)``; a Monte Carlo
row *moved* when ``|new − old| > nse · max(stderr_old, stderr_new)`` (strict; the rerun uses the
same seeds, so the two estimates are strongly correlated and their errors are **not** combined in
quadrature); an exact row — and a Monte Carlo row whose stderr is zero on both sides — moved when
it differs by more than :data:`EXACT_REL_TOL` relative; a NaN on one side only is a move; a row
whose unit changed or that switched between exact and Monte Carlo is ``changed`` (whatever its
value); rows on one side only are ``added`` / ``removed``.  ``nse`` must be finite and positive.

:class:`Column`, :class:`TableSpec` and :class:`FigureSpec` (the runner-module protocol's
declarations) live here because they depend on nothing but :class:`Results`; they are
re-exported by :mod:`volsto.studies.runner`, :mod:`volsto.studies.latex` and
:mod:`volsto.studies.style`.

Checked by ``tests/test_study_runner.py`` (``test_results_invariants``,
``test_diff_results_statuses``, the run / rerun round trip).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from volsto.calibration.cache import atomic_write

if TYPE_CHECKING:
    from matplotlib.figure import Figure

#: Columns of ``results.parquet``, in order.
RESULT_COLUMNS: tuple[str, ...] = (
    "table",
    "row",
    "column",
    "value",
    "stderr",
    "exact",
    "unit",
    "source",
    "note",
    "axes",
)
KEY_COLUMNS: tuple[str, str, str] = ("table", "row", "column")
STRING_COLUMNS: tuple[str, ...] = ("table", "row", "column", "unit", "source", "note", "axes")
#: The unit of a dimensionless Monte Carlo number (a probability, a ratio): a non-exact value must
#: state a unit, and this one is not printed in a table header.
DIMENSIONLESS = "1"
#: ``source`` prefixes (followed by an id) and the bare form, the four provenance kinds.
SOURCE_PREFIXES: tuple[str, ...] = ("store:", "cache:", "artefact:")
SOURCE_COMPUTED = "computed"
#: Relative tolerance under which an exact value is unchanged by a rerun (floating-point noise
#: of a deterministic computation, not a modelling change).
EXACT_REL_TOL = 1e-12
#: The rerun threshold of the owner's M10 Part 1: a number moved when it changed by more than
#: this many standard errors.
DEFAULT_NSE = 2.0
#: Statuses of :func:`diff_results`.
DIFF_STATUSES: tuple[str, ...] = ("unchanged", "moved", "changed", "added", "removed")


class ResultsError(ValueError):
    """A results frame violates the invariants (module docstring); the message lists every
    offending row."""


# --------------------------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------------------------


def _valid_source(source: str) -> bool:
    parts = source.split(";")
    return bool(parts) and all(
        p == SOURCE_COMPUTED or (p.startswith(SOURCE_PREFIXES) and p.partition(":")[2] != "")
        for p in parts
    )


def _axes_json(axes: Mapping[str, Any] | str | None) -> str:
    """Canonical JSON of a row's coordinates (sorted keys, finite scalars only)."""
    if axes is None:
        return "{}"
    data = json.loads(axes) if isinstance(axes, str) else dict(axes)
    if not isinstance(data, dict):
        raise ResultsError(f"axes must be a JSON object, got {axes!r}")
    clean: dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, np.generic):
            v = v.item()
        if v is not None and not isinstance(v, (str, int, float, bool)):
            raise ResultsError(f"axes[{k!r}] must be a scalar, got {v!r}")
        clean[str(k)] = v
    try:
        return json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except ValueError as exc:
        raise ResultsError(f"axes {axes!r}: {exc}") from exc


def _row_problems(r: Mapping[str, Any]) -> list[str]:
    """Invariant violations of one normalised row (module docstring)."""
    out: list[str] = []
    for c in KEY_COLUMNS:
        if not isinstance(r[c], str) or not r[c]:
            out.append(f"empty {c}")
    value, stderr, exact = float(r["value"]), float(r["stderr"]), bool(r["exact"])
    if math.isinf(value):
        out.append(f"infinite value {value}")
    if exact:
        if not math.isnan(stderr):
            out.append(f"exact value with stderr {stderr} (an exact value has none)")
    else:
        if math.isfinite(value) and not math.isfinite(stderr):
            out.append(f"Monte Carlo value {value} without a finite stderr ({stderr})")
        if math.isnan(value) and not math.isnan(stderr):
            out.append(f"NaN value with stderr {stderr} (a missing number has no stderr)")
        if math.isfinite(stderr) and stderr < 0:
            out.append(f"negative stderr {stderr}")
        if not r["unit"]:
            out.append(f"Monte Carlo value without a unit (use {DIMENSIONLESS!r} for a ratio)")
    if not _valid_source(str(r["source"])):
        out.append(
            f"source {r['source']!r} is not one of store:<id>, cache:<key>, artefact:<path>, "
            f"{SOURCE_COMPUTED} (';'-joined)"
        )
    return out


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """``frame.to_dict("records")`` typed with string keys (the frames here have string
    column names)."""
    return cast(list[dict[str, Any]], frame.to_dict("records"))


def _normalise(frame: pd.DataFrame) -> pd.DataFrame:
    """Column order and dtypes of :data:`RESULT_COLUMNS`; exact rows' stderr forced to NaN."""
    missing = [c for c in RESULT_COLUMNS if c not in frame.columns]
    extra = [c for c in frame.columns if c not in RESULT_COLUMNS]
    if missing or extra:
        raise ResultsError(f"results columns: missing {missing}, unexpected {extra}")
    df = frame.loc[:, list(RESULT_COLUMNS)].reset_index(drop=True).copy()
    for c in STRING_COLUMNS:
        df[c] = df[c].map(lambda v: "" if v is None else str(v)).astype(object)
    df["value"] = pd.to_numeric(df["value"], errors="raise").astype(np.float64)
    df["stderr"] = pd.to_numeric(df["stderr"], errors="raise").astype(np.float64)
    df["exact"] = df["exact"].astype(bool)
    return df


def validate_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """The normalised frame, or :class:`ResultsError` listing every violated invariant."""
    df = _normalise(frame)
    problems: list[str] = []
    for i, r in enumerate(records(df)):
        for p in _row_problems(r):
            problems.append(f"  [{i}] {r['table']} / {r['row']} / {r['column']}: {p}")
        try:
            _axes_json(str(r["axes"]))
        except (ResultsError, json.JSONDecodeError) as exc:
            problems.append(f"  [{i}] {r['table']} / {r['row']} / {r['column']}: axes {exc}")
    dup = df.duplicated(list(KEY_COLUMNS), keep=False)
    if dup.any():
        keys = sorted({tuple(k) for k in df.loc[dup, list(KEY_COLUMNS)].to_numpy().tolist()})
        problems.extend(f"  duplicated (table, row, column) {k}" for k in keys)
    if problems:
        raise ResultsError(
            f"{len(problems)} results invariant violation(s):\n" + "\n".join(problems)
        )
    return df


# --------------------------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Results:
    """A validated results frame (module docstring).  Construct through
    :class:`ResultsBuilder` or :meth:`read`; the frame is validated on construction."""

    frame: pd.DataFrame

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame", validate_frame(self.frame))

    def __len__(self) -> int:
        return len(self.frame)

    # -- lookup ------------------------------------------------------------------------------

    def tables(self) -> list[str]:
        """Table names in order of first appearance."""
        return list(dict.fromkeys(self.frame["table"]))

    def rows(self, table: str) -> list[str]:
        """Row labels of ``table`` in order of first appearance."""
        return list(dict.fromkeys(self._table(table)["row"]))

    def columns(self, table: str) -> list[str]:
        """Column keys of ``table`` in order of first appearance."""
        return list(dict.fromkeys(self._table(table)["column"]))

    def _table(self, table: str) -> pd.DataFrame:
        df = self.frame[self.frame["table"] == table]
        if df.empty:
            raise KeyError(f"no results table {table!r}; tables: {self.tables()}")
        return df

    def record(self, table: str, row: str, column: str) -> dict[str, Any]:
        """The full row of one number (``KeyError`` when absent)."""
        f = self.frame
        hit = f[(f["table"] == table) & (f["row"] == row) & (f["column"] == column)]
        if hit.empty:
            raise KeyError(f"no result ({table!r}, {row!r}, {column!r})")
        rec = hit.iloc[0].to_dict()
        return {str(k): v for k, v in rec.items()}

    def value(self, table: str, row: str, column: str) -> tuple[float, float]:
        """``(value, stderr)`` of one number (stderr NaN for an exact value)."""
        rec = self.record(table, row, column)
        return float(rec["value"]), float(rec["stderr"])

    def is_exact(self, table: str, row: str, column: str) -> bool:
        return bool(self.record(table, row, column)["exact"])

    def unit(self, table: str, column: str) -> str:
        """The unit of ``column`` in ``table`` (``ResultsError`` when its rows disagree)."""
        df = self._table(table)
        units = set(df.loc[df["column"] == column, "unit"])
        if not units:
            raise KeyError(f"no column {column!r} in results table {table!r}")
        if len(units) > 1:
            raise ResultsError(f"column {column!r} of table {table!r} mixes units {units}")
        return str(units.pop())

    def axes(self, table: str, row: str) -> dict[str, Any]:
        """The coordinates of a row (the first non-empty ``axes`` among its numbers)."""
        df = self._table(table)
        for a in df.loc[df["row"] == row, "axes"]:
            data = json.loads(str(a))
            if data:
                return dict(data)
        return {}

    def long(self, table: str, column: str | None = None) -> pd.DataFrame:
        """The long rows of ``table`` (optionally one ``column``) with the decoded ``axes``
        expanded as ``axis_<name>`` columns — the input of a figure."""
        df = self._table(table)
        if column is not None:
            df = df[df["column"] == column]
        df = df.reset_index(drop=True).copy()
        decoded = [json.loads(str(a)) for a in df["axes"]]
        for name in dict.fromkeys(k for d in decoded for k in d):
            df[f"axis_{name}"] = [d.get(name, np.nan) for d in decoded]
        return df

    def pivot(self, table: str, value: bool = True) -> pd.DataFrame:
        """Rows × columns of ``table`` (row order and column order of first appearance, index
        named ``row``).  With ``value=True`` each value column is followed by its
        ``<column>_stderr`` twin (NaN for exact values); with ``value=False`` only the stderr
        columns are returned.  A cell a row does not carry is NaN."""
        df = self._table(table)
        rows, cols = self.rows(table), self.columns(table)
        vals = df.pivot(index="row", columns="column", values="value").reindex(
            index=rows, columns=cols
        )
        errs = df.pivot(index="row", columns="column", values="stderr").reindex(
            index=rows, columns=cols
        )
        out = pd.DataFrame(index=pd.Index(rows, name="row"))
        for c in cols:
            if value:
                out[c] = vals[c].to_numpy(dtype=np.float64)
            out[f"{c}_stderr"] = errs[c].to_numpy(dtype=np.float64)
        return out

    # -- io ----------------------------------------------------------------------------------

    def write(self, path: str | Path) -> Path:
        """Write ``results.parquet`` atomically (the project's one write rule,
        :func:`volsto.calibration.cache.atomic_write`)."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        frame = self.frame
        return atomic_write(p, lambda tmp: frame.to_parquet(tmp, index=False))

    @classmethod
    def read(cls, path: str | Path) -> Results:
        """Read and re-validate a ``results.parquet``."""
        return cls(pd.read_parquet(Path(path)))

    @classmethod
    def concat(cls, parts: Iterable[Results]) -> Results:
        frames = [p.frame for p in parts]
        if not frames:
            return ResultsBuilder().build()
        return cls(pd.concat(frames, ignore_index=True))


class ResultsBuilder:
    """Accumulates numbers; :meth:`build` validates them all at once and raises
    :class:`ResultsError` listing every violation (the contract: ``add`` records, ``build``
    checks)."""

    def __init__(self) -> None:
        self._rows: list[dict[str, Any]] = []

    def __len__(self) -> int:
        return len(self._rows)

    def add(
        self,
        table: str,
        row: str,
        column: str,
        value: float,
        stderr: float | None = None,
        *,
        exact: bool = False,
        unit: str,
        source: str,
        note: str = "",
        axes: Mapping[str, Any] | None = None,
    ) -> None:
        """Record one number.  ``stderr`` is required (finite) for a Monte Carlo value; an exact
        value takes ``stderr=None`` (``0`` / NaN are accepted and stored as NaN)."""
        se = float("nan") if stderr is None else float(stderr)
        if exact and (se == 0.0 or math.isnan(se)):
            se = float("nan")
        self._rows.append(
            {
                "table": table,
                "row": row,
                "column": column,
                "value": float(value),
                "stderr": se,
                "exact": bool(exact),
                "unit": unit,
                "source": source,
                "note": note,
                "axes": _axes_json(axes),
            }
        )

    def add_mc(
        self,
        table: str,
        row: str,
        column: str,
        estimate: Any,
        *,
        unit: str,
        source: str,
        scale: float = 1.0,
        note: str = "",
        axes: Mapping[str, Any] | None = None,
    ) -> None:
        """Record a :class:`~volsto.engine.mc.PriceResult`-like estimate (``.mean`` and
        ``.stderr``), both multiplied by ``scale`` (e.g. 100 for % of notional)."""
        self.add(
            table,
            row,
            column,
            scale * float(estimate.mean),
            scale * float(estimate.stderr),
            unit=unit,
            source=source,
            note=note,
            axes=axes,
        )

    def add_exact(
        self,
        table: str,
        row: str,
        column: str,
        value: float,
        *,
        unit: str,
        source: str,
        note: str = "",
        axes: Mapping[str, Any] | None = None,
    ) -> None:
        """Record a closed-form / input / count value."""
        self.add(
            table,
            row,
            column,
            value,
            None,
            exact=True,
            unit=unit,
            source=source,
            note=note,
            axes=axes,
        )

    def extend(self, results: Results) -> None:
        """Append every row of an existing :class:`Results`."""
        self._rows.extend(records(results.frame))

    def build(self) -> Results:
        frame = pd.DataFrame(self._rows, columns=list(RESULT_COLUMNS))
        return Results(frame)


# --------------------------------------------------------------------------------------------
# diff
# --------------------------------------------------------------------------------------------

DIFF_COLUMNS: tuple[str, ...] = (
    "table",
    "row",
    "column",
    "status",
    "old_value",
    "old_stderr",
    "new_value",
    "new_stderr",
    "delta",
    "tolerance",
    "n_stderr",
    "exact",
    "unit",
    "old_unit",
    "why",
)


def _diff_row(
    o: Mapping[str, Any] | None, n: Mapping[str, Any] | None, nse: float
) -> dict[str, Any]:
    nan = float("nan")
    base = o if o is not None else n
    assert base is not None
    out: dict[str, Any] = {c: base[c] for c in KEY_COLUMNS}
    ov = float(o["value"]) if o is not None else nan
    os_ = float(o["stderr"]) if o is not None else nan
    nv = float(n["value"]) if n is not None else nan
    ns = float(n["stderr"]) if n is not None else nan
    exact = bool(o is not None and o["exact"]) and bool(n is not None and n["exact"])
    out.update(
        old_value=ov,
        old_stderr=os_,
        new_value=nv,
        new_stderr=ns,
        exact=exact,
        unit=str((n if n is not None else base)["unit"]),
        old_unit=str(o["unit"]) if o is not None else "",
        why="",
    )
    if o is None or n is None:
        out.update(status="added" if o is None else "removed", delta=nan, tolerance=nan)
        out["n_stderr"] = nan
        return out
    delta = nv - ov
    finite_errs = [e for e in (os_, ns) if math.isfinite(e)]
    scale = max(finite_errs, default=nan)
    n_se = abs(delta) / scale if math.isfinite(scale) and scale > 0 else nan
    reasons = []
    if str(o["unit"]) != str(n["unit"]):
        reasons.append(f"unit {o['unit']!r} -> {n['unit']!r}")
    if bool(o["exact"]) != bool(n["exact"]):
        kinds = ("Monte Carlo", "exact")
        reasons.append(f"{kinds[bool(o['exact'])]} -> {kinds[bool(n['exact'])]}")
    if reasons:
        out.update(
            status="changed", delta=delta, tolerance=nan, n_stderr=n_se, why="; ".join(reasons)
        )
        return out
    if math.isnan(ov) or math.isnan(nv):
        moved = math.isnan(ov) != math.isnan(nv)
        tol = nan
    elif exact or not (math.isfinite(scale) and scale > 0):
        # an exact number, or a Monte Carlo number with a zero stderr on both sides: the exact
        # relative tolerance (2 x 0 would flag floating-point noise)
        tol = EXACT_REL_TOL * max(abs(ov), abs(nv))
        moved = abs(delta) > tol
    else:
        tol = nse * scale
        moved = abs(delta) > tol
    out.update(status="moved" if moved else "unchanged", delta=delta, tolerance=tol, n_stderr=n_se)
    return out


def diff_results(old: Results, new: Results, nse: float = DEFAULT_NSE) -> pd.DataFrame:
    """Every ``(table, row, column)`` of either side with its :data:`DIFF_STATUSES` status
    (module docstring), in the old order then the added rows; columns :data:`DIFF_COLUMNS`
    (``n_stderr`` = ``|delta| / max(stderr_old, stderr_new)``, NaN for exact rows; ``why`` says
    what a ``changed`` row changed).  ``nse`` must be finite and positive."""
    if not (math.isfinite(nse) and nse > 0):
        raise ValueError(f"nse must be finite and positive, got {nse}")
    key = list(KEY_COLUMNS)
    olds = {tuple(r[c] for c in key): r for r in records(old.frame)}
    news = {tuple(r[c] for c in key): r for r in records(new.frame)}
    rows = [_diff_row(o, news.get(k), nse) for k, o in olds.items()]
    rows += [_diff_row(None, n, nse) for k, n in news.items() if k not in olds]
    return pd.DataFrame(rows, columns=list(DIFF_COLUMNS))


def changed(diff: pd.DataFrame) -> pd.DataFrame:
    """The rows of a :func:`diff_results` frame that are not ``unchanged``."""
    return diff[diff["status"] != "unchanged"].reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# table / figure declarations of the runner-module protocol
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    """One column of a :class:`TableSpec`: the results ``key``, its ``header`` text, the
    ``unit`` shown in the header (``""`` = the results' unit of the column), and the significant
    ``digits`` of an exact value (:func:`volsto.studies.latex.format_exact`; Monte Carlo values
    are rounded by their stderr)."""

    key: str
    header: str
    unit: str = ""
    digits: int = 4

    def __post_init__(self) -> None:
        if not self.key or not self.header:
            raise ValueError("a column needs a key and a header")
        if self.digits < 1:
            raise ValueError("digits must be >= 1")


@dataclass(frozen=True)
class TableSpec:
    """A rendered table: ``tables/<name>.tex`` (booktabs) and a markdown table in
    ``study.md``, reading the results table ``table``; ``rows`` fixes the row order (``None`` =
    results order)."""

    name: str
    caption: str
    table: str
    columns: tuple[Column, ...]
    rows: tuple[str, ...] | None = None
    row_header: str = ""

    def __post_init__(self) -> None:
        _check_name(self.name, "table")
        if not self.columns:
            raise ValueError(f"table {self.name!r} has no columns")


@dataclass(frozen=True)
class FigureSpec:
    """A rendered figure: ``figures/<name>.pdf`` and ``.png``, drawn by ``draw(results)`` (which
    uses :func:`volsto.studies.style.new_figure`) from the results alone."""

    name: str
    caption: str
    draw: Callable[[Results], Figure] = field(compare=False)

    def __post_init__(self) -> None:
        _check_name(self.name, "figure")


def _check_name(name: str, kind: str) -> None:
    """A table / figure name is a file stem and a LaTeX label: ``[A-Za-z0-9_-]+``."""
    if not name or not all(ch.isascii() and (ch.isalnum() or ch in "_-") for ch in name):
        raise ValueError(f"{kind} name {name!r} must match [A-Za-z0-9_-]+")
