"""S7 — hedging: the M8b model-mismatch reserve and the delta-regime ranking (SPEC §10.2, owner's
M10 Part 2 "S7. Hedging: the M8b model-mismatch reserve tables and the delta-regime ranking").

**Artefacts only.**  The study reads the M8b summary tables and the raw-slice discriminator's
verdict under the configured ``outputs`` root and computes nothing by Monte Carlo; it never
opens the leverage cache or the results store:

* ``m8b/m8b_table_A.csv`` — study A, the regression ports with the ``q`` sweep: per pricing
  model (``LV`` / ``2F``, world = pricing) and product (``cliquet 1y``, ``fva 1y-2y``) the
  strategies ranked by P&L std;
* ``m8b/m8b_table_B.csv`` — study B, the **model-mismatch reserve**: per world (``same``,
  ``historical``, ``pure LV``, ``nu x1.5``) and product the desk leakage, and the leakage
  relative to the ``same`` world of the product (``leakage_vs_same``: the world = pricing runs
  carry the engine's own baseline, so the reserve of a world is the difference to the ``same``
  row, not the raw leakage — §8.2), the static spread and the dynamic leakage; the gated world
  (ii) rows, skipped with the discriminator's reason;
* the raw-slice discriminator's verdict that gates world (ii) (``real`` enables it) and its
  per-pillar SSR table (raw slices against a fitted surface history, Newey–West standard
  errors): **the repaired eSSVI history's run**, ``essvi_gate/discriminator/
  discriminator_verdict.json`` (M10 Part 0, SPEC §13.1; :data:`VERDICT_REPAIRED_COMMAND`); only
  when that file is absent, the M8b gate's own ``m8b/discriminator_verdict.json`` — the
  pre-repair plain-SSVI history ``outputs/m7/hdn_history_ssvi.csv`` — as a documented fallback.
  The verdict and its reason are the same in both files today (the gate reads the raw 1m pillar
  only), the fitted column differs; study.md names the history the table used;
* ``m8b/m8b_table_D.csv`` — study D, the delta regimes ranked by their **distance to the common
  minimum-variance delta** (the precision-weighted mean of the four regime rows' implied MV
  deltas; the ``min_variance`` row is a fifth line flagged ``mv_valid``, not the reference —
  the owner's decision of 2026-09-16), with the owner's headline
  :data:`volsto.studies.m8b.STUDY_D_HEADLINE` quoted verbatim and checked against the ranking.

The tables are read through :func:`volsto.viewers.api.get_hedging_table`, which renames every
``*_se`` column ``*_stderr`` and re-attaches the stem-named twins of the tables written before
2026-09-16 (``desk_mean`` beside ``mean_se``), so both conventions give the same numbers.  A Monte
Carlo value whose own stderr is absent or not finite is recorded as a *missing* number (NaN, with
a note), never without its error.  The dispersion ``mv_common_spread`` is written upstream without
an error (``volsto.viewers.api.UNPAIRED_UPSTREAM``); it is a difference of two Monte Carlo values
whose correlation (the rows share their world paths) is not measured, so it carries the **sum** of
the two extreme rows' stderrs, a bound whatever that correlation (:func:`spread_stderr`).  A max −
min of noisy values is biased upward: with ``k`` rows of similar stderr ``s`` noise alone gives an
expected range of about ``d_k · s`` (:data:`RANGE_FACTORS`, ``d_4 = 2.06``); the spread is shown
over that noise floor as ``spread_over_noise`` (a Monte Carlo ratio whose stderr is the spread's
over the floor, the floor's own sampling error ignored and said so).

**Derived numbers.**  A z-score carries stderr 1 (its sampling sd, :data:`Z_SD_NOTE`), so a
rerun compares it at 2 stderr like its parents.  Ranks, counts and flags stay exact
(:data:`EXACT_KINDS` declares each); every rank has a companion flag ``<rank>_decided``
(:func:`decided_flags`) saying whether the row's ordering against the next-ranked row is decided
at 2 stderr — the two ``value ± 2 se`` intervals do not overlap, a test valid whatever the rows'
correlation.

Results tables: ``a_ranking``, ``b_reserve``, ``b_matrix`` (world x product of the reserve),
``b_skipped``, ``gate``, ``gate_pillars``, ``d_ranking``, ``d_common``, ``d_summary``.  Figures:
``a_std`` (P&L std per strategy), ``b_reserve`` (reserve per world, one panel per product),
``d_distance`` (distance to the common MV delta per regime, one panel per product).

CI-fast mode (``configs/studies/catalogue/s7_fast.yaml``) reads the synthetic M8b outputs of
``tests/_synthetic_store.py::make_synthetic_outputs`` (written by the sanctioned fixture
``tests/conftest.py::toy_build``).  Checked by ``tests/test_catalogue_s5_s7.py``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volsto.config import ConfigError
from volsto.hedging.strategies import MIN_VARIANCE_REGIME
from volsto.studies import latex, m8b, style
from volsto.studies.latex import LONGTABLE_MIN_ROWS
from volsto.studies.results import (
    DIMENSIONLESS,
    Column,
    FigureSpec,
    Results,
    ResultsBuilder,
    TableSpec,
)
from volsto.studies.runner import Requirement, StudyContext, artefact_path
from volsto.viewers import api
from volsto.viewers.config import ViewerConfig

TITLE = "S7 - Hedging: the model-mismatch reserve and the delta-regime ranking"
QUESTION = (
    "How much does each hedging strategy leak when the world is not the pricing model, and "
    "which delta regime sits closest to the minimum-variance delta?"
)
REQUIRED_PARAMS = ("tables", "headline_regimes")
OPTIONAL_PARAMS: tuple[str, ...] = ()

#: The M8b tables this study can show.
KNOWN_TABLES: tuple[str, ...] = ("A", "B", "D")
TABLE_FILE = "m8b/m8b_table_{name}.csv"
#: The discriminator verdict of the repaired eSSVI history (M10 Part 0) and its producing command.
VERDICT_REPAIRED = "essvi_gate/discriminator/discriminator_verdict.json"
VERDICT_REPAIRED_COMMAND = (
    ".venv/bin/python scripts/m8b_discriminator.py --ssvi-history "
    'outputs/essvi_gate/hdn_history_repaired.csv --fitted-label "eSSVI repaired" '
    "--out outputs/essvi_gate/discriminator --no-world-spec"
)
#: The documented fallback: the M8b gate's own verdict (the pre-repair plain-SSVI history, the
#: script's default), read only when the repaired run's file is absent.
VERDICT_FALLBACK = "m8b/discriminator_verdict.json"
VERDICT_FALLBACK_COMMAND = ".venv/bin/python scripts/m8b_discriminator.py"
#: What the fallback file compares against when it does not name its fitted history (the
#: script's ``DEFAULT_FITTED_HISTORY`` / ``DEFAULT_FITTED_LABEL``).
FALLBACK_FITTED_HISTORY = "outputs/m7/hdn_history_ssvi.csv"
FALLBACK_FITTED_LABEL = "SSVI"
#: The commands that write the artefacts (``--resume`` rebuilds a table from the stored results).
TABLE_COMMAND = ".venv/bin/python scripts/m8b.py --study {name} --resume"
#: Header unit of a column whose rows carry different product units (the caption lists them).
PRODUCT_UNIT = "product unit"
#: ``status_code`` of a table row: 0 ok, 1 anything else (the reason in the note).
STATUS_OK = 0.0
STATUS_NOT_OK = 1.0
#: The world whose leakage is the engine's own baseline (``table_B``'s ``leakage_vs_same``).
SAME_WORLD = "same"
MISSING_SE_NOTE = "not reported: the value or its stderr is not finite in the source table"
SPREAD_NOTE = (
    "largest minus smallest implied MV delta of the regime rows (upstream, no stderr); stderr: "
    "the sum of the two extreme rows' stderrs, a bound whatever their correlation (not measured); "
    "a max - min of noisy values is biased upward (see spread_over_noise)"
)
#: Expected range of ``k`` independent unit normals (the ``d_2`` constants of quality control):
#: the range noise alone gives ``k`` values of stderr ``s`` is about ``d_k · s``.
RANGE_FACTORS: dict[int, float] = {
    2: 1.128,
    3: 1.693,
    4: 2.059,
    5: 2.326,
    6: 2.534,
    7: 2.704,
    8: 2.847,
    9: 2.970,
    10: 3.078,
}
NOISE_FLOOR_NOTE = (
    "the spread over its noise floor d_k x the mean stderr of the k rows (the range noise alone "
    "would give independent rows of similar stderr): a ratio near 1 is not evidence of "
    "dispersion; stderr: the spread's over the floor"
)
Z_NOTE = "z-score: value over its own stderr"
#: A z-score is a Monte Carlo quantity whose sampling sd is 1 (the catalogue's convention).
Z_SD = 1.0
Z_SD_NOTE = "stderr 1: the sampling sd of a z-score"
#: A rank's ordering against the next-ranked row is decided at this many stderrs.
DECIDED_NSE = 2.0
DECIDED_NOTE = (
    "1 = the ordering against the next-ranked row is decided at 2 stderr (the two value +/- 2 se "
    "intervals do not overlap: a test valid for any correlation); 0 = not decided; NaN = no "
    "next-ranked row"
)
RANK_NOTE = (
    "a rank: exact given the table's Monte Carlo values (it changes only when they do, possibly "
    "within their stderrs)"
)

#: What each exact row is (:data:`volsto.studies.catalogue._common.EXACT_KIND_NAMES`, full-match
#: regexes on table and column).  Every rank names its companion "decided at 2 se" flag
#: column (:func:`decided_flags`).
EXACT_KINDS: tuple[tuple[str, ...], ...] = (
    ("sources", "rows", "count"),
    ("a_ranking|b_skipped|d_ranking", "status_code", "flag"),
    ("a_ranking", "rank", "rank", "rank_decided"),
    ("d_ranking", "std_rank", "rank", "std_rank_decided"),
    ("d_ranking", "distance_rank", "rank", "distance_rank_decided"),
    ("d_summary", "model_rank", "rank", "model_rank_decided"),
    ("a_ranking", "rank_decided", "flag"),
    ("d_ranking", "std_rank_decided|distance_rank_decided|mv_valid", "flag"),
    ("d_summary", "model_rank_decided|headline_holds|headline_decided|mv_row_valid", "flag"),
    ("d_common", "spread_matches_rows", "flag"),
    ("d_common", "mv_common_rows", "count"),
    ("gate", "repaired_history|world_ii_enabled|reason_matches_table_b", "flag"),
    ("gate", "n_dates_used|skipped_rows", "count"),
)


def validate_params(params: Mapping[str, Any]) -> None:
    tables = params["tables"]
    if not isinstance(tables, list) or not tables or not set(tables) <= set(KNOWN_TABLES):
        raise ConfigError(f"tables must be a non-empty subset of {list(KNOWN_TABLES)}: {tables}")
    regimes = params["headline_regimes"]
    known = [r for r in m8b.REGIMES_D if r != MIN_VARIANCE_REGIME]
    if not isinstance(regimes, list) or not regimes or not set(regimes) <= set(known):
        raise ConfigError(f"headline_regimes must be a non-empty subset of {known}: {regimes}")


# --------------------------------------------------------------------------------------------
# requirements
# --------------------------------------------------------------------------------------------


def requirements(ctx: StudyContext) -> list[Requirement]:
    out: list[Requirement] = []
    for name in ctx.params["tables"]:
        out.append(
            ctx.artefact_requirement(
                TABLE_FILE.format(name=name),
                f"M8b study {name} summary table",
                TABLE_COMMAND.format(name=name),
            )
        )
    if "B" in ctx.params["tables"]:
        rel = verdict_file(ctx)
        command = VERDICT_REPAIRED_COMMAND if rel == VERDICT_REPAIRED else VERDICT_FALLBACK_COMMAND
        out.append(
            ctx.artefact_requirement(
                rel, "raw-slice discriminator verdict (gates world ii)", command
            )
        )
    return out


def verdict_file(ctx: StudyContext) -> str:
    """The verdict to read (module docstring): the repaired history's run when present, else the
    M8b fallback when present, else the repaired one (whose command a missing requirement
    prints)."""
    if artefact_path(ctx.outputs_root, VERDICT_REPAIRED).is_file():
        return VERDICT_REPAIRED
    if artefact_path(ctx.outputs_root, VERDICT_FALLBACK).is_file():
        return VERDICT_FALLBACK
    return VERDICT_REPAIRED


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def _viewer_cfg(ctx: StudyContext) -> ViewerConfig:
    return ViewerConfig(
        cache_root=ctx.cache_root, store_root=ctx.store_root, outputs_root=ctx.outputs_root
    )


def _num(v: Any) -> float:
    """``v`` as a float (NaN for None, text, booleans stay 0/1)."""
    if v is None:
        return math.nan
    if isinstance(v, bool | np.bool_):
        return float(bool(v))
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _text(v: Any) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    return str(v)


def _unit(u: Any) -> str:
    """A table's unit text for a results row (a Monte Carlo number must carry one)."""
    s = _text(u).strip()
    return s if s else DIMENSIONLESS


def _flag(v: Any) -> float:
    """A boolean-ish cell (``True`` / ``False`` / ``"True"`` / NaN) as 1 / 0 / NaN."""
    if isinstance(v, bool | np.bool_):
        return float(bool(v))
    s = _text(v).strip().lower()
    if s in ("true", "1", "1.0", "yes"):
        return 1.0
    if s in ("false", "0", "0.0", "no"):
        return 0.0
    return math.nan


def add_mc(
    b: ResultsBuilder,
    table: str,
    row: str,
    column: str,
    value: Any,
    stderr: Any,
    *,
    unit: str,
    source: str,
    axes: Mapping[str, Any],
    note: str = "",
) -> None:
    """A Monte Carlo number with its stderr, or a missing number (NaN, noted) when either is
    not finite — never a value without its error."""
    v, se = _num(value), _num(stderr)
    if math.isfinite(v) and math.isfinite(se) and se >= 0.0:
        b.add(table, row, column, v, se, unit=unit, source=source, axes=axes, note=note)
    else:
        b.add(
            table,
            row,
            column,
            math.nan,
            math.nan,
            unit=unit,
            source=source,
            note=MISSING_SE_NOTE,
            axes=axes,
        )


def add_z(
    b: ResultsBuilder,
    table: str,
    row: str,
    column: str,
    z: float,
    *,
    source: str,
    axes: Mapping[str, Any],
    note: str,
) -> None:
    """A z-score with its sampling sd 1 (:data:`Z_SD_NOTE`), or a missing number."""
    add_mc(
        b,
        table,
        row,
        column,
        z,
        Z_SD if math.isfinite(z) else math.nan,
        unit=DIMENSIONLESS,
        source=source,
        axes=axes,
        note=f"{note}; {Z_SD_NOTE}",
    )


def decided_flags(entries: Sequence[tuple[str, float, float, float]]) -> dict[str, float]:
    """For ``(key, rank, value, stderr)`` entries ranked in ascending ``value``: 1 when the
    entry's ordering against the next-ranked entry (rank + 1) is decided at
    :data:`DECIDED_NSE` stderr — the two intervals ``value ± 2 se`` do not overlap, i.e. the
    difference exceeds ``2 (se + se_next)``, a test valid whatever the two errors' correlation —,
    0 when it is not, NaN for the last-ranked entry, an unranked one (rank < 1) or a
    non-finite value."""
    by_rank = {
        int(rank): (key, value, se)
        for key, rank, value, se in entries
        if math.isfinite(rank) and rank >= 1 and float(rank).is_integer()
    }
    out: dict[str, float] = {key: math.nan for key, _, _, _ in entries}
    for rank, (key, value, se) in by_rank.items():
        nxt = by_rank.get(rank + 1)
        if nxt is None or not all(math.isfinite(x) for x in (value, se, nxt[1], nxt[2])):
            continue
        out[key] = 1.0 if value + DECIDED_NSE * se < nxt[1] - DECIDED_NSE * nxt[2] else 0.0
    return out


def set_decided(
    inside: Sequence[tuple[float, float]], outside: Sequence[tuple[float, float]]
) -> float:
    """1 when every ``(value, stderr)`` of ``inside`` is ahead of every one of ``outside`` at
    :data:`DECIDED_NSE` stderr (the largest ``value + 2 se`` inside below the smallest
    ``value - 2 se`` outside: the set is decided whatever the order within it), 0 when not,
    NaN when either side is empty or holds a non-finite number."""
    pairs = [*inside, *outside]
    if not inside or not outside or not all(math.isfinite(x) for p in pairs for x in p):
        return math.nan
    top = max(v + DECIDED_NSE * se for v, se in inside)
    bottom = min(v - DECIDED_NSE * se for v, se in outside)
    return 1.0 if top < bottom else 0.0


def _pair(rec: Mapping[str, Any], col: str) -> tuple[float, float]:
    return _num(rec.get(col)), _num(rec.get(f"{col}_stderr"))


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): v for k, v in r.items()} for r in df.to_dict("records")]


def _units_caption(results: Results, table: str) -> str:
    """``"autocall 3y: % of notional; ..."`` from the rows of ``table`` (axis ``product``)."""
    long = results.long(table)
    if "axis_product" not in long.columns:
        return ""
    seen: dict[str, str] = {}
    for p, u in zip(long["axis_product"], long["unit"]):
        if isinstance(p, str) and u and u != DIMENSIONLESS:
            seen.setdefault(p, str(u))
    return "; ".join(f"{p}: {u}" for p, u in seen.items())


# --------------------------------------------------------------------------------------------
# compute
# --------------------------------------------------------------------------------------------


def _table_a(b: ResultsBuilder, df: pd.DataFrame, source: str) -> None:
    records = _records(df)
    # upstream ranks ``std`` within (pricing, product) over every row with a std
    groups: dict[tuple[str, str], list[tuple[str, float, float, float]]] = {}
    for r in records:
        pricing, product = _text(r.get("pricing")), _text(r["product"])
        groups.setdefault((pricing, product), []).append(
            (
                f"{pricing} | {product} | {_text(r['strategy'])}",
                _num(r.get("rank")),
                *_pair(r, "std"),
            )
        )
    decided: dict[str, float] = {}
    for entries in groups.values():
        decided.update(decided_flags(entries))
    for r in records:
        pricing, product, strategy = (
            _text(r.get("pricing")),
            _text(r["product"]),
            _text(r["strategy"]),
        )
        row = f"{pricing} | {product} | {strategy}"
        axes = {"pricing": pricing, "product": product, "strategy": strategy}
        ok = _text(r.get("status")) == "ok"
        b.add_exact(
            "a_ranking",
            row,
            "status_code",
            STATUS_OK if ok else STATUS_NOT_OK,
            unit="",
            source=source,
            note=_text(r.get("status")) + (f": {_text(r.get('reason'))}" if not ok else ""),
            axes=axes,
        )
        unit = _unit(r.get("unit"))
        std, std_se = _pair(r, "std")
        add_mc(b, "a_ranking", row, "std", std, std_se, unit=unit, source=source, axes=axes)
        mean, mean_se = _pair(r, "desk_mean")
        add_mc(b, "a_ranking", row, "desk_mean", mean, mean_se, unit=unit, source=source, axes=axes)
        b.add_exact(
            "a_ranking",
            row,
            "rank",
            _num(r.get("rank")),
            unit="",
            source=source,
            axes=axes,
            note=RANK_NOTE,
        )
        b.add_exact(
            "a_ranking",
            row,
            "rank_decided",
            decided.get(row, math.nan),
            unit="",
            source=source,
            axes=axes,
            note=DECIDED_NOTE,
        )


B_COLUMNS: tuple[str, ...] = (
    "leakage_desk",
    "leakage_vs_same",
    "static_spread",
    "dynamic_leakage",
    "std",
    "q05_desk",
    "q95_desk",
)


def _table_b(b: ResultsBuilder, df: pd.DataFrame, source: str) -> list[str]:
    """``b_reserve`` / ``b_matrix`` / ``b_skipped``; returns the skipped rows' reasons."""
    reasons: list[str] = []
    for r in _records(df):
        world, product = _text(r["world"]), _text(r["product"])
        row = f"{world} | {product}"
        axes = {"world": world, "product": product}
        if _text(r.get("status")) != "ok":
            reason = _text(r.get("reason"))
            reasons.append(reason)
            b.add_exact(
                "b_skipped",
                row,
                "status_code",
                STATUS_NOT_OK,
                unit="",
                source=source,
                note=f"{_text(r.get('status'))}: {reason}",
                axes=axes,
            )
            continue
        unit = _unit(r.get("unit"))
        for col in B_COLUMNS:
            v, se = _pair(r, col)
            add_mc(b, "b_reserve", row, col, v, se, unit=unit, source=source, axes=axes)
        v, se = _pair(r, "leakage_vs_same")
        add_mc(
            b,
            "b_matrix",
            world,
            product,
            v,
            se,
            unit=unit,
            source=source,
            axes={"world": world},
        )
    return reasons


def _gate(b: ResultsBuilder, doc: Mapping[str, Any], reasons: Sequence[str], rel: str) -> None:
    source = f"artefact:{rel}"
    verdict = str(doc.get("verdict", "")).strip()
    reason = str(doc.get("reason", ""))
    repaired = rel == VERDICT_REPAIRED
    history = str(doc.get("fitted_history") or "")
    label = str(doc.get("fitted_label") or "")
    named = bool(history and label)
    if not named:
        history = history or FALLBACK_FITTED_HISTORY
        label = label or FALLBACK_FITTED_LABEL
    b.add_exact(
        "gate",
        "discriminator",
        "repaired_history",
        1.0 if repaired else 0.0,
        unit="",
        source=source,
        note=(
            f"fitted history: {label} ({history})"
            + ("" if named else " - not named by the file: the script's default history")
            + f"; verdict file {rel}"
            + ("" if repaired else " (fallback: the repaired run's file is absent)")
        ),
    )
    # the gate rule of m8b.discriminator_gate: world (ii) runs only on the verdict "real"
    b.add_exact(
        "gate",
        "discriminator",
        "world_ii_enabled",
        1.0 if verdict.lower() == "real" else 0.0,
        unit="",
        source=source,
        note=f"verdict '{verdict}': {reason}",
    )
    b.add_exact(
        "gate",
        "discriminator",
        "n_dates_used",
        _num(doc.get("n_dates_used")),
        unit="",
        source=source,
    )
    consistent = bool(reasons) and all(reason and reason in r for r in reasons)
    b.add_exact(
        "gate",
        "discriminator",
        "skipped_rows",
        float(len(reasons)),
        unit="",
        source=source,
        note="table B rows carried as skipped",
    )
    b.add_exact(
        "gate",
        "discriminator",
        "reason_matches_table_b",
        1.0 if consistent else 0.0,
        unit="",
        source=source,
        note="1 when every skipped table-B row quotes the verdict's reason",
    )
    for rec in doc.get("table", []) or []:
        T, window = _num(rec.get("T")), _num(rec.get("window"))
        row = f"T={T:.4g}y window={window:g}d"
        axes = {"T": T, "window": window}
        for col, key, se_col, note in (
            ("ssr_raw", "ssr_raw", "se_raw", "raw vendor slices"),
            ("ssr_fitted", "ssr_ssvi", "se_ssvi", f"the fitted history: {label}"),
            (
                "diff",
                "diff",
                "se_diff",
                "raw minus fitted; stderr: upstream's independent-SE approximation",
            ),
        ):
            add_mc(
                b,
                "gate_pillars",
                row,
                col,
                rec.get(key),
                rec.get(se_col),
                unit=DIMENSIONLESS,
                source=source,
                axes=axes,
                note=note,
            )
        add_z(
            b,
            "gate_pillars",
            row,
            "z",
            _num(rec.get("z")),
            source=source,
            axes=axes,
            note="upstream's z: diff over its stderr",
        )


D_COLUMNS: tuple[tuple[str, bool], ...] = (
    ("std", True),
    ("mean_delta", False),
    ("lambda_star", False),
    ("std_at_lambda", True),
    ("mv_delta_implied", False),
    ("distance_to_mv", False),
)


def spread_stderr(rows: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    """``(spread, stderr)`` of the implied minimum-variance deltas of the regime rows that enter
    the common MV delta (finite value, positive stderr; the ``min_variance`` row excluded):
    largest minus smallest, and the sum of the two extreme rows' stderrs — a bound whatever their
    correlation (module docstring).
    NaN when fewer than two rows qualify."""
    vals = []
    for r in rows:
        if _text(r.get("regime")) == MIN_VARIANCE_REGIME:
            continue
        v, se = _pair(r, "mv_delta_implied")
        if math.isfinite(v) and math.isfinite(se) and se > 0.0:
            vals.append((v, se))
    if len(vals) < 2:
        return math.nan, math.nan
    hi, lo = max(vals), min(vals)
    return hi[0] - lo[0], hi[1] + lo[1]


def spread_noise_floor(rows: Sequence[Mapping[str, Any]]) -> float:
    """``d_k`` (:data:`RANGE_FACTORS`) times the mean stderr of the ``k`` regime rows of
    :func:`spread_stderr`: the expected range of ``k`` independent estimates of one value with
    those stderrs (NaN outside the tabulated ``k``)."""
    ses = []
    for r in rows:
        if _text(r.get("regime")) == MIN_VARIANCE_REGIME:
            continue
        v, se = _pair(r, "mv_delta_implied")
        if math.isfinite(v) and math.isfinite(se) and se > 0.0:
            ses.append(se)
    factor = RANGE_FACTORS.get(len(ses))
    return math.nan if factor is None else factor * float(np.mean(ses))


def _table_d(b: ResultsBuilder, df: pd.DataFrame, headline: Sequence[str], source: str) -> None:
    by_product: dict[str, list[dict[str, Any]]] = {}
    for r in _records(df):
        by_product.setdefault(_text(r["product"]), []).append(r)
    for product, rows in by_product.items():
        unit = next((_unit(r.get("unit")) for r in rows if _text(r.get("unit"))), DIMENSIONLESS)
        # upstream ranks the ok rows by std and the ok non-MV rows by |distance_to_mv|; the
        # others carry rank -1 (or 0 for the min_variance row), which decided_flags leaves NaN
        std_decided = decided_flags(
            [(_text(r["regime"]), _num(r.get("std_rank")), *_pair(r, "std")) for r in rows]
        )
        dist_decided = decided_flags(
            [
                (
                    _text(r["regime"]),
                    _num(r.get("distance_rank")),
                    abs(_pair(r, "distance_to_mv")[0]),
                    _pair(r, "distance_to_mv")[1],
                )
                for r in rows
            ]
        )
        for r in rows:
            regime = _text(r["regime"])
            row = f"{product} | {regime}"
            axes = {"product": product, "regime": regime}
            ok = _text(r.get("status")) == "ok"
            b.add_exact(
                "d_ranking",
                row,
                "status_code",
                STATUS_OK if ok else STATUS_NOT_OK,
                unit="",
                source=source,
                note=_text(r.get("status")),
                axes=axes,
            )
            for col, in_product_unit in D_COLUMNS:
                v, se = _pair(r, col)
                u = unit if in_product_unit else DIMENSIONLESS
                add_mc(b, "d_ranking", row, col, v, se, unit=u, source=source, axes=axes)
            for col, flags in (("std_rank", std_decided), ("distance_rank", dist_decided)):
                b.add_exact(
                    "d_ranking",
                    row,
                    col,
                    _num(r.get(col)),
                    unit="",
                    source=source,
                    axes=axes,
                    note=RANK_NOTE,
                )
                b.add_exact(
                    "d_ranking",
                    row,
                    f"{col}_decided",
                    flags.get(regime, math.nan),
                    unit="",
                    source=source,
                    axes=axes,
                    note=DECIDED_NOTE,
                )
            is_mv = regime == MIN_VARIANCE_REGIME
            b.add_exact(
                "d_ranking",
                row,
                "mv_valid",
                _flag(r.get("mv_valid")) if is_mv else math.nan,
                unit="",
                source=source,
                note=_text(r.get("mv_note")) if is_mv else "",
                axes=axes,
            )
            add_z(
                b,
                "d_ranking",
                row,
                "mv_z",
                _num(r.get("mv_z")) if is_mv else math.nan,
                source=source,
                axes=axes,
                note="the min_variance row's mean delta minus the common MV delta over the two "
                "stderrs in quadrature (upstream)",
            )
        first = rows[0]
        paxes = {"product": product}
        v, se = _pair(first, "mv_common")
        add_mc(
            b,
            "d_common",
            product,
            "mv_common",
            v,
            se,
            unit=DIMENSIONLESS,
            source=source,
            axes=paxes,
        )
        spread, spread_se = spread_stderr(rows)
        upstream = _num(first.get("mv_common_spread"))
        agrees = math.isfinite(upstream) and abs(spread - upstream) <= 1e-9 * max(1.0, abs(spread))
        add_mc(
            b,
            "d_common",
            product,
            "mv_common_spread",
            upstream if agrees else math.nan,
            spread_se if agrees else math.nan,
            unit=DIMENSIONLESS,
            source=source,
            axes=paxes,
            note=SPREAD_NOTE,
        )
        floor = spread_noise_floor(rows)
        ratio_ok = agrees and math.isfinite(floor) and floor > 0.0
        add_mc(
            b,
            "d_common",
            product,
            "spread_over_noise",
            upstream / floor if ratio_ok else math.nan,
            spread_se / floor if ratio_ok else math.nan,
            unit=DIMENSIONLESS,
            source=source,
            axes=paxes,
            note=f"{NOISE_FLOOR_NOTE} (floor {floor:.4g}; its own sampling error is ignored)",
        )
        b.add_exact(
            "d_common",
            product,
            "spread_matches_rows",
            1.0 if agrees else 0.0,
            unit="",
            source=source,
            note=(
                SPREAD_NOTE
                if agrees
                else f"the table's spread {upstream!r} differs from its rows' {spread!r}"
            ),
            axes=paxes,
        )
        b.add_exact(
            "d_common",
            product,
            "mv_common_rows",
            _num(first.get("mv_common_rows")),
            unit="",
            source=source,
            axes=paxes,
        )
        ranks = {
            _text(r["regime"]): _num(r.get("distance_rank"))
            for r in rows
            if _text(r["regime"]) != MIN_VARIANCE_REGIME
        }
        closest = {k for k, v in ranks.items() if 1 <= v <= len(headline)}
        b.add_exact(
            "d_summary",
            product,
            "headline_holds",
            1.0 if closest == set(headline) else 0.0,
            unit="",
            source=source,
            note="closest: " + ", ".join(sorted(closest, key=lambda k: ranks[k])),
            axes=paxes,
        )
        k = len(headline)
        distances = {
            _text(r["regime"]): (abs(_pair(r, "distance_to_mv")[0]), _pair(r, "distance_to_mv")[1])
            for r in rows
        }
        inside = [distances[g] for g, v in ranks.items() if 1 <= v <= k]
        outside = [distances[g] for g, v in ranks.items() if v > k]
        b.add_exact(
            "d_summary",
            product,
            "headline_decided",
            set_decided(inside, outside) if len(inside) == k else math.nan,
            unit="",
            source=source,
            note=(
                f"1 = every one of the {k} closest regimes is ahead of every other ranked regime "
                "at 2 stderr (|distance| +/- 2 se intervals do not overlap): the closest set is "
                "decided, whatever the order within it"
            ),
            axes=paxes,
        )
        model = next((r for r in rows if _text(r["regime"]) == "model"), None)
        mv_row = next((r for r in rows if _text(r["regime"]) == MIN_VARIANCE_REGIME), None)
        if model is not None:
            v, se = _pair(model, "distance_to_mv")
            add_mc(
                b,
                "d_summary",
                product,
                "model_minus_mv",
                v,
                se,
                unit=DIMENSIONLESS,
                source=source,
                axes=paxes,
            )
            b.add_exact(
                "d_summary",
                product,
                "model_rank",
                _num(model.get("distance_rank")),
                unit="",
                source=source,
                axes=paxes,
                note=RANK_NOTE,
            )
            b.add_exact(
                "d_summary",
                product,
                "model_rank_decided",
                dist_decided.get("model", math.nan),
                unit="",
                source=source,
                axes=paxes,
                note=DECIDED_NOTE,
            )
        b.add_exact(
            "d_summary",
            product,
            "mv_row_valid",
            _flag(mv_row.get("mv_valid")) if mv_row is not None else math.nan,
            unit="",
            source=source,
            note=_text(mv_row.get("mv_note")) if mv_row is not None else "no min_variance row",
            axes=paxes,
        )


def compute(ctx: StudyContext) -> Results:
    cfg = _viewer_cfg(ctx)
    b = ResultsBuilder()
    tables = list(ctx.params["tables"])
    for name in tables:
        rel = TABLE_FILE.format(name=name)
        path = ctx.artefact(rel)
        df = api.get_hedging_table(cfg, name)
        ctx.log.info("table %s: %d rows from %s", name, len(df), path)
        source = f"artefact:{rel}"
        b.add_exact(
            "sources",
            f"table {name}",
            "rows",
            float(len(df)),
            unit="",
            source=source,
            note=Path(rel).name,
        )
        if df.empty:
            continue
        if name == "A":
            _table_a(b, df, source)
        elif name == "B":
            reasons = _table_b(b, df, source)
            rel = verdict_file(ctx)
            doc = json.loads(ctx.artefact(rel).read_text(encoding="utf-8"))
            _gate(b, doc, reasons, rel)
        else:
            _table_d(b, df, list(ctx.params["headline_regimes"]), source)
    return b.build()


# --------------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------------


def _fitted_label(results: Results) -> str:
    """The fitted history named by the ``gate`` table's note (``fitted history: <label> …``)."""
    if "gate" not in results.tables():
        return "unnamed"
    note = str(results.record("gate", "discriminator", "repaired_history")["note"])
    head = note.partition(";")[0]
    return head.removeprefix("fitted history: ") or "unnamed"


def tables(results: Results) -> list[TableSpec]:
    have = set(results.tables())
    out: list[TableSpec] = [
        TableSpec(
            "sources",
            "M8b tables read (rows per file).",
            "sources",
            (Column("rows", "rows", digits=4),),
            row_header="table",
        )
    ]
    if "a_ranking" in have:
        units = _units_caption(results, "a_ranking")
        out.append(
            TableSpec(
                "a_ranking",
                "Study A (world = pricing): hedged P&L std per strategy, ranked within pricing "
                f"model and product, and whether each rank's ordering against the next is "
                f"decided at 2 stderr (1 yes); desk mean (desk short). Units: {units}.",
                "a_ranking",
                (
                    Column("std", "P&L std", unit=PRODUCT_UNIT),
                    Column("desk_mean", "desk mean", unit=PRODUCT_UNIT),
                    Column("rank", "rank", digits=2),
                    Column("rank_decided", "decided at 2 se", digits=1),
                    Column("status_code", "status (0 ok)", digits=1),
                ),
                row_header="pricing | product | strategy",
            )
        )
    if "b_reserve" in have:
        units = _units_caption(results, "b_reserve")
        out.append(
            TableSpec(
                "b_reserve",
                "Study B, the model-mismatch reserve: desk leakage per path, the leakage minus "
                "the product's same-world leakage (the reserve), the static spread (marked minus "
                "world price at t = 0), the dynamic leakage, the std and the desk 5/95% quantiles. "
                f"Units: {units}.",
                "b_reserve",
                (
                    Column("leakage_desk", "leakage (desk)", unit=PRODUCT_UNIT),
                    Column("leakage_vs_same", "reserve (vs same)", unit=PRODUCT_UNIT),
                    Column("static_spread", "static spread", unit=PRODUCT_UNIT),
                    Column("dynamic_leakage", "dynamic leakage", unit=PRODUCT_UNIT),
                    Column("std", "std", unit=PRODUCT_UNIT),
                    Column("q05_desk", "q05 (desk)", unit=PRODUCT_UNIT),
                    Column("q95_desk", "q95 (desk)", unit=PRODUCT_UNIT),
                ),
                row_header="world | product",
            )
        )
    if "b_matrix" in have:
        cols = tuple(Column(c, c) for c in results.columns("b_matrix"))
        out.append(
            TableSpec(
                "b_matrix",
                "The reserve matrix: leakage minus the same-world leakage, world x product.",
                "b_matrix",
                cols,
                row_header="world",
            )
        )
    if "b_skipped" in have:
        out.append(
            TableSpec(
                "b_skipped",
                "Study B rows not run (the gated world): status 1 = skipped; the reason is in "
                "the narrative.",
                "b_skipped",
                (Column("status_code", "skipped", digits=1),),
                row_header="world | product",
            )
        )
    if "gate" in have:
        out.append(
            TableSpec(
                "gate",
                "The discriminator gate of world (ii): enabled only on the verdict 'real'; "
                "repaired history = 1 when the verdict is the repaired eSSVI history's run "
                "(0: the M8b fallback file, the pre-repair plain-SSVI history).",
                "gate",
                (
                    Column("world_ii_enabled", "world (ii) enabled", digits=1),
                    Column("repaired_history", "repaired history", digits=1),
                    Column("n_dates_used", "dates", digits=4),
                    Column("skipped_rows", "B rows skipped", digits=3),
                    Column("reason_matches_table_b", "reason quoted in B", digits=1),
                ),
            )
        )
    if "gate_pillars" in have:
        out.append(
            TableSpec(
                "gate_pillars",
                "Discriminator: realised SSR from raw vendor slices vs from the fitted surface "
                f"history ({_fitted_label(results)}), their difference (upstream's "
                "independent-SE approximation) and its z (with its sampling sd 1).",
                "gate_pillars",
                (
                    Column("ssr_raw", "SSR raw"),
                    Column("ssr_fitted", "SSR fitted"),
                    Column("diff", "raw - fitted"),
                    Column("z", "z"),
                ),
                row_header="pillar",
            )
        )
    if "d_ranking" in have:
        units = _units_caption(results, "d_ranking")
        out.append(
            TableSpec(
                "d_ranking",
                "Study D (world = pricing, spot-only hedge): P&L std and its rank, the mean "
                "relative delta, the in-sample lambda* and implied MV delta, the distance to the "
                "common MV delta and its rank (1 = closest; 0 = the min_variance row itself); "
                "beside each rank, 1 when its ordering against the next rank is decided at 2 "
                "stderr (the two value +/- 2 se intervals do not overlap); "
                f"mv_valid on the min_variance row (1 valid). Std units: {units}.",
                "d_ranking",
                (
                    Column("std", "P&L std", unit=PRODUCT_UNIT),
                    Column("std_rank", "std rank", digits=2),
                    Column("std_rank_decided", "decided", digits=1),
                    Column("mean_delta", "mean delta"),
                    Column("lambda_star", "lambda*"),
                    Column("mv_delta_implied", "implied MV delta"),
                    Column("distance_to_mv", "distance to common MV"),
                    Column("distance_rank", "distance rank", digits=2),
                    Column("distance_rank_decided", "decided", digits=1),
                    Column("mv_valid", "MV row valid", digits=1),
                    Column("mv_z", "MV row z", digits=3),
                ),
                row_header="product | regime",
            )
        )
    if "d_common" in have:
        out.append(
            TableSpec(
                "d_common",
                "The common minimum-variance delta per product (precision-weighted over the four "
                "regime rows; se the perfect-correlation bound), its spread across the rows "
                "(stderr: the sum of the two extreme rows' stderrs, a bound whatever their "
                "correlation), that spread over its noise floor (d_k x the mean row stderr: a "
                "max - min of noisy values is biased upward, so a ratio near 1 is noise), and "
                "the rows used.",
                "d_common",
                (
                    Column("mv_common", "common MV delta"),
                    Column("mv_common_spread", "spread (bound se)"),
                    Column("spread_over_noise", "spread / noise floor"),
                    Column("mv_common_rows", "rows", digits=2),
                    Column("spread_matches_rows", "spread = rows", digits=1),
                ),
                row_header="product",
            )
        )
    if "d_summary" in have:
        out.append(
            TableSpec(
                "d_summary",
                "Headline check per product: 1 when the closest regimes are exactly the owner's "
                "headline regimes, and 1 when that closest set is decided at 2 stderr (every "
                "member ahead of every other ranked regime); the model delta minus the common MV "
                "delta, its rank and whether that rank is decided at 2 stderr; the min_variance "
                "row's validity.",
                "d_summary",
                (
                    Column("headline_holds", "headline holds", digits=1),
                    Column("headline_decided", "decided at 2 se", digits=1),
                    Column("model_minus_mv", "model - common MV"),
                    Column("model_rank", "model rank", digits=2),
                    Column("model_rank_decided", "rank decided at 2 se", digits=1),
                    Column("mv_row_valid", "MV row valid", digits=1),
                ),
                row_header="product",
            )
        )
    return fit_specs(out, results)


def fit_specs(specs: Sequence[TableSpec], results: Results) -> list[TableSpec]:
    """Every spec with more than :data:`volsto.studies.latex.LONGTABLE_MIN_ROWS` rows split into
    consecutive parts: a longer table would be an unscaled ``longtable`` and overflow the page
    when it is wide."""
    out: list[TableSpec] = []
    for spec in specs:
        rows = list(spec.rows) if spec.rows is not None else results.rows(spec.table)
        if len(rows) <= LONGTABLE_MIN_ROWS:
            out.append(spec)
            continue
        chunks = [rows[i : i + LONGTABLE_MIN_ROWS] for i in range(0, len(rows), LONGTABLE_MIN_ROWS)]
        for k, chunk in enumerate(chunks, 1):
            out.append(
                dataclasses.replace(
                    spec,
                    name=f"{spec.name}_{k}",
                    caption=f"{spec.caption} (part {k} of {len(chunks)})",
                    rows=tuple(chunk),
                )
            )
    return out


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------


def _panels(n: int) -> tuple[int, int]:
    ncols = min(n, 3)
    return (n + ncols - 1) // ncols, ncols


def _axes_list(axes: Any) -> list[Any]:
    return list(np.atleast_1d(axes).ravel())


def _draw_a(results: Results) -> Any:
    long = results.long("a_ranking")
    products = list(dict.fromkeys(long["axis_product"]))
    nrows, ncols = _panels(len(products))
    fig, axes = style.new_figure(nrows, ncols, width=4.4 * ncols, height=3.6 * nrows)
    for ax, product in zip(_axes_list(axes), products):
        sub = long[(long["axis_product"] == product) & (long["column"] == "std")]
        strategies = list(dict.fromkeys(sub["axis_strategy"]))
        pricings = list(dict.fromkeys(sub["axis_pricing"]))
        width = 0.8 / max(len(pricings), 1)
        for i, pricing in enumerate(pricings):
            s = sub[sub["axis_pricing"] == pricing].set_index("axis_strategy")
            x = [strategies.index(k) + (i - (len(pricings) - 1) / 2) * width for k in s.index]
            style.mc_errorbar(
                ax,
                x,
                s["value"].to_numpy(),
                s["stderr"].to_numpy(),
                series=i,
                label=f"pricing = world = {pricing}",
                line=False,
            )
        ax.set_xticks(range(len(strategies)), strategies, rotation=60, ha="right", fontsize=7)
        unit = next((u for u in sub["unit"] if u), "")
        ax.set_ylabel(f"P&L std [{unit}]")
        ax.set_title(product)
        ax.legend()
    for ax in _axes_list(axes)[len(products) :]:
        ax.set_visible(False)
    return fig


def _draw_b(results: Results) -> Any:
    long = results.long("b_reserve", "leakage_vs_same")
    products = list(dict.fromkeys(long["axis_product"]))
    nrows, ncols = _panels(len(products))
    fig, axes = style.new_figure(nrows, ncols, width=3.6 * ncols, height=3.2 * nrows)
    for ax, product in zip(_axes_list(axes), products):
        sub = long[long["axis_product"] == product]
        worlds = list(sub["axis_world"])
        ax.axhline(0.0, color=style.INK["spine"], linewidth=0.8)
        style.mc_errorbar(
            ax,
            list(range(len(worlds))),
            sub["value"].to_numpy(),
            sub["stderr"].to_numpy(),
            label="reserve (leakage - same world)",
            line=False,
        )
        ax.set_xticks(range(len(worlds)), worlds, rotation=30, ha="right")
        unit = next((u for u in sub["unit"] if u), "")
        ax.set_ylabel(f"desk [{unit}]")
        ax.set_title(product)
    for ax in _axes_list(axes)[len(products) :]:
        ax.set_visible(False)
    return fig


def _draw_d(results: Results) -> Any:
    long = results.long("d_ranking", "distance_to_mv")
    products = list(dict.fromkeys(long["axis_product"]))
    nrows, ncols = _panels(len(products))
    fig, axes = style.new_figure(nrows, ncols, width=4.0 * ncols, height=3.4 * nrows)
    for ax, product in zip(_axes_list(axes), products):
        sub = long[long["axis_product"] == product].reset_index(drop=True)
        regimes = list(sub["axis_regime"])
        ax.axhline(0.0, color=style.INK["spine"], linewidth=0.8)
        is_mv = sub["axis_regime"] == MIN_VARIANCE_REGIME
        for i, (mask, label) in enumerate(
            (
                (~is_mv, "regime rows (ranked)"),
                (is_mv, "min_variance row (benchmark, not ranked)"),
            )
        ):
            part = sub[mask]
            if part.empty:
                continue
            style.mc_errorbar(
                ax,
                [regimes.index(r) for r in part["axis_regime"]],
                part["value"].to_numpy(),
                part["stderr"].to_numpy(),
                series=i,
                label=label,
                line=False,
            )
        ax.set_xticks(range(len(regimes)), regimes, rotation=30, ha="right")
        ax.set_ylabel("mean delta - common MV delta")
        ax.set_title(product)
        ax.legend(loc="best")
    for ax in _axes_list(axes)[len(products) :]:
        ax.set_visible(False)
    return fig


def figures(results: Results) -> list[FigureSpec]:
    have = set(results.tables())
    out: list[FigureSpec] = []
    if "a_ranking" in have:
        out.append(
            FigureSpec(
                "a_std",
                "Study A: hedged P&L std per strategy under each pricing model (error bars: 1 "
                "stderr).",
                _draw_a,
            )
        )
    if "b_reserve" in have:
        out.append(
            FigureSpec(
                "b_reserve",
                "Study B: the model-mismatch reserve per world (leakage minus the product's "
                "same-world leakage, desk sign; error bars: 1 stderr).",
                _draw_b,
            )
        )
    if "d_ranking" in have:
        out.append(
            FigureSpec(
                "d_distance",
                "Study D: each regime's mean delta minus the common minimum-variance delta "
                "(error bars: 1 stderr, quadrature).",
                _draw_d,
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# narrative
# --------------------------------------------------------------------------------------------


def _pm(results: Results, table: str, row: str, column: str) -> str:
    v, se = results.value(table, row, column)
    return latex.format_value_text(v, se)


def _declared(results: Results) -> list[str]:
    """``table:<name>`` / ``figure:<name>`` of every declared table and figure, in order (a
    narrative may place only those)."""
    return [f"table:{t.name}" for t in tables(results)] + [
        f"figure:{f.name}" for f in figures(results)
    ]


def _place(declared: Sequence[str], *names: str) -> list[str]:
    """The placeholder lines of the declared ``names`` (``"table:x"``, and the parts
    ``"table:x_<k>"`` of a split table), each followed by a blank line."""
    out: list[str] = []
    for n in names:
        for d in declared:
            if d == n or re.fullmatch(re.escape(n) + r"_\d+", d):
                out += ["{{" + d + "}}", ""]
    return out


def _narrative_a(results: Results, declared: Sequence[str]) -> list[str]:
    long = results.long("a_ranking")
    lines = ["## Study A — regression ports", ""]
    ranks = long[long["column"] == "rank"]
    for (pricing, product), g in ranks.groupby(["axis_pricing", "axis_product"], sort=False):
        best = g.sort_values("value").iloc[0]
        row = str(best["row"])
        delta_row = f"{pricing} | {product} | delta only"
        line = (
            f"- {product}, pricing = world = {pricing}: the lowest P&L std is "
            f"**{best['axis_strategy']}** at {_pm(results, 'a_ranking', row, 'std')} "
            f"{results.record('a_ranking', row, 'std')['unit']}"
        )
        if delta_row in set(long["row"]) and delta_row != row:
            line += f" (delta only: {_pm(results, 'a_ranking', delta_row, 'std')})"
        second = g[g["value"] == 2.0]
        if not second.empty and "rank_decided" in set(long["column"]):
            decided = results.value("a_ranking", row, "rank_decided")[0]
            ahead = str(second.iloc[0]["axis_strategy"])
            if decided == 1.0:
                line += f"; ahead of {ahead} at 2 stderr"
            elif decided == 0.0:
                line += f"; **not decided** at 2 stderr against {ahead}"
        lines.append(line + ".")
    return [*lines, "", *_place(declared, "table:a_ranking", "figure:a_std")]


def _narrative_b(results: Results, declared: Sequence[str]) -> list[str]:
    lines = [
        "## Study B — the model-mismatch reserve",
        "",
        "The hedger runs with world = pricing carry a non-zero hedged mean (the pricing error of "
        "V0 plus the regression drift of the hedge legs), so the reserve a world calls for is "
        "its leakage **minus the product's same-world leakage** (`reserve (vs same)`), not the "
        "raw leakage.",
        "",
    ]
    if "b_reserve" in results.tables():
        long = results.long("b_reserve", "leakage_vs_same")
        for world, g in long.groupby("axis_world", sort=False):
            if world == SAME_WORLD:
                continue
            parts = []
            for rec in g.to_dict("records"):
                v, se = float(rec["value"]), float(rec["stderr"])
                if not math.isfinite(v):
                    parts.append(f"{rec['axis_product']} --")
                    continue
                z = v / se if se > 0 else math.nan
                parts.append(
                    f"{rec['axis_product']} {latex.format_value_text(v, se)} {rec['unit']} "
                    f"(z {z:+.1f})"
                )
            lines.append(f"- world **{world}**: " + "; ".join(parts) + ".")
        finite = long[np.isfinite(long["value"]) & (long["axis_world"] != SAME_WORLD)]
        if not finite.empty:
            top = finite.iloc[int(np.argmax(np.abs(finite["value"].to_numpy())))]
            lines += [
                "",
                f"The largest reserve in magnitude is {top['axis_product']} under world "
                f"**{top['axis_world']}**: "
                f"{latex.format_value_text(float(top['value']), float(top['stderr']))} "
                f"{top['unit']} per path (desk sign: negative = the desk loses).",
            ]
        lines.append("")
    lines += _place(declared, "table:b_reserve", "table:b_matrix", "figure:b_reserve")
    if "b_skipped" in results.tables():
        notes = results.long("b_skipped")
        worlds = sorted(set(notes["axis_world"]))
        reason = str(notes["note"].iloc[0])
        lines += [
            f"**Skipped:** world(s) {', '.join(worlds)} — {len(notes)} row(s) not run. Reason, as "
            f"table B records it: {reason}",
            "",
            *_place(declared, "table:b_skipped"),
        ]
    if "gate" in results.tables():
        gate = results.record("gate", "discriminator", "world_ii_enabled")
        matches = results.value("gate", "discriminator", "reason_matches_table_b")[0] == 1.0
        hist = results.record("gate", "discriminator", "repaired_history")
        which = str(hist["note"])
        lines += [
            (
                "**The discriminator table below uses the repaired eSSVI history** (M10 Part 0): "
                if hist["value"] == 1.0
                else "**The discriminator table below uses the pre-repair history** — the "
                "repaired run's verdict is absent, so the M8b gate's own file is read as the "
                "documented fallback: "
            )
            + which
            + ".",
            "",
            f"The gate's verdict: {gate['note']}. World (ii) is "
            + (
                "enabled."
                if gate["value"] == 1.0
                else "disabled (it runs only on the verdict 'real')."
            )
            + (
                " Every skipped row of table B quotes that reason."
                if matches
                else " **The skipped rows of table B do not all quote the verdict's reason** "
                "(no skipped row, a table older than the verdict file, or a changed verdict)."
            ),
            "",
            *_place(declared, "table:gate", "table:gate_pillars"),
        ]
    return lines


def _ranked(results: Results, product: str) -> int:
    """How many regime rows of ``product`` carry a distance rank (>= 1) in ``d_ranking``."""
    if "d_ranking" not in results.tables():
        return 0
    long = results.long("d_ranking", "distance_rank")
    mine = long[(long["axis_product"] == product) & (long["value"] >= 1)]
    return len(mine)


def _narrative_d(results: Results, declared: Sequence[str]) -> list[str]:
    lines = [
        "## Study D — the delta-regime ranking",
        "",
        f"> Owner's headline (2026-09-16): {m8b.STUDY_D_HEADLINE}.",
        "",
        "The ranking is the distance of each regime's mean delta to the **common** "
        "minimum-variance delta of the product (the precision-weighted mean of the four regime "
        "rows' implied MV deltas); the min_variance row is shown as a fifth line with its "
        "validity flag, not used as the reference.",
        "",
    ]
    if "d_summary" in results.tables():
        cols = set(results.columns("d_summary"))
        for product in results.rows("d_summary"):
            holds = results.value("d_summary", product, "headline_holds")[0] == 1.0
            closest = str(results.record("d_summary", product, "headline_holds")["note"])
            text = f"- **{product}**: {closest}; the headline "
            text += "holds" if holds else "**does not hold** on this product"
            if "headline_decided" in cols:
                decided = results.value("d_summary", product, "headline_decided")[0]
                if decided == 1.0:
                    text += " (the closest set is decided at 2 stderr)"
                elif decided == 0.0:
                    text += (
                        " (the closest set is **not decided** at 2 stderr: one of its members "
                        "overlaps a farther regime)"
                    )
            text += "."
            if "model_minus_mv" in cols:
                v, se = results.value("d_summary", product, "model_minus_mv")
                if math.isfinite(v):
                    side = "BELOW" if v > 0 else "ABOVE"
                    text += (
                        " The model delta minus the common MV delta is "
                        f"{latex.format_value_text(v, se)}: the MV delta lies {side} the model "
                        "delta."
                    )
            if "model_rank_decided" in cols:
                rank = results.value("d_summary", product, "model_rank")[0]
                decided = results.value("d_summary", product, "model_rank_decided")[0]
                if math.isfinite(rank) and rank >= 1:
                    word = {1.0: "decided", 0.0: "**not decided**"}.get(decided)
                    n = _ranked(results, product)
                    text += f" The model's distance ranks {rank:.0f} of {n}" + (
                        f" ({word} at 2 stderr against the next rank)."
                        if word
                        else (" (the farthest)." if rank == n else ".")
                    )
            valid = results.record("d_summary", product, "mv_row_valid")
            if math.isfinite(float(valid["value"])):
                if valid["value"] == 1.0:
                    text += " The min_variance row is a valid benchmark."
                else:
                    why = f" ({valid['note']})" if valid["note"] else ""
                    text += f" The min_variance row is not a valid benchmark{why}."
            lines.append(text)
        lines.append("")
    lines += _place(
        declared, "table:d_ranking", "table:d_common", "table:d_summary", "figure:d_distance"
    )
    return lines


def narrative(results: Results) -> str:
    have = set(results.tables())
    declared = _declared(results)
    lines = [
        "Artefacts only: this study reads the M8b summary tables and the discriminator verdict; "
        "nothing is simulated or calibrated. Every Monte Carlo number is shown with its standard "
        "error; a z-score carries stderr 1 (its sampling sd); ranks, counts and flags are exact "
        "given the tables' values, and each rank says whether its ordering against the next "
        "rank is decided at 2 stderr (the two value +/- 2 se intervals do not overlap). Desk "
        "sign: the desk is short the product.",
        "",
        *_place(declared, "table:sources"),
    ]
    if "a_ranking" in have:
        lines += _narrative_a(results, declared)
    if "b_reserve" in have or "b_skipped" in have or "gate" in have:
        lines += _narrative_b(results, declared)
    if "d_ranking" in have:
        lines += _narrative_d(results, declared)
    return "\n".join(lines)
