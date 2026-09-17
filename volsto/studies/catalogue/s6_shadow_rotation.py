"""S6 — shadow rotation: the M7 greek beside the M8b simulated recalibration P&L (SPEC §10.2,
owner's M10 Part 2 "S6. Shadow rotation: the M7 greek and the M8b simulated P&L side by side,
per policy (sabr_linked vs sticky_breakeven), in the declared desk-P&L convention, with the
first-order agreement and the nonlinearity at +2, +3 rota").

**Artefacts only** (no Monte Carlo, no leverage, no store point):

* ``m7/p1_marking_shadow_rotation.csv`` — the M7 shadow-rotation greek of the 3y autocall on the
  SPX 2022-12-30 marking fit ``(ssr 1, eps 0.10)`` under each policy (§15 Part 3), written in
  **fractions of notional per rota**; shown here in % of notional (:data:`M7_PERCENT`);
* ``m8b/C/static_<product>__<policy>.json`` — the same greek per book product and policy as
  study C cached it (2·10⁵ paths and particles, seed 2024, the product's reporting unit, the
  desk-P&L numbers already signed), with the declared convention string;
* ``m8b/m8b_table_C.csv`` — study C: per product × rota × recalibration the simulated
  recalibration P&L (desk convention), the static prediction ``desk_pnl_shadow × rota``
  (``desk_pnl_usual × rota`` against the total hedged P&L for the ``none`` rows, a reference
  only), the refit counts and the contamination flags ``refits_at_bound`` / ``contaminated`` /
  ``refits_fallback`` / ``refits_capped`` (``-1`` = not recorded by the run).

**Convention** (:data:`volsto.risk.shadow_rotation.ROTATION_CONVENTION`, quoted from the static
records): rota +1 steepens the 6M 90/110 skew by 0.56 vp; fee = P1 price − LV price; the desk
is SHORT the note, its P&L per +1 rota is −(d fee).

**What is recomputed** from the table's values (so a table written before 2026-09-16 — stems
``recal_se`` / ``total_se`` / ``static_se`` and no ratio or nonlinearity error — gives the same
numbers as one written with exact ``<value>_se`` twins; the read API
:func:`volsto.viewers.api.get_hedging_table` normalises both): the first-order agreement
``ratio = simulated / static`` with its delta-method stderr
(:func:`volsto.studies.m8b.ratio_stderr`),
``z = (simulated − static) / sqrt(se² + se²)`` and the ``|ratio − 1| ≤ tolerance`` flag
(:func:`volsto.studies.m8b.first_order_agreement`'s definitions, the tolerance from the config),
and the nonlinearity ``P&L(rota) / (rota × P&L(+1)) − 1`` at +2 / +3 with the delta-method
stderr (conservative: the runs share the world seed).

**The restricted claim** (owner's decision of 2026-09-16, §8.2): a policy column enters the
conclusion only when it is *clean* — every configured row present and ``ok`` and no refit at a
correlation bound (``refits_at_bound == 0``; ``-1`` = not recorded is not clean).  While the
``sabr_linked`` refits are contaminated, the conclusion is drawn on the ``sticky_breakeven``
column only and the ``sabr_linked`` column is reported, marked, and excluded.  On a clean
column every product's ratio is classed **at 2 stderr** (:func:`classify`: above / within /
below the band ``1 ± tolerance``, or undecided when its interval straddles an edge), **per
rota** — the statement at +1 is not extended to +2 / +3, where the same column is classed
separately.  The desk P&L is a loss for some products and a gain for others (the VKO put, the KO
var), so the claim compares signed values and magnitudes, never a "cost": a simulated P&L is
called a gain or a loss only when its sign is significant (``|z| > SIGN_NSE``), the others are
listed as sign-undecided; and the sentence "the static greek does not overstate the simulated
P&L in magnitude" is printed only when **every** decided product's ratio has its lower 2-stderr
bound at or above 1 (a ratio "within" the band but below 1 blocks it — the claim is then stated
class by class).  The section also states, per policy column, the refits, how many took the
guarded fallback (the base fit's correlation target held on that date) and how many had their
correlation target capped (``refits_fallback`` / ``refits_capped``), and the reading those counts
support: on a fallback refit ``sabr_linked`` holds the correlation ``sticky_breakeven`` always
holds, so on that share of refits the two policies differ only through their break-even vol
targets.

**The policy ordering** — the signed ratio ``sticky / sabr`` of the static desk shadow
(``ordering``) and of the simulated re-marking P&L at every rota (``ordering_sim``), each side
decided at 2 stderr (:func:`ordering_side`), with the product-rota pairs where the two decided
sides differ (``ordering_sides``).  The claim section
also states, from the table-state results, how many of the column's rows no longer have their
task result under ``m8b/C`` and how many newer results the table does not show (a pre-re-run
table).  Nothing is hard-coded to a policy name.

**Table C's state** is part of the results: its modification time, the configured rows missing
from it, the rows not ``ok``, and — because study C's thirty recalibration rows are being re-run
(their pre-re-run result files were moved to ``m8b/C_pre_2026-09-16/``) — the task result files
under ``m8b/C`` newer than the table (a re-run in progress the table does not show yet) and the
table rows whose result file is no longer under ``m8b/C``.  The study works on whatever the
table holds.

Results tables: ``table_c_state``, ``table_c_missing``, ``m7_greek``, ``static_greek``,
``m7_vs_static``, ``pnl``, ``agreement``, ``claim_columns``, ``claim``, ``ordering``,
``ordering_sim``, ``ordering_sides``, ``setup``; ``pnl`` and
``agreement`` are rendered one table per product (``pnl_<product>``, ``agreement_<product>``), and
any table above :data:`volsto.studies.latex.LONGTABLE_MIN_ROWS` rows is split into parts (the
renderer would otherwise switch to an unscaled ``longtable``).  Figures: ``pnl_vs_static``,
``ratio``, ``nonlinearity``, ``m7_greek``.  CI-fast mode (``s6_fast.yaml``) reads the synthetic
outputs of ``tests/_synthetic_store.py::make_synthetic_outputs``.  Checked by
``tests/test_catalogue_s5_s7.py``.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volsto.config import ConfigError
from volsto.hedging.hedger import REFIT_CORRELATION_CAP
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
from volsto.studies.runner import Requirement, StudyContext
from volsto.viewers import api
from volsto.viewers.config import ViewerConfig

TITLE = "S6 - Shadow rotation: the M7 greek and the M8b simulated recalibration P&L"
QUESTION = (
    "Does the static M7 shadow-rotation greek predict the simulated M8b recalibration P&L of "
    "each re-marking policy in the desk convention, and how nonlinear is that P&L at +2 and +3 "
    "rota?"
)
REQUIRED_PARAMS = (
    "products",
    "rotas",
    "recalibrations",
    "first_order_tolerance",
    "m7_surface",
    "m7_product",
)
OPTIONAL_PARAMS: tuple[str, ...] = ()

TABLE_C = "m8b/m8b_table_C.csv"
M7_ROTATION = "m7/p1_marking_shadow_rotation.csv"
STATIC_FILE = "m8b/C/static_{product}__{policy}.json"
RESULTS_DIR = "m8b/C"
COMMAND_C = ".venv/bin/python scripts/m8b.py --study C --resume"
COMMAND_M7 = ".venv/bin/python scripts/m7_p1_marking.py --no-stage3"
#: ``p1_marking_shadow_rotation.csv`` writes its greeks and levels in fractions of notional
#: (``per_rota`` −0.000917 is the −0.0917 % of notional of §15 Part 3).
M7_PERCENT = 100.0
M7_UNIT = "% notional"
POLICY_NONE = "none"
#: The two re-marking policies the refit reading compares: ``sticky_breakeven`` holds the base
#: fit's break-even vol and correlation targets on every refit; ``sabr_linked`` re-reads both from
#: the world's state surface, except on a guarded-fallback refit, where it holds the base fit's
#: correlation target (:class:`volsto.hedging.hedger.RecalibrationRule`).
POLICY_SABR = "sabr_linked"
POLICY_STICKY = "sticky_breakeven"
#: The study-C task-result key prefix (``volsto.studies.m8b.Task.key``).
RESULT_PREFIX = "C__"
#: ``refits_*`` value of a run from before the flag was recorded.
NOT_RECORDED = -1.0
PRODUCT_UNIT = "product unit"
#: The M7 greeks shown, in the report's order (those the CSV carries).
M7_GREEKS: tuple[str, ...] = (
    "lv_rotation",
    "usual",
    "recalibrated",
    "fee_usual",
    "fee_recalibrated",
    "fee_shadow",
    "desk_pnl_usual",
    "desk_pnl_recalibrated",
    "desk_pnl_shadow",
)
M7_LEVELS: tuple[str, ...] = ("p1_level", "lv_level", "fee")
#: The static record's ``[value, stderr]`` entries.
STATIC_GREEKS: tuple[str, ...] = (
    "p1_level",
    "lv_level",
    "fee",
    "lv_rotation",
    "usual",
    "recalibrated",
    "fee_shadow",
    "desk_pnl_usual",
    "desk_pnl_shadow",
)
MISSING_NOTE = "not reported: the value or its stderr is not finite (or the row is absent)"
Z_NOTE = "z-score of the difference (stderrs in quadrature)"
#: Relative difference under which the M7 CSV and a static record hold the same number (the same
#: seed and states computed twice: floating-point noise, not two samples).
IDENTICAL_REL_TOL = 1e-9
#: A first-order comparison is **decided** at this many stderrs: the ratio's interval
#: ``ratio ± CLASSIFY_NSE · se`` against the band ``1 ± tolerance`` (:func:`classify`).
CLASSIFY_NSE = 2.0
#: The simulated P&L counts as a desk gain / loss only when ``|value / stderr|`` exceeds this;
#: otherwise its sign is undecided and listed as such.
SIGN_NSE = 2.0
#: Sides of a policy-ordering ratio ``sticky / sabr`` (:func:`ordering_side`), decided at
#: :data:`CLASSIFY_NSE` stderr: +1 the numerator larger in magnitude (same sign), -1 smaller (same
#: sign), -2 the opposite sign, 0 undecided.
SIDE_WORDS: dict[float, str] = {
    1.0: "{num} larger",
    -1.0: "{num} smaller",
    -2.0: "opposite signs",
    0.0: "undecided",
}
#: ``class_2se`` codes (exact numbers; the class name is in the row's note).
CLASS_CODES: dict[str, float] = {"below": -1.0, "within": 0.0, "above": 1.0, "undecided": 2.0}
CLASS_LEGEND = (
    f"class at {CLASSIFY_NSE:g} stderr: +1 above (ratio - 2 se > 1 + tol: same sign, |simulated| "
    "larger), 0 within (the whole interval inside 1 +/- tol), -1 below (ratio + 2 se < 1 - tol: "
    "|simulated| smaller, or the opposite sign), 2 undecided (the interval straddles a band edge)"
)
IDENTICAL_NOTE = "identical to 1e-9 relative: the same seed and states (a reproduction)"


def validate_params(params: Mapping[str, Any]) -> None:
    products = params["products"]
    if (
        not isinstance(products, list)
        or not products
        or not all(isinstance(p, str) for p in products)
    ):
        raise ConfigError(f"products must be a non-empty list of names: {products}")
    rotas = params["rotas"]
    if not isinstance(rotas, list) or not rotas or 1.0 not in [float(r) for r in rotas]:
        raise ConfigError(f"rotas must be a list holding +1 (the nonlinearity base): {rotas}")
    recals = params["recalibrations"]
    if (
        not isinstance(recals, list)
        or not set(recals) <= set(m8b.RECALIBRATIONS_C)
        or not [r for r in recals if r != POLICY_NONE]
    ):
        raise ConfigError(
            f"recalibrations must be a subset of {list(m8b.RECALIBRATIONS_C)} with at least one "
            f"policy: {recals}"
        )
    tol = float(params["first_order_tolerance"])
    if not 0.0 < tol < 1.0:
        raise ConfigError(f"first_order_tolerance must lie in (0, 1): {tol}")
    if params["m7_product"] not in products:
        raise ConfigError(f"m7_product {params['m7_product']!r} is not in products")
    if not isinstance(params["m7_surface"], str) or not params["m7_surface"]:
        raise ConfigError("m7_surface must name the M7 surface (the CSV's `surface` column)")


@dataclasses.dataclass(frozen=True)
class Found:
    """One product's first-order comparison at one rota (:func:`_claim`)."""

    product: str
    ratio: float
    ratio_se: float
    sim: float
    sim_se: float
    cls: str

    @property
    def lower(self) -> float:
        """The ratio's lower :data:`CLASSIFY_NSE`-stderr bound."""
        return self.ratio - CLASSIFY_NSE * self.ratio_se

    @property
    def sign(self) -> int:
        """+1 a desk gain, -1 a desk loss, 0 when ``|simulated / se| <= SIGN_NSE`` (the sign of
        the simulated P&L is not significant)."""
        if not (self.sim_se > 0 and abs(self.sim / self.sim_se) > SIGN_NSE):
            return 0
        return 1 if self.sim > 0 else -1


def classify(ratio: float, stderr: float, tol: float, nse: float = CLASSIFY_NSE) -> str:
    """The class of a simulated / static ratio (:data:`CLASS_CODES`): ``above`` when
    ``ratio − nse·se > 1 + tol``, ``below`` when ``ratio + nse·se < 1 − tol``, ``within`` when
    the whole interval lies inside ``1 ± tol``, ``undecided`` otherwise (and for a ratio or a
    stderr that is not finite)."""
    if not (math.isfinite(ratio) and math.isfinite(stderr)):
        return "undecided"
    lo, hi = ratio - nse * stderr, ratio + nse * stderr
    if lo > 1.0 + tol:
        return "above"
    if hi < 1.0 - tol:
        return "below"
    if lo >= 1.0 - tol and hi <= 1.0 + tol:
        return "within"
    return "undecided"


def ordering_side(ratio: float, stderr: float, nse: float = CLASSIFY_NSE) -> float:
    """The side of a signed ratio ``num / den`` (:data:`SIDE_WORDS`): +1 when
    ``ratio − nse·se > 1``, −2 when ``ratio + nse·se < 0``, −1 when the interval lies inside
    ``(0, 1)``, 0 otherwise (and for a ratio or stderr that is not finite)."""
    if not (math.isfinite(ratio) and math.isfinite(stderr)):
        return 0.0
    lo, hi = ratio - nse * stderr, ratio + nse * stderr
    if lo > 1.0:
        return 1.0
    if hi < 0.0:
        return -2.0
    if lo > 0.0 and hi < 1.0:
        return -1.0
    return 0.0


def _policies(params: Mapping[str, Any]) -> list[str]:
    return [p for p in params["recalibrations"] if p != POLICY_NONE]


def static_rel(product: str, policy: str) -> str:
    return STATIC_FILE.format(product=m8b.slug(product), policy=m8b.slug(policy))


def requirements(ctx: StudyContext) -> list[Requirement]:
    p = ctx.params
    out = [
        ctx.artefact_requirement(TABLE_C, "M8b study C summary table", COMMAND_C),
        ctx.artefact_requirement(M7_ROTATION, "M7 shadow-rotation greek table", COMMAND_M7),
    ]
    for product in p["products"]:
        for policy in _policies(p):
            out.append(
                ctx.artefact_requirement(
                    static_rel(product, policy),
                    f"study-C static greek of {product} under {policy}",
                    COMMAND_C,
                )
            )
    return out


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def _num(v: Any) -> float:
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


def _flag(v: Any) -> float:
    if isinstance(v, bool | np.bool_):
        return float(bool(v))
    s = _text(v).strip().lower()
    if s in ("true", "1", "1.0"):
        return 1.0
    if s in ("false", "0", "0.0"):
        return 0.0
    return math.nan


def _unit(u: Any) -> str:
    s = _text(u).strip()
    return s if s else DIMENSIONLESS


def _rota_label(rota: float) -> str:
    return f"+{rota:g}" if rota > 0 else f"{rota:g}"


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
    """A Monte Carlo number with its stderr, or a missing number (NaN, noted)."""
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
            note=MISSING_NOTE,
            axes=axes,
        )


def _pair(rec: Mapping[str, Any], col: str) -> tuple[float, float]:
    return _num(rec.get(col)), _num(rec.get(f"{col}_stderr"))


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [{str(k): v for k, v in r.items()} for r in df.to_dict("records")]


def _utc(ts: float) -> str:
    return _dt.datetime.fromtimestamp(ts, _dt.UTC).isoformat(timespec="seconds")


def _result_files(root: Path) -> list[tuple[Path, tuple[str, float, str] | None]]:
    """The study-C task results under ``root`` (key prefix ``C__``) with their ``(product, rota,
    policy)`` — ``None`` for a file that does not load (a run writing it right now)."""
    out: list[tuple[Path, tuple[str, float, str] | None]] = []
    if not root.is_dir():
        return out
    for p in sorted(root.glob(f"{RESULT_PREFIX}*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            key = (str(doc["product"]), float(doc["rota"]), str(doc["policy"]))
        except (OSError, ValueError, KeyError, TypeError):
            out.append((p, None))
            continue
        out.append((p, key))
    return out


# --------------------------------------------------------------------------------------------
# compute
# --------------------------------------------------------------------------------------------


def _read_table_c(ctx: StudyContext) -> tuple[pd.DataFrame, Path]:
    path = ctx.artefact(TABLE_C)
    cfg = ViewerConfig(
        cache_root=ctx.cache_root, store_root=ctx.store_root, outputs_root=ctx.outputs_root
    )
    df = api.get_hedging_table(cfg, "C")
    return df, path


def _key(rec: Mapping[str, Any]) -> tuple[str, float, str]:
    return (_text(rec.get("product")), _num(rec.get("rota")), _text(rec.get("recalibration")))


def _table_c_state(
    b: ResultsBuilder,
    ctx: StudyContext,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    expected: Sequence[tuple[str, float, str]],
    files: Sequence[tuple[Path, tuple[str, float, str] | None]],
) -> None:
    source = f"artefact:{TABLE_C}"
    mtime = path.stat().st_mtime
    present = [_key(r) for r in rows]
    present_set = set(present)
    missing = [k for k in expected if k not in present_set]
    extra = [k for k in present if k not in set(expected)]
    not_ok = [r for r in rows if _text(r.get("status")) != "ok"]
    newer = [p for p, _ in files if p.stat().st_mtime > mtime]
    unreadable = [p for p, k in files if k is None]
    file_keys = {k for _, k in files if k is not None}
    without_file = [k for k in present if k not in file_keys]
    recal_rows = [r for r in rows if _text(r.get("recalibration")) != POLICY_NONE]
    row = "table C"
    for col, value, note in (
        ("mtime_epoch_s", mtime, f"modified {_utc(mtime)}"),
        ("rows", float(len(rows)), ""),
        ("rows_expected", float(len(expected)), "products x rotas x recalibrations of the config"),
        ("rows_missing", float(len(missing)), ""),
        ("rows_beyond_config", float(len(extra)), "rows of the table outside the configured set"),
        ("rows_not_ok", float(len(not_ok)), ""),
        ("recalibration_rows", float(len(recal_rows)), ""),
        (
            "contaminated_rows",
            float(sum(_flag(r.get("contaminated")) == 1.0 for r in recal_rows)),
            "refits at a correlation bound",
        ),
        (
            "at_bound_not_recorded",
            float(sum(_num(r.get("refits_at_bound")) == NOT_RECORDED for r in recal_rows)),
            "",
        ),
        (
            "fallback_recorded",
            float(
                sum(
                    math.isfinite(_num(r.get("refits_fallback")))
                    and _num(r.get("refits_fallback")) != NOT_RECORDED
                    for r in recal_rows
                )
            ),
            "rows whose refits_fallback / refits_capped were recorded (the rebuilt refit)",
        ),
        ("result_files", float(len(files)), f"task results under {RESULTS_DIR}"),
        (
            "result_files_newer",
            float(len(newer)),
            "task results written after the table (a run the table does not show yet)",
        ),
        ("result_files_unreadable", float(len(unreadable)), ""),
        (
            "rows_without_result_file",
            float(len(without_file)),
            f"table rows whose task result is no longer under {RESULTS_DIR}",
        ),
    ):
        b.add_exact("table_c_state", row, col, value, unit="", source=source, note=note)
    for product, rota, policy in missing:
        b.add_exact(
            "table_c_missing",
            f"{product} | {policy} | {_rota_label(rota)}",
            "missing",
            1.0,
            unit="",
            source=source,
            note="absent from the table",
            axes={"product": product, "policy": policy, "rota": rota},
        )
    for r in not_ok:
        product, rota, policy = _key(r)
        b.add_exact(
            "table_c_missing",
            f"{product} | {policy} | {_rota_label(rota)}",
            "missing",
            1.0,
            unit="",
            source=source,
            note=f"status {_text(r.get('status'))}: {_text(r.get('reason'))}",
            axes={"product": product, "policy": policy, "rota": rota},
        )
    ctx.record(
        "table_c",
        {
            "modified_utc": _utc(mtime),
            "rows": len(rows),
            "missing": [list(k) for k in missing],
            "result_files_newer": [p.name for p in newer],
        },
    )
    ctx.log.info(
        "table C modified %s: %d rows, %d of %d configured rows missing, %d not ok; %d task "
        "results under %s newer than the table",
        _utc(mtime),
        len(rows),
        len(missing),
        len(expected),
        len(not_ok),
        len(newer),
        RESULTS_DIR,
    )


def artefact_root(ctx: StudyContext) -> Path:
    return ctx.outputs_root


def _m7(b: ResultsBuilder, ctx: StudyContext) -> dict[str, dict[str, tuple[float, float]]]:
    """``m7_greek`` rows (one per policy of the configured surface); returns
    ``{policy: {greek: (value %, se %)}}``."""
    p = ctx.params
    path = ctx.artefact(M7_ROTATION)
    df = api.read_study_csv(path)
    df = df[df["surface"].astype(str) == str(p["m7_surface"])]
    source = f"artefact:{M7_ROTATION}"
    out: dict[str, dict[str, tuple[float, float]]] = {}
    for policy, g in df.groupby("policy", sort=False):
        pol = str(policy)
        axes = {"policy": pol, "product": str(p["m7_product"])}
        greeks = {str(r["greek"]): r for r in _records(g)}
        out[pol] = {}
        for name in M7_GREEKS:
            if name not in greeks:
                continue
            v, se = _pair(greeks[name], "per_rota")
            v, se = M7_PERCENT * v, M7_PERCENT * se
            out[pol][name] = (v, se)
            add_mc(
                b,
                "m7_greek",
                pol,
                name,
                v,
                se,
                unit=M7_UNIT,
                source=source,
                axes=axes,
                note="per +1 rota",
            )
        first = _records(g)[0]
        for name in M7_LEVELS:
            v, se = _pair(first, name)
            add_mc(
                b,
                "m7_greek",
                pol,
                name,
                M7_PERCENT * v,
                M7_PERCENT * se,
                unit=M7_UNIT,
                source=source,
                axes=axes,
                note="level",
            )
    return out


def _statics(b: ResultsBuilder, ctx: StudyContext) -> dict[tuple[str, str], dict[str, Any]]:
    p = ctx.params
    docs: dict[tuple[str, str], dict[str, Any]] = {}
    for product in p["products"]:
        for policy in _policies(p):
            rel = static_rel(product, policy)
            doc = json.loads(ctx.artefact(rel).read_text(encoding="utf-8"))
            docs[(product, policy)] = doc
            row = f"{product} | {policy}"
            axes = {"product": product, "policy": policy}
            source = f"artefact:{rel}"
            unit = _unit(doc.get("unit"))
            for name in STATIC_GREEKS:
                pair = doc.get(name)
                v, se = (
                    (_num(pair[0]), _num(pair[1]))
                    if isinstance(pair, list | tuple) and len(pair) == 2
                    else (math.nan, math.nan)
                )
                add_mc(b, "static_greek", row, name, v, se, unit=unit, source=source, axes=axes)
            b.add_exact(
                "static_greek",
                row,
                "base_fit_is_marking_fit",
                _flag(doc.get("base_fit_equals_marking_fit")),
                unit="",
                source=source,
                note=str(doc.get("convention", "")),
                axes=axes,
            )
            for name in ("n_paths", "n_particles"):
                b.add_exact(
                    "static_greek",
                    row,
                    name,
                    _num(doc.get(name)),
                    unit="",
                    source=source,
                    axes=axes,
                )
    return docs


def _m7_vs_static(
    b: ResultsBuilder,
    ctx: StudyContext,
    m7: Mapping[str, Mapping[str, tuple[float, float]]],
    docs: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    product = str(ctx.params["m7_product"])
    for policy in _policies(ctx.params):
        greeks = m7.get(policy)
        doc = docs.get((product, policy))
        if not greeks or doc is None:
            continue
        axes = {"policy": policy, "product": product}
        source = f"artefact:{M7_ROTATION};artefact:{static_rel(product, policy)}"
        for name in ("usual", "desk_pnl_shadow"):
            if name not in greeks:
                continue
            mv, mse = greeks[name]
            pair = doc.get(name)
            if not isinstance(pair, list | tuple) or len(pair) != 2:
                continue
            sv, sse = _num(pair[0]), _num(pair[1])
            add_mc(
                b,
                "m7_vs_static",
                policy,
                f"m7_{name}",
                mv,
                mse,
                unit=M7_UNIT,
                source=source,
                axes=axes,
            )
            add_mc(
                b,
                "m7_vs_static",
                policy,
                f"static_{name}",
                sv,
                sse,
                unit=M7_UNIT,
                source=source,
                axes=axes,
            )
            identical = abs(sv - mv) <= IDENTICAL_REL_TOL * max(abs(sv), abs(mv), 1e-300)
            if identical:
                b.add_exact(
                    "m7_vs_static",
                    policy,
                    f"diff_{name}",
                    0.0,
                    unit=M7_UNIT,
                    source=source,
                    axes=axes,
                    note=IDENTICAL_NOTE,
                )
                b.add_exact(
                    "m7_vs_static",
                    policy,
                    f"z_{name}",
                    0.0,
                    unit="",
                    source=source,
                    axes=axes,
                    note=IDENTICAL_NOTE,
                )
                continue
            se = math.hypot(mse, sse)
            add_mc(
                b,
                "m7_vs_static",
                policy,
                f"diff_{name}",
                sv - mv,
                se,
                unit=M7_UNIT,
                source=source,
                axes=axes,
                note="treated as independent runs: stderrs in quadrature",
            )
            b.add_exact(
                "m7_vs_static",
                policy,
                f"z_{name}",
                (sv - mv) / se if se > 0 else math.nan,
                unit="",
                source=source,
                note=Z_NOTE,
                axes=axes,
            )


def _sim(rec: Mapping[str, Any]) -> tuple[float, float]:
    """The simulated number a row is compared on: the total hedged desk P&L for ``none``, the
    recalibration P&L otherwise (``table_C``'s rule)."""
    if _text(rec.get("recalibration")) == POLICY_NONE:
        return _pair(rec, "total_pnl_desk")
    return _pair(rec, "recal_pnl_desk")


def _agreement(b: ResultsBuilder, ctx: StudyContext, rows: Sequence[Mapping[str, Any]]) -> None:
    p = ctx.params
    tol = float(p["first_order_tolerance"])
    source = f"artefact:{TABLE_C}"
    ok = {_key(r): r for r in rows if _text(r.get("status")) == "ok"}
    order = [
        (product, float(rota), policy)
        for product in p["products"]
        for policy in p["recalibrations"]
        for rota in p["rotas"]
    ]
    for key in order:
        rec = ok.get(key)
        if rec is None:
            continue
        product, rota, policy = key
        row = f"{product} | {policy} | {_rota_label(rota)}"
        axes = {"product": product, "policy": policy, "rota": rota}
        unit = _unit(rec.get("unit"))
        sim, sim_se = _sim(rec)
        pred, pred_se = _pair(rec, "static_prediction")
        # --- the P&L levels (product unit) -----------------------------------------------------
        if policy == POLICY_NONE:
            b.add_exact(
                "pnl",
                row,
                "recal_pnl",
                0.0,
                unit=unit,
                source=source,
                axes=axes,
                note="no re-marking: exactly zero",
            )
        else:
            v, se = _pair(rec, "recal_pnl_desk")
            add_mc(b, "pnl", row, "recal_pnl", v, se, unit=unit, source=source, axes=axes)
        add_mc(
            b,
            "pnl",
            row,
            "static_x_rota",
            pred,
            pred_se,
            unit=unit,
            source=source,
            axes=axes,
            note=_text(rec.get("prediction_of")),
        )
        v, se = _pair(rec, "total_pnl_desk")
        add_mc(b, "pnl", row, "total_pnl", v, se, unit=unit, source=source, axes=axes)
        # --- first-order agreement (dimensionless) -------------------------------------------
        ratio, _, z = m8b.first_order_agreement(sim, sim_se, pred, pred_se)
        ratio_se = m8b.ratio_stderr(sim, sim_se, pred, pred_se)
        add_mc(
            b,
            "agreement",
            row,
            "ratio",
            ratio,
            ratio_se,
            unit=DIMENSIONLESS,
            source=source,
            axes=axes,
        )
        b.add_exact("agreement", row, "z", z, unit="", source=source, axes=axes, note=Z_NOTE)
        within = abs(ratio - 1.0) <= tol if math.isfinite(ratio) else False
        b.add_exact(
            "agreement",
            row,
            "within_tol_point",
            1.0 if within else 0.0,
            unit="",
            source=source,
            axes=axes,
            note=f"|ratio - 1| <= {tol:g} on the point estimate (M8b's within_30pct); not used "
            "by the claim",
        )
        cls = classify(ratio, ratio_se, tol)
        b.add_exact(
            "agreement",
            row,
            "class_2se",
            CLASS_CODES[cls],
            unit="",
            source=source,
            axes=axes,
            note=cls,
        )
        if rota != 1.0:
            base = ok.get((product, 1.0, policy))
            if base is not None:
                bv, bse = _sim(base)
                nonlin = m8b.nonlinearity(sim, rota, bv)
                nonlin_se = m8b.ratio_stderr(sim, sim_se, rota * bv, rota * bse)
            else:
                nonlin, nonlin_se = math.nan, math.nan
            add_mc(
                b,
                "agreement",
                row,
                "nonlinearity",
                nonlin,
                nonlin_se,
                unit=DIMENSIONLESS,
                source=source,
                axes=axes,
                note="delta-method stderr, conservative (the runs share the world seed)",
            )
        for col in ("n_refits", "refits_at_bound", "refits_fallback", "refits_capped"):
            v = _num(rec.get(col)) if col in rec else NOT_RECORDED
            b.add_exact(
                "agreement",
                row,
                col,
                v if math.isfinite(v) else NOT_RECORDED,
                unit="",
                source=source,
                axes=axes,
                note="-1 = not recorded by the run" if col != "n_refits" else "",
            )
        cont = _flag(rec.get("contaminated"))
        b.add_exact(
            "agreement",
            row,
            "contaminated",
            cont,
            unit="",
            source=source,
            axes=axes,
            note=_text(rec.get("refit_dates")),
        )


def _recorded(rec: Mapping[str, Any], col: str) -> float | None:
    """A refit-flag count as recorded (``None`` for an absent column, NaN or -1)."""
    v = _num(rec.get(col)) if col in rec else math.nan
    return None if not math.isfinite(v) or v == NOT_RECORDED else v


def _refit_counts(ok: Sequence[Mapping[str, Any]]) -> list[tuple[str, float, str]]:
    """``claim_columns`` entries: the column's refits, how many took the guarded fallback and
    how many had their correlation target capped (summed over the rows that record both), the
    rows that do not record them, and the two shares."""
    recorded = [
        r
        for r in ok
        if _recorded(r, "refits_fallback") is not None and _recorded(r, "refits_capped") is not None
    ]
    refits = float(sum(_num(r.get("n_refits")) for r in ok))
    refits_rec = float(sum(_num(r.get("n_refits")) for r in recorded))
    fallback = float(sum(_recorded(r, "refits_fallback") or 0.0 for r in recorded))
    capped = float(sum(_recorded(r, "refits_capped") or 0.0 for r in recorded))
    return [
        ("refits", refits, "refits over the column's ok rows"),
        ("refits_recorded", refits_rec, "refits of the rows recording the fallback / cap counts"),
        (
            "refits_fallback",
            fallback,
            "refits that held the base fit's correlation target (the guarded fallback: step 0 "
            "degenerate on that date's state surface)",
        ),
        (
            "refits_capped",
            capped,
            f"refits whose correlation target was capped at |Corr_BE| <= "
            f"{REFIT_CORRELATION_CAP:g}",
        ),
        (
            "rows_counts_unrecorded",
            float(len(ok) - len(recorded)),
            "rows without the fallback / cap counts (-1: a run from before the rebuilt refit)",
        ),
        ("fallback_share", fallback / refits_rec if refits_rec else math.nan, "fallback / refits"),
        ("capped_share", capped / refits_rec if refits_rec else math.nan, "capped / refits"),
    ]


def _claim(
    b: ResultsBuilder,
    ctx: StudyContext,
    rows: Sequence[Mapping[str, Any]],
    file_keys: set[tuple[str, float, str]],
) -> None:
    """``claim_columns`` (per policy: clean or not, and how many of its rows still have their
    task result under ``m8b/C``) and ``claim`` (per policy and rota: the :func:`classify`
    counts, the smallest and largest ratio with their own stderr, the desk gains and losses)."""
    p = ctx.params
    tol = float(p["first_order_tolerance"])
    source = f"artefact:{TABLE_C}"
    by_key = {_key(r): r for r in rows}
    rotas = [float(r) for r in p["rotas"]]
    for policy in _policies(p):
        keys = [(prod, rota, policy) for prod in p["products"] for rota in rotas]
        present = [by_key[k] for k in keys if k in by_key]
        ok = [r for r in present if _text(r.get("status")) == "ok"]
        at_bound = [_num(r.get("refits_at_bound")) for r in ok]
        contaminated = sum(v > 0 for v in at_bound if math.isfinite(v))
        unrecorded = sum((not math.isfinite(v)) or v == NOT_RECORDED for v in at_bound)
        clean = len(ok) == len(keys) and contaminated == 0 and unrecorded == 0
        axes = {"policy": policy}
        for col, value, note in (
            ("clean", 1.0 if clean else 0.0, "1 = enters the conclusion"),
            ("rows_expected", float(len(keys)), ""),
            ("rows_ok", float(len(ok)), ""),
            ("rows_contaminated", float(contaminated), "refits at a correlation bound"),
            ("rows_at_bound_unrecorded", float(unrecorded), ""),
            (
                "rows_without_result_file",
                float(sum(_key(r) not in file_keys for r in present)),
                f"rows whose task result is no longer under {RESULTS_DIR}",
            ),
            (
                "mean_refits",
                float(np.mean([_num(r.get("n_refits")) for r in ok])) if ok else math.nan,
                "",
            ),
            *_refit_counts(ok),
        ):
            b.add_exact(
                "claim_columns", policy, col, value, unit="", source=source, note=note, axes=axes
            )
        for rota in rotas:
            row = f"{policy} | {_rota_label(rota)}"
            raxes = {"policy": policy, "rota": rota}
            found: list[Found] = []
            for r in ok:
                if _num(r.get("rota")) != rota:
                    continue
                sim, sim_se = _sim(r)
                pred, pred_se = _pair(r, "static_prediction")
                ratio = m8b.first_order_agreement(sim, sim_se, pred, pred_se)[0]
                ratio_se = m8b.ratio_stderr(sim, sim_se, pred, pred_se)
                if math.isfinite(ratio) and math.isfinite(ratio_se):
                    found.append(
                        Found(
                            _text(r.get("product")),
                            ratio,
                            ratio_se,
                            sim,
                            sim_se,
                            classify(ratio, ratio_se, tol),
                        )
                    )
            counts = {c: [f.product for f in found if f.cls == c] for c in CLASS_CODES}
            decided = [f for f in found if f.cls != "undecided"]
            not_larger = [f.product for f in decided if f.lower >= 1.0]
            gains = [f.product for f in found if f.sign == 1]
            losses = [f.product for f in found if f.sign == -1]
            unsigned = [f.product for f in found if f.sign == 0]
            for col, value, note in (
                ("pairs", float(len(found)), "products with a finite ratio and stderr"),
                ("above", float(len(counts["above"])), ", ".join(counts["above"])),
                ("within", float(len(counts["within"])), ", ".join(counts["within"])),
                ("undecided", float(len(counts["undecided"])), ", ".join(counts["undecided"])),
                ("below", float(len(counts["below"])), ", ".join(counts["below"])),
                ("decided", float(len(decided)), "above + within + below"),
                (
                    "decided_static_not_larger",
                    float(len(not_larger)),
                    "decided products whose ratio - 2 se >= 1 (|simulated| >= |static|, same "
                    "sign): " + ", ".join(not_larger),
                ),
                ("desk_gains", float(len(gains)), ", ".join(gains)),
                ("desk_losses", float(len(losses)), ", ".join(losses)),
                ("sign_undecided", float(len(unsigned)), ", ".join(unsigned)),
            ):
                b.add_exact("claim", row, col, value, unit="", source=source, note=note, axes=raxes)
            for col, pick in (("min_ratio", min), ("max_ratio", max)):
                chosen = pick(found, key=lambda f: f.ratio) if found else None
                add_mc(
                    b,
                    "claim",
                    row,
                    col,
                    chosen.ratio if chosen else math.nan,
                    chosen.ratio_se if chosen else math.nan,
                    unit=DIMENSIONLESS,
                    source=source,
                    axes=raxes,
                    note=f"the {chosen.product} row, with its own stderr" if chosen else "",
                )


def _ordering(
    b: ResultsBuilder,
    ctx: StudyContext,
    rows: Sequence[Mapping[str, Any]],
    docs: Mapping[tuple[str, str], Mapping[str, Any]],
) -> None:
    """The policy ordering per product: the static desk shadows and their ratio
    ``sticky / sabr`` (``ordering``), the same ratio of the simulated re-marking P&L at every
    rota (``ordering_sim``), and the side of each ratio with the disagreements between the static
    and the simulated side (``ordering_sides``, :func:`ordering_side`)."""
    pols = _policies(ctx.params)
    if len(pols) != 2:
        return
    a, c = pols
    ok = {_key(r): r for r in rows if _text(r.get("status")) == "ok"}
    tsrc = f"artefact:{TABLE_C}"
    rotas = [float(r) for r in ctx.params["rotas"]]
    for product in ctx.params["products"]:
        da, dc = docs.get((product, a)), docs.get((product, c))
        if da is None or dc is None:
            continue
        unit = _unit(da.get("unit"))
        axes = {"product": product}
        src = f"artefact:{static_rel(product, a)};artefact:{static_rel(product, c)}"
        sa = (_num(da["desk_pnl_shadow"][0]), _num(da["desk_pnl_shadow"][1]))
        sc = (_num(dc["desk_pnl_shadow"][0]), _num(dc["desk_pnl_shadow"][1]))
        add_mc(b, "ordering", product, f"static_{a}", *sa, unit=unit, source=src, axes=axes)
        add_mc(b, "ordering", product, f"static_{c}", *sc, unit=unit, source=src, axes=axes)
        s_ratio = sc[0] / sa[0] if sa[0] else math.nan
        s_se = m8b.ratio_stderr(sc[0], sc[1], sa[0], sa[1])
        add_mc(
            b,
            "ordering",
            product,
            f"static_{c}_over_{a}",
            s_ratio,
            s_se,
            unit=DIMENSIONLESS,
            source=src,
            axes=axes,
            note="delta method, the two static runs treated as independent",
        )
        static_side = ordering_side(s_ratio, s_se)
        b.add_exact(
            "ordering_sides",
            product,
            "static",
            static_side,
            unit="",
            source=src,
            axes=axes,
            note=SIDE_WORDS[static_side].format(num=c),
        )
        contaminated = [
            _flag(ok[(product, r, a)].get("contaminated")) == 1.0
            for r in rotas
            if (product, r, a) in ok
        ]
        b.add_exact(
            "ordering",
            product,
            f"{a}_contaminated_rows",
            float(sum(contaminated)),
            unit="",
            source=tsrc,
            axes=axes,
        )
        for rota in rotas:
            ra, rc = ok.get((product, rota, a)), ok.get((product, rota, c))
            if ra is None or rc is None:
                continue
            va, vc = _pair(ra, "recal_pnl_desk"), _pair(rc, "recal_pnl_desk")
            ratio = vc[0] / va[0] if va[0] else math.nan
            ratio_se = m8b.ratio_stderr(vc[0], vc[1], va[0], va[1])
            label = _rota_label(rota)
            add_mc(
                b,
                "ordering_sim",
                product,
                f"ratio@{label}",
                ratio,
                ratio_se,
                unit=DIMENSIONLESS,
                source=tsrc,
                axes={**axes, "rota": rota},
                note="delta method; conservative (the runs share the world seed)",
            )
            side = ordering_side(ratio, ratio_se)
            b.add_exact(
                "ordering_sides",
                product,
                f"sim@{label}",
                side,
                unit="",
                source=tsrc,
                axes=axes,
                note=SIDE_WORDS[side].format(num=c),
            )
            disagree = static_side != 0.0 and side != 0.0 and side != static_side
            b.add_exact(
                "ordering_sides",
                product,
                f"disagree@{label}",
                1.0 if disagree else 0.0,
                unit="",
                source=f"{src};{tsrc}",
                axes=axes,
                note="1 = the static and the simulated sides are both decided and differ",
            )


def compute(ctx: StudyContext) -> Results:
    p = ctx.params
    b = ResultsBuilder()
    df, path = _read_table_c(ctx)
    rows = _records(df)
    expected = [
        (product, float(rota), policy)
        for product in p["products"]
        for rota in p["rotas"]
        for policy in p["recalibrations"]
    ]
    files = _result_files(artefact_root(ctx) / RESULTS_DIR)
    file_keys = {k for _, k in files if k is not None}
    _table_c_state(b, ctx, rows, path, expected, files)
    m7 = _m7(b, ctx)
    docs = _statics(b, ctx)
    _m7_vs_static(b, ctx, m7, docs)
    if rows:
        _agreement(b, ctx, rows)
        _claim(b, ctx, rows, file_keys)
        _ordering(b, ctx, rows, docs)
    b.add_exact(
        "setup",
        "study C",
        "first_order_tolerance",
        float(p["first_order_tolerance"]),
        unit="",
        source="computed",
        note=f"M8b's own constant: {m8b.FIRST_ORDER_TOLERANCE:g}",
    )
    return b.build()


# --------------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------------


def _units_caption(results: Results, table: str) -> str:
    long = results.long(table)
    if "axis_product" not in long.columns:
        return ""
    seen: dict[str, str] = {}
    for prod, u in zip(long["axis_product"], long["unit"]):
        if isinstance(prod, str) and u and u != DIMENSIONLESS:
            seen.setdefault(prod, str(u))
    return "; ".join(f"{k}: {v}" for k, v in seen.items())


def tables(results: Results) -> list[TableSpec]:
    have = set(results.tables())
    out = [
        TableSpec(
            "table_c_state",
            "The state of study C's table as read: modification time (UTC in the narrative), "
            "rows present / configured / missing / not ok, contamination flags recorded, and the "
            "task results under m8b/C against the table.",
            "table_c_state",
            (
                Column("rows", "rows", digits=3),
                Column("rows_expected", "configured", digits=3),
                Column("rows_missing", "missing", digits=3),
                Column("rows_not_ok", "not ok", digits=3),
                Column("contaminated_rows", "contaminated", digits=3),
                Column("fallback_recorded", "fallback recorded", digits=3),
                Column("result_files", "result files", digits=3),
                Column("result_files_newer", "files newer than table", digits=3),
                Column("rows_without_result_file", "rows without file", digits=3),
            ),
        )
    ]
    if "table_c_missing" in have:
        out.append(
            TableSpec(
                "table_c_missing",
                "Configured study-C rows the table does not carry (absent or not ok).",
                "table_c_missing",
                (Column("missing", "missing", digits=1),),
                row_header="product | policy | rota",
            )
        )
    if "m7_greek" in have:
        cols = [c for c in (*M7_GREEKS, *M7_LEVELS) if c in results.columns("m7_greek")]
        out.append(
            TableSpec(
                "m7_greek",
                "The M7 shadow-rotation greek of the 3y autocall (SPX 2022-12-30, ssr 1, eps "
                "0.10), per +1 rota, and the price levels (M7 CSV, fractions converted to % of "
                "notional).",
                "m7_greek",
                tuple(Column(c, c.replace("_", " ")) for c in cols),
                row_header="policy",
            )
        )
    if "static_greek" in have:
        out.append(
            TableSpec(
                "static_greek",
                "Study C's static greek per product and policy (2e5 paths and particles), per +1 "
                f"rota; levels. Units: {_units_caption(results, 'static_greek')}.",
                "static_greek",
                (
                    Column("fee", "fee level", unit=PRODUCT_UNIT),
                    Column("lv_rotation", "LV rotation", unit=PRODUCT_UNIT),
                    Column("usual", "P1 usual", unit=PRODUCT_UNIT),
                    Column("recalibrated", "P1 recalibrated", unit=PRODUCT_UNIT),
                    Column("fee_shadow", "fee shadow", unit=PRODUCT_UNIT),
                    Column("desk_pnl_usual", "desk usual", unit=PRODUCT_UNIT),
                    Column("desk_pnl_shadow", "desk shadow", unit=PRODUCT_UNIT),
                    Column("base_fit_is_marking_fit", "base = marking fit", digits=1),
                ),
                row_header="product | policy",
            )
        )
    if "m7_vs_static" in have:
        out.append(
            TableSpec(
                "m7_vs_static",
                "Reproduction: the M7 greek of the 3y autocall against study C's static greek; a "
                "difference below 1e-9 relative is the same computation (same seed and states) "
                "and is shown as an exact 0, otherwise the stderrs are combined in quadrature.",
                "m7_vs_static",
                tuple(Column(c, c.replace("_", " ")) for c in results.columns("m7_vs_static")),
                row_header="policy",
            )
        )
    for product, rows in _rows_by(results, "pnl", "product").items():
        units = {str(u) for u in results.long("pnl").query("axis_product == @product")["unit"]}
        unit = next((u for u in sorted(units) if u and u != DIMENSIONLESS), "")
        out.append(
            TableSpec(
                f"pnl_{m8b.slug(product)}",
                f"{product}, study C in the desk convention: the simulated recalibration P&L, the "
                "static prediction x rota (desk_pnl_shadow; desk_pnl_usual for the none rows) and "
                "the total hedged P&L.",
                "pnl",
                (
                    Column("recal_pnl", "recal P&L (desk)", unit=unit),
                    Column("static_x_rota", "static x rota", unit=unit),
                    Column("total_pnl", "total P&L (desk)", unit=unit),
                ),
                rows=tuple(rows),
                row_header="product | policy | rota",
            )
        )
    for product, rows in _rows_by(results, "agreement", "product").items():
        out.append(
            TableSpec(
                f"agreement_{m8b.slug(product)}",
                f"{product}: first-order agreement (simulated / static, delta-method stderr; z; "
                f"the {CLASS_LEGEND}), nonlinearity P&L(rota)/(rota P&L(+1)) - 1, refit counts "
                "and flags (-1 = not recorded; contaminated = refits at a correlation bound). The "
                "none rows compare the total hedged P&L with desk_pnl_usual x rota: a reference.",
                "agreement",
                (
                    Column("ratio", "ratio"),
                    Column("z", "z", digits=3),
                    Column("class_2se", "class (2 se)", digits=1),
                    Column("nonlinearity", "nonlinearity"),
                    Column("n_refits", "refits", digits=3),
                    Column("refits_at_bound", "at bound", digits=3),
                    Column("contaminated", "contaminated", digits=1),
                    Column("refits_fallback", "fallback", digits=3),
                    Column("refits_capped", "capped", digits=3),
                ),
                rows=tuple(rows),
                row_header="product | policy | rota",
            )
        )
    if "claim_columns" in have:
        out.append(
            TableSpec(
                "claim_columns",
                "Which policy column enters the conclusion: clean = every configured row ok and "
                "no refit at a correlation bound; the rows whose task result is no longer under "
                "m8b/C (a table older than its results); the refits, how many took the guarded "
                "fallback (the base fit's correlation target held) and how many had their "
                "correlation target capped, and the rows that do not record those counts.",
                "claim_columns",
                (
                    Column("clean", "clean", digits=1),
                    Column("rows_expected", "configured", digits=3),
                    Column("rows_ok", "rows ok", digits=3),
                    Column("rows_contaminated", "contaminated", digits=3),
                    Column("rows_at_bound_unrecorded", "bound unrecorded", digits=3),
                    Column("rows_without_result_file", "rows without result file", digits=3),
                    Column("refits", "refits", digits=4),
                    Column("refits_fallback", "fallback", digits=4),
                    Column("refits_capped", "capped", digits=4),
                    Column("fallback_share", "fallback share", digits=3),
                    Column("rows_counts_unrecorded", "counts unrecorded", digits=3),
                ),
                row_header="policy",
            )
        )
    if "claim" in have:
        out.append(
            TableSpec(
                "claim",
                f"The first-order comparison per policy and rota, each product classed at "
                f"{CLASSIFY_NSE:g} stderr against the tolerance band (above: same sign, "
                "|simulated| larger; below: |simulated| smaller or the opposite sign; undecided: "
                "the interval straddles a band edge); the smallest and largest ratio with their "
                "own stderr; how many decided products have |simulated| >= |static| at 2 stderr "
                "(ratio - 2 se >= 1); how many simulated desk P&Ls are gains and losses with "
                f"|z| > {SIGN_NSE:g}, and how many signs are undecided.",
                "claim",
                (
                    Column("pairs", "products", digits=2),
                    Column("above", "above", digits=2),
                    Column("within", "within", digits=2),
                    Column("undecided", "undecided", digits=2),
                    Column("below", "below", digits=2),
                    Column("min_ratio", "min ratio"),
                    Column("max_ratio", "max ratio"),
                    Column("decided_static_not_larger", "decided with sim >= static", digits=2),
                    Column("desk_gains", "desk gains", digits=2),
                    Column("desk_losses", "desk losses", digits=2),
                    Column("sign_undecided", "sign undecided", digits=2),
                ),
                row_header="policy | rota",
            )
        )
    if "ordering" in have:
        cols = results.columns("ordering")
        out.append(
            TableSpec(
                "ordering",
                "The static policy ordering: the greek's desk shadow under each policy and their "
                "signed ratio (delta method). Units: "
                f"{_units_caption(results, 'ordering')}.",
                "ordering",
                tuple(
                    Column(
                        k,
                        k.replace("_", " "),
                        unit="" if ("_over_" in k or k.endswith("_rows")) else PRODUCT_UNIT,
                        digits=2 if k.endswith("_rows") else 4,
                    )
                    for k in cols
                ),
                row_header="product",
            )
        )
    if "ordering_sim" in have:
        out.append(
            TableSpec(
                "ordering_sim",
                "The simulated policy ordering: the signed ratio of the simulated re-marking P&L "
                "of the two policies at each rota (delta method, conservative: the runs share "
                "the world seed).",
                "ordering_sim",
                tuple(
                    Column(k, k.replace("ratio@", "rota ")) for k in results.columns("ordering_sim")
                ),
                row_header="product",
            )
        )
    if "ordering_sides" in have:
        out.append(
            TableSpec(
                "ordering_sides",
                f"Sides of the ordering ratios at {CLASSIFY_NSE:g} stderr (+1 numerator larger, "
                "-1 smaller, -2 opposite signs, 0 undecided) and the product-rota pairs where the "
                "decided static and simulated sides differ (1).",
                "ordering_sides",
                tuple(
                    Column(k, k.replace("@", " "), digits=1)
                    for k in results.columns("ordering_sides")
                ),
                row_header="product",
            )
        )
    return fit_specs(out, results)


def _rows_by(results: Results, table: str, axis: str) -> dict[str, list[str]]:
    """The row labels of ``table`` grouped by the value of ``axis`` (first-appearance order)."""
    if table not in results.tables():
        return {}
    long = results.long(table)
    out: dict[str, list[str]] = {}
    for row, value in zip(long["row"], long[f"axis_{axis}"]):
        group = out.setdefault(str(value), [])
        if row not in group:
            group.append(str(row))
    return out


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


def _grid(n: int) -> tuple[int, int]:
    ncols = min(max(n, 1), 3)
    return (max(n, 1) + ncols - 1) // ncols, ncols


def _flat(axes: Any) -> list[Any]:
    return list(np.atleast_1d(axes).ravel())


def _draw_pnl(results: Results) -> Any:
    long = results.long("pnl")
    long = long[long["axis_policy"] != POLICY_NONE]
    products = list(dict.fromkeys(long["axis_product"]))
    nrows, ncols = _grid(len(products))
    fig, axes = style.new_figure(nrows, ncols, width=3.8 * ncols, height=3.3 * nrows)
    contaminated: dict[tuple[str, str], bool] = {}
    if "agreement" in results.tables():
        ag = results.long("agreement", "contaminated")
        for r in ag.to_dict("records"):
            key = (str(r["axis_product"]), str(r["axis_policy"]))
            contaminated[key] = contaminated.get(key, False) or r["value"] == 1.0
    for ax, product in zip(_flat(axes), products):
        sub = long[long["axis_product"] == product]
        series = 0
        for policy in dict.fromkeys(sub["axis_policy"]):
            tag = " (contaminated)" if contaminated.get((product, str(policy))) else ""
            for col, what in (("recal_pnl", "simulated"), ("static_x_rota", "static x rota")):
                part = sub[(sub["axis_policy"] == policy) & (sub["column"] == col)]
                if part.empty or series >= len(style.PALETTE):
                    continue
                style.mc_errorbar(
                    ax,
                    part["axis_rota"].to_numpy(dtype=float),
                    part["value"].to_numpy(),
                    part["stderr"].to_numpy(),
                    exact=part["exact"].to_numpy(),
                    series=series,
                    label=f"{policy}{tag}: {what}",
                )
                series += 1
        ax.axhline(0.0, color=style.INK["spine"], linewidth=0.8)
        unit = next((u for u in sub["unit"] if u), "")
        ax.set_xticks(sorted(set(sub["axis_rota"].astype(float))))
        ax.set_xlabel("rota")
        ax.set_ylabel(f"desk P&L [{unit}]")
        ax.set_title(product)
        ax.legend(fontsize=6)
    for ax in _flat(axes)[len(products) :]:
        ax.set_visible(False)
    return fig


def _draw_ratio(results: Results, column: str) -> Any:
    long = results.long("agreement", column)
    long = long[long["axis_policy"] != POLICY_NONE]
    policies = list(dict.fromkeys(long["axis_policy"]))
    fig, axes = style.new_figure(1, max(len(policies), 1), width=4.2 * max(len(policies), 1))
    tol = math.nan
    if "setup" in results.tables():
        tol = results.value("setup", "study C", "first_order_tolerance")[0]
    for ax, policy in zip(_flat(axes), policies):
        sub = long[long["axis_policy"] == policy]
        if column == "ratio" and math.isfinite(tol):
            ax.axhspan(1.0 - tol, 1.0 + tol, color=style.INK["grid"], alpha=0.6, zorder=0)
            ax.axhline(1.0, color=style.INK["spine"], linewidth=0.8)
        else:
            ax.axhline(0.0, color=style.INK["spine"], linewidth=0.8)
        for i, product in enumerate(dict.fromkeys(sub["axis_product"])):
            part = sub[sub["axis_product"] == product]
            style.mc_errorbar(
                ax,
                part["axis_rota"].to_numpy(dtype=float),
                part["value"].to_numpy(),
                part["stderr"].to_numpy(),
                series=i,
                label=str(product),
            )
        if not sub.empty:
            ax.set_xticks(sorted(set(sub["axis_rota"].astype(float))))
        ax.set_xlabel("rota")
        ax.set_ylabel("simulated / static" if column == "ratio" else "P&L(rota)/(rota P&L(+1)) - 1")
        ax.set_title(str(policy))
        ax.legend(fontsize=7)
    return fig


def _draw_m7(results: Results) -> Any:
    long = results.long("m7_greek")
    names = [
        n
        for n in ("usual", "recalibrated", "fee_shadow", "desk_pnl_shadow")
        if n in set(long["column"])
    ]
    fig, ax = style.new_figure()
    policies = list(dict.fromkeys(long["row"]))
    width = 0.8 / max(len(policies), 1)
    for i, policy in enumerate(policies):
        sub = long[long["row"] == policy].set_index("column")
        xs = [names.index(n) + (i - (len(policies) - 1) / 2) * width for n in names]
        style.mc_errorbar(
            ax,
            xs,
            [float(sub.loc[n, "value"]) for n in names],
            [float(sub.loc[n, "stderr"]) for n in names],
            series=i,
            label=str(policy),
            line=False,
        )
    ax.axhline(0.0, color=style.INK["spine"], linewidth=0.8)
    ax.set_xticks(range(len(names)), [n.replace("_", " ") for n in names])
    ax.set_ylabel(f"per +1 rota [{M7_UNIT}]")
    ax.set_title("M7 greek, 3y autocall (error bars: 1 stderr)")
    ax.legend()
    return fig


def figures(results: Results) -> list[FigureSpec]:
    have = set(results.tables())
    out: list[FigureSpec] = []
    if "pnl" in have:
        out.append(
            FigureSpec(
                "pnl_vs_static",
                "Simulated recalibration P&L (desk) against the static greek x rota, per "
                "product and policy (error bars: 1 stderr).",
                _draw_pnl,
            )
        )
    if "agreement" in have:
        out.append(
            FigureSpec(
                "ratio",
                "First-order agreement simulated / static per rota (band: the tolerance; error "
                "bars: 1 stderr, delta method).",
                lambda r: _draw_ratio(r, "ratio"),
            )
        )
        if "nonlinearity" in results.columns("agreement"):
            out.append(
                FigureSpec(
                    "nonlinearity",
                    "Nonlinearity of the simulated P&L at +2 / +3 rota (error bars: 1 stderr, "
                    "delta method, conservative).",
                    lambda r: _draw_ratio(r, "nonlinearity"),
                )
            )
    if "m7_greek" in have:
        out.append(
            FigureSpec(
                "m7_greek",
                "The M7 shadow-rotation greek of the 3y autocall per policy (P1 price moves and "
                "the desk shadow; error bars: 1 stderr).",
                _draw_m7,
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# narrative
# --------------------------------------------------------------------------------------------


def _declared(results: Results) -> list[str]:
    """``table:<name>`` / ``figure:<name>`` of every declared table and figure, in order."""
    return [f"table:{t.name}" for t in tables(results)] + [
        f"figure:{f.name}" for f in figures(results)
    ]


def _place(declared: Sequence[str], *names: str) -> list[str]:
    """Placeholder lines for the declared ``names``; a name ending in ``*`` places every
    declared name with that prefix (the parts of a split table)."""
    out: list[str] = []
    for n in names:
        hits = (
            [d for d in declared if d.startswith(n[:-1])]
            if n.endswith("*")
            else [d for d in declared if d == n or re.fullmatch(re.escape(n) + r"_\d+", d)]
        )
        for d in hits:
            out += ["{{" + d + "}}", ""]
    return out


def _ex(results: Results, table: str, row: str, column: str) -> float:
    return results.value(table, row, column)[0]


def _state_lines(results: Results) -> list[str]:
    rec = results.record("table_c_state", "table C", "mtime_epoch_s")
    n = int(_ex(results, "table_c_state", "table C", "rows"))
    exp = int(_ex(results, "table_c_state", "table C", "rows_expected"))
    miss = int(_ex(results, "table_c_state", "table C", "rows_missing"))
    notok = int(_ex(results, "table_c_state", "table C", "rows_not_ok"))
    newer = int(_ex(results, "table_c_state", "table C", "result_files_newer"))
    nofile = int(_ex(results, "table_c_state", "table C", "rows_without_result_file"))
    fallback = int(_ex(results, "table_c_state", "table C", "fallback_recorded"))
    lines = [
        "## Study C's table as read",
        "",
        f"`m8b_table_C.csv` was **{rec['note']}**; it holds {n} rows for {exp} configured "
        f"(product x rota x recalibration): "
        + (
            "**no configured row is missing**"
            if miss == 0
            else f"**{miss} configured row(s) are missing** (listed below)"
        )
        + (f", {notok} row(s) not ok" if notok else "")
        + ".",
    ]
    if newer:
        lines.append(
            f"{newer} study-C task result(s) under `m8b/C` are newer than the table: a re-run is "
            "writing results the table does not show yet (the table is rebuilt when the run "
            "ends)."
        )
    if nofile:
        lines.append(
            f"{nofile} table row(s) have no task result under `m8b/C` any more (their files were "
            "moved aside for the re-run): those rows are the **pre-re-run** numbers."
        )
    lines.append(
        f"{fallback} recalibration row(s) record `refits_fallback` / `refits_capped` (the rebuilt "
        "refit of 2026-09-16); -1 marks a run from before the record."
    )
    return [*lines, ""]


def _names(results: Results, row: str, column: str) -> str:
    """The product list a ``claim`` count carries in its note."""
    return str(results.record("claim", row, column)["note"])


def _count(results: Results, row: str, column: str) -> int:
    return int(_ex(results, "claim", row, column))


def _class_sentence(results: Results, row: str) -> str:
    """``a above (…), w within (…), u undecided (…), b below (…)`` for one ``claim`` row."""
    parts = []
    for col in ("above", "within", "undecided", "below"):
        n = _count(results, row, col)
        if n:
            parts.append(f"{n} {col} ({_names(results, row, col)})")
    return ", ".join(parts) if parts else "no product"


def _sign_sentence(results: Results, row: str) -> str:
    """Which simulated P&Ls are desk losses / gains (``|z| > SIGN_NSE``) and which signs are
    undecided."""
    parts = []
    for col, what in (("desk_losses", "a desk loss"), ("desk_gains", "a desk gain")):
        names = _names(results, row, col)
        if names:
            parts.append(f"{what} for {names}")
    text = (
        f"The simulated re-marking P&L is {' and '.join(parts)} (|z| > {SIGN_NSE:g})"
        if parts
        else f"No simulated re-marking P&L has a sign significant at |z| > {SIGN_NSE:g}"
    )
    unsigned = _names(results, row, "sign_undecided")
    if unsigned:
        text += f"; its sign is undecided (|z| <= {SIGN_NSE:g}) for {unsigned}"
    return text + "."


def _refit_lines(results: Results) -> list[str]:
    """Per policy column: the refits, the guarded fallbacks and the capped targets, and the
    reading those numbers support (module docstring)."""
    policies = results.rows("claim_columns")
    lines: list[str] = []
    for p in policies:
        refits = int(_ex(results, "claim_columns", p, "refits"))
        rec = int(_ex(results, "claim_columns", p, "refits_recorded"))
        fb = int(_ex(results, "claim_columns", p, "refits_fallback"))
        cap = int(_ex(results, "claim_columns", p, "refits_capped"))
        rows = int(_ex(results, "claim_columns", p, "rows_ok"))
        unrec = int(_ex(results, "claim_columns", p, "rows_counts_unrecorded"))
        text = f"- **{p}**: {refits} refits over its {rows} rows"
        if rec:
            text += (
                f"; {fb} of the {rec} recorded refits ({fb / rec:.0%}) took the guarded fallback "
                f"and {cap} ({cap / rec:.0%}) had their correlation target capped at "
                f"|Corr_BE| <= {REFIT_CORRELATION_CAP:g}"
            )
        if unrec:
            text += (
                f"; {unrec} row(s) do not record the fallback / cap counts (a run from before the "
                "rebuilt refit)"
            )
        lines.append(text + ".")
    if POLICY_SABR in policies and POLICY_STICKY in policies:
        rec = int(_ex(results, "claim_columns", POLICY_SABR, "refits_recorded"))
        fb = int(_ex(results, "claim_columns", POLICY_SABR, "refits_fallback"))
        if rec:
            lines.append(
                f"  Reading: on a fallback refit the refit holds the base fit's correlation "
                f"target — the target {POLICY_STICKY} holds on every refit — so on {fb} of "
                f"{POLICY_SABR}'s {rec} recorded refits ({fb / rec:.0%}) the two policies share "
                f"their correlation target and {POLICY_SABR} differs from {POLICY_STICKY} only "
                "through its break-even vol targets: on that share of refits a policy comparison "
                "(the ordering below) compares vol targets, not correlation dynamics."
            )
    return [*lines, ""]


def _claim_lines(results: Results) -> list[str]:
    if "claim_columns" not in results.tables():
        return ["No study-C row to draw a conclusion from.", ""]
    tol = _ex(results, "setup", "study C", "first_order_tolerance")
    policies = results.rows("claim_columns")
    clean = [p for p in policies if _ex(results, "claim_columns", p, "clean") == 1.0]
    dirty = [p for p in policies if p not in clean]
    modified = str(results.record("table_c_state", "table C", "mtime_epoch_s")["note"])
    newer = int(_ex(results, "table_c_state", "table C", "result_files_newer"))
    lines = [
        "## The restricted claim",
        "",
        f"Each product's simulated / static ratio is classed at {CLASSIFY_NSE:g} stderr "
        f"against 1 +/- {tol:g}: **above** = same sign and |simulated| larger than "
        f"(1 + {tol:g}) x |static|; **below** = |simulated| smaller than (1 - {tol:g}) x |static|"
        " or the opposite sign; **within** = the whole interval inside the band; "
        "**undecided** = the interval straddles a band edge. The desk P&L is a loss for some "
        "products and a gain for others, so the comparison is on signed values and magnitudes, "
        "not on a cost.",
        "",
    ]
    rotas = sorted(
        {float(v) for v in results.long("claim")["axis_rota"]}
        if "claim" in results.tables()
        else {1.0}
    )
    for p in dirty:
        cont = int(_ex(results, "claim_columns", p, "rows_contaminated"))
        ok = int(_ex(results, "claim_columns", p, "rows_ok"))
        exp = int(_ex(results, "claim_columns", p, "rows_expected"))
        unrec = int(_ex(results, "claim_columns", p, "rows_at_bound_unrecorded"))
        why = []
        if cont:
            why.append(f"{cont} of its {ok} rows are contaminated (refits at a correlation bound)")
        if ok < exp:
            why.append(f"{exp - ok} of its {exp} configured rows are absent or not ok")
        if unrec:
            why.append(f"{unrec} row(s) do not record the bound count")
        lines.append(
            f"- The **{p}** column is reported but **excluded** from the conclusion: "
            + "; ".join(why)
            + " (owner's decision of 2026-09-16: the claim is drawn on clean columns only)."
        )
    if lines[-1] != "":
        lines.append("")
    lines += ["Refits per policy column:", "", *_refit_lines(results)]
    if not clean:
        lines += ["**No policy column is clean: no conclusion is drawn.**", ""]
        return lines
    for p in clean:
        one = f"{p} | +1"
        if one not in results.rows("claim"):
            lines.append(f"- The clean **{p}** column has no +1 row: no claim.")
            continue
        n = _count(results, one, "pairs")
        above, below = _count(results, one, "above"), _count(results, one, "below")
        within, undecided = _count(results, one, "within"), _count(results, one, "undecided")
        decided = _count(results, one, "decided")
        within_not_larger = _count(results, one, "decided_static_not_larger") - above
        lines.append(
            f"- **{p}, +1 rota** ({n} products): {_class_sentence(results, one)}. "
            + _sign_sentence(results, one)
        )
        if decided and above and below == 0 and within_not_larger == within:
            lines.append(
                "  At +1, on every product the comparison decides, |simulated| >= |static| at "
                f"{CLASSIFY_NSE:g} stderr (same sign): the static greek understates the simulated "
                f"re-marking P&L beyond the {tol:.0%} tolerance on {above} product(s)"
                + (f" and within it on {within}" if within else "")
                + (f"; {undecided} undecided" if undecided else "")
                + ". This is a statement about +1 rota and about magnitudes only."
            )
        elif decided and above == 0 and below == 0:
            lines.append(
                f"  At +1 the static greek matches the simulated P&L within {tol:.0%} on "
                f"{within} product(s) ({within_not_larger} of them with |simulated| >= |static| "
                f"at {CLASSIFY_NSE:g} stderr)" + (f"; {undecided} undecided." if undecided else ".")
            )
        elif above and below:
            lines.append(
                "  At +1 the static greek neither estimates nor bounds the simulated P&L: the "
                "decided misses go both ways."
            )
        elif decided:
            lines.append(
                "  At +1 no single direction holds for the decided products: "
                f"{above} above the band, {within} within it ({within_not_larger} with "
                f"|simulated| >= |static| at {CLASSIFY_NSE:g} stderr, "
                f"{within - within_not_larger} where the static greek may be the larger), "
                f"{below} below it; {undecided} undecided."
            )
        else:
            lines.append("  At +1 no comparison is decided at 2 stderr: nothing more is stated.")
        others = [
            r for r in rotas if r != 1.0 and f"{p} | {_rota_label(r)}" in results.rows("claim")
        ]
        if others:
            parts = []
            for r in others:
                row = f"{p} | {_rota_label(r)}"
                lo, lo_se = results.value("claim", row, "min_ratio")
                hi, hi_se = results.value("claim", row, "max_ratio")
                unsigned = _names(results, row, "sign_undecided")
                parts.append(
                    f"at {_rota_label(r)}: {_class_sentence(results, row)} (ratios "
                    f"{latex.format_value_text(lo, lo_se)} to "
                    f"{latex.format_value_text(hi, hi_se)}"
                    + (f"; simulated sign undecided for {unsigned}" if unsigned else "")
                    + ")"
                )
            more_below = sum(_count(results, f"{p} | {_rota_label(r)}", "below") for r in others)
            lines.append(
                "  The static greek scaled linearly (static x rota) does not describe the larger "
                "shocks the same way — "
                + "; ".join(parts)
                + "."
                + (
                    f" Beyond +1 the simulated P&L is smaller in magnitude than static x rota "
                    f"(or of the opposite sign) on {more_below} product-rota pair(s): the +1 "
                    "reading does not extend to the larger shocks."
                    if more_below and below == 0
                    else ""
                )
            )
        missing_files = int(_ex(results, "claim_columns", p, "rows_without_result_file"))
        total = int(_ex(results, "claim_columns", p, "rows_expected"))
        if missing_files or newer:
            lines.append(
                f"  **This is the claim of the table as read ({modified}):** {missing_files} of "
                f"the column's {total} rows have no task result under `m8b/C` any more (moved "
                f"aside for the study-C re-run) and {newer} newer task result(s) are not in the "
                "table yet — the pre-re-run claim, to be redrawn when the re-run's table lands."
            )
        else:
            lines.append(
                f"  Every row of the column has its task result under `m8b/C` and no newer "
                f"result exists (table {modified})."
            )
    return [*lines, ""]


def _ordering_lines(results: Results) -> list[str]:
    if "ordering" not in results.tables():
        return []
    cols = results.columns("ordering")
    static_ratio = next((c for c in cols if c.startswith("static_") and "_over_" in c), None)
    if static_ratio is None:
        return []
    num, _, den = static_ratio[len("static_") :].partition("_over_")
    have_sim = "ordering_sim" in results.tables()
    sides = "ordering_sides" in results.tables()
    lines = [
        "## Policy ordering",
        "",
        f"The signed ratio {num} / {den} of the static desk shadow and of the simulated "
        f"re-marking P&L at each rota; each side decided at {CLASSIFY_NSE:g} stderr "
        f"({num} larger / smaller in magnitude with the same sign, opposite signs, or "
        "undecided).",
        "",
    ]
    disagreements: list[str] = []
    both_decided = 0
    pairs = 0
    sim = results.pivot("ordering_sim") if have_sim else pd.DataFrame()
    side = results.pivot("ordering_sides") if sides else pd.DataFrame()
    side_notes: dict[tuple[str, str], str] = {}
    if sides:
        for rec in _records(results.long("ordering_sides")):
            side_notes[(str(rec["row"]), str(rec["column"]))] = str(rec["note"])
    for product in results.rows("ordering"):
        sv, sse = results.value("ordering", product, static_ratio)
        s_side = side_notes.get((product, "static"), "")
        text = f"- {product}: static {latex.format_value_text(sv, sse)}" + (
            f" ({s_side})" if s_side else ""
        )
        parts = []
        if product in sim.index:
            for col in results.columns("ordering_sim"):
                v, se = _num(sim.at[product, col]), _num(sim.at[product, f"{col}_stderr"])
                if math.isnan(v):
                    continue
                label = col.partition("@")[2]
                word = side_notes.get((product, f"sim@{label}"), "")
                if word:
                    pairs += 1
                    static_code = _num(side.at[product, "static"])
                    if static_code != 0.0 and _num(side.at[product, f"sim@{label}"]) != 0.0:
                        both_decided += 1
                    if _num(side.at[product, f"disagree@{label}"]) == 1.0:
                        disagreements.append(f"{product} at {label}")
                parts.append(
                    f"{label} {latex.format_value_text(v, se)}" + (f" ({word})" if word else "")
                )
        if parts:
            text += "; simulated " + ", ".join(parts)
        cont = f"{den}_contaminated_rows"
        if cont in cols and results.value("ordering", product, cont)[0] > 0:
            n = int(results.value("ordering", product, cont)[0])
            text += f" ({n} contaminated {den} row(s))"
        lines.append(text + ".")
    if sides and pairs:
        lines += [
            "",
            f"The static and the simulated sides are both decided on {both_decided} of "
            f"{pairs} product-rota pairs; they differ on {len(disagreements)}"
            + (f" ({', '.join(disagreements)})" if disagreements else "")
            + "."
            + (
                " The simulated re-marking P&L does not order the policies the way the static "
                "greek does there: the reversal §8.2 records as open."
                if disagreements
                else ""
            ),
        ]
    return [*lines, ""]


def _nonlinearity_lines(results: Results) -> list[str]:
    if "agreement" not in results.tables() or "nonlinearity" not in results.columns("agreement"):
        return []
    long = results.long("agreement", "nonlinearity")
    refits = results.long("agreement", "n_refits")
    lines = ["## Nonlinearity at +2 / +3", ""]
    for policy in dict.fromkeys(long["axis_policy"]):
        if policy == POLICY_NONE:
            continue
        sub = long[(long["axis_policy"] == policy) & np.isfinite(long["value"])]
        if sub.empty:
            continue
        rsub = refits[refits["axis_policy"] == policy]
        by_rota = rsub.groupby("axis_rota")["value"].mean()
        counts = ", ".join(f"{_rota_label(float(k))}: {v:.1f}" for k, v in by_rota.items())
        lines.append(
            f"- {policy}: nonlinearity between {sub['value'].min():+.2f} and "
            f"{sub['value'].max():+.2f} over {len(sub)} rows (each with its stderr in the "
            f"agreement tables); mean refits per run by rota {counts}. The rule refits more "
            "often the larger the shock, so the re-marking P&L need not scale with the rota."
        )
    return [*lines, ""]


def _m7_lines(results: Results) -> list[str]:
    lines = ["## The static greek", ""]
    if "static_greek" in results.tables():
        notes = {
            str(n)
            for n in results.long("static_greek", "base_fit_is_marking_fit")["note"]
            if str(n)
        }
        for n in sorted(notes):
            lines.append(f"Declared convention (static records): *{n}*.")
        base = results.long("static_greek", "base_fit_is_marking_fit")
        bad = base[base["value"] != 1.0]
        lines.append(
            "Every static greek's base fit equals the marking fit."
            if bad.empty
            else f"**{len(bad)} static record(s) do not confirm their base fit equals the "
            "marking fit.**"
        )
        lines.append("")
    if "m7_vs_static" in results.tables():
        for policy in results.rows("m7_vs_static"):
            cols = set(results.columns("m7_vs_static"))
            if "z_desk_pnl_shadow" not in cols:
                continue
            mv, mse = results.value("m7_vs_static", policy, "m7_desk_pnl_shadow")
            sv, sse = results.value("m7_vs_static", policy, "static_desk_pnl_shadow")
            rec = results.record("m7_vs_static", policy, "z_desk_pnl_shadow")
            verdict = (
                "the same number (same seed and states: reproduced bit for bit up to "
                "floating point)"
                if rec["note"] == IDENTICAL_NOTE
                else f"z {float(rec['value']):+.2f}"
            )
            lines.append(
                f"- {policy}: M7 desk shadow {latex.format_value_text(mv, mse)} against study "
                f"C's static {latex.format_value_text(sv, sse)} {M7_UNIT} per rota: {verdict}."
            )
        lines.append("")
    return lines


def narrative(results: Results) -> str:
    declared = _declared(results)
    lines = [
        "Artefacts only: the M7 greek (`m7/p1_marking_shadow_rotation.csv`), study C's static "
        "greeks (`m8b/C/static_*.json`) and study C's table (`m8b/m8b_table_C.csv`); nothing is "
        "simulated or calibrated. Desk convention: the desk is SHORT the note; its P&L per +1 "
        "rota is -(d fee). Every Monte Carlo number carries its stderr; ratios, z-scores and "
        "nonlinearities are recomputed from the table's values with the M8b definitions.",
        "",
        *_state_lines(results),
        *_place(declared, "table:table_c_state", "table:table_c_missing"),
        *_m7_lines(results),
        *_place(
            declared,
            "table:m7_greek",
            "figure:m7_greek",
            "table:static_greek",
            "table:m7_vs_static",
        ),
        *_claim_lines(results),
        *_place(
            declared, "table:claim_columns", "table:claim", "table:pnl_*", "figure:pnl_vs_static"
        ),
        "## First-order agreement",
        "",
        "The none rows compare the total hedged desk P&L with desk_pnl_usual x rota: a "
        "reference, not part of the claim.",
        "",
        *_place(declared, "table:agreement_*", "figure:ratio"),
        *_nonlinearity_lines(results),
        *_place(declared, "figure:nonlinearity"),
        *_ordering_lines(results),
        *_place(declared, "table:ordering", "table:ordering_sim", "table:ordering_sides"),
    ]
    return "\n".join(lines)
