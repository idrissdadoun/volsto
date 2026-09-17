"""S5 — marking: what the SSR mark costs or earns, per product (SPEC §10.2, owner's M10 Part 2
"S5. Marking study: prices and Greeks across (ssr_target, skew_eps) on each SPX snapshot — the
binding map, |L-1|, realised SSR, forward/spot skew ratio, and the product price surface over the
two dials").

**What is computed here and what is read.**

* **The binding map is computed inline.**  For every configured surface and every
  ``(ssr_target, skew_eps)`` of ``params.binding_map`` the study runs the P1 marking fit through
  the one helper the precompute resolves a marking point with
  (:func:`volsto.viewers.grid.marking_fit`, called by :func:`~volsto.viewers.grid.
  resolve_marking`: ``fit_2f_marking`` at k2 0.2, ν cap 3.5, two-point skew constraint)
  **without stage 3**: a deterministic parameter fit of a few
  seconds (first-order break-even closed forms, a 2-D QP per k1 and a ten-start SLSQP), **not a
  leverage calibration** — nothing is simulated and no leverage is built.  The runner's guard
  (:func:`volsto.studies.runner.calibration_forbidden`) stays active around it and would refuse
  any leverage calibration.  The fit's status (interior / binding / infeasible), parameters, the
  naked-skew gaps and binding edges at the two constraint pillars, the ν box / ν cap flags, the
  largest first-order SpotVolCovar miss and the ρ12-collapse flag are exact numbers.  For the
  axis values that are grid points the stored fit is compared with the inline one (largest
  parameter difference, status match) — the store's leverage key depends on those parameters.
* **Everything that needs a leverage is read from the results store**, at the marking points
  ``marking:<surface>:ssr<s>:eps<e>`` of the configured grid (``configs/grids/default.yaml``: the
  4 × 3 marks on each of the three SPX snapshots):

  - the mean ``|L − 1|`` (the M7 convention, ``|k| ≤ 2`` ATM sd per slice) — an exact
    functional of the cached leverage whose particle-seed noise is not estimated upstream: shown
    as exact with that note, and drawn without error bars;
  - the realised SSR of the calibrated LSV at ``params.ssr_pillars`` (``ssr_lsv`` ± stderr)
    beside the fit's first-order P1 SSR;
  - the forward 90/110 skew of the stored forward smiles (``σ(0.9) − σ(1.1)``, stderr the
    **sum** of the two strikes' errors — a bound valid for any correlation between them, which
    quadrature is not: the strikes' estimates are correlated under the LSVs; the S4
    convention) against the spot 90/110 skew of the same tenor on the snapshot (exact,
    :func:`volsto.calibration.fit_2f.spot_skew_90_110`) and their ratio (stderr
    ``fs_se / |spot|``);
  - the product prices (``params.products``, store keys);
  - when ``params.greeks.names`` is not empty, the stored risk rows of the two products the
    precompute's risk step prices (:data:`RISK_PRODUCTS`, notional 1): the stored values are
    fractions of notional per bump unit, shown ×100 as ``% notional <store unit>``, as S4 does.

  A missing marking point prints the ``volsto-precompute`` line (exit 2 before anything is
  computed); a stored point without its light-tier risk rows prints the ``--risk light`` refresh
  line; a Greek name no stored point carries is a config error.
* **The cost of a mark** is the price at a mark minus the price at ``params.reference_mark`` on
  the same snapshot (the M8b marking, ssr 1 / eps 0.10), stderr in quadrature, labelled
  :data:`~volsto.studies.catalogue._common.QUADRATURE_NOTE`: not the exact error — the marks
  share the store's pricing seed and their correlation is not measured, so the quadrature error
  is a bound only if that correlation is non-negative.  Its z-score carries stderr 1 (the
  sampling sd of a z-score).
  Positive = the mark prices the product higher than the reference mark (a desk short the
  product marks a larger liability; at inception it would charge a higher fee).
* **Units.**  Store vols are converted to vol points.  The M6 cells (``autocall 3y:price``,
  ``phoenix 3y:leg:*``) are stored as **fractions** of notional under the label ``% notional``
  (the store's mislabel, left untouched there) and are multiplied by 100 here, with a note.
* **The recorded M7 fits** (``m7/p1_marking_fits.csv``, ``params.recorded_m7_fits``) are shown
  as recorded by ``scripts/m7_p1_marking.py`` (SPX and the placeholder reference surface at
  ``(1.0, 0.10)`` and ``(1.5, 0.05)``, 2·10⁵ particles), labelled as such — not recomputed.

The placeholder SSVI is unsuitable for marking work (M7 report decision vii: ν past the cap,
ρ = −1, ρ12 = +1); the narrative says so whenever a configured surface is the placeholder — the
CI-fast configuration's toy marking grid is one, and fast mode is plumbing, not numbers.

Results tables: ``setup``, ``binding_map``, ``marks``, ``forward_skew``, ``prices``,
``mark_cost``, ``mark_cost_z``, ``greeks_<product>``, ``m7_fits``; the rendered tables are one per
surface (``binding_map_<surface>`` …, split further into parts above
:data:`volsto.studies.latex.LONGTABLE_MIN_ROWS` rows, where the renderer would switch to an
unscaled ``longtable``).  Figures: ``binding_map``, ``realised_ssr``, ``fwd_spot_ratio``,
``leverage_deviation``, ``mark_cost_<surface>``.  Checked by ``tests/test_catalogue_s5_s7.py``.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import re
import time
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from volsto.calibration.cache import build_market
from volsto.calibration.fit_2f import FitResult, spot_skew_90_110
from volsto.config import ConfigError
from volsto.studies import latex, style
from volsto.studies.catalogue._common import QUADRATURE_NOTE
from volsto.studies.latex import LONGTABLE_MIN_ROWS
from volsto.studies.m6 import AUTOCALL_NAME
from volsto.studies.results import (
    DIMENSIONLESS,
    Column,
    FigureSpec,
    Results,
    ResultsBuilder,
    TableSpec,
)
from volsto.studies.runner import MissingRequirements, Requirement, StudyContext
from volsto.viewers import api
from volsto.viewers.grid import (
    GridPoint,
    GridSpec,
    marking_fit,
    marking_summary,
    reference_spec,
    surface_spec,
)
from volsto.viewers.precompute import CLIQUET_1Y_NAME
from volsto.viewers.store import StoreReader

TITLE = "S5 - Marking: what the SSR mark costs or earns, per product"
QUESTION = (
    "What does the choice of SSR mark (ssr_target, skew_eps) cost or earn per product on each "
    "snapshot, and what does each mark imply for the leverage, the realised SSR and the forward "
    "skew?"
)
REQUIRED_PARAMS = (
    "surfaces",
    "binding_map",
    "reference_mark",
    "products",
    "ssr_pillars",
    "forward_windows",
    "greeks",
    "recorded_m7_fits",
)
OPTIONAL_PARAMS: tuple[str, ...] = ()

M7_FITS = "m7/p1_marking_fits.csv"
M7_COMMAND = ".venv/bin/python scripts/m7_p1_marking.py --surfaces spx,reference"
#: Status codes of a marking fit (exact numbers; the text is in the row's note).
STATUS_CODES: dict[str, float] = {"interior": 0.0, "binding": 1.0, "infeasible": 2.0}
STATUS_NAMES: dict[float, str] = {v: k for k, v in STATUS_CODES.items()}
#: Binding edge of a constraint pillar: +1 the steep edge (naked = (1 + eps) x market), -1 the
#: flat edge, 0 inside the slab.
EDGE_STEEP, EDGE_FLAT, EDGE_NONE = 1.0, -1.0, 0.0
#: Strikes of the 90/110 skew (fractions of the forward-start spot, the store's convention).
SKEW_STRIKES: tuple[float, float] = (0.9, 1.1)
#: Vol -> vol points.
VOL_PTS = 100.0
#: The deck's forward / spot 90/110 skew ratio band (the owner's sticky-strike note of M7; §15
#: Part 3 measured 1.0-1.17 against it) — a reference band on the figure, not a target.
DECK_RATIO_BAND: tuple[float, float] = (1.3, 1.5)
#: Two floats name the same axis value / maturity when closer than this.
TOL = 1e-9
#: Surface kinds (the ``setup`` table's exact code).
KIND_CODES: dict[str, float] = {"placeholder": 0.0, "snapshot": 1.0}
M6_UNIT_NOTE = (
    "M6 cell stored as a fraction of notional under the label '% notional' (store mislabel); "
    "x100 here"
)
L_NOTE = (
    "mean |L-1| (M7 convention): exact functional of the cached leverage; its particle-seed noise "
    "is not estimated upstream"
)
#: The products the precompute's risk step prices (``volsto.viewers.precompute.risk_products``:
#: the M6 3y autocall and the study cliquet, both at notional 1) — their stored risk values are
#: **fractions of notional** per unit of the bump, shown here ×100 in % of notional (as S4 does).
RISK_PRODUCTS: tuple[str, ...] = (AUTOCALL_NAME, CLIQUET_1Y_NAME)
RISK_PERCENT = 100.0
RISK_NOTE = "the store's fraction of notional (notional 1) x 100"
MISSING_NOTE = "not in the store for this point (or no finite stderr)"
#: How the forward 90/110 skew's stderr is formed from the two strikes' errors (the store keeps no
#: per-path samples): their **sum**, an upper bound whatever their correlation — the two
#: strikes' estimates are correlated (measured −0.05 to −0.13 under the LSVs by the catalogue
#: verification, where quadrature understates the error by up to ~6%); the S4 convention.
FWD_SKEW_SE_NOTE = (
    "sigma(0.9) - sigma(1.1) of the forward smile; stderr: the sum of the two strikes' errors, an "
    "upper bound whatever their correlation (the store keeps no per-path samples)"
)
BEYOND_NOTE = "1 = the window ends beyond the calibration horizon (last leverage slice held)"
Z_NOTE = "z-score of the difference (stderrs in quadrature)"
#: A z-score is a Monte Carlo quantity whose sampling sd is 1 (the catalogue's convention).
Z_SD = 1.0
Z_SD_NOTE = "stderr 1: the sampling sd of a z-score"
PLACEHOLDER_WARNING = (
    "The placeholder SSVI is unsuitable for marking work (M7 report decision vii: nu past the "
    "cap, rho = -1, rho12 = +1): numbers on it are plumbing, not a marking result."
)


# --------------------------------------------------------------------------------------------
# params
# --------------------------------------------------------------------------------------------


def _floats(v: Any, name: str, *, positive: bool = False) -> list[float]:
    if not isinstance(v, list) or not v:
        raise ConfigError(f"{name} must be a non-empty list of numbers: {v!r}")
    out = []
    for x in v:
        if isinstance(x, bool) or not isinstance(x, int | float):
            raise ConfigError(f"{name}: {x!r} is not a number")
        if (positive and x <= 0) or x < 0:
            raise ConfigError(f"{name}: {x!r} out of range")
        out.append(float(x))
    return out


#: What each exact row is (:data:`volsto.studies.catalogue._common.EXACT_KIND_NAMES`, full-match
#: regexes on table and column).  The marking fit is deterministic given the surface and the
#: config, so its parameters and the numbers read off them are fitted parameters or closed
#: forms; ``mean |L-1|`` is a functional of the cached leverage (a calibration output, whose
#: particle-seed noise is not estimated upstream: :data:`L_NOTE`).  There are no ranks.
EXACT_KINDS: tuple[tuple[str, ...], ...] = (
    ("setup", "fast_mode|surface_kind", "flag"),
    ("setup", "marks", "count"),
    ("setup", "n_particles|horizon|pricing_n_paths|pricing_seed", "input"),
    ("binding_map|marks|m7_fits", "status_code", "flag"),
    ("binding_map|m7_fits", "nu|theta|k1|k2|rho12|rho_SX1|rho_SX2", "fitted parameter"),
    # the fit's constraint gaps, first-order SVC miss, skew gap, and inline-minus-stored fit
    ("binding_map", "gap_T_[sl]|max_svc_miss|mean_skew_gap|store_param_diff", "fitted parameter"),
    (
        "binding_map",
        "edge_T_[sl]|nu_box|nu_at_cap|rho12_collapse|grid_point|store_status_match",
        "flag",
    ),
    ("marks|m7_fits", "mean_abs_L_minus_1", "fitted parameter"),
    ("marks", r"ssr_first_order@.+", "closed form"),  # the fit's first-order P1 SSR
    ("forward_skew", r"spot_skew@.+", "closed form"),  # the snapshot SSVI's 90/110 skew
    ("forward_skew", r"beyond_horizon@.+", "flag"),
)


def validate_params(params: Mapping[str, Any]) -> None:
    surfaces = params["surfaces"]
    if (
        not isinstance(surfaces, list)
        or not surfaces
        or not all(isinstance(s, str) for s in surfaces)
    ):
        raise ConfigError(f"surfaces must be a non-empty list of grid surface names: {surfaces!r}")
    bm = params["binding_map"]
    if not isinstance(bm, Mapping) or set(bm) != {"ssr_target", "skew_eps"}:
        raise ConfigError(f"binding_map must be {{ssr_target: [...], skew_eps: [...]}}: {bm!r}")
    _floats(bm["ssr_target"], "binding_map.ssr_target", positive=True)
    _floats(bm["skew_eps"], "binding_map.skew_eps")
    ref = params["reference_mark"]
    if not isinstance(ref, Mapping) or set(ref) != {"ssr_target", "skew_eps"}:
        raise ConfigError(f"reference_mark must be {{ssr_target: x, skew_eps: y}}: {ref!r}")
    products = params["products"]
    if (
        not isinstance(products, list)
        or not products
        or not all(isinstance(k, str) for k in products)
    ):
        raise ConfigError(f"products must be a non-empty list of store product keys: {products!r}")
    _floats(params["ssr_pillars"], "ssr_pillars", positive=True)
    windows = params["forward_windows"]
    if not isinstance(windows, list) or not windows:
        raise ConfigError(f"forward_windows must be a non-empty list of [t1, t2]: {windows!r}")
    for w in windows:
        if not isinstance(w, list) or len(w) != 2 or not float(w[1]) > float(w[0]) >= 0.0:
            raise ConfigError(f"forward window {w!r} must be [t1, t2] with t2 > t1 >= 0")
    greeks = params["greeks"]
    if not isinstance(greeks, Mapping) or set(greeks) != {"products", "names"}:
        raise ConfigError(f"greeks must be {{products: [...], names: [...]}}: {greeks!r}")
    if not isinstance(greeks["names"], list) or not isinstance(greeks["products"], list):
        raise ConfigError("greeks.products and greeks.names must be lists")
    if greeks["names"] and not greeks["products"]:
        raise ConfigError("greeks.names needs greeks.products")
    unknown = [x for x in greeks["products"] if x not in RISK_PRODUCTS]
    if unknown:
        raise ConfigError(
            f"greeks.products {unknown}: the store's risk rows exist for {list(RISK_PRODUCTS)} only"
        )
    if not isinstance(params["recorded_m7_fits"], bool):
        raise ConfigError("recorded_m7_fits must be true or false")


# --------------------------------------------------------------------------------------------
# grid
# --------------------------------------------------------------------------------------------


def _grid(ctx: StudyContext) -> GridSpec:
    grid = ctx.grid()
    if grid is None:
        raise ConfigError("S5 reads the marking points of a grid: the config's grid is null")
    return grid


def marking_points(ctx: StudyContext) -> list[GridPoint]:
    """The marking points of the configured surfaces (config checked against the grid)."""
    grid = _grid(ctx)
    p = ctx.params
    if grid.marking is None:
        raise ConfigError(f"grid {grid.name!r} has no marking axes")
    names = [s.name for s in grid.surfaces]
    for s in p["surfaces"]:
        if s not in names:
            raise ConfigError(f"surface {s!r} is not in grid {grid.name!r} ({names})")
        if not grid.surface(s).marking:
            raise ConfigError(f"surface {s!r} of grid {grid.name!r} is not flagged marking")
    ref = p["reference_mark"]
    if not any(abs(x - float(ref["ssr_target"])) < TOL for x in grid.marking.ssr_target) or not any(
        abs(x - float(ref["skew_eps"])) < TOL for x in grid.marking.skew_eps
    ):
        raise ConfigError(
            f"reference_mark {dict(ref)} is not a mark of grid {grid.name!r} "
            f"(ssr_target {list(grid.marking.ssr_target)}, skew_eps {list(grid.marking.skew_eps)})"
        )
    wanted = set(p["surfaces"])
    return [pt for pt in ctx.grid_points() if pt.mode == "marking" and pt.surface in wanted]


def requirements(ctx: StudyContext) -> list[Requirement]:
    reqs = [ctx.point_requirement(pt.id, f"marking fit {pt.label}") for pt in marking_points(ctx)]
    if ctx.params["recorded_m7_fits"]:
        reqs.append(ctx.artefact_requirement(M7_FITS, "recorded M7 marking fits", M7_COMMAND))
    return reqs


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def mark_label(surface: str, ssr: float, eps: float) -> str:
    return f"{surface} | ssr {ssr:g} | eps {eps:g}"


def mark_axes(surface: str, ssr: float, eps: float) -> dict[str, Any]:
    return {"surface": surface, "ssr_target": float(ssr), "skew_eps": float(eps)}


def window_label(t1: float, t2: float) -> str:
    """The M7 naming of a forward-start window (``1y-into-1y``)."""
    return f"{t1:g}y-into-{t2 - t1:g}y"


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "x"


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


def add_mc(
    b: ResultsBuilder,
    table: str,
    row: str,
    column: str,
    value: float,
    stderr: float,
    *,
    unit: str,
    source: str,
    axes: Mapping[str, Any],
    note: str = "",
) -> None:
    """A Monte Carlo number with its stderr, or a missing number (NaN, noted)."""
    if math.isfinite(value) and math.isfinite(stderr) and stderr >= 0.0:
        b.add(table, row, column, value, stderr, unit=unit, source=source, axes=axes, note=note)
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


def display(key: str, unit: str) -> tuple[float, str, str]:
    """``(scale, unit, note)`` turning a store product row into its reported unit (module
    docstring)."""
    if unit == "vol":
        return VOL_PTS, "vol pts", ""
    if unit == "% notional":
        if ":" in key:
            return 100.0, "% notional", M6_UNIT_NOTE
        return 1.0, "% notional", ""
    if unit in ("probability", "ratio", ""):
        return 1.0, DIMENSIONLESS, ""
    return 1.0, unit, ""


def _status_code(status: str) -> float:
    return STATUS_CODES.get(status, math.nan)


def _edge(text: str) -> float:
    t = text.strip()
    if t.startswith("(1+"):
        return EDGE_STEEP
    if t.startswith("(1-"):
        return EDGE_FLAT
    return EDGE_NONE


def _by_point(df: pd.DataFrame, ids: Sequence[str]) -> dict[str, pd.DataFrame]:
    if df.empty or "point_id" not in df.columns:
        return {}
    sub = df[df["point_id"].astype(str).isin(set(ids))]
    return {str(k): g.reset_index(drop=True) for k, g in sub.groupby("point_id", sort=False)}


# --------------------------------------------------------------------------------------------
# compute: the parts
# --------------------------------------------------------------------------------------------


def _setup(
    b: ResultsBuilder,
    ctx: StudyContext,
    grid: GridSpec,
    pts: Sequence[GridPoint],
    rows: Mapping[str, Mapping[str, Any]],
) -> None:
    for sname in ctx.params["surfaces"]:
        s = grid.surface(sname)
        mine = [rows[pt.id] for pt in pts if pt.surface == sname and pt.id in rows]
        axes = {"surface": sname}
        src = "computed"
        b.add_exact(
            "setup",
            sname,
            "surface_kind",
            KIND_CODES[s.kind],
            unit="",
            source=src,
            note=s.kind if s.path is None else f"{s.kind}: {s.path}",
            axes=axes,
        )
        b.add_exact(
            "setup",
            sname,
            "marks",
            float(len(mine)),
            unit="",
            source=src,
            axes=axes,
        )
        for col in ("n_particles", "horizon", "pricing_n_paths", "pricing_seed"):
            vals = sorted({_num(r.get(col)) for r in mine if math.isfinite(_num(r.get(col)))})
            b.add_exact(
                "setup",
                sname,
                col,
                vals[0] if len(vals) == 1 else math.nan,
                unit="",
                source=src,
                note="" if len(vals) <= 1 else f"several values: {vals}",
                axes=axes,
            )
        b.add_exact(
            "setup",
            sname,
            "fast_mode",
            1.0 if ctx.mode == "fast" else 0.0,
            unit="",
            source=src,
            axes=axes,
        )


def _fit(surface: Any, ssr: float, eps: float) -> tuple[FitResult, float]:
    """The precompute's own marking fit (:func:`volsto.viewers.grid.marking_fit`), timed."""
    t0 = time.perf_counter()
    r = marking_fit(surface, ssr, eps)
    return r, time.perf_counter() - t0


PARAMS: tuple[str, ...] = ("nu", "theta", "k1", "k2", "rho12", "rho_SX1", "rho_SX2")


def _binding_map(
    b: ResultsBuilder,
    ctx: StudyContext,
    surfaces: Mapping[str, Any],
    grid: GridSpec,
    rows: Mapping[str, Mapping[str, Any]],
    pts: Sequence[GridPoint],
) -> None:
    bm = ctx.params["binding_map"]
    grid_ids = {(pt.surface, pt.axes["ssr_target"], pt.axes["skew_eps"]): pt.id for pt in pts}
    walls: dict[str, float] = {}
    for sname, surface in surfaces.items():
        for ssr in (float(x) for x in bm["ssr_target"]):
            for eps in (float(x) for x in bm["skew_eps"]):
                r, wall = _fit(surface, ssr, eps)
                m = marking_summary(r)
                row = mark_label(sname, ssr, eps)
                walls[row] = round(wall, 3)
                axes = mark_axes(sname, ssr, eps)
                src = "computed"

                def ex(
                    col: str,
                    value: float,
                    note: str = "",
                    row: str = row,
                    axes: Mapping[str, Any] = axes,
                    src: str = src,
                ) -> None:
                    b.add_exact(
                        "binding_map", row, col, value, unit="", source=src, note=note, axes=axes
                    )

                ex(
                    "status_code",
                    _status_code(r.status),
                    r.status + ("; " + " | ".join(m.messages) if m.messages else ""),
                )
                for k in PARAMS:
                    ex(k, float(getattr(r.params, k)))
                cons = {str(c["name"]): c for c in r.constraints.to_dict("records")}
                for name in ("T_s", "T_l"):
                    c = cons.get(name)
                    if c is None:
                        ex(f"gap_{name}", math.nan, "no such constraint")
                        ex(f"edge_{name}", math.nan)
                        continue
                    ex(f"gap_{name}", float(c["gap_rel"]), f"T = {float(c['T']):g}y")
                    ex(
                        f"edge_{name}",
                        _edge(str(c.get("binding_edge", ""))),
                        f"T = {float(c['T']):g}y",
                    )
                ex(
                    "nu_box",
                    1.0 if any(str(a).startswith("nu box") for a in r.first.active) else 0.0,
                )
                ex(
                    "nu_at_cap",
                    1.0 if any("nu at cap" in str(f) for f in r.second.bound_flags) else 0.0,
                )
                miss = np.abs(np.asarray(r.svc_rel_error, dtype=float))
                ex("max_svc_miss", float(np.nanmax(miss)) if miss.size else math.nan, "first order")
                ex("mean_skew_gap", float(m.mean_skew_gap))
                ex(
                    "rho12_collapse",
                    1.0 if abs(float(r.params.rho12)) > float(r.config.rho12_flag) else 0.0,
                    f"|rho12| > {float(r.config.rho12_flag):g}",
                )
                pid = next(
                    (
                        v
                        for (s, gs, ge), v in grid_ids.items()
                        if s == sname and abs(gs - ssr) < TOL and abs(ge - eps) < TOL
                    ),
                    None,
                )
                ex("grid_point", 0.0 if pid is None else 1.0, pid or "")
                stored = rows.get(pid) if pid is not None else None
                if stored is None:
                    ex("store_param_diff", math.nan, "not a stored grid point")
                    ex("store_status_match", math.nan)
                    continue
                diffs = [
                    abs(float(getattr(r.params, k)) - _num(stored.get(k)))
                    for k in PARAMS
                    if math.isfinite(_num(stored.get(k)))
                ]
                ex(
                    "store_param_diff",
                    max(diffs) if diffs else math.nan,
                    "largest |inline fit - stored fit| over the seven parameters",
                )
                ex(
                    "store_status_match",
                    1.0 if _text(stored.get("status")) == r.status else 0.0,
                    f"stored status {_text(stored.get('status'))}",
                )
    ctx.record("binding_map_fit_seconds", walls)
    ctx.log.info(
        "binding map: %d inline fits in %.1f s (parameter fits, no leverage calibration)",
        len(walls),
        sum(walls.values()),
    )


def _marks(
    b: ResultsBuilder,
    ctx: StudyContext,
    pts: Sequence[GridPoint],
    rows: Mapping[str, Mapping[str, Any]],
    ssr: Mapping[str, pd.DataFrame],
) -> None:
    pillars = [float(x) for x in ctx.params["ssr_pillars"]]
    for pt in pts:
        rec = rows[pt.id]
        s_, e_ = pt.axes["ssr_target"], pt.axes["skew_eps"]
        row, axes, src = (
            mark_label(pt.surface, s_, e_),
            mark_axes(pt.surface, s_, e_),
            f"store:{pt.id}",
        )
        status = _text(rec.get("status"))
        msgs = _text(rec.get("fit_messages"))
        b.add_exact(
            "marks",
            row,
            "status_code",
            _status_code(status),
            unit="",
            source=src,
            note=status + (f"; {msgs}" if msgs else ""),
            axes=axes,
        )
        b.add_exact(
            "marks",
            row,
            "mean_abs_L_minus_1",
            _num(rec.get("mean_abs_L_minus_1")),
            unit="",
            source=src,
            note=L_NOTE,
            axes=axes,
        )
        table = ssr.get(pt.id, pd.DataFrame())
        for T in pillars:
            hit = (
                table[np.isclose(table["T"].astype(float), T, atol=TOL)]
                if not table.empty
                else table
            )
            v = _num(hit["ssr_lsv"].iloc[0]) if not hit.empty else math.nan
            se = _num(hit["ssr_lsv_stderr"].iloc[0]) if not hit.empty else math.nan
            add_mc(
                b,
                "marks",
                row,
                f"ssr_lsv@{T:g}y",
                v,
                se,
                unit=DIMENSIONLESS,
                source=src,
                axes=axes,
            )
            fo = (
                _num(hit["ssr_naked_first_order"].iloc[0])
                if not hit.empty and "ssr_naked_first_order" in hit.columns
                else math.nan
            )
            b.add_exact(
                "marks",
                row,
                f"ssr_first_order@{T:g}y",
                fo,
                unit="",
                source=src,
                axes=axes,
                note="the fit's first-order P1 SSR (exact)",
            )


def _forward_skew(
    b: ResultsBuilder,
    ctx: StudyContext,
    pts: Sequence[GridPoint],
    smiles: Mapping[str, pd.DataFrame],
    surfaces: Mapping[str, Any],
) -> None:
    windows = [(float(w[0]), float(w[1])) for w in ctx.params["forward_windows"]]
    for pt in pts:
        s_, e_ = pt.axes["ssr_target"], pt.axes["skew_eps"]
        row, axes, src = (
            mark_label(pt.surface, s_, e_),
            mark_axes(pt.surface, s_, e_),
            f"store:{pt.id}",
        )
        sm = smiles.get(pt.id, pd.DataFrame())
        for t1, t2 in windows:
            w = window_label(t1, t2)
            spot = VOL_PTS * spot_skew_90_110(surfaces[pt.surface], t2 - t1)
            b.add_exact(
                "forward_skew",
                row,
                f"spot_skew@{w}",
                spot,
                unit="vol pts",
                source="computed",
                axes=axes,
                note=f"snapshot SSVI at the window's length {t2 - t1:g}y, exact",
            )
            fs, fs_se, beyond = math.nan, math.nan, math.nan
            if not sm.empty:
                win = sm[
                    np.isclose(sm["t1"].astype(float), t1, atol=TOL)
                    & np.isclose(sm["t2"].astype(float), t2, atol=TOL)
                ]
                lo = win[np.isclose(win["strike_moneyness"].astype(float), SKEW_STRIKES[0])]
                hi = win[np.isclose(win["strike_moneyness"].astype(float), SKEW_STRIKES[1])]
                if not lo.empty and not hi.empty:
                    fs = VOL_PTS * (_num(lo["iv"].iloc[0]) - _num(hi["iv"].iloc[0]))
                    fs_se = VOL_PTS * (
                        _num(lo["iv_stderr"].iloc[0]) + _num(hi["iv_stderr"].iloc[0])
                    )
                if not win.empty and "beyond_horizon" in win.columns:
                    beyond = 1.0 if bool(win["beyond_horizon"].astype(bool).any()) else 0.0
            add_mc(
                b,
                "forward_skew",
                row,
                f"fwd_skew@{w}",
                fs,
                fs_se,
                unit="vol pts",
                source=src,
                axes=axes,
                note=FWD_SKEW_SE_NOTE,
            )
            ratio = fs / spot if spot else math.nan
            ratio_se = fs_se / abs(spot) if spot else math.nan
            add_mc(
                b,
                "forward_skew",
                row,
                f"ratio@{w}",
                ratio,
                ratio_se,
                unit=DIMENSIONLESS,
                source=f"{src};computed",
                axes=axes,
                note="stderr fs_se / |spot| (the spot skew is exact)",
            )
            b.add_exact(
                "forward_skew",
                row,
                f"beyond_horizon@{w}",
                beyond,
                unit="",
                source=src,
                axes=axes,
                note=BEYOND_NOTE,
            )


def _prices(
    b: ResultsBuilder,
    ctx: StudyContext,
    pts: Sequence[GridPoint],
    products: Mapping[str, pd.DataFrame],
) -> None:
    keys = list(ctx.params["products"])
    units: dict[str, tuple[float, str, str]] = {}
    for g in products.values():
        for rec in g.to_dict("records"):
            k = str(rec["key"])
            if k in keys and k not in units:
                units[k] = display(k, _text(rec.get("unit")))
    absent = [k for k in keys if k not in units]
    if absent:
        raise ConfigError(
            f"products {absent} are in no stored marking point (store keys are the M4 headline "
            "columns and '<product>:<column>' M6 cells)"
        )
    ref = ctx.params["reference_mark"]
    values: dict[tuple[str, str], tuple[float, float]] = {}
    for pt in pts:
        s_, e_ = pt.axes["ssr_target"], pt.axes["skew_eps"]
        row, axes, src = (
            mark_label(pt.surface, s_, e_),
            mark_axes(pt.surface, s_, e_),
            f"store:{pt.id}",
        )
        g = products.get(pt.id, pd.DataFrame())
        by_key = {str(r["key"]): r for r in g.to_dict("records")} if not g.empty else {}
        for k in keys:
            scale, unit, note = units[k]
            hit = by_key.get(k)
            v = scale * _num(hit["value"]) if hit is not None else math.nan
            se = scale * _num(hit["value_stderr"]) if hit is not None else math.nan
            values[(row, k)] = (v, se)
            add_mc(b, "prices", row, k, v, se, unit=unit, source=src, axes=axes, note=note)
    for pt in pts:
        s_, e_ = pt.axes["ssr_target"], pt.axes["skew_eps"]
        if abs(s_ - float(ref["ssr_target"])) < TOL and abs(e_ - float(ref["skew_eps"])) < TOL:
            continue
        row, axes = mark_label(pt.surface, s_, e_), mark_axes(pt.surface, s_, e_)
        ref_row = mark_label(pt.surface, float(ref["ssr_target"]), float(ref["skew_eps"]))
        ref_pt = next(
            (
                q.id
                for q in pts
                if q.surface == pt.surface
                and abs(q.axes["ssr_target"] - float(ref["ssr_target"])) < TOL
                and abs(q.axes["skew_eps"] - float(ref["skew_eps"])) < TOL
            ),
            None,
        )
        src = f"store:{pt.id}" + (f";store:{ref_pt}" if ref_pt else "")
        for k in keys:
            v, se = values[(row, k)]
            rv, rse = values.get((ref_row, k), (math.nan, math.nan))
            d, dse = v - rv, math.hypot(se, rse)
            add_mc(
                b,
                "mark_cost",
                row,
                k,
                d,
                dse,
                unit=units[k][1],
                source=src,
                axes=axes,
                note="price minus the reference mark's; " + QUADRATURE_NOTE,
            )
            z = d / dse if math.isfinite(d) and dse > 0 else math.nan
            add_mc(
                b,
                "mark_cost_z",
                row,
                k,
                z,
                Z_SD if math.isfinite(z) else math.nan,
                unit=DIMENSIONLESS,
                source=src,
                axes=axes,
                note=f"{Z_NOTE}; {QUADRATURE_NOTE}; {Z_SD_NOTE}",
            )


def _greeks(
    b: ResultsBuilder,
    ctx: StudyContext,
    pts: Sequence[GridPoint],
    rows: Mapping[str, Mapping[str, Any]],
    risk: Mapping[str, pd.DataFrame],
) -> None:
    g = ctx.params["greeks"]
    names, products = list(g["names"]), list(g["products"])
    if not names:
        return
    feasible = [pt for pt in pts if _text(rows[pt.id].get("status")) != "infeasible"]
    missing = [pt.id for pt in feasible if risk.get(pt.id, pd.DataFrame()).empty]
    if missing:
        cmd = f"{ctx.precompute_line(missing)} --risk light"
        reqs = [
            Requirement("point", pid, "the point's light-tier risk rows (store table risk)", cmd)
            for pid in missing
        ]
        raise MissingRequirements(reqs, [cmd])
    units: dict[tuple[str, str], str] = {}
    for df in risk.values():
        for rec in df.to_dict("records"):
            key = (str(rec["product"]), str(rec["name"]))
            units.setdefault(key, _text(rec.get("unit")))
    absent = [(x, n) for x in products for n in names if (x, n) not in units]
    if absent:
        have = sorted({n for (_, n) in units})
        raise ConfigError(
            f"greeks {absent} are in no stored marking point's risk rows (names there: {have})"
        )
    for product in products:
        table = f"greeks_{slug(product)}"
        for pt in feasible:
            s_, e_ = pt.axes["ssr_target"], pt.axes["skew_eps"]
            row, axes = mark_label(pt.surface, s_, e_), mark_axes(pt.surface, s_, e_)
            df = risk[pt.id]
            sub = df[df["product"].astype(str) == product]
            by_name = {str(r["name"]): r for r in sub.to_dict("records")}
            for name in names:
                hit = by_name.get(name)
                store_unit = units[(product, name)]
                unit = f"% notional {store_unit}".strip()
                v = RISK_PERCENT * _num(hit["value"]) if hit is not None else math.nan
                se = RISK_PERCENT * _num(hit["value_stderr"]) if hit is not None else math.nan
                add_mc(
                    b,
                    table,
                    row,
                    name,
                    v,
                    se,
                    unit=unit,
                    source=f"store:{pt.id}",
                    axes={**axes, "product": product},
                    note=RISK_NOTE,
                )


def _m7_fits(b: ResultsBuilder, ctx: StudyContext) -> None:
    path = ctx.artefact(M7_FITS)
    df = api.read_study_csv(path)
    src = f"artefact:{M7_FITS}"
    stderr_cols = {c for c in df.columns if str(c).endswith("_stderr")}
    for rec in df.to_dict("records"):
        surface = _text(rec.get("surface"))
        ssr, eps = _num(rec.get("ssr_target")), _num(rec.get("skew_eps"))
        row = mark_label(surface, ssr, eps)
        axes = mark_axes(surface, ssr, eps)
        status = _text(rec.get("status"))
        b.add_exact(
            "m7_fits",
            row,
            "status_code",
            _status_code(status),
            unit="",
            source=src,
            note=f"recorded M7 fit: {status}; stage-3 {_text(rec.get('stage3_assertion'))}",
            axes=axes,
        )
        for k in ("nu", "theta", "k1", "rho_SX1", "rho_SX2", "rho12"):
            b.add_exact("m7_fits", row, k, _num(rec.get(k)), unit="", source=src, axes=axes)
        b.add_exact(
            "m7_fits",
            row,
            "mean_abs_L_minus_1",
            _num(rec.get("mean_abs_L_minus_1")),
            unit="",
            source=src,
            note=L_NOTE,
            axes=axes,
        )
        for c in df.columns:
            name = str(c)
            if not (name.startswith("ssr_lsv@") or name.startswith("fwd/spot@")):
                continue
            if name.endswith("_stderr") or f"{name}_stderr" not in stderr_cols:
                continue
            add_mc(
                b,
                "m7_fits",
                row,
                name,
                _num(rec.get(name)),
                _num(rec.get(f"{name}_stderr")),
                unit=DIMENSIONLESS,
                source=src,
                axes=axes,
            )


# --------------------------------------------------------------------------------------------
# compute
# --------------------------------------------------------------------------------------------


def compute(ctx: StudyContext) -> Results:
    p = ctx.params
    grid = _grid(ctx)
    pts = marking_points(ctx)
    ids = [pt.id for pt in pts]
    # whole-table reads through a plain reader, filtered to the configured marking points (all
    # of them already recorded in the manifest as point requirements)
    reader = StoreReader(ctx.store_root)
    pdf = reader.points()
    rows: dict[str, dict[str, Any]] = {
        str(r["point_id"]): {str(k): v for k, v in r.items()}
        for r in (
            pdf[pdf["point_id"].astype(str).isin(set(ids))] if not pdf.empty else pdf
        ).to_dict("records")
    }
    lost = [i for i in ids if i not in rows]
    if lost:  # present at the requirement check, gone since
        raise MissingRequirements(
            [ctx.point_requirement(i, "marking point") for i in lost],
            [ctx.precompute_line(lost)],
        )
    products = _by_point(reader.products(), ids)
    ssr = _by_point(reader.ssr(), ids)
    smiles = _by_point(reader.forward_smile(), ids)
    risk = _by_point(reader.risk(), ids) if p["greeks"]["names"] else {}
    paths = sorted(
        {
            int(_num(r.get("pricing_n_paths")))
            for r in rows.values()
            if math.isfinite(_num(r.get("pricing_n_paths")))
        }
    )
    ctx.record("n_paths", paths[0] if len(paths) == 1 else paths)
    t0 = time.perf_counter()
    ref = reference_spec(grid)
    surfaces = {s: build_market(surface_spec(grid, grid.surface(s), ref))[1] for s in p["surfaces"]}
    ctx.record("surface_build_seconds", round(time.perf_counter() - t0, 3))
    b = ResultsBuilder()
    _setup(b, ctx, grid, pts, rows)
    _binding_map(b, ctx, surfaces, grid, rows, pts)
    _marks(b, ctx, pts, rows, ssr)
    _forward_skew(b, ctx, pts, smiles, surfaces)
    _prices(b, ctx, pts, products)
    _greeks(b, ctx, pts, rows, risk)
    if p["recorded_m7_fits"]:
        _m7_fits(b, ctx)
    return b.build()


# --------------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------------


HEADERS: dict[str, str] = {
    "status_code": "status",
    "mean_abs_L_minus_1": "mean |L-1|",
    "surface_kind": "kind",
    "n_particles": "particles",
    "pricing_n_paths": "pricing paths",
    "pricing_seed": "seed",
    "fast_mode": "fast",
}


def header(key: str) -> str:
    """A column header: the declared name, else the key made readable (``ssr_lsv@1y`` ->
    ``SSR LSV 1y``)."""
    if key in HEADERS:
        return HEADERS[key]
    name, _, at = key.partition("@")
    name = {
        "ssr_lsv": "SSR LSV",
        "ssr_first_order": "SSR 1st order",
        "fwd_skew": "fwd skew",
        "spot_skew": "spot skew",
        "ratio": "fwd/spot",
        "beyond_horizon": "beyond horizon",
    }.get(name, name.replace("_", " "))
    return f"{name} {at.replace('-into-', ' into ')}".strip()


def _cols(results: Results, table: str, *, digits: int = 4) -> tuple[Column, ...]:
    return tuple(Column(c, header(c), digits=digits) for c in results.columns(table))


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


BINDING_COLUMNS: tuple[tuple[str, str, int], ...] = (
    ("status_code", "status", 1),
    ("nu", "nu", 4),
    ("theta", "theta", 3),
    ("k1", "k1", 4),
    ("rho_SX1", "rho SX1", 3),
    ("rho_SX2", "rho SX2", 3),
    ("rho12", "rho12", 3),
    ("gap_T_s", "gap T_s", 3),
    ("edge_T_s", "edge T_s", 1),
    ("gap_T_l", "gap T_l", 3),
    ("edge_T_l", "edge T_l", 1),
    ("nu_box", "nu box", 1),
    ("nu_at_cap", "nu cap", 1),
    ("max_svc_miss", "max SVC miss", 3),
    ("rho12_collapse", "rho12 collapse", 1),
    ("store_param_diff", "vs store", 2),
)
#: Per-surface tables: results table -> caption.
SURFACE_TABLES: dict[str, str] = {
    "binding_map": (
        "the binding map, fitted inline (P1 marking fit without stage 3: exact, no Monte "
        "Carlo): status 0 interior / 1 binding / 2 infeasible; the parameters; the relative "
        "naked-skew gap and binding edge (+1 steep, -1 flat, 0 inside) at the two constraint "
        "pillars (T in the notes); the nu box / nu cap flags; the largest first-order "
        "SpotVolCovar miss; the rho12 collapse flag; for grid points the largest parameter "
        "difference to the stored fit."
    ),
    "marks": (
        "per stored mark: status, mean |L-1| (exact functional of the cached leverage, its "
        "particle-seed noise not estimated upstream), the calibrated LSV's realised SSR (+/- "
        "stderr) and the fit's first-order P1 SSR at the pillars."
    ),
    "forward_skew": (
        "forward 90/110 skew of the stored forward smiles (stderr: the sum of the two strikes' "
        "errors, a bound for any correlation) against the spot 90/110 skew of the same tenor "
        "(exact) and their ratio (stderr fs_se/|spot|); beyond horizon = 1 when the window ends "
        "past the calibration horizon."
    ),
    "prices": (
        "the product price surface over the two dials (store, value +/- stderr; vols in vol "
        "points, M6 cells converted from fractions to % of notional)."
    ),
    "mark_cost": (
        "what each mark costs or earns: price minus the price at the reference mark (stderrs "
        "in quadrature - not a bound: the store's same-seed estimates have an unmeasured "
        "correlation); positive = the mark prices the product higher."
    ),
}


def tables(results: Results) -> list[TableSpec]:
    have = set(results.tables())
    out: list[TableSpec] = []
    if "setup" in have:
        out.append(
            TableSpec(
                "setup",
                "Surfaces studied: kind (0 placeholder, 1 snapshot), stored marks, the store's "
                "particle count, horizon, pricing paths and seed, and the run mode (1 = fast).",
                "setup",
                _cols(results, "setup", digits=6),
                row_header="surface",
            )
        )
    for table, caption in SURFACE_TABLES.items():
        if table not in have:
            continue
        if table == "binding_map":
            cols = set(results.columns(table))
            columns = tuple(Column(k, h, digits=d) for k, h, d in BINDING_COLUMNS if k in cols)
        elif table in ("prices", "mark_cost"):
            columns = tuple(Column(c, c) for c in results.columns(table))
        else:
            columns = _cols(results, table)
        for surface, rows in _rows_by(results, table, "surface").items():
            out.append(
                TableSpec(
                    f"{table}_{slug(surface)}",
                    f"{surface}: {caption}",
                    table,
                    columns,
                    rows=tuple(rows),
                    row_header="surface | ssr | eps",
                )
            )
    for t in sorted(t for t in have if t.startswith("greeks_")):
        product = str(results.long(t)["axis_product"].iloc[0])
        for surface, rows in _rows_by(results, t, "surface").items():
            out.append(
                TableSpec(
                    f"{t}_{slug(surface)}",
                    f"{surface}: stored risk of {product} per mark (value +/- stderr; the store's "
                    "fractions of notional x 100, per the store's bump unit).",
                    t,
                    tuple(Column(c, c) for c in results.columns(t)),
                    rows=tuple(rows),
                    row_header="surface | ssr | eps",
                )
            )
    if "m7_fits" in have:
        out.append(
            TableSpec(
                "m7_fits",
                "The recorded M7 fits (m7/p1_marking_fits.csv, scripts/m7_p1_marking.py; not "
                "recomputed): status, parameters, mean |L-1| (exact, no stderr upstream), the "
                "realised SSR and the forward/spot 90/110 skew ratio.",
                "m7_fits",
                _cols(results, "m7_fits"),
                row_header="surface | ssr | eps",
            )
        )
    return fit_specs(out, results)


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------


def _flat(axes: Any) -> list[Any]:
    return list(np.atleast_1d(axes).ravel())


def _surfaces(long: pd.DataFrame) -> list[str]:
    return [str(s) for s in dict.fromkeys(long["axis_surface"])]


def _grid_shape(n: int, max_cols: int = 3) -> tuple[int, int]:
    ncols = min(max(n, 1), max_cols)
    return (max(n, 1) + ncols - 1) // ncols, ncols


def _draw_binding(results: Results) -> Any:
    long = results.long("binding_map")
    surfaces = _surfaces(long)
    nrows, ncols = _grid_shape(len(surfaces))
    fig, axes = style.new_figure(nrows, ncols, width=4.2 * ncols, height=3.4 * nrows)
    status_series = {0.0: 2, 1.0: 1, 2.0: 7}
    for ax, sname in zip(_flat(axes), surfaces):
        sub = long[long["axis_surface"] == sname]
        vals = {
            (str(r["row"]), str(r["column"])): float(r["value"]) for r in sub.to_dict("records")
        }
        coords = {str(r["row"]): r for r in sub.to_dict("records")}
        ssrs = sorted({float(c["axis_ssr_target"]) for c in coords.values()})
        epss = sorted({float(c["axis_skew_eps"]) for c in coords.values()})
        seen: set[float] = set()
        for row, c in coords.items():
            x = ssrs.index(float(c["axis_ssr_target"]))
            y = epss.index(float(c["axis_skew_eps"]))
            code = vals.get((row, "status_code"), math.nan)
            ser = status_series.get(code, 6)
            label = STATUS_NAMES.get(code, "unknown")
            ax.scatter(
                [x],
                [y],
                s=260,
                color=style.series_color(ser),
                marker=style.MARKERS[ser],
                label=None if code in seen else label,
                zorder=3,
            )
            seen.add(code)
            edges = []
            for name in ("T_s", "T_l"):
                e = vals.get((row, f"edge_{name}"), math.nan)
                if math.isfinite(e):
                    edges.append({EDGE_STEEP: "S", EDGE_FLAT: "F"}.get(e, "."))
            ax.annotate(
                "".join(edges),
                (x, y),
                textcoords="offset points",
                xytext=(0, 11),
                ha="center",
                fontsize=7,
                color=style.INK["secondary"],
            )
        ax.set_xticks(range(len(ssrs)), [f"{v:g}" for v in ssrs])
        ax.set_yticks(range(len(epss)), [f"{v:g}" for v in epss])
        ax.set_xlim(-0.6, len(ssrs) - 0.4)
        ax.set_ylim(-0.6, len(epss) - 0.4)
        ax.set_xlabel("ssr_target")
        ax.set_ylabel("skew_eps")
        ax.set_title(f"{sname} (edges T_s T_l: S steep, F flat)", fontsize=9)
        ax.legend(fontsize=7, loc="upper left")
    for ax in _flat(axes)[len(surfaces) :]:
        ax.set_visible(False)
    return fig


def _pillar_column(results: Results) -> str | None:
    cols = [c for c in results.columns("marks") if c.startswith("ssr_lsv@")]
    if not cols:
        return None
    ts = [float(c[len("ssr_lsv@") : -1]) for c in cols]
    return cols[int(np.argmin([abs(t - 1.0) for t in ts]))]


def _by_eps(ax: Any, sub: pd.DataFrame, *, exact: bool = False, label_prefix: str = "eps") -> None:
    for i, eps in enumerate(sorted(set(sub["axis_skew_eps"].astype(float)))):
        if i >= len(style.PALETTE):
            break
        part = sub[np.isclose(sub["axis_skew_eps"].astype(float), eps)]
        style.mc_errorbar(
            ax,
            part["axis_ssr_target"].to_numpy(dtype=float),
            part["value"].to_numpy(dtype=float),
            part["stderr"].to_numpy(dtype=float),
            exact=np.full(len(part), exact) if exact else part["exact"].to_numpy(dtype=bool),
            series=i,
            label=f"{label_prefix} {eps:g}",
        )


def _draw_ssr(results: Results) -> Any:
    col = _pillar_column(results)
    assert col is not None
    long = results.long("marks", col)
    surfaces = _surfaces(long)
    nrows, ncols = _grid_shape(len(surfaces))
    fig, axes = style.new_figure(nrows, ncols, width=4.0 * ncols, height=3.4 * nrows)
    for ax, sname in zip(_flat(axes), surfaces):
        sub = long[long["axis_surface"] == sname]
        xs = sorted(set(sub["axis_ssr_target"].astype(float)))
        ax.plot(
            xs, xs, linestyle="--", color=style.INK["spine"], linewidth=1.0, label="= ssr_target"
        )
        _by_eps(ax, sub)
        ax.set_xlabel("ssr_target")
        ax.set_ylabel(f"realised LSV SSR at {col[len('ssr_lsv@'):]}")
        ax.set_title(sname)
        ax.legend(fontsize=7)
    for ax in _flat(axes)[len(surfaces) :]:
        ax.set_visible(False)
    return fig


def _draw_ratio(results: Results) -> Any:
    col = next(c for c in results.columns("forward_skew") if c.startswith("ratio@"))
    long = results.long("forward_skew", col)
    surfaces = _surfaces(long)
    nrows, ncols = _grid_shape(len(surfaces))
    fig, axes = style.new_figure(nrows, ncols, width=4.0 * ncols, height=3.4 * nrows)
    for ax, sname in zip(_flat(axes), surfaces):
        sub = long[long["axis_surface"] == sname]
        ax.axhspan(*DECK_RATIO_BAND, color=style.INK["grid"], alpha=0.7, zorder=0)
        ax.axhline(1.0, color=style.INK["spine"], linewidth=0.8)
        _by_eps(ax, sub)
        ax.set_xlabel("ssr_target")
        ax.set_ylabel(f"forward / spot 90/110 skew, {col[len('ratio@'):]}")
        ax.set_title(f"{sname} (band: the deck's 1.3-1.5)", fontsize=9)
        ax.legend(fontsize=7)
    for ax in _flat(axes)[len(surfaces) :]:
        ax.set_visible(False)
    return fig


def _draw_leverage(results: Results) -> Any:
    long = results.long("marks", "mean_abs_L_minus_1")
    surfaces = _surfaces(long)
    nrows, ncols = _grid_shape(len(surfaces))
    fig, axes = style.new_figure(nrows, ncols, width=4.0 * ncols, height=3.4 * nrows)
    for ax, sname in zip(_flat(axes), surfaces):
        sub = long[long["axis_surface"] == sname]
        _by_eps(ax, sub, exact=True)
        ax.set_xlabel("ssr_target")
        ax.set_ylabel("mean |L - 1|")
        ax.set_title(f"{sname} (exact functional of the leverage)", fontsize=9)
        ax.legend(fontsize=7)
    for ax in _flat(axes)[len(surfaces) :]:
        ax.set_visible(False)
    return fig


def _draw_cost(results: Results, surface: str) -> Any:
    long = results.long("mark_cost")
    long = long[long["axis_surface"] == surface]
    keys = list(dict.fromkeys(long["column"]))
    nrows, ncols = _grid_shape(len(keys))
    fig, axes = style.new_figure(nrows, ncols, width=3.6 * ncols, height=3.0 * nrows)
    for ax, key in zip(_flat(axes), keys):
        sub = long[long["column"] == key]
        ax.axhline(0.0, color=style.INK["spine"], linewidth=0.8)
        _by_eps(ax, sub)
        unit = next((u for u in sub["unit"] if u and u != DIMENSIONLESS), "")
        ax.set_xlabel("ssr_target")
        ax.set_ylabel(f"minus reference [{unit}]" if unit else "minus reference")
        ax.set_title(key, fontsize=9)
        ax.legend(fontsize=6)
    for ax in _flat(axes)[len(keys) :]:
        ax.set_visible(False)
    return fig


def figures(results: Results) -> list[FigureSpec]:
    have = set(results.tables())
    out: list[FigureSpec] = []
    if "binding_map" in have:
        out.append(
            FigureSpec(
                "binding_map",
                "The binding map per surface over (ssr_target, skew_eps), fitted inline (exact): "
                "status and the binding edge at the two constraint pillars.",
                _draw_binding,
            )
        )
    if "marks" in have and _pillar_column(results) is not None:
        out.append(
            FigureSpec(
                "realised_ssr",
                "Realised SSR of the calibrated LSV against the SSR mark (dashed: realised = "
                "target; error bars: 1 stderr).",
                _draw_ssr,
            )
        )
        out.append(
            FigureSpec(
                "leverage_deviation",
                "Mean |L-1| per mark (exact functional of the cached leverage; drawn without "
                "error bars because its particle-seed noise is not estimated upstream, not "
                "because it has none).",
                _draw_leverage,
            )
        )
    if "forward_skew" in have and any(
        c.startswith("ratio@") for c in results.columns("forward_skew")
    ):
        out.append(
            FigureSpec(
                "fwd_spot_ratio",
                "Forward / spot 90/110 skew ratio per mark (grey band: the deck's 130-150%; "
                "error bars: 1 stderr).",
                _draw_ratio,
            )
        )
    if "mark_cost" in have:
        for sname in _surfaces(results.long("mark_cost")):
            out.append(
                FigureSpec(
                    f"mark_cost_{slug(sname)}",
                    f"{sname}: price at each mark minus the price at the reference mark, per "
                    "product (error bars: 1 stderr in quadrature - not a bound, the same-seed "
                    "correlation is not measured).",
                    functools.partial(_draw_cost, surface=sname),
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
    declared name with that prefix (the per-surface tables and the parts of a split table)."""
    out: list[str] = []
    for n in names:
        hits = (
            [d for d in declared if d.startswith(n[:-1])]
            if n.endswith("*")
            else [d for d in declared if d == n or re.fullmatch(re.escape(n) + r"_\d+", d)]
        )
        for d in hits:
            if "{{" + d + "}}" not in out:
                out += ["{{" + d + "}}", ""]
    return out


def _ex(results: Results, table: str, row: str, column: str) -> float:
    return results.value(table, row, column)[0]


def _setup_lines(results: Results) -> list[str]:
    lines: list[str] = []
    rows = results.rows("setup")
    kinds = {r: _ex(results, "setup", r, "surface_kind") for r in rows}
    if any(_ex(results, "setup", r, "fast_mode") == 1.0 for r in rows):
        lines.append(
            "**Fast mode** (CI plumbing on a toy grid): the numbers below check the pipeline, "
            "not the marking question."
        )
    placeholders = [r for r, k in kinds.items() if k == KIND_CODES["placeholder"]]
    if placeholders:
        lines.append(f"**{', '.join(placeholders)}:** {PLACEHOLDER_WARNING}")
    for r in rows:
        n = _ex(results, "setup", r, "n_particles")
        paths = _ex(results, "setup", r, "pricing_n_paths")
        hz = _ex(results, "setup", r, "horizon")
        marks = int(_ex(results, "setup", r, "marks"))
        lines.append(
            f"- {r}: {marks} stored marks; leverage at {n:.0f} particles over {hz:g}y; prices at "
            f"{paths:.0f} paths (seed {_ex(results, 'setup', r, 'pricing_seed'):.0f})."
            if math.isfinite(n) and math.isfinite(paths)
            else f"- {r}: {marks} stored marks."
        )
    return [*lines, ""]


EDGE_WORDS: dict[float, str] = {
    EDGE_STEEP: "the steep edge",
    EDGE_FLAT: "the flat edge",
    EDGE_NONE: "the inside of its slab",
}


def _edge_words(frame: pd.DataFrame) -> str:
    return " / ".join(sorted({EDGE_WORDS.get(float(v), "?") for v in frame["value"]}))


def _binding_lines(results: Results) -> list[str]:
    long = results.long("binding_map")
    status = long[long["column"] == "status_code"]
    lines = [
        "## The binding map",
        "",
        "Fitted inline: the P1 marking fit without stage 3 is a parameter fit of a few seconds "
        "(first-order closed forms, a QP per k1, a multi-start SLSQP), **not a leverage "
        "calibration** — nothing is simulated, and the runner's guard refuses any leverage "
        "calibration while the study computes.",
        "",
    ]
    for sname in _surfaces(status):
        sub = status[status["axis_surface"] == sname]
        counts = {name: int((sub["value"] == code).sum()) for name, code in STATUS_CODES.items()}
        interior = [
            f"({r['axis_ssr_target']:g}, {r['axis_skew_eps']:g})"
            for r in sub.to_dict("records")
            if r["value"] == STATUS_CODES["interior"]
        ]
        text = (
            f"- **{sname}**: {counts['binding']} binding, {counts['interior']} interior, "
            f"{counts['infeasible']} infeasible of {len(sub)} pairs"
        )
        text += f"; interior at {', '.join(interior)}." if interior else "."
        edges = long[(long["axis_surface"] == sname) & (long["column"] == "edge_T_s")]
        if not edges.empty:
            lo = edges[edges["axis_ssr_target"] == edges["axis_ssr_target"].min()]
            hi = edges[edges["axis_ssr_target"] == edges["axis_ssr_target"].max()]
            text += (
                f" The first constraint pillar's naked skew sits at {_edge_words(lo)} at the "
                f"lowest ssr_target and at {_edge_words(hi)} at the highest."
            )
        lines.append(text)
    diffs = long[(long["column"] == "store_param_diff") & np.isfinite(long["value"])]
    if not diffs.empty:
        match = long[(long["column"] == "store_status_match") & np.isfinite(long["value"])]
        lines += [
            "",
            f"The inline fit reproduces the stored fit at {len(diffs)} grid point(s): largest "
            f"parameter difference {diffs['value'].max():.2e}; status identical at "
            f"{int(match['value'].sum())} of {len(match)}.",
        ]
    return [*lines, ""]


def _marks_lines(results: Results) -> list[str]:
    lines = ["## Leverage, realised SSR and forward skew", ""]
    col = _pillar_column(results)
    marks = results.long("marks")
    for sname in _surfaces(marks):
        sub = marks[marks["axis_surface"] == sname]
        lev = sub[(sub["column"] == "mean_abs_L_minus_1") & np.isfinite(sub["value"])]
        text = f"- **{sname}**:"
        if not lev.empty:
            text += f" mean |L-1| from {lev['value'].min():.3f} to {lev['value'].max():.3f};"
        if col is not None:
            ssr = sub[(sub["column"] == col) & np.isfinite(sub["value"])]
            if not ssr.empty:
                parts = []
                for target, g in ssr.groupby("axis_ssr_target"):
                    lo, hi = g.loc[g["value"].idxmin()], g.loc[g["value"].idxmax()]
                    parts.append(
                        f"target {float(target):g} -> "
                        f"{latex.format_value_text(float(lo['value']), float(lo['stderr']))}"
                        + (
                            ""
                            if lo["row"] == hi["row"]
                            else " to "
                            f"{latex.format_value_text(float(hi['value']), float(hi['stderr']))}"
                        )
                    )
                text += f" realised SSR at {col[len('ssr_lsv@'):]}: " + "; ".join(parts) + "."
        lines.append(text)
    if "forward_skew" in results.tables():
        fs = results.long("forward_skew")
        for c in [c for c in results.columns("forward_skew") if c.startswith("ratio@")]:
            part = fs[(fs["column"] == c) & np.isfinite(fs["value"])]
            if part.empty:
                continue
            beyond = fs[
                (fs["column"] == c.replace("ratio@", "beyond_horizon@")) & (fs["value"] == 1.0)
            ]
            lines.append(
                f"- forward / spot 90/110 skew ratio {c[len('ratio@'):]}: "
                f"{part['value'].min():.2f} to {part['value'].max():.2f} over {len(part)} marks "
                f"(the deck's band is {DECK_RATIO_BAND[0]:g}-{DECK_RATIO_BAND[1]:g})"
                + (
                    f"; {len(beyond)} mark(s) price this window beyond the calibration horizon"
                    if len(beyond)
                    else ""
                )
                + "."
            )
    return [*lines, ""]


def _cost_lines(results: Results) -> list[str]:
    if "mark_cost" not in results.tables():
        return []
    long = results.long("mark_cost")
    zs = results.long("mark_cost_z")
    lines = [
        "## What the mark costs or earns",
        "",
        "Price at a mark minus the price at the reference mark on the same surface (positive = "
        "the mark prices the product higher: a desk short the product marks a larger liability "
        "and would charge a higher fee at inception).",
        "",
    ]
    for key in dict.fromkeys(long["column"]):
        sub = long[(long["column"] == key) & np.isfinite(long["value"])]
        if sub.empty:
            lines.append(f"- {key}: no finite difference.")
            continue
        lo, hi = sub.loc[sub["value"].idxmin()], sub.loc[sub["value"].idxmax()]
        z = zs[(zs["column"] == key) & np.isfinite(zs["value"])]
        zmax = float(z["value"].abs().max()) if not z.empty else math.nan
        unit = str(lo["unit"]) if lo["unit"] != DIMENSIONLESS else ""
        lines.append(
            f"- **{key}**: from "
            f"{latex.format_value_text(float(lo['value']), float(lo['stderr']))} ({lo['row']}) to "
            f"{latex.format_value_text(float(hi['value']), float(hi['stderr']))} ({hi['row']}) "
            f"{unit}; largest |z| {zmax:.1f}."
        )
    return [*lines, ""]


def narrative(results: Results) -> str:
    declared = _declared(results)
    have = set(results.tables())
    lines = [
        "Read from the results store (every number that needs a calibrated leverage), computed "
        "inline (the binding map and the spot skew of the snapshot, both exact), or read from "
        "the recorded M7 outputs (labelled). Every Monte Carlo number carries its stderr.",
        "",
    ]
    if "setup" in have:
        lines += _setup_lines(results)
        lines += _place(declared, "table:setup")
    if "binding_map" in have:
        lines += _binding_lines(results)
        lines += _place(declared, "table:binding_map_*", "figure:binding_map")
    if "marks" in have:
        lines += _marks_lines(results)
        lines += _place(
            declared,
            "table:marks_*",
            "figure:realised_ssr",
            "figure:leverage_deviation",
            "table:forward_skew_*",
            "figure:fwd_spot_ratio",
        )
    lines += _cost_lines(results)
    lines += _place(declared, "table:prices_*", "table:mark_cost_*", "figure:mark_cost_*")
    greeks = sorted(t for t in have if t.startswith("greeks_"))
    if greeks:
        lines += ["## Greeks per mark", ""]
        lines += _place(declared, "table:greeks_*")
    if "m7_fits" in have:
        lines += [
            "## The recorded M7 fits",
            "",
            "As recorded by `scripts/m7_p1_marking.py` (§15 Part 3), not recomputed here: the "
            "SPX and placeholder-reference fits at (1.0, 0.10) and (1.5, 0.05).",
            "",
            *_place(declared, "table:m7_fits"),
        ]
    return "\n".join(lines)
