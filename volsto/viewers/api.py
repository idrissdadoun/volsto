"""Read API of the viewers (SPEC §9.2 "Read API and app", M9 Part 3; owner's ``viewers/api.py``,
here ``volsto/viewers/api.py``): the small, typed surface the Streamlit pages call and the one
an MCP server (M11) will sit on.  Every function takes a :class:`~volsto.viewers.config.
ViewerConfig` and returns compact DataFrames, dicts or small record dataclasses — never a path
into Monte Carlo output.

**Standing rule.** Nothing here calibrates a leverage or runs a Monte Carlo: the functions read
the results store (:class:`~volsto.viewers.store.StoreReader`), the leverage cache **by file**
(``<cache>/<key>/leverage.npz`` and ``diagnostics.json`` — :class:`~volsto.calibration.cache.
LeverageCache.get_or_calibrate` is not imported), the M7 tables under ``<outputs>/m7`` and the
M8b task results under ``<outputs>/m8b``.  A missing point raises :class:`MissingPoint` whose
``command`` is the exact ``volsto-precompute`` line that produces it — the grid the *store* was
built with (:func:`grid_path_of`), the point named with ``--only <id>``, ``--resume`` for a
point that is not stored and ``--only <id> --risk light`` (no ``--resume``, the point is
already stored) for a missing risk table: the contract in :mod:`volsto.viewers.precompute`'s
docstring.  A missing study file raises :class:`MissingArtefact` naming the script
(``scripts/m7_p1_marking.py``, ``scripts/m8b.py --study ...``); an unknown *surface* is not a
computation at all and its command says to add it to the grid YAML first.
``tests/test_viewers_api.py`` points the cache at an empty directory and asserts it stays
empty.

Functions (the M11 surface):

* :func:`list_grid` — one row per stored point: id, label, surface, mode, status, the model
  parameters and axis values, particle count, ``has_*`` flags per table and ``has_leverage``.
* :func:`get_point` → :class:`PointRecord` — parameters, calibration diagnostics, and the
  ``products``, ``forward_smile``, ``forward_vols``, ``ssr``, ``varv`` frames of one point.
* :func:`list_surfaces` / :func:`get_surface` → :class:`SurfaceRecord` — the target
  :class:`~volsto.market.surface.ImpliedSurface` built from the placeholder / snapshot config
  (pure construction) and its Dupire :class:`~volsto.market.dupire.LocalVolSurface`.
* :func:`get_leverage` → :class:`LeverageRecord` — the cached
  :class:`~volsto.models.leverage.LeverageFunction` of a point with its
  :class:`~volsto.calibration.diagnostics.CalibrationReport` (the error map) when stored.
* :func:`get_products` — the product catalogue (name, class, unit, term-sheet fields) of the
  term-sheet editor, flagged with what the store holds.
* :func:`get_risk` — the stored :class:`~volsto.risk.report.RiskReport` rows of one point and
  product: section, name, T / bucket, value, value_stderr, unit, tier.
* :func:`list_hedging_runs` / :func:`get_hedging_run` → :class:`HedgingRunRecord` /
  :func:`get_hedging_pnl` / :func:`get_hedging_table` — the stored M8b task results.
* :func:`get_marking` → :class:`MarkingRecord` — the M7 Part 3 tables (fits, shadow rotation,
  binding maps).

**Standard errors.** Every Monte Carlo quantity a return carries has a ``<name>_stderr`` twin
(:func:`columns_without_stderr` walks a frame — the twin is required whatever the values, an
all-NaN column included; the test walks every return on the synthetic store).  Exact fields
carry none and are declared: the model parameters and axis values (``nu, theta, k1, k2, rho12,
rho_SX1, rho_SX2, axis_*, ssr_target, skew_eps``), grid axes and maturities (``T, t1, t2, k,
strike_moneyness, log_moneyness, bucket``), counts and settings (``n_particles, horizon,
pricing_n_paths, pricing_seed, n_paths_*, n_dates, size, eps, rota, scale``), measured wall
clocks (``*_seconds``, ``wall_*``), the error-map maxima of the store's ``points`` table
(``max_abs_error_vp, max_z`` — maxima of a map whose cells carry their own stderr in
:class:`LeverageRecord`), the first-order (analytic) SSR ``ssr_naked_first_order``, the
z-scores (``z``, ``*_z_*``) and the closed-form surface quantities of :class:`SurfaceRecord`
(implied and local vols are exact SSVI / Dupire values).  Exactness is **scoped** where a name
is exact in one table only: :data:`TABLE_EXACT` holds the surface tables' ``iv`` / ``atm_vol`` /
``atm_skew`` / ``local_vol`` / ``forward``, so the Monte Carlo ``iv`` of a stored
``forward_smile`` is still walked (review:S2 F6).  The M7 and M8b files are read as written
(:func:`read_study_csv`); their ``*_se`` columns are renamed ``*_stderr``
(:func:`normalise_stderr_names`) and re-attached to the value column of whichever desk-naming
convention the table uses (:func:`reattach_stderr_names`), the ``per_vp_90_110_*`` scalings of
the shadow-rotation table get the scaled stderr of ``per_rota``, and the columns those scripts
report without a standard error are listed in :data:`UNPAIRED_UPSTREAM` /
:data:`MARKING_UNPAIRED_PREFIXES` (an open item for those runners, not silently padded).

Checked by ``tests/test_viewers_api.py``.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pickle
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volsto.calibration.cache import build_market, has_complete_leverage
from volsto.calibration.diagnostics import CalibrationReport
from volsto.config import CalibrationSpec, from_mapping
from volsto.market.curves import ForwardCurve
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import ImpliedSurface, atm_skew_numeric
from volsto.models.leverage import LeverageFunction
from volsto.risk.shadow_rotation import ROTATION_CONVENTION
from volsto.studies.m8b import TaskResult
from volsto.viewers.config import ViewerConfig
from volsto.viewers.grid import (
    REPO_ROOT,
    GridSpec,
    SurfaceSpec,
    load_grid,
    reference_spec,
    surface_spec,
)
from volsto.viewers.store import NON_MC_COLUMNS, TABLES, StoreReader

MODEL_PARAMS: tuple[str, ...] = ("nu", "theta", "k1", "k2", "rho12", "rho_SX1", "rho_SX2")
AXES: tuple[str, ...] = ("axis_nu", "axis_rho", "axis_kappa", "ssr_target", "skew_eps")
#: Per-point tables the ``has_<table>`` flags of :func:`list_grid` report.
POINT_TABLES: tuple[str, ...] = TABLES[1:]
#: M7 output files of :func:`get_marking` (relative to ``<outputs>/m7``).
MARKING_FILES: dict[str, str] = {
    "fits": "p1_marking_fits.csv",
    "shadow_rotation": "p1_marking_shadow_rotation.csv",
    "soft_skew": "spx_soft_skew.csv",
    "skew_tradeoff": "skew_tradeoff.csv",
}
MARKING_SCRIPT = "scripts/m7_p1_marking.py"
HEDGING_SCRIPT = "scripts/m8b.py --study {study}"
#: 6M and 1y 90/110 skew move per +1 rota (vol points): ``per_vp = per_rota / this``
#: (:data:`~volsto.risk.shadow_rotation.ROTATION_CONVENTION`: 2/sqrt(T) vp per unit
#: log-moneyness on a 0.2-wide 90/110 span → 0.56 vp at 6M, 0.4 vp at 1y).
ROTA_VP_PER_UNIT: dict[str, float] = {"0.5y": 0.56, "1y": 0.4}

#: Numeric columns that are exact (no standard error) on top of the store's
#: :data:`~volsto.viewers.store.NON_MC_COLUMNS` — parameters, axes, counts, settings, measured
#: wall clocks and closed-form values.  Value-like names that are exact in *one* table only
#: (the closed-form surface tables) are **not** here but in :data:`TABLE_EXACT`, so that the
#: same name in a Monte Carlo table (``iv`` of ``forward_smile``) is still walked.
#: ``mean_abs_L_minus_1`` is dropped from the store's set: it is a particle-method summary
#: recorded upstream without a standard error, declared in :data:`UNPAIRED_UPSTREAM`
#: (``marking_fits``) and :data:`MARKING_UNPAIRED_PREFIXES` rather than called exact.
EXACT_COLUMNS: frozenset[str] = frozenset(
    (NON_MC_COLUMNS - {"mean_abs_L_minus_1"})
    | {
        "scale",
        "rota",
        "n_dates",
        "n_paths_pricing",
        "n_paths_world",
        "n_refits",
        "refits_at_bound",
        "refits_fallback",  # refits that took the guarded fallback (counts, study C)
        "refits_capped",  # refits whose target correlation hit REFIT_CORRELATION_CAP
        "calibrations",
        "cache_keys_touched",
        "cache_hits",
        "n",
        "n_paths",
        "wall_s",
        "rank",
        "std_rank",  # study D: rank of the hedged std among the ok regime rows
        "distance_rank",  # study D: rank of |distance to the common MV delta|
        "mv_common_rows",  # study D: how many regime rows entered the common MV delta
        "date",
        "t",
        "k",
        "z",  # a z-score: (value - prediction) / the stderr of the difference
        "target_vol",  # the SSVI target of the calibration error map (closed form)
        "window",  # the rolling-window length (days) of the raw-slice discriminator
        "n_points",
        "n_runs",
        "skew_weight",
        "ssr_requested",
        "ssr_first_target",
        "ssr_fitted",
        "objective_first",
        "objective_first_ssr",
        "objective_first_skew",
        "objective_second",
        "mean_skew_gap",
        "max_skew_ratio",
        "fit_seconds",
        "stage3_seconds",
        "T_T_s",
        "T_T_l",
        "skew_gap_T_s",
        "skew_gap_T_l",
    }
)
#: Exact columns of one table only, passed to :func:`columns_without_stderr` as ``exact`` by the
#: caller that walks that table: the closed-form tables of :class:`SurfaceRecord` (SSVI implied
#: vols, Dupire local vols, forwards and the numeric ATM skew of the target surface — built, not
#: simulated).  The same names elsewhere (``iv`` of the stored ``forward_smile``) are Monte Carlo
#: and are walked.
TABLE_EXACT: dict[str, frozenset[str]] = {
    "surface_implied_vol": frozenset({"iv", "forward"}),
    "surface_local_vol": frozenset({"local_vol"}),
    "surface_atm": frozenset({"atm_vol", "atm_skew", "forward"}),
}
#: Prefixes of exact columns (pattern-named): naked / first-order / market quantities of the
#: M7 tables, the skew gaps and the closed-form 90/110 skews of the M8b rotated worlds.
EXACT_PREFIXES: tuple[str, ...] = (
    "skew_gap_",
    "short_gap@",
    "ssr_naked",
    "skew_market_",
    "skew_naked_",
    "skew_6m_",  # volsto.hedging.worlds.skew_90_110 on the base / rotated SSVI (closed form)
    "ssr_achieved_naked",
    "ssr_first_order",
    "wall_",
    # first-order targets, bounds and clamped fits of the M7 binding maps
    "ssr_floor",
    "ssr_ceiling",
    "clamped_",
    "svc_target",
    "svc_model_fo",
    "volvar_target",
    "volvar_model_fo",
    "tracking_segments",
    "strict_tracking",
    "stage3_pricing_paths",
)
#: Unpaired-by-prefix Monte Carlo columns of the M7 binding maps (``soft_skew`` /
#: ``skew_tradeoff``): stage-3 means recorded without their se (``ssr_achieved_lsv_mean``,
#: ``clamped_ssr_achieved_mean``, ``ssr_sim_breakeven_<T>``, ``mean_skew_gap_mixing``, the
#: ``mean_abs_L_minus_1`` variants); the paired ``ssr_lsv_num_<T>`` and ``skew_naked_mixing_<T>``
#: columns carry theirs.
MARKING_UNPAIRED_PREFIXES: tuple[str, ...] = (
    "ssr_achieved_lsv",
    "clamped_ssr_achieved",
    "ssr_sim_breakeven_",
    "mean_skew_gap_mixing",
    "mean_abs_L",
)
#: The unpaired stage-3 SSR means of the binding maps, ``ssr_lsv_<T>`` — matched on a **digit**
#: after the prefix so that the paired ``ssr_lsv_num_<T>`` (and its ``ssr_lsv_num_z_<T>``) are
#: *not* exempted from the walk (review:S2 F5).
MARKING_UNPAIRED_PATTERNS: tuple[re.Pattern[str], ...] = (re.compile(r"^ssr_lsv_\d"),)
#: Monte Carlo columns the upstream scripts report **without** a standard error, per file (an
#: open item for those runners; the API passes them through and names them here rather than
#: padding a NaN).  M7 ``p1_marking_fits.csv``: the stage-3 engine-bias maxima and the reported
#: gap maxima (maxima over pillars of ratios whose stderr is in the per-fit YAML), the
#: particle-method leverage summary ``mean_abs_L_minus_1`` and the stage-3 SVC / vol-of-var
#: error maxima and correlation means (``svc_err_max``, ``volvar_err_max``, ``corr_model_mean``,
#: ``corr_target_mean``) — all recorded upstream without a standard error (review:PC F10);
#: M8b task results: the per-regime ``std`` and mean realised vol (the M8b runner stores the
#: regime mean with its stderr only); the M7 binding maps: :data:`MARKING_UNPAIRED_PREFIXES`
#: and :data:`MARKING_UNPAIRED_PATTERNS`.
UNPAIRED_UPSTREAM: dict[str, frozenset[str]] = {
    "marking_fits": frozenset(
        {
            "stage3_max_engine_bias_svc",
            "stage3_max_engine_bias_volvar",
            "stage3_max_gap_svc_reported",
            "stage3_max_gap_volvar_reported",
            "mean_abs_L_minus_1",
            "svc_err_max",
            "volvar_err_max",
            "corr_model_mean",
            "corr_target_mean",
        }
    ),
    "hedging_regimes": frozenset({"std", "realised_vol_mean"}),
    #: ``m8b_table_D.csv``: the spread (largest minus smallest) of the four regimes' implied
    #: minimum-variance deltas behind ``mv_common`` — a dispersion diagnostic of Monte Carlo
    #: values, reported beside ``mv_common`` ± its se, without an se of its own.
    "hedging_table_D": frozenset({"mv_common_spread"}),
    #: ``discriminator.csv`` (the M8b raw-slice gate): the regression diagnostics of the SSR
    #: fit per window — the runner reports the fitted SSR with its se but not these.
    "hedging_discriminator": frozenset(
        {"slope_raw", "slope_ssvi", "mean_skew_raw", "mean_skew_ssvi", "r2_raw", "r2_ssvi"}
    ),
}
#: ``discriminator.csv`` writes its standard errors as a ``se_<stem>`` **prefix** (the only M8b
#: file that does): the explicit pairing, applied by :func:`get_hedging_table`.
DISCRIMINATOR_STDERR: dict[str, str] = {
    "se_raw": "ssr_raw_stderr",
    "se_ssvi": "ssr_ssvi_stderr",
    "se_diff": "diff_stderr",
}
_SE_AT = re.compile(r"^(.*)_se@(.*)$")
_SE_T = re.compile(r"^(.*)_se_(\d[\d.]*)$")


# --------------------------------------------------------------------------------------------
# errors and commands
# --------------------------------------------------------------------------------------------


class MissingArtefact(FileNotFoundError):  # noqa: N818 — the owner's name
    """A stored result is absent; ``command`` produces it (never run by the viewer)."""

    def __init__(self, what: str, command: str) -> None:
        super().__init__(f"{what} is not available; produce it with: {command}")
        self.what = what
        self.command = command


class MissingPoint(MissingArtefact):
    """A grid point (or one of its tables / its leverage) is not in the store or the cache."""

    def __init__(self, key: str, command: str, what: str | None = None) -> None:
        super().__init__(what or f"point {key!r}", command)
        self.key = key


def grid_path_of(cfg: ViewerConfig) -> str:
    """The grid the printed command must name: the ``grid_path`` the store's run records
    recorded (a repository-relative path, :mod:`volsto.viewers.precompute`) when the store has
    one, else the configured ``cfg.grid_path``.  A store built from another grid (the toy grid,
    a VM grid) must not be refilled with the viewer's default grid (review:S2 F2)."""
    recorded = _reader(cfg).manifest().get("grid_path")
    if isinstance(recorded, str) and recorded:
        return recorded
    return cfg.relative_to_cwd(cfg.grid_path)


def precompute_command(
    cfg: ViewerConfig,
    *,
    only: str | None = None,
    risk: str | None = None,
    resume: bool = True,
) -> str:
    """The ``volsto-precompute`` line that produces what is missing at ``cfg`` (the contract in
    :mod:`volsto.viewers.precompute`'s docstring).  The grid is the one the store was built with
    (:func:`grid_path_of`).

    * ``only`` names one point id, so the line computes that point and nothing else
      (``--only <id>``); without it the line covers the whole grid;
    * ``resume`` (default) adds ``--resume``: stored points are skipped and a point stored below
      the requested risk tier is refreshed for its risk step only — the cheap line for a
      *missing* point;
    * ``risk`` adds ``--risk none|light|full``.  For a stored point whose risk rows are missing
      the API emits ``--only <id> --resume --risk light``: ``--resume`` is what makes it the
      **risk-only refresh** of :func:`volsto.viewers.precompute.pending_steps` (the stored
      pricing and analytics are reused and only the risk step runs), so the cheap line and the
      correct one are the same line.
    """
    cmd = (
        f"volsto-precompute --grid {grid_path_of(cfg)} "
        f"--store {cfg.relative_to_cwd(cfg.store_root)} "
        f"--cache {cfg.relative_to_cwd(cfg.cache_root)}"
    )
    if only is not None:
        cmd += f" --only {only}"
    if resume:
        cmd += " --resume"
    if risk is not None:
        cmd += f" --risk {risk}"
    return cmd


def _reader(cfg: ViewerConfig) -> StoreReader:
    return StoreReader(cfg.store_root)


# --------------------------------------------------------------------------------------------
# standard-error schema
# --------------------------------------------------------------------------------------------


def is_exact_column(name: str, exact: frozenset[str] = frozenset()) -> bool:
    """Exact (no stderr) by declaration: ``exact`` (the caller's per-table set,
    :data:`TABLE_EXACT`), :data:`EXACT_COLUMNS`, :data:`EXACT_PREFIXES`, the ``has_*`` flags and
    the ``*_z_*`` columns (ratios of a paired value to its own stderr)."""
    if name in exact or name in EXACT_COLUMNS:
        return True
    if name.startswith(EXACT_PREFIXES) or name.startswith("has_"):
        return True
    return "_z_" in name or name.endswith("_z")


def columns_without_stderr(
    df: pd.DataFrame,
    *,
    unpaired: frozenset[str] = frozenset(),
    exact: frozenset[str] = frozenset(),
) -> list[str]:
    """Numeric, non-boolean columns that are Monte Carlo quantities (not exact by global or
    per-table (``exact``) declaration, not a ``_stderr`` themselves, not in ``unpaired``)
    lacking their ``<name>_stderr`` twin.

    The twin is required **whatever the values**: a column whose Monte Carlo estimates all
    failed (all NaN) is still a Monte Carlo column (review:S2 F9).  A text column of a CSV that
    holds no value at all is read as ``object`` by :func:`read_study_csv`, not as an all-NaN
    float, so it never reaches this walk as a number."""
    bad: list[str] = []
    for c in df.columns:
        name = str(c)
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        if name.endswith("_stderr") or name in unpaired or is_exact_column(name, exact):
            continue
        if f"{name}_stderr" not in df.columns:
            bad.append(name)
    return bad


def read_study_csv(path: Path) -> pd.DataFrame:
    """Read an M7 / M8b CSV with :func:`normalise_stderr_names` applied and every column that is
    **empty in the file** (no character in any cell: the runners' unfilled ``reason`` /
    ``message`` / ``bounds`` text columns, which pandas otherwise types ``float64`` NaN) kept as
    an ``object`` column, so that :func:`columns_without_stderr` walks numbers only.  A CSV
    written by pandas encodes a computed NaN as an empty cell too, so a numeric column that is
    NaN in *every* row of the file cannot be told apart from an empty text column — the M7 /
    M8b runners would have to write it explicitly (an upstream open item)."""
    df = pd.read_csv(path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)  # the cells as written
    empty = [
        c
        for c in df.columns
        if df[c].isna().all() and c in raw.columns and bool((raw[c] == "").all())
    ]
    if empty:
        df = df.astype({c: "object" for c in empty})
    return normalise_stderr_names(df)


def normalise_stderr_names(df: pd.DataFrame) -> pd.DataFrame:
    """``x_se`` → ``x_stderr``, ``x_se@T`` → ``x@T_stderr`` and ``x_se_T`` → ``x_T_stderr`` (the
    M7 / M8b CSV conventions)."""
    ren: dict[str, str] = {}
    for c in df.columns:
        name = str(c)
        m = _SE_AT.match(name)
        m2 = _SE_T.match(name)
        if m:
            ren[name] = f"{m.group(1)}@{m.group(2)}_stderr"
        elif m2:
            ren[name] = f"{m2.group(1)}_{m2.group(2)}_stderr"
        elif name.endswith("_se"):
            ren[name] = name[:-3] + "_stderr"
    return df.rename(columns=ren)


# --------------------------------------------------------------------------------------------
# grid and points
# --------------------------------------------------------------------------------------------

GRID_COLUMNS: tuple[str, ...] = (
    "id",
    "label",
    "surface",
    "mode",
    "status",
    *MODEL_PARAMS,
    *AXES,
    "n_particles",
    "horizon",
    "cache_key",
    "has_leverage",
    *(f"has_{t}" for t in POINT_TABLES),
    "risk_tier",
    "code_tag",
    "git_commit",
    "created_utc",
)


def list_grid(cfg: ViewerConfig) -> pd.DataFrame:
    """One row per stored point (:data:`GRID_COLUMNS`); empty with those columns when the store
    is empty.  ``has_leverage`` is a file-existence check in the cache (no calibration)."""
    reader = _reader(cfg)
    pts = reader.points()
    if pts.empty:
        return pd.DataFrame(columns=list(GRID_COLUMNS))
    present = {
        t: set(reader.store.table(t).get("point_id", pd.Series(dtype=str))) for t in POINT_TABLES
    }
    rows: list[dict[str, Any]] = []
    for _, r in pts.iterrows():
        pid = str(r["point_id"])
        key = str(r.get("cache_key", "") or "")
        row: dict[str, Any] = {
            "id": pid,
            "label": r.get("label", pid),
            "surface": r.get("surface", ""),
            "mode": r.get("mode", ""),
            "status": r.get("status", ""),
        }
        for c in (*MODEL_PARAMS, *AXES):
            row[c] = float(r[c]) if c in r.index and pd.notna(r[c]) else math.nan
        row["n_particles"] = int(r.get("n_particles", 0) or 0)
        row["horizon"] = float(r.get("horizon", math.nan))
        row["cache_key"] = key
        # the cache's own completeness test: a torn leverage.npz is not a leverage
        row["has_leverage"] = bool(key) and has_complete_leverage(cfg.cache_root, key)
        for t in POINT_TABLES:
            row[f"has_{t}"] = pid in present[t]
        for c in ("risk_tier", "code_tag", "git_commit", "created_utc"):
            row[c] = str(r.get(c, "") or "")
        rows.append(row)
    out = pd.DataFrame(rows, columns=list(GRID_COLUMNS))
    return out.sort_values(["surface", "mode", "label"], kind="stable").reset_index(drop=True)


@dataclass(frozen=True)
class PointRecord:
    """One stored grid point.  ``params`` are the exact model parameters (empty for the LV
    point), ``axes`` the grid position, ``diagnostics`` the calibration summary (particle count,
    horizon, cache key, calibration wall time, ``mean_abs_L_minus_1``, ``max_abs_error_vp``,
    ``max_z``, ``calibrated_this_run``), ``fit`` the marking-fit record (status, messages, skew
    gaps) and ``provenance`` the code tag, git commit, pricing paths / seed, risk tier and
    creation time.  The frames are the store's tables without the ``point_id`` column."""

    id: str
    label: str
    surface: str
    mode: str
    status: str
    params: dict[str, float]
    axes: dict[str, float]
    diagnostics: dict[str, Any]
    fit: dict[str, Any]
    provenance: dict[str, Any]
    products: pd.DataFrame
    forward_smile: pd.DataFrame
    forward_vols: pd.DataFrame
    ssr: pd.DataFrame
    varv: pd.DataFrame

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "products": self.products,
            "forward_smile": self.forward_smile,
            "forward_vols": self.forward_vols,
            "ssr": self.ssr,
            "varv": self.varv,
        }


def _drop_point_id(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=["point_id"], errors="ignore").reset_index(drop=True)


def get_point(cfg: ViewerConfig, key: str) -> PointRecord:
    """The point ``key`` (a store ``point_id``) or :class:`MissingPoint`."""
    reader = _reader(cfg)
    try:
        r = reader.point(key)
    except KeyError:
        raise MissingPoint(key, precompute_command(cfg, only=key)) from None
    params = {
        p: float(r[p])
        for p in MODEL_PARAMS
        if p in r and isinstance(r[p], float) and pd.notna(r[p])
    }
    axes = {a: float(r[a]) for a in AXES if a in r and pd.notna(r[a])}
    diag_keys = (
        "n_particles",
        "horizon",
        "cache_key",
        "calibrated_this_run",
        "calibration_seconds",
        "mean_abs_L_minus_1",
        "max_abs_error_vp",
        "max_z",
        "wall_seconds",
        "wall_calibration_s",
        "wall_pricing_s",
        "wall_analytics_s",
        "wall_risk_s",
    )
    diagnostics = {k: _py(r[k]) for k in diag_keys if k in r}
    fit = {k: _py(v) for k, v in r.items() if k.startswith(("fit_", "skew_gap_"))}
    prov_keys = (
        "code_tag",
        "git_commit",
        "created_utc",
        "pricing_n_paths",
        "pricing_seed",
        "risk_tier",
    )
    provenance = {k: _py(r[k]) for k in prov_keys if k in r}
    return PointRecord(
        key,
        str(r.get("label", key)),
        str(r.get("surface", "")),
        str(r.get("mode", "")),
        str(r.get("status", "")),
        params,
        axes,
        diagnostics,
        fit,
        provenance,
        _drop_point_id(reader.products(key)),
        _drop_point_id(reader.forward_smile(key)),
        _drop_point_id(reader.forward_vols(key)),
        _drop_point_id(reader.ssr(key)),
        _drop_point_id(reader.varv(key)),
    )


def _py(v: Any) -> Any:
    if isinstance(v, np.generic):
        return v.item()
    return v


# --------------------------------------------------------------------------------------------
# surfaces
# --------------------------------------------------------------------------------------------


def _grid_of(cfg: ViewerConfig) -> tuple[GridSpec, str]:
    """The grid the store was computed with (its manifest's mapping), else the configured grid
    YAML; returns the grid and its source."""
    man = _reader(cfg).manifest()
    mapping = man.get("grid")
    if isinstance(mapping, dict) and mapping:
        return from_mapping(GridSpec, mapping), "store manifest"
    return load_grid(cfg.grid_path), str(cfg.grid_path)


def list_surfaces(cfg: ViewerConfig) -> pd.DataFrame:
    """``name, kind, path, source, n_points`` of every surface known to the grid (store manifest
    first, then the grid YAML) plus any surface named by the store but absent from the grid
    (``kind = "unknown"``) and the snapshot files of ``snapshots_root`` not yet on the grid
    (``kind = "snapshot"``, ``source = "snapshots_root"``; the top level only — the history
    sub-directories hold hundreds of daily fits that are not grid surfaces)."""
    grid, source = _grid_of(cfg)
    pts = _reader(cfg).points()
    counts: dict[str, int] = (
        {str(k): int(v) for k, v in pts["surface"].value_counts().items()} if not pts.empty else {}
    )
    rows: list[dict[str, Any]] = []
    known: set[str] = set()
    for s in grid.surfaces:
        known.add(s.name)
        rows.append(
            {
                "name": s.name,
                "kind": s.kind,
                "path": s.path or "",
                "source": source,
                "n_points": int(counts.get(s.name, 0)),
            }
        )
    for name in sorted(set(counts) - known):
        rows.append(
            {
                "name": name,
                "kind": "unknown",
                "path": "",
                "source": "store",
                "n_points": int(counts[name]),
            }
        )
        known.add(name)
    if (
        cfg.snapshots_root.exists()
    ):  # top level only: the history sub-directories are not grid surfaces
        for p in sorted(cfg.snapshots_root.glob("*.yaml")):
            if p.stem not in known:
                rel = p.relative_to(REPO_ROOT) if p.is_relative_to(REPO_ROOT) else p
                rows.append(
                    {
                        "name": p.stem,
                        "kind": "snapshot",
                        "path": str(rel),
                        "source": "snapshots_root",
                        "n_points": 0,
                    }
                )
    return pd.DataFrame(rows, columns=["name", "kind", "path", "source", "n_points"])


@dataclass(frozen=True)
class SurfaceRecord:
    """A target surface: ``spec`` (market, SSVI and local-vol settings with the LV placeholder
    model), the :class:`~volsto.market.surface.ImpliedSurface`, its forward curve and the Dupire
    :class:`~volsto.market.dupire.LocalVolSurface` (built once, no simulation).  The tables are
    closed-form values — exact, no standard error."""

    name: str
    kind: str
    path: str
    spec: CalibrationSpec
    surface: ImpliedSurface
    forward_curve: ForwardCurve
    local_vol: LocalVolSurface

    @property
    def spot(self) -> float:
        return float(self.forward_curve.spot)

    def implied_vol_table(
        self, maturities: Sequence[float], log_moneyness: Sequence[float]
    ) -> pd.DataFrame:
        """Long frame ``T, k, iv, forward`` of the target implied vol (``k = ln K / F``)."""
        T = np.asarray(maturities, dtype=float)
        k = np.asarray(log_moneyness, dtype=float)
        iv = self.surface.implied_vol_k(k[None, :], T[:, None])
        fwd = self.surface.forward(T)
        rows = [
            {"T": float(T[i]), "k": float(k[j]), "iv": float(iv[i, j]), "forward": float(fwd[i])}
            for i in range(T.size)
            for j in range(k.size)
        ]
        return pd.DataFrame(rows, columns=["T", "k", "iv", "forward"])

    def local_vol_table(
        self, times: Sequence[float], log_moneyness: Sequence[float]
    ) -> pd.DataFrame:
        """Long frame ``t, k, local_vol`` of the Dupire local vol (``k = ln S / F(t)``)."""
        t = np.asarray(times, dtype=float)
        k = np.asarray(log_moneyness, dtype=float)
        tt, kk = np.meshgrid(t, k, indexing="ij")
        lv = self.local_vol.local_vol_k(tt.ravel(), kk.ravel())
        return pd.DataFrame({"t": tt.ravel(), "k": kk.ravel(), "local_vol": lv})

    def atm_table(self, maturities: Sequence[float]) -> pd.DataFrame:
        """``T, atm_vol, atm_skew, forward`` of the target surface."""
        T = np.asarray(maturities, dtype=float)
        return pd.DataFrame(
            {
                "T": T,
                "atm_vol": self.surface.atm_vol(T),
                "atm_skew": atm_skew_numeric(self.surface, T),
                "forward": self.surface.forward(T),
            }
        )


def _snapshot_order(p: Path) -> tuple[int, str]:
    """Sort key of an off-grid snapshot candidate: the shallowest path first (the top-level file
    of ``snapshots_root`` before a same-named daily fit under ``history/``), then by name."""
    return (len(p.parts), str(p))


def get_surface(cfg: ViewerConfig, name: str) -> SurfaceRecord:
    """Build the surface ``name`` of the grid (or a snapshot file ``<snapshots_root>/<name>.yaml``
    not yet on the grid) — pure construction: SSVI from its config, Dupire from the SSVI.  The
    snapshot lookup is deterministic: the top-level file of ``snapshots_root`` wins over a file
    of the same name in a history sub-directory (:func:`list_surfaces` lists the top level
    only).  An unknown surface is not a missing computation — the command says to add it to the
    grid YAML first (review:S2 F10)."""
    grid, _ = _grid_of(cfg)
    ref = reference_spec(grid)
    try:
        sspec = grid.surface(name)
    except KeyError:
        candidates = (
            sorted(cfg.snapshots_root.rglob(f"{name}.yaml"), key=_snapshot_order)
            if cfg.snapshots_root.exists()
            else []
        )
        if not candidates:
            known = [s.name for s in grid.surfaces]
            raise MissingArtefact(
                f"surface {name!r} (known: {known}; snapshots under {cfg.snapshots_root})",
                f"add the surface to {grid_path_of(cfg)} (a 'surfaces:' entry naming its "
                f"snapshot YAML under {cfg.snapshots_root}), then {precompute_command(cfg)}",
            ) from None
        rel = (
            candidates[0].relative_to(REPO_ROOT)
            if candidates[0].is_relative_to(REPO_ROOT)
            else candidates[0]
        )
        sspec = SurfaceSpec(name, "snapshot", False, False, False, str(rel))
    spec = surface_spec(grid, sspec, ref)
    fc, surface, _ = build_market(spec)
    local_vol = LocalVolSurface.from_implied(surface, spec.local_vol)
    return SurfaceRecord(name, sspec.kind, sspec.path or "", spec, surface, fc, local_vol)


# --------------------------------------------------------------------------------------------
# leverage
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LeverageRecord:
    """The cached leverage of a point: ``leverage`` (``times × k_grid`` values), its provenance
    ``metadata`` (cache key, wall time, code tag, git commit), the calibration ``report`` when the
    cache holds one and its ``error_map`` (``T, k, cp, target_vol, model_vol, model_vol_stderr,
    error_vp, error_vp_stderr, price, price_stderr``; empty without a report).  ``target_vol`` is
    the closed-form SSVI target (exact); ``model_vol_stderr`` is the recorded ``stderr_vp`` in
    vol units (see :func:`get_leverage`)."""

    point_id: str
    cache_key: str
    leverage: LeverageFunction
    metadata: dict[str, Any]
    report: CalibrationReport | None
    error_map: pd.DataFrame

    def leverage_table(self, k_abs: float = 1.0) -> pd.DataFrame:
        """Wide frame (rows ``t``, columns ``k``) of ``L(t, k)`` for ``|k| ≤ k_abs``."""
        sel = np.abs(self.leverage.k_grid) <= k_abs + 1e-12
        return pd.DataFrame(
            self.leverage.values[:, sel],
            index=pd.Index(self.leverage.times, name="t"),
            columns=pd.Index(self.leverage.k_grid[sel], name="k"),
        )


def get_leverage(cfg: ViewerConfig, key: str) -> LeverageRecord:
    """Load ``<cache>/<cache_key>/leverage.npz`` (+ ``diagnostics.json``) of the point ``key``;
    :class:`MissingPoint` when the point or its cache entry is absent — never calibrates."""
    rec = get_point(cfg, key)
    cache_key = str(rec.diagnostics.get("cache_key", "") or "")
    if not cache_key:
        raise MissingPoint(
            key,
            precompute_command(cfg, only=key),
            f"point {key!r} ({rec.mode}, {rec.status}) has no leverage",
        )
    entry = cfg.cache_root / cache_key
    npz = entry / "leverage.npz"
    if not has_complete_leverage(cfg.cache_root, cache_key):
        raise MissingPoint(
            key,
            precompute_command(cfg, only=key),
            f"leverage {cache_key[:12]}… of point {key!r} in {cfg.cache_root}",
        )
    lev = LeverageFunction.load(npz)
    diag = entry / "diagnostics.json"
    report = CalibrationReport.load(diag) if diag.exists() else None
    err_columns = (
        "T",
        "k",
        "cp",
        "target_vol",
        "model_vol",
        "model_vol_stderr",
        "error_vp",
        "error_vp_stderr",
        "price",
        "price_stderr",
    )
    if report is not None:
        err = report.vanillas.rename(columns={"stderr_vp": "error_vp_stderr"})
        if "model_vol" in err.columns and "error_vp_stderr" in err.columns:
            # error_vp = 100 (model_vol - target_vol) with a deterministic (SSVI) target_vol, so
            # the model vol's stderr is the error's in vol units — a unit change of a recorded
            # standard error, not a filled-in default.
            err["model_vol_stderr"] = err["error_vp_stderr"] / 100.0
        err = err[[c for c in err_columns if c in err.columns]].reset_index(drop=True)
    else:
        err = pd.DataFrame(columns=[c for c in err_columns if c != "cp"])
    return LeverageRecord(key, cache_key, lev, dict(lev.metadata), report, err)


# --------------------------------------------------------------------------------------------
# products
# --------------------------------------------------------------------------------------------

#: Term-sheet catalogue of the product classes (page 6's editor maps its fields onto these
#: constructors); ``store_products`` are the store's product names a class prices.
PRODUCT_CATALOGUE: tuple[dict[str, Any], ...] = (
    {
        "name": "European option",
        "class": "volsto.products.vanilla.EuropeanOption",
        "unit": "price (spot units)",
        "fields": "strike, T, cp (+1 call / -1 put)",
        "store_products": "",
    },
    {
        "name": "Digital option",
        "class": "volsto.products.vanilla.DigitalOption",
        "unit": "price",
        "fields": "strike, T, cp",
        "store_products": "",
    },
    {
        "name": "Additive cliquet",
        "class": "volsto.products.cliquet.AdditiveCliquet",
        "unit": "% notional",
        "fields": (
            "T, periods_per_year (12), local_cap (0.02), local_floor, global_floor (0), "
            "global_cap"
        ),
        "store_products": "cliquet 1y, cliquet 2y",
    },
    {
        "name": "Vol knock-out put",
        "class": "volsto.products.vko.VolKnockOutPut",
        "unit": "% notional",
        "fields": "strike, maturity, vol_ko (0.30), daily fixings (252/y), daily_cap, knock_in",
        "store_products": "VKO 12m 100% put @30%",
    },
    {
        "name": "Knock-out variance swap",
        "class": "volsto.products.conditional_variance.KnockOutVarianceSwap",
        "unit": "fair vol",
        "fields": "fixing_times (daily 1y), barrier (1.1 spot), strike_vol",
        "store_products": "KO var 1y B=110%",
    },
    {
        "name": "Up / down variance swap",
        "class": "volsto.products.conditional_variance.ConditionalVarianceSwap",
        "unit": "fair vol",
        "fields": "fixing_times, barrier, side (up / down), indicator, convention, strike_vol",
        "store_products": "up-var 1y B=100%, down-var 1y B=100%",
    },
    {
        "name": "Autocall",
        "class": "volsto.products.autocall.Autocall",
        "unit": "% notional",
        "fields": (
            "observation_times (annual 3y), coupons (6%), autocall_barriers (1.0), ki_level "
            "(0.6), ki_type (european), memory, non_call_periods"
        ),
        "store_products": "autocall 3y",
    },
    {
        "name": "Phoenix",
        "class": "volsto.products.autocall.Phoenix",
        "unit": "% notional",
        "fields": (
            "observation_times, coupons (6%), coupon_barrier (0.7), memory (true), ki_level "
            "(0.6), ki_type (american daily)"
        ),
        "store_products": "phoenix 3y",
    },
    {
        "name": "Knock-out option",
        "class": "volsto.products.barrier.KnockOutOption",
        "unit": "price",
        "fields": (
            "strike, maturity, cp, barrier, direction (up / down), monitoring (continuous / "
            "discrete), rebate"
        ),
        "store_products": "",
    },
    {
        "name": "Knock-in option",
        "class": "volsto.products.barrier.KnockInOption",
        "unit": "price",
        "fields": "strike, maturity, cp, barrier, direction, monitoring, rebate",
        "store_products": "",
    },
    {
        "name": "Forward-start option",
        "class": "volsto.products.forward_start.ForwardStartOption",
        "unit": "implied vol",
        "fields": "t1, t2, strike_moneyness, cp",
        "store_products": "fwd ATM vol 1y→2y",
    },
)


def get_products(cfg: ViewerConfig) -> pd.DataFrame:
    """The catalogue (:data:`PRODUCT_CATALOGUE`) with ``in_store`` (a store product of the class
    is priced somewhere) and ``n_points`` (points holding one)."""
    prods = _reader(cfg).products()
    per_product: dict[str, int] = {}
    if not prods.empty:
        per_product = {
            str(k): int(v) for k, v in prods.groupby("product")["point_id"].nunique().items()
        }
    rows = []
    for entry in PRODUCT_CATALOGUE:
        names = [s.strip() for s in str(entry["store_products"]).split(",") if s.strip()]
        n = max((per_product.get(nm, 0) for nm in names), default=0)
        rows.append({**entry, "in_store": n > 0, "n_points": int(n)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# risk
# --------------------------------------------------------------------------------------------

RISK_COLUMNS: tuple[str, ...] = (
    "section",
    "name",
    "T",
    "bucket",
    "regime",
    "variant",
    "value",
    "value_stderr",
    "unit",
    "tier",
    "size",
    "scheme",
    "states",
    "key",
)


def get_risk(cfg: ViewerConfig, key: str, product: str) -> pd.DataFrame:
    """The stored RiskReport rows of ``product`` at point ``key`` (:data:`RISK_COLUMNS`);
    :class:`MissingPoint` names the per-point ``--only <id> --risk light`` command when the
    point is stored without risk rows — the point is already in the store, so a whole-grid
    ``--resume`` would only refresh it at the requested tier; the per-point line recomputes
    exactly that point and is what the message states (review:S2 F3)."""
    reader = _reader(cfg)
    if key not in set(reader.points().get("point_id", pd.Series(dtype=str))):
        raise MissingPoint(key, precompute_command(cfg, only=key))
    df = reader.risk(key, product)
    if df.empty:
        have = sorted(set(reader.risk(key).get("product", pd.Series(dtype=str))))
        raise MissingPoint(
            key,
            precompute_command(cfg, only=key, risk="light"),
            f"risk rows of {product!r} at point {key!r} (stored products: {have}); the point is "
            f"stored, so the command below recomputes this point alone at the light risk tier",
        )
    out = df.rename(columns={"group": "section"})
    return out[[c for c in RISK_COLUMNS if c in out.columns]].reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# hedging (M8b task results)
# --------------------------------------------------------------------------------------------


def _m8b_root(cfg: ViewerConfig) -> Path:
    return cfg.outputs_root / "m8b"


def _task_files(cfg: ViewerConfig) -> list[Path]:
    """The stored task results: ``<outputs>/m8b/<study>/<study>__....json``.  The key convention
    of :mod:`volsto.studies.m8b` (``<study>__<product>__...``) is the filter — the study
    directories also hold the runner's own caches (``C/static_<product>__<policy>.json``, the M7
    rotation greek), which are not task results and must not be listed as runs."""
    root = _m8b_root(cfg)
    if not root.exists():
        return []
    return sorted(p for p in root.glob("*/*.json") if p.stem.startswith(f"{p.parent.name}__"))


def _load_task(path: Path) -> TaskResult:
    return TaskResult.from_dict(json.loads(path.read_text()))


HEDGING_RUN_COLUMNS: tuple[str, ...] = (
    "run_id",
    "study",
    "product",
    "world",
    "strategy",
    "pricing",
    "rota",
    "policy",
    "regime",
    "frequency",
    "status",
    "reason",
    "unit",
    "n_dates",
    "n_paths_pricing",
    "n_paths_world",
    "n_particles",
    "value_0",
    "value_0_stderr",
    "mean",
    "mean_stderr",
    "std",
    "std_stderr",
    "recal_total",
    "recal_total_stderr",
    "n_refits",
    "calibrations",
    "wall_seconds",
    "has_paths",
)


def list_hedging_runs(cfg: ViewerConfig) -> pd.DataFrame:
    """One row per stored M8b task result (``<outputs>/m8b/<study>/<key>.json``;
    :data:`HEDGING_RUN_COLUMNS`); ``has_paths`` says whether the ``.pkl`` with the per-path
    P&L sits beside it.  Numbers are in the hedger's sign (long the product) and unit."""
    rows: list[dict[str, Any]] = []
    for p in _task_files(cfg):
        t = _load_task(p)
        rows.append(
            {
                "run_id": t.key,
                "study": t.study,
                "product": t.product,
                "world": t.world,
                "strategy": t.strategy,
                "pricing": t.pricing,
                "rota": math.nan if t.rota is None else float(t.rota),
                "policy": t.policy or "",
                "regime": t.regime or "",
                "frequency": t.frequency,
                "status": t.status,
                "reason": t.reason,
                "unit": t.unit,
                "n_dates": t.n_dates,
                "n_paths_pricing": t.n_paths_pricing,
                "n_paths_world": t.n_paths_world,
                "n_particles": t.n_particles,
                "value_0": t.value_0[0],
                "value_0_stderr": t.value_0[1],
                "mean": t.mean[0],
                "mean_stderr": t.mean[1],
                "std": t.std[0],
                "std_stderr": t.std[1],
                "recal_total": t.recal_total[0],
                "recal_total_stderr": t.recal_total[1],
                "n_refits": t.n_refits,
                "calibrations": t.calibrations,
                "wall_seconds": t.wall_seconds,
                "has_paths": p.with_suffix(".pkl").exists(),
            }
        )
    return pd.DataFrame(rows, columns=list(HEDGING_RUN_COLUMNS))


@dataclass(frozen=True)
class HedgingRunRecord:
    """A stored M8b run: ``meta`` (study, product, world, strategy, pricing, rota, policy,
    regime, frequency, status, unit, counts, wall clock), ``settings`` and ``budget`` as stored,
    ``distribution`` (``statistic, value, value_stderr``: value_0, mean, std, the quantiles,
    zero-cost mean, recalibration total, world value, static spread), ``regimes`` (regime, n,
    mean, mean_stderr, std, realised_vol_mean — see :data:`UNPAIRED_UPSTREAM`), ``attribution``
    (component, value, value_stderr), ``recalibration`` (per refit date, as stored) and the
    ``notes``.  Values are in the hedger's sign (long the product); ``desk`` P&L is the negative
    (:func:`volsto.studies.m8b.to_desk_pnl`)."""

    run_id: str
    meta: dict[str, Any]
    settings: dict[str, Any]
    budget: dict[str, float]
    distribution: pd.DataFrame
    regimes: pd.DataFrame
    attribution: pd.DataFrame
    recalibration: pd.DataFrame
    notes: tuple[str, ...]
    has_paths: bool

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "distribution": self.distribution,
            "regimes": self.regimes,
            "attribution": self.attribution,
            "recalibration": self.recalibration,
        }


def _task_path(cfg: ViewerConfig, run_id: str) -> Path:
    study = run_id.split("__", 1)[0]
    p = _m8b_root(cfg) / study / f"{run_id}.json"
    if not p.exists():
        raise MissingArtefact(
            f"hedging run {run_id!r} under {_m8b_root(cfg)}", HEDGING_SCRIPT.format(study=study)
        )
    return p


def get_hedging_run(cfg: ViewerConfig, run_id: str) -> HedgingRunRecord:
    """The stored task result ``run_id`` (``<study>__...`` keys of :mod:`volsto.studies.m8b`)."""
    p = _task_path(cfg, run_id)
    t = _load_task(p)
    dist: list[dict[str, Any]] = [
        {"statistic": "value_0", "value": t.value_0[0], "value_stderr": t.value_0[1]},
        {"statistic": "mean", "value": t.mean[0], "value_stderr": t.mean[1]},
        {"statistic": "std", "value": t.std[0], "value_stderr": t.std[1]},
    ]
    for q, (v, se) in t.quantiles.items():
        dist.append({"statistic": q, "value": v, "value_stderr": se})
    dist.append(
        {
            "statistic": "zero_cost_mean",
            "value": t.zero_cost_mean[0],
            "value_stderr": t.zero_cost_mean[1],
        }
    )
    dist.append(
        {"statistic": "recal_total", "value": t.recal_total[0], "value_stderr": t.recal_total[1]}
    )
    if t.world_value_0 is not None:
        dist.append(
            {
                "statistic": "world_value_0",
                "value": t.world_value_0[0],
                "value_stderr": t.world_value_0[1],
            }
        )
    if t.static_spread is not None:
        dist.append(
            {
                "statistic": "static_spread",
                "value": t.static_spread[0],
                "value_stderr": t.static_spread[1],
            }
        )
    distribution = pd.DataFrame(dist, columns=["statistic", "value", "value_stderr"])
    distribution["unit"] = t.unit
    regimes = pd.DataFrame(t.regimes).rename(columns={"stderr": "mean_stderr"})
    attribution = pd.DataFrame(t.attribution).rename(
        columns={"mean": "value", "stderr": "value_stderr"}
    )
    recal = normalise_stderr_names(pd.DataFrame(t.recal_by_date))
    meta = {
        k: getattr(t, k)
        for k in (
            "study",
            "product",
            "world",
            "strategy",
            "pricing",
            "rota",
            "policy",
            "regime",
            "frequency",
            "status",
            "reason",
            "unit",
            "scale",
            "n_paths_pricing",
            "n_paths_world",
            "n_particles",
            "n_dates",
            "n_refits",
            "n_refits_at_bound",
            "n_refits_fallback",
            "n_refits_capped",
            "calibrations",
            "cache_hits",
            "wall_seconds",
        )
    }
    meta["world_meta"] = dict(t.world_meta)
    return HedgingRunRecord(
        run_id,
        meta,
        dict(t.settings),
        {str(k): (math.nan if v is None else float(v)) for k, v in t.budget.items()},
        distribution,
        regimes,
        attribution,
        recal,
        tuple(t.notes),
        p.with_suffix(".pkl").exists(),
    )


def get_hedging_pnl(cfg: ViewerConfig, run_id: str) -> pd.DataFrame:
    """Per-path P&L samples of a run (``pnl_total, pnl_product, pnl_hedges, costs,
    pnl_recalibration, termination`` and one ``pnl_hedge:<instrument>`` per hedge; hedger's
    sign, hedger's value unit — multiply by ``meta["scale"]`` for the reporting unit) from the
    ``.pkl`` beside the JSON — the histogram data of page 8.  These are samples, not estimators
    (their summary with stderr is :attr:`HedgingRunRecord.distribution`)."""
    p = _task_path(cfg, run_id).with_suffix(".pkl")
    if not p.exists():
        raise MissingArtefact(
            f"per-path P&L of run {run_id!r} ({p.name})",
            HEDGING_SCRIPT.format(study=run_id.split("__", 1)[0]),
        )
    with p.open("rb") as fh:
        res = pickle.load(fh)  # our own HedgeResult pickles under outputs/
    hedges = np.asarray(res.pnl_hedges, dtype=float)
    if hedges.ndim == 1:
        hedges = hedges[:, None]
    data: dict[str, Any] = {
        "pnl_total": np.asarray(res.pnl_total, dtype=float),
        "pnl_product": np.asarray(res.pnl_product, dtype=float),
        "pnl_hedges": hedges.sum(axis=1),
        "costs": np.asarray(res.costs, dtype=float),
        "pnl_recalibration": np.asarray(res.pnl_recalibration, dtype=float),
        "termination": np.asarray(res.termination, dtype=float),
    }
    names = list(getattr(res, "instruments", ()))
    for j in range(hedges.shape[1]):
        data[f"pnl_hedge:{names[j] if j < len(names) else j}"] = hedges[:, j]
    return pd.DataFrame(data)


def reattach_stderr_names(df: pd.DataFrame) -> pd.DataFrame:
    """Give every orphan ``<stem>_stderr`` of an M8b table the name of the value column it
    belongs to.

    Since 2026-09-16 the M8b writers emit exact twins (``<value>_se``, see
    :func:`volsto.studies.m8b.table_A` … ``table_D``), so a table written today has no orphan.
    This re-attaches the tables written before, whichever desk-naming convention they used:

    * table A / D wrote ``desk_mean`` + ``mean_se`` (the desk sign as a *prefix*) → the twin
      becomes ``desk_mean_stderr``;
    * table B writes ``leakage_desk`` + ``leakage_desk_se`` (the desk sign as a *suffix*), which
      :func:`normalise_stderr_names` already pairs — nothing to do;
    * table C wrote ``recal_pnl_desk`` + ``recal_se`` (and ``total_pnl_desk`` + ``total_se``,
      ``static_prediction`` + ``static_se``): the stem is a *prefix of the value column*.

    A candidate value column that already has its own twin cannot own the orphan (table D's
    ``mean_delta`` next to ``mean_delta_se`` is a prefix match for ``mean_se`` but is not its
    value).  An orphan is re-attached only when exactly one numeric, non-stderr, twinless column
    matches, so an ambiguous stem is left alone and the stderr walk reports the unpaired value
    column instead of pairing it with the wrong twin (review:S2 F1)."""
    numeric = [
        str(c)
        for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
        and not str(c).endswith("_stderr")
    ]
    ren: dict[str, str] = {}
    for c in df.columns:
        name = str(c)
        if not name.endswith("_stderr"):
            continue
        stem = name[: -len("_stderr")]
        if stem in df.columns:
            continue
        hits = [
            v
            for v in numeric
            if (v == f"desk_{stem}" or v.startswith(f"{stem}_")) and f"{v}_stderr" not in df.columns
        ]
        if len(hits) == 1:
            ren[name] = f"{hits[0]}_stderr"
    return df.rename(columns=ren)


def get_hedging_table(cfg: ViewerConfig, name: str) -> pd.DataFrame:
    """The M8b summary table ``A`` … ``D`` (``<outputs>/m8b/m8b_table_<name>.csv``) or the
    discriminator table, ``*_se`` columns renamed ``*_stderr`` (:func:`normalise_stderr_names`)
    and re-attached to their value column (:func:`reattach_stderr_names`; the discriminator's
    ``se_<stem>`` prefix convention through :data:`DISCRIMINATOR_STDERR`, its unpaired
    regression diagnostics in ``UNPAIRED_UPSTREAM["hedging_discriminator"]``); an empty frame
    for a table a study wrote without a single ``ok`` run."""
    fname = "discriminator.csv" if name == "discriminator" else f"m8b_table_{name}.csv"
    p = _m8b_root(cfg) / fname
    if not p.exists():
        raise MissingArtefact(
            f"hedging table {name!r} ({p})",
            HEDGING_SCRIPT.format(study=name if len(name) == 1 else "A"),
        )
    try:
        df = read_study_csv(p)
    except pd.errors.EmptyDataError:  # a study whose table was written empty (no ok run)
        return pd.DataFrame()
    if name == "discriminator":
        df = df.rename(columns={k: v for k, v in DISCRIMINATOR_STDERR.items() if k in df.columns})
    return reattach_stderr_names(df)


# --------------------------------------------------------------------------------------------
# marking (M7 Part 3)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkingRecord:
    """The M7 Part 3 tables: ``fits`` (one row per (surface, ssr_target, skew_eps): status,
    fitted parameters, naked-skew gaps, |L−1|, realised SSR with stderr), ``shadow_rotation``
    (surface, policy, greek, per-rota P&L with stderr, the P1 / LV levels and the fee),
    ``soft_skew`` and ``skew_tradeoff`` (the binding maps: ssr_target × skew weight / eps with
    status flags), the fit-spec YAML names under ``configs/studies/m7_p1_marking`` and the
    rotation ``convention`` string."""

    fits: pd.DataFrame
    shadow_rotation: pd.DataFrame
    soft_skew: pd.DataFrame
    skew_tradeoff: pd.DataFrame
    fit_specs: tuple[str, ...]
    convention: str
    source: str

    def tables(self) -> dict[str, pd.DataFrame]:
        return {
            "fits": self.fits,
            "shadow_rotation": self.shadow_rotation,
            "soft_skew": self.soft_skew,
            "skew_tradeoff": self.skew_tradeoff,
        }


def _scale_per_vp(df: pd.DataFrame) -> pd.DataFrame:
    """``per_vp_90_110_<T>`` is ``per_rota`` divided by the rota's vp move at ``T``: give it the
    same relative stderr as ``per_rota`` (an exact scaling)."""
    out = df.copy()
    if "per_rota_stderr" not in out.columns:
        return out
    for tag in ROTA_VP_PER_UNIT:
        col = f"per_vp_90_110_{tag}"
        if col in out.columns and f"{col}_stderr" not in out.columns:
            out[f"{col}_stderr"] = out["per_rota_stderr"] / ROTA_VP_PER_UNIT[tag]
    return out


def get_marking(cfg: ViewerConfig) -> MarkingRecord:
    """The M7 tables under ``<outputs>/m7`` (:data:`MARKING_FILES`); :class:`MissingArtefact`
    naming :data:`MARKING_SCRIPT` when the fits or the shadow-rotation table is absent (the
    binding maps are optional: empty frames when absent)."""
    root = cfg.outputs_root / "m7"
    frames: dict[str, pd.DataFrame] = {}
    for key, fname in MARKING_FILES.items():
        p = root / fname
        if p.exists():
            frames[key] = read_study_csv(p)
        elif key in ("fits", "shadow_rotation"):
            raise MissingArtefact(f"marking table {fname} under {root}", MARKING_SCRIPT)
        else:
            frames[key] = pd.DataFrame()
    frames["shadow_rotation"] = _scale_per_vp(frames["shadow_rotation"])
    spec_dir = REPO_ROOT / "configs" / "studies" / "m7_p1_marking"
    specs = tuple(sorted(p.name for p in spec_dir.glob("*.yaml"))) if spec_dir.exists() else ()
    return MarkingRecord(
        frames["fits"],
        frames["shadow_rotation"],
        frames["soft_skew"],
        frames["skew_tradeoff"],
        specs,
        ROTATION_CONVENTION,
        str(root),
    )


def marking_unpaired(table: str, df: pd.DataFrame) -> frozenset[str]:
    """The declared unpaired columns of a marking table (:data:`UNPAIRED_UPSTREAM` and, for
    the binding maps, :data:`MARKING_UNPAIRED_PREFIXES` / :data:`MARKING_UNPAIRED_PATTERNS`)."""
    base = UNPAIRED_UPSTREAM.get(f"marking_{table}", frozenset())
    if table in ("soft_skew", "skew_tradeoff"):
        base = base | frozenset(
            str(c)
            for c in df.columns
            if str(c).startswith(MARKING_UNPAIRED_PREFIXES)
            or any(pat.match(str(c)) for pat in MARKING_UNPAIRED_PATTERNS)
        )
    return base


# --------------------------------------------------------------------------------------------
# store header (page_header)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StoreSummary:
    """What the page header shows: roots, the store's code tag / git commit / grid / particle
    count (from the latest run record), the number of points and runs."""

    cfg: ViewerConfig
    code_tag: str
    git_commit: str
    grid_name: str
    n_particles: int | None
    n_points: int
    n_runs: int
    last_run_utc: str
    store_exists: bool
    cache_exists: bool

    def as_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["cfg"] = self.cfg.as_dict()
        return d


def store_summary(cfg: ViewerConfig) -> StoreSummary:
    man = _reader(cfg).manifest()
    runs = man.get("runs") or []
    grid = man.get("grid") if isinstance(man.get("grid"), dict) else {}
    last = runs[-1] if runs else {}
    return StoreSummary(
        cfg,
        str(man.get("code_tag") or ""),
        str(man.get("git_commit") or ""),
        str((grid or {}).get("name", "")),
        int(man["n_particles"]) if man.get("n_particles") else None,
        len(man.get("points") or {}),
        len(runs),
        str(last.get("created_utc", "")),
        (cfg.store_root / "results").exists(),
        cfg.cache_root.exists(),
    )


__all__ = [
    "AXES",
    "DISCRIMINATOR_STDERR",
    "EXACT_COLUMNS",
    "EXACT_PREFIXES",
    "GRID_COLUMNS",
    "HEDGING_RUN_COLUMNS",
    "MARKING_UNPAIRED_PATTERNS",
    "MARKING_UNPAIRED_PREFIXES",
    "MODEL_PARAMS",
    "PRODUCT_CATALOGUE",
    "RISK_COLUMNS",
    "TABLE_EXACT",
    "UNPAIRED_UPSTREAM",
    "HedgingRunRecord",
    "LeverageRecord",
    "MarkingRecord",
    "MissingArtefact",
    "MissingPoint",
    "PointRecord",
    "StoreSummary",
    "SurfaceRecord",
    "columns_without_stderr",
    "get_hedging_pnl",
    "get_hedging_run",
    "get_hedging_table",
    "get_leverage",
    "get_marking",
    "get_point",
    "get_products",
    "get_risk",
    "get_surface",
    "grid_path_of",
    "is_exact_column",
    "list_grid",
    "list_hedging_runs",
    "list_surfaces",
    "marking_unpaired",
    "normalise_stderr_names",
    "precompute_command",
    "read_study_csv",
    "reattach_stderr_names",
    "store_summary",
]
