"""``volsto-precompute`` — build a parameter grid into the leverage cache and the results store
(SPEC §9.2, M9 Part 1; owner's ``viewers/precompute.py``, here ``volsto/viewers/precompute.py``
with the console script in ``pyproject.toml``).

**This CLI is the one place of the viewers layer that calibrates a leverage** — through
:meth:`~volsto.calibration.cache.LeverageCache.get_or_calibrate`, logged per point as a cache
hit or a miss ("calibrating") and recorded in the store's manifest (``calibrated_this_run``) and
in the run record (``n_calibrated`` / ``n_cache_hits``, also on the ``done:`` line).  The pages
and the read API never do; a missing point makes them print the ``volsto-precompute`` command
that produces it (the contract with :mod:`volsto.viewers.api` is below).

Per grid point (:func:`compute_point`), in this order, each step timed (``wall_calibration_s``,
``wall_diagnostics_s``, ``wall_pricing_s``, ``wall_analytics_s``, ``wall_risk_s`` in the
``points`` table):

1. **calibration** — the leverage through the cache (the LV point of a surface builds the Dupire
   local vol instead; an infeasible marking fit is recorded without a model).  A cache **miss**
   calibrates with ``run_diagnostics=True`` at the point's pricing :class:`~volsto.config.
   SimConfig`, so the :class:`~volsto.calibration.diagnostics.CalibrationReport` lands in the
   cache (``diagnostics.json``) and the store's ``max_abs_error_vp`` / ``max_z`` (the M3
   acceptance region ``T ≤ min(2y, horizon)``, ``|k| ≤ 0.2``) are finite.  A cache **hit** is read
   by file (``leverage.npz`` + ``diagnostics.json``; a hit without a report keeps NaN unless
   ``--diagnostics`` backfills it: :func:`~volsto.calibration.diagnostics.reprice_surface` at the
   pricing SimConfig, written next to the leverage — no calibration);
2. **headline pricing** — :func:`volsto.studies.m4.run_headline` on this point's model
   (1y → 2y forward smile on :data:`~volsto.studies.m4.HEADLINE_STRIKES`, forward VS / vol swap,
   the study cliquets, the M4c conditional-variance / VKO set when ``products.conditional``) and
   :func:`volsto.studies.m6.run_m6_headline` (the 3y autocall and Phoenix) when ``products.m6``,
   with the grid's pricing :class:`~volsto.config.SimConfig` (400k paths, seed 2024 — the M4 /
   M6 baseline convention, stated in the manifest).  The pure-LV reference of every surface is a
   grid point of its own (mode ``lv``), priced once and stored like any other;
3. **forward smiles** at 1y → 2y (from step 2) and 2y → 3y (:func:`~volsto.analytics.
   forward_smile.forward_smile`) with the forward ATM vol / forward VS / forward vol-swap triple
   (:func:`~volsto.analytics.forward_smile.forward_vol_comparison`) of each window; both windows
   are stored whatever the calibration horizon, with ``beyond_horizon = t2 > horizon`` flagged
   per row (beyond the horizon the last leverage slice is held constant — the pages must say so);
4. **SSR term structure** — :func:`~volsto.analytics.smile_dynamics.ssr_numerical_many` at
   :data:`SSR_PILLARS` within the calibration horizon, ``eps`` :data:`SSR_EPS` (the stage-3
   convention of M7);
5. **Var(V) decomposition** — :func:`~volsto.analytics.var_decomp.var_decomposition` at
   :data:`VARV_MATURITIES` within the horizon (LSV points; the analytic needs a kernel);
6. **risk** — a :class:`~volsto.risk.report.RiskReport` per risk product (``autocall 3y``,
   ``cliquet 1y``) at the tier ``--risk none|light|full``: *light* = the delta / gamma regimes,
   the forward-variance ladder with ``risk.fwd_var_buckets`` buckets (20 = the M5 ladder, the
   default meaning of "fwd-var ladder"; ``len(light_pillars)`` = the coarse ladder on the light
   pillars) and the skew ladder at ``risk.light_pillars``; *full* = every section of
   :data:`volsto.risk.report.SECTIONS` at the library defaults.  On an LSV point every surface
   bump is a leverage calibration, so the tier dominates the budget (:func:`risk_plan` counts
   the light tier's calibrations and pricings from the configured buckets and pillars; the full
   tier is the measured engine count :data:`RISK_FULL_CALIBRATIONS` / :data:`RISK_FULL_PRICINGS`);
7. **store** — the rows of every table and the point's manifest entry
   (:meth:`~volsto.viewers.store.ResultsStore.write_point`).

**Projection.**  Before computing, ``projected wall clock`` prints, per tier, ``points ×
(overhead + calibration + diagnostics + pricing + analytics + risk) s`` with the per-point cost
model (:class:`CostModel`): overhead = the measured per-point fixed cost (market build with the
ξ₀ strip, leverage load or Dupire build, mean |L − 1|; :data:`OVERHEAD_S_FALLBACK`), charged to
every point; calibration = the median measured ``wall_time`` of the cache manifest at the grid's
particle count and horizon (linearly rescaled from another particle count when the exact one is
absent, else :data:`CALIBRATION_S_FALLBACK`), charged to cache misses only; diagnostics = the
measured repricing of the pillar surface (:data:`DIAGNOSTICS_S_FALLBACK`), charged to every miss
and, under ``--diagnostics``, to every hit without a report; pricing and analytics = the
``--probe`` measurement on the first point of the shard (else the measured fallbacks
:data:`PRICING_S_FALLBACK` / :data:`ANALYTICS_S_FALLBACK` at 400k paths, rescaled by the grid's
paths); risk = bumped states per tier × the budget-independent per-state cost of the point's
mode (:data:`RISK_STATE_OVERHEAD_S` + :data:`RISK_STATE_BUILD_S`) + (states − 1) × calibration
cost (LSV points; the base state is the point's own leverage) + pricings per tier × the risk
pricing of the point's mode (:data:`RISK_PRICING_RATIO` × its pricing step).  The light tier
is projected twice — at the configured bucket count and at the alternative one (20 ↔
``len(light_pillars)``) — so the owner can choose.  A running ETA follows every finished point.
``--dry-run`` enumerates, prints the counts per surface and mode and the projection and
computes nothing (the cache manifest is read, never written).

**Sharding, workers, selection, resume.**  ``--shard i/n`` takes the interleaved shard
:func:`~volsto.viewers.grid.shard`; ``--only ID [ID ...]`` keeps only the named points of the
shard (the stable ids: the leverage-cache key of a calibrated point, ``lv:<hash>``,
``marking:<label>``; an unknown id is an error listing the nearest ids); ``--workers k`` runs the
shard's points in a spawn-based :class:`multiprocessing.pool.Pool` of ``k`` processes, each with
``NUMBA_NUM_THREADS = max(1, cpu_count // k)`` (set in the environment before the pool starts so
that each child's numba import sees it — numba reads the variable once, at import).
``--resume`` is tier-aware (:func:`pending_steps`): a point whose ``points`` row exists in the
store **and** whose leverage sits in the cache (its ``cache_key``; LV and infeasible points need
none) is skipped when its stored ``risk_tier`` is at least the requested one; a point stored at a
lower tier is **refreshed for the risk step only** (:func:`refresh_point`: the stored tables are
reused, the model is rebuilt from the cache — never calibrated — and the point is rewritten with
the new tier, ``updated_utc`` / ``updated_steps``); with ``--diagnostics`` a stored LSV point
whose cache entry has no ``diagnostics.json`` is refreshed for the diagnostics step the same
way.  Nothing else is recomputed on a resume, so the per-point manifest entries of the skipped
points are unchanged.

**Workers and threads.**  ``--workers k`` runs ``k`` spawn processes with ``NUMBA_NUM_THREADS =
--threads-per-worker`` (default ``max(1, cpu_count // k)``); with one process
``--threads-per-worker`` lowers the pool numba started with (``numba.set_num_threads``).  The
per-point work is mostly serial at the production budget (SPEC §9.2 "VM sizing": the measured
thread ratios of :data:`MEASURED_THREAD_RATIO`), so throughput comes from many 1–2-thread
workers, bounded by the physical cores and by memory (the peak RSS each run records).

**Measured cost** (the VM runbook, ``docs/vm_grid_run.md``).  Every run record carries, per
computed point, its mode, the steps run, ``calibrated`` / ``cache_hit``, the wall split, the
numba thread count and the process's peak RSS, and at the top the workers, threads per worker,
CPU count, skipped / failed counts and the peak RSS.  ``volsto-precompute report --store ROOT``
(:func:`report_main`) reads ``results/runs/*.json`` — never computes — and prints per run the
wall clock, host, workers × threads, shard, points computed / skipped / failed, calibrated vs
cache hits (the hit rate), core-hours and peak RSS, and per (kind, mode, how) and step
(:data:`COST_STEPS`) the median, p90 and core-seconds (seconds × threads per worker); it writes
``cost_report.md`` and ``runs.csv`` / ``points.csv`` / ``summary.csv`` under
``<store>/cost_report`` (or ``--out``).  Fields an older record lacks are listed and left absent
(``--assume-threads`` states a thread count, labelled as assumed).  ``--cost-from ROOT``
(:func:`cost_model_from_store`) replaces the manifest / fallback cost model by the medians a
previous store measured, rescaled linearly in particles (calibration) and paths (diagnostics,
pricing, analytics, risk pricing) and by the measured thread ratio :func:`thread_factor` from
the source's thread count to ``--threads-per-worker``, printing the source of every number;
``--cost-from-threads`` states the source's count when its records predate the field.  A dry run
with ``--workers k`` also prints the parallel wall clock of the points it would compute
(their process-hours / k, or the longest point; a lower bound) — the shard's, not the grid's.

**Contract with the read API** (:func:`volsto.viewers.api.precompute_command`): a missing point
is produced by ``volsto-precompute --grid <grid> --store <store> --cache <cache> --only <id>
--resume``; a stored point whose risk table is missing (or below the wanted tier) by the same
line with ``--risk light`` (or ``full``) — ``--resume`` makes it the risk-only refresh above, so
the stored pricing is not redone; a point whose calibration diagnostics are missing by the same
line with ``--diagnostics``.

**Failures and provenance.**  Every point runs under its own ``try``: an exception is logged
with its traceback, recorded as a ``failed`` entry (id, label, error, traceback) of the run
record, the remaining points continue and the CLI exits 1 at the end (a later ``--resume`` picks
the failures up).  The run record (``results/runs/*.json``) stores no absolute path: the grid
path relative to the repository root (the basename, with a warning, for a grid outside it) and
the ``argv`` with the ``--store`` / ``--cache`` / ``--grid`` values replaced by ``<store>`` /
``<cache>`` / that relative grid path, so a store rsync'd from a VM reads unchanged.  The run
header prints the absolute store and cache roots (the defaults are repository-root-relative:
``outputs/store`` and ``cache``, like the viewer's).

Checked by ``tests/test_precompute.py`` on the one sanctioned calibrating fixture of the layer
(``tests/conftest.py::toy_build``: a 3-point toy grid at 2·10⁴ particles into a temporary cache);
the cost report and ``--cost-from`` by ``test_cost_report_on_synthetic_records`` (an old-format
record included), ``test_cost_report_on_the_toy_store``,
``test_cost_from_builds_the_cost_model_from_the_toy_store``,
``test_cost_from_risk_budgets_by_mode`` and ``test_thread_factor_reads_the_measured_ratios``;
the cache manifest's concurrent writers by ``tests/test_cache_concurrency.py``.
"""

from __future__ import annotations

import argparse
import dataclasses
import difflib
import json
import logging
import math
import multiprocessing
import os
import platform
import re
import sys
import time
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volsto.analytics.forward_smile import ForwardSmile, forward_smile, forward_vol_comparison
from volsto.analytics.smile_dynamics import ssr_numerical_many
from volsto.analytics.var_decomp import var_decomposition
from volsto.calibration.cache import LeverageCache, build_market, code_version, spec_key
from volsto.calibration.diagnostics import CalibrationReport, reprice_surface
from volsto.calibration.fit_2f import mean_abs_leverage_deviation
from volsto.calibration.particle import CALIBRATION_CODE_TAG
from volsto.config import BergomiParams, CalibrationSpec, SimConfig
from volsto.market.dupire import LocalVolSurface
from volsto.models.base import Model
from volsto.models.leverage import LeverageFunction
from volsto.models.localvol import LocalVol
from volsto.models.lsv import LSV
from volsto.products.base import Product
from volsto.products.cliquet import AdditiveCliquet
from volsto.risk.engine import LSVBuilder, LVBuilder, ModelBuilder, RiskEngine, RiskState
from volsto.risk.greeks import REGIMES
from volsto.risk.ladders import default_buckets
from volsto.risk.report import SECTIONS, risk_report
from volsto.studies.m4 import HEADLINE_STRIKES, run_headline
from volsto.studies.m6 import AUTOCALL_NAME, headline_products, run_m6_headline
from volsto.viewers.grid import (
    M5_FWD_VAR_BUCKETS,
    REPO_ROOT,
    RISK_TIERS,
    GridPoint,
    GridSpec,
    RiskSettings,
    count_by,
    enumerate_points,
    grid_mapping,
    load_grid,
    parse_shard,
    resolve_marking,
    shard,
)
from volsto.viewers.store import PointResult, ResultsStore, utc_now

log = logging.getLogger(__name__)

DEFAULT_GRID = REPO_ROOT / "configs" / "grids" / "default.yaml"
#: Store and cache defaults resolved against the repository root like the grid (and like the
#: viewer's :mod:`volsto.viewers.config`), so the CLI finds the same cache from any directory.
DEFAULT_STORE = REPO_ROOT / "outputs" / "store"
DEFAULT_CACHE = REPO_ROOT / "cache"
#: Forward-start windows ``(T1, T2)`` of the stored smiles: 1y into 1y and 2y into 1y.
FORWARD_WINDOWS: tuple[tuple[float, float], ...] = ((1.0, 2.0), (2.0, 3.0))
#: SSR pillars (those within the calibration horizon are computed).
SSR_PILLARS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0)
#: State-bump size of the numerical SSR — the M7 stage-3 convention
#: (:class:`volsto.calibration.fit_2f.Stage3Inputs` ``eps``).
SSR_EPS = 0.05
#: Var(V) maturities (those within the calibration horizon are computed).
VARV_MATURITIES: tuple[float, ...] = (1.0, 3.0)
#: Sections of the light risk tier (owner: delta regimes + fwd-var ladder + skew_T at three
#: pillars); the full tier is :data:`volsto.risk.report.SECTIONS`.
LIGHT_SECTIONS: tuple[str, ...] = ("delta", "fwd_var", "skew")
CLIQUET_1Y_NAME = "cliquet 1y"
#: Region of the calibration diagnostics summarised in the store (``max_abs_error_vp`` /
#: ``max_z``): the M3 acceptance region ``T ≤ 2y``, ``|k| ≤ 0.2`` (:meth:`CalibrationReport.
#: max_abs_error` defaults), capped at the calibration horizon (beyond it the leverage is the
#: held last slice, not a calibrated one).
DIAGNOSTICS_T_MAX = 2.0
DIAGNOSTICS_K_ABS = 0.2
#: Distinct leverage calibrations / CRN pricings of a **full** RiskReport of one point (both
#: risk products, whose bumped states are shared), counted with ``RiskEngine.n_calibrations`` /
#: ``n_pricings`` on the LV builder of the placeholder surface (Dupire rebuilds count like
#: recalibrations; the base state included): the §7.13 budget.  The full tier runs every
#: section at the library defaults (9 pillars, 20 buckets, 7 parameters), which no grid setting
#: changes, so the measured count is a constant; the light tier is counted per grid by
#: :func:`risk_plan`.
RISK_FULL_CALIBRATIONS = 90
RISK_FULL_PRICINGS = 356
#: Fallback calibration cost when the cache manifest has no measured entry (s): the median
#: ``wall_time`` at 8·10⁵ particles, 3y horizon, code tag m6, on the owner's laptop (12 cores).
CALIBRATION_S_FALLBACK = 140.0
#: Per-point fixed overhead (s), measured on this machine on the placeholder surface: the market
#: build (SSVI + ξ₀ strip, 3.5 s — the SPX snapshots take 2 s) with the leverage read from the
#: cache and the mean |L − 1| (LSV cache hit, 8·10⁵ particles) or the Dupire build (0.05 s, LV
#: point); charged to every point whatever the step it runs.
OVERHEAD_S_FALLBACK = 3.8
#: Calibration diagnostics cost (s) at 400k paths: ``reprice_surface`` of the 7 × 9 pillar
#: vanillas + 7 variance swaps on common paths, measured on the cached placeholder 1F ω = 2 point
#: (8·10⁵ particles) at the pricing SimConfig; rescaled linearly in the grid's paths.
DIAGNOSTICS_S_FALLBACK = 26.0
#: Fallback pricing cost of steps 2–3 at 400k paths (s), measured on this machine (12 cores) on
#: the cached placeholder 1F ω = 2 point at 8·10⁵ particles: M4 headline (conditional set, two
#: cliquets) 33.8 s + 2y → 3y smile and forward-vol triple 47.4 s + M6 autocall / Phoenix 95.2 s;
#: rescaled linearly in the grid's paths.
PRICING_S_FALLBACK = 176.4
#: Fallback cost of steps 4–5 at 400k paths (s), same measurement: SSR at five pillars 64.3 s +
#: Var(V) at 1y and 3y 60.3 s.
ANALYTICS_S_FALLBACK = 124.6
#: Fallback cost of one risk pricing at 400k paths (s), same measurement: the 3y autocall on one
#: path set 21.7 s, the 1y cliquet CLIQUET_1Y_PRICING_S; the two risk products alternate, so
#: the mean is one pure pricing.  Not used by the projection any more: a risk pricing also
#: carries the bumped state's model build (:data:`RISK_STATE_BUILD_S`), and the pure risk
#: pricing is measured as :data:`RISK_PRICING_RATIO` × the point's pricing step.
AUTOCALL_PRICING_S = 21.7
CLIQUET_1Y_PRICING_S = 10.5
RISK_PRICING_S_FALLBACK = 0.5 * (AUTOCALL_PRICING_S + CLIQUET_1Y_PRICING_S)
#: Paths the fallback costs were measured with.
FALLBACK_N_PATHS = 400_000
#: Wall-clock splits of a point, in order (``wall_<name>_s`` columns of the ``points`` table).
WALL_STEPS: tuple[str, ...] = ("calibration", "diagnostics", "pricing", "analytics", "risk")
#: The steps a ``--resume`` can run on a stored point (:func:`pending_steps`); ``"all"`` is a
#: full computation.
REFRESH_STEPS: tuple[str, ...] = ("risk", "diagnostics")

_T_IN_NAME = re.compile(r"\[(-?\d*\.?\d+(?:e-?\d+)?)y\]")
_BUCKET_IN_NAME = re.compile(r"\[(-?\d*\.?\d+(?:e-?\d+)?)-(-?\d*\.?\d+(?:e-?\d+)?)y\]")

# M4 headline columns → (product, quantity, unit)
M4_PRODUCTS: dict[str, tuple[str, str, str]] = {
    "atm_vol": ("fwd ATM vol 1y→2y", "vol", "vol"),
    "vs_vol": ("fwd VS 1y→2y", "fair vol", "vol"),
    "volswap_vol": ("fwd vol swap 1y→2y", "fair vol", "vol"),
    "upvar_100": ("up-var 1y B=100%", "fair vol", "vol"),
    "downvar_100": ("down-var 1y B=100%", "fair vol", "vol"),
    "kovar_110": ("KO var 1y B=110%", "fair vol", "vol"),
    "kovar_110_p_ko": ("KO var 1y B=110%", "P(KO)", "probability"),
    "vko_30": ("VKO 12m 100% put @30%", "price", "% notional"),
    "vko_30_p_ko": ("VKO 12m 100% put @30%", "P(KO)", "probability"),
}


# --------------------------------------------------------------------------------------------
# risk plan, cost model and projection
# --------------------------------------------------------------------------------------------


def light_buckets(risk: RiskSettings) -> tuple[tuple[float, float], ...]:
    """The light tier's forward-variance buckets: the M5 ladder for ``fwd_var_buckets = 20``,
    else the coarse ladder on the light pillars (:attr:`RiskSettings.light_buckets`)."""
    if risk.light_buckets is None:
        bk = default_buckets()
        if len(bk) != M5_FWD_VAR_BUCKETS:  # pragma: no cover - the library ladder changed
            raise RuntimeError(f"default_buckets() has {len(bk)} buckets, not {M5_FWD_VAR_BUCKETS}")
        return bk
    return risk.light_buckets


def risk_plan(tier: str, risk: RiskSettings, n_buckets: int | None = None) -> tuple[int, int]:
    """``(calibrations, pricings)`` of the RiskReports of one LSV point at ``tier`` for the grid's
    risk settings (``n_buckets`` overrides ``risk.fwd_var_buckets`` for the alternative
    projection).

    The light tier is counted from the plan of :func:`~volsto.risk.report.risk_report` with the
    :class:`~volsto.risk.engine.LSVBuilder` rule (a ``recalibrate`` state is one calibration,
    the ``model`` regime prices the base calibration moved in spot, bumped states are shared
    between the products, every state × product is one CRN pricing): the base state; the delta
    section — of the five regimes the ``model`` one recalibrates nothing and the other four
    recalibrate at both spots (4 × 2 states, 5 × 2 pricings per product; the gamma's base term
    is the memoised base pricing); the forward-variance ladder — one forward bump per bucket
    plus the parallel bump (``buckets + 1``); the skew ladder — one tent per pillar plus the
    global bump (``pillars + 1``).  Hence ``calibrations = 1 + 8 + (buckets + 1) + (pillars +
    1)`` and ``pricings = products × (1 + 10 + (buckets + 1) + (pillars + 1))``: 34 / 72 at the
    defaults (20 buckets, 3 pillars, 2 products) — the count measured with
    ``RiskEngine.n_calibrations`` / ``n_pricings`` on the LV builder of the placeholder surface
    (SPEC §9.2) and re-measured on a non-simulating plan engine at 20 and 3 buckets for this
    formula (``tests/test_precompute.py::test_risk_plan_counts``).  The full tier is the measured
    constant :data:`RISK_FULL_CALIBRATIONS` / :data:`RISK_FULL_PRICINGS` (library defaults, no
    grid setting).
    """
    if tier not in RISK_TIERS:
        raise ValueError(f"tier must be one of {RISK_TIERS}")
    if tier == "none":
        return 0, 0
    if tier == "full":
        return RISK_FULL_CALIBRATIONS, RISK_FULL_PRICINGS
    nb = risk.fwd_var_buckets if n_buckets is None else int(n_buckets)
    n_pillars = len(risk.light_pillars)
    n_products = len(risk.products)
    recal_regimes = len(REGIMES) - 1  # the "model" regime reprices the base calibration
    calibrations = 1 + 2 * recal_regimes + (nb + 1) + (n_pillars + 1)
    pricings = n_products * (1 + 2 * len(REGIMES) + (nb + 1) + (n_pillars + 1))
    return calibrations, pricings


def alternative_buckets(risk: RiskSettings) -> int:
    """The other admissible light-tier bucket count (20 ↔ ``len(light_pillars)``), projected
    beside the configured one so the owner can choose."""
    return len(risk.light_pillars) if risk.fwd_var_buckets == M5_FWD_VAR_BUCKETS else 20


@dataclass(frozen=True)
class CostModel:
    """Per-point costs in seconds, where they come from, and the risk plan per tier
    (module docstring)."""

    overhead_s: float
    calibration_s: float
    calibration_source: str
    diagnostics_s: float
    pricing_s: float
    analytics_s: float
    risk_pricing_s: float
    pricing_source: str
    risk_calibrations: dict[str, int]
    risk_pricings: dict[str, int]
    #: Serial, budget-independent cost of the risk step per bumped state of an LSV point: the work
    #: outside the engine's timer (:data:`RISK_STATE_OVERHEAD_S`) + the state's model build
    #: inside it (:data:`RISK_STATE_BUILD_S`); an LV point's is ``by_mode["lv"]["risk_state"]``.
    risk_state_s: float
    risk_source: str
    #: Per-mode overrides ``{mode: {"overhead" | "pricing" | "analytics" | "risk_pricing" |
    #: "risk_state": s}}`` — set by ``--cost-from`` (:func:`cost_model_from_store`) and by
    #: :func:`default_cost_model` (the LV risk costs); a mode absent here is charged the scalar
    #: costs above.
    by_mode: dict[str, dict[str, float]] = dataclasses.field(default_factory=dict)

    def _mode_s(self, mode: str | None, step: str, default: float) -> float:
        return float(self.by_mode.get(mode or "", {}).get(step, default))

    def risk_s(self, tier: str, calibrates: bool, mode: str | None = None) -> float:
        """The risk step of one point: ``states × risk state + (states − 1) × calibration``
        (LSV points: the base state is the point's own leverage, a cache hit) ``+ pricings ×
        risk pricing``, the risk state and pricing of ``mode`` (an LV point's state cost is its
        Dupire rebuild)."""
        states = self.risk_calibrations[tier]
        pricings = self.risk_pricings[tier]
        if states == 0 and pricings == 0:
            return 0.0
        cal = max(states - 1, 0) * self.calibration_s if calibrates else 0.0
        return (
            states * self._mode_s(mode, "risk_state", self.risk_state_s)
            + cal
            + pricings * self._mode_s(mode, "risk_pricing", self.risk_pricing_s)
        )

    def point_s(
        self,
        tier: str,
        *,
        calibrates: bool,
        miss: bool,
        backfill: bool = False,
        mode: str | None = None,
    ) -> float:
        """A full computation: overhead + (calibration + diagnostics on a miss) + (diagnostics
        on a hit backfilled under ``--diagnostics``) + pricing + analytics + risk (the overhead,
        pricing and analytics of ``mode`` when :attr:`by_mode` has them)."""
        return (
            self._mode_s(mode, "overhead", self.overhead_s)
            + (self.calibration_s + self.diagnostics_s if miss else 0.0)
            + (self.diagnostics_s if backfill and not miss else 0.0)
            + self._mode_s(mode, "pricing", self.pricing_s)
            + self._mode_s(mode, "analytics", self.analytics_s)
            + self.risk_s(tier, calibrates, mode)
        )

    def steps_s(
        self,
        tier: str,
        steps: Sequence[str],
        *,
        calibrates: bool,
        miss: bool,
        backfill: bool = False,
        mode: str | None = None,
    ) -> float:
        """Cost of ``steps`` (:func:`pending_steps`): nothing when done, :meth:`point_s` for a
        full computation, the overhead plus the named refresh steps otherwise."""
        if not steps:
            return 0.0
        if "all" in steps:
            return self.point_s(
                tier, calibrates=calibrates, miss=miss, backfill=backfill, mode=mode
            )
        s = self._mode_s(mode, "overhead", self.overhead_s)
        if "risk" in steps:
            s += self.risk_s(tier, calibrates, mode)
        if "diagnostics" in steps:
            s += self.diagnostics_s
        return s


def calibration_cost(cache: LeverageCache, n_particles: int, horizon: float) -> tuple[float, str]:
    """Median measured calibration wall time at ``(n_particles, horizon)`` from the cache
    manifest (current code tag preferred, any tag otherwise), rescaled linearly from another
    particle count when the exact one is absent, else the fallback."""
    m = cache.manifest()
    if m.empty or "wall_time" not in m:
        return (
            CALIBRATION_S_FALLBACK,
            f"fallback {CALIBRATION_S_FALLBACK:g} s (empty cache manifest)",
        )
    m = m[np.isfinite(m["wall_time"].to_numpy(dtype=float))]
    same_h = m[np.isclose(m["horizon"].to_numpy(dtype=float), horizon)]
    if same_h.empty:
        same_h = m
    exact = same_h[same_h["n_particles"] == n_particles]
    tagged = exact[exact["code_tag"] == CALIBRATION_CODE_TAG]
    if not tagged.empty:
        exact = tagged
    if not exact.empty:
        med = float(exact["wall_time"].median())
        return med, (
            f"cache manifest median of {len(exact)} entries at {n_particles} particles, "
            f"horizon {horizon:g}y (code tag {exact['code_tag'].iloc[0]})"
        )
    counts = same_h["n_particles"].to_numpy(dtype=float)
    nearest = float(counts[np.argmin(np.abs(np.log(counts / n_particles)))])
    sub = same_h[same_h["n_particles"] == nearest]
    med = float(sub["wall_time"].median()) * n_particles / nearest
    return med, (
        f"cache manifest median of {len(sub)} entries at {nearest:g} particles rescaled "
        f"linearly to {n_particles}"
    )


def risk_plan_by_tier(risk: RiskSettings) -> tuple[dict[str, int], dict[str, int]]:
    """``({tier: calibrations}, {tier: pricings})`` for the configured settings, plus the
    ``light@<n>b`` entry of the alternative bucket count."""
    cals: dict[str, int] = {}
    prs: dict[str, int] = {}
    for tier in RISK_TIERS:
        cals[tier], prs[tier] = risk_plan(tier, risk)
    alt = alternative_buckets(risk)
    cals[f"light@{alt}b"], prs[f"light@{alt}b"] = risk_plan("light", risk, alt)
    return cals, prs


def default_cost_model(cache: LeverageCache, grid: GridSpec) -> CostModel:
    """The cost model from the cache manifest and the measured fallbacks (no ``--cost-from``):
    the risk pricing of an LSV point is :data:`RISK_PRICING_RATIO` × the fallback pricing, of an
    LV point :data:`RISK_PRICING_RATIO` × :data:`LV_TO_LSV_PRICING_RATIO` × the fallback pricing
    (the LV ratio is to the LV pricing step, the fallback is an LSV one); each risk state is
    charged :data:`RISK_STATE_OVERHEAD_S` + :data:`RISK_STATE_BUILD_S`."""
    cal, src = calibration_cost(cache, grid.particle.n_particles, grid.particle.horizon)
    scale = grid.pricing.n_paths / FALLBACK_N_PATHS
    cals, prs = risk_plan_by_tier(grid.risk)
    pricing = PRICING_S_FALLBACK * scale
    return CostModel(
        OVERHEAD_S_FALLBACK,
        cal,
        src,
        DIAGNOSTICS_S_FALLBACK * scale,
        pricing,
        ANALYTICS_S_FALLBACK * scale,
        RISK_PRICING_RATIO["lsv"] * pricing,
        f"measured fallbacks at {FALLBACK_N_PATHS} paths rescaled to {grid.pricing.n_paths}",
        cals,
        prs,
        RISK_STATE_OVERHEAD_S + RISK_STATE_BUILD_S["lsv"],
        _RISK_RATIO_SOURCE,
        by_mode={
            "lv": {
                "risk_pricing": RISK_PRICING_RATIO["lv"] * LV_TO_LSV_PRICING_RATIO * pricing,
                "risk_state": RISK_STATE_OVERHEAD_S + RISK_STATE_BUILD_S["lv"],
            }
        },
    )


def _hit(cache: LeverageCache, point: GridPoint) -> bool:
    """Cache hit for a resolved calibrating point (an unresolved marking point counts as a miss)."""
    if not point.calibrates:
        return True
    if point.cache_key is None:
        return False
    return cache.has_key(point.cache_key)


def _has_report(cache: LeverageCache, key: str | None) -> bool:
    return bool(key) and (cache.root / str(key) / "diagnostics.json").exists()


def projection_table(
    points: Sequence[GridPoint],
    cache: LeverageCache,
    cost: CostModel,
    risk: RiskSettings,
    *,
    diagnostics: bool = False,
) -> pd.DataFrame:
    """One row per tier (the light tier twice: the configured bucket count and the alternative
    one): points, cache misses, per-point costs and the total."""
    misses = sum(1 for p in points if p.calibrates and not _hit(cache, p))

    def _backfill(p: GridPoint) -> bool:
        """Only a calibrated point that is a cache hit without a report is repriced."""
        return (
            diagnostics and p.calibrates and _hit(cache, p) and not _has_report(cache, p.cache_key)
        )

    backfills = sum(1 for p in points if _backfill(p))
    n_cal = sum(1 for p in points if p.calibrates)
    alt = alternative_buckets(risk)
    variants: list[tuple[str, str, int]] = [
        ("none", "none", 0),
        ("light", "light", risk.fwd_var_buckets),
        ("light", f"light@{alt}b", alt),
        ("full", "full", 0),
    ]
    rows = []
    for tier, plan_key, n_buckets in variants:
        plan = dataclasses.replace(
            cost,
            risk_calibrations={tier: cost.risk_calibrations[plan_key]},
            risk_pricings={tier: cost.risk_pricings[plan_key]},
        )
        total = sum(
            plan.point_s(
                tier,
                calibrates=p.calibrates,
                miss=p.calibrates and not _hit(cache, p),
                backfill=_backfill(p),
                mode=p.mode,
            )
            for p in points
        )
        n = max(len(points), 1)
        rows.append(
            {
                "tier": tier,
                "fwd_var_buckets": n_buckets,
                "points": len(points),
                "cache_misses": misses,
                "diagnostics_backfills": backfills,
                "overhead_s": round(cost.overhead_s, 1),
                "calibration_s_per_miss": round(cost.calibration_s, 1),
                "diagnostics_s": round(cost.diagnostics_s, 1),
                "pricing_s": round(cost.pricing_s, 1),
                "analytics_s": round(cost.analytics_s, 1),
                "risk_calibrations": cost.risk_calibrations[plan_key],
                "risk_pricings": cost.risk_pricings[plan_key],
                "risk_s_lsv_point": round(plan.risk_s(tier, True), 1),
                "risk_s_lv_point": round(plan.risk_s(tier, False, "lv"), 1),
                "mean_s_per_point": round(total / n, 1),
                "total_h": round(total / 3600.0, 2),
                "calibrating_points": n_cal,
            }
        )
    return pd.DataFrame(rows)


def format_projection(table: pd.DataFrame, cost: CostModel, tier: str, risk: RiskSettings) -> str:
    lines = [
        "projected wall clock (per tier; the requested tier is marked *; the light tier at the "
        "configured fwd-var bucket count and at the alternative one):",
        f"  overhead: {cost.overhead_s:.1f} s per point (market build, leverage load / Dupire)",
        f"  calibration cost: {cost.calibration_s:.1f} s per cache miss — "
        f"{cost.calibration_source}",
        f"  diagnostics cost: {cost.diagnostics_s:.1f} s per calibration (pillar repricing; also "
        "per hit without a report under --diagnostics)",
        f"  pricing / analytics cost: {cost.pricing_s:.1f} / {cost.analytics_s:.1f} s per point"
        f" — {cost.pricing_source}",
        f"  risk step: states x {cost.risk_state_s:.1f} s (LSV) / "
        f"{cost._mode_s('lv', 'risk_state', cost.risk_state_s):.1f} s (LV) + (states - 1) x "
        f"calibration (LSV) + pricings x {cost.risk_pricing_s:.1f} s (LSV) / "
        f"{cost._mode_s('lv', 'risk_pricing', cost.risk_pricing_s):.1f} s (LV) — "
        f"{cost.risk_source}",
    ]
    for _, r in table.iterrows():
        configured = r["tier"] != "light" or int(r["fwd_var_buckets"]) == risk.fwd_var_buckets
        mark = "*" if (r["tier"] == tier and configured) else " "
        ladder = (
            f", {int(r['fwd_var_buckets'])}-bucket fwd-var ladder" if r["tier"] == "light" else ""
        )
        n_back = int(r["diagnostics_backfills"])
        backfills = f", {n_back} backfills" if n_back else ""
        lines.append(
            f" {mark}{r['tier']:<6} {int(r['points'])} points ({int(r['cache_misses'])} misses"
            f"{backfills}) x "
            f"({r['overhead_s']} overhead + {r['calibration_s_per_miss']} cal + "
            f"{r['diagnostics_s']} diag + {r['pricing_s']} pricing + {r['analytics_s']} "
            f"analytics + {r['risk_s_lsv_point']} risk (LV {r['risk_s_lv_point']}) "
            f"[{int(r['risk_calibrations'])} states, "
            f"{int(r['risk_pricings'])} pricings{ladder}] s; mean {r['mean_s_per_point']} "
            f"s/point) = {r['total_h']} h"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# per-point computation
# --------------------------------------------------------------------------------------------


def pricing_sim(grid: GridSpec, spec_sim: SimConfig) -> SimConfig:
    """The pricing SimConfig: the calibration schedule / scheme of the spec with the grid's paths
    and seed (the M4 baseline convention)."""
    return dataclasses.replace(spec_sim, n_paths=grid.pricing.n_paths, seed=grid.pricing.seed)


def _smile_rows(smile: ForwardSmile, horizon: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "t1": smile.t1,
            "t2": smile.t2,
            "beyond_horizon": bool(smile.t2 > horizon + 1e-9),
            "strike_moneyness": smile.strikes,
            "log_moneyness": smile.log_moneyness,
            "cp": smile.cps,
            "iv": smile.vols,
            "iv_stderr": smile.vol_stderr,
            "price": smile.prices,
            "price_stderr": smile.price_stderr,
        }
    )


def _m4_product_rows(row: pd.Series) -> list[dict[str, Any]]:
    """The M4 headline row → ``products`` rows (columns with a ``_stderr`` twin only; the
    ITM-conditional percentiles and ``p_itm`` / ``p_ko_itm_*`` carry none and are not stored)."""
    out: list[dict[str, Any]] = []
    for col in row.index:
        c = str(col)
        if c in ("model", "wall_s") or c.endswith("_stderr") or f"{c}_stderr" not in row.index:
            continue
        if c in M4_PRODUCTS:
            product, quantity, unit = M4_PRODUCTS[c]
        elif c.startswith("cliquet_"):
            product, quantity, unit = f"cliquet {c[len('cliquet_') :]}", "price", "% notional"
        elif c.startswith("vko_ratio_"):
            h = c[len("vko_ratio_") :]
            product, quantity, unit = f"VKO 12m 100% put @{h}%", "ratio to vanilla", "ratio"
        else:
            product, quantity, unit = c, "value", ""
        out.append(
            {
                "product": product,
                "quantity": quantity,
                "key": c,
                "value": float(row[c]),
                "value_stderr": float(row[f"{c}_stderr"]),
                "unit": unit,
            }
        )
    return out


def _m6_product_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _, r in table.iterrows():
        pname = str(r["product"])
        for col in table.columns:
            c = str(col)
            if c.endswith("_stderr") or f"{c}_stderr" not in table.columns:
                continue
            v = r[c]
            if not isinstance(v, int | float | np.floating) or not np.isfinite(float(v)):
                continue
            unit = "% notional" if c == "price" or c.startswith("leg:") else ""
            unit = "years" if c == "expected_life" else unit
            unit = "probability" if c.startswith("p_") else unit
            out.append(
                {
                    "product": pname,
                    "quantity": c,
                    "key": f"{pname}:{c}",
                    "value": float(v),
                    "value_stderr": float(r[f"{c}_stderr"]),
                    "unit": unit,
                }
            )
    return out


def _risk_rows(rows: list[dict[str, Any]], product: str, tier: str) -> pd.DataFrame:
    out: list[dict[str, Any]] = []
    for r in rows:
        name = str(r["name"])
        t = math.nan
        bucket = ""
        mb = _BUCKET_IN_NAME.search(name)
        mt = _T_IN_NAME.search(name)
        if mb:
            bucket = f"{mb.group(1)}-{mb.group(2)}y"
        elif mt:
            t = float(mt.group(1))
        out.append(
            {
                "product": product,
                "tier": tier,
                "group": r["group"],
                "name": name,
                "value": float(r["value"]),
                "value_stderr": float(r["stderr"]),
                "unit": r.get("unit", ""),
                "size": float(r.get("size", math.nan)),
                "scheme": r.get("scheme", ""),
                "states": r.get("states", ""),
                "T": t,
                "bucket": bucket,
                "regime": str(r.get("regime", "")),
                "variant": str(r.get("variant", "")),
                "key": str(r.get("key", "")),
            }
        )
    return pd.DataFrame(out)


def build_lv(point: GridPoint) -> LocalVol:
    _, surface, _ = build_market(point.spec)
    return LocalVol(LocalVolSurface.from_implied(surface, point.spec.local_vol))


def risk_products(model: Model) -> dict[str, Product]:
    discount = model.forward_curve.rate_curve
    return {
        AUTOCALL_NAME: headline_products(discount, model.spot)[AUTOCALL_NAME],
        CLIQUET_1Y_NAME: AdditiveCliquet.study(1.0, discount),
    }


def store_report(
    cache: LeverageCache,
    spec: CalibrationSpec,
    leverage: LeverageFunction,
    report: CalibrationReport,
) -> Path:
    """Write a calibration report next to an existing cache entry (``diagnostics.json``) and
    refresh its manifest row (``max_abs_error_vp``).  Not :meth:`LeverageCache.store`: that
    would re-save the leverage and stamp a new ``created_utc`` / ``git_commit`` on an entry that
    was not recalibrated (the manifest row is rebuilt from the leverage's own metadata, so the
    original stamps survive)."""
    if not cache.has(spec):
        raise FileNotFoundError(f"no cache entry to attach diagnostics to: {cache.key(spec)}")
    path = cache.write_report(spec, report)  # atomic, like every cache write
    cache._append_manifest(cache.key(spec), spec, leverage, report)
    return path


def report_summary(report: CalibrationReport, horizon: float) -> tuple[float, float]:
    """``(max_abs_error_vp, max_z)`` over the diagnostics region (module constants, ``T`` capped
    at the horizon)."""
    t_max = min(DIAGNOSTICS_T_MAX, horizon)
    return (
        float(report.max_abs_error(t_max=t_max, k_abs=DIAGNOSTICS_K_ABS)),
        float(report.max_z(t_max=t_max, k_abs=DIAGNOSTICS_K_ABS)),
    )


def _risk_step(
    spec: CalibrationSpec,
    label: str,
    mode: str,
    grid: GridSpec,
    cache: LeverageCache,
    tier: str,
    model: Model,
    sim: SimConfig,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Step 6: the risk table of one point at ``tier`` and the engine budget.  On an LSV point
    the builder calibrates the bumped states through the cache (the precompute is the one place
    that calibrates); on the LV point every bump is a Dupire rebuild."""
    state = RiskState(spec, None, label)
    builder: ModelBuilder = (
        LVBuilder(state) if mode == "lv" else LSVBuilder(cache, state, allow_calibrate=True)
    )
    engine = RiskEngine(builder, sim)
    prods = risk_products(model)
    frames = []
    for name in grid.risk.products:
        if name not in prods:
            raise ValueError(f"unknown risk product {name!r}; known: {sorted(prods)}")
        if tier == "light":
            rep = risk_report(
                engine,
                prods[name],
                state,
                sections=LIGHT_SECTIONS,
                pillars=grid.risk.light_pillars,
                buckets=light_buckets(grid.risk),
            )
        else:
            rep = risk_report(engine, prods[name], state, sections=SECTIONS)
        frames.append(_risk_rows(rep.rows, name, tier))
        log.info("%s / %s: %s", label, name, rep.summary())
    table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return table, engine.budget()


def compute_point(
    point: GridPoint,
    grid: GridSpec,
    cache: LeverageCache,
    tier: str,
    *,
    diagnostics: bool = False,
) -> PointResult:
    """Steps 1–7 of the module docstring for one point (the store write is the caller's);
    ``diagnostics`` backfills the calibration report of a cache hit that has none."""
    t_start = time.perf_counter()
    walls = dict.fromkeys(WALL_STEPS, 0.0)
    row: dict[str, Any] = {
        "label": point.label,
        "surface": point.surface,
        "mode": point.mode,
        "status": point.status,
    }
    tables: dict[str, pd.DataFrame] = {}
    manifest: dict[str, Any] = {
        "label": point.label,
        "surface": point.surface,
        "mode": point.mode,
        "risk_tier": tier,
        "n_particles": grid.particle.n_particles,
        "horizon": grid.particle.horizon,
        "pricing": {"n_paths": grid.pricing.n_paths, "seed": grid.pricing.seed},
        "git_commit": code_version(),
        "code_tag": CALIBRATION_CODE_TAG,
        "host": platform.node(),
    }
    horizon = grid.particle.horizon
    sim = pricing_sim(grid, point.spec.sim)

    # -- 1. model: the marking fit (if any), then the leverage through the cache -------------
    t0 = time.perf_counter()
    if point.mode == "marking" and not point.resolved:
        _, surface, _ = build_market(point.spec)
        point = resolve_marking(point, surface)
    row["status"] = point.status
    for k in ("nu", "theta", "k1", "k2", "rho12", "rho_SX1", "rho_SX2"):
        row[k] = float(getattr(point.spec.model, k)) if point.params is not None else math.nan
    for axis, col in (
        ("nu", "axis_nu"),
        ("rho", "axis_rho"),
        ("kappa", "axis_kappa"),
        ("ssr_target", "ssr_target"),
        ("skew_eps", "skew_eps"),
    ):
        row[col] = float(point.axes.get(axis, math.nan))
    row["cache_key"] = point.cache_key or ""
    row["n_particles"] = grid.particle.n_particles
    row["horizon"] = grid.particle.horizon
    if point.marking is not None:
        m = point.marking
        row["fit_status"] = m.status
        row["fit_messages"] = " | ".join(m.messages)
        row["fit_mean_skew_gap"] = m.mean_skew_gap
        row["fit_wall_seconds"] = m.wall_seconds
        for k, v in m.skew_gaps.items():
            row[k] = v
        manifest["marking"] = {
            "status": m.status,
            "messages": list(m.messages),
            "notes": list(m.notes),
            "skew_gaps": m.skew_gaps,
            "mean_skew_gap": m.mean_skew_gap,
        }
    model: Model | None = None
    calibrated = False
    backfilled = False
    row["calibration_seconds"] = math.nan
    row["mean_abs_L_minus_1"] = math.nan
    row["max_abs_error_vp"] = math.nan
    row["max_z"] = math.nan
    if point.mode == "lv":
        model = build_lv(point)
        log.info("%s: Dupire local vol built (no calibration)", point.label)
    elif point.calibrates:
        miss = not cache.has(point.spec)
        report: CalibrationReport | None
        if miss:
            log.info(
                "%s: leverage cache miss %s — calibrating at %d particles (the precompute is "
                "the one place that calibrates), diagnostics at %d paths",
                point.label,
                (point.cache_key or "")[:12],
                grid.particle.n_particles,
                sim.n_paths,
            )
            lsv, report = cache.get_or_calibrate(
                point.spec, allow_calibrate=True, run_diagnostics=True, diagnostics_sim=sim
            )
            _, surface, _ = build_market(point.spec)
            if report is not None:
                walls["diagnostics"] = float(report.wall_time)
        else:
            log.info("%s: leverage cache hit %s", point.label, (point.cache_key or "")[:12])
            # the hit is a read: leverage and report by file, the market built once here
            _, surface, kernel = build_market(point.spec)
            lsv = LSV(kernel, cache.load(point.spec))
            report = cache.load_report(point.spec)
            if report is None and diagnostics:
                t_d = time.perf_counter()
                report = reprice_surface(lsv, surface, sim)
                store_report(cache, point.spec, lsv.leverage, report)
                walls["diagnostics"] = time.perf_counter() - t_d
                backfilled = True
                log.info(
                    "%s: calibration diagnostics backfilled in %.1f s (%d paths, no calibration)",
                    point.label,
                    walls["diagnostics"],
                    sim.n_paths,
                )
        calibrated = miss
        model = lsv
        row["calibration_seconds"] = float(lsv.leverage.metadata.get("wall_time", math.nan))
        row["mean_abs_L_minus_1"] = mean_abs_leverage_deviation(lsv.leverage, surface)[0]
        if report is not None:
            row["max_abs_error_vp"], row["max_z"] = report_summary(report, horizon)
        log.info(
            "%s: leverage ready in %.1f s (%s; calibration wall %.1f s, mean |L-1| %.4f, "
            "max |error| %.3f vp, max z %.2f)",
            point.label,
            time.perf_counter() - t0,
            "calibrated" if miss else "cache hit",
            row["calibration_seconds"],
            row["mean_abs_L_minus_1"],
            row["max_abs_error_vp"],
            row["max_z"],
        )
    else:
        log.info("%s: %s — no calibration, nothing priced", point.label, point.status)
    row["calibrated_this_run"] = calibrated
    row["diagnostics_backfilled"] = backfilled
    walls["calibration"] = time.perf_counter() - t0 - walls["diagnostics"]
    manifest["calibrated_this_run"] = calibrated
    manifest["diagnostics_backfilled"] = backfilled
    manifest["cache_key"] = point.cache_key
    manifest["status"] = point.status

    if model is not None:
        # -- 2/3. headline pricing, forward smiles and forward vol triples -------------------
        t0 = time.perf_counter()
        products: list[dict[str, Any]] = []
        smiles: list[pd.DataFrame] = []
        fvols: list[dict[str, Any]] = []
        (t1a, t2a), (t1b, t2b) = FORWARD_WINDOWS
        if grid.products.m4:
            head = run_headline(
                {point.label: model},
                sim,
                t1=t1a,
                t2=t2a,
                cliquet_maturities=grid.products.cliquet_maturities,
                conditional=grid.products.conditional,
            )
            r0 = head.table.iloc[0]
            products += _m4_product_rows(r0)
            smiles.append(_smile_rows(head.smiles[point.label], horizon))
            fvols.append(
                {
                    "t1": t1a,
                    "t2": t2a,
                    "beyond_horizon": bool(t2a > horizon + 1e-9),
                    "fwd_atm_vol": float(r0["atm_vol"]),
                    "fwd_atm_vol_stderr": float(r0["atm_vol_stderr"]),
                    "fwd_vs": float(r0["vs_vol"]),
                    "fwd_vs_stderr": float(r0["vs_vol_stderr"]),
                    "fwd_volswap": float(r0["volswap_vol"]),
                    "fwd_volswap_stderr": float(r0["volswap_vol_stderr"]),
                }
            )
        else:
            smiles.append(
                _smile_rows(forward_smile(model, t1a, t2a, HEADLINE_STRIKES, sim), horizon)
            )
            fvols.append(_fvc_row(model, t1a, t2a, sim, horizon))
        smiles.append(_smile_rows(forward_smile(model, t1b, t2b, HEADLINE_STRIKES, sim), horizon))
        fvols.append(_fvc_row(model, t1b, t2b, sim, horizon))
        if grid.products.m6:
            m6 = run_m6_headline({point.label: model}, sim)
            products += _m6_product_rows(m6.table)
        tables["products"] = pd.DataFrame(products)
        tables["forward_smile"] = pd.concat(smiles, ignore_index=True)
        tables["forward_vols"] = pd.DataFrame(fvols)
        walls["pricing"] = time.perf_counter() - t0
        log.info("%s: pricing done in %.1f s", point.label, walls["pricing"])

        # -- 4/5. SSR term structure and Var(V) ----------------------------------------------
        t0 = time.perf_counter()
        pillars = [T for T in SSR_PILLARS if horizon + 1e-9 >= T]
        ssr_rows: list[dict[str, Any]] = []
        if pillars:
            fo = point.marking.ssr_first_order if point.marking is not None else {}
            for s in ssr_numerical_many(model, pillars, eps=SSR_EPS, sim=sim):
                naked = math.nan
                for T_fit, v in fo.items():
                    if abs(T_fit - s.T) < 1e-9:
                        naked = v
                ssr_rows.append(
                    {
                        "T": s.T,
                        "ssr_lsv": s.R,
                        "ssr_lsv_stderr": s.R_stderr,
                        "skew_lsv": s.skew,
                        "skew_lsv_stderr": s.skew_stderr,
                        "slope": s.slope,
                        "slope_stderr": s.slope_stderr,
                        "atmf_vol": s.atmf_vol,
                        "atmf_vol_stderr": s.atmf_vol_stderr,
                        "eps": s.eps,
                        "ssr_naked_first_order": naked,
                        "ssr_target": float(point.axes.get("ssr_target", math.nan)),
                    }
                )
        tables["ssr"] = pd.DataFrame(ssr_rows)
        varv_rows: list[dict[str, Any]] = []
        if isinstance(model, LSV):
            for T in VARV_MATURITIES:
                if horizon + 1e-9 < T:
                    continue
                vd = var_decomposition(model, T, sim=sim)
                for term in (
                    "mean",
                    "var_total",
                    "var_sv",
                    "var_leverage",
                    "cov_cross",
                    "var_explained",
                ):
                    varv_rows.append(
                        {
                            "T": T,
                            "term": term,
                            "value": float(getattr(vd, term)),
                            "value_stderr": float(getattr(vd, f"{term}_stderr")),
                            "kind": "mc",
                        }
                    )
                for term in ("mean_sv_closed", "var_sv_closed", "var_sv_discrete"):
                    varv_rows.append(
                        {
                            "T": T,
                            "term": term,
                            "value": float(getattr(vd, term)),
                            "value_stderr": 0.0,
                            "kind": "closed_form",
                        }
                    )
        tables["varv"] = pd.DataFrame(varv_rows)
        walls["analytics"] = time.perf_counter() - t0
        log.info("%s: analytics done in %.1f s", point.label, walls["analytics"])

        # -- 6. risk ------------------------------------------------------------------------
        t0 = time.perf_counter()
        if tier != "none":
            tables["risk"], manifest["risk_budget"] = _risk_step(
                point.spec, point.label, point.mode, grid, cache, tier, model, sim
            )
        walls["risk"] = time.perf_counter() - t0

    total = time.perf_counter() - t_start
    row["risk_tier"] = tier
    row["pricing_n_paths"] = grid.pricing.n_paths
    row["pricing_seed"] = grid.pricing.seed
    row["wall_seconds"] = total
    for k, v in walls.items():
        row[f"wall_{k}_s"] = v
    row["git_commit"] = manifest["git_commit"]
    row["code_tag"] = CALIBRATION_CODE_TAG
    row["created_utc"] = utc_now()
    manifest["wall"] = {**walls, "total": total}
    manifest["created_utc"] = row["created_utc"]
    log.info(
        "%s: done in %.1f s (calibration %.1f / diagnostics %.1f / pricing %.1f / analytics "
        "%.1f / risk %.1f)",
        point.label,
        total,
        walls["calibration"],
        walls["diagnostics"],
        walls["pricing"],
        walls["analytics"],
        walls["risk"],
    )
    return PointResult(point.id, row, tables, manifest)


def _fvc_row(model: Model, t1: float, t2: float, sim: SimConfig, horizon: float) -> dict[str, Any]:
    c = forward_vol_comparison(model, t1, t2, sim)
    return {
        "t1": t1,
        "t2": t2,
        "beyond_horizon": bool(t2 > horizon + 1e-9),
        "fwd_atm_vol": c.atm_vol,
        "fwd_atm_vol_stderr": c.atm_stderr,
        "fwd_vs": c.vs_vol,
        "fwd_vs_stderr": c.vs_stderr,
        "fwd_volswap": c.volswap_vol,
        "fwd_volswap_stderr": c.volswap_stderr,
    }


# --------------------------------------------------------------------------------------------
# resume: pending steps and the partial refresh of a stored point
# --------------------------------------------------------------------------------------------


def _stored_row(store: ResultsStore, point_id: str) -> pd.Series:
    return pd.read_parquet(store.point_dir(point_id) / "points.parquet").iloc[0]


def pending_steps(
    point: GridPoint,
    store: ResultsStore,
    cache: LeverageCache,
    tier: str,
    *,
    diagnostics: bool = False,
) -> tuple[str, ...]:
    """The ``--resume`` rule.  ``("all",)`` when the point must be computed (no store row, or a
    calibrating point whose stored ``cache_key`` has no leverage in the cache); ``()`` when it is
    done; otherwise the refresh steps: ``"risk"`` when the stored ``risk_tier`` is below
    ``tier`` (LV / LSV points with a model), ``"diagnostics"`` when ``diagnostics`` is requested
    and the LSV point's cache entry has no ``diagnostics.json``.  Infeasible marking points are
    done once stored (no model)."""
    if not store.has_point(point.id):
        return ("all",)
    row = _stored_row(store, point.id)
    status = str(row.get("status", "") or "")
    has_model = point.mode == "lv" or status != "infeasible"
    key = str(row.get("cache_key", "") or "")
    if point.mode != "lv" and has_model and not (key and cache.has_key(key)):
        return ("all",)
    steps: list[str] = []
    stored_tier = str(row.get("risk_tier", "none") or "none")
    if stored_tier not in RISK_TIERS:
        stored_tier = "none"
    if has_model and RISK_TIERS.index(stored_tier) < RISK_TIERS.index(tier):
        steps.append("risk")
    if diagnostics and point.mode != "lv" and has_model and not _has_report(cache, key):
        steps.append("diagnostics")
    return tuple(steps)


def is_done(
    point: GridPoint,
    store: ResultsStore,
    cache: LeverageCache,
    tier: str,
    *,
    diagnostics: bool = False,
) -> bool:
    """``pending_steps(...) == ()``."""
    return not pending_steps(point, store, cache, tier, diagnostics=diagnostics)


def _stored_spec(point: GridPoint, row: dict[str, Any]) -> CalibrationSpec:
    """The calibration spec of a stored point: the grid point's own for the axis / preset modes,
    the stored fitted parameters for a marking point (its enumeration spec carries the LV
    placeholder model); the rebuilt key must match the stored ``cache_key``."""
    if point.mode != "marking":
        return point.spec
    params = BergomiParams(
        nu=float(row["nu"]),
        theta=float(row["theta"]),
        k1=float(row["k1"]),
        k2=float(row["k2"]),
        rho12=float(row["rho12"]),
        rho_SX1=float(row["rho_SX1"]),
        rho_SX2=float(row["rho_SX2"]),
    )
    spec = dataclasses.replace(point.spec, model=params)
    key = str(row.get("cache_key", "") or "")
    if spec_key(spec) != key:
        raise RuntimeError(
            f"{point.label}: the stored fit parameters do not reproduce the stored cache key "
            f"{key[:12]} (got {spec_key(spec)[:12]}); recompute the point without --resume"
        )
    return spec


def refresh_point(
    point: GridPoint,
    grid: GridSpec,
    cache: LeverageCache,
    tier: str,
    store: ResultsStore,
    steps: Sequence[str],
) -> PointResult:
    """Run only ``steps`` (a subset of :data:`REFRESH_STEPS`) on a stored point and return the
    rewritten result: the stored tables and row are reused, the model is rebuilt from the cache
    (the leverage read by file — **never calibrated**; the LV point rebuilds its Dupire local
    vol), the touched wall clocks and ``risk_tier`` / diagnostics fields are updated and
    ``updated_utc`` / ``updated_steps`` record the refresh (``created_utc`` stays the first
    write's)."""
    unknown = set(steps) - set(REFRESH_STEPS)
    if unknown:
        raise ValueError(f"unknown refresh steps {sorted(unknown)}; known: {REFRESH_STEPS}")
    res = store.read_point(point.id)
    row, manifest, tables = res.row, res.manifest, res.tables
    horizon = grid.particle.horizon
    walls = {
        k: float(manifest.get("wall", {}).get(k, row.get(f"wall_{k}_s", 0.0)) or 0.0)
        for k in WALL_STEPS
    }
    t0 = time.perf_counter()
    spec = _stored_spec(point, row)
    sim = pricing_sim(grid, spec.sim)
    model: Model
    lsv: LSV | None = None
    _, surface, kernel = build_market(spec)
    if point.mode == "lv":
        model = LocalVol(LocalVolSurface.from_implied(surface, spec.local_vol))
    else:
        lsv = LSV(kernel, cache.load(spec))  # CacheMissError if the entry vanished
        model = lsv
    log.info(
        "%s: refresh %s on the stored point (%s, no calibration)",
        point.label,
        list(steps),
        "cache hit" if lsv is not None else "Dupire rebuilt",
    )
    overhead = time.perf_counter() - t0
    done: dict[str, float] = {}
    if "diagnostics" in steps:
        if lsv is None:
            raise ValueError(f"{point.label}: the LV point has no calibration to diagnose")
        t0 = time.perf_counter()
        report = reprice_surface(lsv, surface, sim)
        store_report(cache, spec, lsv.leverage, report)
        walls["diagnostics"] = done["diagnostics"] = time.perf_counter() - t0
        row["max_abs_error_vp"], row["max_z"] = report_summary(report, horizon)
        row["diagnostics_backfilled"] = True
        manifest["diagnostics_backfilled"] = True
        log.info(
            "%s: calibration diagnostics backfilled in %.1f s (%d paths): max |error| %.3f vp, "
            "max z %.2f",
            point.label,
            walls["diagnostics"],
            sim.n_paths,
            row["max_abs_error_vp"],
            row["max_z"],
        )
    if "risk" in steps:
        t0 = time.perf_counter()
        if tier == "none":
            tables.pop("risk", None)
            manifest.pop("risk_budget", None)
        else:
            tables["risk"], manifest["risk_budget"] = _risk_step(
                spec, point.label, point.mode, grid, cache, tier, model, sim
            )
        walls["risk"] = done["risk"] = time.perf_counter() - t0
        row["risk_tier"] = tier
        manifest["risk_tier"] = tier
    total = float(sum(walls.values()))
    row["wall_seconds"] = total
    for k, v in walls.items():
        row[f"wall_{k}_s"] = v
    row["updated_utc"] = utc_now()
    row["updated_steps"] = ",".join(steps)
    row["git_commit"] = code_version()
    manifest["wall"] = {**walls, "total": total}
    manifest["updated"] = {
        "utc": row["updated_utc"],
        "steps": list(steps),
        "git_commit": row["git_commit"],
        "wall": {**done, "overhead": overhead},
    }
    return PointResult(point.id, row, tables, manifest)


def process_point(
    point: GridPoint,
    grid: GridSpec,
    cache: LeverageCache,
    tier: str,
    store: ResultsStore,
    steps: Sequence[str],
    *,
    diagnostics: bool = False,
) -> dict[str, Any]:
    """Compute (``steps == ("all",)``) or refresh a point, write it, and return the run-record
    entry: id, label, steps, ``calibrated`` / ``cache_hit`` / ``diagnostics_backfilled`` and the
    wall clocks of what ran (``total`` = this run's time on the point)."""
    t0 = time.perf_counter()
    if "all" in steps:
        res = compute_point(point, grid, cache, tier, diagnostics=diagnostics)
        walls = {k: float(res.manifest["wall"][k]) for k in WALL_STEPS}
        calibrated = bool(res.manifest["calibrated_this_run"])
        hit = bool(res.row["mode"] != "lv" and res.row["status"] != "infeasible" and not calibrated)
        backfilled = bool(res.manifest.get("diagnostics_backfilled", False))
    else:
        res = refresh_point(point, grid, cache, tier, store, steps)
        upd = res.manifest["updated"]["wall"]
        walls = {k: float(upd.get(k, 0.0)) for k in WALL_STEPS}
        walls["calibration"] = float(upd.get("overhead", 0.0))
        calibrated = False
        hit = point.mode != "lv"
        backfilled = "diagnostics" in steps
    store.write_point(res)
    return {
        "point_id": point.id,
        "label": point.label,
        "mode": point.mode,
        "steps": list(steps),
        "calibrated": calibrated,
        "cache_hit": hit,
        "diagnostics_backfilled": backfilled,
        **walls,
        "total": time.perf_counter() - t0,
        "threads": numba_threads(),
        "pid": os.getpid(),
        "peak_rss_bytes": peak_rss_bytes(),
    }


# --------------------------------------------------------------------------------------------
# measured cost: the run-record report and the cost model built from a previous store
# --------------------------------------------------------------------------------------------

#: Modes of the cost report, in grid order (:data:`volsto.viewers.grid.MODES`).
COST_MODES: tuple[str, ...] = ("lv", "one_factor", "two_factor", "marking")
#: Steps of the cost report.  ``calibration`` is the leverage's particle pass alone (the
#: ``calibration_seconds`` of the store row, for a point calibrated in the run); ``overhead`` is
#: everything else of the timed calibration step (market build, leverage load or Dupire build,
#: mean |L − 1|, the marking fit) plus the untimed remainder ``total − Σ steps`` (the store
#: write); the other steps are the run record's own splits.
COST_STEPS: tuple[str, ...] = (
    "overhead",
    "calibration",
    "diagnostics",
    "pricing",
    "analytics",
    "risk",
    "total",
)
#: Directory the report writes under ``--store`` by default (outside ``results/``, which the
#: store reader walks).
COST_REPORT_DIRNAME = "cost_report"
#: Top-level fields of a current run record the report reads; an older record's missing ones
#: are listed, never filled in.
RUN_RECORD_FIELDS: tuple[str, ...] = (
    "host",
    "shard",
    "workers",
    "threads_per_worker",
    "cpu_count",
    "wall_seconds",
    "points_selected",
    "points_skipped",
    "n_calibrated",
    "n_cache_hits",
    "failed",
    "n_particles",
    "pricing",
    "peak_rss_bytes",
)
#: Fields of a current ``points_computed`` entry the report reads.
POINT_ENTRY_FIELDS: tuple[str, ...] = (
    "mode",
    "steps",
    "calibrated",
    "cache_hit",
    "threads",
    "peak_rss_bytes",
    *WALL_STEPS,
    "total",
)
#: Measured wall-clock ratio ``t(n)/t(1)`` per step at ``n`` numba threads — the thread
#: rescaling of ``--cost-from`` (:func:`thread_factor`).  Owner's laptop (Apple M-series, 12
#: logical cores = 8 performance + 4 efficiency, 24 GiB, shared with other agents: load average
#: 2.5–4.3), one placeholder 1F point (ν 0.5, ρ −0.7, κ 1.5, risk none) at the probe budget
#: (10⁵ particles, 3y, 5·10⁴ paths), a fresh scratch cache per thread count
#: (``scratchpad/scaling/store_{1,2,4,8,12}``): particle pass 25.22 / 23.16 / 20.20 / 18.09 /
#: 16.25 s, diagnostics 4.63 / 3.40 / 2.72 / 2.40 / 2.33 s, pricing 33.19 / 24.94 / 20.44 /
#: 18.39 / 18.08 s, analytics 24.16 / 18.23 / 15.01 / 13.45 / 13.33 s.  A risk pricing is
#: charged the pricing ratio; the overhead (market build, store write) is serial.  The table is
#: used rather than an Amdahl law because Amdahl does not fit the particle pass
#: (:data:`THREAD_PARALLEL_FRACTION`); between the measured counts the ratio is interpolated
#: linearly in ``1/n``.  Production-budget check (8·10⁵ particles, 4·10⁵ paths, 12 vs 1 thread,
#: ``scratchpad/prod/store_{1,12}``): measured 0.709 / 0.488 / 0.540 / 0.546 against the table's
#: 0.644 / 0.503 / 0.545 / 0.552 — the particle pass is less parallel at production size (the
#: projections at 1 thread per worker therefore read a 1-thread source and do not rescale).
#: These are two *real* laptop cores at ``n = 2``: a second thread on an SMT sibling of a server
#: core gains less (unmeasured; the runbook's VM probe A measures it).
MEASURED_THREAD_RATIO: dict[str, dict[int, float]] = {
    "overhead": {1: 1.0, 2: 1.0, 4: 1.0, 8: 1.0, 12: 1.0},
    "calibration": {1: 1.0, 2: 0.9182, 4: 0.8009, 8: 0.7172, 12: 0.6442},
    "diagnostics": {1: 1.0, 2: 0.7328, 4: 0.5861, 8: 0.5185, 12: 0.5029},
    "pricing": {1: 1.0, 2: 0.7516, 4: 0.6159, 8: 0.5541, 12: 0.5447},
    "analytics": {1: 1.0, 2: 0.7544, 4: 0.6213, 8: 0.5566, 12: 0.5516},
    "risk": {1: 1.0, 2: 0.7516, 4: 0.6159, 8: 0.5541, 12: 0.5447},
}
#: Amdahl parallel fraction ``p`` per step (``t(n)/t(1) = 1 − p + p/n``), least-squares fitted
#: to every measured ratio (the probe at 2, 4, 8, 12 threads and the production budget at 12),
#: used only beyond the largest measured count (:func:`thread_factor`).  Fit residuals
#: (measured − fitted ratio at 2 / 4 / 8 / 12 / production-12 threads): particle pass p 0.316,
#: +0.076 / +0.038 / −0.006 / −0.066 / −0.002 — Amdahl misfits it (the second thread gains
#: 8 %, not the 16 % a single fraction implies), hence the table; diagnostics p 0.549, +0.007 /
#: −0.002 / −0.001 / +0.007 / −0.008; pricing p 0.504, +0.004 / −0.006 / −0.005 / +0.007 /
#: +0.002; analytics p 0.498, +0.003 / −0.005 / −0.008 / +0.008 / +0.002.  (The previous
#: 1-vs-12-only fit gave 0.32 / 0.56 / 0.50 / 0.50 and over-predicted the 2-thread particle-pass
#: gain: 0.84 against the measured 0.918.)
THREAD_PARALLEL_FRACTION: dict[str, float] = {
    "overhead": 0.0,
    "calibration": 0.316,
    "diagnostics": 0.549,
    "pricing": 0.504,
    "analytics": 0.498,
    "risk": 0.504,
}
#: Risk-step constants, measured on the laptop at NUMBA_NUM_THREADS = 1 on one placeholder 1F
#: point (ν 0.5, ρ −0.7, κ 1.5) and the surface's LV point, ``--risk light`` on the 3-bucket
#: ladder (17 bumped states and 38 pricings per point), at two budgets: probe C (10⁵ particles,
#: 5·10⁴ paths; ``scratchpad/vmverify/P/C``) and production (8·10⁵ / 4·10⁵;
#: ``scratchpad/vmfix/prodrisk``, run wall 6211 s, peak RSS 2.36 GiB).  Risk step wall /
#: engine timer (state builds + pricings): 1F 726.3 / 642.8 s and 4283.0 / 4200.9 s; LV
#: 206.7 / 123.1 s and 943.5 / 860.8 s.  The 16 risk calibrations took 408.4 s and 3045.5 s
#: (cache manifests).  The engine remainder (LSV: minus those calibrations) is ``38 a + 17 b``
#: with ``a`` a pure pricing (∝ paths) and ``b`` a per-state build (budget-independent); the two
#: budgets give
#:
#: * :data:`RISK_STATE_OVERHEAD_S` — work outside the engine per state (surface arbitrage
#:   checks, the ε sizing of the bucket bumps on the variance-swap strip, the base-model build):
#:   4.913 / 4.830 s (1F, probe / production) and 4.919 / 4.866 s (LV) — mode- and
#:   budget-independent to 2 %, so their mean; serial, never rescaled.
#: * :data:`RISK_STATE_BUILD_S` — ``b``: 6.05 s per LSV state (market build, leverage load or
#:   store), 1.04 s per LV state (the Dupire rebuild).
#: * :data:`RISK_PRICING_RATIO` — ``a`` over the point's pricing step: LSV 0.1009 (probe) /
#:   0.1044 (production), LV 0.1195 / 0.1183; the production value is used.  This is *not* an
#:   independent check of the decomposition: the split imposes ``a_prod = 8 a_probe`` and solves
#:   ``b`` from the same two points, so the two ratios agreeing to 3.5 % only says the pricing
#:   step itself scaled by about 8 between the budgets.
#:
#: The model then reproduces both measurements by construction.  Its one independent check is
#: the outside-engine work, measured directly at both budgets and agreeing to 1.7 %; the grid
#: projections from different stores (177.0 / 180.6 / 185.7 process-hours) all use
#: :data:`RISK_STATE_BUILD_S` and are not independent of it either.  Earlier rules on the
#: production 1F risk step (4283.0 s): the pre-fix rule (17 × calibration + 38 × 0.0913 ×
#: pricing) −0.3 % by cancellation (17 passes charged for 16 run, 921 s charged for 1237.5 s of
#: pricings and per-state work; −31 % on the LV step); probe C × 8 +36 %; probe C with the state
#: build folded into the pricing +17 %.
RISK_STATE_OVERHEAD_S = (4.913 + 4.830 + 4.919 + 4.866) / 4
RISK_STATE_BUILD_S: dict[str, float] = {"lsv": 6.050, "lv": 1.040}
RISK_PRICING_RATIO: dict[str, float] = {"lsv": 0.1044, "lv": 0.1183}
#: LV pricing step / LSV (1F) pricing step of the same production run (1 thread,
#: ``scratchpad/vmfix/prodrisk``): 187.5 / 265.4 s.  Converts an LSV pricing (the fallback, or a
#: ``--cost-from`` source without an LV point, e.g. probe D's ``--only`` store) into the LV
#: pricing step that :data:`RISK_PRICING_RATIO` ``["lv"]`` is measured against.  (At 12 threads,
#: ``outputs/store``: 118.6 / 180.3 = 0.658.)  The default model still charges an LV point's own
#: pricing and analytics steps at the LSV fallbacks: it has no per-mode fallbacks, and the four
#: LV points are under 0.3 % of the grid; ``--cost-from`` charges them their measured costs.
LV_TO_LSV_PRICING_RATIO = 187.5 / 265.4
_RISK_RATIO_SOURCE = (
    f"measured risk-step constants (laptop, 1 thread, probe C + production): "
    f"{RISK_STATE_OVERHEAD_S:.2f} s per state outside the engine + a state build of "
    f"{RISK_STATE_BUILD_S['lsv']:.2f} s (LSV) / {RISK_STATE_BUILD_S['lv']:.2f} s (LV); risk "
    f"pricing = {RISK_PRICING_RATIO['lsv']:.4f} (LSV) x the pricing step, "
    f"{RISK_PRICING_RATIO['lv']:.4f} (LV) x the LV pricing step, which the default model takes as "
    f"{LV_TO_LSV_PRICING_RATIO:.4f} x the LSV pricing step (measured at 1 thread; the 12-thread "
    "ratio is lower — a small mixing of thread counts on 4 LV points)"
)
#: The mode groups of the risk costs: every leverage mode shares the LSV builder.
LSV_MODES: tuple[str, ...] = ("one_factor", "two_factor", "marking")


def _thread_ratio(step: str, n: int) -> float:
    """``t(n)/t(1)`` of ``step`` (:data:`MEASURED_THREAD_RATIO`, linear in ``1/n`` between the
    measured counts, the fitted Amdahl tail beyond the largest)."""
    table = MEASURED_THREAD_RATIO[step]
    if n in table:
        return table[n]
    counts = sorted(table)
    if n > counts[-1]:
        top = counts[-1]
        return table[top] + THREAD_PARALLEL_FRACTION[step] * (1.0 / n - 1.0 / top)
    hi = next(c for c in counts if c > n)
    lo = max(c for c in counts if c < n)
    w = (1.0 / n - 1.0 / hi) / (1.0 / lo - 1.0 / hi)
    return w * table[lo] + (1.0 - w) * table[hi]


def thread_factor(step: str, n_from: int, n_to: int) -> float:
    """Rescaling factor ``t(n_to)/t(n_from)`` of ``step`` (:data:`MEASURED_THREAD_RATIO`); 1
    when the counts are equal."""
    if n_from < 1 or n_to < 1:
        raise ValueError("thread counts must be >= 1")
    if n_from == n_to:
        _thread_ratio(step, n_from)  # an unknown step is still an error
        return 1.0
    return _thread_ratio(step, n_to) / _thread_ratio(step, n_from)


@dataclass(frozen=True)
class CostRecords:
    """The run records of a store, flattened: ``runs`` (one row per ``results/runs/*.json``),
    ``points`` (one row per ``points_computed`` entry, the :data:`COST_STEPS` in seconds and
    ``core_<step>`` = seconds × threads) and ``absent`` (run file → the fields of
    :data:`RUN_RECORD_FIELDS` / :data:`POINT_ENTRY_FIELDS` that record lacks)."""

    root: Path
    runs: pd.DataFrame
    points: pd.DataFrame
    absent: dict[str, list[str]]


def _store_points(store: ResultsStore) -> dict[str, dict[str, Any]]:
    """``point_id → {mode, calibration_seconds}`` from the store's points rows (read by file)."""
    out: dict[str, dict[str, Any]] = {}
    for pid in store.point_ids():
        try:
            row = _stored_row(store, pid)
        except (OSError, ValueError, IndexError):  # pragma: no cover - a half-written point
            continue
        out[pid] = {
            "mode": str(row.get("mode", "")),
            "calibration_seconds": float(row.get("calibration_seconds", math.nan)),
        }
    return out


def _mode_from_id(point_id: str) -> str | None:
    if point_id.startswith("lv:"):
        return "lv"
    if point_id.startswith("marking:"):
        return "marking"
    return None


def load_cost_records(store_root: str | Path, *, assume_threads: int | None = None) -> CostRecords:
    """Flatten the run records of ``store_root`` (module section "measured cost").

    Per entry: ``mode`` from the entry (``mode_source = record``), else the store's points row
    (``store``), else the id prefix ``lv:`` / ``marking:`` (``id``), else ``unknown``;
    ``threads`` from the entry, else the run's ``threads_per_worker``, else ``assume_threads``
    (``assumed``, the caller's statement), else absent (core-seconds NaN); ``calibrated`` /
    ``cache_hit`` from the entry, else absent (``None``).  The calibration step is split into the
    particle pass (``calibration_seconds`` of the store row) and the overhead only for an entry
    that says it calibrated; for one without the flag the step is kept whole
    (``calibration_split = False``).  Nothing is filled in from elsewhere."""
    store = ResultsStore(store_root)
    stored = _store_points(store)
    files = sorted(store.runs_dir.glob("*.json")) if store.runs_dir.exists() else []
    run_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    absent: dict[str, list[str]] = {}
    for path in files:
        rec = json.loads(path.read_text())
        entries = list(rec.get("points_computed") or [])
        missing = [f for f in RUN_RECORD_FIELDS if f not in rec]
        missing_e = sorted({f for e in entries for f in POINT_ENTRY_FIELDS if f not in e})
        absent[path.name] = missing + [f"points_computed[].{f}" for f in missing_e]
        run_threads = rec.get("threads_per_worker")
        pricing = rec.get("pricing") or {}
        n_paths = pricing.get("n_paths") if isinstance(pricing, dict) else None
        peaks: list[float] = []
        for e in entries:
            pid = str(e.get("point_id", ""))
            if "mode" in e:
                mode, mode_src = str(e["mode"]), "record"
            elif pid in stored and stored[pid]["mode"]:
                mode, mode_src = stored[pid]["mode"], "store"
            elif _mode_from_id(pid) is not None:
                mode, mode_src = str(_mode_from_id(pid)), "id"
            else:
                mode, mode_src = "unknown", "absent"
            if "threads" in e:
                threads, th_src = float(e["threads"]), "record"
            elif run_threads is not None:
                threads, th_src = float(run_threads), "record"
            elif assume_threads is not None:
                threads, th_src = float(assume_threads), "assumed"
            else:
                threads, th_src = math.nan, "absent"
            calibrated = e.get("calibrated")
            cache_hit = e.get("cache_hit")
            steps = e.get("steps")
            # a wall the entry does not carry is absent (NaN), never 0; the untimed remainder
            # (and so the overhead) is then unknown too
            walls = {k: float(e[k]) if e.get(k) is not None else math.nan for k in WALL_STEPS}
            total = float(e.get("total", math.nan))
            timed = sum(walls.values())
            remainder = max(0.0, total - timed) if math.isfinite(total + timed) else math.nan
            cal_step = walls["calibration"]
            cal_s = stored.get(pid, {}).get("calibration_seconds", math.nan)
            split = calibrated is not None and math.isfinite(cal_step)
            if not math.isfinite(cal_step):
                particle = math.nan
            elif calibrated and math.isfinite(cal_s):
                particle = min(cal_s, cal_step)
            elif calibrated:  # the flag says it calibrated but the store row is gone
                particle, split = cal_step, False
            elif calibrated is None:
                particle = cal_step
            else:
                particle = 0.0
            if calibrated is None:
                how = "unknown"
            elif calibrated:
                how = "calibrated"
            elif cache_hit:
                how = "cache hit"
            else:
                how = "no leverage"
            if steps is None:
                kind = "unknown"
            else:
                kind = "full" if "all" in steps else "refresh " + "+".join(steps)
            row: dict[str, Any] = {
                "run": path.name,
                "point_id": pid,
                "label": str(e.get("label", "")),
                "mode": mode,
                "mode_source": mode_src,
                "kind": kind,
                "how": how,
                "calibrated": calibrated,
                "cache_hit": cache_hit,
                "calibration_split": split,
                "threads": threads,
                "threads_source": th_src,
                "n_particles": rec.get("n_particles", math.nan),
                "n_paths": n_paths if n_paths is not None else math.nan,
                "peak_rss_bytes": float(e.get("peak_rss_bytes", math.nan)),
                "overhead": cal_step - particle + remainder,
                "calibration": particle,
                "diagnostics": walls["diagnostics"],
                "pricing": walls["pricing"],
                "analytics": walls["analytics"],
                "risk": walls["risk"],
                "total": total,
            }
            for step in COST_STEPS:
                row[f"core_{step}"] = row[step] * threads
            point_rows.append(row)
            peaks.append(row["peak_rss_bytes"])
        peaks.append(float(rec.get("peak_rss_bytes", math.nan)))
        finite = [p for p in peaks if math.isfinite(p)]
        n_computed = len(entries)
        n_failed = len(rec.get("failed") or []) if "failed" in rec else None
        if "points_skipped" in rec:
            skipped: Any = int(rec["points_skipped"])
        elif "points_selected" in rec and n_failed is not None:
            skipped = f"{int(rec['points_selected']) - n_computed - n_failed} (derived)"
        else:
            skipped = None
        n_cal = rec.get("n_calibrated")
        n_hit = rec.get("n_cache_hits")
        hit_rate = (
            n_hit / (n_cal + n_hit)
            if n_cal is not None and n_hit is not None and (n_cal + n_hit) > 0
            else math.nan
        )
        workers = rec.get("workers")
        if run_threads is not None:
            r_threads, r_src = float(run_threads), "record"
        elif assume_threads is not None:
            r_threads, r_src = float(assume_threads), "assumed"
        else:
            r_threads, r_src = math.nan, "absent"
        wall = float(rec.get("wall_seconds", math.nan))
        run_rows.append(
            {
                "run": path.name,
                "created_utc": rec.get("created_utc"),
                "host": rec.get("host"),
                "platform": rec.get("platform"),
                "cpu_count": rec.get("cpu_count"),
                "shard": rec.get("shard"),
                "workers": workers,
                "threads_per_worker": r_threads,
                "threads_source": r_src,
                "risk_tier": rec.get("risk_tier"),
                "n_particles": rec.get("n_particles"),
                "n_paths": n_paths,
                "resume": rec.get("resume"),
                "wall_seconds": wall,
                "core_hours": (
                    wall * float(workers) * r_threads / 3600.0 if workers is not None else math.nan
                ),
                "points_computed": n_computed,
                "points_skipped": skipped,
                "points_failed": n_failed,
                "n_calibrated": n_cal,
                "n_cache_hits": n_hit,
                "cache_hit_rate": hit_rate,
                "peak_rss_gib": max(finite) / 2**30 if finite else math.nan,
            }
        )
    return CostRecords(Path(store_root), pd.DataFrame(run_rows), pd.DataFrame(point_rows), absent)


def cost_summary(points: pd.DataFrame) -> pd.DataFrame:
    """Per ``(kind, mode, how)`` and step: ``n``, ``median_s``, ``p90_s`` (linear
    interpolation), ``median_core_s`` (seconds × threads; NaN when the threads are absent) and
    ``core_hours`` (the sum)."""
    cols = ["kind", "mode", "how", "step", "n", "median_s", "p90_s", "median_core_s", "core_hours"]
    if points.empty:
        return pd.DataFrame(columns=cols)
    rows: list[dict[str, Any]] = []
    order = {m: i for i, m in enumerate(COST_MODES)}
    keys = sorted(
        {
            (str(k), str(m), str(h))
            for k, m, h in zip(points["kind"], points["mode"], points["how"])
        },
        key=lambda t: (t[0] != "full", order.get(t[1], len(order)), t[1], t[2]),
    )
    for kind, mode, how in keys:
        sub = points[(points["kind"] == kind) & (points["mode"] == mode) & (points["how"] == how)]
        for step in COST_STEPS:
            v = sub[step].to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            c = sub[f"core_{step}"].to_numpy(dtype=float)
            c = c[np.isfinite(c)]
            rows.append(
                {
                    "kind": kind,
                    "mode": mode,
                    "how": how,
                    "step": step,
                    "n": int(v.size),
                    "median_s": float(np.median(v)) if v.size else math.nan,
                    "p90_s": float(np.percentile(v, 90)) if v.size else math.nan,
                    "median_core_s": float(np.median(c)) if c.size else math.nan,
                    "core_hours": float(c.sum()) / 3600.0 if c.size else math.nan,
                }
            )
    return pd.DataFrame(rows, columns=cols)


def _fmt(v: Any, digits: int = 1) -> str:
    if v is None:
        return "absent"
    if isinstance(v, bool | np.bool_):
        return str(bool(v))
    if isinstance(v, int | np.integer):
        return str(int(v))
    if isinstance(v, float | np.floating):
        return "absent" if not math.isfinite(float(v)) else f"{float(v):.{digits}f}"
    return str(v)


def _md_table(header: Sequence[str], rows: Sequence[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def format_cost_report(records: CostRecords, summary: pd.DataFrame, wall_s: float) -> str:
    """The markdown report: runs, totals, per-point cost by kind / mode / how and step, the
    fields absent from older records, and the definitions."""
    runs, pts = records.runs, records.points
    lines = [f"# Measured per-point cost — store `{records.root}`", ""]
    if runs.empty:
        lines += ["No run records under `results/runs/` — nothing to report.", ""]
        lines.append(f"Report wall clock {wall_s:.2f} s; nothing recalibrated (reads JSON only).")
        return "\n".join(lines)
    lines += ["## Runs", ""]
    header = [
        "run",
        "host",
        "shard",
        "workers",
        "threads/worker",
        "tier",
        "particles",
        "paths",
        "wall h",
        "core-h",
        "computed",
        "skipped",
        "failed",
        "calibrated",
        "cache hits",
        "hit rate",
        "peak RSS GiB",
    ]
    rows = []
    for _, r in runs.iterrows():
        th = _fmt(r["threads_per_worker"], 0)
        if r["threads_source"] == "assumed":
            th += " (assumed)"
        rows.append(
            [
                str(r["run"]),
                _fmt(r["host"]),
                _fmt(r["shard"]),
                _fmt(r["workers"]),
                th,
                _fmt(r["risk_tier"]),
                _fmt(r["n_particles"]),
                _fmt(r["n_paths"]),
                _fmt(r["wall_seconds"] / 3600.0, 3),
                _fmt(r["core_hours"], 3),
                _fmt(r["points_computed"]),
                _fmt(r["points_skipped"]),
                _fmt(r["points_failed"]),
                _fmt(r["n_calibrated"]),
                _fmt(r["n_cache_hits"]),
                _fmt(r["cache_hit_rate"], 3),
                _fmt(r["peak_rss_gib"], 2),
            ]
        )
    lines += _md_table(header, rows)
    known = runs.dropna(subset=["n_calibrated", "n_cache_hits"])
    if known.empty:
        hits = "calibrated / cache hits absent from every record"
    else:
        n_cal = int(known["n_calibrated"].astype(int).sum())
        n_hit = int(known["n_cache_hits"].astype(int).sum())
        rate = _fmt(n_hit / (n_cal + n_hit), 3) if n_cal + n_hit else "n/a (no leverage point)"
        hits = (
            f"calibrated {n_cal} / cache hits {n_hit} (hit rate {rate}) over the {len(known)} "
            f"runs that record both ({len(runs) - len(known)} do not)"
        )
    core = runs["core_hours"].to_numpy(dtype=float)
    n_core = int(np.isfinite(core).sum())
    cores = (
        f"core-hours {float(np.nansum(core)):.3f} over the {n_core} runs with a thread count"
        if n_core
        else "core-hours absent (no run records its thread count; see --assume-threads)"
    )
    lines += [
        "",
        f"**Total**: {len(runs)} runs, wall {runs['wall_seconds'].sum() / 3600.0:.3f} h summed "
        f"over runs (concurrent shards overlap in time), "
        f"{int(runs['points_computed'].sum())} points computed; {hits}; {cores}.",
        "",
        "## Per-point cost by mode and step (seconds; core-s = seconds x threads per worker)",
        "",
    ]
    rows = []
    for _, r in summary.iterrows():
        # a step no entry of the group carries (n = 0) is shown as absent, never as 0
        rows.append(
            [
                str(r["kind"]),
                str(r["mode"]),
                str(r["how"]),
                str(r["step"]),
                str(int(r["n"])),
                _fmt(r["median_s"]),
                _fmt(r["p90_s"]),
                _fmt(r["median_core_s"]),
                _fmt(r["core_hours"], 3),
            ]
        )
    lines += _md_table(
        ["kind", "mode", "how", "step", "n", "median s", "p90 s", "median core-s", "core-h"],
        rows,
    )
    unsplit = int((~pts["calibration_split"].astype(bool)).sum()) if not pts.empty else 0
    lines += ["", "## Fields absent from the records (reported, never filled in)", ""]
    any_absent = False
    for run, fields in records.absent.items():
        if fields:
            any_absent = True
            lines.append(f"- `{run}`: {', '.join(fields)}")
    if not any_absent:
        lines.append("- none")
    if not pts.empty:
        derived = pts[pts["mode_source"] != "record"]
        if not derived.empty:
            srcs = derived["mode_source"].value_counts().to_dict()
            lines.append(f"- mode not in the record for {len(derived)} entries: taken from {srcs}")
    lines += [
        "",
        "## Definitions",
        "",
        "- *calibration* = the leverage's particle pass (`calibration_seconds` of the store row) "
        "for a point calibrated in the run, 0 for a cache hit or the LV point; *overhead* = the "
        "rest of the timed calibration step (market build, leverage load / Dupire, mean |L-1|, "
        "marking fit) + the untimed remainder (store write).",
        f"- {unsplit} entries have the calibration step unsplit (no `calibrated` flag in the "
        "record, or no store row): their *calibration* includes the overhead.",
        "- *hit rate* = cache hits / (calibrated + cache hits), points with a leverage only.",
        "- *core-h* of a run = wall x workers x threads per worker (cores held, not cores busy).",
        "- *peak RSS* = the largest `ru_maxrss` of the run's processes (per worker process).",
        "",
        f"Report wall clock {wall_s:.2f} s; nothing recalibrated (reads the store's JSON and "
        "points rows only).",
    ]
    return "\n".join(lines)


def build_report_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="volsto-precompute report",
        description="Measured per-point cost and cache hit rate from a store's run records "
        "(results/runs/*.json); reads only, calibrates and simulates nothing.",
    )
    p.add_argument("--store", default=str(DEFAULT_STORE), help="results store root")
    p.add_argument(
        "--out",
        default=None,
        help=f"directory for cost_report.md, runs.csv, points.csv, summary.csv (default "
        f"<store>/{COST_REPORT_DIRNAME}, outside results/)",
    )
    p.add_argument("--no-write", action="store_true", help="print the markdown, write no file")
    p.add_argument(
        "--assume-threads",
        type=int,
        default=None,
        help="thread count per worker for records that predate the 'threads' fields (labelled "
        "'assumed' in the report); without it their core-seconds are reported absent",
    )
    return p


def report_main(argv: Sequence[str]) -> int:
    """``volsto-precompute report``: print the markdown report and write it with the three CSVs."""
    args = build_report_parser().parse_args(list(argv))
    t0 = time.perf_counter()
    root = Path(args.store)
    if not (root / "results").exists():
        print(f"error: no results store under {root.resolve()}", file=sys.stderr)
        return 2
    records = load_cost_records(root, assume_threads=args.assume_threads)
    summary = cost_summary(records.points)
    text = format_cost_report(records, summary, time.perf_counter() - t0)
    print(text)
    if not args.no_write:
        out = Path(args.out) if args.out else root / COST_REPORT_DIRNAME
        out.mkdir(parents=True, exist_ok=True)
        (out / "cost_report.md").write_text(text + "\n")
        records.runs.to_csv(out / "runs.csv", index=False)
        records.points.to_csv(out / "points.csv", index=False)
        summary.to_csv(out / "summary.csv", index=False)
        print(f"\nwritten: {out.resolve()}/{{cost_report.md, runs.csv, points.csv, summary.csv}}")
    return 0


@dataclass(frozen=True)
class RiskBudgetCosts:
    """What the stored risk steps of a store measure (:func:`risk_budget_costs`), unscaled:
    per-state work outside the engine and pure per-pricing costs by mode group."""

    state_s: list[float]
    lsv_per_pricing: list[float]
    lv_per_pricing: list[float]
    notes: list[str]


def risk_budget_costs(
    store: ResultsStore, points: pd.DataFrame, particle_pass_s: float
) -> RiskBudgetCosts:
    """Per stored risk step: the engine budget of the point's manifest (``risk_budget``: states
    = ``recalibrations``, ``cache_misses``, ``pricings``, the engine's ``wall_clock_s``) and the
    step's wall clock in the run records (``points``, the last entry of the point with a risk
    wall).  Per point, by mode:

    * state overhead = ``(step wall − engine wall) / states`` (any mode);
    * LSV (``one_factor`` / ``two_factor`` / ``marking``): ``(engine wall − cache misses ×
      particle_pass_s − states × RISK_STATE_BUILD_S["lsv"]) / pricings`` — a miss is a particle
      pass at the source's budget; a budget with misses is skipped when ``particle_pass_s`` is
      not finite;
    * LV: ``(engine wall − states × RISK_STATE_BUILD_S["lv"]) / pricings`` — its "misses" are
      the Dupire rebuilds, not particle passes.

    A point whose mode is not known, or whose budget has no pricing, is skipped and said so."""
    state_s: list[float] = []
    lsv: list[float] = []
    lv: list[float] = []
    notes: list[str] = []
    step_wall: dict[str, float] = {}
    modes: dict[str, str] = {}
    if not points.empty:
        for _, e in points.iterrows():
            modes[str(e["point_id"])] = str(e["mode"])
            if math.isfinite(float(e["risk"])) and float(e["risk"]) > 0:
                step_wall[str(e["point_id"])] = float(e["risk"])
    for pid, entry in store.point_manifests().items():
        b = entry.get("risk_budget")
        if not b or not float(b.get("pricings", 0.0)):
            continue
        mode = modes.get(pid) or _stored_mode(store, pid)
        engine = float(b.get("wall_clock_s", math.nan))
        n_pr = float(b["pricings"])
        states = float(b.get("recalibrations", math.nan))
        misses = float(b.get("cache_misses", math.nan))
        if pid in step_wall and math.isfinite(engine) and states > 0:
            state_s.append(max(step_wall[pid] - engine, 0.0) / states)
        if mode == "lv":
            lv.append(max(engine - states * RISK_STATE_BUILD_S["lv"], 0.0) / n_pr)
        elif mode in LSV_MODES:
            if not math.isfinite(misses) or (misses and not math.isfinite(particle_pass_s)):
                notes.append(f"risk budget of {pid[:16]} skipped: no particle pass to subtract")
                continue
            builds = states * RISK_STATE_BUILD_S["lsv"]
            lsv.append(max(engine - misses * particle_pass_s - builds, 0.0) / n_pr)
        else:
            notes.append(f"risk budget of {pid[:16]} skipped: mode {mode!r} unknown")
    return RiskBudgetCosts(state_s, lsv, lv, notes)


def _stored_mode(store: ResultsStore, point_id: str) -> str:
    if _mode_from_id(point_id) is not None:
        return str(_mode_from_id(point_id))
    try:
        return str(_stored_row(store, point_id).get("mode", ""))
    except (OSError, ValueError, IndexError, KeyError):  # pragma: no cover - a half-written point
        return ""


def cost_model_from_store(
    source: str | Path,
    grid: GridSpec,
    fallback: CostModel,
    *,
    target_threads: int,
    source_threads: int | None = None,
) -> tuple[CostModel, list[str]]:
    """The :class:`CostModel` of ``grid`` built from the measured full computations of a previous
    store (``--cost-from``), and one line per number saying where it comes from.

    From the source's full-computation entries (:func:`load_cost_records`): calibration = the
    median particle pass of the calibrated entries × ``grid particles / source particles``;
    diagnostics = their median × ``grid paths / source paths``; overhead = the median overhead
    (not rescaled: market build and store write do not depend on the budget); pricing and
    analytics = the median over the calibrating modes × the path ratio, with per-mode values
    (:attr:`CostModel.by_mode`) for every mode measured; the risk step from the stored risk
    budgets by mode (:func:`risk_budget_costs`: the per-state overhead, not rescaled, and the
    LSV and LV risk pricings × the path ratio), else the measured probe-C constants
    :data:`RISK_STATE_OVERHEAD_S` and :data:`RISK_PRICING_RATIO` × that mode's pricing.  The
    linear rules are the existing ones of :func:`default_cost_model`.  Every number is then
    rescaled from the source thread count to ``target_threads`` with :func:`thread_factor`; a
    source without a thread count (and no ``source_threads``) is not rescaled, and the lines say
    so.  A quantity the source did not measure keeps ``fallback``'s value, named as such."""
    rec = load_cost_records(source, assume_threads=source_threads)
    pts = rec.points
    lines = [f"cost model from the measured store {Path(source).resolve()}"]
    if pts.empty:
        lines.append("  no point entries in its run records: the default cost model is kept")
        return fallback, lines
    full = pts[pts["kind"].isin(["full", "unknown"])]
    if full.empty:
        lines.append("  no full computation in its run records: the default cost model is kept")
        return fallback, lines
    th = full["threads"].to_numpy(dtype=float)
    th = th[np.isfinite(th)]
    src_threads: int | None = int(np.median(th)) if th.size else None
    if src_threads is not None and len(set(th.tolist())) > 1:
        lines.append(f"  source thread counts differ {sorted(set(th.tolist()))}: median used")
    if src_threads is None:
        lines.append(
            "  source thread count ABSENT from the records (pass --cost-from-threads N): no "
            f"thread rescaling to the target {target_threads} — the projection below is at the "
            "source's unknown thread count"
        )
    else:
        how = "assumed by --cost-from-threads" if source_threads is not None else "recorded"
        lines.append(
            f"  threads: source {src_threads} ({how}) -> target {target_threads}; rescaled by "
            "the measured laptop thread ratios (MEASURED_THREAD_RATIO; a second thread on an "
            "SMT sibling gains less than the measured real core)"
        )

    def tf(step: str) -> float:
        return 1.0 if src_threads is None else thread_factor(step, src_threads, target_threads)

    def med(frame: pd.DataFrame, col: str) -> float:
        v = frame[col].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        return float(np.median(v)) if v.size else math.nan

    def ratio(frame: pd.DataFrame, col: str, target: float) -> float:
        src = med(frame, col)
        return target / src if math.isfinite(src) and src > 0 else math.nan

    path_r = ratio(full, "n_paths", float(grid.pricing.n_paths))
    lines.append(
        f"  budget: source {med(full, 'n_particles'):.0f} particles / {med(full, 'n_paths'):.0f} "
        f"paths -> grid {grid.particle.n_particles} / {grid.pricing.n_paths} (linear rules)"
    )
    cal = full[(full["how"] == "calibrated") & full["calibration_split"].astype(bool)]
    calibration_s, cal_src = fallback.calibration_s, f"kept: {fallback.calibration_source}"
    diagnostics_s, diag_src = fallback.diagnostics_s, "kept: measured fallback"
    if not cal.empty:
        part_r = ratio(cal, "n_particles", float(grid.particle.n_particles))
        if math.isfinite(part_r):
            calibration_s = med(cal, "calibration") * part_r * tf("calibration")
            cal_src = (
                f"median particle pass {med(cal, 'calibration'):.1f} s of {len(cal)} calibrated "
                f"entries x {part_r:g} (particles) x {tf('calibration'):.3f} (threads)"
            )
        if math.isfinite(path_r):
            diagnostics_s = med(cal, "diagnostics") * path_r * tf("diagnostics")
            diag_src = (
                f"median {med(cal, 'diagnostics'):.1f} s of {len(cal)} calibrated entries x "
                f"{path_r:g} (paths) x {tf('diagnostics'):.3f} (threads)"
            )
    overhead_s = med(full, "overhead") * tf("overhead")
    lines.append(f"  calibration {calibration_s:.1f} s per miss — {cal_src}")
    lines.append(f"  diagnostics {diagnostics_s:.1f} s per miss — {diag_src}")
    lines.append(
        f"  overhead {overhead_s:.1f} s per point — median of {len(full)} entries (not rescaled)"
    )
    lsv = full[full["mode"].isin(["one_factor", "two_factor", "marking"])]
    base = lsv if not lsv.empty else full
    pricing_s, analytics_s = fallback.pricing_s, fallback.analytics_s
    pr_src = f"kept: {fallback.pricing_source}"
    if math.isfinite(path_r):
        pricing_s = med(base, "pricing") * path_r * tf("pricing")
        analytics_s = med(base, "analytics") * path_r * tf("analytics")
        pr_src = (
            f"median of {len(base)} {'LSV' if not lsv.empty else ''} entries "
            f"({med(base, 'pricing'):.1f} / {med(base, 'analytics'):.1f} s) x {path_r:g} (paths) "
            f"x {tf('pricing'):.3f} / {tf('analytics'):.3f} (threads)"
        )
    lines.append(f"  pricing / analytics {pricing_s:.1f} / {analytics_s:.1f} s — {pr_src}")
    by_mode: dict[str, dict[str, float]] = {}
    if math.isfinite(path_r):
        for mode in COST_MODES:
            sub = full[full["mode"] == mode]
            if sub.empty:
                continue
            by_mode[mode] = {
                "overhead": med(sub, "overhead") * tf("overhead"),
                "pricing": med(sub, "pricing") * path_r * tf("pricing"),
                "analytics": med(sub, "analytics") * path_r * tf("analytics"),
            }
            lines.append(
                f"  mode {mode}: overhead / pricing / analytics "
                f"{by_mode[mode]['overhead']:.1f} / {by_mode[mode]['pricing']:.1f} / "
                f"{by_mode[mode]['analytics']:.1f} s ({len(sub)} entries)"
            )
    cal_src_s = med(cal, "calibration") if not cal.empty else math.nan
    budgets = risk_budget_costs(ResultsStore(source), pts, cal_src_s)
    lines.extend(f"  {note}" for note in budgets.notes)
    outside_s = RISK_STATE_OVERHEAD_S
    state_src = f"{RISK_STATE_OVERHEAD_S:.2f} s outside the engine (RISK_STATE_OVERHEAD_S)"
    if budgets.state_s:
        outside_s = float(np.median(budgets.state_s))
        state_src = (
            f"{outside_s:.2f} s outside the engine, median of {len(budgets.state_s)} stored risk "
            "steps (step wall - engine wall) / states"
        )
    state_src += (
        f" + a state build of {RISK_STATE_BUILD_S['lsv']:.2f} s (LSV) / "
        f"{RISK_STATE_BUILD_S['lv']:.2f} s (LV) (RISK_STATE_BUILD_S); budget-independent and "
        "serial, not rescaled"
    )
    risk_state_s = outside_s + RISK_STATE_BUILD_S["lsv"]
    lv_measured = "pricing" in by_mode.get("lv", {})
    lv_pricing_s = by_mode["lv"]["pricing"] if lv_measured else LV_TO_LSV_PRICING_RATIO * pricing_s
    ref_names = {
        "lsv": "LSV pricing",
        "lv": (
            "measured LV pricing"
            if lv_measured
            else f"LV pricing (LSV x {LV_TO_LSV_PRICING_RATIO:.4f}, LV_TO_LSV_PRICING_RATIO)"
        ),
    }
    risk_by_mode: dict[str, tuple[float, str]] = {}
    for group, measured, ref_s in (
        ("lsv", budgets.lsv_per_pricing, pricing_s),
        ("lv", budgets.lv_per_pricing, lv_pricing_s),
    ):
        if measured and math.isfinite(path_r):
            v = float(np.median(measured))
            risk_by_mode[group] = (
                v * path_r * tf("risk"),
                f"median of {len(measured)} stored {group.upper()} risk budgets ({v:.3f} "
                f"s/pricing net of state builds) x {path_r:g} (paths) x {tf('risk'):.3f} "
                "(threads)",
            )
        else:
            risk_by_mode[group] = (
                RISK_PRICING_RATIO[group] * ref_s,
                f"no {group.upper()} risk budget in the source: {RISK_PRICING_RATIO[group]:.4f} "
                f"(RISK_PRICING_RATIO) x the {ref_names[group]} {ref_s:.1f} s",
            )
    risk_pricing_s = risk_by_mode["lsv"][0]
    by_mode.setdefault("lv", {})["risk_pricing"] = risk_by_mode["lv"][0]
    by_mode["lv"]["risk_state"] = outside_s + RISK_STATE_BUILD_S["lv"]
    lines.append(
        f"  risk state {risk_state_s:.2f} s (LSV) / {by_mode['lv']['risk_state']:.2f} s (LV) per "
        f"bumped state — {state_src}"
    )
    for group, (v, src) in risk_by_mode.items():
        lines.append(f"  risk pricing ({group.upper()}) {v:.2f} s each — {src}")
    lines.append(
        "  risk calibrations: (states - 1) x the calibration cost above on an LSV point (the "
        "base state is the point's own leverage); an LV point's Dupire rebuilds are in its risk "
        "pricing"
    )
    risk_src = (
        f"--cost-from {Path(source).name}: states {state_src}; LSV {risk_by_mode['lsv'][1]}; "
        f"LV {risk_by_mode['lv'][1]}"
    )
    cost = dataclasses.replace(
        fallback,
        overhead_s=overhead_s,
        calibration_s=calibration_s,
        calibration_source=f"--cost-from: {cal_src}",
        diagnostics_s=diagnostics_s,
        pricing_s=pricing_s,
        analytics_s=analytics_s,
        risk_pricing_s=risk_pricing_s,
        pricing_source=f"--cost-from {Path(source).name}: {pr_src}",
        risk_state_s=risk_state_s,
        risk_source=risk_src,
        by_mode=by_mode,
    )
    return cost, lines


# --------------------------------------------------------------------------------------------
# workers, provenance, CLI
# --------------------------------------------------------------------------------------------


def numba_threads() -> int:
    """The numba thread count of this process's parallel regions (``numba.get_num_threads()``:
    ``NUMBA_NUM_THREADS`` at import — else the logical CPU count — unless lowered by
    ``--threads-per-worker``) — recorded per point and per run so the cost report can charge
    core-seconds."""
    import numba

    return int(numba.get_num_threads())  # type: ignore[no-untyped-call]


def peak_rss_bytes() -> int:
    """Peak resident set size of this process so far, in bytes (``getrusage`` ``ru_maxrss``: bytes
    on macOS, kilobytes on Linux) — the number that caps the worker count on a VM."""
    import resource

    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss if sys.platform == "darwin" else rss * 1024


@dataclass(frozen=True)
class _Task:
    grid_path: str
    point_id: str
    store_root: str
    cache_root: str
    tier: str
    n_particles: int | None
    steps: tuple[str, ...]
    diagnostics: bool


def _apply_overrides(grid: GridSpec, n_particles: int | None) -> GridSpec:
    if n_particles is None:
        return grid
    return dataclasses.replace(
        grid, particle=dataclasses.replace(grid.particle, n_particles=int(n_particles))
    )


def _failure(point: GridPoint, exc: BaseException) -> dict[str, Any]:
    tb = traceback.format_exc()
    log.error("%s: FAILED — %s: %s\n%s", point.label, type(exc).__name__, exc, tb)
    return {
        "point_id": point.id,
        "label": point.label,
        "error": f"{type(exc).__name__}: {exc}",
        "traceback": tb,
    }


def _run_task(task: _Task) -> dict[str, Any]:
    """Worker entry: re-enumerate the grid, compute or refresh the named point, write it; a
    failure comes back as a ``failed`` entry instead of tearing the pool down."""
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    grid = _apply_overrides(load_grid(task.grid_path), task.n_particles)
    points = {p.id: p for p in enumerate_points(grid)}
    point = points[task.point_id]
    cache = LeverageCache(task.cache_root)
    store = ResultsStore(task.store_root)
    try:
        return process_point(
            point, grid, cache, task.tier, store, task.steps, diagnostics=task.diagnostics
        )
    except Exception as exc:
        return _failure(point, exc)


def grid_path_for_record(path: Path) -> str:
    """The grid path as the run record stores it: relative to the repository root when inside
    it; otherwise the basename (with a warning — such a grid is not relocatable with the store)."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        log.warning(
            "grid %s lies outside the repository %s: the run record keeps its basename only",
            resolved,
            REPO_ROOT,
        )
        return resolved.name


def provenance_argv(argv: Sequence[str], grid_display: str) -> list[str]:
    """``argv`` with the path-valued entries replaced: ``--store`` → ``<store>``, ``--cache`` →
    ``<cache>``, ``--grid`` → ``grid_display`` (``--flag=value`` forms included)."""
    marks = {"--store": "<store>", "--cache": "<cache>", "--grid": grid_display}
    out: list[str] = []
    pending: str | None = None
    for a in argv:
        if pending is not None:
            out.append(pending)
            pending = None
            continue
        flag, eq, _ = a.partition("=")
        if flag in marks:
            if eq:
                out.append(f"{flag}={marks[flag]}")
            else:
                out.append(a)
                pending = marks[flag]
            continue
        out.append(a)
    return out


def select_points(
    points: Sequence[GridPoint], ids: Sequence[str], all_points: Sequence[GridPoint]
) -> list[GridPoint]:
    """``--only``: the points of the shard named by ``ids`` (in shard order); an id that is not
    in the grid raises ``ValueError`` listing the nearest ids, one in the grid but not in this
    shard raises too."""
    by_id = {p.id: p for p in points}
    grid_ids = [p.id for p in all_points]
    wanted = set(ids)
    unknown = [i for i in ids if i not in grid_ids]
    if unknown:
        lines = []
        for i in unknown:
            near = difflib.get_close_matches(i, grid_ids, n=3, cutoff=0.0)
            prefix = [g for g in grid_ids if g.startswith(i)]
            cands = list(dict.fromkeys(prefix + near))[:3]
            lines.append(f"  {i!r}: nearest {cands}")
        raise ValueError("--only: unknown point id(s) (not in the grid):\n" + "\n".join(lines))
    off_shard = [i for i in ids if i not in by_id]
    if off_shard:
        raise ValueError(
            f"--only: {len(off_shard)} point(s) belong to another shard of this grid: "
            f"{[i[:16] for i in off_shard]}"
        )
    return [p for p in points if p.id in wanted]


def _set_numba_threads(n: int) -> str:
    """Lower this process's numba thread count to ``n`` (``numba.set_num_threads``); the error
    message when ``n`` exceeds the pool numba started with (``NUMBA_NUM_THREADS`` at import),
    else ``""``."""
    import numba

    ceiling = int(numba.config.NUMBA_NUM_THREADS)  # type: ignore[attr-defined]
    if n > ceiling:
        return (
            f"--threads-per-worker {n} exceeds NUMBA_NUM_THREADS {ceiling} of this process "
            f"(export NUMBA_NUM_THREADS={n} instead)"
        )
    numba.set_num_threads(n)  # type: ignore[no-untyped-call]
    return ""


def _physical_memory_gb() -> float:
    """Physical memory of this machine in GiB (``sysconf``; NaN where unavailable)."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30
    except (ValueError, OSError, AttributeError):  # pragma: no cover - platform dependent
        return math.nan


def _parallel_line(
    process_s: float,
    longest_s: float,
    n_points: int,
    tier: str,
    shard_label: str,
    workers: int,
    threads: int,
) -> str:
    """The projected wall clock of **the points this invocation computes** (the shard, after
    ``--only`` / ``--resume`` / ``--limit``) on ``workers`` processes: their process-hours
    divided by the workers, or the longest single point when that is longer — a lower bound
    (even packing at laptop per-core speed; contention and slower cores lengthen it)."""
    total_h = process_s / 3600.0
    wall_h = max(total_h / max(workers, 1), longest_s / 3600.0)
    return (
        f"parallel (shard {shard_label}, {n_points} points to compute): {workers} worker(s) x "
        f"{threads} thread(s) -> {wall_h:.2f} h wall for the {tier} tier "
        f"({total_h:.2f} process-hours at {threads} thread(s) / {workers}; a lower bound — even "
        "packing, the slowest point sets the tail; the projection table above is the whole grid)"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="volsto-precompute", description=__doc__.split("\n\n")[0] if __doc__ else None
    )
    p.add_argument("--grid", default=str(DEFAULT_GRID), help="grid YAML (GridSpec)")
    p.add_argument("--store", default=str(DEFAULT_STORE), help="results store root")
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="leverage cache root")
    p.add_argument("--shard", default="1/1", help="i/n: the i-th of n interleaved shards")
    p.add_argument(
        "--only",
        nargs="+",
        metavar="ID",
        default=None,
        help="only these point ids of the shard (cache key, lv:<hash>, marking:<label>)",
    )
    p.add_argument("--workers", type=int, default=1, help="processes over the shard's points")
    p.add_argument(
        "--resume",
        action="store_true",
        help="skip points done at the requested tier; refresh the risk (and, with "
        "--diagnostics, the diagnostics) step of points stored below it",
    )
    p.add_argument("--risk", choices=RISK_TIERS, default=None, help="risk tier (grid default)")
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="backfill the calibration diagnostics of cache hits that have none (pillar "
        "repricing at the pricing SimConfig; no calibration)",
    )
    p.add_argument("--n-particles", type=int, default=None, help="override the grid's count")
    p.add_argument("--dry-run", action="store_true", help="enumerate and project, compute nothing")
    p.add_argument(
        "--probe",
        action="store_true",
        help="measure the per-point cost on the first point of the shard, then project",
    )
    p.add_argument("--limit", type=int, default=None, help="process at most N points")
    p.add_argument(
        "--threads-per-worker",
        type=int,
        default=None,
        help="numba threads per process (default: cpu_count // workers with --workers > 1, "
        "else NUMBA_NUM_THREADS); also the thread count --cost-from rescales to",
    )
    p.add_argument(
        "--cost-from",
        default=None,
        metavar="ROOT",
        help="build the cost model from the measured run records of a previous store (by mode "
        "and step, rescaled by particles, paths and threads) instead of the cache manifest and "
        "the measured fallbacks",
    )
    p.add_argument(
        "--cost-from-threads",
        type=int,
        default=None,
        help="thread count the --cost-from store ran at, for records that predate the "
        "'threads' fields (stated, not inferred)",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    """The CLI; ``volsto-precompute report ...`` dispatches to :func:`report_main`."""
    raw = list(argv) if argv is not None else sys.argv[1:]
    if raw and raw[0] == "report":
        return report_main(raw[1:])
    args = build_parser().parse_args(raw)
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t_run = time.perf_counter()
    grid_path = Path(args.grid)
    grid = _apply_overrides(load_grid(grid_path), args.n_particles)
    tier = args.risk or grid.risk.tier
    diagnostics = bool(args.diagnostics)
    i, n = parse_shard(args.shard)
    cache = LeverageCache(args.cache)
    store = ResultsStore(args.store)
    points = enumerate_points(grid)
    mine = shard(points, i, n)
    counts = count_by(points)
    grid_display = grid_path_for_record(grid_path)

    print(
        f"grid {grid.name!r} ({grid_display}): {len(points)} points, {grid.particle.n_particles} "
        f"particles, horizon {grid.particle.horizon:g}y, pricing {grid.pricing.n_paths} paths "
        f"seed {grid.pricing.seed}, risk tier {tier} ({grid.risk.fwd_var_buckets}-bucket fwd-var "
        f"ladder, skew pillars {list(grid.risk.light_pillars)}); shard {i}/{n}: {len(mine)} points"
    )
    print(f"  store {store.root.resolve()}\n  cache {cache.root.resolve()}")
    for (surface, mode), c in counts.items():
        print(f"  {surface:<20} {mode:<11} {c:5d}")
    if grid.one_factor is not None:
        surfs = [s.name for s in grid.surfaces if s.one_factor]
        n_zero = (
            (1 if 0.0 in grid.one_factor.nu else 0)
            * len(grid.one_factor.rho)
            * len(grid.one_factor.kappa)
        )
        print(
            f"  one_factor axes: {len(grid.one_factor.nu)} nu x {len(grid.one_factor.rho)} rho x "
            f"{len(grid.one_factor.kappa)} kappa = {grid.one_factor.n_combinations} combinations "
            f"per surface on {surfs}: {grid.one_factor.n_combinations - n_zero} LSV calibrations"
            f" + {n_zero} ω=0 combinations collapsing onto the surface's single LV point "
            "(pure Dupire, no calibration); the degenerate ω=1,2,3 points are included"
        )

    workers = max(1, int(args.workers))
    if args.threads_per_worker is not None and args.threads_per_worker < 1:
        print("error: --threads-per-worker must be >= 1", file=sys.stderr)
        return 2
    if workers > 1:
        threads_per_worker = args.threads_per_worker or max(1, (os.cpu_count() or 1) // workers)
    else:
        threads_per_worker = args.threads_per_worker or numba_threads()
        if not args.dry_run and threads_per_worker != numba_threads():
            error = _set_numba_threads(threads_per_worker)
            if error:
                print(f"error: {error}", file=sys.stderr)
                return 2
    cost = default_cost_model(cache, grid)
    if args.cost_from:
        if not (Path(args.cost_from) / "results" / "runs").exists():
            print(f"error: --cost-from: no run records under {args.cost_from}", file=sys.stderr)
            return 2
        cost, cost_lines = cost_model_from_store(
            args.cost_from,
            grid,
            cost,
            target_threads=threads_per_worker,
            source_threads=args.cost_from_threads,
        )
        print("\n".join(cost_lines))
        peak = load_cost_records(args.cost_from).points["peak_rss_bytes"].to_numpy(dtype=float)
        peak = peak[np.isfinite(peak)]
        if peak.size:
            print(
                f"  memory: measured peak RSS {peak.max() / 2**30:.2f} GiB per process x {workers} "
                f"workers = {peak.max() * workers / 2**30:.1f} GiB (this machine: "
                f"{_physical_memory_gb():.1f} GiB)"
            )
        else:
            print("  memory: peak RSS absent from the source records")
    selected = mine
    if args.only:
        try:
            selected = select_points(mine, list(args.only), points)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(f"only: {len(selected)} of the shard's {len(mine)} points selected")
    plan: dict[str, tuple[str, ...]] = {}
    n_skipped = 0
    if args.resume:
        n_skip = n_refresh = 0
        for p in selected:
            steps = pending_steps(p, store, cache, tier, diagnostics=diagnostics)
            if not steps:
                n_skip += 1
                log.info("resume: skipping %s (store row and cache entry present)", p.label)
            elif "all" not in steps:
                n_refresh += 1
                log.info(
                    "resume: %s is stored — refresh %s only (risk tier %s)",
                    p.label,
                    list(steps),
                    tier,
                )
            plan[p.id] = steps
        todo = [p for p in selected if plan[p.id]]
        n_skipped = n_skip
        print(
            f"resume: {n_skip} points already done, {len(todo)} to compute "
            f"({n_refresh} of them refreshed for {list(REFRESH_STEPS)} steps only)"
        )
    else:
        todo = list(selected)
        plan = {p.id: ("all",) for p in todo}
    if args.limit is not None:
        todo = todo[: max(0, args.limit)]

    projection = projection_table(points, cache, cost, grid.risk, diagnostics=diagnostics)

    def point_cost(p: GridPoint) -> float:
        return cost.steps_s(
            tier,
            plan[p.id],
            calibrates=p.calibrates,
            miss=not _hit(cache, p),
            backfill=(
                diagnostics
                and p.calibrates
                and _hit(cache, p)
                and not _has_report(cache, p.cache_key)
            ),
            mode=p.mode,
        )

    if args.dry_run:
        print(format_projection(projection, cost, tier, grid.risk))
        shard_s = sum(point_cost(p) for p in todo)
        longest = max((point_cost(p) for p in todo), default=0.0)
        print(
            _parallel_line(
                shard_s, longest, len(todo), tier, f"{i}/{n}", workers, threads_per_worker
            )
        )
        print(
            f"this shard ({i}/{n}): {len(todo)} points to compute, projected "
            f"{shard_s / 3600.0:.2f} process-hours at {threads_per_worker} thread(s) "
            f"({shard_s / 3600.0 / workers:.2f} h wall on {workers} worker(s), lower bound; "
            f"longest single point {longest / 3600.0:.2f} h)"
        )
        print("dry run: nothing computed")
        return 0

    done_walls: list[dict[str, Any]] = []
    failed: list[dict[str, Any]] = []
    if args.probe and todo:
        first = todo[0]
        print(f"probe: measuring the per-point cost on {first.label!r}")
        try:
            w = process_point(
                first, grid, cache, tier, store, plan[first.id], diagnostics=diagnostics
            )
        except Exception as exc:
            w = _failure(first, exc)
        if "error" in w:
            failed.append(w)
        else:
            done_walls.append(w)
            if "all" in plan[first.id]:
                entry = store.read_point(first.id)
                probe_pts = pd.DataFrame(
                    [{"point_id": first.id, "mode": first.mode, "risk": float(w["risk"])}]
                )
                particle = float(entry.row.get("calibration_seconds", math.nan))
                if not w["calibrated"]:
                    particle = cost.calibration_s
                measured = risk_budget_costs(store, probe_pts, particle)
                by_mode = {m: dict(v) for m, v in cost.by_mode.items()}
                risk_pricing = cost.risk_pricing_s
                risk_state = cost.risk_state_s
                if measured.state_s:
                    risk_state = measured.state_s[0] + RISK_STATE_BUILD_S["lsv"]
                    by_mode.setdefault("lv", {})["risk_state"] = (
                        measured.state_s[0] + RISK_STATE_BUILD_S["lv"]
                    )
                if first.mode == "lv" and measured.lv_per_pricing:
                    by_mode.setdefault("lv", {})["risk_pricing"] = measured.lv_per_pricing[0]
                elif measured.lsv_per_pricing:
                    risk_pricing = measured.lsv_per_pricing[0]
                cost = dataclasses.replace(
                    cost,
                    calibration_s=(
                        float(entry.row["calibration_seconds"])
                        if w["calibrated"]
                        else cost.calibration_s
                    ),
                    calibration_source=(
                        f"probe on {first.label!r}" if w["calibrated"] else cost.calibration_source
                    ),
                    diagnostics_s=(
                        float(w["diagnostics"]) if w["diagnostics"] > 0 else cost.diagnostics_s
                    ),
                    pricing_s=float(w["pricing"]),
                    analytics_s=float(w["analytics"]),
                    risk_pricing_s=risk_pricing,
                    risk_state_s=risk_state,
                    risk_source=(
                        f"probe on {first.label!r} (tier {tier})"
                        if measured.state_s
                        else cost.risk_source
                    ),
                    by_mode=by_mode,
                    pricing_source=f"probe on {first.label!r} (tier {tier})",
                )
                projection = projection_table(
                    points, cache, cost, grid.risk, diagnostics=diagnostics
                )
        todo = todo[1:]
    print(format_projection(projection, cost, tier, grid.risk))
    remaining = sum(point_cost(p) for p in todo)
    print(
        f"this shard: {len(todo)} points to compute, projected "
        f"{remaining / 3600.0:.2f} h at the cost model above"
    )

    t_loop = time.perf_counter()
    n_done = 0

    def report(w: dict[str, Any]) -> None:
        nonlocal n_done
        n_done += 1
        elapsed = time.perf_counter() - t_loop
        mean = elapsed / n_done
        eta = mean * (len(todo) - n_done)
        if "error" in w:
            failed.append(w)
            print(
                f"[{n_done}/{len(todo)}] {w['label']} FAILED: {w['error']}; elapsed "
                f"{elapsed / 60:.1f} min, ETA {eta / 60:.1f} min",
                flush=True,
            )
            return
        done_walls.append(w)
        how = (
            "calibrated" if w["calibrated"] else ("cache hit" if w["cache_hit"] else "no leverage")
        )
        steps = "" if "all" in w["steps"] else f" [refresh {','.join(w['steps'])}]"
        print(
            f"[{n_done}/{len(todo)}] {w['label']}{steps} in {w['total']:.0f} s ({how}; cal "
            f"{w['calibration']:.0f} / diag {w['diagnostics']:.0f} / pricing {w['pricing']:.0f} / "
            f"analytics {w['analytics']:.0f} / risk {w['risk']:.0f}); elapsed "
            f"{elapsed / 60:.1f} min, ETA {eta / 60:.1f} min",
            flush=True,
        )

    if workers == 1 or len(todo) <= 1:
        for p in todo:
            try:
                report(
                    process_point(p, grid, cache, tier, store, plan[p.id], diagnostics=diagnostics)
                )
            except Exception as exc:
                report(_failure(p, exc))
    else:
        threads = threads_per_worker
        previous = os.environ.get("NUMBA_NUM_THREADS")
        os.environ["NUMBA_NUM_THREADS"] = str(threads)
        print(f"workers: {workers} processes, NUMBA_NUM_THREADS={threads} each")
        tasks = [
            _Task(
                str(grid_path),
                p.id,
                str(store.root),
                str(cache.root),
                tier,
                args.n_particles,
                tuple(plan[p.id]),
                diagnostics,
            )
            for p in todo
        ]
        ctx = multiprocessing.get_context("spawn")
        try:
            with ctx.Pool(workers) as pool:
                for w in pool.imap_unordered(_run_task, tasks):
                    report(w)
        finally:
            if previous is None:
                os.environ.pop("NUMBA_NUM_THREADS", None)
            else:
                os.environ["NUMBA_NUM_THREADS"] = previous

    wall = time.perf_counter() - t_run
    n_calibrated = sum(1 for w in done_walls if w["calibrated"])
    n_hits = sum(1 for w in done_walls if w["cache_hit"])
    n_backfilled = sum(1 for w in done_walls if w["diagnostics_backfilled"])
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    record: dict[str, Any] = {
        "argv": provenance_argv(raw_argv, grid_display),
        "grid_path": grid_display,
        "grid": grid_mapping(grid),
        "shard": f"{i}/{n}",
        "only": list(args.only) if args.only else None,
        "workers": workers,
        "threads_per_worker": threads_per_worker,
        "cpu_count": os.cpu_count(),
        "platform": f"{platform.system()} {platform.machine()} {platform.processor()}".strip(),
        "peak_rss_bytes": peak_rss_bytes(),
        "risk_tier": tier,
        "diagnostics": diagnostics,
        "n_particles": grid.particle.n_particles,
        "pricing": {"n_paths": grid.pricing.n_paths, "seed": grid.pricing.seed},
        "code_tag": CALIBRATION_CODE_TAG,
        "git_commit": code_version(),
        "host": platform.node(),
        "cost_model": dataclasses.asdict(cost),
        "risk_plan": {
            k: {"calibrations": cost.risk_calibrations[k], "pricings": cost.risk_pricings[k]}
            for k in cost.risk_calibrations
        },
        "points_total": len(points),
        "points_in_shard": len(mine),
        "points_selected": len(selected),
        "points_skipped": n_skipped,
        "limit": args.limit,
        "cost_from": Path(args.cost_from).name if args.cost_from else None,
        "points_computed": done_walls,
        "n_calibrated": n_calibrated,
        "n_cache_hits": n_hits,
        "n_diagnostics_backfilled": n_backfilled,
        "failed": failed,
        "resume": bool(args.resume),
        "wall_seconds": wall,
        "created_utc": utc_now(),
    }
    store.write_run(record)
    print(
        f"done: {len(done_walls)} points computed ({n_calibrated} calibrated / {n_hits} cache "
        f"hits / {len(done_walls) - n_calibrated - n_hits} without leverage; {n_backfilled} "
        f"diagnostics backfilled), {len(failed)} failed, in {wall / 60:.1f} min; store {store.root}"
    )
    if failed:
        print(
            "failed points (see the traceback in the log and the run record's 'failed' entries; "
            "--resume picks them up):\n  "
            + "\n  ".join(f"{w['label']}: {w['error']}" for w in failed)
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
