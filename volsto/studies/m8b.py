"""M8b hedging studies (SPEC §8.2; the owner's "M8b — hedging studies (studies/m8b.py)" — the
repository keeps library code in ``volsto/studies/m8b.py``, the CLI in ``scripts/m8b.py`` and the
tests in ``tests/test_m8b.py``).

**Reference for everything here:** the SPX 2022-12-30 snapshot marked at ``ssr_target = 1``,
``skew_eps = 0.10`` (``configs/studies/m7_p1_marking/spx_ssr1_eps0.1.yaml``, the M7 P1 marking fit;
NOT the placeholder SSVI).  The **pricing model** is that marking fit — the LSV of the fitted 2F
parameters on the SPX surface, leverage from the cache at :attr:`StudyConfig.n_particles` (the
production convention, 8·10⁵) — unless a study says otherwise.  Every Monte Carlo number carries
its standard error; every task records its wall clock and the calibrations it performed.

**The book** (:func:`StudyEnvironment.book`): the M6 headline ``autocall 3y`` and ``phoenix 3y``
(:func:`~volsto.studies.m6.headline_products` struck on the SPX spot), the M4 study cliquet
``cliquet 1y`` (``AdditiveCliquet.study(1.0, discount)``: monthly, cap 2%, global floor 0), the
``vko put 12m`` (``VolKnockOutPut(spot, 1.0, 0.30, daily, notional = 1/spot)``) and the
``ko var 1y``
(``KnockOutVarianceSwap(daily, 1.1 × spot, strike 0)``, M4's convention: its P&L is in **variance
units** and is reported in vol-notional terms by dividing by ``2 K_vol`` with ``K_vol`` the pricing
model's fair strike at ``t = 0`` — :attr:`TaskResult.unit`).  Units: autocall, Phoenix, cliquet
and VKO P&L in **% of notional** (the products carry notional 1, the VKO ``1/spot``); the KO var
in vol points of vega notional.

**Sign convention (the desk is SHORT the note).**  The hedger prices the product LONG (its
product leg is ``payoff − V₀`` on every path, its recalibration P&L ``V(new) − V(old)``); every
"desk" number here is the negative of the hedger's (:func:`to_desk_pnl`), the convention of
:data:`~volsto.risk.shadow_rotation.ROTATION_CONVENTION`.  The tables say which sign they carry.

**Studies** (:func:`enumerate_tasks`; one :class:`Task` per hedger run, a stable string key):

* **A — regression ports.**  The cliquet and FVA strategies with the ``q`` sweep of owner decision
  (a): cliquet ``delta only``, ``delta + cap calls q`` for ``q ∈ {0.5, 0.75, 1.0}`` without and with
  the net-sized variance swap; FVA 1y → 2y ``delta only``, the forward-start preset, the preset
  ``+ skew`` (the forward risk reversal on the ``skew_T:2`` tent; skew stays opt-in, decision (b)).
  Pricing = world ∈ {``LV`` (Dupire of the SPX surface), ``2F`` (the marking LSV)}; **monthly**
  rebalancing as the original studies (:data:`STUDY_A_FREQUENCY`).  The strategy ranking by P&L
  std is pinned by the slow regression test.
* **B — model mismatch (the model reserve).**  Pricing = the marking fit; world ∈ {(i) ``same``,
  (ii) ``historical`` — the FINAL historical fit on the 2022 H2 history
  (``configs/studies/m8b/world_historical.yaml``) **only if** the raw-slice discriminator's verdict
  (``outputs/essvi_gate/discriminator/discriminator_verdict.json``, the repaired eSSVI history; the
  pre-repair ``outputs/m8b/discriminator_verdict.json`` as a fallback) is ``real``, else the task
  list carries the row "surface artefact, skipped" with the discriminator's reason
  (:func:`discriminator_gate`), (iii) ``pure LV`` (Dupire of the SPX surface), (iv) ``nu x1.5`` (the
  marking fit with ``ν × 1.5``, the leverage recalibrated on the same surface through the cache)} ×
  the five products, the per-product presets (:func:`~volsto.hedging.strategies.default_strategy`).
  Per (world, product): the desk leakage per path (mean ± se), std, the 5 / 95% quantiles, the
  regime breakdown and the attribution; the **static spread** (the marked price minus the world's
  price at ``t = 0``, both with stderr: the hedger's ``V₀`` under the pricing model and a direct
  Monte Carlo price under the world model on the world's own draws — independent, the stderrs add in
  quadrature) next to the **dynamic leakage** (desk leakage − static spread).  The leakage under
  (ii) is the model reserve the SSR = 1 mark implicitly carries.
* **C — shadow rotation as P&L.**  Same book, pricing = the marking fit, world =
  :func:`~volsto.hedging.worlds.skew_shock_world` (the pricing model with +1 / +2 / +3 rota applied
  linearly over 5 business days from ``t0 = T/2``, then held), recalibration ``none`` vs
  ``on_skew_move`` under ``policy ∈ {sabr_linked, sticky_breakeven}`` (the rule's default
  ``skew_move_threshold``; a refit's leverage at :attr:`StudyConfig.refit_particles`).  Reported:
  the recalibration P&L isolated per refit date and summed (a :class:`RecordingHedger` keeps the
  per-date split the M8 hedger only accumulates), in the desk convention, against the static
  prediction ``desk_pnl_shadow × rota`` of
  :func:`~volsto.risk.shadow_rotation.rotation_shadow_sensitivity` under the same policy (the M7
  greek at :attr:`StudyConfig.rotation_particles` / :attr:`StudyConfig.rotation_paths`, the SPX
  rotation states of the M7 run); :func:`first_order_agreement` at +1 rota (30%,
  :data:`FIRST_ORDER_TOLERANCE`) and the nonlinearity ``P&L(rota) / (rota × P&L(+1)) − 1`` at +2 /
  +3.  The ``none`` rows carry the total hedged desk P&L against ``desk_pnl_usual × rota`` (a
  reference, not asserted).  **The recalibration runs use :class:`RecordingHedger`, which also
  corrects the rule's reading of the world's conditional smile** (its docstring: the M8
  ``Hedger._state_surface`` inverts the path-averaged option prices, i.e. the unconditional
  ``(t + τ)``-option, so its ATM vols grow like ``sqrt((t + τ)/τ)`` and the skew proxy moves at
  the first date after 0 in any world — a bug of ``hedging/hedger.py`` reported to the owner,
  invisible to the M8 test because a Black–Scholes world has zero skew at every date; the
  per-path inversion here belongs in the hedger once accepted) **and triggers on the excess
  skew** — the world's conditional skew minus the pricing model's own prediction of it at the
  same date on the same draws (:meth:`~volsto.hedging.hedger.Hedger._world_skew`): the M8
  reference (the
  world's skew at ``t = 0``) refits at every date in any LSV world, whose forward skew differs
  from its spot skew by more than the threshold, so it cannot isolate a shock.  Measured on the
  cached 2·10⁵ shock world (+1 rota, cliquet, monthly, a stub refit): the excess is 0.001–0.009
  while no strip option reaches into the shock window, 0.018 at ``t = 0.417`` — the 3M option
  then spans ``[t0, t_end]`` and the world's conditional smile anticipates its deterministic
  shock, so the refit fires one pillar *before* ``t0`` — and 0.001–0.008 against the reset
  reference afterwards: one refit in twelve dates.  The rule (:func:`study_c_rule`) reads its
  strips at :data:`STUDY_C_STRIP_PATHS` (8·10⁴, independent of the world paths; precomputed and
  released before the hedge loop) and its refit guards the correlation target
  (:func:`~volsto.hedging.hedger.refit_targets`: the base fit's ``Corr_BE`` held on a date whose
  step-0 reduction is degenerate, ``|Corr_BE|`` capped at 0.97), both counted per row
  (``refits_fallback``, ``refits_capped``) next to ``refits_at_bound``.
* **D — delta-regime P&L, read against the minimum-variance delta** (reinstated by the owner's
  decision of 2026-09-16).  The 1y ATM vanilla (strike = spot) and the 3y autocall, world =
  pricing (2F), ``delta only`` under the §7.2 regimes ``model / sticky_strike / sticky_skew /
  sticky_moneyness`` and the benchmark ``min_variance`` — the pricing model's minimum-variance
  spot-only delta (:func:`~volsto.hedging.hedger.min_variance_delta`, the model delta plus the
  vol-correlation term) — (:class:`~volsto.hedging.strategies.GreekTargetStrategy` with
  ``delta_regime``).  **Headline:** the model delta is NOT the minimum-variance spot-only hedge
  under its own model — the market is incomplete, the MV delta carries the vol-correlation term —
  and the regimes rank by their distance to the common minimum-variance delta
  (:data:`STUDY_D_HEADLINE`).  Per row (:func:`delta_diagnostics`): the P&L std, the mean delta,
  the in-sample variance-minimising scale ``λ*`` of the row's hedge leg with the std at ``λ*`` and
  the implied minimum-variance delta ``λ* × mean delta`` (in-sample: a benchmark, not a
  strategy), and ``distance_to_mv`` = the row's mean delta minus the **common**
  minimum-variance delta the four regime rows imply (:func:`common_mv_delta`); the
  ``min_variance`` row is a fifth line with a validity flag (:func:`mv_benchmark_validity`).
  The estimator's own cost is measured by
  ``scripts/m8b_delta_estimator.py``.

**Frequency rule** (:func:`frequency_for`): daily for maturities ≤ 1y, weekly beyond (the engine's
default budget, SPEC §8.1); study A monthly; ``StudyConfig.frequency`` overrides.  The simulation
step is the rebalancing step capped at :data:`SIM_DT_MAX` (weekly; the products' own fixings — the
Phoenix's daily knock-in monitoring, the variance products' daily returns — refine the grid).

**Budget and sharding.**  ``--dry-run`` prints per study the task count, the projected hedger wall
clock (:meth:`~volsto.hedging.hedger.Hedger.projected_wall_clock` summed over the tasks) and the
leverage calibrations the tasks need up front (:func:`required_states`: the cache keys the runs
will ask for — the pricing base, the parallel ±1 vp states of ``vega`` / ``volga`` / ``vanna``, the
autocall preset's ``skew_T`` tent, the regime states of D, the worlds of B and the rotated
leverages of C — checked against the cache; the refits inside C are one calibration each unless
cached and cannot be counted ahead) times the manifest's median wall time at the study's particle
count.  ``--shard i/n`` takes every ``n``-th task of the ordered list starting at ``i``
(interleaved,
so every shard carries a mix of cheap and expensive tasks); ``--resume`` skips tasks whose result
file exists; the tables are written from every result present in the output directory, so shards
merge.  Results: ``<out>/<study>/<key>.json`` (:class:`TaskResult`) and ``<key>.pkl`` (the
:class:`~volsto.hedging.hedger.HedgeResult` without its world paths and per-date Greek arrays).

Tests never calibrate (``allow_calibrate=False``, skip on
:class:`~volsto.calibration.cache.CacheMissError`); the script calibrates and says so.  Checked by
``tests/test_m8b.py``.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import pickle
import re
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.calibration.cache import CacheMissError, LeverageCache
from volsto.calibration.fit_2f import load_fit_spec
from volsto.config import (
    CalibrationSpec,
    SimConfig,
    SurfacePerturbation,
    load_yaml,
    to_mapping,
)
from volsto.engine.grid import TimeGrid
from volsto.engine.mc import MonteCarlo
from volsto.engine.paths import PathSet
from volsto.hedging.hedger import (
    CORRELATION_BOUND,
    FREQUENCIES,
    MAX_HALVINGS,
    REFIT_CORRELATION_CAP,
    SPOT_BUMP,
    TENT_SIZE,
    VOL_BUMP,
    Costs,
    Hedger,
    HedgeResult,
    PricingContext,
    RecalibrationRule,
    Schedule,
)
from volsto.hedging.instruments import Spot
from volsto.hedging.pricing import DEFAULT_CONTROL_DELTA, Bump, ConditionalPricer, union_grid
from volsto.hedging.report import attribution_table, distribution_table, regime_table
from volsto.hedging.state import HedgeState
from volsto.hedging.strategies import (
    DELTA_REGIMES,
    MIN_VARIANCE_REGIME,
    SURFACE_DELTA_REGIMES,
    GreekTargetStrategy,
    Strategy,
    Target,
    default_strategy,
)
from volsto.hedging.worlds import SHOCK_DAYS, shock_state, skew_shock_world
from volsto.market.loaders import snapshot_spec
from volsto.models.base import Model
from volsto.products.base import Product, daily_schedule
from volsto.products.cliquet import AdditiveCliquet
from volsto.products.conditional_variance import KnockOutVarianceSwap
from volsto.products.vanilla import EuropeanOption
from volsto.products.variance import FVA
from volsto.products.vko import VolKnockOutPut
from volsto.risk.engine import RiskState, surface_of
from volsto.risk.greeks import spot_state
from volsto.risk.ladders import PILLARS as RISK_PILLARS
from volsto.risk.ladders import skew_slope
from volsto.risk.shadow_rotation import (
    RECALIBRATION_POLICIES,
    ROTATION_CONVENTION,
    rotation_shadow_sensitivity,
)
from volsto.studies.m6 import AUTOCALL_NAME, PHOENIX_NAME, headline_products

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[2]
#: the M7 P1 marking fit of SPX 2022-12-30 at (ssr_target 1, skew_eps 0.10): the pricing model
MARKING_FIT = ROOT / "configs" / "studies" / "m7_p1_marking" / "spx_ssr1_eps0.1.yaml"
#: the FINAL historical fit on the 2022 H2 history (A4): study B world (ii)
WORLD_HISTORICAL = ROOT / "configs" / "studies" / "m8b" / "world_historical.yaml"
#: the raw-slice discriminator's verdict (A4): gates world (ii).  Since M10 Part 0 the gate reads
#: the run on the REPAIRED eSSVI history (``outputs/essvi_gate/discriminator/``, the owner's
#: condition "that lands with the M10 Part 0 eSSVI repair"); the M8b run on the plain-SSVI history
#: (``outputs/m8b/discriminator_verdict.json``, the same verdict on 2026-09-16) is the fallback
#: when the repaired run is absent (:func:`discriminator_gate`)
DISCRIMINATOR_VERDICT = (
    ROOT / "outputs" / "essvi_gate" / "discriminator" / "discriminator_verdict.json"
)
DISCRIMINATOR_VERDICT_PRE_REPAIR = ROOT / "outputs" / "m8b" / "discriminator_verdict.json"
#: the SPX snapshot and the reference study spec the marking fit was built from
#: (``scripts/m7_p1_marking.py::base_spec``): the base spec of the M7 rotation greek
REF_SPEC = ROOT / "configs" / "studies" / "lsv_reference_2f.yaml"
SPX_SNAPSHOT = ROOT / "configs" / "surfaces" / "snapshots" / "hdn_2022H2" / "spx_2022-12-30.yaml"
DEFAULT_OUT = ROOT / "outputs" / "m8b"
DEFAULT_CACHE = ROOT / "cache"

STUDIES: tuple[str, ...] = ("A", "B", "C", "D")
#: the five products of studies B and C (module docstring)
BOOK: tuple[str, ...] = (AUTOCALL_NAME, PHOENIX_NAME, "cliquet 1y", "vko put 12m", "ko var 1y")
CLIQUET_NAME = "cliquet 1y"
VKO_NAME = "vko put 12m"
KOVAR_NAME = "ko var 1y"
FVA_NAME = "fva 1y-2y"
VANILLA_NAME = "vanilla 1y atm"
#: study B worlds in the owner's order (i)–(iv)
WORLDS_B: tuple[str, ...] = ("same", "historical", "pure LV", "nu x1.5")
WORLD_HISTORICAL_NAME = "historical"
#: study A pricing (= world) models
PRICING_A: tuple[str, ...] = ("LV", "2F")
#: the q sweep of owner decision (a)
Q_SWEEP: tuple[float, ...] = (0.5, 0.75, 1.0)
#: study C shocks (rotas) and recalibration settings (``"none"`` = no rule)
ROTAS: tuple[float, ...] = (1.0, 2.0, 3.0)
RECALIBRATIONS_C: tuple[str, ...] = ("none", "sabr_linked", "sticky_breakeven")
#: study D regimes (``sticky_local_vol`` is not in the owner's list) and the benchmark row, the
#: minimum-variance delta (owner's decision of 2026-09-16: "the benchmark row as a fifth line")
REGIMES_D: tuple[str, ...] = (
    "model",
    "sticky_strike",
    "sticky_skew",
    "sticky_moneyness",
    MIN_VARIANCE_REGIME,
)
#: the study-D headline, in the owner's words (decision of 2026-09-16)
STUDY_D_HEADLINE = (
    "the model delta is NOT the minimum-variance spot-only hedge under its own model — "
    "incomplete market, the MV delta carries the vol-correlation term — and for these products "
    "sticky-strike and sticky-skew sit closest to it"
)
#: the measured two-sided reading of study D (2F, 2·10⁴ paths, 2026-09-16 measurement): the sign
#: of the vol-correlation term follows the product's vega
STUDY_D_READING = (
    "for the 1y ATM call the minimum-variance delta lies BELOW the model delta (the "
    "vol-correlation term is negative: the call is long vega and spot-vol correlation is "
    "negative), for the 3y autocall ABOVE it (the note is short vol); in both cases "
    "sticky-strike and sticky-skew sit closest to it"
)
#: the ``min_variance`` benchmark row is flagged valid (``mv_valid``) when its mean delta lies
#: within this many standard errors (the two se's in quadrature) of the common
#: minimum-variance delta the regime rows imply.  Measured (2026-09-16): the vanilla row at 2·10⁴
#: pricing paths sits on the common value (0.5366 ± 0.0007 against 0.534–0.538), at 5·10³ it
#: sits ~0.01 above it (0.5457 ± 0.0012 and 0.5476 ± 0.0012 against 0.533–0.540, over 4 se),
#: and the autocall row (no Black–Scholes proxy: raw regression gradients) at 0.318 ± 0.002
#: against 0.294–0.309
MV_BENCHMARK_VALIDITY_NSE = 3.0
#: the pricing note of a ``min_variance`` run whose factor gradients came from the raw value
#: regression (:meth:`volsto.hedging.hedger.Hedger._run`, ``factor_gradients``)
MV_RAW_GRADIENT_NOTE = "min_variance delta: no controlled value target"
#: path-pair bootstrap draws (and seed) of the study-D delta diagnostics' standard errors
#: (``λ*``, the std at ``λ*``, the implied minimum-variance delta): 200 resamples put the se's
#: own relative error near 5%
DELTA_DIAG_BOOTSTRAP = 200
DELTA_DIAG_SEED = 11
#: owner: "2e4 paths default"
DEFAULT_N_PATHS = 20_000
#: the largest simulation step (years): weekly, the M8 study-test step (``SIM_SMALL`` of
#: ``tests/test_hedging.py``: monthly rebalancing over weekly steps); a task's step is
#: ``min(rebalancing step, SIM_DT_MAX)`` and the products' fixings refine the grid further (the
#: Phoenix's daily knock-in monitoring, the variance products' daily returns) — a monthly
#: simulation step for study A would discretise the Dupire / mixing dynamics far too coarsely
#: study C: the pillars at which the recalibration rule reads the world's state surface and the
#: two-point skew constraint of the simulated refit — the static greek's SPX marking fit
#: constrains the skew at 1Y / 3Y (5Y lies beyond the snapshot), so both refits solve the same
#: constraint and the 30% first-order comparison is like for like
STUDY_C_RULE_PILLARS: tuple[float, ...] = (0.25, 1.0, 3.0)
STUDY_C_RULE_SKEW_PILLARS: tuple[float, float] = (1.0, 3.0)
#: study C's trigger threshold (vol per unit log-moneyness): the footprint of a +1 rota on the
#: world's 3M forward skew at mid-life is ~0.007 (measured on the 3y autocall's shock world: the
#: 1/sqrt(T) rotation's local-vol footprint at 1.5y is about half its implied-skew move there),
#: below the hedger's general default 0.01 (owner decision (c)); at 0.005 the +1 rota refit fires
#: when the 3M strip first spans the shock window (one pillar early — the deterministic shock
#: world anticipates itself), +2 / +3 rota clear both thresholds; the proxy noise at 2e4 paths is
#: 0.001-0.003 (0.002-0.005 at 4e3)
STUDY_C_SKEW_MOVE_THRESHOLD = 0.005
#: study C: the path count of the recalibration rule's forward-start strips (the world strip and
#: its pricing-model twin; ``RecalibrationRule.strip_paths``), independent of the 2·10⁴ world
#: paths.  Owner's decision of 2026-09-16 on the pinning diagnosis: at 2·10⁴ strip paths the
#: state surface's curvature is unconverged (3M / 1Y / 3Y −1.22 / −0.53 / −0.34 against
#: −0.56 / +0.39 / +0.56 at 1.6·10⁵ on the named autocall state), step 0's radicand guard fires
#: at every pillar and 41 of the 113 ``sabr_linked`` refits of the 2026-09-15 runs pinned their
#: correlations; from 8·10⁴ no pillar is guarded on that named state and the fit is regular — but
#: not everywhere: on the 1y daily VKO put world the 3M pillar still reads a degenerate step 0 on
#: 136 of 252 dates at 8·10⁴ (133 after the forward-moneyness strip fix, 54 with a ±0.10
#: curvature stencil) and the guarded fallback carries it (see
#: :data:`~volsto.hedging.hedger.DEFAULT_STRIP_PATHS`)
STUDY_C_STRIP_PATHS = 80_000
#: study C: the half-width of the recalibration strip's curvature stencil
#: (``RecalibrationRule.curvature_h``); ``None`` means equal to the strip's ``h`` (0.05), the
#: three-strike strip.  THE OWNER'S CHOICE IS PENDING (2026-09-16): a wider stencil cuts the
#: curvature's Monte Carlo noise (its error scales as ``1/(curvature_h √strip_paths)``) but
#: smooths the curvature over the wings — measured on the pricing snapshot (SPX 2022-12-30),
#: the 3M / 1Y / 3Y curvature read on the ``±0.10`` stencil exceeds the ``k → 0`` one by
#: +0.066 / +0.016 / +0.004 (+0.016 / +0.004 / +0.001 at ``±0.05``)
STUDY_C_CURVATURE_H: float | None = None

#: the studies rebalance on the frequency grid only (:class:`~volsto.hedging.hedger.Schedule`
#: ``product_fixings``): the M8 default adds every product fixing as a rebalancing date, which
#: turns the "weekly" 3y Phoenix (757 daily knock-in fixings) into a daily hedge — 900 dates and
#: 1560 s per run against 156 dates for the autocall (measured, study B); the knock-in state at
#: a weekly date includes every fixing up to it
STUDY_PRODUCT_FIXINGS = False

SIM_DT_MAX = 1.0 / 52.0
#: production leverage convention (SPEC §11, owner decision at the M4b acceptance)
DEFAULT_N_PARTICLES = 800_000
#: a P1 refit inside a hedger run is a new parameter set, hence a calibration: production count
DEFAULT_REFIT_PARTICLES = 800_000
#: the M7 rotation greek's states were calibrated at 2·10⁵ particles (scripts/m7_p1_marking.py):
#: the static prediction of study C reuses them (base, ±1 rota, the policies' refits)
DEFAULT_ROTATION_PARTICLES = 200_000
#: the M7 rotation greek's pricing budget (seed 2024, 2·10⁵ paths)
DEFAULT_ROTATION_PATHS = 200_000
DEFAULT_ROTATION_SEED = 2024
#: pricing seed of the hedger runs (the world seed is the pricing seed + 1, the hedger's rule)
DEFAULT_SEED = 2024
#: study A rebalances monthly as the original cliquet / FVA studies (SPEC §8.1 table)
STUDY_A_FREQUENCY = "monthly"
#: the frequency rule: daily up to this maturity (years), weekly beyond (SPEC §8.1 default budget)
DAILY_MAX_MATURITY = 1.0
#: owner: "the marking fit with nu x 1.5"
NU_SCALE = 1.5
#: the VKO's knock-out vol and the KO var's barrier (M4 study conventions, studies/m4.py)
VKO_VOL_KO = 0.30
KOVAR_BARRIER = 1.1
KOVAR_STRIKE_VOL = 0.0
#: owner: "assert first-order agreement within 30% at +1 rota"
FIRST_ORDER_TOLERANCE = 0.30
#: per-calibration wall clock used by the projection when the cache manifest has no entry at
#: the study's particle count: the manifest's 8·10⁵ median (159 s) scaled linearly in N
FALLBACK_CALIBRATION_SECONDS_8E5 = 159.0
_TOL = 1e-9


# --------------------------------------------------------------------------------------------
# configuration and tasks
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StudyConfig:
    """Budget and paths of a study run (module docstring for every default's reason)."""

    n_paths: int = DEFAULT_N_PATHS
    n_particles: int = DEFAULT_N_PARTICLES
    refit_particles: int = DEFAULT_REFIT_PARTICLES
    world_paths: int | None = None
    seed: int = DEFAULT_SEED
    frequency: str | None = None
    out: Path = DEFAULT_OUT
    cache: Path = DEFAULT_CACHE
    allow_calibrate: bool = True
    stream_bumps: bool = False
    control_variate: bool = True
    control_delta: bool = DEFAULT_CONTROL_DELTA
    rotation_particles: int = DEFAULT_ROTATION_PARTICLES
    rotation_paths: int = DEFAULT_ROTATION_PATHS
    marking_fit: Path = MARKING_FIT
    world_historical: Path = WORLD_HISTORICAL
    discriminator_verdict: Path = DISCRIMINATOR_VERDICT
    verbose: bool = True

    def __post_init__(self) -> None:
        if self.n_paths < 2 or self.n_paths % 2:
            raise ValueError("n_paths must be an even number of paths (antithetic draws)")
        if self.world_paths is not None and (self.world_paths < 2 or self.world_paths % 2):
            raise ValueError("world_paths must be an even number of paths")
        if self.frequency is not None and self.frequency not in FREQUENCIES:
            raise ValueError(f"frequency must be one of {tuple(FREQUENCIES)}")

    @property
    def n_world(self) -> int:
        return int(self.world_paths or self.n_paths)


def frequency_for(maturity: float, study: str, cfg: StudyConfig | None = None) -> str:
    """The rebalancing frequency of a product (module docstring): the config's override, else
    monthly in study A, else daily up to :data:`DAILY_MAX_MATURITY` and weekly beyond."""
    if cfg is not None and cfg.frequency is not None:
        return cfg.frequency
    if study == "A":
        return STUDY_A_FREQUENCY
    return "daily" if float(maturity) <= DAILY_MAX_MATURITY + _TOL else "weekly"


def slug(text: str) -> str:
    """A file-system-safe token of a label."""
    s = re.sub(r"[^0-9A-Za-z.+-]+", "_", str(text).strip())
    return s.strip("_") or "x"


@dataclass(frozen=True)
class Task:
    """One hedger run.  ``kwargs`` are the preset / strategy parameters (hashable pairs);
    ``pricing`` names the pricing model (``"2F"`` the marking LSV, ``"LV"`` Dupire of the SPX
    surface); ``rota`` / ``policy`` (study C: ``"none"`` = no rule) and ``regime`` (study D) are
    ``None`` elsewhere."""

    study: str
    product: str
    world: str
    strategy: str
    kwargs: tuple[tuple[str, Any], ...] = ()
    pricing: str = "2F"
    rota: float | None = None
    policy: str | None = None
    regime: str | None = None
    frequency: str = "daily"

    @property
    def key(self) -> str:
        parts = [self.study, slug(self.product), slug(self.world), slug(self.strategy)]
        if self.pricing != "2F":
            parts.append(f"pricing_{slug(self.pricing)}")
        if self.rota is not None:
            parts.append(f"rota{self.rota:+g}")
        if self.policy is not None:
            parts.append(f"recal_{slug(self.policy)}")
        if self.regime is not None:
            parts.append(f"regime_{slug(self.regime)}")
        return "__".join(parts)

    @property
    def kwargs_dict(self) -> dict[str, Any]:
        return dict(self.kwargs)

    @property
    def label(self) -> str:
        extra = []
        if self.rota is not None:
            extra.append(f"rota {self.rota:+g}")
        if self.policy is not None:
            extra.append(f"recalibration {self.policy}")
        if self.regime is not None:
            extra.append(f"regime {self.regime}")
        tail = f" ({', '.join(extra)})" if extra else ""
        return (
            f"{self.study}: {self.product} | world {self.world} | pricing {self.pricing} | "
            f"{self.strategy}{tail} | {self.frequency}"
        )


@dataclass(frozen=True)
class Gate:
    """The discriminator's decision for world (ii): ``enabled`` when the verdict is ``real``."""

    enabled: bool
    verdict: str
    reason: str

    @property
    def row_label(self) -> str:
        return "surface artefact, skipped" if not self.enabled else "real"


def discriminator_gate(path: str | Path | None = None) -> Gate:
    """Read ``discriminator_verdict.json``; a missing or unreadable file disables world (ii)
    with that reason (the owner's rule: (ii) only if the discriminator confirmed the SSR)."""
    p = Path(path) if path is not None else DISCRIMINATOR_VERDICT
    note = ""
    if not p.exists() and p == DISCRIMINATOR_VERDICT and DISCRIMINATOR_VERDICT_PRE_REPAIR.exists():
        # the documented fallback: the M8b run on the pre-repair plain-SSVI history
        p = DISCRIMINATOR_VERDICT_PRE_REPAIR
        note = f" [pre-repair discriminator {p.relative_to(ROOT)}: the repaired run is absent]"
    if not p.exists():
        return Gate(False, "missing", f"no discriminator verdict at {p}: world (ii) not confirmed")
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Gate(False, "unreadable", f"discriminator verdict unreadable ({exc})")
    verdict = str(doc.get("verdict", "")).strip().lower()
    reason = str(doc.get("reason", "")) + note
    return Gate(verdict == "real", verdict or "missing", reason)


def cliquet_strategy_specs() -> list[tuple[str, dict[str, Any]]]:
    """Study A cliquet strategies: label → preset kwargs (``q``, ``var_swap``)."""
    specs: list[tuple[str, dict[str, Any]]] = [("delta only", {})]
    for q in Q_SWEEP:
        specs.append((f"delta + cap calls q={q:g}", {"q": q, "var_swap": False}))
    for q in Q_SWEEP:
        specs.append((f"delta + cap calls q={q:g} + net var swap", {"q": q, "var_swap": True}))
    return specs


def fva_strategy_specs() -> list[tuple[str, dict[str, Any]]]:
    return [
        ("delta only", {}),
        ("forward-start preset", {}),
        ("forward-start preset + skew", {"skew": True}),
    ]


@dataclass(frozen=True)
class TaskList:
    """The ordered tasks of a study plus the rows the gate removed (for the tables)."""

    study: str
    tasks: tuple[Task, ...]
    skipped: tuple[dict[str, Any], ...] = ()


def enumerate_tasks(
    study: str, cfg: StudyConfig | None = None, gate: Gate | None = None
) -> TaskList:
    """The task list of a study (module docstring).  ``gate`` defaults to the config's verdict
    file; study B's world (ii) is dropped with the reason when the gate is closed."""
    cfg = cfg or StudyConfig()
    tasks: list[Task] = []
    skipped: list[dict[str, Any]] = []
    if study == "A":
        for pricing in PRICING_A:
            for label, kw in cliquet_strategy_specs():
                tasks.append(
                    Task(
                        "A",
                        CLIQUET_NAME,
                        "pricing",
                        label,
                        tuple(sorted(kw.items())),
                        pricing,
                        frequency=frequency_for(1.0, "A", cfg),
                    )
                )
            for label, kw in fva_strategy_specs():
                tasks.append(
                    Task(
                        "A",
                        FVA_NAME,
                        "pricing",
                        label,
                        tuple(sorted(kw.items())),
                        pricing,
                        frequency=frequency_for(2.0, "A", cfg),
                    )
                )
    elif study == "B":
        g = gate or discriminator_gate(cfg.discriminator_verdict)
        for world in WORLDS_B:
            for name in BOOK:
                T = product_maturity(name)
                if world == WORLD_HISTORICAL_NAME and not g.enabled:
                    skipped.append(
                        {
                            "world": world,
                            "product": name,
                            "status": g.row_label,
                            "reason": f"discriminator verdict {g.verdict!r}: {g.reason}",
                        }
                    )
                    continue
                tasks.append(
                    Task("B", name, world, "preset", (), frequency=frequency_for(T, "B", cfg))
                )
    elif study == "C":
        for name in BOOK:
            T = product_maturity(name)
            for rota in ROTAS:
                for recal in RECALIBRATIONS_C:
                    tasks.append(
                        Task(
                            "C",
                            name,
                            "skew shock",
                            "preset",
                            (),
                            rota=float(rota),
                            policy=recal,
                            frequency=frequency_for(T, "C", cfg),
                        )
                    )
    elif study == "D":
        for name in (VANILLA_NAME, AUTOCALL_NAME):
            T = product_maturity(name)
            for regime in REGIMES_D:
                tasks.append(
                    Task(
                        "D",
                        name,
                        "pricing",
                        "delta only",
                        (),
                        regime=regime,
                        frequency=frequency_for(T, "D", cfg),
                    )
                )
    else:
        raise ValueError(f"study must be one of {STUDIES}")
    keys = [t.key for t in tasks]
    if len(set(keys)) != len(keys):
        raise RuntimeError(f"duplicate task keys in study {study}")
    return TaskList(study, tuple(tasks), tuple(skipped))


def product_maturity(name: str) -> float:
    """Maturity in years of a named product (for the frequency rule without building it)."""
    if name in (AUTOCALL_NAME, PHOENIX_NAME):
        return 3.0
    if name in (CLIQUET_NAME, VKO_NAME, KOVAR_NAME, VANILLA_NAME):
        return 1.0
    if name == FVA_NAME:
        return 2.0
    raise ValueError(f"unknown product {name!r}")


def shard(tasks: Sequence[Task], index: int, count: int) -> list[Task]:
    """Shard ``index`` of ``count`` (1-based): every ``count``-th task from ``index − 1``."""
    if count < 1 or not 1 <= index <= count:
        raise ValueError(f"shard must be i/n with 1 <= i <= n (got {index}/{count})")
    return list(tasks[index - 1 :: count])


def parse_shard(text: str) -> tuple[int, int]:
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", text)
    if not m:
        raise ValueError(f"--shard expects i/n (got {text!r})")
    return int(m.group(1)), int(m.group(2))


# --------------------------------------------------------------------------------------------
# conventions
# --------------------------------------------------------------------------------------------


def to_desk_pnl(x: Any) -> Any:
    """The desk is SHORT the note: the negative of the hedger's (long-product) P&L, of its
    recalibration P&L ``V(new) − V(old)`` and of any mean built from them (arrays or floats;
    standard errors are unchanged)."""
    return -np.asarray(x, dtype=np.float64) if isinstance(x, np.ndarray) else -float(x)


def first_order_agreement(
    recal_pnl_total: float, recal_se: float, static_prediction: float, static_se: float
) -> tuple[float, bool, float]:
    """``(ratio, within_30pct, z)`` of a simulated recalibration P&L against the static greek's
    prediction (both in the desk convention): ``ratio = simulated / static`` (NaN when the
    prediction is 0), ``within`` is ``|ratio − 1| ≤`` :data:`FIRST_ORDER_TOLERANCE`, ``z`` the
    difference over its combined standard error."""
    se = float(np.sqrt(float(recal_se) ** 2 + float(static_se) ** 2))
    z = (float(recal_pnl_total) - float(static_prediction)) / se if se > 0 else float("nan")
    if static_prediction == 0.0 or not np.isfinite(static_prediction):
        return float("nan"), False, z
    ratio = float(recal_pnl_total) / float(static_prediction)
    return ratio, bool(abs(ratio - 1.0) <= FIRST_ORDER_TOLERANCE), z


def ratio_stderr(num: float, num_se: float, den: float, den_se: float) -> float:
    """Delta-method standard error of ``num / den`` for two Monte Carlo estimates with
    **independent** errors: ``|num / den| · sqrt((se_num / num)² + (se_den / den)²)`` (NaN when
    either estimate is 0 or not finite).  With correlated errors (correlation ``ρ``) the exact
    first-order variance adds ``−2ρ · (num/den)² · se_num se_den / (num · den)``: the independent
    figure is conservative only when ``ρ ≥ 0`` **and** the two estimates have the same sign; for
    opposite signs a positive correlation makes it understate.  The study-C runs at different
    rotas share the world seed (their correlation is not measured), so a ratio of two of them
    takes :func:`ratio_stderr_bound`; the static greek is a separate run.  Test:
    ``tests/test_m8b.py::test_tables_from_synthetic_results``."""
    num, den = float(num), float(den)
    if num == 0.0 or den == 0.0 or not (np.isfinite(num) and np.isfinite(den)):
        return float("nan")
    return float(abs(num / den) * np.hypot(float(num_se) / num, float(den_se) / den))


def ratio_stderr_bound(num: float, num_se: float, den: float, den_se: float) -> float:
    """A first-order standard error of ``num / den`` valid **for any correlation** of the two
    errors: ``|num / den| · (|se_num / num| + |se_den / den|)`` — the maximum over ``ρ ∈ [−1, 1]``
    of the delta-method error, at most ``√2`` times :func:`ratio_stderr` (NaN when either estimate
    is 0 or not finite).  For ratios of estimates that share random numbers with an unmeasured
    correlation (the study-C runs share the world seed).  Test:
    ``tests/test_catalogue_s5_s7.py::test_ratio_stderr_bound``."""
    num, den = float(num), float(den)
    if num == 0.0 or den == 0.0 or not (np.isfinite(num) and np.isfinite(den)):
        return float("nan")
    return float(abs(num / den) * (abs(float(num_se) / num) + abs(float(den_se) / den)))


def nonlinearity(pnl_rota: float, rota: float, pnl_one: float) -> float:
    """``P&L(rota) / (rota × P&L(+1)) − 1`` (NaN when the +1 value is 0)."""
    if pnl_one == 0.0 or not np.isfinite(pnl_one):
        return float("nan")
    return float(pnl_rota) / (float(rota) * float(pnl_one)) - 1.0


def refit_state(state: RiskState, base_params: Any, n_particles: int) -> RiskState:
    """A state whose parameters differ from ``base_params`` (a P1 refit) at ``n_particles``;
    any other state (the base, a surface bump of the base set) unchanged."""
    if state.spec.model != base_params and state.spec.particle.n_particles != n_particles:
        return state.with_particles(n_particles)
    return state


class RefitParticlesBuilder:
    """An :class:`~volsto.risk.engine.LSVBuilder` whose refits (states with other parameters
    than the base set) are built at ``refit_particles`` (:func:`refit_state`); everything else is
    delegated.  Installed only when ``refit_particles != n_particles``."""

    def __init__(self, inner: Any, refit_particles: int) -> None:
        self.inner = inner
        self.refit_particles = int(refit_particles)
        self.cache = inner.cache
        self.base = inner.base
        self.base_model = inner.base_model
        self.allow_calibrate = inner.allow_calibrate

    def build(self, state: RiskState, mode: str) -> Model:
        st = refit_state(state, self.base.spec.model, self.refit_particles)
        model: Model = self.inner.build(st, mode)
        return model

    def has(self, state: RiskState) -> bool:
        """Whether the leverage of ``state`` **as this builder will build it** (the refit particle
        count applied) is cached — the hedger's refit log reads this, not the raw spec."""
        st = refit_state(state, self.base.spec.model, self.refit_particles)
        return bool(self.cache.has(st.spec))

    @property
    def n_calibrations(self) -> int:
        return int(self.inner.n_calibrations)

    @property
    def n_cache_misses(self) -> int:
        return int(self.inner.n_cache_misses)

    @property
    def cache_keys(self) -> list[str]:
        return list(self.inner.cache_keys)


def spx_base_spec(n_particles: int) -> CalibrationSpec:
    """``scripts/m7_p1_marking.py::base_spec("spx", n_particles)``: the reference 2F study spec
    with the SPX 2022-12-30 snapshot's market and surface — the base spec of the M7 rotation
    greek (its model field is replaced by the marking fit inside the greek)."""
    ref = snapshot_spec(load_yaml(REF_SPEC, CalibrationSpec), SPX_SNAPSHOT)
    return dataclasses.replace(
        ref, particle=dataclasses.replace(ref.particle, n_particles=int(n_particles))
    )


# --------------------------------------------------------------------------------------------
# the per-date recalibration P&L (study C)
# --------------------------------------------------------------------------------------------


@dataclass
class _RecordingPricer(ConditionalPricer):
    """A :class:`~volsto.hedging.pricing.ConditionalPricer` that keeps the product's conditional
    value and alive flags at every date it is evaluated on the world paths (object 0)."""

    values0: dict[float, tuple[FloatArray, NDArray[np.bool_]]] = field(
        init=False, default_factory=dict, repr=False
    )

    def evaluate(
        self, obj_index: int, t: float, paths: PathSet, kinds: Sequence[str]
    ) -> tuple[dict[str, FloatArray], HedgeState]:
        out, hs = super().evaluate(obj_index, t, paths, kinds)
        if obj_index == 0 and "value" in out and paths is not self.paths:
            self.values0.setdefault(float(t), (np.array(out["value"]), np.array(hs.alive)))
        return out, hs


@dataclass
class RecordingHedger(Hedger):
    """A :class:`~volsto.hedging.hedger.Hedger` whose pricers record the product's value per
    date, so the recalibration P&L — accumulated by the M8 loop as one per-path total — can be
    **isolated per refit date** afterwards (:func:`recalibration_by_date`): the pricers are
    created in refit order (the base pricer first, one per refit), and the refit at date ``t_k``
    contributed ``V_k(new) − V_k(old)`` on the paths alive at ``t_k``."""

    created: list[_RecordingPricer] = field(default_factory=list, init=False, repr=False)

    def _pricer(
        self,
        model: Model,
        objects: list[Product],
        grid: TimeGrid,
        bumps: Sequence[Bump],
        strategy: Strategy | None = None,
    ) -> ConditionalPricer:
        mv = strategy is not None and strategy.delta_regime == MIN_VARIANCE_REGIME
        p = _RecordingPricer(
            model,
            objects,
            grid,
            self.sim,
            tuple(bumps),
            self.degree,
            stream_bumps=self.stream_bumps,
            scratch_dir=self.scratch_dir,
            control_variate=self.control_variate,
            control_delta=self.control_delta,
            control_value=bool(mv and self.control_variate),
            surface=self.context.surface,
        )
        self.created.append(p)
        return p


_RHO_IN_REPR = re.compile(r"rho(?:12|_SX1|_SX2)=(-?[0-9.eE+]+)")


def refits_at_bound(recalibrations: pd.DataFrame) -> int:
    """How many of a run's refits landed with a fitted correlation at its bound
    (``|rho| >=`` :data:`~volsto.hedging.hedger.CORRELATION_BOUND`): the hedger's own
    ``at_bound`` column when the run recorded it, else the correlations read back out of the
    stored ``params`` repr (the runs of 2026-09-15, before the flag existed).  A refit that
    lands there prices the product under perfectly correlated factors, so the recalibration
    P&L of that row measures the fitter railing as much as the cost of re-marking: study C
    carries the count per row and marks such rows contaminated (§8.2)."""
    if recalibrations.empty or "recalibrated" not in recalibrations:
        return 0
    fired = recalibrations.loc[recalibrations["recalibrated"].astype(bool)]
    if "at_bound" in fired:
        return int(sum(bool(str(v).strip()) for v in fired["at_bound"]))
    n = 0
    for text in fired.get("params", []):
        rhos = [abs(float(x)) for x in _RHO_IN_REPR.findall(str(text))]
        n += int(bool(rhos) and max(rhos) >= CORRELATION_BOUND)
    return n


def _refits_flagged(recalibrations: pd.DataFrame, column: str) -> int:
    """How many fired refits carry ``column`` true; ``-1`` when the run did not record the
    column (a run from before it existed)."""
    if recalibrations.empty or "recalibrated" not in recalibrations:
        return 0
    if column not in recalibrations:
        return -1
    fired = recalibrations.loc[recalibrations["recalibrated"].astype(bool)]
    return int(fired[column].fillna(False).astype(bool).sum())


def refits_fallback(recalibrations: pd.DataFrame) -> int:
    """How many of a run's refits used the guarded correlation fallback (step 0's radicand
    guard or ``rho`` clip at some pillar of that date's state surface: the base fit's
    ``correl_target`` held, :func:`~volsto.hedging.hedger.refit_targets`); ``-1`` when the run
    predates the record."""
    return _refits_flagged(recalibrations, "fallback_applied")


def refits_capped(recalibrations: pd.DataFrame) -> int:
    """How many of a run's refits had their correlation target capped at
    :data:`~volsto.hedging.hedger.REFIT_CORRELATION_CAP`; ``-1`` when the run predates the
    record."""
    return _refits_flagged(recalibrations, "corr_capped")


def recalibration_by_date(
    hedger: RecordingHedger, result: HedgeResult
) -> tuple[list[dict[str, float]], str | None]:
    """Per refit date: the mean recalibration P&L (hedger sign, long the product), its standard
    error and the alive fraction; plus a note when the per-date pieces do not add up to the
    hedger's per-path total (reported, never patched)."""
    rows: list[dict[str, float]] = []
    if result.recalibrations.empty or "recalibrated" not in result.recalibrations:
        return rows, None
    refit_dates = [
        float(t)
        for t in result.recalibrations.loc[result.recalibrations["recalibrated"].astype(bool), "t"]
    ]
    total = np.zeros(result.n_paths)
    note = None
    for k, t in enumerate(refit_dates, start=1):
        if k >= len(hedger.created):
            note = f"refit at t={t:g}: no pricer recorded (created {len(hedger.created)})"
            break
        old, new = hedger.created[k - 1].values0.get(t), hedger.created[k].values0.get(t)
        if old is None or new is None:
            note = f"refit at t={t:g}: value at the refit date not recorded"
            break
        d = np.where(old[1], new[0] - old[0], 0.0)
        total += d
        rows.append(
            {
                "t": t,
                "mean": float(d.mean()),
                "stderr": float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else float("nan"),
                "alive_fraction": float(old[1].mean()),
            }
        )
    if note is None and rows and not np.allclose(total, result.pnl_recalibration, atol=1e-10):
        gap = float(np.max(np.abs(total - result.pnl_recalibration)))
        note = f"per-date recalibration pieces differ from the hedger's total (max |gap| {gap:.3g})"
    return rows, note


# --------------------------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------------------------


def _mean_se(x: FloatArray) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64)
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else 0.0


@dataclass
class TaskResult:
    """The summary numbers of one run (JSON-serialisable; the arrays stay in the ``.pkl``).
    Scaled numbers (``mean``, ``std``, quantiles, ``recal_*``, ``static_spread``) are in the
    product's reporting unit (``unit``: ``scale`` × the hedger's value; % of notional, or vol
    points of vega notional for the KO var) and in the **hedger's sign** (long the product) —
    the tables apply :func:`to_desk_pnl`.  ``status`` is ``ok``, ``skipped`` (with ``reason``: a
    cache miss under ``allow_calibrate=False``) or ``failed``."""

    key: str
    study: str
    product: str
    world: str
    strategy: str
    pricing: str = "2F"
    rota: float | None = None
    policy: str | None = None
    regime: str | None = None
    frequency: str = "daily"
    status: str = "ok"
    reason: str = ""
    unit: str = "% of notional"
    scale: float = 100.0
    n_paths_pricing: int = 0
    n_paths_world: int = 0
    n_particles: int = 0
    n_dates: int = 0
    value_0: tuple[float, float] = (float("nan"), float("nan"))
    mean: tuple[float, float] = (float("nan"), float("nan"))
    std: tuple[float, float] = (float("nan"), float("nan"))
    quantiles: dict[str, tuple[float, float]] = field(default_factory=dict)
    zero_cost_mean: tuple[float, float] = (float("nan"), float("nan"))
    regimes: list[dict[str, Any]] = field(default_factory=list)
    attribution: list[dict[str, Any]] = field(default_factory=list)
    recal_total: tuple[float, float] = (0.0, 0.0)
    recal_by_date: list[dict[str, float]] = field(default_factory=list)
    n_refits: int = 0
    #: refits whose fitted set has a correlation at its bound (-1: not recorded, a run from
    #: before the flag existed whose JSON was not backfilled)
    n_refits_at_bound: int = -1
    #: refits that used the guarded correlation fallback / whose correlation target was capped
    #: (-1: not recorded, a run from before the rebuilt refit of 2026-09-16)
    n_refits_fallback: int = -1
    n_refits_capped: int = -1
    world_value_0: tuple[float, float] | None = None
    static_spread: tuple[float, float] | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, float] = field(default_factory=dict)
    calibrations: int = 0
    cache_keys_touched: int = 0
    cache_hits: int = 0
    wall_seconds: float = 0.0
    notes: list[str] = field(default_factory=list)
    world_meta: dict[str, Any] = field(default_factory=dict)
    #: study D (:func:`delta_diagnostics`): ``mean_delta``, ``lambda_star``, ``std_at_lambda``,
    #: ``mv_delta_implied`` as ``(value, se)`` — in-sample benchmarks
    delta_diag: dict[str, tuple[float, float]] = field(default_factory=dict)
    #: the ``.npy`` of the per-path time-averaged delta (study D; written by
    #: :func:`save_result` next to the JSON, resolved by :func:`load_results`): paired
    #: comparisons of two rows' deltas on their shared world paths (``distance_to_mv`` is read
    #: against the common minimum-variance delta, :func:`common_mv_delta`, and does not use it);
    #: empty when absent
    delta_paths_file: str = ""
    #: the per-path time-averaged delta itself (not serialised)
    delta_path_means: FloatArray | None = field(default=None, repr=False, compare=False)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = _jsonable(
            {
                f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)
                if f.name != "delta_path_means"
            }
        )
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TaskResult:
        """The inverse of :meth:`to_dict` (JSON ``null`` reads back as NaN)."""
        names = {f.name for f in dataclasses.fields(cls)}
        kw = {k: v for k, v in d.items() if k in names}
        for k in (
            "value_0",
            "mean",
            "std",
            "zero_cost_mean",
            "recal_total",
            "world_value_0",
            "static_spread",
        ):
            if kw.get(k) is not None:
                kw[k] = (_fl(kw[k][0]), _fl(kw[k][1]))
        kw["quantiles"] = {k: (_fl(v[0]), _fl(v[1])) for k, v in kw.get("quantiles", {}).items()}
        kw["delta_diag"] = {k: (_fl(v[0]), _fl(v[1])) for k, v in kw.get("delta_diag", {}).items()}
        for k in ("rota", "wall_seconds", "scale"):
            if k in kw and kw[k] is not None:
                kw[k] = _fl(kw[k])
        return cls(**kw)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def desk_mean(self) -> tuple[float, float]:
        return float(to_desk_pnl(self.mean[0])), self.mean[1]

    def desk_recal_total(self) -> tuple[float, float]:
        return float(to_desk_pnl(self.recal_total[0])), self.recal_total[1]


def _fl(v: Any) -> float:
    """A float from JSON (``None`` and the ``"inf"`` tokens of :func:`_jsonable` read back)."""
    if v is None:
        return float("nan")
    if isinstance(v, str):
        return float(v) if v in ("inf", "-inf") else float("nan")
    return float(v)


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, float) and not np.isfinite(x):
        return None if np.isnan(x) else ("inf" if x > 0 else "-inf")
    return x


#: suffix of a study-D result's per-path time-averaged delta, next to its JSON
DELTA_PATHS_SUFFIX = ".delta.npy"


def delta_path_means(res: TaskResult) -> FloatArray | None:
    """A result's per-path time-averaged delta: the in-memory array, else the ``.npy`` its
    ``delta_paths_file`` names (``None`` when neither is there)."""
    if res.delta_path_means is not None:
        return np.asarray(res.delta_path_means, dtype=np.float64)
    if res.delta_paths_file and Path(res.delta_paths_file).is_file():
        return np.asarray(np.load(res.delta_paths_file), dtype=np.float64)
    return None


def _pair_means(x: FloatArray) -> FloatArray:
    """Means of consecutive path pairs (the world draws are antithetic pairs; pairing is valid
    for independent paths too), a trailing odd path dropped."""
    x = np.asarray(x, dtype=np.float64)
    m = x.size // 2
    return np.asarray(0.5 * (x[0 : 2 * m : 2] + x[1 : 2 * m : 2]), dtype=np.float64)


def _pair_mean_se(x: FloatArray) -> tuple[float, float]:
    """Mean and its standard error from the path pairs (:func:`_pair_means`)."""
    pm = _pair_means(x)
    if pm.size < 2:
        return float(np.mean(x)), float("nan")
    return float(np.mean(x)), float(pm.std(ddof=1) / np.sqrt(pm.size))


def delta_diagnostics(
    result: HedgeResult, scale: float, spot: float
) -> tuple[dict[str, tuple[float, float]], FloatArray]:
    """The study-D reading of one delta-only run (owner's decision of 2026-09-16), each as
    ``(value, se)``, and the per-path time-averaged delta:

    * ``mean_delta`` — the hedged ``delta`` over dates and world paths (terminated paths count as
      0), as a **relative delta** ``S₀ Δ × scale / 100``: the value change in the table's unit
      (÷ 100) per unit relative spot move — the plain per-unit-spot delta for the vanilla on one
      share (``scale = 100/S₀``), ``S₀ Δ`` per unit notional for the autocall; se from the path
      pairs;
    * ``lambda_star`` — the in-sample variance-minimising scale of the row's hedge leg
      ``λ* = −Cov(product leg, hedge leg)/Var(hedge leg)`` (the product leg: the P&L without the
      hedge legs);
    * ``std_at_lambda`` — the P&L std with the hedge leg scaled by ``λ*``;
    * ``mv_delta_implied`` — ``λ* × mean_delta``, the minimum-variance delta the row implies.

    ``λ*`` is fitted on the same paths it is scored on (in-sample): a benchmark the regime
    ranking is read against, not a strategy.  The se's of the last three are path-pair bootstrap
    standard deviations (:data:`DELTA_DIAG_BOOTSTRAP` resamples)."""
    if "delta" not in result.greeks_by_date:
        raise ValueError("delta_diagnostics needs the run's per-date delta (greeks_by_date)")
    rel = float(spot) * float(scale) / 100.0
    dmat = np.asarray(result.greeks_by_date["delta"], dtype=np.float64) * rel
    per_path = np.asarray(dmat.mean(axis=0), dtype=np.float64)
    hedge = np.asarray(result.pnl_hedges.sum(axis=1), dtype=np.float64) * scale
    other = np.asarray(result.pnl_total, dtype=np.float64) * scale - hedge

    def stats(o: FloatArray, h: FloatArray, d: FloatArray) -> tuple[float, float, float]:
        var_h = float(h.var(ddof=1))
        lam = -float(np.cov(o, h, ddof=1)[0, 1]) / var_h if var_h > 0.0 else float("nan")
        return lam, float(np.std(o + lam * h, ddof=1)), lam * float(d.mean())

    lam, s_lam, mv = stats(other, hedge, per_path)
    n_pairs = per_path.size // 2
    boot = np.full((DELTA_DIAG_BOOTSTRAP, 3), np.nan)
    if n_pairs >= 2:
        rng = np.random.default_rng(DELTA_DIAG_SEED)
        for b in range(DELTA_DIAG_BOOTSTRAP):
            pick = rng.integers(0, n_pairs, size=n_pairs)
            ix = np.concatenate([2 * pick, 2 * pick + 1])
            boot[b] = stats(other[ix], hedge[ix], per_path[ix])
    se = boot.std(axis=0, ddof=1) if n_pairs >= 2 else np.full(3, np.nan)
    out = {
        "mean_delta": _pair_mean_se(per_path),
        "lambda_star": (lam, float(se[0])),
        "std_at_lambda": (s_lam, float(se[1])),
        "mv_delta_implied": (mv, float(se[2])),
    }
    return out, per_path


def result_paths(out: Path, task_or_key: Task | str, study: str | None = None) -> tuple[Path, Path]:
    key = task_or_key.key if isinstance(task_or_key, Task) else task_or_key
    st = task_or_key.study if isinstance(task_or_key, Task) else (study or key.split("__")[0])
    d = Path(out) / st
    return d / f"{key}.json", d / f"{key}.pkl"


def save_result(res: TaskResult, hedge: HedgeResult | None, out: Path) -> Path:
    jp, pp = result_paths(out, res.key, res.study)
    jp.parent.mkdir(parents=True, exist_ok=True)
    if res.delta_path_means is not None:
        dp = jp.with_suffix(DELTA_PATHS_SUFFIX)
        np.save(dp, np.asarray(res.delta_path_means, dtype=np.float64))
        res.delta_paths_file = dp.name
    jp.write_text(json.dumps(res.to_dict(), indent=1), encoding="utf-8")
    if hedge is not None:
        # without the world paths and the per-date Greek / move arrays (n_dates × n_paths each:
        # 40 MB per Greek at the default budget); the attribution built from them is in the JSON
        small = dataclasses.replace(
            hedge,
            world_paths=None,  # type: ignore[arg-type]
            greeks_by_date={},
            moves_by_date={},
        )
        with pp.open("wb") as fh:
            pickle.dump(small, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return jp


def load_results(out: Path, study: str | None = None) -> list[TaskResult]:
    """Every ``TaskResult`` JSON under ``out`` (all shards), optionally one study."""
    out = Path(out)
    res: list[TaskResult] = []
    for st in STUDIES if study is None else (study,):
        d = out / st
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.json")):
            if p.name.startswith("static_") or p.name.startswith("_"):
                continue
            try:
                r = TaskResult.from_dict(json.loads(p.read_text(encoding="utf-8")))
            except (ValueError, TypeError, KeyError) as exc:
                log.warning("skipping unreadable result %s: %s", p, exc)
                continue
            if r.delta_paths_file and not Path(r.delta_paths_file).is_absolute():
                r.delta_paths_file = str(p.parent / r.delta_paths_file)
            res.append(r)
    return res


# --------------------------------------------------------------------------------------------
# environment: the marking state, the book, the worlds
# --------------------------------------------------------------------------------------------


class StudyEnvironment:
    """Lazily built shared objects of a study run: the cache, the marking state at the
    config's particle count, the LSV pricing context (and the LV one of study A), the book, the
    worlds.  Counts the leverage calibrations performed through it (``calibrations``)."""

    def __init__(self, cfg: StudyConfig) -> None:
        self.cfg = cfg
        self.cache = LeverageCache(cfg.cache)
        self.fit_spec = load_fit_spec(cfg.marking_fit)
        spec = dataclasses.replace(
            self.fit_spec.spec,
            particle=dataclasses.replace(
                self.fit_spec.spec.particle, n_particles=int(cfg.n_particles)
            ),
        )
        self.state = RiskState(spec, None, "marking fit")
        self.surface = surface_of(self.state)
        self.forward_curve = self.surface.forward_curve
        self.discount = self.forward_curve.rate_curve
        self.spot = float(self.state.spot)
        self._lsv: PricingContext | None = None
        self._lv: PricingContext | None = None
        self._book: dict[str, Product] | None = None
        self._worlds: dict[str, tuple[Model, dict[str, Any]]] = {}
        self._kvol: tuple[float, float] | None = None
        self.calibration_keys: list[str] = []
        self.notes: list[str] = []

    # -- calibration bookkeeping ---------------------------------------------------------------

    def _get_model(self, state: RiskState, label: str) -> Model:
        """An LSV from the cache (calibrating when allowed; recorded)."""
        miss = not self.cache.has(state.spec)
        if miss and not self.cfg.allow_calibrate:
            raise CacheMissError(
                f"{label}: no calibrated leverage for key {state.key} "
                f"({state.spec.particle.n_particles} particles; allow_calibrate=False)"
            )
        t0 = time.perf_counter()
        model, _ = self.cache.get_or_calibrate(state.spec, allow_calibrate=self.cfg.allow_calibrate)
        if miss:
            self.calibration_keys.append(state.key)
            msg = f"calibrated {label} ({state.key[:12]}) in {time.perf_counter() - t0:.0f} s"
            log.info(msg)
            self.notes.append(msg)
        return model

    @property
    def calibrations(self) -> int:
        n = len(self.calibration_keys)
        if self._lsv is not None and self._lsv.builder is not None:
            n += int(self._lsv.builder.n_cache_misses)
        return n

    # -- contexts ------------------------------------------------------------------------------

    @property
    def pricing_ctx(self) -> PricingContext:
        """The marking-fit LSV context (built once; a fresh :class:`PricingContext` sharing the
        builder is handed to each run by :meth:`fresh_ctx`)."""
        if self._lsv is None:
            if not self.cache.has(self.state.spec) and not self.cfg.allow_calibrate:
                raise CacheMissError(
                    f"pricing model: no calibrated leverage for the marking fit at "
                    f"{self.cfg.n_particles} particles (key {self.state.key})"
                )
            self._lsv = PricingContext.from_state(
                self.state,
                self.cache,
                "lsv",
                allow_calibrate=self.cfg.allow_calibrate,
                label="2F marking fit (SPX 2022-12-30, ssr 1, eps 0.10)",
            )
        return self._lsv

    @property
    def lv_ctx(self) -> PricingContext:
        if self._lv is None:
            self._lv = PricingContext.from_state(
                self.state, None, "lv", label="LV (Dupire of the SPX 2022-12-30 surface)"
            )
        return self._lv

    def fresh_ctx(self, pricing: str) -> PricingContext:
        base = self.lv_ctx if pricing == "LV" else self.pricing_ctx
        builder = base.builder
        if (
            pricing != "LV"
            and builder is not None
            and self.cfg.refit_particles != self.cfg.n_particles
        ):
            builder = RefitParticlesBuilder(builder, self.cfg.refit_particles)
        return PricingContext(base.model, base.state, builder, base.surface, base.label)

    def sim(self, frequency: str) -> SimConfig:
        """Pricing simulation of a task: the step is the rebalancing step capped at
        :data:`SIM_DT_MAX` (weekly)."""
        return SimConfig(
            n_paths=int(self.cfg.n_paths),
            chunk_size=int(self.cfg.n_paths),
            seed=int(self.cfg.seed),
            dt_max=float(min(FREQUENCIES[frequency], SIM_DT_MAX)),
        )

    # -- the book ------------------------------------------------------------------------------

    def book(self) -> dict[str, Product]:
        if self._book is None:
            disc, spot = self.discount, self.spot
            prods: dict[str, Product] = dict(headline_products(disc, spot))
            prods[CLIQUET_NAME] = AdditiveCliquet.study(1.0, disc)
            prods[VKO_NAME] = VolKnockOutPut(
                spot, 1.0, VKO_VOL_KO, daily_schedule(1.0, 252), disc, notional=1.0 / spot
            )
            prods[KOVAR_NAME] = KnockOutVarianceSwap(
                daily_schedule(1.0, 252), KOVAR_BARRIER * spot, KOVAR_STRIKE_VOL, disc
            )
            self._book = prods
        return self._book

    def product(self, name: str) -> Product:
        if name == FVA_NAME:
            return FVA(
                1.0,
                2.0,
                float(self.surface.atm_vol(2.0)),
                self.discount,
                forward_curve=self.forward_curve,
            )
        if name == VANILLA_NAME:
            return EuropeanOption(self.spot, 1.0, 1, self.discount)
        return self.book()[name]

    def kovar_fair_strike(self, model: Model | None = None) -> tuple[float, float]:
        """``(K_vol, stderr)`` of the KO var under the pricing model at ``t = 0``
        (:func:`~volsto.analytics.conditional_variance.fair_strike`, the study budget)."""
        if self._kvol is None:
            ko = self.book()[KOVAR_NAME]
            assert isinstance(ko, KnockOutVarianceSwap)
            fs = ko.fair_strike(model or self.pricing_ctx.model, self.sim("daily"))
            kvol = float(np.sqrt(fs.variance))
            se = float(fs.variance_stderr / (2.0 * kvol)) if kvol > 0 else float("nan")
            self._kvol = (kvol, se)
        return self._kvol

    def unit_of(self, name: str, pricing_model: Model | None = None) -> tuple[str, float, str]:
        """``(unit label, scale, note)`` of a product's P&L (module docstring)."""
        if name == KOVAR_NAME:
            kvol, se = self.kovar_fair_strike(pricing_model)
            return (
                "vol points of vega notional",
                100.0 / (2.0 * kvol),
                f"variance P&L / (2 K_vol) with K_vol = {kvol:.5f} +/- {se:.5f} (pricing model's "
                f"fair strike at t = 0, {self.cfg.n_paths} paths)",
            )
        if name == VANILLA_NAME:
            return "% of spot", 100.0 / self.spot, "vanilla on one share: value / spot"
        if name == FVA_NAME:
            return "vol points x notional", 100.0, "FVA pays the vol difference: value x 100"
        return "% of notional", 100.0, "notional 1 (VKO 1/spot): value x 100"

    # -- worlds --------------------------------------------------------------------------------

    def world_model(self, name: str, pricing: str = "2F") -> tuple[Model, dict[str, Any]]:
        """Study B worlds by name (and ``"pricing"`` = the pricing model of ``pricing``)."""
        memo = f"{name}|{pricing}"
        if memo in self._worlds:
            return self._worlds[memo]
        meta: dict[str, Any] = {"world": name}
        if name == "pricing":
            model = self.fresh_ctx(pricing).model
        elif name == "same":
            model = self.pricing_ctx.model
        elif name == "pure LV":
            model = self.lv_ctx.model
        elif name == "nu x1.5":
            nu = float(self.state.spec.model.nu) * NU_SCALE
            st = self.state.with_params(label=f"nu x{NU_SCALE:g}", nu=nu)
            meta.update({"nu": nu, "key": st.key, "cached_before": bool(self.cache.has(st.spec))})
            model = self._get_model(st, name)
        elif name == WORLD_HISTORICAL_NAME:
            hs = load_fit_spec(self.cfg.world_historical)
            spec = dataclasses.replace(
                hs.spec,
                particle=dataclasses.replace(
                    hs.spec.particle, n_particles=int(self.cfg.n_particles)
                ),
            )
            if spec.market != self.state.spec.market or spec.surface != self.state.spec.surface:
                raise ValueError(
                    "the historical world spec must carry the SPX marking market/surface"
                )
            st = RiskState(spec, None, "historical fit")
            meta.update(
                {
                    "params": repr(spec.model),
                    "key": st.key,
                    "cached_before": bool(self.cache.has(spec)),
                    "fit_mode": hs.fit.get("mode"),
                }
            )
            model = self._get_model(st, name)
        else:
            raise ValueError(f"unknown world {name!r}")
        self._worlds[memo] = (model, meta)
        return model, meta

    def shock_world(self, rota: float, t0: float) -> tuple[Model, dict[str, Any]]:
        memo = f"skew shock|{rota:g}|{t0:g}"
        if memo in self._worlds:
            return self._worlds[memo]
        rs = shock_state(self.state, float(rota))
        if not self.cache.has(rs.spec) and not self.cfg.allow_calibrate:
            raise CacheMissError(
                f"skew-shock world rota {rota:+g}: no calibrated leverage for the rotated "
                f"surface (key {rs.key}, {self.cfg.n_particles} particles; allow_calibrate=False)"
            )
        model, meta = skew_shock_world(
            self.state,
            self.cache,
            rota=float(rota),
            t0=float(t0),
            days=SHOCK_DAYS,
            allow_calibrate=self.cfg.allow_calibrate,
        )
        if meta.get("calibrated"):
            self.calibration_keys.append(str(meta["rotated_key"]))
        self._worlds[memo] = (model, meta)
        return model, meta


# --------------------------------------------------------------------------------------------
# strategies and the hedger of a task
# --------------------------------------------------------------------------------------------


def build_strategy(task: Task, product: Product, hedger: Hedger) -> Strategy:
    """The strategy of a task (module docstring)."""
    pc = hedger.preset_context(product)
    kw = task.kwargs_dict
    if task.study == "A":
        if task.strategy == "delta only":
            return GreekTargetStrategy((Target("delta"),), [Spot()], name="delta only")
        if task.product == CLIQUET_NAME:
            s = default_strategy(product, pc, q=kw["q"])
            if not kw.get("var_swap", True):
                # without the swap no remaining instrument spans vega: the strategy is the
                # delta hedge of the netted book (the vega target would only cost two
                # pricing simulations and the parallel-bump leverage for nothing)
                s = s.without("var swap").with_targets(["delta"])
                s.preset_kwargs = {"q": kw["q"], "var_swap": False}
            s.name = task.strategy
            return s
        s = default_strategy(product, pc, **({"skew": True} if kw.get("skew") else {}))
        s.name = task.strategy
        return s
    if task.study == "D":
        assert task.regime is not None and task.regime in DELTA_REGIMES
        return GreekTargetStrategy(
            (Target("delta"),),
            [Spot()],
            delta_regime=task.regime,
            name=f"delta only ({task.regime})",
        )
    return default_strategy(product, pc)


def study_c_rule(policy: str, env: StudyEnvironment) -> RecalibrationRule:
    """Study C's recalibration rule under ``policy``: the simulated refit solves the SAME
    constraint as the static greek's marking fit (SPX: the two-point skew constraint at 1Y / 3Y,
    the 5Y pillar being beyond the snapshot), the rule reads the world's state surface at
    :data:`STUDY_C_RULE_PILLARS` from strips of :data:`STUDY_C_STRIP_PATHS` paths (struck at
    forward moneyness, the curvature on :data:`STUDY_C_CURVATURE_H`), triggers at
    :data:`STUDY_C_SKEW_MOVE_THRESHOLD` and caps the refit's correlation target at the hedger's
    :data:`~volsto.hedging.hedger.REFIT_CORRELATION_CAP` (with the guarded fallback,
    :func:`~volsto.hedging.hedger.refit_targets`)."""
    base = env.fit_spec.config
    rcfg = dataclasses.replace(
        base,
        pillars=STUDY_C_RULE_PILLARS,
        mat_min=0.0,
        skew_pillars=STUDY_C_RULE_SKEW_PILLARS,
        # the state surfaces are quadratic smiles without wings: the first-order engine
        engine=None,
    )
    return RecalibrationRule(
        pillars=STUDY_C_RULE_PILLARS,
        fit_config=rcfg,
        step0=env.fit_spec.step0_source(env.surface),
        step0_surface=env.surface if base.step0 is not None else None,
        policy=policy,
        ssr_target=env.fit_spec.ssr_target,
        skew_move_threshold=STUDY_C_SKEW_MOVE_THRESHOLD,
        strip_paths=STUDY_C_STRIP_PATHS,
        correlation_cap=REFIT_CORRELATION_CAP,
        curvature_h=STUDY_C_CURVATURE_H,
    )


def make_hedger(task: Task, env: StudyEnvironment) -> tuple[Hedger, Product, dict[str, Any]]:
    """The hedger, product and world metadata of a task (no run)."""
    cfg = env.cfg
    product = env.product(task.product)
    T = float(product.maturity)
    ctx = env.fresh_ctx(task.pricing)
    meta: dict[str, Any]
    rule: RecalibrationRule | None = None
    cls: type[Hedger] = Hedger
    if task.study == "C":
        assert task.rota is not None and task.policy is not None
        world, meta = env.shock_world(task.rota, 0.5 * T)
        if task.policy != "none":
            if task.policy not in RECALIBRATION_POLICIES:
                raise ValueError(f"policy must be 'none' or one of {RECALIBRATION_POLICIES}")
            rule = study_c_rule(task.policy, env)
            meta["rule_config"] = {
                "pillars": list(STUDY_C_RULE_PILLARS),
                "skew_pillars": list(STUDY_C_RULE_SKEW_PILLARS),
                "skew_eps": rule.config().skew_eps,
                "skew_move_threshold": rule.skew_move_threshold,
                "strip_paths": rule.strip_paths,
                "strip_h": rule.h,
                "curvature_h": rule.curvature_stencil,
                "strip_strikes": "forward moneyness F(t+tau)/F(t) e^k",
                "correlation_cap": rule.correlation_cap,
                "static_greek_config": "fit_2f_marking on the full snapshot surface with the "
                "marking fit spec's config (and its snapshot's SABRW fits when its step 0 "
                "reads them), the skew pillars relocated to the snapshot's last pillar (1Y / 3Y "
                "on SPX)",
            }
            cls = RecordingHedger
    elif task.study == "B":
        world, meta = env.world_model(task.world)
    else:
        world, meta = env.world_model("pricing", task.pricing)
    h = cls(
        ctx,
        world,
        Schedule(task.frequency, product_fixings=STUDY_PRODUCT_FIXINGS),
        Costs(),
        recalibration=rule,
        sim=env.sim(task.frequency),
        world_paths=cfg.n_world,
        verbose=cfg.verbose,
        stream_bumps=cfg.stream_bumps,
        control_variate=cfg.control_variate,
        control_delta=cfg.control_delta,
    )
    return h, product, dict(meta)


def summarize(
    task: Task,
    env: StudyEnvironment,
    hedger: Hedger,
    product: Product,
    result: HedgeResult,
    meta: Mapping[str, Any],
    wall: float,
    calibrations: int,
    builder_before: tuple[int, int] = (0, 0),
) -> TaskResult:
    unit, scale, unit_note = env.unit_of(task.product, hedger.context.model)
    x = result.pnl_total * scale
    dist = distribution_table(x, "total")
    q = {
        str(r["statistic"]): (float(r["value"]), float(r["stderr"]))
        for r in dist.to_dict(orient="records")
        if str(r["statistic"]).startswith("q")
    }
    std_row = dist[dist["statistic"] == "std"].iloc[0]
    reg = regime_table(result)
    for c in ("mean", "stderr", "std"):
        reg[c] = reg[c] * scale
    att = attribution_table(result)
    for c in ("mean", "stderr"):
        att[c] = att[c] * scale
    recal_rows: list[dict[str, float]] = []
    notes = [*result.pricing_notes, unit_note]
    if isinstance(hedger, RecordingHedger):
        recal_rows, note = recalibration_by_date(hedger, result)
        for r in recal_rows:
            r["mean"] *= scale
            r["stderr"] *= scale
        if note:
            notes.append(note)
    n_refits = int(result.budget.get("refits", 0))
    world_v0: tuple[float, float] | None = None
    spread: tuple[float, float] | None = None
    if task.study == "B":
        wsim = hedger._world_sim()
        pr = MonteCarlo(wsim).price(product, hedger.world)
        world_v0 = (float(pr.mean) * scale, float(pr.stderr) * scale)
        spread = (
            (result.value_0 - float(pr.mean)) * scale,
            float(np.hypot(result.value_0_stderr, pr.stderr)) * scale,
        )
    # per-task cache accounting: the builder's counters are cumulative over the environment, so
    # the task's share is the difference to the snapshot taken before its run
    builder = hedger.context.builder
    keys_now = (
        int(builder.n_calibrations)
        if builder is not None and hasattr(builder, "n_calibrations")
        else 0
    )
    miss_now = (
        int(builder.n_cache_misses)
        if builder is not None and hasattr(builder, "n_cache_misses")
        else 0
    )
    touched = max(keys_now - builder_before[0], 0)
    misses = max(miss_now - builder_before[1], 0)
    diag: dict[str, tuple[float, float]] = {}
    per_path: FloatArray | None = None
    if task.study == "D" and "delta" in result.greeks_by_date:
        diag, per_path = delta_diagnostics(result, scale, float(hedger.context.model.spot))
    return TaskResult(
        key=task.key,
        study=task.study,
        product=task.product,
        world=task.world,
        strategy=result.strategy,
        pricing=task.pricing,
        rota=task.rota,
        policy=task.policy,
        regime=task.regime,
        frequency=task.frequency,
        unit=unit,
        scale=scale,
        n_paths_pricing=int(hedger.sim.n_paths),
        n_paths_world=int(result.n_paths),
        n_particles=int(env.cfg.n_particles),
        n_dates=int(result.dates.size),
        value_0=(result.value_0 * scale, result.value_0_stderr * scale),
        mean=_mean_se(x),
        std=(float(std_row["value"]), float(std_row["stderr"])),
        quantiles=q,
        zero_cost_mean=_mean_se(result.pnl_zero_cost * scale),
        regimes=[_jsonable(r) for r in reg.to_dict(orient="records")],
        attribution=[_jsonable(r) for r in att.to_dict(orient="records")],
        recal_total=_mean_se(result.pnl_recalibration * scale),
        recal_by_date=recal_rows,
        n_refits=n_refits,
        n_refits_at_bound=refits_at_bound(result.recalibrations),
        n_refits_fallback=refits_fallback(result.recalibrations),
        n_refits_capped=refits_capped(result.recalibrations),
        world_value_0=world_v0,
        static_spread=spread,
        settings=_jsonable(result.settings),
        budget={k: float(v) for k, v in result.budget.items()},
        calibrations=int(calibrations),
        cache_keys_touched=touched,
        cache_hits=max(touched - misses, 0),
        wall_seconds=float(wall),
        notes=notes,
        world_meta=_jsonable(meta),
        delta_diag=diag,
        delta_path_means=per_path,
    )


def run_task(
    task: Task, cfg: StudyConfig, env: StudyEnvironment | None = None, *, save: bool = True
) -> TaskResult:
    """Run one task under the standing rules: a :class:`CacheMissError` (under
    ``allow_calibrate=False``) becomes a ``skipped`` result with the reason; the numbers are
    written to ``<out>/<study>/<key>.json`` (+ ``.pkl``) when ``save``."""
    env = env or StudyEnvironment(cfg)
    t0 = time.perf_counter()
    cal0 = env.calibrations
    hedge: HedgeResult | None = None
    try:
        hedger, product, meta = make_hedger(task, env)
        strategy = build_strategy(task, product, hedger)
        b = hedger.context.builder
        keys0 = int(getattr(b, "n_calibrations", 0)) if b is not None else 0
        miss0 = int(getattr(b, "n_cache_misses", 0)) if b is not None else 0
        hedge = hedger.run(product, strategy)
        res = summarize(
            task,
            env,
            hedger,
            product,
            hedge,
            meta,
            time.perf_counter() - t0,
            env.calibrations - cal0,
            builder_before=(keys0, miss0),
        )
    except CacheMissError as exc:
        res = TaskResult(
            task.key,
            task.study,
            task.product,
            task.world,
            task.strategy,
            task.pricing,
            task.rota,
            task.policy,
            task.regime,
            task.frequency,
            status="skipped",
            reason=f"leverage not cached (tests never calibrate): {exc}",
            wall_seconds=time.perf_counter() - t0,
            n_particles=int(cfg.n_particles),
        )
    if save:
        save_result(res, hedge, Path(cfg.out))
    return res


# --------------------------------------------------------------------------------------------
# projection (dry run)
# --------------------------------------------------------------------------------------------


def n_pricing_models(strategy: Strategy) -> int:
    """The pricing simulations a strategy needs (the hedger's ``n_models``): base + 2 spot bumps
    + one per one-sided target bump (tents, buckets) + two per two-sided one (vega, volga,
    param, a non-model delta regime); vanna adds vega when absent."""
    names = list(strategy.target_names)
    n = 3
    two = 0
    for t in names:
        if t in ("vega", "volga") or t.startswith("param:"):
            two += 1
        elif t.startswith(("fwd_var:", "skew_T:", "curvature_T:")):
            n += 1
        elif t == "delta" and getattr(strategy, "delta_regime", "model") in SURFACE_DELTA_REGIMES:
            two += 1
    if "vanna" in names and "vega" not in names:
        two += 1
    return n + 2 * two


def required_states(
    task: Task, env: StudyEnvironment, strategy: Strategy
) -> list[tuple[str, RiskState]]:
    """The leverage-cache states a task will ask for before any refit (module docstring): the
    pricing base, the strategy's bump states, the world's.  Only LSV pricing needs any."""
    out: list[tuple[str, RiskState]] = []
    if task.pricing == "LV":
        return out
    base = env.state
    out.append(("pricing base", base))
    names = list(strategy.target_names)
    if any(t in ("vega", "volga", "vanna") for t in names):
        for s in (VOL_BUMP, -VOL_BUMP):
            out.append(
                (
                    f"parallel {s:+g}",
                    base.with_perturbation(SurfacePerturbation("parallel", {"size": s})),
                )
            )
    for t in names:
        if t.startswith("skew_T:"):
            T = float(t.split(":")[1])
            pillars = tuple(sorted(set(RISK_PILLARS) | {T}))
            # the hedger halves a tent that fails the surface's arbitrage checks (at most
            # MAX_HALVINGS times) and calibrates the size that passes: the state the run will
            # ask for is the first cached one along that sequence (a run has been through the
            # checks before), else the nominal size is the best guess
            states = []
            for k in range(MAX_HALVINGS + 1):
                size = TENT_SIZE * 0.5**k
                pert = SurfacePerturbation(
                    "skew_tent",
                    {"pillars": pillars, "index": pillars.index(T), "slope": skew_slope(size)},
                )
                states.append((k, base.with_perturbation(pert)))
            cached = next((ks for ks in states if env.cache.has(ks[1].spec)), None)
            if cached is not None:
                k, st = cached
                lab = f"{t} tent" + (f" (halved {k}x, cached)" if k else "")
            else:
                k, st = states[0]
                lab = f"{t} tent (+halvings if the check fails)"
            out.append((lab, st))
        if t.startswith("param:"):
            out.append((f"{t} (two states)", base))
    regime = getattr(strategy, "delta_regime", "model")
    if "delta" in names and regime in SURFACE_DELTA_REGIMES:
        for h in (SPOT_BUMP, -SPOT_BUMP):
            st, mode = spot_state(base, regime, h)
            if mode == "recalibrate":
                out.append((f"delta regime {regime} {h:+g}", st))
    if task.study == "B" and task.world == "nu x1.5":
        out.append(("world nu x1.5", base.with_params(nu=float(base.spec.model.nu) * NU_SCALE)))
    if task.study == "B" and task.world == WORLD_HISTORICAL_NAME:
        hs = load_fit_spec(env.cfg.world_historical)
        spec = dataclasses.replace(
            hs.spec,
            particle=dataclasses.replace(hs.spec.particle, n_particles=int(env.cfg.n_particles)),
        )
        out.append(("world historical", RiskState(spec)))
    if task.study == "C" and task.rota is not None:
        out.append((f"world rota {task.rota:+g}", shock_state(base, task.rota)))
    return out


@dataclass
class Projection:
    study: str
    n_tasks: int
    hedger_seconds: float
    missing_keys: dict[str, str]
    calibration_seconds: float
    per_task: list[dict[str, Any]]
    notes: list[str]

    @property
    def total_seconds(self) -> float:
        return self.hedger_seconds + self.calibration_seconds

    def summary(self) -> str:
        return (
            f"study {self.study}: {self.n_tasks} tasks; projected hedger wall clock "
            f"{self.hedger_seconds:.0f} s ({self.hedger_seconds / 3600:.2f} h); "
            f"{len(self.missing_keys)} leverage calibration(s) needed up front "
            f"(~{self.calibration_seconds:.0f} s); total ~{self.total_seconds / 3600:.2f} h"
            + (f"; {'; '.join(self.notes)}" if self.notes else "")
        )


def calibration_seconds(cache: LeverageCache, n_particles: int) -> float:
    """The manifest's median wall time at ``n_particles`` (else the 8·10⁵ fallback scaled)."""
    m = cache.manifest()
    if not m.empty and "n_particles" in m and "wall_time" in m:
        sel = m[m["n_particles"] == int(n_particles)]["wall_time"].dropna()
        if len(sel) > 0:
            return float(sel.median())
    return FALLBACK_CALIBRATION_SECONDS_8E5 * n_particles / 800_000.0


def project(tasks: Sequence[Task], env: StudyEnvironment) -> Projection:
    """The projected wall clock of a task list (module docstring, *Budget*); the hedger's
    probe runs on the pricing model (the world of study C is approximated by the pricing model
    for the grid — the shock adds two slices) with the strategy's bump targets when every state
    they need is cached (a dry run never calibrates: a task with a missing state is probed
    without its bumps and the projection says so)."""
    if not tasks:
        return Projection("-", 0, 0.0, {}, 0.0, [], [])
    study = tasks[0].study
    per_task: list[dict[str, Any]] = []
    missing: dict[str, str] = {}
    notes: list[str] = []
    memo: dict[tuple[str, str, str, str], float] = {}
    strip_memo: dict[tuple[str, str], float] = {}
    total = 0.0
    for task in tasks:
        product = env.product(task.product)
        ctx = env.fresh_ctx(task.pricing)
        h = Hedger(
            ctx,
            ctx.model,
            Schedule(task.frequency, product_fixings=STUDY_PRODUCT_FIXINGS),
            Costs(),
            sim=env.sim(task.frequency),
            world_paths=env.cfg.n_world,
            verbose=False,
        )
        strategy = build_strategy(task, product, h)
        need = required_states(task, env, strategy)
        miss_here = [lab for lab, st in need if not env.cache.has(st.spec)]
        mk = (task.pricing, task.product, strategy.name, task.frequency)
        if mk not in memo:
            objects: list[Product] = [product] + [
                i.product for i in strategy.instruments if i.product is not None
            ]
            dates = Schedule(task.frequency, product_fixings=STUDY_PRODUCT_FIXINGS).build(product)
            grid = union_grid([ctx.model], objects, dates, h.sim)
            bumps: list[Bump] = []
            if not miss_here:
                regime = getattr(strategy, "delta_regime", "model")
                for name in strategy.target_names:
                    b = ctx.bump(name, delta_regime=regime)
                    if b is not None:
                        bumps.append(b)
            elif "probe without bump targets where a leverage is missing" not in notes:
                notes.append("probe without bump targets where a leverage is missing")
            memo[mk] = h.projected_wall_clock(
                product, strategy, grid, n_pricing_models(strategy), bumps
            )
        secs = memo[mk]
        strip_secs = 0.0
        if task.study == "C" and task.policy not in (None, "none"):
            # the recalibration rule's two strips (world + twin) at their own path count; the
            # world strip is probed on the pricing model (the shock world has the same grid)
            sk = (task.product, task.frequency)
            if sk not in strip_memo:
                strip_memo[sk] = h.projected_strip_seconds(
                    product, study_c_rule(str(task.policy), env)
                )
            strip_secs = strip_memo[sk]
            secs = secs + strip_secs
        total += secs
        for lab, st in need:
            if not env.cache.has(st.spec):
                missing.setdefault(st.key, lab)
        per_task.append(
            {
                "key": task.key,
                "projected_seconds": secs,
                "strip_seconds": strip_secs,
                "missing": miss_here,
                "targets": list(strategy.target_names),
            }
        )
    if study == "C":
        notes.append(
            "each refit inside a recalibration run is one more calibration at "
            f"{env.cfg.refit_particles} particles unless cached (not counted ahead)"
        )
        notes.append(
            "the recalibration rows count the rule's two strips at "
            f"{STUDY_C_STRIP_PATHS} paths (world + pricing-model twin), "
            f"{sum(r['strip_seconds'] for r in per_task):.0f} s in total"
        )
    if any(t.study == "A" and t.pricing == "LV" for t in tasks):
        notes.append("LV pricing needs no leverage")
    per_cal = calibration_seconds(env.cache, env.cfg.n_particles)
    return Projection(study, len(tasks), total, missing, per_cal * len(missing), per_task, notes)


# --------------------------------------------------------------------------------------------
# the static prediction of study C (the M7 greek)
# --------------------------------------------------------------------------------------------


def static_prediction_path(out: Path, product: str, policy: str) -> Path:
    return Path(out) / "C" / f"static_{slug(product)}__{slug(policy)}.json"


def static_prediction(
    product_name: str,
    policy: str,
    env: StudyEnvironment,
    *,
    save: bool = True,
) -> dict[str, Any]:
    """``d(fee)/d(rota)`` of a product under ``policy`` in the desk convention
    (:func:`~volsto.risk.shadow_rotation.rotation_shadow_sensitivity` on the M7 base spec at the
    config's rotation particles / paths), scaled to the product's unit; cached as JSON under
    ``<out>/C`` (resume) with the marking fit it was computed from — a cached prediction of
    another marking fit (the model or the fit config; files written before 2026-09-27 record
    neither) is recomputed, never reused."""
    cfg = env.cfg
    p = static_prediction_path(cfg.out, product_name, policy)
    marking = {
        "model": to_mapping(env.fit_spec.spec.model),
        "config": to_mapping(env.fit_spec.config),
    }
    if p.exists():
        cached: dict[str, Any] = dict(json.loads(p.read_text(encoding="utf-8")))
        if (
            int(cached.get("n_paths", -1)) == cfg.rotation_paths
            and int(cached.get("n_particles", -1)) == cfg.rotation_particles
            and cached.get("marking") == json.loads(json.dumps(marking))
        ):
            return {**cached, "from_cache": True}
    t0 = time.perf_counter()
    product = env.product(product_name)
    unit, scale, _ = env.unit_of(product_name)
    misses0 = len(env.cache.manifest())
    base_spec = spx_base_spec(cfg.rotation_particles)
    rep = rotation_shadow_sensitivity(
        product,
        base_spec,
        env.fit_spec.config,
        step0=env.fit_spec.step0_source(surface_of(RiskState(base_spec))),
        ssr_target=float(env.fit_spec.ssr_target),
        cache=env.cache,
        pricing_sim=SimConfig(
            n_paths=int(cfg.rotation_paths), chunk_size=100_000, seed=DEFAULT_ROTATION_SEED
        ),
        allow_calibrate=cfg.allow_calibrate,
        product_name=product_name,
        policy=policy,
    )
    base_params = rep.fits["base"].params
    same_fit = all(
        abs(float(getattr(base_params, f.name)) - float(getattr(env.fit_spec.spec.model, f.name)))
        < 1e-6
        for f in dataclasses.fields(base_params)
    )
    doc: dict[str, Any] = {
        "product": product_name,
        "policy": policy,
        "convention": ROTATION_CONVENTION,
        "unit": unit,
        "n_paths": int(cfg.rotation_paths),
        "n_particles": int(cfg.rotation_particles),
        "seed": DEFAULT_ROTATION_SEED,
        "size": rep.size,
        "marking": marking,
        "base_fit_equals_marking_fit": bool(same_fit),
        "n_calibrations": int(rep.n_calibrations),
        "n_cache_misses": int(rep.n_cache_misses),
        "recalibrated": bool(rep.recalibrated_any),
        "manifest_growth": int(len(env.cache.manifest()) - misses0),
        "wall_seconds": float(time.perf_counter() - t0),
    }
    for name in (
        "p1_level",
        "lv_level",
        "fee",
        "lv_rotation",
        "usual",
        "recalibrated",
        "fee_shadow",
        "desk_pnl_usual",
        "desk_pnl_recalibrated",
        "desk_pnl_shadow",
    ):
        s = getattr(rep, name)
        doc[name] = [float(s.value) * scale, float(s.stderr) * scale]
    if rep.recalibrated_any:
        env.calibration_keys.extend(
            [f"rotation greek {product_name} {policy}"] * int(rep.n_cache_misses)
        )
    if save:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(_jsonable(doc), indent=1), encoding="utf-8")
    return doc


def load_static_predictions(out: Path) -> dict[tuple[str, str], dict[str, Any]]:
    d = Path(out) / "C"
    res: dict[tuple[str, str], dict[str, Any]] = {}
    if d.is_dir():
        for p in sorted(d.glob("static_*.json")):
            doc = json.loads(p.read_text(encoding="utf-8"))
            res[(str(doc["product"]), str(doc["policy"]))] = dict(doc)
    return res


# --------------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------------


def _pm(v: float | None, se: float | None, digits: int = 4) -> str:
    if v is None or not np.isfinite(v):
        return "nan"
    return (
        f"{v:+.{digits}f} +/- {se:.{digits}f}"
        if se is not None and np.isfinite(se)
        else f"{v:+.{digits}f}"
    )


def _pm_doc(doc: Mapping[str, Any], name: str, digits: int = 4) -> str:
    """``value +/- stderr`` of a ``[value, stderr]`` entry of a static-prediction record."""
    pair = doc.get(name)
    if not isinstance(pair, list | tuple) or len(pair) != 2:
        return "nan"
    return _pm(_fl(pair[0]), _fl(pair[1]), digits)


def table_A(results: Iterable[TaskResult]) -> pd.DataFrame:
    """Study A: per pricing model and product the strategies ranked by P&L std (with its
    standard error), the mean P&L (hedger sign is irrelevant for the std; the mean is reported
    in the desk convention)."""
    rows = []
    for r in results:
        if r.study != "A":
            continue
        rows.append(
            {
                "pricing": r.pricing,
                "product": r.product,
                "strategy": r.strategy,
                "status": r.status,
                "std": r.std[0],
                "std_se": r.std[1],
                "desk_mean": r.desk_mean()[0],
                "desk_mean_se": r.mean[1],
                "unit": r.unit,
                "n_paths": r.n_paths_world,
                "wall_s": r.wall_seconds,
                "reason": r.reason,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values(["pricing", "product", "std"], na_position="last").reset_index(drop=True)
    df["rank"] = df.groupby(["pricing", "product"])["std"].rank(method="first")
    return df


def _top_attribution(att: Sequence[Mapping[str, Any]], n: int = 3) -> str:
    comps = [
        a
        for a in att
        if str(a.get("component", "")) not in ("total", "product total")
        and a.get("mean") is not None
        and np.isfinite(float(a["mean"]))
    ]
    comps.sort(key=lambda a: -abs(float(a["mean"])))
    return "; ".join(f"{a['component']}: {-float(a['mean']):+.4f}" for a in comps[:n])


def _regime_ref(reg: Sequence[Mapping[str, Any]]) -> str:
    out = []
    for r in reg:
        lab = str(r.get("regime", ""))
        if lab.startswith("realised vol") or lab.startswith("early") or lab.startswith("no term"):
            m = r.get("mean")
            if m is not None and np.isfinite(float(m)):
                out.append(f"{lab.split(' (')[0]}: {-float(m):+.4f}")
    return "; ".join(out)


def table_B(
    results: Iterable[TaskResult], skipped: Sequence[Mapping[str, Any]] = ()
) -> pd.DataFrame:
    """Study B: per world × product the **desk** leakage (mean ± se; ``leakage_desk_incl_v0``
    repeats it with the pricing error of ``V₀`` added in quadrature to its se — the hedged mean
    is measured against the run's own ``V₀`` estimate), std, q05 / q95 (± the distribution
    table's quantile se), the static spread (marked − world price at ``t = 0``), the dynamic
    leakage (leakage − static spread, se in quadrature), the leakage **relative to the ``same``
    world** of
    the product (the engine's own baseline: pricing error plus the hedge legs' regression drift —
    the model reserve is the difference), the regime means (desk sign) and the top attribution
    terms (desk sign); the gated rows."""
    res = [r for r in results if r.study == "B"]
    same = {r.product: r for r in res if r.world == "same" and r.ok}
    rows = []
    for r in res:
        dm, dse = r.desk_mean()
        sp = r.static_spread
        nan2 = (float("nan"), float("nan"))
        q05, q95 = r.quantiles.get("q05", nan2), r.quantiles.get("q95", nan2)
        base = same.get(r.product)
        vs_same = (
            (dm - base.desk_mean()[0], float(np.hypot(dse, base.mean[1])))
            if base is not None and r.ok
            else (float("nan"), float("nan"))
        )
        rows.append(
            {
                "world": r.world,
                "product": r.product,
                "status": r.status,
                "unit": r.unit,
                "leakage_desk": dm,
                "leakage_desk_se": dse,
                "leakage_desk_incl_v0": dm,
                "leakage_desk_incl_v0_se": float(np.hypot(dse, r.value_0[1])),
                "leakage_vs_same": vs_same[0],
                "leakage_vs_same_se": vs_same[1],
                "std": r.std[0],
                "std_se": r.std[1],
                # the desk is short: its q05 is minus the hedger's q95 (same se)
                "q05_desk": -q95[0],
                "q05_desk_se": q95[1],
                "q95_desk": -q05[0],
                "q95_desk_se": q05[1],
                "static_spread": sp[0] if sp else float("nan"),
                "static_spread_se": sp[1] if sp else float("nan"),
                "dynamic_leakage": (dm - sp[0]) if sp else float("nan"),
                "dynamic_leakage_se": float(np.hypot(dse, sp[1])) if sp else float("nan"),
                "value_0": r.value_0[0],
                "value_0_se": r.value_0[1],
                "regimes_desk": _regime_ref(r.regimes),
                "attribution_top_desk": _top_attribution(r.attribution),
                "n_paths": r.n_paths_world,
                "wall_s": r.wall_seconds,
                "calibrations": r.calibrations,
                "reason": r.reason,
            }
        )
    for s in skipped:
        rows.append(
            {
                "world": s["world"],
                "product": s["product"],
                "status": s["status"],
                "reason": s.get("reason", ""),
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    order = {w: i for i, w in enumerate(WORLDS_B)}
    porder = {p: i for i, p in enumerate(BOOK)}
    df["_w"] = df["world"].map(order)
    df["_p"] = df["product"].map(porder)
    return df.sort_values(["_w", "_p"]).drop(columns=["_w", "_p"]).reset_index(drop=True)


def table_C(
    results: Iterable[TaskResult], static: Mapping[tuple[str, str], Mapping[str, Any]] | None = None
) -> pd.DataFrame:
    """Study C: per product × rota × recalibration the recalibration P&L (desk convention, total
    ± se and the number of refits), the static prediction ``desk_pnl_shadow × rota`` (the M7
    greek under the same policy; ``desk_pnl_usual × rota`` for the ``none`` rows, against the
    total hedged desk P&L), the ratio (± its delta-method se, :func:`ratio_stderr`), the 30%
    flag and z at every rota, the nonlinearity at +2 / +3 against the +1 row (± se), and
    ``refits_at_bound`` / ``contaminated`` (:func:`refits_at_bound`): a row whose refits pinned
    a correlation prices under perfectly correlated factors and is **excluded from the study's
    conclusion** — the simulated shadow there measures the fitter railing as much as the cost of
    re-marking (§8.2, owner's decision of 2026-09-16); ``refits_fallback`` / ``refits_capped``
    (:func:`refits_fallback`, :func:`refits_capped`): the refits of the row that held the base
    fit's correlation target because step 0 was degenerate on that date's state surface, and
    those whose correlation target was capped (``-1``: a run from before the record)."""
    static = static or {}
    res = [r for r in results if r.study == "C"]
    by_key = {(r.product, r.policy, r.rota): r for r in res}
    rows = []
    for r in res:
        assert r.rota is not None and r.policy is not None
        recal, recal_se = r.desk_recal_total()
        total, total_se = r.desk_mean()
        if r.policy != "none":
            sp = static.get((r.product, r.policy))
        else:
            # the usual rotation is the same under every policy: any record of the product
            sp = next((v for (p, _), v in static.items() if p == r.product), None)
        if r.policy == "none":
            simulated, sim_se = total, total_se
            pred_name = "desk_pnl_usual"
        else:
            simulated, sim_se = recal, recal_se
            pred_name = "desk_pnl_shadow"
        pred = float(sp[pred_name][0]) * r.rota if sp else float("nan")
        pred_se = float(sp[pred_name][1]) * r.rota if sp else float("nan")
        ratio, within, z = (
            first_order_agreement(simulated, sim_se, pred, pred_se)
            if sp
            else (float("nan"), False, float("nan"))
        )
        one = by_key.get((r.product, r.policy, 1.0))
        base_val, base_se = (
            one.desk_recal_total()
            if (one and r.policy != "none")
            else (one.desk_mean() if one else (float("nan"), float("nan")))
        )
        if r.rota != 1.0:
            nonlin = nonlinearity(simulated, r.rota, base_val)
            nonlin_se = ratio_stderr(simulated, sim_se, r.rota * base_val, r.rota * base_se)
        else:
            nonlin, nonlin_se = 0.0, 0.0
        rows.append(
            {
                "product": r.product,
                "rota": r.rota,
                "recalibration": r.policy,
                "status": r.status,
                "unit": r.unit,
                "recal_pnl_desk": recal,
                "recal_pnl_desk_se": recal_se,
                "n_refits": r.n_refits,
                "refits_at_bound": r.n_refits_at_bound,
                "contaminated": r.n_refits_at_bound > 0,
                "refits_fallback": r.n_refits_fallback,
                "refits_capped": r.n_refits_capped,
                "refit_dates": ";".join(f"{d['t']:.4g}" for d in r.recal_by_date),
                "recal_by_date_desk": ";".join(
                    f"{-d['mean']:+.4f}+/-{d['stderr']:.4f}" for d in r.recal_by_date
                ),
                "total_pnl_desk": total,
                "total_pnl_desk_se": total_se,
                "static_prediction": pred,
                "static_prediction_se": pred_se,
                "prediction_of": pred_name + " x rota",
                "ratio": ratio,
                "ratio_se": ratio_stderr(simulated, sim_se, pred, pred_se) if sp else float("nan"),
                "within_30pct": within,
                "z": z,
                "nonlinearity": nonlin,
                "nonlinearity_se": nonlin_se,
                "skew_6m_base_vp": r.world_meta.get("skew_90_110_6m_base_vp"),
                "skew_6m_rotated_vp": r.world_meta.get("skew_90_110_6m_rotated_vp"),
                "n_paths": r.n_paths_world,
                "wall_s": r.wall_seconds,
                "calibrations": r.calibrations,
                "reason": r.reason,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    porder = {p: i for i, p in enumerate(BOOK)}
    rorder = {p: i for i, p in enumerate(RECALIBRATIONS_C)}
    df["_p"] = df["product"].map(porder)
    df["_r"] = df["recalibration"].map(rorder)
    return df.sort_values(["_p", "rota", "_r"]).drop(columns=["_p", "_r"]).reset_index(drop=True)


def common_mv_delta(rows: Mapping[str, TaskResult]) -> tuple[float, float, float, int]:
    """The **common minimum-variance delta** of a study-D product: the precision-weighted mean of
    the regime rows' in-sample implied minimum-variance deltas ``λ* × mean delta``
    (:func:`delta_diagnostics`; the ``min_variance`` row excluded), as ``(value, se, spread,
    n)`` — ``spread`` the largest minus the smallest row value, ``n`` the rows used (a row
    without a finite value and a positive se is left out; NaN and 0 when none is left).

    The rows share their world paths, so their implied values are positively correlated and the
    independent-rows se ``(Σ w)^{-1/2}`` would understate the error; the se reported is the
    perfect-correlation bound ``Σ w σ / Σ w`` (``w = 1/σ²``), which is conservative.  The
    ``min_variance`` row itself is NOT the reference: it is not robust to the pricing path count
    (:data:`MV_BENCHMARK_VALIDITY_NSE`)."""
    nan = float("nan")
    vals, ses = [], []
    for regime, r in rows.items():
        if regime == MIN_VARIANCE_REGIME or "mv_delta_implied" not in r.delta_diag:
            continue
        v, se = (float(x) for x in r.delta_diag["mv_delta_implied"])
        if np.isfinite(v) and np.isfinite(se) and se > 0.0:
            vals.append(v)
            ses.append(se)
    if not vals:
        return nan, nan, nan, 0
    v_ = np.asarray(vals)
    s_ = np.asarray(ses)
    w = 1.0 / (s_ * s_)
    return (
        float(np.sum(w * v_) / np.sum(w)),
        float(np.sum(w * s_) / np.sum(w)),
        float(v_.max() - v_.min()),
        int(v_.size),
    )


def mv_benchmark_validity(
    mv: TaskResult, common: tuple[float, float]
) -> tuple[bool | None, float, str]:
    """``(valid, z, note)`` of the ``min_variance`` row against the common minimum-variance delta
    ``common = (value, se)``: ``z`` = (its mean delta − the common value) / the two se's in
    quadrature, ``valid`` = ``|z| <= MV_BENCHMARK_VALIDITY_NSE`` (``None`` when either is
    missing); ``note`` names a run whose factor gradients came from the raw value regression
    (no Black–Scholes proxy for the controlled gradients: :data:`MV_RAW_GRADIENT_NOTE`)."""
    note = (
        "raw regression gradients (no Black-Scholes proxy for the controlled value target)"
        if any(MV_RAW_GRADIENT_NOTE in n for n in mv.notes)
        else ""
    )
    if "mean_delta" not in mv.delta_diag or not np.isfinite(common[0]):
        return None, float("nan"), note
    md, md_se = (float(x) for x in mv.delta_diag["mean_delta"])
    se = float(np.hypot(md_se, common[1]))
    if not (np.isfinite(md) and se > 0.0):
        return None, float("nan"), note
    z = (md - common[0]) / se
    return bool(abs(z) <= MV_BENCHMARK_VALIDITY_NSE), float(z), note


def table_D(results: Iterable[TaskResult]) -> pd.DataFrame:
    """Study D (owner's decision of 2026-09-16; :data:`TABLE_HEADERS` ``"D"``): per product one
    row per regime of :data:`REGIMES_D`, the ``min_variance`` benchmark included as a fifth line
    — the P&L std and its rank (``std_rank``, 1 = smallest among the ok rows), the desk mean, the
    :func:`delta_diagnostics` columns (``mean_delta``, ``lambda_star``, ``std_at_lambda``,
    ``mv_delta_implied``; in-sample), the product's **common minimum-variance delta**
    ``mv_common`` (:func:`common_mv_delta` over the ok regime rows, with its se, its ``spread``
    and the rows used), ``distance_to_mv`` = the row's mean delta minus ``mv_common`` (se: the
    two in quadrature, ``distance_se_kind``) and the **distance ranking** ``distance_rank`` (1 =
    closest among the ok regime rows; 0 on the ``min_variance`` row, whose distance is shown but
    not ranked; −1 where no distance is available).  The ``min_variance`` row carries
    ``mv_valid`` / ``mv_z`` / ``mv_note`` (:func:`mv_benchmark_validity`; empty on the regime
    rows).  Before 2026-09-16 the distance was measured to the ``min_variance`` row's own mean
    delta — not a robust reference (hedged std 2.756 ± 0.053 and 2.965 ± 0.068 at 5·10³ pricing
    paths on two seeds, mean delta ~0.01 above the common value; on the autocall 0.318 against
    0.294–0.309).  Every Monte Carlo column carries its ``_se`` twin."""
    res = [r for r in results if r.study == "D"]
    nan = float("nan")
    rows = []
    for product in dict.fromkeys(r.product for r in res):
        rs = {r.regime: r for r in res if r.product == product and r.regime is not None}
        ok = {k: v for k, v in rs.items() if v.ok and np.isfinite(v.std[0])}
        by_std = sorted(ok, key=lambda k: ok[k].std[0])
        mv = ok.get(MIN_VARIANCE_REGIME)
        common, common_se, spread, n_common = common_mv_delta(ok)
        dist: dict[str, tuple[float, float, str]] = {}
        for k, v in ok.items():
            if "mean_delta" in v.delta_diag and n_common:
                md, md_se = (float(x) for x in v.delta_diag["mean_delta"])
                dist[k] = (md - common, float(np.hypot(md_se, common_se)), "quadrature")
            else:
                dist[k] = (nan, nan, "")
        valid, z, mv_note = (
            (None, nan, "") if mv is None else mv_benchmark_validity(mv, (common, common_se))
        )
        ranked = sorted(
            (k for k in ok if k != MIN_VARIANCE_REGIME and np.isfinite(dist[k][0])),
            key=lambda k: abs(dist[k][0]),
        )
        for regime in REGIMES_D:
            r = rs.get(regime)
            dd = r.delta_diag if r is not None else {}

            def pair(name: str, dd: Mapping[str, tuple[float, float]] = dd) -> tuple[float, float]:
                return dd.get(name, (nan, nan))

            d_val, d_se, d_kind = dist.get(regime, (nan, nan, ""))
            is_mv = regime == MIN_VARIANCE_REGIME
            if is_mv and regime in ok and np.isfinite(d_val):
                d_rank = 0
            elif regime in ranked:
                d_rank = ranked.index(regime) + 1
            else:
                d_rank = -1
            rows.append(
                {
                    "product": product,
                    "regime": regime,
                    "status": r.status if r else "missing",
                    "unit": r.unit if r else "",
                    "std": r.std[0] if r else nan,
                    "std_se": r.std[1] if r else nan,
                    "std_rank": by_std.index(regime) + 1 if regime in ok else -1,
                    "desk_mean": r.desk_mean()[0] if r else nan,
                    "desk_mean_se": r.mean[1] if r else nan,
                    "mean_delta": pair("mean_delta")[0],
                    "mean_delta_se": pair("mean_delta")[1],
                    "mv_common": common,
                    "mv_common_se": common_se,
                    "mv_common_spread": spread,
                    "mv_common_rows": n_common,
                    "distance_to_mv": d_val,
                    "distance_to_mv_se": d_se,
                    "distance_se_kind": d_kind,
                    "distance_rank": d_rank,
                    "mv_valid": (valid if is_mv else None),
                    "mv_z": (z if is_mv else nan),
                    "mv_note": (mv_note if is_mv else ""),
                    "lambda_star": pair("lambda_star")[0],
                    "lambda_star_se": pair("lambda_star")[1],
                    "std_at_lambda": pair("std_at_lambda")[0],
                    "std_at_lambda_se": pair("std_at_lambda")[1],
                    "mv_delta_implied": pair("mv_delta_implied")[0],
                    "mv_delta_implied_se": pair("mv_delta_implied")[1],
                    "n_paths": r.n_paths_world if r else 0,
                    "wall_s": r.wall_seconds if r else nan,
                    "reason": r.reason if r else "",
                }
            )
    return pd.DataFrame(rows)


def markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    if df.empty:
        return "(no rows)"
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, rec in df.iterrows():
        cells = []
        for c in cols:
            v = rec[c]
            if isinstance(v, float | np.floating):
                cells.append(f"{v:.{digits}f}" if np.isfinite(v) else "nan")
            elif v is None:
                cells.append("")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


TABLE_HEADERS: dict[str, str] = {
    "A": (
        "Study A — strategy ranking by P&L std per pricing model (= world), monthly rebalancing; "
        "`desk_mean` is the hedged P&L mean for the desk SHORT the product; cliquet in % of "
        "notional, FVA in vol points x notional; `std_se` from the fourth moment."
    ),
    "B": (
        "Study B — model mismatch: pricing = the SPX marking fit, world per row; `leakage_desk` "
        "is the mean hedged P&L per path for the desk SHORT the note (± se), `static_spread` the "
        "marked price minus the world price at t = 0 (both Monte Carlo, stderrs in quadrature), "
        "`dynamic_leakage` = leakage - static spread, `leakage_vs_same` = leakage minus the "
        "product's leakage under world (i) `same` (the engine's own baseline: the pricing error "
        "of V0 and the regression drift of the hedge legs; the model reserve of a world is this "
        "difference); `leakage_desk_incl_v0` repeats the leakage with the V0 pricing error folded "
        "into its se; quantiles are of the desk P&L (± the quantile se of the distribution "
        "table); every `<x>_se` is the standard error of `<x>`; units per row (% of notional; "
        "KO var in vol points of vega notional = variance P&L / 2 K_vol)."
    ),
    "C": (
        "Study C — shadow rotation as P&L: pricing = the marking fit, world = the skew-shock "
        "world (+rota over 5 business days from T/2, then held). "
        + ROTATION_CONVENTION
        + " `recal_pnl_desk` = -Σ_refits [V(new) - V(old)] (the hedger prices the product long; "
        "the desk is short), per refit date in `recal_by_date_desk`; `static_prediction` = the M7 "
        "greek's desk_pnl_shadow x rota under the same policy (desk_pnl_usual x rota against the "
        "total hedged desk P&L on the `none` rows, a reference only); `ratio` = simulated / "
        "static (`ratio_se`: delta method, independent errors), `within_30pct` the first-order "
        "test at +1 rota, `nonlinearity` = P&L(rota)/(rota x P&L(+1)) - 1 (`nonlinearity_se`: "
        "delta method; the rota runs share the world seed, so it is conservative). "
        "`refits_at_bound` counts the refits of the row that landed with a fitted correlation "
        "at its bound and `contaminated` marks the row: such rows are reported but EXCLUDED "
        "from the study's conclusion (owner's decision, 2026-09-16; measured cause on the "
        "2026-09-15 runs: the rule's strip curvature unconverged at 2e4 paths fired step 0's "
        "radicand guard, Corr_SABR clipped to -1 and the collapsed set was step 3's exact "
        "minimiser). The rule's strips run at "
        f"{STUDY_C_STRIP_PATHS} paths; `refits_fallback` counts the refits that held the base "
        "fit's correlation target because step 0 was degenerate on that date's state surface, "
        f"`refits_capped` those whose |Corr_BE| target was capped at {REFIT_CORRELATION_CAP:g} "
        "(-1: not recorded, a run from before 2026-09-16)."
    ),
    "D": (
        "Study D — delta-regime P&L read against the minimum-variance delta (reinstated, owner's "
        "decision of 2026-09-16): world = pricing (2F), delta only under each §7.2 regime and "
        "the benchmark row `min_variance` (the pricing model's minimum-variance spot-only delta: "
        "the model delta + Σ_i dV/dX_i · d<X_i,S>/d<S,S>). Headline: "
        + STUDY_D_HEADLINE
        + ". Measured reading: "
        + STUDY_D_READING
        + ". The reference is the COMMON minimum-variance delta `mv_common` the four regime rows "
        "imply (the precision-weighted mean of their `mv_delta_implied`; se: the "
        "perfect-correlation bound, conservative, the rows sharing their world paths; "
        "`mv_common_spread` = the largest minus the smallest row value). The ranking is "
        "`distance_rank` (1 = the regime whose `mean_delta` sits closest to `mv_common`; 0 = the "
        "min_variance row, shown but not ranked; -1 = no distance), with `distance_to_mv` = mean "
        "delta minus `mv_common` (se in quadrature); the std ranking stays visible as `std_rank` "
        "(1 = smallest std). The `min_variance` row (the model delta + Σ_i dV/dX_i · "
        "d<X_i,S>/d<S,S> on the pricing paths) is a fifth line with a validity flag: `mv_valid` "
        "when its mean delta lies within "
        f"{MV_BENCHMARK_VALIDITY_NSE:g} se of `mv_common` (`mv_z`); `mv_note` marks a run whose "
        "dV/dX came from the raw value regression (no Black-Scholes proxy, e.g. the autocall). "
        "`mean_delta` is the relative delta S0·Δ·scale/100 averaged over dates and paths (the "
        "per-unit-spot delta for the vanilla on one share). IN-SAMPLE benchmarks, fitted on the "
        "paths they are scored on: `lambda_star` = -Cov(product leg, hedge leg)/Var(hedge leg), "
        "`std_at_lambda` the P&L std with the hedge leg scaled by it, `mv_delta_implied` = "
        "lambda_star x mean_delta (path-pair bootstrap se's). `desk_mean` is the hedged mean for "
        "the desk SHORT the product; vanilla in % of spot, autocall in % of notional."
    ),
}


def build_tables(
    results: Sequence[TaskResult],
    skipped_b: Sequence[Mapping[str, Any]] = (),
    static: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> dict[str, pd.DataFrame]:
    return {
        "A": table_A(results),
        "B": table_B(results, skipped_b),
        "C": table_C(results, static),
        "D": table_D(results),
    }


def write_tables(
    out: Path,
    tables: Mapping[str, pd.DataFrame],
    *,
    header_lines: Sequence[str] = (),
    static: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> Path:
    """``<out>/m8b_table_<study>.csv`` per study and ``<out>/m8b.md`` (every table with its
    header and the study C static predictions)."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    md = ["# M8b hedging studies", ""]
    md += [f"- {line}" for line in header_lines]
    md.append("")
    for st in STUDIES:
        df = tables.get(st, pd.DataFrame())
        df.to_csv(out / f"m8b_table_{st}.csv", index=False)
        md += [
            f"## Study {st}",
            "",
            TABLE_HEADERS[st],
            "",
            f"M8B_TABLE_{st}",
            "",
            markdown_table(df),
            "",
        ]
        if st == "C" and static:
            rows = []
            for (prod, pol), doc in sorted(static.items()):
                rows.append(
                    {
                        "product": prod,
                        "policy": pol,
                        "unit": doc.get("unit"),
                        "p1_level": _pm_doc(doc, "p1_level"),
                        "fee": _pm_doc(doc, "fee"),
                        "desk_pnl_usual": _pm_doc(doc, "desk_pnl_usual"),
                        "desk_pnl_shadow": _pm_doc(doc, "desk_pnl_shadow"),
                        "n_paths": doc.get("n_paths"),
                        "n_particles": doc.get("n_particles"),
                        "recalibrated": doc.get("recalibrated"),
                        "wall_s": doc.get("wall_seconds"),
                        "base_fit_equals_marking_fit": doc.get("base_fit_equals_marking_fit"),
                    }
                )
            md += [
                "Static predictions (the M7 rotation greek per product and policy, per +1 rota, "
                "desk convention):",
                "",
                markdown_table(pd.DataFrame(rows)),
                "",
            ]
    (out / "m8b.md").write_text("\n".join(md), encoding="utf-8")
    return out / "m8b.md"


__all__ = [
    "BOOK",
    "DEFAULT_N_PARTICLES",
    "DEFAULT_N_PATHS",
    "DEFAULT_REFIT_PARTICLES",
    "DELTA_DIAG_BOOTSTRAP",
    "DELTA_PATHS_SUFFIX",
    "FIRST_ORDER_TOLERANCE",
    "MV_BENCHMARK_VALIDITY_NSE",
    "MV_RAW_GRADIENT_NOTE",
    "PRICING_A",
    "Q_SWEEP",
    "RECALIBRATIONS_C",
    "REGIMES_D",
    "ROTAS",
    "SIM_DT_MAX",
    "STUDIES",
    "STUDY_A_FREQUENCY",
    "STUDY_C_CURVATURE_H",
    "STUDY_D_HEADLINE",
    "STUDY_D_READING",
    "WORLDS_B",
    "Gate",
    "Projection",
    "RecordingHedger",
    "RefitParticlesBuilder",
    "StudyConfig",
    "StudyEnvironment",
    "Task",
    "TaskList",
    "TaskResult",
    "build_strategy",
    "build_tables",
    "calibration_seconds",
    "common_mv_delta",
    "delta_diagnostics",
    "delta_path_means",
    "discriminator_gate",
    "enumerate_tasks",
    "first_order_agreement",
    "frequency_for",
    "load_results",
    "load_static_predictions",
    "make_hedger",
    "markdown_table",
    "mv_benchmark_validity",
    "n_pricing_models",
    "nonlinearity",
    "parse_shard",
    "product_maturity",
    "project",
    "recalibration_by_date",
    "refit_state",
    "required_states",
    "result_paths",
    "run_task",
    "save_result",
    "shard",
    "slug",
    "spx_base_spec",
    "static_prediction",
    "static_prediction_path",
    "summarize",
    "table_A",
    "table_B",
    "table_C",
    "table_D",
    "to_desk_pnl",
    "write_tables",
]
