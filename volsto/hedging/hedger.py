"""The hedger (SPEC §8, M8 Part 1): world paths, pricing-model Greeks by regression, strategy
rebalancing, P&L accounting in time-0 money, early termination, transaction costs and the
recalibration rule.

**Loop.**  ``Hedger(pricing, world, schedule, costs, recalibration).run(product, strategy)``:

1. the rebalancing dates are the schedule's (daily 1/252, weekly 1/52 or monthly 1/12 by
   default, or explicit) plus every fixing / observation date of the product inside ``(0, T)``
   (rebalance after the fixing) and every instrument roll date; ``t = 0`` included, ``T`` excluded
   (settlement);
2. the pricing model is simulated once with its CRN spot bumps and the strategy's target bumps
   (:class:`~volsto.hedging.pricing.ConditionalPricer`); the world model is simulated once on the
   same grid with its own seed (``world_seed``, default pricing seed + 1);
3. at each date: conditional values and Greeks of the product and of every active instrument at
   the world states → the strategy's quantities (zero after the product has terminated — the
   hedges are unwound on the termination date — and outside an instrument's activity window) →
   costs on the traded quantities → the P&L of the holding period ``[t_k, t_{k+1}]``: product
   revaluation ``V_{k+1} − V_k``, hedge legs ``q_k (I_{k+1} − I_k)``, each in time-0 money; at the
   last date the values are the realised discounted payoffs;
4. with a :class:`RecalibrationRule` the world's conditional skew at each date is measured, the P1
   set refit when it moved by more than ``skew_move_threshold`` under the rule's ``policy``
   (:data:`~volsto.risk.shadow_rotation.RECALIBRATION_POLICIES`: ``"sabr_linked"`` refits every
   target on the state surface; ``"sticky_breakeven"`` holds the break-even targets at the base
   fit's values and lets only the skew constraint follow the state surface;
   ``"sticky_breakeven_skew"`` holds the skew reference too) and the pricing model rebuilt through
   the leverage cache (hit rate reported); the **recalibration P&L** is ``V_k(new set) − V_k(old
   set)`` at the same world state, isolated as its own leg.

Total P&L per path ``= product leg + Σ hedge legs − costs``; the zero-cost total is reported next
to it (costs are additive).  The product leg sums to ``payoff − V₀`` exactly on every path
(telescoping), so the hedged P&L's mean is the product's pricing error plus the hedging noise.

**Compute budget** (:meth:`Hedger.projected_wall_clock`, printed before a run): one pricing-model
simulation per bumped model (``1 + 2`` spot bumps ``+`` one or two per target bump) on
``n_paths`` paths over the grid, one world simulation, and ``n_dates × n_objects`` regressions
of ``O(n_paths × n_basis²)``; the estimate scales a one-chunk timing.  Long runs are shard-able by
path blocks (the M9 ``--shard`` form): ``world_paths`` and ``world_seed`` select the block.
**Memory:** the bumped path sets are kept for the hybrid targets (``budget["pricing_paths_gb"]``);
``stream_bumps=True`` writes them to a scratch directory (``scratch_dir``, the ``VOLSTO_SCRATCH``
variable or the system temp) and memory-maps them back per date instead
(``budget["streamed_paths_gb"]``; :class:`~volsto.hedging.pricing.ConditionalPricer`), passed to
every pricer the run builds, the recalibration rebuilds included; the scratch directories are
removed when the run ends.

Checked by ``tests/test_hedging.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.config import SimConfig, SurfacePerturbation
from volsto.engine.grid import TimeGrid
from volsto.engine.paths import PathSet
from volsto.engine.rng import GaussianDraws
from volsto.hedging.controls import MIN_SHADOW_VOL
from volsto.hedging.instruments import Spot, deduplicate_names, expand_rolls
from volsto.hedging.pricing import Bump, ConditionalPricer, union_grid
from volsto.hedging.strategies import (
    DELTA_REGIMES,
    GreekTargetStrategy,
    HedgeSolution,
    PresetContext,
    Strategy,
    default_strategy,
)
from volsto.market.surface import ArbitrageError
from volsto.models.base import Model
from volsto.products.base import Portfolio, Product
from volsto.risk.engine import BSBuilder, LSVBuilder, LVBuilder, RiskState, surface_of
from volsto.risk.greeks import _spot_state
from volsto.risk.ladders import PILLARS as RISK_PILLARS
from volsto.risk.ladders import bucket_epsilon
from volsto.risk.shadow_rotation import RECALIBRATION_POLICIES, held_targets

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
_TOL = 1e-9
FREQUENCIES = {"daily": 1.0 / 252.0, "weekly": 1.0 / 52.0, "monthly": 1.0 / 12.0}
#: default tent skew / curvature size (vol points of 90/110) and vol bump (vol points)
TENT_SIZE = 0.01
VOL_BUMP = 0.01
SPOT_BUMP = 0.01
#: default ``RecalibrationRule.skew_move_threshold`` (vol per unit log-moneyness): the world's
#: conditional skew is a three-strike regression under the world model whose own noise measured
#: 0.002-0.008 (SPEC §8.1, the Black-Scholes world without a skew move); 0.01 sits above it so a
#: world without a skew move never triggers a spurious refit (owner decision (c), M8 acceptance)
SKEW_MOVE_THRESHOLD = 0.01
#: the state surface logs when fewer than this fraction of the paths invert to a finite vol
STATE_SURFACE_MIN_INVERTED = 0.99
#: halvings of a tent / forward-variance bump that fails the surface's arbitrage checks before
#: giving up — the M5 ``RiskEngine.max_halvings`` default (a +1 vp skew tent at the 2y pillar
#: breaks the calendar condition on the placeholder surface: halved once it passes)
MAX_HALVINGS = 4


# --------------------------------------------------------------------------------------------
# schedule and costs
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Schedule:
    """Rebalancing dates: ``frequency`` in :data:`FREQUENCIES` or explicit ``dates``, plus every
    product fixing.  **Rebalance after the fixing:** at a rebalancing date that coincides with a
    product fixing the hedger sees the post-fixing state (the fixing at ``t`` is part of the
    state, the hybrid-CRN Greeks move the future only), i.e. it rebalances just after the fixing;
    ``after_fixing > 0`` places such a date that much later instead (one business day is 1/252 —
    the setting the fitted-gradient delta estimator needed, its derivative being unidentified
    when every path has the same in-period return).  ``t = 0`` stays; nothing is placed at or
    beyond the maturity."""

    frequency: str = "daily"
    dates: tuple[float, ...] | None = None
    after_fixing: float = 0.0

    def __post_init__(self) -> None:
        if self.dates is None and self.frequency not in FREQUENCIES:
            raise ValueError(f"frequency must be one of {tuple(FREQUENCIES)}")
        if self.after_fixing < 0:
            raise ValueError("after_fixing must be non-negative")

    def build(self, product: Product, extra: Sequence[float] = ()) -> FloatArray:
        T = float(product.maturity)
        if self.dates is not None:
            base = np.asarray(self.dates, dtype=np.float64)
        else:
            dt = FREQUENCIES[self.frequency]
            n = max(1, round(T / dt))
            base = np.linspace(0.0, T, n + 1)
        fix = np.asarray(product.fixing_times, dtype=np.float64)
        fix = fix[(fix > _TOL) & (fix < T - _TOL)]
        cand = np.unique(np.concatenate([base, fix, np.asarray(extra, dtype=np.float64), [0.0]]))
        out = []
        for d in cand:
            if d <= _TOL:
                out.append(0.0)
                continue
            on_fixing = fix.size > 0 and bool(np.min(np.abs(fix - d)) <= _TOL)
            out.append(d + self.after_fixing if on_fixing else d)
        all_ = np.unique(np.asarray(out, dtype=np.float64))
        all_ = all_[(all_ >= 0.0) & (all_ < T - _TOL)]
        return np.asarray(all_, dtype=np.float64)


@dataclass(frozen=True)
class Costs:
    """Default half-spreads: ``spot_bps`` on the spot's traded value, ``vol_points`` on vanillas
    and variance-type swaps (instruments carrying their own ``cost`` keep it)."""

    spot_bps: float = 0.0
    vol_points: float = 0.0


# --------------------------------------------------------------------------------------------
# pricing context: the desk's model and the bumps it can build
# --------------------------------------------------------------------------------------------


@dataclass
class PricingContext:
    """The pricing model plus, when it comes from a :class:`RiskState`, the builder that
    produces surface-bumped and recalibrated variants (through the leverage cache), and the
    surface for reference vols and strikes."""

    model: Model
    state: RiskState | None = None
    builder: Any = None
    surface: Any = None
    label: str = "pricing"
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def from_model(model: Model, surface: Any = None, label: str = "pricing") -> PricingContext:
        return PricingContext(model, None, None, surface, label)

    @staticmethod
    def from_state(
        state: RiskState,
        cache: Any = None,
        kind: str = "lsv",
        *,
        allow_calibrate: bool = True,
        label: str = "pricing",
    ) -> PricingContext:
        """``kind``: ``"lsv"`` (needs ``cache``), ``"lv"`` or ``"bs"``."""
        if kind == "lsv":
            if cache is None:
                raise ValueError("an LSV pricing context needs the leverage cache")
            builder: Any = LSVBuilder(cache, state, allow_calibrate=allow_calibrate)
        elif kind == "lv":
            builder = LVBuilder(state)
        elif kind == "bs":
            builder = BSBuilder(state)
        else:
            raise ValueError("kind must be 'lsv', 'lv' or 'bs'")
        model = builder.build(state, "recalibrate")
        return PricingContext(model, state, builder, surface_of(state), label)

    @property
    def forward_curve(self) -> Any:
        return self.model.forward_curve

    def reference_vol(self) -> Callable[[float, float], float] | None:
        if self.surface is None:
            return None
        surf = self.surface

        def fn(strike: float, expiry: float) -> float:
            T = min(max(float(expiry), 1.0 / 365.0), float(surf.max_maturity))
            return float(np.asarray(surf.implied_vol(float(strike), T)).ravel()[0])

        return fn

    def _bumped(self, pert: SurfacePerturbation, mode: str = "recalibrate") -> tuple[Model, Any]:
        """The model rebuilt on the perturbed state and that state's implied surface (the
        latter travels with the :class:`Bump` for the §7.11 control's proxy vols)."""
        assert self.state is not None and self.builder is not None
        state = self.state.with_perturbation(pert)
        model: Model = self.builder.build(state, mode)
        return model, surface_of(state)

    def _bumped_halving(
        self, make: Callable[[float], SurfacePerturbation], size: float, label: str
    ) -> tuple[Model, Any, float]:
        """:meth:`_bumped` on ``make(size)``, the size halved until the perturbed surface passes
        the no-arbitrage checks (M5's ``RiskEngine.perturbed_state``, at most
        :data:`MAX_HALVINGS` times; noted when halved) — the achieved size is returned so the
        caller scales the bump's ``unit`` and the target stays per the nominal size."""
        s = float(size)
        last: Exception | None = None
        for _ in range(MAX_HALVINGS + 1):
            try:
                model, surface = self._bumped(make(s))
            except ArbitrageError as exc:
                last = exc
                s *= 0.5
                continue
            if s != size:
                note = f"{label}: bump halved to {s:.3g} of {size:.3g} (arbitrage check)"
                if note not in self.notes:
                    self.notes.append(note)
            return model, surface, s
        raise ArbitrageError(
            f"bump {label} fails the arbitrage checks after {MAX_HALVINGS} halvings: {last}"
        )

    def available_targets(self) -> list[str]:
        base = ["delta", "gamma"]
        nf = int(getattr(self.model, "n_factors", 0))
        base += [f"dX{i}" for i in range(1, nf + 1)]
        has_vol_bump = hasattr(self.model, "vol")
        if self.state is not None or has_vol_bump:
            base += ["vega", "volga", "vanna"]
        if self.state is not None:
            base += ["fwd_var:<lo>-<hi>", "skew_T:<T>", "curvature_T:<T>", "param:<name>"]
        return base

    def bump(self, name: str, *, delta_regime: str = "model") -> Bump | None:
        """The :class:`Bump` of a target name (``None`` for the regression-native ones)."""
        if name.startswith("dX"):
            nf = int(getattr(self.model, "n_factors", 0))
            i = int(name[2:]) if name[2:].isdigit() else 0
            if i < 1 or i > nf:
                raise ValueError(f"{name}: the pricing model has {nf} factors")
            return None
        if name in ("delta", "gamma", "vanna"):
            if name == "delta" and delta_regime != "model":
                return self.regime_delta_bump(delta_regime)
            return None
        if name in ("vega", "volga"):
            # what the §7.11 control needs to shadow the bump: the perturbed surfaces, or the
            # flat shifts of a bare Black-Scholes model (its dn vol floored like the model's)
            shadow: dict[str, Any] = {}
            if self.state is not None:
                up, s_up = self._bumped(SurfacePerturbation("parallel", {"size": VOL_BUMP}))
                dn, s_dn = self._bumped(SurfacePerturbation("parallel", {"size": -VOL_BUMP}))
                shadow = {"up_surface": s_up, "dn_surface": s_dn}
            elif hasattr(self.model, "vol"):
                v = float(self.model.vol)
                v_dn = max(v - VOL_BUMP, MIN_SHADOW_VOL)
                up = self.model.bump(vol=v + VOL_BUMP)
                dn = self.model.bump(vol=v_dn)
                shadow = {"vol_shift_up": VOL_BUMP, "vol_shift_dn": v_dn - v}
            else:
                raise ValueError(
                    f"target {name!r} needs a pricing state or a model with a vol; available: "
                    f"{self.available_targets()}"
                )
            if name == "vega":
                return Bump(
                    "vega", up, dn, VOL_BUMP, "parallel +1 vp, central", kind="model", **shadow
                )
            return Bump(
                "volga",
                up,
                dn,
                VOL_BUMP,
                "second difference in the parallel vol",
                kind="second",
                **shadow,
            )
        if self.state is None:
            raise ValueError(
                f"target {name!r} needs a pricing state (surface bumps); available: "
                f"{self.available_targets()}"
            )
        if name.startswith("fwd_var:"):
            lo, hi = (float(x) for x in name.split(":")[1].split("-"))
            eps = bucket_epsilon(self.surface, lo, hi, 0.01)
            up, s_up, achieved = self._bumped_halving(
                lambda e: SurfacePerturbation("total_variance", {"eps": e, "t_lo": lo, "t_hi": hi}),
                eps,
                name,
            )
            return Bump(
                name,
                up,
                None,
                achieved / eps,
                f"forward-variance bucket [{lo:g}, {hi:g}] +1 vp of its VS vol",
                kind="model",
                up_surface=s_up,
            )
        if name.startswith("skew_T:") or name.startswith("curvature_T:"):
            T = float(name.split(":")[1])
            pillars = tuple(sorted(set(RISK_PILLARS) | {T}))
            i = pillars.index(T)
            if name.startswith("skew_T:"):
                from volsto.risk.ladders import skew_slope

                def make(size: float) -> SurfacePerturbation:
                    return SurfacePerturbation(
                        "skew_tent", {"pillars": pillars, "index": i, "slope": skew_slope(size)}
                    )

            else:
                from volsto.risk.ladders import curvature_coefficient

                def make(size: float) -> SurfacePerturbation:
                    return SurfacePerturbation(
                        "curvature_tent",
                        {"pillars": pillars, "index": i, "curv": curvature_coefficient(size)},
                    )

            # the tent size is halved until the surface passes the arbitrage checks (as M5);
            # the unit keeps the target per +1 vp of 90/110 whatever the achieved size
            up, s_up, achieved = self._bumped_halving(make, TENT_SIZE, name)
            return Bump(
                name,
                up,
                None,
                achieved / TENT_SIZE,
                f"{name}: +1 vp of 90/110 at the pillar (bump {achieved / TENT_SIZE:g} vp)",
                kind="model",
                up_surface=s_up,
            )
        if name.startswith("param:"):
            pname = name.split(":")[1]
            from volsto.risk.engine import default_params_bump

            assert self.state is not None
            h = default_params_bump(pname, self.state.spec.model)
            cur = float(getattr(self.state.spec.model, pname))
            up = self.builder.build(_with_params(self.state, {pname: cur + h}), "recalibrate")
            dn = self.builder.build(_with_params(self.state, {pname: cur - h}), "recalibrate")
            return Bump(name, up, dn, h, f"model parameter {pname} ±{h:g}", kind="model")
        raise ValueError(f"unknown target {name!r}; available: {self.available_targets()}")

    def regime_delta_bump(self, regime: str) -> Bump:
        if regime not in DELTA_REGIMES:
            raise ValueError(f"delta_regime must be one of {DELTA_REGIMES}")
        if self.state is None:
            raise ValueError("a non-model delta regime needs a pricing state")
        h = SPOT_BUMP
        up_state, mode_up = _spot_state(self.state, regime, h)
        dn_state, mode_dn = _spot_state(self.state, regime, -h)
        up = self.builder.build(up_state, mode_up)
        dn = self.builder.build(dn_state, mode_dn)
        return Bump("delta", up, dn, h, f"delta under the {regime} regime", kind="spot")


# --------------------------------------------------------------------------------------------
# recalibration rule
# --------------------------------------------------------------------------------------------


@dataclass
class RecalibrationRule:
    """``"on_skew_move"`` (module docstring): at each rebalancing date the world's conditional
    ATM skew at the ``pillars`` (tenors from the date) is measured by regressing a three-strike
    vanilla strip under the **world** model on the world state; when it moved by more than
    ``skew_move_threshold`` (vol per unit log-moneyness, any pillar; default
    :data:`SKEW_MOVE_THRESHOLD`) since the last refit the P1 set is refit on the state surface
    (:func:`~volsto.calibration.fit_2f.fit_2f` on
    :func:`~volsto.calibration.targets.marking_targets` of a quadratic-smile state surface) under
    ``policy`` (:data:`~volsto.risk.shadow_rotation.RECALIBRATION_POLICIES`, the semantics of
    :func:`~volsto.risk.shadow_rotation.refit_on_rotated`): ``"sabr_linked"`` refits every target
    on the state surface; ``"sticky_breakeven"`` holds the break-even targets (``spot_vol_covar``,
    ``vol_var``, ``vovol``, ``vov_be_raw``, ``correl_target``) at the **base** fit's values and
    lets only the skew constraint follow the state surface; ``"sticky_breakeven_skew"`` holds the
    two-point skew reference too (:func:`~volsto.risk.shadow_rotation.held_targets`).  The base
    fit is ``base_fit`` (a :class:`~volsto.calibration.fit_2f.FitResult`); when ``None`` under a
    sticky policy the hedger computes the marking fit of the pricing surface once, at the first
    rebalancing date, with the rule's config and ``ssr_target``
    (:func:`~volsto.calibration.fit_2f.fit_2f_marking`), stores it here and logs it.  The pricing
    model is rebuilt with the new parameters through the cache (``builder``); the reference skew
    is the world's own at ``t = 0``, so a world without a skew move never recalibrates.  ``refit``
    may replace the fit call (``(state_surface, base_params) -> params``; the policy is then the
    callable's business).  ``log_rows`` records every refit with its policy.
    """

    pillars: tuple[float, ...] = (0.25, 1.0)
    skew_move_threshold: float = SKEW_MOVE_THRESHOLD
    h: float = 0.05
    fit_config: Any = None
    ssr_target: float = 1.0
    refit: Callable[..., Any] | None = None
    max_refits: int = 100
    policy: str = "sabr_linked"
    base_fit: Any | None = None
    log_rows: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.policy not in RECALIBRATION_POLICIES:
            raise ValueError(f"policy must be one of {RECALIBRATION_POLICIES}")
        if not (self.skew_move_threshold > 0.0):
            raise ValueError("skew_move_threshold must be positive")

    @property
    def sticky(self) -> bool:
        """Whether the policy holds targets of a base fit."""
        return self.policy != "sabr_linked"

    def config(self) -> Any:
        """The fit config: ``fit_config`` or the default on the rule's pillars."""
        from volsto.calibration.fit_2f import BreakEvenFitConfig

        return self.fit_config or BreakEvenFitConfig(
            pillars=tuple(self.pillars),
            mat_min=0.0,
            skew_pillars=(min(self.pillars), max(self.pillars)),
        )


class _StateSurface:
    """Quadratic smiles ``atf + skew k + ½ curv k²`` per pillar, interpolated in ``T``, with the
    forward curve of the world: the surface a SABR reduction can read at a world state."""

    def __init__(
        self, pillars: FloatArray, atf: FloatArray, skew: FloatArray, curv: FloatArray, fc: Any
    ) -> None:
        self.pillars, self.atf, self.skew, self.curv = pillars, atf, skew, curv
        self.forward_curve = fc
        self.max_maturity = float(pillars[-1])

    def _interp(self, arr: FloatArray, T: Any) -> FloatArray:
        return np.asarray(
            np.interp(np.asarray(T, dtype=np.float64), self.pillars, arr), dtype=np.float64
        )

    def implied_vol_k(self, k: Any, T: Any) -> FloatArray:
        k_, T_ = np.broadcast_arrays(
            np.asarray(k, dtype=np.float64), np.asarray(T, dtype=np.float64)
        )
        v = (
            self._interp(self.atf, T_)
            + self._interp(self.skew, T_) * k_
            + 0.5 * self._interp(self.curv, T_) * k_**2
        )
        return np.asarray(np.maximum(v, 1e-4), dtype=np.float64)

    def atm_vol(self, T: Any) -> FloatArray:
        return self._interp(self.atf, T)

    def atm_skew(self, T: Any) -> FloatArray:
        return self._interp(self.skew, T)


# --------------------------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------------------------


@dataclass
class HedgeResult:
    """Per-path and per-date P&L of one run (module docstring).  Arrays: ``pnl_total``
    (with costs), ``pnl_zero_cost``, ``pnl_product``, ``pnl_hedges`` ``(n_paths, n_instruments)``,
    ``costs``, ``pnl_recalibration``; ``termination`` (years, NaN when none); per date:
    ``dates``, ``by_date`` (mean P&L per leg), ``quantities`` (mean / std per instrument),
    ``residual`` (mean |residual exposure| per target), ``recalibrations``."""

    product: str
    strategy: str
    dates: FloatArray
    instruments: tuple[str, ...]
    targets: tuple[str, ...]
    pnl_product: FloatArray
    pnl_hedges: FloatArray
    costs: FloatArray
    pnl_recalibration: FloatArray
    termination: FloatArray
    value_0: float
    value_0_stderr: float
    by_date: pd.DataFrame
    quantities: pd.DataFrame
    residual: pd.DataFrame
    world_paths: PathSet
    pricing_notes: tuple[str, ...]
    recalibrations: pd.DataFrame
    budget: dict[str, float]
    settings: dict[str, Any] = field(default_factory=dict)
    greeks_by_date: dict[str, FloatArray] = field(default_factory=dict)
    moves_by_date: dict[str, FloatArray] = field(default_factory=dict)

    @property
    def n_paths(self) -> int:
        return int(self.pnl_product.size)

    @property
    def pnl_zero_cost(self) -> FloatArray:
        return np.asarray(self.pnl_product + self.pnl_hedges.sum(axis=1) + self.pnl_recalibration)

    @property
    def pnl_total(self) -> FloatArray:
        return np.asarray(self.pnl_zero_cost - self.costs)

    def mean_se(self, x: FloatArray) -> tuple[float, float]:
        return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(x.size))

    def summary(self) -> str:
        m, se = self.mean_se(self.pnl_total)
        m0, se0 = self.mean_se(self.pnl_zero_cost)
        term = np.isfinite(self.termination)
        return (
            f"hedge {self.product} with {self.strategy}: {self.n_paths} world paths, "
            f"{self.dates.size} rebalancing dates; V0 {self.value_0:.6f} +/- "
            f"{self.value_0_stderr:.6f}; "
            f"P&L mean {m:+.6f} +/- {se:.6f} (std {np.std(self.pnl_total, ddof=1):.6f}); zero-cost "
            f"mean {m0:+.6f} +/- {se0:.6f} (std {np.std(self.pnl_zero_cost, ddof=1):.6f}); costs "
            f"{np.mean(self.costs):.6f}; recalibration P&L {np.mean(self.pnl_recalibration):+.6f}; "
            f"early terminations {int(term.sum())}; {self.budget}"
        )


# --------------------------------------------------------------------------------------------
# the hedger
# --------------------------------------------------------------------------------------------


@dataclass
class Hedger:
    """Module docstring.  ``pricing`` is a :class:`PricingContext` (or a bare model),
    ``world`` a model; ``sim`` the pricing configuration (paths, schedule, scheme), ``world_paths``
    the number of world paths (default ``sim.n_paths``); ``stream_bumps`` / ``scratch_dir`` and
    ``control_variate`` (the §7.11 control on the difference for the surface-driven bump targets
    of the objects with a Black–Scholes proxy, :mod:`volsto.hedging.controls`; its median
    variance reduction over dates and controlled objects is ``budget["cv_reduction_median"]``)
    are passed to every :class:`~volsto.hedging.pricing.ConditionalPricer` the run builds
    (module docstring, *Memory*)."""

    pricing: PricingContext | Model
    world: Model
    schedule: Schedule = field(default_factory=Schedule)
    costs: Costs = field(default_factory=Costs)
    recalibration: RecalibrationRule | None = None
    sim: SimConfig = field(default_factory=lambda: SimConfig(n_paths=20_000, chunk_size=20_000))
    world_paths: int | None = None
    world_seed: int | None = None
    degree: int = 2
    verbose: bool = True
    stream_bumps: bool = False
    scratch_dir: str | Path | None = None
    _twin: ConditionalPricer | None = field(default=None, init=False, repr=False)
    control_variate: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.pricing, PricingContext):
            self.pricing = PricingContext.from_model(self.pricing)
        if self.sim.n_paths % 2 and self.sim.antithetic:
            raise ValueError("n_paths must be even with antithetic draws")

    # -- helpers -------------------------------------------------------------------------------

    @property
    def context(self) -> PricingContext:
        assert isinstance(self.pricing, PricingContext)
        return self.pricing

    def preset_context(self, product: Product) -> PresetContext:
        fc = self.context.forward_curve
        return PresetContext(
            forward_curve=fc,
            discount=product.discount,
            spot=float(fc.spot),
            reference_vol=self.context.reference_vol(),
            cost_spot_bps=self.costs.spot_bps,
            cost_vol_points=self.costs.vol_points,
            surface_targets=self.context.state is not None,
            horizon=float(product.maturity) + 1.0,
        )

    def _world_sim(self) -> SimConfig:
        n = self.world_paths or self.sim.n_paths
        return SimConfig(
            n_paths=n,
            dt_max=self.sim.dt_max,
            chunk_size=min(self.sim.chunk_size, n),
            antithetic=self.sim.antithetic,
            seed=self.sim.seed + 1 if self.world_seed is None else self.world_seed,
            local_var_time_average=self.sim.local_var_time_average,
            predictor_corrector=self.sim.predictor_corrector,
            pc_eta=self.sim.pc_eta,
            weak_order2=self.sim.weak_order2,
            local_var_time_eval=self.sim.local_var_time_eval,
            sv_order2=self.sim.sv_order2,
            record_all_steps=self.sim.record_all_steps,
            chunk_memory_mb=self.sim.chunk_memory_mb,
        )

    def _simulate_world(self, grid: TimeGrid) -> PathSet:
        wsim = self._world_sim()
        draws = GaussianDraws(
            wsim.seed, wsim.n_paths, grid.n_steps, self.world.n_brownians, wsim.antithetic
        )
        parts = [
            self.world.simulate_chunk(grid, draws, p0, p1, wsim.scheme)
            for p0, p1 in wsim.chunk_ranges(grid.n_records, self.world.n_factors)
        ]
        return PathSet.concat(parts)

    def projected_wall_clock(
        self, product: Product, strategy: Strategy, grid: TimeGrid, n_models: int
    ) -> float:
        """Seconds: one 1000-path chunk of the pricing model on the grid, scaled."""
        n_probe = 1000
        probe = SimConfig(
            n_paths=n_probe,
            chunk_size=n_probe,
            seed=self.sim.seed,
            dt_max=self.sim.dt_max,
            antithetic=self.sim.antithetic,
        )
        draws = GaussianDraws(
            probe.seed, n_probe, grid.n_steps, self.context.model.n_brownians, probe.antithetic
        )
        t0 = time.perf_counter()
        self.context.model.simulate_chunk(grid, draws, 0, n_probe, probe.scheme)
        per_path = (time.perf_counter() - t0) / n_probe
        n_w = self.world_paths or self.sim.n_paths
        sims = per_path * (self.sim.n_paths * n_models + n_w)
        n_obj = 1 + len(strategy.instruments)
        n_dates = self.schedule.build(product).size
        regress = n_dates * n_obj * (self.sim.n_paths + n_w) * 40 * 1e-8
        return float(sims + regress)

    # -- run -----------------------------------------------------------------------------------

    def run(
        self, product: Product, strategy: Strategy | None = None, **preset_kwargs: Any
    ) -> HedgeResult:
        t_run = time.perf_counter()
        ctx = self.context
        if strategy is None:
            strategy = default_strategy(product, self.preset_context(product), **preset_kwargs)
        T = float(product.maturity)
        fc = ctx.forward_curve
        # instruments: rolls materialised, names unique, default costs applied
        base_dates = self.schedule.build(product)
        instruments = expand_rolls(strategy.instruments, base_dates, fc, T)
        deduplicate_names(instruments)
        for inst in instruments:
            if inst.cost == 0.0:
                inst.cost = self.costs.spot_bps if isinstance(inst, Spot) else self.costs.vol_points
        roll_dates = [i.start for i in instruments if i.start > 0.0]
        dates = self.schedule.build(product, extra=roll_dates)
        # target bumps
        bumps: list[Bump] = []
        dropped: list[str] = []
        needs_vega = any(t in ("vanna",) for t in strategy.target_names)
        for name in strategy.target_names:
            try:
                b = ctx.bump(name, delta_regime=strategy.delta_regime)
            except ValueError as exc:
                dropped.append(f"{name}: {exc}")
                continue
            if b is not None:
                bumps.append(b)
        if needs_vega and not any(b.name == "vega" for b in bumps):
            try:
                bv = ctx.bump("vega")
                if bv is not None:
                    bumps.append(bv)
            except ValueError as exc:
                dropped.append(f"vanna needs vega: {exc}")
        if dropped:
            for d in dropped:
                strategy.notes.append(f"target dropped (unavailable in this pricing context): {d}")
            kept = [
                t for t in strategy.targets if not any(d.startswith(t.name + ":") for d in dropped)
            ]
            if isinstance(strategy, GreekTargetStrategy):
                strategy = strategy.with_targets(kept)
        objects: list[Product] = [product] + [
            i.product for i in instruments if i.product is not None
        ]
        obj_index: dict[int, int | None] = {}
        k = 1
        for j, inst in enumerate(instruments):
            if inst.product is None:
                obj_index[j] = None
            else:
                obj_index[j] = k
                k += 1
        # the product and its constant static legs are regressed as ONE object: least squares is
        # linear, so netting before the regression makes the hedged Greeks' noise that of the net
        # payoff (tiny for a static replication) instead of the sum of the legs' regression
        # noises (measured on the study cliquet: the strip + put legs regressed separately left
        # the residual delta noise at 3x the delta-only P&L under local vol)
        net_index: int | None = None
        net_legs: list[int] = []
        net_weights: list[float] = []
        for j, inst in enumerate(instruments):
            if inst.name not in strategy.static or inst.product is None:
                continue
            v = strategy.static[inst.name]
            whole_life = inst.start <= _TOL and inst.expiry >= T - _TOL
            if callable(v) or inst.roll != "fixed" or not whole_life:
                continue
            net_legs.append(j)
            net_weights.append(float(v))
        run_notes: list[str] = []
        if net_legs:
            net_index = len(objects)
            objects.append(
                Portfolio(
                    [product] + [instruments[j].product for j in net_legs],  # type: ignore[misc]
                    [1.0, *net_weights],
                )
            )
            run_notes.append(
                "static legs netted with the product for the Greeks (one regression): "
                + ", ".join(instruments[j].name for j in net_legs)
            )
        grid = union_grid([ctx.model, self.world], objects, dates, self.sim)
        n_models = 3 + sum(1 + (b.dn is not None) for b in bumps)
        projected = self.projected_wall_clock(product, strategy, grid, n_models)
        if self.verbose:
            print(
                f"[hedger] {product!r} | strategy {strategy.name}: {dates.size} dates, "
                f"{len(objects)} priced objects, {n_models} pricing simulations of "
                f"{self.sim.n_paths} paths + {self.world_paths or self.sim.n_paths} world paths; "
                f"projected wall clock {projected:.0f} s",
                flush=True,
            )
        t0 = time.perf_counter()
        pricer = self._pricer(ctx.model, objects, grid, bumps)
        pricers = [pricer]
        t_price = time.perf_counter() - t0
        try:
            return self._run(
                product,
                strategy,
                instruments,
                objects,
                obj_index,
                net_index,
                net_legs,
                run_notes,
                dates,
                grid,
                bumps,
                pricer,
                pricers,
                t_price,
                t_run,
                projected,
                n_models,
                dict(preset_kwargs),
            )
        finally:
            for p in pricers:
                p.close()

    def _pricer(
        self, model: Model, objects: list[Product], grid: TimeGrid, bumps: Sequence[Bump]
    ) -> ConditionalPricer:
        return ConditionalPricer(
            model,
            objects,
            grid,
            self.sim,
            tuple(bumps),
            self.degree,
            stream_bumps=self.stream_bumps,
            scratch_dir=self.scratch_dir,
            control_variate=self.control_variate,
            surface=self.context.surface,
        )

    def _run(
        self,
        product: Product,
        strategy: Strategy,
        instruments: list[Any],
        objects: list[Product],
        obj_index: dict[int, int | None],
        net_index: int | None,
        net_legs: list[int],
        run_notes: list[str],
        dates: FloatArray,
        grid: TimeGrid,
        bumps: list[Bump],
        pricer: ConditionalPricer,
        pricers: list[ConditionalPricer],
        t_price: float,
        t_run: float,
        projected: float,
        n_models: int,
        preset_kwargs: dict[str, Any],
    ) -> HedgeResult:
        """The rebalancing loop of :meth:`run` (split so the scratch directories of every pricer
        built — the recalibration rebuilds included — are removed whatever happens)."""
        ctx = self.context
        T = float(product.maturity)
        fc = ctx.forward_curve
        world = self._simulate_world(grid)
        idx = grid.fixing_index
        n_w = world.n_paths
        n_i = len(instruments)
        realised = [obj.payoff(world, idx) for obj in objects]
        # per-path accumulators
        pnl_product = np.zeros(n_w)
        pnl_hedges = np.zeros((n_w, n_i))
        costs = np.zeros(n_w)
        pnl_recal = np.zeros(n_w)
        termination = np.full(n_w, np.nan)
        q_prev = np.zeros((n_w, n_i))
        by_date_rows: list[dict[str, Any]] = []
        q_rows: list[dict[str, Any]] = []
        res_rows: list[dict[str, Any]] = []
        recal_rows: list[dict[str, Any]] = []
        kinds = ["value", *strategy.target_names]
        if "vanna" in kinds:
            kinds = [k_ for k_ in kinds if k_ != "vanna"]
            kinds.append("vega")
        kinds = list(dict.fromkeys(kinds))
        greeks_by_date: dict[str, list[FloatArray]] = {}
        moves: dict[str, list[FloatArray]] = {"dS": [], "dX1": [], "dX2": []}
        spot_inst = Spot()
        n_dates = dates.size
        n_refits = 0
        cache_hits = 0
        skew_ref: FloatArray | None = None
        rule = self.recalibration
        world_skew_pricer = None
        self._twin = None
        if rule is not None:
            world_skew_pricer = self._world_skew_pricer(dates, rule, grid, world)
            pricers.append(world_skew_pricer)
            self._twin = self._twin_skew_pricer(world_skew_pricer)
            pricers.append(self._twin)
            if rule.sticky and rule.base_fit is None and ctx.surface is not None:
                # the base marking fit the sticky policies hold their targets at: computed once,
                # at the first rebalancing date, from the pricing surface with the rule's config
                from volsto.calibration.fit_2f import fit_2f_marking

                t_fit = time.perf_counter()
                rule.base_fit = fit_2f_marking(
                    ctx.surface, rule.config(), ssr_target=rule.ssr_target
                )
                msg = (
                    f"recalibration policy {rule.policy}: base marking fit computed from the "
                    f"pricing surface at t = 0 ({rule.base_fit.status}, "
                    f"{time.perf_counter() - t_fit:.1f} s): {rule.base_fit.params!r}"
                )
                log.info(msg)
                run_notes.append(msg)
                rule.log_rows.append(
                    {
                        "t": 0.0,
                        "event": "base_fit",
                        "policy": rule.policy,
                        "params": repr(rule.base_fit.params),
                        "status": rule.base_fit.status,
                    }
                )

        def values_at(
            pr: ConditionalPricer, obj: int, t: float, want: Sequence[str]
        ) -> dict[str, FloatArray]:
            out, _ = pr.evaluate(obj, t, world, want)
            return out

        def netted(
            pr: ConditionalPricer,
            t: float,
            prod: dict[str, FloatArray],
            inst_g: list[dict[str, FloatArray]],
        ) -> dict[str, FloatArray]:
            """The product's Greeks replaced by the net object's (its own value kept for the
            P&L), the netted static legs' Greeks zeroed (their values kept)."""
            if net_index is None:
                return prod
            net_g, _ = pr.evaluate(net_index, t, world, kinds)
            out = {k_: (prod["value"] if k_ == "value" else v) for k_, v in net_g.items()}
            for j in net_legs:
                inst_g[j] = {
                    k_: (v if k_ == "value" else np.zeros(n_w)) for k_, v in inst_g[j].items()
                }
            return out

        def spot_greeks(t: float) -> dict[str, FloatArray]:
            s_t = np.exp(world.log_spot_at(idx[t]))
            scale = spot_inst.scale(t, fc)
            g: dict[str, FloatArray] = {"value": s_t * scale, "delta": np.full(n_w, scale)}
            for kk in kinds:
                if kk not in g:
                    g[kk] = np.zeros(n_w)
            return g

        cur_pricer = pricer
        for kdx, t in enumerate(dates):
            t = float(t)
            t_next = float(dates[kdx + 1]) if kdx + 1 < n_dates else T
            # product at t
            prod, hs = cur_pricer.evaluate(0, t, world, kinds)
            alive = hs.alive
            newly = (~alive) & np.isnan(termination)
            termination[newly] = t
            # instruments at t
            inst_g: list[dict[str, FloatArray]] = []
            active = []
            for j, inst in enumerate(instruments):
                is_active = inst.active(t)
                active.append(is_active)
                oi = obj_index[j]
                if oi is None:
                    inst_g.append(spot_greeks(t))
                elif is_active:
                    inst_g.append(values_at(cur_pricer, oi, t, kinds))
                else:
                    inst_g.append({kk: np.zeros(n_w) for kk in kinds})
            prod = netted(cur_pricer, t, prod, inst_g)
            # vanna from the vega polynomial gradient in ln S
            if "vanna" in strategy.target_names:
                g_obj = 0 if net_index is None else net_index
                fit = cur_pricer.fit(g_obj, t)
                feats, _ = cur_pricer.features(g_obj, world, t)
                s_t = np.exp(world.log_spot_at(idx[t]))
                prod["vanna"] = fit.gradient("vega", feats, 0) / s_t
                for j in range(n_i):
                    oi = obj_index[j]
                    if oi is None or not active[j] or j in net_legs:
                        inst_g[j]["vanna"] = np.zeros(n_w)
                    else:
                        f_ = cur_pricer.fit(oi, t)
                        ff, _ = cur_pricer.features(oi, world, t)
                        inst_g[j]["vanna"] = f_.gradient("vega", ff, 0) / s_t
            # recalibration
            if rule is not None and world_skew_pricer is not None:
                skew_now = self._world_skew(world_skew_pricer, kdx, t, rule, world)
                if skew_ref is None:
                    skew_ref = skew_now
                moved = float(np.max(np.abs(skew_now - skew_ref)))
                did = False
                if moved > rule.skew_move_threshold and n_refits < rule.max_refits:
                    new_ctx, hit = self._recalibrate(t, world_skew_pricer, kdx, rule, world)
                    if new_ctx is not None:
                        n_refits += 1
                        cache_hits += int(hit)
                        new_pricer = self._pricer(new_ctx.model, objects, grid, bumps)
                        pricers.append(new_pricer)
                        new_prod, _ = new_pricer.evaluate(0, t, world, ["value"])
                        d_recal = new_prod["value"] - prod["value"]
                        pnl_recal += np.where(alive, d_recal, 0.0)
                        cur_pricer = new_pricer
                        prod, hs = cur_pricer.evaluate(0, t, world, kinds)
                        inst_g = []
                        for j in range(n_i):
                            oi = obj_index[j]
                            if oi is None:
                                inst_g.append(spot_greeks(t))
                            elif active[j]:
                                inst_g.append(values_at(cur_pricer, oi, t, kinds))
                            else:
                                inst_g.append({kk: np.zeros(n_w) for kk in kinds})
                        prod = netted(cur_pricer, t, prod, inst_g)
                        skew_ref = skew_now
                        did = True
                recal_rows.append(
                    {
                        "t": t,
                        "skew_move": moved,
                        "threshold": rule.skew_move_threshold,
                        "policy": rule.policy,
                        "recalibrated": did,
                        "params": None if not did else repr(cur_pricer.model),
                    }
                )
            # strategy
            sol: HedgeSolution = strategy.solve(t, prod, inst_g, active, q_prev)
            q = sol.quantities
            q[~alive] = 0.0  # unwind after termination
            dq = q - q_prev
            s_t = np.exp(world.log_spot_at(idx[t]))
            for j, inst in enumerate(instruments):
                if np.any(dq[:, j] != 0.0):
                    costs += inst.transaction_cost(dq[:, j], t, s_t, fc, product.discount)
            # values at t_next (realised at maturity / expiry)
            if kdx + 1 < n_dates:
                prod_next, _ = cur_pricer.evaluate(0, t_next, world, ["value"])
                v_next_prod = prod_next["value"]
            else:
                v_next_prod = realised[0]
            d_prod = v_next_prod - prod["value"]
            pnl_product += d_prod
            leg = np.zeros((n_w, n_i))
            for j, inst in enumerate(instruments):
                if not np.any(q[:, j] != 0.0):
                    continue
                oi = obj_index[j]
                if oi is None:
                    v_now = inst_g[j]["value"]
                    v_next = (
                        np.exp(world.log_spot_at(idx[t_next])) * spot_inst.scale(t_next, fc)
                        if kdx + 1 < n_dates
                        else realised_spot(world, idx, T, fc)
                    )
                else:
                    v_now = inst_g[j]["value"]
                    if kdx + 1 < n_dates and inst.active(t_next):
                        v_next = values_at(cur_pricer, oi, t_next, ["value"])["value"]
                    else:
                        v_next = realised[oi]
                leg[:, j] = q[:, j] * (v_next - v_now)
            pnl_hedges += leg
            # records
            row: dict[str, Any] = {"t": t, "product": float(d_prod.mean()), "costs": 0.0}
            for j, inst in enumerate(instruments):
                row[f"hedge:{inst.name}"] = float(leg[:, j].mean())
            by_date_rows.append(row)
            qrow: dict[str, Any] = {"t": t}
            for j, inst in enumerate(instruments):
                qrow[f"q_mean:{inst.name}"] = float(q[:, j].mean())
                qrow[f"q_std:{inst.name}"] = float(q[:, j].std())
            q_rows.append(qrow)
            rrow: dict[str, Any] = {"t": t, "alive": float(alive.mean())}
            for gi, g in enumerate(sol.target_names):
                rrow[f"residual:{g}"] = (
                    float(np.mean(np.abs(sol.residual[alive, gi]))) if alive.any() else 0.0
                )
                rrow[f"exposure:{g}"] = (
                    float(np.mean(np.abs(prod[g][alive]))) if alive.any() else 0.0
                )
            res_rows.append(rrow)
            for g in strategy.target_names:
                greeks_by_date.setdefault(g, []).append(prod[g].copy())
            moves["dS"].append(np.exp(world.log_spot_at(idx[t_next])) - s_t)
            if world.n_factors >= 1 and world.n_factors == ctx.model.n_factors:
                f_now, f_next = world.factors_at(idx[t]), world.factors_at(idx[t_next])
                moves["dX1"].append(f_next[:, 0] - f_now[:, 0])
                if world.n_factors >= 2:
                    moves["dX2"].append(f_next[:, 1] - f_now[:, 1])
            q_prev = q
        v0, v0_se = pricer.value_at_zero(0)
        budget = {
            "pricing_simulations": float(n_models),
            "pricing_seconds": float(t_price),
            "wall_seconds": float(time.perf_counter() - t_run),
            "projected_seconds": float(projected),
            "n_dates": float(n_dates),
            "n_objects": float(len(objects)),
            "pricing_paths_gb": float(pricer.memory_bytes / 1e9),
            "streamed_paths_gb": float(pricer.streamed_bytes / 1e9),
            "refits": float(n_refits),
            "cache_hit_rate": float(cache_hits / n_refits) if n_refits else float("nan"),
        }
        # the §7.11 control's median variance reduction over dates and controlled objects
        # (NaN when nothing was controlled: control off, no surface-driven bump or no proxy)
        reductions = [r for p in pricers for r in p.cv_reductions()]
        red_ses = [r for p in pricers for r in p.cv_reduction_ses()]
        budget["cv_reduction_median"] = float(np.median(reductions)) if reductions else float("nan")
        # the spread over dates and objects, and the median bootstrap se of one reduction
        budget["cv_reduction_q25"] = (
            float(np.quantile(reductions, 0.25)) if reductions else float("nan")
        )
        budget["cv_reduction_q75"] = (
            float(np.quantile(reductions, 0.75)) if reductions else float("nan")
        )
        budget["cv_reduction_se_median"] = float(np.median(red_ses)) if red_ses else float("nan")
        notes = tuple(
            dict.fromkeys(
                [
                    *pricer.notes,
                    *strategy.notes,
                    *run_notes,
                    *ctx.notes,
                    *[n for i in instruments for n in i.notes],
                ]
            )
        )
        return HedgeResult(
            repr(product),
            strategy.name,
            dates,
            tuple(i.name for i in instruments),
            strategy.target_names,
            pnl_product,
            pnl_hedges,
            costs,
            pnl_recal,
            termination,
            v0,
            v0_se,
            pd.DataFrame(by_date_rows),
            pd.DataFrame(q_rows),
            pd.DataFrame(res_rows),
            world,
            notes,
            pd.DataFrame(recal_rows),
            budget,
            {
                "schedule": self.schedule.frequency if self.schedule.dates is None else "explicit",
                "costs": (self.costs.spot_bps, self.costs.vol_points),
                "delta_regime": strategy.delta_regime,
                "pricing": ctx.label,
                "world": repr(self.world),
                "n_paths_pricing": self.sim.n_paths,
                "n_paths_world": n_w,
                "recalibration": None if rule is None else "on_skew_move",
                "recalibration_policy": None if rule is None else rule.policy,
                "skew_move_threshold": None if rule is None else rule.skew_move_threshold,
                "stream_bumps": self.stream_bumps,
                "scratch_dir": None if self.scratch_dir is None else str(self.scratch_dir),
                "control_variate": self.control_variate,
                # the preset's keyword arguments (run(product, q=0.5)), or those the strategy
                # was built with by default_strategy when it is passed in
                "preset_kwargs": dict(preset_kwargs)
                or dict(getattr(strategy, "preset_kwargs", None) or {}),
            },
            {k_: np.array(v) for k_, v in greeks_by_date.items()},
            {k_: np.array(v) for k_, v in moves.items() if v},
        )

    # -- recalibration internals -------------------------------------------------------------

    def _world_skew_pricer(
        self, dates: FloatArray, rule: RecalibrationRule, grid: TimeGrid, world: PathSet
    ) -> ConditionalPricer:
        """The skew strip under the **world** model: at every rebalancing date ``t`` and pillar
        ``τ`` three forward-start options ``(S_{t+τ}/S_t − e^{k})^±`` at ``k ∈ {−h, 0, +h}`` —
        struck at the *forward moneyness of each path* (a vanilla struck at ``F(T) e^{k}`` would
        sit anywhere from far below to far above a path's own forward at ``t``; inverting it at
        the path's forward mixes moneyness across paths).  Their conditional values at ``t`` are
        homogeneous in ``S_t`` and invert path by path to the world's conditional smile at ``t``
        (:meth:`_state_surface`).  No bumps, no control variate; streamed like the other pricers."""
        from volsto.products.forward_start import ForwardStartOption

        fc = self.world.forward_curve
        disc = fc.rate_curve
        objs: list[Product] = []
        for t in dates:
            for tau in rule.pillars:
                for k in (-rule.h, 0.0, rule.h):
                    objs.append(
                        ForwardStartOption(
                            float(t), float(t) + tau, float(np.exp(k)), 1 if k >= 0 else -1, disc
                        )
                    )
        g = union_grid([self.world], objs, dates, self.sim)
        wsim = self._world_sim()
        return ConditionalPricer(
            self.world,
            objs,
            g,
            wsim,
            (),
            self.degree,
            seed=wsim.seed + 7,
            stream_bumps=self.stream_bumps,
            scratch_dir=self.scratch_dir,
            control_variate=False,
        )

    def _twin_skew_pricer(self, strip: ConditionalPricer) -> ConditionalPricer:
        """The same strip under the **pricing** model on the same grid, paths and seed (a CRN
        twin): the pricing model's own prediction of the conditional smile at every date, so the
        recalibration trigger reads the world's skew *in excess* of it (:meth:`_world_skew`)."""
        return ConditionalPricer(
            self.context.model,
            list(strip.objects),
            strip.grid,
            strip.sim,
            (),
            self.degree,
            seed=strip.seed,
            stream_bumps=self.stream_bumps,
            scratch_dir=self.scratch_dir,
            control_variate=False,
        )

    def _state_surface(
        self, pr: ConditionalPricer, kdx: int, t: float, rule: RecalibrationRule, world: PathSet
    ) -> _StateSurface:
        """The conditional smile at ``t`` read **per path** from the strip of
        :meth:`_world_skew_pricer` and averaged: each forward-start's regressed conditional value
        (time-0 money) is taken
        to time-``t`` money, inverted on the ratio (forward ``F(t + τ)/F(t)``, strike ``e^{k}``,
        maturity ``τ``, discount ``DF(t + τ)/DF(t)``) and the ATMF vol, the central-difference
        skew and the second-difference curvature across the three strikes are averaged over the
        paths where the inversion is finite (a regressed value below intrinsic inverts to NaN and
        is dropped; a kept fraction under :data:`STATE_SURFACE_MIN_INVERTED` is logged).  The M8
        reading inverted the path-*averaged* price — the unconditional ``(t + τ)``-option, whose
        "vol over τ" carries the spot variance over ``[0, t]`` (measured: the 3M ATMF vol read
        0.218 at ``t = 0`` and 0.394 at ``t = 0.5`` on a world without any shock, the factor
        ``sqrt((t + τ)/τ)``), so the proxy moved at every date and the rule refit at every
        date."""
        from volsto.market.bs import implied_vol

        fc = self.world.forward_curve
        n_pill = len(rule.pillars)
        df_t = float(fc.rate_curve.df(t))
        h = float(rule.h)
        atf, skew, curv = [], [], []
        for pi, tau in enumerate(rule.pillars):
            T = t + tau
            base_obj = kdx * n_pill * 3 + pi * 3
            f_ratio = float(fc.forward(T) / fc.forward(t))
            df = float(fc.rate_curve.df(T)) / df_t
            ivs = []
            for si, k in enumerate((-h, 0.0, h)):
                out, _ = pr.evaluate(base_obj + si, t, pr.paths, ["value"])
                iv = implied_vol(
                    out["value"] / df_t, f_ratio, float(np.exp(k)), tau, 1 if k >= 0 else -1, df
                )
                ivs.append(np.asarray(iv, dtype=np.float64))
            lo, mid, hi = ivs
            ok = np.isfinite(lo) & np.isfinite(mid) & np.isfinite(hi)
            if not ok.any():
                raise ValueError(
                    f"state surface at t={t:g}, pillar {tau:g}: no path inverts to a finite vol"
                )
            if ok.mean() < STATE_SURFACE_MIN_INVERTED:
                log.info(
                    "state surface t=%g pillar %g: %.1f%% of the paths invert",
                    t,
                    tau,
                    100 * ok.mean(),
                )
            atf.append(float(mid[ok].mean()))
            skew.append(float(((hi - lo)[ok] / (2.0 * h)).mean()))
            curv.append(float(((hi - 2.0 * mid + lo)[ok] / (h * h)).mean()))
        return _StateSurface(
            np.asarray(rule.pillars, dtype=np.float64),
            np.asarray(atf),
            np.asarray(skew),
            np.asarray(curv),
            fc,
        )

    def _world_skew(
        self, pr: ConditionalPricer, kdx: int, t: float, rule: RecalibrationRule, world: PathSet
    ) -> FloatArray:
        """The world's conditional skew at ``t`` **in excess of the pricing model's own
        prediction** at the same date: the CRN twin strip (:meth:`_twin_skew_pricer`) is read
        by :meth:`_state_surface` the same way and the rule triggers on the *change* of the
        excess since the last refit (the reference is the excess at ``t = 0``, reset at each
        refit).  An LSV world's forward skew at date ``t`` differs from its spot skew by far more
        than the threshold with no shock at all (measured on the SPX marking fit: the M8 rule,
        which compared the world's skew with its own ``t = 0`` value, refit at every one of 11
        monthly dates), whereas the excess is exactly 0 under world = pricing (identical
        simulations), the shock's rota shows up in it after the shock, and the held rotation
        triggers no further refit once the reference has reset."""
        twin = self._twin
        assert twin is not None
        sw = self._state_surface(pr, kdx, t, rule, world)
        sp = self._state_surface(twin, kdx, t, rule, world)
        return np.asarray(sw.skew - sp.skew, dtype=np.float64)

    def _recalibrate(
        self, t: float, pr: ConditionalPricer, kdx: int, rule: RecalibrationRule, world: PathSet
    ) -> tuple[PricingContext | None, bool]:
        ctx = self.context
        if ctx.state is None or ctx.builder is None:
            ctx.notes.append("recalibration rule needs an LSV pricing state: skipped")
            return None, False
        surf = self._state_surface(pr, kdx, t, rule, world)
        from volsto.calibration.fit_2f import fit_2f, marking_targets_for
        from volsto.market.varswap import xi0_curve

        cfg = rule.config()
        held_flags: tuple[str, ...] = ()
        if rule.refit is not None:
            params = rule.refit(surf, ctx.state.spec.model)
        else:
            targets = marking_targets_for(surf, cfg, ssr_target=rule.ssr_target)
            if rule.sticky:
                if rule.base_fit is None:
                    raise ValueError(
                        f"recalibration policy {rule.policy!r} needs base_fit (the hedger computes "
                        "it at the first rebalancing date from the pricing surface)"
                    )
                n_flags = len(targets.flags)
                targets = held_targets(targets, rule.base_fit.targets, rule.policy)
                held_flags = tuple(targets.flags[n_flags:])
            xi0 = xi0_curve(
                ctx.surface, float(min(ctx.surface.max_maturity, max(targets.pillars) + t))
            )
            params = fit_2f(targets, xi0, cfg).params
        changes = {f.name: float(getattr(params, f.name)) for f in dataclasses.fields(params)}
        new_state = _with_params(ctx.state, changes)
        if hasattr(ctx.builder, "has"):
            hit = bool(ctx.builder.has(new_state))
        elif hasattr(ctx.builder, "cache"):
            hit = bool(ctx.builder.cache.has(new_state.spec))
        else:
            hit = False
        model = ctx.builder.build(new_state, "recalibrate")
        rule.log_rows.append(
            {
                "t": t,
                "event": "refit",
                "policy": rule.policy,
                "params": repr(params),
                "cache_hit": hit,
                "held": "; ".join(held_flags),
            }
        )
        return PricingContext(model, new_state, ctx.builder, ctx.surface, ctx.label), hit


def _with_params(state: RiskState, changes: Mapping[str, float]) -> RiskState:
    """``state`` with the model parameters ``changes`` applied (a typed ``with_params``)."""
    model = state.spec.model.replace(**{k: float(v) for k, v in changes.items()})
    label = ",".join(f"{k}={v:g}" for k, v in changes.items())
    return RiskState(dataclasses.replace(state.spec, model=model), state.x0, label)


def realised_spot(world: PathSet, idx: Any, T: float, fc: Any) -> FloatArray:
    """The spot instrument's realised value at ``T`` in time-0 money."""
    return np.asarray(np.exp(world.log_spot_at(idx[T])) * float(fc.spot / fc.forward(T)))


__all__ = [
    "FREQUENCIES",
    "SKEW_MOVE_THRESHOLD",
    "SPOT_BUMP",
    "TENT_SIZE",
    "VOL_BUMP",
    "Costs",
    "HedgeResult",
    "Hedger",
    "PricingContext",
    "RecalibrationRule",
    "Schedule",
    "realised_spot",
]
