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
   target on the state surface; ``"sticky_breakeven"`` holds the SpotVolCovar and correlation
   targets at the base fit's values and lets the skew constraint follow the state surface;
   ``"sticky_breakeven_skew"`` holds the skew reference too) and the pricing model rebuilt through
   the leverage cache (hit rate reported); the **recalibration P&L** is ``V_k(new set) − V_k(old
   set)`` at the same world state, isolated as its own leg.  The rule's state surfaces are read
   from forward-start strips at their own path count (``RecalibrationRule.strip_paths``) and
   **precomputed for every date before the loop** (:class:`StripSurfaces`): each strip pricer is
   released as soon as its surfaces are read, so no strip coexists with the hedge loop; the
   refit's correlation target is guarded (:func:`refit_targets`: the base fit's correlation held
   on a date whose step-0 reduction is degenerate, ``|Corr_BE|`` capped at
   :data:`REFIT_CORRELATION_CAP`).

**Minimum-variance delta** (``delta_regime="min_variance"``, :func:`min_variance_delta`): the
strategy's ``delta`` target — the product's and every priced instrument's — is replaced by the
pricing model's minimum-variance spot-only delta, the model delta plus the vol-correlation term
``Σ_i (∂V/∂X_i) d⟨X_i, S⟩/d⟨S, S⟩`` from the model's own SDE (:func:`spot_factor_projection`), the
value gradients ``∂V/∂X_i`` of the regression; no extra simulation.  The gradients are read from
the **controlled** value regression (``ConditionalPricer(control_value=True)``, the §7.11
Black–Scholes shadow of the value target, whose conditional expectation depends on ``S_t`` only:
the factor dependence stays, the spot-driven payoff noise goes) — measured on the M8b study-D
vanilla, the raw value regression's gradients made the minimum-variance hedge WORSE than the
model delta (P&L std 3.218 ± 0.029 against 3.144 ± 0.011 % of spot, the correction's cross-path
std 0.16 at ``t = 0.1`` against a mean of −0.22; at ``t = 0`` the borrowed gradient had the wrong
sign), the controlled one takes it to 1.736 ± 0.014 (the recorded run: 2·10⁴ paths, before the
§7.11 coefficient fix of 2026-09-16; at 5·10³ pricing paths it measured 2.756 ± 0.053 and
2.965 ± 0.068 on two seeds, worse than sticky-strike — the benchmark row is not robust to the
pricing path count, and study D measures distances to the regimes' common λ*-implied value).  An
object without a Black–Scholes proxy (autocall, cliquet) keeps the raw gradients, noted.  In an
incomplete (stochastic-vol) pricing model the model delta is **not** the minimum-variance
spot-only hedge even when world = pricing (M8b study D, owner's decision of 2026-09-16).

Total P&L per path ``= product leg + Σ hedge legs − costs``; the zero-cost total is reported next
to it (costs are additive).  The product leg sums to ``payoff − V₀`` exactly on every path
(telescoping), so the hedged P&L's mean is the product's pricing error plus the hedging noise.

**Compute budget** (:meth:`Hedger.projected_wall_clock`, printed before a run): one pricing-model
simulation per bumped model (``1 + 2`` spot bumps ``+`` one or two per target bump) on
``n_paths`` paths over the grid, one world simulation, and ``n_dates × n_objects`` regressions
of ``O(n_paths × n_basis²)``; the estimate scales a one-chunk timing.  Long runs are shard-able by
path blocks (the M9 ``--shard`` form): ``world_paths`` and ``world_seed`` select the block.
The recalibration strips are projected separately (:meth:`Hedger.projected_strip_seconds`, at
``strip_paths``).  **Memory:** the bumped path sets are kept for the hybrid targets
(``budget["pricing_paths_gb"]``; the strips' resident footprint before their release is
``budget["strip_world_gb"]`` / ``budget["strip_twin_gb"]``);
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
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.config import SimConfig, SurfacePerturbation
from volsto.engine.grid import FixingIndex, TimeGrid
from volsto.engine.paths import PathSet
from volsto.engine.rng import GaussianDraws
from volsto.hedging.controls import MIN_SHADOW_VOL
from volsto.hedging.instruments import Spot, deduplicate_names, expand_rolls
from volsto.hedging.pricing import (
    DEFAULT_CONTROL_DELTA,
    VALUE_CV,
    Bump,
    ConditionalPricer,
    ObjectPayoffs,
    union_grid,
)
from volsto.hedging.strategies import (
    DELTA_REGIMES,
    MIN_VARIANCE_REGIME,
    SURFACE_DELTA_REGIMES,
    GreekTargetStrategy,
    HedgeSolution,
    PresetContext,
    Strategy,
    default_strategy,
)
from volsto.market.surface import ArbitrageError
from volsto.models.base import Model, ModelState
from volsto.models.bergomi import BergomiSV
from volsto.models.lsv import LSV
from volsto.products.base import Portfolio, Product
from volsto.risk.engine import BSBuilder, LSVBuilder, LVBuilder, RiskState, surface_of
from volsto.risk.greeks import _spot_state
from volsto.risk.ladders import PILLARS as RISK_PILLARS
from volsto.risk.ladders import bucket_epsilon
from volsto.risk.shadow_rotation import RECALIBRATION_POLICIES, held_targets

if TYPE_CHECKING:
    from volsto.calibration.targets import TargetSet

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
#: default ``RecalibrationRule.strip_paths``: the path count of the forward-start strips the rule
#: reads its state surfaces from, independent of the hedger's world paths.  The strip's ATM level
#: and skew converge fast but its curvature — a second difference over ``h² = 0.0025`` of three
#: regressed conditional prices — does not: measured on the M8b study-C 3y autocall shock world
#: (+1 rota, t = 0.9615, pillars 3M / 1Y / 3Y) it reads −1.22 / −0.53 / −0.34 at 2·10⁴ paths
#: against −0.56 / +0.39 / +0.56 at 1.6·10⁵, step 0's radicand guard fires at all three pillars
#: at 2·10⁴, at one at 4·10⁴ and at none from 8·10⁴ (owner's decision of 2026-09-16: 8·10⁴).
#: That holds for that state only.  On the 1y daily VKO put shock world (+3 rota) the 3M pillar's
#: curvature is still unconverged at 8·10⁴: 136 of the 252 dates read a degenerate step 0 on the
#: world strip, three of the four refit dates that used the guarded fallback clear only at 3.2·10⁵,
#: and one refit lands at the correlation bound with no step-0 flag (ρ_SX1 = −0.9944 at
#: t = 0.7302; regular at 3.2·10⁵) — at this count the guarded fallback, not the path count, is
#: what protects the short pillar (M8b verification, 2026-09-16).  After the forward-moneyness
#: fix of the strip (same day) the count is 133 of 252 at ``h = 0.05`` and 54 of 252 with a
#: ``±0.10`` curvature stencil (:attr:`RecalibrationRule.curvature_h`; strip phase 459 s and
#: 751 s, world + twin); the pricing model's own twin strip (no shock) reads a degenerate 3M
#: step 0 on 143 and 99 dates — the short pillar's conditional curvature is negative enough to
#: fire the guard on many dates, not only noisy
DEFAULT_STRIP_PATHS = 80_000
#: the largest ``|Corr_BE|`` a refit may target (:func:`refit_targets`).  At ``Corr_BE = −1`` the
#: marking VolVar target is ``SpotVolCovar²`` and, with ``λ1, λ2`` of the same sign, the fully
#: collapsed set ``ρ_SX1 = ρ_SX2 = −1, ρ12 = +1`` is step 3's exact minimiser.  Measured on the
#: named study-C state (autocall 3y, +1 rota, t = 0.9615, 2·10⁴ strip paths): a cap of 0.975
#: still returns ``ρ_SX1 = −0.9904`` (beyond :data:`CORRELATION_BOUND`), 0.97 returns
#: −0.988 / −0.939 / +0.907, and the cap costs nothing in step 2 (``λ``, ``k1``, the achieved
#: skew gaps and SpotVolCovar misses unchanged; ν +0.9%).  A converged 3M read sits at −0.955
#: (1.6·10⁵ strip paths), inside the cap
REFIT_CORRELATION_CAP = 0.97
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
    beyond the maturity.  ``product_fixings=False`` keeps the frequency grid only (a fixing
    that lies on the grid is still a rebalancing date; the state at a date includes every
    fixing up to it, so a knock-in monitored daily is seen at the next grid date): the M8b
    studies use it — a Phoenix with 757 daily knock-in fixings would otherwise rebalance
    daily under a "weekly" schedule (900 dates, 1560 s per run against 156 dates)."""

    frequency: str = "daily"
    dates: tuple[float, ...] | None = None
    after_fixing: float = 0.0
    product_fixings: bool = True

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
        added = fix if self.product_fixings else np.zeros(0)
        cand = np.unique(np.concatenate([base, added, np.asarray(extra, dtype=np.float64), [0.0]]))
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
            if name == "delta" and delta_regime in SURFACE_DELTA_REGIMES:
                return self.regime_delta_bump(delta_regime)
            if name == "delta" and delta_regime not in DELTA_REGIMES:
                raise ValueError(f"delta_regime must be one of {DELTA_REGIMES}")
            # "model" and "min_variance": the hybrid-CRN spot bumps (the latter adds the
            # vol-correlation term from the value regression, min_variance_delta)
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
        """The spot-kind :class:`Bump` of a §7.2 surface regime (``SURFACE_DELTA_REGIMES``, or
        ``"model"`` through the builder): the states moved by ``±SPOT_BUMP`` under the regime,
        rebuilt (leverage recalibrated through the cache), with the moved states' surfaces for the
        delta control's proxy vols."""
        if regime != "model" and regime not in SURFACE_DELTA_REGIMES:
            raise ValueError(
                f"regime_delta_bump: {regime!r} is neither 'model' nor a surface regime "
                f"{SURFACE_DELTA_REGIMES} (min_variance is not a bump: min_variance_delta)"
            )
        if self.state is None:
            raise ValueError("a non-model delta regime needs a pricing state")
        h = SPOT_BUMP
        up_state, mode_up = _spot_state(self.state, regime, h)
        dn_state, mode_dn = _spot_state(self.state, regime, -h)
        up = self.builder.build(up_state, mode_up)
        dn = self.builder.build(dn_state, mode_dn)
        return Bump(
            "delta",
            up,
            dn,
            h,
            f"delta under the {regime} regime",
            kind="spot",
            up_surface=surface_of(up_state),
            dn_surface=surface_of(dn_state),
        )


# --------------------------------------------------------------------------------------------
# recalibration rule
# --------------------------------------------------------------------------------------------


@dataclass
class RecalibrationRule:
    """``"on_skew_move"`` (module docstring): at each rebalancing date the world's conditional
    smile at the ``pillars`` (tenors from the date) is read from a three-strike forward-start
    strip regressed under the **world** model (:meth:`Hedger._state_surface`), and the trigger is
    its ATM skew in excess of the pricing model's own prediction (a CRN twin strip,
    :meth:`Hedger._world_skew`); when the excess moved by more than ``skew_move_threshold`` (vol
    per unit log-moneyness, any pillar; default :data:`SKEW_MOVE_THRESHOLD`) since the last refit
    the P1 set is refit on the state surface (:func:`~volsto.calibration.fit_2f.fit_2f` on the
    targets of :func:`refit_targets`) under ``policy``
    (:data:`~volsto.risk.shadow_rotation.RECALIBRATION_POLICIES`, the semantics of
    :func:`~volsto.risk.shadow_rotation.refit_on_rotated`): ``"sabr_linked"`` refits every target
    on the state surface; ``"sticky_breakeven"`` holds ``spot_vol_covar`` and ``correl_target`` at
    the **base** fit's values and lets the skew constraint (and the ATMF vols) follow the state
    surface; ``"sticky_breakeven_skew"`` holds the two-point skew reference too
    (:func:`~volsto.risk.shadow_rotation.held_targets`).

    **The strips** run at ``strip_paths`` (default :data:`DEFAULT_STRIP_PATHS`), for the world
    strip and its twin alike and independent of ``Hedger.world_paths``; their state surfaces are
    precomputed for every date before the hedge loop (:class:`StripSurfaces`) and the refit reads
    the same surface the trigger read.  The strip is struck at **forward** moneyness
    (:meth:`Hedger._strip_objects`): the ATMF level and skew are read at forward log-moneyness
    ``{−h, 0, +h}``, the curvature at ``{−curvature_h, 0, +curvature_h}`` (``curvature_h``
    ``None``: equal to ``h``, the three-strike strip; otherwise two more strikes per pillar).
    The curvature is a second difference whose Monte Carlo error scales as
    ``1/(curvature_h √strip_paths)`` while its smoothing of the smile grows with
    ``curvature_h``; the base fit reads the pricing snapshot on the same stencil
    (:meth:`marking_targets`).

    **The refit's correlation target** (:func:`refit_targets`).  Step 0's radicand guard
    (:func:`~volsto.calibration.targets.sabr_reduce`) is correct for a genuinely negative ATM
    curvature, but it must not fire on Monte Carlo noise: once it fires ``Corr_SABR`` is
    ``2 smi/(√3|smi|)`` clipped to ``∓1`` whatever the smile, ``sabr_linked`` then targets
    ``Corr_BE = −1`` and the collapsed set ``ρ_SX1 = ρ_SX2 = −1, ρ12 = +1`` is step 3's exact
    minimiser (measured on 41 of the 113 ``sabr_linked`` refits of the 2026-09-15 study-C runs,
    whose strips ran at 2·10⁴ paths).  Two things protect the refit from that: the strip path
    count (the curvature converges) and the cap ``|Corr_BE| <= correlation_cap`` (default
    :data:`REFIT_CORRELATION_CAP`, every refit, recorded as ``corr_capped``); the last belt is
    the **guarded fallback** — on a date where step 0's guard fired or its ``rho`` clip applied
    at any fitted pillar, the base fit's ``correl_target`` is held for that date only (recorded
    as ``fallback_applied``).  The skew target is not the problem (the two-point QP is feasible
    at the pinned dates) and is not relaxed.

    The base fit is ``base_fit`` (a :class:`~volsto.calibration.fit_2f.FitResult`); when
    ``None`` the hedger computes the marking fit of the pricing surface once, before the first
    rebalancing date, with the rule's config and ``ssr_target``
    (:func:`~volsto.calibration.fit_2f.fit_2f_marking`) — under every policy, since the
    guarded fallback needs it under ``sabr_linked`` too —, stores it here and logs it.  The
    pricing model is rebuilt with the new parameters through the cache (``builder``).  ``refit``
    may replace the fit call (``(state_surface, base_params) -> params``; the policy, the
    fallback and the cap are then the callable's business).  ``log_rows`` records every refit
    with its policy, its step-0 flags, ``fallback_applied``, ``corr_capped`` and ``at_bound``.
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
    strip_paths: int = DEFAULT_STRIP_PATHS
    correlation_cap: float = REFIT_CORRELATION_CAP
    curvature_h: float | None = None

    def __post_init__(self) -> None:
        if self.policy not in RECALIBRATION_POLICIES:
            raise ValueError(f"policy must be one of {RECALIBRATION_POLICIES}")
        if not (self.skew_move_threshold > 0.0):
            raise ValueError("skew_move_threshold must be positive")
        if int(self.strip_paths) != self.strip_paths or self.strip_paths < 2:
            raise ValueError("strip_paths must be an integer of at least 2")
        if not (0.0 < self.correlation_cap <= 1.0):
            raise ValueError("correlation_cap must be in (0, 1]")
        if not (self.h > 0.0):
            raise ValueError("h must be positive")
        if self.curvature_h is not None and not (self.curvature_h > 0.0):
            raise ValueError("curvature_h must be positive (or None: equal to h)")

    @property
    def curvature_stencil(self) -> float:
        """The half-width of the curvature read: ``curvature_h``, or ``h`` when it is ``None``."""
        return float(self.h if self.curvature_h is None else self.curvature_h)

    def strip_log_moneyness(self) -> tuple[float, ...]:
        """The strip's forward log-moneyness per pillar, in object order: ``(−h, 0, +h)``, then
        ``(−curvature_h, +curvature_h)`` when the curvature stencil differs from ``h``."""
        h, ch = float(self.h), self.curvature_stencil
        return (-h, 0.0, h) if ch == h else (-h, 0.0, h, -ch, ch)

    def marking_targets(self, surface: Any) -> TargetSet:
        """The marking targets of ``surface`` read **on the strip's stencil**
        (:func:`~volsto.calibration.fit_2f.marking_targets_for` with the rule's config and
        ``ssr_target``, the skew by the central difference of half-width ``h`` and the curvature
        by that of half-width :attr:`curvature_stencil`): the base fit on the pricing snapshot
        and every refit on a state surface read the smile the same way, so a refit never sees a
        move that is only a stencil mismatch.  On a state surface (a quadratic in ``k``) both
        differences are exact."""
        from volsto.calibration.fit_2f import marking_targets_for

        return marking_targets_for(
            surface,
            self.config(),
            ssr_target=self.ssr_target,
            h=self.curvature_stencil,
            skew_h=float(self.h),
        )

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


@dataclass
class _StripPricer(ConditionalPricer):
    """A :class:`~volsto.hedging.pricing.ConditionalPricer` for the recalibration strips, lean in
    memory and time, whose ``value`` regressions are **identical** to the full pricer's on the same
    grid, model, seed and paths (``tests/test_hedging.py`` checks it bit for bit):

    * only the base model is simulated — the strips read values, never the CRN spot bumps, so the
      two bumped simulations (and their path sets) of the full pricer are skipped; the delta
      estimator is ``"gradient"`` with ``hybrid_bumps=False``, so no hybrid path set or hybrid
      payoff matrix is ever built (the value regression shares nothing with the delta targets:
      same design matrix, its own right-hand side);
    * the paths are simulated chunk by chunk on the full grid, every object's payoff taken on the
      chunk, and only what a strip regression reads is kept: ``ln S`` and the factors at
      ``keep_times`` (the rebalancing dates whose objects the pricer carries — a forward-start
      option at its own start date has the hedge state ``(started = 1, u_start = 0)``);
      ``variance`` / ``int_var`` / ``sum_sq`` are zero-memory placeholders and :attr:`idx` maps
      the kept dates to the kept columns;
    * ``date_slots`` maps a rebalancing-date index to the block of the pricer's objects (three
      strikes per pillar, in order) — a pricer may carry a subset of the dates.

    Measured need (M8b study C): at 2·10⁴ paths the full strip pricer of a 1y daily product (2268
    objects, 756 record columns) held its three path sets, three payoff matrices and three dates
    of hybrid payoffs — the larger part of the 9–13 GB of a recalibration run."""

    keep_times: tuple[float, ...] = ()
    date_slots: dict[int, int] = field(default_factory=dict)
    #: seconds spent simulating and taking payoffs in :meth:`_simulate` (the projection's probe)
    timing: dict[str, float] = field(init=False, default_factory=dict)
    _index: FixingIndex | None = field(init=False, default=None, repr=False)
    _owned_bytes: int = field(init=False, default=0)

    def _simulate(self) -> None:
        full = self.grid.fixing_index
        draws = GaussianDraws(
            self.sim.seed if self.seed is None else self.seed,
            self.sim.n_paths,
            self.grid.n_steps,
            self.model.n_brownians,
            self.sim.antithetic,
        )
        times = np.asarray(sorted(set(float(t) for t in self.keep_times)), dtype=np.float64)
        if times.size == 0:
            raise ValueError("a strip pricer needs the dates it is read at (keep_times)")
        cols = full.indices(times)
        n, n_obj, nf = self.sim.n_paths, len(self.objects), self.model.n_factors
        pay = np.empty((n, n_obj))
        ls = np.empty((n, times.size))
        fac = np.empty((n, times.size, nf))
        t_sim = t_pay = 0.0
        for p0, p1 in self.sim.chunk_ranges(self.grid.n_records, nf):
            t0 = time.perf_counter()
            ps = self.model.simulate_chunk(self.grid, draws, p0, p1, self.sim.scheme)
            t1 = time.perf_counter()
            for j, obj in enumerate(self.objects):
                pay[p0:p1, j] = obj.payoff(ps, full)
            t_pay += time.perf_counter() - t1
            t_sim += t1 - t0
            ls[p0:p1] = ps.log_spot[:, cols]
            fac[p0:p1] = ps.factors[:, cols, :]
            del ps
        self.timing = {"simulate": t_sim, "payoff": t_pay}
        # the columns a strip regression never reads: read-only zero views, no memory
        zero = np.broadcast_to(np.zeros(1), (n, times.size))
        self.paths = PathSet(times, ls, zero, fac, zero, zero)
        self._index = FixingIndex(times)
        self._owned_bytes = int(pay.nbytes + ls.nbytes + fac.nbytes)
        self.n_simulations = 1
        self.bump_shifts = {}
        self.bump_keys = ()
        self.bumped_paths = {}
        self.streamed = {}
        self.payoffs = [ObjectPayoffs(pay[:, j], pay[:, j], pay[:, j], {}) for j in range(n_obj)]
        # the spot bumps are never simulated: the delta / gamma targets the regression carries
        # along are zero (``p_up = p_dn = base``) and never read
        self._s_up = self.model.spot * float(np.exp(self.spot_size))
        self._s_dn = self.model.spot * float(np.exp(-self.spot_size))

    @property
    def idx(self) -> Any:
        assert self._index is not None
        return self._index

    @property
    def memory_bytes(self) -> int:
        """The kept payoff matrix and date columns (after :meth:`release`, what they were)."""
        return self.released_bytes if self.released else self._owned_bytes

    def release(self) -> int:
        """:meth:`ConditionalPricer.release` plus the payoffs and fits: a strip is read once,
        before the hedge loop, and nothing reads it afterwards."""
        n = super().release()
        self.payoffs = []
        self.fits = {}
        return n

    def slot(self, kdx: int) -> int:
        """The object block of rebalancing date ``kdx``."""
        if kdx not in self.date_slots:
            raise KeyError(f"the strip pricer does not carry rebalancing date {kdx}")
        return self.date_slots[kdx]


@dataclass
class StripSurfaces:
    """The recalibration rule's state surfaces per rebalancing-date index, precomputed before the
    hedge loop (:meth:`Hedger.strip_surfaces`): ``world[k]`` read from the world strip, ``twin[k]``
    from the pricing model's CRN twin (empty when not computed), both at ``n_paths`` strip
    paths; ``world_bytes`` / ``twin_bytes`` the resident footprint each strip pricer had before
    it was released, ``seconds`` the wall clock of the whole precomputation."""

    dates: FloatArray
    n_paths: int
    world: dict[int, _StateSurface] = field(default_factory=dict)
    twin: dict[int, _StateSurface] = field(default_factory=dict)
    world_bytes: int = 0
    twin_bytes: int = 0
    seconds: float = 0.0

    def excess_skew(self, kdx: int) -> FloatArray:
        """The world's skew at date ``kdx`` minus the pricing model's prediction of it."""
        return np.asarray(self.world[kdx].skew - self.twin[kdx].skew, dtype=np.float64)


def step0_degenerate_pillars(targets: TargetSet) -> tuple[float, ...]:
    """The fitted pillars whose step-0 SABR reduction is degenerate: the radicand guard fired
    (:attr:`~volsto.calibration.targets.SabrPillar.radicand_guarded`) or ``rho_SABR`` was
    clipped to ``±1`` (a clipped value is exactly ``±1``; a non-finite one — a negative radicand
    with the guard disabled — counts too)."""
    return tuple(
        float(s.T)
        for s in targets.sabr
        if s.radicand_guarded or not np.isfinite(s.rho_sabr) or abs(float(s.rho_sabr)) >= 1.0
    )


@dataclass(frozen=True)
class RefitTargets:
    """What :func:`refit_targets` hands the fit, with its provenance: ``step0_pillars`` (the
    degenerate pillars), ``step0_flags`` (the target set's flags as read), ``held`` (the policy's
    note), ``fallback_applied`` (the base fit's ``correl_target`` held for this date),
    ``corr_capped`` (the cap bit at some pillar), ``correl_read`` (``Corr_BE`` as read, before
    the policy, the fallback and the cap)."""

    targets: TargetSet
    step0_pillars: tuple[float, ...]
    step0_flags: tuple[str, ...]
    held: tuple[str, ...]
    fallback_applied: bool
    corr_capped: bool
    correl_read: FloatArray


def targets_beyond_cap(targets: TargetSet, cap: float) -> tuple[tuple[float, float], ...]:
    """``(T, Corr_BE)`` of the pillars whose break-even correlation target exceeds ``cap`` in
    magnitude: the pillars :func:`refit_targets` caps.  Applied to the base marking fit it tells
    whether a refit can reproduce the base fit at all (on the repaired eSSVI anchor of 2026-09-22
    the base targets sit beyond 0.97 at 1y, 2y and 3y: every refit caps them, the base fit does
    not — SPEC §8.2).  Checked by ``tests/test_hedging.py::test_targets_beyond_cap``."""
    if not 0.0 < cap <= 1.0:
        raise ValueError("cap must be in (0, 1]")
    ct = np.asarray(targets.correl_target, dtype=np.float64)
    return tuple((float(T), float(c)) for T, c in zip(targets.pillars, ct) if abs(float(c)) > cap)


def refit_targets(
    surface: Any, rule: RecalibrationRule, base_targets: TargetSet | None
) -> RefitTargets:
    """The marking targets a recalibration refits to (:class:`RecalibrationRule`, *The refit's
    correlation target*), in this order: the targets read on ``surface`` on the strip's stencil
    (:meth:`RecalibrationRule.marking_targets`); the policy's holding
    (:func:`~volsto.risk.shadow_rotation.held_targets`);
    the **guarded fallback** — when step 0 is degenerate at any fitted pillar
    (:func:`step0_degenerate_pillars`) the base fit's ``correl_target`` is held (a no-op in value
    under the sticky policies, which hold it already; recorded all the same); the **cap**
    ``|correl_target| <= rule.correlation_cap``.  Only ``correl_target`` is touched by the last two
    (measured on the named study-C state: holding it removes the pinning and leaves step 2 —
    ``λ``, ``k1``, the achieved skew and SpotVolCovar — unchanged).  ``base_targets`` is needed
    under a sticky policy and whenever the fallback fires (``ValueError`` otherwise; the pillars
    must match).  Checked by ``tests/test_hedging.py``."""
    targets = rule.marking_targets(surface)
    flags = tuple(targets.flags)
    read = np.array(targets.correl_target, dtype=np.float64)
    degenerate = step0_degenerate_pillars(targets)
    held: tuple[str, ...] = ()
    if rule.sticky:
        if base_targets is None:
            raise ValueError(
                f"recalibration policy {rule.policy!r} needs the base fit's targets (the hedger "
                "computes the base fit before the first rebalancing date)"
            )
        targets = held_targets(targets, base_targets, rule.policy)
        held = tuple(targets.flags[len(flags) :])
    fallback = bool(degenerate)
    if fallback:
        if base_targets is None:
            raise ValueError(
                f"step 0 is degenerate at {degenerate} and no base fit is available for the "
                "guarded correlation fallback"
            )
        b = base_targets
        if targets.pillars.shape != b.pillars.shape or not np.allclose(targets.pillars, b.pillars):
            raise ValueError("the guarded fallback needs the same fitted pillars on both surfaces")
        targets = dataclasses.replace(
            targets,
            correl_target=np.array(b.correl_target, dtype=np.float64),
            flags=(
                *targets.flags,
                f"guarded fallback: step 0 degenerate at T={list(degenerate)} (radicand guard or "
                "rho clip): Corr_BE held at the base fit's values for this date",
            ),
        )
    cap = float(rule.correlation_cap)
    ct = np.asarray(targets.correl_target, dtype=np.float64)
    capped = bool(np.any(np.abs(ct) > cap))
    if capped:
        targets = dataclasses.replace(
            targets,
            correl_target=np.clip(ct, -cap, cap),
            flags=(*targets.flags, f"|Corr_BE| capped at {cap:g}: {np.round(ct, 4).tolist()}"),
        )
    return RefitTargets(targets, degenerate, flags, held, fallback, capped, read)


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
# the minimum-variance delta
# --------------------------------------------------------------------------------------------

#: why a pricing model has no vol-correlation term (the note :func:`spot_factor_projection`
#: returns for a factor-free model)
COMPLETE_MARKET_NOTE = (
    "min_variance delta: the pricing model {name} has no stochastic-vol factor, the market is "
    "complete in the spot and the minimum-variance spot-only delta IS the model delta"
)


def spot_factor_projection(
    model: Model, t: float, log_spot: FloatArray, factors: FloatArray
) -> tuple[FloatArray, str | None]:
    """``d⟨X_i, S⟩_t / d⟨S, S⟩_t`` per path, ``(n_paths, n_factors)``, from the pricing model's own
    SDE, and a note (``None`` when the model has factors).

    **Source.**  :class:`~volsto.models.bergomi.BergomiSV` and :class:`~volsto.models.lsv.LSV`
    (SPEC §3.3, §3.5; ``bergomi_block``): the factors are unit-vol Ornstein–Uhlenbeck processes
    ``dX^i = −k_i X^i dt + dW^i`` and the spot ``d ln S = (…) dt + σ_t dW^S`` with
    ``σ_t² = L(t, S)² ξ_t^t`` (``L ≡ 1`` for the pure model) — the quantity
    :meth:`~volsto.models.base.Model.instantaneous_variance` returns and
    :attr:`PathSet.variance <volsto.engine.paths.PathSet>` records —, ``corr(dW^S, dW^i) = ρ_Si``
    (``kernel.rho_s``).  Hence ``d⟨X_i, S⟩ = ρ_Si S σ_t dt``, ``d⟨S, S⟩ = S² σ_t² dt`` and

        d⟨X_i, S⟩ / d⟨S, S⟩ = ρ_Si / (S_t σ_t).

    ``σ_t`` is the **pricing** model's at the given state (its own leverage and forward
    variance), so a world of another model is read through the desk's model.  A model without
    factors (Black–Scholes, local vol) returns an empty projection and says why
    (:data:`COMPLETE_MARKET_NOTE`); a model with factors whose spot-factor correlations the hedger
    cannot read raises (never a silent model delta)."""
    n = int(np.asarray(log_spot).shape[0])
    nf = int(getattr(model, "n_factors", 0))
    if nf == 0:
        return np.zeros((n, 0)), COMPLETE_MARKET_NOTE.format(name=type(model).__name__)
    kernel = model.kernel if isinstance(model, LSV) else model
    if not isinstance(kernel, BergomiSV):
        raise ValueError(
            f"min_variance delta: the pricing model {type(model).__name__} has {nf} factor(s) but "
            "no spot-factor correlation the hedger can read (Bergomi / LSV only)"
        )
    rho = np.asarray(kernel.rho_s, dtype=np.float64)
    fac = np.asarray(factors, dtype=np.float64).reshape(n, nf)
    ls = np.asarray(log_spot, dtype=np.float64)
    state = ModelState(float(t), ls, np.zeros(n), fac)
    var = np.asarray(model.instantaneous_variance(state), dtype=np.float64)
    denom = np.exp(ls) * np.sqrt(np.maximum(var, 0.0))
    safe = np.where(denom > 0.0, denom, 1.0)
    proj = np.where(denom[:, None] > 0.0, rho[None, :] / safe[:, None], 0.0)
    return np.asarray(proj, dtype=np.float64), None


def min_variance_delta(
    delta: FloatArray, value_gradients: FloatArray, projection: FloatArray
) -> FloatArray:
    """The minimum-variance spot-only delta (per unit spot, time-0 money)

        Δ_MV = Δ_model + Σ_i (∂V/∂X_i) d⟨X_i, S⟩ / d⟨S, S⟩,

    the ``Δ`` minimising the instantaneous variance of ``dV − Δ dS`` when ``V`` moves with the
    spot and the factors, ``dV = Δ_model dS + Σ_i ∂V/∂X_i dX_i + (…) dt``.  ``value_gradients``
    ``(n_paths, n_factors)`` are the regression's ``∂V/∂X_i``, ``projection`` the model's
    :func:`spot_factor_projection`.  Without factors (``n_factors = 0``) it returns ``delta``
    itself, exactly.  For a negative spot-vol correlation and a long-vega value the term is
    negative: the minimum-variance delta sits below the model delta (M8b study D, the 1y ATM
    call on the SPX marking fit: 17.6% below by the in-sample ``λ*`` probe)."""
    d = np.asarray(delta, dtype=np.float64)
    if projection.shape[1] == 0:
        return d
    return np.asarray(d + np.sum(value_gradients * projection, axis=1), dtype=np.float64)


# --------------------------------------------------------------------------------------------
# the hedger
# --------------------------------------------------------------------------------------------


@dataclass
class Hedger:
    """Module docstring.  ``pricing`` is a :class:`PricingContext` (or a bare model),
    ``world`` a model; ``sim`` the pricing configuration (paths, schedule, scheme), ``world_paths``
    the number of world paths (default ``sim.n_paths``); ``stream_bumps`` / ``scratch_dir`` and
    ``control_variate`` (the §7.11 control on the difference for the surface-driven bump targets of
    the objects with a Black–Scholes proxy, :mod:`volsto.hedging.controls`; its median variance
    reduction over dates and controlled objects is ``budget["cv_reduction_median"]``) and
    ``control_delta`` (the same control on the hybrid-CRN / regime delta target,
    :meth:`~volsto.hedging.pricing.ConditionalPricer.delta_control`; its median reduction is
    ``budget["cv_delta_reduction_median"]``, the product's per date in ``residual``'s
    ``cv_delta_reduction`` / ``cv_delta_reduction_se``) are passed to every
    :class:`~volsto.hedging.pricing.ConditionalPricer` the run builds (module docstring,
    *Memory*)."""

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
    control_variate: bool = True
    control_delta: bool = DEFAULT_CONTROL_DELTA

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

    #: paths of the projection probe (probed at this count and at twice it: the per-date cost
    #: is split into a fixed part and a part linear in the paths; small probes over-estimate the
    #: slope — numpy is less efficient per element at 10³ than at 4·10⁴ rows — so the probe is
    #: 2·10³ / 4·10³ paths, a few seconds per product)
    PROBE_PATHS = 2000

    def projected_wall_clock(
        self,
        product: Product,
        strategy: Strategy,
        grid: TimeGrid,
        n_models: int,
        bumps: Sequence[Bump] = (),
    ) -> float:
        """Projected seconds of :meth:`run`: the **simulation** cost from one
        :attr:`PROBE_PATHS`-path chunk of the pricing model on the grid (per path, times the
        ``n_models`` pricing sets and the world set) plus the **per-date** cost measured on a
        probe pricer of the same size — every object of the strategy fitted and evaluated at one
        mid-life date with the given ``bumps`` (the hybrid bump targets, the regressions and the
        state features, a Phoenix's knock-in scan included) — scaled to the pricing plus world
        path counts and the number of rebalancing dates.  ``bumps`` are the strategy's target
        bumps when the caller can build them without calibrating (the study runner passes them
        when every state is cached); without them the per-date cost is that of the value and
        spot targets only.  The M8 model (a constant per path, date and object) projected the
        3y autocall 3x and the Phoenix 8x short of the measured wall clock (study B, 2·10⁴
        paths, weekly); the probe measures what the loop does."""
        n_probe = self.PROBE_PATHS
        n_w = self.world_paths or self.sim.n_paths
        dates = self.schedule.build(product)
        n_dates = int(dates.size)
        objects: list[Product] = [product] + [
            i.product for i in strategy.instruments if i.product is not None
        ]
        kinds = ["value", "delta", *(b.name for b in bumps)]
        # two consecutive mid-life dates: the first carries the one-off costs of a pricer (the
        # payoffs of every object on every set, the control-variate proxies and shadow draws,
        # the first hybrid targets), the second is the marginal cost of a date in the loop;
        # probed at two path counts, the per-date cost splits into a fixed part (Python and
        # regression set-up per date and object) and a part linear in the paths
        i_mid = n_dates // 2
        t_first = float(dates[i_mid]) if n_dates else 0.0
        t_second = float(dates[min(i_mid + 1, n_dates - 1)]) if n_dates else 0.0
        per_path = 0.0
        one_off = 0.0
        per_date: dict[int, float] = {}
        for n in (n_probe, 2 * n_probe):
            probe = SimConfig(
                n_paths=n,
                chunk_size=n,
                seed=self.sim.seed,
                dt_max=self.sim.dt_max,
                antithetic=self.sim.antithetic,
            )
            draws = GaussianDraws(
                probe.seed, n, grid.n_steps, self.context.model.n_brownians, probe.antithetic
            )
            t0 = time.perf_counter()
            self.context.model.simulate_chunk(grid, draws, 0, n, probe.scheme)
            per_path = (time.perf_counter() - t0) / n
            t1 = time.perf_counter()
            pr = ConditionalPricer(
                self.context.model,
                objects,
                grid,
                probe,
                tuple(bumps),
                self.degree,
                control_variate=self.control_variate,
                control_delta=self.control_delta,
                surface=self.context.surface,
            )
            for j in range(len(objects)):
                pr.evaluate(j, t_first, pr.paths, kinds)
            t2 = time.perf_counter()
            if n_dates > 1:
                for j in range(len(objects)):
                    pr.evaluate(j, t_second, pr.paths, kinds)
            per_date[n] = time.perf_counter() - t2
            # the probe's own simulations (base, spot up / down, one set per bump) are the
            # per-path cost; what remains of the first date is the one-off cost of a pricer
            one_off = max(t2 - t1 - per_path * n * (3 + len(bumps)) - per_date[n], 0.0) / n
        sims = per_path * (self.sim.n_paths * n_models + n_w)
        n_loop = self.sim.n_paths + n_w
        slope = max(per_date[2 * n_probe] - per_date[n_probe], 0.0) / n_probe
        fixed = max(per_date[n_probe] - slope * n_probe, 0.0)
        regress = one_off * n_loop + n_dates * (fixed + slope * n_loop)
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
        # the minimum-variance delta: checked before anything is simulated (a model with factors
        # but no readable spot-factor correlation raises; a factor-free one says why the term
        # vanishes)
        if strategy.delta_regime == MIN_VARIANCE_REGIME:
            s0 = np.full(1, np.log(float(ctx.model.spot)))
            _, mv_note = spot_factor_projection(
                ctx.model, 0.0, s0, np.zeros((1, int(ctx.model.n_factors)))
            )
            if mv_note is not None and mv_note not in strategy.notes:
                strategy.notes.append(mv_note)
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
        rule = self.recalibration
        if rule is not None:
            projected += self.projected_strip_seconds(product, rule, dates)
        if self.verbose:
            strip = (
                "" if rule is None else f" + two recalibration strips of {rule.strip_paths} paths"
            )
            print(
                f"[hedger] {product!r} | strategy {strategy.name}: {dates.size} dates, "
                f"{len(objects)} priced objects, {n_models} pricing simulations of "
                f"{self.sim.n_paths} paths + {self.world_paths or self.sim.n_paths} world paths"
                f"{strip}; projected wall clock {projected:.0f} s",
                flush=True,
            )
        # the recalibration strips first, each released as soon as its surfaces are read: no
        # strip pricer ever coexists with the product's pricer or the world paths
        strips = None if rule is None else self.strip_surfaces(dates, rule)
        t0 = time.perf_counter()
        pricer = self._pricer(ctx.model, objects, grid, bumps, strategy)
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
                strips,
            )
        finally:
            for p in pricers:
                p.close()

    def _pricer(
        self,
        model: Model,
        objects: list[Product],
        grid: TimeGrid,
        bumps: Sequence[Bump],
        strategy: Strategy | None = None,
    ) -> ConditionalPricer:
        """The run's pricer; under the ``min_variance`` delta regime with the control variate
        on it also fits the controlled value target (``control_value``) the factor gradients are
        read from."""
        mv = strategy is not None and strategy.delta_regime == MIN_VARIANCE_REGIME
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
            control_delta=self.control_delta,
            control_value=bool(mv and self.control_variate),
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
        strips: StripSurfaces | None = None,
    ) -> HedgeResult:
        """The rebalancing loop of :meth:`run` (split so the scratch directories of every pricer
        built — the recalibration rebuilds included — are removed whatever happens); ``strips``
        the recalibration rule's precomputed state surfaces."""
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
        mv_delta = strategy.delta_regime == MIN_VARIANCE_REGIME and "delta" in strategy.target_names
        spot_inst = Spot()
        n_dates = dates.size
        n_refits = 0
        cache_hits = 0
        skew_ref: FloatArray | None = None
        rule = self.recalibration
        if rule is not None:
            if strips is None:
                strips = self.strip_surfaces(dates, rule)
            if (
                rule.base_fit is None
                and ctx.surface is not None
                and (rule.sticky or rule.refit is None)
            ):
                # the base marking fit the sticky policies hold their targets at, and the
                # guarded fallback's correlation under every policy: computed once, before the
                # first rebalancing date, from the pricing surface with the rule's config
                from volsto.calibration.fit_2f import fit_2f_marking

                t_fit = time.perf_counter()
                # read on the strip's stencil (RecalibrationRule.marking_targets): the held
                # targets and the fallback's correlation compare like for like with a refit's
                rule.base_fit = fit_2f_marking(
                    ctx.surface,
                    rule.config(),
                    ssr_target=rule.ssr_target,
                    h=rule.curvature_stencil,
                    skew_h=float(rule.h),
                )
                msg = (
                    f"recalibration policy {rule.policy}: base marking fit computed from the "
                    f"pricing surface at t = 0 on the strip's stencil (skew h = {rule.h:g}, "
                    f"curvature h = {rule.curvature_stencil:g}; {rule.base_fit.status}, "
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
            if rule.base_fit is not None:
                beyond = targets_beyond_cap(rule.base_fit.targets, rule.correlation_cap)
                if beyond:
                    note = (
                        f"recalibration policy {rule.policy}: the base marking fit's own "
                        f"correlation targets exceed the refit cap {rule.correlation_cap:g} at "
                        + ", ".join(f"T={T:g} ({c:+.4f})" for T, c in beyond)
                        + ": every refit caps what the base fit does not, so a refit on an "
                        "unmoved state already moves the parameters (SPEC §8.2)"
                    )
                    log.warning(note)
                    run_notes.append(note)

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

        def instrument_greeks(
            pr: ConditionalPricer, t: float, active: Sequence[bool]
        ) -> list[dict[str, FloatArray]]:
            """The instruments' Greeks at ``t`` under ``pr`` (an inactive instrument is zero)."""
            out: list[dict[str, FloatArray]] = []
            for j in range(n_i):
                oi = obj_index[j]
                if oi is None:
                    out.append(spot_greeks(t))
                elif active[j]:
                    out.append(values_at(pr, oi, t, kinds))
                else:
                    out.append({kk: np.zeros(n_w) for kk in kinds})
            return out

        def add_vanna(
            pr: ConditionalPricer,
            t: float,
            prod: dict[str, FloatArray],
            inst_g: list[dict[str, FloatArray]],
            active: Sequence[bool],
        ) -> None:
            """The vanna column of the product (or the net object) and of every instrument, from
            the gradient of the fitted vega polynomial in ``ln S``.  Called on every date **and
            again after a recalibration**, which rebuilds ``prod`` / ``inst_g`` under the new
            pricer: the M8b study-C VKO put runs (whose preset targets vanna) failed with
            ``KeyError: 'vanna'`` at the first refit date because the rebuilt dictionaries were
            handed to the solve without it."""
            if "vanna" not in strategy.target_names:
                return
            g_obj = 0 if net_index is None else net_index
            fit = pr.fit(g_obj, t)
            feats, _ = pr.features(g_obj, world, t)
            s_t = np.exp(world.log_spot_at(idx[t]))
            prod["vanna"] = fit.gradient("vega", feats, 0) / s_t
            for j in range(n_i):
                oi = obj_index[j]
                if oi is None or not active[j] or j in net_legs:
                    inst_g[j]["vanna"] = np.zeros(n_w)
                else:
                    f_ = pr.fit(oi, t)
                    ff, _ = pr.features(oi, world, t)
                    inst_g[j]["vanna"] = f_.gradient("vega", ff, 0) / s_t

        def factor_gradients(
            pr: ConditionalPricer, obj: int, t: float, t_next: float | None
        ) -> FloatArray:
            """``∂V/∂X_i`` of object ``obj`` at ``t`` on the world paths, ``(n_w, n_factors)``,
            from the gradient of the fitted value; zero on terminated paths.  A factor column the
            fit dropped as constant across the pricing paths (``t = 0``: every path at one
            state) borrows the alive-path mean of the next date's fitted gradient on the pricing
            paths — an ``O(Δt)`` approximation, noted."""
            nf = int(pr.model.n_factors)
            fit = pr.fit(obj, t)
            feats, hs = pr.features(obj, world, t)
            out = np.zeros((n_w, nf))
            if VALUE_CV not in fit.coefficients and nf:
                note = (
                    f"min_variance delta: no controlled value target for "
                    f"{type(pr.objects[obj]).__name__} (no Black-Scholes proxy, the control "
                    "variate off, or a forward start past T1): dV/dX from the raw value "
                    "regression (measured 2x noisier on the 1y ATM call)"
                )
                if note not in run_notes:
                    run_notes.append(note)
            for i in range(1, nf + 1):
                if i in fit.columns:
                    kind = VALUE_CV if VALUE_CV in fit.coefficients else "value"
                    out[:, i - 1] = fit.gradient(kind, feats, i)
                elif t_next is not None:
                    f2 = pr.fit(obj, t_next)
                    pf, phs = pr.features(obj, pr.paths, t_next)
                    kind = VALUE_CV if VALUE_CV in f2.coefficients else "value"
                    g = f2.gradient(kind, pf, i)[phs.alive]
                    out[:, i - 1] = float(g.mean()) if g.size else 0.0
                    note = (
                        f"min_variance delta at t = {t:g}: the factor state is constant across "
                        "the pricing paths, dV/dX borrowed from the next date's fit (alive-path "
                        "mean, O(dt))"
                    )
                    if note not in run_notes:
                        run_notes.append(note)
            if hs.any_terminated:
                out[~hs.alive] = 0.0
            return out

        def apply_min_variance(
            pr: ConditionalPricer,
            t: float,
            t_next: float | None,
            prod: dict[str, FloatArray],
            inst_g: list[dict[str, FloatArray]],
            active: Sequence[bool],
        ) -> None:
            """``delta`` of the product (or the net object) and of every priced active
            instrument replaced by its minimum-variance delta (:func:`min_variance_delta`); the
            spot's is its own (no factor exposure).  Called on every date and again after a
            recalibration (the rebuilt dictionaries, under the new pricing model)."""
            if not mv_delta:
                return
            col = idx[t]
            # a world without the pricing model's factors: the pricer's imputed state
            # E[X_t | ln S_t] (ConditionalPricer.pricing_factors), the one its fits are read at
            fac = pr.pricing_factors(world, col)
            proj, _ = spot_factor_projection(pr.model, t, world.log_spot_at(col), fac)
            if proj.shape[1] == 0:
                return
            g_obj = 0 if net_index is None else net_index
            prod["delta"] = min_variance_delta(
                prod["delta"], factor_gradients(pr, g_obj, t, t_next), proj
            )
            for j in range(n_i):
                oi = obj_index[j]
                if oi is None or not active[j] or j in net_legs:
                    continue
                inst_g[j]["delta"] = min_variance_delta(
                    inst_g[j]["delta"], factor_gradients(pr, oi, t, t_next), proj
                )

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
            add_vanna(cur_pricer, t, prod, inst_g, active)
            t_after = t_next if kdx + 1 < n_dates else None
            apply_min_variance(cur_pricer, t, t_after, prod, inst_g, active)
            # recalibration
            if rule is not None and strips is not None:
                skew_now = self._world_skew(strips, kdx, t, rule, world)
                if skew_ref is None:
                    skew_ref = skew_now
                moved = float(np.max(np.abs(skew_now - skew_ref)))
                did = False
                if moved > rule.skew_move_threshold and n_refits < rule.max_refits:
                    new_ctx, hit = self._recalibrate(t, strips, kdx, rule, world)
                    if new_ctx is not None:
                        n_refits += 1
                        cache_hits += int(hit)
                        # the replaced pricer is read once more (V_k(old) is in ``prod``
                        # already) and released: its path sets would otherwise stay resident
                        # for the rest of the run, one full pricer per refit
                        old_pricer = cur_pricer
                        new_pricer = self._pricer(new_ctx.model, objects, grid, bumps, strategy)
                        pricers.append(new_pricer)
                        new_prod, _ = new_pricer.evaluate(0, t, world, ["value"])
                        d_recal = new_prod["value"] - prod["value"]
                        pnl_recal += np.where(alive, d_recal, 0.0)
                        cur_pricer = new_pricer
                        old_pricer.release()
                        prod, hs = cur_pricer.evaluate(0, t, world, kinds)
                        inst_g = instrument_greeks(cur_pricer, t, active)
                        prod = netted(cur_pricer, t, prod, inst_g)
                        add_vanna(cur_pricer, t, prod, inst_g, active)
                        apply_min_variance(cur_pricer, t, t_after, prod, inst_g, active)
                        skew_ref = skew_now
                        did = True
                # the refit's own log row carries the flags (absent when the rule's refit hook
                # is stubbed, as the loop tests do)
                last = rule.log_rows[-1] if did and rule.log_rows else {}
                recal_rows.append(
                    {
                        "t": t,
                        "skew_move": moved,
                        "threshold": rule.skew_move_threshold,
                        "policy": rule.policy,
                        "recalibrated": did,
                        "params": None if not did else repr(cur_pricer.model),
                        "at_bound": str(last.get("at_bound", "")),
                        "step0_flags": str(last.get("step0_flags", "")),
                        "fallback_applied": bool(last.get("fallback_applied", False)),
                        "corr_capped": bool(last.get("corr_capped", False)),
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
            # the delta control's variance reduction on the product's (net object's) delta target
            fit_g = cur_pricer.fit(0 if net_index is None else net_index, t)
            if "delta" in fit_g.controlled:
                rrow["cv_delta_reduction"] = fit_g.variance_reduction["delta"]
                rrow["cv_delta_reduction_se"] = fit_g.variance_reduction_se["delta"]
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
        if strips is not None:
            budget["strip_paths"] = float(strips.n_paths)
            budget["strip_seconds"] = float(strips.seconds)
            budget["strip_world_gb"] = float(strips.world_bytes / 1e9)
            budget["strip_twin_gb"] = float(strips.twin_bytes / 1e9)
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
        # the delta control's (NaN when nothing was controlled: control off, no proxy)
        d_red = [r for p in pricers for r in p.cv_reductions(delta=True)]
        d_ses = [r for p in pricers for r in p.cv_reduction_ses(delta=True)]
        budget["cv_delta_reduction_median"] = float(np.median(d_red)) if d_red else float("nan")
        budget["cv_delta_reduction_q25"] = (
            float(np.quantile(d_red, 0.25)) if d_red else float("nan")
        )
        budget["cv_delta_reduction_q75"] = (
            float(np.quantile(d_red, 0.75)) if d_red else float("nan")
        )
        budget["cv_delta_reduction_se_median"] = float(np.median(d_ses)) if d_ses else float("nan")
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
                "strip_paths": None if rule is None else int(rule.strip_paths),
                "correlation_cap": None if rule is None else float(rule.correlation_cap),
                "stream_bumps": self.stream_bumps,
                "scratch_dir": None if self.scratch_dir is None else str(self.scratch_dir),
                "control_variate": self.control_variate,
                "control_delta": self.control_delta,
                # the preset's keyword arguments (run(product, q=0.5)), or those the strategy
                # was built with by default_strategy when it is passed in
                "preset_kwargs": dict(preset_kwargs)
                or dict(getattr(strategy, "preset_kwargs", None) or {}),
            },
            {k_: np.array(v) for k_, v in greeks_by_date.items()},
            {k_: np.array(v) for k_, v in moves.items() if v},
        )

    # -- recalibration internals -------------------------------------------------------------

    #: strip objects per pillar of the three-strike strip ``k ∈ {−h, 0, +h}``
    #: (:meth:`RecalibrationRule.strip_log_moneyness` carries two more when ``curvature_h != h``)
    STRIP_STRIKES = 3

    def _strip_sim(self, rule: RecalibrationRule) -> SimConfig:
        """The world simulation settings (:meth:`_world_sim`) at ``rule.strip_paths`` paths —
        identical to the world's when the counts are equal."""
        wsim = self._world_sim()
        n = int(rule.strip_paths)
        if n % 2 and wsim.antithetic:
            raise ValueError("strip_paths must be even with antithetic draws")
        return dataclasses.replace(wsim, n_paths=n, chunk_size=min(self.sim.chunk_size, n))

    def _strip_objects(self, dates: FloatArray, rule: RecalibrationRule) -> list[Product]:
        """The forward-start strip of every date (module docstring of the rule): per date ``t``,
        pillar ``τ`` and forward log-moneyness ``k`` in
        :meth:`RecalibrationRule.strip_log_moneyness` the option
        ``(S_{t+τ}/S_t − m)^±`` struck at ``m = F(t+τ)/F(t) · e^{k}`` (a put below the money, a
        call at and above), in that order.  The payoff's strike is relative to the SPOT at ``t``;
        the forward ratio puts the strike at forward log-moneyness exactly ``k``.  (Before
        2026-09-16 it was ``m = e^{k}`` — forward log-moneyness ``k − ln(F(t+τ)/F(t))`` — while
        the reads were labelled at ``k``: at ``t = 0`` on the study-C pricing twin the "ATMF" vol
        read 0.2225 / 0.2364 / 0.2373 at 3M / 1Y / 3Y, the snapshot's vol at the spot strike
        0.2230 / 0.2368 / 0.2380 and its ATMF vol 0.2200 / 0.2275 / 0.2218.)"""
        from volsto.products.forward_start import ForwardStartOption

        fc = self.world.forward_curve
        disc = fc.rate_curve
        ks = rule.strip_log_moneyness()
        objs: list[Product] = []
        for t in dates:
            for tau in rule.pillars:
                f_ratio = float(fc.forward(float(t) + tau) / fc.forward(float(t)))
                for k in ks:
                    objs.append(
                        ForwardStartOption(
                            float(t),
                            float(t) + tau,
                            f_ratio * float(np.exp(k)),
                            1 if k >= 0 else -1,
                            disc,
                        )
                    )
        return objs

    def _world_skew_pricer(
        self, dates: FloatArray, rule: RecalibrationRule, only: Sequence[int] | None = None
    ) -> _StripPricer:
        """The skew strip under the **world** model at ``rule.strip_paths`` paths: at every
        rebalancing date ``t`` and pillar ``τ`` the forward-start options of
        :meth:`_strip_objects`, ``(S_{t+τ}/S_t − F(t+τ)/F(t) e^{k})^±`` — struck at the *forward
        moneyness of each path* (a vanilla struck at ``F(T) e^{k}`` would sit anywhere from far
        below to far above a path's own forward at ``t``; inverting it at the path's forward
        mixes moneyness across paths).  Their conditional values at ``t`` are homogeneous in
        ``S_t`` and invert path by path to the world's conditional smile at ``t``
        (:meth:`_state_surface`).  No bumps, no control variate, a lean pricer
        (:class:`_StripPricer`).  ``only`` restricts the priced objects to those dates' (the
        grid — hence every path — is the full strip's either way)."""
        objs = self._strip_objects(dates, rule)
        g = union_grid([self.world], objs, dates, self.sim)
        per = len(rule.pillars) * len(rule.strip_log_moneyness())
        sel = list(range(dates.size)) if only is None else sorted({int(k) for k in only})
        if any(k < 0 or k >= dates.size for k in sel):
            raise ValueError("only: rebalancing-date indices out of range")
        kept = [o for k in sel for o in objs[k * per : (k + 1) * per]]
        wsim = self._strip_sim(rule)
        return _StripPricer(
            self.world,
            kept,
            g,
            wsim,
            (),
            self.degree,
            seed=wsim.seed + 7,
            delta_estimator="gradient",
            hybrid_bumps=False,
            control_variate=False,
            keep_times=tuple(float(dates[k]) for k in sel),
            date_slots={k: i for i, k in enumerate(sel)},
        )

    def _twin_skew_pricer(self, strip: _StripPricer) -> _StripPricer:
        """The same strip under the **pricing** model on the same grid, path count and seed (a
        CRN twin): the pricing model's own prediction of the conditional smile at every date, so
        the recalibration trigger reads the world's skew *in excess* of it
        (:meth:`_world_skew`).  ``strip`` may already be released (its objects, grid and
        settings are what is read)."""
        return _StripPricer(
            self.context.model,
            list(strip.objects),
            strip.grid,
            strip.sim,
            (),
            self.degree,
            seed=strip.seed,
            delta_estimator="gradient",
            hybrid_bumps=False,
            control_variate=False,
            keep_times=strip.keep_times,
            date_slots=dict(strip.date_slots),
        )

    def strip_surfaces(
        self,
        dates: FloatArray,
        rule: RecalibrationRule,
        *,
        only: Sequence[int] | None = None,
        twin: bool = True,
    ) -> StripSurfaces:
        """Every date's state surface (:class:`StripSurfaces`), one strip at a time: the world
        strip is built, read at every date (``only``: those date indices) and released, then the
        pricing model's twin (unless ``twin=False``) the same way.  The strips' only output is a
        few numbers per date (the ATMF vol, skew and curvature per pillar), so the hedge loop
        never holds a strip pricer; the peak is one strip's payoff matrix plus one simulation
        chunk (``StripSurfaces.world_bytes``)."""
        t0 = time.perf_counter()
        sel = list(range(dates.size)) if only is None else sorted({int(k) for k in only})
        out = StripSurfaces(np.asarray(dates, dtype=np.float64), int(rule.strip_paths))
        pr = self._world_skew_pricer(dates, rule, sel)
        out.world = {k: self._state_surface(pr, k, float(dates[k]), rule) for k in sel}
        out.world_bytes = pr.release()
        if twin:
            tw = self._twin_skew_pricer(pr)
            out.twin = {k: self._state_surface(tw, k, float(dates[k]), rule) for k in sel}
            out.twin_bytes = tw.release()
            del tw
        del pr
        out.seconds = time.perf_counter() - t0
        if self.verbose:
            print(
                f"[hedger] recalibration strips: {len(sel)} dates x {len(rule.pillars)} pillars "
                f"at {out.n_paths} paths (world{' + twin' if twin else ''}) in "
                f"{out.seconds:.0f} s; resident before release {out.world_bytes / 1e9:.2f} + "
                f"{out.twin_bytes / 1e9:.2f} GB",
                flush=True,
            )
        return out

    def projected_strip_seconds(
        self, product: Product, rule: RecalibrationRule, dates: FloatArray | None = None
    ) -> float:
        """Projected seconds of :meth:`strip_surfaces` for ``product`` at ``rule.strip_paths``:
        per strip model (the world, then the pricing model for the twin) the simulation and
        payoff cost measured on a lean probe strip carrying two mid-life dates at
        :attr:`PROBE_PATHS` and twice that (the full strip's grid), the per-date read (one
        regression per pillar and strike and per-path inversions) split into a fixed part and a
        part linear in the paths, scaled to the strip's path count, object count and dates."""
        dates = self.schedule.build(product) if dates is None else np.asarray(dates)
        n_dates = int(dates.size)
        if n_dates == 0:
            return 0.0
        per = len(rule.pillars) * len(rule.strip_log_moneyness())
        n_obj = n_dates * per
        n_strip = int(rule.strip_paths)
        i_mid = n_dates // 2
        sel = sorted({i_mid, min(i_mid + 1, n_dates - 1)})
        total = 0.0
        for model_is_world in (True, False):
            read: dict[int, float] = {}
            sim_pp = pay_pp = 0.0
            for n in (self.PROBE_PATHS, 2 * self.PROBE_PATHS):
                probe_rule = dataclasses.replace(rule, strip_paths=n, log_rows=[])
                pr = self._world_skew_pricer(dates, probe_rule, sel)
                if not model_is_world:
                    pr = self._twin_skew_pricer(pr)
                sim_pp = pr.timing["simulate"] / n
                pay_pp = pr.timing["payoff"] / (n * len(pr.objects))
                t0 = time.perf_counter()
                self._state_surface(pr, sel[-1], float(dates[sel[-1]]), probe_rule)
                read[n] = time.perf_counter() - t0
                pr.release()
            n1, n2 = self.PROBE_PATHS, 2 * self.PROBE_PATHS
            slope = max(read[n2] - read[n1], 0.0) / n1
            fixed = max(read[n1] - slope * n1, 0.0)
            total += sim_pp * n_strip + pay_pp * n_strip * n_obj
            total += n_dates * (fixed + slope * n_strip)
        return float(total)

    def _state_surface(
        self,
        pr: ConditionalPricer,
        kdx: int,
        t: float,
        rule: RecalibrationRule,
        world: PathSet | None = None,
    ) -> _StateSurface:
        """The conditional smile at ``t`` read **per path** from the strip of
        :meth:`_world_skew_pricer` (or its twin) and averaged: each forward-start's regressed
        conditional value (time-0 money) on the strip's own paths is taken to time-``t`` money,
        inverted on the ratio (forward ``F(t + τ)/F(t)``, strike ``F(t + τ)/F(t) e^{k}`` — the
        strike the object carries, forward log-moneyness ``k`` —, maturity ``τ``, discount
        ``DF(t + τ)/DF(t)``) and the ATMF vol (``k = 0``), the central-difference skew across
        ``±h`` and the second-difference curvature across ``{−c, 0, +c}`` (``c`` the rule's
        :attr:`~RecalibrationRule.curvature_stencil`) are averaged over the paths where every
        strike's inversion is finite (a regressed value below intrinsic inverts to NaN and is
        dropped; a kept fraction under :data:`STATE_SURFACE_MIN_INVERTED` is logged).  The M8
        reading inverted the path-*averaged* price — the unconditional ``(t + τ)``-option, whose
        "vol over τ" carries the spot variance over ``[0, t]`` (measured: the 3M ATMF vol read
        0.218 at ``t = 0`` and 0.394 at ``t = 0.5`` on a world without any shock, the factor
        ``sqrt((t + τ)/τ)``), so the proxy moved at every date and the rule refit at every date.
        The curvature is the noisy part of this read (:data:`DEFAULT_STRIP_PATHS`).  ``pr`` is a
        :class:`_StripPricer` (its date block by :meth:`_StripPricer.slot`) or a full pricer over
        every date's strip in order; ``world`` is unused (kept for the hook's signature)."""
        from volsto.market.bs import implied_vol

        fc = self.world.forward_curve
        n_pill = len(rule.pillars)
        slot = pr.slot(kdx) if isinstance(pr, _StripPricer) else kdx
        df_t = float(fc.rate_curve.df(t))
        h = float(rule.h)
        ch = rule.curvature_stencil
        ks = rule.strip_log_moneyness()
        n_k = len(ks)
        atf, skew, curv = [], [], []
        for pi, tau in enumerate(rule.pillars):
            T = t + tau
            base_obj = slot * n_pill * n_k + pi * n_k
            f_ratio = float(fc.forward(T) / fc.forward(t))
            df = float(fc.rate_curve.df(T)) / df_t
            ivs = []
            for si, k in enumerate(ks):
                strike = f_ratio * float(np.exp(k))
                carried = float(getattr(pr.objects[base_obj + si], "strike", strike))
                if abs(carried / strike - 1.0) > 1e-12:
                    raise ValueError(
                        f"strip object {base_obj + si} is struck at {carried:.6g}, not at forward "
                        f"log-moneyness {k:+g} ({strike:.6g})"
                    )
                out, _ = pr.evaluate(base_obj + si, t, pr.paths, ["value"])
                iv = implied_vol(out["value"] / df_t, f_ratio, strike, tau, 1 if k >= 0 else -1, df)
                ivs.append(np.asarray(iv, dtype=np.float64))
            lo, mid, hi = ivs[:3]
            c_lo, c_hi = (lo, hi) if n_k == 3 else (ivs[3], ivs[4])
            ok = np.logical_and.reduce([np.isfinite(v) for v in ivs])
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
            curv.append(float(((c_hi - 2.0 * mid + c_lo)[ok] / (ch * ch)).mean()))
        return _StateSurface(
            np.asarray(rule.pillars, dtype=np.float64),
            np.asarray(atf),
            np.asarray(skew),
            np.asarray(curv),
            fc,
        )

    def _world_skew(
        self, strips: StripSurfaces, kdx: int, t: float, rule: RecalibrationRule, world: PathSet
    ) -> FloatArray:
        """The world's conditional skew at ``t`` **in excess of the pricing model's own
        prediction** at the same date: the CRN twin strip (:meth:`_twin_skew_pricer`) is read
        by :meth:`_state_surface` the same way (both precomputed, :meth:`strip_surfaces`) and the
        rule triggers on the *change* of the excess since the last refit (the reference is the
        excess at ``t = 0``, reset at each refit).  An LSV world's forward skew at date ``t``
        differs from its spot skew by far more than the threshold with no shock at all (measured
        on the SPX marking fit: the M8 rule, which compared the world's skew with its own
        ``t = 0`` value, refit at every one of 11 monthly dates), whereas the excess is exactly 0
        under world = pricing (identical simulations), the shock's rota shows up in it after the
        shock, and the held rotation triggers no further refit once the reference has reset."""
        return strips.excess_skew(kdx)

    def _recalibrate(
        self, t: float, strips: StripSurfaces, kdx: int, rule: RecalibrationRule, world: PathSet
    ) -> tuple[PricingContext | None, bool]:
        """One refit at date ``t`` on the world's state surface of that date (the one the trigger
        read): the targets of :func:`refit_targets` — the policy's holding, the **guarded
        fallback** (step 0's radicand guard is right for a genuinely negative ATM curvature but
        must not fire on Monte Carlo noise; the strip path count and the correlation cap are what
        protect the refit from that, the fallback to the base fit's correlation is the last belt)
        and the cap — fitted by :func:`~volsto.calibration.fit_2f.fit_2f`, the pricing model
        rebuilt through the cache.  Logged in ``rule.log_rows`` with the step-0 flags,
        ``fallback_applied``, ``corr_capped`` and a fitted correlation at its bound (warned)."""
        ctx = self.context
        if ctx.state is None or ctx.builder is None:
            ctx.notes.append("recalibration rule needs an LSV pricing state: skipped")
            return None, False
        surf = strips.world[kdx]
        from volsto.calibration.fit_2f import fit_2f
        from volsto.market.varswap import xi0_curve

        cfg = rule.config()
        prov: RefitTargets | None = None
        if rule.refit is not None:
            params = rule.refit(surf, ctx.state.spec.model)
        else:
            base = None if rule.base_fit is None else rule.base_fit.targets
            prov = refit_targets(surf, rule, base)
            targets = prov.targets
            xi0 = xi0_curve(
                ctx.surface, float(min(ctx.surface.max_maturity, max(targets.pillars) + t))
            )
            params = fit_2f(targets, xi0, cfg).params
        changes = {f.name: float(getattr(params, f.name)) for f in dataclasses.fields(params)}
        at_bound = degenerate_correlations(changes)
        step0 = () if prov is None else prov.step0_flags
        guards = (
            ""
            if prov is None
            else (
                f"; guarded fallback {'applied' if prov.fallback_applied else 'not needed'}, "
                f"|Corr_BE| cap {'applied' if prov.corr_capped else 'not binding'}"
            )
        )
        if at_bound:
            msg = (
                f"recalibration at t={t:g} ({rule.policy}): the refit lands with "
                + ", ".join(f"{k} = {v:+.4f}" for k, v in at_bound.items())
                + f" (|rho| >= {CORRELATION_BOUND}) — a degenerate two-factor set, so the "
                "repricing it books is not a measurement of the desk's re-marking. "
                "Step-0 flags of the state surface read at this date: "
                + ("; ".join(step0) if step0 else "none")
                + guards
                + ". (Measured mechanism on the 2026-09-15 study-C runs: an unconverged strip "
                "curvature fires step 0's radicand guard, Corr_SABR clips to -1 and the collapsed "
                "set is step 3's exact minimiser; the skew target is attainable.)"
            )
            log.warning(msg)
            if msg not in ctx.notes:
                ctx.notes.append(msg)
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
                "held": "" if prov is None else "; ".join(prov.held),
                "at_bound": "; ".join(f"{k}={v:+.4f}" for k, v in at_bound.items()),
                "step0_flags": "; ".join(step0),
                "step0_pillars": "" if prov is None else repr(list(prov.step0_pillars)),
                "fallback_applied": False if prov is None else prov.fallback_applied,
                "corr_capped": False if prov is None else prov.corr_capped,
                "correl_read": "" if prov is None else repr(np.round(prov.correl_read, 6).tolist()),
                "correl_target": (
                    "" if prov is None else repr(np.round(prov.targets.correl_target, 6).tolist())
                ),
                "curv": repr(np.round(surf.curv, 6).tolist()),
            }
        )
        return PricingContext(model, new_state, ctx.builder, ctx.surface, ctx.label), hit


#: a fitted correlation at least this large in absolute value is a degenerate two-factor set
#: (perfectly correlated factors): the M7 marking work guards ``rho12`` for the same reason
#: (§15 Part 3), and 41 of the 113 ``sabr_linked`` refits of the M8b study-C runs landed here
#: while none of the 113 ``sticky_breakeven`` refits did — measured cause: step 0's radicand
#: guard firing on an unconverged state-surface curvature (§8.2)
CORRELATION_BOUND = 0.99


def degenerate_correlations(changes: Mapping[str, float]) -> dict[str, float]:
    """The fitted correlations at their bound (``|rho| >= CORRELATION_BOUND``), empty when the
    set is non-degenerate.  A refit that lands here prices the product under perfectly correlated
    factors, so the recalibration P&L it books says more about the policy's attainability than
    about the re-marking; :meth:`Hedger._recalibrate` warns and records it per date
    (``recalibrations["at_bound"]``)."""
    return {
        k: float(v)
        for k, v in changes.items()
        if k.startswith("rho") and abs(float(v)) >= CORRELATION_BOUND
    }


def _with_params(state: RiskState, changes: Mapping[str, float]) -> RiskState:
    """``state`` with the model parameters ``changes`` applied (a typed ``with_params``)."""
    model = state.spec.model.replace(**{k: float(v) for k, v in changes.items()})
    label = ",".join(f"{k}={v:g}" for k, v in changes.items())
    return RiskState(dataclasses.replace(state.spec, model=model), state.x0, label)


def realised_spot(world: PathSet, idx: Any, T: float, fc: Any) -> FloatArray:
    """The spot instrument's realised value at ``T`` in time-0 money."""
    return np.asarray(np.exp(world.log_spot_at(idx[T])) * float(fc.spot / fc.forward(T)))


__all__ = [
    "COMPLETE_MARKET_NOTE",
    "CORRELATION_BOUND",
    "DEFAULT_STRIP_PATHS",
    "FREQUENCIES",
    "REFIT_CORRELATION_CAP",
    "SKEW_MOVE_THRESHOLD",
    "SPOT_BUMP",
    "TENT_SIZE",
    "VOL_BUMP",
    "Costs",
    "HedgeResult",
    "Hedger",
    "PricingContext",
    "RecalibrationRule",
    "RefitTargets",
    "Schedule",
    "StripSurfaces",
    "degenerate_correlations",
    "min_variance_delta",
    "realised_spot",
    "refit_targets",
    "spot_factor_projection",
    "step0_degenerate_pillars",
    "targets_beyond_cap",
]
