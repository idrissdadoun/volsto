"""Risk engine (SPEC v2 §7.1): pricing states, model builders, CRN sensitivities.

* :class:`RiskState` is a :class:`~volsto.config.CalibrationSpec` (market, surface and its
  additive perturbation layer, model parameters, particle and simulation settings) plus an
  initial factor state.  Every state that changes the surface or a model parameter is a
  distinct leverage-cache entry, so the second run of a ladder is free.
* :class:`ModelBuilder` turns a state into a priced model under one of three *modes*:
  ``"recalibrate"`` (leverage refit to the state's surface through
  :meth:`~volsto.calibration.cache.LeverageCache.get_or_calibrate`), ``"sticky_leverage"``
  (the base leverage held, ``ξ₀`` and parameters from the state) and ``"model"`` (spot and factor
  state bumped only, leverage held fixed in spot, factors as given).  :class:`LSVBuilder`,
  :class:`LVBuilder` and :class:`BSBuilder` cover the three model families; the LV and BS
  builders accept every mode so the same tests run on all of them.
* :class:`RiskEngine` prices ``(product, state, mode)`` under common random numbers (same seed,
  same grid) keeping the per-path payoffs, and returns :class:`Sensitivity` objects whose
  standard error is that of the *difference* estimated path by path.  Surface bumps that fail
  the no-arbitrage checks are halved and retried (``max_halvings``); the achieved size is
  reported and the sensitivity is per unit of the *requested* size definition.  The builder
  counts distinct recalibrations and cache misses and the engine keeps the wall clock: that is
  the viewer precompute budget of §7.13.

Default bump sizes (§7.1): delta/gamma 1% of spot in log space (central three-point), vega
1 vol point, forward-variance buckets +1 vol point of the bucket's forward VS vol, model
parameters 5% relative (ν, k1, k2) and 0.05 absolute (θ, correlations); all configurable.
Checked by ``tests/test_risk_greeks.py`` and the ladder / estimator tests.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from volsto.calibration.cache import LeverageCache, build_market, spec_key
from volsto.config import (
    BergomiParams,
    CalibrationSpec,
    CurveConfig,
    MarketConfig,
    SimConfig,
    SurfacePerturbation,
)
from volsto.engine.mc import MonteCarlo, PriceResult
from volsto.market.curves import ForwardCurve
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import ArbitrageError, ImpliedSurface, SSVISurface, perturbed_surface
from volsto.models.base import Model
from volsto.models.bergomi import BergomiSV
from volsto.models.bs import BlackScholes
from volsto.models.localvol import LocalVol
from volsto.models.lsv import LSV
from volsto.products.base import Product

FloatArray = NDArray[np.float64]
log = logging.getLogger(__name__)

MODES = ("recalibrate", "sticky_leverage", "model")


# --------------------------------------------------------------------------------------------
# states
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RiskState:
    """A pricing state: calibration spec (market, surface + perturbation, model parameters) and
    the initial factor state ``x0`` (zero unless bumped)."""

    spec: CalibrationSpec
    x0: tuple[float, ...] | None = None
    label: str = "base"

    @property
    def spot(self) -> float:
        return self.spec.market.spot

    @property
    def key(self) -> str:
        """Leverage-cache key of the state (the factor state is not part of a calibration)."""
        return spec_key(self.spec)

    def with_spot(self, spot: float, label: str | None = None) -> RiskState:
        market = dataclasses.replace(self.spec.market, spot=float(spot))
        return RiskState(
            dataclasses.replace(self.spec, market=market), self.x0, label or f"spot={spot:g}"
        )

    def with_perturbation(
        self, pert: SurfacePerturbation | None, *, replace: bool = False, label: str | None = None
    ) -> RiskState:
        """Add a perturbation layer (composed with the existing one unless ``replace``)."""
        current = self.spec.perturbation
        if pert is None or replace or current is None:
            new = pert
        else:
            items_a = (
                current.params["items"] if current.kind == "composite" else [_as_mapping(current)]
            )
            items_b = pert.params["items"] if pert.kind == "composite" else [_as_mapping(pert)]
            new = SurfacePerturbation("composite", {"items": [*items_a, *items_b]})
        return RiskState(
            dataclasses.replace(self.spec, perturbation=new),
            self.x0,
            label or (pert.kind if pert is not None else "unperturbed"),
        )

    def with_params(self, label: str | None = None, **changes: float) -> RiskState:
        model = self.spec.model.replace(**changes)
        return RiskState(
            dataclasses.replace(self.spec, model=model),
            self.x0,
            label or ",".join(f"{k}={v:g}" for k, v in changes.items()),
        )

    def with_rate_shift(
        self, dr: float = 0.0, dq: float = 0.0, label: str | None = None
    ) -> RiskState:
        m = self.spec.market
        rc = CurveConfig(m.rate_curve.times, tuple(r + dr for r in m.rate_curve.rates))
        qc = CurveConfig(m.dividend_curve.times, tuple(q + dq for q in m.dividend_curve.rates))
        market = MarketConfig(m.spot, rc, qc)
        return RiskState(
            dataclasses.replace(self.spec, market=market),
            self.x0,
            label or f"dr={dr:g},dq={dq:g}",
        )

    def with_zero_rates(self) -> RiskState:
        m = self.spec.market
        market = MarketConfig(
            m.spot,
            CurveConfig(m.rate_curve.times, tuple(0.0 for _ in m.rate_curve.rates)),
            CurveConfig(m.dividend_curve.times, tuple(0.0 for _ in m.dividend_curve.rates)),
        )
        return RiskState(dataclasses.replace(self.spec, market=market), self.x0, "zero rates")

    def with_x0(self, x0: Sequence[float], label: str | None = None) -> RiskState:
        return RiskState(self.spec, tuple(float(x) for x in x0), label or f"x0={tuple(x0)}")

    def with_particles(self, n: int) -> RiskState:
        particle = dataclasses.replace(self.spec.particle, n_particles=int(n))
        return RiskState(dataclasses.replace(self.spec, particle=particle), self.x0, self.label)

    def __repr__(self) -> str:
        return f"RiskState({self.label}, spot={self.spot:g}, key={self.key[:8]})"


def _as_mapping(p: SurfacePerturbation) -> dict[str, Any]:
    return {"kind": p.kind, "params": dict(p.params)}


def surface_of(state: RiskState) -> ImpliedSurface:
    """The (perturbed) SSVI surface of a state, arbitrage-checked."""
    fc = ForwardCurve.from_config(state.spec.market)
    base = SSVISurface.from_config(state.spec.surface, fc, fc.rate_curve)
    return perturbed_surface(state.spec.perturbation, base)


# --------------------------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------------------------


class ModelBuilder(Protocol):
    """Turns a state into a priced model under a mode; counts recalibrations."""

    def build(self, state: RiskState, mode: str) -> Model: ...

    @property
    def n_calibrations(self) -> int: ...

    @property
    def n_cache_misses(self) -> int: ...

    @property
    def cache_keys(self) -> list[str]: ...


class _Counter:
    def __init__(self) -> None:
        self.keys: list[str] = []
        self.misses = 0

    def add(self, key: str, miss: bool) -> None:
        if key not in self.keys:
            self.keys.append(key)
            if miss:
                self.misses += 1


def model_regime_spot_bump(model: Model, spot: float) -> Model:
    """The ``"model"``-regime spot bump of any model: the spot moved with the model's own
    dynamics held.  LSV: ``LSV.bump(spot=)`` re-anchors the leverage so ``L(t, S)`` stays fixed in
    spot; Black–Scholes: the vol unchanged; **pure local vol**: ``σ_loc(t, S)`` held in absolute
    spot through :meth:`~volsto.market.dupire.LocalVolSurface.reanchored` — *not*
    ``LocalVol.bump(spot=)``, which keeps the local vol indexed in ``k`` (the sticky-local-vol
    move; measured on the placeholder surface it left the hedger's LV "model" delta 21% below the
    LV model's own delta at ``t = 0``).  Used by the hedging regression engine and the §7.11
    conditional-Greeks estimator for their CRN spot bumps."""
    if isinstance(model, LocalVol):
        fc = model.forward_curve.with_spot(float(spot))
        return LocalVol(model.local_vol.reanchored(fc), fc)
    return model.bump(spot=float(spot))


class LSVBuilder:
    """LSV models from states through the leverage cache (SPEC v2 §7.1); with
    ``allow_calibrate=False`` a missing entry raises :class:`~volsto.calibration.cache.
    CacheMissError` (tests and viewers never calibrate)."""

    def __init__(
        self, cache: LeverageCache, base: RiskState, *, allow_calibrate: bool = True
    ) -> None:
        self.cache = cache
        self.base = base
        self.allow_calibrate = bool(allow_calibrate)
        self._counter = _Counter()
        self._models: dict[tuple[str, str, tuple[float, ...] | None], Model] = {}
        self.base_model = self._recalibrated(base)

    def _recalibrated(self, state: RiskState) -> LSV:
        miss = not self.cache.has(state.spec)
        model, _ = self.cache.get_or_calibrate(state.spec, allow_calibrate=self.allow_calibrate)
        self._counter.add(state.key, miss)
        return model

    def _reference(self, state: RiskState) -> LSV:
        """The recalibrated model of ``state``'s surface and parameters at the base spot."""
        if state.spot == self.base.spot:
            return self._recalibrated(state)
        market = dataclasses.replace(state.spec.market, spot=self.base.spot)
        return self._recalibrated(
            RiskState(dataclasses.replace(state.spec, market=market), None, state.label)
        )

    def build(self, state: RiskState, mode: str) -> Model:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        memo = (state.key, mode, state.x0)
        if memo in self._models:
            return self._models[memo]
        base = self.base_model
        assert isinstance(base, LSV)
        if mode == "recalibrate":
            lsv = self._recalibrated(state)
        else:
            if mode == "model":
                # the recalibrated model of this surface / parameter state at the BASE spot,
                # spot moved with L held in spot (the base model when nothing but the spot
                # differs; the vega-bumped state's own calibration for vanna's dDelta/dsigma)
                ref = self._reference(state)
                kernel: BergomiSV = ref.kernel.bump(spot=state.spot)
                lev = ref.leverage
            else:
                _, _, kernel = build_market(state.spec)  # ξ₀ and parameters from the state
                lev = base.leverage
            lsv = LSV(kernel, lev.reanchored(kernel.forward_curve))
        if state.x0 is not None:
            lsv = LSV(lsv.kernel.bump(x0=np.asarray(state.x0)), lsv.leverage)
        self._models[memo] = lsv
        return lsv

    @property
    def n_calibrations(self) -> int:
        return len(self._counter.keys)

    @property
    def n_cache_misses(self) -> int:
        return self._counter.misses

    @property
    def cache_keys(self) -> list[str]:
        return list(self._counter.keys)


class LVBuilder:
    """Pure local vol from the state's surface (no leverage; every surface change is a Dupire
    rebuild, counted as a "recalibration" for the budget; ``"model"`` re-anchors the base local
    vol in spot)."""

    def __init__(self, base: RiskState) -> None:
        self.base = base
        self._counter = _Counter()
        self._models: dict[tuple[str, str], Model] = {}
        self.base_model = self.build(base, "recalibrate")

    def build(self, state: RiskState, mode: str) -> Model:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        memo = (state.key, mode)
        if memo in self._models:
            return self._models[memo]
        if mode == "model" and hasattr(self, "base_model"):
            # the Dupire local vol of this surface state at the base spot, held in spot
            if state.spot == self.base.spot:
                ref = self.build(state, "recalibrate")
            else:
                market = dataclasses.replace(state.spec.market, spot=self.base.spot)
                ref = self.build(
                    RiskState(dataclasses.replace(state.spec, market=market), None, state.label),
                    "recalibrate",
                )
            assert isinstance(ref, LocalVol)
            fc = ForwardCurve.from_config(state.spec.market)
            lv: Model = LocalVol(ref.local_vol.reanchored(fc), fc)
        else:
            surface = surface_of(state)
            self._counter.add(state.key, True)
            lv = LocalVol(LocalVolSurface.from_implied(surface, state.spec.local_vol))
        self._models[memo] = lv
        return lv

    @property
    def n_calibrations(self) -> int:
        return len(self._counter.keys)

    @property
    def n_cache_misses(self) -> int:
        return self._counter.misses

    @property
    def cache_keys(self) -> list[str]:
        return list(self._counter.keys)


class BSBuilder:
    """Black–Scholes with the state's ATM vol (flat surfaces): the analytic test bed; every mode
    gives the same model, so all regimes and variants coincide."""

    def __init__(self, base: RiskState, maturity: float = 1.0) -> None:
        self.base = base
        self.maturity = float(maturity)
        self._counter = _Counter()

    def build(self, state: RiskState, mode: str) -> Model:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        surface = surface_of(state)
        if mode == "recalibrate":  # the analogue of a leverage recalibration
            self._counter.add(state.key, False)
        return BlackScholes(float(surface.atm_vol(self.maturity)), surface.forward_curve)

    @property
    def n_calibrations(self) -> int:
        return len(self._counter.keys)

    @property
    def n_cache_misses(self) -> int:
        return 0

    @property
    def cache_keys(self) -> list[str]:
        return list(self._counter.keys)


# --------------------------------------------------------------------------------------------
# sensitivities
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Sensitivity:
    """A bump-and-reprice sensitivity with the standard error of the CRN difference."""

    name: str
    value: float
    stderr: float
    unit: str
    size: float
    scheme: str
    states: tuple[str, ...] = ()
    n_paths: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def z(self) -> float:
        return self.value / self.stderr if self.stderr > 0 else float("nan")

    def __repr__(self) -> str:
        return f"Sensitivity({self.name} = {self.value:.6g} ± {self.stderr:.2g} {self.unit})"


@dataclass
class _Priced:
    result: PriceResult
    payoffs: FloatArray  # antithetic-pair averaged, independent samples
    n_steps: int


class RiskEngine:
    """Prices states under common random numbers and forms finite differences."""

    def __init__(
        self, builder: ModelBuilder, sim: SimConfig, *, max_halvings: int = 4, memo: bool = True
    ) -> None:
        self.builder = builder
        self.sim = sim
        self.max_halvings = int(max_halvings)
        self._memo: dict[tuple[str, str, str, tuple[float, ...] | None], _Priced] = {}
        self._use_memo = memo
        self.wall_clock = 0.0
        self.n_pricings = 0

    # -- pricing -------------------------------------------------------------------------------

    def priced(self, product: Product, state: RiskState, mode: str = "recalibrate") -> _Priced:
        key = (product_key(product), state.key, mode, state.x0)
        if self._use_memo and key in self._memo:
            return self._memo[key]
        t0 = time.perf_counter()
        model = self.builder.build(state, mode)
        # the product discounts with the state's rate curve (rate bumps move the discounting too)
        bound = product.with_discount(model.forward_curve.rate_curve)
        mc = MonteCarlo(self.sim)
        grid = mc.build_grid([bound], model)
        res = mc.price(bound, model, grid=grid, keep_payoffs=True)
        pay = np.asarray(res.payoffs)
        if self.sim.antithetic:
            pay = 0.5 * (pay[0::2] + pay[1::2])
        out = _Priced(res, pay, grid.n_steps)
        self.wall_clock += time.perf_counter() - t0
        self.n_pricings += 1
        if self._use_memo:
            self._memo[key] = out
        return out

    def price(self, product: Product, state: RiskState, mode: str = "recalibrate") -> PriceResult:
        return self.priced(product, state, mode).result

    # -- bumps ---------------------------------------------------------------------------------

    def perturbed_state(
        self,
        state: RiskState,
        make: Callable[[float], SurfacePerturbation],
        size: float,
        *,
        label: str | None = None,
    ) -> tuple[RiskState, float]:
        """``state`` with ``make(size)`` added; halves ``size`` until the perturbed surface passes
        the no-arbitrage checks (SPEC v2 §7.1) and returns the achieved size."""
        s = float(size)
        last: Exception | None = None
        for _ in range(self.max_halvings + 1):
            candidate = state.with_perturbation(make(s), label=label)
            try:
                surface_of(candidate)
                if s != size:
                    log.warning("bump %s halved to %.3g of %.3g (arbitrage check)", label, s, size)
                return candidate, s
            except ArbitrageError as exc:
                last = exc
                s *= 0.5
        raise ArbitrageError(
            f"bump {label} fails the arbitrage checks after {self.max_halvings} halvings: {last}"
        )

    # -- differences ---------------------------------------------------------------------------

    def combination(
        self,
        name: str,
        product: Product,
        terms: Sequence[tuple[RiskState, str, float]],
        *,
        unit: str,
        size: float,
        scheme: str,
        extra: dict[str, Any] | None = None,
    ) -> Sensitivity:
        """``Σ_i c_i · P(state_i, mode_i)`` with the standard error of the per-path combination.

        All terms share the seed, so path ``i`` of every term uses the same normals: the paired
        estimator is valid whatever the grids (its error only shrinks when the grids coincide,
        which they do for every bump except the aged products of theta, whose grids differ by
        one step)."""
        priced = [(self.priced(product, st, mode), c) for st, mode, c in terms]
        combo = sum((c * p.payoffs for p, c in priced), np.zeros_like(priced[0][0].payoffs))
        n = combo.size
        return Sensitivity(
            name,
            float(combo.mean()),
            float(combo.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            unit,
            size,
            scheme,
            tuple(st.label for st, _, _ in terms),
            self.sim.n_paths,
            dict(extra or {}),
        )

    def paired(
        self,
        name: str,
        terms: Sequence[tuple[Product, RiskState, str, float]],
        *,
        unit: str,
        size: float,
        scheme: str,
        extra: dict[str, Any] | None = None,
    ) -> Sensitivity:
        """``Σ_i c_i · P_i`` for terms with *different products* (aged products of theta and the
        fixing risk) — the same paired estimator as :meth:`combination`."""
        priced = [(self.priced(prod, st, mode), c) for prod, st, mode, c in terms]
        combo = sum((c * p.payoffs for p, c in priced), np.zeros_like(priced[0][0].payoffs))
        n = combo.size
        return Sensitivity(
            name,
            float(combo.mean()),
            float(combo.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            unit,
            size,
            scheme,
            tuple(st.label for _, st, _, _ in terms),
            self.sim.n_paths,
            dict(extra or {}),
        )

    # -- budget --------------------------------------------------------------------------------

    @property
    def n_calibrations(self) -> int:
        return self.builder.n_calibrations

    @property
    def n_cache_misses(self) -> int:
        return self.builder.n_cache_misses

    def budget(self) -> dict[str, float]:
        return {
            "recalibrations": float(self.n_calibrations),
            "cache_misses": float(self.n_cache_misses),
            "pricings": float(self.n_pricings),
            "wall_clock_s": self.wall_clock,
        }


def product_key(product: Product) -> str:
    """Memo key of a contract: its class and every attribute at full precision (reprs round
    their floats, so two vol barriers 0.5 vp apart could share a repr), arrays by content hash,
    nested products recursively."""

    def walk(obj: Any) -> str:
        if isinstance(obj, Product):
            items = ", ".join(f"{k}={walk(v)}" for k, v in sorted(vars(obj).items()))
            return f"{type(obj).__name__}({items})"
        if isinstance(obj, np.ndarray):
            digest = hashlib.sha1(np.ascontiguousarray(obj).tobytes()).hexdigest()[:16]
            return f"array{obj.shape}:{digest}"
        if isinstance(obj, float | np.floating):
            return repr(float(obj))
        if isinstance(obj, list | tuple):
            return "[" + ", ".join(walk(v) for v in obj) + "]"
        if isinstance(obj, dict):
            return "{" + ", ".join(f"{k}: {walk(v)}" for k, v in sorted(obj.items())) + "}"
        return repr(obj)

    return walk(product)


def default_state(spec: CalibrationSpec, label: str = "base") -> RiskState:
    return RiskState(spec, None, label)


def default_params_bump(param: str, params: BergomiParams) -> float:
    """§7.1 default model-parameter bump: 5% relative for ν, k1, k2; 0.05 absolute for θ and
    the correlations."""
    if param in ("nu", "k1", "k2"):
        return 0.05 * abs(float(getattr(params, param)))
    if param in ("theta", "rho12", "rho_SX1", "rho_SX2"):
        return 0.05
    raise ValueError(f"unknown Bergomi parameter {param!r}")
