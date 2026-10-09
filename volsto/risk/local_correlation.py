"""Risk of correlation products under the local correlation model (SPEC §8.7, M12 part LC6).

**Machinery.**  An :class:`LCState` is a specification, optional spot factors with a spot
regime, and how ``λ`` follows a surface bump.  An :class:`LCBuilder` turns states into models
through the local correlation cache (and counts calibrations and cache misses);
:class:`FixedLambdaBuilder` does the same for a model given directly (a synthetic world, the
reference regression), where only the spots move.  An :class:`LCRiskEngine` prices products on
states with one pricing seed and one grid for every state, keeps pair-averaged payoffs and
returns :class:`volsto.risk.engine.Sensitivity` objects whose standard error is that of the
per-path combination (the ``RiskEngine.combination`` convention).

**Strikes do not move.**  Every product is priced with its reference levels bound to the base
state's spots (``with_reference``), so a spot bump never moves a strike.

**Common random numbers.**  One pricing seed for all states; every recalibration uses the base
``particle.seed``, so the ``λ`` difference in a vega comes from the bump, not from particle
noise.

**Spot regimes** (:func:`spot_bumped`):

* ``"sticky_strike"`` (default; the owner's and the reference's definition): each name's local
  vol is held in absolute spot (``LocalVolSurface.reanchored``, the ``model_regime_spot_bump``
  rule for pure local vol) and ``λ`` is held in absolute basket level — the basket's shifts and
  forward stay the base state's, so a common move ``e^h`` starts the basket at ``k_B(0) = h``.
  In SPEC §7.2's vocabulary this is the "model" regime of pure local vol.
* ``"sticky_moneyness"``: each name's table is held in ``k`` and the basket's shifts are rebuilt
  on the new forwards (``k_B(0) = 0``, ``λ`` unchanged in ``k``).  For the dispersion forward a
  common move then scales ``D`` by the spot factor path by path, so its elasticity is exactly 1.
* ``"implied_sticky_strike"`` (SPEC §7.2's sense): every name's implied surface and the index
  target are shifted in ``k`` (``SurfacePerturbation("shift_k")``), the Dupire surfaces are
  rebuilt and ``λ`` is recalibrated — one calibration per bump (:class:`LCBuilder` only).

**Deltas** (:func:`name_deltas`, :func:`common_delta`): ``Δ = (P₊ − P₋)/(S₊ − S₋)`` per unit of
spot (spots are 1) and the elasticity ``Δ·S/P₀`` — the percentage of the price per +1 % of spot
(with the arithmetic bump ``S·(1 ± h)`` this is the specification's ``(P₊ − P₋)/(2h·P₀)``; with
the log bump the denominator is the realised ``e^h − e^{−h}``, which makes the homogeneity
identity exact).

**Decomposition** (:func:`delta_decomposition`), with ``Δ^M_ss`` and ``Δ^M_sm`` the
sticky-strike and sticky-moneyness common deltas of the model ``M ∈ {LC, CC}`` (CC: the
constant-correlation companion, the same marginals): homogeneity ``= Δ^LC_sm``; single-name
skew channel ``= Δ^CC_ss − Δ^CC_sm``; correlation channel ``= (Δ^LC_ss − Δ^LC_sm) − (Δ^CC_ss −
Δ^CC_sm)``; level term ``= Δ^LC_sm − Δ^CC_sm`` (zero for the forward).  Each delta is an
elasticity of its own model's price, and by construction ``Δ^LC_ss = homogeneity + skew +
correlation``.

**Vegas** (:func:`name_vegas`, :func:`index_vega`, :func:`index_skew_vega`): each is a
recalibration, cached.  A name's surface gets +1 vol point parallel; ``λ`` is recalibrated to
the unchanged index smile (the single-name vega at a fixed index smile) or held (the index
smile moves with the name); the difference is the correlation re-marking.

**Model risk** (:func:`model_risk_range`): the price under a list of alternative
specifications (``R_low`` choices × particle / parametric family), each repricing the same
single-name smiles — Langnau's "chewing-gum" effect.

Checked by ``tests/test_local_correlation.py`` (the five risk tests of SPEC §8.7).
"""

from __future__ import annotations

import dataclasses
import logging
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Protocol

import numpy as np
from numpy.typing import NDArray

from volsto.calibration.lc_cache import LocalCorrelationCache, build_lc_market, lc_spec_key
from volsto.config import LocalCorrelationSpec, SimConfig, SurfacePerturbation
from volsto.models.localvol import LocalVol
from volsto.multi.lc_function import LocalCorrelationFunction
from volsto.multi.lc_model import BasketSpec, LocalCorrelationModel
from volsto.multi.mc import MultiAssetMonteCarlo
from volsto.multi.products import MultiAssetProduct
from volsto.risk.engine import Sensitivity

FloatArray = NDArray[np.float64]
log = logging.getLogger(__name__)

LC_REGIMES: Final[tuple[str, ...]] = ("sticky_strike", "implied_sticky_strike", "sticky_moneyness")
LAMBDA_MODES: Final[tuple[str, ...]] = ("recalibrate", "held")
#: Bump sizes (SPEC §7.1): spot 1 % (central), vega 1 vol point (forward difference).
SPOT_BUMP: Final[float] = 0.01
VEGA_BUMP: Final[float] = 0.01


# --------------------------------------------------------------------------------------------
# states and spot bumps
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LCState:
    """A state of the world for the risk engine.

    ``spec``: the specification (``None`` only with :class:`FixedLambdaBuilder`).
    ``spot_factors``: one multiplier per name on the base spots (``None``: unchanged), applied
    under ``regime``.  ``lambda_mode``: for a state whose surfaces differ from the base's,
    ``"recalibrate"`` (``λ`` recalibrated to the state's index target) or ``"held"`` (the base
    ``λ``)."""

    spec: LocalCorrelationSpec | None
    spot_factors: tuple[float, ...] | None = None
    regime: str = "sticky_strike"
    lambda_mode: str = "recalibrate"
    label: str = "base"

    def __post_init__(self) -> None:
        if self.regime not in LC_REGIMES:
            raise ValueError(f"regime must be one of {LC_REGIMES}")
        if self.lambda_mode not in LAMBDA_MODES:
            raise ValueError(f"lambda_mode must be one of {LAMBDA_MODES}")
        if self.spot_factors is not None and any(
            not (math.isfinite(f) and f > 0) for f in self.spot_factors
        ):
            raise ValueError("spot factors must be positive and finite")

    @property
    def key(self) -> tuple[Any, ...]:
        spec_key = None if self.spec is None else lc_spec_key(self.spec)
        moved = self.spot_factors is not None and any(f != 1.0 for f in self.spot_factors)
        return (
            spec_key,
            self.spot_factors if moved else None,
            self.regime if moved else None,
            self.lambda_mode,
        )

    def with_spots(self, factors: Sequence[float], regime: str, label: str) -> LCState:
        return dataclasses.replace(
            self, spot_factors=tuple(float(f) for f in factors), regime=regime, label=label
        )

    def with_spec(
        self, spec: LocalCorrelationSpec, label: str, lambda_mode: str = "recalibrate"
    ) -> LCState:
        return dataclasses.replace(self, spec=spec, label=label, lambda_mode=lambda_mode)


def spot_bumped(
    model: LocalCorrelationModel, factors: Sequence[float], regime: str = "sticky_strike"
) -> LocalCorrelationModel:
    """``model`` with every name's spot multiplied by its factor (module docstring):
    ``"sticky_strike"`` holds each local vol in absolute spot and ``λ`` in absolute basket level;
    ``"sticky_moneyness"`` holds the tables in ``k`` and re-centres the basket."""
    f = [float(x) for x in factors]
    if len(f) != model.n_assets:
        raise ValueError("one spot factor per name")
    if regime == "sticky_strike":
        names = []
        for m, x in zip(model.models, f, strict=True):
            curve = m.forward_curve.with_spot(m.forward_curve.spot * x)
            names.append(m if x == 1.0 else LocalVol(m.local_vol.reanchored(curve), curve))
        return model.with_models(names)
    if regime == "sticky_moneyness":
        names = [
            m if x == 1.0 else m.bump(spot=m.forward_curve.spot * x)
            for m, x in zip(model.models, f, strict=True)
        ]
        basket = BasketSpec(
            model.basket.weights, model.basket.mode, [m.forward_curve for m in names]
        )
        return LocalCorrelationModel(
            names,
            model.family,
            model.lam,
            basket,
            model.names,
            block_memory_mb=model.block_memory_mb,
        )
    raise ValueError(f"spot_bumped: regime {regime!r} needs a recalibration (LCBuilder)")


class ModelBuilder(Protocol):
    """What the engine needs of a builder."""

    base: LCState

    @property
    def base_spots(self) -> tuple[float, ...]: ...

    def build(self, state: LCState) -> LocalCorrelationModel: ...


class FixedLambdaBuilder:
    """States of a model given directly: only the spots move, ``λ`` is the model's own."""

    def __init__(self, model: LocalCorrelationModel, label: str = "base") -> None:
        self.model = model
        self.base = LCState(None, label=label)

    @property
    def base_spots(self) -> tuple[float, ...]:
        return tuple(float(m.forward_curve.spot) for m in self.model.models)

    def build(self, state: LCState) -> LocalCorrelationModel:
        if state.spec is not None:
            raise ValueError("FixedLambdaBuilder takes states without a specification")
        if state.spot_factors is None:
            return self.model
        return spot_bumped(self.model, state.spot_factors, state.regime)


class LCBuilder:
    """Local correlation models from states through the cache
    (:meth:`~volsto.calibration.lc_cache.LocalCorrelationCache.get_or_calibrate`); with
    ``allow_calibrate=False`` a missing entry raises ``CacheMissError``.  ``matrices``: the
    ``R_low`` matrix of each specification that needs one (``"historical-scaled"``), by cache
    key.  ``n_built`` counts the models asked through the cache and ``n_missed`` those that were
    calibrated."""

    def __init__(
        self,
        cache: LocalCorrelationCache,
        base: LCState,
        *,
        allow_calibrate: bool = True,
        matrices: dict[str, Any] | None = None,
    ) -> None:
        if base.spec is None:
            raise ValueError("LCBuilder needs a specification")
        self.cache = cache
        self.base = base
        self.allow_calibrate = bool(allow_calibrate)
        self.matrices = dict(matrices or {})
        self.n_built = 0
        self.n_missed = 0
        self._calibrated: dict[str, LocalCorrelationModel] = {}
        self._models: dict[tuple[Any, ...], LocalCorrelationModel] = {}
        self.base_model = self.calibrated(base.spec)

    @property
    def base_spots(self) -> tuple[float, ...]:
        return tuple(float(m.forward_curve.spot) for m in self.base_model.models)

    def calibrated(self, spec: LocalCorrelationSpec) -> LocalCorrelationModel:
        """The model of ``spec`` through the cache (memoised here)."""
        key = lc_spec_key(spec)
        if key not in self._calibrated:
            missed = not self.cache.has(spec)
            model, _ = self.cache.get_or_calibrate(
                spec, allow_calibrate=self.allow_calibrate, r_low_matrix=self.matrices.get(key)
            )
            self.n_built += 1
            self.n_missed += int(missed)
            self._calibrated[key] = model
        return self._calibrated[key]

    def _held(self, spec: LocalCorrelationSpec) -> LocalCorrelationModel:
        """The names of ``spec`` (its perturbed surfaces) with the base ``λ``."""
        market = build_lc_market(spec, r_low_matrix=self.matrices.get(lc_spec_key(spec)))
        return LocalCorrelationModel(
            market.models, market.family, self.base_model.lam, market.basket, spec.names
        )

    def build(self, state: LCState) -> LocalCorrelationModel:
        if state.spec is None:
            raise ValueError("LCBuilder takes states with a specification")
        if state.key in self._models:
            return self._models[state.key]
        same = lc_spec_key(state.spec) == lc_spec_key(self.base.spec)  # type: ignore[arg-type]
        ref = (
            self.calibrated(state.spec)
            if same or state.lambda_mode == "recalibrate"
            else self._held(state.spec)
        )
        f = state.spot_factors
        if f is None or all(x == 1.0 for x in f):
            model = ref
        elif state.regime == "implied_sticky_strike":
            model = spot_bumped(self._implied_sticky(state.spec, f), f, "sticky_moneyness")
        else:
            model = spot_bumped(ref, f, state.regime)
        self._models[state.key] = model
        return model

    def _implied_sticky(
        self, spec: LocalCorrelationSpec, factors: Sequence[float]
    ) -> LocalCorrelationModel:
        """The recalibrated model of the world after the move with every implied smile held in
        strike: each name's surface shifted by ``ln f_i`` in ``k``, the index target by the log
        of the basket's move ``ln Σ w_i f_i``."""
        if spec.perturbations is not None or spec.index_perturbation is not None:
            raise ValueError("implied_sticky_strike: the specification already carries a bump")
        shifts = tuple(
            None if f == 1.0 else SurfacePerturbation("shift_k", {"delta": math.log(f)})
            for f in factors
        )
        move = math.log(math.fsum(w * f for w, f in zip(spec.weights, factors, strict=True)))
        index = None if move == 0.0 else SurfacePerturbation("shift_k", {"delta": move})
        return self.calibrated(
            dataclasses.replace(spec, perturbations=shifts, index_perturbation=index)
        )


# --------------------------------------------------------------------------------------------
# the engine
# --------------------------------------------------------------------------------------------


@dataclass
class _Priced:
    mean: float
    stderr: float
    payoffs: FloatArray  # pair-averaged: independent samples


class LCRiskEngine:
    """Prices products on states under common random numbers and forms combinations.

    ``priced_many(products, state)`` simulates the state's model once on the pricing seed of
    ``sim`` and the grid of the products' fixings and the ``λ`` slices; the payoffs are kept
    pair-averaged, memoised by product and state."""

    def __init__(self, builder: ModelBuilder, sim: SimConfig) -> None:
        self.builder = builder
        self.sim = sim
        self._memo: dict[tuple[int, tuple[Any, ...]], _Priced] = {}
        self._bound: dict[int, MultiAssetProduct] = {}
        self.wall_clock = 0.0
        self.n_pricings = 0

    def bound(self, product: MultiAssetProduct) -> MultiAssetProduct:
        """``product`` with its reference levels bound to the base state's spots (a product
        without performances is returned as it is)."""
        if id(product) not in self._bound:
            try:
                self._bound[id(product)] = product.with_reference(self.builder.base_spots)
            except TypeError:
                self._bound[id(product)] = product
        return self._bound[id(product)]

    def priced_many(self, products: Sequence[MultiAssetProduct], state: LCState) -> list[_Priced]:
        missing = [p for p in products if (id(p), state.key) not in self._memo]
        if missing:
            t0 = time.perf_counter()
            model = self.builder.build(state)
            bound = [self.bound(p) for p in missing]
            mc = MultiAssetMonteCarlo(self.sim)
            grid = mc.build_grid(bound, model)
            results = mc.price_many(bound, model, grid=grid, keep_payoffs=True)
            for p, res in zip(missing, results, strict=True):
                pay = np.asarray(res.payoffs, dtype=np.float64)
                if self.sim.antithetic:
                    pay = 0.5 * (pay[0::2] + pay[1::2])
                self._memo[(id(p), state.key)] = _Priced(float(res.mean), float(res.stderr), pay)
            self.wall_clock += time.perf_counter() - t0
            self.n_pricings += 1
        return [self._memo[(id(p), state.key)] for p in products]

    def priced(self, product: MultiAssetProduct, state: LCState) -> _Priced:
        return self.priced_many([product], state)[0]

    def price(self, product: MultiAssetProduct, state: LCState) -> tuple[float, float]:
        """``(price, standard error)`` of ``product`` on ``state``."""
        p = self.priced(product, state)
        return p.mean, p.stderr

    def combination(
        self,
        name: str,
        product: MultiAssetProduct,
        terms: Sequence[tuple[LCState, float]],
        *,
        unit: str,
        size: float,
        scheme: str,
        extra: dict[str, Any] | None = None,
    ) -> Sensitivity:
        """``Σ_i c_i · P(state_i)`` with the standard error of the per-path combination (every
        term shares the pricing seed)."""
        combo = sum(
            (c * self.priced(product, st).payoffs for st, c in terms),
            np.zeros_like(self.priced(product, terms[0][0]).payoffs),
        )
        n = combo.size
        return Sensitivity(
            name,
            float(combo.mean()),
            float(combo.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
            unit,
            size,
            scheme,
            tuple(st.label for st, _ in terms),
            self.sim.n_paths,
            dict(extra or {}),
        )


# --------------------------------------------------------------------------------------------
# deltas
# --------------------------------------------------------------------------------------------


def _factors(
    n: int, which: int | None, h: float, bump: str
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    if bump == "log":
        up, down = math.exp(h), math.exp(-h)
    elif bump == "arith":
        up, down = 1.0 + h, 1.0 - h
    else:
        raise ValueError('bump must be "log" or "arith"')
    hit = range(n) if which is None else (which,)
    return (
        tuple(up if i in hit else 1.0 for i in range(n)),
        tuple(down if i in hit else 1.0 for i in range(n)),
    )


def _delta(
    engine: LCRiskEngine,
    product: MultiAssetProduct,
    state: LCState,
    which: int | None,
    regime: str,
    size: float,
    bump: str,
    name: str,
) -> Sensitivity:
    n = len(engine.builder.base_spots)
    up, down = _factors(n, which, size, bump)
    tag = "all" if which is None else str(which)
    s_up = state.with_spots(up, regime, f"{state.label} spot[{tag}] up")
    s_dn = state.with_spots(down, regime, f"{state.label} spot[{tag}] down")
    width = (up[0] - down[0]) if which is None else (up[which] - down[which])
    base = engine.priced(product, state).mean
    raw = engine.combination(
        name, product, [(s_up, 1.0 / width), (s_dn, -1.0 / width)],
        unit="per unit of spot (spots are 1)", size=size, scheme=f"central, {bump} bump, {regime}",
    )  # fmt: skip
    raw.extra.update(
        elasticity=raw.value / base if base != 0 else float("nan"),
        elasticity_se=raw.stderr / abs(base) if base != 0 else float("nan"),
        base_price=base,
        regime=regime,
        bump=bump,
    )
    return raw


def common_delta(
    engine: LCRiskEngine,
    product: MultiAssetProduct,
    state: LCState,
    *,
    regime: str = "sticky_strike",
    size: float = SPOT_BUMP,
    bump: str = "log",
) -> Sensitivity:
    """The delta to a common move of every spot: ``(P₊ − P₋)/(S₊ − S₋)``; ``extra`` holds the
    elasticity ``Δ·S/P₀`` (the percentage of the price per +1 % of spot) with its standard
    error (the base price's own error is not in it)."""
    return _delta(engine, product, state, None, regime, size, bump, "common delta")


def name_deltas(
    engine: LCRiskEngine,
    product: MultiAssetProduct,
    state: LCState,
    *,
    regime: str = "sticky_strike",
    size: float = SPOT_BUMP,
    bump: str = "log",
) -> list[Sensitivity]:
    """One delta per name, ``Δ_i = (P₊ − P₋)/(S₊ − S₋)`` with name ``i`` alone bumped."""
    n = len(engine.builder.base_spots)
    return [_delta(engine, product, state, i, regime, size, bump, f"delta[{i}]") for i in range(n)]


@dataclass
class DeltaDecomposition:
    """Output of :func:`delta_decomposition`: the four common deltas and the channels, as
    elasticities — each model's delta in percent of its own price per +1 % of spot — with the
    standard errors of the paired combinations; the raw deltas per unit of spot in ``deltas``."""

    lc_ss: tuple[float, float]
    lc_sm: tuple[float, float]
    cc_ss: tuple[float, float]
    cc_sm: tuple[float, float]
    homogeneity: tuple[float, float]
    skew_channel: tuple[float, float]
    correlation_channel: tuple[float, float]
    level_term: tuple[float, float]
    price_lc: float
    price_cc: float
    deltas: dict[str, float] = field(default_factory=dict)

    @property
    def residual(self) -> float:
        """``Δ^LC_ss − (homogeneity + skew + correlation)`` — zero by construction."""
        return self.lc_ss[0] - (
            self.homogeneity[0] + self.skew_channel[0] + self.correlation_channel[0]
        )

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def delta_decomposition(
    engine_lc: LCRiskEngine,
    engine_cc: LCRiskEngine,
    product: MultiAssetProduct,
    *,
    size: float = SPOT_BUMP,
    bump: str = "log",
) -> DeltaDecomposition:
    """The common delta of the local correlation model split into homogeneity, the single-name
    skew channel, the correlation channel and the level term (module docstring).  The two
    engines hold the same pricing settings (one for the model, one for its constant-correlation
    companion), so every difference is paired.  Each model's deltas are elasticities of its own
    price; ``Δ^LC_ss = homogeneity + skew + correlation`` exactly, with ``homogeneity = Δ^LC_sm``
    (``= Δ^CC_sm + level``: the level term is zero for the forward, where both are 1)."""
    if engine_lc.sim != engine_cc.sim:
        raise ValueError("the two engines must share the pricing settings")
    lc, cc = engine_lc.builder.base, engine_cc.builder.base
    p_lc = engine_lc.priced(product, lc).mean
    p_cc = engine_cc.priced(product, cc).mean
    n = len(engine_lc.builder.base_spots)
    up, down = _factors(n, None, size, bump)
    width = up[0] - down[0]

    def paired(regime: str, engine: LCRiskEngine, state: LCState) -> FloatArray:
        a = engine.priced(product, state.with_spots(up, regime, f"{state.label} {regime} up"))
        b = engine.priced(product, state.with_spots(down, regime, f"{state.label} {regime} down"))
        return np.asarray((a.payoffs - b.payoffs) / width, dtype=np.float64)

    d = {
        "lc_ss": paired("sticky_strike", engine_lc, lc),
        "lc_sm": paired("sticky_moneyness", engine_lc, lc),
        "cc_ss": paired("sticky_strike", engine_cc, cc),
        "cc_sm": paired("sticky_moneyness", engine_cc, cc),
    }

    def stat(x: FloatArray) -> tuple[float, float]:
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))

    # elasticities, each model's deltas per unit of its own price (per-path, so that every
    # channel carries the standard error of its paired combination)
    e = {k: v / (p_lc if k.startswith("lc") else p_cc) for k, v in d.items()}
    return DeltaDecomposition(
        lc_ss=stat(e["lc_ss"]),
        lc_sm=stat(e["lc_sm"]),
        cc_ss=stat(e["cc_ss"]),
        cc_sm=stat(e["cc_sm"]),
        homogeneity=stat(e["lc_sm"]),
        skew_channel=stat(e["cc_ss"] - e["cc_sm"]),
        correlation_channel=stat((e["lc_ss"] - e["lc_sm"]) - (e["cc_ss"] - e["cc_sm"])),
        level_term=stat(e["lc_sm"] - e["cc_sm"]),
        price_lc=p_lc,
        price_cc=p_cc,
        deltas={k: float(v.mean()) for k, v in d.items()},
    )


# --------------------------------------------------------------------------------------------
# vegas
# --------------------------------------------------------------------------------------------


def _bumped_spec(
    spec: LocalCorrelationSpec,
    name_bumps: dict[int, SurfacePerturbation],
    index: SurfacePerturbation | None,
) -> LocalCorrelationSpec:
    if spec.perturbations is not None or spec.index_perturbation is not None:
        raise ValueError("the specification already carries a bump")
    per_name = None
    if name_bumps:
        per_name = tuple(name_bumps.get(i) for i in range(spec.n_names))
    return dataclasses.replace(spec, perturbations=per_name, index_perturbation=index)


def _vega(
    engine: LCRiskEngine,
    product: MultiAssetProduct,
    state: LCState,
    bumped: LocalCorrelationSpec,
    name: str,
    size: float,
    lambda_mode: str,
) -> Sensitivity:
    if state.spec is None:
        raise ValueError("a vega needs a specification (LCBuilder)")
    moved = state.with_spec(bumped, f"{state.label} {name}", lambda_mode)
    out = engine.combination(
        name, product, [(moved, 1.0), (state, -1.0)],
        unit="per vol point", size=size, scheme=f"forward difference, lambda {lambda_mode}",
    )  # fmt: skip
    up, base = engine.priced(product, moved), engine.priced(product, state)
    out.extra.update(
        unpaired_stderr=float(math.hypot(up.stderr, base.stderr)), base_price=base.mean
    )
    scale = 0.01 / size
    return dataclasses.replace(out, value=out.value * scale, stderr=out.stderr * scale)


def name_vegas(
    engine: LCRiskEngine,
    product: MultiAssetProduct,
    state: LCState,
    *,
    size: float = VEGA_BUMP,
    lambda_mode: str = "recalibrate",
    names: Sequence[int] | None = None,
    all_names: bool = True,
) -> list[Sensitivity]:
    """The single-name vegas: name ``i``'s surface +``size`` parallel (in vol; 0.01 is one vol
    point), its Dupire surface rebuilt, ``λ`` recalibrated to the unchanged index smile
    (``lambda_mode="recalibrate"``) or held.  Per vol point, paired on the pricing seed;
    ``extra["unpaired_stderr"]`` is the error two independent runs would have.  ``names``: the
    names to bump (default: all); ``all_names``: also the bump of every name at once (last)."""
    if state.spec is None:
        raise ValueError("a vega needs a specification (LCBuilder)")
    which = range(state.spec.n_names) if names is None else names
    bump = SurfacePerturbation("parallel", {"size": float(size)})
    out = [
        _vega(engine, product, state, _bumped_spec(state.spec, {i: bump}, None),
              f"vega[{state.spec.names[i]}]", size, lambda_mode)
        for i in which
    ]  # fmt: skip
    if all_names:
        every = {i: bump for i in range(state.spec.n_names)}
        out.append(_vega(engine, product, state, _bumped_spec(state.spec, every, None),
                         "vega[all names]", size, lambda_mode))  # fmt: skip
    return out


def index_vega(
    engine: LCRiskEngine, product: MultiAssetProduct, state: LCState, *, size: float = VEGA_BUMP
) -> Sensitivity:
    """The index vega: the index target +``size`` parallel, ``λ`` recalibrated, the names
    unchanged.  Per vol point."""
    if state.spec is None:
        raise ValueError("a vega needs a specification (LCBuilder)")
    bump = SurfacePerturbation("parallel", {"size": float(size)})
    return _vega(
        engine,
        product,
        state,
        _bumped_spec(state.spec, {}, bump),
        "index vega",
        size,
        "recalibrate",
    )


def index_skew_vega(
    engine: LCRiskEngine, product: MultiAssetProduct, state: LCState, *, kind: str = "rotation"
) -> Sensitivity:
    """The index skew vega, ``λ`` recalibrated: ``kind="rotation"`` — the desk's +1 rota,
    ``SurfacePerturbation("rotation", {"size": 1.0, "t_min": 1/12})``; ``kind="put90"`` — a
    1 vol point tent in ``k`` at the 90 % strike (0 / 0.01 / 0 at ``ln 0.85``, ``ln 0.90``,
    ``ln 0.95`` at every pillar of the index target)."""
    if state.spec is None:
        raise ValueError("a vega needs a specification (LCBuilder)")
    if kind == "rotation":
        bump = SurfacePerturbation("rotation", {"size": 1.0, "t_min": 1.0 / 12.0})
    elif kind == "put90":
        times = [float(t) for t in state.spec.index_surface.times]
        ks = [math.log(0.85), math.log(0.90), math.log(0.95)]
        bump = SurfacePerturbation(
            "table", {"ts": times, "ks": ks, "values": [[0.0, 0.01, 0.0] for _ in times]}
        )
    else:
        raise ValueError('kind must be "rotation" or "put90"')
    out = _vega(engine, product, state, _bumped_spec(state.spec, {}, bump),
                f"index skew vega ({kind})", 0.01, "recalibrate")  # fmt: skip
    return dataclasses.replace(
        out, unit="per +1 rota" if kind == "rotation" else "per vol point at 90 %"
    )


# --------------------------------------------------------------------------------------------
# model risk
# --------------------------------------------------------------------------------------------


@dataclass
class ModelRiskRange:
    """Output of :func:`model_risk_range`: per variant the price with its standard error and the
    paired difference to the base; the minimum and the maximum."""

    rows: list[dict[str, Any]]

    @property
    def low(self) -> float:
        return float(min(r["price"] for r in self.rows))

    @property
    def high(self) -> float:
        return float(max(r["price"] for r in self.rows))

    def to_dict(self) -> dict[str, Any]:
        return {"low": self.low, "high": self.high, "rows": self.rows}


def model_risk_range(
    engine: LCRiskEngine,
    product: MultiAssetProduct,
    state: LCState,
    variants: Sequence[tuple[str, LocalCorrelationSpec]],
) -> ModelRiskRange:
    """The price of ``product`` under each alternative specification of ``variants`` (label,
    specification) — the same names and single-name smiles with another ``R_low`` or another
    ``λ`` family — recalibrated through the cache and priced on the common pricing seed: the
    price, its standard error, and the paired difference to the base state's price."""
    rows = []
    for label, spec in variants:
        other = state.with_spec(spec, f"variant {label}")
        price = engine.priced(product, other)
        diff = engine.combination(f"{label} minus base", product, [(other, 1.0), (state, -1.0)],
                                  unit="price", size=0.0, scheme="paired")  # fmt: skip
        rows.append({"label": label, "price": price.mean, "stderr": price.stderr,
                     "minus_base": diff.value, "minus_base_se": diff.stderr})  # fmt: skip
    return ModelRiskRange(rows)


def constant_companion(model: LocalCorrelationModel, lam_c: float) -> LocalCorrelationModel:
    """The constant-correlation companion of ``model``: the same names, family and basket with
    the constant ``λ_c`` on the model's own slices."""
    return model.with_lambda(
        LocalCorrelationFunction.constant(lam_c, model.lam.times, model.lam.k_grid)
    )


__all__ = [
    "LAMBDA_MODES",
    "LC_REGIMES",
    "DeltaDecomposition",
    "FixedLambdaBuilder",
    "LCBuilder",
    "LCRiskEngine",
    "LCState",
    "ModelRiskRange",
    "common_delta",
    "constant_companion",
    "delta_decomposition",
    "index_skew_vega",
    "index_vega",
    "model_risk_range",
    "name_deltas",
    "name_vegas",
    "spot_bumped",
]
