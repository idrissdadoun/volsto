"""P&L attribution (SPEC v2 §7.12, §7.12.1): ``explain(engine, product, state_0, state_1)``.

Sequential CRN revaluation from ``state_0`` to ``state_1`` in the order

1. **spot** — ``state_0`` with ``state_1``'s spot (the surface configuration re-evaluated at the
   new spot, i.e. the ``"sticky_moneyness"`` move of the state's own surface); explained by the
   sticky-moneyness delta and gamma of ``state_0``: ``Δ·δS + ½Γ·δS²``; the remainder is the
   third-order (speed) term, reported;
2. **rates** — ``state_1``'s rate and dividend curves; explained by the first-order
   **directional** sensitivities along the day's actual curve moves
   (:func:`curve_move_sensitivities`): ``rho`` along the rate-curve move, ``repo`` along the
   dividend-curve move, each ``[P(state_0 with the curve moved by h of the way) − P(state_0)]/h``
   (``h =`` :data:`CURVE_MOVE_H`, paired CRN, one pricing each).  A curve move is not a parallel
   shift — the implied dividend curve of an option chain moves by tens of per cent at its short
   end while its long end barely moves — so the mean zero-rate change times a +1 bp
   sensitivity (the pre-M10 reading) is not the first-order term.  **Convexity.**  The
   one-sided difference is ``P′ + (h/2)·P″ + O(h²)`` along the move (``P″`` the second
   derivative in the blend fraction), so the rates bucket carries ``h/2·P″`` of the move's own
   curvature and the rates step's residual keeps ``(1 − h)/2·P″`` — systematic, with the sign of
   ``P″`` — besides the spot × curve cross term (the Greeks are taken at ``state_0``'s spot, the
   actual at ``state_1``'s).  A first-order explanation leaves ``P″/2`` in the residual whatever
   the difference scheme; the one-sided scheme is kept on purpose (:data:`CURVE_MOVE_H`);
3. **surface** — ``state_1``'s surface configuration and perturbation; explained by the parallel
   vega times the average ATM change over the pillars (``detail="parallel"``) or, with
   ``detail="ladders"``, by the vega-T tents times the per-pillar ATM changes, the skew ladder
   times the 90/110 skew changes and the curvature ladder times the butterfly changes;
4. **params** — ``state_1``'s model parameters; explained by the parameter sensitivities times the
   changes;
5. **factors** — ``state_1``'s initial factor state (actual only: no first-order Greek is kept
   for ``x₀``);
6. **time** — the product seen ``dt`` later at ``state_1``: ``product.aged(dt)``, or
   ``product_1`` when given (the product seasoned to date 1 by
   :func:`volsto.products.seasoning.season`, whose day-1 fixings are the realised closes);
   explained by the held-surface theta (decay + carry; the roll-down sits in the surface step
   because ``state_1``'s surface is the observed end surface) times ``dt``, the theta pricing
   ``product_theta`` (the product seasoned with the held spot as its day-1 close) or
   ``product.aged(dt)``;
7. **recalibration** (``mode="sticky_leverage"`` only) — see below.

Every actual step is a paired CRN difference with its standard error; the residual of a step is
``actual − explained`` and the total residual ``actual P&L − Σ explained``.  Greeks are taken at
``state_0`` (start-of-day risk), so cross terms between steps land in the residuals.  The steps'
actuals telescope: ``Σ actual = P(state_1, product_1) − P(state_0, product)``, the total P&L.
Its standard error is that of the **direct paired difference** of the two endpoint pricings (both
already priced: no extra pricing) — the steps share their paths, so the root sum of squares of the
steps' errors ignores their covariance (measured 0.72–1.79× the paired error on real 2022 SPX
dates).  The residual's standard error is paired as well (:attr:`Explain.residual_stderr`): the
per-path total minus every explained item's per-path combination times its move, the combinations
kept by the call's pricing recorder — again at no pricing cost.

**Modes.**  ``mode="recalibrate"`` (the default, unchanged since M5) prices every intermediate
state and every Greek with the leverage recalibrated to that state — one leverage calibration per
distinct state (about 140 s each at 8·10⁵ particles; a ``detail="ladders"`` date needs dozens).
``mode="sticky_leverage"`` (M10 Part 3, for a daily backtest) prices every intermediate state and
every Greek on **state_0's leverage** ``L₀(t, k)``, held in forward log-moneyness, with ``ξ₀`` and
the parameters of the state priced (the builder's ``"frozen_leverage"`` mode; the engine's
builder must be based on ``state_0``): no calibration between the endpoints.  Its Greeks are
**model Greeks with the leverage frozen at state_0, not recalibrated Greeks**: a surface or
parameter bump moves ``ξ₀`` and the SV dynamics while ``L₀`` stays, so the local-vol part of the
bump the recalibration would have put into ``L`` is missing from them (the M5 sticky-leverage
vega; the recalibrated vega of a 1y study cliquet ladder is 4× the sticky one, SPEC §7.13).
Spot and rate moves are exact in this mode — the particle method is homogeneous in ``k``, so the
recalibrated leverage after a sticky-moneyness or rate move IS ``L₀`` in ``k`` (measured to
3·10⁻¹¹ relative on the cached delta-regime pairs) — which is why ``L₀`` is held in ``k`` and
not in spot (the builder's ``"sticky_leverage"`` re-anchoring would book the leverage's own spot
move as recalibration P&L).  The approximation shows in ONE place, the final **recalibration**
step: ``V(state_1, recalibrate) − V(state_1, frozen L₀)``, a paired CRN difference on the end
product; no Greek explains it — the step is its own bucket (its ``explained`` is its ``actual``,
``detail["recalibration"]``, so the total residual stays the Greek residual of steps 1–6).  The
per-date cost is then the one endpoint calibration a backtest already needs
(:func:`attribution_cost`); a pure spot move leaves the bucket at zero within its stderr.

``detail="ladders"`` on the eSSVI surfaces of the 2022 H2 history (SPEC §13.2): the curvature
ladder's 1-vp butterfly bump fails the perturbed-surface calendar check after the engine's
default 4 halvings on 62 of the 127 days (5 halvings on 51 days, 6 on 11) — the 25 days the
calendar repair changed and 37 days it left bit-equal — while the same 127 days flattened to the
mean ``ρ`` pass with 4: the per-pillar ``ρ_T`` narrows the calendar slack between the risk
pillars, not the repair.  Build the engine with ``max_halvings=6`` for a ladders backtest
(measured 2026-09-16; the sensitivities stay per requested unit).

A **settled end** (``product_1`` a :class:`~volsto.products.seasoning.Settled`, e.g. a
knock-out on date 1) is priced as its known cash (:class:`~volsto.products.seasoning.
SettledCash`: ``amount · DF(pay_time)`` on the end state's curve, zero standard error); the time
step is then that cash minus ``P(state_1, product)``, and the recalibration bucket of a settled
end is exactly zero.

:meth:`Explain.buckets` maps every explained item (``spot.delta``, ``spot.gamma``,
``rates.rho``, ``rates.repo``, ``surface.*``, ``params.<name>``, ``time.decay``, ``time.carry``,
``recalibration``) plus ``residual`` and sums to the total P&L.  Checked by
``tests/test_risk_attribution.py`` (pure spot move: residual at the third-order level, reported;
pure parallel vol move: residual within 2 stderr) and ``tests/test_attribution_sticky.py``
(sticky leverage: zero intermediate calibrations counted by the builder, the steps and buckets
summing to the total, a zero recalibration bucket on a pure spot move, the ``product_1`` wiring,
the dry-run cost against the real run).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.config import CurveConfig, SimConfig
from volsto.engine.mc import PriceResult
from volsto.market.curves import DiscountCurve
from volsto.market.surface import ImpliedSurface
from volsto.models.base import Model
from volsto.products.base import Product
from volsto.products.seasoning import Settled
from volsto.risk.engine import (
    RiskEngine,
    RiskState,
    Sensitivity,
    _Priced,
    product_key,
    surface_of,
)
from volsto.risk.greeks import delta_gamma, theta
from volsto.risk.greeks import vega as parallel_vega
from volsto.risk.ladders import K90, K110, PILLARS, curvature_T, skew_T, vega_T
from volsto.risk.volsto_sens import PARAMS, parameter_sensitivity

FloatArray = NDArray[np.float64]

EXPLAIN_MODES: tuple[str, ...] = ("recalibrate", "sticky_leverage")
"""Attribution modes (module docstring)."""
DETAILS: tuple[str, ...] = ("parallel", "ladders")
RECALIBRATION_STEP = "recalibration"
FROZEN_MODE = "frozen_leverage"
"""The builder mode the sticky-leverage attribution prices every intermediate state in."""


@dataclass(frozen=True)
class Step:
    name: str
    actual: float
    actual_stderr: float
    explained: float
    detail: dict[str, float]

    @property
    def residual(self) -> float:
        return self.actual - self.explained


@dataclass(frozen=True)
class Explain:
    steps: tuple[Step, ...]
    total: Sensitivity
    """The P&L ``P(state_1, end product) - P(state_0, product)`` with the standard error of that
    paired CRN difference, formed directly from the two endpoint pricings (not from the steps)."""
    price_0: float
    price_1: float
    mode: str = "recalibrate"
    end_key: str = ""
    """Leverage-cache key of the end state (``state_1`` with ``state_0``'s particle, simulation
    and local-vol settings)."""
    residual_stderr: float = float("nan")
    """Standard error of :attr:`residual`: the per-path residual (the total's paired difference
    minus every explained item's paired combination times its move) when every explained item
    comes from a paired combination (:attr:`residual_paired`), else the root sum of squares of
    the total's, the explained items' and the recalibration's standard errors — which ignores
    their covariance (they share one seed): an indicative number, flagged."""
    residual_paired: bool = False
    bucket_stderrs: Mapping[str, float] = dataclasses.field(default_factory=dict)
    """The standard error of each explained bucket of :meth:`buckets` (``"spot.delta"``,
    ``"surface.vega_T"``, ...), from the per-path sum of its items (a ladder bucket's pillars
    are correlated: they share the seed), no extra pricing; ``time.decay`` / ``time.carry``
    carry theta's own paired errors (:func:`volsto.risk.greeks.theta`)."""
    group_stderrs: Mapping[str, float] = dataclasses.field(default_factory=dict)
    """Paired standard errors of bucket sums, per path: ``spot``, ``rates``, ``surface``,
    ``params`` (the explained items of each step), ``time`` (decay + carry: theta's held-surface
    difference), ``greeks`` (every explained item of steps 1-6) and ``explained`` (with the
    recalibration bucket)."""
    unpaired: tuple[str, ...] = ()
    """The buckets and groups whose standard error above is a root sum of squares (an item
    without a per-path combination: a derived Greek) — indicative only."""

    @property
    def explained(self) -> float:
        return float(sum(s.explained for s in self.steps))

    @property
    def residual(self) -> float:
        return self.total.value - self.explained

    @property
    def recalibration(self) -> float:
        """The recalibration bucket (0 in ``"recalibrate"`` mode)."""
        return float(sum(s.actual for s in self.steps if s.name == RECALIBRATION_STEP))

    def buckets(self) -> dict[str, float]:
        """Every explained item as ``"<step>.<item>"`` (the recalibration bucket as
        ``"recalibration"``) and ``"residual"``; the values sum to the total P&L."""
        out: dict[str, float] = {}
        for s in self.steps:
            for k, v in s.detail.items():
                key = k if s.name == RECALIBRATION_STEP else f"{s.name}.{k}"
                out[key] = out.get(key, 0.0) + float(v)
        out["residual"] = self.residual
        return out

    def as_frame(self) -> pd.DataFrame:
        rows = [
            {
                "step": s.name,
                "actual": s.actual,
                "actual_stderr": s.actual_stderr,
                "explained": s.explained,
                "residual": s.residual,
            }
            for s in self.steps
        ]
        rows.append(
            {
                "step": "total",
                "actual": self.total.value,
                "actual_stderr": self.total.stderr,
                "explained": self.explained,
                "residual": self.residual,
                "residual_stderr": self.residual_stderr,
                "residual_paired": self.residual_paired,
            }
        )
        return pd.DataFrame(rows)


class FrozenLeverageEngine(RiskEngine):
    """The engine as the sticky-leverage attribution sees it: every ``"recalibrate"`` and
    ``"sticky_leverage"`` pricing of a state other than ``state_0`` is priced in the builder's
    ``"frozen_leverage"`` mode (``state_0``'s own requests stay ``"recalibrate"``: the same model,
    one memo entry).  Pricing, memo and budget are the wrapped engine's."""

    def __init__(self, inner: RiskEngine, base: RiskState) -> None:
        super().__init__(inner.builder, inner.sim, max_halvings=inner.max_halvings, memo=False)
        self.inner = inner
        self.base_key = base.key

    def priced(self, product: Product, state: RiskState, mode: str = "recalibrate") -> _Priced:
        if mode in ("recalibrate", "sticky_leverage", FROZEN_MODE):
            mode = "recalibrate" if state.key == self.base_key else FROZEN_MODE
        return self.inner.priced(product, state, mode)


class _PathRecorder(RiskEngine):
    """One :func:`explain` call's view of an engine: every pricing is delegated (and kept, so
    forming a combination again never prices), and the per-path combination behind every
    :class:`Sensitivity` it returns is remembered (:meth:`path_of`) — the total's and the
    residual's paired standard errors then cost no pricing."""

    def __init__(self, inner: RiskEngine) -> None:
        super().__init__(inner.builder, inner.sim, max_halvings=inner.max_halvings, memo=False)
        self.inner = inner
        self._kept: dict[tuple[str, str, str, tuple[float, ...] | None], _Priced] = {}
        self._paths: dict[int, tuple[Sensitivity, FloatArray]] = {}

    def priced(self, product: Product, state: RiskState, mode: str = "recalibrate") -> _Priced:
        key = (product_key(product), state.key, mode, state.x0)
        hit = self._kept.get(key)
        if hit is None:
            hit = self._kept[key] = self.inner.priced(product, state, mode)
        return hit

    def path(self, terms: Sequence[tuple[Product, RiskState, str, float]]) -> FloatArray:
        """``Σ c_i · payoffs_i`` per path (the engine's paired estimator before averaging)."""
        priced = [(self.priced(p, s, m), c) for p, s, m, c in terms]
        out = np.zeros_like(priced[0][0].payoffs)
        for q, c in priced:
            out = out + c * q.payoffs
        return np.asarray(out, dtype=np.float64)

    def _keep(
        self, sens: Sensitivity, terms: Sequence[tuple[Product, RiskState, str, float]]
    ) -> Sensitivity:
        self._paths[id(sens)] = (sens, self.path(terms))
        return sens

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
        s = super().combination(
            name, product, terms, unit=unit, size=size, scheme=scheme, extra=extra
        )
        return self._keep(s, [(product, st, m, c) for st, m, c in terms])

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
        s = super().paired(name, terms, unit=unit, size=size, scheme=scheme, extra=extra)
        return self._keep(s, list(terms))

    def path_of(self, sens: Sensitivity) -> FloatArray | None:
        """The per-path combination behind ``sens`` (``None`` for a derived sensitivity)."""
        hit = self._paths.get(id(sens))
        return hit[1] if hit is not None and hit[0] is sens else None


def _stderr(path: FloatArray) -> float:
    n = path.size
    return float(path.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")


def _actual(
    engine: RiskEngine, product: Product, a: RiskState, b: RiskState, name: str
) -> Sensitivity:
    return engine.combination(
        name,
        product,
        [(b, "recalibrate", 1.0), (a, "recalibrate", -1.0)],
        unit="price",
        size=0.0,
        scheme="revaluation",
    )


def _skew_90_110(surface: ImpliedSurface, ps: FloatArray) -> FloatArray:
    k90, k110 = np.full(ps.size, K90), np.full(ps.size, K110)
    return np.asarray(surface.implied_vol_k(k90, ps)) - np.asarray(surface.implied_vol_k(k110, ps))


def _butterfly_90_110(surface: ImpliedSurface, ps: FloatArray) -> FloatArray:
    k90, k110 = np.full(ps.size, K90), np.full(ps.size, K110)
    wings = 0.5 * (
        np.asarray(surface.implied_vol_k(k90, ps)) + np.asarray(surface.implied_vol_k(k110, ps))
    )
    return np.asarray(wings - np.asarray(surface.atm_vol(ps)), dtype=np.float64)


CURVE_MOVE_H = 0.1
"""Fraction of the day's curve move the directional rate sensitivities bump by, in a one-sided
difference ``[P(h) - P(0)]/h = P' + (h/2) P'' + O(h^2)``: the rates bucket carries ``h/2 P''``
(5 % of the move's curvature) and the rates residual ``(1 - h)/2 P''``, systematic with the sign of
``P''`` (module docstring).  Measured on the 2022-10-27..11-01 SPX moves (4000 paths, leverage
frozen at 20000 particles, x100; ``scratchpad bt4/convexity.py``, 44 (row, curve) pairs):

* the h dependence: a smooth product barely depends on h (the cliquet +0.115 at h = 0.01, 0.1 and
  0.25; the 2-day variance swap -0.355 / -0.339 / -0.311 on 10-28, when the 7-day implied
  dividend yield fell from 49.6 % to -0.35 %), while a knock-out product's difference is noise at
  h = 0.01 (KO variance swap -0.24 +- 0.32 against -0.02 +- 0.09 at 0.1; autocall on 11-01
  -0.16 +- 0.10 against -0.05 +- 0.02);
* the first-order bias ``h/2 P''`` (one-sided minus central, paired): on the variance swaps'
  repo buckets 5.1-5.2 % on 10-28 (0.0183 on the 2-day swap, 0.0007 / 0.0003 on the 6m / 1y) and
  9-14 % of the (tiny) 10-31 / 11-01 ones, deterministic in the paths; on every other row below
  1.8 stderr or below 3e-7 in absolute value;
* the two schemes that remove it at one extra pricing per moved curve: the **central** difference
  ``[P(h) - P(-h)]/(2h)`` extrapolates the move backwards beyond both observed curves, and on the
  barrier products its stderr is up to 34x (autocall 11-01 rho) and 229x (Phoenix 11-01 repo)
  the one-sided one (half of it on two 10-28 repo rows); the **half-step** Richardson
  ``2 D(h/2) - D(h)`` removes the bias at equal stderr on the smooth products but has 2.4-3.0x
  the stderr on the knock-out ones (KO variance swap, autocall), where the bias is below 1.8
  stderr;
* neither removes the rates residual's convexity (``P''/2`` instead of ``(1 - h)/2 P''``).

So the one-sided scheme is kept: one pricing per moved curve, a documented 5 % curvature share in
the bucket, and the curvature in the residual."""
PARAM_MOVE_RTOL = 1e-12
"""A model parameter moved when ``|p1 - p0| > PARAM_MOVE_RTOL * max(1, |p0|)`` (a fit sitting on
a bound jitters by ~1e-16, which is not a move worth two pricings)."""


def param_moved(p0: float, p1: float) -> bool:
    """Whether a model parameter moved between two states (:data:`PARAM_MOVE_RTOL`)."""
    return abs(float(p1) - float(p0)) > PARAM_MOVE_RTOL * max(1.0, abs(float(p0)))


def blend_curves(c0: CurveConfig, c1: CurveConfig, h: float) -> CurveConfig:
    """The zero curve whose ``ln DF`` is ``(1 − h) ln DF₀ + h ln DF₁``, exactly: both curves are
    linear in ``ln DF`` between their own pillars and extrapolate their last forward, so the
    blend is linear on the union of the pillars and extrapolates the blend of the last forwards
    — a :class:`~volsto.config.CurveConfig` on the union grid represents it without error."""
    d0, d1 = DiscountCurve.from_config(c0), DiscountCurve.from_config(c1)
    t = np.union1d(np.asarray(c0.times, dtype=np.float64), np.asarray(c1.times, dtype=np.float64))
    log_df = (1.0 - h) * d0.log_df(t) + h * d1.log_df(t)
    return CurveConfig(tuple(float(x) for x in t), tuple(float(x) for x in -log_df / t))


def curve_move_sensitivities(
    engine: RiskEngine,
    product: Product,
    state_0: RiskState,
    state_1: RiskState,
    h: float = CURVE_MOVE_H,
) -> dict[str, Sensitivity]:
    """First-order P&L of the day's rate-curve move (``"rho"``) and dividend-curve move
    (``"repo"``): ``[P(state_0 with that curve at h of the way to state_1's) − P(state_0)]/h``
    (:func:`blend_curves`), a paired CRN difference, one pricing per curve that moved (a curve
    that did not move gives an exact zero without pricing).  The value is in price units for the
    whole move (a directional derivative times a unit move), so it is the explained rates
    P&L."""
    if not 0.0 < h <= 1.0:
        raise ValueError("h must lie in (0, 1]")
    m0, m1 = state_0.spec.market, state_1.spec.market
    out: dict[str, Sensitivity] = {}
    for name, field in (("rho", "rate_curve"), ("repo", "dividend_curve")):
        c0, c1 = getattr(m0, field), getattr(m1, field)
        if c0 == c1:
            out[name] = Sensitivity(
                f"{name}[curve move]", 0.0, 0.0, "price per curve move", h, "directional"
            )
            continue
        blended = blend_curves(c0, c1, h)
        market = (
            dataclasses.replace(m0, rate_curve=blended)
            if field == "rate_curve"
            else dataclasses.replace(m0, dividend_curve=blended)
        )
        bumped = RiskState(
            dataclasses.replace(state_0.spec, market=market), state_0.x0, f"{name}:{h:g} move"
        )
        out[name] = engine.combination(
            f"{name}[curve move]",
            product,
            [(bumped, "recalibrate", 1.0 / h), (state_0, "recalibrate", -1.0 / h)],
            unit="price per curve move",
            size=h,
            scheme="directional",
            extra={"curve": field},
        )
    return out


def _theta_product(product: Product, dt: float, product_theta: Product | None) -> Product:
    if product_theta is not None:
        return product_theta
    try:
        return product.aged(dt)
    except (ValueError, NotImplementedError) as exc:
        raise ValueError(
            f"explain: {type(product).__name__}.aged({dt:g}) fails ({exc}); pass product_theta, "
            "the product seasoned to date 1 with the held spot as its day-1 close "
            "(volsto.products.seasoning.season on history.extended(date_1, spot_0))"
        ) from exc


def explain(
    engine: RiskEngine,
    product: Product,
    state_0: RiskState,
    state_1: RiskState,
    *,
    dt: float = 0.0,
    detail: str = "parallel",
    pillars: tuple[float, ...] = PILLARS,
    size: float = 0.01,
    mode: str = "recalibrate",
    product_1: Product | Settled | None = None,
    product_theta: Product | Settled | None = None,
) -> Explain:
    """Sequential attribution of ``P(state_1, product_1 or product aged by dt) − P(state_0)``
    (module docstring).  ``mode="sticky_leverage"`` needs ``engine.builder`` based on
    ``state_0``; ``product_1`` / ``product_theta`` need ``dt > 0``, and ``product_theta`` only
    serves (and is only accepted) with ``product_1``; a :class:`~volsto.products.seasoning.
    Settled` for either is priced as its cash (:meth:`Settled.as_product`)."""
    if detail not in DETAILS:
        raise ValueError("detail must be 'parallel' or 'ladders'")
    if mode not in EXPLAIN_MODES:
        raise ValueError(f"mode must be one of {EXPLAIN_MODES}")
    if dt <= 0 and (product_1 is not None or product_theta is not None):
        raise ValueError("product_1 / product_theta are the product one step later: give dt > 0")
    if product_theta is not None and product_1 is None:
        raise ValueError(
            "product_theta is the held-spot product of a seasoned time step: it is used only "
            "with product_1 (without product_1 the theta ages the product itself)"
        )
    if isinstance(product_1, Settled):
        product_1 = product_1.as_product()
    if isinstance(product_theta, Settled):
        product_theta = product_theta.as_product()
    if mode == "sticky_leverage":
        base = getattr(engine.builder, "base", None)
        if not isinstance(base, RiskState) or base.key != state_0.key:
            raise ValueError(
                "the sticky-leverage attribution freezes the builder's base leverage: build the "
                "engine's builder on state_0"
            )
    # the recorders keep every pricing and the per-path combination behind every Greek, so the
    # total's and the residual's paired standard errors cost no pricing
    outer = _PathRecorder(engine)
    eng = (
        _PathRecorder(FrozenLeverageEngine(outer, state_0)) if mode == "sticky_leverage" else outer
    )
    steps: list[Step] = []
    # (bucket, Greek, the move it multiplies) for every explained item of steps 1-6
    explained_paths: list[tuple[str, Sensitivity, float]] = []
    time_stderrs: list[float] = []
    surf0, surf1 = surface_of(state_0), surface_of(state_1)

    # 1. spot
    s_a = state_0.with_spot(state_1.spot, label="explain:spot")
    if state_1.spot != state_0.spot:
        act = _actual(eng, product, state_0, s_a, "explain.spot")
        d, g = delta_gamma(eng, product, state_0, "sticky_moneyness", size)
        ds = state_1.spot - state_0.spot
        det = {"delta": d.value * ds, "gamma": 0.5 * g.value * ds * ds}
        explained_paths += [("spot.delta", d, ds), ("spot.gamma", g, 0.5 * ds * ds)]
        steps.append(Step("spot", act.value, act.stderr, det["delta"] + det["gamma"], det))

    # 2. rates
    m1 = state_1.spec.market
    s_b = RiskState(
        dataclasses.replace(
            s_a.spec,
            market=dataclasses.replace(
                s_a.spec.market, rate_curve=m1.rate_curve, dividend_curve=m1.dividend_curve
            ),
        ),
        s_a.x0,
        "explain:rates",
    )
    if s_b.key != s_a.key:
        act = _actual(eng, product, s_a, s_b, "explain.rates")
        rs = curve_move_sensitivities(eng, product, state_0, state_1)
        det = {"rho": rs["rho"].value, "repo": rs["repo"].value}
        explained_paths += [("rates.rho", rs["rho"], 1.0), ("rates.repo", rs["repo"], 1.0)]
        steps.append(Step("rates", act.value, act.stderr, det["rho"] + det["repo"], det))

    # 3. surface
    s_c = RiskState(
        dataclasses.replace(
            s_b.spec, surface=state_1.spec.surface, perturbation=state_1.spec.perturbation
        ),
        s_b.x0,
        "explain:surface",
    )
    if s_c.key != s_b.key:
        act = _actual(eng, product, s_b, s_c, "explain.surface")
        ps = np.asarray(pillars, dtype=np.float64)
        d_atm = np.asarray(surf1.atm_vol(ps)) - np.asarray(surf0.atm_vol(ps))
        det = {}
        if detail == "parallel":
            v = parallel_vega(eng, product, state_0, "recalibrated", size)
            det["parallel_vega"] = v.value * float(d_atm.mean()) / 0.01
            explained_paths.append(("surface.parallel_vega", v, float(d_atm.mean()) / 0.01))
        else:
            vt = vega_T(eng, product, state_0, pillars, size, with_tents=True)
            det["vega_T"] = float(sum(t.value * dv / 0.01 for t, dv in zip(vt.tents, d_atm)))
            d_skew = _skew_90_110(surf1, ps) - _skew_90_110(surf0, ps)
            d_fly = _butterfly_90_110(surf1, ps) - _butterfly_90_110(surf0, ps)
            sk = skew_T(eng, product, state_0, pillars, size)
            cv = curvature_T(eng, product, state_0, pillars, size)
            det["skew_T"] = float(sum(e.value * x / 0.01 for e, x in zip(sk.entries, d_skew)))
            det["curvature_T"] = float(sum(e.value * x / 0.01 for e, x in zip(cv.entries, d_fly)))
            explained_paths += [
                ("surface.vega_T", t, float(dv) / 0.01) for t, dv in zip(vt.tents, d_atm)
            ]
            explained_paths += [
                ("surface.skew_T", e, float(x) / 0.01) for e, x in zip(sk.entries, d_skew)
            ]
            explained_paths += [
                ("surface.curvature_T", e, float(x) / 0.01) for e, x in zip(cv.entries, d_fly)
            ]
        steps.append(Step("surface", act.value, act.stderr, float(sum(det.values())), det))

    # 4. parameters
    s_d = RiskState(
        dataclasses.replace(s_c.spec, model=state_1.spec.model), s_c.x0, "explain:params"
    )
    if s_d.key != s_c.key:
        act = _actual(eng, product, s_c, s_d, "explain.params")
        det = {}
        for name in PARAMS:
            p0 = float(getattr(state_0.spec.model, name))
            p1 = float(getattr(state_1.spec.model, name))
            if param_moved(p0, p1):
                ps_ = parameter_sensitivity(eng, product, state_0, name)
                det[name] = ps_.value * (p1 - p0)
                explained_paths.append((f"params.{name}", ps_, p1 - p0))
        steps.append(Step("params", act.value, act.stderr, float(sum(det.values())), det))

    # 5. factor state
    s_e = RiskState(s_d.spec, state_1.x0, "explain:factors")
    if s_e.x0 != s_d.x0:
        act = _actual(eng, product, s_d, s_e, "explain.factors")
        steps.append(Step("factors", act.value, act.stderr, 0.0, {}))

    # 6. time
    end_product = product
    held_terms: list[tuple[Product, RiskState, str, float]] = []
    if dt > 0:
        end_product = product_1 if product_1 is not None else product.aged(dt)
        act = eng.paired(
            "explain.time",
            [(end_product, s_e, "recalibrate", 1.0), (product, s_e, "recalibrate", -1.0)],
            unit="price",
            size=dt,
            scheme="revaluation",
        )
        aged = product.aged(dt) if product_1 is None else _theta_product(product, dt, product_theta)
        th = theta(eng, product, state_0, dt, aged=aged)
        det = {"decay": th.decay.value * dt, "carry": th.carry.value * dt}
        # decay and carry are paired combinations of theta; their sum is the held-surface
        # difference, whose paired terms are:
        held_terms = [(aged, state_0, "recalibrate", 1.0), (product, state_0, "recalibrate", -1.0)]
        time_stderrs = [dt * th.decay.stderr, dt * th.carry.stderr]
        steps.append(Step("time", act.value, act.stderr, det["decay"] + det["carry"], det))

    # 7. recalibration (sticky leverage): the endpoint's leverage refit, a bucket of its own
    recal_path: FloatArray | None = None
    recal_stderrs: list[float] = []
    if mode == "sticky_leverage":
        act = outer.paired(
            "explain.recalibration",
            [(end_product, s_e, "recalibrate", 1.0), (end_product, s_e, FROZEN_MODE, -1.0)],
            unit="price",
            size=0.0,
            scheme="revaluation",
        )
        recal_path = outer.path_of(act)
        recal_stderrs = [act.stderr]
        steps.append(
            Step(RECALIBRATION_STEP, act.value, act.stderr, act.value, {"recalibration": act.value})
        )

    end_price = outer.price(end_product, s_e).mean
    p0 = outer.price(product, state_0).mean
    # the total: the paired difference of the two endpoint pricings (both kept), whose standard
    # error includes the covariance of the steps (a root sum of squares of theirs does not)
    total_path = outer.path(
        [(end_product, s_e, "recalibrate", 1.0), (product, state_0, "recalibrate", -1.0)]
    )
    total = Sensitivity(
        "pnl",
        end_price - p0,
        _stderr(total_path),
        "price",
        0.0,
        "revaluation",
        (state_0.label, state_1.label),
        engine.sim.n_paths,
    )
    # per path: every bucket's sum of its items, and the residual (the total minus them all)
    zero = np.zeros_like(total_path)
    bucket_path: dict[str, FloatArray] = {}
    rss: dict[str, float] = {}
    unpaired: set[str] = set()
    for bucket, sens, move in explained_paths:
        rss[bucket] = rss.get(bucket, 0.0) + (move * sens.stderr) ** 2
        path = eng.path_of(sens)
        if path is None:
            if sens.value != 0.0 or sens.stderr != 0.0:  # a zero without pricing has no path
                unpaired.add(bucket)
            continue
        bucket_path[bucket] = bucket_path.get(bucket, zero) + move * path
    held_path = eng.path(held_terms) if held_terms else None
    paired = not unpaired
    bucket_se = {
        b: (
            (_stderr(bucket_path[b]) if b in bucket_path else 0.0)
            if b not in unpaired
            else float(np.sqrt(v))
        )
        for b, v in rss.items()
    }
    if held_terms:
        bucket_se["time.decay"], bucket_se["time.carry"] = time_stderrs
    members: dict[str, list[str]] = {}
    for b in bucket_se:
        if not b.startswith("time."):
            members.setdefault(b.split(".", 1)[0], []).append(b)
    group_se: dict[str, float] = {}
    unpaired_groups: set[str] = set()

    def group(name: str, names: Sequence[str], extra: Sequence[FloatArray | None]) -> None:
        if any(b in unpaired for b in names):
            unpaired_groups.add(name)
            group_se[name] = float(np.sqrt(sum(rss[b] for b in names)))
            return
        path = zero
        for b in names:
            path = path + bucket_path.get(b, zero)
        for x in extra:
            if x is not None:
                path = path + x
        group_se[name] = _stderr(path)

    for name, names in members.items():
        group(name, names, ())
    if held_path is not None:
        group_se["time"] = _stderr(held_path)
    greek_buckets = [b for names in members.values() for b in names]
    group("greeks", greek_buckets, (held_path,))
    group("explained", greek_buckets, (held_path, recal_path))
    residual_path = total_path.copy()
    for path in bucket_path.values():
        residual_path = residual_path - path
    if held_path is not None:
        residual_path = residual_path - held_path
    if recal_path is not None:
        residual_path = residual_path - recal_path
    if paired:
        residual_se = _stderr(residual_path)
    else:
        items = [abs(m) * s.stderr for _, s, m in explained_paths]
        items += time_stderrs + recal_stderrs
        residual_se = float(np.sqrt(total.stderr**2 + sum(x * x for x in items)))
    return Explain(
        tuple(steps),
        total,
        p0,
        end_price,
        mode,
        s_e.key,
        residual_se,
        paired,
        bucket_se,
        group_se,
        tuple(sorted(unpaired | unpaired_groups)),
    )


# --------------------------------------------------------------------------------------------
# cost projection
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AttributionCost:
    """What one :func:`explain` call costs, counted by a dry run (nothing priced or calibrated).

    ``calibration_keys``: the distinct leverage calibrations the run touches — ``state_0``'s and
    the end state's (:attr:`endpoint_keys`) included, the cached ones included (a backtest's
    ``state_0`` is the previous date's end state, so its incremental cost is
    :attr:`calibrations` − 1 when the endpoints differ); ``pricings``: the distinct Monte Carlo
    pricings (product, state, mode, factor state) at ``n_paths`` each."""

    mode: str
    detail: str
    steps: tuple[str, ...]
    calibration_keys: tuple[str, ...]
    endpoint_keys: tuple[str, ...]
    pricings: int
    n_paths: int

    @property
    def calibrations(self) -> int:
        return len(self.calibration_keys)

    @property
    def intermediate_calibrations(self) -> int:
        """Calibrations of states other than the endpoints (0 in sticky-leverage mode)."""
        return len(set(self.calibration_keys) - set(self.endpoint_keys))

    def wall_clock(
        self,
        seconds_per_calibration: float,
        seconds_per_pricing: float,
        *,
        cached: Sequence[str] = (),
    ) -> float:
        """Projected seconds: the calibrations not in ``cached`` plus the pricings."""
        todo = [k for k in self.calibration_keys if k not in set(cached)]
        return len(todo) * seconds_per_calibration + self.pricings * seconds_per_pricing

    def as_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "detail": self.detail,
            "steps": list(self.steps),
            "calibrations": self.calibrations,
            "intermediate_calibrations": self.intermediate_calibrations,
            "pricings": self.pricings,
            "n_paths": self.n_paths,
        }


class DryRunBuilder:
    """A builder that builds nothing (the dry run never prices); ``base`` is ``state_0``."""

    def __init__(self, base: RiskState) -> None:
        self.base = base

    def build(self, state: RiskState, mode: str) -> Model:
        raise AssertionError("the attribution dry run builds no model")

    @property
    def n_calibrations(self) -> int:
        return 0

    @property
    def n_cache_misses(self) -> int:
        return 0

    @property
    def cache_keys(self) -> list[str]:
        return []


class DryRunEngine(RiskEngine):
    """Records every pricing request and the leverage calibration it implies under
    :class:`~volsto.risk.engine.LSVBuilder` semantics, returning zero payoffs."""

    def __init__(self, sim: SimConfig, base: RiskState, max_halvings: int = 4) -> None:
        super().__init__(DryRunBuilder(base), sim, max_halvings=max_halvings, memo=False)
        self.base = base
        self.requests: set[tuple[str, str, str, tuple[float, ...] | None]] = set()
        self.calibration_keys: list[str] = [base.key]

    def _calibrates(self, key: str) -> None:
        if key not in self.calibration_keys:
            self.calibration_keys.append(key)

    def priced(self, product: Product, state: RiskState, mode: str = "recalibrate") -> _Priced:
        self.requests.add((product_key(product), state.key, mode, state.x0))
        if mode == "recalibrate":
            self._calibrates(state.key)
        elif mode == "model":
            market = dataclasses.replace(state.spec.market, spot=self.base.spot)
            self._calibrates(RiskState(dataclasses.replace(state.spec, market=market)).key)
        zeros = np.zeros(2)
        return _Priced(PriceResult(0.0, 0.0, self.sim.n_paths, 2, zeros), zeros, 0)


def attribution_cost(
    product: Product,
    state_0: RiskState,
    state_1: RiskState,
    sim: SimConfig,
    *,
    mode: str = "recalibrate",
    detail: str = "parallel",
    dt: float = 0.0,
    pillars: tuple[float, ...] = PILLARS,
    size: float = 0.01,
    product_1: Product | Settled | None = None,
    product_theta: Product | Settled | None = None,
    max_halvings: int = 4,
) -> AttributionCost:
    """The calibrations and pricings of ``explain(...)`` with the same arguments, counted by a
    dry run of the attribution itself (surfaces are built and arbitrage-checked, nothing is
    simulated or calibrated), so a backtest can project its wall clock per date, mode and detail
    before running (:meth:`AttributionCost.wall_clock`).  Counts follow
    :class:`~volsto.risk.engine.LSVBuilder`: a ``"recalibrate"`` request is a calibration of its
    state, a ``"model"`` request one of its state at the base spot, the frozen and sticky
    leverage modes none.  ``max_halvings`` is the engine's (a bump the arbitrage checks refuse
    after that many halvings raises here as in the real run; the curvature ladder on the repaired
    eSSVI surfaces of the 2022 H2 history needs up to 6, SPEC §7.12.1).  Checked against a real
    run's builder and engine counters by ``tests/test_attribution_sticky.py``."""
    dry = DryRunEngine(sim, state_0, max_halvings)
    ex = explain(
        dry,
        product,
        state_0,
        state_1,
        dt=dt,
        detail=detail,
        pillars=pillars,
        size=size,
        mode=mode,
        product_1=product_1,
        product_theta=product_theta,
    )
    endpoints = tuple(dict.fromkeys((state_0.key, ex.end_key)))
    return AttributionCost(
        mode=mode,
        detail=detail,
        steps=tuple(s.name for s in ex.steps),
        calibration_keys=tuple(dry.calibration_keys),
        endpoint_keys=endpoints,
        pricings=len(dry.requests),
        n_paths=sim.n_paths,
    )


#: The names these classes had before M10 Part 3 made them public (kept as aliases).
_FrozenLeverageEngine = FrozenLeverageEngine
_DryRunBuilder = DryRunBuilder
_DryRunEngine = DryRunEngine

__all__ = [
    "CURVE_MOVE_H",
    "DETAILS",
    "EXPLAIN_MODES",
    "PARAM_MOVE_RTOL",
    "AttributionCost",
    "DryRunBuilder",
    "DryRunEngine",
    "Explain",
    "FrozenLeverageEngine",
    "Step",
    "attribution_cost",
    "blend_curves",
    "curve_move_sensitivities",
    "explain",
    "param_moved",
]
