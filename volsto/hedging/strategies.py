"""Hedging strategies derived from the payoff (SPEC §8, M8 Part 3).

**Layer A — the Greek-targeting builder** (:class:`GreekTargetStrategy`).  ``targets`` is the set
of sensitivities to neutralise, ``instruments`` the candidate hedge set; at every rebalancing
date and on every world path the quantities ``q`` solve the weighted least squares

    min_q  Σ_g w_g (G_g^product + Σ_j q_j G_g^j)²  +  Σ_j ridge ⟨(G^j)²⟩ q_j²
           +  turnover |q − q_prev|²

per path, in closed form (batched normal equations), with **static** legs (given quantities, a
cliquet's cap-call strip) held fixed and the **residual exposure**
``r_g = G_g^product + Σ_j q_j G_g^j`` reported after hedging.  ``ridge`` (relative to the
instrument's cross-path mean squared sensitivity ``⟨(G^j)²⟩``) keeps quantities finite when
instruments are collinear for the chosen targets or their sensitivities vanish on a path;
``turnover`` penalises date-to-date
changes (stability, lower costs).  Target names are the sensitivities of the conditional-pricing
engine (:mod:`volsto.hedging.pricing`): ``delta``, ``gamma`` (per unit spot, ``"model"`` regime;
another delta regime — ``sticky_strike``, ``sticky_skew``, ``sticky_moneyness``,
``sticky_local_vol`` — replaces ``delta`` by the bump of that regime's surface move, §7.2, so the
regime question becomes a P&L comparison; ``min_variance`` replaces it by the pricing model's
minimum-variance spot-only delta, the model delta plus the vol-correlation term
:func:`~volsto.hedging.hedger.min_variance_delta`, the benchmark of M8b study D), ``vega`` (parallel
+1 vp of the pricing surface, leverage recalibrated), ``fwd_var:<lo>-<hi>`` (a forward-variance
bucket, §7.5), ``skew_T:<T>`` / ``curvature_T:<T>`` (the §7.6 tents at the risk pillars), ``vanna``
(``∂vega/∂ln S`` from the regressed vega polynomial), ``volga`` (second difference in the parallel
vol), ``dX1`` / ``dX2`` (the value gradient in the pricing factors), and ``param:<name>`` (a
model-parameter bump, §7.9). Which of these the hedger can build depends on the pricing context (a
Black–Scholes model offers ``vega`` / ``volga`` through its own vol; surface bumps need a
calibration state and the leverage cache); an unavailable target raises with the list of the
available ones.  **Static legs** (``static``: instrument name → quantity, constant or a callable of
``t``) are held as given and
their sensitivities subtracted before the solve — the payoff-derived replication pieces (a
digital's call spread, the cliquet's cap-call strip, a knock-in's in–out parity) live there.

**Layer B — per-product presets** (:func:`default_strategy`, the registry
:data:`PRESETS` keyed by product class; each preset exposes its target and instrument lists, so a
study can drop or add a leg).  Built from the product's structure:

* vanilla / digital: ``delta`` (+ ``vega`` with a same-expiry ATM vanilla when ``vol_hedged``);
  digital: ``delta`` + the call-spread replication (static, short, smoothing width
  ``digital_width``);
* forward-start option / straddle, FVA: before ``T1`` the delta is about zero (reported) — the
  forward-variance bucket ``[T1, T2]`` with a forward variance swap and the forward skew with a
  forward-start risk reversal (target ``skew_T:T2``, else ``dX1`` when tents are unavailable);
  after ``T1`` the product is a vanilla — the preset switches to ``delta`` + ``vega`` with a
  vanilla struck at the fixed ``S_{T1} m`` (the instrument family covers the forward-moneyness
  strikes, the closest one is used);
* variance swap / vol swap: ``delta`` (the dollar-gamma exposure) + the log-contract vanilla strip
  (static, short, ``2/K²`` weights); the vol swap adds ``volga`` with the var-swap-vs-vol-swap
  spread;
* option on realised variance / vol: ``delta`` + a variance swap over the option's window sized
  on ``vega`` (its variance delta); the study compares vol-of-vol overlays;
* cliquet family: ``delta`` + the cap-call strip (static, ``q`` per period, the decomposition's 1
  by default) + a variance swap sized on ``vega`` (the net forward-variance exposure); the
  global-floor leg via the accumulated-sum put (static, from ``decompose()``).  ``q`` is a
  **strategy parameter** (owner decision (a) at the M8 acceptance): ``default_strategy(cliquet,
  ctx, q=0.5)`` — a scalar or one weight per period — is the study's "q-weighted cap calls +
  net-sized var swap"; with ``q = 1`` the replication is exact and the swap only sees regression
  noise.  The strip's instrument name carries ``q`` when it is not 1 (``"cap-call strip q=0.5"``)
  and the hedger records the preset keyword arguments in ``HedgeResult.settings["preset_kwargs"]``
  (carried on the strategy as ``preset_kwargs`` by :func:`default_strategy`);
* up / down variance, convexity spread: ``delta`` + variance swap (``vega``) + ``skew_T`` via an
  option strip + the complementary conditional variance swap where the corridor identity applies
  (static);
* knock-out variance swap: ``delta`` + variance swap + a call spread at the barrier in the solve
  (``delta``, ``gamma``, ``vega``); unwound on knock-out by the hedger;
* VKO put: ``delta`` + the underlying vanilla put (static, short one per unit notional, from
  ``decompose()``) + a variance swap on ``vega`` + ``vanna`` via a risk reversal;
* autocall / Phoenix: per leg from ``decompose()``: call spreads at each observation level
  (smart-gap shifted levels when the note carries a smart gap), the knock-in put and a digital put
  at the barrier, all in the solve on ``delta``, ``gamma``, ``vega``; ``skew_T`` via an option
  strip when available; the bond leg is deterministic (no hedge); unwound at autocall;
* barrier options: ``delta`` + the static replication where it exists (knock-in: the in–out
  parity vanilla, short) + a call spread at the barrier in the solve;
* :class:`CustomStrategy`: a user callable ``(context) -> quantities``.

Every preset is a :class:`GreekTargetStrategy`; unsupported targets in the pricing context are
dropped with a note (never silently) so that every preset runs end to end on every context.

**Static legs carry the hedge's sign.**  The hedger holds the product long and a static quantity is
added to the product's exposure before the solve, so a leg that replicates a piece of the product
is held SHORT that piece (the cliquet's cap calls, which the cliquet is short, are held long).
Until 2026-09-27 the digital's call spread, the variance strip, the VKO's underlying put and the
knock-in's parity vanilla carried the product's own sign, doubling the exposure they were meant to
cancel (hedged / unhedged P&L std 2.06 and 2.12 for the variance swap and the VKO put under monthly
Black–Scholes, recorded then as discrete-hedging noise); ``tests/test_hedging.py`` checks the sign
of every replicating static leg.
Checked by ``tests/test_hedging.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volsto.hedging.instruments import (
    AccumulatedSumPut,
    CapCallStrip,
    ConditionalVarianceSwap,
    Digital,
    ForwardStartRiskReversal,
    ForwardStartVanilla,
    ForwardVarianceSwap,
    HedgeInstrument,
    Spot,
    Vanilla,
    VarianceSwap,
    VolSwap,
    option_strip,
)
from volsto.market.curves import DiscountCurve, ForwardCurve
from volsto.products.base import Product

FloatArray = NDArray[np.float64]

#: relative ridge of the per-path solve: an instrument's quantity is shrunk where its squared
#: sensitivity falls below ``ridge`` times its cross-path mean, the LARGER of today's mean and the
#: mean at the strategy's first date — so an instrument whose sensitivity decays over the life
#: (a variance swap's vega near expiry) or crosses zero on a path (its regressed vega) cannot take
#: a noise-driven quantity (measured: the cliquet preset's var-swap quantity on the SPX marking
#: fit at 6·10³ paths reached hundreds on single paths near expiry, P&L std 11–18 ± 5–8 % of
#: notional); a well-conditioned solve is biased by ~0.1%
DEFAULT_RIDGE = 1e-3
#: an instrument's squared sensitivity below this fraction of its first-date cross-path level
#: on a path makes it dead there (quantity 0): see GreekTargetStrategy.solve
DEAD_SENSITIVITY = 1e-12
#: the §7.2 surface regimes: the delta is the CRN bump of the regime's moved surface, the leverage
#: recalibrated (:meth:`~volsto.hedging.hedger.PricingContext.regime_delta_bump`)
SURFACE_DELTA_REGIMES = ("sticky_strike", "sticky_skew", "sticky_moneyness", "sticky_local_vol")
#: the minimum-variance spot-only delta of the pricing model (owner's decision of 2026-09-16, M8b
#: study D): ``Δ_model + Σ_i (∂V/∂X_i) d⟨X_i, S⟩ / d⟨S, S⟩`` — no extra simulation, the value
#: gradients of the regression (:func:`~volsto.hedging.hedger.min_variance_delta`)
MIN_VARIANCE_REGIME = "min_variance"
DELTA_REGIMES = ("model", *SURFACE_DELTA_REGIMES, MIN_VARIANCE_REGIME)
#: default smoothing width of the digital call-spread replication (fraction of the level, §6.9)
DEFAULT_DIGITAL_WIDTH = 0.02


@dataclass(frozen=True)
class Target:
    name: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or self.weight <= 0:
            raise ValueError("a target needs a name and a positive weight")


@dataclass
class HedgeSolution:
    """Quantities ``(n_paths, n_instruments)`` and the residual exposure per target
    ``(n_paths, n_targets)`` after hedging at one date."""

    quantities: FloatArray
    residual: FloatArray
    target_names: tuple[str, ...]
    instrument_names: tuple[str, ...]


@dataclass
class GreekTargetStrategy:
    """Layer A (module docstring)."""

    targets: tuple[Target, ...]
    instruments: list[HedgeInstrument]
    static: dict[str, float | Callable[[float], float]] = field(default_factory=dict)
    ridge: float = DEFAULT_RIDGE
    turnover: float = 0.0
    delta_regime: str = "model"
    name: str = "greek-target"
    notes: list[str] = field(default_factory=list)
    _level0: dict[str, float] = field(default_factory=dict, init=False, repr=False)
    #: the keyword arguments the preset was built with (set by :func:`default_strategy`; recorded
    #: in ``HedgeResult.settings["preset_kwargs"]``)
    preset_kwargs: dict[str, Any] = field(default_factory=dict)
    #: static legs held only until the product knocks in: zero on the paths whose hedge state
    #: carries ``knocked = 1`` (a knock-in's knock-out replication, switched to the parity vanilla
    #: at the barrier; :func:`volsto.hedging.comparison.barrier_static_strategy`)
    unwind_on_knock: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.delta_regime not in DELTA_REGIMES:
            raise ValueError(f"delta_regime must be one of {DELTA_REGIMES}")
        names = [i.name for i in self.instruments]
        if len(set(names)) != len(names):
            raise ValueError(f"instrument names must be unique: {names}")
        for k in self.static:
            if k not in names:
                raise ValueError(f"static leg {k!r} is not an instrument of the strategy")
        for k in self.unwind_on_knock:
            if k not in self.static:
                raise ValueError(f"unwind_on_knock leg {k!r} is not a static leg of the strategy")
        if self.ridge < 0 or self.turnover < 0:
            raise ValueError("ridge and turnover must be non-negative")

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.targets)

    @property
    def solved_instruments(self) -> list[HedgeInstrument]:
        return [i for i in self.instruments if i.name not in self.static]

    def static_quantity(self, name: str, t: float) -> float:
        v = self.static[name]
        return float(v(t)) if callable(v) else float(v)

    def with_targets(self, targets: Sequence[Target | str]) -> GreekTargetStrategy:
        ts = tuple(Target(x) if isinstance(x, str) else x for x in targets)
        return GreekTargetStrategy(
            ts,
            list(self.instruments),
            dict(self.static),
            self.ridge,
            self.turnover,
            self.delta_regime,
            self.name,
            list(self.notes),
            dict(self.preset_kwargs),
            self.unwind_on_knock,
        )

    def without(self, *instrument_names: str) -> GreekTargetStrategy:
        """The same strategy without the named legs (the study's drop-a-leg comparison)."""
        keep = [i for i in self.instruments if i.name not in instrument_names]
        static = {k: v for k, v in self.static.items() if k not in instrument_names}
        return GreekTargetStrategy(
            self.targets,
            keep,
            static,
            self.ridge,
            self.turnover,
            self.delta_regime,
            self.name,
            list(self.notes),
            dict(self.preset_kwargs),
            tuple(k for k in self.unwind_on_knock if k in static),
        )

    def knock_mask(self, state: Any, n_paths: int) -> FloatArray:
        """``1 − knocked`` per path from the product's hedge state (ones when the state carries
        no ``knocked`` feature, or none is given)."""
        names = tuple(getattr(state, "names", ()) or ())
        if state is None or "knocked" not in names:
            return np.ones(n_paths)
        k = np.asarray(state.features[:, names.index("knocked")], dtype=np.float64)
        return np.asarray(1.0 - k, dtype=np.float64)

    def solve(
        self,
        t: float,
        product_greeks: Mapping[str, FloatArray],
        instrument_greeks: Sequence[Mapping[str, FloatArray]],
        active: Sequence[bool],
        q_prev: FloatArray | None = None,
        *,
        state: Any = None,
    ) -> HedgeSolution:
        """The quantities at ``t`` on every path (module docstring).  ``instrument_greeks[j]``
        holds the same keys as ``product_greeks`` for instrument ``j``; inactive instruments get
        zero; static legs their given quantity (times ``1 − knocked`` for the
        ``unwind_on_knock`` legs, from the product's hedge ``state``)."""
        n_paths = next(iter(product_greeks.values())).size
        names = [i.name for i in self.instruments]
        tn = self.target_names
        q = np.zeros((n_paths, len(names)))
        b = np.column_stack([np.asarray(product_greeks[g], dtype=np.float64) for g in tn])
        if b.ndim == 1:
            b = b[:, None]
        unknocked = self.knock_mask(state, n_paths) if self.unwind_on_knock else None
        for j, inst in enumerate(self.instruments):
            if inst.name in self.static and active[j]:
                q[:, j] = self.static_quantity(inst.name, t)
                if unknocked is not None and inst.name in self.unwind_on_knock:
                    q[:, j] = q[:, j] * unknocked
                for gi, g in enumerate(tn):
                    b[:, gi] = b[:, gi] + q[:, j] * np.asarray(instrument_greeks[j][g])
        solved = [
            j
            for j, inst in enumerate(self.instruments)
            if inst.name not in self.static and active[j]
        ]
        if solved and len(tn):
            w = np.array([tg.weight for tg in self.targets])
            A = np.stack(
                [
                    np.column_stack([np.asarray(instrument_greeks[j][g]) for g in tn])
                    for j in solved
                ],
                axis=2,
            )  # (n_paths, n_targets, n_solved)
            Aw = A * w[None, :, None]
            H = np.einsum("pgj,pgk->pjk", Aw, A)
            rhs = -np.einsum("pgj,pg->pj", Aw, b)
            # the ridge is relative to each instrument's cross-path mean squared sensitivity
            # (scale-free: a per-unit-notional variance swap and a vanilla differ by orders of
            # magnitude), so an instrument whose sensitivities vanish on a path (a deep-OTM
            # option) is shrunk to zero there instead of taking a noise-driven quantity
            level = np.mean(np.einsum("pjj->pj", H), axis=0)
            names_solved = [self.instruments[j].name for j in solved]
            for nm, lv in zip(names_solved, level, strict=True):
                self._level0.setdefault(nm, float(lv))
            floor = np.array(
                [
                    max(float(lv), self._level0[nm])
                    for nm, lv in zip(names_solved, level, strict=True)
                ]
            )
            reg = self.ridge * floor + self.turnover
            H = H + reg[None, :, None] * np.eye(len(solved))[None, :, :]
            if self.turnover > 0 and q_prev is not None:
                rhs = rhs + self.turnover * q_prev[:, solved]
            # scale rows/cols for conditioning; an instrument whose squared sensitivity on a
            # path is below DEAD_SENSITIVITY x its first-date level is DEAD on that path (an
            # expired-in-all-but-name call spread, a regressed vega crossing zero): it is taken
            # out of the path's system with quantity 0 instead of being scaled up by 1/sqrt(H_jj)
            # (measured: quantities of 1e137 on the autocall preset's call spreads)
            diag_h = np.einsum("pjj->pj", H)
            dead = diag_h <= DEAD_SENSITIVITY * floor[None, :]
            if dead.any():
                eye = np.eye(len(solved))
                H = np.where(dead[:, :, None] | dead[:, None, :], eye[None, :, :], H)
                rhs = np.where(dead, 0.0, rhs)
                diag_h = np.einsum("pjj->pj", H)
            d = np.sqrt(np.maximum(diag_h, 1e-300))
            Hs = H / (d[:, :, None] * d[:, None, :])
            rs = rhs / d
            try:
                xs = np.linalg.solve(Hs, rs[:, :, None])[:, :, 0]
            except np.linalg.LinAlgError:
                xs = np.stack(
                    [np.linalg.lstsq(Hs[p], rs[p], rcond=None)[0] for p in range(n_paths)]
                )
            q[:, solved] = np.where(dead, 0.0, xs / d)
            resid = b + np.einsum("pgj,pj->pg", A, q[:, solved])
        else:
            resid = b
        return HedgeSolution(q, resid, tn, tuple(names))


@dataclass
class CustomStrategy:
    """A user callable ``fn(t, product_greeks, instrument_greeks, active, q_prev) ->
    (n_paths, n_instruments)`` quantities (the RiskReport-to-quantities hook)."""

    instruments: list[HedgeInstrument]
    fn: Callable[..., FloatArray]
    targets: tuple[Target, ...] = ()
    static: dict[str, float] = field(default_factory=dict)
    name: str = "custom"
    notes: list[str] = field(default_factory=list)
    delta_regime: str = "model"

    @property
    def target_names(self) -> tuple[str, ...]:
        return tuple(t.name for t in self.targets)

    @property
    def solved_instruments(self) -> list[HedgeInstrument]:
        return list(self.instruments)

    def solve(self, t, product_greeks, instrument_greeks, active, q_prev=None, *, state=None):  # type: ignore[no-untyped-def]
        q = np.asarray(
            self.fn(t, product_greeks, instrument_greeks, active, q_prev), dtype=np.float64
        )
        names = tuple(i.name for i in self.instruments)
        n_paths = next(iter(product_greeks.values())).size
        if q.shape != (n_paths, len(names)):
            raise ValueError(
                f"the custom strategy must return ({n_paths}, {len(names)}) quantities"
            )
        tn = self.target_names
        if tn:
            b = np.column_stack([np.asarray(product_greeks[g]) for g in tn])
            for j in range(len(names)):
                for gi, g in enumerate(tn):
                    b[:, gi] += q[:, j] * np.asarray(instrument_greeks[j][g])
            resid = b
        else:
            resid = np.zeros((n_paths, 0))
        return HedgeSolution(q, resid, tn, names)


Strategy = GreekTargetStrategy | CustomStrategy


# --------------------------------------------------------------------------------------------
# Layer B — presets
# --------------------------------------------------------------------------------------------


@dataclass
class PresetContext:
    """What the presets need to build instruments: curves, the spot, a reference vol per
    ``(strike, expiry)`` (from the pricing surface when available), transaction-cost levels and
    the risk pillars for the tents."""

    forward_curve: ForwardCurve
    discount: DiscountCurve
    spot: float
    reference_vol: Callable[[float, float], float] | None = None
    cost_spot_bps: float = 0.0
    cost_vol_points: float = 0.0
    digital_width: float = DEFAULT_DIGITAL_WIDTH
    surface_targets: bool = True
    horizon: float = 100.0

    def ref(self, strike: float, expiry: float) -> float | None:
        if self.reference_vol is None:
            return None
        return float(self.reference_vol(strike, expiry))

    def vanilla(self, strike: float, expiry: float, cp: int, name: str = "") -> Vanilla:
        return Vanilla(
            strike=float(strike),
            maturity=float(expiry),
            cp=cp,
            discount=self.discount,
            cost=self.cost_vol_points,
            reference_vol=self.ref(strike, expiry),
            name=name,
        )

    def spot_instrument(self) -> Spot:
        return Spot(cost=self.cost_spot_bps, horizon=self.horizon)

    def varswap(self, t1: float, t2: float, strike_vol: float, name: str = "") -> VarianceSwap:
        vs = ForwardVarianceSwap(t1, t2, strike_vol, self.discount, cost=self.cost_vol_points)
        if name:
            vs.name = name
        return vs

    def atm_vol(self, expiry: float) -> float:
        f = float(self.forward_curve.forward(expiry))
        r = self.ref(f, expiry)
        return 0.2 if r is None else r


def _skew_targets(ctx: PresetContext, T: float) -> list[Target]:
    """``skew_T`` when the context has a surface; ``dX1`` (dropped by the hedger under a
    factor-less model) otherwise."""
    return [Target(f"skew_T:{T:g}")] if ctx.surface_targets else [Target("dX1")]


def preset_vanilla(
    product: Any, ctx: PresetContext, *, vol_hedged: bool = False
) -> GreekTargetStrategy:
    from volsto.products.vanilla import DigitalOption, EuropeanOption

    T = float(product.fixing_times[-1])
    inst: list[HedgeInstrument] = [ctx.spot_instrument()]
    targets = [Target("delta")]
    static: dict[str, float | Callable[[float], float]] = {}
    if isinstance(product, DigitalOption):
        w = ctx.digital_width * product.strike
        cs = Digital(
            strike=product.strike,
            maturity=T,
            cp=product.cp,
            width=w,
            discount=ctx.discount,
            cost=ctx.cost_vol_points,
            reference_vol=ctx.ref(product.strike, T),
            name="call spread",
        )
        inst.append(cs)
        # the digital is long the call spread: the hedge is short it
        static["call spread"] = -float(product.notional)
    if vol_hedged or (isinstance(product, EuropeanOption) and vol_hedged):
        f = float(ctx.forward_curve.forward(T))
        inst.append(ctx.vanilla(f, T, 1, "atm vanilla"))
        targets.append(Target("vega"))
    return GreekTargetStrategy(tuple(targets), inst, static, name="vanilla preset")


def preset_forward_start(
    product: Any, ctx: PresetContext, *, skew: bool = False
) -> GreekTargetStrategy:
    """Forward start / FVA ``T1 → T2``: the forward variance swap on the ``[T1, T2]`` bucket
    (targets ``delta``, ``fwd_var:T1-T2``, ``vega``) and, after ``T1``, the vanilla family struck
    at the ``T1`` fixing (:class:`ForwardStartVanilla`, moneyness 0.9 / 1.0 / 1.1 of the product's,
    puts below and calls at or above — a path-dependent strike; measured: a family at strikes
    fixed at inception hedged the post-``T1`` straddle from mismatched strikes and its gross gamma
    made the preset worse than delta only under local vol).  ``skew=True`` adds the forward risk
    reversal with the ``skew_T:T2`` target (the study's forward-skew leg): measured on the FVA
    1y → 2y under the placeholder local vol at 6·10³ pricing paths, the skew-target quantity is
    noise-dominated (risk-reversal leg P&L std 0.34 against a product std 0.16, hedged total
    0.33 vs 0.05 without it), so it is not the default; M8b sizes the budget it needs."""
    t1, t2 = float(product.T1), float(product.T2)
    k_vol = ctx.atm_vol(t2)
    inst: list[HedgeInstrument] = [
        ctx.spot_instrument(),
        ctx.varswap(t1, t2, k_vol, name="fwd var swap"),
    ]
    targets = [Target("delta"), Target(f"fwd_var:{t1:g}-{t2:g}")]
    if skew:
        inst.append(
            ForwardStartRiskReversal(
                t1=t1,
                t2=t2,
                discount=ctx.discount,
                cost=ctx.cost_vol_points,
                reference_vol=k_vol,
                name="fwd risk reversal",
            )
        )
        targets += _skew_targets(ctx, t2)
    targets.append(Target("vega"))
    m = float(getattr(product, "strike", getattr(product, "moneyness", 1.0)))
    for mult in (0.9, 1.0, 1.1):
        inst.append(
            ForwardStartVanilla(
                t1=t1,
                t2=t2,
                moneyness=m * mult,
                cp=1 if mult >= 1.0 else -1,
                discount=ctx.discount,
                cost=ctx.cost_vol_points,
                reference_vol=k_vol,
                name=f"vanilla struck at T1 x{mult:g}",
            )
        )
    s = GreekTargetStrategy(
        tuple(targets), inst, name="forward-start preset" + (" + skew" if skew else "")
    )
    s.notes.append("delta before T1 is reported (about zero for a forward start)")
    if skew:
        s.notes.append(
            "forward risk reversal on the skew_T target: noise-dominated at 6e3 pricing paths "
            "(measured), size the budget"
        )
    return s


def preset_variance_swap(product: Any, ctx: PresetContext) -> GreekTargetStrategy:
    from volsto.products.variance import VolSwap as _Vol

    T = float(product.maturity)
    f = float(ctx.forward_curve.forward(T))
    strikes = [f * m for m in (0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3)]
    ref_vols: dict[tuple[float, float], float] | None = None
    if ctx.reference_vol is not None:
        ref_vols = {(k, T): float(ctx.reference_vol(k, T)) for k in strikes}
    strip = option_strip(
        strikes,
        [T],
        ctx.discount,
        forward=ctx.forward_curve,
        cost=ctx.cost_vol_points,
        reference_vols=ref_vols,
    )
    inst: list[HedgeInstrument] = [ctx.spot_instrument(), *strip]
    static: dict[str, float | Callable[[float], float]] = {}
    # log-contract weights 2/K^2 dK per unit variance notional (annualised / T): the swap is long
    # the strip, the hedge is short it
    dk = np.diff(strikes).mean()
    for v in strip:
        static[v.name] = -float(product.notional) * 2.0 / (v.strike**2) * dk / T
    targets = [Target("delta")]
    if isinstance(product, _Vol):
        k_vol = ctx.atm_vol(T)
        inst.append(ctx.varswap(float(product.start), T, k_vol, name="var swap"))
        inst.append(
            VolSwap(
                start_time=float(product.start),
                end=T,
                strike_vol=k_vol,
                discount=ctx.discount,
                cost=ctx.cost_vol_points,
                name="vol swap",
            )
        )
        targets += [Target("vega"), Target("volga")]
    return GreekTargetStrategy(tuple(targets), inst, static, name="variance preset")


def cap_call_strip_name(q: Sequence[float] | float) -> str:
    """``"cap-call strip"`` for the decomposition's ``q = 1``, else the name carrying ``q``
    (``"cap-call strip q=0.5"``; per-period weights listed)."""
    arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
    if np.all(arr == 1.0):
        return "cap-call strip"
    if arr.size == 1 or np.all(arr == arr[0]):
        return f"cap-call strip q={float(arr[0]):g}"
    return "cap-call strip q=[" + ",".join(f"{w:g}" for w in arr) + "]"


def preset_variance_option(product: Any, ctx: PresetContext) -> GreekTargetStrategy:
    """Option on realised variance or vol (module docstring): ``delta`` + a variance swap over the
    option's window sized on ``vega`` — the option's first-order exposure to the level of implied
    variance; its vol-of-vol (convexity in variance) is left open by this preset."""
    T = float(product.maturity)
    inst: list[HedgeInstrument] = [
        ctx.spot_instrument(),
        ctx.varswap(float(product.start), T, ctx.atm_vol(T), name="var swap"),
    ]
    return GreekTargetStrategy(
        (Target("delta"), Target("vega")), inst, name="variance-option preset"
    )


def preset_cliquet(
    product: Any, ctx: PresetContext, *, q: Sequence[float] | float = 1.0
) -> GreekTargetStrategy:
    """Cliquet family (module docstring): ``delta`` + the cap-call strip weighted ``q`` (static,
    long, one weight per period or a scalar; the decomposition's exact replication at the default
    ``q = 1``) + the accumulated-sum put (static, short) when the cliquet has a global floor + a
    variance swap sized on the **net** ``vega`` — the study's "q-weighted cap calls + net-sized var
    swap" for ``q < 1``.  The strip's name carries ``q`` when it is not 1
    (:func:`cap_call_strip_name`)."""
    from volsto.products.cliquet import AdditiveCliquet, ReverseCliquet

    cl = (
        product
        if isinstance(product, AdditiveCliquet)
        else product._inner() if isinstance(product, ReverseCliquet) else None
    )
    T = float(product.maturity)
    inst: list[HedgeInstrument] = [ctx.spot_instrument()]
    static: dict[str, float | Callable[[float], float]] = {}
    if cl is not None and np.isfinite(cl.local_cap):
        strip = CapCallStrip(
            cliquet=cl,
            q=q,
            cost=ctx.cost_vol_points,
            reference_vol=ctx.atm_vol(T),
            name=cap_call_strip_name(q),
        )
        inst.append(strip)
        # the cliquet is SHORT the cap calls of its decomposition: the hedge is long them
        static[strip.name] = float(product.notional)
    if cl is not None and np.isfinite(cl.global_floor):
        put = AccumulatedSumPut(cliquet=cl, cost=ctx.cost_vol_points, reference_vol=ctx.atm_vol(T))
        inst.append(put)
        # the cliquet is LONG the global-floor put: the hedge is short it
        static[put.name] = -float(product.notional)
    inst.append(ctx.varswap(0.0, T, ctx.atm_vol(T), name="var swap"))
    targets = [Target("delta"), Target("vega")]
    return GreekTargetStrategy(tuple(targets), inst, static, name="cliquet preset")


def preset_conditional_variance(product: Any, ctx: PresetContext) -> GreekTargetStrategy:
    from volsto.products.conditional_variance import ConditionalVarianceSwap as CondVarProduct
    from volsto.products.conditional_variance import ConvexitySpread

    T = float(product.maturity)
    f = float(ctx.forward_curve.forward(T))
    inst: list[HedgeInstrument] = [
        ctx.spot_instrument(),
        ctx.varswap(0.0, T, ctx.atm_vol(T), name="var swap"),
    ]
    strikes = [f * m for m in (0.8, 0.9, 1.0, 1.1, 1.2)]
    inst += option_strip(
        strikes, [T], ctx.discount, forward=ctx.forward_curve, cost=ctx.cost_vol_points
    )
    static: dict[str, float | Callable[[float], float]] = {}
    targets = [Target("delta"), Target("vega"), *_skew_targets(ctx, T)]
    cv = product if isinstance(product, CondVarProduct) else None
    if cv is not None and cv.convention == "corridor":
        other = "down" if cv.side == "up" else "up"
        comp = ConditionalVarianceSwap(
            fixings=cv.fixing_times,
            barrier=cv.barrier,
            side=other,
            strike_vol=cv.strike_vol,
            convention="corridor",
            indicator=cv.indicator,
            discount=ctx.discount,
            cost=ctx.cost_vol_points,
            name="complementary corridor",
        )
        inst.append(comp)
        static["complementary corridor"] = float(product.notional)  # up + down = full variance
    elif isinstance(product, ConvexitySpread):
        pass
    return GreekTargetStrategy(tuple(targets), inst, static, name="conditional-variance preset")


def preset_ko_variance(product: Any, ctx: PresetContext) -> GreekTargetStrategy:
    T = float(product.maturity)
    b = float(product.barrier)
    inst: list[HedgeInstrument] = [
        ctx.spot_instrument(),
        ctx.varswap(0.0, T, ctx.atm_vol(T), name="var swap"),
        Digital(
            strike=b,
            maturity=T,
            cp=1 if product.direction == "up" else -1,
            width=ctx.digital_width * b,
            discount=ctx.discount,
            cost=ctx.cost_vol_points,
            reference_vol=ctx.ref(b, T),
            name="barrier call spread",
        ),
    ]
    targets = [Target("delta"), Target("gamma"), Target("vega")]
    return GreekTargetStrategy(tuple(targets), inst, name="knock-out variance preset")


def preset_vko(product: Any, ctx: PresetContext) -> GreekTargetStrategy:
    T = float(product.T)
    k = float(product.strike)
    put = ctx.vanilla(k, T, -1, "underlying put")
    f = float(ctx.forward_curve.forward(T))
    rr_call = ctx.vanilla(1.1 * f, T, 1, "rr call")
    rr_put = ctx.vanilla(0.9 * f, T, -1, "rr put")
    inst: list[HedgeInstrument] = [
        ctx.spot_instrument(),
        put,
        ctx.varswap(0.0, T, ctx.atm_vol(T), name="var swap"),
        rr_call,
        rr_put,
    ]
    # the VKO is long the underlying put (put − vol-knock-in put): the hedge is short it
    static: dict[str, float | Callable[[float], float]] = {
        "underlying put": -float(product.notional)
    }
    targets = [Target("delta"), Target("vega"), Target("vanna")]
    return GreekTargetStrategy(tuple(targets), inst, static, name="VKO preset")


def preset_autocall(product: Any, ctx: PresetContext) -> GreekTargetStrategy:
    obs = np.asarray(product.observation_times, dtype=np.float64)
    # Autocall.autocall_levels and .ki_barrier are ABSOLUTE levels (the product scales its
    # fractions by spot_reference itself; measured: scaling again put the call spreads at
    # spot x level and the barrier put deep in the money, value 5553 on a spot of 100)
    levels = np.asarray(product.autocall_levels, dtype=np.float64)
    inst: list[HedgeInstrument] = [ctx.spot_instrument()]
    for i, (t_i, lvl) in enumerate(zip(obs, levels, strict=True), start=1):
        inst.append(
            Digital(
                strike=float(lvl),
                maturity=float(t_i),
                cp=1,
                width=ctx.digital_width * float(lvl),
                discount=ctx.discount,
                cost=ctx.cost_vol_points,
                reference_vol=ctx.ref(float(lvl), float(t_i)),
                name=f"call spread AC{i}",
            )
        )
    T = float(obs[-1])
    b = float(product.ki_barrier)
    inst.append(ctx.vanilla(b, T, -1, "put at barrier"))
    inst.append(
        Digital(
            strike=b,
            maturity=T,
            cp=-1,
            width=ctx.digital_width * b,
            discount=ctx.discount,
            cost=ctx.cost_vol_points,
            reference_vol=ctx.ref(b, T),
            name="digital put at barrier",
        )
    )
    inst.append(ctx.varswap(0.0, T, ctx.atm_vol(T), name="var swap"))
    targets = [Target("delta"), Target("gamma"), Target("vega")]
    if ctx.surface_targets:
        f = float(ctx.forward_curve.forward(T))
        inst += option_strip(
            [0.8 * f, 0.9 * f, 1.1 * f],
            [T],
            ctx.discount,
            forward=ctx.forward_curve,
            cost=ctx.cost_vol_points,
        )
        targets.append(Target(f"skew_T:{T:g}"))
    s = GreekTargetStrategy(tuple(targets), inst, name="autocall preset")
    s.notes.append(
        "the bond leg is deterministic: no hedge; smart-gap shifted levels are not applied to "
        "the call spreads"
    )
    return s


def preset_barrier(product: Any, ctx: PresetContext) -> GreekTargetStrategy:
    from volsto.products.barrier import KnockInOption

    T = float(product.T)
    b = float(product.barrier)
    inst: list[HedgeInstrument] = [ctx.spot_instrument()]
    static: dict[str, float | Callable[[float], float]] = {}
    if isinstance(product, KnockInOption):
        van = ctx.vanilla(float(product.strike), T, int(product.cp), "parity vanilla")
        inst.append(van)
        # knock-in = vanilla − knock-out: the hedge is short the parity vanilla
        static["parity vanilla"] = -float(product.notional)
    up = product.direction == "up"
    inst.append(
        Digital(
            strike=b,
            maturity=T,
            cp=1 if up else -1,
            width=ctx.digital_width * b,
            discount=ctx.discount,
            cost=ctx.cost_vol_points,
            reference_vol=ctx.ref(b, T),
            name="barrier call spread",
        )
    )
    targets = [Target("delta"), Target("gamma")]
    return GreekTargetStrategy(tuple(targets), inst, static, name="barrier preset")


def preset_portfolio(product: Any, ctx: PresetContext) -> GreekTargetStrategy:
    legs = [default_strategy(leg, ctx) for leg in product.legs]
    inst: list[HedgeInstrument] = []
    names: set[str] = set()
    static: dict[str, float | Callable[[float], float]] = {}
    targets: list[Target] = []
    for w, s in zip(product.weights, legs, strict=True):
        for i in s.instruments:
            if i.name in names:
                continue
            names.add(i.name)
            inst.append(i)
        for k, v in s.static.items():
            static[k] = float(w) * (v if isinstance(v, float) else v(0.0))
        for tg in s.targets:
            if tg.name not in {x.name for x in targets}:
                targets.append(tg)
    return GreekTargetStrategy(tuple(targets), inst, static, name="portfolio preset")


PRESETS: dict[str, Callable[..., GreekTargetStrategy]] = {
    "EuropeanOption": preset_vanilla,
    "DigitalOption": preset_vanilla,
    "ForwardStartOption": preset_forward_start,
    "ForwardStartStraddle": preset_forward_start,
    "FVA": preset_forward_start,
    "VarianceSwap": preset_variance_swap,
    "VolSwap": preset_variance_swap,
    "VarianceOption": preset_variance_option,
    "AdditiveCliquet": preset_cliquet,
    "ReverseCliquet": preset_cliquet,
    "Napoleon": preset_cliquet,
    "ConditionalVarianceSwap": preset_conditional_variance,
    "ConvexitySpread": preset_conditional_variance,
    "KnockOutVarianceSwap": preset_ko_variance,
    "VolKnockOutPut": preset_vko,
    "Autocall": preset_autocall,
    "KnockOutOption": preset_barrier,
    "KnockInOption": preset_barrier,
    "Portfolio": preset_portfolio,
}


def default_strategy(product: Product, ctx: PresetContext, **kwargs: Any) -> GreekTargetStrategy:
    """The registered preset of the product's class (module docstring) built with ``kwargs`` —
    the preset's strategy parameters (the cliquet family's ``q``, the forward start's ``skew``,
    the vanilla's ``vol_hedged``, …), recorded on the strategy as ``preset_kwargs`` and by the
    hedger in ``HedgeResult.settings``; a class without a preset gets the vanilla preset (delta
    only) with a note."""
    for cls in type(product).__mro__:
        fn = PRESETS.get(cls.__name__)
        if fn is not None:
            s = fn(product, ctx, **kwargs)
            s.preset_kwargs = dict(kwargs)
            return s
    s = preset_vanilla(product, ctx)
    s.notes.append(f"no preset registered for {type(product).__name__}: delta only")
    return s


__all__ = [
    "DEFAULT_DIGITAL_WIDTH",
    "DELTA_REGIMES",
    "MIN_VARIANCE_REGIME",
    "PRESETS",
    "SURFACE_DELTA_REGIMES",
    "CustomStrategy",
    "GreekTargetStrategy",
    "HedgeSolution",
    "PresetContext",
    "Strategy",
    "Target",
    "cap_call_strip_name",
    "default_strategy",
    "preset_autocall",
    "preset_barrier",
    "preset_cliquet",
    "preset_conditional_variance",
    "preset_forward_start",
    "preset_ko_variance",
    "preset_portfolio",
    "preset_vanilla",
    "preset_variance_option",
    "preset_variance_swap",
    "preset_vko",
]
