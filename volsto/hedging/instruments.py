"""Hedge instruments (SPEC §8, M8 Part 2): thin wrappers around the library's products that
carry a transaction-cost rule, an activity window and a roll rule, priced under the pricing model
by the regression engine of :mod:`volsto.hedging.pricing` — the same engine, grid and draws as
the product they hedge.

**Units and accounting** (time-0 money, :mod:`volsto.hedging.pricing`): one unit of an
instrument is one unit of its product's notional; its conditional value is the regressed
``E[discounted payoff | state]``.  :class:`Spot` is the total-return asset normalised to one share
at ``t = 0``: value ``S_t · DF_r(t) / DF_q(t) = S₀ S_t / F(t)`` (a martingale in time-0 money),
delta ``S₀ / F(t)`` per unit spot, no other sensitivity — analytic, never regressed.

**Transaction costs** (half-spread on the traded quantity ``|Δq|``, time-0 money; a zero-cost run
is always reported next to the costed one by the hedger):

* ``Spot``: ``bps × 1e-4 × |Δq| × S_t DF_r/DF_q`` (bps of the traded value);
* vanillas, digitals / call spreads, strips, forward-start straddles, cap-call strips: ``vol_points
  × 0.01 × |Δq| × vega``, the Black vega of the leg at the state (forward moneyness at the
  rebalancing date, remaining maturity, the instrument's ``reference_vol`` — the pricing surface's
  implied vol of the strike at inception when the hedger has a surface, else 20% with a note);
* variance-type swaps: ``vol_points × 0.01 × 2 K_vol × |Δq| × DF(T) × remaining fraction`` of the
  variance window (``dVar/dσ = 2σ``); vol swaps ``vol_points × 0.01 × |Δq| × DF(T) × remaining``;
* static portfolios (the replications of :mod:`volsto.hedging.replication`): ``vol_points × 0.01
  × |Δq| × |net vega|``, the package quoted as one structure (:class:`StaticPortfolio`).

**Roll rules.**  ``"fixed"`` (default): the product is dated at inception and dies at its expiry.
``"constant_maturity"``: a new instrument of the same tenor and forward moneyness is opened every
``roll_every`` years (strikes at the *forward* moneyness of the roll date — deterministic, so the
whole family is priced on one path set; spot-ATM rolling would need path-dependent strikes and is
not offered), the old one closed at the roll; :func:`expand_rolls` materialises the family.

**Activity.**  An instrument is active on ``[start, expiry)``; the hedger forces its quantity to
zero outside (and after the hedged product terminates).  A forward-variance swap on ``[T1, T2]`` is
active to ``T2``, a cap-call strip to the cliquet's maturity, a forward-start straddle to ``T2``.

Checked by ``tests/test_hedging.py`` (``test_instruments_products_and_costs``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from numpy.typing import NDArray

from volsto.market.bs import black_vega
from volsto.market.curves import DiscountCurve, ForwardCurve
from volsto.products.base import Portfolio, Product
from volsto.products.cliquet import AccumulatedSumOption, AdditiveCliquet
from volsto.products.conditional_variance import ConditionalVarianceSwap as _CondVar
from volsto.products.forward_start import ForwardStartOption
from volsto.products.forward_start import ForwardStartOption as _FwdOption
from volsto.products.forward_start import ForwardStartStraddle as _FwdStraddle
from volsto.products.vanilla import DigitalOption, EuropeanOption
from volsto.products.variance import VarianceSwap as _VarSwap
from volsto.products.variance import VolSwap as _VolSwap

FloatArray = NDArray[np.float64]
_TOL = 1e-9
DEFAULT_REFERENCE_VOL = 0.20


@dataclass
class HedgeInstrument(ABC):
    """One hedge instrument (module docstring)."""

    name: str
    cost: float = 0.0
    start: float = 0.0
    roll: str = "fixed"
    roll_every: float = 1.0 / 12.0
    reference_vol: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    @abstractmethod
    def product(self) -> Product | None:
        """The priced product (``None`` for the analytic spot)."""

    @property
    @abstractmethod
    def expiry(self) -> float:
        """The instrument dies (its value is realised) at this time."""

    def active(self, t: float) -> bool:
        return self.start - _TOL <= t < self.expiry - _TOL

    @abstractmethod
    def transaction_cost(
        self, dq: FloatArray, t: float, spot: FloatArray, fc: ForwardCurve, discount: DiscountCurve
    ) -> FloatArray:
        """Half-spread cost of trading ``dq`` units at ``t`` on paths with spot ``spot``."""

    def spawn(self, t: float, fc: ForwardCurve) -> HedgeInstrument:
        """The same instrument re-struck at ``t`` (constant-maturity roll)."""
        raise NotImplementedError(f"{type(self).__name__} does not roll")

    def _ref_vol(self) -> float:
        if self.reference_vol is None:
            note = (
                f"{self.name}: cost vega at the default reference vol {DEFAULT_REFERENCE_VOL:.0%}"
            )
            if note not in self.notes:
                self.notes.append(note)
            return DEFAULT_REFERENCE_VOL
        return float(self.reference_vol)


def _vega_cost(
    inst: HedgeInstrument,
    strike_abs: float,
    expiry: float,
    dq: FloatArray,
    t: float,
    spot: FloatArray,
    fc: ForwardCurve,
    discount: DiscountCurve,
    scale: float = 1.0,
) -> FloatArray:
    tau = max(expiry - t, 1e-6)
    f_t = np.asarray(spot, dtype=np.float64) * float(fc.forward(expiry) / fc.forward(t))
    vega = black_vega(f_t, strike_abs, tau, inst._ref_vol(), float(discount.df(expiry)))
    return np.asarray(inst.cost * 0.01 * np.abs(dq) * np.abs(vega) * scale, dtype=np.float64)


# --------------------------------------------------------------------------------------------
# spot
# --------------------------------------------------------------------------------------------


@dataclass
class Spot(HedgeInstrument):
    """The total-return asset (module docstring); ``cost`` in bps of the traded value."""

    name: str = "spot"
    horizon: float = 100.0

    @property
    def product(self) -> Product | None:
        return None

    @property
    def expiry(self) -> float:
        return self.horizon

    def scale(self, t: float, fc: ForwardCurve) -> float:
        """``S₀ / F(t)``: the time-0-money value of one unit per unit of ``S_t``."""
        return float(fc.spot / fc.forward(t)) if t > 0 else 1.0

    def transaction_cost(
        self, dq: FloatArray, t: float, spot: FloatArray, fc: ForwardCurve, discount: DiscountCurve
    ) -> FloatArray:
        return np.asarray(
            self.cost * 1e-4 * np.abs(dq) * spot * self.scale(t, fc), dtype=np.float64
        )


# --------------------------------------------------------------------------------------------
# options
# --------------------------------------------------------------------------------------------


@dataclass
class Vanilla(HedgeInstrument):
    """European option ``(strike, expiry, cp)``; ``cost`` in vol points."""

    strike: float = 100.0
    maturity: float = 1.0
    cp: int = 1
    discount: DiscountCurve | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{'call' if self.cp > 0 else 'put'} K={self.strike:g} T={self.maturity:g}"

    @property
    def product(self) -> Product:
        assert self.discount is not None
        return EuropeanOption(self.strike, self.maturity, self.cp, self.discount)

    @property
    def expiry(self) -> float:
        return self.maturity

    def transaction_cost(self, dq, t, spot, fc, discount):  # type: ignore[no-untyped-def]
        return _vega_cost(self, self.strike, self.maturity, dq, t, spot, fc, discount)

    def spawn(self, t: float, fc: ForwardCurve) -> HedgeInstrument:
        tenor = self.maturity - self.start
        m = self.strike / float(fc.forward(self.maturity))  # forward moneyness
        new_T = t + tenor
        return Vanilla(
            strike=float(m * fc.forward(new_T)),
            maturity=new_T,
            cp=self.cp,
            discount=self.discount,
            cost=self.cost,
            start=t,
            roll=self.roll,
            roll_every=self.roll_every,
            reference_vol=self.reference_vol,
        )


@dataclass
class Digital(HedgeInstrument):
    """Cash-or-nothing digital ``(strike, expiry, cp)`` or, with ``width > 0``, its call-spread
    replication ``[V(K − w/2) − V(K + w/2)] / w`` (the barrier-digital hedge); ``cost`` in vol
    points on the legs."""

    strike: float = 100.0
    maturity: float = 1.0
    cp: int = 1
    width: float = 0.0
    discount: DiscountCurve | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            kind = "call spread" if self.width > 0 else "digital"
            self.name = f"{kind} K={self.strike:g} T={self.maturity:g}"

    @property
    def product(self) -> Product:
        assert self.discount is not None
        if self.width <= 0:
            return DigitalOption(self.strike, self.maturity, self.cp, self.discount)
        lo, hi = self.strike - 0.5 * self.width, self.strike + 0.5 * self.width
        legs = [
            EuropeanOption(lo, self.maturity, self.cp, self.discount),
            EuropeanOption(hi, self.maturity, self.cp, self.discount),
        ]
        w = 1.0 / self.width
        return Portfolio(legs, [w * self.cp, -w * self.cp])

    @property
    def expiry(self) -> float:
        return self.maturity

    def transaction_cost(self, dq, t, spot, fc, discount):  # type: ignore[no-untyped-def]
        scale = 2.0 / self.width if self.width > 0 else 1.0
        return _vega_cost(self, self.strike, self.maturity, dq, t, spot, fc, discount, scale)


def option_strip(
    strikes: Sequence[float],
    expiries: Sequence[float],
    discount: DiscountCurve,
    *,
    forward: ForwardCurve | None = None,
    cost: float = 0.0,
    reference_vols: dict[tuple[float, float], float] | None = None,
) -> list[Vanilla]:
    """A generic option strip for skew hedges: out-of-the-money vanillas on every ``(strike,
    expiry)`` pair (puts below the forward when ``forward`` is given, else below the first
    strike's midpoint)."""
    out = []
    for T in expiries:
        f = float(forward.forward(T)) if forward is not None else float(np.median(strikes))
        for k in strikes:
            cp = -1 if k < f else 1
            rv = None if reference_vols is None else reference_vols.get((float(k), float(T)))
            out.append(
                Vanilla(
                    strike=float(k),
                    maturity=float(T),
                    cp=cp,
                    discount=discount,
                    cost=cost,
                    reference_vol=rv,
                )
            )
    return out


@dataclass
class ForwardStartStraddle(HedgeInstrument):
    """Forward-start straddle ``T1 → T2`` at moneyness ``m``; ``cost`` in vol points (Black vega of
    the straddle on the ratio, scaled by the forward)."""

    t1: float = 1.0
    t2: float = 2.0
    moneyness: float = 1.0
    discount: DiscountCurve | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"fwd straddle {self.t1:g}->{self.t2:g} m={self.moneyness:g}"

    @property
    def product(self) -> Product:
        assert self.discount is not None
        return _FwdStraddle(self.t1, self.t2, self.moneyness, self.discount)

    @property
    def expiry(self) -> float:
        return self.t2

    def transaction_cost(self, dq, t, spot, fc, discount):  # type: ignore[no-untyped-def]
        tau = max(self.t2 - max(t, self.t1), 1e-6)
        vega = float(
            black_vega(
                self.moneyness, self.moneyness, tau, self._ref_vol(), float(discount.df(self.t2))
            )
        )
        return np.asarray(self.cost * 0.01 * np.abs(dq) * 2.0 * abs(vega) * np.ones_like(spot))


@dataclass
class ForwardStartVanilla(HedgeInstrument):
    """A vanilla **struck at the ``T1`` fixing** (``S_T1 · m``, expiry ``T2``, ``cp`` ±1) and held
    from ``T1`` on: the desk's near-the-money strikes once a forward start has fixed — a
    path-dependent strike, so it is represented by a forward-start option whose activity window
    starts at ``T1`` (before ``T1`` it is not held).  ``cost`` in vol points (Black vega on the
    ratio)."""

    t1: float = 1.0
    t2: float = 2.0
    moneyness: float = 1.0
    cp: int = 1
    discount: DiscountCurve | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            kind = "call" if self.cp > 0 else "put"
            self.name = f"{kind} struck at T1 m={self.moneyness:g} ({self.t1:g}->{self.t2:g})"
        if self.start < self.t1:
            self.start = self.t1

    @property
    def product(self) -> Product:
        assert self.discount is not None
        return _FwdOption(self.t1, self.t2, self.moneyness, self.cp, self.discount)

    @property
    def expiry(self) -> float:
        return self.t2

    def transaction_cost(self, dq, t, spot, fc, discount):  # type: ignore[no-untyped-def]
        tau = max(self.t2 - max(t, self.t1), 1e-6)
        vega = float(
            black_vega(1.0, self.moneyness, tau, self._ref_vol(), float(discount.df(self.t2)))
        )
        return np.asarray(self.cost * 0.01 * np.abs(dq) * abs(vega) * np.ones_like(spot))


@dataclass
class ForwardStartRiskReversal(HedgeInstrument):
    """Forward-start risk reversal ``T1 → T2``: long the call at ``m_call``, short the put at
    ``m_put`` (the forward-skew hedge)."""

    t1: float = 1.0
    t2: float = 2.0
    m_put: float = 0.9
    m_call: float = 1.1
    discount: DiscountCurve | None = None
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"fwd risk reversal {self.t1:g}->{self.t2:g} {self.m_put:g}/{self.m_call:g}"

    @property
    def product(self) -> Product:
        assert self.discount is not None
        return Portfolio(
            [
                ForwardStartOption(self.t1, self.t2, self.m_call, 1, self.discount),
                ForwardStartOption(self.t1, self.t2, self.m_put, -1, self.discount),
            ],
            [1.0, -1.0],
        )

    @property
    def expiry(self) -> float:
        return self.t2

    def transaction_cost(self, dq, t, spot, fc, discount):  # type: ignore[no-untyped-def]
        tau = max(self.t2 - max(t, self.t1), 1e-6)
        df = float(discount.df(self.t2))
        v = float(black_vega(1.0, self.m_call, tau, self._ref_vol(), df)) + float(
            black_vega(1.0, self.m_put, tau, self._ref_vol(), df)
        )
        return np.asarray(self.cost * 0.01 * np.abs(dq) * abs(v) * np.ones_like(spot))


@dataclass
class CapCallStrip(HedgeInstrument):
    """The cliquet's local-cap replication: per period the forward-start call struck at the
    local cap ``1 + LC`` (weights ``q``, one per period; the decomposition's static replication
    has ``q = 1``), paid at the cliquet's maturity; ``cost`` in vol points per leg."""

    cliquet: AdditiveCliquet | None = None
    q: Sequence[float] | float = 1.0
    name: str = "cap-call strip"

    @property
    def product(self) -> Product:
        assert self.cliquet is not None
        c = self.cliquet
        fx = c.fixing_times
        legs = []
        for t1, t2 in pairwise(fx):
            legs.append(
                ForwardStartOption(
                    float(t1), float(t2), 1.0 + c.local_cap, 1, c.discount, 1.0, c.maturity
                )
            )
        q = np.broadcast_to(np.asarray(self.q, dtype=np.float64), (len(legs),))
        return Portfolio(legs, [float(w) for w in q])

    @property
    def expiry(self) -> float:
        assert self.cliquet is not None
        return self.cliquet.maturity

    def transaction_cost(self, dq, t, spot, fc, discount):  # type: ignore[no-untyped-def]
        assert self.cliquet is not None
        c = self.cliquet
        fx = c.fixing_times
        live = [(t1, t2) for t1, t2 in pairwise(fx) if t2 > t + _TOL]
        v = 0.0
        for t1, t2 in live:
            tau = max(t2 - max(t, t1), 1e-6)
            v += float(
                abs(
                    black_vega(
                        1.0, 1.0 + c.local_cap, tau, self._ref_vol(), float(discount.df(c.maturity))
                    )
                )
            )
        return np.asarray(self.cost * 0.01 * np.abs(dq) * v * np.ones_like(spot))


@dataclass
class AccumulatedSumPut(HedgeInstrument):
    """The cliquet's global-floor leg as a hedge instrument (``AccumulatedSumOption``, a put on
    the accumulated local sum struck at the global floor)."""

    cliquet: AdditiveCliquet | None = None
    strike: float | None = None
    name: str = "accumulated-sum put"

    @property
    def product(self) -> Product:
        assert self.cliquet is not None
        k = self.cliquet.global_floor if self.strike is None else self.strike
        return AccumulatedSumOption(self.cliquet, float(k), -1)

    @property
    def expiry(self) -> float:
        assert self.cliquet is not None
        return self.cliquet.maturity

    def transaction_cost(self, dq, t, spot, fc, discount):  # type: ignore[no-untyped-def]
        assert self.cliquet is not None
        tau = max(self.cliquet.maturity - t, 1e-6)
        v = float(
            abs(
                black_vega(
                    1.0, 1.0, tau, self._ref_vol(), float(discount.df(self.cliquet.maturity))
                )
            )
        )
        return np.asarray(self.cost * 0.01 * np.abs(dq) * v * np.ones_like(spot))


@dataclass
class StaticPortfolio(HedgeInstrument):
    """A weighted book of same-expiry European options traded as one leg: the static
    replications of :mod:`volsto.hedging.replication` (a barrier's put-call-symmetry portfolio, a
    corridor's ``2/K²`` strip, a knock-out variance swap's stopped log contract).  ``strikes``,
    ``weights`` (units of each option per unit of the leg) and ``cps`` (+1 call, −1 put) are
    parallel; one unit of the leg pays ``Σ_i w_i (cp_i (S_T − K_i))⁺`` at ``maturity``.  ``cost``
    in vol points on the package's **net** vega ``|Σ_i w_i vega_i|`` per unit traded: the package
    is quoted as one structure, its legs' vegas offsetting — leg-by-leg half-spreads would charge
    a digital spread ``2/width`` times a vanilla's (measured: 6.3% of the spot on the 6m 90%
    down-and-out put's replication at 0.25 vol point, against a hedged P&L std of 0.85%)."""

    strikes: tuple[float, ...] = ()
    weights: tuple[float, ...] = ()
    cps: tuple[int, ...] = ()
    maturity: float = 1.0
    discount: DiscountCurve | None = None
    name: str = "static portfolio"

    def __post_init__(self) -> None:
        self.strikes = tuple(float(k) for k in self.strikes)
        self.weights = tuple(float(w) for w in self.weights)
        self.cps = tuple(int(c) for c in self.cps)
        if not self.strikes or not (len(self.strikes) == len(self.weights) == len(self.cps)):
            raise ValueError("a static portfolio needs parallel, non-empty strikes/weights/cps")
        if any(k <= 0 or not np.isfinite(k) for k in self.strikes):
            raise ValueError("strikes must be positive and finite")
        if any(c not in (-1, 1) for c in self.cps):
            raise ValueError("cps must be +1 (call) or -1 (put)")

    @property
    def product(self) -> Product:
        assert self.discount is not None
        legs = [
            EuropeanOption(k, self.maturity, c, self.discount)
            for k, c in zip(self.strikes, self.cps, strict=True)
        ]
        return Portfolio(legs, list(self.weights))

    @property
    def expiry(self) -> float:
        return self.maturity

    def terminal_payoff(self, spot: FloatArray) -> FloatArray:
        """``Σ_i w_i (cp_i (S − K_i))⁺`` at the expiry, per unit of the leg (the replication
        checks)."""
        s = np.asarray(spot, dtype=np.float64)[..., None]
        k = np.asarray(self.strikes)
        c = np.asarray(self.cps, dtype=np.float64)
        return np.asarray(
            np.sum(np.asarray(self.weights) * np.maximum(c * (s - k), 0.0), axis=-1),
            dtype=np.float64,
        )

    def transaction_cost(self, dq, t, spot, fc, discount):  # type: ignore[no-untyped-def]
        tau = max(self.maturity - t, 1e-6)
        f_t = np.asarray(spot, dtype=np.float64) * float(fc.forward(self.maturity) / fc.forward(t))
        df = float(discount.df(self.maturity))
        net = np.zeros_like(f_t)
        for k, w in zip(self.strikes, self.weights, strict=True):
            net = net + w * black_vega(f_t, k, tau, self._ref_vol(), df)
        return np.asarray(self.cost * 0.01 * np.abs(dq) * np.abs(net), dtype=np.float64)


# --------------------------------------------------------------------------------------------
# variance-type swaps
# --------------------------------------------------------------------------------------------


def _remaining(fixings: FloatArray, t: float) -> float:
    n = fixings.size - 1
    done = int(np.sum(fixings[1:] <= t + _TOL))
    return max(n - done, 0) / max(n, 1)


@dataclass
class VarianceSwap(HedgeInstrument):
    """Variance swap on ``fixings`` (daily by default over ``[start_time, end]``) at ``strike_vol``;
    ``cost`` in vol points."""

    start_time: float = 0.0
    end: float = 1.0
    strike_vol: float = 0.2
    fixings: FloatArray | None = None
    discount: DiscountCurve | None = None
    per_year: int = 252
    name: str = ""

    def __post_init__(self) -> None:
        if self.fixings is None:
            n = max(1, round((self.end - self.start_time) * self.per_year))
            self.fixings = np.linspace(self.start_time, self.end, n + 1)
        if not self.name:
            self.name = f"var swap {self.start_time:g}->{self.end:g} K={self.strike_vol:g}"

    @property
    def product(self) -> Product:
        assert self.discount is not None and self.fixings is not None
        return _VarSwap(
            self.fixings, self.strike_vol**2, self.discount, annualisation=float(self.per_year)
        )

    @property
    def expiry(self) -> float:
        return self.end

    def transaction_cost(self, dq, t, spot, fc, discount):  # type: ignore[no-untyped-def]
        assert self.fixings is not None
        rem = _remaining(np.asarray(self.fixings), t)
        return np.asarray(
            self.cost
            * 0.01
            * 2.0
            * self.strike_vol
            * np.abs(dq)
            * float(discount.df(self.end))
            * rem
            * np.ones_like(spot)
        )


def ForwardVarianceSwap(  # constructor-like helper
    t1: float,
    t2: float,
    strike_vol: float,
    discount: DiscountCurve,
    *,
    cost: float = 0.0,
    per_year: int = 252,
) -> VarianceSwap:
    """A variance swap over ``[T1, T2]`` (the forward-variance bucket hedge)."""
    return VarianceSwap(
        start_time=t1,
        end=t2,
        strike_vol=strike_vol,
        discount=discount,
        cost=cost,
        per_year=per_year,
    )


@dataclass
class VolSwap(HedgeInstrument):
    """Vol swap on daily fixings over ``[start_time, end]`` at ``strike_vol``; ``cost`` in vol
    points."""

    start_time: float = 0.0
    end: float = 1.0
    strike_vol: float = 0.2
    discount: DiscountCurve | None = None
    per_year: int = 252
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"vol swap {self.start_time:g}->{self.end:g} K={self.strike_vol:g}"

    @property
    def product(self) -> Product:
        assert self.discount is not None
        n = max(1, round((self.end - self.start_time) * self.per_year))
        fx = np.linspace(self.start_time, self.end, n + 1)
        return _VolSwap(fx, self.strike_vol, self.discount, annualisation=float(self.per_year))

    @property
    def expiry(self) -> float:
        return self.end

    def transaction_cost(self, dq, t, spot, fc, discount):  # type: ignore[no-untyped-def]
        n = max(1, round((self.end - self.start_time) * self.per_year))
        rem = _remaining(np.linspace(self.start_time, self.end, n + 1), t)
        return np.asarray(
            self.cost * 0.01 * np.abs(dq) * float(discount.df(self.end)) * rem * np.ones_like(spot)
        )


@dataclass
class ConditionalVarianceSwap(HedgeInstrument):
    """Up / down variance swap as a hedge instrument (``side`` "up" / "down" of ``barrier``,
    corridor or conditional ``convention``); ``cost`` in vol points."""

    fixings: FloatArray | None = None
    barrier: float = 100.0
    side: str = "down"
    strike_vol: float = 0.2
    convention: str = "corridor"
    indicator: str = "prev"
    discount: DiscountCurve | None = None
    per_year: int = 252
    name: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            self.name = f"{self.side} var B={self.barrier:g} K={self.strike_vol:g}"

    @property
    def product(self) -> Product:
        assert self.discount is not None and self.fixings is not None
        return _CondVar(
            self.fixings,
            self.barrier,
            self.side,
            self.indicator,
            self.convention,
            self.strike_vol,
            self.discount,
            annualisation=float(self.per_year),
        )

    @property
    def expiry(self) -> float:
        assert self.fixings is not None
        return float(np.max(self.fixings))

    def transaction_cost(self, dq, t, spot, fc, discount):  # type: ignore[no-untyped-def]
        assert self.fixings is not None
        rem = _remaining(np.asarray(self.fixings), t)
        return np.asarray(
            self.cost
            * 0.01
            * 2.0
            * self.strike_vol
            * np.abs(dq)
            * float(discount.df(self.expiry))
            * rem
            * np.ones_like(spot)
        )


# --------------------------------------------------------------------------------------------
# families
# --------------------------------------------------------------------------------------------


def expand_rolls(
    instruments: Sequence[HedgeInstrument], dates: FloatArray, fc: ForwardCurve, horizon: float
) -> list[HedgeInstrument]:
    """Materialise the constant-maturity families: every ``roll_every`` along ``dates`` a new
    instrument is spawned at the roll date (module docstring); fixed instruments pass through."""
    out: list[HedgeInstrument] = []
    for inst in instruments:
        out.append(inst)
        if inst.roll != "constant_maturity":
            continue
        t_next = inst.start + inst.roll_every
        while t_next < horizon - _TOL:
            t_roll = float(dates[np.argmin(np.abs(dates - t_next))])
            if t_roll <= inst.start + _TOL:
                break
            child = inst.spawn(t_roll, fc)
            child.notes = inst.notes
            out.append(child)
            inst = child
            t_next = t_roll + child.roll_every
    return out


def instrument_products(instruments: Sequence[HedgeInstrument]) -> list[Product | None]:
    return [i.product for i in instruments]


def deduplicate_names(instruments: Sequence[HedgeInstrument]) -> None:
    seen: dict[str, int] = {}
    for inst in instruments:
        if inst.name in seen:
            seen[inst.name] += 1
            inst.name = f"{inst.name} #{seen[inst.name]}"
        else:
            seen[inst.name] = 1


__all__ = [
    "DEFAULT_REFERENCE_VOL",
    "AccumulatedSumPut",
    "CapCallStrip",
    "ConditionalVarianceSwap",
    "Digital",
    "ForwardStartRiskReversal",
    "ForwardStartStraddle",
    "ForwardVarianceSwap",
    "HedgeInstrument",
    "Spot",
    "Vanilla",
    "VarianceSwap",
    "VolSwap",
    "deduplicate_names",
    "expand_rolls",
    "instrument_products",
    "option_strip",
]
