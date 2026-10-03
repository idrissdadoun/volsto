"""Vanilla structures — spreads, ratios, butterflies, straddles — as portfolios of European
options, with model-free prices off an implied surface and premium matching (the barrier-versus-
vanilla study, SPEC §8.4).

A structure is a :class:`VanillaStructure`: a :class:`~volsto.products.base.Portfolio` of
:class:`~volsto.products.vanilla.EuropeanOption` legs of one expiry, one weight per leg, one
notional shared by the legs.  Its price needs no model — ``notional · Σ_i w_i Black(K_i, T,
σ̂(K_i, T))`` (:meth:`VanillaStructure.surface_price`) — so its only market input is today's smile
at the legs' strikes, the property the study sets against the path-dependent barrier option.

Conventions (long structures; weights per unit notional):

* ``call_spread(k1, k2)``: ``+C(k1) − C(k2)``, ``k1 < k2``; ``put_spread(k1, k2)``: ``+P(k2) −
  P(k1)``, ``k1 < k2``.
* ``call_ratio(k1, k2, ratio)``: ``+C(k1) − ratio · C(k2)``, ``k1 < k2`` (a 1×2 for ``ratio =
  2``): gain capped at ``k2 − k1`` on ``k2``, loss beyond the upper break-even ``k2 + (k2 −
  k1)/(ratio − 1)`` for ``ratio > 1``.  ``put_ratio(k1, k2, ratio)``: ``+P(k2) − ratio · P(k1)``,
  ``k1 < k2``.
* ``call_fly(k1, k2, k3)``: ``+C(k1) − (k3 − k1)/(k3 − k2) · C(k2) + (k2 − k1)/(k3 − k2) ·
  C(k3)``, ``k1 < k2 < k3`` — the weights that make the payoff vanish beyond ``k3`` (``1, −2, 1``
  when ``k2`` is the midpoint); peak ``k2 − k1`` at ``k2``.  ``put_fly(k1, k2, k3)`` is the mirror
  image: ``+P(k3) − (k3 − k1)/(k2 − k1) · P(k2) + (k3 − k2)/(k2 − k1) · P(k1)``, peak ``k3 − k2``
  at ``k2``, zero below ``k1``.
* ``straddle(k)``: ``+C(k) + P(k)``.

Premium matching (the study compares structures of one premium): :func:`ratio_for_premium`
solves the short-leg ratio in closed form, ``ratio = (C(k1) − premium)/C(k2)`` (the put mirror
with the puts); :func:`fly_wing_for_premium` moves the outer wing ``k3`` (call fly) / ``k1`` (put
fly) by bisection with the other two strikes held; :func:`units_for_premium` is the plain
scaling ``premium / price``.

The European knock-out (:func:`european_knock_out`) is the barrier option observed at expiry
only — an upper bound of the daily and continuous knock-outs — and its model-free price is the
spread minus the digital at the barrier (:func:`european_knock_out_surface_price`;
:func:`surface_digital` is the skew-adjusted cash digital by a centred strike difference).

Checked by ``tests/test_structures.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import pairwise
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.products.barrier import KnockOutOption
from volsto.products.base import Portfolio, Product, parse_cp, shift_times
from volsto.products.vanilla import EuropeanOption

if TYPE_CHECKING:
    from volsto.market.curves import DiscountCurve
    from volsto.market.surface import ImpliedSurface

FloatArray = NDArray[np.float64]

#: half-width of the digital's centred strike difference, as a fraction of the level
DIGITAL_REL_WIDTH = 0.005


def _strikes(*ks: float) -> tuple[float, ...]:
    out = tuple(float(k) for k in ks)
    if any(not np.isfinite(k) or k <= 0 for k in out):
        raise ValueError("strikes must be positive and finite")
    if any(b <= a for a, b in pairwise(out)):
        raise ValueError(f"strikes must be strictly increasing, got {out}")
    return out


class VanillaStructure(Portfolio):
    """A weighted set of same-expiry European options (module docstring)."""

    def __init__(
        self,
        strikes: Sequence[float],
        cps: Sequence[int | str],
        weights: Sequence[float],
        maturity: float,
        discount: DiscountCurve,
        notional: float = 1.0,
        kind: str = "structure",
    ) -> None:
        if not (len(strikes) == len(cps) == len(weights)) or not strikes:
            raise ValueError("strikes, cps and weights must have the same positive length")
        if not np.isfinite(maturity) or maturity <= 0:
            raise ValueError("maturity must be positive")
        if not np.isfinite(notional):
            raise ValueError("notional must be finite")
        self.strikes = tuple(float(k) for k in strikes)
        self.cps = tuple(parse_cp(c) for c in cps)
        self.leg_weights = tuple(float(w) for w in weights)
        self.T = float(maturity)
        self.kind = str(kind)
        legs = [
            EuropeanOption(k, self.T, cp, discount, notional)
            for k, cp in zip(self.strikes, self.cps, strict=True)
        ]
        super().__init__(legs, self.leg_weights)
        self.notional = float(notional)

    # -- model-free quantities ---------------------------------------------------------------

    def payoff_at(self, spot: ArrayLike) -> FloatArray:
        """Undiscounted terminal payoff per unit notional at ``S_T = spot`` (broadcasts)."""
        s = np.asarray(spot, dtype=np.float64)[..., None]
        k = np.asarray(self.strikes)
        c = np.asarray(self.cps, dtype=np.float64)
        w = np.asarray(self.leg_weights)
        return np.asarray(np.sum(w * np.maximum(c * (s - k), 0.0), axis=-1), dtype=np.float64)

    def surface_price(self, surface: ImpliedSurface) -> float:
        """``notional · Σ_i w_i · Black(K_i, T, σ̂(K_i, T))`` — discounted, no model."""
        total = 0.0
        for k, cp, w in zip(self.strikes, self.cps, self.leg_weights, strict=True):
            total += w * float(np.asarray(surface.price(k, self.T, cp)).ravel()[0])
        return self.notional * total

    def surface_vega(self, surface: ImpliedSurface, size: float = 0.01) -> float:
        """Parallel vega per ``size`` of implied vol: every leg repriced at ``σ̂ + size`` (the
        smile shape held), per unit notional of the structure."""
        from volsto.market.bs import black_price

        total = 0.0
        for k, cp, w in zip(self.strikes, self.cps, self.leg_weights, strict=True):
            f = float(surface.forward(self.T))
            df = float(surface.discount.df(self.T))
            sig = float(np.asarray(surface.implied_vol(k, self.T)).ravel()[0])
            up = float(black_price(f, k, self.T, sig + size, cp, df))
            base = float(black_price(f, k, self.T, sig, cp, df))
            total += w * (up - base)
        return self.notional * total

    def max_payoff(self, lo: float, hi: float, n: int = 4001) -> float:
        """Largest payoff per unit notional on ``[lo, hi]`` (a grid; the payoff is piecewise
        linear with kinks at the strikes, which the grid includes)."""
        s = np.unique(np.concatenate((np.linspace(lo, hi, n), np.asarray(self.strikes))))
        return float(np.max(self.payoff_at(s)))

    # -- Product protocol ----------------------------------------------------------------------

    def _rebuild(self, maturity: float, discount: DiscountCurve) -> VanillaStructure:
        return VanillaStructure(
            self.strikes, self.cps, self.leg_weights, maturity, discount, self.notional, self.kind
        )

    def aged(self, dt: float) -> Product:
        return self._rebuild(float(shift_times([self.T], dt)[0]), self.discount)

    def with_discount(self, discount: DiscountCurve) -> Product:
        return self._rebuild(self.T, discount)

    def __repr__(self) -> str:
        legs = ", ".join(
            f"{w:+g} x {'C' if cp > 0 else 'P'}({k:g})"
            for k, cp, w in zip(self.strikes, self.cps, self.leg_weights, strict=True)
        )
        return f"{self.kind} [{legs}] expiry {self.T:g}y, notional {self.notional:g}"


# --------------------------------------------------------------------------------------------
# constructors
# --------------------------------------------------------------------------------------------


def call_spread(
    k1: float, k2: float, maturity: float, discount: DiscountCurve, notional: float = 1.0
) -> VanillaStructure:
    """``+C(k1) − C(k2)``, ``k1 < k2``."""
    ks = _strikes(k1, k2)
    return VanillaStructure(ks, (1, 1), (1.0, -1.0), maturity, discount, notional, "call spread")


def put_spread(
    k1: float, k2: float, maturity: float, discount: DiscountCurve, notional: float = 1.0
) -> VanillaStructure:
    """``+P(k2) − P(k1)``, ``k1 < k2``."""
    ks = _strikes(k1, k2)
    return VanillaStructure(ks, (-1, -1), (-1.0, 1.0), maturity, discount, notional, "put spread")


def call_ratio(
    k1: float,
    k2: float,
    ratio: float,
    maturity: float,
    discount: DiscountCurve,
    notional: float = 1.0,
) -> VanillaStructure:
    """``+C(k1) − ratio · C(k2)``, ``k1 < k2``, ``ratio > 0``."""
    ks = _strikes(k1, k2)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("ratio must be positive")
    kind = f"call ratio 1x{ratio:g}"
    return VanillaStructure(ks, (1, 1), (1.0, -float(ratio)), maturity, discount, notional, kind)


def put_ratio(
    k1: float,
    k2: float,
    ratio: float,
    maturity: float,
    discount: DiscountCurve,
    notional: float = 1.0,
) -> VanillaStructure:
    """``+P(k2) − ratio · P(k1)``, ``k1 < k2``, ``ratio > 0``."""
    ks = _strikes(k1, k2)
    if not np.isfinite(ratio) or ratio <= 0:
        raise ValueError("ratio must be positive")
    kind = f"put ratio 1x{ratio:g}"
    return VanillaStructure(ks, (-1, -1), (-float(ratio), 1.0), maturity, discount, notional, kind)


def call_fly(
    k1: float,
    k2: float,
    k3: float,
    maturity: float,
    discount: DiscountCurve,
    notional: float = 1.0,
) -> VanillaStructure:
    """``+C(k1) − (k3 − k1)/(k3 − k2) C(k2) + (k2 − k1)/(k3 − k2) C(k3)``, ``k1 < k2 < k3``."""
    ks = _strikes(k1, k2, k3)
    a, b, c = ks
    w = (1.0, -(c - a) / (c - b), (b - a) / (c - b))
    return VanillaStructure(ks, (1, 1, 1), w, maturity, discount, notional, "call fly")


def put_fly(
    k1: float,
    k2: float,
    k3: float,
    maturity: float,
    discount: DiscountCurve,
    notional: float = 1.0,
) -> VanillaStructure:
    """``+P(k3) − (k3 − k1)/(k2 − k1) P(k2) + (k3 − k2)/(k2 − k1) P(k1)``, ``k1 < k2 < k3``."""
    ks = _strikes(k1, k2, k3)
    a, b, c = ks
    w = ((c - b) / (b - a), -(c - a) / (b - a), 1.0)
    return VanillaStructure(ks, (-1, -1, -1), w, maturity, discount, notional, "put fly")


def straddle(
    k: float, maturity: float, discount: DiscountCurve, notional: float = 1.0
) -> VanillaStructure:
    """``+C(k) + P(k)``."""
    ks = _strikes(k)
    return VanillaStructure(
        (ks[0], ks[0]), (1, -1), (1.0, 1.0), maturity, discount, notional, "straddle"
    )


# --------------------------------------------------------------------------------------------
# premium matching
# --------------------------------------------------------------------------------------------


def _vanilla(surface: ImpliedSurface, k: float, T: float, cp: int) -> float:
    return float(np.asarray(surface.price(k, T, cp)).ravel()[0])


def ratio_for_premium(
    k1: float, k2: float, maturity: float, surface: ImpliedSurface, premium: float, cp: int = 1
) -> float:
    """The short-leg ratio of a call ratio (``cp = +1``: ``C(k1) − ratio · C(k2) = premium``) or
    a put ratio (``cp = −1``: ``P(k2) − ratio · P(k1) = premium``) per unit notional."""
    ks = _strikes(k1, k2)
    if cp > 0:
        long_leg = _vanilla(surface, ks[0], maturity, 1)
        short_leg = _vanilla(surface, ks[1], maturity, 1)
    else:
        long_leg = _vanilla(surface, ks[1], maturity, -1)
        short_leg = _vanilla(surface, ks[0], maturity, -1)
    if short_leg <= 0:
        raise ValueError("the short leg has no value: no ratio reaches the premium")
    return (long_leg - float(premium)) / short_leg


def fly_wing_for_premium(
    k1: float,
    k2: float,
    maturity: float,
    surface: ImpliedSurface,
    premium: float,
    discount: DiscountCurve,
    cp: int = 1,
    *,
    bound: float = 3.0,
) -> VanillaStructure:
    """The call fly ``(k1, k2, k3)`` (``cp = +1``) whose outer wing ``k3 > k2`` makes its price
    ``premium`` per unit notional, or the put fly ``(k1, k2, k3)`` (``cp = −1``; here ``k1``
    and ``k2`` are the middle and upper strikes and the lower wing is solved).  The price grows
    with the wing's distance (more payoff), so the wing is found by bisection on ``[k2, bound ·
    k2]`` (calls) / ``[k2 / bound, k2]`` (puts); a premium outside the reachable range raises."""
    from scipy.optimize import brentq

    if cp > 0:
        lo, hi = _strikes(k1, k2)

        def price_call(wing: float) -> float:
            return call_fly(lo, hi, wing, maturity, discount).surface_price(surface) - premium

        a, b = hi * (1.0 + 1e-4), hi * bound
        if price_call(a) > 0 or price_call(b) < 0:
            raise ValueError("the premium is outside the range a call fly of these strikes spans")
        wing = float(brentq(price_call, a, b, xtol=1e-10 * hi))
        return call_fly(lo, hi, wing, maturity, discount)
    mid, top = _strikes(k1, k2)

    def price_put(wing: float) -> float:
        return put_fly(wing, mid, top, maturity, discount).surface_price(surface) - premium

    a, b = mid / bound, mid * (1.0 - 1e-4)
    if price_put(a) < 0 or price_put(b) > 0:
        raise ValueError("the premium is outside the range a put fly of these strikes spans")
    wing = float(brentq(price_put, a, b, xtol=1e-10 * mid))
    return put_fly(wing, mid, top, maturity, discount)


def fly_width_for_premium(
    k: float,
    maturity: float,
    surface: ImpliedSurface,
    premium: float,
    discount: DiscountCurve,
    cp: int = 1,
    *,
    bound: float = 2.0,
) -> VanillaStructure:
    """The symmetric fly of inner strike ``k`` whose width matches the premium: the call fly
    ``(k, (k + w)/2, w)`` with ``w > k`` (``cp = +1``) or the put fly ``(w, (k + w)/2, k)`` with
    ``w < k`` (``cp = −1``).  Its price runs from zero (``w → k``) to the fly spanning ``[k,
    bound · k]``, so a premium below that fly's price is always matched (bisection); a fly
    spanning the whole strike-to-barrier range cannot match a knock-out's premium, which lies
    below the European knock-out's — the floor of every fly with the inner strike held
    (``tests/test_structures.py``)."""
    from scipy.optimize import brentq

    k = float(k)
    if cp > 0:

        def price_call(w: float) -> float:
            return (
                call_fly(k, 0.5 * (k + w), w, maturity, discount).surface_price(surface) - premium
            )

        a, b = k * (1.0 + 1e-4), k * bound
        if price_call(b) < 0:
            raise ValueError("the premium exceeds the symmetric fly spanning [k, bound k]")
        w = float(brentq(price_call, a, b, xtol=1e-10 * k))
        return call_fly(k, 0.5 * (k + w), w, maturity, discount)

    def price_put(w: float) -> float:
        return put_fly(w, 0.5 * (k + w), k, maturity, discount).surface_price(surface) - premium

    a, b = k / bound, k * (1.0 - 1e-4)
    if price_put(a) < 0:
        raise ValueError("the premium exceeds the symmetric fly spanning [k / bound, k]")
    w = float(brentq(price_put, a, b, xtol=1e-10 * k))
    return put_fly(w, 0.5 * (k + w), k, maturity, discount)


def units_for_premium(
    structure: VanillaStructure, surface: ImpliedSurface, premium: float
) -> float:
    """``premium / price``: the number of units of the structure one premium buys."""
    p = structure.surface_price(surface)
    if p <= 0:
        raise ValueError("the structure has no positive price")
    return float(premium) / p


# --------------------------------------------------------------------------------------------
# the European knock-out and the digital
# --------------------------------------------------------------------------------------------


def surface_digital(
    surface: ImpliedSurface,
    strike: float,
    maturity: float,
    cp: int = 1,
    rel_width: float = DIGITAL_REL_WIDTH,
) -> float:
    """Discounted price of the cash digital paying 1 when ``cp (S_T − K) > 0``, by the centred
    strike difference ``(C(K − h) − C(K + h))/(2h)`` (``h = rel_width · K``) — the skew-adjusted
    digital, ``−∂C/∂K`` along the smile (the put digital is ``(P(K + h) − P(K − h))/(2h)``)."""
    h = float(rel_width) * float(strike)
    cp = parse_cp(cp)
    lo = _vanilla(surface, strike - h, maturity, cp)
    hi = _vanilla(surface, strike + h, maturity, cp)
    return float(cp * (lo - hi) / (2.0 * h))


def european_knock_out(
    strike: float,
    barrier: float,
    direction: str,
    cp: int,
    maturity: float,
    discount: DiscountCurve,
    notional: float = 1.0,
) -> KnockOutOption:
    """The barrier option observed at expiry only (``fixing_times=[T]``, strict): ``(cp (S_T −
    K))⁺ 1{S_T on the alive side of B}``."""
    return KnockOutOption(
        strike,
        maturity,
        cp,
        barrier,
        direction,
        discount,
        monitoring="discrete",
        fixing_times=[maturity],
        strict=True,
        notional=notional,
    )


def european_knock_out_surface_price(
    surface: ImpliedSurface,
    strike: float,
    barrier: float,
    direction: str,
    cp: int,
    maturity: float,
    notional: float = 1.0,
    rel_width: float = DIGITAL_REL_WIDTH,
) -> float:
    """Model-free price of the reverse European knock-outs: up-and-out call ``B > K``: ``C(K) −
    C(B) − (B − K) D_call(B)``; down-and-out put ``H < K``: ``P(K) − P(H) − (K − H) D_put(H)``
    (:func:`surface_digital`).  Other combinations raise."""
    cp = parse_cp(cp)
    k, b = float(strike), float(barrier)
    if direction == "up" and cp > 0 and b > k:
        spread = _vanilla(surface, k, maturity, 1) - _vanilla(surface, b, maturity, 1)
        dig = surface_digital(surface, b, maturity, 1, rel_width)
        return notional * (spread - (b - k) * dig)
    if direction == "down" and cp < 0 and b < k:
        spread = _vanilla(surface, k, maturity, -1) - _vanilla(surface, b, maturity, -1)
        dig = surface_digital(surface, b, maturity, -1, rel_width)
        return notional * (spread - (k - b) * dig)
    raise NotImplementedError(
        "the model-free European knock-out is implemented for the reverse barriers (up-and-out "
        "call B > K, down-and-out put H < K)"
    )


__all__ = [
    "DIGITAL_REL_WIDTH",
    "VanillaStructure",
    "call_fly",
    "call_ratio",
    "call_spread",
    "european_knock_out",
    "european_knock_out_surface_price",
    "fly_width_for_premium",
    "fly_wing_for_premium",
    "put_fly",
    "put_ratio",
    "put_spread",
    "ratio_for_premium",
    "straddle",
    "surface_digital",
    "units_for_premium",
]
