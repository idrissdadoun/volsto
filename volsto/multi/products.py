"""Basket and dispersion products on a :class:`~volsto.multi.paths.MultiPathSet` (SPEC §8.5).

Performances are ``r_i = S_i(T) / ref_i − 1`` with ``ref_i`` the initial spot (default) or the
reference levels given (the forwards for an at-the-money-forward contract); the basket
performance is ``r_B = Σ w_i r_i`` (``Σ w_i = 1`` is not enforced: the weights are the term
sheet's).

* :class:`Palladium` — the call on dispersion: ``notional · DF(T) · (D − K)⁺`` with the
  dispersion ``D = Σ_i w_i |r_i − r_B|`` (the owner's definition: the weighted sum of the absolute
  performances of the names against the basket's); ``K = 0`` is the dispersion forward.
* :class:`BasketOption` — ``(cp (r_B − K))⁺`` (a call / put on the basket performance);
  :class:`BasketStraddle` — ``|r_B − K|``.
* :class:`SingleNameStraddles` — ``Σ w_i |r_i − K|`` (the single-name straddle package with the
  basket's weights).
* :func:`dispersion_straddles` — the straddle dispersion trade ``Σ w_i |r_i| − λ |r_B|``
  (``λ = 1`` with the basket's weights; ``λ`` the premium- or vega-neutral scaling).  By the
  triangle inequality ``Σ w_i |r_i| − |r_B| ≤ D ≤ Σ w_i |r_i| + |r_B|`` path by path, so the
  palladium forward dominates the straddle dispersion payoff (``tests/test_multi.py``).
* :class:`VarianceDispersion` — the delta-hedged analogue: ``Σ w_i RV_i − RV_B − K`` on the
  fixing schedule (annualised realised variances of the names and of the performance basket
  ``Σ w_i S_i(t)/S_i(0)``).
* :class:`MultiPortfolio` — a weighted sum of these.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.products.base import parse_cp

if TYPE_CHECKING:
    from volsto.engine.grid import FixingIndex
    from volsto.market.curves import DiscountCurve
    from volsto.multi.paths import MultiPathSet

FloatArray = NDArray[np.float64]


def _weights(weights: ArrayLike, n: int | None = None) -> FloatArray:
    w = np.asarray(weights, dtype=np.float64).ravel()
    if w.size == 0 or not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite and non-empty")
    if n is not None and w.size != n:
        raise ValueError(f"expected {n} weights, got {w.size}")
    return w


class MultiAssetProduct(ABC):
    """Abstract product on several underlyings (the single-asset ``Product`` protocol)."""

    requires_all_steps: bool = False

    def __init__(self, discount: DiscountCurve, notional: float = 1.0) -> None:
        if not np.isfinite(notional):
            raise ValueError("notional must be finite")
        self.discount = discount
        self.notional = float(notional)

    @property
    @abstractmethod
    def fixing_times(self) -> FloatArray: ...

    @property
    def maturity(self) -> float:
        return float(np.max(self.fixing_times))

    @abstractmethod
    def payoff(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray: ...

    def df(self, t: float) -> float:
        return float(self.discount.df(t))

    @abstractmethod
    def __repr__(self) -> str: ...


class _Terminal(MultiAssetProduct):
    """Terminal payoffs on the performances at ``T``."""

    def __init__(
        self,
        weights: ArrayLike,
        maturity: float,
        discount: DiscountCurve,
        *,
        notional: float = 1.0,
        reference: Sequence[float] | None = None,
    ) -> None:
        super().__init__(discount, notional)
        if not np.isfinite(maturity) or maturity <= 0:
            raise ValueError("maturity must be positive")
        self.weights = _weights(weights)
        self.T = float(maturity)
        self.reference = None if reference is None else tuple(float(x) for x in reference)
        if self.reference is not None and len(self.reference) != self.weights.size:
            raise ValueError("one reference level per asset")

    @property
    def fixing_times(self) -> FloatArray:
        return np.array([self.T])

    def performances(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        r = paths.performances(idx[self.T], self.reference)
        if r.shape[1] != self.weights.size:
            raise ValueError("the paths carry a different number of assets than the weights")
        return r

    def basket_performance(self, r: FloatArray) -> FloatArray:
        return np.asarray(r @ self.weights, dtype=np.float64)

    def _ref_repr(self) -> str:
        return "spot" if self.reference is None else "given levels"


class Palladium(_Terminal):
    """The call on dispersion ``(Σ w_i |r_i − r_B| − K)⁺`` (module docstring)."""

    def __init__(
        self,
        weights: ArrayLike,
        strike: float,
        maturity: float,
        discount: DiscountCurve,
        *,
        notional: float = 1.0,
        reference: Sequence[float] | None = None,
    ) -> None:
        super().__init__(weights, maturity, discount, notional=notional, reference=reference)
        if not np.isfinite(strike) or strike < 0:
            raise ValueError("the dispersion strike must be non-negative")
        self.strike = float(strike)

    def dispersion(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        """``D = Σ w_i |r_i − r_B|`` per path (undiscounted, before the strike)."""
        r = self.performances(paths, idx)
        rb = self.basket_performance(r)
        return np.asarray(np.abs(r - rb[:, None]) @ self.weights, dtype=np.float64)

    def payoff(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        d = self.dispersion(paths, idx)
        return np.asarray(
            self.notional * self.df(self.T) * np.maximum(d - self.strike, 0.0), dtype=np.float64
        )

    def __repr__(self) -> str:
        return (
            f"Palladium (call on dispersion): strike {self.strike:g}, expiry {self.T:g}y, "
            f"{self.weights.size} names, reference {self._ref_repr()}, notional {self.notional:g}"
        )


class BasketOption(_Terminal):
    """``(cp (r_B − K))⁺`` on the basket performance."""

    def __init__(
        self,
        weights: ArrayLike,
        strike: float,
        maturity: float,
        cp: int | str,
        discount: DiscountCurve,
        *,
        notional: float = 1.0,
        reference: Sequence[float] | None = None,
    ) -> None:
        super().__init__(weights, maturity, discount, notional=notional, reference=reference)
        self.strike = float(strike)
        self.cp = parse_cp(cp)

    def payoff(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        rb = self.basket_performance(self.performances(paths, idx))
        cf = np.maximum(self.cp * (rb - self.strike), 0.0)
        return np.asarray(self.notional * self.df(self.T) * cf, dtype=np.float64)

    def __repr__(self) -> str:
        kind = "call" if self.cp > 0 else "put"
        return (
            f"Basket {kind}: performance strike {self.strike:g}, expiry {self.T:g}y, "
            f"{self.weights.size} names, reference {self._ref_repr()}, notional {self.notional:g}"
        )


class BasketStraddle(_Terminal):
    """``|r_B − K|``."""

    def __init__(
        self,
        weights: ArrayLike,
        maturity: float,
        discount: DiscountCurve,
        *,
        strike: float = 0.0,
        notional: float = 1.0,
        reference: Sequence[float] | None = None,
    ) -> None:
        super().__init__(weights, maturity, discount, notional=notional, reference=reference)
        self.strike = float(strike)

    def payoff(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        rb = self.basket_performance(self.performances(paths, idx))
        return np.asarray(
            self.notional * self.df(self.T) * np.abs(rb - self.strike), dtype=np.float64
        )

    def __repr__(self) -> str:
        return (
            f"Basket straddle: performance strike {self.strike:g}, expiry {self.T:g}y, "
            f"{self.weights.size} names, reference {self._ref_repr()}, notional {self.notional:g}"
        )


class SingleNameStraddles(_Terminal):
    """``Σ w_i |r_i − K|`` (one straddle per name, the basket's weights)."""

    def __init__(
        self,
        weights: ArrayLike,
        maturity: float,
        discount: DiscountCurve,
        *,
        strike: float = 0.0,
        notional: float = 1.0,
        reference: Sequence[float] | None = None,
    ) -> None:
        super().__init__(weights, maturity, discount, notional=notional, reference=reference)
        self.strike = float(strike)

    def payoff(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        r = self.performances(paths, idx)
        return np.asarray(
            self.notional * self.df(self.T) * (np.abs(r - self.strike) @ self.weights),
            dtype=np.float64,
        )

    def __repr__(self) -> str:
        return (
            f"Single-name straddles: performance strike {self.strike:g}, expiry {self.T:g}y, "
            f"{self.weights.size} names, reference {self._ref_repr()}, notional {self.notional:g}"
        )


class MultiPortfolio(MultiAssetProduct):
    """Weighted sum of multi-asset products (each leg discounts itself)."""

    def __init__(
        self, legs: Sequence[MultiAssetProduct], weights: Sequence[float] | None = None
    ) -> None:
        if not legs:
            raise ValueError("a portfolio needs at least one leg")
        super().__init__(legs[0].discount, 1.0)
        self.legs = list(legs)
        self.weights = [1.0] * len(self.legs) if weights is None else [float(w) for w in weights]
        if len(self.weights) != len(self.legs):
            raise ValueError("one weight per leg")
        self.requires_all_steps = any(leg.requires_all_steps for leg in self.legs)

    @property
    def fixing_times(self) -> FloatArray:
        return np.unique(np.concatenate([leg.fixing_times for leg in self.legs]))

    def payoff(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        total = np.zeros(paths.n_paths)
        for w, leg in zip(self.weights, self.legs, strict=True):
            total += w * leg.payoff(paths, idx)
        return total

    def __repr__(self) -> str:
        parts = ", ".join(f"{w:g} x [{leg!r}]" for w, leg in zip(self.weights, self.legs))
        return f"MultiPortfolio({parts})"


def dispersion_straddles(
    weights: ArrayLike,
    maturity: float,
    discount: DiscountCurve,
    *,
    basket_scale: float = 1.0,
    strike: float = 0.0,
    notional: float = 1.0,
    reference: Sequence[float] | None = None,
) -> MultiPortfolio:
    """``Σ w_i |r_i − K| − basket_scale · |r_B − K|``: long the single-name straddles, short
    ``basket_scale`` basket straddles (module docstring)."""
    if not np.isfinite(basket_scale) or basket_scale < 0:
        raise ValueError("basket_scale must be non-negative")
    singles = SingleNameStraddles(
        weights, maturity, discount, strike=strike, notional=notional, reference=reference
    )
    basket = BasketStraddle(
        weights, maturity, discount, strike=strike, notional=notional, reference=reference
    )
    return MultiPortfolio([singles, basket], [1.0, -float(basket_scale)])


class VarianceDispersion(MultiAssetProduct):
    """``notional · DF(T) · (Σ w_i RV_i − basket_scale · RV_B − K)`` with the annualised
    realised variances ``RV = (A/n) Σ ln²(S_{t_j}/S_{t_{j−1}})`` on the fixing schedule, the
    basket's from the performance basket ``Σ w_i S_i(t)/S_i(0)``."""

    def __init__(
        self,
        weights: ArrayLike,
        fixing_times: ArrayLike,
        strike: float,
        discount: DiscountCurve,
        *,
        basket_scale: float = 1.0,
        notional: float = 1.0,
        annualisation: float = 252.0,
    ) -> None:
        super().__init__(discount, notional)
        self.weights = _weights(weights)
        t = np.asarray(fixing_times, dtype=np.float64).ravel()
        if t.size < 2 or np.any(np.diff(t) <= 0) or t[0] < 0:
            raise ValueError("fixing_times must be increasing with at least two dates")
        self.schedule = t if t[0] == 0.0 else np.concatenate(([0.0], t))
        self.strike = float(strike)
        self.basket_scale = float(basket_scale)
        self.annualisation = float(annualisation)

    @property
    def fixing_times(self) -> FloatArray:
        return self.schedule

    def payoff(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        cols = idx.indices(self.schedule)
        n = cols.size - 1
        scale = self.annualisation / n
        rv_names = np.column_stack([p.realised_variance_fixings(cols) for p in paths.assets])
        levels = np.column_stack([paths.basket_level(int(c), self.weights) for c in cols])
        rb = np.diff(np.log(levels), axis=1)
        rv_b = np.sum(rb * rb, axis=1)
        value = scale * (rv_names @ self.weights - self.basket_scale * rv_b) - self.strike
        T = float(self.schedule[-1])
        return np.asarray(self.notional * self.df(T) * value, dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"Variance dispersion: {self.weights.size} names, {self.schedule.size - 1} fixings "
            f"to {self.schedule[-1]:g}y, basket scale {self.basket_scale:g}, strike "
            f"{self.strike:g}, notional {self.notional:g}"
        )


__all__ = [
    "BasketOption",
    "BasketStraddle",
    "MultiAssetProduct",
    "MultiPortfolio",
    "Palladium",
    "SingleNameStraddles",
    "VarianceDispersion",
    "dispersion_straddles",
]
