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

Correlation products of the local correlation model (SPEC §8.7, M12 part LC5; each class states
its payoff and the test that checks it): :class:`PalladiumPut`, :class:`PalladiumCallSpread`,
:class:`DispersionGap`, :class:`WorstOf` / :class:`BestOf` (with a discretely monitored knock-in
on the worst-of), :class:`OutperformanceOption`, :class:`RelativePerformanceStraddle`,
:class:`CorrelationSwap` and :class:`BasketVarianceSwap`.

**Reference levels.**  A product with performances reads ``reference`` (default: the column-0
spots).  ``with_reference(levels)`` returns the same contract with the levels bound: the risk
code binds the base state's spots, so that a spot bump never moves a strike.
"""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING, Self

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

    def with_reference(self, levels: Sequence[float]) -> MultiAssetProduct:
        """The same contract with its performance reference levels bound to ``levels``
        (overridden by every product that has performances)."""
        raise TypeError(f"{type(self).__name__} has no performance reference to bind")

    @abstractmethod
    def __repr__(self) -> str: ...


def _reference(reference: Sequence[float] | None) -> tuple[float, ...] | None:
    if reference is None:
        return None
    ref = tuple(float(x) for x in reference)
    if not ref or not all(np.isfinite(x) and x > 0 for x in ref):
        raise ValueError("reference levels must be positive and finite")
    return ref


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
        self.reference = _reference(reference)
        if self.reference is not None and len(self.reference) != self.weights.size:
            raise ValueError("one reference level per asset")

    def with_reference(self, levels: Sequence[float]) -> Self:
        ref = _reference(levels)
        if ref is None or len(ref) != self.weights.size:
            raise ValueError("one reference level per asset")
        out = copy.copy(self)
        out.reference = ref
        return out

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


class _Dispersion(_Terminal):
    """Terminal payoffs on the dispersion ``D = Σ_i w_i |r_i − r_B|``."""

    def dispersion(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        """``D = Σ w_i |r_i − r_B|`` per path (undiscounted, before the strike)."""
        r = self.performances(paths, idx)
        rb = self.basket_performance(r)
        return np.asarray(np.abs(r - rb[:, None]) @ self.weights, dtype=np.float64)


class Palladium(_Dispersion):
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

    def with_reference(self, levels: Sequence[float]) -> MultiPortfolio:
        return MultiPortfolio([leg.with_reference(levels) for leg in self.legs], self.weights)

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


# ---------------------------------------------------------------------------------------------
# correlation products (SPEC §8.7, M12 part LC5)
# ---------------------------------------------------------------------------------------------


class PalladiumPut(_Dispersion):
    """The put on dispersion: ``N · DF(T) · (K − D)⁺`` with ``D = Σ w_i |r_i − r_B|``.

    Parity ``(D − K)⁺ − (K − D)⁺ = D − K`` holds path by path
    (``tests/test_local_correlation.py::test_dispersion_product_identities``)."""

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

    def payoff(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        d = self.dispersion(paths, idx)
        return np.asarray(
            self.notional * self.df(self.T) * np.maximum(self.strike - d, 0.0), dtype=np.float64
        )

    def __repr__(self) -> str:
        return (
            f"Palladium put (put on dispersion): strike {self.strike:g}, expiry {self.T:g}y, "
            f"{self.weights.size} names, reference {self._ref_repr()}, notional {self.notional:g}"
        )


class PalladiumCallSpread(_Dispersion):
    """The call spread on dispersion: ``N · DF(T) · min((D − K₁)⁺, K₂ − K₁)``, ``0 ≤ K₁ < K₂``.

    Equal to ``Palladium(K₁) − Palladium(K₂)`` path by path
    (``tests/test_local_correlation.py::test_dispersion_product_identities``)."""

    def __init__(
        self,
        weights: ArrayLike,
        lower: float,
        upper: float,
        maturity: float,
        discount: DiscountCurve,
        *,
        notional: float = 1.0,
        reference: Sequence[float] | None = None,
    ) -> None:
        super().__init__(weights, maturity, discount, notional=notional, reference=reference)
        if not (np.isfinite(lower) and np.isfinite(upper) and 0.0 <= lower < upper):
            raise ValueError("need 0 <= lower < upper")
        self.lower, self.upper = float(lower), float(upper)

    def payoff(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        d = self.dispersion(paths, idx)
        cf = np.minimum(np.maximum(d - self.lower, 0.0), self.upper - self.lower)
        return np.asarray(self.notional * self.df(self.T) * cf, dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"Palladium call spread: strikes {self.lower:g} / {self.upper:g}, expiry {self.T:g}y, "
            f"{self.weights.size} names, reference {self._ref_repr()}, notional {self.notional:g}"
        )


class DispersionGap(_Dispersion):
    """The gap between the dispersion and the straddle package:
    ``N · DF(T) · G`` with ``G = D − (Σ_i w_i |r_i − K| − β_B · |r_B − K|)``.

    With ``β_B = 1``, ``K = 0`` and ``Σ w = 1``: ``G = 2 · Σ_i w_i (|r_B| − (ε r_i)⁺)⁺``,
    ``ε = sgn r_B`` (derived; the study's ``disp_payoff.gap_formula``), so ``0 ≤ G ≤ 2|r_B|``.
    Equal to ``Palladium(0) − dispersion_straddles`` and to ``gap_formula`` path by path
    (``tests/test_local_correlation.py::test_dispersion_product_identities``)."""

    def __init__(
        self,
        weights: ArrayLike,
        maturity: float,
        discount: DiscountCurve,
        *,
        basket_scale: float = 1.0,
        strike: float = 0.0,
        notional: float = 1.0,
        reference: Sequence[float] | None = None,
    ) -> None:
        super().__init__(weights, maturity, discount, notional=notional, reference=reference)
        if not np.isfinite(basket_scale) or basket_scale < 0:
            raise ValueError("basket_scale must be non-negative")
        self.basket_scale, self.strike = float(basket_scale), float(strike)

    def payoff(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        r = self.performances(paths, idx)
        rb = self.basket_performance(r)
        d = np.abs(r - rb[:, None]) @ self.weights
        package = np.abs(r - self.strike) @ self.weights - self.basket_scale * np.abs(
            rb - self.strike
        )
        return np.asarray(self.notional * self.df(self.T) * (d - package), dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"Dispersion gap (dispersion minus straddle package): basket scale "
            f"{self.basket_scale:g}, strike {self.strike:g}, expiry {self.T:g}y, "
            f"{self.weights.size} names, reference {self._ref_repr()}, notional {self.notional:g}"
        )


class _Levels(MultiAssetProduct):
    """Products on the levels ``X_i = S_i(t)/ref_i`` of a set of assets ``A``."""

    def __init__(
        self,
        maturity: float,
        discount: DiscountCurve,
        *,
        assets: Sequence[int] | None = None,
        notional: float = 1.0,
        reference: Sequence[float] | None = None,
    ) -> None:
        super().__init__(discount, notional)
        if not np.isfinite(maturity) or maturity <= 0:
            raise ValueError("maturity must be positive")
        self.T = float(maturity)
        self.assets = None if assets is None else tuple(int(i) for i in assets)
        if self.assets is not None and (
            not self.assets or len(set(self.assets)) != len(self.assets) or min(self.assets) < 0
        ):
            raise ValueError("assets must be distinct non-negative indices")
        self.reference = _reference(reference)

    @property
    def fixing_times(self) -> FloatArray:
        return np.array([self.T])

    def with_reference(self, levels: Sequence[float]) -> Self:
        out = copy.copy(self)
        out.reference = _reference(levels)
        return out

    def levels(self, paths: MultiPathSet, col: int) -> FloatArray:
        """``S_i(col)/ref_i`` for the assets of the product: ``(n_paths, |A|)``."""
        spots = paths.spots(col)
        if self.reference is None:
            ref = paths.spots(0)
        else:
            if len(self.reference) != spots.shape[1]:
                raise ValueError("one reference level per asset")
            ref = np.asarray(self.reference, dtype=np.float64)[None, :]
        x = np.asarray(spots / ref, dtype=np.float64)
        if self.assets is None:
            return x
        if max(self.assets) >= x.shape[1]:
            raise ValueError("an asset index is beyond the paths' assets")
        return np.asarray(x[:, list(self.assets)], dtype=np.float64)

    def _set_repr(self) -> str:
        which = "all names" if self.assets is None else f"assets {list(self.assets)}"
        return f"{which}, reference {'spot' if self.reference is None else 'given levels'}"


class WorstOf(_Levels):
    """The option on the worst performer: ``N · DF(T) · (cp · (m − K))⁺`` with
    ``m = min_{i ∈ A} S_i(T)/ref_i``.

    ``knock_in = H``: the option pays only if ``min_{t ∈ monitoring} min_{i ∈ A} S_i(t)/ref_i ≤
    H`` (discrete monitoring on the given times, which become fixings; default: the maturity
    alone).  Tests (``tests/test_local_correlation.py::test_extremum_and_relative_products``):
    one asset gives the vanilla; for two assets ``(M − K)⁺ + (m − K)⁺ = (X₁ − K)⁺ + (X₂ − K)⁺``
    path by path; ``H = 0`` gives 0 and ``H = ∞`` the European; the worst-of call is at most the
    smallest single-name call."""

    def __init__(
        self,
        cp: int | str,
        strike: float,
        maturity: float,
        discount: DiscountCurve,
        *,
        assets: Sequence[int] | None = None,
        knock_in: float | None = None,
        monitoring: ArrayLike | None = None,
        notional: float = 1.0,
        reference: Sequence[float] | None = None,
    ) -> None:
        super().__init__(maturity, discount, assets=assets, notional=notional, reference=reference)
        self.cp, self.strike = parse_cp(cp), float(strike)
        if knock_in is not None and not knock_in >= 0:
            raise ValueError("the knock-in level must be non-negative")
        if monitoring is not None and knock_in is None:
            raise ValueError("monitoring dates need a knock-in level")
        self.knock_in = None if knock_in is None else float(knock_in)
        times = np.array([self.T]) if monitoring is None else np.asarray(monitoring, dtype=float)
        times = np.unique(times.ravel())
        if times.size == 0 or times[0] <= 0 or times[-1] > self.T + 1e-12:
            raise ValueError("monitoring dates must lie in (0, maturity]")
        self.monitoring = times

    @property
    def fixing_times(self) -> FloatArray:
        return np.unique(np.concatenate([self.monitoring, [self.T]]))

    def payoff(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        worst = self.levels(paths, idx[self.T]).min(axis=1)
        cf = np.maximum(self.cp * (worst - self.strike), 0.0)
        if self.knock_in is not None:
            low = np.full(paths.n_paths, np.inf)
            for t in self.monitoring:
                low = np.minimum(low, self.levels(paths, idx[float(t)]).min(axis=1))
            cf = np.where(low <= self.knock_in, cf, 0.0)
        return np.asarray(self.notional * self.df(self.T) * cf, dtype=np.float64)

    def __repr__(self) -> str:
        kind = "call" if self.cp > 0 else "put"
        barrier = (
            ""
            if self.knock_in is None
            else f", knock-in at {self.knock_in:g} on {self.monitoring.size} dates"
        )
        return (
            f"Worst-of {kind}: strike {self.strike:g}{barrier}, expiry {self.T:g}y, "
            f"{self._set_repr()}, notional {self.notional:g}"
        )


class BestOf(_Levels):
    """The option on the best performer: ``N · DF(T) · (cp · (M − K))⁺`` with
    ``M = max_{i ∈ A} S_i(T)/ref_i`` (tests: with :class:`WorstOf`)."""

    def __init__(
        self,
        cp: int | str,
        strike: float,
        maturity: float,
        discount: DiscountCurve,
        *,
        assets: Sequence[int] | None = None,
        notional: float = 1.0,
        reference: Sequence[float] | None = None,
    ) -> None:
        super().__init__(maturity, discount, assets=assets, notional=notional, reference=reference)
        self.cp, self.strike = parse_cp(cp), float(strike)

    def payoff(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        best = self.levels(paths, idx[self.T]).max(axis=1)
        cf = np.maximum(self.cp * (best - self.strike), 0.0)
        return np.asarray(self.notional * self.df(self.T) * cf, dtype=np.float64)

    def __repr__(self) -> str:
        kind = "call" if self.cp > 0 else "put"
        return (
            f"Best-of {kind}: strike {self.strike:g}, expiry {self.T:g}y, {self._set_repr()}, "
            f"notional {self.notional:g}"
        )


class _Relative(MultiAssetProduct):
    """Products on performances relative to a reference performance ``R_ref``: the basket's,
    ``Σ_j w_j R_j`` (``against="basket"``, with ``weights``), or another asset's, ``R_e``
    (``against=e``: any simulated asset, an external one with zero index weight included)."""

    def __init__(
        self,
        maturity: float,
        discount: DiscountCurve,
        *,
        against: str | int = "basket",
        weights: ArrayLike | None = None,
        notional: float = 1.0,
        reference: Sequence[float] | None = None,
    ) -> None:
        super().__init__(discount, notional)
        if not np.isfinite(maturity) or maturity <= 0:
            raise ValueError("maturity must be positive")
        self.T = float(maturity)
        if against == "basket":
            if weights is None:
                raise ValueError('against="basket" needs the basket weights')
            self.weights: FloatArray | None = _weights(weights)
        elif (
            isinstance(against, (int, np.integer))
            and not isinstance(against, bool)
            and against >= 0
        ):
            if weights is not None:
                raise ValueError("weights are read only against the basket")
            self.weights = None
            against = int(against)
        else:
            raise ValueError('against must be "basket" or the index of an asset')
        self.against: str | int = against
        self.reference = _reference(reference)

    @property
    def fixing_times(self) -> FloatArray:
        return np.array([self.T])

    def with_reference(self, levels: Sequence[float]) -> Self:
        out = copy.copy(self)
        out.reference = _reference(levels)
        return out

    def performances(self, paths: MultiPathSet, idx: FixingIndex) -> tuple[FloatArray, FloatArray]:
        """``(R, R_ref)``: every asset's performance ``(n_paths, n)`` and the reference's."""
        r = paths.performances(idx[self.T], self.reference)
        if self.weights is not None:
            if self.weights.size != r.shape[1]:
                raise ValueError("the paths carry a different number of assets than the weights")
            return r, np.asarray(r @ self.weights, dtype=np.float64)
        if int(self.against) >= r.shape[1]:
            raise ValueError("the reference asset is beyond the paths' assets")
        return r, np.asarray(r[:, int(self.against)], dtype=np.float64)

    def _against_repr(self) -> str:
        return "the basket" if self.weights is not None else f"asset {self.against}"


class OutperformanceOption(_Relative):
    """``N · DF(T) · (cp · (R_a − R_ref − K))⁺``: asset ``a`` against the basket or against
    another asset.

    Test (``tests/test_local_correlation.py::test_margrabe_and_correlation_estimators``): two
    Black–Scholes names, ``K = 0``, one against the other — Margrabe's exchange option
    (:func:`volsto.multi.analytics.margrabe_exchange`), within 3 standard errors."""

    def __init__(
        self,
        asset: int,
        cp: int | str,
        strike: float,
        maturity: float,
        discount: DiscountCurve,
        *,
        against: str | int = "basket",
        weights: ArrayLike | None = None,
        notional: float = 1.0,
        reference: Sequence[float] | None = None,
    ) -> None:
        super().__init__(
            maturity,
            discount,
            against=against,
            weights=weights,
            notional=notional,
            reference=reference,
        )
        if asset < 0:
            raise ValueError("asset must be a non-negative index")
        self.asset, self.cp, self.strike = int(asset), parse_cp(cp), float(strike)

    def payoff(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        r, ref = self.performances(paths, idx)
        if self.asset >= r.shape[1]:
            raise ValueError("the asset is beyond the paths' assets")
        cf = np.maximum(self.cp * (r[:, self.asset] - ref - self.strike), 0.0)
        return np.asarray(self.notional * self.df(self.T) * cf, dtype=np.float64)

    def __repr__(self) -> str:
        kind = "call" if self.cp > 0 else "put"
        return (
            f"Outperformance {kind}: asset {self.asset} against {self._against_repr()}, strike "
            f"{self.strike:g}, expiry {self.T:g}y, notional {self.notional:g}"
        )


class RelativePerformanceStraddle(_Relative):
    """``N · DF(T) · Σ_{i ∈ A} v_i · |R_i − β_i R_ref − K|`` (``v_i`` equal by default,
    ``β_i = 1``).

    Tests (``tests/test_local_correlation.py::test_extremum_and_relative_products``): ``β = 0``
    gives :class:`SingleNameStraddles`; one name against itself with ``β = 1`` gives ``|K|``."""

    def __init__(
        self,
        assets: Sequence[int],
        maturity: float,
        discount: DiscountCurve,
        *,
        betas: ArrayLike | None = None,
        name_weights: ArrayLike | None = None,
        against: str | int = "basket",
        weights: ArrayLike | None = None,
        strike: float = 0.0,
        notional: float = 1.0,
        reference: Sequence[float] | None = None,
    ) -> None:
        super().__init__(
            maturity,
            discount,
            against=against,
            weights=weights,
            notional=notional,
            reference=reference,
        )
        self.assets = tuple(int(i) for i in assets)
        if not self.assets or min(self.assets) < 0:
            raise ValueError("assets must be non-negative indices")
        n = len(self.assets)
        self.betas = np.ones(n) if betas is None else _weights(betas, n)
        self.name_weights = (
            np.full(n, 1.0 / n) if name_weights is None else _weights(name_weights, n)
        )
        self.strike = float(strike)

    def payoff(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        r, ref = self.performances(paths, idx)
        if max(self.assets) >= r.shape[1]:
            raise ValueError("an asset is beyond the paths' assets")
        legs = np.abs(r[:, list(self.assets)] - self.betas[None, :] * ref[:, None] - self.strike)
        return np.asarray(
            self.notional * self.df(self.T) * (legs @ self.name_weights), dtype=np.float64
        )

    def __repr__(self) -> str:
        return (
            f"Relative performance straddles: {len(self.assets)} names against "
            f"{self._against_repr()}, strike {self.strike:g}, expiry {self.T:g}y, notional "
            f"{self.notional:g}"
        )


def _schedule(fixing_times: ArrayLike) -> FloatArray:
    t = np.asarray(fixing_times, dtype=np.float64).ravel()
    if t.size < 2 or np.any(np.diff(t) <= 0) or t[0] < 0:
        raise ValueError("fixing_times must be increasing with at least two dates")
    return t if t[0] == 0.0 else np.concatenate(([0.0], t))


class CorrelationSwap(MultiAssetProduct):
    """``N · DF(T) · (ρ̂ − K)``: the realised correlation of the names on a fixing schedule.

    With the log returns ``r_{i,m} = ln(S_i(t_m)/S_i(t_{m−1}))``, ``m = 1..M``, the realised
    variances ``σ̂_i² = (A/M) Σ_m r_{i,m}²`` and the performance basket ``Σ_i w_i S_i(t)/S_i(0)``:

    * ``estimator="cboe"``: ``ρ̂ = (σ̂_B² − Σ w_i² σ̂_i²) / ((Σ w_i σ̂_i)² − Σ w_i² σ̂_i²)`` — the
      study's realised correlation (``disp_payoff.daily_stats(...)["rho_real"]``, equal path by
      path);
    * ``estimator="pairwise"``: ``ρ̂ = Σ_{i<j} w_i w_j ρ̂_ij / Σ_{i<j} w_i w_j`` with ``ρ̂_ij =
      Σ_m r_{i,m} r_{j,m} / √(Σ_m r_{i,m}² · Σ_m r_{j,m}²)`` (zero-mean; ``demean=True``
      subtracts the window means: Pearson).  Computed as ``(Σ_m (Σ_i w_i z_{i,m})² − Σ w_i²) /
      ((Σ w_i)² − Σ w_i²)`` with ``z_i`` the returns normalised over the window (derived: the
      same double sum).

    Tests (``tests/test_local_correlation.py::test_margrabe_and_correlation_estimators``): the
    two forms of each estimator; in a constant-correlation Black–Scholes world with 63 daily
    fixings both are within 0.02 of ``ρ`` (the finite-sample bias is ``O(1/M)``)."""

    def __init__(
        self,
        weights: ArrayLike,
        fixing_times: ArrayLike,
        strike: float,
        discount: DiscountCurve,
        *,
        estimator: str = "cboe",
        demean: bool = False,
        annualisation: float = 252.0,
        notional: float = 1.0,
    ) -> None:
        super().__init__(discount, notional)
        self.weights = _weights(weights)
        if self.weights.size < 2:
            raise ValueError("a correlation needs at least two names")
        if estimator not in ("cboe", "pairwise"):
            raise ValueError('estimator must be "cboe" or "pairwise"')
        if demean and estimator != "pairwise":
            raise ValueError("demean applies to the pairwise estimator")
        self.schedule = _schedule(fixing_times)
        self.strike, self.estimator, self.demean = float(strike), estimator, bool(demean)
        self.annualisation = float(annualisation)

    @property
    def fixing_times(self) -> FloatArray:
        return self.schedule

    def realised_correlation(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        """``ρ̂`` per path."""
        cols = idx.indices(self.schedule)
        w = self.weights
        if paths.n_assets != w.size:
            raise ValueError("the paths carry a different number of assets than the weights")
        r = np.stack([np.diff(p.log_spot_at(cols), axis=1) for p in paths.assets], axis=2)
        if self.estimator == "cboe":
            scale = self.annualisation / (cols.size - 1)
            vols = np.sqrt(scale * np.sum(r * r, axis=1))
            levels = np.column_stack([paths.basket_level(int(c), w) for c in cols])
            rb = np.diff(np.log(levels), axis=1)
            own = np.sum((w * vols) ** 2, axis=1)
            cross = (vols @ w) ** 2 - own
            with np.errstate(divide="ignore", invalid="ignore"):
                rho = (scale * np.sum(rb * rb, axis=1) - own) / cross
            return np.asarray(np.where(cross > 0, rho, np.nan), dtype=np.float64)
        if self.demean:
            r = r - r.mean(axis=1, keepdims=True)
        norm = np.sqrt(np.sum(r * r, axis=1))
        with np.errstate(divide="ignore", invalid="ignore"):
            z = r / norm[:, None, :]
        total = np.sum((z @ w) ** 2, axis=1)
        w2 = float(np.sum(w * w))
        return np.asarray((total - w2) / (float(w.sum()) ** 2 - w2), dtype=np.float64)

    def payoff(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        rho = self.realised_correlation(paths, idx)
        T = float(self.schedule[-1])
        return np.asarray(self.notional * self.df(T) * (rho - self.strike), dtype=np.float64)

    def __repr__(self) -> str:
        kind = (
            "Cboe formula"
            if self.estimator == "cboe"
            else ("pairwise, demeaned" if self.demean else "pairwise, zero-mean")
        )
        return (
            f"Correlation swap ({kind}): {self.weights.size} names, {self.schedule.size - 1} "
            f"fixings to {self.schedule[-1]:g}y, strike {self.strike:g}, notional "
            f"{self.notional:g}"
        )


class BasketVarianceSwap(MultiAssetProduct):
    """``N · DF(T) · ((A/M) Σ_m ln²(B_m/B_{m−1}) − K)``: the realised variance of the performance
    basket ``B(t) = Σ_i w_i S_i(t)/S_i(0)`` on the fixing schedule against a variance strike.

    Checked against the index variance swap of the listed strip
    (:func:`volsto.market.varswap.varswap_strike` on the index target; S9,
    ``tests/test_local_correlation.py::test_s9_basket_variance_swap``)."""

    def __init__(
        self,
        weights: ArrayLike,
        fixing_times: ArrayLike,
        strike: float,
        discount: DiscountCurve,
        *,
        annualisation: float = 252.0,
        notional: float = 1.0,
    ) -> None:
        super().__init__(discount, notional)
        self.weights = _weights(weights)
        self.schedule = _schedule(fixing_times)
        self.strike, self.annualisation = float(strike), float(annualisation)

    @property
    def fixing_times(self) -> FloatArray:
        return self.schedule

    def realised_variance(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        """The annualised realised variance of the basket per path."""
        cols = idx.indices(self.schedule)
        levels = np.column_stack([paths.basket_level(int(c), self.weights) for c in cols])
        rb = np.diff(np.log(levels), axis=1)
        return np.asarray(
            self.annualisation / (cols.size - 1) * np.sum(rb * rb, axis=1), dtype=np.float64
        )

    def payoff(self, paths: MultiPathSet, idx: FixingIndex) -> FloatArray:
        T = float(self.schedule[-1])
        value = self.realised_variance(paths, idx) - self.strike
        return np.asarray(self.notional * self.df(T) * value, dtype=np.float64)

    def __repr__(self) -> str:
        return (
            f"Basket variance swap: {self.weights.size} names, {self.schedule.size - 1} fixings "
            f"to {self.schedule[-1]:g}y, variance strike {self.strike:g}, notional "
            f"{self.notional:g}"
        )


__all__ = [
    "BasketOption",
    "BasketStraddle",
    "BasketVarianceSwap",
    "BestOf",
    "CorrelationSwap",
    "DispersionGap",
    "MultiAssetProduct",
    "MultiPortfolio",
    "OutperformanceOption",
    "Palladium",
    "PalladiumCallSpread",
    "PalladiumPut",
    "RelativePerformanceStraddle",
    "SingleNameStraddles",
    "VarianceDispersion",
    "WorstOf",
    "dispersion_straddles",
]
