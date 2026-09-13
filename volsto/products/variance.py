"""Variance and volatility swaps (SPEC §6).

Realised variance is computed on the **fixing** dates (daily by default), not on the simulation
grid:  ``RV = A · Σ_i ln²(S_{t_i}/S_{t_{i-1}})`` with annualisation ``A = 1/(t_n − t_0)`` by
default or ``A = annualisation / n`` (e.g. 252/n) when given.  A swap whose first fixing is at
``T₁ > 0`` is a forward variance swap over ``[T₁, T₂]``.  FVA (forward-start straddle vs forward
vol) is added with the forward-start products in M4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.products.base import Product, daily_schedule

if TYPE_CHECKING:
    from volsto.engine.grid import FixingIndex
    from volsto.engine.paths import PathSet
    from volsto.market.curves import DiscountCurve

FloatArray = NDArray[np.float64]


class _RealisedVarianceProduct(Product):
    def __init__(
        self,
        fixing_times: ArrayLike,
        discount: DiscountCurve,
        notional: float,
        annualisation: float | None,
        use_simulation_grid: bool,
    ) -> None:
        super().__init__(discount, notional)
        ft = np.unique(np.asarray(fixing_times, dtype=np.float64))
        if ft.size < 2 or ft[0] < 0:
            raise ValueError("need at least two non-negative fixing times")
        self._fixings = ft
        self.annualisation = annualisation
        self.use_simulation_grid = use_simulation_grid

    @property
    def fixing_times(self) -> FloatArray:
        return self._fixings

    @property
    def start(self) -> float:
        return float(self._fixings[0])

    @property
    def n_returns(self) -> int:
        return int(self._fixings.size - 1)

    @property
    def annualisation_factor(self) -> float:
        if self.annualisation is None:
            return 1.0 / (self.maturity - self.start)
        return self.annualisation / self.n_returns

    def realised_variance(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        """Annualised realised variance per path."""
        if self.use_simulation_grid:
            rv = paths.realised_variance_grid(idx[self.start], idx[self.maturity])
        else:
            rv = paths.realised_variance_fixings(idx.indices(self._fixings))
        return self.annualisation_factor * rv


class VarianceSwap(_RealisedVarianceProduct):
    """Pays ``notional · (RV − strike)`` at the last fixing; ``strike`` in variance units.

    ``notional`` is the variance notional (vega notional / (2 K_vol) in market convention).
    """

    def __init__(
        self,
        fixing_times: ArrayLike,
        strike: float,
        discount: DiscountCurve,
        notional: float = 1.0,
        annualisation: float | None = None,
        use_simulation_grid: bool = False,
    ) -> None:
        super().__init__(fixing_times, discount, notional, annualisation, use_simulation_grid)
        if strike < 0:
            raise ValueError("variance strike must be non-negative")
        self.strike = float(strike)

    @classmethod
    def daily(
        cls,
        maturity: float,
        strike: float,
        discount: DiscountCurve,
        *,
        start: float = 0.0,
        per_year: int = 252,
        notional: float = 1.0,
        annualisation: float | None = None,
    ) -> VarianceSwap:
        return cls(
            daily_schedule(maturity, per_year, start), strike, discount, notional, annualisation
        )

    def floating_leg(self) -> VarianceSwap:
        """Same swap with zero strike: its price / DF(T) / notional is the fair strike."""
        return VarianceSwap(
            self._fixings,
            0.0,
            self.discount,
            self.notional,
            self.annualisation,
            self.use_simulation_grid,
        )

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        rv = self.realised_variance(paths, idx)
        return np.asarray(
            self.notional * float(self.df(self.maturity)) * (rv - self.strike), dtype=np.float64
        )

    def __repr__(self) -> str:
        kind = "Forward variance swap" if self.start > 0 else "Variance swap"
        return (
            f"{kind}: [{self.start:g}y, {self.maturity:g}y], {self.n_returns} returns, "
            f"strike {np.sqrt(self.strike) * 100:.4g}% vol ({self.strike:.6g} var), "
            f"variance notional {self.notional:g}"
        )


def ForwardVarianceSwap(
    start: float,
    maturity: float,
    strike: float,
    discount: DiscountCurve,
    *,
    per_year: int = 252,
    notional: float = 1.0,
    annualisation: float | None = None,
) -> VarianceSwap:
    """Variance swap over ``[start, maturity]`` with daily fixings."""
    if start <= 0:
        raise ValueError("forward variance swap needs start > 0")
    return VarianceSwap.daily(
        maturity,
        strike,
        discount,
        start=start,
        per_year=per_year,
        notional=notional,
        annualisation=annualisation,
    )


class VolSwap(_RealisedVarianceProduct):
    """Pays ``notional · (sqrt(RV) − strike_vol)`` at the last fixing."""

    def __init__(
        self,
        fixing_times: ArrayLike,
        strike_vol: float,
        discount: DiscountCurve,
        notional: float = 1.0,
        annualisation: float | None = None,
        use_simulation_grid: bool = False,
    ) -> None:
        super().__init__(fixing_times, discount, notional, annualisation, use_simulation_grid)
        if strike_vol < 0:
            raise ValueError("vol strike must be non-negative")
        self.strike_vol = float(strike_vol)

    @classmethod
    def daily(
        cls,
        maturity: float,
        strike_vol: float,
        discount: DiscountCurve,
        *,
        start: float = 0.0,
        per_year: int = 252,
        notional: float = 1.0,
        annualisation: float | None = None,
    ) -> VolSwap:
        return cls(
            daily_schedule(maturity, per_year, start), strike_vol, discount, notional, annualisation
        )

    def floating_leg(self) -> VolSwap:
        return VolSwap(
            self._fixings,
            0.0,
            self.discount,
            self.notional,
            self.annualisation,
            self.use_simulation_grid,
        )

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        rv = self.realised_variance(paths, idx)
        return np.asarray(
            self.notional * float(self.df(self.maturity)) * (np.sqrt(rv) - self.strike_vol),
            dtype=np.float64,
        )

    def __repr__(self) -> str:
        kind = "Forward vol swap" if self.start > 0 else "Vol swap"
        return (
            f"{kind}: [{self.start:g}y, {self.maturity:g}y], {self.n_returns} returns, "
            f"strike {self.strike_vol * 100:.4g}% vol, vol notional {self.notional:g}"
        )
