"""Product interface (SPEC §6).

A product declares its fixing times and pay times and maps a :class:`~volsto.engine.paths.PathSet`
to *discounted* cash flows per path.  Discounting is done inside ``payoff`` with the product's
:class:`~volsto.market.curves.DiscountCurve`.  ``decompose()`` optionally returns simpler
products whose prices sum to the product's price.  ``__repr__`` reads like a term sheet.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from volsto.engine.grid import FixingIndex
    from volsto.engine.paths import PathSet
    from volsto.market.curves import DiscountCurve

FloatArray = NDArray[np.float64]


def parse_cp(cp: int | float | str) -> int:
    """Normalise ``'call'/'put'/'c'/'p'/±1`` to ``+1`` / ``-1``."""
    if isinstance(cp, str):
        s = cp.strip().lower()
        if s in ("call", "c"):
            return 1
        if s in ("put", "p"):
            return -1
        raise ValueError(f"unknown option type {cp!r}")
    if cp in (1, -1):
        return int(cp)
    raise ValueError(f"cp must be ±1 or 'call'/'put', got {cp!r}")


def uniform_schedule(maturity: float, n_periods: int, start: float = 0.0) -> FloatArray:
    """``n_periods + 1`` equally spaced fixing times from ``start`` to ``maturity`` inclusive."""
    if maturity <= start or n_periods < 1:
        raise ValueError("need maturity > start and n_periods ≥ 1")
    return np.linspace(start, maturity, n_periods + 1)


def daily_schedule(maturity: float, per_year: int = 252, start: float = 0.0) -> FloatArray:
    """Daily fixings (``per_year`` per year) from ``start`` to ``maturity`` inclusive."""
    n = max(1, round((maturity - start) * per_year))
    return uniform_schedule(maturity, n, start)


class Product(ABC):
    """Abstract single-underlying product."""

    #: Set to ``True`` when the payoff needs the spot at every simulation step between fixings
    #: (e.g. Brownian-bridge barrier corrections); the engine then records all grid steps.
    requires_all_steps: bool = False

    def __init__(self, discount: DiscountCurve, notional: float = 1.0) -> None:
        if not np.isfinite(notional):
            raise ValueError("notional must be finite")
        self.discount = discount
        self.notional = float(notional)

    @property
    @abstractmethod
    def fixing_times(self) -> FloatArray:
        """Sorted observation times (years) the path container must record."""

    @property
    def pay_times(self) -> FloatArray:
        """Payment times; default: the last fixing."""
        return np.array([float(self.fixing_times[-1])])

    @property
    def maturity(self) -> float:
        return float(np.max(self.pay_times))

    @abstractmethod
    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        """Discounted cash flows per path, shape ``(n_paths,)``."""

    def decompose(self) -> list[Product] | None:
        """Simpler products whose prices sum to this one (``None`` if not available)."""
        return None

    def df(self, t: ArrayLike) -> FloatArray:
        return self.discount.df(t)

    @abstractmethod
    def __repr__(self) -> str: ...


class CashFlow(Product):
    """Deterministic amount ``notional · amount`` paid at ``pay_time`` (decomposition building
    block: the cash leg of a cliquet or of a forward volatility agreement).  Its only fixing is
    the pay date, so it never extends a grid beyond the product it decomposes."""

    def __init__(
        self, amount: float, pay_time: float, discount: DiscountCurve, notional: float = 1.0
    ) -> None:
        super().__init__(discount, notional)
        if pay_time <= 0 or not np.isfinite(amount):
            raise ValueError("pay_time must be positive and amount finite")
        self.amount = float(amount)
        self.T = float(pay_time)

    @property
    def fixing_times(self) -> FloatArray:
        return np.array([self.T])

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        return np.full(paths.n_paths, self.notional * self.amount * float(self.df(self.T)))

    def __repr__(self) -> str:
        return f"Cash flow: {self.amount:g} x notional {self.notional:g} paid at {self.T:g}y"
