"""European vanilla and digital options (SPEC §6)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from volsto.products.base import Product, parse_cp, shift_times

if TYPE_CHECKING:
    from volsto.engine.grid import FixingIndex
    from volsto.engine.paths import PathSet
    from volsto.market.curves import DiscountCurve

FloatArray = NDArray[np.float64]


class EuropeanOption(Product):
    """``notional · (cp (S_T − K))⁺`` paid at ``T``."""

    def __init__(
        self,
        strike: float,
        maturity: float,
        cp: int | str,
        discount: DiscountCurve,
        notional: float = 1.0,
        *,
        seasoned: bool = False,
    ) -> None:
        super().__init__(discount, notional)
        if strike <= 0 or maturity <= 0:
            raise ValueError("strike and maturity must be positive")
        self.strike = float(strike)
        self.T = float(maturity)
        self.cp = parse_cp(cp)
        #: set by :func:`volsto.products.seasoning.season` (a knocked-in barrier option's vanilla)
        self.seasoned = bool(seasoned)

    @property
    def is_seasoned(self) -> bool:
        return self.seasoned

    @property
    def fixing_times(self) -> FloatArray:
        return np.array([self.T])

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        S = paths.spot_at(idx[self.T])
        cf = np.maximum(self.cp * (S - self.strike), 0.0)
        return np.asarray(self.notional * float(self.df(self.T)) * cf, dtype=np.float64)

    def aged(self, dt: float) -> Product:
        return EuropeanOption(
            self.strike,
            float(shift_times([self.T], dt)[0]),
            self.cp,
            self.discount,
            self.notional,
            seasoned=self.seasoned,
        )

    def __repr__(self) -> str:
        kind = "Call" if self.cp > 0 else "Put"
        return (
            f"European {kind}: strike {self.strike:g}, expiry {self.T:g}y, "
            f"notional {self.notional:g}"
        )


class DigitalOption(Product):
    """Cash-or-nothing: pays ``notional · payout`` at ``T`` if ``cp (S_T − K) > 0``."""

    def __init__(
        self,
        strike: float,
        maturity: float,
        cp: int | str,
        discount: DiscountCurve,
        payout: float = 1.0,
        notional: float = 1.0,
    ) -> None:
        super().__init__(discount, notional)
        if strike <= 0 or maturity <= 0:
            raise ValueError("strike and maturity must be positive")
        self.strike = float(strike)
        self.T = float(maturity)
        self.cp = parse_cp(cp)
        self.payout = float(payout)

    @property
    def fixing_times(self) -> FloatArray:
        return np.array([self.T])

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        S = paths.spot_at(idx[self.T])
        hit = (self.cp * (S - self.strike) > 0.0).astype(np.float64)
        return np.asarray(
            self.notional * self.payout * float(self.df(self.T)) * hit, dtype=np.float64
        )

    def aged(self, dt: float) -> Product:
        return DigitalOption(
            self.strike,
            float(shift_times([self.T], dt)[0]),
            self.cp,
            self.discount,
            self.payout,
            self.notional,
        )

    def __repr__(self) -> str:
        kind = "Call" if self.cp > 0 else "Put"
        return (
            f"Digital {kind}: strike {self.strike:g}, expiry {self.T:g}y, "
            f"payout {self.payout:g}, notional {self.notional:g}"
        )
