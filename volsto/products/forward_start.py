"""Forward-start (cliquet) options on the relative performance ``S_T2 / S_T1`` (SPEC §6, book §3.1).

A forward-start call of *moneyness* ``k`` pays ``(S_T2/S_T1 − k)⁺`` at ``T2`` (Bergomi §3.1, the
ATM case ``k = 1`` is eq. 3.13).  Its price depends only on the distribution of the forward
return, which is why cliquets are the natural probe of forward-smile risk (§3.1.5.2): in a
Black–Scholes model with a term structure the price is ``DF(T2) · Black(F_R, k, τ, σ̂_{T1T2})``
with ``F_R = F(T2)/F(T1)`` (``E[S_T2/S_T1]`` for deterministic rates), ``τ = T2 − T1`` and the
forward volatility ``σ̂²_{T1T2} = (T2 σ̂²_{T2} − T1 σ̂²_{T1}) / τ`` (eqs. 3.1–3.2); checked by
``tests/test_forward_start.py::test_forward_start_black_scholes``.  Inverting a strip of these
prices for a Black volatility gives the *forward smile* used by
:mod:`volsto.analytics.forward_smile`.

``t1 = 0`` is allowed (the option is then a vanilla on ``S_T2/S_0``) so that cliquet
decompositions can start at inception, and ``pay_time`` may be deferred past ``T2`` (a cliquet
settles every period's option on its own maturity date).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from volsto.products.base import Product, parse_cp, shift_times

if TYPE_CHECKING:
    from volsto.engine.grid import FixingIndex
    from volsto.engine.paths import PathSet
    from volsto.market.curves import DiscountCurve

FloatArray = NDArray[np.float64]


def _check_dates(t1: float, t2: float, pay_time: float | None) -> float:
    if t1 < 0 or t2 <= t1:
        raise ValueError("need 0 ≤ t1 < t2")
    pay = float(t2 if pay_time is None else pay_time)
    if pay < t2:
        raise ValueError("pay_time cannot precede t2")
    return pay


class ForwardStartOption(Product):
    """``notional · (cp (S_T2/S_T1 − k))⁺`` paid at ``pay_time`` (default ``T2``).

    ``k`` is the moneyness strike (``k = 1``: struck at the money on ``T1``'s fixing); ``k = 0``
    is allowed for a call, which then pays the forward return itself (used by the additive
    cliquet decomposition when there is no local floor).
    """

    def __init__(
        self,
        t1: float,
        t2: float,
        strike: float,
        cp: int | str,
        discount: DiscountCurve,
        notional: float = 1.0,
        pay_time: float | None = None,
    ) -> None:
        super().__init__(discount, notional)
        self.T1 = float(t1)
        self.T2 = float(t2)
        self.pay_time = _check_dates(self.T1, self.T2, pay_time)
        self.cp = parse_cp(cp)
        if strike < 0 or (strike == 0 and self.cp < 0):
            raise ValueError("strike must be positive (zero allowed for a call)")
        self.strike = float(strike)

    @property
    def fixing_times(self) -> FloatArray:
        return np.array([self.T1, self.T2])

    @property
    def pay_times(self) -> FloatArray:
        return np.array([self.pay_time])

    @property
    def tau(self) -> float:
        """Residual maturity ``T2 − T1`` of the option once it starts."""
        return self.T2 - self.T1

    def forward_return(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        return np.exp(paths.log_return(idx[self.T1], idx[self.T2]))

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        r = self.forward_return(paths, idx)
        cf = np.maximum(self.cp * (r - self.strike), 0.0)
        return np.asarray(self.notional * float(self.df(self.pay_time)) * cf, dtype=np.float64)

    def aged(self, dt: float) -> Product:
        t = shift_times([self.T1, self.T2, self.pay_time], dt)
        return ForwardStartOption(
            float(t[0]),
            float(t[1]),
            self.strike,
            self.cp,
            self.discount,
            self.notional,
            float(t[2]),
        )

    def __repr__(self) -> str:
        kind = "Call" if self.cp > 0 else "Put"
        pay = "" if self.pay_time == self.T2 else f", paid at {self.pay_time:g}y"
        return (
            f"Forward-start {kind}: (S_T2/S_T1 - {self.strike:g})+ with T1 {self.T1:g}y, "
            f"T2 {self.T2:g}y{pay}, notional {self.notional:g}"
        )


class ForwardStartStraddle(Product):
    """``notional · |S_T2/S_T1 − k|`` paid at ``pay_time`` (default ``T2``); ``decompose()`` is
    the call plus the put."""

    def __init__(
        self,
        t1: float,
        t2: float,
        strike: float,
        discount: DiscountCurve,
        notional: float = 1.0,
        pay_time: float | None = None,
    ) -> None:
        super().__init__(discount, notional)
        self.T1 = float(t1)
        self.T2 = float(t2)
        self.pay_time = _check_dates(self.T1, self.T2, pay_time)
        if strike <= 0:
            raise ValueError("strike must be positive")
        self.strike = float(strike)

    @property
    def fixing_times(self) -> FloatArray:
        return np.array([self.T1, self.T2])

    @property
    def pay_times(self) -> FloatArray:
        return np.array([self.pay_time])

    @property
    def tau(self) -> float:
        return self.T2 - self.T1

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        r = np.exp(paths.log_return(idx[self.T1], idx[self.T2]))
        return np.asarray(
            self.notional * float(self.df(self.pay_time)) * np.abs(r - self.strike),
            dtype=np.float64,
        )

    def decompose(self) -> list[Product]:
        return [
            ForwardStartOption(
                self.T1, self.T2, self.strike, cp, self.discount, self.notional, self.pay_time
            )
            for cp in (1, -1)
        ]

    def aged(self, dt: float) -> Product:
        t = shift_times([self.T1, self.T2, self.pay_time], dt)
        return ForwardStartStraddle(
            float(t[0]), float(t[1]), self.strike, self.discount, self.notional, float(t[2])
        )

    def __repr__(self) -> str:
        pay = "" if self.pay_time == self.T2 else f", paid at {self.pay_time:g}y"
        return (
            f"Forward-start straddle: |S_T2/S_T1 - {self.strike:g}| with T1 {self.T1:g}y, "
            f"T2 {self.T2:g}y{pay}, notional {self.notional:g}"
        )


def forward_start_strip(
    t1: float,
    t2: float,
    strikes: Iterable[float],
    cp: int | str,
    discount: DiscountCurve,
    notional: float = 1.0,
) -> list[ForwardStartOption]:
    """Forward-start options of one type over a strike grid (the forward-smile probe)."""
    return [ForwardStartOption(t1, t2, float(k), cp, discount, notional) for k in strikes]
