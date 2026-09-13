"""Volatility knock-out put (SPEC §16 / v2 §6.2).

Realised volatility over the life on the fixing schedule, ``σ_real² = (A/N) Σ_{i=1}^N r_i²``
(``r_i²`` capped at ``c²`` with ``daily_cap``); payoff ``(K − S_T)⁺ · 1{σ_real < vol_ko}``.
``monitoring = "maturity"`` (default, the traded form) checks the condition once at ``T`` on the
full-life realised vol; ``"running"`` knocks the option out on the first day the accrued variance
``Σ_{i≤m} r_i²`` exceeds ``vol_ko² N / A`` — the option is dead as soon as the full-life realised
vol can no longer end below the barrier.  **The running form is not the traded convention**; it
is a flag for the study only.  Because the accrued sum is non-decreasing, exceeding the
full-life budget at some ``m ≤ N`` is the same event as exceeding it at ``N``: the running and
maturity forms have identical terminal payoffs path by path (``tests/test_conditional_variance``
asserts the equality), and the flag only changes *when* the option dies (the ``ko_time``
statistic, the index of the first breach), which matters for risk and hedging, not for price.
A running rule on the annualised realised vol to date, ``sqrt((A/m) Σ_{i≤m} r_i²) > vol_ko``,
would be a different (and cheaper) product; it is not implemented (M4c report).
``decompose()`` = vanilla put − the "vol-knock-in" put ``(K − S_T)⁺ 1{σ_real ≥ vol_ko}``
(``knock_in=True``), exact path by path.

Why it is in the study: the price is ``E[(K − S_T)⁺ 1{RV < H²}]``, the joint law of terminal spot
and realised variance.  Under local vol the realised variance is nearly a deterministic function
of the path's spot levels, so the LV price is close to a hard threshold on ``S_T``; stochastic
vol spreads ``RV`` conditional on ``S_T`` and the spread is governed by ``ν`` and ``ρ`` — the
largest model dependence of any product in the library.  Reported through
:func:`volsto.analytics.conditional_variance.vko_report`: price, ratio to the vanilla put (the
"VKO discount") and ``P(knock-out)``.  Checked by ``tests/test_conditional_variance.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.products.base import Product
from volsto.products.conditional_variance import RealisedVarianceSchedule
from volsto.products.vanilla import EuropeanOption

if TYPE_CHECKING:
    from volsto.engine.grid import FixingIndex
    from volsto.engine.paths import PathSet
    from volsto.market.curves import DiscountCurve

FloatArray = NDArray[np.float64]

MONITORING = ("maturity", "running")


class VolKnockOutPut(RealisedVarianceSchedule):
    """``notional · (K − S_T)⁺ · 1{alive}``, ``alive = {σ_real < vol_ko}`` (module docstring)."""

    def __init__(
        self,
        strike: float,
        maturity: float,
        vol_ko: float,
        fixing_times: ArrayLike,
        discount: DiscountCurve,
        *,
        daily_cap: float | None = None,
        monitoring: str = "maturity",
        annualisation: float = 252.0,
        notional: float = 1.0,
        knock_in: bool = False,
    ) -> None:
        super().__init__(fixing_times, discount, notional, annualisation, daily_cap)
        if strike <= 0 or vol_ko < 0:
            raise ValueError("strike must be positive and vol_ko non-negative")
        if abs(self._fixings[-1] - maturity) > 1e-9:
            raise ValueError("the fixing schedule must end at the maturity")
        if monitoring not in MONITORING:
            raise ValueError(f"monitoring in {MONITORING}")
        self.strike = float(strike)
        self.T = float(maturity)
        self.vol_ko = float(vol_ko)
        self.monitoring = monitoring
        self.knock_in = bool(knock_in)

    def statistics(self, paths: PathSet, idx: FixingIndex) -> dict[str, FloatArray]:
        ls = self.log_spots(paths, idx)
        r2 = self.squared_returns(ls)
        n = self.n_returns
        budget = self.vol_ko**2 * n / self.annualisation
        cum = np.cumsum(r2, axis=1)
        if self.monitoring == "maturity":
            alive = cum[:, -1] < budget
            ko_time = np.where(alive, n + 1, n).astype(np.float64)
        else:
            breach = cum > budget
            alive = ~np.any(breach, axis=1)
            ko_time = np.where(alive, n + 1, np.argmax(breach, axis=1) + 1).astype(np.float64)
        alive_f = alive.astype(np.float64)
        intrinsic = np.maximum(self.strike - np.exp(ls[:, -1]), 0.0)
        return {
            "alive": alive_f,
            "ko": 1.0 - alive_f,
            "ko_time": ko_time,  # 1-based fixing index of the first breach; n + 1 if alive
            "rv": self.annualisation / n * np.sum(r2, axis=1),
            "put": intrinsic,
        }

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        st = self.statistics(paths, idx)
        gate = st["ko"] if self.knock_in else st["alive"]
        return np.asarray(
            self.notional * float(self.df(self.T)) * st["put"] * gate, dtype=np.float64
        )

    def vanilla(self) -> EuropeanOption:
        return EuropeanOption(self.strike, self.T, -1, self.discount, self.notional)

    def decompose(self) -> list[Product] | None:
        if self.knock_in:
            return None
        return [
            self.vanilla(),
            VolKnockOutPut(
                self.strike,
                self.T,
                self.vol_ko,
                self._fixings,
                self.discount,
                daily_cap=self.daily_cap,
                monitoring=self.monitoring,
                annualisation=self.annualisation,
                notional=-self.notional,
                knock_in=True,
            ),
        ]

    def __repr__(self) -> str:
        kind = "knock-in" if self.knock_in else "knock-out"
        cap = "" if self.daily_cap is None else f", daily cap {self.daily_cap:g}"
        return (
            f"Vol {kind} put: strike {self.strike:g}, expiry {self.T:g}y, vol barrier "
            f"{self.vol_ko * 100:.4g}% ({self.monitoring} monitoring, {self.n_returns} fixings, "
            f"A = {self.annualisation:g}{cap}), notional {self.notional:g}"
        )
