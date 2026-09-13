"""Volatility knock-out put (SPEC v2 §6.2).

Realised volatility over the life on the fixing schedule, ``σ_real² = (A/N) Σ_{i=1}^N r_i²``
(``r_i²`` capped at ``c²`` with ``daily_cap``); payoff ``(K − S_T)⁺ · 1{σ_real < vol_ko}``, the
condition checked once at maturity (the traded form; a "running" flag was dropped at the M4c
review because the accrued sum is monotone, so any running check on the full-life budget is the
same event).  The ``ko_time`` statistic — the first fixing at which the accrued variance exceeds
the budget ``vol_ko² N/A``, i.e. the day the knock-out becomes certain — is kept for risk and
hedging.  ``decompose()`` = vanilla put − the "vol-knock-in" put ``(K − S_T)⁺ 1{σ_real ≥ vol_ko}``
(``knock_in=True``), exact path by path.

Why it is in the study: the price is ``E[(K − S_T)⁺ 1{RV < H²}]``, the joint law of terminal spot
and realised variance.  Under local vol the realised variance is nearly a deterministic function
of the path's spot levels; stochastic vol spreads ``RV`` conditional on ``S_T`` around that level,
which is sign-indefinite for the price: the LSV-versus-LV direction depends on where the barrier
sits relative to the ITM-conditional realised-vol distribution (owner, M4c review).  Hence
:func:`volsto.analytics.conditional_variance.vko_analysis` reports, on one path set, the
``vol_ko`` sweep of the ratio to the vanilla put and the distribution of ``σ_real`` conditional on
``S_T < K`` (10/50/90 percentiles and ``P(σ_real > vol_ko | ITM)``).  Checked by
``tests/test_conditional_variance.py``.
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
        annualisation: float = 252.0,
        notional: float = 1.0,
        knock_in: bool = False,
    ) -> None:
        super().__init__(fixing_times, discount, notional, annualisation, daily_cap)
        if strike <= 0 or vol_ko < 0:
            raise ValueError("strike must be positive and vol_ko non-negative")
        if abs(self._fixings[-1] - maturity) > 1e-9:
            raise ValueError("the fixing schedule must end at the maturity")
        self.strike = float(strike)
        self.T = float(maturity)
        self.vol_ko = float(vol_ko)
        self.knock_in = bool(knock_in)

    def statistics(self, paths: PathSet, idx: FixingIndex) -> dict[str, FloatArray]:
        ls = self.log_spots(paths, idx)
        r2 = self.squared_returns(ls)
        n = self.n_returns
        budget = self.vol_ko**2 * n / self.annualisation
        cum = np.cumsum(r2, axis=1)
        alive = cum[:, -1] < budget
        breach = cum > budget
        ko_time = np.where(alive, n + 1, np.argmax(breach, axis=1) + 1).astype(np.float64)
        alive_f = alive.astype(np.float64)
        s_t = np.exp(ls[:, -1])
        return {
            "alive": alive_f,
            "ko": 1.0 - alive_f,
            "ko_time": ko_time,  # 1-based fixing index at which the knock-out became certain
            "rv": self.annualisation / n * cum[:, -1],
            "put": np.maximum(self.strike - s_t, 0.0),
            "itm": (s_t < self.strike).astype(np.float64),
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
            f"{self.vol_ko * 100:.4g}% checked at maturity ({self.n_returns} fixings, "
            f"A = {self.annualisation:g}{cap}), notional {self.notional:g}"
        )
