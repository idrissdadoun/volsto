"""Variance and volatility swaps (SPEC §6).

Realised variance is computed on the **fixing** dates (daily by default), not on the simulation
grid:  ``RV = A · Σ_i ln²(S_{t_i}/S_{t_{i-1}})`` with annualisation ``A = 1/(t_n − t_0)`` by
default or ``A = annualisation / n`` (e.g. 252/n) when given.  A swap whose first fixing is at
``T₁ > 0`` is a forward variance swap over ``[T₁, T₂]``.  :class:`FVA` is the forward volatility
agreement on the relative performance (a forward on the ``T₁``-dated ATM-forward straddle).

Seasoned swaps (M10 Part 3, SPEC §6.10; built by :func:`volsto.products.seasoning.season`): the
state inputs ``reference_fixing`` (the last realised close ``S_ref``; the first remaining return
is ``ln(S_{t_1}/S_ref)``, so a spot bump never moves a realised fixing), ``realised_sum_sq``
(``Σ r_i²`` over the ``realised_count`` realised returns) and ``inception`` (the contract's first
fixing time, negative once seasoned: ``A = 1/(t_n − inception)``) give ``RV = A (realised_sum_sq +
Σ_future r_i²)`` with ``n = realised_count + future returns``.  Their defaults (``None``, 0, 0,
``None``) are the fresh swap bit for bit (``tests/test_seasoning.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.market.bs import black_price
from volsto.products.base import CashFlow, Product, daily_schedule, shift_times
from volsto.products.forward_start import ForwardStartStraddle

if TYPE_CHECKING:
    from volsto.engine.grid import FixingIndex
    from volsto.engine.paths import PathSet
    from volsto.market.curves import DiscountCurve, ForwardCurve

FloatArray = NDArray[np.float64]


class _RealisedVarianceProduct(Product):
    def __init__(
        self,
        fixing_times: ArrayLike,
        discount: DiscountCurve,
        notional: float,
        annualisation: float | None,
        use_simulation_grid: bool,
        *,
        reference_fixing: float | None = None,
        realised_sum_sq: float = 0.0,
        realised_count: int = 0,
        inception: float | None = None,
        seasoned: bool = False,
    ) -> None:
        super().__init__(discount, notional)
        ft = np.unique(np.asarray(fixing_times, dtype=np.float64))
        if reference_fixing is None:
            if ft.size < 2 or ft[0] < 0:
                raise ValueError("need at least two non-negative fixing times")
        elif ft.size < 1 or ft[0] <= 0:
            raise ValueError("a seasoned swap needs its remaining fixings after the time origin")
        self._fixings = ft
        self.annualisation = annualisation
        self.use_simulation_grid = use_simulation_grid
        self.reference_fixing, self.realised_sum_sq, self.realised_count = realised_state(
            reference_fixing, realised_sum_sq, realised_count
        )
        if inception is not None and not (np.isfinite(inception) and inception <= ft[0]):
            raise ValueError("inception must be finite and not after the first fixing")
        self.inception = None if inception is None else float(inception)
        self.seasoned = bool(seasoned)
        if use_simulation_grid and self.is_seasoned:
            raise ValueError("a seasoned swap is realised on its fixings, not the simulation grid")

    @property
    def fixing_times(self) -> FloatArray:
        return self._fixings

    @property
    def is_seasoned(self) -> bool:
        """Whether the swap was seasoned (``seasoned``, set by
        :func:`volsto.products.seasoning.season`, or any realised state; the defaults are the
        fresh swap)."""
        return self.seasoned or self.reference_fixing is not None or self.inception is not None

    @property
    def start(self) -> float:
        """The contract's first fixing time (``inception`` once seasoned)."""
        return float(self._fixings[0]) if self.inception is None else self.inception

    @property
    def n_returns(self) -> int:
        """Returns over the contract's life, realised ones included."""
        future = self._fixings.size - (1 if self.reference_fixing is None else 0)
        return int(self.realised_count + future)

    @property
    def annualisation_factor(self) -> float:
        if self.annualisation is None:
            return 1.0 / (self.maturity - self.start)
        return self.annualisation / self.n_returns

    def state_kwargs(self) -> dict[str, Any]:
        """The realised-state constructor arguments (for :meth:`aged` and the legs)."""
        return {
            "reference_fixing": self.reference_fixing,
            "realised_sum_sq": self.realised_sum_sq,
            "realised_count": self.realised_count,
            "inception": self.inception,
            "seasoned": self.seasoned,
        }

    def realised_variance(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        """Annualised realised variance per path (the realised part included)."""
        if self.use_simulation_grid:
            rv = paths.realised_variance_grid(idx[self.start], idx[self.maturity])
        elif self.reference_fixing is None:
            rv = self.realised_sum_sq + paths.realised_variance_fixings(idx.indices(self._fixings))
        else:
            ls = paths.log_spot_at(idx.indices(self._fixings))
            r = np.diff(with_reference(ls, self.reference_fixing), axis=1)
            rv = self.realised_sum_sq + np.sum(r * r, axis=1)
        return self.annualisation_factor * rv

    def _seasoned_repr(self) -> str:
        if not self.is_seasoned:
            return ""
        ref = self.reference_fixing
        ref_text = "" if ref is None else f", reference fixing {ref:g}"
        return (
            f"; seasoned: {self.realised_count} of {self.n_returns} returns realised "
            f"(sum of squares {self.realised_sum_sq:.6g}{ref_text}), inception {self.start:g}y"
        )


def realised_state(
    reference_fixing: float | None, realised_sum_sq: float, realised_count: int
) -> tuple[float | None, float, int]:
    """Validated realised-state inputs of a seasoned realised-variance product: a positive
    reference close, a non-negative sum of squared returns and count, and no realised return
    without the reference it ends on."""
    if reference_fixing is not None and not (
        np.isfinite(reference_fixing) and reference_fixing > 0
    ):
        raise ValueError("reference_fixing must be a positive close")
    if not (np.isfinite(realised_sum_sq) and realised_sum_sq >= 0):
        raise ValueError("realised_sum_sq must be finite and non-negative")
    if int(realised_count) != realised_count or realised_count < 0:
        raise ValueError("realised_count must be a non-negative integer")
    if reference_fixing is None and (realised_count > 0 or realised_sum_sq != 0.0):
        raise ValueError("realised returns need the reference fixing they end on")
    ref = None if reference_fixing is None else float(reference_fixing)
    return ref, float(realised_sum_sq), int(realised_count)


def with_reference(log_spots: FloatArray, reference_fixing: float | None) -> FloatArray:
    """``ln S`` at the remaining fixings, ``(n_paths, m)``, with the realised reference close
    prepended as a constant column (unchanged when there is none)."""
    if reference_fixing is None:
        return log_spots
    ref = np.full((log_spots.shape[0], 1), np.log(reference_fixing))
    return np.asarray(np.concatenate([ref, log_spots], axis=1), dtype=np.float64)


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
        *,
        reference_fixing: float | None = None,
        realised_sum_sq: float = 0.0,
        realised_count: int = 0,
        inception: float | None = None,
        seasoned: bool = False,
    ) -> None:
        super().__init__(
            fixing_times,
            discount,
            notional,
            annualisation,
            use_simulation_grid,
            reference_fixing=reference_fixing,
            realised_sum_sq=realised_sum_sq,
            realised_count=realised_count,
            inception=inception,
            seasoned=seasoned,
        )
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
            **self.state_kwargs(),
        )

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        rv = self.realised_variance(paths, idx)
        return np.asarray(
            self.notional * float(self.df(self.maturity)) * (rv - self.strike), dtype=np.float64
        )

    def aged(self, dt: float) -> Product:
        state = self.state_kwargs()
        if self.inception is not None:
            state["inception"] = self.inception - dt
        return VarianceSwap(
            shift_times(self._fixings, dt),
            self.strike,
            self.discount,
            self.notional,
            self.annualisation,
            self.use_simulation_grid,
            **state,
        )

    def __repr__(self) -> str:
        kind = "Forward variance swap" if self.start > 0 else "Variance swap"
        return (
            f"{kind}: [{self.start:g}y, {self.maturity:g}y], {self.n_returns} returns, "
            f"strike {np.sqrt(self.strike) * 100:.4g}% vol ({self.strike:.6g} var), "
            f"variance notional {self.notional:g}{self._seasoned_repr()}"
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

    def aged(self, dt: float) -> Product:
        return VolSwap(
            shift_times(self._fixings, dt),
            self.strike_vol,
            self.discount,
            self.notional,
            self.annualisation,
            self.use_simulation_grid,
        )

    def __repr__(self) -> str:
        kind = "Forward vol swap" if self.start > 0 else "Vol swap"
        return (
            f"{kind}: [{self.start:g}y, {self.maturity:g}y], {self.n_returns} returns, "
            f"strike {self.strike_vol * 100:.4g}% vol, vol notional {self.notional:g}"
        )


class FVA(Product):
    """Forward volatility agreement on the relative performance ``R = S_T2/S_T1``.

    Pays at ``T2``::

        notional · ( |R − m| − Straddle_Black(m, m, τ, K_vol) ),   τ = T2 − T1,

    i.e. the holder receives the forward-start straddle struck at the ratio ``m`` and pays its
    Black premium at the agreed volatility ``K_vol`` (settled at ``T2`` together with the
    straddle, so ``DF(T2)`` is the only financing term).  ``m`` defaults to the model's forward
    ratio ``F(T2)/F(T1) = E[R]`` (deterministic rates), the ATM-forward straddle, so the fair
    ``K_vol`` is exactly the forward ATM-forward implied volatility of
    :func:`volsto.analytics.forward_smile.forward_atm_vol` (``E[|R − m|] = Straddle_Black(m, m,
    τ, σ̂_ATMF)`` by definition of the forward smile, book §3.1).  Bergomi §3.1.9 treats the FX
    variant ``(S_T2 − k S_T1)⁺`` whose ``S_T1`` prefactor changes the hedge instruments to
    ``S ln S`` contracts (eq. 3.18); footnote 9 there names this relative-performance form.
    Checked by ``tests/test_forward_start.py::test_fva_fair_strike_is_forward_atm_vol``.
    """

    def __init__(
        self,
        t1: float,
        t2: float,
        strike_vol: float,
        discount: DiscountCurve,
        *,
        forward_curve: ForwardCurve | None = None,
        moneyness: float | None = None,
        notional: float = 1.0,
    ) -> None:
        super().__init__(discount, notional)
        if t1 < 0 or t2 <= t1:
            raise ValueError("need 0 ≤ t1 < t2")
        if strike_vol < 0:
            raise ValueError("strike_vol must be non-negative")
        if moneyness is None:
            if forward_curve is None:
                raise ValueError("give the forward curve (ATM-forward straddle) or a moneyness")
            moneyness = float(forward_curve.forward(t2) / forward_curve.forward(t1))
        if moneyness <= 0:
            raise ValueError("moneyness must be positive")
        self.T1 = float(t1)
        self.T2 = float(t2)
        self.strike_vol = float(strike_vol)
        self.moneyness = float(moneyness)

    @property
    def tau(self) -> float:
        return self.T2 - self.T1

    @property
    def straddle_premium(self) -> float:
        """Undiscounted Black straddle value ``Σ_cp Black(m, m, τ, K_vol, cp)`` paid at ``T2``."""
        m, tau, v = self.moneyness, self.tau, self.strike_vol
        return float(black_price(m, m, tau, v, 1) + black_price(m, m, tau, v, -1))

    @property
    def fixing_times(self) -> FloatArray:
        return np.array([self.T1, self.T2])

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        r = np.exp(paths.log_return(idx[self.T1], idx[self.T2]))
        cf = np.abs(r - self.moneyness) - self.straddle_premium
        return np.asarray(self.notional * float(self.df(self.T2)) * cf, dtype=np.float64)

    def decompose(self) -> list[Product]:
        return [
            ForwardStartStraddle(self.T1, self.T2, self.moneyness, self.discount, self.notional),
            CashFlow(-self.straddle_premium, self.T2, self.discount, self.notional),
        ]

    def aged(self, dt: float) -> Product:
        t = shift_times([self.T1, self.T2], dt)
        return FVA(
            float(t[0]),
            float(t[1]),
            self.strike_vol,
            self.discount,
            moneyness=self.moneyness,
            notional=self.notional,
        )

    def __repr__(self) -> str:
        return (
            f"FVA: forward-start straddle |S_T2/S_T1 - {self.moneyness:.6g}| less its Black "
            f"premium at {self.strike_vol * 100:.4g}% vol, T1 {self.T1:g}y, T2 {self.T2:g}y, "
            f"notional {self.notional:g}"
        )
