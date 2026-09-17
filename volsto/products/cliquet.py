"""Cliquet family (SPEC §6): additive cliquet with local and global caps/floors, reverse cliquet,
Napoleon.

All payoffs are functions of the period returns ``r_i = S_{t_i}/S_{t_{i-1}} − 1`` over consecutive
fixings (book §3.1: "cliquets ... whose prices are purely determined by the distribution of
forward returns"), paid at the last fixing ``T``.  Clipping identities give exact decompositions
into forward-start options settled at ``T``::

    clip(x, LF, LC) = LF + (x − LF)⁺ − (x − LC)⁺            (LF finite)
                    = x − (x − LC)⁺ = (R − 0)⁺ − 1 − (R − (1 + LC))⁺     (no local floor)
    clip(Σ, GF, GC) = Σ + (GF − Σ)⁺ − (Σ − GC)⁺

so an additive cliquet is cash + a strip of long forward-start calls struck ``1 + LF`` (or the
``k = 0`` call, i.e. the return itself) − short forward-start calls struck ``1 + LC`` + a put on
the accumulated sum struck ``GF`` − a call on it struck ``GC`` (:class:`AccumulatedSumOption`).
The identity holds path by path (``tests/test_cliquet.py``), which is what
``decompose()`` is used for in the gamma-profile and hedging analyses (SPEC §7–§8).

The original study's structure is :meth:`AdditiveCliquet.study`: monthly fixings, local cap 2%,
no local floor, global floor 0, maturities 1y and 2y.

Seasoned cliquets (M10 Part 3, SPEC §6.10; :func:`volsto.products.seasoning.season`): the state
inputs ``reference_fixing`` (the realised close that starts the running period: its return is
``S_{t_1}/S_ref − 1``, a spot bump never moves it) and ``accrued`` (``Σ clip(r_i, LF, LC)`` over
the realised periods) give ``clip(accrued + Σ_future clip(r_i, LF, LC), GF, GC)``; the defaults
(``None``, 0) are the fresh cliquet bit for bit (``tests/test_seasoning.py``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.products.base import CashFlow, Product, parse_cp, shift_times, uniform_schedule
from volsto.products.forward_start import ForwardStartOption
from volsto.products.variance import with_reference

if TYPE_CHECKING:
    from volsto.engine.grid import FixingIndex
    from volsto.engine.paths import PathSet
    from volsto.market.curves import DiscountCurve

FloatArray = NDArray[np.float64]


def _bound(x: float | None, default: float) -> float:
    if x is None:
        return default
    if not np.isfinite(x):
        raise ValueError("caps and floors must be finite or None")
    return float(x)


class CliquetBounds(TypedDict):
    """The local and global bounds of an :class:`AdditiveCliquet` as constructor arguments."""

    local_floor: float | None
    local_cap: float | None
    global_floor: float | None
    global_cap: float | None


def _fmt(x: float) -> str:
    return "none" if not np.isfinite(x) else f"{x * 100:g}%"


class _PeriodReturnProduct(Product):
    """Common schedule handling: ``n`` period returns over ``n + 1`` fixings (first may be 0)."""

    def __init__(
        self,
        fixing_times: ArrayLike,
        discount: DiscountCurve,
        notional: float,
        *,
        reference_fixing: float | None = None,
    ) -> None:
        super().__init__(discount, notional)
        ft = np.unique(np.asarray(fixing_times, dtype=np.float64))
        if reference_fixing is None:
            if ft.size < 2 or ft[0] < 0 or ft[-1] <= 0:
                raise ValueError("need at least two non-negative fixing times ending after 0")
        else:
            if ft.size < 1 or ft[0] <= 0:
                raise ValueError("a seasoned cliquet needs its remaining fixings after the origin")
            if not (np.isfinite(reference_fixing) and reference_fixing > 0):
                raise ValueError("reference_fixing must be a positive close")
        self._fixings = ft
        self.reference_fixing = None if reference_fixing is None else float(reference_fixing)

    @property
    def fixing_times(self) -> FloatArray:
        return self._fixings

    @property
    def n_periods(self) -> int:
        """The periods still to fix (all of them when fresh)."""
        return int(self._fixings.size - (1 if self.reference_fixing is None else 0))

    def period_returns(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        """``(n_paths, n_periods)`` simple returns ``S_{t_i}/S_{t_{i-1}} − 1`` (the first from
        the realised reference close when seasoned)."""
        ls = paths.log_spot_at(idx.indices(self._fixings))
        ls = with_reference(ls, self.reference_fixing)
        return np.asarray(np.exp(np.diff(ls, axis=1)) - 1.0, dtype=np.float64)


class AdditiveCliquet(_PeriodReturnProduct):
    """``notional · clip( Σ_i clip(r_i, LF, LC), GF, GC )`` paid at the last fixing.

    ``None`` means no bound.  ``local_floor ≤ local_cap`` and ``global_floor ≤ global_cap`` are
    enforced.  See the module docstring for the decomposition and its test.
    """

    def __init__(
        self,
        fixing_times: ArrayLike,
        discount: DiscountCurve,
        *,
        local_floor: float | None = None,
        local_cap: float | None = None,
        global_floor: float | None = None,
        global_cap: float | None = None,
        notional: float = 1.0,
        reference_fixing: float | None = None,
        accrued: float = 0.0,
        seasoned: bool = False,
    ) -> None:
        super().__init__(fixing_times, discount, notional, reference_fixing=reference_fixing)
        self.local_floor = _bound(local_floor, -np.inf)
        self.local_cap = _bound(local_cap, np.inf)
        self.global_floor = _bound(global_floor, -np.inf)
        self.global_cap = _bound(global_cap, np.inf)
        if self.local_floor > self.local_cap or self.global_floor > self.global_cap:
            raise ValueError("floors must not exceed caps")
        if not np.isfinite(accrued):
            raise ValueError("accrued must be finite")
        if accrued != 0.0 and reference_fixing is None:
            raise ValueError(
                "realised periods need the reference fixing the running period starts on"
            )
        self.accrued = float(accrued)
        self.seasoned = bool(seasoned)

    @classmethod
    def study(
        cls,
        maturity: float,
        discount: DiscountCurve,
        *,
        periods_per_year: int = 12,
        local_cap: float | None = 0.02,
        local_floor: float | None = None,
        global_floor: float | None = 0.0,
        global_cap: float | None = None,
        notional: float = 1.0,
    ) -> AdditiveCliquet:
        """The original study's cliquet: monthly, local cap 2%, no local floor, global floor 0."""
        n = round(maturity * periods_per_year)
        if abs(n - maturity * periods_per_year) > 1e-9 or n < 1:
            raise ValueError("maturity must be a whole number of periods")
        return cls(
            uniform_schedule(maturity, n),
            discount,
            local_floor=local_floor,
            local_cap=local_cap,
            global_floor=global_floor,
            global_cap=global_cap,
            notional=notional,
        )

    @property
    def has_global(self) -> bool:
        return bool(np.isfinite(self.global_floor) or np.isfinite(self.global_cap))

    @property
    def is_seasoned(self) -> bool:
        """Whether the cliquet was seasoned (``seasoned``, set by
        :func:`volsto.products.seasoning.season`, or a realised state; the defaults are the
        fresh cliquet)."""
        return self.seasoned or self.reference_fixing is not None

    def terms(self, *, with_global: bool = True) -> CliquetBounds:
        """The bounds as constructor arguments (``None`` for an absent bound; the global ones
        ``None`` too without ``with_global``)."""

        def b(x: float) -> float | None:
            return None if not np.isfinite(x) else x

        return CliquetBounds(
            local_floor=b(self.local_floor),
            local_cap=b(self.local_cap),
            global_floor=b(self.global_floor) if with_global else None,
            global_cap=b(self.global_cap) if with_global else None,
        )

    def without_global(self) -> AdditiveCliquet:
        """Same local structure without the global floor/cap (the accumulated sum itself)."""
        return AdditiveCliquet(
            self._fixings,
            self.discount,
            **self.terms(with_global=False),
            notional=self.notional,
            reference_fixing=self.reference_fixing,
            accrued=self.accrued,
            seasoned=self.seasoned,
        )

    def local_returns(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        return np.clip(self.period_returns(paths, idx), self.local_floor, self.local_cap)

    def accumulated(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        """``Σ_i clip(r_i, LF, LC)`` per path (before the global floor/cap), the realised
        ``accrued`` included."""
        return np.asarray(
            self.accrued + np.sum(self.local_returns(paths, idx), axis=1), dtype=np.float64
        )

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        total = np.clip(self.accumulated(paths, idx), self.global_floor, self.global_cap)
        return np.asarray(self.notional * float(self.df(self.maturity)) * total, dtype=np.float64)

    def decompose(self) -> list[Product]:
        if self.is_seasoned:
            raise NotImplementedError(
                "decompose() of a seasoned cliquet (its running period starts on a realised close)"
            )
        T = self.maturity
        parts: list[Product] = []
        cash = 0.0
        for t1, t2 in zip(self._fixings[:-1], self._fixings[1:], strict=True):
            if np.isfinite(self.local_floor):
                cash += self.local_floor
                parts.append(
                    ForwardStartOption(
                        t1, t2, 1.0 + self.local_floor, 1, self.discount, self.notional, T
                    )
                )
            else:  # r = (R − 0)⁺ − 1
                cash -= 1.0
                parts.append(ForwardStartOption(t1, t2, 0.0, 1, self.discount, self.notional, T))
            if np.isfinite(self.local_cap):
                parts.append(
                    ForwardStartOption(
                        t1, t2, 1.0 + self.local_cap, 1, self.discount, -self.notional, T
                    )
                )
        if cash != 0.0:
            parts.insert(0, CashFlow(cash, T, self.discount, self.notional))
        if np.isfinite(self.global_floor):
            parts.append(AccumulatedSumOption(self, self.global_floor, -1, self.notional))
        if np.isfinite(self.global_cap):
            parts.append(AccumulatedSumOption(self, self.global_cap, 1, -self.notional))
        return parts

    def aged(self, dt: float) -> Product:
        return AdditiveCliquet(
            shift_times(self._fixings, dt),
            self.discount,
            **self.terms(),
            notional=self.notional,
            reference_fixing=self.reference_fixing,
            accrued=self.accrued,
            seasoned=self.seasoned,
        )

    def __repr__(self) -> str:
        seasoned = (
            ""
            if self.reference_fixing is None
            else f"; seasoned: running period from the realised close {self.reference_fixing:g}, "
            f"accrued capped returns {self.accrued:.6g}"
        )
        return (
            f"Additive cliquet: {self.n_periods} periods to {self.maturity:g}y, local floor "
            f"{_fmt(self.local_floor)}, local cap {_fmt(self.local_cap)}, global floor "
            f"{_fmt(self.global_floor)}, global cap {_fmt(self.global_cap)}, "
            f"notional {self.notional:g}{seasoned}"
        )


class AccumulatedSumOption(Product):
    """``notional · (cp (Σ − K))⁺`` on a cliquet's accumulated local sum ``Σ``, paid at its
    maturity: the global-floor put (``cp = −1``) and global-cap call (``cp = +1``) legs."""

    def __init__(
        self, cliquet: AdditiveCliquet, strike: float, cp: int | str, notional: float = 1.0
    ) -> None:
        super().__init__(cliquet.discount, notional)
        self.cliquet = cliquet.without_global()
        self.strike = float(strike)
        self.cp = parse_cp(cp)

    @property
    def fixing_times(self) -> FloatArray:
        return self.cliquet.fixing_times

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        total = self.cliquet.accumulated(paths, idx)
        cf = np.maximum(self.cp * (total - self.strike), 0.0)
        return np.asarray(
            self.notional * float(self.df(self.cliquet.maturity)) * cf, dtype=np.float64
        )

    def aged(self, dt: float) -> Product:
        inner = self.cliquet.aged(dt)
        assert isinstance(inner, AdditiveCliquet)
        return AccumulatedSumOption(inner, self.strike, self.cp, self.notional)

    def with_discount(self, discount: DiscountCurve) -> Product:
        inner = self.cliquet.with_discount(discount)
        assert isinstance(inner, AdditiveCliquet)
        return AccumulatedSumOption(inner, self.strike, self.cp, self.notional)

    def __repr__(self) -> str:
        kind = "call" if self.cp > 0 else "put"
        return (
            f"{kind.capitalize()} struck {self.strike * 100:g}% on the accumulated sum of "
            f"[{self.cliquet!r}], notional {self.notional:g}"
        )


class ReverseCliquet(_PeriodReturnProduct):
    """``notional · max(GF, C + Σ_i clip(r_i, LF, 0))``: a coupon ``C`` eroded by the negative
    period returns (each optionally floored at ``LF``), floored globally at ``GF`` (0 by default).

    Equals ``C + clip(Σ_i clip(r_i, LF, 0), GF − C, ∞)``, so ``decompose()`` is the coupon cash
    flow plus the decomposition of that additive cliquet (short forward-start puts struck 1 when
    there is no local floor: ``min(r, 0) = −(1 − R)⁺``).
    """

    def __init__(
        self,
        fixing_times: ArrayLike,
        coupon: float,
        discount: DiscountCurve,
        *,
        local_floor: float | None = None,
        global_floor: float | None = 0.0,
        notional: float = 1.0,
    ) -> None:
        super().__init__(fixing_times, discount, notional)
        if coupon < 0:
            raise ValueError("coupon must be non-negative")
        self.coupon = float(coupon)
        self.local_floor = _bound(local_floor, -np.inf)
        self.global_floor = _bound(global_floor, -np.inf)
        if self.local_floor > 0:
            raise ValueError("the local floor of a reverse cliquet must be ≤ 0")

    def _inner(self) -> AdditiveCliquet:
        gf = None if not np.isfinite(self.global_floor) else self.global_floor - self.coupon
        lf = None if not np.isfinite(self.local_floor) else self.local_floor
        return AdditiveCliquet(
            self._fixings,
            self.discount,
            local_floor=lf,
            local_cap=0.0,
            global_floor=gf,
            notional=self.notional,
        )

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        r = np.clip(self.period_returns(paths, idx), self.local_floor, 0.0)
        total = np.maximum(self.coupon + np.sum(r, axis=1), self.global_floor)
        return np.asarray(self.notional * float(self.df(self.maturity)) * total, dtype=np.float64)

    def decompose(self) -> list[Product]:
        return [
            CashFlow(self.coupon, self.maturity, self.discount, self.notional),
            *self._inner().decompose(),
        ]

    def aged(self, dt: float) -> Product:
        return ReverseCliquet(
            shift_times(self._fixings, dt),
            self.coupon,
            self.discount,
            local_floor=None if not np.isfinite(self.local_floor) else self.local_floor,
            global_floor=None if not np.isfinite(self.global_floor) else self.global_floor,
            notional=self.notional,
        )

    def __repr__(self) -> str:
        return (
            f"Reverse cliquet: coupon {self.coupon * 100:g}% less negative returns over "
            f"{self.n_periods} periods to {self.maturity:g}y, local floor "
            f"{_fmt(self.local_floor)}, global floor {_fmt(self.global_floor)}, "
            f"notional {self.notional:g}"
        )


class Napoleon(_PeriodReturnProduct):
    """``notional · max(GF, C + min_i clip(r_i, LF, LC))``: a coupon plus the worst period
    return, floored globally at ``GF`` (0 by default).  The minimum is not additive, so there is
    no forward-start decomposition (``decompose()`` returns ``None``)."""

    def __init__(
        self,
        fixing_times: ArrayLike,
        coupon: float,
        discount: DiscountCurve,
        *,
        local_floor: float | None = None,
        local_cap: float | None = None,
        global_floor: float | None = 0.0,
        notional: float = 1.0,
    ) -> None:
        super().__init__(fixing_times, discount, notional)
        if coupon < 0:
            raise ValueError("coupon must be non-negative")
        self.coupon = float(coupon)
        self.local_floor = _bound(local_floor, -np.inf)
        self.local_cap = _bound(local_cap, np.inf)
        self.global_floor = _bound(global_floor, -np.inf)
        if self.local_floor > self.local_cap:
            raise ValueError("floors must not exceed caps")

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        r = np.clip(self.period_returns(paths, idx), self.local_floor, self.local_cap)
        total = np.maximum(self.coupon + np.min(r, axis=1), self.global_floor)
        return np.asarray(self.notional * float(self.df(self.maturity)) * total, dtype=np.float64)

    def aged(self, dt: float) -> Product:
        return Napoleon(
            shift_times(self._fixings, dt),
            self.coupon,
            self.discount,
            local_floor=None if not np.isfinite(self.local_floor) else self.local_floor,
            local_cap=None if not np.isfinite(self.local_cap) else self.local_cap,
            global_floor=None if not np.isfinite(self.global_floor) else self.global_floor,
            notional=self.notional,
        )

    def __repr__(self) -> str:
        return (
            f"Napoleon: coupon {self.coupon * 100:g}% plus the worst of {self.n_periods} period "
            f"returns to {self.maturity:g}y (local floor {_fmt(self.local_floor)}, local cap "
            f"{_fmt(self.local_cap)}), global floor {_fmt(self.global_floor)}, "
            f"notional {self.notional:g}"
        )
