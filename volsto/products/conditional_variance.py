"""Conditional, corridor and knock-out variance swaps (SPEC §15 / v2 §6.1).

Notation: daily closes ``S_0..S_N`` on the fixing schedule, ``r_i = ln(S_i/S_{i-1})``,
annualisation ``A`` (252 by default), barrier ``B``, strike ``K`` quoted as a volatility, variance
notional ``N_var``.  Realised quantities are computed on the fixing dates (SPEC §6), never on the
simulation grid.

* :class:`ConditionalVarianceSwap` accrues ``r_i²`` only where the spot is in the region
  (``side = "up"``: ``S > B``, ``"down"``: ``S < B``; ``strict`` selects ``>`` / ``<`` versus
  ``>=`` / ``<=``) as decided by the ``indicator``: ``"prev"`` looks at ``S_{i-1}``, ``"curr"`` at
  ``S_i``,
  ``"both"`` at both.  ``D = Σ_i I_i``.  Convention ``"conditional"`` pays
  ``N_var [ (A/N) Σ r_i² I_i − K² D/N ]`` (zero when ``D = 0``), ``"corridor"`` pays
  ``N_var [ (A/N) Σ r_i² I_i − K² ]``.  ``daily_cap c`` replaces ``r_i²`` by ``min(r_i², c²)``.
  The indicator is a required argument; the desk helpers :func:`UpVar` (``"prev"`` or
  ``"both"``) and :func:`DownVar` (``"curr"`` or ``"both"``) supply the conventional one.
* :class:`ConvexitySpread`: long the conditional product, short the plain variance swap on the
  same schedule; ``decompose()`` returns the two legs.
* :class:`KnockOutVarianceSwap`: close-to-close monitoring, ``j = min{i : S_i > B}`` (``S_i <
  B`` for ``direction = "down"``), ``τ = min(j, N)``; variant (b): ``N_var [ (A/N) Σ_{i≤τ} r_i² −
  K² τ/N ]`` where the knock-out day's own return accrues.  ``settlement`` (default
  ``"maturity"``: paid at the last fixing, discounted from there) or ``"knock_out"``: paid at the
  knock-out close ``t_τ`` and discounted from it — the desk's convention (owner, 2026-09-27); an
  unknocked swap pays at maturity either way.  Continuous monitoring and variants (a)/(c) raise
  ``NotImplementedError``.

Seasoned products (M10 Part 3, SPEC §6.10; :func:`volsto.products.seasoning.season`): the
schedule carries ``reference_fixing`` (the last realised close, prepended to the path's fixings
as a constant column), ``realised_sum_sq`` (``Σ r_i²``, capped, over the realised returns) and
``realised_count``; ``N`` counts every return of the life.  The knock-out swap's ``τ`` is then
``realised_count`` plus the stopping index of the remaining returns (a realised knock-out is a
settled trade, never a live seasoned one).  The defaults are the fresh products bit for bit.

Fair strikes are ratios of expectations (``K² = A E[Σ r_i² I_i] / E[D]`` for the conditional
convention, ``A E[Σ_{i≤τ} r_i²] / E[τ]`` for the knock-out swap, ``(A/N) E[Σ r_i² I_i]`` for the
corridor); :func:`volsto.analytics.conditional_variance.fair_strike` estimates them with a
delta-method standard error from the per-path statistics exposed by :meth:`statistics` /
:meth:`leg`.  Path-wise identities (complementarity of up and down swaps, ``B → ∞`` limits,
decompositions) are checked by ``tests/test_conditional_variance.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.products.base import Product, shift_times
from volsto.products.variance import VarianceSwap, realised_state, with_reference

if TYPE_CHECKING:
    from volsto.config import SimConfig
    from volsto.engine.grid import FixingIndex
    from volsto.engine.paths import PathSet
    from volsto.market.curves import DiscountCurve
    from volsto.models.base import Model

FloatArray = NDArray[np.float64]

SIDES = ("up", "down")
#: when a knock-out variance swap pays (:class:`KnockOutVarianceSwap`)
SETTLEMENTS = ("maturity", "knock_out")
INDICATORS = ("prev", "curr", "both")
CONVENTIONS = ("conditional", "corridor")


class RealisedVarianceSchedule(Product):
    """Fixing-schedule handling shared by the products of this module and the VKO put."""

    def __init__(
        self,
        fixing_times: ArrayLike,
        discount: DiscountCurve,
        notional: float,
        annualisation: float,
        daily_cap: float | None,
        *,
        reference_fixing: float | None = None,
        realised_sum_sq: float = 0.0,
        realised_count: int = 0,
        seasoned: bool = False,
    ) -> None:
        super().__init__(discount, notional)
        ft = np.unique(np.asarray(fixing_times, dtype=np.float64))
        if reference_fixing is None:
            if ft.size < 2 or ft[0] < 0 or ft[-1] <= 0:
                raise ValueError("need at least two non-negative fixing times ending after 0")
        elif ft.size < 1 or ft[0] <= 0:
            raise ValueError("a seasoned product needs its remaining fixings after the time origin")
        if annualisation <= 0:
            raise ValueError("annualisation must be positive")
        if daily_cap is not None and daily_cap <= 0:
            raise ValueError("daily_cap must be positive")
        self._fixings = ft
        self.annualisation = float(annualisation)
        self.daily_cap = None if daily_cap is None else float(daily_cap)
        self.reference_fixing, self.realised_sum_sq, self.realised_count = realised_state(
            reference_fixing, realised_sum_sq, realised_count
        )
        self.seasoned = bool(seasoned)

    @property
    def fixing_times(self) -> FloatArray:
        return self._fixings

    @property
    def is_seasoned(self) -> bool:
        """Whether the product was seasoned (``seasoned``, set by
        :func:`volsto.products.seasoning.season`, or a realised state; the defaults are the
        fresh product)."""
        return self.seasoned or self.reference_fixing is not None

    @property
    def n_returns(self) -> int:
        """``N``: returns over the life, the realised ones included."""
        future = self._fixings.size - (1 if self.reference_fixing is None else 0)
        return int(self.realised_count + future)

    def state_kwargs(self) -> dict[str, Any]:
        """The realised-state constructor arguments."""
        return {
            "reference_fixing": self.reference_fixing,
            "realised_sum_sq": self.realised_sum_sq,
            "realised_count": self.realised_count,
            "seasoned": self.seasoned,
        }

    def _seasoned_repr(self) -> str:
        if self.reference_fixing is None:
            return ""
        return (
            f"; seasoned: {self.realised_count} of {self.n_returns} returns realised (sum of "
            f"squares {self.realised_sum_sq:.6g}, reference fixing {self.reference_fixing:g})"
        )

    def log_spots(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        """``ln S`` at the fixings, ``(n_paths, N + 1)``; for a seasoned product the realised
        reference close followed by the remaining fixings, ``(n_paths, m + 1)``."""
        return with_reference(paths.log_spot_at(idx.indices(self._fixings)), self.reference_fixing)

    def squared_returns(self, ls: FloatArray) -> FloatArray:
        """``r_i²`` (capped at ``daily_cap²`` when set), ``(n_paths, N)``."""
        r2 = np.diff(ls, axis=1) ** 2
        if self.daily_cap is not None:
            r2 = np.minimum(r2, self.daily_cap**2)
        return np.asarray(r2, dtype=np.float64)

    def statistics(self, paths: PathSet, idx: FixingIndex) -> dict[str, FloatArray]:
        """Per-path undiscounted statistics (``accrued`` is already annualised by ``A/N``)."""
        raise NotImplementedError

    def leg(self, name: str) -> Product:
        """A product whose (undiscounted) payoff is the statistic ``name``."""
        return StatisticLeg(self, name)

    def fair_strike(self, model: Model, sim: SimConfig) -> Any:
        """Model fair strike with delta-method standard error (see the analytics module)."""
        from volsto.analytics.conditional_variance import fair_strike

        return fair_strike(self, model, sim)


class StatisticLeg(Product):
    """Undiscounted per-path statistic of a schedule product, priced like a product."""

    def __init__(self, parent: RealisedVarianceSchedule, name: str) -> None:
        super().__init__(parent.discount, 1.0)
        self.parent = parent
        self.name = name

    @property
    def fixing_times(self) -> FloatArray:
        return self.parent.fixing_times

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        return self.parent.statistics(paths, idx)[self.name]

    def aged(self, dt: float) -> Product:
        parent = self.parent.aged(dt)
        assert isinstance(parent, RealisedVarianceSchedule)
        return StatisticLeg(parent, self.name)

    def with_discount(self, discount: DiscountCurve) -> Product:
        parent = self.parent.with_discount(discount)
        assert isinstance(parent, RealisedVarianceSchedule)
        return StatisticLeg(parent, self.name)

    def __repr__(self) -> str:
        return f"Statistic '{self.name}' of [{self.parent!r}]"


def _in_region(ls: FloatArray, ln_b: float, side: str, strict: bool) -> NDArray[np.bool_]:
    if side == "up":
        return (ls > ln_b) if strict else (ls >= ln_b)
    return (ls < ln_b) if strict else (ls <= ln_b)


class ConditionalVarianceSwap(RealisedVarianceSchedule):
    """Conditional / corridor variance swap; see the module docstring for the payoff."""

    def __init__(
        self,
        fixing_times: ArrayLike,
        barrier: float,
        side: str,
        indicator: str,
        convention: str,
        strike_vol: float,
        discount: DiscountCurve,
        *,
        strict: bool = True,
        daily_cap: float | None = None,
        annualisation: float = 252.0,
        notional: float = 1.0,
    ) -> None:
        super().__init__(fixing_times, discount, notional, annualisation, daily_cap)
        if barrier <= 0:
            raise ValueError("barrier must be positive")
        if side not in SIDES or indicator not in INDICATORS or convention not in CONVENTIONS:
            raise ValueError(
                f"side in {SIDES}, indicator in {INDICATORS}, convention in {CONVENTIONS}"
            )
        if strike_vol < 0:
            raise ValueError("strike_vol must be non-negative")
        self.barrier = float(barrier)
        self.side = side
        self.indicator = indicator
        self.convention = convention
        self.strike_vol = float(strike_vol)
        self.strict = bool(strict)

    def indicators(self, ls: FloatArray) -> FloatArray:
        """``I_i`` for the ``N`` returns from the log spots ``(n_paths, N + 1)``."""
        inside = _in_region(ls, np.log(self.barrier), self.side, self.strict)
        if self.indicator == "prev":
            ind = inside[:, :-1]
        elif self.indicator == "curr":
            ind = inside[:, 1:]
        else:
            ind = inside[:, :-1] & inside[:, 1:]
        return ind.astype(np.float64)

    def statistics(self, paths: PathSet, idx: FixingIndex) -> dict[str, FloatArray]:
        ls = self.log_spots(paths, idx)
        ind = self.indicators(ls)
        n = self.n_returns
        accrued = self.annualisation / n * np.sum(self.squared_returns(ls) * ind, axis=1)
        return {"accrued": accrued, "count": np.sum(ind, axis=1) / n}

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        st = self.statistics(paths, idx)
        k2 = self.strike_vol**2
        if self.convention == "conditional":
            cf = st["accrued"] - k2 * st["count"]  # zero automatically when D = 0
        else:
            cf = st["accrued"] - k2
        return np.asarray(self.notional * float(self.df(self.maturity)) * cf, dtype=np.float64)

    def aged(self, dt: float) -> Product:
        return ConditionalVarianceSwap(
            shift_times(self._fixings, dt),
            self.barrier,
            self.side,
            self.indicator,
            self.convention,
            self.strike_vol,
            self.discount,
            strict=self.strict,
            daily_cap=self.daily_cap,
            annualisation=self.annualisation,
            notional=self.notional,
        )

    def __repr__(self) -> str:
        op = {"up": ">" if self.strict else ">=", "down": "<" if self.strict else "<="}[self.side]
        cap = "" if self.daily_cap is None else f", daily cap {self.daily_cap:g}"
        return (
            f"{self.convention.capitalize()} {self.side}-variance swap: S {op} {self.barrier:g} "
            f"({self.indicator}), {self.n_returns} returns to {self.maturity:g}y, strike "
            f"{self.strike_vol * 100:.4g}% vol, A = {self.annualisation:g}{cap}, "
            f"variance notional {self.notional:g}"
        )


def UpVar(
    fixing_times: ArrayLike,
    barrier: float,
    strike_vol: float,
    discount: DiscountCurve,
    *,
    indicator: str = "prev",
    convention: str = "conditional",
    **kwargs: Any,
) -> ConditionalVarianceSwap:
    """Desk up-variance swap: accrues where ``S > B`` on the previous close (or both closes)."""
    if indicator not in ("prev", "both"):
        raise ValueError("up-var uses the 'prev' or 'both' indicator")
    return ConditionalVarianceSwap(
        fixing_times, barrier, "up", indicator, convention, strike_vol, discount, **kwargs
    )


def DownVar(
    fixing_times: ArrayLike,
    barrier: float,
    strike_vol: float,
    discount: DiscountCurve,
    *,
    indicator: str = "curr",
    convention: str = "conditional",
    **kwargs: Any,
) -> ConditionalVarianceSwap:
    """Desk down-variance swap: accrues where ``S < B`` on the current close (or both closes)."""
    if indicator not in ("curr", "both"):
        raise ValueError("down-var uses the 'curr' or 'both' indicator")
    return ConditionalVarianceSwap(
        fixing_times, barrier, "down", indicator, convention, strike_vol, discount, **kwargs
    )


class ConvexitySpread(Product):
    """Long a conditional variance swap, short ``notional_ratio`` plain variance swaps on the same
    fixing schedule; the study quantity is the strike differential ``K_up² − K_var²``."""

    def __init__(
        self, upvar: ConditionalVarianceSwap, varswap: VarianceSwap, notional_ratio: float = 1.0
    ) -> None:
        super().__init__(upvar.discount, 1.0)
        if not np.allclose(upvar.fixing_times, varswap.fixing_times, rtol=0, atol=1e-9):
            raise ValueError("both legs must share the fixing schedule")
        if varswap.use_simulation_grid:
            raise ValueError("the variance-swap leg must be realised on the fixing dates")
        self.upvar = upvar
        self.varswap = varswap
        self.notional_ratio = float(notional_ratio)

    @property
    def fixing_times(self) -> FloatArray:
        return self.upvar.fixing_times

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        return np.asarray(
            self.upvar.payoff(paths, idx) - self.notional_ratio * self.varswap.payoff(paths, idx),
            dtype=np.float64,
        )

    def decompose(self) -> list[Product]:
        short = VarianceSwap(
            self.varswap.fixing_times,
            self.varswap.strike,
            self.varswap.discount,
            -self.notional_ratio * self.varswap.notional,
            self.varswap.annualisation,
            self.varswap.use_simulation_grid,
        )
        return [self.upvar, short]

    def aged(self, dt: float) -> Product:
        up = self.upvar.aged(dt)
        vs = self.varswap.aged(dt)
        assert isinstance(up, ConditionalVarianceSwap) and isinstance(vs, VarianceSwap)
        return ConvexitySpread(up, vs, self.notional_ratio)

    def with_discount(self, discount: DiscountCurve) -> Product:
        up = self.upvar.with_discount(discount)
        vs = self.varswap.with_discount(discount)
        assert isinstance(up, ConditionalVarianceSwap) and isinstance(vs, VarianceSwap)
        return ConvexitySpread(up, vs, self.notional_ratio)

    def __repr__(self) -> str:
        return (
            f"Convexity spread: long [{self.upvar!r}] short {self.notional_ratio:g} x "
            f"[{self.varswap!r}]"
        )


class KnockOutVarianceSwap(RealisedVarianceSchedule):
    """Knock-out variance swap, close-to-close monitoring, variant (b), paid at maturity or at the
    knock-out close (``settlement``; module docstring)."""

    def __init__(
        self,
        fixing_times: ArrayLike,
        barrier: float,
        strike_vol: float,
        discount: DiscountCurve,
        *,
        direction: str = "up",
        strict: bool = True,
        monitoring: str = "close",
        variant: str = "b",
        settlement: str = "maturity",
        daily_cap: float | None = None,
        annualisation: float = 252.0,
        notional: float = 1.0,
        reference_fixing: float | None = None,
        realised_sum_sq: float = 0.0,
        realised_count: int = 0,
        seasoned: bool = False,
    ) -> None:
        super().__init__(
            fixing_times,
            discount,
            notional,
            annualisation,
            daily_cap,
            reference_fixing=reference_fixing,
            realised_sum_sq=realised_sum_sq,
            realised_count=realised_count,
            seasoned=seasoned,
        )
        if barrier <= 0 or strike_vol < 0:
            raise ValueError("barrier must be positive and strike_vol non-negative")
        if settlement not in SETTLEMENTS:
            raise ValueError(f"settlement must be one of {SETTLEMENTS}")
        if direction not in SIDES:
            raise ValueError(f"direction in {SIDES}")
        if monitoring != "close":
            raise NotImplementedError("only close-to-close monitoring is implemented (SPEC §15)")
        if variant != "b":
            raise NotImplementedError("only variant (b) is implemented (SPEC §15)")
        self.barrier = float(barrier)
        self.strike_vol = float(strike_vol)
        self.direction = direction
        self.strict = bool(strict)
        self.monitoring = monitoring
        self.variant = variant
        self.settlement = settlement
        if self.reference_fixing is not None and bool(
            self.beyond_barrier(np.array([np.log(self.reference_fixing)]))[0]
        ):
            raise ValueError(
                "the reference fixing lies beyond the barrier: the swap has knocked out "
                "(a settled trade, volsto.products.seasoning.season)"
            )

    def beyond_barrier(self, ls: FloatArray) -> NDArray[np.bool_]:
        """Whether each log close lies in the knock-out region (the monitoring comparison)."""
        return _in_region(ls, np.log(self.barrier), self.direction, self.strict)

    def stopping_index(self, ls: FloatArray) -> NDArray[np.int64]:
        """``τ = min(j, m)`` with ``j`` the first close of ``ls`` (including its first column,
        ``S_0`` or the realised reference) beyond the barrier and ``m = ls.shape[1] − 1`` the
        returns it spans."""
        hit = self.beyond_barrier(ls)
        n = int(ls.shape[1]) - 1
        any_hit = hit.any(axis=1)
        first = np.argmax(hit, axis=1)
        return np.where(any_hit, np.minimum(first, n), n).astype(np.int64)

    def statistics(self, paths: PathSet, idx: FixingIndex) -> dict[str, FloatArray]:
        ls = self.log_spots(paths, idx)
        n = self.n_returns
        m = ls.shape[1] - 1  # the remaining returns (all of them when fresh)
        tau = self.stopping_index(ls)
        r2 = self.squared_returns(ls)
        alive = (np.arange(1, m + 1)[None, :] <= tau[:, None]).astype(np.float64)
        accrued = self.annualisation / n * (self.realised_sum_sq + np.sum(r2 * alive, axis=1))
        tau_life = self.realised_count + tau
        return {
            "accrued": accrued,
            "count": tau_life / n,
            "ko": (tau < m).astype(np.float64),
            "tau": tau_life.astype(np.float64),
        }

    def pay_time(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        """Per path, when the swap pays: the maturity, or the knock-out close ``t_τ`` under
        ``settlement = "knock_out"`` (the maturity when it never knocks)."""
        ls = self.log_spots(paths, idx)
        if self.settlement == "maturity":
            return np.full(ls.shape[0], float(self.maturity))
        cols = self._fixings if self.reference_fixing is None else np.r_[0.0, self._fixings]
        return np.asarray(cols[self.stopping_index(ls)], dtype=np.float64)

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        st = self.statistics(paths, idx)
        cf = st["accrued"] - self.strike_vol**2 * st["count"]
        if self.settlement == "maturity":
            df = np.full(cf.shape, float(self.df(self.maturity)))
        else:
            df = np.asarray(self.df(self.pay_time(paths, idx)), dtype=np.float64)
        return np.asarray(self.notional * df * cf, dtype=np.float64)

    def aged(self, dt: float) -> Product:
        return KnockOutVarianceSwap(
            shift_times(self._fixings, dt),
            self.barrier,
            self.strike_vol,
            self.discount,
            direction=self.direction,
            strict=self.strict,
            settlement=self.settlement,
            daily_cap=self.daily_cap,
            annualisation=self.annualisation,
            notional=self.notional,
            **self.state_kwargs(),
        )

    def __repr__(self) -> str:
        op = {"up": ">" if self.strict else ">=", "down": "<" if self.strict else "<="}[
            self.direction
        ]
        return (
            f"Knock-out variance swap (close-to-close, variant b): out when S {op} "
            f"{self.barrier:g}, {self.n_returns} returns to {self.maturity:g}y, strike "
            f"{self.strike_vol * 100:.4g}% vol, A = {self.annualisation:g}, "
            f"variance notional {self.notional:g}"
            + (", settled at the knock-out" if self.settlement == "knock_out" else "")
            + self._seasoned_repr()
        )
