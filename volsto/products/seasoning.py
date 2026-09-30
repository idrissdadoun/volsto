"""Seasoned trades: a product struck earlier, priced as of a later date given its realised path
(M10 Part 3, SPEC §6.10).

Products carry year fractions from their own trade date (``t = 0``; daily fixings at ``k/252``).
:class:`RealisedHistory` holds the trade date, the trading dates since and their closes, and
states the one date map the module uses: **the k-th trading date after the trade date is trading
index k, and a fixing at ``t`` years fixes on trading index ``j = t · 252``** (every fixing of a
supported product lies on that grid within ``1e-6`` of an integer, otherwise ``ValueError``).  As
of ``dates[n]``:

* fixings with ``j ≤ n`` are **realised** at the closes of ``dates[j]`` (the as-of close
  included: the pricing state of the as-of date carries that close as its spot);
* the others lie ``t − n/252`` years ahead: the seasoned product's time origin is the as-of date
  and its fixing times are the remaining ones, shifted;
* the realised part enters the seasoned product through explicit state inputs whose defaults are
  the fresh product (see each class): the variance products' ``reference_fixing`` /
  ``realised_sum_sq`` / ``realised_count`` (and the variance swap's ``inception``), the cliquet's
  ``reference_fixing`` / ``accrued``, the autocall's ``knocked_in`` / ``memory_coupons`` with its
  remaining dates, coupons and non-call periods re-indexed.  A realised fixing is a constant of
  the seasoned product, so a spot bump of the pricing state moves only the future.

Every product :func:`season` returns carries ``seasoned=True`` (its dates count from the as-of
date) and a seasoned product is refused by a second :func:`season`.  Its discount curve — and a
:class:`Settled` value's — is the as-of curve: the ``discount`` passed (anchored at the as-of
date), by default the product's own curve rolled forward, ``DF(e + τ)/DF(e)``.

A trade whose remaining cash flows the history already determines — a knock-out, an autocall, a
certain vol knock-out, every fixing realised — is returned as :class:`Settled` (deterministic
cash, discounted, standard error 0; :meth:`Settled.as_product` gives it to the pricing machinery
as :class:`SettledCash`).  Cash-flow convention: a cash flow dated ``d`` belongs to
the realised cash flows of every as-of date ``≥ d`` (:attr:`Replay.cash_flows`) and to no
seasoned value from ``d`` on — the product or the settled value priced as of ``d`` holds the
flows after ``d`` only (ex-coupon at the payment date's close), so the P&L of ``(d₀, d₁]`` is
``V(d₁) − V(d₀) + Σ_{d₀ < d ≤ d₁} flows``.  Knock-outs and autocalls observed in the history are
honoured, never dropped; an unsupported product raises with its class name.

Supported (exact type): :class:`~volsto.products.variance.VarianceSwap` (fixing-realised) and
:class:`~volsto.products.variance.VarianceOption`,
:class:`~volsto.products.conditional_variance.ConditionalVarianceSwap` (the up / down variance:
the realised in-region sum and count),
:class:`~volsto.products.barrier.KnockOutOption` / :class:`~volsto.products.barrier.KnockInOption`
(discrete monitoring on the daily closes; a breach settles the knock-out and turns the knock-in
into its vanilla; continuous monitoring and the smart gap raise ``NotImplementedError``),
:class:`~volsto.products.conditional_variance.KnockOutVarianceSwap`,
:class:`~volsto.products.vko.VolKnockOutPut`, :class:`~volsto.products.cliquet.AdditiveCliquet`
and :class:`~volsto.products.autocall.Autocall` (European knock-in, or the discrete American
knock-in of the Phoenix; the continuous knock-in and any gap convention — fixed or smart —
raise ``NotImplementedError``: the history's events are decided on the term-sheet levels, a gap
shifts the priced ones).  The hedge state of a seasoned product
(:func:`volsto.hedging.state.hedge_state`) reads the same realised state.  Checked by
``tests/test_seasoning.py``: age 0 prices bit-identically to the fresh product under the same seed
for every class; a half-way variance swap under Black–Scholes equals the realised part plus the
closed-form remainder within 2 stderr; histories that breach a knock-out barrier or a vol budget
settle; an American knock-in in the history is carried; a cliquet with a realised prefix matches a
brute-force computation in a near-deterministic world; the Phoenix memory accounting on a
synthetic path.
"""

from __future__ import annotations

import datetime as dt
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from volsto.engine.grid import FixingIndex
from volsto.engine.paths import PathSet
from volsto.market.curves import DiscountCurve
from volsto.products.autocall import Autocall
from volsto.products.barrier import KnockInOption, KnockOutOption, _BarrierOption, first_hit_index
from volsto.products.base import Product
from volsto.products.cliquet import AdditiveCliquet
from volsto.products.conditional_variance import ConditionalVarianceSwap, KnockOutVarianceSwap
from volsto.products.vanilla import EuropeanOption
from volsto.products.variance import VarianceOption, VarianceSwap
from volsto.products.vko import VolKnockOutPut

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

TRADING_DAYS_PER_YEAR: Final[int] = 252
"""Trading dates per year of the date map (the library's daily fixing convention, ``k/252``)."""
GRID_TOL: Final[float] = 1e-6
"""Largest distance, in trading days, between ``t · 252`` and the integer a fixing maps to."""


# --------------------------------------------------------------------------------------------
# the history
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RealisedHistory:
    """The realised closes of a trade's underlying from its trade date on.

    ``dates[0]`` is the trade date (its close is the fixing at ``t = 0``) and ``dates[k]`` the
    k-th trading date after it — trading index ``k``, time ``k / per_year`` — strictly
    increasing; ``closes[k]`` is the close of ``dates[k]``.  The history may run past the as-of
    date of a query (only the closes up to it are used)."""

    trade_date: dt.date
    dates: tuple[dt.date, ...]
    closes: tuple[float, ...]
    per_year: int = TRADING_DAYS_PER_YEAR

    def __post_init__(self) -> None:
        dates = tuple(self.dates)
        closes = tuple(float(c) for c in self.closes)
        object.__setattr__(self, "dates", dates)
        object.__setattr__(self, "closes", closes)
        if not dates or dates[0] != self.trade_date:
            raise ValueError("dates must start with the trade date")
        if len(closes) != len(dates):
            raise ValueError("one close per trading date")
        if any(b <= a for a, b in itertools.pairwise(dates)):
            raise ValueError("trading dates must be strictly increasing")
        if not all(np.isfinite(c) and c > 0 for c in closes):
            raise ValueError("closes must be positive and finite")
        if int(self.per_year) != self.per_year or self.per_year <= 0:
            raise ValueError("per_year must be a positive integer")

    def index(self, day: dt.date) -> int:
        """Trading index of ``day`` (a date of the history)."""
        try:
            return self.dates.index(day)
        except ValueError as exc:
            raise ValueError(f"{day} is not a trading date of the history") from exc

    def elapsed(self, day: dt.date) -> float:
        """Years from the trade date to ``day`` on the trading grid: ``index / per_year``."""
        return self.index(day) / self.per_year

    def trading_indices(self, times: FloatArray) -> IntArray:
        """The trading index of each fixing time (``t · per_year``, which must be integral)."""
        x = np.asarray(times, dtype=np.float64) * self.per_year
        j = np.rint(x)
        if np.any(np.abs(x - j) > GRID_TOL):
            bad = np.asarray(times)[np.abs(x - j) > GRID_TOL]
            raise ValueError(
                f"fixing times {bad[:3].tolist()} are not on the {self.per_year}-day trading grid"
            )
        return j.astype(np.int64)

    def closes_at(self, indices: IntArray) -> FloatArray:
        """Closes at trading indices (every index must be in the history)."""
        idx = np.asarray(indices, dtype=np.int64)
        if idx.size and (idx.min() < 0 or idx.max() >= len(self.closes)):
            raise ValueError("a realised fixing lies outside the history")
        return np.asarray([self.closes[int(i)] for i in idx], dtype=np.float64)

    def extended(self, day: dt.date, close: float) -> RealisedHistory:
        """The history with one more trading date (e.g. ``day`` closing at the previous close:
        the held-spot product of a one-day theta, :func:`volsto.risk.attribution.explain`)."""
        return RealisedHistory(
            self.trade_date, (*self.dates, day), (*self.closes, float(close)), self.per_year
        )


# --------------------------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Settled:
    """A trade whose remaining cash is known as of the query date.

    ``amount``: the cash (notional included, undiscounted) paid ``pay_time`` years after the
    as-of date (``≤ 0``: paid on or before it, and then among the realised cash flows).
    :attr:`value` is its present value as of the as-of date — ``amount · DF(pay_time)`` for a
    payment still to come, 0 once paid — with a zero standard error."""

    amount: float
    pay_time: float
    discount: DiscountCurve
    reason: str

    @property
    def stderr(self) -> float:
        """Deterministic cash: no Monte Carlo error."""
        return 0.0

    @property
    def value(self) -> float:
        if self.pay_time <= 0.0:
            return 0.0
        return float(self.amount * float(self.discount.df(self.pay_time)))

    def as_product(self) -> SettledCash:
        """The settled cash as a :class:`~volsto.products.base.Product` (:class:`SettledCash`),
        for the pricing machinery — the risk engine and the P&L attribution (``explain(...,
        product_1=settled)``) price it at its value with a zero standard error."""
        return SettledCash(self.amount, self.pay_time, self.discount, self.reason)

    def __repr__(self) -> str:
        when = "paid" if self.pay_time <= 0 else f"payable in {self.pay_time:.6g}y"
        return f"Settled({self.amount:.8g} {when}, value {self.value:.8g} ± 0: {self.reason})"


SETTLED_GRID_TIME: Final[float] = 1.0 / TRADING_DAYS_PER_YEAR
"""The one fixing time of a :class:`SettledCash` (a grid point for the engine; the payoff reads
no path)."""


class SettledCash(Product):
    """A settled trade's known cash as a product: ``amount · DF(pay_time)`` on every path while
    the cash is to come, 0 once paid (``pay_time ≤ 0``), discounted on the product's curve — so
    the risk engine's rebinding to a state's rate curve discounts it on that state's curve.
    Constant payoff: its Monte Carlo price has a zero standard error.  Built by
    :meth:`Settled.as_product`; never seasoned (``is_seasoned`` is true)."""

    seasoned = True

    def __init__(
        self, amount: float, pay_time: float, discount: DiscountCurve, reason: str = ""
    ) -> None:
        super().__init__(discount, 1.0)
        if not (np.isfinite(amount) and np.isfinite(pay_time)):
            raise ValueError("amount and pay_time must be finite")
        self.amount = float(amount)
        self.pay_time = float(pay_time)
        self.reason = str(reason)

    @property
    def is_seasoned(self) -> bool:
        return True

    @property
    def fixing_times(self) -> FloatArray:
        return np.array([SETTLED_GRID_TIME])

    @property
    def pay_times(self) -> FloatArray:
        return np.array([max(self.pay_time, SETTLED_GRID_TIME)])

    @property
    def value(self) -> float:
        if self.pay_time <= 0.0:
            return 0.0
        return float(self.amount * float(self.df(self.pay_time)))

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        return np.full(paths.n_paths, self.value)

    def aged(self, dt: float) -> Product:
        return SettledCash(self.amount, self.pay_time - dt, self.discount, self.reason)

    def __repr__(self) -> str:
        when = "paid" if self.pay_time <= 0 else f"payable in {self.pay_time:.6g}y"
        return f"Settled cash {self.amount:.8g} {when} ({self.reason})"


@dataclass(frozen=True)
class RealisedCashFlow:
    """A cash flow the history fixed and paid (``date`` ≤ the as-of date)."""

    date: dt.date
    amount: float
    label: str


@dataclass(frozen=True)
class Replay:
    """The replay of a product over its history up to ``as_of``: the seasoned product or the
    settled value (:attr:`result`), the cash flows paid on or before ``as_of`` and the realised
    state for the record (:attr:`state`)."""

    as_of: dt.date
    elapsed: float
    result: Product | Settled
    cash_flows: tuple[RealisedCashFlow, ...] = ()
    state: dict[str, Any] = field(default_factory=dict)

    @property
    def settled(self) -> bool:
        return isinstance(self.result, Settled)


# --------------------------------------------------------------------------------------------
# the replays
# --------------------------------------------------------------------------------------------


@dataclass
class _Ctx:
    history: RealisedHistory
    n: int
    elapsed: float
    discount: DiscountCurve
    as_of: dt.date

    def flows(self, j: int, amount: float, label: str) -> list[RealisedCashFlow]:
        """The flow of trading index ``j`` when it is paid by the as-of date."""
        if j > self.n:
            return []
        return [RealisedCashFlow(self.history.dates[j], float(amount), label)]

    def settled(
        self,
        amount: float,
        pay_time: float,
        reason: str,
        j_pay: int,
        label: str,
        prior: Sequence[RealisedCashFlow] = (),
        **state: Any,
    ) -> Replay:
        """The settled replay: the terminal cash, paid at trading index ``j_pay`` (a realised
        flow once ``j_pay ≤ n``), after the ``prior`` flows."""
        return Replay(
            self.as_of,
            self.elapsed,
            Settled(float(amount), float(pay_time), self.discount, reason),
            (*prior, *self.flows(j_pay, amount, label)),
            {"settled": True, "reason": reason, **state},
        )


def _split(ctx: _Ctx, times: FloatArray) -> tuple[IntArray, NDArray[np.bool_]]:
    j = ctx.history.trading_indices(times)
    return j, j <= ctx.n


def _replay_variance_swap(p: VarianceSwap, ctx: _Ctx) -> Replay:
    if p.use_simulation_grid:
        raise NotImplementedError("a variance swap realised on the simulation grid")
    fix = p.fixing_times
    j, real = _split(ctx, fix)
    remaining = fix[~real] - ctx.elapsed
    if not real.any():  # a forward-start swap before its start
        return Replay(
            ctx.as_of,
            ctx.elapsed,
            VarianceSwap(
                remaining,
                p.strike,
                ctx.discount,
                p.notional,
                p.annualisation,
                inception=float(fix[0]) - ctx.elapsed,
                seasoned=True,
            ),
            (),
            {"realised_returns": 0},
        )
    closes = ctx.history.closes_at(j[real])
    r = np.diff(np.log(closes))
    sum_sq = float(np.sum(r * r))
    count = int(r.size)
    state: dict[str, Any] = {"realised_returns": count, "realised_sum_sq": sum_sq}
    if real.all():
        rv = p.annualisation_factor * sum_sq
        amount = p.notional * (rv - p.strike)
        return ctx.settled(
            amount,
            float(fix[-1]) - ctx.elapsed,
            "every fixing realised",
            int(j[-1]),
            "variance swap settlement",
            realised_variance=rv,
            **state,
        )
    seasoned = VarianceSwap(
        remaining,
        p.strike,
        ctx.discount,
        p.notional,
        p.annualisation,
        reference_fixing=float(closes[-1]),
        realised_sum_sq=sum_sq,
        realised_count=count,
        inception=float(fix[0]) - ctx.elapsed,
        seasoned=True,
    )
    return Replay(ctx.as_of, ctx.elapsed, seasoned, (), state)


def _replay_variance_option(p: VarianceOption, ctx: _Ctx) -> Replay:
    """The option on realised variance: seasoned as the variance swap (the realised sum of squares,
    the reference close, the inception), settled at its intrinsic value once every fixing is
    realised."""
    fix = p.fixing_times
    j, real = _split(ctx, fix)
    remaining = fix[~real] - ctx.elapsed
    terms: dict[str, Any] = {
        "cp": p.cp,
        "underlying": p.underlying,
        "notional": p.notional,
        "annualisation": p.annualisation,
    }
    if not real.any():  # a forward-start option before its start
        seasoned0 = VarianceOption(
            remaining,
            p.strike_vol,
            ctx.discount,
            **terms,
            inception=float(fix[0]) - ctx.elapsed,
            seasoned=True,
        )
        return Replay(ctx.as_of, ctx.elapsed, seasoned0, (), {"realised_returns": 0})
    closes = ctx.history.closes_at(j[real])
    r = np.diff(np.log(closes))
    sum_sq = float(np.sum(r * r))
    count = int(r.size)
    state: dict[str, Any] = {"realised_returns": count, "realised_sum_sq": sum_sq}
    if real.all():
        rv = p.annualisation_factor * sum_sq
        amount = p.notional * float(p.intrinsic(np.array([rv]))[0])
        return ctx.settled(
            amount,
            float(fix[-1]) - ctx.elapsed,
            "every fixing realised",
            int(j[-1]),
            "variance option settlement",
            realised_variance=rv,
            **state,
        )
    seasoned = VarianceOption(
        remaining,
        p.strike_vol,
        ctx.discount,
        **terms,
        reference_fixing=float(closes[-1]),
        realised_sum_sq=sum_sq,
        realised_count=count,
        inception=float(fix[0]) - ctx.elapsed,
        seasoned=True,
    )
    return Replay(ctx.as_of, ctx.elapsed, seasoned, (), state)


def _replay_ko_varswap(p: KnockOutVarianceSwap, ctx: _Ctx) -> Replay:
    fix = p.fixing_times
    j, real = _split(ctx, fix)
    n_life = p.n_returns
    maturity_left = float(fix[-1]) - ctx.elapsed
    if not real.any():
        return Replay(ctx.as_of, ctx.elapsed, _ko_like(p, fix[~real] - ctx.elapsed, ctx), (), {})
    closes = ctx.history.closes_at(j[real])
    ls = np.log(closes)
    r2 = p.squared_returns(ls[None, :])[0]
    hit = p.beyond_barrier(ls)
    if hit.any() or real.all():
        first = int(np.argmax(hit))
        tau = min(first, n_life) if hit.any() else n_life
        accrued = p.annualisation / n_life * float(np.sum(r2[:tau]))
        amount = p.notional * (accrued - p.strike_vol**2 * tau / n_life)
        reason = (
            f"knocked out at fixing {first} ({ctx.history.dates[int(j[first])]})"
            if hit.any()
            else "every fixing realised"
        )
        # paid at maturity, or at the knock-out close when the swap settles there
        at_ko = hit.any() and p.settlement == "knock_out"
        j_pay = int(j[real][first]) if at_ko else int(j[-1])
        pay_time = float(fix[real][first]) - ctx.elapsed if at_ko else maturity_left
        return ctx.settled(
            amount,
            pay_time,
            reason,
            j_pay,
            "knock-out variance swap settlement",
            tau=tau,
            accrued=accrued,
            knocked_out=bool(hit.any()),
        )
    seasoned = _ko_like(
        p,
        fix[~real] - ctx.elapsed,
        ctx,
        reference_fixing=float(closes[-1]),
        realised_sum_sq=float(np.sum(r2)),
        realised_count=int(r2.size),
    )
    state = {"realised_returns": int(r2.size), "realised_sum_sq": float(np.sum(r2))}
    return Replay(ctx.as_of, ctx.elapsed, seasoned, (), state)


def _ko_like(
    p: KnockOutVarianceSwap, times: FloatArray, ctx: _Ctx, **state: Any
) -> KnockOutVarianceSwap:
    return KnockOutVarianceSwap(
        times,
        p.barrier,
        p.strike_vol,
        ctx.discount,
        direction=p.direction,
        strict=p.strict,
        settlement=p.settlement,
        daily_cap=p.daily_cap,
        annualisation=p.annualisation,
        notional=p.notional,
        seasoned=True,
        **state,
    )


def _replay_conditional_varswap(p: ConditionalVarianceSwap, ctx: _Ctx) -> Replay:
    """Up / down (conditional or corridor) variance swap: the realised in-region sum of squares
    and count ``D`` folded into the seasoned product (the reference close decides the next
    return's ``"prev"`` indicator); settled once every fixing is realised."""
    fix = p.fixing_times
    j, real = _split(ctx, fix)
    remaining = fix[~real] - ctx.elapsed
    kw: dict[str, Any] = {
        "strict": p.strict,
        "daily_cap": p.daily_cap,
        "annualisation": p.annualisation,
        "notional": p.notional,
        "seasoned": True,
    }
    if not real.any():  # a forward-start swap before its start
        raise NotImplementedError("a forward-start conditional variance swap before its start")
    closes = ctx.history.closes_at(j[real])
    ls = np.log(closes)[None, :]
    ind = p.indicators(ls)[0]
    r2 = p.squared_returns(ls)[0]
    sum_in = float(np.sum(r2 * ind))
    d_in = int(np.sum(ind))
    state: dict[str, Any] = {
        "realised_returns": int(r2.size),
        "realised_sum_sq_in": sum_in,
        "realised_in_count": d_in,
    }
    if real.all():
        n = int(r2.size)
        accrued = p.annualisation / n * sum_in
        k2 = p.strike_vol**2 * (d_in / n if p.convention == "conditional" else 1.0)
        return ctx.settled(
            p.notional * (accrued - k2),
            float(fix[-1]) - ctx.elapsed,
            "every fixing realised",
            int(j[-1]),
            f"{p.side}-variance swap settlement",
            accrued=accrued,
            **state,
        )
    seasoned = ConditionalVarianceSwap(
        remaining,
        p.barrier,
        p.side,
        p.indicator,
        p.convention,
        p.strike_vol,
        ctx.discount,
        reference_fixing=float(closes[-1]),
        realised_sum_sq=sum_in,
        realised_count=int(r2.size),
        realised_in_count=d_in,
        **kw,
    )
    return Replay(ctx.as_of, ctx.elapsed, seasoned, (), state)


def _replay_barrier_option(p: _BarrierOption, ctx: _Ctx) -> Replay:
    """Discretely monitored knock-out / knock-in option on the daily closes: a breach in the
    history settles the knock-out (its rebate, at the hit or at maturity) and turns the knock-in
    into its vanilla; otherwise the seasoned option monitors the remaining dates (the as-of close,
    realised and not breached, stays at ``t = 0`` like an aged product's); settled once the
    expiry is realised."""
    if p.monitoring != "discrete":
        raise NotImplementedError(
            "a continuously monitored barrier option: the history holds the daily closes only"
        )
    if p.gap is not None and p.gap.smart:
        raise NotImplementedError("a smart-gap barrier option (the gap shifts the priced level)")
    sched = p.schedule
    j, real = _split(ctx, sched)
    j_T = int(ctx.history.trading_indices(np.array([p.T]))[0])
    maturity_left = p.T - ctx.elapsed
    hit = int(real.sum())
    if real.any():
        closes = ctx.history.closes_at(j[real])
        hit = int(
            first_hit_index(
                np.log(closes)[None, :], p.ln_barrier, p.direction, strict=bool(p.strict)
            )[0]
        )
    knocked = hit < int(real.sum())
    state: dict[str, Any] = {"knocked": knocked, "realised_monitoring_dates": int(real.sum())}

    def vanilla_amount() -> float:
        s_T = float(ctx.history.closes_at(np.array([j_T]))[0])
        return float(p.notional * max(p.cp * (s_T - p.strike), 0.0))

    if knocked:
        j_hit = int(j[real][hit])
        reason = f"barrier breached at monitoring date {hit} ({ctx.history.dates[j_hit]})"
        if p.knock == "out":
            at_hit = p.rebate_timing == "hit"
            return ctx.settled(
                p.notional * p.rebate,
                (float(sched[real][hit]) - ctx.elapsed) if at_hit else maturity_left,
                reason,
                j_hit if at_hit else j_T,
                "knock-out rebate",
                **state,
            )
        if j_T <= ctx.n:
            return ctx.settled(
                vanilla_amount(),
                maturity_left,
                reason + "; expiry realised",
                j_T,
                "knock-in settlement",
                **state,
            )
        van = EuropeanOption(p.strike, maturity_left, p.cp, ctx.discount, p.notional, seasoned=True)
        return Replay(ctx.as_of, ctx.elapsed, van, (), {**state, "reason": reason})
    if j_T <= ctx.n:
        amount = vanilla_amount() if p.knock == "out" else float(p.notional * p.rebate)
        return ctx.settled(
            amount,
            maturity_left,
            "expiry realised, never breached",
            j_T,
            f"knock-{p.knock} settlement",
            **state,
        )
    at_origin = np.array([0.0]) if (j[real] == ctx.n).any() else np.array([])
    remaining = np.concatenate([at_origin, sched[~real] - ctx.elapsed])
    seasoned = p._rebuild(type(p), maturity_left, remaining, discount=ctx.discount, seasoned=True)
    return Replay(ctx.as_of, ctx.elapsed, seasoned, (), state)


def _replay_vko(p: VolKnockOutPut, ctx: _Ctx) -> Replay:
    fix = p.fixing_times
    j, real = _split(ctx, fix)
    maturity_left = p.T - ctx.elapsed

    def like(times: FloatArray, **state: Any) -> VolKnockOutPut:
        return VolKnockOutPut(
            p.strike,
            float(times[-1]),
            p.vol_ko,
            times,
            ctx.discount,
            daily_cap=p.daily_cap,
            annualisation=p.annualisation,
            notional=p.notional,
            knock_in=p.knock_in,
            seasoned=True,
            **state,
        )

    if not real.any():
        return Replay(ctx.as_of, ctx.elapsed, like(fix[~real] - ctx.elapsed), (), {})
    closes = ctx.history.closes_at(j[real])
    r2 = p.squared_returns(np.log(closes)[None, :])[0]
    realised = float(np.sum(r2))
    budget = p.variance_budget
    state: dict[str, Any] = {
        "realised_returns": int(r2.size),
        "realised_sum_sq": realised,
        "variance_budget": budget,
    }
    knocked_out = realised >= budget
    if knocked_out and not p.knock_in:
        k = int(np.argmax(np.cumsum(r2) >= budget)) + 1  # the fixing that made it certain
        return ctx.settled(
            0.0,
            maturity_left,
            f"vol knock-out certain at fixing {k} ({ctx.history.dates[int(j[k])]}): realised "
            f"sum of squares {realised:.6g} >= budget {budget:.6g}",
            int(j[-1]),
            "vol knock-out put settlement",
            knocked_out=True,
            **state,
        )
    if real.all():
        put = max(p.strike - float(closes[-1]), 0.0)
        gate = float(knocked_out) if p.knock_in else float(not knocked_out)
        return ctx.settled(
            p.notional * put * gate,
            maturity_left,
            "every fixing realised",
            int(j[-1]),
            "vol knock-out put settlement",
            knocked_out=knocked_out,
            **state,
        )
    seasoned = like(
        fix[~real] - ctx.elapsed,
        reference_fixing=float(closes[-1]),
        realised_sum_sq=realised,
        realised_count=int(r2.size),
    )
    return Replay(ctx.as_of, ctx.elapsed, seasoned, (), {"knocked_out": knocked_out, **state})


def _replay_cliquet(p: AdditiveCliquet, ctx: _Ctx) -> Replay:
    fix = p.fixing_times
    j, real = _split(ctx, fix)
    if not real.any():
        fwd = AdditiveCliquet(
            fix[~real] - ctx.elapsed, ctx.discount, **p.terms(), notional=p.notional, seasoned=True
        )
        return Replay(ctx.as_of, ctx.elapsed, fwd, (), {})
    closes = ctx.history.closes_at(j[real])
    # the product's own arithmetic: exp(Δ ln S) − 1, clipped
    r = np.exp(np.diff(np.log(closes))) - 1.0
    accrued = float(np.sum(np.clip(r, p.local_floor, p.local_cap)))
    state: dict[str, Any] = {"realised_periods": int(r.size), "accrued": accrued}
    if real.all():
        total = float(np.clip(accrued, p.global_floor, p.global_cap))
        return ctx.settled(
            p.notional * total,
            float(fix[-1]) - ctx.elapsed,
            "every fixing realised",
            int(j[-1]),
            "cliquet settlement",
            **state,
        )
    seasoned = AdditiveCliquet(
        fix[~real] - ctx.elapsed,
        ctx.discount,
        **p.terms(),
        notional=p.notional,
        reference_fixing=float(closes[-1]),
        accrued=accrued,
        seasoned=True,
    )
    return Replay(ctx.as_of, ctx.elapsed, seasoned, (), state)


def _replay_autocall(p: Autocall, ctx: _Ctx) -> Replay:
    if p.gap is not None:
        raise NotImplementedError(
            f"seasoning a note under a gap convention ({p.gap!r}): the history's events are "
            "decided on the term-sheet levels, the gap shifts the priced ones"
        )
    if p.ki_type == "american" and p.ki_monitoring != "discrete":
        raise NotImplementedError(
            "seasoning a continuously monitored knock-in (the closes do not decide it)"
        )
    h = ctx.history
    obs = p.observation_times
    n_dates = p.n_dates
    j_obs = h.trading_indices(obs)
    j_ki = None if p.ki_fixing_times is None else h.trading_indices(p.ki_fixing_times)
    c = p.coupon_schedule
    ac_coupon = np.zeros(n_dates) if p.has_coupon_leg else c
    s_ref = p.spot_reference
    barrier = p.ki_barrier
    levels = p.autocall_levels
    cb = None if p.coupon_barrier is None else p.coupon_barrier * s_ref
    knocked = False
    knocked_date: dt.date | None = None
    memory = 0.0
    flows: list[RealisedCashFlow] = []

    def monitor(upto: int) -> None:
        nonlocal knocked, knocked_date
        if j_ki is None or knocked:
            return
        seen = j_ki[j_ki <= upto]
        if seen.size:
            below = ctx.history.closes_at(seen) < barrier
            if below.any():
                knocked = True
                knocked_date = h.dates[int(seen[int(np.argmax(below))])]

    first_live = n_dates
    for i in range(n_dates):
        ji = int(j_obs[i])
        if ji > ctx.n:
            first_live = i
            break
        s = float(h.closes[ji])
        monitor(ji)
        cash = 0.0
        if p.has_coupon_leg:
            pay = p.guaranteed_coupons or (cb is not None and s >= cb)
            if pay:
                cash += float(c[i]) + (memory if p.memory else 0.0)
                memory = 0.0
            elif p.memory:
                memory += float(c[i])
        called = i >= p.non_call_periods and s >= float(levels[i])
        observed = {"knocked_in": knocked, "memory_coupons": memory, "dates_observed": i + 1}
        if called:
            cash += 1.0 + float(ac_coupon[i])
            return ctx.settled(
                p.notional * cash,
                float(obs[i]) - ctx.elapsed,
                f"autocalled at observation {i + 1} ({h.dates[ji]})",
                ji,
                f"autocall redemption {i + 1}",
                flows,
                **observed,
            )
        if i == n_dates - 1:
            ki = (s < barrier) if p.ki_type == "european" else knocked
            gate = p.final_redemption == "knock_in" or (cb is not None and s < cb)
            loss = max(s_ref - s, 0.0) / s_ref if (ki and gate) else 0.0
            cash += 1.0 - loss
            return ctx.settled(
                p.notional * cash,
                float(obs[i]) - ctx.elapsed,
                f"matured ({h.dates[ji]}), knock-in {bool(ki)}",
                ji,
                "final redemption",
                flows,
                **{**observed, "knocked_in": bool(ki), "put_loss": loss},
            )
        if cash:
            flows.append(RealisedCashFlow(h.dates[ji], p.notional * cash, f"coupon {i + 1}"))
    monitor(ctx.n)
    i0 = first_live
    changes: dict[str, Any] = {
        "observation_times": obs[i0:] - ctx.elapsed,
        "autocall_barriers": p.autocall_barriers[i0:].copy(),
        "non_call_periods": max(0, p.non_call_periods - i0),
        "discount": ctx.discount,
        "knocked_in": knocked,
        "memory_coupons": memory,
        "seasoned": True,
    }
    if not (p.has_coupon_leg and isinstance(p.coupons, float)):
        # the growing coupon c_i = i c keeps its original index; a schedule is re-indexed
        changes["coupons"] = tuple(float(x) for x in c[i0:])
    if j_ki is not None and p.ki_fixing_times is not None:
        changes["ki_fixing_times"] = p.ki_fixing_times[j_ki > ctx.n] - ctx.elapsed
    seasoned = p.replace(**changes)
    state: dict[str, Any] = {
        "knocked_in": knocked,
        "knocked_in_date": None if knocked_date is None else knocked_date.isoformat(),
        "memory_coupons": memory,
        "dates_observed": i0,
    }
    return Replay(ctx.as_of, ctx.elapsed, seasoned, tuple(flows), state)


_REPLAYS: dict[type, Callable[[Any, _Ctx], Replay]] = {
    VarianceSwap: _replay_variance_swap,
    VarianceOption: _replay_variance_option,
    ConditionalVarianceSwap: _replay_conditional_varswap,
    KnockOutOption: _replay_barrier_option,
    KnockInOption: _replay_barrier_option,
    KnockOutVarianceSwap: _replay_ko_varswap,
    VolKnockOutPut: _replay_vko,
    AdditiveCliquet: _replay_cliquet,
    Autocall: _replay_autocall,
}
SUPPORTED: tuple[str, ...] = tuple(cls.__name__ for cls in _REPLAYS)
"""The product classes :func:`season` accepts (exact types)."""


# --------------------------------------------------------------------------------------------
# entry points
# --------------------------------------------------------------------------------------------


def replay(
    product: Product,
    history: RealisedHistory,
    as_of: dt.date,
    *,
    discount: DiscountCurve | None = None,
) -> Replay:
    """Replay ``product`` (struck on ``history.trade_date``) over its history up to ``as_of``
    (module docstring): the seasoned product or the settled value, the paid cash flows and the
    realised state.  ``discount`` is the as-of date's curve, anchored at ``as_of`` (``DF(0) =
    1`` on that date): the seasoned product discounts with it (the risk engine rebinds it to
    the priced state's curve anyway) and :attr:`Settled.value` is ``amount · discount.df(τ)``.
    Default: the product's own (trade-date) curve rolled forward to ``as_of``,
    ``DF(e + τ)/DF(e)`` (:meth:`~volsto.market.curves.DiscountCurve.rolled`).  Every product
    returned carries ``seasoned=True``, so it is never seasoned again (its dates count from
    ``as_of``)."""
    handler = _REPLAYS.get(type(product))
    if handler is None:
        raise TypeError(
            f"season: {type(product).__name__} is not supported (supported: {', '.join(SUPPORTED)})"
        )
    if bool(getattr(product, "is_seasoned", False)):
        raise ValueError(
            f"season: {type(product).__name__} already carries a seasoned state (its dates count "
            "from its own as-of date); season the fresh product over the whole history"
        )
    n = history.index(as_of)
    elapsed = n / history.per_year
    as_of_curve = product.discount.rolled(elapsed) if discount is None else discount
    ctx = _Ctx(history, n, elapsed, as_of_curve, as_of)
    return handler(product, ctx)


def season(
    product: Product,
    history: RealisedHistory,
    as_of: dt.date,
    *,
    discount: DiscountCurve | None = None,
) -> Product | Settled:
    """The product to price from ``as_of`` (time origin ``as_of``, the realised state folded in)
    or its :class:`Settled` value (:func:`replay`)."""
    return replay(product, history, as_of, discount=discount).result


__all__ = [
    "GRID_TOL",
    "SETTLED_GRID_TIME",
    "SUPPORTED",
    "TRADING_DAYS_PER_YEAR",
    "RealisedCashFlow",
    "RealisedHistory",
    "Replay",
    "Settled",
    "SettledCash",
    "replay",
    "season",
]
