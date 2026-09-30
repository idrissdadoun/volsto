"""M10 Part 3 (G2): seasoned trades (``volsto.products.seasoning``, SPEC §6.10).

The book's classes priced as of a date after their trade date given the realised closes: age 0
is the fresh product bit for bit under the same seed; the explicit state inputs' defaults are the
fresh product; a half-way variance swap under Black–Scholes equals its realised part plus the
closed-form remainder within 2 stderr (with the pricing spot on and off the realised reference);
knock-outs, vol budgets and autocalls in the history settle; an American knock-in in the history
is carried; a cliquet with a realised prefix matches a brute force in a near-deterministic
world; the Phoenix memory coupons are accounted for on a synthetic path.

Black–Scholes only, at most 2·10⁴ paths; no calibration.
"""

from __future__ import annotations

import datetime as dt
import itertools
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

from volsto.config import SimConfig
from volsto.engine.grid import FixingIndex, TimeGrid
from volsto.engine.mc import MonteCarlo, PriceResult
from volsto.engine.paths import PathSet
from volsto.hedging.state import hedge_state
from volsto.market import DiscountCurve, ForwardCurve
from volsto.models.bs import BlackScholes
from volsto.products import (
    AdditiveCliquet,
    Autocall,
    EuropeanOption,
    KnockOutVarianceSwap,
    Phoenix,
    Product,
    RealisedHistory,
    Settled,
    VarianceOption,
    VarianceSwap,
    VolKnockOutPut,
    VolSwap,
    daily_schedule,
    replay,
    season,
)
from volsto.products.barrier import KnockInOption, KnockOutOption
from volsto.products.conditional_variance import ConditionalVarianceSwap, DownVar, UpVar
from volsto.products.gap import GapSpec
from volsto.products.seasoning import SUPPORTED
from volsto.risk.engine import product_key
from volsto.studies.m6 import headline_products

FloatArray = NDArray[np.float64]
R, Q, SIGMA, S0 = 0.02, 0.01, 0.2, 100.0
DT = 1.0 / 252.0


def _fc(spot: float = S0, r: float = R, q: float = Q) -> ForwardCurve:
    return ForwardCurve.flat(spot, r, q)


def _dates(n: int) -> list[dt.date]:
    days = np.busday_offset("2022-07-01", np.arange(n), roll="forward")
    return [d.astype(dt.date) for d in days]


def _history(closes: FloatArray | list[float]) -> RealisedHistory:
    c = np.asarray(closes, dtype=np.float64)
    days = _dates(c.size)
    return RealisedHistory(days[0], tuple(days), tuple(float(x) for x in c))


def _path(n: int, vol: float, seed: int, start: float = S0) -> FloatArray:
    rng = np.random.default_rng(seed)
    steps = vol * np.sqrt(DT) * rng.standard_normal(n - 1) - 0.5 * vol * vol * DT
    return np.asarray(start * np.exp(np.concatenate(([0.0], np.cumsum(steps)))), dtype=np.float64)


def _sim(n_paths: int = 20_000, seed: int = 11) -> SimConfig:
    return SimConfig(n_paths=n_paths, chunk_size=n_paths, dt_max=DT, seed=seed)


def _payoffs(products: list[Product], model: BlackScholes, sim: SimConfig) -> list[PriceResult]:
    """Every product on ONE path set (the union grid, the same draws)."""
    return MonteCarlo(sim).price_many(products, model, keep_payoffs=True)


def _book(disc: DiscountCurve, spot: float = S0) -> dict[str, Product]:
    """The M8b book (``StudyRunner.book``), a daily variance swap and a put on variance."""
    prods: dict[str, Product] = dict(headline_products(disc, spot))
    prods["cliquet 1y"] = AdditiveCliquet.study(1.0, disc)
    prods["vko put 12m"] = VolKnockOutPut(
        spot, 1.0, 0.30, daily_schedule(1.0, 252), disc, notional=1.0 / spot
    )
    prods["ko var 1y"] = KnockOutVarianceSwap(daily_schedule(1.0, 252), 1.1 * spot, 0.20, disc)
    prods["var swap 1y"] = VarianceSwap.daily(1.0, 0.04, disc, annualisation=252.0)
    prods["put on var 1y"] = VarianceOption.daily(1.0, 0.20, disc, annualisation=252.0)
    prods.update(_payoff_study_barriers(disc, spot))
    prods["up var 1y"] = UpVar(daily_schedule(1.0, 252), spot, 0.2, disc, convention="corridor")
    return prods


def _payoff_study_barriers(disc: DiscountCurve, spot: float = S0) -> dict[str, Product]:
    """The payoff study's barrier options: daily-close monitoring, 6m."""
    kw: dict[str, Any] = {
        "monitoring": "discrete",
        "fixing_times": daily_schedule(0.5, 252),
        "strict": True,
    }
    return {
        "uoc 6m": KnockOutOption(spot, 0.5, 1, 1.2 * spot, "up", disc, **kw),
        "dip 6m": KnockInOption(spot, 0.5, -1, 0.8 * spot, "down", disc, **kw),
    }


# --------------------------------------------------------------------------------------------
# age 0 and the defaults
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(_book(_fc().rate_curve)))
def test_age_zero_is_the_fresh_product_bit_for_bit(name: str) -> None:
    fc = _fc()
    product = _book(fc.rate_curve)[name]
    hist = _history(_path(40, 0.3, 5))  # closes[0] = S0, the pricing spot
    seasoned = season(product, hist, hist.trade_date)
    assert isinstance(seasoned, type(product)) and seasoned is not product
    sim = _sim(4_000, seed=3)
    model = BlackScholes(SIGMA, fc)
    fresh = MonteCarlo(sim).price(product, model, keep_payoffs=True)
    aged0 = MonteCarlo(sim).price(seasoned, model, keep_payoffs=True)
    assert fresh.payoffs is not None and aged0.payoffs is not None
    assert np.array_equal(fresh.payoffs, aged0.payoffs), name
    assert fresh.mean == aged0.mean and fresh.stderr == aged0.stderr


STATE_DEFAULTS: dict[type, dict[str, Any]] = {
    VarianceSwap: {
        "reference_fixing": None,
        "realised_sum_sq": 0.0,
        "realised_count": 0,
        "inception": None,
        "seasoned": False,
    },
    KnockOutVarianceSwap: {
        "reference_fixing": None,
        "realised_sum_sq": 0.0,
        "realised_count": 0,
        "seasoned": False,
    },
    VolKnockOutPut: {
        "reference_fixing": None,
        "realised_sum_sq": 0.0,
        "realised_count": 0,
        "seasoned": False,
    },
    VarianceOption: {
        "reference_fixing": None,
        "realised_sum_sq": 0.0,
        "realised_count": 0,
        "inception": None,
        "seasoned": False,
    },
    AdditiveCliquet: {"reference_fixing": None, "accrued": 0.0, "seasoned": False},
    Autocall: {"knocked_in": False, "memory_coupons": 0.0, "seasoned": False},
    ConditionalVarianceSwap: {
        "reference_fixing": None,
        "realised_sum_sq": 0.0,
        "realised_count": 0,
        "realised_in_count": 0,
        "seasoned": False,
    },
    KnockOutOption: {"seasoned": False},
    KnockInOption: {"seasoned": False},
}


def test_explicit_state_defaults_are_the_fresh_product() -> None:
    fc = _fc()
    disc = fc.rate_curve
    book = _book(disc)
    notes = headline_products(disc, S0)
    explicit: dict[str, Product] = {
        "autocall 3y": notes["autocall 3y"].replace(**STATE_DEFAULTS[Autocall]),
        "phoenix 3y": notes["phoenix 3y"].replace(**STATE_DEFAULTS[Autocall]),
        "cliquet 1y": AdditiveCliquet(
            np.linspace(0.0, 1.0, 13),
            disc,
            local_cap=0.02,
            global_floor=0.0,
            **STATE_DEFAULTS[AdditiveCliquet],
        ),
        "vko put 12m": VolKnockOutPut(
            S0,
            1.0,
            0.30,
            daily_schedule(1.0, 252),
            disc,
            notional=1.0 / S0,
            **STATE_DEFAULTS[VolKnockOutPut],
        ),
        "ko var 1y": KnockOutVarianceSwap(
            daily_schedule(1.0, 252),
            1.1 * S0,
            0.20,
            disc,
            **STATE_DEFAULTS[KnockOutVarianceSwap],
        ),
        "var swap 1y": VarianceSwap(
            daily_schedule(1.0, 252),
            0.04,
            disc,
            annualisation=252.0,
            **STATE_DEFAULTS[VarianceSwap],
        ),
        "put on var 1y": VarianceOption(
            daily_schedule(1.0, 252),
            0.20,
            disc,
            annualisation=252.0,
            **STATE_DEFAULTS[VarianceOption],
        ),
        "uoc 6m": KnockOutOption(
            S0,
            0.5,
            1,
            1.2 * S0,
            "up",
            disc,
            monitoring="discrete",
            fixing_times=daily_schedule(0.5, 252),
            strict=True,
            **STATE_DEFAULTS[KnockOutOption],
        ),
        "dip 6m": KnockInOption(
            S0,
            0.5,
            -1,
            0.8 * S0,
            "down",
            disc,
            monitoring="discrete",
            fixing_times=daily_schedule(0.5, 252),
            strict=True,
            **STATE_DEFAULTS[KnockInOption],
        ),
        "up var 1y": ConditionalVarianceSwap(
            daily_schedule(1.0, 252),
            S0,
            "up",
            "prev",
            "corridor",
            0.2,
            disc,
            **STATE_DEFAULTS[ConditionalVarianceSwap],
        ),
    }
    assert set(STATE_DEFAULTS) == {type(p) for p in book.values()}
    assert set(SUPPORTED) == {cls.__name__ for cls in STATE_DEFAULTS}
    model = BlackScholes(SIGMA, fc)
    for name, product in book.items():
        other = explicit[name]
        assert product_key(other) == product_key(product), name
        assert repr(other) == repr(product), name
        assert not product.is_seasoned  # type: ignore[attr-defined]
        a, b = _payoffs([product, other], model, _sim(2_000))
        assert a.payoffs is not None and b.payoffs is not None
        assert np.array_equal(a.payoffs, b.payoffs), name


# --------------------------------------------------------------------------------------------
# variance swap half-way under Black–Scholes
# --------------------------------------------------------------------------------------------


def _bs_log_moments(spot: float, ref: float) -> tuple[float, float]:
    """``E[r_1²]`` of the first remaining daily return (from the realised reference ``ref`` to a
    close one day after a spot of ``spot``) and ``E[r²]`` of the others, Black–Scholes."""
    m = (R - Q - 0.5 * SIGMA**2) * DT
    first = (np.log(spot / ref) + m) ** 2 + SIGMA**2 * DT
    other = m * m + SIGMA**2 * DT
    return float(first), float(other)


@pytest.mark.parametrize("spot_move", [1.0, 1.03])
def test_variance_swap_half_way_equals_accrued_plus_closed_form(spot_move: float) -> None:
    """A 1y daily variance swap (``A = 252``, strike 20² vol) seasoned 126 days on a 30%-vol
    history: ``DF(½) [A/252 (Σ_realised r² + E[r_1²] + 125 E[r²]) − K]`` within 2 stderr, with the
    pricing spot at the realised reference and 3% away from it (the realised fixing stays)."""
    hist = _history(_path(127, 0.30, 21))
    product = VarianceSwap.daily(1.0, 0.04, _fc().rate_curve, annualisation=252.0)
    as_of = hist.dates[126]
    rep = replay(product, hist, as_of)
    seasoned = rep.result
    assert isinstance(seasoned, VarianceSwap)
    closes = np.asarray(hist.closes)
    realised = float(np.sum(np.diff(np.log(closes)) ** 2))
    assert seasoned.realised_count == 126 and seasoned.n_returns == 252
    assert seasoned.realised_sum_sq == pytest.approx(realised, rel=1e-14)
    assert seasoned.reference_fixing == closes[-1]
    assert seasoned.maturity == pytest.approx(0.5, abs=1e-12)
    spot = float(closes[-1]) * spot_move
    fc = _fc(spot)
    first, other = _bs_log_moments(spot, float(closes[-1]))
    expected_rv = (realised + first + 125 * other) * 252.0 / 252.0
    closed = float(fc.rate_curve.df(0.5)) * (expected_rv - 0.04)
    res = MonteCarlo(_sim()).price(seasoned, BlackScholes(SIGMA, fc))
    z = (res.mean - closed) / res.stderr
    print(
        f"spot x{spot_move}: MC {res.mean:.7f} ± {res.stderr:.2e}, closed form {closed:.7f}, "
        f"z {z:+.2f}; realised vol {np.sqrt(realised * 252 / 126):.4f}"
    )
    assert abs(z) < 2.0
    # the end of the life: every fixing realised, settled and paid
    long_hist = _history(np.concatenate([closes, _path(130, 0.3, 22, start=float(closes[-1]))[1:]]))
    end = replay(product, long_hist, long_hist.dates[252])
    assert isinstance(end.result, Settled) and end.result.value == 0.0
    all_closes = np.asarray(long_hist.closes[:253])
    amount = float(np.sum(np.diff(np.log(all_closes)) ** 2)) - 0.04
    assert end.result.amount == pytest.approx(amount, rel=1e-12)
    assert [(f.date, f.amount) for f in end.cash_flows] == [
        (long_hist.dates[252], end.result.amount)
    ]
    day_before = replay(product, long_hist, long_hist.dates[251]).result
    assert isinstance(day_before, VarianceSwap) and day_before.realised_count == 251


def test_variance_option_seasons_like_the_variance_swap() -> None:
    """The put on variance seasoned 126 days carries the variance swap's realised state (sum of
    squares, count, reference close, inception); priced on common paths the seasoned call minus
    the seasoned put is the seasoned variance swap to round-off; at the end of the life it settles
    at its intrinsic value, paid on the last fixing."""
    hist = _history(_path(127, 0.30, 21))
    disc = _fc().rate_curve
    put = VarianceOption.daily(1.0, 0.25, disc, annualisation=252.0)
    call = VarianceOption.daily(1.0, 0.25, disc, cp=1, annualisation=252.0)
    swap = VarianceSwap.daily(1.0, 0.0625, disc, annualisation=252.0)
    as_of = hist.dates[126]
    sp, sc, sv = (replay(x, hist, as_of).result for x in (put, call, swap))
    assert isinstance(sp, VarianceOption) and isinstance(sc, VarianceOption)
    assert isinstance(sv, VarianceSwap)
    for attr in ("realised_sum_sq", "realised_count", "reference_fixing", "inception", "n_returns"):
        assert getattr(sp, attr) == getattr(sv, attr), attr
    fc = _fc(float(hist.closes[-1]))
    res = MonteCarlo(_sim(4_000)).price_many([sc, sp, sv], BlackScholes(SIGMA, fc))
    assert res[0].mean - res[1].mean == pytest.approx(res[2].mean, abs=1e-12)
    long_hist = _history(
        np.concatenate([hist.closes, _path(130, 0.3, 22, start=float(hist.closes[-1]))[1:]])
    )
    end = replay(put, long_hist, long_hist.dates[252])
    assert isinstance(end.result, Settled) and end.result.value == 0.0
    rv = float(np.sum(np.diff(np.log(np.asarray(long_hist.closes[:253]))) ** 2))
    assert end.result.amount == pytest.approx(max(0.0625 - rv, 0.0), abs=1e-15)
    assert [f.date for f in end.cash_flows] == [long_hist.dates[252]]


# --------------------------------------------------------------------------------------------
# knock-out variance swap
# --------------------------------------------------------------------------------------------


def test_ko_variance_swap_breach_settles() -> None:
    disc = _fc().rate_curve
    product = KnockOutVarianceSwap(daily_schedule(1.0, 252), 110.0, 0.20, disc, notional=2.0)
    closes = np.concatenate([np.linspace(100.0, 109.0, 30), [111.0], np.linspace(108.0, 95.0, 40)])
    hist = _history(closes)
    tau = 30  # the first close strictly above 110 (fixing 30); its return accrues
    r2 = np.diff(np.log(closes[: tau + 1])) ** 2
    amount = 2.0 * (252.0 / 252.0 * float(np.sum(r2)) - 0.04 * tau / 252.0)
    rep = replay(product, hist, hist.dates[60])
    assert isinstance(rep.result, Settled), rep.result
    assert rep.state["knocked_out"] and rep.state["tau"] == tau
    assert rep.result.amount == pytest.approx(amount, rel=1e-12)
    assert rep.result.pay_time == pytest.approx(1.0 - 60 / 252, abs=1e-12)
    assert rep.result.value == pytest.approx(amount * np.exp(-R * (1.0 - 60 / 252)), rel=1e-12)
    assert rep.result.stderr == 0.0 and rep.cash_flows == ()
    # the knock-out day itself settles; the day before is live
    assert isinstance(season(product, hist, hist.dates[tau]), Settled)
    assert isinstance(season(product, hist, hist.dates[tau - 1]), KnockOutVarianceSwap)
    # strictness: a close AT the barrier knocks only a non-strict swap
    touch = _history(np.concatenate([np.linspace(100.0, 109.0, 30), [110.0, 105.0]]))
    assert isinstance(season(product, touch, touch.dates[31]), KnockOutVarianceSwap)
    loose = KnockOutVarianceSwap(daily_schedule(1.0, 252), 110.0, 0.20, disc, strict=False)
    settled = season(loose, touch, touch.dates[31])
    assert isinstance(settled, Settled) and "fixing 30" in settled.reason
    # a live seasoned swap: the reference cannot sit beyond the barrier
    with pytest.raises(ValueError, match="beyond the barrier"):
        KnockOutVarianceSwap(daily_schedule(0.5, 252)[1:], 110.0, 0.2, disc, reference_fixing=111.0)


def test_ko_variance_swap_settled_at_the_knock_out_pays_on_the_knock_out_date() -> None:
    """Settled at the knock-out: a realised knock-out pays the same amount on the knock-out close
    — a realised cash flow dated that day, nothing left to value — while a swap settled at
    maturity keeps the amount to come; a live seasoned swap keeps the convention."""
    disc = _fc().rate_curve
    closes = np.concatenate([np.linspace(100.0, 109.0, 30), [111.0], np.linspace(108.0, 95.0, 40)])
    hist = _history(closes)
    tau = 30
    kw = dict(notional=2.0, settlement="knock_out")
    product = KnockOutVarianceSwap(daily_schedule(1.0, 252), 110.0, 0.20, disc, **kw)  # type: ignore[arg-type]
    at_maturity = KnockOutVarianceSwap(daily_schedule(1.0, 252), 110.0, 0.20, disc, notional=2.0)
    rep = replay(product, hist, hist.dates[60])
    ref = replay(at_maturity, hist, hist.dates[60])
    assert isinstance(rep.result, Settled) and isinstance(ref.result, Settled)
    assert rep.result.amount == pytest.approx(ref.result.amount, rel=1e-15)
    assert rep.result.pay_time == pytest.approx((tau - 60) / 252, abs=1e-12)
    assert rep.result.value == 0.0  # paid on the knock-out date: nothing left to value
    assert ref.result.value == pytest.approx(
        ref.result.amount * np.exp(-R * (1.0 - 60 / 252)), rel=1e-12
    )
    assert len(rep.cash_flows) == 1 and rep.cash_flows[0].date == hist.dates[tau]
    assert rep.cash_flows[0].amount == pytest.approx(ref.result.amount, rel=1e-15)
    live = season(product, hist, hist.dates[tau - 1])
    assert isinstance(live, KnockOutVarianceSwap) and live.settlement == "knock_out"


def test_seasoned_ko_variance_swap_without_barrier_is_the_seasoned_variance_swap() -> None:
    """``B → ∞``: two independent seasonings agree path by path (the §6.1 limit)."""
    fc = _fc()
    disc = fc.rate_curve
    hist = _history(_path(90, 0.25, 7))
    as_of = hist.dates[89]
    ko = season(KnockOutVarianceSwap(daily_schedule(1.0, 252), 1e9, 0.2, disc), hist, as_of)
    vs = season(VarianceSwap.daily(1.0, 0.04, disc, annualisation=252.0), hist, as_of)
    assert isinstance(ko, KnockOutVarianceSwap) and isinstance(vs, VarianceSwap)
    spot = hist.closes[89] * 0.98
    a, b = _payoffs([ko, vs], BlackScholes(SIGMA, _fc(spot)), _sim(4_000))
    assert a.payoffs is not None and b.payoffs is not None
    np.testing.assert_allclose(a.payoffs, b.payoffs, rtol=1e-12, atol=1e-15)


# --------------------------------------------------------------------------------------------
# autocall / Phoenix: knock-in state, autocall events
# --------------------------------------------------------------------------------------------


def _dip_history(n: int, low: float, at: int) -> FloatArray:
    """Closes from 100 down to ``low`` at index ``at``, then back to 90 at ``n − 1``."""
    down = np.linspace(100.0, low, at + 1)
    up = np.linspace(low, 90.0, n - at)[1:]
    return np.asarray(np.concatenate([down, up]), dtype=np.float64)


def test_autocall_history_crossing_the_knock_in_level() -> None:
    fc = _fc()
    disc = fc.rate_curve
    book = headline_products(disc, S0)
    hist = _history(_dip_history(201, 55.0, 100))  # below KI 60 at 100, back to 90 at 200
    as_of = hist.dates[200]
    # the Phoenix's daily American knock-in: knocked, the state carried
    rep = replay(book["phoenix 3y"], hist, as_of)
    phoenix = rep.result
    assert isinstance(phoenix, Autocall) and phoenix.knocked_in and phoenix.is_seasoned
    # the first close below 60 on the 100 → 55 line: index 89
    assert rep.state["knocked_in"] and rep.state["knocked_in_date"] == hist.dates[89].isoformat()
    assert "knocked in" in repr(phoenix)
    assert phoenix.observation_times.tolist() == pytest.approx(
        [1 - 200 / 252, 2 - 200 / 252, 3 - 200 / 252]
    )
    assert phoenix.ki_fixing_times is not None and phoenix.ki_fixing_times[0] == pytest.approx(DT)
    # a knocked-in note is the European knock-in at 100% (SPEC §6.7), path by path
    knocked_equiv = phoenix.replace(
        knocked_in=False, ki_type="european", ki_level=1.0, ki_monitoring=None, ki_fixing_times=None
    )
    alive = phoenix.replace(knocked_in=False)
    model = BlackScholes(SIGMA, _fc(90.0))
    a, b, c = _payoffs([phoenix, knocked_equiv, alive], model, _sim(20_000))
    assert a.payoffs is not None and b.payoffs is not None and c.payoffs is not None
    np.testing.assert_array_equal(a.payoffs, b.payoffs)
    print(
        f"seasoned Phoenix at 90: knocked {a.mean:.5f} ± {a.stderr:.1e}, not knocked "
        f"{c.mean:.5f} ± {c.stderr:.1e}"
    )
    assert a.mean < c.mean - 10 * a.stderr  # the knock-in history is priced, not dropped
    # the European knock-in autocall: a mid-life dip is not a knock-in
    auto = season(book["autocall 3y"], hist, as_of)
    assert isinstance(auto, Autocall) and not auto.knocked_in
    assert auto.coupon_schedule.tolist() == pytest.approx([0.06, 0.12, 0.18])
    # a close below the level ON the as-of date knocks too
    same_day = _history(np.concatenate([np.linspace(100.0, 70.0, 50), [59.0]]))
    note = season(book["phoenix 3y"], same_day, same_day.dates[50])
    assert isinstance(note, Autocall) and note.knocked_in
    with pytest.raises(ValueError, match="American"):
        book["autocall 3y"].replace(knocked_in=True)


def test_autocall_event_in_the_history_settles() -> None:
    disc = _fc().rate_curve
    book = headline_products(disc, S0)
    closes = np.concatenate([np.linspace(100.0, 96.0, 252), [101.0], np.linspace(101.0, 99.0, 60)])
    hist = _history(closes)
    for name, cash in (("autocall 3y", 1.06), ("phoenix 3y", 1.06)):
        rep = replay(book[name], hist, hist.dates[300])
        assert isinstance(rep.result, Settled), name
        assert rep.result.amount == pytest.approx(cash) and rep.result.value == 0.0
        assert rep.result.pay_time == pytest.approx(1.0 - 300 / 252)
        assert [f.date for f in rep.cash_flows] == [hist.dates[252]]
        assert rep.cash_flows[0].amount == pytest.approx(cash)
        on_day = replay(book[name], hist, hist.dates[252]).result
        assert isinstance(on_day, Settled) and on_day.pay_time == pytest.approx(0.0, abs=1e-12)
        assert on_day.value == 0.0  # ex-coupon at the payment date's close
        before = season(book[name], hist, hist.dates[251])
        assert isinstance(before, Autocall) and before.n_dates == 3


# --------------------------------------------------------------------------------------------
# cliquet: realised prefix against a brute force
# --------------------------------------------------------------------------------------------


def _cliquet_brute_force(
    closes: FloatArray, n: int, rq: float, lf: float, lc: float, gf: float, gc: float
) -> float:
    """The monthly cliquet's undiscounted payoff on the path that follows the realised closes up
    to index ``n`` and then grows deterministically at ``rq = r − q`` (fixings every 21 days)."""
    levels = []
    for j in range(0, 253, 21):
        if j <= n:
            levels.append(float(closes[j]))
        else:
            levels.append(float(closes[n]) * np.exp(rq * (j - n) / 252.0))
    total = 0.0
    for a, b in itertools.pairwise(levels):
        total += min(max(b / a - 1.0, lf), lc)
    return min(max(total, gf), gc)


@pytest.mark.parametrize(
    ("r", "q", "global_cap", "shock"),
    [(0.08, 0.02, None, 1.0), (0.08, 0.02, 0.03, 1.0), (0.0, 0.12, None, 0.8)],
)
def test_cliquet_realised_prefix_brute_force(
    r: float, q: float, global_cap: float | None, shock: float
) -> None:
    """Black–Scholes at 0.01% vol (a near-deterministic forward): the seasoned cliquet — the
    running period started on the realised close at index 21, as of index 30 — prices the brute
    force within 2·10⁻⁶ of notional; uncapped, capped globally, and floored."""
    closes = np.concatenate(
        [
            np.linspace(100.0, 104.0, 22),  # period 1: +4% (capped at 2%)
            np.linspace(104.0, 101.0, 10)[1:] * shock,  # period 2 so far (to index 30)
        ]
    )
    hist = _history(closes)
    n = 30
    disc = _fc(r=r, q=q).rate_curve
    product = AdditiveCliquet.study(1.0, disc, global_cap=global_cap)
    rep = replay(product, hist, hist.dates[n])
    seasoned = rep.result
    assert isinstance(seasoned, AdditiveCliquet)
    assert seasoned.reference_fixing == closes[21] and seasoned.n_periods == 11
    assert seasoned.accrued == pytest.approx(0.02, abs=1e-15)
    gc = np.inf if global_cap is None else global_cap
    brute = _cliquet_brute_force(closes, n, r - q, -np.inf, 0.02, 0.0, gc)
    model = BlackScholes(1e-4, ForwardCurve.flat(float(closes[n]), r, q))
    res = MonteCarlo(_sim(2_000, seed=5)).price(seasoned, model)
    expected = float(disc.df(1.0 - n / 252.0)) * brute
    print(
        f"r={r} q={q} cap={global_cap}: MC {res.mean:.8f} ± {res.stderr:.1e}, brute {expected:.8f}"
    )
    assert abs(res.mean - expected) < 2e-6
    # the settlement at maturity is the brute force on the realised path
    full = np.concatenate([closes, closes[n] * np.exp((r - q) * np.arange(1, 223) / 252.0)])
    end = replay(product, _history(full), _dates(full.size)[252])
    assert isinstance(end.result, Settled)
    assert end.result.amount == pytest.approx(
        _cliquet_brute_force(full, 252, 0.0, -np.inf, 0.02, 0.0, gc), abs=1e-12
    )
    with pytest.raises(NotImplementedError):
        seasoned.decompose()


# --------------------------------------------------------------------------------------------
# VKO: the realised variance budget
# --------------------------------------------------------------------------------------------


def test_vko_realised_variance_above_budget_knocks_out() -> None:
    fc = _fc()
    disc = fc.rate_curve
    product = VolKnockOutPut(S0, 1.0, 0.30, daily_schedule(1.0, 252), disc, notional=1.0 / S0)
    hot = _history(_path(141, 0.60, 31))
    r2 = np.diff(np.log(np.asarray(hot.closes))) ** 2
    k = int(np.argmax(np.cumsum(r2) >= 0.09)) + 1
    assert k < 140  # the 60%-vol history exhausts the 30%-vol budget (0.09) before day 140
    rep = replay(product, hot, hot.dates[140])
    assert isinstance(rep.result, Settled) and rep.result.amount == 0.0
    assert rep.result.value == 0.0 and rep.state["knocked_out"]
    assert f"fixing {k}" in rep.result.reason
    assert isinstance(season(product, hot, hot.dates[k]), Settled)
    assert isinstance(season(product, hot, hot.dates[k - 1]), VolKnockOutPut)
    # the knock-in variant is then certainly knocked in: a live put with the gate at 1
    ki = VolKnockOutPut(S0, 1.0, 0.30, daily_schedule(1.0, 252), disc, knock_in=True)
    live = season(ki, hot, hot.dates[140])
    assert isinstance(live, VolKnockOutPut)
    model = BlackScholes(SIGMA, _fc(hot.closes[140]))
    sim = _sim(4_000)
    paths = MonteCarlo(sim).simulate(model, MonteCarlo(sim).build_grid([live], model))
    idx = FixingIndex(paths.times)
    assert np.all(live.statistics(paths, idx)["ko"] == 1.0)
    # below the budget: the survival test is realised + remaining < budget, path by path
    calm = _history(_path(141, 0.25, 32))
    seasoned = season(product, calm, calm.dates[140])
    assert isinstance(seasoned, VolKnockOutPut)
    realised = float(np.sum(np.diff(np.log(np.asarray(calm.closes))) ** 2))
    assert seasoned.realised_sum_sq == pytest.approx(realised, rel=1e-14)
    model = BlackScholes(0.35, _fc(calm.closes[140]))
    grid = MonteCarlo(sim).build_grid([seasoned], model)
    paths = MonteCarlo(sim).simulate(model, grid)
    idx = grid.fixing_index
    ls = paths.log_spot_at(idx.indices(seasoned.fixing_times))
    ls = np.concatenate([np.full((ls.shape[0], 1), np.log(calm.closes[140])), ls], axis=1)
    brute_alive = realised + np.sum(np.diff(ls, axis=1) ** 2, axis=1) < seasoned.variance_budget
    stats = seasoned.statistics(paths, idx)
    np.testing.assert_array_equal(stats["alive"], brute_alive.astype(np.float64))
    assert 0.05 < stats["alive"].mean() < 0.95


# --------------------------------------------------------------------------------------------
# Phoenix memory coupons
# --------------------------------------------------------------------------------------------


def _quarterly_phoenix(disc: DiscountCurve) -> Autocall:
    return Phoenix(
        [0.25, 0.5, 0.75, 1.0],
        disc,
        spot_reference=S0,
        coupon=0.02,
        coupon_barrier=0.8,
        memory=True,
        ki_level=0.6,
        ki_type="american",
        ki_monitoring="discrete",
        ki_fixing_times=daily_schedule(1.0, 252),
        autocall_barriers=1.0,
    )


def _synthetic_paths(times: FloatArray, levels: list[float], last: list[float]) -> PathSet:
    """Paths flat at ``levels`` then at ``last`` on the final column (``n_paths = len(levels)``)."""
    m = times.size
    ls = np.log(np.repeat(np.asarray(levels, dtype=np.float64)[:, None], m, axis=1))
    ls[:, -1] = np.log(np.asarray(last, dtype=np.float64))
    z = np.zeros_like(ls)
    return PathSet(times, ls, z.copy(), np.zeros((len(levels), m, 0)), z.copy(), z.copy())


def test_phoenix_memory_coupon_accounting() -> None:
    disc = _fc().rate_curve
    note = _quarterly_phoenix(disc)
    base = np.full(201, 95.0)
    # quarter 1 (63): 75 < CB, quarter 2 (126): 78 < CB, quarter 3 (189): 85 in [CB, AC)
    paid = base.copy()
    paid[63], paid[126], paid[189] = 75.0, 78.0, 85.0
    rep = replay(note, _history(paid), _dates(201)[200])
    seasoned = rep.result
    assert isinstance(seasoned, Autocall) and seasoned.memory_coupons == 0.0
    assert [(f.date, f.label) for f in rep.cash_flows] == [(_dates(201)[189], "coupon 3")]
    assert rep.cash_flows[0].amount == pytest.approx(0.06)  # 0.02 + the two missed
    # quarter 3 also below the coupon barrier: 0.06 in memory, nothing paid
    missed = paid.copy()
    missed[189] = 70.0
    rep = replay(note, _history(missed), _dates(201)[200])
    seasoned = rep.result
    assert isinstance(seasoned, Autocall) and rep.cash_flows == ()
    assert seasoned.memory_coupons == pytest.approx(0.06) and not seasoned.knocked_in
    assert seasoned.n_dates == 1 and seasoned.observation_times[0] == pytest.approx(52 / 252)
    assert "coupons in memory" in repr(seasoned)
    # the last quarter on synthetic paths: called (105), coupon only (90), nothing (70),
    # knocked in at the end (50)
    times = np.unique(np.concatenate([[0.0], seasoned.fixing_times]))
    paths = _synthetic_paths(times, [105.0, 90.0, 70.0, 70.0], [105.0, 90.0, 70.0, 50.0])
    pay = seasoned.payoff(paths, FixingIndex(times))
    df = float(disc.df(52 / 252))
    expected = df * np.array([1.0 + 0.08, 1.0 + 0.08, 1.0, 0.5])
    np.testing.assert_allclose(pay, expected, rtol=1e-14)
    coupons = seasoned.statistics(paths, FixingIndex(times))["coupons_paid"]
    np.testing.assert_allclose(coupons, [0.08, 0.08, 0.0, 0.0], rtol=1e-14)
    # the same note without the carried memory pays only the period coupon
    plain = seasoned.replace(memory_coupons=0.0).payoff(paths, FixingIndex(times))
    np.testing.assert_allclose(plain, df * np.array([1.02, 1.02, 1.0, 0.5]), rtol=1e-14)
    with pytest.raises(ValueError, match="memory"):
        headline_products(disc, S0)["autocall 3y"].replace(memory_coupons=0.02)
    # an autocall at quarter 3 pays the coupon with the memory and the redemption
    called = paid.copy()
    called[189] = 101.0
    rep = replay(note, _history(called), _dates(201)[200])
    assert isinstance(rep.result, Settled)
    assert rep.result.amount == pytest.approx(1.0 + 0.06)
    assert [f.amount for f in rep.cash_flows] == [pytest.approx(1.06)]


# --------------------------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------------------------


def test_unsupported_and_invalid_inputs_raise() -> None:
    disc = _fc().rate_curve
    hist = _history(_path(30, 0.2, 1))
    for product in (EuropeanOption(S0, 1.0, -1, disc), VolSwap.daily(1.0, 0.2, disc)):
        with pytest.raises(TypeError, match=type(product).__name__):
            season(product, hist, hist.dates[5])
    varswap = VarianceSwap.daily(1.0, 0.04, disc)
    seasoned = season(varswap, hist, hist.dates[10])
    with pytest.raises(ValueError, match="already carries"):
        season(seasoned, hist, hist.dates[20])  # type: ignore[arg-type]
    off_grid = VarianceSwap(np.linspace(0.0, 1.0, 11), 0.04, disc)  # 25.2-day fixings
    with pytest.raises(ValueError, match="trading grid"):
        season(off_grid, hist, hist.dates[5])
    with pytest.raises(ValueError, match="not a trading date"):
        season(varswap, hist, dt.date(2031, 1, 1))
    days = _dates(3)
    with pytest.raises(ValueError, match="trade date"):
        RealisedHistory(days[1], tuple(days), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="increasing"):
        RealisedHistory(days[0], (days[0], days[2], days[1]), (1.0, 1.0, 1.0))
    with pytest.raises(ValueError, match="positive"):
        RealisedHistory(days[0], tuple(days), (1.0, 0.0, 1.0))
    longer = hist.extended(dt.date(2030, 1, 1), 101.0)
    assert longer.index(dt.date(2030, 1, 1)) == 30 and longer.closes[-1] == 101.0
    with pytest.raises(ValueError, match="realised returns need"):
        VarianceSwap(daily_schedule(1.0, 252)[1:], 0.04, disc, realised_count=3)
    with pytest.raises(ValueError, match="simulation grid"):
        VarianceSwap(
            daily_schedule(1.0, 252)[1:],
            0.04,
            disc,
            use_simulation_grid=True,
            reference_fixing=100.0,
        )
    # the book's rolled seasoned products are still ageable (theta) and rebindable
    assert isinstance(seasoned, VarianceSwap)
    aged = seasoned.aged(0.5 * DT)  # the next fixing is one day ahead
    assert isinstance(aged, VarianceSwap)
    assert aged.start == pytest.approx(seasoned.start - 0.5 * DT)
    assert aged.realised_sum_sq == seasoned.realised_sum_sq
    assert seasoned.with_discount(disc).realised_count == seasoned.realised_count  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------------
# verification follow-ups (2026-09-16): hedge state, the seasoned marker, the as-of curve, gaps
# --------------------------------------------------------------------------------------------


def test_conditional_variance_seasoned_statistics_equal_the_full_path() -> None:
    """Up (``"prev"``) and down (``"curr"``) variance, corridor and conditional: the seasoned
    swap's statistics on the future equal the fresh swap's on the realised closes followed by the
    same future, path by path; a fully realised swap settles at its realised statistics."""
    disc = _fc().rate_curve
    closes = _path(90, 0.3, 17)
    hist = _history(closes)
    n = 89
    for side, conv in (("up", "corridor"), ("down", "corridor"), ("up", "conditional")):
        maker = UpVar if side == "up" else DownVar
        fresh = maker(daily_schedule(1.0, 252), 101.0, 0.2, disc, convention=conv)
        seasoned = season(fresh, hist, hist.dates[n])
        assert isinstance(seasoned, ConditionalVarianceSwap) and seasoned.is_seasoned
        assert seasoned.realised_count == n and 0 < seasoned.realised_in_count < n
        fut, fidx = _future_paths(float(closes[n]), 252 - n, 0.25, seed=5)
        full, full_idx = _full_paths(closes, fut, fidx)
        a, b = seasoned.statistics(fut, fidx), fresh.statistics(full, full_idx)
        for key in ("accrued", "count"):
            np.testing.assert_allclose(a[key], b[key], rtol=1e-12, err_msg=f"{side} {conv}")
        assert "realised returns in the region" in repr(seasoned)
    whole = _path(253, 0.3, 19)
    fresh = UpVar(daily_schedule(1.0, 252), 100.0, 0.2, disc, convention="corridor")
    done = season(fresh, _history(whole), _history(whole).dates[252])
    assert isinstance(done, Settled)
    r = np.diff(np.log(whole))
    ind = whole[:-1] > 100.0
    expected = 252.0 / 252 * float(np.sum(r * r * ind)) - 0.04
    assert done.amount == pytest.approx(expected, rel=1e-12)


def test_barrier_option_history() -> None:
    """Daily-close barrier options: a history breaching the barrier settles the knock-out at its
    (zero) rebate and turns the knock-in into its vanilla; an unbreached history seasons the
    option (the remaining dates, the as-of close at ``t = 0``) with a knock state equal to the
    fresh option's on the full path; an expired history settles at the realised payoff."""
    disc = _fc().rate_curve
    book = _payoff_study_barriers(disc)
    closes = np.full(40, 99.0)
    closes[0] = S0
    closes[20] = 79.0  # below the 80 barrier of the down-and-in put
    hist = _history(closes)
    dip = book["dip 6m"]
    van = season(dip, hist, hist.dates[30])
    assert isinstance(van, EuropeanOption) and van.is_seasoned
    assert pytest.approx(0.5 - 30 / 252) == van.T and van.strike == S0 and van.cp == -1
    dop = KnockOutOption(
        S0,
        0.5,
        -1,
        80.0,
        "down",
        disc,
        monitoring="discrete",
        fixing_times=daily_schedule(0.5, 252),
        strict=True,
    )
    rep = replay(dop, hist, hist.dates[30])
    assert rep.settled and isinstance(rep.result, Settled) and rep.result.amount == 0.0
    assert rep.state["knocked"] and "breached at monitoring date 20" in rep.result.reason
    # unbreached: the up-and-out call seasoned 40 days in, knock state = the full path's
    uoc = book["uoc 6m"]
    path = _path(41, 0.2, 23)
    h2 = _history(path)
    seasoned = season(uoc, h2, h2.dates[40])
    assert isinstance(seasoned, KnockOutOption) and seasoned.is_seasoned
    assert pytest.approx(0.5 - 40 / 252) == seasoned.T and seasoned.schedule[0] == 0.0
    fut, fidx = _future_paths(float(path[40]), 126 - 40, 0.3, seed=9)
    full, full_idx = _full_paths(path, fut, fidx)
    np.testing.assert_array_equal(seasoned.monitor(fut, fidx)[0], uoc.monitor(full, full_idx)[0])
    assert np.any(seasoned.monitor(fut, fidx)[0] == 0.0)
    # expired: settled at the realised vanilla payoff (never breached)
    flat = _history(np.full(127, 110.0))
    out = season(uoc, flat, flat.dates[126])
    assert isinstance(out, Settled) and out.amount == pytest.approx(10.0)
    with pytest.raises(NotImplementedError, match="continuously monitored"):
        season(
            KnockOutOption(S0, 0.5, 1, 120.0, "up", disc, monitoring="continuous"),
            hist,
            hist.dates[3],
        )


def _future_paths(spot: float, horizon: int, vol: float, seed: int) -> tuple[PathSet, FixingIndex]:
    """Black–Scholes paths from ``spot`` recorded on every trading day of ``horizon``."""
    model = BlackScholes(vol, _fc(spot))
    sim = SimConfig(n_paths=2_000, chunk_size=2_000, dt_max=DT, seed=seed)
    grid = TimeGrid.build(np.arange(1, horizon + 1) * DT, sim.dt_max)
    return MonteCarlo(sim).simulate(model, grid), grid.fixing_index


def _full_paths(closes: FloatArray, fut: PathSet, fidx: FixingIndex) -> tuple[PathSet, FixingIndex]:
    """The realised closes followed by the simulated future, on the trade's own clock."""
    horizon = fut.n_cols - 1
    ls_f = fut.log_spot[:, fidx.indices(np.arange(1, horizon + 1) * DT)]
    prefix = np.broadcast_to(np.log(closes), (fut.n_paths, closes.size))
    ls = np.concatenate([prefix, ls_f], axis=1)
    times = np.asarray(np.arange(ls.shape[1]) * DT, dtype=np.float64)
    z = np.zeros_like(ls)
    return PathSet(times, ls, z, np.zeros((*ls.shape, 0)), z.copy(), z.copy()), FixingIndex(times)


def _vko_repro_history() -> FloatArray:
    """±3% alternating closes (the verifier's reproducer): 0.0009 per day of the 0.09 budget."""
    steps = np.tile([0.03, -0.03], 100)
    return np.asarray(S0 * np.exp(np.concatenate([[0.0], np.cumsum(steps)])), dtype=np.float64)


def _phoenix_quarterly(disc: DiscountCurve) -> Autocall:
    return Phoenix(
        np.arange(1, 5) / 4.0,
        disc,
        spot_reference=S0,
        coupon=0.02,
        coupon_barrier=0.8,
        memory=True,
        ki_level=0.6,
        ki_type="american",
        ki_monitoring="discrete",
    )


def test_hedge_state_of_seasoned_products() -> None:
    """``volsto.hedging.state.hedge_state`` of a seasoned product equals that of the fresh product
    on the full path (realised closes, then the future) — features and alive flags — for every
    seasoned class; the verifier's two reproducers (a seasoned VKO whose budget is partly spent,
    a Phoenix knocked in by its history) included."""
    disc = _fc().rate_curve
    base = _path(260, 0.3, 41)
    dip = np.full(260, 90.0)
    dip[0], dip[5] = S0, 50.0  # knocked in on day 5; quarter 1 (day 63) below the coupon barrier
    dip[63] = 70.0
    vko_hist = _vko_repro_history()
    cases: list[tuple[str, Product, FloatArray, int, float]] = [
        ("var swap", VarianceSwap.daily(1.0, 0.04, disc, annualisation=252.0), base, 100, 0.25),
        (
            "ko var",
            KnockOutVarianceSwap(daily_schedule(1.0, 252), 140.0, 0.2, disc),
            base,
            100,
            0.25,
        ),
        ("vko", VolKnockOutPut(S0, 1.0, 0.30, daily_schedule(1.0, 252), disc), vko_hist, 40, 0.45),
        (
            "vki",
            VolKnockOutPut(S0, 1.0, 0.30, daily_schedule(1.0, 252), disc, knock_in=True),
            vko_hist,
            40,
            0.45,
        ),
        ("cliquet", AdditiveCliquet.study(1.0, disc), base, 30, 0.25),
        (
            "down var",
            DownVar(daily_schedule(1.0, 252), 102.0, 0.2, disc, convention="corridor"),
            base,
            100,
            0.25,
        ),
        ("uoc", _payoff_study_barriers(disc)["uoc 6m"], base, 40, 0.25),
        ("phoenix knocked", _phoenix_quarterly(disc), dip, 70, 0.2),
        ("autocall", headline_products(disc, S0)["autocall 3y"], base, 100, 0.25),
    ]
    for name, fresh, closes_all, n, vol in cases:
        hist = _history(closes_all[: n + 1])
        seasoned = season(fresh, hist, hist.dates[n])
        assert isinstance(seasoned, Product) and seasoned.is_seasoned, name  # type: ignore[attr-defined]
        horizon = round(float(np.max(fresh.fixing_times)) * 252) - n
        fut, fidx = _future_paths(float(closes_all[n]), horizon, vol, seed=n)
        full, full_idx = _full_paths(closes_all[: n + 1], fut, fidx)
        for k in (0, 1, 7, 33, 60, horizon - 1):
            t = k * DT
            hs = hedge_state(seasoned, fut, fidx, t)
            hf = hedge_state(fresh, full, full_idx, (n + k) * DT)
            assert hs.names == hf.names, name
            np.testing.assert_allclose(
                hs.features, hf.features, rtol=1e-12, atol=1e-14, err_msg=f"{name} k={k}"
            )
            np.testing.assert_array_equal(hs.alive, hf.alive, err_msg=f"{name} k={k}")
            if name == "vko" and k == 60:
                # the reproducer: alive where the whole-life budget is already spent
                ls = full.log_spot_at(full_idx.indices(np.arange(0, n + k + 1) * DT))
                dead = np.sum(np.diff(ls, axis=1) ** 2, axis=1) >= seasoned.variance_budget  # type: ignore[attr-defined]
                assert dead.any() and not np.any(hs.alive & dead)
            if name == "phoenix knocked":
                assert np.all(hs.features[:, 0] == 1.0)  # the ki feature
                assert np.all(hs.features[:, 1] >= 0.02 - 1e-15) or k * DT >= 52 / 252


def test_seasoned_products_are_never_seasoned_again() -> None:
    """Every product ``season`` returns carries the marker (the forward-start case and a note
    without knock-in or memory state included) and is refused by a second ``season``."""
    disc = _fc().rate_curve
    flat = _history(np.full(400, 95.0))
    fwd_var = VarianceSwap(0.25 + np.arange(0, 190) * DT, 0.04, disc)  # starts on day 63
    fwd_cliquet = AdditiveCliquet(0.25 + np.arange(0, 7) * (21 * DT), disc, local_cap=0.02)
    plain = Autocall(
        np.arange(1, 5) / 4.0, disc, spot_reference=S0, coupons=0.03, ki_level=0.6,
        ki_type="european",
    )  # fmt: skip
    products: list[tuple[Product, int]] = [
        (fwd_var, 10),
        (fwd_cliquet, 10),
        (plain, 70),
        (VolKnockOutPut(S0, 1.0, 0.3, daily_schedule(1.0, 252), disc), 0),
        (KnockOutVarianceSwap(daily_schedule(1.0, 252), 140.0, 0.2, disc), 5),
    ]
    for product, n in products:
        once = season(product, flat, flat.dates[n])
        assert isinstance(once, Product), product
        assert once.is_seasoned and once.seasoned  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match="already carries"):
            season(once, flat, flat.dates[n + 70])
    # the plain note's dates count from the as-of date: a second seasoning would have shifted
    # them again (the verifier's reproducer)
    once = season(plain, flat, flat.dates[70])
    assert isinstance(once, Autocall)
    assert once.observation_times.tolist() == pytest.approx(
        [(126 - 70) * DT, (189 - 70) * DT, 1.0 - 70 * DT]
    )
    assert not once.knocked_in and once.memory_coupons == 0.0
    assert once.with_discount(disc).is_seasoned and once.aged(0.5 * DT).is_seasoned
    assert season(fwd_var, flat, flat.dates[10]).start == pytest.approx(0.25 - 10 * DT)  # type: ignore[union-attr]


def test_settled_value_uses_the_as_of_curve() -> None:
    """A term-structure curve: ``Settled.value`` is ``amount · DF_asof(τ)`` — by default the trade
    curve rolled forward, ``DF(e + τ)/DF(e)``; with an explicit as-of curve, that curve."""
    curve = DiscountCurve([0.5, 1.0, 2.0], [0.01, 0.03, 0.05])
    e = 126 * DT
    rolled = curve.rolled(e)
    taus = np.linspace(0.0, 3.0, 61)
    np.testing.assert_allclose(
        rolled.df(taus), curve.df(e + taus) / curve.df(e), rtol=1e-13, atol=1e-15
    )
    assert curve.rolled(0.0) is curve
    far = curve.rolled(2.5)  # beyond the last pillar: the last forward, flat
    np.testing.assert_allclose(far.df(taus), curve.df(2.5 + taus) / curve.df(2.5), rtol=1e-13)
    ko = KnockOutVarianceSwap(daily_schedule(1.0, 252), 105.0, 0.2, curve)
    hist = _history(S0 * np.exp(0.01 * np.arange(400)))  # above 105 from day 5
    default = season(ko, hist, hist.dates[126])
    assert isinstance(default, Settled) and default.pay_time == pytest.approx(1.0 - e)
    expected = default.amount * float(curve.df(1.0) / curve.df(e))
    assert default.value == pytest.approx(expected, rel=1e-12)
    assert default.value != pytest.approx(default.amount * float(curve.df(default.pay_time)))
    as_of_curve = DiscountCurve([0.25, 1.0], [0.04, 0.045])
    explicit = season(ko, hist, hist.dates[126], discount=as_of_curve)
    assert isinstance(explicit, Settled)
    assert explicit.value == pytest.approx(
        explicit.amount * float(as_of_curve.df(explicit.pay_time)), rel=1e-14
    )
    # a live seasoned product discounts on the same as-of curve
    swap = season(VarianceSwap.daily(1.0, 0.04, curve), hist, hist.dates[126])
    assert isinstance(swap, VarianceSwap)
    np.testing.assert_allclose(swap.df(taus), rolled.df(taus), rtol=1e-15)
    assert season(ko, hist, hist.dates[0]).discount is curve


def test_gapped_notes_are_refused() -> None:
    disc = _fc().rate_curve
    note = headline_products(disc, S0)["phoenix 3y"]
    hist = _history(np.full(30, 95.0))
    for gap in (GapSpec(mode="fixed", fixed_shift=-0.01), GapSpec(mode="smart")):
        with pytest.raises(NotImplementedError, match="gap"):
            season(note.replace(gap=gap), hist, hist.dates[20])
