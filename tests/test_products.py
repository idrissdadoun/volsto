from __future__ import annotations

import numpy as np
import pytest

from volsto.engine import FixingIndex, PathSet
from volsto.market import DiscountCurve
from volsto.products import (
    DigitalOption,
    EuropeanOption,
    ForwardVarianceSwap,
    VarianceSwap,
    VolSwap,
    daily_schedule,
    parse_cp,
    uniform_schedule,
)


def _paths(spots: np.ndarray, times: np.ndarray) -> PathSet:
    n, m = spots.shape
    return PathSet(
        times,
        np.log(spots),
        np.zeros((n, m)),
        np.zeros((n, m, 0)),
        np.zeros((n, m)),
        np.zeros((n, m)),
    )


def test_parse_cp_and_schedules() -> None:
    assert parse_cp("call") == 1 and parse_cp("P") == -1 and parse_cp(-1) == -1
    with pytest.raises(ValueError):
        parse_cp("straddle")
    np.testing.assert_allclose(uniform_schedule(1.0, 4), [0, 0.25, 0.5, 0.75, 1.0])
    d = daily_schedule(0.5, per_year=252)
    assert d.size == 127 and d[0] == 0.0 and d[-1] == 0.5
    f = daily_schedule(1.0, per_year=252, start=0.5)
    assert f[0] == 0.5 and f.size == 127


def test_vanilla_payoffs(discount: DiscountCurve) -> None:
    times = np.array([0.0, 1.0])
    ps = _paths(np.array([[100.0, 120.0], [100.0, 80.0]]), times)
    idx = FixingIndex(times)
    df = float(discount.df(1.0))
    call = EuropeanOption(100.0, 1.0, "call", discount, notional=2.0)
    put = EuropeanOption(100.0, 1.0, "put", discount)
    dig = DigitalOption(90.0, 1.0, "put", discount, payout=5.0)
    np.testing.assert_allclose(call.payoff(ps, idx), [2 * 20 * df, 0.0])
    np.testing.assert_allclose(put.payoff(ps, idx), [0.0, 20 * df])
    np.testing.assert_allclose(dig.payoff(ps, idx), [0.0, 5 * df])
    assert "strike 100" in repr(call) and "Put" in repr(put) and "Digital" in repr(dig)
    with pytest.raises(ValueError):
        EuropeanOption(-1.0, 1.0, "call", discount)


def test_variance_swap_payoff(discount: DiscountCurve) -> None:
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    spots = np.array([[100.0, 110.0, 99.0, 104.0, 100.0]])
    ps = _paths(spots, times)
    idx = FixingIndex(times)
    r = np.diff(np.log(spots[0]))
    rv = np.sum(r * r)  # over 1y -> annualised by 1/T = 1
    df = float(discount.df(1.0))
    swap = VarianceSwap(times, strike=0.04, discount=discount, notional=1000.0)
    assert float(swap.payoff(ps, idx)[0]) == pytest.approx(1000.0 * df * (rv - 0.04))
    assert float(swap.floating_leg().payoff(ps, idx)[0]) == pytest.approx(1000.0 * df * rv)
    # market annualisation A/n
    swap252 = VarianceSwap(times, 0.0, discount, annualisation=252.0)
    assert float(swap252.payoff(ps, idx)[0]) == pytest.approx(df * 252.0 / 4 * rv)
    vs = VolSwap(times, strike_vol=0.2, discount=discount)
    assert float(vs.payoff(ps, idx)[0]) == pytest.approx(df * (np.sqrt(rv) - 0.2))
    # forward variance swap over [0.5, 1]: only the last two returns, annualised by 1/0.5
    fwd = VarianceSwap(times[2:], 0.0, discount)
    assert fwd.start == 0.5 and fwd.maturity == 1.0
    assert float(fwd.payoff(ps, idx)[0]) == pytest.approx(df * 2.0 * np.sum(r[2:] ** 2))
    assert "Forward variance swap" in repr(fwd) and "Variance swap" in repr(swap)
    f2 = ForwardVarianceSwap(0.5, 1.0, 0.05, discount)
    assert f2.start == 0.5 and f2.n_returns == 126
    with pytest.raises(ValueError):
        ForwardVarianceSwap(0.0, 1.0, 0.05, discount)
    with pytest.raises(ValueError):
        VarianceSwap([0.0], 0.04, discount)
