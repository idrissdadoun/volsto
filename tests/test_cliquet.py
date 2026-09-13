"""M4: cliquet family (SPEC §6, §10): exact path-wise decomposition into forward-start options,
Black–Scholes closed form for the additive cliquet without global floor, reverse cliquet and
Napoleon payoffs, the study structure."""

from __future__ import annotations

import numpy as np
import pytest

from volsto.config import SimConfig
from volsto.engine import FixingIndex, MonteCarlo, PathSet
from volsto.market import DiscountCurve, ForwardCurve, black_price
from volsto.models import BlackScholes
from volsto.products import (
    AccumulatedSumOption,
    AdditiveCliquet,
    CashFlow,
    ForwardStartOption,
    Napoleon,
    ReverseCliquet,
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


def _random_paths(rng: np.random.Generator, n: int, times: np.ndarray) -> PathSet:
    steps = rng.normal(0.0, 0.06, size=(n, times.size - 1))
    spots = 100.0 * np.exp(np.concatenate([np.zeros((n, 1)), np.cumsum(steps, axis=1)], axis=1))
    return _paths(spots, times)


@pytest.mark.parametrize(
    "bounds",
    [
        dict(local_cap=0.02, global_floor=0.0),  # the study structure
        dict(local_floor=-0.03, local_cap=0.02),
        dict(local_floor=-0.05, local_cap=0.05, global_floor=-0.1, global_cap=0.2),
        dict(global_floor=0.0),
        dict(local_floor=-0.02, global_cap=0.15),
        dict(),
    ],
)
def test_additive_cliquet_decomposition_is_exact_path_by_path(
    discount: DiscountCurve, rng: np.random.Generator, bounds: dict[str, float]
) -> None:
    times = uniform_schedule(1.0, 12)
    ps = _random_paths(rng, 500, times)
    idx = FixingIndex(times)
    cliquet = AdditiveCliquet(times, discount, notional=100.0, **bounds)
    parts = cliquet.decompose()
    total = sum(p.payoff(ps, idx) for p in parts)
    np.testing.assert_allclose(total, cliquet.payoff(ps, idx), atol=1e-10)
    # every leg settles on the cliquet's maturity and the strip has one long call per period
    assert all(p.maturity == pytest.approx(1.0) for p in parts)
    longs = [p for p in parts if isinstance(p, ForwardStartOption) and p.notional > 0]
    assert len(longs) == 12
    n_global = int("global_floor" in bounds) + int("global_cap" in bounds)
    assert sum(isinstance(p, AccumulatedSumOption) for p in parts) == n_global
    if not bounds:
        # no bounds: cash −n + Σ (R − 0)⁺ = Σ (R − 1); no cash leg beyond that
        assert isinstance(parts[0], CashFlow) and parts[0].amount == -12.0


def test_additive_cliquet_hand_values_and_validation(discount: DiscountCurve) -> None:
    times = np.array([0.0, 0.5, 1.0])
    ps = _paths(
        np.array([[100.0, 105.0, 101.85], [100.0, 105.0, 106.05], [100.0, 97.0, 100.0]]), times
    )
    idx = FixingIndex(times)
    df = float(discount.df(1.0))
    study = AdditiveCliquet(times, discount, local_cap=0.02, global_floor=0.0)
    # returns: (+5%, −3%) -> 0.02 − 0.03 floored at 0; (+5%, +1%) -> 0.03; (−3%, +3.09%) -> −0.01
    np.testing.assert_allclose(study.payoff(ps, idx), df * np.array([0.0, 0.03, 0.0]), atol=1e-12)
    np.testing.assert_allclose(study.accumulated(ps, idx), [-0.01, 0.03, -0.03 + 0.02], atol=1e-12)
    assert study.has_global and not study.without_global().has_global
    assert "local cap 2%" in repr(study) and "global floor 0%" in repr(study)
    with pytest.raises(ValueError):
        AdditiveCliquet(times, discount, local_floor=0.03, local_cap=0.02)
    with pytest.raises(ValueError):
        AdditiveCliquet(times, discount, global_floor=0.1, global_cap=0.0)
    with pytest.raises(ValueError):
        AdditiveCliquet([1.0], discount)
    s1 = AdditiveCliquet.study(1.0, discount)
    s2 = AdditiveCliquet.study(2.0, discount)
    assert s1.n_periods == 12 and s2.n_periods == 24 and s1.local_cap == 0.02
    assert s1.global_floor == 0.0 and not np.isfinite(s1.local_floor)
    with pytest.raises(ValueError):
        AdditiveCliquet.study(1.3, discount)


def test_additive_cliquet_black_scholes(forward_curve: ForwardCurve, fast_sim: SimConfig) -> None:
    """Flat BS σ: E[clip(r_i, LF, LC)] = LF + C(F_i, 1 + LF) − C(F_i, 1 + LC) per period with
    F_i = F(t_i)/F(t_{i−1}) (no global floor), summed and discounted; the decomposition
    reprices the product on the same paths."""
    sigma = 0.2
    model = BlackScholes(sigma, forward_curve)
    discount = forward_curve.rate_curve
    times = uniform_schedule(1.0, 12)
    f_i = np.asarray(forward_curve.forward(times[1:]) / forward_curve.forward(times[:-1]))
    tau = np.diff(times)
    df = float(discount.df(1.0))
    cases = [
        (AdditiveCliquet(times, discount, local_cap=0.02), (None, 0.02)),
        (AdditiveCliquet(times, discount, local_floor=-0.01, local_cap=0.02), (-0.01, 0.02)),
    ]
    mc = MonteCarlo(fast_sim)
    for cliquet, (lf, lc) in cases:
        cap_call = np.asarray(black_price(f_i, 1.0 + lc, tau, sigma, 1))
        if lf is None:
            per_period = (f_i - 1.0) - cap_call
        else:
            per_period = lf + np.asarray(black_price(f_i, 1.0 + lf, tau, sigma, 1)) - cap_call
        closed = df * float(np.sum(per_period))
        parts = cliquet.decompose()
        res = mc.price_many([cliquet, *parts], model)
        assert abs(res[0].mean - closed) < 3.5 * res[0].stderr, (cliquet, res[0], closed)
        assert sum(r.mean for r in res[1:]) == pytest.approx(res[0].mean, abs=1e-9)


def test_reverse_cliquet_and_napoleon(discount: DiscountCurve, rng: np.random.Generator) -> None:
    times = uniform_schedule(1.0, 4)
    spots = 100.0 * np.cumprod(
        np.array(
            [[1, 0.97, 1.02, 0.95, 1.01], [1, 0.92, 0.95, 1.10, 1.0], [1, 1.1, 1.1, 1.1, 1.1]]
        ),
        axis=1,
    )
    ps = _paths(spots, times)
    idx = FixingIndex(times)
    df = float(discount.df(1.0))
    rev = ReverseCliquet(times, 0.10, discount)
    # coupon 10% − (3% + 5%) = 2%; 10% − (8% + 5%) < 0 -> 0; no negative return -> 10%
    np.testing.assert_allclose(rev.payoff(ps, idx), df * np.array([0.02, 0.0, 0.10]), atol=1e-12)
    rev_lf = ReverseCliquet(times, 0.10, discount, local_floor=-0.04)
    np.testing.assert_allclose(
        rev_lf.payoff(ps, idx), df * np.array([0.03, 0.02, 0.10]), atol=1e-12
    )
    nap = Napoleon(times, 0.08, discount)
    np.testing.assert_allclose(nap.payoff(ps, idx), df * np.array([0.03, 0.0, 0.18]), atol=1e-12)
    nap_c = Napoleon(times, 0.08, discount, local_cap=0.05, global_floor=None)
    np.testing.assert_allclose(nap_c.payoff(ps, idx), df * np.array([0.03, 0.0, 0.13]), atol=1e-12)
    assert nap.decompose() is None
    # reverse cliquet decomposition (coupon + additive cliquet legs) is exact path by path
    big = _random_paths(rng, 400, times)
    for product in (rev, rev_lf, ReverseCliquet(times, 0.05, discount, global_floor=None)):
        parts = product.decompose()
        np.testing.assert_allclose(
            sum(p.payoff(big, idx) for p in parts), product.payoff(big, idx), atol=1e-10
        )
    assert "Reverse cliquet" in repr(rev) and "Napoleon" in repr(nap)
    with pytest.raises(ValueError):
        ReverseCliquet(times, -0.1, discount)
    with pytest.raises(ValueError):
        ReverseCliquet(times, 0.1, discount, local_floor=0.01)
