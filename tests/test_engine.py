"""Engine tests: CRN-exact draws, time grid, BS MC vs analytic, control variates, CRN bumps."""

from __future__ import annotations

import numpy as np
import pytest

from volsto.config import SimConfig
from volsto.engine import GaussianDraws, MonteCarlo, PathSet, TimeGrid, VanillaControl
from volsto.market import DiscountCurve, ForwardCurve, bs_price, bs_vega, norm_cdf
from volsto.models import BlackScholes
from volsto.products import DigitalOption, EuropeanOption


def test_draws_are_crn_exact() -> None:
    a = GaussianDraws(seed=3, n_paths=8, n_steps=5, n_brownians=2, antithetic=False)
    b = GaussianDraws(seed=3, n_paths=16, n_steps=9, n_brownians=3, antithetic=False)
    za = a.block(0, 5, 0, 8)
    zb = b.block(0, 9, 0, 16)
    # same (seed, path, step, brownian) -> same normal regardless of n_paths / n_steps / n_brownians
    np.testing.assert_array_equal(za, zb[:8, :5, :2])
    # independent of chunking
    np.testing.assert_array_equal(za, np.concatenate([a.block(0, 5, 0, 3), a.block(0, 5, 3, 8)]))
    np.testing.assert_array_equal(
        za, np.concatenate([a.block(0, 2, 0, 8), a.block(2, 5, 0, 8)], axis=1)
    )
    # steps / paths carry different numbers, seeds differ
    assert not np.array_equal(za[:, 0, :], za[:, 1, :])
    assert not np.array_equal(GaussianDraws(4, 8, 5, 2, False).block(0, 5, 0, 8), za)
    # re-reading earlier positions (generator rewinds) gives identical numbers
    np.testing.assert_array_equal(a.normals(4, 0, 8), za[:, 4, :])
    np.testing.assert_array_equal(a.normals(1, 2, 6), za[2:6, 1, :])


def test_antithetic_pairs_and_distribution() -> None:
    d = GaussianDraws(seed=1, n_paths=200_000, n_steps=2, n_brownians=1, antithetic=True)
    z = d.normals(1, 0, 200_000)[:, 0]
    np.testing.assert_array_equal(z[0::2], -z[1::2])
    ind = z[0::2]
    assert abs(ind.mean()) < 4 / np.sqrt(ind.size)
    assert abs(ind.var() - 1.0) < 0.02
    assert abs(np.mean(ind**4) - 3.0) < 0.1
    # antithetic path 2i uses rng path i: same as the non-antithetic stream
    plain = GaussianDraws(seed=1, n_paths=100_000, n_steps=2, n_brownians=1, antithetic=False)
    np.testing.assert_array_equal(plain.normals(1, 0, 100_000)[:, 0], ind)


def test_time_grid_build() -> None:
    fixings = [0.5, 1.0, 1.0 + 1e-13, 0.25]
    g = TimeGrid.build(fixings, dt_max=0.1, calibration_grid=[0.33, 2.0])
    assert g.times[0] == 0.0
    assert g.dt_max <= 0.1 + 1e-12
    np.testing.assert_allclose(g.record_times, [0.0, 0.25, 0.5, 1.0])
    assert 0.33 in set(np.round(g.times, 9))
    assert g.horizon == 1.0  # calibration slices beyond the last fixing are dropped
    idx = g.fixing_index
    assert idx[0.5] == 2 and idx[1.0] == 3 and idx[1.0 + 1e-12] == 3
    with pytest.raises(KeyError):
        idx[0.7]
    # step_record points to the right columns
    for col, step in enumerate(g.record_steps):
        if col:
            assert g.step_record[step - 1] == col
    assert np.sum(g.step_record >= 0) == 3
    with pytest.raises(ValueError):
        TimeGrid.build([], 0.1)


def test_bs_mc_matches_analytic(
    forward_curve: ForwardCurve, discount: DiscountCurve, fast_sim: SimConfig
) -> None:
    mc = MonteCarlo(fast_sim)
    model = BlackScholes(0.25, forward_curve)
    call = EuropeanOption(105.0, 1.0, "call", discount)
    put = EuropeanOption(105.0, 1.0, "put", discount)
    dig = DigitalOption(95.0, 1.0, "put", discount, payout=10.0)
    rc, rp, rd = mc.price_many([call, put, dig], model, keep_payoffs=True)
    ac = float(bs_price(100.0, 105.0, 1.0, 0.25, 0.02, 0.01, 1))
    ap = float(bs_price(100.0, 105.0, 1.0, 0.25, 0.02, 0.01, -1))
    F = 100.0 * np.exp(0.01)
    d2 = (np.log(F / 95.0) - 0.5 * 0.25**2) / 0.25
    ad = 10.0 * np.exp(-0.02) * float(norm_cdf(-d2))
    assert abs(rc.mean - ac) < 3 * rc.stderr
    assert abs(rp.mean - ap) < 3 * rp.stderr
    assert abs(rd.mean - ad) < 3 * rd.stderr
    # parity holds path by path (same paths): C - P = df (S_T - K), whose mean is within noise of
    # df (F - K) and whose antithetic-pair stderr is tiny compared with the option's
    assert rc.payoffs is not None and rp.payoffs is not None
    fwd = rc.payoffs - rp.payoffs
    pairs = 0.5 * (fwd[0::2] + fwd[1::2])
    assert rc.mean - rp.mean == pytest.approx(pairs.mean(), abs=1e-9)
    assert abs(pairs.mean() - np.exp(-0.02) * (F - 105.0)) < 3 * pairs.std(ddof=1) / np.sqrt(
        pairs.size
    )
    # same seed -> identical numbers; price() == price_many()
    assert mc.price(call, model).mean == rc.mean
    assert rc.n_samples == fast_sim.n_paths // 2


def test_control_variate(
    forward_curve: ForwardCurve, discount: DiscountCurve, fast_sim: SimConfig
) -> None:
    mc = MonteCarlo(fast_sim)
    model = BlackScholes(0.25, forward_curve)
    target = EuropeanOption(110.0, 1.0, "call", discount)
    ctrl_opt = EuropeanOption(100.0, 1.0, "call", discount)
    ctrl = VanillaControl(ctrl_opt, float(bs_price(100.0, 100.0, 1.0, 0.25, 0.02, 0.01, 1)))
    raw = mc.price(target, model)
    cv = mc.price(target, model, controls=[ctrl])
    analytic = float(bs_price(100.0, 110.0, 1.0, 0.25, 0.02, 0.01, 1))
    assert cv.cv is not None and cv.cv.variance_reduction > 3
    assert cv.stderr < raw.stderr / 1.7
    assert abs(cv.mean - analytic) < 3 * cv.stderr
    assert cv.cv.raw_mean == pytest.approx(raw.mean)


def test_crn_bump_vega(
    forward_curve: ForwardCurve, discount: DiscountCurve, fast_sim: SimConfig
) -> None:
    """Common random numbers: the stderr of a bumped difference is far below that of the levels."""
    mc = MonteCarlo(fast_sim)
    opt = EuropeanOption(100.0, 1.0, "call", discount)
    eps = 1e-3
    base = BlackScholes(0.2, forward_curve)
    up = mc.price(opt, base.bump(vol=0.2 + eps), keep_payoffs=True)
    dn = mc.price(opt, base.bump(vol=0.2 - eps), keep_payoffs=True)
    assert up.payoffs is not None and dn.payoffs is not None
    diff = (up.payoffs - dn.payoffs) / (2 * eps)
    pairs = 0.5 * (diff[0::2] + diff[1::2])
    vega_mc, vega_err = pairs.mean(), pairs.std(ddof=1) / np.sqrt(pairs.size)
    vega = float(bs_vega(100.0, 100.0, 1.0, 0.2, 0.02, 0.01))
    assert abs(vega_mc - vega) < 3 * vega_err
    assert vega_err < 0.05 * up.stderr / (2 * eps)


def test_pathset_accessors() -> None:
    times = np.array([0.0, 0.5, 1.0])
    ls = np.log(np.array([[100.0, 110.0, 99.0], [100.0, 90.0, 95.0]]))
    ps = PathSet(
        times, ls, np.zeros((2, 3)), np.zeros((2, 3, 0)), np.zeros((2, 3)), np.zeros((2, 3))
    )
    np.testing.assert_allclose(ps.spot_at(2), [99.0, 95.0])
    np.testing.assert_allclose(ps.log_return(0, 1), np.log([1.1, 0.9]))
    rv = ps.realised_variance_fixings([0, 1, 2])
    np.testing.assert_allclose(
        rv, [np.log(1.1) ** 2 + np.log(0.9) ** 2, np.log(0.9) ** 2 + np.log(95 / 90) ** 2]
    )
    with pytest.raises(IndexError):
        ps.spot_at(1, asset=1)
    assert PathSet.concat([ps, ps]).n_paths == 4
