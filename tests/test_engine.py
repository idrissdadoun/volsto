"""Engine tests: CRN-exact draws, time grid, BS MC vs analytic, control variates, CRN bumps."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import ndtri

from volsto.config import SimConfig
from volsto.engine import (
    CoarsenedDraws,
    GaussianDraws,
    MonteCarlo,
    PathSet,
    TimeGrid,
    VanillaControl,
)
from volsto.engine.rng import _TWO_M53, STRIDE_BROWNIAN, STRIDE_STEP
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


class _ReferenceDraws(GaussianDraws):
    """``normals`` and ``block`` exactly as they stood before the lean layout (every one of the 8
    generated columns converted to a uniform, ``ndtri`` on the used ones, the antithetic rows
    filled by two copies, a third copy into the block).  The two method bodies are that source,
    unedited; the generator and the position addressing are the library's."""

    def normals(self, step: int, p0: int, p1: int) -> np.ndarray:
        """Normals for one step, paths ``[p0, p1)``: shape ``(p1 - p0, n_brownians)``."""
        if not 0 <= step < self.n_steps:
            raise ValueError("step out of range")
        q0, q1 = self._rng_range(p0, p1)
        n = q1 - q0
        raw = self._raw(step * STRIDE_STEP + q0 * STRIDE_BROWNIAN, n * STRIDE_BROWNIAN)
        u = ((raw >> np.uint64(11)).astype(np.float64) + 0.5) * _TWO_M53
        z = ndtri(u.reshape(n, STRIDE_BROWNIAN)[:, : self.n_brownians])
        z = np.asarray(z, dtype=np.float64)
        if self.antithetic:
            out = np.empty((2 * n, self.n_brownians))
            out[0::2] = z
            out[1::2] = -z
            return out
        return z

    def block(self, step0: int, step1: int, p0: int, p1: int) -> np.ndarray:
        """Normals for steps ``[step0, step1)``: shape ``(p1 - p0, step1 - step0, n_brownians)``."""
        if not 0 <= step0 < step1 <= self.n_steps:
            raise ValueError("step range out of bounds")
        out = np.empty((p1 - p0, step1 - step0, self.n_brownians))
        for j, s in enumerate(range(step0, step1)):
            out[:, j, :] = self.normals(s, p0, p1)
        return out


def _same_bits(a: np.ndarray, b: np.ndarray) -> None:
    """Equal values, equal sign bits (``-0.0`` is not ``0.0`` here), same shape and layout."""
    assert a.shape == b.shape and a.dtype == b.dtype == np.float64
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.signbit(a), np.signbit(b))
    assert a.flags.c_contiguous == b.flags.c_contiguous


@pytest.mark.parametrize("antithetic", [True, False])
@pytest.mark.parametrize("n_brownians", [1, 3, 8])
def test_lean_draws_equal_the_reference_layout(antithetic: bool, n_brownians: int) -> None:
    """The lean layout (only the used columns converted, ``ndtri`` written into the independent
    rows, the antithetic rows their negation) gives the numbers of the earlier layout bit for
    bit: whole blocks, chunked path ranges, multi-step blocks, single steps, rewinds, and
    through a coarsened stream."""
    n_paths, n_steps, seed = 96, 12, 11
    lean = GaussianDraws(seed, n_paths, n_steps, n_brownians, antithetic)
    ref = _ReferenceDraws(seed, n_paths, n_steps, n_brownians, antithetic)
    whole = ref.block(0, n_steps, 0, n_paths)
    _same_bits(lean.block(0, n_steps, 0, n_paths), whole)
    # chunked path ranges (even-aligned, as antithetic pairs require) and multi-step blocks
    for p0, p1 in ((0, 10), (10, 64), (64, 96), (30, 32)):
        for s0, s1 in ((0, 12), (0, 1), (3, 7), (11, 12)):
            _same_bits(lean.block(s0, s1, p0, p1), ref.block(s0, s1, p0, p1))
            np.testing.assert_array_equal(lean.block(s0, s1, p0, p1), whole[p0:p1, s0:s1, :])
    # single steps, read backwards and out of order: the generator rewinds
    for step in (11, 4, 4, 0, 9, 2):
        for p0, p1 in ((0, n_paths), (20, 48)):
            _same_bits(lean.normals(step, p0, p1), ref.normals(step, p0, p1))
    _same_bits(lean.block(2, 5, 0, n_paths), ref.block(2, 5, 0, n_paths))
    # a coarsened stream over each: its own normals and its own block
    for factor in (2, 3):
        c_lean = CoarsenedDraws(
            GaussianDraws(seed, n_paths, n_steps, n_brownians, antithetic), factor
        )
        c_ref = CoarsenedDraws(
            _ReferenceDraws(seed, n_paths, n_steps, n_brownians, antithetic), factor
        )
        _same_bits(
            c_lean.block(0, n_steps // factor, 0, n_paths),
            c_ref.block(0, n_steps // factor, 0, n_paths),
        )
        _same_bits(c_lean.normals(1, 16, 40), c_ref.normals(1, 16, 40))
        np.testing.assert_array_equal(
            c_lean.block(1, 3, 16, 40)[:, 0, :], c_lean.normals(1, 16, 40)
        )
    # the same errors as before
    for bad in (lambda d: d.normals(n_steps, 0, 2), lambda d: d.block(0, n_steps + 1, 0, 2)):
        with pytest.raises(ValueError):
            bad(lean)
    if antithetic:
        with pytest.raises(ValueError, match="even-aligned"):
            lean.block(0, 1, 1, 3)


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
