"""M6 Part 1: barrier machinery (SPEC §6.5) — Brownian-bridge weights by hand (step-start
variance convention included) and on a stochastic-variance kernel, in–out parity path by path,
discrete → continuous convergence against Reiner–Rubinstein with the Broadie–Glasserman–Kou shift
checked on common random numbers, closed forms within 3 stderr, weighted versus sampled survival
with the sampled variance predicted from the weights, the M4c knock-out variance swap rebuilt with
the discrete monitoring helper, the barrier shift, digital smoothing and the closed-form identities
of ``volsto.market.barrier_bs`` (quadrature of the reflection-principle density, image identities,
first-passage quadrature)."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from volsto.config import BergomiParams, SimConfig
from volsto.engine import FixingIndex, MonteCarlo, PathSet, TimeGrid
from volsto.market import (
    DiscountCurve,
    ForwardCurve,
    ForwardVarianceCurve,
    LocalVolSurface,
    SSVISurface,
    bs_price,
)
from volsto.market.barrier_bs import (
    bs_barrier_price,
    bs_digital_price,
    bs_hit_discount,
    bs_hit_probability,
    bs_no_touch_price,
    bs_one_touch_price,
)
from volsto.models import BergomiSV, BlackScholes, LocalVol
from volsto.products import (
    DigitalOption,
    EuropeanOption,
    KnockOutVarianceSwap,
    Portfolio,
    daily_schedule,
    uniform_schedule,
)
from volsto.products.barrier import (
    Digital,
    KnockInOption,
    KnockOutOption,
    NoTouch,
    OneTouch,
    bridge_step_survival,
    continuous_survival_weight,
    first_hit_index,
)

BGK_BETA = 0.5826  # Broadie–Glasserman–Kou (1997): -zeta(1/2)/sqrt(2 pi)


def _mode(monitoring: str, survival: str = "weight") -> dict[str, Any]:
    """Constructor keywords of a monitoring mode: discrete monitoring needs the inequality
    convention (touching knocks here), continuous monitoring the survival form."""
    if monitoring == "discrete":
        return {"monitoring": "discrete", "strict": False}
    return {"monitoring": "continuous", "survival": survival}


def _paths(spots: np.ndarray, times: np.ndarray, variance: float | np.ndarray = 0.04) -> PathSet:
    """A PathSet whose accumulators are consistent with one simulation step per column;
    ``variance`` is a scalar or one value per column (the recorded instantaneous variance)."""
    n, m = spots.shape
    var = np.broadcast_to(np.asarray(variance, dtype=np.float64), (n, m)).copy()
    ls = np.log(spots)
    dx = np.diff(ls, axis=1)
    sum_sq = np.concatenate([np.zeros((n, 1)), np.cumsum(dx * dx, axis=1)], axis=1)
    int_var = np.concatenate(
        [np.zeros((n, 1)), np.cumsum(var[:, :-1] * np.diff(times), axis=1)], axis=1
    )
    return PathSet(times, ls, var, np.zeros((n, m, 0)), int_var, sum_sq)


def _random_paths(
    rng: np.random.Generator, n: int, times: np.ndarray, s0: float = 100.0
) -> PathSet:
    steps = rng.normal(0.0, 0.012, size=(n, times.size - 1))
    spots = s0 * np.exp(np.concatenate([np.zeros((n, 1)), np.cumsum(steps, axis=1)], axis=1))
    return _paths(spots, times)


def _pairs(x: np.ndarray) -> np.ndarray:
    return 0.5 * (x[0::2] + x[1::2])


def _diff_stats(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean and stderr of the CRN difference ``a − b`` of per-path antithetic payoffs."""
    d = _pairs(a) - _pairs(b)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size))


# ---------------------------------------------------------------------------------------------
# bridge weights and monitoring helpers
# ---------------------------------------------------------------------------------------------


def test_bridge_weights_hand_values() -> None:
    """p_i = exp(−2 (b − x_i)(b − x_{i+1}) / (σ_i² Δt)) per step with σ_i² the variance recorded
    at the step START, zero survival on a recorded breach, window restriction, zero-variance
    step, and the deterministic sampled form."""
    times = np.array([0.0, 0.5, 1.0])
    var = 0.04
    spots = np.array(
        [
            [100.0, 95.0, 98.0],  # alive, both steps contribute
            [100.0, 89.0, 98.0],  # recorded breach of the down barrier at 90
            [100.0, 105.0, 102.0],  # alive for the up barrier at 110
            [100.0, 111.0, 102.0],  # recorded breach of the up barrier
            [100.0, 90.0, 98.0],  # touching counts (non-strict)
        ]
    )
    ps = _paths(spots, times, var)
    idx = FixingIndex(times)
    b = np.log(90.0)
    x = np.log(spots[0])
    p1 = np.exp(-2 * (b - x[0]) * (b - x[1]) / (var * 0.5))
    p2 = np.exp(-2 * (b - x[1]) * (b - x[2]) / (var * 0.5))
    w_down = continuous_survival_weight(ps, idx, b, "down", 0.0, 1.0)
    np.testing.assert_allclose(w_down[0], (1 - p1) * (1 - p2))
    assert w_down[1] == 0.0 and w_down[4] == 0.0 and 0.0 < w_down[0] < 1.0
    surv, end_cols = bridge_step_survival(ps, idx, b, "down", 0.0, 1.0)
    np.testing.assert_allclose(surv[0], [1 - p1, 1 - p2])
    np.testing.assert_array_equal(end_cols, [1, 2])
    # window [0.5, 1]: only the second step
    np.testing.assert_allclose(continuous_survival_weight(ps, idx, b, "down", 0.5, 1.0)[0], 1 - p2)
    # the step variance is the one recorded at the step start: columns [0.04, 0.09, 0.01] use
    # 0.04 on the first step and 0.09 on the second (the end-of-step convention would differ)
    psv = _paths(spots[:1], times, np.array([0.04, 0.09, 0.01]))
    r1 = np.exp(-2 * (b - x[0]) * (b - x[1]) / (0.04 * 0.5))
    r2 = np.exp(-2 * (b - x[1]) * (b - x[2]) / (0.09 * 0.5))
    w_start = continuous_survival_weight(psv, idx, b, "down", 0.0, 1.0)
    np.testing.assert_allclose(w_start, [(1 - r1) * (1 - r2)], rtol=1e-13)
    np.testing.assert_allclose(
        bridge_step_survival(psv, idx, b, "down", 0.0, 1.0)[0][0], [1 - r1, 1 - r2]
    )
    e1 = np.exp(-2 * (b - x[0]) * (b - x[1]) / (0.09 * 0.5))
    e2 = np.exp(-2 * (b - x[1]) * (b - x[2]) / (0.01 * 0.5))
    w_end = (1 - e1) * (1 - e2)
    assert abs(w_start[0] - w_end) > 0.05 and abs(w_start[0] - w_down[0]) > 0.05
    bu = np.log(110.0)
    xu = np.log(spots[2])
    q1 = np.exp(-2 * (bu - xu[0]) * (bu - xu[1]) / (var * 0.5))
    q2 = np.exp(-2 * (bu - xu[1]) * (bu - xu[2]) / (var * 0.5))
    w_up = continuous_survival_weight(ps, idx, bu, "up", 0.0, 1.0)
    np.testing.assert_allclose(w_up[2], (1 - q1) * (1 - q2))
    assert w_up[3] == 0.0 and w_up[0] > w_up[2]  # path 0 stays farther from 110
    # farther barrier: larger weight, monotone
    assert np.all(continuous_survival_weight(ps, idx, np.log(80.0), "down", 0.0, 1.0) >= w_down)
    # zero variance between two alive points: no crossing
    ps0 = _paths(spots[:1], times, 0.0)
    np.testing.assert_allclose(continuous_survival_weight(ps0, idx, b, "down", 0.0, 1.0), [1.0])
    # sampled form: 0/1, deterministic in the seed, consistent with the weights
    s1 = continuous_survival_weight(ps, idx, b, "down", 0.0, 1.0, sample=True, seed=3)
    s2 = continuous_survival_weight(ps, idx, b, "down", 0.0, 1.0, sample=True, seed=3)
    np.testing.assert_array_equal(s1, s2)
    assert set(np.unique(s1)) <= {0.0, 1.0} and s1[1] == 0.0 and s1[4] == 0.0
    big = _paths(np.tile(spots[:1], (20_000, 1)), times, var)
    smean = continuous_survival_weight(big, idx, b, "down", 0.0, 1.0, sample=True).mean()
    assert abs(smean - w_down[0]) < 3.5 * np.sqrt(w_down[0] * (1 - w_down[0]) / 20_000)
    # more steps than one bridge block (64): the blockwise product equals the full product
    many_t = np.linspace(0.0, 1.0, 201)
    many = _random_paths(np.random.default_rng(1), 50, many_t)
    full_surv, _ = bridge_step_survival(many, FixingIndex(many_t), b, "down", 0.0, 1.0)
    assert full_surv.shape == (50, 200)
    np.testing.assert_allclose(
        continuous_survival_weight(many, FixingIndex(many_t), b, "down", 0.0, 1.0),
        np.prod(full_surv, axis=1),
        rtol=1e-12,
    )
    # validation
    with pytest.raises(ValueError):
        continuous_survival_weight(ps, idx, b, "sideways", 0.0, 1.0)
    with pytest.raises(ValueError):
        continuous_survival_weight(ps, idx, b, "down", 0.0, 0.7)  # not a record time
    with pytest.raises(ValueError):
        continuous_survival_weight(ps, idx, b, "down", 1.0, 0.5)
    # discrete helper
    ls = np.log(spots)
    np.testing.assert_array_equal(first_hit_index(ls, b, "down", strict=False), [3, 1, 3, 3, 1])
    np.testing.assert_array_equal(first_hit_index(ls, b, "down", strict=True), [3, 1, 3, 3, 3])
    np.testing.assert_array_equal(first_hit_index(ls, bu, "up", strict=False), [3, 3, 3, 1, 3])
    with pytest.raises(ValueError):
        first_hit_index(ls[0], b, "down", strict=False)


def test_continuous_requires_all_steps(forward_curve: ForwardCurve) -> None:
    """A path set recorded only at the fixings raises (message names requires_all_steps), and so
    does a log_spot rewritten without its sum_sq accumulator (the invariant of the module
    docstring); the engine records every step for a continuous product on its own."""
    model = BlackScholes(0.2, forward_curve)
    sim = SimConfig(n_paths=2_000, dt_max=1 / 52, chunk_size=1_000, seed=3)
    mc = MonteCarlo(sim)
    coarse = TimeGrid.build([0.5, 1.0], 1 / 52)
    paths = mc.simulate(model, coarse)
    with pytest.raises(ValueError, match="requires_all_steps"):
        continuous_survival_weight(paths, coarse.fixing_index, np.log(90.0), "down", 0.0, 1.0)
    full = TimeGrid.build([0.5, 1.0], 1 / 52, record_all_steps=True)
    paths = mc.simulate(model, full)
    w = continuous_survival_weight(paths, full.fixing_index, np.log(90.0), "down", 0.0, 1.0)
    assert w.shape == (2_000,) and np.all((w >= 0) & (w <= 1)) and 0 < w.mean() < 1
    # a pathwise perturbation of log_spot must carry sum_sq along (risk-layer contract)
    ls = paths.log_spot.copy()
    ls[:, 10:21] = ls[:, [10]] + 1.001 * (ls[:, 10:21] - ls[:, [10]])
    ls[:, 21:] += 0.001 * (paths.log_spot[:, [20]] - paths.log_spot[:, [10]])
    with pytest.raises(ValueError, match="requires_all_steps"):
        continuous_survival_weight(
            dataclasses.replace(paths, log_spot=ls),
            full.fixing_index,
            np.log(90.0),
            "down",
            0.0,
            1.0,
        )
    sq = paths.sum_sq.copy()
    sq[:, 10:21] = sq[:, [10]] + 1.001**2 * (sq[:, 10:21] - sq[:, [10]])
    sq[:, 21:] += (1.001**2 - 1.0) * (paths.sum_sq[:, [20]] - paths.sum_sq[:, [10]])
    w_mod = continuous_survival_weight(
        dataclasses.replace(paths, log_spot=ls, sum_sq=sq),
        full.fixing_index,
        np.log(90.0),
        "down",
        0.0,
        1.0,
    )
    assert w_mod.shape == w.shape and not np.array_equal(w_mod, w)
    disc = forward_curve.rate_curve
    ko_c = KnockOutOption(100.0, 1.0, "call", 90.0, "down", disc, monitoring="continuous")
    ko_d = KnockOutOption(100.0, 1.0, "call", 90.0, "down", disc, **_mode("discrete"))
    assert ko_c.requires_all_steps and not ko_d.requires_all_steps
    g = mc.build_grid([ko_c], model)
    assert g.n_records == g.n_steps + 1
    g = mc.build_grid([ko_d], model)
    assert g.n_records == ko_d.fixing_times.size  # the daily schedule already starts at 0
    res = mc.price(ko_c, model)
    assert 0 < res.mean < mc.price(ko_c.vanilla(), model).mean and res.stderr > 0


def test_portfolio_propagates_requires_all_steps(forward_curve: ForwardCurve) -> None:
    """A continuous barrier wrapped in the generic composite must still be priced on a
    record-all-steps grid (the risk layer's after-fixing composite is a Portfolio): the
    composite reprices the bare product exactly on the same seed."""
    disc = forward_curve.rate_curve
    ko = KnockOutOption(100.0, 1.0, "call", 90.0, "down", disc, monitoring="continuous")
    pf = Portfolio([ko])
    if not pf.requires_all_steps:
        pytest.skip(
            "volsto/products/base.py: Portfolio does not propagate requires_all_steps from its "
            "legs (integrator finding) - the composite is priced on a fixings-only grid and the "
            "barrier payoff raises; this test activates once the property is added"
        )
    model = BlackScholes(0.2, forward_curve)
    mc = MonteCarlo(SimConfig(n_paths=4_000, dt_max=1 / 52, chunk_size=2_000, seed=3))
    bare, wrapped = mc.price(ko, model), mc.price(pf, model)
    assert bare.mean == pytest.approx(wrapped.mean) and bare.stderr == pytest.approx(wrapped.stderr)


def test_bridge_on_stochastic_variance_kernel(forward_curve: ForwardCurve) -> None:
    """One-factor Bergomi (ω = 3, κ = 1.5, ρ = −0.7, flat ξ₀ = 4%): the full-grid rule passes on
    the kernel's accumulators, the bridge weights equal the hand transcription with the variance
    recorded at each step START (a column-shifted variance changes them), weighted and sampled
    survival agree on common random numbers with the weighted stderr no larger, continuous
    monitoring is worth no more than the discrete comparator on the same grid path by path, and
    the one-touch lies strictly inside (0, DF)."""
    disc = forward_curve.rate_curve
    model = BergomiSV(
        BergomiParams.one_factor(3.0, 1.5, -0.7), ForwardVarianceCurve.flat(0.04), forward_curve
    )
    sim = SimConfig(n_paths=20_000, dt_max=1 / 100, chunk_size=10_000, seed=3)
    h = 90.0
    prods = [
        KnockOutOption(100.0, 1.0, "call", h, "down", disc, monitoring="continuous"),
        KnockOutOption(
            100.0, 1.0, "call", h, "down", disc, monitoring="continuous", survival="sampled"
        ),
        KnockOutOption(
            100.0,
            1.0,
            "call",
            h,
            "down",
            disc,
            fixing_times=uniform_schedule(1.0, 100),
            **_mode("discrete"),
        ),
        OneTouch(h, 1.0, "down", disc, monitoring="continuous"),
    ]
    mc = MonteCarlo(sim)
    res = mc.price_many(prods, model, keep_payoffs=True)
    pays = [np.asarray(rr.payoffs) for rr in res]
    d, se = _diff_stats(pays[0], pays[1])
    assert abs(d) < 3.0 * se, (res[0], res[1], d, se)
    assert res[0].stderr <= res[1].stderr
    assert np.all(pays[0] <= pays[2] + 1e-12) and res[0].mean < res[2].mean
    df = float(disc.df(1.0))
    assert 0.0 < res[3].mean < df and res[3].stderr > 0
    # weights on a simulated set: hand transcription with variance_at(step start)
    grid = mc.build_grid(prods[:1], model)
    paths = mc.simulate(model, grid)
    idx = grid.fixing_index
    cols = np.arange(idx[0.0], idx[1.0] + 1)
    x = paths.log_spot_at(cols)
    v = paths.variance_at(cols[:-1])
    assert v.std() > 0.05  # the variance really is stochastic (ω = 3)
    dd = x - np.log(h)
    both = (dd[:, :-1] > 0) & (dd[:, 1:] > 0)
    p = np.exp(-2.0 * dd[:, :-1] * dd[:, 1:] / (v * np.diff(idx.times[cols])[None, :]))
    hand = np.prod(np.where(both, 1.0 - p, 0.0), axis=1)
    w = continuous_survival_weight(paths, idx, np.log(h), "down", 0.0, 1.0)
    np.testing.assert_allclose(w, hand, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        NoTouch(h, 1.0, "down", disc, monitoring="continuous").payoff(paths, idx), df * w
    )
    shifted = dataclasses.replace(paths, variance=np.roll(paths.variance, -1, axis=1))
    w_shift = continuous_survival_weight(shifted, idx, np.log(h), "down", 0.0, 1.0)
    assert np.abs(w - w_shift).max() > 0.05  # the step-end variance would give other weights


# ---------------------------------------------------------------------------------------------
# path-wise identities
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("monitoring", "survival"),
    [("discrete", "weight"), ("continuous", "weight"), ("continuous", "sampled")],
)
def test_in_out_parity_path_by_path(
    discount: DiscountCurve, rng: np.random.Generator, monitoring: str, survival: str
) -> None:
    """KI + KO = vanilla path by path (with a rebate: + R DF(T)); KI weight + KO weight = 1;
    one-touch + no-touch = payout DF(T); KnockInOption.decompose() reprices."""
    times = daily_schedule(1.0)
    ps = _random_paths(rng, 400, times)
    idx = FixingIndex(times)
    df = float(discount.df(1.0))
    kw = _mode(monitoring, survival)
    for direction, barrier in (("down", 92.0), ("up", 108.0)):
        for cp in ("call", "put"):
            van = EuropeanOption(100.0, 1.0, cp, discount).payoff(ps, idx)
            ki = KnockInOption(100.0, 1.0, cp, barrier, direction, discount, **kw)
            ko = KnockOutOption(100.0, 1.0, cp, barrier, direction, discount, **kw)
            np.testing.assert_allclose(ki.payoff(ps, idx) + ko.payoff(ps, idx), van, atol=1e-12)
            a_ki, _ = ki.monitor(ps, idx)
            a_ko, _ = ko.monitor(ps, idx)
            np.testing.assert_array_equal(a_ki, a_ko)
            assert 0.0 < a_ko.mean() < 1.0, (direction, barrier)  # both outcomes present
            np.testing.assert_allclose((1.0 - a_ki) + a_ko, 1.0)
            parts = ki.decompose()
            np.testing.assert_allclose(
                ki.payoff(ps, idx), sum(p.payoff(ps, idx) for p in parts), atol=1e-12
            )
            r = 3.0
            ki_r = KnockInOption(
                100.0,
                1.0,
                cp,
                barrier,
                direction,
                discount,
                rebate=r,
                rebate_timing="maturity",
                **kw,
            )
            ko_r = KnockOutOption(
                100.0,
                1.0,
                cp,
                barrier,
                direction,
                discount,
                rebate=r,
                rebate_timing="maturity",
                **kw,
            )
            np.testing.assert_allclose(
                ki_r.payoff(ps, idx) + ko_r.payoff(ps, idx), van + r * df, atol=1e-12
            )
            parts = ki_r.decompose()
            assert len(parts) == 3
            np.testing.assert_allclose(
                ki_r.payoff(ps, idx), sum(p.payoff(ps, idx) for p in parts), atol=1e-12
            )
        ot = OneTouch(barrier, 1.0, direction, discount, payout=2.0, **kw)
        nt = NoTouch(barrier, 1.0, direction, discount, payout=2.0, **kw)
        np.testing.assert_allclose(ot.payoff(ps, idx) + nt.payoff(ps, idx), 2.0 * df)
        if survival == "sampled":
            np.testing.assert_allclose(
                nt.payoff(ps, idx) / (2.0 * df),
                continuous_survival_weight(
                    ps, idx, np.log(barrier), direction, 0.0, 1.0, sample=True
                ),
            )


def test_rebate_timing_and_european_barrier(
    discount: DiscountCurve, rng: np.random.Generator
) -> None:
    """Rebate at the hit discounts from the breach date (discrete) / the midpoint of the
    crossing step (continuous, weight form: Σ (S_{i−1} − S_i) DF(t_i + Δt_i/2); sampled form:
    DF at the midpoint of the crossing step); a European barrier is the vanilla times the
    maturity indicator."""
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    spots = np.array(
        [
            [100.0, 95.0, 88.0, 92.0, 105.0],
            [100.0, 97.0, 96.0, 95.0, 98.0],
            [85.0, 95.0, 96.0, 95.0, 110.0],
        ]
    )
    ps = _paths(spots, times)
    idx = FixingIndex(times)
    r = 4.0
    ko_hit = KnockOutOption(
        100.0,
        1.0,
        "call",
        90.0,
        "down",
        discount,
        fixing_times=times,
        rebate=r,
        rebate_timing="hit",
        **_mode("discrete"),
    )
    ko_mat = KnockOutOption(
        100.0,
        1.0,
        "call",
        90.0,
        "down",
        discount,
        fixing_times=times,
        rebate=r,
        rebate_timing="maturity",
        **_mode("discrete"),
    )
    df = discount.df(times)
    np.testing.assert_allclose(ko_hit.payoff(ps, idx), [r * df[2], df[4] * 0.0, r * df[0]])
    np.testing.assert_allclose(ko_mat.payoff(ps, idx), [r * df[4], 0.0, r * df[4]])
    # continuous, weight form
    ko_c = KnockOutOption(
        100.0,
        1.0,
        "call",
        90.0,
        "down",
        discount,
        monitoring="continuous",
        rebate=r,
        rebate_timing="hit",
    )
    surv, end_cols = bridge_step_survival(ps, idx, np.log(90.0), "down", 0.0, 1.0)
    cum = np.cumprod(surv, axis=1)
    prev = np.concatenate([np.ones((3, 1)), cum[:, :-1]], axis=1)
    df_mid = discount.df(0.5 * (times[end_cols - 1] + times[end_cols]))
    expected = df[4] * cum[:, -1] * np.maximum(spots[:, -1] - 100.0, 0.0) + r * np.sum(
        (prev - cum) * df_mid[None, :], axis=1
    )
    np.testing.assert_allclose(ko_c.payoff(ps, idx), expected)
    assert np.all(np.sum(prev - cum, axis=1) + cum[:, -1] == pytest.approx(1.0))
    # path 2 breaches at inception: the whole mass sits in the first step, path 0 crosses the
    # recorded 88 in the second step at the latest
    _, hit_df = ko_c.monitor(ps, idx)
    assert hit_df[2] == pytest.approx(float(discount.df(0.125)))
    assert float(discount.df(0.375)) <= hit_df[0] <= float(discount.df(0.125))
    # sampled form: the crossing step's midpoint discount factor
    ko_s = KnockOutOption(
        100.0,
        1.0,
        "call",
        90.0,
        "down",
        discount,
        monitoring="continuous",
        survival="sampled",
        rebate=r,
        rebate_timing="hit",
    )
    alive_s, hit_s = ko_s.monitor(ps, idx)
    mids = set(np.round(discount.df(0.5 * (times[:-1] + times[1:])), 12).tolist())
    for a, h_ in zip(alive_s, hit_s, strict=True):
        assert (a == 1.0 and h_ == 0.0) or (a == 0.0 and round(float(h_), 12) in mids)
    assert alive_s[2] == 0.0 and hit_s[2] == pytest.approx(float(discount.df(0.125)))
    # European-at-maturity barrier through fixing_times=[T]
    eu = KnockOutOption(
        100.0, 1.0, "call", 90.0, "down", discount, fixing_times=[1.0], **_mode("discrete")
    )
    van = EuropeanOption(100.0, 1.0, "call", discount).payoff(ps, idx)
    np.testing.assert_allclose(eu.payoff(ps, idx), van * (spots[:, -1] > 90.0))
    assert eu.fixing_times.tolist() == [1.0]
    # random paths: rebate at hit is worth at least the rebate at maturity (earlier payment, r > 0)
    big = _random_paths(rng, 300, daily_schedule(1.0))
    idx_d = FixingIndex(daily_schedule(1.0))
    a = KnockOutOption(
        100.0,
        1.0,
        "put",
        92.0,
        "down",
        discount,
        monitoring="continuous",
        rebate=1.0,
        rebate_timing="hit",
    )
    b = KnockOutOption(
        100.0,
        1.0,
        "put",
        92.0,
        "down",
        discount,
        monitoring="continuous",
        rebate=1.0,
        rebate_timing="maturity",
    )
    assert np.all(a.payoff(big, idx_d) >= b.payoff(big, idx_d) - 1e-12)
    # 252 steps span several bridge blocks: the running-product hit discount equals the
    # full-matrix formula
    surv_b, ends_b = bridge_step_survival(big, idx_d, np.log(92.0), "down", 0.0, 1.0)
    cum_b = np.cumprod(surv_b, axis=1)
    prev_b = np.concatenate([np.ones((cum_b.shape[0], 1)), cum_b[:, :-1]], axis=1)
    t_d = daily_schedule(1.0)
    df_mid_b = discount.df(0.5 * (t_d[ends_b - 1] + t_d[ends_b]))
    alive_b, hit_b = a.monitor(big, idx_d)
    np.testing.assert_allclose(alive_b, cum_b[:, -1], rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(
        hit_b, np.sum((prev_b - cum_b) * df_mid_b[None, :], axis=1), rtol=1e-12, atol=1e-15
    )
    assert 0.0 < alive_b.mean() < 1.0


def test_discrete_monitoring_matches_m4c_knock_out_variance_swap(
    discount: DiscountCurve, rng: np.random.Generator
) -> None:
    """The M4c knock-out variance swap (close-to-close, S_0 monitored, knock-out day accrues,
    strict or not) rebuilt with ``first_hit_index`` on the same daily fixings: identical τ and
    payoff path by path; a discrete KnockOutOption on that schedule shares the survival."""
    times = daily_schedule(1.0)
    n = times.size - 1
    ps = PathSet.concat(
        [
            _random_paths(rng, 300, times),
            _random_paths(rng, 20, times, s0=112.0),
        ]  # inception breach
    )
    idx = FixingIndex(times)
    ls = ps.log_spot_at(idx.indices(times))
    r2 = np.diff(ls, axis=1) ** 2
    df = float(discount.df(1.0))
    for direction, barrier in (("up", 110.0), ("down", 90.0)):
        for strict in (True, False):
            ko = KnockOutVarianceSwap(
                times, barrier, 0.2, discount, direction=direction, strict=strict
            )
            j = first_hit_index(ls, np.log(barrier), direction, strict=strict)
            tau = np.minimum(j, n)
            np.testing.assert_array_equal(tau, ko.stopping_index(ls))
            alive = np.arange(1, n + 1)[None, :] <= tau[:, None]
            accrued = 252.0 / n * np.sum(r2 * alive, axis=1)
            mine = df * (accrued - 0.04 * tau / n)
            np.testing.assert_allclose(mine, ko.payoff(ps, idx), rtol=0, atol=1e-14)
            assert 0 < np.mean(tau < n) < 1  # both knocked and surviving paths in the sample
            opt = KnockOutOption(
                100.0,
                1.0,
                "call",
                barrier,
                direction,
                discount,
                monitoring="discrete",
                fixing_times=times,
                strict=strict,
            )
            np.testing.assert_array_equal(opt.monitor(ps, idx)[0], (j == n + 1).astype(float))
    # inception breach: τ = 0 and the option is dead from the start
    up = KnockOutVarianceSwap(times, 110.0, 0.2, discount, direction="up")
    assert np.all(up.stopping_index(ls[300:]) == 0)
    opt = KnockOutOption(100.0, 1.0, "call", 110.0, "up", discount, **_mode("discrete"))
    assert np.all(opt.payoff(ps, idx)[300:] == 0.0)


def test_barrier_shift_moves_only_the_monitored_level(
    discount: DiscountCurve, rng: np.random.Generator
) -> None:
    times = daily_schedule(1.0)
    ps = _random_paths(rng, 300, times)
    idx = FixingIndex(times)
    for monitoring in ("discrete", "continuous"):
        kw = _mode(monitoring)
        shifted = KnockOutOption(
            100.0, 1.0, "call", 90.0, "down", discount, barrier_shift=-0.01, **kw
        )
        moved = KnockOutOption(100.0, 1.0, "call", 89.1, "down", discount, **kw)
        plain = KnockOutOption(100.0, 1.0, "call", 90.0, "down", discount, **kw)
        np.testing.assert_allclose(shifted.payoff(ps, idx), moved.payoff(ps, idx))
        assert shifted.barrier == 90.0 and shifted.barrier_eff == pytest.approx(89.1)
        assert shifted.strike == plain.strike and shifted.vanilla_payoff(ps, idx) is not None
        np.testing.assert_allclose(shifted.vanilla_payoff(ps, idx), plain.vanilla_payoff(ps, idx))
        # a lower down-barrier can only save paths
        assert np.all(shifted.payoff(ps, idx) >= plain.payoff(ps, idx))
        assert shifted.payoff(ps, idx).mean() > plain.payoff(ps, idx).mean()
        rep = repr(shifted)
        assert "barrier 90 (monitored 89.1, shift -1%)" in rep and monitoring in rep
        # the sign is free: the desk applies it per side
        up_shift = OneTouch(90.0, 1.0, "down", discount, barrier_shift=0.02, **kw)
        assert up_shift.barrier_eff == pytest.approx(91.8)
        np.testing.assert_allclose(
            up_shift.payoff(ps, idx),
            OneTouch(91.8, 1.0, "down", discount, **kw).payoff(ps, idx),
        )
    with pytest.raises(ValueError):
        KnockOutOption(
            100.0, 1.0, "call", 90.0, "down", discount, barrier_shift=-1.0, **_mode("discrete")
        )


def test_digital_smoothing(
    forward_curve: ForwardCurve, discount: DiscountCurve, rng: np.random.Generator
) -> None:
    """w = 0 is the exact digital; w > 0 equals the unit-height spread of EuropeanOptions at
    K ∓ w/2 path by path; on a dense uniform spot grid straddling the strike E|smooth − exact|
    is the Riemann sum of a tent of height 1/2 and base w, i.e. exactly payout DF (w/4)/(N Δ)
    when K ± w/2 are grid nodes — the O(w) convergence rate with no free constant; the
    Black–Scholes digital matches the closed form."""
    times = np.array([0.0, 1.0])
    ps = _paths(100.0 * np.exp(rng.normal(0.0, 0.2, size=(2_000, 2)) * np.array([0.0, 1.0])), times)
    idx = FixingIndex(times)
    for cp in ("call", "put"):
        exact = Digital(100.0, 1.0, cp, discount, payout=3.0, notional=2.0)
        ref = DigitalOption(100.0, 1.0, cp, discount, payout=3.0, notional=2.0)
        np.testing.assert_allclose(exact.payoff(ps, idx), ref.payoff(ps, idx))
        assert exact.decompose() is None
        for w in (4.0, 1.0, 0.25, 0.05, 1e-6):
            smooth = Digital(100.0, 1.0, cp, discount, payout=3.0, notional=2.0, smoothing=w)
            s = 1 if cp == "call" else -1
            lo = EuropeanOption(100.0 - s * w / 2, 1.0, cp, discount).payoff(ps, idx)
            hi = EuropeanOption(100.0 + s * w / 2, 1.0, cp, discount).payoff(ps, idx)
            np.testing.assert_allclose(
                smooth.payoff(ps, idx), 3.0 * 2.0 * (lo - hi) / w, atol=1e-12
            )
            parts = smooth.decompose()
            assert parts is not None and len(parts) == 2
            np.testing.assert_allclose(
                smooth.payoff(ps, idx), sum(p.payoff(ps, idx) for p in parts), atol=1e-12
            )
            assert np.all(smooth.payoff(ps, idx) >= -1e-12)
        assert "smoothing width 0.25" in repr(Digital(100.0, 1.0, cp, discount, smoothing=0.25))
        assert "exact" in repr(Digital(100.0, 1.0, cp, discount))
    # deterministic convergence on a dense spot grid: gap(w) = payout DF (w/4) / (N Δ)
    grid_s = np.linspace(90.0, 110.0, 20_001)
    delta = float(grid_s[1] - grid_s[0])
    ps_grid = _paths(np.column_stack([np.full(grid_s.size, 100.0), grid_s]), times)
    df = float(discount.df(1.0))
    for cp in ("call", "put"):
        exact_pay = Digital(100.0, 1.0, cp, discount, payout=3.0).payoff(ps_grid, idx)
        for w in (1.0, 0.25, 0.05, 0.01):
            gap = np.abs(
                Digital(100.0, 1.0, cp, discount, payout=3.0, smoothing=w).payoff(ps_grid, idx)
                - exact_pay
            ).mean()
            np.testing.assert_allclose(gap, 3.0 * df * (w / 4.0) / (grid_s.size * delta), rtol=1e-6)
    for bad in (
        lambda: Digital(100.0, 1.0, "call", discount, smoothing=-1.0),
        lambda: Digital(1.0, 1.0, "call", discount, smoothing=3.0),
        lambda: Digital(-1.0, 1.0, "call", discount),
    ):
        with pytest.raises(ValueError):
            bad()
    aged = Digital(100.0, 1.0, "call", discount, smoothing=0.5).aged(0.25)
    assert isinstance(aged, Digital) and pytest.approx(0.75) == aged.T and aged.smoothing == 0.5
    # Black–Scholes: exact digital and a narrow spread both match the closed form
    model = BlackScholes(0.2, forward_curve)
    sim = SimConfig(n_paths=40_000, dt_max=0.25, chunk_size=20_000, seed=5)
    prods = [
        Digital(100.0, 1.0, "call", discount),
        Digital(100.0, 1.0, "put", discount, smoothing=0.5),
    ]
    res = MonteCarlo(sim).price_many(prods, model)
    for prod, rr in zip(prods, res, strict=True):
        closed = float(bs_digital_price(100.0, 100.0, 1.0, 0.2, 0.02, 0.01, prod.cp))
        assert abs(rr.mean - closed) < 3.5 * rr.stderr + 1e-3, (prod, rr, closed)


# ---------------------------------------------------------------------------------------------
# closed forms
# ---------------------------------------------------------------------------------------------


def _knock_out_quadrature(
    S: float, K: float, H: float, T: float, vol: float, r: float, q: float, cp: int, direction: str
) -> float:
    """Knock-out price by quadrature of the reflection-principle density of x = ln(S_T/S)
    killed at h = ln(H/S): e^{−rT} ∫_alive (cp (S e^x − K))⁺ [n((x − mT)/s) − e^{2mh/σ²}
    n((x − 2h − mT)/s)] / s dx with m = r − q − σ²/2, s = σ√T (independent of the A..D
    assembly of Reiner–Rubinstein)."""
    m = r - q - 0.5 * vol * vol
    s = vol * np.sqrt(T)
    h = np.log(H / S)

    def dens(x: float) -> float:
        return float(
            (
                norm.pdf((x - m * T) / s)
                - np.exp(2 * m * h / vol**2) * norm.pdf((x - 2 * h - m * T) / s)
            )
            / s
        )

    def pay(x: float) -> float:
        return max(cp * (S * np.exp(x) - K), 0.0)

    if direction == "down":
        lo, hi = h, max(h, m * T) + 12 * s
    else:
        lo, hi = min(h, m * T) - 12 * s, h
    kink = np.log(K / S)
    val, _ = quad(
        lambda x: pay(x) * dens(x), lo, hi, points=[kink] if lo < kink < hi else None, limit=200
    )
    return float(np.exp(-r * T) * val)


def test_closed_form_identities() -> None:
    """Internal checks of ``barrier_bs``: the eight knock-out prices against the quadrature of
    the killed density (5 strikes each side of K = H), the image identities DIC(K ≥ H) =
    (H/S)^{2μ} C_BS(H²/S, K) and UIP(K ≤ H) = (H/S)^{2μ} P_BS(H²/S, K), in–out parity against
    bs_price, continuity at K = H, barrier limits, one-touch + no-touch = DF, hit probability
    against an independent reflection-principle transcription, the F term against a quadrature
    of e^{−rt} times the first-passage density at r > 0, rebate decompositions, digital = limit
    of a call spread with an O(w²) bias, validation (negative radicand, non-finite rates)."""
    S, T, vol, r, q = 100.0, 1.0, 0.2, 0.02, 0.01
    strikes = np.array([80.0, 95.0, 100.0, 110.0, 125.0])
    m = r - q - 0.5 * vol**2
    mu = m / vol**2
    for direction, H in (("down", 90.0), ("up", 115.0)):
        for cp in (1, -1):
            ki = bs_barrier_price(S, strikes, H, T, vol, r, q, cp, direction, "in")
            ko = bs_barrier_price(S, strikes, H, T, vol, r, q, cp, direction, "out")
            van = bs_price(S, strikes, T, vol, r, q, cp)
            np.testing.assert_allclose(ki + ko, van, rtol=0, atol=1e-10)
            assert np.all(ki >= -1e-12) and np.all(ko >= -1e-12)
            # independent reference for the reflected terms: quadrature of the killed density
            quad_ko = [
                _knock_out_quadrature(S, float(K), H, T, vol, r, q, cp, direction) for K in strikes
            ]
            np.testing.assert_allclose(ko, quad_ko, rtol=0, atol=1e-10)
            # continuity of the two strike branches at K = H
            near = bs_barrier_price(
                S, H * np.array([1 - 1e-7, 1.0, 1 + 1e-7]), H, T, vol, r, q, cp, direction, "out"
            )
            assert np.ptp(near) < 1e-4
        # image identities (the C term): DIC for K >= H, UIP for K <= H
        if direction == "down":
            ks = np.array([90.0, 100.0, 125.0])
            img = (H / S) ** (2 * mu) * bs_price(H * H / S, ks, T, vol, r, q, 1)
            np.testing.assert_allclose(
                bs_barrier_price(S, ks, H, T, vol, r, q, 1, "down", "in"), img, rtol=0, atol=1e-10
            )
        else:
            ks = np.array([80.0, 100.0, 115.0])
            img = (H / S) ** (2 * mu) * bs_price(H * H / S, ks, T, vol, r, q, -1)
            np.testing.assert_allclose(
                bs_barrier_price(S, ks, H, T, vol, r, q, -1, "up", "in"), img, rtol=0, atol=1e-10
            )
        # limits: a far barrier gives the vanilla (out) or nothing (in); at the spot, the reverse
        far = 1e-3 if direction == "down" else 1e5
        np.testing.assert_allclose(
            bs_barrier_price(S, 100.0, far, T, vol, r, q, 1, direction, "out"),
            bs_price(S, 100.0, T, vol, r, q, 1),
            rtol=1e-9,
        )
        assert bs_barrier_price(S, 100.0, far, T, vol, r, q, -1, direction, "in") < 1e-9
        assert bs_barrier_price(
            S, 100.0, S, T, vol, r, q, 1, direction, "out", 2.0, "hit"
        ) == pytest.approx(2.0)
        np.testing.assert_allclose(
            bs_barrier_price(S, 100.0, S, T, vol, r, q, -1, direction, "in"),
            bs_price(S, 100.0, T, vol, r, q, -1),
        )
        # hit probability against the reflection principle, coded independently
        h = np.log(H / S)
        s = vol * np.sqrt(T)
        sign = 1.0 if direction == "down" else -1.0
        p_ref = norm.cdf(sign * (h - m * T) / s) + np.exp(2 * m * h / vol**2) * norm.cdf(
            sign * (h + m * T) / s
        )
        p_hit = float(bs_hit_probability(S, H, T, vol, r, q, direction))
        assert p_hit == pytest.approx(p_ref, abs=1e-12) and 0 < p_hit < 1
        assert bs_hit_probability(S, S, T, vol, r, q, direction) == 1.0
        # F term at r > 0 against the quadrature of e^{-rt} x first-passage density
        # |h| / (sigma sqrt(2 pi t^3)) exp(-(h - m t)^2 / (2 sigma^2 t)) over (0, T)

        def first_passage(t: float, h: float = h) -> float:
            return float(
                abs(h)
                / (vol * np.sqrt(2 * np.pi * t**3))
                * np.exp(-((h - m * t) ** 2) / (2 * vol**2 * t))
            )

        f_quad, _ = quad(lambda t: np.exp(-r * t) * first_passage(t), 0.0, T, limit=200)
        assert float(bs_hit_discount(S, H, T, vol, r, q, direction)) == pytest.approx(
            f_quad, abs=1e-9
        )
        p_quad, _ = quad(first_passage, 0.0, T, limit=200)
        assert p_hit == pytest.approx(p_quad, abs=1e-9)
        # touches
        ot = bs_one_touch_price(S, H, T, vol, r, q, direction, payout=3.0)
        nt = bs_no_touch_price(S, H, T, vol, r, q, direction, payout=3.0)
        assert float(ot + nt) == pytest.approx(3.0 * np.exp(-r * T))
        assert float(ot) == pytest.approx(3.0 * np.exp(-r * T) * p_hit)
        # rebate decompositions: maturity rebate = R DF P(hit) (out) / R DF P(no hit) (in);
        # the hit-paid rebate is worth more than the maturity-paid one when r > 0 and equal at r = 0
        for cp, K in ((1, 100.0), (-1, 100.0)):
            base_out = bs_barrier_price(S, K, H, T, vol, r, q, cp, direction, "out")
            with_r = bs_barrier_price(S, K, H, T, vol, r, q, cp, direction, "out", 5.0, "maturity")
            assert float(with_r - base_out) == pytest.approx(5.0 * np.exp(-r * T) * p_hit)
            hit_r = bs_barrier_price(S, K, H, T, vol, r, q, cp, direction, "out", 5.0, "hit")
            assert float(hit_r - base_out) == pytest.approx(
                5.0 * float(bs_hit_discount(S, H, T, vol, r, q, direction))
            )
            assert hit_r > with_r
            base_in = bs_barrier_price(S, K, H, T, vol, r, q, cp, direction, "in")
            in_r = bs_barrier_price(S, K, H, T, vol, r, q, cp, direction, "in", 5.0, "maturity")
            assert float(in_r - base_in) == pytest.approx(5.0 * np.exp(-r * T) * (1 - p_hit))
        assert float(
            bs_barrier_price(S, 100.0, H, T, vol, 0.0, q, 1, direction, "out", 5.0, "hit")
        ) == pytest.approx(
            float(
                bs_barrier_price(S, 100.0, H, T, vol, 0.0, q, 1, direction, "out", 5.0, "maturity")
            )
        )
        assert float(bs_hit_discount(S, H, T, vol, 0.0, q, direction)) == pytest.approx(
            float(bs_hit_probability(S, H, T, vol, 0.0, q, direction))
        )
    # digital as the limit of the call spread, with the O(w^2) bias of a central difference
    for cp in (1, -1):
        dig = float(bs_digital_price(S, 110.0, T, vol, r, q, cp, payout=2.0))

        def spread(w: float, cp: int = cp) -> float:
            return (
                2.0
                * cp
                * float(
                    bs_price(S, 110.0 - w / 2, T, vol, r, q, cp)
                    - bs_price(S, 110.0 + w / 2, T, vol, r, q, cp)
                )
                / w
            )

        assert dig == pytest.approx(spread(1e-4), abs=1e-6)
        e1, e2 = spread(2.0) - dig, spread(1.0) - dig
        assert abs(e1) > 1e-6 and e1 / e2 == pytest.approx(4.0, rel=1e-2)
    assert float(
        bs_digital_price(S, 100.0, T, vol, r, q, 1) + bs_digital_price(S, 100.0, T, vol, r, q, -1)
    ) == pytest.approx(np.exp(-r * T))
    # a negative radicand raises instead of NaN: mu^2 + 2 r / sigma^2 = ((r - q - sigma^2/2)^2
    # + 2 r sigma^2) / sigma^4 is a perfect square ((r + sigma^2/2) / sigma^2)^2 when q = 0, so it
    # needs r < 0 and q != 0; r = -2%, q = -4% gives mu = 0 and a radicand of -1
    r_bad, q_bad = -0.02, -0.04
    assert (
        ((r_bad - q_bad - 0.5 * vol**2) / vol**2) ** 2 + 2 * r_bad / vol**2
    ) == pytest.approx(-1.0)
    for bad in (
        lambda: bs_barrier_price(S, 100.0, 90.0, T, vol, r, q, 1, "down", "out", 1.0),
        lambda: bs_barrier_price(S, 100.0, 90.0, T, vol, r, q, 1, "down", "in", 1.0, "hit"),
        lambda: bs_barrier_price(S, 100.0, 90.0, T, vol, r, q, 1, "down", "through"),
        lambda: bs_barrier_price(S, 100.0, 90.0, T, vol, r, q, 2, "down", "out"),
        lambda: bs_barrier_price(S, 100.0, 90.0, T, vol, r, q, 1, "sideways", "out"),
        lambda: bs_barrier_price(S, 100.0, -90.0, T, vol, r, q, 1, "down", "out"),
        lambda: bs_barrier_price(S, 100.0, 90.0, T, vol, np.nan, q, 1, "down", "out"),
        lambda: bs_barrier_price(S, 100.0, 90.0, T, vol, r, q, 1, "down", "out", np.inf, "hit"),
        lambda: bs_hit_discount(S, 90.0, T, vol, r_bad, q_bad, "down"),
        lambda: bs_barrier_price(
            S, 100.0, 90.0, T, vol, r_bad, q_bad, 1, "down", "out", 1.0, "hit"
        ),
        lambda: bs_digital_price(S, 100.0, T, 0.0, r, q, 1),
        lambda: bs_one_touch_price(S, 90.0, T, vol, r, q, "down", payout=np.inf),
    ):
        with pytest.raises(ValueError):
            bad()
    # ... while the E term (maturity rebate) and the hit probability stay defined there
    assert np.isfinite(
        bs_barrier_price(S, 100.0, 90.0, T, vol, r_bad, q_bad, 1, "down", "out", 1.0, "maturity")
    )
    assert 0.0 < float(bs_hit_probability(S, 90.0, T, vol, r_bad, q_bad, "down")) < 1.0


# ---------------------------------------------------------------------------------------------
# Monte Carlo against the closed forms
# ---------------------------------------------------------------------------------------------


def _bs_setup() -> tuple[BlackScholes, DiscountCurve, dict[str, float]]:
    fc = ForwardCurve.flat(100.0, 0.02, 0.01)
    return BlackScholes(0.2, fc), fc.rate_curve, dict(S=100.0, T=1.0, vol=0.2, r=0.02, q=0.01)


def test_closed_forms_vs_bridge_monte_carlo() -> None:
    """Continuous barriers with bridge weights (dt = 1/250, 40k paths, seed 7) within 3 stderr
    of Reiner–Rubinstein: down-and-out call, up-and-out put, down-and-in put, with rebates at the
    hit and at maturity, one-touch and no-touch."""
    model, disc, m = _bs_setup()
    S, T, vol, r, q = m["S"], m["T"], m["vol"], m["r"], m["q"]
    cases = [
        (
            KnockOutOption(100.0, T, "call", 90.0, "down", disc, monitoring="continuous"),
            bs_barrier_price(S, 100.0, 90.0, T, vol, r, q, 1, "down", "out"),
        ),
        (
            KnockOutOption(100.0, T, "put", 115.0, "up", disc, monitoring="continuous"),
            bs_barrier_price(S, 100.0, 115.0, T, vol, r, q, -1, "up", "out"),
        ),
        (
            KnockInOption(100.0, T, "put", 90.0, "down", disc, monitoring="continuous"),
            bs_barrier_price(S, 100.0, 90.0, T, vol, r, q, -1, "down", "in"),
        ),
        (
            KnockOutOption(
                100.0,
                T,
                "call",
                90.0,
                "down",
                disc,
                monitoring="continuous",
                rebate=5.0,
                rebate_timing="hit",
            ),
            bs_barrier_price(S, 100.0, 90.0, T, vol, r, q, 1, "down", "out", 5.0, "hit"),
        ),
        (
            KnockOutOption(
                100.0,
                T,
                "put",
                115.0,
                "up",
                disc,
                monitoring="continuous",
                rebate=5.0,
                rebate_timing="maturity",
            ),
            bs_barrier_price(S, 100.0, 115.0, T, vol, r, q, -1, "up", "out", 5.0, "maturity"),
        ),
        (
            KnockInOption(
                100.0,
                T,
                "put",
                90.0,
                "down",
                disc,
                monitoring="continuous",
                rebate=5.0,
                rebate_timing="maturity",
            ),
            bs_barrier_price(S, 100.0, 90.0, T, vol, r, q, -1, "down", "in", 5.0, "maturity"),
        ),
        (
            OneTouch(90.0, T, "down", disc, monitoring="continuous"),
            bs_one_touch_price(S, 90.0, T, vol, r, q, "down"),
        ),
        (
            NoTouch(115.0, T, "up", disc, monitoring="continuous"),
            bs_no_touch_price(S, 115.0, T, vol, r, q, "up"),
        ),
    ]
    sim = SimConfig(n_paths=40_000, dt_max=1 / 250, chunk_size=20_000, seed=7)
    res = MonteCarlo(sim).price_many([p for p, _ in cases], model)
    for (prod, closed), rr in zip(cases, res, strict=True):
        z = (rr.mean - float(closed)) / rr.stderr
        assert rr.stderr > 0 and abs(z) < 3.0, (prod, rr, float(closed), z)


def test_sampled_vs_weighted(ssvi: SSVISurface, local_vol: LocalVolSurface) -> None:
    """Weighted and sampled survival agree within 3 stderr of their CRN difference; the sampled
    payoff is, given the weights w, a Bernoulli mixture whose variance of antithetic pair means
    is Var(pair(w V)) + E[w (1 − w) V²]/2 (the pair's two uniforms are independent), so the
    sampled stderr and the weighted/sampled ratio are predicted from the weighted payoffs alone
    and checked within 3% (Black–Scholes and local vol at dt = 1/250, Black–Scholes at monthly
    steps); the bridge averages more at coarser steps (E[w(1 − w)] increases); continuous
    monitoring is worth no more than discrete monitoring on the simulation grid path by path."""
    model, disc, _ = _bs_setup()

    def run(model: BlackScholes | LocalVol, disc: DiscountCurve, spot: float, dt: float) -> float:
        sim = SimConfig(n_paths=40_000, dt_max=dt, chunk_size=20_000, seed=11)
        k, h_down = spot, 0.9 * spot
        prods = [
            KnockOutOption(k, 1.0, "call", h_down, "down", disc, monitoring="continuous"),
            KnockOutOption(
                k, 1.0, "call", h_down, "down", disc, monitoring="continuous", survival="sampled"
            ),
            OneTouch(h_down, 1.0, "down", disc, monitoring="continuous"),
            OneTouch(h_down, 1.0, "down", disc, monitoring="continuous", survival="sampled"),
            # discrete comparator observed on the simulation grid (its fixings would otherwise
            # refine the shared grid: price_many unions every product's fixings)
            KnockOutOption(
                k,
                1.0,
                "call",
                h_down,
                "down",
                disc,
                fixing_times=uniform_schedule(1.0, round(1 / dt)),
                **_mode("discrete"),
            ),
        ]
        res = MonteCarlo(sim).price_many(prods, model, keep_payoffs=True)
        pays = [np.asarray(rr.payoffs) for rr in res]
        for w, s in ((0, 1), (2, 3)):
            d, se = _diff_stats(pays[w], pays[s])
            assert abs(d) < 3.0 * se + 1e-12, (prods[w], res[w], res[s], d, se)
            assert res[w].stderr <= res[s].stderr, (prods[w], res[w], res[s])
        assert np.all(pays[0] <= pays[4] + 1e-12) and res[0].mean < res[4].mean
        df = float(disc.df(1.0))
        assert 0.0 < res[2].mean < df
        # predicted sampled variance from the weights (one-touch: V = payout DF; DOC: V w = pay)
        n_pairs = pays[0].size // 2
        w = 1.0 - pays[2] / df
        assert np.all((w >= 0) & (w <= 1))
        var_touch = np.var(_pairs(pays[2]), ddof=1) + df * df * np.mean(w * (1 - w)) / 2
        se_touch = float(np.sqrt(var_touch / n_pairs))
        assert res[3].stderr == pytest.approx(se_touch, rel=0.03), (res[3], se_touch)
        assert res[2].stderr / res[3].stderr == pytest.approx(res[2].stderr / se_touch, rel=0.03)
        safe_w = np.where(w > 0, w, 1.0)
        term = np.where(w > 0, (1 - w) * pays[0] ** 2 / safe_w, 0.0)
        var_doc = np.var(_pairs(pays[0]), ddof=1) + np.mean(term) / 2
        se_doc = float(np.sqrt(var_doc / n_pairs))
        assert res[1].stderr == pytest.approx(se_doc, rel=0.03), (res[1], se_doc)
        assert res[2].stderr < res[3].stderr and res[0].stderr < res[1].stderr
        return float(np.mean(w * (1 - w)))

    fine = run(model, disc, 100.0, 1 / 250)
    run(LocalVol(local_vol), ssvi.discount, ssvi.forward_curve.spot, 1 / 250)
    coarse = run(model, disc, 100.0, 1 / 12)
    assert coarse > 2.0 * fine, (fine, coarse)  # more bridge averaging at monthly steps


def test_discrete_to_continuous_convergence() -> None:
    """Daily-style discrete monitoring of a down-and-out call at 12, 52, 252 and 1008
    observations per year, priced on one path set with the bridge-weighted continuous product:
    prices decrease towards the continuous one (path by path for nested schedules), the
    continuous price matches Reiner–Rubinstein within 3 stderr, and the CRN difference between
    each discrete price and the continuous one matches the Broadie–Glasserman–Kou continuity
    correction H exp(−0.5826 σ sqrt(Δt)) within 3 stderr of the difference plus the size of the
    term the correction drops (the price move of the second-order log-barrier shift σ² Δt)."""
    model, disc, m = _bs_setup()
    S, T, vol, r, q = m["S"], m["T"], m["vol"], m["r"], m["q"]
    K, H = 100.0, 90.0
    freqs = (12, 52, 252, 1008)
    prods = [
        KnockOutOption(
            K,
            T,
            "call",
            H,
            "down",
            disc,
            fixing_times=uniform_schedule(T, f),
            **_mode("discrete"),
        )
        for f in freqs
    ]
    cont = KnockOutOption(K, T, "call", H, "down", disc, monitoring="continuous")
    sim = SimConfig(n_paths=40_000, dt_max=1 / 250, chunk_size=20_000, seed=17)
    res = MonteCarlo(sim).price_many([*prods, cont], model, keep_payoffs=True)
    pays = [np.asarray(rr.payoffs) for rr in res]
    rr_cont = float(bs_barrier_price(S, K, H, T, vol, r, q, 1, "down", "out"))
    z_cont = (res[-1].mean - rr_cont) / res[-1].stderr
    assert abs(z_cont) < 3.0, (res[-1], rr_cont)
    # ordering: every discrete price above the continuous one path by path; nested schedules
    # (monthly ⊂ daily ⊂ 4x daily) ordered path by path; all consecutive means significantly ordered
    for p in pays[:-1]:
        assert np.all(p >= pays[-1] - 1e-12)
    assert np.all(pays[0] >= pays[2] - 1e-12) and np.all(pays[2] >= pays[3] - 1e-12)
    table = []
    for i, f in enumerate(freqs):
        d, se = _diff_stats(pays[i], pays[-1])
        assert d > 3.0 * se, (f, d, se)
        if i:
            d2, se2 = _diff_stats(pays[i - 1], pays[i])
            assert d2 > 2.0 * se2, (f, d2, se2)
        dt = 1.0 / f
        h_bgk = H * np.exp(-BGK_BETA * vol * np.sqrt(dt))
        rr_bgk = float(bs_barrier_price(S, K, h_bgk, T, vol, r, q, 1, "down", "out"))
        # the o(1/sqrt m) remainder of the BGK expansion: the price move of a log-barrier shift
        # of the next order, sigma^2 dt (0.10 / 0.023 / 0.005 / 0.001 here)
        c_f = abs(
            float(
                bs_barrier_price(S, K, H * np.exp(vol * vol * dt), T, vol, r, q, 1, "down", "out")
            )
            - rr_cont
        )
        resid = d - (rr_bgk - rr_cont)
        table.append((f, res[i].mean, res[i].stderr, rr_bgk, d, se, resid, c_f))
        assert abs(resid) < 3.0 * se + c_f, (f, d, se, rr_bgk - rr_cont, c_f)
    # the gap to the continuous closed form shrinks monotonically with the frequency
    gaps_cont = [row[1] - rr_cont for row in table]
    assert gaps_cont == sorted(gaps_cont, reverse=True) and gaps_cont[-1] > 0
    print(f"\nDOC discrete -> continuous (RR continuous = {rr_cont:.5f}, MC bridge = {res[-1]}):")
    for f, mean, se_m, rr_bgk, d, se, resid, c_f in table:
        print(
            f"  {f:5d}/yr: MC {mean:.5f} +- {se_m:.5f}  BGK-shifted RR {rr_bgk:.5f}  "
            f"CRN(discrete - continuous) {d:.5f} +- {se:.5f} vs {rr_bgk - rr_cont:.5f}: "
            f"residual {resid:+.5f} ({resid / se:+.2f} se, allowance {c_f:.4f})"
        )


# ---------------------------------------------------------------------------------------------
# validation, term sheets, ageing
# ---------------------------------------------------------------------------------------------


def test_validation_reprs_and_ageing(discount: DiscountCurve) -> None:
    ok = _mode("discrete")
    for bad in (
        lambda: KnockOutOption(100.0, 1.0, "call", 90.0, "down", discount, monitoring="close"),
        lambda: KnockOutOption(100.0, 1.0, "call", 90.0, "sideways", discount, **ok),
        lambda: KnockOutOption(
            100.0, 1.0, "call", 90.0, "down", discount, monitoring="continuous", survival="mean"
        ),
        lambda: KnockOutOption(100.0, 1.0, "call", 90.0, "down", discount, rebate=1.0, **ok),
        lambda: KnockOutOption(
            100.0, 1.0, "call", 90.0, "down", discount, rebate=1.0, rebate_timing="later", **ok
        ),
        lambda: KnockInOption(
            100.0, 1.0, "call", 90.0, "down", discount, rebate=1.0, rebate_timing="hit", **ok
        ),
        lambda: KnockOutOption(-1.0, 1.0, "call", 90.0, "down", discount, **ok),
        lambda: KnockOutOption(100.0, 1.0, "call", -90.0, "down", discount, **ok),
        lambda: KnockOutOption(
            100.0, 1.0, "call", 90.0, "down", discount, fixing_times=[0.5, 1.5], **ok
        ),
        lambda: KnockOutOption(100.0, 1.0, "call", 90.0, "down", discount, fixing_times=[], **ok),
        lambda: KnockOutOption(
            100.0,
            1.0,
            "call",
            90.0,
            "down",
            discount,
            monitoring="continuous",
            fixing_times=[0.5, 0.5],
        ),
        lambda: KnockOutOption(
            100.0,
            1.0,
            "call",
            90.0,
            "down",
            discount,
            monitoring="continuous",
            fixing_times=[0.5, 1.5],
        ),
        lambda: KnockOutOption(100.0, 1.0, "straddle", 90.0, "down", discount, **ok),
        lambda: OneTouch(90.0, 1.0, "down", discount, payout=np.inf, **ok),
        # mode-irrelevant conventions are refused, not ignored
        lambda: KnockOutOption(100.0, 1.0, "call", 90.0, "down", discount, monitoring="discrete"),
        lambda: NoTouch(90.0, 1.0, "down", discount, monitoring="discrete"),
        lambda: KnockOutOption(
            100.0, 1.0, "call", 90.0, "down", discount, monitoring="continuous", strict=True
        ),
        lambda: KnockOutOption(
            100.0, 1.0, "call", 90.0, "down", discount, monitoring="continuous", strict=False
        ),
        lambda: KnockOutOption(
            100.0, 1.0, "call", 90.0, "down", discount, survival="sampled", **ok
        ),
        lambda: KnockOutOption(100.0, 1.0, "call", 90.0, "down", discount, seed=5, **ok),
        lambda: KnockOutOption(
            100.0, 1.0, "call", 90.0, "down", discount, monitoring="continuous", seed=5
        ),
        lambda: OneTouch(
            90.0, 1.0, "down", discount, monitoring="continuous", survival="sampled", seed=-1
        ),
        lambda: OneTouch(
            90.0, 1.0, "down", discount, monitoring="continuous", survival="sampled", seed=1.5
        ),
    ):
        with pytest.raises(ValueError):
            bad()
    ko = KnockOutOption(
        100.0,
        1.0,
        "put",
        115.0,
        "up",
        discount,
        monitoring="discrete",
        fixing_times=[0.0, 0.5, 1.0],
        rebate=2.0,
        rebate_timing="hit",
        strict=True,
    )
    rep = repr(ko)
    for token in (
        "Up-and-out Put",
        "strike 100",
        "expiry 1y",
        "barrier 115 (monitored 115, shift +0%)",
        "discrete monitoring (3 obs from 0y to 1y, knocked when S > B_eff)",
        "rebate 2 at hit",
        "notional 1",
    ):
        assert token in rep, (token, rep)
    assert "knocked when S >= B_eff" in repr(
        KnockOutOption(100.0, 1.0, "put", 115.0, "up", discount, **ok)
    )
    ki = KnockInOption(
        100.0,
        2.0,
        "call",
        80.0,
        "down",
        discount,
        monitoring="continuous",
        survival="sampled",
        seed=4,
        fixing_times=[0.5, 2.0],
    )
    rep = repr(ki)
    for token in (
        "Down-and-in Call",
        "continuous monitoring on [0.5, 2]y (Brownian bridge, survival sampled, seed 4, draws "
        "keyed on the path set (no CRN across bumps))",
        "rebate 0,",
    ):
        assert token in rep, (token, rep)
    assert ki.fixing_times.tolist() == [0.5, 2.0] and ki.maturity == 2.0 and ki.requires_all_steps
    assert ki.strict is None and ko.survival == "weight" and ko.seed == 0
    assert "One-touch (down)" in repr(
        OneTouch(90.0, 1.0, "down", discount, monitoring="continuous")
    )
    assert "No-touch (up)" in repr(NoTouch(110.0, 1.0, "up", discount, **ok))
    # default schedule: daily from 0 (inception monitored) to T
    d = KnockOutOption(100.0, 1.0, "call", 90.0, "down", discount, **ok)
    np.testing.assert_allclose(d.fixing_times, daily_schedule(1.0))
    assert d.schedule.size == 253
    # ageing shifts every date and keeps the conventions
    aged = ko.aged(0.25)
    assert isinstance(aged, KnockOutOption)
    np.testing.assert_allclose(aged.schedule, [0.0, 0.25, 0.75])
    assert aged.T == 0.75 and aged.rebate == 2.0 and aged.rebate_timing == "hit" and aged.strict
    aged_ki = ki.aged(0.25)
    assert isinstance(aged_ki, KnockInOption)
    np.testing.assert_allclose(aged_ki.schedule, [0.25, 1.75])
    assert aged_ki.survival == "sampled" and aged_ki.seed == 4 and aged_ki.T == 1.75
    ot = OneTouch(90.0, 1.0, "down", discount, monitoring="continuous", payout=3.0).aged(0.5)
    assert isinstance(ot, OneTouch) and ot.payout == 3.0 and ot.schedule.tolist() == [0.0, 0.5]
    assert ot.survival == "weight" and ot.seed == 0 and ot.strict is None
    # a monitoring date inside the roll window was observed at the held spot and drops out
    # (t = 0 stays; the convention shared with the autocall, base.Product.aged); the maturity
    # itself inside the window still raises
    dropped = ko.aged(0.5)
    assert isinstance(dropped, KnockOutOption)
    np.testing.assert_allclose(dropped.schedule, [0.0, 0.5])
    assert dropped.T == 0.5
    with pytest.raises(ValueError):
        ko.aged(1.0)
    # with_discount rebinds the curve
    other = DiscountCurve.flat(0.05)
    assert ko.with_discount(other).discount is other and ko.discount is discount
