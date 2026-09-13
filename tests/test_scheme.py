"""Spot-step scheme, step schedule and per-step recording (owner amendments after M1).

Measured on the reference surface (ATM, uniform dt = 1/365, 200k paths, bias in vol points):
1m: log-Euler +0.36, time-averaged variance +0.29, predictor-corrector θ=η=½ +0.78, Platen weak
order 2 +0.03; 3m: +0.15 / +0.11 / +0.41 / +0.01.  The default is weak order 2 with the default
step schedule (1/2920 below 3m, 1/730 to 2y, 1/500 after since the pre-M4 amendment).
"""

from __future__ import annotations

import numpy as np
import pytest

from volsto.config import SchemeConfig, SimConfig, StepSchedule
from volsto.engine import MonteCarlo, PathSet, TimeGrid, refinement_study
from volsto.engine.grid import FixingIndex
from volsto.market import (
    DiscountCurve,
    ForwardCurve,
    LocalVolSurface,
    SSVISurface,
    black_vega,
    implied_vol,
)
from volsto.models import BlackScholes, LocalVol
from volsto.products import EuropeanOption, Product

DAYS = 365.0


def _atm_bias_vp(
    ssvi: SSVISurface, local_vol: LocalVolSurface, cfg: SimConfig, T: float, ks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    F = float(ssvi.forward(T))
    df = float(ssvi.discount.df(T))
    cps = np.where(ks >= 0, 1, -1)
    K = F * np.exp(ks)
    prods = [EuropeanOption(k, T, int(cp), ssvi.discount) for k, cp in zip(K, cps)]
    res = MonteCarlo(cfg).price_many(prods, LocalVol(local_vol))
    pr = np.array([r.mean for r in res])
    er = np.array([r.stderr for r in res])
    iv = implied_vol(pr, F, K, T, cps, df)
    return 100 * (iv - ssvi.implied_vol_k(ks, T)), 100 * er / black_vega(F, K, T, iv, df)


# ---------------------------------------------------------------------------------------------
# accuracy of the default scheme + schedule
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("T", [1 / 12, 0.25, 0.5])
def test_default_scheme_reprices_short_maturities(
    ssvi: SSVISurface, local_vol: LocalVolSurface, T: float
) -> None:
    """Pure LV vs SSVI at 1m/3m/6m: ATM within 0.1 vp, ±10% within 0.2 vp (default options)."""
    cfg = SimConfig(n_paths=400_000, chunk_size=50_000, seed=3)
    assert cfg.scheme.weak_order2 and isinstance(cfg.dt_max, StepSchedule)
    bias, se = _atm_bias_vp(ssvi, local_vol, cfg, T, np.array([-0.1, 0.0, 0.1]))
    assert abs(bias[1]) < max(0.10, 3 * se[1]), (bias, se)
    assert np.all(np.abs(bias) < np.maximum(0.20, 3 * se)), (bias, se)


def test_scheme_options_are_exact_for_black_scholes(
    forward_curve: ForwardCurve, discount: DiscountCurve
) -> None:
    """With a flat variance every scheme reduces to the exact log-Euler step, path by path."""
    opt = EuropeanOption(105.0, 0.5, "call", discount)
    model = BlackScholes(0.25, forward_curve)
    means = []
    for opts in (
        dict(local_var_time_average=False, predictor_corrector=False, weak_order2=False),
        dict(local_var_time_average=True, predictor_corrector=False, weak_order2=False),
        dict(local_var_time_average=True, predictor_corrector=True, weak_order2=False, pc_eta=0.5),
        dict(local_var_time_average=False, predictor_corrector=False, weak_order2=True),
    ):
        cfg = SimConfig(n_paths=20_000, dt_max=1 / 52, chunk_size=10_000, seed=5, **opts)
        means.append(MonteCarlo(cfg).price(opt, model).mean)
    np.testing.assert_allclose(means, means[0], rtol=1e-12)


def test_time_average_table_matches_quadrature(local_vol: LocalVolSurface) -> None:
    t0, t1 = 0.2, 0.2 + 1 / DAYS
    avg = local_vol.var_time_average([t0, t1])[0]
    tt = np.linspace(t0, t1, 2001)
    num = np.trapezoid(
        local_vol.local_var_k(tt[:, None], local_vol.k_grid[None, :]), tt, axis=0
    ) / (t1 - t0)
    np.testing.assert_allclose(avg, num, rtol=1e-9)
    np.testing.assert_allclose(
        local_vol.var_at_times([0.3])[0], local_vol.local_var_k(0.3, local_vol.k_grid)
    )


# ---------------------------------------------------------------------------------------------
# refinement diagnostics under CRN
# ---------------------------------------------------------------------------------------------


def test_euler_bias_is_first_order_and_richardson_removes_it(
    ssvi: SSVISurface, local_vol: LocalVolSurface
) -> None:
    """Plain log-Euler: D_i = P(dt_i) - P(dt_{i+1}) halves when dt halves; 2P(dt/2) - P(dt) is
    closer to the surface price than P(dt/2) itself."""
    T = 30 / DAYS
    F = float(ssvi.forward(T))
    opt = EuropeanOption(F, T, "call", ssvi.discount)
    cfg = SimConfig(
        n_paths=200_000,
        dt_max=1 / DAYS,
        chunk_size=50_000,
        seed=9,
        local_var_time_average=False,
        predictor_corrector=False,
        weak_order2=False,
    )
    study = refinement_study(cfg, opt, LocalVol(local_vol), dt=1 / DAYS, levels=3, order=1)
    d0, d1 = study.differences
    assert abs(d0.mean) > 5 * d0.stderr and abs(d1.mean) > 5 * d1.stderr, study
    ratio = d0.mean / d1.mean
    assert 1.5 < ratio < 3.0, study
    exact = float(ssvi.price(F, T, 1))
    fine = study.prices[-1]
    assert abs(study.richardson.mean - exact) < abs(fine.mean - exact), study
    assert abs(study.richardson.mean - exact) < max(
        0.05 / 100 * float(black_vega(F, F, T, 0.22, 1.0)), 3 * study.richardson.stderr
    ), study


def test_weak_order2_residual_is_small_and_higher_order(
    ssvi: SSVISurface, local_vol: LocalVolSurface
) -> None:
    T = 30 / DAYS
    F = float(ssvi.forward(T))
    opt = EuropeanOption(F, T, "call", ssvi.discount)
    cfg = SimConfig(n_paths=200_000, dt_max=1 / DAYS, chunk_size=50_000, seed=9)
    study = refinement_study(cfg, opt, LocalVol(local_vol), dt=1 / DAYS, levels=3, order=2)
    vega = float(black_vega(F, F, T, 0.22, 1.0))
    d0, d1 = study.differences
    # the dt -> dt/2 change is below 0.05 vol points and shrinks faster than first order
    assert abs(d0.mean) < 0.05 / 100 * vega, study
    assert abs(d1.mean) < 0.6 * abs(d0.mean) + 3 * d1.stderr, study
    exact = float(ssvi.price(F, T, 1))
    assert abs(study.prices[0].mean - exact) < max(
        0.1 / 100 * vega, 3 * study.prices[0].stderr
    ), study


# ---------------------------------------------------------------------------------------------
# step schedule and per-step recording
# ---------------------------------------------------------------------------------------------


def test_step_schedule() -> None:
    sch = StepSchedule()
    assert sch.dt_at(0.1) == pytest.approx(1 / 2920)
    assert sch.dt_at(0.25) == pytest.approx(1 / 730)
    assert sch.dt_at(1.99) == pytest.approx(1 / 730)
    assert sch.dt_at(2.0) == pytest.approx(1 / 500)
    assert sch.knots(1.0) == (0.25,) and sch.knots(3.0) == (0.25, 2.0)
    assert StepSchedule.uniform(0.01).dt_at(5.0) == 0.01
    with pytest.raises(ValueError):
        StepSchedule(breaks=(1.0,), dts=(0.1,))
    grid = TimeGrid.build([0.5, 1.0, 2.5], sch)
    t, dts = grid.times[:-1], grid.dts
    assert np.all(dts[t < 0.25] <= 1 / 2920 + 1e-12)
    assert np.all(dts[(t >= 0.25) & (t < 2.0)] <= 1 / 730 + 1e-12)
    assert np.all(dts[t >= 2.0] <= 1 / 500 + 1e-12)
    assert np.max(dts[t >= 2.0]) > 1 / 730  # the coarse segment is really used
    for b in (0.25, 2.0):
        assert np.any(np.abs(grid.times - b) < 1e-12)
    np.testing.assert_allclose(grid.record_times, [0.0, 0.5, 1.0, 2.5])
    # SimConfig accepts a float or a schedule
    assert SimConfig(dt_max=0.01).step_schedule.dt_at(3.0) == 0.01
    assert SimConfig().step_schedule == sch


class _Lookback(Product):
    """Max of the recorded spot over all columns — needs every step."""

    requires_all_steps = True

    def __init__(self, maturity: float, discount: DiscountCurve) -> None:
        super().__init__(discount)
        self.T = maturity

    @property
    def fixing_times(self) -> np.ndarray:
        return np.array([self.T])

    def payoff(self, paths: PathSet, idx: FixingIndex) -> np.ndarray:
        assert paths.n_cols == len(idx) and idx[self.T] == paths.n_cols - 1
        return np.asarray(
            float(self.df(self.T)) * np.max(paths.spot_at(np.arange(paths.n_cols)), axis=1)
        )

    def __repr__(self) -> str:
        return f"Lookback max, expiry {self.T}"


def test_record_all_steps(forward_curve: ForwardCurve, discount: DiscountCurve) -> None:
    grid = TimeGrid.build([0.5], 0.05, record_all_steps=True)
    assert grid.n_records == grid.n_steps + 1
    np.testing.assert_array_equal(grid.record_times, grid.times)
    model = BlackScholes(0.2, forward_curve)
    cfg = SimConfig(n_paths=4_000, dt_max=1 / 52, chunk_size=2_000, seed=1)
    mc = MonteCarlo(cfg)
    lb = _Lookback(0.5, discount)
    g = mc.build_grid([lb], model)
    assert g.n_records == g.n_steps + 1
    res = mc.price(lb, model)
    vanilla = mc.price(EuropeanOption(1e-6, 0.5, "call", discount), model)  # ~ discounted forward
    assert res.mean > vanilla.mean
    # cfg.record_all_steps forces it for any product
    g2 = MonteCarlo(SimConfig(n_paths=4_000, dt_max=1 / 52, record_all_steps=True)).build_grid(
        [EuropeanOption(100.0, 1.0, "call", discount)], model
    )
    assert g2.n_records == g2.n_steps + 1
    # memory budget caps the chunk size
    cfg_small = SimConfig(n_paths=100_000, chunk_size=50_000, chunk_memory_mb=1)
    assert cfg_small.effective_chunk_size(n_cols=1000, n_factors=2) < 50_000
    assert cfg_small.effective_chunk_size(n_cols=1000, n_factors=2) % 2 == 0
    assert sum(b - a for a, b in cfg_small.chunk_ranges(1000, 2)) == 100_000


def test_scheme_config_validation() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        SchemeConfig(predictor_corrector=True, weak_order2=True)
    with pytest.raises(ValueError, match="pc_eta"):
        SchemeConfig(pc_eta=1.5)
    with pytest.raises(ValueError, match="local_var_time_eval"):
        SimConfig(local_var_time_eval="end")
