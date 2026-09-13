"""M5 product-specific risks (SPEC v2 §7.10) under Black–Scholes: the fixing-risk jump of a
forward-start straddle against the after-fixing vanilla straddle, price continuity of the cliquet
legs across a fixing, the realised-variance exposure of a variance swap (flat, ``N · DF``),
barrier sensitivities and the second-order forward-variance ladder against volga."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import flat_state as _flat_state

from volsto.config import SimConfig
from volsto.market import bs_delta, bs_gamma, bs_volga
from volsto.market.bs import norm_cdf
from volsto.market.curves import ForwardCurve
from volsto.models.bs import BlackScholes
from volsto.products import (
    AdditiveCliquet,
    EuropeanOption,
    ForwardStartStraddle,
    KnockOutVarianceSwap,
    VarianceSwap,
    VolKnockOutPut,
    daily_schedule,
)
from volsto.risk import (
    BSBuilder,
    RiskEngine,
    barrier_profile,
    barrier_sensitivity,
    fixing_risk,
    fwd_var_convexity,
    ko_probability_delta,
    realised_variance_exposure,
)
from volsto.risk.engine import surface_of

SIGMA, R, Q = 0.2, 0.02, 0.01


def _engine(n_paths: int = 40_000, dt: float = 1.0 / 50.0, seed: int = 5) -> RiskEngine:
    state = _flat_state(SIGMA)
    sim = SimConfig(n_paths=n_paths, dt_max=dt, chunk_size=20_000, seed=seed)
    return RiskEngine(BSBuilder(state), sim)


def test_fixing_risk_forward_start_straddle() -> None:
    engine = _engine()
    state = engine.builder.base
    disc = surface_of(state).discount
    fs = ForwardStartStraddle(0.5, 1.5, 1.0, disc, notional=100.0)
    table = fixing_risk(engine, fs, state)
    assert len(table) == 1
    row = table.iloc[0]
    # before the fixing a forward start has no delta under homogeneous dynamics
    assert abs(row["delta_before"]) < 3.0 * row["delta_before_stderr"] + 1e-3
    assert abs(row["gamma_before"]) < 3.0 * row["gamma_before_stderr"] + 1e-4
    # after it, the vanilla straddle struck at S0 with T2 − T1 − 1d to run (per unit of S0)
    S, tau = 100.0, 1.0 - 1.0 / 252.0
    d_call = float(bs_delta(S, S, tau, SIGMA, R, Q, 1))
    d_put = float(bs_delta(S, S, tau, SIGMA, R, Q, -1))
    g = 2.0 * float(bs_gamma(S, S, tau, SIGMA, R, Q))
    assert abs(row["delta_after"] - (d_call + d_put)) < 3.0 * row["delta_after_stderr"] + 2e-3
    assert abs(row["gamma_after"] - g) < 3.0 * row["gamma_after_stderr"] + 0.15 * g
    assert row["vega_before"] > 0 and row["vega_after"] > 0
    # the jump is the vega-to-delta conversion: report the ratio
    ratio = row["delta_jump"] / row["vega_before"]
    print(
        f"forward-start straddle fixing jump: delta {row['delta_jump']:.4f}, gamma {row['gamma_jump']:.4f}, vega {row['vega_jump']:.4f}; delta jump / vega before = {ratio:.3f}"
    )
    assert np.isfinite(ratio)


def test_fixing_risk_cliquet_price_continuity() -> None:
    engine = _engine(20_000)
    state = engine.builder.base
    disc = surface_of(state).discount
    cliquet = AdditiveCliquet.study(0.5, disc, notional=100.0)
    table = fixing_risk(engine, cliquet, state)
    assert len(table) == 5  # intermediate fixings of a 6-leg cliquet
    # the legs are the same contract one day apart: the price is continuous across the fixing
    gap = np.abs(table["price_after"] - table["price_before"])
    assert np.all(gap < 0.05 * table["price_before"].abs() + 0.05), table[
        ["price_before", "price_after"]
    ]
    assert np.all(np.isfinite(table[["delta_jump", "gamma_jump", "vega_jump"]].to_numpy()))


def test_realised_variance_exposure_variance_swap_flat() -> None:
    fc = ForwardCurve.flat(100.0, R, Q)
    model = BlackScholes(SIGMA, fc)
    vs = VarianceSwap(daily_schedule(1.0), 0.04, fc.rate_curve, notional=100.0)
    sim = SimConfig(n_paths=20_000, dt_max=1.0 / 252.0, chunk_size=10_000, seed=3)
    prof = realised_variance_exposure(vs, model, sim, n_buckets=12)
    assert len(prof) == 12 and prof["n_periods"].sum() == 252
    target = 100.0 * float(fc.rate_curve.df(1.0))
    for _, r in prof.iterrows():
        assert abs(r["exposure"] - target) < 3.0 * r["stderr"] + 1e-6, (r, target)
    assert prof["exposure"].std() < 0.01 * target


def test_barrier_sensitivities_and_profile() -> None:
    engine = _engine(20_000, 1.0 / 52.0)
    state = engine.builder.base
    disc = surface_of(state).discount
    fix = daily_schedule(1.0, 52)  # weekly closes keep the test fast
    kov = KnockOutVarianceSwap(fix, 110.0, 0.0, disc, annualisation=52.0, notional=100.0)
    # weekly fixings annualised by 52; the vol barrier sits near sigma so the sensitivity is
    # measurable
    vko = VolKnockOutPut(100.0, 1.0, 0.21, fix, disc, annualisation=52.0, notional=1.0)
    b = barrier_sensitivity(engine, kov, state)
    assert (
        b["dprice_dB"].value > 3.0 * b["dprice_dB"].stderr
    )  # accrues longer with a higher barrier
    assert b["dprice_dB_pct"].value == pytest.approx(b["dprice_dB"].value * 0.01 * 110.0)
    v = barrier_sensitivity(engine, vko, state)
    assert v["dprice_dH"].value > 3.0 * v["dprice_dH"].stderr  # alive more often
    # P(KO) of the vol barrier does not depend on the spot level under Black–Scholes
    p_vko = ko_probability_delta(engine, vko, state)
    assert abs(p_vko.value) < 3.0 * p_vko.stderr + 1e-9
    # P(KO) of the spot barrier rises with the spot
    p_kov = ko_probability_delta(engine, kov, state)
    assert p_kov.value > 3.0 * p_kov.stderr
    prof = barrier_profile(engine, kov, state, 110.0)
    assert len(prof) == 21 and abs(prof["spot_over_barrier"].iloc[10] - 1.0) < 1e-9
    assert set(["price", "delta", "gamma", "gamma_fd"]).issubset(prof.columns)


def test_fwd_var_convexity_matches_volga_on_flat_surface() -> None:
    engine = _engine(40_000)
    state = engine.builder.base
    put = EuropeanOption(90.0, 1.0, -1, surface_of(state).discount)
    lad = fwd_var_convexity(engine, put, state, buckets=((0.0, 1.0),))
    e = lad.entries[0]
    ref = float(bs_volga(100.0, 90.0, 1.0, SIGMA, R, Q)) * 1e-4
    assert abs(e.value - ref) < 3.0 * e.stderr + 0.1 * abs(ref), (e, ref)
    assert e.extra["sizes"][1] > e.extra["sizes"][0] > 0
    _ = norm_cdf  # analytic helper kept for the docstring cross-reference
