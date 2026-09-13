"""M5 profiles and parameter sensitivities (SPEC v2 §7.8–7.9): the spot-shift profile against
Black–Scholes, the cliquet gamma profile against the Bachelier cross-check, and the structure of
the parameter-sensitivity table (zero response of Black–Scholes, halving at the PSD boundary)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from helpers import flat_state as _flat_state

from volsto.config import (
    BergomiParams,
    SimConfig,
)
from volsto.market import bs_gamma, bs_price, bs_vega
from volsto.market.curves import ForwardCurve
from volsto.models.bs import BlackScholes
from volsto.products import AdditiveCliquet, EuropeanOption
from volsto.risk import (
    BSBuilder,
    RiskEngine,
    RiskState,
    bs_cliquet_value_mc,
    cliquet_gamma_profile,
    parameter_sensitivities,
    spot_profile,
)
from volsto.risk.engine import surface_of


def test_spot_profile_black_scholes() -> None:
    state = _flat_state()
    sim = SimConfig(n_paths=20_000, dt_max=1.0 / 50.0, chunk_size=10_000, seed=3)
    engine = RiskEngine(BSBuilder(state), sim)
    opt = EuropeanOption(100.0, 1.0, 1, surface_of(state).discount)
    prof = spot_profile(engine, opt, state, shifts=(-0.2, -0.1, 0.0, 0.1, 0.2))
    assert list(prof["shift"]) == [-0.2, -0.1, 0.0, 0.1, 0.2]
    for _, r in prof.iterrows():
        args = (r["spot"], 100.0, 1.0, 0.2, 0.02, 0.01)
        assert abs(r["price"] - float(bs_price(*args, 1))) < 3.5 * r["price_stderr"] + 1e-3
        assert abs(r["vega"] - 0.01 * float(bs_vega(*args))) < 3.5 * r["vega_stderr"] + 5e-3
        assert abs(r["gamma"] - float(bs_gamma(*args))) < 3.5 * r["gamma_stderr"] + 2e-4
    mid = prof.iloc[2]
    assert abs(mid["gamma_fd"] - mid["gamma"]) < 0.3 * mid["gamma"]  # 10% steps: coarse
    assert np.isnan(prof["gamma_fd"].iloc[0]) and np.isnan(prof["gamma_fd"].iloc[-1])
    assert engine.n_calibrations == 1  # the base state only: the profile never recalibrates


def test_cliquet_gamma_profile_bachelier_cross_check() -> None:
    """Study cliquet (monthly, local cap 2%, global floor 0) under Black–Scholes at the 6m
    fixing: the regressed conditional value matches the exact independent-legs reference over
    the 10–90% range of the accumulated sum; the Bachelier (Gaussian-sum) value of the study sits
    below it by a bounded amount (left-skewed remaining sum); the gamma profile peaks in the
    lower half of the range, around the floor."""
    fc = ForwardCurve.flat(100.0, 0.02, 0.01)
    model = BlackScholes(0.2, fc)
    cliquet = AdditiveCliquet.study(1.0, fc.rate_curve, notional=100.0)
    sim = SimConfig(n_paths=100_000, dt_max=1.0 / 24.0, chunk_size=50_000, seed=8)
    prof = cliquet_gamma_profile(cliquet, model, sim, 0.5, bs_sigma=0.2)
    frame = prof.as_frame()
    assert prof.bachelier is not None and np.all(np.isfinite(frame["value"]))
    assert prof.n_paths == 100_000
    a = frame["accumulated"].to_numpy()
    value = frame["value"].to_numpy()
    bach = frame["bachelier"].to_numpy()
    exact, exact_se = bs_cliquet_value_mc(cliquet, a, 0.5, 0.2, fc, seed=5)
    mid = (a > np.percentile(a, 10)) & (a < np.percentile(a, 90))
    # regression against the exact reference, point by point: 1.5% of the value, the reference's
    # noise and 0.03% of notional (the value is exactly 0 below A = −0.12, where the capped legs
    # cannot recover the floor)
    err = np.abs(value[mid] - exact[mid]) - 3.0 * exact_se[mid]
    assert np.all(err < 0.015 * exact[mid] + 0.03), (value[mid], exact[mid])
    assert np.max(np.abs(frame["value_raw"].to_numpy()[mid] - exact[mid])) > np.max(err)
    # the study's Gaussian-sum approximation overprices the floor (measured): one-signed and
    # bounded by 0.25% of notional (5% of the top-of-range value; 20-25% of the value near
    # A = 0, where the value is 1% of notional)
    dev = value[mid] - bach[mid]
    assert np.max(dev) < 0.03 and np.min(dev) > -0.25, dev
    print(
        "bachelier deviation (value minus bachelier) min/max:",
        dev.min(),
        dev.max(),
        "value max",
        value.max(),
    )
    assert np.all(np.diff(value[2:-2]) > -0.05)  # increasing in the accumulated sum
    # gamma profile: positive peak where the Bachelier curvature peaks (A + mu = floor)
    g = frame["gamma"].to_numpy()
    peak = int(np.nanargmax(g))
    bach_gamma = np.gradient(np.gradient(bach, a), a)
    da = a[1] - a[0]
    assert g[peak] > 0 and abs(a[peak] - a[int(np.argmax(bach_gamma))]) <= 2.0 * da + 1e-12


def test_parameter_sensitivities_structure_black_scholes() -> None:
    state = _flat_state()
    sim = SimConfig(n_paths=10_000, dt_max=1.0 / 50.0, chunk_size=10_000, seed=3)
    engine = RiskEngine(BSBuilder(state), sim)
    opt = EuropeanOption(100.0, 1.0, 1, surface_of(state).discount)
    table = parameter_sensitivities(engine, opt, state, params=("nu", "rho_SX1", "theta"))
    assert list(table["param"]) == ["nu", "rho_SX1", "theta"]
    np.testing.assert_allclose(table["value"], 0.0, atol=1e-12)  # BS ignores the SV parameters
    assert table.loc[0, "bump"] == pytest.approx(0.05 * 0.5)
    assert table.loc[1, "bump"] == pytest.approx(0.05)
    assert np.isfinite(table.loc[0, "value_sticky"]) and np.isnan(table.loc[1, "value_sticky"])
    assert list(table["scheme"]) == ["central", "central", "forward"]  # θ = 0 is a boundary
    # a correlation whose down-bump leaves |ρ| ≤ 1 is differenced one-sided at the full bump
    near = RiskState(
        dataclasses.replace(state.spec, model=BergomiParams.one_factor(1.0, 1.0, -0.98))
    )
    t2 = parameter_sensitivities(engine, opt, near, params=("rho_SX1",))
    assert t2.loc[0, "bump"] == pytest.approx(0.05) and t2.loc[0, "scheme"] == "forward"
