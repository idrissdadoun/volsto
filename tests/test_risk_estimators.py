"""M5 precision estimators (SPEC v2 §7.11): likelihood-ratio delta and vega of a digital against
the analytic Black–Scholes values and the bump estimate, conditional Greeks by regression
against the Black–Scholes delta along the paths (and the t → 0 limit), and the control variate on
the difference under local vol."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import flat_state as _flat_state

from volsto.config import CalibrationSpec, SimConfig, load_yaml
from volsto.market import bs_delta, bs_price
from volsto.market.bs import norm_pdf
from volsto.market.curves import ForwardCurve
from volsto.models.bs import BlackScholes
from volsto.products import DigitalOption, EuropeanOption
from volsto.risk import (
    BSBuilder,
    LVBuilder,
    RiskEngine,
    RiskState,
    compare,
    conditional_greeks,
    cv_difference,
    delta_gamma,
    lr_delta,
    lr_vega,
)
from volsto.risk.engine import surface_of

ROOT = Path(__file__).resolve().parents[1]
SPEC_1F = ROOT / "configs" / "studies" / "lsv_reference_1f.yaml"
SIGMA, R, Q = 0.2, 0.02, 0.01


def _digital_greeks(S: float, K: float, T: float) -> tuple[float, float]:
    """Analytic delta and vega (per unit vol) of a cash-or-nothing call."""
    d1 = (np.log(S / K) + (R - Q + 0.5 * SIGMA**2) * T) / (SIGMA * np.sqrt(T))
    d2 = d1 - SIGMA * np.sqrt(T)
    df = np.exp(-R * T)
    delta = df * float(norm_pdf(d2)) / (S * SIGMA * np.sqrt(T))
    vega = -df * float(norm_pdf(d2)) * d1 / SIGMA
    return float(delta), float(vega)


def test_lr_delta_and_vega_digital_black_scholes() -> None:
    fc = ForwardCurve.flat(100.0, R, Q)
    model = BlackScholes(SIGMA, fc)
    dig = DigitalOption(100.0, 0.5, 1, fc.rate_curve)
    sim = SimConfig(n_paths=200_000, dt_max=1.0 / 52.0, chunk_size=50_000, seed=17)
    delta_ref, vega_ref = _digital_greeks(100.0, 100.0, 0.5)
    lr_d = lr_delta(dig, model, sim, first_step=1.0 / 52.0)
    assert lr_d.extra["n_steps"] == 1 and lr_d.size == pytest.approx(1.0 / 52.0)
    assert abs(lr_d.value - delta_ref) < 3.5 * lr_d.stderr, (lr_d, delta_ref)
    # aggregating four weeks is exact under Black–Scholes and less noisy
    lr_d4 = lr_delta(dig, model, sim, first_step=4.0 / 52.0)
    assert lr_d4.extra["n_steps"] == 4 and lr_d4.stderr < lr_d.stderr
    assert abs(lr_d4.value - delta_ref) < 3.5 * lr_d4.stderr
    lr_v = lr_vega(dig, model, sim)
    assert abs(lr_v.value - 0.01 * vega_ref) < 3.5 * lr_v.stderr, (lr_v, 0.01 * vega_ref)
    # the bump cross-check on the same digital through the engine
    state = _flat_state(SIGMA)
    engine = RiskEngine(BSBuilder(state), sim)
    d_bump, _ = delta_gamma(engine, dig, state, "model", 0.01)
    rep = compare(lr_d4, d_bump)
    print("LR vs bump delta on a digital:", rep)
    assert abs(rep["z"]) < 4.0
    assert abs(d_bump.value - delta_ref) < 3.5 * d_bump.stderr + 0.002 * delta_ref


def test_conditional_delta_black_scholes_vanilla_and_t_to_zero() -> None:
    fc = ForwardCurve.flat(100.0, R, Q)
    model = BlackScholes(SIGMA, fc)
    call = EuropeanOption(100.0, 1.0, 1, fc.rate_curve)
    sim = SimConfig(n_paths=100_000, dt_max=1.0 / 52.0, chunk_size=50_000, seed=9)
    cg = conditional_greeks(call, model, sim, 0.5)
    ln_s = cg.features[:, 0] * cg.scale[0] + cg.mean[0]
    s_t = np.exp(ln_s)
    fitted = cg.fitted("delta")
    exact = np.asarray(bs_delta(s_t, 100.0, 0.5, SIGMA, R, Q, 1))
    inner = (s_t > np.percentile(s_t, 5)) & (s_t < np.percentile(s_t, 95))
    rmse = float(np.sqrt(np.mean((fitted[inner] - exact[inner]) ** 2)))
    assert rmse < 0.03, rmse
    # the conditional value is the discounted Black–Scholes price at (S_t, T − t)
    val = cg.fitted("value")
    ref = np.exp(-R * 0.5) * np.asarray(bs_price(s_t, 100.0, 0.5, SIGMA, R, Q, 1))
    rmse_v = float(np.sqrt(np.mean((val[inner] - ref[inner]) ** 2)))
    assert rmse_v < 0.05 * float(ref[inner].mean()), rmse_v  # cubic basis: 4% at 6m
    assert 0 < cg.r2["value"] < 1
    # predict() reproduces the fit and the unconditional mean is the t = 0 delta
    pred = cg.predict("delta", ln_s[:5], cg.features[:5, 1:] * cg.scale[1:] + cg.mean[1:])
    np.testing.assert_allclose(pred, fitted[:5], rtol=1e-10, atol=1e-12)
    d0 = float(bs_delta(100.0, 100.0, 1.0, SIGMA, R, Q, 1))
    m, se = cg.unconditional("delta")
    assert abs(m - d0) < 3.5 * se + 1e-3
    # t → 0: the conditional delta at the first week collapses onto the t = 0 delta
    cg0 = conditional_greeks(call, model, sim, 1.0 / 52.0)
    at_spot = float(
        cg0.predict("delta", np.array([np.log(100.0)]), np.zeros((1, cg0.features.shape[1] - 1)))[0]
    )
    assert abs(at_spot - d0) < 0.02, (at_spot, d0)


@pytest.mark.slow
def test_cv_difference_under_local_vol() -> None:
    """Measured on the reference surface (40k paths): the Black–Scholes control reduces the vega
    variance by 10× (ATM) / 3× (90% put) but hardly touches the delta (VR 1.1–1.3, β < 0 for the
    ATM call under every regime: the pathwise LV difference anti-correlates with the BS one on
    the deep in-the-money paths) or the gamma (β ≈ 0).  The controlled estimate stays within 3
    stderr of the raw one and the control's MC mean matches its analytic value."""
    state = RiskState(load_yaml(SPEC_1F, CalibrationSpec))
    sim = SimConfig(n_paths=40_000, chunk_size=20_000, seed=11)
    engine = RiskEngine(LVBuilder(state), sim)
    call = EuropeanOption(100.0, 1.0, 1, surface_of(state).discount)
    out = {}
    for kind, regime in (("delta", "model"), ("delta", "sticky_moneyness"), ("vega", "model")):
        res = cv_difference(engine, call, state, kind, regime=regime)
        out[(kind, regime)] = res
        print(kind, regime, res.raw, res.controlled, "beta", res.beta, "VR", res.variance_reduction)
        assert res.variance_reduction >= 0.99  # a fitted control never adds variance
        assert abs(res.controlled.value - res.raw.value) < 3.0 * res.raw.stderr + 1e-9
        assert abs(res.control_mean - res.control_value) < 3.0 * res.raw.stderr + 2e-3
    assert out[("vega", "model")].variance_reduction > 2.0
    assert out[("delta", "model")].variance_reduction < 2.0  # the finding, not a target
