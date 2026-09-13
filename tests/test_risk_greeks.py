"""M5 risk engine: delta regimes, vega variants, theta split and cross-Greeks (SPEC v2 §7.1–7.3,
§7.7).  Black–Scholes analytic checks on a flat surface (all regimes and variants coincide),
the forward-start before T1, and the sticky-strike / sticky-moneyness delta relation under
local vol on the skewed reference surface."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from volsto.config import (
    BergomiParams,
    CalibrationSpec,
    CurveConfig,
    MarketConfig,
    SimConfig,
    SSVIConfig,
    load_yaml,
)
from volsto.market import (
    bs_delta,
    bs_gamma,
    bs_price,
    bs_rho,
    bs_theta,
    bs_vanna,
    bs_vega,
    bs_volga,
)
from volsto.products import EuropeanOption, ForwardStartOption, VarianceSwap, daily_schedule
from volsto.risk import (
    REGIMES,
    BSBuilder,
    LVBuilder,
    RiskEngine,
    RiskState,
    cross_greeks,
    delta_gamma,
    delta_table,
    theta,
    vega,
)
from volsto.risk.engine import surface_of

ROOT = Path(__file__).resolve().parents[1]
SPEC_1F = ROOT / "configs" / "studies" / "lsv_reference_1f.yaml"


def _flat_state(sigma: float = 0.2, r: float = 0.02, q: float = 0.01) -> RiskState:
    spec = CalibrationSpec(
        market=MarketConfig(100.0, CurveConfig((1.0,), (r,)), CurveConfig((1.0,), (q,))),
        surface=SSVIConfig((1 / 12, 0.25, 0.5, 1.0, 2.0, 3.0), (sigma,) * 6, 0.0, 0.0, 0.5, 10.0),
        model=BergomiParams.one_factor(1.0, 1.0, 0.0),
    )
    return RiskState(spec)


def _bs(K: float, T: float, cp: int, state: RiskState, sigma: float = 0.2) -> dict[str, float]:
    m = state.spec.market
    r, q = m.rate_curve.rates[0], m.dividend_curve.rates[0]
    args = (m.spot, K, T, sigma, r, q)  # bs_* take (S, K, T, vol, r, q[, cp])
    return {
        "price": float(bs_price(*args, cp)),
        "delta": float(bs_delta(*args, cp)),
        "gamma": float(bs_gamma(*args)),
        "vega": float(bs_vega(*args)),
        "theta": float(bs_theta(*args, cp)),
        "rho": float(bs_rho(*args, cp)),
        "vanna": float(bs_vanna(*args)),
        "volga": float(bs_volga(*args)),
    }


@pytest.fixture(scope="module")
def bs_engine() -> tuple[RiskEngine, RiskState]:
    state = _flat_state()
    sim = SimConfig(n_paths=40_000, dt_max=1.0 / 50.0, chunk_size=20_000, seed=5)
    return RiskEngine(BSBuilder(state), sim), state


def test_bs_delta_gamma_vega_theta_analytic(bs_engine) -> None:
    engine, state = bs_engine
    for K, T, cp in ((100.0, 1.0, 1), (95.0, 0.5, -1)):
        opt = EuropeanOption(K, T, cp, surface_of(state).discount)
        ref = _bs(K, T, cp, state)
        d, g = delta_gamma(engine, opt, state, "model")
        assert abs(d.value - ref["delta"]) < 3.5 * d.stderr + 2e-4, (d, ref["delta"])
        assert abs(g.value - ref["gamma"]) < 3.5 * g.stderr + 2e-5, (g, ref["gamma"])
        v = vega(engine, opt, state, "recalibrated")
        # forward difference: allow the second-order (volga) term of the 1 vp bump
        assert (
            abs(v.value - 0.01 * ref["vega"])
            < 3.5 * v.stderr + 0.5 * abs(ref["volga"]) * 1e-4 + 1e-4
        )
        vs = vega(engine, opt, state, "sticky_leverage")
        assert vs.value == pytest.approx(v.value)  # BS: one model for every variant
        th = theta(engine, opt, state)
        assert (
            abs(th.total.value - ref["theta"])
            < 3.5 * th.total.stderr + 0.02 * abs(ref["theta"]) + 1e-3
        )
        assert th.total.value == pytest.approx(th.decay.value + th.carry.value + th.roll_down.value)
        assert abs(th.roll_down.value) < 3.5 * th.roll_down.stderr + 1e-3  # flat surface: no roll
        # BS identity θ + ½σ²S²Γ + (r − q)Sδ − rP = 0 with the analytic Greeks
        m = state.spec.market
        r, q = m.rate_curve.rates[0], m.dividend_curve.rates[0]
        ident = (
            ref["theta"]
            + 0.5 * 0.04 * 100**2 * ref["gamma"]
            + (r - q) * 100 * ref["delta"]
            - r * ref["price"]
        )
        assert abs(ident) < 1e-8
        assert d.unit == "per unit spot" and "regime" in d.extra
    assert engine.n_calibrations > 0 and engine.budget()["pricings"] > 0


def test_bs_all_regimes_coincide_and_cross_greeks(bs_engine) -> None:
    engine, state = bs_engine
    opt = EuropeanOption(100.0, 1.0, 1, surface_of(state).discount)
    table = delta_table(engine, opt, state)
    assert list(table["regime"]) == list(REGIMES)
    np.testing.assert_allclose(table["delta"], table["delta"].iloc[0], rtol=1e-12)
    np.testing.assert_allclose(table["gamma"], table["gamma"].iloc[0], rtol=1e-12)
    ref = _bs(100.0, 1.0, 1, state)
    cg = cross_greeks(engine, opt, state, params=("rho_SX1",))
    # vanna in both forms (per vol point): ∂vega/∂lnS = S·∂²P/∂S∂σ, ∂delta/∂σ = ∂²P/∂S∂σ
    v1 = cg["vanna_dvega_dlnS"]
    assert (
        abs(v1.value - 100.0 * ref["vanna"] * 0.01)
        < 3.5 * v1.stderr + 0.05 * abs(100 * ref["vanna"] * 0.01) + 1e-3
    )
    v2 = cg["vanna_ddelta_dsigma"]
    assert (
        abs(v2.value - ref["vanna"] * 0.01)
        < 3.5 * v2.stderr + 0.05 * abs(ref["vanna"] * 0.01) + 1e-4
    )
    vo = cg["volga"]
    assert (
        abs(vo.value - ref["volga"] * 1e-4)
        < 3.5 * vo.stderr + 0.1 * abs(ref["volga"] * 1e-4) + 1e-4
    )
    rh = cg["rho"]
    assert (
        abs(rh.value - ref["rho"] * 1e-4) < 3.5 * rh.stderr + 0.05 * abs(ref["rho"] * 1e-4) + 1e-5
    )
    assert np.isfinite(cg["charm"].value) and np.isfinite(cg["veta"].value)
    assert np.isfinite(cg["repo_delta"].value) and np.isfinite(cg["ddelta_drho_SX1"].value)
    # BS does not depend on rho_SX1: the cross term vanishes exactly (same model, same draws)
    assert cg["ddelta_drho_SX1"].value == pytest.approx(0.0, abs=1e-12)


def test_bs_forward_start_delta_small_relative_to_vega(bs_engine) -> None:
    """Before T1 a forward-start option has no spot exposure in BS (homogeneity); its vega is
    material.  Report the ratio; every regime coincides."""
    engine, state = bs_engine
    fs = ForwardStartOption(0.5, 1.0, 1.0, 1, surface_of(state).discount)
    d, _ = delta_gamma(engine, fs, state, "model")
    v = vega(engine, fs, state)
    assert abs(d.value) < 3.5 * d.stderr + 1e-6 and v.value > 0
    d_sm, _ = delta_gamma(engine, fs, state, "sticky_moneyness")
    assert abs(d_sm.value) < 3.5 * d_sm.stderr + 1e-6
    print("forward-start |delta| / vega(per vp) =", abs(d.value) / v.value)


def test_theta_raises_inside_roll_window(bs_engine) -> None:
    engine, state = bs_engine
    vs = VarianceSwap(daily_schedule(0.5), 0.04, surface_of(state).discount)
    with pytest.raises(ValueError):
        theta(engine, vs, state, dt=1.0 / 252.0)  # the first daily fixing lies in the window
    with pytest.raises(ValueError):
        EuropeanOption(100.0, 1 / 365, 1, surface_of(state).discount).aged(1 / 252)


def test_lv_sticky_regimes_relation_on_reference_surface() -> None:
    """Under pure LV on the skewed reference surface, sticky-strike and sticky-moneyness deltas of
    an ATM vanilla differ by the vega times the ATM skew: ``δ_sm − δ_ss ≈ −vega · s_T / S``
    (§14 test), and the ATM-vol shifts per unit Δ are ordered 0 (moneyness) < s_T (strike) <
    2 s_T (local vol)."""
    spec = load_yaml(SPEC_1F, CalibrationSpec)
    state = RiskState(spec)
    sim = SimConfig(n_paths=60_000, chunk_size=30_000, seed=9)
    engine = RiskEngine(LVBuilder(state), sim)
    surface = surface_of(state)
    T, K = 1.0, float(surface.forward(1.0))
    opt = EuropeanOption(K, T, 1, surface.discount)
    d_ss, _ = delta_gamma(engine, opt, state, "sticky_strike")
    d_sm, _ = delta_gamma(engine, opt, state, "sticky_moneyness")
    v = vega(engine, opt, state)  # per vol point -> per unit vol = v / 0.01
    s_t = float((surface.implied_vol_k(1e-3, T) - surface.implied_vol_k(-1e-3, T)) / 2e-3)
    predicted = -v.value / 0.01 * s_t / state.spot
    diff = d_sm.value - d_ss.value
    se = np.hypot(d_sm.stderr, d_ss.stderr)
    assert s_t < 0 and diff > 0, (s_t, diff)
    assert abs(diff - predicted) < 3.5 * se + 0.1 * abs(predicted), (diff, predicted, se)
    # ATM-vol shift per unit Δ under each regime (the vol at the *new* money, k = 0 on the
    # rebuilt surface, minus the old ATM vol): 0 (moneyness), s_T (strike, skew), 2 s_T (local
    # vol) — with s_T < 0 the ordering runs in the direction of the skew
    from volsto.risk.greeks import _spot_state

    h = 0.01
    shifts = {}
    for regime in ("sticky_moneyness", "sticky_strike", "sticky_skew", "sticky_local_vol"):
        up, _ = _spot_state(state, regime, h)
        new = surface_of(up)
        shifts[regime] = (
            float(new.implied_vol_k(0.0, T)) - float(surface.implied_vol_k(0.0, T))
        ) / h
    ratios = {r: v / s_t for r, v in shifts.items()}
    assert abs(ratios["sticky_moneyness"]) < 0.02
    assert ratios["sticky_strike"] == pytest.approx(1.0, abs=0.05)
    assert ratios["sticky_skew"] == pytest.approx(1.0, abs=0.05)
    assert ratios["sticky_local_vol"] == pytest.approx(2.0, abs=0.05)
    assert ratios["sticky_moneyness"] < ratios["sticky_strike"] < ratios["sticky_local_vol"]
    assert engine.n_calibrations >= 5  # four regime states + the base Dupire rebuild
