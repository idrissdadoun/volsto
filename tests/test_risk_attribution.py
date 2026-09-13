"""M5 P&L attribution (SPEC v2 §7.12) under Black–Scholes: a pure spot move leaves a residual at
the third-order (speed) level, a pure parallel vol move a residual within 2 stderr, and a
combined move with one day of aging closes to the revalued P&L."""

from __future__ import annotations

import numpy as np
import pytest
from helpers import flat_state as _flat_state

from volsto.config import SimConfig, SurfacePerturbation
from volsto.market import bs_gamma
from volsto.products import EuropeanOption
from volsto.risk import BSBuilder, RiskEngine, explain
from volsto.risk.engine import surface_of

SIGMA, R, Q = 0.2, 0.02, 0.01


def _engine() -> RiskEngine:
    state = _flat_state(SIGMA)
    return RiskEngine(
        BSBuilder(state), SimConfig(n_paths=40_000, dt_max=1.0 / 50.0, chunk_size=20_000, seed=21)
    )


def test_explain_pure_spot_move_residual_third_order() -> None:
    engine = _engine()
    s0 = engine.builder.base
    call = EuropeanOption(100.0, 1.0, 1, surface_of(s0).discount)
    s1 = s0.with_spot(102.0, label="S=102")
    ex = explain(engine, call, s0, s1)
    frame = ex.as_frame()
    assert list(frame["step"]) == ["spot", "total"]
    step = ex.steps[0]
    # third-order term of a Black–Scholes call: (1/6) C''' dS³, C''' = −Γ (d1/(Sσ√T) + 1/S)
    S, dS = 100.0, 2.0
    gamma = float(bs_gamma(S, 100.0, 1.0, SIGMA, R, Q))
    d1 = (np.log(S / 100.0) + (R - Q + 0.5 * SIGMA**2)) / SIGMA
    third = -gamma * (d1 / (S * SIGMA) + 1.0 / S) * dS**3 / 6.0
    print(
        f"spot step: actual {step.actual:.5f} ± {step.actual_stderr:.2g}, explained {step.explained:.5f} (delta {step.detail['delta']:.5f}, gamma {step.detail['gamma']:.5f}), residual {step.residual:.2e}, analytic third order {third:.2e}"
    )
    assert abs(step.residual - third) < 3.0 * step.actual_stderr + 0.1 * abs(step.detail["gamma"])
    assert ex.total.value == pytest.approx(step.actual, abs=1e-9)
    assert ex.residual == pytest.approx(step.residual, abs=1e-9)


def test_explain_pure_parallel_vol_move_within_2_stderr() -> None:
    engine = _engine()
    s0 = engine.builder.base
    call = EuropeanOption(100.0, 1.0, 1, surface_of(s0).discount)
    s1 = s0.with_perturbation(SurfacePerturbation("parallel", {"size": 0.005}), label="vol+0.5vp")
    ex = explain(engine, call, s0, s1)
    assert [s.name for s in ex.steps] == ["surface"]
    step = ex.steps[0]
    print(
        f"vol step: actual {step.actual:.5f} ± {step.actual_stderr:.2g}, explained {step.explained:.5f}, residual {step.residual:.2e}"
    )
    assert abs(step.residual) < 2.0 * step.actual_stderr + 0.01 * abs(step.explained)
    # the ladder detail explains the same move through the tents (flat term structure)
    ex2 = explain(engine, call, s0, s1, detail="ladders", pillars=(0.5, 1.0, 2.0, 3.0))
    assert abs(ex2.steps[0].residual) < 2.0 * step.actual_stderr + 0.03 * abs(step.explained)


def test_explain_combined_move_with_aging() -> None:
    engine = _engine()
    s0 = engine.builder.base
    call = EuropeanOption(100.0, 1.0, 1, surface_of(s0).discount)
    s1 = s0.with_spot(101.0).with_perturbation(
        SurfacePerturbation("parallel", {"size": 0.005}), label="S=101,vol+0.5vp"
    )
    dt = 1.0 / 252.0
    ex = explain(engine, call, s0, s1, dt=dt)
    frame = ex.as_frame()
    assert list(frame["step"]) == ["spot", "surface", "time", "total"]
    p1 = engine.price(call.aged(dt), s1).mean
    assert ex.total.value == pytest.approx(p1 - ex.price_0, abs=1e-9) and ex.price_1 == p1
    assert abs(ex.residual) < 3.0 * ex.total.stderr + 0.02 * abs(ex.total.value)
    time_step = ex.steps[-1]
    assert time_step.explained < 0 and abs(
        time_step.residual
    ) < 3.0 * time_step.actual_stderr + 0.1 * abs(time_step.explained)
