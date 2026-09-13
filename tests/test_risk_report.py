"""M5 RiskReport (SPEC v2 §7.13): a full report on the Black–Scholes test bed (structure, flat
frame, Excel export, budget) and the slow LSV report at 2·10⁵ paths on the reference surface,
which prints the recalibration count and the wall clock (the viewer precompute budget)."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pytest
from helpers import flat_state as _flat_state

from volsto.calibration import LeverageCache
from volsto.config import CalibrationSpec, SimConfig, load_yaml
from volsto.products import AdditiveCliquet, EuropeanOption, VarianceSwap, daily_schedule
from volsto.risk import SECTIONS, BSBuilder, LSVBuilder, RiskEngine, RiskState, risk_report
from volsto.risk.engine import surface_of

ROOT = Path(__file__).resolve().parents[1]
SPEC_1F = ROOT / "configs" / "studies" / "lsv_reference_1f.yaml"
PILLARS_4 = (0.5, 1.0, 2.0, 3.0)
BUCKETS_4 = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))


def test_risk_report_black_scholes_structure(tmp_path: Path) -> None:
    state = _flat_state()
    engine = RiskEngine(
        BSBuilder(state), SimConfig(n_paths=20_000, dt_max=1.0 / 50.0, chunk_size=20_000, seed=2)
    )
    call = EuropeanOption(100.0, 1.0, 1, surface_of(state).discount)
    rep = risk_report(
        engine, call, state, pillars=PILLARS_4, buckets=BUCKETS_4, params=("nu", "rho_SX1")
    )
    frame = rep.to_dataframe()
    assert list(frame.columns[:8]) == [
        "group",
        "name",
        "value",
        "stderr",
        "unit",
        "size",
        "scheme",
        "states",
    ]
    groups = set(frame["group"])
    assert {
        "delta",
        "gamma",
        "vega",
        "theta",
        "cross",
        "vega_T",
        "fwd_var",
        "fwd_var_sticky",
        "skew",
        "curvature",
        "params",
    } <= groups
    assert (frame["group"] == "delta").sum() == 5 and (frame["group"] == "vega").sum() == 2
    assert np.all(np.isfinite(frame["value"])) and np.all(frame["stderr"] >= 0)
    assert {
        "delta_regimes",
        "vega_T",
        "fwd_var",
        "fwd_var_sticky",
        "skew",
        "curvature",
        "params",
    } <= set(rep.tables)
    assert rep.budget["report_recalibrations"] > 0 and rep.budget["report_wall_clock_s"] > 0
    assert rep.settings["pillars"] == list(PILLARS_4) and "recalibrations" in rep.summary()
    out = rep.to_excel(tmp_path / "report.xlsx")
    assert out.exists() and out.stat().st_size > 0
    # product sections apply by type: a cliquet gets its fixing risk, a variance swap its exposure
    cliquet = AdditiveCliquet.study(0.5, surface_of(state).discount, notional=100.0)
    rep_c = risk_report(engine, cliquet, state, sections=("product",))
    assert "fixing_risk" in rep_c.tables and len(rep_c.tables["fixing_risk"]) == 5
    vs = VarianceSwap(daily_schedule(0.5, 52), 0.04, surface_of(state).discount, notional=100.0)
    rep_v = risk_report(engine, vs, state, sections=("product",))
    assert "realised_variance_exposure" in rep_v.tables
    assert set(SECTIONS) >= {"delta", "product"}


@pytest.mark.slow
def test_risk_report_lsv_budget_2e5() -> None:
    """Full report of a 1y ATM call under the 1F LSV (ω = 3) at 2·10⁵ particles / 2·10⁵ paths:
    prints the recalibration count and the wall clock (SPEC v2 §7.13 budget)."""
    state = RiskState(load_yaml(SPEC_1F, CalibrationSpec))
    engine = RiskEngine(
        LSVBuilder(LeverageCache(ROOT / "cache"), state),
        SimConfig(n_paths=200_000, chunk_size=50_000, seed=2024),
    )
    call = EuropeanOption(100.0, 1.0, 1, surface_of(state).discount)
    t0 = time.perf_counter()
    rep = risk_report(engine, call, state)
    wall = time.perf_counter() - t0
    print(rep.summary(), f"| test wall clock {wall:.0f} s")
    print(rep.to_dataframe().to_string())
    assert rep.budget["report_recalibrations"] >= 60
    assert len(rep.cache_keys) == rep.budget["recalibrations"]
