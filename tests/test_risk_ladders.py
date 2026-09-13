"""M5 ladders (SPEC v2 §7.4–7.6) under local vol on the reference surface: vega-T waves and
projections against single tents, the forward-variance ladder against the variance swap's
analytic sensitivity and against the parallel bump, skew and curvature ladders on vanillas."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from volsto.config import CalibrationSpec, SimConfig, load_yaml
from volsto.products import EuropeanOption, VarianceSwap, daily_schedule
from volsto.risk import (
    LVBuilder,
    RiskEngine,
    RiskState,
    check_pillars_calendar,
    curvature_T,
    fwd_var_ladder,
    skew_T,
    varswap_bucket_sensitivity,
    vega_T,
)
from volsto.risk.engine import surface_of

ROOT = Path(__file__).resolve().parents[1]
SPEC_1F = ROOT / "configs" / "studies" / "lsv_reference_1f.yaml"
PILLARS_5 = (0.25, 0.5, 1.0, 2.0, 3.0)


@pytest.fixture(scope="module")
def lv_engine() -> tuple[RiskEngine, RiskState]:
    state = RiskState(load_yaml(SPEC_1F, CalibrationSpec))
    sim = SimConfig(n_paths=40_000, chunk_size=20_000, seed=11)
    return RiskEngine(LVBuilder(state), sim), state


def test_pillar_calendar_check() -> None:
    check_pillars_calendar((1 / 12, 2 / 12, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0))
    with pytest.raises(ValueError):
        check_pillars_calendar((1.0, 1.05))  # 1.05 < 1.1025 at 20% for a 1 vp bump
    with pytest.raises(ValueError):
        check_pillars_calendar((1.0,))


def test_vega_T_waves_projections_and_parallel(lv_engine) -> None:
    engine, state = lv_engine
    opt = EuropeanOption(100.0, 1.0, 1, surface_of(state).discount)
    vt = vega_T(engine, opt, state, PILLARS_5, with_tents=True)
    # projections sum to the parallel vega (wave_n) by construction
    assert sum(p.value for p in vt.projections) == pytest.approx(vt.parallel.value, abs=1e-12)
    # projections equal single-pillar tents to first order (2 stderr + a small cross term)
    for p, t in zip(vt.projections, vt.tents):
        assert (
            abs(p.value - t.value)
            < 2.0 * np.hypot(p.stderr, t.stderr) + 0.02 * abs(vt.parallel.value) + 1e-4
        ), (p, t)
    # the 1y option loads mainly on the 1y pillar and its neighbours; far pillars carry nothing
    tents = {round(pl, 3): t.value for pl, t in zip(vt.pillars, vt.tents)}
    assert tents[1.0] > 0.5 * vt.parallel.value and abs(tents[3.0]) < 0.05 * vt.parallel.value
    frame = vt.as_frame()
    assert list(frame.columns)[:2] == ["pillar", "wave"] and len(frame) == 5


def test_fwd_var_ladder_variance_swap_analytic_and_sum(lv_engine) -> None:
    """Under local vol a bucket bump of the forward variance moves a plain variance swap by the
    model-independent log-contract strip difference (``N · DF · [(σ_b + 0.01)² − σ_b²]/12`` for
    the monthly buckets): the monthly ladder is flat and its sum equals the parallel bump within
    2 stderr."""
    engine, state = lv_engine
    surface = surface_of(state)
    vs = VarianceSwap(daily_schedule(1.0), 0.06, surface.discount, notional=100.0)
    buckets = tuple((i / 12, (i + 1) / 12) for i in range(12))
    ladder = fwd_var_ladder(engine, vs, state, buckets)
    analytic = [varswap_bucket_sensitivity(vs, surface, lo, hi) for lo, hi in buckets]
    for e, a in zip(ladder.entries, analytic):
        assert abs(e.value - a) < 3.0 * e.stderr + 0.05 * abs(a) + 1e-3, (e, a)
    vals = np.array([e.value for e in ladder.entries])
    assert vals.std() < 0.1 * vals.mean()  # flat across equal monthly buckets
    assert ladder.total is not None and ladder.parallel is not None
    assert abs(ladder.total.value - ladder.parallel.value) < 2.0 * np.hypot(
        ladder.total.stderr, ladder.parallel.stderr
    ) + 0.02 * abs(ladder.parallel.value)
    assert len(ladder.as_frame()) == 12


def test_skew_and_curvature_ladders_on_vanillas(lv_engine) -> None:
    engine, state = lv_engine
    surface = surface_of(state)
    disc = surface.discount
    F = float(surface.forward(1.0))
    atm = EuropeanOption(F, 1.0, 1, disc)
    put90 = EuropeanOption(0.9 * F, 1.0, -1, disc)
    call110 = EuropeanOption(1.1 * F, 1.0, 1, disc)
    pillars = (0.5, 1.0, 2.0)
    sk_atm = skew_T(engine, atm, state, pillars)
    sk_put = skew_T(engine, put90, state, pillars)
    i1 = pillars.index(1.0)
    # ATM vanilla: zero skew vega at its pillar within stderr (+ discretisation allowance);
    # 90% put: positive
    e = sk_atm.entries[i1]
    assert abs(e.value) < 3.0 * e.stderr + 2e-3 * atm.strike * 0.01, e
    assert sk_put.entries[i1].value > 3.0 * sk_put.entries[i1].stderr
    # the ladder sums to the global rotation within 2 stderr
    assert sk_put.total is not None and sk_put.parallel is not None
    assert abs(sk_put.total.value - sk_put.parallel.value) < 2.0 * np.hypot(
        sk_put.total.stderr, sk_put.parallel.stderr
    ) + 0.02 * abs(sk_put.parallel.value)
    # curvature: a 90/110 risk reversal has none, a strangle has positive curvature vega
    cv_put = curvature_T(engine, put90, state, pillars)
    cv_call = curvature_T(engine, call110, state, pillars)
    rr = cv_put.entries[i1].value - cv_call.entries[i1].value
    strangle = cv_put.entries[i1].value + cv_call.entries[i1].value
    se = np.hypot(cv_put.entries[i1].stderr, cv_call.entries[i1].stderr)
    assert abs(rr) < 3.0 * se + 0.1 * strangle, (rr, strangle, se)
    assert strangle > 3.0 * se
    assert 0 < cv_put.entries[i1].size <= cv_put.entries[i1].extra["requested"]  # halving reported
