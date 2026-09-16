"""The conditional-Greeks delta estimator under Black–Scholes and the §7.11 delta control (SPEC
§8.2 "Study D reinstated", owner's decision of 2026-09-16; ``scripts/m8b_delta_estimator.py``
is the full measurement).  Tests never calibrate: the Black–Scholes bed uses the Black–Scholes
builder, the 2F section reads the leverage cache (skipped when absent).  No wall-clock
assertion; every figure printed with its standard error.

* FAST — the regime identity: under Black–Scholes the four §7.2 surface regimes, the
  ``sticky_local_vol`` reference and the ``min_variance`` benchmark give per-path P&L
  bit-identical to the ``model`` regime, with the delta control on and off (the estimator
  contributes no spread between regimes when the regimes coincide);
* SLOW — the estimator's inflation of the hedged std over the analytic-delta hedge on the same
  paths (1y ATM call, daily, 2·10⁴ pricing and world paths; measured 1.133 ± 0.006 without the
  delta control, 1.074 ± 0.005 with it), pinned loosely;
* SLOW — the delta control's variance reduction under the cached 2F marking LSV (> 1 at every
  probed date, stderr-based).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from volsto.calibration.cache import CacheMissError
from volsto.config import SimConfig
from volsto.hedging import ConditionalPricer, PricingContext
from volsto.hedging.hedger import HedgeResult
from volsto.hedging.pricing import union_grid
from volsto.hedging.report import _se_std
from volsto.hedging.strategies import DELTA_REGIMES
from volsto.products.vanilla import EuropeanOption

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "m8b_delta_estimator", ROOT / "scripts" / "m8b_delta_estimator.py"
)
assert _SPEC is not None and _SPEC.loader is not None
est = importlib.util.module_from_spec(_SPEC)
sys.modules["m8b_delta_estimator"] = est
_SPEC.loader.exec_module(est)


def _ctx() -> PricingContext:
    return PricingContext.from_state(est.flat_bs_state(), None, "bs", label="BS 20%")


def _run(ctx: PricingContext, regime: str, n: int, frequency: str, control: bool) -> HedgeResult:
    call = EuropeanOption(est.STRIKE, est.MATURITY, 1, ctx.model.forward_curve.rate_curve)
    return cast(HedgeResult, est.run_hedge(ctx, call, regime, n, n, frequency, control))


def test_bs_regime_identity_bit_identical() -> None:
    """Every delta regime reproduces the ``model`` regime's per-path P&L exactly under
    Black–Scholes (2·10³ paths, weekly), control on and off; the control changes the P&L (it
    does something) but not the identity."""
    ctx = _ctx()
    base: dict[bool, np.ndarray] = {}
    for control in (True, False):
        ref = _run(ctx, "model", 2_000, "weekly", control)
        base[control] = ref.pnl_total
        s, se = float(np.std(ref.pnl_total, ddof=1)), float(_se_std(ref.pnl_total))
        print(f"control {control}: model std {s:.4f} +/- {se:.4f}")
        for regime in DELTA_REGIMES:
            if regime == "model":
                continue
            r = _run(ctx, regime, 2_000, "weekly", control)
            diff = float(np.max(np.abs(r.pnl_total - ref.pnl_total)))
            print(f"  {regime}: max |P&L - model P&L| = {diff:.3e}")
            assert diff == 0.0, (regime, control, diff)
    assert not np.array_equal(base[True], base[False])


@pytest.mark.slow
def test_bs_estimator_inflation_ratio() -> None:
    """The estimator's hedged std over the analytic-delta hedge on the same world paths (1y ATM
    call, daily, 2·10⁴ pricing and world paths, the diagnosis's seeds): 1.133 ± 0.006 without the
    delta control and 1.074 ± 0.005 with it (measured 2026-09-16; the diagnosis's 1.110 was read
    against an analytic hedge whose quantity lacked the ``F(t)/S₀`` factor of the total-return
    spot, a bound 2% too high), pinned loosely; the control helps by more than three standard
    errors of the difference."""
    ctx = _ctx()
    fc = ctx.model.forward_curve
    ratios: dict[bool, tuple[float, float]] = {}
    for control in (False, True):
        r = _run(ctx, "model", 20_000, "daily", control)
        exact, _ = est.analytic_hedge(r, fc)
        ratios[control] = est.std_ratio(r.pnl_total, exact)
        print(
            f"control {control}: std {np.std(r.pnl_total, ddof=1):.4f} +/- "
            f"{_se_std(r.pnl_total):.4f} vs analytic {np.std(exact, ddof=1):.4f} +/- "
            f"{_se_std(exact):.4f}; ratio {ratios[control][0]:.4f} +/- {ratios[control][1]:.4f}"
        )
    off, on = ratios[False], ratios[True]
    assert 1.05 < off[0] < 1.18, off
    assert 0.98 < on[0] < 1.10, on
    assert off[0] - on[0] > 3.0 * float(np.hypot(off[1], on[1])), (off, on)


@pytest.mark.slow
def test_delta_control_variance_reduction_2f() -> None:
    """Under the cached 2F marking LSV (the M8b pricing model, never calibrated here) the delta
    control reduces the variance of the 1y ATM call's hybrid-CRN delta target at every probed
    date: reduction − 3 se > 1 (1·10⁴ paths)."""
    from volsto.studies.m8b import StudyConfig, StudyEnvironment

    env = StudyEnvironment(StudyConfig(allow_calibrate=False, verbose=False))
    try:
        ctx = env.pricing_ctx
    except CacheMissError as exc:
        pytest.skip(f"cached leverage absent (tests never calibrate): {exc}")
    model = ctx.model
    fc = model.forward_curve
    call = EuropeanOption(float(fc.spot), 1.0, 1, fc.rate_curve)
    sim = SimConfig(n_paths=10_000, chunk_size=10_000, seed=3, dt_max=1.0 / 252.0)
    dates = np.array([0.25, 0.5, 0.95])
    grid = union_grid([model], [call], dates, sim)
    pr = ConditionalPricer(model, [call], grid, sim, surface=ctx.surface, control_delta=True)
    for t in dates:
        f = pr.fit(0, float(t))
        vr, se = f.variance_reduction["delta"], f.variance_reduction_se["delta"]
        print(f"t = {t:g}: delta-target variance reduction {vr:.3f} +/- {se:.3f}, beta {f.beta}")
        assert "delta" in f.controlled and vr - 3.0 * se > 1.0, (t, vr, se)
    assert env.calibrations == 0
    pr.release()


def test_script_statistics_helpers() -> None:
    """The script's pair-level ratio estimator and std ratio on known inputs."""
    x = np.arange(1.0, 9.0)
    r, se = est.ratio_stat(x, np.ones_like(x))
    assert r == pytest.approx(4.5) and se > 0.0
    a = np.array([1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    ratio, rse = est.std_ratio(2.0 * a, a)
    assert ratio == pytest.approx(2.0) and rse == pytest.approx(0.0, abs=1e-12)
