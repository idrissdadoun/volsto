"""VIX futures and options in the two-factor model (SPEC §15 Part 5; Bergomi §7.7).

* Quadrature against the factor Monte Carlo: future, calls and ATM implied vol of vol within
  2 stderr (Table 8.2, flat 20% curve, 3-month and 1-year expiries).
* ``E[VIX²]`` equals the forward 1m VS variance exactly and the future lies below its root by
  the convexity gap (Jensen), which grows with the vol of vol.
* Short horizon: the ATM implied vol of vol at one week tends to the instantaneous lognormal
  vol of the forward-starting 30-day VS volatility (eq. 7.39 machinery), and at ``T → 0`` to
  ``vs_vol_of_vol_flat(Δ)``.
* The VIX smile of the lognormal model is flat-to-downward across strikes at short expiry
  (the book's §7.7.4 remark: the model cannot produce the upward VIX skew) — recorded, not a
  target.
"""

from __future__ import annotations

import numpy as np
import pytest

from volsto.analytics.bergomi import forward_vs_vol_of_vol_flat, vs_vol_of_vol_flat
from volsto.analytics.vix import (
    VIX_DELTA,
    black_call,
    forward_vs_variance,
    vix_monte_carlo,
    vix_quadrature,
    vix_term_structure,
)
from volsto.config import BergomiParams, SimConfig
from volsto.market import ForwardCurve, ForwardVarianceCurve
from volsto.models import BergomiSV

P82 = BergomiParams(1.74, 0.245, 5.35, 0.28, 0.0, -0.759, -0.487)
FC0 = ForwardCurve.flat(100.0, 0.0, 0.0)


def _model(p: BergomiParams = P82, xi: float = 0.04) -> BergomiSV:
    return BergomiSV(p, ForwardVarianceCurve.flat(xi, 10.0), FC0)


def test_quadrature_matches_monte_carlo() -> None:
    m = _model()
    sim = SimConfig(n_paths=200_000, dt_max=1.0 / 52.0, chunk_size=100_000, seed=4)
    for T in (0.25, 1.0):
        q = vix_quadrature(m, T)
        mc = vix_monte_carlo(m, T, sim=sim, strikes=q.strikes)
        assert abs(q.future - mc.future) < 2 * mc.future_stderr + 2e-4, (q, mc)
        assert np.all(np.abs(q.calls - mc.calls) < 2 * mc.calls_stderr + 2e-4), (q, mc)
        assert abs(q.atm_vol_of_vol - mc.atm_vol_of_vol) < 2 * mc.atm_vol_of_vol_stderr + 1e-3
        assert q.forward_vs_variance == pytest.approx(mc.forward_vs_variance)
        assert "VIXQuotes" in repr(q) and len(q.to_dataframe()) == q.strikes.size
    # quadrature convergence in the number of Hermite nodes: the future converges at once (a
    # smooth integrand), the kinked option payoffs algebraically (0.4% at 20 vs 60 nodes)
    q60 = vix_quadrature(m, 1.0, n_hermite=60)
    q120 = vix_quadrature(m, 1.0, n_hermite=120)
    assert abs(q60.future - q120.future) < 1e-6
    assert abs(q60.atm_vol_of_vol - q120.atm_vol_of_vol) < 1.5e-3, (q60, q120)


def test_forward_variance_identity_and_convexity() -> None:
    m = _model()
    for T in (0.1, 0.5, 2.0):
        q = vix_quadrature(m, T)
        assert q.forward_vs_variance == pytest.approx(0.04, rel=1e-10)  # flat curve
        # E[VIX²] from the quadrature nodes equals the forward variance (martingale)
        assert q.future < np.sqrt(q.forward_vs_variance) and q.convexity_gap > 0
    # the convexity gap grows with the vol of vol
    gaps = [
        vix_quadrature(
            _model(BergomiParams(nu, 0.245, 5.35, 0.28, 0.0, -0.759, -0.487)), 0.5
        ).convexity_gap
        for nu in (0.5, 1.0, 1.74)
    ]
    assert np.all(np.diff(gaps) > 0), gaps
    assert forward_vs_variance(m, 0.5) == pytest.approx(0.04)
    with pytest.raises(ValueError):
        vix_quadrature(m, -1.0)


def _rms_forward_volvol(p: BergomiParams, T: float, n: int = 64) -> float:
    """``sqrt((1/T) ∫₀ᵀ ν²(T − t, T − t + Δ) dt)``: the Black implied vol of an option on the
    forward-starting 30-day VS vol at horizon ``T`` is the time average of its instantaneous
    lognormal vol, which rises from ``ν(T, T + Δ)`` at ``t = 0`` to ``ν_Δ(0)`` (eq. 7.39 at
    ``Δ``) at ``t = T``."""
    x, w = np.polynomial.legendre.leggauss(n)
    t = 0.5 * T * (x + 1.0)
    v2 = np.array(
        [float(forward_vs_vol_of_vol_flat(p, T - ti, T - ti + VIX_DELTA)) ** 2 for ti in t]
    )
    return float(np.sqrt(0.5 * float(v2 @ w)))


def test_short_horizon_vol_of_vol_limit() -> None:
    """ATM implied vol of vol of the VIX option = the time-averaged instantaneous vol of the
    forward-starting 30-day VS vol (1% at one week, where it lies between the ``t = 0`` value
    1.32 and eq. 7.39 at Δ, 1.44), and tends to eq. 7.39 at Δ as ``T → 0`` (one day: 1%)."""
    m = _model()
    T = 1.0 / 52.0
    q = vix_quadrature(m, T, n_hermite=60)
    inst0 = float(forward_vs_vol_of_vol_flat(P82, T, T + VIX_DELTA))
    limit = float(vs_vol_of_vol_flat(P82, VIX_DELTA))
    rms = _rms_forward_volvol(P82, T)
    assert inst0 < rms < limit, (inst0, rms, limit)
    assert abs(q.atm_vol_of_vol - rms) < 0.01 * rms, (q.atm_vol_of_vol, rms)
    q_day = vix_quadrature(m, 1.0 / 365.0, n_hermite=60)
    assert abs(q_day.atm_vol_of_vol - limit) < 0.01 * limit, (q_day.atm_vol_of_vol, limit)
    ts = vix_term_structure(m, [1 / 52, 0.25, 0.5, 1.0])
    assert list(ts.columns)[:2] == ["T", "future"] and np.all(ts["convexity_gap"] > 0)
    assert np.all(np.diff(ts["atm_vol_of_vol"]) < 0)  # vol of vol of VIX futures decays with expiry
    # the lognormal model's VIX smile: no upward skew at short expiry (book §7.7.4)
    ivs = q.implied_vols
    assert ivs[-1] <= ivs[len(ivs) // 2] + 0.02, ivs
    # Black helper sanity: the ATM call from the quadrature reprices with the implied vol
    k_atm = q.future
    assert float(black_call(q.future, k_atm, T, q.atm_vol_of_vol)) == pytest.approx(
        float(np.maximum(0.0, 0.0) + q.calls[np.argmin(np.abs(q.strikes - k_atm))]), rel=0.05
    )
