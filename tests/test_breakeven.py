"""Break-even engine, reparametrisation and targets (SPEC §15 Part 3, M7 addendum).

* ``reparam``: book ↔ break-even round trip to 1e-12 on Table 8.2, the ch. 7 variance dynamics
  with spot correlations of either sign and ``ρ12 ∈ {−0.7, 0, 0.7}``, a near-degenerate
  ``ρ_SX = 0.999``; hand arithmetic of ``α_θ, ω_i, λ_i, χ``; the validation errors.
* ``breakeven``: the first-order closed forms reduce to the flat-curve formulas of
  :mod:`volsto.analytics.bergomi` (eqs. 7.24/7.39, 8.55, 9.21) to 1e-10 and to the sloping-curve
  quadratures (eqs. 8.54, 9.19) to 5e-4 (quadrature-limited); ``σ_0 = sqrt(ξ_0^0)`` by default
  (SPEC §15); the ``sigma_hat`` override scaling (sensitivities and local skew linear, the SV
  skew unchanged — eq. 8.54 at the order-zero VS vol); affinity in ``(λ1, λ2)``, the quadratic form in
  ``(ω1, ω2)`` with the ``ρ_XY`` cross term; the most-likely-path level and sensitivities
  against the first-order forms (``O(ω²)`` gap, growing with ``ν``); the argument validation
  (``x0`` shape, a reused kernel of another maturity); the local branch on the cached 2F LSV
  against ``simulated_breakevens`` (σ_0 identity with ``initial_vol``, ``ssr`` identity with
  ``ATMFVolOfVol``, the market-skew rescale of the spot sensitivity); the acceptance gate
  against simulation on a ``ν ∈ {0.5, 1, 1.74} × T ∈ {3M, 1Y, 2Y}`` grid (numbers in the test
  docstring).
* ``targets``: the SABR reduction round trip and its flags (finite-differences, finite-``T``),
  the marking targets and the policy check (absolute reading from the stored anchor, pillars
  without 3M, the desk ``VoV_SABR`` classification), the ``ssr = 2`` identity of a 1F ``k → 0``
  Bergomi model with its own SABR pillar, the historical targets from a Gaussian surface history
  (the one-month ``σ_0`` flag).

Runtime (fast suite): the gate ≈ 30 s, the LSV local-branch test ≈ 30 s, the rest < 5 s.
"""

from __future__ import annotations

import dataclasses
import itertools
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volsto.analytics.bergomi import (
    atmf_skew_order1,
    atmf_skew_order1_flat,
    ssr_order1,
    ssr_order1_flat,
    vs_vol_of_vol_flat,
)
from volsto.analytics.breakeven import (
    LocalSlope,
    first_order_breakevens,
    kernels,
    local_slope_from_leverage,
    mlp_atmf_variance,
    mlp_sensitivities,
    sigma_0_of,
    simulated_breakevens,
)
from volsto.analytics.reparam import BreakEvenParams, alpha_theta_of, from_breakeven, to_breakeven
from volsto.analytics.smile_dynamics import atmf_vol_of_vol, initial_vol
from volsto.calibration.cache import CacheMissError, LeverageCache, build_market
from volsto.calibration.history import SurfaceHistory
from volsto.calibration.targets import (
    SABR_CURVATURE_H,
    SabrPillar,
    historical_targets,
    marking_targets,
    sabr_reduce,
)
from volsto.config import BergomiParams, CalibrationSpec, SimConfig, load_yaml
from volsto.market import ForwardCurve, ForwardVarianceCurve
from volsto.market.varswap import xi0_curve
from volsto.models import BergomiSV

ROOT = Path(__file__).resolve().parents[1]

#: Bergomi Table 8.2
P82 = BergomiParams(1.74, 0.245, 5.35, 0.28, 0.0, -0.759, -0.487)
BOOK_FIELDS = tuple(f.name for f in dataclasses.fields(BergomiParams))
BE_FIELDS = tuple(f.name for f in dataclasses.fields(BreakEvenParams))


def _max_diff(a: object, b: object, fields: tuple[str, ...]) -> float:
    return max(abs(float(getattr(a, f)) - float(getattr(b, f))) for f in fields)


def _I(x: float) -> float:
    """``(1 - exp(-x)) / x`` (book eq. 7.25)."""
    return 1.0 if x == 0.0 else -math.expm1(-x) / x


@pytest.fixture(scope="module")
def xs(ssvi) -> ForwardVarianceCurve:  # type: ignore[no-untyped-def]
    """The VS strip of the reference SSVI (a sloping curve; built once, about 5 s)."""
    return xi0_curve(ssvi, 10.0)


# --------------------------------------------------------------------------------------------
# 1. reparametrisation
# --------------------------------------------------------------------------------------------


def test_reparam_round_trip() -> None:
    """book -> break-even -> book (and break-even -> book -> break-even) to 1e-12 on Table 8.2,
    the ch. 7 dynamics (ν 1.74, θ 0.245, k1 5.35, k2 0.28) with ``ρ_SX ∈ {−0.6, 0.5}``, ``ρ_SY ∈
    {−0.5, 0.4}``, ``ρ12 ∈ {−0.7, 0, 0.7}`` (the non-PSD combinations are skipped), a one-factor
    limit ``θ = 0`` and two near-degenerate sets with ``|ρ_SX| = 0.999``; hand arithmetic of
    ``α_θ, ω1, ω2, λ1, λ2, χ`` for Table 8.2; the validation errors."""
    sets = [P82, BergomiParams.one_factor(1.0, 1.5, -0.7)]
    for rho12, rsx, rsy in itertools.product((-0.7, 0.0, 0.7), (-0.6, 0.5), (-0.5, 0.4)):
        try:
            sets.append(BergomiParams(1.74, 0.245, 5.35, 0.28, rho12, rsx, rsy))
        except ValueError:
            continue  # correlation matrix not PSD: not an admissible book set
    sets.append(BergomiParams(1.0, 0.3, 4.0, 0.5, 0.0, 0.999, 0.0))
    sets.append(BergomiParams(0.8, 0.5, 2.0, 0.1, 0.3, -0.999, -0.3))
    assert len(sets) >= 10
    assert {p.rho12 for p in sets} >= {-0.7, 0.0, 0.7}
    assert any(p.rho_SX1 > 0 for p in sets) and any(p.rho_SX2 > 0 for p in sets)
    for p in sets:
        be = to_breakeven(p)
        assert be.k1 == p.k1 and be.k2 == p.k2
        back = be.to_book()
        assert _max_diff(back, p, BOOK_FIELDS) < 1e-12, (p, back)
        again = from_breakeven(be.k1, be.k2, be.omega1, be.omega2, be.lambda1, be.lambda2, be.chi)
        assert _max_diff(again, p, BOOK_FIELDS) < 1e-12
        be2 = to_breakeven(back)
        assert _max_diff(be2, be, BE_FIELDS) < 1e-12
        # properties agree with the book side
        assert abs(be.nu - p.nu) < 1e-12 and abs(be.theta - p.theta) < 1e-12
        assert abs(be.rho_SX - p.rho_SX1) < 1e-12 and abs(be.rho_SY - p.rho_SX2) < 1e-12
        assert abs(be.rho_XY - p.rho12) < 1e-12
        assert abs(be.alpha_theta - alpha_theta_of(p.theta, p.rho12)) < 1e-12
        assert -1.0 <= be.chi <= 1.0
    # hand arithmetic, Table 8.2
    be = to_breakeven(P82)
    a = 1.0 / math.sqrt(0.755**2 + 0.245**2 + 2.0 * 0.0 * 0.245 * 0.755)
    assert abs(a - 1.2598316) < 1e-6 and abs(be.alpha_theta - a) < 1e-12
    om1 = 2.0 * 1.74 * a * 0.755
    om2 = 2.0 * 1.74 * a * 0.245
    assert abs(be.omega1 - om1) < 1e-12 and abs(be.omega2 - om2) < 1e-12
    assert abs(om1 - 3.3100815) < 1e-6 and abs(om2 - 1.0741324) < 1e-6
    assert abs(be.lambda1 - (-0.759) * om1) < 1e-12 and abs(be.lambda2 - (-0.487) * om2) < 1e-12
    chi = (0.0 - (-0.759) * (-0.487)) / (math.sqrt(1 - 0.759**2) * math.sqrt(1 - 0.487**2))
    assert abs(be.chi - chi) < 1e-12 and abs(chi - (-0.6500027)) < 1e-6
    assert abs(be.omega1 + be.omega2 - 2.0 * 1.74 * a) < 1e-12  # omega x_t^t loading, eq. 7.34
    # validation errors
    with pytest.raises(ValueError, match="lambda1"):
        BreakEvenParams(5.35, 0.28, om1, om2, -1.1 * om1, be.lambda2, chi)
    with pytest.raises(ValueError, match="lambda2"):
        from_breakeven(5.35, 0.28, om1, om2, be.lambda1, 1.5 * om2, chi)
    with pytest.raises(ValueError, match="chi"):
        BreakEvenParams(5.35, 0.28, om1, om2, be.lambda1, be.lambda2, 1.2)
    with pytest.raises(ValueError, match="chi"):
        from_breakeven(5.35, 0.28, om1, om2, be.lambda1, be.lambda2, -1.0000001)
    with pytest.raises(ValueError, match="k1 and k2"):
        BreakEvenParams(0.0, 0.28, om1, om2, be.lambda1, be.lambda2, chi)
    with pytest.raises(ValueError, match="k1 and k2"):
        from_breakeven(5.35, -0.1, om1, om2, be.lambda1, be.lambda2, chi)
    with pytest.raises(ValueError, match="omega"):
        BreakEvenParams(5.35, 0.28, -0.1, om2, 0.0, be.lambda2, chi)
    with pytest.raises(ValueError, match="omega"):
        BreakEvenParams(5.35, 0.28, 0.0, 0.0, 0.0, 0.0, chi)
    # one factor in the break-even form: omega2 = 0 is admissible, rho_SY reads 0
    one = BreakEvenParams(1.5, 1.5, 1.0, 0.0, -0.7, 0.0, 0.0)
    assert one.theta == 0.0 and one.rho_SY == 0.0 and one.rho_SX == pytest.approx(-0.7)
    assert one.to_book().is_one_factor
    # theta in {0, 1}: the absent factor's correlations are not identifiable; chi is 0 and the
    # round trip returns rho12 = 0 and rho_S = 0 for that factor whatever the input carried
    for p0 in (
        BergomiParams(1.0, 0.0, 1.5, 0.28, 0.3, -0.7, -0.4),
        BergomiParams(1.0, 1.0, 5.0, 0.5, -0.2, 0.3, -0.6),
    ):
        b0 = to_breakeven(p0)
        assert b0.chi == 0.0
        back0 = b0.to_book()
        assert back0.rho12 == 0.0 and back0.nu == pytest.approx(p0.nu) and back0.theta == p0.theta
        if p0.theta == 0.0:
            assert back0.rho_SX1 == pytest.approx(p0.rho_SX1) and back0.rho_SX2 == 0.0
        else:
            assert back0.rho_SX2 == pytest.approx(p0.rho_SX2) and back0.rho_SX1 == 0.0


# --------------------------------------------------------------------------------------------
# 2-4. first-order engine identities
# --------------------------------------------------------------------------------------------


def test_engine_first_order_identities(xs) -> None:  # type: ignore[no-untyped-def]
    """Flat curve: ``SensiX_i = ½ ω_i I(k_i T) σ̂``, ``sqrt(VolVar)/σ̂ = ν_T`` of eq. 7.39,
    ``Skew = S_T`` of eq. 8.55, ``SSR = R_T`` of eq. 9.21, all to 1e-10.  Sloping curve (the
    reference SSVI's VS strip): skew and SSR against the eq. 8.54 / 9.19 quadratures of
    :mod:`volsto.analytics.bergomi` to 5e-4 relative on fine quadratures (the two codes nest the
    double integral differently; measured 1e-6 at 6M and 1e-4 at 2Y); the engine's default
    ``σ_0`` is ``sqrt(ξ_0^0)`` (SPEC §15; 0.2582 on this strip, not the one-month VS vol 0.2609)
    and an explicit ``sigma_0`` rescales the SSR exactly.  The ``sigma_hat`` override scales
    ``SensiX``, ``SensiSpot`` and the local skew linearly (``SensiSpot/SensiX`` invariant) and
    the SV skew by ``(K.σ̂/σ̂)³`` (the recorded convention); with ``skew_market`` on the slope
    the skew is the market skew and the spot sensitivity carries the factor ``Skew^mkt /
    (Skew^SV + Skew^LV)`` at the order-zero level."""
    xi = ForwardVarianceCurve.flat(0.04)
    for p in (P82, P82.replace(nu=0.5), BergomiParams(1.0, 0.4, 3.0, 0.5, 0.3, -0.5, 0.2)):
        be = to_breakeven(p)
        for T in (0.1, 0.25, 1.0, 2.0, 5.0):
            r = first_order_breakevens(p, xi, T)
            sig = r.sigma_hat
            assert abs(sig - 0.2) < 1e-12 and abs(r.sigma_0 - 0.2) < 1e-12
            assert r.method == "first_order" and r.sensi_spot == 0.0
            assert abs(r.sensi_x - 0.5 * be.omega1 * _I(p.k1 * T) * sig) < 1e-10
            assert abs(r.sensi_y - 0.5 * be.omega2 * _I(p.k2 * T) * sig) < 1e-10
            assert abs(math.sqrt(r.vol_var) / sig - float(vs_vol_of_vol_flat(p, T))) < 1e-10
            assert abs(r.vovol - float(vs_vol_of_vol_flat(p, T))) < 1e-10
            assert abs(r.skew - float(atmf_skew_order1_flat(p, T))) < 1e-10
            assert abs(r.ssr - float(ssr_order1_flat(p, T))) < 1e-10
            assert abs(r.spot_vol_covar - (be.rho_SX * r.sensi_x + be.rho_SY * r.sensi_y)) < 1e-14
            assert -1.0 <= r.correl <= 1.0
            if p.rho_SX1 < 0 and p.rho_SX2 < 0:
                assert r.correl < 0.0 and r.spot_vol_covar < 0.0 and r.skew < 0.0
    # sloping curve
    s0 = float(np.sqrt(xs.xi0(0.0)))
    assert sigma_0_of(xs) == pytest.approx(s0, rel=1e-12) and abs(s0 - 0.2582) < 5e-4
    vs_1m = math.sqrt(float(xs.total_variance(1.0 / 12.0)) * 12.0)
    assert abs(vs_1m - 0.2609) < 5e-4 and vs_1m > 1.01 * s0  # the VS vol is not sigma_0
    for T in (0.5, 2.0):
        K = kernels((P82.k1, P82.k2), xs, T, n_quad=128, n_inner=128)
        r = first_order_breakevens(P82, xs, T, kern=K)
        assert r.sigma_0 == pytest.approx(s0, rel=1e-12)
        assert abs(r.skew / atmf_skew_order1(P82, xs, T, n_quad=128) - 1.0) < 5e-4
        assert abs(r.ssr / ssr_order1(P82, xs, T, n_quad=128) - 1.0) < 5e-4
        r_vs = first_order_breakevens(P82, xs, T, sigma_0=vs_1m, kern=K)
        assert r_vs.ssr == pytest.approx(r.ssr * s0 / vs_1m, rel=1e-12)
        assert r_vs.skew == r.skew and r_vs.vol_var == r.vol_var
        assert r.skew_sv == r.skew and r.skew_lv == 0.0
    # sigma_hat override, naked
    K = kernels((P82.k1, P82.k2), xi, 1.0)
    r0 = first_order_breakevens(P82, xi, 1.0, kern=K)
    for c in (0.8, 1.25):
        r = first_order_breakevens(P82, xi, 1.0, sigma_hat=c * K.sigma_hat, kern=K)
        assert r.sigma_hat == pytest.approx(c * K.sigma_hat)
        assert r.sensi_x == pytest.approx(c * r0.sensi_x, rel=1e-12)
        assert r.sensi_y == pytest.approx(c * r0.sensi_y, rel=1e-12)
        assert r.spot_vol_covar == pytest.approx(c * r0.spot_vol_covar, rel=1e-12)
        assert r.vol_var == pytest.approx(c * c * r0.vol_var, rel=1e-12)
        assert r.skew == pytest.approx(r0.skew, rel=1e-12)  # eq. 8.54 stays at the VS vol
    # sigma_hat override with a toy local slope (l2 = 0.8, b = -2: Bachelier-like), flat curve
    toy = LocalSlope(lambda t: np.full_like(t, 0.8), lambda t: np.full_like(t, -2.0), "toy")
    KL = kernels((P82.k1, P82.k2), xi, 1.0, local=toy)
    assert KL.sigma_hat == pytest.approx(0.2 * math.sqrt(0.8), rel=1e-10)
    assert pytest.approx(-2.0 * KL.W_T, rel=1e-10) == KL.B
    assert pytest.approx(-2.0 * 0.5 * KL.W_T, rel=1e-6) == KL.BK  # W_t/W_T = t/T on a flat curve
    l0 = first_order_breakevens(P82, xi, 1.0, local=toy, kern=KL)
    assert l0.sigma_0 == pytest.approx(sigma_0_of(xi, toy)) == pytest.approx(0.2 * math.sqrt(0.8))
    assert l0.sensi_spot == pytest.approx(l0.sigma_0 * KL.sigma_hat * KL.B / (2 * KL.W_T))
    assert l0.skew_lv == pytest.approx(KL.sigma_hat * KL.BK / (2 * KL.W_T))
    assert l0.skew == pytest.approx(l0.skew_sv + l0.skew_lv) and l0.skew_lv < 0 < -l0.skew_sv
    assert l0.ssr == pytest.approx(l0.spot_vol_covar / (l0.sigma_0 * l0.skew))
    for c in (0.8, 1.25):
        r = first_order_breakevens(P82, xi, 1.0, local=toy, sigma_hat=c * KL.sigma_hat, kern=KL)
        assert r.sensi_spot == pytest.approx(c * l0.sensi_spot, rel=1e-12)
        assert r.sensi_x == pytest.approx(c * l0.sensi_x, rel=1e-12)
        assert r.sensi_spot / r.sensi_x == pytest.approx(l0.sensi_spot / l0.sensi_x, rel=1e-12)
        assert r.skew_lv == pytest.approx(c * l0.skew_lv, rel=1e-12)
        assert r.skew_sv == pytest.approx(l0.skew_sv, rel=1e-12)
    # skew_market: the skew is the market skew, the spot sensitivity carries the rescale
    s_mkt = 1.3 * l0.skew
    toy_m = LocalSlope(toy.l2, toy.b, "toy", lambda T: s_mkt)
    lm = first_order_breakevens(P82, xi, 1.0, local=toy_m, kern=KL)
    assert lm.skew == pytest.approx(s_mkt) and lm.skew_sv == l0.skew_sv
    assert lm.skew_lv == pytest.approx(s_mkt - l0.skew_sv)
    assert lm.sensi_spot == pytest.approx(1.3 * l0.sensi_spot, rel=1e-12)
    assert lm.sensi_x == l0.sensi_x and lm.sensi_y == l0.sensi_y
    mm = mlp_sensitivities(P82, xi, 1.0, local=toy_m)
    m0 = mlp_sensitivities(P82, xi, 1.0, local=toy)
    assert mm.sensi_spot == pytest.approx(
        1.3 * m0.sensi_spot, rel=1e-10
    ) and mm.skew == pytest.approx(s_mkt)


def test_engine_affine_in_lambda(xs) -> None:  # type: ignore[no-untyped-def]
    """At fixed ``(k1, k2, ξ_0, T)``: ``SpotVolCovar`` and ``Skew`` are affine in ``(λ1, λ2)``
    (``f(a, b) = f(0, 0) + a [f(1, 0) − f(0, 0)] + b [f(0, 1) − f(0, 0)]`` to 1e-12 on random
    points) and ``VolVar = ¼ σ̂² (ω1² A1² + ω2² A2² + 2 ρ_XY ω1 ω2 A1 A2)`` at fixed spot
    correlations, with the cross term isolated by polarisation and homogeneous of degree 2."""
    rng = np.random.default_rng(5)
    xi = xs
    k1, k2, om1, om2, chi, T = 5.35, 0.28, 3.3, 1.07, -0.65, 1.5
    K = kernels((k1, k2), xi, T)

    def f(a: float, b: float) -> tuple[float, float]:
        r = first_order_breakevens(BreakEvenParams(k1, k2, om1, om2, a, b, chi), xi, T, kern=K)
        return r.spot_vol_covar, r.skew

    f00, f10, f01 = f(0.0, 0.0), f(1.0, 0.0), f(0.0, 1.0)
    assert f00 == (0.0, 0.0)
    for _ in range(25):
        a = float(rng.uniform(-om1, om1))
        b = float(rng.uniform(-om2, om2))
        got = f(a, b)
        for i in range(2):
            pred = f00[i] + a * (f10[i] - f00[i]) + b * (f01[i] - f00[i])
            assert abs(got[i] - pred) < 1e-12, (a, b, i, got[i], pred)
    # quadratic form in (omega1, omega2) at fixed (rho_SX, rho_SY, rho_XY)
    rsx, rsy = -0.759, -0.487
    for rho_xy in (
        0.0,
        0.3,
        -0.1,
    ):  # -0.1 is the most negative admissible with these spot correlations
        chi_ = (rho_xy - rsx * rsy) / (math.sqrt(1 - rsx * rsx) * math.sqrt(1 - rsy * rsy))
        for _ in range(8):
            w1, w2 = (float(x) for x in rng.uniform(0.2, 4.0, 2))
            be = BreakEvenParams(k1, k2, w1, w2, rsx * w1, rsy * w2, chi_)
            assert abs(be.rho_XY - rho_xy) < 1e-12
            r = first_order_breakevens(be, xi, T, kern=K)
            sig = K.sigma_hat
            a1, a2 = float(K.A[0]), float(K.A[1])
            q = 0.25 * sig * sig * (w1 * w1 * a1 * a1 + w2 * w2 * a2 * a2)
            q += 0.25 * sig * sig * 2.0 * rho_xy * w1 * w2 * a1 * a2
            assert abs(r.vol_var - q) < 1e-12
            r10 = first_order_breakevens(
                BreakEvenParams(k1, k2, w1, 0.0, rsx * w1, 0.0, chi_), xi, T, kern=K
            )
            r01 = first_order_breakevens(
                BreakEvenParams(k1, k2, 0.0, w2, 0.0, rsy * w2, chi_), xi, T, kern=K
            )
            cross = r.vol_var - r10.vol_var - r01.vol_var
            assert abs(cross - 2.0 * rho_xy * r.sensi_x * r.sensi_y) < 1e-12
            c = 1.7
            rc = first_order_breakevens(
                BreakEvenParams(k1, k2, c * w1, c * w2, c * rsx * w1, c * rsy * w2, chi_),
                xi,
                T,
                kern=K,
            )
            assert rc.vol_var == pytest.approx(c * c * r.vol_var, rel=1e-12)
            assert rc.spot_vol_covar == pytest.approx(c * r.spot_vol_covar, rel=1e-12)


def test_mlp_matches_first_order_at_small_volvol() -> None:
    """``mlp_sensitivities`` against ``first_order_breakevens`` (flat 20% curve, T = 1Y): the
    largest relative gap over (SensiX, SensiY, SpotVolCovar, VolVar) is 0.3% at ν = 0.1, 2.1% at
    ν = 0.5 and 13% at ν = 1.74 (measured: SensiX 3.5e-5 / 2.5e-3 / 3.4e-2, SensiY 2.0e-3 /
    1.6e-2 / 1.2e-1, VolVar 2.8e-3 / 2.1e-2 / 1.3e-1) — ``O(ω²)`` and growing with ``ν``.  The
    most-likely-path ATMF variance at ``X_0 = 0`` sits below the VS variance by the order-one
    skew shift ``σ̂ − σ_VS ≈ ½ σ_VS² T Skew_T`` (ratio 1.010 at ν = 0.005, 1.020 at ν = 0.01,
    1.20 at ν = 0.1: the ``O(ν²)`` conditioning term ``−½ (Σ λ_i c̃_i)² / W_T`` has a large
    coefficient on a 20% curve) and the naked model is spot-homogeneous."""
    xi = ForwardVarianceCurve.flat(0.04)
    T = 1.0
    gaps: dict[float, float] = {}
    print("\nMLP vs first order, flat 20%, T = 1Y (relative gaps)")
    for nu in (0.1, 0.5, 1.74):
        p = P82.replace(nu=nu)
        fo = first_order_breakevens(p, xi, T)
        ml = mlp_sensitivities(p, xi, T)
        rel = {
            q: abs(getattr(ml, q) / getattr(fo, q) - 1.0)
            for q in ("sensi_x", "sensi_y", "spot_vol_covar", "vol_var")
        }
        gaps[nu] = max(rel.values())
        print(f"  nu {nu:4.2f}: " + "  ".join(f"{q} {v:.2e}" for q, v in rel.items()))
        assert ml.method == "mlp_fd" and ml.skew == pytest.approx(fo.skew, rel=1e-12)
        assert ml.sensi_spot == 0.0 and abs(ml.sigma_hat - 0.2) < 0.03
    assert gaps[0.1] < 0.01, gaps
    assert gaps[0.1] < gaps[0.5] < gaps[1.74], gaps
    # level: below the VS variance by the order-one skew shift, spot-homogeneous
    for nu in (0.005, 0.01):
        p = P82.replace(nu=nu)
        v = mlp_atmf_variance(p, xi, T)
        assert v < 0.04
        shift = math.sqrt(v) - 0.2
        pred = 0.5 * 0.2**2 * T * first_order_breakevens(p, xi, T).skew
        assert pred < 0 and abs(shift / pred - 1.0) < 0.03, (nu, shift, pred)
    for nu in (0.5, 1.74):
        p = P82.replace(nu=nu)
        base = mlp_atmf_variance(p, xi, T)
        assert base < 0.04
        assert mlp_atmf_variance(p, xi, T, d_ln_s0=0.3) == base
        assert mlp_atmf_variance(p, xi, T, d_ln_s0=-0.3) == base
        assert mlp_atmf_variance(p, xi, T, x0=(0.1, 0.0)) > base
        assert mlp_atmf_variance(p, xi, T, x0=(0.0, -0.1)) < base


def test_engine_argument_validation(xs) -> None:  # type: ignore[no-untyped-def]
    """``x0`` must have exactly two entries (a length-1 ``x0`` used to broadcast to both
    factors silently); a reused ``kern`` of another maturity is refused by
    ``first_order_breakevens`` and ``mlp_atmf_variance`` (``T`` enters the spot sensitivity
    while ``K.T`` entered ``J``); ``T <= 0`` is refused by ``kernels``."""
    K1 = kernels((P82.k1, P82.k2), xs, 1.0)
    assert mlp_atmf_variance(P82, xs, 1.0, x0=(0.1, 0.0), kern=K1) != mlp_atmf_variance(
        P82, xs, 1.0, x0=(0.1, 0.1), kern=K1
    )
    with pytest.raises(ValueError, match="two entries"):
        mlp_atmf_variance(P82, xs, 1.0, x0=(0.1,), kern=K1)
    with pytest.raises(ValueError, match="two entries"):
        mlp_atmf_variance(P82, xs, 1.0, x0=(0.1, 0.0, 0.0), kern=K1)
    with pytest.raises(ValueError, match="built for maturity"):
        first_order_breakevens(P82, xs, 2.0, kern=K1)
    with pytest.raises(ValueError, match="built for maturity"):
        mlp_atmf_variance(P82, xs, 2.0, kern=K1)
    assert first_order_breakevens(P82, xs, 1.0, kern=K1).T == 1.0
    with pytest.raises(ValueError, match="positive"):
        kernels((P82.k1, P82.k2), xs, 0.0)


# --------------------------------------------------------------------------------------------
# 5. local branch on the cached 2F LSV
# --------------------------------------------------------------------------------------------


def _cached_2f_lsv():  # type: ignore[no-untyped-def]
    spec = load_yaml(ROOT / "configs" / "studies" / "lsv_reference_2f.yaml", CalibrationSpec)
    spec = dataclasses.replace(
        spec, particle=dataclasses.replace(spec.particle, n_particles=800_000)
    )
    try:
        lsv, _report = LeverageCache(ROOT / "cache").get_or_calibrate(spec, allow_calibrate=False)
    except CacheMissError:
        pytest.skip(
            "cached 2F Table 8.2 leverage (code tag m6, 800000 particles, reference SSVI) not "
            "found under cache/: run the M6 calibration first; tests never calibrate"
        )
    return lsv


def test_local_slope_lsv_first_order(ssvi) -> None:  # type: ignore[no-untyped-def]
    """Cached 2F Table 8.2 LSV on the reference SSVI (never recalibrated here; 40k paths,
    eps 0.05, dt 1/100, seed 3, about 25 s of simulation).  ``local_slope_from_leverage``:
    finite ``l2 = L²(t, F_t)`` in (0.5, 1.2) (measured 0.68 at 1w rising to 0.99 at 3y), ``b =
    ∂ ln L²/∂ ln S < 0`` everywhere (the surface is negatively skewed), ``|b| < 2`` from 6M on
    (measured −1.48, −1.30, −0.51, −0.48 at 6M/1Y/2Y/3Y; −6.2 at 5w and −23 at 1w where the
    short-dated skew is steep).  σ_0: the engine's default ``sigma_0_of(ξ_0, slope)`` equals
    ``initial_vol`` (``L(0, S_0) sqrt(ξ_0^0)`` = 0.2194) and the state covariance's own; the
    naked default is ``sqrt(ξ_0^0)`` = 0.2582.  ``simulated_breakevens``: ``ssr`` equals
    ``ATMFVolOfVol.ssr`` from the same paths to 1e-12 with the same standard error,
    ``vol_var`` its ``variance_rate``, ``spot_vol_covar_se`` comes from the influence samples
    (smaller than the root-sum-square of the partials' errors); an explicit ``sigma_0``
    moves ``SpotVolCovar``, ``VolVar`` and ``ssr`` together (0.22 against 0.2194: 0.1%).
    First order with the market skew attached (``surface=``): ``skew`` is the market skew
    (−0.696 / −0.343 at 3M / 1Y, reproduced by the simulation), ``SensiSpot`` −0.165 / −0.061
    against the simulated −0.157 ± 0.003 / −0.049 ± 0.002 (+5% / +26%; unrescaled −0.191 /
    −0.071, +22% / +47%), ``SpotVolCovar`` within 5% / 7%, ``VolVar`` within 11% / 12% (the
    ν = 1.74 first-order gap of the gate: ``SensiX`` −13% / −30%), SSR 2.38 / 2.04 against the
    simulated 2.49 ± 0.03 / 2.20 ± 0.03.  Asserted: the σ_0 and ssr identities, the SSR
    between 2 and 3 on both sides (the assignment's prior of 1–2 does not hold on this
    surface), the first-order SSR within 15% of the simulation, the spot sensitivity within
    15% at 3M and 35% at 1Y and the unrescaled one overshooting by more than 15%."""
    lsv = _cached_2f_lsv()
    spec = load_yaml(ROOT / "configs" / "studies" / "lsv_reference_2f.yaml", CalibrationSpec)
    _, surface, _ = build_market(spec)
    ls = local_slope_from_leverage(lsv, surface=surface)
    ls0 = local_slope_from_leverage(lsv)
    assert ls.label == "leverage" and ls.skew_market is not None and ls0.skew_market is None
    t = np.array([0.02, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0])
    l2, b = ls.l2(t), ls.b(t)
    assert l2.shape == t.shape and b.shape == t.shape
    assert np.all(np.isfinite(l2)) and np.all((l2 > 0.5) & (l2 < 1.2)), l2
    assert np.all(np.isfinite(b)) and np.all(b < 0), b
    assert np.all(np.abs(b[t >= 0.5]) < 2.0), b
    xi = lsv.kernel.xi0
    s0 = initial_vol(lsv)
    assert sigma_0_of(xi, ls) == pytest.approx(s0, rel=1e-12) and abs(s0 - 0.2194) < 5e-4
    assert sigma_0_of(xi) == pytest.approx(math.sqrt(float(xi.xi0(0.0)))) and sigma_0_of(xi) > s0
    sim = SimConfig(n_paths=40_000, dt_max=1.0 / 100.0, chunk_size=40_000, seed=3)
    t0 = time.time()
    ref = atmf_vol_of_vol(lsv, 0.25, eps=0.05, sim=sim)
    print("\nLSV break-evens by simulation (no recalibration):")
    for T in (0.25, 1.0):
        s = simulated_breakevens(lsv, T, sim=sim)
        assert s.method == "simulation" and s.sigma_0 == pytest.approx(s0, rel=1e-12)
        assert s.ssr == pytest.approx(s.spot_vol_covar / (s.sigma_0 * s.skew), rel=1e-12)
        assert s.ssr_se > 0 and s.spot_vol_covar_se > 0 and s.vol_var_se > 0 and s.skew_se > 0
        rss = math.hypot(s.sensi_spot_se, math.hypot(-0.759 * s.sensi_x_se, -0.487 * s.sensi_y_se))
        assert s.spot_vol_covar_se < rss  # common paths: the influence-sample error is smaller
        if T == 0.25:
            assert s.ssr == pytest.approx(ref.ssr, rel=1e-12) and s.ssr_se == pytest.approx(
                ref.ssr_stderr, rel=1e-12
            )
            assert s.vol_var == pytest.approx(ref.variance_rate, rel=1e-12)
            assert s.sigma_hat == pytest.approx(ref.atmf_vol, rel=1e-12)
            s22 = simulated_breakevens(lsv, T, sim=sim, sigma_0=0.22)
            assert s22.sigma_0 == 0.22 and abs(s22.ssr / s.ssr - 1.0) < 0.01
            assert abs(s22.vol_var / s.vol_var - 1.0) < 0.01 and s22.vol_var != s.vol_var
            assert s22.spot_vol_covar == pytest.approx(
                0.22 * s.sensi_spot / s0 + (s.spot_vol_covar - s.sensi_spot), rel=1e-12
            )
        mkt_skew = float(ssvi.atm_skew(T))
        assert abs(s.skew - mkt_skew) < 3 * s.skew_se + 0.02, (s.skew, mkt_skew)
        r = first_order_breakevens(P82, xi, T, local=ls, sigma_hat=s.sigma_hat)
        r0 = first_order_breakevens(P82, xi, T, local=ls0, sigma_hat=s.sigma_hat)
        naked = first_order_breakevens(P82, xi, T, sigma_hat=s.sigma_hat)
        assert r.sigma_0 == pytest.approx(s0, rel=1e-12) and naked.sigma_0 == sigma_0_of(xi)
        assert r.skew == pytest.approx(mkt_skew) and r.skew_sv == r0.skew_sv
        assert abs(r.skew_sv / naked.skew - 1.0) < 0.5  # same lambda . J with l2 < 1 in V-hat
        assert r0.skew == pytest.approx(r0.skew_sv + r0.skew_lv) and r0.skew_lv < 0
        assert r.sensi_spot < 0.0 and r0.sensi_spot < r.sensi_spot and naked.sensi_spot == 0.0
        print(
            f"  T {T:4.2f}: sim SensiSpot {s.sensi_spot:+.4f} +- {s.sensi_spot_se:.4f}"
            f"  SpotVolCovar {s.spot_vol_covar:+.4f} +- {s.spot_vol_covar_se:.4f}"
            f"  VolVar {s.vol_var:.4f} +- {s.vol_var_se:.4f}  SSR {s.ssr:.3f} +- {s.ssr_se:.3f}"
            f"  skew {s.skew:+.4f} (mkt {mkt_skew:+.4f})\n"
            f"           first order (market skew) SensiSpot {r.sensi_spot:+.4f}"
            f"  SpotVolCovar {r.spot_vol_covar:+.4f}  VolVar {r.vol_var:.4f}  SSR {r.ssr:.3f}"
            f"  | unrescaled SensiSpot {r0.sensi_spot:+.4f} skew {r0.skew:+.4f} SSR {r0.ssr:.3f}"
        )
        tol_spot = 0.15 if T == 0.25 else 0.35
        assert abs(r.sensi_spot / s.sensi_spot - 1.0) < tol_spot, (T, r.sensi_spot, s.sensi_spot)
        assert r0.sensi_spot / s.sensi_spot - 1.0 > 0.15, (T, r0.sensi_spot, s.sensi_spot)
        assert abs(r.spot_vol_covar / s.spot_vol_covar - 1.0) < 0.10
        assert abs(r.vol_var / s.vol_var - 1.0) < 0.20
        assert 2.0 < s.ssr < 3.0 and 2.0 < r.ssr < 3.0, (s.ssr, r.ssr)
        assert abs(r.ssr - s.ssr) < 0.15 * s.ssr + 2.0 * s.ssr_se, (r.ssr, s.ssr)
    print(f"  ({time.time() - t0:.1f} s of simulation)")


# --------------------------------------------------------------------------------------------
# 6. the acceptance gate: first order against simulation
# --------------------------------------------------------------------------------------------


GATE_NUS = (0.5, 1.0, 1.74)
GATE_TS = (0.25, 1.0, 2.0)


def _gate_rows(nus, Ts, sim):  # type: ignore[no-untyped-def]
    xi = ForwardVarianceCurve.flat(0.04)
    fc = ForwardCurve.flat(100.0, 0.0, 0.0)
    rows = []
    for nu in nus:
        p = P82.replace(nu=nu)
        model = BergomiSV(p, xi, fc)
        for T in Ts:
            t0 = time.time()
            s = simulated_breakevens(model, T, sim=sim)
            wall = time.time() - t0
            a = first_order_breakevens(p, xi, T, sigma_hat=s.sigma_hat)
            o1 = first_order_breakevens(p, xi, T)
            for q, se_name, av in (
                ("spot_vol_covar", "spot_vol_covar_se", a.spot_vol_covar),
                ("vol_var", "vol_var_se", a.vol_var),
                ("sensi_x", "sensi_x_se", a.sensi_x),
                ("sensi_y", "sensi_y_se", a.sensi_y),
                ("skew_order1", "skew_se", o1.skew),
            ):
                sv, se = getattr(s, q if not q.startswith("skew") else "skew"), getattr(s, se_name)
                rows.append(
                    {
                        "nu": nu,
                        "T": T,
                        "quantity": q,
                        "sim": sv,
                        "se": se,
                        "analytic": av,
                        "z": (av - sv) / se if se > 0 else np.nan,
                        "relative": av / sv - 1.0,
                        "sigma_hat_sim": s.sigma_hat,
                        "wall_s": wall,
                    }
                )
    return pd.DataFrame(rows)


def _assert_gate(df: pd.DataFrame) -> None:
    for _, r in df.iterrows():
        q, nu = r["quantity"], r["nu"]
        if q in ("spot_vol_covar", "vol_var"):
            if nu == 0.5:
                tol = 2.0 * r["se"] + 0.005 * abs(r["sim"])
                assert abs(r["analytic"] - r["sim"]) <= tol, dict(r)
            elif nu == 1.0:
                assert abs(r["relative"]) < 0.05, dict(r)
            else:
                assert abs(r["relative"]) < 0.20, dict(r)
        elif q == "skew_order1":
            assert abs(r["relative"]) < 0.10, dict(r)


def test_engine_gate_analytic_vs_simulation() -> None:
    """THE ACCEPTANCE GATE.  Table 8.2 with ``ν ∈ {0.5, 1, 1.74}``, flat 20% curve, spot 100,
    zero rates, ``SimConfig(100000 paths, dt 1/100, seed 3)``; analytic = first order with the
    simulated ATMF vol as prefactor.  Measured (analytic/sim − 1; z = (analytic − sim)/se):

    ======  ====  ==============  ========  =======  ========  ======  =======
    ν       T     quantity        sim       se       analytic  z       rel
    ======  ====  ==============  ========  =======  ========  ======  =======
    0.50    0.25  SpotVolCovar    −0.05340  0.00021  −0.05358  −0.83   +0.3%
    0.50    0.25  VolVar          0.00352   0.00003  0.00354   +0.82   +0.7%
    0.50    1.00  SpotVolCovar    −0.02629  0.00016  −0.02611  +1.14   −0.7%
    0.50    1.00  VolVar          0.00101   0.00001  0.00100   −0.32   −0.3%
    0.50    2.00  SpotVolCovar    −0.01811  0.00013  −0.01785  +1.91   −1.4%
    0.50    2.00  VolVar          0.00061   0.00001  0.00061   +0.01   +0.0%
    1.00    0.25  SpotVolCovar    −0.10426  0.00036  −0.10418  +0.23   −0.1%
    1.00    0.25  VolVar          0.01341   0.00009  0.01338   −0.30   −0.2%
    1.00    1.00  SpotVolCovar    −0.05128  0.00027  −0.05020  +4.00   −2.1%
    1.00    1.00  VolVar          0.00378   0.00003  0.00371   −1.89   −1.7%
    1.00    2.00  SpotVolCovar    −0.03512  0.00023  −0.03405  +4.65   −3.1%
    1.00    2.00  VolVar          0.00223   0.00002  0.00222   −0.53   −0.5%
    1.74    0.25  SpotVolCovar    −0.17187  0.00048  −0.16931  +5.30   −1.5%
    1.74    0.25  VolVar          0.03649   0.00021  0.03535   −5.52   −3.1%
    1.74    1.00  SpotVolCovar    −0.08504  0.00037  −0.07941  +15.4   −6.6%
    1.74    1.00  VolVar          0.01002   0.00007  0.00929   −9.93   −7.3%
    1.74    2.00  SpotVolCovar    −0.05790  0.00032  −0.05307  +15.3   −8.4%
    1.74    2.00  VolVar          0.00565   0.00005  0.00539   −5.81   −4.7%
    ======  ====  ==============  ========  =======  ========  ======  =======

    (``SpotVolCovar``'s standard error is that of ``Σ ρ_i p_i`` from the per-path influence
    samples: larger than the root-sum-square of the partials' errors for the naked model, whose
    factor partials are positively correlated and enter with same-sign loadings.)

    Per-factor: ``SensiX`` (short factor) −1.9% / −4.9% at ν = 0.5 (1Y / 2Y), −5.2% / −9.7% at
    ν = 1, −13.5% / −21.5% at ν = 1.74; ``SensiY`` within +0.6% … +1.7% throughout.  Skew, the
    mixing check: the plain order-one skew (no ``σ̂`` override) is within 10% of the simulated
    ±1% strike difference at every node — 0.6% / 2.2% / 2.5% at ν = 0.5, 1.9% / 3.9% / 4.4% at
    ν = 1, 5.6% / 7.2% / 7.1% at ν = 1.74 (the true skew is *smaller* than the order one; the
    ``sigma_hat`` override leaves it unchanged — a ``(K.σ̂/σ̂)³`` rescale, tried and removed,
    moved it the wrong way by +5–10% / +16–29% / +47–84%).  Gate: at ν = 0.5 SpotVolCovar and
    VolVar within 2 se + 0.5% at T ≤ 2Y, 5% at ν = 1, 20% at ν = 1.74; the
    order-one skew within 10%.  Wall clock ≈ 27 s; no leverage function is calibrated."""
    sim = SimConfig(n_paths=100_000, dt_max=1.0 / 100.0, chunk_size=100_000, seed=3)
    t0 = time.time()
    df = _gate_rows(GATE_NUS, GATE_TS, sim)
    wall = time.time() - t0
    with pd.option_context("display.width", 160, "display.max_rows", 200):
        print(f"\nGate: first order vs simulation ({wall:.1f} s wall, no recalibration)")
        print(df[["nu", "T", "quantity", "sim", "se", "analytic", "z", "relative"]].to_string())
    assert len(df) == len(GATE_NUS) * len(GATE_TS) * 5
    assert np.all(df["se"] > 0)
    _assert_gate(df)
    # the discrepancy grows with nu on the short factor and on the order-one skew
    sx = df[df["quantity"] == "sensi_x"].groupby("nu")["relative"].apply(lambda s: s.abs().max())
    assert sx[0.5] < sx[1.0] < sx[1.74] and sx[1.74] < 0.25
    sk = (
        df[df["quantity"] == "skew_order1"].groupby("nu")["relative"].apply(lambda s: s.abs().max())
    )
    assert sk[0.5] < sk[1.0] < sk[1.74] and sk[1.74] < 0.10
    # wall clock is reported above, not asserted: a machine sleeping mid-test would fail any bound


# --------------------------------------------------------------------------------------------
# 7-9. targets
# --------------------------------------------------------------------------------------------


class _QuadSurface:
    """``σ̂(k, T) = atf + skew k + ½ curv k²`` for every ``T`` (no analytic ``atm_skew``: the
    reduction uses finite differences)."""

    def __init__(self, atf: float, skew: float, curv: float, max_maturity: float = 5.0) -> None:
        self.atf, self.skew, self.curv, self.max_maturity = atf, skew, curv, max_maturity

    @classmethod
    def from_sabr(cls, atf: float, nu: float, rho: float) -> _QuadSurface:
        return cls(atf, 0.5 * rho * nu, (2.0 - 3.0 * rho * rho) * nu * nu / (6.0 * atf))

    def implied_vol_k(self, k, T):  # type: ignore[no-untyped-def]
        k_, T_ = np.broadcast_arrays(np.asarray(k, dtype=np.float64), np.asarray(T, dtype=float))
        return self.atf + self.skew * k_ + 0.5 * self.curv * k_ * k_ + 0.0 * T_

    def atm_vol(self, T):  # type: ignore[no-untyped-def]
        return np.full_like(np.asarray(T, dtype=np.float64), self.atf)


def test_sabr_reduction() -> None:
    """``sabr_reduce`` on a quadratic smile built from ``(atf, ν_SABR, ρ_SABR)`` recovers ``ν``
    and ``ρ`` to 1e-6 (the central differences of a quadratic are exact) with ``skew_sabr ==
    skew``; the radicand ``6 skew² + 3 atf curv`` is negative only for a *concave* smile,
    ``curv < −2 skew²/atf`` (a flat-curvature smile has ``ν = sqrt(6) |skew|`` whatever the
    skew): NaN and the flag; ``−2 skew²/atf < curv < −⅔ skew²/atf`` gives ``|ρ| > 1``, clipped
    with the flag, and ``skew_sabr ≠ skew`` then.  Flags are never silent: a surface without an
    analytic ``atm_skew`` (this quadratic one) is flagged as finite-difference, ``ν_SABR² T >
    1`` is flagged as beyond the ``T → 0`` reduction; the half-width ``h`` (default 1e-3) is
    recorded on the pillar.  The reference SSVI (analytic ``atm_skew``, ``ν² T`` 0.6–0.8) gets
    no flag at any pillar 1M–5Y."""
    fd = "no analytic atm_skew"
    for atf, nu, rho in itertools.product((0.15, 0.25), (0.3, 0.8, 1.5), (-0.9, -0.5, 0.0, 0.6)):
        s = _QuadSurface.from_sabr(atf, nu, rho)
        for T in (0.25, 2.0):
            r = sabr_reduce(s, T)
            assert isinstance(r, SabrPillar) and r.T == T and r.h == SABR_CURVATURE_H == 1e-3
            assert fd in r.flags[0] and len(r.flags) == 1 + (nu * nu * T > 1.0), r.flags
            if nu * nu * T > 1.0:
                assert "finite-T" in r.flags[1]
            assert abs(r.atf - atf) < 1e-12
            assert abs(r.nu_sabr - nu) < 1e-6 and abs(r.rho_sabr - rho) < 1e-6, (atf, nu, rho, r)
            assert abs(r.skew - s.skew) < 1e-10 and abs(r.curv - s.curv) < 1e-6
            assert r.skew_sabr == pytest.approx(s.skew, abs=1e-10)
    # flat curvature: never a negative radicand
    flat = sabr_reduce(_QuadSurface(0.2, -0.5, 0.0), 1.0)
    assert len(flat.flags) == 2 and fd in flat.flags[0] and "finite-T" in flat.flags[1]
    assert flat.nu_sabr == pytest.approx(math.sqrt(6) * 0.5)
    assert flat.rho_sabr == pytest.approx(-2.0 / math.sqrt(6))
    # concave smile with a small skew: negative radicand
    bad = sabr_reduce(_QuadSurface(0.2, -0.05, -0.5), 1.0)
    assert math.isnan(bad.nu_sabr) and math.isnan(bad.rho_sabr) and math.isnan(bad.skew_sabr)
    assert len(bad.flags) == 2 and "negative radicand" in bad.flags[1]
    assert 6 * 0.05**2 + 3 * 0.2 * (-0.5) < 0
    # |rho| > 1: clipped and flagged, skew_sabr no longer the market skew
    clip = sabr_reduce(_QuadSurface(0.2, -0.3, -0.5), 1.0)
    rad = 6 * 0.3**2 + 3 * 0.2 * (-0.5)
    assert rad > 0 and 2 * 0.3 / math.sqrt(rad) > 1
    assert clip.nu_sabr == pytest.approx(math.sqrt(rad)) and clip.rho_sabr == -1.0
    assert len(clip.flags) == 2 and "clipped" in clip.flags[1]
    assert clip.skew_sabr == pytest.approx(-0.5 * math.sqrt(rad)) and clip.skew_sabr != clip.skew
    # the finite-difference half-width is an explicit argument, recorded on the pillar
    r05 = sabr_reduce(_QuadSurface.from_sabr(0.2, 1.0, -0.6), 1.0, h=0.05)
    assert r05.nu_sabr == pytest.approx(1.0, abs=1e-6) and r05.h == 0.05


def test_marking_targets_and_policy_check(ssvi) -> None:  # type: ignore[no-untyped-def]
    """Reference SSVI, pillars 1M…5Y: ``vovol_target = ½ ssr atf ν_SABR A`` exactly,
    ``SpotVolCovar_target = ssr σ_0 Skew_SABR``, ``VolVar = vovol²``, ``σ_0 = atm_vol(1M)``,
    ``A(3M) = 1`` and ``A = (atf(3M)/atf(T))^p``; ``policy_check`` reads ``"absolute"`` at
    ``p = 1`` with ``mismatch_lognormal = 1/atf`` between 4 and 6 here (the lognormal reading
    differs by more than a factor 3 at every pillar — the flag the owner asked for), also on a
    pillar set without 3M (the anchor is stored, not re-interpolated), ``"neither"`` away from
    3M at ``p = 0.5``, and classifies a desk ``VoV_SABR`` (``ν_SABR`` → lognormal, ``atf ν_SABR``
    → absolute, ``2 ν_SABR`` → omega, else unknown); ``ssr_target`` as a mapping (linear
    interpolation in T) and as a callable; ``anchor_power = 0`` gives ``A = 1``; pillars beyond
    ``max_maturity`` are dropped with a flag; no SABR flag on this surface."""
    pillars = (1.0 / 12.0, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0)
    ts = marking_targets(ssvi, pillars, ssr_target=1.0, anchor_power=1.0)
    assert ts.mode == "marking" and ts.anchor_power == 1.0
    assert np.allclose(ts.pillars, sorted(pillars)) and len(ts.sabr) == 7
    assert ts.flags == (), ts.flags
    atf3 = float(ssvi.atm_vol(0.25))
    assert ts.sigma_0 == float(ssvi.atm_vol(1.0 / 12.0)) == pytest.approx(0.22, abs=1e-6)
    assert ts.atf_anchor == atf3
    assert all(s.h == SABR_CURVATURE_H for s in ts.sabr)
    for i, T in enumerate(ts.pillars):
        s = ts.sabr[i]
        assert s.T == T and s.flags == () and abs(s.atf - float(ssvi.atm_vol(T))) < 1e-12
        assert s.skew == pytest.approx(float(ssvi.atm_skew(T))) and s.skew < 0
        assert s.skew_sabr == pytest.approx(s.skew, rel=1e-12)  # no clipping on this surface
        anchor = (atf3 / s.atf) ** 1.0
        assert ts.anchor[i] == pytest.approx(anchor, rel=1e-14)
        assert ts.atf[i] == s.atf and ts.skew_target[i] == s.skew_sabr and ts.ssr_target[i] == 1.0
        assert ts.vovol[i] == pytest.approx(0.5 * 1.0 * s.atf * s.nu_sabr * anchor, rel=1e-14)
        assert ts.spot_vol_covar[i] == pytest.approx(1.0 * ts.sigma_0 * s.skew_sabr, rel=1e-14)
        assert ts.vol_var[i] == pytest.approx(ts.vovol[i] ** 2, rel=1e-14)
        assert ts.spot_vol_covar_se[i] == 0.0 and ts.vol_var_se[i] == 0.0
    assert ts.anchor[1] == pytest.approx(1.0) and ts.pillars[1] == 0.25
    frame = ts.frame()
    assert len(frame) == 7 and {"nu_sabr", "rho_sabr", "anchor", "vovol_target"} <= set(frame)
    # policy check
    pc = ts.policy_check()
    assert list(pc["T"]) == list(ts.pillars)
    assert (pc["reading"] == "absolute").all(), pc
    assert np.allclose(pc["vovol_target_at_ssr_1"], pc["policy_absolute"], rtol=1e-12)
    assert np.allclose(pc["mismatch_lognormal"], 1.0 / ts.atf, rtol=1e-12)
    assert np.all((pc["mismatch_lognormal"] > 4.0) & (pc["mismatch_lognormal"] < 6.0)), pc
    ratio = pc["policy_lognormal"] / pc["vovol_target_at_ssr_1"]
    assert np.all(ratio > 3.0), pc  # the lognormal reading would be off by more than 3x
    # ssr as a mapping (interpolated in T) and as a callable
    tm = marking_targets(ssvi, pillars, ssr_target={0.25: 1.5, 5.0: 0.8})
    exp = np.interp(ts.pillars, [0.25, 5.0], [1.5, 0.8])
    assert np.allclose(tm.ssr_target, exp)
    assert np.allclose(tm.vovol, ts.vovol * exp) and np.allclose(
        tm.spot_vol_covar, ts.spot_vol_covar * exp
    )
    assert np.allclose(tm.vol_var, ts.vol_var * exp * exp)
    tc = marking_targets(ssvi, pillars, ssr_target=lambda T: 1.0 + 0.1 * T)
    assert np.allclose(tc.ssr_target, 1.0 + 0.1 * ts.pillars)
    assert np.allclose(tc.spot_vol_covar, ts.spot_vol_covar * (1.0 + 0.1 * ts.pillars))
    pcm = tm.policy_check()  # vovol / ssr recovers the unit-ssr value whatever the dial
    assert (pcm["reading"] == "absolute").all()
    # pillars without 3M: the stored anchor keeps the reading (was 'neither' by re-interpolation)
    t_no3m = marking_targets(ssvi, (0.5, 1.0, 2.0))
    assert t_no3m.atf_anchor == atf3 and np.allclose(t_no3m.anchor, atf3 / t_no3m.atf)
    pn = t_no3m.policy_check()
    assert (pn["reading"] == "absolute").all(), pn
    assert np.allclose(pn["mismatch_lognormal"], 1.0 / t_no3m.atf, rtol=1e-12)
    # anchor_power 0.5: the formula is not the p = 1 policy away from 3M
    ph = marking_targets(ssvi, pillars, anchor_power=0.5).policy_check()
    assert ph.loc[ph["T"] == 0.25, "reading"].item() == "absolute"
    assert (ph.loc[ph["T"] != 0.25, "reading"] == "neither").all()
    # the desk's VoV_SABR classified against nu_sabr
    nu_s = {float(x.T): x.nu_sabr for x in ts.sabr}
    for scale, kind in ((1.0, "lognormal"), (2.0, "omega")):
        pv = ts.policy_check(vov_sabr={T: scale * v for T, v in nu_s.items()})
        assert (pv["vov_reading"] == kind).all(), pv
        assert np.allclose(pv["vov_over_nu"], scale)
    pa = ts.policy_check(
        vov_sabr=lambda T: float(ssvi.atm_vol(T)) * nu_s[min(nu_s, key=lambda x: abs(x - T))]
    )
    assert (pa["vov_reading"] == "absolute").all(), pa
    pu = ts.policy_check(vov_sabr=lambda T: 1.4 * nu_s[min(nu_s, key=lambda x: abs(x - T))])
    assert (pu["vov_reading"] == "unknown").all(), pu
    assert "vov_sabr" not in pc.columns
    # anchoring power
    t0 = marking_targets(ssvi, pillars, anchor_power=0.0)
    assert np.all(t0.anchor == 1.0) and np.allclose(t0.vovol, 0.5 * t0.atf * ts.frame()["nu_sabr"])
    pc0 = t0.policy_check()
    assert pc0.loc[pc0["T"] == 0.25, "reading"].item() == "absolute"
    assert (pc0.loc[pc0["T"] != 0.25, "reading"] == "neither").all()
    t2 = marking_targets(ssvi, pillars, anchor_power=2.0)
    assert np.allclose(t2.anchor, (atf3 / t2.atf) ** 2) and np.allclose(
        t2.vovol, ts.vovol * ts.anchor
    )
    # explicit sigma_0 and pillars beyond max_maturity
    tsig = marking_targets(ssvi, pillars, sigma_0=0.3)
    assert tsig.sigma_0 == 0.3 and np.allclose(
        tsig.spot_vol_covar, ts.spot_vol_covar * 0.3 / ts.sigma_0
    )
    td = marking_targets(ssvi, (1.0, 0.5, 12.0, 15.0))
    assert td.pillars.tolist() == [0.5, 1.0]
    assert len(td.flags) == 1 and "beyond max_maturity" in td.flags[0] and "12.0" in td.flags[0]
    with pytest.raises(ValueError, match="no pillar"):
        marking_targets(ssvi, (12.0,))


def test_marking_targets_sabr_consistency_at_ssr_2() -> None:
    """A 1F Bergomi model with ``k → 0`` is lognormal SABR with ``ν_Bergomi = ν_SABR``: on a
    quadratic pillar built from ``(atf, ν, ρ)`` at ``T = 1e-3`` the engine gives ``vovol = ν``,
    ``Skew = ½ ρ ν`` and ``SSR = 2`` (the ``T → 0`` SSR of every diffusive model), and the
    marking targets at ``ssr = 2`` (``anchor_power = 0``) equal the model's ``VolVar`` and
    ``SpotVolCovar`` to 1e-5 (the ``k T = 1e-6`` residual of the finite ``k``) — the ½ of
    ``vovol_target`` is the ``ssr = 2`` normalisation, not a ν/ω conversion; at ``ssr = 1`` both
    targets are half the smile's diffusive dynamics."""
    T = 1e-3
    for atf, nu, rho in ((0.2, 1.0, -0.6), (0.25, 1.74, -0.7), (0.15, 0.5, 0.3)):
        q = _QuadSurface.from_sabr(atf, nu, rho)
        p = BergomiParams.one_factor(2.0 * nu, 1e-3, rho)
        r = first_order_breakevens(p, ForwardVarianceCurve.flat(atf * atf), T)
        assert r.vovol == pytest.approx(nu, rel=1e-6) and r.ssr == pytest.approx(2.0, rel=1e-6)
        assert r.skew == pytest.approx(0.5 * rho * nu, rel=1e-6)
        t2 = marking_targets(q, (T,), ssr_target=2.0, anchor_power=0.0, sigma_0=atf)
        assert t2.sabr[0].nu_sabr == pytest.approx(nu, rel=1e-6)
        assert t2.vol_var[0] == pytest.approx(r.vol_var, rel=1e-5)
        assert t2.spot_vol_covar[0] == pytest.approx(r.spot_vol_covar, rel=1e-5)
        assert t2.skew_target[0] == pytest.approx(r.skew, rel=1e-5)
        t1 = marking_targets(q, (T,), ssr_target=1.0, anchor_power=0.0, sigma_0=atf)
        assert t1.vol_var[0] == pytest.approx(0.25 * r.vol_var, rel=1e-5)
        assert t1.spot_vol_covar[0] == pytest.approx(0.5 * r.spot_vol_covar, rel=1e-5)


def _gaussian_history(seed: int, n_days: int = 800, pillars=(0.25, 1.0, 2.0)):  # type: ignore[no-untyped-def]
    """``ln vs_vol`` random walks with annual vols (0.8, 0.5, 0.4) and cross correlation 0.7;
    ATM vol ``0.35 + (vs_vol − 0.2) + β (ln S − ln S_0)`` with β per pillar, constant skew; spot a
    20% random walk.  Returns the history, the generator's vol of vol and ``SSR = β / skew``."""
    rng = np.random.default_rng(seed)
    p = np.asarray(pillars)
    vols = np.array([0.8, 0.5, 0.4]) / np.sqrt(252)
    corr = np.full((3, 3), 0.7) + 0.3 * np.eye(3)
    z = rng.multivariate_normal(np.zeros(3), corr, size=n_days)
    ln_vs = np.log(0.2) + np.cumsum(z * vols, axis=0)
    d_ln_s = 0.2 / np.sqrt(252) * rng.normal(size=n_days)
    ln_s = np.log(100.0) + np.cumsum(d_ln_s)
    beta = np.array([-0.3, -0.15, -0.1])
    skew = np.array([-0.8, -0.4, -0.25])
    atm = 0.35 + (np.exp(ln_vs) - 0.2) + (ln_s - ln_s[0])[:, None] * beta[None, :]
    assert atm.min() > 0.05, "generator produced a non-positive ATM vol: change the seed"
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    hist = SurfaceHistory.from_arrays(
        dates, p, np.exp(ln_vs), atm, np.tile(skew, (n_days, 1)), ln_s
    )
    return hist, vols * np.sqrt(252), beta / skew


def test_historical_targets() -> None:
    """On a Gaussian surface history (pillars 3M/1Y/2Y): ``vol_var = (atf volvol_hist)²`` and
    ``spot_vol_covar = SSR_hist σ_0 skew`` at the last date with the estimators' delta-method
    standard errors (``2 vovol × atf se_volvol``, ``|se_ssr σ_0 skew|``), ``σ_0`` the ATM vol
    interpolated at one month across the pillars (flat below 3M here, hence the 3M value) with a
    flag saying so — and no such flag when one month is a pillar —, the pillar subset and the
    end date honoured, the flag naming the windows and the end date; ``policy_check`` refuses
    historical targets."""
    hist, volvol_true, ssr_true = _gaussian_history(3)
    ts = historical_targets(hist, window_vol=250, window_ssr=60)
    assert ts.mode == "historical" and ts.pillars.tolist() == [0.25, 1.0, 2.0] and ts.sabr == ()
    e = hist.n_dates - 1
    atm = hist.atm_vol.to_numpy()[e]
    skw = hist.skew.to_numpy()[e]
    assert ts.sigma_0 == atm[0]
    assert math.isnan(ts.anchor_power)
    assert np.all(ts.anchor == 1.0)
    for j, T in enumerate(ts.pillars):
        vv = hist.volvol_hist(float(T), 250)
        sr = hist.ssr_hist(float(T), 60)
        assert ts.atf[j] == atm[j] and ts.skew_target[j] == skw[j] == pytest.approx(
            [-0.8, -0.4, -0.25][j]
        )
        assert ts.vovol[j] == pytest.approx(atm[j] * vv.volvol, rel=1e-12)
        assert ts.vol_var[j] == pytest.approx((atm[j] * vv.volvol) ** 2, rel=1e-12)
        assert ts.vol_var_se[j] == pytest.approx(
            2.0 * atm[j] * vv.volvol * atm[j] * vv.se, rel=1e-12
        )
        assert ts.ssr_target[j] == sr.ssr
        assert ts.spot_vol_covar[j] == pytest.approx(sr.ssr * ts.sigma_0 * skw[j], rel=1e-12)
        assert ts.spot_vol_covar_se[j] == pytest.approx(abs(sr.se * ts.sigma_0 * skw[j]), rel=1e-12)
        assert vv.se > 0 and sr.se > 0
    assert len(ts.flags) == 2
    assert "windows vol 250 / ssr 60" in ts.flags[0] and str(hist.dates[-1].date()) in ts.flags[0]
    assert "sigma_0" in ts.flags[1] and "no one-month pillar" in ts.flags[1]
    h1m, _, _ = _gaussian_history(3, pillars=(1.0 / 12.0, 0.25, 1.0))
    t1m = historical_targets(h1m, window_vol=250, window_ssr=60)
    assert len(t1m.flags) == 1 and t1m.sigma_0 == h1m.atm_vol.to_numpy()[-1][0]
    frame = ts.frame()
    assert len(frame) == 3 and "nu_sabr" not in frame and np.all(frame["vol_var_se"] > 0)
    # the estimators themselves recover the generator on a long window (3 se)
    tl = historical_targets(hist, window_vol=750, window_ssr=750)
    for j in range(3):
        vv = hist.volvol_hist(float(ts.pillars[j]), 750)
        assert abs(tl.vovol[j] / atm[j] - volvol_true[j]) < 3 * vv.se
        sr = hist.ssr_hist(float(ts.pillars[j]), 750)
        assert abs(tl.ssr_target[j] - ssr_true[j]) < 3 * sr.se
    # pillar subset (sorted), explicit sigma_0, end date
    sub = historical_targets(hist, pillars=[2.0, 0.25], window_vol=250, window_ssr=60, sigma_0=0.3)
    assert sub.pillars.tolist() == [0.25, 2.0] and sub.sigma_0 == 0.3 and len(sub.flags) == 1
    assert sub.vol_var[1] == ts.vol_var[2]
    assert sub.spot_vol_covar[1] == pytest.approx(ts.spot_vol_covar[2] * 0.3 / ts.sigma_0)
    mid = hist.dates[500]
    tm = historical_targets(hist, end=mid, window_vol=250, window_ssr=60)
    assert str(mid.date()) in tm.flags[0]
    assert tm.atf[1] == hist.atm_vol.to_numpy()[500][1]
    assert tm.ssr_target[1] == hist.ssr_hist(1.0, 60, end=mid).ssr
    with pytest.raises(KeyError):
        historical_targets(hist, pillars=[0.5])
    with pytest.raises(ValueError, match="marking"):
        ts.policy_check()
