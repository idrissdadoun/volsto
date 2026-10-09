"""Cross-dependent volatility prototype (``volsto/multi/cdv.py``, ``docs/cross_dependent_vol.md``):

* ``β = 0`` is the local correlation model of M12, bit for bit;
* the per-name normalisation of a constant is 1;
* with ``β > 0`` each name still reprices its own smile (C2 at ``β > 0``);
* a synthetic truth — a cross-dependent world with a known constant ``λ`` — is recovered by the
  calibration when ``β`` is the truth's, and is not a constant-``λ`` world when ``β = 0``;
* the pure helpers of ``scripts/cdv_scan.py`` and ``scripts/cdv_stratified.py`` (the owner's
  requests (e) and (f) of 2026-10-09): the basket's second moment by region, the attribution of
  the forward's move, the rules of the stratified test, its table and its report.

The worlds, grids and settings are those of ``tests/test_local_correlation.py`` (W5 at 3m).
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import test_local_correlation as t
from scipy.integrate import quad
from scipy.special import ndtr

from volsto.calibration import local_correlation as lcal
from volsto.config import LocalCorrelationConfig, ParticleConfig, SimConfig, SSVIConfig
from volsto.market.bs import black_price, black_vega, implied_vol
from volsto.market.curves import ForwardCurve
from volsto.market.surface import surface_from_config
from volsto.multi.analytics import strip_second_moment
from volsto.multi.cdv import CrossDependence, _name_scales, calibrate_cdv, simulate_cdv
from volsto.multi.family import CorrelationFamily

HORIZON = 0.25
PARTICLE = ParticleConfig(n_particles=50_000, horizon=HORIZON, seed=12345)
SIM = SimConfig(n_paths=100_000, dt_max=t.DAILY, seed=2024, chunk_size=20_000)


def world() -> tuple[list, CorrelationFamily, object, object, object]:  # type: ignore[type-arg]
    models = t.w5_models(HORIZON)
    surface, lv = t.w5_target(HORIZON)
    return models, CorrelationFamily.equi(5), t.w5_basket(models), surface, lv


def test_beta_zero_is_the_local_correlation_model() -> None:
    """``β = 0``: ``λ``, ``λ*``, the clipped masses, the trusted range and the final particle
    cloud equal ``calibrate_local_correlation``'s bit for bit; the scales are exactly 1."""
    models, family, basket, surface, lv = world()
    lc = LocalCorrelationConfig(particle=PARTICLE)
    ref = lcal.calibrate_local_correlation(models, family, basket, surface, lv, PARTICLE, SIM, lc)  # type: ignore[arg-type]
    got = calibrate_cdv(models, family, basket, surface, lv, PARTICLE, SIM, lc, CrossDependence(0.0))  # type: ignore[arg-type]
    np.testing.assert_array_equal(got.lam, ref.lam.values)
    np.testing.assert_array_equal(got.lam_star, ref.lambda_star)
    np.testing.assert_array_equal(got.final_log_spot, ref.final_log_spot)
    np.testing.assert_array_equal(got.clipped_low, ref.clipped_low)
    np.testing.assert_array_equal(got.clipped_high, ref.clipped_high)
    np.testing.assert_array_equal(got.inner_high, ref.clipped_high_inner)
    np.testing.assert_array_equal(got.q_lo, ref.q_lo)
    np.testing.assert_array_equal(got.lambda_mean, ref.lambda_mean)
    assert np.all(got.scale == 1.0)
    with pytest.raises(ValueError):
        CrossDependence(1.0, g_min=1.5)
    with pytest.raises(ValueError):
        calibrate_cdv(models, family, basket, surface, lv, dataclasses.replace(PARTICLE, second_pass=True),  # type: ignore[arg-type]
                      SIM, lc, CrossDependence(0.0))  # fmt: skip


def test_normalisation_of_a_constant_is_one() -> None:
    """``E[g² | k_i]`` of a constant ``g² = c`` is ``c`` (the regression and its tails), so the
    scale is ``1/√c``: to 1e-12 for ``c = 1``, to 1e-10 for another constant (the log-quadratic
    tail extrapolation rounds at 1e-12 there)."""
    rng = np.random.default_rng(3)
    ls = 0.1 * rng.standard_normal((20_000, 3)) + np.array([0.0, 0.01, -0.02])
    grid = np.linspace(-2.0, 2.0, 1601)
    for c in (1.0, 2.25):
        scales = _name_scales(ls, np.zeros(3), np.full(ls.shape[0], c), grid, PARTICLE)
        np.testing.assert_allclose(scales, 1.0 / np.sqrt(c), rtol=1e-12 if c == 1.0 else 1e-10)


@pytest.fixture(scope="module")
def beta3() -> tuple[object, np.ndarray, np.ndarray, np.ndarray]:
    """W5 at 3m calibrated with ``β = 3`` and with ``β = 0``, both simulated on the pricing
    seed: the calibration, and the terminal log-spots under each."""
    models, family, basket, surface, lv = world()
    lc = LocalCorrelationConfig(particle=PARTICLE)
    out = []
    for beta in (3.0, 0.0):
        res = calibrate_cdv(models, family, basket, surface, lv, PARTICLE, SIM, lc, CrossDependence(beta))  # type: ignore[arg-type]
        ls, kb = simulate_cdv(res, models, family, basket, SIM, [HORIZON])  # type: ignore[arg-type]
        out.append((res, ls[0], kb[0]))
    return out[0][0], out[0][1], out[1][1], out[0][2]


def test_names_keep_their_smiles(beta3: tuple[object, np.ndarray, np.ndarray, np.ndarray]) -> None:
    """C2 at ``β = 3``: every name's vanilla prices at 3m under the cross-dependent model are
    those of its own surface — within 0.25 vp plus four standard errors of the analytic smile,
    and within 0.15 vp plus four paired errors of the same name under ``β = 0`` on the same
    normals.  The scales are not trivial (they move by more than 5 %)."""
    res, ls3, ls0, _ = beta3
    assert res.scale.min() < 0.95 and res.scale.max() > 1.05  # type: ignore[attr-defined]
    surfaces = t.w5_surfaces()
    worst_abs, worst_pair = 0.0, 0.0
    for i, surface in enumerate(surfaces):
        atm = float(surface.atm_vol(HORIZON))
        for m in (-1.0, 0.0, 1.0):
            k = m * atm * np.sqrt(HORIZON)
            cp = 1.0 if k >= 0 else -1.0
            target = float(surface.implied_vol_k(k, HORIZON))
            pay3 = np.maximum(cp * (np.exp(ls3[:, i]) - np.exp(k)), 0.0)
            pay0 = np.maximum(cp * (np.exp(ls0[:, i]) - np.exp(k)), 0.0)
            vega = float(black_vega(1.0, np.exp(k), HORIZON, target))
            pair3 = 0.5 * (pay3[0::2] + pay3[1::2])
            diff = 0.5 * ((pay3 - pay0)[0::2] + (pay3 - pay0)[1::2])
            vol3 = float(implied_vol(pair3.mean(), 1.0, np.exp(k), HORIZON, cp))
            se = pair3.std(ddof=1) / np.sqrt(pair3.size) / vega
            err = 100 * (vol3 - target)
            assert abs(err) <= 0.25 + 400 * se, (i, m, err, 100 * se)
            pair_err = 100 * diff.mean() / vega
            pair_se = 100 * diff.std(ddof=1) / np.sqrt(diff.size) / vega
            assert abs(pair_err) <= 0.15 + 4 * pair_se, (i, m, pair_err, pair_se)
            worst_abs, worst_pair = max(worst_abs, abs(err)), max(worst_pair, abs(pair_err))
            assert float(black_price(1.0, np.exp(k), HORIZON, target, cp)) > 0
    print(
        f"\nC2 at beta = 3: worst error against the smile {worst_abs:.3f} vp, against beta = 0 {worst_pair:.3f} vp"
    )


def test_index_smile_and_correlation_at_beta_3(
    beta3: tuple[object, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    """With ``β = 3`` the basket still reprices the index target at 3m (0.30 vp plus four
    standard errors inside ±1.5 sd), and the calibrated ``λ`` asks for less correlation on the
    downside than at ``β = 0``: the names' volatilities carry part of the index skew."""
    res, _, _, kb = beta3
    models, family, basket, surface, lv = world()
    atm = float(surface.atm_vol(HORIZON))  # type: ignore[attr-defined]
    level = np.exp(kb)
    for m in (-1.5, -1.0, 0.0, 1.0, 1.5):
        k = m * atm * np.sqrt(HORIZON)
        cp = 1.0 if k >= 0 else -1.0
        pay = np.maximum(cp * (level - np.exp(k)), 0.0)
        pair = 0.5 * (pay[0::2] + pay[1::2])
        target = float(surface.implied_vol_k(k, HORIZON))  # type: ignore[attr-defined]
        vol = float(implied_vol(pair.mean(), 1.0, np.exp(k), HORIZON, cp))
        se = (
            pair.std(ddof=1)
            / np.sqrt(pair.size)
            / float(black_vega(1.0, np.exp(k), HORIZON, target))
        )
        assert abs(100 * (vol - target)) <= 0.30 + 400 * se, (m, 100 * (vol - target), 100 * se)
    lc = LocalCorrelationConfig(particle=PARTICLE)
    plain = calibrate_cdv(models, family, basket, surface, lv, PARTICLE, SIM, lc, CrossDependence(0.0))  # type: ignore[arg-type]
    j = res.times.size - 1  # type: ignore[attr-defined]
    down = float(np.interp(-1.0 * atm * np.sqrt(HORIZON), res.k_grid, res.lam[j]))  # type: ignore[attr-defined]
    down0 = float(np.interp(-1.0 * atm * np.sqrt(HORIZON), plain.k_grid, plain.lam[j]))
    print(f"\nlambda at -1 sd at 3m: beta 3 {down:.4f}, beta 0 {down0:.4f}")
    assert down < down0 - 0.02


def test_synthetic_truth_round_trip() -> None:
    """A cross-dependent world with ``β = 3`` and a constant ``λ = 0.5`` (its basket local
    variance regressed on its own cloud, seed 777) is recovered by the calibration with
    ``β = 3`` on another seed — ``λ`` within 0.05 of 0.5 inside ±1.5 sd from one month on, no
    clipped mass — while with ``β = 0`` the same target is not a constant-``λ`` world: the
    calibrated ``λ`` at −1.5 sd is above the one at +1.5 sd by more than 0.15."""
    models, family, basket, surface, lv = world()
    lc = LocalCorrelationConfig(particle=PARTICLE)
    dep = CrossDependence(3.0)
    truth = calibrate_cdv(models, family, basket, surface, lv, dataclasses.replace(PARTICLE, seed=777, n_particles=100_000),  # type: ignore[arg-type]
                          SIM, lc, dep, fixed_lambda=0.5)  # fmt: skip
    fit = calibrate_cdv(models, family, basket, surface, lv, PARTICLE, SIM, lc, dep, target_rows=truth.target)  # type: ignore[arg-type]
    plain = calibrate_cdv(models, family, basket, surface, lv, PARTICLE, SIM, lc, CrossDependence(0.0),  # type: ignore[arg-type]
                          target_rows=truth.target)  # fmt: skip
    atm = float(surface.atm_vol(HORIZON))  # type: ignore[attr-defined]
    worst = 0.0
    for j in range(fit.times.size):
        tj = float(fit.times[j])
        if tj < 1 / 12 - 1e-9:
            continue
        inside = np.abs(fit.k_grid) <= 1.5 * atm * np.sqrt(tj)
        worst = max(worst, float(np.abs(fit.lam[j][inside] - 0.5).max()))
    last = fit.times.size - 1
    lo, hi = -1.5 * atm * np.sqrt(HORIZON), 1.5 * atm * np.sqrt(HORIZON)
    slope = float(
        np.interp(lo, plain.k_grid, plain.lam[last]) - np.interp(hi, plain.k_grid, plain.lam[last])
    )
    print(
        f"\nsynthetic truth (beta 3, lambda 0.5): recovered within {worst:.4f} with beta 3 "
        f"(clipped mass inside {fit.max_clipped_mass_inner:.4f}); with beta 0 lambda(-1.5 sd) - lambda(+1.5 sd) "
        f"= {slope:+.4f} at 3m, clipped mass inside {plain.max_clipped_mass_inner:.4f}"
    )
    assert worst <= 0.05 and fit.max_clipped_mass_inner == 0.0
    assert slope > 0.15


# ---------------------------------------------------------------------------------------------
# the scripts of the owner's requests (e) and (f) of 2026-10-09: their pure helpers (fast)
# ---------------------------------------------------------------------------------------------


def cdv_scripts() -> tuple[Any, Any]:
    """``scripts/cdv_scan.py`` and ``scripts/cdv_stratified.py`` as modules."""
    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import cdv_scan
    import cdv_stratified

    return cdv_scan, cdv_stratified


def flat_surface(vol: float, r: float = 0.0, q: float = 0.0) -> Any:
    fc = ForwardCurve.flat(1.0, r, q)
    cfg = SSVIConfig((0.25, 1.0, 2.0), (vol,) * 3, 0.0, 1e-9, 0.5, 5.0)
    return surface_from_config(cfg, fc, fc.rate_curve)


def lognormal_regions(vol: float, T: float, f: float, strikes: list[float]) -> np.ndarray:
    """``E`` of the parts by region of ``(X − f)²`` for a lognormal ``X`` of forward ``f``, in
    closed form: ``E[((K − X)⁺)²] = K²N(−e₂) − 2KfN(−e₁) + f²e^{s²}N(−e₀)`` and
    ``E[((X − K)⁺)²] = f²e^{s²}N(e₀) − 2KfN(e₁) + K²N(e₂)``, ``s = σ√T``,
    ``e₁ = (ln(f/K) + s²/2)/s``, ``e₀ = e₁ + s``, ``e₂ = e₁ − s`` (derived: the lognormal
    partial moments of orders 0, 1 and 2)."""
    s = vol * math.sqrt(T)

    def e(K: float) -> tuple[float, float, float]:
        e1 = (math.log(f / K) + 0.5 * s * s) / s
        return e1 + s, e1, e1 - s

    def put_sq(K: float) -> float:
        if K <= 0.0:
            return 0.0
        e0, e1, e2 = e(K)
        return float(K * K * ndtr(-e2) - 2 * K * f * ndtr(-e1) + f * f * math.exp(s * s) * ndtr(-e0))  # fmt: skip

    def call_sq(K: float) -> float:
        if math.isinf(K):
            return 0.0
        e0, e1, e2 = e(K)
        return float(f * f * math.exp(s * s) * ndtr(e0) - 2 * K * f * ndtr(e1) + K * K * ndtr(e2))

    bounds = [0.0, *strikes, math.inf]
    return np.array(
        [
            put_sq(min(k2, f)) - put_sq(min(k1, f)) + call_sq(max(k1, f)) - call_sq(max(k2, f))
            for k1, k2 in itertools.pairwise(bounds)
        ]
    )


def test_region_second_moments() -> None:
    """The split of ``(L − F)²`` by region of the strike axis (``cdv_scan.region_second_moments``):

    * each part equals the direct numerical integral ``2∫_{K₁}^{K₂} OTM payoff dK`` (puts below
      the forward, calls above; a region that holds the forward is cut at it) to 1e-13, on
      levels below, inside and above the regions, with the forward on an edge and inside a
      region, and the parts sum to ``(L − F)²``;
    * ``basket_regions`` on a lognormal law: the target's parts (the strip of a flat surface by
      region) equal the closed form to 5e-4, with and without carry (the units), and the
      model's parts on 2·10⁵ antithetic paths are within five standard errors of it, the last
      of the regions included; the default strip (2000 intervals) is off by more than 5e-3 in
      a far region, which is why the script uses 10⁵."""
    cs, _ = cdv_scripts()

    def direct(L: float, k1: float, k2: float, F: float) -> float:
        def otm(K: float) -> float:
            return 2.0 * max(K - L, 0.0) if K < F else 2.0 * max(L - K, 0.0)

        hi = k2 if math.isfinite(k2) else max(L, k1, F) + 1.0  # the call payoff is 0 above L
        kinks = sorted({x for x in (L, F) if k1 < x < hi})
        return float(quad(otm, k1, hi, points=kinks or None, epsabs=1e-14, epsrel=1e-13)[0])

    levels = np.array([0.55, 0.8, 0.85, 0.97, 1.0, 1.02, 1.04, 1.1, 1.3])
    worst = 0.0
    for F, edges in ((1.0, (0.8, 0.9, 1.0, 1.1)), (1.0, (0.8, 0.95, 1.1)), (1.03, (0.7, 1.0, 1.2))):
        parts = cs.region_second_moments(levels, edges, F)
        assert parts.shape == (levels.size, len(edges) + 1) and np.all(parts >= 0.0)
        np.testing.assert_allclose(parts.sum(axis=1), (levels - F) ** 2, rtol=0, atol=1e-15)
        bounds = [0.0, *edges, math.inf]
        for i, L in enumerate(levels):
            for j in range(len(edges) + 1):
                worst = max(worst, abs(direct(float(L), bounds[j], bounds[j + 1], F) - parts[i, j]))
    assert worst <= 1e-13, worst
    for bad in ((1.0, 0.9), (0.0, 1.0), ()):
        with pytest.raises(ValueError):
            cs.region_second_moments(levels, bad, 1.0)
    with pytest.raises(ValueError):
        cs.region_second_moments(levels, (0.9, 1.1), 0.0)
    assert cs.region_labels() == ["below -3.5 sd", "-3.5 to -2.5 sd", "-2.5 to -1.5 sd", "-1.5 to +0.0 sd",
                                  "+0.0 to +1.5 sd", "above +1.5 sd"]  # fmt: skip
    rng = np.random.default_rng(11)
    print()
    for r, q, vol, T in ((0.0, 0.0, 0.2, 0.25), (0.04, 0.01, 0.35, 1.0)):
        surface = flat_surface(vol, r, q)
        f = float(surface.forward_curve.forward(T))
        sd = vol * math.sqrt(T)
        z = rng.standard_normal(100_000)
        z = np.column_stack([z, -z]).ravel()  # antithetic pairs (2i, 2i + 1)
        level = np.exp(-0.5 * sd * sd + sd * z)
        got = cs.basket_regions(level, surface, T)
        exact = lognormal_regions(vol, T, f, [f * math.exp(m * sd) for m in cs.REGION_EDGES_SD])
        target, model, se = np.array(got["target"]), np.array(got["model"]), np.array(got["model_se"])  # fmt: skip
        np.testing.assert_allclose(target, exact, rtol=5e-4)
        assert got["total_target"] == pytest.approx(exact.sum(), rel=1e-6)
        assert np.all(np.abs(model - exact) <= 5.0 * se), (model - exact) / se
        assert abs(got["total_model"] - exact.sum()) <= 5.0 * got["total_model_se"]
        np.testing.assert_allclose(got["difference"], model - target, rtol=0, atol=1e-18)
        _, coarse = strip_second_moment(surface, T, splits=[m * sd for m in cs.REGION_EDGES_SD])  # type: ignore[misc]
        print(
            f"regions of a lognormal law (vol {vol}, T {T}, f {f:.4f}): strip/closed form - 1 at 1e5 intervals "
            f"{np.abs(target / exact - 1).max():.1e}, at 2000 {np.abs(coarse / exact - 1).max():.1e}; "
            f"model - closed form in standard errors {np.round((model - exact) / se, 2).tolist()}"
        )
        if r == 0.0:
            assert np.abs(coarse / exact - 1).max() > 5e-3


def synthetic_paths(
    cs: Any, seed: int, vol_names: float, rho: float, skew: float = 0.0, n: int = 40_000
) -> Any:
    """Made-up per-path quantities of a ten-name equicorrelated Gaussian basket (antithetic
    pairs): ``cdv_scan.Paths``.  ``skew``: every name's return is scaled by ``e^{−skew·c}``,
    ``c`` the common normal — volatilities that rise together when the basket falls."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n // 2, 11))
    z = np.concatenate([z, -z], axis=1).reshape(n, 11)
    common, own = z[:, :1], z[:, 1:]
    r = vol_names * np.exp(-skew * common) * (math.sqrt(rho) * common + math.sqrt(1.0 - rho) * own)
    w = np.full(10, 0.1)
    rb = r @ w
    return cs.Paths(np.abs(r - rb[:, None]) @ w, (r * r) @ w, rb, np.exp(rb - 0.5 * rb.var()))


def test_attribution_identities() -> None:
    """``cdv_scan.attribution`` of the forward's move between two models on the same paths:
    ``Δ ln E[D] = Δ ln κ + ½ Δ ln E[V]``; ``½ Δ ln E[V] = wing estimate + names term + short
    term``, so the gap to the wing estimate is ``Δ ln κ + names term + short term``; the forward
    at fixed ``κ`` is ``κ(0)·√E[V](β)``; the standard error of ``Δ ln E[D]`` is the paired
    ratio's; a model against itself moves by nothing, with no error.  With the names' second
    moment held at its ``β = 0`` estimate ``N⁰``: ``Δ ln E[D] = Δ ln κ_h + ½ ln((N⁰ − E^β[R̄²])/
    E⁰[V])``, the second term being the wing estimate plus what the basket is still short of,
    and the names' drift is the difference to the measured ``½ Δ ln E[V]`` (noise here: the two
    made-up worlds have the same names)."""
    cs, _ = cdv_scripts()
    lcd = sys.modules["lcm_diagnostics"]
    base = synthetic_paths(cs, 5, 0.10, 0.50)
    # the same normals: more correlation, and volatilities that rise when the basket falls
    other = synthetic_paths(cs, 5, 0.10, 0.56, skew=0.2)
    m_b = 1.05 * float(np.mean(other.rb**2))
    a = cs.attribution(base, other, m_b)
    ed0, ed1 = float(base.D.mean()), float(other.D.mean())
    ev0, ev1 = float(base.V.mean()), float(other.V.mean())
    n0, n1 = float(base.sq.mean()), float(other.sq.mean())
    assert a["dln_ED"] == pytest.approx(math.log(ed1 / ed0), abs=1e-14)
    assert a["half_dln_EV"] == pytest.approx(0.5 * math.log(ev1 / ev0), abs=1e-14)
    assert a["dln_ED"] == pytest.approx(a["dln_kappa"] + a["half_dln_EV"], abs=1e-14)
    assert a["ED_fixed_kappa"] == pytest.approx(ed0 / math.sqrt(ev0) * math.sqrt(ev1), rel=1e-13)
    assert a["wing_estimate"] == pytest.approx(0.5 * math.log((n0 - m_b) / ev0), abs=1e-14)
    assert a["half_dln_EV"] == pytest.approx(a["wing_estimate"] + a["names_term"] + a["short_term"], abs=1e-13)  # fmt: skip
    assert a["gap_to_wing_estimate"] == pytest.approx(a["dln_kappa"] + a["names_term"] + a["short_term"], abs=1e-13)  # fmt: skip
    assert a["short_term"] > 0  # E[Rbar²] of the other model is below m_b
    assert a["d_E_Rbar2"] == pytest.approx(float(np.mean(other.rb**2) - np.mean(base.rb**2)), abs=1e-16)  # fmt: skip
    assert a["d_sum_w_ER2"] == pytest.approx(n1 - n0, abs=1e-16)
    assert a["shortfall_closed"] == pytest.approx(a["d_E_Rbar2"] / (m_b - float(np.mean(base.rb**2))), rel=1e-12)  # fmt: skip
    # the names' second moment held at its beta = 0 estimate
    b1 = float(np.mean(other.rb**2))
    assert a["dln_ED"] == pytest.approx(a["dln_kappa_names_held"] + a["half_dln_EV_names_held"], abs=1e-14)  # fmt: skip
    assert a["half_dln_EV_names_held"] == pytest.approx(0.5 * math.log((n0 - b1) / ev0), abs=1e-14)
    assert a["half_dln_EV_names_held"] == pytest.approx(a["wing_estimate"] + a["short_term_names_held"], abs=1e-13)  # fmt: skip
    assert a["names_drift"] == pytest.approx(a["half_dln_EV"] - a["half_dln_EV_names_held"], abs=1e-13)  # fmt: skip
    assert a["kappa_base"] == pytest.approx(ed0 / math.sqrt(ev0), rel=1e-13)
    assert a["kappa_names_held"] == pytest.approx(ed1 / math.sqrt(n0 - b1), rel=1e-13)
    assert a["kappa_names_held"] == pytest.approx(a["kappa_base"] * math.exp(a["dln_kappa_names_held"]), rel=1e-12)  # fmt: skip
    assert a["ED_fixed_kappa_names_held"] == pytest.approx(ed0 / math.sqrt(ev0) * math.sqrt(n0 - b1), rel=1e-13)  # fmt: skip
    # two worlds with the same names (only the correlation differs): the drift of the names'
    # second moment is noise, and kappa, which is the same in both, is resolved either way
    same_names = cs.attribution(base, synthetic_paths(cs, 5, 0.10, 0.56), m_b)
    assert 0 < same_names["names_drift_se"] < 2e-3
    assert abs(same_names["names_drift"]) <= 4.0 * same_names["names_drift_se"]
    assert abs(same_names["dln_kappa"]) <= 1e-12
    assert abs(same_names["dln_kappa_names_held"]) <= 4.0 * same_names["dln_kappa_names_held_se"]
    print(
        f"\nattribution on made-up paths: d ln E[D] {a['dln_ED']:+.5f} ({a['dln_ED_se']:.5f}) = d ln kappa {a['dln_kappa']:+.5f} "
        f"({a['dln_kappa_se']:.5f}) + 0.5 d ln E[V] {a['half_dln_EV']:+.5f} ({a['half_dln_EV_se']:.5f}); names held: "
        f"{a['dln_kappa_names_held']:+.5f} ({a['dln_kappa_names_held_se']:.5f}) + {a['half_dln_EV_names_held']:+.5f} "
        f"({a['half_dln_EV_names_held_se']:.5f}); names' drift {a['names_drift']:+.5f} ({a['names_drift_se']:.5f})"
    )
    ratio, ratio_se = lcd.pair_ratio(other.D, base.D)
    assert a["dln_ED_se"] == pytest.approx(ratio_se / ratio, rel=1e-4)
    assert 0 < a["dln_ED_se"] < 0.1 * abs(a["dln_ED"])  # paired: the move is resolved
    same = cs.attribution(base, base, m_b)
    for key in ("dln_ED", "dln_kappa", "half_dln_EV", "names_term", "d_E_Rbar2", "shortfall_closed",
                "dln_kappa_names_held", "half_dln_EV_names_held", "names_drift"):  # fmt: skip
        assert same[key] == pytest.approx(0.0, abs=1e-14) and same[f"{key}_se"] == pytest.approx(0.0, abs=1e-9)  # fmt: skip
    assert same["ED_fixed_kappa"] == pytest.approx(ed0, rel=1e-13)


def test_stratified_rules(tmp_path: Path) -> None:
    """The rules of ``scripts/cdv_stratified.py`` on made-up numbers.

    ``select_beta``: the smallest ``β`` whose clipped mass is at most 1 % (the bound included);
    ``β* = 0`` when ``β = 0`` already satisfies it; when none reaches 1 %, the ``β`` with the
    smallest clipped mass (the smallest on a tie), flagged.  ``priced_betas``: ``(β*, 2β*)``,
    and the two smallest positive betas of the grid when ``β* = 0``.  ``select_dates``: the
    ``n`` largest and the ``n`` smallest ``clip_inner_max`` among the priced rows, ties to the
    earlier date, with the table's flag.  The digest of a resume moves with the grid and
    ``g_max`` and not with the order of the betas; outputs are refused outside
    ``outputs/dispersion_lc``."""
    _, st = cdv_scripts()
    grid = [1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 10.0, 12.0]
    assert st.CLIP_TARGET == 0.01
    assert st.select_beta([(0.0, 0.153), (1.0, 0.08), (2.0, 0.009), (3.0, 0.0)]) == (2.0, True)
    assert st.select_beta([(3.0, 0.0), (0.0, 0.153), (2.0, 0.009), (1.0, 0.08)]) == (2.0, True)
    assert st.select_beta([(0.0, 0.153), (1.0, 0.0100)]) == (1.0, True)  # at the target: reached
    assert st.select_beta([(0.0, 0.153), (1.0, 0.0101)]) == (1.0, False)  # just above: not reached
    assert st.select_beta([(0.0, 0.0182), (1.0, 0.0085)]) == (1.0, True)
    assert st.priced_betas(2.0, grid) == (2.0, 4.0) and st.priced_betas(8.0, grid) == (8.0, 16.0)
    # beta = 0 already inside the target
    assert st.select_beta([(0.0, 0.004)]) == (0.0, True) and st.select_beta([(0.0, 0.0)]) == (0.0, True)  # fmt: skip
    assert st.priced_betas(0.0, grid) == (1.0, 2.0) and st.priced_betas(0.0, [3.0, 1.0, 0.0]) == (1.0, 3.0)  # fmt: skip
    # none reaches 1 %: the best, flagged
    masses = [0.153, 0.12, 0.09, 0.087, 0.06, 0.044, 0.051, 0.092, 0.11]
    tried = list(zip([0.0, *grid], masses, strict=True))
    assert st.select_beta(tried) == (6.0, False) and st.priced_betas(6.0, grid) == (6.0, 12.0)
    assert st.select_beta([(0.0, 0.2), (1.0, 0.05), (2.0, 0.05)]) == (1.0, False)  # a tie
    assert st.select_beta([(0.0, 0.02), (1.0, 0.05), (2.0, 0.03)]) == (
        0.0,
        False,
    )  # beta = 0 the best
    assert st.select_beta(tried, target=0.05) == (6.0, True)
    with pytest.raises(ValueError):
        st.select_beta([])
    with pytest.raises(ValueError):
        st.select_beta([(0.0, float("nan"))])
    assert st.search_betas([3, 1, 2, 2, 0, -1]) == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        st.search_betas([0.0, 3.0])
    # the dates
    table = pd.DataFrame(
        {
            "date": ["2020-01-06", "2020-02-03", "2020-03-02", "2020-04-06", "2020-05-04", "2020-06-01", "2020-07-06", "2020-08-03"],
            "status": ["ok", "check", "failed", "ok", "ok", "ok", "ok", "ok"],
            "clip_inner_max": [0.0, 0.30, np.nan, 0.0, 0.12, 0.30, 0.002, 0.05],
            "n_names_unscreened": [0.0, 1.0, np.nan, np.nan, 0.0, 0.0, 2.0, 0.0],
        }
    )  # fmt: skip
    sel = st.select_dates(table, 2)
    assert sel["date"].tolist() == ["2020-02-03", "2020-06-01", "2020-01-06", "2020-04-06"]
    assert sel["group"].tolist() == ["high", "high", "low", "low"]
    assert sel["table_clip_inner_max"].tolist() == [0.30, 0.30, 0.0, 0.0]
    assert sel["table_unscreened"].tolist() == [True, False, False, False]
    sel3 = st.select_dates(table, 3)
    assert sel3["date"].tolist() == ["2020-02-03", "2020-06-01", "2020-05-04", "2020-01-06", "2020-04-06", "2020-07-06"]  # fmt: skip
    assert sel3["table_unscreened"].tolist() == [True, False, False, False, False, True]
    assert st.select_dates(table.drop(columns="n_names_unscreened"), 1)["table_unscreened"].tolist() == [False, False]  # fmt: skip
    with pytest.raises(ValueError, match="overlap"):
        st.select_dates(table, 4)  # seven priced rows
    with pytest.raises(ValueError):
        st.select_dates(table, 0)
    given = st.given_selection(["2020-06-01", "2020-08-03", "2021-01-04"], table, 2)
    assert given["group"].tolist() == ["high", "given", "given"]
    assert given["table_clip_inner_max"].tolist()[:2] == [0.30, 0.05] and math.isnan(given["table_clip_inner_max"].iloc[2])  # fmt: skip
    assert st.given_selection(["2020-07-06"], table, 2)["table_unscreened"].tolist() == [True]
    assert st.given_selection(["2020-07-06"], None, 2)["group"].tolist() == ["given"]
    # the digest and the folders
    cfg = st.config_for("development", None)
    digest = st.args_digest(cfg, "3m", "development", grid, 3.0)
    assert digest == st.args_digest(cfg, "3m", "development", list(reversed(grid)), 3.0)
    assert digest != st.args_digest(cfg, "3m", "development", grid[:-1], 3.0)
    assert digest != st.args_digest(cfg, "3m", "development", grid, 2.0)
    assert digest != st.args_digest(cfg, "3m", "production", grid, 3.0)
    smoke = st.config_for("development", 20_000)
    assert smoke["budgets"]["development"] == {"n_particles": 20_000, "n_paths": 20_000, "companion_paths": 20_000}  # fmt: skip
    assert cfg["budgets"]["development"]["n_particles"] == 200_000
    assert digest != st.args_digest(smoke, "3m", "development", grid, 3.0)
    with pytest.raises(ValueError):
        st.config_for("development", 2001)
    assert st.checked_dir(tmp_path / "outputs" / "dispersion_lc" / "cdv", "--out").name == "cdv"
    for bad in (tmp_path / "outputs" / "dispersion", tmp_path):
        with pytest.raises(ValueError, match="dispersion_lc"):
            st.checked_dir(bad, "--out")


def test_stratified_table(tmp_path: Path) -> None:
    """The table and the report of ``scripts/cdv_stratified.py`` on made-up rows (no
    calibration): ``model_block`` fills a model's entries with standard errors (the copula's
    error in quadrature in ``E[D]/P_D``); ``with_star_columns`` reads CDV at ``β*`` and ``2β*``
    from the priced models where ``β* > 0`` and from LC where ``β* = 0``; ``group_means`` gives
    the mean, the standard error of the mean and the count per group; ``build_table`` keeps the
    rows of the run's arguments; the report has every group, the averages and the failed date."""
    cs, st = cdv_scripts()
    lcd = sys.modules["lcm_diagnostics"]
    surface, T = flat_surface(0.10), 0.25
    listed = (0.0045, 0.0100, 0.0055)  # E^Q[V], Σ w M_i, M_B^listed
    copula = (0.062, 0.0001)
    digest = "0123456789abcdef"

    def row(date: str, beta_star: float, reached: bool, converged: bool | None, seed: int) -> dict[str, Any]:  # fmt: skip
        betas = st.priced_betas(beta_star, [1.0, 2.0, 3.0, 4.0])
        models = {
            "cc": synthetic_paths(cs, seed, 0.10, 0.50),
            "lc": synthetic_paths(cs, seed, 0.10, 0.53),
            "cdv_a": synthetic_paths(cs, seed, 0.10, 0.55),
            "cdv_b": synthetic_paths(cs, seed, 0.10, 0.58),
        }
        out: dict[str, Any] = {
            "date": date, "tenor": "3m", "budget": "development", "status": "ok", "reason": "", "git_commit": "abc1234",
            "args_digest": digest, "beta_star": beta_star, "reached": reached, "beta_star_zero": beta_star == 0.0,
            "note": "", "clip_inner_lc": 0.15 if beta_star else 0.001, "clip_inner_cdv_a": 0.008, "clip_inner_cdv_b": 0.0,
            "tried": [{"beta": 0.0, "clip_inner": 0.15}], "betas": [1.0, 2.0, 3.0, 4.0], "screen": {"long_maturity": 1.0},
            "n_names_unscreened": None, "P_D": copula[0], "kappa_cop": 0.75, "listed_variance_forward": 0.95,
            "model_s_converged": converged, "P_D_S_over_P_D": 0.94, "seconds_total": 12.0, "n_calibrations": 3,
        }  # fmt: skip
        for tag, beta in zip(st.MODELS, (0.0, 0.0, *betas), strict=True):
            out.update(st.model_block(tag, beta, models[tag], None, surface, T, listed, copula))
        for num, den in (("lc", "cc"), ("cdv_a", "cc"), ("cdv_b", "cc"), ("cdv_a", "lc"), ("cdv_b", "lc")):  # fmt: skip
            out[f"ratio_{num}_{den}"], out[f"ratio_{num}_{den}_se"] = lcd.pair_ratio(models[num].D, models[den].D)  # fmt: skip
        return out

    rows = [row("2020-02-03", 3.0, True, True, 1), row("2020-06-01", 4.0, False, False, 2),
            row("2020-01-06", 0.0, True, True, 3), row("2020-04-06", 0.0, True, None, 4)]  # fmt: skip
    block = rows[0]
    p = synthetic_paths(cs, 1, 0.10, 0.53)
    ed, ed_se = lcd.pair_mean(p.D)
    assert (block["ED_lc"], block["ED_lc_se"]) == (ed, ed_se)
    assert block["kappa_lc"] == pytest.approx(ed / math.sqrt(p.V.mean()), rel=1e-12)
    assert block["EV_over_EQV_lc"] == pytest.approx(p.V.mean() / listed[0], rel=1e-12)
    assert block["E_Rbar2_over_listed_lc"] == pytest.approx(np.mean(p.rb**2) / listed[2], rel=1e-12)
    assert block["names_over_listed_lc"] == pytest.approx(p.sq.mean() / listed[1], rel=1e-12)
    assert block["over_copula_lc"] == pytest.approx(ed / copula[0], rel=1e-12)
    assert block["over_copula_lc_se"] == pytest.approx(ed / copula[0] * math.hypot(ed_se / ed, copula[1] / copula[0]), rel=1e-12)  # fmt: skip
    assert math.isnan(block["clip_low_inner_cc"]) and block["beta_cdv_b"] == 6.0
    assert all(math.isfinite(block[f"idx_{k}_{tag}_se"]) for k in ("atm", "m15", "m25") for tag in st.MODELS)  # fmt: skip
    assert 0 < block["ratio_cdv_a_lc_se"] < 0.2 * abs(block["ratio_cdv_a_lc"] - 1.0)  # paired
    rows.append({"date": "2020-05-04", "tenor": "3m", "budget": "development", "status": "failed",
                 "reason": "ValueError: no usable smile", "git_commit": "abc1234", "args_digest": digest, "seconds_total": 1.0})  # fmt: skip
    stale = {**row("2020-08-03", 2.0, True, True, 5), "args_digest": "another"}
    for r in [*rows, stale]:
        st.write_row(tmp_path / "rows" / f"{r['date']}.json", r)
    selection = pd.DataFrame(
        {
            "date": ["2020-02-03", "2020-06-01", "2020-05-04", "2020-01-06", "2020-04-06", "2020-08-03", "2020-09-01"],
            "group": ["high", "high", "high", "low", "low", "low", "low"],
            "table_clip_inner_max": [0.30, 0.20, 0.25, 0.0, 0.0, 0.001, 0.002],
            "table_unscreened": [False, True, False, False, False, False, False],
        }
    )  # fmt: skip
    frame = st.build_table(tmp_path / "rows", selection, digest)
    assert frame["date"].tolist() == ["2020-02-03", "2020-05-04", "2020-06-01", "2020-01-06", "2020-04-06"]  # fmt: skip
    by = frame.set_index("date")
    assert json.loads(by.at["2020-02-03", "tried_json"]) == [{"beta": 0.0, "clip_inner": 0.15}]
    # beta* > 0: the priced CDV models; beta* = 0: LC itself
    for key in ("ED", "kappa", "EV_over_EQV", "E_Rbar2_over_listed", "over_copula", "idx_m25"):
        for suffix in ("", "_se"):
            assert by.at["2020-02-03", f"{key}_star{suffix}"] == by.at["2020-02-03", f"{key}_cdv_a{suffix}"]  # fmt: skip
            assert by.at["2020-02-03", f"{key}_2star{suffix}"] == by.at["2020-02-03", f"{key}_cdv_b{suffix}"]  # fmt: skip
            assert by.at["2020-01-06", f"{key}_star{suffix}"] == by.at["2020-01-06", f"{key}_lc{suffix}"]  # fmt: skip
            assert by.at["2020-01-06", f"{key}_2star{suffix}"] == by.at["2020-01-06", f"{key}_lc{suffix}"]  # fmt: skip
            assert math.isnan(by.at["2020-05-04", f"{key}_star{suffix}"])
    assert by.at["2020-02-03", "ratio_star_cc"] == by.at["2020-02-03", "ratio_cdv_a_cc"]
    assert by.at["2020-02-03", "ratio_2star_cc_se"] == by.at["2020-02-03", "ratio_cdv_b_cc_se"]
    assert by.at["2020-01-06", "ratio_star_cc"] == by.at["2020-01-06", "ratio_lc_cc"]
    assert by.at["2020-01-06", "ratio_star_lc"] == 1.0 and by.at["2020-01-06", "ratio_2star_lc_se"] == 0.0  # fmt: skip
    assert by.at["2020-02-03", "clip_inner_star"] == 0.008 and by.at["2020-01-06", "clip_inner_star"] == 0.001  # fmt: skip
    assert by.at["2020-02-03", "beta_2star"] == 6.0 and by.at["2020-01-06", "beta_2star"] == 0.0
    assert by["flag_unscreened"].tolist() == [False, False, True, False, False]
    assert by["S_converged"].tolist() == [True, False, False, True, False]
    assert by.at["2020-02-03", "S_over_copula"] == 0.94 and math.isnan(by.at["2020-06-01", "S_over_copula"])  # fmt: skip
    assert by.at["2020-02-03", "star_cc_minus_S"] == by.at["2020-02-03", "ratio_cdv_a_cc"] - 0.94
    assert by.at["2020-01-06", "star_cc_minus_lc_cc"] == 0.0
    # the averages
    means = st.group_means(frame[frame["status"] != "failed"], ["ratio_star_cc", "S_over_copula"])
    hit = means.set_index(["group", "column"])
    x = np.array([by.at["2020-02-03", "ratio_cdv_a_cc"], by.at["2020-06-01", "ratio_cdv_a_cc"]])
    assert hit.at[("high", "ratio_star_cc"), "mean"] == pytest.approx(x.mean(), rel=1e-14)
    assert hit.at[("high", "ratio_star_cc"), "sem"] == pytest.approx(x.std(ddof=1) / math.sqrt(2), rel=1e-12)  # fmt: skip
    assert hit.at[("high", "ratio_star_cc"), "n"] == 2
    assert hit.at[("high", "S_over_copula"), "n"] == 1 and math.isnan(hit.at[("high", "S_over_copula"), "sem"])  # fmt: skip
    assert hit.at[("low", "S_over_copula"), "n"] == 1
    # the report and the table on disk
    meta = {"tenor": "3m", "budget": "development", "commit": "abc1234", "digest": digest, "g_max": 3.0, "betas": [1.0, 2.0, 3.0, 4.0],
            "selection": "made up", "screen": {"long_maturity": 1.0}, "smoke": False, "counts": "none"}  # fmt: skip
    text = st.report(frame, meta)
    for needle in ("## Group `high`", "## Group `low`", "## Averages by group", "## Failed dates", "no usable smile",
                   "## Dates with beta* = 0", "(not converged)", "**no**", "CDV(beta*)/CC - LC/CC", "high: without flagged"):  # fmt: skip
        assert needle in text, needle
    assert "nan" not in text.replace("n/a", "")
    frame.to_parquet(tmp_path / "table.parquet", index=False)
    assert pd.read_parquet(tmp_path / "table.parquet")["date"].tolist() == frame["date"].tolist()
    assert "No row yet." in st.report(pd.DataFrame(), meta)
    only_failed = st.build_table(tmp_path / "rows", selection[selection["date"] == "2020-05-04"], digest)  # fmt: skip
    assert "## Failed dates" in st.report(only_failed, meta) and "## Averages" not in st.report(only_failed, meta)  # fmt: skip
