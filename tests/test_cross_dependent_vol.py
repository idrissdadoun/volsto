"""Cross-dependent volatility prototype (``volsto/multi/cdv.py``, ``docs/cross_dependent_vol.md``):

* ``β = 0`` is the local correlation model of M12, bit for bit;
* the per-name normalisation of a constant is 1;
* with ``β > 0`` each name still reprices its own smile (C2 at ``β > 0``);
* a synthetic truth — a cross-dependent world with a known constant ``λ`` — is recovered by the
  calibration when ``β`` is the truth's, and is not a constant-``λ`` world when ``β = 0``.

The worlds, grids and settings are those of ``tests/test_local_correlation.py`` (W5 at 3m).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import test_local_correlation as t

from volsto.calibration import local_correlation as lcal
from volsto.config import LocalCorrelationConfig, ParticleConfig, SimConfig
from volsto.market.bs import black_price, black_vega, implied_vol
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
