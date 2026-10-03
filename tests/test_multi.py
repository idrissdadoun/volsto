"""The multi-asset layer (``volsto/multi``): correlated draws, the model, the products against
the Gaussian closed forms, the triangle-inequality bound, common random numbers."""

from __future__ import annotations

import numpy as np
import pytest

from volsto.config import SimConfig
from volsto.market.curves import ForwardCurve
from volsto.models.bs import BlackScholes
from volsto.multi import (
    BasketOption,
    BasketStraddle,
    CorrelatedDraws,
    MultiAssetModel,
    MultiAssetMonteCarlo,
    Palladium,
    SingleNameStraddles,
    VarianceDispersion,
    basket_vol,
    dispersion_straddles,
    gaussian_dispersion_moments,
    gaussian_palladium_call,
    gaussian_palladium_forward,
    gaussian_straddle_dispersion,
    implied_correlation,
    pairwise_mean_correlation,
)
from volsto.multi.draws import check_correlation, constant_correlation
from volsto.products.base import daily_schedule

VOLS = (0.25, 0.30, 0.35, 0.40)
RHO = 0.5
T = 0.25


def _world(rho: float = RHO) -> tuple[MultiAssetModel, np.ndarray, object]:  # type: ignore[type-arg]
    fcs = [ForwardCurve.flat(100.0, 0.0, 0.0) for _ in VOLS]
    models = [BlackScholes(v, fc) for v, fc in zip(VOLS, fcs, strict=True)]
    mm = MultiAssetModel(models, constant_correlation(len(VOLS), rho), names=list("ABCD"))
    return mm, np.ones(len(VOLS)) / len(VOLS), fcs[0].rate_curve


def test_correlation_validation() -> None:
    with pytest.raises(ValueError):
        check_correlation([[1.0, 0.5], [0.4, 1.0]])
    with pytest.raises(ValueError):
        check_correlation([[1.0, 1.2], [1.2, 1.0]])
    with pytest.raises(ValueError):
        constant_correlation(4, -0.5)
    c = constant_correlation(3, 0.3)
    assert pairwise_mean_correlation(c) == pytest.approx(0.3)
    assert implied_correlation(
        basket_vol(VOLS[:3], c, np.ones(3) / 3), VOLS[:3], np.ones(3) / 3
    ) == pytest.approx(0.3)


def test_draws_reproduce_the_correlation_and_keep_antithetics() -> None:
    d = CorrelatedDraws(11, 20_000, 3, constant_correlation(3, 0.6))
    z = d.block_all(0, 3, 0, 20_000)
    assert z.shape == (20_000, 3, 3)
    np.testing.assert_allclose(z[0::2], -z[1::2])  # antithetic pairs
    c = np.corrcoef(z[:, 1, :].T)
    assert np.allclose(c[np.triu_indices(3, 1)], 0.6, atol=0.03)
    v = d.asset(2).block(0, 3, 0, 20_000)
    np.testing.assert_array_equal(v[:, :, 0], z[:, :, 2])
    # the independent streams are shared across correlations (common random numbers)
    e = d.with_correlation(constant_correlation(3, 0.2))
    np.testing.assert_array_equal(
        e.independent_block(0, 3, 0, 20_000), d.independent_block(0, 3, 0, 20_000)
    )


def test_model_rejects_factor_models() -> None:
    from volsto.config import BergomiParams
    from volsto.models.bergomi import BergomiSV

    fc = ForwardCurve.flat(100.0, 0.0, 0.0)
    with pytest.raises(ValueError):
        MultiAssetModel(
            [BlackScholes(0.2, fc), BergomiSV(BergomiParams.one_factor(1.0, 1.0, -0.5), fc, 0.04)],
            constant_correlation(2, 0.3),
        )  # type: ignore[call-arg]


def test_products_against_gaussian_closed_forms() -> None:
    mm, w, dc = _world()
    cfg = SimConfig(n_paths=60_000, chunk_size=30_000, seed=3, dt_max=1 / 52)
    mc = MultiAssetMonteCarlo(cfg)
    prods = [
        Palladium(w, 0.0, T, dc),
        Palladium(w, 0.06, T, dc),
        BasketStraddle(w, T, dc),
        SingleNameStraddles(w, T, dc),
        dispersion_straddles(w, T, dc),
        VarianceDispersion(w, daily_schedule(T, 252), 0.0, dc),
        BasketOption(w, 0.0, T, 1, dc),
        BasketOption(w, 0.0, T, -1, dc),
    ]
    res = mc.price_many(prods, mm, keep_payoffs=True)
    C = mm.correlation
    fwd = gaussian_palladium_forward(VOLS, C, w, T)
    # lognormal against Gaussian: within 1.5% at 3m
    assert abs(res[0].mean - fwd) < 0.015 * fwd + 3 * res[0].stderr
    call = gaussian_palladium_call(VOLS, C, w, T, 0.06)
    assert abs(res[1].mean - call) < 0.05 * call + 3 * res[1].stderr
    sd = gaussian_straddle_dispersion(VOLS, C, w, T)
    assert abs(res[4].mean - sd) < 0.03 * sd + 3 * res[4].stderr
    m, s = gaussian_dispersion_moments(VOLS, C, w, T)
    assert m == pytest.approx(fwd) and s > 0
    # straddle = call + put on the basket, path by path
    np.testing.assert_allclose(res[2].payoffs, res[6].payoffs + res[7].payoffs)
    # the triangle inequality: D ≥ Σ w|r_i| − |r_B| on every path
    assert np.all(res[0].payoffs >= res[4].payoffs - 1e-12)
    # realised variance dispersion: Σ w σ_i² − σ_B² (zero rates, no drift)
    theory = float(np.sum(w * np.square(VOLS)) - basket_vol(VOLS, C, w) ** 2)
    assert abs(res[5].mean - theory) < 3 * res[5].stderr + 1e-3


def test_correlation_sensitivity_under_common_random_numbers() -> None:
    mm, w, dc = _world()
    cfg = SimConfig(n_paths=20_000, chunk_size=20_000, seed=3, dt_max=1 / 26)
    mc = MultiAssetMonteCarlo(cfg)
    p = Palladium(w, 0.0, T, dc)
    base = mc.price(p, mm, keep_payoffs=True)
    up = mc.price(p, mm.with_correlation(constant_correlation(4, RHO + 0.1)), keep_payoffs=True)
    diff = np.asarray(up.payoffs) - np.asarray(base.payoffs)
    # the paired difference is far tighter than the quadrature of the two stderrs
    se_paired = float(np.std(0.5 * (diff[0::2] + diff[1::2]), ddof=1) / np.sqrt(diff.size / 2))
    assert se_paired < 0.5 * float(np.hypot(base.stderr, up.stderr))
    assert up.mean < base.mean  # more correlation, less dispersion
