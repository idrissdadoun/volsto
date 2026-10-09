"""M12 (SPEC §8.7): the calibrated local correlation model.

Part LC2 — the correlation family ``ρ(λ) = (1 − λ)·R_low + λ·R_high`` and the draws: the
closed-form equicorrelation factor and the fixed-order mixing (identities I6, bit for bit),
the streams and their antithetics, the statistics of the mixed normals (C1, draws part), the
family's validation and algebra, the historical-scaled ``R_low``.

Part LC3 — the ``λ(t, k)`` function (I8, rows), the basket state, the model and its joint
kernel: ``λ ≡ 0`` is the constant-correlation model bit for bit (I1), each asset's path is the
single-asset kernel on its own mixed normals (I3), one name is the single-asset model (I4),
identical names at ``λ ≡ 1`` are one path (I5), the path-wise identities of the dispersion
payoffs (I9), the mixed normals inside the kernel (C1), the per-particle variance terms.

Part LC4 — the particle calibration: calibration and pricing share the kernel (I2), the
regression helper is the leverage's estimator twice (I7), the row at ``t_0`` is exact (I8),
one name is unidentified (I4), single names keep their local-vol law under a calibrated ``λ``
(C2), the index repricing report, the constant and parametric companions, the configuration,
the cache and the two guards.  The acceptance tests S1–S7 and S10 are slow (run them with
``-m slow -s``: each prints its table): the round trip on a known ``λ`` (S1), the
constant-correlation fixed point (S2), identical names (S3), the reference regression (S4),
the Δt halving (S5), the particle doubling (S6), the bandwidth (S7) and the carry mode (S10).

The synthetic world W5: five names, spots 1, zero rates, an SSVI surface per name (ATM vols
0.20–0.40 and a skew each), weights (0.30, 0.25, 0.20, 0.15, 0.10), ``R_low`` the
equicorrelation at 0.02, ``R_high = 11ᵀ``, daily steps.  Its fast index target is an SSVI
surface within the family's reach (ATM 0.21, ρ −0.7, η 0.9: ``λ`` falls from about 0.8 at
``k = −0.2`` to 0.15 at ``k = +0.1``, no clipping).

Seeds are fixed; a statistical cell that fails is reported with its table, never re-seeded.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import math
import types
from pathlib import Path
from typing import Any

import _lcm_reference as lcm
import numpy as np
import pandas as pd
import pytest

from volsto.calibration import lc_cache
from volsto.calibration import local_correlation as lcal
from volsto.calibration.cache import CacheMissError
from volsto.calibration.guard import CalibrationForbiddenError, calibration_forbidden
from volsto.calibration.particle import _finish_estimate, conditional_variance_estimate
from volsto.config import (
    LC_STEP_SCHEDULE,
    ConfigError,
    CurveConfig,
    LocalCorrelationConfig,
    LocalCorrelationSpec,
    LocalVolConfig,
    MarketConfig,
    ParametricLambdaConfig,
    ParticleConfig,
    SchemeConfig,
    SimConfig,
    SSVIConfig,
    StepSchedule,
    SurfacePerturbation,
    SviSurfaceConfig,
    from_mapping,
    lc_sim_config,
    to_mapping,
)
from volsto.engine.grid import TimeGrid
from volsto.engine.mc import summarize
from volsto.engine.rng import GaussianDraws
from volsto.market.bs import black_price, black_vega, implied_vol
from volsto.market.curves import DiscountCurve, ForwardCurve
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import ArbitrageError, ImpliedSurface, SSVISurface, surface_from_config
from volsto.market.svi_slices import SviSlices, fit_svi_slice
from volsto.models.bs import BlackScholes
from volsto.models.localvol import LocalVol
from volsto.multi import (
    MultiAssetModel,
    MultiAssetMonteCarlo,
    MultiPathSet,
    Palladium,
)
from volsto.multi.analytics import (
    implied_correlation,
    pairwise_mean_correlation,
    strip_second_moment,
)
from volsto.multi.draws import (
    ASSET_SEED_STRIDE,
    CorrelatedDraws,
    cholesky_factor,
    constant_correlation,
    equicorrelation_cholesky,
    equicorrelation_factor,
    is_equicorrelation,
    mix_equi,
    mix_lower,
)
from volsto.multi.family import (
    CorrelationFamily,
    check_psd_correlation,
    family_from_spec,
    file_sha256,
    historical_scaled_correlation,
    ledoit_wolf_identity,
    load_correlation_matrix,
    parse_r_high_spec,
    parse_r_low_spec,
    psd_factor,
)
from volsto.multi.lc_draws import COMMON_FACTOR_SEED_OFFSET, LocalCorrelationDraws
from volsto.multi.lc_function import LocalCorrelationFunction, ParametricLambda
from volsto.multi.lc_kernel import lc_ab
from volsto.multi.lc_model import BasketSpec, LocalCorrelationModel
from volsto.studies.disp_payoff import gap_formula

#: a positive definite, non-equicorrelation ``R_low`` and a rank-2 ``R_high`` above it entrywise
#: (two perfectly correlated groups, 0.6 across)
R_LOW_4 = np.array(
    [
        [1.0, 0.30, 0.10, 0.20],
        [0.30, 1.0, 0.15, 0.05],
        [0.10, 0.15, 1.0, 0.40],
        [0.20, 0.05, 0.40, 1.0],
    ]
)
R_HIGH_4 = np.array(
    [
        [1.0, 1.0, 0.6, 0.6],
        [1.0, 1.0, 0.6, 0.6],
        [0.6, 0.6, 1.0, 1.0],
        [0.6, 0.6, 1.0, 1.0],
    ]
)


# ---------------------------------------------------------------------------------------------
# I6: the factor and the mixing
# ---------------------------------------------------------------------------------------------


def test_equi_prefix_equals_general_mix() -> None:
    """I6: the O(n) equicorrelation prefix recursion equals the general lower-triangular mixing
    with the closed-form factor, bit for bit; ``CorrelatedDraws.block_all`` is that mixing, for
    an equicorrelation and for a general matrix, and does not depend on the block shape (one
    step at a time or several: what the calibration / pricing identity needs)."""
    rng = np.random.default_rng(3)
    for n, rho in ((1, 0.0), (2, 0.5), (5, 0.0), (5, 0.02), (5, -0.2), (30, 0.02), (30, 0.9)):
        z = rng.standard_normal((2000, 3, n))
        d, ell = equicorrelation_factor(n, rho)
        a, b = np.empty_like(z), np.empty_like(z)
        mix_equi(d, ell, z, a)
        mix_lower(equicorrelation_cholesky(n, rho), z, b)
        np.testing.assert_array_equal(a, b, err_msg=f"n={n} rho={rho}")
    # the draws object: equicorrelation
    n_paths, n_steps = 4000, 6
    eq = CorrelatedDraws(17, n_paths, n_steps, constant_correlation(5, 0.3))
    assert eq.equi
    ind = eq.independent_block(0, n_steps, 0, n_paths)
    ref = np.empty_like(ind)
    mix_lower(equicorrelation_cholesky(5, 0.3), ind, ref)
    whole = eq.block_all(0, n_steps, 0, n_paths)
    np.testing.assert_array_equal(whole, ref)
    # a general matrix: numpy's factor, the same summation order
    gen = CorrelatedDraws(17, n_paths, n_steps, R_LOW_4)
    assert not gen.equi
    ind = gen.independent_block(0, n_steps, 0, n_paths)
    ref = np.empty_like(ind)
    mix_lower(np.linalg.cholesky(R_LOW_4), ind, ref)
    np.testing.assert_array_equal(gen.block_all(0, n_steps, 0, n_paths), ref)
    # block shape and path chunks play no part
    for draws, full in ((eq, whole), (gen, ref)):
        steps = np.concatenate(
            [draws.block_all(j, j + 1, 0, n_paths) for j in range(n_steps)], axis=1
        )
        np.testing.assert_array_equal(steps, full)
        halves = np.concatenate(
            [draws.block_all(0, n_steps, 0, 1000), draws.block_all(0, n_steps, 1000, n_paths)]
        )
        np.testing.assert_array_equal(halves, full)
        np.testing.assert_array_equal(
            draws.asset(2).block(1, 4, 0, n_paths)[:, :, 0], full[:, 1:4, 2]
        )


def test_draws_antithetic_and_streams() -> None:
    """I6: the closed-form factor against ``numpy.linalg.cholesky`` (1e-14); the ``ε`` streams
    are the constant-correlation layer's; ``ε`` and ``η`` are negated on antithetic partners;
    the ``η`` streams are distinct from every asset stream; common random numbers across
    families; a rank above 8 spans several streams; coarsening is Brownian-consistent."""
    for n in (2, 5, 30, 100):
        for rho in (0.0, 0.02, 0.5, 0.97, -0.5 / (n - 1)):
            chol = equicorrelation_cholesky(n, rho)
            matrix = constant_correlation(n, rho)
            np.testing.assert_allclose(chol, np.linalg.cholesky(matrix), rtol=0, atol=1e-14)
            np.testing.assert_allclose(chol @ chol.T, matrix, rtol=0, atol=1e-14)
            assert is_equicorrelation(matrix)
            np.testing.assert_array_equal(cholesky_factor(matrix), chol)
    assert is_equicorrelation(np.array([[1.0]])) and not is_equicorrelation(R_LOW_4)
    almost = constant_correlation(4, 0.3)
    almost[0, 1] = almost[1, 0] = np.nextafter(0.3, 1.0)
    assert not is_equicorrelation(almost)
    np.testing.assert_array_equal(cholesky_factor(almost), np.linalg.cholesky(almost))
    np.testing.assert_array_equal(cholesky_factor(R_LOW_4), np.linalg.cholesky(R_LOW_4))
    with pytest.raises(ValueError):
        equicorrelation_factor(4, -0.4)
    # the streams
    assert COMMON_FACTOR_SEED_OFFSET == 104_729 and COMMON_FACTOR_SEED_OFFSET % ASSET_SEED_STRIDE
    seed, n_paths, n_steps = 23, 3000, 5
    fam = CorrelationFamily.equi(5)
    draws = LocalCorrelationDraws(seed, n_paths, n_steps, fam)
    eps = draws.eps_block(0, n_steps, 0, n_paths)
    eta = draws.eta_block(0, n_steps, 0, n_paths)
    assert eps.shape == (n_paths, n_steps, 5) and eta.shape == (n_paths, n_steps, 1)
    low = CorrelatedDraws(seed, n_paths, n_steps, fam.r_low)
    np.testing.assert_array_equal(eps, low.independent_block(0, n_steps, 0, n_paths))
    np.testing.assert_array_equal(
        draws.low.block_all(0, n_steps, 0, n_paths), low.block_all(0, n_steps, 0, n_paths)
    )
    for i in range(5):
        own = GaussianDraws(seed + ASSET_SEED_STRIDE * i, n_paths, n_steps, 1)
        np.testing.assert_array_equal(eps[:, :, i], own.block(0, n_steps, 0, n_paths)[:, :, 0])
        assert not np.array_equal(eps[:, :, i], eta[:, :, 0])
        assert abs(float(np.corrcoef(eps[:, :, i].ravel(), eta[:, :, 0].ravel())[0, 1])) < 0.05
    common = GaussianDraws(seed + COMMON_FACTOR_SEED_OFFSET, n_paths, n_steps, 1)
    np.testing.assert_array_equal(eta, common.block(0, n_steps, 0, n_paths))
    np.testing.assert_array_equal(eps[0::2], -eps[1::2])  # antithetic partners
    np.testing.assert_array_equal(eta[0::2], -eta[1::2])
    # an η seed is never an asset seed of the same base seed, whatever the basket's size
    asset_seeds = {seed + ASSET_SEED_STRIDE * i for i in range(5000)}
    assert not any(
        seed + COMMON_FACTOR_SEED_OFFSET + ASSET_SEED_STRIDE * m in asset_seeds for m in range(64)
    )
    # common random numbers: another family (a correlation bump) leaves ε and η unchanged
    other = draws.with_family(CorrelationFamily.equi(5, 0.10, 0.90))
    np.testing.assert_array_equal(other.eps_block(0, n_steps, 0, n_paths), eps)
    np.testing.assert_array_equal(other.eta_block(0, n_steps, 0, n_paths), eta)
    fresh = LocalCorrelationDraws(seed, n_paths, n_steps, CorrelationFamily.equi(5, 0.10, 0.90))
    np.testing.assert_array_equal(fresh.eta_block(0, n_steps, 0, n_paths), eta)
    # λ = 0 is the constant-correlation layer bit for bit; λ = 1 (cap 1) is the common factor
    zeros = np.zeros((n_paths, n_steps))
    np.testing.assert_array_equal(
        draws.mixed_block(0, n_steps, 0, n_paths, zeros), low.block_all(0, n_steps, 0, n_paths)
    )
    ones = draws.mixed_block(0, n_steps, 0, n_paths, np.ones((n_paths, n_steps)))
    np.testing.assert_array_equal(ones, np.repeat(eta, 5, axis=2))
    with pytest.raises(ValueError):
        draws.mixed_block(0, n_steps, 0, n_paths, np.full((n_paths, n_steps), 1.5))
    # a rank above 8: two η streams (8 + 2 normals)
    # (non-negative loadings: R_high has non-negative entries, so R_low = I lies below it)
    load = np.random.default_rng(1).uniform(0.0, 1.0, (12, 10))
    cov = load @ load.T
    sd = np.sqrt(np.diag(cov))
    high = cov / np.outer(sd, sd)
    np.fill_diagonal(high, 1.0)
    fam12 = CorrelationFamily(np.eye(12), high)
    assert fam12.rank_high == 10
    d12 = LocalCorrelationDraws(seed, 1000, 3, fam12)
    e12 = d12.eta_block(0, 3, 0, 1000)
    assert e12.shape == (1000, 3, 10) and len(d12.eta_streams) == 2
    first = GaussianDraws(seed + COMMON_FACTOR_SEED_OFFSET, 1000, 3, 8)
    second = GaussianDraws(seed + COMMON_FACTOR_SEED_OFFSET + ASSET_SEED_STRIDE, 1000, 3, 2)
    np.testing.assert_array_equal(e12[:, :, :8], first.block(0, 3, 0, 1000))
    np.testing.assert_array_equal(e12[:, :, 8:], second.block(0, 3, 0, 1000))
    # coarsening: the coarse normals are the fine ones summed over the factor, over √factor
    fine = LocalCorrelationDraws(seed, 2000, 6, fam)
    coarse = fine.coarsened(2)
    assert coarse.n_steps == 3 and coarse.n_paths == 2000 and coarse.low.equi
    fe, fh = fine.eps_block(0, 6, 0, 2000), fine.eta_block(0, 6, 0, 2000)
    np.testing.assert_array_equal(
        coarse.eps_block(0, 3, 0, 2000), (fe[:, 0::2] + fe[:, 1::2]) / np.sqrt(2)
    )
    np.testing.assert_array_equal(
        coarse.eta_block(0, 3, 0, 2000), (fh[:, 0::2] + fh[:, 1::2]) / np.sqrt(2)
    )
    with pytest.raises(ValueError):
        CorrelatedDraws.from_streams(fine.low.streams[:3], fam.r_low, seed=seed)
    with pytest.raises(ValueError):
        LocalCorrelationDraws.from_streams(fine.low, [], fam)
    assert "LocalCorrelationDraws" in repr(fine)


# ---------------------------------------------------------------------------------------------
# C1 (draws part): the mixed normals
# ---------------------------------------------------------------------------------------------


def _deciles(values: np.ndarray) -> np.ndarray:
    """The decile (0–9) of each value by rank: ten groups of equal size, also when many values
    are tied (a clipped ``λ``)."""
    order = np.argsort(values, kind="stable")
    bucket = np.empty(values.size, dtype=np.int64)
    bucket[order] = np.arange(values.size) * 10 // values.size
    return bucket


@pytest.mark.parametrize("which", ["equi", "general"])
def test_mixed_normals_statistics(which: str) -> None:
    """C1, draws part: with an exogenous ``λ`` per path and step, each asset's mixed normal has
    mean 0, variance 1 and no lag-1 autocorrelation, and within each decile of ``λ`` the cross
    moments ``E[z_i z_k]`` equal ``ρ_ik(λ)`` averaged over the decile — all within 4 standard
    errors (2·10⁵ independent paths, 4 steps, fixed seeds)."""
    fam = CorrelationFamily.equi(5) if which == "equi" else CorrelationFamily(R_LOW_4, R_HIGH_4)
    n, n_paths, n_steps = fam.n, 200_000, 4
    draws = LocalCorrelationDraws(11, n_paths, n_steps, fam, antithetic=False)
    lam = np.random.default_rng(5).uniform(0.0, fam.lambda_max, (n_paths, n_steps))
    z = draws.mixed_block(0, n_steps, 0, n_paths, lam)
    pooled = z.reshape(-1, n)
    count = pooled.shape[0]
    worst = {"mean": 0.0, "variance": 0.0, "lag-1": 0.0, "cross": 0.0}
    for i in range(n):
        worst["mean"] = max(worst["mean"], abs(float(pooled[:, i].mean())) * np.sqrt(count))
        worst["variance"] = max(
            worst["variance"], abs(float(pooled[:, i].var()) - 1.0) / np.sqrt(2.0 / count)
        )
        prod = (z[:, :-1, i] * z[:, 1:, i]).ravel()
        worst["lag-1"] = max(
            worst["lag-1"], abs(float(prod.mean())) / (float(prod.std()) / np.sqrt(prod.size))
        )
    flat = lam.ravel()
    bucket = _deciles(flat)
    rows = []
    for dec in range(10):
        sel = bucket == dec
        lam_bar = float(flat[sel].mean())
        target = (1.0 - lam_bar) * fam.r_low + lam_bar * fam.r_high
        for i in range(n):
            for k in range(i + 1, n):
                prod = pooled[sel, i] * pooled[sel, k]
                se = float(prod.std(ddof=1)) / np.sqrt(prod.size)
                zscore = (float(prod.mean()) - target[i, k]) / se
                rows.append((dec, i, k, lam_bar, float(prod.mean()), target[i, k], zscore))
                worst["cross"] = max(worst["cross"], abs(zscore))
    table = "\n".join(
        f"decile {d} pair ({i},{k}) lam {lb:.3f}: {m:+.4f} vs {t:+.4f}  z {zs:+.2f}"
        for d, i, k, lb, m, t, zs in rows
    )
    print(f"{which}: worst |z| " + ", ".join(f"{k} {v:.2f}" for k, v in worst.items()))
    assert all(v < 4.0 for v in worst.values()), f"{worst}\n{table}"


# ---------------------------------------------------------------------------------------------
# the family
# ---------------------------------------------------------------------------------------------


def test_correlation_family_algebra_and_validation(tmp_path: Path) -> None:
    """``ρ(λ)`` is a correlation matrix for every ``λ``; ``v_B = a + λ·b`` with the closed forms
    of the equicorrelation family (the Cboe implied-correlation formula on local vols);
    ``b ≥ 0``; the factor of ``R_high``; the validation; the specification strings."""
    fam = CorrelationFamily.equi(5)
    assert fam.low_equi and fam.high_ones and fam.rank_high == 1 and fam.rho_low == 0.02
    assert fam.lambda_max == pytest.approx(0.96 / 0.98) and fam.lambda_max == (0.98 - 0.02) / 0.98
    assert float(fam.equicorrelation(fam.lambda_max)) == pytest.approx(0.98, abs=1e-15)
    assert float(fam.lambda_of_equicorrelation(0.5)) == pytest.approx(0.48 / 0.98)
    np.testing.assert_array_equal(fam.l_high, np.ones((5, 1)))
    rng = np.random.default_rng(2)
    w = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
    sig = np.array([0.20, 0.25, 0.30, 0.35, 0.40])
    for general in (fam, CorrelationFamily(R_LOW_4, R_HIGH_4), CorrelationFamily(R_LOW_4)):
        n = general.n
        for lam in (0.0, 0.3, general.lambda_max, 1.0):
            rho = general.correlation(lam)
            check_psd_correlation(rho)
            assert float(np.linalg.eigvalsh(rho).min()) > -1e-12
            u = rng.uniform(0.01, 0.2, (7, n))
            a, b = general.variance_terms(u)
            assert np.all(b >= -1e-16)  # the basket variance is non-decreasing in λ
            np.testing.assert_allclose(
                np.einsum("pi,ij,pj->p", u, rho, u), a + lam * b, rtol=1e-13, atol=0
            )
        np.testing.assert_allclose(general.l_low @ general.l_low.T, general.r_low, atol=1e-14)
        np.testing.assert_allclose(general.l_high @ general.l_high.T, general.r_high, atol=1e-13)
    # the equicorrelation closed forms and the Cboe formula
    u = w * sig
    a, b = fam.variance_terms(u)
    s1, s2 = float(u.sum()), float(np.sum(u * u))
    assert float(a) == pytest.approx(0.98 * s2 + 0.02 * s1 * s1, rel=1e-14)
    assert float(b) == pytest.approx(0.98 * (s1 * s1 - s2), rel=1e-14)
    lam = 0.4
    v_b = float(a + lam * b)
    rho_t = (v_b - s2) / (s1 * s1 - s2)
    assert rho_t == pytest.approx(float(fam.equicorrelation(lam)), rel=1e-13)
    assert rho_t == pytest.approx(implied_correlation(np.sqrt(v_b), sig, w), rel=1e-13)
    # the factor of a rank-deficient R_high
    high = psd_factor(R_HIGH_4)
    assert high.shape == (4, 2) and CorrelationFamily(R_LOW_4, R_HIGH_4).rank_high == 2
    assert np.all(high[np.argmax(np.abs(high), axis=0), np.arange(2)] > 0)
    assert psd_factor(np.ones((6, 6))).shape == (6, 1)
    # validation
    with pytest.raises(ValueError, match="non-negative entrywise"):
        CorrelationFamily(constant_correlation(4, 0.7), R_HIGH_4)  # 0.7 > 0.6 across groups
    with pytest.raises(ValueError, match="R_low"):
        CorrelationFamily(np.ones((3, 3)))  # singular: no Cholesky factor
    with pytest.raises(ValueError, match="positive semi-definite"):
        CorrelationFamily(np.eye(3), np.array([[1, 0.9, -0.9], [0.9, 1, 0.9], [-0.9, 0.9, 1.0]]))
    with pytest.raises(ValueError, match="same size"):
        CorrelationFamily(np.eye(3), R_HIGH_4)
    for bad in (0.0, 1.2, float("nan")):
        with pytest.raises(ValueError, match="lambda_max"):
            CorrelationFamily(np.eye(3), lambda_max=bad)
    with pytest.raises(ValueError):
        CorrelationFamily.equi(4, 0.5, 0.5)
    with pytest.raises(ValueError):
        fam.correlation(1.5)
    with pytest.raises(ValueError, match="equicorrelation"):
        CorrelationFamily(R_LOW_4).equicorrelation(0.5)
    single = CorrelationFamily.equi(1)
    a1, b1 = single.variance_terms(np.array([0.2]))
    assert float(a1) == pytest.approx(0.04) and float(b1) == 0.0  # n = 1: λ is not identified
    assert "CorrelationFamily" in repr(fam) and fam.describe()["r_low"] == "equi"
    # specification strings
    assert parse_r_low_spec("equi") == ("equi", {})
    assert parse_r_low_spec("historical-scaled:252,0.05") == (
        "historical-scaled",
        {"window": 252, "target": 0.05},
    )
    assert parse_r_low_spec("matrix:/a/b.npy") == ("matrix", {"path": "/a/b.npy"})
    assert parse_r_high_spec("ones") == ("ones", {})
    assert parse_r_high_spec("matrix:x.csv") == ("matrix", {"path": "x.csv"})
    for bad_spec in ("flat", "historical-scaled:252", "historical-scaled:a,b", "matrix:",
                     "historical-scaled:2,0.05", "historical-scaled:252,1.5"):  # fmt: skip
        with pytest.raises(ValueError):
            parse_r_low_spec(bad_spec)
    with pytest.raises(ValueError):
        parse_r_high_spec("identity")
    default = family_from_spec(30)
    assert default.r_low_spec == "equi" and default.lambda_max == (0.98 - 0.02) / 0.98
    np.testing.assert_array_equal(default.r_low, constant_correlation(30, 0.02))
    assert family_from_spec(3, rho_min=0.0, rho_max=1.0).lambda_max == 1.0
    np.save(tmp_path / "low.npy", R_LOW_4)
    import pandas as pd

    names = ("AAA", "BBB", "CCC", "DDD")
    pd.DataFrame(R_HIGH_4, index=list(names), columns=list(names)).to_csv(tmp_path / "high.csv")
    low = load_correlation_matrix(tmp_path / "low.npy")
    high_m = load_correlation_matrix(tmp_path / "high.csv", names)
    np.testing.assert_array_equal(low, R_LOW_4)
    np.testing.assert_array_equal(high_m, R_HIGH_4)
    swapped = load_correlation_matrix(tmp_path / "high.csv", ("CCC", "AAA", "DDD", "BBB"))
    np.testing.assert_array_equal(swapped, R_HIGH_4[np.ix_([2, 0, 3, 1], [2, 0, 3, 1])])
    with pytest.raises(ValueError, match="no row"):
        load_correlation_matrix(tmp_path / "high.csv", ("AAA", "ZZZ"))
    with pytest.raises(ValueError):
        load_correlation_matrix(tmp_path / "high.txt")
    assert len(file_sha256(tmp_path / "low.npy")) == 64
    built = family_from_spec(
        4,
        r_low=f"matrix:{tmp_path / 'low.npy'}",
        r_high=f"matrix:{tmp_path / 'high.csv'}",
        lambda_max=0.9,
        r_low_matrix=low,
        r_high_matrix=high_m,
    )
    assert built.lambda_max == 0.9 and built.rank_high == 2 and not built.low_equi
    assert family_from_spec(4, r_low="matrix:x.npy", r_low_matrix=low).lambda_max == 1.0
    for kwargs in (
        {"r_low": "matrix:x.npy"},  # no matrix given
        {"r_low_matrix": low},  # a matrix for "equi"
        {"lambda_max": 0.5},  # "equi" sets its own cap
        {"r_high": "matrix:x.npy"},
        {"r_high_matrix": high_m},
        {"rho_min": 0.5, "rho_max": 0.4},
    ):
        with pytest.raises(ValueError):
            family_from_spec(4, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="expected 5"):
        family_from_spec(5, r_low="matrix:x.npy", r_low_matrix=low)


def _ledoit_wolf_reference(x: np.ndarray) -> tuple[np.ndarray, float]:
    """The specification's formulas written out observation by observation (an independent
    implementation of :func:`ledoit_wolf_identity`)."""
    t, n = x.shape
    s = np.zeros((n, n))
    for row in x:
        s += np.outer(row, row) / t

    def norm2(a: np.ndarray) -> float:
        return float(np.trace(a @ a.T)) / n

    m = float(np.trace(s)) / n
    d2 = norm2(s - m * np.eye(n))
    b_bar2 = sum(norm2(np.outer(row, row) - s) for row in x) / (t * t)
    b2 = min(b_bar2, d2)
    return (b2 / d2) * m * np.eye(n) + (1.0 - b2 / d2) * s, b2 / d2


def test_historical_scaled_r_low() -> None:
    """The ``"historical-scaled"`` ``R_low``: the Ledoit–Wolf shrinkage against an independent
    implementation (and scikit-learn's when it is installed); the weighted mean pairwise
    correlation of the result is the target; ``s`` outside ``[0, 1]`` is refused with its value;
    a name without a full window gets the target and is listed."""
    rng = np.random.default_rng(8)
    t, n = 252, 8
    market = rng.standard_normal(t)
    load = rng.uniform(0.3, 0.9, n)
    x = 0.01 * (load[None, :] * market[:, None] + rng.standard_normal((t, n)))
    z = (x - x.mean(axis=0)) / x.std(axis=0)
    shrunk, shrink = ledoit_wolf_identity(z)
    ref, ref_shrink = _ledoit_wolf_reference(z)
    np.testing.assert_allclose(shrunk, ref, rtol=0, atol=1e-13)
    assert shrink == pytest.approx(ref_shrink, abs=1e-13) and 0.0 < shrink < 1.0
    try:
        from sklearn.covariance import ledoit_wolf
    except ImportError:
        print("scikit-learn is not installed: the shrinkage is checked against the formulas only")
    else:
        sk, sk_shrink = ledoit_wolf(z, assume_centered=True)
        np.testing.assert_allclose(shrunk, sk, rtol=0, atol=1e-12)
        assert shrink == pytest.approx(float(sk_shrink), abs=1e-12)
    same, zero = ledoit_wolf_identity(np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]))
    assert zero == 0.0 and np.allclose(same, 0.5 * np.eye(2))
    w = rng.uniform(0.5, 1.5, n)
    w /= w.sum()
    target = 0.05
    out = historical_scaled_correlation(x, w, target)
    r_low = out.r_low
    np.testing.assert_array_equal(r_low, r_low.T)
    np.testing.assert_array_equal(np.diag(r_low), np.ones(n))
    assert pairwise_mean_correlation(r_low, w) == pytest.approx(target, abs=1e-14)
    assert 0.0 < out.scale < 1.0 and out.scale == pytest.approx(target / out.mean_correlation)
    assert (
        out.shrinkage == pytest.approx(shrink) and out.short_names == () and out.n_observations == t
    )
    # R_low is the convex combination (1 − s)·I + s·Ĉ of the identity and the shrunk matrix
    c_hat = shrunk / np.outer(np.sqrt(np.diag(shrunk)), np.sqrt(np.diag(shrunk)))
    np.testing.assert_allclose(
        r_low, (1.0 - out.scale) * np.eye(n) + out.scale * c_hat, rtol=0, atol=1e-14
    )
    fam = CorrelationFamily(r_low, r_low_spec="historical-scaled:252,0.05")
    assert not fam.low_equi and float(np.linalg.eigvalsh(r_low).min()) > 0.5
    # the target above the historical mean correlation: s > 1, refused with its value
    with pytest.raises(ValueError, match=r"s = target / mean correlation.*outside \[0, 1\]"):
        historical_scaled_correlation(x, w, 0.9)
    assert np.array_equal(historical_scaled_correlation(x, w, 0.0).r_low, np.eye(n))
    # a name listed during the window: the target on its off-diagonals
    short = x.copy()
    short[:40, 3] = np.nan
    part = historical_scaled_correlation(short, w, target)
    assert part.short_names == (3,) and part.info()["short_names"] == [3]
    np.testing.assert_array_equal(np.delete(part.r_low[3], 3), np.full(n - 1, target))
    assert pairwise_mean_correlation(part.r_low, w) == pytest.approx(target, abs=1e-14)
    CorrelationFamily(part.r_low)
    with pytest.raises(ValueError):
        historical_scaled_correlation(x, w[:-1], target)
    with pytest.raises(ValueError):
        historical_scaled_correlation(x, w, 1.0)


# ---------------------------------------------------------------------------------------------
# the synthetic world W5 and its helpers
# ---------------------------------------------------------------------------------------------

W5_WEIGHTS = np.array([0.30, 0.25, 0.20, 0.15, 0.10])
W5_ATM = (0.20, 0.25, 0.30, 0.35, 0.40)
W5_RHO = (-0.70, -0.60, -0.50, -0.40, -0.30)  # the SSVI correlation: a skew per name
W5_ETA, W5_GAMMA = 1.0, 0.5
W5_PILLARS = (1 / 12, 0.25, 0.5, 1.0, 2.0)
DAILY = 1.0 / 252.0
DEFAULT_SCHEME = SchemeConfig()
LOG_EULER = SchemeConfig(weak_order2=False, local_var_time_average=False)
PREDICTOR_CORRECTOR = SchemeConfig(weak_order2=False, predictor_corrector=True)


#: First time of the shared Dupire grid (1/365, the library's and the specification's).  The
#: owner's hypothesis that S3's dip came from it was tested and refuted (SPEC §8.7, second
#: follow-up); ``scripts/lcm_synthetic.py where --case s3 --t-min`` replaces it to show that.
GRID_T_MIN = 1.0 / 365.0


def lc_grid(horizon: float) -> LocalVolConfig:
    """The shared Dupire grid of the specification for a horizon up to 1y."""
    return LocalVolConfig(
        t_min=GRID_T_MIN, t_max=horizon + 0.02, n_t=120, k_min=-2.0, k_max=2.0, n_k=1601
    )


def w5_surfaces(n: int = 5, curve: ForwardCurve | None = None) -> list:  # type: ignore[type-arg]
    fc = curve or ForwardCurve.flat(1.0, 0.0, 0.0)
    out = []
    for vol, rho in zip(W5_ATM[:n], W5_RHO[:n], strict=True):
        cfg = SSVIConfig(W5_PILLARS, (vol,) * len(W5_PILLARS), rho, W5_ETA, W5_GAMMA, 3.0)
        out.append(surface_from_config(cfg, fc, fc.rate_curve))
    return out


def w5_models(horizon: float, n: int = 5, curve: ForwardCurve | None = None) -> list[LocalVol]:
    """The first ``n`` names of W5 as Dupire local-vol models on the shared grid."""
    cfg = lc_grid(horizon)
    return [
        LocalVol(LocalVolSurface.from_implied(s, cfg), s.forward_curve)
        for s in w5_surfaces(n, curve)
    ]


def flat_local_vol(vol: float, cfg: LocalVolConfig, curve: ForwardCurve) -> LocalVol:
    """A Black–Scholes name as a ``LocalVol`` with a flat table on the shared grid."""
    t = np.array([cfg.t_min, cfg.t_max])
    k = np.linspace(cfg.k_min, cfg.k_max, cfg.n_k)
    return LocalVol(LocalVolSurface(t, k, np.full((2, k.size), vol * vol), curve), curve)


@pytest.fixture(scope="module")
def w5_3m() -> list[LocalVol]:
    return w5_models(0.25)


def lc_model(
    models: list[LocalVol],
    family: CorrelationFamily,
    lam: LocalCorrelationFunction,
    weights: np.ndarray | None = None,
    mode: str = "performance",
) -> LocalCorrelationModel:
    w = W5_WEIGHTS[: len(models)] / W5_WEIGHTS[: len(models)].sum() if weights is None else weights
    basket = BasketSpec(w, mode, [m.forward_curve for m in models])
    return LocalCorrelationModel(models, family, lam, basket)


def k_grid_of(models: list[LocalVol]) -> np.ndarray:
    return models[0].local_vol.k_grid


def sloped_lambda(
    models: list[LocalVol], family: CorrelationFamily, horizon: float, slope: float = 4.0
) -> LocalCorrelationFunction:
    """A time-homogeneous ``λ`` falling with the basket (clipped at 0 and at the family's cap):
    correlation rises when the basket falls."""
    p = ParametricLambda(0.45, slope, 0.0, family.lambda_max)
    return LocalCorrelationFunction.parametric(p, [0.0, horizon], k_grid_of(models))


def _same_paths(a: MultiPathSet, b: MultiPathSet) -> None:
    """Bit for bit: every asset's log-spot, variance, integrated variance and sum of squares in
    every column."""
    assert a.n_assets == b.n_assets and a.n_cols == b.n_cols
    for x, y in zip(a.assets, b.assets, strict=True):
        np.testing.assert_array_equal(x.times, y.times)
        for name in ("log_spot", "variance", "int_var", "sum_sq"):
            np.testing.assert_array_equal(getattr(x, name), getattr(y, name), err_msg=name)


class _Recorded:
    """One asset's recorded mixed normals as the draws object of the single-asset kernel."""

    n_brownians = 1

    def __init__(self, z: np.ndarray) -> None:
        self.z = z  # (n_paths, n_steps)
        self.n_paths, self.n_steps = z.shape

    def block(self, step0: int, step1: int, p0: int, p1: int) -> np.ndarray:
        return np.ascontiguousarray(self.z[p0:p1, step0:step1, None])


# ---------------------------------------------------------------------------------------------
# I8 (rows): the λ function
# ---------------------------------------------------------------------------------------------


def test_lambda_function_rows(tmp_path: Path) -> None:
    """I8: rows are linear in ``t`` between slices, the first slice below ``t_0`` and the last
    beyond the horizon, the slice itself at a slice time; the step rows are the start-of-step
    rows (frozen rule); ``k`` is interpolated linearly and flat outside the grid; the parametric
    family is exact at the nodes; the function survives a save / load round trip."""
    times = np.array([0.1, 0.5, 1.0])
    k = np.linspace(-1.0, 1.0, 5)
    values = np.array(
        [[0.8, 0.6, 0.4, 0.2, 0.0], [0.9, 0.7, 0.5, 0.3, 0.1], [1.0, 0.9, 0.5, 0.1, 0.0]]
    )
    lam = LocalCorrelationFunction(times, k, values, {"seed": 7})
    assert lam.n_slices == 3 and lam.horizon == 1.0 and lam.k0 == -1.0 and lam.dk == 0.5
    np.testing.assert_array_equal(lam.rows(times), values)  # the slices, bit for bit
    np.testing.assert_allclose(lam.rows([0.3])[0], 0.5 * (values[0] + values[1]), atol=1e-15)
    np.testing.assert_array_equal(lam.rows([0.0, 0.05])[0], values[0])  # the first slice below t_0
    np.testing.assert_array_equal(lam.rows([0.05])[0], values[0])
    np.testing.assert_array_equal(lam.rows([1.0, 7.0])[1], values[2])  # held beyond the horizon
    # frozen rule: the row of a step is the row at its start
    nodes = np.array([0.1, 0.3, 0.5, 0.75, 1.0])
    np.testing.assert_array_equal(lam.step_rows(nodes), lam.rows(nodes[:-1]))
    assert lam.step_rows(nodes).shape == (4, 5)
    np.testing.assert_array_equal(lam.step_rows(nodes)[2], values[1])
    with pytest.raises(ValueError):
        lam.step_rows([0.5])
    # in k: linear, flat outside
    assert float(lam(0.5, 0.25)) == pytest.approx(0.4)
    assert float(lam(0.5, -3.0)) == 0.9 and float(lam(0.5, 3.0)) == 0.1
    assert float(lam(0.3, 0.0)) == pytest.approx(0.45)
    np.testing.assert_allclose(lam([0.1, 1.0], [[-1.0], [1.0]]), [[0.8, 1.0], [0.0, 0.0]])
    # one slice: the same row at every time
    single = LocalCorrelationFunction([0.0], k, values[:1])
    np.testing.assert_array_equal(single.rows([0.0, 0.4, 9.0]), np.repeat(values[:1], 3, axis=0))
    # constant and parametric
    const = LocalCorrelationFunction.constant(0.3, [0.0, 0.25], k)
    assert np.all(const.values == 0.3) and const.metadata["kind"] == "constant"
    p = ParametricLambda(0.4, 0.5, 0.05, 0.6)
    np.testing.assert_allclose(p(k), [0.6, 0.6, 0.4, 0.15, 0.05], atol=1e-15)
    par = LocalCorrelationFunction.parametric(p, [0.0, 1.0], k)
    np.testing.assert_array_equal(par.values[0], p(k))
    np.testing.assert_array_equal(par.values[1], p(k))
    fine = np.linspace(-2.0, 2.0, 1601)
    tab = LocalCorrelationFunction.parametric(p, [0.0, 1.0], fine)
    probe = np.linspace(-1.9, 1.9, 4001)
    err = np.abs(tab(0.5, probe) - p(probe))
    kinks = (0.4 - np.array([0.6, 0.05])) / 0.5  # where the clips start to bind
    away = np.all(np.abs(probe[:, None] - kinks[None, :]) > 0.0025, axis=1)
    # exact off the two kink cells, up to the rounding of the uniform-grid lookup (the grid
    # step is read off the grid: 1e-13 relative, the convention of every table in the library)
    assert err[away].max() < 1e-12 and err.max() < 0.5 * 0.0025
    # the reference's form on the equicorrelation family
    q = ParametricLambda.from_rho(0.2, 4.0, 0.02, 0.98)
    fam = CorrelationFamily.equi(5)
    assert q.lam_lo == 0.0 and q.lam_hi == fam.lambda_max
    kk = np.linspace(-0.4, 0.4, 81)
    np.testing.assert_allclose(
        fam.equicorrelation(q(kk)), np.clip(0.2 - 4.0 * kk, 0.02, 0.98), rtol=0, atol=1e-15
    )
    assert q.to_rho(0.02) == pytest.approx((0.2, 4.0), abs=1e-15)
    # average and round trip
    avg = LocalCorrelationFunction.average(lam, lam.with_values(values[::-1], seed=8))
    np.testing.assert_allclose(avg.values, 0.5 * (values + values[::-1]))
    assert avg.metadata["averaged_with"] == 8
    back = LocalCorrelationFunction.load(lam.save(tmp_path / "lambda.npz"))
    np.testing.assert_array_equal(back.values, values)
    np.testing.assert_array_equal(back.times, times)
    np.testing.assert_array_equal(back.k_grid, k)
    assert back.metadata == {"seed": 7} and "LocalCorrelationFunction" in repr(back)
    for bad in (values + 0.5, -values, np.full_like(values, np.nan)):
        with pytest.raises(ValueError):
            LocalCorrelationFunction(times, k, bad)
    with pytest.raises(ValueError):
        LocalCorrelationFunction(times[::-1], k, values)
    with pytest.raises(ValueError):
        LocalCorrelationFunction(times, np.array([-1.0, 0.0, 0.5, 0.6, 1.0]), values)
    with pytest.raises(ValueError):
        LocalCorrelationFunction.average(lam, const)
    with pytest.raises(ValueError):
        ParametricLambda(0.5, 1.0, 0.6, 0.4)


# ---------------------------------------------------------------------------------------------
# the basket state and the model's validation
# ---------------------------------------------------------------------------------------------


def test_basket_spec_and_model_validation(w5_3m: list[LocalVol]) -> None:
    """The two basket modes' shifts and forwards; the model refuses names on different grids,
    names that are not local-vol models, sizes that disagree and a product beyond the last
    ``λ`` slice; ``aux`` is concatenated with the paths."""
    carry = [ForwardCurve.flat(s, 0.04, q) for s, q in ((1.0, 0.00), (50.0, 0.03), (2.0, 0.01))]
    w = np.array([0.5, 0.3, 0.2])
    t = np.array([0.0, 0.25, 1.0])
    perf = BasketSpec(w, "performance", carry)
    np.testing.assert_allclose(
        perf.shifts(t), np.stack([np.asarray(fc.log_forward(t)) for fc in carry]), atol=1e-15
    )
    np.testing.assert_array_equal(perf.log_basket_forward(t), np.zeros(3))
    price = perf.with_mode("carry")
    np.testing.assert_allclose(price.shifts(t), np.log([[1.0] * 3, [50.0] * 3, [2.0] * 3]))
    ratios = np.stack([np.asarray(fc.forward(t)) / fc.spot for fc in carry])
    np.testing.assert_allclose(price.log_basket_forward(t), np.log(w @ ratios), atol=1e-15)
    # at the forwards both baskets are at the money; at the spots the price basket starts at 0
    ls_fwd = np.array([float(fc.log_forward(0.25)) for fc in carry])
    assert float(perf.log_moneyness(ls_fwd, 0.25)) == pytest.approx(0.0, abs=1e-15)
    assert float(price.log_moneyness(ls_fwd, 0.25)) == pytest.approx(0.0, abs=1e-15)
    assert float(price.log_moneyness(np.log([1.0, 50.0, 2.0]), 0.0)) == pytest.approx(
        0.0, abs=1e-15
    )
    # zero carry: the two modes coincide
    flat = [ForwardCurve.flat(1.0, 0.0, 0.0)] * 3
    a, b = BasketSpec(w, "performance", flat), BasketSpec(w, "carry", flat)
    np.testing.assert_array_equal(a.shifts(t), b.shifts(t))
    np.testing.assert_array_equal(a.log_basket_forward(t), b.log_basket_forward(t))
    for bad_w, bad_mode, curves in (
        ([0.5, 0.3, 0.3], "performance", flat),
        ([0.7, 0.5, -0.2], "performance", flat),
        (w, "spot", flat),
        (w, "carry", flat[:2]),
    ):
        with pytest.raises(ValueError):
            BasketSpec(bad_w, bad_mode, curves)
    # the model
    fam = CorrelationFamily.equi(5)
    lam = sloped_lambda(w5_3m, fam, 0.25)
    model = lc_model(w5_3m, fam, lam)
    assert model.n_assets == 5 and model.names == tuple(f"asset{i}" for i in range(5))
    np.testing.assert_array_equal(model.spots, np.ones(5))
    np.testing.assert_array_equal(model.required_times(), [0.0, 0.25])
    assert model.describe()["mode"] == "performance" and "LocalCorrelationModel" in repr(model)
    other_grid = LocalVolConfig(t_min=1 / 365, t_max=0.27, n_t=20, k_min=-1.5, k_max=1.5, n_k=601)
    odd = flat_local_vol(0.2, other_grid, w5_3m[0].forward_curve)
    with pytest.raises(ValueError, match="one uniform k grid"):
        lc_model([*w5_3m[:4], odd], fam, lam)
    bs = BlackScholes(0.2, w5_3m[0].forward_curve)
    with pytest.raises(ValueError, match="local-vol names"):
        lc_model([*w5_3m[:4], bs], fam, lam)  # type: ignore[list-item]
    with pytest.raises(ValueError, match="number of assets"):
        lc_model(w5_3m[:4], fam, lam)
    grid = TimeGrid.build([0.5], DAILY)
    with pytest.raises(ValueError, match="beyond the calibration horizon"):
        model.simulate_chunk(grid, model.draws_for(grid, 1, 100), 0, 100, DEFAULT_SCHEME)
    # aux travels with the paths
    grid = TimeGrid.build([0.1, 0.25], DAILY, calibration_grid=model.required_times())
    draws = model.draws_for(grid, 5, 400)
    whole = model.simulate_chunk(grid, draws, 0, 400, DEFAULT_SCHEME)
    assert whole.aux is not None and sorted(whole.aux) == ["k_basket", "lam_int"]
    assert whole.aux["k_basket"].shape == (400, 3) and np.all(whole.aux["lam_int"][:, 0] == 0.0)
    np.testing.assert_allclose(whole.aux["k_basket"][:, 0], 0.0, atol=3e-16)
    assert np.all(np.diff(whole.aux["lam_int"], axis=1) >= 0.0)
    parts = [model.simulate_chunk(grid, draws, p0, p0 + 200, DEFAULT_SCHEME) for p0 in (0, 200)]
    joined = MultiPathSet.concat(parts)
    _same_paths(joined, whole)
    assert joined.aux is not None
    for key in ("k_basket", "lam_int"):
        np.testing.assert_array_equal(joined.aux[key], whole.aux[key])
    # the recorded basket log-moneyness is the basket's, from the recorded spots
    for col in (1, 2):
        np.testing.assert_allclose(
            whole.aux["k_basket"][:, col],
            model.basket.log_moneyness(np.log(whole.spots(col)), float(grid.record_times[col])),
            rtol=0,
            atol=1e-14,
        )
    plain = MultiPathSet(whole.assets, whole.names)
    with pytest.raises(ValueError, match="aux"):
        MultiPathSet.concat([whole, plain])
    with pytest.raises(ValueError, match="one row per path"):
        MultiPathSet(whole.assets, whole.names, {"x": np.zeros((3, 2))})


# ---------------------------------------------------------------------------------------------
# I1, I3, I4, I5: the kernel's identities (bit for bit)
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scheme", [DEFAULT_SCHEME, LOG_EULER, PREDICTOR_CORRECTOR])
@pytest.mark.parametrize("which", ["equi", "general", "general_high", "carry"])
def test_lambda_zero_reproduces_multi_asset_model(which: str, scheme: SchemeConfig) -> None:
    """I1: with ``λ ≡ 0`` the local correlation model is ``MultiAssetModel(models, R_low)`` on
    the same asset streams (``draws.low``), seed and grid — every asset's log-spot, variance,
    integrated variance and sum of squares in every column, bit for bit — for the
    equicorrelation and for a general ``R_low`` (and a general ``R_high``, and names with
    carry and spots away from 1), under two chunkings of the paths and of the steps."""
    horizon, n_paths, seed = 0.25, 6000, 31
    if which == "equi":
        models = w5_models(horizon)
        family = CorrelationFamily.equi(5)
    elif which == "carry":
        curves = [ForwardCurve.flat(s, 0.04, q) for s, q in ((1.0, 0.0), (80.0, 0.03), (3.0, 0.06))]
        surfaces = [w5_surfaces(i + 1, curves[i])[i] for i in range(3)]
        models = [
            LocalVol(LocalVolSurface.from_implied(s, lc_grid(horizon)), s.forward_curve)
            for s in surfaces
        ]
        family = CorrelationFamily.equi(3, 0.3, 0.9)
    else:
        models = w5_models(horizon, 4)
        family = CorrelationFamily(R_LOW_4, R_HIGH_4 if which == "general_high" else None)
    n = len(models)
    lam = LocalCorrelationFunction.constant(0.0, [0.0, horizon], k_grid_of(models))
    mode = "carry" if which == "carry" else "performance"
    model = lc_model(models, family, lam, mode=mode)
    reference = MultiAssetModel(models, family.r_low)
    grid = TimeGrid.build(
        [horizon / 3, 2 * horizon / 3, horizon], DAILY, calibration_grid=model.required_times()
    )
    assert grid.n_steps >= 63
    draws = model.draws_for(grid, seed, n_paths)
    ref = reference.simulate_chunk(grid, draws.low, 0, n_paths, scheme)
    got = model.simulate_chunk(grid, draws, 0, n_paths, scheme)
    _same_paths(got, ref)
    assert got.aux is not None and np.all(got.aux["lam_int"] == 0.0)
    # a second chunking: three path chunks, seven steps per block
    parts = [
        model.simulate_chunk(grid, draws, p0, p0 + 2000, scheme, step_block=7)
        for p0 in range(0, n_paths, 2000)
    ]
    _same_paths(MultiPathSet.concat(parts), ref)
    # the same draws object through the Monte Carlo loop and a fresh constant-correlation one
    fresh = MultiAssetModel(models, family.r_low).simulate_chunk(
        grid, CorrelatedDraws(seed, n_paths, grid.n_steps, family.r_low), 0, n_paths, scheme
    )
    _same_paths(got, fresh)
    assert n == got.n_assets


def test_asset_paths_equal_single_asset_kernel(w5_3m: list[LocalVol]) -> None:
    """I3: with a state-dependent ``λ`` each asset's path is ``LocalVol.simulate_chunk`` fed with
    that asset's recorded mixed normals, bit for bit (the joint kernel calls the library's
    shared spot step on each name)."""
    horizon, n_paths = 0.25, 4000
    for family in (CorrelationFamily.equi(5), CorrelationFamily(R_LOW_4, R_HIGH_4)):
        models = w5_3m[: family.n]
        model = lc_model(models, family, sloped_lambda(models, family, horizon))
        grid = TimeGrid.build([0.1, horizon], DAILY, calibration_grid=model.required_times())
        draws = model.draws_for(grid, 77, n_paths)
        for scheme in (DEFAULT_SCHEME, LOG_EULER, PREDICTOR_CORRECTOR):
            paths = model.simulate_chunk(grid, draws, 0, n_paths, scheme, record_normals=True)
            assert paths.aux is not None
            z = paths.aux["normals"]
            assert z.shape == (n_paths, grid.n_steps, family.n)
            lam_T = paths.aux["lam_int"][:, -1] / horizon
            assert lam_T.std() > 0.02  # λ did vary across paths
            for i, m in enumerate(models):
                alone = m.simulate_chunk(grid, _Recorded(z[:, :, i]), 0, n_paths, scheme)  # type: ignore[arg-type]
                for name in ("log_spot", "variance", "int_var", "sum_sq"):
                    np.testing.assert_array_equal(
                        getattr(paths.assets[i], name), getattr(alone, name), err_msg=name
                    )
        # the recorded normals do not depend on the block size
        again = model.simulate_chunk(
            grid, draws, 0, n_paths, PREDICTOR_CORRECTOR, step_block=5, record_normals=True
        )
        assert again.aux is not None
        np.testing.assert_array_equal(again.aux["normals"], z)


def test_single_name_degeneracy_of_the_kernel(w5_3m: list[LocalVol]) -> None:
    """I4 (the kernel's half; the calibration's is ``test_single_name_calibration``): with one
    name and ``λ ≡ 0`` the model is ``LocalVol`` on the asset stream, bit for bit."""
    horizon, n_paths, seed = 0.25, 5000, 13
    name = w5_3m[2]
    family = CorrelationFamily.equi(1)
    lam = LocalCorrelationFunction.constant(0.0, [0.0, horizon], k_grid_of(w5_3m))
    model = lc_model([name], family, lam, weights=np.array([1.0]))
    grid = TimeGrid.build([0.1, horizon], DAILY, calibration_grid=model.required_times())
    paths = model.simulate_chunk(
        grid, model.draws_for(grid, seed, n_paths), 0, n_paths, DEFAULT_SCHEME
    )
    alone = name.simulate_chunk(
        grid, GaussianDraws(seed, n_paths, grid.n_steps, 1), 0, n_paths, DEFAULT_SCHEME
    )
    for field in ("log_spot", "variance", "int_var", "sum_sq"):
        np.testing.assert_array_equal(getattr(paths.assets[0], field), getattr(alone, field))
    assert paths.aux is not None
    # the basket is the name: k_B = ln(exp(ln S)) up to the rounding of exp and log
    np.testing.assert_allclose(paths.aux["k_basket"], alone.log_spot, rtol=0, atol=1e-15)


def test_identical_names_lambda_one(w5_3m: list[LocalVol]) -> None:
    """I5: identical names with ``λ ≡ 1`` and ``rho_max = 1`` (so ``λ_max = 1``; the default cap
    of 0.9796 would clip ``λ``): every asset follows the same path, the ``LocalVol`` path on
    the ``η`` stream, bit for bit."""
    horizon, n_paths, seed = 0.25, 5000, 19
    name = w5_3m[1]
    family = CorrelationFamily.equi(4, 0.02, 1.0)
    assert family.lambda_max == 1.0
    lam = LocalCorrelationFunction.constant(1.0, [0.0, horizon], k_grid_of(w5_3m))
    model = lc_model([name] * 4, family, lam, weights=np.full(4, 0.25))
    grid = TimeGrid.build([horizon], DAILY, calibration_grid=model.required_times())
    paths = model.simulate_chunk(
        grid, model.draws_for(grid, seed, n_paths), 0, n_paths, DEFAULT_SCHEME
    )
    common = GaussianDraws(seed + COMMON_FACTOR_SEED_OFFSET, n_paths, grid.n_steps, 1)
    alone = name.simulate_chunk(grid, common, 0, n_paths, DEFAULT_SCHEME)
    for asset in paths.assets:
        for field in ("log_spot", "variance", "int_var", "sum_sq"):
            np.testing.assert_array_equal(getattr(asset, field), getattr(alone, field))
    # with the default cap the kernel clips λ to 0.9796 and the names decouple
    capped = lc_model([name] * 4, CorrelationFamily.equi(4), lam, weights=np.full(4, 0.25))
    out = capped.simulate_chunk(
        grid, capped.draws_for(grid, seed, n_paths), 0, n_paths, DEFAULT_SCHEME
    )
    assert not np.array_equal(out.assets[0].log_spot, out.assets[1].log_spot)
    assert out.aux is not None
    np.testing.assert_allclose(
        out.aux["lam_int"][:, -1], CorrelationFamily.equi(4).lambda_max * horizon, rtol=1e-13
    )


# ---------------------------------------------------------------------------------------------
# I9: the path-wise identities of the dispersion payoffs
# ---------------------------------------------------------------------------------------------


def test_pathwise_identities(w5_3m: list[LocalVol]) -> None:
    """I9, on local correlation paths (W5, 2·10⁴ paths): ``V = Σ w R² − R̄²`` (1e-14),
    ``D ≤ √V`` (1e-15), the sandwich ``Σ w|R| − |R̄| ≤ D ≤ Σ w|R| + |R̄|`` (1e-15) and the gap
    formula ``D − (Σ w|R| − |R̄|) = 2 Σ w (|R̄| − (ε R)⁺)⁺`` with ``ε = sgn R̄`` (1e-14)."""
    horizon, n_paths = 0.25, 20_000
    fam = CorrelationFamily.equi(5)
    model = lc_model(w5_3m, fam, sloped_lambda(w5_3m, fam, horizon))
    sim = SimConfig(n_paths=n_paths, dt_max=DAILY, seed=3, chunk_size=10_000)
    mc = MultiAssetMonteCarlo(sim)
    zero = DiscountCurve.flat(0.0)
    product = Palladium(W5_WEIGHTS, 0.0, horizon, zero)
    grid = mc.build_grid([product], model)
    paths = mc.simulate(model, grid)
    w = W5_WEIGHTS
    r = paths.performances(grid.fixing_index[horizon])
    rb = r @ w
    dev = r - rb[:, None]
    d = np.abs(dev) @ w
    v = (dev * dev) @ w
    abs_r = np.abs(r) @ w
    np.testing.assert_array_equal(d, product.dispersion(paths, grid.fixing_index))
    np.testing.assert_allclose(v, (r * r) @ w - rb * rb, rtol=0, atol=1e-14)
    assert np.all(d <= np.sqrt(v) + 1e-15)
    assert np.all(abs_r - np.abs(rb) <= d + 1e-15) and np.all(d <= abs_r + np.abs(rb) + 1e-15)
    gap = d - (abs_r - np.abs(rb))
    np.testing.assert_allclose(gap, gap_formula(1.0 + r, w), rtol=0, atol=1e-14)
    assert np.all(gap >= -1e-15) and np.all(gap <= 2.0 * np.abs(rb) + 1e-15)
    # through the Monte Carlo loop: the model satisfies the MultiModel protocol
    price = mc.price(product, model)
    assert price.mean == pytest.approx(float(d.mean()), abs=1e-15) and price.stderr > 0


# ---------------------------------------------------------------------------------------------
# C1 (kernel part) and the constant-λ model against the constant-correlation layer
# ---------------------------------------------------------------------------------------------


def test_kernel_normals_statistics(w5_3m: list[LocalVol]) -> None:
    """C1 inside the kernel, where ``λ`` is read on each path's own state: per asset the mixed
    normals have mean 0, variance 1 and no lag-1 autocorrelation, and within each decile of
    the step's ``λ`` the cross moments equal ``ρ(λ)`` — all within 4 standard errors (10⁵
    independent paths, 21 daily steps, a steep ``λ`` so that every level is visited)."""
    horizon, n_paths = 1 / 12, 100_000
    fam = CorrelationFamily.equi(5)
    model = lc_model(w5_3m, fam, sloped_lambda(w5_3m, fam, 0.25, slope=12.0))
    grid = TimeGrid.build([horizon], DAILY, record_all_steps=True)
    draws = model.draws_for(grid, 41, n_paths, antithetic=False)
    paths = model.simulate_chunk(grid, draws, 0, n_paths, DEFAULT_SCHEME, record_normals=True)
    assert paths.aux is not None
    z = paths.aux["normals"]
    lam = np.diff(paths.aux["lam_int"], axis=1) / grid.dts[None, :]
    assert lam.min() >= -1e-12 and lam.max() <= fam.lambda_max + 1e-12
    assert np.all(lam[:, 0] == pytest.approx(0.45))  # every path starts at the money
    n = fam.n
    pooled = z.reshape(-1, n)
    count = pooled.shape[0]
    worst = {"mean": 0.0, "variance": 0.0, "lag-1": 0.0, "cross": 0.0}
    for i in range(n):
        worst["mean"] = max(worst["mean"], abs(float(pooled[:, i].mean())) * np.sqrt(count))
        worst["variance"] = max(
            worst["variance"], abs(float(pooled[:, i].var()) - 1.0) / np.sqrt(2.0 / count)
        )
        prod = (z[:, :-1, i] * z[:, 1:, i]).ravel()
        worst["lag-1"] = max(
            worst["lag-1"], abs(float(prod.mean())) / (float(prod.std()) / np.sqrt(prod.size))
        )
    # deciles over the steps after the first (where λ is the same on every path)
    flat = lam[:, 1:].ravel()
    later = z[:, 1:, :].reshape(-1, n)
    bucket = _deciles(flat)
    means = np.array([float(flat[bucket == dec].mean()) for dec in range(10)])
    rows = []
    for dec in range(10):
        sel = bucket == dec
        target = float(fam.equicorrelation(means[dec]))
        for i in range(n):
            for k in range(i + 1, n):
                prod = later[sel, i] * later[sel, k]
                se = float(prod.std(ddof=1)) / np.sqrt(prod.size)
                zscore = (float(prod.mean()) - target) / se
                rows.append((dec, i, k, target, float(prod.mean()), zscore))
                worst["cross"] = max(worst["cross"], abs(zscore))
    table = "\n".join(
        f"decile {d} pair ({i},{k}): {m:+.4f} vs rho {t:+.4f}  z {zs:+.2f}"
        for d, i, k, t, m, zs in rows
    )
    print(
        f"kernel normals: decile means of lambda {means[0]:.3f} .. {means[-1]:.3f}; worst |z| "
        + ", ".join(f"{k} {v:.2f}" for k, v in worst.items())
    )
    assert len(rows) == 100 and all(np.isfinite(row[-1]) for row in rows)
    assert means[-1] - means[0] > 0.5, "λ did not vary enough for the test to bind"
    assert all(v < 4.0 for v in worst.values()), f"{worst}\n{table}"


def test_constant_lambda_matches_constant_correlation(w5_3m: list[LocalVol]) -> None:
    """A constant ``λ`` is the constant correlation ``ρ = ρ_min + λ(1 − ρ_min)`` in law: the
    Palladium forward, a call and the basket's second moment agree with ``MultiAssetModel`` at
    that correlation within 3 combined standard errors (different mixings of the same asset
    streams, so the two estimates are not paired)."""
    horizon, lam_c = 0.25, 0.5
    fam = CorrelationFamily.equi(5)
    rho = float(fam.equicorrelation(lam_c))
    assert rho == pytest.approx(0.51)
    const = LocalCorrelationFunction.constant(lam_c, [0.0, horizon], k_grid_of(w5_3m))
    model = lc_model(w5_3m, fam, const)
    reference = MultiAssetModel(w5_3m, constant_correlation(5, rho))
    zero = DiscountCurve.flat(0.0)
    products = [
        Palladium(W5_WEIGHTS, 0.0, horizon, zero),
        Palladium(W5_WEIGHTS, 0.08, horizon, zero),
    ]
    mc_a = MultiAssetMonteCarlo(
        SimConfig(n_paths=60_000, dt_max=DAILY, seed=101, chunk_size=20_000)
    )
    mc_b = MultiAssetMonteCarlo(
        SimConfig(n_paths=60_000, dt_max=DAILY, seed=202, chunk_size=20_000)
    )
    got = mc_a.price_many(products, model)
    ref = mc_b.price_many(products, reference)
    for g, r in zip(got, ref, strict=True):
        assert abs(g.mean - r.mean) < 3.0 * float(np.hypot(g.stderr, r.stderr)), (g, r)
    assert got[0].mean > 10 * got[0].stderr


# ---------------------------------------------------------------------------------------------
# the per-particle variance terms
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["performance", "carry"])
def test_lc_ab_matches_the_matrix_form(mode: str) -> None:
    """``lc_ab`` (the calibration's per-particle kernel) against ``u = ω·σ`` and
    ``a = uᵀR_low u``, ``b = uᵀR_high u − a`` in matrix form, for the four combinations of an
    equicorrelation / general ``R_low`` and ``11ᵀ`` / general ``R_high``, with carry."""
    horizon, t = 0.25, 0.1
    curves = [
        ForwardCurve.flat(s, 0.04, q)
        for s, q in ((1.0, 0.0), (80.0, 0.03), (3.0, 0.06), (1.0, 0.02))
    ]
    models = [
        LocalVol(
            LocalVolSurface.from_implied(w5_surfaces(i + 1, curves[i])[i], lc_grid(horizon)),
            curves[i],
        )
        for i in range(4)
    ]
    w = np.array([0.4, 0.3, 0.2, 0.1])
    basket = BasketSpec(w, mode, curves)
    rng = np.random.default_rng(4)
    ln_f = np.array([float(fc.log_forward(t)) for fc in curves])
    ls = ln_f[None, :] + rng.normal(0.0, 0.15, (500, 4))
    var_rows = np.ascontiguousarray(
        np.stack([m.local_vol.var_time_average([t, t + DAILY])[0] for m in models])
    )
    lv0 = models[0].local_vol
    shifts = np.ascontiguousarray(basket.shifts([t])[:, 0])
    ln_fb = float(basket.log_basket_forward([t])[0])
    k_i = ls - ln_f[None, :]
    sigma = np.sqrt(
        np.stack([np.interp(k_i[:, i], lv0.k_grid, var_rows[i]) for i in range(4)], axis=1)
    )
    level = np.exp(ls - shifts[None, :]) @ w
    omega = w[None, :] * np.exp(ls - shifts[None, :]) / level[:, None]
    np.testing.assert_allclose(omega.sum(axis=1), 1.0, atol=1e-14)
    u = omega * sigma
    for family in (
        CorrelationFamily.equi(4),
        CorrelationFamily(constant_correlation(4, 0.1), R_HIGH_4),
        CorrelationFamily(R_LOW_4),
        CorrelationFamily(R_LOW_4, R_HIGH_4),
    ):
        a, b, k = np.empty(500), np.empty(500), np.empty(500)
        lc_ab(
            ls, ln_f, w, shifts, ln_fb, lv0.k0, lv0.dk, var_rows,
            family.low_equi, family.rho_low if family.rho_low is not None else 0.0, family.r_low,
            family.high_ones, family.l_high, a, b, k,
        )  # fmt: skip
        ref_a, ref_b = family.variance_terms(u)
        np.testing.assert_allclose(a, ref_a, rtol=1e-12, atol=0)
        np.testing.assert_allclose(b, ref_b, rtol=1e-10, atol=1e-16)
        np.testing.assert_allclose(k, basket.log_moneyness(ls, t), rtol=0, atol=1e-14)
        assert np.all(a > 0) and np.all(b >= 0)


# ---------------------------------------------------------------------------------------------
# LC4: the particle calibration on W5 (fast: 5·10⁴ particles)
# ---------------------------------------------------------------------------------------------

#: an index target within W5's reach (module docstring)
W5_INDEX = SSVIConfig(W5_PILLARS, (0.21,) * len(W5_PILLARS), -0.7, 0.9, 0.5, 3.0)
FAST_PARTICLE = ParticleConfig(n_particles=50_000, horizon=0.25, seed=12345)
FAST_SIM = SimConfig(n_paths=100_000, dt_max=DAILY, seed=2024, chunk_size=20_000)


def w5_basket(models: list[LocalVol], mode: str = "performance") -> BasketSpec:
    w = W5_WEIGHTS[: len(models)] / W5_WEIGHTS[: len(models)].sum()
    return BasketSpec(w, mode, [m.forward_curve for m in models])


def w5_target(horizon: float, cfg: SSVIConfig = W5_INDEX) -> tuple[ImpliedSurface, LocalVolSurface]:
    """The fast index target of W5: its implied surface and its Dupire surface on the grid."""
    fc = ForwardCurve.flat(1.0, 0.0, 0.0)
    surface = surface_from_config(cfg, fc, fc.rate_curve)
    return surface, LocalVolSurface.from_implied(surface, lc_grid(horizon))


@pytest.fixture(scope="module")
def w5_calibration(w5_3m: list[LocalVol]) -> lcal.LCCalibrationResult:
    """W5 at 3m against its fast target: 5·10⁴ particles, daily steps, the default settings."""
    surface, lv = w5_target(0.25)
    return lcal.calibrate_local_correlation(
        w5_3m,
        CorrelationFamily.equi(5),
        w5_basket(w5_3m),
        surface,
        lv,
        FAST_PARTICLE,
        FAST_SIM,
        LocalCorrelationConfig(particle=FAST_PARTICLE),
        snapshot_times=[0.1],
    )


def test_calibration_and_pricing_share_the_kernel(
    w5_3m: list[LocalVol], w5_calibration: lcal.LCCalibrationResult
) -> None:
    """I2: the calibrated model, simulated with the calibration's seed, particle count and grid,
    reproduces the final particle cloud bit for bit — and the cloud kept at a snapshot time —
    whatever the block size; the engine's pricing grid contains every ``λ`` slice."""
    res = w5_calibration
    fam = CorrelationFamily.equi(5)
    model = LocalCorrelationModel(w5_3m, fam, res.lam, w5_basket(w5_3m))
    n = FAST_PARTICLE.n_particles
    assert res.grid.n_steps == 63 and res.lam.n_slices == 64 and res.passes == 1
    np.testing.assert_array_equal(res.lam.times, res.grid.times)
    grid = TimeGrid.build([FAST_PARTICLE.horizon], FAST_SIM.dt_max)
    np.testing.assert_array_equal(grid.times, res.grid.times)
    draws = LocalCorrelationDraws(
        FAST_PARTICLE.seed, n, grid.n_steps, fam, FAST_PARTICLE.antithetic
    )
    for step_block in (64, 5):
        paths = model.simulate_chunk(grid, draws, 0, n, FAST_SIM.scheme, step_block=step_block)
        for i in range(5):
            np.testing.assert_array_equal(paths.assets[i].log_spot[:, 1], res.final_log_spot[:, i])
    ((t_snap, cloud),) = res.snapshots.items()
    assert t_snap == res.grid.times[int(np.argmin(np.abs(res.grid.times - 0.1)))]
    # a product fixing at the snapshot time: the engine's grid keeps every λ slice as a node
    grid2 = TimeGrid.build(
        [t_snap, FAST_PARTICLE.horizon], FAST_SIM.dt_max, calibration_grid=model.required_times()
    )
    np.testing.assert_array_equal(grid2.times, res.grid.times)
    mid = model.simulate_chunk(grid2, draws, 0, n, FAST_SIM.scheme)
    for i in range(5):
        np.testing.assert_array_equal(mid.assets[i].log_spot[:, 1], cloud[:, i])
    mc = MultiAssetMonteCarlo(FAST_SIM)
    priced = mc.build_grid([Palladium(W5_WEIGHTS, 0.0, 0.25, DiscountCurve.flat(0.0))], model)
    np.testing.assert_array_equal(priced.times, res.grid.times)
    # the diagnostics of the fast calibration: no clipping on the reachable target
    assert res.max_clipped_mass < 0.01 and not res.unidentified.any()
    assert res.drift_abs_mean is None and res.bandwidths.shape == (63,)
    assert np.all(res.q_lo[1:] < 0) and np.all(res.q_hi[1:] > 0) and res.q_lo[0] == res.q_hi[0]
    assert (
        np.all(np.diff(res.q_hi[1:]) > -1e-3) and res.lam.metadata["code_tag"] == lcal.LC_CODE_TAG
    )
    assert 0.3 < res.lambda_mean[-1] < 0.6 and res.wall_time > 0 and set(res.timings) == {
        "tables", "draws", "kernel", "ab", "regression", "rows"
    }  # fmt: skip
    # λ falls with the basket: correlation rises on the downside
    lam_T = res.lam(0.25, np.array([-0.15, 0.0, 0.1]))
    assert lam_T[0] > lam_T[1] > lam_T[2]
    frame = lcal.lambda_surface_frame(res)
    assert list(frame.columns) == [
        "t", "k", "lam", "lam_star", "in_trusted_range", "q_lo", "q_hi", "m_low", "m_high"
    ]  # fmt: skip
    assert len(frame) == 64 * 1601 and frame["lam"].between(0.0, fam.lambda_max).all()
    inside = frame[frame["in_trusted_range"] & (frame["t"] > 0)]
    assert np.allclose(inside["lam"], np.clip(inside["lam_star"], 0.0, fam.lambda_max))
    assert json.loads(json.dumps(res.summary()))["n_slices"] == 64
    assert res.clip_intervals(63) == [] or all(
        x[0] in ("low", "high") for x in res.clip_intervals(63)
    )


def _ab_cloud(
    n: int = 60_000, seed: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = 0.1 * rng.standard_t(6, n) - 0.02 * rng.exponential(1.0, n)  # skewed, fat put tail
    a = 0.02 * np.exp(-1.0 * k) * np.exp(rng.normal(-0.02, 0.2, n))
    b = 0.05 * np.exp(-0.5 * k) * np.exp(rng.normal(-0.02, 0.2, n))
    return k, a, b, np.linspace(-2.0, 2.0, 1601)


def test_regression_helper_equals_two_estimates() -> None:
    """I7: ``conditional_expectations_ab`` equals two calls of the leverage's
    ``conditional_variance_estimate`` bit for bit, for every estimator and post-processing
    option it supports; the trusted range is the sorted quantile rule; and the signature of the
    private ``particle._finish_estimate`` it imports is pinned."""
    assert list(inspect.signature(_finish_estimate).parameters) == [
        "m", "slope", "kreg", "q_lo", "q_hi", "grid", "h", "cfg", "v", "tail_slope"
    ]  # fmt: skip
    k, a, b, grid = _ab_cloud()
    slope_a = float(np.polyfit(k, np.log(a), 1)[0])
    slope_b = float(np.polyfit(k, np.log(b), 1)[0])
    cases: list[tuple[dict, tuple[float | None, float | None]]] = [({}, (None, None))]
    cases += [({"tail_extrapolation": t}, (None, None)) for t in ("flat", "log_linear", "adaptive")]
    cases += [
        ({"tail_extrapolation": "cloud_slope"}, (slope_a, slope_b)),
        ({"bias_correction": False}, (None, None)),
        ({"kernel": "quartic"}, (None, None)),
        ({"regression": "nadaraya_watson", "tail_extrapolation": "flat"}, (None, None)),
        ({"min_window": 500, "min_window_fraction": 0.002}, (None, None)),
        ({"quantile_clip": 0.02, "n_regression_points": 51}, (None, None)),
        ({"estimator": "sorted"}, (None, None)),
        ({"estimator": "binned"}, (None, None)),
        ({"estimator": "binned", "tail_extrapolation": "flat"}, (None, None)),
    ]
    for changes, slopes in cases:
        cfg = ParticleConfig(n_particles=k.size, horizon=1.0, **changes)
        for h in (0.003, 0.02):
            ea, eb, q_lo, q_hi = lcal.conditional_expectations_ab(k, a, b, grid, h, cfg, slopes)
            np.testing.assert_array_equal(
                ea,
                conditional_variance_estimate(k, a, grid, h, cfg, slopes[0]),
                err_msg=str(changes),
            )
            np.testing.assert_array_equal(
                eb,
                conditional_variance_estimate(k, b, grid, h, cfg, slopes[1]),
                err_msg=str(changes),
            )
            ks = np.sort(k)
            assert q_lo == ks[int(cfg.quantile_clip * k.size)]
            assert q_hi == ks[int((1.0 - cfg.quantile_clip) * k.size) - 1]
            assert np.all(ea > 0) and np.all(eb > 0)
    # a degenerate cloud (one point) and a response that is zero everywhere (one name)
    one = np.full(4000, 0.1)
    for estimator in (None, "binned"):
        cfg = ParticleConfig(n_particles=4000, estimator=estimator)
        ea, eb, q_lo, q_hi = lcal.conditional_expectations_ab(
            one, np.full(4000, 0.04), np.full(4000, 0.01), grid, 0.01, cfg
        )
        assert np.all(ea == 0.04) and np.all(eb == 0.01) and q_lo == q_hi == 0.1
        ea, eb, _, _ = lcal.conditional_expectations_ab(
            k[:4000], a[:4000], np.zeros(4000), grid, 0.01, cfg
        )
        assert np.all(eb == 0.0) and np.all(ea > 0)


@pytest.mark.parametrize("average", ["step", "point"])
def test_lambda_t0_exact(w5_3m: list[LocalVol], average: str) -> None:
    """I8: the row at ``t_0`` is one value, ``clip((σ̄²_B(0) − a_0)/b_0, 0, λ_max)`` with ``u_i =
    w_i·σ̄_i(0)`` — the averages over the first step for ``"step"``, the values at ``t_0`` for
    ``"point"`` — computed here from the surfaces."""
    surface, lv = w5_target(0.25)
    fam = CorrelationFamily.equi(5)
    cfg = ParticleConfig(n_particles=2000, horizon=0.02, seed=1, min_window=200)
    lc = LocalCorrelationConfig(particle=cfg, target_average=average)
    res = lcal.calibrate_local_correlation(
        w5_3m, fam, w5_basket(w5_3m), surface, lv, cfg, FAST_SIM, lc
    )
    row = res.lam.values[0]
    assert np.all(row == row[0]) and np.all(res.lambda_star[0] == res.lambda_star[0, 0])
    first = [0.0, float(res.grid.times[1])]

    def at_money(s: LocalVolSurface) -> float:
        table = s.var_time_average(first)[0] if average == "step" else s.var_at_times([0.0])[0]
        return float(np.interp(0.0, s.k_grid, table))

    u = W5_WEIGHTS * np.sqrt([at_money(m.local_vol) for m in w5_3m])
    a0, b0 = fam.variance_terms(u)
    expected = (at_money(lv) - float(a0)) / float(b0)
    assert 0.0 < expected < fam.lambda_max
    # to 1e-10: the kernel's uniform-grid lookup reads the grid step off the grid, so k = 0 is
    # 1e-10 of a cell away from its node (measured difference 4e-12 relative)
    assert row[0] == pytest.approx(expected, rel=1e-10)
    assert res.lambda_mean[0] == row[0] and res.clipped_low[0] == res.clipped_high[0] == 0.0
    assert res.mean_a[0] == pytest.approx(float(a0), rel=1e-10)
    assert res.mean_b[0] == pytest.approx(float(b0), rel=1e-10)


def test_single_name_calibration(w5_3m: list[LocalVol]) -> None:
    """I4 (the calibration's half): with one name ``b ≡ 0``, so ``λ`` is not identified — the
    calibration returns ``λ ≡ 0`` and flags every slice."""
    name = w5_3m[2]
    surface, lv = w5_target(0.25)
    cfg = ParticleConfig(n_particles=4000, horizon=0.05, seed=3)
    basket = BasketSpec([1.0], "performance", [name.forward_curve])
    res = lcal.calibrate_local_correlation(
        [name], CorrelationFamily.equi(1), basket, surface, lv, cfg, FAST_SIM,
        LocalCorrelationConfig(particle=cfg),
    )  # fmt: skip
    assert np.all(res.lam.values == 0.0) and res.unidentified.all()
    assert np.all(res.mean_b == 0.0) and np.all(res.mean_a > 0) and res.max_clipped_mass == 0.0
    # and the calibrated model is the single-asset model on the asset stream (the kernel's half)
    model = LocalCorrelationModel([name], CorrelationFamily.equi(1), res.lam, basket)
    grid = res.grid
    paths = model.simulate_chunk(
        grid, model.draws_for(grid, cfg.seed, 4000), 0, 4000, FAST_SIM.scheme
    )
    alone = name.simulate_chunk(
        grid, GaussianDraws(cfg.seed, 4000, grid.n_steps, 1), 0, 4000, FAST_SIM.scheme
    )
    np.testing.assert_array_equal(paths.assets[0].log_spot, alone.log_spot)
    np.testing.assert_array_equal(res.final_log_spot[:, 0], alone.log_spot[:, 1])


def test_single_names_match_local_vol(
    w5_3m: list[LocalVol], w5_calibration: lcal.LCCalibrationResult
) -> None:
    """C2 (W5): under the calibrated ``λ`` every single-name out-of-the-money vanilla — strikes
    ``k ∈ {−0.15, −0.05, 0, 0.05, 0.15}·√(T/0.25)``, maturities ``T/3``, ``2T/3`` and ``T`` —
    is the single-asset ``LocalVol`` Monte Carlo price on an independent seed: ``|z| < 3`` in
    each of the 75 cells, with the combined standard errors."""
    horizon = 0.25
    model = LocalCorrelationModel(
        w5_3m, CorrelationFamily.equi(5), w5_calibration.lam, w5_basket(w5_3m)
    )
    maturities = [horizon / 3, 2 * horizon / 3, horizon]
    strikes = np.array([-0.15, -0.05, 0.0, 0.05, 0.15]) * np.sqrt(horizon / 0.25)
    grid = TimeGrid.build(maturities, FAST_SIM.dt_max, calibration_grid=model.required_times())
    sim = FAST_SIM
    paths = MultiAssetMonteCarlo(sim).simulate(model, grid)
    rows = []
    for i, name in enumerate(w5_3m):
        alone_draws = GaussianDraws(900_000 + i, sim.n_paths, grid.n_steps, 1)
        alone = name.simulate_chunk(grid, alone_draws, 0, sim.n_paths, sim.scheme)
        for T in maturities:
            col = grid.fixing_index[T]
            for k in strikes:
                cp = 1.0 if k >= 0 else -1.0
                got = summarize(
                    np.maximum(cp * (paths.assets[i].spot_at(col) - np.exp(k)), 0.0), True
                )
                ref = summarize(np.maximum(cp * (alone.spot_at(col) - np.exp(k)), 0.0), True)
                z = (got.mean - ref.mean) / float(np.hypot(got.stderr, ref.stderr))
                rows.append((i, T, k, got.mean, got.stderr, ref.mean, ref.stderr, z))
    table = pd.DataFrame(rows, columns=["name", "T", "k", "lc", "lc_se", "lv", "lv_se", "z"])
    print(
        f"C2 (W5): {len(table)} cells, worst |z| {table['z'].abs().max():.2f}, "
        f"rms z {np.sqrt((table['z'] ** 2).mean()):.2f}, mean z {table['z'].mean():+.2f}"
    )
    assert len(table) == 75
    assert table["z"].abs().max() < 3.0, table.to_string()


def test_index_repricing_report() -> None:
    """The report on a model whose basket law is known: one Black–Scholes name, so every strike
    reprices the flat vol and the forward is 1, within the Monte Carlo error; the pass rule
    fails a cell only when it exceeds both its tolerance and ``z`` standard errors."""
    fc = ForwardCurve.flat(1.0, 0.0, 0.0)
    name = flat_local_vol(0.2, lc_grid(0.25), fc)
    flat = SSVISurface.flat_atm(0.2, 0.0, 0.0, 0.5, fc, fc.rate_curve, max_maturity=1.0)
    lam = LocalCorrelationFunction.constant(0.0, [0.0, 0.25], name.local_vol.k_grid)
    model = LocalCorrelationModel(
        [name], CorrelationFamily.equi(1), lam, BasketSpec([1.0], "performance", [fc])
    )
    sim = SimConfig(n_paths=40_000, dt_max=DAILY, seed=5, chunk_size=20_000)
    rep = lcal.reprice_index_smile(model, flat, sim, maturities=[0.1, 0.25], pricing_seeds=(5, 6))
    assert len(rep.table) == 2 * 11 and rep.seeds == (5, 6) and rep.n_paths == 40_000
    assert np.allclose(rep.table["target_vol"], 0.2) and np.all(rep.table["stderr_vp"] > 0)
    np.testing.assert_allclose(
        rep.table["k"], rep.table["sd"] * 0.2 * np.sqrt(rep.table["T"]), rtol=1e-12, atol=1e-15
    )
    assert (rep.table["error_vp"].abs() < 4.0 * rep.table["stderr_vp"]).all(), rep.summary()
    assert (rep.forwards["forward_error"].abs() < 4.0 * rep.forwards["forward_error_se"]).all()
    assert rep.passes() and rep.violations().empty and rep.max_abs_error(1.5) < 0.15
    assert rep.pivot().shape == (2, 11) and "index implied-vol error" in rep.summary()
    assert json.loads(json.dumps(rep.to_dict()))["seeds"] == [5, 6]
    one = lcal.reprice_index_smile(
        model, flat, sim, maturities=[0.25], sd_multiples=(-1.0, 0.0, 1.0)
    )
    two = lcal.reprice_index_smile(
        model, flat, sim, maturities=[0.25], sd_multiples=(-1.0, 0.0, 1.0), pricing_seeds=(5, 6)
    )
    assert (two.table["stderr_vp"] < 0.8 * one.table["stderr_vp"]).all()  # two seeds pooled
    # the pass rule on a doctored table
    bad = dataclasses.replace(rep, table=rep.table.copy())
    cell = (bad.table["T"] == 0.25) & (bad.table["sd"] == 1.0)
    bad.table.loc[cell, "error_vp"] = 0.2  # above 0.15, and above 3 se?
    bad.table.loc[cell, "stderr_vp"] = 0.1
    assert bad.passes()  # 0.2 < 3 × 0.1: within the noise
    bad.table.loc[cell, "stderr_vp"] = 0.01
    assert not bad.passes() and len(bad.violations()) == 1
    outer = (bad.table["T"] == 0.25) & (bad.table["sd"] == 2.5)
    bad.table.loc[cell, "error_vp"] = 0.0
    bad.table.loc[outer, ["error_vp", "stderr_vp"]] = [0.25, 0.01]
    assert bad.passes() and not bad.passes(tol_outer=0.2)  # the outer cells have their own gate
    bad.table.loc[outer, "error_vp"] = np.nan
    assert not bad.passes()  # a vol that could not be inverted is a violation
    assert lcal.straddle_vol(2.0 * (2.0 * 0.5398278372770290 - 1.0), 1.0) == pytest.approx(0.2)


def test_constant_and_parametric_lambda(w5_3m: list[LocalVol]) -> None:
    """The two fitted companions on W5: the constant ``λ`` reprices the index at-the-money
    straddle to 2e-7 in vol on its own draws; the two-parameter family reprices the straddle and
    the 90 % put to 2e-6, converges, and falls with the basket as the target's skew asks."""
    surface, lv = w5_target(0.25)
    fam = CorrelationFamily.equi(5)
    times = TimeGrid.build([0.25], DAILY).times
    zero = LocalCorrelationFunction.constant(0.0, times, lv.k_grid)
    model = LocalCorrelationModel(w5_3m, fam, zero, w5_basket(w5_3m))
    sim = SimConfig(n_paths=20_000, dt_max=DAILY, seed=11, chunk_size=20_000)
    lam_c = lcal.calibrate_constant_lambda(model, surface, 0.25, sim)
    assert 0.3 < lam_c < 0.7
    sampler = lcal.BasketSampler(model, sim, [0.25])
    const = LocalCorrelationFunction.constant(lam_c, times, lv.k_grid)
    vol, se, price = sampler.straddle(sampler.levels(const)[:, 0], 0.25)
    assert abs(vol - float(surface.atm_vol(0.25))) < 2e-7 and 0 < se < 0.01 and price > 0
    # the constant correlation leaves the 90 % put too cheap: the index skew is not repriced
    put_cc = sampler.otm_vol(sampler.levels(const)[:, 0], 0.9, 0.25)[0]
    assert put_cc < float(surface.implied_vol_k(np.log(0.9), 0.25)) - 0.005
    cfg = ParametricLambdaConfig(n_paths=20_000)
    fit = lcal.calibrate_parametric_lambda(model, surface, 0.25, cfg, sim, seed=11)
    assert fit.converged and len(fit.history) <= cfg.maxit and fit.jacobian.shape == (2, 2)
    assert max(abs(v - t) for v, t in zip(fit.vols, fit.targets, strict=True)) < cfg.tol
    assert fit.targets == (
        float(surface.implied_vol_k(0.0, 0.25)),
        float(surface.implied_vol_k(np.log(0.9), 0.25)),
    )
    assert fit.c > 0 and fit.lam.slope > 0 and fit.lam.lam_hi == fam.lambda_max
    assert fit.lam == ParametricLambda.from_rho(fit.rho0, fit.c, 0.02, 0.98)
    assert fit.start[0] == pytest.approx(0.02 + 0.98 * lam_c, abs=1e-5)  # ρ_ATM: the constant fit
    assert all(s > 0 for s in fit.vol_stderrs) and fit.n_evaluations > 5
    tab = fit.function(times, lv.k_grid)
    np.testing.assert_array_equal(tab.values[0], fit.lam(lv.k_grid))
    assert json.loads(json.dumps(fit.to_dict()))["converged"] is True
    # the parametric model reprices both targets on its own draws; the sampler's λ must hold
    # the model's slices
    level = sampler.levels(tab)[:, 0]
    assert abs(sampler.strike_vol(level, 0.9, 0.25)[0] - fit.targets[1]) < cfg.tol
    with pytest.raises(ValueError, match="slice"):
        sampler.levels(LocalCorrelationFunction.constant(0.3, [0.0, 0.1234, 0.25], lv.k_grid))


def test_calibration_guard_validation_and_options(w5_3m: list[LocalVol]) -> None:
    """The three calibrating functions are refused while calibration is forbidden; the inputs
    are validated; ``clip_policy = "raise"`` fails on a target out of the family's reach;
    ``lambda_tail = "flat"`` holds the row beyond the trusted range; a second pass averages two
    seeds; injected draws must match."""
    surface, lv = w5_target(0.25)
    fam = CorrelationFamily.equi(5)
    basket = w5_basket(w5_3m)
    cfg = ParticleConfig(n_particles=6000, horizon=0.04, seed=2)
    lc = LocalCorrelationConfig(particle=cfg)
    args = (w5_3m, fam, basket, surface, lv, cfg, FAST_SIM)
    times = TimeGrid.build([cfg.horizon], DAILY).times
    model = LocalCorrelationModel(
        w5_3m, fam, LocalCorrelationFunction.constant(0.0, times, lv.k_grid), basket
    )
    with calibration_forbidden():
        with pytest.raises(CalibrationForbiddenError, match="calibrate_local_correlation"):
            lcal.calibrate_local_correlation(*args, lc)
        with pytest.raises(CalibrationForbiddenError, match="calibrate_constant_lambda"):
            lcal.calibrate_constant_lambda(model, surface, cfg.horizon, FAST_SIM)
        with pytest.raises(CalibrationForbiddenError, match="calibrate_parametric_lambda"):
            lcal.calibrate_parametric_lambda(
                model, surface, cfg.horizon, ParametricLambdaConfig(), FAST_SIM, seed=1
            )
    with pytest.raises(ValueError, match="mode"):
        lcal.calibrate_local_correlation(*args, dataclasses.replace(lc, mode="carry"))
    short = LocalVolSurface.from_implied(
        surface,
        LocalVolConfig(t_min=1 / 365, t_max=cfg.horizon, n_t=20, k_min=-2.0, k_max=2.0, n_k=1601),
    )
    with pytest.raises(ValueError, match="horizon plus"):
        lcal.calibrate_local_correlation(w5_3m, fam, basket, surface, short, cfg, FAST_SIM, lc)
    with pytest.raises(ValueError, match="sv_slope"):
        LocalCorrelationConfig(particle=dataclasses.replace(cfg, tail_extrapolation="sv_slope"))
    base = lcal.calibrate_local_correlation(*args, lc)
    # the tail rule: "flat" holds λ* at its values at the ends of the trusted range
    flat = lcal.calibrate_local_correlation(*args, dataclasses.replace(lc, lambda_tail="flat"))
    j = base.grid.n_steps
    k = lv.k_grid
    inside = (k >= base.q_lo[j]) & (k <= base.q_hi[j])
    np.testing.assert_array_equal(flat.lam.values[1][(k >= base.q_lo[1]) & (k <= base.q_hi[1])],
                                  base.lam.values[1][(k >= base.q_lo[1]) & (k <= base.q_hi[1])])  # fmt: skip
    left, right = flat.lambda_star[1][k < base.q_lo[1]], flat.lambda_star[1][k > base.q_hi[1]]
    assert np.all(left == left[0]) and np.all(right == right[0])
    assert not np.all(
        base.lambda_star[1][k < base.q_lo[1]] == base.lambda_star[1][k < base.q_lo[1]][0]
    )
    assert inside.sum() > 10 and flat.lam.metadata["lambda_tail"] == "flat"
    # a second pass: the average of the tables of seeds s and s + 1
    cfg2 = dataclasses.replace(cfg, second_pass=True)
    both = lcal.calibrate_local_correlation(w5_3m, fam, basket, surface, lv, cfg2, FAST_SIM, lc)
    other = lcal.calibrate_local_correlation(
        w5_3m, fam, basket, surface, lv, dataclasses.replace(cfg, seed=cfg.seed + 1), FAST_SIM, lc
    )
    assert both.passes == 2
    np.testing.assert_allclose(
        both.lam.values, 0.5 * (base.lam.values + other.lam.values), rtol=0, atol=1e-15
    )
    np.testing.assert_array_equal(both.final_log_spot, other.final_log_spot)
    # injected draws: the default draws give the default result; a mismatch is refused
    own = LocalCorrelationDraws(cfg.seed, cfg.n_particles, base.grid.n_steps, fam, cfg.antithetic)
    again = lcal.calibrate_local_correlation(*args, lc, draws=own)
    np.testing.assert_array_equal(again.lam.values, base.lam.values)
    wrong = LocalCorrelationDraws(cfg.seed, cfg.n_particles, base.grid.n_steps + 1, fam)
    with pytest.raises(ValueError, match="draws"):
        lcal.calibrate_local_correlation(*args, lc, draws=wrong)
    # a target out of the family's reach (a skew far steeper than the names can deliver)
    steep = SSVIConfig(W5_PILLARS, (0.21,) * len(W5_PILLARS), -0.8, 1.3, 0.5, 3.0)
    s_surface, s_lv = w5_target(0.25, steep)
    reported = lcal.calibrate_local_correlation(
        w5_3m, fam, basket, s_surface, s_lv, cfg, FAST_SIM, lc
    )
    assert reported.clipped_high.max() > 0.05 and reported.clipped_low.max() > 0.05
    assert reported.overshoot_high.max() > 0 and reported.overshoot_low.min() < 0
    worst = int(np.argmax(reported.clipped_high))
    kinds = {x[0] for x in reported.clip_intervals(worst)}
    assert "high" in kinds and reported.max_clipped_mass == max(
        reported.clipped_low.max(), reported.clipped_high.max()
    )
    with pytest.raises(lcal.ClippedMassError, match="out of the"):
        lcal.calibrate_local_correlation(
            w5_3m,
            fam,
            basket,
            s_surface,
            s_lv,
            cfg,
            FAST_SIM,
            dataclasses.replace(lc, clip_policy="raise"),
        )


# ---------------------------------------------------------------------------------------------
# LC4: the configuration, the cache and the code-tag guard
# ---------------------------------------------------------------------------------------------

SVI_PILLARS = (1 / 12, 2 / 12, 3 / 12, 6 / 12)


def svi_config(
    surface: ImpliedSurface, pillars: tuple[float, ...] = SVI_PILLARS
) -> SviSurfaceConfig:
    """An SVI-slice configuration fitted to a smooth surface at the pillars (41 strikes over
    ±3.5 at-the-money standard deviations each): how a synthetic name enters a specification."""
    params = []
    for T in pillars:
        sd = float(surface.atm_vol(T)) * np.sqrt(T)
        k = np.linspace(-3.5 * sd, 3.5 * sd, 41)
        params.append(fit_svi_slice(k, surface.implied_vol_k(k, T), T).params)
    return SviSurfaceConfig(tuple(pillars), tuple(params), max(pillars) + 0.05)


@pytest.fixture(scope="module")
def toy_spec() -> LocalCorrelationSpec:
    """Three W5 names and an index target as SVI slices: a small specification for the cache."""
    flat = MarketConfig(1.0, CurveConfig.flat(0.0), CurveConfig.flat(0.0))
    fc = ForwardCurve.flat(1.0, 0.0, 0.0)
    index = surface_from_config(
        SSVIConfig(W5_PILLARS, (0.20,) * len(W5_PILLARS), -0.7, 0.9, 0.5, 3.0), fc, fc.rate_curve
    )
    particle = ParticleConfig(n_particles=20_000, horizon=0.1, seed=3)
    return LocalCorrelationSpec(
        names=("AAA", "BBB", "CCC"),
        weights=(0.40, 0.35, 0.25),
        markets=(flat, flat, flat),
        surfaces=tuple(svi_config(s) for s in w5_surfaces(3)),
        index_surface=svi_config(index),
        lc=LocalCorrelationConfig(
            particle=particle, parametric=ParametricLambdaConfig(n_paths=20_000)
        ),
        sim=SimConfig(n_paths=20_000, dt_max=DAILY, seed=7, chunk_size=10_000),
        local_vol=LocalVolConfig(t_min=1 / 365, t_max=0.12, n_t=40, k_min=-2.0, k_max=2.0, n_k=801),
        index_forward_ratios=((1 / 12, 1.0), (0.25, 1.02)),
        label="toy 2026-10-08 3 names",
    )


def test_lc_configuration(toy_spec: LocalCorrelationSpec) -> None:
    """The configuration dataclasses: validation, the strict mapping round trip, and what the
    cache key holds — everything that determines ``λ`` and nothing else."""
    spec = toy_spec
    assert from_mapping(LocalCorrelationSpec, to_mapping(spec)) == spec  # strict YAML round trip
    default = to_mapping(LocalCorrelationConfig())
    assert "lambda_max" not in default and "lambda_grid" not in default  # omitted while None
    assert (
        default["family"] == "particle"
        and default["r_low"] == "equi"
        and default["rho_min"] == 0.02
    )
    assert default["lambda_tail"] == "regressions" and default["target_average"] == "step"
    assert "maturity" not in default["parametric"] and default["parametric"]["strikes"] == [
        1.0,
        0.9,
    ]
    with pytest.raises(ConfigError, match="unknown keys"):
        from_mapping(LocalCorrelationConfig, {"familly": "particle"})
    key = lc_cache.lc_spec_key(spec)
    assert len(key) == 64 and lc_cache.lc_spec_key(spec, "lc2") != key

    def keyed(**changes: object) -> str:
        return lc_cache.lc_spec_key(dataclasses.replace(spec, **changes))  # type: ignore[arg-type]

    # not keyed: the label, the report-only forwards, the pricing-only simulation fields, the
    # slices' provenance, the leverage-only particle fields, a market's fixing level
    assert keyed(label="another") == key and keyed(index_forward_ratios=()) == key
    assert keyed(sim=dataclasses.replace(spec.sim, n_paths=10, seed=99, chunk_size=2, antithetic=False,
                                         chunk_memory_mb=64, record_all_steps=True)) == key  # fmt: skip
    with_keys = dataclasses.replace(spec.index_surface, record_keys=("a" * 64,) * 4)
    assert keyed(index_surface=with_keys) == key
    lev = dataclasses.replace(
        spec.lc.particle, leverage_dk=0.01, l_max=5.0, l_min=0.2, leverage_std_span=4.0
    )
    assert keyed(lc=dataclasses.replace(spec.lc, particle=lev)) == key
    closed = dataclasses.replace(spec.markets[0], close=1.01)
    assert keyed(markets=(closed, *spec.markets[1:])) == key
    # keyed: everything else
    bumped = list(spec.surfaces[1].params)
    bumped[2] = (float(np.nextafter(bumped[2][0], 1.0)), *bumped[2][1:])
    one_ulp = dataclasses.replace(spec.surfaces[1], params=tuple(bumped))
    changes: list[dict[str, object]] = [
        {"weights": (0.41, 0.34, 0.25)},
        {"names": ("AAA", "BBB", "DDD")},
        {"surfaces": (spec.surfaces[0], one_ulp, spec.surfaces[2])},
        {"index_surface": dataclasses.replace(spec.index_surface, max_maturity=0.6)},
        {
            "markets": (
                MarketConfig(1.0, CurveConfig.flat(0.01), CurveConfig.flat(0.0)),
                *spec.markets[1:],
            )
        },
        {"sim": dataclasses.replace(spec.sim, dt_max=1 / 504)},
        {"sim": dataclasses.replace(spec.sim, dt_max=StepSchedule.uniform(1 / 365))},
        {"sim": dataclasses.replace(spec.sim, weak_order2=False)},
        {"local_vol": dataclasses.replace(spec.local_vol, n_k=1601)},
        {"perturbations": (SurfacePerturbation("parallel", {"size": 0.01}), None, None)},
        {"perturbations": (None, None, None)},
        {"index_perturbation": SurfacePerturbation("parallel", {"size": 0.01})},
    ]
    for name, value in (
        ("family", "parametric"), ("family", "constant"), ("rho_min", 0.0), ("rho_max", 0.9),
        ("mode", "carry"), ("lambda_tail", "flat"), ("target_average", "point"),
        ("clip_policy", "raise"), ("max_clipped_mass", 0.02), ("arbitrage", "raise"),
        ("lambda_grid", dataclasses.replace(spec.local_vol, n_k=401)),
        ("parametric", ParametricLambdaConfig(n_paths=40_000)),
        ("particle", dataclasses.replace(spec.lc.particle, seed=4)),
        ("particle", dataclasses.replace(spec.lc.particle, n_particles=40_000)),
        ("particle", dataclasses.replace(spec.lc.particle, bandwidth_factor=2.0)),
        ("particle", dataclasses.replace(spec.lc.particle, estimator="binned")),
        ("particle", dataclasses.replace(spec.lc.particle, horizon=0.09)),
    ):  # fmt: skip
        changes.append({"lc": dataclasses.replace(spec.lc, **{name: value})})
    keys = {keyed(**c) for c in changes}
    assert key not in keys and len(keys) == len(changes)
    payload = spec.key_payload()
    assert "leverage_dk" not in payload["lc"]["particle"] and "label" not in payload
    assert "n_paths" not in json.dumps(payload["schedule"]) and "seed" in payload["lc"]["particle"]
    # validation
    for bad in (
        {"weights": (0.5, 0.3, 0.3)},
        {"weights": (0.7, 0.5, -0.2)},
        {"weights": (0.5, 0.5)},
        {"names": ("AAA", "AAA", "CCC")},
        {"perturbations": (None, None)},
        {"local_vol": dataclasses.replace(spec.local_vol, t_max=0.1)},  # no room for the last step
        {"local_vol": dataclasses.replace(spec.local_vol, t_max=0.9)},  # beyond the surfaces
        {"lc": dataclasses.replace(spec.lc, r_low="matrix:/tmp/low.npy")},  # no r_low_source
        {"lc": dataclasses.replace(spec.lc, r_high="matrix:/tmp/high.npy")},  # no r_high_source
    ):
        with pytest.raises(ValueError):
            dataclasses.replace(spec, **bad)  # type: ignore[arg-type]
    for bad_lc in (
        {"family": "copula"}, {"r_low": "identity"}, {"r_high": "twos"}, {"rho_min": 0.5, "rho_max": 0.5},
        {"lambda_max": 0.9}, {"mode": "spot"}, {"clip_policy": "ignore"}, {"lambda_tail": "linear"},
        {"target_average": "mid"}, {"arbitrage": "ignore"}, {"max_clipped_mass": 1.5},
        {"r_low": "matrix:/tmp/low.npy", "lambda_max": 1.5},
    ):  # fmt: skip
        with pytest.raises(ValueError):
            LocalCorrelationConfig(**bad_lc)  # type: ignore[arg-type]
    assert (
        LocalCorrelationConfig(r_low="historical-scaled:252,0.05", lambda_max=0.9).lambda_max == 0.9
    )
    for bad_svi in (
        {"times": (0.25, 0.1)}, {"params": ((0.0, 0.1, 0.0, 0.0, 0.1),)}, {"max_maturity": 0.1},
        {"record_keys": ("x",)},
    ):  # fmt: skip
        base = {
            "times": (0.1, 0.25),
            "params": ((0.0, 0.1, 0.0, 0.0, 0.1),) * 2,
            "max_maturity": 0.3,
        }
        with pytest.raises(ValueError):
            SviSurfaceConfig(**{**base, **bad_svi})  # type: ignore[arg-type]
    for bad_par in ({"strikes": (0.9, 1.0)}, {"strikes": (1.0, 1.0)}, {"n_paths": 3}, {"tol": 0.0},
                    {"h": (0.0, 0.04)}, {"maxit": 0}, {"maturity": -1.0}):  # fmt: skip
        with pytest.raises(ValueError):
            ParametricLambdaConfig(**bad_par)  # type: ignore[arg-type]


def test_lc_cache_round_trip(toy_spec: LocalCorrelationSpec, tmp_path: Path) -> None:
    """The cache: a miss without permission names the key and writes nothing; a calibration
    stores the entry's files, its diagnostics and its manifest row; a hit reads ``λ`` back bit
    for bit and calibrates nothing; the fitted families go through the same door."""
    from volsto.calibration import guard

    spec = toy_spec
    cache = lc_cache.LocalCorrelationCache(tmp_path / "cache" / "lc")
    key = cache.key(spec)
    assert key == lc_cache.lc_spec_key(spec) and not cache.has(spec)
    with pytest.raises(CacheMissError, match=key):
        cache.get_or_calibrate(spec, allow_calibrate=False)
    assert not cache.root.exists() and cache.manifest().empty
    assert cache.svi_records().root == cache.root / "svi_fits"
    started = guard.calibrations()
    model, diag = cache.get_or_calibrate(spec, run_diagnostics=True)
    assert guard.calibrations() == started + 1 and cache.has(spec) and diag is not None
    entry = cache.entry_dir(spec)
    assert sorted(p.name for p in entry.iterdir()) == [
        "diagnostics.json", "lambda.npz", "lambda_surface.parquet", "r_low.npy", "spec.json"
    ]  # fmt: skip
    assert model.names == spec.names and model.lam.metadata["cache_key"] == key
    assert (
        model.lam.metadata["lc_code_tag"] == lcal.LC_CODE_TAG and "git_commit" in model.lam.metadata
    )
    np.testing.assert_array_equal(np.load(entry / "r_low.npy"), constant_correlation(3, 0.02))
    frame = pd.read_parquet(entry / "lambda_surface.parquet")
    assert len(frame) == model.lam.values.size and set(frame.columns) >= {
        "t",
        "k",
        "lam",
        "lam_star",
    }
    rec = diag.record
    assert rec["key"] == key and rec["lc_code_tag"] == "lc1" and rec["particle_seed"] == 3
    assert rec["pricing_seed"] == 7 and rec["n_particles"] == 20_000 and rec["n_paths"] == 20_000
    assert (
        rec["threads"] >= 1 and rec["machine"] and rec["git_commit"] and rec["label"] == spec.label
    )
    assert diag.calibration["family"] == "particle" and 0.0 <= diag.max_clipped_mass <= 1.0
    cal = diag.calibration
    assert cal["clip_gate_sd"] == lcal.CLIP_GATE_SD == 2.5
    assert 0.0 <= cal["max_clipped_mass_inner"] <= cal["max_clipped_mass"]
    assert all(a <= b for a, b in zip(cal["clipped_high_inner"], cal["clipped_high"], strict=True))
    assert all(a <= b for a, b in zip(cal["clipped_low_inner"], cal["clipped_low"], strict=True))
    visited = cal["arbitrage_visited"]
    assert set(visited["ranges"]) == {"AAA", "BBB", "CCC", "index"} and visited["central_sd"] == 3.0
    for label, r in visited["ranges"].items():
        n_slices = len(spec.index_surface.times if label == "index" else spec.surfaces[0].times)
        assert len(r["T"]) == len(r["k_lo"]) == len(r["k_hi"]) == n_slices
        for lo, c_lo, c_hi, hi in zip(
            r["k_lo"], r["k_lo_central"], r["k_hi_central"], r["k_hi"], strict=True
        ):
            assert lo <= c_lo < 0.0 < c_hi <= hi
    assert set(visited["flagged_central"]) <= set(visited["flagged"]) <= set(visited["ranges"])
    assert diag.market["arbitrage_ok"] and diag.market["flagged"] == []
    assert set(diag.market["dupire"]) == {"AAA", "BBB", "CCC", "index"}
    # the alignment of the listed index forwards: zero carry, so δ = −ln(F_I/I_0)
    align = diag.market["alignment"]
    assert [a["T"] for a in align] == [1 / 12, 0.25]
    assert align[0]["delta"] == pytest.approx(0.0, abs=1e-15) and not align[0]["flagged"]
    assert align[1]["delta"] == pytest.approx(-np.log(1.02)) and align[1]["flagged"]
    assert diag.index_report is not None and diag.max_index_error_vp is not None
    assert [round(t, 6) for t in sorted(set(diag.index_report["table"]["T"]))] == [0.083333, 0.1]
    man = cache.manifest()
    assert list(man.columns) == [
        "key", "created_utc", "git_commit", "lc_code_tag", "label", "n_names", "family", "r_low",
        "mode", "n_particles", "seed", "horizon", "max_clipped_mass", "max_index_error_vp", "wall_time",
    ]  # fmt: skip
    assert len(man) == 1 and man.iloc[0]["key"] == key and man.iloc[0]["n_names"] == 3
    # a hit: nothing is calibrated, λ is the stored one, the diagnostics are read back
    again, diag2 = cache.get_or_calibrate(spec, allow_calibrate=False)
    assert guard.calibrations() == started + 1
    np.testing.assert_array_equal(again.lam.values, model.lam.values)
    np.testing.assert_array_equal(again.lam.times, model.lam.times)
    assert diag2 is not None and diag2.to_dict() == json.loads(
        json.dumps(diag.to_dict(), default=str)
    )
    with calibration_forbidden():
        cache.get_or_calibrate(spec)  # a hit never reaches the calibration
        other = dataclasses.replace(spec, lc=dataclasses.replace(spec.lc, rho_min=0.05))
        with pytest.raises(CalibrationForbiddenError):
            cache.get_or_calibrate(other)
    # the model prices: the basket forward is 1 within the noise
    rep = lcal.reprice_index_smile(again, lc_cache.build_lc_market(spec).index_surface, spec.sim,
                                   maturities=[0.1], sd_multiples=(0.0,))  # fmt: skip
    assert abs(rep.forwards["forward_error"][0]) < 4 * rep.forwards["forward_error_se"][0]
    # the two fitted families
    const_spec = dataclasses.replace(spec, lc=dataclasses.replace(spec.lc, family="constant"))
    const_model, const_diag = cache.get_or_calibrate(const_spec)
    assert const_diag is not None and const_diag.calibration["family"] == "constant"
    assert np.all(const_model.lam.values == const_diag.calibration["lambda"])
    assert (
        const_model.lam.metadata["kind"] == "constant"
        and not (cache.entry_dir(const_spec) / "lambda_surface.parquet").exists()
    )
    np.testing.assert_array_equal(const_model.lam.times, model.lam.times)  # the same slices
    par_spec = dataclasses.replace(spec, lc=dataclasses.replace(spec.lc, family="parametric"))
    par_model, par_diag = cache.get_or_calibrate(par_spec)
    assert par_diag is not None and par_diag.calibration["family"] == "parametric"
    assert par_diag.calibration["converged"] and par_model.lam.metadata["kind"] == "parametric"
    assert par_diag.calibration["c"] > 0 and len(cache.manifest()) == 3
    assert set(cache.manifest()["family"]) == {"particle", "constant", "parametric"}


def test_build_lc_market_policies(toy_spec: LocalCorrelationSpec, tmp_path: Path) -> None:
    """The market of a specification: perturbations move one name's local vol; an SVI surface
    with an arbitrage is flagged (or raises under ``arbitrage = "raise"``); a matrix ``R_low`` is
    read from its file; a historical-scaled one must be given; carry mode puts the index target
    on the basket's forward curve."""
    spec = toy_spec
    market = lc_cache.build_lc_market(spec)
    assert market.family.low_equi and market.basket.mode == "performance" and not market.flagged
    assert market.index_lv.k_grid.size == 801 and market.summary()["arbitrage_ok"]
    assert all(0.0 <= d["floored_fraction"] < 0.01 for d in market.dupire.values())
    up = dataclasses.replace(
        spec, perturbations=(SurfacePerturbation("parallel", {"size": 0.01}), None, None)
    )
    bumped = lc_cache.build_lc_market(up)
    assert float(bumped.surfaces[0].atm_vol(0.1)) == pytest.approx(
        float(market.surfaces[0].atm_vol(0.1)) + 0.01
    )
    np.testing.assert_array_equal(
        bumped.models[1].local_vol.local_var, market.models[1].local_vol.local_var
    )
    assert (
        np.mean(bumped.models[0].local_vol.local_var > market.models[0].local_vol.local_var) > 0.9
    )
    index_up = dataclasses.replace(
        spec, index_perturbation=SurfacePerturbation("parallel", {"size": 0.01})
    )
    assert float(lc_cache.build_lc_market(index_up).index_surface.atm_vol(0.1)) == pytest.approx(
        float(market.index_surface.atm_vol(0.1)) + 0.01
    )
    # a calendar crossed in the put wing on one name
    crossed = SviSurfaceConfig(
        (0.25, 0.5), ((0.002, 0.10, -0.6, 0.0, 0.1), (0.010, 0.03, -0.3, 0.0, 0.1)), 0.6
    )
    bad = dataclasses.replace(spec, surfaces=(spec.surfaces[0], crossed, spec.surfaces[2]))
    flagged = lc_cache.build_lc_market(bad)
    assert flagged.flagged == ["BBB"] and not flagged.summary()["arbitrage_ok"]
    assert flagged.summary()["violations"]["BBB"][0].startswith("calendar")
    with pytest.raises(ArbitrageError, match="BBB"):
        lc_cache.build_lc_market(
            dataclasses.replace(bad, lc=dataclasses.replace(spec.lc, arbitrage="raise"))
        )
    # a matrix R_low from its file, keyed by the file's digest
    low = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    np.save(tmp_path / "low.npy", low)
    digest = file_sha256(tmp_path / "low.npy")
    matrix_lc = dataclasses.replace(spec.lc, r_low=f"matrix:{tmp_path / 'low.npy'}")
    matrix_spec = dataclasses.replace(spec, lc=matrix_lc, r_low_source=digest)
    m = lc_cache.build_lc_market(matrix_spec)
    assert not m.family.low_equi and m.family.lambda_max == 1.0
    np.testing.assert_array_equal(m.family.r_low, low)
    assert lc_cache.lc_spec_key(dataclasses.replace(matrix_spec, r_low_source="0" * 64)) != (
        lc_cache.lc_spec_key(matrix_spec)
    )
    # a historical-scaled R_low: the matrix is given on a miss, stored, and read back on a hit
    hist_lc = dataclasses.replace(spec.lc, r_low="historical-scaled:252,0.05")
    hist_spec = dataclasses.replace(
        spec, lc=hist_lc, r_low_source="sha256 of the panel|252|2026-10-01"
    )
    with pytest.raises(ValueError, match="must be given"):
        lc_cache.build_lc_market(hist_spec)
    rng = np.random.default_rng(3)
    returns = 0.01 * (rng.standard_normal((252, 1)) * 0.8 + rng.standard_normal((252, 3)))
    given = historical_scaled_correlation(returns, np.array(spec.weights), 0.05).r_low
    cache = lc_cache.LocalCorrelationCache(tmp_path / "cache")
    with pytest.raises(ValueError, match="must be given"):
        cache.get_or_calibrate(hist_spec)
    first, _ = cache.get_or_calibrate(hist_spec, r_low_matrix=given)
    back, _ = cache.get_or_calibrate(hist_spec, allow_calibrate=False)  # no matrix: the stored one
    np.testing.assert_array_equal(back.family.r_low, given)
    np.testing.assert_array_equal(back.lam.values, first.lam.values)
    # carry mode: the index target sits on the basket's forward curve
    div = MarketConfig(1.0, CurveConfig.flat(0.04), CurveConfig.flat(0.03))
    carry = dataclasses.replace(
        spec,
        markets=(spec.markets[0], div, spec.markets[2]),
        lc=dataclasses.replace(spec.lc, mode="carry"),
    )
    cm = lc_cache.build_lc_market(carry)
    f_basket = 0.40 + 0.35 * np.exp(0.01 * 1.0) + 0.25
    assert float(cm.index_surface.forward_curve.forward(1.0)) == pytest.approx(f_basket, rel=1e-12)
    assert float(cm.basket.log_basket_forward([1.0])[0]) == pytest.approx(
        np.log(f_basket), rel=1e-12
    )
    assert float(lc_cache.build_lc_market(spec).index_surface.forward_curve.forward(1.0)) == 1.0
    # a matrix R_high: keyed by the digest of its file, and a file that changed is refused
    high = 0.5 * np.ones((3, 3)) + 0.5 * np.eye(3)
    np.save(tmp_path / "high.npy", high)
    high_lc = dataclasses.replace(spec.lc, r_high=f"matrix:{tmp_path / 'high.npy'}")
    high_spec = dataclasses.replace(
        spec, lc=high_lc, r_high_source=file_sha256(tmp_path / "high.npy")
    )
    np.testing.assert_array_equal(lc_cache.build_lc_market(high_spec).family.r_high, high)
    assert "r_high_source" in high_spec.key_payload() and "r_high_source" not in spec.key_payload()
    assert from_mapping(LocalCorrelationSpec, to_mapping(high_spec)) == high_spec
    assert lc_cache.lc_spec_key(dataclasses.replace(high_spec, r_high_source="0" * 64)) != (
        lc_cache.lc_spec_key(high_spec)
    )
    np.save(tmp_path / "high.npy", 0.6 * np.ones((3, 3)) + 0.4 * np.eye(3))
    with pytest.raises(ValueError, match="r_high_source"):
        lc_cache.build_lc_market(high_spec)
    np.save(tmp_path / "low.npy", np.eye(3))
    with pytest.raises(ValueError, match="r_low_source"):
        lc_cache.build_lc_market(matrix_spec)


def test_lc_code_tag_guard() -> None:
    """The sources that can move a calibrated ``λ`` are hashed; a change without a bump of
    ``LC_CODE_TAG`` (and a refreshed ``lc_code_tag_guard.json``) fails here.  The leverage's own
    guard is untouched by M12 (``tests/test_lsv.py::test_calibration_code_tag_guard``)."""
    assert lc_cache.LC_GUARD_FILE.exists(), "lc_code_tag_guard.json missing: run write_lc_guard()"
    assert lcal.LC_CODE_TAG == "lc1" and lcal.LC_CODE_TAG in lc_cache.read_lc_guard()
    assert len(lc_cache.lc_source_hash()) == 64
    root = Path(__file__).resolve().parents[1]
    assert all((root / rel).is_file() for rel in lc_cache.LC_GUARDED_MODULES)
    assert {"volsto/multi/lc_kernel.py", "volsto/calibration/local_correlation.py",
            "volsto/calibration/particle.py", "volsto/market/svi_slices.py"} <= set(
        lc_cache.LC_GUARDED_MODULES
    )  # fmt: skip
    lc_cache.check_lc_guard()


def test_lc_step_schedule() -> None:
    """The model's step schedule (owner's decision 2 of 2026-10-08): quarter steps over the
    first two weeks, daily steps after; the monthly pillars are nodes; halving every segment
    gives the grid with each step cut in two (what the Δt checks coarsen back)."""
    assert LC_STEP_SCHEDULE.breaks == (10 / 252,) and LC_STEP_SCHEDULE.dts == (1 / 1008, 1 / 252)
    grid = TimeGrid.build([0.25], LC_STEP_SCHEDULE)
    assert grid.n_steps == 40 + 53
    np.testing.assert_allclose(grid.times[:41], np.arange(41) / 1008, rtol=0, atol=1e-15)
    np.testing.assert_allclose(grid.times[40:], np.arange(10, 64) / 252, rtol=0, atol=1e-15)
    for month in (1, 2, 3):
        assert np.min(np.abs(grid.times - month / 12)) < 1e-15
    half = LC_STEP_SCHEDULE.refined(2)
    assert half.breaks == LC_STEP_SCHEDULE.breaks and half.dts == (1 / 2016, 1 / 504)
    fine = TimeGrid.build([0.25], half)
    assert fine.n_steps == 2 * grid.n_steps
    np.testing.assert_allclose(fine.times[::2], grid.times, rtol=0, atol=1e-15)
    assert TimeGrid.build([2.0], LC_STEP_SCHEDULE).n_steps == 40 + 494
    assert StepSchedule.uniform(DAILY).refined(4).dts == (DAILY / 4,)
    assert LC_STEP_SCHEDULE.refined(1) == LC_STEP_SCHEDULE
    # the model's default simulation settings carry the schedule
    sim = lc_sim_config(1000, 7)
    assert sim.step_schedule == LC_STEP_SCHEDULE and (sim.n_paths, sim.seed) == (1000, 7)
    assert sim.chunk_size == 20_000 and lc_sim_config().n_paths == 800_000
    assert lc_sim_config(1000, 7, dt_max=DAILY).step_schedule == StepSchedule.uniform(DAILY)
    with pytest.raises(ValueError):
        LC_STEP_SCHEDULE.refined(0)


def test_expiry_screen() -> None:
    """The expiry screen of the specification builder (owner's decisions of 2026-10-08): the
    third-Friday rule; the quote quality of an expiry; an expiry is dropped, with the rule and
    the reason, when one of its three listed strikes nearest the forward has no positive bid on
    both sides, or when its median half spread inside ±1 sd is above 2 vol points (6 beyond one
    year); the screen needs the quotes unless it is switched off."""
    from volsto.studies.disp_lc import (
        ExpiryScreen,
        QuoteQuality,
        quote_quality,
        screen_expiries,
        third_friday,
    )

    assert third_friday("2026-10-16") and third_friday("2026-12-18") and third_friday("2019-09-20")
    for other in (
        "2026-10-30",
        "2026-11-30",
        "2026-10-09",
        "2026-10-23",
        "2026-10-21",
        "2026-10-14",
    ):
        assert not third_friday(other)
    # the Saturday of the old convention, and the Thursday before a holiday Friday — unless the
    # Friday beside them is listed too (an index with daily expiries)
    assert third_friday("2008-07-19") and not third_friday("2008-07-19", {"2008-07-18"})
    assert third_friday("2027-06-17") and not third_friday("2027-06-17", {"2027-06-18"})
    assert not third_friday("2026-10-15", {"2026-10-16"}) and not third_friday("2026-10-24")

    forward, rate, atm = 100.0, 0.04, 0.20
    strikes = np.arange(70.0, 131.0, 5.0)
    k = np.log(strikes / forward)

    def expiry(date: str, T: float = 0.25, vol: float = atm) -> Any:
        return types.SimpleNamespace(
            expiry=date, T=T, forward=forward, rate=rate, k=k, vol=np.full(k.size, vol)
        )

    def chain(
        date: str,
        half_vp: float,
        dead_puts: tuple[float, ...] = (),
        T: float = 0.25,
        vol: float = atm,
        listed: np.ndarray = strikes,
    ) -> pd.DataFrame:
        df = math.exp(-rate * T)
        call = black_price(forward, listed, T, vol, 1.0, df)
        put = black_price(forward, listed, T, vol, -1.0, df)
        half = 0.01 * half_vp * black_vega(forward, listed, T, vol, df)
        p_bid = np.where(np.isin(listed, dead_puts), 0.0, np.maximum(put - half, 0.0))
        return pd.DataFrame({"expirDate": date, "strike": listed, "cBidPx": np.maximum(call - half, 0.0),
                             "cAskPx": call + half, "pBidPx": p_bid, "pAskPx": put + half})  # fmt: skip

    good, thin, wide, weekly, gone = (
        expiry("2026-12-18"), expiry("2027-01-15"), expiry("2027-03-19"), expiry("2026-12-24"),
        expiry("2027-06-17"),
    )  # fmt: skip
    long_ok, long_wide = expiry("2028-06-16", 1.5), expiry("2028-12-15", 2.0)
    few = expiry("2027-02-19")
    # 1 sd = 1.5 %: only the strike at the forward is inside, and the puts and calls of its two
    # neighbours are worth less than their half spread — no bid: dropped by the strike rule
    narrow = expiry("2027-04-16", 0.25, 0.03)
    far = expiry("2027-05-21", 0.25, 0.20)
    rows = pd.concat([
        chain("2026-12-18", 1.0), chain("2027-01-15", 1.0, (95.0,)), chain("2027-03-19", 2.5),
        chain("2026-12-24", 0.5), chain("2028-06-16", 2.5, T=1.5), chain("2028-12-15", 6.5, T=2.0),
        chain("2027-02-19", 1.0, listed=np.array([95.0, 105.0])),
        chain("2027-04-16", 1.0, vol=0.03),
        # two-sided at the three nearest strikes, no bid on a put further out: kept
        chain("2027-05-21", 1.0, (85.0,)),
    ])  # fmt: skip
    everything = [good, thin, wide, weekly, gone, long_ok, long_wide, few, narrow, far]
    quotes = quote_quality(rows, everything)
    # 1 sd = 10 %: the strikes 95 to 110 are inside; the three nearest are 100, 105 and 95
    assert quotes["2026-12-18"] == QuoteQuality(
        "2026-12-18", 0.25, 4, 4, pytest.approx(1.0, abs=1e-12), 3, 3, False
    )
    assert (quotes["2027-01-15"].n_nearest, quotes["2027-01-15"].n_nearest_two_sided) == (3, 2)
    assert quotes["2027-03-19"].median_half_spread_vp == pytest.approx(2.5, abs=1e-12)
    assert quotes["2027-06-17"].n_nearest == 0 and math.isnan(
        quotes["2027-06-17"].median_half_spread_vp
    )
    assert (quotes["2027-02-19"].n_nearest, quotes["2027-02-19"].n_nearest_two_sided) == (2, 2)
    assert quotes["2027-04-16"].n_strikes == 1 and not quotes["2027-04-16"].spread_on_nearest
    assert quotes["2027-05-21"].n_nearest_two_sided == 3 and quotes["2027-05-21"].n_two_sided == 4
    screen = ExpiryScreen()
    assert screen.reads_quotes and screen.describe() == {
        "index_third_friday": True, "nearest_two_sided": True, "nearest_strikes": 3,
        "max_half_spread_vp": 2.0, "max_half_spread_long_vp": 6.0, "long_maturity": 1.0, "quote_sd": 1.0,
    }  # fmt: skip
    assert screen.spread_limit(1.0) == 2.0 and screen.spread_limit(1.0001) == 6.0
    kept, dropped = screen_expiries("XYZ", everything, quotes, screen)
    assert [e.expiry for e in kept] == ["2026-12-18", "2026-12-24", "2028-06-16", "2027-05-21"]
    why = {g["expiry"]: (g["rule"], g["reason"]) for g in dropped}
    assert all(
        g["leg"] == "XYZ" and set(g) == {"leg", "expiry", "T", "rule", "reason"} for g in dropped
    )
    assert why["2027-01-15"][0] == "strikes" and why["2027-01-15"][1].startswith(
        "2 of the 3 listed strikes nearest the forward have a positive bid"
    )
    assert why["2027-03-19"] == (
        "spread",
        "median half spread inside ±1 sd 2.50 vol points, above 2",
    )
    assert why["2028-12-15"] == (
        "spread",
        "median half spread inside ±1 sd 6.50 vol points, above 6",
    )
    assert why["2027-06-17"][0] == "strikes" and why["2027-02-19"][0] == "strikes"
    assert why["2027-04-16"][0] == "strikes" and quotes["2027-04-16"].n_nearest_two_sided == 1
    assert "(2 listed)" in why["2027-02-19"][1] and "(0 listed)" in why["2027-06-17"][1]
    # an expiry whose ±1 sd holds no listed strike: the spread is read on the nearest three
    off_grid = types.SimpleNamespace(
        expiry="2027-07-16", T=0.25, forward=102.5, rate=rate, k=np.log(strikes / 102.5),
        vol=np.full(k.size, 0.02),
    )  # fmt: skip
    q = quote_quality(chain("2027-07-16", 1.0), [off_grid])["2027-07-16"]
    assert q.n_strikes == 0 and q.spread_on_nearest and q.n_nearest == 3
    assert math.isfinite(q.median_half_spread_vp)
    # the index reads third Fridays only: the weekly goes first, whatever its quotes
    kept, dropped = screen_expiries("index", everything, quotes, screen, index=True)
    assert [e.expiry for e in kept] == ["2026-12-18", "2028-06-16", "2027-05-21"]
    assert {g["expiry"]: g["rule"] for g in dropped}["2026-12-24"] == "third_friday"
    assert screen_expiries("index", everything, None, ExpiryScreen.off(), index=True) == (
        everything,
        [],
    )
    only_fridays = ExpiryScreen(True, False, math.inf, math.inf)
    assert not only_fridays.reads_quotes and not ExpiryScreen.off().reads_quotes
    kept, _ = screen_expiries("index", everything, None, only_fridays, index=True)
    assert "2026-12-24" not in [e.expiry for e in kept] and len(kept) == len(everything) - 1
    with pytest.raises(ValueError, match="quote screen needs"):
        screen_expiries("XYZ", everything, None, screen)
    for bad in (
        {"max_half_spread_vp": 0.0},
        {"max_half_spread_long_vp": -1.0},
        {"long_maturity": 0.0},
    ):
        with pytest.raises(ValueError):
            ExpiryScreen(**bad)  # type: ignore[arg-type]


def test_strip_second_moment() -> None:
    """``strip_second_moment``: a flat vol gives the lognormal value ``f²·e^{σ²T} − 2f + 1`` (with
    and without carry); the regions of a split sum to the strip; a skewed surface has a larger
    second moment below the forward than its mirror image above it."""
    for r, q, vol, T in ((0.0, 0.0, 0.2, 0.25), (0.04, 0.01, 0.35, 1.0), (0.03, 0.0, 0.15, 2.0)):
        fc = ForwardCurve.flat(1.0, r, q)
        flat = surface_from_config(
            SSVIConfig((0.25, 1.0, 2.0), (vol,) * 3, 0.0, 1e-9, 0.5, 5.0), fc, fc.rate_curve
        )
        f = float(fc.forward(T))
        exact = f * f * math.exp(vol * vol * T) - 2 * f + 1
        assert strip_second_moment(flat, T) == pytest.approx(exact, rel=1e-9)
        total, parts = strip_second_moment(flat, T, splits=[-0.2, 0.0, 0.2])  # type: ignore[misc]
        assert total == pytest.approx(exact, rel=1e-9) and parts.shape == (4,) and np.all(parts > 0)
        assert parts.sum() + (f - 1.0) ** 2 == pytest.approx(total, rel=2e-5)
    skewed = w5_surfaces(1)[0]
    _, parts = strip_second_moment(skewed, 0.25, splits=[0.0])  # type: ignore[misc]
    assert parts[0] > parts[1] > 0
    for bad in ({"n_half": 3}, {"splits": [0.1, 0.0]}):
        with pytest.raises(ValueError):
            strip_second_moment(skewed, 0.25, **bad)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# LC4 acceptance (slow): helpers
# ---------------------------------------------------------------------------------------------

PRODUCTION = 800_000  # particles and pricing paths of the acceptance tests (SPEC §11)
PRICING_SEED = 2024
PARTICLE_SEED = 12345
TRUTH_SEED = 777
SD_INNER = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)
ZERO = DiscountCurve.flat(0.0)


#: The step schedule of the acceptance runs: the model's (quarter steps over the first two
#: weeks, daily after; the owner's decision 2 of 2026-10-08, SPEC §8.7).  ``scripts/
#: lcm_synthetic.py`` replaces it to report the same runs on another schedule, for information.
ACCEPTANCE_SCHEDULE: StepSchedule | float = LC_STEP_SCHEDULE


def production_sim(
    dt: StepSchedule | float | None = None, n_paths: int = PRODUCTION, seed: int = PRICING_SEED
) -> SimConfig:
    """The acceptance tests' simulation settings (``dt``: default :data:`ACCEPTANCE_SCHEDULE`)."""
    return SimConfig(
        n_paths=n_paths,
        dt_max=ACCEPTANCE_SCHEDULE if dt is None else dt,
        seed=seed,
        chunk_size=20_000,
    )


def index_smile_by_simulation(
    model: LocalCorrelationModel, sim: SimConfig, pillars: tuple[float, ...], seed: int = TRUTH_SEED
) -> tuple[SviSlices, pd.DataFrame]:
    """The index smile a model produces, as a target: at each pillar the Black vols of 41
    out-of-the-money options on the basket over ±3.5 at-the-money standard deviations (inverted
    strike by strike), then an SVI slice per pillar.  Returns the surface (on a zero-carry curve
    of spot 1: the basket's forward moneyness) and the fits' table."""
    sampler = lcal.BasketSampler(model, sim, list(pillars), seed=seed)
    level = sampler.levels()
    times, params, rows = [], [], []
    for i, T in enumerate(pillars):
        atm = sampler.straddle(level[:, i], T)[0]
        k = np.linspace(-3.5, 3.5, 41) * atm * np.sqrt(T)
        vol = np.array([sampler.otm_vol(level[:, i], float(np.exp(x)), T)[0] for x in k])
        keep = np.isfinite(vol) & (vol > 0)
        fit = fit_svi_slice(k[keep], vol[keep], T)
        times.append(T)
        params.append(fit.params)
        rows.append({"T": T, "atm_vol": atm, "svi_rms_vp": fit.rms_vp, "n_strikes": fit.n_points})
    fc = ForwardCurve(1.0, ZERO, ZERO)
    return SviSlices(times, params, fc, max(pillars) + 0.05), pd.DataFrame(rows)


def calibrate(
    models: list[LocalVol],
    family: CorrelationFamily,
    basket: BasketSpec,
    index_surface: ImpliedSurface,
    horizon: float,
    *,
    n_particles: int = PRODUCTION,
    sim: SimConfig | None = None,
    draws: LocalCorrelationDraws | None = None,
    **options: object,
) -> lcal.LCCalibrationResult:
    """A particle calibration at the acceptance settings (``options`` change the particle or the
    local correlation configuration: ``seed``, ``bandwidth_factor``, ``lambda_tail``,
    ``target_average``)."""
    particle_fields = {f.name for f in dataclasses.fields(ParticleConfig)}
    p_opts = {k: v for k, v in options.items() if k in particle_fields}
    l_opts = {k: v for k, v in options.items() if k not in particle_fields}
    p_opts.setdefault("seed", PARTICLE_SEED)
    cfg = ParticleConfig(n_particles=n_particles, horizon=horizon, **p_opts)  # type: ignore[arg-type]
    lc = LocalCorrelationConfig(particle=cfg, mode=basket.mode, **l_opts)  # type: ignore[arg-type]
    index_lv = LocalVolSurface.from_implied(index_surface, lc_grid(horizon))
    return lcal.calibrate_local_correlation(
        models,
        family,
        basket,
        index_surface,
        index_lv,
        cfg,
        sim or production_sim(),
        lc,
        draws=draws,
    )


def clip_table(res: lcal.LCCalibrationResult) -> str:
    """The clipped mass per step, summarised: the maxima, where they are, the masses inside
    ±2.5 sd (what the gate reads), and the profile at a few slices."""
    t = res.grid.times
    lo, hi = int(np.argmax(res.clipped_low)), int(np.argmax(res.clipped_high))
    inner = np.maximum(res.clipped_low_inner, res.clipped_high_inner)
    picks = sorted(
        {0, 1, 2, 5, 10, 21, 42, len(t) // 2, len(t) - 1, int(np.argmax(inner))}
        & set(range(len(t)))
    )
    lines = [
        f"clipped mass per slice: max low {res.clipped_low.max():.5f} at t={t[lo]:.4f}, max high "
        f"{res.clipped_high.max():.5f} at t={t[hi]:.4f}; mean low {res.clipped_low.mean():.5f}, "
        f"mean high {res.clipped_high.mean():.5f}; slices above 1 %: "
        f"{int(np.sum(np.maximum(res.clipped_low, res.clipped_high) > 0.01))} of {len(t)}",
        f"clipped mass inside ±{lcal.CLIP_GATE_SD:g} sd (the gate): max low "
        f"{res.clipped_low_inner.max():.5f}, max high {res.clipped_high_inner.max():.5f} at "
        f"t={t[int(np.argmax(res.clipped_high_inner))]:.4f}; slices above 1 %: "
        f"{int(np.sum(inner > 0.01))} of {len(t)}",
    ]
    lines += [
        f"  t={t[j]:.4f}: low {res.clipped_low[j]:.5f} high {res.clipped_high[j]:.5f} (inside: "
        f"{res.clipped_low_inner[j]:.5f} / {res.clipped_high_inner[j]:.5f}) mean lambda "
        f"{res.lambda_mean[j]:.4f} trusted [{res.q_lo[j]:+.3f}, {res.q_hi[j]:+.3f}]"
        for j in picks
    ]
    return "\n".join(lines)


#: the two-week pillar every synthetic target carries (owner's decision of 2026-10-08, second
#: round: a market surface has one)
TWO_WEEKS = 1 / 24


def clip_by_first_pillar(res: lcal.LCCalibrationResult, first: float) -> str:
    """The clipped mass inside ±2.5 sd, reported separately before the target's first pillar
    and from it on (the gate reads every slice)."""
    t = res.grid.times
    inner = np.maximum(res.clipped_low_inner, res.clipped_high_inner)
    before = t < first - 1e-12
    j0, j1 = int(np.argmax(np.where(before, inner, -1.0))), int(
        np.argmax(np.where(before, -1.0, inner))
    )
    return (
        f"clipped mass inside ±{lcal.CLIP_GATE_SD:g} sd: before the first pillar (t < {first:.4f}, "
        f"{int(before.sum())} slices) max {inner[j0]:.5f} at t={t[j0]:.4f}, {int((inner[before] > 0.01).sum())} "
        f"slices over 1 %; from the first pillar on ({int((~before).sum())} slices) max {inner[j1]:.5f} at "
        f"t={t[j1]:.4f}, {int((inner[~before] > 0.01).sum())} slices over 1 %"
    )


#: the particle seeds of the noise-aware gates (four independent calibrations)
NOISE_SEEDS = (PARTICLE_SEED, PARTICLE_SEED + 1, PARTICLE_SEED + 2, PARTICLE_SEED + 3)


def lambda_gate(
    results: list[lcal.LCCalibrationResult],
    truth: object,
    index_surface: ImpliedSurface,
    pillars: tuple[float, ...],
    t_min: float = 1 / 12,
    gate_pillar: float = 0.05,
    gate_between: float = 0.07,
) -> pd.DataFrame:
    """The noise-aware ``λ`` gate (owner's decision 3 of 2026-10-08, SPEC §8.7).  Per slice from
    ``t_min`` on, at 61 points inside ±1.5 at-the-money standard deviations: the error ``λ̂ −
    λ_true`` of the first calibration and ``s_λ``, the standard deviation of ``λ̂`` across the
    calibrations (independent seeds).  A cell passes when ``|error| ≤ gate + 2·s_λ``, the gate
    being ``gate_pillar`` at a pillar maturity of the target and ``gate_between`` elsewhere.
    One row per slice: the largest ``|error|`` and where, ``s_λ`` there, the largest ``s_λ``,
    and ``excess``, the largest ``|error| − gate − 2·s_λ`` (the slice passes when ``≤ 0``)."""
    rows = []
    for t in results[0].grid.times:
        if t < t_min - 1e-12:
            continue
        sd = float(index_surface.atm_vol(t)) * np.sqrt(t)
        k = np.linspace(-1.5 * sd, 1.5 * sd, 61)
        lam = np.array([r.lam(t, k) for r in results])
        err = lam[0] - truth(k)  # type: ignore[operator]
        s = lam.std(axis=0, ddof=1)
        at_pillar = bool(np.any(np.isclose(t, pillars, rtol=0.0, atol=1e-9)))
        gate = gate_pillar if at_pillar else gate_between
        i = int(np.argmax(np.abs(err)))
        rows.append({"t": t, "pillar": at_pillar, "gate": gate, "max_abs_err": float(np.abs(err[i])),
                     "at_k": float(k[i]), "s_at_max": float(s[i]), "max_s": float(s.max()),
                     "excess": float((np.abs(err) - gate - 2.0 * s).max()),
                     "err_atm": float(err[30]), "err_m1p5": float(err[0]), "err_p1p5": float(err[-1])})  # fmt: skip
    return pd.DataFrame(rows)


def print_lambda_gate(label: str, gate: pd.DataFrame) -> None:
    at, between = gate[gate["pillar"]], gate[~gate["pillar"]]
    print(f"{label} lambda against the truth at the pillars (inside ±1.5 sd; s = sd over 4 seeds):")
    print(at.drop(columns="pillar").to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    for name, part in (("at the pillars", at), ("between the pillars", between)):
        if len(part):
            w = part.loc[part["max_abs_err"].idxmax()]
            print(
                f"{label} {name} ({len(part)} slices, gate {w['gate']:.2f} + 2 s): worst |error| "
                f"{w['max_abs_err']:.4f} at t={w['t']:.4f}, k={w['at_k']:+.4f} (s there "
                f"{w['s_at_max']:.4f}); median over slices {part['max_abs_err'].median():.4f}; "
                f"largest excess over the gate {part['excess'].max():+.4f}; slices over it: "
                f"{int((part['excess'] > 0).sum())}"
            )


def dispersion_payoffs(
    model: LocalCorrelationModel,
    sim: SimConfig,
    weights: np.ndarray,
    horizon: float,
    strikes: tuple[float, ...] = (),
    *,
    draws: object = None,
) -> np.ndarray:
    """Per-path payoffs ``(n_paths, 1 + len(strikes))`` of the Palladium forward and calls."""
    mc = MultiAssetMonteCarlo(sim)
    products = [Palladium(weights, k, horizon, ZERO) for k in (0.0, *strikes)]
    grid = mc.build_grid(products, model)
    res = mc.price_many(products, model, grid=grid, draws=draws, keep_payoffs=True)
    return np.column_stack([np.asarray(r.payoffs) for r in res])


def paired(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Mean and standard error of ``a − b`` on antithetic pair means (the same paths)."""
    return lcm.pair_mean(a - b)


def smile_cells(level: np.ndarray, pillars: tuple[float, ...], atm: list[float]) -> np.ndarray:
    """Per-path out-of-the-money payoffs on the basket at the pillars and at the strikes of
    ``SD_INNER``: ``(n_paths, n_pillars, 7)``, with the strikes' ``(k, cp)``."""
    out = np.empty((level.shape[0], len(pillars), len(SD_INNER)))
    for i, T in enumerate(pillars):
        for j, m in enumerate(SD_INNER):
            k = m * atm[i] * np.sqrt(T)
            cp = 1.0 if k >= 0 else -1.0
            out[:, i, j] = np.maximum(cp * (level[:, i] - np.exp(k)), 0.0)
    return out


def smile_difference(
    level_a: np.ndarray, level_b: np.ndarray, pillars: tuple[float, ...], surface: ImpliedSurface
) -> pd.DataFrame:
    """The basket's implied vols under two models on the same paths, cell by cell inside ±1.5
    sd: the two vols, their difference in vol points and its paired standard error (the paired
    price difference through the vega)."""
    atm = [float(surface.atm_vol(T)) for T in pillars]
    pa, pb = smile_cells(level_a, pillars, atm), smile_cells(level_b, pillars, atm)
    rows = []
    for i, T in enumerate(pillars):
        for j, m in enumerate(SD_INNER):
            k = m * atm[i] * np.sqrt(T)
            cp = 1.0 if k >= 0 else -1.0
            va = float(implied_vol(pa[:, i, j].mean(), 1.0, np.exp(k), T, cp))
            vb = float(implied_vol(pb[:, i, j].mean(), 1.0, np.exp(k), T, cp))
            _, se = paired(pa[:, i, j], pb[:, i, j])
            vega = float(black_vega(1.0, np.exp(k), T, va))
            rows.append({"T": T, "sd": m, "vol_a": va, "vol_b": vb, "diff_vp": 100 * (va - vb),
                         "se_vp": 100 * se / vega})  # fmt: skip
    return pd.DataFrame(rows)


def basket_levels(
    model: LocalCorrelationModel, sim: SimConfig, pillars: tuple[float, ...], draws: object = None
) -> np.ndarray:
    """``e^{k_B}`` at the pillars on the pricing draws (or on given draws: the Δt check)."""
    grid = TimeGrid.build(list(pillars), sim.dt_max, calibration_grid=model.required_times())
    d = draws if draws is not None else model.draws_for(grid, sim.seed, sim.n_paths, sim.antithetic)
    cols = [grid.fixing_index[T] for T in pillars]
    out = np.empty((sim.n_paths, len(pillars)))
    for p0, p1 in sim.chunk_ranges(grid.n_records * model.n_assets, 0):
        paths = model.simulate_chunk(grid, d, p0, p1, sim.scheme)
        assert paths.aux is not None
        out[p0:p1] = np.exp(paths.aux["k_basket"][:, cols])
    return out


# ---------------------------------------------------------------------------------------------
# S4: the reference regression
# ---------------------------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("tag", lcm.TAGS)
def test_s4_reference_regression(tag: str) -> None:
    """S4: the parametric model at the reference's ``(ρ0, c)`` and the constant correlation at
    its ``ρ_CC`` reproduce the reference's numbers within three combined standard errors
    (``√(se_volsto² + se_ref²)``): ``E^CC[D]``, ``E^LC[D]`` and their ratio on the four dates; on
    2026-10-02 also the calls at the reference's cash strikes, the LC basket's at-the-money and
    90 % implied vols and the CC basket's 90 % vol.  (The deltas are part LC6's.)"""
    world = lcm.reference_world(tag)
    anchors = world.fx["anchors"]
    T = float(world.fx["T"])
    d_cc, b_cc = lcm.simulate_terminal(world, world.cc)
    d_lc, b_lc = lcm.simulate_terminal(world, world.lc)
    rows: list[tuple[str, float, float, float, float]] = []

    def check(name: str, value: float, se: float, anchor: list[float]) -> None:
        rows.append((name, value, se, anchor[0], anchor[1]))

    check("E_CC[D]", *lcm.pair_mean(d_cc), anchors["mom.cc.ED"])
    check("E_LC[D]", *lcm.pair_mean(d_lc), anchors["mom.lc.ED"])
    check("LC/CC", *lcm.pair_ratio(d_lc, d_cc), anchors["forward_ratio"])
    if tag == "today":
        for m, c in anchors["calls"].items():
            check(f"call {m} CC", *lcm.pair_mean(np.maximum(d_cc - c["K"], 0.0)), c["cc"])
            check(f"call {m} LC", *lcm.pair_mean(np.maximum(d_lc - c["K"], 0.0)), c["lc"])
            check(
                f"call {m} LC/CC",
                *lcm.pair_ratio(np.maximum(d_lc - c["K"], 0.0), np.maximum(d_cc - c["K"], 0.0)),
                c["ratio"],
            )
        for label, b, key in (("LC", b_lc, "fit.lc.iv_put"), ("CC", b_cc, "fit.cc.iv_put")):
            p, se = lcm.pair_mean(np.maximum(0.9 - b, 0.0))
            vol = float(implied_vol(p, 1.0, 0.9, T, -1.0))
            check(
                f"{label} basket 90% vol",
                vol,
                se / float(black_vega(1.0, 0.9, T, vol)),
                anchors[key],
            )
        p, se = lcm.pair_mean(np.abs(b_lc - 1.0))
        vol = lcal.straddle_vol(p, T)
        vega = 2.0 * np.exp(-0.125 * vol * vol * T) * np.sqrt(T / (2.0 * np.pi))
        check("LC basket ATM vol", vol, se / vega, anchors["fit.lc.iv_str"])
    table = pd.DataFrame(rows, columns=["quantity", "volsto", "se", "reference", "ref_se"])
    table["z"] = (table["volsto"] - table["reference"]) / np.hypot(table["se"], table["ref_se"])
    print(
        f"\nS4 {tag} ({world.fx['date']}, T={T:.6f}, {world.fx['steps']} steps, {lcm.N_PATHS} paths)"
    )
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    assert table["z"].abs().max() < 3.0, table.to_string()


@pytest.mark.slow
def test_s4_recalibrated_variant() -> None:
    """S4, the looser variant, on 2026-10-02: ``(ρ0, c)`` and ``ρ_CC`` fitted by the library on
    its own paths to the reference's two index targets; ``E^LC[D]`` within 0.3 % and the ratio
    within 0.002 of the reference (the reference's own independent re-simulation moved them by
    up to 0.185 % and 0.0011)."""
    world = lcm.reference_world("today")
    fx = world.fx
    T = float(fx["T"])
    target = lcm.TwoPointIndex(fx["targets"]["sigB"], fx["targets"]["v90B"])
    skeleton = world.lc.with_lambda(
        LocalCorrelationFunction.constant(0.0, world.grid.times, world.models[0].local_vol.k_grid)
    )
    lam_c = lcal.calibrate_constant_lambda(skeleton, target, T, world.sim)
    rho_cc = float(world.family.equicorrelation(lam_c))
    fit = lcal.calibrate_parametric_lambda(
        skeleton, target, T, ParametricLambdaConfig(n_paths=lcm.N_PATHS), world.sim, seed=lcm.SEED
    )
    assert fit.converged
    k_grid = world.models[0].local_vol.k_grid
    lc = skeleton.with_lambda(fit.function(world.grid.times, k_grid))
    cc = skeleton.with_lambda(LocalCorrelationFunction.constant(lam_c, world.grid.times, k_grid))
    d_cc, _ = lcm.simulate_terminal(world, cc)
    d_lc, _ = lcm.simulate_terminal(world, lc)
    ed_lc, ratio = lcm.pair_mean(d_lc), lcm.pair_ratio(d_lc, d_cc)
    ref_ed, ref_ratio = fx["anchors"]["mom.lc.ED"][0], fx["anchors"]["forward_ratio"][0]
    print(
        f"\nS4 recalibrated (2026-10-02): rho_cc {rho_cc:.8f} (reference {fx['rho_cc']:.8f}), "
        f"(rho0, c) = ({fit.rho0:.8f}, {fit.c:.8f}) (reference {fx['rho0']:.8f}, {fx['c']:.8f}), "
        f"{len(fit.history)} Newton iterations, {fit.n_evaluations} simulations, {fit.wall_time:.0f} s\n"
        f"  E_LC[D] {ed_lc[0]:.6f} ({ed_lc[1]:.6f}) against {ref_ed:.6f}: {100 * (ed_lc[0] / ref_ed - 1):+.3f} %\n"
        f"  LC/CC   {ratio[0]:.5f} ({ratio[1]:.5f}) against {ref_ratio:.5f}: {ratio[0] - ref_ratio:+.5f}"
    )
    assert abs(ed_lc[0] / ref_ed - 1.0) < 0.003
    assert abs(ratio[0] - ref_ratio) < 0.002


# ---------------------------------------------------------------------------------------------
# S1, S2, S3: round trips on known worlds
# ---------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_s1_round_trip() -> None:
    """S1 (as redesigned by the owner's decisions of 2026-10-08, SPEC §8.7): W5 at 1y under the
    known ``λ_true(t, k) = 0.5 − 0.4·tanh(k/0.25)``; its index smiles at two weeks, at every
    month to 1y and at 13 months (a target starts with a short pillar, as a market surface
    does, and does not end at the horizon) from 2·10⁶ paths, fitted with SVI slices; the
    particle calibration at 8·10⁵ particles on four seeds.  Pass: the index reprices at every
    pillar of the target up to the horizon within 0.15 vp inside ±1.5 sd and 0.30 vp inside
    ±2.5 sd (noise-aware, 8·10⁵ pricing paths); the clipped mass inside ±2.5 sd is at most 1 %
    at every step, from the start (printed separately before and from the first pillar);
    ``|λ̂ − λ_true| ≤ gate + 2·s_λ`` inside ±1.5 sd for ``t ≥ 1m``, with the gate 0.05 at the
    pillars and 0.07 between them and ``s_λ`` the standard deviation of ``λ̂`` over the four
    seeds."""
    horizon, beyond = 1.0, 13 / 12
    pillars = tuple(m / 12 for m in range(1, 13))
    models = w5_models(beyond)
    fam = CorrelationFamily.equi(5)
    basket = w5_basket(models)

    def truth(k: np.ndarray) -> np.ndarray:
        return np.asarray(0.5 - 0.4 * np.tanh(np.asarray(k) / 0.25))

    k_grid = k_grid_of(models)
    lam_true = LocalCorrelationFunction([0.0, beyond], k_grid, np.tile(truth(k_grid), (2, 1)))
    true_model = LocalCorrelationModel(models, fam, lam_true, basket)
    index_surface, fits = index_smile_by_simulation(
        true_model, production_sim(n_paths=2_000_000), (TWO_WEEKS, *pillars, beyond)
    )
    print("\nS1 target (2e6 paths):\n" + fits.to_string(index=False))
    results = [calibrate(models, fam, basket, index_surface, horizon, seed=s) for s in NOISE_SEEDS]
    res = results[0]
    model = true_model.with_lambda(res.lam)
    rep = lcal.reprice_index_smile(
        model, index_surface, production_sim(), maturities=(TWO_WEEKS, *pillars)
    )
    gate = lambda_gate(results, truth, index_surface, pillars)
    print(
        f"S1 calibration: {res.wall_time:.0f} s ({res.grid.n_steps} steps), timings {res.timings}"
    )
    print("S1 " + rep.summary())
    print("S1 " + clip_table(res))
    print("S1 " + clip_by_first_pillar(res, TWO_WEEKS))
    print_lambda_gate("S1", gate)
    assert res.max_clipped_mass_inner <= 0.01, clip_table(res)
    assert rep.passes(), rep.summary()
    assert (gate["excess"] <= 0.0).all(), gate.to_string()


@pytest.mark.slow
def test_s2_constant_correlation_fixed_point() -> None:
    """S2 (design of the owner's decisions): W5 at 3m under the constant correlation 0.5
    (``λ = 0.48/0.98``); its index smile at two weeks, 1m, 2m, 3m and 4m from 2·10⁶ paths; the
    calibration
    on four seeds must find the constant back — ``|λ̂ − 0.4898| ≤ gate + 2·s_λ`` inside ±1.5 sd
    for ``t ≥ 1m``, the gate 0.05 at the pillars and 0.07 between them — and price the Palladium
    forward within 0.5 % of the true model's (paired).  The owner confirmed on 2026-10-08 that
    the gate of S1 replaces the specification's 0.03.  Printed: the errors against that first
    gate, the index repricing and the clipped mass."""
    horizon, beyond = 0.25, 4 / 12
    pillars = (1 / 12, 2 / 12, 0.25)
    models = w5_models(beyond)
    fam = CorrelationFamily.equi(5)
    basket = w5_basket(models)
    lam_c = (0.5 - 0.02) / 0.98
    const = LocalCorrelationFunction.constant(lam_c, [0.0, beyond], k_grid_of(models))
    true_model = LocalCorrelationModel(models, fam, const, basket)
    index_surface, fits = index_smile_by_simulation(
        true_model, production_sim(n_paths=2_000_000), (TWO_WEEKS, *pillars, beyond)
    )
    print("\nS2 target (2e6 paths):\n" + fits.to_string(index=False))
    results = [calibrate(models, fam, basket, index_surface, horizon, seed=s) for s in NOISE_SEEDS]
    res = results[0]
    gate = lambda_gate(results, lambda k: np.full(np.shape(k), lam_c), index_surface, pillars)
    model = true_model.with_lambda(res.lam)
    cc = true_model.with_lambda(
        LocalCorrelationFunction.constant(lam_c, res.lam.times, k_grid_of(models))
    )
    sim = production_sim()
    d_lc = dispersion_payoffs(model, sim, W5_WEIGHTS, horizon)[:, 0]
    d_cc = dispersion_payoffs(cc, sim, W5_WEIGHTS, horizon)[:, 0]
    ratio, ratio_se = lcm.pair_ratio(d_lc, d_cc)
    rep = lcal.reprice_index_smile(model, index_surface, sim, maturities=(TWO_WEEKS, *pillars))
    print("S2 " + rep.summary())
    print("S2 " + clip_table(res))
    print("S2 " + clip_by_first_pillar(res, TWO_WEEKS))
    print_lambda_gate("S2", gate)
    print(
        f"S2 against the first gate of 0.03 (no noise allowance): {int((gate['max_abs_err'] > 0.03).sum())} "
        f"of {len(gate)} slices over it, worst {gate['max_abs_err'].max():.4f}\n"
        f"S2 E_LC[D]/E_CC[D] = {ratio:.5f} ({ratio_se:.5f}); E_CC[D] = {lcm.pair_mean(d_cc)[0]:.6f}"
    )
    assert (gate["excess"] <= 0.0).all(), gate.to_string()
    assert abs(ratio - 1.0) <= 0.005


@pytest.mark.slow
def test_s3_identical_names() -> None:
    """S3: five identical names whose common smile is the index smile (the truth is ``λ ≡ 1``;
    ``rho_max = 1`` so that the cap does not bind): on the trusted range ``λ̂ ≥ 0.99`` from the
    end of the quarter-step segment on (``t ≥ 10/252``) and ``λ̂ ≥ 0.985`` inside it (owner's
    decision of 2026-10-08, second round: the first quarter steps carry a small downward bias,
    SPEC §8.7), and the index reprices within 0.05 vp (noise-aware)."""
    horizon = 0.25
    name_surface = w5_surfaces(2)[1]
    name = LocalVol(
        LocalVolSurface.from_implied(name_surface, lc_grid(horizon)), name_surface.forward_curve
    )
    models = [name] * 5
    fam = CorrelationFamily.equi(5, 0.02, 1.0)
    basket = w5_basket(models)
    res = calibrate(models, fam, basket, name_surface, horizon, rho_max=1.0)
    k = res.lam.k_grid
    lows = np.array(
        [
            float(res.lam.values[j][(k >= res.q_lo[j]) & (k <= res.q_hi[j])].min())
            for j in range(1, res.lam.n_slices)
        ]
    )
    early = res.grid.times[1:] < 10 / 252 - 1e-12
    low_early = float(lows[early].min()) if early.any() else 1.0
    low_late = float(lows[~early].min())
    model = LocalCorrelationModel(models, fam, res.lam, basket)
    rep = lcal.reprice_index_smile(
        model, name_surface, production_sim(), maturities=(1 / 12, 2 / 12, 0.25)
    )
    print("\nS3 " + rep.summary())
    print(
        f"S3 smallest lambda on the trusted range: {low_early:.6f} inside the quarter-step "
        f"segment ({int(early.sum())} slices, {int((lows[early] < 0.99).sum())} below 0.99; floor "
        f"0.985), {low_late:.6f} from t = 10/252 on (floor 0.99); lambda at t0 "
        f"{res.lam.values[0, 0]:.6f}; clipped high mass max {res.clipped_high.max():.4f} with "
        f"overshoot {res.overshoot_high.max():.2e} (lambda* against the cap of 1)"
    )
    assert low_late >= 0.99 and low_early >= 0.985
    assert rep.passes(tol_inner=0.05, tol_outer=0.05), rep.summary()


# ---------------------------------------------------------------------------------------------
# S5, S6, S7: step, particles, bandwidth (W5 at 3m against its reachable target)
# ---------------------------------------------------------------------------------------------

W5_3M_PILLARS = (1 / 12, 2 / 12, 0.25)


def _dt_halving(
    models: list[LocalVol],
    fam: CorrelationFamily,
    basket: BasketSpec,
    surface: ImpliedSurface,
    weights: np.ndarray,
    horizon: float,
    pillars: tuple[float, ...],
    *,
    n_particles: int,
    n_paths: int,
    schedule: StepSchedule | float | None = None,
    **options: object,
) -> dict[str, object]:
    """Calibrate and price on ``schedule`` ("dt") and on the schedule with both of its segments
    halved ("dt/2"), on common random numbers (the coarse runs use the Brownian-consistent
    coarsening of the fine runs' draws, in the calibration and in pricing): the Palladium
    forward and calls, and the basket smile, at both."""
    dt = ACCEPTANCE_SCHEDULE if schedule is None else schedule
    dt = dt if isinstance(dt, StepSchedule) else StepSchedule.uniform(dt)
    half = dt.refined(2)
    fine_sim = production_sim(half, n_paths)
    coarse_sim = production_sim(dt, n_paths)
    n_fine = TimeGrid.build([horizon], half).n_steps
    cal_fine = LocalCorrelationDraws(PARTICLE_SEED, n_particles, n_fine, fam)
    res_f = calibrate(models, fam, basket, surface, horizon, n_particles=n_particles, sim=fine_sim,
                      draws=cal_fine, **options)  # fmt: skip
    res_c = calibrate(models, fam, basket, surface, horizon, n_particles=n_particles, sim=coarse_sim,
                      draws=cal_fine.coarsened(2), **options)  # fmt: skip
    m_f = LocalCorrelationModel(models, fam, res_f.lam, basket)
    m_c = LocalCorrelationModel(models, fam, res_c.lam, basket)
    grid_f = TimeGrid.build(list(pillars), half, calibration_grid=m_f.required_times())
    grid_c = TimeGrid.build(list(pillars), dt, calibration_grid=m_c.required_times())
    assert grid_f.n_steps == 2 * grid_c.n_steps
    price_fine = m_f.draws_for(grid_f, PRICING_SEED, n_paths)
    price_coarse = price_fine.coarsened(2)
    lev_f = basket_levels(m_f, fine_sim, pillars, price_fine)
    lev_c = basket_levels(m_c, coarse_sim, pillars, price_coarse)
    mc_f, mc_c = MultiAssetMonteCarlo(fine_sim), MultiAssetMonteCarlo(coarse_sim)
    fwd = Palladium(weights, 0.0, horizon, ZERO)
    base = mc_c.price(fwd, m_c, grid=grid_c, draws=price_coarse).mean
    strikes = tuple(m * base for m in (0.75, 1.0, 1.25, 1.5))
    prods = [Palladium(weights, k, horizon, ZERO) for k in (0.0, *strikes)]
    pay_f = np.column_stack(
        [
            np.asarray(r.payoffs)
            for r in mc_f.price_many(prods, m_f, grid=grid_f, draws=price_fine, keep_payoffs=True)
        ]
    )
    pay_c = np.column_stack(
        [
            np.asarray(r.payoffs)
            for r in mc_c.price_many(prods, m_c, grid=grid_c, draws=price_coarse, keep_payoffs=True)
        ]
    )
    rows = []
    for j, label in enumerate(("E[D]", "call 0.75", "call 1.00", "call 1.25", "call 1.50")):
        coarse, fine = lcm.pair_mean(pay_c[:, j]), lcm.pair_mean(pay_f[:, j])
        diff, se = paired(pay_f[:, j], pay_c[:, j])
        rows.append({"quantity": label, "dt": coarse[0], "dt/2": fine[0], "diff": diff, "diff_se": se,
                     "rel_%": 100 * diff / coarse[0], "rel_se_%": 100 * se / coarse[0]})  # fmt: skip
    smile = smile_difference(lev_f, lev_c, pillars, surface)
    return {
        "prices": pd.DataFrame(rows),
        "smile": smile,
        "coarse": res_c,
        "fine": res_f,
        "report_coarse": lcal.reprice_index_smile(m_c, surface, coarse_sim, maturities=pillars),
        "report_fine": lcal.reprice_index_smile(m_f, surface, fine_sim, maturities=pillars),
    }


def _print_dt(label: str, out: dict[str, object]) -> None:
    prices, smile = out["prices"], out["smile"]
    assert isinstance(prices, pd.DataFrame) and isinstance(smile, pd.DataFrame)
    print(f"\n{label}: prices at dt and dt/2 (common random numbers; diff = dt/2 minus dt)")
    print(prices.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print(f"{label}: basket implied vol, dt/2 minus dt, in vol points (paired se in brackets)")
    piv = smile.pivot(index="T", columns="sd", values="diff_vp")
    se = smile.pivot(index="T", columns="sd", values="se_vp")
    cells = piv.copy().astype(object)
    for i in piv.index:
        for j in piv.columns:
            cells.loc[i, j] = f"{piv.loc[i, j]:+.3f} ({se.loc[i, j]:.3f})"
    print(cells.to_string())
    for name in ("report_coarse", "report_fine"):
        rep = out[name]
        assert isinstance(rep, lcal.IndexRepricingReport)
        print(f"{label} {name}: max |error| against the target within 1.5 sd "
              f"{rep.max_abs_error(1.5):.3f} vp, within 2.5 sd {rep.max_abs_error(2.5):.3f} vp")  # fmt: skip


@pytest.mark.slow
def test_s5_dt_halving_w5() -> None:
    """S5 on W5 at 3m: calibrated and priced on the model's step schedule and on the schedule
    with both segments halved, on common random numbers — ``|E[D](dt/2) − E[D](dt)| < 0.2 %`` of
    ``E[D]`` and the index smile within 0.10 vp inside ±1.5 sd.  Also printed: the same with the
    point-value target (SPEC §8.7, review point 3)."""
    models = w5_models(0.25)
    fam = CorrelationFamily.equi(5)
    basket = w5_basket(models)
    surface, _ = w5_target(0.25)
    args = (models, fam, basket, surface, W5_WEIGHTS, 0.25, W5_3M_PILLARS)
    out = _dt_halving(*args, n_particles=PRODUCTION, n_paths=PRODUCTION)
    _print_dt("S5 W5 3m, step-averaged target", out)
    point = _dt_halving(*args, n_particles=PRODUCTION, n_paths=PRODUCTION, target_average="point")
    _print_dt("S5 W5 3m, point target (diagnostic)", point)
    prices, smile = out["prices"], out["smile"]
    assert isinstance(prices, pd.DataFrame) and isinstance(smile, pd.DataFrame)
    assert abs(prices.iloc[0]["rel_%"]) < 0.2, prices.to_string()
    assert (smile["diff_vp"].abs() <= np.maximum(0.10, 3 * smile["se_vp"])).all(), smile.to_string()


def smile_vols(
    level: np.ndarray, pillars: tuple[float, ...], surface: ImpliedSurface
) -> np.ndarray:
    """The basket's implied vols at the pillars and at the strikes of ``SD_INNER``, flattened
    ``(n_pillars·7,)``, from the out-of-the-money option means."""
    atm = [float(surface.atm_vol(T)) for T in pillars]
    pay = smile_cells(level, pillars, atm).mean(axis=0)
    out = []
    for i, T in enumerate(pillars):
        for j, m in enumerate(SD_INNER):
            k = m * atm[i] * np.sqrt(T)
            out.append(float(implied_vol(pay[i, j], 1.0, np.exp(k), T, 1.0 if k >= 0 else -1.0)))
    return np.array(out)


@pytest.mark.slow
def test_s6_particle_doubling() -> None:
    """S6 on W5 at 3m, with the gate of the owner's decision 4 (SPEC §8.7): the difference
    between the calibrations at 4·10⁵ and at 8·10⁵ particles is read against the calibration
    noise, ``|P(8·10⁵) − P(4·10⁵)| ≤ 3·√(s₄² + s₈²)``, where ``s_N`` is the standard deviation of
    the price over four independent calibration seeds at ``N`` particles, everything priced on
    the same paths — for ``E[D]``, the calls and each cell of the index smile.  Printed as
    materiality checks: the differences against 0.02 % of ``E[D]`` and 0.01 vp."""
    horizon = 0.25
    models = w5_models(horizon)
    fam = CorrelationFamily.equi(5)
    basket = w5_basket(models)
    surface, _ = w5_target(horizon)
    sim = production_sim()
    counts = (400_000, PRODUCTION)
    fitted = {
        (n, seed): LocalCorrelationModel(
            models,
            fam,
            calibrate(models, fam, basket, surface, horizon, n_particles=n, seed=seed).lam,
            basket,
        )
        for n in counts
        for seed in NOISE_SEEDS
    }
    first = (PRODUCTION, NOISE_SEEDS[0])
    base = float(dispersion_payoffs(fitted[first], sim, W5_WEIGHTS, horizon)[:, 0].mean())
    strikes = tuple(m * base for m in (0.75, 1.0, 1.25, 1.5))
    price = {
        key: dispersion_payoffs(m, sim, W5_WEIGHTS, horizon, strikes).mean(axis=0)
        for key, m in fitted.items()
    }
    vol = {
        key: smile_vols(basket_levels(m, sim, W5_3M_PILLARS), W5_3M_PILLARS, surface)
        for key, m in fitted.items()
    }

    def gate(
        values: dict[tuple[int, int], np.ndarray], labels: list[str], scale: float
    ) -> pd.DataFrame:
        rows = []
        for j, label in enumerate(labels):
            lo = np.array([values[(counts[0], seed)][j] for seed in NOISE_SEEDS])
            hi = np.array([values[(counts[1], seed)][j] for seed in NOISE_SEEDS])
            s_lo, s_hi = float(lo.std(ddof=1)), float(hi.std(ddof=1))
            diff = float(hi[0] - lo[0])
            limit = 3.0 * float(np.hypot(s_lo, s_hi))
            rows.append({"quantity": label, "8e5": hi[0] * scale, "8e5 - 4e5": diff * scale,
                         "s_4e5": s_lo * scale, "s_8e5": s_hi * scale, "gate": limit * scale,
                         "ratio": abs(diff) / limit, "mean diff": float(hi.mean() - lo.mean()) * scale})  # fmt: skip
        return pd.DataFrame(rows)

    prices = gate(price, ["E[D]", "call 0.75", "call 1.00", "call 1.25", "call 1.50"], 1.0)
    cells = [f"T={T:.4f} {m:+.1f}sd" for T in W5_3M_PILLARS for m in SD_INNER]
    smile = gate(vol, cells, 100.0)
    rel = 100.0 * abs(prices.iloc[0]["8e5 - 4e5"]) / prices.iloc[0]["8e5"]
    print(
        "\nS6 W5 3m: 8e5 against 4e5 particles (seed 12345), four calibration seeds at each count, the same pricing paths"
    )
    print(prices.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    print("S6 basket implied vol (vol points):")
    print(smile.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print(
        f"S6 materiality: |E[D](8e5) - E[D](4e5)| = {rel:.4f} % of E[D] (0.02 %: "
        f"{'within' if rel <= 0.02 else 'over'}); largest smile difference "
        f"{smile['8e5 - 4e5'].abs().max():.4f} vp (0.01 vp: "
        f"{'within' if smile['8e5 - 4e5'].abs().max() <= 0.01 else 'over'})"
    )
    assert (prices["ratio"] <= 1.0).all(), prices.to_string()
    assert (smile["ratio"] <= 1.0).all(), smile.to_string()


@pytest.mark.slow
def test_s7_bandwidth() -> None:
    """S7 on W5 at 3m: bandwidth factors 1, 1.5 and 2 — the index smiles agree pairwise within
    0.10 vp inside ±1.5 sd (the same particles, the same pricing paths)."""
    horizon = 0.25
    models = w5_models(horizon)
    fam = CorrelationFamily.equi(5)
    basket = w5_basket(models)
    surface, _ = w5_target(horizon)
    sim = production_sim()
    lev, ed = {}, {}
    for c in (1.0, 1.5, 2.0):
        res = calibrate(models, fam, basket, surface, horizon, bandwidth_factor=c)
        model = LocalCorrelationModel(models, fam, res.lam, basket)
        lev[c] = basket_levels(model, sim, W5_3M_PILLARS)
        ed[c] = dispersion_payoffs(model, sim, W5_WEIGHTS, horizon)[:, 0]
    print("\nS7 W5 3m: basket implied vol differences between bandwidth factors (vol points)")
    worst = 0.0
    for a, b in ((1.0, 1.5), (1.5, 2.0), (1.0, 2.0)):
        smile = smile_difference(lev[a], lev[b], W5_3M_PILLARS, surface)
        piv = smile.pivot(index="T", columns="sd", values="diff_vp")
        d, se = paired(ed[a], ed[b])
        print(f"factor {a} minus factor {b}: max |diff| {smile['diff_vp'].abs().max():.4f} vp; "
              f"E[D] diff {d:+.3e} ({se:.1e}), {100 * d / ed[1.5].mean():+.4f} %")  # fmt: skip
        print(piv.to_string(float_format=lambda x: f"{x:+.4f}"))
        worst = max(worst, float(smile["diff_vp"].abs().max()))
    assert worst <= 0.10


# ---------------------------------------------------------------------------------------------
# S10: carry mode
# ---------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_s10_carry_mode() -> None:
    """S10: W5 with a rate of 4 % and dividend yields of 0–4 % in carry mode (the basket is the
    price basket, whose drift the calibration ignores).  The index smile of a known sloped ``λ``
    (2·10⁶ paths, at two weeks, 1m, 2m, 3m and 4m: the target starts with a short pillar and
    does not end at the horizon) is the target.  Pass: ``E[B_T]/F_B(T) − 1`` within 3 standard
    errors and the index at-the-money error at most 0.15 vp at every pillar up to the horizon.  Printed: the per-step mean of
    ``|δ_t|`` and the carry-mode ``λ`` against the performance-mode ``λ`` on the same particles."""
    horizon, beyond = 0.25, 4 / 12
    yields = (0.00, 0.01, 0.02, 0.03, 0.04)
    curves = [ForwardCurve.flat(1.0, 0.04, q) for q in yields]
    cfg = lc_grid(beyond)
    models = [
        LocalVol(LocalVolSurface.from_implied(w5_surfaces(i + 1, curves[i])[i], cfg), curves[i])
        for i in range(5)
    ]
    fam = CorrelationFamily.equi(5)
    basket = BasketSpec(W5_WEIGHTS, "carry", curves)
    k_grid = k_grid_of(models)
    lam_true = LocalCorrelationFunction.parametric(
        ParametricLambda(0.45, 2.0, 0.0, fam.lambda_max), [0.0, beyond], k_grid
    )
    true_model = LocalCorrelationModel(models, fam, lam_true, basket)
    index_surface, fits = index_smile_by_simulation(
        true_model, production_sim(n_paths=2_000_000), (TWO_WEEKS, *W5_3M_PILLARS, beyond)
    )
    print("\nS10 target (carry mode, 2e6 paths):\n" + fits.to_string(index=False))
    res = calibrate(models, fam, basket, index_surface, horizon)
    assert res.drift_abs_mean is not None
    model = true_model.with_lambda(res.lam)
    rep = lcal.reprice_index_smile(
        model, index_surface, production_sim(), maturities=(TWO_WEEKS, *W5_3M_PILLARS)
    )
    perf = calibrate(models, fam, basket.with_mode("performance"), index_surface, horizon)
    kg = res.lam.k_grid
    inside = np.array(
        [(kg >= res.q_lo[j]) & (kg <= res.q_hi[j]) for j in range(1, res.lam.n_slices)]
    )
    gap = np.abs(res.lam.values[1:] - perf.lam.values[1:])[inside]
    print("S10 " + rep.summary())
    print("S10 " + clip_table(res))
    print("S10 " + clip_by_first_pillar(res, TWO_WEEKS))
    print(
        f"S10 mean |delta_t| over the cloud, per year: max over steps {res.drift_abs_mean.max():.3e}, "
        f"at 1m {res.drift_abs_mean[int(np.argmin(np.abs(res.grid.times - 1 / 12)))]:.3e}, at 3m "
        f"{res.drift_abs_mean[-1]:.3e}\n"
        f"S10 carry-mode against performance-mode lambda on the trusted range: max |diff| "
        f"{gap.max():.4f}, mean {gap.mean():.5f}\n"
        f"S10 forward: "
        + ", ".join(
            f"T={r['T']:.4f}: {r['forward_error']:+.2e} ({r['forward_error_se']:.1e})"
            for _, r in rep.forwards.iterrows()
        )
    )
    z = rep.forwards["forward_error"] / rep.forwards["forward_error_se"]
    assert z.abs().max() < 3.0, rep.forwards.to_string()
    atm = rep.table[rep.table["sd"] == 0.0]
    assert (atm["error_vp"].abs() <= np.maximum(0.15, 3 * atm["stderr_vp"])).all(), atm.to_string()


# ---------------------------------------------------------------------------------------------
# the Dow from the study's data (slow; skipped when the data is absent)
# ---------------------------------------------------------------------------------------------

STUDY = Path(__file__).resolve().parents[1] / "outputs" / "dispersion"
TODAY = "2026-10-02"


def _study_data_present() -> bool:
    from volsto.studies import disp_data as dd

    return (
        (STUDY / "prices.parquet").exists()
        and (STUDY / "entries" / "3m" / f"{TODAY}.pkl").exists()
        and dd.day_path(TODAY).exists()
    )


needs_study_data = pytest.mark.skipif(
    not _study_data_present(), reason="the dispersion study's data is absent"
)


def diagnostics() -> Any:
    """``scripts/lcm_diagnostics.py`` as a module: the loader of the study's inputs, the
    specification builder with the expiry screen, and the measurements the Dow tests gate."""
    import sys

    scripts = str(Path(__file__).resolve().parents[1] / "scripts")
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    import lcm_diagnostics

    return lcm_diagnostics


@pytest.mark.slow
@needs_study_data
def test_dow_specification_from_the_study(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The Dow of 2026-10-02 at 3m as a specification.  The smiles are the study's
    (``marginals_for``).  Under the default expiry screen (owner's decision 1): the index target
    holds third-Friday expiries only; every expiry dropped has its leg and its reason; every
    name's curve reproduces the forwards of its kept expiries (1e-12); the SVI slices are C8's
    selection among the kept expiries.  Without the screen the build is the one measured before
    the decision (278 slices).  The second build reads every slice from the records and fits
    nothing, and its key is the first's; the alignment of the listed DJX forwards with the
    basket's is within the 1 % flag."""
    from volsto.calibration.fit_records import FitRecords
    from volsto.market import svi_slices as sv
    from volsto.studies.disp_lc import third_friday

    L = diagnostics()
    inp = L.load_inputs(TODAY, "3m")
    study = L.de.marginals_for(TODAY, [*inp.names, "DJX"], inp.T)
    assert study is not None
    for ticker, mine in [*inp.smiles.items(), ("DJX", inp.index_smiles)]:
        theirs = study[ticker][2]
        assert [e.expiry for e in mine] == [e.expiry for e in theirs]
        for a, b in zip(mine, theirs, strict=True):
            assert a.forward == b.forward
            np.testing.assert_array_equal(a.k, b.k)
            np.testing.assert_array_equal(a.vol, b.vol)
    records = FitRecords(tmp_path / "lc" / "svi_fits")
    spec, info = L.build_spec(inp, records=records)
    assert (
        spec.n_names == 30 and spec.lc.particle.horizon == 0.25 and spec.label == f"{TODAY} B1 3m"
    )
    assert abs(sum(spec.weights) - 1.0) < 1e-12 and spec.local_vol.n_k == 1601
    assert spec.sim.step_schedule == LC_STEP_SCHEDULE
    # the screen: what was dropped, and why
    dropped = {(g["leg"], g["expiry"]): g["reason"] for g in info["dropped"]}
    assert len(dropped) == len(info["dropped"]) and all(dropped.values())
    listed = {e.expiry for e in inp.index_smiles}
    index_kept = [e for e in inp.index_smiles if ("index", e.expiry) not in dropped]
    assert all(third_friday(e.expiry, listed) for e in index_kept)
    assert (
        dropped[("index", "2026-10-30")]
        == dropped[("index", "2026-11-30")]
        == "not a third-Friday expiry"
    )
    # the index expiries beyond one year stay (the 6 vp limit beyond 1y: second-round decision)
    assert not [key for key in dropped if key[0] == "index" and key[1] > "2027-10-02"]
    assert {g["rule"] for g in info["dropped"]} <= {"third_friday", "strikes", "spread", "quotes"}
    assert [round(t, 4) for t in spec.index_surface.times] == [
        0.0384,
        0.1342,
        0.211,
        0.4603,
        0.7068,
    ]
    worst, n_forwards = 0.0, 0
    for name, market, surface, spot in zip(spec.names, spec.markets, spec.surfaces, inp.spots):
        curve = ForwardCurve.from_config(market)
        kept = [e for e in inp.smiles[name] if (name, e.expiry) not in dropped]
        ratio = np.array([e.forward / spot for e in kept])
        got = np.asarray(curve.forward(np.array([e.T for e in kept])))
        worst = max(worst, float(np.max(np.abs(got / ratio - 1.0))))
        n_forwards += len(kept)
        times = [e.T for e in kept if e.T >= 10 / 365]
        expected = [t for t in times if t <= 0.25] + [t for t in times if t > 0.25][:2]
        assert list(surface.times) == expected and surface.record_keys is not None
        assert surface.max_maturity == max(expected[-1], 0.25) + 0.05
    assert worst < 1e-12
    n_slices = sum(len(x.times) for x in spec.surfaces) + len(spec.index_surface.times)
    assert len(records.keys()) == n_slices
    assert len(spec.index_forward_ratios) == len(spec.index_surface.times)
    # without the screen: every expiry the loader returns, as before the decision
    plain, plain_info = L.build_spec(inp, screen="off")
    n_plain = sum(len(x.times) for x in plain.surfaces) + len(plain.index_surface.times)
    assert plain_info["dropped"] == [] and n_plain == 278 and len(plain.index_surface.times) == 7
    assert lc_cache.lc_spec_key(plain) != lc_cache.lc_spec_key(spec)

    def no_fit(*args: object, **kwargs: object) -> object:
        raise AssertionError("a recorded slice was fitted again")

    monkeypatch.setattr(sv, "fit_svi_slice", no_fit)
    again, _ = L.build_spec(inp, records=records)
    assert again == spec and lc_cache.lc_spec_key(again) == lc_cache.lc_spec_key(spec)
    monkeypatch.undo()
    market = lc_cache.build_lc_market(spec)
    deltas = [(round(a["T"], 4), round(100 * a["delta"], 3)) for a in market.alignment]
    by_leg: dict[str, int] = {}
    for leg, _ in dropped:
        by_leg[leg] = by_leg.get(leg, 0) + 1
    print(f"\nDow {TODAY} 3m: {len(dropped)} expiries dropped by the screen (index {by_leg.get('index', 0)}, "
          f"{len(by_leg) - ('index' in by_leg)} names touched); {n_slices} slices ({n_plain} without the screen); "
          f"SVI rms median {info['svi_rms_vp_median']:.2f} vp, max {info['svi_rms_vp_max']:.2f} vp; {n_forwards} "
          f"forwards reproduced to {worst:.1e}; alignment delta (%) by listed DJX expiry {deltas}; flagged for "
          f"arbitrage on |k| <= 1: {len([x for x in market.flagged if x != 'index'])} names, index "
          f"{'index' in market.flagged}")  # fmt: skip
    assert all(abs(a["delta"]) < 0.01 and not a["flagged"] for a in market.alignment)
    assert info["n_names_extrapolated"] == 0 and not info["index_extrapolated"]


@pytest.mark.slow
@needs_study_data
def test_single_names_match_local_vol_on_the_dow() -> None:
    """C2 on the thirty Dow names of 2026-10-02 (3m): under a calibrated ``λ`` every single-name
    out-of-the-money vanilla — forward log-moneyness ``k ∈ {−0.15, −0.05, 0, 0.05, 0.15}``,
    maturities ``T/3``, ``2T/3``, ``T`` — is the single-asset ``LocalVol`` Monte Carlo price on
    an independent seed.  The gate on the 450 cells (owner's decision of 2026-10-08): every
    ``|z| < 4`` and at most four cells with ``|z| > 3`` (1.2 expected by chance), with the
    combined standard errors.  (The index target only shapes ``λ``; the names' law does not
    depend on it.)"""
    L = diagnostics()
    spec, _ = L.build_spec(L.load_inputs(TODAY, "3m"), n_particles=200_000)
    market = lc_cache.build_lc_market(spec)
    res = L.calibrate(spec, market)
    model = LocalCorrelationModel(market.models, market.family, res.lam, market.basket, spec.names)
    horizon = 0.25
    maturities = [horizon / 3, 2 * horizon / 3, horizon]
    strikes = np.array([-0.15, -0.05, 0.0, 0.05, 0.15]) * np.sqrt(horizon / 0.25)
    sim = production_sim(n_paths=400_000)
    grid = TimeGrid.build(maturities, sim.dt_max, calibration_grid=model.required_times())
    cols = [grid.fixing_index[T] for T in maturities]
    draws = model.draws_for(grid, sim.seed, sim.n_paths)
    n = model.n_assets
    spots = np.empty((n, sim.n_paths, len(cols)))
    for p0, p1 in sim.chunk_ranges(grid.n_records * n, 0):
        paths = model.simulate_chunk(grid, draws, p0, p1, sim.scheme)
        for i in range(n):
            spots[i, p0:p1] = paths.assets[i].spot_at(cols)
    rows = []
    for i, name in enumerate(market.models):
        alone_draws = GaussianDraws(900_000 + i, sim.n_paths, grid.n_steps, 1)
        alone = np.empty((sim.n_paths, len(cols)))
        for p0, p1 in sim.chunk_ranges(grid.n_records, 0):
            alone[p0:p1] = name.simulate_chunk(grid, alone_draws, p0, p1, sim.scheme).spot_at(cols)
        for j, T in enumerate(maturities):
            fwd = float(name.forward_curve.forward(T))
            for k in strikes:
                cp = 1.0 if k >= 0 else -1.0
                got = summarize(np.maximum(cp * (spots[i, :, j] / fwd - np.exp(k)), 0.0), True)
                ref = summarize(np.maximum(cp * (alone[:, j] / fwd - np.exp(k)), 0.0), True)
                z = (got.mean - ref.mean) / float(np.hypot(got.stderr, ref.stderr))
                rows.append((spec.names[i], T, k, got.mean, got.stderr, ref.mean, ref.stderr, z))
    table = pd.DataFrame(rows, columns=["name", "T", "k", "lc", "lc_se", "lv", "lv_se", "z"])
    over = table[table["z"].abs() > 3.0]
    print(
        f"\nC2 (Dow {TODAY}, 30 names): {len(table)} cells, worst |z| {table['z'].abs().max():.2f}, "
        f"rms z {np.sqrt((table['z'] ** 2).mean()):.2f}, mean z {table['z'].mean():+.2f}, cells with "
        f"|z| >= 2: {int((table['z'].abs() >= 2).sum())} (expected {0.0455 * len(table):.0f}), "
        f"> 3: {len(over)} (expected {0.0027 * len(table):.1f})"
    )
    if len(over):
        print(over.to_string(index=False))
    assert len(table) == 450
    assert table["z"].abs().max() < 4.0 and len(over) <= 4, over.to_string()


@pytest.mark.slow
@needs_study_data
def test_s5_dt_halving_dow() -> None:
    """S5 on the Dow of 2026-10-02 at 3m (the screened target of the owner's decision 1, 8·10⁵
    particles and paths): calibrated and priced on the model's step schedule and on the schedule
    with both segments halved, on common random numbers — ``|E[D](dt/2) − E[D](dt)| < 0.2 %`` of
    ``E[D]`` and the index smile within 0.10 vp inside ±1.5 sd at 1m, 2m and 3m."""
    L = diagnostics()
    spec, _ = L.build_spec(L.load_inputs(TODAY, "3m"))
    market = lc_cache.build_lc_market(spec)
    T = spec.lc.particle.horizon
    out = L.dt_check(spec, market, [T / 3, 2 * T / 3, T])
    print("\n" + L.format_dt(f"S5 Dow {TODAY} 3m", out))
    prices, smile = out["prices"], out["smile"]
    assert abs(prices.iloc[0]["rel_%"]) < 0.2, prices.to_string()
    assert (smile["diff_vp"].abs() <= np.maximum(0.10, 3 * smile["se_vp"])).all(), smile.to_string()


#: the absolute floor of the baseline's tolerance: 0.02 % of notional (``test_m6_regression``)
BASELINE_FLOOR = 0.0002


@pytest.mark.slow
@needs_study_data
def test_s11_dow_baseline() -> None:
    """S11: the Dow of 2026-10-02 at 3m on the screened target — the full calibration at 8·10⁵
    particles; the index smile acceptance, gated when the largest clipped mass inside ±2.5 sd
    is at most 1 % and reported otherwise (the owner's reading of the rule, 2026-10-08); the basket forward within 3 standard errors; and the M12 baseline
    ``tests/golden/lcm_baseline_2026-10-02.json`` (``E[D]`` and the calls under the model and
    under its constant-correlation companion, their ratio, ``E[V]`` and ``κ``), each cell within
    ``max(2 standard errors, 0.02 % of notional)`` of the recorded value.  The baseline is
    written by ``scripts/lcm_diagnostics.py baseline --write``."""
    L = diagnostics()
    run = L.baseline_run()
    res, rep = run["result"], run["report"]
    print(f"\nS11 Dow {TODAY} 3m ({run['seconds']:.0f} s): rho_CC {run['rho_cc']:.6f}")
    print("S11 " + L.format_clip(res))
    print("S11 " + rep.summary())
    z = rep.forwards["forward_error"] / rep.forwards["forward_error_se"]
    assert z.abs().max() < 3.0, rep.forwards.to_string()
    if res.max_clipped_mass_inner <= 0.01:
        assert rep.passes(), rep.summary()
    else:
        print(
            f"S11 index smile acceptance reported, not gated: the largest clipped mass inside "
            f"±{lcal.CLIP_GATE_SD:g} sd is {res.max_clipped_mass_inner:.4f} (the whole cloud: "
            f"{res.max_clipped_mass:.4f}); {len(rep.violations())} cells over the smile gate"
        )
    assert L.BASELINE_FILE.exists(), "record it: python scripts/lcm_diagnostics.py baseline --write"
    recorded = json.loads(L.BASELINE_FILE.read_text())
    rows = []
    for name, (value, se) in recorded["cells"].items():
        got = run["cells"][name][0]
        tol = max(2.0 * se, BASELINE_FLOOR)
        rows.append({"cell": name, "baseline": value, "se": se, "now": got, "diff": got - value,
                     "tolerance": tol, "ok": abs(got - value) <= tol})  # fmt: skip
    table = pd.DataFrame(rows)
    print("S11 against the recorded baseline:")
    print(table.to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    # the settings behind the recorded numbers (the key itself moves with the last bits of the
    # SVI fits from one machine to another, so it is recorded but not compared)
    now = L.baseline_document(run)
    assert recorded["screen"] == now["screen"] and recorded["n_dropped"] == now["n_dropped"]
    for field in (
        "particle_seed",
        "pricing_seed",
        "n_particles",
        "n_paths",
        "schedule",
        "lc_code_tag",
    ):
        assert recorded["record"][field] == now["record"][field], field
    assert table["ok"].all(), table.to_string()
