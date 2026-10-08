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

The synthetic world W5: five names, spots 1, zero rates, an SSVI surface per name (ATM vols
0.20–0.40 and a skew each), weights (0.30, 0.25, 0.20, 0.15, 0.10), ``R_low`` the
equicorrelation at 0.02, ``R_high = 11ᵀ``, daily steps.

Seeds are fixed; a statistical cell that fails is reported with its table, never re-seeded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from volsto.config import LocalVolConfig, SchemeConfig, SimConfig, SSVIConfig
from volsto.engine.grid import TimeGrid
from volsto.engine.rng import GaussianDraws
from volsto.market.curves import DiscountCurve, ForwardCurve
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import surface_from_config
from volsto.models.bs import BlackScholes
from volsto.models.localvol import LocalVol
from volsto.multi import MultiAssetModel, MultiAssetMonteCarlo, MultiPathSet, Palladium
from volsto.multi.analytics import implied_correlation, pairwise_mean_correlation
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


def lc_grid(horizon: float) -> LocalVolConfig:
    """The shared Dupire grid of the specification for a horizon up to 1y."""
    return LocalVolConfig(
        t_min=1 / 365, t_max=horizon + 0.02, n_t=120, k_min=-2.0, k_max=2.0, n_k=1601
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
