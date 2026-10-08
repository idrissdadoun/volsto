"""M12 (SPEC §8.7): the calibrated local correlation model.

Part LC2 — the correlation family ``ρ(λ) = (1 − λ)·R_low + λ·R_high`` and the draws: the
closed-form equicorrelation factor and the fixed-order mixing (identities I6, bit for bit),
the streams and their antithetics, the statistics of the mixed normals (C1, draws part), the
family's validation and algebra, the historical-scaled ``R_low``.

Seeds are fixed; a statistical cell that fails is reported with its table, never re-seeded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from volsto.engine.rng import GaussianDraws
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
