"""M2 tests (SPEC §10, Bergomi): martingale, exact step vs Euler, χ vs sample variance, diagonal
covariances, θ = 0 reproduces a reference 1F implementation path by path, vol of VS vol vs eq.
7.39, mixing-solution ATMF skew vs eq. 8.55 (Table 8.2, flat 20% VS), mixing smile vs spot MC."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from volsto.analytics import (
    alpha_theta,
    atmf_skew_order1,
    atmf_skew_order1_flat,
    chi,
    cov_x_diag,
    cov_xi_diag,
    eta_u,
    ssr_order1,
    ssr_order1_flat,
    var_integrated_variance,
    vs_vol_of_vol,
    vs_vol_of_vol_flat,
)
from volsto.analytics.mixing import mixing_atmf_skew, mixing_smile
from volsto.config import BergomiParams, SimConfig, load_yaml
from volsto.engine import GaussianDraws, MonteCarlo, TimeGrid, batch_means
from volsto.market import ForwardCurve, ForwardVarianceCurve, black_vega, implied_vol
from volsto.models import BergomiSV
from volsto.models.bergomi import factor_step_covariance, sqrt_covariance
from volsto.products import EuropeanOption, VarianceSwap

TABLE_8_2 = Path(__file__).resolve().parents[1] / "configs" / "models" / "bergomi_table_8_2.yaml"


@pytest.fixture(scope="module")
def p82() -> BergomiParams:
    return load_yaml(TABLE_8_2, BergomiParams)


@pytest.fixture(scope="module")
def fc0() -> ForwardCurve:
    return ForwardCurve.flat(100.0, 0.0, 0.0)


@pytest.fixture(scope="module")
def xi_flat() -> ForwardVarianceCurve:
    return ForwardVarianceCurve.flat(0.04)


@pytest.fixture(scope="module")
def model_2f(p82: BergomiParams, xi_flat: ForwardVarianceCurve, fc0: ForwardCurve) -> BergomiSV:
    return BergomiSV(p82, xi_flat, fc0)


@pytest.fixture(scope="module")
def model_1f(xi_flat: ForwardVarianceCurve, fc0: ForwardCurve) -> BergomiSV:
    return BergomiSV(BergomiParams.one_factor(3.0, 1.5, -0.7), xi_flat, fc0)


# ---------------------------------------------------------------------------------------------
# parameters and closed forms
# ---------------------------------------------------------------------------------------------


def test_params_validation_and_constructors(p82: BergomiParams) -> None:
    assert p82.omega == pytest.approx(3.48)
    with pytest.raises(ValueError, match="PSD"):
        BergomiParams(1.0, 0.3, 5.0, 0.3, 0.9, 0.9, -0.9)
    one = BergomiParams.one_factor(3.0, 1.5, -0.7)
    assert one.is_one_factor and one.nu == 1.5 and one.k1 == 1.5 and one.rho_SX1 == -0.7
    q = BergomiParams.from_chi(1.74, 0.245, 5.35, 0.28, 0.0, -0.759, 0.3)
    assert q.rho_SX2 == pytest.approx(0.3 * np.sqrt(1 - 0.759**2))  # eq. 8.56 with rho12 = 0
    with pytest.raises(ValueError):
        BergomiParams(1.0, 1.5, 5.0, 0.3, 0.0, 0.0, 0.0)


def test_closed_form_identities(p82: BergomiParams, xi_flat: ForwardVarianceCurve) -> None:
    assert float(eta_u(p82, 0.0)) == pytest.approx(1.0)  # eq. 7.31b: eta(0) = 1
    # chi(t, t) = int_0^t eta(u)^2 du (eq. 7.35)
    u = np.linspace(0, 1.3, 20001)
    assert float(chi(p82, 1.3, 1.3)) == pytest.approx(
        float(np.trapezoid(eta_u(p82, u) ** 2, u)), rel=1e-6
    )
    # chi(t, T) = int_{T-t}^{T} eta^2
    u = np.linspace(0.5, 2.0, 20001)
    assert float(chi(p82, 1.5, 2.0)) == pytest.approx(
        float(np.trapezoid(eta_u(p82, u) ** 2, u)), rel=1e-6
    )
    # Cov(x_u^u, x_u^u) = chi(u, u); symmetric in (u, v)
    assert float(cov_x_diag(p82, 0.7, 0.7)) == pytest.approx(float(chi(p82, 0.7, 0.7)))
    assert float(cov_x_diag(p82, 0.3, 1.1)) == pytest.approx(float(cov_x_diag(p82, 1.1, 0.3)))
    # general term-structure forms reduce to the flat ones
    for T in (0.25, 1.0, 3.0):
        assert atmf_skew_order1(p82, xi_flat, T) == pytest.approx(
            float(atmf_skew_order1_flat(p82, T)), rel=1e-6
        )
        assert ssr_order1(p82, xi_flat, T) == pytest.approx(
            float(ssr_order1_flat(p82, T)), rel=1e-6
        )
        assert vs_vol_of_vol(p82, T, xi_flat) == pytest.approx(
            float(vs_vol_of_vol_flat(p82, T)), rel=1e-8
        )
    # SSR limits (book §9.3, §9.5): R_0 = 2, R_inf = 1; ~1.5 shoulder for 1-5y with Table 8.2
    assert float(ssr_order1_flat(p82, 1e-6)) == pytest.approx(2.0, abs=1e-4)
    assert float(ssr_order1_flat(p82, 200.0)) == pytest.approx(1.0, abs=0.02)
    assert 1.4 < float(ssr_order1_flat(p82, 2.0)) < 1.6
    # nu is a global scale: vol of a very short VS vol is nu (book after eq. 7.22)
    assert float(vs_vol_of_vol_flat(p82, 1e-8)) == pytest.approx(p82.nu, rel=1e-6)
    # alpha_theta = 1 for theta = 0
    assert alpha_theta(BergomiParams.one_factor(2.0, 1.0, -0.5)) == 1.0


# ---------------------------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------------------------


def _simulate(model: BergomiSV, fixings: list[float], n_paths: int, seed: int, **opts: object):
    cfg = SimConfig(n_paths=n_paths, dt_max=1 / 365, chunk_size=min(n_paths, 50_000), seed=seed, **opts)  # type: ignore[arg-type]
    grid = TimeGrid.build(fixings, cfg.dt_max)
    return MonteCarlo(cfg).simulate(model, grid), grid


@pytest.mark.parametrize("which", ["2f", "1f"])
def test_forward_variances_are_martingales(
    which: str, model_2f: BergomiSV, model_1f: BergomiSV
) -> None:
    """E[xi_t^T] = xi_0^T (eqs. 7.33-7.35) and E[V_t] = xi_0^t."""
    model = model_2f if which == "2f" else model_1f
    paths, _ = _simulate(model, [0.5, 1.0], 100_000, seed=4)
    for col, t in ((1, 0.5), (2, 1.0)):
        fac = paths.factors_at(col)
        for T in (t, 1.5, 3.0):
            xi = model.forward_variance(t, fac, T)
            se = xi.std(ddof=1) / np.sqrt(xi.size)
            assert abs(xi.mean() - float(model.xi0.xi0(T))) < 3.5 * se, (t, T, xi.mean(), se)
        v = paths.variance_at(col)
        np.testing.assert_allclose(v, model.variance_from_factors(t, fac), rtol=1e-12)


def test_exact_step_covariance_matches_formula_and_fine_euler(
    p82: BergomiParams, xi_flat: ForwardVarianceCurve, fc0: ForwardCurve
) -> None:
    """One exact step (eqs. 7.15-7.18): sample covariance of (dW^S, dX1, dX2) vs the formula,
    and the formula vs a fine Euler discretisation of the OU factors."""
    dt = 0.1
    C = factor_step_covariance(p82, dt, 2)[0]
    # (a) kernel one-step output: dX from the factors, dW^S from the log-spot increment
    model = BergomiSV(p82, xi_flat, fc0)
    n = 200_000
    cfg = SimConfig(n_paths=n, dt_max=dt, chunk_size=50_000, seed=8, local_var_time_average=False)
    grid = TimeGrid.build([dt], dt)
    paths = MonteCarlo(cfg).simulate(model, grid)
    v0 = float(xi_flat.xi0(0.0))
    dws = (paths.log_return(0, 1) + 0.5 * v0 * dt) / np.sqrt(v0)
    sample = np.cov(np.column_stack([dws, paths.factors_at(1)]).T)
    np.testing.assert_allclose(sample, C, rtol=0.03, atol=1e-3)
    # (b) fine Euler for the OU factors and the Brownian motion
    rng = np.random.default_rng(1)
    m, nsub = 100_000, 400
    h = dt / nsub
    L = np.linalg.cholesky(p82.correlation_matrix)
    ws = np.zeros(m)
    x = np.zeros((m, 2))
    ks = np.array([p82.k1, p82.k2])
    for _ in range(nsub):
        dw = rng.standard_normal((m, 3)) @ L.T * np.sqrt(h)
        ws += dw[:, 0]
        x = x - ks * x * h + dw[:, 1:]
    euler = np.cov(np.column_stack([ws, x]).T)
    np.testing.assert_allclose(euler, C, rtol=0.04, atol=1.5e-3)
    # cholesky nesting: the (S, X1) block factor is the leading 2x2 block
    A = sqrt_covariance(C[None])[0]
    np.testing.assert_allclose(A[:2, :2], np.linalg.cholesky(C[:2, :2]))


def test_chi_matches_sample_variance(model_2f: BergomiSV, p82: BergomiParams) -> None:
    """Var[x_t^T] = chi(t, T) (eq. 7.35) and Cov(X^i_t, X^j_t) analytic."""
    paths, _ = _simulate(model_2f, [0.5, 2.0], 200_000, seed=5)
    for col, t in ((1, 0.5), (2, 2.0)):
        fac = paths.factors_at(col)
        np.testing.assert_allclose(
            np.cov(fac.T), model_2f.factor_covariance(t), rtol=0.03, atol=3e-3
        )
        for T in (t, t + 1.0, 5.0):
            x = model_2f.x_factor(t, fac, T)[:, 0]
            assert x.var(ddof=1) == pytest.approx(float(chi(p82, t, T)), rel=0.03)
            assert abs(x.mean()) < 4 * x.std() / np.sqrt(x.size)


@pytest.mark.parametrize("nu", [0.5, 1.74])
def test_diagonal_covariances(
    p82: BergomiParams, xi_flat: ForwardVarianceCurve, fc0: ForwardCurve, nu: float
) -> None:
    """Cov(xi_u^u, xi_v^v) and Var(int xi) closed forms (SPEC §3.3, derived) vs MC, with
    batch-means standard errors and a 2 SE criterion; run at nu = 50% (tame lognormal moments)
    and at the Table 8.2 value (owner request after M2)."""
    p = p82.replace(nu=nu)
    model = BergomiSV(p, xi_flat, fc0)
    paths, _ = _simulate(model, [0.5, 1.0, 2.0], 400_000, seed=6)
    vu, vv = paths.variance_at(1), paths.variance_at(2)
    cov = batch_means(lambda a, b: float(np.cov(a, b)[0, 1]), vu, vv)
    assert abs(cov.z(float(cov_xi_diag(p, xi_flat, 0.5, 1.0)))) < 2.0, (
        nu,
        cov,
        cov_xi_diag(p, xi_flat, 0.5, 1.0),
    )
    var = batch_means(lambda a: float(np.var(a, ddof=1)), vu)
    assert abs(var.z(float(cov_xi_diag(p, xi_flat, 0.5, 0.5)))) < 2.0, (nu, var)
    # variance of the integrated variance over [0, 1] and [1, 2]: the int_var accumulator is a
    # left-point Riemann sum, so compare with its exact discrete moments (2 SE) and check that the
    # continuous closed form differs only by the O(dt) quadrature term
    paths_grid = TimeGrid.build([0.5, 1.0, 2.0], 1 / 365)
    for c0, c1, T1, T2 in ((0, 2, 0.0, 1.0), (2, 3, 1.0, 2.0)):
        iv = paths.integrated_variance(c0, c1)
        est = batch_means(lambda a: float(np.var(a, ddof=1)), iv)
        mean_disc, var_disc = model.integrated_variance_moments(
            paths_grid.times, SimConfig().scheme, T1, T2
        )
        closed = var_integrated_variance(p, xi_flat, T1, T2)
        assert abs(est.z(var_disc)) < 2.0, (nu, T1, T2, est, var_disc, closed)
        assert abs(var_disc / closed - 1.0) < 0.006, (nu, T1, T2, var_disc, closed)
        # the mean of a fat-tailed lognormal sum: batch-means SE is itself noisy, allow 2.5 SE
        mean = batch_means(lambda a: float(np.mean(a)), iv)
        assert abs(mean.z(mean_disc)) < 2.5, (nu, T1, T2, mean, mean_disc)
        assert abs(mean_disc / (0.04 * (T2 - T1)) - 1.0) < 0.003


def test_theta_zero_reproduces_reference_1f_path_by_path(xi_flat: ForwardVarianceCurve) -> None:
    """theta = 0 is the 1F model of the earlier studies: compare with an independent numpy
    implementation (exact OU step, log-Euler spot) driven by the same normals."""
    omega, kappa, rho = 3.0, 1.5, -0.7
    fc = ForwardCurve.flat(100.0, 0.02, 0.01)
    model = BergomiSV(BergomiParams.one_factor(omega, kappa, rho), xi_flat, fc)
    assert model.n_factors == 1 and model.n_brownians == 2
    fixings = [0.25, 0.5, 1.0]
    cfg = SimConfig(
        n_paths=2_000, dt_max=1 / 52, chunk_size=2_000, seed=17, local_var_time_average=False
    )
    grid = TimeGrid.build(fixings, cfg.dt_max)
    draws = GaussianDraws(cfg.seed, cfg.n_paths, grid.n_steps, 2, cfg.antithetic)
    paths = model.simulate_chunk(grid, draws, 0, cfg.n_paths, cfg.scheme)
    # reference: 1F Bergomi, xi_t^t = xi0 exp(omega X_t - omega^2/2 (1 - e^{-2 kappa t})/(2 kappa))
    z = draws.block(0, grid.n_steps, 0, cfg.n_paths)  # (n, steps, 2)
    ls = np.full(cfg.n_paths, np.log(100.0))
    x = np.zeros(cfg.n_paths)
    xi0 = 0.04
    lnF = np.asarray(fc.log_forward(grid.times))
    ref_ls, ref_x = {}, {}
    for j in range(grid.n_steps):
        t, dt = grid.times[j], grid.dts[j]
        var = xi0 * np.exp(omega * x - 0.5 * omega**2 * (1 - np.exp(-2 * kappa * t)) / (2 * kappa))
        c11 = np.sqrt(dt)
        c21 = rho * (1 - np.exp(-kappa * dt)) / kappa / c11
        c22 = np.sqrt((1 - np.exp(-2 * kappa * dt)) / (2 * kappa) - c21**2)
        dws = c11 * z[:, j, 0]
        dx = c21 * z[:, j, 0] + c22 * z[:, j, 1]
        ls = ls + (lnF[j + 1] - lnF[j]) - 0.5 * var * dt + np.sqrt(var) * dws
        x = np.exp(-kappa * dt) * x + dx
        col = grid.step_record[j]
        if col >= 0:
            ref_ls[col], ref_x[col] = ls.copy(), x.copy()
    for col in range(1, grid.n_records):
        np.testing.assert_allclose(paths.log_spot_at(col), ref_ls[col], rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(paths.factors_at(col)[:, 0], ref_x[col], rtol=1e-11, atol=1e-11)


def test_vs_vol_of_vol(model_2f: BergomiSV, p82: BergomiParams) -> None:
    """Instantaneous vol of VS vols from the simulated factors at t = 1/365 vs eq. 7.39."""
    t = 1 / 365
    paths, _ = _simulate(model_2f, [t], 200_000, seed=7)
    fac = paths.factors_at(1)
    for T in (0.25, 1.0, 2.0):
        sig = np.sqrt(model_2f.vs_variance(t, fac, t, T))
        est = np.log(sig).std(ddof=1) / np.sqrt(t)
        assert est == pytest.approx(float(vs_vol_of_vol_flat(p82, T)), rel=0.03), (T, est)
    # spot-start VS variance equals xi0 on average (martingale) and V_0 = xi_0^0
    assert model_2f.vs_variance(t, fac, t, 1.0).mean() == pytest.approx(0.04, rel=0.01)


def test_variance_swap_reprices_xi0(model_2f: BergomiSV, fc0: ForwardCurve) -> None:
    cfg = SimConfig(n_paths=100_000, chunk_size=50_000, seed=2)
    swap = VarianceSwap.daily(1.0, 0.0, fc0.rate_curve, per_year=365)
    res = MonteCarlo(cfg).price(swap, model_2f)
    assert abs(res.mean - 0.04) < 3 * res.stderr + 1e-4, res


# ---------------------------------------------------------------------------------------------
# mixing solution
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize("T", [0.25, 1.0])
def test_mixing_skew_matches_eq_8_55(model_2f: BergomiSV, p82: BergomiParams, T: float) -> None:
    """Table 8.2, flat 20% VS: exact (mixing) ATMF skew vs order-one eq. 8.55 within ~10%
    (book Fig. 8.3: the order-one formula is accurate to a few percent)."""
    skew, _, err = mixing_atmf_skew(model_2f, T, h=0.02, n_paths=400_000, seed=3)
    ref = float(atmf_skew_order1_flat(p82, T))
    assert abs(skew / ref - 1.0) < 0.10 + 3 * err / abs(ref), (T, skew, ref, err)
    assert skew < 0


def test_mixing_five_year_atmf_vol_matches_book(model_2f: BergomiSV) -> None:
    """Book §8.8: 5y ATMF vol 16.0% (MC) for Table 8.2 and a flat 20% VS term structure."""
    _, atm5, _ = mixing_atmf_skew(model_2f, 5.0, h=0.01, n_paths=100_000, seed=1)
    assert atm5 == pytest.approx(0.160, abs=0.003)


@pytest.mark.parametrize("T", [0.25, 1.0])
def test_mixing_smile_matches_spot_simulation(
    model_2f: BergomiSV, fc0: ForwardCurve, T: float
) -> None:
    """Mixing-solution smile vs spot-simulation MC smile within 2 combined stderr (+0.03 vp
    discretisation allowance for the frozen-variance spot step)."""
    F = float(fc0.forward(T))
    ks = np.array([-0.1, -0.05, 0.0, 0.05, 0.1])
    K = F * np.exp(ks)
    mix = mixing_smile(model_2f, T, K, n_paths=200_000, seed=11)
    cps = np.where(K >= F, 1, -1)
    prods = [EuropeanOption(k, T, int(c), fc0.rate_curve) for k, c in zip(K, cps)]
    cfg = SimConfig(n_paths=200_000, dt_max=1 / 365, chunk_size=50_000, seed=12)
    res = MonteCarlo(cfg).price_many(prods, model_2f)
    pr = np.array([r.mean for r in res])
    er = np.array([r.stderr for r in res])
    iv = implied_vol(pr, F, K, T, cps, 1.0)
    iv_se = er / black_vega(F, K, T, iv, 1.0)
    diff = iv - mix.implied_vols
    tol = 2 * np.hypot(iv_se, mix.implied_vol_stderr) + 0.0003
    assert np.all(np.abs(diff) < tol), (T, diff * 100, tol * 100)
