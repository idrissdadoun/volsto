"""Mixing solution for vanilla smiles of the pure SV model (Bergomi ch. 8, Appendix A.1).

Conditional on the factor paths, ``ln S_T`` is Gaussian: splitting ``W^S = λ W^∥ + √(1−λ²) W^⊥``
with ``W^∥`` the part spanned by the factor Brownians (eqs. 8.59–8.62),

    S_T = S*_0 exp(−½ σ*² T + σ* √T Z),
    ln S*_0 = ln F_T − ½ λ² ∫₀ᵀ ξ_t^t dt + λ ∫₀ᵀ √ξ_t^t dW^∥_t,      (8.61a, forward for rates)
    σ*² = (1 − λ²) (1/T) ∫₀ᵀ ξ_t^t dt,                              (8.61b)
    λ W^∥ = c₁ W¹ + c₂ W²,  c₁ = (ρ_SX1 − ρ12 ρ_SX2)/(1−ρ12²),  c₂ = (ρ_SX2 − ρ12 ρ_SX1)/(1−ρ12²),
    λ² = (ρ_SX1² + ρ_SX2² − 2 ρ12 ρ_SX1 ρ_SX2)/(1−ρ12²),             (8.62)

and the price is ``E[Black(S*_0, K, T, σ*)]`` (8.58).  Only the factors are simulated (exactly, with
the Brownian increments drawn jointly, eqs. 7.15–7.18); the two integrals are accrued with the
left-point rule on a fine grid.  This is the low-noise way to get the naked SV smile and its
ATMF skew (SPEC §3.3, §4.4).  Checked by ``tests/test_bergomi.py`` (skew vs eq. 8.55, smile vs
spot-simulation MC).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.engine.mc import PriceResult, summarize
from volsto.engine.rng import GaussianDraws
from volsto.market.bs import black_price, black_vega, implied_vol
from volsto.models.bergomi import BergomiSV, factor_step_covariance, sqrt_covariance

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MixingSmile:
    """Vanilla prices and implied vols from the mixing solution, with standard errors."""

    maturity: float
    forward: float
    strikes: FloatArray
    cp: NDArray[np.int64]
    prices: list[PriceResult]
    implied_vols: FloatArray
    implied_vol_stderr: FloatArray
    effective_forward_mean: float  # E[S*_0]; equals the forward up to discretisation

    @property
    def log_moneyness(self) -> FloatArray:
        return np.log(self.strikes / self.forward)

    def __repr__(self) -> str:
        rows = [
            f"  k={k:+.4f}: iv={iv:.5f} ± {se:.5f}"
            for k, iv, se in zip(self.log_moneyness, self.implied_vols, self.implied_vol_stderr)
        ]
        return f"MixingSmile(T={self.maturity}, F={self.forward:.6g},\n" + "\n".join(rows) + "\n)"


def parallel_coefficients(model: BergomiSV) -> tuple[FloatArray, float]:
    """``(c_i, λ²)`` of eq. 8.62 (1F: ``c = ρ_SX1``, ``λ² = ρ_SX1²``)."""
    p = model.params
    if model.n_factors == 1:
        return np.array([p.rho_SX1]), p.rho_SX1**2
    d = 1.0 - p.rho12**2
    if d <= 0:
        raise ValueError("mixing solution needs |rho12| < 1")
    c = np.array([(p.rho_SX1 - p.rho12 * p.rho_SX2) / d, (p.rho_SX2 - p.rho12 * p.rho_SX1) / d])
    lam2 = (p.rho_SX1**2 + p.rho_SX2**2 - 2.0 * p.rho12 * p.rho_SX1 * p.rho_SX2) / d
    return c, float(lam2)


def mixing_integrals(
    model: BergomiSV,
    T: float,
    *,
    n_paths: int,
    seed: int,
    dt: float,
    antithetic: bool = True,
    chunk_size: int = 50_000,
) -> tuple[FloatArray, FloatArray]:
    """Per-path ``(∫₀ᵀ ξ_t^t dt, λ ∫₀ᵀ √ξ_t^t dW^∥_t)`` on a uniform grid of step ``≤ dt``.

    Left-point Riemann / Itô sums with the exact factor increments: the quadrature error is
    O(dt) (at dt = 1/365 and ω ≈ 3.5 about 0.05–0.1 vol points on a 3m smile), so use
    ``dt ≤ 1/2920`` when the mixing smile serves as the reference for the second-order spot step
    (M4b), or match the allowance.
    """
    n_steps = max(1, int(np.ceil(T / dt - 1e-9)))
    h = T / n_steps
    nf = model.n_factors
    m = 1 + 2 * nf
    cov = factor_step_covariance(model.params, h, nf, with_brownians=True)[0]
    chol = sqrt_covariance(cov[None])[0]  # rows: dW^S, dX^1..nf, dW^1..nf
    c, _ = parallel_coefficients(model)
    times = h * np.arange(n_steps + 1)
    g = model.g(times)
    decay = np.exp(-model.ks * h)
    draws = GaussianDraws(seed, n_paths, n_steps, m, antithetic)
    i1 = np.empty(n_paths)
    i2 = np.empty(n_paths)
    if antithetic and chunk_size % 2:
        chunk_size += 1
    for p0 in range(0, n_paths, chunk_size):
        p1 = min(p0 + chunk_size, n_paths)
        n = p1 - p0
        x = np.zeros((n, nf))
        a1 = np.zeros(n)
        a2 = np.zeros(n)
        for j in range(n_steps):
            z = draws.normals(j, p0, p1)  # (n, m)
            inc = z @ chol.T  # (n, m): dW^S (unused), dX^i, dW^i
            v = g[j] * np.exp(x @ model.coef)
            a1 += v * h
            a2 += np.sqrt(v) * (inc[:, 1 + nf :] @ c)
            x = x * decay + inc[:, 1 : 1 + nf]
        i1[p0:p1] = a1
        i2[p0:p1] = a2
    return i1, i2


def mixing_smile(
    model: BergomiSV,
    T: float,
    strikes: ArrayLike,
    *,
    n_paths: int = 100_000,
    seed: int = 0,
    dt: float = 1.0 / 365.0,
    antithetic: bool = True,
    cp: ArrayLike | None = None,
) -> MixingSmile:
    """Vanilla smile of the pure SV model by the mixing solution (eqs. 8.58–8.62).

    ``cp`` defaults to out-of-the-money options (puts below the forward).  Implied vols are
    inverted from the mean price; their stderr is the price stderr divided by the Black vega.
    """
    K = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
    F = float(model.forward_curve.forward(T))
    df = float(model.forward_curve.df(T))
    cps = np.where(K >= F, 1, -1) if cp is None else np.broadcast_to(np.asarray(cp), K.shape)
    i1, i2 = mixing_integrals(model, T, n_paths=n_paths, seed=seed, dt=dt, antithetic=antithetic)
    _, lam2 = parallel_coefficients(model)
    f_star = F * np.exp(-0.5 * lam2 * i1 + i2)
    sig_star = np.sqrt((1.0 - lam2) * i1 / T)
    prices = []
    for k, c in zip(K, cps):
        pay = black_price(f_star, k, T, sig_star, int(c), df)
        prices.append(summarize(np.asarray(pay, dtype=np.float64), antithetic))
    mean = np.array([p.mean for p in prices])
    err = np.array([p.stderr for p in prices])
    iv = implied_vol(mean, F, K, T, cps, df)
    iv_se = err / black_vega(F, K, T, iv, df)
    return MixingSmile(
        T, F, K, np.asarray(cps, dtype=np.int64), prices, iv, iv_se, float(f_star.mean())
    )


def mixing_atmf_skew(
    model: BergomiSV,
    T: float,
    *,
    h: float = 0.01,
    n_paths: int = 100_000,
    seed: int = 0,
    dt: float = 1.0 / 365.0,
    antithetic: bool = True,
) -> tuple[float, float, float]:
    """``(S_T, σ̂_ATMF, stderr(S_T))`` with ``S_T = ∂σ̂/∂k|_F`` by a central difference of half-width
    ``h`` in log-moneyness on common paths.

    The error is that of the per-path difference ``P₊/vega₊ − P₋/vega₋`` (the two implied vols
    share the paths, so it is far below the quadrature sum of the individual errors).
    """
    F = float(model.forward_curve.forward(T))
    df = float(model.forward_curve.df(T))
    i1, i2 = mixing_integrals(model, T, n_paths=n_paths, seed=seed, dt=dt, antithetic=antithetic)
    _, lam2 = parallel_coefficients(model)
    f_star = F * np.exp(-0.5 * lam2 * i1 + i2)
    sig_star = np.sqrt((1.0 - lam2) * i1 / T)
    K = F * np.exp(np.array([-h, 0.0, h]))
    cps = np.array([-1, 1, 1])
    pays = [
        np.asarray(black_price(f_star, k, T, sig_star, int(c), df), dtype=np.float64)
        for k, c in zip(K, cps)
    ]
    means = np.array([pay.mean() for pay in pays])
    iv = implied_vol(means, F, K, T, cps, df)
    vega = black_vega(F, K, T, iv, df)
    diff = summarize(pays[2] / vega[2] - pays[0] / vega[0], antithetic)
    skew = float((iv[2] - iv[0]) / (2.0 * h))
    return skew, float(iv[1]), float(diff.stderr / (2.0 * h))
