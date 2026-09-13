"""Closed forms of the two-factor lognormal Bergomi model (SPEC §3.3).

Sources: Bergomi, *Stochastic Volatility Modeling*, ch. 7 (eqs. 7.29–7.39), ch. 8 (eq. 8.55),
ch. 9 (eqs. 9.18–9.21); the diagonal covariances are derived here from the OU covariances.
Every function takes a :class:`~volsto.config.BergomiParams` ``p`` and, where the term structure
matters, a :class:`~volsto.market.varswap.ForwardVarianceCurve` ``xi0``.  Weights are
``w1 = 1 − θ``, ``w2 = θ``.  Checked by ``tests/test_bergomi.py``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.config import BergomiParams
from volsto.market.varswap import ForwardVarianceCurve

FloatArray = NDArray[np.float64]


def _weights(p: BergomiParams) -> tuple[float, float]:
    return 1.0 - p.theta, p.theta


def alpha_theta(p: BergomiParams) -> float:
    """``α_θ = 1 / sqrt((1−θ)² + θ² + 2 ρ12 θ (1−θ))`` (eq. 7.29); ``α_0 = 1``."""
    w1, w2 = _weights(p)
    return float(1.0 / np.sqrt(w1 * w1 + w2 * w2 + 2.0 * p.rho12 * w1 * w2))


def eta_u(p: BergomiParams, u: ArrayLike) -> FloatArray:
    """``η(u) = α_θ sqrt((1−θ)² e^{−2k1u} + θ² e^{−2k2u} + 2ρ12θ(1−θ) e^{−(k1+k2)u})`` (eq. 7.31b).

    Instantaneous lognormal volatility of ``ξ_t^T`` divided by ``ω`` at ``u = T − t``; ``η(0) = 1``.
    """
    w1, w2 = _weights(p)
    u_ = np.asarray(u, dtype=np.float64)
    a = alpha_theta(p)
    inner = (
        w1 * w1 * np.exp(-2.0 * p.k1 * u_)
        + w2 * w2 * np.exp(-2.0 * p.k2 * u_)
        + 2.0 * p.rho12 * w1 * w2 * np.exp(-(p.k1 + p.k2) * u_)
    )
    return np.asarray(a * np.sqrt(inner), dtype=np.float64)


def chi(p: BergomiParams, t: ArrayLike, T: ArrayLike) -> FloatArray:
    """``χ(t, T) = ∫_{T−t}^{T} η²(u) du = Var[x_t^T]`` (eq. 7.35):

    ``α_θ² [ (1−θ)² e^{−2k1(T−t)} (1−e^{−2k1 t})/(2k1) + θ² e^{−2k2(T−t)} (1−e^{−2k2 t})/(2k2)
    + 2θ(1−θ)ρ12 e^{−(k1+k2)(T−t)} (1−e^{−(k1+k2) t})/(k1+k2) ]``.
    Checked by ``tests/test_bergomi.py::test_chi_matches_sample_variance``.
    """
    w1, w2 = _weights(p)
    t_ = np.asarray(t, dtype=np.float64)
    T_ = np.asarray(T, dtype=np.float64)
    tau = T_ - t_
    a2 = alpha_theta(p) ** 2
    k1, k2, k12 = p.k1, p.k2, p.k1 + p.k2
    out = a2 * (
        w1 * w1 * np.exp(-2.0 * k1 * tau) * (1.0 - np.exp(-2.0 * k1 * t_)) / (2.0 * k1)
        + w2 * w2 * np.exp(-2.0 * k2 * tau) * (1.0 - np.exp(-2.0 * k2 * t_)) / (2.0 * k2)
        + 2.0 * w1 * w2 * p.rho12 * np.exp(-k12 * tau) * (1.0 - np.exp(-k12 * t_)) / k12
    )
    return np.asarray(out, dtype=np.float64)


def cov_x_diag(p: BergomiParams, u: ArrayLike, v: ArrayLike) -> FloatArray:
    """``Cov(x_u^u, x_v^v)`` — derived from the OU covariances (SPEC §3.3), for any ``u, v``:

    with ``a = min(u,v)``, ``d = |v − u|``:
    ``α_θ² [ (1−θ)² e^{−k1 d} (1−e^{−2k1 a})/(2k1) + θ² e^{−k2 d} (1−e^{−2k2 a})/(2k2)
    + θ(1−θ)ρ12 (1−e^{−(k1+k2) a})/(k1+k2) (e^{−k1 d} + e^{−k2 d}) ]``.
    Checked by ``tests/test_bergomi.py::test_diagonal_covariances``.
    """
    w1, w2 = _weights(p)
    u_ = np.asarray(u, dtype=np.float64)
    v_ = np.asarray(v, dtype=np.float64)
    a = np.minimum(u_, v_)
    d = np.abs(v_ - u_)
    a2 = alpha_theta(p) ** 2
    k1, k2, k12 = p.k1, p.k2, p.k1 + p.k2
    e1, e2 = np.exp(-k1 * d), np.exp(-k2 * d)
    out = a2 * (
        w1 * w1 * e1 * (1.0 - np.exp(-2.0 * k1 * a)) / (2.0 * k1)
        + w2 * w2 * e2 * (1.0 - np.exp(-2.0 * k2 * a)) / (2.0 * k2)
        + w1 * w2 * p.rho12 * (1.0 - np.exp(-k12 * a)) / k12 * (e1 + e2)
    )
    return np.asarray(out, dtype=np.float64)


def cov_xi_diag(
    p: BergomiParams, xi0: ForwardVarianceCurve, u: ArrayLike, v: ArrayLike
) -> FloatArray:
    """``Cov(ξ_u^u, ξ_v^v) = ξ_0^u ξ_0^v (exp(ω² Cov(x_u^u, x_v^v)) − 1)`` (derived, lognormal)."""
    u_ = np.asarray(u, dtype=np.float64)
    v_ = np.asarray(v, dtype=np.float64)
    return np.asarray(
        xi0.xi0(u_) * xi0.xi0(v_) * np.expm1(p.omega**2 * cov_x_diag(p, u_, v_)), dtype=np.float64
    )


def _gauss_legendre(a: float, b: float, n: int) -> tuple[FloatArray, FloatArray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


def var_integrated_variance(
    p: BergomiParams, xi0: ForwardVarianceCurve, T1: float, T2: float, n_quad: int = 48
) -> float:
    """``Var(∫_{T1}^{T2} ξ_t^t dt) = ∫∫ Cov(ξ_u^u, ξ_v^v) du dv`` (book eqs. 7.19–7.20 machinery
    on the diagonal, SPEC §3.3), by Gauss–Legendre on the two triangles (the integrand has a
    derivative kink on ``u = v``).  Checked by ``tests/test_bergomi.py::test_diagonal_covariances``.
    """
    if not 0 <= T1 < T2:
        raise ValueError("need 0 <= T1 < T2")
    u, wu = _gauss_legendre(T1, T2, n_quad)
    total = 0.0
    for ui, wi in zip(u, wu):
        v, wv = _gauss_legendre(float(ui), T2, n_quad)
        total += wi * float(np.sum(wv * cov_xi_diag(p, xi0, ui, v)))
    return 2.0 * total


# --------------------------------------------------------------------------------------------
# volatility of VS volatilities (eqs. 7.38–7.39, 7.24–7.26)
# --------------------------------------------------------------------------------------------


def _I(x: FloatArray) -> FloatArray:
    """``I(x) = (1 − e^{−x}) / x`` (eq. 7.25), ``I(0) = 1``."""
    x_ = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.expm1(-x_) / x_
    return np.asarray(np.where(x_ == 0.0, 1.0, out), dtype=np.float64)


def _A_i(
    p: BergomiParams, xi0: ForwardVarianceCurve | None, t: float, T: float, n_quad: int
) -> tuple[float, float]:
    """``A_i = ∫_t^T ξ_t^τ e^{−k_i(τ−t)} dτ / ∫_t^T ξ_t^τ dτ`` at ``t = 0`` with the initial
    curve (eq. 7.38); flat curve ⇒ ``A_i = I(k_i (T − t))``."""
    if xi0 is None:
        return float(_I(np.array(p.k1 * (T - t)))), float(_I(np.array(p.k2 * (T - t))))
    if t != 0.0:
        raise ValueError("term-structure form is available at t = 0 only (uses ξ_0)")
    tau, w = _gauss_legendre(0.0, T, n_quad)
    xi = xi0.xi0(tau)
    den = float(np.sum(w * xi))
    a1 = float(np.sum(w * xi * np.exp(-p.k1 * tau))) / den
    a2 = float(np.sum(w * xi * np.exp(-p.k2 * tau))) / den
    return a1, a2


def vs_vol_of_vol(
    p: BergomiParams,
    T: float,
    xi0: ForwardVarianceCurve | None = None,
    t: float = 0.0,
    n_quad: int = 64,
) -> float:
    """Instantaneous lognormal volatility of the VS volatility ``σ̂_T(t)`` (eq. 7.39):

    ``ν_T(t) = ν α_θ sqrt((1−θ)² A1² + θ² A2² + 2 ρ12 θ (1−θ) A1 A2)`` with ``A_i`` from eq. 7.38
    (``A_i = I(k_i (T−t))`` for a flat term structure, eq. 7.24).
    Checked by ``tests/test_bergomi.py::test_vs_vol_of_vol``.
    """
    w1, w2 = _weights(p)
    a1, a2 = _A_i(p, xi0, t, T, n_quad)
    return float(
        p.nu
        * alpha_theta(p)
        * np.sqrt(w1 * w1 * a1 * a1 + w2 * w2 * a2 * a2 + 2.0 * p.rho12 * w1 * w2 * a1 * a2)
    )


def vs_vol_of_vol_flat(p: BergomiParams, T: ArrayLike, t: float = 0.0) -> FloatArray:
    """Vectorised flat-term-structure form of :func:`vs_vol_of_vol` (eq. 7.24 / 7.39)."""
    w1, w2 = _weights(p)
    tau = np.asarray(T, dtype=np.float64) - t
    a1, a2 = _I(p.k1 * tau), _I(p.k2 * tau)
    return np.asarray(
        p.nu
        * alpha_theta(p)
        * np.sqrt(w1 * w1 * a1 * a1 + w2 * w2 * a2 * a2 + 2.0 * p.rho12 * w1 * w2 * a1 * a2),
        dtype=np.float64,
    )


def forward_vs_vol_of_vol_flat(p: BergomiParams, T1: float, T2: float, t: float = 0.0) -> float:
    """Instantaneous vol of the forward VS volatility ``σ̂_{T1T2}(t)`` (eq. 7.26, flat curve):

    ``ν α_θ sqrt(Σ_ij w_i w_j ρ_ij I(k_i(T2−T1)) I(k_j(T2−T1)) e^{−(k_i+k_j)(T1−t)})``.
    """
    w1, w2 = _weights(p)
    d = T2 - T1
    g1 = float(_I(np.array(p.k1 * d))) * np.exp(-p.k1 * (T1 - t))
    g2 = float(_I(np.array(p.k2 * d))) * np.exp(-p.k2 * (T1 - t))
    return float(
        p.nu
        * alpha_theta(p)
        * np.sqrt(w1 * w1 * g1 * g1 + w2 * w2 * g2 * g2 + 2.0 * p.rho12 * w1 * w2 * g1 * g2)
    )


# --------------------------------------------------------------------------------------------
# order-one ATMF skew and SSR (eqs. 8.55 / 9.18–9.21)
# --------------------------------------------------------------------------------------------


def _J(x: FloatArray) -> FloatArray:
    """``(x − (1 − e^{−x})) / x²``, the skew kernel of eq. 8.55; ``→ 1/2`` as ``x → 0``."""
    x_ = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = (x_ + np.expm1(-x_)) / (x_ * x_)
    return np.asarray(np.where(x_ == 0.0, 0.5, out), dtype=np.float64)


def atmf_skew_order1_flat(p: BergomiParams, T: ArrayLike) -> FloatArray:
    """Order-one ATMF skew ``S_T = ∂σ̂/∂ln K |_F`` for a flat VS term structure (eq. 8.55 / 9.20):

    ``S_T = ν α_θ [ (1−θ) ρ_SX1 (k1T − (1−e^{−k1T}))/(k1T)²
                  + θ ρ_SX2 (k2T − (1−e^{−k2T}))/(k2T)² ]``.
    Independent of the VS level; rescaling ``(ρ_SX1, ρ_SX2)`` by ``c`` and ``ν`` by ``1/c`` leaves
    it unchanged.  Checked by ``tests/test_bergomi.py::test_mixing_skew_matches_eq_8_55``.
    """
    w1, w2 = _weights(p)
    T_ = np.asarray(T, dtype=np.float64)
    return np.asarray(
        p.nu * alpha_theta(p) * (w1 * p.rho_SX1 * _J(p.k1 * T_) + w2 * p.rho_SX2 * _J(p.k2 * T_)),
        dtype=np.float64,
    )


def _mu_kernel(p: BergomiParams) -> Callable[[FloatArray], FloatArray]:
    w1, w2 = _weights(p)

    def kern(s: FloatArray) -> FloatArray:
        return np.asarray(w1 * p.rho_SX1 * np.exp(-p.k1 * s) + w2 * p.rho_SX2 * np.exp(-p.k2 * s))

    return kern


def _skew_integrals(
    p: BergomiParams, xi0: ForwardVarianceCurve, T: float, n_quad: int
) -> tuple[float, float, float]:
    """Return ``(D, N, σ̂_T²)`` with ``D = ∫₀ᵀ dt sqrt(ξ_0^t) ∫_t^T ξ_0^u k(u−t) du``,
    ``N = ∫₀ᵀ ξ_0^t k(t) dt`` and ``k(s) = (1−θ)ρ_SX1 e^{−k1 s} + θ ρ_SX2 e^{−k2 s}``."""
    kern = _mu_kernel(p)
    t, wt = _gauss_legendre(0.0, T, n_quad)
    xi_t = xi0.xi0(t)
    D = 0.0
    for ti, wi, xti in zip(t, wt, xi_t):
        u, wu = _gauss_legendre(float(ti), T, n_quad)
        D += wi * np.sqrt(xti) * float(np.sum(wu * xi0.xi0(u) * kern(u - ti)))
    N = float(np.sum(wt * xi_t * kern(t)))
    var_T = float(xi0.total_variance(T)) / T
    return D, N, var_T


def atmf_skew_order1(
    p: BergomiParams, xi0: ForwardVarianceCurve, T: float, n_quad: int = 48
) -> float:
    """Order-one ATMF skew for a general term structure (eq. 8.54 / 9.18):

    ``S_T = ν α_θ / (σ̂_T³ T²) ∫₀ᵀ dt sqrt(ξ_0^t) ∫_t^T ξ_0^u k(u−t) du``,
    ``k(s) = (1−θ)ρ_SX1 e^{−k1 s} + θρ_SX2 e^{−k2 s}``.
    Reduces to eq. 8.55 for a flat curve (tested).
    """
    D, _, var_T = _skew_integrals(p, xi0, T, n_quad)
    return float(p.nu * alpha_theta(p) * D / (var_T**1.5 * T * T))


def ssr_order1_flat(p: BergomiParams, T: ArrayLike) -> FloatArray:
    """Order-one skew-stickiness ratio for a flat VS term structure (eq. 9.21):

    ``R_T = [(1−θ)ρ_SX1 I(k1T) + θρ_SX2 I(k2T)] / [(1−θ)ρ_SX1 J(k1T) + θρ_SX2 J(k2T)]`` with
    ``I(x) = (1−e^{−x})/x`` and ``J(x) = (x − (1−e^{−x}))/x²``; ``R_0 = 2``, ``R_∞ = 1``.
    """
    w1, w2 = _weights(p)
    T_ = np.asarray(T, dtype=np.float64)
    num = w1 * p.rho_SX1 * _I(p.k1 * T_) + w2 * p.rho_SX2 * _I(p.k2 * T_)
    den = w1 * p.rho_SX1 * _J(p.k1 * T_) + w2 * p.rho_SX2 * _J(p.k2 * T_)
    return np.asarray(num / den, dtype=np.float64)


def ssr_order1(p: BergomiParams, xi0: ForwardVarianceCurve, T: float, n_quad: int = 48) -> float:
    """Order-one SSR for a general term structure (eq. 9.19):

    ``R_T = σ̂_T² T · N / (sqrt(ξ_0^0) · D)`` with ``N = ∫₀ᵀ ξ_0^t k(t) dt`` and ``D`` as in
    :func:`atmf_skew_order1` — the ratio of ``E[dσ̂_T d ln S]/E[(d ln S)²] = ν N/(σ̂_T T √ξ_0^0)``
    (VS vol as the zeroth-order ATMF vol) to ``S_T`` of eq. 9.18; reduces to eq. 9.21 for a flat
    curve (tested).  The book's typeset prefactor is ``σ̂_T² T / √ξ_0^0``.
    """
    D, N, var_T = _skew_integrals(p, xi0, T, n_quad)
    return float(var_T * T * N / (np.sqrt(float(xi0.xi0(0.0))) * D))
