"""Variance-swap strip by log-contract replication and the forward-variance curve ξ₀ (SPEC §2.4).

.. math::
    K_{var}(T) = \\frac{2}{T}\\int_0^\\infty \\frac{\\tilde O(K, T)}{K^2}\\,dK
               = \\frac{2}{T}\\int_{-\\infty}^{\\infty} \\frac{\\tilde O(k, T)}{F e^{k}}\\,dk ,
               \\qquad k = \\ln(K/F),\\ dK = K\\,dk,

with ``Õ`` the undiscounted out-of-the-money Black price (put below the forward, call above).

Source: Demeterfi–Derman–Kamal–Zou (1999); Bergomi ch. 5 (log-contract replication).
``ξ₀(T) = d/dT [T K_var(T)]`` is the initial forward-variance curve of the Bergomi models
(SPEC §3.3).  Checked by ``tests/test_varswap.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad
from scipy.interpolate import PchipInterpolator

from volsto.market.surface import ImpliedSurface

FloatArray = NDArray[np.float64]


def varswap_strike(
    surface: ImpliedSurface,
    T: float,
    *,
    n_std: float = 25.0,
    k_min_width: float = 3.0,
    epsrel: float = 1e-10,
    epsabs: float = 0.0,
) -> float:
    """Fair variance-swap strike ``K_var(T)`` (annualised variance) by log-contract replication.

    Adaptive quadrature (``scipy.integrate.quad``) in ``k`` on ``[-k_max, 0]`` and ``[0, k_max]``
    with ``k_max = max(n_std · σ_ATM √T, k_min_width)``.  Bounds are deliberately wide: SSVI put
    wings decay slowly (``k_max = 2.4`` at 1y truncated 0.01 vol points on the reference surface).
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if surface.max_maturity < T:
        raise ValueError("T exceeds the surface's max_maturity")
    sig_atm = float(surface.atm_vol(T))
    k_max = max(n_std * sig_atm * math.sqrt(T), k_min_width)
    F = float(surface.forward(T))

    def integrand(k: float, cp: float) -> float:
        # Õ(k, T) / (F e^k): the OTM undiscounted price divided by the strike
        return float(surface.undiscounted_price_k(k, T, cp)) * math.exp(-k) / F

    put, _ = quad(integrand, -k_max, 0.0, args=(-1.0,), epsrel=epsrel, epsabs=epsabs, limit=400)
    call, _ = quad(integrand, 0.0, k_max, args=(1.0,), epsrel=epsrel, epsabs=epsabs, limit=400)
    return float(2.0 / T * (put + call))


def varswap_strip(surface: ImpliedSurface, maturities: ArrayLike, **kwargs: float) -> FloatArray:
    """``K_var`` at each maturity."""
    Ts = np.atleast_1d(np.asarray(maturities, dtype=np.float64))
    return np.array([varswap_strike(surface, float(T), **kwargs) for T in Ts])


class ForwardVarianceCurve:
    """Initial forward-variance curve ``ξ₀(T) = d/dT [T K_var(T)]`` (SPEC §2.4, §3.3).

    Built from the total-variance strip ``W(T) = T K_var(T)`` on a fine maturity grid with a
    shape-preserving (PCHIP) interpolant through ``W(0) = 0``; ``ξ₀ = W'`` is then continuous,
    non-negative whenever the strip is increasing, and ``∫₀ᵀ ξ₀ = W(T)`` exactly at the nodes.
    Checked by ``tests/test_varswap.py::test_xi0_integrates_back_to_strip``.
    """

    def __init__(self, maturities: ArrayLike, total_variances: ArrayLike) -> None:
        T = np.asarray(maturities, dtype=np.float64).ravel()
        W = np.asarray(total_variances, dtype=np.float64).ravel()
        if T.size < 2 or T.shape != W.shape:
            raise ValueError("need ≥ 2 maturities with matching total variances")
        if np.any(T <= 0) or np.any(np.diff(T) <= 0):
            raise ValueError("maturities must be positive and strictly increasing")
        if np.any(W <= 0) or np.any(np.diff(W) <= 0):
            raise ValueError("total variance strip must be positive and strictly increasing")
        self._T = np.concatenate(([0.0], T))
        self._W = np.concatenate(([0.0], W))
        self._spline = PchipInterpolator(self._T, self._W, extrapolate=False)
        self._deriv = self._spline.derivative()
        self._slope_last = float(self._deriv(self._T[-1]))
        fine = np.linspace(0.0, self._T[-1], max(4 * T.size, 1001))
        xi = self.xi0(fine)
        if np.any(xi <= 0):
            raise ValueError(f"forward variance must be positive; min ξ₀ = {xi.min():.3e}")

    @classmethod
    def flat(cls, variance: float, t_max: float = 10.0) -> ForwardVarianceCurve:
        """Flat ``ξ₀ ≡ variance``."""
        T = np.array([0.5 * t_max, t_max])
        return cls(T, variance * T)

    @classmethod
    def from_surface(
        cls,
        surface: ImpliedSurface,
        t_max: float,
        *,
        maturities: Sequence[float] | FloatArray | None = None,
        per_year: int = 52,
    ) -> ForwardVarianceCurve:
        """Strip the surface on a fine grid (weekly by default, denser in the first month)."""
        if maturities is None:
            short = np.array([1.0, 2.0, 3.0, 5.0]) / 365.0
            weekly = np.arange(1, int(np.ceil(per_year * t_max)) + 1) / per_year
            weekly = weekly[weekly < t_max]
            Ts = np.unique(np.concatenate((short, weekly, [t_max])))
        else:
            Ts = np.asarray(maturities, dtype=np.float64)
        strip = varswap_strip(surface, Ts)
        return cls(Ts, Ts * strip)

    @property
    def maturities(self) -> FloatArray:
        return self._T[1:].copy()

    @property
    def t_max(self) -> float:
        return float(self._T[-1])

    def total_variance(self, T: ArrayLike) -> FloatArray:
        """``W(T) = ∫₀ᵀ ξ₀(u) du``; linear (last ξ₀) beyond the last node."""
        T_ = np.asarray(T, dtype=np.float64)
        if np.any(T_ < 0):
            raise ValueError("T must be non-negative")
        inside = self._spline(np.minimum(T_, self._T[-1]))
        beyond = self._W[-1] + self._slope_last * (T_ - self._T[-1])
        return np.asarray(np.where(self._T[-1] < T_, beyond, inside), dtype=np.float64)

    def xi0(self, T: ArrayLike) -> FloatArray:
        """``ξ₀(T) = W'(T)``; flat beyond the last node."""
        T_ = np.asarray(T, dtype=np.float64)
        if np.any(T_ < 0):
            raise ValueError("T must be non-negative")
        inside = self._deriv(np.minimum(T_, self._T[-1]))
        return np.asarray(np.where(self._T[-1] < T_, self._slope_last, inside), dtype=np.float64)

    __call__ = xi0

    def integral(self, t1: ArrayLike, t2: ArrayLike) -> FloatArray:
        """``∫_{t1}^{t2} ξ₀(u) du``."""
        return self.total_variance(t2) - self.total_variance(t1)

    def varswap_strike(self, T: ArrayLike) -> FloatArray:
        """``K_var(T) = W(T)/T``."""
        T_ = np.asarray(T, dtype=np.float64)
        return np.asarray(self.total_variance(T_) / T_, dtype=np.float64)

    def forward_varswap_strike(self, t1: ArrayLike, t2: ArrayLike) -> FloatArray:
        """Forward variance-swap strike over ``[t1, t2]``."""
        a = np.asarray(t1, dtype=np.float64)
        b = np.asarray(t2, dtype=np.float64)
        return np.asarray(self.integral(a, b) / (b - a), dtype=np.float64)

    def __repr__(self) -> str:
        return f"ForwardVarianceCurve(t_max={self.t_max}, n_nodes={self._T.size - 1})"


def xi0_curve(surface: ImpliedSurface, t_max: float, **kwargs: object) -> ForwardVarianceCurve:
    """``ξ₀(T)`` from a surface (SPEC §2.4); see :class:`ForwardVarianceCurve.from_surface`."""
    return ForwardVarianceCurve.from_surface(surface, t_max, **kwargs)  # type: ignore[arg-type]
