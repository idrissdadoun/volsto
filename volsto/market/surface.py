"""Implied-volatility surfaces (SPEC §2.2): ABC, Gatheral–Jacquier SSVI, market slice grid.

Conventions: ``k = ln(K / F(T))`` and ``w(k, T) = σ̂(k, T)² T`` (total implied variance).
Checked by ``tests/test_surface.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.market.bs import black_price, strike_from_delta

if TYPE_CHECKING:
    from volsto.config import SSVIConfig
    from volsto.market.curves import DiscountCurve, ForwardCurve

FloatArray = NDArray[np.float64]


class ImpliedSurface(ABC):
    """Arbitrage-checked implied surface parametrised in ``(k, T)`` total variance."""

    def __init__(
        self, forward_curve: ForwardCurve, discount: DiscountCurve, max_maturity: float
    ) -> None:
        if max_maturity <= 0:
            raise ValueError("max_maturity must be positive")
        self.forward_curve = forward_curve
        self.discount = discount
        self.max_maturity = float(max_maturity)

    @abstractmethod
    def total_variance(self, k: ArrayLike, T: ArrayLike) -> FloatArray:
        """``w(k, T)``, broadcasting over ``k`` and ``T``."""

    def forward(self, T: ArrayLike) -> FloatArray:
        return self.forward_curve.forward(T)

    def log_moneyness(self, K: ArrayLike, T: ArrayLike) -> FloatArray:
        return np.log(np.asarray(K, dtype=np.float64) / self.forward(T))

    def implied_vol_k(self, k: ArrayLike, T: ArrayLike) -> FloatArray:
        """``σ̂(k, T) = sqrt(w(k, T)/T)``."""
        T_ = np.asarray(T, dtype=np.float64)
        return np.sqrt(self.total_variance(k, T_) / T_)

    def implied_vol(self, K: ArrayLike, T: ArrayLike) -> FloatArray:
        return self.implied_vol_k(self.log_moneyness(K, T), T)

    def atm_vol(self, T: ArrayLike) -> FloatArray:
        """ATMF implied vol ``σ̂(0, T)``."""
        return self.implied_vol_k(0.0, T)

    def price(self, K: ArrayLike, T: ArrayLike, cp: ArrayLike) -> FloatArray:
        """Discounted Black price at the surface's implied vol."""
        T_ = np.asarray(T, dtype=np.float64)
        return black_price(
            self.forward(T_), K, T_, self.implied_vol(K, T_), cp, self.discount.df(T_)
        )

    def undiscounted_price_k(self, k: ArrayLike, T: ArrayLike, cp: ArrayLike) -> FloatArray:
        """Undiscounted Black price at log-moneyness ``k`` (used by the log-contract strip)."""
        T_ = np.asarray(T, dtype=np.float64)
        F = self.forward(T_)
        K = F * np.exp(np.asarray(k, dtype=np.float64))
        return black_price(F, K, T_, self.implied_vol_k(k, T_), cp, 1.0)


# --------------------------------------------------------------------------------------------
# SSVI
# --------------------------------------------------------------------------------------------


class SSVISurface(ImpliedSurface):
    r"""Gatheral–Jacquier (2014) power-law SSVI.

    .. math::
        w(k, T) = \tfrac{\theta_T}{2}\Big(1 + \rho\varphi(\theta_T)k
                  + \sqrt{(\varphi(\theta_T)k + \rho)^2 + 1 - \rho^2}\Big),\qquad
        \varphi(\theta) = \frac{\eta}{\theta^\gamma(1+\theta)^{1-\gamma}}

    ``θ_T`` is the ATM total variance, linear in ``T`` between pillars (flat forward variance),
    ``θ_0 = 0`` and the last forward variance extended flat.  No-arbitrage checks (Gatheral–Jacquier
    Thm 4.1/4.2, SPEC §2.2) run at construction on ``θ ∈ [θ(min_maturity), θ(max_maturity)]``:

    * butterfly: ``θφ(θ)(1+|ρ|) < 4`` and ``θφ(θ)²(1+|ρ|) ≤ 4``,
    * calendar: ``θ_T`` non-decreasing and ``∂_θ(θφ(θ)) ≥ 0``.

    Checked by ``tests/test_surface.py``.
    """

    def __init__(
        self,
        atm_maturities: Sequence[float] | FloatArray,
        atm_total_variances: Sequence[float] | FloatArray,
        rho: float,
        eta: float,
        gamma: float,
        forward_curve: ForwardCurve,
        discount: DiscountCurve,
        *,
        max_maturity: float = 10.0,
        min_maturity: float = 1.0 / 365.0,
    ) -> None:
        super().__init__(forward_curve, discount, max_maturity)
        t = np.asarray(atm_maturities, dtype=np.float64).ravel()
        th = np.asarray(atm_total_variances, dtype=np.float64).ravel()
        if t.size == 0 or t.shape != th.shape:
            raise ValueError("atm_maturities and atm_total_variances must match and be non-empty")
        if np.any(t <= 0) or np.any(np.diff(t) <= 0):
            raise ValueError("atm_maturities must be positive and strictly increasing")
        if np.any(th <= 0) or np.any(np.diff(th) <= 0):
            raise ValueError(
                "ATM total variance must be positive and strictly increasing (calendar)"
            )
        if not -1.0 < rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")
        if eta < 0:
            raise ValueError("eta must be non-negative")
        if not 0.0 < gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if not 0 < min_maturity < max_maturity:
            raise ValueError("need 0 < min_maturity < max_maturity")
        self._t = np.concatenate(([0.0], t))
        self._theta = np.concatenate(([0.0], th))
        self._slope_last = (self._theta[-1] - self._theta[-2]) / (self._t[-1] - self._t[-2])
        self.rho = float(rho)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.min_maturity = float(min_maturity)
        self.check_no_arbitrage()

    # -- constructors ------------------------------------------------------------------------

    @classmethod
    def from_config(
        cls, cfg: SSVIConfig, forward_curve: ForwardCurve, discount: DiscountCurve
    ) -> SSVISurface:
        t = np.asarray(cfg.atm_maturities)
        v = np.asarray(cfg.atm_vols)
        return cls(
            t,
            v * v * t,
            cfg.rho,
            cfg.eta,
            cfg.gamma,
            forward_curve,
            discount,
            max_maturity=cfg.max_maturity,
        )

    @classmethod
    def flat_atm(
        cls,
        vol: float,
        rho: float,
        eta: float,
        gamma: float,
        forward_curve: ForwardCurve,
        discount: DiscountCurve,
        *,
        max_maturity: float = 10.0,
    ) -> SSVISurface:
        """Flat ATM vol term structure ``θ_T = σ² T``."""
        return cls(
            [max_maturity],
            [vol * vol * max_maturity],
            rho,
            eta,
            gamma,
            forward_curve,
            discount,
            max_maturity=max_maturity,
        )

    # -- pieces ------------------------------------------------------------------------------

    def theta(self, T: ArrayLike) -> FloatArray:
        """ATM total variance ``θ_T``."""
        T_ = np.asarray(T, dtype=np.float64)
        if np.any(T_ < 0):
            raise ValueError("maturity must be non-negative")
        inside = np.interp(T_, self._t, self._theta)
        beyond = self._theta[-1] + self._slope_last * (T_ - self._t[-1])
        return np.where(self._t[-1] < T_, beyond, inside)

    def phi(self, theta: ArrayLike) -> FloatArray:
        """Power-law ``φ(θ) = η / (θ^γ (1+θ)^{1−γ})``."""
        th = np.asarray(theta, dtype=np.float64)
        return self.eta / (th**self.gamma * (1.0 + th) ** (1.0 - self.gamma))

    def total_variance(self, k: ArrayLike, T: ArrayLike) -> FloatArray:
        k_ = np.asarray(k, dtype=np.float64)
        th = self.theta(T)
        ph = self.phi(th)
        rho = self.rho
        pk = ph * k_
        w = 0.5 * th * (1.0 + rho * pk + np.sqrt((pk + rho) ** 2 + 1.0 - rho * rho))
        return np.asarray(w, dtype=np.float64)

    def atm_skew(self, T: ArrayLike) -> FloatArray:
        """ATMF skew ``∂σ̂/∂k |_{k=0} = ρ φ(θ) sqrt(θ) / (2 √T)`` (from ``∂_k w(0) = θρφ/2``)."""
        T_ = np.asarray(T, dtype=np.float64)
        th = self.theta(T_)
        return np.asarray(
            0.5 * self.rho * self.phi(th) * np.sqrt(th) / np.sqrt(T_), dtype=np.float64
        )

    def check_no_arbitrage(self, n_grid: int = 2001) -> None:
        """Raise ``ValueError`` on butterfly / calendar violations (see class docstring)."""
        th_lo = float(self.theta(self.min_maturity))
        th_hi = float(self.theta(self.max_maturity))
        th = np.linspace(th_lo, th_hi, n_grid)
        ph = self.phi(th)
        a = np.abs(self.rho)
        c1 = th * ph * (1.0 + a)
        c2 = th * ph * ph * (1.0 + a)
        if np.any(c1 >= 4.0):
            i = int(np.argmax(c1))
            raise ValueError(
                f"SSVI butterfly arbitrage: theta*phi*(1+|rho|) = {c1[i]:.4f} >= 4 "
                f"at theta = {th[i]:.4f}"
            )
        if np.any(c2 > 4.0):
            i = int(np.argmax(c2))
            raise ValueError(
                f"SSVI butterfly arbitrage: theta*phi^2*(1+|rho|) = {c2[i]:.4f} > 4 "
                f"at theta = {th[i]:.4f}"
            )
        tp = th * ph
        if np.any(np.diff(tp) < -1e-12):
            raise ValueError("SSVI calendar arbitrage: θφ(θ) must be non-decreasing in θ")

    def __repr__(self) -> str:
        return (
            f"SSVISurface(atm_maturities={self._t[1:].tolist()}, "
            f"atm_vols={np.sqrt(self._theta[1:] / self._t[1:]).round(6).tolist()}, "
            f"rho={self.rho}, eta={self.eta}, gamma={self.gamma})"
        )


# --------------------------------------------------------------------------------------------
# Market slice grid
# --------------------------------------------------------------------------------------------


class GridSurface(ImpliedSurface):
    """Market slices ``(k_ij, w_ij)`` per maturity ``T_i``.

    Interpolation (SPEC §2.2): linear in ``w`` along ``k`` inside a slice, flat in implied vol
    beyond the wings (``w`` constant at fixed ``T``); linear in ``w`` along ``T`` at fixed ``k``
    between slices, linear from ``w = 0`` at ``T = 0`` before the first slice and flat implied vol
    (``w ∝ T``) after the last.  Calendar monotonicity in ``T`` is checked at construction.
    """

    def __init__(
        self,
        maturities: Sequence[float] | FloatArray,
        log_moneyness: Sequence[Sequence[float] | FloatArray],
        total_variances: Sequence[Sequence[float] | FloatArray],
        forward_curve: ForwardCurve,
        discount: DiscountCurve,
        *,
        max_maturity: float | None = None,
    ) -> None:
        t = np.asarray(maturities, dtype=np.float64).ravel()
        if t.size == 0 or np.any(t <= 0) or np.any(np.diff(t) <= 0):
            raise ValueError("maturities must be positive and strictly increasing")
        if len(log_moneyness) != t.size or len(total_variances) != t.size:
            raise ValueError("one (k, w) slice per maturity is required")
        super().__init__(forward_curve, discount, max_maturity or float(t[-1]))
        self._t = t
        self._k: list[FloatArray] = []
        self._w: list[FloatArray] = []
        for i, (kk, ww) in enumerate(zip(log_moneyness, total_variances)):
            k_ = np.asarray(kk, dtype=np.float64).ravel()
            w_ = np.asarray(ww, dtype=np.float64).ravel()
            if k_.size < 2 or k_.shape != w_.shape:
                raise ValueError(f"slice {i}: need ≥ 2 quotes with matching shapes")
            order = np.argsort(k_)
            k_, w_ = k_[order], w_[order]
            if np.any(np.diff(k_) <= 0):
                raise ValueError(f"slice {i}: duplicate log-moneyness")
            if np.any(w_ <= 0):
                raise ValueError(f"slice {i}: total variance must be positive")
            self._k.append(k_)
            self._w.append(w_)
        self._check_calendar()

    @classmethod
    def from_strike_vols(
        cls,
        maturities: Sequence[float],
        strikes: Sequence[Sequence[float] | FloatArray],
        vols: Sequence[Sequence[float] | FloatArray],
        forward_curve: ForwardCurve,
        discount: DiscountCurve,
        *,
        max_maturity: float | None = None,
    ) -> GridSurface:
        ks, ws = [], []
        for T, K, v in zip(maturities, strikes, vols):
            K_ = np.asarray(K, dtype=np.float64)
            v_ = np.asarray(v, dtype=np.float64)
            ks.append(np.log(K_ / forward_curve.forward(T)))
            ws.append(v_ * v_ * T)
        return cls(maturities, ks, ws, forward_curve, discount, max_maturity=max_maturity)

    @classmethod
    def from_delta_vols(
        cls,
        maturities: Sequence[float],
        deltas: Sequence[Sequence[float] | FloatArray],
        vols: Sequence[Sequence[float] | FloatArray],
        forward_curve: ForwardCurve,
        discount: DiscountCurve,
        *,
        max_maturity: float | None = None,
    ) -> GridSurface:
        """Slices quoted in spot delta (calls positive, puts negative)."""
        strikes = []
        for T, d, v in zip(maturities, deltas, vols):
            F = float(forward_curve.forward(T))
            q = float(-forward_curve.dividend_curve.log_df(T) / T)
            strikes.append(strike_from_delta(d, F, T, v, q))
        return cls.from_strike_vols(
            maturities, strikes, vols, forward_curve, discount, max_maturity=max_maturity
        )

    def _slice_w(self, i: int, k: FloatArray) -> FloatArray:
        return np.interp(k, self._k[i], self._w[i])

    def _check_calendar(self) -> None:
        k_all = np.unique(np.concatenate(self._k))
        prev = np.zeros_like(k_all)
        for i in range(self._t.size):
            cur = self._slice_w(i, k_all)
            if np.any(cur < prev - 1e-12):
                j = int(np.argmin(cur - prev))
                raise ValueError(
                    f"calendar arbitrage between slices {i - 1} and {i} at k={k_all[j]:.4f}"
                )
            prev = cur

    def total_variance(self, k: ArrayLike, T: ArrayLike) -> FloatArray:
        k_, T_ = np.broadcast_arrays(
            np.asarray(k, dtype=np.float64), np.asarray(T, dtype=np.float64)
        )
        if np.any(T_ < 0):
            raise ValueError("maturity must be non-negative")
        out = np.empty(k_.shape, dtype=np.float64)
        flat_k = k_.ravel()
        flat_T = T_.ravel()
        flat_out = out.reshape(-1)
        t = self._t
        for Tv in np.unique(flat_T):
            m = flat_T == Tv
            kk = flat_k[m]
            if Tv <= t[0]:
                val = self._slice_w(0, kk) * (Tv / t[0])
            elif Tv >= t[-1]:
                val = self._slice_w(t.size - 1, kk) * (Tv / t[-1])
            else:
                i = int(np.searchsorted(t, Tv, side="right")) - 1
                w0 = self._slice_w(i, kk)
                w1 = self._slice_w(i + 1, kk)
                lam = (Tv - t[i]) / (t[i + 1] - t[i])
                val = (1.0 - lam) * w0 + lam * w1
            flat_out[m] = val
        return out

    @property
    def maturities(self) -> FloatArray:
        return self._t.copy()

    def __repr__(self) -> str:
        return f"GridSurface(maturities={self._t.tolist()}, n_quotes={[k.size for k in self._k]})"
