"""Implied-volatility surfaces (SPEC §2.2): ABC, Gatheral–Jacquier SSVI, market slice grid.

Conventions: ``k = ln(K / F(T))`` and ``w(k, T) = σ̂(k, T)² T`` (total implied variance).
Checked by ``tests/test_surface.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.market.bs import black_price, strike_from_delta

if TYPE_CHECKING:
    from volsto.config import SSVIConfig, SurfacePerturbation
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
    (``w ∝ T``) after the last.  Calendar monotonicity in ``T`` is checked at construction on the
    common quoted ``k`` range of consecutive slices.
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

    def _check_calendar(self, tol: float = 1e-8) -> None:
        """``w`` must not decrease in ``T`` where consecutive slices both have quotes (flat
        extrapolation beyond a slice's wings is a modelling choice and is not compared)."""
        for i in range(1, self._t.size):
            lo = max(self._k[i - 1][0], self._k[i][0])
            hi = min(self._k[i - 1][-1], self._k[i][-1])
            if lo > hi:
                continue
            k_all = np.unique(np.concatenate(self._k[i - 1 : i + 1]))
            k_all = k_all[(k_all >= lo) & (k_all <= hi)]
            prev = self._slice_w(i - 1, k_all)
            cur = self._slice_w(i, k_all)
            if np.any(cur < prev - tol):
                j = int(np.argmin(cur - prev))
                raise ValueError(
                    f"calendar arbitrage between slices {i - 1} and {i} at k={k_all[j]:.4f}"
                )

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


# --------------------------------------------------------------------------------------------
# eSSVI (slice-dependent correlation) — behind the ``essvi`` flag of the importer (SPEC §13)
# --------------------------------------------------------------------------------------------


class ESSVISurface(SSVISurface):
    """SSVI with a maturity-dependent correlation ``ρ_T`` (Hendriks–Martini eSSVI family).

    ``ρ_T`` is piecewise-linear in ``T`` between the ATM pillars (flat outside).  Butterfly
    conditions are checked with ``max|ρ|`` (sufficient); calendar-spread absence is checked
    numerically (``w`` non-decreasing in ``T`` on a ``k`` grid) since the analytic eSSVI
    conditions couple ``ρ_T`` and ``θ_T``.  Intended for later single-stock use.
    """

    def __init__(
        self,
        atm_maturities: Sequence[float] | FloatArray,
        atm_total_variances: Sequence[float] | FloatArray,
        rhos: Sequence[float] | FloatArray,
        eta: float,
        gamma: float,
        forward_curve: ForwardCurve,
        discount: DiscountCurve,
        *,
        max_maturity: float = 10.0,
        min_maturity: float = 1.0 / 365.0,
    ) -> None:
        r = np.asarray(rhos, dtype=np.float64).ravel()
        t = np.asarray(atm_maturities, dtype=np.float64).ravel()
        if r.shape != t.shape:
            raise ValueError("one rho per ATM pillar is required")
        if np.any(np.abs(r) >= 1.0):
            raise ValueError("rhos must lie in (-1, 1)")
        self._rhos = r
        self._rho_t = t
        super().__init__(
            atm_maturities,
            atm_total_variances,
            float(r[np.argmax(np.abs(r))]),
            eta,
            gamma,
            forward_curve,
            discount,
            max_maturity=max_maturity,
            min_maturity=min_maturity,
        )
        self._check_calendar_numeric()

    def rho_T(self, T: ArrayLike) -> FloatArray:
        return np.asarray(
            np.interp(np.asarray(T, dtype=np.float64), self._rho_t, self._rhos), dtype=np.float64
        )

    def total_variance(self, k: ArrayLike, T: ArrayLike) -> FloatArray:
        k_ = np.asarray(k, dtype=np.float64)
        T_ = np.asarray(T, dtype=np.float64)
        th = self.theta(T_)
        ph = self.phi(th)
        rho = self.rho_T(T_)
        pk = ph * k_
        w = 0.5 * th * (1.0 + rho * pk + np.sqrt((pk + rho) ** 2 + 1.0 - rho * rho))
        return np.asarray(w, dtype=np.float64)

    def atm_skew(self, T: ArrayLike) -> FloatArray:
        T_ = np.asarray(T, dtype=np.float64)
        th = self.theta(T_)
        return np.asarray(
            0.5 * self.rho_T(T_) * self.phi(th) * np.sqrt(th) / np.sqrt(T_), dtype=np.float64
        )

    def _check_calendar_numeric(self) -> None:
        ks = np.linspace(-1.0, 1.0, 81)
        Ts = np.concatenate((np.linspace(self.min_maturity, self._t[-1], 200), [self.max_maturity]))
        w = self.total_variance(ks[None, :], Ts[:, None])
        if np.any(np.diff(w, axis=0) < -1e-10):
            raise ValueError("eSSVI calendar arbitrage: total variance decreases in T for some k")

    def __repr__(self) -> str:
        return (
            f"ESSVISurface(atm_maturities={self._t[1:].tolist()}, "
            f"rhos={np.round(self._rhos, 4).tolist()}, eta={self.eta}, gamma={self.gamma})"
        )


# --------------------------------------------------------------------------------------------
# additive perturbation layer (SPEC v2 §7.1, M5)
# --------------------------------------------------------------------------------------------


class ArbitrageError(ValueError):
    """A perturbed surface failed the butterfly or calendar check."""


def tent(T: ArrayLike, pillars: Sequence[float], i: int) -> FloatArray:
    """``tent_i(T)``: 0 at ``T_{i−1}``, 1 at ``T_i``, 0 at ``T_{i+1}``, flat (1) beyond the last
    pillar for the last tent and before the first pillar for the first (SPEC v2 §7.4)."""
    t = np.asarray(T, dtype=np.float64)
    ps = np.asarray(pillars, dtype=np.float64)
    n = ps.size
    if not 0 <= i < n:
        raise ValueError("tent index out of range")
    out = np.zeros_like(t)
    lo = ps[i - 1] if i > 0 else -np.inf
    hi = ps[i + 1] if i < n - 1 else np.inf
    left = (t <= ps[i]) & (t > lo)
    right = (t > ps[i]) & (t < hi)
    up = (t - lo) / (ps[i] - lo) if i > 0 else np.ones_like(t)
    down = (hi - t) / (hi - ps[i]) if i < n - 1 else np.ones_like(t)
    out = np.where(left, up, out)
    out = np.where(right, down, out)
    return np.asarray(np.clip(out, 0.0, 1.0), dtype=np.float64)


class PerturbedSurface(ImpliedSurface):
    """``σ(k, T) = σ_base(k, T) + δσ(k, T)`` (floored at 1e-4); everything else from the base."""

    def __init__(
        self,
        base: ImpliedSurface,
        delta_sigma: Callable[[FloatArray, FloatArray], FloatArray],
        name: str = "perturbed",
        *,
        check: bool = True,
    ) -> None:
        super().__init__(base.forward_curve, base.discount, base.max_maturity)
        self.base = base
        self.delta_sigma = delta_sigma
        self.name = name
        self.min_maturity = getattr(base, "min_maturity", 1.0 / 365.0)
        if check:
            self.check_no_arbitrage()

    def implied_vol_k(self, k: ArrayLike, T: ArrayLike) -> FloatArray:
        k_, T_ = np.broadcast_arrays(
            np.asarray(k, dtype=np.float64), np.asarray(T, dtype=np.float64)
        )
        sig = self.base.implied_vol_k(k_, T_) + self.delta_sigma(k_, T_)
        return np.asarray(np.maximum(sig, 1e-4), dtype=np.float64)

    def total_variance(self, k: ArrayLike, T: ArrayLike) -> FloatArray:
        T_ = np.asarray(T, dtype=np.float64)
        v = self.implied_vol_k(k, T_)
        return np.asarray(v * v * T_, dtype=np.float64)

    def atm_skew(self, T: ArrayLike, h: float = 1e-3) -> FloatArray:
        """``∂σ/∂k`` at ``k = 0`` by central differences."""
        return (self.implied_vol_k(h, T) - self.implied_vol_k(-h, T)) / (2.0 * h)

    def check_no_arbitrage(
        self,
        k_range: float = 1.0,
        n_k: int = 201,
        maturities: Sequence[float] | None = None,
        tol: float = 1e-6,
    ) -> None:
        """Butterfly (Gatheral's density condition, finite differences) and calendar (``w`` non-
        decreasing in ``T`` at fixed ``k``) checks on a ``|k| ≤ k_range`` grid at the risk pillars;
        raise :class:`ArbitrageError` (the risk engine then halves the bump and retries).  The
        skew/curvature bumps saturate beyond ``k_cap`` (0.5) because a rotation extended linearly
        into the far wings breaks the calendar condition where the base surface is nearly
        calendar-flat."""
        ks = np.linspace(-k_range, k_range, n_k)
        ts = (
            np.asarray(maturities, dtype=np.float64)
            if maturities is not None
            else np.array([1 / 12, 2 / 12, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
        )
        ts = ts[(ts >= self.min_maturity) & (ts <= self.max_maturity)]
        dk = ks[1] - ks[0]
        prev = None
        for T in ts:
            w = self.total_variance(ks, T)
            wk = np.gradient(w, dk)
            wkk = np.gradient(wk, dk)
            g = (1.0 - ks * wk / (2.0 * w)) ** 2 - 0.25 * wk * wk * (1.0 / w + 0.25) + 0.5 * wkk
            inner = slice(2, -2)
            if np.any(g[inner] < -tol):
                raise ArbitrageError(
                    f"{self.name}: butterfly condition violated at T={T:g} "
                    f"(min g = {g[inner].min():.2e})"
                )
            if prev is not None and np.any(w - prev < -tol):
                raise ArbitrageError(f"{self.name}: calendar condition violated at T={T:g}")
            prev = w

    def __repr__(self) -> str:
        return f"PerturbedSurface({self.name} on {self.base!r})"


def saturated_k(k: ArrayLike, k_cap: float) -> FloatArray:
    """``k_cap · tanh(k / k_cap)``: the log-moneyness profile of the skew and curvature bumps —
    linear (unit slope) at the money, saturating smoothly at ``±k_cap`` in the wings (a kink
    would spike the butterfly density check)."""
    return np.asarray(k_cap * np.tanh(np.asarray(k, dtype=np.float64) / k_cap), dtype=np.float64)


def atm_skew_numeric(surface: ImpliedSurface, T: ArrayLike, h: float = 1e-3) -> FloatArray:
    """``∂σ/∂k`` at ``k = 0`` by central differences (any surface)."""
    return np.asarray(
        (surface.implied_vol_k(h, T) - surface.implied_vol_k(-h, T)) / (2.0 * h), dtype=np.float64
    )


def _bilinear(
    ks: FloatArray, ts: FloatArray, values: FloatArray, k: FloatArray, t: FloatArray
) -> FloatArray:
    """Bilinear interpolation of ``values[t, k]`` (flat outside the grid)."""
    ki = np.clip(np.searchsorted(ks, k) - 1, 0, max(ks.size - 2, 0))
    ti = np.clip(np.searchsorted(ts, t) - 1, 0, max(ts.size - 2, 0))
    if ks.size == 1:
        wk = np.zeros_like(k)
    else:
        wk = np.clip((k - ks[ki]) / (ks[ki + 1] - ks[ki]), 0.0, 1.0)
    if ts.size == 1:
        wt = np.zeros_like(t)
    else:
        wt = np.clip((t - ts[ti]) / (ts[ti + 1] - ts[ti]), 0.0, 1.0)
    ki1 = np.minimum(ki + 1, ks.size - 1)
    ti1 = np.minimum(ti + 1, ts.size - 1)
    v = (1 - wt) * ((1 - wk) * values[ti, ki] + wk * values[ti, ki1]) + wt * (
        (1 - wk) * values[ti1, ki] + wk * values[ti1, ki1]
    )
    return np.asarray(v, dtype=np.float64)


def delta_sigma_from_config(
    cfg: SurfacePerturbation, base: ImpliedSurface
) -> Callable[[FloatArray, FloatArray], FloatArray]:
    """The ``δσ(k, T)`` function of a perturbation config (kinds: :class:`SurfacePerturbation`)."""
    kind, p = cfg.kind, cfg.params
    if kind == "parallel":
        size = float(p["size"])
        return lambda k, T: np.full(np.broadcast(k, T).shape, size)
    if kind == "tent":
        pillars, i, size = tuple(p["pillars"]), int(p["index"]), float(p["size"])
        return lambda k, T: size * tent(T, pillars, i) * np.ones_like(k)
    if kind == "skew_tent":
        pillars, i, slope = tuple(p["pillars"]), int(p["index"]), float(p["slope"])
        k_cap = float(p.get("k_cap", 0.5))  # rotation around the money, saturating in the wings
        return lambda k, T: slope * saturated_k(k, k_cap) * tent(T, pillars, i)
    if kind == "rotation":
        # the desk's rotation (SPEC §15 Part 3, shadow-rotation greek): ``size`` rotas, one rota
        # being an ATM-skew move of ``2/sqrt(T)`` vol points per unit log-moneyness (``2/sqrt(T)
        # ln(110/90)`` vol points of 90/110 skew, 0.57 at 6M), the maturity floored at ``t_min``,
        # saturating beyond ``k_cap``; positive size steepens (puts up, calls down)
        size, t_min = float(p["size"]), float(p["t_min"])
        k_cap = float(p.get("k_cap", 0.5))
        if t_min <= 0:
            raise ValueError("rotation needs a positive t_min")

        def ds_rot(k: FloatArray, T: FloatArray) -> FloatArray:
            T_ = np.maximum(np.asarray(T, dtype=np.float64), t_min)
            return np.asarray(-size * 0.02 / np.sqrt(T_) * saturated_k(k, k_cap), dtype=np.float64)

        return ds_rot
    if kind == "curvature_tent":
        pillars, i, curv = tuple(p["pillars"]), int(p["index"]), float(p["curv"])
        k_cap = float(p.get("k_cap", 0.5))
        return lambda k, T: curv * saturated_k(k, k_cap) ** 2 * tent(T, pillars, i)
    if kind == "shift_k":
        delta = float(p["delta"])
        return lambda k, T: base.implied_vol_k(k + delta, T) - base.implied_vol_k(k, T)
    if kind == "atm_shift":
        # ``factor · delta · s_T`` for every k, with the ATM skew ``s_T`` evaluated at
        # ``max(T, t_min)``: the SSVI skew grows like ``1/√T`` towards zero maturity and an
        # unfloored shift drives the 1–5 day vols of the variance-swap strip to zero (found by
        # the M5 budget run, sticky-local-vol regime); ``t_min`` is the first vega pillar of the
        # regime (1m), an explicit parameter of the layer
        delta, factor = float(p["delta"]), float(p.get("factor", 1.0))
        t_min = float(p["t_min"])
        if t_min <= 0:
            raise ValueError("atm_shift needs a positive t_min (skew evaluation floor)")

        def ds_atm(k: FloatArray, T: FloatArray) -> FloatArray:
            T_ = np.maximum(np.asarray(T, dtype=np.float64), t_min)
            return np.asarray(
                factor * delta * atm_skew_numeric(base, T_) * np.ones_like(k), dtype=np.float64
            )

        return ds_atm
    if kind == "total_variance":
        from volsto.market.varswap import xi0_curve

        eps, t_lo, t_hi = float(p["eps"]), float(p["t_lo"]), float(p["t_hi"])
        xi0 = xi0_curve(base, min(base.max_maturity, max(t_hi + 1.0, 5.0)))

        def ds(k: FloatArray, T: FloatArray) -> FloatArray:
            T_ = np.asarray(T, dtype=np.float64)
            hi = np.minimum(T_, t_hi)
            dw = eps * np.where(hi > t_lo, xi0.integral(t_lo, np.maximum(hi, t_lo)), 0.0)
            w = base.total_variance(k, T_)
            return np.asarray(
                np.sqrt(np.maximum(w + dw, 1e-12) / T_) - np.sqrt(w / T_), dtype=np.float64
            )

        return ds
    if kind == "roll":
        dt = float(p["dt"])
        fc = base.forward_curve

        def ds_roll(k: FloatArray, T: FloatArray) -> FloatArray:
            T_ = np.asarray(T, dtype=np.float64)
            shift = np.asarray(fc.drift(T_, T_ + dt))  # ln F(T+dt) − ln F(T)
            return base.implied_vol_k(k - shift, T_ + dt) - base.implied_vol_k(k, T_)

        return ds_roll
    if kind == "table":
        ks = np.asarray(p["ks"], dtype=np.float64)
        ts = np.asarray(p["ts"], dtype=np.float64)
        values = np.asarray(p["values"], dtype=np.float64).reshape(ts.size, ks.size)
        return lambda k, T: _bilinear(
            ks, ts, values, np.asarray(k, dtype=np.float64), np.asarray(T, dtype=np.float64)
        )
    if kind == "composite":
        from volsto.config import SurfacePerturbation

        parts = [
            delta_sigma_from_config(SurfacePerturbation(**it) if isinstance(it, dict) else it, base)
            for it in p["items"]
        ]
        return lambda k, T: sum((f(k, T) for f in parts), np.zeros(np.broadcast(k, T).shape))
    raise ValueError(f"unknown perturbation kind {kind!r}")


def perturbed_surface(
    cfg: SurfacePerturbation | None, base: ImpliedSurface, *, check: bool = True
) -> ImpliedSurface:
    """``base`` itself when ``cfg`` is None, else the checked :class:`PerturbedSurface`."""
    if cfg is None:
        return base
    return PerturbedSurface(base, delta_sigma_from_config(cfg, base), cfg.kind, check=check)
