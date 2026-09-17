"""Implied-volatility surfaces (SPEC §2.2): ABC, Gatheral–Jacquier SSVI, market slice grid.

Conventions: ``k = ln(K / F(T))`` and ``w(k, T) = σ̂(k, T)² T`` (total implied variance).
Checked by ``tests/test_surface.py``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
        """Alias of :func:`surface_from_config` (the one construction site of a surface from a
        config): an eSSVI config gives an :class:`ESSVISurface`, never its flattened SSVI.
        Library code calls the factory directly (``tests/test_surface_config.py`` walks the
        package)."""
        return surface_from_config(cfg, forward_curve, discount)

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


CALENDAR_GRID_N_T: int = 200
"""Maturities from ``min_maturity`` to the last pillar in the eSSVI constructor's calendar grid
(M3b)."""
CALENDAR_SEGMENT_N: int = 48
"""Maturities per knot segment, both ends included, of the repair's constraint grid and of the
initial partition of :func:`certify_calendar` (M10 Part 0: 7 segments of 48 points match the
200-point grid's density on the 1m-2y segments and are denser on the short ones)."""
CERT_N_K: int = 121
"""Log-moneyness points of the initial partition of :func:`certify_calendar` (step 0.05 on
``±3``: the repair's ``k`` grid, so the cell corners are its constraint points)."""
CERT_MAX_LEVELS: int = 60
"""Bisection levels of :func:`certify_calendar` before it gives up (``inconclusive``); a cell
edge halves per level, so 60 levels reach below ``1e-19`` of the initial cell."""
CERT_MAX_CELLS: int = 4_000_000
"""Cells :func:`certify_calendar` may examine in total before it gives up (``inconclusive``;
bounds memory and time, never turned into a pass)."""
CERT_MAX_POINTS: int = 64
"""Largest number of violating points :func:`certify_calendar` returns (lowest first)."""
DENSE_CHECK_K_ABS: float = 3.0
"""Half-width in ``k`` of :meth:`ESSVISurface.calendar_dense_check`: the Dupire range."""
DENSE_CHECK_N_K: int = 1201
"""Log-moneyness points of the dense check (step 0.005; the M10 Part 0 verifier's grid)."""
DENSE_CHECK_N_T: int = 3000
"""Maturities of the dense check on ``[min_maturity, max_maturity]`` (the verifier's grid), plus
every knot, evaluated with the slopes of both adjacent segments."""


def calendar_t_grid(
    min_maturity: float,
    pillars: Sequence[float] | FloatArray,
    max_maturity: float,
    n_t: int = CALENDAR_GRID_N_T,
) -> FloatArray:
    """Maturity grid of the eSSVI constructor's calendar check:
    ``unique(linspace(min_maturity, last_pillar, n_t) ∪ pillars ∪ {max_maturity})``.

    The pillars are included (M10 Part 0 fix): ``θ_T`` and ``ρ_T`` kink there, and ``∂_T w`` is
    evaluated there with both one-sided slopes.  The ``unique`` removes the zero-length step the
    M3b grid had when the last pillar equals ``max_maturity`` (3y on seven-pillar days).  Checked
    by ``test_essvi_calendar_grid_dedup`` and ``test_essvi_calendar_grid_sees_pillars``."""
    pil = np.atleast_1d(np.asarray(pillars, dtype=np.float64))
    return np.unique(
        np.concatenate(
            (np.linspace(min_maturity, float(pil[-1]), int(n_t)), pil, [float(max_maturity)])
        )
    ).astype(np.float64)


def calendar_knots(
    min_maturity: float, pillars: Sequence[float] | FloatArray, max_maturity: float
) -> FloatArray:
    """``{min_maturity} ∪ {pillars strictly inside} ∪ {max_maturity}``: the maturities between
    which ``θ_T`` and ``ρ_T`` are affine, so ``w(k, T)`` is smooth on every knot segment and
    ``∂_T w`` may jump only at a knot."""
    pil = np.atleast_1d(np.asarray(pillars, dtype=np.float64))
    inside = pil[(pil > min_maturity) & (pil < max_maturity)]
    return np.unique(np.concatenate(([float(min_maturity)], inside, [float(max_maturity)])))


def ssvi_theta(
    pillars: Sequence[float] | FloatArray,
    theta_p: Sequence[float] | FloatArray,
    T: ArrayLike,
) -> FloatArray:
    """``θ_T`` exactly as :meth:`SSVISurface.theta` evaluates it: linear from ``θ_0 = 0``
    through the pillars, the last forward variance extended beyond the last pillar."""
    t = np.concatenate(([0.0], np.asarray(pillars, dtype=np.float64)))
    th = np.concatenate(([0.0], np.asarray(theta_p, dtype=np.float64)))
    T_ = np.asarray(T, dtype=np.float64)
    slope = (th[-1] - th[-2]) / (t[-1] - t[-2])
    beyond = th[-1] + slope * (T_ - t[-1])
    return np.asarray(np.where(t[-1] < T_, beyond, np.interp(T_, t, th)), dtype=np.float64)


def essvi_dw_dt(
    k: ArrayLike,
    theta: ArrayLike,
    dtheta: ArrayLike,
    rho: ArrayLike,
    drho: ArrayLike,
    eta: float,
    gamma: float,
) -> FloatArray:
    r"""Exact ``∂_T w`` of eSSVI where ``θ_T`` and ``ρ_T`` are affine in ``T`` with slopes
    ``dtheta`` and ``drho`` (inside a knot segment; at a knot, the one-sided limit on the side
    whose slopes are passed).

    With ``x = kφ(θ)``, ``S = sqrt((x+ρ)² + 1 − ρ²)``, ``g = 1 + ρx + S`` (so ``w = θg/2``) and
    ``χ = θ|φ'/φ| = γ + (1−γ)θ/(1+θ)``, the chain rule ``∂_T w = θ' ∂_θ w + ρ' ∂_ρ w`` (``φ``
    depends on ``T`` through ``θ``: ``∂_T x = −x θ' χ / θ``) collapses to

    .. math::
        \partial_T w = \tfrac{\theta'}{2}\, g \Big(1 - \chi + \frac{\chi}{S}\Big)
                     + \tfrac{\theta\rho'}{2}\, x \Big(1 + \frac{1}{S}\Big),

    using ``S² − x² − ρx = 1 + ρx``.  The first term is positive whenever ``θ' > 0`` (SSVI with
    one ``ρ`` is calendar-free); only ``ρ' x < 0`` can make ``∂_T w`` negative.  Checked against
    finite differences by ``tests/test_surface.py::test_essvi_dw_dt_is_the_derivative``."""
    k_ = np.asarray(k, dtype=np.float64)
    th = np.asarray(theta, dtype=np.float64)
    r = np.asarray(rho, dtype=np.float64)
    phi = eta / (th**gamma * (1.0 + th) ** (1.0 - gamma))
    x = phi * k_
    s = np.sqrt((x + r) ** 2 + 1.0 - r * r)
    g = 1.0 + r * x + s
    chi = gamma + (1.0 - gamma) * th / (1.0 + th)
    out = 0.5 * np.asarray(dtheta) * g * (1.0 - chi + chi / s) + 0.5 * th * np.asarray(drho) * x * (
        1.0 + 1.0 / s
    )
    return np.asarray(out, dtype=np.float64)


@dataclass(frozen=True)
class CalendarSegments:
    """The knot segments ``[a_i, b_i]`` of an eSSVI surface with ``θ_T = θ_a + θ'(T − a)`` and
    ``ρ_T = ρ_a + ρ'(T − a)`` on each (see :func:`calendar_knots`)."""

    a: FloatArray
    b: FloatArray
    theta_a: FloatArray
    dtheta: FloatArray
    rho_a: FloatArray
    drho: FloatArray

    @classmethod
    def from_knots(
        cls, knots: FloatArray, theta_knots: FloatArray, rho_knots: FloatArray
    ) -> CalendarSegments:
        """Segments between consecutive knots from ``θ`` and ``ρ`` at the knots."""
        kn = np.asarray(knots, dtype=np.float64)
        th = np.asarray(theta_knots, dtype=np.float64)
        r = np.asarray(rho_knots, dtype=np.float64)
        dt = np.diff(kn)
        return cls(kn[:-1], kn[1:], th[:-1], np.diff(th) / dt, r[:-1], np.diff(r) / dt)

    @property
    def n(self) -> int:
        return int(self.a.size)

    def theta(self, seg: NDArray[np.intp], T: FloatArray) -> FloatArray:
        return np.asarray(self.theta_a[seg] + self.dtheta[seg] * (T - self.a[seg]))

    def rho(self, seg: NDArray[np.intp], T: FloatArray) -> FloatArray:
        return np.asarray(self.rho_a[seg] + self.drho[seg] * (T - self.a[seg]))

    def points(self, Ts: FloatArray) -> tuple[NDArray[np.intp], FloatArray]:
        """``(seg, T)`` for every maturity of ``Ts`` inside ``[a_i, b_i]``, per segment; a knot
        appears once for each segment it bounds (both one-sided limits)."""
        segs, ts = [], []
        for i in range(self.n):
            m = Ts[(Ts >= self.a[i]) & (Ts <= self.b[i])]
            segs.append(np.full(m.size, i, dtype=np.intp))
            ts.append(m)
        return np.concatenate(segs), np.concatenate(ts).astype(np.float64)

    def segment_grid(self, n_t: int) -> tuple[NDArray[np.intp], FloatArray]:
        """``(seg, T)`` of ``linspace(a_i, b_i, n_t)`` on every segment (both ends, so both
        one-sided limits at every knot)."""
        segs = np.repeat(np.arange(self.n, dtype=np.intp), int(n_t))
        u = np.linspace(0.0, 1.0, int(n_t))
        ts = (self.a[:, None] + (self.b - self.a)[:, None] * u[None, :]).ravel()
        ts[int(n_t) - 1 :: int(n_t)] = self.b  # the right end exactly on the knot
        return segs, ts

    def dw_dt(
        self, k: ArrayLike, seg: NDArray[np.intp], T: FloatArray, eta: float, gamma: float
    ) -> FloatArray:
        """Exact ``∂_T w`` at ``(k, T)`` with the slopes of segment ``seg`` (broadcast)."""
        return essvi_dw_dt(
            k,
            self.theta(seg, T),
            self.dtheta[seg],
            self.rho(seg, T),
            self.drho[seg],
            eta,
            gamma,
        )


def essvi_segments(
    pillars: Sequence[float] | FloatArray,
    theta_p: Sequence[float] | FloatArray,
    rhos: Sequence[float] | FloatArray,
    min_maturity: float,
    max_maturity: float,
) -> CalendarSegments:
    """:class:`CalendarSegments` of the eSSVI surface with these pillar parameters (``θ`` as
    :func:`ssvi_theta`, ``ρ_T = interp(T, pillars, rhos)``, flat outside the pillars)."""
    kn = calendar_knots(min_maturity, pillars, max_maturity)
    pil = np.asarray(pillars, dtype=np.float64)
    return CalendarSegments.from_knots(
        kn, ssvi_theta(pil, theta_p, kn), np.interp(kn, pil, np.asarray(rhos, dtype=np.float64))
    )


def _imul(
    al: FloatArray, ah: FloatArray, bl: FloatArray, bh: FloatArray
) -> tuple[FloatArray, FloatArray]:
    """Interval product ``[al, ah] × [bl, bh]``."""
    p = np.stack([al * bl, al * bh, ah * bl, ah * bh])
    return np.asarray(p.min(axis=0)), np.asarray(p.max(axis=0))


def dw_dt_lower_bound(
    segs: CalendarSegments,
    seg: NDArray[np.intp],
    T0: FloatArray,
    T1: FloatArray,
    k0: FloatArray,
    k1: FloatArray,
    eta: float,
    gamma: float,
) -> FloatArray:
    r"""A lower bound of the exact ``∂_T w`` (:func:`essvi_dw_dt`) over each cell
    ``[T0, T1] × [k0, k1]`` inside segment ``seg`` (closed: both knot limits included).

    Every factor's range over the cell is exact, from monotonicity (``η ≥ 0``,
    ``0 < γ ≤ 1``): ``θ`` and ``ρ`` are affine in ``T``; ``φ`` decreases in ``θ``, so
    ``x = kφ`` spans the four products of the ends; ``χ`` increases in ``θ``; ``S² = x² + 2ρx + 1``
    is linear in ``ρ`` and convex in ``x`` (minimum at ``x = −ρ``, clipped); ``g = 1 + ρx + S``
    is monotone in ``ρ`` (``∂_ρ g = x(1 + 1/S)``) and convex in ``x`` (minimum at ``x = −2ρ``,
    clipped).  The two terms are then combined by interval arithmetic, treating the factors as
    independent, which can only widen the range: the result is a valid lower bound (up to
    floating-point rounding, ~1e-16 relative), and it converges to the point value at rate
    O(cell size) as the cell shrinks — which is what lets :func:`certify_calendar` terminate."""
    dth = segs.dtheta[seg]
    th0 = segs.theta_a[seg] + dth * (T0 - segs.a[seg])
    th1 = segs.theta_a[seg] + dth * (T1 - segs.a[seg])
    thl, thh = np.minimum(th0, th1), np.maximum(th0, th1)
    drh = segs.drho[seg]
    r0 = segs.rho_a[seg] + drh * (T0 - segs.a[seg])
    r1 = segs.rho_a[seg] + drh * (T1 - segs.a[seg])
    rl, rh = np.minimum(r0, r1), np.maximum(r0, r1)

    def phi(t: FloatArray) -> FloatArray:
        return np.asarray(eta / (t**gamma * (1.0 + t) ** (1.0 - gamma)))

    def chi(t: FloatArray) -> FloatArray:
        return np.asarray(gamma + (1.0 - gamma) * t / (1.0 + t))

    xl, xh = _imul(k0, k1, phi(thh), phi(thl))

    def q(x: FloatArray, r: FloatArray) -> FloatArray:
        return np.asarray((x + r) ** 2 + 1.0 - r * r)

    corners = [(xl, rl), (xl, rh), (xh, rl), (xh, rh)]
    q_lo = np.minimum(q(np.clip(-rl, xl, xh), rl), q(np.clip(-rh, xl, xh), rh))
    q_hi = np.max(np.stack([q(x, r) for x, r in corners]), axis=0)
    s_lo, s_hi = np.sqrt(q_lo), np.sqrt(q_hi)

    def gf(x: FloatArray, r: FloatArray) -> FloatArray:
        return np.asarray(1.0 + r * x + np.sqrt(q(x, r)))

    g_lo = np.minimum(gf(np.clip(-2.0 * rl, xl, xh), rl), gf(np.clip(-2.0 * rh, xl, xh), rh))
    g_hi = np.max(np.stack([gf(x, r) for x, r in corners]), axis=0)
    # h1 = 1 − χ (1 − 1/S)
    cu_lo, cu_hi = _imul(chi(thl), chi(thh), 1.0 - 1.0 / s_lo, 1.0 - 1.0 / s_hi)
    gh_lo, gh_hi = _imul(g_lo, g_hi, 1.0 - cu_hi, 1.0 - cu_lo)
    t1_lo, _ = _imul(0.5 * dth, 0.5 * dth, gh_lo, gh_hi)
    # x (1 + 1/S)
    b_lo, b_hi = _imul(xl, xh, 1.0 + 1.0 / s_hi, 1.0 + 1.0 / s_lo)
    tb_lo, tb_hi = _imul(thl, thh, b_lo, b_hi)
    t2_lo, _ = _imul(0.5 * drh, 0.5 * drh, tb_lo, tb_hi)
    return np.asarray(t1_lo + t2_lo, dtype=np.float64)


@dataclass(frozen=True)
class CalendarCertificate:
    """Outcome of :func:`certify_calendar` (``∂_T w`` per year).

    ``status`` is ``"certified"`` (``∂_T w ≥ floor`` proven on the whole range, knot limits
    included; ``lower_bound`` is the smallest cell bound that proved it), ``"violated"`` (points
    with ``∂_T w < stop_below`` were found: ``cut_*``, lowest first) or ``"inconclusive"`` (the
    level or cell budget ran out: never a pass).  ``min_value`` / ``min_k`` / ``min_T`` /
    ``min_seg`` locate the smallest exact ``∂_T w`` evaluated (grid points and cell centres) —
    an upper bound of the true minimum."""

    status: str
    floor: float
    stop_below: float
    k_abs: float
    lower_bound: float
    min_value: float
    min_k: float
    min_T: float
    min_seg: int
    cut_k: FloatArray
    cut_T: FloatArray
    cut_seg: NDArray[np.intp]
    levels: int
    n_cells: int

    @property
    def certified(self) -> bool:
        return self.status == "certified"


def certify_calendar(
    segs: CalendarSegments,
    eta: float,
    gamma: float,
    *,
    k_abs: float,
    floor: float,
    stop_below: float,
    n_k: int = CERT_N_K,
    n_t: int = CALENDAR_SEGMENT_N,
    max_levels: int = CERT_MAX_LEVELS,
    max_cells: int = CERT_MAX_CELLS,
    max_points: int = CERT_MAX_POINTS,
) -> CalendarCertificate:
    """Prove ``∂_T w(k, T) ≥ floor`` for every ``|k| ≤ k_abs`` and every ``T`` of every knot
    segment, both one-sided limits at every knot included — or find points below
    ``stop_below`` (M10 Part 0, third pass: the invariant itself, not a grid).

    Branch and bound on the exact derivative: the exact ``∂_T w`` is evaluated on
    ``linspace(−k_abs, k_abs, n_k)`` × ``linspace(a_i, b_i, n_t)`` of every segment (with that
    segment's slopes, so a knot is evaluated from both sides); the cells between these points
    get the lower bound of :func:`dw_dt_lower_bound`; a cell whose bound is ``≥ floor`` is
    proven, every other cell is bisected in ``T`` and ``k`` and its centre evaluated exactly.
    It stops when every cell is proven (``certified``), when an evaluated point falls below
    ``stop_below`` (``violated``), or when ``max_levels`` / ``max_cells`` run out
    (``inconclusive``).  With ``stop_below ≥ floor`` the outcome is decided for every surface
    except those whose minimum lies in ``[floor, stop_below)`` up to the budget: a true minimum
    above ``floor`` is eventually proven (the bound converges to the point value), one below
    ``stop_below`` eventually sampled.  Checked by ``tests/test_surface.py`` and
    ``tests/test_calendar_repair.py``."""
    ks = np.linspace(-float(k_abs), float(k_abs), int(n_k))
    gseg, gT = segs.segment_grid(int(n_t))
    D = segs.dw_dt(ks[None, :], gseg[:, None], gT[:, None], eta, gamma)
    i, j = np.unravel_index(int(np.argmin(D)), D.shape)
    best = (float(D[i, j]), float(ks[j]), float(gT[i]), int(gseg[i]))
    n_cells = 0
    lb_min = np.inf
    empty_f = np.empty(0, dtype=np.float64)
    empty_i = np.empty(0, dtype=np.intp)

    def done(
        status: str, level: int, cut: tuple[FloatArray, FloatArray, NDArray[np.intp]] | None
    ) -> CalendarCertificate:
        ck, cT, cs = cut if cut is not None else (empty_f, empty_f, empty_i)
        return CalendarCertificate(
            status=status,
            floor=float(floor),
            stop_below=float(stop_below),
            k_abs=float(k_abs),
            lower_bound=float(lb_min) if status == "certified" else float("nan"),
            min_value=best[0],
            min_k=best[1],
            min_T=best[2],
            min_seg=best[3],
            cut_k=ck,
            cut_T=cT,
            cut_seg=cs,
            levels=level,
            n_cells=n_cells,
        )

    def lowest(
        vals: FloatArray, kk: FloatArray, tt: FloatArray, ss: NDArray[np.intp]
    ) -> tuple[FloatArray, FloatArray, NDArray[np.intp]]:
        bad = np.flatnonzero(vals < stop_below)
        order = bad[np.argsort(vals[bad], kind="stable")][:max_points]
        return kk[order], tt[order], ss[order]

    if best[0] < stop_below:
        KK, TT = np.broadcast_to(ks[None, :], D.shape), np.broadcast_to(gT[:, None], D.shape)
        SS = np.broadcast_to(gseg[:, None], D.shape)
        return done("violated", 0, lowest(D.ravel(), KK.ravel(), TT.ravel(), SS.ravel()))
    # level-0 cells: between consecutive grid points of each segment × consecutive ks
    nt = int(n_t)
    ti = np.arange(segs.n * nt).reshape(segs.n, nt)
    lo_idx, hi_idx = ti[:, :-1].ravel(), ti[:, 1:].ravel()
    nkc = ks.size - 1
    c_seg = np.repeat(gseg[lo_idx], nkc)
    c_T0 = np.repeat(gT[lo_idx], nkc)
    c_T1 = np.repeat(gT[hi_idx], nkc)
    c_k0 = np.tile(ks[:-1], lo_idx.size)
    c_k1 = np.tile(ks[1:], lo_idx.size)
    for level in range(int(max_levels)):
        n_cells += c_seg.size
        lb = dw_dt_lower_bound(segs, c_seg, c_T0, c_T1, c_k0, c_k1, eta, gamma)
        proven = lb >= floor
        if np.any(proven):
            lb_min = min(lb_min, float(np.min(lb[proven])))
        keep = ~proven
        if not np.any(keep):
            return done("certified", level, None)
        c_seg, c_T0, c_T1 = c_seg[keep], c_T0[keep], c_T1[keep]
        c_k0, c_k1 = c_k0[keep], c_k1[keep]
        if n_cells + 4 * c_seg.size > max_cells:
            return done("inconclusive", level, None)
        Tm, km = 0.5 * (c_T0 + c_T1), 0.5 * (c_k0 + c_k1)
        v = segs.dw_dt(km, c_seg, Tm, eta, gamma)
        m = int(np.argmin(v))
        if float(v[m]) < best[0]:
            best = (float(v[m]), float(km[m]), float(Tm[m]), int(c_seg[m]))
        if best[0] < stop_below:
            return done("violated", level + 1, lowest(v, km, Tm, c_seg))
        c_seg = np.tile(c_seg, 4)
        c_T0, c_T1 = np.concatenate([c_T0, Tm, c_T0, Tm]), np.concatenate([Tm, c_T1, Tm, c_T1])
        c_k0, c_k1 = np.concatenate([c_k0, c_k0, km, km]), np.concatenate([km, km, c_k1, c_k1])
    return done("inconclusive", int(max_levels), None)


@dataclass(frozen=True)
class CalendarDenseCheck:
    """Outcome of :meth:`ESSVISurface.calendar_dense_check` (``∂_T w`` per year).

    All derivative fields use the exact ``∂_T w`` (:func:`essvi_dw_dt`): ``min_interior`` over
    the dense maturities strictly inside a knot segment, ``min_left`` / ``min_right`` the
    one-sided limits at every interior knot (a pillar), each over every ``k``; ``worst_*``
    locate the smallest of the three.  ``min_slope`` (secant ``Δw/ΔT``) and ``min_dw`` (step
    ``Δw``) are the verifier's finite-difference metrics on the same grid, kept as an
    independent cross-check.  ``certificate`` is :func:`certify_calendar` with floor 0 on the
    same ``k`` range: ``ok`` requires it to prove ``∂_T w ≥ 0`` everywhere."""

    min_interior: float
    min_left: float
    min_right: float
    min_slope: float
    min_dw: float
    worst_k: float
    worst_T: float
    k_abs: float
    n_k: int
    n_t: int
    certificate: CalendarCertificate

    @property
    def min_dw_dt(self) -> float:
        """The smallest exact ``∂_T w`` on the dense grid, knot limits included (per year)."""
        return min(self.min_interior, self.min_left, self.min_right)

    @property
    def ok(self) -> bool:
        """``∂_T w ≥ 0`` proven on the whole range (and seen on the dense grid)."""
        return self.certificate.certified and self.min_dw_dt >= 0.0


def calendar_grid_dw_dt(
    segs: CalendarSegments, eta: float, gamma: float, ks: FloatArray, Ts: FloatArray
) -> tuple[FloatArray, NDArray[np.intp], FloatArray]:
    """Exact ``∂_T w`` on ``ks`` × every maturity of ``Ts``, per segment (knots from both
    sides): returns ``(D, seg, T)`` with ``D`` of shape ``(n_points, ks.size)``."""
    seg, T = segs.points(Ts)
    return segs.dw_dt(ks[None, :], seg[:, None], T[:, None], eta, gamma), seg, T


class ESSVISurface(SSVISurface):
    """SSVI with a maturity-dependent correlation ``ρ_T`` (Hendriks–Martini eSSVI family).

    ``ρ_T`` is piecewise-linear in ``T`` between the ATM pillars (flat outside).  Butterfly
    conditions are checked with ``max|ρ|`` (sufficient); calendar-spread absence is checked
    on the exact derivative ``∂_T w`` (:func:`essvi_dw_dt`), since the analytic eSSVI conditions
    couple ``ρ_T`` and ``θ_T``.  Intended for later single-stock use.

    The constructor's calendar check (:meth:`calendar_min_dw_dt` below ``−CALENDAR_TOL``) runs
    on ``k ∈ [−CALENDAR_K_ABS, CALENDAR_K_ABS]`` at the maturities of :func:`calendar_t_grid`,
    every knot from both sides; the importer's repair
    (:func:`volsto.market.import_hdn.repair_calendar`) proves ``∂_T w ≥ margin`` on the Dupire
    range ``±3`` with :func:`certify_calendar`, and :meth:`calendar_dense_check` reports it.
    Checked by ``tests/test_surface.py`` and ``tests/test_calendar_repair.py``.
    """

    CALENDAR_K_ABS: float = 1.0
    """Half-width in ``k`` of the constructor's calendar check (unchanged since M3b; the Dupire
    range ``±3`` is enforced by the importer's repair and reported by the M10 gate)."""
    CALENDAR_N_K: int = 81
    """Log-moneyness points of the constructor's calendar check."""
    CALENDAR_N_T: int = CALENDAR_GRID_N_T
    """Maturities from ``min_maturity`` to the last pillar in the calendar check (plus every
    pillar and ``max_maturity``)."""
    CALENDAR_TOL: float = 1e-10
    """Most negative exact ``∂_T w`` (per year) the constructor tolerates as round-off."""

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

    @property
    def pillars(self) -> FloatArray:
        """The ATM pillar maturities (where ``θ_T`` and ``ρ_T`` kink)."""
        return self._rho_t.copy()

    def calendar_segments(self) -> CalendarSegments:
        """The knot segments of this surface (``θ_T`` and ``ρ_T`` affine on each)."""
        kn = calendar_knots(self.min_maturity, self._rho_t, self.max_maturity)
        return CalendarSegments.from_knots(kn, self.theta(kn), self.rho_T(kn))

    def dw_dT(self, k: ArrayLike, T: ArrayLike, side: str = "right") -> FloatArray:
        """Exact ``∂_T w(k, T)``; at a knot, the ``"left"`` or ``"right"`` limit.  Checked by
        ``test_essvi_dw_dt_is_the_derivative``."""
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        segs = self.calendar_segments()
        T_ = np.asarray(T, dtype=np.float64)
        if side == "left":
            seg = np.searchsorted(segs.b, T_, side="left")
        else:
            seg = np.searchsorted(segs.a, T_, side="right") - 1
        seg = np.clip(seg, 0, segs.n - 1).astype(np.intp)
        return segs.dw_dt(k, seg, T_, self.eta, self.gamma)

    def calendar_grid(
        self, k_abs: float | None = None, n_k: int | None = None
    ) -> tuple[FloatArray, FloatArray]:
        """The calendar-check grid ``(ks, Ts)``: ``ks = linspace(−k_abs, k_abs, n_k)`` (defaults
        ``CALENDAR_K_ABS``, ``CALENDAR_N_K``) and the maturities of :func:`calendar_t_grid`
        (every pillar included, no zero-length step).  Checked by
        ``test_essvi_calendar_grid_dedup`` and ``test_essvi_calendar_grid_sees_pillars``."""
        ka = self.CALENDAR_K_ABS if k_abs is None else float(k_abs)
        nk = self.CALENDAR_N_K if n_k is None else int(n_k)
        ks = np.linspace(-ka, ka, nk)
        Ts = calendar_t_grid(self.min_maturity, self._rho_t, self.max_maturity, self.CALENDAR_N_T)
        return ks, Ts

    def calendar_min_dw_dt(self, k_abs: float | None = None, n_k: int | None = None) -> float:
        """The constructor's calendar quantity: the smallest exact ``∂_T w`` (per year) over
        :meth:`calendar_grid`, every knot evaluated with the slopes of both adjacent segments
        (negative = calendar arbitrage).  Returns instead of raising.  Checked by
        ``test_essvi_calendar_min_dw_dt_matches_check``."""
        ks, Ts = self.calendar_grid(k_abs, n_k)
        D, _, _ = calendar_grid_dw_dt(self.calendar_segments(), self.eta, self.gamma, ks, Ts)
        return float(np.min(D))

    def calendar_min_dw(self, k_abs: float | None = None, n_k: int | None = None) -> float:
        """The M3b-era slack, for the record only: the smallest step
        ``w(k, T_{i+1}) − w(k, T_i)`` over :meth:`calendar_grid` (the constructor no longer
        decides on it)."""
        ks, Ts = self.calendar_grid(k_abs, n_k)
        w = self.total_variance(ks[None, :], Ts[:, None])
        return float(np.min(np.diff(w, axis=0)))

    def calendar_certificate(
        self, k_abs: float = DENSE_CHECK_K_ABS, floor: float = 0.0
    ) -> CalendarCertificate:
        """:func:`certify_calendar` of this surface: prove ``∂_T w ≥ floor`` on ``|k| ≤ k_abs``
        and every ``T`` in ``[min_maturity, max_maturity]`` (stops at the first point below
        ``floor``)."""
        return certify_calendar(
            self.calendar_segments(),
            self.eta,
            self.gamma,
            k_abs=k_abs,
            floor=floor,
            stop_below=floor,
        )

    def calendar_dense_check(
        self,
        k_abs: float = DENSE_CHECK_K_ABS,
        n_k: int = DENSE_CHECK_N_K,
        n_t: int = DENSE_CHECK_N_T,
    ) -> CalendarDenseCheck:
        """Verification of ``∂_T w ≥ 0`` (M10 Part 0): the exact derivative on
        ``linspace(−k_abs, k_abs, n_k)`` × ``linspace(min_maturity, max_maturity, n_t)`` plus
        every knot from both sides, the verifier's secant and step metrics on the same grid, and
        the certificate of :func:`certify_calendar` (floor 0) on the same ``k`` range.  Reports,
        never raises.  Checked by ``tests/test_surface.py`` and
        ``tests/test_calendar_repair.py``; run on every sample day by
        ``scripts/essvi_calendar_gate.py``."""
        ks = np.linspace(-float(k_abs), float(k_abs), int(n_k))
        segs = self.calendar_segments()
        knots = np.concatenate((segs.a, segs.b[-1:]))
        Ts = np.union1d(np.linspace(self.min_maturity, self.max_maturity, int(n_t)), knots)
        D, seg, tp = calendar_grid_dw_dt(segs, self.eta, self.gamma, ks, Ts)
        at_left_end = tp == segs.a[seg]
        at_right_end = tp == segs.b[seg]
        interior = ~(at_left_end | at_right_end)
        right = at_left_end & (seg > 0)  # right limit at an interior knot
        left = at_right_end & (seg < segs.n - 1)  # left limit at an interior knot
        ends = (at_left_end & (seg == 0)) | (at_right_end & (seg == segs.n - 1))

        def rowmin(mask: NDArray[np.bool_]) -> float:
            return float(np.min(D[mask])) if np.any(mask) else float("inf")

        # the range ends (min_maturity, max_maturity) count with the interior
        min_interior = rowmin(interior | ends)
        i, j = np.unravel_index(int(np.argmin(D)), D.shape)
        w = self.total_variance(ks[None, :], Ts[:, None])
        dw = np.diff(w, axis=0)
        return CalendarDenseCheck(
            min_interior=min_interior,
            min_left=rowmin(left),
            min_right=rowmin(right),
            min_slope=float(np.min(dw / np.diff(Ts)[:, None])),
            min_dw=float(np.min(dw)),
            worst_k=float(ks[j]),
            worst_T=float(tp[i]),
            k_abs=float(k_abs),
            n_k=int(n_k),
            n_t=int(n_t),
            certificate=self.calendar_certificate(float(k_abs), 0.0),
        )

    def _check_calendar_numeric(self) -> None:
        if self.calendar_min_dw_dt() < -self.CALENDAR_TOL:
            raise ValueError("eSSVI calendar arbitrage: total variance decreases in T for some k")

    def __repr__(self) -> str:
        return (
            f"ESSVISurface(atm_maturities={self._t[1:].tolist()}, "
            f"rhos={np.round(self._rhos, 4).tolist()}, eta={self.eta}, gamma={self.gamma})"
        )


def surface_from_config(
    cfg: SSVIConfig, forward_curve: ForwardCurve, discount: DiscountCurve
) -> SSVISurface:
    """The implied surface of a surface config — **the one place a surface is built from a
    config** (M10 Part 3, SPEC §13.2).

    ``θ_i = σ_ATM,i² T_i`` at the pillars of ``cfg``; with ``cfg.rhos`` set the result is the
    :class:`ESSVISurface` with ``ρ_T`` through the pillar correlations (its constructor re-runs
    the calendar check on the exact ``∂_T w``), otherwise the plain :class:`SSVISurface` with the
    scalar ``cfg.rho``.  The leverage cache's market (:func:`volsto.calibration.cache.
    build_market`), the risk engine's states (:func:`volsto.risk.engine.surface_of`) and the
    snapshot loader (:func:`volsto.market.loaders.load_ssvi_surface`) all build through here, so
    an eSSVI snapshot is never flattened to one ``ρ`` on the way to a leverage calibration or a
    risk bump.  ``tests/test_surface_config.py::test_every_surface_is_built_by_the_factory``
    walks ``volsto/`` and ``scripts/`` for any other construction from a config;
    ``test_calendar_certificate_survives_the_config_path`` re-proves the importer's calendar
    certificate (SPEC §13.1) on the surface rebuilt here."""
    t = np.asarray(cfg.atm_maturities)
    v = np.asarray(cfg.atm_vols)
    if cfg.rhos is None:
        return SSVISurface(
            t,
            v * v * t,
            cfg.rho,
            cfg.eta,
            cfg.gamma,
            forward_curve,
            discount,
            max_maturity=cfg.max_maturity,
        )
    return ESSVISurface(
        t,
        v * v * t,
        cfg.rhos,
        cfg.eta,
        cfg.gamma,
        forward_curve,
        discount,
        max_maturity=cfg.max_maturity,
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
