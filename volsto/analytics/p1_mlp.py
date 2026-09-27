"""The desk's P1 break-evens by the most-likely-path closed forms (the owner's note
*P1ParametersCalibration*, section 4 and appendices C–D; SPEC §15 Part 3, *The note's engine*).
Not wired into the marking fit: an engine to compare with volsto's first-order one
(:class:`volsto.calibration.fit_2f.P1Maps`) and with the stage-3 simulations.

**The approximation (the note's eq. 24).** The ATMF implied variance of the P1LV model is

    σ̂_T² ≈ (1/T) ∫₀ᵀ E[V_t | S̃_T = S_0] dt,

``S̃`` the Black–Scholes process on the ATMF forward variance ``ξ̂ = ∂_t Q̂``, ``Q̂(t) = σ̂_t² t``
(eqs. 20–22), ``V`` the instantaneous variance.

**SensiX, SensiY (§4.1.1, appendices C–D).** ``V`` is replaced by ``Ṽ_t = ξ_t^t σ̃²(t, S̃_t)``
(eq. 26), the decoration proxy matching the market's corridor variances (eqs. 25, 27)::

    σ̃²(t, K) = φ^market(t, K) / E[ξ_t^t 1{S̃_t = K}],
    E[ξ_t^t 1{S̃_t = K}] = N′(d̃_t(K)) / (K √Q̂(t)),   d̃_t(K) = (ln(K/S_0) + ½ Q̂(t) − c(t)) / √Q̂(t),
    c(t) = λ1 ∫₀ᵗ √ξ̂(s) e^{−k1(t−s)} ds + λ2 ∫₀ᵗ √ξ̂(s) e^{−k2(t−s)} ds                (eq. 28)

with ``φ^market(t, K) = E[V_t 1{S_t = K}] = N′(d^mkt) ∂_t w^mkt / (K √w^mkt)``, ``w^mkt = t σ̂(t,
K)²``, ``d^mkt = ln(K/S_0)/√w^mkt + ½ √w^mkt`` — the Dupire identity for the driftless process in
the note's notation (the line defining ``φ^market`` is cut off in the copy seen: to confirm).
With ``X0``, ``Y0`` the factors' initial values (eq. 33)::

    σ̂_T(X0, Y0)² = (1/T) ∫₀ᵀ a(t) f(t, T) dt,   a(t) = exp(ω1 X0 e^{−k1 t} + ω2 Y0 e^{−k2 t}),
    f(t, T) = η(t, T) E_u[σ̃²(t, S_0 e^{μ(t,T) + γ(t,T) u})],   u ~ N(0, 1),
    η = exp(½ c(t) − ½ c(t)²/Q̂(T)),   μ = (1 − Q̂(t)/Q̂(T)) c(t),   γ = √(Q̂(t) − Q̂(t)²/Q̂(T)),
    Q̂(t) = ∫₀ᵗ a ξ̂,   c(t) = Σ_i λ_i ∫₀ᵗ √(a ξ̂)(s) e^{−k_i(t−s)} ds,

the decoration ``σ̃²`` fixed at ``X0 = Y0 = 0``; ``SensiX = ∂_{X0} σ̂_T² / (2 σ̂_T(0, 0)²)``
(eq. 34), ``SensiY`` likewise in ``Y0`` (eq. 35).  The derivative is eq. 57 (after the note's
integration by parts in ``u``)::

    ∂σ̂_T² = (1/T) ∫₀ᵀ [∂a η I0 + ∂η I0 + (η/γ) ∂μ I1 + (η/γ) ∂γ I2] dt,
    I0 = E[σ̃²], I1 = E[u σ̃²], I2 = E[(u² − 1) σ̃²],
    ∂_{X0} a = ω1 e^{−k1 t},   ∂_{X0} Q̂(t) = ω1 ∫₀ᵗ e^{−k1 s} ξ̂(s) ds,
    ∂_{X0} c(t) = ½ ω1 [λ1 e^{−k1 t} ∫₀ᵗ √ξ̂ ds + λ2 e^{−k2 t} ∫₀ᵗ √ξ̂(s) e^{(k2−k1)s} ds],

and ``∂η``, ``∂μ``, ``∂γ`` the exact derivatives of the definitions above (the printed lines for
``∂η`` and ``∂γ`` read differently in the photographs; the exact forms are checked against a
finite difference of eq. 33, ``tests/test_p1_mlp.py``).  SensiY is appendix D: ``∂_{X0} a`` replaced
by ``∂_{Y0} a = ω2 e^{−k2 t}`` everywhere (so ``∂_{Y0} Q̂ = ω2 ∫₀ᵗ e^{−k2 s} ξ̂``, ``∂_{Y0} c = ½ ω2
[λ1 e^{−k1 t} ∫₀ᵗ √ξ̂ e^{(k1−k2)s} ds + λ2 e^{−k2 t} ∫₀ᵗ √ξ̂ ds]``).

**SensiSpot (§4.1.2, eq. 43)** ``= σ_0/(2 Q̂(T)) ∫₀ᵀ (β(t) − c(t) ξ̂(t))/Q̂(t) dt`` with ``β`` the
time derivative of the market skew term that makes eqs. 38 and 41 hold, ``∂_t(2 t² σ̂_t³ S_t)``
(``S`` the ATM skew ``∂σ̂/∂k``; printed ``∂_T[2T σ̂_T S_T]``, whose literal reading makes eq. 43
diverge at 0); computed integrated by parts, ``σ_0 [S_T/σ̂_T + (1/Q̂(T)) ∫ S_t ξ̂/σ̂_t dt −
(1/(2Q̂(T))) ∫ c ξ̂/Q̂ dt]`` — volsto's eq. 12.52 form.

The note's break-evens are on log-vol (``SpotVolCovar = ⟨d ln S, d ln σ̂_T⟩/(σ_0 dt) = SensiSpot +
ρ_SX SensiX + ρ_SY SensiY``, ``VolVar = ⟨d ln σ̂_T⟩/dt``, the full quadratic form, §4);
:meth:`MlpBreakEvens.absolute` gives volsto's absolute forms (the market ATMF prefactor).

**Numerics.** The outer ``t``-integrals by Gauss–Legendre on each segment between the surface's
pillars (where ``ξ̂`` kinks), with ``t = b v²`` on the first one (the ATM skew grows like
``t^{−1/2}`` at short maturities); the inner integrals exactly per cell of a fine grid (``ξ̂``
constant on a cell, its midpoint value — exact on an (e)SSVI surface, whose ``θ`` is piecewise
linear); the bridge expectation by Gauss–Hermite.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from volsto.analytics.reparam import BreakEvenParams

FloatArray = NDArray[np.float64]

N_FINE_DEFAULT: Final[int] = 4000
"""Uniform cells on ``[0, T]`` for the inner integrals (the pillars and the outer nodes are
added to the grid)."""
N_GL_DEFAULT: Final[int] = 48
"""Gauss-Legendre nodes per segment of the outer ``t``-integral."""
N_U_DEFAULT: Final[int] = 48
"""Gauss-Hermite nodes of the bridge expectation ``E_u``."""


@dataclass(frozen=True)
class MlpGrid:
    """The quadrature of :func:`mlp_breakevens` (module docstring, *Numerics*)."""

    n_fine: int = N_FINE_DEFAULT
    n_gl: int = N_GL_DEFAULT
    n_u: int = N_U_DEFAULT

    def __post_init__(self) -> None:
        if self.n_fine < 10 or self.n_gl < 4 or self.n_u < 8:
            raise ValueError("MlpGrid needs n_fine >= 10, n_gl >= 4, n_u >= 8")


@dataclass(frozen=True)
class MlpBreakEvens:
    """The note's break-evens at one maturity (log-vol, module docstring): the market ATMF vol,
    the closed form's own ``σ̂_T(0, 0)`` (eq. 33), the three sensitivities and the correlations."""

    T: float
    atmf_vol: float
    implied_vol: float
    sensi_spot: float
    sensi_x: float
    sensi_y: float
    rho_sx: float
    rho_sy: float
    rho_xy: float

    @property
    def spot_vol_covar(self) -> float:
        return self.sensi_spot + self.rho_sx * self.sensi_x + self.rho_sy * self.sensi_y

    @property
    def vol_var(self) -> float:
        s, x, y = self.sensi_spot, self.sensi_x, self.sensi_y
        return (
            s * s
            + x * x
            + y * y
            + 2.0 * self.rho_sx * s * x
            + 2.0 * self.rho_sy * s * y
            + 2.0 * self.rho_xy * x * y
        )

    def absolute(self) -> tuple[float, float]:
        """``(SpotVolCovar, VolVar)`` in volsto's absolute units (the market ATMF prefactor:
        ``σ̂_T`` and ``σ̂_T²`` times the log-vol forms)."""
        v = self.atmf_vol
        return v * self.spot_vol_covar, v * v * self.vol_var


def _dw_dt(surface: Any, k: FloatArray, t: FloatArray) -> FloatArray:
    """``∂_t w(k, t)``: the surface's exact ``dw_dT`` when it has one, else a central difference."""
    fn = getattr(surface, "dw_dT", None)
    if callable(fn):
        return np.asarray(fn(k, t), dtype=np.float64)
    h = 1e-5 * np.maximum(t, 1e-3)
    up = np.asarray(surface.total_variance(k, t + h), dtype=np.float64)
    dn = np.asarray(surface.total_variance(k, t - h), dtype=np.float64)
    return np.asarray((up - dn) / (2.0 * h), dtype=np.float64)


def _atm_skew(surface: Any, t: FloatArray) -> FloatArray:
    fn = getattr(surface, "atm_skew", None)
    if callable(fn):
        return np.asarray(fn(t), dtype=np.float64)
    h = 1e-3
    up = np.asarray(surface.implied_vol_k(np.full_like(t, h), t), dtype=np.float64)
    dn = np.asarray(surface.implied_vol_k(np.full_like(t, -h), t), dtype=np.float64)
    return np.asarray((up - dn) / (2.0 * h), dtype=np.float64)


def _kernel_cum(s: FloatArray, g: FloatArray, ka: float, kb: float) -> FloatArray:
    """``E(s_j) = ∫₀^{s_j} g(s) e^{−ka (s_j − s)} e^{−kb s} ds`` at every grid point, ``g``
    constant on each cell (one value per cell), the kernel integrated exactly per cell."""
    h = np.diff(s)
    decay = np.exp(-ka * h)
    if abs(ka - kb) > 1e-12:
        cell = (np.exp(-kb * s[1:]) - np.exp(-ka * h - kb * s[:-1])) / (ka - kb)
    else:
        cell = h * np.exp(-kb * s[1:])
    x = g * cell
    out = np.empty(s.size)
    out[0] = 0.0
    acc = 0.0
    for j in range(h.size):
        acc = decay[j] * acc + x[j]
        out[j + 1] = acc
    return out


def _exp_cum(s: FloatArray, g: FloatArray, k: float) -> FloatArray:
    """``∫₀^{s_j} g(s) e^{−k s} ds`` at every grid point, ``g`` constant on each cell."""
    cell = (np.exp(-k * s[:-1]) - np.exp(-k * s[1:])) / k if k > 0.0 else np.diff(s)
    return np.concatenate([[0.0], np.cumsum(g * cell)])


class _Setup:
    """The grids and the ``X0 = Y0 = 0`` quantities of one maturity."""

    def __init__(self, surface: Any, p: BreakEvenParams, T: float, grid: MlpGrid) -> None:
        if not T > 0.0:
            raise ValueError("T must be positive")
        self.surface, self.p, self.T = surface, p, float(T)
        pillars = np.asarray(getattr(surface, "pillars", np.zeros(0)), dtype=np.float64)
        knots = pillars[(pillars > 0.0) & (pillars < self.T)]
        bounds = np.unique(np.concatenate([[0.0], knots, [self.T]]))
        xg, wg = np.polynomial.legendre.leggauss(grid.n_gl)
        nodes, weights = [], []
        for a, b in itertools.pairwise(bounds):
            v, wv = 0.5 * (xg + 1.0), 0.5 * wg
            if a == 0.0:
                # t = b v² on the first segment: the ATM skew grows like t^(-1/2) at short
                # maturities (SensiSpot's integrand), regular in v
                nodes.append(b * v * v)
                weights.append(2.0 * b * v * wv)
            else:
                nodes.append(a + (b - a) * v)
                weights.append((b - a) * wv)
        self.t = np.concatenate(nodes)
        self.wt = np.concatenate(weights)
        fine = np.unique(
            np.concatenate([np.linspace(0.0, self.T, grid.n_fine + 1), bounds, self.t])
        )
        self.s = fine
        self.idx = np.searchsorted(fine, self.t)
        if not np.allclose(fine[self.idx], self.t, rtol=0.0, atol=1e-14):
            raise AssertionError("outer nodes must lie on the fine grid")
        self.mid = 0.5 * (fine[1:] + fine[:-1])
        self.xi_mid = _dw_dt(surface, np.zeros_like(self.mid), self.mid)
        if np.any(self.xi_mid <= 0.0):
            raise ValueError("the ATMF forward variance must be positive on (0, T)")
        u, wu = np.polynomial.hermite_e.hermegauss(grid.n_u)
        self.u = np.asarray(u, dtype=np.float64)
        self.wu = np.asarray(wu, dtype=np.float64) / math.sqrt(2.0 * math.pi)
        h = np.diff(fine)
        self.Q0 = np.concatenate([[0.0], np.cumsum(h * self.xi_mid)])
        g = np.sqrt(self.xi_mid)
        self.c0 = p.lambda1 * _kernel_cum(fine, g, p.k1, 0.0) + p.lambda2 * _kernel_cum(
            fine, g, p.k2, 0.0
        )
        self.Q0t, self.c0t, self.Q0T = self.Q0[self.idx], self.c0[self.idx], float(self.Q0[-1])
        self.eta0, self.mu0, self.gam0 = self._bridge(self.Q0t, self.c0t, self.Q0T)
        k = self.mu0[:, None] + self.gam0[:, None] * self.u[None, :]
        self.sig2 = self.decoration(k)
        self.I0 = self.sig2 @ self.wu
        self.I1 = self.sig2 @ (self.wu * self.u)
        self.I2 = self.sig2 @ (self.wu * (self.u**2 - 1.0))

    @staticmethod
    def _bridge(
        Qt: FloatArray, ct: FloatArray, QT: float
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        eta = np.exp(0.5 * ct - 0.5 * ct * ct / QT)
        mu = (1.0 - Qt / QT) * ct
        gam = np.sqrt(np.maximum(Qt - Qt * Qt / QT, 0.0))
        return eta, mu, gam

    def decoration(self, k: FloatArray) -> FloatArray:
        """``σ̃²(t_i, K)`` at the outer nodes (rows) for forward log-moneyness ``k = ln(K/S_0)``
        (eq. 27 with the Dupire ``φ^market``; the ``X0 = Y0 = 0`` decoration)."""
        tt = np.broadcast_to(self.t[:, None], k.shape)
        w = np.asarray(self.surface.total_variance(k, tt), dtype=np.float64)
        dw = _dw_dt(self.surface, k, tt)
        q0 = self.Q0t[:, None]
        c0 = self.c0t[:, None]
        sw, sq = np.sqrt(w), np.sqrt(q0)
        d_mkt = k / sw + 0.5 * sw
        d_til = (k + 0.5 * q0 - c0) / sq
        return np.asarray(
            dw * (sq / sw) * np.exp(-0.5 * d_mkt**2 + 0.5 * d_til**2), dtype=np.float64
        )

    def implied_variance(self, x0: float = 0.0, y0: float = 0.0) -> float:
        """``σ̂_T(X0, Y0)²`` of eq. 33 (the decoration fixed; the finite-difference reference)."""
        p = self.p

        def a_of(t: FloatArray) -> FloatArray:
            return np.asarray(
                np.exp(p.omega1 * x0 * np.exp(-p.k1 * t) + p.omega2 * y0 * np.exp(-p.k2 * t)),
                dtype=np.float64,
            )

        a_mid = a_of(self.mid)
        h = np.diff(self.s)
        Q = np.concatenate([[0.0], np.cumsum(h * a_mid * self.xi_mid)])
        g = np.sqrt(a_mid * self.xi_mid)
        c = p.lambda1 * _kernel_cum(self.s, g, p.k1, 0.0) + p.lambda2 * _kernel_cum(
            self.s, g, p.k2, 0.0
        )
        eta, mu, gam = self._bridge(Q[self.idx], c[self.idx], float(Q[-1]))
        sig2 = self.decoration(mu[:, None] + gam[:, None] * self.u[None, :])
        f = eta * (sig2 @ self.wu)
        return float(np.sum(self.wt * a_of(self.t) * f) / self.T)

    def d_implied_variance(self, factor: str) -> float:
        """``∂_{X0} σ̂_T²`` (``factor="X"``, eq. 57) or ``∂_{Y0} σ̂_T²`` (``"Y"``, appendix D)
        at ``X0 = Y0 = 0``."""
        p = self.p
        if factor == "X":
            om, kf = p.omega1, p.k1
        elif factor == "Y":
            om, kf = p.omega2, p.k2
        else:
            raise ValueError("factor must be 'X' or 'Y'")
        dQ = om * _exp_cum(self.s, self.xi_mid, kf)
        g = np.sqrt(self.xi_mid)
        dc = (
            0.5
            * om
            * (
                p.lambda1 * _kernel_cum(self.s, g, p.k1, kf)
                + p.lambda2 * _kernel_cum(self.s, g, p.k2, kf)
            )
        )
        Qt, ct, QT = self.Q0t, self.c0t, self.Q0T
        dQt, dQT, dct = dQ[self.idx], float(dQ[-1]), dc[self.idx]
        eta, gam = self.eta0, self.gam0
        # the exact derivatives of η = exp(½c − ½c²/Q̂(T)), μ = (1 − Q̂(t)/Q̂(T)) c,
        # γ² = Q̂(t) − Q̂(t)²/Q̂(T)
        deta = eta * (0.5 * dct - ct * dct / QT + 0.5 * ct * ct * dQT / QT**2)
        dmu = dct * (1.0 - Qt / QT) - ct * (dQt / QT - Qt * dQT / QT**2)
        dgam = (dQt - (2.0 * Qt * dQt / QT - Qt * Qt * dQT / QT**2)) / (2.0 * gam)
        integrand = (
            om * np.exp(-kf * self.t) * eta * self.I0
            + deta * self.I0
            + (eta / gam) * dmu * self.I1
            + (eta / gam) * dgam * self.I2
        )
        return float(np.sum(self.wt * integrand) / self.T)

    def leading_sensi(self, factor: str) -> float:
        """``½ ω_i A_i(T)``, ``A_i = (1/Q̂(T)) ∫₀ᵀ e^{−k_i t} ξ̂ dt``: volsto's first-order
        ``SensiX_i`` (book eq. 7.38, log-vol) on the same grid."""
        p = self.p
        om, kf = (p.omega1, p.k1) if factor == "X" else (p.omega2, p.k2)
        return float(0.5 * om * _exp_cum(self.s, self.xi_mid, kf)[-1] / self.Q0T)

    def sensi_spot(self, sigma_0: float) -> float:
        """Eq. 43 integrated by parts (module docstring)."""
        t = self.t
        xi_t = _dw_dt(self.surface, np.zeros_like(t), t)
        sig_t = np.sqrt(self.Q0t / t)
        sig_T = math.sqrt(self.Q0T / self.T)
        S_t = _atm_skew(self.surface, t)
        S_T = float(_atm_skew(self.surface, np.array([self.T]))[0])
        market = S_T / sig_T + float(np.sum(self.wt * S_t * xi_t / sig_t)) / self.Q0T
        naked = float(np.sum(self.wt * self.c0t * xi_t / self.Q0t)) / (2.0 * self.Q0T)
        return float(sigma_0 * (market - naked))


class MlpPillar:
    """The note's break-evens of one maturity at fixed ``(k1, k2)`` as functions of ``λ`` alone —
    the marking fit's step 1 with ``engine="mlp"`` (SPEC §15 Part 3).  SensiX and SensiY are
    linear in ``ω1`` and ``ω2`` (every ``∂`` of eq. 57 carries ``ω_i``), so :meth:`evaluate`
    returns ``(SensiSpot, G_X, G_Y)`` with ``SensiX = ω1 G_X``, ``SensiY = ω2 G_Y`` and
    ``SpotVolCovar = SensiSpot + λ1 G_X + λ2 G_Y`` (log-vol units): the covariance depends on
    ``(k1, k2, λ1, λ2)`` only, as the note says.  Everything that does not depend on ``λ`` — the
    grids, ``Q̄``, ``γ``, the kernel integrals of ``c`` and of its derivatives, the market part of
    SensiSpot — is computed once."""

    def __init__(
        self,
        surface: Any,
        T: float,
        k1: float,
        k2: float,
        *,
        sigma_0: float,
        grid: MlpGrid | None = None,
    ) -> None:
        g_ = grid or MlpGrid()
        base = BreakEvenParams(float(k1), float(k2), 1.0, 1.0, 0.0, 0.0, 0.0)
        st = _Setup(surface, base, float(T), g_)
        self.surface, self.T, self.k1, self.k2 = surface, float(T), float(k1), float(k2)
        self.sigma_0 = float(sigma_0)
        self.t, self.wt, self.idx, self.u, self.wu = st.t, st.wt, st.idx, st.u, st.wu
        self.Q0t, self.Q0T, self.gam = st.Q0t, st.Q0T, st.gam0
        gsq = np.sqrt(st.xi_mid)
        s_ = st.s
        self.E1 = _kernel_cum(s_, gsq, self.k1, 0.0)[st.idx]
        self.E2 = _kernel_cum(s_, gsq, self.k2, 0.0)[st.idx]
        self.der: dict[str, tuple[FloatArray, FloatArray, FloatArray, float, float]] = {}
        for f, kf in (("X", self.k1), ("Y", self.k2)):
            dq = _exp_cum(s_, st.xi_mid, kf)
            self.der[f] = (
                0.5 * _kernel_cum(s_, gsq, self.k1, kf)[st.idx],
                0.5 * _kernel_cum(s_, gsq, self.k2, kf)[st.idx],
                dq[st.idx],
                float(dq[-1]),
                kf,
            )
        xi_t = _dw_dt(surface, np.zeros_like(st.t), st.t)
        sig_t = np.sqrt(st.Q0t / st.t)
        sig_T = math.sqrt(st.Q0T / self.T)
        S_t = _atm_skew(surface, st.t)
        S_T = float(_atm_skew(surface, np.array([self.T]))[0])
        self.market = S_T / sig_T + float(np.sum(st.wt * S_t * xi_t / sig_t)) / st.Q0T
        self.N1 = float(np.sum(st.wt * self.E1 * xi_t / st.Q0t))
        self.N2 = float(np.sum(st.wt * self.E2 * xi_t / st.Q0t))
        self.atmf_vol = sig_T
        self._setup = st

    def evaluate(self, lam: FloatArray) -> tuple[float, float, float]:
        """``(SensiSpot, G_X, G_Y)`` at ``λ = (λ1, λ2)`` (log-vol units, class docstring)."""
        l1, l2 = float(lam[0]), float(lam[1])
        c = l1 * self.E1 + l2 * self.E2
        Qt, QT, gam = self.Q0t, self.Q0T, self.gam
        eta = np.exp(0.5 * c - 0.5 * c * c / QT)
        mu = (1.0 - Qt / QT) * c
        tt = np.broadcast_to(self.t[:, None], (self.t.size, self.u.size))
        k = mu[:, None] + gam[:, None] * self.u[None, :]
        w = np.asarray(self.surface.total_variance(k, tt), dtype=np.float64)
        dw = _dw_dt(self.surface, k, tt)
        sw, sq = np.sqrt(w), np.sqrt(Qt)[:, None]
        d_mkt = k / sw + 0.5 * sw
        d_til = (k + 0.5 * Qt[:, None] - c[:, None]) / sq
        sig2 = dw * (sq / sw) * np.exp(-0.5 * d_mkt**2 + 0.5 * d_til**2)
        I0 = sig2 @ self.wu
        I1 = sig2 @ (self.wu * self.u)
        I2 = sig2 @ (self.wu * (self.u**2 - 1.0))
        v00 = float(np.sum(self.wt * eta * I0)) / self.T
        out = []
        for f in ("X", "Y"):
            a1, a2, dQt, dQT, kf = self.der[f]
            dc = l1 * a1 + l2 * a2
            deta = eta * (0.5 * dc - c * dc / QT + 0.5 * c * c * dQT / QT**2)
            dmu = dc * (1.0 - Qt / QT) - c * (dQt / QT - Qt * dQT / QT**2)
            dgam = (dQt - (2.0 * Qt * dQt / QT - Qt * Qt * dQT / QT**2)) / (2.0 * gam)
            integrand = (
                np.exp(-kf * self.t) * eta * I0
                + deta * I0
                + (eta / gam) * dmu * I1
                + (eta / gam) * dgam * I2
            )
            out.append(float(np.sum(self.wt * integrand)) / self.T / (2.0 * v00))
        spot = self.sigma_0 * (self.market - (l1 * self.N1 + l2 * self.N2) / (2.0 * QT))
        return spot, out[0], out[1]


def mlp_breakevens(
    surface: Any,
    p: BreakEvenParams,
    T: float,
    *,
    sigma_0: float,
    grid: MlpGrid | None = None,
) -> MlpBreakEvens:
    """The note's break-evens of ``p`` at maturity ``T`` on ``surface`` (module docstring);
    ``sigma_0`` the ultra-short vol of SensiSpot (the note takes the 3M ATMF vol, §6; volsto's
    marking targets carry the 1M one)."""
    s = _Setup(surface, p, T, grid or MlpGrid())
    v00 = s.implied_variance()
    rho_sx = p.rho_SX
    rho_sy = 0.0 if p.omega2 == 0.0 else p.lambda2 / p.omega2
    rho_xy = rho_sx * rho_sy + p.chi * math.sqrt(max(1.0 - rho_sx**2, 0.0)) * math.sqrt(
        max(1.0 - rho_sy**2, 0.0)
    )
    return MlpBreakEvens(
        T=float(T),
        atmf_vol=math.sqrt(s.Q0T / float(T)),
        implied_vol=math.sqrt(v00),
        sensi_spot=s.sensi_spot(sigma_0),
        sensi_x=s.d_implied_variance("X") / (2.0 * v00),
        sensi_y=s.d_implied_variance("Y") / (2.0 * v00),
        rho_sx=rho_sx,
        rho_sy=rho_sy,
        rho_xy=rho_xy,
    )


def mlp_setup(surface: Any, p: BreakEvenParams, T: float, grid: MlpGrid | None = None) -> _Setup:
    """The internal state of one maturity — for the finite-difference checks and the
    comparison with the first-order terms (``implied_variance``, ``d_implied_variance``,
    ``leading_sensi``)."""
    return _Setup(surface, p, T, grid or MlpGrid())


__all__ = [
    "N_FINE_DEFAULT",
    "N_GL_DEFAULT",
    "N_U_DEFAULT",
    "MlpBreakEvens",
    "MlpGrid",
    "MlpPillar",
    "mlp_breakevens",
    "mlp_setup",
]
