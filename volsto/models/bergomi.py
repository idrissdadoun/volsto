"""Two-factor lognormal Bergomi forward-variance model (SPEC §3.3; Bergomi ch. 7).

Factors (eq. 7.30 and below): ``dX^i = −k_i X^i dt + dW^i``, ``X^i_0 = 0``,
``corr(dW¹, dW²) = ρ12``;
``x_t^T = α_θ [(1−θ) e^{−k1(T−t)} X¹_t + θ e^{−k2(T−t)} X²_t]`` (7.30), ``α_θ`` from (7.29).
Forward variances (7.32–7.35): ``ξ_t^T = ξ_0^T exp(ω x_t^T − ½ ω² χ(t,T))``, ``ω = 2ν``,
``χ(t,T) = Var[x_t^T]`` so ``E[ξ_t^T] = ξ_0^T`` (martingale; tested).  ``V_t = ξ_t^t``.

Exact simulation (§7.3.1, eqs. 7.15–7.18) over a step ``δτ``:
``X^i_{τ+δτ} = e^{−k_i δτ} X^i_τ + δX^i``, ``E[δX^i δX^j] = ρ_ij (1−e^{−(k_i+k_j)δτ})/(k_i+k_j)``,
``E[δW^S δX^i] = ρ_iS (1−e^{−k_i δτ})/k_i``, ``E[(δW^S)²] = δτ``; the joint Gaussian
``(δW^S, δX¹, δX²)`` is drawn from the Cholesky factor of that covariance, computed once per
distinct step size.  The spot step is the second-order SV step of M4b by default
(``SchemeConfig.sv_order2``: factors advanced first, trapezoidal variance in the drift, explicit
weak order-2 spot/variance cross terms; see :func:`bergomi_block`); with ``sv_order2=False`` it
is log-Euler with ``ξ_t^t`` frozen over the step (§7.3.1) and, under ``local_var_time_average``,
the deterministic prefactor ``g(u) = ξ_0^u e^{−½ω²χ(u,u)}`` averaged over the step with the
factors frozen (owner amendment, M1→M2).  ``θ = 0`` is the 1F model
(``α_0 = 1``, ``x_t^T = e^{−k1(T−t)} X¹_t``); the factor count is then 1.
Checked by ``tests/test_bergomi.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto._numba import njit, prange
from volsto.analytics.bergomi import alpha_theta, chi, cov_x_diag
from volsto.config import BergomiParams, SchemeConfig
from volsto.engine.grid import TimeGrid
from volsto.engine.paths import PathSet
from volsto.engine.rng import GaussianDraws
from volsto.market.curves import ForwardCurve
from volsto.market.varswap import ForwardVarianceCurve
from volsto.models.base import Model, ModelState
from volsto.models.localvol import interp_uniform, spot_step

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


# --------------------------------------------------------------------------------------------
# step covariances (eqs. 7.17–7.18)
# --------------------------------------------------------------------------------------------


def factor_step_covariance(
    p: BergomiParams, dt: ArrayLike, n_factors: int, *, with_brownians: bool = False
) -> FloatArray:
    """Covariance of ``(δW^S, δX¹..δX^n[, δW¹..δW^n])`` over steps ``dt`` — shape ``(…, m, m)``.

    ``E[δX^i δX^j] = ρ_ij (1−e^{−(k_i+k_j)δτ})/(k_i+k_j)`` (7.17), ``E[δW^S δX^i] = ρ_iS
    (1−e^{−k_i δτ})/k_i`` (7.18), ``E[(δW^S)²] = δτ``; with the Brownian increments themselves
    (mixing solution): ``E[δW^i δW^j] = ρ_ij δτ``, ``E[δX^j δW^i] = ρ_ij (1−e^{−k_j δτ})/k_j``.
    """
    dt_ = np.atleast_1d(np.asarray(dt, dtype=np.float64))
    ks = np.array([p.k1, p.k2][:n_factors])
    corr = p.correlation_matrix[: 1 + n_factors, : 1 + n_factors]
    nf = n_factors
    m = 1 + nf + (nf if with_brownians else 0)
    C = np.zeros((*dt_.shape, m, m))
    C[..., 0, 0] = dt_
    for i in range(nf):
        gi = -np.expm1(-ks[i] * dt_) / ks[i]
        C[..., 0, 1 + i] = C[..., 1 + i, 0] = corr[0, 1 + i] * gi
        for j in range(nf):
            kij = ks[i] + ks[j]
            C[..., 1 + i, 1 + j] = corr[1 + i, 1 + j] * (-np.expm1(-kij * dt_) / kij)
        if with_brownians:
            C[..., 0, 1 + nf + i] = C[..., 1 + nf + i, 0] = corr[0, 1 + i] * dt_
            for j in range(nf):
                C[..., 1 + nf + i, 1 + nf + j] = corr[1 + i, 1 + j] * dt_
                gj = -np.expm1(-ks[j] * dt_) / ks[j]
                C[..., 1 + j, 1 + nf + i] = C[..., 1 + nf + i, 1 + j] = corr[1 + j, 1 + i] * gj
    return C


def sqrt_covariance(C: FloatArray) -> FloatArray:
    """Lower-triangular Cholesky factors of a stack of covariances; eigen-sqrt fallback when
    singular (e.g. ``|ρ| = 1``), in which case the nesting of factors is lost."""
    try:
        return np.asarray(np.linalg.cholesky(C), dtype=np.float64)
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(C)
        w = np.clip(w, 0.0, None)
        return np.asarray(v * np.sqrt(w)[..., None, :], dtype=np.float64)


# --------------------------------------------------------------------------------------------
# kernel
# --------------------------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def bergomi_block(
    log_spot: FloatArray,
    factors: FloatArray,
    int_var: FloatArray,
    sum_sq: FloatArray,
    z: FloatArray,
    t_nodes: FloatArray,
    ln_f_nodes: FloatArray,
    drifts: FloatArray,
    step_record: IntArray,
    chol: FloatArray,
    decay: FloatArray,
    g_step: FloatArray,
    g_node: FloatArray,
    dlng: FloatArray,
    coef: FloatArray,
    ks: FloatArray,
    rho_s: FloatArray,
    corr_x: FloatArray,
    sv2: int,
    use_lev: int,
    k0: float,
    dk: float,
    lev_a: FloatArray,
    lev_b: FloatArray,
    lev_rec: FloatArray,
    mode: int,
    eta: float,
    out_log_spot: FloatArray,
    out_var: FloatArray,
    out_factors: FloatArray,
    out_int_var: FloatArray,
    out_sum_sq: FloatArray,
) -> None:  # pragma: no cover - numba
    """Advance paths over a block of steps: exact OU factors (7.15–7.18), then the spot step.

    ``z`` is ``(n_paths, n_block, 1 + n_factors)``; ``chol[j]`` the ``(1+nf)×(1+nf)`` factor of the
    step covariance (spot Brownian first, so ``δW^S = chol[j,0,0] z_0``); ``decay[j, i] =
    e^{−k_i δτ_j}``; ``g_node`` the deterministic prefactor of ``ξ_t^t`` at the nodes, ``g_step``
    its time average over the step (frozen-variance step only) and ``dlng[j] = Δ ln g / δτ``;
    ``coef[i] = ω α_θ w_i`` so that ``ξ_t^t = g exp(Σ coef_i X^i)``; ``ks``, ``rho_s = corr(W^S,
    W^i)`` and ``corr_x`` the factor mean reversions and correlations.  With ``use_lev`` the spot
    variance is ``L²(t, x) ξ_t^t`` from the ``lev_*`` tables (LSV, M3) and the spot step uses the
    shared :func:`~volsto.models.localvol.spot_step` in ``mode``.

    ``sv2 = 0``: the variance is frozen at the step start (``ξ_{t_n}`` with the time-averaged
    prefactor), the original M2/M3 step.  ``sv2 = 1`` (M4b, second-order SV step): the factors are
    advanced first, so ``ξ_{t_{n+1}}`` is known exactly; the drift uses the trapezoid of the
    variance, and the weak order-2 Itô–Taylor terms of the spot increment that involve the
    variance factors are added explicitly (Kloeden–Platen 14.2, correlated drivers written in
    terms of ``δW^S`` and ``δW̃_i = ΔX_i − (e^{−k_i δ} − 1) X_i ≈ δW^i``)::

        Δx += ¼ b0 Σ_i c_i (δW̃_i δW^S − ρ_Si δ)                    (spot/variance cross term)
            + ½ δ b0 [½ ∂_t ln g − ½ Σ_i c_i k_i X_i + ⅛ Σ_ij c_i c_j ρ_ij
                      + ½ (b_x/b0) Σ_i ρ_Si c_i] δW^S

    with ``b0 = L(x_n) sqrt(ξ_{t_n})`` and ``b0 b_x`` from the leverage slope; the end-of-step
    variance also enters the drift ``a_{n+1}`` (which supplies the ``−¼ c δ b0² δW^i`` term).  The
    zero-mean Lévy areas are dropped.  Checked against the mixing solution and a fine grid by
    ``tests/test_scheme.py::test_second_order_sv_step``.
    """
    n = log_spot.shape[0]
    nb = z.shape[1]
    nf = factors.shape[1]
    cc = 0.0
    for i in range(nf):
        for l in range(nf):
            cc += coef[i] * coef[l] * corr_x[i, l]
    cc *= 0.125
    rs = 0.0
    for i in range(nf):
        rs += rho_s[i] * coef[i]
    for p in prange(n):
        ls = log_spot[p]
        iv = int_var[p]
        sq = sum_sq[p]
        x0 = np.empty(nf)
        dwt = np.empty(nf)
        for j in range(nb):
            dt = t_nodes[j + 1] - t_nodes[j]
            e0 = 0.0
            for i in range(nf):
                x0[i] = factors[p, i]
                e0 += coef[i] * x0[i]
            # exact factor step: X^i <- e^{-k_i dt} X^i + delta X^i, delta X = chol rows 1.. of z
            e1 = 0.0
            for i in range(nf):
                dx = 0.0
                for l in range(i + 2):
                    dx += chol[j, 1 + i, l] * z[p, j, l]
                factors[p, i] = decay[j, i] * x0[i] + dx
                dwt[i] = dx  # = ΔX_i − (e^{−k_i δ} − 1) X_i = ∫ e^{−k_i(δ−u)} dW_i ≈ δW_i
                e1 += coef[i] * factors[p, i]
            if sv2:
                vsv0 = g_node[j] * np.exp(e0)
                vsv1 = g_node[j + 1] * np.exp(e1)
            else:
                vsv0 = g_step[j] * np.exp(e0)
                vsv1 = vsv0
            dws = chol[j, 0, 0] * z[p, j, 0]
            if use_lev:
                dls, v, b0, bbx = spot_step(
                    ls,
                    dt,
                    drifts[j],
                    z[p, j, 0],
                    k0,
                    dk,
                    lev_a[j],
                    lev_b[j],
                    vsv0,
                    vsv1,
                    ln_f_nodes[j],
                    ln_f_nodes[j + 1],
                    mode,
                    eta,
                )
            else:
                b0 = np.sqrt(vsv0)
                bbx = 0.0
                v = 0.5 * (vsv0 + vsv1)
                dls = drifts[j] - 0.5 * v * dt + b0 * dws
            if sv2:
                cross = 0.0
                lin = 0.0
                for i in range(nf):
                    cross += coef[i] * (dwt[i] * dws - rho_s[i] * dt)
                    lin += coef[i] * ks[i] * x0[i]
                lterm = 0.5 * dlng[j] - 0.5 * lin + cc + 0.5 * (bbx / b0) * rs
                dls += 0.25 * b0 * cross + 0.5 * dt * b0 * lterm * dws
            ls += dls
            iv += v * dt
            sq += dls * dls
            col = step_record[j]
            if col >= 0:
                for i in range(nf):
                    out_factors[p, col, i] = factors[p, i]
                vnew = g_node[j + 1] * np.exp(e1)
                if use_lev:
                    vnew *= interp_uniform(k0, dk, lev_rec[j], ls - ln_f_nodes[j + 1])
                out_log_spot[p, col] = ls
                out_var[p, col] = vnew
                out_int_var[p, col] = iv
                out_sum_sq[p, col] = sq
        log_spot[p] = ls
        int_var[p] = iv
        sum_sq[p] = sq


# --------------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------------


class BergomiSV(Model):
    """Pure two-factor (or 1F when ``θ = 0``) lognormal Bergomi model with ``L ≡ 1``."""

    def __init__(
        self, params: BergomiParams, xi0: ForwardVarianceCurve, forward_curve: ForwardCurve
    ) -> None:
        self.params = params
        self.xi0 = xi0
        self.forward_curve = forward_curve
        self.n_factors = 1 if params.is_one_factor else 2
        self.n_brownians = 1 + self.n_factors
        self.alpha = alpha_theta(params)
        self.weights = np.array([1.0 - params.theta, params.theta][: self.n_factors])
        self.ks = np.array([params.k1, params.k2][: self.n_factors])
        self.coef = params.omega * self.alpha * self.weights
        corr = params.correlation_matrix
        self.rho_s = np.ascontiguousarray(corr[0, 1 : 1 + self.n_factors])
        self.corr_x = np.ascontiguousarray(corr[1 : 1 + self.n_factors, 1 : 1 + self.n_factors])

    # -- deterministic pieces ------------------------------------------------------------------

    def g(self, t: ArrayLike) -> FloatArray:
        """``g(t) = ξ_0^t exp(−½ ω² χ(t,t))``: ``ξ_t^t = g(t) exp(ω x_t^t)``."""
        t_ = np.asarray(t, dtype=np.float64)
        return np.asarray(
            self.xi0.xi0(t_) * np.exp(-0.5 * self.params.omega**2 * chi(self.params, t_, t_)),
            dtype=np.float64,
        )

    def x_factor(self, t: float, factors: FloatArray, T: ArrayLike) -> FloatArray:
        """``x_t^T = α_θ Σ_i w_i e^{−k_i(T−t)} X^i_t`` (eq. 7.30); ``factors`` is ``(n, nf)``."""
        T_ = np.atleast_1d(np.asarray(T, dtype=np.float64))
        decay = np.exp(-self.ks[None, :] * (T_[:, None] - t))  # (nT, nf)
        return np.asarray(self.alpha * (factors @ (self.weights * decay).T), dtype=np.float64)

    def variance_from_factors(self, t: float, factors: FloatArray) -> FloatArray:
        """``ξ_t^t`` from the factor values."""
        return np.asarray(float(self.g(t)) * np.exp(factors @ self.coef), dtype=np.float64)

    def forward_variance(self, t: float, factors: FloatArray, T: ArrayLike) -> FloatArray:
        """``ξ_t^T = ξ_0^T exp(ω x_t^T − ½ω² χ(t,T))`` (eqs. 7.33–7.35); shape ``(n, nT)``."""
        T_ = np.atleast_1d(np.asarray(T, dtype=np.float64))
        w = self.params.omega
        out = self.xi0.xi0(T_)[None, :] * np.exp(
            w * self.x_factor(t, factors, T_) - 0.5 * w * w * chi(self.params, t, T_)[None, :]
        )
        return np.asarray(out[:, 0] if np.ndim(T) == 0 else out, dtype=np.float64)

    def vs_variance(
        self, t: float, factors: FloatArray, T1: float, T2: float, n_quad: int = 32
    ) -> FloatArray:
        """``σ̂²_{T1T2}(t) = (1/(T2−T1)) ∫_{T1}^{T2} ξ_t^τ dτ`` by Gauss–Legendre quadrature."""
        x, w = np.polynomial.legendre.leggauss(n_quad)
        tau = 0.5 * (T2 - T1) * x + 0.5 * (T1 + T2)
        wt = 0.5 * (T2 - T1) * w / (T2 - T1)
        return np.asarray(self.forward_variance(t, factors, tau) @ wt, dtype=np.float64)

    def integrated_variance_moments(
        self, times: FloatArray, scheme: SchemeConfig, T1: float, T2: float
    ) -> tuple[float, float]:
        """Exact mean and variance of the accumulator ``Σ_n v_n Δ_n`` over the steps of ``times``
        inside ``[T1, T2]``, where ``v_n = g_step_n exp(ω x_{t_n}^{t_n})`` is the step variance the
        kernel uses (factors frozen at the step start, prefactor time-averaged under ``scheme``):

        ``E = Σ_n Δ_n g_step_n e^{½ω²χ(t_n)}``,
        ``Var = Σ_{n,m} Δ_n Δ_m g_n g_m e^{½ω²(χ(t_n)+χ(t_m))} (e^{ω² Cov(x_n, x_m)} − 1)``
        with ``g_n = g_step_n``.

        Discrete counterpart of :func:`volsto.analytics.bergomi.var_integrated_variance`;
        the two differ by the O(Δ) left-point quadrature of the pathwise integral (−0.3% on [0, 1y]
        at Δ = 1/365 for Table 8.2, checked in ``tests/test_bergomi.py``).
        """
        t = np.asarray(times, dtype=np.float64)
        sel = (t[:-1] >= T1 - 1e-12) & (t[1:] <= T2 + 1e-12)
        idx = np.flatnonzero(sel)
        if idx.size == 0:
            raise ValueError("no steps inside [T1, T2]")
        t_nodes = t[idx[0] : idx[-1] + 2]
        _, _, g_step, _, _ = self.step_tables(t_nodes, scheme)
        tn = t_nodes[:-1]
        dts = np.diff(t_nodes)
        w = self.params.omega
        a = dts * g_step * np.exp(0.5 * w * w * chi(self.params, tn, tn))
        cov = cov_x_diag(self.params, tn[:, None], tn[None, :])
        var = float(a @ np.expm1(w * w * cov) @ a)
        return float(a.sum()), var

    def factor_covariance(self, t: float) -> FloatArray:
        """``Cov(X^i_t, X^j_t) = ρ_ij (1−e^{−(k_i+k_j)t})/(k_i+k_j)`` (analytic, for tests)."""
        return factor_step_covariance(self.params, t, self.n_factors)[0, 1:, 1:]

    # -- Model interface -----------------------------------------------------------------------

    def initial_state(self, n_paths: int) -> ModelState:
        return ModelState(
            0.0,
            np.full(n_paths, np.log(self.spot)),
            np.full(n_paths, float(self.xi0.xi0(0.0))),
            np.zeros((n_paths, self.n_factors)),
        )

    def instantaneous_variance(self, state: ModelState) -> FloatArray:
        return self.variance_from_factors(state.t, state.factors)

    def bump(self, **kwargs: Any) -> BergomiSV:
        """Supported: any :class:`BergomiParams` field, ``xi0`` (curve), ``xi0_scale``, ``spot``."""
        params = self.params
        xi0 = self.xi0
        fc = self.forward_curve
        changes: dict[str, float] = {}
        for key, val in kwargs.items():
            if key in BergomiParams.__dataclass_fields__:
                changes[key] = float(val)
            elif key == "xi0":
                xi0 = val
            elif key == "xi0_scale":
                xi0 = ForwardVarianceCurve(
                    xi0.maturities, float(val) * xi0.total_variance(xi0.maturities)
                )
            elif key == "spot":
                fc = fc.with_spot(float(val))
            else:
                raise ValueError(f"BergomiSV.bump: unknown parameter {key!r}")
        if changes:
            params = params.replace(**changes)
        return BergomiSV(params, xi0, fc)

    def step_tables(
        self, t_nodes: FloatArray, scheme: SchemeConfig
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray, FloatArray]:
        """``(chol, decay, g_step, g_node, dlng)`` for the steps between ``t_nodes``;
        ``dlng = Δ ln g / δτ`` feeds the second-order SV step."""
        dts = np.diff(t_nodes)
        chol = sqrt_covariance(factor_step_covariance(self.params, dts, self.n_factors))
        decay = np.exp(-dts[:, None] * self.ks[None, :])
        g_node = self.g(t_nodes)
        if scheme.local_var_time_average and not scheme.sv_order2:
            # Simpson on the smooth prefactor g over each step (frozen-variance step only: with
            # the second-order step the time dependence of g enters through ½ ∂_t ln g and the
            # end-of-step variance; averaging g with the factors frozen at t_n biases E[ξ] by
            # −¼ ω² δ χ'(t), 0.6% at 1m for ω = 3 and δ = 1/365)
            g_mid = self.g(0.5 * (t_nodes[:-1] + t_nodes[1:]))
            g_step = (g_node[:-1] + 4.0 * g_mid + g_node[1:]) / 6.0
        else:
            g_step = g_node[:-1].copy()
        dlng = np.diff(np.log(g_node)) / dts
        return chol, decay, np.ascontiguousarray(g_step), g_node, np.ascontiguousarray(dlng)

    def simulate_chunk(
        self,
        grid: TimeGrid,
        draws: GaussianDraws,
        p0: int,
        p1: int,
        scheme: SchemeConfig,
        *,
        step_block: int = 64,
    ) -> PathSet:
        n = p1 - p0
        nf = self.n_factors
        out = PathSet.empty(n, grid.record_times, nf)
        state = self.initial_state(n)
        out.log_spot[:, 0] = state.log_spot
        out.variance[:, 0] = state.variance
        out.factors[:, 0, :] = 0.0
        out.int_var[:, 0] = 0.0
        out.sum_sq[:, 0] = 0.0
        ls = state.log_spot.copy()
        fac = np.zeros((n, nf))
        iv = np.zeros(n)
        sq = np.zeros(n)
        ln_f = np.asarray(self.forward_curve.log_forward(grid.times), dtype=np.float64)
        drifts = np.diff(ln_f)
        dummy = np.ones((1, 2))
        for s0 in range(0, grid.n_steps, step_block):
            s1 = min(s0 + step_block, grid.n_steps)
            t_nodes = grid.times[s0 : s1 + 1]
            chol, decay, g_step, g_node, dlng = self.step_tables(t_nodes, scheme)
            z = draws.block(s0, s1, p0, p1)
            bergomi_block(
                ls,
                fac,
                iv,
                sq,
                z,
                t_nodes,
                ln_f[s0 : s1 + 1],
                drifts[s0:s1],
                grid.step_record[s0:s1],
                chol,
                decay,
                g_step,
                g_node,
                dlng,
                self.coef,
                self.ks,
                self.rho_s,
                self.corr_x,
                int(scheme.sv_order2),
                0,
                0.0,
                1.0,
                dummy,
                dummy,
                dummy,
                0,
                0.0,
                out.log_spot,
                out.variance,
                out.factors,
                out.int_var,
                out.sum_sq,
            )
        return out

    def __repr__(self) -> str:
        kind = "1F" if self.params.is_one_factor else "2F"
        return f"BergomiSV({kind}, {self.params}, xi0={self.xi0!r}, spot={self.spot})"
