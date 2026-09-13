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
distinct step size.  The spot is log-Euler with ``ξ_t^t`` frozen over the step (§7.3.1); with
``local_var_time_average`` the deterministic prefactor ``g(u) = ξ_0^u e^{−½ω²χ(u,u)}`` is averaged
over the step with the factors frozen (owner amendment).  ``θ = 0`` is the 1F model
(``α_0 = 1``, ``x_t^T = e^{−k1(T−t)} X¹_t``); the factor count is then 1.
Checked by ``tests/test_bergomi.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto._numba import njit, prange
from volsto.analytics.bergomi import alpha_theta, chi
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
    coef: FloatArray,
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
    """Advance paths over a block of steps: exact OU factors (7.15–7.18), log-Euler spot.

    ``z`` is ``(n_paths, n_block, 1 + n_factors)``; ``chol[j]`` the ``(1+nf)×(1+nf)`` factor of the
    step covariance; ``decay[j, i] = e^{−k_i δτ_j}``; ``g_step[j]`` the (time-averaged)
    deterministic prefactor of ``ξ_t^t`` for the step and ``g_node`` its value at the nodes;
    ``coef[i] = ω α_θ w_i`` so that ``ξ_t^t = g exp(Σ coef_i X^i)``.  With ``use_lev`` the spot
    variance is ``L²(t, x) ξ_t^t`` from the ``lev_*`` tables (LSV, M3) and the spot step uses the
    shared :func:`~volsto.models.localvol.spot_step` in ``mode``.
    """
    n = log_spot.shape[0]
    nb = z.shape[1]
    nf = factors.shape[1]
    for p in prange(n):
        ls = log_spot[p]
        iv = int_var[p]
        sq = sum_sq[p]
        for j in range(nb):
            dt = t_nodes[j + 1] - t_nodes[j]
            e = 0.0
            for i in range(nf):
                e += coef[i] * factors[p, i]
            vsv = g_step[j] * np.exp(e)
            if use_lev:
                dls, v = spot_step(
                    ls,
                    dt,
                    drifts[j],
                    z[p, j, 0],
                    k0,
                    dk,
                    lev_a[j],
                    lev_b[j],
                    vsv,
                    ln_f_nodes[j],
                    ln_f_nodes[j + 1],
                    mode,
                    eta,
                )
            else:
                v = vsv
                dls = drifts[j] - 0.5 * v * dt + np.sqrt(v * dt) * z[p, j, 0]
            # exact factor step: X^i <- e^{-k_i dt} X^i + delta X^i, delta X = chol rows 1.. of z
            for i in range(nf):
                dx = 0.0
                for l in range(i + 2):
                    dx += chol[j, 1 + i, l] * z[p, j, l]
                factors[p, i] = decay[j, i] * factors[p, i] + dx
            ls += dls
            iv += v * dt
            sq += dls * dls
            col = step_record[j]
            if col >= 0:
                e = 0.0
                for i in range(nf):
                    e += coef[i] * factors[p, i]
                    out_factors[p, col, i] = factors[p, i]
                vnew = g_node[j + 1] * np.exp(e)
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
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """``(chol, decay, g_step, g_node)`` for the steps between ``t_nodes``."""
        dts = np.diff(t_nodes)
        chol = sqrt_covariance(factor_step_covariance(self.params, dts, self.n_factors))
        decay = np.exp(-dts[:, None] * self.ks[None, :])
        g_node = self.g(t_nodes)
        if scheme.local_var_time_average:
            # Simpson on the smooth prefactor g over each step
            g_mid = self.g(0.5 * (t_nodes[:-1] + t_nodes[1:]))
            g_step = (g_node[:-1] + 4.0 * g_mid + g_node[1:]) / 6.0
        else:
            g_step = g_node[:-1].copy()
        return chol, decay, np.ascontiguousarray(g_step), g_node

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
            chol, decay, g_step, g_node = self.step_tables(t_nodes, scheme)
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
                self.coef,
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
