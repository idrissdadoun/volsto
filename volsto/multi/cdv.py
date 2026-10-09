"""Cross-dependent volatility on top of the local correlation model — a prototype (SPEC §8.7,
the extension named in the owner's decision 7 of the second round; ``docs/cross_dependent_vol.md``).

**Why.**  With own-level local vols, ``σ_i = σ_i(t, S_i)``, the basket's variance in a sell-off
can rise only through the names' own skews and through the correlation, which is capped at 1:
on the Dow the index downside wing is then out of reach (the clipped mass of M12).  Guyon's
cross-dependent volatility (Risk, 2016) lets a name's volatility depend on the index as well:

    σ_i(t, S_i, B) = σ_Dup,i(t, k_i) · g(k_B) · s_i(t, k_i),
    g(k) = clip(e^{−β k}, g_min, g_max),        s_i(t, k) = 1 / √E[g(k_B(t))² | k_i(t) = k].

**Each name keeps its smile.**  By Gyöngy's theorem name ``i`` has the marginals of its Dupire
diffusion iff ``E[σ_i² | S_i] = σ_Dup,i²``, which the normalisation ``s_i`` enforces: one
conditional expectation per name and per time, estimated on the particle cloud like the
leverage of a local-stochastic volatility model (SPEC §4.1).

**The index.**  With ``m_i = g(k_B)·s_i(t, k_i)`` and ``u_i = ω_i σ_Dup,i m_i`` the basket
variance is still ``v_B = a + λ b`` (``a = uᵀR_low u``, ``b = uᵀ(R_high − R_low)u``) and the
local correlation follows from the same formula, ``λ = (σ_B² − E[a|k_B])/E[b|k_B]``.  With
``β > 0`` every name's volatility rises when the index falls, so less correlation is asked for
on the downside.

**Discretisation.**  ``λ`` and ``m_i`` are frozen over a step at their start-of-step values;
``m_i²`` is passed to the library's shared step (``models.localvol.spot_step``) as the variance
factor, where the local correlation kernel passes 1.  With ``β = 0``: ``g ≡ 1``, ``s_i ≡ 1``,
``m_i² = 1.0`` exactly, and every statement is M12's — the calibrated ``λ`` and the particle
cloud equal ``calibrate_local_correlation``'s bit for bit
(``tests/test_cross_dependent_vol.py::test_beta_zero_is_the_local_correlation_model``).

**Status.**  A prototype: off by default, used by nothing in M12, no cache, no products layer
(the simulation returns the names' log-spots at record times), one step per kernel call.  The
local correlation sources are not modified.
"""

# ruff: noqa: E501
from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volsto._numba import njit, prange
from volsto.calibration.guard import check_calibration_allowed
from volsto.calibration.local_correlation import (
    CLIP_GATE_SD,
    MIN_ATM_TIME,
    UNIDENTIFIED_RATIO,
    _check_inputs,
    conditional_expectations_ab,
)
from volsto.calibration.particle import conditional_variance_estimate
from volsto.config import LocalCorrelationConfig, ParticleConfig, SimConfig
from volsto.engine.grid import TimeGrid
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import ImpliedSurface
from volsto.models.localvol import LocalVol, interp_uniform, spot_step
from volsto.models.lsv import scheme_mode
from volsto.multi.draws import lower_row_dot
from volsto.multi.family import CorrelationFamily
from volsto.multi.lc_draws import LocalCorrelationDraws, high_row_dot
from volsto.multi.lc_model import BasketSpec, block_tables, grid_arrays

FloatArray = NDArray[np.float64]
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class CrossDependence:
    """``g(k_B) = clip(e^{−β k_B}, g_min, g_max)``: the factor every name's volatility carries
    before its own normalisation.  ``β = 0`` is the local correlation model."""

    beta: float = 0.0
    g_min: float = 0.5
    g_max: float = 2.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.beta):
            raise ValueError("beta must be finite")
        if not 0.0 < self.g_min <= 1.0 <= self.g_max:
            raise ValueError("the clip must satisfy 0 < g_min <= 1 <= g_max")

    def g(self, k_basket: FloatArray) -> FloatArray:
        return np.asarray(
            np.clip(np.exp(-self.beta * np.asarray(k_basket)), self.g_min, self.g_max),
            dtype=np.float64,
        )


@njit(parallel=True, cache=True)
def cdv_ab(
    log_spot: FloatArray,
    ln_f: FloatArray,
    weights: FloatArray,
    shifts: FloatArray,
    ln_fb: float,
    k0: float,
    dk: float,
    var_rows: FloatArray,
    scale_rows: FloatArray,
    beta: float,
    g_min: float,
    g_max: float,
    low_equi: bool,
    rho_low: float,
    r_low: FloatArray,
    high_ones: bool,
    l_high: FloatArray,
    out_a: FloatArray,
    out_b: FloatArray,
    out_k: FloatArray,
) -> None:  # pragma: no cover - numba
    """``lc_ab`` with ``u_i = ω_i·√var_rows[i](k_i)·m_i``, ``m_i = g(k_B)·scale_rows[i](k_i)``:
    the two terms of ``v_B = a + λ b`` and the basket log-moneyness per particle.  The same
    statements in the same order as ``volsto.multi.lc_kernel.lc_ab``; with ``m_i = 1.0`` the
    same numbers bit for bit."""
    n_paths = log_spot.shape[0]
    n = log_spot.shape[1]
    r = l_high.shape[1]
    for p in prange(n_paths):
        level = 0.0
        for i in range(n):
            level += weights[i] * np.exp(log_spot[p, i] - shifts[i])
        kb = np.log(level) - ln_fb
        out_k[p] = kb
        g = min(max(np.exp(-beta * kb), g_min), g_max)
        u = np.empty(n)
        s1 = 0.0
        s2 = 0.0
        for i in range(n):
            ls = log_spot[p, i]
            omega = weights[i] * np.exp(ls - shifts[i]) / level
            ki = ls - ln_f[i]
            m = g * interp_uniform(k0, dk, scale_rows[i], ki)
            u[i] = omega * np.sqrt(interp_uniform(k0, dk, var_rows[i], ki)) * m
            s1 += u[i]
            s2 += u[i] * u[i]
        if low_equi:
            a = (1.0 - rho_low) * s2 + rho_low * (s1 * s1)
        else:
            a = 0.0
            for i in range(n):
                for j in range(n):
                    a += u[i] * r_low[i, j] * u[j]
        if high_ones:
            high = s1 * s1
        else:
            high = 0.0
            for q in range(r):
                proj = 0.0
                for i in range(n):
                    proj += l_high[i, q] * u[i]
                high += proj * proj
        out_a[p] = a
        out_b[p] = high - a


@njit(parallel=True, cache=True)
def cdv_step(
    log_spot: FloatArray,
    int_var: FloatArray,
    sum_sq: FloatArray,
    eps: FloatArray,
    eta: FloatArray,
    dt: float,
    ln_f0: FloatArray,
    ln_f1: FloatArray,
    drifts: FloatArray,
    k0: float,
    dk: float,
    var_a: FloatArray,
    var_b: FloatArray,
    mode: int,
    pc_eta: float,
    weights: FloatArray,
    shifts0: FloatArray,
    ln_fb0: float,
    lam_row: FloatArray,
    lam_k0: float,
    lam_dk: float,
    lam_max: float,
    scale_rows: FloatArray,
    beta: float,
    g_min: float,
    g_max: float,
    low_equi: bool,
    d_low: FloatArray,
    ell_low: FloatArray,
    l_low: FloatArray,
    l_high: FloatArray,
) -> None:  # pragma: no cover - numba
    """One step of every particle and every name: ``lc_diffuse_block``'s statements for a block
    of one step, with the variance factor ``m_i² = (g(k_B)·scale_rows[i](k_i))²`` in place of
    ``1.0`` in the names' step.  ``eps`` is ``(n_paths, n)``, ``eta`` ``(n_paths, r)``;
    ``var_a``, ``var_b`` and ``scale_rows`` are ``(n, n_k)`` on the shared grid ``(k0, dk)``."""
    n_paths = log_spot.shape[0]
    n = log_spot.shape[1]
    for p in prange(n_paths):
        level = 0.0
        for i in range(n):
            level += weights[i] * np.exp(log_spot[p, i] - shifts0[i])
        kb = np.log(level) - ln_fb0
        lam = interp_uniform(lam_k0, lam_dk, lam_row, kb)
        lam = min(max(lam, 0.0), lam_max)
        c1 = np.sqrt(1.0 - lam)
        c2 = np.sqrt(lam)
        g = min(max(np.exp(-beta * kb), g_min), g_max)
        er = eps[p]
        hr = eta[p]
        pref = 0.0
        for i in range(n):
            if low_equi:
                x = pref + d_low[i] * er[i]
                pref = pref + ell_low[i] * er[i]
            else:
                x = lower_row_dot(l_low, er, i)
            zj = c1 * x + c2 * high_row_dot(l_high, hr, i)
            ls = log_spot[p, i]
            m = g * interp_uniform(k0, dk, scale_rows[i], ls - ln_f0[i])
            va = m * m
            dls, v, _b0, _bbx = spot_step(
                ls, dt, drifts[i], zj, k0, dk, var_a[i], var_b[i], va, va,
                ln_f0[i], ln_f1[i], mode, pc_eta,
            )  # fmt: skip
            log_spot[p, i] = ls + dls
            int_var[p, i] += v * dt
            sum_sq[p, i] += dls * dls


@dataclass
class CDVResult:
    """Output of :func:`calibrate_cdv`.  ``lam`` and ``lam_star`` (clipped and raw) are
    ``(n_slices, n_k)`` on ``k_grid``; ``scale[j, i]`` is ``s_i(t_j, ·)`` on ``name_k_grid``;
    ``target[j]`` the basket local variance the row was matched to (with ``fixed_lambda``: the
    regressed ``E[a|k] + λ·E[b|k]`` of the run itself); the clipped masses per slice, on the
    whole cloud and inside ±2.5 at-the-money standard deviations; the trusted range; the final
    cloud."""

    dependence: CrossDependence
    times: FloatArray
    k_grid: FloatArray
    name_k_grid: FloatArray
    lam: FloatArray
    lam_star: FloatArray
    scale: FloatArray
    target: FloatArray
    clipped_low: FloatArray
    clipped_high: FloatArray
    inner_low: FloatArray
    inner_high: FloatArray
    lambda_mean: FloatArray
    q_lo: FloatArray
    q_hi: FloatArray
    final_log_spot: FloatArray
    final_k_basket: FloatArray
    wall_time: float
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def max_clipped_mass_inner(self) -> float:
        return float(max(self.inner_low.max(), self.inner_high.max()))

    def summary(self) -> dict[str, Any]:
        return {
            "beta": self.dependence.beta,
            "g_min": self.dependence.g_min,
            "g_max": self.dependence.g_max,
            "n_slices": int(self.times.size),
            "max_clipped_low": float(self.clipped_low.max()),
            "max_clipped_high": float(self.clipped_high.max()),
            "max_clipped_low_inner": float(self.inner_low.max()),
            "max_clipped_high_inner": float(self.inner_high.max()),
            "lambda_mean_last": float(self.lambda_mean[-1]),
            "scale_min": float(self.scale.min()),
            "scale_max": float(self.scale.max()),
            "wall_time": self.wall_time,
            "timings": dict(self.timings),
        }


def _name_scales(
    ls: FloatArray,
    ln_f: FloatArray,
    g2: FloatArray,
    grid: FloatArray,
    cfg: ParticleConfig,
    bounds: tuple[float, float] = (0.0, np.inf),
) -> FloatArray:
    """``s_i(k) = 1/√E[g² | k_i = k]`` on the shared grid, one regression per name on the cloud
    (the leverage's estimator; the bandwidth is the cloud's own width in ``k_i`` times
    ``bandwidth_factor · N^{−1/5}``).  ``bounds``: the range ``g²`` lies in — a conditional
    expectation of ``g²`` lies in it too, so the estimate is clipped to it (the regression's
    tail extrapolation beyond the cloud does not respect it)."""
    n_particles, n = ls.shape
    out = np.empty((n, grid.size))
    n_exp = float(n_particles) ** (-0.2)
    for i in range(n):
        k = ls[:, i] - ln_f[i]
        h = max(cfg.bandwidth_factor * float(k.std()) * n_exp, cfg.bandwidth_min)
        e = conditional_variance_estimate(k, g2, grid, h, cfg)
        out[i] = 1.0 / np.sqrt(np.clip(np.maximum(e, 1e-12), bounds[0], bounds[1]))
    return out


def calibrate_cdv(
    models: Sequence[LocalVol],
    family: CorrelationFamily,
    basket: BasketSpec,
    index_surface: ImpliedSurface,
    index_lv: LocalVolSurface,
    cfg: ParticleConfig,
    sim: SimConfig,
    lc: LocalCorrelationConfig,
    dependence: CrossDependence,
    *,
    fixed_lambda: float | None = None,
    target_rows: FloatArray | None = None,
    draws: LocalCorrelationDraws | None = None,
) -> CDVResult:
    """The particle calibration of ``λ(t, k)`` and of the names' normalisations ``s_i(t, k)``
    under the cross-dependence ``dependence`` (module docstring) — one pass, the loop of
    ``calibrate_local_correlation`` with the per-name regressions added.

    ``fixed_lambda``: no calibration of ``λ`` — the cloud is moved with that constant and the
    result's ``target`` is the basket local variance it produces, ``E[a|k] + λ·E[b|k]`` per
    slice (a synthetic truth).  ``target_rows``: ``(n_slices, n_k)`` on ``index_lv.k_grid``,
    used in place of the rows of ``index_lv`` (the round trip of a synthetic truth).
    ``index_surface`` gives the at-the-money vol of the bandwidth rule and of the ±2.5 sd range.

    Refused while calibration is forbidden: the check below must stay the first statement."""
    check_calibration_allowed("calibrate_cdv")
    t_start = time.perf_counter()
    T = cfg.horizon
    grid = TimeGrid.build([T], sim.dt_max)
    _check_inputs(models, family, basket, index_lv, cfg, lc, grid)
    if cfg.second_pass:
        raise ValueError("the prototype runs one pass (second_pass = False)")
    if lc.lambda_tail != "regressions":
        raise ValueError('the prototype implements lambda_tail = "regressions" only')
    scheme = sim.scheme
    mode = scheme_mode(scheme)
    times = grid.times
    n_steps = grid.n_steps
    n_slices = n_steps + 1
    n = len(models)
    N = cfg.n_particles
    weights = basket.weights
    lv0 = models[0].local_vol
    name_grid = np.asarray(lv0.k_grid, dtype=np.float64)
    k_grid = index_lv.k_grid
    lam_k0, lam_dk = index_lv.k0, index_lv.dk
    lam_max = family.lambda_max
    rho_low = family.rho_low if family.rho_low is not None else 0.0
    ln_f, drifts, shifts, ln_fb = grid_arrays(models, basket, times)
    sig_atm = np.asarray(index_surface.atm_vol(np.maximum(times, MIN_ATM_TIME)), dtype=np.float64)
    n_exp = float(N) ** (-0.2)
    step_average = lc.target_average == "step"
    dep = dependence
    plain = dep.beta == 0.0
    if target_rows is not None and target_rows.shape != (n_slices, k_grid.size):
        raise ValueError("target_rows must be (n_slices, n_k) on the index grid")

    def interval(j: int) -> list[float]:
        end = float(times[j + 1]) if j < n_steps else float(times[j] + grid.dts[-1])
        return [float(times[j]), end]

    def name_rows(j: int) -> FloatArray:
        if step_average:
            rows = [m.local_vol.var_time_average(interval(j))[0] for m in models]
        else:
            rows = [m.local_vol.var_at_times([times[j]])[0] for m in models]
        return np.ascontiguousarray(np.stack(rows))

    def target_row(j: int) -> FloatArray:
        if target_rows is not None:
            return np.asarray(target_rows[j], dtype=np.float64)
        if step_average:
            return np.asarray(index_lv.var_time_average(interval(j))[0], dtype=np.float64)
        return np.asarray(index_lv.var_at_times([times[j]])[0], dtype=np.float64)

    L = np.empty((n_slices, k_grid.size))
    L_star = np.empty((n_slices, k_grid.size))
    scale = np.ones((n_slices, n, name_grid.size))
    target = np.empty((n_slices, k_grid.size))
    clipped_low, clipped_high = np.zeros(n_slices), np.zeros(n_slices)
    inner_low, inner_high = np.zeros(n_slices), np.zeros(n_slices)
    lambda_mean = np.empty(n_slices)
    q_lo, q_hi = np.empty(n_slices), np.empty(n_slices)
    a, b, kb = np.empty(N), np.empty(N), np.empty(N)
    timings = {"tables": 0.0, "draws": 0.0, "kernel": 0.0, "scales": 0.0, "ab": 0.0, "regression": 0.0}  # fmt: skip

    d = draws if draws is not None else LocalCorrelationDraws(cfg.seed, N, n_steps, family, cfg.antithetic)  # fmt: skip
    if (d.n_paths, d.n_steps, d.n_assets) != (N, n_steps, n):
        raise ValueError("the draws do not match the particles, the grid or the family")
    ls = np.empty((N, n))
    for i, m in enumerate(models):
        ls[:, i] = np.log(m.spot)
    iv = np.zeros((N, n))
    sq = np.zeros((N, n))

    def terms(j: int) -> None:
        cdv_ab(
            ls, np.ascontiguousarray(ln_f[:, j]), weights, np.ascontiguousarray(shifts[:, j]),
            float(ln_fb[j]), lv0.k0, lv0.dk, name_rows(j), scale[j], dep.beta, dep.g_min, dep.g_max,
            family.low_equi, rho_low, family.r_low, family.high_ones, family.l_high, a, b, kb,
        )  # fmt: skip

    # slice 0: one state, g = 1 there, so the scales are 1 and the row is one value
    terms(0)
    a0, b0, k_0 = float(a[0]), float(b[0]), float(kb[0])
    if fixed_lambda is not None:
        L_star[0] = L[0] = fixed_lambda
        target[0] = a0 + fixed_lambda * b0
    else:
        target[0] = target_row(0)
        flat0 = b0 <= UNIDENTIFIED_RATIO * a0
        star0 = 0.0 if flat0 else (float(np.interp(k_0, k_grid, target[0])) - a0) / b0
        L_star[0], L[0] = star0, min(max(star0, 0.0), lam_max)
        clipped_low[0], clipped_high[0] = float(star0 < 0.0), float(star0 > lam_max)
        inner_low[0], inner_high[0] = clipped_low[0], clipped_high[0]
    lambda_mean[0] = float(L[0, 0])
    q_lo[0] = q_hi[0] = k_0
    for j in range(n_steps):
        t_nodes = times[j : j + 2]
        t1 = float(times[j + 1])
        c0 = time.perf_counter()
        var_a, var_b, _ = block_tables(models, t_nodes, scheme, False)
        c1 = time.perf_counter()
        eps = d.eps_block(j, j + 1, 0, N)
        eta = d.eta_block(j, j + 1, 0, N)
        c2 = time.perf_counter()
        cdv_step(
            ls, iv, sq, np.ascontiguousarray(eps[:, 0, :]), np.ascontiguousarray(eta[:, 0, :]),
            float(t_nodes[1] - t_nodes[0]), np.ascontiguousarray(ln_f[:, j]),
            np.ascontiguousarray(ln_f[:, j + 1]), np.ascontiguousarray(drifts[:, j]), lv0.k0, lv0.dk,
            np.ascontiguousarray(var_a[:, 0, :]), np.ascontiguousarray(var_b[:, 0, :]), mode,
            scheme.pc_eta, weights, np.ascontiguousarray(shifts[:, j]), float(ln_fb[j]), L[j],
            lam_k0, lam_dk, lam_max, scale[j], dep.beta, dep.g_min, dep.g_max, family.low_equi,
            family.d_low, family.ell_low, family.l_low, family.l_high,
        )  # fmt: skip
        c3 = time.perf_counter()
        del eps, eta
        if not plain:
            level = np.exp(ls - shifts[:, j + 1][None, :]) @ weights
            g2 = dep.g(np.log(level) - ln_fb[j + 1]) ** 2
            bounds = (dep.g_min**2, dep.g_max**2)
            scale[j + 1] = _name_scales(ls, ln_f[:, j + 1], g2, name_grid, cfg, bounds)
        c4 = time.perf_counter()
        terms(j + 1)
        c5 = time.perf_counter()
        h = max(cfg.bandwidth_factor * float(sig_atm[j + 1]) * np.sqrt(t1) * n_exp, cfg.bandwidth_min)  # fmt: skip
        ea, eb, lo, hi = conditional_expectations_ab(kb, a, b, k_grid, h, cfg, (None, None))
        q_lo[j + 1], q_hi[j + 1] = lo, hi
        if fixed_lambda is not None:
            L_star[j + 1] = L[j + 1] = fixed_lambda
            target[j + 1] = ea + fixed_lambda * eb
            lambda_mean[j + 1] = fixed_lambda
        else:
            target[j + 1] = target_row(j + 1)
            flat = eb <= UNIDENTIFIED_RATIO * ea
            with np.errstate(divide="ignore", invalid="ignore"):
                star = np.where(flat, 0.0, (target[j + 1] - ea) / np.where(flat, 1.0, eb))
            L_star[j + 1], L[j + 1] = star, np.clip(star, 0.0, lam_max)
            per = np.interp(kb, k_grid, star)
            below, above = per < 0.0, per > lam_max
            clipped_low[j + 1], clipped_high[j + 1] = float(below.mean()), float(above.mean())
            inner = np.abs(kb) <= CLIP_GATE_SD * float(sig_atm[j + 1]) * np.sqrt(t1)
            inner_low[j + 1] = float((below & inner).mean())
            inner_high[j + 1] = float((above & inner).mean())
            lambda_mean[j + 1] = float(np.clip(per, 0.0, lam_max).mean())
        c6 = time.perf_counter()
        for key, dt in zip(timings, (c1 - c0, c2 - c1, c3 - c2, c4 - c3, c5 - c4, c6 - c5), strict=True):  # fmt: skip
            timings[key] += dt
    wall = time.perf_counter() - t_start
    log.info(
        "cross-dependent calibration: beta %.3g, %d steps, N=%d, %d names, %.1f s; clipped mass "
        "inside ±%.1f sd: %.4f low / %.4f high; scales in [%.4f, %.4f]",
        dep.beta, n_steps, N, n, wall, CLIP_GATE_SD, float(inner_low.max()), float(inner_high.max()),
        float(scale.min()), float(scale.max()),
    )  # fmt: skip
    return CDVResult(
        dep, times.copy(), np.asarray(k_grid, dtype=np.float64).copy(), name_grid.copy(), L, L_star,
        scale, target, clipped_low, clipped_high, inner_low, inner_high, lambda_mean, q_lo, q_hi,
        ls, kb.copy(), wall, timings,
    )  # fmt: skip


def simulate_cdv(
    result: CDVResult,
    models: Sequence[LocalVol],
    family: CorrelationFamily,
    basket: BasketSpec,
    sim: SimConfig,
    record_times: Sequence[float],
) -> tuple[FloatArray, FloatArray]:
    """The calibrated model on the pricing seed: the names' log-spots ``(n_record, n_paths, n)``
    and the basket log-moneyness ``(n_record, n_paths)`` at ``record_times`` (grid times of the
    calibration).  ``λ`` and the scales are those of the calibration's slices, frozen over each
    step as in the calibration."""
    times = result.times
    n_steps = times.size - 1
    n = len(models)
    P = sim.n_paths
    dep = result.dependence
    scheme = sim.scheme
    mode = scheme_mode(scheme)
    lv0 = models[0].local_vol
    lam_k0, lam_dk = float(result.k_grid[0]), float(result.k_grid[1] - result.k_grid[0])
    ln_f, drifts, shifts, ln_fb = grid_arrays(models, basket, times)
    cols: dict[int, int] = {}
    for t in record_times:
        j = int(np.argmin(np.abs(times - float(t))))
        if abs(times[j] - float(t)) > 1e-10:
            raise ValueError(f"record time {t} is not a grid time of the calibration")
        cols[j] = len(cols)
    out_ls = np.empty((len(cols), P, n))
    out_kb = np.empty((len(cols), P))
    d = LocalCorrelationDraws(sim.seed, P, n_steps, family, sim.antithetic)
    ls = np.empty((P, n))
    for i, m in enumerate(models):
        ls[:, i] = np.log(m.spot)
    iv = np.zeros((P, n))
    sq = np.zeros((P, n))
    for j in range(n_steps):
        t_nodes = times[j : j + 2]
        var_a, var_b, _ = block_tables(models, t_nodes, scheme, False)
        eps = d.eps_block(j, j + 1, 0, P)
        eta = d.eta_block(j, j + 1, 0, P)
        cdv_step(
            ls, iv, sq, np.ascontiguousarray(eps[:, 0, :]), np.ascontiguousarray(eta[:, 0, :]),
            float(t_nodes[1] - t_nodes[0]), np.ascontiguousarray(ln_f[:, j]),
            np.ascontiguousarray(ln_f[:, j + 1]), np.ascontiguousarray(drifts[:, j]), lv0.k0, lv0.dk,
            np.ascontiguousarray(var_a[:, 0, :]), np.ascontiguousarray(var_b[:, 0, :]), mode,
            scheme.pc_eta, basket.weights, np.ascontiguousarray(shifts[:, j]), float(ln_fb[j]),
            result.lam[j], lam_k0, lam_dk, family.lambda_max, result.scale[j], dep.beta, dep.g_min,
            dep.g_max, family.low_equi, family.d_low, family.ell_low, family.l_low, family.l_high,
        )  # fmt: skip
        if j + 1 in cols:
            c = cols[j + 1]
            out_ls[c] = ls
            level = np.exp(ls - shifts[:, j + 1][None, :]) @ basket.weights
            out_kb[c] = np.log(level) - ln_fb[j + 1]
    return out_ls, out_kb


__all__ = ["CDVResult", "CrossDependence", "calibrate_cdv", "cdv_ab", "cdv_step", "simulate_cdv"]
