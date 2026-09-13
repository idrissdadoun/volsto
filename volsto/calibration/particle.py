"""Particle-method calibration of the leverage function (SPEC §4.1).

Target (Guyon & Henry-Labordère 2012; Bergomi §12.2.5):
``L(t, S)² = σ_loc²(t, S) / E[ξ_t^t | S_t = S]``
with ``σ_loc`` the Dupire local volatility of the target surface and ``ξ_t^t`` the SV kernel's
instantaneous variance.  Algorithm, on the simulation time grid of the shared
:class:`~volsto.config.StepSchedule`:

1. ``t_0``: ``L(0, S) = σ_loc(0, S) / sqrt(ξ_0^0)``.
2. Step all ``N`` particles from ``t_i`` to ``t_{i+1}`` with the current ``L(t_i, ·)`` — using the
   pricing kernel itself (:func:`volsto.models.lsv.step_lsv_block`, frozen-leverage rule, same
   :class:`~volsto.config.SchemeConfig`), so pricing reproduces the calibration step for step.
3. At ``t_{i+1}`` estimate ``E[ξ | S]`` by kernel regression in ``k = ln(S/F)`` (local-linear by
   default, Nadaraya–Watson optional; Gaussian kernel truncated at ``4h`` or quartic), bandwidth
   ``h_i = c σ_ATM(t_{i+1}) sqrt(t_{i+1}) N^{−1/5}``
   floored at ``h_min``, on a 201-point grid; outside the cloud's ``[q, 1−q]`` quantiles the
   estimate is held flat.  The (smooth) estimate is interpolated onto the fine leverage grid where
   ``σ_loc²`` is resolved (dk = 0.0025, as in the Dupire grid), and ``L(t_{i+1}, ·)`` is set.
4. Continue.  Optional second pass: repeat with a fresh seed and average the two ``L`` surfaces.

The regression kernel is a ``numba`` ``prange`` loop over grid points with a truncated window on
the sorted particles: ``O(N log N)`` for the sort plus ``O(N × window fraction × n_grid)``.
Checked by ``tests/test_lsv.py`` (§4.2 acceptance with the pricing kernel, calibration / pricing
path identity, variance-swap invariance across ω).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from volsto._numba import njit, prange
from volsto.config import LocalVolConfig, ParticleConfig, SimConfig
from volsto.engine.grid import TimeGrid
from volsto.engine.rng import GaussianDraws
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import ImpliedSurface
from volsto.models.bergomi import BergomiSV
from volsto.models.leverage import LeverageFunction
from volsto.models.lsv import step_lsv_block

FloatArray = NDArray[np.float64]
log = logging.getLogger(__name__)

#: Bumped whenever the calibration numerics change; part of the cache key (SPEC §4.3).
#: Guarded by ``tests/test_lsv.py::test_calibration_code_tag_guard`` (source hash of the
#: calibration and stepping modules).
CALIBRATION_CODE_TAG = "m4b"


@njit(parallel=True, cache=True)
def kernel_regression(
    ks: FloatArray,
    vs: FloatArray,
    grid: FloatArray,
    h: float,
    gaussian: bool,
    local_linear: bool,
    min_window: int,
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:  # pragma: no cover - numba
    """``E[v | k]`` on ``grid`` from particles sorted by ``k`` with a truncated window (``4h``
    Gaussian, ``h`` quartic): Nadaraya–Watson or local-linear (weighted least squares with a
    slope, which cancels the design bias ``h² m' f'/f``).  Returns the estimate (``nan`` where the
    window is empty or degenerate), its slope ``dm/dk`` (local-linear only, else 0) and the
    window counts."""
    ng = grid.shape[0]
    out = np.empty(ng)
    slope = np.zeros(ng)
    cnt = np.empty(ng, dtype=np.int64)
    w = 4.0 if gaussian else 1.0
    n = ks.shape[0]
    for g in prange(ng):
        hg = h
        lo = np.searchsorted(ks, grid[g] - w * hg)
        hi = np.searchsorted(ks, grid[g] + w * hg)
        if hi - lo < min_window and n >= min_window:
            # k-NN floor: widen to the nearest min_window particles and rescale the bandwidth
            c = np.searchsorted(ks, grid[g])
            lo = max(0, c - min_window // 2)
            hi = min(n, lo + min_window)
            lo = max(0, hi - min_window)
            span = max(abs(ks[lo] - grid[g]), abs(ks[hi - 1] - grid[g]))
            hg = max(h, span / w)
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        t0 = 0.0
        t1 = 0.0
        for i in range(lo, hi):
            u = (ks[i] - grid[g]) / hg
            if gaussian:
                wt = np.exp(-0.5 * u * u)
            else:
                d = 1.0 - u * u
                wt = d * d if d > 0.0 else 0.0
            s0 += wt
            s1 += wt * u
            s2 += wt * u * u
            t0 += wt * vs[i]
            t1 += wt * u * vs[i]
        cnt[g] = hi - lo
        if s0 <= 0.0:
            out[g] = np.nan
        elif local_linear:
            det = s0 * s2 - s1 * s1
            if det > 1e-12 * s0 * s2:
                out[g] = (s2 * t0 - s1 * t1) / det
                slope[g] = (s0 * t1 - s1 * t0) / det / hg
            else:
                out[g] = t0 / s0
        else:
            out[g] = t0 / s0
    return out, slope, cnt


def conditional_variance_estimate(
    k: FloatArray, v: FloatArray, grid: FloatArray, h: float, cfg: ParticleConfig
) -> FloatArray:
    """``E[v | k]`` on the output ``grid`` (the leverage ``k`` grid).

    The regression itself runs on a dense grid of ``cfg.n_regression_points`` spanning the
    cloud's trusted ``[q, 1−q]`` quantile range (M4b: a fixed grid over the whole leverage range
    put 0.025 between regression points while the cloud at the first slices is 0.006 wide and
    ``E[ξ|k]`` varies like ``exp(ω ρ k / (σ√t))``; linear interpolation of that exponential between
    far-apart nodes biased the short-end leverage low by ~1% in variance at ω = 3 — 0.09 vol
    points on the 1m ATM vol — independently of the step size); values inside the range are
    interpolated onto ``grid``, the tails are extended flat or log-linearly / log-quadratically
    (SPEC §4.1 and :class:`~volsto.config.ParticleConfig`).
    """
    order = np.argsort(k, kind="stable")
    ks = np.ascontiguousarray(k[order])
    vs = np.ascontiguousarray(v[order])
    n = ks.size
    q_lo = float(ks[min(int(cfg.quantile_clip * n), n - 1)])
    q_hi = float(ks[max(int((1.0 - cfg.quantile_clip) * n) - 1, 0)])
    if not q_hi > q_lo:
        # degenerate cloud (e.g. all particles at one point): use the plain mean everywhere
        return np.full(grid.shape, float(v.mean()))
    kreg = np.linspace(q_lo, q_hi, cfg.n_regression_points)
    window = max(cfg.min_window, int(cfg.min_window_fraction * n))
    m, slope, _ = kernel_regression(
        ks, vs, kreg, h, cfg.kernel == "gaussian", cfg.regression == "local_linear", window
    )
    good = np.isfinite(m) & (m > 0)
    if not np.any(good):
        return np.full(grid.shape, float(v.mean()))
    if not np.all(good):
        m = np.interp(kreg, kreg[good], m[good])
    if cfg.bias_correction and cfg.regression == "local_linear" and kreg.size >= 5:
        # plug-in correction of the local-linear bias 1/2 h^2 kappa_2 m'' (kappa_2 = 1 Gaussian,
        # 1/7 quartic); m'' by central second differences on a stencil of width ~h (the
        # regression grid is much finer than the bandwidth: differencing neighbouring points
        # would amplify the regression noise by (h/d)^2 before the one-sided clip below)
        kappa2 = 1.0 if cfg.kernel == "gaussian" else 1.0 / 7.0
        d = kreg[1] - kreg[0]
        st = int(np.clip(round(h / d), 1, (kreg.size - 1) // 2))
        m2 = np.zeros_like(m)
        m2[st:-st] = (m[2 * st :] - 2.0 * m[st:-st] + m[: -2 * st]) / (st * d) ** 2
        m2[:st], m2[-st:] = m2[st], m2[-st - 1]
        m = np.maximum(m - 0.5 * h * h * kappa2 * m2, 0.5 * m)
    out = np.interp(grid, kreg, m)
    lo = grid < q_lo
    hi = grid > q_hi
    if cfg.tail_extrapolation in ("log_linear", "log_quadratic", "adaptive"):
        # ln m over the outer 10% of the trusted points (at least 5): linear fit for the edge
        # slope, or a quadratic whose slope is only allowed to decay towards zero (then flat)
        n_fit = max(5, kreg.size // 10)
        for edge, mask, pts, sign in (
            (0, lo, slice(0, min(n_fit, kreg.size)), -1.0),
            (kreg.size - 1, hi, slice(max(kreg.size - n_fit, 0), kreg.size), 1.0),
        ):
            if not np.any(mask):
                continue
            x = kreg[pts] - kreg[edge]
            y = np.log(m[pts])
            d = sign * (grid[mask] - kreg[edge])  # distance into the tail (>= 0)
            if x.size >= 3 and np.ptp(x) > 0:
                b = float(np.polyfit(x, y, 1)[0])  # d ln m / dk at the edge
            else:
                b = float(slope[edge] / m[edge])
            b = float(np.clip(b, -50.0, 50.0))
            expo = sign * b * d
            # adaptive: saturating quadratic only where ln m falls into the tail
            quadratic = cfg.tail_extrapolation == "log_quadratic" or (
                cfg.tail_extrapolation == "adaptive" and sign * b < 0
            )
            if quadratic and x.size >= 4 and np.ptp(x) > 0:
                c2, b2, _ = np.polyfit(x, y, 2)
                b2 = float(np.clip(b2, -50.0, 50.0))
                curv = float(c2)
                # keep the quadratic only if it makes |slope| decay into the tail
                if sign * b2 * curv < 0 and abs(b2) > 0:
                    d_flat = abs(b2) / (2.0 * abs(curv))  # slope vanishes here
                    dd = np.minimum(d, d_flat)
                    expo = sign * b2 * dd + curv * dd * dd
                else:
                    expo = sign * b2 * d
            out[mask] = m[edge] * np.exp(np.clip(expo, -3.0, 3.0))
    else:
        out[lo] = m[0]
        out[hi] = m[-1]
    return np.asarray(out, dtype=np.float64)


@dataclass
class CalibrationResult:
    """Output of :func:`calibrate_leverage`."""

    leverage: LeverageFunction
    grid: TimeGrid
    final_log_spot: FloatArray
    final_factors: FloatArray
    bandwidths: FloatArray
    wall_time: float
    passes: int
    #: particle clouds ``(log_spot, factors)`` at requested slice times (diagnostics: the
    #: pricing kernel must reproduce them to 1e-12 on the calibration grid, M6 Part 0)
    snapshots: dict[float, tuple[FloatArray, FloatArray]] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"CalibrationResult({self.leverage!r}, n_steps={self.grid.n_steps}, "
            f"passes={self.passes}, wall_time={self.wall_time:.1f}s)"
        )


def leverage_grid_config(
    surface: ImpliedSurface, cfg: ParticleConfig, base: LocalVolConfig | None = None
) -> LocalVolConfig:
    """Leverage ``k`` grid: ``±leverage_std_span · σ_ATM(T) √T`` with step ``leverage_dk``."""
    if base is not None:
        return base
    T = cfg.horizon
    # at least +-2.5: the far put wing feeds long-dated variance swaps (LocalVolConfig note)
    half = max(float(cfg.leverage_std_span * surface.atm_vol(T) * np.sqrt(T)), 2.5)
    n_half = int(np.ceil(half / cfg.leverage_dk))
    n_k = 2 * n_half + 1
    return LocalVolConfig(
        t_max=min(max(T, 2.0 / 365.0), surface.max_maturity),
        k_min=-n_half * cfg.leverage_dk,
        k_max=n_half * cfg.leverage_dk,
        n_k=n_k,
    )


def calibrate_leverage(
    surface: ImpliedSurface,
    kernel: BergomiSV,
    cfg: ParticleConfig,
    sim: SimConfig,
    *,
    local_vol_cfg: LocalVolConfig | None = None,
    local_vol: LocalVolSurface | None = None,
    snapshot_times: Sequence[float] | None = None,
) -> CalibrationResult:
    """Calibrate ``L(t, S)`` so that the LSV model reprices ``surface`` (SPEC §4.1).

    ``snapshot_times``: the particle cloud (last pass) is stored at the grid time nearest each
    requested time (``CalibrationResult.snapshots``, keyed by the actual grid time) — a
    diagnostic hook, no effect on the calibration.

    ``sim`` supplies the step schedule and scheme shared with pricing; ``cfg`` the particle
    settings.  ``local_vol`` may be passed to reuse a Dupire surface (its ``k`` grid becomes the
    leverage grid).
    """
    t_start = time.perf_counter()
    if kernel.forward_curve is not surface.forward_curve and not np.isclose(
        kernel.forward_curve.spot, surface.forward_curve.spot
    ):
        raise ValueError("kernel and surface must share the forward curve")
    lv_cfg = leverage_grid_config(surface, cfg, local_vol_cfg)
    lv = local_vol or LocalVolSurface.from_implied(surface, lv_cfg)
    if lv.t_grid[-1] < cfg.horizon - 1e-9:
        raise ValueError("local vol grid must extend to the calibration horizon")
    T = cfg.horizon
    grid = TimeGrid.build([T], sim.dt_max)
    scheme = sim.scheme
    times = grid.times
    n_steps = grid.n_steps
    nf = kernel.n_factors
    N = cfg.n_particles
    fc = kernel.forward_curve
    ln_f = np.asarray(fc.log_forward(times), dtype=np.float64)
    drifts = np.diff(ln_f)
    k_grid = lv.k_grid
    k0, dk = lv.k0, lv.dk
    xi00 = float(kernel.xi0.xi0(0.0))
    sig_atm = np.asarray(surface.atm_vol(np.maximum(times, 1.0 / 365.0)), dtype=np.float64)
    n_exp = float(N) ** (-0.2)
    no_record = np.array([-1], dtype=np.int64)
    dummy = np.empty((N, 1))
    dummy_f = np.empty((N, 1, nf))

    def target_var(j: int) -> FloatArray:
        """Local variance the slice ``L(t_j, ·)`` reproduces: the point value ``σ_loc²(t_j, ·)``
        (its step average was tried at M4b and rejected: it shifted the coarse-schedule
        variance swaps 0.02–0.05 vol points low without curing the ω = 3 residual)."""
        return np.asarray(lv.var_at_times([times[j]])[0], dtype=np.float64)

    results: list[LeverageFunction] = []
    bandwidths = np.empty(n_steps)
    final_ls = final_fac = None
    snap_idx: dict[int, float] = {}
    if snapshot_times:
        for ts in snapshot_times:
            j_snap = int(np.argmin(np.abs(times - float(ts))))
            snap_idx[j_snap] = float(times[j_snap])
    snapshots: dict[float, tuple[FloatArray, FloatArray]] = {}
    n_pass = 2 if cfg.second_pass else 1
    for p in range(n_pass):
        seed = cfg.seed + p
        draws = GaussianDraws(seed, N, n_steps, kernel.n_brownians, cfg.antithetic)
        ls = np.full(N, np.log(fc.spot))
        fac = np.zeros((N, nf))
        iv = np.zeros(N)
        sq = np.zeros(N)
        L = np.empty((n_steps + 1, k_grid.size))
        L[0] = np.clip(np.sqrt(target_var(0) / xi00), cfg.l_min, cfg.l_max)
        for j in range(n_steps):
            t_nodes = times[j : j + 2]
            t1 = float(times[j + 1])
            row = np.ascontiguousarray((L[j] * L[j])[None, :])
            step_lsv_block(
                kernel,
                scheme,
                ls,
                fac,
                iv,
                sq,
                draws.block(j, j + 1, 0, N),
                t_nodes,
                ln_f[j : j + 2],
                drifts[j : j + 1],
                no_record,
                k0,
                dk,
                row,
                row,
                row,
                dummy,
                dummy,
                dummy_f,
                dummy,
                dummy,
            )
            v = kernel.variance_from_factors(t1, fac)
            k = ls - ln_f[j + 1]
            h = max(
                cfg.bandwidth_factor * float(sig_atm[j + 1]) * np.sqrt(t1) * n_exp,
                cfg.bandwidth_min,
            )
            bandwidths[j] = h
            ev = conditional_variance_estimate(k, v, k_grid, h, cfg)
            L[j + 1] = np.clip(np.sqrt(target_var(j + 1) / ev), cfg.l_min, cfg.l_max)
            if j + 1 in snap_idx:
                snapshots[snap_idx[j + 1]] = (ls.copy(), fac.copy())
        final_ls, final_fac = ls, fac
        results.append(
            LeverageFunction(
                times,
                k_grid,
                L,
                fc,
                {
                    "seed": seed,
                    "n_particles": N,
                    "horizon": T,
                    "kernel": cfg.kernel,
                    "bandwidth_factor": cfg.bandwidth_factor,
                    "code_tag": CALIBRATION_CODE_TAG,
                    "scheme": scheme.__dict__.copy(),
                    "schedule": repr(sim.step_schedule),
                },
            )
        )
        log.info("particle pass %d/%d done (%d steps, N=%d)", p + 1, n_pass, n_steps, N)
    lev = results[0] if n_pass == 1 else LeverageFunction.average(results[0], results[1])
    assert final_ls is not None and final_fac is not None
    wall = time.perf_counter() - t_start
    lev.metadata["wall_time"] = wall
    return CalibrationResult(lev, grid, final_ls, final_fac, bandwidths, wall, n_pass, snapshots)
