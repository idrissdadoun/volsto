"""Particle calibration of the local correlation function ``λ(t, k)`` (SPEC §8.7, M12 part LC4).

**Condition** (derived; Gyöngy 1986, recalled).  In performance mode ``Y = ln B`` is an Itô
process with variance ``v_B = Σ_ij ω_i ω_j σ_i σ_j ρ_ij`` and drift ``−½ v_B``; the listed index
in forward moneyness is the Dupire diffusion of local variance ``σ_B²(t, k)``.  The two have the
same one-dimensional marginals iff ``E[v_B(t) | B_t] = σ_B²(t, ln B_t)`` for all ``t``.  With
``v_B = a + λ·b`` (:mod:`volsto.multi.family`) and ``λ`` a function of ``(t, k)`` only:

    λ*(t, k) = (σ_B²(t, k) − E[a_t | k]) / E[b_t | k],        λ(t, k) = clip(λ*(t, k), 0, λ_max)

The right-hand side depends on the law of the solution — a McKean equation, solved by the
particle method (Guyon & Henry-Labordère; SPEC §4.1): the condition can be met iff ``E[a|k] ≤
σ_B² ≤ E[a + λ_max·b | k]``; where it cannot, the index smile is out of the family's reach and
the clip records it.

**Algorithm** (:func:`calibrate_local_correlation`), on the grid of the shared step schedule:

0. ``t_0``: every particle is at the same state, so the whole row is one value,
   ``λ(t_0, ·) ≡ clip((σ̄²_{B,0}(k_0) − a_0)/b_0, 0, λ_max)``.
1. Step the cloud over ``[t_j, t_{j+1}]`` with the pricing kernel itself
   (:func:`volsto.multi.lc_kernel.lc_diffuse_block`, a one-step block, the row ``λ(t_j, ·)``
   frozen over the step, the same scheme) — pricing reproduces the calibration step for step.
2. Per particle at ``t_{j+1}``: ``a``, ``b`` and the basket log-moneyness ``k_B``
   (:func:`volsto.multi.lc_kernel.lc_ab`).
3. Bandwidth ``h = max(c·σ_ATM,B(t_{j+1})·√t_{j+1}·N^{−1/5}, h_min)`` (the leverage's rule).
4. ``Ê[a | k]`` and ``Ê[b | k]`` on the ``λ`` grid (:func:`conditional_expectations_ab`: the
   leverage's estimators, unchanged).
5. The new row, clipped; where ``Ê[b|k] ≤ 1e-12·Ê[a|k]`` (one name, or ``R_high = R_low`` on
   the support) the row is 0 and the slice is flagged unidentified.

**Step-averaged target.**  ``λ(t_j)`` governs the whole of step ``j``, so every variance in the
condition is the average over ``[t_j, t_{j+1}]`` at the start state
(``LocalVolSurface.var_time_average``; the last row uses a virtual step of the last length) —
for the names and for the index alike (``target_average = "step"``; ``"point"`` uses the values
at ``t_j``, a diagnostic).

Also here: the index repricing report (:func:`reprice_index_smile`), the constant-correlation
companion (:func:`calibrate_constant_lambda`: the constant ``λ`` that reprices the index
at-the-money-forward straddle, on the kernel and the draws of the calibrated model, so that
every LC − CC difference is a paired estimate) and the two-parameter parametric family
(:func:`calibrate_parametric_lambda`: the reference implementation's Newton iteration on two
index implied vols).

Every calibrating function calls :func:`volsto.calibration.guard.check_calibration_allowed` as
its first statement.  Checked by ``tests/test_local_correlation.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import ndtri

from volsto.calibration.binned import binned_regression
from volsto.calibration.guard import check_calibration_allowed
from volsto.calibration.particle import _finish_estimate, kernel_regression
from volsto.config import LocalCorrelationConfig, ParametricLambdaConfig, ParticleConfig, SimConfig
from volsto.engine.grid import TimeGrid
from volsto.market.bs import black_vega, implied_vol
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import ImpliedSurface
from volsto.models.localvol import LocalVol
from volsto.models.lsv import scheme_mode
from volsto.multi.family import CorrelationFamily
from volsto.multi.lc_draws import LocalCorrelationDraws
from volsto.multi.lc_function import LocalCorrelationFunction, ParametricLambda
from volsto.multi.lc_kernel import lc_ab, lc_diffuse_block
from volsto.multi.lc_model import BasketSpec, LocalCorrelationModel, block_tables, grid_arrays

FloatArray = NDArray[np.float64]
log = logging.getLogger(__name__)

#: Bumped whenever a change can move a calibrated ``λ``; part of the cache key (SPEC §8.7).
#: Guarded by ``tests/test_local_correlation.py::test_lc_code_tag_guard`` (source hash of the
#: modules of ``lc_cache.LC_GUARDED_MODULES``).
LC_CODE_TAG: Final[str] = "lc1"
#: ``Ê[b|k] ≤ UNIDENTIFIED_RATIO·Ê[a|k]``: ``λ`` does not move the basket variance there.
UNIDENTIFIED_RATIO: Final[float] = 1e-12
#: The at-the-money vol of the bandwidth rule is read at ``max(t, 1/365)`` (the leverage's rule).
MIN_ATM_TIME: Final[float] = 1.0 / 365.0
#: Tolerance of the constant-``λ`` secant, in implied vol (0.00002 vol points; the reference's).
CONSTANT_LAMBDA_TOL: Final[float] = 2e-7
#: Default strikes of the index repricing report, in at-the-money standard deviations.
SD_MULTIPLES: Final[tuple[float, ...]] = (
    -2.5,
    -2.0,
    -1.5,
    -1.0,
    -0.5,
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
)


class ClippedMassError(ValueError):
    """``clip_policy = "raise"``: the clipped mass of a slice exceeded ``max_clipped_mass``."""


# --------------------------------------------------------------------------------------------
# the regression of a and b on the basket log-moneyness
# --------------------------------------------------------------------------------------------


def conditional_expectations_ab(
    k: FloatArray,
    a: FloatArray,
    b: FloatArray,
    grid: FloatArray,
    h: float,
    cfg: ParticleConfig,
    tail_slopes: tuple[float | None, float | None] = (None, None),
) -> tuple[FloatArray, FloatArray, float, float]:
    """``(Ê[a | k], Ê[b | k], q_lo, q_hi)`` on ``grid``: the leverage's estimator
    (:func:`volsto.calibration.particle.conditional_variance_estimate`) applied to the two
    responses on the same particles.

    *Sorted* (``cfg.estimator`` ``None`` or ``"sorted"``): one stable sort of ``k``, the nodes
    ``linspace(q_lo, q_hi, n_regression_points)`` on the trusted quantiles with the index rule
    of the leverage's sorted regression, the window ``max(min_window, min_window_fraction·N)``,
    :func:`~volsto.calibration.particle.kernel_regression` once per response, then the shared
    post-processing ``particle._finish_estimate`` (bad nodes, the ``½h²m″`` correction, the
    interpolation onto ``grid``, the tails of ``cfg.tail_extrapolation``).  *Binned*
    (``cfg.estimator = "binned"``): :func:`~volsto.calibration.binned.binned_regression` per
    response, then the same post-processing.

    Bit-identical to two calls of ``conditional_variance_estimate``
    (``tests/test_local_correlation.py::test_regression_helper_equals_two_estimates``).  A
    degenerate cloud (one point) gives the plain means and ``q_lo = q_hi``.  ``a`` and ``b`` are
    positive conditional "variances" (``b > 0`` for two names or more under the family's
    validation), so the positivity-based post-processing applies; a response that is zero
    everywhere (one name) comes back as zeros.
    """
    if cfg.estimator == "binned":
        ra = binned_regression(k, a, h, cfg)
        rb = binned_regression(k, b, h, cfg)
        if ra is None or rb is None:
            point = float(k[0])
            return (
                np.full(grid.shape, float(a.mean())),
                np.full(grid.shape, float(b.mean())),
                point,
                point,
            )
        kreg, q_lo, q_hi, ma, sa = ra
        _, _, _, mb, sb = rb
    else:
        order = np.argsort(k, kind="stable")
        ks = np.ascontiguousarray(k[order])
        n = ks.size
        q_lo = float(ks[min(int(cfg.quantile_clip * n), n - 1)])
        q_hi = float(ks[max(int((1.0 - cfg.quantile_clip) * n) - 1, 0)])
        if not q_hi > q_lo:
            return (
                np.full(grid.shape, float(a.mean())),
                np.full(grid.shape, float(b.mean())),
                q_lo,
                q_lo,
            )
        kreg = np.linspace(q_lo, q_hi, cfg.n_regression_points)
        window = max(cfg.min_window, int(cfg.min_window_fraction * n))
        gaussian, linear = cfg.kernel == "gaussian", cfg.regression == "local_linear"
        ma, sa, _ = kernel_regression(
            ks, np.ascontiguousarray(a[order]), kreg, h, gaussian, linear, window
        )
        mb, sb, _ = kernel_regression(
            ks, np.ascontiguousarray(b[order]), kreg, h, gaussian, linear, window
        )
    ea = _finish_estimate(ma, sa, kreg, q_lo, q_hi, grid, h, cfg, a, tail_slopes[0])
    eb = _finish_estimate(mb, sb, kreg, q_lo, q_hi, grid, h, cfg, b, tail_slopes[1])
    return ea, eb, float(q_lo), float(q_hi)


# --------------------------------------------------------------------------------------------
# the result
# --------------------------------------------------------------------------------------------


@dataclass
class LCCalibrationResult:
    """Output of :func:`calibrate_local_correlation`.

    Per slice ``j`` (the row ``λ(t_j, ·)``, built from the cloud at ``t_j``): the trusted range
    ``q_lo[j]``, ``q_hi[j]``; the clipped masses ``clipped_low[j] = #{p: λ*(t_j, k_p) < 0}/N`` and
    ``clipped_high[j] = #{p: λ*(t_j, k_p) > λ_max}/N`` with their mass-weighted overshoots
    ``E[(λ* − clip λ*)·1_clipped]``; the mean of ``λ`` over the particles; the cloud means of
    ``a`` and ``b``; the unidentified flag; in carry mode the mean of ``|δ_t|`` (the basket
    drift the calibration ignores), ``None`` otherwise.  ``bandwidths[j − 1]`` is the bandwidth
    of slice ``j ≥ 1``.  ``lambda_star`` is the unclipped table.  With a second pass ``lam`` and
    ``lambda_star`` are the averages and the diagnostics are the last pass's.

    How to read the clipped masses: ``clipped_high > 0`` — the index is more volatile there than
    the family can deliver (the single-name vols are too low for that index level; typically
    the far downside); ``clipped_low > 0`` — the index is less volatile than ``R_low`` gives
    (typically the upside of a steep index skew).
    """

    lam: LocalCorrelationFunction
    grid: TimeGrid
    final_log_spot: FloatArray
    bandwidths: FloatArray
    q_lo: FloatArray
    q_hi: FloatArray
    clipped_low: FloatArray
    clipped_high: FloatArray
    lambda_mean: FloatArray
    unidentified: NDArray[np.bool_]
    drift_abs_mean: FloatArray | None
    wall_time: float
    passes: int
    snapshots: dict[float, FloatArray] = field(default_factory=dict)
    lambda_star: FloatArray = field(default_factory=lambda: np.empty((0, 0)))
    overshoot_low: FloatArray = field(default_factory=lambda: np.empty(0))
    overshoot_high: FloatArray = field(default_factory=lambda: np.empty(0))
    mean_a: FloatArray = field(default_factory=lambda: np.empty(0))
    mean_b: FloatArray = field(default_factory=lambda: np.empty(0))
    lambda_max: float = 1.0
    timings: dict[str, float] = field(default_factory=dict)

    @property
    def max_clipped_mass(self) -> float:
        """The largest clipped mass (low or high) over the slices."""
        return float(max(self.clipped_low.max(), self.clipped_high.max()))

    def clip_intervals(self, j: int) -> list[tuple[str, float, float]]:
        """The ``k``-intervals of the trusted range of slice ``j`` where the clip binds:
        ``("low" | "high", k_from, k_to)`` on the ``λ`` grid."""
        k = self.lam.k_grid
        inside = (k >= self.q_lo[j]) & (k <= self.q_hi[j])
        out: list[tuple[str, float, float]] = []
        for label, mask in (
            ("low", (self.lambda_star[j] < 0.0) & inside),
            ("high", (self.lambda_star[j] > self.lambda_max) & inside),
        ):
            idx = np.flatnonzero(mask)
            if idx.size == 0:
                continue
            breaks = np.flatnonzero(np.diff(idx) > 1)
            starts = np.concatenate(([0], breaks + 1))
            ends = np.concatenate((breaks, [idx.size - 1]))
            out += [(label, float(k[idx[s]]), float(k[idx[e]])) for s, e in zip(starts, ends)]
        return out

    def summary(self) -> dict[str, Any]:
        """The JSON-able diagnostics of the calibration (stored with a cache entry)."""
        t = self.grid.times
        worst_lo, worst_hi = int(np.argmax(self.clipped_low)), int(np.argmax(self.clipped_high))
        return {
            "n_slices": int(t.size),
            "horizon": float(t[-1]),
            "passes": self.passes,
            "wall_time": self.wall_time,
            "timings": dict(self.timings),
            "lambda_max": self.lambda_max,
            "max_clipped_mass": self.max_clipped_mass,
            "max_clipped_low": float(self.clipped_low.max()),
            "max_clipped_low_time": float(t[worst_lo]),
            "max_clipped_high": float(self.clipped_high.max()),
            "max_clipped_high_time": float(t[worst_hi]),
            "mean_clipped_low": float(self.clipped_low.mean()),
            "mean_clipped_high": float(self.clipped_high.mean()),
            "lambda_mean_first": float(self.lambda_mean[0]),
            "lambda_mean_last": float(self.lambda_mean[-1]),
            "unidentified_slices": int(self.unidentified.sum()),
            "drift_abs_mean_max": (
                None if self.drift_abs_mean is None else float(self.drift_abs_mean.max())
            ),
            "clipped_low": [float(x) for x in self.clipped_low],
            "clipped_high": [float(x) for x in self.clipped_high],
            "lambda_mean": [float(x) for x in self.lambda_mean],
            "q_lo": [float(x) for x in self.q_lo],
            "q_hi": [float(x) for x in self.q_hi],
            "bandwidths": [float(x) for x in self.bandwidths],
            "drift_abs_mean": (
                None if self.drift_abs_mean is None else [float(x) for x in self.drift_abs_mean]
            ),
        }

    def __repr__(self) -> str:
        return (
            f"LCCalibrationResult({self.lam!r}, n_steps={self.grid.n_steps}, passes={self.passes}, "
            f"max clipped mass {self.max_clipped_mass:.4f}, wall_time={self.wall_time:.1f}s)"
        )


def lambda_surface_frame(result: LCCalibrationResult) -> pd.DataFrame:
    """The plot data of a calibration, one row per ``(t, k)``: ``t``, ``k``, ``lam`` (clipped),
    ``lam_star`` (unclipped), ``in_trusted_range``, ``q_lo``, ``q_hi`` and the slice's clipped
    masses ``m_low``, ``m_high`` (stored with a cache entry as ``lambda_surface.parquet``)."""
    lam = result.lam
    n_t, n_k = lam.values.shape
    t = np.repeat(lam.times, n_k)
    k = np.tile(lam.k_grid, n_t)
    q_lo = np.repeat(result.q_lo, n_k)
    q_hi = np.repeat(result.q_hi, n_k)
    return pd.DataFrame(
        {
            "t": t,
            "k": k,
            "lam": lam.values.ravel(),
            "lam_star": result.lambda_star.ravel(),
            "in_trusted_range": (k >= q_lo) & (k <= q_hi),
            "q_lo": q_lo,
            "q_hi": q_hi,
            "m_low": np.repeat(result.clipped_low, n_k),
            "m_high": np.repeat(result.clipped_high, n_k),
        }
    )


# --------------------------------------------------------------------------------------------
# the particle calibration
# --------------------------------------------------------------------------------------------


def _check_inputs(
    models: Sequence[LocalVol],
    family: CorrelationFamily,
    basket: BasketSpec,
    index_lv: LocalVolSurface,
    cfg: ParticleConfig,
    lc: LocalCorrelationConfig,
    grid: TimeGrid,
) -> LocalCorrelationModel:
    """Validate the skeleton (the model's own checks: one shared grid, sizes) and what the
    calibration adds; returns the skeleton model at ``λ ≡ 0``."""
    if cfg.tail_extrapolation == "sv_slope":
        raise ValueError(
            "tail_extrapolation='sv_slope' is the leverage's (a stochastic-vol kernel's slope): "
            "not defined for the local correlation regressions"
        )
    if basket.mode != lc.mode:
        raise ValueError(
            f"the basket state is in {basket.mode!r} mode and the configuration in {lc.mode!r}"
        )
    zero = LocalCorrelationFunction.constant(0.0, grid.times, index_lv.k_grid)
    skeleton = LocalCorrelationModel(models, family, zero, basket)
    need = float(grid.times[-1] + grid.dts[-1]) - 1e-9
    for name, lv in [("the index target", index_lv)] + [
        (f"asset {i}", m.local_vol) for i, m in enumerate(models)
    ]:
        if lv.t_grid[-1] < need:
            raise ValueError(
                f"{name}: the local-vol grid ends at {lv.t_grid[-1]:.6g}y, before the horizon plus "
                f"one step ({need:.6g}y) that the last row's step average needs"
            )
    return skeleton


def calibrate_local_correlation(
    models: Sequence[LocalVol],
    family: CorrelationFamily,
    basket: BasketSpec,
    index_surface: ImpliedSurface,
    index_lv: LocalVolSurface,
    cfg: ParticleConfig,
    sim: SimConfig,
    lc: LocalCorrelationConfig,
    *,
    snapshot_times: Sequence[float] | None = None,
    draws: LocalCorrelationDraws | None = None,
) -> LCCalibrationResult:
    """Calibrate ``λ(t, k)`` so that the basket of ``models`` reprices the index target (module
    docstring).

    ``models``, ``family`` and ``basket`` are the model's skeleton.  The index target is two
    objects: ``index_surface`` (the implied surface in the basket's forward moneyness: its
    at-the-money vol sets the bandwidth) and ``index_lv`` (its Dupire local variance
    ``σ_B²(t, k)``, whose ``k`` grid becomes the ``λ`` grid).  ``cfg`` holds the particle
    settings (the leverage's fields ``leverage_std_span``, ``leverage_dk``, ``l_min``, ``l_max``
    are not read), ``sim`` the step schedule and the scheme shared with pricing, ``lc`` the tail
    rule, the target average and the clip policy.

    ``snapshot_times``: the particle cloud of the last pass (log-spots, ``(N, n)``) is kept at
    the grid time nearest each requested time (a diagnostic).  ``draws``: the first pass's draws
    (default: ``LocalCorrelationDraws(cfg.seed, …)``) — the Δt check passes the coarsening of a
    finer calibration's draws.

    Refused with :class:`~volsto.calibration.guard.CalibrationForbiddenError` while calibration
    is forbidden: the check below must stay the first statement.
    """
    check_calibration_allowed("calibrate_local_correlation")
    t_start = time.perf_counter()
    T = cfg.horizon
    grid = TimeGrid.build([T], sim.dt_max)
    skeleton = _check_inputs(models, family, basket, index_lv, cfg, lc, grid)
    scheme = sim.scheme
    mode = scheme_mode(scheme)
    times = grid.times
    n_steps = grid.n_steps
    n_slices = n_steps + 1
    n = len(models)
    N = cfg.n_particles
    weights = basket.weights
    lv0 = models[0].local_vol
    k_grid = index_lv.k_grid
    lam_k0, lam_dk = index_lv.k0, index_lv.dk
    lam_max = family.lambda_max
    rho_low = family.rho_low if family.rho_low is not None else 0.0
    ln_f, drifts, shifts, ln_fb = grid_arrays(models, basket, times)
    sig_atm = np.asarray(index_surface.atm_vol(np.maximum(times, MIN_ATM_TIME)), dtype=np.float64)
    n_exp = float(N) ** (-0.2)
    step_average = lc.target_average == "step"
    carry = basket.mode == "carry"

    def interval(j: int) -> list[float]:
        """The step row ``j`` governs (the last row: a virtual step of the last length)."""
        end = float(times[j + 1]) if j < n_steps else float(times[j] + grid.dts[-1])
        return [float(times[j]), end]

    def name_rows(j: int) -> FloatArray:
        """Each name's variance on the shared grid for row ``j``: ``(n, n_k)``."""
        if step_average:
            rows = [m.local_vol.var_time_average(interval(j))[0] for m in models]
        else:
            rows = [m.local_vol.var_at_times([times[j]])[0] for m in models]
        return np.ascontiguousarray(np.stack(rows))

    def target_row(j: int) -> FloatArray:
        """The index target variance on the ``λ`` grid for row ``j``."""
        if step_average:
            return np.asarray(index_lv.var_time_average(interval(j))[0], dtype=np.float64)
        return np.asarray(index_lv.var_at_times([times[j]])[0], dtype=np.float64)

    snap_idx: dict[int, float] = {}
    for ts in snapshot_times or ():
        j_snap = int(np.argmin(np.abs(times - float(ts))))
        snap_idx[j_snap] = float(times[j_snap])
    snapshots: dict[float, FloatArray] = {}

    no_record = np.array([-1], dtype=np.int64)
    dummy3 = np.empty((1, 1, 1))
    dummy2 = np.empty((1, 1))
    n_pass = 2 if cfg.second_pass else 1
    tables_l: list[FloatArray] = []
    tables_star: list[FloatArray] = []
    timings = {
        "tables": 0.0,
        "draws": 0.0,
        "kernel": 0.0,
        "ab": 0.0,
        "regression": 0.0,
        "rows": 0.0,
    }
    bandwidths = np.empty(n_steps)
    q_lo, q_hi = np.empty(n_slices), np.empty(n_slices)
    clipped_low, clipped_high = np.empty(n_slices), np.empty(n_slices)
    over_low, over_high = np.empty(n_slices), np.empty(n_slices)
    lambda_mean = np.empty(n_slices)
    mean_a, mean_b = np.empty(n_slices), np.empty(n_slices)
    unidentified = np.zeros(n_slices, dtype=np.bool_)
    drift_abs = np.zeros(n_slices) if carry else None
    final_ls: FloatArray | None = None
    a, b, kb = np.empty(N), np.empty(N), np.empty(N)

    def new_row(
        j: int, ea: FloatArray, eb: FloatArray, lo: float, hi: float, k_cloud: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Row ``j`` from the regressions and the diagnostics of the slice."""
        target = target_row(j)
        flat = eb <= UNIDENTIFIED_RATIO * ea
        with np.errstate(divide="ignore", invalid="ignore"):
            star = np.where(flat, 0.0, (target - ea) / np.where(flat, 1.0, eb))
        if lc.lambda_tail == "flat" and hi > lo:
            # hold λ* at its values at the ends of the trusted range
            at_lo, at_hi = np.interp([lo, hi], k_grid, star)
            star = np.where(k_grid < lo, at_lo, np.where(k_grid > hi, at_hi, star))
        row = np.clip(star, 0.0, lam_max)
        inside = (k_grid >= lo) & (k_grid <= hi)
        unidentified[j] = bool(np.any(flat[inside])) if np.any(inside) else bool(np.all(flat))
        per = np.interp(k_cloud, k_grid, star)
        below, above = per < 0.0, per > lam_max
        clipped_low[j], clipped_high[j] = float(below.mean()), float(above.mean())
        over_low[j] = float(np.where(below, per, 0.0).mean())
        over_high[j] = float(np.where(above, per - lam_max, 0.0).mean())
        lambda_mean[j] = float(np.clip(per, 0.0, lam_max).mean())
        q_lo[j], q_hi[j] = lo, hi
        if lc.clip_policy == "raise" and max(clipped_low[j], clipped_high[j]) > lc.max_clipped_mass:
            raise ClippedMassError(
                f"local correlation calibration: at t = {times[j]:.6g} the clipped mass is "
                f"{clipped_low[j]:.4f} (low) / {clipped_high[j]:.4f} (high), above "
                f"max_clipped_mass = {lc.max_clipped_mass:g}: the index smile is out of the "
                "family's reach there"
            )
        return row, star

    def first_row(a0: float, b0: float, k_0: float) -> tuple[float, float]:
        """``(λ*, λ)`` of slice 0 from the one state of the cloud, and its diagnostics."""
        target0 = float(np.interp(k_0, k_grid, target_row(0)))
        flat0 = b0 <= UNIDENTIFIED_RATIO * a0
        star0 = 0.0 if flat0 else (target0 - a0) / b0
        value = min(max(star0, 0.0), lam_max)
        unidentified[0] = flat0
        q_lo[0] = q_hi[0] = k_0
        clipped_low[0], clipped_high[0] = float(star0 < 0.0), float(star0 > lam_max)
        over_low[0], over_high[0] = min(star0, 0.0), max(star0 - lam_max, 0.0)
        lambda_mean[0] = value
        if lc.clip_policy == "raise" and (star0 < 0.0 or star0 > lam_max):
            raise ClippedMassError(
                f"local correlation calibration: at t = 0 lambda* = {star0:.6g} is outside "
                f"[0, {lam_max:.6g}]: the index at-the-money variance is out of the family's reach"
            )
        return star0, value

    def cloud_terms(ls: FloatArray, j: int) -> None:
        """``a``, ``b``, ``k_B`` of every particle at ``t_j`` (and the carry diagnostic)."""
        lc_ab(
            ls,
            np.ascontiguousarray(ln_f[:, j]),
            weights,
            np.ascontiguousarray(shifts[:, j]),
            float(ln_fb[j]),
            lv0.k0,
            lv0.dk,
            name_rows(j),
            family.low_equi,
            rho_low,
            family.r_low,
            family.high_ones,
            family.l_high,
            a,
            b,
            kb,
        )
        mean_a[j], mean_b[j] = float(a.mean()), float(b.mean())
        if drift_abs is not None:
            drift_abs[j] = _mean_abs_drift(ls, j)

    def _mean_abs_drift(ls: FloatArray, j: int) -> float:
        """Carry mode: the cloud mean of ``|δ_t|``, ``δ_t = Σ (ω_i − θ_i)·μ_i`` with ``μ_i`` the
        names' carry over the step row ``j`` governs (the last row: the last step's)."""
        jj = min(j, n_steps - 1)
        mu = drifts[:, jj] / grid.dts[jj]
        level = np.exp(ls - shifts[:, j][None, :]) @ weights
        omega = weights[None, :] * np.exp(ls - shifts[:, j][None, :]) / level[:, None]
        theta = weights * np.exp(ln_f[:, j] - shifts[:, j]) / np.exp(ln_fb[j])
        return float(np.mean(np.abs((omega - theta[None, :]) @ mu)))

    for p in range(n_pass):
        seed = cfg.seed + p
        d = (
            draws
            if draws is not None and p == 0
            else LocalCorrelationDraws(seed, N, n_steps, family, cfg.antithetic)
        )
        if (d.n_paths, d.n_steps, d.n_assets) != (N, n_steps, n) or (
            d.family.rank_high != family.rank_high
        ):
            raise ValueError("the draws do not match the particles, the grid or the family")
        ls = np.empty((N, n))
        for i, m in enumerate(models):
            ls[:, i] = np.log(m.spot)
        iv = np.zeros((N, n))
        sq = np.zeros((N, n))
        lam_int = np.zeros(N)
        L = np.empty((n_slices, k_grid.size))
        L_star = np.empty((n_slices, k_grid.size))
        # slice 0: every particle is at the same state, so the whole row is one value,
        # (target(k_0) − a_0) / b_0 at that state, clipped
        cloud_terms(ls, 0)
        L_star[0], L[0] = first_row(float(a[0]), float(b[0]), float(kb[0]))
        for j in range(n_steps):
            t_nodes = times[j : j + 2]
            t1 = float(times[j + 1])
            c0 = time.perf_counter()
            var_a, var_b, var_rec = block_tables(models, t_nodes, scheme, False)
            c1 = time.perf_counter()
            eps = d.eps_block(j, j + 1, 0, N)
            eta = d.eta_block(j, j + 1, 0, N)
            c2 = time.perf_counter()
            lc_diffuse_block(
                ls,
                iv,
                sq,
                lam_int,
                eps,
                eta,
                t_nodes,
                np.ascontiguousarray(ln_f[:, j : j + 2]),
                np.ascontiguousarray(drifts[:, j : j + 1]),
                no_record,
                lv0.k0,
                lv0.dk,
                var_a,
                var_b,
                var_rec,
                mode,
                scheme.pc_eta,
                weights,
                np.ascontiguousarray(shifts[:, j : j + 2]),
                np.ascontiguousarray(ln_fb[j : j + 2]),
                np.ascontiguousarray(L[j][None, :]),
                lam_k0,
                lam_dk,
                lam_max,
                family.low_equi,
                family.d_low,
                family.ell_low,
                family.l_low,
                family.l_high,
                dummy3,
                dummy3,
                dummy3,
                dummy3,
                dummy2,
                dummy2,
                False,
                dummy3,
                0,
            )
            c3 = time.perf_counter()
            del eps, eta
            cloud_terms(ls, j + 1)
            c4 = time.perf_counter()
            h = max(
                cfg.bandwidth_factor * float(sig_atm[j + 1]) * np.sqrt(t1) * n_exp,
                cfg.bandwidth_min,
            )
            bandwidths[j] = h
            slopes: tuple[float | None, float | None] = (None, None)
            if cfg.tail_extrapolation == "cloud_slope":
                slopes = (
                    float(np.polyfit(kb, np.log(np.maximum(a, 1e-300)), 1)[0]),
                    float(np.polyfit(kb, np.log(np.maximum(b, 1e-300)), 1)[0]),
                )
            ea, eb, lo, hi = conditional_expectations_ab(kb, a, b, k_grid, h, cfg, slopes)
            c5 = time.perf_counter()
            L[j + 1], L_star[j + 1] = new_row(j + 1, ea, eb, lo, hi, kb)
            c6 = time.perf_counter()
            for key, dt in zip(timings, (c1 - c0, c2 - c1, c3 - c2, c4 - c3, c5 - c4, c6 - c5)):
                timings[key] += dt
            if j + 1 in snap_idx and p == n_pass - 1:
                snapshots[snap_idx[j + 1]] = ls.copy()
            log.debug(
                "lc step %d/%d t=%.5f h=%.5f q=[%.4f, %.4f] clipped %.4f/%.4f mean lambda %.4f",
                j + 1,
                n_steps,
                t1,
                h,
                lo,
                hi,
                clipped_low[j + 1],
                clipped_high[j + 1],
                lambda_mean[j + 1],
            )
        final_ls = ls
        tables_l.append(L)
        tables_star.append(L_star)
        log.info(
            "local correlation pass %d/%d done (%d steps, N=%d, %d names; max clipped mass "
            "%.4f low / %.4f high)",
            p + 1,
            n_pass,
            n_steps,
            N,
            n,
            float(clipped_low.max()),
            float(clipped_high.max()),
        )
    assert final_ls is not None
    values = tables_l[0] if n_pass == 1 else 0.5 * (tables_l[0] + tables_l[1])
    star = tables_star[0] if n_pass == 1 else 0.5 * (tables_star[0] + tables_star[1])
    wall = time.perf_counter() - t_start
    metadata: dict[str, Any] = {
        "kind": "particle",
        "seed": cfg.seed,
        "n_particles": N,
        "horizon": T,
        "passes": n_pass,
        "code_tag": LC_CODE_TAG,
        "family": family.describe(),
        "mode": basket.mode,
        "names": list(skeleton.names),
        "kernel": cfg.kernel,
        "bandwidth_factor": cfg.bandwidth_factor,
        "tail_extrapolation": cfg.tail_extrapolation,
        "lambda_tail": lc.lambda_tail,
        "target_average": lc.target_average,
        "scheme": scheme.__dict__.copy(),
        "schedule": repr(sim.step_schedule),
        "wall_time": wall,
    }
    if cfg.estimator is not None:
        metadata["estimator"] = cfg.estimator
    lam = LocalCorrelationFunction(times, k_grid, values, metadata)
    log.info(
        "local correlation calibrated in %.1f s (tables %.1f, draws %.1f, kernel %.1f, a/b %.1f, "
        "regression %.1f, rows %.1f)",
        wall,
        *(timings[key] for key in ("tables", "draws", "kernel", "ab", "regression", "rows")),
    )
    return LCCalibrationResult(
        lam=lam,
        grid=grid,
        final_log_spot=final_ls,
        bandwidths=bandwidths,
        q_lo=q_lo,
        q_hi=q_hi,
        clipped_low=clipped_low,
        clipped_high=clipped_high,
        lambda_mean=lambda_mean,
        unidentified=unidentified,
        drift_abs_mean=drift_abs,
        wall_time=wall,
        passes=n_pass,
        snapshots=snapshots,
        lambda_star=star,
        overshoot_low=over_low,
        overshoot_high=over_high,
        mean_a=mean_a,
        mean_b=mean_b,
        lambda_max=lam_max,
        timings=timings,
    )


# --------------------------------------------------------------------------------------------
# the basket's terminal law on fixed draws: implied vols with standard errors
# --------------------------------------------------------------------------------------------


def straddle_vol(price: float, maturity: float) -> float:
    """The lognormal vol of an at-the-money-forward straddle on a forward of 1, inverted
    exactly (derived: the straddle is ``2·(2N(σ√T/2) − 1)``, so ``σ = 2·N⁻¹(½(p/2 + 1))/√T``)."""
    return float(2.0 * ndtri(min(0.5 * (0.5 * price + 1.0), 1.0 - 1e-16)) / np.sqrt(maturity))


def _pair_stats(x: FloatArray, antithetic: bool) -> tuple[float, float]:
    """Mean and standard error of per-path values (antithetic pairs averaged first)."""
    y = 0.5 * (x[0::2] + x[1::2]) if antithetic else x
    return float(y.mean()), float(y.std(ddof=1) / np.sqrt(y.size))


class BasketSampler:
    """The basket's forward-moneyness level ``B_T/F_B(T) = e^{k_B(T)}`` at given maturities for
    any ``λ`` on **fixed draws** (common random numbers across ``λ``): the engine of the
    constant and parametric calibrations and of the index repricing report.  ``model`` supplies
    the names, the family and the basket; ``sim`` the path count, the seed, the schedule, the
    scheme and the chunking."""

    def __init__(
        self,
        model: LocalCorrelationModel,
        sim: SimConfig,
        maturities: Sequence[float],
        *,
        seed: int | None = None,
    ) -> None:
        self.model = model
        self.sim = sim
        self.maturities = tuple(float(t) for t in maturities)
        self.seed = sim.seed if seed is None else int(seed)
        self.grid = TimeGrid.build(
            list(self.maturities), sim.dt_max, calibration_grid=model.required_times()
        )
        self.draws = model.draws_for(self.grid, self.seed, sim.n_paths, sim.antithetic)
        self.columns = [self.grid.fixing_index[t] for t in self.maturities]
        self.n_evaluations = 0

    def levels(self, lam: LocalCorrelationFunction | None = None) -> FloatArray:
        """``e^{k_B}`` at the maturities, ``(n_paths, n_maturities)``, under ``lam`` (default:
        the model's own ``λ``).  ``lam`` must hold the model's slices (the grid is fixed)."""
        m = self.model if lam is None else self.model.with_lambda(lam)
        if lam is not None and not np.all(
            np.isin(lam.times[lam.times <= self.grid.horizon + 1e-12], self.grid.times)
        ):
            raise ValueError("λ has a slice that is not a time of the sampler's grid")
        out = np.empty((self.sim.n_paths, len(self.columns)))
        ranges = self.sim.chunk_ranges(self.grid.n_records * m.n_assets, 0)
        for p0, p1 in ranges:
            paths = m.simulate_chunk(self.grid, self.draws, p0, p1, self.sim.scheme)
            assert paths.aux is not None
            out[p0:p1] = np.exp(paths.aux["k_basket"][:, self.columns])
        self.n_evaluations += 1
        return out

    def straddle(self, level: FloatArray, maturity: float) -> tuple[float, float, float]:
        """``(vol, vol stderr, price)`` of the at-the-money-forward straddle ``E|B/F_B − 1|``."""
        price, se = _pair_stats(np.abs(level - 1.0), self.sim.antithetic)
        vol = straddle_vol(price, maturity)
        vega = 2.0 * float(np.exp(-0.125 * vol * vol * maturity)) * np.sqrt(maturity / (2 * np.pi))
        return vol, se / vega, price

    def otm_vol(
        self, level: FloatArray, strike: float, maturity: float
    ) -> tuple[float, float, float]:
        """``(vol, vol stderr, price)`` of the out-of-the-money option at ``strike`` (a fraction
        of the forward): a put below 1, a call at or above."""
        cp = 1.0 if strike >= 1.0 else -1.0
        price, se = _pair_stats(np.maximum(cp * (level - strike), 0.0), self.sim.antithetic)
        vol = float(implied_vol(price, 1.0, strike, maturity, cp))
        vega = float(black_vega(1.0, strike, maturity, vol)) if np.isfinite(vol) else float("nan")
        return vol, (se / vega if vega > 0 else float("nan")), price

    def strike_vol(
        self, level: FloatArray, strike: float, maturity: float
    ) -> tuple[float, float, float]:
        """The implied vol at ``strike``: the straddle's at exactly 1, the out-of-the-money
        option's elsewhere (the reference's two targets)."""
        if strike == 1.0:
            return self.straddle(level, maturity)
        return self.otm_vol(level, strike, maturity)


# --------------------------------------------------------------------------------------------
# the index repricing report
# --------------------------------------------------------------------------------------------


@dataclass
class IndexRepricingReport:
    """Output of :func:`reprice_index_smile`: per maturity and strike (in at-the-money standard
    deviations) the basket's implied vol under the model against the index target, in vol
    points with its Monte Carlo standard error; per maturity the forward error ``E[B_T]/F_B(T) −
    1`` with its standard error."""

    table: pd.DataFrame
    forwards: pd.DataFrame
    n_paths: int
    seeds: tuple[int, ...]
    wall_time: float

    def violations(
        self,
        tol_inner: float = 0.15,
        sd_inner: float = 1.5,
        tol_outer: float = 0.30,
        sd_outer: float = 2.5,
        z: float = 3.0,
    ) -> pd.DataFrame:
        """The cells inside ``±sd_outer`` whose ``|error|`` exceeds both their tolerance
        (``tol_inner`` vol points inside ``±sd_inner``, ``tol_outer`` beyond) and ``z`` standard
        errors — the convention of ``CalibrationReport.passes`` (SPEC §4.2)."""
        r = self.table[self.table["sd"].abs() <= sd_outer + 1e-12]
        tol = np.where(r["sd"].abs() <= sd_inner + 1e-12, tol_inner, tol_outer)
        bad = (r["error_vp"].abs() > np.maximum(tol, z * r["stderr_vp"])) | r["error_vp"].isna()
        return r[bad]

    def passes(
        self,
        tol_inner: float = 0.15,
        sd_inner: float = 1.5,
        tol_outer: float = 0.30,
        sd_outer: float = 2.5,
        z: float = 3.0,
    ) -> bool:
        return self.violations(tol_inner, sd_inner, tol_outer, sd_outer, z).empty

    def max_abs_error(self, sd_max: float = 1.5) -> float:
        """Largest ``|error|`` in vol points inside ``±sd_max``."""
        r = self.table[self.table["sd"].abs() <= sd_max + 1e-12]
        return float(r["error_vp"].abs().max())

    def pivot(self, value: str = "error_vp") -> pd.DataFrame:
        return self.table.pivot(index="T", columns="sd", values=value)

    def summary(self) -> str:
        err, se = self.pivot("error_vp"), self.pivot("stderr_vp")
        cells = err.copy().astype(object)
        for i in err.index:
            for j in err.columns:
                cells.loc[i, j] = f"{err.loc[i, j]:+.3f}±{se.loc[i, j]:.3f}"
        head = (
            f"index implied-vol error (vol points ± MC se) by maturity and strike in ATM sd, "
            f"n_paths={self.n_paths} x {len(self.seeds)} seeds, wall {self.wall_time:.1f}s; "
            f"max |err| within 1.5 sd {self.max_abs_error(1.5):.3f}, within 2.5 sd "
            f"{self.max_abs_error(2.5):.3f}; violations {len(self.violations())}"
        )
        return head + "\n" + cells.to_string() + "\n" + self.forwards.to_string(index=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "table": self.table.to_dict(orient="list"),
            "forwards": self.forwards.to_dict(orient="list"),
            "n_paths": self.n_paths,
            "seeds": list(self.seeds),
            "wall_time": self.wall_time,
            "max_abs_error_vp_1p5sd": self.max_abs_error(1.5),
            "max_abs_error_vp_2p5sd": self.max_abs_error(2.5),
        }


def reprice_index_smile(
    model: LocalCorrelationModel,
    index_surface: ImpliedSurface,
    sim: SimConfig,
    *,
    maturities: Sequence[float],
    sd_multiples: Sequence[float] = SD_MULTIPLES,
    pricing_seeds: Sequence[int] | None = None,
) -> IndexRepricingReport:
    """Out-of-the-money options on the model's basket against the index target.

    At each maturity ``T`` and strike ``k = m·σ_ATM,B(T)·√T`` (``m`` in ``sd_multiples``): a put
    for ``k < 0`` and a call for ``k ≥ 0`` on ``e^{k_B(T)}`` against ``e^{k}``, in forward units
    (read from ``aux["k_basket"]``); the price is inverted with
    :func:`volsto.market.bs.implied_vol` (forward 1, no discounting) and its standard error
    goes through the vega; the error is against ``index_surface.implied_vol_k(k, T)``.  The
    samples of several ``pricing_seeds`` (default: ``sim.seed`` alone) are pooled.  No
    calibration: the model is priced as given."""
    t0 = time.perf_counter()
    mats = [float(t) for t in maturities]
    seeds = tuple(int(s) for s in (pricing_seeds if pricing_seeds is not None else (sim.seed,)))
    sds = [float(m) for m in sd_multiples]
    atm = [float(index_surface.atm_vol(t)) for t in mats]
    ks = np.array([[m * atm[i] * np.sqrt(t) for m in sds] for i, t in enumerate(mats)])
    strikes = np.exp(ks)
    cps = np.where(ks >= 0.0, 1.0, -1.0)
    anti = sim.antithetic
    n_ind = (sim.n_paths // 2 if anti else sim.n_paths) * len(seeds)
    s1 = np.zeros(ks.shape)
    s2 = np.zeros(ks.shape)
    f1 = np.zeros(len(mats))
    f2 = np.zeros(len(mats))
    for seed in seeds:
        sampler = BasketSampler(model, sim, mats, seed=seed)
        level = sampler.levels()
        for i in range(len(mats)):
            x = level[:, i]
            fwd = 0.5 * (x[0::2] + x[1::2]) if anti else x
            f1[i] += float(fwd.sum())
            f2[i] += float(np.sum(fwd * fwd))
            for jj in range(len(sds)):
                pay = np.maximum(cps[i, jj] * (x - strikes[i, jj]), 0.0)
                y = 0.5 * (pay[0::2] + pay[1::2]) if anti else pay
                s1[i, jj] += float(y.sum())
                s2[i, jj] += float(np.sum(y * y))
    price = s1 / n_ind
    price_se = np.sqrt(np.maximum(s2 / n_ind - price * price, 0.0) / (n_ind - 1))
    rows = []
    for i, t in enumerate(mats):
        for jj, m in enumerate(sds):
            vol = float(implied_vol(price[i, jj], 1.0, strikes[i, jj], t, cps[i, jj]))
            vega = float(black_vega(1.0, strikes[i, jj], t, vol)) if np.isfinite(vol) else np.nan
            target = float(index_surface.implied_vol_k(ks[i, jj], t))
            rows.append(
                {
                    "T": t,
                    "sd": m,
                    "k": float(ks[i, jj]),
                    "model_vol": vol,
                    "target_vol": target,
                    "error_vp": 100.0 * (vol - target),
                    "stderr_vp": 100.0 * price_se[i, jj] / vega if vega and vega > 0 else np.nan,
                    "price": float(price[i, jj]),
                    "price_se": float(price_se[i, jj]),
                }
            )
    mean_f = f1 / n_ind
    se_f = np.sqrt(np.maximum(f2 / n_ind - mean_f * mean_f, 0.0) / (n_ind - 1))
    forwards = pd.DataFrame({"T": mats, "forward_error": mean_f - 1.0, "forward_error_se": se_f})
    return IndexRepricingReport(
        pd.DataFrame(rows), forwards, sim.n_paths, seeds, time.perf_counter() - t0
    )


# --------------------------------------------------------------------------------------------
# the constant-correlation companion
# --------------------------------------------------------------------------------------------


def _secant(
    f: Callable[[float], float],
    x0: float,
    x1: float,
    lo: float,
    hi: float,
    tol: float,
    maxit: int,
) -> tuple[float, float, list[tuple[float, float]]]:
    """Secant iteration on ``f`` clipped to ``[lo, hi]`` (the reference's ``calibrate_cc``):
    ``(x, f(x), history)``; stops when ``|f| < tol`` or the iterate stops moving."""
    f0, f1 = f(x0), f(x1)
    hist = [(x0, f0), (x1, f1)]
    for _ in range(maxit):
        if abs(f1) < tol or f1 == f0:
            break
        x2 = min(max(x1 - f1 * (x1 - x0) / (f1 - f0), lo), hi)
        if x2 == x1:
            break
        x0, f0 = x1, f1
        x1, f1 = x2, f(x2)
        hist.append((x1, f1))
    return x1, f1, hist


def _constant_lambda_for(
    sampler: BasketSampler,
    target_vol: float,
    strike: float,
    maturity: float,
    start: float,
    tol: float,
    maxit: int = 20,
) -> tuple[float, float, list[tuple[float, float]]]:
    """The constant ``λ`` at which the basket's implied vol at ``strike`` equals ``target_vol``
    on the sampler's draws (secant, clipped to ``[0, λ_max]``)."""
    model = sampler.model
    cap = model.family.lambda_max
    col = sampler.maturities.index(float(maturity))

    def gap(value: float) -> float:
        lam = LocalCorrelationFunction.constant(value, model.lam.times, model.lam.k_grid)
        level = sampler.levels(lam)[:, col]
        return sampler.strike_vol(level, strike, maturity)[0] - target_vol

    x0 = min(max(start, 0.0), cap)
    x1 = x0 + 0.02 * cap if x0 + 0.02 * cap <= cap else x0 - 0.02 * cap
    return _secant(gap, x0, x1, 0.0, cap, tol, maxit)


def _lambda_at_inception(model: LocalCorrelationModel, target_var: float) -> float:
    """The exact ``λ`` at ``t_0`` for an index variance ``target_var`` (the ``t_0`` formula of
    the particle calibration on instantaneous variances): a starting value."""
    w = model.basket.weights
    ls = np.log(model.spots)
    level = float(np.exp(ls - model.basket.shifts([0.0])[:, 0]) @ w)
    omega = w * np.exp(ls - model.basket.shifts([0.0])[:, 0]) / level
    sig = np.sqrt(
        [float(m.local_vol.local_var_k(m.local_vol.t_grid[0], 0.0)) for m in model.models]
    )
    a, b = model.family.variance_terms(omega * sig)
    if float(b) <= UNIDENTIFIED_RATIO * float(a):
        return 0.0
    return float(np.clip((target_var - float(a)) / float(b), 0.0, model.family.lambda_max))


def calibrate_constant_lambda(
    model: LocalCorrelationModel,
    index_surface: ImpliedSurface,
    maturity: float,
    sim: SimConfig,
    *,
    tol: float = CONSTANT_LAMBDA_TOL,
) -> float:
    """The constant ``λ_c`` that reprices the index at-the-money-forward straddle at
    ``maturity`` (secant on common random numbers: ``sim``'s paths and seed; tolerance 2e-7 in
    implied vol).  The constant-correlation model is ``model.with_lambda(
    LocalCorrelationFunction.constant(λ_c, model.lam.times, model.lam.k_grid))``: the kernel and
    the ``ε`` and ``η`` of ``model``, so every LC − CC difference is a paired estimate.  For the
    equicorrelation family it is the constant correlation ``ρ_c = ρ_min + λ_c(1 − ρ_min)`` — the
    dispersion study's correlation marked to the index straddle, on a diffusion.

    Refused while calibration is forbidden: the check below must stay the first statement."""
    check_calibration_allowed("calibrate_constant_lambda")
    target = float(index_surface.atm_vol(maturity))
    sampler = BasketSampler(model, sim, [maturity])
    start = _lambda_at_inception(model, target * target)
    value, gap, hist = _constant_lambda_for(sampler, target, 1.0, float(maturity), start, tol)
    if abs(gap) >= tol:
        log.warning(
            "constant lambda: the secant stopped at lambda = %.8f with an ATM vol error of %.6f "
            "vol points (%d evaluations): the target is out of the family's reach",
            value,
            100.0 * gap,
            len(hist),
        )
    log.info(
        "constant lambda %.8f (ATM vol error %.6f vp, %d evaluations)",
        value,
        100.0 * gap,
        len(hist),
    )
    return float(value)


# --------------------------------------------------------------------------------------------
# the parametric family
# --------------------------------------------------------------------------------------------


@dataclass
class ParametricFit:
    """Output of :func:`calibrate_parametric_lambda`: the two parameters iterated on (``rho0``,
    ``c`` of the reference's form for the equicorrelation family; ``λ0`` and the slope
    otherwise), the ``λ`` parameters, the Jacobian of the two implied vols, the history
    ``[(parameters, residuals)]``, the converged flag, and the achieved implied vols with their
    standard errors and targets."""

    rho0: float
    c: float
    lam: ParametricLambda
    jacobian: FloatArray
    history: list[tuple[list[float], list[float]]]
    converged: bool
    strikes: tuple[float, ...]
    vols: tuple[float, ...]
    vol_stderrs: tuple[float, ...]
    targets: tuple[float, ...]
    start: tuple[float, float]
    n_evaluations: int
    wall_time: float

    def function(self, times: Any, k_grid: Any) -> LocalCorrelationFunction:
        """The fitted ``λ`` tabulated on a grid."""
        return LocalCorrelationFunction.parametric(self.lam, times, k_grid)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rho0": self.rho0,
            "c": self.c,
            "lambda0": self.lam.lambda0,
            "slope": self.lam.slope,
            "lam_lo": self.lam.lam_lo,
            "lam_hi": self.lam.lam_hi,
            "jacobian": self.jacobian.tolist(),
            "history": self.history,
            "converged": self.converged,
            "strikes": list(self.strikes),
            "vols": list(self.vols),
            "vol_stderrs": list(self.vol_stderrs),
            "targets": list(self.targets),
            "start": list(self.start),
            "n_evaluations": self.n_evaluations,
            "wall_time": self.wall_time,
        }


def calibrate_parametric_lambda(
    model: LocalCorrelationModel,
    index_surface: ImpliedSurface,
    maturity: float,
    cfg: ParametricLambdaConfig,
    sim: SimConfig,
    *,
    seed: int,
) -> ParametricFit:
    """The two-parameter ``λ(t, k) = clip(λ0 − slope·k, 0, λ_max)`` that reprices two index
    implied vols at ``maturity`` — the reference implementation's algorithm (``lcm_lib.
    calibrate_lc``).

    * Targets: ``index_surface``'s implied vols at ``cfg.strikes`` (fractions of the basket
      forward; default the at-the-money-forward straddle and the 90 % put).
    * Residuals: model minus target implied vols on common random numbers (``cfg.n_paths``
      paths of seed ``seed`` — the calibration seed, ``particle.seed`` — with ``sim``'s schedule
      and scheme).  The straddle vol is inverted exactly (:func:`straddle_vol`).
    * Newton in two dimensions on ``(ρ0, c)`` for the equicorrelation family
      (``ParametricLambda.from_rho``; ``(λ0, slope)`` otherwise) with a forward-difference
      Jacobian, steps ``cfg.h``, recomputed on the first ``cfg.fd_iters`` iterations and then
      held; damping ``min(1, 0.25/|Δρ0|, 3/|Δc|)``; stop at ``max |residual| < cfg.tol``, at most
      ``cfg.maxit`` iterations.
    * Start: ``ρ0 = ρ_ATM`` and ``c = 2(ρ_K − ρ_ATM)/ln(1/K)`` with ``ρ_ATM`` and ``ρ_K`` the
      constant correlations that reprice each target alone (secants on the same draws).

    Refused while calibration is forbidden: the check below must stay the first statement."""
    check_calibration_allowed("calibrate_parametric_lambda")
    t0 = time.perf_counter()
    fam = model.family
    equi = fam.rho_low is not None and fam.high_ones
    rho_min = float(fam.rho_low) if fam.rho_low is not None and equi else 0.0
    span = 1.0 - rho_min
    run = dataclasses.replace(sim, n_paths=cfg.n_paths, seed=int(seed))
    sampler = BasketSampler(model, run, [maturity])
    strikes = tuple(float(s) for s in cfg.strikes)
    targets = tuple(float(index_surface.implied_vol_k(np.log(s), maturity)) for s in strikes)
    times, k_grid = model.lam.times, model.lam.k_grid

    def to_lambda(theta: FloatArray) -> ParametricLambda:
        """``(ρ0, c)`` (or ``(λ0, slope)``) as the ``λ`` parameters."""
        return ParametricLambda((theta[0] - rho_min) / span, theta[1] / span, 0.0, fam.lambda_max)

    def evaluate(theta: FloatArray) -> tuple[FloatArray, list[tuple[float, float, float]]]:
        level = sampler.levels(LocalCorrelationFunction.parametric(to_lambda(theta), times, k_grid))
        cells = [sampler.strike_vol(level[:, 0], s, maturity) for s in strikes]
        return np.array([cells[i][0] - targets[i] for i in range(2)]), cells

    # start: the constant correlations that reprice each target alone
    first = _lambda_at_inception(model, targets[0] * targets[0])
    lam_atm, _, _ = _constant_lambda_for(
        sampler, targets[0], strikes[0], float(maturity), first, 1e-6
    )
    lam_k, _, _ = _constant_lambda_for(
        sampler, targets[1], strikes[1], float(maturity), lam_atm, 1e-6
    )
    rho_atm, rho_k = rho_min + lam_atm * span, rho_min + lam_k * span
    theta = np.array([rho_atm, 2.0 * (rho_k - rho_atm) / np.log(1.0 / strikes[1])])
    start = (float(theta[0]), float(theta[1]))
    jac = np.zeros((2, 2))
    have_jac = False
    hist: list[tuple[list[float], list[float]]] = []
    converged = False
    cells: list[tuple[float, float, float]] = []
    for it in range(cfg.maxit):
        res, cells = evaluate(theta)
        hist.append((theta.tolist(), res.tolist()))
        log.debug(
            "parametric lambda iter %d: (%.6f, %.6f), vol errors %.6f / %.6f vp",
            it,
            theta[0],
            theta[1],
            100 * res[0],
            100 * res[1],
        )
        if float(np.abs(res).max()) < cfg.tol:
            converged = True
            break
        if it == cfg.maxit - 1:
            break
        if not have_jac or it < cfg.fd_iters:
            ra, _ = evaluate(theta + np.array([cfg.h[0], 0.0]))
            rb, _ = evaluate(theta + np.array([0.0, cfg.h[1]]))
            jac = np.column_stack([(ra - res) / cfg.h[0], (rb - res) / cfg.h[1]])
            have_jac = True
        step = np.linalg.solve(jac, res)
        damp = min(1.0, 0.25 / max(abs(step[0]), 1e-12), 3.0 / max(abs(step[1]), 1e-12))
        theta = theta - damp * step
    if not converged:
        log.warning(
            "parametric lambda did not converge in %d iterations (vol errors %.6f / %.6f vp)",
            cfg.maxit,
            100 * hist[-1][1][0],
            100 * hist[-1][1][1],
        )
    return ParametricFit(
        rho0=float(theta[0]),
        c=float(theta[1]),
        lam=to_lambda(theta),
        jacobian=jac,
        history=hist,
        converged=converged,
        strikes=strikes,
        vols=tuple(c[0] for c in cells),
        vol_stderrs=tuple(c[1] for c in cells),
        targets=targets,
        start=start,
        n_evaluations=sampler.n_evaluations,
        wall_time=time.perf_counter() - t0,
    )


__all__ = [
    "LC_CODE_TAG",
    "SD_MULTIPLES",
    "BasketSampler",
    "ClippedMassError",
    "IndexRepricingReport",
    "LCCalibrationResult",
    "ParametricFit",
    "calibrate_constant_lambda",
    "calibrate_local_correlation",
    "calibrate_parametric_lambda",
    "conditional_expectations_ab",
    "lambda_surface_frame",
    "reprice_index_smile",
    "straddle_vol",
]
