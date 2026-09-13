"""Dupire local-volatility model and the shared spot-step kernel (SPEC §3.1–3.2, owner
amendment after M1).

Step over ``[t_n, t_{n+1}]`` (log-Euler with the variance frozen over the step):

    ln S_{n+1} = ln S_n + ∫ (r − q) du − ½ v_n Δt_n + √(v_n Δt_n) Z_n

where the step variance ``v_n`` is chosen by :class:`~volsto.config.SchemeConfig`:

* plain: ``v_n = σ²(t_n, S_n)`` (``local_var_time_average=False``, ``time_eval="start"``);
* midpoint diagnostic: ``v_n = σ²(t_n + Δt/2, S_n)``;
* time average (default): ``v_n = (1/Δt) ∫_{t_n}^{t_{n+1}} σ²(u, S_n) du``, exact for the
  piecewise-linear-in-``t`` local-variance grid, evaluated at ``k_n = ln S_n − ln F(t_n)``;
* predictor–corrector (default; weak predictor–corrector of Kloeden–Platen §15.5 with
  θ = η = ½, the local-vol analogue of Andersen's θ = ½ step): Euler predictor ``x̄`` with
  ``v_n``, then

      x_{n+1} = x_n + [μ − ½ v̂ − ¼ ∂ₓv̂] Δt + √(v̂ Δt) Z,   v̂ = ½ [v_n(x_n) + v_n'(x̄)],

  where ``v_n'`` is the end-of-step (or, with time averaging, the same step-averaged) variance
  at ``x̄`` and ``∂ₓv̂`` averages the ``k``-slopes at ``x_n`` and ``x̄``.  The drift term
  ``−¼ ∂ₓv̂`` is the Itô correction ``−η b b'`` required because the corrector's diffusion
  coefficient depends on the predictor (and hence on ``Z``); without it ``E[√v̂ Z] ≠ 0`` and the
  forward is not preserved.  One extra lookup per step.

The kernel also accumulates ``∫ v dt`` and ``Σ (Δ ln S)²`` and records the state at fixing
columns.  Variance tables are per-step rows on the uniform ``k`` grid, so a lookup is O(1).
Black–Scholes uses the same kernel with a flat table.  Checked by ``tests/test_surface.py``
(local-vol MC reprices SSVI) and ``tests/test_scheme.py`` (bias vs scheme options and dt).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from volsto._numba import njit, prange
from volsto.config import SchemeConfig
from volsto.engine.grid import TimeGrid
from volsto.engine.paths import PathSet
from volsto.engine.rng import GaussianDraws
from volsto.market.curves import ForwardCurve
from volsto.market.dupire import LocalVolSurface
from volsto.models.base import Model, ModelState

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@njit(cache=True)
def interp_uniform(k0: float, dk: float, row: FloatArray, k: float) -> float:  # pragma: no cover
    """Linear interpolation on a uniform grid ``k0 + i dk`` with flat extrapolation."""
    n = row.shape[0]
    x = (k - k0) / dk
    if x <= 0.0:
        return float(row[0])
    if x >= n - 1:
        return float(row[n - 1])
    i = int(x)
    w = x - i
    return float((1.0 - w) * row[i] + w * row[i + 1])


@njit(cache=True)
def slope_uniform(k0: float, dk: float, row: FloatArray, k: float) -> float:  # pragma: no cover
    """Slope ``∂row/∂k`` of the linear interpolant in the cell containing ``k`` (0 outside)."""
    n = row.shape[0]
    x = (k - k0) / dk
    if x <= 0.0 or x >= n - 1:
        return 0.0
    i = int(x)
    return float((row[i + 1] - row[i]) / dk)


@njit(inline="always")
def spot_step(
    ls: float,
    dt: float,
    drift: float,
    zj: float,
    k0: float,
    dk: float,
    row_a: FloatArray,
    row_b: FloatArray,
    scale: float,
    scale_b: float,
    ln_f_a: float,
    ln_f_b: float,
    mode: int,
    eta: float,
) -> tuple[float, float, float, float]:  # pragma: no cover - numba
    """One spot step with variance ``v(x) = scale · row(x − ln F)``.

    Returns ``(Δ ln S, v_used, b0, b0·b_x)`` with ``b0 = sqrt(v(x_n))`` the start-point diffusion
    coefficient and ``b0 b_x = ½ scale · d row/dx`` (needed by the second-order SV terms of
    :func:`~volsto.models.bergomi.bergomi_block`).  ``row_a`` is the start-point table, ``row_b``
    the end/predictor table (see module docstring); ``scale`` is 1 for local vol and the SV
    variance ``ξ_{t_n}`` for LSV (rows then hold ``L²``); ``scale_b`` is the SV variance used
    for the end-of-step *drift* evaluations (``ξ_{t_{n+1}}`` under the second-order SV step,
    equal to ``scale`` otherwise, which reproduces the frozen-variance step exactly).  The
    diffusion supporting values always use ``scale`` (deterministic offsets, Platen's device);
    the end-of-step variance enters them through the explicit weak order-2 cross terms in the
    caller.  ``mode``: 0 plain log-Euler, 1 weak predictor–corrector, 2 Platen weak order 2.
    """
    k = ls - ln_f_a
    row_k = interp_uniform(k0, dk, row_a, k)
    v = scale * row_k
    b0 = np.sqrt(v)
    bbx = 0.5 * scale * slope_uniform(k0, dk, row_a, k)
    if mode == 1:
        # weak predictor-corrector (Kloeden-Platen 15.5.4): drift averaged with theta = 1/2
        # between (t_n, x_n) and (t_{n+1}, x_pred), diffusion variance weighted (1-eta, eta);
        # the drift carries the Ito correction -eta b b' = -(eta/2) dv/dx because the
        # corrector's diffusion depends on the predictor and hence on Z
        x_pred = ls + drift - 0.5 * v * dt + np.sqrt(v * dt) * zj
        k_pred = x_pred - ln_f_b
        v_pred = scale * interp_uniform(k0, dk, row_b, k_pred)
        slope = (
            0.5 * scale * (slope_uniform(k0, dk, row_a, k) + slope_uniform(k0, dk, row_b, k_pred))
        )
        v_diff = (1.0 - eta) * v + eta * v_pred
        a_avg = -0.25 * (v + v_pred) - 0.5 * eta * slope
        return drift + a_avg * dt + np.sqrt(v_diff * dt) * zj, v_diff, b0, bbx
    if mode == 2:
        # Platen explicit weak order-2 scheme (Kloeden-Platen 15.1.3) for
        # dx = a dt + b dW, a = mu - v/2, b = sqrt(v); supporting values at t_{n+1}
        sdt = np.sqrt(dt)
        b0 = np.sqrt(v)
        a0 = drift - 0.5 * v * dt  # a * dt at (t_n, x_n)
        x_bar = ls + a0 + b0 * sdt * zj
        x_up = ls + a0 + b0 * sdt
        x_dn = ls + a0 - b0 * sdt
        v_bar = scale_b * interp_uniform(k0, dk, row_b, x_bar - ln_f_b)
        b_up = np.sqrt(scale * interp_uniform(k0, dk, row_b, x_up - ln_f_b))
        b_dn = np.sqrt(scale * interp_uniform(k0, dk, row_b, x_dn - ln_f_b))
        a1 = drift - 0.5 * v_bar * dt  # a * dt at (t_{n+1}, x_bar)
        dls = (
            0.5 * (a0 + a1)
            + 0.25 * (b_up + b_dn + 2.0 * b0) * sdt * zj
            + 0.25 * (b_up - b_dn) * sdt * (zj * zj - 1.0)
        )
        return dls, 0.5 * (v + v_bar), b0, bbx
    v1 = scale_b * row_k
    return drift - 0.25 * (v + v1) * dt + b0 * np.sqrt(dt) * zj, 0.5 * (v + v1), b0, bbx


@njit(parallel=True, cache=True)
def diffuse_block(
    log_spot: FloatArray,
    int_var: FloatArray,
    sum_sq: FloatArray,
    z: FloatArray,
    t_nodes: FloatArray,
    ln_f_nodes: FloatArray,
    drifts: FloatArray,
    step_record: IntArray,
    k0: float,
    dk: float,
    var_a: FloatArray,
    var_b: FloatArray,
    var_rec: FloatArray,
    mode: int,
    eta: float,
    out_log_spot: FloatArray,
    out_var: FloatArray,
    out_int_var: FloatArray,
    out_sum_sq: FloatArray,
) -> None:  # pragma: no cover - numba
    """Advance all paths over a block of steps (pure array function; see module docstring).

    ``z`` is ``(n_paths, n_block, ≥1)``; ``t_nodes``/``ln_f_nodes`` have ``n_block + 1`` entries;
    ``drifts[j] = ∫ (r − q)`` over step ``j``; ``var_a[j]``/``var_b[j]`` are the start-point and
    corrector variance rows for step ``j`` on the uniform ``k`` grid ``(k0, dk)``; ``var_rec[j]``
    is the instantaneous variance at ``t_{j+1}`` used when ``step_record[j] ≥ 0`` selects an
    output column; ``mode`` is 0 (plain), 1 (predictor–corrector with diffusion weight ``eta``)
    or 2 (Platen weak order 2).  State arrays are updated in place.
    """
    n = log_spot.shape[0]
    nb = z.shape[1]
    for p in prange(n):
        ls = log_spot[p]
        iv = int_var[p]
        sq = sum_sq[p]
        for j in range(nb):
            dt = t_nodes[j + 1] - t_nodes[j]
            zj = z[p, j, 0]
            dls, v, _b0, _bbx = spot_step(
                ls,
                dt,
                drifts[j],
                zj,
                k0,
                dk,
                var_a[j],
                var_b[j],
                1.0,
                1.0,
                ln_f_nodes[j],
                ln_f_nodes[j + 1],
                mode,
                eta,
            )
            ls += dls
            iv += v * dt
            sq += dls * dls
            col = step_record[j]
            if col >= 0:
                out_log_spot[p, col] = ls
                out_var[p, col] = interp_uniform(k0, dk, var_rec[j], ls - ln_f_nodes[j + 1])
                out_int_var[p, col] = iv
                out_sum_sq[p, col] = sq
        log_spot[p] = ls
        int_var[p] = iv
        sum_sq[p] = sq


def step_variance_tables(
    lv: LocalVolSurface, t_nodes: FloatArray, scheme: SchemeConfig
) -> tuple[FloatArray, FloatArray]:
    """``(var_a, var_b)`` rows for the steps between ``t_nodes`` under ``scheme``.

    ``var_a`` is looked up at ``(t_n, x_n)``, ``var_b`` at the predictor/supporting points.  The
    weak order-2 scheme always uses start/end-of-step values (its trapezoidal time treatment
    replaces time averaging).
    """
    if scheme.local_var_time_average and not scheme.weak_order2:
        va = lv.var_time_average(t_nodes)
        return va, va
    if scheme.local_var_time_eval == "midpoint":
        vm = lv.var_at_times(0.5 * (t_nodes[:-1] + t_nodes[1:]))
        return vm, vm
    vs = lv.var_at_times(t_nodes)
    return np.ascontiguousarray(vs[:-1]), np.ascontiguousarray(vs[1:])


class LocalVol(Model):
    """Dupire local-vol model driven by a :class:`~volsto.market.dupire.LocalVolSurface`."""

    n_factors = 0
    n_brownians = 1

    def __init__(
        self, local_vol: LocalVolSurface, forward_curve: ForwardCurve | None = None
    ) -> None:
        self.local_vol = local_vol
        self.forward_curve = forward_curve or local_vol.forward_curve

    def initial_state(self, n_paths: int) -> ModelState:
        ls = np.full(n_paths, np.log(self.spot))
        var = np.full(n_paths, float(self.local_vol.local_var_k(0.0, 0.0)))
        return ModelState(0.0, ls, var, np.empty((n_paths, 0)))

    def instantaneous_variance(self, state: ModelState) -> FloatArray:
        k = state.log_spot - float(self.forward_curve.log_forward(state.t))
        return self.local_vol.local_var_k(state.t, k)

    def bump(self, **kwargs: Any) -> LocalVol:
        """Supported: ``spot`` (sticky local vol in ``k``), ``local_vol``."""
        fc = self.forward_curve
        lv = self.local_vol
        for key, val in kwargs.items():
            if key == "spot":
                fc = fc.with_spot(float(val))
            elif key == "local_vol":
                lv = val
            else:
                raise ValueError(f"LocalVol.bump: unknown parameter {key!r}")
        return LocalVol(lv, fc)

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
        out = PathSet.empty(n, grid.record_times, 0)
        state = self.initial_state(n)
        out.log_spot[:, 0] = state.log_spot
        out.variance[:, 0] = state.variance
        out.int_var[:, 0] = 0.0
        out.sum_sq[:, 0] = 0.0
        ls = state.log_spot.copy()
        iv = np.zeros(n)
        sq = np.zeros(n)
        ln_f = np.asarray(self.forward_curve.log_forward(grid.times), dtype=np.float64)
        drifts = np.diff(ln_f)
        lv = self.local_vol
        mode = 2 if scheme.weak_order2 else (1 if scheme.predictor_corrector else 0)
        for s0 in range(0, grid.n_steps, step_block):
            s1 = min(s0 + step_block, grid.n_steps)
            t_nodes = grid.times[s0 : s1 + 1]
            rec = grid.step_record[s0:s1]
            var_a, var_b = step_variance_tables(lv, t_nodes, scheme)
            var_rec = lv.var_at_times(t_nodes[1:]) if np.any(rec >= 0) else var_a
            z = draws.block(s0, s1, p0, p1)
            diffuse_block(
                ls,
                iv,
                sq,
                z,
                t_nodes,
                ln_f[s0 : s1 + 1],
                drifts[s0:s1],
                rec,
                lv.k0,
                lv.dk,
                var_a,
                var_b,
                var_rec,
                mode,
                scheme.pc_eta,
                out.log_spot,
                out.variance,
                out.int_var,
                out.sum_sq,
            )
        return out

    def __repr__(self) -> str:
        return f"LocalVol({self.local_vol!r}, spot={self.spot})"
