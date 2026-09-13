"""Dupire local-volatility model and the shared log-Euler spot kernel (SPEC §3.2).

Step (SPEC §3.1, variance frozen over the step):

    ln S_{i+1} = ln S_i + ∫_{t_i}^{t_{i+1}} (r − q) du − ½ σ²(t_i, S_i) Δt_i + σ(t_i, S_i) √Δt_i Z_i

with ``σ = σ_loc(t, S)`` interpolated bilinearly on the ``(t, k = ln S/F(t))`` grid.  The kernel
also accumulates ``∫ σ² dt`` and ``Σ (Δ ln S)²``.  Black–Scholes uses the same kernel with a flat
grid.  Checked by ``tests/test_surface.py::test_local_vol_mc_reprices_ssvi``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from volsto._numba import njit, prange
from volsto.engine.grid import TimeGrid
from volsto.engine.paths import PathSet
from volsto.engine.rng import GaussianDraws
from volsto.market.curves import ForwardCurve
from volsto.market.dupire import LocalVolSurface, bilinear_flat
from volsto.models.base import Model, ModelState

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


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
    tg: FloatArray,
    kg: FloatArray,
    local_var: FloatArray,
    out_log_spot: FloatArray,
    out_var: FloatArray,
    out_int_var: FloatArray,
    out_sum_sq: FloatArray,
) -> None:  # pragma: no cover - numba
    """Advance all paths over a block of steps (pure array function; see module docstring).

    ``z`` is ``(n_paths, n_block, ≥1)``; ``t_nodes``/``ln_f_nodes`` have ``n_block + 1`` entries;
    ``drifts[j] = ∫ (r − q)`` over step ``j``; ``step_record[j] ≥ 0`` is the output column to fill
    after step ``j``.  State arrays are updated in place.
    """
    n = log_spot.shape[0]
    nb = z.shape[1]
    for p in prange(n):
        ls = log_spot[p]
        iv = int_var[p]
        sq = sum_sq[p]
        for j in range(nb):
            dt = t_nodes[j + 1] - t_nodes[j]
            var = bilinear_flat(tg, kg, local_var, t_nodes[j], ls - ln_f_nodes[j])
            dls = drifts[j] - 0.5 * var * dt + np.sqrt(var * dt) * z[p, j, 0]
            ls += dls
            iv += var * dt
            sq += dls * dls
            col = step_record[j]
            if col >= 0:
                out_log_spot[p, col] = ls
                out_var[p, col] = bilinear_flat(
                    tg, kg, local_var, t_nodes[j + 1], ls - ln_f_nodes[j + 1]
                )
                out_int_var[p, col] = iv
                out_sum_sq[p, col] = sq
        log_spot[p] = ls
        int_var[p] = iv
        sum_sq[p] = sq


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
        self, grid: TimeGrid, draws: GaussianDraws, p0: int, p1: int, *, step_block: int = 64
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
        for s0 in range(0, grid.n_steps, step_block):
            s1 = min(s0 + step_block, grid.n_steps)
            z = draws.block(s0, s1, p0, p1)
            diffuse_block(
                ls,
                iv,
                sq,
                z,
                grid.times[s0 : s1 + 1],
                ln_f[s0 : s1 + 1],
                drifts[s0:s1],
                grid.step_record[s0:s1],
                lv.t_grid,
                lv.k_grid,
                lv.local_var,
                out.log_spot,
                out.variance,
                out.int_var,
                out.sum_sq,
            )
        return out

    def __repr__(self) -> str:
        return f"LocalVol({self.local_vol!r}, spot={self.spot})"
