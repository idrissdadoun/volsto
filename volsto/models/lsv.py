"""LSV wrapper (SPEC §3.5): spot variance ``L(t, S)² ξ_t^t`` over the Bergomi SV kernel.

Frozen-leverage rule (owner amendment before M3): within a step ``[t_n, t_{n+1}]`` every leverage
lookup — start point, predictor / weak order-2 supporting values — uses the ``t_n`` slice
``L(t_n, ·)``.  During particle calibration ``L(t_{n+1}, ·)`` is not yet known when stepping to
``t_{n+1}``, and pricing applies the identical rule so that a calibrated leverage reprices the
surface with the pricing kernel.  The time dependence of ``L`` is therefore left-point; the
calibration absorbs the corresponding discretisation.  The SV kernel keeps its exact factor step
and its (optionally time-averaged) deterministic prefactor.  :func:`step_lsv_block` is the single
stepping routine used by both :class:`LSV.simulate_chunk` and
:func:`volsto.calibration.particle.calibrate_leverage`.  Checked by ``tests/test_lsv.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from volsto.config import SchemeConfig
from volsto.engine.grid import TimeGrid
from volsto.engine.paths import PathSet
from volsto.engine.rng import GaussianDraws
from volsto.models.base import Model, ModelState
from volsto.models.bergomi import BergomiSV, bergomi_block
from volsto.models.leverage import LeverageFunction

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def scheme_mode(scheme: SchemeConfig) -> int:
    return 2 if scheme.weak_order2 else (1 if scheme.predictor_corrector else 0)


def step_lsv_block(
    kernel: BergomiSV,
    scheme: SchemeConfig,
    log_spot: FloatArray,
    factors: FloatArray,
    int_var: FloatArray,
    sum_sq: FloatArray,
    z: FloatArray,
    t_nodes: FloatArray,
    ln_f_nodes: FloatArray,
    drifts: FloatArray,
    step_record: IntArray,
    k0: float,
    dk: float,
    lev_a: FloatArray,
    lev_b: FloatArray,
    lev_rec: FloatArray,
    out_log_spot: FloatArray,
    out_var: FloatArray,
    out_factors: FloatArray,
    out_int_var: FloatArray,
    out_sum_sq: FloatArray,
) -> None:
    """Advance the LSV state over the steps between ``t_nodes`` (in place).

    ``lev_a`` / ``lev_b`` / ``lev_rec`` are ``L²`` rows per step on the uniform ``(k0, dk)`` grid;
    under the frozen-leverage rule ``lev_a[j] = lev_b[j] = L²(t_j, ·)`` and ``lev_rec[j] =
    L²(t_{j+1}, ·)`` (the recorded instantaneous variance at the new time).
    """
    chol, decay, g_step, g_node = kernel.step_tables(t_nodes, scheme)
    bergomi_block(
        log_spot,
        factors,
        int_var,
        sum_sq,
        z,
        t_nodes,
        ln_f_nodes,
        drifts,
        step_record,
        chol,
        decay,
        g_step,
        g_node,
        kernel.coef,
        1,
        k0,
        dk,
        np.ascontiguousarray(lev_a),
        np.ascontiguousarray(lev_b),
        np.ascontiguousarray(lev_rec),
        scheme_mode(scheme),
        scheme.pc_eta,
        out_log_spot,
        out_var,
        out_factors,
        out_int_var,
        out_sum_sq,
    )


class LSV(Model):
    """``d ln S = (r − q − ½ L² ξ_t^t) dt + L(t, S) sqrt(ξ_t^t) dW^S`` over a Bergomi kernel."""

    def __init__(self, kernel: BergomiSV, leverage: LeverageFunction) -> None:
        self.kernel = kernel
        self.leverage = leverage
        self.forward_curve = kernel.forward_curve
        self.n_factors = kernel.n_factors
        self.n_brownians = kernel.n_brownians

    def required_times(self) -> FloatArray:
        """The leverage slices: the pricing grid contains them so the frozen rule matches."""
        return self.leverage.times[self.leverage.times > 0.0]

    def initial_state(self, n_paths: int) -> ModelState:
        s0 = self.kernel.initial_state(n_paths)
        l0 = float(self.leverage(0.0, self.spot))
        return ModelState(0.0, s0.log_spot, s0.variance * l0 * l0, s0.factors)

    def instantaneous_variance(self, state: ModelState) -> FloatArray:
        lev = self.leverage(state.t, np.exp(state.log_spot))
        return np.asarray(lev * lev * self.kernel.variance_from_factors(state.t, state.factors))

    def bump(self, **kwargs: Any) -> LSV:
        """``leverage=`` replaces ``L``; any other key is passed to the kernel (sticky leverage)."""
        lev = kwargs.pop("leverage", self.leverage)
        kernel = self.kernel.bump(**kwargs) if kwargs else self.kernel
        return LSV(kernel, lev)

    def leverage_tables(self, t_nodes: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``(lev_a, lev_b, lev_rec)`` for the steps between ``t_nodes`` under the frozen rule."""
        rows = self.leverage.l2_rows(t_nodes)
        start = np.ascontiguousarray(rows[:-1])
        return start, start, np.ascontiguousarray(rows[1:])

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
        for s0 in range(0, grid.n_steps, step_block):
            s1 = min(s0 + step_block, grid.n_steps)
            t_nodes = grid.times[s0 : s1 + 1]
            lev_a, lev_b, lev_rec = self.leverage_tables(t_nodes)
            step_lsv_block(
                self.kernel,
                scheme,
                ls,
                fac,
                iv,
                sq,
                draws.block(s0, s1, p0, p1),
                t_nodes,
                ln_f[s0 : s1 + 1],
                drifts[s0:s1],
                grid.step_record[s0:s1],
                self.leverage.k0,
                self.leverage.dk,
                lev_a,
                lev_b,
                lev_rec,
                out.log_spot,
                out.variance,
                out.factors,
                out.int_var,
                out.sum_sq,
            )
        return out

    def __repr__(self) -> str:
        return f"LSV({self.kernel!r}, {self.leverage!r})"
