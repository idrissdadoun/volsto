"""Dupire local volatility from a total-variance surface (SPEC §2.3, Gatheral 2006 eq. 1.10).

.. math::
    \\sigma_{loc}^2(k, T) = \\frac{\\partial_T w}
    {1 - \\frac{k}{w}\\partial_k w + \\tfrac14\\big(-\\tfrac14 - \\tfrac1w + \\tfrac{k^2}{w^2}\\big)
     (\\partial_k w)^2 + \\tfrac12 \\partial_{kk} w}

Derivatives are central finite differences of the analytic surface (``dk``, ``dT`` from
:class:`~volsto.config.LocalVolConfig`), the result is floored/capped and stored on a ``(T, k)``
grid with bilinear interpolation (flat extrapolation).  The ``T`` grid is square-root spaced
(dense at short maturities, where the local vol varies fastest in ``t``); a uniform 200-point
grid was measured to bias 3m repricing by ~0.1 vol point, and a ``k`` spacing coarser than
0.0025 biased the 1y variance-swap strike (see :class:`~volsto.config.LocalVolConfig`).
Checked by ``tests/test_surface.py``
(positivity, local-vol MC repricing of the SSVI pillars).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto._numba import njit
from volsto.config import LocalVolConfig
from volsto.market.curves import ForwardCurve
from volsto.market.surface import ImpliedSurface

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class DupireDiagnostics:
    """Output of :meth:`LocalVolSurface.check_positive`."""

    min_local_var: float
    max_local_var: float
    min_denominator: float
    min_dw_dt: float
    n_floored: int
    n_capped: int
    n_total: int

    @property
    def ok(self) -> bool:
        return self.min_denominator > 0 and self.min_dw_dt > 0 and self.n_floored == 0

    @property
    def floored_fraction(self) -> float:
        return self.n_floored / self.n_total


@njit(cache=True)
def bilinear_flat(
    tg: FloatArray, kg: FloatArray, vals: FloatArray, t: float, k: float
) -> float:  # pragma: no cover - numba
    """Bilinear interpolation on a rectangular ``(t, k)`` grid with flat extrapolation."""
    nt = tg.shape[0]
    nk = kg.shape[0]
    if t <= tg[0]:
        it = 0
        wt = 0.0
    elif t >= tg[nt - 1]:
        it = nt - 2
        wt = 1.0
    else:
        it = int(np.searchsorted(tg, t)) - 1
        wt = (t - tg[it]) / (tg[it + 1] - tg[it])
    if k <= kg[0]:
        ik = 0
        wk = 0.0
    elif k >= kg[nk - 1]:
        ik = nk - 2
        wk = 1.0
    else:
        ik = int(np.searchsorted(kg, k)) - 1
        wk = (k - kg[ik]) / (kg[ik + 1] - kg[ik])
    v00 = vals[it, ik]
    v01 = vals[it, ik + 1]
    v10 = vals[it + 1, ik]
    v11 = vals[it + 1, ik + 1]
    return float((1.0 - wt) * ((1.0 - wk) * v00 + wk * v01) + wt * ((1.0 - wk) * v10 + wk * v11))


@njit(cache=True)
def _bilinear_many(
    tg: FloatArray, kg: FloatArray, vals: FloatArray, t: FloatArray, k: FloatArray
) -> FloatArray:  # pragma: no cover - numba
    out = np.empty(t.shape[0])
    for i in range(t.shape[0]):
        out[i] = bilinear_flat(tg, kg, vals, t[i], k[i])
    return out


class LocalVolSurface:
    """Local variance ``σ_loc²(t, k)`` on a ``(t, k)`` grid, ``k = ln(S / F(t))``."""

    def __init__(
        self,
        t_grid: FloatArray,
        k_grid: FloatArray,
        local_var: FloatArray,
        forward_curve: ForwardCurve,
        diagnostics: DupireDiagnostics | None = None,
    ) -> None:
        t_grid = np.asarray(t_grid, dtype=np.float64)
        k_grid = np.asarray(k_grid, dtype=np.float64)
        local_var = np.asarray(local_var, dtype=np.float64)
        if t_grid.ndim != 1 or k_grid.ndim != 1 or local_var.shape != (t_grid.size, k_grid.size):
            raise ValueError("local_var must have shape (n_t, n_k)")
        if t_grid.size < 2 or k_grid.size < 2:
            raise ValueError("grids need at least two points")
        if np.any(np.diff(t_grid) <= 0) or np.any(np.diff(k_grid) <= 0):
            raise ValueError("grids must be strictly increasing")
        dk = float(k_grid[1] - k_grid[0])
        if not np.allclose(np.diff(k_grid), dk, rtol=1e-8, atol=1e-14):
            raise ValueError("k_grid must be uniformly spaced (the step kernel uses O(1) lookups)")
        if np.any(local_var <= 0) or not np.all(np.isfinite(local_var)):
            raise ValueError("local variance must be positive and finite")
        self.t_grid = t_grid
        self.k_grid = k_grid
        self.k0 = float(k_grid[0])
        self.dk = dk
        self.local_var = np.ascontiguousarray(local_var)
        self.forward_curve = forward_curve
        self.diagnostics = diagnostics
        # cumulative time integral of the (piecewise-linear in t, flat outside) interpolant
        w = np.empty_like(self.local_var)
        w[0] = t_grid[0] * self.local_var[0]
        w[1:] = w[0] + np.cumsum(
            0.5 * np.diff(t_grid)[:, None] * (self.local_var[1:] + self.local_var[:-1]), axis=0
        )
        self._cum_var = w

    @classmethod
    def from_implied(
        cls, surface: ImpliedSurface, cfg: LocalVolConfig | None = None
    ) -> LocalVolSurface:
        """Dupire local variance from the surface (formula in the module docstring)."""
        cfg = cfg or LocalVolConfig()
        if cfg.t_max > surface.max_maturity:
            raise ValueError("LocalVolConfig.t_max exceeds the surface's max_maturity")
        # square-root spacing: dense where the local vol varies fastest in t (short maturities)
        u = np.linspace(0.0, 1.0, cfg.n_t)
        t_grid = cfg.t_min + (cfg.t_max - cfg.t_min) * u * u
        k_grid = np.linspace(cfg.k_min, cfg.k_max, cfg.n_k)
        T = t_grid[:, None]
        k = k_grid[None, :]
        dk = cfg.dk
        dT = np.minimum(cfg.dt, 0.5 * T)
        w = surface.total_variance(k, T)
        w_kp = surface.total_variance(k + dk, T)
        w_km = surface.total_variance(k - dk, T)
        w_Tp = surface.total_variance(k, T + dT)
        w_Tm = surface.total_variance(k, T - dT)
        dw_dk = (w_kp - w_km) / (2.0 * dk)
        d2w_dk2 = (w_kp - 2.0 * w + w_km) / (dk * dk)
        dw_dT = (w_Tp - w_Tm) / (2.0 * dT)
        denom = (
            1.0
            - (k / w) * dw_dk
            + 0.25 * (-0.25 - 1.0 / w + (k * k) / (w * w)) * dw_dk**2
            + 0.5 * d2w_dk2
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            raw = dw_dT / denom
        raw = np.where(np.isfinite(raw), raw, cfg.floor)
        raw = np.where(denom > 0, raw, cfg.floor)
        n_floored = int(np.sum(raw < cfg.floor))
        n_capped = int(np.sum(raw > cfg.cap))
        local_var = np.clip(raw, cfg.floor, cfg.cap)
        diag = DupireDiagnostics(
            min_local_var=float(np.min(raw)),
            max_local_var=float(np.max(raw)),
            min_denominator=float(np.min(denom)),
            min_dw_dt=float(np.min(dw_dT)),
            n_floored=n_floored,
            n_capped=n_capped,
            n_total=int(raw.size),
        )
        return cls(t_grid, k_grid, local_var, surface.forward_curve, diag)

    @classmethod
    def flat(cls, vol: float, forward_curve: ForwardCurve, t_max: float = 10.0) -> LocalVolSurface:
        t_grid = np.array([0.0, t_max])
        k_grid = np.array([-10.0, 10.0])
        return cls(t_grid, k_grid, np.full((2, 2), vol * vol), forward_curve)

    def check_positive(self) -> DupireDiagnostics:
        """Positivity diagnostic of the raw Dupire formula (before floor/cap)."""
        if self.diagnostics is None:
            return DupireDiagnostics(
                float(self.local_var.min()),
                float(self.local_var.max()),
                float("inf"),
                float("inf"),
                0,
                0,
                int(self.local_var.size),
            )
        return self.diagnostics

    # -- per-step tables for the spot kernel ---------------------------------------------------

    def _t_weights(self, t: FloatArray) -> tuple[NDArray[np.int64], FloatArray]:
        tg = self.t_grid
        i = np.clip(np.searchsorted(tg, t, side="right") - 1, 0, tg.size - 2)
        w = np.clip((t - tg[i]) / (tg[i + 1] - tg[i]), 0.0, 1.0)
        return i.astype(np.int64), np.asarray(w, dtype=np.float64)

    def reanchored(self, forward_curve: ForwardCurve) -> LocalVolSurface:
        """The same ``σ_loc(t, S)`` re-expressed on the ``k`` grid of a new forward curve (a spot
        bump with the local vol held fixed in spot, the "model" regime for pure LV): each slice is
        shifted by ``ln F_new(t) − ln F_old(t)`` by linear interpolation in ``k``, flat outside."""
        shift = np.asarray(forward_curve.log_forward(self.t_grid)) - np.asarray(
            self.forward_curve.log_forward(self.t_grid)
        )
        out = np.empty_like(self.local_var)
        for j in range(self.t_grid.size):
            out[j] = np.interp(self.k_grid + shift[j], self.k_grid, self.local_var[j])
        return LocalVolSurface(self.t_grid, self.k_grid, out, forward_curve, self.diagnostics)

    def var_at_times(self, times: ArrayLike) -> FloatArray:
        """``σ_loc²(t_j, k)`` for every ``t_j`` on the ``k`` grid: shape ``(len(times), n_k)``."""
        t = np.atleast_1d(np.asarray(times, dtype=np.float64))
        i, w = self._t_weights(t)
        v = self.local_var
        return np.asarray((1.0 - w)[:, None] * v[i] + w[:, None] * v[i + 1], dtype=np.float64)

    def cumulative_var(self, times: ArrayLike) -> FloatArray:
        """``W(t, k) = ∫₀ᵗ σ_loc²(s, k) ds`` of the interpolant (exact), ``(len(times), n_k)``."""
        t = np.atleast_1d(np.asarray(times, dtype=np.float64))
        tg = self.t_grid
        v = self.local_var
        i, w = self._t_weights(t)
        v_t = (1.0 - w)[:, None] * v[i] + w[:, None] * v[i + 1]
        inside = self._cum_var[i] + 0.5 * (t - tg[i])[:, None] * (v[i] + v_t)
        below = t[:, None] * v[0][None, :]
        above = self._cum_var[-1][None, :] + (t - tg[-1])[:, None] * v[-1][None, :]
        out = np.where(
            (t <= tg[0])[:, None], below, np.where((t >= tg[-1])[:, None], above, inside)
        )
        return np.asarray(out, dtype=np.float64)

    def var_time_average(self, t_nodes: ArrayLike) -> FloatArray:
        """Exact time average of ``σ_loc²(·, k)`` over each ``[t_j, t_{j+1}]``: ``(len - 1, n_k)``.

        Used by the ``local_var_time_average`` scheme option (owner amendment after M1).
        """
        t = np.atleast_1d(np.asarray(t_nodes, dtype=np.float64))
        if t.size < 2 or np.any(np.diff(t) <= 0):
            raise ValueError("t_nodes must be strictly increasing with ≥ 2 entries")
        W = self.cumulative_var(t)
        return np.asarray(np.diff(W, axis=0) / np.diff(t)[:, None], dtype=np.float64)

    def local_var_k(self, t: ArrayLike, k: ArrayLike) -> FloatArray:
        """``σ_loc²(t, k)`` by bilinear interpolation (flat outside the grid)."""
        t_, k_ = np.broadcast_arrays(
            np.asarray(t, dtype=np.float64), np.asarray(k, dtype=np.float64)
        )
        out = _bilinear_many(
            self.t_grid,
            self.k_grid,
            self.local_var,
            np.ascontiguousarray(t_.ravel()),
            np.ascontiguousarray(k_.ravel()),
        )
        return np.asarray(out.reshape(t_.shape), dtype=np.float64)

    def local_vol_k(self, t: ArrayLike, k: ArrayLike) -> FloatArray:
        return np.sqrt(self.local_var_k(t, k))

    def local_vol(self, t: ArrayLike, S: ArrayLike) -> FloatArray:
        """``σ_loc(t, S)`` with ``k = ln(S / F(t))``."""
        t_ = np.asarray(t, dtype=np.float64)
        k = np.log(np.asarray(S, dtype=np.float64)) - self.forward_curve.log_forward(t_)
        return self.local_vol_k(t_, k)

    def __repr__(self) -> str:
        return (
            f"LocalVolSurface(t∈[{self.t_grid[0]:.4g}, {self.t_grid[-1]:.4g}] x{self.t_grid.size}, "
            f"k∈[{self.k_grid[0]:.3g}, {self.k_grid[-1]:.3g}] x{self.k_grid.size})"
        )
