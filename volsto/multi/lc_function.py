"""The local correlation function ``λ(t, k)`` (SPEC §8.7, M12 part LC3) — the counterpart of
:class:`~volsto.models.leverage.LeverageFunction`.

``values`` is ``(n_slices, n_k)`` in ``[0, 1]``: the slices are the calibration grid's times
``t_0 < … < t_N`` and ``k`` is a uniform grid in **basket** log-moneyness ``k = ln(B_t / F_B(t))``
(the basket state of :class:`~volsto.multi.lc_model.BasketSpec`; by default the shared Dupire
grid of the names).

* In ``t``: linear between slices, the first slice below ``t_0``, the last slice held beyond the
  horizon (:meth:`LocalCorrelationFunction.rows`).  The model itself refuses to price beyond
  the horizon; the rule only says what a row is.
* In ``k``: linear with flat extrapolation (``interp_uniform`` in the kernel, which also clips
  the interpolated value to ``[0, λ_max]`` of the family — interpolating clipped rows stays in
  range; the clip guards against a table loaded from disk).
* **Frozen rule**: over a step ``[t_j, t_{j+1}]`` the kernel uses the row at ``t_j``
  (:meth:`LocalCorrelationFunction.step_rows`), read at the path's basket log-moneyness at
  ``t_j`` — in the particle calibration and in pricing alike, whose grid contains every slice.
  A fixing between two slices adds a step, whose row is interpolated in ``t``.

:class:`ParametricLambda` is the time-homogeneous two-parameter family ``λ(t, k) = clip(λ0 −
slope·k, lam_lo, lam_hi)`` (the reference implementation's ``ρ_t = clip(ρ0 − c·k, ρ_min,
ρ_max)`` on an equicorrelation ``R_low``, :meth:`ParametricLambda.from_rho`), tabulated on the
grid: exact at the nodes and linear in between, hence exact everywhere except in the two cells
that hold the kinks.

Checked by ``tests/test_local_correlation.py::test_lambda_function_rows``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class ParametricLambda:
    """``λ(t, k) = clip(lambda0 − slope·k, lam_lo, lam_hi)``, the same at every ``t``."""

    lambda0: float
    slope: float
    lam_lo: float = 0.0
    lam_hi: float = 1.0

    def __post_init__(self) -> None:
        if not all(np.isfinite(x) for x in (self.lambda0, self.slope, self.lam_lo, self.lam_hi)):
            raise ValueError("the parameters must be finite")
        if not 0.0 <= self.lam_lo <= self.lam_hi <= 1.0:
            raise ValueError("need 0 <= lam_lo <= lam_hi <= 1")

    @classmethod
    def from_rho(cls, rho0: float, c: float, rho_min: float, rho_max: float) -> ParametricLambda:
        """The reference's form ``ρ_t = clip(ρ0 − c·k, ρ_min, ρ_max)`` on the equicorrelation
        family ``ρ = ρ_min + λ(1 − ρ_min)`` (derived, exact): ``λ0 = (ρ0 − ρ_min)/(1 − ρ_min)``,
        ``slope = c/(1 − ρ_min)``, ``lam_lo = 0``, ``lam_hi = (ρ_max − ρ_min)/(1 − ρ_min)``."""
        if not 0.0 <= rho_min < rho_max <= 1.0:
            raise ValueError("need 0 <= rho_min < rho_max <= 1")
        span = 1.0 - rho_min
        return cls((rho0 - rho_min) / span, c / span, 0.0, (rho_max - rho_min) / span)

    def to_rho(self, rho_min: float) -> tuple[float, float]:
        """``(ρ0, c)`` of :meth:`from_rho` for the equicorrelation level ``rho_min``."""
        span = 1.0 - rho_min
        return rho_min + self.lambda0 * span, self.slope * span

    def __call__(self, k: ArrayLike) -> FloatArray:
        kk = np.asarray(k, dtype=np.float64)
        return np.asarray(
            np.clip(self.lambda0 - self.slope * kk, self.lam_lo, self.lam_hi), dtype=np.float64
        )


class LocalCorrelationFunction:
    """``λ(t, k)`` on a ``(t, k)`` grid; ``values`` is ``(n_slices, n_k)`` in ``[0, 1]`` (module
    docstring)."""

    def __init__(
        self,
        times: ArrayLike,
        k_grid: ArrayLike,
        values: ArrayLike,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        t = np.asarray(times, dtype=np.float64).ravel()
        k = np.asarray(k_grid, dtype=np.float64).ravel()
        v = np.asarray(values, dtype=np.float64)
        if t.size < 1 or np.any(np.diff(t) <= 0) or t[0] < 0:
            raise ValueError("times must be non-negative and strictly increasing")
        if k.size < 2 or not np.allclose(np.diff(k), k[1] - k[0], rtol=1e-8, atol=1e-14):
            raise ValueError("k_grid must be uniform with at least two points")
        if v.shape != (t.size, k.size):
            raise ValueError(f"values must have shape {(t.size, k.size)}")
        if not np.all(np.isfinite(v)) or np.any(v < 0.0) or np.any(v > 1.0):
            raise ValueError("local correlation values must be finite and lie in [0, 1]")
        self.times = t
        self.k_grid = k
        self.values = np.ascontiguousarray(v)
        self.metadata: dict[str, Any] = dict(metadata or {})

    # -- grid ----------------------------------------------------------------------------------

    @property
    def k0(self) -> float:
        return float(self.k_grid[0])

    @property
    def dk(self) -> float:
        return float(self.k_grid[1] - self.k_grid[0])

    @property
    def horizon(self) -> float:
        return float(self.times[-1])

    @property
    def n_slices(self) -> int:
        return int(self.times.size)

    # -- evaluation ------------------------------------------------------------------------------

    def rows(self, times: ArrayLike) -> FloatArray:
        """``λ(t_j, ·)`` rows for each ``t_j``: linear in ``t`` between slices, the first slice
        below ``t_0``, the last slice beyond the horizon.  At a slice time the row is the
        slice, bit for bit."""
        t = np.atleast_1d(np.asarray(times, dtype=np.float64))
        tg = self.times
        if tg.size == 1:
            return np.repeat(self.values, t.size, axis=0)
        i = np.clip(np.searchsorted(tg, t, side="right") - 1, 0, tg.size - 2)
        w = np.clip((t - tg[i]) / (tg[i + 1] - tg[i]), 0.0, 1.0)
        out = (1.0 - w)[:, None] * self.values[i] + w[:, None] * self.values[i + 1]
        return np.ascontiguousarray(out, dtype=np.float64)

    def step_rows(self, t_nodes: ArrayLike) -> FloatArray:
        """The start-of-step rows of the steps between ``t_nodes`` (frozen rule): ``(len − 1,
        n_k)``."""
        t = np.atleast_1d(np.asarray(t_nodes, dtype=np.float64))
        if t.size < 2:
            raise ValueError("t_nodes needs at least two entries")
        return self.rows(t[:-1])

    def __call__(self, t: ArrayLike, k: ArrayLike) -> FloatArray:
        """``λ(t, k)``: rows in ``t``, linear in ``k``, flat outside the grid."""
        t_, k_ = np.broadcast_arrays(
            np.asarray(t, dtype=np.float64), np.asarray(k, dtype=np.float64)
        )
        rows = self.rows(t_.ravel())
        kk = k_.ravel()
        x = np.clip((kk - self.k0) / self.dk, 0.0, self.k_grid.size - 1.0)
        i = np.minimum(x.astype(np.int64), self.k_grid.size - 2)
        w = x - i
        idx = np.arange(kk.size)
        out = (1.0 - w) * rows[idx, i] + w * rows[idx, i + 1]
        return np.asarray(out.reshape(t_.shape), dtype=np.float64)

    # -- constructors ----------------------------------------------------------------------------

    @classmethod
    def constant(
        cls,
        value: float,
        times: ArrayLike,
        k_grid: ArrayLike,
        metadata: dict[str, Any] | None = None,
    ) -> LocalCorrelationFunction:
        """``λ ≡ value`` on the given slices and grid (the constant-correlation companion: the
        same slices as a calibrated ``λ`` give it the same pricing grid)."""
        t = np.asarray(times, dtype=np.float64).ravel()
        k = np.asarray(k_grid, dtype=np.float64).ravel()
        meta = {"kind": "constant", "value": float(value), **(metadata or {})}
        return cls(t, k, np.full((t.size, k.size), float(value)), meta)

    @classmethod
    def parametric(
        cls,
        p: ParametricLambda,
        times: ArrayLike,
        k_grid: ArrayLike,
        metadata: dict[str, Any] | None = None,
    ) -> LocalCorrelationFunction:
        """``p`` tabulated on the grid at every slice."""
        t = np.asarray(times, dtype=np.float64).ravel()
        k = np.asarray(k_grid, dtype=np.float64).ravel()
        meta = {
            "kind": "parametric",
            "lambda0": p.lambda0,
            "slope": p.slope,
            "lam_lo": p.lam_lo,
            "lam_hi": p.lam_hi,
            **(metadata or {}),
        }
        return cls(t, k, np.tile(p(k), (t.size, 1)), meta)

    def with_values(self, values: ArrayLike, **metadata: Any) -> LocalCorrelationFunction:
        return LocalCorrelationFunction(
            self.times, self.k_grid, values, {**self.metadata, **metadata}
        )

    @staticmethod
    def average(
        a: LocalCorrelationFunction, b: LocalCorrelationFunction
    ) -> LocalCorrelationFunction:
        """Slice-wise average of two calibrations on the same grid (the second pass)."""
        if not (np.array_equal(a.times, b.times) and np.array_equal(a.k_grid, b.k_grid)):
            raise ValueError("local correlation functions must share the same grid")
        return a.with_values(0.5 * (a.values + b.values), averaged_with=b.metadata.get("seed"))

    # -- serialisation ---------------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write ``.npz`` with the grid, the values and the JSON metadata."""
        p = Path(path)
        np.savez_compressed(
            p,
            times=self.times,
            k_grid=self.k_grid,
            values=self.values,
            metadata=np.array(json.dumps(self.metadata, sort_keys=True, default=str)),
        )
        return p if p.suffix == ".npz" else p.with_suffix(p.suffix + ".npz")

    @classmethod
    def load(cls, path: str | Path) -> LocalCorrelationFunction:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["metadata"]))
            return cls(z["times"], z["k_grid"], z["values"], meta)

    def __repr__(self) -> str:
        return (
            f"LocalCorrelationFunction(n_slices={self.n_slices}, horizon={self.horizon:.4g}, "
            f"k∈[{self.k_grid[0]:.3g}, {self.k_grid[-1]:.3g}] x{self.k_grid.size}, "
            f"λ∈[{self.values.min():.3g}, {self.values.max():.3g}])"
        )


__all__ = ["LocalCorrelationFunction", "ParametricLambda"]
