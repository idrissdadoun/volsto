"""Path container (SPEC §3.1).

``PathSet`` stores the model state at the *record columns* (``t = 0`` and every fixing time)
plus two cumulative accumulators along the simulation grid: ``int_var = ∫₀ᵗ V_u du`` (the spot's
instantaneous variance, leverage included) and ``sum_sq = Σ (Δ ln S)²`` over simulation steps.
Products access the container only through the accessor methods below; all of them take an
``asset`` argument (only ``0`` today) so that a second underlying can later be added as an extra
leading axis without touching product code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _cols(col: int | ArrayLike) -> int | IntArray:
    if isinstance(col, (int, np.integer)):
        return int(col)
    return np.asarray(col, dtype=np.int64)


@dataclass
class PathSet:
    """Recorded paths: ``(n_paths, n_cols)`` arrays; factors ``(n_paths, n_cols, n_factors)``."""

    times: FloatArray
    log_spot: FloatArray
    variance: FloatArray
    factors: FloatArray
    int_var: FloatArray
    sum_sq: FloatArray

    def __post_init__(self) -> None:
        n, m = self.log_spot.shape
        if self.times.shape != (m,):
            raise ValueError("times must have one entry per column")
        for name in ("variance", "int_var", "sum_sq"):
            if getattr(self, name).shape != (n, m):
                raise ValueError(f"{name} must have shape {(n, m)}")
        if self.factors.ndim != 3 or self.factors.shape[:2] != (n, m):
            raise ValueError("factors must have shape (n_paths, n_cols, n_factors)")

    # -- shape -------------------------------------------------------------------------------

    @property
    def n_paths(self) -> int:
        return int(self.log_spot.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.log_spot.shape[1])

    @property
    def n_factors(self) -> int:
        return int(self.factors.shape[2])

    @property
    def n_assets(self) -> int:
        return 1

    def _check_asset(self, asset: int) -> None:
        if asset != 0:
            raise IndexError("only a single underlying (asset=0) is supported in v1")

    # -- accessors (the only API products use) -------------------------------------------------

    def log_spot_at(self, col: int | ArrayLike, asset: int = 0) -> FloatArray:
        """``ln S`` at column(s) ``col``: ``(n_paths,)`` or ``(n_paths, len(col))``."""
        self._check_asset(asset)
        return self.log_spot[:, _cols(col)]

    def spot_at(self, col: int | ArrayLike, asset: int = 0) -> FloatArray:
        return np.exp(self.log_spot_at(col, asset))

    def log_return(self, col0: int, col1: int, asset: int = 0) -> FloatArray:
        """``ln(S_{col1} / S_{col0})``."""
        self._check_asset(asset)
        return self.log_spot[:, col1] - self.log_spot[:, col0]

    def variance_at(self, col: int | ArrayLike, asset: int = 0) -> FloatArray:
        """Instantaneous variance of the spot (``L² ξ_t^t`` for LSV) at column(s)."""
        self._check_asset(asset)
        return self.variance[:, _cols(col)]

    def factors_at(self, col: int) -> FloatArray:
        """``(n_paths, n_factors)`` model factors at a column."""
        return self.factors[:, col, :]

    def integrated_variance(self, col0: int, col1: int, asset: int = 0) -> FloatArray:
        """``∫_{t_col0}^{t_col1} V_u du`` along the simulation grid."""
        self._check_asset(asset)
        return self.int_var[:, col1] - self.int_var[:, col0]

    def realised_variance_fixings(self, cols: ArrayLike, asset: int = 0) -> FloatArray:
        """``Σ_i ln²(S_{c_i}/S_{c_{i-1}})`` over the given consecutive columns (not annualised)."""
        self._check_asset(asset)
        c = np.asarray(cols, dtype=np.int64)
        if c.size < 2:
            raise ValueError("need at least two fixing columns")
        r = np.diff(self.log_spot[:, c], axis=1)
        return np.asarray(np.sum(r * r, axis=1), dtype=np.float64)

    def realised_variance_grid(self, col0: int, col1: int, asset: int = 0) -> FloatArray:
        """``Σ (Δ ln S)²`` over simulation steps between two columns (not annualised)."""
        self._check_asset(asset)
        return self.sum_sq[:, col1] - self.sum_sq[:, col0]

    # -- assembly ----------------------------------------------------------------------------

    @staticmethod
    def concat(parts: Sequence[PathSet]) -> PathSet:
        if not parts:
            raise ValueError("nothing to concatenate")
        t = parts[0].times
        for p in parts[1:]:
            if not np.array_equal(p.times, t):
                raise ValueError("all parts must share the same record times")
        return PathSet(
            times=t,
            log_spot=np.concatenate([p.log_spot for p in parts]),
            variance=np.concatenate([p.variance for p in parts]),
            factors=np.concatenate([p.factors for p in parts]),
            int_var=np.concatenate([p.int_var for p in parts]),
            sum_sq=np.concatenate([p.sum_sq for p in parts]),
        )

    @staticmethod
    def empty(n_paths: int, times: FloatArray, n_factors: int) -> PathSet:
        m = int(times.size)
        return PathSet(
            times=np.asarray(times, dtype=np.float64),
            log_spot=np.empty((n_paths, m)),
            variance=np.empty((n_paths, m)),
            factors=np.empty((n_paths, m, n_factors)),
            int_var=np.empty((n_paths, m)),
            sum_sq=np.empty((n_paths, m)),
        )

    def __repr__(self) -> str:
        return f"PathSet(n_paths={self.n_paths}, n_cols={self.n_cols}, n_factors={self.n_factors})"
