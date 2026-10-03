"""Correlated Brownians for several assets from the library's CRN-keyed streams.

Asset ``i`` owns the independent stream ``GaussianDraws(seed + ASSET_SEED_STRIDE · i, …,
n_brownians=1)`` and the assets' normals are mixed with the Cholesky factor ``L`` of the
correlation matrix: ``W = Z Lᵀ`` per step.  Hence (common random numbers across the multi-asset
world) a bump of one asset's volatility, of a strike or of the correlation itself leaves every
independent stream unchanged — a correlation bump moves only the mixing, which is what a
correlation sensitivity under CRN needs.  Antithetics are preserved by the mixing (it is
linear).  Checked by ``tests/test_multi.py``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.engine.rng import GaussianDraws

FloatArray = NDArray[np.float64]

#: seed offset between the assets' independent streams
ASSET_SEED_STRIDE = 7919


def check_correlation(correlation: ArrayLike) -> FloatArray:
    """The matrix as a float array, validated: square, symmetric, unit diagonal, positive
    definite (its Cholesky factor exists)."""
    c = np.asarray(correlation, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] == 0:
        raise ValueError("correlation must be a non-empty square matrix")
    if not np.all(np.isfinite(c)) or not np.allclose(c, c.T, atol=1e-12):
        raise ValueError("correlation must be finite and symmetric")
    if not np.allclose(np.diag(c), 1.0, atol=1e-12):
        raise ValueError("correlation must have a unit diagonal")
    try:
        np.linalg.cholesky(c)
    except np.linalg.LinAlgError as exc:
        raise ValueError("correlation must be positive definite") from exc
    return c


def constant_correlation(n: int, rho: float) -> FloatArray:
    """The ``n × n`` matrix with ``rho`` off the diagonal (positive definite for ``rho >
    −1/(n−1)``)."""
    if n < 1:
        raise ValueError("n must be positive")
    if n > 1 and not -1.0 / (n - 1) < rho < 1.0:
        raise ValueError(f"rho must lie in (-1/(n-1), 1) = ({-1.0 / (n - 1):.4g}, 1) for n = {n}")
    c = np.full((n, n), float(rho))
    np.fill_diagonal(c, 1.0)
    return c


class _AssetDraws:
    """Asset ``i``'s view of a :class:`CorrelatedDraws`: the ``.block`` the single-asset
    kernels call, shaped ``(n_paths, n_steps, 1)``."""

    n_brownians = 1

    def __init__(self, parent: CorrelatedDraws, asset: int) -> None:
        self.parent = parent
        self.asset = int(asset)
        self.seed = parent.seed
        self.n_paths = parent.n_paths
        self.n_steps = parent.n_steps
        self.antithetic = parent.antithetic

    def block(self, step0: int, step1: int, p0: int, p1: int) -> FloatArray:
        return self.parent.block_all(step0, step1, p0, p1)[:, :, self.asset : self.asset + 1]

    def normals(self, step: int, p0: int, p1: int) -> FloatArray:
        return self.block(step, step + 1, p0, p1)[:, 0, :]


class CorrelatedDraws:
    """``n_assets`` correlated standard normals per step (module docstring)."""

    def __init__(
        self,
        seed: int,
        n_paths: int,
        n_steps: int,
        correlation: ArrayLike,
        antithetic: bool = True,
    ) -> None:
        self.correlation = check_correlation(correlation)
        self.n_assets = int(self.correlation.shape[0])
        self.chol = np.linalg.cholesky(self.correlation)
        self.seed = int(seed)
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.antithetic = bool(antithetic)
        self.streams = [
            GaussianDraws(self.seed + ASSET_SEED_STRIDE * i, n_paths, n_steps, 1, antithetic)
            for i in range(self.n_assets)
        ]
        self._memo_key: tuple[int, int, int, int] | None = None
        self._memo: FloatArray | None = None

    def independent_block(self, step0: int, step1: int, p0: int, p1: int) -> FloatArray:
        """The uncorrelated normals ``(n_paths, n_steps, n_assets)`` of the block."""
        out = np.empty((p1 - p0, step1 - step0, self.n_assets))
        for i, s in enumerate(self.streams):
            out[:, :, i] = s.block(step0, step1, p0, p1)[:, :, 0]
        return out

    def block_all(self, step0: int, step1: int, p0: int, p1: int) -> FloatArray:
        """The correlated normals ``(n_paths, n_steps, n_assets)`` of the block (the last block
        is memoised: the assets' kernels request the same block in turn)."""
        key = (int(step0), int(step1), int(p0), int(p1))
        if self._memo is not None and self._memo_key == key:
            return self._memo
        z = self.independent_block(step0, step1, p0, p1)
        w = np.asarray(z @ self.chol.T, dtype=np.float64)
        self._memo_key, self._memo = key, w
        return w

    def asset(self, i: int) -> _AssetDraws:
        if not 0 <= i < self.n_assets:
            raise IndexError("asset out of range")
        return _AssetDraws(self, i)

    def with_correlation(self, correlation: ArrayLike) -> CorrelatedDraws:
        """The same independent streams mixed with another correlation (CRN across
        correlations)."""
        return CorrelatedDraws(self.seed, self.n_paths, self.n_steps, correlation, self.antithetic)

    def __repr__(self) -> str:
        return (
            f"CorrelatedDraws(seed={self.seed}, n_paths={self.n_paths}, n_steps={self.n_steps}, "
            f"n_assets={self.n_assets}, antithetic={self.antithetic})"
        )


__all__ = ["ASSET_SEED_STRIDE", "CorrelatedDraws", "check_correlation", "constant_correlation"]
