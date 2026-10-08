"""Correlated Brownians for several assets from the library's CRN-keyed streams.

Asset ``i`` owns the independent stream ``GaussianDraws(seed + ASSET_SEED_STRIDE · i, …,
n_brownians=1)`` and the assets' normals are mixed with the Cholesky factor ``L`` of the
correlation matrix: ``W = Z Lᵀ`` per step.  Hence (common random numbers across the multi-asset
world) a bump of one asset's volatility, of a strike or of the correlation itself leaves every
independent stream unchanged — a correlation bump moves only the mixing, which is what a
correlation sensitivity under CRN needs.  Antithetics are preserved by the mixing (it is
linear).  Checked by ``tests/test_multi.py``.

**The mixing arithmetic is fixed** (SPEC §8.7, M12 part LC2).  ``W_i = Σ_{m ≤ i} L[i, m]·Z_m`` is
accumulated from 0.0 in ascending ``m`` by a numba loop (:func:`lower_row_dot`,
:func:`mix_lower`), not by a BLAS product: a BLAS result depends on the block shape and on the
BLAS build (SPEC §13.3), while the local-correlation kernel (``volsto/multi/lc_kernel.py``)
must reproduce these normals bit for bit, one step at a time in the calibration and 64 steps
at a time in pricing, on every machine.  For an exact equicorrelation matrix the factor is the
closed form of :func:`equicorrelation_factor` (no LAPACK call, the same doubles everywhere)
and the mixing is the O(n) prefix recursion :func:`mix_equi`, which performs the additions of
:func:`lower_row_dot` in the same order and is therefore bit-identical to it
(``tests/test_local_correlation.py::test_equi_prefix_equals_general_mix``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto._numba import njit, prange
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


# --------------------------------------------------------------------------------------------
# the factor and the mixing (fixed arithmetic; module docstring)
# --------------------------------------------------------------------------------------------


def is_equicorrelation(correlation: ArrayLike) -> bool:
    """Whether the matrix is an *exact* equicorrelation: a diagonal of exactly 1.0 and every
    off-diagonal entry the same double (``constant_correlation`` builds such a matrix).  A
    ``1 × 1`` matrix is one (with correlation 0)."""
    c = np.asarray(correlation, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] == 0:
        return False
    n = c.shape[0]
    if not np.all(np.diag(c) == 1.0):
        return False
    if n == 1:
        return True
    off = c[~np.eye(n, dtype=bool)]
    return bool(np.all(off == off[0]))


def equicorrelation_factor(n: int, rho: float) -> tuple[FloatArray, FloatArray]:
    """The Cholesky factor of ``(1 − ρ)·I + ρ·11ᵀ`` in closed form, as its diagonal ``d`` and
    its column values ``ℓ`` (``L[j, j] = d_j`` and ``L[i, j] = ℓ_j`` for ``i > j``):

        s_0 = 0;   d_j = √(1 − s_j),   ℓ_j = (ρ − s_j)/d_j,   s_{j+1} = s_j + ℓ_j²

    Derived (SPEC §8.7): row ``i`` of ``L Lᵀ`` against row ``j < i`` gives ``Σ_{m<j} ℓ_m² +
    ℓ_j d_j = s_j + ρ − s_j = ρ``, and the diagonal ``s_j + d_j² = 1``.  Only correctly rounded
    operations (``+ − × ÷ √``) on doubles: the same factor on every machine.  Checked against
    ``numpy.linalg.cholesky`` to 1e-14 by ``tests/test_local_correlation.py::
    test_draws_antithetic_and_streams``."""
    if n < 1:
        raise ValueError("n must be positive")
    rho = float(rho)
    if n > 1 and not -1.0 / (n - 1) < rho < 1.0:
        raise ValueError(f"rho must lie in (-1/(n-1), 1) for n = {n}")
    d = np.empty(n)
    ell = np.empty(n)
    s = 0.0
    for j in range(n):
        dj = math.sqrt(1.0 - s)
        lj = (rho - s) / dj
        d[j] = dj
        ell[j] = lj
        s = s + lj * lj
    return d, ell


def equicorrelation_cholesky(n: int, rho: float) -> FloatArray:
    """The lower factor ``L`` of :func:`equicorrelation_factor` as a matrix."""
    d, ell = equicorrelation_factor(n, rho)
    chol = np.zeros((n, n))
    for j in range(n):
        chol[j, j] = d[j]
        chol[j + 1 :, j] = ell[j]
    return chol


def cholesky_factor(correlation: ArrayLike) -> FloatArray:
    """The lower Cholesky factor used by every mixing of the multi-asset layer: the closed form
    for an exact equicorrelation (:func:`is_equicorrelation`), ``numpy.linalg.cholesky``
    otherwise (LAPACK: machine-dependent in its last bits, SPEC §13.3 — identities across
    machines hold for the equicorrelation case only)."""
    c = np.asarray(correlation, dtype=np.float64)
    if is_equicorrelation(c):
        n = c.shape[0]
        return equicorrelation_cholesky(n, float(c[0, 1]) if n > 1 else 0.0)
    return np.asarray(np.linalg.cholesky(c), dtype=np.float64)


@njit(inline="always")
def lower_row_dot(chol: FloatArray, z: FloatArray, i: int) -> float:  # pragma: no cover - numba
    """``Σ_{m=0}^{i} chol[i, m]·z[m]``, accumulated from 0.0 in ascending ``m`` (the one
    summation order of the multi-asset mixing; module docstring)."""
    acc = 0.0
    for m in range(i + 1):
        acc += chol[i, m] * z[m]
    return acc


@njit(parallel=True, cache=True)
def mix_lower(chol: FloatArray, z: FloatArray, out: FloatArray) -> None:  # pragma: no cover
    """``out[p, j, i] = lower_row_dot(chol, z[p, j], i)`` for independent normals ``z`` of
    shape ``(n_paths, n_steps, n_assets)``: the mixing with a general lower factor."""
    n_paths, n_steps, n = z.shape
    for p in prange(n_paths):
        for j in range(n_steps):
            zr = z[p, j]
            for i in range(n):
                out[p, j, i] = lower_row_dot(chol, zr, i)


@njit(parallel=True, cache=True)
def mix_equi(
    d: FloatArray, ell: FloatArray, z: FloatArray, out: FloatArray
) -> None:  # pragma: no cover - numba
    """The mixing with the closed-form equicorrelation factor in O(n) per step: with the prefix
    ``P_0 = 0``, ``out_i = P_i + d_i·z_i`` and ``P_{i+1} = P_i + ℓ_i·z_i``.  These are the
    additions of :func:`lower_row_dot` on :func:`equicorrelation_cholesky` in the same order, so
    the result is bit-identical to :func:`mix_lower` with that factor (tested)."""
    n_paths, n_steps, n = z.shape
    for p in prange(n_paths):
        for j in range(n_steps):
            pref = 0.0
            for i in range(n):
                zi = z[p, j, i]
                out[p, j, i] = pref + d[i] * zi
                pref = pref + ell[i] * zi


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
        matrix = check_correlation(correlation)
        streams = [
            GaussianDraws(int(seed) + ASSET_SEED_STRIDE * i, n_paths, n_steps, 1, antithetic)
            for i in range(matrix.shape[0])
        ]
        self._setup(streams, matrix, int(seed), int(n_paths), int(n_steps), bool(antithetic))

    def _setup(
        self,
        streams: Sequence[GaussianDraws],
        correlation: FloatArray,
        seed: int,
        n_paths: int,
        n_steps: int,
        antithetic: bool,
    ) -> None:
        """The state of the object from validated inputs (``correlation`` has passed
        :func:`check_correlation`)."""
        self.correlation = correlation
        self.n_assets = int(self.correlation.shape[0])
        if len(streams) != self.n_assets:
            raise ValueError("one independent stream per asset")
        self.chol = cholesky_factor(self.correlation)
        #: exact equicorrelation: the closed-form factor's diagonal and column values
        self.equi = is_equicorrelation(self.correlation)
        self._d = np.ascontiguousarray(np.diag(self.chol))
        self._ell = np.ascontiguousarray(
            np.append(self.chol[-1, :-1], 0.0) if self.equi else np.zeros(self.n_assets)
        )
        self.seed = seed
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.antithetic = antithetic
        self.streams = list(streams)
        self._memo_key: tuple[int, int, int, int] | None = None
        self._memo: FloatArray | None = None

    @classmethod
    def from_streams(
        cls, streams: Sequence[GaussianDraws], correlation: ArrayLike, *, seed: int
    ) -> CorrelatedDraws:
        """Correlated draws on given independent streams (one per asset, one Brownian each, the
        same paths, steps and antithetics) — e.g. the :class:`~volsto.engine.rng.CoarsenedDraws`
        of another object's streams, for a step-refinement study under common random numbers.
        ``seed`` is the label the streams were derived from."""
        if not streams:
            raise ValueError("at least one stream")
        first = streams[0]
        for s in streams:
            if s.n_brownians != 1:
                raise ValueError("each asset stream carries one Brownian")
            if (s.n_paths, s.n_steps, s.antithetic) != (
                first.n_paths,
                first.n_steps,
                first.antithetic,
            ):
                raise ValueError("the streams must share paths, steps and antithetics")
        obj = cls.__new__(cls)
        obj._setup(
            streams,
            check_correlation(correlation),
            int(seed),
            first.n_paths,
            first.n_steps,
            first.antithetic,
        )
        return obj

    def independent_block(self, step0: int, step1: int, p0: int, p1: int) -> FloatArray:
        """The uncorrelated normals ``(n_paths, n_steps, n_assets)`` of the block."""
        out = np.empty((p1 - p0, step1 - step0, self.n_assets))
        for i, s in enumerate(self.streams):
            out[:, :, i] = s.block(step0, step1, p0, p1)[:, :, 0]
        return out

    def block_all(self, step0: int, step1: int, p0: int, p1: int) -> FloatArray:
        """The correlated normals ``(n_paths, n_steps, n_assets)`` of the block (the last block
        is memoised: the assets' kernels request the same block in turn), mixed in the fixed
        summation order of the module docstring."""
        key = (int(step0), int(step1), int(p0), int(p1))
        if self._memo is not None and self._memo_key == key:
            return self._memo
        z = self.independent_block(step0, step1, p0, p1)
        w = np.empty_like(z)
        if self.equi:
            mix_equi(self._d, self._ell, z, w)
        else:
            mix_lower(self.chol, z, w)
        self._memo_key, self._memo = key, w
        return w

    def asset(self, i: int) -> _AssetDraws:
        if not 0 <= i < self.n_assets:
            raise IndexError("asset out of range")
        return _AssetDraws(self, i)

    def with_correlation(self, correlation: ArrayLike) -> CorrelatedDraws:
        """The same independent streams mixed with another correlation (CRN across
        correlations)."""
        return CorrelatedDraws.from_streams(self.streams, correlation, seed=self.seed)

    def __repr__(self) -> str:
        return (
            f"CorrelatedDraws(seed={self.seed}, n_paths={self.n_paths}, n_steps={self.n_steps}, "
            f"n_assets={self.n_assets}, antithetic={self.antithetic})"
        )


__all__ = [
    "ASSET_SEED_STRIDE",
    "CorrelatedDraws",
    "check_correlation",
    "cholesky_factor",
    "constant_correlation",
    "equicorrelation_cholesky",
    "equicorrelation_factor",
    "is_equicorrelation",
    "lower_row_dot",
    "mix_equi",
    "mix_lower",
]
