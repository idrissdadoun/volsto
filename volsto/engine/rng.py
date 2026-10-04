"""CRN-keyed Gaussian draws (SPEC §5).

Design: one PCG64 stream per seed, addressed by an *absolute position*

    pos(path, step, brownian) = step · STRIDE_STEP + rng_path · STRIDE_BROWNIAN + brownian

with ``STRIDE_BROWNIAN = 8`` (max Brownians per step) and ``STRIDE_STEP = 2²⁴ · 8``.  Normals are
obtained by inverse transform of one 64-bit output each (``ndtri``), so exactly one raw output is
consumed per normal and the position is deterministic.  Hence the same ``(seed, path, step,
brownian)`` always gives the same normal, independently of ``n_paths``, chunking, ``n_steps`` or
the number of Brownians requested — common-random-number bumps are exact.

Antithetics: path ``2i+1`` uses the negated normals of path ``2i`` (rng path ``i``).
Checked by ``tests/test_engine.py::test_draws_are_crn_exact``.
"""

from __future__ import annotations

import numpy as np
from numpy.random import PCG64
from numpy.typing import NDArray
from scipy.special import ndtri

FloatArray = NDArray[np.float64]

STRIDE_BROWNIAN = 8
MAX_PATHS = 1 << 24
STRIDE_STEP = MAX_PATHS * STRIDE_BROWNIAN
_TWO_M53 = 2.0**-53


class GaussianDraws:
    """Deterministic, position-addressed standard normals for a simulation."""

    def __init__(
        self,
        seed: int,
        n_paths: int,
        n_steps: int,
        n_brownians: int,
        antithetic: bool = True,
    ) -> None:
        if seed < 0:
            raise ValueError("seed must be non-negative")
        if n_paths <= 0 or n_steps <= 0:
            raise ValueError("n_paths and n_steps must be positive")
        if not 1 <= n_brownians <= STRIDE_BROWNIAN:
            raise ValueError(f"n_brownians must lie in [1, {STRIDE_BROWNIAN}]")
        if antithetic and n_paths % 2:
            raise ValueError("n_paths must be even with antithetic=True")
        n_rng = n_paths // 2 if antithetic else n_paths
        if n_rng > MAX_PATHS:
            raise ValueError(f"at most {MAX_PATHS} independent paths are supported")
        self.seed = int(seed)
        self.n_paths = int(n_paths)
        self.n_steps = int(n_steps)
        self.n_brownians = int(n_brownians)
        self.antithetic = bool(antithetic)
        self._bg = PCG64(self.seed)
        self._pos = 0

    def _raw(self, position: int, count: int) -> NDArray[np.uint64]:
        delta = position - self._pos
        if delta < 0:
            self._bg = PCG64(self.seed)
            self._pos = 0
            delta = position
        if delta:
            self._bg.advance(delta)
        out = self._bg.random_raw(count)
        self._pos = position + count
        return np.asarray(out, dtype=np.uint64)

    def _rng_range(self, p0: int, p1: int) -> tuple[int, int]:
        if not 0 <= p0 < p1 <= self.n_paths:
            raise ValueError("path range out of bounds")
        if self.antithetic:
            if p0 % 2 or p1 % 2:
                raise ValueError("antithetic path ranges must be even-aligned")
            return p0 // 2, p1 // 2
        return p0, p1

    def _fill_normals(self, step: int, q0: int, n: int, out: FloatArray) -> None:
        """Write the normals of one step, generator paths ``[q0, q0 + n)``, into ``out`` (shape
        ``(n, n_brownians)``, or ``(2n, n_brownians)`` with antithetics).

        The generator still produces ``STRIDE_BROWNIAN`` outputs per path (the position of a
        normal is fixed by the addressing scheme), but only the ``n_brownians`` columns used are
        converted to uniforms; ``ndtri`` writes straight into the rows of the independent paths
        and the antithetic rows are their negation.  Bit for bit the numbers of the earlier
        layout, which converted every column and copied twice
        (``tests/test_engine.py::test_lean_draws_equal_the_reference_layout``)."""
        raw = self._raw(step * STRIDE_STEP + q0 * STRIDE_BROWNIAN, n * STRIDE_BROWNIAN)
        used = raw.reshape(n, STRIDE_BROWNIAN)[:, : self.n_brownians]
        u = ((used >> np.uint64(11)).astype(np.float64) + 0.5) * _TWO_M53
        if self.antithetic:
            ndtri(u, out=out[0::2])
            np.negative(out[0::2], out=out[1::2])
        else:
            ndtri(u, out=out)

    def normals(self, step: int, p0: int, p1: int) -> FloatArray:
        """Normals for one step, paths ``[p0, p1)``: shape ``(p1 - p0, n_brownians)``."""
        if not 0 <= step < self.n_steps:
            raise ValueError("step out of range")
        q0, q1 = self._rng_range(p0, p1)
        out = np.empty((p1 - p0, self.n_brownians))
        self._fill_normals(step, q0, q1 - q0, out)
        return out

    def block(self, step0: int, step1: int, p0: int, p1: int) -> FloatArray:
        """Normals for steps ``[step0, step1)``: shape ``(p1 - p0, step1 - step0, n_brownians)``."""
        if not 0 <= step0 < step1 <= self.n_steps:
            raise ValueError("step range out of bounds")
        q0, q1 = self._rng_range(p0, p1)
        out = np.empty((p1 - p0, step1 - step0, self.n_brownians))
        for j, s in enumerate(range(step0, step1)):
            self._fill_normals(s, q0, q1 - q0, out[:, j, :])
        return out

    def __repr__(self) -> str:
        return (
            f"GaussianDraws(seed={self.seed}, n_paths={self.n_paths}, n_steps={self.n_steps}, "
            f"n_brownians={self.n_brownians}, antithetic={self.antithetic})"
        )


class CoarsenedDraws(GaussianDraws):
    """Brownian-consistent coarsening of a finer draw stream for refinement studies.

    Coarse step ``s`` aggregates fine steps ``[m s, m s + m)`` as ``Σ z_i / √m``, so the coarse
    and fine simulations see the *same* Brownian path at the coarse times.  Differences such as
    ``P(dt) − P(dt/2)`` are then common-random-number exact, which is what Talay–Tubaro Richardson
    extrapolation and the order diagnostics in :mod:`volsto.engine.richardson` need.
    """

    def __init__(self, fine: GaussianDraws, factor: int) -> None:
        if factor < 1 or fine.n_steps % factor:
            raise ValueError("factor must divide the fine number of steps")
        self.fine = fine
        self.factor = int(factor)
        self.seed = fine.seed
        self.n_paths = fine.n_paths
        self.n_steps = fine.n_steps // self.factor
        self.n_brownians = fine.n_brownians
        self.antithetic = fine.antithetic

    def normals(self, step: int, p0: int, p1: int) -> FloatArray:
        if not 0 <= step < self.n_steps:
            raise ValueError("step out of range")
        m = self.factor
        z = self.fine.block(step * m, (step + 1) * m, p0, p1)
        return np.asarray(z.sum(axis=1) / np.sqrt(m), dtype=np.float64)

    def block(self, step0: int, step1: int, p0: int, p1: int) -> FloatArray:
        """Coarse normals for steps ``[step0, step1)``, one :meth:`normals` call per coarse step
        (the base class fills its block from the generator, which a coarsened stream has not)."""
        if not 0 <= step0 < step1 <= self.n_steps:
            raise ValueError("step range out of bounds")
        out = np.empty((p1 - p0, step1 - step0, self.n_brownians))
        for j, s in enumerate(range(step0, step1)):
            out[:, j, :] = self.normals(s, p0, p1)
        return out

    def __repr__(self) -> str:
        return f"CoarsenedDraws(factor={self.factor}, fine={self.fine!r})"
