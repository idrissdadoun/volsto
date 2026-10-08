"""Draws of the local correlation model (SPEC §8.7, M12 part LC2).

For each path and step

    dW = √(1 − λ)·L_low ε + √λ·L_high η

with ``ε ∈ ℝⁿ`` and ``η ∈ ℝʳ`` independent standard normals and ``λ`` the path's frozen value
at the step start: conditionally on the past, ``Cov(dW) = (1 − λ)·L_low L_lowᵀ + λ·L_high
L_highᵀ = ρ(λ)`` (derived), and each ``dW_i`` is exactly ``N(0, 1)``.

**Streams** (common random numbers):

* ``ε_i``: the asset streams of ``CorrelatedDraws(seed, …, R_low)``, i.e. ``GaussianDraws(seed
  + 7919·i, …, 1)`` — unchanged from the constant-correlation layer, so that ``λ ≡ 0`` is that
  layer exactly (:attr:`LocalCorrelationDraws.low`).
* ``η``: ``GaussianDraws(seed + COMMON_FACTOR_SEED_OFFSET + 7919·m, …, min(8, r − 8m))`` for
  ``m = 0 … ⌈r/8⌉ − 1``.  ``104 729 = 13·7919 + 1 782`` is not a multiple of 7919, so an ``η``
  stream never coincides with an asset stream of the same seed, whatever ``n``.
* Antithetics: path ``2p + 1`` uses ``−ε`` and ``−η`` of path ``2p``.  The mixing is linear, so
  the ``R_low`` part of the partner is the exact negation; ``λ`` is read on each path's own
  state, so after the first step the partner's ``dW`` is not the negation — the pairing is on
  the driving normals, which is what the pair-averaged standard error needs.
* A change of ``λ`` (a recalibration, a bump), of a spot or of a surface leaves ``ε`` and ``η``
  unchanged.

The mixing itself runs inside the joint kernel (``lc_kernel.lc_diffuse_block``), with the
summation order of :mod:`volsto.multi.draws`; :func:`mix_local` is the same arithmetic on
arrays, for a given ``λ`` per path and step (tests and diagnostics).

Checked by ``tests/test_local_correlation.py`` (identities I6, statistics C1).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np
from numpy.typing import NDArray

from volsto._numba import njit, prange
from volsto.engine.rng import STRIDE_BROWNIAN, CoarsenedDraws, GaussianDraws
from volsto.multi.draws import ASSET_SEED_STRIDE, CorrelatedDraws, lower_row_dot
from volsto.multi.family import CorrelationFamily

FloatArray = NDArray[np.float64]

#: Seed offset of the ``η`` streams (module docstring: not a multiple of the asset stride).
COMMON_FACTOR_SEED_OFFSET: Final[int] = 104_729


@njit(inline="always")
def high_row_dot(l_high: FloatArray, eta: FloatArray, i: int) -> float:  # pragma: no cover
    """``Σ_m l_high[i, m]·eta[m]``, accumulated from 0.0 in ascending ``m`` (the ``R_high`` part
    of asset ``i``'s normal; with ``R_high = 11ᵀ`` it is ``eta[0]`` exactly)."""
    acc = 0.0
    for m in range(l_high.shape[1]):
        acc += l_high[i, m] * eta[m]
    return acc


@njit(parallel=True, cache=True)
def mix_local(
    eps: FloatArray,
    eta: FloatArray,
    lam: FloatArray,
    low_equi: bool,
    d_low: FloatArray,
    ell_low: FloatArray,
    l_low: FloatArray,
    l_high: FloatArray,
    out: FloatArray,
) -> None:  # pragma: no cover - numba
    """``out[p, j, i] = √(1 − λ)·x_i + √λ·y_i`` with ``λ = lam[p, j]``, ``x = L_low ε`` (the
    equicorrelation prefix or :func:`~volsto.multi.draws.lower_row_dot`) and ``y = L_high η``
    (:func:`high_row_dot`): the statements of the joint kernel's mixing, for a given ``λ``.
    ``eps`` is ``(n_paths, n_steps, n)``, ``eta`` ``(n_paths, n_steps, r)``."""
    n_paths, n_steps, n = eps.shape
    for p in prange(n_paths):
        for j in range(n_steps):
            c1 = np.sqrt(1.0 - lam[p, j])
            c2 = np.sqrt(lam[p, j])
            er = eps[p, j]
            hr = eta[p, j]
            pref = 0.0
            for i in range(n):
                if low_equi:
                    x = pref + d_low[i] * er[i]
                    pref = pref + ell_low[i] * er[i]
                else:
                    x = lower_row_dot(l_low, er, i)
                out[p, j, i] = c1 * x + c2 * high_row_dot(l_high, hr, i)


class LocalCorrelationDraws:
    """The ``ε`` and ``η`` normals of a simulation of the local correlation model (module
    docstring).  ``low`` is the constant-correlation object on the same ``ε`` streams (the
    model at ``λ ≡ 0``)."""

    def __init__(
        self,
        seed: int,
        n_paths: int,
        n_steps: int,
        family: CorrelationFamily,
        antithetic: bool = True,
    ) -> None:
        low = CorrelatedDraws(seed, n_paths, n_steps, family.r_low, antithetic)
        r = family.rank_high
        eta = [
            GaussianDraws(
                int(seed) + COMMON_FACTOR_SEED_OFFSET + ASSET_SEED_STRIDE * m,
                n_paths,
                n_steps,
                min(STRIDE_BROWNIAN, r - STRIDE_BROWNIAN * m),
                antithetic,
            )
            for m in range(math.ceil(r / STRIDE_BROWNIAN))
        ]
        self._setup(low, eta, family)

    def _setup(
        self, low: CorrelatedDraws, eta_streams: Sequence[GaussianDraws], family: CorrelationFamily
    ) -> None:
        if low.n_assets != family.n:
            raise ValueError("the asset streams and the family disagree on the number of assets")
        if sum(s.n_brownians for s in eta_streams) != family.rank_high:
            raise ValueError("the R_high streams must carry rank(R_high) normals")
        for s in eta_streams:
            if (s.n_paths, s.n_steps, s.antithetic) != (low.n_paths, low.n_steps, low.antithetic):
                raise ValueError(
                    "the R_high streams must share paths, steps and antithetics with the "
                    "asset streams"
                )
        self.family = family
        self.low = low
        self.eta_streams = list(eta_streams)
        self.seed = low.seed
        self.n_paths = low.n_paths
        self.n_steps = low.n_steps
        self.antithetic = low.antithetic
        self.n_assets = low.n_assets

    @classmethod
    def from_streams(
        cls, low: CorrelatedDraws, eta_streams: Sequence[GaussianDraws], family: CorrelationFamily
    ) -> LocalCorrelationDraws:
        """Draws on given streams (``low`` carries the ``ε`` streams and ``R_low``)."""
        obj = cls.__new__(cls)
        obj._setup(low, eta_streams, family)
        return obj

    def eps_block(self, step0: int, step1: int, p0: int, p1: int) -> FloatArray:
        """The independent asset normals ``ε``: ``(n_paths, n_steps, n_assets)``."""
        return self.low.independent_block(step0, step1, p0, p1)

    def eta_block(self, step0: int, step1: int, p0: int, p1: int) -> FloatArray:
        """The independent ``R_high`` normals ``η``: ``(n_paths, n_steps, rank_high)``."""
        out = np.empty((p1 - p0, step1 - step0, self.family.rank_high))
        col = 0
        for s in self.eta_streams:
            out[:, :, col : col + s.n_brownians] = s.block(step0, step1, p0, p1)
            col += s.n_brownians
        return out

    def mixed_block(self, step0: int, step1: int, p0: int, p1: int, lam: FloatArray) -> FloatArray:
        """The mixed normals of the block for given ``λ`` values ``(n_paths, n_steps)`` in
        ``[0, 1]`` (:func:`mix_local`; in a simulation the kernel reads ``λ`` on each path's
        state)."""
        eps = self.eps_block(step0, step1, p0, p1)
        eta = self.eta_block(step0, step1, p0, p1)
        lam_ = np.ascontiguousarray(lam, dtype=np.float64)
        if lam_.shape != eps.shape[:2] or np.any(lam_ < 0.0) or np.any(lam_ > 1.0):
            raise ValueError("lam must be (n_paths, n_steps) with values in [0, 1]")
        out = np.empty_like(eps)
        f = self.family
        mix_local(eps, eta, lam_, f.low_equi, f.d_low, f.ell_low, f.l_low, f.l_high, out)
        return out

    def coarsened(self, factor: int) -> LocalCorrelationDraws:
        """The Brownian-consistent coarsening of every stream (``ε`` and ``η``) by ``factor``
        (:class:`~volsto.engine.rng.CoarsenedDraws`): the coarse and the fine simulations see
        the same driving Brownian paths at the coarse times (the Δt check of SPEC §8.7)."""
        low = CorrelatedDraws.from_streams(
            [CoarsenedDraws(s, factor) for s in self.low.streams],
            self.family.r_low,
            seed=self.seed,
        )
        eta = [CoarsenedDraws(s, factor) for s in self.eta_streams]
        return LocalCorrelationDraws.from_streams(low, eta, self.family)

    def with_family(self, family: CorrelationFamily) -> LocalCorrelationDraws:
        """The same ``ε`` streams with another family (common random numbers across families;
        the ``η`` streams are kept when the rank of ``R_high`` is unchanged)."""
        low = self.low.with_correlation(family.r_low)
        if family.rank_high == self.family.rank_high:
            return LocalCorrelationDraws.from_streams(low, self.eta_streams, family)
        return LocalCorrelationDraws(self.seed, self.n_paths, self.n_steps, family, self.antithetic)

    def __repr__(self) -> str:
        return (
            f"LocalCorrelationDraws(seed={self.seed}, n_paths={self.n_paths}, "
            f"n_steps={self.n_steps}, n_assets={self.n_assets}, "
            f"rank_high={self.family.rank_high}, antithetic={self.antithetic})"
        )


__all__ = ["COMMON_FACTOR_SEED_OFFSET", "LocalCorrelationDraws", "high_row_dot", "mix_local"]
