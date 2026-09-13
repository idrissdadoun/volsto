"""Batch-means standard errors for nonlinear Monte Carlo statistics (covariances, variances).

For a statistic ``f`` of the sample (not a plain mean) the estimator's standard error is taken
from ``n_batches`` contiguous, equally sized batches: ``se = std(f(batch_b)) / sqrt(n_batches)``.
Batches are aligned to antithetic pairs when ``pair_aligned`` (pairs are adjacent paths), so
each batch is an independent replicate.  Used by ``tests/test_bergomi.py`` for the closed-form
covariance checks (owner request after M2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class BatchEstimate:
    value: float
    stderr: float
    n_batches: int

    def z(self, target: float) -> float:
        """Distance from ``target`` in standard errors."""
        return (self.value - target) / self.stderr if self.stderr > 0 else float("inf")

    def __repr__(self) -> str:
        return f"{self.value:.6g} ± {self.stderr:.2g} (batch means, {self.n_batches} batches)"


def batch_means(
    fn: Callable[..., float], *samples: FloatArray, n_batches: int = 40, pair_aligned: bool = True
) -> BatchEstimate:
    """Full-sample value of ``fn(*samples)`` with a batch-means standard error.

    ``samples`` are per-path arrays of equal length ``n``; ``fn`` maps the (sliced) arrays to a
    scalar, e.g. ``lambda a, b: np.cov(a, b)[0, 1]``.  ``n`` must be divisible by ``n_batches``
    (and by ``2 n_batches`` when ``pair_aligned``); trailing paths are dropped otherwise.
    """
    n = int(samples[0].shape[0])
    if any(s.shape[0] != n for s in samples):
        raise ValueError("all samples must have the same length")
    if n_batches < 2:
        raise ValueError("need at least 2 batches")
    unit = 2 if pair_aligned else 1
    size = (n // (n_batches * unit)) * unit
    if size < unit:
        raise ValueError("too few paths for the requested number of batches")
    used = size * n_batches
    full = float(fn(*(s[:used] for s in samples)))
    vals = np.array(
        [fn(*(s[b * size : (b + 1) * size] for s in samples)) for b in range(n_batches)],
        dtype=np.float64,
    )
    se = float(vals.std(ddof=1) / np.sqrt(n_batches))
    return BatchEstimate(full, se, n_batches)
