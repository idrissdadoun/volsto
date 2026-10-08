"""The affine correlation family of the local correlation model (SPEC §8.7, M12 part LC2).

    ρ(λ) = (1 − λ)·R_low + λ·R_high,        λ ∈ [0, λ_max] ⊂ [0, 1]

**A correlation matrix in every state, by construction** (derived): with ``R_low`` and
``R_high`` correlation matrices (symmetric, positive semi-definite, unit diagonal) and
``λ ∈ [0, 1]``, ``ρ(λ)`` is symmetric with a unit diagonal and ``xᵀρ(λ)x = (1 − λ)·xᵀR_low x +
λ·xᵀR_high x ≥ 0``.  No repair step exists.

**The basket variance is affine in λ** (derived).  With ``u_i = ω_i σ_i ≥ 0`` (current weight
times local vol of each name):

    v_B = uᵀρ(λ)u = a + λ·b,     a = uᵀR_low u,     b = uᵀR_high u − a

and ``b ≥ 0`` whenever ``R_high − R_low ≥ 0`` entrywise — the condition validated here — so
``v_B`` is non-decreasing in ``λ`` and the calibration's inversion is monotone.  With
``R_high = 11ᵀ`` (the default) the condition holds for every correlation matrix ``R_low``, and
``uᵀR_high u = (Σ_i u_i)²``.  For the equicorrelation ``R_low = (1 − ρ_min)·I + ρ_min·11ᵀ``:

    a = (1 − ρ_min)·Σ u_i² + ρ_min·(Σ u_i)²,     b = (1 − ρ_min)·[(Σ u_i)² − Σ u_i²]

and ``ρ(λ)`` is the equicorrelation ``ρ_min + λ(1 − ρ_min)`` — the Cboe implied-correlation
formula applied to local vols (:func:`volsto.multi.analytics.implied_correlation`).

**Draws.**  ``dW = √(1 − λ)·L_low ε + √λ·L_high η`` with ``L_low`` the Cholesky factor of
``R_low`` (:func:`volsto.multi.draws.cholesky_factor`: ``R_low`` must be positive definite) and
``L_high`` a factor of ``R_high`` with as many columns as its rank (:func:`psd_factor`; one
column of ones for ``11ᵀ``): the conditional covariance is ``ρ(λ)``.

**Specifications** (the strings of ``LocalCorrelationConfig``; :func:`family_from_spec`):
``R_low`` is ``"equi"`` (level ``rho_min``), ``"historical-scaled:<window>,<target>"``
(:func:`historical_scaled_correlation`) or ``"matrix:<path>"``; ``R_high`` is ``"ones"`` or
``"matrix:<path>"`` (:func:`load_correlation_matrix`).

Checked by ``tests/test_local_correlation.py`` (the family tests).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.multi.analytics import pairwise_mean_correlation
from volsto.multi.draws import (
    check_correlation,
    cholesky_factor,
    constant_correlation,
    is_equicorrelation,
)

FloatArray = NDArray[np.float64]

#: ``R_high − R_low ≥ −1e-14`` entrywise is required (rounding of a unit diagonal).
HIGH_MINUS_LOW_TOL: Final[float] = 1e-14
#: A correlation matrix whose smallest eigenvalue is below ``−PSD_TOL`` is refused.
PSD_TOL: Final[float] = 1e-10
#: Eigenvalues at or below ``RANK_TOL`` are dropped from the factor of ``R_high``.
RANK_TOL: Final[float] = 1e-12
#: Level of the default ``"equi"`` ``R_low`` (the reference implementation's floor).
DEFAULT_RHO_MIN: Final[float] = 0.02
#: Cap of the equicorrelation ``ρ(λ)`` for ``"equi"`` (the reference's cap): ``λ_max =
#: (rho_max − rho_min)/(1 − rho_min)``.
DEFAULT_RHO_MAX: Final[float] = 0.98


def check_psd_correlation(matrix: ArrayLike, name: str = "correlation") -> FloatArray:
    """The matrix as a float array, validated: square, symmetric, unit diagonal, positive
    semi-definite (smallest eigenvalue at least ``−PSD_TOL``).  Unlike
    :func:`~volsto.multi.draws.check_correlation` a singular matrix (``11ᵀ``) passes."""
    c = np.asarray(matrix, dtype=np.float64)
    if c.ndim != 2 or c.shape[0] != c.shape[1] or c.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty square matrix")
    if not np.all(np.isfinite(c)) or not np.allclose(c, c.T, atol=1e-12):
        raise ValueError(f"{name} must be finite and symmetric")
    if not np.allclose(np.diag(c), 1.0, atol=1e-12):
        raise ValueError(f"{name} must have a unit diagonal")
    low = float(np.linalg.eigvalsh(c).min())
    if low < -PSD_TOL:
        raise ValueError(f"{name} is not positive semi-definite (smallest eigenvalue {low:.3e})")
    return c


def psd_factor(matrix: ArrayLike, tol: float = RANK_TOL) -> FloatArray:
    """``L = U_r Λ_r^{1/2}`` (``n × r``) with ``L Lᵀ = matrix`` for a positive semi-definite
    matrix of rank ``r``: the eigenvectors of the eigenvalues above ``tol``, largest first, each
    column signed so that its largest entry in absolute value is positive.  From LAPACK:
    machine-dependent in its last bits (SPEC §13.3)."""
    c = np.asarray(matrix, dtype=np.float64)
    vals, vecs = np.linalg.eigh(c)
    keep = vals > tol
    if not np.any(keep):
        raise ValueError("the matrix has no eigenvalue above the rank tolerance")
    order = np.argsort(vals[keep])[::-1]
    lam = vals[keep][order]
    u = vecs[:, keep][:, order]
    sign = np.sign(u[np.argmax(np.abs(u), axis=0), np.arange(u.shape[1])])
    return np.ascontiguousarray(u * sign[None, :] * np.sqrt(lam)[None, :])


class CorrelationFamily:
    """``ρ(λ) = (1 − λ)·R_low + λ·R_high`` on ``[0, λ_max]`` (module docstring).

    Attributes: ``r_low``, ``r_high`` (``n × n``); ``l_low`` the lower Cholesky factor of
    ``R_low``; ``low_equi`` whether ``R_low`` is an exact equicorrelation, then ``rho_low`` its
    level and ``d_low`` / ``ell_low`` the closed-form factor's diagonal and column values
    (:func:`volsto.multi.draws.equicorrelation_factor`); ``l_high`` (``n × rank_high``) the
    factor of ``R_high``, ``high_ones`` whether ``R_high = 11ᵀ``; ``lambda_max``; the
    specification strings ``r_low_spec`` / ``r_high_spec`` (metadata).
    """

    def __init__(
        self,
        r_low: ArrayLike,
        r_high: ArrayLike | None = None,
        *,
        lambda_max: float = 1.0,
        r_low_spec: str = "matrix",
        r_high_spec: str | None = None,
    ) -> None:
        try:
            self.r_low = check_correlation(r_low)
        except ValueError as exc:
            raise ValueError(f"R_low: {exc}") from exc
        n = int(self.r_low.shape[0])
        if r_high is None:
            self.r_high = np.ones((n, n))
            self.high_ones = True
        else:
            self.r_high = check_psd_correlation(r_high, "R_high")
            if self.r_high.shape != (n, n):
                raise ValueError("R_low and R_high must have the same size")
            self.high_ones = bool(np.all(self.r_high == 1.0))
        gap = float((self.r_high - self.r_low).min())
        if gap < -HIGH_MINUS_LOW_TOL:
            raise ValueError(
                f"R_high - R_low must be non-negative entrywise (smallest entry {gap:.3e}): the "
                "basket variance would not be monotone in lambda"
            )
        if not (np.isfinite(lambda_max) and 0.0 < lambda_max <= 1.0):
            raise ValueError("lambda_max must lie in (0, 1]")
        self.lambda_max = float(lambda_max)
        self.l_low = cholesky_factor(self.r_low)
        self.low_equi = is_equicorrelation(self.r_low)
        self.rho_low: float | None = (
            (float(self.r_low[0, 1]) if n > 1 else 0.0) if self.low_equi else None
        )
        self.d_low = np.ascontiguousarray(np.diag(self.l_low))
        self.ell_low = np.ascontiguousarray(
            np.append(self.l_low[-1, :-1], 0.0) if self.low_equi else np.zeros(n)
        )
        self.l_high = np.ones((n, 1)) if self.high_ones else psd_factor(self.r_high)
        self.r_low_spec = str(r_low_spec)
        self.r_high_spec = (
            ("ones" if self.high_ones else "matrix") if r_high_spec is None else r_high_spec
        )

    # -- constructors ------------------------------------------------------------------------

    @classmethod
    def equi(
        cls, n: int, rho_min: float = DEFAULT_RHO_MIN, rho_max: float = DEFAULT_RHO_MAX
    ) -> CorrelationFamily:
        """The default family: ``R_low`` the equicorrelation at ``rho_min``, ``R_high = 11ᵀ``,
        so that ``ρ(λ)`` is the equicorrelation ``rho_min + λ(1 − rho_min)``, capped at
        ``rho_max``: ``λ_max = (rho_max − rho_min)/(1 − rho_min)``."""
        if not 0.0 <= rho_min < rho_max <= 1.0:
            raise ValueError("need 0 <= rho_min < rho_max <= 1")
        return cls(
            constant_correlation(n, rho_min),
            None,
            lambda_max=(rho_max - rho_min) / (1.0 - rho_min),
            r_low_spec="equi",
            r_high_spec="ones",
        )

    # -- the family --------------------------------------------------------------------------

    @property
    def n(self) -> int:
        return int(self.r_low.shape[0])

    @property
    def rank_high(self) -> int:
        return int(self.l_high.shape[1])

    def correlation(self, lam: float) -> FloatArray:
        """``ρ(λ)`` for one ``λ ∈ [0, 1]``."""
        if not 0.0 <= lam <= 1.0:
            raise ValueError("lambda must lie in [0, 1]")
        return np.asarray((1.0 - lam) * self.r_low + lam * self.r_high, dtype=np.float64)

    def equicorrelation(self, lam: ArrayLike) -> FloatArray:
        """The level ``ρ_low + λ(1 − ρ_low)`` of ``ρ(λ)`` for the ``"equi"`` / ``"ones"``
        family (refused otherwise: ``ρ(λ)`` is then not an equicorrelation)."""
        if self.rho_low is None or not self.high_ones:
            raise ValueError(
                "rho(lambda) is an equicorrelation only for equi R_low and R_high = ones"
            )
        lam_ = np.asarray(lam, dtype=np.float64)
        return np.asarray(self.rho_low + lam_ * (1.0 - self.rho_low), dtype=np.float64)

    def lambda_of_equicorrelation(self, rho: ArrayLike) -> FloatArray:
        """The inverse of :meth:`equicorrelation`: ``λ = (ρ − ρ_low)/(1 − ρ_low)``."""
        if self.rho_low is None or not self.high_ones:
            raise ValueError(
                "rho(lambda) is an equicorrelation only for equi R_low and R_high = ones"
            )
        rho_ = np.asarray(rho, dtype=np.float64)
        return np.asarray((rho_ - self.rho_low) / (1.0 - self.rho_low), dtype=np.float64)

    def variance_terms(self, u: ArrayLike) -> tuple[FloatArray, FloatArray]:
        """``(a, b)`` with ``a = uᵀR_low u`` and ``b = uᵀR_high u − a`` for ``u`` of shape
        ``(…, n)``: the basket variance is ``a + λ·b`` (module docstring).  The plain matrix
        form; the calibration's per-particle kernel (``lc_kernel.lc_ab``) is checked against
        it."""
        u_ = np.asarray(u, dtype=np.float64)
        if u_.shape[-1] != self.n:
            raise ValueError("u must have one entry per asset")
        a = np.einsum("...i,ij,...j->...", u_, self.r_low, u_)
        high = np.einsum("...i,ij,...j->...", u_, self.r_high, u_)
        return np.asarray(a, dtype=np.float64), np.asarray(high - a, dtype=np.float64)

    def describe(self) -> dict[str, Any]:
        """Metadata of the family (specifications, size, rank, cap)."""
        return {
            "n": self.n,
            "r_low": self.r_low_spec,
            "r_high": self.r_high_spec,
            "rho_low": self.rho_low,
            "rank_high": self.rank_high,
            "lambda_max": self.lambda_max,
        }

    def __repr__(self) -> str:
        return (
            f"CorrelationFamily(n={self.n}, r_low={self.r_low_spec!r}, "
            f"r_high={self.r_high_spec!r}, lambda_max={self.lambda_max:.6g})"
        )


# --------------------------------------------------------------------------------------------
# historical-scaled R_low
# --------------------------------------------------------------------------------------------


def ledoit_wolf_identity(x: ArrayLike) -> tuple[FloatArray, float]:
    """Ledoit–Wolf (2004) shrinkage of the second-moment matrix of the rows of ``x`` (``T × n``,
    already centred) towards a multiple of the identity: ``(S*, shrinkage)`` with

        S = (1/T)·Σ_t x_t x_tᵀ,        ⟨A, B⟩ = tr(ABᵀ)/n,        m = ⟨S, I⟩,
        d² = ‖S − m·I‖²,     b̄² = (1/T²)·Σ_t ‖x_t x_tᵀ − S‖²,     b² = min(b̄², d²),
        S* = (b²/d²)·m·I + (1 − b²/d²)·S

    (recalled; the lemma-3.3 estimators of Ledoit & Wolf, "A well-conditioned estimator for
    large-dimensional covariance matrices", JMVA 2004).  ``shrinkage = b²/d²`` (0 when ``S`` is
    already a multiple of the identity).  Checked against an independent implementation by
    ``tests/test_local_correlation.py::test_historical_scaled_r_low``."""
    xx = np.asarray(x, dtype=np.float64)
    if xx.ndim != 2 or xx.shape[0] < 2:
        raise ValueError("x must be (T, n) with at least two rows")
    t, n = xx.shape
    s = xx.T @ xx / t
    m = float(np.trace(s)) / n
    dev = s - m * np.eye(n)
    d2 = float(np.sum(dev * dev)) / n
    # ‖x xᵀ − S‖² = ‖x‖⁴ − 2 xᵀS x + ‖S‖_F²  (Frobenius, before the 1/n of the inner product)
    norms = np.sum(xx * xx, axis=1)
    quad = np.einsum("ti,ij,tj->t", xx, s, xx)
    b_bar2 = float(np.sum(norms * norms - 2.0 * quad + np.sum(s * s))) / n / (t * t)
    if d2 <= 0.0:
        return s, 0.0
    shrink = min(b_bar2, d2) / d2
    return np.asarray(shrink * m * np.eye(n) + (1.0 - shrink) * s, dtype=np.float64), float(shrink)


@dataclass(frozen=True)
class HistoricalScaled:
    """Output of :func:`historical_scaled_correlation`: the matrix and how it was made."""

    r_low: FloatArray
    scale: float
    shrinkage: float
    mean_correlation: float
    short_names: tuple[int, ...]
    n_observations: int

    def info(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "shrinkage": self.shrinkage,
            "mean_correlation": self.mean_correlation,
            "short_names": list(self.short_names),
            "n_observations": self.n_observations,
        }


def historical_scaled_correlation(
    log_returns: ArrayLike, weights: ArrayLike, target: float
) -> HistoricalScaled:
    """``R_low`` of the ``"historical-scaled:<window>,<target>"`` specification from the
    window's daily log returns (``window × n``, NaN where a name has no observation):

    1. the names with a full window are standardised (mean 0, variance 1 over the window) and
       their correlation is shrunk towards the identity (:func:`ledoit_wolf_identity`), then
       rescaled to a unit diagonal: ``Ĉ``;
    2. the off-diagonals are scaled, ``R_low = (1 − s)·I + s·Ĉ`` with ``s = target / ρ̄_w(Ĉ)``
       and ``ρ̄_w`` the weighted mean pairwise correlation
       (:func:`~volsto.multi.analytics.pairwise_mean_correlation`), so that the weighted mean
       pairwise correlation of ``R_low`` is the target;
    3. ``0 ≤ s ≤ 1`` is required — ``R_low`` is then a convex combination of ``I`` and ``Ĉ``,
       hence positive semi-definite — and refused otherwise, naming ``s``;
    4. a name without a full window gets the target on its off-diagonals (``short_names``, to
       be recorded with the entry).

    The matrix has machine-dependent last bits (a BLAS product): a cache key hashes the data's
    digest, never the matrix (SPEC §13.3)."""
    x = np.asarray(log_returns, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64).ravel()
    if x.ndim != 2 or x.shape[1] != w.size or x.shape[0] < 3:
        raise ValueError("log_returns must be (window, n) with one weight per name")
    if np.any(w < 0) or not np.all(np.isfinite(w)):
        raise ValueError("weights must be non-negative and finite")
    if not 0.0 <= target < 1.0:
        raise ValueError("the target correlation must lie in [0, 1)")
    n = w.size
    full = np.all(np.isfinite(x), axis=0)
    idx = np.flatnonzero(full)
    if idx.size < 2:
        raise ValueError("fewer than two names have a full window of returns")
    xs = x[:, idx]
    sd = xs.std(axis=0)
    if np.any(sd <= 0):
        raise ValueError("a name has constant returns over the window")
    z = (xs - xs.mean(axis=0)) / sd
    shrunk, shrinkage = ledoit_wolf_identity(z)
    dg = np.sqrt(np.diag(shrunk))
    c_hat = shrunk / np.outer(dg, dg)
    np.fill_diagonal(c_hat, 1.0)
    mean = pairwise_mean_correlation(c_hat, w[idx])
    if target == 0.0:
        scale = 0.0
    elif mean <= 0.0:
        raise ValueError(
            f"historical-scaled R_low refused: the mean pairwise correlation is {mean:.4g} <= 0"
        )
    else:
        scale = target / mean
    if not 0.0 <= scale <= 1.0:
        raise ValueError(
            f"historical-scaled R_low refused: s = target / mean correlation = {target:.4g} / "
            f"{mean:.4g} = {scale:.4g} is outside [0, 1] (R_low would not be a convex combination "
            "of I and the historical matrix)"
        )
    r_low = np.full((n, n), float(target))
    r_low[np.ix_(idx, idx)] = scale * c_hat
    np.fill_diagonal(r_low, 1.0)
    r_low = 0.5 * (r_low + r_low.T)
    return HistoricalScaled(
        r_low,
        float(scale),
        float(shrinkage),
        float(mean),
        tuple(int(i) for i in np.flatnonzero(~full)),
        int(x.shape[0]),
    )


# --------------------------------------------------------------------------------------------
# specifications
# --------------------------------------------------------------------------------------------


def parse_r_low_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """``("equi", {})``, ``("historical-scaled", {"window": int, "target": float})`` or
    ``("matrix", {"path": str})`` from the configuration string."""
    if spec == "equi":
        return "equi", {}
    if spec.startswith("historical-scaled:"):
        parts = spec.split(":", 1)[1].split(",")
        try:
            if len(parts) != 2:
                raise ValueError
            window, target = int(parts[0]), float(parts[1])
        except ValueError:
            raise ValueError(
                f"r_low {spec!r}: expected 'historical-scaled:<window>,<target>'"
            ) from None
        if window < 3 or not 0.0 <= target < 1.0:
            raise ValueError(f"r_low {spec!r}: need window >= 3 and 0 <= target < 1")
        return "historical-scaled", {"window": window, "target": target}
    if spec.startswith("matrix:") and len(spec) > len("matrix:"):
        return "matrix", {"path": spec.split(":", 1)[1]}
    raise ValueError(
        f"r_low {spec!r}: expected 'equi', 'historical-scaled:<window>,<target>' or 'matrix:<path>'"
    )


def parse_r_high_spec(spec: str) -> tuple[str, dict[str, Any]]:
    """``("ones", {})`` or ``("matrix", {"path": str})`` from the configuration string."""
    if spec == "ones":
        return "ones", {}
    if spec.startswith("matrix:") and len(spec) > len("matrix:"):
        return "matrix", {"path": spec.split(":", 1)[1]}
    raise ValueError(f"r_high {spec!r}: expected 'ones' or 'matrix:<path>'")


def file_sha256(path: str | Path) -> str:
    """SHA-256 of a file's bytes (a matrix file's digest enters the cache key)."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_correlation_matrix(path: str | Path, names: tuple[str, ...] | None = None) -> FloatArray:
    """A correlation matrix from an ``.npy`` file (``n × n``, in the basket's order) or from a
    ``.csv`` with a ticker header and index (reordered to ``names`` when given).  Not validated
    here: the family validates it."""
    p = Path(path)
    if p.suffix == ".npy":
        return np.asarray(np.load(p, allow_pickle=False), dtype=np.float64)
    if p.suffix == ".csv":
        import pandas as pd

        frame = pd.read_csv(p, index_col=0)
        frame.index = frame.index.astype(str)
        frame.columns = frame.columns.astype(str)
        if names is not None:
            missing = [t for t in names if t not in frame.index or t not in frame.columns]
            if missing:
                raise ValueError(f"{p}: no row / column for {missing}")
            frame = frame.loc[list(names), list(names)]
        elif list(frame.index) != list(frame.columns):
            raise ValueError(f"{p}: the header and the index must list the same tickers")
        return np.asarray(frame.to_numpy(dtype=np.float64), dtype=np.float64)
    raise ValueError(f"{p}: a correlation matrix file is .npy or .csv")


def family_from_spec(
    n: int,
    *,
    r_low: str = "equi",
    r_high: str = "ones",
    rho_min: float = DEFAULT_RHO_MIN,
    rho_max: float = DEFAULT_RHO_MAX,
    lambda_max: float | None = None,
    r_low_matrix: ArrayLike | None = None,
    r_high_matrix: ArrayLike | None = None,
) -> CorrelationFamily:
    """The family of a configuration (SPEC §8.7).

    * ``r_low = "equi"``: the equicorrelation at ``rho_min``; ``λ_max = (rho_max − rho_min)/(1 −
      rho_min)`` and ``lambda_max`` must be left unset.
    * otherwise ``r_low_matrix`` holds the matrix the caller built
      (:func:`historical_scaled_correlation` on the study's returns, or
      :func:`load_correlation_matrix`), and ``λ_max = lambda_max`` (1.0 when unset).
    * ``r_high = "ones"`` or a matrix given as ``r_high_matrix``.
    """
    low_kind, _ = parse_r_low_spec(r_low)
    high_kind, _ = parse_r_high_spec(r_high)
    if not 0.0 <= rho_min < rho_max <= 1.0:
        raise ValueError("need 0 <= rho_min < rho_max <= 1")
    if high_kind == "ones":
        if r_high_matrix is not None:
            raise ValueError("r_high = 'ones' takes no matrix")
        high: ArrayLike | None = None
    else:
        if r_high_matrix is None:
            raise ValueError(f"r_high = {r_high!r} needs its matrix")
        high = r_high_matrix
    if low_kind == "equi":
        if r_low_matrix is not None:
            raise ValueError("r_low = 'equi' takes no matrix")
        if lambda_max is not None:
            raise ValueError(
                "r_low = 'equi' sets lambda_max from rho_min and rho_max: leave it unset"
            )
        low: ArrayLike = constant_correlation(n, rho_min)
        cap = (rho_max - rho_min) / (1.0 - rho_min)
    else:
        if r_low_matrix is None:
            raise ValueError(f"r_low = {r_low!r} needs its matrix")
        low = r_low_matrix
        cap = 1.0 if lambda_max is None else float(lambda_max)
    family = CorrelationFamily(low, high, lambda_max=cap, r_low_spec=r_low, r_high_spec=r_high)
    if family.n != n:
        raise ValueError(f"the correlation matrices are {family.n} x {family.n}, expected {n}")
    return family


__all__ = [
    "DEFAULT_RHO_MAX",
    "DEFAULT_RHO_MIN",
    "CorrelationFamily",
    "HistoricalScaled",
    "check_psd_correlation",
    "family_from_spec",
    "file_sha256",
    "historical_scaled_correlation",
    "ledoit_wolf_identity",
    "load_correlation_matrix",
    "parse_r_high_spec",
    "parse_r_low_spec",
    "psd_factor",
]
