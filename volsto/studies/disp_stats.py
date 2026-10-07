"""Dispersion study: statistics for overlapping windows (spec §8).

Weekly entries of an ``h``-week window overlap: means and regression slopes carry
Hansen–Hodrick standard errors (uniform kernel, lag ``h − 1``), Newey–West at twice that lag
where the Hansen–Hodrick variance is not positive; ratios carry a block bootstrap (blocks of
the window length).  Terciles of an indicator are cut on its in-sample values and applied
everywhere.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


def _long_run(u: FloatArray, lag: int, bartlett: bool) -> FloatArray:
    """``Σ_k w_k Γ_k`` of the rows of ``u`` (``n × p``), ``|k| ≤ lag``."""
    n = u.shape[0]
    s = u.T @ u
    for k in range(1, min(lag, n - 1) + 1):
        g = u[k:].T @ u[:-k]
        wt = 1.0 - k / (lag + 1.0) if bartlett else 1.0
        s = s + wt * (g + g.T)
    return s


def mean_se(x: Any, lag: int) -> tuple[float, float, int]:
    """Mean of a weekly series, its Hansen–Hodrick standard error (Newey–West at ``2·lag``
    when that variance is not positive) and the number of observations.  NaN dropped."""
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.size
    if n < 3:
        return (float(v.mean()) if n else float("nan")), float("nan"), n
    u = (v - v.mean())[:, None]
    var = float(_long_run(u, lag, False)[0, 0]) / n**2
    if var <= 0:
        var = float(_long_run(u, 2 * lag, True)[0, 0]) / n**2
    return float(v.mean()), float(np.sqrt(max(var, 0.0))), n


def ols(y: Any, X: Any, lag: int) -> dict[str, Any]:
    """Least squares of ``y`` on ``X`` (a constant is **not** added) with Hansen–Hodrick
    standard errors (Newey–West at ``2·lag`` for a coefficient whose variance is not
    positive): ``coef``, ``se``, ``t``, ``r2``, ``n``.  Rows with a NaN are dropped."""
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X = y[ok], X[ok]
    n, p = X.shape
    nan = np.full(p, np.nan)
    if n <= p + 2:
        return {"coef": nan, "se": nan, "t": nan, "r2": float("nan"), "n": n}
    xtx_inv = np.linalg.pinv(X.T @ X)
    coef = xtx_inv @ X.T @ y
    e = y - X @ coef
    u = X * e[:, None]
    cov = xtx_inv @ _long_run(u, lag, False) @ xtx_inv
    bad = np.diag(cov) <= 0
    if bad.any():
        cov_nw = xtx_inv @ _long_run(u, 2 * lag, True) @ xtx_inv
        cov = np.where(bad[:, None] | bad[None, :], cov_nw, cov)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    var = float(np.var(y))
    return {
        "coef": coef, "se": se, "t": coef / np.where(se > 0, se, np.nan),
        "r2": 1.0 - float(np.var(e)) / var if var > 0 else float("nan"), "n": n,
    }  # fmt: skip


def bootstrap_ratio(
    num: Any, den: Any, block: int, n_resamples: int = 2000, seed: int = 11
) -> tuple[float, float, float]:
    """``Σ num / Σ den`` with its 95 % circular-block-bootstrap interval."""
    a = np.column_stack([np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)])
    a = a[np.isfinite(a).all(axis=1)]
    n = len(a)
    if n < 10 or a[:, 1].sum() == 0:
        return float("nan"), float("nan"), float("nan")
    block = max(1, min(block, n))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_resamples, int(np.ceil(n / block))))
    idx = ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(n_resamples, -1)[
        :, :n
    ]
    s = a[idx].sum(axis=1)
    draws = s[:, 0] / s[:, 1]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return float(a[:, 0].sum() / a[:, 1].sum()), float(lo), float(hi)


def tercile_cuts(values_is: Any) -> tuple[float, float]:
    """The two in-sample cut points of an indicator (NaN ignored)."""
    v = np.asarray(values_is, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 30:
        return float("nan"), float("nan")
    lo, hi = np.quantile(v, [1.0 / 3.0, 2.0 / 3.0])
    return float(lo), float(hi)


def tercile(values: pd.Series, cuts: tuple[float, float]) -> pd.Series:
    """``low`` / ``mid`` / ``high`` by the cut points (NaN stays NaN)."""
    lo, hi = cuts
    out = pd.Series(
        np.where(values <= lo, "low", np.where(values > hi, "high", "mid")),
        index=values.index,
        dtype=object,
    )
    out[values.isna() | ~np.isfinite(lo)] = np.nan
    return out


def expanding_percentile(values: pd.Series, min_obs: int = 104) -> pd.Series:
    """Percentile of each value against the values before it (at least ``min_obs`` of them)."""
    v = values.to_numpy(float)
    out = np.full(v.size, np.nan)
    for i in range(v.size):
        past = v[:i]
        past = past[np.isfinite(past)]
        if past.size >= min_obs and np.isfinite(v[i]):
            out[i] = float(np.mean(past <= v[i]))
    return pd.Series(out, index=values.index)


def bh_qvalues(p: Any) -> FloatArray:
    """Benjamini–Hochberg q-values (NaN kept)."""
    p = np.asarray(p, dtype=np.float64)
    q = np.full(p.size, np.nan)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * m / (np.arange(m) + 1.0)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def describe(x: Any, lag: int) -> dict[str, float]:
    """Mean, Hansen–Hodrick t, sd, mean/sd, skew, hit rate, 5 % quantile, worst, n."""
    v = np.asarray(x, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 3:
        return {
            "mean": np.nan,
            "t": np.nan,
            "sd": np.nan,
            "mean/sd": np.nan,
            "skew": np.nan,
            "hit": np.nan,
            "q05": np.nan,
            "worst": np.nan,
            "n": float(v.size),
        }
    m, se, n = mean_se(v, lag)
    sd = float(v.std(ddof=1))
    return {
        "mean": m, "t": m / se if se > 0 else np.nan, "sd": sd,
        "mean/sd": m / sd if sd > 0 else np.nan,
        "skew": float(np.mean(((v - v.mean()) / sd) ** 3)) if sd > 0 else float("nan"),
        "hit": float(np.mean(v > 0)), "q05": float(np.quantile(v, 0.05)),
        "worst": float(v.min()), "n": float(n),
    }  # fmt: skip


def subset_mean_se(y: Any, mask: Any, lag: int) -> tuple[float, float, int]:
    """Mean of ``y`` over the rows of ``mask`` with a Hansen–Hodrick standard error that keeps
    the rows at their place in time (a regression of ``y`` on the indicator of the subset and
    of its complement over the whole series): the right error for a tercile or a rule's leaf,
    whose dates are scattered."""
    v = np.asarray(y, dtype=np.float64)
    m = np.asarray(mask, dtype=bool) & np.isfinite(v)
    n = int(m.sum())
    if n < 3:
        return (float(v[m].mean()) if n else float("nan")), float("nan"), n
    ok = np.isfinite(v)
    X = np.column_stack([m[ok].astype(float), (~m[ok]).astype(float)])
    if X[:, 1].sum() == 0:
        return mean_se(v[ok], lag)
    fit = ols(v[ok], X, lag)
    return float(fit["coef"][0]), float(fit["se"][0]), n


def describe_subset(y: Any, mask: Any, lag: int) -> dict[str, float]:
    """:func:`describe` of the rows of ``mask``, the t-statistic from :func:`subset_mean_se`."""
    v = np.asarray(y, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    out = describe(v[m], 0)
    mean, se, _ = subset_mean_se(v, m, lag)
    out["t"] = mean / se if np.isfinite(se) and se > 0 else float("nan")
    return out
