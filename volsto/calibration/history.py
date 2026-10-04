"""Historical estimators from a surface history (SPEC §15 Part 2; Bergomi ch. 7 and ch. 9).

A :class:`SurfaceHistory` holds, per date and per constant time-to-maturity pillar ``T``, the
variance-swap volatility ``vs_vol`` (log-contract strip, :func:`~volsto.market.varswap.
varswap_strike`), the ATMF volatility ``atm_vol`` (``k = 0``), the ATMF skew ``skew =
∂σ̂/∂k|_{k=0}`` (:func:`~volsto.market.surface.atm_skew_numeric`) and ``ln_spot``.  The estimators
read daily increments of these series:

* ``volvol_hist(T) = sqrt(days_per_year) · std(Δ ln vs_vol(·, T))`` — the historical counterpart
  of the instantaneous lognormal volatility of the VS volatility, eq. 7.39 (flat curve: eq. 7.24);
* the cross-pillar correlation matrix of ``Δ ln vs_vol`` — counterpart of the model correlation
  derived from the forward-variance covariances (eq. 7.20 machinery, :func:`vs_vol_correlation`);
* ``SSR_hist(T) = slope(T) / mean skew(·, T)`` with ``slope`` the OLS coefficient of
  ``Δ atm_vol(·, T)`` on ``Δ ln_spot`` — book eq. 9.3 read historically (eq. 9.22 with the skew
  taken out of the sum); the model counterpart is eq. 9.21 (flat) / 9.19 (sloping curve);
* the realised spot/vol correlation ``corr(Δ ln_spot, Δ ln vs_vol(·, T))`` (diagnostic).

Every estimate carries a Newey–West (Bartlett kernel) standard error
(:func:`newey_west_covariance`); the truncation lag follows the rule of thumb
``L = floor(4 (n/100)^(2/9))`` (:func:`newey_west_lags`) unless given.  Correlations also report
the Fisher-z standard error ``(1 − r²)/sqrt(n − 3)`` (serially uncorrelated increments).

The synthetic generator :func:`synthetic_2f_history` simulates one daily path of the pure 2F
model, computes each day's VS vols from the factor state (``ξ_t^u`` explicit in the factors,
eqs. 7.30–7.35) and the ATMF vol and skew by the mixing solution refreshed with that day's
forward-variance curve (:func:`mixing_atmf_batch`); it also records the model's instantaneous
quantities along the path so that the identifiability test can separate estimator error from
the term-structure dependence of the flat-curve formulas.  Checked by ``tests/test_history.py``.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import pandas as pd
import yaml
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

from volsto.analytics.bergomi import (
    alpha_theta,
    atmf_skew_order1,
    atmf_skew_order1_flat,
    ssr_order1_flat,
    vs_vol_of_vol_flat,
)
from volsto.analytics.mixing import parallel_coefficients
from volsto.config import BergomiParams, SimConfig
from volsto.engine.grid import TimeGrid
from volsto.engine.rng import GaussianDraws
from volsto.market.bs import black_vega, implied_vol
from volsto.market.curves import ForwardCurve
from volsto.market.import_hdn import DEFAULT_CALENDAR_REPAIR, CalendarRepairConfig
from volsto.market.surface import ImpliedSurface, atm_skew_numeric
from volsto.market.varswap import ForwardVarianceCurve, varswap_strike
from volsto.market.vendor import HdnSource, VendorSource
from volsto.models.bergomi import BergomiSV, factor_step_covariance, sqrt_covariance

FloatArray = NDArray[np.float64]
log = logging.getLogger(__name__)

DEFAULT_PILLARS: tuple[float, ...] = (1.0 / 12.0, 0.25, 0.5, 1.0, 2.0, 3.0)
DAYS_PER_YEAR = 252
WINDOW_VOL = 250  # SPEC §15 Part 2: vol of vol and correlations
WINDOW_SSR = 60  # SPEC §15 Part 2: SSR (regime dependent)
MIN_INCREMENTS = 20  # an estimate needs at least this many daily increments
_TOL_T = 1e-9


# --------------------------------------------------------------------------------------------
# Newey–West long-run covariance, HAC regression, correlations with standard errors
# --------------------------------------------------------------------------------------------


def newey_west_lags(n: int) -> int:
    """Bartlett truncation lag ``L = floor(4 (n/100)^(2/9))`` (Newey–West 1994 rule of thumb as
    quoted by Stock–Watson), capped at ``n − 2`` and non-negative.  ``n = 60 → 3``, ``250 → 4``,
    ``756 → 6``.  Checked by ``tests/test_history.py::test_newey_west_lag_rule``."""
    if n < 2:
        raise ValueError("need at least two observations")
    return int(min(max(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)), 0), n - 2))


def newey_west_covariance(x: ArrayLike, lags: int) -> FloatArray:
    """Long-run covariance of the rows of ``x`` (``(n, k)`` or ``(n,)``), Bartlett kernel:

    ``S = Γ₀ + Σ_{l=1}^{L} (1 − l/(L+1)) (Γ_l + Γ_lᵀ)``, ``Γ_l = (1/n) Σ_{t=l}^{n−1} x̃_t x̃_{t−l}ᵀ``
    on the demeaned series ``x̃``.  ``Var(sample mean) ≈ S / n``.  Source: Newey–West (1987).
    Checked by ``tests/test_history.py::test_newey_west_hand_computation`` and
    ``::test_newey_west_ar1_long_run_variance``.
    """
    X = np.asarray(x, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError("x must be (n,) or (n, k) with n >= 2")
    n = X.shape[0]
    if not 0 <= lags <= n - 1:
        raise ValueError(f"lags must lie in [0, n - 1] (got {lags} for n = {n})")
    Xc = X - X.mean(axis=0)
    S = Xc.T @ Xc / n
    for lag in range(1, lags + 1):
        G = Xc[lag:].T @ Xc[:-lag] / n
        S += (1.0 - lag / (lags + 1.0)) * (G + G.T)
    return np.asarray(S, dtype=np.float64)


def newey_west_variance(x: ArrayLike, lags: int) -> float:
    """Scalar long-run variance of a series (see :func:`newey_west_covariance`)."""
    X = np.asarray(x, dtype=np.float64)
    if X.ndim != 1:
        raise ValueError("x must be one-dimensional")
    return float(newey_west_covariance(X, lags)[0, 0])


@dataclass(frozen=True)
class HacRegression:
    """OLS of ``y`` on ``x`` with a Newey–West standard error of the slope.

    ``influence`` is the slope's influence function ``x̃_t e_t / mean(x̃²)`` (so that
    ``Var(slope) = LRV(influence)/n``); it lets callers combine the slope with other sample
    means by the delta method.
    """

    slope: float
    slope_se: float
    intercept: float
    r2: float
    n: int
    lags: int
    influence: FloatArray

    def __repr__(self) -> str:
        return (
            f"HacRegression(slope={self.slope:.6g} ± {self.slope_se:.3g}, r2={self.r2:.3f}, "
            f"n={self.n}, lags={self.lags})"
        )


def hac_ols(x: ArrayLike, y: ArrayLike, lags: int, *, intercept: bool) -> HacRegression:
    """Slope of ``y`` on ``x`` (with or without an intercept) and its HAC standard error
    ``sqrt(LRV(x̃ e)/n) / mean(x̃²)`` with ``x̃`` the (de)meaned regressor and ``e`` the residual.
    Checked by ``tests/test_history.py::test_hac_ols_recovers_slope``."""
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.ndim != 1 or xa.shape != ya.shape or xa.size < 3:
        raise ValueError("x and y must be one-dimensional of equal length >= 3")
    n = xa.size
    xc = xa - xa.mean() if intercept else xa
    yc = ya - ya.mean() if intercept else ya
    mxx = float(np.mean(xc * xc))
    if mxx <= 0.0:
        raise ValueError("regressor has zero variance")
    slope = float(np.mean(xc * yc) / mxx)
    e = yc - slope * xc
    psi = xc * e / mxx
    se = float(np.sqrt(newey_west_variance(psi, lags) / n))
    sst = float(np.sum(yc * yc))
    r2 = float(1.0 - np.sum(e * e) / sst) if sst > 0 else float("nan")
    c = float(ya.mean() - slope * xa.mean()) if intercept else 0.0
    return HacRegression(slope, se, c, r2, n, lags, psi)


@dataclass(frozen=True)
class CorrelationEstimate:
    """Sample correlation with two standard errors: Fisher-z (``(1 − r²)/sqrt(n − 3)``, serially
    uncorrelated increments) and HAC (delta method on the moments ``(x̃ỹ, x̃², ỹ²)`` with their
    Newey–West covariance)."""

    corr: float
    se_fisher: float
    se_hac: float
    n: int
    lags: int

    def __repr__(self) -> str:
        return (
            f"CorrelationEstimate({self.corr:+.4f}, se_fisher={self.se_fisher:.4f}, "
            f"se_hac={self.se_hac:.4f}, n={self.n})"
        )


def correlation_with_se(x: ArrayLike, y: ArrayLike, lags: int) -> CorrelationEstimate:
    """Pearson correlation of two series with Fisher-z and HAC standard errors.

    HAC: with ``m = (mean x̃ỹ, mean x̃², mean ỹ²)``, ``r = m₁/sqrt(m₂m₃)``, gradient
    ``g = (1/sqrt(m₂m₃), −r/(2m₂), −r/(2m₃))`` and ``Var(r) = gᵀ S g / n``, ``S`` the Newey–West
    covariance of the three moment series.  Checked by
    ``tests/test_history.py::test_gaussian_synthetic_recovery``.
    """
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if xa.ndim != 1 or xa.shape != ya.shape or xa.size < 4:
        raise ValueError("x and y must be one-dimensional of equal length >= 4")
    n = xa.size
    xc = xa - xa.mean()
    yc = ya - ya.mean()
    mom = np.column_stack([xc * yc, xc * xc, yc * yc])
    mu = mom.mean(axis=0)
    if mu[1] <= 0 or mu[2] <= 0:
        raise ValueError("a series has zero variance")
    r = float(mu[0] / np.sqrt(mu[1] * mu[2]))
    g = np.array([1.0 / np.sqrt(mu[1] * mu[2]), -r / (2.0 * mu[1]), -r / (2.0 * mu[2])])
    S = newey_west_covariance(mom, lags) / n
    se_hac = float(np.sqrt(max(g @ S @ g, 0.0)))
    se_fisher = float((1.0 - r * r) / np.sqrt(n - 3))
    return CorrelationEstimate(r, se_fisher, se_hac, n, lags)


# --------------------------------------------------------------------------------------------
# Pillar sources: what a dated snapshot must provide
# --------------------------------------------------------------------------------------------


@runtime_checkable
class PillarSource(Protocol):
    """Per-date provider of the pillar quantities the estimators read."""

    @property
    def spot(self) -> float: ...

    def atm_vol(self, T: float) -> float: ...

    def atm_skew(self, T: float) -> float: ...

    def vs_vol(self, T: float) -> float: ...


class SurfacePillarSource:
    """Pillar quantities of an :class:`~volsto.market.surface.ImpliedSurface`: ``atm_vol`` at
    ``k = 0``, ``atm_skew`` by central differences of half-width ``skew_h`` in ``k``
    (:func:`~volsto.market.surface.atm_skew_numeric`), ``vs_vol = sqrt(K_var(T))`` from the
    log-contract strip (:func:`~volsto.market.varswap.varswap_strike`), ``spot`` from the forward
    curve."""

    def __init__(
        self, surface: ImpliedSurface, *, skew_h: float = 1e-3, close: float | None = None
    ) -> None:
        if skew_h <= 0:
            raise ValueError("skew_h must be positive")
        if close is not None and close <= 0:
            raise ValueError("close must be positive")
        self.surface = surface
        self.skew_h = float(skew_h)
        self.close = None if close is None else float(close)

    @property
    def spot(self) -> float:
        """The realised level the history's ``ln_spot`` uses: the official close when the
        source was built from an importer snapshot (:meth:`SurfaceHistory.from_snapshots`;
        the snapshot's ``spot`` is the level the option quotes imply, SPEC §13.1), else the
        surface's spot (synthetic surfaces, where the two coincide)."""
        if self.close is not None:
            return self.close
        return float(self.surface.forward_curve.spot)

    def atm_vol(self, T: float) -> float:
        return float(self.surface.atm_vol(T))

    def atm_skew(self, T: float) -> float:
        return float(atm_skew_numeric(self.surface, T, self.skew_h))

    def vs_vol(self, T: float) -> float:
        return float(np.sqrt(varswap_strike(self.surface, T)))


@dataclass(frozen=True)
class PillarQuotes:
    """A lightweight per-date source holding the pillar quantities directly (synthetic
    histories: the VS vol comes from the model's forward-variance curve, not from a strip)."""

    pillars: FloatArray
    vs_vols: FloatArray
    atm_vols: FloatArray
    skews: FloatArray
    spot: float

    def _index(self, T: float) -> int:
        i = int(np.argmin(np.abs(self.pillars - T)))
        if abs(self.pillars[i] - T) > _TOL_T:
            raise KeyError(f"no quote at T={T}; pillars are {self.pillars.tolist()}")
        return i

    def atm_vol(self, T: float) -> float:
        return float(self.atm_vols[self._index(T)])

    def atm_skew(self, T: float) -> float:
        return float(self.skews[self._index(T)])

    def vs_vol(self, T: float) -> float:
        return float(self.vs_vols[self._index(T)])


# --------------------------------------------------------------------------------------------
# Estimate containers
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VolVolEstimate:
    """``volvol_hist(T)`` with its Newey–West standard error (delta method on the variance of
    ``Δ ln vs_vol``: ``se(σ̂) = se(σ̂²)/(2σ̂)``, ``se(σ̂²) = sqrt(LRV((x − x̄)²)/n)``)."""

    T: float
    window: int
    end_date: pd.Timestamp
    volvol: float
    se: float
    n: int
    lags: int
    days_per_year: int

    def __repr__(self) -> str:
        return (
            f"VolVolEstimate(T={self.T:.4g}, {self.volvol:.4f} ± {self.se:.4f}, "
            f"window={self.window}, end={self.end_date.date()})"
        )


@dataclass(frozen=True)
class SSREstimate:
    """``SSR_hist(T) = slope / mean skew`` with the delta-method standard error
    ``Var(R) = Var(b)/s̄² + b² Var(s̄)/s̄⁴ − 2 b Cov(b, s̄)/s̄³`` from the Newey–West covariance of
    the slope's influence function and the skew series."""

    T: float
    window: int
    end_date: pd.Timestamp
    ssr: float
    se: float
    slope: float
    slope_se: float
    intercept: float
    r2: float
    mean_skew: float
    mean_skew_se: float
    n: int
    lags: int

    def __repr__(self) -> str:
        return (
            f"SSREstimate(T={self.T:.4g}, SSR={self.ssr:.3f} ± {self.se:.3f}, "
            f"slope={self.slope:+.4f} ± {self.slope_se:.4f}, r2={self.r2:.2f}, "
            f"mean_skew={self.mean_skew:+.4f}, window={self.window}, end={self.end_date.date()})"
        )


@dataclass(frozen=True)
class PillarCorrelations:
    """Cross-pillar correlation matrix of ``Δ ln vs_vol`` with Fisher-z and HAC standard errors
    (matrices indexed by pillar in both directions)."""

    window: int
    end_date: pd.Timestamp
    corr: pd.DataFrame
    se_fisher: pd.DataFrame
    se_hac: pd.DataFrame
    n: int
    lags: int

    def __repr__(self) -> str:
        return (
            f"PillarCorrelations(window={self.window}, end={self.end_date.date()}, n={self.n})\n"
            + self.corr.round(4).to_string()
        )


# --------------------------------------------------------------------------------------------
# SurfaceHistory
# --------------------------------------------------------------------------------------------

REQUIRED_COLUMNS = ("date", "T", "vs_vol", "atm_vol", "skew", "ln_spot")


class SurfaceHistory:
    """Dated pillar quantities of an implied-surface history (SPEC §15 Part 2).

    ``frame`` is the tidy table with columns ``date, T, vs_vol, atm_vol, skew, ln_spot`` (one row
    per date and pillar, every date carrying every pillar); it is validated at construction:
    strictly increasing dates, positive ``vs_vol`` and ``atm_vol``, no NaN, ``ln_spot`` constant
    within a date — each error names the offending date (and pillar).  Wide views are exposed
    as ``(n_dates, n_pillars)`` arrays and DataFrames.  Checked by
    ``tests/test_history.py::test_history_construction`` and ``::test_history_validation``.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
        if missing:
            raise ValueError(f"history frame lacks columns {missing}")
        df = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
        df["date"] = pd.to_datetime(df["date"])
        df["T"] = df["T"].astype(float)
        for col in ("vs_vol", "atm_vol", "skew", "ln_spot"):
            df[col] = df[col].astype(float)
        df = df.sort_values(["date", "T"], kind="stable").reset_index(drop=True)
        dates = pd.DatetimeIndex(pd.unique(df["date"]))
        pillars = np.unique(df["T"].to_numpy())
        if dates.size == 0:
            raise ValueError("history is empty")
        if np.any(pillars <= 0):
            raise ValueError(f"pillars must be positive: {pillars.tolist()}")
        n_d, n_p = dates.size, pillars.size
        if len(df) != n_d * n_p:
            counts = df.groupby("date").size()
            bad = counts[counts != n_p]
            first = bad.index[0] if len(bad) else dates[0]
            raise ValueError(
                f"every date must carry every pillar ({n_p} pillars): date "
                f"{pd.Timestamp(first).date()} has {int(counts.get(first, 0))} rows"
            )
        if not np.all(np.diff(dates.to_numpy().astype("datetime64[ns]")) > np.timedelta64(0, "ns")):
            d = dates.to_numpy()
            i = int(np.flatnonzero(np.diff(d) <= np.timedelta64(0))[0])
            raise ValueError(
                f"dates must be strictly increasing: {pd.Timestamp(d[i]).date()} is followed "
                f"by {pd.Timestamp(d[i + 1]).date()}"
            )
        # NaN / positivity checks, naming the date and pillar
        for col in ("vs_vol", "atm_vol", "skew", "ln_spot"):
            vals = df[col].to_numpy()
            bad_mask = ~np.isfinite(vals)
            if col in ("vs_vol", "atm_vol"):
                bad_mask |= vals <= 0.0
            if np.any(bad_mask):
                i = int(np.flatnonzero(bad_mask)[0])
                what = "NaN/inf" if not np.isfinite(vals[i]) else f"non-positive ({vals[i]:g})"
                raise ValueError(
                    f"{col} is {what} on {pd.Timestamp(df['date'].iloc[i]).date()} "
                    f"at T={df['T'].iloc[i]:g}"
                )
        # layout is complete: reshape to (n_dates, n_pillars)
        pillar_check = np.tile(pillars, n_d)
        if not np.allclose(df["T"].to_numpy(), pillar_check, rtol=0, atol=_TOL_T):
            raise ValueError("the set of pillars differs between dates")
        self._vs_vol = df["vs_vol"].to_numpy().reshape(n_d, n_p)
        self._atm_vol = df["atm_vol"].to_numpy().reshape(n_d, n_p)
        self._skew = df["skew"].to_numpy().reshape(n_d, n_p)
        ln_s = df["ln_spot"].to_numpy().reshape(n_d, n_p)
        if np.any(np.abs(ln_s - ln_s[:, :1]) > 1e-12):
            i = int(np.flatnonzero(np.any(np.abs(ln_s - ln_s[:, :1]) > 1e-12, axis=1))[0])
            raise ValueError(f"ln_spot differs across pillars on {dates[i].date()}")
        self._ln_spot = ln_s[:, 0].copy()
        self._dates = dates
        self._pillars = pillars
        self._frame = df

    # -- constructors --------------------------------------------------------------------------

    @classmethod
    def from_arrays(
        cls,
        dates: Sequence[pd.Timestamp] | pd.DatetimeIndex | Sequence[str],
        pillars: ArrayLike,
        vs_vol: ArrayLike,
        atm_vol: ArrayLike,
        skew: ArrayLike,
        ln_spot: ArrayLike,
    ) -> SurfaceHistory:
        """From ``(n_dates, n_pillars)`` arrays and a per-date ``ln_spot``."""
        d = pd.DatetimeIndex(pd.to_datetime([pd.Timestamp(x) for x in dates]))
        p = np.asarray(pillars, dtype=np.float64).ravel()
        vs = np.asarray(vs_vol, dtype=np.float64)
        atm = np.asarray(atm_vol, dtype=np.float64)
        sk = np.asarray(skew, dtype=np.float64)
        ls = np.asarray(ln_spot, dtype=np.float64).ravel()
        shape = (d.size, p.size)
        for name, arr in (("vs_vol", vs), ("atm_vol", atm), ("skew", sk)):
            if arr.shape != shape:
                raise ValueError(f"{name} must have shape {shape}, got {arr.shape}")
        if ls.shape != (d.size,):
            raise ValueError(f"ln_spot must have shape {(d.size,)}, got {ls.shape}")
        frame = pd.DataFrame(
            {
                "date": np.repeat(d.to_numpy(), p.size),
                "T": np.tile(p, d.size),
                "vs_vol": vs.ravel(),
                "atm_vol": atm.ravel(),
                "skew": sk.ravel(),
                "ln_spot": np.repeat(ls, p.size),
            }
        )
        return cls(frame)

    @classmethod
    def from_sources(
        cls,
        dates: Iterable[pd.Timestamp | str],
        sources: Iterable[PillarSource],
        pillars: Sequence[float] = DEFAULT_PILLARS,
    ) -> SurfaceHistory:
        """From dated :class:`PillarSource` objects (one per date)."""
        p = np.asarray(pillars, dtype=np.float64)
        if p.ndim != 1 or p.size == 0 or np.any(p <= 0) or np.any(np.diff(p) <= 0):
            raise ValueError("pillars must be positive and strictly increasing")
        ds = [pd.Timestamp(d) for d in dates]
        srcs = list(sources)
        if len(ds) != len(srcs):
            raise ValueError("one source per date is required")
        n_d, n_p = len(ds), p.size
        vs = np.empty((n_d, n_p))
        atm = np.empty((n_d, n_p))
        sk = np.empty((n_d, n_p))
        ls = np.empty(n_d)
        for i, (d, s) in enumerate(zip(ds, srcs)):
            try:
                ls[i] = np.log(float(s.spot))
                for j, T in enumerate(p):
                    vs[i, j] = s.vs_vol(float(T))
                    atm[i, j] = s.atm_vol(float(T))
                    sk[i, j] = s.atm_skew(float(T))
            except Exception as exc:  # re-raise naming the date
                raise ValueError(f"{d.date()}: {exc}") from exc
        return cls.from_arrays(ds, p, vs, atm, sk, ls)

    @classmethod
    def from_surfaces(
        cls,
        pairs: Iterable[tuple[pd.Timestamp | str, ImpliedSurface]],
        pillars: Sequence[float] = DEFAULT_PILLARS,
        *,
        skew_h: float = 1e-3,
    ) -> SurfaceHistory:
        """From ``(date, ImpliedSurface)`` pairs (any surface with ``implied_vol_k`` / ``atm_vol``
        / ``forward`` / ``forward_curve``): per pillar the log-contract VS vol, the ATMF vol and
        the numerical ATMF skew."""
        items = list(pairs)
        return cls.from_sources(
            [d for d, _ in items],
            [SurfacePillarSource(s, skew_h=skew_h) for _, s in items],
            pillars,
        )

    @classmethod
    def from_snapshots(
        cls,
        paths: Iterable[str | Path],
        pillars: Sequence[float] = DEFAULT_PILLARS,
        *,
        skew_h: float = 1e-3,
    ) -> SurfaceHistory:
        """From importer snapshot YAML configs (:mod:`volsto.market.import_hdn`, loaded with
        :func:`volsto.market.loaders.load_ssvi_surface`); the date is ``provenance.quote_date``,
        else a ``YYYY-MM-DD`` in the file name.  Files are sorted by date."""
        from volsto.config import MarketConfig, load_yaml
        from volsto.market.loaders import load_ssvi_surface

        dated = sorted((snapshot_date(p), Path(p)) for p in paths)
        if not dated:
            raise ValueError("no snapshot files given")
        sources: list[PillarSource] = []
        for _, p in dated:
            close = load_yaml(p, MarketConfig, section="market").close
            if close is None:
                raise ValueError(
                    f"{p}: the snapshot has no market.close (imported before 2026-09-22, when "
                    "spot became the option-implied level): re-import it"
                )
            sources.append(SurfacePillarSource(load_ssvi_surface(p), skew_h=skew_h, close=close))
        return cls.from_sources([d for d, _ in dated], sources, pillars)

    @classmethod
    def from_callable(
        cls,
        dates: Iterable[pd.Timestamp | str],
        surface_for_date: Callable[[pd.Timestamp], PillarSource | ImpliedSurface],
        pillars: Sequence[float] = DEFAULT_PILLARS,
        *,
        skew_h: float = 1e-3,
    ) -> SurfaceHistory:
        """Synthetic constructor: ``surface_for_date(date)`` returns an ``ImpliedSurface`` or a
        :class:`PillarSource` (e.g. :class:`PillarQuotes`) for each date."""
        ds = [pd.Timestamp(d) for d in dates]
        srcs: list[PillarSource] = []
        for d in ds:
            obj = surface_for_date(d)
            if isinstance(obj, ImpliedSurface):
                srcs.append(SurfacePillarSource(obj, skew_h=skew_h))
            elif isinstance(obj, PillarSource):
                srcs.append(obj)
            else:
                raise TypeError(f"{d.date()}: expected an ImpliedSurface or PillarSource")
        return cls.from_sources(ds, srcs, pillars)

    # -- views -----------------------------------------------------------------------------------

    @property
    def frame(self) -> pd.DataFrame:
        """Tidy table ``date, T, vs_vol, atm_vol, skew, ln_spot`` (a copy)."""
        return self._frame.copy()

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self._dates

    @property
    def pillars(self) -> FloatArray:
        return self._pillars.copy()

    @property
    def n_dates(self) -> int:
        return int(self._dates.size)

    @property
    def ln_spot(self) -> pd.Series:
        return pd.Series(self._ln_spot.copy(), index=self._dates, name="ln_spot")

    def _wide(self, arr: FloatArray, name: str) -> pd.DataFrame:
        df = pd.DataFrame(arr.copy(), index=self._dates, columns=list(self._pillars))
        df.columns.name = "T"
        df.index.name = "date"
        df.attrs["name"] = name
        return df

    @property
    def vs_vol(self) -> pd.DataFrame:
        """``(n_dates × n_pillars)`` variance-swap volatilities."""
        return self._wide(self._vs_vol, "vs_vol")

    @property
    def atm_vol(self) -> pd.DataFrame:
        return self._wide(self._atm_vol, "atm_vol")

    @property
    def skew(self) -> pd.DataFrame:
        return self._wide(self._skew, "skew")

    def pillar_index(self, T: float) -> int:
        i = int(np.argmin(np.abs(self._pillars - T)))
        if abs(self._pillars[i] - T) > _TOL_T:
            raise KeyError(f"T={T} is not a pillar of this history: {self._pillars.tolist()}")
        return i

    def date_index(self, end: pd.Timestamp | str | None) -> int:
        if end is None:
            return self.n_dates - 1
        ts = pd.Timestamp(end)
        pos = self._dates.get_indexer(pd.DatetimeIndex([ts]))
        if pos[0] < 0:
            raise KeyError(f"{ts.date()} is not a date of this history")
        return int(pos[0])

    def subset(
        self, start: pd.Timestamp | str | None = None, end: pd.Timestamp | str | None = None
    ) -> SurfaceHistory:
        """Dates in ``[start, end]`` (inclusive)."""
        df = self._frame
        if start is not None:
            df = df[df["date"] >= pd.Timestamp(start)]
        if end is not None:
            df = df[df["date"] <= pd.Timestamp(end)]
        return SurfaceHistory(df)

    def __len__(self) -> int:
        return self.n_dates

    def __repr__(self) -> str:
        return (
            f"SurfaceHistory({self.n_dates} dates {self._dates[0].date()} → "
            f"{self._dates[-1].date()}, pillars={np.round(self._pillars, 4).tolist()})"
        )

    # -- increments and windows ---------------------------------------------------------------

    def _window(self, window: int, end: pd.Timestamp | str | None) -> tuple[slice, int]:
        """Increment slice ``[e − window, e)`` (increment ``i`` is between dates ``i`` and
        ``i + 1``) for the window of ``window`` increments ending at date ``end``."""
        if window < MIN_INCREMENTS:
            raise ValueError(
                f"window must be at least {MIN_INCREMENTS} daily increments, got {window}"
            )
        e = self.date_index(end)
        if window > e:
            raise ValueError(
                f"a window of {window} increments needs {window + 1} dates ending "
                f"{self._dates[e].date()}; the history has {e + 1} date(s) up to that date "
                f"({self.n_dates} in total): nothing can be estimated"
            )
        return slice(e - window, e), e

    def d_ln_vs_vol(self) -> FloatArray:
        """``Δ ln vs_vol`` — ``(n_dates − 1, n_pillars)``."""
        return np.asarray(np.diff(np.log(self._vs_vol), axis=0), dtype=np.float64)

    def d_atm_vol(self) -> FloatArray:
        return np.asarray(np.diff(self._atm_vol, axis=0), dtype=np.float64)

    def d_ln_spot(self) -> FloatArray:
        return np.asarray(np.diff(self._ln_spot), dtype=np.float64)

    # -- estimators ----------------------------------------------------------------------------

    def volvol_hist(
        self,
        T: float,
        window: int = WINDOW_VOL,
        *,
        end: pd.Timestamp | str | None = None,
        days_per_year: int = DAYS_PER_YEAR,
        lags: int | None = None,
    ) -> VolVolEstimate:
        """``sqrt(days_per_year) · std(Δ ln vs_vol(·, T))`` over the last ``window`` increments
        ending at ``end`` (SPEC §15 Part 2 item 1; model counterpart eq. 7.39 / 7.24), with the
        Newey–West standard error of :class:`VolVolEstimate`.  Checked by
        ``tests/test_history.py::test_gaussian_synthetic_recovery`` and the identifiability
        tests."""
        if days_per_year <= 0:
            raise ValueError("days_per_year must be positive")
        sl, e = self._window(window, end)
        j = self.pillar_index(T)
        x = self.d_ln_vs_vol()[sl, j]
        n = x.size
        L = newey_west_lags(n) if lags is None else lags
        s2 = float(x.var(ddof=1))
        y = (x - x.mean()) ** 2
        se_s2 = float(np.sqrt(newey_west_variance(y, L) / n))
        vv = float(np.sqrt(days_per_year * s2))
        se = float(np.sqrt(days_per_year) * se_s2 / (2.0 * np.sqrt(s2)))
        return VolVolEstimate(float(T), window, self._dates[e], vv, se, n, L, days_per_year)

    def pillar_correlations(
        self,
        window: int = WINDOW_VOL,
        *,
        end: pd.Timestamp | str | None = None,
        lags: int | None = None,
    ) -> PillarCorrelations:
        """Correlation matrix of ``Δ ln vs_vol`` across pillars (SPEC item 1) with Fisher-z and
        HAC standard errors (:func:`correlation_with_se`); model counterpart
        :func:`vs_vol_correlation` (eq. 7.20 machinery)."""
        sl, e = self._window(window, end)
        X = self.d_ln_vs_vol()[sl]
        n = X.shape[0]
        L = newey_west_lags(n) if lags is None else lags
        m = self._pillars.size
        corr = np.eye(m)
        sef = np.zeros((m, m))
        seh = np.zeros((m, m))
        for i in range(m):
            for j in range(i + 1, m):
                c = correlation_with_se(X[:, i], X[:, j], L)
                corr[i, j] = corr[j, i] = c.corr
                sef[i, j] = sef[j, i] = c.se_fisher
                seh[i, j] = seh[j, i] = c.se_hac
        idx = list(self._pillars)
        return PillarCorrelations(
            window,
            self._dates[e],
            pd.DataFrame(corr, index=idx, columns=idx),
            pd.DataFrame(sef, index=idx, columns=idx),
            pd.DataFrame(seh, index=idx, columns=idx),
            n,
            L,
        )

    def spot_vol_correlation(
        self,
        T: float,
        window: int = WINDOW_VOL,
        *,
        end: pd.Timestamp | str | None = None,
        lags: int | None = None,
    ) -> CorrelationEstimate:
        """``corr(Δ ln_spot, Δ ln vs_vol(·, T))`` (SPEC item 4, diagnostic); model counterpart
        :func:`spot_vs_vol_correlation_flat`."""
        sl, _ = self._window(window, end)
        j = self.pillar_index(T)
        x = self.d_ln_spot()[sl]
        y = self.d_ln_vs_vol()[sl, j]
        L = newey_west_lags(x.size) if lags is None else lags
        return correlation_with_se(x, y, L)

    def ssr_hist(
        self,
        T: float,
        window: int = WINDOW_SSR,
        *,
        end: pd.Timestamp | str | None = None,
        lags: int | None = None,
        intercept: bool = True,
    ) -> SSREstimate:
        """Skew-stickiness ratio ``R_T = slope / mean skew`` (SPEC item 2; book eq. 9.3 read
        historically, eq. 9.22 with the skew taken out of the sum): ``slope`` is the OLS
        coefficient of ``Δ atm_vol(·, T)`` on ``Δ ln_spot`` over the window (with an intercept
        by default, absorbing drifts), ``mean skew`` the average of ``skew(·, T)`` over the
        increments' start dates (the ``S_{T,i}`` of eq. 9.22).  Standard errors: Newey–West for
        the slope, delta method for the ratio (:class:`SSREstimate`); ``r2`` is the regression's.
        Model counterparts: eq. 9.21 (flat curve), 9.19 (sloping).  Checked by
        ``tests/test_history.py::test_gaussian_synthetic_recovery`` and the identifiability
        tests."""
        sl, e = self._window(window, end)
        j = self.pillar_index(T)
        x = self.d_ln_spot()[sl]
        y = self.d_atm_vol()[sl, j]
        n = x.size
        L = newey_west_lags(n) if lags is None else lags
        reg = hac_ols(x, y, L, intercept=intercept)
        skews = self._skew[sl, j]  # skew at the start date of each increment
        s_bar = float(skews.mean())
        if s_bar == 0.0:
            raise ValueError("mean skew is zero: the SSR is undefined")
        psi = np.column_stack([reg.influence, skews - s_bar])
        S = newey_west_covariance(psi, L) / n
        b = reg.slope
        var_r = S[0, 0] / s_bar**2 + b * b * S[1, 1] / s_bar**4 - 2.0 * b * S[0, 1] / s_bar**3
        return SSREstimate(
            float(T),
            window,
            self._dates[e],
            float(b / s_bar),
            float(np.sqrt(max(var_r, 0.0))),
            b,
            reg.slope_se,
            reg.intercept,
            reg.r2,
            s_bar,
            float(np.sqrt(S[1, 1])),
            n,
            L,
        )

    def skew_term_structure(
        self,
        window: int = WINDOW_VOL,
        *,
        end: pd.Timestamp | str | None = None,
        lags: int | None = None,
    ) -> pd.DataFrame:
        """Mean ATMF skew per pillar over the last ``window`` dates (Newey–West standard error
        of the mean of a persistent level series — indicative) and the latest skew (SPEC item
        3).  Columns: ``mean_skew, se, latest, n, lags``; index ``T``."""
        sl, e = self._window(window, end)
        rows = slice(sl.start + 1, e + 1)  # the last `window` dates ending at e
        S = self._skew[rows]
        n = S.shape[0]
        L = newey_west_lags(n) if lags is None else lags
        mean = S.mean(axis=0)
        se = np.array([np.sqrt(newey_west_variance(S[:, j], L) / n) for j in range(S.shape[1])])
        return pd.DataFrame(
            {
                "mean_skew": mean,
                "se": se,
                "latest": self._skew[e],
                "n": n,
                "lags": L,
            },
            index=pd.Index(self._pillars, name="T"),
        )


def snapshot_date(path: str | Path) -> pd.Timestamp:
    """Date of an importer snapshot: ``provenance.quote_date``, else ``YYYY-MM-DD`` in the name."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if isinstance(raw, dict):
        prov = raw.get("provenance")
        if isinstance(prov, dict) and prov.get("quote_date"):
            return pd.Timestamp(str(prov["quote_date"]))
    m = re.search(r"(\d{4}-\d{2}-\d{2})", p.name)
    if m is None:
        raise ValueError(f"{p}: no provenance.quote_date and no date in the file name")
    return pd.Timestamp(m.group(1))


# --------------------------------------------------------------------------------------------
# Summary and rolling versions
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HistoryEstimates:
    """All Part 2 estimates at one end date: vol of vol and correlations on ``window_vol``, the
    SSR on both ``window_ssr`` and ``window_vol``, the skew term structure on ``window_vol``.
    ``to_frame()`` returns a tidy long table ``statistic, T, T2, window, value, se``."""

    end_date: pd.Timestamp
    window_vol: int
    window_ssr: int
    days_per_year: int
    volvol: pd.DataFrame
    correlations: PillarCorrelations
    spot_vol_corr: pd.DataFrame
    ssr_short: pd.DataFrame
    ssr_long: pd.DataFrame
    skew_ts: pd.DataFrame

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for T, r in self.volvol.iterrows():
            rows.append(
                {
                    "statistic": "volvol",
                    "T": float(str(T)),
                    "T2": np.nan,
                    "window": self.window_vol,
                    "value": r["volvol"],
                    "se": r["se"],
                }
            )
        for T, r in self.spot_vol_corr.iterrows():
            rows.append(
                {
                    "statistic": "spot_vol_corr",
                    "T": float(str(T)),
                    "T2": np.nan,
                    "window": self.window_vol,
                    "value": r["corr"],
                    "se": r["se_fisher"],
                }
            )
        pil = list(self.correlations.corr.index)
        for i, Ti in enumerate(pil):
            for Tj in pil[i + 1 :]:
                rows.append(
                    {
                        "statistic": "pillar_corr",
                        "T": float(Ti),
                        "T2": float(Tj),
                        "window": self.window_vol,
                        "value": float(self.correlations.corr.loc[Ti, Tj]),
                        "se": float(self.correlations.se_fisher.loc[Ti, Tj]),
                    }
                )
        for name, df, w in (
            ("ssr", self.ssr_short, self.window_ssr),
            ("ssr", self.ssr_long, self.window_vol),
        ):
            for T, r in df.iterrows():
                rows.append(
                    {
                        "statistic": name,
                        "T": float(str(T)),
                        "T2": np.nan,
                        "window": w,
                        "value": r["ssr"],
                        "se": r["se"],
                    }
                )
        for T, r in self.skew_ts.iterrows():
            rows.append(
                {
                    "statistic": "mean_skew",
                    "T": float(str(T)),
                    "T2": np.nan,
                    "window": self.window_vol,
                    "value": r["mean_skew"],
                    "se": r["se"],
                }
            )
            rows.append(
                {
                    "statistic": "latest_skew",
                    "T": float(str(T)),
                    "T2": np.nan,
                    "window": 1,
                    "value": r["latest"],
                    "se": np.nan,
                }
            )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        parts = [
            f"HistoryEstimates(end={self.end_date.date()}, window_vol={self.window_vol}, "
            f"window_ssr={self.window_ssr}, days_per_year={self.days_per_year})",
            "vol of vol of VS vols:",
            self.volvol.round(4).to_string(),
            "cross-pillar correlation of d ln vs_vol (Fisher se in .correlations.se_fisher):",
            self.correlations.corr.round(3).to_string(),
            "spot/vol correlation:",
            self.spot_vol_corr.round(4).to_string(),
            f"SSR (window {self.window_ssr}):",
            self.ssr_short.round(4).to_string(),
            f"SSR (window {self.window_vol}):",
            self.ssr_long.round(4).to_string(),
            "ATMF skew term structure:",
            self.skew_ts.round(4).to_string(),
        ]
        return "\n".join(parts)


def _ssr_frame(history: SurfaceHistory, window: int, end: pd.Timestamp | None) -> pd.DataFrame:
    rows = [history.ssr_hist(float(T), window, end=end) for T in history.pillars]
    return pd.DataFrame(
        {
            "ssr": [r.ssr for r in rows],
            "se": [r.se for r in rows],
            "slope": [r.slope for r in rows],
            "slope_se": [r.slope_se for r in rows],
            "r2": [r.r2 for r in rows],
            "mean_skew": [r.mean_skew for r in rows],
            "mean_skew_se": [r.mean_skew_se for r in rows],
            "n": [r.n for r in rows],
            "lags": [r.lags for r in rows],
        },
        index=pd.Index(history.pillars, name="T"),
    )


def estimate_history(
    history: SurfaceHistory,
    *,
    end: pd.Timestamp | str | None = None,
    window_vol: int = WINDOW_VOL,
    window_ssr: int = WINDOW_SSR,
    days_per_year: int = DAYS_PER_YEAR,
) -> HistoryEstimates:
    """All estimators at ``end`` (default: the last date) — SPEC §15 Part 2 windows: 250 for the
    vol of vol and correlations, 60 for the SSR (also reported on the long window)."""
    e = history.date_index(end)
    end_ts = history.dates[e]
    vv = [
        history.volvol_hist(float(T), window_vol, end=end_ts, days_per_year=days_per_year)
        for T in history.pillars
    ]
    volvol = pd.DataFrame(
        {
            "volvol": [v.volvol for v in vv],
            "se": [v.se for v in vv],
            "n": [v.n for v in vv],
            "lags": [v.lags for v in vv],
        },
        index=pd.Index(history.pillars, name="T"),
    )
    sv = [history.spot_vol_correlation(float(T), window_vol, end=end_ts) for T in history.pillars]
    spot_vol = pd.DataFrame(
        {
            "corr": [c.corr for c in sv],
            "se_fisher": [c.se_fisher for c in sv],
            "se_hac": [c.se_hac for c in sv],
            "n": [c.n for c in sv],
        },
        index=pd.Index(history.pillars, name="T"),
    )
    return HistoryEstimates(
        end_ts,
        window_vol,
        window_ssr,
        days_per_year,
        volvol,
        history.pillar_correlations(window_vol, end=end_ts),
        spot_vol,
        _ssr_frame(history, window_ssr, end_ts),
        _ssr_frame(history, window_vol, end_ts),
        history.skew_term_structure(window_vol, end=end_ts),
    )


def _rolling_ends(history: SurfaceHistory, window: int, step: int) -> list[int]:
    if step < 1:
        raise ValueError("step must be >= 1")
    if window < MIN_INCREMENTS or window > history.n_dates - 1:
        raise ValueError(
            f"window must lie in [{MIN_INCREMENTS}, {history.n_dates - 1}] for this history"
        )
    return list(range(window, history.n_dates, step))


def rolling_volvol(
    history: SurfaceHistory,
    window: int = WINDOW_VOL,
    *,
    step: int = 1,
    days_per_year: int = DAYS_PER_YEAR,
    pillars: Sequence[float] | None = None,
) -> pd.DataFrame:
    """``volvol_hist`` on every window end (rows ``end_date, T, volvol, se, n, lags``; for Part
    4).  Checked by ``tests/test_history.py::test_rolling_shapes``."""
    ps = history.pillars if pillars is None else np.asarray(pillars, dtype=np.float64)
    rows: list[dict[str, object]] = []
    for e in _rolling_ends(history, window, step):
        end = history.dates[e]
        for T in ps:
            v = history.volvol_hist(float(T), window, end=end, days_per_year=days_per_year)
            rows.append(
                {
                    "end_date": end,
                    "T": float(str(T)),
                    "volvol": v.volvol,
                    "se": v.se,
                    "n": v.n,
                    "lags": v.lags,
                }
            )
    return pd.DataFrame(rows)


def rolling_ssr(
    history: SurfaceHistory,
    window: int = WINDOW_SSR,
    *,
    step: int = 1,
    pillars: Sequence[float] | None = None,
    intercept: bool = True,
) -> pd.DataFrame:
    """``ssr_hist`` on every window end (rows ``end_date, T, ssr, se, slope, slope_se, r2,
    mean_skew, n, lags``)."""
    ps = history.pillars if pillars is None else np.asarray(pillars, dtype=np.float64)
    rows: list[dict[str, object]] = []
    for e in _rolling_ends(history, window, step):
        end = history.dates[e]
        for T in ps:
            r = history.ssr_hist(float(T), window, end=end, intercept=intercept)
            rows.append(
                {
                    "end_date": end,
                    "T": float(str(T)),
                    "ssr": r.ssr,
                    "se": r.se,
                    "slope": r.slope,
                    "slope_se": r.slope_se,
                    "r2": r.r2,
                    "mean_skew": r.mean_skew,
                    "n": r.n,
                    "lags": r.lags,
                }
            )
    return pd.DataFrame(rows)


def rolling_spot_vol_corr(
    history: SurfaceHistory,
    window: int = WINDOW_VOL,
    *,
    step: int = 1,
    pillars: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Realised spot/vol correlation on every window end (rows ``end_date, T, corr, se_fisher,
    se_hac, n``)."""
    ps = history.pillars if pillars is None else np.asarray(pillars, dtype=np.float64)
    rows: list[dict[str, object]] = []
    for e in _rolling_ends(history, window, step):
        end = history.dates[e]
        for T in ps:
            c = history.spot_vol_correlation(float(T), window, end=end)
            rows.append(
                {
                    "end_date": end,
                    "T": float(str(T)),
                    "corr": c.corr,
                    "se_fisher": c.se_fisher,
                    "se_hac": c.se_hac,
                    "n": c.n,
                }
            )
    return pd.DataFrame(rows)


def rolling_pillar_corr(
    history: SurfaceHistory, T1: float, T2: float, window: int = WINDOW_VOL, *, step: int = 1
) -> pd.DataFrame:
    """Correlation of ``Δ ln vs_vol(T1)`` and ``Δ ln vs_vol(T2)`` on every window end (rows
    ``end_date, corr, se_fisher, se_hac, n``)."""
    i, j = history.pillar_index(T1), history.pillar_index(T2)
    X = history.d_ln_vs_vol()
    rows: list[dict[str, object]] = []
    for e in _rolling_ends(history, window, step):
        sl = slice(e - window, e)
        c = correlation_with_se(X[sl, i], X[sl, j], newey_west_lags(window))
        rows.append(
            {
                "end_date": history.dates[e],
                "corr": c.corr,
                "se_fisher": c.se_fisher,
                "se_hac": c.se_hac,
                "n": c.n,
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Model counterparts (2F lognormal model, Bergomi ch. 7 / ch. 9)
# --------------------------------------------------------------------------------------------


def _I(x: FloatArray) -> FloatArray:
    """``(1 − e^{−x})/x`` (eq. 7.25), ``I(0) = 1``."""
    x_ = np.asarray(x, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -np.expm1(-x_) / x_
    return np.asarray(np.where(x_ == 0.0, 1.0, out), dtype=np.float64)


def _factor_correlation(p: BergomiParams) -> FloatArray:
    nf = 1 if p.is_one_factor else 2
    return np.asarray(p.correlation_matrix[1 : 1 + nf, 1 : 1 + nf], dtype=np.float64)


def vs_vol_loadings(
    p: BergomiParams,
    T: float,
    *,
    xi: Callable[[FloatArray], FloatArray] | None = None,
    n_quad: int = 64,
) -> FloatArray:
    """Loadings ``ℓ_i = ν α_θ w_i A_i`` of ``d ln σ̂_T = Σ_i ℓ_i dW^i`` (the integrand of eq. 7.39
    before squaring): ``A_i = ∫₀ᵀ ξ(s) e^{−k_i s} ds / ∫₀ᵀ ξ(s) ds`` (eq. 7.38) with ``ξ(s)`` the
    current forward-variance curve in time-to-maturity ``s`` (``xi``; flat when ``None``:
    ``A_i = I(k_i T)``, eq. 7.24).  ``sqrt(ℓᵀ ρ ℓ)`` is eq. 7.39; checked against
    :func:`volsto.analytics.bergomi.vs_vol_of_vol` by
    ``tests/test_history.py::test_day_curve_loadings_match_vs_vol_of_vol``.
    """
    nf = 1 if p.is_one_factor else 2
    ks = np.array([p.k1, p.k2][:nf])
    w = np.array([1.0 - p.theta, p.theta][:nf])
    if xi is None:
        A = _I(ks * T)
    else:
        x, wq = np.polynomial.legendre.leggauss(n_quad)
        s = 0.5 * T * (x + 1.0)
        ws = 0.5 * T * wq
        v = xi(s)
        den = float(np.sum(ws * v))
        A = np.array([float(np.sum(ws * v * np.exp(-k * s))) / den for k in ks])
    return np.asarray(p.nu * alpha_theta(p) * w * A, dtype=np.float64)


def vs_vol_correlation(
    p: BergomiParams,
    T1: float,
    T2: float,
    *,
    xi: Callable[[FloatArray], FloatArray] | None = None,
) -> float:
    """Instantaneous correlation of ``d ln σ̂_{T1}`` and ``d ln σ̂_{T2}`` in the 2F model — the
    model counterpart of the cross-pillar correlation of ``Δ ln vs_vol``:

    ``corr = ℓ(T1)ᵀ ρ ℓ(T2) / sqrt(ℓ(T1)ᵀ ρ ℓ(T1) · ℓ(T2)ᵀ ρ ℓ(T2))`` with the loadings of
    :func:`vs_vol_loadings` and ``ρ`` the factor correlation (``ρ12``).  Derivation: from
    ``dξ_t^u = ω ξ_t^u α_θ Σ_i w_i e^{−k_i(u−t)} dW^i`` (eq. 7.30–7.33; the covariance structure of
    eq. 7.20), ``d σ̂²_T = (1/T)∫ dξ_t^u du`` hence ``d ln σ̂_T = ν α_θ Σ_i w_i A_i dW^i`` with
    ``A_i`` of eq. 7.38; flat curve: ``A_i = I(k_i T)`` and, for ``ρ12 = 0``,
    ``corr = Σ_i w_i² I(k_i T1) I(k_i T2) / sqrt(Σ_i w_i² I(k_i T1)² · Σ_i w_i² I(k_i T2)²)``.
    Checked by the identifiability tests in ``tests/test_history.py``.
    """
    C = _factor_correlation(p)
    l1 = vs_vol_loadings(p, T1, xi=xi)
    l2 = vs_vol_loadings(p, T2, xi=xi)
    return float(l1 @ C @ l2 / np.sqrt((l1 @ C @ l1) * (l2 @ C @ l2)))


def vs_vol_correlation_matrix(
    p: BergomiParams,
    pillars: Sequence[float] = DEFAULT_PILLARS,
    *,
    xi: Callable[[FloatArray], FloatArray] | None = None,
) -> pd.DataFrame:
    """:func:`vs_vol_correlation` on a pillar grid (DataFrame indexed by pillar)."""
    ps = [float(T) for T in pillars]
    M = np.array([[vs_vol_correlation(p, a, b, xi=xi) for b in ps] for a in ps])
    return pd.DataFrame(M, index=ps, columns=ps)


def spot_vs_vol_correlation_flat(p: BergomiParams, T: float) -> float:
    """Instantaneous ``corr(d ln S, d ln σ̂_T)`` for a flat curve:
    ``Σ_i ℓ_i ρ_{SX_i} / sqrt(ℓᵀ ρ ℓ)`` (from ``E[dW^S dW^i] = ρ_{SX_i} dt``)."""
    nf = 1 if p.is_one_factor else 2
    rho_s = np.array([p.rho_SX1, p.rho_SX2][:nf])
    ell = vs_vol_loadings(p, T)
    C = _factor_correlation(p)
    return float(ell @ rho_s / np.sqrt(ell @ C @ ell))


def day_forward_variance_curve(
    model: BergomiSV, t: float, factors: FloatArray, t_max: float, *, n_nodes: int = 1201
) -> ForwardVarianceCurve:
    """The forward-variance curve seen at time ``t`` in state ``factors`` (``(n_factors,)``),
    re-based to time-to-maturity ``s ∈ (0, t_max]``: ``ξ(s) = ξ_t^{t+s}`` (eqs. 7.33–7.35 via
    :meth:`BergomiSV.forward_variance`), as a :class:`ForwardVarianceCurve` through the
    trapezoidal total variance on ``n_nodes`` nodes (so that the library's general-curve formulas
    — eqs. 7.39, 9.18, 9.19 — can be evaluated on it).  Checked by
    ``tests/test_history.py::test_day_curve_loadings_match_vs_vol_of_vol``."""
    s = np.linspace(0.0, t_max, n_nodes)
    xi = model.forward_variance(t, np.asarray(factors, dtype=np.float64)[None, :], t + s)[0]
    W = np.concatenate(([0.0], np.cumsum(0.5 * (xi[1:] + xi[:-1]) * np.diff(s))))
    return ForwardVarianceCurve(s[1:], W[1:])


# --------------------------------------------------------------------------------------------
# Synthetic 2F history
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MixingBatch:
    """Per-day ATMF vol and skew from :func:`mixing_atmf_batch` with Monte Carlo standard
    errors (common paths across days: the errors are almost perfectly correlated between days,
    so daily *changes* are far more precise than the levels)."""

    T: float
    atm_vol: FloatArray
    atm_vol_se: FloatArray
    skew: FloatArray
    skew_se: FloatArray
    n_paths: int
    n_steps: int


def mixing_atmf_batch(
    model: BergomiSV,
    T: float,
    t_days: ArrayLike,
    factors: ArrayLike,
    *,
    n_paths: int,
    seed: int,
    dt: float,
    chunk_size: int = 5_000,
) -> MixingBatch:
    """ATMF implied vol and skew ``∂σ̂/∂k|_{k=0}`` of the pure SV model for many dates at once,
    each by the mixing solution (eqs. 8.58–8.62) refreshed with that date's forward-variance
    curve, sharing one set of factor paths across dates (common random numbers).

    On date ``t`` in state ``X_t`` the future instantaneous variance factorises (eqs. 7.15,
    7.33–7.35): ``ξ_{t+s}^{t+s} = g(t+s) · exp(Σ_i coef_i e^{−k_i s} X^i_t) · exp(Σ_i coef_i
    Y^i_s)``
    with ``Y`` a fresh OU process from 0 (``g`` as in :meth:`BergomiSV.g`, ``coef_i = ω α_θ
    w_i``); this is exactly the mixing solution of a fresh ``BergomiSV`` whose initial curve is
    the date's ``ξ_t^{t+s}`` (``χ(t, t+s) + χ(s, s) = χ(t+s, t+s)``), so the two mixing integrals
    of eq. 8.61 are matrix products of the path arrays ``exp(coef·Y_s)`` and ``sqrt(exp(coef·Y_s))
    (c·dW_s)`` with the date-specific deterministic weights (left-point rule of step ``h ≤ dt``,
    as :func:`volsto.analytics.mixing.mixing_integrals`).  With ``F = K = 1`` the ATMF vol inverts
    the mean call price and the skew is the exact strike derivative
    ``σ̂'(0) = [N(d₂(σ̂)) − E N(d₂*)]/vega(σ̂)`` (from ``∂_k Black(f*, e^k, σ*) = −e^k N(d₂*)``),
    whose per-path spread is bounded — unlike the OTM call/put difference of
    :func:`~volsto.analytics.mixing.mixing_atmf_skew`, whose put–call-parity term ``f* − K`` adds
    noise (about 20× at 1y).  Standard errors from antithetic pair averages.  Checked against
    the library on the flat state and against a fresh model on the day's curve by
    ``tests/test_history.py::test_mixing_batch_matches_library``.
    """
    if n_paths < 2 or n_paths % 2 or chunk_size < 2 or chunk_size % 2:
        raise ValueError("n_paths and chunk_size must be even and >= 2 (antithetic pairs)")
    if T <= 0 or dt <= 0:
        raise ValueError("T and dt must be positive")
    t_d = np.atleast_1d(np.asarray(t_days, dtype=np.float64))
    X = np.asarray(factors, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    nf = model.n_factors
    if X.shape != (t_d.size, nf):
        raise ValueError(f"factors must have shape {(t_d.size, nf)}, got {X.shape}")
    n_steps = max(1, int(np.ceil(T / dt - 1e-9)))
    h = T / n_steps
    cov = factor_step_covariance(model.params, h, nf, with_brownians=True)[0]
    chol = sqrt_covariance(cov[None])[0]  # rows: dW^S, dX^1..nf, dW^1..nf
    c, lam2 = parallel_coefficients(model)
    if lam2 >= 1.0:
        raise ValueError("mixing solution needs λ² < 1 (|spot/vol correlation| < 1)")
    decay = np.exp(-model.ks * h)
    s = h * np.arange(n_steps)
    # date-specific deterministic weights (n_steps, n_days)
    tilt = np.exp(np.einsum("di,ji,i->dj", X, np.exp(-np.outer(s, model.ks)), model.coef))
    g_ts = model.g(t_d[:, None] + s[None, :])
    W = (g_ts * tilt).T
    U = W * h
    V = np.sqrt(W)
    rng = np.random.default_rng(seed)
    nd = t_d.size
    sum_c0 = np.zeros(nd)
    sum_c0_sq = np.zeros(nd)
    sum_n2 = np.zeros(nd)
    sum_n2_sq = np.zeros(nd)
    n_done = 0
    sqrt_1ml = np.sqrt(1.0 - lam2)
    while n_done < n_paths:
        n = min(chunk_size, n_paths - n_done)
        half = n // 2
        A = np.empty((n, n_steps))
        B = np.empty((n, n_steps))
        y = np.zeros((n, nf))
        for j in range(n_steps):
            z = rng.standard_normal((half, 1 + 2 * nf))
            inc = np.concatenate((z, -z)) @ chol.T
            a = np.exp(y @ model.coef)
            A[:, j] = a
            B[:, j] = np.sqrt(a) * (inc[:, 1 + nf :] @ c)
            y = y * decay + inc[:, 1 : 1 + nf]
        I1 = A @ U
        I2 = B @ V
        ln_f = -0.5 * lam2 * I1 + I2
        sq = sqrt_1ml * np.sqrt(I1)  # σ* √T
        d1 = ln_f / sq + 0.5 * sq
        d2 = d1 - sq
        n2 = ndtr(d2)
        c0 = np.exp(ln_f) * ndtr(d1) - n2  # call, F = K = 1
        c0p = 0.5 * (c0[0::2] + c0[1::2])
        n2p = 0.5 * (n2[0::2] + n2[1::2])
        sum_c0 += c0.sum(axis=0)
        sum_c0_sq += (c0p * c0p).sum(axis=0)
        sum_n2 += n2.sum(axis=0)
        sum_n2_sq += (n2p * n2p).sum(axis=0)
        n_done += n
    n_pairs = n_paths // 2
    mean_c0 = sum_c0 / n_paths
    mean_n2 = sum_n2 / n_paths
    var_c0 = np.maximum(sum_c0_sq / n_pairs - mean_c0 * mean_c0, 0.0) * n_pairs / (n_pairs - 1)
    var_n2 = np.maximum(sum_n2_sq / n_pairs - mean_n2 * mean_n2, 0.0) * n_pairs / (n_pairs - 1)
    se_c0 = np.sqrt(var_c0 / n_pairs)
    se_n2 = np.sqrt(var_n2 / n_pairs)
    iv = implied_vol(mean_c0, 1.0, 1.0, T, 1, 1.0)
    vega = black_vega(1.0, 1.0, T, iv, 1.0)
    d2_bs = -0.5 * iv * np.sqrt(T)
    skew = (ndtr(d2_bs) - mean_n2) / vega
    return MixingBatch(
        float(T),
        np.asarray(iv, dtype=np.float64),
        np.asarray(se_c0 / vega, dtype=np.float64),
        np.asarray(skew, dtype=np.float64),
        np.asarray(se_n2 / vega, dtype=np.float64),
        n_paths,
        n_steps,
    )


def simulate_daily_path(
    model: BergomiSV, years: float, days_per_year: int, seed: int
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """One path of ``(ln S, X)`` at daily steps ``1/days_per_year`` over ``years`` with the
    library kernel (:meth:`BergomiSV.simulate_chunk`: exact OU factor transition, eqs.
    7.15–7.18, and the second-order SV spot step of M4b, ``SimConfig`` defaults).  Returns
    ``(times, ln_spot, factors)`` with ``factors`` of shape ``(n_days + 1, n_factors)``.

    The spot step's daily error is irrelevant for the estimators' statistics: they depend on the
    covariance of the daily increments, which the exact joint Gaussian draw ``(δW^S, δX)``
    reproduces up to the ``O(k_i δt)`` difference between ``ρ_iS (1 − e^{−k_i δt})/k_i`` and
    ``ρ_iS δt`` (about 1% for ``k1 = 5.35`` at ``δt = 1/252``, checked in the M7 Part 2 report).
    """
    if years <= 0 or days_per_year <= 0:
        raise ValueError("years and days_per_year must be positive")
    grid = TimeGrid.build([years], 1.0 / days_per_year, record_all_steps=True)
    cfg = SimConfig(n_paths=1, antithetic=False, chunk_size=1, seed=seed)
    draws = GaussianDraws(seed, 1, grid.n_steps, model.n_brownians, antithetic=False)
    paths = model.simulate_chunk(grid, draws, 0, 1, cfg.scheme)
    return (
        np.asarray(grid.times, dtype=np.float64),
        np.asarray(paths.log_spot[0], dtype=np.float64),
        np.asarray(paths.factors[0], dtype=np.float64),
    )


@dataclass(frozen=True)
class SyntheticHistory:
    """A synthetic 2F history with the model quantities recorded along the path.

    Per date ``d`` and pillar ``j`` (arrays ``(n_dates, n_pillars)``):

    * ``volvol_inst`` — instantaneous vol of the VS vol, eq. 7.39 with the date's own curve;
    * ``beta_inst`` — instantaneous regression coefficient ``E[dσ̂_T d ln S]/E[(d ln S)²]`` at
      order one (VS vol for the ATMF vol, book §9.2): ``ν α_θ (σ̂_T/√ξ_t^t) Σ_i w_i ρ_{SX_i} A_i``;
    * ``skew_order1`` — eq. 9.18 with the date's curve;
    * ``loadings`` ``(n_dates, n_pillars, n_factors)`` — :func:`vs_vol_loadings` per date.

    ``targets(window)`` averages these the way the estimators do (path averages, the slope's
    realised-variance weighting replaced by ``ξ_t^t``), next to the flat-curve formulas.
    """

    history: SurfaceHistory
    params: BergomiParams
    atm_source: str
    states: pd.DataFrame
    volvol_inst: FloatArray
    beta_inst: FloatArray
    skew_order1: FloatArray
    loadings: FloatArray
    atm_vol_mc_se: FloatArray
    skew_mc_se: FloatArray
    days_per_year: int

    @property
    def pillars(self) -> FloatArray:
        return self.history.pillars

    def targets(self, window: int | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
        """Per pillar: ``volvol_in`` (``sqrt(mean volvol_inst²)`` over the window's increment
        start dates), ``volvol_flat`` (eq. 7.24/7.39), ``ssr_order1_in`` (``ξ``-weighted mean of
        ``beta_inst`` over the mean ``skew_order1``), ``ssr_flat`` (eq. 9.21), ``skew_order1_in``,
        ``skew_flat`` (eq. 8.55), ``spot_vol_corr_in``, ``spot_vol_corr_flat``."""
        sl = self._slice(window, end)
        xi = self.states["xi_tt"].to_numpy()[sl]
        p = self.params
        nf = self.loadings.shape[2]
        rho_s = np.array([p.rho_SX1, p.rho_SX2][:nf])
        C = _factor_correlation(p)
        rows = []
        for j, T in enumerate(self.pillars):
            ell = self.loadings[sl, j, :]
            var_inst = np.einsum("di,ik,dk->d", ell, C, ell)
            cov_spot = np.sqrt(xi) * (ell @ rho_s)
            rows.append(
                {
                    "volvol_in": float(np.sqrt(np.mean(self.volvol_inst[sl, j] ** 2))),
                    "volvol_flat": float(vs_vol_of_vol_flat(p, T)),
                    "ssr_order1_in": float(
                        np.sum(self.beta_inst[sl, j] * xi)
                        / np.sum(xi)
                        / np.mean(self.skew_order1[sl, j])
                    ),
                    "ssr_flat": float(ssr_order1_flat(p, T)),
                    "skew_order1_in": float(np.mean(self.skew_order1[sl, j])),
                    "skew_flat": float(atmf_skew_order1_flat(p, T)),
                    "spot_vol_corr_in": float(
                        np.mean(cov_spot) / np.sqrt(np.mean(xi) * np.mean(var_inst))
                    ),
                    "spot_vol_corr_flat": spot_vs_vol_correlation_flat(p, float(T)),
                }
            )
        return pd.DataFrame(rows, index=pd.Index(self.pillars, name="T"))

    def correlation_targets(
        self, window: int | None = None, end: pd.Timestamp | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """``(in-sample, flat)`` model correlation matrices of ``d ln σ̂`` across pillars:
        in-sample ``mean_t[ℓ_iᵀρℓ_j] / sqrt(mean_t[ℓ_iᵀρℓ_i] mean_t[ℓ_jᵀρℓ_j])``."""
        sl = self._slice(window, end)
        C = _factor_correlation(self.params)
        L = self.loadings[sl]
        cov = np.einsum("dia,ab,djb->ij", L, C, L) / L.shape[0]
        d = np.sqrt(np.diag(cov))
        idx = list(self.pillars)
        return (
            pd.DataFrame(cov / np.outer(d, d), index=idx, columns=idx),
            vs_vol_correlation_matrix(self.params, idx),
        )

    def _slice(self, window: int | None, end: pd.Timestamp | None) -> slice:
        e = self.history.date_index(end)
        if window is None:
            return slice(0, e)
        if window > e:
            raise ValueError(f"window {window} exceeds the {e} increments up to the end date")
        return slice(e - window, e)


def synthetic_2f_history(
    params: BergomiParams,
    *,
    years: float,
    seed: int,
    start_date: str | pd.Timestamp = "2020-01-02",
    spot: float = 100.0,
    vs_vol: float = 0.20,
    pillars: Sequence[float] = DEFAULT_PILLARS,
    days_per_year: int = DAYS_PER_YEAR,
    atm_source: Literal["mixing", "order_one"] = "mixing",
    mixing_paths: int = 20_000,
    mixing_dt: float = 1.0 / 365.0,
    forward_curve: ForwardCurve | None = None,
    n_quad: int = 64,
) -> SyntheticHistory:
    """Synthetic history of the pure 2F (or 1F) model for the identifiability test (SPEC §15
    Part 2): one daily path of ``(ln S, X)`` over ``years`` (:func:`simulate_daily_path`) on a
    flat VS curve at ``vs_vol`` and ``forward_curve`` (default: flat ``r = q = 0`` at ``spot``);
    each date's pillar VS vols ``σ̂_T(t)² = (1/T)∫_t^{t+T} ξ_t^u du`` from the factor state
    (:meth:`BergomiSV.vs_variance`), and the ATMF vol and skew either

    * ``atm_source="mixing"`` — by the mixing solution refreshed with the date's curve
      (:func:`mixing_atmf_batch`, ``mixing_paths`` common paths, step ``mixing_dt``); or
    * ``atm_source="order_one"`` — at order one in the vol of vol: the VS vol stands in for the
      ATMF vol (book §9.2) and the skew is eq. 9.18 on the date's curve (no Monte Carlo).

    Dates are business days from ``start_date`` (one per simulation step of ``1/days_per_year``
    years; holidays ignored).  The history is wrapped through :meth:`SurfaceHistory.
    from_callable` with :class:`PillarQuotes` per date.  Cost (Table 8.2, 3y, 6 pillars, 20k
    mixing paths): about 5 s for the mixing and 10 s for the per-date order-one skews.
    Checked by ``tests/test_history.py`` (identifiability tests).
    """
    if atm_source not in ("mixing", "order_one"):
        raise ValueError("atm_source must be 'mixing' or 'order_one'")
    if vs_vol <= 0 or spot <= 0:
        raise ValueError("vs_vol and spot must be positive")
    ps = np.asarray(pillars, dtype=np.float64)
    if ps.ndim != 1 or ps.size == 0 or np.any(ps <= 0) or np.any(np.diff(ps) <= 0):
        raise ValueError("pillars must be positive and strictly increasing")
    t_max = float(ps[-1])
    fc = forward_curve if forward_curve is not None else ForwardCurve.flat(spot, 0.0, 0.0)
    xi0 = ForwardVarianceCurve.flat(vs_vol * vs_vol, max(10.0, years + t_max + 1.0))
    model = BergomiSV(params, xi0, fc)
    times, ln_s, X = simulate_daily_path(model, years, days_per_year, seed)
    nd, nf = X.shape
    n_p = ps.size
    dates = pd.bdate_range(pd.Timestamp(start_date), periods=nd)
    # per-date model quantities from the factor state
    vs = np.empty((nd, n_p))
    loadings = np.empty((nd, n_p, nf))
    skew1 = np.empty((nd, n_p))
    xi_tt = np.empty(nd)
    rho_s = np.array([params.rho_SX1, params.rho_SX2][:nf])
    C = _factor_correlation(params)
    curve_nodes = int(np.ceil(t_max * 365)) + 1
    for d in range(nd):
        t = float(times[d])
        fac = X[d]
        xi_tt[d] = float(model.variance_from_factors(t, fac[None, :])[0])

        def xi_day(s: FloatArray, _t: float = t, _f: FloatArray = fac) -> FloatArray:
            return np.asarray(model.forward_variance(_t, _f[None, :], _t + s)[0])

        curve = day_forward_variance_curve(model, t, fac, t_max + 0.05, n_nodes=curve_nodes)
        for j, T in enumerate(ps):
            vs[d, j] = float(model.vs_variance(t, fac[None, :], t, t + float(T), n_quad)[0])
            loadings[d, j] = vs_vol_loadings(params, float(T), xi=xi_day, n_quad=n_quad)
            skew1[d, j] = atmf_skew_order1(params, curve, float(T))
    vs_sig = np.sqrt(vs)
    volvol_inst = np.sqrt(np.einsum("dji,ik,djk->dj", loadings, C, loadings))
    beta_inst = np.sqrt(vs / xi_tt[:, None]) * (loadings @ rho_s)
    if atm_source == "mixing":
        atm = np.empty((nd, n_p))
        skew = np.empty((nd, n_p))
        atm_se = np.empty(n_p)
        skew_se = np.empty(n_p)
        for j, T in enumerate(ps):
            mb = mixing_atmf_batch(
                model, float(T), times, X, n_paths=mixing_paths, seed=seed + 1000 + j, dt=mixing_dt
            )
            atm[:, j] = mb.atm_vol
            skew[:, j] = mb.skew
            atm_se[j] = float(mb.atm_vol_se.mean())
            skew_se[j] = float(mb.skew_se.mean())
    else:
        atm = vs_sig.copy()
        skew = skew1.copy()
        atm_se = np.zeros(n_p)
        skew_se = np.zeros(n_p)
    spots = np.exp(ln_s)
    quotes = {
        dates[d]: PillarQuotes(ps, vs_sig[d], atm[d], skew[d], float(spots[d])) for d in range(nd)
    }
    history = SurfaceHistory.from_callable(dates, lambda d: quotes[d], ps.tolist())
    states = pd.DataFrame(
        {
            "t": times,
            "ln_spot": ln_s,
            **{f"X{i + 1}": X[:, i] for i in range(nf)},
            "xi_tt": xi_tt,
        },
        index=dates,
    )
    states.index.name = "date"
    return SyntheticHistory(
        history,
        params,
        atm_source,
        states,
        volvol_inst,
        beta_inst,
        skew1,
        loadings,
        atm_se,
        skew_se,
        days_per_year,
    )


# --------------------------------------------------------------------------------------------
# Real data: HDN sample day by day
# --------------------------------------------------------------------------------------------


def hdn_available_dates(root: str | Path) -> list[str]:
    """Trading dates (``YYYY-MM-DD``) the HistoricalData.net source at ``root`` holds
    (:meth:`volsto.market.vendor.HdnSource.available_dates`: the source owns the layout)."""
    return HdnSource(root).available_dates()


def build_history(
    source: VendorSource,
    dates: Sequence[str],
    out_dir: str | Path,
    *,
    underlying: str = "SPX",
    pillars: Sequence[float] = DEFAULT_PILLARS,
    essvi: bool = True,
    skip_failures: bool = True,
    calendar_repair: CalendarRepairConfig | None = DEFAULT_CALENDAR_REPAIR,
) -> tuple[SurfaceHistory, dict[str, str]]:
    """Import each date through a vendor source (:meth:`volsto.market.vendor.VendorSource.
    import_day`), write the snapshot YAML into ``out_dir`` and build the history from the
    snapshots.  Returns the history and the failures ``{date: error}``; with
    ``skip_failures=False`` the first importer error propagates.  About 1 s per day plus 0.1 s
    per day for the pillar strip (measured on the 2022 H2 sample) — a test should run a handful
    of days only.  ``calendar_repair`` is passed to the importer (eSSVI calendar repair, M10
    Part 0; ``None`` = pre-M10 behaviour)."""
    from volsto.market.import_hdn import write_snapshot

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    failures: dict[str, str] = {}
    for date in dates:
        try:
            cfg, _, _, _ = source.import_day(
                date, underlying, essvi=essvi, calendar_repair=calendar_repair
            )
            written.append(write_snapshot(cfg, out / f"{underlying.lower()}_{date}.yaml"))
        except Exception as exc:
            if not skip_failures:
                raise
            failures[date] = f"{type(exc).__name__}: {exc}"
            log.warning("import of %s failed: %s", date, exc)
    if not written:
        raise ValueError(f"no day could be imported: {failures}")
    return SurfaceHistory.from_snapshots(written, pillars), failures


def build_hdn_history(
    root: str | Path,
    dates: Sequence[str],
    out_dir: str | Path,
    *,
    underlying: str = "SPX",
    pillars: Sequence[float] = DEFAULT_PILLARS,
    essvi: bool = True,
    skip_failures: bool = True,
    calendar_repair: CalendarRepairConfig | None = DEFAULT_CALENDAR_REPAIR,
) -> tuple[SurfaceHistory, dict[str, str]]:
    """:func:`build_history` on the HistoricalData.net source at ``root``."""
    return build_history(
        HdnSource(root),
        dates,
        out_dir,
        underlying=underlying,
        pillars=pillars,
        essvi=essvi,
        skip_failures=skip_failures,
        calendar_repair=calendar_repair,
    )


__all__ = [
    "DAYS_PER_YEAR",
    "DEFAULT_PILLARS",
    "MIN_INCREMENTS",
    "WINDOW_SSR",
    "WINDOW_VOL",
    "CorrelationEstimate",
    "HacRegression",
    "HistoryEstimates",
    "MixingBatch",
    "PillarCorrelations",
    "PillarQuotes",
    "PillarSource",
    "SSREstimate",
    "SurfaceHistory",
    "SurfacePillarSource",
    "SyntheticHistory",
    "VolVolEstimate",
    "build_hdn_history",
    "build_history",
    "correlation_with_se",
    "day_forward_variance_curve",
    "estimate_history",
    "hac_ols",
    "hdn_available_dates",
    "mixing_atmf_batch",
    "newey_west_covariance",
    "newey_west_lags",
    "newey_west_variance",
    "rolling_pillar_corr",
    "rolling_spot_vol_corr",
    "rolling_ssr",
    "rolling_volvol",
    "simulate_daily_path",
    "snapshot_date",
    "spot_vs_vol_correlation_flat",
    "synthetic_2f_history",
    "vs_vol_correlation",
    "vs_vol_correlation_matrix",
    "vs_vol_loadings",
]
