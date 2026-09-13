"""Discount and forward curves (SPEC §2.1).

``DiscountCurve(times, rates)`` stores continuously compounded zero rates at pillar times and
interpolates linearly in ``log DF``; this is equivalent to piecewise-flat instantaneous forward
rates between pillars (and flat extrapolation of the last forward rate).  All times are year
fractions; :class:`Calendar` converts dates with ACT/365 fixed.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

if TYPE_CHECKING:
    from volsto.config import CurveConfig, MarketConfig

FloatArray = NDArray[np.float64]


def _as_float_array(x: ArrayLike) -> FloatArray:
    return np.asarray(x, dtype=np.float64)


class DiscountCurve:
    """Zero curve with log-linear discount-factor interpolation (piecewise-flat forwards).

    Args:
        times: strictly increasing positive pillar times (years).
        rates: continuously compounded zero rates at the pillars, ``DF(t_i) = exp(-r_i t_i)``.
    """

    def __init__(self, times: Sequence[float] | FloatArray, rates: Sequence[float] | FloatArray):
        t = _as_float_array(times).ravel()
        r = _as_float_array(rates).ravel()
        if t.size == 0 or t.shape != r.shape:
            raise ValueError("times and rates must be non-empty with equal length")
        if np.any(t <= 0) or np.any(np.diff(t) <= 0):
            raise ValueError("times must be positive and strictly increasing")
        if not np.all(np.isfinite(r)):
            raise ValueError("rates must be finite")
        self._times = np.concatenate(([0.0], t))
        self._log_df = np.concatenate(([0.0], -r * t))
        # piecewise-flat instantaneous forward rates on [t_{i-1}, t_i)
        self._fwd = -np.diff(self._log_df) / np.diff(self._times)

    @classmethod
    def flat(cls, rate: float) -> DiscountCurve:
        """Flat continuously compounded rate."""
        return cls([1.0], [rate])

    @classmethod
    def from_config(cls, cfg: CurveConfig) -> DiscountCurve:
        return cls(cfg.times, cfg.rates)

    @classmethod
    def from_instantaneous(
        cls, times: Sequence[float] | FloatArray, fwd_rates: Sequence[float] | FloatArray
    ) -> DiscountCurve:
        """Build from piecewise-flat instantaneous forward rates on ``[t_{i-1}, t_i)``."""
        t = _as_float_array(times).ravel()
        f = _as_float_array(fwd_rates).ravel()
        if t.size == 0 or t.shape != f.shape:
            raise ValueError("times and fwd_rates must be non-empty with equal length")
        t0 = np.concatenate(([0.0], t))
        integ = np.cumsum(f * np.diff(t0))
        return cls(t, integ / t)

    @property
    def times(self) -> FloatArray:
        return self._times[1:].copy()

    @property
    def zero_rates(self) -> FloatArray:
        return (-self._log_df[1:] / self._times[1:]).copy()

    def log_df(self, t: ArrayLike) -> FloatArray:
        """``ln DF(t)``; linear between pillars, flat last forward beyond the last pillar."""
        tt = _as_float_array(t)
        if np.any(tt < 0):
            raise ValueError("time must be non-negative")
        inside = np.interp(tt, self._times, self._log_df)
        t_last = self._times[-1]
        beyond = self._log_df[-1] - self._fwd[-1] * (tt - t_last)
        return np.where(tt > t_last, beyond, inside)

    def df(self, t: ArrayLike) -> FloatArray:
        """Discount factor ``DF(t) = exp(-∫₀ᵗ r)``."""
        return np.exp(self.log_df(t))

    def integrated(self, t1: ArrayLike, t2: ArrayLike) -> FloatArray:
        """``∫_{t1}^{t2} r(u) du = ln DF(t1) - ln DF(t2)``."""
        return self.log_df(t1) - self.log_df(t2)

    def zero_rate(self, t: ArrayLike) -> FloatArray:
        """Continuously compounded zero rate ``-ln DF(t)/t`` (instantaneous rate at t=0)."""
        tt = _as_float_array(t)
        with np.errstate(divide="ignore", invalid="ignore"):
            z = -self.log_df(tt) / tt
        return np.where(tt > 0, z, self._fwd[0])

    def forward_rate(self, t1: ArrayLike, t2: ArrayLike) -> FloatArray:
        """Continuously compounded forward rate between ``t1 < t2``."""
        a = _as_float_array(t1)
        b = _as_float_array(t2)
        if np.any(b <= a):
            raise ValueError("need t2 > t1")
        return self.integrated(a, b) / (b - a)

    def instantaneous_forward(self, t: ArrayLike) -> FloatArray:
        """Piecewise-flat instantaneous forward rate ``r(t)`` (right-continuous)."""
        tt = _as_float_array(t)
        idx = np.searchsorted(self._times, tt, side="right") - 1
        idx = np.clip(idx, 0, self._fwd.size - 1)
        return self._fwd[idx]

    def __repr__(self) -> str:
        return f"DiscountCurve(times={self.times.tolist()}, rates={self.zero_rates.tolist()})"


class ForwardCurve:
    """Equity forward ``F(T) = S0 · exp(∫₀ᵀ (r − q))`` with deterministic rate and repo curves.

    Source: SPEC §2.1.  Checked by ``tests/test_curves.py``.
    """

    def __init__(self, spot: float, rate_curve: DiscountCurve, dividend_curve: DiscountCurve):
        if not np.isfinite(spot) or spot <= 0:
            raise ValueError("spot must be positive")
        self.spot = float(spot)
        self.rate_curve = rate_curve
        self.dividend_curve = dividend_curve

    @classmethod
    def from_config(cls, cfg: MarketConfig) -> ForwardCurve:
        return cls(
            cfg.spot,
            DiscountCurve.from_config(cfg.rate_curve),
            DiscountCurve.from_config(cfg.dividend_curve),
        )

    @classmethod
    def flat(cls, spot: float, r: float, q: float) -> ForwardCurve:
        return cls(spot, DiscountCurve.flat(r), DiscountCurve.flat(q))

    def drift(self, t1: ArrayLike, t2: ArrayLike) -> FloatArray:
        """``∫_{t1}^{t2} (r − q) du``: the log-forward increment used by the log-Euler step."""
        return self.rate_curve.integrated(t1, t2) - self.dividend_curve.integrated(t1, t2)

    def log_forward(self, t: ArrayLike) -> FloatArray:
        """``ln F(t)``."""
        return np.asarray(np.log(self.spot) + self.drift(0.0, t), dtype=np.float64)

    def forward(self, t: ArrayLike) -> FloatArray:
        """``F(t) = S0 exp(∫₀ᵗ (r − q))``."""
        return np.asarray(np.exp(self.log_forward(t)), dtype=np.float64)

    def df(self, t: ArrayLike) -> FloatArray:
        """Discount factor of the rate curve (convenience)."""
        return self.rate_curve.df(t)

    def with_spot(self, spot: float) -> ForwardCurve:
        """Same curves, new spot (used by spot bumps)."""
        return ForwardCurve(spot, self.rate_curve, self.dividend_curve)

    def __repr__(self) -> str:
        return (
            f"ForwardCurve(spot={self.spot}, rate_curve={self.rate_curve!r}, "
            f"dividend_curve={self.dividend_curve!r})"
        )


class Calendar:
    """ACT/365 fixed year-fraction helper anchored at a valuation date (SPEC §2.1)."""

    DAYS_PER_YEAR = 365.0

    def __init__(self, anchor: _dt.date):
        self.anchor = anchor

    @staticmethod
    def year_fraction(d0: _dt.date, d1: _dt.date) -> float:
        return (d1 - d0).days / Calendar.DAYS_PER_YEAR

    def t(self, d: _dt.date) -> float:
        return self.year_fraction(self.anchor, d)

    def times(self, dates: Iterable[_dt.date]) -> FloatArray:
        return np.array([self.t(d) for d in dates], dtype=np.float64)

    def date(self, t: float) -> _dt.date:
        return self.anchor + _dt.timedelta(days=round(t * self.DAYS_PER_YEAR))
