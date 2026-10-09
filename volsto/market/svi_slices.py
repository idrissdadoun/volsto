"""SVI slices per listed expiry as an implied surface (SPEC §8.7, M12 part LC1).

Promoted from the dispersion study's check C8 (``scripts/disp_c8.py``, SPEC §8.5), numerics
unchanged: the library's Dupire local vol needs a smooth implied surface, and a listed smile is
a handful of strikes per expiry.

**Slice** (raw SVI, Gatheral 2004), ``params = (a, b, ρ, m, σ)``:

    w(k)  = a + b·(ρ(k − m) + √((k − m)² + σ²))                (:func:`svi_total_variance`)
    w′(k) = b·(ρ + (k − m)/√((k − m)² + σ²))                    (:func:`svi_derivatives`)
    w″(k) = b·σ² / ((k − m)² + σ²)^{3/2}

with ``k = ln(K/F(T))`` and ``w = σ̂²T`` (the library's conventions, ``market/surface.py``).

**Fit** (:func:`fit_svi_slice`; C8's ``fit_svi`` on arrays): least squares on the total variance
of the strikes within ``width_sd`` at-the-money standard deviations of the forward (at least
``min_width`` in ``k``; every strike when fewer than ``min_points`` qualify), from the start
point and inside the bounds stated there.

**Surface** (:class:`SviSlices`): total variance linear in ``T`` at fixed ``k`` between the
slices, proportional to ``T`` before the first and after the last, floored at
:data:`W_FLOOR`.  :meth:`SviSlices.arbitrage_report` checks each slice for butterfly arbitrage
with the analytic derivatives and consecutive slices for calendar arbitrage.

**Records** (:func:`recorded_svi_fit`): a fitted slice is a stored record keyed by its inputs
(SPEC §13.4): the solver's last bits are machine-dependent (SPEC §13.3), and the parameters
enter the local-correlation cache key, so every machine reads them back and none refits.

Checked by ``tests/test_svi_slices.py`` (the fit against the frozen C8 code bit for bit, the
derivatives, the interpolation, the arbitrage report, the records).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

from volsto.market.surface import ArbitrageError, ImpliedSurface

if TYPE_CHECKING:
    from volsto.calibration.fit_records import FitRecords
    from volsto.market.curves import ForwardCurve

FloatArray = NDArray[np.float64]

SVI_FIT_CODE_TAG: Final[str] = "svi1"
"""The SVI slice fit's numerics tag, part of every SVI record key.  Bump it whenever a change
moves any fitted slice (start point, bounds, strike selection, solver settings)."""
W_FLOOR: Final[float] = 1e-8
"""Floor on a slice's total variance (C8's value): a raw SVI slice may cross zero in a wing."""
FIT_WIDTH_SD: Final[float] = 3.0
"""Fitted strikes: within this many at-the-money standard deviations of the forward (C8)."""
FIT_MIN_WIDTH: Final[float] = 0.15
"""… and at least this far in log-moneyness (C8)."""
FIT_MIN_POINTS: Final[int] = 5
"""Fewer qualifying strikes than this: every strike of the expiry is fitted (C8)."""
MIN_SLICE_T: Final[float] = 10 / 365.0
"""Expiries shorter than ten days are not fitted (C8)."""
N_BEYOND: Final[int] = 2
"""Listed expiries kept beyond the horizon (C8: the Dupire time derivative at the horizon needs
the bracketing slice, and one more keeps the interpolation away from the extrapolation)."""
MAX_MATURITY_MARGIN: Final[float] = 0.05
"""``max_maturity = max(last slice, horizon) + 0.05`` (C8): ``LocalVolSurface.from_implied``
refuses a grid that ends beyond the surface's ``max_maturity``."""
ARBITRAGE_TOL: Final[float] = 1e-12
"""A butterfly or calendar quantity below ``-ARBITRAGE_TOL`` is a violation (the derivatives
are analytic: the tolerance covers rounding only)."""


# --------------------------------------------------------------------------------------------
# the slice
# --------------------------------------------------------------------------------------------


def svi_total_variance(params: ArrayLike, k: ArrayLike) -> FloatArray:
    """Raw SVI total variance ``w(k) = a + b·(ρ(k − m) + √((k − m)² + σ²))`` with ``params =
    (a, b, ρ, m, σ)`` (Gatheral 2004; C8's ``svi``, the same statement).  Checked by
    ``tests/test_svi_slices.py::test_svi_matches_c8``."""
    a, b, rho, m, s = np.asarray(params, dtype=np.float64)
    kk = np.asarray(k, dtype=np.float64)
    return np.asarray(a + b * (rho * (kk - m) + np.sqrt((kk - m) ** 2 + s * s)), dtype=np.float64)


def svi_derivatives(params: ArrayLike, k: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
    """``(w, w′, w″)`` of the raw SVI slice (module docstring; derived by differentiating
    :func:`svi_total_variance`).  Checked against central differences by
    ``tests/test_svi_slices.py::test_svi_derivatives``."""
    a, b, rho, m, s = np.asarray(params, dtype=np.float64)
    x = np.asarray(k, dtype=np.float64) - m
    r = np.sqrt(x * x + s * s)
    w = a + b * (rho * x + r)
    w1 = b * (rho + x / r)
    w2 = b * s * s / (r * r * r)
    return (
        np.asarray(w, dtype=np.float64),
        np.asarray(w1, dtype=np.float64),
        np.asarray(w2, dtype=np.float64),
    )


@dataclass(frozen=True)
class SviSliceFit:
    """One fitted slice: the maturity, the raw SVI parameters ``(a, b, ρ, m, σ)``, the
    root-mean-square error in vol points on the fitted strikes, their number and their range in
    log-moneyness."""

    T: float
    params: tuple[float, float, float, float, float]
    rms_vp: float
    n_points: int
    k_lo: float
    k_hi: float

    def summary(self) -> dict[str, Any]:
        """The JSON summary a fit record stores (floats survive the round trip bit for bit)."""
        return {
            "T": self.T,
            "params": list(self.params),
            "rms_vp": self.rms_vp,
            "n_points": self.n_points,
            "k_lo": self.k_lo,
            "k_hi": self.k_hi,
        }

    @classmethod
    def from_summary(cls, doc: Mapping[str, Any]) -> SviSliceFit:
        p = [float(x) for x in doc["params"]]
        if len(p) != 5:
            raise ValueError("an SVI slice has five parameters (a, b, rho, m, sigma)")
        return cls(
            float(doc["T"]),
            (p[0], p[1], p[2], p[3], p[4]),
            float(doc["rms_vp"]),
            int(doc["n_points"]),
            float(doc["k_lo"]),
            float(doc["k_hi"]),
        )


def _smile_arrays(k: ArrayLike, vol: ArrayLike, T: float) -> tuple[FloatArray, FloatArray]:
    kk = np.asarray(k, dtype=np.float64)
    vv = np.asarray(vol, dtype=np.float64)
    if kk.ndim != 1 or kk.shape != vv.shape or kk.size == 0:
        raise ValueError("k and vol must be one-dimensional, non-empty and of equal length")
    if not (np.isfinite(T) and T > 0):
        raise ValueError("the maturity must be positive")
    if not np.all(np.isfinite(kk)) or np.any(np.diff(kk) <= 0):
        raise ValueError("k must be finite and strictly increasing")
    if not np.all(np.isfinite(vv)) or np.any(vv <= 0):
        raise ValueError("vols must be positive and finite")
    return kk, vv


def fit_svi_slice(
    k: ArrayLike,
    vol: ArrayLike,
    T: float,
    *,
    width_sd: float = FIT_WIDTH_SD,
    min_width: float = FIT_MIN_WIDTH,
    min_points: int = FIT_MIN_POINTS,
) -> SviSliceFit:
    """Raw-SVI parameters of one expiry's total variance (C8's ``fit_svi``, on arrays).

    * Fitted strikes: ``|k| ≤ max(width_sd·σ_atm·√T, min_width)`` with ``σ_atm`` the vol
      linearly interpolated at ``k = 0``; every strike when fewer than ``min_points`` qualify.
    * Residuals: ``w_SVI(k) − vol²·T`` on those strikes.
    * Start point, with ``w₀ = σ_atm²·T``: ``(½w₀, max(w₀, 1e-4)/0.2, −0.4, 0, 0.2)``.
    * Bounds: ``(−w₀, 1e-6, −0.999, −1, 1e-3)`` to ``(4w₀ + 1e-6, 10, 0.999, 1, 3)``.
    * Solver: ``scipy.optimize.least_squares`` with its default settings.

    ``rms_vp`` is the root-mean-square vol error on the fitted strikes, in vol points.  The
    result is deterministic on one machine and reproduced to the solver's last bits only on
    another (SPEC §13.3): pipelines read it through :func:`recorded_svi_fit`.  Checked by
    ``tests/test_svi_slices.py::test_svi_matches_c8`` (bit for bit against the frozen C8 code).
    """
    kk, vv = _smile_arrays(k, vol, T)
    atm = float(np.interp(0.0, kk, vv))
    width = max(width_sd * atm * np.sqrt(T), min_width)
    sel = np.abs(kk) <= width
    if sel.sum() < min_points:
        sel = np.ones(kk.size, dtype=bool)
    ks, w = kk[sel], vv[sel] ** 2 * T
    w0 = atm * atm * T
    x0 = np.array([0.5 * w0, max(w0, 1e-4) / 0.2, -0.4, 0.0, 0.2])
    lo = np.array([-w0, 1e-6, -0.999, -1.0, 1e-3])
    hi = np.array([4.0 * w0 + 1e-6, 10.0, 0.999, 1.0, 3.0])

    def residuals(p: FloatArray) -> FloatArray:
        return np.asarray(svi_total_variance(p, ks) - w, dtype=np.float64)

    fit = least_squares(residuals, x0, bounds=(lo, hi))
    vol_fit = np.sqrt(np.maximum(svi_total_variance(fit.x, ks), 1e-10) / T)
    rms = float(100.0 * np.sqrt(np.mean((vol_fit - vv[sel]) ** 2)))
    a, b, rho, m, s = (float(x) for x in fit.x)
    return SviSliceFit(float(T), (a, b, rho, m, s), rms, int(ks.size), float(ks[0]), float(ks[-1]))


# --------------------------------------------------------------------------------------------
# the surface
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ArbitrageReport:
    """Output of :meth:`SviSlices.arbitrage_report` on the grid of ``n_k`` points over
    ``|k| ≤ k_range`` — or, when the report was asked on per-slice ranges, over ``[k_lo[s],
    k_hi[s]]`` for the butterfly of slice ``s`` and over the union of the two slices' ranges for
    the calendar of a pair (``k_lo``, ``k_hi``; ``None`` for the symmetric range).

    * ``min_g[s]``: the smallest value on the grid of slice ``s``'s butterfly function
      ``g(k) = (1 − k w′/(2w))² − (w′²/4)(1/w + 1/4) + w″/2`` (the density of the slice has the
      sign of ``g``; Gatheral 2006, the formula of ``PerturbedSurface.check_no_arbitrage``,
      here with the analytic derivatives), reached at ``argmin_g[s]``.
    * ``min_calendar[s]``: the smallest ``w_{s+1}(k) − w_s(k)`` on the grid, reached at
      ``argmin_calendar[s]`` (one entry per consecutive pair).  Under the surface's
      linear-in-``T`` interpolation ``w_{s+1} ≥ w_s`` is necessary and sufficient for
      ``∂_T w ≥ 0`` between the two slices (derived: ``∂_T w = (w_{s+1} − w_s)/(t_{s+1} − t_s)``
      there).
    * ``n_floored[s]``: grid points where the raw slice is at or below :data:`W_FLOOR` (the
      surface is flat there; ``g`` is evaluated on the floored slice).

    A quantity below ``−tol`` is a violation."""

    times: tuple[float, ...]
    min_g: tuple[float, ...]
    argmin_g: tuple[float, ...]
    min_calendar: tuple[float, ...]
    argmin_calendar: tuple[float, ...]
    n_floored: tuple[int, ...]
    k_range: float
    n_k: int
    tol: float
    k_lo: tuple[float, ...] | None = None
    k_hi: tuple[float, ...] | None = None

    @property
    def butterfly_ok(self) -> bool:
        return all(g >= -self.tol for g in self.min_g)

    @property
    def calendar_ok(self) -> bool:
        return all(c >= -self.tol for c in self.min_calendar)

    @property
    def ok(self) -> bool:
        return self.butterfly_ok and self.calendar_ok

    def violations(self) -> list[str]:
        """One line per violated slice or pair (empty when :attr:`ok`)."""
        out = [
            f"butterfly: slice T={t:.6g}, min g = {g:.3e} at k = {k:.4g}"
            for t, g, k in zip(self.times, self.min_g, self.argmin_g, strict=True)
            if g < -self.tol
        ]
        out += [
            f"calendar: slices T={t0:.6g} -> T={t1:.6g}, min dw = {c:.3e} at k = {k:.4g}"
            for t0, t1, c, k in zip(
                self.times[:-1],
                self.times[1:],
                self.min_calendar,
                self.argmin_calendar,
                strict=True,
            )
            if c < -self.tol
        ]
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "times": list(self.times),
            "min_g": list(self.min_g),
            "argmin_g": list(self.argmin_g),
            "min_calendar": list(self.min_calendar),
            "argmin_calendar": list(self.argmin_calendar),
            "n_floored": list(self.n_floored),
            "k_range": self.k_range,
            "n_k": self.n_k,
            "tol": self.tol,
            "violations": self.violations(),
            **({} if self.k_lo is None else {"k_lo": list(self.k_lo)}),
            **({} if self.k_hi is None else {"k_hi": list(self.k_hi)}),
        }


class SviSlices(ImpliedSurface):
    """Total variance from SVI slices: linear in time between them at fixed log-moneyness,
    proportional to time before the first and after the last (flat implied vol at fixed ``k``
    there), floored at :data:`W_FLOOR` (C8's class; module docstring).

    ``times`` are the slices' maturities (strictly increasing), ``params`` their raw SVI
    parameters, shape ``(n_slices, 5)``.  The discount curve is the forward curve's rate curve.
    """

    def __init__(
        self,
        times: ArrayLike,
        params: ArrayLike,
        forward_curve: ForwardCurve,
        max_maturity: float,
    ) -> None:
        super().__init__(forward_curve, forward_curve.rate_curve, max_maturity)
        t = np.asarray(times, dtype=np.float64)
        p = np.asarray(params, dtype=np.float64)
        if t.ndim != 1 or t.size == 0 or p.shape != (t.size, 5):
            raise ValueError("times must be (n_slices,) and params (n_slices, 5)")
        if not np.all(np.isfinite(t)) or t[0] <= 0 or np.any(np.diff(t) <= 0):
            raise ValueError("slice times must be positive and strictly increasing")
        if not np.all(np.isfinite(p)):
            raise ValueError("SVI parameters must be finite")
        self.times, self.params = t, p

    @property
    def n_slices(self) -> int:
        return int(self.times.size)

    def total_variance(self, k: ArrayLike, T: ArrayLike) -> FloatArray:
        k = np.asarray(k, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        k_b, T_b = np.broadcast_arrays(k, T)
        slices = np.stack(
            [np.maximum(svi_total_variance(p, k_b), W_FLOOR) for p in self.params]
        )  # (n_slices, …)
        j = (
            np.clip(np.searchsorted(self.times, T_b), 1, len(self.times) - 1)
            if len(self.times) > 1
            else np.zeros_like(T_b, dtype=int)
        )
        if len(self.times) == 1:
            return np.asarray(slices[0] * T_b / self.times[0], dtype=np.float64)
        t0, t1 = self.times[j - 1], self.times[j]
        w0 = np.take_along_axis(slices, (j - 1)[None, ...], axis=0)[0]
        w1 = np.take_along_axis(slices, j[None, ...], axis=0)[0]
        inside = w0 + (w1 - w0) * (T_b - t0) / (t1 - t0)
        before = slices[0] * T_b / self.times[0]
        after = slices[-1] * T_b / self.times[-1]
        return np.asarray(
            np.where(
                T_b <= self.times[0],
                before,
                np.where(T_b >= self.times[-1], after, np.maximum(inside, W_FLOOR)),
            ),
            dtype=np.float64,
        )

    def arbitrage_report(
        self,
        k_range: float = 1.0,
        n_k: int = 401,
        tol: float = ARBITRAGE_TOL,
        *,
        k_lo: Sequence[float] | None = None,
        k_hi: Sequence[float] | None = None,
    ) -> ArbitrageReport:
        """Butterfly per slice and calendar per consecutive pair on ``n_k`` points over
        ``|k| ≤ k_range`` (:class:`ArbitrageReport`) — or, with ``k_lo`` and ``k_hi`` (one bound
        per slice, e.g. the range a particle cloud visits at that maturity), over ``[k_lo[s],
        k_hi[s]]`` for slice ``s`` and over the union of the two ranges for a pair.  Checked by
        ``tests/test_svi_slices.py::test_arbitrage_report_detects_violations``."""
        if k_range <= 0 or n_k < 3:
            raise ValueError("need k_range > 0 and n_k >= 3")
        if (k_lo is None) != (k_hi is None):
            raise ValueError("give both k_lo and k_hi, or neither")
        if k_lo is None or k_hi is None:
            lo = np.full(self.n_slices, -float(k_range))
            hi = np.full(self.n_slices, float(k_range))
        else:
            lo, hi = np.asarray(k_lo, dtype=float), np.asarray(k_hi, dtype=float)
            if lo.shape != (self.n_slices,) or hi.shape != (self.n_slices,) or np.any(hi < lo):
                raise ValueError("k_lo and k_hi: one bound per slice, with k_lo <= k_hi")

        def floored_slice(p: ArrayLike, ks: FloatArray) -> tuple[FloatArray, FloatArray, int]:
            """The floored slice on ``ks``, its butterfly function and its floored points."""
            w, w1, w2 = svi_derivatives(p, ks)
            low = w <= W_FLOOR
            # on the floor the surface is flat in k: w = W_FLOOR, w' = w'' = 0, hence g = 1
            ws = np.where(low, W_FLOOR, w)
            w1 = np.where(low, 0.0, w1)
            w2 = np.where(low, 0.0, w2)
            g = (1.0 - ks * w1 / (2.0 * ws)) ** 2 - 0.25 * w1 * w1 * (1.0 / ws + 0.25) + 0.5 * w2
            return ws, g, int(low.sum())

        min_g, arg_g, n_floored = [], [], []
        for s, p in enumerate(self.params):
            ks = np.linspace(lo[s], hi[s], n_k)
            _, g, n_low = floored_slice(p, ks)
            i = int(np.argmin(g))
            min_g.append(float(g[i]))
            arg_g.append(float(ks[i]))
            n_floored.append(n_low)
        min_c, arg_c = [], []
        for s in range(self.n_slices - 1):
            ks = np.linspace(min(lo[s], lo[s + 1]), max(hi[s], hi[s + 1]), n_k)
            d = floored_slice(self.params[s + 1], ks)[0] - floored_slice(self.params[s], ks)[0]
            i = int(np.argmin(d))
            min_c.append(float(d[i]))
            arg_c.append(float(ks[i]))
        return ArbitrageReport(
            tuple(float(t) for t in self.times),
            tuple(min_g),
            tuple(arg_g),
            tuple(min_c),
            tuple(arg_c),
            tuple(n_floored),
            float(k_range),
            int(n_k),
            float(tol),
            None if k_lo is None else tuple(float(x) for x in lo),
            None if k_hi is None else tuple(float(x) for x in hi),
        )

    def check_no_arbitrage(
        self, k_range: float = 1.0, n_k: int = 401, tol: float = ARBITRAGE_TOL
    ) -> None:
        """Raise :class:`~volsto.market.surface.ArbitrageError` naming every violation of
        :meth:`arbitrage_report`."""
        report = self.arbitrage_report(k_range, n_k, tol)
        if not report.ok:
            raise ArbitrageError("SVI slices: " + "; ".join(report.violations()))

    def __repr__(self) -> str:
        return (
            f"SviSlices(n_slices={self.n_slices}, T∈[{self.times[0]:.4g}, {self.times[-1]:.4g}], "
            f"max_maturity={self.max_maturity:.4g})"
        )


# --------------------------------------------------------------------------------------------
# records (SPEC §13.4 pattern)
# --------------------------------------------------------------------------------------------


def svi_fit_settings(
    width_sd: float = FIT_WIDTH_SD,
    min_width: float = FIT_MIN_WIDTH,
    min_points: int = FIT_MIN_POINTS,
) -> dict[str, Any]:
    """The fit settings as the record key reads them."""
    return {
        "width_sd": float(width_sd),
        "min_width": float(min_width),
        "min_points": int(min_points),
    }


def svi_fit_inputs(
    T: float, k: ArrayLike, vol: ArrayLike, settings: Mapping[str, Any]
) -> dict[str, Any]:
    """What a slice fit reads, as a JSON-able mapping: the maturity, the strikes'
    log-moneyness, their vols and the fit settings."""
    return {
        "T": float(T),
        "k": [float(x) for x in np.asarray(k, dtype=np.float64).ravel()],
        "vol": [float(x) for x in np.asarray(vol, dtype=np.float64).ravel()],
        "settings": dict(settings),
    }


def svi_fit_key(T: float, k: ArrayLike, vol: ArrayLike, settings: Mapping[str, Any]) -> str:
    """The record key of one slice fit: the SHA-256 of the canonical encoding
    (:func:`volsto.calibration.fit_records.canonical`: sorted keys, every float as the shortest
    round-trip ``repr`` of its double) of the inputs and :data:`SVI_FIT_CODE_TAG`.  A one-ulp
    change of the maturity, of a strike or of a vol is another key (tested)."""
    from volsto.calibration.fit_records import canonical

    payload = {"inputs": svi_fit_inputs(T, k, vol, settings), "fit_code_tag": SVI_FIT_CODE_TAG}
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


def recorded_svi_fit(
    k: ArrayLike,
    vol: ArrayLike,
    T: float,
    *,
    records: FitRecords | None,
    origin: str,
    width_sd: float = FIT_WIDTH_SD,
    min_width: float = FIT_MIN_WIDTH,
    min_points: int = FIT_MIN_POINTS,
) -> SviSliceFit:
    """The slice fit of these inputs: read from ``records`` when its record exists, else fitted
    (:func:`fit_svi_slice`) and recorded under :data:`SVI_FIT_CODE_TAG`.  ``records=None``
    always fits and stores nothing.  A writer that lost a race returns the winner's parameters
    (records are immutable, SPEC §13.4)."""
    if records is None:
        return fit_svi_slice(
            k, vol, T, width_sd=width_sd, min_width=min_width, min_points=min_points
        )
    settings = svi_fit_settings(width_sd, min_width, min_points)
    inputs = svi_fit_inputs(T, k, vol, settings)
    key = svi_fit_key(T, k, vol, settings)
    doc = records.get(key)
    if doc is not None:
        return SviSliceFit.from_summary(doc["fit"])
    fit = fit_svi_slice(k, vol, T, width_sd=width_sd, min_width=min_width, min_points=min_points)
    stored = records.put(key, inputs, fit.summary(), origin=origin, code_tag=SVI_FIT_CODE_TAG)
    return SviSliceFit.from_summary(stored["fit"])


class SmileSlice(Protocol):
    """What :func:`fit_svi_surface` reads of a listed expiry (the study's ``ExpirySmile`` is
    one): the maturity and the vols at the log-moneyness of its strikes."""

    @property
    def T(self) -> float: ...

    @property
    def k(self) -> FloatArray: ...

    @property
    def vol(self) -> FloatArray: ...


def fit_svi_surface(
    expiries: Sequence[SmileSlice],
    forward_curve: ForwardCurve,
    *,
    horizon: float,
    min_T: float = MIN_SLICE_T,
    n_beyond: int = N_BEYOND,
    records: FitRecords | None = None,
    origin: str,
) -> tuple[SviSlices, list[SviSliceFit]]:
    """The SVI surface of a list of listed expiries (sorted by maturity) and its slice fits.

    C8's slice selection: the expiries of at least ``min_T`` (ten days), those up to the
    ``horizon`` and the next ``n_beyond`` (two).  Each slice goes through ``records``
    (:func:`recorded_svi_fit`; ``None`` fits in place and stores nothing).  ``max_maturity =
    max(last slice, horizon) + 0.05`` as in C8 (:data:`MAX_MATURITY_MARGIN`).  ``k`` is the
    log-moneyness against each expiry's own forward; ``forward_curve`` is the curve the surface
    is used on (SPEC §8.7: the index target sits on the basket's forward curve)."""
    if not (np.isfinite(horizon) and horizon > 0):
        raise ValueError("the horizon must be positive")
    maturities = [float(e.T) for e in expiries]
    use = [i for i, t in enumerate(maturities) if t >= min_T]
    kept = [i for i in use if maturities[i] <= horizon]
    kept += [i for i in use if maturities[i] > horizon][:n_beyond]
    if not kept:
        raise ValueError(f"no listed expiry of at least {min_T:.4g} years to fit")
    fits = [
        recorded_svi_fit(
            expiries[i].k, expiries[i].vol, maturities[i], records=records, origin=origin
        )
        for i in kept
    ]
    times = np.array([maturities[i] for i in kept])
    surface = SviSlices(
        times,
        np.stack([np.array(f.params) for f in fits]),
        forward_curve,
        max_maturity=max(float(times[-1]), horizon) + MAX_MATURITY_MARGIN,
    )
    return surface, fits


__all__ = [
    "ARBITRAGE_TOL",
    "SVI_FIT_CODE_TAG",
    "W_FLOOR",
    "ArbitrageReport",
    "SmileSlice",
    "SviSliceFit",
    "SviSlices",
    "fit_svi_slice",
    "fit_svi_surface",
    "recorded_svi_fit",
    "svi_derivatives",
    "svi_fit_inputs",
    "svi_fit_key",
    "svi_fit_settings",
    "svi_total_variance",
]
