"""Frozen copy of check C8's SVI code (``scripts/disp_c8.py`` at commit 390d71c, before M12
part LC1 moved it into ``volsto/market/svi_slices.py``): ``svi``, ``fit_svi`` and ``SviSlices``,
statement for statement.  ``tests/test_svi_slices.py::test_svi_matches_c8`` compares the
library with it bit for bit.  Not library code: never import it outside the tests, never edit
it (a change of the SVI numerics bumps ``SVI_FIT_CODE_TAG`` and re-records, it does not touch
this file)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from volsto.market.curves import ForwardCurve
from volsto.market.surface import ImpliedSurface


@dataclass(frozen=True)
class Smile:
    """The three fields of the study's ``ExpirySmile`` that C8's fit reads."""

    T: float
    k: np.ndarray
    vol: np.ndarray


def svi(p: np.ndarray, k: np.ndarray) -> np.ndarray:
    a, b, rho, m, s = p
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s * s))


def fit_svi(e: Any) -> tuple[np.ndarray, float]:
    """Raw-SVI parameters of one expiry's total variance, fitted on the strikes within three
    standard deviations of the money (at least ±0.15), and the root-mean-square error in vol
    points."""
    atm = float(np.interp(0.0, e.k, e.vol))
    width = max(3.0 * atm * np.sqrt(e.T), 0.15)
    sel = np.abs(e.k) <= width
    if sel.sum() < 5:
        sel = np.ones(e.k.size, dtype=bool)
    k, w = e.k[sel], e.vol[sel] ** 2 * e.T
    w0 = atm * atm * e.T
    x0 = np.array([0.5 * w0, max(w0, 1e-4) / 0.2, -0.4, 0.0, 0.2])
    lo = np.array([-w0, 1e-6, -0.999, -1.0, 1e-3])
    hi = np.array([4.0 * w0 + 1e-6, 10.0, 0.999, 1.0, 3.0])
    fit = least_squares(lambda p: svi(p, k) - w, x0, bounds=(lo, hi))
    vol_fit = np.sqrt(np.maximum(svi(fit.x, k), 1e-10) / e.T)
    return fit.x, float(100.0 * np.sqrt(np.mean((vol_fit - e.vol[sel]) ** 2)))


class SviSlices(ImpliedSurface):
    """Total variance from SVI slices, linear in time between them at fixed log-moneyness
    (proportional to time before the first slice and after the last)."""

    def __init__(
        self,
        times: np.ndarray,
        params: np.ndarray,
        forward_curve: ForwardCurve,
        max_maturity: float,
    ) -> None:
        super().__init__(forward_curve, forward_curve.rate_curve, max_maturity)
        self.times, self.params = times, params

    def total_variance(self, k: Any, T: Any) -> Any:
        k = np.asarray(k, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        k_b, T_b = np.broadcast_arrays(k, T)
        slices = np.stack([np.maximum(svi(p, k_b), 1e-8) for p in self.params])  # (n_slices, …)
        j = (
            np.clip(np.searchsorted(self.times, T_b), 1, len(self.times) - 1)
            if len(self.times) > 1
            else np.zeros_like(T_b, dtype=int)
        )
        if len(self.times) == 1:
            return slices[0] * T_b / self.times[0]
        t0, t1 = self.times[j - 1], self.times[j]
        w0 = np.take_along_axis(slices, (j - 1)[None, ...], axis=0)[0]
        w1 = np.take_along_axis(slices, j[None, ...], axis=0)[0]
        inside = w0 + (w1 - w0) * (T_b - t0) / (t1 - t0)
        before = slices[0] * T_b / self.times[0]
        after = slices[-1] * T_b / self.times[-1]
        return np.where(
            T_b <= self.times[0],
            before,
            np.where(T_b >= self.times[-1], after, np.maximum(inside, 1e-8)),
        )
