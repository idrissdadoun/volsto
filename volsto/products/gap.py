"""Barrier gap conventions (owner addendum at the M6 review): the fixed signed shift of M6 and the
**smart (signed) gap**.

``GapSpec(mode="fixed")`` is the M6 ``barrier_shift`` (a signed fraction of the monitored level,
the default, unchanged).  ``mode="smart"``: at each monitoring date and each barrier the
effective shift is computed per path from the seller's mark-to-market jump across the level in
that path's state,

    ΔV(date, state) = L_knocked(date, state) − L_continuing(date, state),

the seller's liabilities just inside (barrier event happened) and just outside (structure
continues) the level, both conditional expectations at the date given the state ``(ln S, X,
knock-in status, memory state)`` obtained by the §7.11 conditional-pricing regression of the
remaining discounted cash flows on a polynomial basis in the state (the liability of the
knocked side is analytic when the event settles the structure, e.g. the autocall redemption).
The **sign** of the shift is whichever is conservative for the seller in that state: when
``ΔV > 0`` the knock costs more, so the level is moved to enlarge the knocked region (an
up-barrier moves down, a down-barrier moves up); when ``ΔV < 0`` the opposite.  The sign is
therefore never fixed per barrier: it follows the measured ``ΔV`` of the state, which can take
either sign at the same barrier (time to observation, distance of the spot, coupon / memory
state, knock-in status).  The **size** is a configurable function of ``|ΔV|``
(:data:`GAP_FUNCTIONS`); the placeholder default is linear in ``|ΔV|`` with a floor for
overnight / liquidity slippage and a cap — **to be replaced by the desk's sizing function**
(open question recorded in SPEC §6.9: linear in |ΔV|, a number of vegas, a bid/ask in the
digital; and which barriers it applies to).  Every quantity is reported per barrier and per
date (:class:`GapReport`).  Checked by ``tests/test_gap.py``.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

GAP_MODES: tuple[str, ...] = ("fixed", "smart")
DIRECTIONS: tuple[str, ...] = ("up", "down")


def gap_linear(abs_dv: FloatArray, params: dict[str, float]) -> FloatArray:
    """Placeholder sizing: ``min(cap, floor + scale · |ΔV|)`` in fractions of the barrier, with
    ``|ΔV|`` in fractions of notional; ``floor_bp`` / ``cap_bp`` in basis points of the level."""
    floor = float(params["floor_bp"]) * 1e-4
    cap = float(params["cap_bp"]) * 1e-4
    scale = float(params["scale"])
    return np.asarray(np.minimum(cap, floor + scale * np.abs(abs_dv)), dtype=np.float64)


#: sizing functions ``|ΔV| -> shift size``; the desk's function registers here
GAP_FUNCTIONS: dict[str, Callable[[FloatArray, dict[str, float]], FloatArray]] = {
    "linear": gap_linear,
}


@dataclass(repr=False, frozen=True)
class GapSpec:
    """Gap convention of a barrier product.

    ``mode``: ``"fixed"`` (the M6 signed ``fixed_shift``) or ``"smart"``.  Smart parameters:
    ``function`` (key of :data:`GAP_FUNCTIONS`) and ``params`` (its parameters: for
    ``"linear"`` ``scale`` in barrier fraction per unit ``|ΔV|``, ``floor_bp``, ``cap_bp``);
    ``apply_to``: which barriers of an autocall get the smart gap (``"autocall"``, ``"ki"``; a
    barrier not listed keeps ``fixed_shift``; barrier options have one barrier and ignore it);
    ``degree``: polynomial degree of the conditional-pricing regression in the state;
    ``band``: the log-moneyness half-width around the level of the paths used by the regression
    (the continuation value is needed at the level, so the fit is local); ``ki_grid_months``:
    months between the regression dates of a daily-monitored barrier (the monitoring dates use
    the nearest earlier grid date's state-dependent shift, piecewise constant in time);
    ``min_paths``: below this many paths on a side at a date the regression is skipped and the
    fixed shift applies (reported with ``n_fit = 0``); ``report_band``: log-moneyness half-width
    of the "near the barrier" population summarised in :class:`GapReport`; ``dv_min`` and ``z_min``:
    no shift is applied where ``|ΔV| < dv_min`` or ``|ΔV| < z_min ×`` the regression standard
    error of ``ΔV`` at the level (the sign is not identified there; a zero shift is recorded,
    the share of shifted paths is reported).
    """

    mode: str = "fixed"
    fixed_shift: float = 0.0
    function: str = "linear"
    params: dict[str, float] = field(
        default_factory=lambda: {"scale": 0.2, "floor_bp": 10.0, "cap_bp": 500.0}
    )
    apply_to: tuple[str, ...] = ("autocall", "ki")
    degree: int = 2
    band: float = 0.3
    ki_grid_months: int = 1
    min_paths: int = 500
    report_band: float = 0.05
    dv_min: float = 1e-3
    z_min: float = 2.0

    def __post_init__(self) -> None:
        if self.mode not in GAP_MODES:
            raise ValueError(f"mode must be one of {GAP_MODES}, got {self.mode!r}")
        if not np.isfinite(self.fixed_shift) or self.fixed_shift <= -1.0:
            raise ValueError("fixed_shift is a signed fraction of the level and must exceed -1")
        # the smart parameters are validated in both modes (no silently broken spec)
        if self.function not in GAP_FUNCTIONS:
            raise ValueError(f"function must be one of {tuple(GAP_FUNCTIONS)}")
        for key in ("scale", "floor_bp", "cap_bp"):
            if key not in self.params or not np.isfinite(self.params[key]):
                raise ValueError(f"smart gap needs a finite params[{key!r}]")
        if self.params["scale"] < 0 or self.params["floor_bp"] < 0:
            raise ValueError("scale and floor_bp must be non-negative")
        if self.params["cap_bp"] < self.params["floor_bp"]:
            raise ValueError("cap_bp must be at least floor_bp")
        if not self.apply_to or any(a not in ("autocall", "ki") for a in self.apply_to):
            raise ValueError("apply_to entries must be 'autocall' and/or 'ki'")
        if self.degree < 1 or self.band <= 0 or self.ki_grid_months < 1 or self.min_paths < 10:
            raise ValueError("degree >= 1, band > 0, ki_grid_months >= 1, min_paths >= 10")
        if not 0 < self.report_band <= self.band:
            raise ValueError("report_band must lie in (0, band]")
        if not np.isfinite(self.dv_min) or self.dv_min < 0:
            raise ValueError("dv_min must be a non-negative number (fraction of the reference)")
        if not np.isfinite(self.z_min) or self.z_min < 0:
            raise ValueError("z_min must be a non-negative number of standard errors")

    @property
    def smart(self) -> bool:
        return self.mode == "smart"

    def size(self, abs_dv: FloatArray) -> FloatArray:
        """Shift size (fraction of the level) for ``|ΔV|`` (fraction of notional)."""
        return GAP_FUNCTIONS[self.function](
            np.abs(np.asarray(abs_dv, dtype=np.float64)), self.params
        )

    def shift(
        self, dv_seller: FloatArray, direction: str, se: FloatArray | None = None
    ) -> FloatArray:
        """Signed shift per path (fraction of the level) for the seller's ``ΔV``: the sign of
        :func:`conservative_shift`, the size of :meth:`size`; no shift where ``|ΔV| < dv_min``
        or, with the regression standard errors ``se``, where ``|ΔV| < z_min · se`` (the sign is
        not identified there — neither the floor nor a direction applies)."""
        dv = np.asarray(dv_seller, dtype=np.float64)
        weak = np.abs(dv) < self.dv_min
        if se is not None:
            weak |= np.abs(dv) < self.z_min * np.asarray(se, dtype=np.float64)
        dv = np.where(weak, 0.0, dv)
        return conservative_shift(dv, direction, self.size(np.abs(dv)))

    def __repr__(self) -> str:
        if self.mode == "fixed":
            return f"gap fixed {self.fixed_shift:+.4%}"
        p = ", ".join(f"{k}={v:g}" for k, v in self.params.items())
        return (
            f"gap smart ({self.function}: {p}; on {'/'.join(self.apply_to)}, degree {self.degree}, "
            f"band {self.band:g}, grid {self.ki_grid_months}m, z_min {self.z_min:g}) - sizing "
            "placeholder"
        )


def conservative_shift(dv_seller: FloatArray, direction: str, size: FloatArray) -> FloatArray:
    """Signed shift (fraction of the level): enlarge the knocked region when the knock costs the
    seller more (``ΔV > 0``): an up-barrier moves down, a down-barrier moves up; the opposite
    when ``ΔV < 0``; no shift where ``ΔV = 0``."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}")
    sgn = np.sign(np.asarray(dv_seller, dtype=np.float64))
    out = -sgn * np.asarray(size, dtype=np.float64) if direction == "up" else sgn * size
    return np.asarray(out, dtype=np.float64)


def _basis(features: FloatArray, degree: int) -> FloatArray:
    n, d = features.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(d), deg):
            cols.append(np.prod(features[:, list(combo)], axis=1))
    return np.column_stack(cols)


def conditional_value_at_level(
    y: FloatArray,
    features: FloatArray,
    eval_features: FloatArray,
    *,
    degree: int,
    band: float | None = None,
    min_paths: int = 500,
) -> tuple[FloatArray, int, FloatArray]:
    """§7.11 conditional-pricing regression: fit ``y`` (remaining discounted cash flows in
    date money) on a polynomial basis in ``features`` (first column ``ln S − ln B``, then the
    factors and state indicators) and evaluate at ``eval_features`` (first column 0: at the
    level).  With ``band`` only the paths with ``|ln S − ln B| ≤ band`` enter the fit (local).
    Returns the fitted values, the number of paths used (0 when fewer than ``min_paths`` or
    when ``ln S − ln B`` has no spread in the cloud — e.g. every path at the initial spot — the
    caller then falls back to the fixed shift) and the standard error of each fitted value
    (``sqrt(s² xᵉ (XᵀX)⁻¹ xᵉ)``, ``s²`` the residual variance).  Ridge 1e-10 on standardised
    features."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(features, dtype=np.float64)
    if band is not None:
        keep = np.abs(x[:, 0]) <= band
        y, x = y[keep], x[keep]
    n = y.size
    n_eval = eval_features.shape[0]
    if n < min_paths or x[:, 0].std() < 1e-4:
        return np.full(n_eval, np.nan), 0, np.full(n_eval, np.nan)
    mean = x.mean(axis=0)
    scale = np.where(x.std(axis=0) > 0, x.std(axis=0), 1.0)
    xb = _basis((x - mean) / scale, degree)
    xe = _basis((np.asarray(eval_features, dtype=np.float64) - mean) / scale, degree)
    a = xb.T @ xb + 1e-10 * np.eye(xb.shape[1])
    beta = np.linalg.solve(a, xb.T @ y)
    resid = y - xb @ beta
    dof = max(n - xb.shape[1], 1)
    s2 = float(resid @ resid) / dof
    # standard error of the fitted value at each evaluation point
    cov_xe = np.linalg.solve(a, xe.T)  # (k, n_eval)
    se = np.sqrt(np.maximum(s2 * np.sum(xe.T * cov_xe, axis=0), 0.0))
    return np.asarray(xe @ beta, dtype=np.float64), int(n), np.asarray(se, dtype=np.float64)


@dataclass(frozen=True)
class LevelFactors:
    """Effective level factors ``1 + shift`` of a monitored level: a fixed factor, or per-path
    factors on regression grid dates (``factors[p, j]`` applies from ``grid[j]`` until the next
    grid date: piecewise constant in time, the first grid date's factor before it)."""

    fixed: float = 1.0
    grid: FloatArray | None = None
    factors: FloatArray | None = None  # (n_paths, n_grid)

    def at(self, times: ArrayLike, n_paths: int) -> FloatArray:
        """``(n_paths, len(times))`` factors at the monitoring ``times``."""
        t = np.atleast_1d(np.asarray(times, dtype=np.float64))
        if self.grid is None or self.factors is None:
            return np.full((n_paths, t.size), self.fixed)
        j = np.clip(np.searchsorted(self.grid, t + 1e-9, side="right") - 1, 0, self.grid.size - 1)
        return np.asarray(self.factors[:, j], dtype=np.float64)


def regress_at_level(
    gap: GapSpec,
    y: FloatArray,
    features: FloatArray,
    fit: NDArray[np.bool_],
    evaluate: NDArray[np.bool_],
) -> tuple[FloatArray, int, FloatArray]:
    """:func:`conditional_value_at_level` with the spec's regression settings: fit on the paths
    ``fit``, evaluate at the level (first feature 0) for the paths ``evaluate``; returns the
    values, the number of paths fitted and the standard errors of the values."""
    ev = np.asarray(features[evaluate], dtype=np.float64).copy()
    ev[:, 0] = 0.0
    return conditional_value_at_level(
        y[fit], features[fit], ev, degree=gap.degree, band=gap.band, min_paths=gap.min_paths
    )


@dataclass
class GapReport:
    """Per (barrier, date) summary of the smart gap over the paths that reached the date:
    mean / std of ``ΔV`` (fraction of notional), share with ``ΔV > 0``, mean signed shift and
    mean |shift| (fractions of the level), the number of paths regressed."""

    rows: list[dict[str, Any]] = field(default_factory=list)

    def add(
        self,
        barrier: str,
        date: float,
        dv: FloatArray,
        shift: FloatArray,
        n_fit: int,
        near: NDArray[np.bool_] | None = None,
    ) -> None:
        """One row: statistics over the paths with a finite ``ΔV`` (alive at the date) and, with
        ``near`` (paths within ``report_band`` of the level), over the paths at the barrier —
        the states in which the level is actually at stake (``*_near`` columns)."""
        dv = np.asarray(dv, dtype=np.float64)
        shift = np.asarray(shift, dtype=np.float64)
        ok = np.isfinite(dv)
        nr = ok if near is None else (ok & np.asarray(near, dtype=bool))
        row: dict[str, Any] = {
            "barrier": barrier,
            "date": float(date),
            "n_paths": int(ok.sum()),
            "n_fit": int(n_fit),
            "dv_mean": float(dv[ok].mean()) if ok.any() else np.nan,
            "dv_std": float(dv[ok].std(ddof=1)) if ok.sum() > 1 else np.nan,
            "share_dv_positive": float((dv[ok] > 0).mean()) if ok.any() else np.nan,
            "shift_mean": float(shift[ok].mean()) if ok.any() else np.nan,
            "abs_shift_mean": float(np.abs(shift[ok]).mean()) if ok.any() else np.nan,
            "shift_min": float(shift[ok].min()) if ok.any() else np.nan,
            "shift_max": float(shift[ok].max()) if ok.any() else np.nan,
            "share_shifted": float((shift[ok] != 0.0).mean()) if ok.any() else np.nan,
            "n_near": int(nr.sum()),
            "dv_near_mean": float(dv[nr].mean()) if nr.any() else np.nan,
            "share_dv_positive_near": float((dv[nr] > 0).mean()) if nr.any() else np.nan,
            "shift_near_mean": float(shift[nr].mean()) if nr.any() else np.nan,
            "abs_shift_near_mean": float(np.abs(shift[nr]).mean()) if nr.any() else np.nan,
            "share_shifted_near": float((shift[nr] != 0.0).mean()) if nr.any() else np.nan,
        }
        row["sign_near"] = (
            int(np.sign(row["dv_near_mean"])) if np.isfinite(row["dv_near_mean"]) else 0
        )
        self.rows.append(row)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def month_grid(times: ArrayLike, months: int) -> FloatArray:
    """Regression dates for the knock-in barrier: every ``months`` months from 0 to the last
    monitoring time, snapped to the nearest monitoring time (unique, sorted)."""
    t = np.unique(np.atleast_1d(np.asarray(times, dtype=np.float64)).ravel())
    if t.size == 0:
        return t
    targets = np.arange(0.0, t[-1] + 1e-12, months / 12.0)
    snapped = np.array([t[np.argmin(np.abs(t - g))] for g in targets])
    return np.asarray(np.unique(snapped), dtype=np.float64)


__all__ = [
    "DIRECTIONS",
    "GAP_FUNCTIONS",
    "GAP_MODES",
    "GapReport",
    "GapSpec",
    "LevelFactors",
    "conditional_value_at_level",
    "conservative_shift",
    "gap_linear",
    "month_grid",
    "regress_at_level",
]
