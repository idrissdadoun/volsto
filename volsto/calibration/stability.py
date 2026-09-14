"""Stability of the two-factor break-even fit over time (SPEC §15 Part 4, M7 addendum).

:func:`rolling_fit` runs :func:`~volsto.calibration.fit_2f.fit_2f_historical` every ``every``
dates on the trailing ``window_vol`` / ``window_ssr`` windows and returns one row per fitted
date with the break-even parameters ``(k1, λ1, λ2, ω1, ω2, χ)``, their standard errors (``k1``
from the curvature of the ``k1`` profile — NaN with the fit's note when it has none —, ``λ``
from the inner linear least squares, ``ω`` / ``χ`` from the second fit's Jacobian), the book
parameters ``(ν, θ, ρ_SX1, ρ_SX2, ρ12)`` for reading, both objectives, the active skew
constraints and the bound flags.  :func:`flag_unidentified` marks the parameters whose
consecutive changes exceed their standard-error band on more than ``share`` of the fitted dates
(a parameter that moves by more than its own uncertainty from one fit to the next is driven by
noise, not by information — unidentified in the SPEC's sense).  Checked by
``tests/test_fit_2f.py::test_rolling_fit_and_flags``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

import numpy as np
import pandas as pd

from volsto.calibration.fit_2f import BreakEvenFitConfig, fit_2f_historical
from volsto.calibration.history import WINDOW_SSR, WINDOW_VOL, SurfaceHistory

#: the break-even parameters carried by the rolling frame (with ``<name>_se`` columns)
PARAM_COLUMNS = ("k1", "lambda1", "lambda2", "omega1", "omega2", "chi")
#: the book parameters carried for reading (no standard errors)
BOOK_COLUMNS = ("nu", "theta", "rho_SX1", "rho_SX2", "rho12", "k2")


def rolling_fit(
    history: SurfaceHistory,
    cfg: BreakEvenFitConfig | None = None,
    *,
    every: int = 1,
    window_vol: int = WINDOW_VOL,
    window_ssr: int = WINDOW_SSR,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Parameter time series in historical mode: columns ``date``, :data:`PARAM_COLUMNS`,
    their ``_se``, :data:`BOOK_COLUMNS`, ``first_objective, second_objective, n_active,
    k1_at_bound, bound_flags, wall_seconds``.  ``start`` defaults to the first date with
    ``window_vol`` increments behind it; ``every`` is the step in dates."""
    c = cfg or BreakEvenFitConfig()
    if every < 1:
        raise ValueError("every must be positive")
    first = window_vol if start is None else history.date_index(start)
    last = history.date_index(end)
    if first < window_vol:
        raise ValueError(
            f"start must leave {window_vol} increments behind it (first possible date "
            f"{history.dates[window_vol].date()})"
        )
    rows: list[dict[str, object]] = []
    for i in range(first, last + 1, every):
        date = history.dates[i]
        t0 = time.perf_counter()
        r = fit_2f_historical(history, c, end=date, window_vol=window_vol, window_ssr=window_ssr)
        b, p, f, s = r.breakeven, r.params, r.first, r.second
        rows.append(
            {
                "date": date,
                "k1": b.k1,
                "lambda1": b.lambda1,
                "lambda2": b.lambda2,
                "omega1": b.omega1,
                "omega2": b.omega2,
                "chi": b.chi,
                "k1_se": f.k1_se,
                "lambda1_se": f.lambda1_se,
                "lambda2_se": f.lambda2_se,
                "omega1_se": s.stderr["omega1"],
                "omega2_se": s.stderr["omega2"],
                "chi_se": s.stderr["chi"],
                "nu": p.nu,
                "theta": p.theta,
                "rho_SX1": p.rho_SX1,
                "rho_SX2": p.rho_SX2,
                "rho12": p.rho12,
                "k2": p.k2,
                "first_objective": f.objective,
                "second_objective": s.objective,
                "n_active": len(f.active),
                "k1_at_bound": f.k1_at_bound,
                "bound_flags": ";".join(s.bound_flags),
                "wall_seconds": time.perf_counter() - t0,
            }
        )
    return pd.DataFrame(rows)


def flag_unidentified(
    frame: pd.DataFrame,
    params: Sequence[str] = PARAM_COLUMNS,
    *,
    share: float = 0.5,
    band: float = 1.0,
) -> pd.DataFrame:
    """Per parameter: the share of consecutive changes exceeding ``band`` × the (larger of the
    two dates') standard error, the median absolute change, the median standard error, the
    number of dates without a standard error and the flag ``unidentified`` when the share
    exceeds ``share`` (NaN share — no usable standard error — is reported, not flagged)."""
    rows = []
    for name in params:
        se_col = f"{name}_se"
        if name not in frame or se_col not in frame:
            raise KeyError(f"frame lacks {name} / {se_col}")
        x = frame[name].to_numpy(dtype=float)
        se = frame[se_col].to_numpy(dtype=float)
        n_no_se = int(np.sum(~np.isfinite(se) | (se <= 0)))
        if x.size < 2:
            rows.append(
                {
                    "param": name,
                    "n_changes": 0,
                    "share_beyond_se": np.nan,
                    "median_abs_change": np.nan,
                    "median_se": float(np.nanmedian(se)) if se.size else np.nan,
                    "n_without_se": n_no_se,
                    "unidentified": False,
                }
            )
            continue
        d = np.abs(np.diff(x))
        s = band * np.maximum(se[1:], se[:-1])
        ok = np.isfinite(d) & np.isfinite(s) & (s > 0)
        beyond = float(np.mean(d[ok] > s[ok])) if ok.any() else np.nan
        rows.append(
            {
                "param": name,
                "n_changes": int(ok.sum()),
                "share_beyond_se": beyond,
                "median_abs_change": float(np.nanmedian(d)),
                "median_se": float(np.nanmedian(se)) if np.any(np.isfinite(se)) else np.nan,
                "n_without_se": n_no_se,
                "unidentified": bool(np.isfinite(beyond) and beyond > share),
            }
        )
    return pd.DataFrame(rows)


__all__ = ["BOOK_COLUMNS", "PARAM_COLUMNS", "flag_unidentified", "rolling_fit"]
