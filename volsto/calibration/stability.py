"""Stability of the two-factor fit over time (SPEC §15 Part 4).

:func:`rolling_fit` refits stage 1 every ``stage1_every`` dates on the trailing ``window_vol``
window and stage 2 every ``stage2_every`` dates (stage 1 held at its latest refit), returning
one row per date with the parameters and their standard errors.  :func:`flag_unidentified`
marks the parameters whose day-to-day changes exceed their standard-error band on more than
``share`` of the days (a parameter that moves by more than its own uncertainty from one day to
the next is being driven by noise, not by information — unidentified in the SPEC's sense);
:func:`degeneracy_profile` is the objective along the ``ν ⟷ correlations`` direction of
:func:`~volsto.calibration.fit_2f.skew_scale_degeneracy` (re-exported for the backtest
study).  Checked by ``tests/test_fit_2f.py::test_rolling_fit_and_flags``.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from volsto.calibration.fit_2f import (
    Fit2FConfig,
    Stage1Report,
    fit_stage1,
    fit_stage2,
    skew_scale_degeneracy,
)
from volsto.calibration.history import SurfaceHistory

PARAM_COLUMNS = ("nu", "theta", "k1", "k2", "rho12", "rho_SX1", "rho_SX2", "chi")


def rolling_fit(
    history: SurfaceHistory,
    cfg: Fit2FConfig | None = None,
    *,
    stage1_every: int = 21,
    stage2_every: int = 1,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    refine_mixing: bool = False,
) -> pd.DataFrame:
    """Parameter time series: columns ``date, nu, theta, k1, k2, rho12, rho_SX1, rho_SX2, chi,
    nu_se, theta_se, k1_se, k2_se, rho_SX1_se, chi_se, stage1_objective, stage2_objective,
    stage1_refit`` (``True`` on the dates stage 1 was refit).  ``start`` defaults to the first
    date with ``window_vol`` increments behind it; the mixing refinement is off by default
    (a daily loop over Monte Carlo skews would dominate the cost)."""
    c = cfg or Fit2FConfig()
    if refine_mixing != c.refine_mixing:
        c = Fit2FConfig(**{**c.__dict__, "refine_mixing": refine_mixing})
    if stage1_every < 1 or stage2_every < 1:
        raise ValueError("stage1_every and stage2_every must be positive")
    first = c.window_vol if start is None else history.date_index(start)
    last = history.date_index(end)
    if first < c.window_vol:
        raise ValueError(
            f"start must leave {c.window_vol} increments behind it (first possible date "
            f"{history.dates[c.window_vol].date()})"
        )
    rows: list[dict[str, object]] = []
    s1: Stage1Report | None = None
    for i in range(first, last + 1):
        date = history.dates[i]
        refit = s1 is None or (i - first) % stage1_every == 0
        if refit:
            s1 = fit_stage1(history, c, date)
        assert s1 is not None
        if (i - first) % stage2_every == 0:
            s2 = fit_stage2(history, c, s1.best.params, date)
            p = s2.params
            rows.append(
                {
                    "date": date,
                    "nu": p.nu,
                    "theta": p.theta,
                    "k1": p.k1,
                    "k2": p.k2,
                    "rho12": p.rho12,
                    "rho_SX1": p.rho_SX1,
                    "rho_SX2": p.rho_SX2,
                    "chi": s2.chi,
                    "nu_se": s1.best.stderr["nu"],
                    "theta_se": s1.best.stderr["theta"],
                    "k1_se": s1.best.stderr["k1"],
                    "k2_se": s1.best.stderr["k2"],
                    "rho_SX1_se": s2.stderr["rho_SX1"],
                    "chi_se": s2.stderr["chi"],
                    "stage1_objective": s1.best.objective,
                    "stage2_objective": s2.objective,
                    "stage1_refit": refit,
                    "k2_flag": s1.best.k2_flag,
                }
            )
    return pd.DataFrame(rows)


def flag_unidentified(
    frame: pd.DataFrame,
    params: Sequence[str] = ("nu", "theta", "k1", "k2", "rho_SX1", "chi"),
    *,
    share: float = 0.5,
    band: float = 1.0,
) -> pd.DataFrame:
    """Per parameter: the share of consecutive changes exceeding ``band`` × the (larger of the
    two dates') standard error, the median absolute change, the median standard error and the
    flag ``unidentified`` when the share exceeds ``share``.  Stage-1 parameters are compared on
    the stage-1 refit dates only (between refits they are constant by construction)."""
    rows = []
    for name in params:
        se_col = f"{name}_se"
        if name not in frame or se_col not in frame:
            raise KeyError(f"frame lacks {name} / {se_col}")
        sub = frame
        if name in ("nu", "theta", "k1", "k2") and "stage1_refit" in frame:
            sub = frame[frame["stage1_refit"].astype(bool)]
        x = sub[name].to_numpy(dtype=float)
        se = sub[se_col].to_numpy(dtype=float)
        if x.size < 2:
            rows.append(
                {
                    "param": name,
                    "n_changes": 0,
                    "share_beyond_se": np.nan,
                    "median_abs_change": np.nan,
                    "median_se": float(np.nanmedian(se)) if se.size else np.nan,
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
                "median_se": float(np.nanmedian(se)),
                "unidentified": bool(np.isfinite(beyond) and beyond > share),
            }
        )
    return pd.DataFrame(rows)


degeneracy_profile = skew_scale_degeneracy

__all__ = ["PARAM_COLUMNS", "degeneracy_profile", "flag_unidentified", "rolling_fit"]
