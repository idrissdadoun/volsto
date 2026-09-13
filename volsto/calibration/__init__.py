"""Leverage calibration (SPEC §4): particle method, repricing diagnostics, cache."""

from __future__ import annotations

from volsto.calibration.cache import CacheMissError, LeverageCache, code_version
from volsto.calibration.diagnostics import CalibrationReport, reprice_surface
from volsto.calibration.fit_2f import (
    Fit2FConfig,
    FitResult,
    Stage3Inputs,
    fit_2f,
    fit_stage1,
    fit_stage2,
    skew_scale_degeneracy,
    stage3_validation,
)
from volsto.calibration.history import (
    HistoryEstimates,
    SurfaceHistory,
    SyntheticHistory,
    estimate_history,
    rolling_ssr,
    rolling_volvol,
    synthetic_2f_history,
)
from volsto.calibration.particle import CALIBRATION_CODE_TAG, CalibrationResult, calibrate_leverage
from volsto.calibration.stability import flag_unidentified, rolling_fit

__all__ = [
    "CALIBRATION_CODE_TAG",
    "CacheMissError",
    "CalibrationReport",
    "CalibrationResult",
    "Fit2FConfig",
    "FitResult",
    "HistoryEstimates",
    "LeverageCache",
    "Stage3Inputs",
    "SurfaceHistory",
    "SyntheticHistory",
    "calibrate_leverage",
    "code_version",
    "estimate_history",
    "fit_2f",
    "fit_stage1",
    "fit_stage2",
    "flag_unidentified",
    "reprice_surface",
    "rolling_fit",
    "rolling_ssr",
    "rolling_volvol",
    "skew_scale_degeneracy",
    "stage3_validation",
    "synthetic_2f_history",
]
