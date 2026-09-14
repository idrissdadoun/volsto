"""Leverage calibration (SPEC §4): particle method, repricing diagnostics, cache; the
break-even fit of the two-factor model (SPEC §15 Part 3) and its stability (Part 4)."""

from __future__ import annotations

from volsto.calibration.cache import CacheMissError, LeverageCache, code_version
from volsto.calibration.diagnostics import CalibrationReport, reprice_surface
from volsto.calibration.fit_2f import (
    BreakEvenFitConfig,
    FitResult,
    Stage3Inputs,
    Stage3Report,
    attainable_ssr,
    fit_2f,
    fit_2f_historical,
    fit_2f_marking,
    k1_profile,
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
from volsto.calibration.targets import TargetSet, historical_targets, marking_targets

__all__ = [
    "CALIBRATION_CODE_TAG",
    "BreakEvenFitConfig",
    "CacheMissError",
    "CalibrationReport",
    "CalibrationResult",
    "FitResult",
    "HistoryEstimates",
    "LeverageCache",
    "Stage3Inputs",
    "Stage3Report",
    "SurfaceHistory",
    "SyntheticHistory",
    "TargetSet",
    "attainable_ssr",
    "calibrate_leverage",
    "code_version",
    "estimate_history",
    "fit_2f",
    "fit_2f_historical",
    "fit_2f_marking",
    "flag_unidentified",
    "historical_targets",
    "k1_profile",
    "marking_targets",
    "reprice_surface",
    "rolling_fit",
    "rolling_ssr",
    "rolling_volvol",
    "stage3_validation",
    "synthetic_2f_history",
]
