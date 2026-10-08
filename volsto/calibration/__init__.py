"""Leverage calibration (SPEC §4): particle method, repricing diagnostics, cache; the
break-even fit of the two-factor model (SPEC §15 Part 3) and its stability (Part 4); the
particle calibration of the local correlation model and its cache (SPEC §8.7)."""

from __future__ import annotations

from volsto.calibration.cache import CacheMissError, LeverageCache, code_version
from volsto.calibration.diagnostics import CalibrationReport, reprice_surface
from volsto.calibration.fit_2f import (
    BreakEvenFitConfig,
    FitResult,
    FitSpec,
    Stage3Inputs,
    Stage3Report,
    fit_2f,
    fit_2f_historical,
    fit_2f_marking,
    k1_profile,
    load_fit_spec,
    mean_abs_leverage_deviation,
    stage3_validation,
    write_fit_spec,
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
from volsto.calibration.lc_cache import LCDiagnostics, LocalCorrelationCache, build_lc_market
from volsto.calibration.local_correlation import (
    LC_CODE_TAG,
    IndexRepricingReport,
    LCCalibrationResult,
    ParametricFit,
    calibrate_constant_lambda,
    calibrate_local_correlation,
    calibrate_parametric_lambda,
    reprice_index_smile,
)
from volsto.calibration.particle import CALIBRATION_CODE_TAG, CalibrationResult, calibrate_leverage
from volsto.calibration.stability import flag_unidentified, rolling_fit
from volsto.calibration.targets import TargetSet, historical_targets, marking_targets

__all__ = [
    "CALIBRATION_CODE_TAG",
    "LC_CODE_TAG",
    "BreakEvenFitConfig",
    "CacheMissError",
    "CalibrationReport",
    "CalibrationResult",
    "FitResult",
    "FitSpec",
    "HistoryEstimates",
    "IndexRepricingReport",
    "LCCalibrationResult",
    "LCDiagnostics",
    "LeverageCache",
    "LocalCorrelationCache",
    "ParametricFit",
    "Stage3Inputs",
    "Stage3Report",
    "SurfaceHistory",
    "SyntheticHistory",
    "TargetSet",
    "build_lc_market",
    "calibrate_constant_lambda",
    "calibrate_leverage",
    "calibrate_local_correlation",
    "calibrate_parametric_lambda",
    "code_version",
    "estimate_history",
    "fit_2f",
    "fit_2f_historical",
    "fit_2f_marking",
    "flag_unidentified",
    "historical_targets",
    "k1_profile",
    "load_fit_spec",
    "marking_targets",
    "mean_abs_leverage_deviation",
    "reprice_index_smile",
    "reprice_surface",
    "rolling_fit",
    "rolling_ssr",
    "rolling_volvol",
    "stage3_validation",
    "synthetic_2f_history",
    "write_fit_spec",
]
