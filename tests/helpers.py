"""Shared helpers for the risk-layer tests: the flat Black–Scholes test bed, and the
comparison of fitted parameters against a stored record."""

from __future__ import annotations

import dataclasses

import pytest

from volsto.config import BergomiParams, CalibrationSpec, CurveConfig, MarketConfig, SSVIConfig
from volsto.risk import RiskState


def flat_state(sigma: float = 0.2, r: float = 0.02, q: float = 0.01) -> RiskState:
    """Flat surface at ``sigma`` with flat curves ``r``, ``q`` and a one-factor model whose
    parameters the Black–Scholes builder ignores."""
    spec = CalibrationSpec(
        market=MarketConfig(100.0, CurveConfig((1.0,), (r,)), CurveConfig((1.0,), (q,))),
        surface=SSVIConfig((1 / 12, 0.25, 0.5, 1.0, 2.0, 3.0), (sigma,) * 6, 0.0, 0.0, 0.5, 10.0),
        model=BergomiParams.one_factor(1.0, 1.0, 0.0),
    )
    return RiskState(spec)


#: Relative tolerance of a refit against its stored record.  The optimiser's last digits depend
#: on the machine (the same lock file gives fits that differ by 1e-14 to 1e-7 between two arm64
#: Macs, at any numba thread count), so the record is reproduced to this tolerance, not bit for
#: bit.
FIT_RTOL = 1e-6


def assert_params_close(
    got: BergomiParams, want: BergomiParams, msg: object = "", rel: float = FIT_RTOL
) -> None:
    """``got`` equals ``want`` field by field within the relative tolerance ``rel``."""
    assert dataclasses.asdict(got) == pytest.approx(dataclasses.asdict(want), rel=rel), msg
