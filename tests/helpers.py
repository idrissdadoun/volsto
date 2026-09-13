"""Shared helpers for the risk-layer tests: the flat Black–Scholes test bed."""

from __future__ import annotations

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
