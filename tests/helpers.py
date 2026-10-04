"""Shared helpers for the risk-layer tests: the flat Black–Scholes test bed, and the
comparison of a refit against a stored record (parameters where the fit is interior, what the
fit is judged on everywhere)."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterable
from typing import Any

import pytest

from volsto.analytics.bergomi import ssr_order1
from volsto.analytics.breakeven import first_order_breakevens
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


#: Relative tolerances of what a marking fit is judged on — spot/vol covariance, vol variance,
#: skew, first-order SSR — between a refit and its stored record (owner's decision 2026-10-03).
#: Measured that day on the five study specs across two Macs (SPEC §13.3):
#: interior fits (three specs): 2.1e-08 at most (vol variance; 6.8e-14 on the covariance, the
#: skew and the SSR);
FIT_JUDGED_RTOL_INTERIOR = 1e-6
#: fits flagged at the boundary (two specs): 5.7e-05 at most (the vol variance of
#: ``reference_ssr1.5_eps0.05``; 9.7e-10 on the other; 7.8e-16 on the covariance, skew and SSR).
FIT_JUDGED_RTOL_BOUNDARY = 1e-3
#: A fit with a correlation within this of ±1 is at the boundary (owner's decision 2026-10-03).
BOUNDARY_CORRELATION = 1e-3
#: ... and so is a fit whose ν is within this relative distance of its cap.
BOUNDARY_NU_REL = 1e-6
#: The maturities of the two-point skew constraint, always among the judged maturities.
SKEW_PILLARS = (1.0, 5.0)


def fit_at_boundary(params: BergomiParams, nu_cap: float | None) -> str:
    """Why the fit is at the boundary of its parameter domain (``""`` when it is interior):
    ``ν`` at its cap or a correlation within :data:`BOUNDARY_CORRELATION` of ±1.  There the
    parameters are not identified to the last digits and are not compared across machines."""
    why = []
    if nu_cap is not None and abs(params.nu - nu_cap) <= BOUNDARY_NU_REL * nu_cap:
        why.append(f"nu at its cap {nu_cap:g}")
    for name in ("rho12", "rho_SX1", "rho_SX2"):
        if 1.0 - abs(getattr(params, name)) < BOUNDARY_CORRELATION:
            why.append(f"{name} within {BOUNDARY_CORRELATION:g} of ±1")
    return "; ".join(why)


def assert_fit_judged_close(
    got: BergomiParams,
    want: BergomiParams,
    xi0: Any,
    pillars: Iterable[float],
    msg: object = "",
    *,
    rel: float,
) -> None:
    """The first-order break-evens (spot/vol covariance, vol variance, skew) and SSR of ``got``
    equal those of ``want`` within ``rel`` (:data:`FIT_JUDGED_RTOL_INTERIOR` or
    :data:`FIT_JUDGED_RTOL_BOUNDARY`) at every pillar and at the 1Y and 5Y skew points."""
    for T in sorted({*(float(t) for t in pillars), *SKEW_PILLARS}):
        a, b = first_order_breakevens(got, xi0, T), first_order_breakevens(want, xi0, T)
        for name in ("spot_vol_covar", "vol_var", "skew"):
            assert getattr(a, name) == pytest.approx(getattr(b, name), rel=rel), (msg, T, name)
        assert ssr_order1(got, xi0, T) == pytest.approx(ssr_order1(want, xi0, T), rel=rel), (
            msg,
            T,
            "ssr",
        )
