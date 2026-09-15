"""Shadow-rotation greek of the P1 marking calibration (SPEC §15 Part 3, owner's "M7 Part 3
FINAL", KEEP list; the recalibrating machinery of the M5 vega ladder §7.5 and parameter
sensitivities §7.9).

**Rota.**  One rota is a fixed-ATM rotation of the implied surface whose ATM skew moves by
``2/sqrt(T)`` vol points per unit of log-moneyness, i.e. ``2/sqrt(T) ln(110/90)`` vol points of
90/110 skew (0.568 at 6M, 0.401 at 1Y; the owner's "1 rota = 2/sqrt(6M) ln(110/90) = 0.56 at 6M"),
applied at every maturity with the maturity floored at ``t_min`` (1M) and the profile saturating
beyond ``k_cap`` (0.5) as the M5 skew bumps (§7.6; a rotation extended linearly into the far
wings breaks the calendar check).  Positive rotas steepen the skew (puts up, calls down):
``δσ(k, T) = −size · 0.02/sqrt(max(T, t_min)) · k_cap tanh(k/k_cap)`` (surface perturbation kind
``"rotation"``).  :func:`rota_skew_vol_points` converts per rota into per vol point of 90/110 skew
at a tenor (the exact saturated profile: 0.560 at 6M, 1.3% below the linear 0.568).

**The greek** (:func:`rotation_shadow_sensitivity`), central differences at ``±size`` rotas under
common random numbers, every leverage through the cache:

* *usual rotation* — the surface rotated, the P1 set **held**, the leverage recalibrated to the
  rotated surface (the ordinary recalibrated skew greek of §7.6);
* *recalibrated rotation* — the surface rotated, the SABR reduction → break-evens → the P1 set
  **refit** by :func:`~volsto.calibration.fit_2f.fit_2f_marking` with the same config and
  ``ssr_target``, the leverage recalibrated for the refit set;
* *shadow* = recalibrated − usual, estimated path by path on the four states (the desk's
  uncomputed rotation risk: the part of the fee move that comes from the marking parameters
  following the skew).

Every price and difference carries its standard error; the report states the number of
calibrations, cache misses, whether anything was recalibrated and the wall clock.  Bumps failing
the no-arbitrage checks are halved (the engine's rule) and both sides use the smaller size.
Checked by ``tests/test_shadow_rotation.py`` (the rotation surface round trip, the SABR-skew move,
the refit symmetry and d(fee)/d(rota) on cached calibrations).
"""

from __future__ import annotations

import dataclasses
import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from volsto.calibration.cache import LeverageCache
from volsto.calibration.fit_2f import BreakEvenFitConfig, FitResult, fit_2f_marking
from volsto.config import BergomiParams, CalibrationSpec, SimConfig, SurfacePerturbation
from volsto.market.surface import saturated_k
from volsto.products.base import Product
from volsto.risk.engine import LSVBuilder, RiskEngine, RiskState, Sensitivity, surface_of

#: maturity floor of the rota scaling (1M, the M5 convention of ``atm_shift``)
ROTA_T_MIN = 1.0 / 12.0
ROTA_K_CAP = 0.5
_LN_90_110 = (math.log(0.9), math.log(1.1))


def rotation_perturbation(
    size: float, *, t_min: float = ROTA_T_MIN, k_cap: float = ROTA_K_CAP
) -> SurfacePerturbation:
    """The ``"rotation"`` surface perturbation of ``size`` rotas (module docstring)."""
    return SurfacePerturbation(
        "rotation", {"size": float(size), "t_min": float(t_min), "k_cap": float(k_cap)}
    )


def rota_slope(T: float, *, t_min: float = ROTA_T_MIN) -> float:
    """ATM-skew move per rota in vol per unit log-moneyness: ``0.02/sqrt(max(T, t_min))``."""
    return 0.02 / math.sqrt(max(float(T), t_min))


def rota_skew_vol_points(
    T: float, *, t_min: float = ROTA_T_MIN, k_cap: float = ROTA_K_CAP, linear: bool = False
) -> float:
    """Vol points of 90/110 skew ``σ(ln 0.9) − σ(ln 1.1)`` per rota at tenor ``T`` (the linear
    value ``2/sqrt(T) ln(110/90)`` with ``linear=True``)."""
    lo, hi = _LN_90_110
    if linear:
        return 100.0 * rota_slope(T, t_min=t_min) * (hi - lo)
    kap = saturated_k(np.array([lo, hi]), k_cap)
    return float(100.0 * rota_slope(T, t_min=t_min) * (kap[1] - kap[0]))


@dataclass(frozen=True)
class ShadowRotationReport:
    """The shadow-rotation greek (module docstring).  Sensitivities are per rota in the product's
    price units; ``per_vol_point`` converts them per vol point of 90/110 skew at the listed
    tenors; ``fits`` holds the base and rotated P1 fits, ``param_moves`` the parameter changes per
    rota of the refit."""

    product: str
    size: float
    fee: Sensitivity
    usual: Sensitivity
    recalibrated: Sensitivity
    shadow: Sensitivity
    fits: dict[str, FitResult]
    param_moves: dict[str, float]
    per_vol_point: dict[float, float]
    n_calibrations: int
    n_cache_misses: int
    recalibrated_any: bool
    wall_seconds: float

    def frame(self) -> pd.DataFrame:
        rows = []
        for name, s in (
            ("usual", self.usual),
            ("recalibrated", self.recalibrated),
            ("shadow", self.shadow),
        ):
            row: dict[str, Any] = {"greek": name, "per_rota": s.value, "per_rota_se": s.stderr}
            for T, vp in self.per_vol_point.items():
                row[f"per_vp_90_110_{T:g}y"] = s.value / vp
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [
            f"shadow rotation on {self.product}: fee {self.fee.value:.6f} +/- {self.fee.stderr:.6f}"
            f"; central difference at +/-{self.size:g} rota; calibrations {self.n_calibrations} "
            f"(cache misses {self.n_cache_misses}, recalibrated: "
            f"{'yes' if self.recalibrated_any else 'no'}); wall clock {self.wall_seconds:.0f} s",
            self.frame().to_string(index=False),
            "rota in vol points of 90/110 skew: "
            + ", ".join(f"{T:g}y {vp:.3f}" for T, vp in self.per_vol_point.items()),
            "refit parameter moves per rota: "
            + ", ".join(f"{k} {v:+.4f}" for k, v in self.param_moves.items()),
            "fit status: " + ", ".join(f"{k} {r.status}" for k, r in self.fits.items()),
        ]
        return "\n".join(lines)


def rotation_states(
    base_spec: CalibrationSpec,
    cfg: BreakEvenFitConfig,
    *,
    ssr_target: float,
    size: float = 1.0,
    t_min: float = ROTA_T_MIN,
    k_cap: float = ROTA_K_CAP,
    engine: RiskEngine | None = None,
) -> tuple[dict[str, RiskState], dict[str, FitResult], float]:
    """The five states of the greek (``base``, ``up``, ``down`` with the base set; ``up_refit``,
    ``down_refit`` with the refit sets) and the three fits; the base state's model is the fit on
    the base surface (``base_spec.model`` is replaced).  ``engine`` (when given) halves bumps that
    fail the arbitrage checks; the achieved size is returned."""
    base_surface = surface_of(RiskState(base_spec))
    fits = {"base": fit_2f_marking(base_surface, cfg, ssr_target=ssr_target)}
    base = RiskState(dataclasses.replace(base_spec, model=fits["base"].params), None, "base")
    s = float(size)
    if engine is not None:
        _, s_up = engine.perturbed_state(
            base, lambda x: rotation_perturbation(x, t_min=t_min, k_cap=k_cap), s, label="rota+"
        )
        _, s_dn = engine.perturbed_state(
            base, lambda x: rotation_perturbation(-x, t_min=t_min, k_cap=k_cap), s, label="rota-"
        )
        s = min(s_up, s_dn)
    up = base.with_perturbation(rotation_perturbation(s, t_min=t_min, k_cap=k_cap), label="rota+")
    dn = base.with_perturbation(rotation_perturbation(-s, t_min=t_min, k_cap=k_cap), label="rota-")
    states = {"base": base, "up": up, "down": dn}
    for name, st in (("up", up), ("down", dn)):
        f = fit_2f_marking(surface_of(st), cfg, ssr_target=ssr_target)
        fits[name] = f
        states[f"{name}_refit"] = RiskState(
            dataclasses.replace(st.spec, model=f.params), None, f"{st.label} refit"
        )
    return states, fits, s


def rotation_shadow_sensitivity(
    product: Product,
    base_spec: CalibrationSpec,
    cfg: BreakEvenFitConfig,
    *,
    ssr_target: float,
    cache: LeverageCache,
    pricing_sim: SimConfig,
    size: float = 1.0,
    t_min: float = ROTA_T_MIN,
    k_cap: float = ROTA_K_CAP,
    allow_calibrate: bool = True,
    product_name: str = "product",
    vol_point_tenors: tuple[float, ...] = (0.5, 1.0),
) -> ShadowRotationReport:
    """``d(fee)/d(rota)``: usual, recalibrated and shadow (module docstring)."""
    t0 = time.perf_counter()
    base_surface = surface_of(RiskState(base_spec))
    base_fit = fit_2f_marking(base_surface, cfg, ssr_target=ssr_target)
    base = RiskState(dataclasses.replace(base_spec, model=base_fit.params), None, "base")
    builder = LSVBuilder(cache, base, allow_calibrate=allow_calibrate)
    engine = RiskEngine(builder, pricing_sim)
    states, fits, s = rotation_states(
        base_spec,
        cfg,
        ssr_target=ssr_target,
        size=size,
        t_min=t_min,
        k_cap=k_cap,
        engine=engine,
    )
    c = 1.0 / (2.0 * s)
    mode = "recalibrate"
    unit = "price per rota"

    def diff(name: str, terms: list[tuple[RiskState, str, float]]) -> Sensitivity:
        return engine.combination(name, product, terms, unit=unit, size=s, scheme="central")

    fee = engine.combination(
        "fee", product, [(states["base"], mode, 1.0)], unit="price", size=0.0, scheme="level"
    )
    usual = diff("usual rotation", [(states["up"], mode, c), (states["down"], mode, -c)])
    recal = diff(
        "recalibrated rotation", [(states["up_refit"], mode, c), (states["down_refit"], mode, -c)]
    )
    shadow = diff(
        "shadow rotation",
        [
            (states["up_refit"], mode, c),
            (states["down_refit"], mode, -c),
            (states["up"], mode, -c),
            (states["down"], mode, c),
        ],
    )
    moves = {
        name: (getattr(fits["up"].params, name) - getattr(fits["down"].params, name)) * c
        for name in (f.name for f in dataclasses.fields(BergomiParams))
    }
    return ShadowRotationReport(
        product_name,
        s,
        fee,
        usual,
        recal,
        shadow,
        fits,
        moves,
        {float(T): rota_skew_vol_points(T, t_min=t_min, k_cap=k_cap) for T in vol_point_tenors},
        engine.n_calibrations,
        engine.n_cache_misses,
        bool(engine.n_cache_misses > 0 and allow_calibrate),
        time.perf_counter() - t0,
    )


__all__ = [
    "ROTA_K_CAP",
    "ROTA_T_MIN",
    "ShadowRotationReport",
    "rota_skew_vol_points",
    "rota_slope",
    "rotation_perturbation",
    "rotation_shadow_sensitivity",
    "rotation_states",
]
