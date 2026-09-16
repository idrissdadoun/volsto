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
* *recalibrated rotation* — the surface rotated and the P1 set **refit** under a
  ``recalibration_policy`` (:data:`RECALIBRATION_POLICIES`): ``"sabr_linked"`` (the new marking
  policy: SABR reduction of the rotated surface → break-evens → refit with the same config and
  ``ssr_target``) or ``"sticky_breakeven"`` (the old policy: the ``SpotVolCovar`` target and the
  correlation target ``Corr_BE`` held at their pre-rotation values, the set recalibrated to them
  on the rotated surface, whose skew term structure, two-point ``Skew_SABR`` and ATMF vols are
  those of the rotated surface; the variance targets — ``VolVar``, ``VoV_BE`` — are not held in
  marking mode because the fitter derives them: step 3's VolVar target is
  ``(SpotVolCovar_model / Corr_BE)²``, so holding ``Corr_BE`` is what holds it —
  :func:`held_targets`), or ``"sticky_breakeven_skew"`` (the implementer's diagnostic variant,
  not one of the owner's two: the same two targets **and** the two-point ``Skew_SABR`` reference
  of the constraint held at their pre-rotation values, so the naked kernel is pinned to the old
  skew while the leverage absorbs the rotation — only the leverage integrals of the P1
  break-even see the rotated skew); the leverage recalibrated for the refit set;
* *shadow* = recalibrated − usual, estimated path by path on the four states (the desk's
  uncomputed rotation risk: the part of the fee move that comes from the marking parameters
  following the skew);
* *LV rotation* — the same product priced under pure local vol
  (:class:`~volsto.risk.engine.LVBuilder`, a Dupire rebuild on each of the base / up / down
  surfaces, same seed) and differenced centrally.

**Convention** (owner's decision on the M7 Part 3 report, :data:`ROTATION_CONVENTION`; a
convention issue, not a model result).  The deck's quantity is the *fee* = P1 price − LV price
and its P&L for a desk **short** the note.  The report states every quantity in one declared
convention: rota sign (+1 = the 6M 90/110 skew steepens by 0.56 vp), fee = P1 − LV, desk P&L per
+1 rota for a short position = −(d fee).  Hence ``fee_usual = usual − lv_rotation``,
``fee_recalibrated = recalibrated − lv_rotation``, ``fee_shadow = fee_recalibrated − fee_usual``
(**identical to the P1 shadow**: the LV rotation cancels path by path, so ``shadow`` is kept as
an alias of ``fee_shadow``), and ``desk_pnl_* = −fee_*`` (the same standard errors, sign flipped).
A positive P1 shadow therefore means the fee *rises* on a +1 rota when the marking set follows the
skew — a **negative** desk P&L for the short note, the deck's "fee increases, negative P&L" claim.
The P1 and LV pricings share the seed and the position-addressed draws (SPEC §5), so the spot
Brownian is common path by path wherever the two grids coincide; the standard error of every
per-path combination is exact either way.

Every level and difference carries its standard error; the report states the policy, the number
of leverage calibrations, LV builds, cache misses, whether anything was recalibrated and the wall
clock.  The two owner policies on the SPX 2022-12-30 3y autocall are the sign test of the greek
(owner, report decision ix, restated in the convention): the desk-P&L shadow **negative** under
``sticky_breakeven`` and ``|desk_pnl_shadow(sticky_breakeven)| > |desk_pnl_shadow(sabr_linked)|``
(measured 1.4×, the deck's ordering), verified in ``scripts/m7_p1_marking.py`` and
``tests/test_shadow_rotation.py``.  Bumps failing the no-arbitrage checks are halved (the engine's
rule) and both sides use the smaller size.  Checked by ``tests/test_shadow_rotation.py`` (the
rotation surface round trip, the SABR-skew move, the refit symmetry, the convention algebra on a
synthetic report and d(fee)/d(rota) on cached calibrations).
"""

from __future__ import annotations

import dataclasses
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from volsto.calibration.cache import LeverageCache
from volsto.calibration.fit_2f import (
    BreakEvenFitConfig,
    FitResult,
    fit_2f,
    fit_2f_marking,
    marking_targets_for,
)
from volsto.calibration.targets import TargetSet
from volsto.config import BergomiParams, CalibrationSpec, SimConfig, SurfacePerturbation
from volsto.market.surface import saturated_k
from volsto.market.varswap import xi0_curve
from volsto.products.base import Product
from volsto.risk.engine import (
    FloatArray,
    LSVBuilder,
    LVBuilder,
    RiskEngine,
    RiskState,
    Sensitivity,
    surface_of,
)

#: maturity floor of the rota scaling (1M, the M5 convention of ``atm_shift``)
ROTA_T_MIN = 1.0 / 12.0
ROTA_K_CAP = 0.5
RECALIBRATION_POLICIES = ("sabr_linked", "sticky_breakeven", "sticky_breakeven_skew")
#: the declared reporting convention (owner's text, verbatim; module docstring)
ROTATION_CONVENTION = (
    "rota +1: 6M 90/110 skew steepens by 0.56 vp (2/sqrt(T) vp per unit log-moneyness); "
    "fee = P1 price - LV price; desk P&L per +1 rota for a SHORT position = -(d fee)"
)
#: the P1 (LSV) states of the greek and the LV states, in the order the report lists them
P1_STATES = ("base", "up", "down", "up_refit", "down_refit")
LV_STATES = ("base", "up", "down")
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
    """The shadow-rotation greek in the declared ``convention`` (module docstring).  Levels
    (``p1_level``, ``lv_level``, ``fee = p1_level − lv_level``) are in the product's price units;
    the sensitivities per rota in the same units; ``per_vol_point`` converts them per vol point of
    90/110 skew at the listed tenors; ``fits`` holds the base and rotated P1 fits, ``param_moves``
    the parameter changes per rota of the refit.  ``shadow`` is the P1-price shadow, identical to
    ``fee_shadow`` (kept under its old name); the ``desk_pnl_*`` quantities are the negatives of
    the ``fee_*`` ones for a short position."""

    product: str
    policy: str
    size: float
    convention: str
    p1_level: Sensitivity
    lv_level: Sensitivity
    fee: Sensitivity
    lv_rotation: Sensitivity
    usual: Sensitivity
    recalibrated: Sensitivity
    shadow: Sensitivity
    fee_usual: Sensitivity
    fee_recalibrated: Sensitivity
    fee_shadow: Sensitivity
    desk_pnl_usual: Sensitivity
    desk_pnl_recalibrated: Sensitivity
    desk_pnl_shadow: Sensitivity
    fits: dict[str, FitResult]
    param_moves: dict[str, float]
    per_vol_point: dict[float, float]
    n_calibrations: int
    n_lv_builds: int
    n_cache_misses: int
    recalibrated_any: bool
    wall_seconds: float

    def rows(self) -> tuple[tuple[str, str, Sensitivity], ...]:
        """``(name, what it differentiates, sensitivity)`` per reported quantity, in order."""
        return (
            ("lv_rotation", "LV price", self.lv_rotation),
            ("usual", "P1 price", self.usual),
            ("recalibrated", "P1 price", self.recalibrated),
            ("fee_usual", "fee = P1 - LV", self.fee_usual),
            ("fee_recalibrated", "fee = P1 - LV", self.fee_recalibrated),
            ("fee_shadow", "fee = P1 - LV", self.fee_shadow),
            ("desk_pnl_usual", "desk P&L (short)", self.desk_pnl_usual),
            ("desk_pnl_recalibrated", "desk P&L (short)", self.desk_pnl_recalibrated),
            ("desk_pnl_shadow", "desk P&L (short)", self.desk_pnl_shadow),
        )

    def frame(self) -> pd.DataFrame:
        rows = []
        for name, of, s in self.rows():
            row: dict[str, Any] = {
                "greek": name,
                "of": of,
                "per_rota": s.value,
                "per_rota_se": s.stderr,
            }
            for T, vp in self.per_vol_point.items():
                row[f"per_vp_90_110_{T:g}y"] = s.value / vp
            rows.append(row)
        return pd.DataFrame(rows)

    def summary(self) -> str:
        lines = [
            f"convention: {self.convention}",
            f"shadow rotation on {self.product}; policy {self.policy}; central difference at "
            f"+/-{self.size:g} rota; leverage calibrations {self.n_calibrations} (cache misses "
            f"{self.n_cache_misses}, recalibrated: {'yes' if self.recalibrated_any else 'no'}), "
            f"LV builds {self.n_lv_builds}; wall clock {self.wall_seconds:.0f} s",
            f"levels: P1 {self.p1_level.value:.6f} +/- {self.p1_level.stderr:.6f}; LV "
            f"{self.lv_level.value:.6f} +/- {self.lv_level.stderr:.6f}; fee = P1 - LV "
            f"{self.fee.value:.6f} +/- {self.fee.stderr:.6f}",
            self.frame().to_string(index=False),
            "rota in vol points of 90/110 skew: "
            + ", ".join(f"{T:g}y {vp:.3f}" for T, vp in self.per_vol_point.items()),
            "refit parameter moves per rota: "
            + ", ".join(f"{k} {v:+.4f}" for k, v in self.param_moves.items()),
            "fit status: " + ", ".join(f"{k} {r.status}" for k, r in self.fits.items()),
        ]
        return "\n".join(lines)


def _paired(
    name: str,
    terms: Sequence[tuple[FloatArray, float]],
    *,
    unit: str,
    size: float,
    scheme: str,
    states: tuple[str, ...],
    n_paths: int,
) -> Sensitivity:
    """``Σ_i c_i · payoffs_i`` path by path with the standard error of the combination (the
    estimator of :meth:`RiskEngine.combination`, here across the P1 and LV engines)."""
    combo = sum((c * p for p, c in terms), np.zeros_like(terms[0][0]))
    n = combo.size
    return Sensitivity(
        name,
        float(combo.mean()),
        float(combo.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan"),
        unit,
        size,
        scheme,
        states,
        n_paths,
    )


def _negated(s: Sensitivity, name: str) -> Sensitivity:
    """``−s`` (the same standard error, sign flipped)."""
    return replace(s, name=name, value=-s.value)


def shadow_quantities(
    p1: Mapping[str, FloatArray],
    lv: Mapping[str, FloatArray],
    *,
    size: float,
    n_paths: int,
    p1_labels: Mapping[str, str] | None = None,
    lv_labels: Mapping[str, str] | None = None,
) -> dict[str, Sensitivity]:
    """Every level and per-rota quantity of the report from the per-path payoffs of the five P1
    states (:data:`P1_STATES`) and the three LV states (:data:`LV_STATES`), all priced on the
    same paths (module docstring, convention): ``p1_level``, ``lv_level``, ``fee``,
    ``lv_rotation``, ``usual``, ``recalibrated``, ``shadow``, ``fee_usual``,
    ``fee_recalibrated``, ``fee_shadow`` (= ``shadow``, the same combination), ``desk_pnl_usual``,
    ``desk_pnl_recalibrated``, ``desk_pnl_shadow`` (= −``fee_*``)."""
    missing = [k for k in P1_STATES if k not in p1] + [f"lv {k}" for k in LV_STATES if k not in lv]
    if missing:
        raise ValueError(f"payoffs missing for {missing}")
    pl = dict(p1_labels or {k: k for k in P1_STATES})
    ll = dict(lv_labels or {k: f"lv {k}" for k in LV_STATES})
    c = 1.0 / (2.0 * float(size))
    unit = "price per rota"
    lvl: dict[str, Any] = dict(unit="price", size=0.0, scheme="level", n_paths=n_paths)
    dif: dict[str, Any] = dict(unit=unit, size=float(size), scheme="central", n_paths=n_paths)
    up_dn = [(p1["up"], c), (p1["down"], -c)]
    up_dn_refit = [(p1["up_refit"], c), (p1["down_refit"], -c)]
    lv_up_dn = [(lv["up"], c), (lv["down"], -c)]
    lv_minus = [(lv["up"], -c), (lv["down"], c)]
    q: dict[str, Sensitivity] = {}
    q["p1_level"] = _paired("P1 price", [(p1["base"], 1.0)], states=(pl["base"],), **lvl)
    q["lv_level"] = _paired("LV price", [(lv["base"], 1.0)], states=(ll["base"],), **lvl)
    q["fee"] = _paired(
        "fee = P1 - LV",
        [(p1["base"], 1.0), (lv["base"], -1.0)],
        states=(pl["base"], ll["base"]),
        **lvl,
    )
    q["lv_rotation"] = _paired("LV rotation", lv_up_dn, states=(ll["up"], ll["down"]), **dif)
    q["usual"] = _paired("usual rotation (P1)", up_dn, states=(pl["up"], pl["down"]), **dif)
    q["recalibrated"] = _paired(
        "recalibrated rotation (P1)", up_dn_refit, states=(pl["up_refit"], pl["down_refit"]), **dif
    )
    q["shadow"] = _paired(
        "shadow rotation (P1)",
        [*up_dn_refit, (p1["up"], -c), (p1["down"], c)],
        states=(pl["up_refit"], pl["down_refit"], pl["up"], pl["down"]),
        **dif,
    )
    q["fee_usual"] = _paired(
        "usual rotation (fee)",
        [*up_dn, *lv_minus],
        states=(pl["up"], pl["down"], ll["up"], ll["down"]),
        **dif,
    )
    q["fee_recalibrated"] = _paired(
        "recalibrated rotation (fee)",
        [*up_dn_refit, *lv_minus],
        states=(pl["up_refit"], pl["down_refit"], ll["up"], ll["down"]),
        **dif,
    )
    # the LV terms cancel exactly: the fee shadow is the P1 shadow, the same per-path combination
    q["fee_shadow"] = replace(q["shadow"], name="shadow rotation (fee)")
    q["desk_pnl_usual"] = _negated(q["fee_usual"], "usual rotation (desk P&L, short)")
    q["desk_pnl_recalibrated"] = _negated(
        q["fee_recalibrated"], "recalibrated rotation (desk P&L, short)"
    )
    q["desk_pnl_shadow"] = _negated(q["fee_shadow"], "shadow rotation (desk P&L, short)")
    return q


def refit_on_rotated(
    rotated_surface: Any,
    base_fit: FitResult,
    cfg: BreakEvenFitConfig,
    *,
    ssr_target: float,
    policy: str,
) -> FitResult:
    """The P1 set refit on a rotated surface under ``policy`` (module docstring)."""
    if policy not in RECALIBRATION_POLICIES:
        raise ValueError(f"policy must be one of {RECALIBRATION_POLICIES}")
    if policy == "sabr_linked":
        return fit_2f_marking(rotated_surface, cfg, ssr_target=ssr_target)
    tg = marking_targets_for(rotated_surface, cfg, ssr_target=ssr_target)
    # the holding rule lives in held_targets (shared with the hedger's RecalibrationRule)
    sticky = held_targets(tg, base_fit.targets, policy)
    t_max = float(min(rotated_surface.max_maturity, max(sticky.pillars)))
    c = replace(cfg, pillars=tuple(float(t) for t in sticky.pillars))
    return fit_2f(sticky, xi0_curve(rotated_surface, t_max), c)


def held_targets(targets: TargetSet, base_targets: TargetSet, policy: str) -> TargetSet:
    """``targets`` (the marking targets read on a moved surface — a rotated surface, or the
    hedger's world state surface) with the quantities a recalibration ``policy`` holds at the
    **base** fit's values (:data:`RECALIBRATION_POLICIES`, module docstring).

    * ``"sabr_linked"`` holds nothing (``targets`` returned unchanged);
    * ``"sticky_breakeven"`` holds ``spot_vol_covar`` (step 2's covariance target) and
      ``correl_target`` (``Corr_BE``, from which step 3 builds its VolVar target
      ``(SpotVolCovar_model / Corr_BE)²``), so the skew constraint and the ATMF vols follow the
      moved surface.  The variance targets ``vol_var`` / ``vovol`` / ``vov_be_raw`` are **not**
      held: in marking mode the fitter never reads them (``fit_2f`` derives the VolVar target;
      ``vol_var`` only fills the reported ``volvar_target_requested`` column), so holding them
      was inert — measured on SPX 2022-12-30 at +1 rota: the five-array and the two-array
      holdings give identical parameters, and each of the two held arrays moves the fit on its
      own (``tests/test_shadow_rotation.py``).  Owner's decision of 2026-09-16: the definition
      names what the policy actually holds; no fitter support for held variance targets;
    * ``"sticky_breakeven_skew"`` holds the two-point ``Skew_SABR`` reference (``skew_target``)
      too.

    Marking-mode targets only (``ValueError`` otherwise: a historical fit keeps its empirical
    VolVar target, which this holding rule does not describe); the held pillars must match (the
    targets are per pillar); the policy's flag is appended to ``flags``.  Shared by
    :func:`refit_on_rotated` and the hedger's
    :class:`~volsto.hedging.hedger.RecalibrationRule`."""
    if policy not in RECALIBRATION_POLICIES:
        raise ValueError(f"policy must be one of {RECALIBRATION_POLICIES}")
    if policy == "sabr_linked":
        return targets
    if targets.mode != "marking" or base_targets.mode != "marking":
        raise ValueError(f"{policy} holds marking-mode targets only")
    b = base_targets
    if targets.pillars.shape != b.pillars.shape or not np.allclose(targets.pillars, b.pillars):
        raise ValueError(f"{policy} needs the same fitted pillars on both surfaces")
    held: dict[str, Any] = {
        "spot_vol_covar": b.spot_vol_covar.copy(),
        "correl_target": b.correl_target.copy(),
    }
    note = (
        "sticky_breakeven: SpotVolCovar and Corr_BE held at the base fit's values (the skew term "
        "structure, Skew_SABR of the two-point constraint and the ATMF vols are the moved "
        "surface's; the VolVar target follows from the held Corr_BE)"
    )
    if policy == "sticky_breakeven_skew":
        held["skew_target"] = b.skew_target.copy()
        note = (
            "sticky_breakeven_skew: SpotVolCovar, Corr_BE and the two-point Skew_SABR reference "
            "held at the base fit's values (only the leverage integrals see the moved skew)"
        )
    return replace(targets, flags=(*targets.flags, note), **held)


def rotation_states(
    base_spec: CalibrationSpec,
    cfg: BreakEvenFitConfig,
    *,
    ssr_target: float,
    size: float = 1.0,
    t_min: float = ROTA_T_MIN,
    k_cap: float = ROTA_K_CAP,
    engine: RiskEngine | None = None,
    policy: str = "sabr_linked",
) -> tuple[dict[str, RiskState], dict[str, FitResult], float]:
    """The five states of the greek (``base``, ``up``, ``down`` with the base set; ``up_refit``,
    ``down_refit`` with the refit sets under ``policy``) and the three fits; the base state's
    model is the fit on the base surface (``base_spec.model`` is replaced).  ``engine`` (when
    given) halves bumps that fail the arbitrage checks; the achieved size is returned."""
    if policy not in RECALIBRATION_POLICIES:
        raise ValueError(f"policy must be one of {RECALIBRATION_POLICIES}")
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
        f = refit_on_rotated(
            surface_of(st), fits["base"], cfg, ssr_target=ssr_target, policy=policy
        )
        fits[name] = f
        states[f"{name}_refit"] = RiskState(
            dataclasses.replace(st.spec, model=f.params), None, f"{st.label} refit {policy}"
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
    policy: str = "sabr_linked",
) -> ShadowRotationReport:
    """``d(fee)/d(rota)`` in the declared convention: the P1 usual, recalibrated and shadow
    rotations under ``policy``, the LV rotation of the same product, the fee and desk-P&L
    quantities (module docstring, :func:`shadow_quantities`)."""
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
        policy=policy,
    )
    mode = "recalibrate"
    p1 = {k: engine.priced(product, states[k], mode).payoffs for k in P1_STATES}
    # pure local vol of the same surfaces, same seed (the LV states carry the base model field,
    # which LVBuilder ignores: it rebuilds Dupire from each state's surface)
    lv_engine = RiskEngine(LVBuilder(states["base"]), pricing_sim)
    lv = {k: lv_engine.priced(product, states[k], mode).payoffs for k in LV_STATES}
    q = shadow_quantities(
        p1,
        lv,
        size=s,
        n_paths=pricing_sim.n_paths,
        p1_labels={k: states[k].label for k in P1_STATES},
        lv_labels={k: f"LV {states[k].label}" for k in LV_STATES},
    )
    moves = {
        name: (getattr(fits["up"].params, name) - getattr(fits["down"].params, name)) / (2.0 * s)
        for name in (f.name for f in dataclasses.fields(BergomiParams))
    }
    return ShadowRotationReport(
        product=product_name,
        policy=policy,
        size=s,
        convention=ROTATION_CONVENTION,
        p1_level=q["p1_level"],
        lv_level=q["lv_level"],
        fee=q["fee"],
        lv_rotation=q["lv_rotation"],
        usual=q["usual"],
        recalibrated=q["recalibrated"],
        shadow=q["shadow"],
        fee_usual=q["fee_usual"],
        fee_recalibrated=q["fee_recalibrated"],
        fee_shadow=q["fee_shadow"],
        desk_pnl_usual=q["desk_pnl_usual"],
        desk_pnl_recalibrated=q["desk_pnl_recalibrated"],
        desk_pnl_shadow=q["desk_pnl_shadow"],
        fits=fits,
        param_moves=moves,
        per_vol_point={
            float(T): rota_skew_vol_points(T, t_min=t_min, k_cap=k_cap) for T in vol_point_tenors
        },
        n_calibrations=engine.n_calibrations,
        n_lv_builds=lv_engine.n_calibrations,
        n_cache_misses=engine.n_cache_misses,
        recalibrated_any=bool(engine.n_cache_misses > 0 and allow_calibrate),
        wall_seconds=time.perf_counter() - t0,
    )


__all__ = [
    "LV_STATES",
    "P1_STATES",
    "RECALIBRATION_POLICIES",
    "ROTATION_CONVENTION",
    "ROTA_K_CAP",
    "ROTA_T_MIN",
    "ShadowRotationReport",
    "held_targets",
    "refit_on_rotated",
    "rota_skew_vol_points",
    "rota_slope",
    "rotation_perturbation",
    "rotation_shadow_sensitivity",
    "rotation_states",
    "shadow_quantities",
]
