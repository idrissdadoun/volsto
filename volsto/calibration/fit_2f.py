"""P1 (two-factor LSV, "P1 deco") marking calibration by SABR break-evens (SPEC §15 Part 3; the
owner's "M7 Part 3 FINAL" specification, which supersedes every earlier Part 3 design; Bergomi
ch. 7 §7.4, ch. 8 eq. 8.54, ch. 9 §9.2, ch. 12 eq. 12.52, in the break-even parametrisation of
:mod:`volsto.analytics.reparam`).

**Model.**  ``dS/S = (r − q) dt + sqrt(ξ_t^t) σ(t, S) dW`` with the two-factor Bergomi forward
variance and the leverage ("decoration") ``σ(t, S)``; seven parameters ``(k1, k2, ν, θ, ρ_SX,
ρ_SY, ρ_XY)`` fitted in the break-even parametrisation ``(k1, k2, ω1, ω2, λ1, λ2, χ)``.  The
library is its own system (SSVI/eSSVI surfaces, its own conventions, SSR and skew tolerance as
free inputs), a research / risk tool, not a replica of a production marking engine.

**Targets** (:mod:`volsto.calibration.targets`, steps 0–1): per pillar ``T ≥ mat_min`` (3M) to
10Y, ``VoV_BE = (atf_3M/atf_T)(ssr_target/2) VoV_SABR`` (absolute), ``Corr_BE = Corr_SABR``,
``SpotVolCovar_target = Corr_BE VoV_BE``, ``VolVar_target = VoV_BE²`` (SmoothBreakEven on).
``ssr_target`` feeds ``VoV_BE`` only.

**The P1 break-evens** (first order, absolute vol units, ``σ_0 = atf(1M)``, the market ATMF vol
as prefactor of the sensitivities — the gate-validated convention of
:mod:`volsto.analytics.breakeven`)::

    SensiX_i(T)  = ½ ω_i A_i(T) atf_T                                            (eq. 7.38)
    Skew_naked(T) = λ1 J1(T) + λ2 J2(T)                                  (eq. 8.54, σ̂³ in J)
    SensiSpot(T) = σ_0 [ (S_T − λ·J_T) + (1/T) ∫₀ᵀ f(t) (S_t − λ·J_t) dt ]   (leverage, eq. 12.52)
    SpotVolCovar_P1(T) = SensiSpot + ρ_SX SensiX + ρ_SY SensiY = SensiSpot + ½ atf_T λ·A_T
    VolVar_P1(T) = SensiSpot² + 2 SensiSpot ½ atf_T λ·A_T + SensiX² + SensiY² + 2 ρ_XY SensiX SensiY

with ``S_t`` the market ATM skew, ``f(t) = σ²(t)/(σ̂_t σ̂_T)`` on the market ATMF curve and ``J_t``
the kernel's order-one skew at every ``t ∈ (0, T]``. ``SensiSpot`` is the spot sensitivity the
leverage generates when it absorbs the skew residual ``S − Skew_naked`` (eq.  12.52 in covariance
form: ``SpotVolCovar_P1 = σ_0 R_LSV S`` at order one); it is zero when the naked skew equals the
market skew at every maturity, and ``SpotVolCovar_P1`` and ``Skew_naked`` are affine in ``λ`` at
fixed ``(k1, k2)`` (:class:`P1Maps`).  *Reading of the specification (flagged):* the owner's
``VolVar_P1`` carries ``σ_0² f(T)²`` and ``σ_0 f(T) A`` terms, i.e.  a non-zero ``SensiSpot = σ_0
f(T)`` — the leverage-included (P1-deco) break-even, the ``SVC_PILV`` of the previous message — so
``SpotVolCovar_P1`` is this form, not the naked kernel's (``SensiSpot = 0``, reported as
``svc_naked``) — **confirmed by the owner (report decision i: production's definition; the naked
reading is trivially satisfied by ρ = Corr_SABR)**.  The owner's ``VolVar_P1`` as written has both
cross terms at half weight (``(λ_i/2) σ_0 f A_i`` and ``(ω1 ω2/4) ρ_XY A1 A2``; the quadratic form
``(SensiSpot + SensiX + SensiY)²`` has ``2 SensiSpot ρ_Si SensiX_i = SensiSpot λ_i A_i atf`` and ``2
ρ_XY SensiX SensiY = (ω1 ω2/2) ρ_XY A1 A2 atf²``); the engine's full quadratic form — the one the
analytic-vs- simulation gate validated — is used (owner, report decision ii; the slide's half-weight
cross-term coefficients remain an **unresolved normalisation** of that formula, recorded here).  In
historical mode (no surface) the skew residual is interpolated linearly in ``t`` between the pillars
(flat outside): ``(1/T) ∫ f (S − λ·J)`` uses the pillar values of ``S`` and ``J``, so the leverage
term vanishes whenever the naked skew matches the market at the pillars (the synthetic recovery
test).

**Step 2 — first minimisation** over ``(k1, λ1, λ2)``, ``k2`` fixed (0.2)::

    min Σ_i w_i^Covar (SpotVolCovar_P1(T_i) − SpotVolCovar_target(T_i))²
    s.t. Skew_naked(T_s) ∈ [(1 − eps_s), (1 + eps_s)] · Skew_SABR(T_s)      T_s = 1Y
         Skew_naked(T_l) ∈ [(1 − eps_l), (1 + eps_l)] · Skew_SABR(T_l)      T_l = 5Y
         |λ1| + |λ2| ≤ 2 ν_cap

(``skew_mode="twopoint"``, default; the interval is read between the two bounds whatever the
sign of the skew).  ``w^Covar`` is ``"relative"`` (``1/target²``, default: every pillar's relative
error counts alike), ``"uniform"`` (normalised by the mean squared target) or explicit per-pillar
values.  At fixed ``k1`` the problem is a 2-D convex QP solved exactly by candidate enumeration
(:func:`_qp2`: unconstrained point, every edge, every vertex); the outer ``k1`` runs on a
geometric grid then a bounded scalar refinement.  The box row is the ν feasibility of step 3:
``min ν`` over ``ω_i ≥ |λ_i|`` is ``|λ1 + λ2|/2``, equal to ``(|λ1| + |λ2|)/2`` for same-sign
loadings (conservative for opposite signs).  The skew constraint pillars must be fitted pillars;
a constraint maturity outside them (5Y on a surface quoted to 3Y, a history to 3Y) is moved to
the nearest fitted pillar with a note, never dropped silently.  ``skew_mode="soft"`` replaces the
two-point constraint by the all-pillar penalty ``skew_weight Σ_i (Skew_naked_i/Skew_SABR_i − 1)²``
added to the covariance objective (normalised as the relative weights).  ``skew_mode="auto"``
(default) is two-point in marking mode and **soft in historical mode** (owner, report decision v:
the two-point configuration leaves ρ_SX1 unidentified on the three-year synthetic recovery,
−0.664 against −0.759; soft at the default weight 10 gives −0.716 / −0.461, ν 1.798, θ 0.245, k1
5.78, inside the owner's tolerances).

**Binding and infeasibility (never a silent railed fit).**  With ``eps ≥ 0`` the two skew slabs
always intersect in the ``λ`` plane (their normals ``J(1Y)``, ``J(5Y)`` are independent for ``k1 ≠
k2``), so ``ssr_target`` cannot make the constraint set empty: an incompatible ``(ssr_target,
skew_eps)`` pair shows as a **binding** constraint (the SpotVolCovar the SSR asks for needs a
naked skew outside the tolerance) and the fit carries :data:`BINDING_MESSAGE` naming the edge,
the maturity, the naked-vs-market skew and the achieved-vs-target SpotVolCovar per pillar.  The
set becomes **infeasible** only with the ν box (the market skew within ``eps`` needs ``|λ1| +
|λ2| > 2 ν_cap`` at every ``k1``): the least-violation fit (the skew rows uniformly relaxed by the
smallest ``δ`` in units of ``|Skew_SABR|``, the box kept hard so that step 3 stays admissible, by a
linear programme) is returned with :data:`INFEASIBLE_MESSAGE`.  :attr:`FitResult.status` is
``"interior"``, ``"binding"`` or ``"infeasible"``.

**Step 3 — second minimisation** over ``(ω1, ω2, χ)`` with ``(k1, k2, λ1, λ2)`` fixed::

    min Σ_i w_i^VolVar (VolVar_P1(T_i) − VolVar_target(T_i))²
    s.t. ω_i ≥ |λ_i|,  ν = ½ sqrt(ω1² + ω2² + 2 ρ_XY ω1 ω2) ≤ ν_cap,  χ ∈ [−0.99, 0.99]

(SLSQP from ten starts, ``ρ_Si = λ_i/ω_i``, ``ρ_XY = ρ_SX ρ_SY + χ sqrt(1 − ρ_SX²) sqrt(1 −
ρ_SY²)``), then the inverse reparametrisation to ``(ν, θ, ρ_SX, ρ_SY, ρ_XY)``.  **Correlation kept
at ``ρ_SABR`` always** (owner): the VolVar target is rebuilt from the achieved covariance,
``VolVar_target_i = (SpotVolCovar_P1,i / Corr_BE,i)²`` — equal to ``VoV_BE²`` when step 2 meets its
target — so a missed covariance never forces ``|ρ| = 1``; the requested ``VoV_BE²`` is reported next
to it.  Historical mode keeps its empirical VolVar target (no SABR correlation there; the empirical
``SpotVolCovar/sqrt(VolVar)`` reaches −1.02 at 3M on the one-year synthetic history, and the rebuild
from it biased ν by −20%). ``ν_cap`` (default 3.5 since report decision viii, was 2.5) is a
config value, not a hard-wired rail: when a fit binds it (step 2 box or step 3)
:data:`NU_CAP_WARNING` is logged and attached to the messages.  **Stage-3 assertion** (decision
viii, restricted to the engine-bias term by the owner on the M7 Part 3 report): the simulated
SpotVolCovar and VolVar of the calibrated model must be within ``stage3_tolerance`` (10%) of the
**analytic (first-order) break-evens evaluated at the fitted parameters** — the engine bias
(:func:`breakeven_check`).  The miss of the fit's *targets* by the first-order fit is a binding
fit, already reported by the binding status: it is reported per pillar in the check table
(``gap_vs_target``, ``first_order_miss``) and never asserted.  The split makes a higher-ν fit
self-checking since the first-order engine is about 15% biased beyond ν ≈ 4; :func:`fit_2f`
raises :class:`BreakEvenValidationError` (carrying the result) when it fails.
``iterate_against_simulation=k`` refits ``k`` times with the targets divided by the cumulative
simulated/analytic factors measured at each solution and reports the convergence.  **ρ12
diagnostic** (decision vii): ``|ρ12| > rho12_flag`` (0.9) is noted as the two-factor structure
collapsing.

**Always reported** (:meth:`FitResult.summary`, :meth:`FitResult.config_yaml`): the fitted
parameters; the naked skew at 1Y / 5Y against the market (within ``eps`` or the binding message);
the free short-end naked skew (1M and the pillars below 1Y); per pillar the SpotVolCovar /
VolVar targets and achieved values, the implied correlation and the first-order P1 SSR; with
stage 3 the calibrated LSV's numerical SSR (a diagnostic, never a target: at ``ssr_target = 1`` the
model cannot realise SSR 1 without leverage, pure SV floors near 1.5), the mean ``|L − 1|`` and the
forward 90/110 skew at 1y-into-1y and 2y-into-1y against the spot 1y skew — a diagnostic,
not a target (owner, report decision vi: the measured 1.0–1.17 is reported; the 130–150% figure
was a long-maturity heuristic with different SSR inputs).

**Stage 3 — validation, nothing refit** (:func:`stage3_validation`): the leverage is calibrated
for the fitted parameters (or a cached ``model=`` is used), and the report carries (a) mean ``|L −
1|``, (b) the LSV's numerical SSR per pillar with standard errors next to ``ssr_target``, the SSR
the targets imply and the first-order P1 SSR, (c) the naked kernel's skew by the exact mixing
derivative, (d) the LSV's simulated SpotVolCovar / VolVar against the targets, (e) the forward
smile diagnostic.  Every report states its wall clock and whether it recalibrated.

Checked by ``tests/test_fit_2f.py`` and, for the shadow-rotation greek built on this fitter,
``tests/test_shadow_rotation.py``.
"""

from __future__ import annotations

import itertools
import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from scipy.optimize import linprog, minimize, minimize_scalar

from volsto.analytics.breakeven import Kernels, _gl
from volsto.analytics.p1_mlp import MlpGrid, MlpPillar
from volsto.analytics.reparam import BreakEvenParams
from volsto.calibration.history import WINDOW_SSR, WINDOW_VOL, SurfaceHistory
from volsto.calibration.targets import (
    DEFAULT_ATF_REF,
    DEFAULT_MAT_MIN,
    DEFAULT_RADICAND_FLOOR,
    DEFAULT_SABRW_POWER,
    DEFAULT_TARGET_PILLARS,
    SABR_CURVATURE_H,
    SIGMA0_MATURITY,
    SsrInput,
    Step0Triplets,
    TargetSet,
    historical_targets,
    marking_targets,
)
from volsto.config import BergomiParams, from_mapping, to_mapping
from volsto.market.varswap import ForwardVarianceCurve, xi0_curve

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

#: the fixed slow mean reversion (owner default)
DEFAULT_K2 = 0.2
#: the ν cap (config; owner default 3.5 since the M7 Part 3 report decision viii, was 2.5)
DEFAULT_NU_CAP = 3.5
#: the soft-mode skew weight (historical-mode default; weight 10 meets the owner's recovery
#: tolerances on the three-year mixing history, weight 1 misses ρ_SX1 by 0.073)
DEFAULT_SKEW_WEIGHT = 10.0
#: stage-3 assertion: simulated SpotVolCovar and VolVar within this relative distance of the
#: first-order break-evens at the fitted parameters — the engine bias (owner, report decision
#: viii; restricted to the engine-bias term on the M7 Part 3 report, the target miss reported)
DEFAULT_STAGE3_TOLERANCE = 0.10
#: ``|ρ12|`` above which the two-factor structure is flagged as collapsing (report decision vii)
RHO12_COLLAPSE = 0.9
#: smallest eigenvalue of the fitted Brownian correlation matrix of ``(S, X¹, X²)`` below which the
#: fit is noted as near-singular (:func:`correlation_min_eigenvalue`).  At the repaired eSSVI anchor
#: of 2026-09-22 the SPX marking fit reads 6·10⁻⁴ (``χ`` at its −0.99 bound, every break-even
#: correlation target at or beyond −0.97 from 1y): the pricing paths' ``(ln S, X¹, X²)`` are then
#: nearly collinear, and the regressions on the factors (the hedger's conditional pricer, the
#: minimum-variance delta) are ill-identified — SPEC §8.2.  The plain-SSVI anchor read 0.030
CORRELATION_EIGEN_FLAG = 1e-2
#: points of the geometric ``k2`` grid when ``BreakEvenFitConfig.k2_bounds`` is set and ``k2_grid``
#: is ``None`` (:func:`_optimise_k2`)
K2_GRID_DEFAULT = 9
#: ``note`` of a stage-3 check row without a finite non-zero first-order value: the verdict
#: falls back to the gap vs the fit's target (:func:`breakeven_check`)
NO_FIRST_ORDER_NOTE = "no first-order value: target gap used"
#: the two-point skew tolerance (owner default, both points)
DEFAULT_SKEW_EPS = 0.10
#: the two constraint maturities ``(T_s, T_l)``
DEFAULT_SKEW_PILLARS: tuple[float, float] = (1.0, 5.0)
SKEW_MODES = ("twopoint", "soft")
#: ``skew_mode`` choices: ``"auto"`` resolves to ``"twopoint"`` in marking mode and ``"soft"`` in
#: historical mode (report decision v; :func:`resolve_skew_mode`)
SKEW_MODE_CHOICES = ("auto", *SKEW_MODES)
WEIGHT_KINDS = ("relative", "uniform")
#: term-structure factor ``f(t)`` of the leverage integrals
TERM_STRUCTURE_KINDS = ("atmf", "flat", "vs")
KERNEL_CURVES: tuple[str, ...] = ("atmf",)
"""Values of :attr:`BreakEvenFitConfig.kernel_curve` other than ``None`` (the caller's curve)."""
#: the risk regime paired with a marking fit (sticky strike: ``ssr_target = 1``)
RISK_REGIME = "sticky_strike"
#: standard errors above this are reported as NaN (numerically unidentified; :mod:`stability`)
MAX_FINITE_SE = 1e3
#: the owner's ν-cap warning (logged and attached to the messages when a fit binds the cap)
NU_CAP_WARNING = (
    "nu at cap {cap:g}; first-order break-even engine ~15% biased beyond ~4; raise only if "
    "stage-3 simulation validates"
)
#: a skew constraint binding (module docstring); filled per binding edge
BINDING_MESSAGE = (
    "skew constraint binds at T={T:g} ({edge} edge: naked skew = {factor} x Skew_SABR = "
    "{bound:+.5f}): naked skew {naked:+.5f} vs market {market:+.5f} ({gap:+.1%}, eps {eps:g}); "
    "SpotVolCovar achieved vs target: {svc}"
)
#: the QP infeasible at every k1 (module docstring)
INFEASIBLE_MESSAGE = (
    "infeasible: no (lambda1, lambda2) meets {what} inside {box} "
    "at any k1 in [{k1_lo:g}, {k1_hi:g}]; returned the least-violation fit (skew bounds "
    "relaxed by {delta:.4f} x |Skew_SABR| at k1 = {k1:.4g}, box kept): naked skew {skews}; "
    "SpotVolCovar achieved vs target: {svc}"
)
_BOX_LABELS = ("nu box l1+l2", "nu box l1-l2", "nu box -l1+l2", "nu box -l1-l2")
_FACTOR_BOX_LABELS = ("lambda1 upper", "lambda1 lower", "lambda2 upper", "lambda2 lower")
#: the desk note's step-1 box ``|λ_i| ≤ 0.99 ω_max`` (its §3 parameter table), used without a ν cap
LAMBDA_BOX_FRACTION = 0.99
#: values of :attr:`BreakEvenFitConfig.engine` other than ``None`` (the first-order engine)
FIT_ENGINES: tuple[str, ...] = ("mlp",)
#: the quadrature of the note's closed forms inside the fit (SensiX / SensiY to ~1e-7, SensiSpot
#: to ~1e-5 of the fine grid; ``volsto.analytics.p1_mlp``)
MLP_FIT_GRID = MlpGrid(n_fine=400, n_gl=24, n_u=32)
#: values of :attr:`BreakEvenFitConfig.volvar_target` other than ``None`` (the M7 rebuild)
VOLVAR_TARGETS: tuple[str, ...] = ("direct",)
#: values of :attr:`BreakEvenFitConfig.step0` other than ``None`` (the surface's ATM derivatives)
STEP0_SOURCES: tuple[str, ...] = ("sabrw",)
#: the desk note's bounds (its §3 table): no ν cap, ``ω_i ≤ 500 %`` per factor, ``k1 ≤ 100``
PRODUCTION_BOUNDS: dict[str, Any] = {"nu_cap": None, "omega_max": 5.0, "k1_bounds": (0.3, 100.0)}
_TOL_T = 1e-9
_TOL_ACTIVE = 1e-9
_TOL_FEAS = 1e-9
#: the repository root (a fit spec names its snapshot relative to it)
_REPO_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BreakEvenFitConfig:
    """Settings of the P1 marking calibration (module docstring).

    Targets: ``pillars`` (default 3M–10Y), ``mat_min`` (drop pillars below; 3M),
    ``smooth_breakeven`` (SmoothBreakEven, on), ``sabrw_power`` / ``atf_ref`` (the convexity
    rescaling of step 0; 1 / 0.3).  Step 2: ``k2`` fixed (0.2) unless ``k2_bounds`` is set — then
    ``k2`` is fitted inside it (``k2_grid`` points of a geometric grid — :data:`K2_GRID_DEFAULT`
    when ``None`` —, the fixed ``k2`` among them when inside, then a bounded refinement; each
    candidate runs the ``k1`` search with ``k1 > k2 + k1_min_gap``; owner's decision of
    2026-09-26, SPEC §15 Part 3) —, ``k1_bounds``
    (``k1_bounds[0]`` must exceed ``k2 + k1_min_gap``), ``k1_grid`` points of the coarse geometric
    grid,
    ``skew_mode`` (:data:`SKEW_MODE_CHOICES`; ``"auto"`` → two-point in marking mode, soft in
    historical mode), ``skew_pillars`` the constrained maturities (default ``(T_s, T_l)`` =
    ``(1, 5)``; any increasing list — the owner's decision of 2026-09-26 extends the band to the
    short end, SPEC §15 Part 3), ``skew_eps`` a float (every point) or one value per skew pillar
    (normalised to a tuple), ``skew_weight`` (soft mode only; 10), ``radicand_floor``
    (the step-0 guard ``c``),
    ``weights_covar`` (:data:`WEIGHT_KINDS` or one weight per fitted pillar), ``term_structure``
    (``f(t)`` of the leverage integrals), ``kernel_curve`` (the forward variance of the naked
    kernels ``A``, ``J``: ``None`` the variance-swap curve the caller passes, the M7 engine;
    ``"atmf"`` the ATMF one, the desk note's ``ξ̂`` — option of 2026-09-27, SPEC §15 Part 3),
    ``sigma0_maturity`` (the maturity of the ATMF vol taken as ``σ_0``: ``None`` 1M,
    :data:`~volsto.calibration.targets.SIGMA0_MATURITY`; the desk note takes 3M, its §6).
    Step 3: ``weights_volvar``, ``nu_cap`` (config cap of
    both minimisations, 3.5, warning when bound; ``None``: no ``ν`` cap, the first fit's box then
    ``|λ_i| ≤ 0.99 omega_max`` and the second's ``|λ_i| ≤ ω_i ≤ omega_max`` — the desk note's
    bounds, :data:`PRODUCTION_BOUNDS`), ``chi_bounds``, ``omega_max``, ``volvar_target``
    (``None``: rebuilt from the achieved covariance, the M7 rule; ``"direct"``: the targets'
    ``VoV_BE²``, the note's step 2), ``engine`` (``None``: the first-order break-evens;
    ``"mlp"``: the desk note's most-likely-path closed forms, :class:`MlpMaps`, with
    ``kernel_curve="atmf"``), ``step0`` (``None``: step 0 reads the surface's ATM derivatives;
    ``"sabrw"``: the date's stored SABRW fits, which the caller passes as the ``step0`` source —
    :func:`volsto.market.loaders.load_step0_source`, or
    :class:`~volsto.calibration.targets.ShiftedTriplets` on a moved surface —, a marking fit
    without one raising).  Diagnostics:
    ``stage3_tolerance`` (the stage-3 assertion, 10%), ``rho12_flag`` (``|ρ12|`` above which the
    two-factor structure is flagged as collapsing, 0.9).  Quadrature orders
    ``n_quad`` / ``n_inner`` (pillar kernels) and ``n_ts`` / ``n_quad_ts`` / ``n_inner_ts`` (the
    term-structure integrals, ``t = T u^p``)."""

    pillars: tuple[float, ...] = DEFAULT_TARGET_PILLARS
    mat_min: float = DEFAULT_MAT_MIN
    smooth_breakeven: bool = True
    sabrw_power: float = DEFAULT_SABRW_POWER
    atf_ref: float = DEFAULT_ATF_REF
    k2: float = DEFAULT_K2
    k2_bounds: tuple[float, float] | None = None
    k2_grid: int | None = None
    k1_bounds: tuple[float, float] = (0.3, 20.0)
    k1_min_gap: float = 0.05
    k1_grid: int = 25
    radicand_floor: float | None = DEFAULT_RADICAND_FLOOR
    skew_mode: str = "auto"
    skew_eps: float | tuple[float, ...] = DEFAULT_SKEW_EPS
    skew_pillars: tuple[float, ...] = DEFAULT_SKEW_PILLARS
    skew_weight: float = DEFAULT_SKEW_WEIGHT
    weights_covar: str | tuple[float, ...] = "relative"
    weights_volvar: str | tuple[float, ...] = "relative"
    term_structure: str = "atmf"
    kernel_curve: str | None = None
    sigma0_maturity: float | None = None
    volvar_target: str | None = None
    engine: str | None = None
    step0: str | None = None
    nu_cap: float | None = DEFAULT_NU_CAP
    chi_bounds: tuple[float, float] = (-0.99, 0.99)
    omega_max: float = 20.0
    stage3_tolerance: float = DEFAULT_STAGE3_TOLERANCE
    rho12_flag: float = RHO12_COLLAPSE
    n_quad: int = 64
    n_inner: int = 32
    n_ts: int = 64
    n_quad_ts: int = 32
    n_inner_ts: int = 24

    OMIT_WHEN_NONE: ClassVar[frozenset[str]] = frozenset(
        {
            "k2_bounds",
            "k2_grid",
            "kernel_curve",
            "sigma0_maturity",
            "volvar_target",
            "engine",
            "step0",
        }
    )
    """Options left out of the config's mapping while unset, so a config without them maps — and
    hashes (the backtest's config hash includes the resolved fit config) — exactly as before they
    existed (:func:`volsto.config.to_mapping`)."""

    def __post_init__(self) -> None:
        if not self.pillars or any(t <= 0 for t in self.pillars):
            raise ValueError("pillars must be positive maturities")
        if self.mat_min < 0:
            raise ValueError("mat_min must be non-negative")
        if self.atf_ref <= 0:
            raise ValueError("atf_ref must be positive")
        if self.k2 <= 0:
            raise ValueError("k2 must be positive")
        lo, hi = self.k1_bounds
        if not lo < hi:
            raise ValueError("k1_bounds must be increasing")
        if lo < self.k2 + self.k1_min_gap:
            raise ValueError(
                f"k1_bounds[0] = {lo:g} must exceed k2 + k1_min_gap = {self.k2 + self.k1_min_gap:g}"
            )
        if self.k1_grid < 3:
            raise ValueError("k1_grid must be at least 3")
        if self.k2_bounds is not None:
            k2lo, k2hi = (float(x) for x in self.k2_bounds)
            if not 0.0 < k2lo < k2hi:
                raise ValueError("k2_bounds must be increasing positive values")
            if not k2hi + self.k1_min_gap < hi:
                raise ValueError(
                    f"k2_bounds[1] + k1_min_gap = {k2hi + self.k1_min_gap:g} must stay below "
                    f"k1_bounds[1] = {hi:g}"
                )
            if self.k2_grid is not None and self.k2_grid < 3:
                raise ValueError("k2_grid must be at least 3")
            object.__setattr__(self, "k2_bounds", (k2lo, k2hi))
        elif self.k2_grid is not None:
            raise ValueError("k2_grid needs k2_bounds (k2 is fixed without them)")
        if self.skew_mode not in SKEW_MODE_CHOICES:
            raise ValueError(f"skew_mode must be one of {SKEW_MODE_CHOICES}")
        if self.radicand_floor is not None and not 0.0 <= self.radicand_floor < 1.0:
            raise ValueError("radicand_floor must be in [0, 1) or None")
        if not 0.0 < self.stage3_tolerance < 1.0:
            raise ValueError("stage3_tolerance must be in (0, 1)")
        if not 0.0 < self.rho12_flag <= 1.0:
            raise ValueError("rho12_flag must be in (0, 1]")
        sp = tuple(float(t) for t in self.skew_pillars)
        if not sp or sp[0] <= 0.0 or any(b <= a for a, b in itertools.pairwise(sp)):
            raise ValueError("skew_pillars must be increasing positive maturities")
        # stored as given: the config's mapping (hashed by the backtest) must not change
        eps = self.skew_eps
        vals = (
            (float(eps),) * len(sp)
            if isinstance(eps, int | float)
            else tuple(float(e) for e in eps)
        )
        if len(vals) != len(sp) or not all(math.isfinite(e) and e >= 0.0 for e in vals):
            raise ValueError(
                "skew_eps must be a non-negative float or one non-negative value per skew pillar"
            )
        object.__setattr__(self, "skew_eps", vals)
        if not (math.isfinite(self.skew_weight) and self.skew_weight >= 0.0):
            raise ValueError("skew_weight must be a finite non-negative number")
        for w in (self.weights_covar, self.weights_volvar):
            if isinstance(w, str):
                if w not in WEIGHT_KINDS:
                    raise ValueError(f"weights must be one of {WEIGHT_KINDS} or per-pillar values")
            elif not all(math.isfinite(x) and x > 0 for x in w):
                raise ValueError("explicit weights must be finite and positive")
        if self.term_structure not in TERM_STRUCTURE_KINDS:
            raise ValueError(f"term_structure must be one of {TERM_STRUCTURE_KINDS}")
        if self.kernel_curve is not None and self.kernel_curve not in KERNEL_CURVES:
            raise ValueError(f"kernel_curve must be None or one of {KERNEL_CURVES}")
        if self.sigma0_maturity is not None and not self.sigma0_maturity > 0.0:
            raise ValueError("sigma0_maturity must be positive (or None: the 1M ATMF vol)")
        if self.volvar_target is not None and self.volvar_target not in VOLVAR_TARGETS:
            raise ValueError(f"volvar_target must be None or one of {VOLVAR_TARGETS}")
        if self.engine is not None and self.engine not in FIT_ENGINES:
            raise ValueError(f"engine must be None or one of {FIT_ENGINES}")
        if self.engine == "mlp" and self.kernel_curve != "atmf":
            raise ValueError(
                "engine='mlp' (the desk note's closed forms) reads the ATMF forward variance: set "
                "kernel_curve='atmf'"
            )
        if self.step0 is not None and self.step0 not in STEP0_SOURCES:
            raise ValueError(f"step0 must be None or one of {STEP0_SOURCES}")
        if self.nu_cap is not None and self.nu_cap <= 0:
            raise ValueError("nu_cap must be positive (or None: no nu cap, the desk note's bounds)")
        clo, chi_ = self.chi_bounds
        if not -1.0 <= clo < chi_ <= 1.0:
            raise ValueError("chi_bounds must be increasing inside [-1, 1]")
        if not self.omega_max > 0.0:
            raise ValueError("omega_max must be positive")
        if self.nu_cap is not None and self.omega_max <= 2.0 * self.nu_cap:
            raise ValueError(
                "omega_max must exceed 2 nu_cap (the first fit allows |lambda_i| <= 2 nu_cap)"
            )
        if min(self.n_quad, self.n_inner, self.n_ts, self.n_quad_ts, self.n_inner_ts) < 4:
            raise ValueError("quadrature orders must be at least 4")

    @property
    def eps_pair(self) -> tuple[float, float]:
        """``(eps_s, eps_l)`` of the two-point constraint (a two-pillar band only)."""
        e = self.skew_eps
        assert isinstance(e, tuple)
        if len(e) != 2:
            raise ValueError(f"eps_pair needs a two-point band; this one has {len(e)} points")
        return float(e[0]), float(e[1])


#: the desk's marking fit (SPEC §15 Part 3, the owner's default of 2026-09-27): step 0 from the
#: date's stored SABRW fits (``step0="sabrw"``: the caller passes them), ``k2`` fitted inside
#: ``(0.05, 5)``, the naked kernels on the ATMF forward variance, ``σ_0`` the 3M ATMF vol and the
#: desk note's bounds (:data:`PRODUCTION_BOUNDS`); the pipelines use it with their ``skew_eps``
#: (the backtest's ``marking.fit``, the viewer grid's ``marking.fit``)
DESK_FIT = BreakEvenFitConfig(
    k2_bounds=(0.05, 5.0),
    kernel_curve="atmf",
    sigma0_maturity=0.25,
    step0="sabrw",
    **PRODUCTION_BOUNDS,
)
#: :data:`DESK_FIT` for the surfaces the hedger simulates (quadratic smiles per pillar, no wings):
#: the first-order engine, which reads the smile at the money only
DESK_FIT_STENCIL = replace(DESK_FIT, engine=None)
#: the named marking fits of the pipelines' configs: ``"m7"`` the M7 fit (step 0 from the
#: surface, ``k2`` 0.2, ν cap 3.5, the variance-swap kernels), ``"desk"`` :data:`DESK_FIT`
FIT_PRESETS: dict[str, BreakEvenFitConfig] = {"m7": BreakEvenFitConfig(), "desk": DESK_FIT}


def fit_preset(name: str, **changes: Any) -> BreakEvenFitConfig:
    """The named marking fit (:data:`FIT_PRESETS`) with ``changes`` (``skew_eps``, pillars…)."""
    if name not in FIT_PRESETS:
        raise ValueError(f"fit preset {name!r} must be one of {sorted(FIT_PRESETS)}")
    return replace(FIT_PRESETS[name], **changes)


def resolve_skew_mode(cfg: BreakEvenFitConfig, targets: TargetSet) -> tuple[str, str]:
    """The skew mode of a fit and a note: an explicit ``cfg.skew_mode`` is kept; ``"auto"``
    gives ``"twopoint"`` on marking targets and ``"soft"`` on historical targets (owner, report
    decision v: the soft all-pillar penalty is the historical-mode default, since the two-point
    configuration leaves ρ_SX1 unidentified on the synthetic recovery)."""
    if cfg.skew_mode != "auto":
        return cfg.skew_mode, ""
    mode = "twopoint" if targets.mode == "marking" else "soft"
    return mode, f"skew_mode 'auto' resolved to '{mode}' ({targets.mode} mode)"


def _resolved(cfg: BreakEvenFitConfig, targets: TargetSet) -> tuple[BreakEvenFitConfig, str]:
    mode, note = resolve_skew_mode(cfg, targets)
    return (cfg if mode == cfg.skew_mode else replace(cfg, skew_mode=mode)), note


# --------------------------------------------------------------------------------------------
# kernels along k1: the quadrature of ``kernels`` precomputed once per pillar
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PillarQuad:
    """The ``k``-independent part of :func:`~volsto.analytics.breakeven.kernels` for one
    maturity of the naked model (same Gauss-Legendre nodes, so ``A(k)`` and ``J(k)`` reproduce
    ``kernels((k1, k2), xi0, T).A / .J`` to round-off — ``test_affine_maps_match_engine``)."""

    T: float
    t: FloatArray
    w: FloatArray
    v: FloatArray
    W_T: float
    sigma_hat: float
    U: FloatArray  # (n_quad, n_inner) inner nodes
    S: FloatArray  # (n_quad, n_inner) inner weights times sqrt(xi0(u))

    def A(self, k: float) -> float:
        """``A(k) = ∫₀ᵀ ξ₀ e^{−kt} dt / W_T`` (eq. 7.38)."""
        return float(np.sum(self.w * self.v * np.exp(-k * self.t)) / self.W_T)

    def c(self, k: float) -> FloatArray:
        """``c̃(t_j; k) = ∫₀^{t_j} sqrt(ξ₀^u) e^{−k(t_j − u)} du`` at the nodes."""
        return np.asarray(
            np.sum(self.S * np.exp(-k * (self.t[:, None] - self.U)), axis=1), dtype=np.float64
        )

    def J(self, k: float) -> float:
        """``J(k) = ∫₀ᵀ ξ₀ c̃(t; k) dt / (2 σ̂³ T²)`` (eq. 8.54)."""
        return float(np.sum(self.w * self.v * self.c(k)) / (2.0 * self.sigma_hat**3 * self.T**2))


def pillar_quad(xi0: ForwardVarianceCurve, T: float, *, n_quad: int, n_inner: int) -> PillarQuad:
    """Precompute :class:`PillarQuad` for ``T`` on ``xi0`` (naked model)."""
    if T <= 0:
        raise ValueError("T must be positive")
    t, w = _gl(0.0, T, n_quad)
    v = np.asarray(xi0.xi0(t), dtype=np.float64)
    W_T = float(np.sum(w * v))
    U = np.empty((t.size, n_inner))
    S = np.empty((t.size, n_inner))
    for j, tj in enumerate(t):
        u, wu = _gl(0.0, float(tj), n_inner)
        U[j] = u
        S[j] = wu * np.sqrt(np.asarray(xi0.xi0(u), dtype=np.float64))
    return PillarQuad(float(T), t, w, v, W_T, float(np.sqrt(W_T / T)), U, S)


@dataclass(frozen=True)
class AffineMaps:
    """The naked affine maps at fixed ``(k1, k2)``: ``svc(λ) = a @ λ`` (``½ atf (λ1 A1 + λ2
    A2)``, eq. 7.38) and ``skew(λ) = j @ λ`` (``λ1 J1 + λ2 J2``, eq. 8.54), ``a, j`` of shape
    ``(n, 2)``; ``A``, ``J`` the raw kernels, ``sigma_hat`` the kernel's VS vols."""

    k1: float
    k2: float
    T: FloatArray
    atf: FloatArray
    a: FloatArray
    j: FloatArray
    A: FloatArray
    J: FloatArray
    sigma_hat: FloatArray

    def svc(self, lam: FloatArray) -> FloatArray:
        return np.asarray(self.a @ lam, dtype=np.float64)

    def skew(self, lam: FloatArray) -> FloatArray:
        return np.asarray(self.j @ lam, dtype=np.float64)


def _maps(
    k1: float,
    k2: float,
    T: FloatArray,
    atf: FloatArray,
    A: FloatArray,
    J: FloatArray,
    sig: FloatArray,
) -> AffineMaps:
    # the SV skew is eq. 8.54 at the kernel's order-zero VS vol; the sensitivities take the
    # market ATMF vol as prefactor (the gate-validated convention of analytics/breakeven.py)
    return AffineMaps(float(k1), float(k2), T, atf, 0.5 * atf[:, None] * A, J.copy(), A, J, sig)


def affine_maps(quads: Sequence[PillarQuad], atf: FloatArray, k1: float, k2: float) -> AffineMaps:
    """:class:`AffineMaps` from the precomputed quadratures and the market ATMF vols."""
    A = np.array([[q.A(k1), q.A(k2)] for q in quads])
    J = np.array([[q.J(k1), q.J(k2)] for q in quads])
    sig = np.array([q.sigma_hat for q in quads])
    T = np.array([q.T for q in quads])
    return _maps(k1, k2, T, atf, A, J, sig)


def affine_maps_from_kernels(
    ks: tuple[float, float], kerns: Sequence[Kernels], atf: FloatArray
) -> AffineMaps:
    """The same maps read from :func:`~volsto.analytics.breakeven.kernels` objects (the slow
    reference route of the tests)."""
    A = np.array([k.A for k in kerns])
    J = np.array([k.J for k in kerns])
    sig = np.array([k.sigma_hat for k in kerns])
    T = np.array([k.T for k in kerns])
    return _maps(ks[0], ks[1], T, atf, A, J, sig)


# --------------------------------------------------------------------------------------------
# the leverage integrals (1/T) ∫ f(t) (S_t − λ·J_t) dt
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TermStructureBank:
    """The ``k``-independent quadrature of ``(1/T) ∫₀ᵀ f(t) X_t dt`` per fitted pillar, nodes
    ``t = T u^p`` (shape ``(n, m)``) and weights ``p u^{p−1} w_u f(t)`` (``f`` included).
    ``source == "surface"``: the market skew at the nodes and the eq. 8.54 quadrature of ``J(t;
    k)`` at every node (vectorised :class:`PillarQuad`); ``source == "pillars"`` (historical mode):
    ``Q`` with ``(1/T) ∫ f X = Q @ X(pillars)`` for ``X`` interpolated linearly in ``t`` between
    the pillars, flat outside."""

    T: FloatArray
    t: FloatArray
    weight: FloatArray
    f: FloatArray
    source: str
    skew_market_nodes: FloatArray
    tq: FloatArray
    wv: FloatArray
    U: FloatArray
    S: FloatArray
    denom: FloatArray
    Q: FloatArray
    skew_pillars: FloatArray
    kind: str

    def J_nodes(self, k: float) -> FloatArray:
        """``J(t; k)`` at every node, shape ``(n, m)`` (surface source)."""
        c = np.sum(self.S * np.exp(-k * (self.tq[:, :, None] - self.U)), axis=2)
        return np.asarray(
            (np.sum(self.wv * c, axis=1) / self.denom).reshape(self.t.shape), dtype=np.float64
        )

    def integral(self, k: float, J_pillars: FloatArray) -> FloatArray:
        """``(1/T) ∫₀ᵀ f(t) J(t; k) dt`` per pillar (``J_pillars`` used by the pillars source)."""
        if self.source == "pillars":
            return np.asarray(self.Q @ J_pillars, dtype=np.float64)
        return np.asarray(np.sum(self.weight * self.J_nodes(k), axis=1), dtype=np.float64)

    @property
    def I_market(self) -> FloatArray:
        """``(1/T) ∫₀ᵀ f(t) S^mkt_t dt`` per pillar."""
        if self.source == "pillars":
            return np.asarray(self.Q @ self.skew_pillars, dtype=np.float64)
        return np.asarray(np.sum(self.weight * self.skew_market_nodes, axis=1), dtype=np.float64)


def ts_substitution_power(targets: TargetSet) -> float:
    """``p = clip(1/(1 − γ), 2, 4)`` of the substitution ``t = T u^p`` with ``γ = −d ln|S|/d ln t``
    the market skew's short-end exponent between ``t = 1e-4`` and ``1e-3`` (surface source; the
    integrand in ``u`` is then smooth); 2 for the pillars source (flat residual below the first
    pillar)."""
    if targets.term_structure_source != "surface":
        return 2.0
    t = np.array([1e-4, 1e-3])
    S = np.abs(targets.market_skew(t))
    gamma = float(-np.log(S[1] / S[0]) / np.log(t[1] / t[0]))
    return float(np.clip(1.0 / (1.0 - min(gamma, 0.9)), 2.0, 4.0))


def term_structure_bank(
    targets: TargetSet,
    T: FloatArray,
    xi0: ForwardVarianceCurve,
    *,
    kind: str,
    n_ts: int,
    n_quad: int,
    n_inner: int,
    skew_pillars: FloatArray | None = None,
) -> TermStructureBank:
    """Build :class:`TermStructureBank` (``f``: ``"flat"`` 1, ``"atmf"`` ``σ²(t)/(σ̂_t σ̂_T)`` on
    the targets' ATMF curve, ``"vs"`` the same on ``xi0``)."""
    if kind not in TERM_STRUCTURE_KINDS:
        raise ValueError(f"kind must be one of {TERM_STRUCTURE_KINDS}")
    T = np.asarray(T, dtype=np.float64)
    u, wu = np.polynomial.legendre.leggauss(n_ts)
    u = 0.5 * (u + 1.0)
    wu = 0.5 * wu
    p = ts_substitution_power(targets)
    t = T[:, None] * (u**p)[None, :]
    weight = np.broadcast_to(p * u ** (p - 1) * wu, t.shape)
    if kind == "flat":
        f = np.ones_like(t)
    else:
        curve = targets.atmf_curve(max(float(T.max()), 1.5 / 12.0)) if kind == "atmf" else xi0
        sig_t = np.sqrt(np.asarray(curve.total_variance(t), dtype=np.float64) / t)
        sig_T = np.sqrt(np.asarray(curve.total_variance(T), dtype=np.float64) / T)
        f = np.asarray(curve.xi0(t), dtype=np.float64) / (sig_t * sig_T[:, None])
    wf = np.asarray(weight * f, dtype=np.float64)
    empty = np.zeros((0,))
    if targets.term_structure_source != "surface":
        sp = np.asarray(skew_pillars, dtype=np.float64) if skew_pillars is not None else empty
        if sp.shape != T.shape:
            raise ValueError("the pillars source needs the market skew at every fitted pillar")
        P = np.stack([np.interp(t, T, np.eye(T.size)[i]) for i in range(T.size)], axis=-1)
        Q = np.sum(wf[:, :, None] * P, axis=1)
        return TermStructureBank(
            T, t, wf, f, "pillars", empty, empty, empty, empty, empty, empty, Q, sp, kind
        )
    tau = t.ravel()
    xq, wq = np.polynomial.legendre.leggauss(n_quad)
    xq, wq = 0.5 * (xq + 1.0), 0.5 * wq
    xi, wi = np.polynomial.legendre.leggauss(n_inner)
    xi, wi = 0.5 * (xi + 1.0), 0.5 * wi
    tq = tau[:, None] * xq[None, :]
    w = tau[:, None] * wq[None, :]
    v = np.asarray(xi0.xi0(tq), dtype=np.float64)
    W = np.sum(w * v, axis=1)
    sig = np.sqrt(W / tau)
    U = tq[:, :, None] * xi[None, None, :]
    S = (
        tq[:, :, None]
        * wi[None, None, :]
        * np.sqrt(np.asarray(xi0.xi0(U.ravel()), dtype=np.float64)).reshape(U.shape)
    )
    denom = 2.0 * sig**3 * tau * tau
    skew_nodes = targets.market_skew(tau).reshape(t.shape)
    return TermStructureBank(
        T, t, wf, f, "surface", skew_nodes, tq, w * v, U, S, denom, np.zeros((0, 0)), empty, kind
    )


@dataclass(frozen=True)
class P1Maps:
    """The P1 break-evens at fixed ``(k1, k2)`` (module docstring), affine in ``λ``:
    ``svc(λ) = a @ λ + b`` with ``a = ½ atf A − σ_0 (J + I)`` and ``b = σ_0 (S + I^mkt)``;
    ``svc_naked(λ) = ½ atf λ·A``; ``sensi_spot = svc − svc_naked``; ``skew_naked(λ) = λ·J``."""

    naked: AffineMaps
    I: FloatArray  # (n, 2)
    skew_market: FloatArray
    I_market: FloatArray
    sigma_0: float

    @property
    def T(self) -> FloatArray:
        return self.naked.T

    @property
    def k1(self) -> float:
        return self.naked.k1

    def rows(self) -> tuple[FloatArray, FloatArray]:
        a = self.naked.a - self.sigma_0 * (self.naked.j + self.I)
        b = self.sigma_0 * (self.skew_market + self.I_market)
        return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)

    def svc(self, lam: FloatArray) -> FloatArray:
        a, b = self.rows()
        return np.asarray(a @ lam + b, dtype=np.float64)

    def svc_naked(self, lam: FloatArray) -> FloatArray:
        return self.naked.svc(lam)

    def skew_naked(self, lam: FloatArray) -> FloatArray:
        return self.naked.skew(lam)

    def sensi_spot(self, lam: FloatArray) -> FloatArray:
        """The leverage spot sensitivity ``σ_0 [(S − λ·J) + (I^mkt − λ·I)]``."""
        return np.asarray(self.svc(lam) - self.svc_naked(lam), dtype=np.float64)

    def ssr_first_order(self, lam: FloatArray) -> FloatArray:
        """``SpotVolCovar_P1 / (σ_0 S^mkt)``: the first-order SSR of the P1 model."""
        return np.asarray(self.svc(lam) / (self.sigma_0 * self.skew_market), dtype=np.float64)


@dataclass(frozen=True)
class MlpMaps(P1Maps):
    """The step-2 maps of ``engine="mlp"`` (SPEC §15 Part 3): SpotVolCovar, its spot and naked
    parts and the first-order SSR from the desk note's closed forms at any ``λ``
    (:class:`volsto.analytics.p1_mlp.MlpPillar`: ``SpotVolCovar = atf (SensiSpot + λ1 G_X + λ2
    G_Y)``); the naked skew stays ``λ·J`` (the note's §8); ``naked.A`` is ``2 G`` at ``lam_star``,
    the solution, so step 3's ``SensiX_i = ½ ω_i A_i atf`` is the note's ``ω_i G_i atf``."""

    pillars: tuple[Any, ...] = ()
    atf_mlp: FloatArray = field(default_factory=lambda: np.zeros(0))
    lam_star: FloatArray = field(default_factory=lambda: np.zeros(2))

    def values(self, lam: FloatArray) -> tuple[FloatArray, FloatArray]:
        """``(SensiSpot, G)`` per pillar at ``λ``: absolute SensiSpot and ``G`` of shape (n, 2)."""
        v = np.array([p.evaluate(lam) for p in self.pillars], dtype=np.float64)
        return np.asarray(v[:, 0] * self.atf_mlp, dtype=np.float64), v[:, 1:]

    def svc(self, lam: FloatArray) -> FloatArray:
        spot, G = self.values(lam)
        return np.asarray(spot + self.atf_mlp * (G @ lam), dtype=np.float64)

    def svc_naked(self, lam: FloatArray) -> FloatArray:
        _, G = self.values(lam)
        return np.asarray(self.atf_mlp * (G @ lam), dtype=np.float64)

    def sensi_spot(self, lam: FloatArray) -> FloatArray:
        return self.values(lam)[0]


def _mlp_maps(fo: P1Maps, pillars: tuple[Any, ...], lam: FloatArray) -> MlpMaps:
    atf = np.asarray(fo.naked.atf, dtype=np.float64)
    G = np.array([p.evaluate(lam)[1:] for p in pillars], dtype=np.float64)
    A_eff = 2.0 * G
    nk = fo.naked
    naked = AffineMaps(
        nk.k1, nk.k2, nk.T, atf, 0.5 * atf[:, None] * A_eff, nk.j, A_eff, nk.J, nk.sigma_hat
    )
    return MlpMaps(
        naked, fo.I, fo.skew_market, fo.I_market, fo.sigma_0, pillars, atf, np.asarray(lam)
    )


# --------------------------------------------------------------------------------------------
# the inner problem: exact 2-D QP
# --------------------------------------------------------------------------------------------


def _qp2(
    H: FloatArray, g: FloatArray, G: FloatArray, h: FloatArray
) -> tuple[FloatArray, tuple[int, ...], bool]:
    """``argmin ½ xᵀHx − gᵀx`` over ``G x ≤ h`` for ``x ∈ R²`` by candidate enumeration (convex
    QP: the optimum is the unconstrained point, the equality optimum on one edge or a vertex).
    Returns ``(x, active constraint indices, feasible)``; when no candidate is feasible the
    unconstrained solution comes back with ``feasible=False``."""
    scale = 1.0 + np.abs(h)

    def feasible(x: FloatArray) -> bool:
        return bool(np.all(G @ x <= h + _TOL_ACTIVE * scale))

    def obj(x: FloatArray) -> float:
        return float(0.5 * x @ H @ x - g @ x)

    def active(x: FloatArray) -> tuple[int, ...]:
        return tuple(int(i) for i in np.flatnonzero(np.abs(G @ x - h) <= 1e-7 * scale))

    try:
        x0 = np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        x0 = np.linalg.lstsq(H, g, rcond=None)[0]
    cands: list[tuple[float, FloatArray]] = []
    if feasible(x0):
        cands.append((obj(x0), x0))
    m = G.shape[0]
    for i in range(m):
        K = np.zeros((3, 3))
        K[:2, :2] = H
        K[:2, 2] = G[i]
        K[2, :2] = G[i]
        rhs = np.array([g[0], g[1], h[i]])
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            continue
        x = sol[:2]
        if feasible(x):
            cands.append((obj(x), x))
    for i in range(m):
        for j in range(i + 1, m):
            Gij = G[[i, j]]
            if abs(np.linalg.det(Gij)) <= 1e-14 * (np.abs(Gij).max() ** 2 + 1e-300):
                continue
            x = np.linalg.solve(Gij, h[[i, j]])
            if feasible(x):
                cands.append((obj(x), x))
    if not cands:
        return np.asarray(x0, dtype=np.float64), (), False
    best = min(cands, key=lambda c: c[0])[1]
    return np.asarray(best, dtype=np.float64), active(best), True


def _min_violation(Gn: FloatArray, hn: FloatArray, relax: NDArray[np.bool_]) -> float:
    """The smallest uniform relaxation ``δ ≥ 0`` of the rows flagged ``relax`` with ``{λ : Gn λ ≤
    hn + δ relax}`` non-empty (a linear programme; ``Gn``, ``hn`` normalised rows; the other rows
    — the ν box — stay hard, so the relaxed point is always admissible for step 3)."""
    m = Gn.shape[0]
    A = np.hstack((Gn, -relax.astype(np.float64)[:, None]))
    res = linprog(
        c=np.array([0.0, 0.0, 1.0]),
        A_ub=A,
        b_ub=hn,
        bounds=[(None, None), (None, None), (0.0, None)],
        method="highs",
    )
    if not res.success or m == 0:
        raise RuntimeError(f"least-violation programme failed: {res.message}")
    return float(np.asarray(res.x, dtype=np.float64)[2])


@dataclass(frozen=True)
class SkewConstraint:
    """One point of the two-point constraint: the requested maturity, the fitted pillar it is
    applied at, its index, the tolerance and the market skew."""

    requested: float
    T: float
    index: int
    eps: float
    skew_market: float
    name: str

    @property
    def relocated(self) -> bool:
        return abs(self.requested - self.T) > _TOL_T

    @property
    def bounds(self) -> tuple[float, float]:
        a, b = (1.0 - self.eps) * self.skew_market, (1.0 + self.eps) * self.skew_market
        return (min(a, b), max(a, b))

    def edge_factor(self, upper: bool) -> str:
        """The factor of ``Skew_SABR`` at the upper / lower bound of the interval."""
        a = (1.0 - self.eps) * self.skew_market
        at_a = (a >= (1.0 + self.eps) * self.skew_market) == upper
        return f"(1{'-' if at_a else '+'}{self.eps:g})"


def _skew_constraints(
    T: FloatArray, skew: FloatArray, cfg: BreakEvenFitConfig
) -> tuple[tuple[SkewConstraint, ...], list[str]]:
    notes: list[str] = []
    out = []
    n = len(cfg.skew_pillars)
    names = ("T_s", "T_l") if n == 2 else tuple(f"T_{i + 1}" for i in range(n))
    assert isinstance(cfg.skew_eps, tuple)
    for name, req, eps in zip(names, cfg.skew_pillars, cfg.skew_eps, strict=True):
        i = int(np.argmin(np.abs(T - req)))
        if abs(T[i] - req) > _TOL_T:
            notes.append(
                f"skew constraint {name} = {req:g}y is not a fitted pillar (pillars "
                f"{[round(float(x), 4) for x in T]}): applied at the nearest pillar {T[i]:g}y"
            )
        out.append(SkewConstraint(float(req), float(T[i]), i, float(eps), float(skew[i]), name))
    by_index: dict[int, list[str]] = {}
    for c in out:
        by_index.setdefault(c.index, []).append(c.name)
    for idx, on_pillar in by_index.items():
        if len(on_pillar) < 2:
            continue
        if n == 2:
            notes.append(
                f"both skew constraints fall on the pillar {out[0].T:g}y: one point constrained"
            )
        else:
            notes.append(
                f"skew constraints {on_pillar} fall on the pillar {float(T[idx]):g}y: the "
                "intersection of their bands applies"
            )
    return tuple(out), notes


@dataclass(frozen=True)
class InnerSolution:
    """The inner problem at one ``k1``: ``λ``, the objective (normalised units), the active
    constraint labels, feasibility and the least violation (0 when feasible), the
    unconstrained-LS covariance of ``λ`` (historical mode) and the maps."""

    k1: float
    lam: FloatArray
    objective: float
    objective_skew: float
    active: tuple[str, ...]
    feasible: bool
    violation: float
    cov: FloatArray
    maps: P1Maps


def _weights(spec: str | tuple[float, ...], target: FloatArray) -> FloatArray:
    if not isinstance(spec, str):
        w = np.asarray(spec, dtype=np.float64)
        if w.shape != target.shape:
            raise ValueError(f"explicit weights need one value per fitted pillar ({target.size})")
        return w
    if np.any(target == 0.0) or not np.all(np.isfinite(target)):
        raise ValueError("weights need finite, non-zero targets")
    if spec == "relative":
        return np.asarray(1.0 / target**2, dtype=np.float64)
    return np.full(target.shape, 1.0 / float(np.mean(target**2)))


@dataclass(frozen=True)
class _FirstProblem:
    quads: tuple[PillarQuad, ...]
    bank: TermStructureBank
    T: FloatArray
    atf: FloatArray
    sigma_0: float
    skew_market: FloatArray
    svc_target: FloatArray
    svc_se: FloatArray
    wc: FloatArray
    skew_mode: str
    skew_weight: float
    constraints: tuple[SkewConstraint, ...]
    nu_cap: float | None
    k2: float
    skew_band: FloatArray | None = None
    omega_max: float = 20.0
    engine: str | None = None
    surface: Any = field(default=None, compare=False, repr=False)
    cache: dict[float, P1Maps] = field(default_factory=dict, compare=False, repr=False)
    mlp_cache: dict[float, tuple[Any, ...]] = field(default_factory=dict, compare=False, repr=False)

    @property
    def band(self) -> FloatArray:
        """The skews the band (and the soft penalty) compare the naked skew with: the targets'
        ``band_skew`` when set (a step-0 source), else the market skew of the leverage term."""
        return self.skew_market if self.skew_band is None else self.skew_band

    def maps(self, k1: float) -> P1Maps:
        key = float(k1)
        mm = self.cache.get(key)
        if mm is None:
            naked = affine_maps(self.quads, self.atf, key, self.k2)
            I = np.stack(
                [
                    self.bank.integral(key, naked.J[:, 0]),
                    self.bank.integral(self.k2, naked.J[:, 1]),
                ],
                axis=1,
            )
            mm = P1Maps(naked, I, self.skew_market, self.bank.I_market, self.sigma_0)
            if len(self.cache) > 4096:
                self.cache.clear()
            self.cache[key] = mm
        return mm

    @property
    def has_noise(self) -> bool:
        se = self.svc_se
        return bool(se.size and np.all(np.isfinite(se)) and np.all(se > 0))

    def stacked(self, k1: float) -> tuple[P1Maps, FloatArray, FloatArray, int]:
        """``(maps, R, y, n_cov)``: residuals ``R λ − y``, the covariance rows ``sqrt(w) (a λ + b −
        target)`` then (soft mode) ``sqrt(skew_weight) (λ·J_i/S_i − 1)``."""
        mm = self.maps(k1)
        a, b = mm.rows()
        sw = np.sqrt(self.wc)
        R = sw[:, None] * a
        y = sw * (self.svc_target - b)
        if self.skew_mode == "soft":
            s = math.sqrt(self.skew_weight)
            R = np.vstack((R, s * mm.naked.j / self.band[:, None]))
            y = np.concatenate((y, np.full(self.T.size, s)))
        return mm, R, y, self.T.size

    def constraint_rows(self, mm: P1Maps) -> tuple[FloatArray, FloatArray, FloatArray, list[str]]:
        """``(G, h, scale, labels)``: the ν box and, in two-point mode, the two slabs."""
        if self.nu_cap is not None:
            rows = [np.array([1.0, 1.0]), np.array([1.0, -1.0]), np.array([-1.0, 1.0])]
            rows.append(np.array([-1.0, -1.0]))
            box = 2.0 * self.nu_cap
            labels = list(_BOX_LABELS)
        else:
            rows = [np.array([1.0, 0.0]), np.array([-1.0, 0.0])]
            rows += [np.array([0.0, 1.0]), np.array([0.0, -1.0])]
            box = LAMBDA_BOX_FRACTION * self.omega_max
            labels = list(_FACTOR_BOX_LABELS)
        h = [box] * 4
        scale = [box] * 4
        if self.skew_mode == "twopoint":
            for c in self.constraints:
                j = mm.naked.j[c.index]
                lo, hi = c.bounds
                sc = abs(c.skew_market)
                rows += [j, -j]
                h += [hi, -lo]
                scale += [sc, sc]
                labels += [
                    f"skew T={c.T:g} {c.edge_factor(True)}",
                    f"skew T={c.T:g} {c.edge_factor(False)}",
                ]
        return np.array(rows), np.array(h), np.array(scale), labels

    def solve(self, k1: float) -> InnerSolution:
        mm, R, y, n = self.stacked(k1)
        H = 2.0 * R.T @ R
        g = 2.0 * R.T @ y
        G, h, scale, labels = self.constraint_rows(mm)
        lam, active, feas = _qp2(H, g, G, h)
        violation = 0.0
        relax = np.array([not lab.startswith(("nu box", "lambda")) for lab in labels])
        if not feas:
            violation = _min_violation(G / scale[:, None], h / scale, relax)
            lam, active, feas2 = _qp2(H, g, G, h + (violation + _TOL_FEAS) * scale * relax)
            if not feas2:  # numerical: take the LP point
                lam = np.zeros(2)
        if self.engine == "mlp":
            h_used = h if violation <= 0.0 else h + (violation + _TOL_FEAS) * scale * relax
            return _solve_mlp(self, float(k1), mm, lam, G, h_used, scale, labels, violation)
        r = R @ lam - y
        o_skew = float(r[n:] @ r[n:]) if r.size > n else 0.0
        cov = (
            _sandwich(R[:n], np.diag(self.wc * self.svc_se**2))
            if self.has_noise
            else np.full((2, 2), np.nan)
        )
        return InnerSolution(
            float(k1),
            lam,
            float(r @ r),
            o_skew,
            tuple(labels[i] for i in active),
            violation <= 0.0,
            violation,
            np.asarray(cov, dtype=np.float64),
            mm,
        )


def _mlp_pillars(prob: _FirstProblem, k1: float) -> tuple[Any, ...]:
    key = float(k1)
    pl = prob.mlp_cache.get(key)
    if pl is None:
        pl = tuple(
            MlpPillar(prob.surface, float(T), key, prob.k2, sigma_0=prob.sigma_0, grid=MLP_FIT_GRID)
            for T in prob.T
        )
        if len(prob.mlp_cache) > 512:
            prob.mlp_cache.clear()
        prob.mlp_cache[key] = pl
    return pl


def _solve_mlp(
    prob: _FirstProblem,
    k1: float,
    fo: P1Maps,
    lam0: FloatArray,
    G: FloatArray,
    h: FloatArray,
    scale: FloatArray,
    labels: list[str],
    violation: float,
) -> InnerSolution:
    """Step 2's inner problem on the desk note's closed forms: SLSQP from the first-order QP
    solution ``lam0`` under the same linear constraints ``G λ ≤ h`` (the band and the box)."""
    pillars = _mlp_pillars(prob, k1)
    atf = np.asarray(fo.naked.atf, dtype=np.float64)
    sw = np.sqrt(prob.wc)
    soft = prob.skew_mode == "soft"
    s_w = math.sqrt(prob.skew_weight)

    def resid(lam: FloatArray) -> FloatArray:
        v = np.array([p.evaluate(lam) for p in pillars], dtype=np.float64)
        svc = atf * (v[:, 0] + v[:, 1:] @ lam)
        r = sw * (svc - prob.svc_target)
        if soft:
            r = np.concatenate((r, s_w * (fo.naked.j @ lam / prob.band - 1.0)))
        return np.asarray(r, dtype=np.float64)

    def obj(lam: FloatArray) -> float:
        r = resid(lam)
        return float(r @ r)

    cons: list[Any] = [{"type": "ineq", "fun": lambda x: h - G @ x, "jac": lambda x: -G}]
    res = minimize(
        obj,
        np.asarray(lam0, dtype=np.float64),
        method="SLSQP",
        constraints=cons,
        options={"ftol": 1e-14, "maxiter": 200},
    )
    lam = np.asarray(res.x, dtype=np.float64)
    if not np.all(G @ lam <= h + 1e-9 * (1.0 + np.abs(h))) or obj(lam) > obj(lam0):
        lam = np.asarray(lam0, dtype=np.float64)
    r = resid(lam)
    n = prob.T.size
    o_skew = float(r[n:] @ r[n:]) if r.size > n else 0.0
    act = np.flatnonzero(np.abs(G @ lam - h) <= 1e-7 * scale)
    return InnerSolution(
        float(k1),
        lam,
        float(r @ r),
        o_skew,
        tuple(labels[i] for i in act),
        violation <= 0.0,
        violation,
        np.full((2, 2), np.nan),
        _mlp_maps(fo, pillars, lam),
    )


def _sandwich(J: FloatArray, sigma: FloatArray) -> FloatArray:
    """``(JᵀJ)⁻¹ Jᵀ Σ J (JᵀJ)⁻¹`` (NaN when ``JᵀJ`` is singular)."""
    try:
        N = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return np.full((J.shape[1], J.shape[1]), np.nan)
    return np.asarray(N @ J.T @ sigma @ J @ N, dtype=np.float64)


def _select_pillars(targets: TargetSet, pillars: Sequence[float]) -> tuple[FloatArray, list[str]]:
    """Indices into ``targets.pillars`` of the requested pillars; the absent ones are noted."""
    tp = np.asarray(targets.pillars, dtype=np.float64)
    idx: list[int] = []
    dropped: list[float] = []
    for T in sorted(float(t) for t in pillars):
        i = int(np.argmin(np.abs(tp - T)))
        if abs(tp[i] - T) <= _TOL_T:
            idx.append(i)
        else:
            dropped.append(T)
    notes = []
    if dropped:
        notes.append(f"config pillars absent from the targets: {dropped}")
    if len(idx) < 2:
        raise ValueError(f"the fit needs at least two pillars, {len(idx)} available")
    return np.asarray(idx, dtype=np.int64), notes


def _engine_curve(
    targets: TargetSet, xi0: ForwardVarianceCurve, cfg: BreakEvenFitConfig
) -> ForwardVarianceCurve:
    """The forward variance the first-order engine's naked kernels ``A``, ``J`` read: the
    caller's ``xi0`` (``kernel_curve`` ``None``: the variance-swap curve, the M7 engine) or the
    ATMF one (``"atmf"``: :meth:`TargetSet.atmf_curve`, the desk note's ``ξ̂`` — measured against
    the stage-3 simulations, SPEC §15 Part 3)."""
    if cfg.kernel_curve is None:
        return xi0
    return targets.atmf_curve(float(np.max(np.asarray(targets.pillars, dtype=np.float64))))


def _first_problem(
    targets: TargetSet,
    cfg: BreakEvenFitConfig,
    xi0: ForwardVarianceCurve,
    svc_correction: FloatArray | None = None,
) -> tuple[_FirstProblem, list[str]]:
    cfg, mode_note = _resolved(cfg, targets)
    idx, notes = _select_pillars(targets, cfg.pillars)
    if mode_note:
        notes.append(mode_note)
    T = np.asarray(targets.pillars, dtype=np.float64)[idx]
    skew = np.asarray(targets.skew_target, dtype=np.float64)[idx]
    if np.any(skew == 0) or not np.all(np.isfinite(skew)):
        raise ValueError("the fit needs finite, non-zero market skews at every pillar")
    band = (
        None if targets.band_skew is None else np.asarray(targets.band_skew, dtype=np.float64)[idx]
    )
    if band is not None and (np.any(band == 0) or not np.all(np.isfinite(band))):
        raise ValueError("the fit needs finite, non-zero band skews at every pillar")
    quads = tuple(pillar_quad(xi0, float(t), n_quad=cfg.n_quad, n_inner=cfg.n_inner) for t in T)
    bank = term_structure_bank(
        targets,
        T,
        xi0,
        kind=cfg.term_structure,
        n_ts=cfg.n_ts,
        n_quad=cfg.n_quad_ts,
        n_inner=cfg.n_inner_ts,
        skew_pillars=skew,
    )
    if bank.source == "pillars":
        notes.append(
            "historical mode: the leverage integrals interpolate the skew residual (S - lambda.J) "
            "linearly between the pillars, flat outside"
        )
    constraints, cnotes = _skew_constraints(T, skew if band is None else band, cfg)
    if cfg.skew_mode == "twopoint":
        notes += cnotes
    svc = np.asarray(targets.spot_vol_covar, dtype=np.float64)[idx]
    if svc_correction is not None:
        corr = np.asarray(svc_correction, dtype=np.float64)
        if corr.shape != svc.shape or np.any(~np.isfinite(corr)) or np.any(corr <= 0):
            raise ValueError("svc_correction needs one finite positive factor per fitted pillar")
        svc = svc / corr
        notes.append(
            "SpotVolCovar targets divided by the simulated/analytic factors "
            f"{np.round(corr, 4).tolist()} (iterate_against_simulation)"
        )
    svc_se = np.asarray(targets.spot_vol_covar_se, dtype=np.float64)[idx]
    prob = _FirstProblem(
        quads,
        bank,
        T,
        np.asarray(targets.atf, dtype=np.float64)[idx],
        float(targets.sigma_0),
        skew,
        svc,
        svc_se,
        _weights(cfg.weights_covar, svc),
        cfg.skew_mode,
        float(cfg.skew_weight),
        constraints,
        None if cfg.nu_cap is None else float(cfg.nu_cap),
        float(cfg.k2),
        band,
        float(cfg.omega_max),
        cfg.engine,
        targets.surface,
    )
    if cfg.engine == "mlp":
        if targets.surface is None:
            raise ValueError(
                "engine='mlp' needs marking targets read from a surface (the note's closed forms "
                "read its smile)"
            )
        notes.append(
            "engine 'mlp': step 2 minimises the desk note's closed-form SpotVolCovar (most-likely "
            "path: SensiX, SensiY eq. 57 / appendix D, SensiSpot eq. 43) from the first-order QP "
            "solution at each k1, under the same band and box"
        )
    return prob, notes


def _k1_grid(cfg: BreakEvenFitConfig) -> FloatArray:
    lo, hi = cfg.k1_bounds
    return np.asarray(np.geomspace(lo, hi, cfg.k1_grid), dtype=np.float64)


def _rank(s: InnerSolution) -> tuple[float, float]:
    """Feasible solutions first (by objective), infeasible ones by violation."""
    return (0.0, s.objective) if s.feasible else (1.0, s.violation)


def _optimise_k1(
    prob: _FirstProblem, cfg: BreakEvenFitConfig
) -> tuple[InnerSolution, list[InnerSolution]]:
    """Coarse geometric grid in ``k1``, bounded scalar refinement around the best grid point."""
    grid = _k1_grid(cfg)
    sols = [prob.solve(float(k)) for k in grid]
    i = min(range(len(sols)), key=lambda j: _rank(sols[j]))
    best = sols[i]
    lo = float(grid[max(i - 1, 0)])
    hi = float(grid[min(i + 1, grid.size - 1)])
    if hi > lo:

        def score(k: float) -> float:
            s = prob.solve(float(k))
            return s.objective if s.feasible else 1e12 * (1.0 + s.violation)

        res = minimize_scalar(
            score, bounds=(lo, hi), method="bounded", options={"xatol": 1e-5 * hi}
        )
        cand = prob.solve(float(res.x))
        if _rank(cand) <= _rank(best):
            best = cand
    return best, sols


def k1_profile(
    targets: TargetSet,
    cfg: BreakEvenFitConfig,
    xi0: ForwardVarianceCurve,
    k1s: Sequence[float] | FloatArray | None = None,
) -> pd.DataFrame:
    """The first-minimisation objective along ``k1`` (default: the config's coarse grid)."""
    prob, _ = _first_problem(targets, cfg, _engine_curve(targets, xi0, cfg))
    grid = _k1_grid(cfg) if k1s is None else np.asarray(k1s, dtype=np.float64)
    return _profile_frame([prob.solve(float(k)) for k in grid])


def _profile_frame(sols: Sequence[InnerSolution]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "k1": [s.k1 for s in sols],
            "objective": [s.objective for s in sols],
            "lambda1": [float(s.lam[0]) for s in sols],
            "lambda2": [float(s.lam[1]) for s in sols],
            "nu_min": [0.5 * abs(float(np.sum(s.lam))) for s in sols],
            "feasible": [s.feasible for s in sols],
            "violation": [s.violation for s in sols],
            "active": [";".join(s.active) for s in sols],
        }
    )


# --------------------------------------------------------------------------------------------
# first fit
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FirstFit:
    """Step 2: ``(k1, λ1, λ2)``, the objective, the active constraints, feasibility / least
    violation, standard errors (historical mode), the ``k1`` profile, the per-pillar table
    (``T, atf, svc_target, svc_model, svc_naked, sensi_spot, ssr_first_order, ssr_implied,
    skew_market, skew_naked, skew_gap_rel``), the constraint table, the short-end table and the
    maps."""

    k1: float
    lambda1: float
    lambda2: float
    objective: float
    objective_skew: float
    k1_se: float
    lambda_cov: FloatArray
    active: tuple[str, ...]
    feasible: bool
    violation: float
    k1_at_bound: bool
    profile: pd.DataFrame
    table: pd.DataFrame
    constraints: pd.DataFrame
    short_end: pd.DataFrame
    maps: P1Maps
    skew_mode: str
    notes: tuple[str, ...]
    wall_seconds: float
    k2: float

    @property
    def lam(self) -> FloatArray:
        return np.array([self.lambda1, self.lambda2])

    @property
    def lambda1_se(self) -> float:
        return float(np.sqrt(self.lambda_cov[0, 0]))

    @property
    def lambda2_se(self) -> float:
        return float(np.sqrt(self.lambda_cov[1, 1]))

    @property
    def nu_min(self) -> float:
        """The smallest ``ν`` compatible with ``(λ1, λ2)``: ``|λ1 + λ2|/2``."""
        return 0.5 * abs(self.lambda1 + self.lambda2)

    @property
    def box_binding(self) -> bool:
        return any(a.startswith("nu box") for a in self.active)

    @property
    def skew_binding(self) -> tuple[str, ...]:
        return tuple(a for a in self.active if a.startswith("skew"))

    @property
    def mean_skew_gap(self) -> float:
        """Mean over pillars of ``|skew_naked / skew_market − 1|``."""
        return float(np.mean(np.abs(self.table["skew_gap_rel"])))


def _first_table(prob: _FirstProblem, lam: FloatArray, mm: P1Maps) -> pd.DataFrame:
    naked = mm.skew_naked(lam)
    return pd.DataFrame(
        {
            "T": prob.T,
            "atf": prob.atf,
            "svc_target": prob.svc_target,
            "svc_model": mm.svc(lam),
            "svc_naked": mm.svc_naked(lam),
            "sensi_spot": mm.sensi_spot(lam),
            "ssr_first_order": mm.ssr_first_order(lam),
            "ssr_implied": prob.svc_target / (prob.sigma_0 * prob.skew_market),
            "skew_market": prob.band,
            "skew_naked": naked,
            "skew_gap_rel": naked / prob.band - 1.0,
        }
    )


def _constraint_table(prob: _FirstProblem, sol: InnerSolution) -> pd.DataFrame:
    rows = []
    naked = sol.maps.skew_naked(sol.lam)
    for c in prob.constraints:
        lo, hi = c.bounds
        s = float(naked[c.index])
        tol = 1e-7 * (1.0 + abs(c.skew_market))
        edge = ""
        if prob.skew_mode == "twopoint":
            if abs(s - hi) <= tol:
                edge = c.edge_factor(True)
            elif abs(s - lo) <= tol:
                edge = c.edge_factor(False)
        rows.append(
            {
                "name": c.name,
                "T_requested": c.requested,
                "T": c.T,
                "eps": c.eps,
                "skew_market": c.skew_market,
                "lower": lo,
                "upper": hi,
                "skew_naked": s,
                "gap_rel": s / c.skew_market - 1.0,
                "within_eps": bool(lo - tol <= s <= hi + tol),
                "binding_edge": edge,
            }
        )
    return pd.DataFrame(rows)


def _short_end_table(
    prob: _FirstProblem,
    targets: TargetSet,
    xi0: ForwardVarianceCurve,
    lam: FloatArray,
    k1: float,
    cfg: BreakEvenFitConfig,
) -> pd.DataFrame:
    """The free short-end naked skew: 1M and every fitted pillar below ``T_s``."""
    t_s = cfg.skew_pillars[0]
    mats = [SIGMA0_MATURITY] + [float(t) for t in prob.T if t < t_s - _TOL_T]
    rows = []
    for T in sorted(set(mats)):
        q = pillar_quad(xi0, T, n_quad=cfg.n_quad, n_inner=cfg.n_inner)
        naked = float(lam[0] * q.J(k1) + lam[1] * q.J(cfg.k2))
        mkt = float(targets.market_skew(T)[0])
        rows.append(
            {
                "T": T,
                "skew_market": mkt,
                "skew_naked": naked,
                "gap_rel": naked / mkt - 1.0,
                "source": (
                    targets.term_structure_source
                    if targets.term_structure_source == "surface"
                    else "pillar power law"
                ),
                "fitted_pillar": bool(np.any(np.abs(prob.T - T) <= _TOL_T)),
            }
        )
    return pd.DataFrame(rows)


def _first_stderr(
    prob: _FirstProblem, best: InnerSolution, bounds: tuple[float, float], at_bound: bool
) -> tuple[float, FloatArray, list[str]]:
    """Standard errors of ``(k1, λ1, λ2)``: NaN in marking mode (the targets carry no sampling
    error); in historical mode the sandwich on the covariance rows with the target standard
    errors, ``J = [∂r/∂k1, R]``; the active constraints are ignored (noted)."""
    notes: list[str] = []
    nan2 = np.full((2, 2), np.nan)
    if not prob.has_noise:
        notes.append(
            "no standard errors for (k1, lambda): the targets carry no sampling error "
            "(marking mode)"
        )
        return float("nan"), nan2, notes
    k1 = best.k1
    n = prob.T.size
    _, R, _, _ = prob.stacked(k1)
    sigma = np.diag(prob.wc * prob.svc_se**2)
    if best.active:
        notes.append("standard errors ignore the active constraints (kink possible)")
    if at_bound:
        notes.append("k1 standard error: k1 at a bound; lambda standard errors at fixed k1")
        return float("nan"), _sandwich(R[:n], sigma), notes
    h = 1e-4 * k1
    lo, hi = max(k1 - h, bounds[0]), min(k1 + h, bounds[1])
    _, Ra, ya, _ = prob.stacked(lo)
    _, Rb, yb, _ = prob.stacked(hi)
    dr = ((Rb @ best.lam - yb) - (Ra @ best.lam - ya))[:n] / (hi - lo)
    cov = _sandwich(np.column_stack((dr, R[:n])), sigma)
    k1_var = float(cov[0, 0])
    k1_se = math.sqrt(k1_var) if np.isfinite(k1_var) and k1_var >= 0 else float("nan")
    return k1_se, np.asarray(cov[1:, 1:], dtype=np.float64), notes


def _k2_config(cfg: BreakEvenFitConfig, k2: float) -> BreakEvenFitConfig:
    """``cfg`` with ``k2`` fixed at ``k2`` and ``k1`` kept above ``k2 + k1_min_gap``."""
    lo = max(cfg.k1_bounds[0], float(k2) + cfg.k1_min_gap * (1.0 + 1e-9))
    return replace(cfg, k2=float(k2), k2_bounds=None, k1_bounds=(lo, cfg.k1_bounds[1]))


def _optimise_k2(
    prob: _FirstProblem, cfg: BreakEvenFitConfig
) -> tuple[BreakEvenFitConfig, _FirstProblem, InnerSolution, list[InnerSolution], list[str]]:
    """``k2`` fitted inside ``cfg.k2_bounds`` (:class:`BreakEvenFitConfig`): a geometric grid of
    ``k2_grid`` points (the fixed ``cfg.k2`` added when inside the bounds, so the fitted
    objective never exceeds the fixed-``k2`` one), then a bounded scalar refinement between the
    best grid point's neighbours; every candidate runs :func:`_optimise_k1` on the same problem
    (the pillar quadratures and the term-structure bank do not depend on ``k2``).  Returns the
    winning ``k2``'s config and problem, its solution and ``k1`` profile, and the notes."""
    assert cfg.k2_bounds is not None
    lo, hi = cfg.k2_bounds
    grid = np.geomspace(lo, hi, cfg.k2_grid or K2_GRID_DEFAULT)
    if lo < cfg.k2 < hi:
        grid = np.unique(np.concatenate((grid, [cfg.k2])))
    Run = tuple[BreakEvenFitConfig, _FirstProblem, InnerSolution, list[InnerSolution]]

    def run(k2: float) -> Run:
        c = _k2_config(cfg, k2)
        p = replace(prob, k2=float(k2), cache={}, mlp_cache={})
        best, sols = _optimise_k1(p, c)
        return c, p, best, sols

    runs = [run(float(k)) for k in grid]
    i = min(range(len(runs)), key=lambda j: _rank(runs[j][2]))
    best = runs[i]
    a, b = float(grid[max(i - 1, 0)]), float(grid[min(i + 1, grid.size - 1)])
    if b > a:

        def score(k: float) -> float:
            s = run(float(k))[2]
            return s.objective if s.feasible else 1e12 * (1.0 + s.violation)

        res = minimize_scalar(score, bounds=(a, b), method="bounded", options={"xatol": 1e-4 * b})
        cand = run(float(res.x))
        if _rank(cand[2]) <= _rank(best[2]):
            best = cand
    c, p, sol, sols = best
    notes = [
        f"k2 fitted: {c.k2:.4g} in {cfg.k2_bounds} (grid of {grid.size} points, bounded "
        f"refinement; grid objectives "
        f"{[round(float(r[2].objective), 6) if r[2].feasible else None for r in runs]})"
    ]
    if c.k2 <= lo * (1.0 + 1e-6) or c.k2 >= hi * (1.0 - 1e-6):
        notes.append(f"k2 = {c.k2:.4g} sits on a bound of {cfg.k2_bounds}")
    return c, p, sol, sols, notes


def fit_first(
    targets: TargetSet,
    cfg: BreakEvenFitConfig,
    xi0: ForwardVarianceCurve,
    *,
    svc_correction: FloatArray | None = None,
) -> FirstFit:
    """Step 2 (module docstring); ``svc_correction`` divides the SpotVolCovar targets (the
    iteration against simulation of :func:`fit_2f`)."""
    t0 = time.perf_counter()
    xi0 = _engine_curve(targets, xi0, cfg)
    prob, notes = _first_problem(targets, cfg, xi0, svc_correction)
    if cfg.kernel_curve is not None:
        notes.append(
            f"naked kernels A, J on the {cfg.kernel_curve.upper()} forward variance "
            "(kernel_curve; the M7 engine reads the variance-swap curve)"
        )
    if cfg.k2_bounds is None:
        best, sols = _optimise_k1(prob, cfg)
    else:
        cfg, prob, best, sols, k2_notes = _optimise_k2(prob, cfg)
        notes += k2_notes
    k1 = best.k1
    at_bound = bool(k1 <= cfg.k1_bounds[0] * (1 + 1e-6) or k1 >= cfg.k1_bounds[1] * (1 - 1e-6))
    if at_bound:
        notes.append(f"k1 = {k1:.4g} sits on a bound of {cfg.k1_bounds}")
    k1_se, lam_cov, se_notes = _first_stderr(prob, best, cfg.k1_bounds, at_bound)
    notes += se_notes
    return FirstFit(
        k1,
        float(best.lam[0]),
        float(best.lam[1]),
        best.objective,
        best.objective_skew,
        k1_se,
        lam_cov,
        best.active,
        best.feasible,
        best.violation,
        at_bound,
        _profile_frame(sols),
        _first_table(prob, best.lam, best.maps),
        _constraint_table(prob, best),
        _short_end_table(prob, targets, xi0, best.lam, k1, cfg),
        best.maps,
        prob.skew_mode,
        tuple(notes),
        time.perf_counter() - t0,
        float(cfg.k2),
    )


# --------------------------------------------------------------------------------------------
# second fit
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SecondFit:
    """Step 3: ``(ω1, ω2, χ)``, the weighted objective, standard errors (historical mode),
    bound flags, whether ``ν`` sits on the cap, the number of starts and distinct optima and the
    per-pillar table (``T, volvar_target_requested, volvar_target, volvar_model, vov_target,
    vov_model, corr_target, corr_model, sensi_spot``)."""

    omega1: float
    omega2: float
    chi: float
    nu: float
    objective: float
    stderr: dict[str, float]
    bound_flags: tuple[str, ...]
    nu_at_cap: bool
    n_starts: int
    n_distinct: int
    table: pd.DataFrame
    notes: tuple[str, ...]
    wall_seconds: float


def _rho_xy(lam: FloatArray, om1: float, om2: float, chi: float) -> tuple[float, float, float]:
    r1 = float(np.clip(lam[0] / om1, -1.0, 1.0))
    r2 = float(np.clip(lam[1] / om2, -1.0, 1.0))
    rxy = r1 * r2 + chi * math.sqrt(max(1.0 - r1 * r1, 0.0)) * math.sqrt(max(1.0 - r2 * r2, 0.0))
    return r1, r2, rxy


def volvar_p1(
    x: FloatArray, lam: FloatArray, A: FloatArray, atf: FloatArray, spot: FloatArray
) -> tuple[FloatArray, float]:
    """``VolVar_P1`` per pillar and ``ν`` for ``x = (ω1, ω2, χ)`` at fixed ``λ`` (module
    docstring: the full quadratic form with ``SensiSpot = spot``)."""
    om1, om2, chi = float(x[0]), float(x[1]), float(x[2])
    _, _, rxy = _rho_xy(lam, om1, om2, chi)
    sx = 0.5 * om1 * A[:, 0] * atf
    sy = 0.5 * om2 * A[:, 1] * atf
    vv = (
        spot * spot + 2.0 * spot * (0.5 * atf * (A @ lam)) + sx * sx + sy * sy + 2.0 * rxy * sx * sy
    )
    nu = 0.5 * math.sqrt(max(om1 * om1 + om2 * om2 + 2.0 * rxy * om1 * om2, 0.0))
    return np.asarray(vv, dtype=np.float64), nu


def fit_second(
    targets: TargetSet,
    cfg: BreakEvenFitConfig,
    first: FirstFit,
    *,
    volvar_correction: FloatArray | None = None,
) -> SecondFit:
    """Step 3 (module docstring): SLSQP with the ν cap as an inequality from ten starts; the
    best optimum is kept and the number of distinct optima reported.  ``volvar_correction``
    divides the VolVar targets (the iteration against simulation of :func:`fit_2f`)."""
    t0 = time.perf_counter()
    idx, _ = _select_pillars(targets, cfg.pillars)
    notes: list[str] = []
    vv_req = np.asarray(targets.vol_var, dtype=np.float64)[idx]
    vv_se = np.asarray(targets.vol_var_se, dtype=np.float64)[idx]
    corr = (
        np.asarray(targets.correl_target, dtype=np.float64)[idx]
        if targets.correl_target.size == targets.pillars.size
        else np.full(idx.size, np.nan)
    )
    svc = first.table["svc_model"].to_numpy()
    if targets.mode != "marking":
        vv_t = vv_req
        notes.append(
            "historical mode: the empirical VolVar target is kept (the correlation-preserving "
            "rebuild applies to the SABR correlation of marking mode)"
        )
    elif cfg.volvar_target == "direct":
        vv_t = vv_req
        notes.append(
            "VolVar target: the targets' VoV_BE^2 itself (volvar_target='direct', the desk note's "
            "step 2); the M7 rule rebuilds it from the achieved covariance"
        )
    elif np.all(np.isfinite(corr) & (np.abs(corr) > 1e-12)):
        vv_t = np.asarray((svc / corr) ** 2, dtype=np.float64)
    else:
        raise ValueError("marking targets need a finite, non-zero Corr_BE at every pillar")
    if volvar_correction is not None:
        corr_vv = np.asarray(volvar_correction, dtype=np.float64)
        if corr_vv.shape != vv_t.shape or np.any(~np.isfinite(corr_vv)) or np.any(corr_vv <= 0):
            raise ValueError("volvar_correction needs one finite positive factor per fitted pillar")
        vv_t = vv_t / corr_vv
        notes.append(
            "VolVar targets divided by the simulated/analytic factors "
            f"{np.round(corr_vv, 4).tolist()} (iterate_against_simulation)"
        )
    w = _weights(cfg.weights_volvar, vv_t)
    lam = first.lam
    A = first.maps.naked.A
    atf = first.maps.naked.atf
    spot = first.table["sensi_spot"].to_numpy()
    lower = np.array([abs(lam[0]) + 1e-9, abs(lam[1]) + 1e-9, cfg.chi_bounds[0]])
    upper = np.array([cfg.omega_max, cfg.omega_max, cfg.chi_bounds[1]])
    if np.any(lower >= upper):
        raise ValueError("|lambda_i| exceeds omega_max: enlarge omega_max")

    def objective(x: FloatArray) -> float:
        vv, _ = volvar_p1(x, lam, A, atf, spot)
        r = vv - vv_t
        return float(np.sum(w * r * r))

    cap = cfg.nu_cap

    def slack(x: FloatArray) -> float:
        return (np.inf if cap is None else cap) - volvar_p1(x, lam, A, atf, spot)[1]

    cons: list[Any] = [] if cap is None else [{"type": "ineq", "fun": slack}]

    base = np.maximum(np.abs(lam), 0.05)
    starts = [np.array([lower[0] * (1 + 1e-6), lower[1] * (1 + 1e-6), 0.0])]
    for scale in (1.0 / 0.7, 2.0, 1.2):
        for chi0 in (0.0, -0.6, 0.6):
            starts.append(np.array([base[0] * scale, base[1] * scale, chi0]))
    bounds = list(zip(lower, upper))
    results = []
    for x0 in starts:
        x0 = np.clip(x0, lower, upper)
        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=cons,
            options={"ftol": 1e-15, "maxiter": 1000},
        )
        x = np.clip(np.asarray(res.x, dtype=np.float64), lower, upper)
        if slack(x) >= -1e-7:
            results.append((objective(x), x))
    if not results:
        raise RuntimeError("the second minimisation found no point inside the nu cap")
    results.sort(key=lambda r: r[0])
    best_obj, x = results[0]
    distinct = 1
    for o, xx in results[1:]:
        if abs(o - best_obj) <= 0.01 * max(best_obj, 1e-12) and not np.allclose(
            xx, x, atol=1e-3, rtol=1e-3
        ):
            distinct += 1
    vv_m, nu = volvar_p1(x, lam, A, atf, spot)
    names = ("omega1", "omega2", "chi")
    stderr: dict[str, float] = dict.fromkeys(names, float("nan"))
    n = vv_t.size
    sw = np.sqrt(w)
    if np.all(np.isfinite(vv_se)) and np.all(vv_se > 0):
        Jm = np.empty((n, 3))
        for i in range(3):
            hstep = 1e-6 * max(abs(x[i]), 1.0)
            xp, xm = x.copy(), x.copy()
            xp[i] += hstep
            xm[i] -= hstep
            Jm[:, i] = sw * (
                volvar_p1(xp, lam, A, atf, spot)[0] - volvar_p1(xm, lam, A, atf, spot)[0]
            )
            Jm[:, i] /= 2.0 * hstep
        try:
            JtJ_inv = np.linalg.inv(Jm.T @ Jm)
            cov = JtJ_inv @ (Jm.T @ ((w * vv_se * vv_se)[:, None] * Jm)) @ JtJ_inv
            stderr = {nm: float(np.sqrt(cov[i, i])) for i, nm in enumerate(names)}
        except np.linalg.LinAlgError:
            notes.append("singular Jacobian in the second fit: no standard errors")
    else:
        notes.append(
            "no standard errors for (omega1, omega2, chi): the VolVar targets carry no sampling "
            "error (marking mode)"
        )
    flags = []
    for i, nm in enumerate(names[:2]):
        if x[i] <= lower[i] * (1 + 1e-5) + 1e-8:
            flags.append(f"{nm} at |lambda{i + 1}| (|rho_SX{i + 1}| = 1)")
    if x[2] <= lower[2] + 1e-6:
        flags.append("chi at lower bound")
    if x[2] >= upper[2] - 1e-6:
        flags.append("chi at upper bound")
    at_cap = cfg.nu_cap is not None and bool(nu >= cfg.nu_cap * (1.0 - 1e-5))
    if at_cap:
        flags.append(f"nu at cap {cfg.nu_cap:g}")
    for i, nm in enumerate(names[:2]):
        if x[i] >= upper[i] * (1 - 1e-5):
            flags.append(f"{nm} at omega_max {cfg.omega_max:g}")
    table = pd.DataFrame(
        {
            "T": first.maps.T,
            "volvar_target_requested": vv_req,
            "volvar_target": vv_t,
            "volvar_model": vv_m,
            "vov_target": np.sqrt(vv_t),
            "vov_model": np.sqrt(vv_m),
            "corr_target": corr,
            "corr_model": svc / np.sqrt(vv_m),
            "sensi_spot": spot,
        }
    )
    return SecondFit(
        float(x[0]),
        float(x[1]),
        float(x[2]),
        float(nu),
        float(best_obj),
        stderr,
        tuple(flags),
        at_cap,
        len(starts),
        distinct,
        table,
        tuple(notes),
        time.perf_counter() - t0,
    )


# --------------------------------------------------------------------------------------------
# messages
# --------------------------------------------------------------------------------------------


def _svc_text(table: pd.DataFrame) -> str:
    return ", ".join(
        f"{T:g}y {m:+.5f}/{t:+.5f} ({m / t - 1.0:+.1%})"
        for T, m, t in zip(table["T"], table["svc_model"], table["svc_target"])
    )


def fit_messages(
    first: FirstFit, second: SecondFit, cfg: BreakEvenFitConfig
) -> tuple[str, list[str]]:
    """``(status, messages)`` of a fit (module docstring): the infeasible message, one binding
    message per active skew edge, the ν-cap warning (logged)."""
    msgs: list[str] = []
    svc = _svc_text(first.table)
    ct = first.constraints
    if not first.feasible:
        if first.skew_mode != "twopoint":
            what = "the soft-mode box"
        elif len(ct) == 2:
            what = (
                f"the two-point skew constraint (T={ct['T'].iloc[0]:g}y within "
                f"eps_s={float(ct['eps'].iloc[0]):g}, T={ct['T'].iloc[1]:g}y within "
                f"eps_l={float(ct['eps'].iloc[1]):g})"
            )
        else:
            what = (
                f"the {len(ct)}-point skew constraint ("
                + ", ".join(
                    f"T={float(t):g}y within eps={float(e):g}" for t, e in zip(ct["T"], ct["eps"])
                )
                + ")"
            )
        skews = ", ".join(
            f"{r.T:g}y {r.skew_naked:+.5f} vs market {r.skew_market:+.5f} ({r.gap_rel:+.1%})"
            for r in ct.itertuples()
        )
        msgs.append(
            INFEASIBLE_MESSAGE.format(
                what=what,
                box=(
                    f"|lambda1| + |lambda2| <= 2 nu_cap = {2.0 * cfg.nu_cap:g}"
                    if cfg.nu_cap is not None
                    else f"|lambda_i| <= {LAMBDA_BOX_FRACTION:g} omega_max = "
                    f"{LAMBDA_BOX_FRACTION * cfg.omega_max:g}"
                ),
                k1_lo=cfg.k1_bounds[0],
                k1_hi=cfg.k1_bounds[1],
                delta=first.violation,
                k1=first.k1,
                skews=skews,
                svc=svc,
            )
        )
        status = "infeasible"
    else:
        status = "interior"
        if first.skew_mode == "twopoint":
            for rec in ct.to_dict(orient="records"):
                if not rec["binding_edge"]:
                    continue
                naked, lo, hi = float(rec["skew_naked"]), float(rec["lower"]), float(rec["upper"])
                upper = abs(naked - hi) <= abs(naked - lo)
                msgs.append(
                    BINDING_MESSAGE.format(
                        T=float(rec["T"]),
                        edge="upper" if upper else "lower",
                        factor=rec["binding_edge"],
                        bound=hi if upper else lo,
                        naked=naked,
                        market=float(rec["skew_market"]),
                        gap=float(rec["gap_rel"]),
                        eps=float(rec["eps"]),
                        svc=svc,
                    )
                )
                status = "binding"
    hits = [f for f in second.bound_flags if "at omega_max" in f]
    if cfg.nu_cap is None and (first.box_binding or hits):
        where = ", ".join(
            (["step 2 box |lambda_i| <= 0.99 omega_max"] if first.box_binding else []) + hits
        )
        msgs.append(
            f"per-factor bound binds ({where}; the desk note's bounds, omega_max "
            f"{cfg.omega_max:g}): first-order break-evens biased at large vol of vol"
        )
        if status == "interior":
            status = "binding"
    elif first.box_binding or second.nu_at_cap:
        warn = NU_CAP_WARNING.format(cap=cfg.nu_cap)
        where = "step 2 box |lambda1|+|lambda2| <= 2 nu_cap" if first.box_binding else "step 3"
        log.warning("%s (%s)", warn, where)
        msgs.append(f"{warn} ({where})")
        if status == "interior":
            status = "binding"
    return status, msgs


# --------------------------------------------------------------------------------------------
# stage 3
# --------------------------------------------------------------------------------------------


def mean_abs_leverage_deviation(
    leverage: Any, surface: Any, *, n_sd: float = 2.0
) -> tuple[float, pd.DataFrame]:
    """Per leverage time ``t > 0`` the mean of ``|L(t, k) − 1|`` over the grid points with ``|k|
    ≤ n_sd atf(t) sqrt(t)``, and the unweighted mean over times (columns ``t,
    mean_abs_L_minus_1, n``)."""
    rows = []
    for i, t in enumerate(leverage.times):
        if t <= 0:
            continue
        sd = float(surface.atm_vol(t)) * np.sqrt(t)
        inside = np.abs(leverage.k_grid) <= n_sd * sd
        vals = leverage.values[i][inside]
        if vals.size == 0:
            continue
        rows.append(
            {
                "t": float(t),
                "mean_abs_L_minus_1": float(np.mean(np.abs(vals - 1.0))),
                "n": int(inside.sum()),
            }
        )
    table = pd.DataFrame(rows)
    mean = float(table["mean_abs_L_minus_1"].mean()) if len(table) else float("nan")
    return mean, table


@dataclass(frozen=True)
class Stage3Inputs:
    """What the validation needs beyond the fit: the surface, particle / simulation settings,
    the pricing configuration, the pillars of each check, the forward-start windows and,
    optionally, an already calibrated ``model`` (then nothing is calibrated and
    ``recalibrated`` is ``False``)."""

    surface: Any
    particle: Any
    sim: Any
    pricing_sim: Any
    local_vol: Any | None = None
    ssr_pillars: tuple[float, ...] = (0.25, 1.0)
    breakeven_pillars: tuple[float, ...] = (0.25, 1.0)
    forward_starts: tuple[tuple[float, float], ...] = ((1.0, 2.0), (2.0, 3.0))
    eps: float = 0.05
    model: Any | None = None
    mixing_paths: int = 100_000
    mixing_seed: int = 11
    mixing_dt: float = 1.0 / 365.0


class BreakEvenValidationError(RuntimeError):
    """The stage-3 assertion failed (report decision viii, engine-bias term): the simulated
    break-evens of the calibrated model are not within the tolerance of the first-order
    break-evens at the fitted parameters.  ``result`` carries the :class:`FitResult` (with its
    stage 3) when raised by :func:`fit_2f`, so nothing is lost."""

    def __init__(self, message: str, result: Any = None) -> None:
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class Stage3Report:
    """Stage-3 tables (module docstring) with the wall clocks and the recalibration flag.
    ``check`` is the assertion table (per breakeven pillar and quantity: simulated value and
    standard error, the fit's target, the first-order value at the solution, ``engine_bias`` =
    simulated / first-order − 1 — the asserted term —, ``gap_vs_target`` and
    ``first_order_miss`` = first-order / target − 1 — reported, never asserted: a binding fit —,
    ``within`` = ``|engine_bias| <= tolerance`` and ``note``, which flags a row without a
    first-order value where the target gap was used instead), ``within_tolerance`` its verdict
    at ``tolerance`` and ``check_message`` the clear message when it fails (names the engine-bias
    failures only, then lists the first-order misses above the tolerance as reported, not
    asserted)."""

    mean_abs_l_minus_1: float
    leverage_table: pd.DataFrame
    ssr_table: pd.DataFrame
    skew_table: pd.DataFrame
    breakeven_table: pd.DataFrame
    forward_table: pd.DataFrame
    calibration_seconds: float
    wall_seconds: float
    recalibrated: bool
    n_particles: int
    n_paths: int
    check: pd.DataFrame = field(default_factory=pd.DataFrame)
    tolerance: float = DEFAULT_STAGE3_TOLERANCE
    within_tolerance: bool = True
    check_message: str = ""

    def summary(self) -> str:
        lines = [
            f"stage 3: mean |L - 1| {self.mean_abs_l_minus_1:.4f}; recalibrated: "
            f"{'yes' if self.recalibrated else 'no'} (calibration "
            f"{self.calibration_seconds:.0f} s, {self.n_particles} particles); pricing "
            f"{self.n_paths} paths; wall clock {self.wall_seconds:.0f} s",
            "numerical LSV SSR (diagnostic, not a target):",
            self.ssr_table.round(4).to_string(index=False),
            "naked skew (mixing derivative) vs market skew:",
            self.skew_table.drop(columns=["note"]).round(4).to_string(index=False),
            "simulated break-evens of the LSV vs targets:",
            self.breakeven_table.round(5).to_string(index=False),
            "stage-3 check (engine_bias = sim / first-order - 1 asserted; gap_vs_target and "
            "first_order_miss reported, not asserted):",
            self.check.round(5).to_string(index=False),
            f"stage-3 assertion (engine bias: simulated within {self.tolerance:.0%} of the "
            "first-order break-evens at the fitted parameters; the target miss of a binding fit "
            "is reported, not asserted): "
            + ("PASS" if self.within_tolerance else "FAIL - " + self.check_message),
        ]
        if len(self.forward_table):
            lines += [
                "forward 90/110 skew vs spot skew:",
                self.forward_table.round(4).to_string(index=False),
            ]
        return "\n".join(lines)


def _target_at(targets: TargetSet, T: float, values: FloatArray) -> float:
    tp = np.asarray(targets.pillars, dtype=np.float64)
    i = int(np.argmin(np.abs(tp - T)))
    return float(values[i]) if abs(tp[i] - T) <= _TOL_T else float("nan")


def _table_at(table: pd.DataFrame | None, T: float, column: str) -> float:
    if table is None or column not in table:
        return float("nan")
    tt = table["T"].to_numpy(dtype=float)
    i = int(np.argmin(np.abs(tt - T)))
    return float(table[column].iloc[i]) if abs(tt[i] - T) <= _TOL_T else float("nan")


def spot_skew_90_110(surface: Any, T: float) -> float:
    """``σ̂_T(0.9 S_0) − σ̂_T(1.1 S_0)`` (spot moneyness, the convention of the forward-start
    strikes relative to ``S_{T1}``)."""
    fwd = float(surface.forward_curve.forward(T))
    s0 = float(surface.forward_curve.spot)
    k = np.log(np.array([0.9, 1.1]) * s0 / fwd)
    v = np.asarray(surface.implied_vol_k(k, np.full(2, float(T))), dtype=np.float64)
    return float(v[0] - v[1])


def _calibrated(params: BergomiParams, inputs: Stage3Inputs) -> tuple[Any, float, bool, int]:
    """``(lsv, calibration seconds, recalibrated, n_particles)``: the leverage calibrated on
    ``inputs.surface`` for ``params``, or ``inputs.model`` as given."""
    from volsto.calibration.particle import calibrate_leverage
    from volsto.models.bergomi import BergomiSV
    from volsto.models.lsv import LSV

    surface = inputs.surface
    if inputs.model is not None:
        lsv = inputs.model
        return lsv, 0.0, False, int(lsv.leverage.metadata.get("n_particles", 0))
    fc = surface.forward_curve
    xi0 = xi0_curve(surface, min(surface.max_maturity, max(inputs.particle.horizon + 1.0, 5.0)))
    kernel = BergomiSV(params, xi0, fc)
    t0 = time.perf_counter()
    result = calibrate_leverage(
        surface, kernel, inputs.particle, inputs.sim, local_vol_cfg=inputs.local_vol
    )
    lsv = LSV(kernel, result.leverage)
    return lsv, time.perf_counter() - t0, True, int(inputs.particle.n_particles)


def simulated_breakeven_table(
    lsv: Any,
    pillars: Sequence[float],
    targets: TargetSet,
    psim: Any,
    *,
    eps: float,
    fit_table: pd.DataFrame | None,
) -> pd.DataFrame:
    """The simulated break-evens of ``lsv`` per pillar against the fit's targets and first-order
    values: ``svc_target`` the SpotVolCovar target, ``volvar_target`` the requested ``VoV_BE²``,
    ``volvar_target_fit`` the step-3 target the fit solved (the ρ_SABR rebuild in marking mode),
    ``*_p1_first_order`` the fit's analytic values at its solution."""
    from volsto.analytics.breakeven import simulated_breakevens

    rows = []
    for T in pillars:
        b = simulated_breakevens(lsv, float(T), sim=psim, eps=eps, sigma_0=targets.sigma_0)
        rows.append(
            {
                "T": float(T),
                "svc_sim": b.spot_vol_covar,
                "svc_se": b.spot_vol_covar_se,
                "svc_target": _target_at(targets, float(T), targets.spot_vol_covar),
                "svc_p1_first_order": _table_at(fit_table, float(T), "svc_model"),
                "volvar_sim": b.vol_var,
                "volvar_se": b.vol_var_se,
                "volvar_target": _target_at(targets, float(T), targets.vol_var),
                "volvar_target_fit": _table_at(fit_table, float(T), "volvar_target"),
                "volvar_p1_first_order": _table_at(fit_table, float(T), "volvar_model"),
                "corr_sim": b.correl,
                "corr_target": _target_at(targets, float(T), targets.correl_target),
                "ssr_sim": b.ssr,
            }
        )
    return pd.DataFrame(rows)


def breakeven_check(be_table: pd.DataFrame, tolerance: float) -> tuple[pd.DataFrame, bool, str]:
    """The stage-3 assertion (report decision viii, restricted to the engine-bias term by the
    owner on the M7 Part 3 report): per pillar and quantity (SpotVolCovar, VolVar), the
    simulated value against the **first-order value at the fitted parameters** within
    ``tolerance`` relative — ``engine_bias = sim / first_order − 1``, ``within = |engine_bias| <=
    tolerance``.  The gap to the fit's target (``svc_target``; ``volvar_target_fit``, the step-3
    target the fit solved, or ``volvar_target`` when the fit carried none) and the first-order
    miss of that target (``first_order_miss = first_order / target − 1``, a binding fit, already
    reported by the binding status) are REPORTED columns, never asserted.  A row without a
    finite non-zero first-order value falls back to the target gap and is flagged in ``note``
    (``NO_FIRST_ORDER_NOTE``).  Returns the check table, the verdict and the message (empty when
    passed), which names the engine-bias failures only and appends one sentence listing the
    first-order misses above the tolerance as reported, not asserted (binding fit)."""
    rows = []
    for rec in be_table.to_dict(orient="records"):
        vv_t = rec.get("volvar_target_fit", float("nan"))
        if not np.isfinite(vv_t):
            vv_t = rec["volvar_target"]
        pairs = (
            (
                "SpotVolCovar",
                rec["svc_sim"],
                rec["svc_se"],
                rec["svc_target"],
                rec["svc_p1_first_order"],
            ),
            ("VolVar", rec["volvar_sim"], rec["volvar_se"], vv_t, rec["volvar_p1_first_order"]),
        )
        for name, sim, se, target, fo in pairs:
            gap_t = sim / target - 1.0 if target else float("nan")
            has_fo = bool(np.isfinite(fo) and fo)
            gap_fo = sim / fo - 1.0 if has_fo else float("nan")
            miss_fo = fo / target - 1.0 if has_fo and target else float("nan")
            if has_fo:
                within = bool(np.isfinite(gap_fo) and abs(gap_fo) <= tolerance)
                note = ""
            else:
                within = bool(np.isfinite(gap_t) and abs(gap_t) <= tolerance)
                note = NO_FIRST_ORDER_NOTE
            rows.append(
                {
                    "T": rec["T"],
                    "quantity": name,
                    "sim": sim,
                    "se": se,
                    "target": target,
                    "first_order": fo,
                    "gap_vs_target": gap_t,
                    "first_order_miss": miss_fo,
                    "engine_bias": gap_fo,
                    "within": within,
                    "note": note,
                }
            )
    table = pd.DataFrame(rows)
    ok = bool(table["within"].all()) if len(table) else True
    if ok:
        return table, True, ""
    bad = table[~table["within"]]
    parts = []
    for r in bad.to_dict(orient="records"):
        if r["note"]:
            parts.append(
                f"{r['quantity']} at T={r['T']:g}: simulated {r['sim']:.5f} +/- {r['se']:.5f} vs "
                f"target {r['target']:.5f} ({r['gap_vs_target']:+.1%}; {r['note']})"
            )
        else:
            parts.append(
                f"{r['quantity']} at T={r['T']:g}: simulated {r['sim']:.5f} +/- {r['se']:.5f} vs "
                f"first-order {r['first_order']:.5f} at the fitted parameters (engine bias "
                f"{r['engine_bias']:+.1%}; target {r['target']:.5f}, {r['gap_vs_target']:+.1%})"
            )
    # the header names the rule that judged the failing rows: the engine bias when any of them
    # had a first-order value, the target gap when none had (the fallback rows say so)
    any_fo = bool((bad["note"] == "").any())
    any_fb = bool((bad["note"] != "").any())
    if any_fo and any_fb:
        rule = "engine bias; target gap where no first-order value"
    elif any_fo:
        rule = "engine bias"
    else:
        rule = "no first-order value: target gap"
    msg = (
        f"simulated break-evens outside {tolerance:.0%} of the first-order break-evens at the "
        f"fitted parameters ({rule}): " + "; ".join(parts)
    )
    misses = table[np.abs(table["first_order_miss"].to_numpy(dtype=float)) > tolerance]
    if len(misses):
        listed = ", ".join(
            f"{r['quantity']} at T={r['T']:g} {r['first_order_miss']:+.1%}"
            for r in misses.to_dict(orient="records")
        )
        msg += (
            f". First-order misses of the targets above {tolerance:.0%}, reported, not asserted "
            f"(the first-order fit's miss of its target, a binding constraint when the fit says "
            f"so): {listed}."
        )
    return table, False, msg


def stage3_validation(
    params: BergomiParams,
    inputs: Stage3Inputs,
    targets: TargetSet,
    *,
    fit_table: pd.DataFrame | None = None,
    tolerance: float = DEFAULT_STAGE3_TOLERANCE,
    assert_breakevens: bool = False,
) -> Stage3Report:
    """Stage 3 (module docstring): calibrate the leverage on ``inputs.surface`` for ``params``
    (or take ``inputs.model``) and report the validation tables; nothing is refit.  The
    break-even assertion (:func:`breakeven_check`) is always evaluated and reported; with
    ``assert_breakevens`` a failure raises :class:`BreakEvenValidationError`."""
    from volsto.analytics.forward_smile import forward_smile
    from volsto.analytics.smile_dynamics import ssr_numerical_many
    from volsto.calibration.history import mixing_atmf_batch

    t_all = time.perf_counter()
    surface = inputs.surface
    lsv, cal_s, recalibrated, n_particles = _calibrated(params, inputs)
    kernel = lsv.kernel
    mean_abs, lev_table = mean_abs_leverage_deviation(lsv.leverage, surface)
    psim = inputs.pricing_sim
    ssr_rows = ssr_numerical_many(lsv, list(inputs.ssr_pillars), eps=inputs.eps, sim=psim)
    ssr_table = pd.DataFrame(
        {
            "T": [r.T for r in ssr_rows],
            "ssr_lsv": [r.R for r in ssr_rows],
            "ssr_lsv_se": [r.R_stderr for r in ssr_rows],
            "ssr_target": [_target_at(targets, r.T, targets.ssr_target) for r in ssr_rows],
            "ssr_implied_by_targets": [
                _target_at(targets, r.T, targets.ssr_implied) for r in ssr_rows
            ],
            "ssr_first_order_p1": [_table_at(fit_table, r.T, "ssr_first_order") for r in ssr_rows],
            "skew_lsv": [r.skew for r in ssr_rows],
            "skew_lsv_se": [r.skew_stderr for r in ssr_rows],
        }
    )
    nf = int(kernel.n_factors)
    skew_rows = []
    for T in inputs.breakeven_pillars:
        st = float(targets.market_skew(float(T))[0])
        try:
            mb = mixing_atmf_batch(
                kernel,
                float(T),
                np.array([0.0]),
                np.zeros((1, nf)),
                n_paths=inputs.mixing_paths,
                seed=inputs.mixing_seed,
                dt=inputs.mixing_dt,
            )
            sk, sk_se, note = float(mb.skew[0]), float(mb.skew_se[0]), ""
        except ValueError as exc:  # the mixing solution needs |spot/vol correlation| < 1
            sk, sk_se, note = float("nan"), float("nan"), str(exc)
        skew_rows.append(
            {
                "T": float(T),
                "skew_naked": sk,
                "skew_naked_se": sk_se,
                "skew_market": st,
                "gap": sk / st - 1.0 if st else float("nan"),
                "skew_naked_first_order": _table_at(fit_table, float(T), "skew_naked"),
                "note": note,
            }
        )
    skew_table = pd.DataFrame(skew_rows)
    be_table = simulated_breakeven_table(
        lsv, inputs.breakeven_pillars, targets, psim, eps=inputs.eps, fit_table=fit_table
    )
    check, within, check_msg = breakeven_check(be_table, tolerance)
    fwd_rows = []
    for t1, t2 in inputs.forward_starts:
        sm = forward_smile(lsv, t1, t2, [0.9, 1.1], psim)
        i_atm = int(np.argmin(np.abs(sm.strikes - sm.forward_ratio)))
        i_lo = int(np.argmin(np.abs(sm.strikes - 0.9)))
        i_hi = int(np.argmin(np.abs(sm.strikes - 1.1)))
        fs = float(sm.vols[i_lo] - sm.vols[i_hi])
        fs_se = float(np.hypot(sm.vol_stderr[i_lo], sm.vol_stderr[i_hi]))
        spot = spot_skew_90_110(surface, t2 - t1)
        fwd_rows.append(
            {
                "t1": t1,
                "t2": t2,
                "atm_fwd_vol": float(sm.vols[i_atm]),
                "atm_fwd_vol_se": float(sm.vol_stderr[i_atm]),
                "fwd_skew_90_110": fs,
                "fwd_skew_90_110_se": fs_se,
                "spot_skew_90_110": spot,
                "ratio_fwd_to_spot": fs / spot,
                "ratio_se": fs_se / abs(spot),
            }
        )
    report = Stage3Report(
        mean_abs,
        lev_table,
        ssr_table,
        skew_table,
        be_table,
        pd.DataFrame(fwd_rows),
        cal_s,
        time.perf_counter() - t_all,
        recalibrated,
        n_particles,
        int(psim.n_paths),
        check,
        float(tolerance),
        within,
        check_msg,
    )
    if assert_breakevens and not within:
        raise BreakEvenValidationError(check_msg)
    return report


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FitResult:
    """The fit: book and break-even parameters, targets, ``ξ₀``, config, both minimisations,
    the per-pillar table (step-2 columns, ``volvar_target_requested, volvar_target,
    volvar_model, vov_target, vov_model, corr_target, corr_model``), the status and messages,
    the paired risk regime, the optional stage 3, the wall clock and whether a leverage was
    recalibrated."""

    params: BergomiParams
    breakeven: BreakEvenParams
    targets: TargetSet
    xi0: ForwardVarianceCurve
    config: BreakEvenFitConfig
    first: FirstFit
    second: SecondFit
    table: pd.DataFrame
    status: str
    messages: tuple[str, ...]
    risk_regime: str
    stage3: Stage3Report | None
    wall_seconds: float
    recalibrated: bool
    pricing_date: pd.Timestamp | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    iterations: pd.DataFrame | None = None

    @property
    def constraints(self) -> pd.DataFrame:
        return self.first.constraints

    @property
    def stage3_passed(self) -> bool | None:
        """The stage-3 assertion verdict (None without stage 3)."""
        return None if self.stage3 is None else bool(self.stage3.within_tolerance)

    @property
    def short_end(self) -> pd.DataFrame:
        return self.first.short_end

    @property
    def mean_skew_gap(self) -> float:
        return self.first.mean_skew_gap

    @property
    def svc_rel_error(self) -> FloatArray:
        return np.asarray(self.table["svc_model"] / self.table["svc_target"] - 1.0)

    @property
    def min_correlation_eigenvalue(self) -> float:
        """The smallest eigenvalue of the fitted correlation matrix of ``(S, X¹, X²)``
        (:func:`correlation_min_eigenvalue`)."""
        return correlation_min_eigenvalue(self.params)

    @property
    def config_yaml(self) -> str:
        """A loadable model config (``model`` :class:`BergomiParams`, ``breakeven``,
        ``risk_regime``) and the provenance (targets, settings, status, messages, the constraint
        and short-end tables, objectives, standard errors, flags, stage-3 headline numbers)."""
        f, s, tg, cfg = self.first, self.second, self.targets, self.config
        doc: dict[str, Any] = {
            "model": to_mapping(self.params),
            "breakeven": to_mapping(self.breakeven),
            "risk_regime": self.risk_regime,
            "provenance": {
                "mode": tg.mode,
                "pricing_date": (
                    None if self.pricing_date is None else str(self.pricing_date.date())
                ),
                "config": to_mapping(cfg),
                "pillars": [float(t) for t in f.maps.T],
                "sigma_0": float(tg.sigma_0),
                "ssr_target": [float(x) for x in tg.ssr_target],
                "status": self.status,
                "messages": list(self.messages),
                "constraints": _records(f.constraints),
                "short_end": _records(f.short_end),
                "table": _records(self.table),
                "first": {
                    "objective": float(f.objective),
                    "k1_se": float(f.k1_se),
                    "lambda1_se": float(f.lambda1_se),
                    "lambda2_se": float(f.lambda2_se),
                    "active": list(f.active),
                    "violation": float(f.violation),
                    "k1_at_bound": bool(f.k1_at_bound),
                    "notes": list(f.notes),
                },
                "second": {
                    "objective": float(s.objective),
                    "nu": float(s.nu),
                    "stderr": {k: float(v) for k, v in s.stderr.items()},
                    "bound_flags": list(s.bound_flags),
                    "n_starts": int(s.n_starts),
                    "n_distinct_optima": int(s.n_distinct),
                    "notes": list(s.notes),
                },
                "target_flags": list(tg.flags),
                "notes": list(self.notes),
                "stage3": (
                    None
                    if self.stage3 is None
                    else {
                        "mean_abs_L_minus_1": float(self.stage3.mean_abs_l_minus_1),
                        "assertion_passed": bool(self.stage3.within_tolerance),
                        "assertion_message": self.stage3.check_message,
                        "ssr_lsv": _records(self.stage3.ssr_table),
                        "forward_skew": _records(self.stage3.forward_table),
                    }
                ),
                "iterations": None if self.iterations is None else _records(self.iterations),
                "recalibrated": bool(self.recalibrated),
                "wall_seconds": float(self.wall_seconds),
            },
        }
        return str(yaml.safe_dump(doc, sort_keys=False))

    def write_yaml(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.config_yaml, encoding="utf-8")
        return p

    def summary(self) -> str:
        p, b, f, s, cfg = self.params, self.breakeven, self.first, self.second, self.config
        when = "" if self.pricing_date is None else f" @ {self.pricing_date.date()}"
        cap_text = (
            f"nu_cap {cfg.nu_cap:g}"
            if cfg.nu_cap is not None
            else f"no nu cap (omega_i <= {cfg.omega_max:g}, |lambda_i| <= "
            f"{LAMBDA_BOX_FRACTION:g} omega_max)"
        )
        lines = [
            f"fit_2f [{self.targets.mode}]{when}: status {self.status.upper()}; ssr_target "
            f"{np.round(self.targets.ssr_target, 3).tolist()}, skew_mode {cfg.skew_mode}, "
            f"skew_eps {tuple(float(e) for e in cfg.skew_eps)} at {cfg.skew_pillars}, "  # type: ignore[union-attr]
            f"{cap_text}",
        ]
        lines += [f"MESSAGE: {m}" for m in self.messages]
        lines += [
            f"params: nu {p.nu:.4f} theta {p.theta:.4f} k1 {p.k1:.4f} (se {f.k1_se:.3f}) k2 "
            f"{p.k2:.3f} ({'fitted in ' + str(cfg.k2_bounds) if cfg.k2_bounds else 'fixed'}) "
            f"rho_SX1 {p.rho_SX1:+.4f} rho_SX2 {p.rho_SX2:+.4f} rho12 {p.rho12:+.4f}",
            f"break-even: omega1 {b.omega1:.4f} omega2 {b.omega2:.4f} lambda1 {b.lambda1:+.4f} "
            f"lambda2 {b.lambda2:+.4f} chi {b.chi:+.4f}; objectives {f.objective:.3e} / "
            f"{s.objective:.3e}; active {list(f.active)}; bounds {list(s.bound_flags)}; risk "
            f"regime {self.risk_regime}; wall clock {self.wall_seconds:.1f} s; recalibrated: "
            f"{'yes' if self.recalibrated else 'no'}",
            (
                "two-point skew constraint (naked vs market):"
                if len(f.constraints) == 2
                else f"{len(f.constraints)}-point skew constraint (naked vs market):"
            ),
            f.constraints.round(5).to_string(index=False),
            "free short-end naked skew:",
            f.short_end.round(5).to_string(index=False),
            "per pillar (first order):",
            self.table.round(5).to_string(index=False),
        ]
        notes = (*self.notes, *f.notes, *s.notes, *self.targets.flags)
        if notes:
            lines.append("notes: " + "; ".join(notes))
        if self.iterations is not None and len(self.iterations):
            lines += [
                "iteration against simulation (targets corrected by the simulated/analytic gap):",
                self.iterations.round(4).to_string(index=False),
            ]
        if self.stage3 is not None:
            lines.append(self.stage3.summary())
        return "\n".join(lines)


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for rec in frame.to_dict(orient="records"):
        out.append(
            {
                str(k): (
                    float(v)
                    if isinstance(v, float | np.floating)
                    else bool(v) if isinstance(v, bool | np.bool_) else v
                )
                for k, v in rec.items()
            }
        )
    return out


def fit_2f(
    targets: TargetSet,
    xi0: ForwardVarianceCurve,
    cfg: BreakEvenFitConfig | None = None,
    *,
    stage3: Stage3Inputs | None = None,
    pricing_date: pd.Timestamp | None = None,
    iterate_against_simulation: int = 0,
    assert_stage3: bool = True,
) -> FitResult:
    """Steps 2 and 3 on ``targets`` with the forward-variance curve ``xi0``, the status and
    messages, and stage 3 when ``stage3`` is given (module docstring).  ``skew_mode="auto"`` is
    resolved on the targets' mode (:func:`resolve_skew_mode`) and the resolved config is
    returned.  ``iterate_against_simulation=k`` (needs ``stage3``): after the fit, ``k`` times, the
    leverage is calibrated, the break-evens simulated at every fitted pillar and the targets
    divided by the cumulative simulated/analytic factors before a refit (so the simulated
    break-evens of the final model land on the original targets up to first-order curvature);
    the per-iteration gaps are in :attr:`FitResult.iterations`; when an iteration moves the worst
    simulated gap vs target up (a binding fit cannot follow corrected targets) the loop stops and
    keeps the best iteration, with a note — measured on SPX 2022-12-30 ``(1.0, 0.10)`` (binding):
    the raw iteration diverged, ν 1.94 → 2.08 → 2.28 with ρ → −1 / −1 and the VolVar gap 0.31 →
    1.53.  With ``assert_stage3`` (default) a failed stage-3 assertion raises
    :class:`BreakEvenValidationError` carrying the full result.
    """
    t0 = time.perf_counter()
    c, mode_note = _resolved(cfg or BreakEvenFitConfig(), targets)
    idx, notes = _select_pillars(targets, c.pillars)
    if mode_note:
        notes.append(mode_note)
    if iterate_against_simulation < 0:
        raise ValueError("iterate_against_simulation must be a non-negative integer")
    if iterate_against_simulation and stage3 is None:
        raise ValueError("iterate_against_simulation needs stage3 inputs (the simulation)")
    n = idx.size
    factors_svc = np.ones(n)
    factors_vv = np.ones(n)

    def solve(f_svc: FloatArray | None, f_vv: FloatArray | None) -> tuple[Any, ...]:
        first = fit_first(targets, c, xi0, svc_correction=f_svc)
        second = fit_second(targets, c, first, volvar_correction=f_vv)
        status, messages = fit_messages(first, second, c)
        be = BreakEvenParams(
            first.k1,
            first.k2,
            second.omega1,
            second.omega2,
            first.lambda1,
            first.lambda2,
            second.chi,
        )
        params = be.to_book()
        st = second.table
        table = first.table.copy()
        for col in (
            "volvar_target_requested",
            "volvar_target",
            "volvar_model",
            "vov_target",
            "vov_model",
            "corr_target",
            "corr_model",
        ):
            table[col] = st[col].to_numpy()
        if f_svc is not None:
            table["svc_correction"] = f_svc
            table["volvar_correction"] = f_vv
        return first, second, status, messages, be, params, table

    def gaps(check: pd.DataFrame, quantity: str, column: str) -> float:
        return float(np.max(np.abs(check.loc[check["quantity"] == quantity, column])))

    first, second, status, messages, be, params, table = solve(None, None)
    iterations: pd.DataFrame | None = None
    if iterate_against_simulation:
        assert stage3 is not None
        pillars = tuple(float(t) for t in table["T"])
        rows = []
        best: tuple[float, Any] | None = None
        diverged = False
        for it in range(1, iterate_against_simulation + 1):
            lsv, cal_s, _, _ = _calibrated(params, stage3)
            bt = simulated_breakeven_table(
                lsv, pillars, targets, stage3.pricing_sim, eps=stage3.eps, fit_table=table
            )
            chk, _, _ = breakeven_check(bt, c.stage3_tolerance)
            worst = max(
                gaps(chk, "SpotVolCovar", "gap_vs_target"), gaps(chk, "VolVar", "gap_vs_target")
            )
            rows.append(
                {
                    "iteration": it - 1,
                    "nu": params.nu,
                    "theta": params.theta,
                    "k1": params.k1,
                    "rho_SX1": params.rho_SX1,
                    "rho_SX2": params.rho_SX2,
                    "rho12": params.rho12,
                    "status": status,
                    "max_gap_svc_vs_target": gaps(chk, "SpotVolCovar", "gap_vs_target"),
                    "max_gap_volvar_vs_target": gaps(chk, "VolVar", "gap_vs_target"),
                    "max_engine_bias_svc": gaps(chk, "SpotVolCovar", "engine_bias"),
                    "max_engine_bias_volvar": gaps(chk, "VolVar", "engine_bias"),
                    "calibration_seconds": cal_s,
                }
            )
            current = (first, second, status, messages, be, params, table, factors_svc, factors_vv)
            if best is not None and worst > best[0] * (1.0 + 1e-9):
                # the correction moved the simulated break-evens away from the targets (a binding
                # fit cannot follow the corrected targets): keep the best iteration and stop
                diverged = True
                first, second, status, messages, be, params, table, factors_svc, factors_vv = best[
                    1
                ]
                break
            best = (worst, current)
            r_svc = bt["svc_sim"].to_numpy() / table["svc_model"].to_numpy()
            r_vv = bt["volvar_sim"].to_numpy() / table["volvar_model"].to_numpy()
            factors_svc = factors_svc * r_svc
            factors_vv = factors_vv * r_vv
            first, second, status, messages, be, params, table = solve(factors_svc, factors_vv)
        iterations = pd.DataFrame(rows)
        if diverged:
            worst_by_it = np.maximum(
                iterations["max_gap_svc_vs_target"], iterations["max_gap_volvar_vs_target"]
            )
            notes.append(
                f"iterate_against_simulation diverged at iteration {len(rows) - 1}: the worst "
                f"simulated gap vs target grew beyond the best {float(worst_by_it.min()):.3f} "
                f"(the fit is {status}, a constrained covariance cannot follow corrected "
                f"targets); kept iteration {int(np.argmin(worst_by_it))}"
            )
        else:
            notes.append(
                f"iterated {iterate_against_simulation}x against simulation: cumulative target "
                f"corrections svc {np.round(factors_svc, 4).tolist()}, volvar "
                f"{np.round(factors_vv, 4).tolist()}"
            )
    if abs(params.rho12) > c.rho12_flag:
        notes.append(
            f"rho12 = {params.rho12:+.3f}: |rho12| > {c.rho12_flag:g}, two-factor structure "
            "collapsing (the two factors are nearly one)"
        )
    lam_min = correlation_min_eigenvalue(params)
    if lam_min < CORRELATION_EIGEN_FLAG:
        notes.append(
            f"correlation of (S, X1, X2) near-singular: smallest eigenvalue {lam_min:.2e} < "
            f"{CORRELATION_EIGEN_FLAG:g} (rho_SX1 {params.rho_SX1:+.3f}, rho_SX2 "
            f"{params.rho_SX2:+.3f}, rho12 {params.rho12:+.3f}): the spot and the two factors are "
            "nearly collinear, regressions on the factors are ill-identified (SPEC §8.2)"
        )
    s3 = None
    recalibrated = False
    if stage3 is not None:
        s3 = stage3_validation(
            params, stage3, targets, fit_table=table, tolerance=c.stage3_tolerance
        )
        recalibrated = s3.recalibrated
        if iterations is not None:
            last = {
                "iteration": iterate_against_simulation,
                "nu": params.nu,
                "theta": params.theta,
                "k1": params.k1,
                "rho_SX1": params.rho_SX1,
                "rho_SX2": params.rho_SX2,
                "rho12": params.rho12,
                "status": status,
                "max_gap_svc_vs_target": gaps(s3.check, "SpotVolCovar", "gap_vs_target"),
                "max_gap_volvar_vs_target": gaps(s3.check, "VolVar", "gap_vs_target"),
                "max_engine_bias_svc": gaps(s3.check, "SpotVolCovar", "engine_bias"),
                "max_engine_bias_volvar": gaps(s3.check, "VolVar", "engine_bias"),
                "calibration_seconds": s3.calibration_seconds,
            }
            iterations = pd.concat([iterations, pd.DataFrame([last])], ignore_index=True)
    result = FitResult(
        params,
        be,
        targets,
        xi0,
        c,
        first,
        second,
        table,
        status,
        tuple(messages),
        RISK_REGIME,
        s3,
        time.perf_counter() - t0,
        recalibrated,
        pricing_date,
        tuple(notes),
        iterations,
    )
    if assert_stage3 and s3 is not None and not s3.within_tolerance:
        raise BreakEvenValidationError(s3.check_message, result=result)
    return result


def naked_kernel(result: FitResult, forward_curve: Any) -> Any:
    """The fitted pure-SV kernel :class:`~volsto.models.bergomi.BergomiSV` on the fit's ``ξ₀``."""
    from volsto.models.bergomi import BergomiSV

    return BergomiSV(result.params, result.xi0, forward_curve)


def correlation_min_eigenvalue(params: BergomiParams) -> float:
    """The smallest eigenvalue of the Brownian correlation matrix of ``(S, X¹, X²)``,
    ``[[1, ρ_SX1, ρ_SX2], [ρ_SX1, 1, ρ12], [ρ_SX2, ρ12, 1]]``: 0 on the boundary of the admissible
    set (the collapsed ``ρ_SX1 = ρ_SX2 = −1, ρ12 = +1`` included), noted below
    :data:`CORRELATION_EIGEN_FLAG`.  Checked by
    ``tests/test_fit_2f.py::test_correlation_min_eigenvalue``."""
    r1, r2, r12 = float(params.rho_SX1), float(params.rho_SX2), float(params.rho12)
    corr = np.array([[1.0, r1, r2], [r1, 1.0, r12], [r2, r12, 1.0]])
    return float(np.linalg.eigvalsh(corr)[0])


def fit_2f_marking(
    surface: Any,
    cfg: BreakEvenFitConfig | None = None,
    *,
    ssr_target: SsrInput = 1.0,
    anchor_power: float = 1.0,
    stage3: Stage3Inputs | None = None,
    h: float = SABR_CURVATURE_H,
    iterate_against_simulation: int = 0,
    assert_stage3: bool = True,
    skew_h: float | None = None,
    step0: Step0Triplets | None = None,
) -> FitResult:
    """Marking mode on a surface: targets by :func:`~volsto.calibration.targets.marking_targets`
    (the config's pillars, ``mat_min``, SmoothBreakEven, the radicand guard and step-0
    conventions), ``ξ₀`` the surface's variance-swap strip to the last fitted pillar.
    ``skew_h`` (default ``None``: the M7 marking fit, unchanged) is the stencil-consistent
    argument of :func:`marking_targets_for`.  ``step0`` (default ``None``: the surface's ATM
    derivatives) takes step 0 from a triplet source — the desk's SABRW fits
    (:class:`volsto.market.sabrw.SabrwTermStructure`, SPEC §15 Part 3)."""
    c = cfg or BreakEvenFitConfig()
    targets = marking_targets_for(
        surface,
        c,
        ssr_target=ssr_target,
        anchor_power=anchor_power,
        h=h,
        skew_h=skew_h,
        step0=step0,
    )
    t_max = float(min(surface.max_maturity, max(targets.pillars)))
    xi0 = xi0_curve(surface, t_max)
    c = replace(c, pillars=tuple(float(t) for t in targets.pillars))
    try:
        r = fit_2f(
            targets,
            xi0,
            c,
            stage3=stage3,
            iterate_against_simulation=iterate_against_simulation,
            assert_stage3=assert_stage3,
        )
    except BreakEvenValidationError as exc:
        if exc.result is not None:
            exc.result = _with_target_notes(exc.result, targets)
        raise
    return _with_target_notes(r, targets)


def _with_target_notes(r: FitResult, targets: TargetSet) -> FitResult:
    extra = tuple(f"target: {f}" for f in targets.flags if "dropped" in f or "guard" in f)
    return replace(r, notes=(*r.notes, *extra)) if extra else r


def marking_targets_for(
    surface: Any,
    cfg: BreakEvenFitConfig,
    *,
    ssr_target: SsrInput = 1.0,
    anchor_power: float = 1.0,
    h: float = SABR_CURVATURE_H,
    skew_h: float | None = None,
    step0: Step0Triplets | None = None,
) -> TargetSet:
    """:func:`~volsto.calibration.targets.marking_targets` with the config's target settings
    (``step0``: an optional step-0 triplet source, the desk's SABRW fits).

    ``h`` is the curvature stencil's half-width; ``skew_h`` (default ``None``: the surface's
    analytic ATM skew when it has one — the M7 reading, unchanged) reads the skew by the central
    difference of that half-width instead.  Passing both reads a surface on the stencil of a
    finite strike strip: the M8b recalibration rule's base fit does so on the pricing snapshot
    with the strip's ``(h, curvature_h)`` so its held targets and a refit's compare like for
    like (:meth:`volsto.hedging.hedger.RecalibrationRule.marking_targets`).  A config whose
    ``step0`` names a source needs one: without ``step0`` it raises (no silent fall-back to the
    surface's derivatives)."""
    if cfg.step0 is not None and step0 is None:
        raise ValueError(
            f"the fit config's step 0 reads the date's {cfg.step0.upper()} fits: pass them as "
            "step0 (volsto.market.loaders.load_step0_source; ShiftedTriplets on a moved surface)"
        )
    return marking_targets(
        surface,
        cfg.pillars,
        ssr_target=ssr_target,
        anchor_power=anchor_power,
        h=h,
        mat_min=cfg.mat_min,
        smooth_breakeven=cfg.smooth_breakeven,
        sabrw_power=cfg.sabrw_power,
        atf_ref=cfg.atf_ref,
        radicand_floor=cfg.radicand_floor,
        skew_h=skew_h,
        step0=step0,
        sigma_0=(
            None
            if cfg.sigma0_maturity is None
            else float(np.asarray(surface.atm_vol(cfg.sigma0_maturity)))
        ),
    )


def _curve_from_vs(pillars: FloatArray, vs_vol: FloatArray) -> ForwardVarianceCurve:
    """Forward-variance curve through the pillar VS vols (flat VS vol before the first and beyond
    the last pillar, made monotone by a tiny ramp)."""
    mats = np.concatenate(([0.5 * pillars[0]], pillars, [pillars[-1] + 1.0, pillars[-1] + 5.0]))
    vs = np.concatenate(([vs_vol[0]], vs_vol, [vs_vol[-1], vs_vol[-1]]))
    W = vs * vs * mats
    W = np.maximum.accumulate(W + 1e-12 * np.arange(W.size))
    return ForwardVarianceCurve(mats, W)


def fit_2f_historical(
    history: SurfaceHistory,
    cfg: BreakEvenFitConfig | None = None,
    *,
    end: pd.Timestamp | str | None = None,
    window_vol: int = WINDOW_VOL,
    window_ssr: int = WINDOW_SSR,
    stage3: Stage3Inputs | None = None,
) -> FitResult:
    """Historical mode on a :class:`~volsto.calibration.history.SurfaceHistory` at ``end``
    (default: the last date): targets by :func:`~volsto.calibration.targets.historical_targets`
    on the config's pillars present in the history and at or above ``mat_min``, ``ξ₀`` from the
    pricing date's VS vols at all the history's pillars.  Step 0 is a marking step: a config
    with a ``step0`` source raises."""
    c = cfg or BreakEvenFitConfig()
    if c.step0 is not None:
        raise ValueError("step0 applies to marking mode: the historical targets have no step 0")
    e = history.date_index(end)
    end_ts = history.dates[e]
    hp = np.asarray(history.pillars, dtype=np.float64)
    pillars = [
        float(T)
        for T in c.pillars
        if np.any(np.abs(hp - float(T)) <= _TOL_T) and float(T) >= c.mat_min - 1e-12
    ]
    if len(pillars) < 2:
        raise ValueError(f"fewer than two of the config pillars {c.pillars} are in the history")
    targets = historical_targets(
        history, pillars, end=end_ts, window_vol=window_vol, window_ssr=window_ssr
    )
    xi0 = _curve_from_vs(hp, np.asarray(history.vs_vol.to_numpy()[e], dtype=np.float64))
    c = replace(c, pillars=tuple(pillars))
    return fit_2f(targets, xi0, c, stage3=stage3, pricing_date=end_ts)


# --------------------------------------------------------------------------------------------
# study specs: a fitted parameter set as a calibration spec plus the fit provenance
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FitSpec:
    """A study entry (``configs/studies/m7_p1_marking/*.yaml``): the :class:`~volsto.config.
    CalibrationSpec` of the fitted parameters (section ``spec``, the leverage-cache key) and the
    fit provenance (section ``fit``: the loadable ``config``, ``ssr_target``, the surface label,
    ``breakeven``, ``status``, ``messages``)."""

    path: Path
    spec: Any
    fit: dict[str, Any]

    @property
    def config(self) -> BreakEvenFitConfig:
        return from_mapping(BreakEvenFitConfig, self.fit["config"], path=str(self.path))

    @property
    def ssr_target(self) -> float:
        return float(self.fit["ssr_target"])

    @property
    def breakeven(self) -> BreakEvenParams:
        return BreakEvenParams(**{k: float(v) for k, v in self.fit["breakeven"].items()})

    @property
    def snapshot(self) -> str | None:
        """The snapshot the fit read, relative to the repository root (``fit.snapshot``, recorded
        since the named fits of 2026-09-27), ``None`` when the entry names none."""
        v = self.fit.get("snapshot")
        return None if v is None else str(v)

    def step0_source(self, surface: Any) -> Step0Triplets | None:
        """Step 0's source of the entry's fit on ``surface`` (the fit's surface): ``None`` when
        the config's step 0 reads the surface, else the SABRW fits of the entry's snapshot
        (:func:`volsto.market.loaders.load_step0_source`); an entry without one raises."""
        if self.config.step0 is None:
            return None
        if self.snapshot is None:
            raise ValueError(
                f"{self.path}: the fit's step 0 reads a snapshot's SABRW fits and the entry names "
                "no snapshot (fit.snapshot)"
            )
        from volsto.market.loaders import load_step0_source

        return load_step0_source(_REPO_ROOT / self.snapshot, surface)


def fit_spec_document(
    result: FitResult,
    base_spec: Any,
    *,
    n_particles: int,
    ssr_target: float,
    label: str,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """The YAML document of :class:`FitSpec` for ``result`` on ``base_spec`` (its market,
    surface, perturbation, particle and simulation settings; the model replaced by the fitted
    parameters and the particle count by ``n_particles``)."""
    spec = replace(
        base_spec,
        model=result.params,
        particle=replace(base_spec.particle, n_particles=int(n_particles)),
    )
    return {
        "spec": to_mapping(spec),
        "fit": {
            "label": label,
            "config": to_mapping(result.config),
            "ssr_target": float(ssr_target),
            "breakeven": to_mapping(result.breakeven),
            "status": result.status,
            "messages": list(result.messages),
            **dict(extra or {}),
        },
    }


def write_fit_spec(
    result: FitResult,
    base_spec: Any,
    path: str | Path,
    *,
    n_particles: int,
    ssr_target: float,
    label: str,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write :func:`fit_spec_document` to ``path`` (directories created): the study script's
    hand-off to the tests, which read the leverage with ``allow_calibrate=False``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = fit_spec_document(
        result, base_spec, n_particles=n_particles, ssr_target=ssr_target, label=label, extra=extra
    )
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return p


def load_fit_spec(path: str | Path) -> FitSpec:
    from volsto.config import CalibrationSpec, load_yaml

    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or "spec" not in raw or "fit" not in raw:
        raise ValueError(f"{p}: not a fit spec (needs 'spec' and 'fit' sections)")
    return FitSpec(p, load_yaml(p, CalibrationSpec, section="spec"), dict(raw["fit"]))


__all__ = [
    "BINDING_MESSAGE",
    "CORRELATION_EIGEN_FLAG",
    "DEFAULT_K2",
    "DEFAULT_NU_CAP",
    "DEFAULT_SKEW_EPS",
    "DEFAULT_SKEW_PILLARS",
    "DEFAULT_SKEW_WEIGHT",
    "DEFAULT_STAGE3_TOLERANCE",
    "DESK_FIT",
    "DESK_FIT_STENCIL",
    "FIT_ENGINES",
    "FIT_PRESETS",
    "INFEASIBLE_MESSAGE",
    "K2_GRID_DEFAULT",
    "LAMBDA_BOX_FRACTION",
    "MAX_FINITE_SE",
    "MLP_FIT_GRID",
    "NO_FIRST_ORDER_NOTE",
    "NU_CAP_WARNING",
    "PRODUCTION_BOUNDS",
    "RHO12_COLLAPSE",
    "RISK_REGIME",
    "SKEW_MODES",
    "SKEW_MODE_CHOICES",
    "STEP0_SOURCES",
    "TERM_STRUCTURE_KINDS",
    "VOLVAR_TARGETS",
    "WEIGHT_KINDS",
    "AffineMaps",
    "BreakEvenFitConfig",
    "BreakEvenValidationError",
    "FirstFit",
    "FitResult",
    "FitSpec",
    "InnerSolution",
    "MlpMaps",
    "P1Maps",
    "PillarQuad",
    "SecondFit",
    "SkewConstraint",
    "Stage3Inputs",
    "Stage3Report",
    "TermStructureBank",
    "affine_maps",
    "affine_maps_from_kernels",
    "breakeven_check",
    "correlation_min_eigenvalue",
    "fit_2f",
    "fit_2f_historical",
    "fit_2f_marking",
    "fit_first",
    "fit_messages",
    "fit_preset",
    "fit_second",
    "fit_spec_document",
    "k1_profile",
    "load_fit_spec",
    "marking_targets_for",
    "mean_abs_leverage_deviation",
    "naked_kernel",
    "pillar_quad",
    "resolve_skew_mode",
    "simulated_breakeven_table",
    "spot_skew_90_110",
    "stage3_validation",
    "term_structure_bank",
    "ts_substitution_power",
    "volvar_p1",
    "write_fit_spec",
]
