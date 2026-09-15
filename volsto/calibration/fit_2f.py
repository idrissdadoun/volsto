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
the kernel's order-one skew at every ``t ∈ (0, T]``.  ``SensiSpot`` is the spot sensitivity the
leverage generates when it absorbs the skew residual ``S − Skew_naked`` (eq. 12.52 in covariance
form: ``SpotVolCovar_P1 = σ_0 R_LSV S`` at order one); it is zero when the naked skew equals the
market skew at every maturity, and ``SpotVolCovar_P1`` and ``Skew_naked`` are affine in ``λ`` at
fixed ``(k1, k2)`` (:class:`P1Maps`).  *Reading of the specification (flagged):* the owner's
``VolVar_P1`` carries ``σ_0² f(T)²`` and ``σ_0 f(T) A`` terms, i.e. a non-zero ``SensiSpot = σ_0
f(T)`` — the leverage-included (P1-deco) break-even, the ``SVC_PILV`` of the previous message —
so ``SpotVolCovar_P1`` is this form, not the naked kernel's (``SensiSpot = 0``, reported as
``svc_naked``).  The owner's ``VolVar_P1`` as written has both cross terms at half weight
(``(λ_i/2) σ_0 f A_i`` and ``(ω1 ω2/4) ρ_XY A1 A2``; the quadratic form ``(SensiSpot + SensiX +
SensiY)²`` has ``2 SensiSpot ρ_Si SensiX_i = SensiSpot λ_i A_i atf`` and ``2 ρ_XY SensiX SensiY =
(ω1 ω2/2) ρ_XY A1 A2 atf²``); the engine's full quadratic form — the one the analytic-vs-
simulation gate validated — is used.  In historical mode (no surface) the skew residual is
interpolated linearly in ``t`` between the pillars (flat outside): ``(1/T) ∫ f (S − λ·J)`` uses
the pillar values of ``S`` and ``J``, so the leverage term vanishes whenever the naked skew
matches the market at the pillars (the synthetic recovery test).

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
added to the covariance objective (normalised as the relative weights), not the default.

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
from it biased ν by −20%). ``ν_cap`` (default 2.5) is a config value, not a hard-wired rail: when a
fit binds it (step 2 box or step 3) :data:`NU_CAP_WARNING` is logged and attached to the messages.

**Always reported** (:meth:`FitResult.summary`, :meth:`FitResult.config_yaml`): the fitted
parameters; the naked skew at 1Y / 5Y against the market (within ``eps`` or the binding message);
the free short-end naked skew (1M and the pillars below 1Y); per pillar the SpotVolCovar /
VolVar targets and achieved values, the implied correlation and the first-order P1 SSR; with
stage 3 the calibrated LSV's numerical SSR (a diagnostic, never a target: at ``ssr_target = 1`` the
model cannot realise SSR 1 without leverage, pure SV floors near 1.5), the mean ``|L − 1|`` and the
forward 90/110 skew at 1y-into-1y and 2y-into-1y against the spot 1y skew (sticky-strike marking
gives about 130–150% of spot skew, book eq. 12.52).

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

import logging
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from scipy.optimize import linprog, minimize, minimize_scalar

from volsto.analytics.breakeven import Kernels, _gl
from volsto.analytics.reparam import BreakEvenParams
from volsto.calibration.history import WINDOW_SSR, WINDOW_VOL, SurfaceHistory
from volsto.calibration.targets import (
    DEFAULT_ATF_REF,
    DEFAULT_MAT_MIN,
    DEFAULT_SABRW_POWER,
    DEFAULT_TARGET_PILLARS,
    SABR_CURVATURE_H,
    SIGMA0_MATURITY,
    SsrInput,
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
#: the ν cap (config; owner default 2.5)
DEFAULT_NU_CAP = 2.5
#: the two-point skew tolerance (owner default, both points)
DEFAULT_SKEW_EPS = 0.10
#: the two constraint maturities ``(T_s, T_l)``
DEFAULT_SKEW_PILLARS: tuple[float, float] = (1.0, 5.0)
SKEW_MODES = ("twopoint", "soft")
WEIGHT_KINDS = ("relative", "uniform")
#: term-structure factor ``f(t)`` of the leverage integrals
TERM_STRUCTURE_KINDS = ("atmf", "flat", "vs")
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
    "infeasible: no (lambda1, lambda2) meets {what} inside |lambda1| + |lambda2| <= 2 nu_cap = "
    "{box:g} at any k1 in [{k1_lo:g}, {k1_hi:g}]; returned the least-violation fit (skew bounds "
    "relaxed by {delta:.4f} x |Skew_SABR| at k1 = {k1:.4g}, box kept): naked skew {skews}; "
    "SpotVolCovar achieved vs target: {svc}"
)
_BOX_LABELS = ("nu box l1+l2", "nu box l1-l2", "nu box -l1+l2", "nu box -l1-l2")
_TOL_T = 1e-9
_TOL_ACTIVE = 1e-9
_TOL_FEAS = 1e-9


# --------------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BreakEvenFitConfig:
    """Settings of the P1 marking calibration (module docstring).

    Targets: ``pillars`` (default 3M–10Y), ``mat_min`` (drop pillars below; 3M),
    ``smooth_breakeven`` (SmoothBreakEven, on), ``sabrw_power`` / ``atf_ref`` (the convexity
    rescaling of step 0; 1 / 0.3).  Step 2: ``k2`` fixed (0.2), ``k1_bounds`` (``k1_bounds[0]``
    must exceed ``k2 + k1_min_gap``), ``k1_grid`` points of the coarse geometric grid,
    ``skew_mode`` (:data:`SKEW_MODES`), ``skew_eps`` a float (both points) or ``(eps_s, eps_l)``
    (normalised to a pair), ``skew_pillars`` ``(T_s, T_l)``, ``skew_weight`` (soft mode only),
    ``weights_covar`` (:data:`WEIGHT_KINDS` or one weight per fitted pillar), ``term_structure``
    (``f(t)`` of the leverage integrals).  Step 3: ``weights_volvar``, ``nu_cap`` (config cap of
    both minimisations, warning when bound), ``chi_bounds``, ``omega_max``.  Quadrature orders
    ``n_quad`` / ``n_inner`` (pillar kernels) and ``n_ts`` / ``n_quad_ts`` / ``n_inner_ts`` (the
    term-structure integrals, ``t = T u^p``)."""

    pillars: tuple[float, ...] = DEFAULT_TARGET_PILLARS
    mat_min: float = DEFAULT_MAT_MIN
    smooth_breakeven: bool = True
    sabrw_power: float = DEFAULT_SABRW_POWER
    atf_ref: float = DEFAULT_ATF_REF
    k2: float = DEFAULT_K2
    k1_bounds: tuple[float, float] = (0.3, 20.0)
    k1_min_gap: float = 0.05
    k1_grid: int = 25
    skew_mode: str = "twopoint"
    skew_eps: float | tuple[float, float] = DEFAULT_SKEW_EPS
    skew_pillars: tuple[float, float] = DEFAULT_SKEW_PILLARS
    skew_weight: float = 1.0
    weights_covar: str | tuple[float, ...] = "relative"
    weights_volvar: str | tuple[float, ...] = "relative"
    term_structure: str = "atmf"
    nu_cap: float = DEFAULT_NU_CAP
    chi_bounds: tuple[float, float] = (-0.99, 0.99)
    omega_max: float = 20.0
    n_quad: int = 64
    n_inner: int = 32
    n_ts: int = 64
    n_quad_ts: int = 32
    n_inner_ts: int = 24

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
        if self.skew_mode not in SKEW_MODES:
            raise ValueError(f"skew_mode must be one of {SKEW_MODES}")
        eps = self.skew_eps
        pair = (float(eps), float(eps)) if isinstance(eps, int | float) else tuple(eps)
        if len(pair) != 2 or not all(math.isfinite(e) and e >= 0.0 for e in pair):
            raise ValueError("skew_eps must be a non-negative float or an (eps_s, eps_l) pair")
        object.__setattr__(self, "skew_eps", (float(pair[0]), float(pair[1])))
        ts, tl = self.skew_pillars
        if not 0 < ts < tl:
            raise ValueError("skew_pillars must be increasing positive maturities (T_s, T_l)")
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
        if self.nu_cap <= 0:
            raise ValueError("nu_cap must be positive")
        clo, chi_ = self.chi_bounds
        if not -1.0 <= clo < chi_ <= 1.0:
            raise ValueError("chi_bounds must be increasing inside [-1, 1]")
        if self.omega_max <= 2.0 * self.nu_cap:
            raise ValueError(
                "omega_max must exceed 2 nu_cap (the first fit allows |lambda_i| <= 2 nu_cap)"
            )
        if min(self.n_quad, self.n_inner, self.n_ts, self.n_quad_ts, self.n_inner_ts) < 4:
            raise ValueError("quadrature orders must be at least 4")

    @property
    def eps_pair(self) -> tuple[float, float]:
        e = self.skew_eps
        assert isinstance(e, tuple)
        return float(e[0]), float(e[1])


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
    for name, req, eps in zip(("T_s", "T_l"), cfg.skew_pillars, cfg.eps_pair):
        i = int(np.argmin(np.abs(T - req)))
        if abs(T[i] - req) > _TOL_T:
            notes.append(
                f"skew constraint {name} = {req:g}y is not a fitted pillar (pillars "
                f"{[round(float(x), 4) for x in T]}): applied at the nearest pillar {T[i]:g}y"
            )
        out.append(SkewConstraint(float(req), float(T[i]), i, float(eps), float(skew[i]), name))
    if out[0].index == out[1].index:
        notes.append(
            f"both skew constraints fall on the pillar {out[0].T:g}y: one point constrained"
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
    nu_cap: float
    k2: float
    cache: dict[float, P1Maps] = field(default_factory=dict, compare=False, repr=False)

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
            R = np.vstack((R, s * mm.naked.j / self.skew_market[:, None]))
            y = np.concatenate((y, np.full(self.T.size, s)))
        return mm, R, y, self.T.size

    def constraint_rows(self, mm: P1Maps) -> tuple[FloatArray, FloatArray, FloatArray, list[str]]:
        """``(G, h, scale, labels)``: the ν box and, in two-point mode, the two slabs."""
        rows = [np.array([1.0, 1.0]), np.array([1.0, -1.0]), np.array([-1.0, 1.0])]
        rows.append(np.array([-1.0, -1.0]))
        h = [2.0 * self.nu_cap] * 4
        scale = [2.0 * self.nu_cap] * 4
        labels = list(_BOX_LABELS)
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
        if not feas:
            relax = np.array([not lab.startswith("nu box") for lab in labels])
            violation = _min_violation(G / scale[:, None], h / scale, relax)
            lam, active, feas2 = _qp2(H, g, G, h + (violation + _TOL_FEAS) * scale * relax)
            if not feas2:  # numerical: take the LP point
                lam = np.zeros(2)
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


def _first_problem(
    targets: TargetSet, cfg: BreakEvenFitConfig, xi0: ForwardVarianceCurve
) -> tuple[_FirstProblem, list[str]]:
    idx, notes = _select_pillars(targets, cfg.pillars)
    T = np.asarray(targets.pillars, dtype=np.float64)[idx]
    skew = np.asarray(targets.skew_target, dtype=np.float64)[idx]
    if np.any(skew == 0) or not np.all(np.isfinite(skew)):
        raise ValueError("the fit needs finite, non-zero market skews at every pillar")
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
    constraints, cnotes = _skew_constraints(T, skew, cfg)
    if cfg.skew_mode == "twopoint":
        notes += cnotes
    svc = np.asarray(targets.spot_vol_covar, dtype=np.float64)[idx]
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
        float(cfg.nu_cap),
        float(cfg.k2),
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
    prob, _ = _first_problem(targets, cfg, xi0)
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
            "skew_market": prob.skew_market,
            "skew_naked": naked,
            "skew_gap_rel": naked / prob.skew_market - 1.0,
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


def fit_first(targets: TargetSet, cfg: BreakEvenFitConfig, xi0: ForwardVarianceCurve) -> FirstFit:
    """Step 2 (module docstring)."""
    t0 = time.perf_counter()
    prob, notes = _first_problem(targets, cfg, xi0)
    best, sols = _optimise_k1(prob, cfg)
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
        cfg.skew_mode,
        tuple(notes),
        time.perf_counter() - t0,
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


def fit_second(targets: TargetSet, cfg: BreakEvenFitConfig, first: FirstFit) -> SecondFit:
    """Step 3 (module docstring): SLSQP with the ν cap as an inequality from ten starts; the
    best optimum is kept and the number of distinct optima reported."""
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
    elif np.all(np.isfinite(corr) & (np.abs(corr) > 1e-12)):
        vv_t = np.asarray((svc / corr) ** 2, dtype=np.float64)
    else:
        raise ValueError("marking targets need a finite, non-zero Corr_BE at every pillar")
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

    def slack(x: FloatArray) -> float:
        return cfg.nu_cap - volvar_p1(x, lam, A, atf, spot)[1]

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
            constraints=[{"type": "ineq", "fun": slack}],
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
    at_cap = bool(nu >= cfg.nu_cap * (1.0 - 1e-5))
    if at_cap:
        flags.append(f"nu at cap {cfg.nu_cap:g}")
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
        eps_s, eps_l = cfg.eps_pair
        what = (
            f"the two-point skew constraint (T={ct['T'].iloc[0]:g}y within eps_s={eps_s:g}, "
            f"T={ct['T'].iloc[1]:g}y within eps_l={eps_l:g})"
            if cfg.skew_mode == "twopoint"
            else "the soft-mode box"
        )
        skews = ", ".join(
            f"{r.T:g}y {r.skew_naked:+.5f} vs market {r.skew_market:+.5f} ({r.gap_rel:+.1%})"
            for r in ct.itertuples()
        )
        msgs.append(
            INFEASIBLE_MESSAGE.format(
                what=what,
                box=2.0 * cfg.nu_cap,
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
        if cfg.skew_mode == "twopoint":
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
    if first.box_binding or second.nu_at_cap:
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


@dataclass(frozen=True)
class Stage3Report:
    """Stage-3 tables (module docstring) with the wall clocks and the recalibration flag."""

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


def stage3_validation(
    params: BergomiParams,
    inputs: Stage3Inputs,
    targets: TargetSet,
    *,
    fit_table: pd.DataFrame | None = None,
) -> Stage3Report:
    """Stage 3 (module docstring): calibrate the leverage on ``inputs.surface`` for ``params``
    (or take ``inputs.model``) and report the validation tables; nothing is refit."""
    from volsto.analytics.breakeven import simulated_breakevens
    from volsto.analytics.forward_smile import forward_smile
    from volsto.analytics.smile_dynamics import ssr_numerical_many
    from volsto.calibration.history import mixing_atmf_batch
    from volsto.calibration.particle import calibrate_leverage
    from volsto.models.bergomi import BergomiSV
    from volsto.models.lsv import LSV

    t_all = time.perf_counter()
    surface = inputs.surface
    if inputs.model is None:
        fc = surface.forward_curve
        xi0 = xi0_curve(surface, min(surface.max_maturity, max(inputs.particle.horizon + 1.0, 5.0)))
        kernel = BergomiSV(params, xi0, fc)
        t0 = time.perf_counter()
        result = calibrate_leverage(
            surface, kernel, inputs.particle, inputs.sim, local_vol_cfg=inputs.local_vol
        )
        cal_s = time.perf_counter() - t0
        lsv = LSV(kernel, result.leverage)
        recalibrated = True
        n_particles = int(inputs.particle.n_particles)
    else:
        lsv = inputs.model
        kernel = lsv.kernel
        cal_s = 0.0
        recalibrated = False
        n_particles = int(lsv.leverage.metadata.get("n_particles", 0))
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
    be_rows = []
    for T in inputs.breakeven_pillars:
        b = simulated_breakevens(lsv, float(T), sim=psim, eps=inputs.eps, sigma_0=targets.sigma_0)
        svc_t = _target_at(targets, float(T), targets.spot_vol_covar)
        vv_t = _target_at(targets, float(T), targets.vol_var)
        be_rows.append(
            {
                "T": float(T),
                "svc_sim": b.spot_vol_covar,
                "svc_se": b.spot_vol_covar_se,
                "svc_target": svc_t,
                "svc_p1_first_order": _table_at(fit_table, float(T), "svc_model"),
                "volvar_sim": b.vol_var,
                "volvar_se": b.vol_var_se,
                "volvar_target": vv_t,
                "volvar_p1_first_order": _table_at(fit_table, float(T), "volvar_model"),
                "corr_sim": b.correl,
                "corr_target": _target_at(targets, float(T), targets.correl_target),
                "ssr_sim": b.ssr,
            }
        )
    be_table = pd.DataFrame(be_rows)
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
    return Stage3Report(
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
    )


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

    @property
    def constraints(self) -> pd.DataFrame:
        return self.first.constraints

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
                        "ssr_lsv": _records(self.stage3.ssr_table),
                        "forward_skew": _records(self.stage3.forward_table),
                    }
                ),
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
        eps_s, eps_l = cfg.eps_pair
        lines = [
            f"fit_2f [{self.targets.mode}]{when}: status {self.status.upper()}; ssr_target "
            f"{np.round(self.targets.ssr_target, 3).tolist()}, skew_mode {cfg.skew_mode}, "
            f"skew_eps ({eps_s:g}, {eps_l:g}) at {cfg.skew_pillars}, nu_cap {cfg.nu_cap:g}",
        ]
        lines += [f"MESSAGE: {m}" for m in self.messages]
        lines += [
            f"params: nu {p.nu:.4f} theta {p.theta:.4f} k1 {p.k1:.4f} (se {f.k1_se:.3f}) k2 "
            f"{p.k2:.3f} (fixed) rho_SX1 {p.rho_SX1:+.4f} rho_SX2 {p.rho_SX2:+.4f} rho12 "
            f"{p.rho12:+.4f}",
            f"break-even: omega1 {b.omega1:.4f} omega2 {b.omega2:.4f} lambda1 {b.lambda1:+.4f} "
            f"lambda2 {b.lambda2:+.4f} chi {b.chi:+.4f}; objectives {f.objective:.3e} / "
            f"{s.objective:.3e}; active {list(f.active)}; bounds {list(s.bound_flags)}; risk "
            f"regime {self.risk_regime}; wall clock {self.wall_seconds:.1f} s; recalibrated: "
            f"{'yes' if self.recalibrated else 'no'}",
            "two-point skew constraint (naked vs market):",
            f.constraints.round(5).to_string(index=False),
            "free short-end naked skew:",
            f.short_end.round(5).to_string(index=False),
            "per pillar (first order):",
            self.table.round(5).to_string(index=False),
        ]
        notes = (*self.notes, *f.notes, *s.notes, *self.targets.flags)
        if notes:
            lines.append("notes: " + "; ".join(notes))
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
) -> FitResult:
    """Steps 2 and 3 on ``targets`` with the forward-variance curve ``xi0``, the status and
    messages, and stage 3 when ``stage3`` is given (module docstring)."""
    t0 = time.perf_counter()
    c = cfg or BreakEvenFitConfig()
    _, notes = _select_pillars(targets, c.pillars)
    first = fit_first(targets, c, xi0)
    second = fit_second(targets, c, first)
    status, messages = fit_messages(first, second, c)
    be = BreakEvenParams(
        first.k1, c.k2, second.omega1, second.omega2, first.lambda1, first.lambda2, second.chi
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
    s3 = None
    recalibrated = False
    if stage3 is not None:
        s3 = stage3_validation(params, stage3, targets, fit_table=table)
        recalibrated = s3.recalibrated
    return FitResult(
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
    )


def naked_kernel(result: FitResult, forward_curve: Any) -> Any:
    """The fitted pure-SV kernel :class:`~volsto.models.bergomi.BergomiSV` on the fit's ``ξ₀``."""
    from volsto.models.bergomi import BergomiSV

    return BergomiSV(result.params, result.xi0, forward_curve)


def fit_2f_marking(
    surface: Any,
    cfg: BreakEvenFitConfig | None = None,
    *,
    ssr_target: SsrInput = 1.0,
    anchor_power: float = 1.0,
    stage3: Stage3Inputs | None = None,
    h: float = SABR_CURVATURE_H,
) -> FitResult:
    """Marking mode on a surface: targets by :func:`~volsto.calibration.targets.marking_targets`
    (the config's pillars, ``mat_min``, SmoothBreakEven and step-0 conventions), ``ξ₀`` the
    surface's variance-swap strip to the last fitted pillar."""
    c = cfg or BreakEvenFitConfig()
    targets = marking_targets(
        surface,
        c.pillars,
        ssr_target=ssr_target,
        anchor_power=anchor_power,
        h=h,
        mat_min=c.mat_min,
        smooth_breakeven=c.smooth_breakeven,
        sabrw_power=c.sabrw_power,
        atf_ref=c.atf_ref,
    )
    t_max = float(min(surface.max_maturity, max(targets.pillars)))
    xi0 = xi0_curve(surface, t_max)
    c = replace(c, pillars=tuple(float(t) for t in targets.pillars))
    r = fit_2f(targets, xi0, c, stage3=stage3)
    return replace(r, notes=(*r.notes, *(f"target: {f}" for f in targets.flags if "dropped" in f)))


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
    pricing date's VS vols at all the history's pillars."""
    c = cfg or BreakEvenFitConfig()
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
    "DEFAULT_K2",
    "DEFAULT_NU_CAP",
    "DEFAULT_SKEW_EPS",
    "DEFAULT_SKEW_PILLARS",
    "INFEASIBLE_MESSAGE",
    "MAX_FINITE_SE",
    "NU_CAP_WARNING",
    "RISK_REGIME",
    "SKEW_MODES",
    "TERM_STRUCTURE_KINDS",
    "WEIGHT_KINDS",
    "AffineMaps",
    "BreakEvenFitConfig",
    "FirstFit",
    "FitResult",
    "FitSpec",
    "InnerSolution",
    "P1Maps",
    "PillarQuad",
    "SecondFit",
    "SkewConstraint",
    "Stage3Inputs",
    "Stage3Report",
    "TermStructureBank",
    "affine_maps",
    "affine_maps_from_kernels",
    "fit_2f",
    "fit_2f_historical",
    "fit_2f_marking",
    "fit_first",
    "fit_messages",
    "fit_second",
    "fit_spec_document",
    "k1_profile",
    "load_fit_spec",
    "mean_abs_leverage_deviation",
    "naked_kernel",
    "pillar_quad",
    "spot_skew_90_110",
    "stage3_validation",
    "term_structure_bank",
    "ts_substitution_power",
    "volvar_p1",
    "write_fit_spec",
]
