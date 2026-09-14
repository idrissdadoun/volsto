"""Break-even fit of the two-factor Bergomi model (SPEC §15 Part 3, M7 addendum; Bergomi ch. 7
§7.4, ch. 8 eq. 8.54, ch. 9 §9.2 in the break-even parametrisation of
:mod:`volsto.analytics.reparam`).

The fit works on the break-even targets of :mod:`volsto.calibration.targets` — ``SpotVolCovar``,
``VolVar`` and the market skew per pillar, in absolute vol units — through the first-order
engine of :mod:`volsto.analytics.breakeven` with the **market** ATMF vol ``atf(T)`` as the
prefactor of the sensitivities.  ``k2`` is **fixed** (default 0.2) to remove the ``(k1, k2)``
degeneracy of a term-structure fit, and the seven parameters are found in two minimisations
that never compete:

**First minimisation** — ``(k1, λ1, λ2)`` on ``SpotVolCovar``.  At fixed ``(k1, k2)`` both the
naked spot/vol break-even and the naked order-one skew are *affine* in ``(λ1, λ2)``::

    SpotVolCovar_i(λ) = ½ atf_i (λ1 A1_i + λ2 A2_i)                 (A_i of eq. 7.38)
    Skew_i(λ)         = (λ1 J1_i + λ2 J2_i) (σ̂_i / atf_i)³           (J_i of eq. 8.54)

so the inner problem is the two-variable weighted least squares ``min Σ_i w_i [SpotVolCovar_i(λ)
− SpotVolCovar_target_i]²`` under the four linear skew inequalities at a short and a long
guard maturity, ``min((1∓ε) S) ≤ Skew(λ) ≤ max((1∓ε) S)`` with ``S`` the market skew target
(negative: the band lies between ``(1+ε) S`` and ``(1−ε) S``), solved *exactly* by enumerating
the candidates of the convex 2-D QP (unconstrained optimum, one active edge, one vertex).  The
outer variable ``k1`` is minimised on a coarse geometric grid followed by a bounded scalar
refinement; the profile ``objective(k1)`` is reported (:func:`k1_profile`).  With the
correlation from SABR and ``ssr ≈ 1`` the naked skew comes out close to the market skew by
itself — the inequalities are a guard, not the driver — and the gap is reported per pillar.

**Second minimisation** — ``(ω1, ω2, χ)`` on ``VolVar`` with ``(k1, k2, λ1, λ2)`` frozen::

    VolVar_i = SensiX_i² + SensiY_i² + 2 ρ_XY SensiX_i SensiY_i,   SensiX_i = ½ ω1 A1_i atf_i

with ``ρ_Si = λ_i / ω_i`` and ``ρ_XY = ρ_SX ρ_SY + χ sqrt(1−ρ_SX²) sqrt(1−ρ_SY²)`` (the partial
correlation, so any ``|χ| ≤ 1`` gives a PSD correlation matrix); bounds ``ω_i ≥ |λ_i|`` (i.e.
``|ρ_Si| ≤ 1``), ``χ ∈ [−0.99, 0.99]`` and a soft penalty above the vol-of-vol cap ``ν ≤ 500%``
(residual ``NU_PENALTY_WEIGHT × (ν − ν_cap)⁺``); ``scipy.optimize.least_squares`` from several
starts.  The book parameters ``(ν, θ, ρ_SX, ρ_SY, ρ_XY)`` follow from the inverse
reparametrisation and the paired risk regime is recorded (``sticky_strike``: the fit is a
marking of the smile dynamics, so the risk layer pairs it with the sticky-strike bump).

**Prefactor convention** (``sigma_hat_prefactor``).  The affine maps are ``SpotVolCovar_i =
½ σ̂_i (λ1 A1 + λ2 A2)`` and ``Skew_i = λ1 J1 + λ2 J2``.  The SV skew is eq. 8.54 at the kernel's
order-zero VS vol under both conventions — the gate (``scripts/m7_breakeven_gate.py``) measured
the plain order-one skew within 6–7% of the mixing skew at ν = 1.74 and a ``(σ̂_VS/atf)³``
rescale +47–84% off, so no rescale is applied.  The conventions differ in the prefactor of the
sensitivities only: ``"market"`` (default, the owner's formula) takes the market ATMF vol
``atf_i`` (the order-two level correction of the ATMF vol supplied by the market; for the
calibrated LSV the ATMF vol *is* ``atf_i``), ``"model"`` the kernel's own order-zero ``σ̂_i``
(the VS vol of the strip).  The fit's first-order ``ssr_model = SpotVolCovar/(σ_0 Skew)`` then
reads ``(atf_i/σ̂_i) × R^{order one}`` under ``"market"`` and ``R^{order one}`` under
``"model"``; eq. 9.21 puts ``R^{order one}`` in ``[1, 2]`` (``→ 2`` as ``T → 0``), so
``ssr_target = 1`` is a floor a naked kernel reaches only at long maturities and the marking
fit at ``ssr_target = 1`` with the skew guards runs to its bounds on the reference SSVI and on
SPX (numbers in the tests and in SPEC §15) — a property of the target construction, recorded
for the owner, not of the fitter.  On the owner's recovery test (three-year mixing history of
Table 8.2, seed 13, 10⁵ mixing paths, windows 250 / 250, k2 fixed at 0.28, where the mixing
ATMF vol sits 2–7 vol points below the VS vol) ``"market"`` recovers ν 1.768, θ 0.248, k1 5.51,
ρ_SX1 −0.720, ρ_SX2 −0.482, ρ12 +0.09 (true 1.74, 0.245, 5.35, −0.759, −0.487, 0 — all inside
the owner's 10% / 0.05 tolerances) and ``"model"`` ν 1.717, θ 0.212, k1 6.63, ρ −0.766 / −0.550
(``tests/test_fit_2f.py::test_recovery_three_year_mixing``); the choice is recorded in the
provenance.  :func:`attainable_ssr` reports, per pillar, the range of first-order SSR a naked
kernel can deliver with the skew inside the guard band — the feasible dial for the owner.

**Weights.**  ``uniform`` (``w_i = 1``) or ``relative`` (``w_i = 1/target_i²``); the default
is uniform for ``SpotVolCovar`` (one order of magnitude across pillars) and relative for
``VolVar`` (which spans two).  **Standard errors** (no silent defaults): ``λ`` from the inner
linear least squares — target standard errors propagated when the target set carries them
(historical mode), else the residual variance ``s² = objective / (n − 3)``; ``k1`` from the
curvature of the ``k1`` profile (``var = s² / (½ f'')``, the profile Hessian being the Schur
complement of the full one), NaN with a note at a bound, a kink or without degrees of freedom;
``(ω1, ω2, χ)`` from the Jacobian of the second fit (``s² (JᵀJ)⁻¹``).

**Stage 3 — validation, nothing refit** (:func:`stage3_validation`): the leverage is calibrated
on the fitted parameters (or a ``model=`` override is used without calibrating) and the report
carries (a) the mean ``|L − 1|`` inside ±2 sd of log-moneyness per time and overall, (b) the
LSV's numerical SSR (state bump, :func:`~volsto.analytics.smile_dynamics.ssr_numerical_many`)
against ``ssr_target`` per pillar — self-consistency of the target construction, (c) the naked
kernel's skew by the exact mixing derivative (:func:`~volsto.calibration.history.
mixing_atmf_batch`) against the market skew, (d) the LSV's ``SpotVolCovar`` and ``VolVar`` by
**simulation** (:func:`~volsto.analytics.breakeven.simulated_breakevens`) against the targets
with standard errors and z-scores — the first-order engine is only the fit target, this is the
truth check — (e) the forward ATM vol and 90/110 skew 1y-into-1y and 2y-into-1y, and optionally
the M6 headline table.  Every report states its wall clock and whether it recalibrated.

The former ``ν ⟷ correlations`` degeneracy profile is gone by construction: ``SpotVolCovar``
fixes ``λ_i = ρ_Si ω_i`` and ``VolVar`` fixes ``ω_i``, so the scale of ``ν`` and the correlations
are separated by the two targets; :func:`k1_profile` is the remaining one-dimensional
diagnostic.  Checked by ``tests/test_fit_2f.py`` (affine maps against the engine, marking
self-consistency and policy check, the SSR dial, historical recovery on synthetic histories,
the stage-3 machinery on the cached Table 8.2 model, the rolling fit).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from scipy.optimize import least_squares, minimize_scalar

from volsto.analytics.breakeven import Kernels, _gl
from volsto.analytics.reparam import BreakEvenParams
from volsto.calibration.history import WINDOW_SSR, WINDOW_VOL, SurfaceHistory
from volsto.calibration.targets import (
    DEFAULT_TARGET_PILLARS,
    SABR_CURVATURE_H,
    TargetSet,
    historical_targets,
    marking_targets,
)
from volsto.config import BergomiParams, to_mapping
from volsto.market.varswap import ForwardVarianceCurve, xi0_curve

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

#: the fixed slow mean reversion (owner default; Table 8.2 has 0.28)
DEFAULT_K2 = 0.2
#: weight of the soft residual ``(ν − ν_cap)⁺`` in the second minimisation (per unit of ν;
#: with relative weights a 0.1 excess in ν costs as much as a 100% relative VolVar error)
NU_PENALTY_WEIGHT = 10.0
#: the risk regime paired with a break-even fit (SPEC §15 Part 3)
RISK_REGIME = "sticky_strike"
WEIGHT_KINDS = ("uniform", "relative")
PREFACTOR_KINDS = ("market", "model")
_TOL_T = 1e-9
_TOL_ACTIVE = 1e-9


# --------------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BreakEvenFitConfig:
    """Settings of the two minimisations (module docstring).

    ``pillars`` not present in the target set are dropped with a note; ``skew_guard`` is the
    pair of guard maturities of the skew inequalities (default: the first and last fitted
    pillar); ``k1_bounds[0]`` must exceed ``k2 + k1_min_gap``; ``k1_grid`` is the number of
    points of the coarse geometric grid before the bounded refinement; ``weights_svc`` /
    ``weights_volvar`` are ``"uniform"`` or ``"relative"``; ``sigma_hat_prefactor`` is
    ``"market"`` or ``"model"`` (module docstring); ``omega_max`` is the upper bound of
    the loadings in the second minimisation; ``n_quad`` / ``n_inner`` are the quadrature orders
    of :func:`volsto.analytics.breakeven.kernels`."""

    pillars: tuple[float, ...] = DEFAULT_TARGET_PILLARS
    k2: float = DEFAULT_K2
    k1_bounds: tuple[float, float] = (0.3, 20.0)
    k1_min_gap: float = 0.05
    k1_grid: int = 25
    eps_skew: float = 0.10
    skew_guard: tuple[float, float] | None = None
    weights_svc: str = "uniform"
    weights_volvar: str = "relative"
    sigma_hat_prefactor: str = "market"
    nu_cap: float = 5.0
    chi_bounds: tuple[float, float] = (-0.99, 0.99)
    omega_max: float = 20.0
    n_quad: int = 64
    n_inner: int = 32

    def __post_init__(self) -> None:
        if len(self.pillars) < 2 or any(t <= 0 for t in self.pillars):
            raise ValueError("pillars must be at least two positive maturities")
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
        if not 0.0 < self.eps_skew < 1.0:
            raise ValueError("eps_skew must lie in (0, 1)")
        if self.skew_guard is not None and (
            len(self.skew_guard) != 2 or self.skew_guard[0] >= self.skew_guard[1]
        ):
            raise ValueError("skew_guard must be (T_short, T_long) with T_short < T_long")
        for w in (self.weights_svc, self.weights_volvar):
            if w not in WEIGHT_KINDS:
                raise ValueError(f"weights must be one of {WEIGHT_KINDS}, got {w!r}")
        if self.sigma_hat_prefactor not in PREFACTOR_KINDS:
            raise ValueError(f"sigma_hat_prefactor must be one of {PREFACTOR_KINDS}")
        if self.nu_cap <= 0:
            raise ValueError("nu_cap must be positive")
        clo, chi_ = self.chi_bounds
        if not -1.0 <= clo < chi_ <= 1.0:
            raise ValueError("chi_bounds must be increasing inside [-1, 1]")
        if self.omega_max <= 0:
            raise ValueError("omega_max must be positive")


# --------------------------------------------------------------------------------------------
# kernels along k1: the quadrature of ``kernels`` precomputed once per pillar
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PillarQuad:
    """The ``k``-independent part of :func:`~volsto.analytics.breakeven.kernels` for one
    maturity of the naked model (same Gauss-Legendre nodes and inner nodes, so ``A(k)`` and
    ``J(k)`` reproduce ``kernels((k1, k2), xi0, T).A / .J`` to round-off — checked by
    ``tests/test_fit_2f.py::test_affine_maps_match_engine``)."""

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
    """The affine maps of the module docstring at fixed ``(k1, k2)`` for the fitted pillars:
    ``svc(λ) = a @ λ`` and ``skew(λ) = j @ λ`` with ``a, j`` of shape ``(n, 2)``; ``prefactor``
    names the convention and ``sig`` the vol that multiplies the sensitivities (``atf`` under
    ``"market"``, ``sigma_hat`` under ``"model"``)."""

    k1: float
    k2: float
    T: FloatArray
    atf: FloatArray
    a: FloatArray
    j: FloatArray
    A: FloatArray  # (n, 2) raw kernels
    J: FloatArray
    sigma_hat: FloatArray
    prefactor: str = "market"

    @property
    def sig(self) -> FloatArray:
        return self.atf if self.prefactor == "market" else self.sigma_hat

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
    prefactor: str,
) -> AffineMaps:
    if prefactor not in PREFACTOR_KINDS:
        raise ValueError(f"prefactor must be one of {PREFACTOR_KINDS}")
    # the SV skew is eq. 8.54 at the kernel's order-zero VS vol under both conventions (the
    # cube rescale (sig/atf)^3 was measured to move the skew the wrong way: gate table); the
    # conventions differ in the prefactor of the sensitivities only
    j = J.copy()
    pref = atf if prefactor == "market" else sig
    a = 0.5 * pref[:, None] * A
    return AffineMaps(float(k1), float(k2), T, atf, a, j, A, J, sig, prefactor)


def affine_maps(
    quads: Sequence[PillarQuad],
    atf: FloatArray,
    k1: float,
    k2: float,
    prefactor: str = "market",
) -> AffineMaps:
    """:class:`AffineMaps` from the precomputed quadratures and the market ATMF vols."""
    A = np.array([[q.A(k1), q.A(k2)] for q in quads])
    J = np.array([[q.J(k1), q.J(k2)] for q in quads])
    sig = np.array([q.sigma_hat for q in quads])
    T = np.array([q.T for q in quads])
    return _maps(k1, k2, T, atf, A, J, sig, prefactor)


def affine_maps_from_kernels(
    ks: tuple[float, float],
    kerns: Sequence[Kernels],
    atf: FloatArray,
    prefactor: str = "market",
) -> AffineMaps:
    """The same maps read from :func:`~volsto.analytics.breakeven.kernels` objects (the slow
    reference route; used by the tests to check :class:`PillarQuad`)."""
    A = np.array([k.A for k in kerns])
    J = np.array([k.J for k in kerns])
    sig = np.array([k.sigma_hat for k in kerns])
    T = np.array([k.T for k in kerns])
    return _maps(ks[0], ks[1], T, atf, A, J, sig, prefactor)


# --------------------------------------------------------------------------------------------
# the inner problem: 2-D weighted least squares under four linear inequalities
# --------------------------------------------------------------------------------------------


def _weights(kind: str, target: FloatArray) -> FloatArray:
    if kind == "uniform":
        return np.ones_like(target)
    if np.any(target == 0.0) or not np.all(np.isfinite(target)):
        raise ValueError("relative weights need finite, non-zero targets")
    return np.asarray(1.0 / target**2, dtype=np.float64)


def _qp2(
    H: FloatArray, g: FloatArray, G: FloatArray, h: FloatArray
) -> tuple[FloatArray, tuple[int, ...], bool]:
    """``argmin ½ xᵀHx − gᵀx`` over ``G x ≤ h`` for ``x ∈ R²`` by candidate enumeration
    (convex QP: the optimum is the unconstrained point, the equality optimum on one edge or a
    vertex).  Returns ``(x, active constraint indices, feasible)``; when no candidate is
    feasible (parallel guard rows) the unconstrained solution comes back with ``feasible=False``.
    """
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


@dataclass(frozen=True)
class InnerSolution:
    """The inner least squares at one ``k1``: ``λ``, the weighted objective, the active guard
    constraints (indices into ``[skew_s ≤ hi, −skew_s ≤ −lo, skew_l ≤ hi, −skew_l ≤ −lo]``),
    feasibility and the unconstrained-LS covariance of ``λ``."""

    k1: float
    lam: FloatArray
    objective: float
    active: tuple[int, ...]
    feasible: bool
    cov: FloatArray
    maps: AffineMaps


@dataclass(frozen=True)
class _FirstProblem:
    quads: tuple[PillarQuad, ...]
    T: FloatArray
    atf: FloatArray
    svc_target: FloatArray
    svc_se: FloatArray
    skew_target: FloatArray
    w: FloatArray
    guard: tuple[int, int]
    eps: float
    k2: float
    prefactor: str

    def band(self) -> tuple[FloatArray, FloatArray]:
        """``(G, h)`` of the four skew inequalities in terms of the skew rows (filled per k1)."""
        S = self.skew_target[list(self.guard)]
        lo = np.minimum((1.0 - self.eps) * S, (1.0 + self.eps) * S)
        hi = np.maximum((1.0 - self.eps) * S, (1.0 + self.eps) * S)
        return lo, hi

    def solve(self, k1: float) -> InnerSolution:
        maps = affine_maps(self.quads, self.atf, k1, self.k2, self.prefactor)
        M = maps.a
        W = self.w
        H = M.T @ (W[:, None] * M)
        g = M.T @ (W * self.svc_target)
        lo, hi = self.band()
        js = maps.j[self.guard[0]]
        jl = maps.j[self.guard[1]]
        G = np.array([js, -js, jl, -jl])
        h = np.array([hi[0], -lo[0], hi[1], -lo[1]])
        lam, active, feas = _qp2(H, g, G, h)
        r = M @ lam - self.svc_target
        obj = float(np.sum(W * r * r))
        n, p = self.T.size, 3
        try:
            Hinv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            Hinv = np.full((2, 2), np.nan)
        se = self.svc_se
        if np.all(np.isfinite(se)) and np.all(se > 0):
            cov = Hinv @ (M.T @ ((W * W * se * se)[:, None] * M)) @ Hinv
        elif n > p:
            cov = Hinv * obj / (n - p)
        else:
            cov = np.full((2, 2), np.nan)
        return InnerSolution(float(k1), lam, obj, active, feas, np.asarray(cov), maps)


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
        notes.append(f"pillars absent from the targets dropped: {dropped}")
    if len(idx) < 2:
        raise ValueError(f"the fit needs at least two pillars, {len(idx)} available")
    return np.asarray(idx, dtype=np.int64), notes


def _first_problem(
    targets: TargetSet, cfg: BreakEvenFitConfig, xi0: ForwardVarianceCurve
) -> tuple[_FirstProblem, list[str]]:
    idx, notes = _select_pillars(targets, cfg.pillars)
    T = np.asarray(targets.pillars, dtype=np.float64)[idx]
    if cfg.skew_guard is None:
        guard = (0, int(T.size - 1))
    else:
        g = []
        for tg in cfg.skew_guard:
            i = int(np.argmin(np.abs(T - tg)))
            if abs(T[i] - tg) > _TOL_T:
                raise ValueError(f"skew_guard maturity {tg:g} is not a fitted pillar")
            g.append(i)
        guard = (g[0], g[1])
    quads = tuple(pillar_quad(xi0, float(t), n_quad=cfg.n_quad, n_inner=cfg.n_inner) for t in T)
    svc = np.asarray(targets.spot_vol_covar, dtype=np.float64)[idx]
    prob = _FirstProblem(
        quads,
        T,
        np.asarray(targets.atf, dtype=np.float64)[idx],
        svc,
        np.asarray(targets.spot_vol_covar_se, dtype=np.float64)[idx],
        np.asarray(targets.skew_target, dtype=np.float64)[idx],
        _weights(cfg.weights_svc, svc),
        guard,
        cfg.eps_skew,
        cfg.k2,
        cfg.sigma_hat_prefactor,
    )
    return prob, notes


def _k1_grid(cfg: BreakEvenFitConfig) -> FloatArray:
    lo, hi = cfg.k1_bounds
    return np.asarray(np.geomspace(lo, hi, cfg.k1_grid), dtype=np.float64)


def k1_profile(
    targets: TargetSet,
    cfg: BreakEvenFitConfig,
    xi0: ForwardVarianceCurve,
    k1s: Sequence[float] | FloatArray | None = None,
) -> pd.DataFrame:
    """The first-minimisation objective along ``k1`` (default: the config's coarse grid) with
    the inner optimum per point: columns ``k1, objective, lambda1, lambda2, n_active,
    feasible``.  This replaces the former ``ν ⟷ correlations`` degeneracy profile, which the
    break-even method removes by construction (module docstring)."""
    prob, _ = _first_problem(targets, cfg, xi0)
    grid = _k1_grid(cfg) if k1s is None else np.asarray(k1s, dtype=np.float64)
    rows = []
    for k1 in grid:
        s = prob.solve(float(k1))
        rows.append(
            {
                "k1": float(k1),
                "objective": s.objective,
                "lambda1": float(s.lam[0]),
                "lambda2": float(s.lam[1]),
                "n_active": len(s.active),
                "feasible": s.feasible,
            }
        )
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class FirstFit:
    """Result of the first minimisation: ``(k1, λ1, λ2)``, the weighted objective, the standard
    errors, the active skew constraints, the ``k1`` profile and the per-pillar table (``T, atf,
    svc_target, svc_model, skew_target, skew_naked, skew_gap`` — the gap relative to the
    target — and ``ssr_model = svc_model / (σ_0 skew_naked)``)."""

    k1: float
    lambda1: float
    lambda2: float
    objective: float
    k1_se: float
    lambda_cov: FloatArray
    active: tuple[int, ...]
    feasible: bool
    k1_at_bound: bool
    profile: pd.DataFrame
    table: pd.DataFrame
    maps: AffineMaps
    guard: tuple[float, float]
    notes: tuple[str, ...]
    wall_seconds: float

    @property
    def lambda1_se(self) -> float:
        return float(np.sqrt(self.lambda_cov[0, 0]))

    @property
    def lambda2_se(self) -> float:
        return float(np.sqrt(self.lambda_cov[1, 1]))

    @property
    def active_labels(self) -> tuple[str, ...]:
        names = ("skew_short_upper", "skew_short_lower", "skew_long_upper", "skew_long_lower")
        return tuple(names[i] for i in self.active)


def fit_first(targets: TargetSet, cfg: BreakEvenFitConfig, xi0: ForwardVarianceCurve) -> FirstFit:
    """The first minimisation (module docstring): coarse geometric grid in ``k1``, bounded
    scalar refinement around the best grid point, exact inner QP."""
    t0 = time.perf_counter()
    prob, notes = _first_problem(targets, cfg, xi0)
    grid = _k1_grid(cfg)
    sols = [prob.solve(float(k)) for k in grid]
    objs = np.array([s.objective for s in sols])
    i = int(np.argmin(objs))
    lo = float(grid[max(i - 1, 0)])
    hi = float(grid[min(i + 1, grid.size - 1)])
    best = sols[i]
    if hi > lo:
        res = minimize_scalar(
            lambda k: prob.solve(float(k)).objective,
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-5 * hi},
        )
        cand = prob.solve(float(res.x))
        if cand.objective <= best.objective:
            best = cand
    k1 = best.k1
    at_bound = bool(k1 <= cfg.k1_bounds[0] * (1 + 1e-6) or k1 >= cfg.k1_bounds[1] * (1 - 1e-6))
    if at_bound:
        notes.append(f"k1 = {k1:.4g} sits on a bound of {cfg.k1_bounds}")
    if best.active:
        notes.append(f"skew guard active: {best.active}")
    if not best.feasible:
        notes.append("skew guard infeasible (parallel guard rows): unconstrained lambda kept")
    n, p = prob.T.size, 3
    s2 = best.objective / (n - p) if n > p else float("nan")
    k1_se, se_note = _k1_curvature_se(prob, k1, cfg.k1_bounds, s2, bool(best.active))
    if se_note:
        notes.append(se_note)
    if n <= p and not (np.all(np.isfinite(prob.svc_se)) and np.all(prob.svc_se > 0)):
        notes.append(f"{n} pillars for 3 parameters: no residual-based standard errors")
    profile = pd.DataFrame(
        {
            "k1": grid,
            "objective": objs,
            "lambda1": [float(s.lam[0]) for s in sols],
            "lambda2": [float(s.lam[1]) for s in sols],
            "n_active": [len(s.active) for s in sols],
            "feasible": [s.feasible for s in sols],
        }
    )
    maps = best.maps
    svc_model = maps.svc(best.lam)
    skew_naked = maps.skew(best.lam)
    table = pd.DataFrame(
        {
            "T": prob.T,
            "atf": prob.atf,
            "svc_target": prob.svc_target,
            "svc_model": svc_model,
            "skew_target": prob.skew_target,
            "skew_naked": skew_naked,
            "skew_gap": skew_naked / prob.skew_target - 1.0,
            "ssr_model": svc_model / (targets.sigma_0 * skew_naked),
        }
    )
    return FirstFit(
        k1,
        float(best.lam[0]),
        float(best.lam[1]),
        best.objective,
        k1_se,
        best.cov,
        best.active,
        best.feasible,
        at_bound,
        profile,
        table,
        maps,
        (float(prob.T[prob.guard[0]]), float(prob.T[prob.guard[1]])),
        tuple(notes),
        time.perf_counter() - t0,
    )


def _k1_curvature_se(
    prob: _FirstProblem,
    k1: float,
    bounds: tuple[float, float],
    s2: float,
    constrained: bool,
) -> tuple[float, str]:
    """``sqrt(s² / (½ f''))`` from central differences of the profile (module docstring)."""
    if not np.isfinite(s2):
        return float("nan"), "k1 standard error: no degrees of freedom"
    h = max(1e-3, 0.02 * k1)
    if k1 - h < bounds[0] or k1 + h > bounds[1]:
        return float("nan"), "k1 standard error: k1 at a bound, no curvature"
    f0 = prob.solve(k1).objective
    fa = prob.solve(k1 - h).objective
    fb = prob.solve(k1 + h).objective
    curv = (fa - 2.0 * f0 + fb) / (h * h)
    if not np.isfinite(curv) or curv <= 0:
        return float("nan"), "k1 standard error: non-positive profile curvature"
    note = "k1 standard error from a profile with active skew constraints (kink possible)"
    return float(np.sqrt(s2 / (0.5 * curv))), note if constrained else ""


# --------------------------------------------------------------------------------------------
# second minimisation
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SecondFit:
    """Result of the second minimisation: ``(ω1, ω2, χ)``, the weighted objective, standard
    errors from the Jacobian, bound flags, the number of starts and distinct optima, the
    per-pillar table (``T, volvar_target, volvar_model, vovol_target, vovol_model``)."""

    omega1: float
    omega2: float
    chi: float
    objective: float
    stderr: dict[str, float]
    bound_flags: tuple[str, ...]
    n_starts: int
    n_distinct: int
    nu_penalised: bool
    table: pd.DataFrame
    notes: tuple[str, ...]
    wall_seconds: float


def _volvar_model(
    x: FloatArray, lam: FloatArray, A: FloatArray, sig: FloatArray
) -> tuple[FloatArray, float]:
    """``VolVar`` per pillar and ``ν`` for ``x = (ω1, ω2, χ)`` at fixed ``λ`` (module
    docstring); ``sig`` is the prefactor vol of the sensitivities."""
    om1, om2, chi = float(x[0]), float(x[1]), float(x[2])
    r1 = lam[0] / om1
    r2 = lam[1] / om2
    rxy = r1 * r2 + chi * math.sqrt(max(1.0 - r1 * r1, 0.0)) * math.sqrt(max(1.0 - r2 * r2, 0.0))
    sx = 0.5 * om1 * A[:, 0] * sig
    sy = 0.5 * om2 * A[:, 1] * sig
    vv = sx * sx + sy * sy + 2.0 * rxy * sx * sy
    th = om2 / (om1 + om2)
    alpha = 1.0 / math.sqrt((1 - th) ** 2 + th * th + 2.0 * rxy * th * (1 - th))
    return np.asarray(vv, dtype=np.float64), (om1 + om2) / (2.0 * alpha)


def fit_second(targets: TargetSet, cfg: BreakEvenFitConfig, first: FirstFit) -> SecondFit:
    """The second minimisation (module docstring) from several starts (``ω_i = max(|λ_i|/0.7,
    0.05)`` scaled by 0.5, 1, 2 and ``χ ∈ {0, −0.6, +0.6}``); the best optimum is kept and the
    number of distinct optima (objective within 1% and parameters within 1e-3) reported."""
    t0 = time.perf_counter()
    idx, _ = _select_pillars(targets, cfg.pillars)
    vv_t = np.asarray(targets.vol_var, dtype=np.float64)[idx]
    vv_se = np.asarray(targets.vol_var_se, dtype=np.float64)[idx]
    w = _weights(cfg.weights_volvar, vv_t)
    sw = np.sqrt(w)
    lam = np.array([first.lambda1, first.lambda2])
    A = first.maps.A
    atf = first.maps.atf
    sig = first.maps.sig
    lower = np.array([abs(lam[0]) + 1e-9, abs(lam[1]) + 1e-9, cfg.chi_bounds[0]])
    upper = np.array([cfg.omega_max, cfg.omega_max, cfg.chi_bounds[1]])
    if np.any(lower >= upper):
        raise ValueError("|lambda_i| exceeds omega_max: enlarge omega_max")

    def resid(x: FloatArray) -> FloatArray:
        vv, nu = _volvar_model(x, lam, A, sig)
        r = sw * (vv - vv_t)
        pen = NU_PENALTY_WEIGHT * max(nu - cfg.nu_cap, 0.0)
        return np.concatenate((r, [pen]))

    base = np.maximum(np.abs(lam) / 0.7, 0.05)
    starts = []
    for scale in (1.0, 2.0, 0.5):
        for chi0 in (0.0, -0.6, 0.6):
            x0 = np.array([base[0] * scale, base[1] * scale, chi0])
            starts.append(np.clip(x0, lower * (1 + 1e-6) + 1e-9, upper * (1 - 1e-6)))
    results = []
    for x0 in starts:
        res = least_squares(resid, x0, bounds=(lower, upper), method="trf", x_scale="jac")
        results.append(res)
    results.sort(key=lambda r: float(r.cost))
    best = results[0]
    x = np.asarray(best.x, dtype=np.float64)
    distinct = 1
    for r in results[1:]:
        if abs(r.cost - best.cost) <= 0.01 * max(best.cost, 1e-12) and not np.allclose(
            r.x, best.x, atol=1e-3, rtol=1e-3
        ):
            distinct += 1
    vv_m, nu = _volvar_model(x, lam, A, sig)
    r_fit = sw * (vv_m - vv_t)
    objective = float(np.sum(r_fit * r_fit))
    n, p = vv_t.size, 3
    Jm = np.asarray(best.jac, dtype=np.float64)[:n]
    names = ("omega1", "omega2", "chi")
    stderr: dict[str, float] = dict.fromkeys(names, float("nan"))
    notes: list[str] = []
    try:
        JtJ_inv = np.linalg.inv(Jm.T @ Jm)
        if np.all(np.isfinite(vv_se)) and np.all(vv_se > 0):
            cov = JtJ_inv @ (Jm.T @ ((w * vv_se * vv_se)[:, None] * Jm)) @ JtJ_inv
        elif n > p:
            cov = JtJ_inv * objective / (n - p)
        else:
            cov = np.full((3, 3), np.nan)
            notes.append(f"{n} pillars for 3 parameters: no residual-based standard errors")
        stderr = {nm: float(np.sqrt(cov[i, i])) for i, nm in enumerate(names)}
    except np.linalg.LinAlgError:
        notes.append("singular Jacobian in the second fit: no standard errors")
    flags = []
    for i, nm in enumerate(names):
        if x[i] <= lower[i] * (1 + 1e-6) + 1e-9:
            flags.append(f"{nm} at lower bound")
        if x[i] >= upper[i] * (1 - 1e-6):
            flags.append(f"{nm} at upper bound")
    penalised = nu > cfg.nu_cap + 1e-9
    if penalised:
        notes.append(f"nu {nu:.3f} above the cap {cfg.nu_cap:g}: penalty active")
    table = pd.DataFrame(
        {
            "T": first.maps.T,
            "volvar_target": vv_t,
            "volvar_model": vv_m,
            "vovol_target": np.sqrt(vv_t) / atf,
            "vovol_model": np.sqrt(vv_m) / atf,
        }
    )
    return SecondFit(
        float(x[0]),
        float(x[1]),
        float(x[2]),
        objective,
        stderr,
        tuple(flags),
        len(starts),
        distinct,
        bool(penalised),
        table,
        tuple(notes),
        time.perf_counter() - t0,
    )


# --------------------------------------------------------------------------------------------
# stage 3
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage3Inputs:
    """What the validation needs beyond the fit: the pricing date's implied surface, the
    particle / simulation settings, the optional local-vol config, the pricing configuration,
    the pillars of each check and, optionally, an already calibrated ``model`` (then nothing
    is calibrated and ``recalibrated`` is ``False``; the model's own kernel is used for the
    naked-skew check).  ``mixing_paths`` / ``mixing_seed`` / ``mixing_dt`` set the exact
    mixing derivative of check (c)."""

    surface: Any  # ImpliedSurface
    particle: Any  # ParticleConfig
    sim: Any  # SimConfig (calibration schedule)
    pricing_sim: Any  # SimConfig
    local_vol: Any | None = None
    ssr_pillars: tuple[float, ...] = (0.25, 1.0)
    breakeven_pillars: tuple[float, ...] = (0.25, 1.0)
    forward_starts: tuple[tuple[float, float], ...] = ((1.0, 2.0), (2.0, 3.0))
    headline: bool = False
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
    headline: pd.DataFrame | None
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
            "numerical SSR of the LSV vs ssr_target:",
            self.ssr_table.round(4).to_string(index=False),
            "naked skew (mixing derivative) vs market skew:",
            self.skew_table.round(4).to_string(index=False),
            "simulated break-evens of the LSV vs targets:",
            self.breakeven_table.round(5).to_string(index=False),
        ]
        if len(self.forward_table):
            lines += ["forward smiles:", self.forward_table.round(4).to_string(index=False)]
        return "\n".join(lines)


def _target_at(targets: TargetSet, T: float, values: FloatArray) -> float:
    tp = np.asarray(targets.pillars, dtype=np.float64)
    i = int(np.argmin(np.abs(tp - T)))
    return float(values[i]) if abs(tp[i] - T) <= _TOL_T else float("nan")


def stage3_validation(
    params: BergomiParams, inputs: Stage3Inputs, targets: TargetSet
) -> Stage3Report:
    """Stage 3 of the module docstring: calibrate the leverage on ``inputs.surface`` for
    ``params`` (or take ``inputs.model``) and report the validation tables; nothing is refit."""
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
    lev = lsv.leverage
    # (a) mean |L - 1| over the leverage grid inside +/-2 sd of log-moneyness per time
    rows = []
    for i, t in enumerate(lev.times):
        if t <= 0:
            continue
        sd = float(surface.atm_vol(t)) * np.sqrt(t)
        inside = np.abs(lev.k_grid) <= 2.0 * sd
        vals = lev.values[i][inside]
        rows.append(
            {
                "t": float(t),
                "mean_abs_L_minus_1": float(np.mean(np.abs(vals - 1.0))),
                "n": int(inside.sum()),
            }
        )
    lev_table = pd.DataFrame(rows)
    mean_abs = float(lev_table["mean_abs_L_minus_1"].mean()) if len(lev_table) else float("nan")
    psim = inputs.pricing_sim
    # (b) numerical SSR of the LSV vs ssr_target
    ssr_rows = ssr_numerical_many(lsv, list(inputs.ssr_pillars), eps=inputs.eps, sim=psim)
    ssr_t = np.array([_target_at(targets, r.T, targets.ssr_target) for r in ssr_rows])
    ssr_m = np.array([r.R for r in ssr_rows])
    ssr_se = np.array([r.R_stderr for r in ssr_rows])
    ssr_table = pd.DataFrame(
        {
            "T": [r.T for r in ssr_rows],
            "ssr_model": ssr_m,
            "ssr_model_se": ssr_se,
            "ssr_target": ssr_t,
            "z": (ssr_m - ssr_t) / ssr_se,
            "skew_lsv": [r.skew for r in ssr_rows],
            "skew_lsv_se": [r.skew_stderr for r in ssr_rows],
        }
    )
    # (c) naked skew (exact mixing derivative) vs the market skew target
    nf = int(kernel.n_factors)
    skew_rows = []
    for T in inputs.breakeven_pillars:
        st = _target_at(targets, float(T), targets.skew_target)
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
            sk, sk_se, av, note = float(mb.skew[0]), float(mb.skew_se[0]), float(mb.atm_vol[0]), ""
        except ValueError as exc:  # the mixing solution needs |spot/vol correlation| < 1
            sk, sk_se, av, note = float("nan"), float("nan"), float("nan"), str(exc)
        skew_rows.append(
            {
                "T": float(T),
                "skew_naked": sk,
                "skew_naked_se": sk_se,
                "skew_target": st,
                "gap": sk / st - 1.0 if st else float("nan"),
                "atmf_vol_naked": av,
                "note": note,
            }
        )
    skew_table = pd.DataFrame(skew_rows)
    # (d) simulated break-evens of the LSV vs the targets
    be_rows = []
    for T in inputs.breakeven_pillars:
        b = simulated_breakevens(lsv, float(T), sim=psim, eps=inputs.eps, sigma_0=targets.sigma_0)
        svc_t = _target_at(targets, float(T), targets.spot_vol_covar)
        vv_t = _target_at(targets, float(T), targets.vol_var)
        atf_t = _target_at(targets, float(T), targets.atf)
        be_rows.append(
            {
                "T": float(T),
                "svc_sim": b.spot_vol_covar,
                "svc_se": b.spot_vol_covar_se,
                "svc_target": svc_t,
                "svc_z": (
                    (b.spot_vol_covar - svc_t) / b.spot_vol_covar_se
                    if b.spot_vol_covar_se > 0
                    else float("nan")
                ),
                "volvar_sim": b.vol_var,
                "volvar_se": b.vol_var_se,
                "volvar_target": vv_t,
                "volvar_z": (b.vol_var - vv_t) / b.vol_var_se if b.vol_var_se > 0 else float("nan"),
                "vovol_sim": b.vovol,
                "vovol_target": np.sqrt(vv_t) / atf_t if atf_t > 0 else float("nan"),
                "ssr_sim": b.ssr,
                "atmf_vol_sim": b.sigma_hat,
            }
        )
    be_table = pd.DataFrame(be_rows)
    # (e) forward smiles
    fwd_rows = []
    for t1, t2 in inputs.forward_starts:
        sm = forward_smile(lsv, t1, t2, [0.9, 1.1], psim)  # the ATM-forward strike is added
        i_atm = int(np.argmin(np.abs(sm.strikes - sm.forward_ratio)))
        i_lo = int(np.argmin(np.abs(sm.strikes - 0.9)))
        i_hi = int(np.argmin(np.abs(sm.strikes - 1.1)))
        fwd_rows.append(
            {
                "t1": t1,
                "t2": t2,
                "atm_fwd_vol": float(sm.vols[i_atm]),
                "atm_fwd_vol_se": float(sm.vol_stderr[i_atm]),
                "skew_90_110": float(sm.vols[i_lo] - sm.vols[i_hi]),
                "skew_90_110_se": float(np.hypot(sm.vol_stderr[i_lo], sm.vol_stderr[i_hi])),
            }
        )
    forward_table = pd.DataFrame(fwd_rows)
    headline = None
    if inputs.headline:
        from volsto.studies.m6 import run_m6_headline

        headline = run_m6_headline({"fit": lsv}, psim).table
    return Stage3Report(
        mean_abs,
        lev_table,
        ssr_table,
        skew_table,
        be_table,
        forward_table,
        headline,
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
    """The fit: book and break-even parameters, the targets, both minimisations, the per-pillar
    table (``T, atf, svc_target, svc_model, volvar_target, volvar_model, skew_target,
    skew_naked, skew_gap, ssr_target, ssr_model, vovol_target, vovol_model``), the paired risk
    regime, the optional stage 3, the wall clock and whether a leverage was recalibrated."""

    params: BergomiParams
    breakeven: BreakEvenParams
    targets: TargetSet
    xi0: ForwardVarianceCurve
    config: BreakEvenFitConfig
    first: FirstFit
    second: SecondFit
    table: pd.DataFrame
    risk_regime: str
    stage3: Stage3Report | None
    wall_seconds: float
    recalibrated: bool
    pricing_date: pd.Timestamp | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def config_yaml(self) -> str:
        """A loadable model config: ``model`` (:class:`BergomiParams`, ``load_yaml(path,
        BergomiParams, section="model")``), ``breakeven`` (:class:`BreakEvenParams`), the
        ``risk_regime`` and the provenance (mode, targets, settings, objectives, standard
        errors, flags)."""
        f, s = self.first, self.second
        tg = self.targets
        doc: dict[str, Any] = {
            "model": to_mapping(self.params),
            "breakeven": to_mapping(self.breakeven),
            "risk_regime": self.risk_regime,
            "provenance": {
                "mode": tg.mode,
                "pricing_date": (
                    None if self.pricing_date is None else str(self.pricing_date.date())
                ),
                "pillars": [float(t) for t in f.maps.T],
                "sigma_0": float(tg.sigma_0),
                "ssr_target": [float(x) for x in tg.ssr_target],
                "anchor_power": (
                    None if not np.isfinite(tg.anchor_power) else float(tg.anchor_power)
                ),
                "k2_fixed": float(self.config.k2),
                "k1_bounds": [float(x) for x in self.config.k1_bounds],
                "eps_skew": float(self.config.eps_skew),
                "skew_guard": [float(x) for x in f.guard],
                "weights": {"svc": self.config.weights_svc, "volvar": self.config.weights_volvar},
                "nu_cap": float(self.config.nu_cap),
                "sigma_hat_prefactor": self.config.sigma_hat_prefactor,
                "first": {
                    "objective": float(f.objective),
                    "k1_se": float(f.k1_se),
                    "lambda1_se": float(f.lambda1_se),
                    "lambda2_se": float(f.lambda2_se),
                    "active_constraints": list(f.active_labels),
                    "feasible": bool(f.feasible),
                    "k1_at_bound": bool(f.k1_at_bound),
                    "notes": list(f.notes),
                },
                "second": {
                    "objective": float(s.objective),
                    "stderr": {k: float(v) for k, v in s.stderr.items()},
                    "bound_flags": list(s.bound_flags),
                    "n_starts": int(s.n_starts),
                    "n_distinct_optima": int(s.n_distinct),
                    "nu_penalised": bool(s.nu_penalised),
                    "notes": list(s.notes),
                },
                "target_flags": list(tg.flags),
                "notes": list(self.notes),
                "stage3_mean_abs_L_minus_1": (
                    None if self.stage3 is None else float(self.stage3.mean_abs_l_minus_1)
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
        p, b, f, s = self.params, self.breakeven, self.first, self.second
        when = "" if self.pricing_date is None else f" @ {self.pricing_date.date()}"
        lines = [
            f"fit_2f [{self.targets.mode}]{when}: nu {p.nu:.3f} theta {p.theta:.3f} "
            f"k1 {p.k1:.3f} (se {f.k1_se:.3f}) k2 {p.k2:.3f} (fixed) rho12 {p.rho12:+.3f} "
            f"rho_SX1 {p.rho_SX1:+.3f} rho_SX2 {p.rho_SX2:+.3f}; "
            f"break-even omega1 {b.omega1:.3f} omega2 {b.omega2:.3f} "
            f"lambda1 {b.lambda1:+.3f} (se {f.lambda1_se:.3f}) lambda2 {b.lambda2:+.3f} "
            f"(se {f.lambda2_se:.3f}) chi {b.chi:+.3f} (se {s.stderr['chi']:.3f}); "
            f"objectives {f.objective:.3e} / {s.objective:.3e}; active {list(f.active_labels)}; "
            f"bounds {list(s.bound_flags)}; risk regime {self.risk_regime}; "
            f"wall clock {self.wall_seconds:.1f} s; recalibrated: "
            f"{'yes' if self.recalibrated else 'no'}",
            self.table.round(5).to_string(index=False),
        ]
        if self.notes or f.notes or s.notes:
            lines.append("notes: " + "; ".join((*self.notes, *f.notes, *s.notes)))
        if self.stage3 is not None:
            lines.append(self.stage3.summary())
        return "\n".join(lines)


def attainable_ssr(
    targets: TargetSet,
    cfg: BreakEvenFitConfig | None = None,
    xi0: ForwardVarianceCurve | None = None,
    *,
    k1s: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Per pillar, the range of the first-order ``SSR = SpotVolCovar / (σ_0 Skew_target)`` a
    naked kernel can deliver while its skew sits inside the guard band at the two guard
    maturities: for each ``k1`` of the grid the two guard equalities pin ``(λ1, λ2)`` (any
    ``(1 ± ε)`` corner), the SSR at every pillar follows; columns ``T, ssr_min, ssr_max,
    ssr_at_skew_match_min / _max`` (both guards exactly on the target skew) over the ``k1``
    grid, over the loadings on the skew's side (``sign λ_i = sign Skew``: a loading against the
    skew is not a marking candidate).  The marking dial is attainable only inside ``[ssr_min,
    ssr_max]`` — on the reference SSVI the band at an exact skew match is 1.24–1.35 at 3M and
    1.03–1.43 at 3Y, the guard corners reach 0.92 only at 3Y (the structural floor of eq. 9.21,
    ``→ 2`` at short T), which is why ``ssr_target = 1`` runs the fit to its bounds."""
    c = cfg or BreakEvenFitConfig()
    if xi0 is None:
        raise ValueError("attainable_ssr needs the forward-variance curve xi0 of the targets")
    prob, _notes = _first_problem(targets, c, xi0)
    ks = np.asarray(k1s, dtype=np.float64) if k1s is not None else _k1_grid(c)
    T = prob.T
    n = T.size
    i_s, i_l = prob.guard
    sign = float(np.sign(np.mean(prob.skew_target)))  # both loadings on the skew's side
    rows = []
    for k1 in ks:
        maps = affine_maps(prob.quads, prob.atf, float(k1), prob.k2, prob.prefactor)
        Js = maps.j[[i_s, i_l]]
        for f_s in (1.0 - c.eps_skew, 1.0, 1.0 + c.eps_skew):
            for f_l in (1.0 - c.eps_skew, 1.0, 1.0 + c.eps_skew):
                rhs = np.array([f_s * prob.skew_target[i_s], f_l * prob.skew_target[i_l]])
                try:
                    lam = np.linalg.solve(Js, rhs)
                except np.linalg.LinAlgError:
                    continue
                if np.any(np.sign(lam) * sign != 0) and np.any(np.sign(lam) != sign):
                    continue  # a loading against the skew's sign: not a marking candidate
                svc = maps.a @ lam
                ssr = svc / (targets.sigma_0 * prob.skew_target)
                rows.append(
                    {
                        "k1": float(k1),
                        "f_s": f_s,
                        "f_l": f_l,
                        **{f"ssr_{j}": ssr[j] for j in range(n)},
                    }
                )
    df = pd.DataFrame(rows)
    out = []
    for j in range(n):
        col = df[f"ssr_{j}"]
        exact = df[(df["f_s"] == 1.0) & (df["f_l"] == 1.0)][f"ssr_{j}"]
        out.append(
            {
                "T": float(T[j]),
                "ssr_min": float(col.min()),
                "ssr_max": float(col.max()),
                "ssr_at_skew_match_min": float(exact.min()),
                "ssr_at_skew_match_max": float(exact.max()),
            }
        )
    return pd.DataFrame(out)


def fit_2f(
    targets: TargetSet,
    xi0: ForwardVarianceCurve,
    cfg: BreakEvenFitConfig | None = None,
    *,
    stage3: Stage3Inputs | None = None,
    pricing_date: pd.Timestamp | None = None,
) -> FitResult:
    """Both minimisations on ``targets`` with the forward-variance curve ``xi0`` (and stage 3
    when ``stage3`` is given)."""
    c = cfg or BreakEvenFitConfig()
    t0 = time.perf_counter()
    first = fit_first(targets, c, xi0)
    second = fit_second(targets, c, first)
    be = BreakEvenParams(
        first.k1, c.k2, second.omega1, second.omega2, first.lambda1, first.lambda2, second.chi
    )
    params = be.to_book()
    ft, st = first.table, second.table
    idx, notes = _select_pillars(targets, c.pillars)
    ssr_t = np.asarray(targets.ssr_target, dtype=np.float64)[idx]
    table = pd.DataFrame(
        {
            "T": ft["T"],
            "atf": ft["atf"],
            "svc_target": ft["svc_target"],
            "svc_model": ft["svc_model"],
            "volvar_target": st["volvar_target"],
            "volvar_model": st["volvar_model"],
            "skew_target": ft["skew_target"],
            "skew_naked": ft["skew_naked"],
            "skew_gap": ft["skew_gap"],
            "ssr_target": ssr_t,
            "ssr_model": ft["ssr_model"],
            "vovol_target": st["vovol_target"],
            "vovol_model": st["vovol_model"],
        }
    )
    s3 = None
    recalibrated = False
    if stage3 is not None:
        s3 = stage3_validation(params, stage3, targets)
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
        RISK_REGIME,
        s3,
        time.perf_counter() - t0,
        recalibrated,
        pricing_date,
        tuple(notes),
    )


def naked_kernel(result: FitResult, forward_curve: Any) -> Any:
    """The fitted pure-SV kernel :class:`~volsto.models.bergomi.BergomiSV` on the fit's ``ξ₀``
    (for the numerical SSR / mixing checks of the naked model)."""
    from volsto.models.bergomi import BergomiSV

    return BergomiSV(result.params, result.xi0, forward_curve)


def fit_2f_marking(
    surface: Any,
    cfg: BreakEvenFitConfig | None = None,
    *,
    ssr_target: float | Mapping[float, float] | Callable[[float], float] = 1.0,
    anchor_power: float = 1.0,
    stage3: Stage3Inputs | None = None,
    h: float = SABR_CURVATURE_H,
) -> FitResult:
    """Marking mode on a surface: targets by :func:`~volsto.calibration.targets.
    marking_targets` (pillars of the config inside the surface's range), ``ξ₀`` the surface's
    variance-swap strip to the last fitted pillar."""
    c = cfg or BreakEvenFitConfig()
    targets = marking_targets(
        surface, c.pillars, ssr_target=ssr_target, anchor_power=anchor_power, h=h
    )
    t_max = float(min(surface.max_maturity, max(targets.pillars)))
    xi0 = xi0_curve(surface, t_max)
    return fit_2f(targets, xi0, c, stage3=stage3)


def _curve_from_vs(pillars: FloatArray, vs_vol: FloatArray) -> ForwardVarianceCurve:
    """Forward-variance curve through the pillar VS vols (total variance ``T σ̂_T²`` interpolated
    by :class:`ForwardVarianceCurve`, flat VS vol before the first and beyond the last pillar,
    made monotone by a tiny ramp)."""
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
    on the config's pillars present in the history, ``ξ₀`` from the pricing date's VS vols at
    all the history's pillars (:func:`_curve_from_vs`)."""
    c = cfg or BreakEvenFitConfig()
    e = history.date_index(end)
    end_ts = history.dates[e]
    hp = np.asarray(history.pillars, dtype=np.float64)
    pillars = [float(T) for T in c.pillars if np.any(np.abs(hp - float(T)) <= _TOL_T)]
    if len(pillars) < 2:
        raise ValueError(f"fewer than two of the config pillars {c.pillars} are in the history")
    targets = historical_targets(
        history, pillars, end=end_ts, window_vol=window_vol, window_ssr=window_ssr
    )
    xi0 = _curve_from_vs(hp, np.asarray(history.vs_vol.to_numpy()[e], dtype=np.float64))
    return fit_2f(targets, xi0, c, stage3=stage3, pricing_date=end_ts)


__all__ = [
    "DEFAULT_K2",
    "NU_PENALTY_WEIGHT",
    "PREFACTOR_KINDS",
    "RISK_REGIME",
    "WEIGHT_KINDS",
    "AffineMaps",
    "BreakEvenFitConfig",
    "FirstFit",
    "FitResult",
    "InnerSolution",
    "PillarQuad",
    "SecondFit",
    "Stage3Inputs",
    "Stage3Report",
    "affine_maps",
    "affine_maps_from_kernels",
    "attainable_ssr",
    "fit_2f",
    "fit_2f_historical",
    "fit_2f_marking",
    "fit_first",
    "fit_second",
    "k1_profile",
    "naked_kernel",
    "pillar_quad",
    "stage3_validation",
]
