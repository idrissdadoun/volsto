"""Break-even targets of the P1 (two-factor LSV) marking calibration (SPEC §15 Part 3, owner's
"M7 Part 3 FINAL" specification): the SABR reduction of a surface pillar (step 0), the
SABR break-evens (step 1) and the historical alternative.

**Step 0 — SABR reduction per pillar** (library conventions; the owner: "this library is its
OWN system ... its own conventions", no production value to match).  Maturities are ACT/365
years (``T_365 = T``); log-moneyness ``k = ln(K/F)``.  From the surface, per pillar ``T``, the
library's 365-day quotes (``x = k/√T`` the normalised log-moneyness)::

    Atf_365    = 100 σ̂(0, T)                      (vol points)
    Smile_365  = 100 · 2 · ∂σ̂/∂x = 100 · 2 √T · ∂σ̂/∂k
    Convex_365 = 100 · ∂²σ̂/∂x² = 100 · T · ∂²σ̂/∂k²

converted to physical with the owner's formulas::

    atf_T = Atf_365 / 100
    smi_T = Smile_365 / (100 · 2 · √T)               = ∂σ̂/∂k           (the ATM skew)
    cvx_T = Convex_365 / (100 · T · (atf_T/atf_ref)^p) = ∂²σ̂/∂k² (atf_ref/atf_T)^p

``atf_ref = 0.3`` and ``p = SabrW_Power`` (config, default 1 — the library's convexity-rescaling
convention; ``p = 0`` is the plain Hagan inversion).  Then::

    ν_SABR(T)    = sqrt(6 smi_T² + 3 atf_T cvx_T)          lognormal vol of vol of the ATM vol
    VoV_SABR(T)  = atf_T ν_SABR(T)                          ABSOLUTE vol of vol (confirmed)
    Corr_SABR(T) = 2 smi_T / ν_SABR(T)                      (= ρ_SABR, negative for equities)
    Skew_SABR(T) = smi_T

*Reading of the owner's formulas (flagged in the report):* written with ``smi, cvx`` in the units
above, ``sqrt(6 smi² + 3 atf cvx)`` is the lognormal ``ν_SABR`` (Hagan β = 1 at order one: ``σ̂(k) ≈
atf + ½ρν k + (2 − 3ρ²)ν²/(12 atf) k²`` inverts to exactly this at ``p = 0``); the owner's confirmed
fact "VoV_SABR is the ABSOLUTE vol of vol (ν_SABR · atf)" is the same formula in absolute smile
units ``smi_abs = atf smi``, ``cvx_abs = atf² cvx``, and the owner's ``Corr_SABR = −2 smi /
VoV_SABR`` is that formula with the smile quoted positive for a put skew (``smi_abs = −atf
∂σ̂/∂k``).  The library keeps ``Skew_SABR = ∂σ̂/∂k`` (negative), the units of the naked model skew
of book eq.  8.54 it is compared with, and ``Corr_SABR = ρ_SABR``.  With ``p = 1`` the curvature
term ``3 atf cvx = 3 atf_ref ∂²σ̂/∂k²`` no longer depends on ``atf``; against ``p = 0`` it moves
``ν_SABR`` by +0.4% to +0.5% on the reference SSVI (``atf`` 0.20–0.22, positive ATM curvature) and
by −2.5% to −3.8% on SPX 2022-12-30 (negative ATM vol curvature at ``h = 1e-3``), ``Corr_SABR`` by
−0.004 / −0.02 to −0.03 (measured in ``tests/test_breakeven.py::test_sabr_reduction``).  The slope
is the surface's analytic ``atm_skew`` when it has one, the curvature the central difference of
half-width ``h`` (default ``1e-3``: the ``k → 0`` curvature; ``h = 0.02`` overstates it by 2.5× at
1M on the reference SSVI).  **Radicand guard** (owner, report decision iii): ``6 smi² + 3 atf cvx``
is floored at ``6 smi² (1 − c)``, ``c = radicand_floor`` (config, default 0.5; ``None`` disables),
logged at WARNING and flagged when it fires — a negative ATM curvature large enough to pull the
radicand below the floor.  Note that ``|Corr_SABR| ≤ 1`` needs a radicand of at least ``4 smi²``, so
the guard at ``c > 1/3`` (the default 0.5 gives ``3 smi²``) can still produce ``|Corr| > 1``, which
is then clipped with its own flag; on the 127 SPX 2022 H2 snapshots and the reference SSVI the guard
does not fire at any pillar (measured, ``tests/test_breakeven.py``).  Flags, never silent: the
guard, negative radicand (NaN, only with the guard off), ``|Corr_SABR| > 1`` (clipped), no analytic
``atm_skew`` (central differences), ``ν_SABR² T > 1`` (the ``T → 0`` reduction ignores Hagan's
finite-``T`` correction).  The step-0 readings above were confirmed by the owner (report decision
iii: "step 0 as built").

**Step 1 — break-even targets** (marking mode, the default)::

    VoV_BE(T)  = (atf_3M / atf_T)^q · (ssr_target(T) / 2) · VoV_SABR(T)       q = anchor_power = 1
    Corr_BE(T) = Corr_SABR(T)
    SpotVolCovar_target(T) = Corr_BE(T) · VoV_BE(T)
    VolVar_target(T)       = VoV_BE(T)²

*Rationale:* SABR-implied dynamics are SSR = 2 (``Corr_SABR · VoV_SABR = 2 atf Skew_SABR``: a 1F
Bergomi model with ``k → 0`` matching the pillar meets both targets at ``ssr = 2``, checked to 1e-5
in ``tests/test_breakeven.py``); ``ssr_target = 1`` marks sticky-strike by halving ``VoV_SABR``.
The SSR the targets imply is ``SpotVolCovar_target / (σ_0 Skew_SABR) = ssr_target · atf_3M / σ_0``
before smoothing (``σ_0 = atf_1M``): 0.955 ssr on the reference SSVI, 1.084 ssr on SPX 2022-12-30;
SmoothBreakEven moves it by less than 1% (0.947–0.962 and 1.083–1.085 at ssr 1,
:meth:`TargetSet.ssr_implied`). ``ssr_target`` enters ``VoV_BE`` only; it is not a model-realised
SSR target (the calibrated LSV's numerical SSR is a stage-3 diagnostic of
:mod:`volsto.calibration.fit_2f`).  3M anchoring and **SmoothBreakEven** regularise the VoV term
structure: with ``smooth_breakeven=True`` (default) ``ln VoV_BE`` is replaced by its least-squares
polynomial in ``ln T`` of degree ``min(2, n − 2)`` over the ``n`` pillars (no smoothing below three
pillars, flagged); the raw curve is kept (:attr:`TargetSet.vov_be_raw`) and the largest relative
adjustment is flagged.  The correlation is not smoothed.  *SmoothBreakEven's log-log quadratic is
the library's form*, accepted by the owner (report decision iv).  **MatMin**: pillars below
``mat_min`` (default 3M, the owner's ``removeVolatilityPillarsBelow``) are dropped with a flag, as
are pillars beyond the surface's ``max_maturity``; the default pillars run 3M–10Y.

**σ_0** is the ATMF vol at :data:`SIGMA0_MATURITY` (one month) in both modes (the market proxy
of ``σ_0 = L(0, S_0) sqrt(ξ_0^0)``; 0.2200 against 0.2194 for the cached 2F LSV on the reference
SSVI); a history without a one-month pillar gets the ATM vol interpolated across its pillars with
a flag.

**Historical mode** (non-default alternative, the marked-vs-historical model-reserve study):
``VolVar_target(T) = (atf(T) volvol_hist(T))²``, ``SpotVolCovar_target(T) = SSR_hist(T) σ_0
skew_market(T)``, ``Corr_BE = SpotVolCovar_target / sqrt(VolVar_target)``, ``skew_target =
skew_market`` at the pricing date, with the estimators' standard errors carried; pillars below
``mat_min`` dropped.

**Market term structures.**  The P1 break-even of :mod:`volsto.calibration.fit_2f` needs the
market ATM skew and ATMF vol on ``(0, T]``: marking mode reads them from the surface
(:meth:`TargetSet.market_skew`, :meth:`TargetSet.atmf_curve`); historical mode has the pillars
only (``term_structure_source == "pillars"``; the fitter interpolates the skew residual between
pillars, :func:`pillar_power_law_skew` extends the market skew for reporting).

Checked by ``tests/test_breakeven.py`` (``test_sabr_reduction``,
``test_marking_targets_and_policy_check``, ``test_marking_targets_sabr_consistency_at_ssr_2``,
``test_historical_targets``) and ``tests/test_fit_2f.py::test_breakeven_targets_final``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.market.varswap import ForwardVarianceCurve

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

#: the owner's pillar set (3M–10Y, "break-even used to 10Y")
DEFAULT_TARGET_PILLARS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
ANCHOR_MATURITY = 0.25
#: MatMin / removeVolatilityPillarsBelow: pillars below are dropped (owner default 3M)
DEFAULT_MAT_MIN = 0.25
#: the maturity whose ATMF vol is the market proxy of ``σ_0`` (one month)
SIGMA0_MATURITY = 1.0 / 12.0
#: ``atf_ref`` of the convexity rescaling ``(atf/atf_ref)^SabrW_Power``
DEFAULT_ATF_REF = 0.3
#: ``SabrW_Power`` (library convention, default 1)
DEFAULT_SABRW_POWER = 1.0
#: the radicand guard ``c`` (owner, M7 Part 3 report decision iii): ``6 smi² + 3 atf cvx`` is
#: floored at ``6 smi² (1 − c)``; ``None`` disables the guard (NaN on a negative radicand)
DEFAULT_RADICAND_FLOOR = 0.5
#: pillars with ``ν_SABR² T`` above this are flagged (the ``T → 0`` reduction, module docstring)
SABR_FINITE_T_LIMIT = 1.0
#: default half-width in log-moneyness of the curvature stencil (the ``k → 0`` curvature)
SABR_CURVATURE_H = 1e-3
#: the largest relative SmoothBreakEven adjustment reported without a flag
SMOOTH_FLAG_REL = 0.05
#: relative tolerance of the policy-check readings
_POLICY_RTOL = 1e-9
#: relative tolerance of the ``VoV_SABR / ν_SABR`` classification
_VOV_RTOL = 0.05
MODES = ("marking", "historical")
#: clip of the power-law exponent ``γ`` in ``|S(T)| ∝ T^−γ`` used to extend the pillar skews
#: beyond the first and last pillar in historical mode (``γ < 1`` keeps ``∫₀ᵀ S`` finite)
SKEW_EXPONENT_BOUNDS: tuple[float, float] = (0.0, 0.75)


@dataclass(frozen=True)
class SabrPillar:
    """Step 0 of one pillar (module docstring).  ``skew`` is ``smi_T = ∂σ̂/∂k``, ``curv`` the raw
    ``∂²σ̂/∂k²``, ``cvx`` the rescaled ``curv (atf_ref/atf)^p``, ``nu_sabr`` the lognormal
    ``sqrt(6 smi² + 3 atf cvx)`` and ``rho_sabr`` ``Corr_SABR`` (clipped to [−1, 1])."""

    T: float
    atf: float
    skew: float
    curv: float
    nu_sabr: float
    rho_sabr: float
    flags: tuple[str, ...] = ()
    h: float = float("nan")
    cvx: float = float("nan")
    sabrw_power: float = DEFAULT_SABRW_POWER
    atf_ref: float = DEFAULT_ATF_REF
    radicand_guarded: bool = False

    @property
    def vov_sabr(self) -> float:
        """``VoV_SABR = atf ν_SABR`` (absolute)."""
        return self.atf * self.nu_sabr

    @property
    def corr_sabr(self) -> float:
        return self.rho_sabr

    @property
    def skew_sabr(self) -> float:
        """``Skew_SABR = smi_T`` (the market ATM skew)."""
        return self.skew

    @property
    def atf_365(self) -> float:
        return 100.0 * self.atf

    @property
    def smile_365(self) -> float:
        return float(100.0 * 2.0 * np.sqrt(self.T) * self.skew)

    @property
    def convex_365(self) -> float:
        return 100.0 * self.T * self.curv


def surface_atm_derivatives(
    surface: Any, T: float, h: float = SABR_CURVATURE_H
) -> tuple[float, float, float, bool]:
    """``(atf, ∂σ̂/∂k, ∂²σ̂/∂k², analytic_skew)`` at ``k = 0``: the surface's analytic
    ``atm_skew`` when it has one (``analytic_skew`` True), else the central difference of
    half-width ``h``; the curvature always by the ``h`` stencil (module docstring)."""
    k = np.array([-h, 0.0, h])
    v = np.asarray(surface.implied_vol_k(k, np.full(3, float(T))), dtype=np.float64)
    atf = float(v[1])
    fn = getattr(surface, "atm_skew", None)
    analytic = callable(fn)
    skew = float(np.asarray(fn(float(T)))) if callable(fn) else float((v[2] - v[0]) / (2 * h))
    curv = float((v[2] - 2 * v[1] + v[0]) / (h * h))
    return atf, skew, curv, analytic


def sabr_from_365(
    T: float,
    atf_365: float,
    smile_365: float,
    convex_365: float,
    *,
    sabrw_power: float = DEFAULT_SABRW_POWER,
    atf_ref: float = DEFAULT_ATF_REF,
) -> tuple[float, float, float]:
    """The owner's conversion of the 365-day quotes to physical ``(atf, smi, cvx)``."""
    atf = atf_365 / 100.0
    smi = smile_365 / (100.0 * 2.0 * np.sqrt(T))
    cvx = convex_365 / (100.0 * T * (atf / atf_ref) ** sabrw_power)
    return float(atf), float(smi), float(cvx)


def sabr_reduce(
    surface: Any,
    T: float,
    h: float = SABR_CURVATURE_H,
    *,
    sabrw_power: float = DEFAULT_SABRW_POWER,
    atf_ref: float = DEFAULT_ATF_REF,
    radicand_floor: float | None = DEFAULT_RADICAND_FLOOR,
) -> SabrPillar:
    """Step 0 of one pillar (module docstring) with its flags; ``radicand_floor`` is the guard
    ``c`` (the radicand floored at ``6 smi² (1 − c)``, logged and flagged when it fires; ``None``
    disables it)."""
    if atf_ref <= 0:
        raise ValueError("atf_ref must be positive")
    if radicand_floor is not None and not 0.0 <= radicand_floor < 1.0:
        raise ValueError("radicand_floor must be in [0, 1) or None")
    atf0, skew0, curv0, analytic = surface_atm_derivatives(surface, T, h)
    flags: list[str] = []
    if not analytic:
        flags.append(
            f"skew and curvature by central differences of half-width {h:g} (no analytic "
            "atm_skew): unreliable on an interpolated grid"
        )
    atf, skew, cvx = sabr_from_365(
        float(T),
        100.0 * atf0,
        100.0 * 2.0 * np.sqrt(T) * skew0,
        100.0 * T * curv0,
        sabrw_power=sabrw_power,
        atf_ref=atf_ref,
    )
    rad = 6.0 * skew * skew + 3.0 * atf * cvx
    guarded = False
    if radicand_floor is not None:
        floor = 6.0 * skew * skew * (1.0 - radicand_floor)
        if rad < floor:
            guarded = True
            msg = (
                f"T={T:g}: radicand 6 smi^2 + 3 atf cvx = {rad:.5f} below the guard "
                f"6 smi^2 (1 - {radicand_floor:g}) = {floor:.5f} (ATM curvature "
                f"{curv0:+.3f}): floored"
            )
            log.warning("sabr_reduce: %s", msg)
            flags.append("radicand guard fired: " + msg)
            rad = floor
    if rad <= 0:
        flags.append("negative radicand: smile too flat for its skew (nu_sabr NaN)")
        nu = float("nan")
        rho = float("nan")
    else:
        nu = float(np.sqrt(rad))
        rho = 2.0 * skew / nu
        if abs(rho) > 1.0:
            flags.append(f"rho_sabr {rho:+.3f} outside [-1, 1]: clipped")
            rho = float(np.clip(rho, -1.0, 1.0))
        if nu * nu * T > SABR_FINITE_T_LIMIT:
            flags.append(
                f"nu_sabr^2 T = {nu * nu * T:.2f} > {SABR_FINITE_T_LIMIT:g}: the T -> 0 SABR "
                "reduction ignores Hagan's finite-T correction (nu_sabr biased)"
            )
    return SabrPillar(
        float(T),
        atf,
        skew,
        float(curv0),
        nu,
        rho,
        tuple(flags),
        float(h),
        cvx,
        float(sabrw_power),
        float(atf_ref),
        guarded,
    )


def smooth_breakeven_curve(pillars: FloatArray, vov: FloatArray) -> tuple[FloatArray, int]:
    """SmoothBreakEven (module docstring): ``ln VoV`` fitted by a polynomial in ``ln T`` of
    degree ``min(2, n − 2)``; returns ``(smoothed, degree)``, the input and degree ``−1`` below
    three pillars."""
    p = np.asarray(pillars, dtype=np.float64)
    v = np.asarray(vov, dtype=np.float64)
    if p.size < 3:
        return v.copy(), -1
    if np.any(~np.isfinite(v)) or np.any(v <= 0):
        raise ValueError("SmoothBreakEven needs finite positive VoV values")
    deg = min(2, p.size - 2)
    c = np.polyfit(np.log(p), np.log(v), deg)
    return np.asarray(np.exp(np.polyval(c, np.log(p))), dtype=np.float64), deg


def _curve(x: float | Mapping[float, float] | Callable[[float], float], T: float) -> float:
    if callable(x):
        return float(x(T))
    if isinstance(x, Mapping):
        keys = np.array(sorted(x))
        vals = np.array([x[k] for k in keys])
        return float(np.interp(T, keys, vals))
    return float(x)


def _close(a: float, b: float, rtol: float) -> bool:
    return bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) <= rtol * max(abs(b), 1e-300))


SsrInput = float | Sequence[float] | FloatArray | Mapping[float, float] | Callable[[float], float]


@dataclass(frozen=True)
class TargetSet:
    """Break-even targets per pillar (absolute vol units, module docstring).  ``vovol`` is
    ``VoV_BE`` (smoothed when ``smooth_breakeven``), ``vov_be_raw`` before smoothing,
    ``correl_target`` ``Corr_BE``; ``atf_anchor`` the 3M ATMF vol of the anchoring (NaN in
    historical mode)."""

    mode: str
    pillars: FloatArray
    sigma_0: float
    atf: FloatArray
    skew_target: FloatArray
    spot_vol_covar: FloatArray
    vol_var: FloatArray
    vovol: FloatArray
    ssr_target: FloatArray
    spot_vol_covar_se: FloatArray
    vol_var_se: FloatArray
    sabr: tuple[SabrPillar, ...] = ()
    anchor_power: float = 1.0
    anchor: FloatArray = field(default_factory=lambda: np.ones(0))
    flags: tuple[str, ...] = ()
    atf_anchor: float = float("nan")
    correl_target: FloatArray = field(default_factory=lambda: np.zeros(0))
    skew_fn: Callable[[FloatArray], FloatArray] | None = field(
        default=None, repr=False, compare=False
    )
    atm_vol_fn: Callable[[FloatArray], FloatArray] | None = field(
        default=None, repr=False, compare=False
    )
    vov_be_raw: FloatArray = field(default_factory=lambda: np.zeros(0))
    smooth_breakeven: bool = False
    smooth_degree: int = -1
    mat_min: float = 0.0
    sabrw_power: float = float("nan")
    atf_ref: float = float("nan")

    @property
    def term_structure_source(self) -> str:
        """``"surface"`` when the market skew / ATMF vol on ``(0, T]`` come from the surface,
        ``"pillars"`` otherwise (historical mode)."""
        return "surface" if self.skew_fn is not None else "pillars"

    @property
    def vov_sabr(self) -> FloatArray:
        return (
            np.array([s.vov_sabr for s in self.sabr])
            if self.sabr
            else np.full_like(self.pillars, np.nan)
        )

    @property
    def ssr_implied(self) -> FloatArray:
        """``SpotVolCovar_target / (σ_0 skew_target)``: ``ssr_target · atf_3M / σ_0`` in marking
        mode (module docstring), ``SSR_hist`` in historical mode."""
        return np.asarray(self.spot_vol_covar / (self.sigma_0 * self.skew_target))

    def market_skew(self, t: FloatArray | float) -> FloatArray:
        """Market ATM skew ``∂σ̂/∂k`` at maturities ``t > 0`` (surface, else the pillar power
        law of :func:`pillar_power_law_skew`)."""
        t_ = np.atleast_1d(np.asarray(t, dtype=np.float64))
        if np.any(t_ <= 0):
            raise ValueError("maturities must be positive")
        if self.skew_fn is not None:
            return np.asarray(self.skew_fn(t_), dtype=np.float64)
        return pillar_power_law_skew(self.pillars, self.skew_target, t_)

    def atmf_curve(self, t_max: float) -> ForwardVarianceCurve:
        """The ATMF total-variance curve ``σ̂_t² t`` as a :class:`ForwardVarianceCurve` to
        ``t_max`` (so ``xi0(t) = d(σ̂_t² t)/dt``, book p. 475): the surface's ATMF vols on a weekly
        grid (denser in the first month) in marking mode, the pillar ATMF vols with a flat vol
        before the first pillar and beyond the last in historical mode."""
        if self.atm_vol_fn is not None:
            short = np.array([1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 22.0]) / 365.0
            weekly = np.arange(1, int(np.ceil(52 * t_max)) + 2) / 52.0
            mats = np.unique(
                np.concatenate((short[short < t_max], weekly[weekly < t_max], [t_max]))
            )
            W = np.asarray(self.atm_vol_fn(mats), dtype=np.float64) ** 2 * mats
            if np.any(np.diff(W) <= 0):
                raise ValueError("the ATMF total variance is not increasing in T")
            return ForwardVarianceCurve(mats, W)
        p = np.asarray(self.pillars, dtype=np.float64)
        a = np.asarray(self.atf, dtype=np.float64)
        end = max(float(t_max), float(p[-1])) + 1.0
        mats = np.concatenate(([0.5 * p[0]], p, [end]))
        vols = np.concatenate(([a[0]], a, [a[-1]]))
        W = vols * vols * mats
        W = np.maximum.accumulate(W + 1e-12 * np.arange(W.size))
        return ForwardVarianceCurve(mats, W)

    def with_ssr_target(self, ssr_target: SsrInput) -> TargetSet:
        """The same targets at another ``ssr_target`` (scalar, one value per pillar, mapping
        ``T → value`` interpolated, or callable): marking ``VoV_BE`` rebuilt (and re-smoothed),
        ``SpotVolCovar = Corr_BE VoV_BE``; historical ``SpotVolCovar = ssr σ_0 skew``, ``VolVar``
        unchanged."""
        ssr = _ssr_array(ssr_target, self.pillars)
        if self.mode == "marking" and self.sabr:
            raw = self.anchor * 0.5 * ssr * self.vov_sabr
            vov = smooth_breakeven_curve(self.pillars, raw)[0] if self.smooth_breakeven else raw
            return replace(
                self,
                ssr_target=ssr,
                vov_be_raw=raw,
                vovol=vov,
                vol_var=vov * vov,
                spot_vol_covar=self.correl_target * vov,
            )
        svc = ssr * self.sigma_0 * self.skew_target
        with np.errstate(divide="ignore", invalid="ignore"):
            correl = svc / self.vovol
        return replace(self, ssr_target=ssr, spot_vol_covar=svc, correl_target=correl)

    def frame(self) -> pd.DataFrame:
        d: dict[str, Any] = {
            "T": self.pillars,
            "atf": self.atf,
            "skew_target": self.skew_target,
            "ssr_target": self.ssr_target,
            "ssr_implied": self.ssr_implied,
            "vov_be": self.vovol,
            "spot_vol_covar_target": self.spot_vol_covar,
            "spot_vol_covar_se": self.spot_vol_covar_se,
            "vol_var_target": self.vol_var,
            "vol_var_se": self.vol_var_se,
        }
        if self.correl_target.size == self.pillars.size:
            d["corr_be"] = self.correl_target
        if self.sabr:
            d["vov_be_raw"] = self.vov_be_raw
            d["nu_sabr"] = np.array([s.nu_sabr for s in self.sabr])
            d["vov_sabr"] = self.vov_sabr
            d["rho_sabr"] = np.array([s.rho_sabr for s in self.sabr])
            d["smile_365"] = np.array([s.smile_365 for s in self.sabr])
            d["convex_365"] = np.array([s.convex_365 for s in self.sabr])
            d["anchor"] = self.anchor
        return pd.DataFrame(d)

    def policy_check(
        self,
        vov_sabr: Mapping[float, float] | Callable[[float], float] | None = None,
    ) -> pd.DataFrame:
        """Marking mode: the raw (unsmoothed) ``VoV_BE / ssr`` against the policy ``½ VoV_SABR
        atf(3M)/atf(T)`` under the two readings of ``VoV_SABR`` — absolute (``ν_SABR atf``,
        reproduced exactly at ``anchor_power = 1``) and lognormal (``ν_SABR``, off by
        ``1/atf(T)``); ``reading`` names the one reproduced to :data:`_POLICY_RTOL`.  With the
        desk's ``VoV_SABR`` per pillar the columns ``vov_sabr``, ``vov_over_nu`` and
        ``vov_reading`` classify it (``"lognormal"`` ≈ 1, ``"absolute"`` ≈ atf, ``"omega"`` ≈ 2,
        else ``"unknown"``).  Raises in historical mode."""
        if self.mode != "marking" or not self.sabr:
            raise ValueError("policy_check applies to marking-mode targets")
        atf3 = self.atf_anchor
        rows = []
        for s, vv, ssr in zip(self.sabr, self.vov_be_raw, self.ssr_target):
            at_unit = vv / ssr if ssr != 0 else np.nan
            policy_abs = 0.5 * s.nu_sabr * atf3
            policy_ln = 0.5 * s.nu_sabr * atf3 / s.atf
            if _close(at_unit, policy_abs, _POLICY_RTOL):
                reading = "absolute"
            elif _close(at_unit, policy_ln, _POLICY_RTOL):
                reading = "lognormal"
            else:
                reading = "neither"
            row: dict[str, object] = {
                "T": s.T,
                "nu_sabr": s.nu_sabr,
                "atf": s.atf,
                "vovol_target_at_ssr_1": at_unit,
                "policy_absolute": policy_abs,
                "policy_lognormal": policy_ln,
                "reading": reading,
                "mismatch_lognormal": policy_ln / policy_abs if policy_abs else np.nan,
            }
            if vov_sabr is not None:
                vov = _curve(vov_sabr, float(s.T))
                ratio = vov / s.nu_sabr if s.nu_sabr else np.nan
                if _close(ratio, 1.0, _VOV_RTOL):
                    kind = "lognormal"
                elif _close(ratio, s.atf, _VOV_RTOL):
                    kind = "absolute"
                elif _close(ratio, 2.0, _VOV_RTOL):
                    kind = "omega"
                else:
                    kind = "unknown"
                row.update({"vov_sabr": vov, "vov_over_nu": ratio, "vov_reading": kind})
            rows.append(row)
        return pd.DataFrame(rows)


def _ssr_array(ssr_target: SsrInput, pillars: FloatArray) -> FloatArray:
    if isinstance(ssr_target, Mapping) or callable(ssr_target):
        ssr = np.array([_curve(ssr_target, float(T)) for T in pillars])
    else:
        arr = np.asarray(ssr_target, dtype=np.float64)
        ssr = np.full(pillars.size, float(arr)) if arr.ndim == 0 else arr.copy()
    if ssr.shape != pillars.shape:
        raise ValueError("ssr_target must be a scalar, one value per pillar, a mapping or callable")
    if not np.all(np.isfinite(ssr)) or np.any(ssr <= 0):
        raise ValueError("ssr_target must be finite and positive")
    return np.asarray(ssr, dtype=np.float64)


def _filter_pillars(
    pillars: Sequence[float], mat_min: float, tmax: float | None
) -> tuple[FloatArray, list[str]]:
    ps = np.asarray(sorted(float(t) for t in pillars), dtype=np.float64)
    flags: list[str] = []
    below = ps < float(mat_min) - 1e-12
    if below.any():
        flags.append(
            f"pillars below mat_min {mat_min:g} dropped (removeVolatilityPillarsBelow): "
            f"{ps[below].tolist()}"
        )
        ps = ps[~below]
    if tmax is not None:
        keep = ps <= float(tmax) + 1e-9
        if not keep.all():
            flags.append(f"pillars beyond max_maturity {tmax:g} dropped: {ps[~keep].tolist()}")
            ps = ps[keep]
    if ps.size == 0:
        raise ValueError("no pillar inside the surface's maturity range and above mat_min")
    return ps, flags


def marking_targets(
    surface: Any,
    pillars: Sequence[float] = DEFAULT_TARGET_PILLARS,
    *,
    ssr_target: SsrInput = 1.0,
    anchor_power: float = 1.0,
    h: float = SABR_CURVATURE_H,
    sigma_0: float | None = None,
    mat_min: float = DEFAULT_MAT_MIN,
    smooth_breakeven: bool = True,
    sabrw_power: float = DEFAULT_SABRW_POWER,
    atf_ref: float = DEFAULT_ATF_REF,
    radicand_floor: float | None = DEFAULT_RADICAND_FLOOR,
) -> TargetSet:
    """Marking-mode targets from a surface (steps 0 and 1 of the module docstring).
    ``ssr_target`` is a scalar, one value per retained pillar, a mapping ``T → value``
    (interpolated) or a callable; ``sigma_0`` defaults to the surface's ATMF vol at
    :data:`SIGMA0_MATURITY`."""
    ps, flags = _filter_pillars(pillars, mat_min, getattr(surface, "max_maturity", None))
    sabr = tuple(
        sabr_reduce(
            surface,
            float(T),
            h,
            sabrw_power=sabrw_power,
            atf_ref=atf_ref,
            radicand_floor=radicand_floor,
        )
        for T in ps
    )
    for s in sabr:
        flags += [f"T={s.T:g}: {f}" for f in s.flags]
    atf = np.array([s.atf for s in sabr])
    atf3 = float(surface.atm_vol(ANCHOR_MATURITY))
    s0 = float(surface.atm_vol(SIGMA0_MATURITY)) if sigma_0 is None else float(sigma_0)
    ssr = _ssr_array(ssr_target, ps)
    anchor = (atf3 / atf) ** float(anchor_power)
    vov_sabr = np.array([s.vov_sabr for s in sabr])
    raw = anchor * 0.5 * ssr * vov_sabr
    degree = -1
    vov = raw.copy()
    if smooth_breakeven:
        vov, degree = smooth_breakeven_curve(ps, raw)
        if degree < 0:
            flags.append("SmoothBreakEven requested but fewer than three pillars: not smoothed")
        else:
            adj = float(np.max(np.abs(vov / raw - 1.0)))
            if adj > SMOOTH_FLAG_REL:
                flags.append(
                    f"SmoothBreakEven moved VoV_BE by up to {adj:.1%} (degree {degree} in ln T)"
                )
    corr = np.array([s.rho_sabr for s in sabr])
    skew = np.array([s.skew_sabr for s in sabr])
    fn = getattr(surface, "atm_skew", None)
    if callable(fn):

        def skew_fn(t: FloatArray) -> FloatArray:
            return np.asarray(surface.atm_skew(t), dtype=np.float64)

    else:

        def skew_fn(t: FloatArray) -> FloatArray:
            return np.array([surface_atm_derivatives(surface, float(x), h)[1] for x in t])

    def atm_vol_fn(t: FloatArray) -> FloatArray:
        return np.asarray(surface.atm_vol(t), dtype=np.float64)

    return TargetSet(
        mode="marking",
        pillars=ps,
        sigma_0=s0,
        atf=atf,
        skew_target=skew,
        spot_vol_covar=corr * vov,
        vol_var=vov * vov,
        vovol=vov,
        ssr_target=ssr,
        spot_vol_covar_se=np.zeros(ps.size),
        vol_var_se=np.zeros(ps.size),
        sabr=sabr,
        anchor_power=float(anchor_power),
        anchor=anchor,
        flags=tuple(flags),
        atf_anchor=atf3,
        correl_target=corr,
        skew_fn=skew_fn,
        atm_vol_fn=atm_vol_fn,
        vov_be_raw=raw,
        smooth_breakeven=bool(smooth_breakeven),
        smooth_degree=int(degree),
        mat_min=float(mat_min),
        sabrw_power=float(sabrw_power),
        atf_ref=float(atf_ref),
    )


def historical_targets(
    history: Any,
    pillars: Sequence[float] | None = None,
    *,
    end: Any = None,
    window_vol: int = 250,
    window_ssr: int = 60,
    sigma_0: float | None = None,
    mat_min: float = 0.0,
) -> TargetSet:
    """Historical-mode targets from a :class:`~volsto.calibration.history.SurfaceHistory` at
    ``end`` (default: its last date): ``volvol_hist`` and ``SSR_hist`` of the Part 2 estimators,
    the market skew and ATMF vol of the pricing date (module docstring); ``pillars`` default to
    the history's pillars at or above ``mat_min``.  ``sigma_0`` defaults to the ATM vol at
    :data:`SIGMA0_MATURITY` — interpolated across the history's pillars with a flag when one
    month is not a pillar."""
    e = history.date_index(end)
    end_ts = history.dates[e]
    hp = np.asarray(history.pillars, dtype=np.float64)
    ps, pflags = _filter_pillars(hp.tolist() if pillars is None else pillars, mat_min, None)
    atm = history.atm_vol.to_numpy()[e]
    skw = history.skew.to_numpy()[e]
    atf = np.array([atm[history.pillar_index(float(T))] for T in ps])
    skew = np.array([skw[history.pillar_index(float(T))] for T in ps])
    flags = [f"windows vol {window_vol} / ssr {window_ssr}, end {end_ts.date()}", *pflags]
    if sigma_0 is None:
        try:
            s0 = float(atm[history.pillar_index(SIGMA0_MATURITY)])
        except KeyError:
            order = np.argsort(hp)
            s0 = float(np.interp(SIGMA0_MATURITY, hp[order], np.asarray(atm)[order]))
            flags.append(
                f"sigma_0 = {s0:.4f}: ATM vol interpolated at T = {SIGMA0_MATURITY:.4f} across "
                f"the history pillars {hp.tolist()} (no one-month pillar; flat outside)"
            )
    else:
        s0 = float(sigma_0)
    vv = [history.volvol_hist(float(T), window_vol, end=end_ts) for T in ps]
    sr = [history.ssr_hist(float(T), window_ssr, end=end_ts) for T in ps]
    vovol = np.array([atf_i * v.volvol for atf_i, v in zip(atf, vv)])
    vovol_se = np.array([atf_i * v.se for atf_i, v in zip(atf, vv)])
    ssr = np.array([r.ssr for r in sr])
    ssr_se = np.array([r.se for r in sr])
    svc = ssr * s0 * skew
    with np.errstate(divide="ignore", invalid="ignore"):
        correl = svc / vovol
    return TargetSet(
        mode="historical",
        pillars=ps,
        sigma_0=s0,
        atf=atf,
        skew_target=skew,
        spot_vol_covar=svc,
        vol_var=vovol * vovol,
        vovol=vovol,
        ssr_target=ssr,
        spot_vol_covar_se=np.abs(ssr_se * s0 * skew),
        vol_var_se=2.0 * vovol * vovol_se,
        sabr=(),
        anchor_power=float("nan"),
        anchor=np.ones(ps.size),
        flags=tuple(flags),
        atf_anchor=float("nan"),
        correl_target=np.asarray(correl, dtype=np.float64),
        vov_be_raw=vovol.copy(),
        mat_min=float(mat_min),
    )


def pillar_power_law_skew(
    pillars: FloatArray,
    skews: FloatArray,
    t: FloatArray,
    exponent_bounds: tuple[float, float] = SKEW_EXPONENT_BOUNDS,
) -> FloatArray:
    """The pillar skews extended to maturities ``t`` as a power law in ``T``: ``ln|S|`` linear in
    ``ln T`` between pillars, ``|S(T)| = |S(T_end)| (T/T_end)^−γ`` beyond the first / last pillar
    with ``γ`` the end segment's exponent clipped to ``exponent_bounds``; the sign is the
    pillars' common sign (a sign change raises).  One pillar gives ``γ`` at the lower bound."""
    p = np.asarray(pillars, dtype=np.float64)
    s = np.asarray(skews, dtype=np.float64)
    t_ = np.atleast_1d(np.asarray(t, dtype=np.float64))
    if p.size == 0 or np.any(s == 0) or not (np.all(s < 0) or np.all(s > 0)):
        raise ValueError("pillar skews must be non-zero and of one sign")
    order = np.argsort(p)
    p, s = p[order], s[order]
    sign = float(np.sign(s[0]))
    lp, ls = np.log(p), np.log(np.abs(s))
    lo_b, hi_b = exponent_bounds
    if p.size == 1:
        g_lo = g_hi = lo_b
    else:
        g_lo = float(np.clip(-(ls[1] - ls[0]) / (lp[1] - lp[0]), lo_b, hi_b))
        g_hi = float(np.clip(-(ls[-1] - ls[-2]) / (lp[-1] - lp[-2]), lo_b, hi_b))
    lt = np.log(t_)
    inner = np.interp(lt, lp, ls)
    out = np.where(lt < lp[0], ls[0] - g_lo * (lt - lp[0]), inner)
    out = np.where(lt > lp[-1], ls[-1] - g_hi * (lt - lp[-1]), out)
    return np.asarray(sign * np.exp(out), dtype=np.float64)


__all__ = [
    "ANCHOR_MATURITY",
    "DEFAULT_ATF_REF",
    "DEFAULT_MAT_MIN",
    "DEFAULT_RADICAND_FLOOR",
    "DEFAULT_SABRW_POWER",
    "DEFAULT_TARGET_PILLARS",
    "MODES",
    "SABR_CURVATURE_H",
    "SABR_FINITE_T_LIMIT",
    "SIGMA0_MATURITY",
    "SKEW_EXPONENT_BOUNDS",
    "SMOOTH_FLAG_REL",
    "SabrPillar",
    "TargetSet",
    "historical_targets",
    "marking_targets",
    "pillar_power_law_skew",
    "sabr_from_365",
    "sabr_reduce",
    "smooth_breakeven_curve",
]
