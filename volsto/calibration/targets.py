"""Break-even targets for the two-factor fit (SPEC §15 Part 3, M7 addendum): the SABR
reduction of a surface pillar and the two target modes.

**SABR reduction** (per pillar ``T``, lognormal SABR at order one in ``ν``, Hagan): from the
surface's ATM level ``atf = σ̂(k = 0)``, ATM slope ``skew = ∂σ̂/∂k`` and curvature ``curv =
∂²σ̂/∂k²`` in log-moneyness ``k = ln(K/F)``::

    ν_SABR = sqrt(6 skew² + 3 atf curv),   ρ_SABR = 2 skew / ν_SABR,   Skew_SABR = ½ ρ_SABR ν_SABR

(``σ̂(k) ≈ atf + ½ ρ ν k + (2 − 3ρ²) ν² / (12 atf) k²`` inverts to exactly these; ``ν_SABR`` is
the lognormal vol of vol of the SABR ATM vol — the ``ν`` of ``dα = ν α dZ``, the same ``ν`` as
Bergomi's for a 1F model with ``k → 0`` — and ``Skew_SABR`` reproduces the surface skew by
construction, in vol per unit log-strike, the units of book eq. 8.54).  The slope is the
surface's analytic ``atm_skew`` when it has one, the curvature the central difference of
half-width ``h`` in ``k`` (default ``1e-3``: the ``k → 0`` curvature, consistent with the
analytic slope; on the reference SSVI ``h = 0.02`` overstates it by 2.5x at 1M and 1.6x at 3M,
muted to 1.5% / 0.6% on ``ν_SABR`` because the radicand is skew-dominated at ``ρ ≈ −0.8``; the
value used is recorded in :attr:`SabrPillar.h`).  Flags, never silent: a negative radicand (a
smile too flat for its skew) gives NaN; ``|ρ_SABR| > 1`` is clipped; a surface without an
analytic ``atm_skew`` gets both derivatives from the stencil, which is meaningless on a
piecewise-linear grid (a :class:`~volsto.market.surface.GridSurface` with an ATM knot measures
the kink: curvature growing like ``1/h``); pillars with ``ν_SABR² T > SABR_FINITE_T_LIMIT`` are
flagged because the ``T → 0`` reduction ignores Hagan's finite-``T`` correction (measured bias
on ``ν_SABR``: −0.7% at 3M and −2.7% at 1y for ``(ν, ρ) = (0.5, −0.9)``, +12% at 3M and +50% at
1y for ``(2.5, +0.3)``; ``ρ_SABR`` is unaffected).  *Convention to confirm (owner):* accept the
order-one reduction on its domain ``ν_SABR² T ≲ 1`` (0.6–0.8 on the reference surface at every
pillar), or invert the full Hagan ATM formula per pillar.

**Marking mode** (default): the correlation comes from SABR, the SSR is a dial, the vol of
vol is the *output* of the SABR level, the anchoring and the dial::

    A(T)                   = (atf(3M) / atf(T))^p               anchor_power p (1 = 3M anchoring)
    vovol_target(T)        = ½ ssr(T) atf(T) ν_SABR(T) A(T)     absolute vol of the ATMF vol
    SpotVolCovar_target(T) = ssr(T) σ_0 Skew_SABR(T)     (correl cancels: it enters vovol only)
    VolVar_target(T)       = vovol_target(T)²
    skew_target(T)         = Skew_SABR(T)

so that a model matching these has ``SSR_T = SpotVolCovar/(σ_0 Skew) = ssr(T)`` and the
spot/vol correlation ``ρ_SABR σ_0/atf`` at every dial value.  **The ½ is not a ν-versus-ω
conversion** (``ν_SABR`` already is a vol of vol): it is the value at which ``ssr = 2`` — the
``T → 0`` SSR of *every* diffusive model (book ``R_0 = 2``; SPEC Part 1 ``R_1w = 1.999 ± 0.016``)
— reproduces the smile's own dynamics exactly: a 1F Bergomi model with ``k → 0`` that reproduces
a SABR pillar satisfies ``VolVar_target`` and ``SpotVolCovar_target`` at ``ssr = 2`` to 1e-12
(``tests/test_breakeven.py::test_marking_targets_sabr_consistency_at_ssr_2``).  At ``ssr = 1``
(the sticky-strike marking policy) both targets are therefore **half** the diffusive
short-maturity dynamics of the smile: the vol of vol demanded is ``½ atf ν_SABR`` and the
spot/vol covariance ``σ_0 Skew``, whereas the smallest value any diffusive model with the
market skew can show at 1M is about ``2 σ_0 Skew`` (engine SSR of the pure-SV kernels at 1M:
1.96 / 1.89 for 1F / 2F, 2.5–2.6 for the cached LSVs at 3M).  *Convention to confirm
(owner):* whether the dial is meant relative to the diffusive short-``T`` value (then write it
``ssr/2`` with ``ssr = 2`` at ``T → 0``) or as an absolute SSR (then the short pillars are
infeasible for a diffusive model and should be weighted down or excluded); both halves come
from the same dial.  :meth:`TargetSet.policy_check` compares ``vovol_target/ssr`` with the desk
policy ``½ VoV_SABR atf(3M)/atf(T)`` under the two readings of ``VoV_SABR`` — absolute
(``ν_SABR atf``, which the formula above reproduces exactly at ``p = 1``) and lognormal
(``ν_SABR``, off by the factor ``1/atf(T)``) — from the stored 3M anchor, and, given the desk's
own ``VoV_SABR`` per pillar, reports ``VoV_SABR/ν_SABR`` (1 → lognormal ν-like, ``atf`` →
absolute, 2 → ω-like); without the desk's number the ``reading`` column is an identity check of
the formula only (``"absolute"`` at ``p = 1``), not a discrimination between conventions.

**Historical mode**: ``VolVar_target(T) = (atf(T) volvol_hist(T))²`` (the lognormal vol of the
VS vol of the M7 Part 2 estimator as the ATMF vol's, order one), ``SpotVolCovar_target(T) =
SSR_hist(T) σ_0 skew_market(T)`` and ``skew_target = skew_market`` at the pricing date, with the
estimators' standard errors carried.

**σ_0** is the ATMF vol at :data:`SIGMA0_MATURITY` (one month) in both modes — the market
proxy of SPEC §15's ``σ_0 = L(0, S_0) sqrt(ξ_0^0)``, the instantaneous spot vol (0.2200 against
0.2194 for the cached 2F LSV on the reference SSVI), the same ``σ_0`` the engine
:mod:`volsto.analytics.breakeven` defaults to; a history without a one-month pillar gets the
ATM vol interpolated across its pillars (flat outside) with a flag naming it, never a silent
nearest pillar.  Checked by ``tests/test_breakeven.py`` (``test_sabr_reduction``,
``test_marking_targets_and_policy_check``, ``test_marking_targets_sabr_consistency_at_ssr_2``,
``test_historical_targets``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

DEFAULT_TARGET_PILLARS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0)
ANCHOR_MATURITY = 0.25
#: the maturity whose ATMF vol is the market proxy of ``σ_0`` (one month)
SIGMA0_MATURITY = 1.0 / 12.0
#: pillars with ``ν_SABR² T`` above this are flagged (the ``T → 0`` reduction, module docstring)
SABR_FINITE_T_LIMIT = 1.0
#: default half-width in log-moneyness of the curvature stencil (the ``k → 0`` curvature)
SABR_CURVATURE_H = 1e-3
#: relative tolerance of the policy-check readings
_POLICY_RTOL = 1e-9
#: relative tolerance of the ``VoV_SABR / ν_SABR`` classification
_VOV_RTOL = 0.05
MODES = ("marking", "historical")


@dataclass(frozen=True)
class SabrPillar:
    T: float
    atf: float
    skew: float
    curv: float
    nu_sabr: float
    rho_sabr: float
    flags: tuple[str, ...] = ()
    h: float = float("nan")

    @property
    def skew_sabr(self) -> float:
        """``½ ρ_SABR ν_SABR`` (= ``skew`` unless ``ρ`` was clipped)."""
        return 0.5 * self.rho_sabr * self.nu_sabr


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


def sabr_reduce(surface: Any, T: float, h: float = SABR_CURVATURE_H) -> SabrPillar:
    """The SABR reduction of one pillar (module docstring) with its flags."""
    atf, skew, curv, analytic = surface_atm_derivatives(surface, T, h)
    flags: list[str] = []
    if not analytic:
        flags.append(
            f"skew and curvature by central differences of half-width {h:g} (no analytic "
            "atm_skew): unreliable on an interpolated grid"
        )
    rad = 6.0 * skew * skew + 3.0 * atf * curv
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
    return SabrPillar(float(T), atf, skew, curv, nu, rho, tuple(flags), float(h))


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


@dataclass(frozen=True)
class TargetSet:
    """Break-even targets per pillar (absolute vol units, see the module docstring).
    ``atf_anchor`` is the ATMF vol at :data:`ANCHOR_MATURITY` used by ``anchor`` (marking mode;
    NaN in historical mode)."""

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

    def frame(self) -> pd.DataFrame:
        d = {
            "T": self.pillars,
            "atf": self.atf,
            "skew_target": self.skew_target,
            "ssr_target": self.ssr_target,
            "vovol_target": self.vovol,
            "spot_vol_covar_target": self.spot_vol_covar,
            "spot_vol_covar_se": self.spot_vol_covar_se,
            "vol_var_target": self.vol_var,
            "vol_var_se": self.vol_var_se,
        }
        if self.sabr:
            d["nu_sabr"] = np.array([s.nu_sabr for s in self.sabr])
            d["rho_sabr"] = np.array([s.rho_sabr for s in self.sabr])
            d["anchor"] = self.anchor
        return pd.DataFrame(d)

    def policy_check(
        self,
        vov_sabr: Mapping[float, float] | Callable[[float], float] | None = None,
    ) -> pd.DataFrame:
        """Marking mode: ``vovol_target / ssr`` against the marking policy ``½ VoV_SABR
        atf(3M)/atf(T)`` (the stored 3M anchor) under the two readings of ``VoV_SABR``
        (absolute ``ν_SABR atf``; lognormal ``ν_SABR``); ``reading`` names the one the target
        formula reproduces to :data:`_POLICY_RTOL` relative (``"absolute"`` at ``p = 1``,
        ``"neither"`` at other anchoring powers — an identity check of the formula, not a
        discrimination between conventions), ``mismatch_lognormal`` the factor ``1/atf(T)``
        by which the other differs.  With the desk's ``VoV_SABR`` per pillar (a mapping ``T →
        value``, interpolated, or a callable) the columns ``vov_sabr``, ``vov_over_nu`` and
        ``vov_reading`` classify it: ``"lognormal"`` (``VoV/ν_SABR ≈ 1``), ``"absolute"``
        (``≈ atf(T)``), ``"omega"`` (``≈ 2``) within :data:`_VOV_RTOL`, else ``"unknown"``.
        Raises in historical mode."""
        if self.mode != "marking" or not self.sabr:
            raise ValueError("policy_check applies to marking-mode targets")
        atf3 = self.atf_anchor
        rows = []
        for s, vv, ssr in zip(self.sabr, self.vovol, self.ssr_target):
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


def marking_targets(
    surface: Any,
    pillars: Sequence[float] = DEFAULT_TARGET_PILLARS,
    *,
    ssr_target: float | Mapping[float, float] | Callable[[float], float] = 1.0,
    anchor_power: float = 1.0,
    h: float = SABR_CURVATURE_H,
    sigma_0: float | None = None,
) -> TargetSet:
    """Marking-mode targets from a surface (module docstring).  ``pillars`` beyond the
    surface's ``max_maturity`` are dropped with a flag; ``sigma_0`` defaults to the surface's
    ATMF vol at :data:`SIGMA0_MATURITY`."""
    ps = np.asarray(sorted(float(t) for t in pillars), dtype=np.float64)
    flags: list[str] = []
    tmax = getattr(surface, "max_maturity", None)
    if tmax is not None:
        keep = ps <= float(tmax) + 1e-9
        if not keep.all():
            flags.append(f"pillars beyond max_maturity {tmax:g} dropped: {ps[~keep].tolist()}")
            ps = ps[keep]
    if ps.size == 0:
        raise ValueError("no pillar inside the surface's maturity range")
    sabr = tuple(sabr_reduce(surface, float(T), h) for T in ps)
    for s in sabr:
        flags += [f"T={s.T:g}: {f}" for f in s.flags]
    atf = np.array([s.atf for s in sabr])
    atf3 = float(surface.atm_vol(ANCHOR_MATURITY))
    s0 = float(surface.atm_vol(SIGMA0_MATURITY)) if sigma_0 is None else float(sigma_0)
    ssr = np.array([_curve(ssr_target, float(T)) for T in ps])
    anchor = (atf3 / atf) ** float(anchor_power)
    nu = np.array([s.nu_sabr for s in sabr])
    skew = np.array([s.skew_sabr for s in sabr])
    vovol = 0.5 * ssr * atf * nu * anchor
    svc = ssr * s0 * skew
    return TargetSet(
        "marking",
        ps,
        s0,
        atf,
        skew,
        svc,
        vovol * vovol,
        vovol,
        ssr,
        np.zeros(ps.size),
        np.zeros(ps.size),
        sabr,
        float(anchor_power),
        anchor,
        tuple(flags),
        atf3,
    )


def historical_targets(
    history: Any,
    pillars: Sequence[float] | None = None,
    *,
    end: Any = None,
    window_vol: int = 250,
    window_ssr: int = 60,
    sigma_0: float | None = None,
) -> TargetSet:
    """Historical-mode targets from a :class:`~volsto.calibration.history.SurfaceHistory` at
    ``end`` (default: its last date): ``volvol_hist`` and ``SSR_hist`` of the Part 2 estimators,
    the market skew and ATMF vol of the pricing date (module docstring).  ``sigma_0`` defaults
    to the ATM vol at :data:`SIGMA0_MATURITY` — interpolated across the history's pillars with
    a flag when one month is not a pillar."""
    e = history.date_index(end)
    end_ts = history.dates[e]
    hp = np.asarray(history.pillars, dtype=np.float64)
    ps = hp if pillars is None else np.asarray(sorted(float(t) for t in pillars), dtype=np.float64)
    atm = history.atm_vol.to_numpy()[e]
    skw = history.skew.to_numpy()[e]
    atf = np.array([atm[history.pillar_index(float(T))] for T in ps])
    skew = np.array([skw[history.pillar_index(float(T))] for T in ps])
    flags = [f"windows vol {window_vol} / ssr {window_ssr}, end {end_ts.date()}"]
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
    return TargetSet(
        "historical",
        ps,
        s0,
        atf,
        skew,
        svc,
        vovol * vovol,
        vovol,
        ssr,
        np.abs(ssr_se * s0 * skew),
        2.0 * vovol * vovol_se,
        (),
        float("nan"),
        np.ones(ps.size),
        tuple(flags),
    )


__all__ = [
    "ANCHOR_MATURITY",
    "DEFAULT_TARGET_PILLARS",
    "MODES",
    "SABR_CURVATURE_H",
    "SABR_FINITE_T_LIMIT",
    "SIGMA0_MATURITY",
    "SabrPillar",
    "TargetSet",
    "historical_targets",
    "marking_targets",
    "sabr_reduce",
    "surface_atm_derivatives",
]
