"""Staged fitting of the two-factor Bergomi parameters to a surface history (SPEC §15 Part 3;
Bergomi ch. 7 §7.4, ch. 8 §8.7, ch. 9, ch. 12 conclusion).

In an LSV the leverage absorbs the vanilla surface, so vanillas identify none of the seven
parameters; each group is pinned by a non-vanilla observable and fitted in stages:

* **Stage 1 — variance dynamics** ``(ν, θ, k1, k2 | ρ12)``: weighted least squares in
  ``log volvol`` of the model's instantaneous vol of VS vols (eq. 7.39 — flat curve, or the
  non-flat form eq. 7.36–7.38 on the window's mean VS curve) against ``volvol_hist(T)`` at the
  pillars, weights ``1/SE²`` (the standard error of ``log volvol``); optional VIX-implied vol of
  vol targets at short horizons with their own weight.  ``ρ12`` is fixed (0 or the user's value)
  or, with ``rho12_mode="from_correlation"``, chosen so that the model's correlation of
  ``Δ ln vs_vol`` between two pillars (eq. 7.20 machinery, :func:`~volsto.calibration.history.
  vs_vol_correlation`) matches the historical one — two alternations of the two fits.  Bounds:
  ``ν ∈ [0.3, 4]``, ``θ ∈ [0, 1]``, ``k1 ∈ [1, 20]``, ``k2 ∈ [0.05, 1.5]``, ``k1 > k2``.  The
  report carries the Jacobian singular values, the parameter standard errors from
  ``(JᵀJ)⁻¹`` and a flag on ``k2`` when it sits on a bound or is insensitive (the smallest
  singular direction loads on it).  Degeneracy check: the fit is rerun from Bergomi's Table 7.1
  Sets I–III (``ν, θ, k1, k2, ρ12`` = (1.50, 0.312, 2.63, 0.42, −0.7), (1.74, 0.245, 5.35, 0.28,
  0), (1.86, 0.230, 7.54, 0.24, +0.7)) and every optimum is reported — for a given decay
  exponent many sets fit the vol-of-vol curve equally well; the correlation target separates
  them (§7.4.2: higher forward-variance vols go with lower correlations, and correlations are
  invariant under a common shift of the ``k``'s).
* **Stage 2 — spot/vol correlations** ``(ρ_SX1, ρ_SX2)`` with stage 1 frozen, in the eq. 8.56
  form ``ρ_SX2 = ρ12 ρ_SX1 + χ sqrt(1 − ρ12²) sqrt(1 − ρ_SX1²)``, ``χ ∈ [−1, 1]``, so the
  correlation matrix is positive semi-definite by construction.  Targets: the ATMF skew term
  structure of the pricing date and ``SSR_hist(T)`` on the short window; model: order-one skew
  eq. 8.55 (9.18 on a sloping curve) and SSR eq. 9.21 (9.19); weights put the skew residuals in
  vol points per unit ``k`` divided by 10 and the SSR residuals in units of 0.1, SSR pillars
  beyond ``ssr_max_T`` down-weighted.  Refinement: the order-one skew is replaced by the
  mixing-solution skew of the naked 2F model (:func:`~volsto.calibration.history.mixing_atmf_batch`,
  the exact strike derivative; a multiplicative correction per pillar re-evaluated at the current
  parameters, two passes) and, with ``refine_ssr``, the order-one SSR by the numerical SSR of the
  naked kernel (:func:`~volsto.analytics.smile_dynamics.ssr_numerical_many`) the same way — the
  order-one SSR overstates the model's own by about 0.1 at ν ≈ 1.7.
  Output: the correlations, and the naked-skew-versus-market-skew table — the leverage absorbs
  the residual, a small one means the SV carries the skew (eq. 12.52: ``R_T ≈ R_T^SV``).  The
  book's remark after eq. 8.55 — rescaling ``(ρ_SX1, ρ_SX2)`` by a constant and ``ν`` by its
  inverse leaves the order-one skew unchanged — is the direction stage 1 pins (the ``ν``
  scale); :func:`skew_scale_degeneracy` reports the stage-2 objective along it.
* **Stage 3 — validation on the calibrated LSV** (:func:`stage3_validation`; nothing is
  refit): calibrate the leverage to the pricing date's surface, report the mean ``|L − 1|``
  over the leverage grid, the model SSR (numerical, :func:`~volsto.analytics.smile_dynamics.
  ssr_numerical`) against ``SSR_hist``, the model vol of vol of VS vols with leverage
  (:func:`~volsto.analytics.smile_dynamics.volvol_term_structure`) against ``volvol_hist``,
  the forward-start ATM vol and the forward 90/110 skew at 1y-into-1y and 2y-into-1y, and
  optionally the M6 headline product table on this fit.

:func:`fit_2f` runs stages 1–2 (stage 3 through ``stage3=``) and returns a :class:`FitResult`
whose ``config_yaml`` is a loadable model config (``model`` section = :class:`BergomiParams`)
with the provenance (window, settings, objectives, optima).  Checked by
``tests/test_fit_2f.py``: recovery on the synthetic history, the Table 7.1 degeneracy with and
without the correlation target, and the real-data end-to-end run on the 2022 H2 sample.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from scipy.optimize import brentq, least_squares

from volsto.analytics.bergomi import (
    atmf_skew_order1,
    atmf_skew_order1_flat,
    ssr_order1,
    ssr_order1_flat,
    vs_vol_of_vol,
    vs_vol_of_vol_flat,
)
from volsto.calibration.history import (
    DEFAULT_PILLARS,
    WINDOW_SSR,
    WINDOW_VOL,
    HistoryEstimates,
    SurfaceHistory,
    estimate_history,
    vs_vol_correlation,
)
from volsto.config import BergomiParams, to_mapping
from volsto.market.varswap import ForwardVarianceCurve

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

#: Bergomi Table 7.1 (book p. 230): (ν, θ, k1, k2, ρ12) matching the benchmark vol-of-vol curve
TABLE_7_1: dict[str, tuple[float, float, float, float, float]] = {
    "Set I": (1.50, 0.312, 2.63, 0.42, -0.70),
    "Set II": (1.74, 0.245, 5.35, 0.28, 0.00),
    "Set III": (1.86, 0.230, 7.54, 0.24, 0.70),
}
RHO12_MODES = ("fixed", "from_correlation")
CURVES = ("flat", "window_mean")


@dataclass(frozen=True)
class Fit2FConfig:
    """Settings of :func:`fit_2f` (every default is the SPEC §15 Part 3 convention)."""

    pillars: tuple[float, ...] = tuple(DEFAULT_PILLARS)
    window_vol: int = WINDOW_VOL
    window_ssr: int = WINDOW_SSR
    rho12: float = 0.0
    rho12_mode: str = "fixed"
    corr_pillars: tuple[float, float] = (0.25, 2.0)
    curve: str = "window_mean"
    bounds_nu: tuple[float, float] = (0.3, 4.0)
    bounds_theta: tuple[float, float] = (0.0, 1.0)
    bounds_k1: tuple[float, float] = (1.0, 20.0)
    bounds_k2: tuple[float, float] = (0.05, 1.5)
    starts: tuple[str, ...] = ("Set I", "Set II", "Set III")
    vix_targets: tuple[tuple[float, float, float], ...] = ()  # (T, implied vol of vol, se)
    vix_weight: float = 1.0
    skew_scale: float = 0.1  # skew residuals: vol per unit k / 10 → units of 0.1
    ssr_scale: float = 0.1
    ssr_max_T: float = 1.0
    ssr_far_weight: float = 0.5
    refine_mixing: bool = True
    mixing_paths: int = 100_000
    mixing_seed: int = 0
    mixing_passes: int = 2
    refine_ssr: bool = True
    ssr_paths: int = 100_000
    k_separation: float = 0.05  # k1 − k2 ≥ this

    def __post_init__(self) -> None:
        if self.rho12_mode not in RHO12_MODES:
            raise ValueError(f"rho12_mode must be one of {RHO12_MODES}")
        if self.curve not in CURVES:
            raise ValueError(f"curve must be one of {CURVES}")
        if not -1.0 < self.rho12 < 1.0:
            raise ValueError("rho12 must lie in (-1, 1)")
        for name in self.starts:
            if name not in TABLE_7_1:
                raise ValueError(f"unknown start {name!r}; Table 7.1 sets are {list(TABLE_7_1)}")
        if len(self.pillars) < 3 or any(t <= 0 for t in self.pillars):
            raise ValueError("at least three positive pillars are needed")
        for lo, hi in (self.bounds_nu, self.bounds_theta, self.bounds_k1, self.bounds_k2):
            if not lo < hi:
                raise ValueError("bounds must be increasing")


# --------------------------------------------------------------------------------------------
# targets
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage1Targets:
    """``volvol_hist`` per pillar with the standard error of its log, the historical
    correlation of ``Δ ln vs_vol`` between the correlation pillars, the window's mean VS
    curve and the VIX targets."""

    pillars: FloatArray
    volvol: FloatArray
    log_se: FloatArray
    corr_hist: float
    corr_se: float
    curve: ForwardVarianceCurve | None
    vix: tuple[tuple[float, float, float], ...]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"T": self.pillars, "volvol_hist": self.volvol, "log_se": self.log_se})


@dataclass(frozen=True)
class Stage2Targets:
    pillars: FloatArray
    skew: FloatArray  # latest ATMF skew per pillar (vol per unit k)
    ssr: FloatArray
    ssr_se: FloatArray
    curve: ForwardVarianceCurve | None  # the pricing date's VS curve (None: flat expansion)
    sigma_ref: float


def _corr_entry(
    corr: pd.DataFrame, se: pd.DataFrame, pillars: tuple[float, float]
) -> tuple[float, float]:
    """Historical correlation (and Fisher se) between two pillars of the correlation matrix."""
    idx = np.asarray(corr.index, dtype=np.float64)
    i = int(np.argmin(np.abs(idx - pillars[0])))
    j = int(np.argmin(np.abs(idx - pillars[1])))
    if abs(idx[i] - pillars[0]) > 1e-9 or abs(idx[j] - pillars[1]) > 1e-9:
        return float("nan"), float("nan")
    return float(corr.to_numpy()[i, j]), float(se.to_numpy()[i, j])


def _lookup(frame: pd.DataFrame, T: float, column: str) -> float:
    """``frame[column]`` at the pillar ``T`` of a ``T``-indexed frame (NaN when absent)."""
    idx = np.asarray(frame.index, dtype=np.float64)
    i = int(np.argmin(np.abs(idx - T)))
    if abs(idx[i] - T) > 1e-9:
        return float("nan")
    return float(frame[column].to_numpy()[i])


def _mean_vs_curve(
    history: SurfaceHistory, pillars: FloatArray, window: int, end: pd.Timestamp
) -> ForwardVarianceCurve:
    """Forward-variance curve of the window's mean VS vols per pillar (total variance
    ``T σ̂_T²`` interpolated by :class:`ForwardVarianceCurve`)."""
    e = history.date_index(end)
    rows = slice(max(e + 1 - window, 0), e + 1)
    vs = np.array(
        [history.vs_vol.to_numpy()[rows, history.pillar_index(float(T))].mean() for T in pillars]
    )
    return _curve_from_vs(pillars, vs)


def _curve_from_vs(pillars: FloatArray, vs_vol: FloatArray) -> ForwardVarianceCurve:
    mats = np.concatenate(([0.5 * pillars[0]], pillars, [pillars[-1] + 1.0, pillars[-1] + 5.0]))
    vs = np.concatenate(([vs_vol[0]], vs_vol, [vs_vol[-1], vs_vol[-1]]))
    W = vs * vs * mats
    W = np.maximum.accumulate(W + 1e-12 * np.arange(W.size))  # monotone total variance
    return ForwardVarianceCurve(mats, W)


def stage1_targets(
    history: SurfaceHistory, cfg: Fit2FConfig, end: pd.Timestamp | str | None = None
) -> Stage1Targets:
    e = history.date_index(end)
    end_ts = history.dates[e]
    ps = np.asarray(cfg.pillars, dtype=np.float64)
    vv = [history.volvol_hist(float(T), cfg.window_vol, end=end_ts) for T in ps]
    volvol = np.array([v.volvol for v in vv])
    log_se = np.array([v.se / v.volvol for v in vv])
    pc = history.pillar_correlations(cfg.window_vol, end=end_ts)
    corr, corr_se = _corr_entry(pc.corr, pc.se_fisher, cfg.corr_pillars)
    curve = (
        _mean_vs_curve(history, ps, cfg.window_vol, end_ts) if cfg.curve == "window_mean" else None
    )
    return Stage1Targets(ps, volvol, log_se, corr, corr_se, curve, cfg.vix_targets)


def stage2_targets(
    history: SurfaceHistory, cfg: Fit2FConfig, end: pd.Timestamp | str | None = None
) -> Stage2Targets:
    e = history.date_index(end)
    end_ts = history.dates[e]
    ps = np.asarray(cfg.pillars, dtype=np.float64)
    skew = np.array([history.skew.to_numpy()[e, history.pillar_index(float(T))] for T in ps])
    ssr_rows = [history.ssr_hist(float(T), cfg.window_ssr, end=end_ts) for T in ps]
    vs_today = np.array([history.vs_vol.to_numpy()[e, history.pillar_index(float(T))] for T in ps])
    curve = _curve_from_vs(ps, vs_today) if cfg.curve == "window_mean" else None
    return Stage2Targets(
        ps,
        skew,
        np.array([r.ssr for r in ssr_rows]),
        np.array([r.se for r in ssr_rows]),
        curve,
        float(vs_today[0]),
    )


# --------------------------------------------------------------------------------------------
# stage 1
# --------------------------------------------------------------------------------------------


def _params(x: FloatArray, rho12: float, rho_s: tuple[float, float] = (0.0, 0.0)) -> BergomiParams:
    # stage 1 never touches the spot correlations: zeros keep the matrix PSD for any ρ12
    nu, theta, k1, k2 = (float(v) for v in x)
    return BergomiParams(nu, theta, k1, k2, rho12, rho_s[0], rho_s[1])


def model_volvol(
    p: BergomiParams, pillars: FloatArray, curve: ForwardVarianceCurve | None
) -> FloatArray:
    """Instantaneous vol of VS vols at ``t = 0`` per pillar: eq. 7.39 (flat) or on ``curve``."""
    if curve is None:
        return np.asarray(vs_vol_of_vol_flat(p, pillars), dtype=np.float64)
    return np.array([float(vs_vol_of_vol(p, float(T), curve)) for T in pillars])


def _vix_model_volvol(p: BergomiParams, T: float, curve: ForwardVarianceCurve | None) -> float:
    """Model implied vol of vol of the VIX option at ``T`` (quadrature, flat curve at the
    reference level of ``curve`` — the stage-1 target uses the model's own dynamics only)."""
    from volsto.analytics.vix import vix_quadrature
    from volsto.market import ForwardCurve
    from volsto.models.bergomi import BergomiSV

    xi = curve if curve is not None else ForwardVarianceCurve.flat(0.04, 10.0)
    model = BergomiSV(p, xi, ForwardCurve.flat(100.0, 0.0, 0.0))
    return float(vix_quadrature(model, T, n_hermite=30).atm_vol_of_vol)


@dataclass(frozen=True)
class Stage1Optimum:
    start: str
    params: BergomiParams
    objective: float
    residuals: FloatArray
    stderr: dict[str, float]
    singular_values: FloatArray
    k2_flag: str
    n_evaluations: int
    rho12_iterations: list[tuple[float, float]]  # (ρ12, model corr) per alternation


@dataclass(frozen=True)
class Stage1Report:
    targets: Stage1Targets
    optima: list[Stage1Optimum]
    best: Stage1Optimum
    rho12_mode: str
    curve: str
    unique: bool  # the optima agree on (ν, θ, k1, k2) within 5% / 0.05 of ν, θ

    def table(self) -> pd.DataFrame:
        rows = []
        for o in self.optima:
            p = o.params
            rows.append(
                {
                    "start": o.start,
                    "nu": p.nu,
                    "theta": p.theta,
                    "k1": p.k1,
                    "k2": p.k2,
                    "rho12": p.rho12,
                    "objective": o.objective,
                    "nu_se": o.stderr["nu"],
                    "theta_se": o.stderr["theta"],
                    "k1_se": o.stderr["k1"],
                    "k2_se": o.stderr["k2"],
                    "k2_flag": o.k2_flag,
                    "sv_min": float(o.singular_values.min()),
                    "sv_max": float(o.singular_values.max()),
                }
            )
        return pd.DataFrame(rows)

    def fit_table(self) -> pd.DataFrame:
        t = self.targets
        p = self.best.params
        return pd.DataFrame(
            {
                "T": t.pillars,
                "volvol_hist": t.volvol,
                "volvol_model": model_volvol(p, t.pillars, t.curve),
                "log_se": t.log_se,
            }
        )


def _stage1_residuals(
    cfg: Fit2FConfig, tg: Stage1Targets
) -> Callable[[FloatArray, float], FloatArray]:
    def resid(x: FloatArray, rho12: float) -> FloatArray:
        p = _params(x, rho12)
        r = (np.log(model_volvol(p, tg.pillars, tg.curve)) - np.log(tg.volvol)) / tg.log_se
        extra = [100.0 * max(0.0, cfg.k_separation - (x[2] - x[3]))]  # k1 > k2 soft wall
        for T, target, se in tg.vix:
            extra.append(
                cfg.vix_weight
                * (np.log(_vix_model_volvol(p, T, tg.curve)) - np.log(target))
                / max(se / target, 1e-3)
            )
        return np.concatenate((r, extra))

    return resid


def _fit_stage1_fixed_rho(
    cfg: Fit2FConfig, tg: Stage1Targets, x0: FloatArray, rho12: float
) -> tuple[FloatArray, Any]:
    resid = _stage1_residuals(cfg, tg)
    lo = [cfg.bounds_nu[0], cfg.bounds_theta[0], cfg.bounds_k1[0], cfg.bounds_k2[0]]
    hi = [cfg.bounds_nu[1], cfg.bounds_theta[1], cfg.bounds_k1[1], cfg.bounds_k2[1]]
    x0c = np.clip(x0, np.asarray(lo) + 1e-6, np.asarray(hi) - 1e-6)
    res = least_squares(  # type: ignore[call-overload]
        lambda x: resid(x, rho12), x0c, bounds=(lo, hi), x_scale="jac", max_nfev=400
    )
    return np.asarray(res.x, dtype=np.float64), res


def _rho12_from_correlation(cfg: Fit2FConfig, tg: Stage1Targets, x: FloatArray) -> float:
    """``ρ12`` matching the model's correlation of ``Δ ln vs_vol`` between the correlation
    pillars to the historical one (a root in ``(−0.99, 0.99)``; the closer bound if none)."""
    t1, t2 = cfg.corr_pillars
    xi = (
        None if tg.curve is None else (lambda s, c=tg.curve: np.asarray(c.xi0(s), dtype=np.float64))
    )

    def gap(r: float) -> float:
        return vs_vol_correlation(_params(x, r), t1, t2, xi=xi) - tg.corr_hist

    lo, hi = -0.99, 0.99
    g_lo, g_hi = gap(lo), gap(hi)
    if g_lo * g_hi > 0:
        return lo if abs(g_lo) < abs(g_hi) else hi
    return float(brentq(gap, lo, hi, xtol=1e-6))


def _stage1_uncertainty(
    res: Any, x: FloatArray, cfg: Fit2FConfig
) -> tuple[dict[str, float], FloatArray, str]:
    J = np.asarray(res.jac, dtype=np.float64)
    sv = np.linalg.svd(J, compute_uv=False)
    names = ("nu", "theta", "k1", "k2")
    try:
        cov = np.linalg.pinv(J.T @ J)
        se = {n: float(np.sqrt(max(cov[i, i], 0.0))) for i, n in enumerate(names)}
    except np.linalg.LinAlgError:  # pragma: no cover
        se = dict.fromkeys(names, float("nan"))
    _, _, vt = np.linalg.svd(J)
    weakest = vt[-1]
    flags = []
    lo, hi = cfg.bounds_k2
    if x[3] <= lo * 1.01 or x[3] >= hi * 0.99:
        flags.append("on bound")
    if abs(weakest[3]) > 0.7:
        flags.append("weakest direction")
    if se["k2"] > 0.5 * x[3]:
        flags.append("se > 50%")
    return se, sv, ", ".join(flags) if flags else "identified"


def fit_stage1(
    history: SurfaceHistory, cfg: Fit2FConfig, end: pd.Timestamp | str | None = None
) -> Stage1Report:
    """Stage 1 from every configured Table 7.1 start (module docstring)."""
    tg = stage1_targets(history, cfg, end)
    optima = []
    for name in cfg.starts:
        nu0, th0, k10, k20, r0 = TABLE_7_1[name]
        x = np.array([nu0, th0, k10, k20])
        rho12 = cfg.rho12 if cfg.rho12_mode == "fixed" else r0
        its: list[tuple[float, float]] = []
        n_ev = 0
        passes = 1 if cfg.rho12_mode == "fixed" else 3
        for _ in range(passes):
            x, res = _fit_stage1_fixed_rho(cfg, tg, x, rho12)
            n_ev += int(res.nfev)
            if cfg.rho12_mode == "from_correlation" and np.isfinite(tg.corr_hist):
                rho12 = _rho12_from_correlation(cfg, tg, x)
                t1, t2 = cfg.corr_pillars
                xi = (
                    None
                    if tg.curve is None
                    else (lambda s, c=tg.curve: np.asarray(c.xi0(s), dtype=np.float64))
                )
                its.append((rho12, vs_vol_correlation(_params(x, rho12), t1, t2, xi=xi)))
        x, res = _fit_stage1_fixed_rho(cfg, tg, x, rho12)  # final fit at the settled ρ12
        n_ev += int(res.nfev)
        se, sv, flag = _stage1_uncertainty(res, x, cfg)
        optima.append(
            Stage1Optimum(
                name,
                _params(x, rho12),
                float(2.0 * res.cost),
                np.asarray(res.fun),
                se,
                sv,
                flag,
                n_ev,
                its,
            )
        )
    best = min(optima, key=lambda o: o.objective)
    nus = np.array([o.params.nu for o in optima])
    ths = np.array([o.params.theta for o in optima])
    k1s = np.array([o.params.k1 for o in optima])
    unique = bool(
        np.ptp(nus) < 0.05 * best.params.nu
        and np.ptp(ths) < 0.05
        and np.ptp(k1s) < 0.1 * best.params.k1
    )
    return Stage1Report(tg, optima, best, cfg.rho12_mode, cfg.curve, unique)


# --------------------------------------------------------------------------------------------
# stage 2
# --------------------------------------------------------------------------------------------


def rho_sx2_from_chi(rho12: float, rho_sx1: float, chi: float) -> float:
    """Eq. 8.56: ``ρ_SX2 = ρ12 ρ_SX1 + χ sqrt(1 − ρ12²) sqrt(1 − ρ_SX1²)``."""
    return float(
        rho12 * rho_sx1 + chi * np.sqrt(1.0 - rho12 * rho12) * np.sqrt(1.0 - rho_sx1 * rho_sx1)
    )


def chi_from_rhos(rho12: float, rho_sx1: float, rho_sx2: float) -> float:
    d = np.sqrt(1.0 - rho12 * rho12) * np.sqrt(1.0 - rho_sx1 * rho_sx1)
    return float((rho_sx2 - rho12 * rho_sx1) / d) if d > 0 else 0.0


def model_skew_ssr(
    p: BergomiParams, pillars: FloatArray, curve: ForwardVarianceCurve | None
) -> tuple[FloatArray, FloatArray]:
    """Order-one ATMF skew (eq. 8.55 / 9.18) and SSR (eq. 9.21 / 9.19) per pillar."""
    if curve is None:
        return (
            np.asarray(atmf_skew_order1_flat(p, pillars), dtype=np.float64),
            np.asarray(ssr_order1_flat(p, pillars), dtype=np.float64),
        )
    skew = np.array([float(atmf_skew_order1(p, curve, float(T))) for T in pillars])
    ssr = np.array([float(ssr_order1(p, curve, float(T))) for T in pillars])
    return skew, ssr


@dataclass(frozen=True)
class Stage2Coefficients:
    """Per pillar, the curve-dependent integrals of the order-one skew and SSR, which are
    linear / rational in the spot correlations: ``skew = a1 ρ_SX1 + a2 ρ_SX2`` and
    ``SSR = (n1 ρ_SX1 + n2 ρ_SX2) / (d1 ρ_SX1 + d2 ρ_SX2)`` (eqs. 8.54 / 9.19: ``D`` and ``N``
    of :func:`~volsto.analytics.bergomi._skew_integrals` split by factor).  Computed once per
    stage-2 call with the stage-1 parameters frozen; the optimisation then costs nothing."""

    pillars: FloatArray
    a: FloatArray  # (n, 2)
    n: FloatArray  # (n, 2)
    d: FloatArray  # (n, 2)

    def skew_ssr(self, rho1: float, rho2: float) -> tuple[FloatArray, FloatArray]:
        r = np.array([rho1, rho2])
        num = self.n @ r
        den = self.d @ r
        with np.errstate(divide="ignore", invalid="ignore"):
            ssr = np.where(np.abs(den) > 0, num / den, np.nan)
        return np.asarray(self.a @ r, dtype=np.float64), np.asarray(ssr, dtype=np.float64)


def stage2_coefficients(
    p: BergomiParams, pillars: FloatArray, curve: ForwardVarianceCurve | None
) -> Stage2Coefficients:
    """:class:`Stage2Coefficients` of ``p``'s variance dynamics (its own correlations are
    ignored): two evaluations of the order-one formulas with unit loadings on each factor."""
    probe = 0.1  # keeps the correlation matrix PSD for any ρ12 in (−1, 1)
    a = np.empty((pillars.size, 2))
    n = np.empty((pillars.size, 2))
    d = np.empty((pillars.size, 2))
    for j, (r1, r2) in enumerate(((probe, 0.0), (0.0, probe))):
        q = dataclasses.replace(p, rho_SX1=r1, rho_SX2=r2)
        skew, ssr = model_skew_ssr(q, pillars, curve)
        a[:, j] = skew / probe
        # SSR = N/D: recover N and D up to the common factor by fixing D_j = skew_j's scale
        d[:, j] = skew / probe
        n[:, j] = ssr * skew / probe
    return Stage2Coefficients(pillars, a, n, d)


@dataclass(frozen=True)
class Stage2Report:
    targets: Stage2Targets
    params: BergomiParams
    chi: float
    objective: float
    residuals: FloatArray
    stderr: dict[str, float]
    skew_correction: FloatArray  # mixing / order-one skew ratio per pillar (1 without refinement)
    skew_mixing: FloatArray | None
    skew_mixing_se: FloatArray | None
    passes: int
    n_distinct_optima: int = 1
    ssr_correction: FloatArray | None = None  # numerical / order-one SSR per pillar
    ssr_numerical: FloatArray | None = None
    ssr_numerical_se: FloatArray | None = None

    def table(self) -> pd.DataFrame:
        """Naked 2F skew (order one and mixing) against the market skew, model SSR against the
        historical one, per pillar."""
        t = self.targets
        skew1, ssr = model_skew_ssr(self.params, t.pillars, t.curve)
        return pd.DataFrame(
            {
                "T": t.pillars,
                "skew_market": t.skew,
                "skew_2f_order1": skew1,
                "skew_2f_mixing": skew1 * self.skew_correction,
                "skew_residual": skew1 * self.skew_correction - t.skew,
                "ssr_hist": t.ssr,
                "ssr_hist_se": t.ssr_se,
                "ssr_2f_order1": ssr,
                "ssr_2f": ssr * (self.ssr_correction if self.ssr_correction is not None else 1.0),
            }
        )


def _stage2_weights(
    cfg: Fit2FConfig, pillars: FloatArray, ssr_se: FloatArray | None = None
) -> tuple[FloatArray, FloatArray]:
    """Skew residuals in units of ``skew_scale`` (0.1 vol per unit k); SSR residuals in units of
    ``max(ssr_scale, SE)`` — the SPEC's 0.1 unless the short window's standard error is larger
    (a noisy 60-day SSR must not dominate the skew term structure; recorded in §15 notes) —
    and ``ssr_far_weight`` beyond ``ssr_max_T``."""
    w_skew = np.full(pillars.size, 1.0 / cfg.skew_scale)
    scale = np.full(pillars.size, cfg.ssr_scale)
    if ssr_se is not None:
        scale = np.maximum(scale, np.where(np.isfinite(ssr_se), ssr_se, cfg.ssr_scale))
    w_ssr = np.where(pillars > cfg.ssr_max_T, cfg.ssr_far_weight, 1.0) / scale
    return w_skew, w_ssr


def fit_stage2(
    history: SurfaceHistory,
    cfg: Fit2FConfig,
    stage1: BergomiParams,
    end: pd.Timestamp | str | None = None,
    *,
    x0: tuple[float, float] | Sequence[tuple[float, float]] | None = None,
) -> Stage2Report:
    """Stage 2 (module docstring): ``(ρ_SX1, χ)`` by weighted least squares on the skew term
    structure and the SSR from several starts (the objective is not convex in ``(ρ_SX1, χ)``:
    a second basin with ``ρ_SX2 > 0`` exists; the best optimum is kept and the number of
    distinct optima reported), then the mixing refinement from the best."""
    tg = stage2_targets(history, cfg, end)
    w_skew, w_ssr = _stage2_weights(cfg, tg.pillars, tg.ssr_se)
    rho12 = stage1.rho12
    correction = np.ones(tg.pillars.size)
    skew_mix: FloatArray | None = None
    skew_mix_se: FloatArray | None = None

    def with_rhos(y: FloatArray) -> BergomiParams:
        r1, chi = float(y[0]), float(y[1])
        return dataclasses.replace(stage1, rho_SX1=r1, rho_SX2=rho_sx2_from_chi(rho12, r1, chi))

    coef = stage2_coefficients(stage1, tg.pillars, tg.curve)
    ssr_corr = np.ones(tg.pillars.size)
    ssr_num: FloatArray | None = None
    ssr_num_se: FloatArray | None = None

    def resid(y: FloatArray) -> FloatArray:
        r1 = float(y[0])
        skew, ssr = coef.skew_ssr(r1, rho_sx2_from_chi(rho12, r1, float(y[1])))
        r = np.concatenate(
            ((skew * correction - tg.skew) * w_skew, (ssr * ssr_corr - tg.ssr) * w_ssr)
        )
        # both correlations zero: the order-one SSR is 0/0 — a wall, never an optimum
        return np.where(np.isfinite(r), r, 1e3)

    bounds = ([-0.999, -1.0], [0.999, 1.0])
    starts: list[tuple[float, float]]
    if x0 is None:
        starts = [(r, c) for r in (-0.9, -0.6, -0.3, -0.05) for c in (-0.5, 0.0, 0.5)]
    elif len(x0) == 2 and all(isinstance(v, float | int) for v in x0):
        starts = [(float(x0[0]), float(x0[1]))]  # type: ignore[arg-type]
    else:
        starts = [(float(a), float(b)) for a, b in x0]  # type: ignore[misc]
    fits = []
    for st in starts:
        r = least_squares(
            resid, np.array(st, dtype=np.float64), bounds=bounds, x_scale="jac", max_nfev=200
        )
        fits.append(r)
    fits.sort(key=lambda r: r.cost)
    res = fits[0]
    y = np.asarray(res.x)
    n_distinct = len(
        {
            (round(float(r.x[0]), 2), round(float(r.x[1]), 2))
            for r in fits
            if r.cost < 10 * res.cost + 1e-9
        }
    )
    passes = 0
    if cfg.refine_mixing:
        from volsto.calibration.history import mixing_atmf_batch
        from volsto.market import ForwardCurve
        from volsto.models.bergomi import BergomiSV

        xi = tg.curve if tg.curve is not None else ForwardVarianceCurve.flat(tg.sigma_ref**2, 10.0)
        nf = 1 if stage1.is_one_factor else 2
        for _ in range(cfg.mixing_passes):
            p = with_rhos(y)
            model = BergomiSV(p, xi, ForwardCurve.flat(100.0, 0.0, 0.0))
            skew1, _ = model_skew_ssr(p, tg.pillars, tg.curve)
            # the exact strike derivative of the mixing batch (20× less noisy than the OTM
            # difference of mixing_atmf_skew, whose ±0.05 per pillar would swamp the fit)
            mix = [
                mixing_atmf_batch(
                    model,
                    float(T),
                    np.array([0.0]),
                    np.zeros((1, nf)),
                    n_paths=cfg.mixing_paths,
                    seed=cfg.mixing_seed + j,
                    dt=1.0 / 365.0,
                )
                for j, T in enumerate(tg.pillars)
            ]
            skew_mix = np.array([float(m.skew[0]) for m in mix])
            skew_mix_se = np.array([float(m.skew_se[0]) for m in mix])
            correction = np.where(np.abs(skew1) > 1e-8, skew_mix / skew1, 1.0)
            if cfg.refine_ssr:
                # the same multiplicative correction for the SSR: the order-one formula (VS vol
                # as the ATMF vol) overstates the model's own SSR by ≈ 0.1 at ν ≈ 1.7
                from volsto.analytics.smile_dynamics import ssr_numerical_many
                from volsto.config import SimConfig

                _, ssr1 = model_skew_ssr(p, tg.pillars, tg.curve)
                sim = SimConfig(
                    n_paths=cfg.ssr_paths,
                    dt_max=1.0 / 100.0,
                    chunk_size=cfg.ssr_paths,
                    seed=cfg.mixing_seed + 100,
                )
                rows = ssr_numerical_many(model, list(tg.pillars), eps=0.05, sim=sim)
                ssr_num = np.array([r.R for r in rows])
                ssr_num_se = np.array([r.R_stderr for r in rows])
                ssr_corr = np.where(np.abs(ssr1) > 1e-8, ssr_num / ssr1, 1.0)
            fits = [
                least_squares(
                    resid,
                    np.array(st, dtype=np.float64),
                    bounds=bounds,
                    x_scale="jac",
                    max_nfev=200,
                )
                for st in [tuple(y), *starts[:4]]
            ]
            fits.sort(key=lambda r: r.cost)
            res = fits[0]
            y = np.asarray(res.x)
            passes += 1
    J = np.asarray(res.jac)
    cov = np.linalg.pinv(J.T @ J)
    se = {
        "rho_SX1": float(np.sqrt(max(cov[0, 0], 0.0))),
        "chi": float(np.sqrt(max(cov[1, 1], 0.0))),
    }
    p = with_rhos(y)
    return Stage2Report(
        tg,
        p,
        float(y[1]),
        float(2.0 * res.cost),
        np.asarray(res.fun),
        se,
        correction,
        skew_mix,
        skew_mix_se,
        passes,
        n_distinct,
        ssr_corr if cfg.refine_ssr and passes else None,
        ssr_num,
        ssr_num_se,
    )


def skew_scale_degeneracy(
    history: SurfaceHistory,
    cfg: Fit2FConfig,
    params: BergomiParams,
    scales: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0),
    end: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """The book's remark after eq. 8.55: along ``ν → c ν``, ``(ρ_SX1, ρ_SX2) → (ρ_SX1, ρ_SX2)/c``
    the order-one skew is unchanged; stage 2 alone cannot tell the points apart, stage 1
    (vol of vol ∝ ν) does.  Per scale: the stage-1 and stage-2 objectives (weighted sums of
    squares) and the PSD status of the correlation matrix."""
    tg1 = stage1_targets(history, cfg, end)
    tg2 = stage2_targets(history, cfg, end)
    w_skew, w_ssr = _stage2_weights(cfg, tg2.pillars, tg2.ssr_se)
    rows = []
    for c in scales:
        r1, r2 = params.rho_SX1 / c, params.rho_SX2 / c
        psd = abs(r1) < 1 and abs(r2) < 1
        try:
            p = dataclasses.replace(params, nu=params.nu * c, rho_SX1=r1, rho_SX2=r2)
        except ValueError:
            psd = False
            p = None
        if p is None:
            rows.append(
                {
                    "scale": c,
                    "nu": params.nu * c,
                    "rho_SX1": r1,
                    "rho_SX2": r2,
                    "psd": False,
                    "stage1_objective": np.nan,
                    "stage2_objective": np.nan,
                }
            )
            continue
        r_1 = (np.log(model_volvol(p, tg1.pillars, tg1.curve)) - np.log(tg1.volvol)) / tg1.log_se
        skew, ssr = stage2_coefficients(p, tg2.pillars, tg2.curve).skew_ssr(r1, r2)
        r_2 = np.concatenate(((skew - tg2.skew) * w_skew, (ssr - tg2.ssr) * w_ssr))
        rows.append(
            {
                "scale": c,
                "nu": p.nu,
                "rho_SX1": r1,
                "rho_SX2": r2,
                "psd": psd,
                "stage1_objective": float(r_1 @ r_1),
                "stage2_objective": float(r_2 @ r_2),
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# stage 3
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage3Inputs:
    """What the validation needs beyond the fit: the pricing date's implied surface, the
    particle / simulation settings, the optional local-vol config, the pricing configuration
    and the estimates to compare with."""

    surface: Any  # ImpliedSurface
    particle: Any  # ParticleConfig
    sim: Any  # SimConfig (calibration schedule; pricing uses n_paths of pricing_sim)
    pricing_sim: Any  # SimConfig
    local_vol: Any | None = None
    ssr_pillars: tuple[float, ...] = (0.25, 1.0)
    volvol_pillars: tuple[float, ...] = (0.25, 1.0, 2.0)
    forward_starts: tuple[tuple[float, float], ...] = ((1.0, 2.0), (2.0, 3.0))
    headline: bool = False
    eps: float = 0.05


@dataclass(frozen=True)
class Stage3Report:
    mean_abs_l_minus_1: float
    leverage_table: pd.DataFrame
    ssr_table: pd.DataFrame
    volvol_table: pd.DataFrame
    forward_table: pd.DataFrame
    headline: pd.DataFrame | None
    calibration_seconds: float
    wall_seconds: float


def stage3_validation(
    params: BergomiParams, inputs: Stage3Inputs, estimates: HistoryEstimates
) -> Stage3Report:
    """Stage 3 (module docstring): calibrate the leverage on ``inputs.surface`` for ``params``
    and report the validation tables; nothing is refit."""
    from volsto.analytics.forward_smile import forward_smile
    from volsto.analytics.smile_dynamics import ssr_numerical_many, volvol_term_structure
    from volsto.calibration.particle import calibrate_leverage
    from volsto.market.varswap import xi0_curve
    from volsto.models.bergomi import BergomiSV
    from volsto.models.lsv import LSV

    t_all = time.perf_counter()
    surface = inputs.surface
    fc = surface.forward_curve
    xi0 = xi0_curve(surface, min(surface.max_maturity, max(inputs.particle.horizon + 1.0, 5.0)))
    kernel = BergomiSV(params, xi0, fc)
    t0 = time.perf_counter()
    result = calibrate_leverage(
        surface, kernel, inputs.particle, inputs.sim, local_vol_cfg=inputs.local_vol
    )
    cal_s = time.perf_counter() - t0
    lsv = LSV(kernel, result.leverage)
    lev = result.leverage
    # mean |L − 1| over the leverage grid inside ±2 sd of log-moneyness per time
    rows = []
    for i, t in enumerate(lev.times):
        if t <= 0:
            continue
        sd = float(surface.atm_vol(t)) * np.sqrt(t)
        k = lev.k_grid
        inside = np.abs(k) <= 2.0 * sd
        vals = lev.values[i][inside] if lev.values.ndim == 2 else lev.values[inside]
        rows.append(
            {
                "t": float(t),
                "mean_abs_L_minus_1": float(np.mean(np.abs(vals - 1.0))),
                "n": int(inside.sum()),
            }
        )
    lev_table = pd.DataFrame(rows)
    mean_abs = float(lev_table["mean_abs_L_minus_1"].mean()) if len(lev_table) else float("nan")
    # SSR numerical vs SSR_hist
    psim = inputs.pricing_sim
    ssr_rows = ssr_numerical_many(lsv, list(inputs.ssr_pillars), eps=inputs.eps, sim=psim)
    ssr_hist = estimates.ssr_short
    ssr_table = pd.DataFrame(
        {
            "T": [r.T for r in ssr_rows],
            "ssr_model": [r.R for r in ssr_rows],
            "ssr_model_se": [r.R_stderr for r in ssr_rows],
            "ssr_hist": [_lookup(ssr_hist, r.T, "ssr") for r in ssr_rows],
            "ssr_hist_se": [_lookup(ssr_hist, r.T, "se") for r in ssr_rows],
            "skew_model": [r.skew for r in ssr_rows],
        }
    )
    vt = volvol_term_structure(lsv, list(inputs.volvol_pillars), eps=inputs.eps, sim=psim)
    vv_hist = estimates.volvol
    volvol_table = pd.DataFrame(
        {
            "T": vt.Ts,
            "volvol_model_with_leverage": vt.model_volvol,
            "volvol_model_se": vt.model_volvol_stderr,
            "volvol_kernel_eq_7_39": vt.kernel_volvol,
            "volvol_hist": [_lookup(vv_hist, float(T), "volvol") for T in vt.Ts],
            "volvol_hist_se": [_lookup(vv_hist, float(T), "se") for T in vt.Ts],
        }
    )
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
        volvol_table,
        forward_table,
        headline,
        cal_s,
        time.perf_counter() - t_all,
    )


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FitResult:
    params: BergomiParams
    pricing_date: pd.Timestamp
    config: Fit2FConfig
    stage1: Stage1Report
    stage2: Stage2Report
    stage3: Stage3Report | None
    degeneracy: pd.DataFrame
    wall_seconds: float

    @property
    def config_yaml(self) -> str:
        """A loadable model config: ``model`` (:class:`BergomiParams`, ``load_yaml(path,
        BergomiParams, section="model")``) with the provenance."""
        s1, s2 = self.stage1, self.stage2
        doc: dict[str, Any] = {
            "model": to_mapping(self.params),
            "omega": 2.0 * self.params.nu,
            "provenance": {
                "pricing_date": str(self.pricing_date.date()),
                "window_vol": self.config.window_vol,
                "window_ssr": self.config.window_ssr,
                "pillars": list(map(float, self.config.pillars)),
                "rho12_mode": self.config.rho12_mode,
                "curve": self.config.curve,
                "stage1": {
                    "objective": float(s1.best.objective),
                    "start": s1.best.start,
                    "unique_across_starts": bool(s1.unique),
                    "stderr": {k: float(v) for k, v in s1.best.stderr.items()},
                    "k2_flag": s1.best.k2_flag,
                    "optima": [
                        {"start": o.start, "objective": float(o.objective), **to_mapping(o.params)}
                        for o in s1.optima
                    ],
                },
                "stage2": {
                    "objective": float(s2.objective),
                    "chi": float(s2.chi),
                    "stderr": {k: float(v) for k, v in s2.stderr.items()},
                    "mixing_passes": int(s2.passes),
                    "distinct_optima": int(s2.n_distinct_optima),
                },
                "stage3_mean_abs_L_minus_1": (
                    None if self.stage3 is None else float(self.stage3.mean_abs_l_minus_1)
                ),
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
        p = self.params
        lines = [
            f"fit_2f @ {self.pricing_date.date()}: nu {p.nu:.3f} theta {p.theta:.3f} k1 {p.k1:.2f} "
            f"k2 {p.k2:.3f} rho12 {p.rho12:+.2f} rho_SX1 {p.rho_SX1:+.3f} rho_SX2 {p.rho_SX2:+.3f} "
            f"(chi {self.stage2.chi:+.3f}); stage-1 objective {self.stage1.best.objective:.2f} "
            f"({'unique' if self.stage1.unique else 'degenerate'} across starts, k2 "
            f"{self.stage1.best.k2_flag}); stage-2 objective {self.stage2.objective:.2f}",
            "stage 1:",
            self.stage1.table().round(4).to_string(index=False),
            self.stage1.fit_table().round(4).to_string(index=False),
            "stage 2:",
            self.stage2.table().round(4).to_string(index=False),
        ]
        if self.stage3 is not None:
            s3 = self.stage3
            lines += [
                f"stage 3: mean |L - 1| {s3.mean_abs_l_minus_1:.4f} "
                f"(calibration {s3.calibration_seconds:.0f} s)",
                s3.ssr_table.round(4).to_string(index=False),
                s3.volvol_table.round(4).to_string(index=False),
                s3.forward_table.round(4).to_string(index=False),
            ]
        return "\n".join(lines)


def fit_2f(
    history: SurfaceHistory,
    pricing_date: pd.Timestamp | str | None,
    config: Fit2FConfig | None = None,
    *,
    stage3: Stage3Inputs | None = None,
) -> FitResult:
    """Stages 1–2 (and 3 with ``stage3``) at ``pricing_date`` (default: the last date)."""
    cfg = config or Fit2FConfig()
    t0 = time.perf_counter()
    e = history.date_index(pricing_date)
    end = history.dates[e]
    s1 = fit_stage1(history, cfg, end)
    s2 = fit_stage2(history, cfg, s1.best.params, end)
    params = s2.params
    deg = skew_scale_degeneracy(history, cfg, params, end=end)
    s3 = None
    if stage3 is not None:
        est = estimate_history(
            history, end=end, window_vol=cfg.window_vol, window_ssr=cfg.window_ssr
        )
        s3 = stage3_validation(params, stage3, est)
    return FitResult(params, end, cfg, s1, s2, s3, deg, time.perf_counter() - t0)


__all__ = [
    "CURVES",
    "RHO12_MODES",
    "TABLE_7_1",
    "Fit2FConfig",
    "FitResult",
    "Stage1Optimum",
    "Stage1Report",
    "Stage1Targets",
    "Stage2Coefficients",
    "Stage2Report",
    "Stage2Targets",
    "Stage3Inputs",
    "Stage3Report",
    "chi_from_rhos",
    "fit_2f",
    "fit_stage1",
    "fit_stage2",
    "model_skew_ssr",
    "model_volvol",
    "rho_sx2_from_chi",
    "skew_scale_degeneracy",
    "stage1_targets",
    "stage2_coefficients",
    "stage2_targets",
    "stage3_validation",
]
