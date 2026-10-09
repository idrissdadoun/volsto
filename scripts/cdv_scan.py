"""Cross-dependent volatility prototype on the dispersion study's Dow basket: a scan of ``β``
(``volsto/multi/cdv.py``, ``docs/cross_dependent_vol.md``).

    python scripts/cdv_scan.py --date 2026-10-02 [--tenor 3m] [--betas 0,1,2,3,5]
        [--budget development] [--g-max 2.0] [--rows <M12 rows folder>] [--out <dir>]

For each ``β``: the particle calibration of ``λ`` and of the names' normalisations on the
date's specification (the one ``scripts/lcm_price.py`` builds: the same smiles, screen,
schedule, seeds and budget), then one pass of the pricing paths.  Reported with standard errors
on antithetic pair means: the clipped mass inside ±2.5 sd by side (how much of the index target
is out of reach), the index smile at the horizon against its target, the Palladium forward
``E[D]`` and its ratio to the constant-correlation companion's of the M12 row when that row
exists, ``E[V]`` against the listed ``E^Q[V]`` with its single-name and basket parts (the names
must keep their strips; the basket part is the wing), and the wall time.  ``β = 0`` is M12's
model on the calibration grid.  Outputs: a table in the log and
``<out>/cdv_scan_<date>_<tenor>.json`` (default ``outputs/dispersion_lc/cdv`` of this worktree;
never the study's own folders).

The owner's request (e) of 2026-10-09 (SPEC §8.7) adds, for each ``β``:

* the index smile errors at the horizon at −3.0 and −3.5 at-the-money standard deviations;
* ``E[R̄²]`` under the model against ``M_B^listed = Σ w M_i − E^Q[V]`` (the study's listed index
  second moment): value, ratio, standard error;
* the attribution of the forward's move from ``β = 0`` (:func:`attribution`): with
  ``E[D] = κ·√E[V]``, ``Δ ln E[D] = Δ ln κ + ½ Δ ln E[V]``, the forward at fixed ``κ``,
  ``κ(0)·√E[V](β)``, and the gap to the fixed-``κ`` estimate that takes the basket's second
  moment to ``M_B^listed``, split into ``κ``, the names' second moment and what the basket is
  still short of — as measured, and with the names' second moment held at its ``β = 0``
  estimate (the model keeps it; its Monte Carlo noise is what blurs ``κ``);
* where the basket's second moment is short of its own index target, by region of
  log-moneyness (:func:`basket_regions`: edges at −3.5, −2.5, −1.5, 0 and +1.5 sd).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lcm_diagnostics as lcd
import lcm_price as lp

from volsto.calibration.cache import code_version
from volsto.calibration.lc_cache import build_lc_market, lc_spec_key
from volsto.market.bs import black_vega, implied_vol
from volsto.multi.analytics import strip_second_moment
from volsto.multi.cdv import CrossDependence, calibrate_cdv, simulate_cdv

log = logging.getLogger("cdv_scan")
SD = (-3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
#: The edges of the regions of the basket's second moment, in at-the-money standard deviations.
REGION_EDGES_SD = (-3.5, -2.5, -1.5, 0.0, 1.5)
#: Intervals each side of the forward of the target's strip by region.  The strip gives a cell
#: to the region its midpoint is in: at the default 2000 a far region is off by up to 1 % (one
#: cell at its edge), at 10⁵ by 2·10⁻⁴ (measured on a lognormal law against the closed form,
#: ``tests/test_cross_dependent_vol.py::test_region_second_moments``).
STRIP_N_HALF = 100_000


@dataclass
class Paths:
    """Per pricing path at the horizon: the dispersion ``D = Σ w|R_i − R̄|``, ``Σ w R_i²``,
    ``R̄`` (``R_i = S_i(T)/S_i(0) − 1``) and the basket's forward-moneyness level
    ``L = e^{k_B}``."""

    D: np.ndarray
    sq: np.ndarray
    rb: np.ndarray
    level: np.ndarray

    @property
    def V(self) -> np.ndarray:
        return np.asarray(self.sq - self.rb**2)


def paths_of(log_spot: np.ndarray, k_basket: np.ndarray, market: Any, weights: np.ndarray) -> Paths:
    """The per-path quantities of one record time of ``simulate_cdv``: ``log_spot`` is
    ``(n_paths, n)``, ``k_basket`` ``(n_paths,)``."""
    spot0 = np.array([float(m.forward_curve.spot) for m in market.models])
    r = np.exp(log_spot) / spot0[None, :] - 1.0
    rb = r @ weights
    return Paths(np.abs(r - rb[:, None]) @ weights, (r * r) @ weights, rb, np.exp(k_basket))


def listed_moments(inp: lcd.DowInputs) -> tuple[float, float, float]:
    """``(E^Q[V], Σ w M_i, M_B^listed)`` of the study's entry of the date: the listed dispersion
    variance, the names' listed second moments and their difference, the listed index second
    moment (``scripts/lcm_price.py``, the same three numbers)."""
    eqv = float(inp.entry["B1"]["EQV"])
    sum_wm = float(np.sum(inp.weights / inp.weights.sum() * np.asarray(inp.entry["legs"]["M"])))
    return eqv, sum_wm, sum_wm - eqv


def smile_errors(
    level: np.ndarray, surface: Any, T: float, sds: Sequence[float] = SD
) -> dict[str, tuple[float, float]]:
    """The basket's implied vol minus the target at the horizon, vol points with standard
    errors, at strikes in at-the-money standard deviations and at the 90 % strike."""
    atm = float(surface.atm_vol(T))
    out = {}
    for label, k in [(f"{m:+.1f}", m * atm * math.sqrt(T)) for m in sds] + [("90%", math.log(0.9))]:
        cp = 1.0 if k >= 0 else -1.0
        price, se = lcd.pair_mean(np.maximum(cp * (level - math.exp(k)), 0.0))
        target = float(surface.implied_vol_k(k, T))
        vol = float(implied_vol(price, 1.0, math.exp(k), T, cp))
        vega = float(black_vega(1.0, math.exp(k), T, target))
        out[label] = (100 * (vol - target), 100 * se / vega)
    return out


def region_second_moments(
    level: np.ndarray, strikes: Sequence[float], forward: float = 1.0
) -> np.ndarray:
    """Per path, the parts of ``(L − F)²`` by region of the strike axis: ``(n_paths,
    len(strikes) + 1)`` — below the first strike (from ``K = 0``), between consecutive strikes,
    above the last.

    Derivation.  For every level ``L > 0`` the square is a strip of out-of-the-money payoffs
    (Carr–Madan, path by path): ``(L − F)² = 2∫_0^F (K − L)⁺ dK + 2∫_F^∞ (L − K)⁺ dK``.  The
    integrand has an explicit primitive on each side, ``d/dK ((K − L)⁺)² = 2(K − L)⁺`` and
    ``d/dK ((L − K)⁺)² = −2(L − K)⁺``, so the part of a region ``[K₁, K₂]`` is exact:

        below the forward (K₂ ≤ F):  2∫_{K₁}^{K₂} (K − L)⁺ dK = ((K₂ − L)⁺)² − ((K₁ − L)⁺)²,
        above the forward (K₁ ≥ F):  2∫_{K₁}^{K₂} (L − K)⁺ dK = ((L − K₁)⁺)² − ((L − K₂)⁺)².

    A region that holds the forward is cut at it (puts below, calls above); the outer regions
    reach ``K = 0`` (``(0 − L)⁺ = 0``) and ``K = ∞`` (``(L − ∞)⁺ = 0``).  The parts telescope to
    ``(L − F)²`` exactly.  Their expectations are the parts by region of the model's own strip
    ``2∫ P(K) dK + 2∫ C(K) dK``, comparable term by term with
    ``analytics.strip_second_moment(surface, T, splits=ln(strikes/F))`` of a target surface.

    Checked against a direct numerical integral of the payoffs and against the strip of a
    lognormal law in ``tests/test_cross_dependent_vol.py::test_region_second_moments``."""
    L = np.asarray(level, dtype=np.float64).ravel()
    cuts = np.asarray(strikes, dtype=np.float64).ravel()
    if cuts.size == 0 or np.any(cuts <= 0.0) or np.any(np.diff(cuts) <= 0.0):
        raise ValueError("strikes must be positive and increasing")
    if not forward > 0.0:
        raise ValueError("the forward must be positive")
    f = float(forward)

    def put_sq(K: float) -> np.ndarray:
        return np.asarray(np.maximum(K - L, 0.0) ** 2)

    def call_sq(K: float) -> np.ndarray:
        return np.zeros(L.size) if math.isinf(K) else np.asarray(np.maximum(L - K, 0.0) ** 2)

    bounds = [0.0, *[float(c) for c in cuts], math.inf]
    out = np.empty((L.size, cuts.size + 1))
    for j, (k1, k2) in enumerate(pairwise(bounds)):
        below = put_sq(min(k2, f)) - put_sq(min(k1, f))  # the region's part of [0, F]: puts
        above = call_sq(max(k1, f)) - call_sq(max(k2, f))  # its part of [F, ∞): calls
        out[:, j] = below + above
    return out


def region_labels(edges_sd: Sequence[float] = REGION_EDGES_SD) -> list[str]:
    """``"below -3.5 sd"``, ``"-3.5 to -2.5 sd"``, …, ``"above +1.5 sd"``."""
    e = [f"{m:+.1f}" for m in edges_sd]
    return [f"below {e[0]} sd", *[f"{a} to {b} sd" for a, b in pairwise(e)], f"above {e[-1]} sd"]


def basket_regions(
    level: np.ndarray, surface: Any, T: float, edges_sd: Sequence[float] = REGION_EDGES_SD
) -> dict[str, Any]:
    """Where the model's basket second moment is short of its index target: ``E[(L − 1)²]`` of
    the basket level ``L = e^{k_B}`` (forward moneyness) by region of log-moneyness, edges in
    at-the-money standard deviations ``σ_ATM√T`` of the target.

    The model's part of a region is the pair mean of :func:`region_second_moments` (exact per
    path); the target's is ``strip_second_moment(surface, T, splits=…)`` — the same strip
    ``2∫ OTM(K) dK`` of the index surface, by the same regions (its lognormal tails in the
    outer ones), on ``STRIP_N_HALF`` intervals each side.  Both are in the surface's units: the
    strikes are ``f·e^k`` with ``f = F(T)/S_0`` of the surface's curve (``f = 1`` in
    performance mode, the Dow runs').

    Returns the labels, the edges, per region ``model``, ``model_se``, ``target`` and
    ``difference = model − target``, and the totals (``total_target`` is the sum of the
    regions: trapezoids, within the quadrature error of the Simpson value)."""
    f = float(surface.forward_curve.forward(T)) / float(surface.forward_curve.spot)
    sd = float(surface.atm_vol(T)) * math.sqrt(T)
    ks = [float(m) * sd for m in edges_sd]
    _, target = strip_second_moment(surface, T, splits=ks, n_half=STRIP_N_HALF)  # type: ignore[misc]
    parts = region_second_moments(f * level, [f * math.exp(k) for k in ks], f)
    stats = [lcd.pair_mean(parts[:, j]) for j in range(parts.shape[1])]
    total, total_se = lcd.pair_mean((f * level - f) ** 2)
    return {
        "labels": region_labels(edges_sd), "edges_sd": [float(m) for m in edges_sd], "sd": sd,
        "model": [s[0] for s in stats], "model_se": [s[1] for s in stats],
        "target": [float(x) for x in target],
        "difference": [s[0] - float(x) for s, x in zip(stats, target, strict=True)],
        "total_model": total, "total_model_se": total_se, "total_target": float(np.sum(target)),
    }  # fmt: skip


def _half_log(numerator: float, denominator: float) -> float:
    """``½ ln(numerator/denominator)``, ``nan`` when either is not positive."""
    if not (numerator > 0.0 and denominator > 0.0):
        return float("nan")
    return 0.5 * math.log(numerator / denominator)


def attribution(base: Paths, other: Paths, m_b: float) -> dict[str, float]:
    """The move of the Palladium forward between two models on the same pricing paths (``base``:
    ``β = 0``; ``other``: a ``β > 0``), attributed.  With ``E[D] = κ·√E[V]``:

        Δ ln E[D] = Δ ln κ + ½ Δ ln E[V]                                   (an identity),

    reported as ``dln_ED``, ``dln_kappa``, ``half_dln_EV``, with the forward at fixed ``κ``,
    ``ED_fixed_kappa = κ(0)·√E[V](β)``.

    The fixed-``κ`` estimate of M12's rows (``ED_wing``) takes the basket's second moment to
    the listed one at unchanged names: ``wing_estimate = ½ ln((Σ w E⁰[R_i²] − M_B^listed)/E⁰[V])``.
    The gap of the realised move to it is, exactly,

        Δ ln E[D] − wing_estimate = Δ ln κ + names_term + short_term,
        names_term = ½ ln((Σ w E^β[R_i²] − M_B^listed)/(Σ w E⁰[R_i²] − M_B^listed)),
        short_term = ½ ln(E^β[V]/(Σ w E^β[R_i²] − M_B^listed)),

    ``short_term`` being positive when ``E^β[R̄²]`` is still below ``M_B^listed``: the three
    terms say whether the forward moved less than the estimate because ``κ`` rose, because the
    names' second moment moved, or because the basket is still short of the listed moment.
    ``shortfall_closed`` is ``(E^β[R̄²] − E⁰[R̄²])/(M_B^listed − E⁰[R̄²])``.

    **Names held.**  ``Σ w E[R_i²]`` is the noisy term (the names' upper tails) and is the same
    under every ``β`` in the model — each name keeps its marginal law — so the split between
    ``κ`` and ``E[V]`` is sharper with the names' second moment held at its ``β = 0`` estimate
    ``N⁰``: ``E_h[V](β) = N⁰ − E^β[R̄²]``, ``κ_h(β) = E^β[D]/√E_h[V](β)`` (``kappa_names_held``;
    ``κ_h(0) = κ(0)``), and, exactly,

        Δ ln E[D] = dln_kappa_names_held + half_dln_EV_names_held,
        half_dln_EV_names_held = ½ ln((N⁰ − E^β[R̄²])/(N⁰ − E⁰[R̄²]))
                               = wing_estimate + short_term_names_held,
        short_term_names_held  = ½ ln((N⁰ − E^β[R̄²])/(N⁰ − M_B^listed)),

    so the gap to the wing estimate is ``dln_kappa_names_held + short_term_names_held``: ``κ``
    up, or the basket still short.  ``names_drift = half_dln_EV − half_dln_EV_names_held`` is
    what the measured move of the names' second moment adds (zero in the model; its standard
    error says how well the paths resolve it), and ``ED_fixed_kappa_names_held =
    κ(0)·√E_h[V](β)``.

    Every number carries a delta-method standard error on antithetic pair means of the two
    models' paths taken together (``lcm_price.delta_method``): the difference is paired.
    Checked in ``tests/test_cross_dependent_vol.py::test_attribution_identities``."""
    samples = [other.D, other.sq, other.rb**2, base.D, base.sq, base.rb**2]

    def ln(x: float) -> float:
        return math.log(x) if x > 0.0 else float("nan")

    terms = {
        "dln_ED": lambda m: ln(m[0] / m[3]),
        "dln_kappa": lambda m: ln(m[0] / m[3]) - _half_log(m[1] - m[2], m[4] - m[5]),
        "half_dln_EV": lambda m: _half_log(m[1] - m[2], m[4] - m[5]),
        "ED_fixed_kappa": lambda m: m[3] * math.sqrt(max(m[1] - m[2], 0.0) / (m[4] - m[5])),
        "wing_estimate": lambda m: _half_log(m[4] - m_b, m[4] - m[5]),
        "names_term": lambda m: _half_log(m[1] - m_b, m[4] - m_b),
        "short_term": lambda m: _half_log(m[1] - m[2], m[1] - m_b),
        "gap_to_wing_estimate": lambda m: ln(m[0] / m[3]) - _half_log(m[4] - m_b, m[4] - m[5]),
        "d_E_Rbar2": lambda m: m[2] - m[5],
        "d_sum_w_ER2": lambda m: m[1] - m[4],
        "shortfall_closed": lambda m: (m[2] - m[5]) / (m_b - m[5]) if m_b != m[5] else float("nan"),
        # the names' second moment held at its beta = 0 estimate
        "half_dln_EV_names_held": lambda m: _half_log(m[4] - m[2], m[4] - m[5]),
        "dln_kappa_names_held": lambda m: ln(m[0] / m[3]) - _half_log(m[4] - m[2], m[4] - m[5]),
        "short_term_names_held": lambda m: _half_log(m[4] - m[2], m[4] - m_b),
        "names_drift": lambda m: _half_log(m[1] - m[2], m[4] - m[2]),
        "kappa_names_held": lambda m: m[0] / math.sqrt(m[4] - m[2]) if m[4] > m[2] else float("nan"),
        "kappa_base": lambda m: m[3] / math.sqrt(m[4] - m[5]) if m[4] > m[5] else float("nan"),
        "ED_fixed_kappa_names_held": lambda m: m[3] * math.sqrt(max(m[4] - m[2], 0.0) / (m[4] - m[5])),
    }  # fmt: skip
    out: dict[str, float] = {}
    for name, fn in terms.items():
        out[name], out[f"{name}_se"] = lp.delta_method(fn, samples)
    return out


def scan(
    date: str, tenor: str, betas: Sequence[float], budget: str, g_max: float, rows_dir: str | None
) -> dict[str, Any]:
    cfg = lp.load_config()
    inp = lcd.load_inputs(date, tenor, cfg["index"])
    spec, _ = lp.spec_for(inp, cfg, budget, None)
    market = build_lc_market(spec)
    T = spec.lc.particle.horizon
    w = np.array(spec.weights)
    eqv, sum_wm, m_b = listed_moments(inp)
    row_file = None if rows_dir is None else Path(rows_dir) / f"{tenor}_{budget}" / f"{date}.json"
    if rows_dir is not None and (Path(rows_dir) / f"{date}.json").exists():
        row_file = Path(rows_dir) / f"{date}.json"  # the folder of one pass, given itself
    ed_cc = (
        json.loads(row_file.read_text()).get("ED_cc")
        if row_file is not None and row_file.exists()
        else None
    )
    log.info(
        "=== cross-dependent scan %s %s (%s budget: N = %d, paths = %d; key of the M12 specification %s); "
        "E_CC[D] of the M12 row: %s; screen %s",
        date, tenor, budget, spec.lc.particle.n_particles, spec.sim.n_paths, lc_spec_key(spec)[:12], ed_cc, cfg["screen"],
    )  # fmt: skip
    rows = []
    kept: dict[float, Paths] = {}
    for beta in betas:
        t0 = time.perf_counter()
        dep = CrossDependence(float(beta), g_max=g_max, g_min=1.0 / g_max)
        res = calibrate_cdv(market.models, market.family, market.basket, market.index_surface,
                            market.index_lv, spec.lc.particle, spec.sim, spec.lc, dep)  # fmt: skip
        t1 = time.perf_counter()
        ls, kb = simulate_cdv(res, market.models, market.family, market.basket, spec.sim, [T])
        t2 = time.perf_counter()
        fwd = np.array(
            [float(m.forward_curve.forward(T)) / float(m.forward_curve.spot) for m in market.models]
        )
        spot0 = np.array([float(m.forward_curve.spot) for m in market.models])
        p = paths_of(ls[0], kb[0], market, w)
        D, sq, rb, level = p.D, p.sq, p.rb, p.level
        errs = smile_errors(level, market.index_surface, T)
        ed, ed_se = lcd.pair_mean(D)
        ev, ev_se = lcd.pair_mean(sq - rb**2)
        names, names_se = lcd.pair_mean(sq)
        rbar2, rbar2_se = lcd.pair_mean(rb**2)
        kappa, kappa_se = lcd.pair_ratio_sqrt(D, sq - rb**2)
        f_err, f_se = lcd.pair_mean(level - 1.0)
        regions = basket_regions(level, market.index_surface, T)
        f_b = float(w @ fwd)
        row = {
            "beta": float(beta), "g_max": g_max,
            "clip_high_inner": float(res.inner_high.max()), "clip_low_inner": float(res.inner_low.max()),
            "clip_high": float(res.clipped_high.max()), "clip_low": float(res.clipped_low.max()),
            "lambda_mean_T": float(res.lambda_mean[-1]), "scale_min": float(res.scale.min()), "scale_max": float(res.scale.max()),
            "ED": ed, "ED_se": ed_se, "ED_over_cc": None if ed_cc is None else ed / ed_cc,
            "EV_over_EQV": ev / eqv, "EV_over_EQV_se": ev_se / eqv,
            "names_part": names - sum_wm, "names_part_se": names_se, "names_over_listed": names / sum_wm - 1.0,
            "basket_part": rbar2 - m_b, "basket_part_se": rbar2_se, "kappa": kappa, "kappa_se": kappa_se,
            "forward_error": f_err, "forward_error_se": f_se,
            **{f"idx_{k}": v[0] for k, v in errs.items()}, **{f"idx_{k}_se": v[1] for k, v in errs.items()},
            "seconds_calibration": t1 - t0, "seconds_pricing": t2 - t1, "timings": res.timings,
            "mean_forward_check": float(np.max(np.abs(np.exp(ls[0]).mean(axis=0) / (spot0 * fwd) - 1.0))),
            # request (e) of 2026-10-09
            "EV": ev, "EV_se": ev_se, "sum_w_ER2": names, "sum_w_ER2_se": names_se,
            "E_Rbar2": rbar2, "E_Rbar2_se": rbar2_se,
            "E_Rbar2_over_listed": rbar2 / m_b, "E_Rbar2_over_listed_se": rbar2_se / m_b,
            "regions": regions,
            # the bridge from the level L to Rbar: Rbar = F_B·L − 1 when the names share one carry, F_B = Σ w f_i
            "basket_forward": f_b, "E_Rbar2_of_target": (f_b - 1.0) ** 2 + f_b**2 * regions["total_target"],
            "E_Rbar2_of_model_level": (f_b - 1.0) ** 2 + f_b**2 * regions["total_model"],
        }  # fmt: skip
        rows.append(row)
        log.info(
            "beta %.2f: clipped mass inside ±2.5 sd high %.4f, low %.4f; index at T: ATM %+.3f (%.3f), 90%% %+.3f (%.3f), -1.5 sd %+.3f, -2.5 sd %+.3f (%.3f), +1.5 sd %+.3f; "
            "E[D] %.6f (%.6f)%s; E[V]/E_Q[V] %.4f (%.4f): names %+.6f (%.6f), basket %+.6f (%.6f); kappa %.4f; scales [%.3f, %.3f]; %.0f + %.0f s",
            beta, row["clip_high_inner"], row["clip_low_inner"], errs["+0.0"][0], errs["+0.0"][1], errs["90%"][0], errs["90%"][1],
            errs["-1.5"][0], errs["-2.5"][0], errs["-2.5"][1], errs["+1.5"][0], ed, ed_se,
            "" if ed_cc is None else f" = {ed / ed_cc:.5f} x E_CC[D]", row["EV_over_EQV"], row["EV_over_EQV_se"],
            row["names_part"], row["names_part_se"], row["basket_part"], row["basket_part_se"], kappa,
            row["scale_min"], row["scale_max"], t1 - t0, t2 - t1,
        )  # fmt: skip
        log.info(
            "beta %.2f: index at T: -2.5 sd %+.3f (%.3f), -3.0 sd %+.3f (%.3f), -3.5 sd %+.3f (%.3f) vp; "
            "E[Rbar^2] %.6f (%.6f) against M_B^listed %.6f: ratio %.4f (%.4f); kappa %.4f (%.4f)",
            beta, errs["-2.5"][0], errs["-2.5"][1], errs["-3.0"][0], errs["-3.0"][1], errs["-3.5"][0], errs["-3.5"][1],
            rbar2, rbar2_se, m_b, row["E_Rbar2_over_listed"], row["E_Rbar2_over_listed_se"], kappa, kappa_se,
        )  # fmt: skip
        kept[float(beta)] = p
    # the forward's move from beta = 0, on the same pricing paths
    attributions: list[dict[str, Any]] = (
        [{"beta": b, **attribution(kept[0.0], q, m_b)} for b, q in kept.items() if b != 0.0]
        if 0.0 in kept
        else []
    )
    table = pd.DataFrame(rows)
    cols = [
        "beta",
        "clip_high_inner",
        "clip_low_inner",
        "idx_+0.0",
        "idx_90%",
        "idx_-1.5",
        "idx_-2.5",
        "idx_+1.5",
        "ED",
        "ED_se",
        "ED_over_cc",
        "EV_over_EQV",
        "names_over_listed",
        "basket_part",
        "kappa",
    ]
    log.info("\n%s", table[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    log_request_e(rows, attributions, m_b)
    return {
        "date": date, "tenor": tenor, "budget": budget, "rows": rows, "E_CC_D_of_the_M12_row": ed_cc,
        "EQV": eqv, "sum_w_M": sum_wm, "M_B_listed": m_b,
        "attribution_from_beta_0": attributions, "region_labels": region_labels(), "region_edges_sd": list(REGION_EDGES_SD),
        "screen": dict(cfg["screen"]),
        "record": {"git_commit": code_version(), "spec_key": lc_spec_key(spec), "particle_seed": spec.lc.particle.seed,
                   "pricing_seed": spec.sim.seed, "n_particles": spec.lc.particle.n_particles, "n_paths": spec.sim.n_paths,
                   "schedule": repr(spec.sim.step_schedule)},
    }  # fmt: skip


def log_request_e(rows: list[dict[str, Any]], attributions: list[dict[str, Any]], m_b: float) -> None:  # fmt: skip
    """The tables of the owner's request (e): the wing of the index smile, the second moments,
    the attribution of the forward's move from ``β = 0`` and the basket's regions."""
    wing = pd.DataFrame(
        [{"beta": r["beta"], **{f"{k} sd": r[f"idx_{k}"] for k in ("-2.5", "-3.0", "-3.5")},
          **{f"se {k}": r[f"idx_{k}_se"] for k in ("-2.5", "-3.0", "-3.5")}} for r in rows]
    )  # fmt: skip
    log.info("index smile at the horizon, model minus target (vp):\n%s", wing.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))  # fmt: skip
    moments = pd.DataFrame(
        [{"beta": r["beta"], "E[V]/E_Q[V]": r["EV_over_EQV"], "se": r["EV_over_EQV_se"], "names part": r["names_part"],
          "se names": r["names_part_se"], "basket part": r["basket_part"], "se basket": r["basket_part_se"],
          "E[Rbar^2]": r["E_Rbar2"], "/M_B^listed": r["E_Rbar2_over_listed"], "se ratio": r["E_Rbar2_over_listed_se"],
          "kappa": r["kappa"], "se kappa": r["kappa_se"]} for r in rows]
    )  # fmt: skip
    log.info("second moments (M_B^listed = %.6f):\n%s", m_b, moments.to_string(index=False, float_format=lambda x: f"{x:.6f}"))  # fmt: skip
    log.info(
        "bridge to the regions below: with F_B = sum w f_i = %.6f, the index target's strip gives E[Rbar^2] = (F_B-1)^2 + F_B^2 E[(L-1)^2] "
        "= %.6f (M_B^listed %.6f); the model's level gives %s against its measured E[Rbar^2] %s (the difference: the names' carries differ)",
        rows[0]["basket_forward"], rows[0]["E_Rbar2_of_target"], m_b,
        ", ".join(f"{r['E_Rbar2_of_model_level']:.6f}" for r in rows), ", ".join(f"{r['E_Rbar2']:.6f}" for r in rows),
    )  # fmt: skip
    if attributions:
        keys = (
            "dln_ED",
            "dln_kappa",
            "half_dln_EV",
            "wing_estimate",
            "gap_to_wing_estimate",
            "names_term",
            "short_term",
        )
        att = pd.DataFrame(
            [{"beta": a["beta"], **{k: 100 * a[k] for k in keys}, "ED at fixed kappa": a["ED_fixed_kappa"],
              "shortfall closed": a["shortfall_closed"]} for a in attributions]
        )  # fmt: skip
        err = pd.DataFrame(
            [{"beta": a["beta"], **{k: 100 * a[f"{k}_se"] for k in keys}, "ED at fixed kappa": a["ED_fixed_kappa_se"],
              "shortfall closed": a["shortfall_closed_se"]} for a in attributions]
        )  # fmt: skip
        log.info(
            "the forward's move from beta = 0 on the same paths, in %% (d ln E[D] = d ln kappa + 0.5 d ln E[V]; the gap to the "
            "fixed-kappa wing estimate = d ln kappa + names term + basket still short):\n%s\nstandard errors:\n%s",
            att.to_string(index=False, float_format=lambda x: f"{x:+.4f}"),
            err.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        )  # fmt: skip
        held_keys = ("dln_ED", "wing_estimate", "short_term_names_held", "dln_kappa_names_held", "half_dln_EV_names_held", "names_drift")  # fmt: skip
        held = pd.DataFrame(
            [{"beta": a["beta"], **{k: 100 * a[k] for k in held_keys}, "kappa(0)": a["kappa_base"], "kappa names held": a["kappa_names_held"],
              "ED at fixed kappa": a["ED_fixed_kappa_names_held"]} for a in attributions]
        )  # fmt: skip
        held_err = pd.DataFrame(
            [{"beta": a["beta"], **{k: 100 * a[f"{k}_se"] for k in held_keys}, "kappa(0)": a["kappa_base_se"],
              "kappa names held": a["kappa_names_held_se"], "ED at fixed kappa": a["ED_fixed_kappa_names_held_se"]} for a in attributions]
        )  # fmt: skip
        log.info(
            "the same with the names' second moment held at its beta = 0 estimate, in %% (d ln E[D] = wing estimate + basket still "
            "short + d ln kappa; the names' drift is what the measured move of sum w E[R_i^2] adds to 0.5 d ln E[V]):\n%s\nstandard errors:\n%s",
            held.to_string(index=False, float_format=lambda x: f"{x:+.4f}"),
            held_err.to_string(index=False, float_format=lambda x: f"{x:.4f}"),
        )  # fmt: skip
    else:
        log.info(
            "no attribution of the forward's move: it needs beta = 0 and another beta in the scan"
        )
    for r in rows:
        log.info(
            "beta %.2f: the basket's second moment E[(L-1)^2] against its index target, by region of log-moneyness "
            "(at-the-money sd %.4f):\n%s",
            r["beta"], r["regions"]["sd"], regions_frame(r["regions"]).to_string(index=False, float_format=lambda x: f"{x:.7f}"),
        )  # fmt: skip


def regions_frame(regions: dict[str, Any]) -> pd.DataFrame:
    """:func:`basket_regions` as a table: one line per region and the total."""
    frame = pd.DataFrame(
        {"region": regions["labels"], "model": regions["model"], "se": regions["model_se"],
         "target": regions["target"], "model - target": regions["difference"]}
    )  # fmt: skip
    total = {"region": "total", "model": regions["total_model"], "se": regions["total_model_se"],
             "target": regions["total_target"], "model - target": regions["total_model"] - regions["total_target"]}  # fmt: skip
    frame = pd.concat([frame, pd.DataFrame([total])], ignore_index=True)
    frame["in se"] = frame["model - target"] / frame["se"]
    return frame


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--date", default="2026-10-02")
    ap.add_argument("--tenor", default="3m")
    ap.add_argument("--betas", default="0,1,2,3,5")
    ap.add_argument("--budget", default="development", choices=("production", "development"))
    ap.add_argument("--g-max", type=float, default=2.0)
    ap.add_argument(
        "--rows",
        default=None,
        help="the rows folder of an M12 sweep, or of one of its passes (for the companion's E[D])",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("volsto").setLevel(logging.WARNING)
    betas = [float(b) for b in args.betas.split(",")]
    out = Path(args.out) if args.out else lp.ROOT / "outputs" / "dispersion_lc" / "cdv"
    if "outputs/dispersion_lc" not in str(out.resolve()):
        raise ValueError("outputs must go under outputs/dispersion_lc")
    doc = scan(args.date, args.tenor, betas, args.budget, args.g_max, args.rows)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"cdv_scan_{args.date}_{args.tenor}.json"
    path.write_text(json.dumps(doc, indent=1, default=lp.jsonable))
    log.info("written %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
