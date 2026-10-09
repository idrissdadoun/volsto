"""Local correlation model (SPEC §8.7, M12 part LC7): one date of the dispersion study's Dow
basket — calibration through the cache, prices with the constant-correlation companion, risk.

    python scripts/lcm_price.py --date 2026-10-02 [--tenor 3m] [--budget production|development]
        [--risk none|deltas|full] [--varswap] [--config configs/studies/dispersion/lcm.yaml]
        [--root <dir>] [--row-out <json>] [--profile]

What it does (the row of ``scripts/disp_lcm.py`` is its output):

1. builds the specification from the study's smiles under the expiry screen and calibrates
   ``λ(t, k)`` through the cache (``<root>/outputs/dispersion_lc/cache``); the
   constant-correlation companion is the cache's ``"constant"`` family (the constant ``λ_c``
   that reprices the index at-the-money straddle at the horizon);
2. prices, on one pass of the pricing paths per model (the same seed: every LC − CC difference
   is paired): the Palladium forward; the calls at the study's cash strikes ``K_050 … K_200``;
   the basket straddle and the 90 % put; the single-name straddles and the gap; ``E[V]``, ``κ``
   and the conditional profile ``E[D/B | bucket]/E[D/B]`` on the study's buckets; the index
   smile against its target; the wing-corrected forwards
   ``ED_wing = κ_LC·√(Σ w E^LC[R_i²] − M_B^listed)`` and ``ED_eqv = κ_LC·√EQV``;
3. runs the risk asked for: ``deltas`` — the sticky-strike common deltas of the forward and of
   the call at ``K_100`` under LC and CC with the decomposition of the forward's (homogeneity
   exactly 1, skew channel ``Δ^CC_ss − 1``, correlation channel ``Δ^LC_ss − Δ^CC_ss``);
   ``full`` — also the single-name vegas (``λ`` recalibrated and held), the all-names bump, the
   index vega, the two index skew vegas and the model-risk range over four ``R_low`` × two
   families;
4. logs a table and writes the row as JSON with the reproducibility record.

Standard errors are on antithetic pair means; ratios and ``κ`` use the delta method.  A date
that fails returns a row with ``status = "failed"`` and the reason.  Outputs go under
``outputs/dispersion_lc/`` — never into the study's own ``outputs/dispersion/``.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import sys
import time
import traceback
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lcm_diagnostics as lcd  # noqa: E402

from volsto._numba import num_threads  # noqa: E402
from volsto.calibration.cache import code_version  # noqa: E402
from volsto.calibration.fit_records import FitRecords  # noqa: E402
from volsto.calibration.lc_cache import (  # noqa: E402
    LocalCorrelationCache,
    build_lc_market,
    lc_spec_key,
)
from volsto.calibration.local_correlation import CLIP_GATE_SD, LC_CODE_TAG  # noqa: E402
from volsto.config import (  # noqa: E402
    LocalCorrelationSpec,
    ParametricLambdaConfig,
    StepSchedule,
)
from volsto.engine.grid import TimeGrid  # noqa: E402
from volsto.market.bs import black_vega, implied_vol  # noqa: E402
from volsto.market.curves import DiscountCurve  # noqa: E402
from volsto.market.varswap import varswap_strike  # noqa: E402
from volsto.multi.analytics import strip_second_moment  # noqa: E402
from volsto.multi.family import historical_scaled_correlation  # noqa: E402
from volsto.multi.lc_model import LocalCorrelationModel  # noqa: E402
from volsto.multi.mc import MultiAssetMonteCarlo  # noqa: E402
from volsto.multi.products import BasketVarianceSwap, Palladium  # noqa: E402
from volsto.risk import local_correlation as lcr  # noqa: E402
from volsto.studies import disp_copula as dcop  # noqa: E402
from volsto.studies import disp_data as dd  # noqa: E402
from volsto.studies.disp_lc import ExpiryScreen  # noqa: E402

log = logging.getLogger("lcm_price")

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "studies" / "dispersion" / "lcm.yaml"
ZERO = DiscountCurve.flat(0.0)
CONFIG_KEYS = {
    "basket", "index", "carry", "discount", "lc", "particle", "sim", "screen", "budgets",
    "products", "risk", "delta_bump", "outputs", "cache",
}  # fmt: skip
SD_GRID = (-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
MULT_TAGS = ("050", "075", "100", "125", "150", "200")


# ---------------------------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------------------------


def load_config(path: str | Path = CONFIG) -> dict[str, Any]:
    """The YAML of the runs, strictly: an unknown or a missing top-level key is an error."""
    cfg = yaml.safe_load(Path(path).read_text())
    if not isinstance(cfg, dict) or set(cfg) != CONFIG_KEYS:
        raise ValueError(f"{path}: keys {sorted(set(cfg) ^ CONFIG_KEYS)} are unknown or missing")
    if cfg["basket"] != "B1" or cfg["discount"] != "zero" or cfg["carry"] not in ("study", "zero"):
        raise ValueError(f"{path}: basket B1, discount zero and carry study|zero are supported")
    if list(cfg["products"]["call_multiples"]) != [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]:
        raise ValueError(f"{path}: the call multiples are the study's six")
    return cfg


def config_digest(cfg: dict[str, Any], tenor: str, budget: str) -> str:
    """The digest a sweep's resume compares: the YAML, the tenor and the budget."""
    doc = {"config": cfg, "tenor": tenor, "budget": budget}
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


def out_root(cfg: dict[str, Any], root: str | Path | None) -> Path:
    base = Path(root) if root else ROOT
    out = (base / cfg["outputs"]).resolve()
    if "outputs/dispersion_lc" not in str(out):
        raise ValueError(f"outputs must go under outputs/dispersion_lc (got {out})")
    return out


def spec_for(
    inp: lcd.DowInputs,
    cfg: dict[str, Any],
    budget: str,
    records: FitRecords | None,
    **override: Any,
) -> tuple[LocalCorrelationSpec, dict[str, Any]]:
    """The specification of a date under the configuration and a budget."""
    b = cfg["budgets"][budget]
    sched = cfg["sim"]["schedule"]
    options: dict[str, Any] = {
        **{k: v for k, v in cfg["lc"].items() if k != "family"},
        **{k: v for k, v in cfg["particle"].items() if k != "seed"},
        "parametric": ParametricLambdaConfig(n_paths=int(b["companion_paths"])),
        "family": cfg["lc"]["family"],
    }
    options.update(override)
    # quantile_clip is a ParticleConfig field the diagnostics' builder does not route: route it
    spec, info = lcd.build_spec(
        inp,
        n_particles=int(b["n_particles"]),
        n_paths=int(b["n_paths"]),
        screen=ExpiryScreen(**cfg["screen"]),
        schedule=StepSchedule(tuple(sched["breaks"]), tuple(sched["dts"])),
        particle_seed=int(cfg["particle"]["seed"]),
        pricing_seed=int(cfg["sim"]["seed"]),
        records=records,
        **{k: v for k, v in options.items() if k != "quantile_clip"},
    )
    if spec.lc.particle.quantile_clip != cfg["particle"]["quantile_clip"]:
        particle = dataclasses.replace(
            spec.lc.particle, quantile_clip=cfg["particle"]["quantile_clip"]
        )
        spec = dataclasses.replace(spec, lc=dataclasses.replace(spec.lc, particle=particle))
    return spec, info


# ---------------------------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------------------------


def pairs(x: np.ndarray) -> np.ndarray:
    return np.asarray(0.5 * (x[0::2] + x[1::2]), dtype=np.float64)


def delta_method(
    fn: Callable[[np.ndarray], float], samples: Sequence[np.ndarray]
) -> tuple[float, float]:
    """``(fn(means), standard error)`` of a smooth function of the means of per-path samples on
    the same paths: the gradient by central differences, the error from the pair means."""
    y = np.column_stack([pairs(np.asarray(s, dtype=np.float64)) for s in samples])
    m = y.mean(axis=0)
    value = float(fn(m))
    grad = np.empty(m.size)
    for i in range(m.size):
        h = 1e-6 * max(abs(m[i]), 1e-12)
        up, dn = m.copy(), m.copy()
        up[i] += h
        dn[i] -= h
        grad[i] = (fn(up) - fn(dn)) / (2 * h)
    u = (y - m) @ grad
    return value, float(u.std(ddof=1) / np.sqrt(u.size))


class Pass:
    """One pass of the pricing paths: per path at the horizon ``D``, ``Σ w R_i²``, ``R̄`` and
    ``Σ w |R_i|``; and the basket's forward-moneyness level at the pillars."""

    def __init__(
        self, model: LocalCorrelationModel, sim: Any, weights: np.ndarray, pillars: Sequence[float]
    ) -> None:
        t0 = time.perf_counter()
        mats = [float(t) for t in pillars]
        grid = TimeGrid.build(mats, sim.dt_max, calibration_grid=model.required_times())
        draws = model.draws_for(grid, sim.seed, sim.n_paths, sim.antithetic)
        cols = [grid.fixing_index[t] for t in mats]
        n = sim.n_paths
        self.D, self.sq, self.rb, self.absr = (np.empty(n) for _ in range(4))
        self.levels = np.empty((n, len(mats)))
        for p0, p1 in sim.chunk_ranges(grid.n_records * model.n_assets, 0):
            paths = model.simulate_chunk(grid, draws, p0, p1, sim.scheme)
            r = paths.performances(cols[-1])
            mean = r @ weights
            self.D[p0:p1] = np.abs(r - mean[:, None]) @ weights
            self.sq[p0:p1] = (r * r) @ weights
            self.absr[p0:p1] = np.abs(r) @ weights
            self.rb[p0:p1] = mean
            assert paths.aux is not None
            self.levels[p0:p1] = np.exp(paths.aux["k_basket"][:, cols])
        self.pillars = mats
        self.seconds = time.perf_counter() - t0

    @property
    def V(self) -> np.ndarray:
        return np.asarray(self.sq - self.rb**2)


def index_errors(levels: np.ndarray, pillars: Sequence[float], surface: Any) -> pd.DataFrame:
    """The basket's implied vol under the model against the index target at each pillar and at
    strikes in at-the-money standard deviations, and at the 90 % strike: error and standard
    error in vol points."""
    rows = []
    for i, T in enumerate(pillars):
        atm = float(surface.atm_vol(T))
        ks = [(f"{m:+.1f}", m * atm * math.sqrt(T)) for m in SD_GRID] + [("90%", math.log(0.9))]
        for label, k in ks:
            cp = 1.0 if k >= 0 else -1.0
            pay = np.maximum(cp * (levels[:, i] - math.exp(k)), 0.0)
            price, se = lcd.pair_mean(pay)
            vol = float(implied_vol(price, 1.0, math.exp(k), T, cp))
            target = float(surface.implied_vol_k(k, T))
            vega = float(black_vega(1.0, math.exp(k), T, vol)) if np.isfinite(vol) else float("nan")
            rows.append({"T": float(T), "strike": label, "k": k, "model_vol": vol, "target_vol": target,
                         "error_vp": 100 * (vol - target), "stderr_vp": 100 * se / vega if vega > 0 else float("nan")})  # fmt: skip
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# one date
# ---------------------------------------------------------------------------------------------


def run_date(
    date: str,
    tenor: str,
    cfg: dict[str, Any],
    *,
    budget: str = "production",
    risk: str = "none",
    root: str | Path | None = None,
    varswap: bool = False,
    allow_calibrate: bool = True,
) -> dict[str, Any]:
    """The row of ``date`` (module docstring).  Raises on a failure (:func:`safe_row` wraps it)."""
    t_start = time.perf_counter()
    out = out_root(cfg, root)
    cache_root = (Path(root) if root else ROOT) / cfg["cache"]
    records = FitRecords(cache_root / "svi_fits")
    cache = LocalCorrelationCache(cache_root)
    inp = lcd.load_inputs(date, tenor, cfg["index"])
    spec, info = spec_for(inp, cfg, budget, records)
    T, w = spec.lc.particle.horizon, np.array(spec.weights)
    b1 = inp.entry["B1"]
    row: dict[str, Any] = {
        "date": date, "tenor": tenor, "T": T, "n_names": spec.n_names, "mode": spec.lc.mode,
        "family": spec.lc.family, "r_low": spec.lc.r_low, "budget": budget, "risk": risk,
        "particle_seed": spec.lc.particle.seed, "pricing_seed": spec.sim.seed,
        "n_particles": spec.lc.particle.n_particles, "n_paths": spec.sim.n_paths,
        "companion_paths": spec.lc.parametric.n_paths, "git_commit": code_version(),
        "spec_key": lc_spec_key(spec), "lc_code_tag": LC_CODE_TAG, "schedule": repr(spec.sim.step_schedule),
        "threads": num_threads(), "status": "ok", "reason": "",
    }  # fmt: skip
    # --- the inputs
    by_rule: dict[str, int] = {}
    for g in info["dropped"]:
        by_rule[g["rule"]] = by_rule.get(g["rule"], 0) + 1
    last_index = float(max(info["index_slices"]))
    row.update(
        n_dropped=len(info["dropped"]), n_dropped_third_friday=by_rule.get("third_friday", 0),
        n_dropped_strikes=by_rule.get("strikes", 0), n_dropped_spread=by_rule.get("spread", 0),
        n_dropped_calendar=by_rule.get("calendar", 0), calendar_repair=bool(cfg["screen"].get("calendar_repair", False)),
        n_dropped_index=sum(g["leg"] == "index" for g in info["dropped"]),
        svi_rms_vp_median=info["svi_rms_vp_median"], svi_rms_vp_max=info["svi_rms_vp_max"],
        svi_rms_vp_index=float(np.max(info["svi_rms_vp_index"])),
        n_names_extrapolated=info["n_names_extrapolated"], index_extrapolated=bool(info["index_extrapolated"]),
        n_names_unscreened=info["n_names_unscreened"], names_unscreened=",".join(info["names_unscreened"]),
        index_last_slice=last_index,
    )  # fmt: skip
    # --- calibration, through the cache
    t0 = time.perf_counter()
    hit = cache.has(spec)
    model, diag = cache.get_or_calibrate(spec, allow_calibrate=allow_calibrate)
    assert diag is not None
    t_cal = time.perf_counter() - t0
    cal, mkt = diag.calibration, diag.market
    market = build_lc_market(spec)
    surface = market.index_surface
    sd_T = float(surface.atm_vol(T)) * math.sqrt(T)
    lam_T = [float(model.lam(T, np.array([m * sd_T]))[0]) for m in (-1.0, 0.0, 1.0)]
    align = mkt["alignment"]
    below = [a for a in align if a["T"] <= T + 1e-9]
    above = [a for a in align if a["T"] > T + 1e-9]
    row.update(
        cache_hit=hit, lambda_t0=float(model.lam.values[0, 0]), lambda_mean_T=float(cal["lambda_mean"][-1]),
        lambda_T_m1sd=lam_T[0], lambda_T_atm=lam_T[1], lambda_T_p1sd=lam_T[2],
        clip_low_max=cal["max_clipped_low"], clip_high_max=cal["max_clipped_high"],
        clip_low_mean=cal["mean_clipped_low"], clip_high_mean=cal["mean_clipped_high"],
        clip_inner_max=cal["max_clipped_mass_inner"], clip_low_inner_max=cal["max_clipped_low_inner"],
        clip_high_inner_max=cal["max_clipped_high_inner"],
        n_slices=cal["n_slices"], wing_binds=bool(cal["max_clipped_mass_inner"] > 0.01),
        align_below=below[-1]["delta"] if below else float("nan"),
        align_above=above[0]["delta"] if above else float("nan"),
        align_flagged=bool(mkt["alignment_flagged"]),
        floored_names_max=mkt["max_floored_fraction_names"], floored_index=mkt["floored_fraction_index"],
        n_flagged_names=len([x for x in mkt["flagged"] if x != "index"]), index_flagged="index" in mkt["flagged"],
        n_flagged_central=len(cal.get("arbitrage_visited", {}).get("flagged_central", [])),
        seconds_calibration=float(cal["wall_time"]), seconds_calibration_step=t_cal,
    )  # fmt: skip
    # --- the constant-correlation companion, through the cache
    t0 = time.perf_counter()
    cc_spec = dataclasses.replace(spec, lc=dataclasses.replace(spec.lc, family="constant"))
    model_cc, diag_cc = cache.get_or_calibrate(cc_spec, allow_calibrate=allow_calibrate)
    assert diag_cc is not None
    lam_c = float(diag_cc.calibration["lambda"])
    row.update(lambda_c=lam_c, rho_cc=float(market.family.equicorrelation(lam_c)),
               seconds_companion=time.perf_counter() - t0)  # fmt: skip
    # --- prices: one pass per model on the same paths
    pillars = lcd.repricing_pillars(spec)
    lc, cc = Pass(model, spec.sim, w, pillars), Pass(model_cc, spec.sim, w, pillars)
    row.update(seconds_pricing_lc=lc.seconds, seconds_pricing_cc=cc.seconds)

    def put(name: str, value: tuple[float, float]) -> None:
        row[name], row[f"{name}_se"] = float(value[0]), float(value[1])

    put("ED_lc", lcd.pair_mean(lc.D))
    put("ED_cc", lcd.pair_mean(cc.D))
    put("ratio", lcd.pair_ratio(lc.D, cc.D))
    row["sd_D_lc"], row["sd_D_cc"] = float(lc.D.std(ddof=1)), float(cc.D.std(ddof=1))
    strikes = [float(k) for k in b1["strikes"]]
    for tag, K in zip(MULT_TAGS, strikes, strict=True):
        a, b = np.maximum(lc.D - K, 0.0), np.maximum(cc.D - K, 0.0)
        row[f"K_{tag}"] = K
        put(f"C_{tag}_lc", lcd.pair_mean(a))
        put(f"C_{tag}_cc", lcd.pair_mean(b))
        put(
            f"C_{tag}_ratio", lcd.pair_ratio(a, b) if b.mean() > 0 else (float("nan"), float("nan"))
        )
    ed_cc = row["ED_cc"]
    for mult in (0.75, 1.0, 1.25, 1.5):  # at multiples of the CC forward (the reference's strikes)
        a, b = np.maximum(lc.D - mult * ed_cc, 0.0), np.maximum(cc.D - mult * ed_cc, 0.0)
        put(
            f"Cfwd_{round(100 * mult):03d}_ratio",
            lcd.pair_ratio(a, b) if b.mean() > 0 else (float("nan"), float("nan")),
        )
    for tag, s in (("lc", lc), ("cc", cc)):
        put(f"EV_{tag}", lcd.pair_mean(s.V))
        put(f"kappa_{tag}", lcd.pair_ratio_sqrt(s.D, s.V))
        put(f"straddle_B_{tag}", lcd.pair_mean(np.abs(s.rb)))
        put(f"put90_B_{tag}", lcd.pair_mean(np.maximum(-0.10 - s.rb, 0.0)))
        put(f"SS_{tag}", lcd.pair_mean(s.absr))
        put(f"gap_{tag}", lcd.pair_mean(s.D - s.absr + np.abs(s.rb)))
        rel = s.D / (1.0 + s.rb)
        bucket = np.digitize(s.rb, dcop.PROFILE_EDGES)
        row[f"E_DB_{tag}"] = float(rel.mean())
        for j in range(5):
            hit_j = bucket == j
            row[f"profile_{tag}_{j}"] = (
                float(rel[hit_j].mean() / rel.mean()) if hit_j.any() else float("nan")
            )
            row[f"profile_share_{tag}_{j}"] = float(hit_j.mean())
    # --- E[V] against the listed strips, and the wing-corrected forwards
    eqv = float(b1["EQV"])
    sum_wm = float(np.sum(inp.weights / inp.weights.sum() * np.asarray(inp.entry["legs"]["M"])))
    m_b = sum_wm - eqv
    put("sum_w_ER2_lc", lcd.pair_mean(lc.sq))
    # the same second moment from the names' own SVI surfaces (no Monte Carlo): what the model
    # must reprice; its distance to the listed strips is the smiles' extrapolation
    sum_svi = float(sum(wi * float(strip_second_moment(s_i, T)) for wi, s_i in zip(w, market.surfaces, strict=True)))  # type: ignore[arg-type]
    row.update(
        sum_w_M_svi=sum_svi, names_mc_over_svi=row["sum_w_ER2_lc"] / sum_svi - 1.0,
        names_mc_z=(row["sum_w_ER2_lc"] - sum_svi) / row["sum_w_ER2_lc_se"], names_svi_over_listed=sum_svi / sum_wm - 1.0,
        names_mc_over_listed=row["sum_w_ER2_lc"] / sum_wm - 1.0,
    )  # fmt: skip
    put("E_Rbar2_lc", lcd.pair_mean(lc.rb**2))
    row.update(
        EQV=eqv, sum_w_M=sum_wm, M_B_listed=m_b, EV_over_EQV=row["EV_lc"] / eqv,
        EV_over_EQV_se=row["EV_lc_se"] / eqv, EV_single_part=row["sum_w_ER2_lc"] - sum_wm,
        EV_single_part_se=row["sum_w_ER2_lc_se"], EV_basket_part=row["E_Rbar2_lc"] - m_b,
        EV_basket_part_se=row["E_Rbar2_lc_se"], P_D_copula=float(b1["P_D"]), EV_copula=float(b1["EV"]),
        rho_cop=float(inp.entry["rho_cop"]),
    )  # fmt: skip
    rb2 = lc.rb**2
    put(
        "ED_wing",
        delta_method(
            lambda m: m[0] * math.sqrt(max(m[1] - m_b, 0.0) / (m[1] - m[2])), [lc.D, lc.sq, rb2]
        ),
    )
    put("ED_eqv", delta_method(lambda m: m[0] * math.sqrt(eqv / (m[1] - m[2])), [lc.D, lc.sq, rb2]))
    put(
        "ED_wing_ratio",
        delta_method(
            lambda m: m[0] * math.sqrt(max(m[1] - m_b, 0.0) / (m[1] - m[2])) / m[3],
            [lc.D, lc.sq, rb2, cc.D],
        ),
    )
    put(
        "ED_eqv_ratio",
        delta_method(
            lambda m: m[0] * math.sqrt(eqv / (m[1] - m[2])) / m[3], [lc.D, lc.sq, rb2, cc.D]
        ),
    )
    # --- the index smile against its target
    err = index_errors(lc.levels, pillars, surface)
    at_T = err[np.isclose(err["T"], T)]
    inner = err[err["strike"].isin([f"{m:+.1f}" for m in SD_GRID if abs(m) <= 1.5])]
    outer = err[err["strike"] != "90%"]
    fwd, fwd_se = lcd.pair_mean(lc.levels[:, -1] - 1.0)
    row.update(
        idx_err_max_1p5=float(inner["error_vp"].abs().max()), idx_err_max_2p5=float(outer["error_vp"].abs().max()),
        idx_err_atm=float(at_T[at_T["strike"] == "+0.0"]["error_vp"].iloc[0]),
        idx_err_atm_se=float(at_T[at_T["strike"] == "+0.0"]["stderr_vp"].iloc[0]),
        idx_err_90=float(at_T[at_T["strike"] == "90%"]["error_vp"].iloc[0]),
        idx_err_90_se=float(at_T[at_T["strike"] == "90%"]["stderr_vp"].iloc[0]),
        forward_error=fwd, forward_error_se=fwd_se,
    )  # fmt: skip
    for m in SD_GRID:
        cell = at_T[at_T["strike"] == f"{m:+.1f}"]
        tag = f"{'m' if m < 0 else 'p'}{abs(m):.1f}".replace(".", "")
        row[f"idx_err_{tag}"] = float(cell["error_vp"].iloc[0])
        row[f"idx_err_{tag}_se"] = float(cell["stderr_vp"].iloc[0])
    row["index_errors"] = err.to_dict("records")
    # --- S9: the basket variance swap (reference dates)
    if varswap:
        nodes = np.asarray(model.required_times())
        n_fix = round(T * 252)
        fixings = sorted(
            {
                float(nodes[int(np.argmin(np.abs(nodes - j * T / n_fix)))])
                for j in range(1, n_fix + 1)
            }
        )
        swap = BasketVarianceSwap(w, fixings, 0.0, ZERO, annualisation=len(fixings) / T)
        rv = MultiAssetMonteCarlo(spec.sim).price(swap, model)
        # -2/T E[ln(B_T/F)], the forward as control variate (E[B_T/F] = 1; the forward error is
        # reported on its own): the estimator's noise falls from that of ln to that of its square
        level = lc.levels[:, -1]
        logk, logk_se = lcd.pair_mean(-2.0 / T * (np.log(level) - (level - 1.0)))
        listed = float(varswap_strike(surface, T))
        row.update(
            RV_B_lc=float(rv.mean), RV_B_lc_se=float(rv.stderr), n_fixings=len(fixings),
            varswap_model_smile=logk, varswap_model_smile_se=logk_se, varswap_listed=listed,
            varswap_vol_rv=100 * math.sqrt(rv.mean), varswap_vol_model_smile=100 * math.sqrt(logk),
            varswap_vol_listed=100 * math.sqrt(listed),
            varswap_vol_rv_se=100 * float(rv.stderr) / (2 * math.sqrt(rv.mean)),
            varswap_vol_model_smile_se=100 * logk_se / (2 * math.sqrt(logk)),
        )  # fmt: skip
    # --- risk
    if risk in ("deltas", "full"):
        t0 = time.perf_counter()
        base_prices = {
            "lc": (row["ED_lc"], row["C_100_lc"]),
            "cc": (row["ED_cc"], row["C_100_cc"]),
        }
        row.update(deltas(model, model_cc, spec, strikes[2], cfg["delta_bump"], base_prices))
        row["seconds_deltas"] = time.perf_counter() - t0
    if risk == "full":
        t0 = time.perf_counter()
        row["full_risk"] = full_risk(spec, cache, inp, cfg)
        row["seconds_full_risk"] = time.perf_counter() - t0
    # --- sanity checks (SPEC §8.7; the specification's §9.4)
    numbers = [
        v
        for k, v in row.items()
        if isinstance(v, float) and not k.startswith(("align_", "C_200", "Cfwd", "profile_"))
    ]
    row["check_no_nan"] = bool(all(np.isfinite(v) for v in numbers))
    row["check_forward"] = bool(abs(fwd) <= 3.0 * fwd_se)
    row["check_index"] = bool(
        row["wing_binds"] or (abs(row["idx_err_atm"]) <= 0.15 and abs(row["idx_err_90"]) <= 0.15)
    )
    row["check_names"] = bool(abs(row["sum_w_ER2_lc"] / sum_wm - 1.0) <= 0.02)
    if not (
        row["check_no_nan"] and row["check_forward"] and row["check_index"] and row["check_names"]
    ):
        failed = [
            k for k in ("check_no_nan", "check_forward", "check_index", "check_names") if not row[k]
        ]
        row["status"], row["reason"] = "check", "sanity checks: " + ", ".join(failed)
    row["seconds_total"] = time.perf_counter() - t_start
    row["output_root"] = str(out)
    return row


def deltas(
    model: LocalCorrelationModel,
    model_cc: LocalCorrelationModel,
    spec: LocalCorrelationSpec,
    k_100: float,
    bump: str,
    base_prices: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    """The sticky-strike common deltas of the forward and of the call at ``K_100`` under LC and
    CC, as elasticities (percent of each model's price per +1 % of spot; the base prices are the
    pricing pass's), and the decomposition of the forward's: homogeneity exactly 1 (a common
    move at sticky moneyness scales ``D``), skew channel ``Δ^CC_ss − 1``, correlation channel
    ``Δ^LC_ss − Δ^CC_ss`` (paired: the two models are bumped on the same paths)."""
    T, w = spec.lc.particle.horizon, np.array(spec.weights)
    products = [Palladium(w, 0.0, T, ZERO), Palladium(w, k_100, T, ZERO)]
    n = spec.n_names
    h = lcr.SPOT_BUMP
    up, dn = (1.0 + h, 1.0 - h) if bump == "arith" else (math.exp(h), math.exp(-h))
    out: dict[str, Any] = {"delta_bump": bump, "delta_bump_size": h, "delta_homogeneity": 1.0}
    forward: dict[str, np.ndarray] = {}

    def stat(x: np.ndarray) -> tuple[float, float]:
        return float(x.mean()), float(x.std(ddof=1) / np.sqrt(x.size))

    for tag, m in (("lc", model), ("cc", model_cc)):
        engine = lcr.LCRiskEngine(lcr.FixedLambdaBuilder(m, tag.upper()), spec.sim)
        base = engine.builder.base
        p_up = engine.priced_many(products, base.with_spots([up] * n, "sticky_strike", f"{tag} up"))
        p_dn = engine.priced_many(
            products, base.with_spots([dn] * n, "sticky_strike", f"{tag} down")
        )
        for name, a, b, p0 in zip(("fwd", "C100"), p_up, p_dn, base_prices[tag], strict=True):
            x = (a.payoffs - b.payoffs) / (up - dn) / p0
            out[f"delta_{name}_{tag}"], out[f"delta_{name}_{tag}_se"] = stat(x)
            if name == "fwd":
                forward[tag] = x
    out["delta_skew_channel"], out["delta_skew_channel_se"] = stat(forward["cc"] - 1.0)
    out["delta_correlation_channel"], out["delta_correlation_channel_se"] = stat(
        forward["lc"] - forward["cc"]
    )
    return out


def full_risk(
    spec: LocalCorrelationSpec,
    cache: LocalCorrelationCache,
    inp: lcd.DowInputs,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Today's full risk of the Palladium forward (SPEC §8.7): the single-name vegas with ``λ``
    recalibrated and held, the all-names bump, the index vega, the two index skew vegas, and the
    model-risk range over ``R_low`` ∈ {equi 0, equi 0.02, equi 0.10, historical-scaled:252,0.05}
    × family ∈ {particle, parametric}."""
    T, w = spec.lc.particle.horizon, np.array(spec.weights)
    forward = Palladium(w, 0.0, T, ZERO)
    state = lcr.LCState(spec)
    # the historical-scaled R_low: the 252 daily log returns of the names up to the date; the
    # key carries the digest of that window, never the matrix (SPEC §13.3)
    panel = dd.prices().ffill()
    rets = np.log(panel[inp.names]).diff().loc[: inp.date].iloc[-252:]
    window = np.ascontiguousarray(rets.to_numpy(), dtype=np.float64)
    hist = historical_scaled_correlation(window, w, 0.05)
    source = hashlib.sha256(window.tobytes()).hexdigest()
    variants: list[tuple[str, LocalCorrelationSpec]] = []
    matrices: dict[str, Any] = {}
    for family in ("particle", "parametric"):
        for label, changes, extra in (
            ("equi 0", {"rho_min": 0.0}, {}),
            ("equi 0.02", {"rho_min": 0.02}, {}),
            ("equi 0.10", {"rho_min": 0.10}, {}),
            (
                "historical-scaled:252,0.05",
                {"r_low": "historical-scaled:252,0.05"},
                {"r_low_source": source},
            ),
        ):
            other = dataclasses.replace(
                spec, lc=dataclasses.replace(spec.lc, family=family, **changes), **extra
            )
            if lc_spec_key(other) == lc_spec_key(spec):
                continue
            variants.append((f"{label} / {family}", other))
            if "r_low_source" in extra:
                matrices[lc_spec_key(other)] = hist.r_low
    builder = lcr.LCBuilder(cache, state, matrices=matrices)
    engine = lcr.LCRiskEngine(builder, spec.sim)
    out: dict[str, Any] = {
        "historical_scaled": {
            **hist.info(),
            "window": [str(rets.index[0])[:10], str(rets.index[-1])[:10]],
            "digest": source,
        }
    }

    def rows(items: Sequence[Any]) -> list[dict[str, Any]]:
        return [{"name": s.name, "value": s.value, "stderr": s.stderr, "unit": s.unit,
                 "unpaired_stderr": s.extra.get("unpaired_stderr")} for s in items]  # fmt: skip

    out["name_vegas_recalibrated"] = rows(lcr.name_vegas(engine, forward, state))
    out["name_vegas_held"] = rows(lcr.name_vegas(engine, forward, state, lambda_mode="held"))
    out["index_vega"] = rows([lcr.index_vega(engine, forward, state)])
    out["index_skew_vegas"] = rows(
        [lcr.index_skew_vega(engine, forward, state, kind=k) for k in ("rotation", "put90")]
    )
    out["model_risk"] = lcr.model_risk_range(engine, forward, state, variants).to_dict()
    out["n_calibrated"], out["n_built"], out["n_pricings"] = (
        builder.n_missed,
        builder.n_built,
        engine.n_pricings,
    )
    return out


def safe_row(date: str, tenor: str, cfg: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """:func:`run_date`, a failure turned into a row with its status and reason."""
    try:
        return run_date(date, tenor, cfg, **kwargs)
    except Exception as exc:
        log.error("%s %s failed: %s\n%s", date, tenor, exc, traceback.format_exc())
        return {"date": date, "tenor": tenor, "budget": kwargs.get("budget", "production"), "risk": kwargs.get("risk", "none"),
                "git_commit": code_version(), "lc_code_tag": LC_CODE_TAG, "status": "failed",
                "reason": f"{type(exc).__name__}: {exc}"[:500]}  # fmt: skip


def log_row(row: dict[str, Any]) -> None:
    if row["status"] == "failed":
        log.info("%s %s: FAILED — %s", row["date"], row["tenor"], row["reason"])
        return
    r = row
    log.info(
        "=== %s %s (T = %.4f, %s budget, key %s): status %s %s",
        r["date"], r["tenor"], r["T"], r["budget"], r["spec_key"][:12], r["status"], r["reason"],
    )  # fmt: skip
    log.info(
        "calibration %.0f s (cache hit %s); lambda(t0) %.4f, cloud mean at T %.4f; clipped mass max low %.4f, high %.4f, "
        "inside ±%.1f sd %.4f; rho_CC %.5f; %d expiries dropped; index extrapolated %s",
        r["seconds_calibration"], r["cache_hit"], r["lambda_t0"], r["lambda_mean_T"], r["clip_low_max"], r["clip_high_max"],
        CLIP_GATE_SD, r["clip_inner_max"], r["rho_cc"], r["n_dropped"], r["index_extrapolated"],
    )  # fmt: skip
    log.info(
        "E_LC[D] %.6f (%.6f); E_CC[D] %.6f (%.6f); LC/CC %.5f (%.5f); copula P_D %.6f; kappa LC %.4f (%.4f), CC %.4f (%.4f)",
        r["ED_lc"], r["ED_lc_se"], r["ED_cc"], r["ED_cc_se"], r["ratio"], r["ratio_se"], r["P_D_copula"],
        r["kappa_lc"], r["kappa_lc_se"], r["kappa_cc"], r["kappa_cc_se"],
    )  # fmt: skip
    log.info(
        "calls LC/CC at K_050..K_200: %s",
        "  ".join(f"{r[f'C_{t}_ratio']:.4f} ({r[f'C_{t}_ratio_se']:.4f})" for t in MULT_TAGS),
    )
    log.info(
        "calls LC/CC at 0.75/1/1.25/1.5 x the CC forward: %s",
        "  ".join(
            f"{r[f'Cfwd_{t}_ratio']:.4f} ({r[f'Cfwd_{t}_ratio_se']:.4f})"
            for t in ("075", "100", "125", "150")
        ),
    )
    log.info(
        "E_LC[V]/E_Q[V] %.4f (%.4f): single-name part %+.6f (%.6f), basket part %+.6f (%.6f); ED_wing %.6f (%.6f) = %.4f (%.4f) x ED_cc; "
        "ED_eqv %.6f (%.6f) = %.4f (%.4f) x ED_cc",
        r["EV_over_EQV"], r["EV_over_EQV_se"], r["EV_single_part"], r["EV_single_part_se"], r["EV_basket_part"], r["EV_basket_part_se"],
        r["ED_wing"], r["ED_wing_se"], r["ED_wing_ratio"], r["ED_wing_ratio_se"], r["ED_eqv"], r["ED_eqv_se"], r["ED_eqv_ratio"], r["ED_eqv_ratio_se"],
    )  # fmt: skip
    log.info(
        "names: sum w E[R_i^2] MC %.6f (%.6f), own SVI strips %.6f (z %+.2f), listed strips %.6f: MC/listed %+.2f %%, SVI/listed %+.2f %%",
        r["sum_w_ER2_lc"], r["sum_w_ER2_lc_se"], r["sum_w_M_svi"], r["names_mc_z"], r["sum_w_M"],
        100 * r["names_mc_over_listed"], 100 * r["names_svi_over_listed"],
    )  # fmt: skip
    log.info(
        "index smile at T: ATM %+.3f (%.3f), 90%% %+.3f (%.3f), -1.5 sd %+.3f, -2.5 sd %+.3f; max inside ±1.5 sd %.3f, ±2.5 sd %.3f vp; forward %+.1e (%.1e)",
        r["idx_err_atm"], r["idx_err_atm_se"], r["idx_err_90"], r["idx_err_90_se"], r["idx_err_m15"], r["idx_err_m25"],
        r["idx_err_max_1p5"], r["idx_err_max_2p5"], r["forward_error"], r["forward_error_se"],
    )  # fmt: skip
    if "delta_fwd_lc" in r:
        log.info(
            "sticky-strike common deltas (%% of price per +1 %%): forward LC %+.4f (%.4f), CC %+.4f (%.4f); call K_100 LC %+.3f (%.3f), CC %+.3f (%.3f); "
            "decomposition of the forward: homogeneity +1, skew %+.4f (%.4f), correlation %+.4f (%.4f)",
            r["delta_fwd_lc"], r["delta_fwd_lc_se"], r["delta_fwd_cc"], r["delta_fwd_cc_se"], r["delta_C100_lc"], r["delta_C100_lc_se"],
            r["delta_C100_cc"], r["delta_C100_cc_se"], r["delta_skew_channel"], r["delta_skew_channel_se"],
            r["delta_correlation_channel"], r["delta_correlation_channel_se"],
        )  # fmt: skip
    if "RV_B_lc" in r:
        log.info(
            "basket variance swap: realised under LC %.4f (%.4f) vp; log contract of the model's own smile %.4f (%.4f) vp (difference %+.4f); listed strip %.4f vp (model smile %+.4f)",
            r["varswap_vol_rv"], r["varswap_vol_rv_se"], r["varswap_vol_model_smile"], r["varswap_vol_model_smile_se"],
            r["varswap_vol_rv"] - r["varswap_vol_model_smile"], r["varswap_vol_listed"], r["varswap_vol_model_smile"] - r["varswap_vol_listed"],
        )  # fmt: skip
    log.info(
        "seconds: calibration %.0f, companion %.0f, pricing LC %.0f + CC %.0f, deltas %.0f, total %.0f",
        r["seconds_calibration_step"], r["seconds_companion"], r["seconds_pricing_lc"], r["seconds_pricing_cc"],
        r.get("seconds_deltas", 0.0), r["seconds_total"],
    )  # fmt: skip


def jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(f"not JSON-able: {type(x).__name__}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--date", required=True)
    ap.add_argument("--tenor", default="3m")
    ap.add_argument("--budget", default="production", choices=("production", "development"))
    ap.add_argument("--risk", default=None, choices=("none", "deltas", "full"))
    ap.add_argument("--varswap", action="store_true", help="also the basket variance swap (S9)")
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument(
        "--root",
        default=None,
        help="the worktree that holds outputs/dispersion_lc (default: this one)",
    )
    ap.add_argument(
        "--row-out",
        default=None,
        help="where the row goes (default: <outputs>/price_<date>_<tenor>_<budget>.json)",
    )
    ap.add_argument("--strict", action="store_true", help="exit 1 when the row's status is not ok")
    ap.add_argument("--profile", action="store_true", help="log the library's timings")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("volsto").setLevel(logging.INFO if args.profile else logging.WARNING)
    cfg = load_config(args.config)
    risk = args.risk or cfg["risk"]
    row = safe_row(
        args.date,
        args.tenor,
        cfg,
        budget=args.budget,
        risk=risk,
        root=args.root,
        varswap=args.varswap,
    )
    row["config_digest"] = config_digest(cfg, args.tenor, args.budget)
    log_row(row)
    path = (
        Path(args.row_out)
        if args.row_out
        else out_root(cfg, args.root) / f"price_{args.date}_{args.tenor}_{args.budget}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(row, indent=1, default=jsonable))
    tmp.replace(path)
    log.info("written %s", path)
    return 1 if (args.strict and row["status"] != "ok") else 0


if __name__ == "__main__":
    sys.exit(main())
