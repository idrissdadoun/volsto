"""Local correlation model (SPEC §8.7, M12): diagnostics on the dispersion study's Dow basket.

    python scripts/lcm_diagnostics.py screen    [--dates D ...] [--tenor 3m]
    python scripts/lcm_diagnostics.py calibrate --date D [--tenor 3m] [--screen default] [--rho-max 0.98]
    python scripts/lcm_diagnostics.py dt        --date D [--target-average step|point]
    python scripts/lcm_diagnostics.py tail      --date D
    python scripts/lcm_diagnostics.py wing      --date D
    python scripts/lcm_diagnostics.py baseline  [--write]
    python scripts/lcm_diagnostics.py strips    --date D [--screen repair|norepair|...] [--paths N] [--model zero|lc]
    python scripts/lcm_diagnostics.py calendar  --date D

``--screen`` names one of ``SCREENS``: ``default`` (the M12 default of ``ExpiryScreen``),
``norepair`` and ``repair`` (the default quote rules without and with the names' calendar repair
and the unscreened fallback, both stated), ``third-friday``, ``off``.

Every Dow number of SPEC §8.7 comes from one of these commands (the synthetic ones come from the
slow tests of ``tests/test_local_correlation.py`` and from ``scripts/lcm_synthetic.py``):

* ``screen`` — which listed expiries enter the model on a date (the expiry screen of
  ``volsto/studies/disp_lc.py``: third-Friday index expiries, the quote screen on every leg),
  the quote quality of the index expiries, the alignment of the listed index forwards with the
  basket's, the arbitrage flags.  No calibration.
* ``calibrate`` — the particle calibration at a given budget with its timings; the clipped mass
  per step (total and inside ±2.5 sd); the index repricing table; the constant-correlation
  companion and the Palladium forward and calls under both, paired; ``E[V]`` against the listed
  ``E^Q[V]`` with the split into its single-name and basket parts.
* ``dt`` — the Δt check: calibration and pricing on the step schedule and on the schedule with
  every step halved, on common random numbers.
* ``tail`` — the ``λ`` tail rule, ``"regressions"`` against ``"flat"``, on the same particles and
  the same pricing paths.
* ``wing`` — the three diagnostics of the downside wing (owner's decision 5 of 2026-10-08):
  (a) the SVI fits of the names against their quotes between −2.5 and −1.5 standard deviations;
  (b) the comonotonic upper bound on the index puts implied by the names' marginals,
  ``E[(K − Σ w_i q_i(U))⁺]``, against the DJX quotes; (c) the sensitivity to ``rho_max = 0.995``.
* ``baseline`` — the run behind the M12 baseline of test S11 (``tests/golden/``).
* ``strips`` — the single-name part of ``E[V] − E^Q[V]``: each name's second moment from its SVI
  surface against the study's strip (vols linear between listed strikes, flat beyond them), by
  region of the strike axis, and the model's Monte Carlo against the SVI strip — in total and,
  exactly per path, by the same regions (``model_regions``: puts and calls beyond the last
  listed strike, and inside the listed range), with the names that carry the difference (the
  owner's decision 3 of 2026-10-09: which tail puts ``Σ w E[R_i²]`` above the strips), and how
  few paths carry the region above the listed strikes (``call_wing_paths``).  No calibration by
  default, the names' law does not depend on the correlation; ``--model lc`` calibrates and
  splits the calibrated model's pricing paths, the ones the sweep's check is computed on.
* ``calendar`` — the effect of the calendar crossings inside ±2 sd of the cloud: the offending
  slices dropped, the model recalibrated, ``E[D]``, the calls and the index smile compared on
  the same particles and the same pricing paths.  Its base is the chosen screen with the
  calendar repair off, whatever the default: the command drops the slices itself.

The smiles are the study's own (``scripts/disp_entries.py::smiles_of``, the loader behind
``marginals_for``; its defaults are untouched).  Results are logged and written as JSON under
``outputs/dispersion_lc/diagnostics/`` (git-ignored) with the reproducibility record.  Standard errors are on
antithetic pair means throughout.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import pickle
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import ndtr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_entries as de  # noqa: E402

from volsto._numba import num_threads  # noqa: E402
from volsto.calibration.cache import code_version  # noqa: E402
from volsto.calibration.lc_cache import (  # noqa: E402
    LCMarket,
    build_lc_market,
    lc_spec_key,
    visited_arbitrage,
)
from volsto.calibration.local_correlation import (  # noqa: E402
    CLIP_GATE_SD,
    LC_CODE_TAG,
    LCCalibrationResult,
    calibrate_constant_lambda,
    calibrate_local_correlation,
    reprice_index_smile,
)
from volsto.config import (  # noqa: E402
    LC_STEP_SCHEDULE,
    LocalCorrelationConfig,
    LocalCorrelationSpec,
    ParticleConfig,
    SimConfig,
    StepSchedule,
    lc_sim_config,
)
from volsto.engine.grid import TimeGrid  # noqa: E402
from volsto.market.bs import black_price, black_vega, implied_vol  # noqa: E402
from volsto.market.curves import DiscountCurve, ForwardCurve  # noqa: E402
from volsto.market.svi_slices import SviSlices, svi_total_variance  # noqa: E402
from volsto.multi.analytics import strip_second_moment  # noqa: E402
from volsto.multi.lc_draws import LocalCorrelationDraws  # noqa: E402
from volsto.multi.lc_function import LocalCorrelationFunction  # noqa: E402
from volsto.multi.lc_model import LocalCorrelationModel  # noqa: E402
from volsto.studies import disp_data as dd  # noqa: E402
from volsto.studies import disp_smile as ds  # noqa: E402
from volsto.studies.disp_lc import (  # noqa: E402
    ExpiryScreen,
    QuoteQuality,
    lc_spec_from_smiles,
    quote_quality,
    third_friday,
)

log = logging.getLogger("lcm_diagnostics")

ROOT = Path(__file__).resolve().parents[1]
#: Every output of the local correlation model on the study's data goes under this folder of the
#: worktree (owner's decision of 2026-10-08, second round) — never into ``outputs/dispersion``,
#: the study's own folder.
LC_OUT = ROOT / "outputs" / "dispersion_lc"
OUT = LC_OUT / "diagnostics"
#: The four dates of the reference implementation (today, typical, steep skew, tied alternate).
REFERENCE_DATES = ("2026-10-02", "2019-09-03", "2017-04-03", "2008-07-07")
PARTICLE_SEED = 12345
PRICING_SEED = 2024
PRODUCTION = 800_000
DAILY = 1.0 / 252.0
#: The expiry screens of ``--screen``, each stated in full so that none but ``default`` moves
#: with the defaults of ``ExpiryScreen``.  ``default``: the M12 default, whatever it is;
#: ``norepair``: the default quote rules, no calendar repair of the names and no unscreened
#: fallback (the screen of the development pass ``lcm_3m_dev_norepair``); ``repair``: the same
#: quote rules with both on (the variant pass ``lcm_3m_dev_repair``; the owner's decisions 1 and
#: 2 of 2026-10-09); ``third-friday``: the index's third-Friday rule alone, no quote rule,
#: neither option; ``off``: ``ExpiryScreen.off()``, every expiry the loader returns.
SCREENS: dict[str, ExpiryScreen] = {
    "default": ExpiryScreen(),
    "norepair": dataclasses.replace(
        ExpiryScreen(), calendar_repair=False, unscreened_fallback=False
    ),
    "repair": dataclasses.replace(ExpiryScreen(), calendar_repair=True, unscreened_fallback=True),
    "third-friday": ExpiryScreen(
        index_third_friday=True,
        nearest_two_sided=False,
        max_half_spread_vp=math.inf,
        max_half_spread_long_vp=math.inf,
        calendar_repair=False,
        unscreened_fallback=False,
    ),
    "off": ExpiryScreen.off(),
}
#: ``lc``: the model's schedule (quarter steps over the first two weeks); ``week``: the
#: one-week variant, reported for information; ``daily``: 1/252 throughout (the specification's
#: first choice, kept to reproduce the numbers measured before the owner's decision).
SCHEDULES: dict[str, StepSchedule | float] = {
    "lc": LC_STEP_SCHEDULE,
    "week": StepSchedule(breaks=(5.0 / 252.0,), dts=(1.0 / 1008.0, DAILY)),
    "daily": DAILY,
}
ZERO_CURVE = DiscountCurve.flat(0.0)
PARTICLE_FIELDS = frozenset({"bandwidth_factor", "estimator", "second_pass", "tail_extrapolation"})
SD_INNER = (-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5)


# ---------------------------------------------------------------------------------------------
# the study's inputs of a date
# ---------------------------------------------------------------------------------------------


@dataclass
class DowInputs:
    """The study's inputs of a date and tenor: the entry (names, maturity, price weights, the
    study's own prices), each leg's expiry smiles through the study's loader, the day they are
    from, and the quote quality of every expiry on that day."""

    date: str
    tenor: str
    entry: dict[str, Any]
    names: list[str]
    T: float
    weights: np.ndarray
    spots: list[float]
    smiles: dict[str, list[Any]]
    index_smiles: list[Any]
    index_spot: float
    quotes: dict[str, dict[str, QuoteQuality]]
    index_quotes: dict[str, QuoteQuality]
    smile_days: dict[str, str]


def load_inputs(date: str, tenor: str = "3m", index: str = "DJX") -> DowInputs:
    """The inputs of ``date``: the smiles are ``disp_entries.smiles_of`` — what
    ``marginals_for`` reads — and the quote quality is measured on the chain of the day the
    smiles are from."""
    with (dd.OUT / "entries" / tenor / f"{date}.pkl").open("rb") as fh:
        entry = pickle.load(fh)
    names = list(entry["names"])
    chains = dd.read_chains(date, [*names, index])
    panel = dd.prices().ffill()
    smiles: dict[str, list[Any]] = {}
    quotes: dict[str, dict[str, QuoteQuality]] = {}
    days: dict[str, str] = {}
    for t in [*names, index]:
        sm, day, _ = de.smiles_of(t, date, chains.get(t))
        if not sm or t not in panel.columns or not np.isfinite(panel.at[date, t]):
            raise ValueError(f"{date}: no usable smile or price for {t}")
        chain = chains[t] if day == date else dd.read_chains(day, [t])[t]
        smiles[t], quotes[t], days[t] = sm, quote_quality(chain, sm), day
    return DowInputs(
        date=date,
        tenor=tenor,
        entry=entry,
        names=names,
        T=float(entry["T"]),
        weights=np.asarray(entry["w_B1"], dtype=float),
        spots=[float(panel.at[date, t]) for t in names],
        smiles={t: smiles[t] for t in names},
        index_smiles=smiles[index],
        index_spot=float(panel.at[date, index]),
        quotes={t: quotes[t] for t in names},
        index_quotes=quotes[index],
        smile_days=days,
    )


def build_spec(
    inp: DowInputs,
    *,
    n_particles: int = PRODUCTION,
    n_paths: int = PRODUCTION,
    screen: ExpiryScreen | str = "default",
    schedule: StepSchedule | float = LC_STEP_SCHEDULE,
    particle_seed: int = PARTICLE_SEED,
    pricing_seed: int = PRICING_SEED,
    records: Any = None,
    **options: Any,
) -> tuple[LocalCorrelationSpec, dict[str, Any]]:
    """The specification of the model on the date's basket and the build's information.
    ``options`` go to ``ParticleConfig`` (bandwidth, estimator, second pass, tails) or to
    ``LocalCorrelationConfig`` (``rho_max``, ``lambda_tail``, ``target_average``, …)."""
    p_opts = {k: v for k, v in options.items() if k in PARTICLE_FIELDS}
    l_opts = {k: v for k, v in options.items() if k not in PARTICLE_FIELDS}
    particle = ParticleConfig(n_particles=n_particles, horizon=inp.T, seed=particle_seed, **p_opts)
    lc = LocalCorrelationConfig(particle=particle, **l_opts)
    sim = lc_sim_config(n_paths, pricing_seed, dt_max=schedule)
    return lc_spec_from_smiles(
        inp.names,
        inp.weights,
        inp.spots,
        inp.smiles,
        inp.index_smiles,
        inp.index_spot,
        inp.T,
        lc=lc,
        sim=sim,
        records=records,
        label=f"{inp.date} B1 {inp.tenor}",
        screen=SCREENS[screen] if isinstance(screen, str) else screen,
        quotes=inp.quotes,
        index_quotes=inp.index_quotes,
    )


# ---------------------------------------------------------------------------------------------
# statistics on antithetic pairs
# ---------------------------------------------------------------------------------------------


def pair_mean(x: np.ndarray) -> tuple[float, float]:
    """Mean and standard error on antithetic pair means."""
    y = 0.5 * (x[0::2] + x[1::2])
    return float(y.mean()), float(y.std(ddof=1) / np.sqrt(y.size))


def pair_ratio(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """``E[a]/E[b]`` with the delta-method standard error on pair means (the same pairs)."""
    ya, yb = 0.5 * (a[0::2] + a[1::2]), 0.5 * (b[0::2] + b[1::2])
    r = float(ya.mean() / yb.mean())
    u = (ya - r * yb) / yb.mean()
    return r, float(u.std(ddof=1) / np.sqrt(u.size))


# ---------------------------------------------------------------------------------------------
# simulation and the measured quantities
# ---------------------------------------------------------------------------------------------


@dataclass
class Sample:
    """Per path at the horizon: the dispersion ``D = Σ w|R_i − R̄|``, ``Σ w R_i²`` and ``R̄``
    (``R_i = S_i(T)/S_i(0) − 1``); and the basket's forward-moneyness level ``e^{k_B}`` at the
    pillars.  ``R``: the names' performances themselves, ``(n_paths, n_names)``, when the pass
    was asked to keep them (the ``strips`` command; ``None`` otherwise)."""

    D: np.ndarray
    sq: np.ndarray
    rb: np.ndarray
    levels: np.ndarray
    pillars: list[float]
    seconds: float
    R: np.ndarray | None = None

    @property
    def V(self) -> np.ndarray:
        return np.asarray(self.sq - self.rb**2)


def sample(
    model: LocalCorrelationModel,
    sim: SimConfig,
    weights: np.ndarray,
    pillars: Sequence[float],
    draws: LocalCorrelationDraws | None = None,
    *,
    keep_performances: bool = False,
) -> Sample:
    """One pass over the pricing paths (or over given draws: the Δt check).
    ``keep_performances``: also keep every name's ``R_i`` at the horizon, per path (8 bytes per
    path and name); the other fields are what the pass returns without it."""
    t0 = time.perf_counter()
    mats = [float(t) for t in pillars]
    grid = TimeGrid.build(mats, sim.dt_max, calibration_grid=model.required_times())
    d = draws if draws is not None else model.draws_for(grid, sim.seed, sim.n_paths, sim.antithetic)
    cols = [grid.fixing_index[t] for t in mats]
    n = sim.n_paths
    D, sq, rb, lev = np.empty(n), np.empty(n), np.empty(n), np.empty((n, len(mats)))
    R = np.empty((n, model.n_assets)) if keep_performances else None
    for p0, p1 in sim.chunk_ranges(grid.n_records * model.n_assets, 0):
        paths = model.simulate_chunk(grid, d, p0, p1, sim.scheme)
        r = paths.performances(cols[-1])
        mean = r @ weights
        D[p0:p1] = np.abs(r - mean[:, None]) @ weights
        sq[p0:p1] = (r * r) @ weights
        rb[p0:p1] = mean
        if R is not None:
            R[p0:p1] = r
        assert paths.aux is not None
        lev[p0:p1] = np.exp(paths.aux["k_basket"][:, cols])
    return Sample(D, sq, rb, lev, mats, time.perf_counter() - t0, R)


def smile_cells(
    level: np.ndarray, T: float, atm: float, sds: Sequence[float] = SD_INNER
) -> tuple[list[float], list[float], np.ndarray]:
    """``(strikes, cps, payoffs (n_paths, n_strikes))`` of the out-of-the-money options at
    ``k = m·atm·√T``."""
    ks = [m * atm * math.sqrt(T) for m in sds]
    cps = [1.0 if k >= 0 else -1.0 for k in ks]
    pay = np.column_stack(
        [np.maximum(cp * (level - math.exp(k)), 0.0) for k, cp in zip(ks, cps, strict=True)]
    )
    return [math.exp(k) for k in ks], cps, pay


def smile_difference(
    lev_a: np.ndarray, lev_b: np.ndarray, pillars: Sequence[float], surface: Any
) -> pd.DataFrame:
    """Implied vol of ``a`` minus ``b`` (the same paths) by pillar and strike, in vol points,
    with the paired standard error (the price difference's through the vega)."""
    rows = []
    for i, T in enumerate(pillars):
        atm = float(surface.atm_vol(T))
        strikes, cps, pa = smile_cells(lev_a[:, i], T, atm)
        _, _, pb = smile_cells(lev_b[:, i], T, atm)
        for j, (K, cp, m) in enumerate(zip(strikes, cps, SD_INNER, strict=True)):
            va = float(implied_vol(pa[:, j].mean(), 1.0, K, T, cp))
            vb = float(implied_vol(pb[:, j].mean(), 1.0, K, T, cp))
            se = pair_mean(pa[:, j] - pb[:, j])[1] / float(black_vega(1.0, K, T, vb))
            rows.append({"T": T, "sd": m, "diff_vp": 100 * (va - vb), "se_vp": 100 * se})
    return pd.DataFrame(rows)


def clip_summary(res: LCCalibrationResult) -> dict[str, Any]:
    """The clipped mass of a calibration: total and inside ``±CLIP_GATE_SD`` sd."""
    t = res.grid.times
    lo, hi = res.clipped_low, res.clipped_high
    lo_in, hi_in = res.clipped_low_inner, res.clipped_high_inner
    return {
        "n_slices": int(t.size),
        "max_low": float(lo.max()),
        "max_low_time": float(t[int(np.argmax(lo))]),
        "max_high": float(hi.max()),
        "max_high_time": float(t[int(np.argmax(hi))]),
        "mean_low": float(lo.mean()),
        "mean_high": float(hi.mean()),
        "slices_low_over_1pct": int(np.sum(lo > 0.01)),
        "slices_high_over_1pct": int(np.sum(hi > 0.01)),
        "inner_sd": CLIP_GATE_SD,
        "max_low_inner": float(lo_in.max()),
        "max_high_inner": float(hi_in.max()),
        "max_high_inner_time": float(t[int(np.argmax(hi_in))]),
        "slices_inner_over_1pct": int(np.sum(np.maximum(lo_in, hi_in) > 0.01)),
    }


def format_clip(res: LCCalibrationResult) -> str:
    """The clipped mass per step, summarised: the maxima, where they are, the masses inside
    ``±CLIP_GATE_SD`` sd, and the profile at a few slices."""
    c = clip_summary(res)
    t = res.grid.times
    lines = [
        f"clipped mass over {c['n_slices']} slices: low max {c['max_low']:.4f} at t={c['max_low_time']:.4f} "
        f"(mean {c['mean_low']:.4f}, {c['slices_low_over_1pct']} slices over 1 %); high max {c['max_high']:.4f} at "
        f"t={c['max_high_time']:.4f} (mean {c['mean_high']:.4f}, {c['slices_high_over_1pct']} slices over 1 %)",
        f"clipped mass inside ±{CLIP_GATE_SD:.1f} sd: low max {c['max_low_inner']:.4f}, high max "
        f"{c['max_high_inner']:.4f} at t={c['max_high_inner_time']:.4f}; {c['slices_inner_over_1pct']} slices over 1 %",
    ]
    step = max(1, (t.size - 1) // 9)
    for j in sorted({*range(0, t.size, step), 1, t.size - 1}):
        cuts = [(a, round(b, 3), round(c_, 3)) for a, b, c_ in res.clip_intervals(j)][:3]
        lines.append(
            f"   t={t[j]:.4f}: low {res.clipped_low[j]:.4f} high {res.clipped_high[j]:.4f} (inside ±{CLIP_GATE_SD:.1f} sd: "
            f"{res.clipped_low_inner[j]:.4f} / {res.clipped_high_inner[j]:.4f}) mean lambda {res.lambda_mean[j]:.4f} "
            f"trusted [{res.q_lo[j]:+.3f}, {res.q_hi[j]:+.3f}] clipped on {cuts}"
        )
    return "\n".join(lines)


def calibrate(
    spec: LocalCorrelationSpec, market: LCMarket, draws: LocalCorrelationDraws | None = None
) -> LCCalibrationResult:
    return calibrate_local_correlation(
        market.models, market.family, market.basket, market.index_surface, market.index_lv,
        spec.lc.particle, spec.sim, spec.lc, draws=draws,
    )  # fmt: skip


def model_of(
    spec: LocalCorrelationSpec, market: LCMarket, lam: LocalCorrelationFunction
) -> LocalCorrelationModel:
    return LocalCorrelationModel(market.models, market.family, lam, market.basket, spec.names)


def repricing_pillars(spec: LocalCorrelationSpec) -> list[float]:
    """The index target's slices before the horizon, and the horizon."""
    T = spec.lc.particle.horizon
    return [t for t in spec.index_surface.times if t < T - 1e-9] + [T]


def variance_split(s: Sample, inp: DowInputs) -> dict[str, Any]:
    """``E[V]`` against the study's listed ``E^Q[V] = Σ w M_i − M_B^listed`` and the split of the
    difference, ``(Σ w E[R_i²] − Σ w M_i) − (E[R̄²] − M_B^listed)`` (the reference's)."""
    b1 = inp.entry["B1"]
    eqv = float(b1["EQV"])
    sum_wm = float(np.sum(inp.weights / inp.weights.sum() * np.asarray(inp.entry["legs"]["M"])))
    m_b = sum_wm - eqv
    esq, erb2, ev = pair_mean(s.sq), pair_mean(s.rb**2), pair_mean(s.V)
    ed = pair_mean(s.D)
    return {
        "EV": ev, "EQV": eqv, "EV_over_EQV": ev[0] / eqv, "EV_over_EQV_se": ev[1] / eqv,
        "sum_wM": sum_wm, "M_B_listed": m_b, "E_sum_w_R2": esq, "E_Rbar2": erb2,
        "single_name_part": esq[0] - sum_wm, "single_name_part_se": esq[1],
        "basket_part": erb2[0] - m_b, "basket_part_se": erb2[1],
        "kappa": ed[0] / math.sqrt(ev[0]),
        "forward_at_listed_EQV": ed[0] / math.sqrt(ev[0]) * math.sqrt(eqv), "ED": ed,
    }  # fmt: skip


def log_split(tag: str, v: dict[str, Any]) -> None:
    log.info(
        "%s: E[V] %.6f (%.6f) against the listed E^Q[V] %.6f: ratio %.4f (%.4f); split (Σ w E[R_i²] - Σ w M_i) - "
        "(E[R̄²] - M_B^listed) = %+.6f (%.6f) - (%+.6f (%.6f)) = %+.6f; kappa = E[D]/sqrt(E[V]) = %.4f; the forward "
        "the listed E^Q[V] implies at that kappa: %.6f (E[D] %.6f)",
        tag, v["EV"][0], v["EV"][1], v["EQV"], v["EV_over_EQV"], v["EV_over_EQV_se"],
        v["single_name_part"], v["single_name_part_se"], v["basket_part"], v["basket_part_se"],
        v["EV"][0] - v["EQV"], v["kappa"], v["forward_at_listed_EQV"], v["ED"][0],
    )  # fmt: skip


def record(spec: LocalCorrelationSpec, **extra: Any) -> dict[str, Any]:
    """The reproducibility record of a run."""
    return {
        "git_commit": code_version(),
        "lc_code_tag": LC_CODE_TAG,
        "spec_key": lc_spec_key(spec),
        "label": spec.label,
        "particle_seed": spec.lc.particle.seed,
        "pricing_seed": spec.sim.seed,
        "n_particles": spec.lc.particle.n_particles,
        "n_paths": spec.sim.n_paths,
        "schedule": repr(spec.sim.step_schedule),
        "threads": num_threads(),
        "load": list(os.getloadavg()),
        **extra,
    }


def write_json(name: str, payload: dict[str, Any], out: str | None) -> Path:
    path = Path(out) if out else OUT / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, default=_jsonable))
    log.info("written %s", path)
    return path


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return dataclasses.asdict(x)
    raise TypeError(f"not JSON-able: {type(x).__name__}")


def log_market(
    spec: LocalCorrelationSpec, market: LCMarket, info: dict[str, Any]
) -> dict[str, Any]:
    s = market.summary()
    names_flagged = [x for x in s["flagged"] if x != "index"]
    by_leg: dict[str, int] = {}
    for g in info["dropped"]:
        by_leg[g["leg"]] = by_leg.get(g["leg"], 0) + 1
    by_rule: dict[str, int] = {}
    for g in info["dropped"]:
        by_rule[g["rule"]] = by_rule.get(g["rule"], 0) + 1
    log.info(
        "screen %s: %d expiries dropped %s (index %d; %d names touched)",
        info["screen"], len(info["dropped"]), by_rule, by_leg.get("index", 0),
        len([k for k in by_leg if k != "index"]),
    )  # fmt: skip
    horizon = spec.lc.particle.horizon
    for g in info["dropped"]:
        if 10 / 365 <= g["T"] <= horizon + 0.75:
            log.info(
                "    dropped %-5s %s (T = %.3f): %s", g["leg"], g["expiry"], g["T"], g["reason"]
            )
    log.info(
        "T = %.6f; index slices %s; %d slices in all",
        spec.lc.particle.horizon, np.round(spec.index_surface.times, 4).tolist(),
        sum(len(x.times) for x in spec.surfaces) + len(spec.index_surface.times),
    )  # fmt: skip
    log.info(
        "arbitrage flags on |k| <= 1: %d of %d names, index %s %s",
        len(names_flagged), spec.n_names, "index" in s["flagged"], s["violations"].get("index", [])[:3],
    )  # fmt: skip
    log.info(
        "Dupire floored fraction: names max %.4f, index %.4f; SVI rms (vp) median %.2f max %.2f, index %s; "
        "names extrapolated %d, index extrapolated %s",
        s["max_floored_fraction_names"], s["floored_fraction_index"], info["svi_rms_vp_median"],
        info["svi_rms_vp_max"], np.round(info["svi_rms_vp_index"], 2).tolist(),
        info["n_names_extrapolated"], info["index_extrapolated"],
    )  # fmt: skip
    log.info(
        "alignment delta (%%) by listed index expiry: %s",
        [(round(a["T"], 4), round(100 * a["delta"], 3)) for a in s["alignment"]],
    )
    return s


# ---------------------------------------------------------------------------------------------
# screen
# ---------------------------------------------------------------------------------------------


def run_screen(args: argparse.Namespace) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for date in args.dates:
        inp = load_inputs(date, args.tenor)
        spec, info = build_spec(inp, screen=args.screen, n_particles=PRODUCTION)
        market = build_lc_market(spec)
        log.info(
            "=== %s %s: T = %.6f; %d listed index expiries",
            date,
            args.tenor,
            inp.T,
            len(inp.index_smiles),
        )
        used = set(np.round(spec.index_surface.times, 6))
        prev = None
        rows = []
        last_slice = max(spec.index_surface.times)
        listed = {e.expiry for e in inp.index_smiles}
        for e in inp.index_smiles:
            maturity = float(e.T)
            if not 10 / 365 <= maturity <= last_slice + 1e-9:
                continue
            q = inp.index_quotes[e.expiry]
            atm = float(np.interp(0.0, e.k, e.vol))
            w0 = atm * atm * e.T
            falls = prev is not None and w0 < prev
            rows.append(
                {"expiry": e.expiry, "days": round(e.T * 365), "third_friday": third_friday(e.expiry, listed),
                 "used": round(e.T, 6) in used, "atm_vol": atm, "forward_ratio": e.forward / inp.index_spot,
                 "n_two_sided": q.n_two_sided, "n_strikes_1sd": q.n_strikes,
                 "n_nearest_two_sided": q.n_nearest_two_sided,
                 "median_half_spread_vp": q.median_half_spread_vp, "total_variance_falls": falls}
            )  # fmt: skip
            log.info(
                "    %s (%3dd) third Friday %-5s used %-5s ATM %6.2f%%  F/I0 %.5f  two-sided: %d of the 3 nearest strikes, "
                "%2d of %2d in ±1 sd  median half spread %5.2f vp%s",
                e.expiry, round(e.T * 365), third_friday(e.expiry, listed), round(e.T, 6) in used, 100 * atm,
                e.forward / inp.index_spot, q.n_nearest_two_sided, q.n_two_sided, q.n_strikes, q.median_half_spread_vp,
                "  <-- total variance below the previous expiry's" if falls else "",
            )  # fmt: skip
            prev = w0
        s = log_market(spec, market, info)
        n_listed = sum(len(v) for v in inp.smiles.values()) + len(inp.index_smiles)
        by_rule: dict[str, int] = {}
        for g in info["dropped"]:
            by_rule[g["rule"]] = by_rule.get(g["rule"], 0) + 1
        log.info(
            "    %d expiries listed on the 31 legs, %d dropped by the screen: %s; index expiries kept: %s",
            n_listed, len(info["dropped"]), by_rule,
            [e.expiry for e in inp.index_smiles if not any(g["leg"] == "index" and g["expiry"] == e.expiry for g in info["dropped"])],
        )  # fmt: skip
        out[date] = {"index_expiries": rows, "dropped": info["dropped"], "n_listed": n_listed, "by_rule": by_rule,
                     "alignment": s["alignment"], "flagged": s["flagged"], "record": record(spec)}  # fmt: skip
    write_json(f"screen_{args.tenor}", out, args.out)
    return out


# ---------------------------------------------------------------------------------------------
# calibrate
# ---------------------------------------------------------------------------------------------


def run_calibrate(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    inp = load_inputs(args.date, args.tenor)
    options: dict[str, Any] = {"rho_max": args.rho_max, "lambda_tail": args.lambda_tail,
                               "target_average": args.target_average}  # fmt: skip
    spec, info = build_spec(
        inp, n_particles=args.particles, n_paths=args.paths, screen=args.screen,
        schedule=SCHEDULES[args.schedule], **options,
    )  # fmt: skip
    market = build_lc_market(spec)
    T, w = spec.lc.particle.horizon, np.array(spec.weights)
    log.info(
        "=== %s %s: screen %s, schedule %s, N = %d, paths = %d, rho_max %.4g; %d threads, load %s",
        args.date, args.tenor, args.screen, spec.sim.step_schedule, args.particles, args.paths,
        args.rho_max, num_threads(), [round(x, 2) for x in os.getloadavg()],
    )  # fmt: skip
    summary = log_market(spec, market, info)
    res = calibrate(spec, market)
    tm = res.timings
    log.info(
        "calibration %.1f s (%d steps): draws %.1f, kernel %.1f, a/b %.1f, regression %.1f, rows %.1f, tables %.1f",
        res.wall_time, res.grid.n_steps, tm["draws"], tm["kernel"], tm["ab"], tm["regression"], tm["rows"], tm["tables"],
    )  # fmt: skip
    rho0 = float(market.family.equicorrelation(res.lam.values[0, 0]))
    log.info(
        "lambda(t0) %.4f (rho %.4f); cloud mean of lambda at T %.4f",
        res.lam.values[0, 0],
        rho0,
        res.lambda_mean[-1],
    )
    clip = clip_summary(res)
    log.info("%s", format_clip(res))
    visited = visited_arbitrage(spec, res)
    log.info(
        "arbitrage on the range the cloud visits [min, max]: %d of %d names flagged, index %s; "
        "inside ±%.0f sd of the cloud: %d names, index %s",
        len([x for x in visited["flagged"] if x != "index"]), spec.n_names, "index" in visited["flagged"],
        visited["central_sd"], len([x for x in visited["flagged_central"] if x != "index"]),
        "index" in visited["flagged_central"],
    )  # fmt: skip
    for label in visited["flagged_central"]:
        log.info(
            "    inside ±%.0f sd, %s: %s",
            visited["central_sd"],
            label,
            visited["violations_central"][label][:4],
        )
    inner = visited_arbitrage(spec, res, central_sd=2.0)
    log.info(
        "    inside ±2 sd of the cloud: %d names, index %s %s",
        len([x for x in inner["flagged_central"] if x != "index"]), "index" in inner["flagged_central"],
        {k: v[:2] for k, v in inner["violations_central"].items()},
    )  # fmt: skip
    out: dict[str, Any] = {
        "date": args.date, "tenor": args.tenor, "T": T, "screen": info["screen"], "dropped": info["dropped"],
        "market": {k: v for k, v in summary.items() if k != "dupire"}, "svi_rms_vp_median": info["svi_rms_vp_median"],
        "svi_rms_vp_max": info["svi_rms_vp_max"], "calibration_seconds": res.wall_time, "timings": tm,
        "n_steps": res.grid.n_steps, "clip": clip, "times": res.grid.times, "clipped_low": res.clipped_low,
        "clipped_high": res.clipped_high, "clipped_low_inner": res.clipped_low_inner,
        "clipped_high_inner": res.clipped_high_inner, "lambda_mean": res.lambda_mean, "lambda_t0": res.lam.values[0, 0],
        "arbitrage_visited": {k: v for k, v in visited.items() if k != "ranges"},
        "arbitrage_visited_2sd": {k: inner[k] for k in ("flagged_central", "violations_central")},
    }  # fmt: skip
    if not args.no_price:
        model = model_of(spec, market, res.lam)
        pillars = repricing_pillars(spec)
        rep = reprice_index_smile(model, market.index_surface, spec.sim, maturities=pillars)
        log.info("%s", rep.summary())
        log.info(
            "index repricing passes the smile gate: %s (%d cells over it); %.1f s",
            rep.passes(),
            len(rep.violations()),
            rep.wall_time,
        )
        s_lc = sample(model, spec.sim, w, pillars)
        log.info(
            "one pricing pass under LC: %.1f s for %d paths; calibration + one pricing = %.1f s",
            s_lc.seconds,
            args.paths,
            res.wall_time + s_lc.seconds,
        )
        split = variance_split(s_lc, inp)
        log_split("LC", split)
        out.update(
            report=rep.to_dict(),
            pricing_seconds=s_lc.seconds,
            ED_lc=pair_mean(s_lc.D),
            variance_lc=split,
        )
        if not args.no_companion:
            tc = time.perf_counter()
            lam_c = calibrate_constant_lambda(model, market.index_surface, T, spec.sim)
            cc = model.with_lambda(
                LocalCorrelationFunction.constant(lam_c, res.lam.times, res.lam.k_grid)
            )
            s_cc = sample(cc, spec.sim, w, pillars)
            rho_cc = float(market.family.equicorrelation(lam_c))
            ed_lc, ed_cc, ratio = pair_mean(s_lc.D), pair_mean(s_cc.D), pair_ratio(s_lc.D, s_cc.D)
            log.info(
                "constant-correlation companion: lambda_c %.6f, rho_CC %.6f (%.1f s)",
                lam_c,
                rho_cc,
                time.perf_counter() - tc,
            )
            log.info(
                "E_CC[D] %.6f (%.6f); E_LC[D] %.6f (%.6f); LC/CC %.5f (%.5f)",
                *ed_cc,
                *ed_lc,
                *ratio,
            )
            split_cc = variance_split(s_cc, inp)
            log_split("CC", split_cc)
            base = inp.entry["B1"]
            log.info(
                "the study (copula at rho_cop %.4f): P_D %.6f, E[V] %.6f",
                inp.entry["rho_cop"],
                base["P_D"],
                base["EV"],
            )
            rows = []
            for mlt, K in zip((0.5, 0.75, 1.0, 1.25, 1.5, 2.0), base["strikes"], strict=True):
                a, b = np.maximum(s_lc.D - K, 0.0), np.maximum(s_cc.D - K, 0.0)
                r = pair_ratio(a, b) if b.mean() > 0 else (float("nan"), float("nan"))
                rows.append({"multiple": mlt, "K": float(K), "C_lc": pair_mean(a)[0], "C_lc_se": pair_mean(a)[1],
                             "C_cc": pair_mean(b)[0], "C_cc_se": pair_mean(b)[1], "ratio": r[0], "ratio_se": r[1]})  # fmt: skip
            log.info(
                "calls at the study's strikes:\n%s",
                pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6g}"),
            )
            out.update(lambda_c=lam_c, rho_cc=rho_cc, ED_cc=ed_cc, ratio=ratio, calls=rows, variance_cc=split_cc,
                       study={"P_D": float(base["P_D"]), "EV": float(base["EV"]), "EQV": float(base["EQV"]),
                              "rho_cop": float(inp.entry["rho_cop"])})  # fmt: skip
    out["record"] = record(spec, total_seconds=time.perf_counter() - t0)
    log.info("total %.1f s", time.perf_counter() - t0)
    tag = f"calibrate_{args.date}_{args.tenor}_{args.screen}_{args.schedule}_rho{args.rho_max:g}_{args.particles}"
    write_json(tag, out, args.out)
    return out


# ---------------------------------------------------------------------------------------------
# dt
# ---------------------------------------------------------------------------------------------


def dt_check(
    spec: LocalCorrelationSpec, market: LCMarket, pillars_wanted: Sequence[float]
) -> dict[str, Any]:
    """Calibrate and price on the schedule of ``spec`` and on the schedule with every step
    halved, on common random numbers (the coarse runs use the Brownian-consistent coarsening of
    the fine runs' draws, in the calibration and in pricing): the Palladium forward and calls
    and the basket smile at both.  The pillars are moved to nodes of the coarse grid."""
    T, w = spec.lc.particle.horizon, np.array(spec.weights)
    N, P = spec.lc.particle.n_particles, spec.sim.n_paths
    coarse = spec.sim
    fine = dataclasses.replace(coarse, dt_max=coarse.step_schedule.refined(2))
    n_fine = TimeGrid.build([T], fine.dt_max).n_steps
    cal_fine = LocalCorrelationDraws(spec.lc.particle.seed, N, n_fine, market.family)
    res = {
        "dt/2": calibrate(dataclasses.replace(spec, sim=fine), market, cal_fine),
        "dt": calibrate(spec, market, cal_fine.coarsened(2)),
    }
    assert res["dt/2"].grid.n_steps == 2 * res["dt"].grid.n_steps
    nodes = np.asarray(res["dt"].grid.times)
    pillars = sorted({float(nodes[int(np.argmin(np.abs(nodes - x)))]) for x in pillars_wanted})
    models = {k: model_of(spec, market, r.lam) for k, r in res.items()}
    sims = {"dt": coarse, "dt/2": fine}
    grid_f = TimeGrid.build(pillars, fine.dt_max, calibration_grid=models["dt/2"].required_times())
    price_fine = models["dt/2"].draws_for(grid_f, coarse.seed, P)
    draws = {"dt/2": price_fine, "dt": price_fine.coarsened(2)}
    s = {k: sample(models[k], sims[k], w, pillars, draws[k]) for k in ("dt", "dt/2")}
    base = pair_mean(s["dt"].D)[0]
    rows = []
    for mlt in (0.0, 0.75, 1.0, 1.25, 1.5):
        a, b = np.maximum(s["dt/2"].D - mlt * base, 0.0), np.maximum(s["dt"].D - mlt * base, 0.0)
        diff, se = pair_mean(a - b)
        rows.append({"quantity": "E[D]" if mlt == 0 else f"call {mlt:.2f}", "dt": pair_mean(b)[0],
                     "dt/2": pair_mean(a)[0], "diff": diff, "diff_se": se,
                     "rel_%": 100 * diff / pair_mean(b)[0], "rel_se_%": 100 * se / pair_mean(b)[0]})  # fmt: skip
    smile = smile_difference(s["dt/2"].levels, s["dt"].levels, pillars, market.index_surface)
    reports = {
        k: reprice_index_smile(models[k], market.index_surface, sims[k], maturities=pillars)
        for k in ("dt", "dt/2")
    }
    return {
        "prices": pd.DataFrame(rows),
        "smile": smile,
        "pillars": pillars,
        "results": res,
        "reports": reports,
    }


def format_dt(label: str, out: dict[str, Any]) -> str:
    res = out["results"]
    lines = [
        f"{label} calibrated at {k}: {res[k].grid.n_steps} steps, {res[k].wall_time:.0f} s; clipped mass max low "
        f"{res[k].clipped_low.max():.4f}, high {res[k].clipped_high.max():.4f} (inside ±{CLIP_GATE_SD:.1f} sd: "
        f"{res[k].max_clipped_mass_inner:.4f})"
        for k in ("dt", "dt/2")
    ]
    lines.append(f"{label}: prices at dt and dt/2 (common random numbers; diff = dt/2 minus dt)")
    lines.append(out["prices"].to_string(index=False, float_format=lambda x: f"{x:.6g}"))
    sm = out["smile"]
    cells = sm.assign(
        cell=[f"{d:+.3f} ({e:.3f})" for d, e in zip(sm["diff_vp"], sm["se_vp"], strict=True)]
    )
    lines.append(f"{label}: basket implied vol, dt/2 minus dt, vol points (paired se)")
    lines.append(cells.pivot(index="T", columns="sd", values="cell").to_string())
    for k, report in out["reports"].items():
        atm = report.table[report.table["sd"] == 0.0]["error_vp"]
        lines.append(
            f"{label} against the target at {k}: max |error| inside ±1.5 sd {report.max_abs_error(1.5):.3f} vp, "
            f"inside ±2.5 sd {report.max_abs_error(2.5):.3f} vp; ATM by pillar "
            + " ".join(f"{x:+.3f}" for x in atm)
        )
    return "\n".join(lines)


def run_dt(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    inp = load_inputs(args.date, args.tenor)
    spec, _ = build_spec(
        inp, n_particles=args.particles, n_paths=args.paths, screen=args.screen,
        schedule=SCHEDULES[args.schedule], target_average=args.target_average,
    )  # fmt: skip
    market = build_lc_market(spec)
    T = spec.lc.particle.horizon
    wanted = (
        [T / 3, 2 * T / 3, T]
        if T <= 0.5
        else [x for x in (0.25, 0.5, 1.0, 1.5) if x < T - 0.1] + [T]
    )
    label = f"{args.date} {args.tenor} ({args.target_average} target, {spec.sim.step_schedule})"
    log.info("=== dt check %s: N = %d, paths = %d", label, args.particles, args.paths)
    out = dt_check(spec, market, wanted)
    log.info("%s", format_dt(label, out))
    payload = {
        "prices": out["prices"].to_dict("records"), "smile": out["smile"].to_dict("records"), "pillars": out["pillars"],
        "reports": {k: r.to_dict() for k, r in out["reports"].items()},
        "clip": {k: clip_summary(r) for k, r in out["results"].items()},
        "record": record(spec, total_seconds=time.perf_counter() - t0),
    }  # fmt: skip
    write_json(
        f"dt_{args.date}_{args.tenor}_{args.schedule}_{args.target_average}_{args.particles}",
        payload,
        args.out,
    )
    return payload


# ---------------------------------------------------------------------------------------------
# tail rule
# ---------------------------------------------------------------------------------------------


def run_tail(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    inp = load_inputs(args.date, args.tenor)
    got: dict[str, Any] = {}
    for tail in ("regressions", "flat"):
        spec, _ = build_spec(inp, n_particles=args.particles, n_paths=args.paths, screen=args.screen,
                             schedule=SCHEDULES[args.schedule], lambda_tail=tail)  # fmt: skip
        market = build_lc_market(spec)
        res = calibrate(spec, market)
        model = model_of(spec, market, res.lam)
        pillars = repricing_pillars(spec)
        got[tail] = (spec, market, res, sample(model, spec.sim, np.array(spec.weights), pillars),
                     reprice_index_smile(model, market.index_surface, spec.sim, maturities=pillars))  # fmt: skip
    a, b = got["regressions"][3], got["flat"][3]
    base = pair_mean(a.D)[0]
    rows = []
    for mlt in (0.0, 0.75, 1.0, 1.25, 1.5, 2.0):
        x, y = np.maximum(a.D - mlt * base, 0.0), np.maximum(b.D - mlt * base, 0.0)
        d, se = pair_mean(y - x)
        rows.append({"quantity": "E[D]" if mlt == 0 else f"call {mlt:.2f}", "regressions": pair_mean(x)[0],
                     "se": pair_mean(x)[1], "flat": pair_mean(y)[0], "flat_minus_regressions": d, "paired_se": se,
                     "rel_%": 100 * d / pair_mean(x)[0]})  # fmt: skip
    log.info("=== tail rule on %s %s: N = %d, paths = %d\n%s", args.date, args.tenor, args.particles, args.paths,
             pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6g}"))  # fmt: skip
    wings = {}
    for tail, (_, _, res, _, rep) in got.items():
        t = rep.table
        far = t[t["sd"].abs() >= 2.0]
        wings[tail] = far[["T", "sd", "error_vp", "stderr_vp"]].to_dict("records")
        log.info("%s: clipped mass max low %.4f high %.4f; index error at |sd| >= 2 (vp):\n%s", tail,
                 res.clipped_low.max(), res.clipped_high.max(),
                 far.pivot(index="T", columns="sd", values="error_vp").to_string(float_format=lambda x: f"{x:+.3f}"))  # fmt: skip
    payload = {
        "prices": rows,
        "wings": wings,
        "record": record(got["regressions"][0], total_seconds=time.perf_counter() - t0),
    }
    write_json(f"tail_{args.date}_{args.tenor}_{args.particles}", payload, args.out)
    return payload


# ---------------------------------------------------------------------------------------------
# the downside wing
# ---------------------------------------------------------------------------------------------


def wing_fit_errors(
    inp: DowInputs, spec: LocalCorrelationSpec, lo: float = -2.5, hi: float = -1.5
) -> pd.DataFrame:
    """(a) Per leg and fitted slice: the SVI vol minus the quoted vol at the listed strikes whose
    standardised log-moneyness ``k/(σ_ATM·√T)`` lies in ``[lo, hi]`` (mean, root mean square and
    count), and the same within ±1 sd for scale.  Vol points."""
    rows = []
    legs = [(n, inp.smiles[n], s) for n, s in zip(spec.names, spec.surfaces, strict=True)]
    legs.append(("index", inp.index_smiles, spec.index_surface))
    for name, expiries, cfg in legs:
        by_t = {round(float(e.T), 9): e for e in expiries}
        for t, params in zip(cfg.times, cfg.params, strict=True):
            e = by_t[round(float(t), 9)]
            atm = float(np.interp(0.0, e.k, e.vol))
            z = e.k / (atm * math.sqrt(e.T))
            fit = np.sqrt(svi_total_variance(np.array(params), e.k) / e.T)
            err = 100.0 * (fit - e.vol)
            wing, core = (z >= lo) & (z <= hi), np.abs(z) <= 1.0
            rows.append({
                "leg": name, "T": float(t), "n_wing": int(wing.sum()),
                "wing_mean_vp": float(err[wing].mean()) if wing.any() else float("nan"),
                "wing_rms_vp": float(np.sqrt((err[wing] ** 2).mean())) if wing.any() else float("nan"),
                "core_mean_vp": float(err[core].mean()) if core.any() else float("nan"),
                "core_rms_vp": float(np.sqrt((err[core] ** 2).mean())) if core.any() else float("nan"),
            })  # fmt: skip
    return pd.DataFrame(rows)


_Z = np.linspace(-8.0, 8.0, 2**14 + 1)
_PDF = np.exp(-0.5 * _Z * _Z) / math.sqrt(2.0 * math.pi)


def quantile_table(surface: Any, T: float) -> np.ndarray:
    """``q(Φ(z))`` on the latent grid: the quantile function of ``S_T/F(T)`` under the surface's
    marginal at ``T``.  ``P(S_T/F ≤ K) = ∂Put/∂K`` on a fine strike grid over ±12 at-the-money
    standard deviations, made monotone outwards from the money (a fitted wing can carry a
    butterfly arbitrage far out: the distribution function is held flat there instead of being
    allowed to come back), and scaled to mean 1."""
    sd = math.sqrt(float(surface.total_variance(0.0, T)))
    k = np.linspace(-12.0 * sd, 12.0 * sd, 48_001)
    K = np.exp(k)
    vol = np.sqrt(np.asarray(surface.total_variance(k, T), dtype=float) / T)
    cdf = np.gradient(black_price(1.0, K, T, vol, -1.0), K)
    mid = k.size // 2
    cdf[: mid + 1] = np.minimum.accumulate(cdf[: mid + 1][::-1])[::-1]
    cdf[mid:] = np.maximum.accumulate(cdf[mid:])
    q = np.interp(ndtr(_Z), np.clip(cdf, 0.0, 1.0), K)
    mean = float(np.trapezoid(q * _PDF, _Z))
    if abs(mean - 1.0) > 0.005:
        log.warning("quantile table at T = %.4f: mean %.5f before scaling", T, mean)
    return np.asarray(q / mean)


def comonotonic_bound(
    inp: DowInputs, spec: LocalCorrelationSpec, market: LCMarket, res: LCCalibrationResult
) -> pd.DataFrame:
    """(b) At each index slice up to the first one beyond the horizon and at each listed index
    strike between −3 and −0.5 sd: the quoted vol, the index SVI vol, and the vol of the
    comonotonic upper bound on the put ``E[(K − Σ w_i q_i(U))⁺]`` built from the names' own
    marginals (the basket of the model: forward performances with the index weights).  A quote
    above the bound cannot be reached by any dependence between the names.  ``capped``: the
    calibrated ``λ*`` exceeds the cap at that strike and time."""
    T, w = spec.lc.particle.horizon, np.array(spec.weights)
    times = res.grid.times
    by_t = {round(float(e.T), 9): e for e in inp.index_smiles}
    slices = [t for t in spec.index_surface.times if t <= T + 1e-9]
    slices += [t for t in spec.index_surface.times if t > T + 1e-9][:1]
    rows = []
    for t in slices:
        e = by_t[round(float(t), 9)]
        qb = sum(wi * quantile_table(s, t) for wi, s in zip(w, market.surfaces, strict=True))
        atm = float(np.interp(0.0, e.k, e.vol))
        z = e.k / (atm * math.sqrt(t))
        j = int(np.argmin(np.abs(times - min(t, T))))
        capped = [(a, b) for label, a, b in res.clip_intervals(j) if label == "high"]
        for k, zq, vq in zip(e.k, z, e.vol, strict=True):
            if not -3.0 <= zq <= -0.5:
                continue
            strike = math.exp(k)
            bound = float(np.trapezoid(np.maximum(strike - qb, 0.0) * _PDF, _Z))
            v_bound = float(implied_vol(bound, 1.0, strike, t, -1.0))
            rows.append({
                "T": float(t), "k": float(k), "sd": float(zq), "quote_vol": float(vq),
                "svi_vol": float(market.index_surface.implied_vol_k(k, t)), "bound_vol": v_bound,
                "quote_minus_bound_vp": 100.0 * (float(vq) - v_bound),
                "capped": bool(any(a <= k <= b for a, b in capped)),
            })  # fmt: skip
    return pd.DataFrame(rows)


def run_wing(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    inp = load_inputs(args.date, args.tenor)
    runs: dict[float, Any] = {}
    for rho_max in (0.98, 0.995):
        spec, info = build_spec(inp, n_particles=args.particles, n_paths=args.paths, screen=args.screen,
                                schedule=SCHEDULES[args.schedule], rho_max=rho_max)  # fmt: skip
        market = build_lc_market(spec)
        res = calibrate(spec, market)
        model = model_of(spec, market, res.lam)
        pillars = repricing_pillars(spec)
        rep = reprice_index_smile(model, market.index_surface, spec.sim, maturities=pillars)
        runs[rho_max] = (
            spec,
            market,
            res,
            rep,
            sample(model, spec.sim, np.array(spec.weights), pillars),
            info,
        )
    spec, market, res, _, _, info = runs[0.98]
    log.info(
        "=== downside wing on %s %s: N = %d, paths = %d, screen %s",
        args.date,
        args.tenor,
        args.particles,
        args.paths,
        args.screen,
    )
    # (a)
    fit = wing_fit_errors(inp, spec)
    names = fit[fit["leg"] != "index"]
    wmap = dict(zip(spec.names, spec.weights, strict=True))
    near = names[names["T"] <= max(t for t in spec.index_surface.times if t <= inp.T + 1e-9) + 0.5]
    has = near[near["n_wing"] > 0]
    weighted = float(np.sum([wmap[n] * m for n, m in has.groupby("leg")["wing_mean_vp"].mean().items()]) /
                     np.sum([wmap[n] for n in has["leg"].unique()]))  # fmt: skip
    log.info(
        "(a) SVI fit minus quotes between -2.5 and -1.5 sd, names: %d quotes on %d slices of %d names; mean %+.3f vp, "
        "rms %.3f vp; index-weighted mean of the names' means %+.3f vp; inside ±1 sd: mean %+.3f, rms %.3f",
        int(has["n_wing"].sum()), len(has), has["leg"].nunique(),
        float(np.average(has["wing_mean_vp"], weights=has["n_wing"])),
        float(np.sqrt(np.average(has["wing_rms_vp"] ** 2, weights=has["n_wing"]))), weighted,
        float(near["core_mean_vp"].mean()), float(np.sqrt((near["core_rms_vp"] ** 2).mean())),
    )  # fmt: skip
    per_name = has.groupby("leg").apply(
        lambda g: pd.Series({"n": int(g["n_wing"].sum()), "mean_vp": float(np.average(g["wing_mean_vp"], weights=g["n_wing"])),
                             "rms_vp": float(np.sqrt(np.average(g["wing_rms_vp"] ** 2, weights=g["n_wing"])))}),
        include_groups=False,
    )  # fmt: skip
    log.info(
        "    by name (n, mean, rms in vp):\n%s",
        per_name.T.to_string(float_format=lambda x: f"{x:.2f}"),
    )
    log.info(
        "    the index's own slices:\n%s",
        fit[fit["leg"] == "index"].to_string(index=False, float_format=lambda x: f"{x:.3f}"),
    )
    # (b)
    bound = comonotonic_bound(inp, spec, market, res)
    over = bound[bound["quote_minus_bound_vp"] > 0]
    log.info(
        "(b) comonotonic bound on the index puts: %d listed strikes between -3 and -0.5 sd on %d slices; %d quotes above "
        "the bound (%d of the %d capped strikes); smallest margin (bound - quote) %.2f vp",
        len(bound), bound["T"].nunique(), len(over), int(over["capped"].sum()), int(bound["capped"].sum()),
        float(-bound["quote_minus_bound_vp"].max()),
    )  # fmt: skip
    log.info("\n%s", bound.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    # (c)
    out_c = {}
    for rho_max, (sp, _, r, rep, s, _) in runs.items():
        c = clip_summary(r)
        split = variance_split(s, inp)
        ed = pair_mean(s.D)
        t = rep.table
        wing = t[t["sd"] <= -1.0]
        log.info(
            "(c) rho_max %.3f: clipped high max %.4f (inside ±%.1f sd %.4f; %d slices over 1 %%), low max %.4f; "
            "E[D] %.6f (%.6f); index error on the downside (vp):\n%s",
            rho_max, c["max_high"], CLIP_GATE_SD, c["max_high_inner"], c["slices_high_over_1pct"], c["max_low"], *ed,
            wing.pivot(index="T", columns="sd", values="error_vp").to_string(float_format=lambda x: f"{x:+.3f}"),
        )  # fmt: skip
        log_split(f"    rho_max {rho_max:.3f}", split)
        out_c[str(rho_max)] = {
            "clip": c,
            "ED": ed,
            "variance": split,
            "report": rep.to_dict(),
            "record": record(sp),
        }
    a, b = runs[0.98][4], runs[0.995][4]
    d = pair_mean(b.D - a.D)
    log.info(
        "    E[D](0.995) - E[D](0.98) = %+.3e (%.1e) = %+.4f %% (the same pricing paths)",
        d[0],
        d[1],
        100 * d[0] / pair_mean(a.D)[0],
    )
    payload = {
        "date": args.date, "tenor": args.tenor, "fit_errors": fit.to_dict("records"), "fit_by_name": per_name.reset_index().to_dict("records"),
        "bound": bound.to_dict("records"), "sensitivity": out_c, "ED_difference": d, "dropped": info["dropped"],
        "record": record(spec, total_seconds=time.perf_counter() - t0),
    }  # fmt: skip
    write_json(f"wing_{args.date}_{args.tenor}_{args.particles}", payload, args.out)
    return payload


# ---------------------------------------------------------------------------------------------
# the single-name strips (S8's diagnostic)
# ---------------------------------------------------------------------------------------------


class StudySmile:
    """The study's smile of a name at the horizon as a surface the strip can read: vols linear
    between the listed strikes and flat beyond them, total variance linear in time between the
    two bracketing expiries (``disp_smile.smile_at``); the forward is the study's."""

    def __init__(self, smile: Any) -> None:
        self.smile = smile
        self.forward_curve = ForwardCurve.from_forwards(
            1.0, [smile.T], [smile.forward / smile.spot], ZERO_CURVE
        )

    def atm_vol(self, T: float) -> float:
        return float(self.smile.vol(0.0))

    def total_variance(self, k: Any, T: float) -> np.ndarray:
        return np.asarray(self.smile.vol(np.asarray(k, dtype=float)) ** 2 * T)


REGIONS = (
    "below the listed strikes",
    "listed, below the forward",
    "listed, above the forward",
    "above the listed strikes",
)
#: The same four, as the options that carry them (the columns of the tables by name): puts and
#: calls beyond the last listed strike, and inside the listed range.
REGION_SHORT = ("puts beyond", "puts listed", "calls listed", "calls beyond")


def strip_regions(inp: DowInputs, market: LCMarket) -> pd.DataFrame:
    """Per name at the horizon: ``E[R_i²]`` from the model's SVI surface and from the study's
    smile (all the study's listed expiries), each split into ``(f − 1)²`` and the strip by
    region of the strike axis — below the listed strikes, listed below the forward, listed
    above it, above the listed strikes (the listed range is the one the two bracketing expiries
    share) — and the entry's own ``M_i``.  The cuts are log-moneyness levels ``k_lo < 0 < k_hi``
    on each smile's own forward (``f_study``, ``f_svi``: the two forwards over the spot)."""
    T = inp.T
    entry_m = np.asarray(inp.entry["legs"]["M"], dtype=float)
    rows = []
    for i, name in enumerate(inp.names):
        smile = ds.smile_at(inp.smiles[name], inp.spots[i], T)
        k_lo = max(float(smile.lo.k.min()), float(smile.hi.k.min()))
        k_hi = min(float(smile.lo.k.max()), float(smile.hi.k.max()))
        cuts = [k_lo, 0.0, k_hi]
        study_total, study_parts = strip_second_moment(StudySmile(smile), T, splits=cuts)  # type: ignore[arg-type,misc]
        svi_total, svi_parts = strip_second_moment(market.surfaces[i], T, splits=cuts)  # type: ignore[misc]
        curve = market.surfaces[i].forward_curve
        row: dict[str, Any] = {"name": name, "weight": float(inp.weights[i] / inp.weights.sum()),
                               "k_lo": k_lo, "k_hi": k_hi, "M_entry": float(entry_m[i]),
                               "M_study": float(study_total), "M_svi": float(svi_total),
                               "sd": float(smile.vol(0.0)) * math.sqrt(T),
                               "f_study": float(smile.forward / smile.spot),
                               "f_svi": float(curve.forward(T)) / float(curve.spot)}  # fmt: skip
        for j, label in enumerate(REGIONS):
            row[f"study: {label}"] = float(study_parts[j])
            row[f"svi: {label}"] = float(svi_parts[j])
        rows.append(row)
    return pd.DataFrame(rows)


FORWARD_TERM = "forward term"


def model_regions(
    R: np.ndarray, weights: np.ndarray, forwards: np.ndarray, k_lo: np.ndarray, k_hi: np.ndarray
) -> tuple[pd.DataFrame, dict[str, tuple[float, float]], float]:
    """The model's Monte Carlo ``E[R_i²]`` split by the regions of :func:`strip_regions`, exactly
    per path.  With ``X = S_T/S_0 = 1 + R`` and any level ``f``,

        (X − 1)² = (f − 1)² + 2(f − 1)(X − f) + ((f − X)⁺)² + ((X − f)⁺)²,

    and, since ``d/dK ((K − X)⁺)² = 2(K − X)⁺`` and ``d/dK ((X − K)⁺)² = −2(X − K)⁺``,

        2·∫_{K1}^{K2} (K − X)⁺ dK = ((K2 − X)⁺)² − ((K1 − X)⁺)²        (the put side, K2 ≤ f),
        2·∫_{K1}^{K2} (X − K)⁺ dK = ((X − K1)⁺)² − ((X − K2)⁺)²        (the call side, K1 ≥ f).

    Taking expectations with ``f`` the forward over the spot gives the strip of
    ``volsto.multi.analytics.strip_second_moment`` (Carr–Madan: ``E[R²] = (f − 1)² +
    2∫_0^f P(K) dK + 2∫_f^∞ C(K) dK``), so the part of ``E[R²]`` carried by the strikes
    ``[K1, K2]`` is the mean of the right-hand sides above.  With ``K_lo = f·e^{k_lo}`` and
    ``K_hi = f·e^{k_hi}`` (the cuts of the SVI strip, on the model's forward) the four regions
    are, per path,

        below the listed strikes     ((K_lo − X)⁺)²
        listed, below the forward    ((f − X)⁺)² − ((K_lo − X)⁺)²
        listed, above the forward    ((X − f)⁺)² − ((X − K_hi)⁺)²
        above the listed strikes     ((X − K_hi)⁺)²

    and what is left of ``R²`` is ``(f − 1)²`` and the forward term ``2(f − 1)(X − f)``, whose
    expectation is zero under the model and whose sample mean ``2(f − 1)(X̄ − f)`` is reported
    with the rest: the four regions, ``(f − 1)²`` and the forward term sum to the sample's
    ``E[R²]`` path by path.

    Returns ``(per name, weighted over the names, gap)``: per name the mean and the standard
    error (antithetic pair means) of ``R²``, of each region and of the forward term; the same
    for ``Σ_i w_i ·`` (the weighted sum is formed per path, then averaged over pairs); and the
    largest ``|regions + (f − 1)² + forward term − E[R²]|`` over the names, which is rounding
    (``run_strips`` logs it and refuses a run where it is above 1e-12)."""
    n, m = R.shape
    labels = [*REGIONS, FORWARD_TERM]
    acc = np.zeros((n, len(labels)))
    rows, gap = [], 0.0
    for i in range(m):
        f, x = float(forwards[i]), 1.0 + R[:, i]
        lo, hi = f * math.exp(float(k_lo[i])), f * math.exp(float(k_hi[i]))
        below, above = np.maximum(lo - x, 0.0) ** 2, np.maximum(x - hi, 0.0) ** 2
        parts = [
            below,
            np.maximum(f - x, 0.0) ** 2 - below,
            np.maximum(x - f, 0.0) ** 2 - above,
            above,
            2.0 * (f - 1.0) * (x - f),
        ]
        total = pair_mean(R[:, i] ** 2)
        row: dict[str, Any] = {"M_mc": total[0], "M_mc_se": total[1], "f_mc": float(x.mean()), "x_max": float(x.max())}  # fmt: skip
        s = (f - 1.0) ** 2
        for j, label in enumerate(labels):
            row[f"mc: {label}"], row[f"mc_se: {label}"] = pair_mean(parts[j])
            s += row[f"mc: {label}"]
            acc[:, j] += weights[i] * parts[j]
        gap = max(gap, abs(s - total[0]))
        rows.append(row)
    return (
        pd.DataFrame(rows),
        {label: pair_mean(acc[:, j]) for j, label in enumerate(labels)},
        gap,
    )


WING_SHARES = (1, 10, 100, 1000)


def call_wing_paths(
    R: np.ndarray,
    weights: np.ndarray,
    forwards: np.ndarray,
    k_hi: np.ndarray,
    names: Sequence[str],
    n_top: int = 8,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """How few paths carry the region above the listed strikes.  Per path and name the cell
    ``c_i = w_i·((X_i − K_hi,i)⁺)²`` of :func:`model_regions` (the mean over the paths of
    ``Σ_i c_i`` is the region's part of ``Σ w E[R_i²]``).  Returns the number of paths with a
    name above its last listed strike and the share of the region's sum carried by its 1, 10,
    100 and 1000 largest paths (``WING_SHARES``); and the ``n_top`` largest cells: the name, the
    path, ``X = S_T/S_0``, its log-moneyness on the forward ``ln(X/f)`` and the cell's part of
    the region's mean, ``c_i/n``.  A second moment whose sample mean a handful of paths carry
    has a standard error that is itself not estimated (it is the same handful)."""
    n = R.shape[0]
    f = np.asarray(forwards, dtype=float)
    cells = np.asarray(weights) * np.maximum(1.0 + R - f * np.exp(np.asarray(k_hi)), 0.0) ** 2
    per_path = cells.sum(axis=1)
    ranked = np.sort(per_path)[::-1]
    total = float(ranked.sum())
    summary: dict[str, Any] = {"n_paths_above": int(np.count_nonzero(per_path)), "region_mean": total / n}  # fmt: skip
    for k in WING_SHARES:
        summary[f"share_top_{k}"] = float(ranked[:k].sum() / total) if total > 0 else float("nan")
    flat = np.argsort(cells, axis=None)[::-1][:n_top]
    rows = []
    for path, i in zip(*np.unravel_index(flat, cells.shape), strict=True):
        x = 1.0 + float(R[path, i])
        rows.append({"name": names[i], "path": int(path), "X": x, "ln(X/f)": math.log(x / float(f[i])),
                     "part of the region": float(cells[path, i]) / n,
                     "share": float(cells[path, i]) / total if total > 0 else float("nan")})  # fmt: skip
    return summary, pd.DataFrame(rows)


def _cell(value: float, se: float | None = None) -> str:
    return f"{value:+.6f}" if se is None else f"{value:+.6f} ({se:.6f})"


def run_strips(args: argparse.Namespace) -> dict[str, Any]:
    """The ``strips`` command.  First the SVI surfaces against the study's strips (the second
    round's table, unchanged); then the model's Monte Carlo ``Σ w E[R_i²]`` in total and — the
    owner's decision 3 of 2026-10-09 — by region of the strike axis (:func:`model_regions`),
    against both strips, with the names that carry the difference."""
    t0 = time.perf_counter()
    inp = load_inputs(args.date, args.tenor)
    spec, info = build_spec(inp, n_particles=args.particles, n_paths=args.paths, screen=args.screen, schedule=SCHEDULES[args.schedule])  # fmt: skip
    market = build_lc_market(spec)
    table = strip_regions(inp, market)
    w = table["weight"].to_numpy()
    log.info("=== single-name strips on %s %s (T = %.6f)", args.date, args.tenor, inp.T)
    by_rule: dict[str, int] = {}
    for g in info["dropped"]:
        by_rule[g["rule"]] = by_rule.get(g["rule"], 0) + 1
    log.info(
        "screen %s %s: expiries dropped %s; by the calendar repair: %s; names kept unscreened: %s",
        args.screen, info["screen"], by_rule,
        [(g["leg"], g["expiry"]) for g in info["dropped"] if g["rule"] == "calendar"], info["names_unscreened"],
    )  # fmt: skip
    check = float(np.max(np.abs(table["M_study"] / table["M_entry"] - 1.0)))
    log.info(
        "Σ w M_i: the entry %.6f; the study's strip rebuilt here %.6f (largest relative difference by name %.1e); "
        "the SVI surfaces %.6f; SVI - study = %+.6f (%+.2f %%)",
        float(w @ table["M_entry"]), float(w @ table["M_study"]), check, float(w @ table["M_svi"]),
        float(w @ (table["M_svi"] - table["M_study"])), 100 * float(w @ (table["M_svi"] - table["M_study"])) / float(w @ table["M_study"]),
    )  # fmt: skip
    by_region = {}
    for label in REGIONS:
        d = table[f"svi: {label}"] - table[f"study: {label}"]
        by_region[label] = float(w @ d)
        log.info("    %-28s Σ w (SVI - study) = %+.6f   (study %.6f, SVI %.6f)", label, by_region[label],
                 float(w @ table[f"study: {label}"]), float(w @ table[f"svi: {label}"]))  # fmt: skip
    rest = float(w @ (table["M_svi"] - table["M_study"])) - sum(by_region.values())
    log.info("    %-28s %+.6f   (the (f - 1)² terms and the quadrature)", "the rest", rest)
    table["diff"] = table["M_svi"] - table["M_study"]
    table["weighted_diff"] = table["weight"] * table["diff"]
    top = table.reindex(table["weighted_diff"].abs().sort_values(ascending=False).index).head(8)
    show = top[["name", "weight", "sd", "k_lo", "k_hi", "M_study", "M_svi", "weighted_diff"]].copy()
    show["k_lo_sd"], show["k_hi_sd"] = show["k_lo"] / show["sd"], show["k_hi"] / show["sd"]
    log.info(
        "the eight names with the largest weighted difference:\n%s",
        show.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
    )
    # the model's own second moments: the names' law does not depend on the correlation, so the
    # paths are those of lambda = 0 and nothing is calibrated; ``--model lc`` calibrates and takes
    # the calibrated model's pricing paths at the repricing pillars, the paths of the sweep's check
    w_model = np.array(spec.weights)
    if args.model == "lc":
        res = calibrate(spec, market)
        lam, pillars = res.lam, repricing_pillars(spec)
        log.info(
            "the paths below are the calibrated model's (%d particles, %.0f s; clipped mass inside ±%.1f sd %.4f)",
            spec.lc.particle.n_particles, res.wall_time, CLIP_GATE_SD, res.max_clipped_mass_inner,
        )  # fmt: skip
    else:
        res_times = TimeGrid.build([inp.T], spec.sim.dt_max).times
        lam, pillars = LocalCorrelationFunction.constant(0.0, res_times, market.index_lv.k_grid), [inp.T]  # fmt: skip
    smp = sample(model_of(spec, market, lam), spec.sim, w_model, pillars, keep_performances=True)
    mc = pair_mean(smp.sq)
    log.info(
        "the model's Monte Carlo Σ w E[R_i²] = %.6f (%.6f): against the SVI strips %+.6f, against the study's %+.6f "
        "(the single-name part of E[V] - E^Q[V])",
        mc[0], mc[1], mc[0] - float(w @ table["M_svi"]), mc[0] - float(w @ table["M_entry"]),
    )  # fmt: skip
    # the same Monte Carlo by region of the strike axis, exactly per path (model_regions)
    assert smp.R is not None
    per_name, mc_region, gap = model_regions(
        smp.R,
        w_model,
        table["f_svi"].to_numpy(),
        table["k_lo"].to_numpy(),
        table["k_hi"].to_numpy(),
    )
    table = pd.concat([table, per_name], axis=1)
    const = float(w_model @ (table["f_svi"] - 1.0) ** 2)
    gap_total = abs(sum(v[0] for v in mc_region.values()) + const - mc[0])
    fwd = mc_region[FORWARD_TERM]
    log.info(
        "the model's Monte Carlo by region (%d paths): the four regions + (f - 1)² + the forward term 2(f - 1)(X̄ - f) "
        "against E[R_i²], largest gap by name %.1e, weighted %.1e; without the forward term the gap is the term itself: "
        "largest by name %.1e, Σ w %+.2e (%.1e); the names' forwards: max |X̄/f - 1| = %.1e",
        smp.R.shape[0], gap, gap_total, float(table[f"mc: {FORWARD_TERM}"].abs().max()), fwd[0], fwd[1],
        float((table["f_mc"] / table["f_svi"] - 1.0).abs().max()),
    )  # fmt: skip
    if max(gap, gap_total) > 1e-12:
        raise RuntimeError(f"the regions do not sum to the Monte Carlo E[R²]: gap {gap:.2e}, weighted {gap_total:.2e}")  # fmt: skip
    sums = {"study": float(w @ table["M_study"]), "svi": float(w @ table["M_svi"])}
    region_rows = []
    for label in REGIONS:
        st, sv = float(w @ table[f"study: {label}"]), float(w @ table[f"svi: {label}"])
        region_rows.append({"region": label, "study": st, "svi": sv, "mc": mc_region[label][0], "mc_se": mc_region[label][1]})  # fmt: skip
    inner = {k: sum(r[k] for r in region_rows) for k in ("study", "svi", "mc")}
    # what is not strip: (f − 1)², the quadrature of the two strips, the forward term of the sample
    region_rows.append({"region": "the rest", "study": sums["study"] - inner["study"], "svi": sums["svi"] - inner["svi"],
                        "mc": mc[0] - inner["mc"], "mc_se": fwd[1]})  # fmt: skip
    region_rows.append({"region": "total", "study": sums["study"], "svi": sums["svi"], "mc": mc[0], "mc_se": mc[1]})  # fmt: skip
    for r in region_rows:
        r["svi_minus_study"], r["mc_minus_svi"], r["mc_minus_study"] = r["svi"] - r["study"], r["mc"] - r["svi"], r["mc"] - r["study"]  # fmt: skip
    regions = pd.DataFrame(region_rows)
    shown = pd.DataFrame({
        "region": regions["region"],
        "study": [f"{x:.6f}" for x in regions["study"]],
        "SVI": [f"{x:.6f}" for x in regions["svi"]],
        "model MC (se)": [f"{x:.6f} ({e:.6f})" for x, e in zip(regions["mc"], regions["mc_se"], strict=True)],
        "SVI - study": [_cell(x) for x in regions["svi_minus_study"]],
        "model - SVI (se)": [_cell(x, e) for x, e in zip(regions["mc_minus_svi"], regions["mc_se"], strict=True)],
        "model - study (se)": [_cell(x, e) for x, e in zip(regions["mc_minus_study"], regions["mc_se"], strict=True)],
    })  # fmt: skip
    log.info(
        "Σ w E[R_i²] by region of the strike axis: the study's strip, the SVI strip, the model's Monte Carlo (standard "
        "errors on antithetic pair means of the weighted sum; the rest: (f - 1)², the strips' quadrature, the forward term):\n%s",
        shown.to_string(index=False),
    )  # fmt: skip
    four = regions.iloc[: len(REGIONS)]
    lead = four.loc[four["mc_minus_study"].abs().idxmax()]
    total_ms = float(regions.iloc[-1]["mc_minus_study"])
    log.info(
        "model - study = %+.6f (%.6f), %+.2f %% (%.2f %%) of the study's strips; the region that carries most of it: "
        "%s, %+.6f (%.6f), of which SVI - study %+.6f and model - SVI %+.6f (%.6f)",
        total_ms, mc[1], 100 * total_ms / sums["study"], 100 * mc[1] / sums["study"], lead["region"], lead["mc_minus_study"],
        lead["mc_se"], lead["svi_minus_study"], lead["mc_minus_svi"], lead["mc_se"],
    )  # fmt: skip
    # the names: weighted differences, in total and by region
    by_name: dict[str, pd.DataFrame] = {}
    for tag, ref in (("study", "study"), ("svi", "svi")):
        t = pd.DataFrame({"name": table["name"], "weight": table["weight"]})
        t["total"] = table["weight"] * (table["M_mc"] - table[f"M_{ref}"])
        t["total_se"] = table["weight"] * table["M_mc_se"]
        for label, short in zip(REGIONS, REGION_SHORT, strict=True):
            t[short] = table["weight"] * (table[f"mc: {label}"] - table[f"{ref}: {label}"])
            t[f"{short}_se"] = table["weight"] * table[f"mc_se: {label}"]
        by_name[tag] = t
        top8 = t.reindex(t["total"].abs().sort_values(ascending=False).index).head(8)
        cells = pd.DataFrame({"name": top8["name"], "weight": [f"{x:.4f}" for x in top8["weight"]],
                              "total (se)": [_cell(x, e) for x, e in zip(top8["total"], top8["total_se"], strict=True)]})  # fmt: skip
        for short in REGION_SHORT:
            cells[f"{short} (se)"] = [_cell(x, e) for x, e in zip(top8[short], top8[f"{short}_se"], strict=True)]  # fmt: skip
        log.info(
            "w_i (model MC - %s strip) by name, the eight largest, in total and by region:\n%s",
            "the study's" if tag == "study" else "the SVI", cells.to_string(index=False),
        )  # fmt: skip
    svi_name = pd.DataFrame({"name": table["name"], "total": table["weight"] * (table["M_svi"] - table["M_study"])})  # fmt: skip
    for label, short in zip(REGIONS, REGION_SHORT, strict=True):
        svi_name[short] = table["weight"] * (table[f"svi: {label}"] - table[f"study: {label}"])
    top3 = by_name["study"].reindex(by_name["study"]["total"].abs().sort_values(ascending=False).index).head(3)  # fmt: skip
    for _, r in top3.iterrows():
        where = max(REGION_SHORT, key=lambda s: abs(float(r[s])))
        v = svi_name.loc[svi_name["name"] == r["name"]].iloc[0]
        log.info(
            "    %-5s w (model - study) %+.6f (%.6f), most of it %s: %+.6f (%.6f); there SVI - study %+.6f",
            r["name"], r["total"], r["total_se"], where, r[where], r[f"{where}_se"], v[where],
        )  # fmt: skip
    wing, wing_cells = call_wing_paths(
        smp.R, w_model, table["f_svi"].to_numpy(), table["k_hi"].to_numpy(), list(table["name"])
    )
    log.info(
        "the region above the listed strikes in the sample: %d of %d paths have a name above its last listed strike; "
        "share of the region's %.6f carried by the largest path %.1f %%, the 10 largest %.1f %%, the 100 largest %.1f %%, "
        "the 1000 largest %.1f %%; the largest cells (X = S_T/S_0):\n%s",
        wing["n_paths_above"], smp.R.shape[0], wing["region_mean"], *(100 * wing[f"share_top_{k}"] for k in WING_SHARES),
        wing_cells.to_string(index=False, float_format=lambda x: f"{x:.6f}"),
    )  # fmt: skip
    payload = {"date": args.date, "tenor": args.tenor, "table": table.to_dict("records"), "by_region": by_region, "rest": rest,
               "sum_w_M_entry": float(w @ table["M_entry"]), "sum_w_M_study": float(w @ table["M_study"]),
               "sum_w_M_svi": float(w @ table["M_svi"]), "mc_sum_w_R2": mc, "model": args.model,
               "call_wing": wing, "call_wing_cells": wing_cells.to_dict("records"),
               "regions": region_rows, "mc_by_region": mc_region, "mc_identity_gap": max(gap, gap_total),
               "model_minus_study_by_name": by_name["study"].to_dict("records"),
               "model_minus_svi_by_name": by_name["svi"].to_dict("records"),
               "svi_minus_study_by_name": svi_name.to_dict("records"),
               "screen": args.screen, "screen_settings": info["screen"], "dropped": info["dropped"],
               "names_unscreened": info["names_unscreened"],
               "record": record(spec, total_seconds=time.perf_counter() - t0)}  # fmt: skip
    write_json(f"strips_{args.date}_{args.tenor}_{args.screen}_{args.model}_{args.paths}", payload, args.out)  # fmt: skip
    return payload


# ---------------------------------------------------------------------------------------------
# calendar crossings
# ---------------------------------------------------------------------------------------------


def central_crossings(
    spec: LocalCorrelationSpec, res: LCCalibrationResult, central_sd: float = 2.0
) -> dict[str, list[tuple[float, float, float, float]]]:
    """Per name, the calendar crossings inside the cloud mean ``± central_sd`` standard
    deviations: ``(T_s, T_{s+1}, min dw, k)`` for each pair of consecutive slices whose total
    variance falls somewhere in that range."""
    ranges = visited_arbitrage(spec, res, central_sd=central_sd)["ranges"]
    out: dict[str, list[tuple[float, float, float, float]]] = {}
    for name, cfg, market in zip(spec.names, spec.surfaces, spec.markets, strict=True):
        r = ranges[name]
        surface = SviSlices(
            np.array(cfg.times),
            np.array(cfg.params),
            ForwardCurve.from_config(market),
            cfg.max_maturity,
        )
        rep = surface.arbitrage_report(k_lo=r["k_lo_central"], k_hi=r["k_hi_central"])
        found = [
            (float(cfg.times[s]), float(cfg.times[s + 1]), float(c), float(k))
            for s, (c, k) in enumerate(zip(rep.min_calendar, rep.argmin_calendar, strict=True))
            if c < -rep.tol
        ]
        if found:
            out[name] = found
    return out


def slice_to_drop(expiry_of: dict[float, str], pair: tuple[float, float]) -> str:
    """Of two crossing slices, the one to drop: the one that is not a third-Friday expiry — the
    shorter one when both are, or neither is (the rule is fixed in advance, SPEC §8.7)."""
    a, b = expiry_of[round(pair[0], 9)], expiry_of[round(pair[1], 9)]
    if third_friday(a) and not third_friday(b):
        return b
    return a


def run_calendar(args: argparse.Namespace) -> dict[str, Any]:
    t0 = time.perf_counter()
    inp = load_inputs(args.date, args.tenor)
    # the base is unrepaired whatever the screen's default: this command drops the crossing
    # slices itself, round by round, and measures the effect against the surfaces as fitted
    unrepaired = dataclasses.replace(SCREENS[args.screen], calendar_repair=False)
    options = {
        "n_particles": args.particles,
        "n_paths": args.paths,
        "screen": unrepaired,
        "schedule": SCHEDULES[args.schedule],
    }
    spec, _ = build_spec(inp, **options)
    market = build_lc_market(spec)
    res = calibrate(spec, market)
    base_cross = central_crossings(spec, res)
    log.info(
        "=== calendar crossings inside ±2 sd of the cloud on %s %s: %d names",
        args.date,
        args.tenor,
        len(base_cross),
    )
    log.info("the base: screen %s with the calendar repair off %s", args.screen, unrepaired.describe())  # fmt: skip
    removed: dict[str, list[str]] = {}
    work, cross = inp, base_cross
    for round_ in range(1, 6):
        if not cross:
            break
        smiles = dict(work.smiles)
        for name, pairs in cross.items():
            expiry_of = {round(float(e.T), 9): e.expiry for e in work.smiles[name]}
            gone = {slice_to_drop(expiry_of, (a, b)) for a, b, _, _ in pairs}
            for a, b, c, k in pairs:
                log.info("    round %d %-5s slices T = %.4f -> %.4f: w falls by %.2e at k = %+.4f; dropped: %s",
                         round_, name, a, b, -c, k, slice_to_drop(expiry_of, (a, b)))  # fmt: skip
            removed.setdefault(name, []).extend(sorted(gone))
            smiles[name] = [e for e in work.smiles[name] if e.expiry not in gone]
        work = dataclasses.replace(work, smiles=smiles)
        spec2, _ = build_spec(work, **options)
        market2 = build_lc_market(spec2)
        res2 = calibrate(spec2, market2)
        cross = central_crossings(spec2, res2)
    log.info(
        "slices dropped: %s; crossings left inside ±2 sd: %d names %s",
        removed,
        len(cross),
        sorted(cross),
    )
    w = np.array(spec.weights)
    pillars = repricing_pillars(spec)
    a = sample(model_of(spec, market, res.lam), spec.sim, w, pillars)
    b = sample(model_of(spec2, market2, res2.lam), spec2.sim, w, pillars)
    strikes = [float(x) for x in inp.entry["B1"]["strikes"]]
    rows = []
    for label, K in [("E[D]", 0.0)] + [
        (f"call {m:g}", K) for m, K in zip((0.5, 0.75, 1.0, 1.25, 1.5, 2.0), strikes, strict=True)
    ]:
        x, y = np.maximum(a.D - K, 0.0), np.maximum(b.D - K, 0.0)
        d, se = pair_mean(y - x)
        rows.append({"quantity": label, "base": pair_mean(x)[0], "base_se": pair_mean(x)[1], "repaired": pair_mean(y)[0],
                     "repaired_minus_base": d, "paired_se": se, "rel_%": 100 * d / pair_mean(x)[0], "rel_se_%": 100 * se / pair_mean(x)[0]})  # fmt: skip
    prices = pd.DataFrame(rows)
    log.info("prices, the offending slices dropped against the base (the same particles and pricing paths):\n%s",
             prices.to_string(index=False, float_format=lambda x: f"{x:.6g}"))  # fmt: skip
    smile = smile_difference(b.levels, a.levels, pillars, market.index_surface)
    cells = smile.assign(
        cell=[f"{d:+.4f} ({e:.4f})" for d, e in zip(smile["diff_vp"], smile["se_vp"], strict=True)]
    )
    log.info(
        "index smile, repaired minus base, vol points (paired se):\n%s",
        cells.pivot(index="T", columns="sd", values="cell").to_string(),
    )
    sq = pair_mean(b.sq - a.sq)
    log.info("Σ w E[R_i²]: %+.3e (%.1e); clipped mass inside ±%.1f sd: %.4f -> %.4f; verdict on E[D] against 0.05 %%: %s",
             sq[0], sq[1], CLIP_GATE_SD, res.max_clipped_mass_inner, res2.max_clipped_mass_inner,
             "above: a calendar repair is proposed" if abs(prices.iloc[0]["rel_%"]) > 0.05 else "below: recorded")  # fmt: skip
    payload = {"date": args.date, "tenor": args.tenor, "crossings": {k: v for k, v in base_cross.items()}, "removed": removed,
               "left": {k: v for k, v in cross.items()}, "prices": rows, "smile": smile.to_dict("records"),
               "record": record(spec, total_seconds=time.perf_counter() - t0)}  # fmt: skip
    write_json(f"calendar_{args.date}_{args.tenor}_{args.particles}", payload, args.out)
    return payload


# ---------------------------------------------------------------------------------------------
# the M12 baseline (S11)
# ---------------------------------------------------------------------------------------------

BASELINE_DATE = "2026-10-02"
BASELINE_FILE = ROOT / "tests" / "golden" / f"lcm_baseline_{BASELINE_DATE}.json"
CALL_MULTIPLES = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)


def baseline_run(
    date: str = BASELINE_DATE,
    tenor: str = "3m",
    *,
    n_particles: int = PRODUCTION,
    n_paths: int = PRODUCTION,
) -> dict[str, Any]:
    """The run behind the M12 baseline (S11): the full calibration on the screened target, the
    index repricing report, and the cells ``{name: (value, standard error)}`` — under LC the
    Palladium forward, the calls at the study's strikes, ``E[V]`` and ``κ = E[D]/√E[V]``; the
    constant-correlation companion's ``ρ_CC`` and forward; the paired LC/CC ratios."""
    t0 = time.perf_counter()
    inp = load_inputs(date, tenor)
    spec, info = build_spec(inp, n_particles=n_particles, n_paths=n_paths)
    market = build_lc_market(spec)
    T, w = spec.lc.particle.horizon, np.array(spec.weights)
    res = calibrate(spec, market)
    model = model_of(spec, market, res.lam)
    pillars = repricing_pillars(spec)
    report = reprice_index_smile(model, market.index_surface, spec.sim, maturities=pillars)
    s_lc = sample(model, spec.sim, w, pillars)
    lam_c = calibrate_constant_lambda(model, market.index_surface, T, spec.sim)
    cc = model.with_lambda(LocalCorrelationFunction.constant(lam_c, res.lam.times, res.lam.k_grid))
    s_cc = sample(cc, spec.sim, w, pillars)
    cells: dict[str, tuple[float, float]] = {
        "ED_lc": pair_mean(s_lc.D),
        "ED_cc": pair_mean(s_cc.D),
        "ratio": pair_ratio(s_lc.D, s_cc.D),
        "EV_lc": pair_mean(s_lc.V),
        "kappa_lc": pair_ratio_sqrt(s_lc.D, s_lc.V),
        "kappa_cc": pair_ratio_sqrt(s_cc.D, s_cc.V),
    }
    strikes = [float(k) for k in inp.entry["B1"]["strikes"]]
    for mlt, K in zip(CALL_MULTIPLES, strikes, strict=True):
        a, b = np.maximum(s_lc.D - K, 0.0), np.maximum(s_cc.D - K, 0.0)
        tag = f"{round(100 * mlt):03d}"
        cells[f"C_{tag}_lc"] = pair_mean(a)
        cells[f"C_{tag}_cc"] = pair_mean(b)
    return {
        "cells": cells,
        "strikes": strikes,
        "rho_cc": float(market.family.equicorrelation(lam_c)),
        "lambda_c": float(lam_c),
        "report": report,
        "result": res,
        "clip": clip_summary(res),
        "variance": variance_split(s_lc, inp),
        "info": info,
        "spec": spec,
        "inputs": inp,
        "seconds": time.perf_counter() - t0,
    }


def pair_ratio_sqrt(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """``E[a]/√E[b]`` with the delta-method standard error on pair means (``κ = E[D]/√E[V]``)."""
    ya, yb = 0.5 * (a[0::2] + a[1::2]), 0.5 * (b[0::2] + b[1::2])
    ma, mb = float(ya.mean()), float(yb.mean())
    value = ma / math.sqrt(mb)
    u = (ya - ma) / math.sqrt(mb) - 0.5 * value * (yb - mb) / mb
    return value, float(u.std(ddof=1) / np.sqrt(u.size))


def baseline_document(run: dict[str, Any]) -> dict[str, Any]:
    """What ``tests/golden/lcm_baseline_<date>.json`` holds."""
    spec = run["spec"]
    return {
        "date": run["inputs"].date,
        "tenor": run["inputs"].tenor,
        "convention": "fractions of notional; (value, standard error on antithetic pair means); tolerance max(2 se, 0.0002)",
        "cells": {k: [float(v[0]), float(v[1])] for k, v in run["cells"].items()},
        "strikes": run["strikes"],
        "rho_cc": run["rho_cc"],
        "max_clipped_mass": (
            run["clip"]["max_high"]
            if run["clip"]["max_high"] > run["clip"]["max_low"]
            else run["clip"]["max_low"]
        ),
        "max_clipped_mass_inner": max(run["clip"]["max_low_inner"], run["clip"]["max_high_inner"]),
        "index_error_vp_1p5sd": run["report"].max_abs_error(1.5),
        "index_error_vp_2p5sd": run["report"].max_abs_error(2.5),
        "screen": run["info"]["screen"],
        "n_dropped": len(run["info"]["dropped"]),
        "record": record(spec),
    }


def run_baseline(args: argparse.Namespace) -> dict[str, Any]:
    run = baseline_run(args.date, args.tenor, n_particles=args.particles, n_paths=args.paths)
    doc = baseline_document(run)
    log.info("=== M12 baseline %s %s (%.0f s)", args.date, args.tenor, run["seconds"])
    log.info("%s", format_clip(run["result"]))
    log.info("%s", run["report"].summary())
    for k, (v, se) in run["cells"].items():
        log.info("    %-10s %.6f (%.6f)", k, v, se)
    log.info("rho_CC %.6f", run["rho_cc"])
    if args.write:
        BASELINE_FILE.write_text(json.dumps(doc, indent=1) + "\n")
        log.info("written %s", BASELINE_FILE)
    else:
        write_json(f"baseline_{args.date}_{args.tenor}_{args.particles}", doc, args.out)
    return doc


# ---------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser, with_date: bool = True) -> None:
        if with_date:
            p.add_argument("--date", required=True)
        p.add_argument("--tenor", default="3m")
        p.add_argument("--screen", default="default", choices=sorted(SCREENS))
        p.add_argument("--schedule", default="lc", choices=sorted(SCHEDULES))
        p.add_argument("--particles", type=int, default=PRODUCTION)
        p.add_argument("--paths", type=int, default=PRODUCTION)
        p.add_argument(
            "--out",
            default=None,
            help="JSON path (default: outputs/dispersion_lc/diagnostics/<command>_….json)",
        )

    p = sub.add_parser("screen", help="the expiry screen, the quote quality and the alignment")
    p.add_argument("--dates", nargs="+", default=list(REFERENCE_DATES))
    p.add_argument("--tenor", default="3m")
    p.add_argument("--screen", default="default", choices=sorted(SCREENS))
    p.add_argument("--out", default=None)
    p.set_defaults(run=run_screen)
    p = sub.add_parser("calibrate", help="calibration, repricing, prices and timings")
    common(p)
    p.add_argument("--rho-max", type=float, default=0.98)
    p.add_argument("--lambda-tail", default="regressions", choices=("regressions", "flat"))
    p.add_argument("--target-average", default="step", choices=("step", "point"))
    p.add_argument("--no-price", action="store_true", help="calibrate only")
    p.add_argument(
        "--no-companion", action="store_true", help="skip the constant-correlation companion"
    )
    p.set_defaults(run=run_calibrate)
    p = sub.add_parser("dt", help="the step-halving check on common random numbers")
    common(p)
    p.add_argument("--target-average", default="step", choices=("step", "point"))
    p.set_defaults(run=run_dt)
    p = sub.add_parser("tail", help="the lambda tail rule: regressions against flat")
    common(p)
    p.set_defaults(run=run_tail)
    p = sub.add_parser("wing", help="the three diagnostics of the downside wing")
    common(p)
    p.set_defaults(run=run_wing)
    p = sub.add_parser(
        "strips",
        help="the single-name second moments by region: SVI, the study's strips, the model",
    )
    common(p)
    p.add_argument(
        "--model",
        default="zero",
        choices=("zero", "lc"),
        help="the paths of the Monte Carlo split: lambda = 0, nothing calibrated (the default; "
        "the names' law does not depend on the correlation), or the calibrated model's",
    )
    p.set_defaults(run=run_strips)
    p = sub.add_parser("calendar", help="the effect of the calendar crossings inside ±2 sd")
    common(p)
    p.set_defaults(run=run_calendar)
    p = sub.add_parser(
        "baseline",
        help="the M12 baseline (S11): --write records tests/golden/lcm_baseline_<date>.json",
    )
    p.add_argument("--date", default=BASELINE_DATE)
    p.add_argument("--tenor", default="3m")
    p.add_argument("--particles", type=int, default=PRODUCTION)
    p.add_argument("--paths", type=int, default=PRODUCTION)
    p.add_argument("--write", action="store_true")
    p.add_argument("--out", default=None)
    p.set_defaults(run=run_baseline)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("volsto").setLevel(logging.WARNING)
    args.run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
