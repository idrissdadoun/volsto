"""Cross-dependent volatility prototype: the stratified test of the owner's request (f) of
2026-10-09 (SPEC §8.7; ``volsto/multi/cdv.py``, ``docs/cross_dependent_vol.md``).

    python scripts/cdv_stratified.py [--table <the variant pass's table>] [--n 20]
        [--betas 1,2,3,4,6,8,10,12] [--g-max 3] [--budget development] [--workers 6]
        [--threads N] [--out <dir>] [--cache <dir>] [--model-s <parquet>] [--tenor 3m]
        [--dates d1,d2] [--smoke N] [--report-only]

The question.  On the dates where M12's local correlation runs into its cap (the clipped mass
inside ±2.5 sd), does closing the index wing with cross-dependence take the model's discount to
its constant-correlation companion from about 3 % towards the 6 % of the study's model S?

Selection (:func:`select_dates`).  Among the priced rows of ``--table`` (``status`` other than
``"failed"``): the ``--n`` dates with the largest ``clip_inner_max`` (group ``high``) and the
``--n`` with the smallest (group ``low``; ties by date).  Each date carries its group, the
table's ``clip_inner_max`` and the table's flag ``n_names_unscreened > 0``.  ``--dates``
replaces the selection (a date keeps the group the selection gives it, else ``given``).

Per date (:func:`run_date`; one process each, ``NUMBA_NUM_THREADS = cores // workers`` unless
``--threads``, one BLAS thread).  The specification of ``scripts/lcm_price.py`` at the budget
(``lcm_price.spec_for`` on the recorded SVI fits of ``--cache``, as ``lcm_price.run_date``) and
its market; the constant-correlation companion's ``λ_c`` through ``LocalCorrelationCache`` with
the ``"constant"`` family, as ``lcm_price.run_date``.  Then, all on the calibration grid and on
the same pricing seed (``cdv.simulate_cdv``: every ratio is paired):

* **CC** — ``λ ≡ λ_c``, ``β = 0``, scales 1 (the ``β = 0`` calibration's arrays with the rows of
  ``λ`` replaced);
* **LC** — CDV at ``β = 0``: M12's model;
* the betas of the grid in increasing order, calibration only, until the clipped mass inside
  ±2.5 sd (the larger of the two sides' maxima over the slices) is at most 1 %: that is ``β*``
  (:func:`select_beta`).  If no beta reaches 1 %, ``β*`` is the one with the smallest clipped
  mass and the row is flagged (``reached = False``);
* **CDV a, CDV b** — priced at ``β*`` and ``2β*`` (:func:`priced_betas`).  When ``β = 0``
  already satisfies the target, ``β* = 0`` and the two priced betas are the two smallest
  positive ones of the grid, for information: the row says so (``beta_star_zero``, ``note``).

Row (``<out>/rows/<date>.json``).  Per priced model ``cc``, ``lc``, ``cdv_a``, ``cdv_b``:
``E[D]``, ``κ = E[D]/√E[V]``, ``E[V]``, ``Σ w E[R_i²]``, ``E[R̄²]``, the clipped masses inside
±2.5 sd by side, the index smile errors at the horizon at the money and at −1.5 and −2.5 sd,
``E[V]/E^Q[V]``, ``E[R̄²]/M_B^listed`` and ``E[D]/P_D`` (the copula's forward of the entry B1;
its own standard error is added in quadrature).  The paired ratios CDV/CC, LC/CC and CDV/LC.
From the study: ``P_D``, the listed-variance forward ``√(EQV/EV)`` of the entry, model S's
``P_D_S/P_D`` with its ``converged`` flag.  Every beta tried with its clipped mass; seconds.
Standard errors are on antithetic pair means; ratios and ``κ`` by the delta method.

Resume.  A date is skipped when its row was written by the same git commit with the same
arguments (:func:`args_digest`) and did not fail; a date that fails gets a row with
``status = "failed"`` and its reason, and the run goes on.  One log line per date with its
seconds and the elapsed time, on stdout and in ``<out>/stratified.log``; each date's own output
is in ``<out>/logs/<date>.log``.

Outputs, all in ``--out`` (which must be under an ``outputs/dispersion_lc`` folder, as
``--cache`` must; never the study's own folders): ``cdv_stratified_<tenor>.parquet`` — the rows
with, as columns ``…_star`` and ``…_2star``, CDV at ``β*`` and at ``2β*`` (:func:`with_star_columns`:
M12 itself where ``β* = 0``) — and ``report_cdv_stratified_<tenor>.md``: per date, and the
averages by group with the standard error of the mean across dates.
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
import subprocess
import sys
import time
import traceback
from collections.abc import Sequence
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cdv_scan as cs  # noqa: E402
import lcm_diagnostics as lcd  # noqa: E402
import lcm_price as lp  # noqa: E402

from volsto._numba import num_threads  # noqa: E402
from volsto.calibration.cache import code_version  # noqa: E402
from volsto.calibration.fit_records import FitRecords  # noqa: E402
from volsto.calibration.lc_cache import (  # noqa: E402
    LocalCorrelationCache,
    build_lc_market,
    lc_spec_key,
)
from volsto.calibration.local_correlation import CLIP_GATE_SD, LC_CODE_TAG  # noqa: E402
from volsto.multi.cdv import CDVResult, CrossDependence, calibrate_cdv, simulate_cdv  # noqa: E402

log = logging.getLogger("cdv_stratified")

#: The worktree of M12: its variant pass's table, its cache and the study's tables are the
#: defaults of the arguments (the cache and the outputs under ``outputs/dispersion_lc``).
LC_ROOT = Path("/Users/idrissdadoun/Code/volsto-lc")
LC_OUT = LC_ROOT / "outputs" / "dispersion_lc"
#: The clipped mass inside ±2.5 sd that defines ``β*`` (the owner's 1 %; M12's gate).
CLIP_TARGET = 0.01
#: ``--fixed a,b``: no search - CDV is priced at these two betas on every date (the owner's request of
#: 2026-10-09: beta = 3 and beta = 6 on all dates).  ``None``: the search for the smallest beta.
FIXED: tuple[float, float] | None = None
MODELS = ("cc", "lc", "cdv_a", "cdv_b")
#: Row entries that are not scalars: left to the JSON rows, as text in the table.
NESTED = ("tried", "screen", "betas", "fixed_betas")
#: The per-model columns that get a ``…_star`` / ``…_2star`` view in the table.
STAR_KEYS = (
    "ED", "kappa", "EV", "EV_over_EQV", "E_Rbar2_over_listed", "names_over_listed", "over_copula",
    "idx_atm", "idx_m15", "idx_m25", "clip_inner", "clip_low_inner", "clip_high_inner",
)  # fmt: skip


# ---------------------------------------------------------------------------------------------
# the rules: the dates, the beta, the digest
# ---------------------------------------------------------------------------------------------


def select_dates(table: pd.DataFrame, n: int) -> pd.DataFrame:
    """The stratified sample of a pass's table: among its priced rows (``status`` other than
    ``"failed"``, a finite ``clip_inner_max``) the ``n`` dates with the largest
    ``clip_inner_max`` (group ``"high"``, largest first) and the ``n`` with the smallest (group
    ``"low"``, smallest first); ties go to the earlier date.  Columns: ``date``, ``group``,
    ``table_clip_inner_max``, ``table_unscreened`` (the table's ``n_names_unscreened > 0``;
    ``False`` when the table has no such column).  Fewer than ``2n`` priced rows is an error
    (the groups would overlap).  Checked in
    ``tests/test_cross_dependent_vol.py::test_stratified_rules``."""
    if n < 1:
        raise ValueError("n must be at least 1")
    priced = table[(table["status"] != "failed") & np.isfinite(table["clip_inner_max"])].copy()
    if len(priced) < 2 * n:
        raise ValueError(f"{len(priced)} priced rows: fewer than 2 x {n}, the groups would overlap")
    priced["date"] = priced["date"].astype(str).str[:10]
    if "n_names_unscreened" in priced:
        flag = priced["n_names_unscreened"].fillna(0.0) > 0
    else:
        flag = pd.Series(False, index=priced.index)
    priced["table_unscreened"] = flag.astype(bool)
    priced = priced.rename(columns={"clip_inner_max": "table_clip_inner_max"})
    cols = ["date", "group", "table_clip_inner_max", "table_unscreened"]
    high = priced.sort_values(["table_clip_inner_max", "date"], ascending=[False, True]).head(n)
    low = priced.sort_values(["table_clip_inner_max", "date"], ascending=[True, True]).head(n)
    return pd.concat([high.assign(group="high")[cols], low.assign(group="low")[cols]], ignore_index=True)  # fmt: skip


def given_selection(dates: Sequence[str], table: pd.DataFrame | None, n: int) -> pd.DataFrame:
    """The selection's frame for dates given on the command line: a date keeps the group that
    :func:`select_dates` gives it in ``table`` (``"given"`` otherwise) and carries the table's
    ``clip_inner_max`` and flag when the table has a row for it."""
    known: dict[str, dict[str, Any]] = {}
    lookup: dict[str, dict[str, Any]] = {}
    if table is not None:
        try:
            known = {r["date"]: r for r in select_dates(table, n).to_dict("records")}
        except ValueError:
            known = {}
        lookup = {str(r["date"])[:10]: r for r in table.to_dict("records")}
    rows = []
    for d in dates:
        if d in known:
            rows.append(known[d])
            continue
        r = lookup.get(d, {})
        clip = r.get("clip_inner_max")
        unscreened = r.get("n_names_unscreened")
        rows.append({
            "date": d, "group": "given",
            "table_clip_inner_max": float(clip) if clip is not None else float("nan"),
            "table_unscreened": bool(unscreened is not None and float(unscreened) > 0),
        })  # fmt: skip
    return pd.DataFrame(rows, columns=["date", "group", "table_clip_inner_max", "table_unscreened"])


def select_beta(
    tried: Sequence[tuple[float, float]], target: float = CLIP_TARGET
) -> tuple[float, bool]:
    """``(β*, reached)`` from the betas tried, ``(β, clipped mass inside ±2.5 sd)``: the
    smallest ``β`` whose clipped mass is at most ``target``; if none reaches it, the ``β`` with
    the smallest clipped mass (the smallest such ``β`` on a tie) and ``reached = False``.
    Checked in ``tests/test_cross_dependent_vol.py::test_stratified_rules``."""
    if not tried:
        raise ValueError("no beta tried")
    ordered = sorted((float(b), float(m)) for b, m in tried)
    if any(not math.isfinite(m) for _, m in ordered):
        raise ValueError("a clipped mass is not finite")
    for beta, mass in ordered:
        if mass <= target:
            return beta, True
    return min(ordered, key=lambda x: (x[1], x[0]))[0], False


def search_betas(betas: Sequence[float]) -> list[float]:
    """The positive betas of the grid, increasing, without repeats."""
    grid = sorted({float(b) for b in betas if float(b) > 0.0})
    if len(grid) < 2:
        raise ValueError("the grid needs at least two positive betas")
    return grid


def priced_betas(beta_star: float, betas: Sequence[float]) -> tuple[float, float]:
    """The two betas priced beside ``β = 0``: ``(β*, 2β*)``; when ``β* = 0`` (M12 already inside
    the clip target) the two smallest positive betas of the grid, priced for information."""
    if beta_star > 0.0:
        return float(beta_star), 2.0 * float(beta_star)
    grid = search_betas(betas)
    return grid[0], grid[1]


def args_digest(cfg: dict[str, Any], tenor: str, budget: str, betas: Sequence[float], g_max: float) -> str:  # fmt: skip
    """The digest a resume compares: the YAML (with ``--smoke``'s counts in its budget), the
    tenor, the budget, the grid of betas, ``g_max`` and the clip target."""
    doc = {"config": lp.config_digest(cfg, tenor, budget), "betas": search_betas(betas),
           "g_max": float(g_max), "clip_target": CLIP_TARGET, "fixed": FIXED}  # fmt: skip
    return hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()[:16]


def config_for(budget: str, smoke: int | None) -> dict[str, Any]:
    """The YAML of the runs; ``smoke`` replaces the budget's particle, path and companion-path
    counts (a test of the plumbing: its numbers are not results)."""
    cfg = lp.load_config()
    if smoke is not None:
        if smoke < 4000 or smoke % 2:
            raise ValueError("--smoke: an even count of at least 4000")
        counts = {"n_particles": int(smoke), "n_paths": int(smoke), "companion_paths": int(smoke)}
        cfg = {**cfg, "budgets": {**cfg["budgets"], budget: counts}}
    return cfg


def checked_dir(path: str | Path, what: str) -> Path:
    """``path`` resolved; refused unless it is under an ``outputs/dispersion_lc`` folder."""
    out = Path(path).resolve()
    if "outputs/dispersion_lc" not in str(out):
        raise ValueError(f"{what} must be under outputs/dispersion_lc (got {out})")
    return out


# ---------------------------------------------------------------------------------------------
# one date
# ---------------------------------------------------------------------------------------------


def model_block(
    tag: str,
    beta: float,
    paths: cs.Paths,
    result: CDVResult | None,
    surface: Any,
    T: float,
    listed: tuple[float, float, float],
    copula: tuple[float, float],
) -> dict[str, Any]:
    """The row entries of one priced model, suffixed ``_<tag>`` (module docstring).  ``result``:
    its calibration (``None`` for the companion: no ``λ`` is calibrated, no clipped mass)."""
    eqv, sum_wm, m_b = listed
    p_d, p_d_se = copula
    out: dict[str, Any] = {f"beta_{tag}": float(beta)}

    def put(name: str, value: tuple[float, float], scale: float = 1.0) -> None:
        out[f"{name}_{tag}"], out[f"{name}_{tag}_se"] = float(value[0]) / scale, float(value[1]) / scale  # fmt: skip

    ed = lcd.pair_mean(paths.D)
    ev = lcd.pair_mean(paths.V)
    names = lcd.pair_mean(paths.sq)
    rbar2 = lcd.pair_mean(paths.rb**2)
    put("ED", ed)
    put("kappa", lcd.pair_ratio_sqrt(paths.D, paths.V))
    put("EV", ev)
    put("sum_w_ER2", names)
    put("E_Rbar2", rbar2)
    put("EV_over_EQV", ev, eqv)
    put("E_Rbar2_over_listed", rbar2, m_b)
    put("names_over_listed", names, sum_wm)
    ratio = ed[0] / p_d
    rel = math.hypot(ed[1] / ed[0], p_d_se / p_d if math.isfinite(p_d_se) else 0.0)
    put("over_copula", (ratio, ratio * rel))
    errs = cs.smile_errors(paths.level, surface, T, (-2.5, -1.5, 0.0))
    for name, label in (("idx_atm", "+0.0"), ("idx_m15", "-1.5"), ("idx_m25", "-2.5")):
        put(name, errs[label])
    put("forward_error", lcd.pair_mean(paths.level - 1.0))
    low = float(result.inner_low.max()) if result is not None else float("nan")
    high = float(result.inner_high.max()) if result is not None else float("nan")
    out[f"clip_low_inner_{tag}"], out[f"clip_high_inner_{tag}"] = low, high
    return out


def model_s_of(path: Path, date: str) -> tuple[float, bool | None]:
    """``(P_D_S, converged)`` of the study's model S table for ``date``; ``(nan, None)`` when
    the table has no row for it."""
    if not path.exists():
        return float("nan"), None
    table = pd.read_parquet(path, columns=["date", "converged", "P_D_S"])
    hit = table[table["date"].astype(str).str[:10] == date]
    if hit.empty:
        return float("nan"), None
    return float(hit["P_D_S"].iloc[0]), bool(hit["converged"].iloc[0])


def run_date(
    date: str,
    tenor: str,
    cfg: dict[str, Any],
    *,
    budget: str,
    betas: Sequence[float],
    g_max: float,
    cache_root: Path,
    model_s: Path,
) -> dict[str, Any]:
    """The row of ``date`` (module docstring).  Raises on a failure (:func:`safe_row` wraps it)."""
    t_start = time.perf_counter()
    grid = search_betas(betas)
    records = FitRecords(cache_root / "svi_fits")
    cache = LocalCorrelationCache(cache_root)
    inp = lcd.load_inputs(date, tenor, cfg["index"])
    spec, info = lp.spec_for(inp, cfg, budget, records)
    market = build_lc_market(spec)
    surface = market.index_surface
    T, w = spec.lc.particle.horizon, np.array(spec.weights)
    listed = cs.listed_moments(inp)
    eqv, sum_wm, m_b = listed
    b1 = inp.entry["B1"]
    p_d, p_d_se = float(b1["P_D"]), float(b1.get("P_D_se", float("nan")))
    ev_cop = float(b1["EV"])
    p_d_s, converged = model_s_of(model_s, date)
    row: dict[str, Any] = {
        "date": date, "tenor": tenor, "T": T, "n_names": spec.n_names, "budget": budget,
        "g_max": float(g_max), "betas": grid, "clip_target": CLIP_TARGET, "clip_gate_sd": CLIP_GATE_SD,
        "particle_seed": spec.lc.particle.seed, "pricing_seed": spec.sim.seed,
        "n_particles": spec.lc.particle.n_particles, "n_paths": spec.sim.n_paths,
        "companion_paths": spec.lc.parametric.n_paths, "git_commit": code_version(),
        "spec_key": lc_spec_key(spec), "lc_code_tag": LC_CODE_TAG, "schedule": repr(spec.sim.step_schedule),
        "threads": num_threads(), "screen": dict(cfg["screen"]),
        "n_names_unscreened": info.get("n_names_unscreened"), "n_dropped": len(info["dropped"]),
        "status": "ok", "reason": "",
        "P_D": p_d, "P_D_se": p_d_se, "EV_copula": ev_cop, "EQV": eqv, "sum_w_M": sum_wm, "M_B_listed": m_b,
        "kappa_cop": p_d / math.sqrt(ev_cop), "listed_variance_forward": math.sqrt(eqv / ev_cop),
        "P_D_S": p_d_s, "model_s_converged": converged, "P_D_S_over_P_D": p_d_s / p_d,
    }  # fmt: skip
    # --- the constant-correlation companion, through the cache (as lcm_price.run_date)
    t0 = time.perf_counter()
    cc_spec = dataclasses.replace(spec, lc=dataclasses.replace(spec.lc, family="constant"))
    hit = cache.has(cc_spec)
    _, diag_cc = cache.get_or_calibrate(cc_spec, allow_calibrate=True)
    assert diag_cc is not None
    lam_c = float(diag_cc.calibration["lambda"])
    row.update(lambda_c=lam_c, rho_cc=float(market.family.equicorrelation(lam_c)),
               companion_cache_hit=bool(hit), seconds_companion=time.perf_counter() - t0)  # fmt: skip
    # --- the calibrations: beta = 0, then the grid until the clip target
    results: dict[float, CDVResult] = {}
    tried: list[dict[str, Any]] = []
    empty = np.empty(0)

    def calibrate(beta: float, in_search: bool) -> CDVResult:
        if beta in results:
            return results[beta]
        c0 = time.perf_counter()
        dep = CrossDependence(float(beta), g_max=g_max, g_min=1.0 / g_max)
        res = calibrate_cdv(market.models, market.family, market.basket, surface, market.index_lv,
                            spec.lc.particle, spec.sim, spec.lc, dep)  # fmt: skip
        # the final cloud is not needed again: the pricing pass reads the rows and the scales
        res = dataclasses.replace(res, final_log_spot=empty, final_k_basket=empty)
        results[beta] = res
        tried.append({"beta": float(beta), "clip_low_inner": float(res.inner_low.max()),
                      "clip_high_inner": float(res.inner_high.max()), "clip_inner": res.max_clipped_mass_inner,
                      "in_search": in_search, "seconds": time.perf_counter() - c0})  # fmt: skip
        log.info("%s: beta %.3g calibrated in %.0f s: clipped mass inside %.4f low, %.4f high",
                 date, beta, tried[-1]["seconds"], tried[-1]["clip_low_inner"], tried[-1]["clip_high_inner"])  # fmt: skip
        return res

    t0 = time.perf_counter()
    res0 = calibrate(0.0, True)
    if FIXED is not None:
        # no search: the two given betas on every date; "reached" says whether the first one
        # brings the clipped mass inside the target
        beta_a, beta_b = FIXED
        res_a, res_b = calibrate(beta_a, False), calibrate(beta_b, False)
        beta_star, reached, zero = beta_a, bool(res_a.max_clipped_mass_inner <= CLIP_TARGET), False
        note = f"fixed betas {beta_a:g} and {beta_b:g} (no search)"
    else:
        for beta in grid:
            if tried[-1]["clip_inner"] <= CLIP_TARGET:
                break
            calibrate(beta, True)
        beta_star, reached = select_beta([(x["beta"], x["clip_inner"]) for x in tried])
        beta_a, beta_b = priced_betas(beta_star, grid)
        res_a, res_b = calibrate(beta_a, False), calibrate(beta_b, False)
        zero = beta_star == 0.0
        note = "" if reached else f"no beta of the grid reaches {CLIP_TARGET:.0%}: beta* = {beta_star:g} has the smallest clipped mass"  # fmt: skip
    if zero:
        why = "M12 is inside the clip target" if reached else note
        note = f"beta* = 0 ({why}); CDV priced at the two smallest positive betas of the grid ({beta_a:g}, {beta_b:g}) for information"  # fmt: skip
    row.update(
        beta_star=beta_star, reached=bool(reached), beta_star_zero=bool(zero), note=note,
        fixed_betas=list(FIXED) if FIXED is not None else None, beta_a=float(beta_a), beta_b=float(beta_b),
        clip_inner_lc=res0.max_clipped_mass_inner, clip_inner_cdv_a=res_a.max_clipped_mass_inner,
        clip_inner_cdv_b=res_b.max_clipped_mass_inner, tried=tried, n_calibrations=len(tried),
        seconds_calibrations=time.perf_counter() - t0,
    )  # fmt: skip
    # --- prices: CC, LC and the two CDV models on the same pricing paths (the calibration grid)
    t0 = time.perf_counter()
    cc_res = dataclasses.replace(
        res0, lam=np.full_like(res0.lam, lam_c), lam_star=np.full_like(res0.lam_star, lam_c),
        scale=np.ones_like(res0.scale),
    )  # fmt: skip
    copula = (p_d, p_d_se)
    priced: dict[str, cs.Paths] = {}
    for tag, beta, res, cal in (("cc", 0.0, cc_res, None), ("lc", 0.0, res0, res0),
                                ("cdv_a", beta_a, res_a, res_a), ("cdv_b", beta_b, res_b, res_b)):  # fmt: skip
        ls, kb = simulate_cdv(res, market.models, market.family, market.basket, spec.sim, [T])
        priced[tag] = cs.paths_of(ls[0], kb[0], market, w)
        del ls, kb
        row.update(model_block(tag, beta, priced[tag], cal, surface, T, listed, copula))
    for num, den in (("lc", "cc"), ("cdv_a", "cc"), ("cdv_b", "cc"), ("cdv_a", "lc"), ("cdv_b", "lc")):  # fmt: skip
        value, se = lcd.pair_ratio(priced[num].D, priced[den].D)
        row[f"ratio_{num}_{den}"], row[f"ratio_{num}_{den}_se"] = value, se
    row["seconds_pricing"] = time.perf_counter() - t0
    numbers = [v for k, v in row.items() if isinstance(v, float) and k.startswith(("ED_", "kappa_", "ratio_"))]  # fmt: skip
    if not all(math.isfinite(v) for v in numbers):
        row["status"], row["reason"] = "check", "a forward, a kappa or a ratio is not finite"
    row["seconds_total"] = time.perf_counter() - t_start
    return row


def safe_row(date: str, tenor: str, cfg: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    """:func:`run_date`, a failure turned into a row with its status and reason."""
    t0 = time.perf_counter()
    try:
        return run_date(date, tenor, cfg, **kwargs)
    except Exception as exc:
        log.error("%s %s failed: %s\n%s", date, tenor, exc, traceback.format_exc())
        return {"date": date, "tenor": tenor, "budget": kwargs.get("budget"), "git_commit": code_version(),
                "lc_code_tag": LC_CODE_TAG, "status": "failed", "reason": f"{type(exc).__name__}: {exc}"[:500],
                "seconds_total": time.perf_counter() - t0}  # fmt: skip


def write_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(row, indent=1, default=lp.jsonable))
    tmp.replace(path)


def read_row(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return dict(json.loads(path.read_text()))
    except (OSError, ValueError):
        return None


# ---------------------------------------------------------------------------------------------
# the table and the report
# ---------------------------------------------------------------------------------------------


def with_star_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """The table with CDV at ``β*`` and at ``2β*`` as columns ``<key>_star`` and
    ``<key>_2star`` (``STAR_KEYS`` and the ratios to CC and to LC): the priced models ``cdv_a``
    and ``cdv_b`` where ``β* > 0``, and M12 itself (LC, the CDV model at ``β = 0``) where
    ``β* = 0`` — there the wing is already inside the clip target, ``2β* = 0`` too, and
    ``cdv_a``, ``cdv_b`` are the two smallest positive betas, priced for information.  Also
    ``flag_unscreened`` (the selection table's flag or the run's own count), the model S ratio
    on its converged dates only (``S_over_copula``) and the paired differences the report
    averages.  Checked in ``tests/test_cross_dependent_vol.py::test_stratified_table``."""
    out = frame.copy()
    if "beta_star" not in out:
        return out
    zero = (out["beta_star"] == 0.0).to_numpy()

    def pick(cdv: str, lc: Any) -> Any:
        return out[cdv].where(~zero, lc)

    for key in STAR_KEYS:
        for suffix in ("", "_se"):
            if f"{key}_lc{suffix}" not in out:
                continue
            out[f"{key}_star{suffix}"] = pick(f"{key}_cdv_a{suffix}", out[f"{key}_lc{suffix}"])
            out[f"{key}_2star{suffix}"] = pick(f"{key}_cdv_b{suffix}", out[f"{key}_lc{suffix}"])
    out["beta_2star"] = 2.0 * out["beta_star"]
    for suffix in ("", "_se"):
        out[f"ratio_star_cc{suffix}"] = pick(f"ratio_cdv_a_cc{suffix}", out[f"ratio_lc_cc{suffix}"])
        out[f"ratio_2star_cc{suffix}"] = pick(f"ratio_cdv_b_cc{suffix}", out[f"ratio_lc_cc{suffix}"])  # fmt: skip
        own = 1.0 if suffix == "" else 0.0  # LC over itself
        out[f"ratio_star_lc{suffix}"] = pick(f"ratio_cdv_a_lc{suffix}", own)
        out[f"ratio_2star_lc{suffix}"] = pick(f"ratio_cdv_b_lc{suffix}", own)
    flag = as_bool(out["table_unscreened"]) if "table_unscreened" in out else pd.Series(False, index=out.index)  # fmt: skip
    if "n_names_unscreened" in out:
        flag = flag | (pd.to_numeric(out["n_names_unscreened"], errors="coerce").fillna(0.0) > 0)
    out["flag_unscreened"] = flag
    out["S_converged"] = as_bool(out["model_s_converged"])
    out["S_over_copula"] = out["P_D_S_over_P_D"].where(out["S_converged"])
    out["star_cc_minus_lc_cc"] = out["ratio_star_cc"] - out["ratio_lc_cc"]
    out["two_star_cc_minus_lc_cc"] = out["ratio_2star_cc"] - out["ratio_lc_cc"]
    out["lc_cc_minus_S"] = out["ratio_lc_cc"] - out["S_over_copula"]
    out["star_cc_minus_S"] = out["ratio_star_cc"] - out["S_over_copula"]
    out["two_star_cc_minus_S"] = out["ratio_2star_cc"] - out["S_over_copula"]
    out["star_copula_minus_S"] = out["over_copula_star"] - out["S_over_copula"]
    out["two_star_copula_minus_S"] = out["over_copula_2star"] - out["S_over_copula"]
    return out


def as_bool(series: pd.Series) -> pd.Series:
    """A column of flags as booleans: ``True`` where the entry is a true boolean, ``False``
    elsewhere (a missing entry, a failed row)."""
    values = [isinstance(v, (bool, np.bool_)) and bool(v) for v in series]
    return pd.Series(values, index=series.index, dtype=bool)


def group_means(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Per group and column: the mean across dates, the standard error of the mean
    (``sd/√n`` with ``ddof = 1``; ``nan`` for one date) and ``n``, the count of finite values.
    Checked in ``tests/test_cross_dependent_vol.py::test_stratified_table``."""
    rows = []
    for group, part in frame.groupby("group", sort=True):
        for col in columns:
            x = pd.to_numeric(part[col], errors="coerce").to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            mean = float(x.mean()) if x.size else float("nan")
            sem = float(x.std(ddof=1) / math.sqrt(x.size)) if x.size > 1 else float("nan")
            rows.append({"group": group, "column": col, "mean": mean, "sem": sem, "n": int(x.size)})
    return pd.DataFrame(rows, columns=["group", "column", "mean", "sem", "n"])


def build_table(rows_dir: Path, selection: pd.DataFrame, digest: str) -> pd.DataFrame:
    """The rows on disk of the selection's dates written with these arguments (``digest``; the
    commit is a column: a table rebuilt after a later commit keeps its rows), with the
    selection's columns and the star columns."""
    rows = []
    for sel in selection.to_dict("records"):
        r = read_row(rows_dir / f"{sel['date']}.json")
        if r is None or r.get("args_digest") != digest:
            continue
        flat = {k: v for k, v in r.items() if k not in NESTED}
        for k in NESTED:
            if k in r:
                flat[f"{k}_json"] = json.dumps(r[k], default=lp.jsonable)
        rows.append({**flat, **sel})
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    order = frame["group"].map({"high": 0, "low": 1}).fillna(2)
    frame = frame.assign(_o=order).sort_values(["_o", "date"]).drop(columns="_o").reset_index(drop=True)  # fmt: skip
    return with_star_columns(frame)


def cell(value: Any, se: Any = None, digits: int = 4) -> str:
    """``"0.9703 (0.0004)"``; ``"n/a"`` for a missing value."""
    if value is None or not isinstance(value, (int, float, np.floating)) or not math.isfinite(value):  # fmt: skip
        return "n/a"
    text = f"{value:.{digits}f}"
    if se is not None and isinstance(se, (int, float, np.floating)) and math.isfinite(se):
        text += f" ({se:.{digits}f})"
    return text


def md_table(header: Sequence[str], lines: Sequence[Sequence[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(line) + " |" for line in lines]
    return "\n".join(out)


#: The columns of the report's averages: (column, label).
SUMMARY = (
    ("table_clip_inner_max", "clipped mass of M12 in the selection table"),
    ("clip_inner_lc", "clipped mass of LC in this run"),
    ("beta_star", "beta*"),
    ("clip_inner_star", "clipped mass at beta*"),
    ("clip_inner_2star", "clipped mass at 2 beta*"),
    ("ratio_lc_cc", "LC/CC"),
    ("ratio_star_cc", "CDV(beta*)/CC"),
    ("ratio_2star_cc", "CDV(2 beta*)/CC"),
    ("star_cc_minus_lc_cc", "CDV(beta*)/CC - LC/CC"),
    ("two_star_cc_minus_lc_cc", "CDV(2 beta*)/CC - LC/CC"),
    ("S_over_copula", "P_D_S/P_D (converged dates)"),
    ("listed_variance_forward", "listed-variance forward sqrt(EQV/EV), over P_D"),
    ("over_copula_cc", "CC/P_D"),
    ("over_copula_lc", "LC/P_D"),
    ("over_copula_star", "CDV(beta*)/P_D"),
    ("over_copula_2star", "CDV(2 beta*)/P_D"),
    ("lc_cc_minus_S", "LC/CC - P_D_S/P_D"),
    ("star_cc_minus_S", "CDV(beta*)/CC - P_D_S/P_D"),
    ("two_star_cc_minus_S", "CDV(2 beta*)/CC - P_D_S/P_D"),
    ("star_copula_minus_S", "CDV(beta*)/P_D - P_D_S/P_D"),
    ("two_star_copula_minus_S", "CDV(2 beta*)/P_D - P_D_S/P_D"),
    ("kappa_cop", "kappa: copula"),
    ("kappa_cc", "kappa: CC"),
    ("kappa_lc", "kappa: LC"),
    ("kappa_star", "kappa: CDV(beta*)"),
    ("kappa_2star", "kappa: CDV(2 beta*)"),
    ("EV_over_EQV_cc", "E[V]/E_Q[V]: CC"),
    ("EV_over_EQV_lc", "E[V]/E_Q[V]: LC"),
    ("EV_over_EQV_star", "E[V]/E_Q[V]: CDV(beta*)"),
    ("EV_over_EQV_2star", "E[V]/E_Q[V]: CDV(2 beta*)"),
    ("E_Rbar2_over_listed_cc", "E[Rbar^2]/M_B^listed: CC"),
    ("E_Rbar2_over_listed_lc", "E[Rbar^2]/M_B^listed: LC"),
    ("E_Rbar2_over_listed_star", "E[Rbar^2]/M_B^listed: CDV(beta*)"),
    ("E_Rbar2_over_listed_2star", "E[Rbar^2]/M_B^listed: CDV(2 beta*)"),
    ("names_over_listed_lc", "sum w E[R_i^2] / sum w M_i: LC"),
    ("names_over_listed_star", "sum w E[R_i^2] / sum w M_i: CDV(beta*)"),
    ("names_over_listed_2star", "sum w E[R_i^2] / sum w M_i: CDV(2 beta*)"),
    ("idx_m25_lc", "index smile error at -2.5 sd (vp): LC"),
    ("idx_m25_star", "index smile error at -2.5 sd (vp): CDV(beta*)"),
    ("idx_m25_2star", "index smile error at -2.5 sd (vp): CDV(2 beta*)"),
)


def report(frame: pd.DataFrame, meta: dict[str, Any]) -> str:
    """The markdown report: what was run, per date by group, the averages by group with the
    standard error of the mean across dates (all the dates, and without the flagged ones), the
    informative betas of the dates with ``β* = 0``, and the failed dates."""
    L: list[str] = [
        f"# Cross-dependent volatility: the stratified test ({meta['tenor']}, {meta['budget']} budget)",
        "",
        "The owner's request (f) of 2026-10-09 (SPEC 8.7): on the dates where M12's local correlation binds, does closing "
        "the index wing with cross-dependence take the discount to the constant-correlation companion from about 3 % towards "
        "model S's 6 %?  Written by `scripts/cdv_stratified.py`; nothing here is written into the study's folders.",
        "",
        f"- commit `{meta['commit']}`, arguments digest `{meta['digest']}`; g_max = {meta['g_max']:g}; grid of beta: "
        f"{', '.join(f'{b:g}' for b in meta['betas'])}; clip target {CLIP_TARGET:.0%} inside +-{CLIP_GATE_SD:g} sd.",
        f"- selection: {meta['selection']}",
        f"- screen of the specification (the YAML's): `{json.dumps(meta['screen'], sort_keys=True)}`.",
        f"- particles and paths: {meta.get('counts', 'n/a')}." + (" **Smoke run: the numbers are not results.**" if meta.get("smoke") else ""),
        "",
        "Models, all simulated on the calibration grid with the same pricing seed (every ratio between them is paired): "
        "**CC** the constant-correlation companion (lambda_c of the M12 cache's constant family); **LC** M12 (CDV at beta = 0); "
        "**CDV(beta\\*)** the cross-dependent model at the smallest beta of the grid whose clipped mass inside +-2.5 sd is at most "
        "1 % (if none: the beta with the smallest clipped mass, `reached` = no), and **CDV(2 beta\\*)**.  Where beta\\* = 0, M12 is "
        "already inside the target and CDV(beta\\*) = CDV(2 beta\\*) = LC; the two smallest positive betas are priced for information "
        "(last table).  `P_D` is the copula's forward of the entry B1, `P_D_S` model S's (on its converged dates), the "
        "listed-variance forward is sqrt(EQV/EV) of the entry.  Standard errors in brackets: Monte Carlo (antithetic pair means, "
        "delta method) in the per-date tables, the standard error of the mean across dates in the averages.",
        "",
    ]  # fmt: skip
    if frame.empty:
        return "\n".join([*L, "No row yet."]) + "\n"
    commits = sorted({str(c) for c in frame["git_commit"].dropna()})
    L += [f"Rows: {len(frame)}, written by commit(s) {', '.join(f'`{c}`' for c in commits)}.", ""]
    ok = frame[frame["status"] != "failed"]
    if not ok.empty:
        ok = ok.assign(reached=as_bool(ok["reached"]), beta_star_zero=as_bool(ok["beta_star_zero"]))
        L += _report_priced(ok)
    failed = frame[frame["status"] == "failed"]
    if not failed.empty:
        L += ["## Failed dates", "", md_table(["date", "group", "reason"], [[r["date"], str(r["group"]), str(r["reason"])] for r in failed.to_dict("records")]), ""]  # fmt: skip
    return "\n".join(L) + "\n"


GROUP_TITLES = {
    "high": "the largest clipped mass of M12",
    "low": "the smallest clipped mass of M12",
    "given": "dates given on the command line",
}


def _report_priced(ok: pd.DataFrame) -> list[str]:
    """The report's sections on the priced rows: per date by group, the averages, the
    informative betas of the dates with ``β* = 0``."""
    L: list[str] = []
    groups = [g for g in GROUP_TITLES if (ok["group"] == g).any()]
    for group in groups:
        part = ok[ok["group"] == group]
        L += [f"## Group `{group}`: {GROUP_TITLES[group]} ({len(part)} dates)", ""]
        lines = [
            [r["date"], cell(r["table_clip_inner_max"]), "yes" if r["flag_unscreened"] else "", cell(r["clip_inner_lc"]),
             f"{r['beta_star']:g}", "yes" if r["reached"] else "**no**", cell(r["clip_low_inner_star"]) + " / " + cell(r["clip_high_inner_star"]),
             cell(r["ratio_lc_cc"], r["ratio_lc_cc_se"]), cell(r["ratio_star_cc"], r["ratio_star_cc_se"]),
             cell(r["ratio_2star_cc"], r["ratio_2star_cc_se"]),
             cell(r["P_D_S_over_P_D"]) + ("" if r["S_converged"] else " (not converged)"),
             cell(r["over_copula_cc"], r["over_copula_cc_se"]), cell(r["over_copula_lc"], r["over_copula_lc_se"]),
             cell(r["over_copula_star"], r["over_copula_star_se"]), cell(r["over_copula_2star"], r["over_copula_2star_se"]),
             cell(r["listed_variance_forward"])]
            for r in part.to_dict("records")
        ]  # fmt: skip
        L += [md_table(
            ["date", "M12 clip (table)", "unscreened", "LC clip (run)", "beta*", "reached", "clip at beta* low / high", "LC/CC", "CDV(beta*)/CC",
             "CDV(2 beta*)/CC", "P_D_S/P_D", "CC/P_D", "LC/P_D", "CDV(beta*)/P_D", "CDV(2 beta*)/P_D", "listed-variance forward"], lines), ""]  # fmt: skip
        lines = [
            [r["date"], cell(r["kappa_cop"]), cell(r["kappa_cc"], r["kappa_cc_se"]), cell(r["kappa_lc"], r["kappa_lc_se"]),
             cell(r["kappa_star"], r["kappa_star_se"]), cell(r["kappa_2star"], r["kappa_2star_se"]),
             cell(r["EV_over_EQV_lc"], r["EV_over_EQV_lc_se"]), cell(r["EV_over_EQV_star"], r["EV_over_EQV_star_se"]),
             cell(r["EV_over_EQV_2star"], r["EV_over_EQV_2star_se"]),
             cell(r["E_Rbar2_over_listed_lc"], r["E_Rbar2_over_listed_lc_se"]), cell(r["E_Rbar2_over_listed_star"], r["E_Rbar2_over_listed_star_se"]),
             cell(r["E_Rbar2_over_listed_2star"], r["E_Rbar2_over_listed_2star_se"]),
             cell(r["idx_atm_lc"], r["idx_atm_lc_se"], 2), cell(r["idx_m15_lc"], r["idx_m15_lc_se"], 2) + " / " + cell(r["idx_m25_lc"], r["idx_m25_lc_se"], 2),
             cell(r["idx_m15_star"], r["idx_m15_star_se"], 2) + " / " + cell(r["idx_m25_star"], r["idx_m25_star_se"], 2),
             cell(r["idx_m15_2star"], r["idx_m15_2star_se"], 2) + " / " + cell(r["idx_m25_2star"], r["idx_m25_2star_se"], 2),
             f"{r['seconds_total']:.0f}"]
            for r in part.to_dict("records")
        ]  # fmt: skip
        L += [md_table(
            ["date", "kappa copula", "kappa CC", "kappa LC", "kappa CDV(beta*)", "kappa CDV(2 beta*)", "E[V]/E_Q[V] LC", "CDV(beta*)", "CDV(2 beta*)",
             "E[Rbar^2]/M_B^listed LC", "CDV(beta*)", "CDV(2 beta*)", "index ATM LC (vp)", "index -1.5 / -2.5 sd LC (vp)", "CDV(beta*)", "CDV(2 beta*)", "seconds"], lines), ""]  # fmt: skip
    L += ["## Averages by group", "",
          "Mean across dates with the standard error of the mean in brackets and the number of dates in square brackets.  "
          "`without flagged`: without the dates that carry a name kept unscreened (owner's decision 2).", ""]  # fmt: skip
    cols = [c for c, _ in SUMMARY if c in ok]
    views = [("all dates", ok), ("without flagged", ok[~ok["flag_unscreened"]])]
    means = {name: group_means(view, cols) for name, view in views if not view.empty}
    lines = []
    for col, label in SUMMARY:
        if col not in ok:
            continue
        line = [label]
        for g in groups:
            for m in means.values():
                hit = m[(m["group"] == g) & (m["column"] == col)]
                line.append("n/a" if hit.empty or hit["n"].iloc[0] == 0 else f"{cell(hit['mean'].iloc[0], hit['sem'].iloc[0])} [{int(hit['n'].iloc[0])}]")  # fmt: skip
        lines.append(line)
    L += [md_table(["quantity"] + [f"{g}: {name}" for g in groups for name in means], lines), ""]
    lines = []
    for g in groups:
        part = ok[ok["group"] == g]
        lines.append([g, str(len(part)), str(int(part["reached"].sum())), str(int(part["beta_star_zero"].sum())),
                      str(int(part["flag_unscreened"].sum())), str(int(part["S_converged"].sum()))])  # fmt: skip
    L += [md_table(["group", "dates", "beta* reaches the target", "beta* = 0", "flagged (unscreened)", "model S converged"], lines), ""]  # fmt: skip
    zero = ok[ok["beta_star_zero"]]
    if not zero.empty:
        L += ["## Dates with beta* = 0: the two smallest positive betas, for information", ""]
        lines = [
            [r["date"], r["group"], f"{r['beta_cdv_a']:g}", cell(r["clip_inner_cdv_a"]), cell(r["ratio_cdv_a_cc"], r["ratio_cdv_a_cc_se"]),
             cell(r["ratio_cdv_a_lc"], r["ratio_cdv_a_lc_se"]), cell(r["kappa_cdv_a"], r["kappa_cdv_a_se"]), f"{r['beta_cdv_b']:g}",
             cell(r["clip_inner_cdv_b"]), cell(r["ratio_cdv_b_cc"], r["ratio_cdv_b_cc_se"]), cell(r["ratio_cdv_b_lc"], r["ratio_cdv_b_lc_se"]),
             cell(r["kappa_cdv_b"], r["kappa_cdv_b_se"])]
            for r in zero.to_dict("records")
        ]  # fmt: skip
        L += [md_table(["date", "group", "beta", "clipped mass", "CDV/CC", "CDV/LC", "kappa", "beta", "clipped mass", "CDV/CC", "CDV/LC", "kappa"], lines), ""]  # fmt: skip
        m = group_means(zero, ["ratio_lc_cc", "ratio_cdv_a_cc", "ratio_cdv_b_cc", "ratio_cdv_a_lc", "ratio_cdv_b_lc"])  # fmt: skip
        L += [md_table(["group", "quantity", "mean (sem) [n]"],
                       [[str(r["group"]), str(r["column"]), f"{cell(r['mean'], r['sem'])} [{r['n']}]"] for r in m.to_dict("records")]), ""]  # fmt: skip
    return L


# ---------------------------------------------------------------------------------------------
# the run
# ---------------------------------------------------------------------------------------------


def hms(seconds: float) -> str:
    s = int(max(seconds, 0))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


class Run:
    """A run: its folders, its commit and digest, the rows it has."""

    def __init__(self, args: argparse.Namespace, cfg: dict[str, Any]) -> None:
        self.args, self.cfg = args, cfg
        self.out = checked_dir(args.out, "--out")
        self.cache = checked_dir(args.cache, "--cache")
        self.rows, self.logs = self.out / "rows", self.out / "logs"
        self.betas = search_betas([float(b) for b in args.betas.split(",")])
        self.commit = code_version()
        self.digest = args_digest(cfg, args.tenor, args.budget, self.betas, args.g_max)
        self.table = self.out / f"cdv_stratified_{args.tenor}.parquet"
        self.report = self.out / f"report_cdv_stratified_{args.tenor}.md"

    def done(self, date: str) -> bool:
        r = read_row(self.rows / f"{date}.json")
        return bool(
            r is not None
            and r.get("status") in ("ok", "check")
            and r.get("git_commit") == self.commit
            and r.get("args_digest") == self.digest
        )

    def command(self, date: str) -> list[str]:
        a = self.args
        cmd = [
            sys.executable, str(Path(__file__).resolve()), "--date", date, "--tenor", a.tenor, "--budget", a.budget,
            "--betas", a.betas, "--g-max", repr(float(a.g_max)), "--out", str(self.out), "--cache", str(self.cache),
            *(["--fixed", a.fixed] if a.fixed else []),
            "--model-s", str(a.model_s),
        ]  # fmt: skip
        return [*cmd, "--smoke", str(a.smoke)] if a.smoke is not None else cmd

    def write_outputs(self, selection: pd.DataFrame, selection_text: str) -> int:
        frame = build_table(self.rows, selection, self.digest)
        if frame.empty:
            return 0
        tmp = self.table.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(self.table)
        b = self.cfg["budgets"][self.args.budget]
        meta = {"tenor": self.args.tenor, "budget": self.args.budget, "commit": self.commit, "digest": self.digest,
                "g_max": float(self.args.g_max), "betas": self.betas, "selection": selection_text, "screen": dict(self.cfg["screen"]),
                "smoke": self.args.smoke is not None,
                "counts": f"{b['n_particles']} particles, {b['n_paths']} pricing paths, {b['companion_paths']} paths of the companion's fit"}  # fmt: skip
        self.report.write_text(report(frame, meta))
        return len(frame)


def run_dates(run: Run, selection: pd.DataFrame, selection_text: str, todo: Sequence[str], workers: int, threads: int) -> None:  # fmt: skip
    """Run ``todo`` on ``workers`` processes of this script (``--date``); one log line per date."""
    env = {**os.environ, "NUMBA_NUM_THREADS": str(threads)}
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[v] = "1"
    group = dict(zip(selection["date"], selection["group"], strict=True))
    queue = list(todo)
    running: dict[str, tuple[subprocess.Popen[bytes], float, Any]] = {}
    t_start = time.perf_counter()
    n_done = 0
    while queue or running:
        while queue and len(running) < workers:
            d = queue.pop(0)
            fh = (run.logs / f"{d}.log").open("wb")
            proc = subprocess.Popen(run.command(d), stdout=fh, stderr=subprocess.STDOUT, env=env)
            running[d] = (proc, time.perf_counter(), fh)
        time.sleep(1.0)
        for d, (proc, t0, fh) in list(running.items()):
            if proc.poll() is None:
                continue
            fh.close()
            del running[d]
            seconds = time.perf_counter() - t0
            r = read_row(run.rows / f"{d}.json")
            if r is None or r.get("args_digest") != run.digest or r.get("git_commit") != run.commit:
                # the process died before its row: the row is written here
                r = {"date": d, "tenor": run.args.tenor, "budget": run.args.budget, "git_commit": run.commit,
                     "args_digest": run.digest, "status": "failed", "seconds_total": seconds,
                     "reason": f"process exited with code {proc.returncode} and no row"}  # fmt: skip
                write_row(run.rows / f"{d}.json", r)
            n_done += 1
            n_rows = run.write_outputs(selection, selection_text)
            elapsed = time.perf_counter() - t_start
            left = (len(todo) - n_done) * elapsed / n_done
            if r["status"] == "failed":
                detail = f"FAILED: {r['reason'][:200]}"
            else:
                star = "a" if r["beta_star"] > 0 else None
                at_star = r[f"ratio_cdv_{star}_cc"] if star else r["ratio_lc_cc"]
                at_star_se = r[f"ratio_cdv_{star}_cc_se"] if star else r["ratio_lc_cc_se"]
                detail = (
                    f"beta* {r['beta_star']:g} ({'reached' if r['reached'] else 'NOT reached'}; {r['n_calibrations']} calibrations), "
                    f"clipped mass LC {r['clip_inner_lc']:.4f} -> {r['clip_inner_cdv_a'] if star else r['clip_inner_lc']:.4f}; "
                    f"LC/CC {r['ratio_lc_cc']:.5f} ({r['ratio_lc_cc_se']:.5f}), CDV(beta*)/CC {at_star:.5f} ({at_star_se:.5f}), "
                    f"P_D_S/P_D {r['P_D_S_over_P_D']:.4f}" + (f"; {r['note']}" if r["note"] else "")
                )  # fmt: skip
            log.info(
                "%s %s [%d/%d] %s %s %.0f s (workers %d x %d threads); elapsed %s, left about %s; table %d rows; %s",
                time.strftime("%H:%M:%S"), d, n_done, len(todo), group.get(d, "given"), r["status"], seconds, workers, threads,
                hms(elapsed), hms(left), n_rows, detail,
            )  # fmt: skip


def child(args: argparse.Namespace) -> int:
    """One date (``--date``): its row, written to ``<out>/rows/<date>.json``."""
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("volsto").setLevel(logging.WARNING)
    logging.getLogger("volsto.multi.cdv").setLevel(logging.INFO)
    cfg = config_for(args.budget, args.smoke)
    run = Run(args, cfg)
    row = safe_row(args.date, args.tenor, cfg, budget=args.budget, betas=run.betas, g_max=float(args.g_max),
                   cache_root=run.cache, model_s=Path(args.model_s))  # fmt: skip
    row["args_digest"] = run.digest
    row["smoke"] = args.smoke
    write_row(run.rows / f"{args.date}.json", row)
    log.info("%s: %s %s in %.0f s; written %s", args.date, row["status"], row.get("reason", ""), row["seconds_total"], run.rows / f"{args.date}.json")  # fmt: skip
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--table", default=str(LC_OUT / "lcm_3m_dev_repair.parquet"), help="the pass whose clip_inner_max selects the dates")  # fmt: skip
    ap.add_argument("--n", type=int, default=20, help="dates per group")
    ap.add_argument("--betas", default="1,2,3,4,6,8,10,12")
    ap.add_argument("--g-max", type=float, default=3.0)
    ap.add_argument(
        "--fixed", default=None, help="a,b: no search - price CDV at these two betas on every date"
    )
    ap.add_argument("--budget", default="development", choices=("production", "development"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--threads", type=int, default=None, help="numba threads per process (default: cores // workers)")  # fmt: skip
    ap.add_argument("--out", default=str(LC_OUT / "cdv" / "stratified"))
    ap.add_argument("--cache", default=str(LC_OUT / "cache"), help="the M12 cache (the companion, the SVI fit records)")  # fmt: skip
    ap.add_argument("--model-s", default=None, help="the study's model S table (default: <LC root>/outputs/dispersion/model_s_<tenor>.parquet)")  # fmt: skip
    ap.add_argument("--tenor", default="3m")
    ap.add_argument("--dates", default=None, help="d1,d2,...: these dates instead of the selection")
    ap.add_argument("--smoke", type=int, default=None, help="particle and path counts of a test run")  # fmt: skip
    ap.add_argument("--report-only", action="store_true", help="rebuild the table and the report from the rows on disk")  # fmt: skip
    ap.add_argument("--date", default=None, help=argparse.SUPPRESS)  # one date: the worker
    args = ap.parse_args(argv)
    if args.fixed:
        global FIXED
        a, b = (float(x) for x in args.fixed.split(","))
        FIXED = (a, b)
    if args.model_s is None:
        args.model_s = str(LC_ROOT / "outputs" / "dispersion" / f"model_s_{args.tenor}.parquet")
    if args.date is not None:
        return child(args)
    cfg = config_for(args.budget, args.smoke)
    run = Run(args, cfg)
    run.rows.mkdir(parents=True, exist_ok=True)
    run.logs.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run.out / "stratified.log")],
    )  # fmt: skip
    table_path = Path(args.table)
    selection = select_dates(pd.read_parquet(table_path), args.n) if table_path.exists() else None
    selection_text = f"the {args.n} largest and the {args.n} smallest `clip_inner_max` among the priced rows of `{table_path}`"  # fmt: skip
    if args.dates is not None:
        given = [d.strip() for d in args.dates.split(",") if d.strip()]
        selection = given_selection(given, pd.read_parquet(table_path) if table_path.exists() else None, args.n)  # fmt: skip
        selection_text = f"the dates given on the command line ({', '.join(given)}); groups and clipped masses from `{table_path}` where it has them"  # fmt: skip
    if selection is None:
        raise FileNotFoundError(f"{table_path}: no selection table and no --dates")
    dates = selection["date"].tolist()
    todo = [] if args.report_only else [d for d in dates if not run.done(d)]
    cores = os.cpu_count() or 1
    workers = max(min(args.workers, len(todo)), 1)
    threads = args.threads if args.threads is not None else max(cores // max(args.workers, 1), 1)
    log.info(
        "=== %s stratified test %s (%s budget%s): %d dates (%s), %d to run; commit %s, arguments %s; betas %s, g_max %g; "
        "%d workers x %d threads on %d cores; rows %s; cache %s; screen %s",
        time.strftime("%Y-%m-%d %H:%M:%S"), args.tenor, args.budget, "" if args.smoke is None else f", SMOKE {args.smoke}",
        len(dates), ", ".join(f"{g} {n}" for g, n in selection["group"].value_counts().items()), len(todo), run.commit, run.digest,
        ",".join(f"{b:g}" for b in run.betas), args.g_max, workers, threads, cores, run.rows, run.cache, cfg["screen"],
    )  # fmt: skip
    t0 = time.perf_counter()
    run_dates(run, selection, selection_text, todo, workers, threads)
    n_rows = run.write_outputs(selection, selection_text)
    statuses: dict[str, int] = {}
    for d in dates:
        r = read_row(run.rows / f"{d}.json")
        key = "missing" if r is None or r.get("args_digest") != run.digest else str(r.get("status"))
        statuses[key] = statuses.get(key, 0) + 1
    log.info(
        "=== %s stratified test finished in %s: %d rows in %s; statuses %s; report %s",
        time.strftime("%Y-%m-%d %H:%M:%S"), hms(time.perf_counter() - t0), n_rows, run.table, statuses, run.report,
    )  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
