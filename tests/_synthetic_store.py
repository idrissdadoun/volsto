"""Synthetic, schema-complete results store for the viewer tests (M9 Parts 2–3).

:func:`make_synthetic_store` writes — with **no calibration and no Monte Carlo**, from a seeded
generator — a :class:`~volsto.viewers.store.ResultsStore` holding three points on two surfaces
(the placeholder's LV point and its 1F ω = 1 point, and the SPX 2022-09-15 snapshot's marking
point at ``ssr_target 1.0 × skew_eps 0.10``), every table of the store (``products``,
``forward_smile``, ``forward_vols``, ``ssr``, ``varv``, ``risk`` in the light tier) with the
columns :mod:`volsto.viewers.precompute` writes and a ``_stderr`` twin on every Monte Carlo
column, a per-point manifest entry and one run record carrying the grid mapping and the grid
path :data:`GRID_PATH` (so the read API recovers the surface catalogue and names the grid the
store was built with).  The ``risk`` rows carry the naming the precompute stores — the
:class:`~volsto.risk.report.RiskReport` groups ``delta`` / ``gamma`` / ``fwd_var`` / ``skew``
with the names ``delta[<regime>]``, ``gamma[<regime>]``, ``fwd_var[<lo>-<hi>y]``,
``skew_T[<T>y]`` (:func:`volsto.viewers.precompute._risk_rows`), and ``forward_smile`` /
``forward_vols`` carry the ``beyond_horizon`` flag of the store (``t2`` beyond the calibration
horizon — false on this grid's 3y horizon, true on the toy grid's 1y one).  Point ids are the
real ids the
grid would give (the LV key of the surface, the leverage-cache key of the 1F spec, the marking
label); nothing is written to any leverage cache by :func:`make_synthetic_store`.
:func:`make_leverage_entry` writes one cache entry (``leverage.npz`` + ``diagnostics.json``)
from arrays for the tests that need the read API's leverage path.  Beside the store,
:func:`make_synthetic_outputs` writes an M8b task result (``m8b/A/<key>.json``, a summary table)
and the M7 marking tables (``m7/*.csv``) with the real headers, so the marking and hedging API
paths render against them.

Used by ``tests/test_viewers_api.py`` and the page tests; wall clock of the writer is a few
hundred milliseconds.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volsto.calibration.cache import spec_key
from volsto.calibration.diagnostics import CalibrationReport
from volsto.config import BergomiParams
from volsto.market.curves import ForwardCurve
from volsto.models.leverage import LeverageFunction
from volsto.studies.m4 import HEADLINE_STRIKES
from volsto.studies.m8b import TaskResult
from volsto.viewers.grid import (
    GridSpec,
    MarkingAxes,
    OneFactorAxes,
    ParticleSettings,
    PricingSettings,
    ProductFlags,
    RiskSettings,
    SurfaceSpec,
    enumerate_points,
    grid_mapping,
)
from volsto.viewers.store import PointResult, ResultsStore

SNAPSHOT_NAME = "spx_2022-09-15"
SNAPSHOT_PATH = "configs/surfaces/snapshots/spx_2022-09-15.yaml"
MARKING_KEY = f"marking:{SNAPSHOT_NAME}:ssr1:eps0.1"
#: The synthetic marking fit's parameters (the shape of the M7 SPX fit, not its values).
MARKING_PARAMS = BergomiParams(
    nu=1.9, theta=0.15, k1=8.7, k2=0.2, rho12=0.74, rho_SX1=-0.94, rho_SX2=-0.88
)
HEDGING_RUN_ID = "A__cliquet_1y__pricing__delta_only"
SSR_PILLARS = (0.25, 0.5, 1.0, 2.0, 3.0)
REGIMES = ("model", "sticky_strike", "sticky_moneyness", "sticky_skew", "sticky_local_vol")
CODE_TAG = "synthetic"
GIT_COMMIT = "0000000"
#: The grid path the run record carries — repository-relative, like the one
#: :mod:`volsto.viewers.precompute` writes; :func:`volsto.viewers.api.precompute_command` quotes
#: it, so the store names the grid it was built with and never the viewer's default grid.
GRID_PATH = "configs/grids/synthetic.yaml"


def synthetic_grid() -> GridSpec:
    """Placeholder (1F axes ν 0.5 × ρ −0.7 × κ 1.5) + the SPX 2022-09-15 snapshot (marking at
    1.0 × 0.10); production particle count, 3y horizon, light risk."""
    return GridSpec(
        name="synthetic",
        reference_spec="configs/studies/lsv_reference_2f.yaml",
        surfaces=(
            SurfaceSpec("placeholder", "placeholder", True, False, False),
            SurfaceSpec(SNAPSHOT_NAME, "snapshot", False, False, True, SNAPSHOT_PATH),
        ),
        particle=ParticleSettings(),
        products=ProductFlags(),
        pricing=PricingSettings(),
        risk=RiskSettings(),
        one_factor=OneFactorAxes((0.5,), (-0.7,), (1.5,)),
        marking=MarkingAxes((1.0,), (0.1,)),
        include_degenerate=False,
    )


def _products(rng: np.random.Generator, omega: float) -> pd.DataFrame:
    m4 = {
        "atm_vol": ("fwd ATM vol 1y→2y", "vol", "vol", 0.2146 - 0.0064 * omega),
        "vs_vol": ("fwd VS 1y→2y", "fair vol", "vol", 0.2525),
        "volswap_vol": ("fwd vol swap 1y→2y", "fair vol", "vol", 0.2254 - 0.002 * omega),
        "cliquet_1y": ("cliquet 1y", "price", "% notional", 1.199 + 0.13 * omega),
        "cliquet_2y": ("cliquet 2y", "price", "% notional", 0.749 + 0.2 * omega),
        "upvar_100": ("up-var 1y B=100%", "fair vol", "vol", 0.1504),
        "downvar_100": ("down-var 1y B=100%", "fair vol", "vol", 0.3492),
        "kovar_110": ("KO var 1y B=110%", "fair vol", "vol", 0.2907),
        "kovar_110_p_ko": ("KO var 1y B=110%", "P(KO)", "probability", 0.657),
        "vko_30": ("VKO 12m 100% put @30%", "price", "% notional", 2.437),
        "vko_30_p_ko": ("VKO 12m 100% put @30%", "P(KO)", "probability", 0.175),
    }
    rows = [
        {
            "product": p,
            "quantity": q,
            "key": k,
            "value": v,
            "value_stderr": abs(v) * 0.002 + 1e-5,
            "unit": u,
        }
        for k, (p, q, u, v) in m4.items()
    ]
    for prod in ("autocall 3y", "phoenix 3y"):
        cells = {
            "price": (96.5 + 0.4 * omega + rng.normal(0, 0.01), "% notional"),
            "expected_life": (1.8 - 0.05 * omega, "years"),
            "p_ki": (0.08 + 0.01 * omega, "probability"),
        }
        for c, (v, u) in cells.items():
            rows.append(
                {
                    "product": prod,
                    "quantity": c,
                    "key": f"{prod}:{c}",
                    "value": float(v),
                    "value_stderr": abs(float(v)) * 0.001 + 1e-4,
                    "unit": u,
                }
            )
    return pd.DataFrame(rows)


def _smiles(omega: float, horizon: float) -> pd.DataFrame:
    rows = []
    for t1, t2 in ((1.0, 2.0), (2.0, 3.0)):
        for kk in HEADLINE_STRIKES:
            lm = math.log(kk)
            iv = 0.215 - 0.006 * omega - 0.25 * lm + 0.8 * lm * lm
            rows.append(
                {
                    "t1": t1,
                    "t2": t2,
                    "beyond_horizon": bool(t2 > horizon + 1e-9),
                    "strike_moneyness": kk,
                    "log_moneyness": lm,
                    "cp": 1.0 if kk >= 1.0 else -1.0,
                    "iv": iv,
                    "iv_stderr": 0.0005,
                    "price": 8.0 * iv,
                    "price_stderr": 0.02,
                }
            )
    return pd.DataFrame(rows)


def _forward_vols(omega: float, horizon: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "t1": t1,
                "t2": t2,
                "beyond_horizon": bool(t2 > horizon + 1e-9),
                "fwd_atm_vol": 0.2146 - 0.0064 * omega,
                "fwd_atm_vol_stderr": 0.00045,
                "fwd_vs": 0.2525,
                "fwd_vs_stderr": 0.0002,
                "fwd_volswap": 0.2254 - 0.002 * omega,
                "fwd_volswap_stderr": 0.00012,
            }
            for t1, t2 in ((1.0, 2.0), (2.0, 3.0))
        ]
    )


def _ssr(omega: float, ssr_target: float, naked: dict[float, float] | None) -> pd.DataFrame:
    rows = []
    for T in SSR_PILLARS:
        rows.append(
            {
                "T": T,
                "ssr_lsv": 1.6 - 0.15 * omega - 0.1 * T,
                "ssr_lsv_stderr": 0.012,
                "skew_lsv": -0.3 / math.sqrt(T),
                "skew_lsv_stderr": 0.004,
                "slope": -0.5 / math.sqrt(T),
                "slope_stderr": 0.006,
                "atmf_vol": 0.21,
                "atmf_vol_stderr": 0.0004,
                "eps": 0.05,
                "ssr_naked_first_order": (naked or {}).get(T, math.nan),
                "ssr_target": ssr_target,
            }
        )
    return pd.DataFrame(rows)


def _varv() -> pd.DataFrame:
    rows = []
    for T in (1.0, 3.0):
        for term, v in (
            ("mean", 0.044),
            ("var_total", 1.2e-4),
            ("var_sv", 0.9e-4),
            ("var_leverage", 0.2e-4),
            ("cov_cross", 0.1e-4),
            ("var_explained", 1.1e-4),
        ):
            rows.append({"T": T, "term": term, "value": v, "value_stderr": v * 0.03, "kind": "mc"})
        for term, v in (
            ("mean_sv_closed", 0.044),
            ("var_sv_closed", 0.9e-4),
            ("var_sv_discrete", 0.89e-4),
        ):
            rows.append(
                {"T": T, "term": term, "value": v, "value_stderr": 0.0, "kind": "closed_form"}
            )
    return pd.DataFrame(rows)


def _risk(rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def add(product: str, group: str, name: str, value: float, unit: str, **extra: Any) -> None:
        row = {
            "product": product,
            "tier": "light",
            "group": group,
            "name": name,
            "value": value,
            "value_stderr": abs(value) * 0.02 + 1e-4,
            "unit": unit,
            "size": 0.01,
            "scheme": "central",
            "states": "",
            "T": math.nan,
            "bucket": "",
            "regime": "",
            "variant": "",
            "key": f"{group}:{name}",
        }
        row.update(extra)
        rows.append(row)

    for product in ("autocall 3y", "cliquet 1y"):
        for regime in REGIMES:
            add(
                product,
                "delta",
                f"delta[{regime}]",
                float(rng.normal(0.3, 0.05)),
                "per spot",
                regime=regime,
            )
            add(
                product,
                "gamma",
                f"gamma[{regime}]",
                float(rng.normal(-0.01, 0.002)),
                "per spot²",
                regime=regime,
            )
        edges = np.linspace(0.0, 3.0, 21)
        for lo, hi in itertools.pairwise(edges):
            add(
                product,
                "fwd_var",
                f"fwd_var[{lo:g}-{hi:g}y]",
                float(rng.normal(0.02, 0.005)),
                "per 1% variance",
                bucket=f"{lo:g}-{hi:g}y",
            )
        for T in (0.25, 1.0, 3.0):
            add(product, "skew", f"skew_T[{T:g}y]", float(rng.normal(-0.05, 0.01)), "per vp", T=T)
    return pd.DataFrame(rows)


def _row(
    point: Any, *, model: BergomiParams | None, omega: float, grid: GridSpec
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "label": point.label,
        "surface": point.surface,
        "mode": point.mode,
        "status": "ok" if point.mode != "marking" else "binding",
    }
    for k in ("nu", "theta", "k1", "k2", "rho12", "rho_SX1", "rho_SX2"):
        row[k] = float(getattr(model, k)) if model is not None else math.nan
    for axis, col in (
        ("nu", "axis_nu"),
        ("rho", "axis_rho"),
        ("kappa", "axis_kappa"),
        ("ssr_target", "ssr_target"),
        ("skew_eps", "skew_eps"),
    ):
        row[col] = float(point.axes.get(axis, math.nan))
    row["cache_key"] = point.cache_key or ""
    row["n_particles"] = grid.particle.n_particles
    row["horizon"] = grid.particle.horizon
    row["calibration_seconds"] = math.nan if model is None else 138.0 + 10.0 * omega
    row["mean_abs_L_minus_1"] = math.nan if model is None else 0.05 + 0.02 * omega
    row["max_abs_error_vp"] = math.nan if model is None else 0.08
    row["max_z"] = math.nan if model is None else 1.9
    row["calibrated_this_run"] = False
    row["risk_tier"] = "light"
    row["pricing_n_paths"] = grid.pricing.n_paths
    row["pricing_seed"] = grid.pricing.seed
    row["wall_seconds"] = 430.0
    row["wall_calibration_s"] = 0.5
    row["wall_pricing_s"] = 176.4
    row["wall_analytics_s"] = 124.6
    row["wall_risk_s"] = 128.5
    row["git_commit"] = GIT_COMMIT
    row["code_tag"] = CODE_TAG
    row["created_utc"] = "2026-09-15T00:00:00+00:00"
    return row


def make_synthetic_store(root: str | Path, *, seed: int = 7) -> dict[str, Any]:
    """Write the store under ``root`` and return ``{"grid", "point_ids", "labels", "root"}``."""
    rng = np.random.default_rng(seed)
    grid = synthetic_grid()
    store = ResultsStore(root)
    points = enumerate_points(grid)
    by_mode = {(p.surface, p.mode): p for p in points}
    lv = by_mode[("placeholder", "lv")]
    one = by_mode[("placeholder", "one_factor")]
    marking = by_mode[(SNAPSHOT_NAME, "marking")]
    spec_marking = dataclasses.replace(marking.spec, model=MARKING_PARAMS)
    marking = dataclasses.replace(
        marking,
        spec=spec_marking,
        cache_key=spec_key(spec_marking),
        status="binding",
        resolved=True,
    )
    naked = {0.25: 1.35, 0.5: 1.3, 1.0: 1.2, 2.0: 1.1, 3.0: 1.05}
    written: list[str] = []
    labels: dict[str, str] = {}
    for point, model, omega in (
        (lv, None, 0.0),
        (one, one.spec.model, 1.0),
        (marking, MARKING_PARAMS, 3.8),
    ):
        row = _row(point, model=model, omega=omega, grid=grid)
        tables: dict[str, pd.DataFrame] = {
            "products": _products(rng, omega),
            "forward_smile": _smiles(omega, grid.particle.horizon),
            "forward_vols": _forward_vols(omega, grid.particle.horizon),
            "ssr": _ssr(
                omega,
                float(point.axes.get("ssr_target", math.nan)),
                naked if point.mode == "marking" else None,
            ),
            "risk": _risk(rng),
        }
        if model is not None:
            tables["varv"] = _varv()
        if point.mode == "marking":
            row["fit_status"] = "binding"
            row["fit_messages"] = "nu at cap 3.5 | rho12 collapse guard"
            row["fit_mean_skew_gap"] = 0.031
            row["fit_wall_seconds"] = 1.9
            row["skew_gap_T_s@1y"] = 0.1
            row["skew_gap_T_l@3y"] = -0.1
        manifest = {
            "label": point.label,
            "surface": point.surface,
            "mode": point.mode,
            "risk_tier": "light",
            "n_particles": grid.particle.n_particles,
            "horizon": grid.particle.horizon,
            "pricing": {"n_paths": grid.pricing.n_paths, "seed": grid.pricing.seed},
            "git_commit": GIT_COMMIT,
            "code_tag": CODE_TAG,
            "calibrated_this_run": False,
            "cache_key": point.cache_key,
            "status": row["status"],
            "wall": {
                "calibration": 0.5,
                "pricing": 176.4,
                "analytics": 124.6,
                "risk": 128.5,
                "total": 430.0,
            },
            "created_utc": row["created_utc"],
        }
        store.write_point(PointResult(point.id, row, tables, manifest))
        written.append(point.id)
        labels[point.id] = point.label
    store.write_run(
        {
            "argv": ["--grid", GRID_PATH, "--store", "<store>"],
            "grid_path": GRID_PATH,
            "grid": grid_mapping(grid),
            "shard": "1/1",
            "workers": 1,
            "risk_tier": "light",
            "n_particles": grid.particle.n_particles,
            "pricing": {"n_paths": grid.pricing.n_paths, "seed": grid.pricing.seed},
            "code_tag": CODE_TAG,
            "git_commit": GIT_COMMIT,
            "host": "synthetic",
            "points_total": len(points),
            "points_in_shard": len(points),
            "points_computed": [],
            "resume": False,
            "wall_seconds": 0.0,
            "created_utc": "2026-09-15T00:00:00+00:00",
        }
    )
    return {"grid": grid, "point_ids": written, "labels": labels, "root": Path(root)}


# --------------------------------------------------------------------------------------------
# leverage cache entry (arrays only)
# --------------------------------------------------------------------------------------------


def make_leverage_entry(
    cache_root: str | Path, key: str, *, n_particles: int = 400_000, seed: int = 2024
) -> Path:
    """Write ``<cache_root>/<key>/{leverage.npz, diagnostics.json}`` from arrays — **no
    calibration and no Monte Carlo**, the same construction as
    ``tests/test_viewers_pages_a.py::_write_leverage_entry``.

    The :class:`~volsto.calibration.diagnostics.CalibrationReport` carries the columns
    :func:`~volsto.calibration.diagnostics.reprice_surface` writes (``T, k, K, cp, price,
    price_stderr, model_vol, target_vol, error_vp, stderr_vp``), so the success path of
    :func:`volsto.viewers.api.get_leverage` — the error map and its ``stderr_vp`` →
    ``error_vp_stderr`` rename — is exercised by a test that calibrates nothing."""
    entry = Path(cache_root) / key
    entry.mkdir(parents=True, exist_ok=True)
    times = np.linspace(0.05, 3.0, 24)
    k_grid = np.linspace(-2.0, 2.0, 41)
    values = 1.0 + 0.15 * np.sin(k_grid)[None, :] * np.exp(-times)[:, None]
    lev = LeverageFunction(
        times,
        k_grid,
        values,
        ForwardCurve.flat(100.0, 0.02, 0.0),
        {
            "wall_time": 2.2,
            "code_tag": CODE_TAG,
            "git_commit": GIT_COMMIT,
            "n_particles": n_particles,
            "seed": seed,
        },
    )
    lev.save(entry / "leverage.npz")
    rows = [
        {
            "T": T,
            "k": k,
            "K": 100.0 * math.exp(k),
            "cp": 1 if k >= 0 else -1,
            "price": 5.0,
            "price_stderr": 0.01,
            "model_vol": 0.2 + 0.05 * k * T / 100.0,
            "target_vol": 0.2,
            "error_vp": 0.05 * k * T,
            "stderr_vp": 0.02,
        }
        for T in (1 / 12, 0.25, 0.5, 1.0, 2.0, 3.0)
        for k in (-0.2, -0.1, 0.0, 0.1, 0.2)
    ]
    CalibrationReport(pd.DataFrame(rows), pd.DataFrame(), n_particles, seed, 130.0).save(
        entry / "diagnostics.json"
    )
    return entry


# --------------------------------------------------------------------------------------------
# M7 / M8b outputs
# --------------------------------------------------------------------------------------------


def make_synthetic_outputs(root: str | Path) -> dict[str, Path]:
    """Write ``m8b/A/<HEDGING_RUN_ID>.json`` + ``m8b/m8b_table_A.csv`` and the four M7 tables
    (real headers, synthetic rows) under ``root``; returns the paths."""
    root = Path(root)
    out: dict[str, Path] = {}
    m8b = root / "m8b" / "A"
    m8b.mkdir(parents=True, exist_ok=True)
    task = TaskResult(
        key=HEDGING_RUN_ID,
        study="A",
        product="cliquet 1y",
        world="pricing",
        strategy="delta only",
        frequency="monthly",
        n_paths_pricing=20_000,
        n_paths_world=20_000,
        n_particles=800_000,
        n_dates=12,
        value_0=(1.55, 0.004),
        mean=(-0.05, 0.05),
        std=(7.1, 0.04),
        quantiles={
            "q01": (-19.0, 0.4),
            "q05": (-11.0, 0.2),
            "q50": (0.1, 0.05),
            "q95": (12.0, 0.2),
            "q99": (20.0, 0.5),
        },
        zero_cost_mean=(-0.05, 0.05),
        regimes=[
            {
                "regime": f"realised vol {lab}",
                "n": 6667,
                "mean": m,
                "stderr": 0.03,
                "std": 2.4,
                "realised_vol_mean": rv,
            }
            for lab, m, rv in (("low", 6.1, 0.15), ("mid", -0.2, 0.2), ("high", -6.0, 0.27))
        ],
        attribution=[
            {"component": "product delta x dS", "mean": 2.44, "stderr": 0.12},
            {"component": "gamma x dS²", "mean": -1.1, "stderr": 0.05},
            {"component": "residual", "mean": -1.39, "stderr": 0.1},
        ],
        recal_total=(0.0, 0.0),
        settings={"schedule": "monthly", "pricing": "2F", "world": "pricing"},
        budget={"pricing_simulations": 24.0, "wall_seconds": 3.4, "n_dates": 12.0},
        wall_seconds=3.4,
        notes=["synthetic run of tests/_synthetic_store.py"],
    )
    p = m8b / f"{HEDGING_RUN_ID}.json"
    p.write_text(json.dumps(task.to_dict(), indent=1))
    out["hedging_run"] = p
    table_a = pd.DataFrame(
        [
            {
                "pricing": "2F",
                "product": "cliquet 1y",
                "strategy": "delta only",
                "status": "ok",
                "std": 7.1,
                "std_se": 0.04,
                "desk_mean": 0.05,
                "mean_se": 0.05,
                "unit": "% of notional",
                "n_paths": 20000,
                "wall_s": 3.4,
                "reason": "",
                "rank": 1,
            }
        ]
    )
    pa = root / "m8b" / "m8b_table_A.csv"
    table_a.to_csv(pa, index=False)
    out["hedging_table_A"] = pa

    m7 = root / "m7"
    m7.mkdir(parents=True, exist_ok=True)
    fits = pd.DataFrame(
        [
            {
                "surface": "spx",
                "ssr_target": 1.0,
                "skew_eps": 0.1,
                "status": "binding",
                "nu": 1.94,
                "theta": 0.15,
                "k1": 8.72,
                "rho_SX1": -0.939,
                "rho_SX2": -0.875,
                "rho12": 0.74,
                "active": "skew T=1 (1+0.1);skew T=3 (1-0.1)",
                "bounds": "",
                "T_T_s": 1.0,
                "skew_gap_T_s": 0.1,
                "T_T_l": 3.0,
                "skew_gap_T_l": -0.1,
                "short_gap@0.0833y": 0.676,
                "short_gap@0.25y": 0.568,
                "short_gap@0.5y": 0.334,
                "svc_err_max": 0.132,
                "volvar_err_max": 0.0055,
                "corr_model_mean": -0.902,
                "corr_target_mean": -0.902,
                "rho12_collapse": False,
                "fit_seconds": 1.8,
                "stage3_assertion": "FAIL",
                "stage3_max_engine_bias_svc": 0.171,
                "stage3_max_engine_bias_volvar": 0.29,
                "stage3_max_gap_svc_reported": 0.214,
                "stage3_max_gap_volvar_reported": 0.297,
                "mean_abs_L_minus_1": 0.156,
                "recalibrated": False,
                "calibration_seconds": 0.0,
                "stage3_seconds": 57.5,
                "ssr_lsv@0.25y": 1.333,
                "ssr_lsv_se@0.25y": 0.0147,
                "ssr_lsv@1y": 1.133,
                "ssr_lsv_se@1y": 0.0123,
                # the stage-3 forward / spot 90/110 skew ratio, the real header of
                # scripts/m7_p1_marking.py: one '<name>@<window>' + '<name>_se@<window>' pair
                # per forward-start window (a Monte Carlo measurement, never exact)
                "fwd/spot@1y-into-1y": 1.41,
                "fwd/spot_se@1y-into-1y": 0.05,
                "fwd/spot@2y-into-1y": 1.22,
                "fwd/spot_se@2y-into-1y": 0.06,
            }
        ]
    )
    pf = m7 / "p1_marking_fits.csv"
    fits.to_csv(pf, index=False)
    out["marking_fits"] = pf
    rot_rows = []
    for greek in (
        "lv_rotation",
        "usual",
        "recalibrated",
        "fee_usual",
        "fee_shadow",
        "desk_pnl_shadow",
    ):
        rot_rows.append(
            {
                "surface": "spx",
                "policy": "sabr_linked",
                "greek": greek,
                "of": "LV price" if greek == "lv_rotation" else "P1 price",
                "per_rota": -0.0009,
                "per_rota_se": 4.8e-5,
                "per_vp_90_110_0.5y": -0.0009 / 0.56,
                "per_vp_90_110_1y": -0.0009 / 0.4,
                "p1_level": 0.928,
                "p1_level_se": 0.00045,
                "lv_level": 0.921,
                "lv_level_se": 0.00047,
                "fee": 0.0073,
                "fee_se": 0.00027,
            }
        )
    pr = m7 / "p1_marking_shadow_rotation.csv"
    pd.DataFrame(rot_rows).to_csv(pr, index=False)
    out["shadow_rotation"] = pr
    binding = pd.DataFrame(
        [
            {
                "fit": "unclamped",
                "ssr_target": s,
                "ssr_measure": "lsv",
                "skew_weight": w,
                "ssr_requested": s,
                "ssr_first_target": s,
                "ssr_fitted": s,
                "clamped": False,
                "first_order_valid": True,
                "nu": 1.9,
                "theta": 0.23,
                "k1": 20.0,
                "k2": 0.2,
                "rho12": 0.61,
                "rho_SX1": -0.8,
                "rho_SX2": -0.75,
                "k1_at_bound": True,
                "nu_limit_binding": False,
                "nu_above_flag": False,
                "objective_first": 2.1,
                "ssr_achieved_naked_vs_market_mean": 1.7,
                "ssr_naked_own_mean": 1.66,
                "ssr_achieved_lsv_mean": 1.57,
                "mean_skew_gap": 0.117,
                "max_skew_ratio": 1.23,
                "ssr_lsv_0.25": 1.6,
                "skew_market_0.25": -0.46,
                "skew_naked_0.25": -0.57,
            }
            for s in (0.75, 1.0)
            for w in (10.0, 100.0)
        ]
    )
    for name in ("spx_soft_skew.csv", "skew_tradeoff.csv"):
        pb = m7 / name
        binding.to_csv(pb, index=False)
        out[name] = pb
    return out
