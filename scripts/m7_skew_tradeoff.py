"""M7 Part 3 study: the skew-weight trade-off of the soft-skew break-even fitter on the reference
SSVI (owner's M7 Part 3 redesign, trade-off study; fitter and formulas in the
:mod:`volsto.calibration.fit_2f` module docstring; Bergomi ch. 12 eq. 12.52 for the LSV SSR).

Setting: the reference SSVI (``configs/surfaces/reference_ssvi.yaml``, identical to the market /
surface sections of ``configs/studies/lsv_reference_2f.yaml``, checked at start-up), marking
mode, pillars 3M / 6M / 1Y / 2Y / 3Y, ``k2 = 0.2`` fixed, the fitter's defaults (``ssr_measure
"auto"`` → lsv on a surface, ``nu_cap`` 2.5, ``volvar_target "achieved"``) unless
``--ssr-measure`` is given.  For every ``ssr_target`` of ``--targets`` and
every skew weight of ``--weights`` (default :data:`~volsto.calibration.fit_2f.
DEFAULT_TRADEOFF_WEIGHTS`, 100 / 10 / 1 / 0.1, skew-tight to skew-loose):

1. the **unclamped** fit at the requested target (``clamp_to_attainable=False``: the trade-off at
   the requested target is what the study measures), with the first-order leverage proxy;
2. the attainable SSR floor / ceiling (:func:`~volsto.calibration.fit_2f.attainable_ssr`: the
   lowest / highest first-order mean SSR over the scan, with the scan target attaining it, the
   scan-edge flag and the binding limits) and the owner's floor / ceiling message with the
   clamped first-order refit (the fitter's default behaviour; first order only, no leverage for
   the clamped parameters).  Floors, achieved SSRs and messages are **first-order estimates**;
   the stage-3 numerical LSV SSR is printed next to them and is the truth (at ν = 5 the review
   measured the first-order lsv SSR 10–27% low);
3. the study spec ``configs/studies/m7_skew_tradeoff/ssr{target:g}_w{weight:g}.yaml``
   (:func:`~volsto.calibration.fit_2f.write_tradeoff_spec`: a ``spec`` section — the
   :class:`~volsto.config.CalibrationSpec` of ``lsv_reference_2f.yaml`` with the fitted model and
   ``--n-particles`` — and a ``fit`` section with the provenance read back by
   ``tests/test_fit_2f.py::test_tradeoff_study_cached_leverages``); the round trip
   ``load_yaml(path, CalibrationSpec, section="spec") == spec`` (same cache key) and the
   ``fit`` section (config, break-even parameters) are checked after writing;
4. stage 3 (unless ``--no-stage3``): the leverage is **calibrated into the leverage cache**
   (``LeverageCache('cache').get_or_calibrate(spec, allow_calibrate=True)``, 2·10⁵ particles by
   default — the development count —, ``ParticleConfig(n_particles, horizon=3.0)`` and the
   default ``SimConfig()`` schedule) and the returned LSV is passed as ``Stage3Inputs(model=...)``
   so that stage 3 does not calibrate a second time; pricing ``SimConfig(n_paths=100000,
   chunk_size=100000, seed=7)``: the LSV's numerical SSR per pillar with standard errors, the
   naked kernel's mixing skew, the simulated ``SpotVolCovar`` / ``VolVar`` of the LSV against the
   targets, the actual mean ``|L − 1|`` (:func:`~volsto.calibration.fit_2f.
   mean_abs_leverage_deviation`, ``|k| ≤ 2 sd`` per time);
5. particle noise (unless ``--no-noise`` or ``--no-stage3``): for each target the tightest and the
   loosest weight's spec is also calibrated into the cache at the particle seed
   :data:`~volsto.calibration.fit_2f.TRADEOFF_NOISE_SEED`; the two-seed difference of the mean
   ``|L − 1|`` is reported and gives the test's monotonicity slack (three standard deviations).

Outputs: ``<out>/skew_tradeoff.md`` and ``<out>/skew_tradeoff.csv`` (one row per (target,
weight): book and break-even parameters with standard errors, achieved SSR under the three
first-order measures (naked vs market, naked own, lsv) mean and per pillar, the numerical LSV
SSR per pillar with standard errors, the mean relative naked-skew gap (first order and mixing),
mean ``|L − 1|`` actual and proxy, simulated ``SpotVolCovar`` / ``VolVar`` against targets with
standard errors, floor / ceiling, message, calibration seconds, wall clock, recalibrated flag).
Every Monte Carlo number carries its standard error; the particle noise of the calibrated
``|L − 1|`` is the two-seed difference of step 5.

Full run (run under ``caffeinate -i``; up to 12 calibrations of about 60 s plus 8 stage-3 pricings
of about 150 s, 30–40 min)::

    caffeinate -i .venv/bin/python scripts/m7_skew_tradeoff.py

Smoke test (first order only, outputs and specs to a scratch directory)::

    .venv/bin/python scripts/m7_skew_tradeoff.py --no-stage3 --targets 1.0 --weights 1 \\
        --out /tmp/x --spec-dir /tmp/x/specs

The helpers :func:`fit_row`, :func:`stage3_row` and :func:`markdown_table` are reused by
``scripts/m7_spx_soft_skew.py`` (imported from the scripts directory).
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volsto.calibration.cache import LeverageCache, build_market, spec_key
from volsto.calibration.fit_2f import (
    DEFAULT_TRADEOFF_WEIGHTS,
    SSR_MEASURES,
    TRADEOFF_NOISE_SEED,
    AttainableSSR,
    BreakEvenFitConfig,
    FitResult,
    Stage3Inputs,
    Stage3Report,
    attainable_ssr,
    fit_2f_marking,
    load_tradeoff_specs,
    mean_abs_leverage_deviation,
    stage3_validation,
    write_tradeoff_spec,
)
from volsto.config import CalibrationSpec, ParticleConfig, SimConfig, SSVIConfig, load_yaml

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SURFACE = ROOT / "configs" / "surfaces" / "reference_ssvi.yaml"
BASE_SPEC = ROOT / "configs" / "studies" / "lsv_reference_2f.yaml"
PILLARS = (0.25, 0.5, 1.0, 2.0, 3.0)
K2 = 0.2


def _floats(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    """A GitHub markdown table of ``df`` (floats rounded to ``digits``; no tabulate dependency)."""

    def fmt(v: Any) -> str:
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "no"
        if isinstance(v, (float, np.floating)):
            return "nan" if not math.isfinite(float(v)) else f"{float(v):.{digits}g}"
        return str(v)

    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.to_numpy()) + " |")
    return "\n".join(lines)


def fit_row(
    r: FitResult, att: AttainableSSR | None, clamped: FitResult | None = None
) -> dict[str, Any]:
    """First-order columns of one study row: parameters with standard errors, the three achieved
    SSRs (mean and per pillar), skew gaps, the leverage proxy, floor / ceiling and the owner's
    message of the clamped fit (``clamped``; the message of ``r`` itself when None)."""
    p, b, f, s = r.params, r.breakeven, r.first, r.second
    tab = r.table
    row: dict[str, Any] = {
        "ssr_measure": r.config.ssr_measure,
        "skew_weight": float(r.config.skew_weight),
        "ssr_requested": float(np.mean(r.ssr_requested)),
        "ssr_first_target": float(tab["ssr_target"].mean()),
        "ssr_fitted": float(tab["ssr_fitted"].mean()),
        "clamped": bool(np.any(np.abs(tab["ssr_target"].to_numpy() - r.ssr_requested) > 1e-12)),
        "first_order_valid": bool(f.first_order_valid),
        "nu": p.nu,
        "theta": p.theta,
        "k1": p.k1,
        "k2": p.k2,
        "rho12": p.rho12,
        "rho_SX1": p.rho_SX1,
        "rho_SX2": p.rho_SX2,
        "be_k1": b.k1,
        "be_k1_se": f.k1_se,
        "be_lambda1": b.lambda1,
        "be_lambda1_se": f.lambda1_se,
        "be_lambda2": b.lambda2,
        "be_lambda2_se": f.lambda2_se,
        "be_omega1": b.omega1,
        "be_omega1_se": s.stderr.get("omega1", float("nan")),
        "be_omega2": b.omega2,
        "be_omega2_se": s.stderr.get("omega2", float("nan")),
        "be_chi": b.chi,
        "be_chi_se": s.stderr.get("chi", float("nan")),
        "k1_at_bound": bool(f.k1_at_bound),
        "nu_limit_binding": bool(f.nu_limit_binding),
        "nu_above_flag": bool(p.nu > r.config.nu_flag),
        "second_bound_flags": ";".join(s.bound_flags),
        "objective_first": f.objective,
        "objective_first_ssr": f.objective_ssr,
        "objective_first_skew": f.objective_skew,
        "objective_second": s.objective,
        "ssr_achieved_naked_vs_market_mean": float(tab["ssr_achieved_naked_vs_market"].mean()),
        "ssr_naked_own_mean": float(tab["ssr_naked_own"].mean()),
        "ssr_achieved_lsv_mean": float(tab["ssr_achieved_lsv"].mean()),
        "mean_skew_gap": r.mean_skew_gap,
        "max_skew_ratio": float(np.max(tab["skew_naked"] / tab["skew_market"])),
    }
    for _, x in tab.iterrows():
        T = f"{float(x['T']):g}"
        row[f"ssr_naked_vs_market_{T}"] = float(x["ssr_achieved_naked_vs_market"])
        row[f"ssr_naked_own_{T}"] = float(x["ssr_naked_own"])
        row[f"ssr_lsv_{T}"] = float(x["ssr_achieved_lsv"])
        row[f"skew_market_{T}"] = float(x["skew_market"])
        row[f"skew_naked_{T}"] = float(x["skew_naked"])
        row[f"skew_gap_{T}"] = float(x["skew_gap_rel"])
    px = r.leverage_proxy
    row["mean_abs_L_minus_1_proxy"] = float("nan") if px is None else px.mean_abs_l_minus_1
    row["proxy_valid"] = None if px is None else bool(px.valid)
    if att is not None:

        def segs(x: tuple[tuple[float, float], ...]) -> str:
            return ";".join(f"{a:g}-{b:g}" for a, b in x)

        row.update(
            {
                "ssr_floor": att.floor,
                "ssr_floor_target": att.floor_target,
                "ssr_floor_at_scan_edge": att.floor_at_scan_edge,
                "ssr_floor_bounds": ";".join(att.floor_bounds),
                "ssr_ceiling": att.ceiling,
                "ssr_ceiling_target": att.ceiling_target,
                "ssr_ceiling_at_scan_edge": att.ceiling_at_scan_edge,
                "ssr_ceiling_bounds": ";".join(att.ceiling_bounds),
                "tracking_segments": segs(att.tracking_segments),
                "strict_tracking_segments": segs(att.strict_tracking_segments),
            }
        )
    c = clamped if clamped is not None else r
    row["message"] = c.message or ""
    if clamped is not None:
        row["clamped_ssr_first_target"] = float(clamped.table["ssr_target"].mean())
        row["clamped_ssr_fitted"] = float(clamped.table["ssr_fitted"].mean())
        row["clamped_ssr_achieved_mean"] = clamped.ssr_achieved_mean
        row["clamped_mean_skew_gap"] = clamped.mean_skew_gap
        row["clamped_k1"] = clamped.params.k1
        row["clamped_nu"] = clamped.params.nu
    return row


def stage3_row(r: FitResult, s3: Stage3Report) -> dict[str, Any]:
    """Stage-3 columns: actual mean ``|L − 1|``, numerical LSV SSR per pillar (± se, z against
    the fitted target), the naked mixing skew gap, simulated SpotVolCovar / VolVar of the LSV
    against the requested targets and the fit's first-order model values."""
    row: dict[str, Any] = {"mean_abs_L_minus_1": s3.mean_abs_l_minus_1}
    for _, x in s3.ssr_table.iterrows():
        T = f"{float(x['T']):g}"
        row[f"ssr_lsv_num_{T}"] = float(x["ssr_model"])
        row[f"ssr_lsv_num_se_{T}"] = float(x["ssr_model_se"])
        row[f"ssr_lsv_num_z_{T}"] = float(x["z"])
    gaps = s3.skew_table["gap"].to_numpy(dtype=float)
    row["mean_skew_gap_mixing"] = float(np.nanmean(np.abs(gaps))) if gaps.size else float("nan")
    for _, x in s3.skew_table.iterrows():
        T = f"{float(x['T']):g}"
        row[f"skew_naked_mixing_{T}"] = float(x["skew_naked"])
        row[f"skew_naked_mixing_se_{T}"] = float(x["skew_naked_se"])
    tab = r.table.set_index("T")
    for _, x in s3.breakeven_table.iterrows():
        Tf = float(x["T"])
        T = f"{Tf:g}"
        i = int(np.argmin(np.abs(tab.index.to_numpy() - Tf)))
        fit = tab.iloc[i] if abs(float(tab.index[i]) - Tf) < 1e-9 else None
        row[f"svc_sim_{T}"] = float(x["svc_sim"])
        row[f"svc_sim_se_{T}"] = float(x["svc_se"])
        row[f"svc_target_{T}"] = float(x["svc_target"])
        row[f"svc_model_fo_{T}"] = float("nan") if fit is None else float(fit["svc_model"])
        row[f"volvar_sim_{T}"] = float(x["volvar_sim"])
        row[f"volvar_sim_se_{T}"] = float(x["volvar_se"])
        row[f"volvar_target_{T}"] = float(x["volvar_target"])
        row[f"volvar_target_fit_{T}"] = float("nan") if fit is None else float(fit["volvar_target"])
        row[f"volvar_model_fo_{T}"] = float("nan") if fit is None else float(fit["volvar_model"])
        row[f"ssr_sim_breakeven_{T}"] = float(x["ssr_sim"])
    row["stage3_pricing_paths"] = s3.n_paths
    row["stage3_seconds"] = s3.wall_seconds
    return row


def summary_frame(df: pd.DataFrame, pillars: Sequence[float]) -> pd.DataFrame:
    """The headline columns of the study frame (numerical SSR as ``R ± se`` strings)."""
    cols = {
        "ssr_requested": "target",
        "skew_weight": "w_skew",
        "ssr_achieved_naked_vs_market_mean": "SSR naked/mkt",
        "ssr_naked_own_mean": "SSR naked own",
        "ssr_achieved_lsv_mean": "SSR lsv (1st order)",
        "first_order_valid": "1st order validated",
        "mean_skew_gap": "skew gap",
        "mean_abs_L_minus_1": "mean abs(L-1) actual",
        "mean_abs_L_minus_1_noise": "abs(L-1) 2-seed diff",
        "mean_abs_L_minus_1_proxy": "mean abs(L-1) proxy",
        "ssr_floor": "floor (1st order)",
        "ssr_ceiling": "ceiling (1st order)",
    }
    out = pd.DataFrame({v: df[k] for k, v in cols.items() if k in df})
    for T in pillars:
        t = f"{T:g}"
        if f"ssr_lsv_num_{t}" in df:
            out[f"SSR LSV num {t}"] = [
                f"{a:.3f} ± {b:.3f}" if math.isfinite(a) else "nan"
                for a, b in zip(df[f"ssr_lsv_num_{t}"], df[f"ssr_lsv_num_se_{t}"])
            ]
    out["message"] = ["yes" if m else "no" for m in df["message"]]
    for k in ("calibration_seconds", "wall_seconds", "recalibrated"):
        if k in df:
            out[k] = df[k]
    return out


def pillar_frame(df: pd.DataFrame, pillars: Sequence[float]) -> pd.DataFrame:
    """Long per-pillar table: achieved SSRs, skew, numerical LSV SSR and simulated break-evens."""
    rows = []
    for _, x in df.iterrows():
        for T in pillars:
            t = f"{T:g}"
            row = {
                "target": x["ssr_requested"],
                "w_skew": x["skew_weight"],
                "T": T,
                "ssr_naked_vs_mkt": x.get(f"ssr_naked_vs_market_{t}", float("nan")),
                "ssr_naked_own": x.get(f"ssr_naked_own_{t}", float("nan")),
                "ssr_lsv_fo": x.get(f"ssr_lsv_{t}", float("nan")),
                "ssr_lsv_num": x.get(f"ssr_lsv_num_{t}", float("nan")),
                "se": x.get(f"ssr_lsv_num_se_{t}", float("nan")),
                "skew_mkt": x.get(f"skew_market_{t}", float("nan")),
                "skew_naked_fo": x.get(f"skew_naked_{t}", float("nan")),
                "skew_gap": x.get(f"skew_gap_{t}", float("nan")),
                "svc_sim": x.get(f"svc_sim_{t}", float("nan")),
                "svc_se": x.get(f"svc_sim_se_{t}", float("nan")),
                "svc_target": x.get(f"svc_target_{t}", float("nan")),
                "svc_fo": x.get(f"svc_model_fo_{t}", float("nan")),
                "volvar_sim": x.get(f"volvar_sim_{t}", float("nan")),
                "volvar_se": x.get(f"volvar_sim_se_{t}", float("nan")),
                "volvar_target": x.get(f"volvar_target_{t}", float("nan")),
                "volvar_target_fit": x.get(f"volvar_target_fit_{t}", float("nan")),
                "volvar_fo": x.get(f"volvar_model_fo_{t}", float("nan")),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def _check_spec_round_trip(path: Path, spec: CalibrationSpec, r: FitResult) -> None:
    loaded = load_yaml(path, CalibrationSpec, section="spec")
    if loaded != spec or spec_key(loaded) != spec_key(spec):
        raise RuntimeError(f"{path}: the spec section does not round trip (cache key would differ)")
    entries = [e for e in load_tradeoff_specs(path.parent) if e.path == path]
    if len(entries) != 1:
        raise RuntimeError(f"{path}: not read back by load_tradeoff_specs")
    e = entries[0]
    if e.config != r.config or e.breakeven != r.breakeven or e.spec != spec:
        raise RuntimeError(f"{path}: the fit section does not round trip")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--targets", default="1.0,1.5", help="comma-separated constant ssr_targets")
    ap.add_argument(
        "--weights",
        default=",".join(f"{w:g}" for w in DEFAULT_TRADEOFF_WEIGHTS),
        help="comma-separated skew weights, skew-tight to skew-loose",
    )
    ap.add_argument("--ssr-measure", choices=SSR_MEASURES, default=None, help="default: fitter's")
    ap.add_argument("--no-stage3", action="store_true", help="first order only, no calibration")
    ap.add_argument(
        "--no-noise",
        action="store_true",
        help="skip the second-seed calibrations of the tightest / loosest weight",
    )
    ap.add_argument("--n-particles", type=int, default=200_000)
    ap.add_argument("--n-paths", type=int, default=100_000, help="stage-3 pricing paths")
    ap.add_argument("--mixing-paths", type=int, default=100_000, help="naked mixing-skew paths")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "m7"))
    ap.add_argument("--spec-dir", default=str(ROOT / "configs" / "studies" / "m7_skew_tradeoff"))
    ap.add_argument("--cache", default=str(ROOT / "cache"), help="leverage cache root")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    t_all = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec_dir = Path(args.spec_dir)
    targets = _floats(args.targets)
    weights = _floats(args.weights)

    base = load_yaml(BASE_SPEC, CalibrationSpec)
    ref_surface_cfg = load_yaml(REFERENCE_SURFACE, SSVIConfig, section="ssvi")
    if base.surface != ref_surface_cfg or base.perturbation is not None:
        raise RuntimeError(f"{BASE_SPEC} surface differs from {REFERENCE_SURFACE}")
    spec_base = dataclasses.replace(
        base, particle=dataclasses.replace(base.particle, n_particles=int(args.n_particles))
    )
    if spec_base.particle != ParticleConfig(n_particles=int(args.n_particles), horizon=3.0):
        raise RuntimeError("the base spec's particle settings are not ParticleConfig defaults")
    if spec_base.sim != SimConfig():
        raise RuntimeError("the base spec's simulation settings are not SimConfig() defaults")
    _, surface, _ = build_market(spec_base)
    cache = LeverageCache(args.cache)
    pricing = SimConfig(n_paths=int(args.n_paths), chunk_size=100_000, seed=7)
    cfg_kw: dict[str, Any] = {"pillars": PILLARS, "k2": K2}
    if args.ssr_measure is not None:
        cfg_kw["ssr_measure"] = args.ssr_measure

    rows: list[dict[str, Any]] = []
    details: list[str] = []
    for tgt in targets:
        for w in weights:
            t0 = time.perf_counter()
            cfg = BreakEvenFitConfig(skew_weight=float(w), clamp_to_attainable=False, **cfg_kw)
            r = fit_2f_marking(surface, cfg, ssr_target=float(tgt), proxy=True)
            rc = fit_2f_marking(
                surface,
                dataclasses.replace(cfg, clamp_to_attainable=True),
                ssr_target=float(tgt),
                proxy=False,
            )
            att = r.attainable or attainable_ssr(r.targets, cfg, r.xi0)
            row: dict[str, Any] = {"ssr_target": float(tgt)}
            row.update(fit_row(r, att, rc))
            spec = dataclasses.replace(spec_base, model=r.params)
            path = spec_dir / f"ssr{tgt:g}_w{w:g}.yaml"
            write_tradeoff_spec(
                r, spec_base, path, n_particles=int(args.n_particles), ssr_target=float(tgt)
            )
            _check_spec_round_trip(path, spec, r)
            row["spec_path"] = str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
            row["cache_key"] = spec_key(spec)
            fo_s = time.perf_counter() - t0
            print(
                f"[target {tgt:g}, weight {w:g}] first order {fo_s:.1f} s: SSR lsv "
                f"{row['ssr_achieved_lsv_mean']:.3f}, naked/mkt "
                f"{row['ssr_achieved_naked_vs_market_mean']:.3f}, skew gap "
                f"{row['mean_skew_gap']:.3f}, proxy {row['mean_abs_L_minus_1_proxy']:.3f}, floor "
                f"{att.floor:.3f} (all first order)",
                flush=True,
            )
            block = [
                f"### ssr_target {tgt:g}, skew_weight {w:g}",
                "",
                f"Spec `{row['spec_path']}` (cache key `{row['cache_key'][:12]}`).",
                "",
                "```",
                r.summary(),
                "",
                att.summary(),
                "",
                "clamped fit (fitter default): "
                + (rc.message or "no message (the requested target is tracked)"),
                *(f"  - {d}" for d in rc.message_details),
                "```",
            ]
            row["recalibrated"] = False
            row["calibration_seconds"] = 0.0
            if not args.no_stage3:
                hit = cache.has(spec)
                t1 = time.perf_counter()
                lsv, _ = cache.get_or_calibrate(spec, allow_calibrate=True)
                cal_s = time.perf_counter() - t1
                print(
                    f"  leverage {'loaded from' if hit else 'calibrated into'} the cache "
                    f"({spec.particle.n_particles} particles) in {cal_s:.0f} s",
                    flush=True,
                )
                inputs = Stage3Inputs(
                    surface=surface,
                    particle=spec.particle,
                    sim=spec.sim,
                    pricing_sim=pricing,
                    ssr_pillars=PILLARS,
                    breakeven_pillars=PILLARS,
                    forward_starts=(),
                    model=lsv,
                    mixing_paths=int(args.mixing_paths),
                )
                s3 = stage3_validation(r.params, inputs, r.targets, fit_table=r.table)
                row.update(stage3_row(r, s3))
                row["recalibrated"] = not hit
                row["calibration_seconds"] = cal_s
                block += [
                    "",
                    f"Stage 3: leverage {'calibrated' if not hit else 'read from the cache'} in "
                    f"{cal_s:.0f} s ({spec.particle.n_particles} particles; recalibrated: "
                    f"{'yes' if not hit else 'no'}); stage-3 pricing {s3.wall_seconds:.0f} s.",
                    "",
                    "```",
                    s3.summary(),
                    "```",
                ]
                print(
                    f"  stage 3 {s3.wall_seconds:.0f} s: mean |L-1| {s3.mean_abs_l_minus_1:.3f}; "
                    "numerical SSR "
                    + ", ".join(
                        f"{x.T:g}: {x.ssr_model:.3f}±{x.ssr_model_se:.3f}"
                        for x in s3.ssr_table.itertuples()
                    ),
                    flush=True,
                )
            noise = (not args.no_stage3) and (not args.no_noise) and w in (weights[0], weights[-1])
            if noise:
                spec2 = dataclasses.replace(
                    spec, particle=dataclasses.replace(spec.particle, seed=TRADEOFF_NOISE_SEED)
                )
                hit2 = cache.has(spec2)
                t2 = time.perf_counter()
                lsv2, _ = cache.get_or_calibrate(spec2, allow_calibrate=True)
                cal2 = time.perf_counter() - t2
                a2, _ = mean_abs_leverage_deviation(lsv2.leverage, surface)
                diff = abs(a2 - row["mean_abs_L_minus_1"])
                row["mean_abs_L_minus_1_seed2"] = a2
                row["mean_abs_L_minus_1_noise"] = diff
                row["recalibrated"] = bool(row["recalibrated"] or not hit2)
                row["calibration_seconds"] = float(row["calibration_seconds"]) + cal2
                block += [
                    "",
                    f"Particle noise: seed {TRADEOFF_NOISE_SEED} leverage "
                    f"{'read from the cache' if hit2 else 'calibrated'} in {cal2:.0f} s; mean "
                    f"|L - 1| {a2:.4f} against {row['mean_abs_L_minus_1']:.4f} (difference "
                    f"{diff:.4f}, one-calibration standard deviation about "
                    f"{diff / np.sqrt(2):.4f}).",
                ]
                print(f"  particle noise: |L-1| seed 2 {a2:.4f}, difference {diff:.4f}", flush=True)
            row["wall_seconds"] = time.perf_counter() - t0
            rows.append(row)
            details += ["", *block]

    df = pd.DataFrame(rows)
    df.to_csv(out / "skew_tradeoff.csv", index=False)
    total = time.perf_counter() - t_all
    any_recal = bool(df["recalibrated"].any()) if len(df) else False
    measure = rows[0]["ssr_measure"] if rows else "?"
    lines = [
        "# M7 Part 3 - skew-weight trade-off of the soft-skew fitter (reference SSVI)",
        "",
        f"Surface `configs/surfaces/reference_ssvi.yaml` (marking mode), pillars "
        f"{list(PILLARS)}, k2 fixed {K2:g}, ssr_measure `{measure}`, targets {targets}, skew "
        f"weights {weights} (skew-tight to skew-loose).  Study fits are **unclamped** "
        "(`clamp_to_attainable=False`); the floor / ceiling and the owner's message come from "
        "the clamped first-order refit (the fitter's default).  Floors, achieved SSRs and "
        "messages are first-order estimates; the numerical LSV SSR (stage 3, with standard "
        "errors) is the truth.",
        (
            "Stage 3 skipped (`--no-stage3`): first order only, nothing calibrated."
            if args.no_stage3
            else f"Stage 3: leverage calibrated into (or, on a hit, read from) the cache "
            f"`{args.cache}` at "
            f"{args.n_particles} particles (ParticleConfig(horizon=3.0), SimConfig() schedule; "
            f"the development count), pricing {args.n_paths} paths (seed 7), mixing skew "
            f"{args.mixing_paths} paths.  Particle noise of |L-1|: "
            + (
                "not measured (`--no-noise`)."
                if args.no_noise
                else f"second seed {TRADEOFF_NOISE_SEED} at the tightest and loosest weight."
            )
        ),
        f"Specs in `{args.spec_dir}` (round trip checked).  Total wall clock {total:.0f} s; "
        f"recalibrated: {'yes' if any_recal else 'no'}.",
        "",
        "## Summary",
        "",
        markdown_table(summary_frame(df, PILLARS)) if len(df) else "(no rows)",
        "",
        "## Parameters (book and break-even, first-order standard errors)",
        "",
        (
            markdown_table(
                df[
                    [
                        "ssr_requested",
                        "skew_weight",
                        "nu",
                        "theta",
                        "k1",
                        "k2",
                        "rho12",
                        "rho_SX1",
                        "rho_SX2",
                        "be_lambda1",
                        "be_lambda1_se",
                        "be_lambda2",
                        "be_lambda2_se",
                        "be_k1_se",
                        "be_omega1",
                        "be_omega2",
                        "be_chi",
                        "be_chi_se",
                        "k1_at_bound",
                        "nu_limit_binding",
                    ]
                ]
            )
            if len(df)
            else ""
        ),
        "",
        "## Per pillar (fo = first order; num = LSV numerical with standard error)",
        "",
        markdown_table(pillar_frame(df, PILLARS)) if len(df) else "",
        "",
        "## Messages (clamped refit)",
        "",
        *(
            f"- target {x['ssr_requested']:g}, weight {x['skew_weight']:g}: "
            + (x["message"] or "none (tracked)")
            for _, x in df.iterrows()
        ),
        "",
        "## Details",
        *details,
    ]
    (out / "skew_tradeoff.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out / 'skew_tradeoff.md'} and .csv; total wall clock {total:.0f} s")


if __name__ == "__main__":
    main()
