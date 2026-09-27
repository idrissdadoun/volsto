"""M7 Part 3 final: the P1 marking calibration (SABR break-evens, two-point 1Y/5Y hard skew
constraint) on SPX 2022-12-30 (the marking reference since the owner's report decision vii; the
placeholder SSVI on request with ``--surfaces spx,reference``), with every diagnostic, the
stage-3 assertion and the shadow-rotation greek under both recalibration policies (owner's "M7
Part 3 FINAL" and report decisions; methodology in the module docstrings of
:mod:`volsto.calibration.fit_2f`, :mod:`volsto.calibration.targets` and
:mod:`volsto.risk.shadow_rotation`).

Fits: ``--pairs`` (default ``(ssr_target, skew_eps) = (1.0, 0.10)`` and ``(1.5, 0.05)``) with the
named fit ``--fit`` (:data:`volsto.calibration.fit_2f.FIT_PRESETS`; default ``desk`` since
2026-09-27 — step 0 from the snapshot's SABRW fits, k2 fitted, ATMF kernels, 3M ``σ_0``, the desk
note's bounds —, ``m7`` the M7 fit: k2 = 0.2, ν cap 3.5; pillars 3M–10Y inside the surface,
MatMin 3M, SmoothBreakEven either way).  The reference SSVI has no quotes: it runs with
``--fit m7`` only.  Surfaces: the repaired eSSVI snapshot (since 2026-09-22; plain SSVI before)
``configs/surfaces/snapshots/hdn_2022H2/spx_2022-12-30.yaml`` (quoted to 3Y: the 5Y / 10Y
pillars are dropped and the 5Y skew constraint moves to 3Y, with the fitter's note) and, on
request, the reference SSVI of ``configs/studies/lsv_reference_2f.yaml`` (to 10Y).

Per fit: the fitted parameter set is written as a study spec
(``configs/studies/m7_p1_marking/<surface>_ssr<s>_eps<e>.yaml``), its leverage calibrated into the
cache (``ParticleConfig`` of the reference study spec, ``--n-particles`` default 2·10⁵, horizon 3y;
a cache hit is reported as not recalibrated), and stage 3 runs on the cached LSV: numerical SSR at
3M / 6M / 1Y / 2Y / 3Y (diagnostic), mean ``|L − 1|``, naked mixing skew and simulated
SpotVolCovar / VolVar at 3M / 1Y, forward 90/110 skew at 1y-into-1y and 2y-into-1y against the
spot 1y skew (``--n-paths`` default 10⁵, seed 7); the stage-3 assertion (simulated SpotVolCovar
and VolVar within the config tolerance of the fit's targets) is reported per fit, never raised
here.  ``--iterate k`` runs the iteration against simulation on every fit (``k`` extra
calibrations each, not cached) and reports its convergence.

Shadow rotation (``--rotation``, default ``spx``; ``--policies`` default all three): the M6
headline 3y autocall on the ``(1.0, 0.10)`` fit, central differences at ±1 rota, the rotated
states calibrated into the cache, the same product under pure local vol on the same surfaces
(``--rotation-paths`` default 2·10⁵, seed 2024).  Reported in the owner's declared convention
(``ROTATION_CONVENTION``: rota +1 = 6M 90/110 skew steepens by 0.56 vp; fee = P1 price − LV
price; desk P&L per +1 rota for a SHORT position = −(d fee)): the P1 / LV / fee levels, the LV
rotation, the P1 usual and recalibrated rotations, the fee and desk-P&L usual / recalibrated /
shadow rotations, each with its standard error
(``configs/studies/m7_p1_marking/rotation_<surface>_<policy>.yaml`` records the refit sets, the
convention and every quantity as ``[value, stderr]``).

Outputs: ``<out>/p1_marking.md``, ``p1_marking_fits.csv``, ``p1_marking_shadow_rotation.csv``.
Every section states its wall clock and whether it recalibrated.  Full run (10–20 min; under
``caffeinate -i``)::

    caffeinate -i .venv/bin/python scripts/m7_p1_marking.py --surfaces spx,reference

Smoke test (first order only)::

    .venv/bin/python scripts/m7_p1_marking.py --no-stage3 --no-rotation --out /tmp/x
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from volsto.calibration.cache import LeverageCache, build_market
from volsto.calibration.fit_2f import (
    FIT_PRESETS,
    BreakEvenValidationError,
    FitResult,
    Stage3Inputs,
    fit_2f_marking,
    fit_preset,
    stage3_validation,
    write_fit_spec,
)
from volsto.config import (
    CalibrationSpec,
    SimConfig,
    load_yaml,
    to_mapping,
)
from volsto.market.loaders import load_step0_source, snapshot_spec
from volsto.risk.shadow_rotation import (
    RECALIBRATION_POLICIES,
    ROTATION_CONVENTION,
    rotation_shadow_sensitivity,
)
from volsto.studies.m6 import AUTOCALL_NAME, headline_products

ROOT = Path(__file__).resolve().parents[1]
REF_SPEC = ROOT / "configs" / "studies" / "lsv_reference_2f.yaml"
SPX_SNAPSHOT = ROOT / "configs" / "surfaces" / "snapshots" / "hdn_2022H2" / "spx_2022-12-30.yaml"
STUDY_DIR = ROOT / "configs" / "studies" / "m7_p1_marking"
FITS: tuple[tuple[float, float], ...] = ((1.0, 0.10), (1.5, 0.05))
SURFACES = ("spx", "reference")
SSR_PILLARS = (0.25, 0.5, 1.0, 2.0, 3.0)
BE_PILLARS = (0.25, 1.0)
FORWARD_STARTS = ((1.0, 2.0), (2.0, 3.0))


def base_spec(surface: str, n_particles: int) -> CalibrationSpec:
    """The calibration spec of a surface (reference study particle / simulation settings)."""
    ref = load_yaml(REF_SPEC, CalibrationSpec)
    if surface == "spx":
        ref = snapshot_spec(ref, SPX_SNAPSHOT)
    elif surface != "reference":
        raise ValueError(f"unknown surface {surface!r}")
    return dataclasses.replace(
        ref, particle=dataclasses.replace(ref.particle, n_particles=n_particles)
    )


def step0_of(surface: str, fit: str, surf: Any) -> Any:
    """Step 0's source of the named fit on ``surface`` (built on ``surf``): the SPX snapshot's
    SABRW fits when the fit reads them, else ``None``."""
    if FIT_PRESETS[fit].step0 is None:
        return None
    if surface != "spx":
        raise SystemExit(f"--fit {fit} reads a snapshot's SABRW fits: {surface!r} has no quotes")
    return load_step0_source(SPX_SNAPSHOT, surf)


def label(surface: str, ssr: float, eps: float) -> str:
    return f"{surface}_ssr{ssr:g}_eps{eps:g}"


def fit_row(surface: str, ssr: float, eps: float, r: FitResult) -> dict[str, Any]:
    p, ct, se = r.params, r.constraints, r.short_end
    row: dict[str, Any] = {
        "surface": surface,
        "ssr_target": ssr,
        "skew_eps": eps,
        "status": r.status,
        "nu": p.nu,
        "theta": p.theta,
        "k1": p.k1,
        "rho_SX1": p.rho_SX1,
        "rho_SX2": p.rho_SX2,
        "rho12": p.rho12,
        "active": ";".join(r.first.active),
        "bounds": ";".join(r.second.bound_flags),
    }
    for rec in ct.to_dict(orient="records"):
        row[f"T_{rec['name']}"] = rec["T"]
        row[f"skew_gap_{rec['name']}"] = rec["gap_rel"]
    for rec in se.to_dict(orient="records"):
        row[f"short_gap@{rec['T']:.3g}y"] = rec["gap_rel"]
    row["svc_err_max"] = float(np.max(np.abs(r.svc_rel_error)))
    row["volvar_err_max"] = float(
        np.max(np.abs(r.table["volvar_model"] / r.table["volvar_target"] - 1))
    )
    row["corr_model_mean"] = float(r.table["corr_model"].mean())
    row["corr_target_mean"] = float(r.table["corr_target"].mean())
    row["rho12_collapse"] = any("collapsing" in n for n in r.notes)
    return row


def _pairs(text: str) -> list[tuple[float, float]]:
    out = []
    for item in text.split(","):
        if item.strip():
            a, b = item.split(":")
            out.append((float(a), float(b)))
    return out


def _transpose(df: pd.DataFrame) -> pd.DataFrame:
    named = df.copy()
    named.index = [
        f"{r['surface']} ssr {r['ssr_target']:g} eps {r['skew_eps']:g}"
        for r in df.to_dict(orient="records")
    ]
    t = named.T.reset_index()
    t.columns = ["field", *[str(c) for c in named.index]]
    return t


def markdown_table(df: pd.DataFrame, digits: int = 4) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(str(c) for c in cols) + " |", "|" + "---|" * len(cols)]
    for _, rec in df.iterrows():
        cells = []
        for c in cols:
            v = rec[c]
            if isinstance(v, float | np.floating):
                cells.append(f"{v:.{digits}f}" if np.isfinite(v) else "nan")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--surfaces", default="spx", help="comma-separated: spx, reference")
    ap.add_argument("--fit", default="desk", choices=sorted(FIT_PRESETS), help="named fit")
    ap.add_argument("--pairs", default="1.0:0.10,1.5:0.05", help="ssr_target:skew_eps pairs")
    ap.add_argument("--iterate", type=int, default=0, help="iterate_against_simulation k")
    ap.add_argument("--n-particles", type=int, default=200_000)
    ap.add_argument("--n-paths", type=int, default=100_000)
    ap.add_argument("--mixing-paths", type=int, default=100_000)
    ap.add_argument("--rotation", default="spx", help="surfaces, or 'none'")
    ap.add_argument("--policies", default=",".join(RECALIBRATION_POLICIES))
    ap.add_argument("--rotation-paths", type=int, default=200_000)
    ap.add_argument("--no-stage3", action="store_true")
    ap.add_argument("--no-rotation", action="store_true")
    ap.add_argument("--cache", default=str(ROOT / "cache"))
    ap.add_argument("--out", default=str(ROOT / "outputs" / "m7"))
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    t_all = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache = LeverageCache(args.cache)
    surfaces = [s for s in args.surfaces.split(",") if s]
    pairs = _pairs(args.pairs)
    pricing = SimConfig(n_paths=int(args.n_paths), chunk_size=100_000, seed=7)
    rows: list[dict[str, Any]] = []
    details: list[str] = []
    any_recal = False
    for surface in surfaces:
        spec = base_spec(surface, int(args.n_particles))
        _, surf, _ = build_market(spec)
        step0 = step0_of(surface, args.fit, surf)
        for ssr, eps in pairs:
            t0 = time.perf_counter()
            cfg = fit_preset(args.fit, skew_eps=eps)
            r = fit_2f_marking(surf, cfg, ssr_target=ssr, step0=step0)
            fo_s = time.perf_counter() - t0
            iterated: FitResult | None = None
            if args.iterate and not args.no_stage3:
                t_it = time.perf_counter()
                try:
                    iterated = fit_2f_marking(
                        surf,
                        cfg,
                        ssr_target=ssr,
                        stage3=Stage3Inputs(
                            surface=surf,
                            particle=spec.particle,
                            sim=spec.sim,
                            pricing_sim=pricing,
                            ssr_pillars=(0.25, 1.0),
                            breakeven_pillars=tuple(float(t) for t in r.table["T"]),
                            forward_starts=(),
                            mixing_paths=int(args.mixing_paths),
                        ),
                        iterate_against_simulation=int(args.iterate),
                        assert_stage3=False,
                        step0=step0,
                    )
                except BreakEvenValidationError as exc:
                    iterated = exc.result
                print(
                    f"  iterated {args.iterate}x ({time.perf_counter() - t_it:.0f} s, "
                    f"recalibrated {args.iterate + 1}x):\n"
                    + (
                        iterated.iterations.round(4).to_string(index=False)
                        if iterated is not None and iterated.iterations is not None
                        else "-"
                    ),
                    flush=True,
                )
            name = label(surface, ssr, eps)
            row = fit_row(surface, ssr, eps, r)
            row["fit_seconds"] = fo_s
            block = [
                "",
                f"### {name} (first order {fo_s:.1f} s)",
                "",
                "```",
                r.summary(),
                "",
                "targets:",
                r.targets.frame().round(5).to_string(index=False),
                "```",
            ]
            if iterated is not None and iterated.iterations is not None:
                s3i = iterated.stage3
                block += [
                    "",
                    f"Iteration against simulation (k = {args.iterate}; every iteration "
                    "recalibrates the leverage, not cached):",
                    "",
                    "```",
                    iterated.iterations.round(4).to_string(index=False),
                    "final params: " + repr(iterated.params),
                    "stage-3 assertion on the final model: "
                    + (
                        "PASS"
                        if s3i is not None and s3i.within_tolerance
                        else "FAIL - " + (s3i.check_message if s3i is not None else "no stage 3")
                    ),
                    "```",
                ]
            spec_path = write_fit_spec(
                r,
                spec,
                STUDY_DIR / f"{name}.yaml",
                n_particles=int(args.n_particles),
                ssr_target=ssr,
                label=name,
                extra={
                    "surface": surface,
                    "skew_eps": eps,
                    "fit": args.fit,
                    **(
                        {"snapshot": SPX_SNAPSHOT.relative_to(ROOT).as_posix()}
                        if surface == "spx"
                        else {}
                    ),
                },
            )
            print(f"[{name}] {r.status}; fit {fo_s:.1f} s; spec {spec_path.name}", flush=True)
            if not args.no_stage3:
                fspec = dataclasses.replace(spec, model=r.params)
                miss = not cache.has(fspec)
                t1 = time.perf_counter()
                lsv, _ = cache.get_or_calibrate(fspec)
                cal_s = time.perf_counter() - t1
                any_recal = any_recal or miss
                s3 = stage3_validation(
                    r.params,
                    Stage3Inputs(
                        surface=surf,
                        particle=fspec.particle,
                        sim=fspec.sim,
                        pricing_sim=pricing,
                        ssr_pillars=tuple(T for T in SSR_PILLARS if surf.max_maturity + 1e-9 >= T),
                        breakeven_pillars=BE_PILLARS,
                        forward_starts=FORWARD_STARTS,
                        model=lsv,
                        mixing_paths=int(args.mixing_paths),
                    ),
                    r.targets,
                    fit_table=r.table,
                    tolerance=cfg.stage3_tolerance,
                )
                chk = s3.check
                row.update(
                    {
                        "stage3_assertion": "pass" if s3.within_tolerance else "FAIL",
                        # asserted (owner decision 2026-09-15): the engine bias per quantity;
                        # the target gaps are reported, not asserted (binding fits)
                        "stage3_max_engine_bias_svc": float(
                            chk.query("quantity == 'SpotVolCovar'")["engine_bias"].abs().max()
                        ),
                        "stage3_max_engine_bias_volvar": float(
                            chk.query("quantity == 'VolVar'")["engine_bias"].abs().max()
                        ),
                        "stage3_max_gap_svc_reported": float(
                            chk.query("quantity == 'SpotVolCovar'")["gap_vs_target"].abs().max()
                        ),
                        "stage3_max_gap_volvar_reported": float(
                            chk.query("quantity == 'VolVar'")["gap_vs_target"].abs().max()
                        ),
                        "mean_abs_L_minus_1": s3.mean_abs_l_minus_1,
                        "recalibrated": miss,
                        "calibration_seconds": cal_s if miss else 0.0,
                        "stage3_seconds": s3.wall_seconds,
                    }
                )
                for rec in s3.ssr_table.to_dict(orient="records"):
                    row[f"ssr_lsv@{rec['T']:g}y"] = rec["ssr_lsv"]
                    row[f"ssr_lsv_se@{rec['T']:g}y"] = rec["ssr_lsv_se"]
                for rec in s3.forward_table.to_dict(orient="records"):
                    key = f"{rec['t1']:g}y-into-{rec['t2'] - rec['t1']:g}y"
                    row[f"fwd/spot@{key}"] = rec["ratio_fwd_to_spot"]
                    row[f"fwd/spot_se@{key}"] = rec["ratio_se"]
                block += [
                    "",
                    f"Leverage: {'calibrated' if miss else 'cache hit'} ({cal_s:.0f} s, "
                    f"{fspec.particle.n_particles} particles); stage 3 {s3.wall_seconds:.0f} s; "
                    f"recalibrated: {'yes' if miss else 'no'}.",
                    "",
                    "```",
                    s3.summary(),
                    "```",
                ]
                print(
                    f"  leverage {'calibrated' if miss else 'hit'} {cal_s:.0f} s; stage 3 "
                    f"{s3.wall_seconds:.0f} s; assertion "
                    f"{'PASS' if s3.within_tolerance else 'FAIL'}; "
                    f"|L-1| {s3.mean_abs_l_minus_1:.3f}; SSR "
                    + ", ".join(
                        f"{x['T']:g}: {x['ssr_lsv']:.3f}±{x['ssr_lsv_se']:.3f}"
                        for x in s3.ssr_table.to_dict(orient="records")
                    ),
                    flush=True,
                )
            rows.append(row)
            details += block
    fits_df = pd.DataFrame(rows)
    fits_df.to_csv(out / "p1_marking_fits.csv", index=False)

    rot_lines: list[str] = []
    rot_rows: list[dict[str, Any]] = []
    if not args.no_rotation and args.rotation != "none":
        rsim = SimConfig(n_paths=int(args.rotation_paths), chunk_size=100_000, seed=2024)
        policies = [p for p in args.policies.split(",") if p]
        combos = [(sf, pol) for sf in args.rotation.split(",") if sf for pol in policies]
        for surface, policy in combos:
            spec = base_spec(surface, int(args.n_particles))
            fc, rsurf, _ = build_market(spec)
            product = headline_products(fc.rate_curve, spec.market.spot)[AUTOCALL_NAME]
            misses_before = len(cache.manifest())
            rep = rotation_shadow_sensitivity(
                product,
                spec,
                fit_preset(args.fit, skew_eps=0.10),
                step0=step0_of(surface, args.fit, rsurf),
                ssr_target=1.0,
                cache=cache,
                pricing_sim=rsim,
                size=1.0,
                product_name=f"{AUTOCALL_NAME} on {surface}",
                policy=policy,
            )
            any_recal = any_recal or rep.recalibrated_any
            doc = {
                "surface": surface,
                "policy": policy,
                "fit": args.fit,
                "ssr_target": 1.0,
                "skew_eps": 0.10,
                "size": rep.size,
                "fits": {k: to_mapping(f.params) for k, f in rep.fits.items()},
                "status": {k: f.status for k, f in rep.fits.items()},
                "n_particles": int(args.n_particles),
                "pricing": {"n_paths": int(args.rotation_paths), "seed": 2024},
                "convention": rep.convention,
                "n_lv_builds": rep.n_lv_builds,
                "p1_level": [rep.p1_level.value, rep.p1_level.stderr],
                "lv_level": [rep.lv_level.value, rep.lv_level.stderr],
                "fee": [rep.fee.value, rep.fee.stderr],
                "lv_rotation": [rep.lv_rotation.value, rep.lv_rotation.stderr],
                "usual": [rep.usual.value, rep.usual.stderr],
                "recalibrated": [rep.recalibrated.value, rep.recalibrated.stderr],
                "shadow": [rep.shadow.value, rep.shadow.stderr],
                "fee_usual": [rep.fee_usual.value, rep.fee_usual.stderr],
                "fee_recalibrated": [rep.fee_recalibrated.value, rep.fee_recalibrated.stderr],
                "fee_shadow": [rep.fee_shadow.value, rep.fee_shadow.stderr],
                "desk_pnl_usual": [rep.desk_pnl_usual.value, rep.desk_pnl_usual.stderr],
                "desk_pnl_recalibrated": [
                    rep.desk_pnl_recalibrated.value,
                    rep.desk_pnl_recalibrated.stderr,
                ],
                "desk_pnl_shadow": [rep.desk_pnl_shadow.value, rep.desk_pnl_shadow.stderr],
            }
            (STUDY_DIR / f"rotation_{surface}_{policy}.yaml").write_text(
                yaml.safe_dump(doc, sort_keys=False), encoding="utf-8"
            )
            fr = rep.frame()
            fr.insert(0, "policy", policy)
            fr.insert(0, "surface", surface)
            fr["p1_level"] = rep.p1_level.value
            fr["p1_level_se"] = rep.p1_level.stderr
            fr["lv_level"] = rep.lv_level.value
            fr["lv_level_se"] = rep.lv_level.stderr
            fr["fee"] = rep.fee.value
            fr["fee_se"] = rep.fee.stderr
            rot_rows += [{str(k): v for k, v in r.items()} for r in fr.to_dict(orient="records")]
            rot_lines += [
                "",
                f"### Shadow rotation, {surface}, {policy} (cache entries before {misses_before}; "
                f"{rep.n_cache_misses} misses; recalibrated: "
                f"{'yes' if rep.recalibrated_any else 'no'}; {rep.wall_seconds:.0f} s)",
                "",
                "```",
                rep.summary(),
                "```",
            ]
            print(rep.summary(), flush=True)
        pd.DataFrame(rot_rows).to_csv(out / "p1_marking_shadow_rotation.csv", index=False)

    wall = time.perf_counter() - t_all
    head_cols = [c for c in fits_df.columns if not c.startswith(("ssr_lsv_se", "fwd/spot_se"))]
    md = [
        "# M7 Part 3 final - P1 marking calibration (SABR break-evens, two-point skew)",
        "",
        f"Wall clock {wall:.0f} s; recalibrated: {'yes' if any_recal else 'no'} "
        f"(leverages at {args.n_particles} particles into the cache; stage-3 pricing "
        f"{args.n_paths} paths; rotation {args.rotation_paths} paths).",
        "",
        "## Fits",
        "",
        markdown_table(_transpose(fits_df[head_cols])),
    ]
    if rot_rows:
        md += [
            "",
            "## Shadow rotation (3y autocall, per rota)",
            "",
            f"Convention: {ROTATION_CONVENTION}.",
            "",
            markdown_table(pd.DataFrame(rot_rows), 6),
        ]
    md += ["", "## Details", *details, *rot_lines]
    (out / "p1_marking.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"done in {wall:.0f} s -> {out / 'p1_marking.md'}", flush=True)


if __name__ == "__main__":
    main()
