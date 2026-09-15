"""M8b gate: the raw-slice-skew discriminator for the 2022 H2 SSR, and the historical P1 fit
spec of study B world (ii) (SPEC §15 Part 2, "Raw-slice discriminator (M8b)").

Steps: (1) :func:`volsto.calibration.raw_history.raw_pillar_frame` on every sample day (the
importer pipeline up to the retained quotes, a local quadratic per slice, no SSVI fit; about
1 s per day); (2) :func:`~volsto.calibration.raw_history.discriminator` against the SSVI history
(``outputs/m7/hdn_history_ssvi.csv``) on the common dates, written to ``<out>/discriminator.md``,
``.csv`` and ``discriminator_verdict.json`` (plus ``raw_history.csv`` and
``raw_history_diagnostics.csv``); (3) unless ``--no-world-spec``, the FINAL historical fit
(``fit_2f_historical`` with the owner's historical defaults: soft skew, weight 10, k2 0.2,
pillars 3M–1Y, windows 100 / 60 — the same call ``scripts/m7_fit_2f_hdn.py --skew-mode soft
--skew-weight 10 --no-stage3`` makes) is rerun (0.1 s, first order, no leverage), checked against
``outputs/m7/fit_2f_hdn_historical.yaml`` and written as the study spec
``configs/studies/m8b/world_historical.yaml`` (SPX 2022-12-30 CalibrationSpec with the
historical BergomiParams and ``--n-particles``; the ``fit`` section records mode ``historical``,
the discriminator verdict and the fit provenance).  Nothing here calibrates a leverage.

Usage::

    caffeinate -i .venv/bin/python scripts/m8b_discriminator.py
    .venv/bin/python scripts/m8b_discriminator.py --limit 5 --out /tmp/x --no-world-spec
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

from volsto.calibration.fit_2f import (
    DEFAULT_K2,
    DEFAULT_SKEW_EPS,
    BreakEvenFitConfig,
    FitResult,
    fit_2f_historical,
    fit_spec_document,
)
from volsto.calibration.history import SurfaceHistory, hdn_available_dates
from volsto.calibration.raw_history import (
    FIT_BAND,
    WINDOW_LONG,
    WINDOW_SSR,
    Verdict,
    discriminator,
    raw_pillar_frame,
    verdict_from_table,
    write_discriminator_report,
)
from volsto.config import CalibrationSpec, MarketConfig, SSVIConfig, load_yaml

ROOT = Path(__file__).resolve().parents[1]
REF_SPEC = ROOT / "configs" / "studies" / "lsv_reference_2f.yaml"
SPX_SNAPSHOT = (
    ROOT / "configs" / "surfaces" / "snapshots" / "hdn_2022H2_ssvi" / "spx_2022-12-30.yaml"
)
HISTORICAL_PILLARS: tuple[float, ...] = (0.25, 0.5, 1.0)
"""``scripts/m7_fit_2f_hdn.py``: the history's pillars between MatMin 3M and ``--max-pillar`` 1y."""
HISTORICAL_SKEW_WEIGHT = 10.0
"""Owner decision: soft skew with weight 10 is the historical-mode default (FINAL fitter)."""
WORLD_N_PARTICLES = 800_000
"""Study B particle count (the owner's M8b spec)."""


def spx_base_spec(n_particles: int) -> CalibrationSpec:
    """``scripts/m7_p1_marking.py::base_spec('spx', n_particles)``: the reference 2F study spec
    with the SPX 2022-12-30 snapshot's market and surface."""
    ref = load_yaml(REF_SPEC, CalibrationSpec)
    ref = dataclasses.replace(
        ref,
        market=load_yaml(SPX_SNAPSHOT, MarketConfig, section="market"),
        surface=load_yaml(SPX_SNAPSHOT, SSVIConfig, section="ssvi"),
    )
    return dataclasses.replace(
        ref, particle=dataclasses.replace(ref.particle, n_particles=n_particles)
    )


def historical_fit(
    history: SurfaceHistory, *, window_long: int, window_ssr: int, skew_weight: float
) -> FitResult:
    cfg = BreakEvenFitConfig(
        pillars=HISTORICAL_PILLARS,
        k2=DEFAULT_K2,
        skew_mode="soft",
        skew_weight=skew_weight,
        skew_eps=DEFAULT_SKEW_EPS,
    )
    return fit_2f_historical(history, cfg, window_vol=window_long, window_ssr=window_ssr)


def compare_with_m7(result: FitResult, path: Path) -> list[str]:
    """Differences between this fit's parameters and the model block of ``path`` (empty when
    ``path`` is absent or agrees to 1e-6)."""
    if not path.exists():
        return [f"{path} absent: no cross-check with the m7 script's output"]
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    mine = dataclasses.asdict(result.params)
    diffs = [
        f"{k}: {mine[k]:.6g} here vs {float(v):.6g} in {path.name}"
        for k, v in doc["model"].items()
        if abs(float(v) - float(mine[k])) > 1e-6 * max(1.0, abs(float(v)))
    ]
    if str(doc.get("provenance", {}).get("status", "")) != result.status:
        diffs.append(f"status {result.status} here vs {doc['provenance'].get('status')}")
    return diffs


def world_spec_document(
    result: FitResult,
    verdict: Verdict,
    *,
    n_particles: int,
    checks: list[str],
    history_path: str = "outputs/m7/hdn_history_ssvi.csv",
    n_dates: int | None = None,
    window_long: int = WINDOW_LONG,
    window_ssr: int = WINDOW_SSR,
    limit: int | None = None,
) -> dict[str, Any]:
    """The world-(ii) FitSpec document; the provenance strings are formatted from the run's
    actual inputs (history path and length, windows, the fit's config), stamped 'partial' when
    ``--limit`` truncated the sample."""
    tg = result.targets
    cfg = result.config
    days = f"{n_dates} days" if n_dates is not None else "length unknown"
    partial = f"; PARTIAL SAMPLE (--limit {limit})" if limit is not None else ""
    ssr_pillars = [float(x) for x in tg.ssr_target]
    extra = {
        "mode": "historical",
        "world": "(ii) historical",
        "ssr_target_note": (
            "historical mode has one SSR target per pillar (ssr_target_pillars); the scalar "
            "ssr_target above is their mean and is informational only"
        ),
        "ssr_target_pillars": {f"{T:g}": s for T, s in zip(result.config.pillars, ssr_pillars)},
        "pricing_date": None if result.pricing_date is None else str(result.pricing_date.date()),
        "discriminator": {"verdict": verdict.verdict, "reason": verdict.reason},
        "provenance": {
            "history": f"{history_path} (SSVI snapshots, {days}{partial})",
            "fit_call": (
                f"fit_2f_historical(history, BreakEvenFitConfig(pillars={tuple(cfg.pillars)}, "
                f"k2={cfg.k2:g}, skew_mode={cfg.skew_mode!r}, skew_weight={cfg.skew_weight:g}, "
                f"skew_eps={cfg.skew_eps!r}), window_vol={window_long}, window_ssr={window_ssr})"
            ),
            "m7_script": (
                f"scripts/m7_fit_2f_hdn.py --skew-mode {cfg.skew_mode} --skew-weight "
                f"{cfg.skew_weight:g} --no-stage3 -> outputs/m7/fit_2f_hdn_historical.yaml"
            ),
            "cross_check": checks or ["parameters equal outputs/m7/fit_2f_hdn_historical.yaml"],
            "first": {
                "objective": float(result.first.objective),
                "active": list(result.first.active),
                "k1_at_bound": bool(result.first.k1_at_bound),
                "notes": list(result.first.notes),
            },
            "second": {
                "objective": float(result.second.objective),
                "bound_flags": list(result.second.bound_flags),
                "n_distinct_optima": int(result.second.n_distinct),
                "notes": list(result.second.notes),
            },
            "target_flags": list(tg.flags),
            "notes": list(result.notes),
            "risk_regime": result.risk_regime,
            "wall_seconds": float(result.wall_seconds),
            "recalibrated": bool(result.recalibrated),
        },
    }
    return fit_spec_document(
        result,
        spx_base_spec(n_particles),
        n_particles=n_particles,
        ssr_target=float(np.mean(ssr_pillars)),
        label="world_historical",
        extra=extra,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="data/hdn_sample/options_sample_2022H2")
    ap.add_argument("--ssvi-history", default="outputs/m7/hdn_history_ssvi.csv")
    ap.add_argument("--out", default="outputs/m8b")
    ap.add_argument("--limit", type=int, default=None, help="first N sample days only")
    ap.add_argument("--fit-band", type=float, default=FIT_BAND)
    ap.add_argument("--window-ssr", type=int, default=WINDOW_SSR)
    ap.add_argument("--window-long", type=int, default=WINDOW_LONG)
    ap.add_argument("--world-spec", default="configs/studies/m8b/world_historical.yaml")
    ap.add_argument("--no-world-spec", action="store_true")
    ap.add_argument("--historical-yaml", default="outputs/m7/fit_2f_hdn_historical.yaml")
    ap.add_argument("--n-particles", type=int, default=WORLD_N_PARTICLES)
    ap.add_argument("--skew-weight", type=float, default=HISTORICAL_SKEW_WEIGHT)
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    dates = hdn_available_dates(args.root)
    if args.limit:
        dates = dates[: args.limit]
    raw_frame, diag = raw_pillar_frame(
        args.root, dates, fit_band=args.fit_band, ssvi_history=args.ssvi_history
    )
    t_raw = time.perf_counter() - t0
    raw_frame.to_csv(out / "raw_history.csv", index=False)
    diag.to_csv(out / "raw_history_diagnostics.csv", index=False)
    print(
        f"raw history: {len(dates)} dates requested, {int(diag['kept'].sum())} kept, "
        f"{int((~diag['kept']).sum())} dropped, {t_raw:.0f} s "
        f"(vs_vol from SSVI on {raw_frame.attrs.get('vs_vol_from_ssvi', 0)} rows)"
    )
    ssvi_frame = pd.read_csv(args.ssvi_history)
    table, verdict = discriminator(
        raw_frame, ssvi_frame, window_ssr=args.window_ssr, window_long=args.window_long
    )
    wall = time.perf_counter() - t0
    # sensitivity, not the gate: the rule on the historical fit's pillars only (MatMin 3M)
    fitted = verdict_from_table(table[table["T"] >= HISTORICAL_PILLARS[0] - 1e-9], args.window_ssr)
    paths = write_discriminator_report(
        out,
        table,
        verdict,
        diag,
        wall_seconds=wall,
        fit_band=args.fit_band,
        notes=(
            f"Raw import {t_raw:.0f} s for {len(dates)} dates; the SSR estimation is "
            f"instantaneous.  Common dates: {table.attrs.get('n_dates')}, end "
            f"{table.attrs.get('end_date')}.",
            "Sensitivity (reported, not the gate): the same rule restricted to the pillars the "
            f"historical fit uses ({', '.join(f'{T:g}' for T in HISTORICAL_PILLARS)}y, MatMin 3M) "
            f"reads **{fitted.verdict}** - {fitted.reason}",
        ),
    )
    print(f"sensitivity on the fitted pillars: {fitted.verdict} -- {fitted.reason}")
    print(table.round(4).to_string(index=False))
    print(f"VERDICT: {verdict.verdict} -- {verdict.reason}")
    print(f"written {', '.join(str(p) for p in paths.values())}; wall clock {wall:.0f} s")

    if args.no_world_spec:
        return
    t1 = time.perf_counter()
    hist = SurfaceHistory(ssvi_frame)
    r = historical_fit(
        hist,
        window_long=args.window_long,
        window_ssr=args.window_ssr,
        skew_weight=args.skew_weight,
    )
    checks = compare_with_m7(r, Path(args.historical_yaml))
    doc = world_spec_document(
        r,
        verdict,
        n_particles=args.n_particles,
        checks=checks,
        history_path=str(args.ssvi_history),
        n_dates=hist.n_dates,
        window_long=int(args.window_long),
        window_ssr=int(args.window_ssr),
        limit=args.limit,
    )
    spec_path = Path(args.world_spec)
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    print(
        f"historical fit ({time.perf_counter() - t1:.1f} s, recalibrated: no): status "
        f"{r.status}; {r.params}; messages {list(r.messages)}"
    )
    for c in checks:
        print(f"  cross-check: {c}")
    print(f"written {spec_path}")


if __name__ == "__main__":
    main()
