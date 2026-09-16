"""M10 Part 0 gate: the eSSVI calendar repair on the 127 days of the 2022 H2 HDN sample.

For every sample day the importer pipeline runs once up to the retained quotes and the surface
is fitted three ways on the identical ``GridSurface`` / ``SurfacePoints``:

* ``base`` — eSSVI without the repair (``calendar_repair=None``).  Its constructor would raise on
  a violating day, so this run alone builds the surface with a subclass whose calendar check
  records instead of raising (a measurement probe: the fit itself is untouched);
* ``rep`` — eSSVI with the default :class:`~volsto.market.import_hdn.CalendarRepairConfig`
  (what ``import_day`` now returns; checked constructor);
* ``ssvi`` — the plain-SSVI fallback the history used until now (``essvi=False``).

Reported (``<out>/essvi_gate.md``, ``essvi_gate_days.csv``, ``essvi_gate.json``): days passing
(base vs repaired, gate ≥ 95%; the constructor's exact-derivative check and, for the record, the
step checks ``Δw`` of the second pass (pillar-aware grid) and of M3b (no pillars)), the exact
``min ∂_T w`` on the constructor grid on ``k ∈ ±1`` and ``±3`` (both limits at every knot), the
certificate of the invariant (:func:`~volsto.market.surface.certify_calendar`: ``∂_T w ≥ margin``
on ``±3`` for every repaired or identity day, ``≥ 0`` on ``±3`` and ``±1`` for the unrepaired
fit), the DENSE check (:meth:`~volsto.market.surface.ESSVISurface.calendar_dense_check`: exact
``∂_T w`` on 1201 ``k`` on ``±3`` × 3000 ``T`` plus both limits at every knot, the secant and
step metrics of the verifier, and the floor-0 certificate), the status of every committed eSSVI
snapshot under ``configs/surfaces`` under the three constructor checks, RMS and
max error inside ``|k| ≤ 0.20`` from 3m to 3y (the fit's reported convention, and on the surface
actually returned — the ``θ`` clamp defect of ``fit_ssvi``), the 1y/2y/3y ATMF skew of each
variant, the anchor day 2022-12-30, and the pillar-count distribution (days whose 3y skew is
extrapolation whatever the parametrisation).  Surface fitting only, single core: nothing
calibrates, nothing simulates, nothing is written outside ``--out`` (refused under
``configs/``).

Usage::

    .venv/bin/python scripts/essvi_calendar_gate.py --out outputs/essvi_gate [--limit N]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd
import yaml

from volsto.calibration.history import hdn_available_dates
from volsto.market import import_hdn as ih
from volsto.market.loaders import load_ssvi_surface
from volsto.market.surface import CALENDAR_GRID_N_T, ESSVISurface, SSVISurface

ROOT = Path(__file__).resolve().parents[1]
ANCHOR = "2022-12-30"
"""The SPX anchor day of M7/M8b (``configs/surfaces/snapshots/hdn_2022H2_ssvi``)."""
GATE_PASS_FRACTION = 0.95
"""Owner's gate: eSSVI passes the calendar check on at least 95% of days."""
ERR_T_MIN, ERR_T_MAX, ERR_K_ABS = 0.25, 3.0, 0.20
"""Error window of the owner's gate: 3m to 3y, ``|k| ≤ 20%``."""
SKEW_TENORS = (1.0, 2.0, 3.0)
"""ATMF skew tenors: 1y is the dense-data control, 2y / 3y the owner's question."""
K_WIDE = 3.0
"""Dupire range (``LocalVolConfig``), the repair's constraint half-width."""
N_K_WIDE = 121
"""Points of the wide grid (the repair's default)."""
MARGIN = ih.DEFAULT_CALENDAR_REPAIR.margin
"""The repair's proven floor on ``∂_T w`` (per year)."""
SNAPSHOT_ROOT = ROOT / "configs" / "surfaces"
"""Committed surfaces whose eSSVI status under the pillar-aware constructor grid is reported."""
DENSE_FIELDS = (
    "min_dw_dt",
    "min_interior",
    "min_left",
    "min_right",
    "min_slope",
    "min_dw",
    "worst_k",
    "worst_T",
)
"""Fields of :class:`~volsto.market.surface.CalendarDenseCheck` written per day."""
TOL = ESSVISurface.CALENDAR_TOL


def m3b_min_dw(s: ESSVISurface) -> float:
    """The constructor's slack on the grid it used before the M10 Part 0 fix
    (``unique(linspace(min_maturity, last pillar, 200) ∪ {max_maturity})``, no pillars, ±1)."""
    ks = np.linspace(-s.CALENDAR_K_ABS, s.CALENDAR_K_ABS, s.CALENDAR_N_K)
    Ts = np.unique(
        np.concatenate(
            (np.linspace(s.min_maturity, float(s.pillars[-1]), CALENDAR_GRID_N_T), [s.max_maturity])
        )
    )
    return float(np.min(np.diff(s.total_variance(ks[None, :], Ts[:, None]), axis=0)))


class _RecordingESSVI(ESSVISurface):
    """ESSVISurface whose constructor calendar check is skipped (probe for the ``base`` run
    only; its quantity is read back with :meth:`calendar_min_dw_dt`)."""

    def _check_calendar_numeric(self) -> None:
        return None


@contextmanager
def _unchecked_essvi() -> Iterator[None]:
    with mock.patch.object(ih, "ESSVISurface", _RecordingESSVI):
        yield


def _surface_errors(fit: ih.SSVIFit) -> tuple[float, float]:
    """RMS / max error in the gate window on the surface ``fit`` returns (the fit's ``points``
    clamp ``θ`` beyond the last pillar; the surface extrapolates it)."""
    p = fit.points
    m = (p["T"] >= ERR_T_MIN - 1e-9) & (p["T"] <= ERR_T_MAX + 1e-9) & (p["k"].abs() <= ERR_K_ABS)
    k = p.loc[m, "k"].to_numpy(float)
    T = p.loc[m, "T"].to_numpy(float)
    e = 100.0 * (fit.surface.implied_vol_k(k, T) - p.loc[m, "iv_mid"].to_numpy(float))
    return float(np.sqrt(np.mean(e**2))), float(np.max(np.abs(e)))


def _metrics(tag: str, fit: ih.SSVIFit, seconds: float) -> dict[str, Any]:
    s = fit.surface
    out: dict[str, Any] = {
        f"{tag}_essvi": isinstance(s, ESSVISurface),
        f"{tag}_rms": fit.rms_error(ERR_T_MAX, ERR_K_ABS, ERR_T_MIN),
        f"{tag}_max": fit.max_error(ERR_T_MAX, ERR_K_ABS, ERR_T_MIN),
        f"{tag}_seconds": seconds,
    }
    out[f"{tag}_rms_surface"], out[f"{tag}_max_surface"] = _surface_errors(fit)
    if isinstance(s, ESSVISurface):
        out.update(_calendar_columns(tag, s))
    for T in SKEW_TENORS:
        out[f"{tag}_skew{T:g}"] = float(s.atm_skew(T))
    return out


def _calendar_columns(tag: str, s: ESSVISurface) -> dict[str, Any]:
    """Every calendar quantity of one eSSVI surface (per-year derivatives, ``Δw`` steps)."""
    out: dict[str, Any] = {
        f"{tag}_min_dwdt1": s.calendar_min_dw_dt(),
        f"{tag}_min_dwdt3": s.calendar_min_dw_dt(K_WIDE, N_K_WIDE),
        f"{tag}_min_dw1_legacy": s.calendar_min_dw(),
        f"{tag}_min_dw1_m3b": m3b_min_dw(s),
        f"{tag}_min_dw3_legacy": s.calendar_min_dw(K_WIDE, N_K_WIDE),
    }
    dense = s.calendar_dense_check()
    for name in DENSE_FIELDS:
        out[f"{tag}_dense_{name}"] = float(getattr(dense, name))
    out[f"{tag}_dense_ok"] = dense.ok
    out[f"{tag}_cert0_k3"] = dense.certificate.status
    out[f"{tag}_cert0_k3_lb"] = dense.certificate.lower_bound
    c1 = s.calendar_certificate(s.CALENDAR_K_ABS, 0.0)
    out[f"{tag}_cert0_k1"] = c1.status
    cm = s.calendar_certificate(K_WIDE, MARGIN)
    out[f"{tag}_certm_k3"] = cm.status
    out[f"{tag}_certm_k3_lb"] = cm.lower_bound
    return out


def committed_snapshots() -> dict[str, Any]:
    """Every committed YAML under ``configs/surfaces``: the plain-SSVI ones are counted (their
    constructor has no numeric calendar check, so the grid change cannot move them); each eSSVI
    one is loaded with the checked constructor (does it still load on the pillar-aware grid?)
    and with an unchecked one for the slacks on both grids and the dense check."""
    rows: list[dict[str, Any]] = []
    n_plain = 0
    for path in sorted(SNAPSHOT_ROOT.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not (isinstance(raw, dict) and "ssvi" in raw and "market" in raw):
            continue
        if "essvi" not in raw:
            n_plain += 1
            continue
        row: dict[str, Any] = {"path": str(path.relative_to(ROOT))}
        try:
            load_ssvi_surface(path)
            row["loads"] = True
        except ValueError as exc:
            row["loads"] = False
            row["error"] = str(exc)
        with mock.patch("volsto.market.loaders.ESSVISurface", _RecordingESSVI):
            u = load_ssvi_surface(path)
        assert isinstance(u, ESSVISurface)
        row.update({k.removeprefix("s_"): v for k, v in _calendar_columns("s", u).items()})
        row["loads_legacy"] = bool(row["min_dw1_legacy"] >= -TOL)
        row["loads_m3b"] = bool(row["min_dw1_m3b"] >= -TOL)
        rows.append(row)
    return {
        "n_plain_ssvi": n_plain,
        "essvi": rows,
        "status_changed_vs_legacy": [r["path"] for r in rows if r["loads"] != r["loads_legacy"]],
        "status_changed_vs_m3b": [r["path"] for r in rows if r["loads"] != r["loads_m3b"]],
    }


def run_day(root: Path, date: str, *, with_ssvi: bool) -> dict[str, Any]:
    f = ih.HdnFilters()
    manifest = ih.load_manifest(root)
    chain = ih.load_day(root / "day_by_date" / f"{date}_options.csv", "SPX", manifest=manifest)
    fwds = ih.implied_forwards(chain, max_years=f.max_years, band=f.near_atm_band)
    grid, pts = ih.to_grid_surface(chain, fwds, f)
    row: dict[str, Any] = {"date": date}
    t = time.perf_counter()
    with _unchecked_essvi():
        base = ih.fit_ssvi(grid, pts, filters=f, essvi=True, calendar_repair=None)
    row.update(_metrics("base", base, time.perf_counter() - t))
    row["base_pass"] = bool(row["base_min_dwdt1"] >= -TOL)
    row["base_pass_legacy"] = bool(row["base_min_dw1_legacy"] >= -TOL)
    row["base_pass_m3b"] = bool(row["base_min_dw1_m3b"] >= -TOL)
    t = time.perf_counter()
    rep = ih.fit_ssvi(grid, pts, filters=f, essvi=True)
    row.update(_metrics("rep", rep, time.perf_counter() - t))
    pil = list(rep.params["atm_maturities"])
    row.update(
        P=len(pil),
        last_pillar=float(pil[-1]),
        max_quoted_T=float(pts.table["T"].max()),
        rep_pass=bool(isinstance(rep.surface, ESSVISurface)),
        repaired=bool(rep.params.get("calendar_repaired", False)),
        stages=int(rep.params.get("calendar_stages", 0)),
        cuts=int(rep.params.get("calendar_cuts", 0)),
        fallback=rep.params.get("calendar_fallback"),
        cost_delta=rep.params.get("calendar_cost_delta"),
        min_dw_dt=rep.params.get("calendar_min_dw_dt"),
        min_dw_dt_before_repair=rep.params.get("calendar_min_dw_dt_before"),
        proven_floor=rep.params.get("calendar_floor"),
        proven_k_abs=rep.params.get("calendar_k_abs"),
        proven_lower_bound=rep.params.get("calendar_lower_bound"),
        rho_base=json.dumps([round(float(r), 5) for r in base.params["rho"]]),
        rho_rep=json.dumps(
            [round(float(r), 5) for r in np.atleast_1d(np.asarray(rep.params["rho"]))]
        ),
    )
    if with_ssvi:
        t = time.perf_counter()
        ssvi = ih.fit_ssvi(grid, pts, filters=f, essvi=False)
        assert isinstance(ssvi.surface, SSVISurface)
        row.update(_metrics("ssvi", ssvi, time.perf_counter() - t))
    return row


def _med(x: pd.Series) -> float:
    return float(np.median(x.to_numpy(float))) if len(x) else float("nan")


def _dense_summary(df: pd.DataFrame, v: str) -> dict[str, Any]:
    """Dense-check summary of variant ``v`` over the days it returned an eSSVI surface (a day
    that fell back to plain SSVI has no ``ρ_T`` to check and is counted apart)."""
    e = df[df[f"{v}_essvi"].astype(bool)]
    neg = e[f"{v}_dense_min_dw_dt"] < 0.0
    out: dict[str, Any] = {
        "n_essvi_days": len(e),
        "n_not_essvi": len(df) - len(e),
        "days_negative": int(neg.sum()),
        "dates_negative": e.loc[neg, "date"].tolist(),
    }
    if len(e):
        out.update(
            {
                name: float(e[f"{v}_dense_{name}"].min())
                for name in (
                    "min_dw_dt",
                    "min_interior",
                    "min_left",
                    "min_right",
                    "min_slope",
                    "min_dw",
                )
            }
        )
        out["worst_date"] = str(e.loc[e[f"{v}_dense_min_dw_dt"].idxmin(), "date"])
        out["days_not_ok"] = int((~e[f"{v}_dense_ok"].astype(bool)).sum())
        for c in ("cert0_k3", "cert0_k1", "certm_k3"):
            out[c] = {str(k): int(n) for k, n in e[f"{v}_{c}"].value_counts().items()}
        proven = e[e[f"{v}_certm_k3"] == "certified"]
        out["certm_k3_min_lower_bound"] = (
            float(proven[f"{v}_certm_k3_lb"].min()) if len(proven) else float("nan")
        )
        # the constructor's point check passes but the floor-0 certificate on its own range fails
        out["ctor_pass_cert_k1_not"] = e.loc[
            (e[f"{v}_min_dwdt1"] >= -TOL) & (e[f"{v}_cert0_k1"] != "certified"), "date"
        ].tolist()
    return out


def summarise(df: pd.DataFrame, wall: float, with_ssvi: bool) -> dict[str, Any]:
    n = len(df)
    ch = df[df["repaired"]]
    s: dict[str, Any] = {
        "n_days": n,
        "wall_seconds": wall,
        "recalibrated": False,
        "passing": {
            "base": int(df["base_pass"].sum()),
            "repaired": int(df["rep_pass"].sum()),
            "repaired_fraction": float(df["rep_pass"].mean()),
            "gate_fraction": GATE_PASS_FRACTION,
            "gate_met": bool(df["rep_pass"].mean() >= GATE_PASS_FRACTION),
            "n_repaired": int(df["repaired"].sum()),
            "n_identity": int((~df["repaired"] & df["rep_pass"]).sum()),
            "n_identity_bit_equal": int(
                (
                    ~df["repaired"]
                    & df["rep_pass"]
                    & (df["rho_rep"] == df["rho_base"])
                    & (df["rep_rms"] == df["base_rms"])
                    & (df["rep_max"] == df["base_max"])
                    & np.logical_and.reduce(
                        [df[f"rep_skew{T:g}"] == df[f"base_skew{T:g}"] for T in SKEW_TENORS]
                    )
                ).sum()
            ),
            "fallbacks": {str(k): int(v) for k, v in df["fallback"].value_counts().items()},
            "stages_on_repaired_days": {
                str(k): int(v) for k, v in ch["stages"].value_counts().sort_index().items()
            },
            "cuts_on_repaired_days": {
                str(k): int(v) for k, v in ch["cuts"].value_counts().sort_index().items()
            },
            "proven_floor": {str(k): int(v) for k, v in df["proven_floor"].value_counts().items()},
            "min_proven_lower_bound": float(df["proven_lower_bound"].min()),
        },
        "slack": {
            "base_days_neg_k1": int((df["base_min_dwdt1"] < -TOL).sum()),
            "base_days_neg_k3": int((df["base_min_dwdt3"] < -TOL).sum()),
            "base_min_k1": float(df["base_min_dwdt1"].min()),
            "base_min_k3": float(df["base_min_dwdt3"].min()),
            "rep_days_neg_k1": int((df["rep_min_dwdt1"] < -TOL).sum()),
            "rep_days_neg_k3": int((df["rep_min_dwdt3"] < -TOL).sum()),
            "rep_min_k1": float(df["rep_min_dwdt1"].min()),
            "rep_min_k3": float(df["rep_min_dwdt3"].min()),
            "base_pass_legacy": int(df["base_pass_legacy"].sum()),
            "base_pass_m3b": int(df["base_pass_m3b"].sum()),
            "base_status_changed_vs_legacy": df.loc[
                df["base_pass"] != df["base_pass_legacy"], "date"
            ].tolist(),
            "base_status_changed_vs_m3b": df.loc[
                df["base_pass"] != df["base_pass_m3b"], "date"
            ].tolist(),
        },
        "dense": {v: _dense_summary(df, v) for v in ("base", "rep")},
        "errors": {},
        "skew": {},
        "pillars": {
            "P_distribution": {
                str(k): int(v) for k, v in df["P"].value_counts().sort_index().items()
            },
            "last_pillar_distribution": {
                f"{k:g}": int(v) for k, v in df["last_pillar"].value_counts().sort_index().items()
            },
            "days_3y_extrapolated": int((df["last_pillar"] < 3.0 - 1e-9).sum()),
            "days_2y_extrapolated": int((df["last_pillar"] < 2.0 - 1e-9).sum()),
            "days_quote_at_or_beyond_3y": int((df["max_quoted_T"] >= 3.0 - 1e-9).sum()),
            "median_max_quoted_T": _med(df["max_quoted_T"]),
        },
    }
    variants = ["base", "rep"] + (["ssvi"] if with_ssvi else [])
    for v in variants:
        s["errors"][v] = {
            "median_rms": _med(df[f"{v}_rms"]),
            "median_max": _med(df[f"{v}_max"]),
            "median_rms_surface": _med(df[f"{v}_rms_surface"]),
            "median_max_surface": _med(df[f"{v}_max_surface"]),
        }
        s["skew"][v] = {f"{T:g}y": _med(df[f"{v}_skew{T:g}"]) for T in SKEW_TENORS}
    if len(ch):
        worst = ch.loc[ch["rep_rms"].idxmax()]
        d_rms = ch["rep_rms"] - ch["base_rms"]
        d_max = ch["rep_max"] - ch["base_max"]
        s["errors"]["changed_days"] = {
            "n": len(ch),
            "dates": ch["date"].tolist(),
            "median_rms_base": _med(ch["base_rms"]),
            "median_rms_rep": _med(ch["rep_rms"]),
            "median_max_base": _med(ch["base_max"]),
            "median_max_rep": _med(ch["rep_max"]),
            "worst_day_rms": {"date": worst["date"], "rms": float(ch["rep_rms"].max())},
            "worst_day_max": {
                "date": ch.loc[ch["rep_max"].idxmax(), "date"],
                "max": float(ch["rep_max"].max()),
            },
            "worst_rms_increase": {
                "date": ch.loc[d_rms.idxmax(), "date"],
                "delta": float(d_rms.max()),
            },
            "worst_max_increase": {
                "date": ch.loc[d_max.idxmax(), "date"],
                "delta": float(d_max.max()),
            },
            "median_rms_increase": _med(d_rms),
            "seconds_per_repaired_fit_median": _med(ch["rep_seconds"]),
        }
    gaps: dict[str, Any] = {}
    if with_ssvi:
        for v in ("base", "rep"):
            gaps[v] = {
                f"{T:g}y": _med(df[f"{v}_skew{T:g}"] - df[f"ssvi_skew{T:g}"]) for T in SKEW_TENORS
            }
        passing = df[df["rep_pass"]]
        gaps["rep_on_passing_days_n"] = len(passing)
    s["skew"]["median_gap_vs_ssvi"] = gaps
    s["skew"]["median_change_rep_minus_base"] = {
        f"{T:g}y": _med(df[f"rep_skew{T:g}"] - df[f"base_skew{T:g}"]) for T in SKEW_TENORS
    }
    s["skew"]["max_abs_change_rep_minus_base"] = {
        f"{T:g}y": float(np.max(np.abs(df[f"rep_skew{T:g}"] - df[f"base_skew{T:g}"])))
        for T in SKEW_TENORS
    }
    if ANCHOR in set(df["date"]):
        a = df[df["date"] == ANCHOR].iloc[0]
        s["anchor"] = {
            k: (a[k].item() if hasattr(a[k], "item") else a[k])
            for k in a.index
            if not k.endswith("_seconds")
        }
    return s


def compare_baseline(df: pd.DataFrame, path: Path) -> dict[str, Any]:
    """Fit error of this run's repaired eSSVI against an earlier gate run's per-day table
    (``essvi_gate_days.csv``): what the proven exact-derivative invariant costs relative to that
    run's repair (all days, days repaired in either run, worst day)."""
    b = pd.read_csv(path, dtype={"date": str})
    m = df.merge(b[["date", "rep_rms", "rep_max", "repaired"]], on="date", suffixes=("", "_b"))
    either = m[m["repaired"].astype(bool) | m["repaired_b"].astype(bool)]
    d_rms = m["rep_rms"] - m["rep_rms_b"]
    d_max = m["rep_max"] - m["rep_max_b"]
    return {
        "path": str(path),
        "n_common": len(m),
        "n_repaired_now": int(m["repaired"].sum()),
        "n_repaired_baseline": int(m["repaired_b"].sum()),
        "newly_repaired": m.loc[
            m["repaired"].astype(bool) & ~m["repaired_b"].astype(bool), "date"
        ].tolist(),
        "median_rms_all": [_med(m["rep_rms_b"]), _med(m["rep_rms"])],
        "median_max_all": [_med(m["rep_max_b"]), _med(m["rep_max"])],
        "median_rms_either": [_med(either["rep_rms_b"]), _med(either["rep_rms"])],
        "median_max_either": [_med(either["rep_max_b"]), _med(either["rep_max"])],
        "n_either": len(either),
        "median_rms_change_either": _med(either["rep_rms"] - either["rep_rms_b"]),
        "worst_rms_change": {
            "date": str(m.loc[d_rms.idxmax(), "date"]),
            "delta": float(d_rms.max()),
        },
        "worst_max_change": {
            "date": str(m.loc[d_max.idxmax(), "date"]),
            "delta": float(d_max.max()),
        },
        "min_rms_change": float(d_rms.min()),
    }


def _fmt(x: Any, nd: int = 4) -> str:
    if isinstance(x, float):
        return f"{x:.{nd}g}" if abs(x) < 1e-3 and x != 0.0 else f"{x:.{nd}f}"
    return str(x)


def write_report(out: Path, df: pd.DataFrame, s: dict[str, Any], args: argparse.Namespace) -> Path:
    p, sl, er, sk, pl = s["passing"], s["slack"], s["errors"], s["skew"], s["pillars"]
    lines = [
        "# M10 Part 0 — eSSVI calendar repair gate",
        "",
        f"Command: `.venv/bin/python scripts/essvi_calendar_gate.py --out {args.out}"
        + (f" --limit {args.limit}" if args.limit else "")
        + ("" if not args.no_ssvi else " --no-ssvi")
        + (f" --baseline-days {args.baseline_days}" if args.baseline_days else "")
        + "`",
        f"Wall clock {s['wall_seconds_total']:.1f} s ({s['wall_seconds']:.1f} s for the day "
        "fits), single core, recalibrated: **no** "
        "(surface fitting only, no Monte Carlo — so no standard errors apply).",
        "",
        "## Days passing",
        "",
        f"* eSSVI constructs (calendar check on ±1 passes): base **{p['base']}/{s['n_days']}**, "
        f"repaired **{p['repaired']}/{s['n_days']}** ({100 * p['repaired_fraction']:.1f}%; gate "
        f"≥ {100 * GATE_PASS_FRACTION:.0f}%: **{'met' if p['gate_met'] else 'NOT met'}**).",
        f"* Repaired days {p['n_repaired']}, exact identity {p['n_identity']} (rho, RMS, max and "
        f"skews equal to the unrepaired fit on {p['n_identity_bit_equal']}), fallbacks "
        f"{p['fallbacks'] or 'none'}; penalty solves used {p['stages_on_repaired_days']}; "
        f"certificate cuts {p['cuts_on_repaired_days']}.",
        f"* Proven invariant (∂_T w ≥ floor on ±3, every T, both knot limits): floors "
        f"{p['proven_floor']}; smallest certificate lower bound "
        f"{_fmt(p['min_proven_lower_bound'])} per year.",
        f"* Constructor quantity (exact min ∂_T w, both knot limits) below -1e-10: base "
        f"{sl['base_days_neg_k1']} days on ±1, {sl['base_days_neg_k3']} on ±3; repaired "
        f"{sl['rep_days_neg_k1']} on ±1, {sl['rep_days_neg_k3']} on ±3.  Smallest: base "
        f"{_fmt(sl['base_min_k1'])} (±1), {_fmt(sl['base_min_k3'])} (±3); repaired "
        f"{_fmt(sl['rep_min_k1'])} (±1), {_fmt(sl['rep_min_k3'])} (±3) per year.",
        f"* Constructor check: exact ∂_T w.  Unrepaired eSSVI passes on {p['base']} days, against "
        f"{sl['base_pass_legacy']} under the second pass's Δw check (pillar-aware grid) and "
        f"{sl['base_pass_m3b']} under M3b's (no pillars); status changed vs the second pass on "
        f"{sl['base_status_changed_vs_legacy'] or 'no day'}, vs M3b on "
        f"{sl['base_status_changed_vs_m3b'] or 'no day'}.",
        "",
        "## Dense check (exact ∂_T w on 1201 k on ±3 x 3000 T plus both limits at every knot; "
        "secant and Δw on the same grid; floor-0 certificate), per year",
        "",
        "| variant | eSSVI days | days with ∂_T w < 0 | not ok | min ∂_T w | min interior "
        "| min left | min right | min secant | min Δw | worst day |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for v in ("base", "rep"):
        dn = s["dense"][v]
        nan = float("nan")
        lines.append(
            f"| {v} | {dn['n_essvi_days']} | **{dn['days_negative']}** | "
            f"{dn.get('days_not_ok', '-')} | "
            f"{_fmt(dn.get('min_dw_dt', nan))} | {_fmt(dn.get('min_interior', nan))} | "
            f"{_fmt(dn.get('min_left', nan))} | {_fmt(dn.get('min_right', nan))} | "
            f"{_fmt(dn.get('min_slope', nan))} | {_fmt(dn.get('min_dw', nan))} | "
            f"{dn.get('worst_date', '-')} |"
        )
    lines += [""]
    for v in ("base", "rep"):
        dn = s["dense"][v]
        lines.append(
            f"* {v}: certificate floor 0 on ±3 {dn.get('cert0_k3')}, floor 0 on ±1 "
            f"{dn.get('cert0_k1')}, floor {MARGIN:g} on ±3 {dn.get('certm_k3')} (smallest proven "
            f"bound {_fmt(dn.get('certm_k3_min_lower_bound', float('nan')))}); constructor "
            f"passes but the ±1 certificate does not: {dn.get('ctor_pass_cert_k1_not') or 'none'}."
        )
    lines += [
        "",
        f"Repaired days with ∂_T w < 0 somewhere: {s['dense']['rep']['dates_negative'] or 'none'}"
        f"; days not eSSVI after the repair: {s['dense']['rep']['n_not_essvi']}.",
    ]
    cs = s["committed"]
    lines += [
        "",
        "## Committed snapshots under configs/surfaces",
        "",
        f"{cs['n_plain_ssvi']} plain-SSVI snapshots (no numeric calendar check: unaffected by the "
        f"check change); {len(cs['essvi'])} eSSVI snapshots; status changed vs the second pass: "
        f"**{cs['status_changed_vs_legacy'] or 'none'}**, vs M3b: "
        f"**{cs['status_changed_vs_m3b'] or 'none'}**.",
        "",
        "| eSSVI snapshot | loads (exact) | loads (Δw pillar grid) | loads (M3b) | min ∂_T w ±1 "
        "| min ∂_T w ±3 | dense min ∂_T w | cert ≥ 0 ±3 | cert ≥ margin ±3 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in cs["essvi"]:
        lines.append(
            f"| {r['path']} | {r['loads']} | {r['loads_legacy']} | {r['loads_m3b']} | "
            f"{_fmt(r['min_dwdt1'])} | {_fmt(r['min_dwdt3'])} | {_fmt(r['dense_min_dw_dt'])} | "
            f"{r['cert0_k3']} | {r['certm_k3']} |"
        )
    lines += [
        "",
        "## Fit error, |k| ≤ 0.20, 3m-3y, vol points (medians over days)",
        "",
        "| variant | RMS (fit's points) | max (fit's points) | RMS (surface returned) "
        "| max (surface returned) |",
        "|---|---|---|---|---|",
    ]
    for v in [k for k in ("base", "rep", "ssvi") if k in er]:
        e = er[v]
        lines.append(
            f"| {v} | {e['median_rms']:.4f} | {e['median_max']:.4f} | "
            f"{e['median_rms_surface']:.4f} | {e['median_max_surface']:.4f} |"
        )
    if "changed_days" in er:
        c = er["changed_days"]
        lines += [
            "",
            f"Changed days (n = {c['n']}): median RMS {c['median_rms_base']:.4f} → "
            f"{c['median_rms_rep']:.4f}, median max {c['median_max_base']:.4f} → "
            f"{c['median_max_rep']:.4f}; worst day RMS {c['worst_day_rms']['rms']:.4f} "
            f"({c['worst_day_rms']['date']}), worst max {c['worst_day_max']['max']:.4f} "
            f"({c['worst_day_max']['date']}); worst RMS increase "
            f"{c['worst_rms_increase']['delta']:+.4f} ({c['worst_rms_increase']['date']}), worst "
            f"max increase {c['worst_max_increase']['delta']:+.4f} "
            f"({c['worst_max_increase']['date']}); median repaired fit "
            f"{c['seconds_per_repaired_fit_median']:.2f} s.",
            "",
            "The 'fit's points' columns are the repository's reported convention (`fit_ssvi` "
            "clamps θ beyond the last pillar); the 'surface returned' columns evaluate the "
            "surface the importer hands back (θ extrapolated) — the defect is reported, not "
            "fixed, in this change.",
        ]
    lines += [
        "",
        "## ATMF skew (d sigma / dk at k = 0), medians over days",
        "",
        "| variant | 1y | 2y | 3y |",
        "|---|---|---|---|",
    ]
    for v in [k for k in ("base", "rep", "ssvi") if k in sk]:
        lines.append(
            f"| {v} | " + " | ".join(f"{sk[v][f'{T:g}y']:.4f}" for T in SKEW_TENORS) + " |"
        )
    g = sk["median_gap_vs_ssvi"]
    for v in ("base", "rep"):
        if v in g:
            lines.append(
                f"| {v} - ssvi (median of daily gaps) | "
                + " | ".join(f"{g[v][f'{T:g}y']:+.4f}" for T in SKEW_TENORS)
                + " |"
            )
    lines.append(
        "| rep - base (median / max abs) | "
        + " | ".join(
            f"{sk['median_change_rep_minus_base'][f'{T:g}y']:+.4f} / "
            f"{sk['max_abs_change_rep_minus_base'][f'{T:g}y']:.4f}"
            for T in SKEW_TENORS
        )
        + " |"
    )
    if "anchor" in s:
        a = s["anchor"]
        lines += [
            "",
            f"## Anchor {ANCHOR}",
            "",
            f"P = {a['P']}, repaired {a['repaired']} in {a['stages']} stages; RMS "
            f"{a['base_rms']:.4f} → {a['rep_rms']:.4f}; max {a['base_max']:.4f} → "
            f"{a['rep_max']:.4f}; skew 2y {a['base_skew2']:.4f} → {a['rep_skew2']:.4f}"
            + (f" (SSVI {a['ssvi_skew2']:.4f})" if "ssvi_skew2" in a else "")
            + f", 3y {a['base_skew3']:.4f} → {a['rep_skew3']:.4f}"
            + (f" (SSVI {a['ssvi_skew3']:.4f})" if "ssvi_skew3" in a else "")
            + f"; exact min ∂_T w ±3 (constructor grid) {_fmt(a['base_min_dwdt3'])} → "
            f"{_fmt(a['rep_min_dwdt3'])}; dense min ∂_T w {_fmt(a['base_dense_min_dw_dt'])} → "
            f"{_fmt(a['rep_dense_min_dw_dt'])} per year; proven floor {a['proven_floor']} "
            f"(lower bound {_fmt(a['proven_lower_bound'])}); cuts {a['cuts']}.",
            f"rho_T base {a['rho_base']}; repaired {a['rho_rep']}.",
        ]
    if "baseline" in s:
        bl = s["baseline"]
        lines += [
            "",
            "## Against the baseline run (cost of the exact, proven invariant)",
            "",
            f"Baseline `{bl['path']}` ({bl['n_common']} common days): repaired "
            f"{bl['n_repaired_baseline']} → {bl['n_repaired_now']} days (newly repaired: "
            f"{bl['newly_repaired'] or 'none'}).  Repaired-eSSVI median RMS, all days "
            f"{bl['median_rms_all'][0]:.4f} → {bl['median_rms_all'][1]:.4f}, median max "
            f"{bl['median_max_all'][0]:.4f} → {bl['median_max_all'][1]:.4f}; on the "
            f"{bl['n_either']} days repaired in either run RMS {bl['median_rms_either'][0]:.4f} → "
            f"{bl['median_rms_either'][1]:.4f} (median change "
            f"{bl['median_rms_change_either']:+.4f}), max {bl['median_max_either'][0]:.4f} → "
            f"{bl['median_max_either'][1]:.4f}; largest RMS increase "
            f"{bl['worst_rms_change']['delta']:+.4f} ({bl['worst_rms_change']['date']}), largest "
            f"max increase {bl['worst_max_change']['delta']:+.4f} "
            f"({bl['worst_max_change']['date']}); largest RMS decrease "
            f"{bl['min_rms_change']:+.4f}.",
        ]
    lines += [
        "",
        "## Pillars (why the 3y skew stays extrapolated)",
        "",
        f"Pillar count P: {pl['P_distribution']}; last pillar: {pl['last_pillar_distribution']}.  "
        f"3y is beyond the last pillar on **{pl['days_3y_extrapolated']}/{s['n_days']}** days "
        f"(flat rho_T, linear theta_T there, whatever the repair); 2y on "
        f"{pl['days_2y_extrapolated']}.  Days with a quote at or beyond 3y: "
        f"{pl['days_quote_at_or_beyond_3y']}; median longest quoted maturity "
        f"{pl['median_max_quoted_T']:.3f} y.  What the repair delivers on every passing day is "
        "the 2y/3y skew from the eSSVI rho_T instead of the single-rho fallback.",
        "",
        "Per-day table: `essvi_gate_days.csv`; summary: `essvi_gate.json`.",
    ]
    path = out / "essvi_gate.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default="data/hdn_sample/options_sample_2022H2")
    ap.add_argument("--out", default="outputs/essvi_gate")
    ap.add_argument("--limit", type=int, default=None, help="first N sample days only")
    ap.add_argument("--no-ssvi", action="store_true", help="skip the plain-SSVI comparison fit")
    ap.add_argument(
        "--baseline-days",
        type=Path,
        default=None,
        help="essvi_gate_days.csv of an earlier run: report the repaired fit error against it",
    )
    args = ap.parse_args()
    out = Path(args.out)
    if out.resolve().is_relative_to((ROOT / "configs").resolve()):
        sys.exit(f"refusing to write under configs/: {out}")
    out.mkdir(parents=True, exist_ok=True)
    root = Path(args.root)
    dates = hdn_available_dates(root)
    if args.limit:
        dates = dates[: args.limit]
    t0 = time.perf_counter()
    rows = []
    for i, d in enumerate(dates):
        rows.append(run_day(root, d, with_ssvi=not args.no_ssvi))
        r = rows[-1]
        print(
            f"[{i + 1}/{len(dates)}] {d} P={r['P']} base_pass={r['base_pass']} "
            f"repaired={r['repaired']} stages={r['stages']} rms {r['base_rms']:.4f}->"
            f"{r['rep_rms']:.4f} ({time.perf_counter() - t0:.0f} s)",
            flush=True,
        )
    wall = time.perf_counter() - t0
    df = pd.DataFrame(rows)
    df.to_csv(out / "essvi_gate_days.csv", index=False)
    s = summarise(df, wall, not args.no_ssvi)
    s["committed"] = committed_snapshots()
    if args.baseline_days:
        s["baseline"] = compare_baseline(df, args.baseline_days)
    s["wall_seconds_total"] = time.perf_counter() - t0
    (out / "essvi_gate.json").write_text(json.dumps(s, indent=2, default=str), encoding="utf-8")
    md = write_report(out, df, s, args)
    print(md.read_text(encoding="utf-8"))
    print(f"written {md}; wall clock {s['wall_seconds_total']:.1f} s; recalibrated: no")


if __name__ == "__main__":
    main()
