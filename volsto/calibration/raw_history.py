"""Raw-slice-skew discriminator for the historical SSR (SPEC §15 Part 2, M8b gate).

The 2022 H2 SSR of 0.81–0.84 (60 days) was estimated from SSVI-fitted daily snapshots; the owner
asks whether that ``SSR < 1`` is a property of the quotes or of the fit.  This module rebuilds
the pillar history **without any surface parametrisation**: per date the importer pipeline is
run up to the retained quotes (:func:`volsto.market.import_hdn.to_grid_surface`, no SSVI fit),
each retained slice gets a local quadratic ``iv(k) = a + b k + c k²`` by unweighted least squares
on the raw mid implied vols inside ``|k| ≤ fit_band`` (``a`` the raw ATMF vol, ``b`` the raw
ATMF skew per unit log-moneyness — the convention of
:class:`~volsto.calibration.history.SurfacePillarSource`), and the pillars are interpolated from
the bracketing slices (skew linear in ``T``, ATM vol linear in total variance ``a² T``).  The
same :meth:`~volsto.calibration.history.SurfaceHistory.ssr_hist` estimator is then run on the
raw and the SSVI frames over the common dates and the two SSRs compared pillar by pillar.

``vs_vol`` is not a raw quantity (it needs the whole strip): it is copied from the SSVI history
when one is given, otherwise set to the raw ATM vol — the SSR estimator reads ``atm_vol``,
``skew`` and ``ln_spot`` only, so this choice cannot affect the discriminator.

Verdict rule (owner's gate for M8b study B world (ii)): ``"real"`` when, at every pillar
``≤ 1y`` on the short window, ``|SSR_raw − SSR_ssvi| ≤ Z_AGREE · sqrt(se_raw² + se_ssvi²)``
(independent-standard-error approximation: the two estimates share the spot increments and are
positively correlated, so the true standard error of the difference is smaller and this test is
conservative) **and** ``SSR_raw + Z_AGREE · se_raw < 1``; ``"surface artefact"`` otherwise, with
the reason.  Checked by ``tests/test_raw_history.py``; run by ``scripts/m8b_discriminator.py``.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.calibration.history import REQUIRED_COLUMNS, SurfaceHistory, estimate_history
from volsto.market.import_hdn import (
    HdnFilters,
    SurfacePoints,
    implied_forwards,
    load_day,
    load_manifest,
    to_grid_surface,
)

FloatArray = NDArray[np.float64]

RAW_PILLARS: tuple[float, ...] = (1.0 / 12.0, 0.25, 0.5, 1.0)
"""The pillars the SSVI history supports on the 2022 H2 sample (SPEC §15: 2y / 3y sit on the
SSVI extrapolation there)."""
FIT_BAND = 0.10
"""``|k|`` band of the local quadratic: wide enough for 20-130 retained quotes per SPX slice on
the sample (5 is the floor), narrow enough that the quadratic's slope at ``k = 0`` is the ATMF
skew and not a wing average — the same ``±10%`` window the importer's parity regression uses."""
MIN_SLICE_POINTS = 5
"""A quadratic has three coefficients; five points leave two degrees of freedom."""
WINDOW_SSR = 60
"""Short SSR window (SPEC §15 Part 2)."""
WINDOW_LONG = 100
"""Long window: the 127-day sample has no 250-day window (SPEC §15 real-data run)."""
Z_AGREE = 2.0
"""Two-standard-error agreement and below-one tests (the owner's gate)."""
MAX_PILLAR_GATE = 1.0
"""The gate applies to the pillars up to 1y (the SSVI-supported ones)."""
_TOL_T = 1e-9


@dataclass(frozen=True)
class SliceFit:
    """Local quadratic of one retained slice: ``iv(k) ≈ atm_vol + skew k + curvature k²``."""

    T: float
    atm_vol: float
    skew: float
    curvature: float
    n: int


def fit_slice_quadratic(k: FloatArray, iv: FloatArray) -> tuple[float, float, float]:
    """Unweighted least-squares ``(a, b, c)`` of ``iv = a + b k + c k²``; exact on a quadratic."""
    k = np.asarray(k, dtype=np.float64)
    iv = np.asarray(iv, dtype=np.float64)
    if k.size < MIN_SLICE_POINTS:
        raise ValueError(f"a slice fit needs at least {MIN_SLICE_POINTS} points, got {k.size}")
    c, b, a = np.polyfit(k, iv, 2)
    return float(a), float(b), float(c)


def slice_fits(points: SurfacePoints, fit_band: float = FIT_BAND) -> list[SliceFit]:
    """One :class:`SliceFit` per retained maturity with at least :data:`MIN_SLICE_POINTS` quotes
    inside ``|k| ≤ fit_band`` (``iv = sqrt(w / T)`` from the table's total variance)."""
    t = points.table
    near = t[t["k"].abs() <= fit_band]
    T_all = near["T"].to_numpy(dtype=np.float64)
    out: list[SliceFit] = []
    for T in np.unique(T_all):
        g = near[T_all == T]
        if len(g) < MIN_SLICE_POINTS:
            continue
        k = g["k"].to_numpy(dtype=np.float64)
        iv = np.sqrt(g["w"].to_numpy(dtype=np.float64) / float(T))
        a, b, c = fit_slice_quadratic(k, iv)
        out.append(SliceFit(float(T), a, b, c, len(g)))
    return out


def interpolate_pillars(
    fits: list[SliceFit], pillars: tuple[float, ...]
) -> tuple[FloatArray, FloatArray]:
    """``(atm_vol, skew)`` at each pillar from the bracketing slices: skew linear in ``T``,
    ATM vol linear in total variance ``a² T``; NaN outside ``[T_min, T_max]`` of the slices."""
    order = np.argsort([f.T for f in fits], kind="stable")  # np.interp needs increasing T
    Ts = np.array([fits[i].T for i in order], dtype=np.float64)
    a = np.array([fits[i].atm_vol for i in order], dtype=np.float64)
    b = np.array([fits[i].skew for i in order], dtype=np.float64)
    if np.any(np.diff(Ts) <= 0):
        raise ValueError("slice maturities must be distinct")
    pill = np.asarray(pillars, dtype=np.float64)
    atm = np.full(pill.shape, np.nan)
    skew = np.full(pill.shape, np.nan)
    if Ts.size == 0:
        return atm, skew
    inside = (pill >= Ts[0] - _TOL_T) & (pill <= Ts[-1] + _TOL_T)
    w = np.interp(pill[inside], Ts, a * a * Ts)
    atm[inside] = np.sqrt(w / pill[inside])
    skew[inside] = np.interp(pill[inside], Ts, b)
    return atm, skew


def raw_day(
    root: str | Path,
    date: str,
    *,
    underlying: str = "SPX",
    fit_band: float = FIT_BAND,
    manifest: dict[str, Any] | None = None,
    filters: HdnFilters | None = None,
) -> tuple[list[SliceFit], float, SurfacePoints]:
    """Importer pipeline for one date up to the retained quotes, then the slice fits.
    Returns ``(fits, spot, points)``."""
    root = Path(root)
    f = filters or HdnFilters()
    chain = load_day(
        root / "day_by_date" / f"{date}_options.csv",
        underlying,
        manifest=manifest or load_manifest(root),
    )
    fwds = implied_forwards(chain, max_years=f.max_years, band=f.near_atm_band)
    _, points = to_grid_surface(chain, fwds, f)
    return slice_fits(points, fit_band), float(chain.attrs["spot"]), points


def raw_pillar_frame(
    root: str | Path,
    dates: list[str] | tuple[str, ...],
    pillars: tuple[float, ...] = RAW_PILLARS,
    *,
    underlying: str = "SPX",
    fit_band: float = FIT_BAND,
    ssvi_history: str | Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The raw pillar history in the :class:`SurfaceHistory` frame format and the per-date
    diagnostics ``date, n_slices, n_points, t_min, t_max, kept, reason``.

    A date with any pillar outside the bracket of its retained slices (or an importer error) is
    dropped for all pillars and recorded in the diagnostics.  ``vs_vol`` is copied from
    ``ssvi_history`` (same date and pillar) when given, else equals ``atm_vol`` (module
    docstring).  About 1 s per date (the importer)."""
    root = Path(root)
    manifest = load_manifest(root)
    P = tuple(float(p) for p in pillars)
    rows: list[dict[str, float | str]] = []
    diag: list[dict[str, Any]] = []
    for date in dates:
        try:
            fits, spot, _points = raw_day(
                root, date, underlying=underlying, fit_band=fit_band, manifest=manifest
            )
        except (ValueError, FileNotFoundError) as exc:
            diag.append(
                {
                    "date": date,
                    "n_slices": 0,
                    "n_points": 0,
                    "t_min": np.nan,
                    "t_max": np.nan,
                    "kept": False,
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        atm, skew = interpolate_pillars(fits, P)
        ok = bool(np.all(np.isfinite(atm)) and np.all(np.isfinite(skew)))
        t_min = min(f.T for f in fits) if fits else np.nan
        t_max = max(f.T for f in fits) if fits else np.nan
        reason = ""
        if not ok:
            missing = [f"{p:g}" for p, x in zip(P, atm) if not np.isfinite(x)]
            reason = f"pillars {missing} outside the slice bracket [{t_min:.3f}, {t_max:.3f}]"
        diag.append(
            {
                "date": date,
                "n_slices": len(fits),
                "n_points": int(sum(f.n for f in fits)),
                "t_min": t_min,
                "t_max": t_max,
                "kept": ok,
                "reason": reason,
            }
        )
        if not ok:
            continue
        for p, a, b in zip(P, atm, skew):
            rows.append(
                {
                    "date": date,
                    "T": p,
                    "vs_vol": a,
                    "atm_vol": a,
                    "skew": b,
                    "ln_spot": float(np.log(spot)),
                }
            )
    frame = pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))
    diagnostics = pd.DataFrame(
        diag, columns=["date", "n_slices", "n_points", "t_min", "t_max", "kept", "reason"]
    )
    if ssvi_history is not None and len(frame):
        ssvi = pd.read_csv(ssvi_history, usecols=["date", "T", "vs_vol"])
        ssvi["date"] = ssvi["date"].astype(str)
        ssvi["_T"] = ssvi["T"].round(9)
        frame["_T"] = frame["T"].round(9)
        merged = frame.merge(
            ssvi[["date", "_T", "vs_vol"]].rename(columns={"vs_vol": "vs_ssvi"}),
            on=["date", "_T"],
            how="left",
        )
        have = merged["vs_ssvi"].notna()
        frame["vs_vol"] = np.where(have, merged["vs_ssvi"], frame["vs_vol"])
        frame = frame.drop(columns="_T")
        frame.attrs["vs_vol_from_ssvi"] = int(have.sum())
        frame.attrs["vs_vol_fallback_atm"] = int((~have).sum())
    else:
        frame.attrs["vs_vol_from_ssvi"] = 0
        frame.attrs["vs_vol_fallback_atm"] = len(frame)
    return frame, diagnostics


# --------------------------------------------------------------------------------------------
# discriminator
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Verdict:
    verdict: str  # "real" | "surface artefact"
    reason: str


def _ssr_table(est_raw: pd.DataFrame, est_ssvi: pd.DataFrame, window: int) -> pd.DataFrame:
    # positional alignment: both frames are indexed by the same pillars (the discriminator
    # restricts to the common ones by rounded T; exact float equality is not relied upon)
    T_raw = est_raw.index.to_numpy(dtype=np.float64)
    T_ssvi = est_ssvi.index.to_numpy(dtype=np.float64)
    if T_raw.shape != T_ssvi.shape or not np.allclose(T_raw, T_ssvi, rtol=0, atol=_TOL_T):
        raise ValueError(f"pillar sets differ: raw {T_raw.tolist()} vs SSVI {T_ssvi.tolist()}")
    r = {c: est_raw[c].to_numpy(dtype=np.float64) for c in est_raw.columns}
    s = {c: est_ssvi[c].to_numpy(dtype=np.float64) for c in est_ssvi.columns}
    se_diff = np.sqrt(r["se"] ** 2 + s["se"] ** 2)
    diff = r["ssr"] - s["ssr"]
    with np.errstate(divide="ignore", invalid="ignore"):
        z = diff / se_diff
    return pd.DataFrame(
        {
            "T": T_raw,
            "window": window,
            "ssr_raw": r["ssr"],
            "se_raw": r["se"],
            "ssr_ssvi": s["ssr"],
            "se_ssvi": s["se"],
            "diff": diff,
            "se_diff": se_diff,
            "z": z,
            "slope_raw": r["slope"],
            "slope_ssvi": s["slope"],
            "mean_skew_raw": r["mean_skew"],
            "mean_skew_ssvi": s["mean_skew"],
            "r2_raw": r["r2"],
            "r2_ssvi": s["r2"],
            "n": r["n"].astype(int),
        }
    )


def verdict_from_table(
    table: pd.DataFrame,
    window_ssr: int = WINDOW_SSR,
    *,
    z_agree: float = Z_AGREE,
    max_pillar: float = MAX_PILLAR_GATE,
) -> Verdict:
    """The gate on a discriminator table (module docstring): both conditions at every pillar
    ``≤ max_pillar`` on the ``window_ssr`` rows.  Reading of the owner's rule: BOTH legs (the
    raw / SSVI agreement within ``z_agree`` combined standard errors and ``SSR_raw + 2 se < 1``)
    are judged on the 60-day window; the 100-day rows are tabulated for information only."""
    rows = table[(table["window"] == window_ssr) & (table["T"] <= max_pillar + _TOL_T)]
    if rows.empty:
        return Verdict(
            "surface artefact", f"no pillar <= {max_pillar:g} on the {window_ssr}-day window"
        )
    fails: list[str] = []
    col = {
        c: rows[c].to_numpy(dtype=np.float64)
        for c in ("T", "ssr_raw", "se_raw", "ssr_ssvi", "se_ssvi", "se_diff")
    }
    for i in range(len(rows)):
        T, raw, se_r = float(col["T"][i]), float(col["ssr_raw"][i]), float(col["se_raw"][i])
        sv, se_s, se_d = (
            float(col["ssr_ssvi"][i]),
            float(col["se_ssvi"][i]),
            float(col["se_diff"][i]),
        )
        if not abs(raw - sv) <= z_agree * se_d:
            z = (raw - sv) / se_d if se_d > 0 else float("inf")
            fails.append(
                f"T={T:g}: raw {raw:.3f} ± {se_r:.3f} vs SSVI {sv:.3f} ± {se_s:.3f} disagree "
                f"(z = {z:+.2f})"
            )
        if not raw + z_agree * se_r < 1.0:
            fails.append(
                f"T={T:g}: raw SSR {raw:.3f} + {z_agree:g} x {se_r:.3f} = "
                f"{raw + z_agree * se_r:.3f} is not below 1"
            )
    if fails:
        return Verdict("surface artefact", "; ".join(fails))
    return Verdict(
        "real",
        f"raw and SSVI SSR agree within {z_agree:g} combined SE and the raw SSR + {z_agree:g} SE "
        f"is below 1 at every pillar <= {max_pillar:g}y on the {window_ssr}-day window",
    )


def discriminator(
    raw_frame: pd.DataFrame,
    ssvi_frame: pd.DataFrame,
    window_ssr: int = WINDOW_SSR,
    window_long: int = WINDOW_LONG,
) -> tuple[pd.DataFrame, Verdict]:
    """Restrict both frames to the common dates and pillars, estimate the SSR on each with
    :func:`~volsto.calibration.history.estimate_history` (windows ``window_long`` /
    ``window_ssr``) and tabulate ``SSR_raw, SSR_ssvi``, their standard errors, the difference and
    its z-score (independent-SE approximation) per pillar and window; the verdict by
    :func:`verdict_from_table`."""
    raw = raw_frame.copy()
    ssvi = ssvi_frame.copy()
    raw["date"] = pd.to_datetime(raw["date"])
    ssvi["date"] = pd.to_datetime(ssvi["date"])
    dates = sorted(set(raw["date"]) & set(ssvi["date"]))
    pill = sorted(set(raw["T"].round(9)) & set(ssvi["T"].round(9)))
    raw = raw[raw["date"].isin(dates) & raw["T"].round(9).isin(pill)]
    ssvi = ssvi[ssvi["date"].isin(dates) & ssvi["T"].round(9).isin(pill)]
    h_raw = SurfaceHistory(raw)
    h_ssvi = SurfaceHistory(ssvi)
    e_raw = estimate_history(h_raw, window_vol=window_long, window_ssr=window_ssr)
    e_ssvi = estimate_history(h_ssvi, window_vol=window_long, window_ssr=window_ssr)
    table = pd.concat(
        [
            _ssr_table(e_raw.ssr_short, e_ssvi.ssr_short, window_ssr),
            _ssr_table(e_raw.ssr_long, e_ssvi.ssr_long, window_long),
        ],
        ignore_index=True,
    )
    table.attrs["n_dates"] = len(dates)
    table.attrs["end_date"] = str(pd.Timestamp(dates[-1]).date())
    return table, verdict_from_table(table, window_ssr)


def write_discriminator_report(
    out_dir: str | Path,
    table: pd.DataFrame,
    verdict: Verdict,
    diagnostics: pd.DataFrame,
    *,
    wall_seconds: float,
    fit_band: float = FIT_BAND,
    notes: tuple[str, ...] = (),
) -> dict[str, Path]:
    """``discriminator.md`` (table, verdict, dates used / dropped, wall clock), ``.csv`` and
    ``discriminator_verdict.json``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kept = diagnostics[diagnostics["kept"]]
    dropped = diagnostics[~diagnostics["kept"]]
    lines = [
        "# M8b raw-slice-skew discriminator for the 2022 H2 SSR",
        "",
        f"Generated {time.strftime('%Y-%m-%d %H:%M:%S')}; wall clock {wall_seconds:.0f} s; "
        "recalibrated: no (no leverage, no SSVI fit: raw quotes only).",
        "",
        f"Raw pillars from the local quadratic on |k| <= {fit_band:g} per retained slice "
        "(unweighted LS on the mid implied vols; ATM vol = a, skew = b per unit log-moneyness), "
        "interpolated to the pillars (skew linear in T, ATM vol linear in total variance).",
        f"Dates used: {len(kept)} ({kept['date'].min() if len(kept) else '-'} .. "
        f"{kept['date'].max() if len(kept) else '-'}), dropped: {len(dropped)}"
        + (
            ": " + ", ".join(f"{r.date} ({r.reason})" for r in dropped.itertuples())
            if len(dropped)
            else "."
        ),
        "",
        "Slices per date (retained, >= 5 points in the band): "
        f"{int(kept['n_slices'].min()) if len(kept) else 0}-"
        f"{int(kept['n_slices'].max()) if len(kept) else 0}, "
        f"points {int(kept['n_points'].min()) if len(kept) else 0}-"
        f"{int(kept['n_points'].max()) if len(kept) else 0}, slice bracket "
        f"[{kept['t_min'].max() if len(kept) else float('nan'):.3f}, "
        f"{kept['t_max'].min() if len(kept) else float('nan'):.3f}] y on the worst date.",
        "",
        "## SSR: raw slices vs SSVI snapshots (same estimator, common dates)",
        "",
        "z = diff / sqrt(se_raw^2 + se_ssvi^2) — independent-SE approximation (the two series "
        "share the spot increments, so this overstates the SE of the difference).",
        "",
        "```",
        table.round(4).to_string(index=False),
        "```",
        "",
        f"## Verdict: **{verdict.verdict}**",
        "",
        verdict.reason,
    ]
    for n in notes:
        lines += ["", n]
    md = out / "discriminator.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    csv = out / "discriminator.csv"
    table.to_csv(csv, index=False)
    js = out / "discriminator_verdict.json"
    js.write_text(
        json.dumps(
            {
                **asdict(verdict),
                "n_dates_used": len(kept),
                "n_dates_dropped": len(dropped),
                "wall_seconds": float(wall_seconds),
                "recalibrated": False,
                "table": json.loads(table.to_json(orient="records")),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {"md": md, "csv": csv, "json": js}


__all__ = [
    "FIT_BAND",
    "MAX_PILLAR_GATE",
    "MIN_SLICE_POINTS",
    "RAW_PILLARS",
    "WINDOW_LONG",
    "WINDOW_SSR",
    "Z_AGREE",
    "SliceFit",
    "Verdict",
    "discriminator",
    "fit_slice_quadratic",
    "interpolate_pillars",
    "raw_day",
    "raw_pillar_frame",
    "slice_fits",
    "verdict_from_table",
    "write_discriminator_report",
]
