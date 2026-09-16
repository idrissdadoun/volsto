"""Page-local helpers shared by the viewer pages (M9 Part 2; owner's ``viewers/pages/``, here
``volsto/viewers/pages/``): cached read-API accessors, the surface / point selectors with grid
snapping, the stored-smile utilities the forward-smile and forward-skew pages both need, and a
plotly heatmap builder.  Everything here reads through :mod:`volsto.viewers.api` only — nothing
calibrates or simulates; a point whose snapped axes are not in the store is reported with
:func:`~volsto.viewers.components.missing_point_notice` (the ``volsto-precompute`` command).

* :func:`grid_table` / :func:`point` / :func:`surface` — ``list_grid`` / ``get_point`` /
  ``get_surface`` behind ``st.cache_data`` / ``st.cache_resource`` keyed on the configured roots
  and :func:`store_stamp` — the newest mtime over the merged ``results/manifest.json``, every
  per-point ``results/points/*/manifest.json`` and the cache root directory, so a point written
  by a still-running shard, a crashed run or an rsync'd cache entry invalidates the cache too.
* :data:`WINDOWS` / :func:`window_label` — the two stored forward-start windows and their
  canonical labels ``1y→1y`` / ``2y→1y`` (``t1 → t1 + (t2 − t1)``), the one naming used on the
  axes, the selectors and the fits tables.
* :func:`select_surface` — a selectbox over :func:`~volsto.viewers.api.list_surfaces`, surfaces
  holding points first.
* :func:`select_point` — model family (``lv / one_factor / two_factor / marking``) then the grid
  axes as :func:`~volsto.viewers.components.snap_slider` positions (ν, ρ, κ for the 1F grid;
  ``ssr_target``, ``skew_eps`` for the marking fits; an LSV family is the default); returns the
  store id or ``None`` after printing the missing-point notice.
* :func:`smile_long` — the stored ``forward_smile`` frames of several points in one long frame
  (``point, label, mode`` added), plus the ids without a stored smile.
* :func:`quadratic_fit` → :class:`SmileFit` — weighted least squares of ``σ(x) = a + b x + c x²``
  in log-moneyness ``x`` at the stored strikes, weights ``1/stderr²``; forward ATM level ``a``,
  skew ``b = ∂σ/∂x`` and curvature ``2c = ∂²σ/∂x²`` with the standard errors of the regression
  covariance ``(XᵀWX)⁻¹`` (:data:`QUADRATIC_METHOD` states the assumption: independent errors
  across strikes — the strikes share one path set, so the quoted skew stderr is conservative for
  positively correlated errors) and the misfit flag ``misfit = chi2_dof > QUADRATIC_MISFIT_CHI2``
  (:data:`QUADRATIC_MISFIT_CHI2`; when the quadratic does not describe the smile within its MC
  noise the regression stderr is noise-only and says nothing about the model error of calling
  the global slope the forward ATM skew).
* :func:`wing_spread` — the M4 put-wing table recomputed from stored smiles: per strike the
  models' vols, ``spread = max − min``, ``spread_stderr`` (hypot of the two extreme models'
  stderr), ``rss_stderr`` and ``spread_z = spread / rss_stderr`` — the definition of
  :func:`volsto.analytics.forward_smile.put_wing_table`.
* :func:`spot_skew_90_110` — ``σ̂_T(0.9 S₀) − σ̂_T(1.1 S₀)`` of a target surface, the spot-moneyness
  convention of :func:`volsto.calibration.fit_2f.spot_skew_90_110` (re-stated here so a page
  never imports the calibration module; asserted equal in the tests), and
  :func:`forward_spot_skew_ratio` — the deck's forward / spot 90/110 skew ratio per point with
  its stderr (``fs_stderr / |spot skew|``, the M7 ``ratio_se`` convention).
* :func:`mc_rows` / :func:`missing_mc_notice` / :func:`nothing_plottable` — **the one Monte
  Carlo guard of the eight pages** (grown on pages 7 and 8, promoted here): the rows a figure
  may draw for a ``(value, stderr)`` pair, the number it must leave out for a NaN in either,
  and the columns the frame does not carry — named in an ``st.info`` with the command that
  writes them instead of a ``KeyError`` mid-render, counted in a caption instead of an
  invisible error bar, and kept as written in the table beside the figure.  The four helpers
  above return the same ``(frame, dropped, absent)`` triple for the derived quantities they
  build out of several stored rows (:func:`finite_rows` is their row filter), so no page draws
  a Monte Carlo number whose own standard error it does not have.
* :func:`heatmap` — a ``go.Heatmap`` from a wide frame with optional ``customdata`` hover.

Checked by ``tests/test_viewers_pages_a.py`` (``test_common_helpers``, ``test_quadratic_fit``,
``test_quadratic_misfit_flag``, ``test_fits_on_synthetic_store``, ``test_spot_skew_convention``,
``test_window_label``, ``test_store_stamp``).

Appended by the page stream of pages 4–6 (``tests/test_viewers_pages_b.py``), on top of the
selectors above:

* :func:`grid_or_none` — :func:`grid_table` or ``None`` on an empty store (the header has
  already printed the precompute command).
* :func:`products_on_surface` — every stored ``products`` row of a surface joined with the
  point's id, label, mode, status and parameters (:func:`point` per point).
* :func:`model_risk_table` — the model-risk page's table: one row per model of the surface for
  a ``(product, quantity)`` with the store's ``value ± stderr`` unchanged and the ``LSV − LV``
  difference whose stderr is :func:`difference_stderr` ``= sqrt(se_model² + se_LV²)`` (two
  separate Monte Carlo runs treated as independent; with the store's common pricing seed the
  estimate is conservative); sortable by grid order, price or label.  Every row stays in the
  table; the triple says how many of them no figure may draw and which column the store does
  not carry.
* :func:`bar_stderr_trace` — a ``go.Bar`` with ``error_y`` = stderr and the ``± stderr`` hover.
* :func:`pm_display` / :func:`fmt_pm` — a display copy of a frame with each ``(col,
  col_stderr)`` pair joined into one ``value ± stderr`` text column (the numeric frame is what
  the Excel export receives).
* :data:`MODE_COLOR` / :data:`AXIS_LABEL` / :data:`PARAM_COLUMNS` — fixed family colours (a
  colour follows the family, never its rank), axis display names, the seven parameters.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from volsto.viewers import api
from volsto.viewers.api import MissingPoint, PointRecord, SurfaceRecord, precompute_command
from volsto.viewers.components import hover_stderr, missing_point_notice, snap_slider
from volsto.viewers.config import ViewerConfig
from volsto.viewers.grid import MODES

#: The two stored forward-start windows ``(t1, t2, label)``.
WINDOWS: tuple[tuple[float, float, str], ...] = ((1.0, 2.0, "1y→1y"), (2.0, 3.0, "2y→1y"))
#: Display names of the store's ``mode`` column.
MODE_LABELS: dict[str, str] = {
    "lv": "LV (ω=0)",
    "one_factor": "1F grid",
    "two_factor": "2F presets",
    "marking": "marking fits",
}
#: Slider axes per mode: display name → store column.
ONE_FACTOR_AXES: dict[str, str] = {"nu": "axis_nu", "rho": "axis_rho", "kappa": "axis_kappa"}
MARKING_AXES: dict[str, str] = {"ssr_target": "ssr_target", "skew_eps": "skew_eps"}
#: Vol → vol points.
VP = 100.0
#: The put wing of the M4 study: strikes at or below this moneyness.
PUT_WING_MAX_STRIKE = 0.9
#: The deck's forward / spot 90/110 skew ratio band (the question the marking fits measure).
DECK_RATIO_BAND: tuple[float, float] = (1.3, 1.5)
#: Misfit threshold on the quadratic fit's χ²/dof.  With the stored smiles (7 strikes 0.8–1.2
#: and an iv stderr of a few hundredths of a vol point at 400k paths) a global quadratic in
#: log-moneyness is a model of the smile, not an interpolant: χ²/dof ≈ 1 means the residuals are
#: within the MC noise; well above it the regression stderr describes the noise only and the
#: skew / curvature carry an unquantified model error.  5 is a deliberately loose flag (a χ² of
#: 5 per degree of freedom is a > 2σ residual per strike), chosen so that a smile flagged
#: ``misfit`` is unambiguously not quadratic — it is a warning marker, not a rejection rule.
QUADRATIC_MISFIT_CHI2 = 5.0
QUADRATIC_METHOD = (
    "Method: weighted least squares of vol(x) = a + b·x + c·x² in log-moneyness x at the stored "
    "strikes (weights 1/stderr²). Forward ATM level = a, skew = b = d vol/dx, curvature = 2c = "
    "d² vol/dx²; standard errors from the regression covariance (XᵀWX)⁻¹, which assumes "
    "independent errors across strikes — the strikes were priced on one path set, so for "
    "positively correlated errors the quoted skew stderr is conservative. chi2/dof ≫ 1 means the "
    "quadratic does not describe the smile within its MC noise; the stderr is then noise-only "
    f"(it omits the model error of calling the global slope the forward ATM skew) — rows with "
    f"chi2/dof > {QUADRATIC_MISFIT_CHI2:g} are flagged misfit in the table and the hover. No new "
    "pricing: stored smiles only."
)


# --------------------------------------------------------------------------------------------
# cached API accessors
# --------------------------------------------------------------------------------------------


def _cfg_key(cfg: ViewerConfig) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(cfg.as_dict().items()))


def _cfg_from_key(key: tuple[tuple[str, str], ...]) -> ViewerConfig:
    return ViewerConfig(**{k: Path(v) for k, v in key})


def _mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def store_stamp(cfg: ViewerConfig) -> float:
    """The cache-invalidation stamp of the store state: the **maximum mtime** over the merged
    ``results/manifest.json``, every per-point ``results/points/*/manifest.json`` and the cache
    root directory (an absent path counts 0.0, so an empty store stamps 0.0).

    The merged manifest alone is not enough: the precompute rewrites it only at the end of a
    shard (``store.write_run``), while ``store.write_point`` writes the per-point manifest as
    each point lands — a running viewer would otherwise not see the points of a shard still in
    progress, of a crashed run, or of a store rsync'd without its merged manifest.  The cache
    root's mtime changes when an entry directory is added (an rsync'd leverage), which flips
    ``has_leverage`` — a value computed inside the cached ``list_grid`` call."""
    results = cfg.store_root / "results"
    stamps = [_mtime(results / "manifest.json"), _mtime(cfg.cache_root)]
    points = results / "points"
    if points.is_dir():
        stamps.extend(_mtime(d / "manifest.json") for d in points.iterdir() if d.is_dir())
    return max(stamps)


@st.cache_data(show_spinner=False)
def _grid_cached(key: tuple[tuple[str, str], ...], stamp: float) -> pd.DataFrame:
    return api.list_grid(_cfg_from_key(key))


@st.cache_data(show_spinner=False)
def _point_cached(key: tuple[tuple[str, str], ...], point_id: str, stamp: float) -> PointRecord:
    return api.get_point(_cfg_from_key(key), point_id)


@st.cache_resource(show_spinner="Building the target surface (SSVI + Dupire, no simulation)…")
def _surface_cached(key: tuple[tuple[str, str], ...], name: str, stamp: float) -> SurfaceRecord:
    return api.get_surface(_cfg_from_key(key), name)


def grid_table(cfg: ViewerConfig) -> pd.DataFrame:
    """:func:`~volsto.viewers.api.list_grid` (cached per store state)."""
    out: pd.DataFrame = _grid_cached(_cfg_key(cfg), store_stamp(cfg))
    return out.copy()


def point(cfg: ViewerConfig, point_id: str) -> PointRecord:
    """:func:`~volsto.viewers.api.get_point` (cached per store state)."""
    rec: PointRecord = _point_cached(_cfg_key(cfg), point_id, store_stamp(cfg))
    return rec


def surface(cfg: ViewerConfig, name: str) -> SurfaceRecord:
    """:func:`~volsto.viewers.api.get_surface` (cached per process; pure construction)."""
    rec: SurfaceRecord = _surface_cached(_cfg_key(cfg), name, store_stamp(cfg))
    return rec


# --------------------------------------------------------------------------------------------
# selectors
# --------------------------------------------------------------------------------------------


def _fmt(v: Any) -> str:
    return f"{float(v):g}" if isinstance(v, (int, float, np.generic)) else str(v)


def seq(values: Any) -> list[float]:
    """A pandas / numpy column as ``list[float]`` (the plotly helpers take sequences)."""
    return [float(v) for v in values]


def labels(values: Any) -> list[str]:
    return [str(v) for v in values]


def select_surface(cfg: ViewerConfig, *, key: str) -> str:
    """Selectbox over the known surfaces (those with stored points first)."""
    surfaces = api.list_surfaces(cfg)
    counts = {str(n): int(c) for n, c in zip(surfaces["name"], surfaces["n_points"])}
    kinds = {str(n): str(k) for n, k in zip(surfaces["name"], surfaces["kind"])}
    names = sorted(counts, key=lambda n: (counts[n] == 0, n))
    if not names:
        raise MissingPoint("any surface", precompute_command(cfg), "no surface is known")
    chosen: str = st.selectbox(
        "Surface",
        names,
        key=key,
        format_func=lambda n: f"{n} ({kinds[n]}, {counts[n]} stored points)",
    )
    return chosen


def points_of(grid: pd.DataFrame, surface_name: str, modes: Sequence[str] | None = None) -> Any:
    """Rows of ``grid`` on ``surface_name`` (optionally restricted to ``modes``), grid order."""
    sub = grid[grid["surface"] == surface_name]
    if modes is not None:
        sub = sub[sub["mode"].isin(list(modes))]
    order = {m: i for i, m in enumerate(MODES)}
    return sub.assign(_o=sub["mode"].map(order)).sort_values(["_o", "label"]).drop(columns="_o")


def label_of(grid: pd.DataFrame, point_id: str) -> str:
    hit = grid[grid["id"] == point_id]
    return str(hit["label"].iloc[0]) if not hit.empty else point_id


def lv_point_of(grid: pd.DataFrame, surface_name: str) -> str | None:
    lv = points_of(grid, surface_name, ("lv",))
    return None if lv.empty else str(lv["id"].iloc[0])


def select_point(
    cfg: ViewerConfig,
    grid: pd.DataFrame,
    surface_name: str,
    *,
    key: str,
    modes: Sequence[str] | None = None,
) -> str | None:
    """Model family + snapped axes → a store id of ``surface_name``; ``None`` (after the
    missing-point notice) when the snapped combination has no stored point."""
    rows = points_of(grid, surface_name, modes)
    present = [m for m in MODES if m in set(rows["mode"])]
    if not present:
        missing_point_notice(
            MissingPoint(
                surface_name, precompute_command(cfg), f"no stored point on {surface_name!r}"
            )
        )
        return None
    default = next((i for i, m in enumerate(present) if m != "lv"), 0)  # an LSV family first
    mode: str = st.selectbox(
        "Model family",
        present,
        index=default,
        key=f"{key}_mode",
        format_func=lambda m: MODE_LABELS.get(m, m),
    )
    sub = rows[rows["mode"] == mode]
    axes = ONE_FACTOR_AXES if mode == "one_factor" else MARKING_AXES if mode == "marking" else {}
    if not axes:
        labels = list(sub["label"])
        if len(labels) == 1:
            st.caption(f"{MODE_LABELS[mode]}: {labels[0]}")
            return str(sub["id"].iloc[0])
        chosen: str = st.selectbox("Point", labels, key=f"{key}_label")
        return str(sub.loc[sub["label"] == chosen, "id"].iloc[0])
    match = sub
    chosen_axes: dict[str, float] = {}
    cols = st.columns(len(axes))
    for col, (name, column) in zip(cols, axes.items()):
        values = [float(v) for v in sub[column].dropna().unique()]
        with col:
            v = snap_slider(name, values, key=f"{key}_{name}", format_func=_fmt)
        chosen_axes[name] = float(v)
        match = match[np.isclose(match[column].astype(float), float(v))]
    if match.empty:
        where = ", ".join(f"{k}={v:g}" for k, v in chosen_axes.items())
        missing_point_notice(
            MissingPoint(
                f"{surface_name} {mode} {where}",
                precompute_command(cfg),
                f"{MODE_LABELS[mode]} point at ({where}) on {surface_name!r}",
            )
        )
        return None
    return str(match["id"].iloc[0])


# --------------------------------------------------------------------------------------------
# stored smiles
# --------------------------------------------------------------------------------------------


def window_label(t1: float, t2: float) -> str:
    """The canonical label of the forward-start window ``t1 → t2``: the :data:`WINDOWS` entry
    matching ``(t1, t2)`` (``np.isclose``), else ``f"{t1:g}y→{t2 - t1:g}y"`` — the same
    ``start → tenor`` naming, so a window not on the grid is labelled, never dropped."""
    for w1, w2, lab in WINDOWS:
        if np.isclose(t1, w1) and np.isclose(t2, w2):
            return lab
    return f"{t1:g}y→{t2 - t1:g}y"


def smile_long(
    cfg: ViewerConfig, grid: pd.DataFrame, ids: Sequence[str]
) -> tuple[pd.DataFrame, list[str]]:
    """The stored ``forward_smile`` frames of ``ids`` stacked (``point, label, mode`` added, in
    the order given) and the ids with no stored smile."""
    frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for pid in ids:
        rec = point(cfg, pid)
        if rec.forward_smile.empty:
            missing.append(pid)
            continue
        frames.append(rec.forward_smile.assign(point=pid, label=rec.label, mode=rec.mode))
    if not frames:
        return pd.DataFrame(), missing
    return pd.concat(frames, ignore_index=True), missing


@dataclass(frozen=True)
class SmileFit:
    """Quadratic fit of one stored smile (module docstring); ``chi2_dof`` is the weighted
    residual χ² per degree of freedom (≈ 1 when the stderrs describe the scatter) and
    :attr:`misfit` flags ``chi2_dof > QUADRATIC_MISFIT_CHI2`` (never with zero degrees of
    freedom, where χ²/dof is NaN)."""

    level: float
    level_stderr: float
    skew: float
    skew_stderr: float
    curvature: float
    curvature_stderr: float
    n: int
    chi2_dof: float

    @property
    def misfit(self) -> bool:
        """``chi2_dof > QUADRATIC_MISFIT_CHI2``: the quadratic misses the smile beyond its MC
        noise (the regression stderr is then noise-only)."""
        return bool(self.chi2_dof > QUADRATIC_MISFIT_CHI2)

    def as_dict(self) -> dict[str, float | bool]:
        return {
            "fwd_atm_level": self.level,
            "fwd_atm_level_stderr": self.level_stderr,
            "fwd_skew": self.skew,
            "fwd_skew_stderr": self.skew_stderr,
            "fwd_curvature": self.curvature,
            "fwd_curvature_stderr": self.curvature_stderr,
            "n_strikes": float(self.n),
            "chi2_dof": self.chi2_dof,
            "misfit": self.misfit,
        }


def quadratic_fit(x: Sequence[float], y: Sequence[float], se: Sequence[float]) -> SmileFit | None:
    """Weighted quadratic regression (module docstring); ``None`` with fewer than three usable
    strikes (finite value, positive stderr)."""
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    ss = np.asarray(se, dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy) & np.isfinite(ss) & (ss > 0)
    xx, yy, ss = xx[ok], yy[ok], ss[ok]
    n = int(xx.size)
    if n < 3 or np.unique(xx).size < 3:
        return None
    design = np.column_stack([np.ones(n), xx, xx * xx])
    w = 1.0 / (ss * ss)
    normal = design.T @ (w[:, None] * design)
    beta = np.linalg.solve(normal, design.T @ (w * yy))
    cov = np.linalg.inv(normal)
    resid = yy - design @ beta
    chi2 = float(np.sum(w * resid * resid))
    dof = n - 3
    return SmileFit(
        float(beta[0]),
        float(math.sqrt(cov[0, 0])),
        float(beta[1]),
        float(math.sqrt(cov[1, 1])),
        float(2.0 * beta[2]),
        float(2.0 * math.sqrt(cov[2, 2])),
        n,
        chi2 / dof if dof > 0 else math.nan,
    )


def fit_smiles(long: pd.DataFrame) -> tuple[pd.DataFrame, int, list[str]]:
    """:func:`quadratic_fit` per ``(point, t1, t2)`` of a :func:`smile_long` frame → ``(fits,
    dropped, absent)``; one row each with ``label, mode, t1, t2, window`` (:func:`window_label`)
    and :meth:`SmileFit.as_dict` (incl. the ``misfit`` flag).

    The guard of :func:`quadratic_fit`, reported to the caller: a strike whose ``iv`` or
    ``iv_stderr`` is missing or not finite carries no weight and is left out of the regression,
    and a ``(point, window)`` left with fewer than three usable strikes produces no row at all —
    ``dropped`` counts those windows so the page can say how many it leaves out, and ``absent``
    names the columns the frame does not carry (``iv`` / ``iv_stderr`` / ``log_moneyness``: no
    fit is possible, the caller names them and the producing command instead of raising a
    ``KeyError`` mid-render)."""
    rows: list[dict[str, Any]] = []
    absent = [c for c in ("log_moneyness", "iv", "iv_stderr") if c not in long.columns]
    if long.empty or absent:
        return pd.DataFrame(), 0, absent
    dropped = 0
    for _, g in long.groupby(["point", "t1", "t2"], sort=False):
        fit = quadratic_fit(seq(g["log_moneyness"]), seq(g["iv"]), seq(g["iv_stderr"]))
        if fit is None:  # fewer than three strikes with a finite vol and a positive stderr
            dropped += 1
            continue
        rows.append(
            {
                "point": str(g["point"].iloc[0]),
                "label": str(g["label"].iloc[0]),
                "mode": str(g["mode"].iloc[0]),
                "t1": float(g["t1"].iloc[0]),
                "t2": float(g["t2"].iloc[0]),
                "window": window_label(float(g["t1"].iloc[0]), float(g["t2"].iloc[0])),
                **fit.as_dict(),
            }
        )
    return pd.DataFrame(rows), dropped, absent


def wing_spread(long: pd.DataFrame, t1: float, t2: float) -> tuple[pd.DataFrame, int, list[str]]:
    """The put-wing table of the M4 study (module docstring) for the window ``t1 → t2`` over the
    points of ``long`` → ``(table, dropped, absent)``; the table is empty with fewer than two
    points.

    ``spread_stderr`` is the hypot of two models' stderr and ``rss_stderr`` a sum over all of
    them, so one missing stderr would make a NaN error bar out of finite vols: a ``(point,
    strike)`` entry enters the pivots only when its ``iv`` and ``iv_stderr`` are both present
    and finite, and a strike left with fewer than two models produces no row.  ``dropped``
    counts the entries left out (the strikes that lost their spread with them), ``absent`` the
    columns the frame does not carry."""
    win = long[np.isclose(long["t1"], t1) & np.isclose(long["t2"], t2)]
    win, dropped, absent = finite_rows(win, "iv", "iv_stderr")
    if absent:
        return pd.DataFrame(), dropped, absent
    if win.empty or win["label"].nunique() < 2:
        return pd.DataFrame(), dropped, absent
    vol = win.pivot_table(index="strike_moneyness", columns="label", values="iv", aggfunc="first")
    se = win.pivot_table(
        index="strike_moneyness", columns="label", values="iv_stderr", aggfunc="first"
    )
    se = se.reindex(columns=vol.columns)
    out = pd.DataFrame({"strike_moneyness": vol.index.to_numpy(dtype=float)})
    for lab in vol.columns:
        out[f"{lab}_vol"] = vol[lab].to_numpy(dtype=float)
        out[f"{lab}_stderr"] = se[lab].to_numpy(dtype=float)
    v = vol.to_numpy(dtype=float)
    e = se.to_numpy(dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        i_max = np.nanargmax(np.where(np.isfinite(v), v, -np.inf), axis=1)
        i_min = np.nanargmin(np.where(np.isfinite(v), v, np.inf), axis=1)
        spread = np.nanmax(v, axis=1) - np.nanmin(v, axis=1)
        rows = np.arange(v.shape[0])
        out["spread"] = spread
        out["spread_stderr"] = np.hypot(e[rows, i_max], e[rows, i_min])
        out["rss_stderr"] = np.sqrt(np.nansum(e * e, axis=1))
        out["spread_z"] = spread / out["rss_stderr"].to_numpy()
    out["put_wing"] = out["strike_moneyness"] <= PUT_WING_MAX_STRIKE + 1e-12
    # a strike priced by one model alone has no spread across models: its max and min are the
    # same number and its "spread" would be a zero with a hypot of one stderr beside it
    n_models = np.isfinite(v).sum(axis=1)
    keep = n_models >= 2
    dropped += int(n_models[~keep].sum())  # in entries, as above: what those strikes still held
    return out[keep].reset_index(drop=True), dropped, absent


def spot_skew_90_110(rec: SurfaceRecord, T: float) -> float:
    """``σ̂_T(0.9 S₀) − σ̂_T(1.1 S₀)`` — spot moneyness, the convention of
    :func:`volsto.calibration.fit_2f.spot_skew_90_110` (an exact SSVI value)."""
    fwd = float(rec.forward_curve.forward(T))
    k = np.log(np.array([0.9, 1.1]) * rec.spot / fwd)
    v = np.asarray(rec.surface.implied_vol_k(k, np.full(2, float(T))), dtype=float)
    return float(v[0] - v[1])


def forward_spot_skew_ratio(
    long: pd.DataFrame, t1: float, t2: float, spot_skew: float
) -> tuple[pd.DataFrame, int, list[str]]:
    """Per point of ``long``: the forward 90/110 skew of the stored ``t1 → t2`` smile
    (``σ(0.9) − σ(1.1)``, stderr = hypot of the two), the spot skew given, and the ratio with
    stderr ``fs_stderr / |spot_skew|`` (the M7 ``ratio_se`` convention: the spot skew is exact)
    → ``(ratios, dropped, absent)``.

    The ratio's stderr is a hypot over the two strikes, NaN as soon as one of them is: a point
    enters the frame only when both strikes carry a finite ``iv`` and a finite ``iv_stderr``.
    ``dropped`` counts the points left out (a strike missing, or a strike without one of the
    two numbers) so the page can say how many, ``absent`` names the columns the frame does not
    carry at all."""
    win = long[np.isclose(long["t1"], t1) & np.isclose(long["t2"], t2)]
    win, _, absent = finite_rows(win, "iv", "iv_stderr")
    if absent:
        return pd.DataFrame(), 0, absent
    rows: list[dict[str, Any]] = []
    dropped = 0
    for pid, g in win.groupby("point", sort=False):
        lo = g[np.isclose(g["strike_moneyness"], 0.9)]
        hi = g[np.isclose(g["strike_moneyness"], 1.1)]
        if lo.empty or hi.empty:  # a strike absent, or dropped for a missing vol / stderr
            dropped += 1
            continue
        fs = float(lo["iv"].iloc[0] - hi["iv"].iloc[0])
        fs_se = float(math.hypot(lo["iv_stderr"].iloc[0], hi["iv_stderr"].iloc[0]))
        rows.append(
            {
                "point": pid,
                "label": str(g["label"].iloc[0]),
                "mode": str(g["mode"].iloc[0]),
                "t1": t1,
                "t2": t2,
                "fwd_skew_90_110": fs,
                "fwd_skew_90_110_stderr": fs_se,
                "spot_skew_90_110": spot_skew,
                "ratio_fwd_to_spot": fs / spot_skew if spot_skew else math.nan,
                "ratio_fwd_to_spot_stderr": fs_se / abs(spot_skew) if spot_skew else math.nan,
            }
        )
    return pd.DataFrame(rows), dropped, absent


# --------------------------------------------------------------------------------------------
# the Monte Carlo guard shared by every page
# --------------------------------------------------------------------------------------------


def mc_rows(df: pd.DataFrame, value: str, stderr: str) -> tuple[pd.DataFrame, int, list[str]]:
    """The rows of ``df`` a figure may draw for the Monte Carlo column ``value``, the number of
    rows it must leave out, and the columns ``df`` does not carry at all.

    A Monte Carlo number is plotted only together with its own standard error, so a row enters
    the figure only when ``value`` and ``stderr`` are both present and not NaN.  A column absent
    from the frame is returned in ``missing`` — the caller names it and the command that writes
    it (:func:`missing_mc_notice`) rather than raising a ``KeyError`` mid-render — and the NaN
    rows are counted so the caller's caption can say how many the figure leaves out; they stay
    as written in the table beside it (no padded zero, no invisible error bar).  Pages 7 and 8
    grew this pair; it is the one implementation of the rule for all eight."""
    missing = [c for c in (value, stderr) if c not in df.columns]
    if missing:
        return df.iloc[:0], 0, missing
    ok = df[value].notna() & df[stderr].notna()
    return df[ok], int((~ok).sum()), missing


def missing_mc_notice(what: str, missing: list[str], source: str, command: str) -> None:
    """One ``st.info`` for a figure dropped because its value or stderr column is not in the
    table at all: the columns, the table they are missing from and the command that rewrites
    it (the table itself is still shown — only the figure goes)."""
    st.info(
        f"{what}: {source} carries no {', '.join(missing)} column — nothing is plotted "
        f"(no MC number is drawn without its stderr); regenerate with {command}."
    )


def nothing_plottable(what: str, command: str) -> None:
    """One ``st.info`` for a figure whose every row was left out by :func:`mc_rows`: an empty
    figure would read as "no data", so the page says which numbers are missing and how to
    produce them instead of drawing axes with nothing in them."""
    st.info(
        f"{what}: no row carries both a value and its own stderr — nothing is plotted "
        f"(no MC number is drawn without its stderr); regenerate with {command}."
    )


def finite_rows(df: pd.DataFrame, *columns: str) -> tuple[pd.DataFrame, int, list[str]]:
    """:func:`mc_rows` over more than two columns: the rows whose every ``columns`` entry is
    present and finite, the number left out, and the columns ``df`` does not carry at all
    (used by the helpers below, which read a value and its twin per *strike* before they build
    a derived quantity out of several rows)."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        return df.iloc[:0], 0, missing
    ok = pd.Series(True, index=df.index)
    for c in columns:
        ok &= np.isfinite(pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float))
    return df[ok], int((~ok).sum()), missing


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------


def heatmap(
    wide: pd.DataFrame,
    *,
    title: str,
    x_title: str,
    y_title: str,
    colorbar: str,
    customdata: np.ndarray[Any, Any] | None = None,
    hovertemplate: str | None = None,
    colorscale: str = "Viridis",
    zmid: float | None = None,
) -> go.Figure:
    """``go.Heatmap`` of a wide frame (index → y, columns → x)."""
    fig = go.Figure(
        data=go.Heatmap(
            z=wide.to_numpy(dtype=float),
            x=[float(c) for c in wide.columns],
            y=[float(i) for i in wide.index],
            colorscale=colorscale,
            zmid=zmid,
            colorbar={"title": colorbar},
            customdata=customdata,
            hovertemplate=hovertemplate
            or (f"{x_title}=%{{x:.3g}}, {y_title}=%{{y:.3g}}: %{{z:.4g}}<extra></extra>"),
        )
    )
    fig.update_layout(
        title=title, xaxis_title=x_title, yaxis_title=y_title, margin={"t": 50, "b": 40}
    )
    return fig


def band(fig: go.Figure, lo: float, hi: float, label: str) -> go.Figure:
    """Shade the horizontal band ``[lo, hi]`` (the deck's ratio band)."""
    fig.add_hrect(
        y0=lo,
        y1=hi,
        fillcolor="LightSalmon",
        opacity=0.25,
        line_width=0,
        annotation_text=label,
        annotation_position="top left",
    )
    return fig


__all__ = [
    "DECK_RATIO_BAND",
    "MARKING_AXES",
    "MODE_LABELS",
    "ONE_FACTOR_AXES",
    "PUT_WING_MAX_STRIKE",
    "QUADRATIC_METHOD",
    "QUADRATIC_MISFIT_CHI2",
    "VP",
    "WINDOWS",
    "SmileFit",
    "band",
    "finite_rows",
    "fit_smiles",
    "forward_spot_skew_ratio",
    "grid_table",
    "heatmap",
    "label_of",
    "labels",
    "lv_point_of",
    "mc_rows",
    "missing_mc_notice",
    "nothing_plottable",
    "point",
    "points_of",
    "quadratic_fit",
    "select_point",
    "select_surface",
    "seq",
    "smile_long",
    "spot_skew_90_110",
    "store_stamp",
    "surface",
    "window_label",
    "wing_spread",
]


# --------------------------------------------------------------------------------------------
# pages 4–6: model families, products across a surface, the model-risk table
# --------------------------------------------------------------------------------------------

#: Fixed colour per model family (plotly's default qualitative palette in a fixed assignment).
MODE_COLOR: dict[str, str] = {
    "lv": "#636EFA",
    "one_factor": "#EF553B",
    "two_factor": "#00CC96",
    "marking": "#AB63FA",
}
#: Grid axes of the store's ``points`` table and their display names.
AXIS_LABEL: dict[str, str] = {
    "axis_nu": "nu (omega = 2 nu)",
    "axis_rho": "rho",
    "axis_kappa": "kappa",
    "ssr_target": "SSR target",
    "skew_eps": "skew eps",
}
#: The seven model parameters shown in a point's short description.
PARAM_COLUMNS: tuple[str, ...] = ("nu", "theta", "k1", "k2", "rho12", "rho_SX1", "rho_SX2")
#: Point columns joined onto the products rows (identity, parameters and the grid axes).
POINT_COLUMNS: tuple[str, ...] = ("id", "label", "mode", "status", *PARAM_COLUMNS, *AXIS_LABEL)
MODEL_RISK_COLUMNS: tuple[str, ...] = (
    "label",
    "family",
    "value",
    "value_stderr",
    "minus_lv",
    "minus_lv_stderr",
    "unit",
    "status",
    "params",
    "key",
    "id",
)


def grid_or_none(cfg: ViewerConfig) -> pd.DataFrame | None:
    """:func:`grid_table`, or ``None`` when the store holds no point."""
    grid = grid_table(cfg)
    return None if grid.empty else grid


def mode_rank(mode: str) -> int:
    return MODES.index(mode) if mode in MODES else len(MODES)


def params_text(row: Any) -> str:
    """``nu=… theta=… k1=… …`` of a point row (``—`` for the LV point)."""
    parts = []
    for c in PARAM_COLUMNS:
        v = row.get(c, math.nan)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            parts.append(f"{c}={float(v):g}")
    return " ".join(parts) if parts else "—"


def products_on_surface(cfg: ViewerConfig, grid: pd.DataFrame, surface_name: str) -> pd.DataFrame:
    """Every stored ``products`` row of the points on ``surface_name`` joined with the point's
    :data:`POINT_COLUMNS`; empty (with the columns) when none."""
    frames = []
    for _, r in points_of(grid, surface_name).iterrows():
        rec = point(cfg, str(r["id"]))
        if rec.products.empty:
            continue
        df = rec.products.copy()
        for c in POINT_COLUMNS:
            df[c] = r[c] if c in r.index else math.nan
        frames.append(df)
    cols = [*POINT_COLUMNS, "product", "quantity", "key", "value", "value_stderr", "unit"]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=cols)


def difference_stderr(se_a: float, se_b: float) -> float:
    """Stderr of ``a − b`` for two independent Monte Carlo estimates: ``sqrt(se_a² + se_b²)``."""
    return math.hypot(float(se_a), float(se_b))


def model_risk_table(
    products: pd.DataFrame, product: str, quantity: str, *, sort_by: str = "grid"
) -> tuple[pd.DataFrame, int, list[str]]:
    """One row per model of the surface pricing ``(product, quantity)`` from
    :func:`products_on_surface` (:data:`MODEL_RISK_COLUMNS`) → ``(table, unplottable, absent)``:
    ``value, value_stderr`` are the store's numbers unchanged; ``minus_lv`` is ``value −
    value_LV`` with :func:`difference_stderr` (NaN when the surface has no LV point holding the
    product, and NaN as well when the LV row's own price or stderr is not finite — a difference
    to an unknown baseline has no error bar).  ``sort_by``: ``grid`` (family order), ``price``
    (ascending) or ``label``.

    Every row stays in the table — it is the page's exported table, NaNs and all — so what the
    caller is told is how many of them a figure may not draw (``unplottable``: a row whose
    ``value`` or ``value_stderr`` is not finite) and which source columns the products frame
    does not carry at all (``absent``; the table then holds the column as NaN rather than
    raising a ``KeyError`` mid-render).  :func:`mc_rows` over the returned table is what keeps
    those rows out of the bars."""
    sel = products[(products["product"] == product) & (products["quantity"] == quantity)]
    absent = [c for c in ("value", "value_stderr") if c not in products.columns]
    values = sel["value"] if "value" in sel.columns else pd.Series(math.nan, index=sel.index)
    errors = (
        sel["value_stderr"]
        if "value_stderr" in sel.columns
        else pd.Series(math.nan, index=sel.index)
    )
    is_lv = sel["mode"] == "lv"
    lv_v = float(values[is_lv].iloc[0]) if int(is_lv.sum()) else math.nan
    lv_se = float(errors[is_lv].iloc[0]) if int(is_lv.sum()) else math.nan
    # the LV baseline needs both its numbers: with a non-finite LV stderr the difference of two
    # Monte Carlo prices has no stderr, and hypot(se, nan) would hide that behind every row
    lv_paired = math.isfinite(lv_v) and math.isfinite(lv_se)
    rows: list[dict[str, Any]] = []
    for (_, r), v, se in zip(sel.iterrows(), values, errors):
        v, se = float(v), float(se)
        rows.append(
            {
                "label": str(r["label"]),
                "family": MODE_LABELS.get(str(r["mode"]), str(r["mode"])),
                "value": v,
                "value_stderr": se,
                "minus_lv": v - lv_v if math.isfinite(lv_v) else math.nan,
                "minus_lv_stderr": difference_stderr(se, lv_se) if lv_paired else math.nan,
                "unit": str(r.get("unit", "")),
                "status": str(r.get("status", "")),
                "params": params_text(r),
                "key": str(r.get("key", "")),
                "id": str(r["id"]),
                "_rank": mode_rank(str(r["mode"])),
            }
        )
    out = pd.DataFrame(rows, columns=[*MODEL_RISK_COLUMNS, "_rank"])
    if sort_by == "price":
        out = out.sort_values(["value", "_rank"], kind="stable")
    elif sort_by == "label":
        out = out.sort_values("label", kind="stable")
    else:
        out = out.sort_values(["_rank", "label"], kind="stable")
    out = out.drop(columns=["_rank"]).reset_index(drop=True)
    finite = np.isfinite(
        np.column_stack(
            [
                pd.to_numeric(out[c], errors="coerce").to_numpy(dtype=float)
                for c in ("value", "value_stderr")
            ]
        )
    ).all(axis=1)
    unplottable = int((~finite).sum())
    return out, unplottable, absent


def bar_stderr_trace(
    x: Iterable[Any],
    y: Iterable[float],
    se: Iterable[float],
    *,
    name: str,
    unit: str = "",
    **kwargs: Any,
) -> go.Bar:
    """A bar trace with ``error_y`` = stderr and the ``± stderr`` hover text (the bar
    counterpart of :func:`~volsto.viewers.components.error_trace`)."""
    ys = [float(v) for v in y]
    ses = [float(s) for s in se]
    return go.Bar(
        x=[str(v) for v in x],
        y=ys,
        error_y={"type": "data", "array": ses, "visible": True},
        name=name,
        text=hover_stderr(ys, ses, unit),
        hovertemplate="%{x}: %{text}<extra>" + name + "</extra>",
        **kwargs,
    )


def fmt_pm(value: float, stderr: float, digits: int = 5) -> str:
    """``value ± stderr`` (``nan`` for a missing value; never a silently padded zero)."""
    if math.isnan(value):
        return "nan"
    if math.isnan(stderr):
        return f"{value:.{digits}g} (no se)"
    return f"{value:.{digits}g} ± {stderr:.2g}"


def pm_display(df: pd.DataFrame, digits: int = 5) -> pd.DataFrame:
    """A display copy of ``df`` with each ``(col, col_stderr)`` pair joined into one text column
    ``col`` = ``value ± stderr``; other columns untouched."""
    out = df.copy()
    cols = [str(c) for c in df.columns]
    for col in cols:
        se = f"{col}_stderr"
        if se in cols:
            out[col] = [
                fmt_pm(
                    float(v) if pd.notna(v) else math.nan,
                    float(s) if pd.notna(s) else math.nan,
                    digits,
                )
                for v, s in zip(df[col], df[se])
            ]
            out = out.drop(columns=[se])
    return out


__all__ += [
    "AXIS_LABEL",
    "MODEL_RISK_COLUMNS",
    "MODE_COLOR",
    "PARAM_COLUMNS",
    "POINT_COLUMNS",
    "bar_stderr_trace",
    "difference_stderr",
    "fmt_pm",
    "grid_or_none",
    "mode_rank",
    "model_risk_table",
    "params_text",
    "pm_display",
    "products_on_surface",
]
