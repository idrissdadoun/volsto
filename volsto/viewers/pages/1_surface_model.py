# ruff: noqa: N999 — Streamlit multipage file name
"""Page 1 — Surface & model (SPEC §9 page 1, M9 Part 2): the target implied-vol surface of the
placeholder or a snapshot (heatmap and slices at chosen maturities), its Dupire local vol, and
for a chosen calibrated point of that surface the leverage heatmap ``L(t, k)`` — a particle
Monte Carlo estimate whose cache entry stores no per-cell stderr, said on the page with the
entry's particle count and seed; its Monte Carlo check is the calibration error map (vol-point
errors vs ``(T, k)`` with their stderr and z-scores) shown below it.

Every Monte Carlo number on the page carries its own standard error or is not drawn
(:func:`~volsto.viewers.pages._common.mc_rows`, the one guard of the eight pages): a diagnostics
cell whose ``error_vp`` or ``error_vp_stderr`` is NaN is blank in the error map *and* in the z
map — the two figures agree on which cells exist — counted in a caption and kept as written in
the error table, and an entry whose diagnostics carry no per-cell stderr at all names the column
and the precompute command in an ``st.info`` (:func:`~volsto.viewers.pages._common.
missing_mc_notice`) instead of raising mid-render; the report line then states that its max z is
not available rather than inventing one.

Everything comes from the read API: :func:`~volsto.viewers.api.get_surface` (pure SSVI / Dupire
construction — exact values, no standard error, said on the page), :func:`~volsto.viewers.api.
get_leverage` (the cache entry read by file: ``leverage.npz`` + ``diagnostics.json``) and
:func:`~volsto.viewers.api.get_point`.  The point selector lists only the points of the chosen
surface; a point whose leverage is not in the cache prints the ``volsto-precompute`` command
through ``missing_point_notice`` — nothing calibrates.  Every table has the Excel export, every
figure the PNG / SVG download.  Rendered headless by ``tests/test_viewers_pages_a.py``
(``test_page1_renders_with_cache``, ``test_page1_missing_leverage``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from volsto.viewers.api import (
    LeverageRecord,
    MissingArtefact,
    MissingPoint,
    SurfaceRecord,
    get_leverage,
    precompute_command,
)
from volsto.viewers.components import (
    figure_with_download,
    missing_point_notice,
    page_header,
    snap_slider,
    table_with_export,
)
from volsto.viewers.config import ViewerConfig
from volsto.viewers.pages import run_if_streamlit
from volsto.viewers.pages._common import (
    VP,
    grid_table,
    heatmap,
    mc_rows,
    missing_mc_notice,
    nothing_plottable,
    point,
    points_of,
    select_surface,
    surface,
)

TITLE = "Surface & model"
#: The ``(value, stderr)`` column pairs this page reads from the read API, per API frame
#: (``<accessor>.<table>``).  Declared rather than only spelled inline so that
#: ``tests/test_viewers_pages_c.py::test_page_twin_names_match_api`` can walk them against what
#: :mod:`volsto.viewers.api` actually returns for the synthetic store and the repository's real
#: store and outputs: a twin the API renames (its ``reattach_stderr_names`` pairing was rewritten
#: once already) then fails a test instead of a page.
MC_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "leverage.error_map": (("error_vp", "error_vp_stderr"),),
}
#: Slice maturities offered (years); the default selection is the M7 pillar trio.
SLICE_MATURITIES: tuple[float, ...] = (1 / 12, 0.25, 0.5, 1.0, 2.0, 3.0)
DEFAULT_SLICES: tuple[float, ...] = (0.25, 1.0, 3.0)
#: Heatmap grids: maturities 1m … 3y, log-moneyness ±0.5 (the calibration horizon of the grid).
HEATMAP_MATURITIES: list[float] = np.linspace(1 / 12, 3.0, 36).tolist()
LOG_MONEYNESS: list[float] = np.linspace(-0.5, 0.5, 41).tolist()
LOCAL_VOL_TIMES: list[float] = np.linspace(0.02, 3.0, 40).tolist()
#: Half-widths of the leverage heatmap's k range (snap positions).
LEVERAGE_K_ABS: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0)


def _fmt_T(T: float) -> str:
    return f"{T:g}y"


def _target_surface(rec: SurfaceRecord, name: str) -> None:
    st.subheader("Target implied-vol surface")
    st.caption(
        f"{name}: {rec.kind} surface, spot {rec.spot:g}. Exact SSVI values (closed form) — no "
        "standard error applies to this section."
    )
    iv = rec.implied_vol_table(HEATMAP_MATURITIES, LOG_MONEYNESS)
    wide = iv.pivot(index="T", columns="k", values="iv") * VP
    figure_with_download(
        heatmap(
            wide,
            title="Target implied vol (k, T) (vol points)",
            x_title="k = ln K/F",
            y_title="T (years)",
            colorbar="vp",
        ),
        f"{name} implied vol heatmap",
    )
    slices: list[float] = st.multiselect(
        "Slice maturities",
        list(SLICE_MATURITIES),
        default=list(DEFAULT_SLICES),
        format_func=_fmt_T,
        key="p1_slices",
    )
    if slices:
        sl = rec.implied_vol_table(sorted(slices), LOG_MONEYNESS)
        fig = go.Figure()
        for _, g in sl.groupby("T", sort=True):
            T = float(g["T"].iloc[0])
            fig.add_trace(
                go.Scatter(
                    x=g["k"],
                    y=g["iv"] * VP,
                    mode="lines",
                    name=_fmt_T(T),
                    hovertemplate="k=%{x:.3f}: %{y:.3f} vp (exact)<extra>" + _fmt_T(T) + "</extra>",
                )
            )
        fig.update_layout(
            title="Implied vol slices", xaxis_title="k = ln K/F", yaxis_title="implied vol (vp)"
        )
        figure_with_download(fig, f"{name} implied vol slices")
        atm = rec.atm_table(sorted(slices))
        table_with_export(
            atm,
            f"{name} ATM table",
            caption="ATM vol, ATM skew d vol/dk and forward per slice maturity (exact).",
        )
        table_with_export(sl, f"{name} implied vol slices", caption="Slice values (exact).")


def _local_vol(rec: SurfaceRecord, name: str) -> None:
    st.subheader("Dupire local vol")
    lv = rec.local_vol_table(LOCAL_VOL_TIMES, LOG_MONEYNESS)
    wide = lv.pivot(index="t", columns="k", values="local_vol") * VP
    figure_with_download(
        heatmap(
            wide,
            title="Dupire local vol (t, k) (vol points)",
            x_title="k = ln S/F(t)",
            y_title="t (years)",
            colorbar="vp",
        ),
        f"{name} local vol heatmap",
    )
    table_with_export(
        lv, f"{name} local vol table", caption="Dupire local vol on the heatmap grid (exact)."
    )


def _error_map(lev: LeverageRecord, label: str, command: str) -> None:
    st.markdown("**Calibration error map**")
    err = lev.error_map
    if err.empty:
        st.info(
            "This cache entry holds the leverage only (no diagnostics.json): the calibration "
            "error map is not stored for it. The viewer never re-prices; the entry's "
            "CalibrationReport is written by the calibration that produces it."
        )
        return
    # every cell of the two maps is a Monte Carlo vol error: a cell is coloured only where the
    # error and its own stderr are both there, so the hover never reads "± nan (z=nan)" and the
    # error map and the z map agree on which cells exist (the table below keeps every row)
    what = f"Calibration error map of {label}"
    source = f"the diagnostics of the leverage cache entry {lev.cache_key[:12]}…"
    shown, n_nan, absent = mc_rows(err, "error_vp", "error_vp_stderr")
    if absent:
        missing_mc_notice(what, absent, source, command)
    elif shown.empty:
        nothing_plottable(what, command)
    else:
        shown = shown.copy()
        with np.errstate(divide="ignore", invalid="ignore"):
            shown["z"] = shown["error_vp"] / shown["error_vp_stderr"]
        piv_e = shown.pivot_table(index="T", columns="k", values="error_vp", aggfunc="first")
        piv_s = shown.pivot_table(index="T", columns="k", values="error_vp_stderr", aggfunc="first")
        piv_z = shown.pivot_table(index="T", columns="k", values="z", aggfunc="first")
        piv_s = piv_s.reindex(index=piv_e.index, columns=piv_e.columns)
        piv_z = piv_z.reindex(index=piv_e.index, columns=piv_e.columns)
        custom = np.dstack([piv_s.to_numpy(dtype=float), piv_z.to_numpy(dtype=float)])
        figure_with_download(
            heatmap(
                piv_e,
                title="Model - target implied vol (vp) with MC stderr and z",
                x_title="k",
                y_title="T",
                colorbar="vp",
                colorscale="RdBu",
                zmid=0.0,
                customdata=custom,
                hovertemplate=(
                    "T=%{y:.3g}, k=%{x:.3g}: %{z:+.3f} ± %{customdata[0]:.3f} vp "
                    "(z=%{customdata[1]:.1f})<extra></extra>"
                ),
            ),
            f"{label} error map",
        )
        figure_with_download(
            heatmap(
                piv_z,
                title="z = error / stderr",
                x_title="k",
                y_title="T",
                colorbar="z",
                colorscale="RdBu",
                zmid=0.0,
                hovertemplate="T=%{y:.3g}, k=%{x:.3g}: z=%{z:.2f}<extra></extra>",
            ),
            f"{label} z map",
        )
    if n_nan:
        st.caption(
            f"{n_nan} cell(s) with a NaN error or stderr are left out of the error map and the "
            "z map (no MC number is shown without its stderr) and kept as written in the table."
        )
    err = err.copy()
    if not absent:
        with np.errstate(divide="ignore", invalid="ignore"):
            err["z"] = err["error_vp"] / err["error_vp_stderr"]
    if lev.report is not None:
        # both maxima read a column the entry's diagnostics may not carry (max_abs_error needs
        # error_vp, max_z needs stderr_vp as well): each is named, never invented
        have = set(lev.report.vanillas.columns)
        err_text = (
            f"max |error| (T ≤ 2y, |k| ≤ 0.2) {lev.report.max_abs_error():.3f} vp"
            if "error_vp" in have
            else "no max |error| (the entry's diagnostics carry no error_vp column)"
        )
        z_text = (
            f", max z {lev.report.max_z():.2f}"
            if {"error_vp", "stderr_vp"} <= have
            else " (no max z: the entry's diagnostics carry no per-cell stderr)"
        )
        st.caption(
            f"Report: {lev.report.n_paths} paths, seed {lev.report.seed}, {err_text}{z_text}."
        )
    table_with_export(err, f"{label} error table", caption="Per cell: error ± stderr (vp) and z.")


def _leverage(cfg: ViewerConfig, grid: pd.DataFrame, name: str) -> None:
    st.subheader("Leverage function of a stored point")
    cands = points_of(grid, name)
    cands = cands[cands["cache_key"].astype(str) != ""]
    if cands.empty:
        st.info(f"No calibrated point of {name!r} in the store (LV points have no leverage).")
        return
    labels = list(cands["label"])
    cached = dict(zip(cands["label"], cands["has_leverage"]))
    chosen: str = st.selectbox(
        "Point (this surface only)",
        labels,
        key="p1_point",
        format_func=lambda lab: f"{lab} — leverage {'cached' if cached[lab] else 'NOT in cache'}",
    )
    pid = str(cands.loc[cands["label"] == chosen, "id"].iloc[0])
    try:
        lev = get_leverage(cfg, pid)
    except MissingPoint as exc:
        missing_point_notice(exc)
        return
    rec = point(cfg, pid)
    meta = lev.metadata
    st.caption(
        f"cache key {lev.cache_key[:12]}… · {lev.leverage.n_slices} slices to "
        f"{lev.leverage.horizon:g}y · k grid {lev.leverage.k_grid.size} points · "
        f"wall {meta.get('wall_time', 'n/a')} s · code tag {meta.get('code_tag', 'n/a')} · "
        f"commit {meta.get('git_commit', 'n/a')}"
    )
    n_part = meta.get("n_particles", rec.diagnostics.get("n_particles", "not recorded"))
    seed = meta.get("seed", "not recorded")
    st.caption(
        f"L(t, k) = sigma_Dupire / sqrt(E[xi | S]) is a particle-method Monte Carlo estimate at "
        f"{n_part} particles, seed {seed}; the cache stores no per-cell stderr for it "
        "(LeverageFunction holds values only). Its Monte Carlo check is the calibration error "
        "map below: model - target implied vol per (T, k) with its MC stderr and z."
    )
    k_abs = snap_slider("|k| range of the heatmap", LEVERAGE_K_ABS, value=1.0, key="p1_kabs")
    wide = lev.leverage_table(float(k_abs))
    figure_with_download(
        heatmap(
            wide,
            title=f"Leverage L(t, k) — {chosen}",
            x_title="k = ln S/F(t)",
            y_title="t (years)",
            colorbar="L",
            zmid=1.0,
            colorscale="RdBu",
        ),
        f"{chosen} leverage heatmap",
    )
    diag = pd.DataFrame(
        [
            {"quantity": k, "value": f"{v:.6g}" if isinstance(v, float) else str(v)}
            for k, v in {**rec.params, **rec.diagnostics}.items()
        ]
    )
    table_with_export(
        diag,
        f"{chosen} point diagnostics",
        caption=(
            "Model parameters and calibration summary (exact settings and maxima of the error "
            "map below, whose cells carry their own stderr)."
        ),
    )
    lev_table = wide.reset_index()
    lev_table.columns = [c if isinstance(c, str) else f"k={float(c):g}" for c in lev_table.columns]
    table_with_export(
        lev_table,
        f"{chosen} leverage table",
        caption=(
            f"L(t, k) on the cache grid: particle estimate ({n_part} particles, seed {seed}), "
            "no per-cell stderr recorded upstream — see the error map for its MC check."
        ),
    )
    _error_map(lev, chosen, precompute_command(cfg))


def render(cfg: ViewerConfig) -> None:
    page_header(
        TITLE,
        cfg,
        subtitle=(
            "Target surface, Dupire local vol, the leverage heatmap L(t, k) and the calibration "
            "error map of a stored point; snapshot / placeholder switch. Read-only."
        ),
    )
    grid = grid_table(cfg)
    try:
        name = select_surface(cfg, key="p1_surface")
        rec = surface(cfg, name)
    except MissingArtefact as exc:
        missing_point_notice(exc)
        return
    _target_surface(rec, name)
    _local_vol(rec, name)
    if grid.empty:
        return  # the header printed the precompute command
    _leverage(cfg, grid, name)


run_if_streamlit(render)
