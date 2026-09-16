# ruff: noqa: N999 — Streamlit multipage file name
"""Page 3 — Forward skew (SPEC §9 page 3, M9 Part 2): forward ATM skew and curvature of the
stored forward smiles (``1y→1y``, ``2y→1y``, the one naming of
:func:`~volsto.viewers.pages._common.window_label` on the axes, the selector and the tables)
from a weighted quadratic fit in log-moneyness per (point, window) —
:func:`~volsto.viewers.pages._common.quadratic_fit`, the method stated on the page, stderr from
the regression covariance, rows with ``chi2_dof`` above
:data:`~volsto.viewers.pages._common.QUADRATIC_MISFIT_CHI2` flagged ``misfit`` in the fits table
and marked in the hover — as a term structure and against a chosen grid axis
(ν / ρ / κ of the 1F grid, ``ssr_target`` / ``skew_eps`` of the marking fits) or any fitted
parameter (θ, k1, k2, ρ's) with the other axes snapped; and the deck's forward / spot 90/110
skew ratio: the forward 90/110 skew of the stored ``1y→1y`` smile over the target surface's spot
1y 90/110 skew (spot-moneyness convention of ``fit_2f.spot_skew_90_110``), with stderr, against
the 1.3–1.5 band drawn as a shaded region — plus the same ratio recorded by the M7 marking runner
(``fwd/spot@1y-into-1y`` columns of ``p1_marking_fits.csv`` through
:func:`~volsto.viewers.api.get_marking`) when present — a ratio column is plotted only with its
``_stderr`` twin (otherwise an ``st.info`` names the column and the producing stage) and rows
whose value or stderr is NaN are left out of the figure and kept as NaN in the table: no
zero-filled error bar anywhere.  The figure's key columns (:data:`M7_RATIO_KEYS`: the ``surface``
/ ``skew_eps`` groups and the ``ssr_target`` x axis) are required rather than optional — a fits
table missing one of them names it in the same ``st.info`` and shows the ratio table alone.

The same rule holds for the store-side figures (:func:`~volsto.viewers.pages._common.mc_rows`,
the one guard of the eight pages): the weighted quadratic ignores a strike without a finite vol
and a positive stderr and skips a window left with fewer than three of them (counted in a
caption), the forward / spot ratio — a hypot over the two strikes — is computed only for a point
whose 0.9 and 1.1 strikes both carry a vol and its stderr, and a store written without
``iv_stderr`` names the column with the precompute command instead of raising mid-render.

Everything is read from the store (``forward_smile`` tables) and the surface config; nothing is
priced or calibrated; missing artefacts print the producing command.  Rendered headless by
``tests/test_viewers_pages_a.py`` (``test_page3_renders``, ``test_page3_m7_ratio_without_stderr``,
``test_page3_m7_ratio_missing_key_column``).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from volsto.viewers.api import MODEL_PARAMS, MissingArtefact, get_marking, precompute_command
from volsto.viewers.components import (
    error_trace,
    figure_with_download,
    missing_point_notice,
    page_header,
    snap_slider,
    table_with_export,
)
from volsto.viewers.config import ViewerConfig
from volsto.viewers.grid import MODES
from volsto.viewers.pages import run_if_streamlit
from volsto.viewers.pages._common import (
    DECK_RATIO_BAND,
    MARKING_AXES,
    MODE_LABELS,
    ONE_FACTOR_AXES,
    QUADRATIC_METHOD,
    QUADRATIC_MISFIT_CHI2,
    VP,
    WINDOWS,
    band,
    fit_smiles,
    forward_spot_skew_ratio,
    grid_table,
    labels,
    lv_point_of,
    mc_rows,
    missing_mc_notice,
    nothing_plottable,
    points_of,
    select_surface,
    seq,
    smile_long,
    spot_skew_90_110,
    surface,
)

TITLE = "Forward skew"
#: The ``(value, stderr)`` column pairs this page reads from the read API, per API frame
#: (``<accessor>.<table>``).  Declared rather than only spelled inline so that
#: ``tests/test_viewers_pages_c.py::test_page_twin_names_match_api`` can walk them against what
#: :mod:`volsto.viewers.api` actually returns for the synthetic store and the repository's real
#: store and outputs: a twin the API renames (its ``reattach_stderr_names`` pairing was rewritten
#: once already) then fails a test instead of a page.
#: (the M7 ``fwd/spot@<T>`` ratios are optional columns of the fits file whose twin the
#: section derives per column, so they are not a fixed pair to walk.)
MC_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "point.forward_smile": (("iv", "iv_stderr"),),
}
#: The ratio's window: forward 1y→1y skew over the spot 1y skew.
RATIO_WINDOW = WINDOWS[0]
FITS_RATIO_COLUMNS: tuple[str, ...] = ("fwd/spot@1y-into-1y", "fwd/spot@2y-into-1y")
#: Columns the M7 ratio figure is keyed on — the two group keys and the x axis.  They are
#: required, not optional: a fits table missing one of them cannot be grouped or ordered, so the
#: section names it in an ``st.info`` and shows the table alone (never a KeyError mid-render).
M7_RATIO_KEYS: tuple[str, ...] = ("surface", "ssr_target", "skew_eps")
#: Where the M7 fits table's ``fwd/spot@…`` ratio and ``fwd/spot_se@…`` stderr columns come from.
M7_RATIO_STAGE = "scripts/m7_p1_marking.py stage 3"


def _fmt(v: float) -> str:
    return f"{v:g}"


def _axis_subset(grid: pd.DataFrame, name: str) -> tuple[pd.DataFrame, str, str]:
    """Mode + axis + snapped other axes → ``(rows, axis name, x column)``; for modes without
    axes every point of the mode, ``x`` = label."""
    rows = points_of(grid, name)
    present = [m for m in MODES if m in set(rows["mode"]) and m != "lv"]
    if not present:
        st.info(f"Only the LV point of {name!r} is stored: no axis to vary.")
        return rows[rows["mode"] == "lv"], "label", "label"
    mode: str = st.selectbox(
        "Model family", present, key="p3_mode", format_func=lambda m: MODE_LABELS.get(m, m)
    )
    sub = rows[rows["mode"] == mode]
    axes = ONE_FACTOR_AXES if mode == "one_factor" else MARKING_AXES if mode == "marking" else {}
    if not axes:
        return sub, "label", "label"
    axis: str = st.radio("Axis", list(axes), horizontal=True, key="p3_axis")
    others = [a for a in axes if a != axis]
    cols = st.columns(max(len(others), 1))
    for col, a in zip(cols, others):
        values = [float(v) for v in sub[axes[a]].dropna().unique()]
        with col:
            v = snap_slider(f"{a} (snapped)", values, key=f"p3_{a}", format_func=_fmt)
        sub = sub[np.isclose(sub[axes[a]].astype(float), float(v))]
    return sub.sort_values(axes[axis]), axis, axes[axis]


def _fit_trace(
    g: pd.DataFrame,
    x: list[str] | list[float],
    quantity: str,
    *,
    name: str,
    mode: str = "lines+markers",
) -> go.Scatter:
    """:func:`error_trace` of a fits frame's ``quantity`` (vp) whose hover carries the χ²/dof and
    the ``misfit`` marker (an ``x`` symbol) for the rows above :data:`QUADRATIC_MISFIT_CHI2`."""
    tr = error_trace(
        x,
        seq(g[quantity] * VP),
        seq(g[f"{quantity}_stderr"] * VP),
        name=name,
        unit="vp",
        mode=mode,
    )
    flags = [bool(m) for m in g["misfit"]]
    # a fit with no degrees of freedom (three strikes) has no chi2/dof: say so, never 'nan'
    tr.text = [
        f"{t} · chi2/dof "
        + (f"{c:.2g}" if math.isfinite(c) else "not defined (dof = 0)")
        + (" MISFIT (stderr noise-only)" if m else "")
        for t, c, m in zip(tr.text, g["chi2_dof"], flags)
    ]
    tr.marker = {"symbol": ["x" if m else "circle" for m in flags], "size": 9}
    return tr


def _term_structure(fits: pd.DataFrame, quantity: str, title: str, command: str) -> None:
    # the fitted level / skew / curvature carry the regression stderr: a fit whose stderr did
    # not come out finite is left out of the figure and kept as written in the fits table
    shown, n_nan, absent = mc_rows(fits, quantity, f"{quantity}_stderr")
    if absent:
        missing_mc_notice(title, absent, "the quadratic fits of the stored smiles", command)
        return
    if shown.empty:
        nothing_plottable(title, command)
        return
    fig = go.Figure()
    for lab, g in shown.groupby("label", sort=False):
        g = g.sort_values("t1")
        fig.add_trace(_fit_trace(g, labels(g["window"]), quantity, name=str(lab)))
    fig.update_layout(title=title, xaxis_title="window", yaxis_title="vp per unit log-moneyness")
    figure_with_download(fig, title)
    if n_nan:
        st.caption(
            f"{n_nan} fit(s) with a NaN {quantity} or stderr are left out of the figure and kept "
            "as written in the fits table."
        )


def _vs_axis(fits: pd.DataFrame, x_col: str, x_name: str, wlabel: str, command: str) -> None:
    win = fits[fits["window"] == wlabel]
    if win.empty:
        st.info(f"No fitted {wlabel} smile on the axis.")
        return
    cols = st.columns(2)
    for col, (q, qname) in zip(
        cols, (("fwd_skew", "forward ATM skew"), ("fwd_curvature", "forward curvature"))
    ):
        with col:
            title = f"{qname} {wlabel} vs {x_name}"
            shown, n_nan, absent = mc_rows(win, q, f"{q}_stderr")
            if absent:
                missing_mc_notice(title, absent, "the quadratic fits of the stored smiles", command)
                continue
            if shown.empty:
                nothing_plottable(title, command)
                continue
            fig = go.Figure()
            for mode, g in shown.groupby("mode", sort=False):
                g = g.sort_values(x_col) if x_col != "label" else g
                fig.add_trace(
                    _fit_trace(
                        g,
                        labels(g[x_col]) if x_col == "label" else seq(g[x_col]),
                        q,
                        name=MODE_LABELS.get(str(mode), str(mode)),
                        mode="markers+lines" if x_col != "label" else "markers",
                    )
                )
            fig.update_layout(title=title, xaxis_title=x_name, yaxis_title="vp")
            figure_with_download(fig, title)
            if n_nan:
                st.caption(
                    f"{n_nan} fit(s) with a NaN {q} or stderr are left out of the figure and "
                    "kept as written in the fits table."
                )


def _skew_section(cfg: ViewerConfig, grid: pd.DataFrame, name: str) -> None:
    st.subheader("Forward ATM skew and curvature")
    st.caption(QUADRATIC_METHOD)
    sub, axis, x_col = _axis_subset(grid, name)
    ids = list(sub["id"])
    lv = lv_point_of(grid, name)
    if lv is not None and lv not in ids:
        ids = [lv, *ids]
    long, missing = smile_long(cfg, grid, ids)
    if missing:
        st.warning(f"{len(missing)} selected point(s) have no stored forward smile (skipped).")
    command = precompute_command(cfg)
    fits, n_unfitted, absent = fit_smiles(long)
    if absent:
        missing_mc_notice(
            "Forward ATM skew and curvature",
            absent,
            "the store's forward_smile table",
            command,
        )
        return
    if fits.empty:
        st.info(
            "No stored smile with at least three strikes carrying a vol and a positive stderr "
            "on this axis (a strike without its own stderr carries no weight in the fit)."
        )
        return
    if n_unfitted:
        st.caption(
            f"{n_unfitted} (point, window) smile(s) with fewer than three strikes carrying both "
            "a vol and a positive stderr are not fitted and are absent from the figures and the "
            "fits table."
        )
    meta_cols = ["id", "label", *MODEL_PARAMS, "axis_nu", "axis_rho", "axis_kappa"]
    meta_cols += ["ssr_target", "skew_eps"]
    fits = fits.merge(
        grid[meta_cols].rename(columns={"id": "point", "label": "_l"}), on="point", how="left"
    ).drop(columns="_l")
    if x_col != "label":
        fits.loc[fits["mode"] == "lv", "axis_nu"] = 0.0  # the LV point is ν = 0 on the ν axis
    _term_structure(fits, "fwd_skew", "Forward ATM skew term structure", command)
    _term_structure(fits, "fwd_curvature", "Forward curvature term structure", command)
    x_options = [x_col] + [p for p in MODEL_PARAMS if p not in (x_col,) and fits[p].notna().any()]
    x_choice: str = st.selectbox(
        "x variable",
        x_options,
        key="p3_x",
        format_func=lambda c: f"grid axis {axis}" if c == x_col else f"parameter {c}",
    )
    wlabel: str = st.selectbox("Window", [w for _, _, w in WINDOWS], key="p3_window")
    x_name = axis if x_choice == x_col else x_choice
    plot = fits if x_choice == "label" else fits[fits[x_choice].notna()]
    _vs_axis(plot, x_choice, x_name, wlabel, command)
    show = [
        "label",
        "mode",
        "window",
        "fwd_atm_level",
        "fwd_atm_level_stderr",
        "fwd_skew",
        "fwd_skew_stderr",
        "fwd_curvature",
        "fwd_curvature_stderr",
        "n_strikes",
        "chi2_dof",
        "misfit",
        *MODEL_PARAMS,
        "axis_nu",
        "axis_rho",
        "axis_kappa",
        "ssr_target",
        "skew_eps",
    ]
    n_misfit = int(fits["misfit"].sum())
    if n_misfit:
        st.warning(
            f"{n_misfit} of {len(fits)} fits have chi2/dof > {QUADRATIC_MISFIT_CHI2:g} (misfit): "
            "the quadratic does not describe those smiles within their MC noise, so their "
            "skew / curvature stderr is noise-only (x markers in the figures)."
        )
    table_with_export(
        fits[show],
        "forward skew fits",
        caption=(
            "Quadratic fits, vol units; value ± stderr. misfit = chi2_dof > "
            f"{QUADRATIC_MISFIT_CHI2:g} (the stderr is then noise-only)."
        ),
    )


def _ratio_section(cfg: ViewerConfig, grid: pd.DataFrame, name: str) -> None:
    st.subheader("Forward / spot 90/110 skew ratio (the deck's 1.3-1.5 question)")
    t1, t2, wlabel = RATIO_WINDOW
    rec = surface(cfg, name)
    spot = spot_skew_90_110(rec, t2 - t1)
    st.caption(
        f"Forward 90/110 skew vol(0.9) - vol(1.1) of the stored {wlabel} smile over the target "
        f"surface's spot {t2 - t1:g}y 90/110 skew ({spot * VP:.3f} vp, exact, spot-moneyness "
        "convention of fit_2f.spot_skew_90_110); ratio stderr = fwd-skew stderr / |spot skew|. "
        f"Shaded: the deck's {DECK_RATIO_BAND[0]}-{DECK_RATIO_BAND[1]} band."
    )
    command = precompute_command(cfg)
    rows = points_of(grid, name)
    long, _ = smile_long(cfg, grid, list(rows["id"]))
    empty: tuple[pd.DataFrame, int, list[str]] = (pd.DataFrame(), 0, [])
    ratios, n_dropped, absent = (
        forward_spot_skew_ratio(long, t1, t2, spot) if not long.empty else empty
    )
    title = f"Forward {wlabel} / spot 90/110 skew ratio per stored point"
    if absent:
        missing_mc_notice(title, absent, "the store's forward_smile table", command)
        return
    if ratios.empty:
        st.info(
            f"No stored {wlabel} smile with the 0.9 and 1.1 strikes, each with its own stderr, "
            f"on {name!r}" + (f" ({n_dropped} point(s) left out)." if n_dropped else ".")
        )
        return
    ratios = ratios.merge(
        grid[["id", "ssr_target", "skew_eps"]].rename(columns={"id": "point"}),
        on="point",
        how="left",
    )
    # the ratio's stderr is a hypot over the two strikes: a point is a marker only when its
    # ratio and that stderr are both finite (the table below keeps every row as computed)
    shown, n_nan, _ = mc_rows(ratios, "ratio_fwd_to_spot", "ratio_fwd_to_spot_stderr")
    if shown.empty:
        nothing_plottable(title, command)
    else:
        fig = go.Figure()
        for mode, g in shown.groupby("mode", sort=False):
            fig.add_trace(
                error_trace(
                    labels(g["label"]),
                    seq(g["ratio_fwd_to_spot"]),
                    seq(g["ratio_fwd_to_spot_stderr"]),
                    name=MODE_LABELS.get(str(mode), str(mode)),
                    mode="markers",
                )
            )
        band(fig, *DECK_RATIO_BAND, "deck 1.3-1.5")
        fig.update_layout(title=title, yaxis_title="ratio", xaxis_title="point")
        figure_with_download(fig, f"fwd spot skew ratio {name}")
    mk = shown[(shown["mode"] == "marking") & shown["ssr_target"].notna()]
    if not mk.empty:
        fig2 = go.Figure()
        for _, g in mk.groupby("skew_eps", sort=True):
            g = g.sort_values("ssr_target")
            fig2.add_trace(
                error_trace(
                    seq(g["ssr_target"]),
                    seq(g["ratio_fwd_to_spot"]),
                    seq(g["ratio_fwd_to_spot_stderr"]),
                    name=f"skew_eps {float(g['skew_eps'].iloc[0]):g}",
                )
            )
        band(fig2, *DECK_RATIO_BAND, "deck 1.3-1.5")
        fig2.update_layout(
            title="Marking fits: ratio vs ssr_target", xaxis_title="ssr_target", yaxis_title="ratio"
        )
        figure_with_download(fig2, f"fwd spot skew ratio marking {name}")
    if n_nan or n_dropped:
        st.caption(
            f"{n_nan} point(s) with a NaN ratio or stderr are left out of the two ratio figures "
            "and kept as written in the table"
            + (
                f"; {n_dropped} further point(s) have no 0.9 / 1.1 strike pair with both a vol "
                "and its stderr and are not in the table either."
                if n_dropped
                else " (no MC number is drawn without its error bar)."
            )
        )
    table_with_export(ratios, "fwd spot skew ratios", caption="Vol units; value ± stderr.")


def _m7_ratio_section(cfg: ViewerConfig) -> None:
    st.subheader("The same ratio recorded by the M7 marking runner")
    try:
        mk = get_marking(cfg)
    except MissingArtefact as exc:
        missing_point_notice(exc)
        return
    fits = mk.fits
    have = [c for c in FITS_RATIO_COLUMNS if c in fits.columns]
    if not have:
        st.info(
            f"{mk.source}/p1_marking_fits.csv carries no fwd/spot@… columns (its ratio columns "
            "are written by scripts/m7_p1_marking.py stage 3)."
        )
        return
    keep = [c for c in (*M7_RATIO_KEYS, "status") if c in fits.columns]
    cols = [c for col in have for c in (col, f"{col}_stderr") if c in fits.columns]
    tab = fits[keep + cols].copy()
    absent = [c for c in M7_RATIO_KEYS if c not in fits.columns]
    if absent:
        st.info(
            f"{mk.source}/p1_marking_fits.csv has no {', '.join(absent)} column: the M7 ratio "
            f"figure is keyed on {', '.join(M7_RATIO_KEYS)} (all written by {M7_RATIO_STAGE}), so "
            "the ratios are shown in the table only."
        )
    paired = [c for c in have if f"{c}_stderr" in tab.columns]
    for col in have:
        if col not in paired:
            st.info(
                f"{col} is in {mk.source}/p1_marking_fits.csv without its stderr column "
                f"fwd/spot_se@{col.split('@')[1]} (written by {M7_RATIO_STAGE}): shown in the "
                "table, not plotted — no MC number is drawn without its stderr."
            )
    plotted: list[str] = [] if absent else paired
    if plotted:
        fig = go.Figure()
        n_nan = n_traces = 0
        for col in plotted:
            # the same shared guard as every other figure of the eight pages
            shown, dropped, _ = mc_rows(tab, col, f"{col}_stderr")  # NaN rows stay in the table
            n_nan += dropped
            for _, g in shown.groupby(["surface", "skew_eps"], sort=True):
                g = g.sort_values("ssr_target")
                surf, eps = str(g["surface"].iloc[0]), float(g["skew_eps"].iloc[0])
                fig.add_trace(
                    error_trace(
                        seq(g["ssr_target"]),
                        seq(g[col]),
                        seq(g[f"{col}_stderr"]),
                        name=f"{surf} eps {eps:g} {col.split('@')[1]}",
                        mode="markers+lines",
                    )
                )
                n_traces += 1
        if n_traces:
            band(fig, *DECK_RATIO_BAND, "deck 1.3-1.5")
            fig.update_layout(
                title="M7 fits: fwd/spot ratio", xaxis_title="ssr_target", yaxis_title="ratio"
            )
            figure_with_download(fig, "m7 fwd spot ratio")
        else:
            nothing_plottable("M7 fits: fwd/spot ratio", M7_RATIO_STAGE)
        if n_nan:
            st.caption(
                f"{n_nan} value(s) with a NaN ratio or stderr (an infeasible or unmarked fit) are "
                "left out of the figure and kept as NaN in the table."
            )
    table_with_export(
        tab,
        "m7 fwd spot ratio",
        caption=f"From {mk.source}; value ± stderr (a column without its stderr is not plotted).",
    )


def render(cfg: ViewerConfig) -> None:
    page_header(
        TITLE,
        cfg,
        subtitle=(
            "Forward ATM skew and curvature term structure vs (nu, rho, theta, k's); the forward "
            "/ spot 90/110 skew ratio of the marking fits against the deck's 1.3-1.5 band. "
            "Read-only."
        ),
    )
    grid = grid_table(cfg)
    if grid.empty:
        return
    try:
        name = select_surface(cfg, key="p3_surface")
    except MissingArtefact as exc:
        missing_point_notice(exc)
        return
    _skew_section(cfg, grid, name)
    try:
        _ratio_section(cfg, grid, name)
    except MissingArtefact as exc:
        missing_point_notice(exc)
    _m7_ratio_section(cfg)


run_if_streamlit(render)
