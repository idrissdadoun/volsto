# ruff: noqa: N999 — Streamlit multipage file name
"""Page 2 — Forward smile (SPEC §9 page 2, M9 Part 2): the stored forward-start smiles
``1y→1y`` and ``2y→1y`` of several points of one surface overlaid with error bars (the surface's
LV point always included), the forward ATM vol / forward variance-swap / forward vol-swap triple
as a table and bar chart with stderr (error bars; the ``± stderr`` text is hover-only,
``textposition="none"``), and the M4 put-wing invariance: the put wing (strikes
≤ 0.9) overlaid across models with the spread of the models' vols quoted against its stderr
(``spread``, ``spread_stderr``, ``rss_stderr``, ``spread_z`` — the definition of
:func:`volsto.analytics.forward_smile.put_wing_table`).

Every plotted number is a stored Monte Carlo estimate shown with its own standard error or not
shown at all (:func:`~volsto.viewers.pages._common.mc_rows`, the one guard of the eight pages): a
strike whose ``iv`` or ``iv_stderr`` is NaN is not a marker of the smile, a triple cell whose
value or stderr is NaN is not a bar, a strike left with fewer than two models carries no spread,
each is counted in a caption and kept as written in the tables, and a stored table without a
stderr column names it with the precompute command (:func:`~volsto.viewers.pages._common.
missing_mc_notice`) and costs the figure, not the render.

Point selector: model family and snapped grid axes (:func:`~volsto.viewers.pages._common.
select_point`) plus a multi-select of further points of the surface.  Everything is read from
the store's ``forward_smile`` / ``forward_vols`` tables through :func:`~volsto.viewers.api.
get_point`; a point without a stored smile prints the ``volsto-precompute`` command — nothing is
priced.  The two :data:`~volsto.viewers.pages._common.WINDOWS` are selected out of the stored
tables with :func:`numpy.isclose` on ``(t1, t2)`` — the tolerance of
:func:`~volsto.viewers.pages._common.window_label`, :func:`~volsto.viewers.pages._common.
wing_spread` and page 3, so a stored ``t1`` of ``1 + 1e-10`` is the ``1y→1y`` window here too and
is never silently dropped.  Rendered headless by ``tests/test_viewers_pages_a.py``
(``test_page2_renders``, ``test_page2_window_isclose``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from volsto.viewers.api import MissingArtefact, MissingPoint, precompute_command
from volsto.viewers.components import (
    error_trace,
    figure_with_download,
    hover_stderr,
    missing_point_notice,
    page_header,
    table_with_export,
)
from volsto.viewers.config import ViewerConfig
from volsto.viewers.pages import run_if_streamlit
from volsto.viewers.pages._common import (
    PUT_WING_MAX_STRIKE,
    VP,
    WINDOWS,
    grid_table,
    label_of,
    lv_point_of,
    mc_rows,
    missing_mc_notice,
    nothing_plottable,
    point,
    points_of,
    select_point,
    select_surface,
    seq,
    smile_long,
    wing_spread,
)

TITLE = "Forward smile"
#: The ``(value, stderr)`` column pairs this page reads from the read API, per API frame
#: (``<accessor>.<table>``).  Declared rather than only spelled inline so that
#: ``tests/test_viewers_pages_c.py::test_page_twin_names_match_api`` can walk them against what
#: :mod:`volsto.viewers.api` actually returns for the synthetic store and the repository's real
#: store and outputs: a twin the API renames (its ``reattach_stderr_names`` pairing was rewritten
#: once already) then fails a test instead of a page.
MC_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "point.forward_smile": (("iv", "iv_stderr"), ("price", "price_stderr")),
    "point.forward_vols": (
        ("fwd_atm_vol", "fwd_atm_vol_stderr"),
        ("fwd_vs", "fwd_vs_stderr"),
        ("fwd_volswap", "fwd_volswap_stderr"),
    ),
}
#: The triple's store columns → display names.
TRIPLE: tuple[tuple[str, str], ...] = (
    ("fwd_atm_vol", "fwd ATM vol"),
    ("fwd_vs", "fwd VS"),
    ("fwd_volswap", "fwd vol swap"),
)


def _select_points(cfg: ViewerConfig, grid: pd.DataFrame, name: str) -> list[str]:
    """The LV point of the surface, the axis-selected point and the multi-selected overlays,
    de-duplicated in that order."""
    primary = select_point(cfg, grid, name, key="p2")
    rows = points_of(grid, name)
    others: list[str] = st.multiselect(
        "Overlay further points of this surface",
        list(rows["label"]),
        default=[],
        key="p2_overlay",
    )
    ids: list[str] = []
    lv = lv_point_of(grid, name)
    if lv is not None:
        ids.append(lv)
    if primary is not None:
        ids.append(primary)
    ids += [str(rows.loc[rows["label"] == lab, "id"].iloc[0]) for lab in others]
    out: list[str] = []
    for pid in ids:
        if pid not in out:
            out.append(pid)
    if lv is None:
        st.info(f"No LV point of {name!r} in the store: the LV reference is not overlaid.")
    return out


def _smiles(long: pd.DataFrame, command: str) -> None:
    st.subheader("Forward-start smiles")
    st.caption(
        "Implied vol of the forward-start options (strike relative to S(T1)) with MC error bars; "
        "vol points."
    )
    n_nan = 0
    cols = st.columns(len(WINDOWS))
    for col, (t1, t2, wlabel) in zip(cols, WINDOWS):
        win = long[np.isclose(long["t1"], t1) & np.isclose(long["t2"], t2)]
        with col:
            if win.empty:
                st.info(f"No stored {wlabel} smile for the selected points.")
                continue
            what = f"Forward smile {wlabel}"
            # a stored vol is a Monte Carlo number: a strike is a marker only with its own
            # stderr (the store's table is what the notice names, the rows stay in the table)
            shown, dropped, absent = mc_rows(win, "iv", "iv_stderr")
            n_nan += dropped
            if absent:
                missing_mc_notice(what, absent, "the store's forward_smile table", command)
                continue
            if shown.empty:
                nothing_plottable(what, command)
                continue
            fig = go.Figure()
            for _, g in shown.groupby("label", sort=False):
                g = g.sort_values("strike_moneyness")
                fig.add_trace(
                    error_trace(
                        seq(g["strike_moneyness"]),
                        seq(g["iv"] * VP),
                        seq(g["iv_stderr"] * VP),
                        name=str(g["label"].iloc[0]),
                        unit="vp",
                    )
                )
            fig.update_layout(
                title=f"Forward smile {wlabel}",
                xaxis_title="strike / S(T1)",
                yaxis_title="implied vol (vp)",
                legend={"orientation": "h", "y": -0.25},
            )
            figure_with_download(fig, f"forward smile {wlabel}")
    if n_nan:
        st.caption(
            f"{n_nan} stored strike(s) with a NaN vol or stderr are left out of the smiles (no "
            "MC number is drawn without its error bar) and kept as written in the table."
        )
    show = [
        c
        for c in (
            "label",
            "mode",
            "t1",
            "t2",
            "strike_moneyness",
            "log_moneyness",
            "iv",
            "iv_stderr",
            "price",
            "price_stderr",
        )
        if c in long.columns  # a store written without a stderr column still exports its rows
    ]
    table_with_export(
        long[show],
        "forward smiles",
        caption="Stored forward smiles (vol and price with stderr).",
    )


def _triple_cells(win: pd.DataFrame) -> tuple[pd.DataFrame, int, list[str]]:
    """The ``(label, quantity, value, stderr)`` cells of the forward-vol triple a figure may
    draw, the number left out for a NaN in either, and the columns the stored table does not
    carry at all — :func:`~volsto.viewers.pages._common.mc_rows` per quantity of the triple
    (one bar is one stored Monte Carlo number, so the guard is per cell, not per point)."""
    absent = [c for name, _ in TRIPLE for c in (name, f"{name}_stderr") if c not in win.columns]
    if absent:
        return pd.DataFrame(), 0, absent
    parts: list[pd.DataFrame] = []
    dropped = 0
    for name, display in TRIPLE:
        shown, n_nan, _ = mc_rows(win, name, f"{name}_stderr")
        dropped += n_nan
        parts.append(
            pd.DataFrame(
                {
                    "label": shown["label"].astype(str),
                    "quantity": display,
                    "value": shown[name].astype(float),
                    "stderr": shown[f"{name}_stderr"].astype(float),
                }
            )
        )
    cells = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if not cells.empty:  # the three quantities stay in the deck's order within each point
        order = {d: i for i, (_, d) in enumerate(TRIPLE)}
        cells = cells.sort_values(
            ["label", "quantity"], key=lambda c: c.map(order) if c.name == "quantity" else c
        ).reset_index(drop=True)
    return cells, dropped, absent


def _triple(cfg: ViewerConfig, ids: list[str], grid: pd.DataFrame, command: str) -> None:
    st.subheader("Forward ATM vol vs forward VS vs forward vol swap")
    frames = []
    for pid in ids:
        rec = point(cfg, pid)
        if rec.forward_vols.empty:
            missing_point_notice(
                MissingPoint(
                    pid, precompute_command(cfg), f"forward_vols of {label_of(grid, pid)!r}"
                )
            )
            continue
        frames.append(rec.forward_vols.assign(label=rec.label, mode=rec.mode))
    if not frames:
        return
    tv = pd.concat(frames, ignore_index=True)
    n_nan = 0
    cols = st.columns(len(WINDOWS))
    for col, (t1, t2, wlabel) in zip(cols, WINDOWS):
        win = tv[np.isclose(tv["t1"], t1) & np.isclose(tv["t2"], t2)]
        with col:
            if win.empty:
                st.info(f"No stored {wlabel} triple.")
                continue
            what = f"Forward vol triple {wlabel}"
            # the three quantities of one point are three Monte Carlo numbers: each is a bar
            # only with its own stderr, so the triple is walked cell by cell (never a row of
            # three bars of which one has an invisible error bar)
            cells, dropped, absent = _triple_cells(win)
            n_nan += dropped
            if absent:
                missing_mc_notice(what, absent, "the store's forward_vols table", command)
                continue
            if cells.empty:
                nothing_plottable(what, command)
                continue
            fig = go.Figure()
            for _, g in cells.groupby("label", sort=False):
                y = [float(v) * VP for v in g["value"]]
                se = [float(v) * VP for v in g["stderr"]]
                fig.add_trace(
                    go.Bar(
                        x=list(g["quantity"]),
                        y=y,
                        error_y={"type": "data", "array": se, "visible": True},
                        name=str(g["label"].iloc[0]),
                        text=hover_stderr(y, se, "vp"),
                        textposition="none",  # the ± stderr text is hover-only, not printed
                        hovertemplate="%{x}: %{text}<extra>" + str(g["label"].iloc[0]) + "</extra>",
                    )
                )
            fig.update_layout(
                title=f"Forward vol triple {wlabel}",
                yaxis_title="vol (vp)",
                barmode="group",
                legend={"orientation": "h", "y": -0.25},
            )
            figure_with_download(fig, f"forward vol triple {wlabel}")
    if n_nan:
        st.caption(
            f"{n_nan} stored value(s) of the triple with a NaN value or stderr are left out of "
            "the bars (no MC number is drawn without its error bar) and kept as written in the "
            "table."
        )
    order = ["label", "mode", "t1", "t2"] + [
        c for name, _ in TRIPLE for c in (name, f"{name}_stderr") if c in tv.columns
    ]
    table_with_export(tv[order], "forward vol triple", caption="Value ± stderr (vol units).")


def _put_wing(long: pd.DataFrame, command: str) -> None:
    st.subheader("Put-wing invariance")
    st.caption(
        f"The M4 study's observation: across models calibrated to the same surface the forward "
        f"put wing (strike ≤ {PUT_WING_MAX_STRIKE:g}) moves far less than the ATM and call side. "
        "Per strike: spread = max - min of the models' vols, spread_stderr = hypot of the two "
        "extreme models' stderr, rss_stderr = root-sum-square of all stderrs, spread_z = spread / "
        "rss_stderr (put_wing_table's definition)."
    )
    labels = [w for _, _, w in WINDOWS]
    wlabel: str = st.selectbox("Window", labels, key="p2_wing_window")
    t1, t2, _ = next(w for w in WINDOWS if w[2] == wlabel)
    tab, dropped, absent = wing_spread(long, t1, t2)
    what = f"Spread of the forward smile across models, {wlabel}"
    if absent:
        missing_mc_notice(what, absent, "the store's forward_smile table", command)
        return
    if tab.empty:
        st.info(
            "Select at least two points with a stored smile (vol and stderr) for this window."
            + (
                f" {dropped} stored vol(s) of this window carry no finite stderr and are left "
                "out of the spread: no MC number is drawn without its error bar."
                if dropped
                else ""
            )
        )
        return
    wing = tab[tab["put_wing"]]
    rest = tab[~tab["put_wing"]]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=tab["strike_moneyness"],
            y=tab["spread"] * VP,
            error_y={"type": "data", "array": tab["spread_stderr"] * VP, "visible": True},
            marker_color=["indianred" if w else "steelblue" for w in tab["put_wing"]],
            text=hover_stderr(list(tab["spread"] * VP), list(tab["spread_stderr"] * VP), "vp"),
            textposition="none",  # hover-only (plotly's default "auto" prints it on the bars)
            hovertemplate="strike %{x}: spread %{text}, z=%{customdata:.1f}<extra></extra>",
            customdata=tab["spread_z"],
            name="spread across models",
        )
    )
    fig.add_vrect(
        x0=float(tab["strike_moneyness"].min()) - 0.025,
        x1=PUT_WING_MAX_STRIKE + 0.025,
        fillcolor="LightSalmon",
        opacity=0.25,
        line_width=0,
        annotation_text="put wing",
        annotation_position="top left",
    )
    fig.update_layout(
        title=f"Spread of the forward smile across models, {wlabel}",
        xaxis_title="strike / S(T1)",
        yaxis_title="max - min (vp)",
    )
    figure_with_download(fig, f"put wing spread {wlabel}")
    if dropped:
        st.caption(
            f"{dropped} stored vol(s) with a NaN vol or stderr are left out of the spread (a "
            "strike left with one model alone has no spread across models and is not a bar); "
            "the stored smiles are in the table of the section above as written."
        )
    if not wing.empty and not rest.empty:
        st.markdown(
            f"Put wing (k ≤ {PUT_WING_MAX_STRIKE:g}): mean spread "
            f"**{wing['spread'].mean() * VP:.2f} vp** vs rss stderr "
            f"{wing['rss_stderr'].mean() * VP:.2f} vp (mean z {wing['spread_z'].mean():.1f}); "
            f"ATM and call side: mean spread **{rest['spread'].mean() * VP:.2f} vp** "
            f"(mean z {rest['spread_z'].mean():.1f})."
        )
    table_with_export(tab, f"put wing table {wlabel}", caption="Per strike, vol units.")


def render(cfg: ViewerConfig) -> None:
    page_header(
        TITLE,
        cfg,
        subtitle=(
            "Forward-start smiles T1→T2 vs parameters; forward ATM vol vs forward VS vs forward "
            "vol swap; the put-wing invariance. Read-only over the results store."
        ),
    )
    grid = grid_table(cfg)
    if grid.empty:
        return
    try:
        name = select_surface(cfg, key="p2_surface")
    except MissingArtefact as exc:
        missing_point_notice(exc)
        return
    ids = _select_points(cfg, grid, name)
    if not ids:
        return
    long, missing = smile_long(cfg, grid, ids)
    for pid in missing:
        missing_point_notice(
            MissingPoint(pid, precompute_command(cfg), f"forward_smile of {label_of(grid, pid)!r}")
        )
    if long.empty:
        return
    command = precompute_command(cfg)
    _smiles(long, command)
    _triple(cfg, ids, grid, command)
    _put_wing(long, command)


run_if_streamlit(render)
