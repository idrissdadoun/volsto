# ruff: noqa: N999 — Streamlit multipage file name
"""Page 4 — Smile dynamics (SPEC §9 page 4, M9 Part 2): for one stored point, the SSR term
structure — the LSV numerical SSR of the store's ``ssr`` table (with stderr) against the naked
first-order P1 SSR (``ssr_naked_first_order``, exact, marking points) and the ``ssr_target``
input line —, an overlay of the SSR curves of other points on the same surface, the Var(V)
decomposition of the ``varv`` table as a stacked bar per maturity with error bars, and what the
store does **not** hold: the vol-of-vol term structure (the page shows the Var(V) terms it is
made of and says the term structure itself is not precomputed) and the conditional smile after
a spot move (not in the store; nothing is computed here).

Every plotted number carries its own standard error (:func:`~volsto.viewers.pages._common.
mc_rows`, the one guard of the eight pages): an SSR pillar or a Var(V) term whose value or stderr
is NaN is not drawn (counted in a caption, kept as written in the table below), and a stored
table without its stderr column names the column and the precompute command in an ``st.info``
(:func:`~volsto.viewers.pages._common.missing_mc_notice`) and costs the figure, not the render.

Everything comes from :func:`volsto.viewers.api.get_point` — no calibration, no Monte Carlo; a
missing point prints the ``volsto-precompute`` command through ``missing_point_notice``.
Rendered headless by ``tests/test_viewers_pages_b.py::test_page_renders[4_smile_dynamics.py]``.
"""

from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from volsto.viewers import api
from volsto.viewers.components import (
    error_trace,
    figure_with_download,
    page_header,
    table_with_export,
)
from volsto.viewers.config import ViewerConfig
from volsto.viewers.pages import run_if_streamlit
from volsto.viewers.pages._common import (
    MODE_COLOR,
    bar_stderr_trace,
    grid_or_none,
    mc_rows,
    missing_mc_notice,
    nothing_plottable,
    point,
    points_of,
    select_point,
    select_surface,
)

TITLE = "Smile dynamics"
#: The ``(value, stderr)`` column pairs this page reads from the read API, per API frame
#: (``<accessor>.<table>``).  Declared rather than only spelled inline so that
#: ``tests/test_viewers_pages_c.py::test_page_twin_names_match_api`` can walk them against what
#: :mod:`volsto.viewers.api` actually returns for the synthetic store and the repository's real
#: store and outputs: a twin the API renames (its ``reattach_stderr_names`` pairing was rewritten
#: once already) then fails a test instead of a page.
MC_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "point.ssr": (("ssr_lsv", "ssr_lsv_stderr"),),
    "point.varv": (("value", "value_stderr"),),
}
#: Var(V) terms of the stacked bar (they sum to ``var_explained``; ``var_total`` is drawn as a
#: marker with its own error bar so the residual is visible).
STACK_TERMS: tuple[str, ...] = ("var_sv", "var_leverage", "cov_cross")
STACK_COLOR: dict[str, str] = {
    "var_sv": "#636EFA",
    "var_leverage": "#EF553B",
    "cov_cross": "#FFA15A",
}
#: The analytics the precompute does not store today (owner's page 4 list); the page names the
#: option that would produce them instead of computing.  Neither option exists in
#: ``volsto-precompute`` yet — an S1 follow-up, stated as such on the page.
NOT_PRECOMPUTED: dict[str, str] = {
    "conditional smile": "--analytics conditional",
    "vol-of-vol term structure": "--analytics volvol",
}


def ssr_figure(ssr: pd.DataFrame, label: str) -> go.Figure:
    """LSV numerical SSR (error bars) vs the naked first-order SSR (exact) vs ``ssr_target``
    over the pillars a figure may draw — the caller filters with
    :func:`~volsto.viewers.pages._common.mc_rows` (``ssr_lsv``, ``ssr_lsv_stderr``) first, so
    every pillar here carries both its value and its own standard error."""
    fig = go.Figure()
    fig.add_trace(
        error_trace(
            list(ssr["T"]),
            list(ssr["ssr_lsv"]),
            list(ssr["ssr_lsv_stderr"]),
            name="LSV numerical (MC)",
        )
    )
    naked = (
        ssr["ssr_naked_first_order"] if "ssr_naked_first_order" in ssr else pd.Series(dtype=float)
    )
    if naked.notna().any():
        sel = ssr[naked.notna()]
        fig.add_trace(
            go.Scatter(
                x=list(sel["T"]),
                y=list(sel["ssr_naked_first_order"]),
                mode="lines+markers",
                line={"dash": "dash"},
                name="naked first order (exact)",
                hovertemplate="%{x}: %{y:.4g} (first order, exact)<extra></extra>",
            )
        )
    targets: list[float] = (
        [float(t) for t in ssr["ssr_target"].dropna().unique()] if "ssr_target" in ssr else []
    )
    for t in targets:
        fig.add_hline(
            y=float(t), line={"dash": "dot", "color": "grey"}, annotation_text=f"ssr_target {t:g}"
        )
    fig.update_layout(
        title=f"SSR term structure — {label}",
        xaxis_title="T (years)",
        yaxis_title="SSR (skew-stickiness ratio)",
        legend={"orientation": "h"},
    )
    return fig


def varv_figure(varv: pd.DataFrame, label: str) -> tuple[go.Figure | None, int, list[str]]:
    """Stacked bar of the MC Var(V) terms per maturity, ``var_total`` as a marker →
    ``(figure, dropped, absent)``.

    Every term drawn here is a Monte Carlo estimate: a maturity enters a bar (and ``var_total``
    its diamond) only when the store's ``varv`` table carries both the value and its own stderr
    and neither is NaN (:func:`~volsto.viewers.pages._common.mc_rows`).  ``dropped`` counts the
    rows left out so the caller's caption can say how many, ``absent`` names the columns the
    table does not carry at all, and the figure is ``None`` when nothing is left to draw — an
    empty stacked bar would read as a Var(V) of zero."""
    mc = varv[varv["kind"] == "mc"] if "kind" in varv.columns else varv.iloc[:0]
    shown, dropped, absent = mc_rows(mc, "value", "value_stderr")
    if absent:
        return None, 0, absent
    fig = go.Figure()
    n_traces = 0
    for term in STACK_TERMS:
        sel = shown[shown["term"] == term]
        if sel.empty:
            continue
        fig.add_trace(
            bar_stderr_trace(
                [f"{t:g}y" for t in sel["T"]],
                sel["value"],
                sel["value_stderr"],
                name=term,
                marker_color=STACK_COLOR.get(term),
            )
        )
        n_traces += 1
    tot = shown[shown["term"] == "var_total"]
    if not tot.empty:
        fig.add_trace(
            error_trace(
                [f"{t:g}y" for t in tot["T"]],
                list(tot["value"]),
                list(tot["value_stderr"]),
                name="var_total",
                mode="markers",
                marker={"size": 10, "color": "black", "symbol": "diamond"},
            )
        )
        n_traces += 1
    if not n_traces:
        return None, dropped, absent
    fig.update_layout(
        barmode="relative",
        title=f"Var(V) decomposition — {label}",
        xaxis_title="T",
        yaxis_title="variance of the integrated variance",
        legend={"orientation": "h"},
    )
    return fig, dropped, absent


def _ssr_overlay(cfg: ViewerConfig, grid: pd.DataFrame, surface: str, current: str) -> None:
    pts = points_of(grid[grid["has_ssr"]], surface)
    labels = dict(zip(pts["id"], pts["label"]))
    others = [i for i in pts["id"] if i != current]
    if not others:
        st.caption("No other point with an SSR table on this surface.")
        return
    chosen = st.multiselect(
        "Overlay the SSR of other points on this surface",
        others,
        default=others[: min(4, len(others))],
        key="p4_overlay",
        format_func=lambda i: labels[i],
    )
    if not chosen:
        return
    title = f"LSV numerical SSR across points — {surface}"
    command = api.precompute_command(cfg)
    fig = go.Figure()
    rows = []
    n_nan = n_traces = 0
    absent: list[str] = []
    for pid in [current, *chosen]:
        try:
            rec = point(cfg, pid)
        except api.MissingPoint:
            continue
        if rec.ssr.empty:
            continue
        # one overlaid pillar is one stored MC number: drawn only with its own stderr
        shown, dropped, miss = mc_rows(rec.ssr, "ssr_lsv", "ssr_lsv_stderr")
        n_nan += dropped
        absent += [c for c in miss if c not in absent]
        cols = [c for c in ("T", "ssr_lsv", "ssr_lsv_stderr") if c in rec.ssr.columns]
        df = rec.ssr[cols].copy()
        df.insert(0, "point", rec.label)
        rows.append(df)
        if shown.empty:
            continue
        fig.add_trace(
            error_trace(
                list(shown["T"]),
                list(shown["ssr_lsv"]),
                list(shown["ssr_lsv_stderr"]),
                name=rec.label,
                line={"color": MODE_COLOR.get(rec.mode)} if pid == current else None,
            )
        )
        n_traces += 1
    if absent:
        missing_mc_notice(title, absent, "the store's ssr table of a selected point", command)
    elif not n_traces:
        nothing_plottable(title, command)
    else:
        fig.update_layout(title=title, xaxis_title="T", yaxis_title="SSR")
        figure_with_download(fig, f"ssr_overlay_{surface}", key="p4_fig_overlay")
    if n_nan:
        st.caption(
            f"{n_nan} pillar(s) with a NaN SSR or stderr are left out of the overlay (no MC "
            "number is drawn without its error bar) and kept as written in the table."
        )
    if rows:
        table_with_export(pd.concat(rows, ignore_index=True), f"ssr_overlay_{surface}")


def render(cfg: ViewerConfig) -> None:
    page_header(
        TITLE,
        cfg,
        subtitle=(
            "SSR term structure (LSV numerical vs naked first order vs ssr_target), the Var(V) "
            "decomposition, and what the store does not hold (vol-of-vol term structure, "
            "conditional smile) — read from the results store, never computed here."
        ),
    )
    grid = grid_or_none(cfg)
    if grid is None:
        return
    surface = select_surface(cfg, key="p4_surface")
    pid = select_point(cfg, grid, surface, key="p4_point")
    if pid is None:
        return
    rec = point(cfg, pid)
    st.caption(
        f"{rec.label} · mode {rec.mode} · status {rec.status} · "
        f"{rec.diagnostics.get('n_particles', '?')} particles · pricing "
        f"{rec.provenance.get('pricing_n_paths', '?')} paths seed "
        f"{rec.provenance.get('pricing_seed', '?')}"
    )

    # -- SSR term structure -------------------------------------------------------------------
    st.subheader("SSR term structure")
    if rec.ssr.empty:
        st.warning(
            f"No SSR table stored for {rec.label!r}. The viewer never computes; run "
            f"`{api.precompute_command(cfg)}`."
        )
    else:
        eps = rec.ssr["eps"].iloc[0] if "eps" in rec.ssr else math.nan
        st.caption(
            f"LSV numerical SSR: state bump ε = {eps:g} (M7 stage-3 convention), Monte Carlo "
            "with stderr. Naked first order: the fit's P1 first-order SSR at the pillar (exact, "
            "marking points only). ssr_target: the marking-fit input (marking points only)."
        )
        title = f"SSR term structure — {rec.label}"
        # the store's realised SSR is a Monte Carlo number: a pillar is a marker only with its
        # own stderr, the NaN pillars are counted below and stay as written in the table
        shown, n_nan, absent = mc_rows(rec.ssr, "ssr_lsv", "ssr_lsv_stderr")
        if absent:
            missing_mc_notice(
                title,
                absent,
                f"the store's ssr table of {rec.label!r}",
                api.precompute_command(cfg),
            )
        elif shown.empty:
            nothing_plottable(title, api.precompute_command(cfg))
        else:
            figure_with_download(ssr_figure(shown, rec.label), f"ssr_{rec.label}", key="p4_fig_ssr")
        if n_nan:
            st.caption(
                f"{n_nan} pillar(s) with a NaN SSR or stderr are left out of the figure (no MC "
                "number is drawn without its error bar) and kept as NaN in the table."
            )
        table_with_export(rec.ssr, f"ssr_{rec.label}")
        with st.expander("Compare with other points on this surface"):
            _ssr_overlay(cfg, grid, surface, pid)

    # -- vol-of-vol term structure ------------------------------------------------------------
    st.subheader("Vol-of-vol term structure")
    st.info(
        "Not precomputed: the store holds the Var(V) decomposition terms only (below); the "
        "vol-of-vol term structure (`volvol_term_structure` / `atmf_vol_of_vol`) is not produced "
        f"by `volsto-precompute` today (a `{NOT_PRECOMPUTED['vol-of-vol term structure']}` option "
        "does not exist yet — an S1 follow-up). The page shows nothing rather than computing."
    )

    # -- Var(V) decomposition -----------------------------------------------------------------
    st.subheader("Var(V) decomposition")
    if rec.varv.empty:
        if rec.mode == "lv":
            st.caption("The LV point has no Var(V) decomposition (it needs an SV / LSV kernel).")
        else:
            st.warning(
                f"No Var(V) table stored for {rec.label!r}. The viewer never computes; run "
                f"`{api.precompute_command(cfg)}`."
            )
    else:
        st.caption(
            "Monte Carlo terms with stderr (stacked: var_sv + var_leverage + cov_cross = "
            "var_explained; the diamond is var_total). Closed-form terms are exact (stderr 0) "
            "and listed in the table with kind = closed_form."
        )
        title = f"Var(V) decomposition — {rec.label}"
        fig, n_nan, absent = varv_figure(rec.varv, rec.label)
        if absent:
            missing_mc_notice(
                title,
                absent,
                f"the store's varv table of {rec.label!r}",
                api.precompute_command(cfg),
            )
        elif fig is None:
            nothing_plottable(title, api.precompute_command(cfg))
        else:
            figure_with_download(fig, f"varv_{rec.label}", key="p4_fig_varv")
        if n_nan:
            st.caption(
                f"{n_nan} Monte Carlo term(s) with a NaN value or stderr are left out of the "
                "stacked bars and the var_total diamond (no MC number is drawn without its "
                "error bar) and kept as written in the table."
            )
        table_with_export(rec.varv, f"varv_{rec.label}")

    # -- conditional smile --------------------------------------------------------------------
    st.subheader("Conditional smile after a spot move")
    st.info(
        "Not in the store: conditional smiles are not precomputed by `volsto-precompute` today "
        f"(they would be produced by `{api.precompute_command(cfg)} "
        f"{NOT_PRECOMPUTED['conditional smile']}`, an option that does not exist yet — an S1 "
        "follow-up). Nothing is shown rather than computed."
    )


run_if_streamlit(render)
