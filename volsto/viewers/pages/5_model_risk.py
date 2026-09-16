# ruff: noqa: N999 — Streamlit multipage file name
"""Page 5 — Model risk (SPEC §9 page 5, M9 Part 2; the page the studies are about): one product
and one quantity (the store's ``products`` rows: the M4 set — forward ATM / VS / vol-swap vols,
cliquets, VKO, KO var, up / down var — and the M6 autocall / Phoenix cells), one surface; every
model calibrated to that surface (LV, the 1F grid points, the 2F presets, the marking fits) side
by side with its price and error bar, the ``LSV − LV`` column with the stderr of the difference
``sqrt(se_model² + se_LV²)`` (two separate Monte Carlo runs treated as independent; with the
store's common pricing seed the estimate is conservative), a table with Excel export and sorting
by price / grid order / label.

A model is a bar only with its own standard error (:func:`~volsto.viewers.pages._common.
mc_rows`, the one guard of the eight pages): a stored price or difference whose value or stderr
is NaN is left out of the figure, counted in a caption and kept as written in the table, and a
store written without ``value_stderr`` names the column with the precompute command instead of
raising mid-render.  ``sqrt(se² + se_LV²)`` is NaN as soon as the LV point's own stderr is, so an
unpaired LV row costs the whole difference figure — said in an ``st.info``, never a row of bars
without error bars.

The numbers are the store's ``value`` / ``value_stderr`` unchanged
(:func:`volsto.viewers.pages._common.model_risk_table`); nothing is calibrated or simulated —
a missing point prints the ``volsto-precompute`` command.  Rendered headless by
``tests/test_viewers_pages_b.py::test_page_renders[5_model_risk.py]``; the table equals the
store rows in ``test_model_risk_table_equals_store`` and the M4 placeholder baselines in the
slow ``test_model_risk_matches_m4_baselines`` (skipped unless the store holds the cached 8·10⁵
placeholder models — never computed by a test).
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from volsto.viewers import api
from volsto.viewers.components import figure_with_download, page_header, table_with_export
from volsto.viewers.config import ViewerConfig
from volsto.viewers.grid import MODES
from volsto.viewers.pages import run_if_streamlit
from volsto.viewers.pages._common import (
    MODE_COLOR,
    MODE_LABELS,
    bar_stderr_trace,
    grid_or_none,
    mc_rows,
    missing_mc_notice,
    model_risk_table,
    nothing_plottable,
    pm_display,
    products_on_surface,
    select_surface,
)

TITLE = "Model risk"
#: The ``(value, stderr)`` column pairs this page reads from the read API, per API frame
#: (``<accessor>.<table>``).  Declared rather than only spelled inline so that
#: ``tests/test_viewers_pages_c.py::test_page_twin_names_match_api`` can walk them against what
#: :mod:`volsto.viewers.api` actually returns for the synthetic store and the repository's real
#: store and outputs: a twin the API renames (its ``reattach_stderr_names`` pairing was rewritten
#: once already) then fails a test instead of a page.
MC_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "point.products": (("value", "value_stderr"),),
}
SORT_OPTIONS: dict[str, str] = {
    "grid": "grid order (LV, 1F, 2F, marking)",
    "price": "price (ascending)",
    "label": "label",
}
#: Default quantity when a product has several (price first, then the fair vol, then the first).
PREFERRED_QUANTITIES: tuple[str, ...] = ("price", "fair vol", "vol")


def price_figure(table: pd.DataFrame, product: str, quantity: str, unit: str) -> go.Figure:
    """Bars with error bars, one colour per model family (fixed assignment); ``table`` holds
    the rows :func:`~volsto.viewers.pages._common.mc_rows` cleared — every bar carries its own
    standard error, the rows without one are counted in the caller's caption and stay in the
    exported table."""
    fig = go.Figure()
    for mode in MODES:
        fam = MODE_LABELS[mode]
        sel = table[table["family"] == fam]
        if sel.empty:
            continue
        fig.add_trace(
            bar_stderr_trace(
                sel["label"],
                sel["value"],
                sel["value_stderr"],
                name=fam,
                unit=unit,
                marker_color=MODE_COLOR[mode],
            )
        )
    fig.update_layout(
        title=f"{product} — {quantity} by model",
        xaxis_title="model",
        yaxis_title=f"{quantity} ({unit})" if unit else quantity,
        xaxis={"categoryorder": "array", "categoryarray": list(table["label"])},
        legend={"orientation": "h"},
    )
    return fig


def difference_figure(table: pd.DataFrame, product: str, quantity: str, unit: str) -> go.Figure:
    """``LSV − LV`` with the stderr of the difference; ``table`` holds the non-LV rows
    :func:`~volsto.viewers.pages._common.mc_rows` cleared for the ``(minus_lv,
    minus_lv_stderr)`` pair — one NaN LV stderr makes the hypot NaN for *every* LSV row, so
    those rows leave the figure together and are counted in the caller's caption."""
    sel = table
    fig = go.Figure()
    for mode in MODES[1:]:
        fam = MODE_LABELS[mode]
        s = sel[sel["family"] == fam]
        if s.empty:
            continue
        fig.add_trace(
            bar_stderr_trace(
                s["label"],
                s["minus_lv"],
                s["minus_lv_stderr"],
                name=fam,
                unit=unit,
                marker_color=MODE_COLOR[mode],
            )
        )
    fig.add_hline(y=0.0, line={"color": "grey", "width": 1})
    fig.update_layout(
        title=f"{product} — {quantity}: LSV minus LV (stderr = sqrt(se² + se_LV²))",
        xaxis_title="model",
        yaxis_title=f"difference to LV ({unit})" if unit else "difference to LV",
        xaxis={"categoryorder": "array", "categoryarray": list(sel["label"])},
        legend={"orientation": "h"},
    )
    return fig


def _default_quantity(quantities: list[str]) -> int:
    for q in PREFERRED_QUANTITIES:
        if q in quantities:
            return quantities.index(q)
    return 0


def render(cfg: ViewerConfig) -> None:
    page_header(
        TITLE,
        cfg,
        subtitle=(
            "One product, every model calibrated to the same surface (LV, 1F grid, 2F presets, "
            "marking fits), price with error bars side by side and the LSV minus LV column — the "
            "store's numbers, never recomputed."
        ),
    )
    grid = grid_or_none(cfg)
    if grid is None:
        return
    surface = select_surface(cfg, key="p5_surface")
    products = products_on_surface(cfg, grid, surface)
    if products.empty:
        st.warning(
            f"No products stored for the points of {surface!r}. The viewer never computes; run "
            "the precompute command printed above."
        )
        return
    names = list(dict.fromkeys(str(p) for p in products["product"]))
    c1, c2, c3 = st.columns(3)
    with c1:
        product = str(st.selectbox("Product", names, key="p5_product"))
    quantities = list(
        dict.fromkeys(str(q) for q in products.loc[products["product"] == product, "quantity"])
    )
    with c2:
        quantity = str(
            st.selectbox(
                "Quantity", quantities, index=_default_quantity(quantities), key="p5_quantity"
            )
        )
    with c3:
        sort_by = str(
            st.selectbox(
                "Sort by",
                list(SORT_OPTIONS),
                key="p5_sort",
                format_func=lambda k: SORT_OPTIONS[k],
            )
        )
    table, unplottable, absent = model_risk_table(products, product, quantity, sort_by=sort_by)
    unit = str(table["unit"].iloc[0]) if not table.empty else ""
    n_lv = int((table["family"] == MODE_LABELS["lv"]).sum())
    st.caption(
        f"{len(table)} models on {surface!r} price {product!r} ({quantity}, {unit or 'unitless'}); "
        f"store key `{table['key'].iloc[0] if not table.empty else ''}`. "
        + (
            "LSV minus LV uses the surface's LV point; its stderr is sqrt(se_model² + se_LV²) — "
            "two separate Monte Carlo runs treated as independent (the store prices every "
            "model with the same seed, so the estimate is conservative)."
            if n_lv
            else "This surface has no LV point in the store: the LSV minus LV column is empty."
        )
    )
    # a stored price is a Monte Carlo number: a model is a bar only with its own stderr, and
    # the rows left out are counted below and kept in the table (never an invisible error bar)
    command = api.precompute_command(cfg)
    title = f"{product} — {quantity} by model"
    shown, n_nan, _ = mc_rows(table, "value", "value_stderr")
    if absent:
        missing_mc_notice(title, absent, f"the store's products rows of {surface!r}", command)
    elif shown.empty:
        nothing_plottable(title, command)
    else:
        figure_with_download(
            price_figure(shown, product, quantity, unit),
            f"model_risk_{surface}_{product}_{quantity}",
        )
    if n_nan or unplottable:
        st.caption(
            f"{max(n_nan, unplottable)} model(s) with a NaN price or stderr are left out of the "
            "bars (no MC number is drawn without its error bar) and kept as written in the table."
        )
    if n_lv:
        diff_title = f"{product} — {quantity}: LSV minus LV"
        lsv = table[table["family"] != MODE_LABELS["lv"]]
        diff, n_diff, _ = mc_rows(lsv, "minus_lv", "minus_lv_stderr")
        if diff.empty:
            st.info(
                f"{diff_title}: no model carries both a difference to LV and its stderr "
                "(sqrt(se² + se_LV²) is NaN as soon as the LV point's own stderr is) — nothing "
                f"is plotted; regenerate with {command}."
            )
        else:
            figure_with_download(
                difference_figure(diff, product, quantity, unit),
                f"model_risk_minus_lv_{surface}_{product}_{quantity}",
            )
        if n_diff:
            st.caption(
                f"{n_diff} model(s) with a NaN LSV minus LV or its stderr are left out of the "
                "difference bars and kept as written in the table."
            )
    st.subheader("Table")
    table_with_export(
        table,
        f"model_risk_{surface}_{product}_{quantity}",
        caption="value ± value_stderr are the store's numbers; minus_lv ± minus_lv_stderr the "
        "difference to the LV point.",
    )
    with st.expander("Compact reading (value ± stderr)"):
        st.dataframe(
            pm_display(
                table[
                    [
                        "label",
                        "family",
                        "value",
                        "value_stderr",
                        "minus_lv",
                        "minus_lv_stderr",
                        "unit",
                    ]
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    with st.expander(f"All quantities of {product!r} on {surface!r}"):
        cols = [
            c
            for c in ("label", "mode", "quantity", "key", "value", "value_stderr", "unit")
            if c in products.columns  # a store written without a stderr column still exports
        ]
        allq = products[products["product"] == product][cols].reset_index(drop=True)
        table_with_export(allq, f"products_{surface}_{product}")


run_if_streamlit(render)
