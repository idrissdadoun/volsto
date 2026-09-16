# ruff: noqa: N999 — Streamlit multipage file name
"""Page 6 — Product grid (SPEC §9 page 6, M9 Part 2): (1) heatmaps of a product's stored price
and of its stderr over any two grid axes of a surface (ν, ρ, κ of the 1F axes; SSR target and
skew ε of the marking fits) with the other axes snapped to grid values — a cell of the face
with no stored point is listed in a missing-point notice (the ``volsto-precompute`` command that
fills it) and is absent from the exported table; (2) the RiskReport summaries of the store's
``risk`` table for a point and product — the five delta regimes and the five gamma regimes
(two store sections, :func:`risk_section_rows`), the fwd-var ladder, skew_T (and curvature_T
when stored), the vega-T waves — as bar charts with error bars, the tier shown and "not
precomputed at this tier" per absent section; the point's risk rows are read once per store
state (:func:`point_risk`, a page-local cached accessor), never once per product name; (3) a
term-sheet editor: one form per product class (autocall / Phoenix / cliquet / VKO put / KO var /
up-var / down-var / barrier KO-KI) mapped onto the product constructors — the page builds the
:class:`~volsto.products.base.Product` and shows its repr and fixing schedule — and prices it
**off the cache**.

"Prices off the cache" is read as: when the term sheet is exactly one of the store's products
(:func:`store_product_name`, the M4 / M6 headline conventions), the stored price ± stderr of
every model on the surface is shown; otherwise the page says that pricing a custom term sheet
needs a precompute run with a products list and prints the YAML snippet for it — **no Monte
Carlo runs in the page** (the ``--products <yaml>`` option is not in ``volsto-precompute`` yet;
the page says so).  The discount curve and spot the constructors need come from
:func:`volsto.viewers.api.get_surface` (pure construction).

Every stored price shown here is a Monte Carlo number quoted with its own standard error or not
quoted at all (:func:`~volsto.viewers.pages._common.mc_rows`, the one guard of the eight pages):
a face cell whose stderr is not finite is blank in the value map *and* in the stderr map
(:func:`paired_cells` — the two figures agree on which cells exist, and no hover reads
``value ± nan``), counted in the table's caption and kept as a row of the exported table; the LV
line beside the face prints its number only when the price and the stderr are both finite; a
risk row without its own stderr is not a bar (counted in a caption); and a table written without
the stderr column names it with the producing command (:func:`~volsto.viewers.pages._common.
missing_mc_notice`) instead of raising mid-render.  Rendered headless by
``tests/test_viewers_pages_b.py::test_page_renders[6_product_grid.py]``; the heatmap pivot, the
missing-cell notice, the regime sections under both risk namings, the read count of the risk
section and the store-product matching are tested there.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from volsto.market.curves import DiscountCurve
from volsto.products.autocall import Autocall, Phoenix
from volsto.products.barrier import KnockInOption, KnockOutOption
from volsto.products.base import Product, daily_schedule
from volsto.products.cliquet import AdditiveCliquet
from volsto.products.conditional_variance import DownVar, KnockOutVarianceSwap, UpVar
from volsto.products.vko import VolKnockOutPut
from volsto.viewers import api
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
    AXIS_LABEL,
    _cfg_from_key,
    _cfg_key,
    bar_stderr_trace,
    grid_or_none,
    mc_rows,
    missing_mc_notice,
    model_risk_table,
    nothing_plottable,
    point,
    products_on_surface,
    select_point,
    select_surface,
    store_stamp,
    surface,
)
from volsto.viewers.store import StoreReader

TITLE = "Product grid"
#: The ``(value, stderr)`` column pairs this page reads from the read API, per API frame
#: (``<accessor>.<table>``).  Declared rather than only spelled inline so that
#: ``tests/test_viewers_pages_c.py::test_page_twin_names_match_api`` can walk them against what
#: :mod:`volsto.viewers.api` actually returns for the synthetic store and the repository's real
#: store and outputs: a twin the API renames (its ``reattach_stderr_names`` pairing was rewritten
#: once already) then fails a test instead of a page.
MC_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "point.products": (("value", "value_stderr"),),
    "point.risk": (("value", "value_stderr"),),
}
FIXINGS_PER_YEAR = 252
#: Risk sections shown per tier (owner's list), keyed on the store's ``section`` column (the
#: RiskReport group: ``report.add('delta', …)`` / ``report.add('gamma', …)`` are two groups); a
#: section absent from the store's rows is reported as not precomputed at the stored tier.
RISK_SECTIONS: tuple[tuple[str, str], ...] = (
    ("delta", "delta regimes"),
    ("gamma", "gamma regimes"),
    ("fwd_var", "fwd-var ladder"),
    ("skew", "skew_T"),
    ("curvature", "curvature_T"),
    ("vega_T", "vega-T waves"),
)
#: The two regime sections; their rows are selected by the ``delta`` / ``gamma`` name prefix
#: across both sections (:func:`risk_section_rows`).
REGIME_SECTIONS: tuple[str, ...] = ("delta", "gamma")
TERM_SHEETS: tuple[str, ...] = (
    "autocall",
    "Phoenix",
    "cliquet",
    "VKO put",
    "KO var",
    "up-var",
    "down-var",
    "barrier KO/KI",
)


# --------------------------------------------------------------------------------------------
# heatmap
# --------------------------------------------------------------------------------------------


def heatmap_pivot(
    products: pd.DataFrame, product: str, quantity: str, x: str, y: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(value, stderr)`` wide frames (rows ``y``, columns ``x``) of the store's rows for
    ``(product, quantity)`` with both axes set; rows sharing a cell must be filtered first."""
    sel = products[
        (products["product"] == product)
        & (products["quantity"] == quantity)
        & products[x].notna()
        & products[y].notna()
    ]
    if sel.duplicated([x, y]).any():
        raise ValueError(f"several points share a ({x}, {y}) cell: snap the other axes first")
    v = sel.pivot(index=y, columns=x, values="value").sort_index().sort_index(axis=1)
    s = sel.pivot(index=y, columns=x, values="value_stderr").reindex_like(v)
    return v, s


def missing_cells(v: pd.DataFrame) -> list[tuple[float, float]]:
    """The ``(x, y)`` grid combinations of a :func:`heatmap_pivot` face with no stored point
    (a shard still running, a ``--limit`` run): the NaN cells, column-major."""
    mask = v.isna()
    return [
        (float(col), float(row)) for col in v.columns for row in v.index if bool(mask.loc[row, col])
    ]


def paired_cells(v: pd.DataFrame, s: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """``v`` with every cell whose stderr is not finite blanked, and how many those are.

    A stored price is a Monte Carlo number: a cell the stderr map cannot colour is a cell the
    value map must not colour either — otherwise the two figures disagree on which cells exist
    and the value map's hover reads ``value ± nan``.  The blanked cells stay in the exported
    table as written (:func:`~volsto.viewers.pages._common.mc_rows`'s rule, cell-wise)."""
    paired = np.isfinite(s.to_numpy(dtype=float))
    unpaired = int((np.isfinite(v.to_numpy(dtype=float)) & ~paired).sum())
    return v.where(pd.DataFrame(paired, index=v.index, columns=v.columns)), unpaired


def heatmap_figure(
    v: pd.DataFrame, s: pd.DataFrame, title: str, x: str, y: str, *, stderr_only: bool = False
) -> go.Figure:
    """Heatmap of ``v`` whose hover shows ``value ± stderr`` (``stderr_only``: the stderr map);
    ``v`` has been through :func:`paired_cells`, so a cell with a value carries a finite stderr
    and the hover never reads ``± nan``."""
    if stderr_only:
        text = np.vectorize(lambda b: f"stderr {b:.3g}" if np.isfinite(b) else "")(
            s.to_numpy(float)
        )
    else:
        text = np.vectorize(
            lambda a, b: f"{a:.5g} ± {b:.2g}" if np.isfinite(a) and np.isfinite(b) else ""
        )(v.to_numpy(float), s.to_numpy(float))
    fig = go.Figure(
        go.Heatmap(
            z=v.to_numpy(float),
            x=[f"{c:g}" for c in v.columns],
            y=[f"{r:g}" for r in v.index],
            text=text,
            hovertemplate=f"{AXIS_LABEL.get(x, x)}=%{{x}}, {AXIS_LABEL.get(y, y)}=%{{y}}: "
            "%{text}<extra></extra>",
            colorscale="Viridis",
            colorbar={"title": title},
        )
    )
    fig.update_layout(
        title=title, xaxis_title=AXIS_LABEL.get(x, x), yaxis_title=AXIS_LABEL.get(y, y)
    )
    return fig


def _heatmaps(cfg: ViewerConfig, products: pd.DataFrame, surface_name: str) -> None:
    st.subheader("Price heatmaps over two grid axes")
    names = list(dict.fromkeys(str(p) for p in products["product"]))
    c1, c2 = st.columns(2)
    with c1:
        product = str(st.selectbox("Product", names, key="p6_hm_product"))
    quantities = list(
        dict.fromkeys(str(q) for q in products.loc[products["product"] == product, "quantity"])
    )
    with c2:
        quantity = str(st.selectbox("Quantity", quantities, key="p6_hm_quantity"))
    sel = products[(products["product"] == product) & (products["quantity"] == quantity)]
    axes = [a for a in AXIS_LABEL if a in sel.columns and sel[a].dropna().nunique() >= 2]
    if len(axes) < 2:
        st.info(
            f"A heatmap needs two grid axes with at least two values each on {surface_name!r}; the "
            f"store has {len(axes)} ({', '.join(AXIS_LABEL[a] for a in axes) or 'none'}) for "
            f"{product!r}. Run the precompute on more grid points to fill the axes."
        )
        return
    c1, c2 = st.columns(2)
    with c1:
        x = str(
            st.selectbox(
                "x axis", axes, index=0, key="p6_hm_x", format_func=lambda a: AXIS_LABEL[a]
            )
        )
    with c2:
        y = str(
            st.selectbox(
                "y axis",
                [a for a in axes if a != x],
                key="p6_hm_y",
                format_func=lambda a: AXIS_LABEL[a],
            )
        )
    for other in (a for a in axes if a not in (x, y)):
        val = snap_slider(
            AXIS_LABEL[other],
            [float(v) for v in sel[other].dropna().unique()],
            key=f"p6_hm_{other}",
        )
        sel = sel[np.isclose(sel[other].astype(float), val) | sel[other].isna()]
    title = f"{product} — {quantity}"
    command = api.precompute_command(cfg)
    absent = [c for c in ("value", "value_stderr") if c not in sel.columns]
    if absent:
        missing_mc_notice(
            f"price heatmaps of {title}",
            absent,
            f"the store's products rows of {surface_name!r}",
            command,
        )
        table_with_export(
            sel[
                [c for c in (x, y, "label", "value", "value_stderr", "unit") if c in sel.columns]
            ].reset_index(drop=True),
            f"heatmap_{surface_name}_{product}_{quantity}",
        )
        return
    try:
        v, s = heatmap_pivot(sel, product, quantity, x, y)
    except ValueError as exc:
        st.warning(str(exc))
        return
    holes = missing_cells(v)
    v, unpaired = paired_cells(v, s)  # the two maps agree on which cells exist
    if holes:
        # a hole in the face: the page names the command that fills it and plots what is stored
        missing_point_notice(
            api.MissingPoint(
                f"{surface_name} {product}",
                api.precompute_command(cfg),
                f"heatmap cells of {product!r} {quantity} on {surface_name!r} at "
                f"({AXIS_LABEL[x]}, {AXIS_LABEL[y]}) = "
                f"{', '.join(f'({a:g}, {b:g})' for a, b in holes)} — plotted blank and absent "
                "from the exported table",
            )
        )
    lv = sel[sel["mode"] == "lv"]
    if not lv.empty:
        lv_v, lv_se = float(lv["value"].iloc[0]), float(lv["value_stderr"].iloc[0])
        # the LV price is a Monte Carlo number too: printed only with its own stderr, never as
        # a "nan ± nan" beside the face
        st.caption(
            f"nu = 0 is the surface's LV point (off the axes): {product} {quantity} = "
            f"{lv_v:.5g} ± {lv_se:.2g}"
            if math.isfinite(lv_v) and math.isfinite(lv_se)
            else (
                f"nu = 0 is the surface's LV point (off the axes): its stored {product} "
                f"{quantity} carries no finite value and stderr, so no number is shown for it "
                f"(regenerate with {command})."
            )
        )
    if not np.isfinite(v.to_numpy(dtype=float)).any():
        nothing_plottable(f"price heatmaps of {title}", command)
    else:
        c1, c2 = st.columns(2)
        with c1:
            figure_with_download(heatmap_figure(v, s, title, x, y), f"heatmap_{product}_{quantity}")
        with c2:
            figure_with_download(
                heatmap_figure(s, s, f"{title} stderr", x, y, stderr_only=True),
                f"heatmap_stderr_{product}_{quantity}",
            )
    notes = []
    if holes:
        notes.append(
            f"{len(holes)} missing cell(s) of the face are not rows of this table (see the "
            "notice above)."
        )
    if unpaired:
        notes.append(
            f"{unpaired} stored cell(s) with a price but no finite stderr are blank in both "
            "maps (no MC number is drawn without its stderr) and kept as written in this table."
        )
    table_with_export(
        sel[[x, y, "label", "value", "value_stderr", "unit"]].reset_index(drop=True),
        f"heatmap_{surface_name}_{product}_{quantity}",
        caption=" ".join(notes) if notes else None,
    )


# --------------------------------------------------------------------------------------------
# risk summaries
# --------------------------------------------------------------------------------------------


def _risk_bars(rows: pd.DataFrame, x: str, title: str, key: str, command: str) -> None:
    """One ``value ± stderr`` bar figure of a point's risk rows: a greek is a bar only when the
    stored row carries both its value and its own stderr (:func:`mc_rows`), an absent column is
    named with the command that writes it instead of a ``KeyError`` mid-render, and the rows
    left out are counted in a caption under it (they stay in the exported risk table)."""
    if rows.empty:  # a section whose rows did not parse (bucket / pillar): say so, no IndexError
        st.info(f"{title}: no parsable rows in the stored risk table")
        return
    shown, n_nan, absent = mc_rows(rows, "value", "value_stderr")
    if absent:
        missing_mc_notice(title, absent, "the store's risk table of this point", command)
        return
    if shown.empty:
        nothing_plottable(title, command)
        return
    fig = go.Figure()
    for name, grp in shown.groupby("name" if x != "name" else "section", sort=False):
        fig.add_trace(
            bar_stderr_trace(
                grp[x].astype(str),
                grp["value"],
                grp["value_stderr"],
                name=str(name),
                unit=str(grp["unit"].iloc[0]),
            )
        )
    fig.update_layout(
        title=title,
        barmode="group",
        xaxis_title=x,
        yaxis_title=str(shown["unit"].iloc[0]),
        legend={"orientation": "h"},
    )
    figure_with_download(fig, key, key=f"p6_fig_{key}")
    if n_nan:
        st.caption(
            f"{n_nan} row(s) with a NaN value or stderr are left out of these bars (no MC "
            "number is drawn without its error bar) and kept as written in the risk table."
        )


def risk_section_rows(risk: pd.DataFrame, section: str) -> pd.DataFrame:
    """The rows of one :data:`RISK_SECTIONS` entry of a point's risk table (``section`` is the
    store's column, the RiskReport group).  The two regime sections are selected by the name
    prefix across both regime sections: the precompute stores ``delta[<regime>]`` under section
    ``delta`` and ``gamma[<regime>]`` under section ``gamma`` (:mod:`volsto.risk.greeks`,
    :func:`volsto.risk.report.risk_report`); a store whose ``gamma`` rows sit under section
    ``delta`` with the name ``gamma [<regime>]`` (the synthetic store of the tests before its
    naming was aligned) renders the same bars."""
    if section in REGIME_SECTIONS:
        return risk[risk["section"].isin(REGIME_SECTIONS) & risk["name"].str.startswith(section)]
    return risk[risk["section"] == section]


@st.cache_data(show_spinner=False)
def _point_risk_cached(
    key: tuple[tuple[str, str], ...], point_id: str, stamp: float
) -> dict[str, pd.DataFrame]:
    """The risk tables of every product stored at ``point_id`` (product → the
    :func:`~volsto.viewers.api.get_risk` frame), cached per store state (:func:`store_stamp`).
    The read API has no ``list_risk_products`` yet, so the product names come from the
    ``product`` values of the point's rows in one :class:`~volsto.viewers.store.StoreReader`
    read (a read of the store, nothing computed); the frames themselves come from ``get_risk``
    so the columns stay the API's."""
    cfg = _cfg_from_key(key)
    rows = StoreReader(cfg.store_root).risk(point_id)
    names = list(dict.fromkeys(str(p) for p in rows.get("product", pd.Series(dtype=str))))
    return {name: api.get_risk(cfg, point_id, name) for name in names}


def point_risk(cfg: ViewerConfig, point_id: str) -> dict[str, pd.DataFrame]:
    """:func:`_point_risk_cached` for ``cfg`` (empty when the point has no risk rows)."""
    out: dict[str, pd.DataFrame] = _point_risk_cached(_cfg_key(cfg), point_id, store_stamp(cfg))
    return out


def _risk(cfg: ViewerConfig, grid: pd.DataFrame, surface_name: str) -> None:
    st.subheader("RiskReport summaries")
    with_risk = grid[(grid["surface"] == surface_name) & grid["has_risk"]]
    if with_risk.empty:
        missing_point_notice(
            api.MissingPoint(
                surface_name,
                api.precompute_command(cfg, risk="light"),
                f"risk rows of any point on {surface_name!r}",
            )
        )
        return
    st.caption("Point (among those with stored risk rows):")
    pid = select_point(cfg, with_risk, surface_name, key="p6_risk_point")
    if pid is None:
        return
    rec = point(cfg, pid)
    found = point_risk(cfg, pid)  # one cached read of the point's risk rows, not one per name
    if not found:
        missing_point_notice(
            api.MissingPoint(
                pid, api.precompute_command(cfg, risk="light"), f"risk rows of point {pid!r}"
            )
        )
        return
    product = str(st.selectbox("Risk product", list(found), key="p6_risk_product"))
    risk = found[product]
    tier = str(risk["tier"].iloc[0])
    st.caption(
        f"{rec.label} · {product} · stored tier **{tier}** · bump size "
        f"{risk['size'].iloc[0]:g} · {len(risk)} rows"
    )
    risk_command = api.precompute_command(cfg, risk="full")
    for section, title in RISK_SECTIONS:
        rows = risk_section_rows(risk, section)
        if rows.empty:
            st.info(
                f"{title}: not precomputed at this tier ({tier}) — "
                f"`{api.precompute_command(cfg, risk='full')}`."
            )
            continue
        if section in REGIME_SECTIONS:
            _risk_bars(
                rows.assign(name=section),
                "regime",
                f"{section} — {rows['regime'].nunique()} regimes ({product})",
                f"{section}_{product}",
                risk_command,
            )
        elif section == "fwd_var":
            _risk_bars(
                rows[rows["bucket"] != ""].assign(name="fwd-var vega"),
                "bucket",
                f"fwd-var ladder ({product}, "
                f"{str(rows['variant'].iloc[0]) or 'variant not recorded'})",
                f"fwd_var_{product}",
                risk_command,
            )
        elif section in ("skew", "curvature"):
            _risk_bars(
                rows[rows["T"].notna()].assign(name=f"{section}_T"),
                "T",
                f"{section}_T ladder ({product})",
                f"{section}_{product}",
                risk_command,
            )
        else:
            part = rows[rows["T"].notna()].copy()
            part["name"] = part["name"].str.replace(r"\[.*\]", "", regex=True)
            _risk_bars(
                part,
                "T",
                f"vega-T waves / projections / tents ({product})",
                f"vega_T_{product}",
                risk_command,
            )
    table_with_export(risk, f"risk_{rec.label}_{product}")


# --------------------------------------------------------------------------------------------
# term-sheet editor
# --------------------------------------------------------------------------------------------


def store_product_name(kind: str, f: dict[str, Any]) -> str | None:
    """The store's product name when the term sheet is exactly a headline product (M4 / M6
    conventions: monthly 2%-capped cliquet with global floor 0 at 1y / 2y; the 12m 100% VKO put
    at 30% on daily fixings; KO var 1y at 110%; up / down var 1y at 100%; the 3y annual autocall
    c = 6%, AC 100%, European KI 60%; the 3y Phoenix c = 6%, CB 70%, memory, American daily KI
    60%), else ``None``."""
    eq = math.isclose
    if (
        kind == "cliquet"
        and f["ppy"] == 12
        and eq(f["cap"], 0.02)
        and f["local_floor"] is None
        and eq(f["global_floor"], 0.0)
        and f["T"] in (1.0, 2.0)
    ):
        return f"cliquet {f['T']:g}y"
    if (
        kind == "VKO put"
        and eq(f["strike"], 1.0)
        and eq(f["T"], 1.0)
        and eq(f["vol_ko"], 0.30)
        and not f["knock_in"]
    ):
        return "VKO 12m 100% put @30%"
    if kind == "KO var" and eq(f["barrier"], 1.1) and eq(f["T"], 1.0) and eq(f["strike_vol"], 0.0):
        return "KO var 1y B=110%"
    if (
        kind in ("up-var", "down-var")
        and eq(f["barrier"], 1.0)
        and eq(f["T"], 1.0)
        and eq(f["strike_vol"], 0.0)
    ):
        return f"{kind} 1y B=100%"
    if (
        kind == "autocall"
        and f["T"] == 3
        and eq(f["coupon"], 0.06)
        and eq(f["ac"], 1.0)
        and eq(f["ki"], 0.6)
    ):
        return "autocall 3y"
    if (
        kind == "Phoenix"
        and f["T"] == 3
        and eq(f["coupon"], 0.06)
        and eq(f["ac"], 1.0)
        and eq(f["ki"], 0.6)
        and eq(f["cb"], 0.7)
        and f["memory"]
    ):
        return "phoenix 3y"
    return None


def build_product(kind: str, f: dict[str, Any], spot: float, discount: DiscountCurve) -> Product:
    """Map the form fields onto the product constructors (levels are fractions of spot)."""
    T = float(f["T"])
    daily = daily_schedule(T, FIXINGS_PER_YEAR)
    if kind == "autocall":
        return Autocall(
            np.arange(1, int(f["T"]) + 1, dtype=float),
            discount,
            spot_reference=spot,
            coupons=f["coupon"],
            ki_level=f["ki"],
            ki_type="european",
            autocall_barriers=f["ac"],
            final_redemption="knock_in",
        )
    if kind == "Phoenix":
        return Phoenix(
            np.arange(1, int(f["T"]) + 1, dtype=float),
            discount,
            spot_reference=spot,
            coupon=f["coupon"],
            coupon_barrier=f["cb"],
            memory=f["memory"],
            ki_level=f["ki"],
            ki_type="american",
            ki_monitoring="discrete",
            ki_fixing_times=daily,
            autocall_barriers=f["ac"],
        )
    if kind == "cliquet":
        return AdditiveCliquet.study(
            T,
            discount,
            periods_per_year=int(f["ppy"]),
            local_cap=f["cap"],
            local_floor=f["local_floor"],
            global_floor=f["global_floor"],
        )
    if kind == "VKO put":
        return VolKnockOutPut(
            f["strike"] * spot,
            T,
            f["vol_ko"],
            daily,
            discount,
            notional=1.0 / spot,
            knock_in=f["knock_in"],
        )
    if kind == "KO var":
        return KnockOutVarianceSwap(daily, f["barrier"] * spot, f["strike_vol"], discount)
    if kind == "up-var":
        return UpVar(daily, f["barrier"] * spot, f["strike_vol"], discount)
    if kind == "down-var":
        return DownVar(daily, f["barrier"] * spot, f["strike_vol"], discount)
    # discrete monitoring needs the knock convention: strict=True (S < B / S > B knocks, the
    # M4c swap convention); continuous monitoring (Brownian bridge) takes none
    strict = True if f["monitoring"] == "discrete" else None
    cls = KnockOutOption if f["knock"] == "out" else KnockInOption
    return cls(
        f["strike"] * spot,
        T,
        f["cp"],
        f["barrier"] * spot,
        f["direction"],
        discount,
        monitoring=f["monitoring"],
        fixing_times=daily if f["monitoring"] == "discrete" else None,
        rebate=f["rebate"],
        strict=strict,
    )


def _fields(kind: str) -> dict[str, Any]:
    """The form of one product class (``st.number_input`` per field, keyed by class)."""

    def n(label: str, v: float | int, **kw: Any) -> float:
        return float(st.number_input(label, value=v, key=f"p6_ts_{kind}_{label}", **kw))

    f: dict[str, Any] = {}
    if kind in ("autocall", "Phoenix"):
        f["T"] = int(n("maturity (years, annual observations)", 3, min_value=1, step=1))
        f["coupon"] = n("coupon (per period)", 0.06, step=0.005, format="%.3f")
        f["ac"] = n("autocall barrier (x spot)", 1.0, step=0.05)
        f["ki"] = n("knock-in level (x spot)", 0.6, step=0.05)
        if kind == "Phoenix":
            f["cb"] = n("coupon barrier (x spot)", 0.7, step=0.05)
            f["memory"] = bool(st.checkbox("memory", value=True, key="p6_ts_memory"))
    elif kind == "cliquet":
        f["T"] = n("maturity (years)", 1.0, step=0.5)
        f["ppy"] = int(n("periods per year", 12, min_value=1, step=1))
        f["cap"] = n("local cap", 0.02, step=0.005, format="%.3f")
        lf = n("local floor (-1 = none)", -1.0, step=0.01)
        f["local_floor"] = None if lf <= -1.0 else lf
        f["global_floor"] = n("global floor", 0.0, step=0.01)
    elif kind == "VKO put":
        f["strike"], f["T"] = n("strike (x spot)", 1.0, step=0.05), n(
            "maturity (years)", 1.0, step=0.5
        )
        f["vol_ko"] = n("vol knock-out level", 0.30, step=0.01)
        f["knock_in"] = bool(st.checkbox("knock-in variant", value=False, key="p6_ts_vko_ki"))
    elif kind in ("KO var", "up-var", "down-var"):
        f["T"] = n("maturity (years)", 1.0, step=0.5)
        f["barrier"] = n("barrier (x spot)", 1.1 if kind == "KO var" else 1.0, step=0.05)
        f["strike_vol"] = n("strike vol", 0.0, step=0.01)
    else:
        f["knock"] = str(st.selectbox("knock", ["out", "in"], key="p6_ts_knock"))
        f["cp"] = str(st.selectbox("call / put", ["call", "put"], key="p6_ts_cp"))
        f["strike"], f["T"] = n("strike (x spot)", 1.0, step=0.05), n(
            "maturity (years)", 1.0, step=0.5
        )
        f["barrier"] = n("barrier (x spot)", 1.2, step=0.05)
        f["direction"] = str(st.selectbox("direction", ["up", "down"], key="p6_ts_dir"))
        f["monitoring"] = str(
            st.selectbox("monitoring", ["continuous", "discrete"], key="p6_ts_mon")
        )
        f["rebate"] = n("rebate", 0.0, step=0.01)
    return f


def _term_sheet(cfg: ViewerConfig, products: pd.DataFrame, surface_name: str) -> None:
    st.subheader("Term-sheet editor (prices off the cache)")
    st.caption(
        "Build a product from its term sheet (levels as fractions of spot; daily fixings 252/y). "
        "'Prices off the cache' = the stored price of every model when the term sheet is exactly "
        "one of the store's products; a custom term sheet needs a precompute run — no Monte "
        "Carlo runs in this page."
    )
    kind = str(st.selectbox("Product class", TERM_SHEETS, key="p6_ts_kind"))
    f = _fields(kind)
    try:
        rec = surface(cfg, surface_name)  # cached; SSVI + Dupire construction, no simulation
        spot, discount = rec.spot, rec.forward_curve.rate_curve
        product = build_product(kind, f, spot, discount)
    except (api.MissingArtefact, ValueError, NotImplementedError) as exc:
        st.error(f"Cannot build this term sheet: {exc}")
        return
    st.code(repr(product), language="text")
    ft = np.asarray(product.fixing_times, dtype=float)
    pay = np.asarray(product.pay_times).round(4).tolist()[:5]
    st.caption(
        f"spot {spot:g} · {ft.size} fixings from {ft[0]:g}y to {ft[-1]:g}y · pay times {pay}"
    )
    with st.expander("Fixing schedule"):
        table_with_export(pd.DataFrame({"fixing_time": ft}), f"fixings_{kind}")
    name = store_product_name(kind, f)
    if name is not None and (products["product"] == name).any():
        quantities = list(
            dict.fromkeys(str(q) for q in products.loc[products["product"] == name, "quantity"])
        )
        q = "price" if "price" in quantities else quantities[0]
        st.success(
            f"This term sheet is the store's product {name!r}: stored {q} per model on "
            f"{surface_name!r} (no pricing run)."
        )
        priced, unplottable, _ = model_risk_table(products, name, q)
        table_with_export(priced, f"cache_price_{name}")
        if unplottable:
            st.caption(
                f"{unplottable} stored price(s) of {name!r} carry no finite value and stderr "
                "(shown as stored; no MC number is quoted without its stderr)."
            )
        return
    snippet = yaml.safe_dump(
        {
            "products": {
                "custom": [
                    {
                        "name": f"{kind} custom",
                        "class": type(product).__name__,
                        "args": {k: v for k, v in f.items()},
                    }
                ]
            }
        },
        sort_keys=False,
    )
    st.warning(
        (
            "This term sheet matches the headline product "
            f"{name!r} but the store holds no price for it on {surface_name!r}. "
            if name
            else "Not one of the store's products. "
        )
        + "Pricing a custom term sheet needs a precompute run with a products list — the "
        "viewer never runs Monte Carlo. The `--products <yaml>` option is not in "
        "`volsto-precompute` yet (an S1 follow-up); the snippet below is its proposed input:"
    )
    st.code(
        f"{api.precompute_command(cfg)} --products custom_products.yaml\n\n"
        f"# custom_products.yaml\n{snippet}",
        language="yaml",
    )


def render(cfg: ViewerConfig) -> None:
    page_header(
        TITLE,
        cfg,
        subtitle=(
            "Price heatmaps over two grid axes; RiskReport summaries (delta and gamma regimes, "
            "fwd-var ladder, skew_T, vega-T waves) from the store's risk table; a term-sheet "
            "editor priced off the cache — nothing computed here."
        ),
    )
    grid = grid_or_none(cfg)
    if grid is None:
        return
    surface_name = select_surface(cfg, key="p6_surface")
    products = products_on_surface(cfg, grid, surface_name)
    if products.empty:
        st.warning(
            f"No products stored for the points of {surface_name!r}; run the precompute command "
            "printed above."
        )
    else:
        _heatmaps(cfg, products, surface_name)
    _risk(cfg, grid, surface_name)
    _term_sheet(cfg, products, surface_name)


run_if_streamlit(render)
