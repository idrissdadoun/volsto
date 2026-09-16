# ruff: noqa: N999 — Streamlit multipage file name
"""Page 7 — Marking / calibration (SPEC §9 page 7, M9 Part 2; owner's ``viewers/pages/7_marking``
= this module): everything from M7 Part 3, browsable, through :func:`volsto.viewers.api.get_marking`
and the store's marking points (:func:`~volsto.viewers.api.list_grid` / :func:`~volsto.viewers.api.
get_point`).

Per (surface, ssr_target, skew_eps) of ``outputs/m7/p1_marking_fits.csv``: the status (interior /
binding / infeasible) with the active constraints, the bound flags and the rho12-collapse guard, the
fitted parameters, the naked-skew gaps at the two constraint pillars and the short-end gaps, the
mean |L - 1|, the realised LSV SSR by pillar (± stderr) against the ``ssr_target`` input, and the
stage-3 verdict with the engine-bias and reported-gap columns (recorded upstream without a standard
error — shown as such, never padded).  A ``<prefix>@<T>`` Monte Carlo pillar is shown only with
its ``<prefix>_se@<T>`` twin (:func:`at_columns`): one without it is named in an ``st.info`` with
the twin and :data:`M7_STAGE3` and is neither plotted nor tabulated, and a pillar whose value or
stderr is NaN is left out of the figure (counted in a caption) and kept as NaN in the table — no
invisible error bar, no padded zero.  The binding map as a status-coded heatmap over
(ssr_target, skew_eps) per snapshot from the store's marking points and from the M7 fits, and the
flag maps of ``spx_soft_skew.csv`` / ``skew_tradeoff.csv`` over (ssr_target, skew_weight) when those
files exist.  The shadow-rotation table per policy (lv_rotation, usual, recalibrated, fee_shadow,
desk_pnl_shadow, … with stderr) under the printed
:data:`~volsto.risk.shadow_rotation.ROTATION_CONVENTION`.

The same rule holds wherever this page draws a Monte Carlo number
(:func:`~volsto.viewers.pages._common.mc_rows`, the one guard of the eight pages — this page and
page 8 grew it, ``pages/_common.py`` now owns it): the store point's SSR term structure plots a
pillar only when the store's ``ssr`` table carries both ``ssr_lsv`` and ``ssr_lsv_stderr`` for it,
and the shadow-rotation bars a greek only when the file carries both ``per_rota`` and
``per_rota_se``; a missing column is named with the producing script
(:func:`~volsto.viewers.pages._common.missing_mc_notice`) and costs the figure, not the render,
and the NaN rows are counted in a caption and kept as written in the table.

The fit selectors are dependent: the surface defaults to :data:`DEFAULT_SURFACE` when it is
fitted (else the first snapshot), and the ``skew_eps`` slider offers only the values fitted for
the chosen ``ssr_target`` on that surface (the M7 file is a sparse list of pairs, not a grid —
an independent cross product lands on an absent pair by default); the store section snaps the
same way.  Read-only: nothing here fits, calibrates or simulates; a missing M7 file prints
``scripts/m7_p1_marking.py`` (a missing fit the exact ``--surfaces <surface> --pairs
<ssr>:<eps>`` line, :func:`fit_command`) and a missing store point the ``volsto-precompute``
command (:func:`~volsto.viewers.components.missing_point_notice`).  Every table has the Excel
export, every figure the PNG / SVG download; the ``value ± stderr`` display columns, the bar
traces with error bars come from :mod:`volsto.viewers.pages._common`.  Rendered headless by
``tests/test_viewers_pages_c.py``.
"""

from __future__ import annotations

import math
import re

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from volsto.viewers import api
from volsto.viewers.components import (
    error_trace,
    figure_with_download,
    missing_point_notice,
    page_header,
    snap_slider,
    table_with_export,
)
from volsto.viewers.config import ViewerConfig
from volsto.viewers.pages import run_if_streamlit
from volsto.viewers.pages._common import (
    bar_stderr_trace,
    mc_rows,
    missing_mc_notice,
    pm_display,
)

TITLE = "Marking / calibration"
#: The ``(value, stderr)`` column pairs this page reads from the read API, per API frame
#: (``<accessor>.<table>``).  Declared rather than only spelled inline so that
#: ``tests/test_viewers_pages_c.py::test_page_twin_names_match_api`` can walk them against what
#: :mod:`volsto.viewers.api` actually returns for the synthetic store and the repository's real
#: store and outputs: a twin the API renames (its ``reattach_stderr_names`` pairing was rewritten
#: once already) then fails a test instead of a page.
#: (the fits file's ``<prefix>@<T>`` pillars are walked by :func:`at_columns` per row.)
MC_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "point.ssr": (("ssr_lsv", "ssr_lsv_stderr"),),
    "marking.shadow_rotation": (("per_rota", "per_rota_stderr"),),
}
#: Surface / snapshot the fit section opens on when it is among the fits (the deck's SPX fit is
#: the primary result; the reference surface is its diagnostic twin); else the first snapshot.
DEFAULT_SURFACE = "spx"
#: Significant digits of the ``value ± stderr`` display columns on this page.
PM_DIGITS = 4
#: The M7 fits report the mean |L - 1| of the leverage function and the fit-time SVC / vol-of-var
#: / correlation errors as single numbers: particle-method summaries without a standard error.
#: The API declares them exact (``api.EXACT_PREFIXES`` / ``api.EXACT_COLUMNS``); whether they
#: belong in ``api.UNPAIRED_UPSTREAM['marking_fits']`` instead is an open API-side item — the page
#: shows them with this note rather than bare.
LEVERAGE_SUMMARY_NOTE = (
    "leverage-function summary recorded upstream without a standard error "
    "(mean |L - 1|, svc_err_max, volvar_err_max, corr_*: particle-method summaries the API "
    "declares exact — not padded here)"
)
#: Status code of the binding-map heatmaps (colour scale below); anything else is "other".
STATUS_CODES: dict[str, int] = {"interior": 0, "binding": 1, "infeasible": 2, "other": 3}
STATUS_SCALE = [[0.0, "#2a9d8f"], [0.34, "#e9c46a"], [0.67, "#e76f51"], [1.0, "#8d99ae"]]
#: Fitted parameters of the M7 fits file (k2 is fixed at 0.2 by the fit and not recorded).
FIT_PARAMS: tuple[str, ...] = ("nu", "theta", "k1", "rho_SX1", "rho_SX2", "rho12")
#: The M7 fit columns of the stage-3 verdict (SPEC §15 Part 3).
STAGE3_COLUMNS: tuple[str, ...] = (
    "stage3_assertion",
    "stage3_max_engine_bias_svc",
    "stage3_max_engine_bias_volvar",
    "stage3_max_gap_svc_reported",
    "stage3_max_gap_volvar_reported",
    "stage3_seconds",
)
#: Flag columns of the binding maps (``soft_skew`` / ``skew_tradeoff``) composing a cell's label.
BINDING_FLAGS: tuple[str, ...] = (
    "clamped",
    "k1_at_bound",
    "nu_limit_binding",
    "nu_above_flag",
    "second_bound_flags",
)
#: Shadow-rotation greeks in the report's order (others present in the file are appended).
ROTATION_GREEKS: tuple[str, ...] = (
    "lv_rotation",
    "usual",
    "recalibrated",
    "fee_usual",
    "fee_recalibrated",
    "fee_shadow",
    "desk_pnl_usual",
    "desk_pnl_recalibrated",
    "desk_pnl_shadow",
)
#: The stage of the M7 runner that writes the ``@T`` Monte Carlo columns (the realised LSV SSR
#: by pillar, the forward / spot ratios) and their ``_se@T`` twins.
M7_STAGE3 = f"{api.MARKING_SCRIPT} stage 3"
_AT = re.compile(r"^(?P<name>.+)@(?P<T>[^_]+)$")


# --------------------------------------------------------------------------------------------
# helpers (page-specific; the ± display and bar traces are ``_common``'s)
# --------------------------------------------------------------------------------------------


def fit_command(surface: str, ssr: float, eps: float) -> str:
    """The exact ``scripts/m7_p1_marking.py`` line that fits ``(surface, ssr_target, skew_eps)``
    (its ``--surfaces`` / ``--pairs ssr_target:skew_eps`` arguments)."""
    return f"{api.MARKING_SCRIPT} --surfaces {surface} --pairs {ssr:g}:{eps:g}"


def dependent_axes(
    sub: pd.DataFrame, *, first: str, second: str, keys: tuple[str, str], suffix: str = ""
) -> tuple[float, float]:
    """The two snap sliders of a sparse (``first``, ``second``) list: the second offers only the
    values stored for the chosen first value, so the pair is always one of ``sub``'s rows."""
    a = snap_slider(f"{first}{suffix}", sub[first].tolist(), key=keys[0])
    b = snap_slider(f"{second}{suffix}", sub.loc[sub[first] == a, second].tolist(), key=keys[1])
    return float(a), float(b)


def status_heatmap(
    df: pd.DataFrame, *, x: str, y: str, status: str, title: str, text: str | None = None
) -> go.Figure:
    """A heatmap over (``x``, ``y``) coloured by :data:`STATUS_CODES` of ``status``, the status
    (or ``text``) written in the cell; missing combinations are blank."""
    xs = sorted(df[x].dropna().unique().tolist())
    ys = sorted(df[y].dropna().unique().tolist())
    z = [[math.nan] * len(xs) for _ in ys]
    labels = [[""] * len(xs) for _ in ys]
    for _, r in df.iterrows():
        i, j = ys.index(r[y]), xs.index(r[x])
        s = str(r[status])
        z[i][j] = STATUS_CODES.get(s, STATUS_CODES["other"])
        labels[i][j] = str(r[text]) if text and pd.notna(r[text]) else s
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=[f"{v:g}" for v in xs],
            y=[f"{v:g}" for v in ys],
            text=labels,
            texttemplate="%{text}",
            zmin=0,
            zmax=3,
            colorscale=STATUS_SCALE,
            colorbar={
                "tickvals": list(STATUS_CODES.values()),
                "ticktext": list(STATUS_CODES),
            },
            hovertemplate=f"{x}=%{{x}}, {y}=%{{y}}: %{{text}}<extra></extra>",
        )
    )
    fig.update_layout(title=title, xaxis_title=x, yaxis_title=y, height=320)
    return fig


def file_stderr_name(column: str) -> str:
    """The **file** column whose absence leaves ``column`` unpaired: ``<prefix>@<T>`` →
    ``<prefix>_se@<T>``, the spelling ``scripts/m7_p1_marking.py`` writes and
    :func:`volsto.viewers.api.normalise_stderr_names` renames to ``<prefix>@<T>_stderr``."""
    m = _AT.match(column)
    return f"{m.group('name')}_se@{m.group('T')}" if m else f"{column}_se"


def at_columns(
    row: pd.Series, prefix: str, *, paired: bool = True
) -> tuple[pd.DataFrame, list[str]]:
    """``<prefix>@<T>`` cells of a fit row → ``(frame, unpaired)``.

    ``paired`` (the default, a Monte Carlo quantity): a pillar enters the ``(T, value, stderr)``
    frame only when its ``<prefix>@<T>_stderr`` twin is in the row; a pillar without one is
    returned by name in ``unpaired`` instead of being padded with a NaN stderr — the caller names
    it and the producing script and plots nothing for it.  ``paired=False`` (an exact first-order
    quantity, e.g. the naked-skew short gaps): the frame is ``(T, value)`` and ``unpaired`` is
    empty."""
    out: list[dict[str, object]] = []
    unpaired: list[str] = []
    for c in row.index:
        m = _AT.match(str(c))
        if not (m and m.group("name") == prefix and not str(c).endswith("_stderr")):
            continue
        if not paired:
            out.append({"T": m.group("T"), prefix: float(row[c])})
            continue
        se = f"{c}_stderr"
        if se not in row.index:
            unpaired.append(str(c))
            continue
        out.append({"T": m.group("T"), prefix: float(row[c]), f"{prefix}_stderr": float(row[se])})
    cols = ["T", prefix, f"{prefix}_stderr"] if paired else ["T", prefix]
    return pd.DataFrame(out, columns=cols), unpaired


def unpaired_notice(source: str, columns: list[str], what: str) -> None:
    """One ``st.info`` per ``<prefix>@<T>`` column of the fits file that has no stderr twin: the
    column, the twin the file should carry (:func:`file_stderr_name`), the producing stage and
    the consequence — the Monte Carlo number is neither plotted nor tabulated, never drawn with
    an invisible NaN error bar (page 3's ``_m7_ratio_section`` prints the same notice)."""
    for col in columns:
        st.info(
            f"{col} is in {source}/p1_marking_fits.csv without its stderr column "
            f"{file_stderr_name(col)} (written by {M7_STAGE3}): the {what} of that pillar is "
            "neither plotted nor tabulated — no MC number is shown without its stderr."
        )


def flags_label(r: pd.Series) -> str:
    on = [f for f in BINDING_FLAGS if f in r.index and pd.notna(r[f]) and bool(r[f])]
    if "first_order_valid" in r.index and pd.notna(r["first_order_valid"]):
        on.append("fo valid" if bool(r["first_order_valid"]) else "fo invalid")
    return "; ".join(on) or "interior"


def flags_status(r: pd.Series) -> str:
    if any(f in r.index and pd.notna(r[f]) and bool(r[f]) for f in BINDING_FLAGS):
        return "binding"
    return "interior"


# --------------------------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------------------------


def fit_section(rec: api.MarkingRecord) -> None:
    fits = rec.fits
    st.subheader("Marking fits (M7 Part 3)")
    st.caption(f"source: {rec.source}/p1_marking_fits.csv — {len(fits)} fits")
    surfaces = sorted(fits["surface"].unique())
    surface = st.selectbox(
        "Surface / snapshot",
        surfaces,
        index=surfaces.index(DEFAULT_SURFACE) if DEFAULT_SURFACE in surfaces else 0,
        key="m7_surface",
    )
    sub = fits[fits["surface"] == surface]
    if sub.empty:  # a header-only fits CSV: no axis to snap to, so say what produces one
        missing_point_notice(
            api.MissingArtefact(f"any fit for {surface}", api.MARKING_SCRIPT),
        )
        return
    ssr, eps = dependent_axes(sub, first="ssr_target", second="skew_eps", keys=("m7_ssr", "m7_eps"))
    hit = sub[(sub["ssr_target"] == ssr) & (sub["skew_eps"] == eps)]
    if hit.empty:  # unreachable through the dependent sliders; kept as the loud guard
        missing_point_notice(
            api.MissingArtefact(
                f"fit ({surface}, ssr_target {ssr:g}, skew_eps {eps:g})",
                fit_command(surface, ssr, eps),
            )
        )
        return
    row = hit.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("status", str(row["status"]))
    c2.metric("mean |L - 1|", f"{float(row['mean_abs_L_minus_1']):.4f}")
    c2.caption(LEVERAGE_SUMMARY_NOTE)
    c3.metric("stage-3 verdict", str(row.get("stage3_assertion", "n/a")))
    st.markdown(
        f"**active constraints:** {row.get('active', '') or '—'}  \n"
        f"**bound flags:** {row.get('bounds', '') if pd.notna(row.get('bounds')) else '—'}  \n"
        f"**rho12 collapse guard:** {row.get('rho12_collapse', '')} · "
        f"fit {float(row.get('fit_seconds', math.nan)):.2f} s · "
        f"recalibrated: {row.get('recalibrated', '')}"
    )
    params = pd.DataFrame([{p: float(row[p]) for p in FIT_PARAMS if p in row.index}])
    table_with_export(
        params, f"fit params {surface} ssr{ssr:g} eps{eps:g}", caption="fitted params"
    )
    gaps = pd.DataFrame(
        [
            {
                "constraint": "short pillar",
                "T": f"{float(row['T_T_s']):g}y",
                "skew_gap": float(row["skew_gap_T_s"]),
            },
            {
                "constraint": "long pillar",
                "T": f"{float(row['T_T_l']):g}y",
                "skew_gap": float(row["skew_gap_T_l"]),
            },
        ]
    )
    short, _ = at_columns(row, "short_gap", paired=False)
    for _, g in short.iterrows():
        gaps.loc[len(gaps)] = {"constraint": "short end", "T": g["T"], "skew_gap": g["short_gap"]}
    table_with_export(
        gaps,
        f"naked skew gaps {surface} ssr{ssr:g} eps{eps:g}",
        caption="naked-skew gaps (exact first-order quantities): at the two constraint pillars "
        "and the short end",
    )
    ssr_tab, ssr_unpaired = at_columns(row, "ssr_lsv")
    unpaired_notice(rec.source, ssr_unpaired, "realised LSV SSR")
    if not ssr_tab.empty:
        ssr_tab["ssr_target"] = ssr
        # the NaN pillars stay NaN in the table below
        shown, n_nan, _ = mc_rows(ssr_tab, "ssr_lsv", "ssr_lsv_stderr")
        if shown.empty:
            st.info(
                "no realised-SSR pillar with both a value and its stderr: nothing is plotted "
                f"(regenerate with {fit_command(surface, ssr, eps)})."
            )
        else:
            fig = go.Figure(
                error_trace(
                    shown["T"].tolist(),
                    shown["ssr_lsv"].tolist(),
                    shown["ssr_lsv_stderr"].tolist(),
                    name="realised LSV SSR",
                )
            )
            fig.add_hline(y=ssr, line_dash="dash", annotation_text=f"ssr_target {ssr:g}")
            fig.update_layout(
                title="realised LSV SSR by pillar (stage 3) vs the ssr_target input",
                xaxis_title="pillar",
                yaxis_title="SSR",
            )
            figure_with_download(fig, f"realised SSR {surface} ssr{ssr:g} eps{eps:g}")
        if n_nan:
            st.caption(
                f"{n_nan} pillar(s) with a NaN realised SSR or stderr (an infeasible or unmarked "
                "pillar) are left out of the figure and kept as NaN in the table."
            )
        table_with_export(ssr_tab, f"realised SSR {surface} ssr{ssr:g} eps{eps:g}")
    fwd, fwd_unpaired = at_columns(row, "fwd/spot")
    unpaired_notice(rec.source, fwd_unpaired, "forward / spot skew ratio")
    if not fwd.empty:
        table_with_export(
            fwd,
            f"fwd-spot skew ratio {surface} ssr{ssr:g} eps{eps:g}",
            caption="forward / spot 90-110 skew ratio (the deck's 1.3-1.5 question) ± stderr",
        )
    stage3 = pd.DataFrame([{c: row[c] for c in STAGE3_COLUMNS if c in row.index}])
    table_with_export(
        stage3,
        f"stage3 {surface} ssr{ssr:g} eps{eps:g}",
        caption="stage-3 verdict; the engine-bias and reported-gap maxima are recorded upstream "
        "without a standard error (api.UNPAIRED_UPSTREAM['marking_fits']) — an open item of "
        "scripts/m7_p1_marking.py, not padded here",
    )
    with st.expander("all fits (raw table, value ± stderr)"):
        table_with_export(
            pm_display(fits, digits=PM_DIGITS), "all marking fits", caption=LEVERAGE_SUMMARY_NOTE
        )


def store_points_section(cfg: ViewerConfig, grid: pd.DataFrame) -> None:
    st.subheader("Store marking points (volsto-precompute)")
    marks = grid[grid["mode"] == "marking"]
    if marks.empty:
        missing_point_notice(api.MissingPoint("marking points", api.precompute_command(cfg)))
        return
    for surface, sub in marks.groupby("surface"):
        figure_with_download(
            status_heatmap(
                sub,
                x="skew_eps",
                y="ssr_target",
                status="status",
                title=f"binding map — store points on {surface}",
            ),
            f"binding map store {surface}",
        )
    surface = st.selectbox("Snapshot", sorted(marks["surface"].unique()), key="store_surface")
    sub = marks[marks["surface"] == surface]
    ssr, eps = dependent_axes(
        sub,
        first="ssr_target",
        second="skew_eps",
        keys=("store_ssr", "store_eps"),
        suffix=" (store)",
    )
    hit = sub[(sub["ssr_target"] == ssr) & (sub["skew_eps"] == eps)]
    if hit.empty:  # unreachable through the dependent sliders; kept as the loud guard
        missing_point_notice(
            api.MissingPoint(
                f"marking:{surface}:ssr{ssr:g}:eps{eps:g}", api.precompute_command(cfg)
            )
        )
        return
    rec = api.get_point(cfg, str(hit.iloc[0]["id"]))
    cache_key = rec.diagnostics.get("cache_key", "")
    st.markdown(f"**{rec.label}** — status `{rec.status}` · cache key `{cache_key}`")
    table_with_export(
        pd.DataFrame([rec.params]), f"store fit params {rec.id}", caption="fitted params"
    )
    if rec.fit:
        table_with_export(
            pd.DataFrame([rec.fit]), f"store fit record {rec.id}", caption="fit record"
        )
    if not rec.ssr.empty:
        # the store's realised SSR is a Monte Carlo number: a pillar is drawn only with its own
        # stderr (the same rule as the M7 fits above), never with an invisible NaN error bar
        shown, n_nan, missing = mc_rows(rec.ssr, "ssr_lsv", "ssr_lsv_stderr")
        if missing:
            missing_mc_notice(
                "SSR term structure of the store point",
                missing,
                f"the store's ssr table of {rec.id}",
                api.precompute_command(cfg),
            )
        elif shown.empty:
            st.info(
                "no stored SSR pillar carries both a value and its stderr: nothing is plotted "
                f"(regenerate with {api.precompute_command(cfg)})."
            )
        else:
            fig = go.Figure(
                error_trace(
                    shown["T"].tolist(),
                    shown["ssr_lsv"].tolist(),
                    shown["ssr_lsv_stderr"].tolist(),
                    name="LSV numerical",
                )
            )
            naked = (
                shown[shown["ssr_naked_first_order"].notna()]
                if "ssr_naked_first_order" in shown.columns
                else shown.iloc[:0]
            )
            if not naked.empty:  # exact first-order values; a NaN pillar is a gap, not a number
                fig.add_trace(
                    go.Scatter(
                        x=naked["T"],
                        y=naked["ssr_naked_first_order"],
                        name="naked (first order, exact)",
                        mode="lines+markers",
                    )
                )
            fig.add_hline(y=ssr, line_dash="dash", annotation_text=f"ssr_target {ssr:g}")
            fig.update_layout(
                title="SSR term structure of the store point",
                xaxis_title="T (y)",
                yaxis_title="SSR",
            )
            figure_with_download(fig, f"store SSR {rec.id}")
        if n_nan:
            st.caption(
                f"{n_nan} pillar(s) with a NaN stored SSR or stderr are left out of the figure "
                "and kept as NaN in the table."
            )
        table_with_export(rec.ssr, f"store SSR table {rec.id}")


def binding_maps_section(rec: api.MarkingRecord) -> None:
    st.subheader("Binding maps")
    fits = rec.fits
    for surface, sub in fits.groupby("surface"):
        figure_with_download(
            status_heatmap(
                sub,
                x="skew_eps",
                y="ssr_target",
                status="status",
                title=f"binding map — M7 fits on {surface}",
            ),
            f"binding map fits {surface}",
        )
    for name, df in (
        ("spx_soft_skew.csv", rec.soft_skew),
        ("skew_tradeoff.csv", rec.skew_tradeoff),
    ):
        if df.empty:
            st.info(f"{name} not present under {rec.source} (produced by {api.MARKING_SCRIPT}).")
            continue
        view = df
        if "fit" in df.columns and df["fit"].nunique() > 1:
            fit = st.selectbox(f"{name}: fit", sorted(df["fit"].unique()), key=f"fit_{name}")
            view = df[df["fit"] == fit]
        # the API's 150-column maps arrive block-fragmented; ``copy()`` consolidates them so the
        # two label columns insert without pandas' PerformanceWarning
        view = view.copy()
        view["flags"] = view.apply(flags_label, axis=1)
        view["flag_status"] = view.apply(flags_status, axis=1)
        figure_with_download(
            status_heatmap(
                view,
                x="skew_weight",
                y="ssr_target",
                status="flag_status",
                text="flags",
                title=f"flag map — {name} (ssr_target x skew_weight)",
            ),
            f"flag map {name}",
        )
        with st.expander(f"{name} (raw table)"):
            table_with_export(pm_display(view, digits=PM_DIGITS), name)


def shadow_rotation_section(rec: api.MarkingRecord) -> None:
    st.subheader("Shadow rotation")
    st.code(rec.convention, language="text")
    rot = rec.shadow_rotation
    source = f"{rec.source}/p1_marking_shadow_rotation.csv"
    if rot.empty or not {"policy", "greek"} <= set(rot.columns):
        st.info(f"{source} carries no policy / greek rows (written by {api.MARKING_SCRIPT}).")
        return
    policies = sorted(rot["policy"].unique())
    chosen = st.multiselect("policies", policies, default=policies, key="rot_policies")
    view = rot[rot["policy"].isin(chosen)]
    order = [g for g in ROTATION_GREEKS if g in set(view["greek"])] + sorted(
        set(view["greek"]) - set(ROTATION_GREEKS)
    )
    title = "P&L per +1 rota by greek and policy (± stderr)"
    absent = [c for c in ("per_rota", "per_rota_stderr") if c not in view.columns]
    if absent:
        missing_mc_notice(title, absent, source, api.MARKING_SCRIPT)
    else:
        # a per-rota P&L is a Monte Carlo number: a greek is a bar only when it carries both the
        # value and its own stderr — ``reindex`` alone would pad the missing greeks back in
        fig = go.Figure()
        n_nan = n_traces = 0
        for policy, sub in view.groupby("policy"):
            rows, dropped, _ = mc_rows(sub, "per_rota", "per_rota_stderr")
            n_nan += dropped
            if rows.empty:
                continue
            rows = rows.set_index("greek")
            rows = rows.reindex([g for g in order if g in rows.index])
            fig.add_trace(
                bar_stderr_trace(
                    rows.index, rows["per_rota"], rows["per_rota_stderr"], name=str(policy)
                )
            )
            n_traces += 1
        if n_traces:
            fig.update_layout(barmode="group", title=title, yaxis_title="per rota (price units)")
            figure_with_download(fig, "shadow rotation per rota")
        else:
            st.info(
                "no shadow-rotation row carries both a per-rota P&L and its stderr: nothing is "
                f"plotted (regenerate with {api.MARKING_SCRIPT})."
            )
        if n_nan:
            st.caption(
                f"{n_nan} greek/policy row(s) with a NaN per-rota P&L or stderr are left out of "
                "the bars (no MC number is drawn without its error bar) and kept as written in "
                "the table."
            )
    cols = [
        c
        for c in (
            "surface",
            "policy",
            "greek",
            "of",
            "per_rota",
            "per_rota_stderr",
            "per_vp_90_110_0.5y",
            "per_vp_90_110_0.5y_stderr",
            "per_vp_90_110_1y",
            "per_vp_90_110_1y_stderr",
            "p1_level",
            "p1_level_stderr",
            "lv_level",
            "lv_level_stderr",
            "fee",
            "fee_stderr",
        )
        if c in view.columns
    ]
    table_with_export(view[cols].reset_index(drop=True), "shadow rotation table")


def render(cfg: ViewerConfig) -> None:
    page_header(
        TITLE,
        cfg,
        subtitle="The marking-fit diagnostics per (ssr_target, skew_eps, snapshot): status, fitted "
        "params, naked-skew gaps, |L - 1|, realised SSR, the binding map, the shadow-rotation "
        "table — read from outputs/m7 and the results store; nothing here fits or calibrates.",
    )
    try:
        rec = api.get_marking(cfg)
    except api.MissingArtefact as exc:
        missing_point_notice(exc)
        rec = None
    if rec is not None:
        fit_section(rec)
    store_points_section(cfg, api.list_grid(cfg))
    if rec is not None:
        binding_maps_section(rec)
        shadow_rotation_section(rec)


run_if_streamlit(render)
