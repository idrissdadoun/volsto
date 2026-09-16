# ruff: noqa: N999 — Streamlit multipage file name
"""Page 8 — Hedging (SPEC §9 page 8, M9 Part 2; owner's ``viewers/pages/8_hedging`` = this
module): the stored M8 / M8b hedging runs, read through
:func:`volsto.viewers.api.list_hedging_runs`, :func:`~volsto.viewers.api.get_hedging_run`,
:func:`~volsto.viewers.api.get_hedging_pnl` and :func:`~volsto.viewers.api.get_hedging_table`.

Sections: the list of stored runs (one row per ``outputs/m8b/<study>/<key>.json``); for a chosen
run the P&L histogram of the per-path samples (the ``.pkl`` beside the JSON) with the mean ±
stderr band and the stored quantiles (± their bootstrap stderr as recorded by the M8b runner)
annotated, the distribution table, the regime tables (mean ± stderr; the per-regime ``std`` and
mean realised vol are recorded upstream without a stderr — flagged, not padded), the attribution
table with its bar chart, and the per-refit-date recalibration P&L when the run refitted; the
study-C comparison table ``m8b_table_C.csv`` (recalibration P&L vs the static desk-P&L shadow x
rota, the ratio with a first-order delta-method stderr added here, z and the 30 % flag) as a table
and a grouped bar chart; the study-A ranking table (std ± stderr per strategy) and the study-B
leakage table when those tables exist — the leakage bars hold only the rows carrying both the desk
leakage and its own ``leakage_desk_stderr``; a row with a leakage but a NaN stderr is left out of
the bars, counted in a caption and kept as written in the table (never a bar without its error
bar).  Every figure on the page holds to that rule through
:func:`~volsto.viewers.pages._common.mc_rows` (the one guard of the eight pages — this page and
page 7 grew it, ``pages/_common.py`` now owns it): the regime, attribution, study-A and study-C
bars and the per-refit-date recalibration line draw a row only when the stored table carries both
its value and its own stderr and neither is NaN, the histogram annotates a stored statistic only
when both are finite, and a table written without the stderr column names it with the
``scripts/m8b.py`` line that writes it
(:func:`~volsto.viewers.pages._common.missing_mc_notice`) and costs the figure, not the render.
P&L numbers are in the hedger's sign (long the product) and the run's reporting unit (``scale``
x the hedger's raw P&L, both read from the run's metadata — a run without them fails loudly
rather than plotting mis-scaled samples); the *desk convention*
toggle flips the sign of the histogram and its annotations (:func:`volsto.studies.m8b.to_desk_pnl`;
standard errors unchanged) and names each stored quantile's desk counterpart (the hedger's ``q05``
is the desk's ``q95``, :func:`desk_quantile`).

Read-only: nothing here hedges, prices, calibrates or simulates; when no run exists the page names
``scripts/m8b.py --study A|B|C|D`` (:func:`~volsto.viewers.components.missing_point_notice`).
Every table has the Excel export, every figure the PNG / SVG download; the ``value ± stderr``
display columns and the bar traces with error bars come from
:mod:`volsto.viewers.pages._common`.  Rendered headless by ``tests/test_viewers_pages_c.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from volsto.studies.m8b import STUDIES
from volsto.viewers import api
from volsto.viewers.components import (
    error_trace,
    figure_with_download,
    missing_point_notice,
    page_header,
    table_with_export,
)
from volsto.viewers.config import ViewerConfig
from volsto.viewers.pages import run_if_streamlit
from volsto.viewers.pages._common import (
    bar_stderr_trace,
    fmt_pm,
    mc_rows,
    missing_mc_notice,
    pm_display,
)

TITLE = "Hedging"
#: Significant digits of the ``value ± stderr`` texts on this page.
PM_DIGITS = 4
#: Per-path P&L components of :func:`~volsto.viewers.api.get_hedging_pnl` offered for the histogram.
PNL_COMPONENTS: tuple[str, ...] = (
    "pnl_total",
    "pnl_product",
    "pnl_hedges",
    "costs",
    "pnl_recalibration",
)
#: Histogram bins (Freedman-Diaconis would depend on the sample; a fixed count keeps the plot
#: comparable across runs).
HIST_BINS = 80
#: Study-C table columns (``table_C`` of :mod:`volsto.studies.m8b`): the simulated
#: recalibration P&L, the static prediction and the total, each with the stderr twin the read
#: API emits.  ``table_C`` writes exact twins since 2026-09-16 (``recal_pnl_desk_se``); a table
#: written before carries the stem (``recal_se``) and
#: :func:`volsto.viewers.api.reattach_stderr_names` gives it the value column's own name, so
#: what a page reads is ``<value>_stderr`` either way — the names below are
#: walked against :func:`volsto.viewers.api.get_hedging_table` by
#: ``tests/test_viewers_pages_c.py::test_page_twin_names_match_api``, so the next rename of the
#: read API fails a test instead of this page.
C_SIMULATED = ("recal_pnl_desk", "recal_pnl_desk_stderr")
C_STATIC = ("static_prediction", "static_prediction_stderr")
C_TOTAL = ("total_pnl_desk", "total_pnl_desk_stderr")
#: Study-B table: the desk leakage column; its stderr is the paired ``leakage_desk_stderr``
#: (``leakage_desk_se`` of :func:`volsto.studies.m8b.table_B` — a table written by an older
#: builder without that pair gets the explicit caption below, never a guessed twin).
B_LEAKAGE = "leakage_desk"


#: The ``(value, stderr)`` column pairs this page reads from the read API, per API frame
#: (``<accessor>.<table>``).  Declared rather than only spelled inline so that
#: ``tests/test_viewers_pages_c.py::test_page_twin_names_match_api`` can walk them against what
#: :mod:`volsto.viewers.api` actually returns for the synthetic store and the repository's real
#: store and outputs: a twin the API renames (its ``reattach_stderr_names`` pairing was rewritten
#: once already) then fails a test instead of a page.
#: (the per-refit-date recalibration table is written with either ``stderr`` or
#: ``mean_stderr``; the section reads whichever the stored table carries, so that one is
#: not a fixed pair.)
MC_PAIRS: dict[str, tuple[tuple[str, str], ...]] = {
    "hedging_table.A": (("std", "std_stderr"), ("desk_mean", "desk_mean_stderr")),
    "hedging_table.B": (("leakage_desk", "leakage_desk_stderr"),),
    "hedging_table.C": (C_SIMULATED, C_STATIC, C_TOTAL),
    "hedging_table.D": (("std", "std_stderr"), ("desk_mean", "desk_mean_stderr")),
    "hedging_run.distribution": (("value", "value_stderr"),),
    "hedging_run.regimes": (("mean", "mean_stderr"),),
    "hedging_run.attribution": (("value", "value_stderr"),),
}


def m8b_commands() -> str:
    return "\n".join(api.HEDGING_SCRIPT.format(study=s) for s in STUDIES)


# -- page-specific helpers (the ± display and bar traces are ``_common``'s) -------------------


def desk_quantile(q: str) -> str:
    """The desk-convention name of the hedger's stored quantile ``qNN``: the sign flip maps the
    hedger's ``NN``-th percentile onto the desk's ``(100 - NN)``-th (``q05`` → ``q95``)."""
    if not (q.startswith("q") and q[1:].isdigit()):
        raise ValueError(f"not a stored quantile name: {q!r}")
    return f"q{100 - int(q[1:]):02d}"


def bar_section(
    df: pd.DataFrame,
    *,
    x: str,
    value: str,
    stderr: str,
    title: str,
    unit: str,
    name: str,
    download: str,
    source: str,
    command: str,
) -> None:
    """One ``value ± stderr`` bar figure of a stored table (the regime means, the attribution):
    only the rows :func:`mc_rows` clears are drawn, an absent column costs the figure and not the
    render, and the rows left out are counted in a caption under it."""
    shown, n_nan, absent = mc_rows(df, value, stderr)
    if x not in df.columns:
        absent = [*absent, x]
    if absent:
        missing_mc_notice(title, absent, source, command)
    elif shown.empty:
        st.info(f"no row of {title!r} carries both a value and its stderr: nothing is plotted.")
    else:
        fig = go.Figure(bar_stderr_trace(shown[x], shown[value], shown[stderr], name=name))
        fig.update_layout(title=title, yaxis_title=unit, height=380)
        figure_with_download(fig, download)
    if n_nan:
        st.caption(
            f"{n_nan} row(s) with a NaN {value} or {stderr} are left out of the bars (no MC "
            "number is drawn without its error bar) and kept as written in the table."
        )


def ratio_stderr(sim: float, sim_se: float, pred: float, pred_se: float) -> float:
    """First-order (delta-method) stderr of ``ratio = sim / pred`` from independent errors:
    ``|ratio| · sqrt((sim_se / sim)² + (pred_se / pred)²)``; NaN when either value is 0."""
    if pred == 0.0 or sim == 0.0 or not (math.isfinite(pred) and math.isfinite(sim)):
        return math.nan
    return abs(sim / pred) * math.sqrt((sim_se / sim) ** 2 + (pred_se / pred) ** 2)


# --------------------------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------------------------


def runs_section(cfg: ViewerConfig) -> pd.DataFrame | None:
    st.subheader("Stored runs")
    runs = api.list_hedging_runs(cfg)
    if runs.empty:
        missing_point_notice(
            api.MissingArtefact(f"hedging runs under {cfg.outputs_root / 'm8b'}", m8b_commands())
        )
        return None
    st.caption(
        f"{len(runs)} task results under {cfg.outputs_root / 'm8b'} (hedger's sign — long the "
        f"product; {int(runs['has_paths'].sum())} with per-path P&L)"
    )
    table_with_export(runs, "hedging runs")
    return runs


def histogram_section(cfg: ViewerConfig, rec: api.HedgingRunRecord, desk: bool) -> None:
    try:
        pnl = api.get_hedging_pnl(cfg, rec.run_id)
    except api.MissingArtefact as exc:
        missing_point_notice(exc)
        return
    comp = st.selectbox(
        "P&L component", [c for c in PNL_COMPONENTS if c in pnl.columns], key="hist_component"
    )
    sign = -1.0 if desk else 1.0
    scale = float(rec.meta["scale"])  # hedger units → reporting unit; absent = loud KeyError
    x = sign * scale * pnl[comp].to_numpy(dtype=float)
    x = x[np.isfinite(x)]
    unit = str(rec.meta["unit"])
    fig = go.Figure(go.Histogram(x=x, nbinsx=HIST_BINS, name=comp, opacity=0.8))
    dist = rec.distribution.set_index("statistic")
    values = {str(k): float(v) for k, v in dist["value"].items()}
    errors = {str(k): float(v) for k, v in dist["value_stderr"].items()}
    # a stored statistic is annotated on the histogram only when it and its stderr are finite:
    # a NaN would draw a band of zero width and a "± nan" label beside a Monte Carlo number
    skipped: list[str] = []
    if comp == "pnl_total" and "mean" in values:
        m, se = sign * values["mean"], errors["mean"]
        if math.isfinite(m) and math.isfinite(se):
            fig.add_vrect(x0=m - se, x1=m + se, fillcolor="#e76f51", opacity=0.25, line_width=0)
            fig.add_vline(
                x=m,
                line_color="#e76f51",
                annotation_text=f"mean {fmt_pm(m, se, PM_DIGITS)}",
                annotation_position="top",
            )
        else:
            skipped.append("mean")
        for q in [k for k in values if k.startswith("q")]:
            v, qse = sign * values[q], errors[q]
            if not (math.isfinite(v) and math.isfinite(qse)):
                skipped.append(q)
                continue
            label = q if not desk else f"{q} -> desk {desk_quantile(q)}"
            fig.add_vline(
                x=v,
                line_dash="dot",
                line_color="#8d99ae",
                annotation_text=f"{label} {fmt_pm(v, qse, PM_DIGITS)}",
                annotation_position="bottom",
                annotation_textangle=-90,
            )
    if x.size > 1:
        sample_se = float(np.std(x, ddof=1) / math.sqrt(x.size))
        sample = f"sample mean {fmt_pm(float(np.mean(x)), sample_se, PM_DIGITS)} {unit}"
    else:  # one finite path has no sample stderr — say so rather than print a "± nan"
        sample = "too few finite paths for a sample mean ± stderr"
    convention = "desk sign" if desk else "hedger sign"
    fig.update_layout(
        title=f"{comp} of {rec.run_id} — {x.size} world paths, {sample} ({convention})",
        xaxis_title=f"{comp} ({unit})",
        yaxis_title="paths",
    )
    figure_with_download(fig, f"pnl histogram {rec.run_id} {comp}")
    if skipped:
        st.caption(
            f"{len(skipped)} stored statistic(s) ({', '.join(skipped)}) with a NaN value or "
            "stderr are not annotated on the histogram (no MC number is shown without its "
            "stderr); they are in the distribution table as stored."
        )


def run_section(cfg: ViewerConfig, runs: pd.DataFrame) -> None:
    st.subheader("One run")
    study = st.selectbox("Study", sorted(runs["study"].unique()), key="run_study")
    ids = runs.loc[runs["study"] == study, "run_id"].tolist()
    run_id = st.selectbox("Run", ids, key="run_id")
    try:
        rec = api.get_hedging_run(cfg, str(run_id))
    except api.MissingArtefact as exc:
        missing_point_notice(exc)
        return
    m = rec.meta
    st.caption(
        f"{m['product']} · world {m['world']} · strategy {m['strategy']} · "
        f"pricing {m['pricing']} · {m['frequency']} · status {m['status']} {m['reason']} · "
        f"unit {m['unit']} · {m['n_paths_world']} world paths · {m['n_dates']} dates · "
        f"{m['n_particles']} particles · refits {m['n_refits']} · "
        f"calibrations {m['calibrations']} · wall {m['wall_seconds']:.1f} s"
    )
    desk = st.checkbox(
        "desk convention (short the product: histogram sign flipped, stderr unchanged)",
        value=False,
        key="desk_sign",
    )
    histogram_section(cfg, rec, desk)
    table_with_export(
        rec.distribution,
        f"distribution {rec.run_id}",
        caption="distribution summary (hedger's sign): value ± stderr as stored by the M8b runner "
        "(quantile stderr by bootstrap)",
    )
    command = api.HEDGING_SCRIPT.format(study=m["study"])
    source = f"the stored task result {rec.run_id}.json"
    reg = rec.regimes
    if not reg.empty:
        c1, c2 = st.columns([3, 2])
        with c1:
            table_with_export(
                reg,
                f"regimes {rec.run_id}",
                caption="regime means ± stderr; 'std' and 'realised_vol_mean' are recorded "
                "upstream without a stderr (api.UNPAIRED_UPSTREAM['hedging_regimes'])",
            )
        with c2:
            bar_section(
                reg,
                x="regime",
                value="mean",
                stderr="mean_stderr",
                title="P&L mean by regime (± stderr)",
                unit=str(m["unit"]),
                name="regime mean",
                download=f"regimes {rec.run_id}",
                source=source,
                command=command,
            )
    att = rec.attribution
    if not att.empty:
        c1, c2 = st.columns([3, 2])
        with c1:
            table_with_export(att, f"attribution {rec.run_id}", caption="P&L attribution ± stderr")
        with c2:
            bar_section(
                att,
                x="component",
                value="value",
                stderr="value_stderr",
                title="attribution (± stderr)",
                unit=str(m["unit"]),
                name="attribution",
                download=f"attribution {rec.run_id}",
                source=source,
                command=command,
            )
    recal = rec.recalibration
    if not recal.empty and "t" in recal.columns:
        se_col = "stderr" if "stderr" in recal.columns else "mean_stderr"
        title = "per-date recalibration P&L (hedger's sign, ± stderr)"
        shown, n_nan, absent = mc_rows(recal, "mean", se_col)
        if absent:
            missing_mc_notice(title, absent, source, command)
        elif shown.empty:
            st.info(
                "no refit date carries both a recalibration P&L and its stderr: nothing is "
                f"plotted — regenerate with {command}."
            )
        else:
            fig = go.Figure(
                error_trace(
                    shown["t"].tolist(),
                    shown["mean"].tolist(),
                    shown[se_col].tolist(),
                    name="recalibration P&L per refit date",
                )
            )
            fig.update_layout(title=title, xaxis_title="t (y)", yaxis_title=m["unit"])
            figure_with_download(fig, f"recalibration by date {rec.run_id}")
        if n_nan:
            st.caption(
                f"{n_nan} refit date(s) with a NaN recalibration P&L or stderr are left out of "
                "the figure and kept as written in the table."
            )
        table_with_export(recal, f"recalibration by date {rec.run_id}")
    else:
        st.caption(f"no per-date recalibration P&L stored (n_refits {m['n_refits']}).")
    with st.expander("settings, budget, notes"):
        st.json({"settings": rec.settings, "budget": rec.budget, "notes": list(rec.notes)})


def study_c_section(cfg: ViewerConfig, runs: pd.DataFrame) -> None:
    st.subheader("Study C — recalibration P&L vs the static desk-P&L shadow x rota")
    try:
        c = api.get_hedging_table(cfg, "C")
    except api.MissingArtefact as exc:
        missing_point_notice(exc)
        return
    if c.empty or C_SIMULATED[0] not in c.columns:
        st.info(
            "m8b_table_C.csv is empty (no ok study-C run summarised yet): "
            + api.HEDGING_SCRIPT.format(study="C")
        )
        cr = runs[runs["study"] == "C"]
        if not cr.empty:
            table_with_export(
                cr,
                "study C task results",
                caption="study-C task results present (recal_total ± stderr)",
            )
        return
    c = c.copy()
    title = "study C: simulated vs static (desk sign, ± stderr)"
    command = api.HEDGING_SCRIPT.format(study="C")
    keys = ("product", "rota", "recalibration")
    absent = [k for k in (C_SIMULATED[1], *C_STATIC, *keys) if k not in c.columns]
    if absent:
        missing_mc_notice(title, absent, "m8b_table_C.csv", command)
    else:
        c["ratio_stderr"] = [
            ratio_stderr(float(s), float(sse), float(p), float(pse))
            for s, sse, p, pse in zip(
                c[C_SIMULATED[0]], c[C_SIMULATED[1]], c[C_STATIC[0]], c[C_STATIC[1]]
            )
        ]
        st.caption(
            "desk convention (short the note): simulated recalibration P&L (total over the refit "
            "dates) against the M7 static greek x rota under the same policy; ratio = simulated / "
            "static, its stderr by the first-order delta method (added here, independent errors), "
            "z = difference / combined stderr, within_30pct = |ratio - 1| <= 0.30."
        )

        def label(frame: pd.DataFrame) -> list[str]:
            return [
                f"{p} · rota {r:g} · {pol}"
                for p, r, pol in zip(frame["product"], frame["rota"], frame["recalibration"])
            ]

        # both series are Monte Carlo numbers (the static prediction carries the M7 greek's
        # stderr): a row is a bar only where its own value and stderr are both there
        sim, n_sim, _ = mc_rows(c, *C_SIMULATED)
        pred, n_pred, _ = mc_rows(c, *C_STATIC)
        traces = [
            bar_stderr_trace(label(frame), frame[value], frame[se], name=series)
            for frame, (value, se), series in (
                (sim, C_SIMULATED, "simulated recalibration P&L"),
                (pred, C_STATIC, "static prediction (greek x rota)"),
            )
            if not frame.empty
        ]
        if traces:
            fig = go.Figure(traces)
            fig.update_layout(
                barmode="group",
                title=title,
                yaxis_title=str(c["unit"].iloc[0]) if "unit" in c else "",
            )
            figure_with_download(fig, "study C comparison")
        else:
            st.info(
                "no study-C row carries a simulated or static value with its own stderr: "
                f"nothing is plotted — regenerate with {command}."
            )
        if n_sim or n_pred:
            st.caption(
                f"{n_sim} simulated and {n_pred} static value(s) with a NaN value or stderr are "
                "left out of the bars (no MC number is drawn without its error bar) and kept as "
                "written in the table."
            )
    first = [
        "product",
        "rota",
        "recalibration",
        "status",
        "unit",
        *C_SIMULATED,
        *C_STATIC,
        "ratio",
        "ratio_stderr",
        "within_30pct",
        "z",
        *C_TOTAL,
        "n_refits",
        "nonlinearity",
    ]
    cols = [k for k in first if k in c.columns] + [k for k in c.columns if k not in first]
    table_with_export(c[cols], "study C table")


def study_a_section(cfg: ViewerConfig) -> None:
    st.subheader("Study A — strategy ranking by P&L std")
    try:
        a = api.get_hedging_table(cfg, "A")
    except api.MissingArtefact as exc:
        missing_point_notice(exc)
        return
    if a.empty or "std" not in a.columns:
        st.info(f"m8b_table_A.csv is empty: {api.HEDGING_SCRIPT.format(study='A')}")
        return
    title = "P&L std by strategy (± stderr), ranked"
    command = api.HEDGING_SCRIPT.format(study="A")
    absent = [c for c in ("std_stderr", "strategy", "pricing", "product") if c not in a.columns]
    if absent:
        missing_mc_notice(title, absent, "m8b_table_A.csv", command)
    else:
        # a P&L std is a Monte Carlo number: a strategy is a bar only with its own stderr
        fig = go.Figure()
        n_nan = n_traces = 0
        for (pricing, product), sub in a.groupby(["pricing", "product"]):
            rows, dropped, _ = mc_rows(sub, "std", "std_stderr")
            n_nan += dropped
            if rows.empty:
                continue
            fig.add_trace(
                bar_stderr_trace(
                    rows["strategy"], rows["std"], rows["std_stderr"], name=f"{pricing} · {product}"
                )
            )
            n_traces += 1
        if n_traces:
            fig.update_layout(barmode="group", title=title, yaxis_title="std")
            figure_with_download(fig, "study A ranking")
        else:
            st.info(
                "no study-A row carries both a P&L std and its 'std_stderr': nothing is plotted "
                f"— regenerate with {command}."
            )
        if n_nan:
            st.caption(
                f"{n_nan} row(s) with a NaN std or std_stderr are left out of the bars (no MC "
                "number is drawn without its error bar) and kept as written in the table."
            )
    table_with_export(
        a, "study A table", caption="ranked by std within (pricing, product); desk_mean ± stderr"
    )


def study_b_section(cfg: ViewerConfig) -> None:
    st.subheader("Study B — leakage across worlds")
    try:
        b = api.get_hedging_table(cfg, "B")
    except api.MissingArtefact as exc:
        missing_point_notice(exc)
        return
    if b.empty:
        st.info(f"m8b_table_B.csv is empty: {api.HEDGING_SCRIPT.format(study='B')}")
        return
    se_col = f"{B_LEAKAGE}_stderr"
    if B_LEAKAGE in b.columns and se_col in b.columns:
        # a leakage is an MC number: a row is drawn only with its own stderr, never bare (the
        # shared guard; the caption below counts the rows that *have* a leakage but no stderr)
        ok, _, _ = mc_rows(b, B_LEAKAGE, se_col)
        n_nan = int((b[B_LEAKAGE].notna() & b[se_col].isna()).sum())
        if ok.empty:
            st.info(
                f"no study-B row carries both the desk leakage and its '{se_col}': nothing is "
                f"plotted — regenerate with {api.HEDGING_SCRIPT.format(study='B')}."
            )
        else:
            fig = go.Figure()
            for world, sub in ok.groupby("world"):
                fig.add_trace(
                    bar_stderr_trace(sub["product"], sub[B_LEAKAGE], sub[se_col], name=str(world))
                )
            fig.update_layout(
                barmode="group",
                title="desk leakage by world and product (± stderr)",
                yaxis_title="leakage",
            )
            figure_with_download(fig, "study B leakage")
        if n_nan:
            st.caption(
                f"{n_nan} row(s) with a desk leakage but a NaN {se_col} are left out of the bars "
                "(no MC number is drawn without its error bar) and kept as written in the table."
            )
    else:
        st.caption(
            f"no study-B row with the desk leakage and its stderr ('{B_LEAKAGE}' / '{se_col}'): "
            f"gated or skipped rows only, or a table written before table_B paired them — "
            f"regenerate with {api.HEDGING_SCRIPT.format(study='B')}."
        )
    table_with_export(b, "study B table")
    with st.expander("study B (value ± stderr view)"):
        table_with_export(pm_display(b, digits=PM_DIGITS), "study B (value ± stderr)")


def render(cfg: ViewerConfig) -> None:
    page_header(
        TITLE,
        cfg,
        subtitle="P&L distributions and attribution of the stored M8b runs: histograms, regime "
        "tables, the recalibration-vs-static shadow-rotation comparison (study C), the study-A "
        "ranking and the study-B leakage — read from outputs/m8b; nothing here hedges or prices.",
    )
    runs = runs_section(cfg)
    if runs is None:
        return
    run_section(cfg, runs)
    study_c_section(cfg, runs)
    study_a_section(cfg)
    study_b_section(cfg)


run_if_streamlit(render)
