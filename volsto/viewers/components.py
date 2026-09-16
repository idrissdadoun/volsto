"""Shared Streamlit / Plotly components of the viewer pages (SPEC §9.2 "Read API and app", M9
Part 2): a grid-snapping slider, plotly error bars, tables with an Excel export button, figures
with PNG / SVG download buttons, the missing-point notice and the page header.  Pure
presentation — nothing here reads a model, calibrates or simulates; every number a page plots
through these helpers arrives with its standard error (``stderr_bar`` draws it, ``hover_stderr``
shows it in the hover text).

* :func:`snap_slider` — ``st.select_slider`` over the grid's values only (the owner's "sliders
  snap to grid points"): a page can never ask for a point that is not on the grid.
* :func:`stderr_bar` / :func:`error_trace` — a scatter trace with ``error_y`` from the stderr.
* :func:`table_with_export` — ``st.dataframe`` plus an *Excel* ``st.download_button`` built in
  memory with openpyxl (the owner's "Excel export button on every table").
* :func:`figure_with_download` — the figure plus PNG and SVG download buttons rendered lazily
  through ``fig.to_image`` (kaleido); when the image engine fails the buttons fall back to an
  HTML download and say so.
* :func:`missing_point_notice` — the exact ``volsto-precompute`` command of a
  :class:`~volsto.viewers.api.MissingArtefact`, in a copyable code block; never runs it.
* :func:`page_header` — title, the store / cache / outputs roots, the store's code tag, git
  commit, grid and particle count, the point and run counts (from
  :func:`~volsto.viewers.api.store_summary`).
* :func:`excel_bytes` / :func:`figure_bytes` — the export encoders (testable without Streamlit).
* :func:`default_key` — the widget key of a table / figure the caller did not key explicitly:
  the display name, made unique within the script run so that two tables (or two figures) of
  the same name on one page cannot raise ``StreamlitDuplicateElementKey``.

Checked by ``tests/test_viewers_api.py`` (``test_export_helpers``, ``test_pages_render``).
"""

from __future__ import annotations

import io
import re
import threading
from collections.abc import Callable, Sequence
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx

from volsto.viewers.api import MissingArtefact, MissingPoint, store_summary
from volsto.viewers.config import ViewerConfig

#: Sheet-name rule of openpyxl (31 chars, no ``[]:*?/\``).
_SHEET_BAD = re.compile(r"[\[\]:*?/\\]")
_FILE_BAD = re.compile(r"[^A-Za-z0-9_.+=-]+")
#: Image export scale of the PNG download (2 = retina-sharp).
PNG_SCALE = 2
#: Download formats offered per figure and their MIME types.
IMAGE_FORMATS: tuple[tuple[str, str], ...] = (("png", "image/png"), ("svg", "image/svg+xml"))


#: Per-script-run state of :func:`default_key`, **thread-local** because Streamlit runs each
#: browser session's script in its own ScriptRunner thread: the run whose keys are counted (the
#: context's ``cursors`` dict, replaced by a fresh object on every run — held by reference, so
#: the identity test cannot be fooled by a recycled address) and the counts per base key.
_KEY_RUN = threading.local()


def safe_filename(name: str) -> str:
    return _FILE_BAD.sub("_", name).strip("_") or "table"


def default_key(base: str) -> str:
    """A widget key for a helper the caller gave no ``key``: ``base`` the first time that base is
    used in this script run, then ``base#2``, ``base#3`` …

    Two tables (or figures) with the same display name on one page would otherwise raise
    ``StreamlitDuplicateElementKey`` (review:S2 F11).  The counter is per script run — the same
    sequence of elements produces the same keys on a rerun, so widget state is stable — and it
    is reset by the identity of the run's element cursors.  The state is **thread-local**:
    Streamlit runs each browser session's script in its own ScriptRunner thread, and a
    process-wide counter would interleave two viewers' keys.  Outside a Streamlit run (a unit
    test calling the helper directly) the counter simply keeps running in that thread."""
    ctx = get_script_run_ctx(suppress_warning=True)
    token = None if ctx is None else ctx.cursors
    if getattr(_KEY_RUN, "token", "unset") is not token:
        _KEY_RUN.token = token
        _KEY_RUN.counts = {}
    counts: dict[str, int] = _KEY_RUN.counts
    n = counts.get(base, 0) + 1
    counts[base] = n
    return base if n == 1 else f"{base}#{n}"


def sheet_name(name: str) -> str:
    return (_SHEET_BAD.sub("_", name) or "table")[:31]


# --------------------------------------------------------------------------------------------
# sliders and error bars
# --------------------------------------------------------------------------------------------


def snap_slider[T](
    label: str,
    values: Sequence[T],
    *,
    value: T | None = None,
    key: str | None = None,
    format_func: Callable[[Any], Any] = str,
    help: str | None = None,
) -> T:
    """A slider whose positions are exactly ``values`` (sorted, de-duplicated) — the grid's
    axis; with a single value it renders the value and returns it."""
    opts = sorted(set(values))  # type: ignore[type-var]  # grid values are ordered numbers
    if not opts:
        raise ValueError(f"{label}: no grid values to snap to")
    if len(opts) == 1:
        st.caption(f"{label}: {format_func(opts[0])} (single grid value)")
        return opts[0]
    default = value if value is not None and value in opts else opts[0]
    out = st.select_slider(
        label, options=opts, value=default, key=key, format_func=format_func, help=help
    )
    return out


def hover_stderr(y: Sequence[float], se: Sequence[float], unit: str = "") -> list[str]:
    """Hover strings ``value ± stderr unit`` (the owner's "every plotted number carries its
    stderr, error bars or a hover value")."""
    return [f"{v:.6g} ± {s:.2g} {unit}".strip() for v, s in zip(y, se)]


def error_trace(
    x: Sequence[Any],
    y: Sequence[float],
    se: Sequence[float],
    *,
    name: str,
    unit: str = "",
    mode: str = "lines+markers",
    **kwargs: Any,
) -> go.Scatter:
    """A scatter trace with ``error_y`` = stderr and the ``± stderr`` hover text."""
    return go.Scatter(
        x=list(x),
        y=list(y),
        error_y={"type": "data", "array": list(se), "visible": True},
        mode=mode,
        name=name,
        text=hover_stderr(y, se, unit),
        hovertemplate="%{x}: %{text}<extra>" + name + "</extra>",
        **kwargs,
    )


def stderr_bar(
    fig: go.Figure,
    x: Sequence[Any],
    y: Sequence[float],
    se: Sequence[float],
    *,
    name: str,
    unit: str = "",
    **kwargs: Any,
) -> go.Figure:
    """Add :func:`error_trace` to ``fig`` and return it."""
    fig.add_trace(error_trace(x, y, se, name=name, unit=unit, **kwargs))
    return fig


# --------------------------------------------------------------------------------------------
# exports
# --------------------------------------------------------------------------------------------


def excel_bytes(frames: dict[str, pd.DataFrame] | pd.DataFrame, name: str = "table") -> bytes:
    """An ``.xlsx`` in memory (openpyxl), one sheet per frame."""
    sheets = {name: frames} if isinstance(frames, pd.DataFrame) else frames
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for sname, df in sheets.items():
            out = df if not isinstance(df.index, pd.MultiIndex) else df.reset_index()
            out.to_excel(
                xw,
                sheet_name=sheet_name(sname),
                index=isinstance(df.index, pd.MultiIndex) or df.index.name is not None,
            )
    return buf.getvalue()


def figure_bytes(fig: go.Figure, fmt: str) -> bytes:
    """``fig.to_image`` (kaleido) for ``png`` / ``svg``; ``html`` is the fallback that needs no
    image engine.  Raises what the engine raises (callers catch and fall back)."""
    if fmt == "html":
        html: str = fig.to_html(include_plotlyjs="cdn")
        return html.encode()
    data: bytes = fig.to_image(format=fmt, scale=PNG_SCALE if fmt == "png" else None)
    return data


def image_engine_ok() -> bool:
    """Whether ``fig.to_image`` works in this process (cached: one tiny render)."""
    return _image_engine_ok()


@st.cache_resource(show_spinner=False)
def _image_engine_ok() -> bool:
    try:
        figure_bytes(go.Figure(data=[go.Scatter(x=[0, 1], y=[0, 1])]), "svg")
    except Exception:  # kaleido / chromium failures of any kind
        return False
    return True


def table_with_export(
    df: pd.DataFrame,
    name: str,
    *,
    caption: str | None = None,
    hide_index: bool = True,
    height: int | None = None,
    key: str | None = None,
) -> None:
    """``st.dataframe(df)`` with an Excel download button (built when clicked).  Without an
    explicit ``key`` the button's key is ``xlsx_<safe name>`` made unique within the script run
    by :func:`default_key`, so two tables of the same name on one page do not collide."""
    if caption:
        st.caption(caption)
    if height is None:
        st.dataframe(df, hide_index=hide_index, width="stretch")
    else:
        st.dataframe(df, hide_index=hide_index, height=height, width="stretch")
    st.download_button(
        f"Excel: {name}",
        data=lambda: excel_bytes(df, name),
        file_name=f"{safe_filename(name)}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key=key or default_key(f"xlsx_{safe_filename(name)}"),
        help="Download this table as .xlsx (openpyxl)",
    )


def figure_with_download(fig: go.Figure, name: str, *, key: str | None = None) -> None:
    """``st.plotly_chart(fig)`` with PNG and SVG download buttons; an HTML download with a
    notice when the image engine (kaleido) is unavailable.  Without an explicit ``key`` the
    chart and its buttons are keyed on the display name made unique within the script run
    (:func:`default_key`), so two figures of the same name on one page do not collide."""
    base = safe_filename(name)  # the download file name: the display name, never the widget key
    wkey = key or default_key(f"fig_{base}")
    st.plotly_chart(fig, width="stretch", key=wkey)
    if image_engine_ok():
        cols = st.columns(len(IMAGE_FORMATS))
        for col, (fmt, mime) in zip(cols, IMAGE_FORMATS):
            with col:
                st.download_button(
                    f"{fmt.upper()}: {name}",
                    data=_deferred_image(fig, fmt),
                    file_name=f"{base}.{fmt}",
                    mime=mime,
                    key=f"{wkey}_{fmt}",
                )
    else:
        st.info(
            "Image export engine (kaleido) unavailable in this process — HTML download instead."
        )
        st.download_button(
            f"HTML: {name}",
            data=_deferred_image(fig, "html"),
            file_name=f"{base}.html",
            mime="text/html",
            key=f"{wkey}_html",
        )


def _deferred_image(fig: go.Figure, fmt: str) -> Callable[[], bytes]:
    def _make() -> bytes:
        try:
            return figure_bytes(fig, fmt)
        except Exception:  # engine failure at click time: hand back HTML
            return figure_bytes(fig, "html")

    return _make


# --------------------------------------------------------------------------------------------
# notices and header
# --------------------------------------------------------------------------------------------


def missing_point_notice(exc: MissingArtefact) -> None:
    """What a page shows instead of computing: what is missing and the command that produces
    it (``volsto-precompute ...`` for a point, the study script for an M7 / M8b file)."""
    what = exc.what
    if isinstance(exc, MissingPoint):
        st.warning(f"Not in the store / cache: {what}. The viewer never computes; run:")
    else:
        st.warning(f"Not available: {what}. The viewer never computes; run:")
    st.code(exc.command, language="bash")


def page_header(title: str, cfg: ViewerConfig, *, subtitle: str | None = None) -> None:
    """Title + the provenance line: roots, code tag, git commit, grid, particle count, points
    and runs of the store (an empty store says so and shows the precompute command)."""
    st.title(title)
    if subtitle:
        st.caption(subtitle)
    s = store_summary(cfg)
    st.caption(
        f"store `{cfg.store_root}` · cache `{cfg.cache_root}` · outputs `{cfg.outputs_root}` · "
        f"grid `{cfg.grid_path}`"
    )
    if s.n_points == 0:
        st.info(
            "The results store is empty: nothing to show yet. Read-only viewer — produce the "
            "store with the command below (the precompute is the one place that calibrates)."
        )
        from volsto.viewers.api import precompute_command

        st.code(precompute_command(cfg), language="bash")
        return
    parts = [f"{s.n_points} points", f"{s.n_runs} runs"]
    if s.grid_name:
        parts.append(f"grid '{s.grid_name}'")
    if s.n_particles:
        parts.append(f"{s.n_particles} particles")
    if s.code_tag:
        parts.append(f"code tag {s.code_tag}")
    if s.git_commit:
        parts.append(f"commit {s.git_commit}")
    if s.last_run_utc:
        parts.append(f"last run {s.last_run_utc}")
    st.caption(" · ".join(parts))


__all__ = [
    "IMAGE_FORMATS",
    "PNG_SCALE",
    "default_key",
    "error_trace",
    "excel_bytes",
    "figure_bytes",
    "figure_with_download",
    "hover_stderr",
    "image_engine_ok",
    "missing_point_notice",
    "page_header",
    "safe_filename",
    "sheet_name",
    "snap_slider",
    "stderr_bar",
    "table_with_export",
]
