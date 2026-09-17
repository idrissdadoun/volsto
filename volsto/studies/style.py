"""House style of the study figures (SPEC §10.1, M10 Part 1): one matplotlib style for every
study, applied by :func:`apply`, used through :func:`new_figure` / :func:`save_figure`, plus the
one plotting helper for Monte Carlo numbers, :func:`mc_errorbar`.

matplotlib is imported **lazily** (inside the functions, on the ``Agg`` backend), so importing
this module — and :mod:`volsto.studies.runner`, whose ``list`` / ``rerun`` never draw — does not
require the ``studies`` extra (``pip install volsto[studies]``).

Choices:

* :data:`PALETTE` — the validated categorical order of the project's data-viz reference (eight
  hues, adjacent pairs clear the colour-vision-deficiency separation target; checked with the
  reference validator, light mode).  Series take the hues **in this order, never cycled**: a
  ninth series is an error (:func:`series_color`); a scatter with more than
  :data:`SCATTER_SERIES_CAP` series should be faceted.  Three of the hues sit below 3:1 contrast
  on white, which the reports relieve with the table view (every study figure is drawn from
  numbers its tables also print) and with a distinct marker per series (:data:`MARKERS`).
* Thin marks (1.5 pt lines, 5 pt markers), a recessive grid, text in ink colours
  (:data:`INK`), never in a series colour; a legend whenever there are two series or more.
* Output: ``figures/<name>.pdf`` and ``.png`` (:data:`PNG_DPI`), written atomically with
  fixed metadata (no creation date), so re-rendering the same results gives the same files.
* :func:`mc_errorbar` draws a Monte Carlo series **only** with its error bars: a non-exact
  point without a finite stderr raises (the invariant already holds in
  :class:`~volsto.studies.results.Results`; this is the figure-side check), and exact points
  are drawn without bars.

Checked by ``tests/test_study_runner.py`` (the example study's figure, ``test_render_rebuilds``,
``test_mc_errorbar_refuses_missing_stderr``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from volsto.calibration.cache import atomic_write
from volsto.studies.results import FigureSpec

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

__all__ = [
    "INK",
    "MARKERS",
    "PALETTE",
    "FigureSpec",
    "apply",
    "mc_errorbar",
    "new_figure",
    "save_figure",
    "series_color",
]

#: Categorical hues in their fixed order (blue, orange, aqua, yellow, magenta, green, violet,
#: red) — the validated reference order (module docstring).
PALETTE: tuple[str, ...] = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#4a3aa7",
    "#e34948",
)
#: Series markers, one per palette slot (the secondary encoding).
MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X", "*")
#: Scatter / small-multiple series cap of the palette (the first three hues validate all-pairs).
SCATTER_SERIES_CAP = 3
#: Ink colours: primary text, secondary text (ticks, legend), grid and spines.
INK: dict[str, str] = {
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "grid": "#e4e3df",
    "spine": "#9a9893",
    "surface": "#ffffff",
}
#: Default figure size (inches): a single panel at about 0.85 of an A4 text width.
FIGSIZE: tuple[float, float] = (6.4, 4.0)
#: PNG resolution.
PNG_DPI = 150
#: The rcParams of the house style (:func:`apply` adds the colour cycle from :data:`PALETTE`).
RC: dict[str, Any] = {
    "figure.figsize": FIGSIZE,
    "figure.dpi": 100,
    "figure.facecolor": INK["surface"],
    "savefig.facecolor": INK["surface"],
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "axes.facecolor": INK["surface"],
    "axes.edgecolor": INK["spine"],
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK["primary"],
    "axes.titlesize": 11,
    "axes.titleweight": "regular",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": INK["grid"],
    "grid.linewidth": 0.6,
    "xtick.color": INK["secondary"],
    "ytick.color": INK["secondary"],
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "text.color": INK["primary"],
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.labelcolor": INK["secondary"],
    "lines.linewidth": 1.5,
    "lines.markersize": 5,
    "errorbar.capsize": 2.5,
    "pdf.fonttype": 42,
    "svg.hashsalt": "volsto",
}


def apply() -> None:
    """Select the ``Agg`` backend and install :data:`RC` (idempotent)."""
    import matplotlib

    matplotlib.use("Agg", force=True)
    from cycler import cycler

    rc = dict(RC)
    rc["axes.prop_cycle"] = cycler(color=list(PALETTE))
    cast(Any, matplotlib.rcParams).update(rc)


def series_color(i: int) -> str:
    """The ``i``-th series colour (0-based); more than :data:`PALETTE` series is an error (fold
    them or facet instead of generating hues)."""
    if not 0 <= i < len(PALETTE):
        raise ValueError(
            f"series {i}: the palette has {len(PALETTE)} hues; fold the rest into 'other' or facet"
        )
    return PALETTE[i]


def new_figure(
    nrows: int = 1,
    ncols: int = 1,
    *,
    width: float | None = None,
    height: float | None = None,
    sharex: bool = False,
    sharey: bool = False,
) -> tuple[Figure, Any]:
    """A figure in the house style: ``(fig, ax)`` for one panel, ``(fig, axes)`` (a numpy array
    of ``Axes``) for a grid.  The default size is :data:`FIGSIZE` per panel row, scaled for
    columns."""
    apply()
    from matplotlib.figure import Figure

    w = width if width is not None else FIGSIZE[0] if ncols == 1 else 3.4 * ncols
    h = height if height is not None else FIGSIZE[1] if nrows == 1 else 3.0 * nrows
    fig = Figure(figsize=(w, h), layout="constrained")
    axes = fig.subplots(nrows, ncols, sharex=sharex, sharey=sharey, squeeze=True)
    return fig, axes


def save_figure(fig: Figure, directory: str | Path, name: str) -> list[Path]:
    """Write ``<directory>/<name>.pdf`` and ``.png`` atomically with fixed metadata; returns the
    two paths.  The figure is not reused afterwards (a ``Figure`` without pyplot needs no
    closing)."""
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    pdf, png = d / f"{name}.pdf", d / f"{name}.png"
    atomic_write(
        pdf,
        lambda p: fig.savefig(
            p, format="pdf", metadata={"CreationDate": None, "ModDate": None, "Creator": "volsto"}
        ),
    )
    atomic_write(
        png,
        lambda p: fig.savefig(p, format="png", dpi=PNG_DPI, metadata={"Software": "volsto"}),
    )
    return [pdf, png]


def mc_errorbar(
    ax: Axes,
    x: Sequence[float] | np.ndarray,
    value: Sequence[float] | np.ndarray,
    stderr: Sequence[float] | np.ndarray,
    *,
    exact: Sequence[bool] | np.ndarray | None = None,
    series: int = 0,
    label: str | None = None,
    nse: float = 1.0,
    line: bool = True,
) -> None:
    """Plot one series as markers (joined by a line when ``line``) with ``± nse·stderr`` error
    bars on its Monte Carlo points; exact points (``exact[i]``) get no bar.  NaN values are
    skipped; a Monte Carlo point with a finite value and a non-finite or negative stderr raises
    ``ValueError``."""
    xs = np.asarray(x, dtype=np.float64)
    ys = np.asarray(value, dtype=np.float64)
    es = np.asarray(stderr, dtype=np.float64)
    ex = np.zeros(len(xs), dtype=bool) if exact is None else np.asarray(exact, dtype=bool)
    if not (len(xs) == len(ys) == len(es) == len(ex)):
        raise ValueError("x, value, stderr and exact must have the same length")
    keep = np.isfinite(ys)
    bad = keep & ~ex & ~(np.isfinite(es) & (es >= 0))
    if bad.any():
        raise ValueError(
            f"{int(bad.sum())} Monte Carlo point(s) without a finite stderr at x = "
            f"{xs[bad].tolist()}: not drawn without their error bars"
        )
    color, marker = series_color(series), MARKERS[series]
    order = np.argsort(xs[keep], kind="stable")
    xk, yk = xs[keep][order], ys[keep][order]
    ek = np.where(ex[keep], 0.0, es[keep])[order]
    if line and len(xk) > 1:
        ax.plot(xk, yk, color=color, linewidth=1.5, zorder=2)
    ax.errorbar(
        xk,
        yk,
        yerr=nse * ek if np.any(ek > 0) else None,
        fmt=marker,
        color=color,
        markersize=5,
        markeredgecolor=INK["surface"],
        markeredgewidth=0.8,
        elinewidth=1.0,
        label=label,
        zorder=3,
    )
    if math.isclose(nse, 1.0):
        return
    ax.text(
        0.99,
        0.01,
        f"error bars ±{nse:g} stderr",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK["secondary"],
    )
