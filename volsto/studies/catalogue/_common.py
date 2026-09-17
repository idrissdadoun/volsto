"""Helpers shared by the catalogue studies S1–S4 (SPEC §10.2, owner's M10 Part 2): the model
selection over a precompute grid, the row labels and coordinates, loading a model **without
calibrating** (the local vol is a Dupire construction from the grid's surface, an LSV is read
from the leverage cache through :meth:`volsto.studies.runner.StudyContext.leverage`), the pricing
settings, the requirement records, the store readers and the figure helpers.

**Model selection** (the ``models`` params entry of S1–S4, validated by
:func:`validate_selection`; every key required, no defaults)::

    models:
      surface: placeholder          # a surface of the configured grid
      lv: true                      # the surface's pure local-vol point (ω = 0)
      one_factor: {nu: all, rho: [-0.7], kappa: [1.5]}   # "all" or a list per axis; null = none
      two_factor: all               # "all", a list of preset names, or null

The 1F axis ``nu`` is the grid's (``ω = 2ν``); ``two_factor`` names the presets of the grid
(``"Table 8.2"``).  :func:`select_models` returns the grid's points in grid order (LV, 1F by
``(ν, ρ, κ)``, 2F by preset order).

**Row labels** (:func:`model_label`): ``LV (ω=0)``, ``1F ω=<2ν> ρ=<ρ> κ=<κ>``,
``2F <preset name>``.  :func:`baseline_name` maps a point onto the model names of
``tests/test_m4_regression.py`` (``LV (ω=0)``, ``1F ω=1``, …, ``2F Table 8.2``).

**Where a number comes from** (the results' ``source``): ``store:<point id>`` for a number read
from the M9 results store, ``cache:<key>`` for a number priced here from a cached leverage,
``computed`` for a number priced here under the local vol (a Dupire build of the grid's surface,
no cache entry) or derived from other results.

**Differences between models share random numbers.**  The Gaussian draws are addressed by
``(seed, path, step, Brownian)`` (:mod:`volsto.engine.rng`), so two models priced with the same
seed are *not* independent: their estimates are correlated, with a sign and a size that depend on
the product and on the two grids, and the two standard errors added in quadrature are not the
error of their difference.  Hence two rules:

* **priced here** — :func:`price_paired` prices every model of a study on the same step
  schedule: each grid carries the leverage-slice times of the study's reference LSV (for the
  LSV models, their own; the local vol has none and takes them — so the local vol is stepped on
  a finer grid than it would be alone, and its numbers are not the store's local-vol numbers).
  Every statistic is an :class:`Estimate` (value plus its per-antithetic-pair influence
  function), and :func:`paired_difference` gives the stderr of a difference from the per-pair
  differences of the influences — valid whatever the grids, because the pair samples are
  independent across the path index (the grid only changes how strongly they are correlated);
  the quadrature error, the measured correlation and the grid identity are recorded beside it;
* **read from the store** — the store keeps no per-path samples, so a difference of two stored
  values carries the quadrature error, labelled :data:`QUADRATURE_NOTE`: it is not the exact
  error, and it bounds the exact error only if the two estimates are non-negatively correlated,
  which no study here measures for stored values.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.config import ConfigError, SimConfig
from volsto.studies.runner import Requirement, StudyContext
from volsto.viewers.grid import GridPoint

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from volsto.engine.grid import TimeGrid
    from volsto.engine.mc import PriceResult
    from volsto.models.base import Model
    from volsto.products.base import Product
    from volsto.studies.results import Results

FloatArray = NDArray[np.float64]

__all__ = [
    "LV_GRID_NOTE",
    "LV_LABEL",
    "PAIRED_NOTE",
    "QUADRATURE_NOTE",
    "VP",
    "Difference",
    "Estimate",
    "ModelPoint",
    "baseline_name",
    "cache_source",
    "leverage_requirements",
    "load_model",
    "lv_grid_sentence",
    "model_axes",
    "model_label",
    "model_source",
    "own_steps",
    "paired_difference",
    "paired_grid",
    "point_requirements",
    "price_paired",
    "pricing_sim",
    "quadrature_comparison",
    "rss",
    "select_models",
    "store_source",
    "validate_selection",
]

#: Vol points per unit of volatility (tables and figures show vols in vol points).
VP = 100.0
#: Row label of a surface's pure local-vol point.
LV_LABEL = "LV (ω=0)"
#: Keys of the ``models`` params entry.
SELECTION_KEYS: tuple[str, ...] = ("surface", "lv", "one_factor", "two_factor")
ONE_FACTOR_AXES: tuple[str, ...] = ("nu", "rho", "kappa")
ALL = "all"
#: Note of a difference of two stored estimates (module docstring).
QUADRATURE_NOTE = (
    "stderr: the two errors in quadrature - not the exact error: the estimates share random "
    "numbers (same seed, draws addressed by path and step) and their correlation is not "
    "measured here; the quadrature error bounds the exact one only if it is non-negative"
)
#: Note of a local-vol number priced on the reference LSV's grid (module docstring).
LV_GRID_NOTE = (
    "the local vol is stepped on the reference LSV's grid (its leverage-slice times added), not "
    "on the grid it would use alone"
)
#: Note of a difference priced here (module docstring).
PAIRED_NOTE = "paired stderr: per-path differences on the shared random numbers"


# --------------------------------------------------------------------------------------------
# selection
# --------------------------------------------------------------------------------------------


def _axis_ok(value: Any) -> bool:
    if value == ALL:
        return True
    return (
        isinstance(value, list)
        and len(value) > 0
        and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value)
    )


def validate_selection(sel: Any, where: str = "models") -> None:
    """The ``models`` params entry (module docstring); a :class:`ConfigError` names the fault."""
    if not isinstance(sel, Mapping):
        raise ConfigError(f"{where}: expected a mapping with keys {list(SELECTION_KEYS)}")
    missing = [k for k in SELECTION_KEYS if k not in sel]
    unknown = sorted(set(sel) - set(SELECTION_KEYS))
    if missing or unknown:
        raise ConfigError(f"{where}: missing keys {missing}, unknown keys {unknown}")
    if not isinstance(sel["surface"], str) or not sel["surface"]:
        raise ConfigError(f"{where}.surface: expected a surface name")
    if not isinstance(sel["lv"], bool):
        raise ConfigError(f"{where}.lv: expected true or false")
    of = sel["one_factor"]
    if of is not None:
        if not isinstance(of, Mapping) or set(of) != set(ONE_FACTOR_AXES):
            raise ConfigError(
                f"{where}.one_factor: expected null or a mapping with exactly the keys "
                f"{list(ONE_FACTOR_AXES)}"
            )
        for axis in ONE_FACTOR_AXES:
            if not _axis_ok(of[axis]):
                raise ConfigError(
                    f"{where}.one_factor.{axis}: expected {ALL!r} or a non-empty list of numbers"
                )
    tf = sel["two_factor"]
    if (
        tf is not None
        and tf != ALL
        and (not isinstance(tf, list) or not tf or not all(isinstance(n, str) for n in tf))
    ):
        raise ConfigError(f"{where}.two_factor: expected null, {ALL!r} or preset names")


def _axis_match(value: float, wanted: Any) -> bool:
    return wanted == ALL or any(math.isclose(value, float(w), abs_tol=1e-12) for w in wanted)


@dataclass(frozen=True)
class ModelPoint:
    """A selected grid point with its table label and coordinates."""

    point: GridPoint
    label: str
    axes: dict[str, Any]

    @property
    def id(self) -> str:
        return self.point.id

    @property
    def mode(self) -> str:
        return self.point.mode

    @property
    def is_lv(self) -> bool:
        return self.point.mode == "lv"


def preset_name(point: GridPoint) -> str:
    """The 2F preset name of a two-factor point (its label without the surface prefix)."""
    prefix = f"{point.surface} 2F "
    return point.label[len(prefix) :] if point.label.startswith(prefix) else point.label


def model_label(point: GridPoint) -> str:
    """The table row label of a grid point (module docstring)."""
    if point.mode == "lv":
        return LV_LABEL
    m = point.spec.model
    if point.mode == "one_factor":
        return f"1F ω={m.omega:g} ρ={m.rho_SX1:g} κ={m.k1:g}"  # noqa: RUF001
    if point.mode == "two_factor":
        return f"2F {preset_name(point)}"
    return point.label


def model_axes(point: GridPoint) -> dict[str, Any]:
    """The coordinates of a model row (JSON scalars): ``mode``, and ``nu`` / ``omega`` /
    ``rho`` / ``kappa`` (1F; the LV point sits at ``nu = omega = 0``) or the seven 2F
    parameters and ``preset``."""
    if point.mode == "lv":
        return {"mode": "lv", "nu": 0.0, "omega": 0.0}
    m = point.spec.model
    if point.mode == "one_factor":
        return {
            "mode": "one_factor",
            "nu": float(m.nu),
            "omega": float(m.omega),
            "rho": float(m.rho_SX1),
            "kappa": float(m.k1),
        }
    out: dict[str, Any] = {"mode": point.mode, "preset": preset_name(point)}
    for k in ("nu", "theta", "k1", "k2", "rho12", "rho_SX1", "rho_SX2"):
        out[k] = float(getattr(m, k))
    return out


def select_models(ctx: StudyContext, sel: Mapping[str, Any]) -> list[ModelPoint]:
    """The grid points the ``models`` entry selects (module docstring), in grid order; an empty
    selection or an unknown surface / preset is a :class:`ConfigError`."""
    points = [p for p in ctx.grid_points() if p.surface == sel["surface"]]
    if not points:
        raise ConfigError(f"models.surface {sel['surface']!r} has no point in the grid")
    out: list[ModelPoint] = []
    of = sel["one_factor"]
    tf = sel["two_factor"]
    presets_seen: set[str] = set()
    for p in points:
        if p.mode == "lv":
            keep = bool(sel["lv"])
        elif p.mode == "one_factor":
            m = p.spec.model
            keep = of is not None and all(
                _axis_match(v, of[a])
                for a, v in (("nu", m.nu), ("rho", m.rho_SX1), ("kappa", m.k1))
            )
        elif p.mode == "two_factor":
            name = preset_name(p)
            presets_seen.add(name)
            keep = tf is not None and (tf == ALL or name in tf)
        else:
            keep = False
        if keep:
            out.append(ModelPoint(p, model_label(p), model_axes(p)))
    if isinstance(tf, list):
        unknown = sorted(set(tf) - presets_seen)
        if unknown:
            raise ConfigError(f"models.two_factor: presets {unknown} are not in the grid")
    if not out:
        raise ConfigError("models: the selection is empty on this grid")
    labels = [m.label for m in out]
    if len(set(labels)) != len(labels):
        raise ConfigError(f"models: duplicate row labels {labels}")
    return out


def baseline_name(mp: ModelPoint) -> str | None:
    """The ``tests/test_m4_regression.py`` model name of a point, or ``None`` (the baselines
    hold LV, 1F ω ∈ {1, 2, 3} at ρ −0.7, κ 1.5, and the Table 8.2 preset)."""
    if mp.is_lv:
        return LV_LABEL
    m = mp.point.spec.model
    if mp.mode == "one_factor":
        if math.isclose(m.rho_SX1, -0.7) and math.isclose(m.k1, 1.5):
            for w in (1.0, 2.0, 3.0):
                if math.isclose(m.omega, w):
                    return f"1F ω={w:g}"
        return None
    if mp.mode == "two_factor" and preset_name(mp.point) == "Table 8.2":
        return "2F Table 8.2"
    return None


# --------------------------------------------------------------------------------------------
# models, pricing, sources, requirements
# --------------------------------------------------------------------------------------------


def pricing_sim(ctx: StudyContext, mp: ModelPoint, n_paths: int) -> SimConfig:
    """The pricing settings: the point's calibration schedule and scheme (shared with the
    calibration, SPEC §3.1) with ``n_paths`` and the config's ``pricing`` seed."""
    return dataclasses.replace(mp.point.spec.sim, n_paths=int(n_paths), seed=ctx.seed("pricing"))


def load_model(ctx: StudyContext, mp: ModelPoint) -> Model:
    """The priced model of a point: the Dupire local vol of the point's surface (a construction,
    no calibration and no cache entry) or the cached LSV (``ctx.leverage``: a miss raises
    :class:`~volsto.studies.runner.MissingRequirements` with the precompute line)."""
    if mp.is_lv:
        from volsto.viewers.precompute import build_lv

        return build_lv(mp.point)
    return ctx.leverage(mp.point.spec, mp.label)


def model_source(mp: ModelPoint) -> str:
    """``computed`` for the local vol, ``cache:<key>`` for an LSV (module docstring)."""
    return "computed" if mp.is_lv else cache_source(mp.point.cache_key or mp.id)


def cache_source(key: str) -> str:
    return f"cache:{key}"


def store_source(point_id: str) -> str:
    return f"store:{point_id}"


def leverage_requirements(ctx: StudyContext, models: Sequence[ModelPoint]) -> list[Requirement]:
    """One ``leverage`` requirement per LSV point (the LV point needs none)."""
    return [
        ctx.leverage_requirement(mp.point.spec, f"leverage of {mp.label}")
        for mp in models
        if not mp.is_lv
    ]


def point_requirements(ctx: StudyContext, models: Sequence[ModelPoint]) -> list[Requirement]:
    """One ``point`` requirement per model (the stored point)."""
    return [ctx.point_requirement(mp.id, f"store point {mp.label}") for mp in models]


def stored_ids(ctx: StudyContext) -> set[str]:
    """The point ids the configured store holds (no point recorded as read)."""
    from volsto.viewers.store import StoreReader

    df = StoreReader(ctx.store_root).points()
    return set() if df.empty else {str(i) for i in df["point_id"]}


# --------------------------------------------------------------------------------------------
# paired estimates
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Estimate:
    """A Monte Carlo statistic: ``value`` and its influence function per antithetic pair
    (mean zero, in the value's units), so ``stderr = std(influence) / sqrt(n)`` and a difference
    with another estimate on the same path index is paired (:func:`paired_difference`)."""

    value: float
    influence: FloatArray

    @property
    def stderr(self) -> float:
        n = self.influence.size
        if n < 2 or not math.isfinite(self.value):
            return math.nan
        return float(self.influence.std(ddof=1) / math.sqrt(n))

    @property
    def pair(self) -> tuple[float, float]:
        return self.value, self.stderr

    def scaled(self, c: float) -> Estimate:
        return Estimate(c * self.value, c * self.influence)

    def minus(self, other: Estimate) -> Estimate:
        """The difference of two statistics of the **same** path set."""
        return Estimate(self.value - other.value, self.influence - other.influence)


def mean_estimate(x: FloatArray) -> Estimate:
    """The mean of per-pair samples."""
    m = float(x.mean())
    return Estimate(m, x - m)


def ratio_estimate(a: FloatArray, d: FloatArray) -> Estimate:
    """``mean(a) / mean(d)`` with the delta-method influence ``(a − R d) / mean(d)``
    (:func:`volsto.analytics.conditional_variance.ratio_of_means`)."""
    dbar = float(d.mean())
    if dbar <= 0:
        return Estimate(math.nan, np.full(a.size, math.nan))
    r = float(a.mean()) / dbar
    return Estimate(r, (a - r * d) / dbar)


def sqrt_estimate(e: Estimate) -> Estimate:
    """``sqrt`` of a variance estimate (a vol), influence ``δv / (2 sqrt v)``."""
    if not (math.isfinite(e.value) and e.value > 0):
        return Estimate(math.nan, np.full(e.influence.size, math.nan))
    k = math.sqrt(e.value)
    return Estimate(k, e.influence / (2.0 * k))


@dataclass(frozen=True)
class Difference:
    """``a − b`` of two estimates on the same path index: the paired stderr, the quadrature
    error for comparison, the per-pair correlation and whether the two grids coincided."""

    value: float
    stderr: float
    stderr_quadrature: float
    correlation: float
    same_grid: bool

    @property
    def z(self) -> float:
        if not (math.isfinite(self.value) and math.isfinite(self.stderr)) or self.stderr <= 0:
            return math.nan
        return self.value / self.stderr


def paired_difference(a: Estimate, b: Estimate, *, same_grid: bool) -> Difference:
    """``a − b`` with the stderr of the per-pair influence differences (module docstring)."""
    n = a.influence.size
    if n != b.influence.size:
        raise ValueError("paired estimates need the same number of path pairs")
    value = a.value - b.value
    if not (math.isfinite(a.value) and math.isfinite(b.value)) or n < 2:
        return Difference(math.nan, math.nan, math.nan, math.nan, same_grid)
    diff = a.influence - b.influence
    se = float(diff.std(ddof=1) / math.sqrt(n))
    quad = float(math.hypot(a.stderr, b.stderr))
    sa, sb = float(a.influence.std()), float(b.influence.std())
    corr = float(np.mean(a.influence * b.influence) / (sa * sb)) if sa > 0 and sb > 0 else math.nan
    return Difference(value, se, quad, corr, same_grid)


def paired_grid(
    model: Model,
    sim: SimConfig,
    products: Sequence[Product],
    reference_times: FloatArray,
) -> TimeGrid:
    """The grid of :func:`price_paired`: the products' fixings, the step schedule, the model's
    own required times and ``reference_times``."""
    from volsto.engine.grid import TimeGrid

    fixings = np.unique(np.concatenate([np.asarray(p.fixing_times, float) for p in products]))
    record_all = sim.record_all_steps or any(p.requires_all_steps for p in products)
    calib = np.unique(np.concatenate([np.asarray(model.required_times(), float), reference_times]))
    return TimeGrid.build(fixings, sim.dt_max, calibration_grid=calib, record_all_steps=record_all)


def lv_grid_sentence(results: Results, table: str = "pairing") -> str:
    """The disclosure, read from ``table`` (rows = model labels, columns ``steps`` and
    ``own_steps``), that the local vol is stepped on the reference LSV's grid; empty when the
    local vol is absent or steps on its own grid."""
    if table not in results.tables() or LV_LABEL not in results.rows(table):
        return ""
    steps = results.value(table, LV_LABEL, "steps")[0]
    own = results.value(table, LV_LABEL, "own_steps")[0]
    if steps == own:
        return ""
    return (
        f"the local vol is stepped on the reference LSV's grid ({steps:.0f} steps to the "
        f"maturity against the {own:.0f} it would use alone, the LSV's leverage-slice times "
        "added), so its numbers are not the store's local-vol computation"
    )


def own_steps(model: Model, sim: SimConfig, products: Sequence[Product]) -> int:
    """The steps of the grid the model would use alone for ``products`` (no reference times)."""
    return int(paired_grid(model, sim, products, np.empty(0)).n_steps)


def price_paired(
    model: Model,
    sim: SimConfig,
    products: Sequence[Product],
    reference_times: FloatArray,
    *,
    grid_products: Sequence[Product] | None = None,
) -> list[PriceResult]:
    """``products`` priced with their per-path payoffs on :func:`paired_grid` (the model's own
    required times and ``reference_times``, the reference LSV's leverage slices), so models
    sharing those slices — and the local vol, which has none — step on the same times with the
    same normals (module docstring).  ``grid_products`` (default ``products``) fix the grid's
    fixings, e.g. both headline notes when one of them is priced."""
    from volsto.engine.mc import MonteCarlo

    mc = MonteCarlo(sim)
    on = list(grid_products if grid_products is not None else products)
    grid = paired_grid(model, sim, on, reference_times)
    return mc.price_many(
        list(products), model, grid=grid, draws=mc.draws_for(grid, model), keep_payoffs=True
    )


def same_times(model: Model, reference_times: FloatArray) -> bool:
    """Whether the model's required times are ``reference_times`` (or it has none)."""
    own = np.asarray(model.required_times(), float)
    if own.size == 0:
        return True
    return own.shape == reference_times.shape and bool(
        np.allclose(own, reference_times, rtol=0.0, atol=1e-12)
    )


def pair_samples(result: PriceResult, antithetic: bool) -> FloatArray:
    """The per-antithetic-pair samples of a priced product."""
    from volsto.analytics.conditional_variance import pair_average

    return pair_average(np.asarray(result.payoffs, dtype=np.float64), antithetic)


# --------------------------------------------------------------------------------------------
# numbers and commands
# --------------------------------------------------------------------------------------------


def quadrature_comparison(paired: FloatArray, quadrature: FloatArray, corr: FloatArray) -> str:
    """A sentence comparing the quadrature errors of some differences with their paired stderrs,
    from the run's own numbers (empty when none is finite)."""
    ok = np.isfinite(paired) & np.isfinite(quadrature) & (paired > 0)
    if not ok.any():
        return ""
    ratio = quadrature[ok] / paired[ok]
    c = corr[ok & np.isfinite(corr)]
    larger, smaller = int((ratio > 1).sum()), int((ratio < 1).sum())
    text = (
        f"The quadrature error is {float(ratio.min()):.2f} to {float(ratio.max()):.2f} times the "
        f"paired stderr (median {float(np.median(ratio)):.2f}). It is the larger one in {larger} "
        f"of {ratio.size} differences, which adding the errors in quadrature would have made "
        "look less significant"
    )
    if smaller:
        text += (
            f", and the smaller one in {smaller}, where the two estimates are negatively "
            "correlated and quadrature would have overstated the significance"
        )
    if c.size:
        text += (
            f"; the per-pair correlation runs from {float(c.min()):+.2f} to {float(c.max()):+.2f}"
        )
    return text + "."


def rss(*errors: float) -> float:
    """Root sum of squares: the error of a difference of independent estimates (for correlated
    ones, a bound only when the correlation is non-negative)."""
    return float(math.sqrt(sum(float(e) ** 2 for e in errors)))


def finite(x: float) -> bool:
    """Whether ``x`` is a finite number."""
    return bool(np.isfinite(x))


def axis(frame: pd.DataFrame, name: str) -> pd.Series:
    """The ``axis_<name>`` column of a :meth:`Results.long` frame as floats (NaN where a row
    lacks the coordinate, all NaN when no row has it)."""
    col = f"axis_{name}"
    if col not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[col], errors="coerce").astype(float)


def axis_text(frame: pd.DataFrame, name: str) -> pd.Series:
    """The ``axis_<name>`` column as strings (``""`` where absent)."""
    col = f"axis_{name}"
    if col not in frame.columns:
        return pd.Series("", index=frame.index, dtype=object)
    return frame[col].map(
        lambda v: "" if v is None or (isinstance(v, float) and v != v) else str(v)
    )


#: Width of a command line in a narrative code block (fits the LaTeX text width in the
#: typewriter font, so the document has no overfull line).
COMMAND_WIDTH = 68


def shell_block(command: str, width: int = COMMAND_WIDTH) -> str:
    """``command`` as a fenced markdown code block of lines no longer than ``width``, joined by
    shell line continuations: words are kept whole when they fit, a longer word (an absolute
    path) is cut with a backslash-newline attached to the chunk, which a POSIX shell removes —
    the block pastes back into the same command."""
    room = width - 2  # room for the " \\" continuation
    lines: list[tuple[str, bool]] = []  # (text, ends inside a word)
    cur = ""
    for word in command.split():
        candidate = f"{cur} {word}" if cur else word
        if len(candidate) <= room:
            cur = candidate
            continue
        if cur:
            lines.append((cur, False))
            cur = ""
        while len(word) > room:
            lines.append((word[:room], True))
            word = word[room:]
        cur = word
    if cur:
        lines.append((cur, False))
    body = [f"{t}\\" if cut else f"{t} \\" for t, cut in lines[:-1]]
    body += [t for t, _ in lines[-1:]]
    return "\n".join(["```", *body, "```"])


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------


def series_by(
    ax: Axes,
    x: str,
    groups: Sequence[tuple[str, pd.DataFrame]],
    *,
    scale: float = 1.0,
    max_series: int = 8,
) -> int:
    """Draw each ``(label, sub-frame)`` of ``groups`` as one Monte Carlo series of ``value`` /
    ``stderr`` against the column ``x`` (house style, error bars ±1 stderr); at most
    ``max_series`` series (the palette); returns the number drawn."""
    from volsto.studies import style

    n = 0
    for label, sub in groups[:max_series]:
        sub = sub.sort_values(x)
        style.mc_errorbar(
            ax,
            sub[x].to_numpy(dtype=float),
            scale * sub["value"].to_numpy(dtype=float),
            scale * sub["stderr"].to_numpy(dtype=float),
            exact=sub["exact"].to_numpy(dtype=bool),
            series=n,
            label=label,
        )
        n += 1
    return n


def band(ax: Axes, value: float, stderr: float, label: str, series: int) -> None:
    """A horizontal reference line with a ±1 stderr band (a point without an x coordinate on the
    axis, e.g. a 2F preset on a 1F ν axis)."""
    from volsto.studies import style

    if not (finite(value) and finite(stderr)):
        return
    color = style.series_color(series)
    ax.axhline(value, color=color, linewidth=1.0, linestyle="--", label=label)
    ax.axhspan(value - stderr, value + stderr, color=color, alpha=0.15, linewidth=0)


def empty_panel(ax: Axes, text: str) -> None:
    """A panel that states why it draws nothing."""
    from volsto.studies import style

    ax.text(
        0.5,
        0.5,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        color=style.INK["secondary"],
        wrap=True,
    )
    ax.set_xticks([])
    ax.set_yticks([])
