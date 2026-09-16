"""Grid specification and point enumeration of the viewer precompute (SPEC §9.2, M9 Part 1).

A :class:`GridSpec` is loaded strictly from YAML (``configs/grids/default.yaml``,
``configs/grids/toy.yaml``; :func:`load_grid`) and expanded by :func:`enumerate_points` into a
deterministic, ordered list of :class:`GridPoint` — one per model calibrated to one surface:

* ``lv`` — the pure Dupire local vol of a surface (no leverage, no calibration).  Every surface
  gets exactly one LV point: it is the reference of the model-risk page (LSV − LV) and the
  ``ω = 0`` row of the old headline tables.  **The one-factor axis value ``nu = 0`` is this point**
  (``ω = 2ν = 0`` is the LV limit in the model sense; the particle calibration of a zero
  vol-of-vol kernel would only reproduce the Dupire local vol up to particle noise), so every
  ``(nu = 0, rho, kappa)`` combination of the axes collapses onto the surface's single LV point,
  keyed ``lv:<sha256 of market + surface + local-vol config>`` (:func:`lv_key`);
  :func:`resolve_axis_point` maps a slider position with ``nu = 0`` to it.
* ``one_factor`` — :meth:`~volsto.config.BergomiParams.one_factor` ``(ω = 2ν, κ, ρ)`` over the
  ``nu × rho × kappa`` axes (``nu > 0``), on the surfaces flagged ``one_factor: true``
  (the default grid: the placeholder only; a snapshot opts in through its YAML flag).  The
  degenerate points of the earlier studies (``ω ∈ {1, 2, 3}``, ``ρ = −0.7``, ``κ = 1.5``,
  :data:`DEGENERATE_ONE_FACTOR`) are always appended when the axes miss them.
* ``two_factor`` — named :class:`~volsto.config.BergomiParams` presets (Table 8.2) on the
  surfaces flagged ``two_factor: true``.
* ``marking`` — the P1 marking fits :func:`~volsto.calibration.fit_2f.fit_2f_marking` at
  ``ssr_target × skew_eps`` on the surfaces flagged ``marking: true`` (the SPX snapshots).  The
  fit defines the model parameters, so a marking point is *resolved* by :func:`resolve_marking`
  (about 2 s per fit) only when it is computed; its id is the stable label
  ``marking:<surface>:ssr<s>:eps<e>`` (a cache key would need the fit first), the fitted
  parameters, the status (``interior`` / ``binding`` / ``infeasible``) and the leverage-cache key
  are recorded in the store.  An infeasible fit is recorded as a point with status
  ``infeasible`` and **no** calibration.

Point ids of calibrated points (``one_factor``, ``two_factor``) are the leverage-cache key of
their :class:`~volsto.config.CalibrationSpec` (market, surface, model, particle settings,
schedule and scheme, calibration code tag), so a code-tag bump or a particle-count change is a
new grid.  ``enumerate_points`` sorts by surface (grid order), mode (:data:`MODES` order) and
axis values; :func:`shard` takes the interleaved ``i``-th of ``n`` shards, ``points[i-1::n]``
(the owner's ``--shard i/n``, 1-based).

Checked by ``tests/test_precompute.py`` (``test_default_grid_counts_and_shards``,
``test_toy_precompute_end_to_end``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import volsto
from volsto.calibration.cache import spec_key
from volsto.calibration.fit_2f import BreakEvenFitConfig, FitResult, fit_2f_marking
from volsto.config import (
    BergomiParams,
    CalibrationSpec,
    ConfigError,
    LocalVolConfig,
    MarketConfig,
    SSVIConfig,
    load_yaml,
    to_mapping,
)
from volsto.market.surface import ImpliedSurface

log = logging.getLogger(__name__)

#: Repository root (the grid YAML names surfaces and the reference spec relative to it).
REPO_ROOT = Path(volsto.__file__).resolve().parents[1]
#: Point modes in enumeration order.
MODES: tuple[str, ...] = ("lv", "one_factor", "two_factor", "marking")
SURFACE_KINDS: tuple[str, ...] = ("placeholder", "snapshot")
RISK_TIERS: tuple[str, ...] = ("none", "light", "full")
#: Production particle count (SPEC §11: headline tables, baselines and the viewer precompute
#: use 8·10⁵ particles, single seed).
PRODUCTION_N_PARTICLES = 800_000
#: The 1F degenerate points of the M4 / M6 studies, always included: ω ∈ {1, 2, 3} → ν = ω / 2,
#: ρ = −0.7, κ = 1.5 (``tests/test_m4_regression.py`` baselines).
DEGENERATE_ONE_FACTOR: tuple[tuple[float, float, float], ...] = (
    (0.5, -0.7, 1.5),
    (1.0, -0.7, 1.5),
    (1.5, -0.7, 1.5),
)
#: Bucket count of the M5 forward-variance ladder (:func:`volsto.risk.ladders.default_buckets`:
#: monthly to 1y, quarterly to 3y = 20 buckets) — the default meaning of "fwd-var ladder" in the
#: light risk tier; the other admissible ``risk.fwd_var_buckets`` value is ``len(light_pillars)``,
#: the coarse ladder ``(0, p₁), (p₁, p₂), …`` built from the light pillars (SPEC §9.2, S1 F2).
M5_FWD_VAR_BUCKETS = 20
#: Model parameters carried by the LV point's :class:`CalibrationSpec` (the LV model ignores
#: them; a zero vol of vol so that ``BergomiParams.omega == 0`` reads as "no SV").  Not a
#: calibration input: the LV point never calibrates and is keyed without the model.
LV_PLACEHOLDER_MODEL = BergomiParams.one_factor(0.0, 1.0, 0.0)


# --------------------------------------------------------------------------------------------
# YAML schema
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceSpec:
    """One target surface.  ``kind = "placeholder"`` takes market and SSVI from the reference
    spec; ``kind = "snapshot"`` loads the ``market`` / ``ssvi`` sections of ``path`` (relative to
    the repository root).  The flags say which model families run on it: no defaults — the YAML
    states each (the default grid: 1F axes and the 2F presets on the placeholder, the marking
    presets on the snapshots; the LV point is always built)."""

    name: str
    kind: str
    one_factor: bool
    two_factor: bool
    marking: bool
    path: str | None = None
    local_vol: LocalVolConfig | None = None

    def __post_init__(self) -> None:
        if self.kind not in SURFACE_KINDS:
            raise ValueError(f"surface kind must be one of {SURFACE_KINDS}")
        if self.kind == "snapshot" and not self.path:
            raise ValueError(f"surface {self.name!r}: a snapshot needs a path")
        if self.kind == "placeholder" and self.path:
            raise ValueError(f"surface {self.name!r}: the placeholder takes no path")
        if not self.name or "/" in self.name or ":" in self.name:
            raise ValueError("surface names are non-empty and contain neither '/' nor ':'")


@dataclass(frozen=True)
class OneFactorAxes:
    """``nu × rho × kappa`` (``ω = 2ν``); ``nu = 0`` is the LV point (module docstring)."""

    nu: tuple[float, ...]
    rho: tuple[float, ...]
    kappa: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (self.nu and self.rho and self.kappa):
            raise ValueError("one_factor axes must be non-empty")
        if any(n < 0 for n in self.nu) or any(k <= 0 for k in self.kappa):
            raise ValueError("nu must be non-negative and kappa positive")
        if any(not -1.0 <= r <= 1.0 for r in self.rho):
            raise ValueError("rho must lie in [-1, 1]")

    @property
    def n_combinations(self) -> int:
        return len(self.nu) * len(self.rho) * len(self.kappa)


@dataclass(frozen=True)
class TwoFactorPreset:
    name: str
    model: BergomiParams


@dataclass(frozen=True)
class MarkingAxes:
    """``ssr_target × skew_eps`` of :func:`~volsto.calibration.fit_2f.fit_2f_marking` (the
    fitter's other settings at their documented defaults: k2 = 0.2, ν cap 3.5, two-point skew
    constraint)."""

    ssr_target: tuple[float, ...]
    skew_eps: tuple[float, ...]

    def __post_init__(self) -> None:
        if not (self.ssr_target and self.skew_eps):
            raise ValueError("marking axes must be non-empty")
        if any(s <= 0 for s in self.ssr_target) or any(e < 0 for e in self.skew_eps):
            raise ValueError("ssr_target must be positive and skew_eps non-negative")


@dataclass(frozen=True)
class ExtraPoint:
    """An explicit extra ``(surface, model)`` point (``label`` names it in the store)."""

    label: str
    surface: str
    model: BergomiParams


@dataclass(frozen=True)
class ParticleSettings:
    """Particle count and horizon; the other :class:`~volsto.config.ParticleConfig` settings come
    from the reference spec.  ``n_particles`` defaults to :data:`PRODUCTION_N_PARTICLES`."""

    n_particles: int = PRODUCTION_N_PARTICLES
    horizon: float = 3.0


@dataclass(frozen=True)
class ProductFlags:
    """Which headline sets the store carries: the M4 set (:func:`volsto.studies.m4.run_headline`;
    ``conditional`` adds the M4c conditional-variance / VKO columns), the M6 notes
    (:func:`volsto.studies.m6.run_m6_headline`)."""

    m4: bool = True
    m6: bool = True
    conditional: bool = True
    cliquet_maturities: tuple[float, ...] = (1.0, 2.0)


@dataclass(frozen=True)
class PricingSettings:
    """Pricing paths and seed of every stored number — the M4 / M6 baseline convention
    (``tests/test_m4_regression.py``: 400k paths, seed 2024)."""

    n_paths: int = 400_000
    seed: int = 2024

    def __post_init__(self) -> None:
        if self.n_paths <= 0 or self.n_paths % 2:
            raise ValueError("n_paths must be a positive even number (antithetic pricing)")


@dataclass(frozen=True)
class RiskSettings:
    """Risk tier of the stored :class:`~volsto.risk.report.RiskReport` per point (``none`` /
    ``light`` / ``full``; :mod:`volsto.viewers.precompute` documents each tier's sections and
    calibration count) for ``products`` (``"autocall 3y"``, ``"cliquet 1y"``); ``light_pillars``
    are the skew-ladder pillars of the light tier and ``fwd_var_buckets`` the bucket count of its
    forward-variance ladder: :data:`M5_FWD_VAR_BUCKETS` (20, the M5 ladder — the default because
    "fwd-var ladder" has meant that ladder since M5 and the stored risk tables stay comparable
    with the M5 / M6 reports) or ``len(light_pillars)`` (the coarse ladder on the light pillars,
    17 instead of 34 leverage recalibrations per LSV point).  On an LSV point every bucket is a
    leverage recalibration, so the count is the single largest lever on the precompute budget:
    ``volsto-precompute --dry-run`` projects both."""

    tier: str = "light"
    products: tuple[str, ...] = ("autocall 3y", "cliquet 1y")
    light_pillars: tuple[float, ...] = (0.25, 1.0, 3.0)
    fwd_var_buckets: int = M5_FWD_VAR_BUCKETS

    def __post_init__(self) -> None:
        if self.tier not in RISK_TIERS:
            raise ValueError(f"risk tier must be one of {RISK_TIERS}")
        if not self.light_pillars or any(p <= 0 for p in self.light_pillars):
            raise ValueError("light_pillars must be non-empty and positive")
        if list(self.light_pillars) != sorted(self.light_pillars):
            raise ValueError("light_pillars must be increasing")
        if self.fwd_var_buckets not in (M5_FWD_VAR_BUCKETS, len(self.light_pillars)):
            raise ValueError(
                f"fwd_var_buckets must be {M5_FWD_VAR_BUCKETS} (the M5 ladder) or "
                f"{len(self.light_pillars)} (= len(light_pillars), the coarse ladder), got "
                f"{self.fwd_var_buckets}"
            )

    @property
    def light_buckets(self) -> tuple[tuple[float, float], ...] | None:
        """The light tier's forward-variance buckets: ``None`` for the M5 ladder (the ladder's
        own default, :func:`volsto.risk.ladders.default_buckets`), else ``(0, p₁), (p₁, p₂), …``
        over ``light_pillars``."""
        if self.fwd_var_buckets == M5_FWD_VAR_BUCKETS:
            return None
        edges = (0.0, *self.light_pillars)
        return tuple((edges[i], edges[i + 1]) for i in range(len(edges) - 1))


@dataclass(frozen=True)
class GridSpec:
    """The grid (module docstring).  ``reference_spec`` is the study spec whose particle,
    simulation (calibration schedule and scheme) and local-vol settings every point shares and
    whose market / surface is the placeholder."""

    name: str
    reference_spec: str
    surfaces: tuple[SurfaceSpec, ...]
    particle: ParticleSettings
    products: ProductFlags
    pricing: PricingSettings
    risk: RiskSettings
    one_factor: OneFactorAxes | None = None
    two_factor_presets: tuple[TwoFactorPreset, ...] = ()
    marking: MarkingAxes | None = None
    extra_points: tuple[ExtraPoint, ...] = ()
    include_degenerate: bool = True

    def __post_init__(self) -> None:
        names = [s.name for s in self.surfaces]
        if not names:
            raise ValueError("a grid needs at least one surface")
        if len(set(names)) != len(names):
            raise ValueError("surface names must be unique")
        if len({p.name for p in self.two_factor_presets}) != len(self.two_factor_presets):
            raise ValueError("preset names must be unique")
        for e in self.extra_points:
            if e.surface not in names:
                raise ValueError(f"extra point {e.label!r} names an unknown surface")

    def surface(self, name: str) -> SurfaceSpec:
        for s in self.surfaces:
            if s.name == name:
                return s
        raise KeyError(name)

    def one_factor_combinations(self) -> int:
        """Axis combinations per surface flagged ``one_factor`` (``nu = 0`` included)."""
        return 0 if self.one_factor is None else self.one_factor.n_combinations


def load_grid(path: str | Path) -> GridSpec:
    """Strict YAML → :class:`GridSpec` (unknown / missing keys raise
    :class:`~volsto.config.ConfigError`)."""
    return load_yaml(path, GridSpec)


def grid_mapping(grid: GridSpec) -> dict[str, Any]:
    """The grid as the plain mapping the manifest records."""
    out = to_mapping(grid)
    assert isinstance(out, dict)
    return out


# --------------------------------------------------------------------------------------------
# specs and keys
# --------------------------------------------------------------------------------------------


def reference_spec(grid: GridSpec) -> CalibrationSpec:
    ref = load_yaml(REPO_ROOT / grid.reference_spec, CalibrationSpec)
    particle = dataclasses.replace(
        ref.particle, n_particles=grid.particle.n_particles, horizon=grid.particle.horizon
    )
    return dataclasses.replace(ref, particle=particle)


def surface_spec(grid: GridSpec, surface: SurfaceSpec, ref: CalibrationSpec) -> CalibrationSpec:
    """The reference spec with the surface's market / SSVI / local-vol settings (model: the LV
    placeholder; callers replace it)."""
    spec = dataclasses.replace(ref, model=LV_PLACEHOLDER_MODEL)
    if surface.kind == "snapshot":
        assert surface.path is not None
        path = REPO_ROOT / surface.path
        if not path.exists():
            raise ConfigError(f"surface {surface.name!r}: snapshot {path} does not exist")
        spec = dataclasses.replace(
            spec,
            market=load_yaml(path, MarketConfig, section="market"),
            surface=load_yaml(path, SSVIConfig, section="ssvi"),
        )
    if surface.local_vol is not None:
        spec = dataclasses.replace(spec, local_vol=surface.local_vol)
    return spec


def lv_key(spec: CalibrationSpec) -> str:
    """Id of a surface's LV point: ``lv:`` + SHA-256 of market, surface and local-vol config
    (model, particle and scheme settings play no part in a Dupire local vol)."""
    payload = {
        "market": to_mapping(spec.market),
        "surface": to_mapping(spec.surface),
        "local_vol": to_mapping(spec.local_vol) if spec.local_vol is not None else None,
        "perturbation": (to_mapping(spec.perturbation) if spec.perturbation is not None else None),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return "lv:" + hashlib.sha256(blob).hexdigest()


def marking_id(surface: str, ssr_target: float, skew_eps: float) -> str:
    return f"marking:{surface}:ssr{ssr_target:g}:eps{skew_eps:g}"


# --------------------------------------------------------------------------------------------
# points
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class MarkingSummary:
    """What the store keeps of a marking fit: status, messages, the naked-skew gaps at the
    constraint pillars, the mean skew gap and the per-pillar first-order P1 SSR."""

    status: str
    messages: tuple[str, ...]
    notes: tuple[str, ...]
    skew_gaps: dict[str, float]
    mean_skew_gap: float
    ssr_first_order: dict[float, float]
    wall_seconds: float


@dataclass(frozen=True)
class GridPoint:
    """One store row.  ``spec`` is the calibration spec (the LV placeholder model for ``lv``
    points; for an unresolved marking point the model is the placeholder too and ``resolved`` is
    False); ``axes`` holds the axis values that address the point on the pages; ``cache_key`` is
    the leverage-cache key (None for ``lv`` and unresolved / infeasible marking points)."""

    id: str
    label: str
    surface: str
    mode: str
    spec: CalibrationSpec
    axes: dict[str, float] = field(default_factory=dict)
    cache_key: str | None = None
    status: str = "ok"
    resolved: bool = True
    marking: MarkingSummary | None = None

    @property
    def params(self) -> BergomiParams | None:
        return None if self.mode == "lv" or not self.resolved else self.spec.model

    @property
    def calibrates(self) -> bool:
        """Whether computing the point needs a leverage calibration."""
        return self.mode != "lv" and self.status != "infeasible"


def _one_factor_triples(grid: GridSpec) -> list[tuple[float, float, float]]:
    axes = grid.one_factor
    triples: list[tuple[float, float, float]] = []
    if axes is not None:
        triples = [(n, r, k) for n in axes.nu for r in axes.rho for k in axes.kappa]
    if grid.include_degenerate:
        for t in DEGENERATE_ONE_FACTOR:
            if t not in triples:
                triples.append(t)
    return triples


def one_factor_spec(base: CalibrationSpec, nu: float, rho: float, kappa: float) -> CalibrationSpec:
    return dataclasses.replace(base, model=BergomiParams.one_factor(2.0 * nu, kappa, rho))


def enumerate_points(grid: GridSpec) -> list[GridPoint]:
    """All points of the grid in the canonical order (module docstring); marking points
    unresolved (no fit run here)."""
    ref = reference_spec(grid)
    points: list[GridPoint] = []
    for surface in grid.surfaces:
        base = surface_spec(grid, surface, ref)
        per_surface: list[tuple[tuple[Any, ...], GridPoint]] = []
        per_surface.append(
            (
                (0, ()),
                GridPoint(lv_key(base), f"{surface.name} LV (ω=0)", surface.name, "lv", base),
            )
        )
        if surface.one_factor:
            seen: set[tuple[float, float, float]] = set()
            for nu, rho, kappa in _one_factor_triples(grid):
                if nu == 0.0 or (nu, rho, kappa) in seen:
                    continue  # nu = 0 is the LV point above
                seen.add((nu, rho, kappa))
                spec = one_factor_spec(base, nu, rho, kappa)
                label = f"{surface.name} 1F nu={nu:g} rho={rho:g} kappa={kappa:g}"
                pt = GridPoint(
                    spec_key(spec),
                    label,
                    surface.name,
                    "one_factor",
                    spec,
                    {"nu": nu, "rho": rho, "kappa": kappa},
                    spec_key(spec),
                )
                per_surface.append(((1, (nu, rho, kappa)), pt))
        if surface.two_factor:
            for i, preset in enumerate(grid.two_factor_presets):
                spec = dataclasses.replace(base, model=preset.model)
                pt = GridPoint(
                    spec_key(spec),
                    f"{surface.name} 2F {preset.name}",
                    surface.name,
                    "two_factor",
                    spec,
                    {"preset": float(i)},
                    spec_key(spec),
                )
                per_surface.append(((2, (i,)), pt))
        if surface.marking and grid.marking is not None:
            for ssr in grid.marking.ssr_target:
                for eps in grid.marking.skew_eps:
                    pt = GridPoint(
                        marking_id(surface.name, ssr, eps),
                        f"{surface.name} marking ssr={ssr:g} eps={eps:g}",
                        surface.name,
                        "marking",
                        base,
                        {"ssr_target": ssr, "skew_eps": eps},
                        None,
                        "unresolved",
                        False,
                    )
                    per_surface.append(((3, (ssr, eps)), pt))
        for j, extra in enumerate(e for e in grid.extra_points if e.surface == surface.name):
            spec = dataclasses.replace(base, model=extra.model)
            mode = "one_factor" if extra.model.is_one_factor else "two_factor"
            pt = GridPoint(
                spec_key(spec),
                f"{surface.name} {extra.label}",
                surface.name,
                mode,
                spec,
                {"extra": float(j)},
                spec_key(spec),
            )
            per_surface.append(((4, (j,)), pt))
        per_surface.sort(key=lambda kv: kv[0])
        points.extend(pt for _, pt in per_surface)
    ids = [p.id for p in points]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate grid points (same calibration spec): {dup}")
    return points


def count_by(points: Sequence[GridPoint]) -> dict[tuple[str, str], int]:
    """``{(surface, mode): count}`` in enumeration order."""
    out: dict[tuple[str, str], int] = {}
    for p in points:
        out[(p.surface, p.mode)] = out.get((p.surface, p.mode), 0) + 1
    return out


def shard(points: Sequence[GridPoint], i: int, n: int) -> list[GridPoint]:
    """The ``i``-th of ``n`` interleaved shards (1-based): ``points[i-1::n]``."""
    if n < 1 or not 1 <= i <= n:
        raise ValueError(f"shard must satisfy 1 <= i <= n, got {i}/{n}")
    return list(points[i - 1 :: n])


def parse_shard(text: str) -> tuple[int, int]:
    """``"i/n"`` → ``(i, n)`` validated."""
    try:
        a, b = text.split("/")
        i, n = int(a), int(b)
    except ValueError as exc:
        raise ValueError(f"--shard expects i/n, got {text!r}") from exc
    if n < 1 or not 1 <= i <= n:
        raise ValueError(f"--shard must satisfy 1 <= i <= n, got {text!r}")
    return i, n


def resolve_axis_point(
    points: Sequence[GridPoint], surface: str, nu: float, rho: float, kappa: float
) -> GridPoint | None:
    """The one-factor point at ``(nu, rho, kappa)`` on ``surface`` — the surface's LV point when
    ``nu == 0`` whatever ``rho`` and ``kappa`` (module docstring)."""
    for p in points:
        if p.surface != surface:
            continue
        if nu == 0.0 and p.mode == "lv":
            return p
        if (
            p.mode == "one_factor"
            and p.axes.get("nu") == nu
            and p.axes.get("rho") == rho
            and p.axes.get("kappa") == kappa
        ):
            return p
    return None


# --------------------------------------------------------------------------------------------
# marking fits
# --------------------------------------------------------------------------------------------


def marking_summary(r: FitResult) -> MarkingSummary:
    gaps: dict[str, float] = {}
    for rec in r.constraints.to_dict(orient="records"):
        gaps[f"skew_gap_{rec['name']}@{float(rec['T']):g}y"] = float(rec["gap_rel"])
    ssr_fo: dict[float, float] = {}
    if "ssr_first_order" in r.table:
        for T, v in zip(r.table["T"], r.table["ssr_first_order"]):
            ssr_fo[float(T)] = float(v)
    return MarkingSummary(
        r.status,
        tuple(r.messages),
        tuple(r.notes),
        gaps,
        float(r.mean_skew_gap),
        ssr_fo,
        float(r.wall_seconds),
    )


def resolve_marking(point: GridPoint, surface: ImpliedSurface) -> GridPoint:
    """Run the marking fit of an unresolved marking point on its (already built) surface and
    return the resolved point: fitted model in ``spec``, ``status`` from the fit, ``cache_key``
    set unless the fit is infeasible (then the point is recorded without a calibration)."""
    if point.mode != "marking":
        raise ValueError("only marking points are resolved by a fit")
    if point.resolved:
        return point
    ssr, eps = point.axes["ssr_target"], point.axes["skew_eps"]
    r = fit_2f_marking(surface, BreakEvenFitConfig(skew_eps=eps), ssr_target=ssr)
    summary = marking_summary(r)
    spec = dataclasses.replace(point.spec, model=r.params)
    key = None if r.status == "infeasible" else spec_key(spec)
    log.info(
        "marking fit %s: %s (nu=%.4g theta=%.4g k1=%.4g rho_SX1=%.4g rho_SX2=%.4g "
        "rho12=%.4g) in %.1f s",
        point.label,
        r.status,
        r.params.nu,
        r.params.theta,
        r.params.k1,
        r.params.rho_SX1,
        r.params.rho_SX2,
        r.params.rho12,
        r.wall_seconds,
    )
    return dataclasses.replace(
        point, spec=spec, cache_key=key, status=r.status, resolved=True, marking=summary
    )
