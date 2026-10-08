"""Configuration dataclasses and strict YAML loading (SPEC §11).

Every model / engine / market parameter set is a frozen dataclass.  Loading from YAML is strict:
unknown keys, missing keys and wrong scalar types raise :class:`ConfigError` with the offending
path.  Model parameters never have defaults (no silent defaults); engine settings carry the
documented SPEC defaults.
"""

from __future__ import annotations

import bisect
import dataclasses
import math
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar, get_args, get_origin, get_type_hints

import numpy as np
import yaml
from numpy.typing import NDArray


class ConfigError(ValueError):
    """Raised when a configuration file or mapping fails validation."""


# --------------------------------------------------------------------------------------------
# Generic strict dataclass construction
# --------------------------------------------------------------------------------------------


def _type_name(tp: Any) -> str:
    return getattr(tp, "__name__", repr(tp))


def _coerce(value: Any, tp: Any, path: str) -> Any:
    """Validate/coerce ``value`` against the annotation ``tp``; raise ConfigError on mismatch."""
    origin = get_origin(tp)
    if tp is Any:
        return value
    if origin in (types.UnionType, typing.Union):
        args = get_args(tp)
        if value is None and type(None) in args:
            return None
        errors = []
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce(value, arg, path)
            except ConfigError as exc:
                errors.append(str(exc))
        raise ConfigError(f"{path}: value {value!r} matches none of {tp}; " + "; ".join(errors))
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        out = float(value)
        if not math.isfinite(out):
            raise ConfigError(f"{path}: expected a finite number, got {value!r}")
        return out
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            if isinstance(value, float) and value.is_integer():
                return int(value)
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected a boolean, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    if tp is Path:
        if not isinstance(value, (str, Path)):
            raise ConfigError(f"{path}: expected a path string, got {value!r}")
        return Path(value)
    if origin in (list, tuple, Sequence) or tp in (list, tuple):
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list, got {value!r}")
        args = get_args(tp)
        if origin is tuple and args and not (len(args) == 2 and args[1] is Ellipsis):
            if len(args) != len(value):
                raise ConfigError(f"{path}: expected {len(args)} items, got {len(value)}")
            return tuple(_coerce(v, a, f"{path}[{i}]") for i, (v, a) in enumerate(zip(value, args)))
        item_tp = args[0] if args else Any
        items = [_coerce(v, item_tp, f"{path}[{i}]") for i, v in enumerate(value)]
        return tuple(items) if origin is tuple or tp is tuple else items
    if origin in (dict, Mapping) or tp is dict:
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping, got {value!r}")
        args = get_args(tp)
        k_tp, v_tp = args if args else (Any, Any)
        return {
            _coerce(k, k_tp, f"{path}.<key>"): _coerce(v, v_tp, f"{path}.{k}")
            for k, v in value.items()
        }
    if is_dataclass(tp) and isinstance(tp, type):
        if not isinstance(value, Mapping):
            raise ConfigError(f"{path}: expected a mapping for {_type_name(tp)}, got {value!r}")
        return from_mapping(tp, value, path=path)
    if isinstance(tp, type):
        if isinstance(value, tp):
            return value
        raise ConfigError(f"{path}: expected {_type_name(tp)}, got {value!r}")
    return value


def from_mapping[T](cls: type[T], data: Mapping[str, Any], *, path: str = "") -> T:
    """Build dataclass ``cls`` from ``data`` strictly (unknown / missing keys are errors)."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass")
    hints = get_type_hints(cls)
    known = {f.name: f for f in fields(cls) if f.init}
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ConfigError(f"{path or _type_name(cls)}: unknown keys {unknown}")
    kwargs: dict[str, Any] = {}
    for name, f in known.items():
        sub_path = f"{path}.{name}" if path else f"{_type_name(cls)}.{name}"
        if name in data:
            kwargs[name] = _coerce(data[name], hints[name], sub_path)
        elif f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            raise ConfigError(f"{sub_path}: missing required key")
    try:
        return cls(**kwargs)
    except (ValueError, TypeError) as exc:
        raise ConfigError(f"{path or _type_name(cls)}: {exc}") from exc


def load_yaml[T](path: str | Path, cls: type[T], *, section: str | None = None) -> T:
    """Load ``path`` as YAML and build dataclass ``cls`` (optionally from a top-level section)."""
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if raw is None:
        raise ConfigError(f"{p}: empty file")
    if section is not None:
        if not isinstance(raw, Mapping) or section not in raw:
            raise ConfigError(f"{p}: missing section {section!r}")
        raw = raw[section]
    if not isinstance(raw, Mapping):
        raise ConfigError(f"{p}: top level must be a mapping")
    return from_mapping(cls, raw, path=str(p))


def omitted_when_none(obj: Any) -> frozenset[str]:
    """The fields of a config dataclass that :func:`to_mapping` leaves out while they hold
    ``None``: its ``OMIT_WHEN_NONE`` class attribute (empty by default).

    The invariant it enforces, in this one place: **a field added to a config after cache keys
    were computed is absent from every mapping of a config that does not use it**, so the
    leverage-cache key (:meth:`CalibrationSpec.key_payload`), the viewers' LV id, ``spec.json``
    and every YAML dump of such a config are byte-identical to what they were before the field
    existed.  The field must default to ``None``.  Checked by
    ``tests/test_surface_config.py::test_keys_of_committed_specs_unchanged``."""
    omit: frozenset[str] = frozenset(getattr(type(obj), "OMIT_WHEN_NONE", frozenset()))
    return omit


def to_mapping(obj: Any) -> Any:
    """Recursively convert a dataclass (with numpy scalars/arrays) to plain YAML-safe types.

    A field named in the class's ``OMIT_WHEN_NONE`` is left out while it is ``None``
    (:func:`omitted_when_none`)."""
    if is_dataclass(obj) and not isinstance(obj, type):
        omit = omitted_when_none(obj)
        return {
            f.name: to_mapping(getattr(obj, f.name))
            for f in fields(obj)
            if not (f.name in omit and getattr(obj, f.name) is None)
        }
    if isinstance(obj, np.ndarray):
        return [to_mapping(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [to_mapping(v) for v in obj]
    if isinstance(obj, Mapping):
        return {str(k): to_mapping(v) for k, v in obj.items()}
    return obj


def dump_yaml(obj: Any, path: str | Path) -> None:
    """Write a dataclass to YAML."""
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(to_mapping(obj), fh, sort_keys=False)


# --------------------------------------------------------------------------------------------
# Engine configuration (defaults documented in SPEC §3.1 / §5 and the owner's post-M1 amendments)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StepSchedule:
    """Piecewise-constant maximum simulation step ``dt_max(t)``.

    ``dt_max = dts[i]`` for ``t ∈ [breaks[i-1], breaks[i])`` with ``breaks[-1] = ∞``.  The default
    is 1/1460 below 3m, 1/365 up to 2y and 1/250 beyond (M4b).  History: the schedule was halved
    before M4 because the frozen-variance LSV step showed a −0.14 vp 1y variance-swap error on
    this schedule; M4b traced most of that to one particle seed and to the regression grid, made
    the SV spot step second order and the regression grid adaptive, and measured (1F ω = 3,
    three particle seeds, N = 2·10⁵) that this schedule reprices as well as the halved one on
    average — the halved one even carries a +0.06–0.08 vp ATM bias at 1y–3y with the
    second-order step — at half the cost (calibration ≈ 33 s for 3y); see SPEC §4.2.  The Monte
    Carlo engine and the particle calibration use the same schedule so that a calibrated leverage
    function reprices the surface on the grid it is used on.
    """

    breaks: tuple[float, ...] = (0.25, 2.0)
    dts: tuple[float, ...] = (1.0 / 1460.0, 1.0 / 365.0, 1.0 / 250.0)

    def __post_init__(self) -> None:
        if len(self.dts) != len(self.breaks) + 1:
            raise ValueError("need len(dts) == len(breaks) + 1")
        if any(d <= 0 for d in self.dts):
            raise ValueError("step sizes must be positive")
        if any(b <= 0 for b in self.breaks) or any(np.diff(self.breaks) <= 0):
            raise ValueError("breaks must be positive and strictly increasing")

    @classmethod
    def uniform(cls, dt: float) -> StepSchedule:
        return cls(breaks=(), dts=(float(dt),))

    def dt_at(self, t: float) -> float:
        """Maximum step applying at time ``t``."""
        return self.dts[bisect.bisect_right(self.breaks, t)]

    def knots(self, horizon: float) -> tuple[float, ...]:
        """Break times strictly inside ``(0, horizon)`` (they become grid points)."""
        return tuple(b for b in self.breaks if 0.0 < b < horizon)

    @property
    def finest(self) -> float:
        return min(self.dts)

    def __repr__(self) -> str:
        if not self.breaks:
            return f"StepSchedule(uniform dt={self.dts[0]:.6g})"
        parts = [f"dt={d:.6g} until {b:g}y" for b, d in zip(self.breaks, self.dts)]
        parts.append(f"dt={self.dts[-1]:.6g} after")
        return "StepSchedule(" + ", ".join(parts) + ")"


DEFAULT_STEP_SCHEDULE = StepSchedule()


@dataclass(frozen=True)
class SchemeConfig:
    """Spot-step discretisation options, shared verbatim by pricing and calibration kernels.

    Default: Platen weak order 2.  Measured on the reference SSVI surface (1m ATM, uniform
    dt = 1/365, 200k paths): plain log-Euler +0.36 vol points, time-averaged variance +0.29,
    predictor–corrector θ = η = ½ +0.78 (it over-corrects the local-variance curvature by 2×),
    weak order 2 +0.03; see ``tests/test_scheme.py`` and the M1→M2 report.

    Attributes:
        local_var_time_average: use the time average of the local (or forward) variance over
            ``[t_n, t_{n+1}]`` at the frozen state instead of its value at ``t_n``.
        predictor_corrector: weak predictor–corrector (Kloeden–Platen §15.5) — Euler predictor
            for ``x_{n+1}``, drift averaged with θ = ½ between ``(t_n, x_n)`` and
            ``(t_{n+1}, x_pred)`` and carrying the Itô correction ``−η b b'``, diffusion variance
            weighted ``(1 − η, η)`` between the two points.  The correction is required because
            the corrector's diffusion coefficient depends on the predictor and hence on the
            Brownian increment (without it the forward is not preserved; the original
            specification omitted it).  With θ = η = ½ the scheme over-corrects the curvature
            of the local variance in ``ln S`` (coefficient 1.5 η against the exact ¼), so it is
            kept as an option, not the default.
        pc_eta: the diffusion weight ``η`` of the predictor–corrector.
        weak_order2: Platen's explicit weak order-2 scheme (Kloeden–Platen eq. 15.1.3) with the
            coefficients at ``t_n`` and ``t_{n+1}``; excludes ``predictor_corrector`` and ignores
            time averaging.  Three extra variance lookups per step.
        local_var_time_eval: when ``local_var_time_average`` is off, evaluate the variance at the
            step ``"start"`` (plain log-Euler) or at the ``"midpoint"`` in time (diagnostic).
        sv_order2: second-order SV step for the Bergomi / LSV kernels (M4b): exact factor
            increments first, trapezoidal variance in the drift and the explicit weak order-2
            spot/variance cross terms (see :func:`volsto.models.bergomi.bergomi_block`).  Off:
            the variance is frozen at the step start (the M2/M3 step).  Independent of the spot
            ``mode`` (applies to log-Euler and Platen; ignored by the predictor–corrector).
    """

    local_var_time_average: bool = True
    predictor_corrector: bool = False
    pc_eta: float = 0.5
    weak_order2: bool = True
    local_var_time_eval: str = "start"
    sv_order2: bool = True

    def __post_init__(self) -> None:
        if self.local_var_time_eval not in ("start", "midpoint"):
            raise ValueError("local_var_time_eval must be 'start' or 'midpoint'")
        if not 0.0 <= self.pc_eta <= 1.0:
            raise ValueError("pc_eta must lie in [0, 1]")
        if self.weak_order2 and self.predictor_corrector:
            raise ValueError("weak_order2 and predictor_corrector are mutually exclusive")


@dataclass(frozen=True)
class SimConfig:
    """Monte Carlo engine settings (SPEC §3.1, §5).

    Attributes:
        n_paths: number of paths (SPEC default 2·10⁵).  Must be even when antithetic.
        dt_max: maximum simulation step — a :class:`StepSchedule` (default: 1/1460 below 3m,
            1/365 to 2y, 1/250 after) or a float for a uniform step.
        chunk_size: paths simulated per block to bound memory (SPEC default 5·10⁴); further
            capped by ``chunk_memory_mb`` when the path container is large.
        antithetic: use antithetic pairs (path 2i+1 uses the negated normals of path 2i).
        seed: base seed of the CRN-keyed PCG64 stream (SPEC §5).
        local_var_time_average, predictor_corrector, pc_eta, weak_order2, local_var_time_eval,
        sv_order2: see :class:`SchemeConfig`.
        record_all_steps: record the state at every grid step, not only at fixing times.
        chunk_memory_mb: budget for one chunk's path container (all recorded arrays).
    """

    n_paths: int = 200_000
    dt_max: StepSchedule | float = field(default_factory=StepSchedule)
    chunk_size: int = 50_000
    antithetic: bool = True
    seed: int = 0
    local_var_time_average: bool = True
    predictor_corrector: bool = False
    pc_eta: float = 0.5
    weak_order2: bool = True
    local_var_time_eval: str = "start"
    sv_order2: bool = True
    record_all_steps: bool = False
    chunk_memory_mb: int = 512

    def __post_init__(self) -> None:
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if self.antithetic and self.n_paths % 2:
            raise ValueError("n_paths must be even when antithetic=True")
        if isinstance(self.dt_max, (int, float)) and self.dt_max <= 0:
            raise ValueError("dt_max must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.antithetic and self.chunk_size % 2:
            raise ValueError("chunk_size must be even when antithetic=True")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.chunk_memory_mb <= 0:
            raise ValueError("chunk_memory_mb must be positive")
        _ = self.scheme  # validates the scheme options

    @property
    def step_schedule(self) -> StepSchedule:
        if isinstance(self.dt_max, StepSchedule):
            return self.dt_max
        return StepSchedule.uniform(float(self.dt_max))

    @property
    def scheme(self) -> SchemeConfig:
        return SchemeConfig(
            local_var_time_average=self.local_var_time_average,
            predictor_corrector=self.predictor_corrector,
            pc_eta=self.pc_eta,
            weak_order2=self.weak_order2,
            local_var_time_eval=self.local_var_time_eval,
            sv_order2=self.sv_order2,
        )

    def effective_chunk_size(self, n_cols: int, n_factors: int) -> int:
        """``chunk_size`` capped so that one chunk's recorded arrays fit ``chunk_memory_mb``."""
        bytes_per_path = 8 * n_cols * (4 + n_factors)
        cap = max(2, (self.chunk_memory_mb * 2**20) // bytes_per_path)
        chunk = min(self.chunk_size, cap)
        if self.antithetic and chunk % 2:
            chunk -= 1
        return max(chunk, 2 if self.antithetic else 1)

    def chunk_ranges(self, n_cols: int, n_factors: int) -> list[tuple[int, int]]:
        """``[(p0, p1), …]`` path ranges covering ``n_paths`` with the effective chunk size."""
        c = self.effective_chunk_size(n_cols, n_factors)
        return [(p0, min(p0 + c, self.n_paths)) for p0 in range(0, self.n_paths, c)]


# --------------------------------------------------------------------------------------------
# Market configuration
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CurveConfig:
    """Piecewise-flat zero curve: continuously compounded zero rates at pillar times (years)."""

    times: tuple[float, ...]
    rates: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.times) != len(self.rates) or not self.times:
            raise ValueError("times and rates must be non-empty and of equal length")
        if any(t <= 0 for t in self.times) or any(np.diff(self.times) <= 0):
            raise ValueError("times must be strictly increasing and positive")

    @classmethod
    def flat(cls, rate: float) -> CurveConfig:
        return cls(times=(1.0,), rates=(float(rate),))


@dataclass(frozen=True)
class MarketConfig:
    """Spot plus rate and dividend-yield curves (SPEC §2.1).

    ``close`` (2026-09-22, SPEC §13.1): the official close of the day — the level trades fix
    on — when it differs from ``spot``, the level the option quotes imply (the importer's
    snapshots: the HDN close is the 16:00 print, SPX options quote until 16:15).  Pricing never
    reads it; the realised history, the backtest's fixings and its strikes do.  It is not a
    pricing input, so :meth:`CalibrationSpec.key_payload` leaves it out of the cache key, and
    a config without it maps and hashes exactly as before (``OMIT_WHEN_NONE``)."""

    spot: float
    rate_curve: CurveConfig
    dividend_curve: CurveConfig
    close: float | None = None

    OMIT_WHEN_NONE: ClassVar[frozenset[str]] = frozenset({"close"})

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive")
        if self.close is not None and self.close <= 0:
            raise ValueError("close must be positive")


@dataclass(frozen=True)
class SSVIConfig:
    """Gatheral–Jacquier power-law SSVI (SPEC §2.2), or its eSSVI extension (SPEC §13.2).

    ``atm_maturities`` / ``atm_vols`` give the ATM implied-vol term structure; θ_T = σ_ATM(T)² T
    is interpolated linearly in T (flat forward variance) between pillars.

    ``rhos`` (M10 Part 3): one correlation per ATM pillar, ``|ρ_i| < 1`` — the eSSVI surface the
    importer fits by default (``ρ_T`` piecewise linear between the pillars, flat outside;
    :class:`volsto.market.surface.ESSVISurface`).  ``None`` is the plain SSVI with the scalar
    ``rho``.  With ``rhos`` set the scalar ``rho`` stays in the config (the importer writes the
    mean of the pillar values there) and the surface ignores it.  ``rhos`` is in
    :data:`OMIT_WHEN_NONE`: a plain SSVI config maps — and hashes into the leverage-cache key —
    exactly as before the field existed.  Surfaces are built from a config only by
    :func:`volsto.market.surface.surface_from_config`; a snapshot's ``essvi`` section is read
    only by :func:`volsto.market.loaders.load_surface_config`.  Checked by
    ``tests/test_surface_config.py``.
    """

    atm_maturities: tuple[float, ...]
    atm_vols: tuple[float, ...]
    rho: float
    eta: float
    gamma: float
    max_maturity: float = 10.0
    rhos: tuple[float, ...] | None = None

    OMIT_WHEN_NONE: ClassVar[frozenset[str]] = frozenset({"rhos"})
    """Fields left out of every mapping while ``None`` (:func:`omitted_when_none`)."""

    def __post_init__(self) -> None:
        if self.rhos is not None:
            rhos = tuple(float(r) for r in self.rhos)
            object.__setattr__(self, "rhos", rhos)
            if len(rhos) != len(self.atm_maturities):
                raise ValueError(
                    f"rhos needs one correlation per ATM pillar: {len(rhos)} given for "
                    f"{len(self.atm_maturities)} pillars"
                )
            if not all(math.isfinite(r) and -1.0 < r < 1.0 for r in rhos):
                raise ValueError("rhos must be finite and lie in (-1, 1)")
        if len(self.atm_maturities) != len(self.atm_vols) or not self.atm_maturities:
            raise ValueError("atm_maturities and atm_vols must be non-empty and of equal length")
        if any(t <= 0 for t in self.atm_maturities) or any(np.diff(self.atm_maturities) <= 0):
            raise ValueError("atm_maturities must be strictly increasing and positive")
        if any(v <= 0 for v in self.atm_vols):
            raise ValueError("atm_vols must be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")
        if self.eta < 0:
            raise ValueError("eta must be non-negative")
        if not 0.0 < self.gamma <= 1.0:
            raise ValueError("gamma must lie in (0, 1]")
        if self.max_maturity < max(self.atm_maturities):
            raise ValueError("max_maturity must cover the last ATM pillar")


@dataclass(frozen=True)
class LocalVolConfig:
    """Dupire local-vol grid and finite-difference settings (SPEC §2.3).

    The ``t`` grid has ``n_t`` points square-root spaced between ``t_min`` and ``t_max``.  The
    ``k`` spacing must resolve the short-dated local skew: variance-swap repricing under local-vol
    MC converged only for ``dk ≤ 0.0025`` on the reference surface (``dk = 0.01`` biased the 1y
    strike by 0.5% of variance).  The ``k`` range must cover the far put wing of long-dated
    variance swaps: with ``±1.5`` the 2y / 3y strikes were 0.08 / 0.22 vol points low (flat local
    vol beyond the grid), with ``±3.0`` within 0.02 — hence 2401 points over ``±3.0``.
    """

    t_min: float = 1.0 / 365.0
    t_max: float = 3.0
    n_t: int = 400
    k_min: float = -3.0
    k_max: float = 3.0
    n_k: int = 2401
    dk: float = 1e-3
    dt: float = 1e-3
    floor: float = 1e-4  # variance floor (vol 1%)
    cap: float = 25.0  # variance cap (vol 500%)

    def __post_init__(self) -> None:
        if not 0 < self.t_min < self.t_max:
            raise ValueError("need 0 < t_min < t_max")
        if self.n_t < 2 or self.n_k < 3:
            raise ValueError("grid too small")
        if self.k_min >= self.k_max:
            raise ValueError("need k_min < k_max")
        if self.dk <= 0 or self.dt <= 0:
            raise ValueError("finite-difference steps must be positive")
        if not 0 < self.floor < self.cap:
            raise ValueError("need 0 < floor < cap")


# --------------------------------------------------------------------------------------------
# Model configurations (M1: BS and local vol; Bergomi/Heston added in later milestones)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BSModelConfig:
    """Black–Scholes flat volatility."""

    vol: float

    def __post_init__(self) -> None:
        if self.vol <= 0:
            raise ValueError("vol must be positive")


@dataclass(frozen=True)
class LocalVolModelConfig:
    """Dupire local vol built from a surface config (SPEC §2.3)."""

    grid: LocalVolConfig = field(default_factory=LocalVolConfig)


# --------------------------------------------------------------------------------------------
# Bergomi two-factor lognormal forward-variance model (SPEC §3.3; Bergomi ch. 7–8)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BergomiVarianceDynamics:
    """Variance-curve dynamics only: ``(ν, θ, k1, k2, ρ12)`` — book Table 7.1 Sets I–III."""

    nu: float
    theta: float
    k1: float
    k2: float
    rho12: float

    def __post_init__(self) -> None:
        if self.nu < 0:
            raise ValueError("nu must be non-negative")
        if not 0.0 <= self.theta <= 1.0:
            raise ValueError("theta must lie in [0, 1]")
        if self.k1 <= 0 or self.k2 <= 0:
            raise ValueError("mean reversions k1, k2 must be positive")
        if not -1.0 <= self.rho12 <= 1.0:
            raise ValueError("rho12 must lie in [-1, 1]")


@dataclass(frozen=True)
class BergomiParams:
    """Two-factor lognormal Bergomi parameters (book notation, SPEC §3.3).

    ``ν`` is the lognormal volatility of a VS volatility of vanishing maturity; ``ω = 2ν`` is the
    lognormal vol of vol of ``ξ_t^t`` (eq. 7.12a).  ``k1 > k2``: ``X¹`` is the short factor.
    ``ρ12 = corr(dW¹, dW²)``, ``ρ_SX1 = corr(dW^S, dW¹)``, ``ρ_SX2 = corr(dW^S, dW²)``; the 3×3
    correlation matrix of ``(W^S, W¹, W²)`` must be positive semi-definite (checked here).
    ``θ = 0`` is the one-factor model of the earlier studies with ``κ = k1`` and ``ρ = ρ_SX1``.
    """

    nu: float
    theta: float
    k1: float
    k2: float
    rho12: float
    rho_SX1: float
    rho_SX2: float

    def __post_init__(self) -> None:
        BergomiVarianceDynamics(self.nu, self.theta, self.k1, self.k2, self.rho12)
        for name in ("rho_SX1", "rho_SX2"):
            if not -1.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must lie in [-1, 1]")
        eig = np.linalg.eigvalsh(self.correlation_matrix)
        if eig.min() < -1e-12:
            raise ValueError(
                f"correlation matrix of (W^S, W1, W2) is not PSD (min eigenvalue {eig.min():.3e})"
            )

    @property
    def omega(self) -> float:
        """``ω = 2ν`` (eq. 7.12a): lognormal vol of vol of the instantaneous variance."""
        return 2.0 * self.nu

    @property
    def is_one_factor(self) -> bool:
        return self.theta == 0.0

    @property
    def correlation_matrix(self) -> NDArray[np.float64]:
        """Correlation of ``(W^S, W¹, W²)``."""
        return np.array(
            [
                [1.0, self.rho_SX1, self.rho_SX2],
                [self.rho_SX1, 1.0, self.rho12],
                [self.rho_SX2, self.rho12, 1.0],
            ]
        )

    @property
    def dynamics(self) -> BergomiVarianceDynamics:
        return BergomiVarianceDynamics(self.nu, self.theta, self.k1, self.k2, self.rho12)

    @classmethod
    def one_factor(cls, omega: float, kappa: float, rho: float) -> BergomiParams:
        """The 1F model of the earlier studies: ``ω = 2ν``, ``κ = k1``, ``ρ = ρ_SX1``, ``θ = 0``."""
        return cls(0.5 * omega, 0.0, kappa, kappa, 0.0, rho, 0.0)

    @classmethod
    def from_chi(
        cls,
        nu: float,
        theta: float,
        k1: float,
        k2: float,
        rho12: float,
        rho_SX1: float,
        chi: float,
    ) -> BergomiParams:
        """Admissible parametrisation, book eq. 8.56:
        ``ρ_SX2 = ρ12 ρ_SX1 + χ sqrt(1 − ρ12²) sqrt(1 − ρ_SX1²)``, ``χ ∈ [−1, 1]``."""
        if not -1.0 <= chi <= 1.0:
            raise ValueError("chi must lie in [-1, 1]")
        rho_sx2 = rho12 * rho_SX1 + chi * math.sqrt(1.0 - rho12**2) * math.sqrt(1.0 - rho_SX1**2)
        return cls(nu, theta, k1, k2, rho12, rho_SX1, rho_sx2)

    @classmethod
    def from_dynamics(
        cls, dyn: BergomiVarianceDynamics, rho_SX1: float, rho_SX2: float
    ) -> BergomiParams:
        return cls(dyn.nu, dyn.theta, dyn.k1, dyn.k2, dyn.rho12, rho_SX1, rho_SX2)

    def replace(self, **changes: float) -> BergomiParams:
        return dataclasses.replace(self, **changes)


# --------------------------------------------------------------------------------------------
# Particle calibration and calibration specification (SPEC §4.1–4.3)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ParticleConfig:
    """Particle-method settings (SPEC §4.1, Guyon–Henry-Labordère; Bergomi §12.2.5).

    Attributes:
        n_particles: ``N`` (SPEC default 2·10⁵).
        horizon: calibration horizon ``T_cal`` in years; ``L`` is held at its last slice beyond.
        bandwidth_factor: ``c`` in ``h_i = c · σ_ref · sqrt(t_{i+1}) · N^{−1/5}`` (default 1.5).
        bandwidth_min: floor ``h_min`` on the bandwidth (log-moneyness units).
        kernel: regression kernel, ``"gaussian"`` (truncated at 4h) or ``"quartic"``.
        regression: ``"local_linear"`` (default) or ``"nadaraya_watson"``.  Local-linear removes
            the design bias ``h² m'(k) f'(k)/f(k)`` of Nadaraya–Watson, which with a steeply
            sloped ``E[V|S]`` and ``c = 1.5`` biased the ±10% repricing by ~0.3 vol points.
        quantile_clip: the regression is trusted inside the ``[q, 1−q]`` quantiles of the cloud.
        min_window, min_window_fraction: the regression window is widened to the nearest
            ``max(min_window, min_window_fraction · N)`` particles wherever the ``4h`` window holds
            fewer (the tails): a k-NN floor whose *fraction* of the cloud is fixed, so the tail
            bandwidth is set by the local particle density and does not shrink as ``N`` grows
            (a fixed count did, and made the variance-swap residual non-monotone in ``N``:
            −0.14 vp at 2·10⁵, +0.23 vp at 8·10⁵ particles at 1y).  The absolute floor of 2000
            keeps small clouds (5·10⁴ particles) from tail noise that inflates ``L`` through
            ``1/E[V|S]`` (a 500-particle floor gave +0.17 vp on the 1y variance swap at 5·10⁴).
        bias_correction: subtract the plug-in local-linear curvature bias ``½ h² m''(k)`` (Gaussian
            kernel second moment 1; ``m''`` by second differences on the regression grid).  The
            uncorrected estimate over-states the convex ``E[V|S]`` by ``O(h²)``, a level bias in
            the repriced vols growing with ``c²`` (−0.10 vp at 1y ATM for ``c = 1.5``).
        tail_extrapolation: outside those quantiles hold ``E[V|S]`` ``"flat"`` (SPEC §4.1
            wording), continue ``ln E[V|S]`` linearly with the slope fitted over the outer trusted
            points (``"log_linear"``), or continue it with a quadratic fitted there whose slope is
            only allowed to decay, going flat where it would vanish (``"log_quadratic"``,
            default), or choose per tail (``"adaptive"``: log-linear where ``E[V|S]`` rises into
            the tail, saturating quadratic where it falls), or continue ``ln E[V|S]`` with the
            pure-SV kernel's own conditional slope (``"sv_slope"``, M6 Part 0 item 4: the
            model-consistent tail ``β(t) = Σ_i coef_i ρ_Si (1 − e^{−k_i t}) / (k_i σ_ATM(t) t)``,
            the Gaussian-approximation slope of ``ln E[ξ_t^t | ln S_t]``, which overstates the
            measured pure-SV slope by 28% at ω = 3, 1y), or with the slope of ``ln V`` on ``k``
            fitted over the whole particle cloud at the slice (``"cloud_slope"``, the LSV's own
            conditional slope, measured).  By repricing error on the reference
            surface (200k particles, 1y horizon, 3m–1y, |k| ≤ 0.2) the quadratic rule is the most
            robust: 1F 0.11 vp, 2F 0.09 vp, against 0.13 / 0.37 vp for log-linear (which
            over-extrapolates the 2F's decaying right tail: 3m +20% calls 0.37 vp rich).
            Measured on the reference surface: the flat rule understates ``E[V|S]`` in both tails
            (+0.2 vp at 3m ±20%, +0.2–0.3 vp on variance swaps); the linear rule keeps the edge
            slope, which over-extrapolates the decaying right tail of the 2F model (E[V|S] at 3m
            +20% under-estimated by 11%, calls +0.3 vp rich); the saturating quadratic follows the
            observed flattening of ``ln E[V|S]``.
        n_regression_points: dense grid on which ``E[V|S]`` is regressed, spanning the particle
            cloud's trusted quantile range at each slice (M4b; interpolated onto the leverage grid).
        leverage_std_span: half-width of the leverage grid in ATM standard deviations.
        leverage_dk: spacing of the leverage log-moneyness grid (same as the Dupire grid).
        l_min, l_max: clip bounds on ``L``.
        second_pass: re-run with the calibrated ``L`` and a fresh seed and average the two.
        seed: seed of the particle draws (CRN key).
        antithetic: antithetic particle pairs.
        estimator: how the regression stage is computed.  ``None`` (the default) and
            ``"sorted"`` are the same numbers: the particles are sorted and each node sums its
            window (:func:`volsto.calibration.particle.kernel_regression`).  ``"binned"`` is the
            estimator of :mod:`volsto.calibration.binned`: exact quantile nodes, floor windows
            and floor bandwidths by counting and selection, kernel sums on linear bins of width
            ``h/4`` for the unfloored nodes and over the tail particles for the floored ones — no
            sort, 2.6 times faster per calibration at 8·10⁵ particles on one thread, and within
            0.0005 vol points of the sorted path on every repriced pillar (measured, 24 paired
            calibrations).  It needs the Gaussian kernel and the local-linear regression.  The
            field is left out of every mapping while ``None`` (:func:`omitted_when_none`), so the
            cache key of a spec that does not set it is what it was before the field existed;
            an explicit ``"sorted"`` is keyed, and so addresses other cache entries than ``None``.
    """

    n_particles: int = 200_000
    horizon: float = 3.0
    bandwidth_factor: float = 1.5
    bandwidth_min: float = 5e-4
    kernel: str = "gaussian"
    regression: str = "local_linear"
    min_window: int = 2000
    min_window_fraction: float = 0.01
    bias_correction: bool = True
    quantile_clip: float = 0.005
    tail_extrapolation: str = "log_quadratic"
    n_regression_points: int = 201
    leverage_std_span: float = 6.0
    leverage_dk: float = 0.0025
    l_min: float = 0.05
    l_max: float = 20.0
    second_pass: bool = False
    seed: int = 12345
    antithetic: bool = True
    estimator: str | None = None

    OMIT_WHEN_NONE: ClassVar[frozenset[str]] = frozenset({"estimator"})
    """Fields left out of every mapping while ``None`` (:func:`omitted_when_none`)."""

    def __post_init__(self) -> None:
        if self.n_particles < 100:
            raise ValueError("n_particles too small")
        if self.estimator not in (None, "sorted", "binned"):
            raise ValueError("estimator must be None, 'sorted' or 'binned'")
        if self.estimator == "binned" and (
            self.kernel != "gaussian" or self.regression != "local_linear"
        ):
            raise ValueError(
                "estimator='binned' needs kernel='gaussian' and regression='local_linear'"
            )
        if self.antithetic and self.n_particles % 2:
            raise ValueError("n_particles must be even when antithetic")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")
        if self.bandwidth_factor <= 0 or self.bandwidth_min <= 0:
            raise ValueError("bandwidths must be positive")
        if self.kernel not in ("gaussian", "quartic"):
            raise ValueError("kernel must be 'gaussian' or 'quartic'")
        if self.regression not in ("local_linear", "nadaraya_watson"):
            raise ValueError("regression must be 'local_linear' or 'nadaraya_watson'")
        if self.tail_extrapolation not in (
            "flat",
            "log_linear",
            "log_quadratic",
            "adaptive",
            "sv_slope",
            "cloud_slope",
        ):
            raise ValueError(
                "tail_extrapolation must be 'flat', 'log_linear', 'log_quadratic', 'adaptive', "
                "'sv_slope' or 'cloud_slope'"
            )
        if self.min_window < 0 or not 0.0 <= self.min_window_fraction < 0.5:
            raise ValueError("need min_window >= 0 and 0 <= min_window_fraction < 0.5")
        if not 0 <= self.quantile_clip < 0.5:
            raise ValueError("quantile_clip must lie in [0, 0.5)")
        if self.n_regression_points < 11:
            raise ValueError("n_regression_points too small")
        if self.leverage_std_span <= 0 or self.leverage_dk <= 0:
            raise ValueError("leverage grid settings must be positive")
        if not 0 < self.l_min < self.l_max:
            raise ValueError("need 0 < l_min < l_max")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass(frozen=True)
class SurfacePerturbation:
    """Additive perturbation layer on the implied surface (SPEC v2 §7.1): ``δσ(k, T)`` added to
    the implied vol of the base SSVI surface, reused by every risk bump and delta regime.

    ``kind`` and the YAML-safe ``params`` are hashed into the leverage cache key, so every
    recalibrating bump is a cache entry.  Kinds (built in :func:`volsto.market.surface.
    perturbed_surface`): ``parallel`` (size), ``tent`` (pillars, index, size), ``skew_tent``
    (pillars, index, slope: ``slope · k · tent``), ``curvature_tent`` (pillars, index, curv: ``curv
    · k² · tent``), ``rotation`` (size, t_min, k_cap: ``−size · 0.02/sqrt(max(T, t_min)) · k_cap
    tanh(k/k_cap)``, the desk rota of the shadow-rotation greek), ``shift_k`` (delta: ``σ(k + delta,
    T) − σ(k, T)``, the sticky-strike regime), ``atm_shift`` (delta, factor: ``factor · s_T ·
    delta`` with ``s_T`` the base ATM skew, the sticky-skew / sticky-local-vol regimes),
    ``total_variance`` (eps, t_lo, t_hi: the forward-variance bucket bump ``w → w + eps ∫_{bucket ∩
    [0,T]} ξ₀``), ``roll`` (dt: the surface held in (K, absolute expiry) seen ``dt`` later),
    ``table`` (ks, ts, values: bilinear ``δσ``), ``composite`` (items: list of perturbation
    mappings, summed).
    """

    kind: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.kind:
            raise ValueError("perturbation kind must be a non-empty string")


@dataclass(frozen=True)
class CalibrationSpec:
    """Everything that determines a calibrated leverage function (SPEC §4.3 cache key).

    The cache key hashes ``key_payload()``: market, surface, its perturbation layer (M5), model
    parameters, particle settings, the local-vol grid, and the parts of :class:`SimConfig` the
    kernel depends on (step schedule and scheme).  Pricing-only settings (``n_paths``, pricing
    seed, chunking) are excluded.  An eSSVI surface (``surface.rhos``, SPEC §13.2) enters the
    key through its pillar correlations; a plain SSVI surface hashes as it did before eSSVI
    configs existed (:func:`omitted_when_none`).
    """

    market: MarketConfig
    surface: SSVIConfig
    model: BergomiParams
    particle: ParticleConfig = field(default_factory=ParticleConfig)
    sim: SimConfig = field(default_factory=SimConfig)
    local_vol: LocalVolConfig | None = None
    perturbation: SurfacePerturbation | None = None

    def key_payload(self) -> dict[str, Any]:
        sim = self.sim
        market = to_mapping(self.market)
        market.pop("close", None)  # a fixing level, not a pricing input (MarketConfig)
        payload = {
            "market": market,
            "surface": to_mapping(self.surface),
            "model": to_mapping(self.model),
            "particle": to_mapping(self.particle),
            "local_vol": to_mapping(self.local_vol) if self.local_vol is not None else None,
            "schedule": to_mapping(sim.step_schedule),
            "scheme": to_mapping(sim.scheme),
        }
        if self.perturbation is not None:
            payload["perturbation"] = to_mapping(self.perturbation)
        return payload


# --------------------------------------------------------------------------------------------
# Local correlation model (SPEC §8.7, M12)
# --------------------------------------------------------------------------------------------

#: Values of :attr:`LocalCorrelationConfig.family`.
LC_FAMILIES: tuple[str, ...] = ("particle", "parametric", "constant")
#: ``ParticleConfig`` fields the local correlation calibration does not read (the leverage
#: grid and its clip bounds): left out of :meth:`LocalCorrelationSpec.key_payload`.
LC_UNUSED_PARTICLE_FIELDS: tuple[str, ...] = ("leverage_std_span", "leverage_dk", "l_min", "l_max")


@dataclass(frozen=True)
class SviSurfaceConfig:
    """An SVI-slice surface (:class:`volsto.market.svi_slices.SviSlices`): the slices'
    maturities, their raw SVI parameters ``(a, b, ρ, m, σ)`` and the surface's last maturity.

    ``record_keys``: provenance — the fit-record key of each slice when the parameters were read
    from records (SPEC §13.4 pattern); left out of every mapping while ``None`` and never part
    of a cache key (the parameters are)."""

    times: tuple[float, ...]
    params: tuple[tuple[float, float, float, float, float], ...]
    max_maturity: float
    record_keys: tuple[str, ...] | None = None

    OMIT_WHEN_NONE: ClassVar[frozenset[str]] = frozenset({"record_keys"})

    def __post_init__(self) -> None:
        if not self.times or len(self.times) != len(self.params):
            raise ValueError("times and params must be non-empty and of equal length")
        if any(t <= 0 for t in self.times) or any(np.diff(self.times) <= 0):
            raise ValueError("slice times must be positive and strictly increasing")
        if any(len(p) != 5 or not all(math.isfinite(x) for x in p) for p in self.params):
            raise ValueError("each slice has five finite SVI parameters (a, b, rho, m, sigma)")
        if not self.max_maturity >= self.times[-1]:
            raise ValueError("max_maturity must cover the last slice")
        if self.record_keys is not None and len(self.record_keys) != len(self.times):
            raise ValueError("one record key per slice")


@dataclass(frozen=True)
class ParametricLambdaConfig:
    """Settings of the two-parameter fit ``λ(t, k) = clip(λ0 − slope·k, 0, λ_max)``
    (:func:`volsto.calibration.local_correlation.calibrate_parametric_lambda`; the reference
    implementation's values).

    ``strikes``: the two fitted strikes as fractions of the basket forward — the
    at-the-money-forward straddle and the 90 % put.  ``maturity``: the fitted maturity (``None``:
    the calibration horizon).  ``n_paths``: paths of the fit's common random numbers.  ``h``:
    forward-difference steps of the Jacobian in ``(ρ0, c)``.  ``tol``: stopping tolerance on the
    implied-vol residuals (decimal: 0.0002 vol points).  ``maxit``: iterations at most.
    ``fd_iters``: iterations on which the Jacobian is recomputed before it is held."""

    strikes: tuple[float, ...] = (1.0, 0.9)
    maturity: float | None = None
    n_paths: int = 400_000
    h: tuple[float, float] = (0.004, 0.04)
    tol: float = 2e-6
    maxit: int = 15
    fd_iters: int = 2

    OMIT_WHEN_NONE: ClassVar[frozenset[str]] = frozenset({"maturity"})

    def __post_init__(self) -> None:
        if len(self.strikes) != 2 or self.strikes[0] != 1.0:
            raise ValueError("strikes must be (1.0, K): the ATM-forward straddle and one strike")
        if not (self.strikes[1] > 0 and self.strikes[1] != 1.0):
            raise ValueError("the second strike must be positive and differ from 1")
        if self.maturity is not None and self.maturity <= 0:
            raise ValueError("maturity must be positive")
        if self.n_paths <= 0 or self.n_paths % 2:
            raise ValueError("n_paths must be positive and even")
        if any(x <= 0 for x in self.h) or self.tol <= 0:
            raise ValueError("Jacobian steps and tolerance must be positive")
        if self.maxit < 1 or self.fd_iters < 1:
            raise ValueError("need maxit >= 1 and fd_iters >= 1")


@dataclass(frozen=True)
class LocalCorrelationConfig:
    """The local correlation model's settings (SPEC §8.7).

    Attributes:
        family: how ``λ`` is obtained — ``"particle"`` (the particle calibration to the whole
            index smile), ``"parametric"`` (two parameters fitted to two index vols) or
            ``"constant"`` (one constant fitted to the index at-the-money straddle).
        r_low: ``"equi"`` (the equicorrelation at ``rho_min``),
            ``"historical-scaled:<window>,<target>"`` or ``"matrix:<path>"``.
        r_high: ``"ones"`` or ``"matrix:<path>"``.
        rho_min: level of the ``"equi"`` ``R_low`` (0.02: the reference's floor).
        rho_max: cap of the equicorrelation ``ρ(λ)`` for ``"equi"`` — ``λ_max = (rho_max −
            rho_min)/(1 − rho_min)`` (0.98: the reference's cap).
        lambda_max: the cap for a non-``"equi"`` ``R_low`` (``None``: 1.0); must be left unset
            for ``"equi"``.
        mode: the basket state — ``"performance"`` (the study's convention) or ``"carry"``.
        particle: the particle settings; the leverage-only fields are not read.
        lambda_grid: the ``λ`` grid (the Dupire grid of the index target); ``None``: the shared
            local-vol grid of the names.
        clip_policy: ``"report"`` (clip and record) or ``"raise"`` (fail when the clipped mass
            of a slice exceeds ``max_clipped_mass``).
        max_clipped_mass: the threshold of ``"raise"`` and of the acceptance gates (1 %).
        lambda_tail: beyond the trusted quantiles, ``"regressions"`` (``λ*`` from the
            extrapolated regressions and the target, then clipped) or ``"flat"`` (``λ`` held at
            its values at the ends of the trusted range).
        target_average: ``"step"`` (every variance averaged over the step the row governs) or
            ``"point"`` (values at the step start; a diagnostic).
        arbitrage: ``"flag"`` (an SVI surface with a butterfly or calendar violation is priced
            and reported) or ``"raise"``.
        parametric: the settings of the parametric family.
    """

    family: str = "particle"
    r_low: str = "equi"
    r_high: str = "ones"
    rho_min: float = 0.02
    rho_max: float = 0.98
    lambda_max: float | None = None
    mode: str = "performance"
    particle: ParticleConfig = field(default_factory=ParticleConfig)
    lambda_grid: LocalVolConfig | None = None
    clip_policy: str = "report"
    max_clipped_mass: float = 0.01
    lambda_tail: str = "regressions"
    target_average: str = "step"
    arbitrage: str = "flag"
    parametric: ParametricLambdaConfig = field(default_factory=ParametricLambdaConfig)

    OMIT_WHEN_NONE: ClassVar[frozenset[str]] = frozenset({"lambda_max", "lambda_grid"})

    def __post_init__(self) -> None:
        # the specification strings are parsed by the family module (imported here, at call
        # time: volsto.multi imports this module)
        from volsto.multi.family import parse_r_high_spec, parse_r_low_spec

        if self.family not in LC_FAMILIES:
            raise ValueError(f"family must be one of {LC_FAMILIES}")
        low_kind, _ = parse_r_low_spec(self.r_low)
        parse_r_high_spec(self.r_high)
        if not 0.0 <= self.rho_min < self.rho_max <= 1.0:
            raise ValueError("need 0 <= rho_min < rho_max <= 1")
        if self.lambda_max is not None:
            if low_kind == "equi":
                raise ValueError("r_low = 'equi' sets the cap from rho_max: leave lambda_max unset")
            if not 0.0 < self.lambda_max <= 1.0:
                raise ValueError("lambda_max must lie in (0, 1]")
        if self.mode not in ("performance", "carry"):
            raise ValueError("mode must be 'performance' or 'carry'")
        if self.clip_policy not in ("report", "raise"):
            raise ValueError("clip_policy must be 'report' or 'raise'")
        if not 0.0 <= self.max_clipped_mass <= 1.0:
            raise ValueError("max_clipped_mass must lie in [0, 1]")
        if self.lambda_tail not in ("regressions", "flat"):
            raise ValueError("lambda_tail must be 'regressions' or 'flat'")
        if self.target_average not in ("step", "point"):
            raise ValueError("target_average must be 'step' or 'point'")
        if self.arbitrage not in ("flag", "raise"):
            raise ValueError("arbitrage must be 'flag' or 'raise'")
        if self.particle.tail_extrapolation == "sv_slope":
            raise ValueError("tail_extrapolation 'sv_slope' is the leverage's: not available here")


@dataclass(frozen=True)
class LocalCorrelationSpec:
    """Everything that determines a calibrated ``λ`` (the local correlation cache key, SPEC
    §8.7): the names and their index weights, one market (spot 1 and the curves that reproduce
    the listed forwards) and one SVI surface per name, the index target surface in its own
    forward moneyness, the model settings, the simulation settings (the step schedule and the
    scheme are keyed) and the shared local-vol grid.

    ``perturbations`` (one per name or ``None``) and ``index_perturbation`` are the additive
    surface bumps of the vegas; ``r_low_source`` the digest of the data behind a
    ``"historical-scaled"`` ``R_low`` (file SHA-256, window, end date) or the SHA-256 of a
    ``"matrix"`` file — the matrix itself is machine-dependent in its last bits and is never
    hashed (SPEC §13.3).  ``index_forward_ratios`` (``(T_e, F_I(T_e)/I_0)`` per listed index
    expiry) and ``label`` are for reports only and are not keyed.

    The calibration seed is ``lc.particle.seed`` (keyed); the pricing seed is ``sim.seed`` (not
    keyed)."""

    names: tuple[str, ...]
    weights: tuple[float, ...]
    markets: tuple[MarketConfig, ...]
    surfaces: tuple[SviSurfaceConfig, ...]
    index_surface: SviSurfaceConfig
    lc: LocalCorrelationConfig
    sim: SimConfig
    local_vol: LocalVolConfig
    perturbations: tuple[SurfacePerturbation | None, ...] | None = None
    index_perturbation: SurfacePerturbation | None = None
    r_low_source: str | None = None
    index_forward_ratios: tuple[tuple[float, float], ...] = ()
    label: str = ""

    OMIT_WHEN_NONE: ClassVar[frozenset[str]] = frozenset(
        {"perturbations", "index_perturbation", "r_low_source"}
    )

    def __post_init__(self) -> None:
        n = len(self.names)
        if n == 0 or len(set(self.names)) != n:
            raise ValueError("names must be non-empty and distinct")
        if not (len(self.weights) == len(self.markets) == len(self.surfaces) == n):
            raise ValueError("names, weights, markets and surfaces must have the same length")
        if self.perturbations is not None and len(self.perturbations) != n:
            raise ValueError("perturbations: one entry (or None) per name")
        if any(not math.isfinite(w) or w < 0 for w in self.weights):
            raise ValueError("weights must be finite and non-negative")
        if abs(math.fsum(self.weights) - 1.0) > 1e-12:
            raise ValueError(f"weights must sum to 1 (sum = {math.fsum(self.weights)!r})")
        horizon = self.lc.particle.horizon
        schedule = self.sim.step_schedule
        starts = (0.0, *schedule.breaks)
        max_dt = max(d for s, d in zip(starts, schedule.dts) if s < horizon)
        if self.local_vol.t_max < horizon + max_dt - 1e-12:
            raise ValueError(
                f"local_vol.t_max = {self.local_vol.t_max:g} must reach the calibration horizon "
                f"plus one step ({horizon:g} + {max_dt:g}): the last row averages over it"
            )
        if self.lc.lambda_grid is not None and self.lc.lambda_grid.t_max < horizon + max_dt - 1e-12:
            raise ValueError("lc.lambda_grid.t_max must reach the horizon plus one step")
        for name, s in (*zip(self.names, self.surfaces), ("the index", self.index_surface)):
            if s.max_maturity < self.local_vol.t_max:
                raise ValueError(
                    f"{name}: the surface ends at {s.max_maturity:g}y, before local_vol.t_max = "
                    f"{self.local_vol.t_max:g}y"
                )
        if self.r_low_source is None and not self.lc.r_low == "equi":
            raise ValueError(
                f"r_low = {self.lc.r_low!r} needs r_low_source (the digest of its data or file)"
            )

    @property
    def n_names(self) -> int:
        return len(self.names)

    def key_payload(self) -> dict[str, Any]:
        """What the cache key hashes: everything except ``label`` and ``index_forward_ratios``,
        the slices' ``record_keys`` (provenance), the pricing-only fields of ``sim`` (as in
        :meth:`CalibrationSpec.key_payload`: ``n_paths``, ``seed``, ``chunk_size``,
        ``antithetic``, ``chunk_memory_mb``, ``record_all_steps``) and the particle fields the
        calibration does not read (:data:`LC_UNUSED_PARTICLE_FIELDS`)."""

        def surface(s: SviSurfaceConfig) -> dict[str, Any]:
            m: dict[str, Any] = to_mapping(s)
            m.pop("record_keys", None)
            return m

        def market(m: MarketConfig) -> dict[str, Any]:
            out: dict[str, Any] = to_mapping(m)
            out.pop("close", None)
            return out

        lc: dict[str, Any] = to_mapping(self.lc)
        for name in LC_UNUSED_PARTICLE_FIELDS:
            lc["particle"].pop(name)
        payload: dict[str, Any] = {
            "names": list(self.names),
            "weights": [float(w) for w in self.weights],
            "markets": [market(m) for m in self.markets],
            "surfaces": [surface(s) for s in self.surfaces],
            "index_surface": surface(self.index_surface),
            "lc": lc,
            "schedule": to_mapping(self.sim.step_schedule),
            "scheme": to_mapping(self.sim.scheme),
            "local_vol": to_mapping(self.local_vol),
        }
        if self.perturbations is not None:
            payload["perturbations"] = [
                None if p is None else to_mapping(p) for p in self.perturbations
            ]
        if self.index_perturbation is not None:
            payload["index_perturbation"] = to_mapping(self.index_perturbation)
        if self.r_low_source is not None:
            payload["r_low_source"] = self.r_low_source
        return payload
