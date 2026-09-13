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
from typing import Any, get_args, get_origin, get_type_hints

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


def to_mapping(obj: Any) -> Any:
    """Recursively convert a dataclass (with numpy scalars/arrays) to plain YAML-safe types."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_mapping(getattr(obj, f.name)) for f in fields(obj)}
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
    (owner amendment after M1) is 1/1460 below 3m, 1/365 up to 2y and 1/250 beyond; the Monte
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
            weighted ``(1 − η, η)`` between the two points.
        pc_eta: the diffusion weight ``η`` of the predictor–corrector.
        weak_order2: Platen's explicit weak order-2 scheme (Kloeden–Platen eq. 15.1.3) with the
            coefficients at ``t_n`` and ``t_{n+1}``; excludes ``predictor_corrector`` and ignores
            time averaging.  Three extra variance lookups per step.
        local_var_time_eval: when ``local_var_time_average`` is off, evaluate the variance at the
            step ``"start"`` (plain log-Euler) or at the ``"midpoint"`` in time (diagnostic).
    """

    local_var_time_average: bool = True
    predictor_corrector: bool = False
    pc_eta: float = 0.5
    weak_order2: bool = True
    local_var_time_eval: str = "start"

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
        local_var_time_average, predictor_corrector, pc_eta, weak_order2, local_var_time_eval:
            see :class:`SchemeConfig`.
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
    """Spot plus rate and dividend-yield curves (SPEC §2.1)."""

    spot: float
    rate_curve: CurveConfig
    dividend_curve: CurveConfig

    def __post_init__(self) -> None:
        if self.spot <= 0:
            raise ValueError("spot must be positive")


@dataclass(frozen=True)
class SSVIConfig:
    """Gatheral–Jacquier power-law SSVI (SPEC §2.2).

    ``atm_maturities`` / ``atm_vols`` give the ATM implied-vol term structure; θ_T = σ_ATM(T)² T
    is interpolated linearly in T (flat forward variance) between pillars.
    """

    atm_maturities: tuple[float, ...]
    atm_vols: tuple[float, ...]
    rho: float
    eta: float
    gamma: float
    max_maturity: float = 10.0

    def __post_init__(self) -> None:
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
    strike by 0.5% of variance), hence the 1201-point default over ``±1.5``.
    """

    t_min: float = 1.0 / 365.0
    t_max: float = 3.0
    n_t: int = 400
    k_min: float = -1.5
    k_max: float = 1.5
    n_k: int = 1201
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
