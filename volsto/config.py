"""Configuration dataclasses and strict YAML loading (SPEC §11).

Every model / engine / market parameter set is a frozen dataclass.  Loading from YAML is strict:
unknown keys, missing keys and wrong scalar types raise :class:`ConfigError` with the offending
path.  Model parameters never have defaults (no silent defaults); engine settings carry the
documented SPEC defaults.
"""

from __future__ import annotations

import dataclasses
import math
import types
import typing
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import numpy as np
import yaml

T = TypeVar("T")


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


def from_mapping(cls: type[T], data: Mapping[str, Any], *, path: str = "") -> T:
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


def load_yaml(path: str | Path, cls: type[T], *, section: str | None = None) -> T:
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
# Engine configuration (defaults documented in SPEC §3.1 / §5)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SimConfig:
    """Monte Carlo engine settings (SPEC §3.1, §5).

    Attributes:
        n_paths: number of paths (SPEC default 2·10⁵).  Must be even when antithetic.
        dt_max: maximum simulation step in years (SPEC default 1/365).
        chunk_size: paths simulated per block to bound memory (SPEC default 5·10⁴).
        antithetic: use antithetic pairs (path 2i+1 uses the negated normals of path 2i).
        seed: base seed of the CRN-keyed PCG64 stream (SPEC §5).
    """

    n_paths: int = 200_000
    dt_max: float = 1.0 / 365.0
    chunk_size: int = 50_000
    antithetic: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if self.antithetic and self.n_paths % 2:
            raise ValueError("n_paths must be even when antithetic=True")
        if self.dt_max <= 0:
            raise ValueError("dt_max must be positive")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.antithetic and self.chunk_size % 2:
            raise ValueError("chunk_size must be even when antithetic=True")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


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
