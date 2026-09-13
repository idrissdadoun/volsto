from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from volsto.config import (
    ConfigError,
    CurveConfig,
    MarketConfig,
    SimConfig,
    SSVIConfig,
    dump_yaml,
    from_mapping,
    load_yaml,
)


@dataclass(frozen=True)
class _Inner:
    a: float
    b: int = 2


@dataclass(frozen=True)
class _Outer:
    name: str
    inner: _Inner
    flags: tuple[bool, ...] = ()
    maybe: float | None = None


def test_from_mapping_nested_and_defaults() -> None:
    out = from_mapping(_Outer, {"name": "x", "inner": {"a": 1}, "flags": [True, False]})
    assert out.inner == _Inner(1.0, 2)
    assert out.flags == (True, False)
    assert out.maybe is None


def test_unknown_key_rejected() -> None:
    with pytest.raises(ConfigError, match="unknown keys"):
        from_mapping(_Outer, {"name": "x", "inner": {"a": 1}, "typo": 3})


def test_missing_key_rejected() -> None:
    with pytest.raises(ConfigError, match="missing required key"):
        from_mapping(_Outer, {"inner": {"a": 1}})


def test_wrong_type_rejected() -> None:
    with pytest.raises(ConfigError, match="expected a number"):
        from_mapping(_Outer, {"name": "x", "inner": {"a": "one"}})
    with pytest.raises(ConfigError, match="expected an integer"):
        from_mapping(_Outer, {"name": "x", "inner": {"a": 1, "b": 2.5}})


def test_yaml_round_trip(tmp_path: Path) -> None:
    cfg = SSVIConfig((0.5, 1.0), (0.2, 0.21), -0.7, 1.0, 0.5)
    p = tmp_path / "s.yaml"
    dump_yaml(cfg, p)
    assert load_yaml(p, SSVIConfig) == cfg
    with pytest.raises(ConfigError, match="missing section"):
        load_yaml(p, SSVIConfig, section="nope")


def test_validation_errors() -> None:
    with pytest.raises(ValueError, match="even"):
        SimConfig(n_paths=1001)
    with pytest.raises(ValueError):
        CurveConfig(times=(1.0, 0.5), rates=(0.0, 0.0))
    with pytest.raises(ValueError):
        MarketConfig(
            spot=-1.0, rate_curve=CurveConfig.flat(0.0), dividend_curve=CurveConfig.flat(0.0)
        )
    with pytest.raises(ConfigError):
        from_mapping(
            SSVIConfig,
            {"atm_maturities": [1.0], "atm_vols": [0.2], "rho": 1.5, "eta": 1, "gamma": 0.5},
        )
