"""Build market objects from YAML configs (SPEC §11: explicit, validated configs).

A surface config is read from a file only by :func:`load_surface_config` (which honours an
``essvi`` section) and a surface is built from a config only by
:func:`volsto.market.surface.surface_from_config` (SPEC §13.2); :func:`snapshot_spec` puts a
snapshot's market and surface on a calibration spec.  ``tests/test_surface_config.py`` walks the
package for any other reader.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path

import yaml

from volsto.config import (
    CalibrationSpec,
    ConfigError,
    LocalVolModelConfig,
    MarketConfig,
    SSVIConfig,
    from_mapping,
    load_yaml,
)
from volsto.market.curves import DiscountCurve, ForwardCurve
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import SSVISurface, surface_from_config

#: Keys of a snapshot's ``essvi`` section (the importer writes the pillar correlations only).
ESSVI_SECTION_KEYS: frozenset[str] = frozenset({"rhos"})


def load_market(path: str | Path, section: str = "market") -> ForwardCurve:
    """Spot + rate + dividend curves from the ``market`` section of a surface file."""
    return ForwardCurve.from_config(load_yaml(path, MarketConfig, section=section))


def load_surface_config(path: str | Path) -> SSVIConfig:
    """The surface config of a file with an ``ssvi`` section — the one reader of a surface config
    from a file (M10 Part 3, SPEC §13.2).

    An ``essvi`` section (``rhos``: one correlation per ATM pillar, written by the importer for
    its default eSSVI fit, ``volsto.market.import_hdn.snapshot_config``) sets
    :attr:`~volsto.config.SSVIConfig.rhos`; the result then builds the eSSVI surface everywhere
    (leverage cache, risk engine, loaders) and hashes the pillar correlations into the cache key.
    Without it the config is the plain SSVI, whose cache key is unchanged.  Strict: the section
    holds exactly ``rhos`` (validated with the config: one per pillar, ``|ρ| < 1``), and ``rhos``
    may not also appear inside the ``ssvi`` section.  Checked by
    ``tests/test_surface_config.py::test_essvi_snapshot_config_carries_rhos``."""
    p = Path(path)
    cfg = load_yaml(p, SSVIConfig, section="ssvi")
    with p.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not (isinstance(raw, Mapping) and "essvi" in raw):
        return cfg
    section = raw["essvi"]
    if not isinstance(section, Mapping) or set(section) != ESSVI_SECTION_KEYS:
        raise ConfigError(f"{p}: the essvi section must hold exactly {sorted(ESSVI_SECTION_KEYS)}")
    ssvi = raw["ssvi"]
    if "rhos" in ssvi:
        raise ConfigError(f"{p}: rhos given in both the ssvi and the essvi sections")
    return from_mapping(SSVIConfig, {**ssvi, "rhos": section["rhos"]}, path=f"{p}:ssvi+essvi")


def snapshot_spec(base: CalibrationSpec, path: str | Path) -> CalibrationSpec:
    """``base`` with the market and the surface of a snapshot file (``market`` + ``ssvi``
    sections, an optional ``essvi`` section): the calibration spec of a dated surface.  The
    model, particle, simulation, local-vol and perturbation settings stay those of ``base``."""
    return dataclasses.replace(
        base,
        market=load_yaml(path, MarketConfig, section="market"),
        surface=load_surface_config(path),
    )


def load_ssvi_surface(path: str | Path) -> SSVISurface:
    """The surface of a file with ``market`` and ``ssvi`` sections: an ``ESSVISurface`` when the
    file has an ``essvi`` section (:func:`load_surface_config`), the plain SSVI otherwise."""
    fc = load_market(path)
    return surface_from_config(load_surface_config(path), fc, fc.rate_curve)


def load_local_vol(
    surface_path: str | Path, model_path: str | Path | None = None
) -> LocalVolSurface:
    """Dupire local vol from an SSVI surface file and an optional local-vol grid config."""
    surface = load_ssvi_surface(surface_path)
    cfg = load_yaml(model_path, LocalVolModelConfig) if model_path else LocalVolModelConfig()
    return LocalVolSurface.from_implied(surface, cfg.grid)


def default_discount(fc: ForwardCurve) -> DiscountCurve:
    return fc.rate_curve
