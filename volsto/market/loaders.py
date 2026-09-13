"""Build market objects from YAML configs (SPEC §11: explicit, validated configs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from volsto.config import LocalVolModelConfig, MarketConfig, SSVIConfig, load_yaml
from volsto.market.curves import DiscountCurve, ForwardCurve
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import ESSVISurface, SSVISurface


def load_market(path: str | Path, section: str = "market") -> ForwardCurve:
    """Spot + rate + dividend curves from the ``market`` section of a surface file."""
    return ForwardCurve.from_config(load_yaml(path, MarketConfig, section=section))


def load_ssvi_surface(path: str | Path) -> SSVISurface:
    """SSVI surface from a file with ``market`` and ``ssvi`` sections; an ``essvi`` section
    (``rhos`` per ATM pillar, written by the importer's ``--essvi``) gives an ``ESSVISurface``."""
    fc = load_market(path)
    cfg = load_yaml(path, SSVIConfig, section="ssvi")
    with Path(path).open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if isinstance(raw, dict) and "essvi" in raw:
        rhos = [float(r) for r in raw["essvi"]["rhos"]]
        t = np.asarray(cfg.atm_maturities)
        v = np.asarray(cfg.atm_vols)
        return ESSVISurface(
            t, v * v * t, rhos, cfg.eta, cfg.gamma, fc, fc.rate_curve, max_maturity=cfg.max_maturity
        )
    return SSVISurface.from_config(cfg, fc, fc.rate_curve)


def load_local_vol(
    surface_path: str | Path, model_path: str | Path | None = None
) -> LocalVolSurface:
    """Dupire local vol from an SSVI surface file and an optional local-vol grid config."""
    surface = load_ssvi_surface(surface_path)
    cfg = load_yaml(model_path, LocalVolModelConfig) if model_path else LocalVolModelConfig()
    return LocalVolSurface.from_implied(surface, cfg.grid)


def default_discount(fc: ForwardCurve) -> DiscountCurve:
    return fc.rate_curve
