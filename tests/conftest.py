"""Shared fixtures: reference surface, flat curves, fast simulation settings."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from volsto.config import SimConfig
from volsto.market import DiscountCurve, ForwardCurve, LocalVolSurface, SSVISurface
from volsto.market.loaders import load_ssvi_surface

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SURFACE = ROOT / "configs" / "surfaces" / "reference_ssvi.yaml"


@pytest.fixture(scope="session")
def forward_curve() -> ForwardCurve:
    return ForwardCurve.flat(100.0, 0.02, 0.01)


@pytest.fixture(scope="session")
def discount(forward_curve: ForwardCurve) -> DiscountCurve:
    return forward_curve.rate_curve


@pytest.fixture(scope="session")
def ssvi() -> SSVISurface:
    return load_ssvi_surface(REFERENCE_SURFACE)


@pytest.fixture(scope="session")
def local_vol(ssvi: SSVISurface) -> LocalVolSurface:
    return LocalVolSurface.from_implied(ssvi)


@pytest.fixture
def fast_sim() -> SimConfig:
    return SimConfig(n_paths=40_000, dt_max=1.0 / 100.0, chunk_size=20_000, antithetic=True, seed=7)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)
