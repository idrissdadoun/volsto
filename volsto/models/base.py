"""Model interface (SPEC §3.1).

A model owns its :class:`~volsto.market.curves.ForwardCurve`, declares how many Brownians it
consumes per step and how many factors it records, and simulates *chunks* of paths on a
:class:`~volsto.engine.grid.TimeGrid`.  All spot dynamics are log-Euler with the variance frozen
over each step (SPEC §3.1); factor stepping is model specific.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volsto.config import SchemeConfig, SimConfig
from volsto.engine.grid import TimeGrid
from volsto.engine.paths import PathSet
from volsto.engine.rng import GaussianDraws
from volsto.market.curves import ForwardCurve

FloatArray = NDArray[np.float64]


@dataclass
class ModelState:
    """State of ``n_paths`` paths at time ``t``."""

    t: float
    log_spot: FloatArray
    variance: FloatArray
    factors: FloatArray  # (n_paths, n_factors)

    @property
    def n_paths(self) -> int:
        return int(self.log_spot.shape[0])


class Model(ABC):
    """Abstract single-underlying model."""

    n_factors: int
    n_brownians: int
    forward_curve: ForwardCurve

    @property
    def spot(self) -> float:
        return self.forward_curve.spot

    @abstractmethod
    def initial_state(self, n_paths: int) -> ModelState:
        """State at ``t = 0`` replicated over ``n_paths``."""

    @abstractmethod
    def instantaneous_variance(self, state: ModelState) -> FloatArray:
        """Instantaneous variance of ``d ln S`` in ``state`` (leverage included for LSV)."""

    @abstractmethod
    def bump(self, **kwargs: Any) -> Model:
        """New model with the given parameters changed (curves shared)."""

    @abstractmethod
    def simulate_chunk(
        self,
        grid: TimeGrid,
        draws: GaussianDraws,
        p0: int,
        p1: int,
        scheme: SchemeConfig,
        *,
        step_block: int = 64,
    ) -> PathSet:
        """Simulate paths ``[p0, p1)`` on ``grid`` under ``scheme``, recording at the columns."""

    def required_times(self) -> FloatArray:
        """Extra times the grid must contain (e.g. leverage slices).  Default: none."""
        return np.empty(0)

    def simulate(self, grid: TimeGrid, rng: GaussianDraws, cfg: SimConfig) -> PathSet:
        """All ``cfg.n_paths`` paths, simulated in chunks of ``cfg.chunk_size``."""
        if rng.n_paths != cfg.n_paths or rng.n_steps != grid.n_steps:
            raise ValueError("draws do not match cfg / grid")
        parts = [
            self.simulate_chunk(grid, rng, p0, p1, cfg.scheme)
            for p0, p1 in cfg.chunk_ranges(grid.n_records, self.n_factors)
        ]
        return PathSet.concat(parts)
