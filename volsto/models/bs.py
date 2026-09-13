"""Black–Scholes model (SPEC §3.2): flat volatility, same log-Euler kernel as local vol."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from volsto.engine.grid import TimeGrid
from volsto.engine.paths import PathSet
from volsto.engine.rng import GaussianDraws
from volsto.market.curves import ForwardCurve
from volsto.market.dupire import LocalVolSurface
from volsto.models.base import Model, ModelState
from volsto.models.localvol import LocalVol

FloatArray = NDArray[np.float64]


class BlackScholes(Model):
    """``d ln S = (r − q − σ²/2) dt + σ dW``; the log-Euler step is exact for this model."""

    n_factors = 0
    n_brownians = 1

    def __init__(self, vol: float, forward_curve: ForwardCurve) -> None:
        if not np.isfinite(vol) or vol <= 0:
            raise ValueError("vol must be positive")
        self.vol = float(vol)
        self.forward_curve = forward_curve
        self._lv = LocalVol(LocalVolSurface.flat(self.vol, forward_curve), forward_curve)

    def initial_state(self, n_paths: int) -> ModelState:
        return ModelState(
            0.0,
            np.full(n_paths, np.log(self.spot)),
            np.full(n_paths, self.vol**2),
            np.empty((n_paths, 0)),
        )

    def instantaneous_variance(self, state: ModelState) -> FloatArray:
        return np.full(state.n_paths, self.vol**2)

    def bump(self, **kwargs: Any) -> BlackScholes:
        """Supported: ``vol``, ``spot``."""
        vol = self.vol
        fc = self.forward_curve
        for key, val in kwargs.items():
            if key == "vol":
                vol = float(val)
            elif key == "spot":
                fc = fc.with_spot(float(val))
            else:
                raise ValueError(f"BlackScholes.bump: unknown parameter {key!r}")
        return BlackScholes(vol, fc)

    def simulate_chunk(
        self, grid: TimeGrid, draws: GaussianDraws, p0: int, p1: int, *, step_block: int = 64
    ) -> PathSet:
        return self._lv.simulate_chunk(grid, draws, p0, p1, step_block=step_block)

    def __repr__(self) -> str:
        return f"BlackScholes(vol={self.vol}, spot={self.spot})"
