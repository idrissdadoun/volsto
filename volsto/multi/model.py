"""The multi-asset model: factor-free single-asset models (Black–Scholes, local vol) driven by
correlated Brownians (:class:`~volsto.multi.draws.CorrelatedDraws`).

Each asset is simulated by its own kernel on the shared grid with its view of the correlated
draws, so every single-asset convention (step schedule, scheme, the Dupire grid, the forward
curve) applies unchanged; the correlation is the only new input.  A stochastic-volatility
component per asset is not supported (its factor Brownians would need the cross-asset vol
correlations the study does not model); ``n_brownians == 1`` and ``n_factors == 0`` are
required.  Checked by ``tests/test_multi.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.config import SchemeConfig
from volsto.engine.grid import TimeGrid
from volsto.models.base import Model
from volsto.multi.draws import CorrelatedDraws, check_correlation
from volsto.multi.paths import MultiPathSet

FloatArray = NDArray[np.float64]


class MultiAssetModel:
    """``models[i]`` drives asset ``i``; ``correlation`` is the Brownian correlation."""

    def __init__(
        self,
        models: Sequence[Model],
        correlation: ArrayLike,
        names: Sequence[str] | None = None,
    ) -> None:
        if not models:
            raise ValueError("at least one asset")
        for i, m in enumerate(models):
            if m.n_brownians != 1 or m.n_factors != 0:
                raise ValueError(
                    f"asset {i}: the multi-asset layer drives factor-free single-asset models "
                    f"(Black-Scholes, local vol); got {type(m).__name__} with "
                    f"{m.n_brownians} Brownians and {m.n_factors} factors"
                )
        self.models = list(models)
        self.correlation = check_correlation(correlation)
        if self.correlation.shape[0] != len(self.models):
            raise ValueError("correlation must be n_assets x n_assets")
        self.names = (
            tuple(str(n) for n in names)
            if names is not None
            else tuple(f"asset{i}" for i in range(len(self.models)))
        )
        if len(self.names) != len(self.models):
            raise ValueError("one name per asset")

    @property
    def n_assets(self) -> int:
        return len(self.models)

    @property
    def spots(self) -> FloatArray:
        return np.array([m.spot for m in self.models])

    def forwards(self, T: float) -> FloatArray:
        return np.array([float(m.forward_curve.forward(T)) for m in self.models])

    def required_times(self) -> FloatArray:
        parts = [np.asarray(m.required_times(), dtype=np.float64) for m in self.models]
        return np.unique(np.concatenate(parts)) if parts else np.empty(0)

    def draws_for(
        self, grid: TimeGrid, seed: int, n_paths: int, antithetic: bool = True
    ) -> CorrelatedDraws:
        return CorrelatedDraws(seed, n_paths, grid.n_steps, self.correlation, antithetic)

    def simulate_chunk(
        self,
        grid: TimeGrid,
        draws: CorrelatedDraws,
        p0: int,
        p1: int,
        scheme: SchemeConfig,
        *,
        step_block: int = 64,
    ) -> MultiPathSet:
        if draws.n_assets != self.n_assets:
            raise ValueError("draws and model disagree on the number of assets")
        parts = tuple(
            m.simulate_chunk(grid, draws.asset(i), p0, p1, scheme, step_block=step_block)  # type: ignore[arg-type]
            for i, m in enumerate(self.models)
        )
        return MultiPathSet(parts, self.names)

    def with_correlation(self, correlation: ArrayLike) -> MultiAssetModel:
        return MultiAssetModel(self.models, correlation, self.names)

    def bump(self, asset: int, **kwargs: Any) -> MultiAssetModel:
        """Asset ``asset``'s model bumped (its own ``bump``), the others shared."""
        models = list(self.models)
        models[asset] = models[asset].bump(**kwargs)
        return MultiAssetModel(models, self.correlation, self.names)

    def __repr__(self) -> str:
        return f"MultiAssetModel({list(self.names)}, {[repr(m) for m in self.models]})"


__all__ = ["MultiAssetModel"]
