"""The Monte Carlo loop over a multi-asset model (:class:`~volsto.multi.model.MultiModel`: the
constant-correlation :class:`~volsto.multi.model.MultiAssetModel` or the local correlation
model of SPEC §8.7): the single-asset engine's chunking, antithetics and standard errors
(:func:`volsto.engine.mc.summarize`) applied to :class:`~volsto.multi.products.
MultiAssetProduct` payoffs on :class:`~volsto.multi.paths.MultiPathSet` chunks.  The chunk size
is the single-asset one for ``n_assets`` times the record columns (the memory of one chunk
scales with the assets).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from volsto.config import SimConfig
from volsto.engine.grid import TimeGrid
from volsto.engine.mc import PriceResult, summarize
from volsto.multi.model import MultiModel
from volsto.multi.paths import MultiPathSet
from volsto.multi.products import MultiAssetProduct


class MultiAssetMonteCarlo:
    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg

    def build_grid(self, products: Sequence[MultiAssetProduct], model: MultiModel) -> TimeGrid:
        fixings = np.unique(np.concatenate([p.fixing_times for p in products]))
        record_all = self.cfg.record_all_steps or any(p.requires_all_steps for p in products)
        return TimeGrid.build(
            fixings,
            self.cfg.dt_max,
            calibration_grid=model.required_times(),
            record_all_steps=record_all,
        )

    def draws_for(self, grid: TimeGrid, model: MultiModel, seed: int | None = None) -> Any:
        """The model's own draws object (``CorrelatedDraws`` or ``LocalCorrelationDraws``)."""
        return model.draws_for(
            grid, self.cfg.seed if seed is None else seed, self.cfg.n_paths, self.cfg.antithetic
        )

    def _chunks(self, grid: TimeGrid, model: MultiModel) -> list[tuple[int, int]]:
        return self.cfg.chunk_ranges(grid.n_records * model.n_assets, 0)

    def simulate(self, model: MultiModel, grid: TimeGrid, draws: Any = None) -> MultiPathSet:
        draws = draws or self.draws_for(grid, model)
        parts = [
            model.simulate_chunk(grid, draws, p0, p1, self.cfg.scheme)
            for p0, p1 in self._chunks(grid, model)
        ]
        return MultiPathSet.concat(parts)

    def price_many(
        self,
        products: Sequence[MultiAssetProduct],
        model: MultiModel,
        *,
        grid: TimeGrid | None = None,
        draws: Any = None,
        keep_payoffs: bool = False,
    ) -> list[PriceResult]:
        """Every product on the same paths (common random numbers across the products)."""
        if not products:
            return []
        grid = grid or self.build_grid(products, model)
        draws = draws or self.draws_for(grid, model)
        if draws.n_paths != self.cfg.n_paths or draws.n_steps != grid.n_steps:
            raise ValueError("draws do not match the configuration / grid")
        payoffs = np.empty((self.cfg.n_paths, len(products)))
        idx = grid.fixing_index
        for p0, p1 in self._chunks(grid, model):
            paths = model.simulate_chunk(grid, draws, p0, p1, self.cfg.scheme)
            for j, prod in enumerate(products):
                payoffs[p0:p1, j] = prod.payoff(paths, idx)
        return [
            summarize(payoffs[:, j], self.cfg.antithetic, keep_payoffs=keep_payoffs)
            for j in range(len(products))
        ]

    def price(
        self,
        product: MultiAssetProduct,
        model: MultiModel,
        *,
        grid: TimeGrid | None = None,
        draws: Any = None,
        keep_payoffs: bool = False,
    ) -> PriceResult:
        return self.price_many([product], model, grid=grid, draws=draws, keep_payoffs=keep_payoffs)[
            0
        ]


__all__ = ["MultiAssetMonteCarlo"]
