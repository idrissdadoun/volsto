"""The multi-asset path container: one :class:`~volsto.engine.paths.PathSet` per asset on a
shared record grid, with the accessors the basket and dispersion products use (performances
``S_t / S_0 − 1`` against the initial spots or against given reference levels)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.engine.paths import PathSet

FloatArray = NDArray[np.float64]


@dataclass
class MultiPathSet:
    """``assets[i]`` is asset ``i``'s paths; all share ``times``."""

    assets: tuple[PathSet, ...]
    names: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.assets:
            raise ValueError("at least one asset")
        t = self.assets[0].times
        n = self.assets[0].n_paths
        for p in self.assets[1:]:
            if not np.array_equal(p.times, t) or p.n_paths != n:
                raise ValueError("every asset must share the record times and the path count")
        if len(self.names) != len(self.assets):
            raise ValueError("one name per asset")

    @property
    def times(self) -> FloatArray:
        return self.assets[0].times

    @property
    def n_paths(self) -> int:
        return self.assets[0].n_paths

    @property
    def n_cols(self) -> int:
        return self.assets[0].n_cols

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    def spot_at(self, col: int | ArrayLike, asset: int) -> FloatArray:
        return self.assets[asset].spot_at(col)

    def log_spot_at(self, col: int | ArrayLike, asset: int) -> FloatArray:
        return self.assets[asset].log_spot_at(col)

    def spots(self, col: int) -> FloatArray:
        """``(n_paths, n_assets)`` spots at a column."""
        return np.column_stack([p.spot_at(col) for p in self.assets])

    def performances(self, col: int, reference: Sequence[float] | None = None) -> FloatArray:
        """``S_col / ref − 1`` per asset, ``(n_paths, n_assets)``; ``reference`` defaults to the
        column-0 spots (the initial levels)."""
        s = self.spots(col)
        if reference is None:
            ref = self.spots(0)
        else:
            r = np.asarray(reference, dtype=np.float64)
            if r.shape != (self.n_assets,):
                raise ValueError("one reference level per asset")
            ref = np.broadcast_to(r, s.shape)
        return np.asarray(s / ref - 1.0, dtype=np.float64)

    def basket_level(self, col: int, weights: ArrayLike) -> FloatArray:
        """``Σ w_i S_i(col)/S_i(0)`` per path (a performance basket, level 1 at inception)."""
        w = np.asarray(weights, dtype=np.float64)
        return np.asarray((self.spots(col) / self.spots(0)) @ w, dtype=np.float64)

    @staticmethod
    def concat(parts: Sequence[MultiPathSet]) -> MultiPathSet:
        if not parts:
            raise ValueError("nothing to concatenate")
        n = parts[0].n_assets
        assets = tuple(PathSet.concat([p.assets[i] for p in parts]) for i in range(n))
        return MultiPathSet(assets, parts[0].names)

    def __repr__(self) -> str:
        return (
            f"MultiPathSet(n_assets={self.n_assets}, n_paths={self.n_paths}, "
            f"n_cols={self.n_cols}, names={list(self.names)})"
        )


__all__ = ["MultiPathSet"]
