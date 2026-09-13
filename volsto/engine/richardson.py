"""Step-refinement diagnostics under common random numbers (owner amendment after M1).

``refinement_study`` prices a product at ``dt, dt/2, …`` on the *same* Brownian paths
(:class:`~volsto.engine.rng.CoarsenedDraws`), so successive differences ``D_i = P(dt_i) −
P(dt_{i+1})`` are low-noise.  For a weak order-``p`` scheme ``D_i / D_{i+1} → 2^p`` and the
Talay–Tubaro extrapolant ``R = (2^p P(dt/2) − P(dt)) / (2^p − 1)`` removes the leading bias.
Checked by ``tests/test_scheme.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from volsto.config import SimConfig
from volsto.engine.grid import TimeGrid
from volsto.engine.mc import MonteCarlo, PriceResult, summarize
from volsto.engine.rng import CoarsenedDraws, GaussianDraws

if TYPE_CHECKING:
    from volsto.models.base import Model
    from volsto.products.base import Product

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RefinementResult:
    """Prices at successive halvings of the step, CRN differences and the extrapolant."""

    dts: FloatArray
    prices: list[PriceResult]
    differences: list[PriceResult]
    richardson: PriceResult
    order_estimate: float | None

    def __repr__(self) -> str:
        lines = [f"  dt={dt:.6g}: {p}" for dt, p in zip(self.dts, self.prices)]
        lines += [f"  D{i} = {d}" for i, d in enumerate(self.differences)]
        lines.append(f"  richardson = {self.richardson}, order ≈ {self.order_estimate}")
        return "RefinementResult(\n" + "\n".join(lines) + "\n)"


def refinement_study(
    cfg: SimConfig,
    product: Product,
    model: Model,
    dt: float,
    *,
    levels: int = 3,
    order: int = 1,
) -> RefinementResult:
    """Price ``product`` at ``dt / 2^i`` (``i < levels``) on Brownian-consistent paths.

    All fixing times must be multiples of ``dt`` (uniform grids only).  The extrapolant uses the
    two finest levels with the assumed weak ``order``; ``order_estimate = log2(D_0 / D_1)`` when
    three or more levels are available and ``D_1`` is resolved (≥ 3 stderr).
    """
    if levels < 2:
        raise ValueError("need at least two refinement levels")
    fixings = product.fixing_times
    n_coarse = np.round(fixings / dt)
    if not np.allclose(n_coarse * dt, fixings, rtol=0, atol=1e-9):
        raise ValueError("fixing times must be multiples of dt")
    mc = MonteCarlo(cfg)
    grids = [TimeGrid.build(fixings, dt / 2**i) for i in range(levels)]
    n_fine = grids[-1].n_steps
    fine = GaussianDraws(cfg.seed, cfg.n_paths, n_fine, model.n_brownians, cfg.antithetic)
    prices: list[PriceResult] = []
    payoffs: list[FloatArray] = []
    for i, grid in enumerate(grids):
        factor = 2 ** (levels - 1 - i)
        if grid.n_steps * factor != n_fine:
            raise ValueError("grid refinement is not dyadic; use a uniform dt")
        draws = fine if factor == 1 else CoarsenedDraws(fine, factor)
        res = mc.price(product, model, grid=grid, draws=draws, keep_payoffs=True)
        assert res.payoffs is not None
        prices.append(res)
        payoffs.append(res.payoffs)
    diffs = [summarize(payoffs[i] - payoffs[i + 1], cfg.antithetic) for i in range(levels - 1)]
    w = 2.0**order
    rich = summarize((w * payoffs[-1] - payoffs[-2]) / (w - 1.0), cfg.antithetic)
    est: float | None = None
    if (
        levels >= 3
        and abs(diffs[1].mean) > 3 * diffs[1].stderr
        and diffs[0].mean * diffs[1].mean > 0
    ):
        est = float(np.log2(diffs[0].mean / diffs[1].mean))
    return RefinementResult(np.array([dt / 2**i for i in range(levels)]), prices, diffs, rich, est)
