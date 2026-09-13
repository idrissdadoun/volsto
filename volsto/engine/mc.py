"""Monte Carlo pricer (SPEC §5): chunked simulation, antithetics, control variates, stderr.

Every price is a :class:`PriceResult` carrying its standard error (SPEC §11: the library never
returns a bare float for a Monte Carlo quantity).  With antithetics the standard error is
computed on pair-averaged payoffs, which are independent.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from volsto.config import SimConfig
from volsto.engine.cv import ControlVariate, CVReport, apply_controls
from volsto.engine.grid import TimeGrid
from volsto.engine.paths import PathSet
from volsto.engine.rng import GaussianDraws

if TYPE_CHECKING:
    from volsto.models.base import Model
    from volsto.products.base import Product

FloatArray = NDArray[np.float64]
log = logging.getLogger(__name__)


@dataclass(frozen=True)
class PriceResult:
    """Monte Carlo estimate with its standard error.

    Attributes:
        mean: estimate of the discounted expectation.
        stderr: standard error of ``mean`` (on independent samples; antithetic pairs averaged).
        n_paths: paths simulated.
        n_samples: independent samples behind ``stderr``.
        payoffs: optional per-path discounted payoffs (raw, before control-variate adjustment).
        cv: control-variate report when controls were used.
    """

    mean: float
    stderr: float
    n_paths: int
    n_samples: int
    payoffs: FloatArray | None = None
    cv: CVReport | None = None

    def ci(self, z: float = 1.96) -> tuple[float, float]:
        return (self.mean - z * self.stderr, self.mean + z * self.stderr)

    def __repr__(self) -> str:
        return f"PriceResult({self.mean:.8g} ± {self.stderr:.3g}, n_paths={self.n_paths})"

    def __str__(self) -> str:
        return f"{self.mean:.8g} ± {self.stderr:.3g}"


def _pair_average(x: FloatArray, antithetic: bool) -> FloatArray:
    if antithetic:
        return 0.5 * (x[0::2] + x[1::2])
    return x


def summarize(
    payoffs: FloatArray,
    antithetic: bool,
    controls: Sequence[ControlVariate] = (),
    control_payoffs: FloatArray | None = None,
    keep_payoffs: bool = False,
) -> PriceResult:
    """Mean/stderr of discounted payoffs, with optional control-variate adjustment."""
    samples = _pair_average(payoffs, antithetic)
    report: CVReport | None = None
    if controls:
        if control_payoffs is None:
            raise ValueError("control payoffs are required when controls are given")
        cs = np.column_stack(
            [_pair_average(control_payoffs[:, j], antithetic) for j in range(len(controls))]
        )
        samples, report = apply_controls(samples, cs, np.array([c.expectation for c in controls]))
    n = samples.size
    mean = float(samples.mean())
    stderr = float(samples.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
    return PriceResult(
        mean, stderr, int(payoffs.size), int(n), payoffs if keep_payoffs else None, report
    )


class MonteCarlo:
    """Runs a model on a grid in chunks and evaluates products on each chunk's :class:`PathSet`."""

    def __init__(self, cfg: SimConfig) -> None:
        self.cfg = cfg

    # -- helpers -----------------------------------------------------------------------------

    def build_grid(self, products: Sequence[Product], model: Model) -> TimeGrid:
        fixings = np.unique(np.concatenate([p.fixing_times for p in products]))
        return TimeGrid.build(fixings, self.cfg.dt_max, calibration_grid=model.required_times())

    def draws_for(self, grid: TimeGrid, model: Model, seed: int | None = None) -> GaussianDraws:
        return GaussianDraws(
            self.cfg.seed if seed is None else seed,
            self.cfg.n_paths,
            grid.n_steps,
            model.n_brownians,
            self.cfg.antithetic,
        )

    def _chunks(self) -> list[tuple[int, int]]:
        n, c = self.cfg.n_paths, self.cfg.chunk_size
        return [(p0, min(p0 + c, n)) for p0 in range(0, n, c)]

    # -- simulation --------------------------------------------------------------------------

    def simulate(self, model: Model, grid: TimeGrid, draws: GaussianDraws | None = None) -> PathSet:
        """Full :class:`PathSet` (all chunks concatenated) for analytics with modest ``n_paths``."""
        draws = draws or self.draws_for(grid, model)
        parts = [model.simulate_chunk(grid, draws, p0, p1) for p0, p1 in self._chunks()]
        return PathSet.concat(parts)

    # -- pricing -----------------------------------------------------------------------------

    def price_many(
        self,
        products: Sequence[Product],
        model: Model,
        *,
        grid: TimeGrid | None = None,
        draws: GaussianDraws | None = None,
        controls: Sequence[ControlVariate] = (),
        keep_payoffs: bool = False,
    ) -> list[PriceResult]:
        """Price several products on the same paths (same grid, same draws)."""
        if not products:
            return []
        all_products: list[Product] = list(products) + [c.product for c in controls]
        grid = grid or self.build_grid(all_products, model)
        draws = draws or self.draws_for(grid, model)
        if draws.n_paths != self.cfg.n_paths or draws.n_steps != grid.n_steps:
            raise ValueError("draws do not match the configuration / grid")
        n = self.cfg.n_paths
        payoffs = np.empty((n, len(products)))
        cpay = np.empty((n, len(controls)))
        idx = grid.fixing_index
        for p0, p1 in self._chunks():
            paths = model.simulate_chunk(grid, draws, p0, p1)
            for j, prod in enumerate(products):
                payoffs[p0:p1, j] = prod.payoff(paths, idx)
            for j, ctrl in enumerate(controls):
                cpay[p0:p1, j] = ctrl.payoff(paths, idx)
        results = []
        for j in range(len(products)):
            results.append(
                summarize(
                    payoffs[:, j],
                    self.cfg.antithetic,
                    controls,
                    cpay if controls else None,
                    keep_payoffs,
                )
            )
        return results

    def price(
        self,
        product: Product,
        model: Model,
        *,
        grid: TimeGrid | None = None,
        draws: GaussianDraws | None = None,
        controls: Sequence[ControlVariate] = (),
        keep_payoffs: bool = False,
    ) -> PriceResult:
        """Discounted price of ``product`` under ``model`` with standard error."""
        return self.price_many(
            [product], model, grid=grid, draws=draws, controls=controls, keep_payoffs=keep_payoffs
        )[0]
