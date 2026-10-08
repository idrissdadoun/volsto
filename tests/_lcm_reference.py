"""The reference regression of the local correlation model (test S4, SPEC §8.7): the world of
the stand-alone reference implementation (``outputs/interview/lcm_reference/``, a run of
2026-10-07 on four Dow dates) rebuilt on the library, from the fixtures
``tests/golden/lcm_reference/<tag>.json``.

The reference: thirty names with the two-parameter local vol ``σ_i(x) = σ_i0·clip(1 − b_i·x,
0.25, 4)`` in ``x = ln S_i`` (zero rates, spots 1), log-Euler at the step start, 63 daily steps,
the equicorrelation ``ρ_t = clip(ρ0 − c·ln B_t, 0.02, 0.98)`` read at the step start (LC) or a
constant ``ρ`` (CC), 4·10⁵ antithetic paths.  On the library: a ``LocalVol`` per name with the
``t``-independent table ``σ²(k) = (σ0·min(max(1 − b·k, 0.25), 4))²`` on the model's grid, the
family ``equi(0.02, 0.98)``, ``λ = ParametricLambda.from_rho(ρ0, c, 0.02, 0.98)`` (exact map),
the scheme with every refinement switched off (mode 0: ``ln S += −½σ²Δt + σ√Δt·Z``), a uniform
grid of ``T/steps``, performance mode (with zero carry the two modes coincide).

The reference's random numbers are its own (float32 normals of another generator): the
comparison is statistical, within three combined standard errors.  Not library code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from volsto.config import LocalVolConfig, SimConfig
from volsto.engine.grid import TimeGrid
from volsto.market.curves import DiscountCurve, ForwardCurve
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import ImpliedSurface
from volsto.models.localvol import LocalVol
from volsto.multi.family import CorrelationFamily
from volsto.multi.lc_function import LocalCorrelationFunction, ParametricLambda
from volsto.multi.lc_model import BasketSpec, LocalCorrelationModel

FIXTURES = Path(__file__).resolve().parent / "golden" / "lcm_reference"
TAGS = ("today", "typical", "steep", "typical_alt")
RHO_MIN, RHO_MAX = 0.02, 0.98
LV_FLOOR, LV_CAP = 0.25, 4.0
#: the pricing seed of the regression (the configuration example's)
SEED = 2024
N_PATHS = 400_000


def load_fixture(tag: str) -> dict[str, Any]:
    doc: dict[str, Any] = json.loads((FIXTURES / f"{tag}.json").read_text())
    return doc


def reference_grid(T: float) -> LocalVolConfig:
    """The model's shared grid for a horizon up to 1y (SPEC §8.7): only ``k`` matters here, the
    tables do not depend on ``t``."""
    return LocalVolConfig(t_min=1 / 365, t_max=T + 0.02, n_t=2, k_min=-2.0, k_max=2.0, n_k=1601)


def two_point_local_vol(sig0: float, b: float, cfg: LocalVolConfig) -> LocalVol:
    """A name of the reference: ``σ²(k) = (σ0·min(max(1 − b·k, 0.25), 4))²`` at every ``t``, on
    the grid of ``cfg``; zero rates, spot 1 (so ``k = ln S``)."""
    fc = ForwardCurve(1.0, DiscountCurve.flat(0.0), DiscountCurve.flat(0.0))
    k = np.linspace(cfg.k_min, cfg.k_max, cfg.n_k)
    vol = sig0 * np.minimum(np.maximum(1.0 - b * k, LV_FLOOR), LV_CAP)
    t = np.array([cfg.t_min, cfg.t_max])
    return LocalVol(LocalVolSurface(t, k, np.tile(vol * vol, (2, 1)), fc), fc)


def reference_sim(fx: dict[str, Any], n_paths: int = N_PATHS, seed: int = SEED) -> SimConfig:
    """The reference's scheme and grid: log-Euler at the step start, ``T/steps`` uniform."""
    return SimConfig(
        n_paths=n_paths,
        dt_max=fx["T"] / fx["steps"],
        chunk_size=20_000,
        seed=seed,
        weak_order2=False,
        predictor_corrector=False,
        local_var_time_average=False,
        local_var_time_eval="start",
    )


@dataclass
class ReferenceWorld:
    """The library's rebuild of one reference date."""

    fx: dict[str, Any]
    models: list[LocalVol]
    weights: np.ndarray
    family: CorrelationFamily
    basket: BasketSpec
    sim: SimConfig
    grid: TimeGrid
    lc: LocalCorrelationModel
    cc: LocalCorrelationModel


def reference_world(tag: str, n_paths: int = N_PATHS, seed: int = SEED) -> ReferenceWorld:
    fx = load_fixture(tag)
    T = float(fx["T"])
    cfg = reference_grid(T)
    single = fx["single"]
    models = [
        two_point_local_vol(s, b, cfg) for s, b in zip(single["sig0"], single["b"], strict=True)
    ]
    w = np.array(single["w"])
    family = CorrelationFamily.equi(len(models), RHO_MIN, RHO_MAX)
    basket = BasketSpec(w, "performance", [m.forward_curve for m in models])
    sim = reference_sim(fx, n_paths, seed)
    grid = TimeGrid.build([T], sim.dt_max)
    assert grid.n_steps == fx["steps"]
    k_grid = models[0].local_vol.k_grid
    lam_lc = LocalCorrelationFunction.parametric(
        ParametricLambda.from_rho(fx["rho0"], fx["c"], RHO_MIN, RHO_MAX), grid.times, k_grid
    )
    lam_cc = LocalCorrelationFunction.constant(
        (fx["rho_cc"] - RHO_MIN) / (1.0 - RHO_MIN), grid.times, k_grid
    )
    lc = LocalCorrelationModel(models, family, lam_lc, basket, single["tickers"])
    return ReferenceWorld(fx, models, w, family, basket, sim, grid, lc, lc.with_lambda(lam_cc))


def simulate_terminal(
    world: ReferenceWorld, model: LocalCorrelationModel
) -> tuple[np.ndarray, np.ndarray]:
    """``(D, B)`` per path at ``T``: the dispersion ``Σ w_i |R_i − R̄|`` and the basket level
    ``Σ w_i S_i(T)`` (the same draws for every model of the world: common random numbers)."""
    sim, grid = world.sim, world.grid
    draws = model.draws_for(grid, sim.seed, sim.n_paths, sim.antithetic)
    d = np.empty(sim.n_paths)
    b = np.empty(sim.n_paths)
    w = world.weights
    for p0, p1 in sim.chunk_ranges(grid.n_records * model.n_assets, 0):
        paths = model.simulate_chunk(grid, draws, p0, p1, sim.scheme)
        r = paths.performances(1)
        rb = r @ w
        d[p0:p1] = np.abs(r - rb[:, None]) @ w
        b[p0:p1] = 1.0 + rb
    return d, b


def pair_mean(x: np.ndarray) -> tuple[float, float]:
    """Mean and standard error on antithetic pair means."""
    y = 0.5 * (x[0::2] + x[1::2])
    return float(y.mean()), float(y.std(ddof=1) / np.sqrt(y.size))


def pair_ratio(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """``E[a]/E[b]`` with the delta-method standard error on pair means (``a`` and ``b`` on the
    same pairs): the reference's ``ratio_pm``."""
    ya, yb = 0.5 * (a[0::2] + a[1::2]), 0.5 * (b[0::2] + b[1::2])
    r = float(ya.mean() / yb.mean())
    u = (ya - r * yb) / yb.mean()
    return r, float(u.std(ddof=1) / np.sqrt(u.size))


class TwoPointIndex(ImpliedSurface):
    """The index target of the reference as a surface: the implied vol linear in ``k`` through
    the two targets — the at-the-money-forward vol and the vol at 90 % of the forward — at every
    maturity (only these two points are read by the parametric fit)."""

    def __init__(self, atm: float, vol_90: float) -> None:
        fc = ForwardCurve(1.0, DiscountCurve.flat(0.0), DiscountCurve.flat(0.0))
        super().__init__(fc, fc.rate_curve, 10.0)
        self.atm, self.slope = float(atm), float((vol_90 - atm) / np.log(0.9))

    def total_variance(self, k: Any, T: Any) -> Any:
        k_, t_ = np.broadcast_arrays(np.asarray(k, dtype=float), np.asarray(T, dtype=float))
        vol = self.atm + self.slope * k_
        return vol * vol * t_
