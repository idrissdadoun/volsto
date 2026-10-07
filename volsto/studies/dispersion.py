"""The dispersion study (SPEC §8.5; the owner's question of 2026-10-03): a framework deciding,
from parameters and market expectations, whether to buy a palladium — the call on dispersion
``(Σ_i w_i |r_i − r_B| − K)⁺`` — rather than single-name straddles against a basket straddle.
Stage 1, the computation (``scripts/dispersion_study.py``), whose outputs the report reads.

**Worlds** (:class:`World`).  A basket of ``n`` names with vols ``σ_i``, a correlation (a
constant ``ρ`` or a matrix), a horizon ``T``, and three departures from the constant-
correlation lognormal that the question turns on:

* *local correlation* ``λ``: the correlation moves with the basket, ``ρ_t = clip(ρ − λ (B_t −
  1))`` — correlation rises when the basket falls (the index skew's cross-asset side);
* *uncertain correlation* ``ρ_sd``: the path's correlation is drawn at inception from ``ρ ±
  ρ_sd`` (a two-point mixture: the vol-of-correlation the call on dispersion is convex in);
* *idiosyncratic events* ``(p_J, μ_J, σ_J)``: each name jumps once over the horizon with
  probability ``p_J`` by ``exp(μ_J + σ_J Z) − 1`` on its terminal performance (earnings, M&A:
  dispersion without basket vol).

The constant-correlation world with per-name local vol (skew) runs on the library's multi-
asset layer (:mod:`volsto.multi`, Cholesky mixing); the local / uncertain-correlation worlds
run on the one-factor representation ``W_i = √ρ_t Z_0 + √(1 − ρ_t) Z_i`` (equicorrelation,
exact per path and step) in :func:`simulate_one_factor`, Black–Scholes per name, daily steps.
Both record the terminal performances and the realised variances.

**The trades** (:func:`trade_values`; per unit of basket notional, basket weights ``w``):
``palladium`` (the dispersion forward, ``K = 0``), ``palladium call`` (``K`` a fraction of the
market forward dispersion), ``basket straddle`` ``|r_B|``, ``single straddles`` ``Σ w_i
|r_i|``, ``straddle package`` ``Σ w_i |r_i| − λ_B |r_B|`` with ``λ_B = 1`` (basket weights) and
``λ_B`` premium-neutral (the package costs nothing in the market world), and ``variance
dispersion`` ``Σ w_i RV_i − RV_B`` (the delta-hedged analogue).

**Parts** (each writes ``<out>/<part>.csv``):

* ``sensitivities`` — the market world's prices (Monte Carlo and the Gaussian closed forms)
  and their sensitivities: ``∂/∂ρ`` (common random numbers across correlations), ``∂/∂σ``
  (all names), the convexity in ``ρ``, the response to ``λ``, ``ρ_sd``, jumps, per-name
  skew, the basket size and the horizon, and the palladium call across strikes.
* ``expectations`` — the framework's grid: the trades bought at the market world's prices and
  realised under expectation worlds that differ by the realised correlation (``ρ_impl ±``),
  the realised vols (``σ_impl ×``), the local-correlation ``λ``, the correlation uncertainty
  and the idiosyncratic-event probability; per cell the expected P&L per unit premium, the
  P&L std and the loss probability of each trade, and the best trade.
* ``history`` — the basket's realised statistics 2006–2026 (:func:`volsto.studies.
  history_stats.dispersion_statistics`): realised dispersion against the straddle package and
  the basket move, realised against Cboe-implied correlation (``COR3M``), by VIX tercile and
  by the sign and size of the basket move; and the ex-post P&L of each trade entered monthly
  at proxy market prices (Gaussian closed forms at ``COR3M`` and the trailing realised vols
  times :data:`IMPLIED_PREMIUM`).

**Market world.** Single-name implied vols are not in the repository (SPX options only): the
market world takes each name's trailing 1y realised vol at the anchor date times
:data:`IMPLIED_PREMIUM` (1.15, the single-name implied-realised ratio stated as an input, not
measured) and the Cboe ``COR3M`` of the anchor date as the implied correlation — both printed
in ``setup.csv``; the framework's conclusions are stated relative to these inputs.

Checked by ``tests/test_dispersion.py``.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.config import SimConfig, SSVIConfig
from volsto.market.curves import ForwardCurve
from volsto.market.surface import surface_from_config
from volsto.models.bs import BlackScholes
from volsto.models.localvol import LocalVol
from volsto.multi import (
    MultiAssetModel,
    MultiAssetMonteCarlo,
    basket_vol,
    gaussian_palladium_call,
    gaussian_palladium_forward,
    gaussian_straddle_dispersion,
)
from volsto.multi.draws import constant_correlation
from volsto.studies.history_stats import (
    HISTORY_DIR,
    TRADING_DAYS,
    dispersion_statistics,
    load_closes,
    load_series,
    regime_bins,
)

log = logging.getLogger(__name__)
FloatArray = NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "dispersion"
PARTS: tuple[str, ...] = ("sensitivities", "expectations", "history")
BASKET: tuple[str, ...] = ("AAPL", "MSFT", "AMZN", "NVDA", "JPM", "XOM", "JNJ", "PG", "HD", "UNH")
ANCHOR_DATE = "2022-12-30"
#: the single-name implied-over-realised ratio of the market world (an input, not measured)
IMPLIED_PREMIUM = 1.15
#: the call on dispersion's strike as a fraction of the market world's forward dispersion
CALL_STRIKE_FRACTION = 0.8
HORIZON_DAYS = 63
TRADES: tuple[str, ...] = (
    "palladium",
    "palladium call",
    "basket straddle",
    "single straddles",
    "straddle package",
    "straddle package (premium-neutral)",
    "variance dispersion",
)


# --------------------------------------------------------------------------------------------
# worlds
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class World:
    """Module docstring.  ``correlation`` is a constant ``ρ`` (one-factor worlds allowed) or a
    full matrix (constant-correlation world only); ``skew`` is the per-name SSVI ``(rho, eta,
    gamma)`` of a local-vol world (``None``: Black–Scholes)."""

    vols: tuple[float, ...]
    correlation: float | tuple[tuple[float, ...], ...]
    T: float = HORIZON_DAYS / TRADING_DAYS
    weights: tuple[float, ...] | None = None
    local_corr: float = 0.0
    corr_sd: float = 0.0
    jump_prob: float = 0.0
    jump_mean: float = 0.0
    jump_sd: float = 0.0
    skew: tuple[float, float, float] | None = None
    label: str = "world"

    @property
    def n(self) -> int:
        return len(self.vols)

    @property
    def w(self) -> FloatArray:
        if self.weights is None:
            return np.ones(self.n) / self.n
        return np.asarray(self.weights, dtype=np.float64)

    @property
    def matrix(self) -> FloatArray:
        if isinstance(self.correlation, float | int):
            return constant_correlation(self.n, float(self.correlation))
        return np.asarray(self.correlation, dtype=np.float64)

    @property
    def rho(self) -> float:
        """The constant correlation (the mean off-diagonal entry for a matrix)."""
        if isinstance(self.correlation, float | int):
            return float(self.correlation)
        from volsto.multi import pairwise_mean_correlation

        return pairwise_mean_correlation(self.matrix, self.w)

    @property
    def one_factor(self) -> bool:
        return self.local_corr != 0.0 or self.corr_sd != 0.0 or self.jump_prob != 0.0

    def replace(self, **changes: Any) -> World:
        return dataclasses.replace(self, **changes)


@dataclass
class Simulated:
    """Terminal performances ``(n_paths, n)`` and annualised realised variances of the names
    ``(n_paths, n)`` and of the basket ``(n_paths,)``."""

    perf: FloatArray
    rv_names: FloatArray
    rv_basket: FloatArray
    world: World

    @property
    def n_paths(self) -> int:
        return int(self.perf.shape[0])


def simulate_one_factor(
    world: World, n_paths: int, seed: int, steps: int | None = None
) -> Simulated:
    """Black–Scholes names on the one-factor correlation ``W_i = √ρ_t Z_0 + √(1−ρ_t) Z_i``
    with the local / uncertain correlation and the jump overlay of the module docstring; daily
    steps (``steps`` defaults to ``round(T · 252)``); antithetic on the Brownian draws."""
    if not isinstance(world.correlation, float | int):
        raise ValueError("the one-factor simulator needs a constant correlation")
    n, w, T = world.n, world.w, world.T
    m = round(T * TRADING_DAYS) if steps is None else int(steps)
    dt = T / m
    rng = np.random.default_rng(seed)
    half = (n_paths + 1) // 2
    sig = np.asarray(world.vols, dtype=np.float64)
    rho0 = float(world.correlation)
    # the path's correlation level: ρ ± ρ_sd with equal probability
    if world.corr_sd > 0:
        signs = rng.choice([-1.0, 1.0], size=half)
        rho_path = np.repeat(rho0 + world.corr_sd * signs, 2)[:n_paths]
    else:
        rho_path = np.full(n_paths, rho0)
    ls = np.zeros((n_paths, n))
    sq = np.zeros((n_paths, n))
    level = np.ones(n_paths)
    sq_b = np.zeros(n_paths)
    for _ in range(m):
        z = rng.standard_normal((half, n + 1))
        z = np.concatenate((z, -z), axis=0)[:n_paths]
        rho_t = np.clip(rho_path - world.local_corr * (level - 1.0), 0.0, 0.999)
        wi = np.sqrt(rho_t)[:, None] * z[:, :1] + np.sqrt(1.0 - rho_t)[:, None] * z[:, 1:]
        dx = -0.5 * sig * sig * dt + sig * np.sqrt(dt) * wi
        ls += dx
        sq += dx * dx
        new_level = np.exp(ls) @ w
        lb = np.log(new_level / level)
        sq_b += lb * lb
        level = new_level
    perf = np.exp(ls) - 1.0
    if world.jump_prob > 0:
        hit = rng.random((n_paths, n)) < world.jump_prob
        j = np.exp(world.jump_mean + world.jump_sd * rng.standard_normal((n_paths, n))) - 1.0
        perf = (1.0 + perf) * (1.0 + hit * j) - 1.0
        # the jump's return enters the names' realised variance (one daily return)
        lj = np.log1p(hit * j)
        sq += lj * lj
        rb_j = np.log(((1.0 + perf) @ w) / level)
        sq_b += rb_j * rb_j
    scale = 1.0 / T  # Σ (Δ ln S)² over the horizon, annualised
    return Simulated(perf, scale * sq, scale * sq_b, world)


def simulate_library(world: World, cfg: SimConfig) -> Simulated:
    """The constant-correlation world on the library's multi-asset layer: Black–Scholes or
    local-vol names (per-name SSVI skew: ``atm`` the name's vol, ``world.skew`` its ``(rho,
    eta, gamma)``) with Cholesky-mixed draws, daily fixings."""
    from volsto.config import LocalVolConfig
    from volsto.market.dupire import LocalVolSurface
    from volsto.multi.products import VarianceDispersion
    from volsto.products.base import daily_schedule

    n, T = world.n, world.T
    fcs = [ForwardCurve.flat(100.0, 0.0, 0.0) for _ in range(n)]
    models: list[Any] = []
    for v, fc in zip(world.vols, fcs, strict=True):
        if world.skew is None:
            models.append(BlackScholes(v, fc))
        else:
            r, eta, gam = world.skew
            ts = (1 / 12, 0.25, 0.5, 1.0, 2.0)
            cfg_s = SSVIConfig(ts, (v,) * len(ts), r, eta, gam, max_maturity=3.0)
            surf = surface_from_config(cfg_s, fc, fc.rate_curve)
            lv = LocalVolSurface.from_implied(
                surf, LocalVolConfig(t_max=2.0, n_t=120, n_k=1201, k_min=-2.0, k_max=2.0)
            )
            models.append(LocalVol(lv, fc))
    mm = MultiAssetModel(models, world.matrix, names=[f"n{i}" for i in range(n)])
    mc = MultiAssetMonteCarlo(cfg)
    sched = daily_schedule(T, TRADING_DAYS)
    vd = VarianceDispersion(world.w, sched, 0.0, fcs[0].rate_curve)
    grid = mc.build_grid([vd], mm)
    paths = mc.simulate(mm, grid)
    idx = grid.fixing_index
    perf = paths.performances(idx[T])
    cols = idx.indices(sched)
    rv = np.column_stack([p.realised_variance_fixings(cols) for p in paths.assets]) / T
    levels = np.column_stack([paths.basket_level(int(c), world.w) for c in cols])
    lb = np.diff(np.log(levels), axis=1)
    rv_b = np.sum(lb * lb, axis=1) / T
    return Simulated(np.asarray(perf), np.asarray(rv), np.asarray(rv_b), world)


def simulate(world: World, n_paths: int, seed: int) -> Simulated:
    """One-factor simulator for the worlds that need it, the library's otherwise."""
    if world.one_factor:
        return simulate_one_factor(world, n_paths, seed)
    cfg = SimConfig(n_paths=n_paths, chunk_size=min(n_paths, 10_000), seed=seed, dt_max=1 / 252)
    return simulate_library(world, cfg)


# --------------------------------------------------------------------------------------------
# the trades
# --------------------------------------------------------------------------------------------


def trade_payoffs(
    sim: Simulated, w: FloatArray, *, call_strike: float, package_scale: float
) -> dict[str, FloatArray]:
    """Per-path undiscounted payoffs of the trades (module docstring)."""
    r = sim.perf
    rb = r @ w
    d = np.abs(r - rb[:, None]) @ w
    singles = np.abs(r) @ w
    basket = np.abs(rb)
    return {
        "palladium": d,
        "palladium call": np.maximum(d - call_strike, 0.0),
        "basket straddle": basket,
        "single straddles": singles,
        "straddle package": singles - basket,
        "straddle package (premium-neutral)": singles - package_scale * basket,
        "variance dispersion": sim.rv_names @ w - sim.rv_basket,
    }


def trade_values(
    sim: Simulated, *, call_strike: float, package_scale: float
) -> dict[str, tuple[float, float]]:
    """Mean and standard error of each trade's payoff (antithetic pairs averaged)."""
    pay = trade_payoffs(sim, sim.world.w, call_strike=call_strike, package_scale=package_scale)
    out = {}
    for k, v in pay.items():
        x = 0.5 * (v[0::2] + v[1::2]) if v.size % 2 == 0 else v
        out[k] = (float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(x.size)))
    return out


def gaussian_values(world: World, call_strike: float, package_scale: float) -> dict[str, float]:
    """The Gaussian closed forms of the trades' forwards (the variance dispersion exactly:
    ``Σ w σ_i² − σ_B²``)."""
    v, c, w, T = world.vols, world.matrix, world.w, world.T
    sb = basket_vol(v, c, w)
    s = np.asarray(v)
    a = np.sqrt(2.0 * T / np.pi)
    return {
        "palladium": gaussian_palladium_forward(v, c, w, T),
        "palladium call": gaussian_palladium_call(v, c, w, T, call_strike),
        "basket straddle": a * sb,
        "single straddles": a * float(np.sum(w * s)),
        "straddle package": gaussian_straddle_dispersion(v, c, w, T),
        "straddle package (premium-neutral)": gaussian_straddle_dispersion(
            v, c, w, T, package_scale
        ),
        "variance dispersion": float(np.sum(w * s * s) - sb * sb),
    }


# --------------------------------------------------------------------------------------------
# the market world from the data
# --------------------------------------------------------------------------------------------


def market_world(
    history: Path = HISTORY_DIR, anchor: str = ANCHOR_DATE
) -> tuple[World, dict[str, Any]]:
    """The market world of the module docstring: trailing 1y realised vols at the anchor times
    :data:`IMPLIED_PREMIUM`, the Cboe ``COR3M`` of the anchor as the correlation, 3m horizon."""
    closes = load_closes(BASKET, history)
    past = closes[closes.index <= pd.Timestamp(anchor)]
    logp = pd.DataFrame(
        np.log(past.to_numpy(dtype=np.float64)), index=past.index, columns=past.columns
    )
    lr = logp.diff().dropna().iloc[-TRADING_DAYS:]
    rv = lr.std(ddof=1).to_numpy(dtype=np.float64) * np.sqrt(TRADING_DAYS)
    corr_realised = float(np.mean(np.corrcoef(lr.to_numpy().T)[np.triu_indices(len(BASKET), 1)]))
    cor3m = load_series("COR3M", history)
    rho_impl = float(cor3m[cor3m.index <= pd.Timestamp(anchor)].iloc[-1]) / 100.0
    vols = tuple(float(x) for x in IMPLIED_PREMIUM * rv)
    world = World(vols, rho_impl, label="market")
    setup = {
        "anchor": anchor,
        "names": list(BASKET),
        "realised_vol_1y": [float(x) for x in rv],
        "implied_premium": IMPLIED_PREMIUM,
        "implied_vols": list(vols),
        "corr_realised_1y": corr_realised,
        "cor3m_implied": rho_impl,
        "horizon_days": HORIZON_DAYS,
    }
    return world, setup


# --------------------------------------------------------------------------------------------
# parts
# --------------------------------------------------------------------------------------------


@dataclass
class DispersionConfig:
    out: Path = DEFAULT_OUT
    history: Path = HISTORY_DIR
    n_paths: int = 100_000
    seed: int = 7
    anchor: str = ANCHOR_DATE


@dataclass
class Environment:
    cfg: DispersionConfig
    market: World = field(init=False)
    setup: dict[str, Any] = field(init=False)
    call_strike: float = field(init=False)
    package_scale: float = field(init=False)

    def __post_init__(self) -> None:
        self.market, self.setup = market_world(self.cfg.history, self.cfg.anchor)
        g = gaussian_values(self.market, 0.0, 1.0)
        self.call_strike = CALL_STRIKE_FRACTION * g["palladium"]
        # premium-neutral package: the basket leg scaled so the package costs nothing
        self.package_scale = g["single straddles"] / g["basket straddle"]
        self.setup["call_strike"] = self.call_strike
        self.setup["package_scale"] = self.package_scale

    def values(self, world: World, seed_offset: int = 0) -> dict[str, tuple[float, float]]:
        sim = simulate(world, self.cfg.n_paths, self.cfg.seed + seed_offset)
        return trade_values(sim, call_strike=self.call_strike, package_scale=self.package_scale)

    def payoffs(self, world: World, seed_offset: int = 0) -> dict[str, FloatArray]:
        sim = simulate(world, self.cfg.n_paths, self.cfg.seed + seed_offset)
        return trade_payoffs(
            sim, world.w, call_strike=self.call_strike, package_scale=self.package_scale
        )


def _rows(
    label: str,
    vals: Mapping[str, tuple[float, float]],
    gauss: Mapping[str, float] | None,
    **tags: Any,
) -> list[dict[str, Any]]:
    rows = []
    for t in TRADES:
        m, se = vals[t]
        rows.append(
            {
                "world": label,
                "trade": t,
                "value": m,
                "stderr": se,
                "gaussian": (gauss or {}).get(t, np.nan),
                **tags,
            }
        )
    return rows


def run_sensitivities(env: Environment) -> pd.DataFrame:
    """Part ``sensitivities`` (module docstring)."""
    mk = env.market
    rows: list[dict[str, Any]] = []
    base = env.values(mk)
    rows += _rows(
        "market", base, gaussian_values(mk, env.call_strike, env.package_scale), axis="base", x=0.0
    )
    # correlation (CRN: the library world shares the independent streams; the one-factor
    # simulator shares the generator seed)
    for d in (-0.2, -0.1, 0.1, 0.2):
        w = mk.replace(
            correlation=float(np.clip(mk.rho + d, 0.0, 0.99)), label=f"rho {mk.rho + d:+.2f}"
        )
        rows += _rows(
            w.label,
            env.values(w),
            gaussian_values(w, env.call_strike, env.package_scale),
            axis="rho",
            x=mk.rho + d,
        )
    for f in (0.8, 0.9, 1.1, 1.2):
        w = mk.replace(vols=tuple(f * v for v in mk.vols), label=f"vols x{f:g}")
        rows += _rows(
            w.label,
            env.values(w),
            gaussian_values(w, env.call_strike, env.package_scale),
            axis="vol_scale",
            x=f,
        )
    for lam in (1.0, 2.0, 4.0):
        w = mk.replace(local_corr=lam, label=f"local corr lambda {lam:g}")
        rows += _rows(w.label, env.values(w), None, axis="local_corr", x=lam)
    for sd in (0.1, 0.2, 0.3):
        w = mk.replace(corr_sd=sd, label=f"corr sd {sd:g}")
        rows += _rows(w.label, env.values(w), None, axis="corr_sd", x=sd)
    for p in (0.1, 0.2, 0.4):
        w = mk.replace(jump_prob=p, jump_mean=0.0, jump_sd=0.10, label=f"jumps p {p:g} (sd 10%)")
        rows += _rows(w.label, env.values(w), None, axis="jump_prob", x=p)
    for sk in (-0.3, -0.6):
        w = mk.replace(skew=(sk, 1.0, 0.5), label=f"skew rho {sk:g}")
        rows += _rows(w.label, env.values(w), None, axis="skew", x=sk)
    for T in (1 / 12, 0.5, 1.0):
        w = mk.replace(T=T, label=f"T {T:g}")
        rows += _rows(
            w.label,
            env.values(w),
            gaussian_values(w, env.call_strike, env.package_scale),
            axis="T",
            x=T,
        )
    # basket size: the first n names (heterogeneity kept)
    for n in (3, 5):
        w = World(mk.vols[:n], mk.correlation, mk.T, label=f"n {n}")
        rows += _rows(
            w.label,
            env.values(w),
            gaussian_values(w, env.call_strike, env.package_scale),
            axis="n",
            x=n,
        )
    # the call on dispersion across strikes (fractions of the forward dispersion)
    g0 = gaussian_values(mk, 0.0, 1.0)["palladium"]
    sim = simulate(mk, env.cfg.n_paths, env.cfg.seed)
    for frac in (0.0, 0.5, 0.8, 1.0, 1.2, 1.5):
        k = frac * g0
        pay = np.maximum(
            trade_payoffs(sim, mk.w, call_strike=k, package_scale=1.0)["palladium"] - k, 0.0
        )
        x = 0.5 * (pay[0::2] + pay[1::2])
        rows.append(
            {
                "world": "market",
                "trade": "palladium call",
                "value": float(x.mean()),
                "stderr": float(x.std(ddof=1) / np.sqrt(x.size)),
                "gaussian": gaussian_palladium_call(mk.vols, mk.matrix, mk.w, mk.T, k),
                "axis": "strike_fraction",
                "x": frac,
            }
        )
    return pd.DataFrame(rows)


#: the expectations grid (module docstring)
EXPECT_RHO: tuple[float, ...] = (-0.2, -0.1, 0.0, 0.1, 0.2)
EXPECT_VOL: tuple[float, ...] = (0.8, 1.0, 1.2)
EXPECT_LOCAL: tuple[float, ...] = (0.0, 3.0)
EXPECT_CORR_SD: tuple[float, ...] = (0.0, 0.2)
EXPECT_JUMP: tuple[float, ...] = (0.0, 0.2)


def run_expectations(env: Environment) -> pd.DataFrame:
    """Part ``expectations`` (module docstring)."""
    mk = env.market
    price = env.values(mk)
    rows: list[dict[str, Any]] = []
    cells = [
        (dr, f, lam, sd, p)
        for dr in EXPECT_RHO
        for f in EXPECT_VOL
        for lam in EXPECT_LOCAL
        for sd in EXPECT_CORR_SD
        for p in EXPECT_JUMP
    ]
    for i, (dr, f, lam, sd, p) in enumerate(cells, start=1):
        rho = float(np.clip(mk.rho + dr, 0.0, 0.99))
        w = mk.replace(
            correlation=rho,
            vols=tuple(f * v for v in mk.vols),
            local_corr=lam,
            corr_sd=sd,
            jump_prob=p,
            jump_mean=0.0,
            jump_sd=0.10,
            label=f"exp rho{dr:+.1f} vol x{f:g} lam {lam:g} sd {sd:g} p {p:g}",
        )
        pay = env.payoffs(w, seed_offset=1)
        for t in TRADES:
            prem, prem_se = price[t]
            x = pay[t] - prem
            xa = 0.5 * (x[0::2] + x[1::2])
            rows.append(
                {
                    "d_rho": dr,
                    "vol_scale": f,
                    "local_corr": lam,
                    "corr_sd": sd,
                    "jump_prob": p,
                    "rho_realised": rho,
                    "trade": t,
                    "premium": prem,
                    "premium_stderr": prem_se,
                    "pnl_mean": float(xa.mean()),
                    "pnl_stderr": float(xa.std(ddof=1) / np.sqrt(xa.size)),
                    "pnl_std": float(np.std(x, ddof=1)),
                    "pnl_per_premium": float(xa.mean() / prem) if prem > 1e-12 else np.nan,
                    "p_loss": float(np.mean(x < 0)),
                }
            )
        if i % 10 == 0:
            log.info("expectations %d/%d", i, len(cells))
    df = pd.DataFrame(rows)
    # the best trade per cell by P&L per unit premium among the premium-paying trades, and by
    # mean P&L among all
    keys = ["d_rho", "vol_scale", "local_corr", "corr_sd", "jump_prob"]
    paying = df[df["premium"] > 1e-9]
    best = paying.loc[paying.groupby(keys)["pnl_per_premium"].idxmax(), [*keys, "trade"]].rename(
        columns={"trade": "best_per_premium"}
    )
    best2 = df.loc[df.groupby(keys)["pnl_mean"].idxmax(), [*keys, "trade"]].rename(
        columns={"trade": "best_mean"}
    )
    return df.merge(best, on=keys).merge(best2, on=keys)


def run_history(env: Environment) -> pd.DataFrame:
    """Part ``history`` (module docstring): the per-entry-date statistics with the implied
    correlation and the proxy-priced ex-post P&L of each trade."""
    closes = load_closes(BASKET, env.cfg.history)
    st = dispersion_statistics(closes, HORIZON_DAYS)
    cor = load_series("COR3M", env.cfg.history).reindex(st.index).ffill() / 100.0
    st["corr_implied"] = cor
    st["corr_premium"] = st["corr_implied"] - st["corr_realised"]
    vix = load_series("VIX", env.cfg.history).reindex(st.index).ffill()
    st["vix"] = vix
    q = vix.quantile([1 / 3, 2 / 3]).to_numpy()
    st["vix_tercile"] = regime_bins(vix, [0, q[0], q[1], 1e9], ["low", "mid", "high"])
    st["basket_move"] = regime_bins(
        st["r_basket"],
        [-10, -0.1, -0.03, 0.03, 0.1, 10],
        ["< -10%", "-10..-3%", "-3..3%", "3..10%", "> 10%"],
    )
    # trailing realised vols at entry (the previous 252 days), proxy implied = × premium
    logc = pd.DataFrame(
        np.log(closes.to_numpy(dtype=np.float64)), index=closes.index, columns=closes.columns
    )
    lr = logc.diff()
    trailing = lr.rolling(TRADING_DAYS).std(ddof=1) * np.sqrt(TRADING_DAYS)
    trailing = trailing.reindex(st.index)
    T = HORIZON_DAYS / TRADING_DAYS
    prem: dict[str, list[float]] = {t: [] for t in TRADES}
    for d, row in trailing.iterrows():
        v = row.to_numpy(dtype=np.float64) * IMPLIED_PREMIUM
        rho = float(cor.loc[d]) if np.isfinite(cor.loc[d]) else np.nan
        if not np.all(np.isfinite(v)) or not np.isfinite(rho):
            for t in TRADES:
                prem[t].append(np.nan)
            continue
        wd = World(tuple(float(x) for x in v), float(np.clip(rho, 0.0, 0.99)), T)
        g = gaussian_values(wd, env.call_strike, env.package_scale)
        for t in TRADES:
            prem[t].append(g[t])
    for t in TRADES:
        st[f"premium {t}"] = prem[t]
    # ex-post payoffs
    st["pay palladium"] = st["dispersion"]
    st["pay palladium call"] = np.maximum(st["dispersion"] - env.call_strike, 0.0)
    st["pay basket straddle"] = st["basket_abs"]
    st["pay single straddles"] = st["singles_abs"]
    st["pay straddle package"] = st["straddle_package"]
    st["pay straddle package (premium-neutral)"] = (
        st["singles_abs"] - env.package_scale * st["basket_abs"]
    )
    st["pay variance dispersion"] = np.nan  # needs the window's variances: vol_mean² proxy below
    st["pay variance dispersion"] = st["vol_mean"] ** 2 - st["vol_basket"] ** 2
    for t in TRADES:
        st[f"pnl {t}"] = st[f"pay {t}"] - st[f"premium {t}"]
    return st.reset_index()


def history_summary(hist: pd.DataFrame) -> pd.DataFrame:
    """Means with block standard errors by regime of the realised quantities and the proxy
    P&Ls (overlapping windows: ``n / horizon`` effective)."""
    cols = [
        "dispersion",
        "straddle_package",
        "basket_abs",
        "singles_abs",
        "corr_realised",
        "corr_implied",
        "corr_premium",
        "vol_mean",
        "vol_basket",
    ] + [f"pnl {t}" for t in TRADES]
    rows = []
    for key, groups in (
        ("all", {"all": hist}),
        ("vix_tercile", dict(tuple(hist.groupby("vix_tercile")))),
        ("basket_move", dict(tuple(hist.groupby("basket_move")))),
    ):
        for g, sub in groups.items():
            n = len(sub)
            n_eff = max(n / HORIZON_DAYS, 1.0)
            row: dict[str, Any] = {"regime": key, "group": g, "n": n, "n_eff": n_eff}
            for c in cols:
                x = sub[c].to_numpy(dtype=np.float64)
                x = x[np.isfinite(x)]
                row[c] = float(x.mean()) if x.size else np.nan
                row[f"{c}_stderr"] = float(x.std(ddof=1) / np.sqrt(n_eff)) if x.size > 1 else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


@dataclass
class PartResult:
    part: str
    wall_s: float
    path: Path


def run_part(env: Environment, part: str) -> PartResult:
    out = env.cfg.out
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"key": k, "value": str(v)} for k, v in env.setup.items()]).to_csv(
        out / "setup.csv", index=False
    )
    t0 = time.perf_counter()
    target = out / f"{part}.csv"
    if target.exists():
        log.info("%s exists: skipped", target)
    elif part == "sensitivities":
        run_sensitivities(env).to_csv(target, index=False)
    elif part == "expectations":
        run_expectations(env).to_csv(target, index=False)
    elif part == "history":
        h = run_history(env)
        h.to_csv(target, index=False)
        history_summary(h).to_csv(out / "history_summary.csv", index=False)
    else:
        raise ValueError(f"unknown part {part!r}")
    return PartResult(part, time.perf_counter() - t0, target)


__all__ = [
    "ANCHOR_DATE",
    "BASKET",
    "CALL_STRIKE_FRACTION",
    "HORIZON_DAYS",
    "IMPLIED_PREMIUM",
    "PARTS",
    "TRADES",
    "DispersionConfig",
    "Environment",
    "PartResult",
    "Simulated",
    "World",
    "gaussian_values",
    "history_summary",
    "market_world",
    "run_part",
    "simulate",
    "simulate_library",
    "simulate_one_factor",
    "trade_payoffs",
    "trade_values",
]
