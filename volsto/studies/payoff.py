"""The payoff study (owner's request of 2026-09-27): the reverse barriers (up-and-out call,
down-and-out and down-and-in puts), the put on realised variance, the knock-out variance swap,
the up and down variance swaps and the VKO put, through every line of the study catalogue on the
SPX 2022-12-30 anchor under the desk's marking fit — stage 1, the computation
(``scripts/payoff_study.py``; it calibrates the leverages it lacks, like ``scripts/m8b.py``), whose
outputs the report reads.

**The book** (:func:`payoff_book`; values ×100 in % of the spot for the options on the spot, in
vol points of vega notional 1 for the variance products — variance notional ``1/(2 K_vol)``):

* reverse barriers, strike 100%, daily-close monitoring (strict), notional 1/spot: up-and-out
  calls at 110% / 120%, down-and-out and down-and-in puts at 90% / 80%, 6m and 1y;
* the put on realised variance at 1y, ``(K² − RV)⁺`` struck at the 1y log-contract (VS) vol and
  at 80% of it;
* knock-out variance swaps over the owner's range: up barriers 101% / 103% / 105% at 3m / 6m /
  1y, close to close, the knock-out day's return counted, **settled at the knock-out** (owner,
  2026-09-27), struck at the VS vol of the maturity;
* up and down (corridor) variance swaps at 1y above / below 100% (the desk indicators ``prev`` /
  ``curr``), struck at the 1y VS vol;
* the 12m VKO put, strike 100%, vol barrier 30%, notional 1/spot.

**Parts** (each writes under ``<out>/<part>/`` and is resumable):

* ``prices`` — every product under Black–Scholes at the ATM vol of its maturity (the reference),
  local vol (Dupire of the snapshot), a one-factor LSV (:data:`ONE_FACTOR_PARAMS`, the library's
  reference one-factor Bergomi parameters, leverage calibrated to the same surface — not a marked
  model: the desk's P1 fit is two-factor) and the desk-marked two-factor LSV; one path set per
  model shared by the products; the model risk is the spread across LV / 1F / 2F.
* ``greeks`` — :func:`volsto.risk.report.risk_report` of one product per family under the 2F
  mark in the ``"recalibrate"`` mode (the owner's risk convention of 2026-09-27: after a market
  move the P1 parameters are held and the leverage is recalibrated so the vanillas reprice):
  delta / gamma in three regimes, parallel vega, theta (barrier options only: a daily-fixing
  variance product's one-day theta needs the day's return realised — the backtest attribution's
  seasoned theta), the vega-T tents and the skew and curvature ladders on the pillars up to the
  book's 1y; the barrier sensitivity and the
  barrier-shift table (the reserve of a conservative mark) of the barrier options and knock-out
  swaps and ``∂P(KO)/∂ln S`` of the knock-out swaps.
* ``dials`` — the prices across the marking dials: the desk fit re-run at the ``(ssr_target,
  skew_eps)`` cross :data:`DIAL_MARKS` around the reference (1.0, 0.10), each with its own
  leverage, the products priced on each.
* ``rotation`` — the shadow rotation with the P1 parameters held (owner's convention): the
  surface rotated by ``rota`` (:func:`volsto.hedging.worlds.shock_state`), the leverage
  recalibrated, the products repriced; the rotation greek is the central difference at ±1 on
  common random numbers.
* ``hedge`` — the hedge comparison (:func:`volsto.hedging.comparison.comparison_strategies`) of
  one product per family under the 2F pricing model, daily rebalancing, with and without
  transaction costs (:data:`HEDGE_COSTS`), in the 2F world (pricing = world) for every strategy
  and in the pure-LV world for the delta baseline and the static / best hedges (the reserve: the
  mean P&L when the world is not the pricing model), and the delta regimes of the delta baseline.

Checked by ``tests/test_payoff_study.py`` (the book, the task list, the fast plumbing on
Black–Scholes).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volsto.calibration.cache import LeverageCache
from volsto.calibration.fit_2f import load_fit_spec
from volsto.config import SimConfig
from volsto.engine.mc import MonteCarlo
from volsto.hedging.comparison import comparison_strategies
from volsto.hedging.hedger import Costs, Hedger, PricingContext, Schedule
from volsto.hedging.report import distribution_table
from volsto.hedging.strategies import GreekTargetStrategy, Target
from volsto.hedging.worlds import shock_state
from volsto.market.varswap import varswap_strike
from volsto.models.base import Model
from volsto.models.bs import BlackScholes
from volsto.products.barrier import KnockInOption, KnockOutOption, _BarrierOption
from volsto.products.base import Product, daily_schedule
from volsto.products.conditional_variance import DownVar, KnockOutVarianceSwap, UpVar
from volsto.products.variance import VarianceOption
from volsto.products.vko import VolKnockOutPut
from volsto.risk.engine import LSVBuilder, RiskEngine, RiskState, surface_of

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
#: the desk's marking of the SPX 2022-12-30 anchor (M7, ``--fit desk``, ssr 1, eps 0.10)
MARKING_FIT = ROOT / "configs" / "studies" / "m7_p1_marking" / "spx_ssr1_eps0.1.yaml"
DEFAULT_OUT = ROOT / "outputs" / "payoff"
DEFAULT_CACHE = ROOT / "cache"
PARTS: tuple[str, ...] = ("prices", "greeks", "dials", "rotation", "hedge")
#: the library's reference one-factor Bergomi parameters (``configs/studies/lsv_reference_1f
#: .yaml``: θ = 0, ν = 1.5, κ = 1.5, ρ = −0.7), the leverage calibrated to the anchor surface
ONE_FACTOR_PARAMS: dict[str, float] = {
    "nu": 1.5,
    "theta": 0.0,
    "k1": 1.5,
    "k2": 1.5,
    "rho12": 0.0,
    "rho_SX1": -0.7,
    "rho_SX2": 0.0,
}
#: the marking-dial cross around the reference mark (1.0, 0.10) (the S5 grid's axes)
DIAL_MARKS: tuple[tuple[float, float], ...] = (
    (1.0, 0.10),
    (0.75, 0.10),
    (1.25, 0.10),
    (1.5, 0.10),
    (1.0, 0.05),
    (1.0, 0.20),
)
#: the shadow-rotation states (rota 0 is the base)
ROTATIONS: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)
#: the Greeks' pillars (the book's maturities are at most 1y)
GREEK_PILLARS: tuple[float, ...] = (0.25, 0.5, 1.0)
GREEK_REGIMES: tuple[str, ...] = ("model", "sticky_strike", "sticky_moneyness")
GREEK_SECTIONS: tuple[str, ...] = ("delta", "vega", "theta", "vega_T", "skew", "curvature")
GREEK_HALVINGS = 6
#: half-spreads of the costed hedge runs: 1 bp on the spot, 0.25 vol point on options and swaps
HEDGE_COSTS = Costs(spot_bps=1.0, vol_points=0.25)
#: one product per family: the Greeks and the hedge comparison
REPRESENTATIVES: tuple[str, ...] = (
    "uoc 6m 110",
    "dop 6m 90",
    "dip 6m 90",
    "put on var 1y 100",
    "ko var 6m 103",
    "up var 1y",
    "down var 1y",
    "vko put 12m",
)
#: the strategies run in the pure-LV world besides the delta baseline (the reserve)
LV_WORLD_STRATEGIES: tuple[str, ...] = (
    "delta",
    "PCS static (carry, BGK) no delta",
    "preset",
    "corridor strip",
    "stopped log strip",
    "put static + delta",
)
#: the delta regimes of the delta baseline (``"model"`` is the baseline itself)
HEDGE_REGIMES: tuple[str, ...] = ("sticky_strike", "min_variance")
#: the products whose delta baseline is also run in the other regimes
REGIME_PRODUCTS: tuple[str, ...] = ("uoc 6m 110", "dop 6m 90", "up var 1y", "ko var 6m 103")


# --------------------------------------------------------------------------------------------
# the book
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BookEntry:
    """A product of the book with its unit and scale (the reported value is ``scale × value``)
    and its family."""

    name: str
    product: Product
    family: str
    unit: str
    scale: float
    strike_vol: float = float("nan")


def payoff_book(
    spot: float,
    discount: Any,
    vs_vol: Callable[[float], float],
) -> dict[str, BookEntry]:
    """The study's book (module docstring); ``vs_vol(T)`` is the anchor's log-contract vol."""
    out: dict[str, BookEntry] = {}
    pct = ("% of spot", 100.0)
    vp = ("vol pts (vega notional 1)", 100.0)

    def barrier(
        cls: type[_BarrierOption], cp: int, level: float, direction: str, T: float
    ) -> Product:
        return cls(
            spot,
            T,
            cp,
            level * spot,
            direction,
            discount,
            monitoring="discrete",
            fixing_times=daily_schedule(T, 252),
            strict=True,
            notional=1.0 / spot,
        )

    for T, tag in ((0.5, "6m"), (1.0, "1y")):
        for lvl in (1.10, 1.20):
            nm = f"uoc {tag} {round(100 * lvl)}"
            out[nm] = BookEntry(nm, barrier(KnockOutOption, 1, lvl, "up", T), "uoc", *pct)
        for lvl in (0.90, 0.80):
            nm = f"dop {tag} {round(100 * lvl)}"
            out[nm] = BookEntry(nm, barrier(KnockOutOption, -1, lvl, "down", T), "dop", *pct)
            nm = f"dip {tag} {round(100 * lvl)}"
            out[nm] = BookEntry(nm, barrier(KnockInOption, -1, lvl, "down", T), "dip", *pct)
    k1 = float(vs_vol(1.0))
    prod: Product
    for ratio in (1.0, 0.8):
        k = ratio * k1
        nm = f"put on var 1y {round(100 * ratio)}"
        prod = VarianceOption(
            daily_schedule(1.0, 252),
            k,
            discount,
            cp=-1,
            notional=1.0 / (2 * k),
            annualisation=252.0,
        )
        out[nm] = BookEntry(nm, prod, "put on var", *vp, strike_vol=k)
    for T, tag in ((0.25, "3m"), (0.5, "6m"), (1.0, "1y")):
        k = float(vs_vol(T))
        for b in (1.01, 1.03, 1.05):
            nm = f"ko var {tag} {round(100 * b)}"
            prod = KnockOutVarianceSwap(
                daily_schedule(T, 252),
                b * spot,
                k,
                discount,
                settlement="knock_out",
                notional=1.0 / (2 * k),
            )
            out[nm] = BookEntry(nm, prod, "ko var", *vp, strike_vol=k)
    for side, maker in (("up", UpVar), ("down", DownVar)):
        nm = f"{side} var 1y"
        prod = maker(
            daily_schedule(1.0, 252),
            spot,
            k1,
            discount,
            convention="corridor",
            notional=1.0 / (2 * k1),
        )
        out[nm] = BookEntry(nm, prod, f"{side} var", *vp, strike_vol=k1)
    nm = "vko put 12m"
    out[nm] = BookEntry(
        nm,
        VolKnockOutPut(spot, 1.0, 0.30, daily_schedule(1.0, 252), discount, notional=1.0 / spot),
        "vko",
        *pct,
    )
    return out


# --------------------------------------------------------------------------------------------
# configuration and environment
# --------------------------------------------------------------------------------------------


@dataclass
class PayoffConfig:
    """Budgets and paths (module docstring)."""

    out: Path = DEFAULT_OUT
    cache: Path = DEFAULT_CACHE
    marking_fit: Path = MARKING_FIT
    n_particles: int = 800_000
    seed: int = 2024
    price_paths: int = 100_000
    greek_paths: int = 50_000
    hedge_paths: int = 20_000
    world_paths: int = 10_000
    allow_calibrate: bool = True
    verbose: bool = False

    def sim(self, n_paths: int) -> SimConfig:
        return SimConfig(
            n_paths=int(n_paths),
            chunk_size=min(int(n_paths), 20_000),
            seed=int(self.seed),
            dt_max=1.0 / 52.0,
        )


class PayoffEnvironment:
    """The anchor's marking state, its models and the book (built lazily; counts the leverage
    calibrations it triggers)."""

    def __init__(self, cfg: PayoffConfig) -> None:
        self.cfg = cfg
        self.cache = LeverageCache(cfg.cache)
        self.fit_spec = load_fit_spec(cfg.marking_fit)
        spec = dataclasses.replace(
            self.fit_spec.spec,
            particle=dataclasses.replace(
                self.fit_spec.spec.particle, n_particles=int(cfg.n_particles)
            ),
        )
        self.state = RiskState(spec, None, "desk mark")
        self.surface = surface_of(self.state)
        self.forward_curve = self.surface.forward_curve
        self.discount = self.forward_curve.rate_curve
        self.spot = float(self.state.spot)
        self.calibrated: list[str] = []
        self._models: dict[str, Model] = {}
        self._book: dict[str, BookEntry] | None = None

    def vs_vol(self, T: float) -> float:
        return float(np.sqrt(float(varswap_strike(self.surface, T))))

    @property
    def book(self) -> dict[str, BookEntry]:
        if self._book is None:
            self._book = payoff_book(self.spot, self.discount, self.vs_vol)
        return self._book

    def lsv(self, state: RiskState, label: str) -> Model:
        """The LSV of ``state`` from the cache, calibrating when allowed (recorded)."""
        if state.key in self._models:
            return self._models[state.key]
        miss = not self.cache.has(state.spec)
        t0 = time.perf_counter()
        model, _ = self.cache.get_or_calibrate(state.spec, allow_calibrate=self.cfg.allow_calibrate)
        if miss:
            self.calibrated.append(state.key)
            log.info(
                "calibrated %s (%s) in %.0f s", label, state.key[:12], time.perf_counter() - t0
            )
        self._models[state.key] = model
        return model

    def model(self, name: str) -> Model:
        """``"LV"``, ``"1F"`` or ``"2F"`` (the desk mark)."""
        if name == "2F":
            return self.lsv(self.state, "2F desk mark")
        if name == "1F":
            st = self.state.with_params(label="1F reference", **ONE_FACTOR_PARAMS)
            return self.lsv(st, "1F reference parameters")
        if name == "LV":
            if "LV" not in self._models:
                self._models["LV"] = PricingContext.from_state(self.state, None, "lv").model
            return self._models["LV"]
        raise ValueError(f"unknown model {name!r}")

    def bs_model(self, T: float) -> BlackScholes:
        return BlackScholes(float(self.surface.atm_vol(T)), self.forward_curve)

    def pricing_ctx(self) -> PricingContext:
        return PricingContext.from_state(
            self.state,
            self.cache,
            "lsv",
            allow_calibrate=self.cfg.allow_calibrate,
            label="2F desk mark (SPX 2022-12-30, ssr 1, eps 0.10)",
        )


# --------------------------------------------------------------------------------------------
# prices, dials, rotation
# --------------------------------------------------------------------------------------------


def _price_rows(
    env: PayoffEnvironment,
    priced_by: Model,
    names: Sequence[str],
    sim: SimConfig,
    **tags: Any,
) -> list[dict[str, Any]]:
    """Every named product on ONE path set of ``priced_by`` (common random numbers across
    them); ``tags`` label the rows."""
    entries = [env.book[n] for n in names]
    t0 = time.perf_counter()
    res = MonteCarlo(sim).price_many([e.product for e in entries], priced_by)
    wall = time.perf_counter() - t0
    rows = []
    for e, r in zip(entries, res, strict=True):
        rows.append(
            {
                "product": e.name,
                "family": e.family,
                "unit": e.unit,
                "value": e.scale * float(r.mean),
                "stderr": e.scale * float(r.stderr),
                "n_paths": sim.n_paths,
                "wall_s": wall / len(entries),
                **tags,
            }
        )
    return rows


def run_prices(env: PayoffEnvironment) -> pd.DataFrame:
    """Part ``prices`` (module docstring)."""
    sim = env.cfg.sim(env.cfg.price_paths)
    names = list(env.book)
    rows: list[dict[str, Any]] = []
    for m in ("LV", "1F", "2F"):
        rows += _price_rows(env, env.model(m), names, sim, model=m)
    for n in names:
        e = env.book[n]
        T = float(e.product.maturity)
        rows += _price_rows(env, env.bs_model(T), [n], sim, model="BS (ATM vol)")
    return pd.DataFrame(rows)


def model_risk_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Per product: the LSV prices, the spread across LV / 1F / 2F and 2F − LV (stderr in
    quadrature: independent path sets per model)."""
    rows = []
    for name, g in prices.groupby("product", sort=False):
        by = {r["model"]: (r["value"], r["stderr"]) for _, r in g.iterrows()}
        lsv = [by[m] for m in ("LV", "1F", "2F") if m in by]
        vals = [v for v, _ in lsv]
        i_max, i_min = int(np.argmax(vals)), int(np.argmin(vals))
        spread = vals[i_max] - vals[i_min]
        spread_se = float(np.hypot(lsv[i_max][1], lsv[i_min][1]))
        d = by["2F"][0] - by["LV"][0]
        d_se = float(np.hypot(by["2F"][1], by["LV"][1]))
        rows.append(
            {
                "product": name,
                "unit": g["unit"].iloc[0],
                "BS": by.get("BS (ATM vol)", (np.nan, np.nan))[0],
                "LV": by["LV"][0],
                "1F": by["1F"][0],
                "2F": by["2F"][0],
                "2F_stderr": by["2F"][1],
                "spread": spread,
                "spread_stderr": spread_se,
                "2F_minus_LV": d,
                "2F_minus_LV_stderr": d_se,
            }
        )
    return pd.DataFrame(rows)


def run_dials(env: PayoffEnvironment) -> pd.DataFrame:
    """Part ``dials``: the desk fit at each mark of :data:`DIAL_MARKS`, its leverage, the book."""
    from volsto.viewers.grid import marking_fit

    sim = env.cfg.sim(env.cfg.price_paths)
    names = list(env.book)
    rows: list[dict[str, Any]] = []
    for ssr, eps in DIAL_MARKS:
        fit = marking_fit(env.surface, ssr, eps, fit="desk", snapshot=env.fit_spec.snapshot)
        if fit.status == "infeasible":
            rows.append({"ssr_target": ssr, "skew_eps": eps, "status": fit.status})
            continue
        spec = dataclasses.replace(env.state.spec, model=fit.params)
        st = RiskState(spec, None, f"mark ssr{ssr:g} eps{eps:g}")
        model = env.lsv(st, st.label)
        rows += _price_rows(
            env,
            model,
            names,
            sim,
            ssr_target=ssr,
            skew_eps=eps,
            status=fit.status,
            params=repr(fit.params),
        )
    return pd.DataFrame(rows)


def run_rotation(env: PayoffEnvironment) -> pd.DataFrame:
    """Part ``rotation``: the book at each rotated state, P1 held, the leverage recalibrated."""
    sim = env.cfg.sim(env.cfg.price_paths)
    names = list(env.book)
    rows: list[dict[str, Any]] = []
    for rota in ROTATIONS:
        st = env.state if rota == 0.0 else shock_state(env.state, rota)
        rows += _price_rows(env, env.lsv(st, f"rota {rota:+g}"), names, sim, rota=rota)
    return pd.DataFrame(rows)


def rotation_greek(rot: pd.DataFrame) -> pd.DataFrame:
    """``(V(+1) − V(−1))/2`` per product with the common-random-numbers caveat: the states share
    the seed but the paired error is not stored (the stderr shown is the quadrature bound)."""
    rows = []
    for name, g in rot.groupby("product", sort=False):
        v = {float(r["rota"]): (float(r["value"]), float(r["stderr"])) for _, r in g.iterrows()}
        if 1.0 not in v or -1.0 not in v:
            continue
        rows.append(
            {
                "product": name,
                "unit": g["unit"].iloc[0],
                "rotation_greek": 0.5 * (v[1.0][0] - v[-1.0][0]),
                "stderr_bound": 0.5 * float(np.hypot(v[1.0][1], v[-1.0][1])),
                "convexity": v.get(2.0, (np.nan,))[0] - 2 * v[0.0][0] + v.get(-2.0, (np.nan,))[0],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# greeks
# --------------------------------------------------------------------------------------------


def run_greeks(env: PayoffEnvironment, names: Sequence[str] = REPRESENTATIVES) -> pd.DataFrame:
    """Part ``greeks`` (module docstring)."""
    from volsto.risk.product_risk import (
        barrier_sensitivity,
        barrier_shift_table,
        ko_probability_delta,
    )
    from volsto.risk.report import risk_report

    ctx = env.pricing_ctx()
    builder = ctx.builder
    assert isinstance(builder, LSVBuilder)
    # the eSSVI curvature ladders need up to six halvings (SPEC §13.2; the backtest's setting)
    engine = RiskEngine(builder, env.cfg.sim(env.cfg.greek_paths), max_halvings=GREEK_HALVINGS)
    rows: list[dict[str, Any]] = []
    for n in names:
        e = env.book[n]
        t0 = time.perf_counter()
        # a daily-fixing variance product has no one-day ``aged`` (its next fixing lies inside
        # the roll window: the day's return must be realised, which the seasoning does — the
        # backtest's attribution carries that theta); the barrier options age by dropping the
        # monitoring dates the day passes
        sections = (
            GREEK_SECTIONS
            if isinstance(e.product, _BarrierOption)
            else tuple(x for x in GREEK_SECTIONS if x != "theta")
        )
        rep = risk_report(
            engine,
            e.product,
            env.state,
            sections=sections,
            pillars=GREEK_PILLARS,
            regimes=GREEK_REGIMES,
        )
        df = rep.to_dataframe()
        for _, r in df.iterrows():
            row = {str(k): (v.item() if hasattr(v, "item") else v) for k, v in r.to_dict().items()}
            rows.append({"product": n, "family": e.family, "unit": e.unit, "scale": e.scale, **row})
        base = {"product": n, "family": e.family, "unit": e.unit, "scale": e.scale}
        if isinstance(e.product, _BarrierOption | KnockOutVarianceSwap | VolKnockOutPut):
            sens = barrier_sensitivity(engine, e.product, env.state)
            for key, sv in sens.items():
                rows.append(
                    {
                        **base,
                        "group": "barrier",
                        "name": key,
                        "value": float(sv.value),
                        "stderr": float(sv.stderr),
                        "unit_sens": sv.unit,
                    }
                )
        if isinstance(e.product, _BarrierOption | KnockOutVarianceSwap):
            tab = barrier_shift_table(engine, e.product, env.state)
            for _, r in tab.iterrows():
                rows.append(
                    {
                        **base,
                        "group": "barrier_shift",
                        "name": f"shift {float(r['shift']):+.3f}",
                        "value": float(r["delta_price"]),
                        "stderr": float(r["stderr"]),
                    }
                )
        if isinstance(e.product, KnockOutVarianceSwap | VolKnockOutPut):
            kp = ko_probability_delta(engine, e.product, env.state)
            rows.append(
                {
                    **base,
                    "scale": 1.0,
                    "group": "product",
                    "name": "dP(KO)/dlnS",
                    "value": float(kp.value),
                    "stderr": float(kp.stderr),
                    "unit_sens": "probability per unit ln S",
                }
            )
        log.info("greeks %s in %.0f s", n, time.perf_counter() - t0)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# hedging
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HedgeTask:
    product: str
    strategy: str
    world: str = "2F"
    regime: str = "model"

    @property
    def key(self) -> str:
        text = f"{self.product}__{self.strategy}__{self.world}__{self.regime}"
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def hedge_tasks(strategy_names: Mapping[str, Sequence[str]]) -> list[HedgeTask]:
    """The hedge comparison's tasks (module docstring): every strategy in the 2F world; the
    :data:`LV_WORLD_STRATEGIES` a product has in the pure-LV world; the delta baseline of
    :data:`REGIME_PRODUCTS` in the :data:`HEDGE_REGIMES`."""
    tasks = []
    for p in REPRESENTATIVES:
        names = list(strategy_names[p])
        tasks += [HedgeTask(p, s) for s in names]
        tasks += [HedgeTask(p, s, "LV") for s in names if s in LV_WORLD_STRATEGIES]
        if p in REGIME_PRODUCTS:
            tasks += [HedgeTask(p, "delta", "2F", r) for r in HEDGE_REGIMES]
    return tasks


def _summary(task: HedgeTask, e: BookEntry, r: Any, wall: float) -> dict[str, Any]:
    out: dict[str, Any] = {
        **dataclasses.asdict(task),
        "unit": e.unit,
        "scale": e.scale,
        "wall_s": wall,
        "value_0": e.scale * r.value_0,
        "value_0_stderr": e.scale * r.value_0_stderr,
        "n_paths": r.n_paths,
        "n_dates": int(r.dates.size),
        "early_terminations": int(np.isfinite(r.termination).sum()),
        "product_std": e.scale * float(np.std(r.pnl_product, ddof=1)),
        "instruments": list(r.instruments),
        "leg_std": [e.scale * float(x) for x in np.std(r.pnl_hedges, axis=0, ddof=1)],
        "costs_mean": e.scale * float(np.mean(r.costs)),
        "notes": list(r.pricing_notes),
        "budget": {k: float(v) for k, v in r.budget.items()},
    }
    for series, x in (("with costs", r.pnl_total), ("zero cost", r.pnl_zero_cost)):
        dist = distribution_table(e.scale * np.asarray(x), series)
        out[series] = {
            str(row["statistic"]): [float(row["value"]), float(row["stderr"])]
            for _, row in dist.iterrows()
        }
    return out


def run_hedge_task(env: PayoffEnvironment, task: HedgeTask, out: Path) -> dict[str, Any]:
    """One hedge run, saved as ``<out>/hedge/<key>.json`` (skipped when it exists)."""
    path = out / "hedge" / f"{task.key}.json"
    if path.exists():
        return dict(json.loads(path.read_text()))
    e = env.book[task.product]
    ctx = env.pricing_ctx()
    world = env.model("LV") if task.world == "LV" else ctx.model
    h = Hedger(
        ctx,
        world,
        Schedule("daily"),
        HEDGE_COSTS,
        sim=env.cfg.sim(env.cfg.hedge_paths),
        world_paths=env.cfg.world_paths,
        verbose=env.cfg.verbose,
    )
    pc = h.preset_context(e.product)
    sets = comparison_strategies(e.product, pc)
    if task.regime != "model":
        strat: GreekTargetStrategy = GreekTargetStrategy(
            (Target("delta"),),
            [pc.spot_instrument()],
            delta_regime=task.regime,
            name=f"delta ({task.regime})",
        )
    else:
        strat = sets[task.strategy]
    t0 = time.perf_counter()
    r = h.run(e.product, strat)
    res = _summary(task, e, r, time.perf_counter() - t0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, default=str))
    return res


def strategy_names(env: PayoffEnvironment) -> dict[str, list[str]]:
    """The comparison set's names per representative product (no pricing)."""
    ctx = PricingContext(env.model("LV"), env.state, None, env.surface, "names only")
    h = Hedger(ctx, ctx.model, Schedule("daily"), HEDGE_COSTS, sim=env.cfg.sim(2), verbose=False)
    return {
        p: list(comparison_strategies(env.book[p].product, h.preset_context(env.book[p].product)))
        for p in REPRESENTATIVES
    }


def hedge_table(out: Path) -> pd.DataFrame:
    """Every saved hedge run as a row: mean, std, q01, es01 with their stderrs, with and without
    costs, the product's std and the ratio."""
    rows = []
    for p in sorted((out / "hedge").glob("*.json")):
        d = json.loads(p.read_text())
        row: dict[str, Any] = {k: d[k] for k in ("product", "strategy", "world", "regime", "unit")}
        for series, tag in (("zero cost", "zc"), ("with costs", "tc")):
            for stat in ("mean", "std", "q01", "es01", "es05"):
                v, se = d[series][stat]
                row[f"{stat}_{tag}"] = v
                row[f"{stat}_{tag}_stderr"] = se
        row["product_std"] = d["product_std"]
        row["std_ratio_zc"] = row["std_zc"] / d["product_std"] if d["product_std"] else np.nan
        row["costs_mean"] = d["costs_mean"]
        row["wall_s"] = d["wall_s"]
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------


@dataclass
class PartResult:
    part: str
    wall_s: float
    calibrations: int
    path: Path
    notes: list[str] = field(default_factory=list)


def setup_table(env: PayoffEnvironment) -> pd.DataFrame:
    """The run's setup (the spot converts the per-unit-spot delta and gamma of the report to a
    1% spot move)."""
    c = env.cfg
    rows = [
        ("spot", env.spot),
        ("n_particles", c.n_particles),
        ("seed", c.seed),
        ("price_paths", c.price_paths),
        ("greek_paths", c.greek_paths),
        ("hedge_paths", c.hedge_paths),
        ("world_paths", c.world_paths),
        ("hedge_cost_spot_bps", HEDGE_COSTS.spot_bps),
        ("hedge_cost_vol_points", HEDGE_COSTS.vol_points),
    ]
    return pd.DataFrame(
        [{"key": k, "value": float(v)} for k, v in rows]
        + [{"key": "marking_fit", "value": np.nan, "text": str(c.marking_fit)}]
    )


def run_part(env: PayoffEnvironment, part: str) -> PartResult:
    """Run one part and write its table(s) (a part whose table exists is skipped; the hedge
    part resumes task by task); ``setup.csv`` is (re)written with every part."""
    out = env.cfg.out
    out.mkdir(parents=True, exist_ok=True)
    setup_table(env).to_csv(out / "setup.csv", index=False)
    n_cal0 = len(env.calibrated)
    t0 = time.perf_counter()
    target = out / f"{part}.csv"
    if part == "hedge":
        tasks = hedge_tasks(strategy_names(env))
        for i, t in enumerate(tasks, start=1):
            t1 = time.perf_counter()
            res = run_hedge_task(env, t, out)
            log.info(
                "hedge %d/%d %s: std %.4f (product %.4f) in %.0f s",
                i,
                len(tasks),
                t.key,
                res["zero cost"]["std"][0],
                res["product_std"],
                time.perf_counter() - t1,
            )
        hedge_table(out).to_csv(target, index=False)
    elif target.exists():
        log.info("%s exists: skipped", target)
    else:
        fn = {
            "prices": run_prices,
            "greeks": run_greeks,
            "dials": run_dials,
            "rotation": run_rotation,
        }[part]
        df = fn(env)
        df.to_csv(target, index=False)
        if part == "prices":
            model_risk_table(df).to_csv(out / "model_risk.csv", index=False)
        if part == "rotation":
            rotation_greek(df).to_csv(out / "rotation_greek.csv", index=False)
    return PartResult(part, time.perf_counter() - t0, len(env.calibrated) - n_cal0, target)


__all__ = [
    "DIAL_MARKS",
    "GREEK_PILLARS",
    "HEDGE_COSTS",
    "MARKING_FIT",
    "ONE_FACTOR_PARAMS",
    "PARTS",
    "REPRESENTATIVES",
    "ROTATIONS",
    "BookEntry",
    "HedgeTask",
    "PartResult",
    "PayoffConfig",
    "PayoffEnvironment",
    "hedge_table",
    "hedge_tasks",
    "model_risk_table",
    "payoff_book",
    "rotation_greek",
    "run_hedge_task",
    "run_part",
    "setup_table",
    "strategy_names",
]
