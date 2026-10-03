"""The barrier-versus-vanilla study (SPEC §8.4; the owner's question of 2026-10-03): a framework
deciding, from pure-vol, market and statistical metrics, whether and when to trade a call ratio
or a call fly instead of an up-and-out call (daily or continuous observation) — and, by the
same logic, a put ratio or a put fly instead of a down-and-out put.  Stage 1, the computation
(``scripts/barrier_vs_vanilla_study.py``; it calibrates the leverages it lacks like the payoff
study), whose outputs the report reads.

**The book** (:func:`study_book`; values ×100 in % of the spot, notional ``1/spot``).  For each
maturity 3m / 6m / 1y and each up barrier 110% / 115% / 120% (strike 100%): the up-and-out call
observed on the daily closes (strict) and continuously (Brownian bridge), its European knock-out
(observed at expiry only — the upper bound of both), the call spread ``K–B``, the call fly ``K /
(K+B)/2 / B`` and the 1×2 call ratio ``K / (K+B)/2``; for each down barrier 90% / 85% / 80%
the mirror image (down-and-out put, European knock-out, put spread, put fly, 1×2 put ratio).

**Parts** (each writes ``<out>/<part>.csv``; resumable):

* ``anchor`` — the barrier products under Black–Scholes (ATM vol of the maturity), local vol, the
  reference one-factor LSV and the desk-marked two-factor LSV on one path set per model (10⁵
  paths); the vanilla structures priced **off the surface** (model-free) and, as a check, under
  local vol; the touch probability and the *regret* probability (touched, alive side at expiry,
  in the money: the paths on which the barrier threw away a payoff) under the 2F mark.  The
  decomposition table (:func:`decomposition_table`): European knock-out, regret value ``EKO −
  UOC``, the continuous-monitoring discount, the model spread, the premium-matched
  alternatives (the ratio in closed form, the fly's outer wing, the fly's units).
* ``map`` — the *when* of the question: the daily 6m 110% up-and-out call and 6m 90% down-and-
  out put and their premium-matched flies / ratios across spot (−10% … +10%, the leverage held
  in spot: the ``"model"`` regime) and time to expiry (6m, 3m, 1m, the surface held in strike
  and calendar time), under the 2F mark.
* ``greeks`` — :func:`volsto.risk.report.risk_report` (delta regimes, gamma, vega, theta, cross
  Greeks, vega tents, skew and curvature ladders) of the 6m representatives: the up-and-out call
  110%, its fly and ratio, the down-and-out put 90%, its put fly and put ratio.
* ``hedge`` — the hedger under the 2F pricing model (daily, with and without costs) for the fly,
  the ratio and the spread on the delta (2F and pure-LV worlds), and for the barrier option
  with the premium-matched fly (``"fly static + delta"``) or ratio as a static proxy; the
  barrier's own baselines (delta, the put-call-symmetry replication) are read from the payoff
  study's ``outputs/payoff/hedge.csv`` when present.
* ``daily`` — the 127 snapshots of 2022 H2 (``configs/surfaces/snapshots/hdn_2022H2``): each
  day's Dupire local vol prices the daily barriers and the European knock-outs (2·10⁴ paths),
  the surface prices the vanilla structures; with the day's market metrics (ATM vols, 90–110
  skew, fly curvature, term slope, 1m / 3m realised vol, implied minus realised, VIX, VVIX,
  SKEW, 3m momentum, distance to the barrier in standard deviations) and, where the horizon
  fits inside the history, the realised outcome of each structure entered that day.
* ``history`` — the statistical layer (:mod:`volsto.studies.history_stats`): touch and regret
  frequencies and the structures' realised payoffs over 1990–2026 by regime (VIX tercile,
  implied-minus-realised sign, 3m momentum sign), and the filtered historical simulation at the
  anchor's realised and implied vols against the 2F mark's touch / regret probabilities.

**The framework** (:func:`decision_table`).  For a buyer, the up-and-out call and its
premium-matched vanilla alternative cost the same; what differs is (i) the paths on which they
pay — the barrier loses the regret paths, the fly loses above its wing, the ratio loses without
limit above its break-even — so the choice is a view on the *real-world* path distribution
against the one the mark implies: the score is the P-measure expected payoff of each per unit
premium (filtered historical simulation at the chosen vol, or the regime-conditional history)
minus one, the barrier preferred when its expected-payoff-to-premium exceeds the alternative's
by more than (ii) its model-risk band (the LV / 1F / 2F spread — the vanilla has none) and (iii)
its hedging cost (the hedged P&L std and the transaction costs of the hedger, against the
fly's static hold).  The daily series of these inputs over 2022 H2 is the *when*: the regret
value shrinks with the distance to the barrier in standard deviations and with implied vol
below realised, the fly's premium rises with the smile's curvature, the ratio's with the skew.

Checked by ``tests/test_barrier_vs_vanilla.py`` (the book, the tables, the fast plumbing on
Black–Scholes).
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volsto.config import LocalVolConfig, SimConfig
from volsto.engine.mc import MonteCarlo
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import ImpliedSurface
from volsto.models.base import Model
from volsto.models.localvol import LocalVol
from volsto.products.barrier import KnockOutOption, OneTouch
from volsto.products.base import Product, daily_schedule
from volsto.products.structures import (
    VanillaStructure,
    call_fly,
    call_ratio,
    call_spread,
    european_knock_out,
    european_knock_out_surface_price,
    fly_width_for_premium,
    put_fly,
    put_ratio,
    put_spread,
    ratio_for_premium,
)
from volsto.studies.history_stats import (
    HISTORY_DIR,
    TRADING_DAYS,
    barrier_path_statistics,
    fhs_barrier_table,
    filtered_historical_paths,
    frequency_table,
    load_series,
    log_returns,
    realised_vol,
    regime_bins,
)
from volsto.studies.payoff import PayoffConfig, PayoffEnvironment

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "outputs" / "barrier_vs_vanilla"
PAYOFF_OUT = ROOT / "outputs" / "payoff"
SNAPSHOT_DIR = ROOT / "configs" / "surfaces" / "snapshots" / "hdn_2022H2"
PARTS: tuple[str, ...] = ("anchor", "map", "greeks", "hedge", "daily", "history")
MATURITIES: tuple[tuple[float, str], ...] = ((0.25, "3m"), (0.5, "6m"), (1.0, "1y"))
UP_BARRIERS: tuple[float, ...] = (1.10, 1.15, 1.20)
DOWN_BARRIERS: tuple[float, ...] = (0.90, 0.85, 0.80)
#: the horizons of the statistical layer in trading days, matched to the maturities
HORIZONS: dict[str, int] = {"3m": 63, "6m": 126, "1y": 252}
#: the representatives (the Greeks, the hedge comparison, the map)
REPRESENTATIVES: tuple[str, ...] = (
    "uoc 6m 110",
    "fly 6m 110",
    "ratio 6m 110",
    "dop 6m 90",
    "pfly 6m 90",
    "pratio 6m 90",
)
GREEK_PILLARS: tuple[float, ...] = (0.25, 0.5, 1.0)
GREEK_REGIMES: tuple[str, ...] = ("model", "sticky_strike", "sticky_moneyness")
GREEK_SECTIONS: tuple[str, ...] = ("delta", "vega", "theta", "vega_T", "skew", "curvature")
GREEK_HALVINGS = 6
MAP_SPOTS: tuple[float, ...] = (0.90, 0.95, 0.975, 1.0, 1.025, 1.05, 1.075, 1.10)
MAP_TAUS: tuple[float, ...] = (0.5, 0.25, 1.0 / 12.0)
#: filtered-historical-simulation settings of the statistical layer
FHS_PATHS = 50_000
FHS_SEED = 11
#: the local-vol grid of the daily part (the snapshots' surfaces reach 3y; 1y products)
DAILY_LV = LocalVolConfig(t_max=1.5)
DAILY_PATHS = 20_000


# --------------------------------------------------------------------------------------------
# the book
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Entry:
    """A product of the book: its family (``barrier`` — priced by Monte Carlo — or ``vanilla``
    — priced off the surface), maturity tag, side, barrier level and the structure's role."""

    name: str
    product: Product
    family: str
    tag: str
    T: float
    side: str
    level: float
    role: str

    @property
    def structure(self) -> VanillaStructure:
        assert isinstance(self.product, VanillaStructure)
        return self.product


def _barrier(
    spot: float, discount: Any, T: float, level: float, side: str, monitoring: str
) -> KnockOutOption:
    cp, direction = (1, "up") if side == "up" else (-1, "down")
    if monitoring == "continuous":
        return KnockOutOption(
            spot,
            T,
            cp,
            level * spot,
            direction,
            discount,
            monitoring="continuous",
            notional=1 / spot,
        )
    return KnockOutOption(
        spot,
        T,
        cp,
        level * spot,
        direction,
        discount,
        monitoring="discrete",
        fixing_times=daily_schedule(T, 252),
        strict=True,
        notional=1 / spot,
    )


def study_book(spot: float, discount: Any) -> dict[str, Entry]:
    """The book of the module docstring (strike at the spot, notional ``1/spot``)."""
    out: dict[str, Entry] = {}
    k = float(spot)

    def add(
        name: str,
        product: Product,
        family: str,
        tag: str,
        T: float,
        side: str,
        lvl: float,
        role: str,
    ) -> None:
        out[name] = Entry(name, product, family, tag, T, side, lvl, role)

    for T, tag in MATURITIES:
        for lvl in UP_BARRIERS:
            b, m, p = lvl * k, 0.5 * (1.0 + lvl) * k, round(100 * lvl)
            add(
                f"uoc {tag} {p}",
                _barrier(k, discount, T, lvl, "up", "discrete"),
                "barrier",
                tag,
                T,
                "up",
                lvl,
                "uoc",
            )
            add(
                f"uoc {tag} {p} cont",
                _barrier(k, discount, T, lvl, "up", "continuous"),
                "barrier",
                tag,
                T,
                "up",
                lvl,
                "uoc cont",
            )
            add(
                f"eko {tag} {p}",
                european_knock_out(k, b, "up", 1, T, discount, 1 / k),
                "barrier",
                tag,
                T,
                "up",
                lvl,
                "eko",
            )
            add(
                f"cs {tag} {p}",
                call_spread(k, b, T, discount, 1 / k),
                "vanilla",
                tag,
                T,
                "up",
                lvl,
                "spread",
            )
            add(
                f"fly {tag} {p}",
                call_fly(k, m, b, T, discount, 1 / k),
                "vanilla",
                tag,
                T,
                "up",
                lvl,
                "fly",
            )
            add(
                f"ratio {tag} {p}",
                call_ratio(k, m, 2.0, T, discount, 1 / k),
                "vanilla",
                tag,
                T,
                "up",
                lvl,
                "ratio",
            )
        for lvl in DOWN_BARRIERS:
            h, m, p = lvl * k, 0.5 * (1.0 + lvl) * k, round(100 * lvl)
            add(
                f"dop {tag} {p}",
                _barrier(k, discount, T, lvl, "down", "discrete"),
                "barrier",
                tag,
                T,
                "down",
                lvl,
                "uoc",
            )
            add(
                f"dop {tag} {p} cont",
                _barrier(k, discount, T, lvl, "down", "continuous"),
                "barrier",
                tag,
                T,
                "down",
                lvl,
                "uoc cont",
            )
            add(
                f"ekp {tag} {p}",
                european_knock_out(k, h, "down", -1, T, discount, 1 / k),
                "barrier",
                tag,
                T,
                "down",
                lvl,
                "eko",
            )
            add(
                f"ps {tag} {p}",
                put_spread(h, k, T, discount, 1 / k),
                "vanilla",
                tag,
                T,
                "down",
                lvl,
                "spread",
            )
            add(
                f"pfly {tag} {p}",
                put_fly(h, m, k, T, discount, 1 / k),
                "vanilla",
                tag,
                T,
                "down",
                lvl,
                "fly",
            )
            add(
                f"pratio {tag} {p}",
                put_ratio(m, k, 2.0, T, discount, 1 / k),
                "vanilla",
                tag,
                T,
                "down",
                lvl,
                "ratio",
            )
    return out


def touch_products(
    spot: float, discount: Any, T: float, level: float, side: str
) -> tuple[Product, Product]:
    """The one-touch (``P(touch)`` undiscounted at the maturity's discount factor) and the regret
    indicator of a daily barrier: ``1{touched} 1{alive side at T} 1{in the money}`` — the
    knock-out's complement inside the European knock-out: ``EKO_digital − UOC_digital``."""
    direction = "up" if side == "up" else "down"
    sched = daily_schedule(T, 252)
    touch = OneTouch(
        level * spot, T, direction, discount, monitoring="discrete", fixing_times=sched, strict=True
    )
    return touch, _Regret(spot, T, level, side, discount)


class _Regret(Product):
    """``DF(T) · 1{touched} · 1{alive side at T} · 1{ITM}`` of the daily barrier (the digital
    counterpart of the regret value)."""

    def __init__(self, spot: float, T: float, level: float, side: str, discount: Any) -> None:
        super().__init__(discount, 1.0)
        self.spot, self.T, self.level, self.side = float(spot), float(T), float(level), side
        self.sched = daily_schedule(T, 252)

    @property
    def fixing_times(self) -> Any:
        return self.sched

    def payoff(self, paths: Any, idx: Any) -> Any:
        cols = idx.indices(self.sched)
        s = paths.spot_at(cols) / self.spot
        b = self.level
        inner = s[:, 1:]
        if self.side == "up":
            touched = np.any(inner > b, axis=1)
            alive = (s[:, -1] < b) & (s[:, -1] > 1.0)
        else:
            touched = np.any(inner < b, axis=1)
            alive = (s[:, -1] > b) & (s[:, -1] < 1.0)
        return np.asarray(float(self.df(self.T)) * (touched & alive), dtype=np.float64)

    def __repr__(self) -> str:
        return f"Regret indicator: {self.side} barrier {self.level:g}, expiry {self.T:g}y"


# --------------------------------------------------------------------------------------------
# environment
# --------------------------------------------------------------------------------------------


@dataclass
class StudyConfig(PayoffConfig):
    """The payoff study's budgets and paths with this study's defaults."""

    out: Path = DEFAULT_OUT
    payoff_out: Path = PAYOFF_OUT
    snapshots: Path = SNAPSHOT_DIR
    history: Path = HISTORY_DIR
    daily_paths: int = DAILY_PATHS
    daily_dates: int | None = None


class StudyEnvironment(PayoffEnvironment):
    """The payoff environment (the desk mark's models) with this study's book."""

    def __init__(self, cfg: StudyConfig) -> None:
        super().__init__(cfg)
        self.scfg = cfg
        self._study_book: dict[str, Entry] | None = None

    @property
    def entries(self) -> dict[str, Entry]:
        if self._study_book is None:
            self._study_book = study_book(self.spot, self.discount)
        return self._study_book


# --------------------------------------------------------------------------------------------
# anchor
# --------------------------------------------------------------------------------------------


def _price_rows(
    entries: Sequence[Entry], priced_by: Model, sim: SimConfig, **tags: Any
) -> list[dict[str, Any]]:
    t0 = time.perf_counter()
    res = MonteCarlo(sim).price_many([e.product for e in entries], priced_by)
    wall = time.perf_counter() - t0
    return [
        {
            "product": e.name,
            "family": e.family,
            "tag": e.tag,
            "T": e.T,
            "side": e.side,
            "level": e.level,
            "role": e.role,
            "value": 100.0 * float(r.mean),
            "stderr": 100.0 * float(r.stderr),
            "n_paths": sim.n_paths,
            "wall_s": wall / len(entries),
            **tags,
        }
        for e, r in zip(entries, res, strict=True)
    ]


def surface_rows(
    entries: Sequence[Entry], surface: ImpliedSurface, **tags: Any
) -> list[dict[str, Any]]:
    """The vanilla structures and the European knock-outs priced off the surface."""
    rows = []
    for e in entries:
        if e.family == "vanilla":
            v = e.structure.surface_price(surface)
        elif e.role == "eko":
            k = float(e.product.strike)  # type: ignore[attr-defined]
            v = european_knock_out_surface_price(
                surface,
                k,
                e.level * k,
                "up" if e.side == "up" else "down",
                1 if e.side == "up" else -1,
                e.T,
                notional=1 / k,
            )
        else:
            continue
        rows.append(
            {
                "product": e.name,
                "family": e.family,
                "tag": e.tag,
                "T": e.T,
                "side": e.side,
                "level": e.level,
                "role": e.role,
                "value": 100.0 * v,
                "stderr": 0.0,
                "n_paths": 0,
                "wall_s": 0.0,
                **tags,
            }
        )
    return rows


def run_anchor(env: StudyEnvironment) -> pd.DataFrame:
    """Part ``anchor`` (module docstring)."""
    sim = env.cfg.sim(env.cfg.price_paths)
    book = env.entries
    barriers = [e for e in book.values() if e.family == "barrier"]
    daily_only = [e for e in barriers if e.role != "uoc cont"]
    cont = [e for e in barriers if e.role == "uoc cont"]
    vanillas = [e for e in book.values() if e.family == "vanilla"]
    rows: list[dict[str, Any]] = []
    rows += surface_rows(barriers + vanillas, env.surface, model="surface")
    for m in ("LV", "1F", "2F"):
        model = env.model(m)
        rows += _price_rows(daily_only + (vanillas if m == "LV" else []), model, sim, model=m)
        # the continuous barriers record every step: priced apart (memory), same seed
        rows += _price_rows(cont, model, sim, model=m)
    for e in daily_only:
        if e.role == "uoc":
            rows += _price_rows([e], env.bs_model(e.T), sim, model="BS (ATM vol)")
    # touch and regret probabilities under the 2F mark (undiscounted)
    model2 = env.model("2F")
    probs: list[Product] = []
    labels: list[tuple[str, str]] = []
    for e in daily_only:
        if e.role != "uoc":
            continue
        touch, regret = touch_products(env.spot, env.discount, e.T, e.level, e.side)
        probs += [touch, regret]
        labels += [(e.name, "p_touch"), (e.name, "p_regret")]
    res = MonteCarlo(sim).price_many(probs, model2)
    for (name, kind), r in zip(labels, res, strict=True):
        e = book[name]
        df = float(env.discount.df(e.T))
        rows.append(
            {
                "product": name,
                "family": "probability",
                "tag": e.tag,
                "T": e.T,
                "side": e.side,
                "level": e.level,
                "role": kind,
                "value": float(r.mean) / df,
                "stderr": float(r.stderr) / df,
                "n_paths": sim.n_paths,
                "wall_s": 0.0,
                "model": "2F",
            }
        )
    return pd.DataFrame(rows)


def _strkeys(d: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(k): v for k, v in d.items()}


def _pick(df: pd.DataFrame, product: str, model: str) -> tuple[float, float]:
    g = df[(df["product"] == product) & (df["model"] == model)]
    if g.empty:
        return float("nan"), float("nan")
    return float(g["value"].iloc[0]), float(g["stderr"].iloc[0])


def decomposition_table(
    anchor: pd.DataFrame, surface: ImpliedSurface, spot: float, discount: Any
) -> pd.DataFrame:
    """Per (maturity, side, level): the European knock-out, the daily and continuous knock-outs
    under the 2F mark, the regret value and share, the continuous discount, the model spread
    and 2F − LV, the touch and regret probabilities, the spread / fly / ratio prices and the
    premium-matched alternatives (module docstring).  Values ×100 in % of the spot."""
    rows = []
    for T, tag in MATURITIES:
        for side, levels in (("up", UP_BARRIERS), ("down", DOWN_BARRIERS)):
            for lvl in levels:
                p = round(100 * lvl)
                uo = f"uoc {tag} {p}" if side == "up" else f"dop {tag} {p}"
                eko = f"eko {tag} {p}" if side == "up" else f"ekp {tag} {p}"
                sp = f"cs {tag} {p}" if side == "up" else f"ps {tag} {p}"
                fl = f"fly {tag} {p}" if side == "up" else f"pfly {tag} {p}"
                ra = f"ratio {tag} {p}" if side == "up" else f"pratio {tag} {p}"
                v2, se2 = _pick(anchor, uo, "2F")
                vc, sec = _pick(anchor, uo + " cont", "2F")
                lv, selv = _pick(anchor, uo, "LV")
                f1, _se1 = _pick(anchor, uo, "1F")
                bs, _ = _pick(anchor, uo, "BS (ATM vol)")
                ek, _ = _pick(anchor, eko, "surface")
                ek2, sek2 = _pick(anchor, eko, "2F")
                vals = [x for x in (lv, f1, v2) if np.isfinite(x)]
                pt, _ = _pick(anchor[anchor["role"] == "p_touch"], uo, "2F")
                pr, _ = _pick(anchor[anchor["role"] == "p_regret"], uo, "2F")
                s_sp, _ = _pick(anchor, sp, "surface")
                s_fl, _ = _pick(anchor, fl, "surface")
                s_ra, _ = _pick(anchor, ra, "surface")
                k = spot
                m = 0.5 * (1.0 + lvl) * k
                premium = v2 / 100.0 * k  # per unit notional (the book's notional is 1/spot)
                row: dict[str, Any] = {
                    "tag": tag,
                    "T": T,
                    "side": side,
                    "level": lvl,
                    "barrier": uo,
                    "BS": bs,
                    "LV": lv,
                    "1F": f1,
                    "2F": v2,
                    "2F_stderr": se2,
                    "2F_cont": vc,
                    "2F_cont_stderr": sec,
                    "EKO_surface": ek,
                    "EKO_2F": ek2,
                    "EKO_2F_stderr": sek2,
                    "regret_value": ek - v2,
                    "regret_share": (ek - v2) / ek if ek else np.nan,
                    "continuous_discount": v2 - vc,
                    "model_spread": (max(vals) - min(vals)) if vals else np.nan,
                    "2F_minus_LV": v2 - lv,
                    "2F_minus_LV_stderr": float(np.hypot(se2, selv)),
                    "p_touch": pt,
                    "p_regret": pr,
                    "spread": s_sp,
                    "fly": s_fl,
                    "ratio_1x2": s_ra,
                    "fly_over_uoc": s_fl / v2 if v2 else np.nan,
                    "fly_units_for_premium": v2 / s_fl if s_fl > 0 else np.nan,
                }
                try:
                    if side == "up":
                        row["ratio_matched"] = ratio_for_premium(k, m, T, surface, premium, cp=1)
                        fly_m = fly_width_for_premium(k, T, surface, premium, discount, cp=1)
                        row["fly_matched_top"] = fly_m.strikes[2] / k
                    else:
                        row["ratio_matched"] = ratio_for_premium(m, k, T, surface, premium, cp=-1)
                        fly_m = fly_width_for_premium(k, T, surface, premium, discount, cp=-1)
                        row["fly_matched_top"] = fly_m.strikes[0] / k
                    row["fly_matched_max_payoff"] = 100.0 * fly_m.max_payoff(0.5 * k, 2.0 * k) / k
                except ValueError as exc:
                    row["ratio_matched"] = np.nan
                    row["fly_matched_top"] = np.nan
                    row["fly_matched_max_payoff"] = np.nan
                    row["note"] = str(exc)
                # the ratio's break-even per unit spot (ratio > 1): m + (m − k)/(ratio − 1) up,
                # m − (k − m)/(ratio − 1) down
                r_m = row["ratio_matched"]
                if np.isfinite(r_m) and r_m > 1:
                    row["ratio_matched_breakeven"] = (
                        (m + (m - k) / (r_m - 1)) / k
                        if side == "up"
                        else (m - (k - m) / (r_m - 1)) / k
                    )
                else:
                    row["ratio_matched_breakeven"] = np.nan
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# map: spot × time to expiry
# --------------------------------------------------------------------------------------------


def run_map(env: StudyEnvironment) -> pd.DataFrame:
    """Part ``map`` (module docstring): the daily barrier under the 2F mark with the leverage
    held in spot (``"model"`` regime), the vanilla alternatives off the surface held in strike;
    at each (spot, τ) the premium-matched fly wing and ratio are re-solved."""
    from volsto.risk.engine import LSVBuilder, RiskEngine

    ctx = env.pricing_ctx()
    builder = ctx.builder
    assert isinstance(builder, LSVBuilder)
    engine = RiskEngine(builder, env.cfg.sim(env.cfg.greek_paths))
    rows: list[dict[str, Any]] = []
    k0 = env.spot
    for side, lvl in (("up", 1.10), ("down", 0.90)):
        b, m = lvl * k0, 0.5 * (1.0 + lvl) * k0
        cp = 1 if side == "up" else -1
        for tau in MAP_TAUS:
            prod = _barrier(k0, env.discount, tau, lvl, side, "discrete")
            # the strike and barrier stay at the inception levels: strike k0, barrier lvl·k0
            for s_rel in MAP_SPOTS:
                s = s_rel * k0
                state = env.state.with_spot(s, label=f"map {side} tau {tau:g} spot {s_rel:g}")
                pr = engine.price(prod, state, "model")
                # the surface of the moved state (sticky strike: the surface in strike is the
                # anchor's, re-read at the new spot through the state's spot)
                from volsto.risk.engine import surface_of

                surf = surface_of(state)
                fly = (
                    call_fly(k0, m, b, tau, env.discount, 1 / k0)
                    if side == "up"
                    else put_fly(b, m, k0, tau, env.discount, 1 / k0)
                )
                ratio = (
                    call_ratio(k0, m, 2.0, tau, env.discount, 1 / k0)
                    if side == "up"
                    else put_ratio(m, k0, 2.0, tau, env.discount, 1 / k0)
                )
                sp = (
                    call_spread(k0, b, tau, env.discount, 1 / k0)
                    if side == "up"
                    else put_spread(b, k0, tau, env.discount, 1 / k0)
                )
                eko = european_knock_out_surface_price(
                    surf, k0, b, "up" if side == "up" else "down", cp, tau, 1 / k0
                )
                v = 100.0 * float(pr.mean)
                row: dict[str, Any] = {
                    "side": side,
                    "level": lvl,
                    "tau": tau,
                    "spot_rel": s_rel,
                    "distance_sd": float(
                        np.log(float(lvl) / float(s_rel))
                        / (float(surf.atm_vol(tau)) * np.sqrt(tau))
                    ),
                    "uoc_2F": v,
                    "uoc_2F_stderr": 100.0 * float(pr.stderr),
                    "EKO": 100.0 * eko,
                    "regret_value": 100.0 * eko - v,
                    "spread": 100.0 * sp.surface_price(surf),
                    "fly": 100.0 * fly.surface_price(surf),
                    "ratio_1x2": 100.0 * ratio.surface_price(surf),
                }
                row["fly_over_uoc"] = row["fly"] / v if v > 0 else np.nan
                premium = float(pr.mean) * k0
                try:
                    if side == "up":
                        row["ratio_matched"] = ratio_for_premium(k0, m, tau, surf, premium, cp=1)
                    else:
                        row["ratio_matched"] = ratio_for_premium(m, k0, tau, surf, premium, cp=-1)
                except ValueError:
                    row["ratio_matched"] = np.nan
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# greeks
# --------------------------------------------------------------------------------------------


def run_greeks(env: StudyEnvironment, names: Sequence[str] = REPRESENTATIVES) -> pd.DataFrame:
    from volsto.risk.engine import LSVBuilder, RiskEngine
    from volsto.risk.report import risk_report

    ctx = env.pricing_ctx()
    builder = ctx.builder
    assert isinstance(builder, LSVBuilder)
    engine = RiskEngine(builder, env.cfg.sim(env.cfg.greek_paths), max_halvings=GREEK_HALVINGS)
    rows: list[dict[str, Any]] = []
    for n in names:
        e = env.entries[n]
        t0 = time.perf_counter()
        rep = risk_report(
            engine,
            e.product,
            env.state,
            sections=GREEK_SECTIONS,
            pillars=GREEK_PILLARS,
            regimes=GREEK_REGIMES,
        )
        for _, r in rep.to_dataframe().iterrows():
            row = {str(k): (v.item() if hasattr(v, "item") else v) for k, v in r.to_dict().items()}
            rows.append({"product": n, "family": e.family, "role": e.role, "scale": 100.0, **row})
        log.info("greeks %s in %.0f s", n, time.perf_counter() - t0)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# hedge
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class HedgeTask:
    product: str
    strategy: str
    world: str = "2F"
    #: the delta regime (the payoff study's field, read by its hedge table; always the model's)
    regime: str = "model"

    @property
    def key(self) -> str:
        text = f"{self.product}__{self.strategy}__{self.world}"
        return "".join(c if c.isalnum() or c in "-_." else "_" for c in text)


def hedge_tasks() -> list[HedgeTask]:
    """The vanilla representatives on the delta in both worlds; the barrier representatives
    with the premium-matched fly and ratio as static proxies plus the delta (2F world) and the
    fly proxy alone."""
    tasks = []
    for p in ("fly 6m 110", "ratio 6m 110", "pfly 6m 90", "pratio 6m 90"):
        tasks += [HedgeTask(p, "delta", "2F"), HedgeTask(p, "delta", "LV")]
    for p in ("uoc 6m 110", "dop 6m 90"):
        tasks += [
            HedgeTask(p, "fly proxy static", "2F"),
            HedgeTask(p, "fly proxy static + delta", "2F"),
            HedgeTask(p, "ratio proxy static + delta", "2F"),
            HedgeTask(p, "fly proxy static", "LV"),
        ]
    return tasks


def proxy_structure(env: StudyEnvironment, e: Entry, kind: str) -> VanillaStructure:
    """The premium-matched fly (``kind="fly"``, wing solved) or ratio of the barrier entry at
    the 2F anchor price read from ``anchor.csv``."""
    anchor = pd.read_csv(env.scfg.out / "anchor.csv")
    v2, _ = _pick(anchor, e.name, "2F")
    k = env.spot
    m = 0.5 * (1.0 + e.level) * k
    premium = v2 / 100.0 * k
    if e.side == "up":
        if kind == "fly":
            return fly_width_for_premium(k, e.T, env.surface, premium, env.discount, cp=1)
        return call_ratio(
            k, m, ratio_for_premium(k, m, e.T, env.surface, premium, cp=1), e.T, env.discount
        )
    if kind == "fly":
        return fly_width_for_premium(k, e.T, env.surface, premium, env.discount, cp=-1)
    return put_ratio(
        m, k, ratio_for_premium(m, k, e.T, env.surface, premium, cp=-1), e.T, env.discount
    )


def run_hedge_task(env: StudyEnvironment, task: HedgeTask, out: Path) -> dict[str, Any]:
    from volsto.hedging.hedger import Hedger, Schedule
    from volsto.hedging.instruments import HedgeInstrument, StaticPortfolio
    from volsto.hedging.strategies import GreekTargetStrategy, Target
    from volsto.studies.payoff import HEDGE_COSTS, _summary

    path = out / "hedge" / f"{task.key}.json"
    if path.exists():
        return dict(json.loads(path.read_text()))
    e = env.entries[task.product]
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
    inst: list[HedgeInstrument] = []
    static: dict[str, float | Callable[[float], float]] = {}
    targets: tuple[Target, ...] = (Target("delta"),)
    if task.strategy.startswith(("fly proxy", "ratio proxy")):
        kind = "fly" if task.strategy.startswith("fly") else "ratio"
        st = proxy_structure(env, e, kind)
        leg = StaticPortfolio(
            name=f"{kind} proxy",
            cost=pc.cost_vol_points,
            strikes=st.strikes,
            weights=st.leg_weights,
            cps=st.cps,
            maturity=st.T,
            discount=e.product.discount,
        )
        inst.append(leg)
        # the product is held long (notional 1/spot): the proxy is sold against it
        static[leg.name] = -float(e.product.notional)
    if task.strategy.endswith("delta") or task.strategy == "delta":
        inst.append(pc.spot_instrument())
    strat = GreekTargetStrategy(targets, inst, static, name=task.strategy)
    t0 = time.perf_counter()
    r = h.run(e.product, strat)
    from volsto.studies.payoff import BookEntry

    be = BookEntry(e.name, e.product, e.role, "% of spot", 100.0)
    res = _summary(task, be, r, time.perf_counter() - t0)  # type: ignore[arg-type]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1, default=str))
    return res


def hedge_table(out: Path, payoff_out: Path | None = None) -> pd.DataFrame:
    """Every saved run as a row (the payoff study's rows of the two barrier representatives
    appended when its table exists)."""
    from volsto.studies.payoff import hedge_table as payoff_hedge_table

    frames = []
    if (out / "hedge").exists() and any((out / "hedge").glob("*.json")):
        df = payoff_hedge_table(out)
        df["source"] = "this study"
        frames.append(df)
    if payoff_out is not None and (payoff_out / "hedge.csv").exists():
        pf = pd.read_csv(payoff_out / "hedge.csv")
        pf = pf[pf["product"].isin(["uoc 6m 110", "dop 6m 90"])].copy()
        pf["source"] = "payoff study"
        frames.append(pf)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# --------------------------------------------------------------------------------------------
# daily (the 127 snapshots)
# --------------------------------------------------------------------------------------------


def snapshot_dates(directory: Path) -> list[tuple[str, Path]]:
    out = []
    for p in sorted(directory.glob("spx_*.yaml")):
        out.append((p.stem.split("_", 1)[1], p))
    return out


def market_metrics(
    surface: ImpliedSurface, spot: float, hist: Mapping[str, pd.Series], date: pd.Timestamp
) -> dict[str, float]:
    """The day's pure-vol metrics off the surface and the statistical / market metrics off the
    histories (realised vols, VIX family, momentum)."""
    out: dict[str, float] = {}
    for T, tag in MATURITIES:
        atm = float(surface.atm_vol(T))
        s90 = float(surface.implied_vol(0.9 * spot, T))
        s110 = float(surface.implied_vol(1.1 * spot, T))
        out[f"atm_{tag}"] = atm
        out[f"skew_{tag}"] = s110 - s90
        out[f"curv_{tag}"] = s90 + s110 - 2.0 * atm
    out["term_1y_3m"] = out["atm_1y"] - out["atm_3m"]
    spx = hist["SPX"]
    past = spx[spx.index <= date]
    for w, nm in ((21, "rv_1m"), (63, "rv_3m")):
        out[nm] = float(realised_vol(past, w).iloc[-1]) if len(past) > w else np.nan
    out["iv_minus_rv_3m"] = out["atm_3m"] - out["rv_3m"]
    out["mom_3m"] = float(past.iloc[-1] / past.iloc[-64] - 1.0) if len(past) > 64 else np.nan
    for nm in ("VIX", "VIX3M", "VVIX", "SKEW"):
        s = hist.get(nm)
        if s is not None:
            sub = s[s.index <= date]
            out[nm.lower()] = float(sub.iloc[-1]) if len(sub) else np.nan
    if np.isfinite(out.get("vix", np.nan)) and np.isfinite(out.get("vix3m", np.nan)):
        out["vix_term"] = out["vix3m"] - out["vix"]
    return out


def load_histories(directory: Path) -> dict[str, pd.Series]:
    out = {}
    for nm in ("SPX", "VIX", "VIX3M", "VVIX", "SKEW"):
        try:
            out[nm] = load_series(nm, directory)
        except FileNotFoundError:
            continue
    return out


def realised_outcomes(
    spx: pd.Series, date: pd.Timestamp, horizon: int, level: float, side: str
) -> dict[str, float]:
    """The realised payoffs (per unit spot) of the structures entered on ``date`` with
    ``horizon`` trading days to run, when the history reaches the expiry."""
    from volsto.studies.history_stats import structure_payoffs, touched_strict

    fut = spx[spx.index >= date]
    if len(fut) <= horizon:
        return {}
    ratios = (fut.iloc[: horizon + 1] / fut.iloc[0]).to_numpy(dtype=np.float64)[None, :]
    touched = touched_strict(ratios, level, side)
    pay = structure_payoffs(ratios[:, -1], touched, strike=1.0, barrier=level, direction=side)
    return {f"realised_{k}": 100.0 * float(v[0]) for k, v in pay.items()}


def run_daily(env: StudyEnvironment) -> pd.DataFrame:
    """Part ``daily`` (module docstring)."""
    from volsto.market.loaders import load_ssvi_surface

    hist = load_histories(env.scfg.history)
    dates = snapshot_dates(env.scfg.snapshots)
    if env.scfg.daily_dates is not None:
        dates = dates[: env.scfg.daily_dates]
    sim = env.cfg.sim(env.scfg.daily_paths)
    rows: list[dict[str, Any]] = []
    for i, (d, path) in enumerate(dates, start=1):
        t0 = time.perf_counter()
        surf = load_ssvi_surface(path)
        fc = surf.forward_curve
        spot, disc = float(fc.spot), fc.rate_curve
        lv = LocalVol(LocalVolSurface.from_implied(surf, DAILY_LV), fc)
        book = study_book(spot, disc)
        daily_bar = [e for e in book.values() if e.family == "barrier" and e.role == "uoc"]
        res = MonteCarlo(sim).price_many([e.product for e in daily_bar], lv)
        date = pd.Timestamp(d)
        metrics = market_metrics(surf, spot, hist, date)
        for e, r in zip(daily_bar, res, strict=True):
            p = round(100 * e.level)
            eko_name = f"eko {e.tag} {p}" if e.side == "up" else f"ekp {e.tag} {p}"
            sp_name = f"cs {e.tag} {p}" if e.side == "up" else f"ps {e.tag} {p}"
            fl_name = f"fly {e.tag} {p}" if e.side == "up" else f"pfly {e.tag} {p}"
            ra_name = f"ratio {e.tag} {p}" if e.side == "up" else f"pratio {e.tag} {p}"
            eko_v = surface_rows([book[eko_name]], surf)[0]["value"]
            v = 100.0 * float(r.mean)
            fl = 100.0 * book[fl_name].structure.surface_price(surf)
            k, m = spot, 0.5 * (1.0 + e.level) * spot
            premium = float(r.mean) * spot
            try:
                rm = (
                    ratio_for_premium(k, m, e.T, surf, premium, cp=1)
                    if e.side == "up"
                    else ratio_for_premium(m, k, e.T, surf, premium, cp=-1)
                )
            except ValueError:
                rm = np.nan
            row: dict[str, Any] = {
                "date": d,
                "product": e.name,
                "tag": e.tag,
                "T": e.T,
                "side": e.side,
                "level": e.level,
                "spot": spot,
                "uoc_LV": v,
                "uoc_LV_stderr": 100.0 * float(r.stderr),
                "EKO": eko_v,
                "regret_value": eko_v - v,
                "regret_share": (eko_v - v) / eko_v if eko_v else np.nan,
                "spread": 100.0 * book[sp_name].structure.surface_price(surf),
                "fly": fl,
                "ratio_1x2": 100.0 * book[ra_name].structure.surface_price(surf),
                "fly_over_uoc": fl / v if v > 0 else np.nan,
                "ratio_matched": rm,
                "distance_sd": float(np.log(e.level) / (metrics[f"atm_{e.tag}"] * np.sqrt(e.T))),
                **metrics,
            }
            if "SPX" in hist:
                row.update(realised_outcomes(hist["SPX"], date, HORIZONS[e.tag], e.level, e.side))
            rows.append(row)
        log.info("daily %d/%d %s in %.1f s", i, len(dates), d, time.perf_counter() - t0)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# history (the statistical layer)
# --------------------------------------------------------------------------------------------


def run_history(env: StudyEnvironment, anchor: pd.DataFrame | None = None) -> pd.DataFrame:
    """Part ``history`` (module docstring): the unconditional and regime-conditional
    frequency tables over the SPX history and the filtered historical simulation at the
    anchor's vols, with the 2F mark's probabilities beside them when ``anchor`` is given."""
    hist = load_histories(env.scfg.history)
    spx = hist["SPX"]
    r = log_returns(spx)
    vix = hist.get("VIX")
    rv = realised_vol(spx, 63)
    rows: list[dict[str, Any]] = []
    # regimes, aligned on the entry date
    regimes: dict[str, pd.Series] = {}
    if vix is not None:
        v = vix.reindex(spx.index).ffill()
        q = v.quantile([1 / 3, 2 / 3]).to_numpy()
        regimes["vix_tercile"] = regime_bins(v, [0, q[0], q[1], 1e9], ["low", "mid", "high"])
        ivrv = (v / 100.0 - rv).dropna()
        regimes["iv_minus_rv"] = regime_bins(ivrv, [-10, 0, 10], ["negative", "positive"])
    mom = (spx / spx.shift(63) - 1.0).dropna()
    regimes["momentum_3m"] = regime_bins(mom, [-10, 0, 10], ["down", "up"])
    anchor_date = pd.Timestamp("2022-12-30")
    r_anchor = r[r.index <= anchor_date]
    rv_anchor = float(realised_vol(spx[spx.index <= anchor_date], 63).iloc[-1])
    for T, tag in MATURITIES:
        n = HORIZONS[tag]
        iv = float(env.surface.atm_vol(T))
        for side, levels in (("up", UP_BARRIERS), ("down", DOWN_BARRIERS)):
            for lvl in levels:
                st = barrier_path_statistics(spx, n, barrier=lvl, direction=side)
                base = {"tag": tag, "T": T, "side": side, "level": lvl, "horizon_days": n}
                ft = frequency_table(st, n)
                rows.append(
                    {
                        **base,
                        "layer": "history",
                        "regime": "all",
                        "group": "all",
                        **_strkeys(ft.iloc[0].to_dict()),
                    }
                )
                for rname, reg in regimes.items():
                    gt = frequency_table(st, n, groups=reg)
                    for _, g in gt.iterrows():
                        rows.append(
                            {**base, "layer": "history", "regime": rname, **_strkeys(g.to_dict())}
                        )
                # the window since 2022-12-30 only (the anchor's own future, 2023–2026)
                st_after = st[st.index >= anchor_date]
                if len(st_after) > n:
                    ft = frequency_table(st_after, n)
                    rows.append(
                        {
                            **base,
                            "layer": "history since anchor",
                            "regime": "all",
                            "group": "all",
                            **_strkeys(ft.iloc[0].to_dict()),
                        }
                    )
                fhs_layers = (
                    ("FHS at realised 3m", rv_anchor, True),
                    ("FHS at implied ATM", iv, True),
                    ("FHS at implied ATM, historical drift", iv, False),
                )
                for layer, vol, demean in fhs_layers:
                    paths = filtered_historical_paths(
                        r_anchor, n, FHS_PATHS, vol_target=vol, seed=FHS_SEED, demean=demean
                    )
                    tab = fhs_barrier_table(paths, barrier=lvl, direction=side)
                    lt = np.log(paths[:, -1])
                    row = {
                        **base,
                        "layer": layer,
                        "fhs_terminal_vol": float(lt.std(ddof=1) / np.sqrt(n / TRADING_DAYS)),
                        "fhs_terminal_mean": float(lt.mean()),
                        "regime": "all",
                        "group": "all",
                        "n": FHS_PATHS,
                        "n_eff": FHS_PATHS,
                        "vol_target": vol,
                    }
                    for k2, (mu, se) in tab.items():
                        row[k2] = mu
                        row[f"{k2}_stderr"] = se
                    rows.append(row)
                if anchor is not None:
                    uo = (
                        f"uoc {tag} {round(100 * lvl)}"
                        if side == "up"
                        else f"dop {tag} {round(100 * lvl)}"
                    )
                    pt, pt_se = _pick(anchor[anchor["role"] == "p_touch"], uo, "2F")
                    pr, pr_se = _pick(anchor[anchor["role"] == "p_regret"], uo, "2F")
                    v2, se2 = _pick(anchor, uo, "2F")
                    rows.append(
                        {
                            **base,
                            "layer": "2F mark (Q)",
                            "regime": "all",
                            "group": "all",
                            "touched": pt,
                            "touched_stderr": pt_se,
                            "regret": pr,
                            "regret_stderr": pr_se,
                            "uoc": v2 / 100.0,
                            "uoc_stderr": se2 / 100.0,
                        }
                    )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# the decision table
# --------------------------------------------------------------------------------------------


def decision_table(
    decomp: pd.DataFrame, history: pd.DataFrame, hedge: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Per (maturity, side, level) and P-measure layer: the expected payoff per unit premium of
    the barrier and of its premium-matched fly and 1×2 ratio (the fly re-solved on the layer's
    own payoffs is not available: the layer's fly is the midpoint fly scaled to the premium —
    ``fly_units × E^P[fly payoff]`` — and the ratio the 1×2; the matched ratio's payoff is
    ``E^P[vanilla] − ratio · E^P[call(mid)]``, reconstructed from the layer's spread / ratio
    legs), the P-minus-Q touch and regret gaps, the model-risk band and the hedge cost as
    shares of the premium, and the verdict (module docstring)."""
    rows = []
    for _, d in decomp.iterrows():
        key = (d["tag"], d["side"], d["level"])
        hist = history[
            (history["tag"] == key[0]) & (history["side"] == key[1]) & (history["level"] == key[2])
        ]
        q = hist[hist["layer"] == "2F mark (Q)"]
        p_touch_q = float(q["touched"].iloc[0]) if len(q) else np.nan
        p_regret_q = float(q["regret"].iloc[0]) if len(q) else np.nan
        premium = float(d["2F"])  # % of spot
        fly_units = float(d["fly_units_for_premium"])
        for _, h in hist[(hist["group"] == "all") & (hist["layer"] != "2F mark (Q)")].iterrows():
            e_uoc = 100.0 * float(h["uoc"])
            e_fly = 100.0 * float(h["fly"]) * fly_units
            # the matched ratio: vanilla − ratio × call(mid); the layer gives vanilla = spread +
            # call(B) … reconstruct from the 1×2: ratio_1x2 = vanilla − 2 call(mid), spread =
            # vanilla − call(B).  With fly = vanilla − a call(mid) + b call(B) (a, b the fly
            # weights) the three layer payoffs identify vanilla, call(mid), call(B).
            lvl = float(d["level"])
            e_sp, e_r2, e_f = (
                100.0 * float(h["spread"]),
                100.0 * float(h["ratio"]),
                100.0 * float(h["fly"]),
            )
            k, m, b = 1.0, 0.5 * (1.0 + lvl), lvl
            if d["side"] == "up":
                wa, wb = (b - k) / (b - m), (m - k) / (b - m)
            else:
                wa, wb = (k - b) / (m - b), (k - m) / (m - b)
            # unknowns V (vanilla), C_m (mid option), C_b (barrier-strike option):
            # spread = V − C_b ; ratio = V − 2 C_m ; fly = V − wa C_m + wb C_b
            A = np.array([[1.0, 0.0, -1.0], [1.0, -2.0, 0.0], [1.0, -wa, wb]])
            sol = np.linalg.solve(A, np.array([e_sp, e_r2, e_f]))
            e_vanilla, e_mid = float(sol[0]), float(sol[1])
            rm = float(d["ratio_matched"])
            e_ratio_m = e_vanilla - rm * e_mid if np.isfinite(rm) else np.nan
            row: dict[str, Any] = {
                "tag": d["tag"],
                "side": d["side"],
                "level": lvl,
                "layer": h["layer"],
                "premium_2F": premium,
                "E_uoc": e_uoc,
                "E_fly_matched_units": e_fly,
                "E_ratio_matched": e_ratio_m,
                "uoc_payoff_to_premium": e_uoc / premium if premium > 0 else np.nan,
                "fly_payoff_to_premium": e_fly / premium if premium > 0 else np.nan,
                "ratio_payoff_to_premium": e_ratio_m / premium if premium > 0 else np.nan,
                "p_touch_P": float(h["touched"]),
                "p_touch_Q": p_touch_q,
                "p_regret_P": float(h["regret"]),
                "p_regret_Q": p_regret_q,
                "touch_gap_P_minus_Q": float(h["touched"]) - p_touch_q,
                "regret_gap_P_minus_Q": float(h["regret"]) - p_regret_q,
                "model_band_share": float(d["model_spread"]) / premium if premium > 0 else np.nan,
                "regret_share_Q": float(d["regret_share"]),
            }
            if hedge is not None and len(hedge):
                hb = hedge[
                    (hedge["product"] == d["barrier"])
                    & (hedge["strategy"] == "delta")
                    & (hedge["world"] == "2F")
                ]
                if len(hb):
                    row["hedge_std_share"] = float(hb["std_zc"].iloc[0]) / premium
                    row["hedge_cost_share"] = float(hb["costs_mean"].iloc[0]) / premium
            edge = row["uoc_payoff_to_premium"] - row["fly_payoff_to_premium"]
            row["edge_uoc_minus_fly"] = edge
            row["verdict_vs_fly"] = (
                "barrier"
                if edge > row["model_band_share"]
                else ("fly" if edge < 0 else "indifferent (inside the model band)")
            )
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


def setup_table(env: StudyEnvironment) -> pd.DataFrame:
    """The run's setup: the payoff study's rows plus this study's path counts."""
    from volsto.studies.payoff import setup_table as payoff_setup

    df = payoff_setup(env)
    extra = pd.DataFrame(
        [
            {"key": "daily_paths", "value": float(env.scfg.daily_paths)},
            {"key": "fhs_paths", "value": float(FHS_PATHS)},
            {"key": "snapshots", "value": float(len(snapshot_dates(env.scfg.snapshots)))},
        ]
    )
    return pd.concat([df, extra], ignore_index=True)


def run_part(env: StudyEnvironment, part: str) -> PartResult:
    out = env.scfg.out
    out.mkdir(parents=True, exist_ok=True)
    setup_table(env).to_csv(out / "setup.csv", index=False)
    n_cal0 = len(env.calibrated)
    t0 = time.perf_counter()
    target = out / f"{part}.csv"
    if part == "hedge":
        tasks = hedge_tasks()
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
        hedge_table(out, env.scfg.payoff_out).to_csv(target, index=False)
    elif target.exists():
        log.info("%s exists: skipped", target)
    else:
        if part == "anchor":
            df = run_anchor(env)
            df.to_csv(target, index=False)
            decomposition_table(df, env.surface, env.spot, env.discount).to_csv(
                out / "decomposition.csv", index=False
            )
        elif part == "map":
            run_map(env).to_csv(target, index=False)
        elif part == "greeks":
            run_greeks(env).to_csv(target, index=False)
        elif part == "daily":
            run_daily(env).to_csv(target, index=False)
        elif part == "history":
            anchor = pd.read_csv(out / "anchor.csv") if (out / "anchor.csv").exists() else None
            df = run_history(env, anchor)
            df.to_csv(target, index=False)
            if (out / "decomposition.csv").exists():
                hedge = pd.read_csv(out / "hedge.csv") if (out / "hedge.csv").exists() else None
                decision_table(pd.read_csv(out / "decomposition.csv"), df, hedge).to_csv(
                    out / "decision.csv", index=False
                )
        else:
            raise ValueError(f"unknown part {part!r}")
    return PartResult(part, time.perf_counter() - t0, len(env.calibrated) - n_cal0, target)


__all__ = [
    "DOWN_BARRIERS",
    "HORIZONS",
    "MATURITIES",
    "PARTS",
    "REPRESENTATIVES",
    "UP_BARRIERS",
    "Entry",
    "HedgeTask",
    "PartResult",
    "StudyConfig",
    "StudyEnvironment",
    "decision_table",
    "decomposition_table",
    "hedge_tasks",
    "market_metrics",
    "run_part",
    "setup_table",
    "study_book",
    "surface_rows",
    "touch_products",
]
