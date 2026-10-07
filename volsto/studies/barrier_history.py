"""The barrier study on 2007–2026 (``outputs/interview/BARRIER_STUDY_SPEC.md``; SPEC §8.6): the
knock-out against its vanilla alternatives on every weekly entry of the ORATS history.

This module holds the study's definitions and its day engine; ``scripts/barrier_history.py``
runs the passes and ``volsto/studies/barrier_history_report.py`` writes the tables.

**Trades** (:func:`entry_dates`, :func:`expiry_date`, :func:`barrier_levels`, :func:`cells_of`).
Strike 100 % of the entry snapshot spot, maturities 1 / 3 / 6 / 12 calendar months (the last
trading day on or before), barriers in percent of spot and in standard deviations
(``K·exp(±c·s·√T)``, ``s`` the at-the-money-forward vol of the maturity), both sides.

**Structures** (:func:`structure_legs`): every vanilla structure is a tuple of :class:`Leg` —
an option of the trade's side (calls on the call side, puts on the put side), a cash digital
(priced by the library's centred strike difference, paid exactly at expiry) or a forward (the
pilot's accounting check) — so a structure's mark, its bumped marks and its payoff are sums over
legs (:func:`leg_values`, :func:`leg_payoffs`).

**The day engine** (:func:`simulate_day`, :func:`knock_out_values`): one local-vol path set per
(day, bump) on the grid of the future trading days, shared by every live trade; the daily
knock-out is a functional of the running extreme of the simulated closes (strict), the
continuous one of the running extreme of Brownian-bridge step extremes (sampled with common
uniforms, so the three bumped sets differ only by the surface).  The sticky-strike bump scales
the spot and the forward curve by ``1 ± e`` and keeps the implied vol of every absolute strike
(:class:`MovedSurface`); the local vol is rebuilt from the bumped surface.

**Accounting** (:func:`knock_dates`, :func:`hedge_pnl`): observation on the official closes
(daily) or highs / lows (continuous); the hedge is the forward to the trade's own expiry, so
every cash flow is valued at expiry.

Checked by ``tests/test_barrier_history.py`` (the pure functions: dates, structures, payoffs,
the model-free facts, the knock rules, the hedge accounting, the bucketing, the bootstrap).
"""

from __future__ import annotations

import datetime as _dt
import json
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from scipy.special import ndtr

from volsto.config import LocalVolConfig, SimConfig
from volsto.engine.grid import TimeGrid
from volsto.engine.mc import MonteCarlo
from volsto.market.curves import ForwardCurve
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import ImpliedSurface
from volsto.models.base import Model
from volsto.models.localvol import LocalVol

FloatArray = NDArray[np.float64]

ROOT: Final = Path(__file__).resolve().parents[2]
OUT: Final = ROOT / "outputs" / "interview"
DAY_CACHE: Final = OUT / "cache" / "days"
RESULTS: Final = OUT / "barrier_results"
HISTORY: Final = ROOT / "data" / "history"

MATURITY_MONTHS: Final[tuple[int, ...]] = (1, 3, 6, 12)
PCT_BARRIERS: Final[dict[int, tuple[float, ...]]] = {
    1: (1.05, 1.10, 1.15, 1.20),
    -1: (0.95, 0.90, 0.85, 0.80),
}
SD_BARRIERS: Final[tuple[float, ...]] = (0.25, 0.5, 1.0, 1.5, 2.0)
FLY_N: Final[tuple[int, ...]] = (2, 3, 4, 6)
#: Relative bump of spot and forward curve for the sticky-strike delta (spec §5.2).
BUMP: Final[float] = 0.01
#: Half-width of the centred strike difference pricing a digital (the library's).
DIGITAL_REL_WIDTH: Final[float] = 0.005
#: The agreed tolerances of the surface (tests/test_import_orats.py; owner's decision F).
TOL_FORWARD_MEDIAN_BP: Final[float] = 3.0
TOL_FORWARD_MAX_BP: Final[float] = 15.0
TOL_RMS_3M_2Y_VP: Final[float] = 0.25
TOL_RMS_6M_2Y_VP: Final[float] = 0.20
PERIODS: Final[tuple[tuple[str, str, str], ...]] = (
    ("2007 to 2011-10-03", "2007-01-01", "2011-10-03"),
    ("2011-10-04 to 2017-05-09", "2011-10-04", "2017-05-09"),
    ("2017-05-10 to 2019-02-04", "2017-05-10", "2019-02-04"),
    ("2019-02-05 to 2021-05-27", "2019-02-05", "2021-05-27"),
    ("2021-05-28 on", "2021-05-28", "2100-01-01"),
)
#: Local-vol grid of the study: the horizon is one year (plus the bump's margin).
LV_CONFIG: Final = LocalVolConfig(t_max=1.1)
#: Paths of a day's local-vol set (antithetic pairs).
LV_PATHS: Final[int] = 100_000


# --------------------------------------------------------------------------------------------
# dates
# --------------------------------------------------------------------------------------------


def period_of(date: str) -> str:
    """The settlement period (:data:`PERIODS`) of an ISO date."""
    for name, lo, hi in PERIODS:
        if lo <= date <= hi:
            return name
    raise ValueError(f"{date} is before the study's first period")


def load_ohlc(path: Path | None = None) -> pd.DataFrame:
    """``date``-indexed open, high, low, close of the SPX (``data/history/SPX_ohlc.csv``)."""
    frame = pd.read_csv(path or HISTORY / "SPX_ohlc.csv", dtype={"date": str})
    return frame.set_index("date")[["open", "high", "low", "close"]].astype(np.float64)


def trading_calendar(known: Sequence[str], until: str) -> list[str]:
    """``known`` (ISO trading days, ascending) extended by the weekdays up to ``until``: the
    future has no holiday calendar (spec contradiction 9)."""
    out = list(known)
    day = _dt.date.fromisoformat(out[-1]) + _dt.timedelta(days=1)
    end = _dt.date.fromisoformat(until)
    while day <= end:
        if day.weekday() < 5:
            out.append(day.isoformat())
        day += _dt.timedelta(days=1)
    return out


def entry_dates(calendar: Sequence[str], first: str, last: str) -> list[str]:
    """The first trading day of each ISO week between ``first`` and ``last`` (included)."""
    out: list[str] = []
    seen: set[tuple[int, int]] = set()
    for d in calendar:
        if d < first or d > last:
            continue
        iso = _dt.date.fromisoformat(d).isocalendar()
        key = (iso[0], iso[1])
        if key not in seen:
            seen.add(key)
            out.append(d)
    return out


def monthly_subset(entries: Sequence[str]) -> list[str]:
    """The first entry date of each calendar month."""
    out: list[str] = []
    seen: set[str] = set()
    for d in entries:
        if d[:7] not in seen:
            seen.add(d[:7])
            out.append(d)
    return out


def add_months(day: _dt.date, months: int) -> _dt.date:
    """``day`` plus calendar months, the day of month capped at the month's end."""
    y, m = divmod(day.month - 1 + months, 12)
    year, month = day.year + y, m + 1
    nxt = _dt.date(year + (month == 12), month % 12 + 1, 1)
    return _dt.date(year, month, min(day.day, (nxt - _dt.timedelta(days=1)).day))


def expiry_date(entry: str, months: int, calendar: Sequence[str]) -> str:
    """The last trading day on or before the entry date plus ``months`` calendar months."""
    target = add_months(_dt.date.fromisoformat(entry), months).isoformat()
    i = int(np.searchsorted(np.asarray(calendar), target, side="right")) - 1
    if i < 0 or calendar[i] <= entry:
        raise ValueError(f"no trading day after {entry} on or before {target}")
    return str(calendar[i])


def year_fraction(d0: str, d1: str) -> float:
    """ACT/365 between two ISO dates."""
    return (_dt.date.fromisoformat(d1) - _dt.date.fromisoformat(d0)).days / 365.0


# --------------------------------------------------------------------------------------------
# structures
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Leg:
    """``qty`` of an option (``"opt"``, of the trade's side), a cash digital paying 1 beyond
    the strike on the trade's side (``"dig"``) or a forward (``"fwd"``, payoff ``S − strike``)."""

    kind: str
    strike: float
    qty: float


def barrier_levels(side: int, strike: float, atm_vol: float, T: float) -> dict[str, float]:
    """The barriers of a cell: ``p105`` … in percent of the strike, ``s0.25`` … in standard
    deviations ``K·exp(side·c·s·√T)`` (spec §3.3)."""
    out = {f"p{round(100 * b)}": strike * b for b in PCT_BARRIERS[side]}
    for c in SD_BARRIERS:
        out[f"s{c:g}"] = strike * float(np.exp(side * c * atm_vol * np.sqrt(T)))
    return out


def structure_legs(side: int, K: float, B: float, S0: float) -> dict[str, tuple[Leg, ...]]:
    """The vanilla structures of spec §3.4 / §3.5 for strike ``K`` and barrier ``B`` (``B > K``
    on the call side, ``B < K`` on the put side), per unit of the underlying (the caller
    divides by ``S0``).  ``VAN`` is the vanilla at ``K`` and ``FWD`` the forward struck at
    ``K`` (pilot checks 2, 3 and 7)."""
    if side not in (1, -1) or side * (B - K) <= 0:
        raise ValueError("the barrier must be beyond the strike on the trade's side")
    w = abs(B - K)
    mid = 0.5 * (K + B)

    def o(x: float, q: float) -> Leg:
        return Leg("opt", float(x), float(q))

    out: dict[str, tuple[Leg, ...]] = {
        "VAN": (o(K, 1.0),),
        "FWD": (Leg("fwd", float(K), 1.0),),
        "B3": (o(K, 1.0), o(B, -1.0)),
        "B4": (o(K, 1.0), o(B, -2.0)),
        "B6": (o(K, 1.0), o(B, -1.0), Leg("dig", float(B), -w)),
        "C7": (o(K, 1.0), o(B, -1.0), Leg("dig", float(B), -2.0 * w)),
        "C8": (
            o(K, 1.0),
            o(B, -(1.0 - K / B)),
            o(B * B / K, -(K / B)),
            Leg("dig", float(B), -2.0 * w),
        ),
        "E12": (o(K, 1.0), o(mid, -2.0), o(B, 1.0)),
        "E13": (o(K, 1.0), o(mid, -2.0)),
    }
    for n in FLY_N:
        far = B + side * w / (n - 1)
        out[f"B5_{n}"] = (o(K, 1.0), o(B, -float(n)), o(far, float(n - 1)))
    for tag, pct in (("1", 0.01), ("2", 0.02)):
        d = min(pct * S0, w)
        # the digital replaced by a tradable spread of width d centred on the barrier
        spread = (o(B - side * d / 2.0, -2.0 * w / d), o(B + side * d / 2.0, 2.0 * w / d))
        out[f"C9_7_{tag}"] = (o(K, 1.0), o(B, -1.0), *spread)
        out[f"C9_8_{tag}"] = (o(K, 1.0), o(B, -(1.0 - K / B)), o(B * B / K, -(K / B)), *spread)
    return out


def black(
    F: FloatArray, K: FloatArray, T: FloatArray, vol: FloatArray, cp: int, df: FloatArray
) -> FloatArray:
    """Black-76 ``df · cp · [F N(cp d1) − K N(cp d2)]``; intrinsic at zero total variance."""
    s = vol * np.sqrt(np.maximum(T, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log(F / K) + 0.5 * s * s) / s
    live = cp * (F * ndtr(cp * d1) - K * ndtr(cp * (d1 - s)))
    return np.asarray(df * np.where(s > 0, live, np.maximum(cp * (F - K), 0.0)), dtype=np.float64)


def leg_value(leg: Leg, side: int, surface: ImpliedSurface, T: float, ratio: float = 1.0) -> float:
    """Discounted value of ``leg`` (with its quantity) off ``surface`` at maturity ``T``, the
    forward scaled by ``ratio`` with the implied vol of each absolute strike unchanged (the
    sticky-strike bump; ``ratio`` 1 is the mark)."""
    F = np.asarray(float(surface.forward(T)) * ratio)
    df = np.asarray(float(surface.discount.df(T)))
    Ta = np.asarray(T)
    if leg.kind == "fwd":
        return float(leg.qty * df * (F - leg.strike))

    def opt(x: float) -> float:
        vol = np.asarray(surface.implied_vol(x, T), dtype=np.float64).reshape(())
        return float(black(F, np.asarray(x), Ta, vol, side, df))

    if leg.kind == "opt":
        return leg.qty * opt(leg.strike)
    if leg.kind == "dig":
        h = DIGITAL_REL_WIDTH * leg.strike
        return leg.qty * side * (opt(leg.strike - h) - opt(leg.strike + h)) / (2.0 * h)
    raise ValueError(f"unknown leg kind {leg.kind!r}")


def legs_value(
    legs: Sequence[Leg], side: int, surface: ImpliedSurface, T: float, ratio: float = 1.0
) -> float:
    return float(sum(leg_value(leg, side, surface, T, ratio) for leg in legs))


def legs_payoff(legs: Sequence[Leg], side: int, spot: float | FloatArray) -> FloatArray:
    """Payoff at expiry on ``spot`` (broadcasts): options ``(side (S − X))⁺``, digitals
    ``1{side (S − X) > 0}``, forwards ``S − X``."""
    s = np.asarray(spot, dtype=np.float64)
    total = np.zeros_like(s)
    for leg in legs:
        if leg.kind == "opt":
            total = total + leg.qty * np.maximum(side * (s - leg.strike), 0.0)
        elif leg.kind == "dig":
            total = total + leg.qty * (side * (s - leg.strike) > 0.0)
        elif leg.kind == "fwd":
            total = total + leg.qty * (s - leg.strike)
        else:
            raise ValueError(f"unknown leg kind {leg.kind!r}")
    return np.asarray(total, dtype=np.float64)


def knock_out_payoff(side: int, K: float, spot: float, knocked: bool) -> float:
    """``(side (S_T − K))⁺`` unless knocked; no rebate."""
    return 0.0 if knocked else float(max(side * (spot - K), 0.0))


def lower_bound(
    side: int, K: float, B: float, surface: ImpliedSurface, T: float, n: int = 81
) -> tuple[float, float]:
    """The model-free lower bound of the knock-out (Brown, Hobson, Rogers 2001) and its strike:
    ``sup_b [C(K) − (B − K)/(B − b)·C(b)]`` over ``b`` between ``K`` and ``B`` (``b = K`` gives
    zero), by a search on ``n`` strikes."""
    base = leg_value(Leg("opt", K, 1.0), side, surface, T)
    best, at = 0.0, K
    for b in np.linspace(K, B, n + 1)[:-1]:
        value = base - (B - K) / (B - b) * leg_value(Leg("opt", float(b), 1.0), side, surface, T)
        if value > best:
            best, at = float(value), float(b)
    return best, at


# --------------------------------------------------------------------------------------------
# realised path: knocks, payoffs, hedge accounting
# --------------------------------------------------------------------------------------------


def knock_dates(
    side: int, B: float, ohlc: pd.DataFrame, observations: Sequence[str]
) -> tuple[str | None, str | None]:
    """``(daily knock date, continuous knock date)`` on the observation dates (spec §5.4):
    daily on the first close strictly beyond ``B``; continuous on the first high at or above
    (calls) / low at or below (puts) ``B``.  ``None`` when not knocked."""
    obs = ohlc.reindex(list(observations))
    if side > 0:
        daily, cont = obs["close"] > B, obs["high"] >= B
    else:
        daily, cont = obs["close"] < B, obs["low"] <= B
    d = daily.to_numpy()
    c = cont.to_numpy()
    return (
        str(obs.index[int(np.argmax(d))]) if d.any() else None,
        str(obs.index[int(np.argmax(c))]) if c.any() else None,
    )


def hedge_pnl(delta_f: FloatArray, forwards: FloatArray, closing_forward: float) -> FloatArray:
    """Hedge P&L per interval, settled at the trade's expiry (spec §5.3): ``h_i = −ΔF_i ·
    (F_{i+1} − F_i)`` for the snapshots ``i = 0 … n−1`` of the life, the last interval closed at
    ``closing_forward`` (the official close at expiry, the knock-day forward, …).
    ``forwards[i]`` is the forward to the trade's expiry seen on day ``i``."""
    nxt = np.append(forwards[1:], closing_forward)
    return np.asarray(-delta_f * (nxt - forwards), dtype=np.float64)


# --------------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------------


def expanding_tercile(values: pd.Series, min_history: int) -> pd.Series:
    """Label each value ``low`` / ``mid`` / ``high`` against the terciles of the values strictly
    before it (expanding window); ``no history`` until ``min_history`` earlier values exist.
    ``values`` is in time order; NaN stays NaN-labelled ``no history``."""
    x = values.to_numpy(dtype=np.float64)
    out = np.full(x.size, "no history", dtype=object)
    for i in range(x.size):
        past = x[:i]
        past = past[~np.isnan(past)]
        if past.size < min_history or np.isnan(x[i]):
            continue
        lo, hi = np.quantile(past, [1.0 / 3.0, 2.0 / 3.0])
        out[i] = "low" if x[i] <= lo else ("high" if x[i] > hi else "mid")
    return pd.Series(out, index=values.index)


def sd_bin(distance: float) -> str:
    """Fixed bins of the distance to the barrier in standard deviations (spec §7)."""
    if not np.isfinite(distance):
        return "nan"
    if distance < 0.5:
        return "<0.5"
    if distance < 1.0:
        return "0.5-1"
    if distance < 1.5:
        return "1-1.5"
    return ">1.5"


def block_bootstrap(
    x: FloatArray, block: int, n_resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """Mean and circular-block-bootstrap standard error of the mean of ``x`` (time-ordered,
    NaN dropped), blocks of ``block`` consecutive observations."""
    x = np.asarray(x, dtype=np.float64)
    x = x[~np.isnan(x)]
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(x[0]), float("nan")
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n, size=(n_resamples, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    means = x[idx.reshape(n_resamples, -1)[:, :n]].mean(axis=1)
    return float(x.mean()), float(means.std(ddof=1))


def effective_sample(n: int, spacing_days: float, maturity_days: float) -> float:
    """``n × spacing / maturity`` (capped at ``n``): the number of independent windows."""
    return float(min(n, n * spacing_days / maturity_days))


# --------------------------------------------------------------------------------------------
# the day's market
# --------------------------------------------------------------------------------------------


class MovedSurface(ImpliedSurface):
    """``base`` with the spot and forward curve scaled by ``ratio`` and the clock advanced by
    ``dt`` years, the implied vol of every absolute strike and expiry date unchanged:
    ``σ_new(K, T) = σ_base(K, T + dt)``.  ``dt = 0`` is the sticky-strike bump (spec §5.2);
    ``dt > 0`` carries the previous day's vols to a day whose own surface cannot be built
    (spec §2.4)."""

    def __init__(self, base: ImpliedSurface, ratio: float, dt: float = 0.0) -> None:
        fc = base.forward_curve
        if dt > 0.0:
            # forwards of the same expiry dates, scaled: F_new(T) = ratio · F_base(T + dt)
            rate = fc.rate_curve.rolled(dt)
            div = fc.dividend_curve.rolled(dt)
            spot = float(ratio * fc.forward(dt))
            curve = ForwardCurve(spot, rate, div)
            discount = base.discount.rolled(dt)
        else:
            curve = fc.with_spot(float(fc.spot) * float(ratio))
            discount = base.discount
        super().__init__(curve, discount, base.max_maturity - dt)
        self.base = base
        self.ratio = float(ratio)
        self.dt = float(dt)

    def total_variance(self, k: Any, T: Any) -> FloatArray:
        T_ = np.asarray(T, dtype=np.float64)
        K = np.asarray(self.forward_curve.forward(T_)) * np.exp(np.asarray(k, dtype=np.float64))
        vol = np.asarray(self.base.implied_vol(K, T_ + self.dt), dtype=np.float64)
        return np.asarray(vol * vol * T_, dtype=np.float64)


@dataclass
class DayMarket:
    """One trading day's market: the implied surface (the importer's, or the previous day's
    carried by the bad-day rule), its snapshot spot and what the import measured."""

    date: str
    surface: ImpliedSurface
    info: dict[str, Any]

    @property
    def spot(self) -> float:
        return float(self.surface.forward_curve.spot)

    @property
    def carried(self) -> bool:
        return bool(self.info.get("carried_from"))

    def forward(self, T: float) -> float:
        return float(self.surface.forward(T))

    def df(self, T: float) -> float:
        return float(self.surface.discount.df(T))

    def tolerances_ok(self) -> bool:
        """The agreed eSSVI tolerances of the entry's surface (spec §2.4): the *strict* sample.
        The forward cross-check against the vendor's ``stkPx`` is stored, not applied: it
        measures the vendor's convention, which changes in 2024-04 (PROGRESS, assumption A1)."""
        i = self.info
        return bool(
            i.get("built")
            and not self.carried
            and i["rms_3m_2y"] <= TOL_RMS_3M_2Y_VP
            and i["rms_6m_2y"] <= TOL_RMS_6M_2Y_VP
        )


#: Constant maturities (years) of the daily surface metrics.
METRIC_TENORS: Final[dict[str, float]] = {
    "1m": 1.0 / 12.0,
    "3m": 0.25,
    "6m": 0.5,
    "1y": 1.0,
    "2y": 2.0,
}


def surface_metrics(surface: ImpliedSurface) -> dict[str, float]:
    """Constant-maturity readings of a surface: the at-the-money-forward vol (``atm_*``), the
    at-the-money skew ``∂σ/∂ln K`` by a centred ±1 % difference (``skew_*``) and the convexity
    ``σ(+1 sd) + σ(−1 sd) − 2 σ(0)`` (``conv_*``), per tenor the surface reaches."""
    out: dict[str, float] = {}
    for name, T in METRIC_TENORS.items():
        if surface.max_maturity < T:
            continue
        atm = float(np.asarray(surface.implied_vol_k(0.0, T)).reshape(()))
        up = float(np.asarray(surface.implied_vol_k(0.01, T)).reshape(()))
        dn = float(np.asarray(surface.implied_vol_k(-0.01, T)).reshape(()))
        sd = atm * np.sqrt(T)
        wing = float(np.asarray(surface.implied_vol_k(sd, T)).reshape(())) + float(
            np.asarray(surface.implied_vol_k(-sd, T)).reshape(())
        )
        out[f"atm_{name}"] = atm
        out[f"skew_{name}"] = (up - dn) / 0.02
        out[f"conv_{name}"] = wing - 2.0 * atm
    return out


def entry_conditions(surface: ImpliedSurface, T: float, side: int) -> dict[str, float]:
    """The surface conditions of a cell at entry (spec §7): the skew on the trade's side
    ``σ(F e^{side·1sd}) − σ(F)`` and the convexity, in vol points; the term slope (1y minus 3m);
    the forward vol over the second half of the life from the at-the-money total variances;
    the decay of the at-the-money skew with maturity (log-log slope between 1m and 2y)."""

    def v(k: float, t: float) -> float:
        return float(np.asarray(surface.implied_vol_k(k, t)).reshape(()))

    atm = v(0.0, T)
    sd = atm * np.sqrt(T)
    w_full, w_half = atm * atm * T, v(0.0, 0.5 * T) ** 2 * 0.5 * T

    def skew(t: float) -> float:
        return (v(0.01, t) - v(-0.01, t)) / 0.02

    t_long = min(2.0, surface.max_maturity)
    s_short, s_long = abs(skew(1.0 / 12.0)), abs(skew(t_long))
    decay = (
        float(np.log(s_long / s_short) / np.log(t_long * 12.0))
        if s_short > 0 and s_long > 0
        else float("nan")
    )
    return {
        "skew_side": 100.0 * (v(side * sd, T) - atm),
        "convexity": 100.0 * (v(sd, T) + v(-sd, T) - 2.0 * atm),
        "term_slope": 100.0 * (v(0.0, 1.0) - v(0.0, 0.25)),
        "fwd_vol_2nd_half": float(np.sqrt(max(w_full - w_half, 0.0) / (0.5 * T))),
        "skew_decay": decay,
        "atm_skew": skew(T),
    }


def import_day(date: str, cache: Path = DAY_CACHE, *, sabrw: bool = False) -> dict[str, Any]:
    """Run the ORATS importer for ``date`` and cache its snapshot (``<cache>/<date>.yaml``) and
    what it measured (``<date>.json``); returns the measurements.  Never raises on a day that
    cannot be built: ``built`` is false and ``error`` says why.  ``sabrw`` adds the SABRW fits
    (needed on parameter re-mark days only)."""
    from volsto.market import import_hdn as ih
    from volsto.market import import_orats as io

    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    info: dict[str, Any] = {"date": date, "built": False, "sabrw": False}
    try:
        f = ih.HdnFilters()
        # the importer is the one writer of a snapshot and of its SABRW section
        doc, fit, points, chain = io.import_day(date, "SPX", filters=f, essvi=True, sabrw=sabrw)
        t_max = float(points.table["T"].max())
        # the forwards again, for the cross-check against the vendor's only (the importer
        # does not return them; same function of the same chain)
        fwds = ih.implied_forwards(chain, max_years=f.max_years, band=f.near_atm_band)
        xc = io.forward_crosscheck(chain, fwds)
        year = xc[xc["T"] <= 1.0]["bp"].abs()
        info.update(
            built=True,
            # the section is always written; without fits its list is empty
            sabrw=bool(doc.get("sabrw", {}).get("fits")),
            spot=float(doc["market"]["spot"]),
            close=float(chain.attrs["spot"]),
            t_max=t_max,
            n_expiries=int(points.table["expiry"].nunique()),
            n_points=len(points.table),
            fwd_median_bp=float(year.median()) if len(year) else float("inf"),
            fwd_max_bp=float(year.max()) if len(year) else float("inf"),
            fwd_slices=len(year),
            rms_3m_2y=float(fit.rms_error(2.0, 0.2, 0.25)),
            rms_6m_2y=float(fit.rms_error(2.0, 0.2, 0.5)),
            calendar_fallback=doc["provenance"]["fit"].get("calendar_fallback"),
            dropped_expiries=list(chain.attrs.get("dropped_expiries", [])),
            settlement_uncertain=bool(chain.attrs.get("settlement_uncertain", False)),
            settled_by=dict(chain.attrs.get("settled_by", {})),
            **surface_metrics(fit.surface),
        )
        (cache / f"{date}.yaml").write_text(yaml.safe_dump(doc, sort_keys=False))
    except Exception as exc:
        info["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    info["seconds"] = round(time.perf_counter() - t0, 2)
    (cache / f"{date}.json").write_text(json.dumps(info, indent=1) + "\n")
    return info


def day_info(date: str, cache: Path = DAY_CACHE) -> dict[str, Any] | None:
    p = cache / f"{date}.json"
    if not p.exists():
        return None
    out: dict[str, Any] = json.loads(p.read_text())
    return out


def load_day(
    date: str, calendar: Sequence[str], ohlc: pd.DataFrame, cache: Path = DAY_CACHE
) -> DayMarket:
    """The market of ``date`` from the day cache.  A day that was not built (or whose surface
    does not reach the study's horizon) takes the last built day's implied vols by absolute
    strike and expiry date, with spot and forwards scaled by the ratio of the official closes
    (spec §2.4; contradiction 7).  Raises ``LookupError`` when no earlier built day exists."""
    from volsto.market.loaders import load_ssvi_surface

    info = day_info(date, cache)
    if info is None:
        raise LookupError(f"{date} is not in the day cache {cache}")
    if info.get("built") and info.get("t_max", 0.0) >= LV_CONFIG.t_max + 0.05:
        return DayMarket(date, load_ssvi_surface(cache / f"{date}.yaml"), info)
    i = list(calendar).index(date)
    for back in range(i - 1, max(i - 30, -1), -1):
        prev = calendar[back]
        pinfo = day_info(prev, cache)
        if pinfo and pinfo.get("built") and pinfo.get("t_max", 0.0) >= LV_CONFIG.t_max + 0.05:
            base = load_ssvi_surface(cache / f"{prev}.yaml")
            closes = ohlc["close"].to_dict()
            ratio = float(closes[date]) / float(closes[prev])
            # spot_new = base spot × close ratio; MovedSurface scales the base *forward* at dt
            dt = year_fraction(prev, date)
            scale = ratio * float(base.forward_curve.spot) / float(base.forward_curve.forward(dt))
            carried = dict(info, carried_from=prev, carried_days=i - back)
            return DayMarket(date, MovedSurface(base, scale, dt), carried)
    raise LookupError(f"no built day within 30 trading days before {date}")


# --------------------------------------------------------------------------------------------
# the day engine: one path set, every live trade
# --------------------------------------------------------------------------------------------


@dataclass
class DayPaths:
    """The functionals of a day's path set on the future trading days ``times`` (years):
    the simulated closes and the running extremes (log), per path."""

    times: FloatArray  # (D,)
    close: FloatArray  # (N, D) spot at each future trading day
    run_max_close: FloatArray  # (N, D) running max of the closes (log)
    run_min_close: FloatArray
    run_max_cont: FloatArray  # (N, D) running max of the bridge-sampled step maxima (log)
    run_min_cont: FloatArray
    antithetic: bool


#: Simulation step of the day engine: ``None`` is the library's schedule (a quarter of a day
#: below three months, a day to two years).  One step per calendar day was tried in the pilot
#: and rejected: it lowers the near-barrier continuous knock-outs by about 5 % of their value
#: (1m, half a standard deviation: 3.60 ± 0.05 bp against 3.81 with the library's step and 3.79
#: by PDE), PROGRESS log, check 5.
DAY_STEP: Final[float | None] = None
#: Paths simulated and reduced at a time (bounds the memory of a day's set).
PATH_CHUNK: Final[int] = 20_000


def _reduce(
    x: FloatArray,
    var: FloatArray,
    grid_times: FloatArray,
    times: FloatArray,
    rng: np.random.Generator,
    antithetic: bool,
) -> DayPaths:
    """Path functionals of recorded log-spots ``x`` (paths × grid times) and step-start
    variances ``var`` (module docstring of :func:`simulate_day`)."""
    cols = np.searchsorted(grid_times, times)
    dts = np.diff(grid_times)
    n, d = x.shape[0], times.size
    day_max = np.empty((n, d))
    day_min = np.empty((n, d))
    a = 0
    for j, b in enumerate(cols):
        x0, x1 = x[:, a:b], x[:, a + 1 : b + 1]
        vdt = var[:, a:b] * dts[None, a:b]
        dx2 = (x1 - x0) ** 2
        u = rng.random((2, n // 2 if antithetic else n, b - a))
        if antithetic:  # one uniform per antithetic pair: the pair stays a pair
            u = np.repeat(u, 2, axis=1)
        up = 0.5 * (x0 + x1 + np.sqrt(dx2 - 2.0 * vdt * np.log(u[0])))
        dn = 0.5 * (x0 + x1 - np.sqrt(dx2 - 2.0 * vdt * np.log(u[1])))
        day_max[:, j] = up.max(axis=1)
        day_min[:, j] = dn.min(axis=1)
        a = int(b)
    logc = x[:, cols]
    x_start = x[:, :1]
    return DayPaths(
        times=np.asarray(times, dtype=np.float64),
        close=np.exp(logc),
        run_max_close=np.maximum.accumulate(logc, axis=1),
        run_min_close=np.minimum.accumulate(logc, axis=1),
        run_max_cont=np.maximum(np.maximum.accumulate(day_max, axis=1), x_start),
        run_min_cont=np.minimum(np.minimum.accumulate(day_min, axis=1), x_start),
        antithetic=antithetic,
    )


def simulate_day_chunks(
    model: Model,
    times: FloatArray,
    *,
    n_paths: int,
    seed: int,
    antithetic: bool = True,
    chunk: int = PATH_CHUNK,
    dt_max: float | None = DAY_STEP,
) -> Iterator[DayPaths]:
    """Simulate ``model`` on the grid of ``times`` (every step recorded), ``chunk`` paths at a
    time, each chunk reduced to the closes and the running extremes.  The continuous extremes
    use the Brownian bridge between consecutive steps with the step-start variance: the maximum
    of a step is sampled as ``½ (x₀ + x₁ + √((x₁ − x₀)² − 2 v Δt ln U))`` (and the mirror for
    the minimum), ``U`` from a generator keyed on ``seed`` and the chunk only — common across
    bumped models, like the normals."""
    kw: dict[str, Any] = {} if dt_max is None else {"dt_max": float(dt_max)}
    sim = SimConfig(
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
        record_all_steps=True,
        chunk_size=chunk,
        **kw,
    )
    grid = TimeGrid.build(
        times, sim.dt_max, calibration_grid=model.required_times(), record_all_steps=True
    )
    draws = MonteCarlo(sim).draws_for(grid, model)
    n_steps = grid.times.size - 1
    for c, p0 in enumerate(range(0, n_paths, chunk)):
        part = model.simulate_chunk(grid, draws, p0, min(p0 + chunk, n_paths), sim.scheme)
        x = part.log_spot_at(np.arange(n_steps + 1))
        var = part.variance_at(np.arange(n_steps))
        rng = np.random.default_rng([int(seed), 7919, c])
        yield _reduce(x, var, grid.times, np.asarray(times, dtype=np.float64), rng, antithetic)


def simulate_day(
    model: Model,
    times: FloatArray,
    *,
    n_paths: int,
    seed: int,
    antithetic: bool = True,
    dt_max: float | None = DAY_STEP,
) -> DayPaths:
    """:func:`simulate_day_chunks` as one set (tests and small runs)."""
    (paths,) = simulate_day_chunks(
        model,
        times,
        n_paths=n_paths,
        seed=seed,
        antithetic=antithetic,
        chunk=n_paths,
        dt_max=dt_max,
    )
    return paths


def knock_out_sums(
    paths: DayPaths, j: int, side: int, K: FloatArray, B: FloatArray
) -> dict[str, FloatArray]:
    """Sufficient statistics, over the (pair-averaged) paths of one chunk, of the payoffs of
    the knock-outs expiring on the ``j``-th future trading day — daily ``1``, continuous ``2``,
    European ``e`` — for strikes ``K`` and barriers ``B``: sums, sums of squares, cross sums
    with the European one, knock counts.  Chunks add (:func:`knock_out_from_sums`)."""
    s = paths.close[:, j][:, None]
    lnb = np.log(B)[None, :]
    if side > 0:
        alive1 = paths.run_max_close[:, j][:, None] <= lnb
        alive2 = paths.run_max_cont[:, j][:, None] < lnb
        alive_e = s <= B[None, :]
    else:
        alive1 = paths.run_min_close[:, j][:, None] >= lnb
        alive2 = paths.run_min_cont[:, j][:, None] > lnb
        alive_e = s >= B[None, :]
    pay = np.maximum(side * (s - K[None, :]), 0.0)
    x1, x2, xe = pay * alive1, pay * alive2, pay * alive_e
    if paths.antithetic:
        x1, x2, xe = (0.5 * (x[0::2] + x[1::2]) for x in (x1, x2, xe))
    return {
        "n": np.full(K.size, float(x1.shape[0])),
        "paths": np.full(K.size, float(alive1.shape[0])),
        "s1": x1.sum(axis=0),
        "s2": x2.sum(axis=0),
        "se": xe.sum(axis=0),
        "q1": (x1 * x1).sum(axis=0),
        "q2": (x2 * x2).sum(axis=0),
        "qe": (xe * xe).sum(axis=0),
        "c1": (x1 * xe).sum(axis=0),
        "c2": (x2 * xe).sum(axis=0),
        "k1": (~alive1).sum(axis=0).astype(np.float64),
        "k2": (~alive2).sum(axis=0).astype(np.float64),
    }


def knock_out_from_sums(
    sums: Mapping[str, FloatArray],
    european: FloatArray | None = None,
    beta: Mapping[str, FloatArray] | None = None,
) -> dict[str, FloatArray]:
    """Values and standard errors from :func:`knock_out_sums` (see :func:`knock_out_values`)."""
    n = sums["n"]
    me = sums["se"] / n
    var_e = np.maximum(sums["qe"] / n - me * me, 0.0)
    out: dict[str, FloatArray] = {
        "p1": sums["k1"] / sums["paths"],
        "p2": sums["k2"] / sums["paths"],
        "e": me,
    }
    for tag in ("1", "2"):
        name = f"a{tag}"
        mean = sums[f"s{tag}"] / n
        var = np.maximum(sums[f"q{tag}"] / n - mean * mean, 0.0)
        if european is None:
            out[name] = mean
            out[f"{name}_se"] = np.sqrt(var / (n - 1.0))
            continue
        cov = sums[f"c{tag}"] / n - mean * me
        if beta is None:
            b = np.where(var_e > 0, cov / np.where(var_e > 0, var_e, 1.0), 0.0)
        else:
            b = np.asarray(beta[name], dtype=np.float64)
        out[name] = mean - b * (me - european)
        out[f"{name}_se"] = np.sqrt(
            np.maximum(var - 2.0 * b * cov + b * b * var_e, 0.0) / (n - 1.0)
        )
        out[f"{name}_beta"] = b
    return out


def knock_out_values(
    paths: DayPaths,
    j: int,
    side: int,
    K: FloatArray,
    B: FloatArray,
    european: FloatArray | None = None,
    beta: Mapping[str, FloatArray] | None = None,
) -> dict[str, FloatArray]:
    """Undiscounted values (and standard errors) of the knock-outs expiring on the ``j``-th
    future trading day, strikes ``K`` and barriers ``B`` (arrays of one length): ``a1`` daily
    (knocked on a close strictly beyond ``B``), ``a2`` continuous (knocked at or beyond), with
    ``p1`` / ``p2`` the knock probabilities.

    With ``european`` — the undiscounted value of the European knock-out ``(side (S_T − K))⁺
    1{S_T not beyond B}`` of each trade, known from the surface the model is calibrated to —
    the same payoff on the paths is a control variate: ``a − β (e_paths − european)``, ``β`` the
    regression coefficient on these paths unless given (the bumped sets reuse the base set's,
    so a delta is a difference of like with like); the standard errors are the controlled
    ones."""
    return knock_out_from_sums(knock_out_sums(paths, j, side, K, B), european, beta)


def local_vol_model(surface: ImpliedSurface) -> LocalVol:
    """Dupire local vol of ``surface`` on the study's grid (:data:`LV_CONFIG`)."""
    return LocalVol(LocalVolSurface.from_implied(surface, LV_CONFIG), surface.forward_curve)


def future_times(
    date: str, calendar: Sequence[str], horizon_days: int = 372
) -> tuple[list[str], FloatArray]:
    """The trading days after ``date`` within ``horizon_days`` calendar days and their ACT/365
    times."""
    i = list(calendar).index(date)
    end = (_dt.date.fromisoformat(date) + _dt.timedelta(days=horizon_days)).isoformat()
    days = [d for d in calendar[i + 1 :] if d <= end]
    return days, np.array([year_fraction(date, d) for d in days])


def seed_of(date: str) -> int:
    """The Monte Carlo seed of a day: its date as an integer (common across bumps and runs)."""
    return int(date.replace("-", ""))


def cells_of(entry: str, market: DayMarket, calendar: Sequence[str]) -> pd.DataFrame:
    """The cells of an entry date: one row per (maturity, side, barrier) with the strike, the
    barrier, the expiry and its time, the at-the-money-forward vol."""
    S0 = market.spot
    rows = []
    for months in MATURITY_MONTHS:
        expiry = expiry_date(entry, months, calendar)
        T = year_fraction(entry, expiry)
        atm = float(np.asarray(market.surface.atm_vol(T)).reshape(()))
        for side in (1, -1):
            for name, B in barrier_levels(side, S0, atm, T).items():
                rows.append(
                    {
                        "entry": entry,
                        "months": months,
                        "side": side,
                        "barrier": name,
                        "K": S0,
                        "B": B,
                        "expiry": expiry,
                        "T": T,
                        "atm": atm,
                    }
                )
    return pd.DataFrame(rows)


def matched_legs(
    side: int, K: float, B: float, premium: float, surface: ImpliedSurface, T: float
) -> dict[str, tuple[Leg, ...]]:
    """D10 and D11: the ratio ``K / (K+B)/2`` and the fly ``K / (K+B)/2 / wing`` whose surface
    price is ``premium`` (the library's premium-matched structures).  A structure the premium
    cannot reach is absent."""
    from volsto.products.structures import fly_wing_for_premium, ratio_for_premium

    mid = 0.5 * (K + B)
    out: dict[str, tuple[Leg, ...]] = {}
    lo, hi = (K, mid) if side > 0 else (mid, K)
    try:
        r = ratio_for_premium(lo, hi, T, surface, premium, cp=side)
        if np.isfinite(r) and r > 0:
            out["D10"] = (Leg("opt", K, 1.0), Leg("opt", mid, -float(r)))
    except ValueError:
        pass
    try:
        fly = fly_wing_for_premium(lo, hi, T, surface, premium, surface.discount, cp=side)
        out["D11"] = tuple(
            Leg("opt", float(k), float(w))
            for k, w in zip(fly.strikes, fly.leg_weights, strict=True)
        )
    except ValueError:
        pass
    return out


def legs_to_json(legs: Mapping[str, Sequence[Leg]]) -> str:
    return json.dumps({k: [[leg.kind, leg.strike, leg.qty] for leg in v] for k, v in legs.items()})


def legs_from_json(text: str) -> dict[str, tuple[Leg, ...]]:
    return {
        k: tuple(Leg(str(a), float(b), float(c)) for a, b, c in v)
        for k, v in json.loads(text).items()
    }


# --------------------------------------------------------------------------------------------
# bulk marks of the vanilla structures
# --------------------------------------------------------------------------------------------


class ParallelVolSurface(ImpliedSurface):
    """``base`` with every implied vol moved by ``shift`` (the +1 vol point vega of spec §6.5)."""

    def __init__(self, base: ImpliedSurface, shift: float) -> None:
        super().__init__(base.forward_curve, base.discount, base.max_maturity)
        self.base = base
        self.shift = float(shift)

    def total_variance(self, k: Any, T: Any) -> FloatArray:
        T_ = np.asarray(T, dtype=np.float64)
        vol = np.asarray(self.base.implied_vol_k(k, T_), dtype=np.float64) + self.shift
        return np.asarray(vol * vol * T_, dtype=np.float64)


STRUCTURES: Final[tuple[str, ...]] = (
    "VAN",
    "FWD",
    "B3",
    "B4",
    "B5_2",
    "B5_3",
    "B5_4",
    "B5_6",
    "B6",
    "C7",
    "C8",
    "C9_7_1",
    "C9_7_2",
    "C9_8_1",
    "C9_8_2",
    "D10",
    "D11",
    "E12",
    "E13",
)


def expand_legs(
    cells: Sequence[tuple[int, Mapping[str, Sequence[Leg]]]],
) -> dict[str, NDArray[Any]]:
    """Flatten the cells' structures — ``(side, {structure: legs})`` per cell — into arrays of
    elementary pieces: an option of strike ``x`` and side ``cp`` with weight ``q`` (digitals as
    the two options of their centred difference), or a forward.  ``slot`` is ``cell index ×
    len(STRUCTURES) + structure index``."""
    slot: list[int] = []
    cp: list[int] = []
    x: list[float] = []
    q: list[float] = []
    fwd: list[bool] = []
    n_s = len(STRUCTURES)
    index = {name: i for i, name in enumerate(STRUCTURES)}
    for c, (side, legs_by) in enumerate(cells):
        for name, legs in legs_by.items():
            s = c * n_s + index[name]
            for leg in legs:
                if leg.kind == "dig":
                    h = DIGITAL_REL_WIDTH * leg.strike
                    for strike, sign in ((leg.strike - h, 1.0), (leg.strike + h, -1.0)):
                        slot.append(s)
                        cp.append(side)
                        x.append(strike)
                        q.append(leg.qty * side * sign / (2.0 * h))
                        fwd.append(False)
                else:
                    slot.append(s)
                    cp.append(side)
                    x.append(leg.strike)
                    q.append(leg.qty)
                    fwd.append(leg.kind == "fwd")
    return {
        "slot": np.asarray(slot, dtype=np.int64),
        "cp": np.asarray(cp, dtype=np.float64),
        "x": np.asarray(x, dtype=np.float64),
        "q": np.asarray(q, dtype=np.float64),
        "fwd": np.asarray(fwd, dtype=bool),
        "n_cells": np.asarray(len(cells)),
    }


def bulk_values(
    pieces: Mapping[str, NDArray[Any]],
    piece_T: FloatArray,
    surface: ImpliedSurface,
    ratios: Sequence[float] = (1.0,),
    vol_shift: float = 0.0,
) -> FloatArray:
    """Discounted values of every (cell, structure) for each forward ratio: array ``(len(ratios),
    n_cells, len(STRUCTURES))``, NaN where a cell lacks the structure.  ``piece_T`` is the
    maturity of each piece (its cell's remaining life).  The implied vol of each absolute
    strike is the surface's (plus ``vol_shift``), whatever the ratio: the sticky-strike bump."""
    n_cells = int(pieces["n_cells"])
    n_s = len(STRUCTURES)
    x, q, cp, slot, is_fwd = (pieces[k] for k in ("x", "q", "cp", "slot", "fwd"))
    F = np.empty_like(x)
    df = np.empty_like(x)
    vol = np.empty_like(x)
    for T in np.unique(piece_T):
        m = piece_T == T
        F[m] = float(surface.forward(T))
        df[m] = float(surface.discount.df(T))
        vol[m] = np.asarray(surface.implied_vol(x[m], T), dtype=np.float64) + vol_shift
    out = np.full((len(ratios), n_cells * n_s), np.nan)
    present = np.zeros(n_cells * n_s, dtype=bool)
    present[slot] = True
    for r, ratio in enumerate(ratios):
        Fr = F * ratio
        d1 = np.empty_like(x)
        s = vol * np.sqrt(piece_T)
        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = (np.log(Fr / x) + 0.5 * s * s) / s
        opt = cp * (Fr * ndtr(cp * d1) - x * ndtr(cp * (d1 - s)))
        opt = np.where(s > 0, opt, np.maximum(cp * (Fr - x), 0.0))
        value = q * df * np.where(is_fwd, Fr - x, opt)
        total = np.bincount(slot, weights=value, minlength=n_cells * n_s)
        out[r] = np.where(present, total, np.nan)
    return out.reshape(len(ratios), n_cells, n_s)


def bs_legs_value(legs: Sequence[Leg], side: int, spot: float, vol: float, T: float) -> float:
    """Black–Scholes value of ``legs`` with zero rates and carry, the digitals exact
    (``N(side·d2)``): the closed form pilot check 1 is stated in."""
    total = 0.0
    s = vol * np.sqrt(T)
    for leg in legs:
        d1 = (np.log(spot / leg.strike) + 0.5 * s * s) / s
        if leg.kind == "opt":
            total += leg.qty * float(
                side * (spot * ndtr(side * d1) - leg.strike * ndtr(side * (d1 - s)))
            )
        elif leg.kind == "dig":
            total += leg.qty * float(ndtr(side * (d1 - s)))
        elif leg.kind == "fwd":
            total += leg.qty * (spot - leg.strike)
        else:
            raise ValueError(f"unknown leg kind {leg.kind!r}")
    return total


# --------------------------------------------------------------------------------------------
# one position's accounting over its life
# --------------------------------------------------------------------------------------------


def position_paths(
    value: FloatArray,
    delta_f: FloatArray,
    forward: FloatArray,
    df: FloatArray,
    *,
    premium: float,
    df0: float,
    terminal: float,
    closing_forward: float,
    n_hedge: int,
) -> tuple[FloatArray, FloatArray]:
    """``(unhedged, hedged)`` running P&L of a position (arguments of
    :func:`position_outcome`): one point per snapshot held (``n_hedge`` of them, the entry
    first: the mark over the discount factor minus the premium carried to expiry, plus — for
    the hedged one — the hedge P&L accrued so far) and a last point at the end (``terminal``).
    The daily P&L of a book is the sum over its trades of the differences of these paths."""
    cost = premium / df0
    pnl_u = terminal - cost
    if n_hedge <= 0:
        return np.array([pnl_u]), np.array([pnl_u])
    h = hedge_pnl(delta_f[:n_hedge], forward[:n_hedge], closing_forward)
    cum = np.concatenate(([0.0], np.cumsum(h)))  # hedge P&L accrued before snapshot i
    mtm = value[:n_hedge] / df[:n_hedge] - cost
    return np.append(mtm, pnl_u), np.append(mtm + cum[:n_hedge], pnl_u + float(cum[-1]))


def position_outcome(
    value: FloatArray,
    delta_f: FloatArray,
    forward: FloatArray,
    df: FloatArray,
    *,
    premium: float,
    df0: float,
    terminal: float,
    closing_forward: float,
    n_hedge: int,
    n_life: int,
) -> dict[str, float]:
    """Spec §5.3 for one position.  ``value`` / ``delta_f`` / ``forward`` / ``df`` are its
    mark, forward delta, forward to its expiry and discount factor to its expiry on the
    snapshots of its life (entry first).  It is held over ``n_hedge`` intervals — to its expiry
    (``n_hedge`` = the number of snapshots) or to the knock / sale — the last one closed at
    ``closing_forward``, and is then worth ``terminal``, valued at expiry (the payoff, zero for
    a knocked trade, the sale value over the discount factor for a structure sold at the
    touch).  ``n_life`` is the number of intervals of a full life (for the quartiles).

    Returns ``pnl_u`` (``terminal − premium / df0``), ``pnl_h`` (plus the hedge), the hedged
    running P&L at 25 / 50 / 75 % of the life and its best and worst, and ``hedge`` (the sum of
    the hedge P&L).  Everything in the units of ``value``."""
    pnl_u = terminal - premium / df0
    _, path = position_paths(
        value,
        delta_f,
        forward,
        df,
        premium=premium,
        df0=df0,
        terminal=terminal,
        closing_forward=closing_forward,
        n_hedge=n_hedge,
    )
    final = float(path[-1])
    if n_hedge <= 0:
        flat = {f"run_{q}": pnl_u for q in (25, 50, 75)}
        return {"pnl_u": pnl_u, "pnl_h": pnl_u, "hedge": 0.0, "best": pnl_u, "worst": pnl_u, **flat}

    def at(q: float) -> float:
        i = round(q * n_life)
        return float(path[i]) if i < path.size else final

    return {
        "pnl_u": float(pnl_u),
        "pnl_h": final,
        "hedge": final - float(pnl_u),
        "best": float(path.max()),
        "worst": float(path.min()),
        "run_25": at(0.25),
        "run_50": at(0.50),
        "run_75": at(0.75),
    }


# --------------------------------------------------------------------------------------------
# forward-start smile and vanilla repricing on a path set
# --------------------------------------------------------------------------------------------

FORWARD_MONEYNESS: Final[tuple[float, ...]] = (0.95, 1.0, 1.05)


def forward_start_sums(paths: DayPaths, j_half: int, j_end: int) -> FloatArray:
    """Sums over a chunk's paths for the forward-start smile from the ``j_half``-th to the
    ``j_end``-th future trading day: ``[n, Σ R, Σ (k − R)⁺ or (R − k)⁺ for each k of
    FORWARD_MONEYNESS]`` with ``R = S_end / S_half`` (puts below 1, calls at and above)."""
    r = paths.close[:, j_end] / paths.close[:, j_half]
    out = [float(r.size), float(r.sum())]
    for k in FORWARD_MONEYNESS:
        out.append(float(np.maximum(k - r, 0.0).sum() if k < 1.0 else np.maximum(r - k, 0.0).sum()))
    return np.asarray(out)


def forward_start_vols(sums: FloatArray, tau: float) -> dict[str, float]:
    """Forward implied vols at :data:`FORWARD_MONEYNESS` (moneyness against the forward ratio
    ``E[R]``) from :func:`forward_start_sums`, and the forward skew ``[σ(0.95) − σ(1.05)]/0.10``
    (spec §7.1.4); ``tau`` is the length of the forward period in years."""
    from scipy.optimize import brentq

    n, mean = sums[0], sums[1] / sums[0]
    out: dict[str, float] = {}
    for k, total in zip(FORWARD_MONEYNESS, sums[2:], strict=True):
        cp = -1 if k < 1.0 else 1
        price = total / n / mean  # an option on R / E[R] struck at k / E[R]
        strike = k / mean

        def gap(v: float, strike: float = strike, cp: int = cp, price: float = price) -> float:
            one = np.asarray(1.0)
            value = black(one, np.asarray(strike), np.asarray(tau), np.asarray(v), cp, one)
            return float(value) - price

        try:
            out[f"fvol_{k:g}"] = float(brentq(gap, 1e-4, 5.0, xtol=1e-10))
        except ValueError:
            out[f"fvol_{k:g}"] = float("nan")
    out["fskew"] = (out["fvol_0.95"] - out["fvol_1.05"]) / 0.10
    return out


def vanilla_sums(paths: DayPaths, j: int, side: int, strikes: FloatArray) -> FloatArray:
    """``[n, Σ payoff, Σ payoff²]`` per strike (columns) of the vanilla ``(side (S − X))⁺`` on
    the ``j``-th future trading day, over the pair-averaged paths of a chunk."""
    pay = np.maximum(side * (paths.close[:, j][:, None] - strikes[None, :]), 0.0)
    if paths.antithetic:
        pay = 0.5 * (pay[0::2] + pay[1::2])
    return np.vstack(
        [np.full(strikes.size, float(pay.shape[0])), pay.sum(axis=0), (pay * pay).sum(axis=0)]
    )
