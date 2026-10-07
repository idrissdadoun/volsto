"""Closed forms and path statistics of the barrier study's addendum 1
(``outputs/interview/BARRIER_STUDY_ADDENDUM.md`` §4 to §7): the forward-skew premium of the
knock-out over its static hedge, the carry control under a flat smile, the weights over touch
dates, the annuity that prices one unit of normalised skew at the barrier, the bucket
statistics of a path set by third of the life, the directional skew-stickiness ratio and the
frozen forecast of realised vol.

Units: fractions of the entry spot unless a function says otherwise.  ``side`` is +1 on the
call side (up-and-out call, ``B > K``) and −1 on the put side.  This module adds quantities;
it changes no price of :mod:`volsto.studies.barrier_history`.

Checked by ``tests/test_barrier_theory.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import ndtr

from volsto.market.barrier_bs import bs_barrier_price
from volsto.market.curves import ForwardCurve
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import ImpliedSurface
from volsto.models.localvol import LocalVol
from volsto.studies import barrier_history as bh

FloatArray = NDArray[np.float64]

#: A ratio whose denominator is below this (0.2 bp of spot) is NaN (addendum §4.6).
RATIO_FLOOR: Final[float] = 0.2e-4
#: Shortest tenor of a normalised-skew reading (addendum §4.4): 7 calendar days.
MIN_TENOR: Final[float] = 7.0 / 365.0
#: Vega-like constant of the annuity (addendum §4.3).
ANNUITY_CONSTANT: Final[float] = 0.8


def carry_rates(df0: float, f0: float, s0: float, T: float) -> tuple[float, float]:
    """``(r, q)`` of maturity ``T`` from the entry day's discount factor and forward:
    ``r = −ln(DF0)/T``, ``q = r − ln(F0/S0)/T`` (addendum §4.1b)."""
    r = -np.log(df0) / T
    return float(r), float(r - np.log(f0 / s0) / T)


def ko_rr(
    side: int, S: float, K: float, B: float, T: float, sig: float, r: float, q: float
) -> float:
    """Reiner–Rubinstein value of the continuous knock-out of the study's side: the up-and-out
    call (``K < B``) or the down-and-out put (``K > B``), cost of carry ``r − q``."""
    direction = "up" if side > 0 else "down"
    return float(bs_barrier_price(S, K, B, T, sig, r, q, side, direction, "out"))


def legs_bs(
    legs: Sequence[bh.Leg], side: int, S: float, T: float, sig: float, r: float, q: float
) -> float:
    """Black–Scholes value of ``legs`` at flat vol ``sig`` with rates and dividends, the
    digitals exact (``e^{−rT} N(side·d2)``)."""
    F = S * np.exp((r - q) * T)
    df = np.exp(-r * T)
    s = sig * np.sqrt(T)
    total = 0.0
    for leg in legs:
        if leg.kind == "fwd":
            total += leg.qty * df * (F - leg.strike)
            continue
        if s <= 0:
            value = (
                max(side * (F - leg.strike), 0.0)
                if leg.kind == "opt"
                else float(side * (F - leg.strike) > 0)
            )
            total += leg.qty * df * value
            continue
        d1 = (np.log(F / leg.strike) + 0.5 * s * s) / s
        if leg.kind == "opt":
            total += (
                leg.qty
                * df
                * float(side * (F * ndtr(side * d1) - leg.strike * ndtr(side * (d1 - s))))
            )
        elif leg.kind == "dig":
            total += leg.qty * df * float(ndtr(side * (d1 - s)))
        else:
            raise ValueError(f"unknown leg kind {leg.kind!r}")
    return float(total)


def c8_bs(
    side: int, S: float, K: float, B: float, T: float, sig: float, r: float, q: float
) -> float:
    """Black–Scholes value at spot ``S`` of the put-call-symmetry hedge C8 of strike ``K`` and
    barrier ``B`` (its legs are those of :func:`volsto.studies.barrier_history.structure_legs`)."""
    return legs_bs(bh.structure_legs(side, K, B, K)["C8"], side, S, T, sig, r, q)


def pi_flat(
    side: int, S: float, K: float, B: float, T: float, sig: float, r: float, q: float
) -> float:
    """The carry control ``sgn·(KO_RR − C8_BS)`` at flat vol ``sig`` (addendum §4.1b): what the
    forward-skew premium would be with no smile.  Zero when ``r = q``."""
    return side * (ko_rr(side, S, K, B, T, sig, r, q) - c8_bs(side, S, K, B, T, sig, r, q))


def touch_weights(S0: float, K: float, B: float, T: float, sig_b: float) -> dict[str, Any]:
    """Addendum §4.2: the life split in thirds — the driftless probability that the first
    touch falls in each (``pi``), the time left at the middle of each (``theta``), the damping
    ``kap_j = 1 − exp(−k_j²/2)`` with ``k_j = |ln(B/K)| / (σ_B √θ_j)``, their average ``kapbar``
    and the weights ``om_j = π_j κ_j / κ̄``; ``c`` is the distance in standard deviations."""
    c = abs(np.log(B / S0)) / (sig_b * np.sqrt(T))

    def Q(t: float) -> float:
        return float(2.0 * ndtr(-c * np.sqrt(T / t)))

    qT = Q(T)
    theta = T * np.array([5.0 / 6.0, 0.5, 1.0 / 6.0])
    if qT <= 0.0:
        nan = np.full(3, np.nan)
        return {"c": float(c), "pi": nan, "theta": theta, "kap": nan, "kapbar": np.nan, "om": nan}
    p1 = Q(T / 3.0) / qT
    p2 = (Q(2.0 * T / 3.0) - Q(T / 3.0)) / qT
    pi = np.array([p1, p2, 1.0 - p1 - p2])
    k = abs(np.log(B / K)) / (sig_b * np.sqrt(theta))
    kap = 1.0 - np.exp(-0.5 * k * k)
    kapbar = float(np.sum(pi * kap))
    om = pi * kap / kapbar if kapbar > 0 else np.full(3, np.nan)
    return {"c": float(c), "pi": pi, "theta": theta, "kap": kap, "kapbar": kapbar, "om": om}


def kap(K: float, B: float, sig_b: float, time_left: FloatArray | float) -> FloatArray:
    """The damping ``1 − exp(−k²/2)``, ``k = |ln(B/K)| / (σ_B √θ)``, at time left ``θ``."""
    theta = np.maximum(np.asarray(time_left, dtype=np.float64), 1e-12)
    k = abs(np.log(B / K)) / (sig_b * np.sqrt(theta))
    return np.asarray(1.0 - np.exp(-0.5 * k * k), dtype=np.float64)


def guarded_ratio(num: Any, den: Any, floor: float = RATIO_FLOOR) -> Any:
    """``num / den``, NaN where ``|den| < floor`` (addendum §4.6); broadcasts."""
    n = np.asarray(num, dtype=np.float64)
    d = np.asarray(den, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(d) >= floor, n / np.where(d == 0, np.nan, d), np.nan)
    return float(out) if out.ndim == 0 else out


def atm_skew(
    surface: ImpliedSurface, theta: float, half_width: float = 0.01
) -> tuple[float, float]:
    """``(at-the-money-forward vol, slope of implied vol in log-strike)`` at tenor ``theta`` by a
    centred difference of ``half_width`` in log-moneyness."""

    def v(k: float) -> float:
        return float(np.asarray(surface.implied_vol_k(k, theta)).reshape(()))

    return v(0.0), (v(half_width) - v(-half_width)) / (2.0 * half_width)


def normalised_skew(surface: ImpliedSurface, theta: float) -> tuple[float, bool]:
    """``n(θ) = −skew(θ)·√θ`` (addendum §4.4) and whether ``θ`` was clamped at
    :data:`MIN_TENOR`.  Positive for an equity index."""
    t = max(float(theta), MIN_TENOR)
    _, skew = atm_skew(surface, t)
    return float(-skew * np.sqrt(t)), bool(t > theta)


# --------------------------------------------------------------------------------------------
# bucket statistics of a path set (addendum §5.2)
# --------------------------------------------------------------------------------------------

BUCKET_STRUCTURES: Final[tuple[str, ...]] = ("B6", "B5_2", "C8")
RULES: Final[tuple[str, ...]] = ("c", "d")  # continuous, daily


class Buckets:
    """Accumulates, over the chunks of a path set, the first-touch statistics of each cell by
    third of its life — under the continuous rule (``c``: the bridge-sampled running extreme
    at or beyond the barrier) and the daily rule (``d``: a close strictly beyond): the
    probability that the first touch falls in the third, the probability of finishing beyond
    the barrier given that, the mean time left at the touch (at the day's resolution), and
    the mean undiscounted payoff of the tight limit, the fly (n = 2) and C8 given that.

    Also the unconditional means the identities of addendum §11 need: the tight-limit payoff,
    the daily knock-out payoff, the share of paths finishing beyond the barrier."""

    def __init__(self, cells: pd.DataFrame, days: list[str], times: FloatArray) -> None:
        index = {d: j for j, d in enumerate(days)}
        self.n_cells = len(cells)
        self.groups: list[dict[str, Any]] = []
        for (expiry, side), g in cells.groupby(["expiry", "side"]):
            j = index[str(expiry)]
            legs = [bh.legs_from_json(text) for text in g["legs"]]
            self.groups.append(
                {
                    "j": j,
                    "side": int(str(side)),
                    "T": float(times[j]),
                    "times": np.asarray(times[: j + 1], dtype=np.float64),
                    "K": g["K"].to_numpy(float),
                    "B": g["B"].to_numpy(float),
                    "legs": [{s: lg[s] for s in BUCKET_STRUCTURES} for lg in legs],
                    "pos": cells.index.get_indexer(g.index),
                }
            )
        shape = (self.n_cells, len(RULES), 3)
        self.n = np.zeros(shape)
        self.beyond = np.zeros(shape)
        self.tl = np.zeros(shape)
        self.pay = {s: np.zeros(shape) for s in BUCKET_STRUCTURES}
        self.paths = 0
        self.sum_b6 = np.zeros(self.n_cells)
        self.sum_a1 = np.zeros(self.n_cells)
        self.sum_beyond = np.zeros(self.n_cells)

    def __call__(self, paths: bh.DayPaths) -> None:
        self.paths += paths.close.shape[0]
        for g in self.groups:
            j, side, T, t_grid = g["j"], g["side"], g["T"], g["times"]
            s_T = paths.close[:, j]
            if side > 0:
                run_d, run_c = paths.run_max_close[:, : j + 1], paths.run_max_cont[:, : j + 1]
            else:
                run_d, run_c = paths.run_min_close[:, : j + 1], paths.run_min_cont[:, : j + 1]
            for i, pos in enumerate(g["pos"]):
                K, B = g["K"][i], g["B"][i]
                lnb = np.log(B)
                beyond = side * (s_T - B) > 0
                pays = {s: bh.legs_payoff(g["legs"][i][s], side, s_T) for s in BUCKET_STRUCTURES}
                hit_d = (run_d > lnb) if side > 0 else (run_d < lnb)
                hit_c = (run_c >= lnb) if side > 0 else (run_c <= lnb)
                self.sum_b6[pos] += pays["B6"].sum()
                self.sum_beyond[pos] += beyond.sum()
                alive_d = ~hit_d[:, -1]
                self.sum_a1[pos] += (np.maximum(side * (s_T - K), 0.0) * alive_d).sum()
                for r_i, hit in enumerate((hit_c, hit_d)):
                    touched = hit[:, -1]
                    if not touched.any():
                        continue
                    first = (j + 1) - hit[touched].sum(axis=1)  # the running extreme is monotone
                    t_touch = t_grid[first]
                    # thirds of the life: (0, T/3], (T/3, 2T/3], (2T/3, T]
                    third = np.searchsorted(
                        np.array([T / 3.0, 2.0 * T / 3.0]), t_touch, side="left"
                    )
                    self.n[pos, r_i] += np.bincount(third, minlength=3)
                    self.beyond[pos, r_i] += np.bincount(
                        third, weights=beyond[touched], minlength=3
                    )
                    self.tl[pos, r_i] += np.bincount(third, weights=T - t_touch, minlength=3)
                    for s in BUCKET_STRUCTURES:
                        self.pay[s][pos, r_i] += np.bincount(
                            third, weights=pays[s][touched], minlength=3
                        )

    def columns(self, prefix: str) -> dict[str, FloatArray]:
        """Per cell: ``<prefix>_<rule>_{p,d,tl,vB6,vB5_2,vC8}<j>`` (``j`` = 1, 2, 3) and the
        unconditional ``<prefix>_bk_{b6,a1,beyond}`` (means over the paths)."""
        out: dict[str, FloatArray] = {}
        with np.errstate(divide="ignore", invalid="ignore"):
            for r_i, rule in enumerate(RULES):
                for j in range(3):
                    n = self.n[:, r_i, j]
                    out[f"{prefix}_{rule}_p{j + 1}"] = n / self.paths
                    out[f"{prefix}_{rule}_d{j + 1}"] = np.where(
                        n > 0, self.beyond[:, r_i, j] / n, np.nan
                    )
                    out[f"{prefix}_{rule}_tl{j + 1}"] = np.where(
                        n > 0, self.tl[:, r_i, j] / n, np.nan
                    )
                    for s in BUCKET_STRUCTURES:
                        out[f"{prefix}_{rule}_v{s}{j + 1}"] = np.where(
                            n > 0, self.pay[s][:, r_i, j] / n, np.nan
                        )
        out[f"{prefix}_bk_b6"] = self.sum_b6 / self.paths
        out[f"{prefix}_bk_a1"] = self.sum_a1 / self.paths
        out[f"{prefix}_bk_beyond"] = self.sum_beyond / self.paths
        return out


# --------------------------------------------------------------------------------------------
# indicators (addendum §7.1, §7.2)
# --------------------------------------------------------------------------------------------


def directional_ssr(
    d_sig: FloatArray, ret: FloatArray, mean_skew: float, min_days: int = 20
) -> tuple[float, float]:
    """``(ssr_up, ssr_dn)``: on the up-days (``ret > 0``) ``β = Σ dσ·r / Σ r²`` over the mean
    at-the-money skew, and the same on the down-days; NaN with fewer than ``min_days`` days on
    the side or a zero skew."""
    out = []
    ok = np.isfinite(d_sig) & np.isfinite(ret)
    for mask in (ret > 0, ret < 0):
        m = mask & ok
        if m.sum() < min_days or not np.isfinite(mean_skew) or mean_skew == 0:
            out.append(float("nan"))
            continue
        out.append(float(np.sum(d_sig[m] * ret[m]) / np.sum(ret[m] ** 2) / mean_skew))
    return out[0], out[1]


RV_WINDOWS: Final[tuple[int, ...]] = (21, 63, 252)
RV_HORIZONS: Final[tuple[int, ...]] = (21, 63, 126, 252)
RV_FLOOR: Final[float] = 0.05


def realised_variance_table(closes: pd.Series) -> pd.DataFrame:
    """Per date: the trailing realised variances (annualised, close-to-close, mean of squared
    log returns × 252) over :data:`RV_WINDOWS` ending that day, and the realised variance of
    the next ``H`` days for each horizon (NaN where the window is incomplete)."""
    r2 = np.log(closes).diff() ** 2 * 252.0
    out = pd.DataFrame(index=closes.index)
    for w in RV_WINDOWS:
        out[f"rv_{w}"] = r2.rolling(w).mean()
    for h in RV_HORIZONS:
        out[f"fwd_{h}"] = r2.rolling(h).mean().shift(-h)
    return out


def fit_rv_forecast(
    closes: pd.Series, last_fit_date: str = "2006-12-29"
) -> dict[int, dict[str, Any]]:
    """Addendum §7.2: for each horizon, ordinary least squares of the next-``H``-day realised
    variance on a constant and the trailing realised variances, on dates whose *future window
    ends* on or before ``last_fit_date``.  Returns the frozen coefficients and the in-sample R²."""
    tab = realised_variance_table(closes)
    dates = list(closes.index)
    out: dict[int, dict[str, Any]] = {}
    for h in RV_HORIZONS:
        cut = [d for d in dates if d <= last_fit_date]
        usable = set(cut[: max(len(cut) - h, 0)])  # the future window stays inside the sample
        fit = tab[tab.index.isin(usable)].dropna(
            subset=[f"fwd_{h}", *[f"rv_{w}" for w in RV_WINDOWS]]
        )
        X = np.column_stack([np.ones(len(fit)), *[fit[f"rv_{w}"] for w in RV_WINDOWS]])
        y = fit[f"fwd_{h}"].to_numpy()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ beta
        out[h] = {
            "coef": [float(b) for b in beta],
            "r2": float(1.0 - resid.var() / y.var()),
            "n": len(fit),
            "first": str(fit.index[0]),
            "last": str(fit.index[-1]),
        }
    return out


def rv_forecast(trailing: Sequence[float], coef: Sequence[float]) -> float:
    """``rv_fc`` = square root of the fitted variance (floored at 5 % vol): ``trailing`` are
    the realised variances over :data:`RV_WINDOWS` known before the entry."""
    var = coef[0] + float(np.dot(coef[1:], trailing))
    return float(np.sqrt(max(var, RV_FLOOR**2)))


# --------------------------------------------------------------------------------------------
# local-vol restart (addendum §6.3)
# --------------------------------------------------------------------------------------------


def restart_model(base: LocalVolSurface, tau: float, spot: float) -> LocalVol:
    """The local-vol model of ``base`` (an entry day's Dupire surface) restarted at time ``tau``
    with spot ``spot``: the same ``σ_loc(t, S)`` seen from ``tau`` — time shifted by ``tau``,
    the forward curve rolled to ``tau`` and re-based on ``spot``, each slice shifted in
    log-moneyness by the constant ``ln(spot / F_base(tau))`` (linear interpolation, flat outside
    the grid).  No change to the engine."""
    fc = base.forward_curve
    curve = ForwardCurve(float(spot), fc.rate_curve.rolled(tau), fc.dividend_curve.rolled(tau))
    t_new = base.t_grid[base.t_grid <= base.t_grid[-1] - tau + 1e-12]
    if t_new.size < 2:
        raise ValueError("the restart leaves no local-vol slice")
    # the slices at tau + t_new, linear in time between the base slices
    tq = t_new + tau
    hi = np.clip(np.searchsorted(base.t_grid, tq), 1, base.t_grid.size - 1)
    lo = hi - 1
    wgt = np.clip((tq - base.t_grid[lo]) / (base.t_grid[hi] - base.t_grid[lo]), 0.0, 1.0)
    at_tq = (1.0 - wgt)[:, None] * base.local_var[lo] + wgt[:, None] * base.local_var[hi]
    shift = float(np.log(spot / float(fc.forward(tau))))
    rows = np.empty_like(at_tq)
    for i, row in enumerate(at_tq):
        rows[i] = np.interp(base.k_grid + shift, base.k_grid, row)
    surface = LocalVolSurface(t_new, base.k_grid, rows, curve, base.diagnostics)
    return LocalVol(surface, curve)
