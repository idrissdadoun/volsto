"""Dispersion study: smiles, vanilla strips and risk-neutral marginals at entry (spec §1.1, §2).

One convention for every leg (single names, DJX, DIA): Black vols inverted from the vendor's
option values — the call at or above the forward, the put below — with the discount
``exp(−iRate·T)`` and ``T`` = calendar days / 365 (:func:`expiry_smiles`).  The forward of an
expiry is put–call parity on the values near the money for a European basket, and a carry
rule for American names (:data:`FORWARD_RULES`; check C1 chooses).  At a target maturity the
total variance is linear in time between the two bracketing expiries at fixed log-moneyness
(:func:`smile_at`).  From the smile on a fixed grid: the at-the-money-spot straddle, the strip
``E[R²]`` by Simpson's rule on each side of the forward, the log-contract variance, and the
marginal distribution of ``S_T/S_0`` as a quantile table for the copula (:class:`Marginal`).
All prices are undiscounted and in units of the spot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.special import ndtr, ndtri

from volsto.market.bs import implied_vol

FloatArray = NDArray[np.float64]

#: Points of each half of the log-moneyness grid (the two halves share ``k = 0``).
HALF_GRID: Final = 1000
#: Smallest option value inverted (half a cent), fewest strikes for an expiry's smile.
MIN_VALUE: Final = 0.005
MIN_STRIKES: Final = 3
#: Data hygiene (set before any result was looked at; counted in the entry files): a vol is
#: kept within a factor of the vendor's smoothed vol at the same strike, an expiry within a
#: band of the median carry of the ticker's expiries.
VENDOR_BAND: Final = 2.0
FORWARD_TOLERANCE: Final = (0.03, 0.05)
#: Added after report v2 (PROGRESS_Q2, "Expiries that are not smiles"; measured on every
#: expiry of every entry date by ``scripts/disp_scan_smiles.py``).  (1) When a ticker has, on
#: the day, expiries with usable strikes on both sides of the forward, the expiries with
#: strikes on one side only are dropped: their at-the-money vol would be the flat
#: extrapolation of a wing (UNH's November 2017 expiry on 2017-08-28: five strikes from 45 to
#: 65 for a share at 195, vols of 160 %).  (2) An expiry of five weeks or more whose
#: at-the-money vol is more than a factor ``TERM_BAND`` from the median of the four expiries
#: nearest in order is dropped (XOM's September 2017 expiry in March 2017: 200 % between two
#: expiries at 16 %).  ``EXPIRY_GUARDS`` is the switch of the scan that measures both.
TERM_BAND: Final = 2.0
TERM_MIN_T: Final = 0.1
EXPIRY_GUARDS: bool = True
#: Carry rules for the forward of an American name: ``F = S·exp(carry·T)``.
FORWARD_RULES: Final = ("rate", "rate_plus_residual", "parity")
#: Latent grid of the quantile tables: ``2^14`` points on ``[−8, 8]``.
Z_MAX: Final = 8.0
Z_POINTS: Final = 2**14


@dataclass(frozen=True)
class ExpirySmile:
    """One listed expiry: ``vol`` at the log-moneyness ``k = ln(K/F)`` of its strikes."""

    expiry: str
    T: float
    forward: float
    rate: float
    k: FloatArray
    vol: FloatArray
    vendor_vol: FloatArray
    n_dropped: int

    def at(self, k: FloatArray, vendor: bool = False) -> FloatArray:
        """Vol at ``k``: linear between strikes, flat outside the listed range."""
        return np.interp(k, self.k, self.vendor_vol if vendor else self.vol)


def money_reach(k: FloatArray) -> tuple[float, float]:
    """Log-moneyness of the nearest strike at or below the forward and at or above it
    (``-inf`` / ``+inf`` when a side has none)."""
    below, above = k[k <= 0.0], k[k >= 0.0]
    return (
        float(below.max()) if below.size else float("-inf"),
        float(above.min()) if above.size else float("inf"),
    )


def two_sided(k: FloatArray) -> bool:
    """Whether the strikes of one expiry stand on both sides of the forward."""
    below, above = money_reach(k)
    return bool(np.isfinite(below) and np.isfinite(above))


def term_ratio(atm: FloatArray) -> FloatArray:
    """Each expiry's at-the-money vol over the median of the four expiries nearest in order
    (two on each side; towards an end, the four nearest); NaN with fewer than three expiries."""
    v = np.asarray(atm, dtype=np.float64)
    n = v.size
    out = np.full(n, np.nan)
    if n < 3:
        return out
    for i in range(n):
        lo, hi = max(0, i - 2), min(n, i + 3)
        while hi - lo < 5 and (lo > 0 or hi < n):
            if lo > 0:
                lo -= 1
            if hi - lo < 5 and hi < n:
                hi += 1
        out[i] = v[i] / np.median(np.concatenate([v[lo:i], v[i + 1 : hi]]))
    return out


def parity_forward(g: pd.DataFrame, spot: float, df: float, band: float = 0.10) -> float:
    """Forward of one expiry from ``cValue − pValue = df·(F − K)`` on the strikes within
    ``band`` of the spot (the median over strikes; the discount is fixed)."""
    near = g[(g["strike"] / spot - 1.0).abs() <= band]
    if len(near) < 2:
        near = g.iloc[np.argsort((g["strike"] / spot - 1.0).abs().to_numpy())[:3]]
    f = near["strike"] + (near["cValue"] - near["pValue"]) / df
    return float(np.median(f.to_numpy(float)))


def expiry_smiles(
    chain: pd.DataFrame,
    trade_date: str,
    rule: str = "parity",
    min_days: int = 5,
    dividends: tuple[FloatArray, FloatArray] | None = None,
    spot: float | None = None,
) -> list[ExpirySmile]:
    """The smiles of every usable expiry of one ticker on one day.

    ``chain`` holds the vendor's rows of the ticker (``expirDate, strike, stkPx, cValue,
    pValue, smoothSmvVol, iRate, residualRateData``).  ``rule`` is one of
    :data:`FORWARD_RULES`, or ``"dividends"`` with ``dividends = (times in years, amounts)``
    of the projected dividends: ``F = S·exp(rT) − Σ d_j·exp(r(T − t_j))``.  Values below half a
    cent and failed inversions are dropped (counted in ``n_dropped``); an expiry needs
    :data:`MIN_STRIKES` vols and at least ``min_days`` calendar days."""
    if chain.empty:
        return []
    spot = float(chain["stkPx"].iloc[0]) if spot is None else float(spot)
    day = pd.Timestamp(trade_date)
    out: list[ExpirySmile] = []
    for expiry, g in chain.groupby("expirDate", sort=True):
        days = (pd.Timestamp(str(expiry)) - day).days
        if days < min_days:
            continue
        T = days / 365.0
        g = g.sort_values("strike").drop_duplicates("strike")
        r = float(np.nanmedian(g["iRate"].to_numpy(float)))
        df = float(np.exp(-r * T))
        if rule == "parity":
            F = parity_forward(g, spot, df)
        elif rule == "rate":
            F = spot * float(np.exp(r * T))
        elif rule == "rate_plus_residual":
            res = float(np.nanmedian(g["residualRateData"].to_numpy(float)))
            F = spot * float(np.exp((r + (res if np.isfinite(res) else 0.0)) * T))
        elif rule == "dividends":
            F = spot * float(np.exp(r * T))
            if dividends is not None:
                t, d = dividends
                inside = (t > 0) & (t <= T)
                F -= float(np.sum(d[inside] * np.exp(r * (T - t[inside]))))
        else:
            raise ValueError(f"unknown forward rule {rule!r}")
        if not np.isfinite(F) or F <= 0:
            continue
        K = g["strike"].to_numpy(float)
        call = K >= F
        value = np.where(call, g["cValue"].to_numpy(float), g["pValue"].to_numpy(float))
        ok = np.isfinite(value) & (value >= MIN_VALUE) & (K > 0)
        vol = np.full(K.size, np.nan)
        if ok.any():
            vol[ok] = implied_vol(value[ok], F, K[ok], T, np.where(call[ok], 1.0, -1.0), df)
        good = np.isfinite(vol) & (vol > 0.005) & (vol < 5.0)
        vendor = g["smoothSmvVol"].to_numpy(float)
        has_vendor = np.isfinite(vendor) & (vendor > 0)
        # an inversion more than a factor VENDOR_BAND away from the vendor's smoothed vol is a
        # one-cent value far out of the money or a non-standard row, not a vol
        good &= ~has_vendor | ((vol >= vendor / VENDOR_BAND) & (vol <= vendor * VENDOR_BAND))
        if int(good.sum()) < MIN_STRIKES:
            continue
        vendor = np.where(has_vendor, vendor, vol)
        out.append(
            ExpirySmile(
                str(expiry)[:10],
                T,
                F,
                r,
                np.log(K[good] / F),
                vol[good],
                vendor[good],
                int(K.size - good.sum()),
            )
        )
    if EXPIRY_GUARDS:
        both = [e for e in out if two_sided(e.k)]
        out = both or out
    out = consistent_forwards(out, spot)
    return consistent_vols(out) if EXPIRY_GUARDS else out


def consistent_forwards(smiles: list[ExpirySmile], spot: float) -> list[ExpirySmile]:
    """Drop the expiries whose forward is off the term structure of the others: with the median
    carry ``m`` of the expiries, an expiry is kept when ``|ln(F/S) − m·T| ≤ 0.03 + 0.05·T``
    (a parity forward read from non-standard rows can be 10 % away)."""
    if len(smiles) < 3:
        return smiles
    carry = np.array([np.log(e.forward / spot) / e.T for e in smiles])
    m = float(np.median(carry))
    return [
        e
        for e in smiles
        if abs(np.log(e.forward / spot) - m * e.T)
        <= FORWARD_TOLERANCE[0] + FORWARD_TOLERANCE[1] * e.T
    ]


def consistent_vols(smiles: list[ExpirySmile]) -> list[ExpirySmile]:
    """Drop the expiries of :data:`TERM_MIN_T` or more whose at-the-money vol (linear between
    strikes) is more than a factor :data:`TERM_BAND` from the median of the four expiries
    nearest in order (:func:`term_ratio`)."""
    if len(smiles) < 3:
        return smiles
    ratio = term_ratio(np.array([float(np.interp(0.0, e.k, e.vol)) for e in smiles]))
    return [
        e
        for e, x in zip(smiles, ratio, strict=True)
        if e.T < TERM_MIN_T or not np.isfinite(x) or 1.0 / TERM_BAND <= x <= TERM_BAND
    ]


@dataclass(frozen=True)
class TenorSmile:
    """The smile of one ticker at a target maturity: forward and rate interpolated in time,
    total variance linear in time between ``lo`` and ``hi`` at fixed log-moneyness."""

    spot: float
    T: float
    forward: float
    rate: float
    lo: ExpirySmile
    hi: ExpirySmile
    vendor: bool = False

    def vol(self, k: FloatArray | float) -> FloatArray:
        k = np.asarray(k, dtype=np.float64)
        lo, hi = self.lo, self.hi
        if lo is hi or hi.T == lo.T:
            return lo.at(k, self.vendor)
        w_lo = lo.at(k, self.vendor) ** 2 * lo.T
        w_hi = hi.at(k, self.vendor) ** 2 * hi.T
        a = (self.T - lo.T) / (hi.T - lo.T)
        return np.sqrt(np.maximum((1.0 - a) * w_lo + a * w_hi, 1e-10) / self.T)

    def vol_at_strike(self, strike_over_spot: float) -> float:
        return float(self.vol(np.log(strike_over_spot * self.spot / self.forward)))


def smile_at(
    expiries: list[ExpirySmile], spot: float, T: float, vendor: bool = False
) -> TenorSmile:
    """The :class:`TenorSmile` at ``T``.  Before the first expiry: the first expiry's vol;
    after the last: the last expiry's vol (flat).  The carry ``ln(F/S)/T`` and the rate are
    interpolated linearly in time (flat outside)."""
    if not expiries:
        raise ValueError("no usable expiry")
    ts = np.array([e.T for e in expiries])
    j = int(np.searchsorted(ts, T))
    if j == 0:
        lo = hi = expiries[0]
    elif j == len(expiries):
        lo = hi = expiries[-1]
    else:
        lo, hi = expiries[j - 1], expiries[j]
    carry = np.interp(T, ts, [np.log(e.forward / spot) / e.T for e in expiries])
    rate = float(np.interp(T, ts, [e.rate for e in expiries]))
    return TenorSmile(spot, T, spot * float(np.exp(carry * T)), rate, lo, hi, vendor)


def _lognormal_tail_second_moments(f: float, K: float, s: float) -> tuple[float, float]:
    """``(E[((X − K)⁺)²], E[((K − X)⁺)²])`` for a lognormal ``X`` of mean ``f`` and total
    vol ``s``."""
    d1 = (np.log(f / K) + 0.5 * s * s) / s
    d2, d0 = d1 - s, d1 + s
    up = f * f * np.exp(s * s) * ndtr(d0) - 2.0 * K * f * ndtr(d1) + K * K * ndtr(d2)
    dn = K * K * ndtr(-d2) - 2.0 * K * f * ndtr(-d1) + f * f * np.exp(s * s) * ndtr(-d0)
    return float(up), float(dn)


def _simpson(y: FloatArray, h: float) -> float:
    """Simpson's rule on an odd number of equally spaced points."""
    return float(h / 3.0 * (y[0] + y[-1] + 4.0 * y[1:-1:2].sum() + 2.0 * y[2:-1:2].sum()))


def convex_envelope(x: FloatArray, y: FloatArray) -> FloatArray:
    """The lower convex envelope of the points ``(x, y)`` (``x`` increasing), evaluated at
    ``x``."""
    keep: list[int] = []
    for i in range(x.size):
        while len(keep) >= 2:
            a, b = keep[-2], keep[-1]
            # drop b when it lies on or above the chord from a to i
            if (y[b] - y[a]) * (x[i] - x[a]) >= (y[i] - y[a]) * (x[b] - x[a]):
                keep.pop()
            else:
                break
        keep.append(i)
    return np.interp(x, x[keep], y[keep])


@dataclass
class Marginal:
    """The distribution of ``X = S_T/S_0`` implied by one smile, and what the study reads
    from it.  Prices undiscounted, in units of the spot."""

    T: float
    f: float  # forward over spot
    rate: float
    k: FloatArray
    vol: FloatArray
    strike: FloatArray  # K/S_0 on the grid
    call: FloatArray
    cdf: FloatArray  # at ``cdf_strike`` (the middles of the grid intervals)
    cdf_strike: FloatArray
    M: float  # E[(X − 1)²]
    var_vs: float  # log-contract variance, annualised
    tail_share: float
    straddle: float  # at-the-money-spot straddle
    atm_vol: float  # vol at K = S_0
    vega: float  # of the straddle, per vol point
    vol_90: float
    vol_110: float
    skew: float  # d vol / d ln K at the money
    _table: FloatArray | None = field(default=None, repr=False)

    def quantile(self, u: FloatArray) -> FloatArray:
        """``Q(u)`` in units of ``K/S_0`` (monotone interpolation of the cleaned CDF; beyond
        the grid a lognormal tail at the edge vol)."""
        u = np.asarray(u, dtype=np.float64)
        q = np.interp(u, self.cdf, self.cdf_strike)
        lo, hi = self.cdf[0], self.cdf[-1]
        s_lo, s_hi = self.vol[0] * np.sqrt(self.T), self.vol[-1] * np.sqrt(self.T)
        if lo > 1e-300:
            below = u < lo
            if below.any():
                q[below] = self.cdf_strike[0] * np.exp(
                    s_lo * (ndtri(np.maximum(u[below], 1e-300)) - ndtri(lo))
                )
        if hi < 1.0:
            above = u > hi
            if above.any():
                q[above] = self.cdf_strike[-1] * np.exp(
                    s_hi * (ndtri(np.minimum(u[above], 1 - 1e-16)) - ndtri(hi))
                )
        return q

    def table(self) -> FloatArray:
        """``Q(Φ(z))`` on the latent grid of :data:`Z_POINTS` points over ``±Z_MAX``."""
        if self._table is None:
            z = np.linspace(-Z_MAX, Z_MAX, Z_POINTS)
            self._table = self.quantile(ndtr(z))
        return self._table


def build_marginal(smile: TenorSmile, shift: float = 0.0) -> Marginal:
    """The :class:`Marginal` of ``smile``, every vol moved by ``shift`` (the parallel bump of
    the single-name sensitivity: 0.01 is one vol point).

    Grid (spec §2.4): ``k = ln(K/F)`` over ``±8a`` with ``a = max(σ_ATM√T, 0.1)`` and at least
    ``K/S_0`` in ``[0.2, 3]``; ``HALF_GRID`` intervals on each side of the forward."""
    T, f = smile.T, smile.forward / smile.spot
    sig0 = float(smile.vol(0.0)) + shift
    a = max(sig0 * np.sqrt(T), 0.1)
    k_lo = min(-8.0 * a, float(np.log(0.2 / f)))
    k_hi = max(8.0 * a, float(np.log(3.0 / f)))
    k = np.concatenate(
        [np.linspace(k_lo, 0.0, HALF_GRID + 1), np.linspace(0.0, k_hi, HALF_GRID + 1)[1:]]
    )
    vol = np.maximum(smile.vol(k) + shift, 1e-4)
    s = vol * np.sqrt(T)
    d1 = -k / s + 0.5 * s
    strike = f * np.exp(k)
    call = f * ndtr(d1) - strike * ndtr(d1 - s)
    put = call - (f - strike)
    n = HALF_GRID
    h_lo, h_hi = -k_lo / n, k_hi / n
    # strips: ∫ P dK below the forward, ∫ C dK above, with dK = K dk
    put_side = _simpson(put[: n + 1] * strike[: n + 1], h_lo)
    call_side = _simpson(call[n:] * strike[n:], h_hi)
    up_tail, _ = _lognormal_tail_second_moments(f, float(strike[-1]), float(s[-1]))
    _, dn_tail = _lognormal_tail_second_moments(f, float(strike[0]), float(s[0]))
    var = 2.0 * (put_side + call_side) + up_tail + dn_tail
    tail_share = (up_tail + dn_tail) / var if var > 0 else 0.0
    M = var + (f - 1.0) ** 2
    vs = (
        2.0
        * (_simpson(put[: n + 1] / strike[: n + 1], h_lo) + _simpson(call[n:] / strike[n:], h_hi))
        / T
    )
    # the distribution: F(K) = 1 + dC/dK on the convex envelope of the call prices (a smile
    # that is linear between strikes is not free of butterfly arbitrage everywhere; the lower
    # convex hull is the nearest arbitrage-free call curve from below and keeps the forward),
    # read at the middle of each grid interval, then clipped to [0, 1]
    hull = convex_envelope(strike, call)
    slope = np.diff(hull) / np.diff(strike)
    cdf = np.maximum.accumulate(np.clip(1.0 + slope, 0.0, 1.0))
    strike_mid = 0.5 * (strike[1:] + strike[:-1])
    # at-the-money-spot quantities
    k0 = float(np.log(1.0 / f))
    v0 = float(np.maximum(smile.vol(k0) + shift, 1e-4))
    s0 = v0 * np.sqrt(T)
    d10 = -k0 / s0 + 0.5 * s0
    c0 = f * ndtr(d10) - ndtr(d10 - s0)
    straddle = float(2.0 * c0 - (f - 1.0))
    vega = float(2.0 * f * np.exp(-0.5 * d10 * d10) / np.sqrt(2.0 * np.pi) * np.sqrt(T) * 0.01)
    eps = 0.01
    skew = float((smile.vol(k0 + eps) - smile.vol(k0 - eps)) / (2 * eps))
    return Marginal(
        T, f, smile.rate, k, vol, strike, call, cdf, strike_mid,
        float(M), float(vs), float(tail_share),
        straddle, v0, vega,
        float(smile.vol(np.log(0.9 / f)) + shift), float(smile.vol(np.log(1.1 / f)) + shift), skew,
    )  # fmt: skip


def straddle_greeks(m: Marginal) -> dict[str, float]:
    """Vega (per vol point), theta (per year, of the undiscounted forward premium) and cash
    gamma of the at-the-money-spot straddle of ``m``, Black–Scholes at its own vol."""
    s0 = m.atm_vol * np.sqrt(m.T)
    d1 = np.log(m.f) / s0 + 0.5 * s0
    pdf = float(np.exp(-0.5 * d1 * d1) / np.sqrt(2.0 * np.pi))
    gamma_cash = 2.0 * pdf * m.f / s0  # X² × d²/dX² of the straddle at X = 1 (forward f)
    return {
        "vega": m.vega,
        "theta": -0.5 * gamma_cash * m.atm_vol**2,
        "gamma_cash": gamma_cash,
    }


def flat_smile(vol: float, T: float, f: float = 1.0, rate: float = 0.0) -> TenorSmile:
    """A flat smile (tests and the synthetic checks C2, C6, C12)."""
    e = ExpirySmile(
        "flat", T, f, rate, np.array([-1.0, 0.0, 1.0]), np.full(3, vol), np.full(3, vol), 0
    )
    return TenorSmile(1.0, T, f, rate, e, e)
