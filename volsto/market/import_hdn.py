"""HistoricalData.net end-of-day option-chain importer (SPEC §13, milestone M3b).

Pipeline, one function per step (each testable):

1. :func:`load_day` — read one daily 34-column CSV, keep the SPX/SPXW roots (or another
   underlying's root), drop rows without both quotes, record ``settlement_time`` and use it in
   the time-to-expiry convention (vendor README §5.2: calendar days / 365, minus one day for AM
   settlement), attach the manifest's Treasury par-yield curve interpolated at ``T`` (README §6).
   Slices are grouped by **(root, expiration)** as the vendor does: SPX (AM-settled monthlies)
   and SPXW (PM weeklies / end-of-month) share expiration dates but differ by one day in ``T``.
2. :func:`implied_forward` — put–call parity on mid prices near the money: regression of
   ``C − P`` on ``K`` gives ``F = −a/b`` and the implied discount factor ``−b`` (the vendor's
   ``iv`` / Greeks are cross-checks only; its quotes are not a synchronised snapshot).
3. :func:`to_grid_surface` — OTM options only, mid implied vols against the implied forward, a
   liquidity filter (minimum bid, maximum relative bid/ask spread in vol terms, from
   ``iv_bid``/``iv_ask`` where present), butterfly (convex OTM prices in strike) and calendar
   (total variance non-decreasing in ``T``) checks on the retained points.
4. :func:`fit_ssvi` — ``θ_T`` from the ATM total variance per expiry, global ``(ρ, η, γ)`` by
   least squares in vol space with the no-arbitrage constraints enforced through a bounded
   reparametrisation; residuals per expiry.  Imported surfaces default to eSSVI (``ρ`` per
   pillar, ``essvi=True``); synthetic configs keep plain SSVI.  Expiries under 3m are reported
   but sit outside the acceptance region (a single power-law φ cannot follow them).  An eSSVI
   fit whose total variance decreases in ``T`` somewhere on the Dupire range ``|k| ≤ 3`` is
   refitted under the calendar constraint by :func:`repair_calendar` (M10 Part 0; ``ρ_T``,
   ``γ``, ``η`` move, ``θ_T`` is frozen).
5. :func:`snapshot_config` — a dated market YAML (surface params, forward and rate curves,
   provenance: vendor, file checksum, filters, code version) loadable with
   :func:`volsto.market.loaders.load_ssvi_surface`.

CLI: ``volsto-import --vendor hdn --date 2022-09-15 --underlying SPX --root <sample dir>``.
Checked by ``tests/test_import_hdn.py``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from scipy.optimize import least_squares

import volsto
from volsto.config import to_mapping
from volsto.market.bs import black_vega, implied_vol
from volsto.market.curves import DiscountCurve, ForwardCurve
from volsto.market.surface import (
    CALENDAR_SEGMENT_N,
    CalendarCertificate,
    CalendarSegments,
    ESSVISurface,
    GridSurface,
    SSVISurface,
    calendar_knots,
    calendar_t_grid,
    certify_calendar,
    essvi_dw_dt,
    ssvi_theta,
)

FloatArray = NDArray[np.float64]
log = logging.getLogger(__name__)

HDN_COLUMNS: tuple[str, ...] = (
    "contract",
    "underlying",
    "expiration",
    "type",
    "strike",
    "style",
    "quote_date",
    "bid",
    "bid_size",
    "ask",
    "ask_size",
    "quote_time",
    "volume",
    "open_interest",
    "open",
    "high",
    "low",
    "close",
    "trade_vwap",
    "transactions",
    "multileg_volume",
    "active_minutes",
    "last_trade_date",
    "underlying_close",
    "settlement_time",
    "iv_bid",
    "iv_ask",
    "iv",
    "iv_flag",
    "delta",
    "gamma",
    "theta",
    "vega",
    "rho",
)
RATE_TENORS: FloatArray = np.array([1.0 / 12.0, 0.25, 1.0, 2.0, 5.0, 10.0, 30.0])
INDEX_ROOTS: dict[str, tuple[str, ...]] = {"SPX": ("SPX", "SPXW")}


@dataclass(frozen=True)
class HdnFilters:
    """Liquidity and range filters applied in :func:`to_grid_surface`."""

    min_bid: float = 0.05
    max_rel_spread_vol: float = 0.25  # (iv_ask − iv_bid) / iv_mid
    max_abs_log_moneyness: float = 0.6
    min_days: int = 7
    max_years: float = 3.0
    min_points_per_expiry: int = 8
    near_atm_band: float = 0.10  # |K/S − 1| band for the parity regression
    fit_min_days: int = 21  # SSVI fit uses expiries from here (weeklies keep the grid surface)
    fit_max_abs_log_moneyness: float = 0.25
    pillars: tuple[float, ...] = (1.0 / 12.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


# --------------------------------------------------------------------------------------------
# 1. load
# --------------------------------------------------------------------------------------------


def contract_root(contract: str) -> str:
    """OSI root: the symbol with its trailing 15 characters removed (README N1, §5.4)."""
    return contract[:-15]


def load_manifest(root: str | Path) -> dict[str, Any]:
    """``day_by_date/manifest.json`` of a sample / archive directory."""
    p = Path(root)
    for cand in (p / "day_by_date" / "manifest.json", p / "manifest.json"):
        if cand.exists():
            data: dict[str, Any] = json.loads(cand.read_text())
            return data
    raise FileNotFoundError(f"no manifest.json under {p}")


def rate_curve(manifest: dict[str, Any], quote_date: str) -> tuple[FloatArray, FloatArray]:
    """``(tenors, zero rates)`` of the manifest's Treasury curve for ``quote_date`` (README §6):
    percent → decimal, ``null`` tenors dropped."""
    rates = manifest.get("rates", {})
    if quote_date not in rates:
        raise KeyError(f"no rate curve for {quote_date} in the manifest")
    raw = rates[quote_date]
    tenors = [t for t, r in zip(RATE_TENORS, raw) if r is not None]
    vals = [float(r) / 100.0 for r in raw if r is not None]
    if not vals:
        raise ValueError(f"empty rate curve for {quote_date}")
    return np.array(tenors), np.array(vals)


def interpolate_rate(tenors: FloatArray, rates: FloatArray, T: FloatArray) -> FloatArray:
    """Vendor convention: linear in ``T`` between tenors, flat outside (README §6)."""
    return np.asarray(np.interp(T, tenors, rates), dtype=np.float64)


def time_to_expiry(
    quote_date: pd.Series, expiration: pd.Series, settlement: pd.Series
) -> pd.Series:
    """README §5.2: calendar days / 365, one day less for AM settlement (Saturday expirations
    predate this sample)."""
    days = (pd.to_datetime(expiration) - pd.to_datetime(quote_date)).dt.days.astype(float)
    days = days - (settlement.fillna("").str.upper() == "AM").astype(float)
    return days / 365.0


def load_day(
    path: str | Path, underlying: str = "SPX", *, manifest: dict[str, Any] | None = None
) -> pd.DataFrame:
    """One trading day's chain for ``underlying`` (SPX keeps the SPX and SPXW roots)."""
    p = Path(path)
    df = pd.read_csv(
        p,
        usecols=list(HDN_COLUMNS),
        dtype={
            "contract": str,
            "underlying": str,
            "type": str,
            "style": str,
            "settlement_time": str,
            "quote_date": str,
            "expiration": str,
        },
        low_memory=False,
    )
    if list(df.columns) != list(HDN_COLUMNS):
        raise ValueError("unexpected column layout (expected the 34 HistoricalData.net columns)")
    roots = INDEX_ROOTS.get(underlying.upper(), (underlying.upper(),))
    df["root"] = df["contract"].map(contract_root)
    df = df[df["root"].isin(roots)].copy()
    if df.empty:
        raise ValueError(f"no rows for roots {roots} in {p.name}")
    df = df[(df["bid"] > 0) & (df["ask"] > 0)].copy()
    df["cp"] = np.where(df["type"].str.lower() == "call", 1, -1)
    # SPX (AM) and SPXW (PM) contracts can share an expiration date with different T: group by
    # (root, expiration) as the vendor does (README §5.4)
    df["expiry"] = df["root"] + "|" + df["expiration"]
    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["T"] = time_to_expiry(df["quote_date"], df["expiration"], df["settlement_time"])
    df = df[df["T"] > 0].copy()
    quote_date = str(df["quote_date"].iloc[0])
    man = manifest or load_manifest(p.parents[1] if p.parent.name == "day_by_date" else p.parent)
    tenors, zeros = rate_curve(man, quote_date)
    df["r"] = interpolate_rate(tenors, zeros, df["T"].to_numpy())
    df["df"] = np.exp(-df["r"] * df["T"])
    df.attrs.update(
        {
            "quote_date": quote_date,
            "underlying": underlying.upper(),
            "file": p.name,
            "rate_tenors": tenors.tolist(),
            "rate_zeros": zeros.tolist(),
            "spot": float(df["underlying_close"].iloc[0]),
        }
    )
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------------------------
# 2. implied forward
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ForwardEstimate:
    expiry: str
    T: float
    forward: float
    forward_stderr: float
    discount: float
    n_pairs: int
    vendor_style_forward: float  # closest-strike parity with the manifest rate (README §5.4)

    @property
    def implied_rate(self) -> float:
        return float(-np.log(self.discount) / self.T)


def implied_forward(
    chain: pd.DataFrame, expiry: str, *, band: float = 0.10, min_pairs: int = 3
) -> ForwardEstimate:
    """``C_mid − P_mid = a + b K`` on strikes with both sides quoted within ``±band`` of spot:
    ``F = −a/b``, ``DF = −b``.  Falls back to the closest-strike pair when fewer than ``min_pairs``.
    """
    g = chain[chain["expiry"] == expiry]
    if g.empty:
        raise KeyError(f"no rows for expiry {expiry}")
    T = float(g["T"].iloc[0])
    S = float(g["underlying_close"].iloc[0])
    r = float(g["r"].iloc[0])
    calls = g[g["cp"] == 1].set_index("strike")["mid"]
    puts = g[g["cp"] == -1].set_index("strike")["mid"]
    pairs = pd.concat([calls.rename("C"), puts.rename("P")], axis=1, join="inner").sort_index()
    if pairs.empty:
        raise ValueError(f"no put-call pairs for expiry {expiry}")
    # vendor-style estimate: strike closest to spot, F = K + e^{rT}(C − P)
    i_near = int(np.argmin(np.abs(pairs.index.to_numpy() - S)))
    k_near = float(pairs.index.to_numpy()[i_near])
    c_near = float(pairs["C"].to_numpy()[i_near])
    p_near = float(pairs["P"].to_numpy()[i_near])
    vendor_f = float(k_near + np.exp(r * T) * (c_near - p_near))
    near = pairs[np.abs(pairs.index.to_numpy() / S - 1.0) <= band]
    if len(near) < min_pairs:
        near = pairs.iloc[
            np.argsort(np.abs(pairs.index.to_numpy() - S))[: max(min_pairs, 1)]
        ].sort_index()
    K = near.index.to_numpy(dtype=float)
    y = (near["C"] - near["P"]).to_numpy(dtype=float)
    if len(K) >= 2 and np.ptp(K) > 0:
        A = np.column_stack([np.ones_like(K), K])
        coef, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(coef[0]), float(coef[1])
        if b >= 0:  # degenerate: use the theoretical discount
            b = -np.exp(-r * T)
            a = float(np.mean(y - b * K))
        F = -a / b
        dof = max(len(K) - 2, 1)
        sigma2 = float(res[0]) / dof if len(res) else float(np.sum((y - A @ coef) ** 2)) / dof
        cov = sigma2 * np.linalg.inv(A.T @ A)
        # delta method for F = -a/b
        grad = np.array([-1.0 / b, a / (b * b)])
        se = float(np.sqrt(max(grad @ cov @ grad, 0.0)))
        disc = -b
    else:
        disc = float(np.exp(-r * T))
        F = float(K[0] + (y[0]) / disc)
        se = float("nan")
    return ForwardEstimate(expiry, T, float(F), se, float(disc), len(K), vendor_f)


def implied_forwards(
    chain: pd.DataFrame, *, max_years: float | None = None, **kwargs: Any
) -> dict[str, ForwardEstimate]:
    """Forward estimates per expiry key; expiries without any two-sided put–call pair (typically
    the longest listed ones) are skipped with a log message."""
    keys = sorted(
        chain["expiry"].unique(),
        key=lambda e: float(chain.loc[chain["expiry"] == e, "T"].iloc[0]),
    )
    out: dict[str, ForwardEstimate] = {}
    for e in keys:
        T = float(chain.loc[chain["expiry"] == e, "T"].iloc[0])
        if max_years is not None and max_years * 1.05 < T:
            continue
        try:
            out[e] = implied_forward(chain, e, **kwargs)
        except ValueError as exc:
            log.info("skipping expiry %s: %s", e, exc)
    return out


# --------------------------------------------------------------------------------------------
# 3. grid surface
# --------------------------------------------------------------------------------------------


@dataclass
class SurfacePoints:
    """Retained quotes, one row per (expiry, strike): mid implied vol vs the implied forward."""

    table: pd.DataFrame
    forwards: dict[str, ForwardEstimate]
    dropped: dict[str, int] = field(default_factory=dict)


def _butterfly_prune(g: pd.DataFrame) -> pd.DataFrame:
    """Iteratively drop points until OTM prices are convex in strike (puts below F, calls above)."""
    g = g.sort_values("strike").reset_index(drop=True)
    for _ in range(len(g)):
        if len(g) < 3:
            break
        K = g["strike"].to_numpy()
        P = g["otm_price"].to_numpy()
        # second differences of the OTM price in strike must be >= 0 within each wing
        worst, worst_val = -1, -1e-12
        for side in (-1, 1):
            m = g["cp"].to_numpy() == side
            idx = np.flatnonzero(m)
            if idx.size < 3:
                continue
            k, p = K[idx], P[idx]
            d2 = (p[2:] - p[1:-1]) / (k[2:] - k[1:-1]) - (p[1:-1] - p[:-2]) / (k[1:-1] - k[:-2])
            j = int(np.argmin(d2))
            if d2[j] < worst_val:
                worst_val = d2[j]
                worst = int(idx[j + 1])
        if worst < 0:
            break
        g = g.drop(index=worst).reset_index(drop=True)
    return g


def to_grid_surface(
    chain: pd.DataFrame, forwards: dict[str, ForwardEstimate], filters: HdnFilters | None = None
) -> tuple[GridSurface, SurfacePoints]:
    """Market slices from OTM mid quotes; see the module docstring for the filters and checks."""
    f = filters or HdnFilters()
    spot = float(chain["underlying_close"].iloc[0])
    rows = []
    dropped: dict[str, int] = {
        "expiry_range": 0,
        "not_otm": 0,
        "min_bid": 0,
        "spread": 0,
        "moneyness": 0,
        "no_iv": 0,
        "butterfly": 0,
        "few_points": 0,
        "calendar": 0,
    }
    for expiry, fe in sorted(forwards.items(), key=lambda kv: kv[1].T):
        if f.min_days / 365.0 > fe.T or f.max_years < fe.T:
            dropped["expiry_range"] += 1
            continue
        g = chain[chain["expiry"] == expiry].copy()
        F, df_ = fe.forward, fe.discount
        otm = ((g["cp"] == -1) & (g["strike"] < F)) | ((g["cp"] == 1) & (g["strike"] >= F))
        dropped["not_otm"] += int((~otm).sum())
        g = g[otm]
        liquid = g["bid"] >= f.min_bid
        dropped["min_bid"] += int((~liquid).sum())
        g = g[liquid]
        g["k"] = np.log(g["strike"] / F)
        inrange = g["k"].abs() <= f.max_abs_log_moneyness
        dropped["moneyness"] += int((~inrange).sum())
        g = g[inrange]
        if g.empty:
            continue
        K, cp, T = g["strike"].to_numpy(float), g["cp"].to_numpy(int), fe.T
        iv_mid = implied_vol(g["mid"].to_numpy(float), F, K, T, cp, df_)
        iv_b = implied_vol(g["bid"].to_numpy(float), F, K, T, cp, df_)
        iv_a = implied_vol(g["ask"].to_numpy(float), F, K, T, cp, df_)
        # prefer the vendor's one-sided vols where present (same convention, cross-check only)
        ivb = np.where(np.isfinite(g["iv_bid"].to_numpy(float)), g["iv_bid"].to_numpy(float), iv_b)
        iva = np.where(np.isfinite(g["iv_ask"].to_numpy(float)), g["iv_ask"].to_numpy(float), iv_a)
        ok = np.isfinite(iv_mid) & (iv_mid > 0)
        dropped["no_iv"] += int((~ok).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            rel = (iva - ivb) / iv_mid
        tight = ok & (np.isnan(rel) | (rel <= f.max_rel_spread_vol))
        dropped["spread"] += int((ok & ~tight).sum())
        g = g.assign(
            iv_mid=iv_mid, iv_bid_used=ivb, iv_ask_used=iva, otm_price=g["mid"].to_numpy(float)
        )[tight]
        n0 = len(g)
        g = _butterfly_prune(g)
        dropped["butterfly"] += n0 - len(g)
        if len(g) < f.min_points_per_expiry:
            dropped["few_points"] += len(g)
            continue
        g["expiry"] = expiry
        g["T"] = T
        g["forward"] = F
        g["w"] = g["iv_mid"] ** 2 * T
        rows.append(
            g[
                [
                    "expiry",
                    "T",
                    "forward",
                    "strike",
                    "cp",
                    "k",
                    "bid",
                    "ask",
                    "mid",
                    "iv_mid",
                    "iv_bid_used",
                    "iv_ask_used",
                    "w",
                    "iv",
                    "iv_flag",
                ]
            ]
        )
    if not rows:
        raise ValueError("no expiries survived the filters")
    table = pd.concat(rows, ignore_index=True)
    # calendar: on the common quoted k range of consecutive slices (union of their k points, as
    # GridSurface checks), w must not decrease in T.  Global fixed point: the worst violating
    # point of any consecutive pair is dropped from the slice it belongs to (a stale / wide quote)
    # until no pair violates; a slice left with too few points is removed.
    slices = [g.sort_values("k").reset_index(drop=True) for _, g in table.groupby("T", sort=True)]
    changed = True
    while changed:
        changed = False
        slices = [g for g in slices if len(g) >= f.min_points_per_expiry]
        for i in range(1, len(slices)):
            prev, cur = slices[i - 1], slices[i]
            lo, hi = max(prev["k"].min(), cur["k"].min()), min(prev["k"].max(), cur["k"].max())
            if lo > hi:
                continue
            k_all = np.unique(np.concatenate([prev["k"].to_numpy(), cur["k"].to_numpy()]))
            k_all = k_all[(k_all >= lo) & (k_all <= hi)]
            gap = np.interp(k_all, cur["k"].to_numpy(), cur["w"].to_numpy()) - np.interp(
                k_all, prev["k"].to_numpy(), prev["w"].to_numpy()
            )
            j = int(np.argmin(gap))
            if gap[j] >= -1e-8:
                continue
            k_bad = k_all[j]
            dropped["calendar"] += 1
            if np.any(np.isclose(cur["k"].to_numpy(), k_bad)):
                slices[i] = cur[~np.isclose(cur["k"].to_numpy(), k_bad)].reset_index(drop=True)
            else:
                slices[i - 1] = prev[~np.isclose(prev["k"].to_numpy(), k_bad)].reset_index(
                    drop=True
                )
            changed = True
            break
    dropped["few_points"] += int(len(table) - sum(len(g) for g in slices) - dropped["calendar"])
    table = pd.concat(slices, ignore_index=True)
    mats = sorted(table["T"].unique())
    ks = [table[table["T"] == T]["k"].to_numpy() for T in mats]
    ws = [table[table["T"] == T]["w"].to_numpy() for T in mats]
    fc = forward_curve_from_forwards(
        spot,
        chain.attrs["rate_tenors"],
        chain.attrs["rate_zeros"],
        {T: float(table[table["T"] == T]["forward"].iloc[0]) for T in mats},
    )
    surface = GridSurface(mats, ks, ws, fc, fc.rate_curve, max_maturity=max(mats[-1], 3.0))
    return surface, SurfacePoints(table, forwards, dropped)


def forward_curve_from_forwards(
    spot: float, tenors: list[float], zeros: list[float], forwards: dict[float, float]
) -> ForwardCurve:
    """Rate curve from the Treasury tenors (zero rates); dividend curve implied by the forwards:
    ``DF_q(T) = F(T) DF_r(T) / S``, log-linear between expiries."""
    rate = DiscountCurve(tenors, zeros)
    Ts = np.array(sorted(forwards))
    q_zero = np.array([-np.log(forwards[T] * float(rate.df(T)) / spot) / T for T in Ts])
    return ForwardCurve(spot, rate, DiscountCurve(Ts, q_zero))


# --------------------------------------------------------------------------------------------
# 4. SSVI fit
# --------------------------------------------------------------------------------------------


@dataclass
class SSVIFit:
    surface: SSVISurface | ESSVISurface
    params: dict[str, Any]
    residuals: pd.DataFrame  # per expiry: rms / max error in vol points, n points
    points: pd.DataFrame  # per point: market vol, model vol, error_vp

    def _mask(self, t_max: float, k_abs: float, t_min: float) -> pd.Series:
        p = self.points
        return (p["T"] >= t_min - 1e-9) & (p["T"] <= t_max + 1e-9) & (p["k"].abs() <= k_abs + 1e-9)

    def max_error(self, t_max: float = 2.0, k_abs: float = 0.2, t_min: float = 1.0 / 12.0) -> float:
        """Largest |model − market| in vol points for ``t_min ≤ T ≤ t_max``, ``|k| ≤ k_abs``."""
        return float(self.points.loc[self._mask(t_max, k_abs, t_min), "error_vp"].abs().max())

    def rms_error(self, t_max: float = 2.0, k_abs: float = 0.2, t_min: float = 1.0 / 12.0) -> float:
        return float(
            np.sqrt(np.mean(self.points.loc[self._mask(t_max, k_abs, t_min), "error_vp"] ** 2))
        )


def _eta_max(rho: float, gamma: float, theta_lo: float, theta_hi: float) -> float:
    """Largest ``η`` such that both SSVI butterfly conditions hold on ``[θ_lo, θ_hi]``."""
    th = np.linspace(theta_lo, theta_hi, 401)
    a = 1.0 + abs(rho)
    m1 = np.max(th ** (1.0 - gamma) * (1.0 + th) ** (gamma - 1.0))
    m2 = np.max(th ** (1.0 - 2.0 * gamma) * (1.0 + th) ** (2.0 * gamma - 2.0))
    return float(min(4.0 / (a * m1), np.sqrt(4.0 / (a * m2))))


def _isotonic_increasing(y: FloatArray, w: FloatArray) -> FloatArray:
    """Weighted pool-adjacent-violators: the non-decreasing sequence closest to ``y``."""
    vals = [float(v) for v in y]
    wts = [float(v) for v in w]
    blocks = [[i] for i in range(len(vals))]
    i = 0
    while i < len(vals) - 1:
        if vals[i] > vals[i + 1] + 1e-15:
            tot = wts[i] + wts[i + 1]
            vals[i] = (vals[i] * wts[i] + vals[i + 1] * wts[i + 1]) / tot
            wts[i] = tot
            blocks[i] += blocks[i + 1]
            del vals[i + 1], wts[i + 1], blocks[i + 1]
            i = max(i - 1, 0)
        else:
            i += 1
    out = np.empty(len(y))
    for v, b in zip(vals, blocks):
        out[b] = v
    return out


def pillar_atm_variances(
    mats: FloatArray, theta_slices: FloatArray, weights: FloatArray, pillars: tuple[float, ...]
) -> tuple[FloatArray, FloatArray]:
    """ATM total variance at the pillar tenors: weighted least squares of the piecewise-linear
    (in ``T``) interpolant through the slice values, then made non-decreasing (isotonic)."""
    pil = np.array([q for q in pillars if mats.min() * 0.95 <= q <= mats.max() * 1.05])
    if pil.size < 2:
        pil = np.array([mats.min(), mats.max()])
    knots = np.concatenate(([0.0], pil))
    A = np.zeros((mats.size, knots.size))
    for i, T in enumerate(mats):
        j = int(np.clip(np.searchsorted(knots, T) - 1, 0, knots.size - 2))
        lam = (T - knots[j]) / (knots[j + 1] - knots[j])
        A[i, j] = 1.0 - lam
        A[i, j + 1] = lam
    A = A[:, 1:]  # theta(0) = 0
    sw = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], theta_slices * sw, rcond=None)
    coef = np.maximum(coef, 1e-8)
    theta_p = _isotonic_increasing(coef, np.ones_like(coef))
    theta_p = theta_p + 1e-10 * np.arange(theta_p.size)  # strictly increasing
    return pil, theta_p


@dataclass(frozen=True)
class CalendarRepairConfig:
    """Settings of the eSSVI calendar repair (M10 Part 0, SPEC §13; measured on the 127 days of
    the 2022 H2 sample by ``scripts/essvi_calendar_gate.py``).  Checked by
    ``tests/test_calendar_repair.py``.

    Not part of :class:`volsto.config.SSVIConfig` on purpose: the repair changes the fitted
    parameters, not the surface definition, and a new ``SSVIConfig`` field would change every
    leverage cache key."""

    k_max: float = 3.0
    """Half-width in ``k`` of the invariant: the Dupire range (``LocalVolConfig.k_min`` /
    ``k_max``).  The fit window ``|k| ≤ 0.25`` never violates (0 of 127 days); ``±1`` (the
    constructor's check) misses days whose ``∂_T w < 0`` Dupire would silently floor."""
    n_k: int = 121
    """Log-moneyness points of the initial constraint grid and of the certificate's initial
    partition (step 0.05)."""
    n_seg: int = CALENDAR_SEGMENT_N
    """Maturities per knot segment of the initial constraint grid, both ends included (each
    end evaluated with the segment's own slopes: both one-sided limits at every knot)."""
    margin: float = 1e-4
    """The invariant: exact ``∂_T w ≥ margin`` per year on ``|k| ≤ k_max`` and every ``T``,
    both one-sided limits at every knot, PROVEN by
    :func:`volsto.market.surface.certify_calendar` on every repaired or identity result.  1e-4
    per year is the margin of the previous passes (1e-6 per step of about 0.01 y)."""
    headroom: float = 1e-5
    """The constraint points are held at ``margin + headroom``, so that between them the
    derivative has room to dip while staying above ``margin``; the certificate's cuts are the
    points it finds below ``margin + headroom/2`` (always new: every constraint point is above
    ``margin + headroom - tol``)."""
    tol: float = 1e-7
    """Feasibility tolerance at the constraint points (a penalty method reaches a bound only
    from the infeasible side); ``tol < headroom/2``."""
    mu0: float = 1e2
    """First penalty weight (objective ~1e2 vp², violation ~1e-2 per year)."""
    mu_growth: float = 10.0
    """Penalty growth per stage."""
    max_stages: int = 14
    """Penalty weights tried per attempt (each a ``least_squares`` solve)."""
    max_cuts: int = 30
    """Certificate rounds per attempt that add the violating points to the constraint set and
    re-solve at the same penalty weight (exchange method)."""
    max_nfev: int = 400
    """Function evaluations per solve (``scipy.optimize.least_squares``)."""
    rho_bound: float = 0.9999
    """Box on the pillar rho (the unconstrained tanh fit saturates at ``rho = -1`` on
    2022-12-22, which the constructor refuses)."""
    fallback_k_max: float = 1.0
    """Escalation step (ii): the constructor's own ``k`` range."""


DEFAULT_CALENDAR_REPAIR = CalendarRepairConfig()
"""The repair every import runs unless told otherwise (M10 Part 0)."""

CALENDAR_FALLBACKS: tuple[str, ...] = ("margin0", "k_abs_1", "ssvi")
"""Escalation order when the constrained refit finds no certified point: (i) zero margin,
(ii) ``k`` narrowed to ``fallback_k_max`` (zero margin), (iii) plain SSVI for the day
(:func:`fit_ssvi` refits with ``essvi=False``; one rho cannot violate)."""


@dataclass(frozen=True)
class CalendarRepair:
    """Outcome of :func:`repair_calendar`; every ``∂_T w`` is the exact derivative, per year.

    ``min_dw_dt_*`` are its smallest values on the initial constraint grid (``±k_max`` × the
    per-segment maturities, both knot limits); ``min_dw_dt1_*`` the constructor's own quantity
    (:meth:`volsto.market.surface.ESSVISurface.calendar_min_dw_dt`: ``±1``, 81 points,
    :func:`volsto.market.surface.calendar_t_grid`), refused below ``−1e-10``; ``floor`` and
    ``k_abs`` the invariant the result was PROVEN to satisfy (``∂_T w ≥ floor`` on
    ``|k| ≤ k_abs``; ``certificate`` the proof, ``lower_bound`` its smallest cell bound), both
    ``None``-valued / NaN when ``feasible`` is false; ``cost_*`` are ``½ Σ r²`` of the stage-2
    residual; ``stages`` counts ``least_squares`` solves, ``cuts`` the certificate rounds that
    added constraint points; ``fallback`` is ``None`` or the escalation step that produced the
    result (:data:`CALENDAR_FALLBACKS`)."""

    rhos: FloatArray
    gamma: float
    eta: float
    repaired: bool
    feasible: bool
    stages: int
    cuts: int
    min_dw_dt_before: float
    min_dw_dt_after: float
    min_dw_dt1_before: float
    min_dw_dt1_after: float
    lower_bound: float
    floor: float | None
    k_abs: float | None
    cost_before: float
    cost_after: float
    k_max: float
    margin: float
    fallback: str | None
    certificate: CalendarCertificate | None


def calendar_constraint_grid(
    pillars: FloatArray,
    max_maturity: float,
    cfg: CalendarRepairConfig,
    min_maturity: float = 1.0 / 365.0,
    *,
    k_max: float | None = None,
    n_k: int | None = None,
) -> tuple[FloatArray, NDArray[np.intp], FloatArray]:
    """``(ks, seg, ts)`` of the repair's initial constraint set: ``ks = linspace(−k_max, k_max,
    n_k)`` and, for every knot segment ``seg`` (:func:`volsto.market.surface.calendar_knots`),
    ``linspace(a, b, n_seg)`` — a knot appears once per segment it bounds, so the exact
    ``∂_T w`` is constrained from both sides there.  Checked by
    ``tests/test_calendar_repair.py``."""
    km = cfg.k_max if k_max is None else float(k_max)
    nk = cfg.n_k if n_k is None else int(n_k)
    kn = calendar_knots(min_maturity, pillars, max_maturity)
    zero = np.zeros_like(kn)
    seg, ts = CalendarSegments.from_knots(kn, zero, zero).segment_grid(cfg.n_seg)
    return np.linspace(-km, km, nk), seg, ts


def _eta_theta_range(theta_p: FloatArray) -> tuple[float, float]:
    """``θ`` range on which :func:`fit_ssvi` caps ``η`` (half the smallest, 1.5× the largest
    pillar variance)."""
    return float(theta_p.min()) * 0.5, float(theta_p.max()) * 1.5


def _surface_theta(pillars: FloatArray, theta_p: FloatArray, Tq: FloatArray) -> FloatArray:
    """``θ_T`` exactly as :meth:`volsto.market.surface.SSVISurface.theta` evaluates it (linear
    from 0, last forward variance extended beyond the last pillar) — the repair constrains the
    surface the importer returns, not the fit's clamped ``θ`` (see :func:`fit_ssvi`)."""
    return ssvi_theta(pillars, theta_p, Tq)


def repair_calendar(
    pillars: FloatArray,
    theta_p: FloatArray,
    rhos: FloatArray,
    gamma: float,
    eta: float,
    resid_fn: Callable[[FloatArray, float, float], FloatArray],
    *,
    gamma_bounds: tuple[float, float],
    cfg: CalendarRepairConfig,
    max_maturity: float | None = None,
    min_maturity: float = 1.0 / 365.0,
) -> CalendarRepair:
    """Constrained refit of the eSSVI stage-2 parameters under the calendar INVARIANT
    ``∂_T w(k, T) ≥ margin`` for every ``|k| ≤ k_max`` and every ``T`` in
    ``[min_maturity, max_maturity]``, both one-sided limits at every knot included (M10 Part 0,
    third pass, SPEC §13).

    ``θ_T`` and ``ρ_T`` are affine on every knot segment, so ``∂_T w`` there is the closed form
    :func:`volsto.market.surface.essvi_dw_dt` with the segment's slopes — no finite difference
    anywhere.  The invariant is enforced as a semi-infinite constraint by an exchange method:

    1. the exact ``∂_T w ≥ margin + headroom`` is imposed on the initial set
       :func:`calendar_constraint_grid` (both limits at every knot, ``n_seg`` maturities per
       segment, ``n_k`` log-moneyness points);
    2. once a solve meets it (to ``tol``), :func:`volsto.market.surface.certify_calendar`
       either PROVES ``∂_T w ≥ margin`` on the whole continuous range (the result is returned)
       or returns the points where ``∂_T w < margin + headroom/2``, which join the constraint
       set before the next solve at the same penalty weight.

    A result is returned only with a proof, so the guarantee holds by construction; the bound
    between grid points is the certificate's cell bound (:func:`dw_dt_lower_bound`), not an
    assumed curvature.  ``θ_T`` is frozen (it is already isotonic; the violation comes from
    ``θ_T(1+ρ_T)``).  The unknowns are the bounded natural variables ``z = (ρ_1..ρ_P, γ, u)``
    with ``η = 0.999 η_max(max|ρ|, γ) u``: ``ρ_T`` and ``γ`` are free inside their boxes, and
    only ``η`` is capped, so both butterfly conditions hold at the current ``max|ρ|`` on the
    union of the fit's θ range and the constructor's ``[θ(min_maturity), θ(max_maturity)]``.
    The objective is ``resid_fn(ρ, γ, η)`` (the fit's vega-weighted vol-point residuals)
    augmented by ``√μ max(0, margin + headroom − ∂_T w)`` on the constraint set, with
    ``μ = mu0·mu_growth^s`` over at most ``max_stages`` penalty weights, each a bounded
    ``least_squares`` started from the previous solve.  An input the certificate already
    proves is returned unchanged (``repaired=False``, bit identical).  Without a proof the
    escalation of :data:`CALENDAR_FALLBACKS` runs (each step proves its own weaker invariant);
    ``fallback='ssvi'`` returns the input parameters with ``feasible=False`` and leaves the SSVI
    refit to the caller.  ``θ`` is never moved, ``ρ`` never clipped after the fact.

    Measured on the 2022 H2 sample: ``scripts/essvi_calendar_gate.py`` (which also runs the
    dense check on every day).  Checked by ``tests/test_calendar_repair.py``."""
    pil = np.asarray(pillars, dtype=np.float64)
    th_p = np.asarray(theta_p, dtype=np.float64)
    rho0 = np.asarray(rhos, dtype=np.float64)
    n_p = pil.size
    if rho0.shape != pil.shape or th_p.shape != pil.shape:
        raise ValueError("one theta and one rho per pillar are required")
    if not 0.0 <= cfg.tol < 0.5 * cfg.headroom:
        raise ValueError("need 0 <= tol < headroom / 2 (a cut must be a new point)")
    max_mat = max(float(pil[-1]), 3.0) if max_maturity is None else float(max_maturity)
    # η cap on the fit's θ range AND on the constructor's butterfly range [θ(min), θ(max)], so
    # the repaired surface constructs (the unconstrained fit caps on the former only)
    theta_lo, theta_hi = _eta_theta_range(th_p)
    th_ends = _surface_theta(pil, th_p, np.array([min_maturity, max_mat]))
    theta_lo = min(theta_lo, float(th_ends[0]))
    theta_hi = max(theta_hi, float(th_ends[1]))
    g_lo, g_hi = gamma_bounds
    lo = np.concatenate([np.full(n_p, -cfg.rho_bound), [g_lo, 0.0]])
    hi = np.concatenate([np.full(n_p, cfg.rho_bound), [g_hi, 1.0]])

    # knot segments: θ frozen; ρ at the knots is linear in the pillar ρ (one row per knot)
    kn = calendar_knots(min_maturity, pil, max_mat)
    th_kn = ssvi_theta(pil, th_p, kn)
    E = np.stack([np.interp(kn, pil, e) for e in np.eye(n_p)], axis=1)
    frozen = CalendarSegments.from_knots(kn, th_kn, np.zeros_like(kn))
    seg_len = frozen.b - frozen.a

    def segments(rp: FloatArray) -> CalendarSegments:
        return CalendarSegments.from_knots(kn, th_kn, E @ rp)

    def eta_cap(rp: FloatArray, g: float) -> float:
        return 0.999 * _eta_max(float(np.max(np.abs(rp))), g, theta_lo, theta_hi)

    def unpack(z: FloatArray) -> tuple[FloatArray, float, float]:
        rp = np.asarray(z[:n_p], dtype=np.float64)
        g = float(z[n_p])
        return rp, g, float(eta_cap(rp, g) * float(z[n_p + 1]))

    Model = Callable[[FloatArray, float, float], FloatArray]

    def dw_dt_at(
        k: FloatArray, seg: NDArray[np.intp], T: FloatArray, *, paired: bool = False
    ) -> Model:
        """Exact ``∂_T w`` at the points ``(seg, T)`` (rows) × ``k`` (columns), or at one ``k``
        per row when ``paired``."""
        kk = k[:, None] if paired else k[None, :]
        a = frozen.a[seg]
        lam = ((T - a) / seg_len[seg])[:, None]
        th = frozen.theta(seg, T)[:, None]
        dth = frozen.dtheta[seg][:, None]
        R = E[seg] + (E[seg + 1] - E[seg]) * lam
        dR = (E[seg + 1] - E[seg]) / seg_len[seg][:, None]

        def D(rp: FloatArray, g: float, e: float) -> FloatArray:
            return essvi_dw_dt(kk, th, dth, (R @ rp)[:, None], (dR @ rp)[:, None], e, g)

        return D

    def grid_model(k_abs: float, n_k: int) -> Model:
        ks, seg, ts = calendar_constraint_grid(
            pil, max_mat, cfg, min_maturity, k_max=k_abs, n_k=n_k
        )
        return dw_dt_at(ks, seg, ts)

    wide = grid_model(cfg.k_max, cfg.n_k)
    # the constructor's own check, for the record (ESSVISurface.calendar_min_dw_dt)
    ks_c = np.linspace(
        -ESSVISurface.CALENDAR_K_ABS, ESSVISurface.CALENDAR_K_ABS, ESSVISurface.CALENDAR_N_K
    )
    seg_c, ts_c = frozen.points(calendar_t_grid(min_maturity, pil, max_mat))
    ctor = dw_dt_at(ks_c, seg_c, ts_c)

    def min_of(model: Model, rp: FloatArray, g: float, e: float) -> float:
        return float(np.min(model(rp, g, e)))

    def cost(rp: FloatArray, g: float, e: float) -> float:
        r = resid_fn(rp, g, e)
        return float(0.5 * np.dot(r, r))

    def certify(
        rp: FloatArray, g: float, e: float, k_abs: float, n_k: int, floor: float, stop: float
    ) -> CalendarCertificate:
        return certify_calendar(
            segments(rp), e, g, k_abs=k_abs, floor=floor, stop_below=stop, n_k=n_k, n_t=cfg.n_seg
        )

    u0 = eta / eta_cap(np.clip(rho0, lo[:n_p], hi[:n_p]), gamma)
    z0 = np.clip(np.concatenate([rho0, [gamma, u0]]), lo, hi)
    before = min_of(wide, rho0, gamma, eta)
    before1 = min_of(ctor, rho0, gamma, eta)
    cost0 = cost(rho0, gamma, eta)

    def result(
        rp: FloatArray,
        g: float,
        e: float,
        repaired: bool,
        stages: int,
        cuts: int,
        fb: str | None,
        cert: CalendarCertificate | None,
        k_abs: float | None,
    ) -> CalendarRepair:
        return CalendarRepair(
            rhos=rp,
            gamma=g,
            eta=e,
            repaired=repaired,
            feasible=cert is not None,
            stages=stages,
            cuts=cuts,
            min_dw_dt_before=before,
            min_dw_dt_after=min_of(wide, rp, g, e),
            min_dw_dt1_before=before1,
            min_dw_dt1_after=min_of(ctor, rp, g, e),
            lower_bound=cert.lower_bound if cert is not None else float("nan"),
            floor=cert.floor if cert is not None else None,
            k_abs=k_abs,
            cost_before=cost0,
            cost_after=cost(rp, g, e),
            k_max=cfg.k_max,
            margin=cfg.margin,
            fallback=fb,
            certificate=cert,
        )

    attempts: list[tuple[str | None, float, float, int]] = [
        (None, cfg.margin, cfg.k_max, cfg.n_k),
        ("margin0", 0.0, cfg.k_max, cfg.n_k),
        ("k_abs_1", 0.0, cfg.fallback_k_max, ESSVISurface.CALENDAR_N_K),
    ]
    total = 0
    cuts_total = 0
    for label, margin, k_abs, n_k in attempts:
        cert = certify(rho0, gamma, eta, k_abs, n_k, margin, margin)
        if cert.certified:
            # the unconstrained fit already satisfies this attempt: returned bit-identical
            return result(rho0, gamma, eta, False, total, cuts_total, label, cert, k_abs)
        target = margin + cfg.headroom
        models: list[Model] = [grid_model(k_abs, n_k)]
        z = z0.copy()
        mu = cfg.mu0
        weights = 0
        cuts = 0
        while weights < cfg.max_stages:
            root_mu = float(np.sqrt(mu))

            def aug(
                zz: FloatArray,
                root_mu: float = root_mu,
                target: float = target,
                models: tuple[Model, ...] = tuple(models),
            ) -> FloatArray:
                rp, g, e = unpack(zz)
                d = np.concatenate([m(rp, g, e).ravel() for m in models])
                return np.concatenate([resid_fn(rp, g, e), root_mu * np.maximum(0.0, target - d)])

            z = least_squares(aug, z, bounds=(lo, hi), method="trf", max_nfev=cfg.max_nfev).x
            total += 1
            rp, g, e = unpack(z)
            if min(min_of(m, rp, g, e) for m in models) >= target - cfg.tol:
                cert = certify(rp, g, e, k_abs, n_k, margin, margin + 0.5 * cfg.headroom)
                if cert.certified:
                    return result(rp, g, e, True, total, cuts_total + cuts, label, cert, k_abs)
                if cert.status != "violated" or cuts >= cfg.max_cuts:
                    log.warning("eSSVI calendar certificate %s after %d cuts", cert.status, cuts)
                    break
                models.append(dw_dt_at(cert.cut_k, cert.cut_seg, cert.cut_T, paired=True))
                cuts += 1
                continue  # the new points are violated: re-solve at the same weight
            mu *= cfg.mu_growth
            weights += 1
        cuts_total += cuts
    log.warning("eSSVI calendar repair found no certified point after %d solves", total)
    return result(rho0, gamma, eta, False, total, cuts_total, "ssvi", None, None)


def fit_ssvi(
    grid: GridSurface,
    points: SurfacePoints,
    *,
    filters: HdnFilters | None = None,
    essvi: bool = False,
    gamma_bounds: tuple[float, float] = (0.05, 1.0),
    weights: str = "vega",
    calendar_repair: CalendarRepairConfig | None = DEFAULT_CALENDAR_REPAIR,
) -> SSVIFit:
    """Power-law SSVI fit in vol space (SPEC §13): ``θ_T`` at the pillar tenors from the slices'
    ATM total variances (isotonic least squares), global ``(ρ, η, γ)`` by least squares with the
    butterfly constraints enforced by a bounded reparametrisation; ``essvi`` fits ``ρ`` at the
    pillars instead (piecewise-linear ``ρ_T``).  Expiries below ``fit_min_days`` and points beyond
    ``fit_max_abs_log_moneyness`` are excluded from the fit but reported in the residuals.

    eSSVI only: when ``calendar_repair`` is set (the default) the stage-2 parameters go through
    :func:`repair_calendar`, which leaves a fit whose calendar invariant is already proven
    bit-identical and otherwise refits ``(ρ_T, γ, η)`` until the invariant ``∂_T w ≥ margin``
    (exact derivative, ``|k| ≤ k_max``, every ``T``, both knot limits) is proven (``ρ_T`` and
    ``γ`` free, ``η`` capped); if no proven point exists the day is refitted as plain SSVI.
    ``params`` then carries ``calendar_repaired``, ``calendar_stages``, ``calendar_cuts``,
    ``calendar_min_dw_dt`` and ``calendar_min_dw_dt_before`` (smallest exact ``∂_T w`` per year
    on the repair's initial grid, ``±k_max``), ``calendar_min_dw_dt_1`` (the constructor's own
    quantity, ``±1``), ``calendar_lower_bound``, ``calendar_floor`` and ``calendar_k_abs`` (the
    proven invariant: ``∂_T w ≥ calendar_floor`` on ``|k| ≤ calendar_k_abs``, with the
    certificate's smallest cell bound), ``calendar_cost_delta``, ``calendar_fallback``,
    ``calendar_k_max`` and ``calendar_margin``.  ``calendar_repair=None`` restores the pre-M10
    behaviour (the constructor raises on a violation).

    Known defect, kept on purpose (M10 Part 0, owner's decision pending): ``theta_of`` below
    CLAMPS ``θ`` beyond the last pillar while the returned surface extrapolates it, so the
    residuals and ``points`` beyond the last pillar describe a slightly different surface
    (median over the 127 days of the 2022 H2 sample, ``|k| ≤ 0.2``, 3m-3y: RMS 0.1957 vp
    reported vs 0.1384 vp on the surface returned, ``scripts/essvi_calendar_gate.py``).
    Fixing it moves every snapshot.  Checked by ``tests/test_import_hdn.py``."""
    f = filters or HdnFilters()
    tbl = points.table
    mats_all = np.array(sorted(tbl["T"].unique()))
    theta_slices = np.array([float(grid.total_variance(0.0, T)) for T in mats_all])
    n_slice = np.array([int((tbl["T"] == T).sum()) for T in mats_all], dtype=float)
    fit_slice = mats_all >= f.fit_min_days / 365.0
    if not np.any(fit_slice):
        raise ValueError("no expiry at or beyond fit_min_days")
    pil, theta_p = pillar_atm_variances(
        mats_all[fit_slice], theta_slices[fit_slice], n_slice[fit_slice], f.pillars
    )
    fc = grid.forward_curve
    sel = (tbl["T"] >= f.fit_min_days / 365.0) & (tbl["k"].abs() <= f.fit_max_abs_log_moneyness)
    fit_tbl = tbl[sel]
    k = fit_tbl["k"].to_numpy(float)
    T = fit_tbl["T"].to_numpy(float)
    sig_mkt = fit_tbl["iv_mid"].to_numpy(float)
    knots_t = np.concatenate(([0.0], pil))
    knots_th = np.concatenate(([0.0], theta_p))

    def theta_of(Tq: FloatArray) -> FloatArray:
        return np.asarray(np.interp(Tq, knots_t, knots_th), dtype=np.float64)

    if weights == "vega":
        F = fit_tbl["forward"].to_numpy(float)
        wts = black_vega(F, F * np.exp(k), T, sig_mkt, 1.0) / F
        wts = np.sqrt(wts / wts.max())
    else:
        wts = np.ones_like(k)
    theta_lo, theta_hi = float(theta_p.min()) * 0.5, float(theta_p.max()) * 1.5
    n_rho = pil.size if essvi else 1

    def unpack(x: FloatArray) -> tuple[FloatArray, float, float]:
        rho_p = np.tanh(x[:n_rho])
        lo, hi = gamma_bounds
        gamma = lo + (hi - lo) / (1.0 + np.exp(-x[n_rho]))
        emax = _eta_max(float(np.max(np.abs(rho_p))), gamma, theta_lo, theta_hi)
        eta = emax * 0.999 / (1.0 + np.exp(-x[n_rho + 1]))
        return rho_p, float(gamma), float(eta)

    def rho_at(rho_p: FloatArray, Tq: FloatArray) -> FloatArray:
        if not essvi:
            return np.full_like(Tq, rho_p[0])
        return np.asarray(np.interp(Tq, pil, rho_p), dtype=np.float64)

    def model_vol_p(
        rho_p: FloatArray, gamma: float, eta: float, kq: FloatArray, Tq: FloatArray
    ) -> FloatArray:
        th = theta_of(Tq)
        r = rho_at(rho_p, Tq)
        phi = eta / (th**gamma * (1.0 + th) ** (1.0 - gamma))
        pk = phi * kq
        w = 0.5 * th * (1.0 + r * pk + np.sqrt((pk + r) ** 2 + 1.0 - r * r))
        return np.asarray(np.sqrt(np.maximum(w, 1e-12) / Tq), dtype=np.float64)

    def resid_p(rho_p: FloatArray, gamma: float, eta: float) -> FloatArray:
        return np.asarray(
            wts * (model_vol_p(rho_p, gamma, eta, k, T) - sig_mkt) * 100.0, dtype=np.float64
        )

    def resid(x: FloatArray) -> FloatArray:
        return resid_p(*unpack(x))

    x0 = np.concatenate([np.full(n_rho, np.arctanh(-0.6)), [0.0, 0.0]])
    sol = least_squares(resid, x0, method="trf", max_nfev=2000)
    rho_p, gamma, eta = unpack(sol.x)
    fit_cost = float(sol.cost)
    max_mat = max(float(pil[-1]), 3.0)
    calendar: dict[str, Any] = {}
    surface: SSVISurface | ESSVISurface
    if essvi and calendar_repair is not None:
        rep = repair_calendar(
            pil,
            theta_p,
            rho_p,
            gamma,
            eta,
            resid_p,
            gamma_bounds=gamma_bounds,
            cfg=calendar_repair,
            max_maturity=max_mat,
        )
        failure: str | None = None if rep.feasible else "no feasible point"
        if failure is None:
            try:
                surface = ESSVISurface(
                    pil,
                    theta_p,
                    rep.rhos,
                    rep.eta,
                    rep.gamma,
                    fc,
                    fc.rate_curve,
                    max_maturity=max_mat,
                )
            except ValueError as exc:
                failure = f"constructor: {exc}"
        if failure is not None:
            log.warning("eSSVI calendar repair failed (%s): plain SSVI for this day", failure)
            fb = fit_ssvi(
                grid, points, filters=f, essvi=False, gamma_bounds=gamma_bounds, weights=weights
            )
            # the slacks after are not evaluated: one rho is calendar-free by construction
            fb.params.update(
                calendar_repaired=False,
                calendar_stages=rep.stages,
                calendar_cuts=rep.cuts,
                calendar_min_dw_dt=None,
                calendar_min_dw_dt_1=None,
                calendar_min_dw_dt_before=rep.min_dw_dt_before,
                calendar_lower_bound=None,
                calendar_floor=None,
                calendar_k_abs=None,
                calendar_cost_delta=None,
                calendar_fallback="ssvi",
                calendar_fallback_reason=failure,
                calendar_k_max=rep.k_max,
                calendar_margin=rep.margin,
            )
            return fb
        if rep.repaired:
            rho_p, gamma, eta = rep.rhos, rep.gamma, rep.eta
            fit_cost = rep.cost_after
        calendar = {
            "calendar_repaired": rep.repaired,
            "calendar_stages": rep.stages,
            "calendar_cuts": rep.cuts,
            "calendar_min_dw_dt": rep.min_dw_dt_after,
            "calendar_min_dw_dt_1": rep.min_dw_dt1_after,
            "calendar_min_dw_dt_before": rep.min_dw_dt_before,
            "calendar_lower_bound": rep.lower_bound,
            "calendar_floor": rep.floor,
            "calendar_k_abs": rep.k_abs,
            "calendar_cost_delta": rep.cost_after - rep.cost_before,
            "calendar_fallback": rep.fallback,
            "calendar_k_max": rep.k_max,
            "calendar_margin": rep.margin,
        }
    elif essvi:
        surface = ESSVISurface(
            pil, theta_p, rho_p, eta, gamma, fc, fc.rate_curve, max_maturity=max_mat
        )
    else:
        surface = SSVISurface(
            pil, theta_p, float(rho_p[0]), eta, gamma, fc, fc.rate_curve, max_maturity=max_mat
        )
    pts = tbl[["expiry", "T", "k", "strike", "iv_mid"]].copy()
    pts["model_vol"] = model_vol_p(
        rho_p, gamma, eta, tbl["k"].to_numpy(float), tbl["T"].to_numpy(float)
    )
    pts["error_vp"] = 100.0 * (pts["model_vol"] - pts["iv_mid"])
    pts["in_fit"] = sel.to_numpy()
    res = pts.groupby(["expiry", "T"], as_index=False).agg(
        n=("k", "size"),
        rms_vp=("error_vp", lambda e: float(np.sqrt(np.mean(e**2)))),
        max_abs_vp=("error_vp", lambda e: float(np.abs(e).max())),
    )
    inner = (
        pts[pts["k"].abs() <= 0.2]
        .groupby("expiry")["error_vp"]
        .apply(lambda e: float(np.abs(e).max()))
    )
    res["max_abs_vp_20pct"] = res["expiry"].map(inner)
    res["in_fit"] = res["T"] >= f.fit_min_days / 365.0
    params = {
        "atm_maturities": pil.tolist(),
        "atm_vols": np.sqrt(theta_p / pil).tolist(),
        "rho": rho_p.tolist() if essvi else float(rho_p[0]),
        "eta": eta,
        "gamma": gamma,
        "essvi": essvi,
        "n_points_fit": len(k),
        "cost": fit_cost,
        **calendar,
    }
    return SSVIFit(surface, params, res, pts)


# --------------------------------------------------------------------------------------------
# 5. snapshot config
# --------------------------------------------------------------------------------------------


def snapshot_config(
    chain: pd.DataFrame,
    fit: SSVIFit,
    points: SurfacePoints,
    filters: HdnFilters,
    *,
    source_file: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    """Dated market config: ``market`` + ``ssvi`` sections (loadable by ``load_ssvi_surface``)
    plus ``provenance``."""
    fc = fit.surface.forward_curve
    sha = hashlib.sha256(Path(source_file).read_bytes()).hexdigest()
    listed: dict[str, Any] = next(
        (f for f in manifest.get("files", []) if f.get("name") == Path(source_file).name), {}
    )
    rho_param = fit.params["rho"]
    essvi = isinstance(rho_param, list)
    ssvi = {
        "atm_maturities": fit.params["atm_maturities"],
        "atm_vols": fit.params["atm_vols"],
        # eSSVI: the scalar rho keeps the section loadable as SSVIConfig; the pillar rhos go to
        # the essvi section, which load_ssvi_surface prefers when present
        "rho": float(np.mean(rho_param)) if essvi else float(rho_param),
        "eta": fit.params["eta"],
        "gamma": fit.params["gamma"],
        "max_maturity": float(max(fit.params["atm_maturities"][-1], 3.0)),
    }
    out: dict[str, Any] = {
        "market": {
            "spot": fc.spot,
            "rate_curve": {
                "times": fc.rate_curve.times.tolist(),
                "rates": fc.rate_curve.zero_rates.tolist(),
            },
            "dividend_curve": {
                "times": fc.dividend_curve.times.tolist(),
                "rates": fc.dividend_curve.zero_rates.tolist(),
            },
        },
        "ssvi": ssvi,
        "provenance": {
            "vendor": "historicaldata.net",
            "product": manifest.get("product"),
            "file": Path(source_file).name,
            "file_sha256": sha,
            "manifest_sha256": listed.get("sha256"),
            "quote_date": chain.attrs["quote_date"],
            "underlying": chain.attrs["underlying"],
            "roots": list(INDEX_ROOTS.get(chain.attrs["underlying"], (chain.attrs["underlying"],))),
            "filters": asdict(filters),
            "n_points": len(points.table),
            "n_expiries": int(points.table["expiry"].nunique()),
            "dropped": points.dropped,
            "fit": {
                "essvi": fit.params["essvi"],
                "cost": fit.params["cost"],
                "n_points_fit": fit.params["n_points_fit"],
                "max_abs_error_vp_20pct_1m_2y": fit.max_error(2.0, 0.2),
                "rms_error_vp_20pct_1m_2y": fit.rms_error(2.0, 0.2),
                **{key: val for key, val in fit.params.items() if key.startswith("calendar_")},
            },
            "forwards": {
                fe.expiry: {
                    "T": fe.T,
                    "forward": fe.forward,
                    "stderr": fe.forward_stderr,
                    "discount": fe.discount,
                    "n_pairs": fe.n_pairs,
                    "vendor_style": fe.vendor_style_forward,
                }
                for fe in points.forwards.values()
                if fe.T in set(points.table["T"])
            },
            "rate_curve_percent": {
                "tenors_years": chain.attrs["rate_tenors"],
                "zeros": chain.attrs["rate_zeros"],
            },
            "code_version": volsto.__version__,
            "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        },
    }
    if essvi:
        out["essvi"] = {"rhos": [float(r) for r in rho_param]}
    return out


def write_snapshot(cfg: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(to_mapping(cfg), fh, sort_keys=False)
    return p


def import_day(
    root: str | Path,
    date: str,
    underlying: str = "SPX",
    *,
    filters: HdnFilters | None = None,
    essvi: bool = True,
    calendar_repair: CalendarRepairConfig | None = DEFAULT_CALENDAR_REPAIR,
) -> tuple[dict[str, Any], SSVIFit, SurfacePoints, pd.DataFrame]:
    """Run the whole pipeline for one day; returns ``(config, fit, points, chain)``.
    ``calendar_repair`` is passed to :func:`fit_ssvi` (eSSVI only)."""
    root = Path(root)
    manifest = load_manifest(root)
    path = root / "day_by_date" / f"{date}_options.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    f = filters or HdnFilters()
    chain = load_day(path, underlying, manifest=manifest)
    fwds = implied_forwards(chain, max_years=f.max_years, band=f.near_atm_band)
    grid, points = to_grid_surface(chain, fwds, f)
    fit = fit_ssvi(grid, points, filters=f, essvi=essvi, calendar_repair=calendar_repair)
    cfg = snapshot_config(chain, fit, points, f, source_file=path, manifest=manifest)
    return cfg, fit, points, chain


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="volsto-import",
        description="Import a vendor option chain into a dated volsto market config",
    )
    ap.add_argument("--vendor", default="hdn", choices=["hdn"])
    ap.add_argument("--date", required=True, help="trading date YYYY-MM-DD")
    ap.add_argument("--underlying", default="SPX")
    ap.add_argument(
        "--root",
        default="data/hdn_sample/options_sample_2022H2",
        help="sample / archive directory (contains day_by_date/)",
    )
    ap.add_argument("--out", default="configs/surfaces/snapshots", help="output directory")
    ap.add_argument(
        "--ssvi",
        action="store_true",
        help="fit a single rho (SSVI); imported surfaces default to eSSVI (rho per pillar)",
    )
    ap.add_argument(
        "--no-calendar-repair",
        action="store_true",
        help="eSSVI: skip the calendar repair (pre-M10; the import fails on a violation)",
    )
    ap.add_argument("--min-bid", type=float, default=HdnFilters.min_bid)
    ap.add_argument("--max-rel-spread", type=float, default=HdnFilters.max_rel_spread_vol)
    ap.add_argument("--max-years", type=float, default=HdnFilters.max_years)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    filters = HdnFilters(
        min_bid=args.min_bid, max_rel_spread_vol=args.max_rel_spread, max_years=args.max_years
    )
    cfg, fit, points, _ = import_day(
        args.root,
        args.date,
        args.underlying,
        filters=filters,
        essvi=not args.ssvi,
        calendar_repair=None if args.no_calendar_repair else DEFAULT_CALENDAR_REPAIR,
    )
    suffix = "_ssvi" if args.ssvi else ""
    out = Path(args.out) / f"{args.underlying.lower()}_{args.date}{suffix}.yaml"
    write_snapshot(cfg, out)
    log.info(
        "wrote %s (%d points, %d expiries)",
        out,
        len(points.table),
        points.table["expiry"].nunique(),
    )
    log.info(
        "SSVI residuals per expiry (vol points):\n%s", fit.residuals.round(3).to_string(index=False)
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
