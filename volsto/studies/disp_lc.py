"""Dispersion study: the local correlation model's specification from the study's smiles
(SPEC §8.7, M12).

No input or output here: :func:`lc_spec_from_smiles` turns expiry smiles that are already
loaded (the study's :class:`~volsto.studies.disp_smile.ExpirySmile` lists, through
``scripts/disp_entries.py::marginals_for`` for the names and for DJX alike) into a
:class:`~volsto.config.LocalCorrelationSpec`:

* **each name**: spot 1; the rate curve of its expiries' rates and the dividend curve that
  reproduces its listed (parity) forwards ``F_e/S_0`` (:meth:`~volsto.market.curves.
  ForwardCurve.from_forwards`: piecewise-flat carry between expiries, flat beyond) — or no carry
  at all (``carry="zero"``, the reference implementation's world); an SVI slice per listed
  expiry by C8's selection (:func:`~volsto.market.svi_slices.fit_svi_surface`), read from the
  fit records when given;
* **the index target**: the listed index expiries fitted the same way in their own forward
  moneyness ``k = ln(K/F_I(T_e))``, to be used at the basket's forward moneyness (the alignment
  of SPEC §8.7); the listed forwards ``F_I(T_e)/I_0`` are kept for the alignment report;
* **the shared grid** of SPEC §8.7 for the horizon (:func:`default_lc_grid`).

**Which expiries enter** (:class:`ExpiryScreen`; owner's decisions of 2026-10-08, on by default):
the index target reads the standard monthly (third-Friday) expiries only, and an expiry of any
leg — index or name — is dropped when one of its three listed strikes nearest the forward has
no two-sided market (a positive bid on both the call and the put), or when its median half
bid-ask spread inside ±1 at-the-money standard deviation of the forward is above 2 vol points
(up to one year; above 6 beyond) (:func:`quote_quality` measures both on the vendor's chain).
Every expiry dropped is logged with its reason and returned in the build's information.  The
screen lives here and not in the study's loader: the study's published numbers were made with
every expiry its own guards kept, and its loader is unchanged.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.config import (
    CurveConfig,
    LocalCorrelationConfig,
    LocalCorrelationSpec,
    LocalVolConfig,
    MarketConfig,
    SimConfig,
    SviSurfaceConfig,
)
from volsto.market.bs import black_vega
from volsto.market.curves import DiscountCurve, ForwardCurve
from volsto.market.svi_slices import (
    FIT_MIN_POINTS,
    FIT_MIN_WIDTH,
    FIT_WIDTH_SD,
    SviSliceFit,
    fit_svi_surface,
    svi_fit_key,
    svi_fit_settings,
)

if TYPE_CHECKING:
    from volsto.calibration.fit_records import FitRecords

FloatArray = NDArray[np.float64]
log = logging.getLogger(__name__)

CARRY_RULES: tuple[str, ...] = ("study", "zero")
#: The quote screen reads the strikes inside this many at-the-money standard deviations of the
#: expiry's forward.
QUOTE_SD: Final[float] = 1.0


class ListedExpiry(Protocol):
    """What the builder reads of a listed expiry (the study's ``ExpirySmile``)."""

    @property
    def expiry(self) -> str: ...

    @property
    def T(self) -> float: ...

    @property
    def forward(self) -> float: ...

    @property
    def rate(self) -> float: ...

    @property
    def k(self) -> FloatArray: ...

    @property
    def vol(self) -> FloatArray: ...


def third_friday(expiry: str, listed: Collection[str] = ()) -> bool:
    """A standard monthly expiry: the third Friday of its month — or the Saturday after it (the
    convention until 2015), or the Thursday before it when that Friday is a market holiday.
    ``listed``: the leg's listed expiries; a Saturday or a Thursday counts only when the Friday
    beside it is not listed too (an index with daily expiries lists the Thursday every month)."""
    d = pd.Timestamp(expiry)
    if d.dayofweek == 4:
        return 15 <= d.day <= 21
    if d.dayofweek == 5 and 16 <= d.day <= 22:
        return (d - pd.Timedelta(days=1)).strftime("%Y-%m-%d") not in listed
    if d.dayofweek == 3 and 14 <= d.day <= 20:
        return (d + pd.Timedelta(days=1)).strftime("%Y-%m-%d") not in listed
    return False


#: The strike rule reads this many listed strikes nearest the forward.
NEAREST_STRIKES: Final[int] = 3


@dataclass(frozen=True)
class QuoteQuality:
    """The quotes of one listed expiry (:func:`quote_quality`).

    ``n_nearest``: how many of the ``NEAREST_STRIKES`` listed strikes nearest the forward exist
    (an expiry may list fewer); ``n_nearest_two_sided``: how many of them have a positive bid on
    both the call and the put.  ``n_strikes``: the listed strikes inside ``±QUOTE_SD``
    at-the-money standard deviations of the forward; ``n_two_sided``: those of them with a
    positive bid on both sides.  ``median_half_spread_vp``: the median over the strikes inside
    ``±QUOTE_SD`` sd — over the nearest strikes when none is inside (``spread_on_nearest``) — of
    the half bid-ask spread of the option the smile reads (the call at or above the forward, the
    put below) in vol points (the half spread divided by that option's Black vega; NaN when no
    strike has a valid two-way quote)."""

    expiry: str
    T: float
    n_strikes: int
    n_two_sided: int
    median_half_spread_vp: float
    n_nearest: int = 0
    n_nearest_two_sided: int = 0
    spread_on_nearest: bool = False


def quote_quality(chain: pd.DataFrame, expiries: Sequence[ListedExpiry]) -> dict[str, QuoteQuality]:
    """:class:`QuoteQuality` of each of ``expiries`` from the vendor's rows of the ticker on the
    day the smiles are from (columns ``expirDate, strike, cBidPx, cAskPx, pBidPx, pAskPx``).
    The forward, the rate and the at-the-money vol are the expiry smile's own."""
    out: dict[str, QuoteQuality] = {}
    by_expiry = {str(e)[:10]: g for e, g in chain.groupby("expirDate", sort=True)}
    for e in expiries:
        g = by_expiry.get(e.expiry)
        if g is None:
            out[e.expiry] = QuoteQuality(e.expiry, float(e.T), 0, 0, float("nan"))
            continue
        g = g.sort_values("strike").drop_duplicates("strike")
        strike = g["strike"].to_numpy(float)
        k = np.log(strike / e.forward)
        atm = float(np.interp(0.0, e.k, e.vol))
        near = np.abs(k) <= QUOTE_SD * atm * math.sqrt(e.T)
        c_bid, c_ask = g["cBidPx"].to_numpy(float), g["cAskPx"].to_numpy(float)
        p_bid, p_ask = g["pBidPx"].to_numpy(float), g["pAskPx"].to_numpy(float)
        both = (c_bid > 0) & (p_bid > 0)
        nearest = np.zeros(strike.size, dtype=bool)
        nearest[np.argsort(np.abs(k), kind="stable")[:NEAREST_STRIKES]] = True
        call = strike >= e.forward
        bid, ask = np.where(call, c_bid, p_bid), np.where(call, c_ask, p_ask)
        vol = np.interp(k, e.k, e.vol)
        vega = black_vega(e.forward, strike, e.T, vol, math.exp(-e.rate * e.T))
        quoted = np.isfinite(bid) & np.isfinite(ask) & (ask > 0) & (ask >= bid) & (vega > 0)
        on_nearest = not bool(near.any())
        valid = (nearest if on_nearest else near) & quoted
        half = 100.0 * 0.5 * (ask[valid] - bid[valid]) / vega[valid]
        median = float(np.median(half)) if valid.any() else float("nan")
        out[e.expiry] = QuoteQuality(
            e.expiry,
            float(e.T),
            int(near.sum()),
            int((near & both).sum()),
            median,
            int(nearest.sum()),
            int((nearest & both).sum()),
            on_nearest,
        )
    return out


@dataclass(frozen=True)
class ExpiryScreen:
    """Which listed expiries enter the model (module docstring; owner's decisions of 2026-10-08,
    second round).

    ``index_third_friday``: the index target reads standard monthly expiries only.
    ``nearest_two_sided``: the ``NEAREST_STRIKES`` listed strikes nearest the forward must each
    have a positive bid on both the call and the put (an expiry that lists fewer is dropped).
    ``max_half_spread_vp``: the median half spread inside ±1 sd is at most this many vol points
    for an expiry of up to ``long_maturity`` years, and at most ``max_half_spread_long_vp``
    beyond (``inf``: not read)."""

    index_third_friday: bool = True
    nearest_two_sided: bool = True
    max_half_spread_vp: float = 2.0
    max_half_spread_long_vp: float = 6.0
    long_maturity: float = 1.0

    def __post_init__(self) -> None:
        if not (self.max_half_spread_vp > 0 and self.max_half_spread_long_vp > 0):
            raise ValueError("the half-spread limits must be positive")
        if not self.long_maturity > 0:
            raise ValueError("long_maturity must be positive")

    @classmethod
    def off(cls) -> ExpiryScreen:
        """No screen: every expiry the loader returns (the study's own selection)."""
        return cls(False, False, math.inf, math.inf)

    @property
    def reads_quotes(self) -> bool:
        return (
            self.nearest_two_sided
            or math.isfinite(self.max_half_spread_vp)
            or math.isfinite(self.max_half_spread_long_vp)
        )

    def spread_limit(self, maturity: float) -> float:
        """The half-spread limit of an expiry of ``maturity`` years, in vol points."""
        if maturity <= self.long_maturity + 1e-12:
            return self.max_half_spread_vp
        return self.max_half_spread_long_vp

    def describe(self) -> dict[str, Any]:
        return {
            "index_third_friday": self.index_third_friday,
            "nearest_two_sided": self.nearest_two_sided,
            "nearest_strikes": NEAREST_STRIKES,
            "max_half_spread_vp": self.max_half_spread_vp,
            "max_half_spread_long_vp": self.max_half_spread_long_vp,
            "long_maturity": self.long_maturity,
            "quote_sd": QUOTE_SD,
        }


def screen_reason(e: ListedExpiry, q: QuoteQuality | None, screen: ExpiryScreen) -> tuple[str, str]:
    """``(rule, reason)`` of the quote rule that drops ``e`` — ``("", "")`` when it passes.
    ``rule`` is ``"strikes"`` or ``"spread"`` (or ``"quotes"`` when the day has none)."""
    if q is None:
        return "quotes", "no quotes"
    if screen.nearest_two_sided and q.n_nearest_two_sided < NEAREST_STRIKES:
        return "strikes", (
            f"{q.n_nearest_two_sided} of the {NEAREST_STRIKES} listed strikes nearest the forward "
            f"have a positive bid on both the call and the put ({q.n_nearest} listed)"
        )
    limit = screen.spread_limit(float(e.T))
    if math.isfinite(limit) and not (q.median_half_spread_vp <= limit):
        where = "on the nearest strikes" if q.spread_on_nearest else f"inside ±{QUOTE_SD:g} sd"
        return "spread", (
            f"median half spread {where} {q.median_half_spread_vp:.2f} vol points, above "
            f"{limit:g}"
        )
    return "", ""


def screen_expiries(
    leg: str,
    expiries: Sequence[ListedExpiry],
    quotes: Mapping[str, QuoteQuality] | None,
    screen: ExpiryScreen,
    *,
    index: bool = False,
) -> tuple[list[ListedExpiry], list[dict[str, Any]]]:
    """``(kept, dropped)`` of one leg's expiries under ``screen``; each dropped expiry is a
    record ``{"leg", "expiry", "T", "rule", "reason"}`` (``rule``: ``"third_friday"``,
    ``"strikes"``, ``"spread"`` or ``"quotes"``) and is logged."""
    if screen.reads_quotes and quotes is None:
        raise ValueError(
            f"{leg}: the quote screen needs the day's quote quality (quote_quality on the "
            "vendor's chain); pass screen=ExpiryScreen.off() to build without it"
        )
    kept: list[ListedExpiry] = []
    dropped: list[dict[str, Any]] = []
    listed = {e.expiry for e in expiries}
    for e in expiries:
        rule, reason = "", ""
        if index and screen.index_third_friday and not third_friday(e.expiry, listed):
            rule, reason = "third_friday", "not a third-Friday expiry"
        elif screen.reads_quotes and quotes is not None:
            rule, reason = screen_reason(e, quotes.get(e.expiry), screen)
        if reason:
            dropped.append(
                {"leg": leg, "expiry": e.expiry, "T": float(e.T), "rule": rule, "reason": reason}
            )
            log.info("expiry screen: %s %s (T = %.4f) dropped: %s", leg, e.expiry, e.T, reason)
        else:
            kept.append(e)
    return kept, dropped


def default_lc_grid(horizon: float) -> LocalVolConfig:
    """The shared Dupire grid of SPEC §8.7: up to 1y, ``t`` from 1/365 to ``T + 0.02`` on 120
    points and ``k`` over ±2 on 1601 points (``dk = 0.0025``, check C8's grid); beyond, 240
    points in ``t`` and ``k`` over ±3 on 2401 points (the library's default width: ±1.5 priced
    2y and 3y variance swaps low, ``LocalVolConfig``)."""
    if horizon <= 1.0 + 1e-12:
        return LocalVolConfig(
            t_min=1 / 365, t_max=horizon + 0.02, n_t=120, k_min=-2.0, k_max=2.0, n_k=1601
        )
    return LocalVolConfig(
        t_min=1 / 365, t_max=horizon + 0.02, n_t=240, k_min=-3.0, k_max=3.0, n_k=2401
    )


def name_market(
    expiries: Sequence[ListedExpiry], spot: float, carry: str = "study"
) -> MarketConfig:
    """The market of one name in units of its spot: spot 1, the zero rates of its expiries, and
    the dividend zero rates ``z_q(T_e) = z_r(T_e) − ln(F_e/S_0)/T_e`` that reproduce its listed
    forwards (``carry="study"``); no rates and no dividends for ``carry="zero"``."""
    if carry not in CARRY_RULES:
        raise ValueError(f"carry must be one of {CARRY_RULES}")
    if carry == "zero":
        return MarketConfig(1.0, CurveConfig.flat(0.0), CurveConfig.flat(0.0))
    if not expiries:
        raise ValueError("a name needs at least one listed expiry")
    times = [float(e.T) for e in expiries]
    rates = DiscountCurve(times, [float(e.rate) for e in expiries])
    curve = ForwardCurve.from_forwards(
        1.0, times, [float(e.forward) / spot for e in expiries], rates
    )
    return MarketConfig(
        1.0,
        CurveConfig(tuple(times), tuple(float(r) for r in rates.zero_rates)),
        CurveConfig(tuple(times), tuple(float(q) for q in curve.dividend_curve.zero_rates)),
    )


def _surface_config(
    expiries: Sequence[ListedExpiry],
    curve: ForwardCurve,
    horizon: float,
    records: FitRecords | None,
    origin: str,
) -> tuple[SviSurfaceConfig, list[SviSliceFit]]:
    surface, fits = fit_svi_surface(
        expiries, curve, horizon=horizon, records=records, origin=origin
    )
    keys: tuple[str, ...] | None = None
    if records is not None:
        settings = svi_fit_settings(FIT_WIDTH_SD, FIT_MIN_WIDTH, FIT_MIN_POINTS)
        by_time = {float(e.T): e for e in expiries}
        keys = tuple(svi_fit_key(f.T, by_time[f.T].k, by_time[f.T].vol, settings) for f in fits)
    cfg = SviSurfaceConfig(
        tuple(float(t) for t in surface.times),
        tuple(f.params for f in fits),
        float(surface.max_maturity),
        keys,
    )
    return cfg, fits


def lc_spec_from_smiles(
    names: Sequence[str],
    weights: Sequence[float],
    spots: Sequence[float],
    smiles: Mapping[str, Sequence[ListedExpiry]],
    index_smiles: Sequence[ListedExpiry],
    index_spot: float,
    horizon: float,
    *,
    lc: LocalCorrelationConfig,
    sim: SimConfig,
    local_vol: LocalVolConfig | None = None,
    carry: str = "study",
    records: FitRecords | None = None,
    label: str = "",
    origin: str = "disp_lc",
    screen: ExpiryScreen | None = None,
    quotes: Mapping[str, Mapping[str, QuoteQuality]] | None = None,
    index_quotes: Mapping[str, QuoteQuality] | None = None,
) -> tuple[LocalCorrelationSpec, dict[str, Any]]:
    """The specification of the local correlation model on a basket of the study (module
    docstring) and what the build measured: per name the SVI fits' root-mean-square errors in
    vol points, the names whose last listed expiry is before the horizon (their surface is
    extrapolated flat in implied vol), the same for the index, and the expiries the screen
    dropped (``"dropped"``: one record per expiry with its leg and reason; ``"screen"``: the
    settings).

    ``screen``: the expiry screen (default :class:`ExpiryScreen` — third-Friday index expiries
    and the quote screen on every leg); it needs ``quotes`` (per name) and ``index_quotes``,
    the :func:`quote_quality` of the day.  ``ExpiryScreen.off()`` takes every expiry given.
    ``weights`` are normalised to sum to 1; ``lc.particle.horizon`` is replaced by ``horizon``.
    """
    if not (len(names) == len(weights) == len(spots)):
        raise ValueError("names, weights and spots must have the same length")
    screen = ExpiryScreen() if screen is None else screen
    dropped: list[dict[str, Any]] = []
    w = [float(x) for x in weights]
    total = math.fsum(w)
    w = [x / total for x in w]
    w[-1] += 1.0 - math.fsum(w)  # the last ulp, so that the weights sum to 1 exactly enough
    grid = local_vol or default_lc_grid(horizon)
    markets: list[MarketConfig] = []
    surfaces: list[SviSurfaceConfig] = []
    rms: dict[str, list[float]] = {}
    extrapolated: list[str] = []
    for name, spot in zip(names, spots, strict=True):
        expiries, gone = screen_expiries(
            name, list(smiles[name]), None if quotes is None else quotes.get(name, {}), screen
        )
        dropped += gone
        if not expiries:
            raise ValueError(f"{name}: no listed expiry passes the screen")
        market = name_market(expiries, float(spot), carry)
        cfg, fits = _surface_config(
            expiries, ForwardCurve.from_config(market), horizon, records, f"{origin}:{name}"
        )
        markets.append(market)
        surfaces.append(cfg)
        rms[name] = [f.rms_vp for f in fits]
        if max(float(e.T) for e in expiries) < horizon:
            extrapolated.append(name)
    flat = ForwardCurve(1.0, DiscountCurve.flat(0.0), DiscountCurve.flat(0.0))
    index_expiries, gone = screen_expiries(
        "index", list(index_smiles), index_quotes, screen, index=True
    )
    dropped += gone
    if not index_expiries:
        raise ValueError("the index: no listed expiry passes the screen")
    index_cfg, index_fits = _surface_config(
        index_expiries, flat, horizon, records, f"{origin}:index"
    )
    kept = set(index_cfg.times)
    ratios = tuple(
        (float(e.T), float(e.forward) / float(index_spot))
        for e in index_expiries
        if float(e.T) in kept
    )
    spec = LocalCorrelationSpec(
        names=tuple(str(n) for n in names),
        weights=tuple(w),
        markets=tuple(markets),
        surfaces=tuple(surfaces),
        index_surface=index_cfg,
        lc=dataclasses.replace(
            lc, particle=dataclasses.replace(lc.particle, horizon=float(horizon))
        ),
        sim=sim,
        local_vol=grid,
        index_forward_ratios=ratios,
        label=label,
    )
    all_rms = [x for v in rms.values() for x in v]
    info = {
        "svi_rms_vp_median": float(np.median(all_rms)),
        "svi_rms_vp_max": float(np.max(all_rms)),
        "svi_rms_vp_index": [f.rms_vp for f in index_fits],
        "svi_rms_vp_by_name": {k: float(np.max(v)) for k, v in rms.items()},
        "n_slices_by_name": {k: len(v) for k, v in rms.items()},
        "names_extrapolated": extrapolated,
        "n_names_extrapolated": len(extrapolated),
        "index_extrapolated": bool(max(float(e.T) for e in index_expiries) < horizon),
        "index_slices": list(index_cfg.times),
        "screen": screen.describe(),
        "dropped": dropped,
    }
    return spec, info


__all__ = [
    "CARRY_RULES",
    "NEAREST_STRIKES",
    "QUOTE_SD",
    "ExpiryScreen",
    "ListedExpiry",
    "QuoteQuality",
    "default_lc_grid",
    "lc_spec_from_smiles",
    "name_market",
    "quote_quality",
    "screen_expiries",
    "screen_reason",
    "third_friday",
]
