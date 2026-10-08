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
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
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

CARRY_RULES: tuple[str, ...] = ("study", "zero")


class ListedExpiry(Protocol):
    """What the builder reads of a listed expiry (the study's ``ExpirySmile``)."""

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
) -> tuple[LocalCorrelationSpec, dict[str, Any]]:
    """The specification of the local correlation model on a basket of the study (module
    docstring) and what the build measured: per name the SVI fits' root-mean-square errors in
    vol points, the names whose last listed expiry is before the horizon (their surface is
    extrapolated flat in implied vol), and the same for the index.

    ``weights`` are normalised to sum to 1; ``lc.particle.horizon`` is replaced by ``horizon``.
    """
    if not (len(names) == len(weights) == len(spots)):
        raise ValueError("names, weights and spots must have the same length")
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
        expiries = list(smiles[name])
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
    index_cfg, index_fits = _surface_config(
        list(index_smiles), flat, horizon, records, f"{origin}:index"
    )
    kept = set(index_cfg.times)
    ratios = tuple(
        (float(e.T), float(e.forward) / float(index_spot))
        for e in index_smiles
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
        "index_extrapolated": bool(max(float(e.T) for e in index_smiles) < horizon),
        "index_slices": list(index_cfg.times),
    }
    return spec, info


__all__ = ["CARRY_RULES", "ListedExpiry", "default_lc_grid", "lc_spec_from_smiles", "name_market"]
