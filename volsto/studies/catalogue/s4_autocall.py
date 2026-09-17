"""S4 — autocall and Phoenix model risk (SPEC §6.6 / §6.8 / §10.2, owner's M10 Part 2): the M6
headline notes (3y annual autocall, European KI 60%; 3y Phoenix, CB 70% with memory, American
daily KI 60%) across the models — price, ``P(KI)``, expected life, the autocall probabilities —,
the LSV-minus-LV difference by leg, and the forward-skew diagnostic.

**Data.**

* The M6 cells (``autocall 3y:<column>`` / ``phoenix 3y:<column>`` of the store's ``products``,
  :func:`volsto.studies.m6.baseline_values` keys) are read from the store where a point holds
  them (``source = store:<point id>``).  **The store labels the price and the legs "% notional"
  while it stores fractions of notional** (the M6 table's convention): this study multiplies
  them by 100 and labels them "% notional" correctly; the store is left as it is.  A selected
  point without M6 cells is, with ``price_missing``, priced here by
  :func:`volsto.studies.m6.run_m6_headline` from its cached leverage (``cache:<key>``;
  ``computed`` for the local vol).
* **LSV minus LV by leg**: priced here, every model runs on the study's shared step schedule and
  seed and the difference carries the **paired** stderr (the quadrature error is reported beside it;
  both errors carry their own delta-method stderrs); read from the store, the difference carries the
  quadrature error, which is not the exact error: the two estimates share random numbers and the
  store keeps no per-path samples to measure their correlation.
* **Beyond the horizon**: a leverage calibrated to a horizon shorter than the notes' 3y maturity
  (the CI toy build, 1y) holds its last slice to maturity; every number of such a model says so
  in its note, the ``horizon`` table lists the models and the narrative states it.
* **Forward-skew exposure** (``lv_exposure``):
  :func:`volsto.analytics.autocall.forward_skew_exposure` of each note and each leg under the
  **local vol** — a skew tent at each observation date, the model rebuilt by
  :class:`~volsto.risk.engine.LVBuilder` (a Dupire construction of the bumped surface, no
  leverage calibration), common random numbers; % of notional per vol point of 90/110 skew.
  The LSV counterpart needs a leverage recalibrated on every bumped surface, which a study never
  does: it is read from the store's ``risk`` table (``skew_T`` of ``autocall 3y``, the
  precompute's light / full tier, pillars of the grid) when present (``store_risk``), and
  otherwise reported as missing with the ``--risk light`` precompute line.  The two are not
  like-for-like: the store's tents sit on the grid's risk pillars, the local-vol tents on the
  observation dates, and a tent's support runs between its neighbouring pillars.
* **Forward-skew diagnostic**: the forward 90/110 skew ``σ(0.9) − σ(1.1)`` of the 1y → 2y and
  2y → 3y smiles (the windows between the observation dates) and its ratio to the surface's
  spot 1y 90/110 skew (exact) — how much forward skew each model carries into the dates where
  the notes' digitals sit.  Priced here, the two strikes share a path set and the stderr is
  paired; read from the store, the stderr is the **sum** of the two strikes' errors, a bound
  whatever the sign of their correlation, which the stored values do not give.  The window's
  ``beyond_horizon`` flag is the store's when read, else ``t2`` against the leverage's
  horizon.

**Params** (all required)::

    models: {surface, lv, one_factor, two_factor}
    price_missing: true
    pricing: {n_paths: 400000}
    lv_exposure: {n_paths: 100000, size: 0.01} | null
    store_risk: true

Seed: ``pricing``.  Checked by ``tests/test_catalogue_s1_s4.py``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from volsto.config import ConfigError, SimConfig
from volsto.models.base import Model
from volsto.products.base import Product
from volsto.studies import style
from volsto.studies.catalogue._common import (
    ERROR_STAT_NOTE,
    PAIRED_NOTE,
    QUADRATURE_NOTE,
    VP,
    Estimate,
    FloatArray,
    ModelPoint,
    empty_panel,
    leverage_requirements,
    load_model,
    mean_estimate,
    model_source,
    pair_samples,
    paired_difference,
    point_requirements,
    price_paired,
    pricing_sim,
    rss,
    same_times,
    select_models,
    shell_block,
    store_source,
    stored_ids,
    validate_selection,
)
from volsto.studies.results import (
    DIMENSIONLESS,
    Column,
    FigureSpec,
    Results,
    ResultsBuilder,
    TableSpec,
    records,
)
from volsto.studies.runner import Requirement, StudyContext

TITLE = "S4 - Autocall and Phoenix model risk"
QUESTION = (
    "How much do the 3y autocall and Phoenix prices, knock-in probabilities and expected lives "
    "move from local vol to stochastic-local vol at a fixed smile, which legs carry the "
    "difference, and how exposed is each leg to forward skew?"
)
REQUIRED_PARAMS = ("models", "price_missing", "pricing", "lv_exposure", "store_risk")
OPTIONAL_PARAMS: tuple[str, ...] = ()
#: What every exact row of this study is (``_common.unclassified_exact_rows``; the walking test
#: fails on any other exact row): ``(table regex, column regex, kind)``.
EXACT_KINDS: tuple[tuple[str, str, str], ...] = (
    ("horizon", "horizon", "input"),
    ("horizon", "notes_beyond", "flag"),
    ("fwd_skew", "beyond_horizon", "flag"),
    ("spot_skew", r"skew_[\d_]+", "closed form"),  # of the target surface
    # a leg whose bumped and base payoffs coincide on every path: exactly zero
    ("fwd_skew_lv", r"exposure_[\d.]+y", "closed form"),
    ("fwd_skew_lv_size", r"achieved_[\d.]+y", "closed form"),  # the bump actually applied
    ("setup", "value", "input"),  # the selection's counts and the config's inputs
)

PCT = "% notional"
VOL = "vol pts"
YEARS = "years"
#: The two notes, their results-table slug and their legs (SPEC §6.8 headline).
PRODUCTS: tuple[tuple[str, str], ...] = (("autocall 3y", "autocall"), ("phoenix 3y", "phoenix"))
SCALARS: tuple[str, ...] = ("price", "expected_life", "p_ki", "p_breach", "p_no_autocall")
#: Forward windows of the diagnostic (between the observation dates) and its strikes.
WINDOWS: tuple[tuple[float, float], ...] = ((1.0, 2.0), (2.0, 3.0))
SKEW_STRIKES: tuple[float, float] = (0.9, 1.1)
#: The spot skew maturity the forward skews are compared with (the windows' length).
SPOT_SKEW_T = 1.0
RISK_PRODUCT = "autocall 3y"
#: The headline notes' maturity (years).
NOTES_MATURITY = 3.0


# --------------------------------------------------------------------------------------------
# params and requirements
# --------------------------------------------------------------------------------------------


def validate_params(params: Mapping[str, Any]) -> None:
    validate_selection(params["models"])
    if not isinstance(params["price_missing"], bool):
        raise ConfigError("price_missing: expected true or false")
    pr = params["pricing"]
    if not isinstance(pr, Mapping) or set(pr) != {"n_paths"}:
        raise ConfigError("pricing: expected {n_paths}")
    if int(pr["n_paths"]) <= 0 or int(pr["n_paths"]) % 2:
        raise ConfigError("pricing.n_paths: a positive even number")
    lv = params["lv_exposure"]
    if lv is not None:
        if not isinstance(lv, Mapping) or set(lv) != {"n_paths", "size"}:
            raise ConfigError("lv_exposure: expected null or {n_paths, size}")
        if int(lv["n_paths"]) <= 0 or int(lv["n_paths"]) % 2 or float(lv["size"]) <= 0:
            raise ConfigError("lv_exposure: a positive even n_paths and a positive size")
    if not isinstance(params["store_risk"], bool):
        raise ConfigError("store_risk: expected true or false")


def _has_m6(ctx: StudyContext, ids: Sequence[str]) -> set[str]:
    """The stored ids whose ``products`` table carries the M6 cells."""
    from volsto.viewers.store import StoreReader

    df = StoreReader(ctx.store_root).products()
    if df.empty:
        return set()
    keys = df[df["key"].astype(str) == f"{PRODUCTS[0][0]}:price"]
    return {str(i) for i in keys["point_id"]} & set(ids)


def requirements(ctx: StudyContext) -> list[Requirement]:
    """A stored point with the M6 cells per model; with ``price_missing`` a model without them
    needs its cached leverage instead."""
    ctx.seed("pricing")
    models = select_models(ctx, ctx.params["models"])
    have = stored_ids(ctx)
    with_m6 = _has_m6(ctx, [mp.id for mp in models])
    if not ctx.params["price_missing"]:
        lacking = [mp.label for mp in models if mp.id in have and mp.id not in with_m6]
        if lacking:
            raise ConfigError(
                f"the store holds {lacking} without the M6 cells (a grid with products.m6 "
                "false); a --resume precompute skips stored points, so set price_missing: true "
                "to price them from the cached leverage"
            )
        return point_requirements(ctx, models)
    present = [mp for mp in models if mp.id in with_m6]
    absent = [mp for mp in models if mp.id not in with_m6]
    return point_requirements(ctx, present) + leverage_requirements(ctx, absent)


# --------------------------------------------------------------------------------------------
# compute
# --------------------------------------------------------------------------------------------


@dataclass
class _Model:
    """One model's M6 cells ``{"<product>:<column>": (value, stderr)}`` (fractions of notional
    for the price and the legs), the paired estimates behind them when priced here (``None``
    for store reads), its forward-smile rows, where they came from, and whether the 3y notes
    run beyond the leverage's calibration horizon."""

    cells: dict[str, tuple[float, float]]
    est: dict[str, Estimate] | None
    smile: pd.DataFrame
    source: str
    stored: bool
    horizon: float
    notes_beyond: bool
    same_grid: bool
    skew: dict[tuple[float, float], Estimate] | None = None


def _price_notes(model: Model, sim: SimConfig, reference_times: FloatArray) -> dict[str, Estimate]:
    """The M6 cells of both headline notes as paired estimates: the product list of
    :func:`volsto.analytics.autocall.autocall_report` priced on the grid of both notes (as
    :func:`volsto.studies.m6.run_m6_headline` does) carrying the shared calibration times."""
    from volsto.analytics.autocall import _leg_name
    from volsto.products.autocall import AutocallStatistic, KIPutLeg
    from volsto.studies.m6 import headline_products

    notes = headline_products(model.forward_curve.rate_curve, model.spot)
    out: dict[str, Estimate] = {}
    for pname, note in notes.items():
        legs = note.decompose()
        names = ["price", *[f"leg:{_leg_name(leg)}" for leg in legs]]
        items: list[Product] = [note, *legs]
        if note.ki_type != "european":
            items.append(KIPutLeg(note).european_counterpart())
            names.append("leg:put_european")
        n = note.n_dates
        items += [
            AutocallStatistic(note, "life"),
            AutocallStatistic(note, "ki_hit"),
            AutocallStatistic(note, "ki_breach"),
            *[AutocallStatistic(note, "autocall_at", i) for i in range(1, n + 2)],
        ]
        names += ["expected_life", "p_ki", "p_breach"]
        names += [f"p_autocall_{i}" for i in range(1, n + 1)] + ["p_no_autocall"]
        res = price_paired(model, sim, items, reference_times, grid_products=list(notes.values()))
        for name, r in zip(names, res, strict=True):
            out[f"{pname}:{name}"] = mean_estimate(pair_samples(r, sim.antithetic))
    return out


def _paired_skew(
    model: Model, sim: SimConfig, reference_times: FloatArray
) -> dict[tuple[float, float], Estimate]:
    """The forward 90/110 skew of each window as an :class:`Estimate` in vol units: the 0.9
    put and the 1.1 call priced on one path set, inverted to implied vols, and the influence of
    each implied vol taken as the price influence over the Black vega — so the skew's stderr
    accounts for the two strikes' correlation, whatever its sign."""
    from volsto.analytics.forward_smile import forward_ratio, forward_smile_from_prices
    from volsto.products.forward_start import ForwardStartOption

    disc = model.forward_curve.rate_curve
    items: list[Product] = []
    for t1, t2 in WINDOWS:
        items += [
            ForwardStartOption(t1, t2, SKEW_STRIKES[0], -1, disc),
            ForwardStartOption(t1, t2, SKEW_STRIKES[1], 1, disc),
        ]
    res = price_paired(model, sim, items, reference_times)
    out: dict[tuple[float, float], Estimate] = {}
    for w, (t1, t2) in enumerate(WINDOWS):
        pair = res[2 * w : 2 * w + 2]
        smile = forward_smile_from_prices(
            t1,
            t2,
            forward_ratio(model, t1, t2),
            list(SKEW_STRIKES),
            [-1, 1],
            pair,
            float(disc.df(t2)),
            sim.n_paths,
        )
        vols = []
        for i, r in enumerate(pair):
            e = mean_estimate(pair_samples(r, sim.antithetic))
            ratio = float(smile.vol_stderr[i] / smile.price_stderr[i])
            vols.append(Estimate(float(smile.vols[i]), e.influence * ratio))
        out[(t1, t2)] = vols[0].minus(vols[1])
    return out


def _price_model(
    ctx: StudyContext, mp: ModelPoint, n_paths: int, reference_times: FloatArray, model: Model
) -> _Model:
    sim = pricing_sim(ctx, mp, n_paths)
    ctx.log.info("%s: no M6 cells in the store, pricing at %d paths", mp.label, n_paths)
    est = _price_notes(model, sim, reference_times)
    cells = {k: e.pair for k, e in est.items()}
    horizon = float(mp.point.spec.particle.horizon)
    return _Model(
        cells,
        est,
        pd.DataFrame(),
        model_source(mp),
        False,
        horizon,
        _notes_beyond(mp, horizon),
        same_times(model, reference_times),
        _paired_skew(model, sim, reference_times),
    )


def _notes_beyond(mp: ModelPoint, horizon: float) -> bool:
    """Whether the notes' maturity lies beyond the leverage's calibration horizon (the local
    vol has no leverage)."""
    return (not mp.is_lv) and horizon + 1e-9 < NOTES_MATURITY


def _stored_model(ctx: StudyContext, mp: ModelPoint) -> _Model:
    reader = ctx.store()
    prod = reader.products(mp.id)
    cells = {
        str(r["key"]): (float(r["value"]), float(r["value_stderr"]))
        for r in records(prod)
        if ":" in str(r["key"])
    }
    point = reader.point(mp.id)
    horizon = float(point.get("horizon", mp.point.spec.particle.horizon))
    return _Model(
        cells,
        None,
        reader.forward_smile(mp.id),
        store_source(mp.id),
        True,
        horizon,
        _notes_beyond(mp, horizon),
        False,
    )


def _unit(column: str) -> tuple[str, float]:
    """``(unit, scale)`` of an M6 cell: price and legs in % of notional (the store's fractions
    times 100), the life in years, probabilities as such."""
    if column == "price" or column.startswith("leg:"):
        return PCT, 100.0
    if column == "expected_life":
        return YEARS, 1.0
    return DIMENSIONLESS, 1.0


def compute(ctx: StudyContext) -> Results:
    p = ctx.params
    models = select_models(ctx, p["models"])
    with_m6 = _has_m6(ctx, [mp.id for mp in models])
    n_paths = int(p["pricing"]["n_paths"])
    to_price = [mp for mp in models if mp.id not in with_m6]
    lsv_first = next((mp for mp in to_price if not mp.is_lv), None)
    loaded: dict[str, Model] = {}
    ref_times: FloatArray = np.empty(0)
    if lsv_first is not None:
        loaded[lsv_first.label] = load_model(ctx, lsv_first)
        ref_times = np.asarray(loaded[lsv_first.label].required_times(), dtype=np.float64)
    data: dict[str, _Model] = {}
    for mp in models:
        if mp.id in with_m6:
            data[mp.label] = _stored_model(ctx, mp)
        else:
            model = loaded.pop(mp.label, None) or load_model(ctx, mp)
            data[mp.label] = _price_model(ctx, mp, n_paths, ref_times, model)
    n_priced = sum(1 for d in data.values() if not d.stored)
    b = ResultsBuilder()
    note_unit = "the store's fraction of notional x 100 (the store labels it % notional)"
    for mp in models:
        d = data[mp.label]
        for pname, slug in PRODUCTS:
            for key, (v, s) in d.cells.items():
                prod, _, col = key.partition(":")
                if prod != pname or col.endswith("_minus_ref"):
                    continue
                unit, scale = _unit(col)
                table = f"legs_{slug}" if col.startswith("leg:") else slug
                notes = []
                if d.stored and unit == PCT:
                    notes.append(note_unit)
                if d.notes_beyond:
                    notes.append(_beyond_note(d.horizon))
                b.add(
                    table,
                    mp.label,
                    col,
                    scale * v,
                    scale * s,
                    unit=unit,
                    source=d.source,
                    note="; ".join(notes),
                    axes={**mp.axes, "horizon": d.horizon},
                )
        b.add_exact(
            "horizon",
            mp.label,
            "horizon",
            d.horizon,
            unit="y",
            source=d.source,
            note=(
                "the grid's horizon; the local vol has no leverage"
                if mp.is_lv
                else "the leverage's calibration horizon"
                + ("; " + _beyond_note(d.horizon) if d.notes_beyond else "")
            ),
            axes=mp.axes,
        )
        b.add_exact(
            "horizon",
            mp.label,
            "notes_beyond",
            float(d.notes_beyond),
            unit="",
            source=d.source,
            note=_beyond_note(d.horizon) if d.notes_beyond else "",
            axes=mp.axes,
        )
    lv = next((mp for mp in models if mp.is_lv), None)
    if lv is not None:
        _differences(b, models, data, lv)
    _forward_skew_diag(ctx, b, models, data)
    if p["store_risk"]:
        _store_risk(ctx, b, models, data)
    exposure = p["lv_exposure"]
    if exposure is not None and lv is not None:
        _lv_exposure(ctx, b, lv, int(exposure["n_paths"]), float(exposure["size"]))
    for row, val in (
        ("models", float(len(models))),
        ("read from the store", float(len(models) - n_priced)),
        ("priced from the cache", float(n_priced)),
        ("inline pricing paths", float(n_paths)),
        ("lv exposure paths", math.nan if exposure is None else float(exposure["n_paths"])),
        ("notes beyond the horizon", float(sum(d.notes_beyond for d in data.values()))),
    ):
        b.add_exact("setup", row, "value", val, unit="", source="computed")
    ctx.record("n_paths", {"inline": n_paths if n_priced else None, "priced_models": n_priced})
    return b.build()


#: Note of a forward-skew window ending beyond the calibration horizon.
WINDOW_BEYOND = "the window ends beyond the calibration horizon"


def _beyond_note(horizon: float) -> str:
    return (
        f"the 3y notes run beyond the leverage's {horizon:g}y calibration horizon: the last "
        "leverage slice is held to maturity"
    )


def _differences(
    b: ResultsBuilder, models: Sequence[ModelPoint], data: Mapping[str, _Model], lv: ModelPoint
) -> None:
    ref = data[lv.label]
    for mp in models:
        if mp.is_lv:
            continue
        d = data[mp.label]
        src = f"{d.source};{ref.source}"
        paired = d.est is not None and ref.est is not None
        for pname, slug in PRODUCTS:
            for key, (v, s) in d.cells.items():
                prod, _, col = key.partition(":")
                if prod != pname or key not in ref.cells:
                    continue
                if col not in SCALARS and not col.startswith(("leg:", "p_autocall_")):
                    continue
                unit, scale = _unit(col)
                ax = {**mp.axes, "paired": float(paired)}
                beyond = "; " + _beyond_note(d.horizon) if d.notes_beyond else ""
                if d.est is not None and ref.est is not None:
                    diff = paired_difference(
                        d.est[key], ref.est[key], same_grid=d.same_grid and ref.same_grid
                    )
                    note = PAIRED_NOTE + (
                        "" if diff.same_grid else "; the grids differ (still paired by path)"
                    )
                    value, se = diff.value, diff.stderr
                    finite = math.isfinite(diff.value)
                    for suffix, err, err_se, what in (
                        (
                            "quadrature",
                            diff.stderr_quadrature,
                            diff.quadrature_se,
                            "the quadrature error, for comparison",
                        ),
                        ("paired", diff.stderr, diff.stderr_se, "the paired stderr"),
                    ):
                        b.add(
                            f"se_{slug}",
                            mp.label,
                            f"{col}_{suffix}",
                            scale * err if finite else math.nan,
                            scale * err_se if finite else math.nan,
                            unit=unit,
                            source=src,
                            note=f"{what}; {ERROR_STAT_NOTE}{beyond}",
                            axes=ax,
                        )
                else:
                    rv, rs = ref.cells[key]
                    value, se = v - rv, rss(s, rs)
                    note = "LSV minus LV of stored values; " + QUADRATURE_NOTE
                note += beyond
                b.add(
                    f"diff_{slug}",
                    mp.label,
                    col,
                    scale * value,
                    scale * se,
                    unit=unit,
                    source=src,
                    note=note,
                    axes=ax,
                )


def _forward_skew_diag(
    ctx: StudyContext, b: ResultsBuilder, models: Sequence[ModelPoint], data: Mapping[str, _Model]
) -> None:
    from volsto.calibration.cache import build_market
    from volsto.calibration.fit_2f import spot_skew_90_110

    _, surface, _ = build_market(models[0].point.spec)
    spot_skew = float(spot_skew_90_110(surface, SPOT_SKEW_T))
    b.add_exact(
        "spot_skew",
        f"{SPOT_SKEW_T:g}y",
        "skew_90_110",
        VP * spot_skew,
        unit=VOL,
        source="computed",
        note="sigma(0.9 S0) - sigma(1.1 S0) of the target surface (fit_2f.spot_skew_90_110)",
    )
    for mp in models:
        d = data[mp.label]
        for t1, t2 in WINDOWS:
            row = f"{mp.label} | {t1:g}y→{t2:g}y"
            ax = {**mp.axes, "t1": t1, "t2": t2}
            if d.skew is not None:
                e = d.skew[(t1, t2)]
                fs, fs_se = e.value, e.stderr
                how_se = "paired: both strikes on one path set"
                beyond = (not mp.is_lv) and t2 > d.horizon + 1e-9
                how = "t2 against the leverage's horizon (the local vol has none)"
            else:
                sm = d.smile
                if sm.empty:
                    continue
                win = sm[np.isclose(sm["t1"], t1) & np.isclose(sm["t2"], t2)]
                lo = win[np.isclose(win["strike_moneyness"], SKEW_STRIKES[0])]
                hi = win[np.isclose(win["strike_moneyness"], SKEW_STRIKES[1])]
                if lo.empty or hi.empty:
                    continue
                fs = float(lo["iv"].iloc[0]) - float(hi["iv"].iloc[0])
                fs_se = float(lo["iv_stderr"].iloc[0]) + float(hi["iv_stderr"].iloc[0])
                how_se = (
                    "the sum of the two strikes' errors, a bound whatever the sign of their "
                    "correlation (the store keeps no per-path samples)"
                )
                if "beyond_horizon" in win.columns:
                    beyond = bool(lo["beyond_horizon"].iloc[0])
                    how = "the store's flag"
                else:  # a store written before the flag
                    beyond = (not mp.is_lv) and t2 > d.horizon + 1e-9
                    how = "derived from the point's horizon (the store predates the flag)"
            window = f"; {WINDOW_BEYOND} ({how})" if beyond else ""
            note = f"sigma(0.9) - sigma(1.1) of the forward smile; stderr: {how_se}{window}"
            b.add(
                "fwd_skew",
                row,
                "fwd_skew",
                VP * fs,
                VP * fs_se,
                unit=VOL,
                source=d.source,
                note=note,
                axes=ax,
            )
            b.add(
                "fwd_skew",
                row,
                "ratio_to_spot",
                fs / spot_skew,
                fs_se / abs(spot_skew),
                unit=DIMENSIONLESS,
                source=d.source,
                note=f"over the exact spot {SPOT_SKEW_T:g}y 90/110 skew{window}",
                axes=ax,
            )
            b.add_exact(
                "fwd_skew",
                row,
                "beyond_horizon",
                float(beyond),
                unit="",
                source=d.source,
                note=how + window,
                axes=ax,
            )


def _store_risk(
    ctx: StudyContext, b: ResultsBuilder, models: Sequence[ModelPoint], data: Mapping[str, _Model]
) -> None:
    have = stored_ids(ctx)
    reader = ctx.store()
    missing: list[str] = []
    for mp in models:
        risk = reader.risk(mp.id, RISK_PRODUCT) if mp.id in have else pd.DataFrame()
        rows = risk[risk["group"] == "skew"] if not risk.empty else risk
        if rows.empty:
            missing.append(mp.id)
            continue
        d = data[mp.label]
        beyond = "; " + _beyond_note(d.horizon) if d.notes_beyond else ""
        for r in records(rows):
            T = float(r["T"])
            b.add(
                "skew_store",
                mp.label,
                f"skew_T_{T:g}y",
                100.0 * float(r["value"]),
                100.0 * float(r["value_stderr"]),
                unit="% notional per vol pt",
                source=store_source(mp.id),
                note=f"store risk tier {r.get('tier', '')}, variant {r.get('variant', '')}; "
                "the store's fraction of notional x 100; a tent on the grid's risk pillar "
                f"{T:g}y" + beyond,
                axes={**mp.axes, "T": T},
            )
    cmd = ""
    if missing:
        cmd = f"{ctx.precompute_line(missing)} --risk light"
    b.add_exact(
        "setup",
        "models without stored skew_T",
        "value",
        float(len(missing)),
        unit="",
        source="computed",
        note=cmd,
    )


def _lv_exposure(
    ctx: StudyContext, b: ResultsBuilder, lv: ModelPoint, n_paths: int, size: float
) -> None:
    from volsto.analytics.autocall import forward_skew_exposure, state_model_factory
    from volsto.risk.engine import LVBuilder, RiskState
    from volsto.studies.m6 import headline_products

    state = RiskState(lv.point.spec, None, lv.label)
    builder = LVBuilder(state)
    base = builder.base_model
    factory = state_model_factory(builder, state, "recalibrate")
    sim = pricing_sim(ctx, lv, n_paths)
    ctx.log.info("forward-skew exposure under the local vol at %d paths", n_paths)
    products = headline_products(base.forward_curve.rate_curve, base.spot)
    for pname, slug in PRODUCTS:
        frame = forward_skew_exposure(products[pname], factory, sim, size=size, with_legs=True)
        for r in records(frame):
            leg = str(r["leg"])
            T = float(r["T"])
            row = f"{slug} / {leg}"
            ax = {"product": pname, "leg": leg, "T": T}
            if float(r["exposure"]) == 0.0 and float(r["exposure_stderr"]) == 0.0:
                b.add_exact(
                    "fwd_skew_lv",
                    row,
                    f"exposure_{T:g}y",
                    0.0,
                    unit="% notional per vol pt",
                    source="computed",
                    note="exactly zero on every path: the leg settles before the tent's "
                    "support, so the bumped and base payoffs coincide",
                    axes=ax,
                )
                b.add_exact(
                    "fwd_skew_lv_size",
                    row,
                    f"achieved_{T:g}y",
                    VP * float(r["achieved"]),
                    unit=VOL,
                    source="computed",
                    note="skew bump actually applied (halved on arbitrage failures)",
                    axes=ax,
                )
                continue
            b.add(
                "fwd_skew_lv",
                row,
                f"exposure_{T:g}y",
                100.0 * float(r["exposure"]),
                100.0 * float(r["exposure_stderr"]),
                unit="% notional per vol pt",
                source="computed",
                note="local vol rebuilt on the bumped surface (Dupire), common random numbers",
                axes=ax,
            )
            b.add_exact(
                "fwd_skew_lv_size",
                row,
                f"achieved_{T:g}y",
                VP * float(r["achieved"]),
                unit=VOL,
                source="computed",
                note="skew bump actually applied (halved on arbitrage failures)",
                axes=ax,
            )
    ctx.record("lv_exposure", {"n_paths": n_paths, "size": size})


# --------------------------------------------------------------------------------------------
# tables, figures, narrative (results only)
# --------------------------------------------------------------------------------------------


def _has(results: Results, table: str) -> bool:
    return table in results.tables()


HEADERS = {
    "price": "price",
    "expected_life": "E[life]",
    "p_ki": "P(KI)",
    "p_breach": "P(breach)",
    "p_no_autocall": "P(no AC)",
    "leg:put_european": "put (Eur.)",
}
#: Scalar columns of the first table of a note; the others go to the second.
FIRST_SCALARS: tuple[str, ...] = ("price", "expected_life", "p_ki", "p_breach")


def _header(col: str) -> str:
    if col in HEADERS:
        return HEADERS[col]
    if col.startswith("p_autocall_"):
        return f"date {col.rsplit('_', 1)[1]}"
    if col.startswith("leg:autocall_"):
        return f"AC {col.rsplit('_', 1)[1]}"
    if col.startswith("leg:"):
        return col[4:].replace("_", " ")
    return col


def _cols_where(results: Results, table: str, keep: Any) -> tuple[Column, ...]:
    def unit(c: str) -> str:
        if c == "price" or c.startswith("leg:"):
            return "%"
        return "y" if c == "expected_life" else ""

    return tuple(Column(c, _header(c), unit=unit(c)) for c in results.columns(table) if keep(c))


def _is_ac_leg(c: str) -> bool:
    return c.startswith("leg:autocall_") or c == "leg:bond"


def _is_other_leg(c: str) -> bool:
    return c.startswith("leg:") and not _is_ac_leg(c)


def _pairing(results: Results, table: str) -> str:
    """How the differences of ``table`` are errored, from the rows' ``paired`` coordinate."""
    flags = {float(results.axes(table, r).get("paired", 0.0)) for r in results.rows(table)}
    if flags == {1.0}:
        return "paired stderr: both models priced here on the shared random numbers"
    if flags == {0.0}:
        return (
            "stderr: the quadrature error of two stored values, not the exact error - they "
            "share random numbers and the store keeps no per-path samples to measure their "
            "correlation"
        )
    return (
        "paired where both models were priced here, else the quadrature error of the stored "
        "values, not the exact error (see notes)"
    )


def tables(results: Results) -> list[TableSpec]:
    specs: list[TableSpec] = []
    for pname, slug in PRODUCTS:
        if not _has(results, slug):
            continue
        specs.append(
            TableSpec(
                slug,
                f"{pname}: price (% of notional), expected life (years), P(KI) (knock-in and "
                "no autocall) and P(breach) (the knock-in level breached, autocalled or not).",
                slug,
                _cols_where(results, slug, lambda c: c in FIRST_SCALARS),
                row_header="model",
            )
        )
        specs.append(
            TableSpec(
                f"{slug}_probabilities",
                f"{pname}: the probability of the first autocall at each date and of no "
                "autocall (they add up to 1).",
                slug,
                _cols_where(
                    results,
                    slug,
                    lambda c: c == "p_no_autocall" or c.startswith("p_autocall_"),
                ),
                row_header="model",
            )
        )
        for suffix, keep, what in (
            ("", _is_ac_leg, "the autocall digitals and the bond"),
            ("_put", _is_other_leg, "the coupon and knock-in put legs"),
        ):
            if _has(results, f"legs_{slug}"):
                specs.append(
                    TableSpec(
                        f"legs_{slug}{suffix}",
                        f"{pname}: {what} of decompose(), % of notional (all legs add up to "
                        "the price path by path).",
                        f"legs_{slug}",
                        _cols_where(results, f"legs_{slug}", keep),
                        row_header="model",
                    )
                )
            if _has(results, f"diff_{slug}"):
                specs.append(
                    TableSpec(
                        f"diff_legs_{slug}{suffix}",
                        f"{pname}, LSV minus LV of {what}, % of notional "
                        f"({_pairing(results, f'diff_{slug}')}).",
                        f"diff_{slug}",
                        _cols_where(results, f"diff_{slug}", keep),
                        row_header="model",
                    )
                )
        if _has(results, f"diff_{slug}"):
            specs.append(
                TableSpec(
                    f"diff_{slug}",
                    f"{pname}, LSV minus LV: price (% of notional), life (years), P(KI) and "
                    f"P(breach) ({_pairing(results, f'diff_{slug}')}).",
                    f"diff_{slug}",
                    _cols_where(results, f"diff_{slug}", lambda c: c in FIRST_SCALARS),
                    row_header="model",
                )
            )
            specs.append(
                TableSpec(
                    f"diff_{slug}_probabilities",
                    f"{pname}, LSV minus LV of the first-autocall probabilities and of P(no "
                    f"autocall) ({_pairing(results, f'diff_{slug}')}).",
                    f"diff_{slug}",
                    _cols_where(
                        results,
                        f"diff_{slug}",
                        lambda c: c == "p_no_autocall" or c.startswith("p_autocall_"),
                    ),
                    row_header="model",
                )
            )
        if _has(results, f"se_{slug}"):
            specs.append(
                TableSpec(
                    f"se_{slug}",
                    f"{pname}: the paired stderr of the LSV-minus-LV price and knock-in "
                    "probability against the quadrature error of the same differences.",
                    f"se_{slug}",
                    tuple(
                        Column(c, c.replace("_", " "), digits=2)
                        for c in (
                            "price_paired",
                            "price_quadrature",
                            "p_ki_paired",
                            "p_ki_quadrature",
                        )
                        if c in results.columns(f"se_{slug}")
                    ),
                    row_header="model",
                )
            )
    if _has(results, "horizon"):
        specs.append(
            TableSpec(
                "horizon",
                "The calibration horizon behind each model (years) and 1 when the 3y notes run "
                "beyond it (the last leverage slice held to maturity; the local vol has no "
                "leverage).",
                "horizon",
                (
                    Column("horizon", "horizon", digits=3),
                    Column("notes_beyond", "notes beyond", digits=1),
                ),
                row_header="model",
            )
        )
    if _has(results, "fwd_skew"):
        specs.append(
            TableSpec(
                "fwd_skew",
                "Forward-skew diagnostic: forward 90/110 skew (vol points) of the windows between "
                "the observation dates, its ratio to the surface's spot 1y 90/110 skew, and 1 "
                "when the window ends beyond the calibration horizon.",
                "fwd_skew",
                (
                    Column("fwd_skew", "fwd 90/110 skew", unit="vp"),
                    Column("ratio_to_spot", "fwd / spot"),
                    Column("beyond_horizon", "beyond", digits=1),
                ),
                row_header="model | window",
            )
        )
    if _has(results, "fwd_skew_lv"):
        specs.append(
            TableSpec(
                "fwd_skew_lv",
                "Forward-skew exposure under the local vol, % of notional per vol point of "
                "90/110 skew added at each observation date (tent), per note and leg. Not "
                "like-for-like with the store's skew_T, whose tents sit on the grid's risk "
                "pillars.",
                "fwd_skew_lv",
                tuple(
                    Column(c, c.replace("exposure_", "tent at "), unit="%")
                    for c in results.columns("fwd_skew_lv")
                ),
                row_header="note / leg",
            )
        )
    if _has(results, "skew_store"):
        specs.append(
            TableSpec(
                "skew_store",
                "skew_T of the 3y autocall from the store's risk table (a leverage recalibrated "
                "per bumped surface by the precompute's risk tier), % of notional per vol point. "
                "The tents sit on the grid's risk pillars, not on the observation dates: not "
                "like-for-like with the local-vol exposure.",
                "skew_store",
                tuple(
                    Column(c, c.replace("skew_T_", "pillar "), unit="%")
                    for c in results.columns("skew_store")
                ),
                row_header="model",
            )
        )
    specs.append(
        TableSpec(
            "setup",
            "Sources and budgets.",
            "setup",
            (Column("value", "value", digits=6),),
            row_header="setting",
        )
    )
    return specs


def _draw_diff(results: Results) -> Any:
    fig, axes = style.new_figure(1, 2)
    for ax, (pname, slug) in zip(axes, PRODUCTS):
        t = f"diff_{slug}"
        if not _has(results, t):
            empty_panel(ax, "no local-vol reference in the selection")
            continue
        cols = ["price"] + [c for c in results.columns(t) if c.startswith("leg:")]
        cols = [c for c in cols if c in results.columns(t)]
        xs = np.arange(len(cols), dtype=float)
        rows = results.rows(t)[:8]
        ax.axhline(0.0, color=style.INK["spine"], linewidth=0.8)
        for i, r in enumerate(rows):
            v = np.array([results.value(t, r, c)[0] for c in cols])
            s = np.array([results.value(t, r, c)[1] for c in cols])
            off = 0.8 * (i - (len(rows) - 1) / 2) / max(len(rows), 1)
            style.mc_errorbar(ax, xs + off, v, s, series=i, label=r, line=False)
        ax.set_xticks(xs)
        ax.set_xticklabels([_header(c) for c in cols], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("LSV - LV [% notional]")
        ax.set_title(pname)
        ax.legend(fontsize=7)
    return fig


def _draw_probabilities(results: Results) -> Any:
    fig, axes = style.new_figure(1, 2)
    for ax, (pname, slug) in zip(axes, PRODUCTS):
        if not _has(results, slug):
            empty_panel(ax, f"no {pname} rows")
            continue
        rows = results.rows(slug)
        xs = np.arange(len(rows), dtype=float)
        cols = [c for c in ("p_ki", "p_breach", "p_no_autocall") if c in results.columns(slug)]
        for i, c in enumerate(cols):
            v = np.array([results.value(slug, r, c)[0] for r in rows])
            s = np.array([results.value(slug, r, c)[1] for r in rows])
            style.mc_errorbar(ax, xs, v, s, series=i, label=_header(c))
        ax.set_xticks(xs)
        ax.set_xticklabels(rows, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("probability")
        ax.set_title(pname)
        ax.legend(fontsize=7)
    return fig


def _draw_exposure(results: Results) -> Any:
    fig, axes = style.new_figure(1, 2, sharey=True)
    for ax, (pname, slug) in zip(axes, PRODUCTS):
        if not _has(results, "fwd_skew_lv"):
            empty_panel(ax, "no local-vol exposure in this run")
            continue
        long = results.long("fwd_skew_lv")
        long = long[long["row"].str.startswith(f"{slug} / ")]
        ax.axhline(0.0, color=style.INK["spine"], linewidth=0.8)
        for i, row in enumerate(list(dict.fromkeys(long["row"]))[:8]):
            sub = long[long["row"] == row]
            xs = np.array([float(json.loads(str(a))["T"]) for a in sub["axes"]])
            style.mc_errorbar(
                ax,
                xs,
                sub["value"].to_numpy(float),
                sub["stderr"].to_numpy(float),
                exact=sub["exact"].to_numpy(bool),
                series=i,
                label=row.split(" / ", 1)[1],
            )
        ax.set_xlabel("observation date of the skew tent [y]")
        ax.set_ylabel("% notional per vol pt of 90/110 skew")
        ax.set_title(f"{pname} under local vol")
        ax.legend(fontsize=7)
    return fig


def _draw_fwd_skew(results: Results) -> Any:
    fig, ax = style.new_figure()
    if not _has(results, "fwd_skew"):
        empty_panel(ax, "no forward smile at 0.9 / 1.1 in the results")
        return fig
    long = results.long("fwd_skew", "fwd_skew")
    labels = list(dict.fromkeys(r.split(" | ")[0] for r in long["row"]))
    xs = np.arange(len(labels), dtype=float)
    for i, w in enumerate(list(dict.fromkeys(r.split(" | ")[1] for r in long["row"]))[:8]):
        v = np.full(len(labels), np.nan)
        s = np.full(len(labels), np.nan)
        for j, lab in enumerate(labels):
            hit = long[long["row"] == f"{lab} | {w}"]
            if not hit.empty:
                v[j], s[j] = float(hit["value"].iloc[0]), float(hit["stderr"].iloc[0])
        style.mc_errorbar(ax, xs, v, s, series=i, label=f"forward {w}")
    if _has(results, "spot_skew"):
        spot = results.long("spot_skew")
        ax.axhline(
            float(spot["value"].iloc[0]),
            color=style.INK["spine"],
            linestyle="--",
            linewidth=1.0,
            label=f"spot {spot['row'].iloc[0]} (exact)",
        )
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
    ax.set_ylabel("90/110 skew [vol pts]")
    ax.set_title("Forward skew between the observation dates")
    ax.legend(fontsize=7)
    return fig


def figures(results: Results) -> list[FigureSpec]:
    return [
        FigureSpec(
            "lsv_minus_lv_by_leg",
            "LSV minus LV of the price and of every leg, per model, bars 1 stderr (paired when "
            "both models were priced here, else the quadrature error, not the exact one).",
            _draw_diff,
        ),
        FigureSpec(
            "probabilities",
            "P(KI), P(breach) and P(no autocall) per model, error bars 1 stderr.",
            _draw_probabilities,
        ),
        FigureSpec(
            "fwd_skew_exposure_lv",
            "Forward-skew exposure per leg under the local vol at each observation date, "
            "error bars 1 stderr of the common-random-number difference.",
            _draw_exposure,
        ),
        FigureSpec(
            "forward_skew",
            "Forward 90/110 skew of each model between the observation dates, error bars 1 "
            "stderr (paired when priced here, the sum of the two strike errors when read from the "
            "store); dashed: the surface's spot 1y skew (exact).",
            _draw_fwd_skew,
        ),
    ]


def _horizon_lines(results: Results) -> list[str]:
    """The disclosure of notes priced beyond a leverage's calibration horizon (from the
    results)."""
    if not _has(results, "horizon"):
        return []
    piv = results.pivot("horizon")
    beyond = [r for r in piv.index if piv.loc[r, "notes_beyond"] == 1.0]
    if not beyond:
        return [
            "Every leverage is calibrated to at least the notes' 3y maturity.",
            "",
        ]
    hz = sorted({float(piv.loc[r, "horizon"]) for r in beyond})
    windows = 0
    if _has(results, "fwd_skew"):
        windows = int((results.pivot("fwd_skew")["beyond_horizon"] == 1.0).sum())
    return [
        f"**{len(beyond)} model(s) price the 3y notes beyond their leverage's calibration "
        f"horizon ({', '.join(f'{h:g}y' for h in hz)}): the last leverage slice is held to "
        f"maturity**, so their numbers are plumbing, not model results ({', '.join(beyond)}). "
        f"{windows} forward-skew window(s) likewise end beyond the horizon. Every such number "
        "says so in its note.",
        "",
        "{{table:horizon}}",
        "",
    ]


def _pillars(results: Results, table: str, prefix: str) -> list[float]:
    return sorted(
        float(c[len(prefix) : -1])
        for c in results.columns(table)
        if c.startswith(prefix) and c.endswith("y")
    )


def _neighbours(xs: Sequence[float], x: float) -> tuple[float | None, float | None]:
    i = list(xs).index(x)
    return (xs[i - 1] if i > 0 else None, xs[i + 1] if i + 1 < len(xs) else None)


def _like_for_like(results: Results) -> str:
    """Whether the store's skew_T tents and the local-vol exposure tents are comparable: a
    tent runs from its previous pillar to its next (flat beyond the ends), so two tents match
    only at a common pillar with common neighbours."""
    store = _pillars(results, "skew_store", "skew_T_")
    dates = _pillars(results, "fwd_skew_lv", "exposure_")

    def text(xs: Sequence[float]) -> str:
        return ", ".join(f"{x:g}y" for x in xs) or "none"

    same = [x for x in store if x in dates and _neighbours(store, x) == _neighbours(dates, x)]
    if store == dates:
        return (
            "The store's `skew_T` tents and the local-vol tents above sit on the same pillars "
            f"({text(store)}), so the two exposures are comparable tent by tent."
        )
    return (
        "**The two exposures are not like-for-like**: the store's `skew_T` tents sit on the "
        f"grid's risk pillars ({text(store)}), the local-vol tents above on the notes' "
        f"observation dates ({text(dates)}). A tent runs from the previous pillar to the next, "
        "so two tents match only at a common pillar with common neighbours: "
        f"{text(same)}."
    )


def narrative(results: Results) -> str:
    setup = {r: results.value("setup", r, "value")[0] for r in results.rows("setup")}
    lines = [
        "## Data",
        "",
        f"{int(setup['models'])} models: {int(setup['read from the store'])} read from the "
        f"store's M6 cells (`source = store:<point id>`), {int(setup['priced from the cache'])} "
        "priced here from the cached leverage (the product list of `run_m6_headline`, on one "
        "shared step schedule and seed). Nothing was calibrated. "
        '**The store labels the M6 price and legs "% notional" but stores fractions of '
        "notional**; they are multiplied by 100 here (noted on every such number) and the store "
        "is left unchanged.",
        "",
        *_horizon_lines(results),
        "{{table:autocall}}",
        "",
        "{{table:phoenix}}",
        "",
        "{{table:autocall_probabilities}}",
        "",
        "{{figure:probabilities}}",
        "",
        "## LSV minus LV, by leg",
        "",
    ]
    for pname, slug in PRODUCTS:
        t = f"diff_{slug}"
        if not _has(results, t):
            continue
        piv = results.pivot(t)
        legs = [c for c in results.columns(t) if c.startswith("leg:")]
        for r in piv.index:
            dv, ds = float(piv.loc[r, "price"]), float(piv.loc[r, "price_stderr"])
            big = max(legs, key=lambda c: abs(float(piv.loc[r, c]))) if legs else ""
            line = f"- {pname}, {r}: price {dv:+.3f} ± {ds:.3f} % of notional"
            if big:
                line += (
                    f"; the largest leg move is {_header(big)} "
                    f"{float(piv.loc[r, big]):+.3f} ± {float(piv.loc[r, big + '_stderr']):.3f}"
                )
            if "p_ki" in piv.columns:
                line += (
                    f"; P(KI) {float(piv.loc[r, 'p_ki']):+.4f} ± "
                    f"{float(piv.loc[r, 'p_ki_stderr']):.4f}"
                )
            lines.append(line + ".")
        lines += [
            "",
            f"{{{{table:diff_{slug}}}}}",
            "",
            f"{{{{table:diff_{slug}_probabilities}}}}",
            "",
        ]
        lines += [f"{{{{table:diff_legs_{slug}}}}}", "", f"{{{{table:diff_legs_{slug}_put}}}}", ""]
    for pname, slug in PRODUCTS:
        if _has(results, f"diff_{slug}"):
            lines += [f"{pname}: {_pairing(results, f'diff_{slug}')}.", ""]
        if _has(results, f"se_{slug}"):
            se = results.pivot(f"se_{slug}")
            if {"price_paired", "price_quadrature"} <= set(se.columns):
                r = (se["price_quadrature"] / se["price_paired"]).to_numpy(float)
                lines += [
                    f"The quadrature error of the {pname} price difference is "
                    f"{float(np.nanmin(r)):.2f} to {float(np.nanmax(r)):.2f} times the paired "
                    "one.",
                    "",
                    f"{{{{table:se_{slug}}}}}",
                    "",
                ]
    lines += [
        "{{figure:lsv_minus_lv_by_leg}}",
        "",
        "## Forward skew",
        "",
    ]
    if _has(results, "fwd_skew_lv"):
        lines += [
            "Under the local vol, a vol point of 90/110 skew added at one observation date moves "
            "each leg as below (common random numbers, the local vol rebuilt on the bumped "
            "surface).",
            "",
            "{{table:fwd_skew_lv}}",
            "",
            "{{figure:fwd_skew_exposure_lv}}",
            "",
        ]
    if _has(results, "skew_store"):
        lines += [
            "The LSV exposure needs a leverage recalibrated on every bumped surface, which a "
            "study never does; the store's risk table holds it for "
            f"{len(results.rows('skew_store'))} model(s).",
            "",
            "{{table:skew_store}}",
            "",
        ]
        if _has(results, "fwd_skew_lv"):
            lines += [_like_for_like(results), ""]
    n_missing = int(setup.get("models without stored skew_T", 0.0))
    if n_missing:
        rec = results.record("setup", "models without stored skew_T", "value")
        lines += [
            f"{n_missing} model(s) have no stored `skew_T` (the LSV exposure needs a leverage "
            "recalibrated on every bumped surface, which a study never does). Produce it with:",
            "",
            shell_block(str(rec["note"])),
            "",
            "Its tents would sit on the grid's risk pillars, not on the observation dates of "
            "the local-vol table above: the two are not like-for-like.",
            "",
        ]
    if _has(results, "fwd_skew"):
        lines += [
            "How much forward skew each model carries into the dates where the digitals sit "
            "(the diagnostic):",
            "",
            "{{table:fwd_skew}}",
            "",
            "{{figure:forward_skew}}",
            "",
        ]
    return "\n".join(lines)
