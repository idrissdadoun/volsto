"""S2 — the volatility knock-out put (SPEC §6.2 / §10.2, owner's M10 Part 2): the ``vol_ko``
sweep × models, the VKO / vanilla discount, ``P(KO)``, and the distribution of the full-life
realised vol **conditional on** ``S_T < K`` — its 10 / 50 / 90 percentiles and
``P(σ_real > vol_ko | ITM)``, the quantity that sets the sign of the LSV-minus-LV difference —
with the 2022 H2 window as the realised-outcome case.

**Data.**  The store holds the headline VKO only at the 12m 100% term sheet (the ratio sweep,
the 30% price and ``P(KO)``) and none of the ITM-conditional distribution (no stderr, dropped by
the precompute).  So every model is priced here, one path set per model on the study's shared
step schedule and seed (:func:`volsto.studies.catalogue._common.price_paired`;
:class:`~volsto.products.vko.VolKnockOutPut` legs ``put``, ``rv``, ``itm``, annualised with the
term sheet's ``per_year``): the LSV from its cached leverage (``source = cache:<key>``), the
local vol from a Dupire build of the grid's surface (``computed``).  With ``store_check`` the
inline sweep is compared with the store's where the store has the same term sheet: a
bitwise-equal number is an exact zero difference (the same computation), any other difference
carries the quadrature error (not the exact error: the two estimates share random numbers and
their correlation is not measured).  The local vol is stepped on the reference LSV's grid (its
leverage-slice times added), so its store-check rows are not the store's computation: the
``pairing`` table gives both step counts.

**Statistics** (antithetic pairs averaged before every error):

* ratio to the vanilla put — delta method, the :func:`~volsto.analytics.conditional_variance.
  vko_analysis` estimator; prices in % of the spot notional; ``P(KO) = P(σ_real ≥ vol_ko)``;
* ``P(ITM)`` and ``P(σ_real > h | ITM)`` — ratio of means with the delta-method error;
* the ITM-conditional percentiles of ``σ_real`` — the lower empirical quantile (the smallest
  value whose cumulative ITM count reaches ``q``), with a **pair bootstrap** stderr: ``replicates``
  resamples of the antithetic pairs, replicate ``b`` drawn with ``default_rng([seeds.bootstrap,
  b])`` — the same resampling for every model, so quantile differences are paired;
* LSV minus LV — **paired** (the same seed and step schedule; the stderr of the per-pair
  difference of the influence functions; the quadrature error and the per-pair correlation are
  reported beside it, each with its own delta-method stderr, and every ``|d| / se`` is a z-score
  with the nominal stderr 1);
* the mechanism table scores two readings of the sign of the ratio difference ``d``, per LSV
  model and barrier, where ``d`` exceeds 2 stderr:

  1. ``−sign(dP)`` with ``dP`` the LSV-minus-LV difference of ``P(σ_real > vol_ko | ITM)``
     (more in-the-money paths above the barrier are knocked out, so the ratio falls), used
     where ``|dP|`` exceeds 2 stderr;
  2. the superseded reading of SPEC §6.2 (replaced by reading 1 on 2026-09-17)
     ``sign(d width) × sign(p50_LV − vol_ko)`` with ``width = p90 − p10`` (a wider
     in-the-money distribution saves paths below its bulk and knocks more out above it), used
     where both factors exceed 2 stderr.

**Realised outcome** (``realised``): the daily closes of the 2022 H2 surface history
(``ln_spot`` of ``essvi_gate/hdn_history_repaired.csv``, an artefact under the outputs root) from
``start``; the 12m trade struck on ``start`` matures after the data, so the **payoff is not
observable**: the table reports the returns observed, the realised vol to date, the share of the
variance budget ``vol_ko² N / A`` used, the knock-out state at the window end (a knock-out is
certain once the budget is spent), the largest vol over the remaining life that keeps the put
alive, the spot move, and the 12m ATM and variance-swap vols marked on ``start`` — all exact
(data, no Monte Carlo).

**Params** (all required)::

    models: {surface, lv, one_factor, two_factor}
    vko: {strike: 1.0, maturity: 1.0, per_year: 252, vol_ko: [0.20, 0.25, 0.30, 0.35, 0.40]}
    n_paths: 400000
    bootstrap: {replicates: 200}
    store_check: true
    realised: {history: essvi_gate/hdn_history_repaired.csv, command: "...", start: 2022-07-01,
               pillar: 1.0} | null

Seeds: ``pricing``, ``bootstrap``.  Checked by ``tests/test_catalogue_s1_s4.py``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.analytics.conditional_variance import pair_average
from volsto.config import ConfigError, SimConfig
from volsto.models.base import Model
from volsto.products.base import daily_schedule
from volsto.products.vko import VolKnockOutPut
from volsto.studies import style
from volsto.studies.catalogue._common import (
    ERROR_STAT_NOTE,
    LV_GRID_NOTE,
    LV_LABEL,
    PAIRED_NOTE,
    QUADRATURE_NOTE,
    VP,
    Estimate,
    FloatArray,
    ModelPoint,
    add_z,
    axis,
    empty_panel,
    leverage_requirements,
    load_model,
    lv_grid_sentence,
    mean_estimate,
    model_source,
    own_steps,
    paired_difference,
    paired_grid,
    price_paired,
    pricing_sim,
    quadrature_comparison,
    ratio_estimate,
    rss,
    same_times,
    select_models,
    series_by,
    store_source,
    stored_ids,
    validate_selection,
)
from volsto.studies.latex import format_value_text
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

TITLE = "S2 - The volatility knock-out put"
QUESTION = (
    "How much cheaper than the vanilla is the vol knock-out put across barriers and models, and "
    "where does each barrier sit in the in-the-money realised-vol distribution that sets the "
    "sign of the LSV-minus-LV difference?"
)
REQUIRED_PARAMS = ("models", "vko", "n_paths", "bootstrap", "store_check", "realised")
OPTIONAL_PARAMS: tuple[str, ...] = ()
#: What every exact row of this study is (``_common.unclassified_exact_rows``; the walking test
#: fails on any other exact row): ``(table regex, column regex, kind)``.
EXACT_KINDS: tuple[tuple[str, str, str], ...] = (
    ("itm_rv", "n_itm", "count"),
    ("mechanism", "predicted_[pw]|agrees_[pw]|significant", "flag"),
    ("pairing", "steps|own_steps", "count"),
    ("pairing", "same_grid", "flag"),
    # an identical inline and stored number: the difference and its z are exactly 0
    ("store_check_(ratio|price|p_ko)", "diff|n_se", "closed form"),
    ("realised", "knocked_out", "flag"),
    ("realised", "budget_used|max_remaining_vol", "closed form"),  # of the price history
    ("realised_path", "rv_to_date|spot", "closed form"),
    ("realised_summary", "count", "count"),
    ("realised_summary", "date", "input"),
    ("realised_summary", "percent|vol", "closed form"),
    ("setup", "value", "input"),  # the selection's counts and the config's inputs
)

VOL = "vol pts"
PCT = "% notional"
QUANTILES: tuple[tuple[str, float], ...] = (("p10", 0.10), ("p50", 0.50), ("p90", 0.90))
#: The store's VKO term sheet (the M4c headline): 100% strike, 12m, daily fixings.
STORE_TERMS: tuple[float, float, int] = (1.0, 1.0, 252)
#: Significance of an LSV-minus-LV difference in the mechanism table (stderr multiples).
SIGNIFICANCE = 2.0


def htag(h: float) -> str:
    return f"{round(100 * h):d}"


# --------------------------------------------------------------------------------------------
# params and requirements
# --------------------------------------------------------------------------------------------


def validate_params(params: Mapping[str, Any]) -> None:
    validate_selection(params["models"])
    v = params["vko"]
    if not isinstance(v, Mapping) or set(v) != {"strike", "maturity", "per_year", "vol_ko"}:
        raise ConfigError("vko: expected {strike, maturity, per_year, vol_ko}")
    if float(v["strike"]) <= 0 or float(v["maturity"]) <= 0 or int(v["per_year"]) <= 0:
        raise ConfigError("vko: strike, maturity and per_year must be positive")
    hs = v["vol_ko"]
    if not isinstance(hs, list) or not hs or any(float(h) <= 0 for h in hs):
        raise ConfigError("vko.vol_ko: a non-empty list of positive barriers")
    if len({htag(float(h)) for h in hs}) != len(hs):
        raise ConfigError("vko.vol_ko: barriers must differ at the percent level")
    n = int(params["n_paths"])
    if n <= 0 or n % 2:
        raise ConfigError("n_paths: a positive even number (antithetic pairs)")
    bs = params["bootstrap"]
    if not isinstance(bs, Mapping) or set(bs) != {"replicates"} or int(bs["replicates"]) < 2:
        raise ConfigError("bootstrap: expected {replicates >= 2}")
    if not isinstance(params["store_check"], bool):
        raise ConfigError("store_check: expected true or false")
    r = params["realised"]
    if r is not None and (
        not isinstance(r, Mapping) or set(r) != {"history", "command", "start", "pillar"}
    ):
        raise ConfigError("realised: expected null or {history, command, start, pillar}")


def requirements(ctx: StudyContext) -> list[Requirement]:
    """The LSV leverages and, with ``realised``, the surface history artefact."""
    models = select_models(ctx, ctx.params["models"])
    ctx.seed("pricing")
    ctx.seed("bootstrap")
    reqs = leverage_requirements(ctx, models)
    r = ctx.params["realised"]
    if r is not None:
        reqs.append(
            ctx.artefact_requirement(
                str(r["history"]), "2022 H2 surface history (daily ln_spot)", str(r["command"])
            )
        )
    return reqs


# --------------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VKOStats:
    """Every VKO statistic of one model as an :class:`Estimate` (prices in % of notional);
    ``quantiles`` the ITM-conditional point percentiles (vol units) and ``replicates`` their
    pair-bootstrap replicates ``(B, 3)`` (p10, p50, p90), drawn with the same resampling for
    every model so that differences are paired."""

    vanilla: Estimate
    ratio: dict[float, Estimate]
    price: dict[float, Estimate]
    p_ko: dict[float, Estimate]
    p_gt: dict[float, Estimate]
    p_itm: Estimate
    quantiles: dict[str, float]
    replicates: FloatArray
    n_itm: int
    same_grid: bool
    steps: int
    own_steps: int

    def quantile(self, key: str) -> tuple[float, float]:
        j = [k for k, _ in QUANTILES].index(key)
        return self.quantiles[key], _std(self.replicates[:, j])

    @property
    def width(self) -> tuple[float, float]:
        return (
            self.quantiles["p90"] - self.quantiles["p10"],
            _std(self.replicates[:, 2] - self.replicates[:, 0]),
        )


def _std(x: FloatArray) -> float:
    x = x[np.isfinite(x)]
    return float(x.std(ddof=1)) if x.size > 1 else math.nan


def lower_quantile_index(cum: FloatArray, q: float) -> int:
    """The first index whose cumulative weight reaches ``q`` of the total."""
    return int(np.searchsorted(cum, q * cum[-1], side="left"))


def itm_quantiles(
    sigma: FloatArray,
    itm: NDArray[np.bool_],
    antithetic: bool,
    seed: int,
    replicates: int,
) -> tuple[dict[str, float], FloatArray]:
    """The ITM-conditional 10 / 50 / 90 percentiles of ``sigma`` (the lower empirical quantile)
    and their pair-bootstrap replicates ``(replicates, 3)``: replicate ``b`` resamples the
    antithetic pairs with ``numpy.random.default_rng([seed, b])``, the same resampling for any
    path set of the same size (module docstring)."""
    n = sigma.size
    pair_of = np.arange(n) // 2 if antithetic else np.arange(n)
    n_pairs = int(pair_of[-1]) + 1
    sel = np.flatnonzero(itm)
    reps = np.full((replicates, len(QUANTILES)), math.nan)
    if sel.size == 0:
        return {k: math.nan for k, _ in QUANTILES}, reps
    order = np.argsort(sigma[sel], kind="stable")
    s_sorted = sigma[sel][order]
    p_sorted = pair_of[sel][order]
    ones = np.arange(1, s_sorted.size + 1, dtype=np.float64)
    point = {k: float(s_sorted[lower_quantile_index(ones, q)]) for k, q in QUANTILES}
    for b in range(replicates):
        rng = np.random.default_rng([seed, b])
        counts = np.bincount(rng.integers(0, n_pairs, n_pairs), minlength=n_pairs)
        cum = np.cumsum(counts[p_sorted], dtype=np.float64)
        if cum[-1] <= 0:
            continue
        for j, (_, q) in enumerate(QUANTILES):
            reps[b, j] = s_sorted[min(lower_quantile_index(cum, q), s_sorted.size - 1)]
    return point, reps


def vko_statistics(
    model: Model,
    sim: SimConfig,
    terms: Mapping[str, Any],
    seed: int,
    replicates: int,
    reference_times: FloatArray,
) -> VKOStats:
    """One path set of the shared step schedule, every statistic of the module docstring."""
    spot = float(model.spot)
    T = float(terms["maturity"])
    per_year = int(terms["per_year"])
    barriers = [float(h) for h in terms["vol_ko"]]
    discount = model.forward_curve.rate_curve
    times = daily_schedule(T, per_year)
    vko = VolKnockOutPut(
        float(terms["strike"]) * spot,
        T,
        barriers[0],
        times,
        discount,
        notional=1.0 / spot,
        annualisation=float(per_year),
    )
    legs = [vko.leg("put"), vko.leg("rv"), vko.leg("itm")]
    res = price_paired(model, sim, legs, reference_times)
    steps = int(paired_grid(model, sim, legs, reference_times).n_steps)
    put_raw = np.asarray(res[0].payoffs, dtype=np.float64)
    rv_raw = np.asarray(res[1].payoffs, dtype=np.float64)
    itm_raw = np.asarray(res[2].payoffs, dtype=np.float64) > 0.5
    anti = sim.antithetic
    scale = 100.0 * vko.notional * float(discount.df(T))
    put = pair_average(put_raw, anti)
    itm_p = pair_average(itm_raw.astype(np.float64), anti)
    ratio, price, p_ko, p_gt = {}, {}, {}, {}
    for h in barriers:
        alive = rv_raw < h * h
        pa = pair_average(put_raw * alive, anti)
        price[h] = mean_estimate(pa).scaled(scale)
        ratio[h] = ratio_estimate(pa, put)
        p_ko[h] = mean_estimate(pair_average((~alive).astype(np.float64), anti))
        above = pair_average(((rv_raw > h * h) & itm_raw).astype(np.float64), anti)
        p_gt[h] = ratio_estimate(above, itm_p)
    point, reps = itm_quantiles(np.sqrt(rv_raw), itm_raw, anti, seed, replicates)
    return VKOStats(
        mean_estimate(put).scaled(scale),
        ratio,
        price,
        p_ko,
        p_gt,
        mean_estimate(itm_p),
        point,
        reps,
        int(itm_raw.sum()),
        same_times(model, reference_times),
        steps,
        own_steps(model, sim, legs),
    )


# --------------------------------------------------------------------------------------------
# realised outcome
# --------------------------------------------------------------------------------------------


def realised_outcome(
    history: pd.DataFrame, start: str, pillar: float, terms: Mapping[str, Any]
) -> dict[str, Any]:
    """The realised-outcome figures of the module docstring from a surface history frame
    (``date, T, vs_vol, atm_vol, ..., ln_spot``)."""
    need = {"date", "T", "atm_vol", "vs_vol", "ln_spot"}
    if not need <= set(history.columns):
        raise ConfigError(f"realised history: columns {sorted(need - set(history.columns))} absent")
    h = history.copy()
    h["date"] = pd.to_datetime(h["date"])
    start_ts = pd.Timestamp(str(start))
    spots = h.groupby("date")["ln_spot"].first().sort_index()
    spots = spots[spots.index >= start_ts]
    if spots.size < 2:
        raise ConfigError(f"realised history: fewer than two closes from {start}")
    first = spots.index[0]
    marks = h[(h["date"] == first) & np.isclose(h["T"].astype(float), float(pillar))]
    if marks.empty:
        raise ConfigError(f"realised history: no {pillar:g}y pillar on {first.date()}")
    ls = spots.to_numpy(dtype=np.float64)
    r = np.diff(ls)
    per_year = int(terms["per_year"])
    T = float(terms["maturity"])
    n_life = len(daily_schedule(T, per_year)) - 1
    annualisation = float(per_year)
    cum = np.cumsum(r * r)
    days = np.arange(1, r.size + 1)
    return {
        "first": first,
        "last": spots.index[-1],
        "maturity": first + pd.DateOffset(days=round(365 * T)),
        "n_obs": int(r.size),
        "n_life": n_life,
        "annualisation": annualisation,
        "sum_sq": float(cum[-1]),
        "rv_to_date": float(math.sqrt(annualisation * cum[-1] / r.size)),
        "spot_return": float(math.exp(ls[-1] - ls[0]) - 1.0),
        "atm_mark": float(marks["atm_vol"].iloc[0]),
        "vs_mark": float(marks["vs_vol"].iloc[0]),
        "path_dates": [d.strftime("%Y-%m-%d") for d in spots.index[1:]],
        "path_rv": np.sqrt(annualisation * cum / days),
        "path_spot": np.exp(ls[1:] - ls[0]),
        "path_cum": cum,
    }


def _date_number(ts: pd.Timestamp) -> float:
    return float(ts.year * 10000 + ts.month * 100 + ts.day)


# --------------------------------------------------------------------------------------------
# compute
# --------------------------------------------------------------------------------------------


def compute(ctx: StudyContext) -> Results:
    p = ctx.params
    models = select_models(ctx, p["models"])
    terms = p["vko"]
    barriers = [float(h) for h in terms["vol_ko"]]
    n_paths = int(p["n_paths"])
    replicates = int(p["bootstrap"]["replicates"])
    seed = ctx.seed("bootstrap")
    lsvs = [mp for mp in models if not mp.is_lv]
    lv = next((mp for mp in models if mp.is_lv), None)
    loaded: dict[str, Model] = {}
    ref_times: FloatArray = np.empty(0)
    if lsvs:
        loaded[lsvs[0].label] = load_model(ctx, lsvs[0])
        ref_times = np.asarray(loaded[lsvs[0].label].required_times(), dtype=np.float64)
    b = ResultsBuilder()
    ref: VKOStats | None = None
    for mp in ([lv] if lv is not None else []) + lsvs:
        model = loaded.pop(mp.label, None) or load_model(ctx, mp)
        sim = pricing_sim(ctx, mp, n_paths)
        ctx.log.info("%s: VKO sweep at %d paths", mp.label, n_paths)
        st = vko_statistics(model, sim, terms, seed, replicates, ref_times)
        src = model_source(mp)
        _model_tables(b, mp, st, src, barriers)
        _pairing_rows(b, mp, st, src)
        if p["store_check"]:
            _store_check(ctx, b, mp, st, src, terms)
        if mp.is_lv:
            ref = st
        elif ref is not None and lv is not None:
            _lsv_minus_lv(b, mp, st, ref, f"{src};{model_source(lv)}", barriers)
    ctx.record("n_paths", n_paths)
    ctx.record("bootstrap_replicates", replicates)
    r = p["realised"]
    if r is not None:
        path = ctx.artefact(str(r["history"]))
        out = realised_outcome(pd.read_csv(path), str(r["start"]), float(r["pillar"]), terms)
        _realised_tables(b, out, barriers, f"artefact:{r['history']}")
    for row, val in (
        ("models", float(len(models))),
        ("paths per model", float(n_paths)),
        ("bootstrap replicates", float(replicates)),
        ("strike (fraction of spot)", float(terms["strike"])),
        ("maturity [y]", float(terms["maturity"])),
        ("fixings per year", float(terms["per_year"])),
        ("significance [stderr]", SIGNIFICANCE),
    ):
        b.add_exact("setup", row, "value", val, unit="", source="computed")
    return b.build()


def _model_tables(
    b: ResultsBuilder, mp: ModelPoint, st: VKOStats, src: str, barriers: Sequence[float]
) -> None:
    ax = mp.axes

    def add(table: str, col: str, e: Estimate, unit: str, **extra: float) -> None:
        b.add(table, mp.label, col, e.value, e.stderr, unit=unit, source=src, axes={**ax, **extra})

    add("vko_price", "vanilla", st.vanilla, PCT)
    for h in barriers:
        t = htag(h)
        add("vko_ratio", f"ratio_{t}", st.ratio[h], DIMENSIONLESS, vol_ko=h)
        add("vko_price", f"price_{t}", st.price[h], PCT, vol_ko=h)
        add("p_ko", f"p_ko_{t}", st.p_ko[h], DIMENSIONLESS, vol_ko=h)
        add("itm_rv", f"p_gt_{t}", st.p_gt[h], DIMENSIONLESS, vol_ko=h)
    add("itm_rv", "p_itm", st.p_itm, DIMENSIONLESS)
    for k, _ in QUANTILES:
        v, s = st.quantile(k)
        b.add(
            "itm_rv",
            mp.label,
            k,
            VP * v,
            VP * s,
            unit=VOL,
            source=src,
            note="lower empirical quantile; stderr: pair bootstrap",
            axes=ax,
        )
    w, ws = st.width
    b.add(
        "itm_rv",
        mp.label,
        "width",
        VP * w,
        VP * ws,
        unit=VOL,
        source=src,
        note="p90 - p10; stderr: pair bootstrap",
        axes=ax,
    )
    b.add_exact("itm_rv", mp.label, "n_itm", float(st.n_itm), unit="", source=src, axes=ax)


def _pairing_rows(b: ResultsBuilder, mp: ModelPoint, st: VKOStats, src: str) -> None:
    note = LV_GRID_NOTE if mp.is_lv else ""
    for col, val, what in (
        ("steps", float(st.steps), "steps to the maturity on the study's shared grid"),
        ("own_steps", float(st.own_steps), "steps on the grid the model would use alone"),
        ("same_grid", float(st.same_grid), "1: the model's grid is the shared one"),
    ):
        b.add_exact(
            "pairing",
            mp.label,
            col,
            val,
            unit="",
            source=src,
            note="; ".join(x for x in (what, note) if x),
            axes=mp.axes,
        )


def _sign_if(value: float, stderr: float) -> float:
    """``sign(value)`` when ``|value|`` exceeds :data:`SIGNIFICANCE` stderr, else 0."""
    if not (math.isfinite(value) and math.isfinite(stderr)):
        return 0.0
    return float(np.sign(value)) if abs(value) > SIGNIFICANCE * stderr else 0.0


def _lsv_minus_lv(
    b: ResultsBuilder,
    mp: ModelPoint,
    st: VKOStats,
    ref: VKOStats,
    src: str,
    barriers: Sequence[float],
) -> None:
    same = st.same_grid and ref.same_grid
    grid_note = "" if same else "; the two grids differ (still paired by path)"
    # paired bootstrap: the same resampling of the pairs for both models
    w_st = st.replicates[:, 2] - st.replicates[:, 0]
    w_ref = ref.replicates[:, 2] - ref.replicates[:, 0]
    wd = st.width[0] - ref.width[0]
    wd_se = _std(w_st - w_ref)
    p50_lv, p50_lv_se = ref.quantile("p50")
    for k, _ in QUANTILES:
        j = [q for q, _ in QUANTILES].index(k)
        dq = st.quantiles[k] - ref.quantiles[k]
        dq_se = _std(st.replicates[:, j] - ref.replicates[:, j])
        b.add(
            "lsv_minus_lv_quantiles",
            mp.label,
            k,
            VP * dq,
            VP * dq_se,
            unit=VOL,
            source=src,
            note="LSV minus LV; paired pair bootstrap" + grid_note,
            axes=mp.axes,
        )
    b.add(
        "lsv_minus_lv_quantiles",
        mp.label,
        "width",
        VP * wd,
        VP * wd_se,
        unit=VOL,
        source=src,
        note="LSV minus LV; paired pair bootstrap" + grid_note,
        axes=mp.axes,
    )
    for h in barriers:
        t = htag(h)
        hax = {**mp.axes, "vol_ko": h, "same_grid": float(same)}
        d = paired_difference(st.ratio[h], ref.ratio[h], same_grid=same)
        dk = paired_difference(st.p_gt[h], ref.p_gt[h], same_grid=same)
        note = PAIRED_NOTE + grid_note
        b.add(
            "lsv_minus_lv",
            mp.label,
            f"ratio_{t}",
            d.value,
            d.stderr,
            unit=DIMENSIONLESS,
            source=src,
            note=note,
            axes=hax,
        )
        for col, value, se, what in (
            (f"se_paired_{t}", d.stderr, d.stderr_se, "paired stderr of the ratio difference"),
            (
                f"se_quadrature_{t}",
                d.stderr_quadrature,
                d.quadrature_se,
                "the quadrature error of the same difference, for comparison",
            ),
            (f"correlation_{t}", d.correlation, d.correlation_se, "per-pair correlation"),
        ):
            b.add(
                "lsv_minus_lv_se",
                mp.label,
                col,
                value,
                se if math.isfinite(value) else math.nan,
                unit=DIMENSIONLESS,
                source=src,
                note=f"{what}; {ERROR_STAT_NOTE}",
                axes=hax,
            )
        b.add(
            "lsv_minus_lv_itm",
            mp.label,
            f"p_gt_{t}",
            dk.value,
            dk.stderr,
            unit=DIMENSIONLESS,
            source=src,
            note=note,
            axes=hax,
        )
        row = f"{mp.label} | {t}%"
        b.add(
            "mechanism",
            row,
            "ratio_diff",
            d.value,
            d.stderr,
            unit=DIMENSIONLESS,
            source=src,
            note=note,
            axes=hax,
        )
        b.add(
            "mechanism",
            row,
            "p_gt_diff",
            dk.value,
            dk.stderr,
            unit=DIMENSIONLESS,
            source=src,
            note="P(sigma_real > vol_ko | ITM), LSV minus LV; " + note,
            axes=hax,
        )
        b.add(
            "mechanism",
            row,
            "width_diff",
            VP * wd,
            VP * wd_se,
            unit=VOL,
            source=src,
            note="(p90 - p10) minus the LV's; paired pair bootstrap",
            axes=hax,
        )
        b.add(
            "mechanism",
            row,
            "p50_lv_minus_h",
            VP * (p50_lv - h),
            VP * p50_lv_se,
            unit=VOL,
            source=src.split(";")[-1],
            note="LV ITM median minus the barrier; bootstrap",
            axes=hax,
        )
        significant = abs(d.value) > SIGNIFICANCE * d.stderr
        measured = float(np.sign(d.value)) if significant else 0.0
        # reading 1: more in-the-money paths above the barrier -> more knocked out -> lower ratio
        pred_p = -_sign_if(dk.value, dk.stderr)
        # reading 2 (superseded SPEC 6.2): a wider ITM distribution saves paths below its bulk and
        # knocks more of them out above it
        pred_w = _sign_if(wd, wd_se) * _sign_if(p50_lv - h, p50_lv_se)
        for col, val, what in (
            ("significant", float(significant), f"|ratio_diff| > {SIGNIFICANCE:g} stderr"),
            ("predicted_p", pred_p, "-sign(dP) when |dP| > 2 stderr, else 0"),
            (
                "agrees_p",
                float(measured == pred_p) if measured and pred_p else math.nan,
                "1: the measured sign is reading 1's (-- when either is not resolved)",
            ),
            (
                "predicted_w",
                pred_w,
                "sign(d width) x sign(p50_LV - vol_ko), each beyond 2 stderr, else 0",
            ),
            (
                "agrees_w",
                float(measured == pred_w) if measured and pred_w else math.nan,
                "1: the measured sign is reading 2's (-- when either is not resolved)",
            ),
        ):
            b.add_exact("mechanism", row, col, val, unit="", source="computed", note=what, axes=hax)


def _store_check(
    ctx: StudyContext,
    b: ResultsBuilder,
    mp: ModelPoint,
    st: VKOStats,
    src: str,
    terms: Mapping[str, Any],
) -> None:
    same_terms = (
        math.isclose(float(terms["strike"]), STORE_TERMS[0])
        and math.isclose(float(terms["maturity"]), STORE_TERMS[1])
        and int(terms["per_year"]) == STORE_TERMS[2]
    )
    if not same_terms or mp.id not in stored_ids(ctx):
        return
    prod = ctx.store().products(mp.id)
    vals = {str(r["key"]): (float(r["value"]), float(r["value_stderr"])) for r in records(prod)}
    for h in st.ratio:
        t = htag(h)
        checks = (
            ("store_check_ratio", f"vko_ratio_{t}", st.ratio[h], DIMENSIONLESS),
            ("store_check_price", f"vko_{t}", st.price[h], PCT),
            ("store_check_p_ko", f"vko_{t}_p_ko", st.p_ko[h], DIMENSIONLESS),
        )
        for table, key, e, unit in checks:
            if key not in vals:
                continue
            v, s = e.value, e.stderr
            sv, ss = vals[key]
            row = f"{mp.label} | {t}%"
            ax = {**mp.axes, "vol_ko": h, "key": key}
            both = f"{src};{store_source(mp.id)}"
            grid_note = (
                f"{LV_GRID_NOTE} ({st.steps} steps against {st.own_steps}): not the store's "
                "computation"
                if mp.is_lv and st.steps != st.own_steps
                else ""
            )
            b.add(table, row, "inline", v, s, unit=unit, source=src, note=grid_note, axes=ax)
            b.add(table, row, "store", sv, ss, unit=unit, source=store_source(mp.id), axes=ax)
            if v == sv:
                b.add_exact(
                    table,
                    row,
                    "diff",
                    0.0,
                    unit=unit,
                    source=both,
                    note=f"identical to the store's {key} (same computation: same "
                    "seed, same steps to the maturity)",
                    axes=ax,
                )
                b.add_exact(
                    table,
                    row,
                    "n_se",
                    0.0,
                    unit=DIMENSIONLESS,
                    source="computed",
                    note="identical",
                    axes=ax,
                )
                continue
            d_se = rss(s, ss)
            b.add(
                table,
                row,
                "diff",
                v - sv,
                d_se,
                unit=unit,
                source=both,
                note=f"inline minus store {key}; "
                + QUADRATURE_NOTE
                + (f"; {grid_note}" if grid_note else ""),
                axes=ax,
            )
            add_z(
                b,
                table,
                row,
                "n_se",
                abs(v - sv) / d_se if d_se > 0 else math.nan,
                source="computed",
                note="|d| over the quadrature error (not an exact significance)",
                axes=ax,
            )


def _realised_tables(
    b: ResultsBuilder, out: Mapping[str, Any], barriers: Sequence[float], src: str
) -> None:
    budget_note = "sum of squared daily log returns over the budget vol_ko^2 N / A"
    n_obs, n_life, A = int(out["n_obs"]), int(out["n_life"]), float(out["annualisation"])
    for h in barriers:
        row = f"vol_ko {htag(h)}%"
        budget = h * h * n_life / A
        used = float(out["sum_sq"]) / budget
        ko = float(out["sum_sq"]) >= budget
        remaining = (
            math.sqrt(A * (budget - float(out["sum_sq"])) / (n_life - n_obs))
            if not ko and n_life > n_obs
            else math.nan
        )
        ax = {"vol_ko": h}
        b.add_exact(
            "realised",
            row,
            "budget_used",
            100.0 * used,
            unit="%",
            source=src,
            note=budget_note,
            axes=ax,
        )
        b.add_exact(
            "realised",
            row,
            "knocked_out",
            float(ko),
            unit="",
            source=src,
            note="1: the knock-out is certain at the window end",
            axes=ax,
        )
        b.add_exact(
            "realised",
            row,
            "max_remaining_vol",
            VP * remaining,
            unit=VOL,
            source=src,
            note="largest constant vol over the remaining life that keeps the put alive "
            "(-- once knocked out)",
            axes=ax,
        )
    first, last, mat = out["first"], out["last"], out["maturity"]
    summary = (
        ("start date", "date", _date_number(first), "", "the trade is struck on this close"),
        ("last close", "date", _date_number(last), "", "the history ends here"),
        (
            "maturity date",
            "date",
            _date_number(mat),
            "",
            "after the data: the payoff is not observable",
        ),
        ("returns observed", "count", float(n_obs), "", f"of {n_life} over the life"),
        ("realised vol to date", "vol", VP * float(out["rv_to_date"]), VOL, "sqrt(A/n sum r^2)"),
        (
            "spot move to date",
            "percent",
            100.0 * float(out["spot_return"]),
            "%",
            "S_last / S_start - 1",
        ),
        ("marked ATM vol", "vol", VP * float(out["atm_mark"]), VOL, "fitted surface, start"),
        ("marked VS vol", "vol", VP * float(out["vs_mark"]), VOL, "fitted surface, start"),
    )
    for row, col, val, unit, note in summary:
        b.add_exact("realised_summary", row, col, val, unit=unit, source=src, note=note)
    for i, (d, rv, sp) in enumerate(
        zip(out["path_dates"], out["path_rv"], out["path_spot"], strict=True)
    ):
        ax = {"day": float(i + 1), "date": d}
        b.add_exact("realised_path", d, "rv_to_date", VP * float(rv), unit=VOL, source=src, axes=ax)
        b.add_exact(
            "realised_path",
            d,
            "spot",
            100.0 * float(sp),
            unit="%",
            source=src,
            note="close over the start close",
            axes=ax,
        )


# --------------------------------------------------------------------------------------------
# tables, figures, narrative (results only)
# --------------------------------------------------------------------------------------------


def _has(results: Results, table: str) -> bool:
    return table in results.tables()


def _barrier_cols(results: Results, table: str, prefix: str) -> list[str]:
    return [c for c in results.columns(table) if c.startswith(prefix)]


def _pct(col: str) -> str:
    return col.rsplit("_", 1)[1] + "%"


def tables(results: Results) -> list[TableSpec]:
    specs = [
        TableSpec(
            "vko_ratio",
            "VKO / vanilla put ratio per vol_ko barrier (the VKO discount is 1 minus the ratio).",
            "vko_ratio",
            tuple(Column(c, f"@{_pct(c)}") for c in _barrier_cols(results, "vko_ratio", "ratio_")),
            row_header="model",
        ),
        TableSpec(
            "vko_price",
            "Vanilla put and VKO prices per barrier, % of the spot notional.",
            "vko_price",
            (
                Column("vanilla", "vanilla", unit="%"),
                *(
                    Column(c, f"@{_pct(c)}", unit="%")
                    for c in _barrier_cols(results, "vko_price", "price_")
                ),
            ),
            row_header="model",
        ),
        TableSpec(
            "p_ko",
            "P(KO) = P(realised vol over the life >= vol_ko) per barrier.",
            "p_ko",
            tuple(Column(c, f"@{_pct(c)}") for c in _barrier_cols(results, "p_ko", "p_ko_")),
            row_header="model",
        ),
        TableSpec(
            "itm_quantiles",
            "Full-life realised vol conditional on S_T < K: P(ITM), the 10/50/90 percentiles "
            "and their width p90 - p10, vol points (pair-bootstrap stderr).",
            "itm_rv",
            (
                Column("p_itm", "P(ITM)"),
                Column("p10", "p10", unit="vp"),
                Column("p50", "p50", unit="vp"),
                Column("p90", "p90", unit="vp"),
                Column("width", "width", unit="vp"),
            ),
            row_header="model",
        ),
        TableSpec(
            "itm_p_gt",
            "P(realised vol > vol_ko | S_T < K) per barrier: the knock-out probability of the "
            "in-the-money paths.",
            "itm_rv",
            tuple(Column(c, f"@{_pct(c)}") for c in _barrier_cols(results, "itm_rv", "p_gt_")),
            row_header="model",
        ),
    ]
    if _has(results, "lsv_minus_lv"):
        specs.append(
            TableSpec(
                "lsv_minus_lv",
                "LSV minus LV of the VKO / vanilla ratio per barrier (paired stderr).",
                "lsv_minus_lv",
                tuple(
                    Column(c, f"@{_pct(c)}")
                    for c in _barrier_cols(results, "lsv_minus_lv", "ratio_")
                ),
                row_header="model",
            )
        )
        specs.append(
            TableSpec(
                "lsv_minus_lv_quantiles",
                "LSV minus LV of the ITM-conditional percentiles and their width, vol points "
                "(paired pair bootstrap).",
                "lsv_minus_lv_quantiles",
                tuple(Column(c, c, unit="vp") for c in results.columns("lsv_minus_lv_quantiles")),
                row_header="model",
            )
        )
        specs.append(
            TableSpec(
                "lsv_minus_lv_se",
                "The same ratio differences: paired stderr against the quadrature error, and "
                "the per-pair correlation, per barrier.",
                "lsv_minus_lv_se",
                tuple(
                    Column(
                        c,
                        c.replace("se_paired_", "paired ")
                        .replace("se_quadrature_", "quadr. ")
                        .replace("correlation_", "corr. ")
                        + "%",
                        digits=2,
                    )
                    for c in results.columns("lsv_minus_lv_se")
                ),
                row_header="model",
            )
        )
        specs.append(
            TableSpec(
                "mechanism",
                "Reading 1 per model and barrier: the ratio difference d, its predictor dP "
                "(the difference of P(realised vol > vol_ko | ITM)), the predicted sign "
                "-sign(dP) where |dP| > 2 stderr (else 0), whether |d| > 2 stderr, and whether "
                "the signs agree (-- where either is not resolved).",
                "mechanism",
                (
                    Column("ratio_diff", "d ratio"),
                    Column("p_gt_diff", "dP"),
                    Column("predicted_p", "predicted", digits=1),
                    Column("significant", "d > 2 se", digits=1),
                    Column("agrees_p", "agrees", digits=1),
                ),
                row_header="model | vol_ko",
            )
        )
        specs.append(
            TableSpec(
                "mechanism_reading",
                "Reading 2 (the superseded SPEC 6.2 reading) of the same pairs: the width "
                "p90 - p10 of the ITM "
                "distribution minus the local vol's, where the barrier sits against the local "
                "vol's ITM median (vol points), the predicted sign (each factor beyond 2 stderr, "
                "else 0) and whether it agrees with d.",
                "mechanism",
                (
                    Column("width_diff", "d width", unit="vp"),
                    Column("p50_lv_minus_h", "p50 LV - vol_ko", unit="vp"),
                    Column("predicted_w", "predicted", digits=1),
                    Column("agrees_w", "agrees", digits=1),
                ),
                row_header="model | vol_ko",
            )
        )
    for t, what, unit in (
        ("store_check_ratio", "VKO / vanilla ratio", ""),
        ("store_check_price", "VKO price, % of notional", "%"),
        ("store_check_p_ko", "P(KO)", ""),
    ):
        if _has(results, t):
            specs.append(
                TableSpec(
                    t,
                    f"Inline sweep against the store's headline VKO ({what}): an exact 0 is the "
                    "same computation; otherwise the stderr is the quadrature error, not the "
                    "exact one (the estimates share random numbers). The local vol is stepped on "
                    "the reference LSV's grid here, not the store's, so its rows are not the same "
                    "computation.",
                    t,
                    (
                        Column("inline", "inline", unit=unit),
                        Column("store", "store", unit=unit),
                        Column("diff", "d", unit=unit),
                        Column("n_se", "|d| / se", digits=3),
                    ),
                    row_header="model | vol_ko",
                )
            )
    if _has(results, "pairing"):
        specs.append(
            TableSpec(
                "pairing",
                "The shared step schedule: steps to the maturity on the study's grid, on the grid "
                "each model would use alone, and 1 when the model's grid is the shared one (the "
                "local vol takes the reference LSV's leverage-slice times).",
                "pairing",
                (
                    Column("steps", "steps", digits=6),
                    Column("own_steps", "own steps", digits=6),
                    Column("same_grid", "shared", digits=1),
                ),
                row_header="model",
            )
        )
    if _has(results, "realised"):
        specs.append(
            TableSpec(
                "realised_summary",
                "2022 H2 realised-outcome case: the 12m VKO struck on the first close of the "
                "window (data, exact; dates as yyyymmdd).",
                "realised_summary",
                (
                    Column("date", "date", digits=8),
                    Column("count", "count", digits=4),
                    Column("vol", "vol", digits=4),
                    Column("percent", "spot move", digits=3),
                ),
                row_header="quantity",
            )
        )
        specs.append(
            TableSpec(
                "realised",
                "2022 H2 knock-out state at the window end per barrier (data, exact): the share "
                "of the variance budget used, 1 when knocked out, and the largest constant vol "
                "over the remaining life that keeps the put alive.",
                "realised",
                (
                    Column("budget_used", "budget used", digits=4),
                    Column("knocked_out", "knocked out", digits=1),
                    Column("max_remaining_vol", "max vol", digits=4),
                ),
                row_header="barrier",
            )
        )
    specs.append(
        TableSpec(
            "setup",
            "Term sheet and budgets.",
            "setup",
            (Column("value", "value", digits=6),),
            row_header="setting",
        )
    )
    return specs


def _draw_ratio(results: Results) -> Any:
    fig, ax = style.new_figure()
    long = results.long("vko_ratio")
    long = long.assign(axis_h=VP * axis(long, "vol_ko"))
    series_by(ax, "axis_h", [(r, long[long["row"] == r]) for r in results.rows("vko_ratio")])
    ax.set_xlabel("vol_ko [vol pts]")
    ax.set_ylabel("VKO / vanilla put")
    ax.set_title("VKO / vanilla ratio per barrier")
    ax.legend()
    return fig


def _draw_quantiles(results: Results) -> Any:
    fig, ax = style.new_figure()
    rows = results.rows("itm_rv")[:8]
    xs = np.arange(len(rows), dtype=float)
    for i, (k, lab) in enumerate((("p10", "p10"), ("p50", "median"), ("p90", "p90"))):
        v = np.array([results.value("itm_rv", r, k)[0] for r in rows])
        s = np.array([results.value("itm_rv", r, k)[1] for r in rows])
        style.mc_errorbar(ax, xs, v, s, series=i, label=f"{lab} (bootstrap stderr)", line=False)
    hs = sorted({float(x) for x in axis(results.long("vko_ratio"), "vol_ko").dropna()})
    for h in hs:
        ax.axhline(VP * h, color=style.INK["grid"], linewidth=1.0, linestyle=":")
    ax.set_xticks(xs)
    ax.set_xticklabels(rows, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("realised vol given S_T < K [vol pts]")
    ax.set_title("ITM-conditional realised vol (dotted: the vol_ko barriers)")
    ax.legend()
    return fig


def _draw_p_gt(results: Results) -> Any:
    fig, ax = style.new_figure()
    long = results.long("itm_rv")
    long = long[long["column"].str.startswith("p_gt_")]
    long = long.assign(axis_h=VP * axis(long, "vol_ko"))
    series_by(ax, "axis_h", [(r, long[long["row"] == r]) for r in results.rows("itm_rv")])
    ax.set_xlabel("vol_ko [vol pts]")
    ax.set_ylabel("P(realised vol > vol_ko | ITM)")
    ax.set_title("Knock-out probability of the in-the-money paths")
    ax.legend()
    return fig


def _draw_diff(results: Results) -> Any:
    fig, ax = style.new_figure()
    if not _has(results, "lsv_minus_lv"):
        empty_panel(ax, "no local-vol reference in the selection")
        return fig
    long = results.long("lsv_minus_lv")
    long = long.assign(axis_h=VP * axis(long, "vol_ko"))
    ax.axhline(0.0, color=style.INK["spine"], linewidth=0.8)
    series_by(ax, "axis_h", [(r, long[long["row"] == r]) for r in results.rows("lsv_minus_lv")])
    ax.set_xlabel("vol_ko [vol pts]")
    ax.set_ylabel("ratio LSV - ratio LV")
    ax.set_title("LSV minus LV of the VKO / vanilla ratio")
    ax.legend()
    return fig


def _draw_realised(results: Results) -> Any:
    fig, axes = style.new_figure(1, 2)
    if not _has(results, "realised_path"):
        empty_panel(axes[0], "no realised-outcome case in this run")
        empty_panel(axes[1], "no realised-outcome case in this run")
        return fig
    piv = results.pivot("realised_path")
    long = results.long("realised_path", "rv_to_date")
    days = axis(long, "day").to_numpy(dtype=float)
    ex = np.ones_like(days, dtype=bool)
    style.mc_errorbar(
        axes[0],
        days,
        piv["rv_to_date"].to_numpy(dtype=float),
        np.zeros_like(days),
        exact=ex,
        series=0,
        label="realised vol to date",
    )
    for i, h in enumerate(sorted({float(x) for x in axis(results.long("realised"), "vol_ko")})):
        axes[0].axhline(
            VP * h,
            color=style.series_color(min(i + 1, 7)),
            linewidth=1.0,
            linestyle="--",
            label=f"vol_ko {VP * h:.0f}",
        )
    axes[0].set_xlabel("business days from the start close")
    axes[0].set_ylabel("vol [vol pts]")
    axes[0].set_title("Realised vol to date (data: no error bars)")
    axes[0].legend(fontsize=7)
    style.mc_errorbar(
        axes[1],
        days,
        piv["spot"].to_numpy(dtype=float),
        np.zeros_like(days),
        exact=ex,
        series=0,
        label="S / S_start",
    )
    axes[1].axhline(100.0, color=style.INK["spine"], linewidth=0.8, linestyle="--")
    axes[1].set_xlabel("business days from the start close")
    axes[1].set_ylabel("% of the start close (the 100% strike)")
    axes[1].set_title("Spot (data)")
    return fig


def figures(results: Results) -> list[FigureSpec]:
    return [
        FigureSpec(
            "vko_ratio_vs_barrier",
            "VKO / vanilla put ratio against the barrier per model, error bars 1 stderr "
            "(often smaller than the markers).",
            _draw_ratio,
        ),
        FigureSpec(
            "itm_rv_quantiles",
            "10/50/90 percentiles of the full-life realised vol on the paths ending in the money, "
            "per model, pair-bootstrap error bars 1 stderr; dotted lines: the barriers.",
            _draw_quantiles,
        ),
        FigureSpec(
            "p_ko_itm",
            "P(realised vol > vol_ko | S_T < K) against the barrier, error bars 1 stderr.",
            _draw_p_gt,
        ),
        FigureSpec(
            "lsv_minus_lv",
            "LSV minus LV of the VKO / vanilla ratio against the barrier (paired, "
            "error bars 1 stderr of the difference).",
            _draw_diff,
        ),
        FigureSpec(
            "realised_2022h2",
            "Realised-outcome case: realised vol to date against the barriers and the spot path "
            "from the start close (historical data, exact: no error bars).",
            _draw_realised,
        ),
    ]


def _se_comparison(results: Results) -> list[str]:
    if not _has(results, "lsv_minus_lv_se"):
        return []
    long = results.long("lsv_minus_lv_se")

    def col(prefix: str) -> pd.Series:
        sub = long[long["column"].str.startswith(prefix)]
        keyed = sub.assign(vol_ko=axis(sub, "vol_ko").to_numpy(float))
        return keyed.set_index(["row", "vol_ko"])["value"]

    frame = pd.concat(
        {"p": col("se_paired_"), "q": col("se_quadrature_"), "c": col("correlation_")}, axis=1
    )
    text = quadrature_comparison(
        frame["p"].to_numpy(float), frame["q"].to_numpy(float), frame["c"].to_numpy(float)
    )
    return [text, ""] if text else []


def _score(piv: pd.DataFrame, pred: str, agree: str) -> tuple[int, int, int]:
    """``(resolved pairs, agreements, significant differences)`` of one reading."""
    sig = int((piv["significant"] == 1.0).sum())
    testable = piv[np.isfinite(piv[agree].to_numpy(dtype=float))]
    return len(testable), int((testable[agree] == 1.0).sum()), sig


def narrative(results: Results) -> str:
    setup = {r: results.value("setup", r, "value")[0] for r in results.rows("setup")}
    grid = lv_grid_sentence(results)
    lines = [
        "## Data",
        "",
        f"The {setup['maturity [y]']:g}y {100 * setup['strike (fraction of spot)']:g}% put with "
        f"daily fixings ({setup['fixings per year']:g} per year) is priced for "
        f"{int(setup['models'])} models on {int(setup['paths per model'])} paths each, one path "
        "set per model on one shared step schedule and seed, so LSV-minus-LV differences are "
        "paired path by path"
        + (f"; {grid}" if grid else "")
        + " (`source = cache:<key>` for an LSV read from the leverage cache, "
        "`computed` for the local vol, a Dupire construction). Nothing was calibrated. The store "
        "keeps neither the in-the-money realised-vol distribution nor any stderr for it, so it "
        "is recomputed here; its percentile errors come from "
        f"{int(setup['bootstrap replicates'])} pair-bootstrap replicates.",
        "",
        *(["{{table:pairing}}", ""] if _has(results, "pairing") else []),
        "{{table:vko_ratio}}",
        "",
        "{{figure:vko_ratio_vs_barrier}}",
        "",
        "## Where the barrier sits in the in-the-money distribution",
        "",
        "{{table:itm_quantiles}}",
        "",
        "{{table:itm_p_gt}}",
        "",
    ]
    if LV_LABEL in results.rows("itm_rv"):
        p50, s50 = results.value("itm_rv", LV_LABEL, "p50")
        p10, _ = results.value("itm_rv", LV_LABEL, "p10")
        p90, _ = results.value("itm_rv", LV_LABEL, "p90")
        lines += [
            f"Under the local vol the realised vol of the in-the-money paths has median "
            f"{format_value_text(p50, s50)} vol points and a 10-90 band of {p10:.1f} to "
            f"{p90:.1f}. The sign of the LSV-minus-LV ratio difference is not a model property "
            "(SPEC §6.2); two readings of it are scored below.",
            "",
        ]
    if _has(results, "mechanism"):
        piv = results.pivot("mechanism")
        n1, a1, sig = _score(piv, "predicted_p", "agrees_p")
        n2, a2, _ = _score(piv, "predicted_w", "agrees_w")
        lines += [
            f"Of {len(piv)} (model, barrier) pairs, {sig} ratio differences exceed "
            f"{SIGNIFICANCE:g} stderr (paired).",
            "",
            f"- Reading 1 (SPEC §6.2), the knock-out probability of the in-the-money paths: "
            f"resolved on {n1} of them, the predicted sign is the measured one in "
            f"**{a1} of {n1}**.",
            f"- Reading 2, the width and median position (the superseded SPEC §6.2 reading): "
            f"resolved on {n2}, right in **{a2} of {n2}**.",
            "",
            "A reading resolved on few pairs says little either way; the tables list every "
            "pair.",
            "",
            "{{table:lsv_minus_lv}}",
            "",
            *_se_comparison(results),
            "{{table:lsv_minus_lv_se}}",
            "",
            "{{figure:lsv_minus_lv}}",
            "",
            "{{table:mechanism}}",
            "",
            "{{table:mechanism_reading}}",
            "",
            "{{table:lsv_minus_lv_quantiles}}",
            "",
        ]
    checks = [
        t
        for t in ("store_check_ratio", "store_check_price", "store_check_p_ko")
        if _has(results, t)
    ]
    if checks:
        frames = [results.long(t, "diff") for t in checks]
        diffs = pd.concat(frames, ignore_index=True)
        identical = int(diffs["exact"].sum())
        other = diffs[~diffs["exact"]]
        n_se = np.abs(other["value"].to_numpy(float)) / other["stderr"].to_numpy(float)
        lv_rows = int(other["row"].str.startswith(f"{LV_LABEL} | ").sum())
        tail = (
            f"; the other {len(other)} ({lv_rows} of them local-vol rows, priced on the "
            "reference LSV's grid and so not the store's computation) differ by at most "
            f"{float(np.nanmax(n_se)):.2f} times their quadrature error, which is not the exact "
            "error (the estimates share random numbers)"
            if len(other)
            else ""
        )
        lines += [
            f"Against the store's headline VKO, {identical} of {len(diffs)} inline numbers are "
            "identical to the stored ones (the same computation)" + tail + ".",
            "",
        ]
    if _has(results, "realised_summary"):
        rs: dict[str, float] = {}
        for rec in records(results.long("realised_summary")):
            rs[str(rec["row"])] = float(rec["value"])
        piv = results.pivot("realised")
        ko = [r for r in piv.index if piv.loc[r, "knocked_out"] == 1.0]
        alive = [r for r in piv.index if piv.loc[r, "knocked_out"] == 0.0]

        def date(x: float) -> str:
            s = f"{int(x):08d}"
            return f"{s[:4]}-{s[4:6]}-{s[6:]}"

        lines += [
            "## The 2022 H2 realised outcome",
            "",
            f"A 12m VKO struck on {date(rs['start date'])} matures on "
            f"{date(rs['maturity date'])}, after the last close of the history "
            f"({date(rs['last close'])}): **the payoff is not observable**. Over the "
            f"{int(rs['returns observed'])} observed returns the realised vol is "
            f"{rs['realised vol to date']:.2f} vol points (the surface marked "
            f"{rs['marked ATM vol']:.2f} ATM and {rs['marked VS vol']:.2f} for the variance "
            f"swap on the start date) and the spot moved {rs['spot move to date']:+.1f}%. "
            f"Knocked out at the window end: {', '.join(ko) or 'none'}; still alive: "
            f"{', '.join(alive) or 'none'}.",
            "",
            "{{table:realised_summary}}",
            "",
            "{{table:realised}}",
            "",
        ]
    return "\n".join(lines)
