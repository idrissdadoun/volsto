"""S3 — conditional and knock-out variance (SPEC §6.1 / §10.2, owner's M10 Part 2): up-variance,
down-variance and convexity-spread fair strikes across the models, the Gyöngy check of each
leg against the local vol, and the knock-out variance swap's LSV-minus-LV strike against ν, ρ
and θ.

**Data.**  The store keeps four conditional numbers per point (up-var and down-var at
``B = 100%``, the knock-out swap at ``B = 110%`` and its ``P(KO)``).  It lacks the spot-start
variance swap, the convexity spread, the corridor strikes and every other barrier, so every model
is priced here from its cached leverage (``source = cache:<key>``; the local vol is a Dupire
build, ``computed``) on the study's shared step schedule
(:func:`volsto.studies.catalogue._common.price_paired`): the daily variance swap and, per
barrier, the :func:`~volsto.products.conditional_variance.UpVar` (``"prev"`` indicator) and
:func:`~volsto.products.conditional_variance.DownVar` (``"curr"``) statistics and the
:class:`~volsto.products.conditional_variance.KnockOutVarianceSwap` statistics.  With
``store_check`` the four stored numbers are compared with the inline ones: a bitwise-equal
number is reported as an exact zero difference (the same computation), any other difference
with the quadrature error, which is not the exact error (the estimates share random numbers and
their correlation is not measured).  The local vol is stepped on the reference LSV's grid (its
leverage-slice times added), so its store-check rows are not the store's computation: the
``pairing`` table gives both step counts.  The θ sweep needs 2F presets that no production
grid holds: ``configs/grids/s3_ko_var.yaml`` carries them.

**Estimators** (antithetic pairs averaged before every error; ``A = 252``; every statistic an
:class:`~volsto.studies.catalogue._common.Estimate` with its influence function):

* variance swap ``K_var² = E[RV]``; conditional up / down ``K² = E[accrued] / E[count]``
  (delta method); the **corridor** up-var ``K² = E[accrued]`` (the same accrued leg);
* convexity spread ``K_up² − K_var²`` and ``K_KO − K_var`` — joint delta method on one path set;
* knock-out swap ``K_KO² = E[accrued] / E[τ/N]``, ``P(KO)``, ``E[τ/N]``;
* LSV minus LV — **paired**: the same seed, the same step schedule, the stderr of the per-pair
  difference of the influence functions (the quadrature error and the per-pair correlation are
  reported beside it, for comparison).

**What Gyöngy pins** (the theory the check is read against).

1. *Continuous monitoring.*  The LSV is calibrated so that ``E[σ_t² | S_t] = σ_loc²(t, S_t)``
   (Gyöngy), so it shares the local vol's marginals.  Any leg accruing ``∫ f(t, S_t) σ_t² dt``
   with a weight that is a function of ``(t, S_t)`` has the expectation
   ``∫ E[f(t, S_t) σ_loc²(t, S_t)] dt``, the same in both models: the up-, down- and corridor
   variance (``f`` an indicator), the variance swap (``f = 1``) and the counts ``∫ f dt``, hence
   the conditional strikes.  Pinned exactly.
2. *Daily fixings, previous-close indicator* (``"prev"``: up-var, corridor).
   ``E[r_i² 1{S_{i−1} ∈ R}] = E[1{S_{i−1} ∈ R} E[r_i² | F_{i−1}]]`` with
   ``E[r_i² | F_{i−1}] = σ²_{t_{i−1}} Δt + O(Δt²)``.  The ``O(Δt²)`` term (the drift of the
   instantaneous variance, the squared drift) depends on the model.  So the leg is pinned up to
   ``O(Δt)`` in aggregate, plus the leverage calibration residual (``E[L² V | S] = σ_loc²`` holds
   only up to particle and regression error) and the scheme's bias.  The variance swap obeys the
   same rule: **its LSV-minus-LV residual measures this floor**.
3. *Daily fixings, current-close indicator* (``"curr"``: down-var).  Relative to the previous-close
   leg, each step adds ``E[r_i² (1{S_i ∈ R} − 1{S_{i−1} ∈ R})]``.  With ``s² = σ_loc² Δt``,
   ``Z = r_i / s`` and ``p`` the density of ``ln(B / S_{t_{i−1}})``, this expands as
   ``s³ p(0) ∫(E[Z² 1{Z < u}] − H(u)) du + s⁴ p'(0) ∫ u (…) du + …``, which equals
   ``−s³ p(0) E[Z³] − ½ s⁴ p'(0) E[Z⁴] + …`` (for the region below ``B``).  The boundary layer is
   ``O(√Δt)`` wide, but its leading term integrates to ``−E[Z³]``, and the skewness of a one-step
   increment is itself ``O(√Δt)``.  So the extra term is ``O(Δt²)`` per step and ``O(Δt)`` in
   aggregate, **not** ``O(√Δt)``.  Its coefficients are the conditional skewness and kurtosis of
   the one-step return at the barrier; the kurtosis involves ``E[Z⁴] = 3 E[σ⁴ | S] / E[σ² | S]²``,
   which the marginals do not fix.  It is therefore a model-dependent ``O(Δt)`` term on top of
   the previous-close one.  The ``monitoring_order`` table measures it at several fixing
   frequencies (``curr − prev`` accrual, paired on each path set).
4. *The knock-out swap* stops at a hitting time: it depends on the law of the path, not on the
   marginals, and is never pinned.

**The per-cell verdict** (``gyongy`` table; one cell per LSV model × leg × barrier):

* ``z = d / se`` with the paired stderr;
* the tolerance comes from the model's variance-swap residual in variance units,
  ``floor = |δ(K_var²)| + 2 se``, converted to the leg's strike as
  ``floor / (2 K_leg c_leg)`` (``c`` the LV count, 1 for the corridor);
* verdict 0 = consistent with zero (``|z| ≤ 2``); 1 = significant but within the floor;
  2 = beyond the floor.

``z`` is recorded with the nominal stderr 1 of a z-score; the floor and the tolerance are Monte
Carlo numbers with delta-method stderrs (the influence of ``|d| + 2 se`` and of the LV's leg
strike and count); the verdict is an exact flag, the decision at the stated thresholds.

The floor is a heuristic, not a bound: the residual of the variance swap can sit in part of the
spot range and move one leg by more than the total.  A verdict of 0 or 1 therefore does not prove
the invariance below the floor, and a 2 is a flag, not a proof of non-invariance.

**Params** (all required)::

    models: {surface, lv, one_factor, two_factor}
    maturity: 1.0
    per_year: 252
    conditional_barriers: [0.9, 1.0, 1.1]      # up / down-var barriers (fractions of spot)
    ko_barriers: [1.05, 1.10, 1.20]           # up-and-out barriers
    n_paths: 400000
    sweeps: {nu: {rho: -0.7, kappa: 1.5}, rho: {nu: 1.0, kappa: 1.5}}
    store_check: true
    monitoring_order: {barrier: 0.9, per_year: [63, 126, 252, 504], n_paths: 100000,
                       model: {nu: 1.5, rho: -0.7, kappa: 1.5}} | null

Seed: ``pricing``.  Checked by ``tests/test_catalogue_s1_s4.py``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from volsto.config import ConfigError, SimConfig
from volsto.models.base import Model
from volsto.products.base import Product, daily_schedule
from volsto.products.conditional_variance import (
    ConditionalVarianceSwap,
    DownVar,
    KnockOutVarianceSwap,
    UpVar,
)
from volsto.products.variance import VarianceSwap
from volsto.studies import style
from volsto.studies.catalogue._common import (
    ERROR_STAT_NOTE,
    LV_GRID_NOTE,
    LV_LABEL,
    PAIRED_NOTE,
    QUADRATURE_NOTE,
    VP,
    Difference,
    Estimate,
    FloatArray,
    ModelPoint,
    add_z,
    axis,
    axis_text,
    empty_panel,
    floor_estimate,
    leverage_requirements,
    load_model,
    lv_grid_sentence,
    mean_estimate,
    model_source,
    own_steps,
    pair_samples,
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
    sqrt_estimate,
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

TITLE = "S3 - Conditional and knock-out variance"
QUESTION = (
    "Which conditional and knock-out variance strikes does vol-of-vol move at a fixed smile, "
    "and which does Gyongy's theorem pin to the local-vol value?"
)
REQUIRED_PARAMS = (
    "models",
    "maturity",
    "per_year",
    "conditional_barriers",
    "ko_barriers",
    "n_paths",
    "sweeps",
    "store_check",
    "monitoring_order",
)
OPTIONAL_PARAMS: tuple[str, ...] = ()
#: What every exact row of this study is (``_common.unclassified_exact_rows``; the walking test
#: fails on any other exact row): ``(table regex, column regex, kind)``.
EXACT_KINDS: tuple[tuple[str, str, str], ...] = (
    ("gyongy", "verdict", "flag"),  # the decision at the stated thresholds
    ("gyongy_floor", "same_grid", "flag"),
    ("pairing", "steps|own_steps", "count"),
    ("pairing", "same_grid", "flag"),
    # an identical inline and stored number: the difference and its z are exactly 0
    ("store_check_(vol|ratio)", "diff|n_se", "closed form"),
    ("setup", "value", "input"),  # the selection's counts and the config's inputs
)

VOL = "vol pts"
VAR = "vol pts^2"
#: The sentence the owner asked to record verbatim in study.md.
KO_NOT_A_THEOREM = (
    "K_KO > K_var holds under negative skew and a non-inverted term structure but is not a "
    "theorem."
)
#: Store keys of the M4c conditional set: (key, statistic, barrier).
STORE_KEYS: tuple[tuple[str, str, float], ...] = (
    ("upvar_100", "up", 1.0),
    ("downvar_100", "down", 1.0),
    ("kovar_110", "ko", 1.1),
    ("kovar_110_p_ko", "p_ko", 1.1),
)
ANNUALISATION = 252.0
#: Significance threshold of the per-cell verdict (stderr multiples).
Z_THRESHOLD = 2.0
#: Stderr multiples added to the variance-swap residual to form the floor.
FLOOR_SE = 2.0
#: The legs of the Gyöngy check: (key, label, indicator).
LEGS: tuple[tuple[str, str, str], ...] = (
    ("corr", "corridor up", "prev"),
    ("up", "conditional up", "prev"),
    ("down", "conditional down", "curr"),
)
VERDICTS: dict[int, str] = {
    0: f"consistent with zero (|z| <= {Z_THRESHOLD:g})",
    1: "significant but within the variance-swap floor",
    2: "beyond the variance-swap floor",
}
#: The O(dt) and O(sqrt dt) predictions of the ratio D(dt/2) / D(dt) of the monitoring term.
ORDER_RATIOS: dict[str, float] = {"O(dt)": 0.5, "O(sqrt dt)": math.sqrt(0.5)}


def btag(B: float) -> str:
    return f"{round(100 * B):d}"


# --------------------------------------------------------------------------------------------
# params and requirements
# --------------------------------------------------------------------------------------------


def _barriers(values: Any, name: str) -> None:
    if not isinstance(values, list) or not values or any(float(x) <= 0 for x in values):
        raise ConfigError(f"{name}: a non-empty list of positive barriers (fractions of spot)")
    if len({btag(float(x)) for x in values}) != len(values):
        raise ConfigError(f"{name}: barriers must differ at the percent level")


def validate_params(params: Mapping[str, Any]) -> None:
    validate_selection(params["models"])
    if float(params["maturity"]) <= 0 or int(params["per_year"]) <= 0:
        raise ConfigError("maturity and per_year must be positive")
    _barriers(params["conditional_barriers"], "conditional_barriers")
    _barriers(params["ko_barriers"], "ko_barriers")
    if any(float(x) <= 1.0 for x in params["ko_barriers"]):
        raise ConfigError("ko_barriers: up-and-out barriers above the spot (> 1)")
    n = int(params["n_paths"])
    if n <= 0 or n % 2:
        raise ConfigError("n_paths: a positive even number (antithetic pairs)")
    sw = params["sweeps"]
    if (
        not isinstance(sw, Mapping)
        or set(sw) != {"nu", "rho"}
        or set(sw["nu"]) != {"rho", "kappa"}
        or set(sw["rho"]) != {"nu", "kappa"}
    ):
        raise ConfigError("sweeps: expected {nu: {rho, kappa}, rho: {nu, kappa}}")
    if not isinstance(params["store_check"], bool):
        raise ConfigError("store_check: expected true or false")
    mo = params["monitoring_order"]
    if mo is not None:
        if not isinstance(mo, Mapping) or set(mo) != {"barrier", "per_year", "n_paths", "model"}:
            raise ConfigError(
                "monitoring_order: expected null or {barrier, per_year, n_paths, model}"
            )
        freqs = mo["per_year"]
        if not isinstance(freqs, list) or len(freqs) < 2 or sorted(freqs) != freqs:
            raise ConfigError("monitoring_order.per_year: at least two increasing frequencies")
        if any(int(b) != 2 * int(a) for a, b in itertools.pairwise(freqs)):
            raise ConfigError("monitoring_order.per_year: each frequency doubles the previous")
        if int(mo["n_paths"]) <= 0 or int(mo["n_paths"]) % 2 or float(mo["barrier"]) <= 0:
            raise ConfigError("monitoring_order: a positive even n_paths and a positive barrier")
        if not isinstance(mo["model"], Mapping) or set(mo["model"]) != {"nu", "rho", "kappa"}:
            raise ConfigError("monitoring_order.model: expected {nu, rho, kappa} (a 1F point)")


def requirements(ctx: StudyContext) -> list[Requirement]:
    """The LSV leverages of the selection (the local vol needs none)."""
    ctx.seed("pricing")
    models = select_models(ctx, ctx.params["models"])
    if ctx.params["monitoring_order"] is not None:
        _monitoring_models(models, ctx.params["monitoring_order"])
    return leverage_requirements(ctx, models)


def _monitoring_models(
    models: Sequence[ModelPoint], mo: Mapping[str, Any]
) -> tuple[ModelPoint, ModelPoint]:
    """The LV and the 1F model of the monitoring-order measurement (both must be selected)."""
    lv = next((mp for mp in models if mp.is_lv), None)
    want = mo["model"]
    lsv = next(
        (
            mp
            for mp in models
            if mp.mode == "one_factor"
            and math.isclose(mp.axes["nu"], float(want["nu"]))
            and math.isclose(mp.axes["rho"], float(want["rho"]))
            and math.isclose(mp.axes["kappa"], float(want["kappa"]))
        ),
        None,
    )
    if lv is None or lsv is None:
        raise ConfigError(
            "monitoring_order needs the LV point and the 1F point "
            f"{dict(want)} in the model selection"
        )
    return lv, lsv


# --------------------------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class VarStats:
    """One model's statistics as :class:`Estimate` (strikes in vol units, spreads in variance
    units), keyed ``"var"``, ``"var2"`` (``K_var²``), ``"<leg>_<B>"`` for the legs of
    :data:`LEGS` (the strike), ``"<leg>_acc_<B>"`` (the accrued variance), ``"<leg>_cnt_<B>"``
    (the count), ``"conv_<B>"``, ``"ko_<B>"``, ``"p_ko_<B>"``, ``"life_<B>"``, ``"kmv_<B>"``."""

    est: dict[str, Estimate]
    same_grid: bool
    steps: int
    own_steps: int


def variance_statistics(
    model: Model,
    sim: SimConfig,
    maturity: float,
    per_year: int,
    cond_barriers: Sequence[float],
    ko_barriers: Sequence[float],
    reference_times: FloatArray,
) -> VarStats:
    """Every statistic of the module docstring on one path set of the shared schedule."""
    spot = float(model.spot)
    discount = model.forward_curve.rate_curve
    times = daily_schedule(maturity, per_year)
    vs = VarianceSwap(times, 0.0, discount, annualisation=ANNUALISATION)
    legs: list[Product] = [vs]
    names = ["rv"]
    for B in cond_barriers:
        up = UpVar(times, B * spot, 0.0, discount, annualisation=ANNUALISATION)
        dn = DownVar(times, B * spot, 0.0, discount, annualisation=ANNUALISATION)
        legs += [up.leg("accrued"), up.leg("count"), dn.leg("accrued"), dn.leg("count")]
        names += [f"up_a_{B}", f"up_d_{B}", f"dn_a_{B}", f"dn_d_{B}"]
    for B in ko_barriers:
        ko = KnockOutVarianceSwap(times, B * spot, 0.0, discount, annualisation=ANNUALISATION)
        legs += [ko.leg("accrued"), ko.leg("count"), ko.leg("ko")]
        names += [f"ko_a_{B}", f"ko_d_{B}", f"ko_k_{B}"]
    res = price_paired(model, sim, legs, reference_times)
    steps = int(paired_grid(model, sim, legs, reference_times).n_steps)
    df = float(discount.df(times[-1]))
    x = {n: pair_samples(r, sim.antithetic) for n, r in zip(names, res, strict=True)}
    x["rv"] = x["rv"] / df  # the zero-strike swap pays DF(T) * RV
    var2 = mean_estimate(x["rv"])
    est: dict[str, Estimate] = {"var2": var2.scaled(VP * VP), "var": sqrt_estimate(var2)}
    for B in cond_barriers:
        for leg, a_name, d_name in (("up", "up_a", "up_d"), ("down", "dn_a", "dn_d")):
            a, d = x[f"{a_name}_{B}"], x[f"{d_name}_{B}"]
            r = ratio_estimate(a, d)
            est[f"{leg}_{B}"] = sqrt_estimate(r)
            est[f"{leg}_acc_{B}"] = mean_estimate(a).scaled(VP * VP)
            est[f"{leg}_cnt_{B}"] = mean_estimate(d)
            if leg == "up":
                est[f"corr_{B}"] = sqrt_estimate(mean_estimate(a))
                est[f"corr_acc_{B}"] = est[f"up_acc_{B}"]
                est[f"corr_cnt_{B}"] = Estimate(1.0, np.zeros_like(a))
                est[f"conv_{B}"] = r.minus(var2).scaled(VP * VP)
    for B in ko_barriers:
        a, d = x[f"ko_a_{B}"], x[f"ko_d_{B}"]
        k = sqrt_estimate(ratio_estimate(a, d))
        est[f"ko_{B}"] = k
        est[f"p_ko_{B}"] = mean_estimate(x[f"ko_k_{B}"])
        est[f"life_{B}"] = mean_estimate(d)
        est[f"kmv_{B}"] = k.minus(est["var"])
    return VarStats(est, same_times(model, reference_times), steps, own_steps(model, sim, legs))


# --------------------------------------------------------------------------------------------
# compute
# --------------------------------------------------------------------------------------------


def compute(ctx: StudyContext) -> Results:
    p = ctx.params
    models = select_models(ctx, p["models"])
    cond = [float(B) for B in p["conditional_barriers"]]
    kob = [float(B) for B in p["ko_barriers"]]
    n_paths = int(p["n_paths"])
    T, per_year = float(p["maturity"]), int(p["per_year"])
    lsvs = [mp for mp in models if not mp.is_lv]
    lv = next((mp for mp in models if mp.is_lv), None)
    loaded: dict[str, Model] = {}
    ref_times: FloatArray = np.empty(0)
    if lsvs:
        loaded[lsvs[0].label] = load_model(ctx, lsvs[0])
        ref_times = np.asarray(loaded[lsvs[0].label].required_times(), dtype=np.float64)
    b = ResultsBuilder()
    ref: VarStats | None = None
    order = ([lv] if lv is not None else []) + lsvs
    for mp in order:
        model = loaded.pop(mp.label, None) or load_model(ctx, mp)
        ctx.log.info("%s: conditional / KO variance at %d paths", mp.label, n_paths)
        sim = pricing_sim(ctx, mp, n_paths)
        st = variance_statistics(model, sim, T, per_year, cond, kob, ref_times)
        src = model_source(mp)
        _model_rows(b, mp, st, src, cond, kob)
        _pairing_rows(b, mp, st, src)
        if p["store_check"] and math.isclose(T, 1.0) and per_year == 252:
            _store_check(ctx, b, mp, st, src)
        if mp.is_lv:
            ref = st
        elif ref is not None and lv is not None:
            _diff_rows(b, mp, st, ref, f"{src};{model_source(lv)}", cond, kob)
    mo = p["monitoring_order"]
    if mo is not None:
        _monitoring_order(ctx, b, models, mo, T, ref_times)
    ctx.record("n_paths", n_paths)
    sw = p["sweeps"]
    for row, val in (
        ("models", float(len(models))),
        ("paths per model", float(n_paths)),
        ("maturity [y]", T),
        ("fixings per year", float(per_year)),
        ("nu sweep rho", float(sw["nu"]["rho"])),
        ("nu sweep kappa", float(sw["nu"]["kappa"])),
        ("rho sweep nu", float(sw["rho"]["nu"])),
        ("rho sweep kappa", float(sw["rho"]["kappa"])),
        ("z threshold", Z_THRESHOLD),
        ("floor stderr multiple", FLOOR_SE),
    ):
        b.add_exact("setup", row, "value", val, unit="", source="computed")
    return b.build()


def _model_rows(
    b: ResultsBuilder,
    mp: ModelPoint,
    st: VarStats,
    src: str,
    cond: Sequence[float],
    kob: Sequence[float],
) -> None:
    ax = mp.axes
    e = st.est

    def add(table: str, col: str, est: Estimate, scale: float, unit: str, **kw: Any) -> None:
        note = str(kw.pop("note", ""))
        b.add(
            table,
            mp.label,
            col,
            scale * est.value,
            scale * est.stderr,
            unit=unit,
            source=src,
            note=note,
            axes={**ax, **kw},
        )

    add("updown", "var", e["var"], VP, VOL)
    for B in cond:
        t = btag(B)
        add("updown", f"up_{t}", e[f"up_{B}"], VP, VOL, barrier=B)
        add("updown", f"down_{t}", e[f"down_{B}"], VP, VOL, barrier=B)
        add("convexity", f"corridor_{t}", e[f"corr_{B}"], VP, VOL, barrier=B)
        add(
            "convexity",
            f"spread_{t}",
            e[f"conv_{B}"],
            1.0,
            VAR,
            barrier=B,
            note="K_up^2 - K_var^2, joint delta method on one path set",
        )
    for B in kob:
        t = btag(B)
        add("ko_var", f"ko_{t}", e[f"ko_{B}"], VP, VOL, barrier=B)
        add("ko_var", f"p_ko_{t}", e[f"p_ko_{B}"], 1.0, DIMENSIONLESS, barrier=B)
        add(
            "ko_var",
            f"life_{t}",
            e[f"life_{B}"],
            1.0,
            DIMENSIONLESS,
            barrier=B,
            note="E[tau/N]: expected fraction of the life accrued",
        )
        add(
            "ko_var",
            f"ko_minus_var_{t}",
            e[f"kmv_{B}"],
            VP,
            VOL,
            barrier=B,
            note="joint delta method on one path set",
        )


def _pairing_rows(b: ResultsBuilder, mp: ModelPoint, st: VarStats, src: str) -> None:
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


def _add_difference(
    b: ResultsBuilder,
    table: str,
    row: str,
    col: str,
    d: Difference,
    scale: float,
    unit: str,
    src: str,
    axes: Mapping[str, Any],
) -> None:
    """A paired difference, with its quadrature error and correlation beside it (Monte Carlo
    numbers with their own delta-method stderrs)."""
    note = PAIRED_NOTE + ("" if d.same_grid else "; the two grids differ (still paired by path)")
    b.add(
        table,
        row,
        col,
        scale * d.value,
        scale * d.stderr,
        unit=unit,
        source=src,
        note=note,
        axes=axes,
    )
    finite = math.isfinite(d.value)
    b.add(
        table,
        row,
        f"{col}_se_quadrature",
        scale * d.stderr_quadrature if finite else math.nan,
        scale * d.quadrature_se if finite else math.nan,
        unit=unit,
        source=src,
        note="the quadrature error of the same difference, for comparison; " + ERROR_STAT_NOTE,
        axes=axes,
    )
    b.add(
        table,
        row,
        f"{col}_correlation",
        d.correlation if finite else math.nan,
        d.correlation_se if finite else math.nan,
        unit=DIMENSIONLESS,
        source=src,
        note="per-pair correlation of the two estimates; " + ERROR_STAT_NOTE,
        axes=axes,
    )


def _tolerance(floor: Estimate, k_leg: Estimate, count: Estimate) -> Estimate:
    """``floor / (2 VP K_leg c_leg)`` (vol points) with its delta-method influence; every
    input is a statistic of the same path set (the floor of the paired difference, the LV's
    leg strike and count)."""
    k, c = k_leg.value, count.value
    n = floor.influence.size
    if not (k > 0 and c > 0 and math.isfinite(floor.value)) or n != k_leg.influence.size:
        return Estimate(math.nan, np.full(n, math.nan))
    scale = 1.0 / (2.0 * VP * k * c)
    t = scale * floor.value
    infl = scale * floor.influence - t * (k_leg.influence / k + count.influence / c)
    return Estimate(t, infl)


def _diff_rows(
    b: ResultsBuilder,
    mp: ModelPoint,
    st: VarStats,
    ref: VarStats,
    src: str,
    cond: Sequence[float],
    kob: Sequence[float],
) -> None:
    same = st.same_grid and ref.same_grid

    def diff(key: str) -> Difference:
        return paired_difference(st.est[key], ref.est[key], same_grid=same)

    ax = {**mp.axes, "same_grid": float(same)}
    d_var = diff("var")
    d_var2 = diff("var2")
    floor_est = floor_estimate(d_var2, FLOOR_SE)
    floor = floor_est.value
    _add_difference(b, "gyongy_floor", mp.label, "d_var", d_var, VP, VOL, src, ax)
    b.add(
        "gyongy_floor",
        mp.label,
        "d_var2",
        d_var2.value,
        d_var2.stderr,
        unit=VAR,
        source=src,
        note=PAIRED_NOTE,
        axes=ax,
    )
    add_z(
        b, "gyongy_floor", mp.label, "z_var", d_var.z, source=src, note="d / paired stderr", axes=ax
    )
    b.add(
        "gyongy_floor",
        mp.label,
        "floor",
        floor,
        floor_est.stderr if math.isfinite(floor) else math.nan,
        unit=VAR,
        source=src,
        note=f"|d(K_var^2)| + {FLOOR_SE:g} stderr; a Monte Carlo number (delta-method stderr)",
        axes=ax,
    )
    b.add_exact(
        "gyongy_floor",
        mp.label,
        "same_grid",
        float(same),
        unit="",
        source="computed",
        note="1: the two models stepped on the same times",
        axes=ax,
    )
    for B in cond:
        for leg, label, indicator in LEGS:
            d = diff(f"{leg}_{B}")
            tol_est = _tolerance(floor_est, ref.est[f"{leg}_{B}"], ref.est[f"{leg}_cnt_{B}"])
            tol = tol_est.value
            z = d.z
            if not math.isfinite(z):
                verdict = math.nan
            elif abs(z) <= Z_THRESHOLD:
                verdict = 0.0
            elif abs(VP * d.value) <= tol:
                verdict = 1.0
            else:
                verdict = 2.0
            row = f"{mp.label} | {label} {btag(B)}%"
            cax = {**ax, "leg": leg, "indicator": indicator, "barrier": B}
            _add_difference(b, "gyongy", row, "d", d, VP, VOL, src, cax)
            add_z(b, "gyongy", row, "z", z, source=src, note="d / paired stderr", axes=cax)
            b.add(
                "gyongy",
                row,
                "tolerance",
                tol,
                tol_est.stderr if math.isfinite(tol) else math.nan,
                unit=VOL,
                source=src,
                note="the variance-swap floor at this leg's level; a Monte Carlo number "
                "(delta-method stderr)",
                axes=cax,
            )
            b.add_exact(
                "gyongy",
                row,
                "verdict",
                verdict,
                unit="",
                source="computed",
                note=VERDICTS.get(int(verdict), "") if math.isfinite(verdict) else "",
                axes=cax,
            )
            acc = diff(f"{leg}_acc_{B}")
            b.add(
                "gyongy_accrued",
                row,
                "d_acc",
                acc.value,
                acc.stderr,
                unit=VAR,
                source=src,
                note=PAIRED_NOTE,
                axes=cax,
            )
        dc = diff(f"conv_{B}")
        _add_difference(
            b,
            "convexity_diff",
            mp.label,
            f"d_spread_{btag(B)}",
            dc,
            1.0,
            VAR,
            src,
            {**ax, "barrier": B},
        )
    for B in kob:
        t = btag(B)
        _add_difference(
            b, "ko_diff", mp.label, f"d_ko_{t}", diff(f"ko_{B}"), VP, VOL, src, {**ax, "barrier": B}
        )
        _add_difference(
            b,
            "ko_diff",
            mp.label,
            f"d_p_ko_{t}",
            diff(f"p_ko_{B}"),
            1.0,
            DIMENSIONLESS,
            src,
            {**ax, "barrier": B},
        )


def _store_check(
    ctx: StudyContext, b: ResultsBuilder, mp: ModelPoint, st: VarStats, src: str
) -> None:
    if mp.id not in stored_ids(ctx):
        return
    vals = {
        str(r["key"]): (float(r["value"]), float(r["value_stderr"]))
        for r in records(ctx.store().products(mp.id))
    }
    for key, stat, B in STORE_KEYS:
        name = f"{stat}_{B}"
        if key not in vals or name not in st.est:
            continue
        scale, unit = (1.0, DIMENSIONLESS) if stat == "p_ko" else (VP, VOL)
        e = st.est[name]
        v, s = e.value, e.stderr
        sv, ss = vals[key]
        row = f"{mp.label} / {key}"
        ax = {**mp.axes, "key": key}
        tbl = "store_check_ratio" if stat == "p_ko" else "store_check_vol"
        both = f"{src};{store_source(mp.id)}"
        grid_note = (
            f"{LV_GRID_NOTE} ({st.steps} steps against {st.own_steps}): not the store's "
            "computation"
            if mp.is_lv and st.steps != st.own_steps
            else ""
        )
        b.add(
            tbl,
            row,
            "inline",
            scale * v,
            scale * s,
            unit=unit,
            source=src,
            note=grid_note,
            axes=ax,
        )
        b.add(
            tbl,
            row,
            "store",
            scale * sv,
            scale * ss,
            unit=unit,
            source=store_source(mp.id),
            axes=ax,
        )
        if v == sv:
            b.add_exact(
                tbl,
                row,
                "diff",
                0.0,
                unit=unit,
                source=both,
                note="identical (same computation: same seed, same steps to the " "maturity)",
                axes=ax,
            )
            b.add_exact(
                tbl,
                row,
                "n_se",
                0.0,
                unit=DIMENSIONLESS,
                source="computed",
                note="identical",
                axes=ax,
            )
            continue
        d_se = scale * rss(s, ss)
        b.add(
            tbl,
            row,
            "diff",
            scale * (v - sv),
            d_se,
            unit=unit,
            source=both,
            note="inline minus store; " + QUADRATURE_NOTE + (f"; {grid_note}" if grid_note else ""),
            axes=ax,
        )
        add_z(
            b,
            tbl,
            row,
            "n_se",
            scale * abs(v - sv) / d_se if d_se > 0 else math.nan,
            source="computed",
            note="|d| over the quadrature error (not the exact error)",
            axes=ax,
        )


def _monitoring_order(
    ctx: StudyContext,
    b: ResultsBuilder,
    models: Sequence[ModelPoint],
    mo: Mapping[str, Any],
    maturity: float,
    ref_times: FloatArray,
) -> None:
    """``D(Δt) = E[curr accrual] − E[prev accrual]`` of the down corridor at ``barrier`` per
    fixing frequency, for the LV and one 1F model, paired on each path set; the ratios of
    successive frequencies and the LSV-minus-LV difference, paired over the path index."""
    lv, lsv = _monitoring_models(models, mo)
    B = float(mo["barrier"])
    n = int(mo["n_paths"])
    freqs = [int(f) for f in mo["per_year"]]
    per_model: dict[str, list[Estimate]] = {}
    grids: dict[str, bool] = {}
    for mp in (lv, lsv):
        model = load_model(ctx, mp)
        sim = pricing_sim(ctx, mp, n)
        grids[mp.label] = same_times(model, ref_times)
        ests: list[Estimate] = []
        for f in freqs:
            times = daily_schedule(maturity, f)
            disc = model.forward_curve.rate_curve
            legs: list[Product] = [
                ConditionalVarianceSwap(
                    times,
                    B * model.spot,
                    "down",
                    ind,
                    "corridor",
                    0.0,
                    disc,
                    annualisation=float(f),
                ).leg("accrued")
                for ind in ("prev", "curr")
            ]
            res = price_paired(model, sim, legs, ref_times)
            prev, curr = (mean_estimate(pair_samples(r, sim.antithetic)) for r in res)
            dd = curr.minus(prev).scaled(VP * VP)
            ests.append(dd)
            row = f"{mp.label} | {f}/y"
            ax = {**mp.axes, "per_year": float(f), "barrier": B}
            src = model_source(mp)
            b.add(
                "monitoring_order",
                row,
                "D",
                dd.value,
                dd.stderr,
                unit=VAR,
                source=src,
                note="curr minus prev down-corridor accrual, paired on one path set",
                axes=ax,
            )
            b.add(
                "monitoring_order",
                row,
                "prev",
                VP * VP * prev.value,
                VP * VP * prev.stderr,
                unit=VAR,
                source=src,
                axes=ax,
            )
        for i in range(1, len(freqs)):
            r = _ratio(ests[i], ests[i - 1])
            row = f"{mp.label} | {freqs[i]}/y"
            ax = {**mp.axes, "per_year": float(freqs[i]), "barrier": B}
            b.add(
                "monitoring_order",
                row,
                "ratio",
                r.value,
                r.stderr,
                unit=DIMENSIONLESS,
                source=model_source(mp),
                note=f"D at {freqs[i]}/y over D at {freqs[i - 1]}/y (paired by path)",
                axes=ax,
            )
        per_model[mp.label] = ests
    same = grids[lv.label] and grids[lsv.label]
    for f, a, c in zip(freqs, per_model[lsv.label], per_model[lv.label], strict=True):
        d = paired_difference(a, c, same_grid=same)
        _add_difference(
            b,
            "monitoring_diff",
            f"{f}/y",
            "d_D",
            d,
            1.0,
            VAR,
            f"{model_source(lsv)};{model_source(lv)}",
            {**lsv.axes, "per_year": float(f), "barrier": B},
        )
    ctx.record("monitoring_order", {"barrier": B, "per_year": freqs, "n_paths": n})


def _ratio(num: Estimate, den: Estimate) -> Estimate:
    """``num / den`` of two estimates on the same path index (delta method)."""
    if den.value == 0 or not (math.isfinite(num.value) and math.isfinite(den.value)):
        return Estimate(math.nan, np.full(num.influence.size, math.nan))
    r = num.value / den.value
    return Estimate(r, (num.influence - r * den.influence) / den.value)


# --------------------------------------------------------------------------------------------
# tables, figures, narrative (results only)
# --------------------------------------------------------------------------------------------


def _has(results: Results, table: str) -> bool:
    return table in results.tables()


def _cols(results: Results, table: str, prefix: str) -> list[str]:
    if not _has(results, table):
        return []
    return [c for c in results.columns(table) if c.startswith(prefix)]


def _b(col: str) -> str:
    return col.rsplit("_", 1)[1] + "%"


def _setup(results: Results) -> dict[str, float]:
    return {r: results.value("setup", r, "value")[0] for r in results.rows("setup")}


def _sweep_rows(results: Results, table: str, kind: str) -> list[str]:
    """The LSV rows of ``table`` on the ν sweep (``kind = "nu"``), the ρ sweep (``"rho"``) or the
    θ sweep (``"theta"``, every 2F row)."""
    s = _setup(results)
    out: list[str] = []
    for r in results.rows(table):
        a = results.axes(table, r)
        mode = a.get("mode")
        if kind == "theta":
            if mode == "two_factor":
                out.append(r)
            continue
        if mode != "one_factor":
            continue
        if (
            kind == "nu"
            and math.isclose(a["rho"], s["nu sweep rho"])
            and math.isclose(a["kappa"], s["nu sweep kappa"])
        ):
            out.append(r)
        if (
            kind == "rho"
            and math.isclose(a["nu"], s["rho sweep nu"])
            and math.isclose(a["kappa"], s["rho sweep kappa"])
        ):
            out.append(r)
    return out


def _group(
    results: Results,
    name: str,
    table: str,
    prefix: str,
    caption: str,
    unit: str,
    extra: tuple[Column, ...] = (),
) -> TableSpec | None:
    """One narrow table: the ``prefix<B>`` columns of ``table`` headed by their barrier."""
    cols = [c for c in _cols(results, table, prefix) if c[len(prefix) :].isdigit()]
    if not cols and not extra:
        return None
    return TableSpec(
        name,
        caption,
        table,
        (*extra, *(Column(c, f"B={_b(c)}", unit=unit) for c in cols)),
        row_header="model",
    )


def tables(results: Results) -> list[TableSpec]:
    specs: list[TableSpec | None] = [
        _group(
            results,
            "strikes_up",
            "updown",
            "up_",
            "Fair strikes, vol points: the daily variance swap and the conditional up-var "
            "(previous close above B) per barrier.",
            "vp",
            (Column("var", "var swap", unit="vp"),),
        ),
        _group(
            results,
            "strikes_down",
            "updown",
            "down_",
            "Conditional down-var fair strikes (current close below B) per barrier, vol points.",
            "vp",
        ),
        _group(
            results,
            "corridor",
            "convexity",
            "corridor_",
            "Corridor up-var fair strikes (previous close above B, K^2 = E[accrued]) per "
            "barrier, vol points.",
            "vp",
        ),
        _group(
            results,
            "convexity",
            "convexity",
            "spread_",
            "Convexity spread K_up^2 - K_var^2 per barrier, vol points squared (joint delta "
            "method on one path set).",
            "vp²",
        ),
        _group(
            results,
            "ko_strike",
            "ko_var",
            "ko_",
            "Up-and-out knock-out variance swap fair strikes per barrier, vol points.",
            "vp",
        ),
        _group(
            results,
            "ko_probability",
            "ko_var",
            "p_ko_",
            "Knock-out variance swap: P(KO) per barrier.",
            "",
        ),
        _group(
            results,
            "ko_life",
            "ko_var",
            "life_",
            "Knock-out variance swap: the expected accrued fraction of the life E[tau/N].",
            "",
        ),
        _group(
            results,
            "ko_minus_var",
            "ko_var",
            "ko_minus_var_",
            "K_KO - K_var per barrier, vol points (paired: joint delta method on one path set).",
            "vp",
        ),
    ]
    if _has(results, "gyongy"):
        specs += [
            TableSpec(
                "gyongy_floor",
                "The floor: LSV minus LV of the daily variance swap (paired stderr; the "
                "quadrature error beside it for comparison), its z, and the floor "
                "|d(K_var^2)| + 2 stderr in vol points squared.",
                "gyongy_floor",
                (
                    Column("d_var", "d var swap", unit="vp"),
                    Column("d_var_se_quadrature", "se quadrature", unit="vp", digits=2),
                    Column("z_var", "z", digits=3),
                    Column("floor", "floor", unit="vp²", digits=3),
                ),
                row_header="model",
            ),
            TableSpec(
                "gyongy",
                "Gyongy check per cell, LSV minus LV of the strike (vol points, paired "
                "stderr), z, the tolerance (the model's variance-swap floor at the leg's level) "
                "and the verdict: 0 = consistent with zero (|z| <= 2), 1 = significant but "
                "within the floor, 2 = beyond the floor.",
                "gyongy",
                (
                    Column("d", "d", unit="vp"),
                    Column("z", "z", digits=3),
                    Column("tolerance", "tolerance", unit="vp", digits=2),
                    Column("verdict", "verdict", digits=1),
                ),
                row_header="model | leg",
            ),
            TableSpec(
                "gyongy_se",
                "Paired stderr against the quadrature error of the same differences (vol "
                "points) and the per-pair correlation of the two estimates.",
                "gyongy",
                (
                    Column("d", "d (paired se)", unit="vp"),
                    Column("d_se_quadrature", "se quadrature", unit="vp", digits=2),
                    Column("d_correlation", "correlation", digits=2),
                ),
                row_header="model | leg",
            ),
            _group(
                results,
                "ko_diff",
                "ko_diff",
                "d_ko_",
                "Knock-out variance swap, LSV minus LV: fair strike per barrier, vol points "
                "(paired stderr). Never pinned by Gyongy.",
                "vp",
            ),
            _group(
                results,
                "ko_diff_p",
                "ko_diff",
                "d_p_ko_",
                "Knock-out variance swap, LSV minus LV: P(KO) per barrier (paired stderr).",
                "",
            ),
        ]
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
    if _has(results, "monitoring_order"):
        specs.append(
            TableSpec(
                "monitoring_order",
                "The current-close monitoring term D = E[curr accrual] - E[prev accrual] of the "
                "down corridor per fixing frequency (vol points squared, paired on each path "
                "set), and the ratio to D at half the frequency: 0.5 for an O(dt) term, 0.71 "
                "for O(sqrt dt).",
                "monitoring_order",
                (
                    Column("D", "D", unit="vp²"),
                    Column("prev", "prev accrual", unit="vp²"),
                    Column("ratio", "ratio"),
                ),
                row_header="model | fixings",
            )
        )
        specs.append(
            TableSpec(
                "monitoring_diff",
                "LSV minus LV of the monitoring term D per fixing frequency (vol points squared, "
                "paired).",
                "monitoring_diff",
                (
                    Column("d_D", "d D", unit="vp²"),
                    Column("d_D_se_quadrature", "se quadrature", unit="vp²", digits=2),
                ),
                row_header="fixings per year",
            )
        )
    for t, what, unit in (
        ("store_check_vol", "fair vols, vol points", "vp"),
        ("store_check_ratio", "P(KO)", ""),
    ):
        if _has(results, t):
            specs.append(
                TableSpec(
                    t,
                    f"Inline strikes against the store's M4c conditional set ({what}): an exact "
                    "0 difference is the same computation (same seed, same steps); otherwise the "
                    "stderr is the quadrature error, not the exact one (the estimates share "
                    "random numbers). The local vol is stepped on the reference LSV's grid here, "
                    "not the store's, so its rows are not the same computation.",
                    t,
                    (
                        Column("inline", "inline", unit=unit),
                        Column("store", "store", unit=unit),
                        Column("diff", "d", unit=unit),
                        Column("n_se", "|d| / se", digits=3),
                    ),
                    row_header="model / key",
                )
            )
    specs.append(
        TableSpec(
            "setup",
            "Budgets, sweep slices and the verdict thresholds.",
            "setup",
            (Column("value", "value"),),
            row_header="setting",
        )
    )
    return [t for t in specs if t is not None]


def _by_model(long: pd.DataFrame, rows: Sequence[str]) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for r in rows:
        sub = long[long["row"] == r]
        out.append((r, sub.assign(x=VP * axis(sub, "barrier"))))
    return out


def _draw_strikes(results: Results) -> Any:
    fig, axes = style.new_figure(1, 3, sharey=False)
    rows = results.rows("updown")[:8]
    for ax, (table, prefix, title) in zip(
        axes,
        (
            ("updown", "up_", "conditional up-var"),
            ("updown", "down_", "conditional down-var"),
            ("ko_var", "ko_", "knock-out var"),
        ),
    ):
        long = results.long(table)
        long = long[long["column"].str.match(rf"^{prefix}\d+$")]
        series_by(ax, "x", _by_model(long, rows))
        ax.set_title(title)
        ax.set_xlabel("barrier [% of spot]")
        ax.set_ylabel("fair strike [vol pts]")
    axes[0].legend(fontsize=7)
    return fig


def _draw_gyongy(results: Results) -> Any:
    fig, axes = style.new_figure(1, 3, sharey=True)
    if not _has(results, "gyongy"):
        for ax in axes:
            empty_panel(ax, "no local-vol reference in the selection")
        return fig
    long = results.long("gyongy")
    models = list(dict.fromkeys(r.split(" | ")[0] for r in long["row"]))[:8]
    leg_axes = axis_text(long, "leg")
    for ax, (leg, label, indicator) in zip(axes, LEGS):
        ax.axhline(0.0, color=style.INK["spine"], linewidth=0.8)
        sub = long[leg_axes == leg]
        for i, m in enumerate(models):
            cells = sub[sub["row"].str.startswith(f"{m} | ")]
            d = cells[cells["column"] == "d"]
            tol = cells[cells["column"] == "tolerance"]
            xs = VP * axis(d, "barrier").to_numpy(float) + 1.2 * (i - (len(models) - 1) / 2)
            style.mc_errorbar(
                ax,
                xs,
                d["value"].to_numpy(float),
                d["stderr"].to_numpy(float),
                series=i,
                label=m,
                line=False,
            )
            tx = VP * axis(tol, "barrier").to_numpy(float) + 1.2 * (i - (len(models) - 1) / 2)
            for x, t in zip(tx, tol["value"].to_numpy(float)):
                if math.isfinite(t):
                    ax.plot([x - 0.4, x + 0.4], [t, t], color=style.series_color(i), lw=0.8)
                    ax.plot([x - 0.4, x + 0.4], [-t, -t], color=style.series_color(i), lw=0.8)
        ax.set_title(f"{label} ({indicator} close)", fontsize=9)
        ax.set_xlabel("barrier [% of spot]")
    axes[0].set_ylabel("LSV - LV [vol pts]")
    axes[0].legend(fontsize=6)
    return fig


def _sweep_groups(long: pd.DataFrame, kind: str, kappa: float) -> list[tuple[str, pd.DataFrame]]:
    """The series of a sweep figure: per rho against nu, per nu against rho (1F rows at the
    sweep's kappa), or the 2F rows against theta."""
    mode = axis_text(long, "mode")
    if kind == "theta":
        two = long[mode == "two_factor"]
        return [("2F presets", two.assign(xv=axis(two, "theta")))] if len(two) else []
    one = long[(mode == "one_factor") & np.isclose(axis(long, "kappa"), kappa)]
    by, x = ("rho", "nu") if kind == "nu" else ("nu", "rho")
    groups = []
    for g in sorted({float(v) for v in axis(one, by)}):
        sub = one[np.isclose(axis(one, by), g)]
        groups.append((f"{by} {g:g}", sub.assign(xv=axis(sub, x))))
    return groups


def _draw_sweep(results: Results, kind: str) -> Any:
    kob = [c for c in _cols(results, "ko_diff", "d_ko_") if c[len("d_ko_") :].isdigit()]
    fig, axes = style.new_figure(1, max(len(kob), 1), sharey=True)
    axes_list = list(np.atleast_1d(axes))
    if not kob:
        empty_panel(axes_list[0], "no local-vol reference in the selection")
        return fig
    s = _setup(results)
    kappa = s["nu sweep kappa"] if kind == "nu" else s["rho sweep kappa"]
    for ax, col in zip(axes_list, kob):
        groups = _sweep_groups(results.long("ko_diff", col), kind, kappa)
        if not groups:
            empty_panel(ax, f"no {kind} sweep point in the selection")
            continue
        ax.axhline(0.0, color=style.INK["spine"], linewidth=0.8)
        series_by(ax, "xv", groups)
        ax.set_title(f"KO var B={_b(col)}")
        ax.set_xlabel({"nu": "nu (omega = 2 nu)", "rho": "rho", "theta": "theta"}[kind])
        ax.set_ylabel("K_KO LSV - LV [vol pts]")
        ax.legend(fontsize=7)
    return fig


def _draw_order(results: Results) -> Any:
    fig, ax = style.new_figure()
    if not _has(results, "monitoring_order"):
        empty_panel(ax, "no monitoring-order measurement in this run")
        return fig
    long = results.long("monitoring_order", "D")
    groups = []
    for m in dict.fromkeys(r.split(" | ")[0] for r in long["row"]):
        sub = long[long["row"].str.startswith(f"{m} | ")]
        groups.append((m, sub.assign(xv=1.0 / axis(sub, "per_year"))))
    series_by(ax, "xv", groups)
    ax.set_xscale("log")
    if bool((long["value"] > 0).all()):
        ax.set_yscale("log")
    ax.set_xlabel("dt = 1 / fixings per year")
    ax.set_ylabel("curr - prev accrual D [vol pts^2]")
    ax.set_title("Current-close monitoring term against dt (a slope of 1 is O(dt))")
    ax.legend(fontsize=7)
    return fig


def figures(results: Results) -> list[FigureSpec]:
    return [
        FigureSpec(
            "strikes_vs_barrier",
            "Conditional up-var, down-var and knock-out variance fair strikes against the "
            "barrier per model, error bars 1 stderr.",
            _draw_strikes,
        ),
        FigureSpec(
            "gyongy_check",
            "LSV minus LV of the corridor, conditional up and conditional down strikes per "
            "barrier and model, error bars 1 paired stderr; the short horizontal ticks at +/- "
            "the tolerance mark each model's variance-swap floor at the leg's level (a Monte "
            "Carlo number whose own error is in the gyongy table, not drawn).",
            _draw_gyongy,
        ),
        FigureSpec(
            "monitoring_order",
            "The current-close monitoring term D against dt on log axes, error bars 1 paired "
            "stderr.",
            _draw_order,
        ),
        FigureSpec(
            "ko_vs_nu",
            "Knock-out variance swap, LSV minus LV, against nu at the sweep's kappa, one series "
            "per rho, error bars 1 paired stderr.",
            lambda r: _draw_sweep(r, "nu"),
        ),
        FigureSpec(
            "ko_vs_rho",
            "Knock-out variance swap, LSV minus LV, against rho at the sweep's kappa, one series "
            "per nu, error bars 1 paired stderr.",
            lambda r: _draw_sweep(r, "rho"),
        ),
        FigureSpec(
            "ko_vs_theta",
            "Knock-out variance swap, LSV minus LV, against theta over the 2F presets of the "
            "selection (the other parameters as in each preset), error bars 1 paired stderr.",
            lambda r: _draw_sweep(r, "theta"),
        ),
    ]


THEORY = """### What Gyongy pins

1. **Continuous monitoring.** The LSV shares the local vol's marginals (E[sigma_t^2 | S_t] =
sigma_loc^2(t, S_t)), so a leg accruing the integral of f(t, S_t) sigma_t^2 dt, with a weight
that is a function of (t, S_t), has the same expectation in both models. This covers the up-,
down- and corridor variance, the variance swap and the counts, hence the conditional strikes.
They are pinned exactly.
2. **Daily fixings, previous-close indicator** (up-var, corridor). The conditional expectation
of r_i^2 given the previous close is sigma^2 dt + O(dt^2), with a model-dependent O(dt^2)
term. So the leg is pinned up to O(dt) in aggregate, plus the leverage calibration residual and
the scheme's bias. The variance swap follows the same rule, and its LSV-minus-LV residual is
the floor every leg is judged against.
3. **Daily fixings, current-close indicator** (down-var). The indicator of the end of the
return adds, per step, -s^3 p(0) E[Z^3] - (1/2) s^4 p'(0) E[Z^4] + ... (s^2 = sigma_loc^2 dt,
Z the standardised one-step return, p the density of ln(B / S) at the barrier). The O(sqrt dt)
boundary layer integrates to -E[Z^3], which is itself O(sqrt dt). The term is therefore O(dt^2)
per step and O(dt) in aggregate, not O(sqrt dt). Its coefficients (the conditional skewness and
the kurtosis 3 E[sigma^4 | S] / E[sigma^2 | S]^2) are not fixed by the marginals: this is a
model-dependent O(dt) term on top of the previous-close one.
4. **The knock-out swap** depends on the hitting time, not on the marginals, and is never
pinned.

The verdict per cell: 0 = |z| <= 2 (paired stderr); 1 = significant but within the model's
variance-swap floor, |d(K_var^2)| + 2 stderr converted to the leg's strike; 2 = beyond that
floor. The floor is a heuristic, not a bound: a residual concentrated in part of the spot range
can move one leg by more than the total. A 0 or a 1 does not prove the invariance below the
floor, and a 2 is a flag, not a proof of non-invariance."""


def _verdict_counts(results: Results) -> list[str]:
    long = results.long("gyongy", "verdict")
    long = long.assign(ind=axis_text(long, "indicator"), leg=axis_text(long, "leg"))
    z_long = results.long("gyongy", "z")
    lines: list[str] = []
    for leg, label, indicator in LEGS:
        sub = long[long["leg"] == leg]
        v = sub["value"].to_numpy(float)
        zs = np.abs(z_long[axis_text(z_long, "leg") == leg]["value"].to_numpy(float))
        counts = [int((v == k).sum()) for k in (0, 1, 2)]
        worst = float(np.nanmax(zs)) if zs.size and np.isfinite(zs).any() else math.nan
        lines.append(
            f"- {label} ({indicator} close): {len(v)} cells, {counts[0]} consistent with zero, "
            f"{counts[1]} significant but within the floor, {counts[2]} beyond the floor; "
            f"max |z| {worst:.1f}."
        )
    return lines


def _beyond(results: Results) -> list[str]:
    piv = results.pivot("gyongy")
    rows = [r for r in piv.index if piv.loc[r, "verdict"] == 2.0]
    return [
        f"{r}: {piv.loc[r, 'd']:+.3f} ± {piv.loc[r, 'd_stderr']:.3f} vp "
        f"(z {piv.loc[r, 'z']:+.1f}, tolerance {piv.loc[r, 'tolerance']:.3f} vp)"
        for r in rows
    ]


def narrative(results: Results) -> str:
    s = _setup(results)
    grid = lv_grid_sentence(results)
    lines = [
        "## Data",
        "",
        f"{int(s['models'])} models, one path set each ({int(s['paths per model'])} paths, "
        f"daily fixings over {s['maturity [y]']:g}y), priced from the cached leverage "
        "(`source = cache:<key>`; the local vol is a Dupire build, `computed`) on one shared "
        "step schedule and seed, so every LSV-minus-LV difference is paired path by path"
        + (f"; {grid}" if grid else "")
        + ". Nothing was calibrated. The store keeps four of these numbers (up-var and down-var "
        "at 100%, the knock-out swap at 110% and its P(KO)); the rest is computed here.",
        "",
        *(["{{table:pairing}}", ""] if _has(results, "pairing") else []),
        "{{table:strikes_up}}",
        "",
        "{{table:strikes_down}}",
        "",
        "{{figure:strikes_vs_barrier}}",
        "",
        "## The Gyongy check",
        "",
        THEORY,
        "",
    ]
    if _has(results, "monitoring_order"):
        piv = results.pivot("monitoring_order")
        ratios = piv[np.isfinite(piv["ratio"].to_numpy(float))]
        vals = ratios["ratio"].to_numpy(float)
        errs = ratios["ratio_stderr"].to_numpy(float)
        parts = [
            f"{r} {format_value_text(float(v), float(e))}"
            for r, v, e in zip(ratios.index, vals, errs, strict=True)
        ]
        dt_ok = int((np.abs(vals - ORDER_RATIOS["O(dt)"]) <= Z_THRESHOLD * errs).sum())
        z_sqrt = np.abs(vals - ORDER_RATIOS["O(sqrt dt)"]) / errs
        excl_sqrt = int((z_sqrt > Z_THRESHOLD).sum())
        far = float(np.max(np.abs(vals - ORDER_RATIOS["O(dt)"]))) if vals.size else math.nan
        lines += [
            "**The monitoring term, measured.** Halving dt multiplies D by: "
            + "; ".join(parts)
            + f". {excl_sqrt} of {vals.size} ratios exclude the O(sqrt dt) value 0.71 at "
            f"{Z_THRESHOLD:g} stderr (the nearest by {float(np.min(z_sqrt)):.0f} stderr); "
            f"{dt_ok} of {vals.size} are within {Z_THRESHOLD:g} stderr of the O(dt) value 0.5, "
            f"and the largest distance from 0.5 is {far:.3f}.",
            "",
            "{{table:monitoring_order}}",
            "",
            "{{table:monitoring_diff}}",
            "",
            "{{figure:monitoring_order}}",
            "",
        ]
    if _has(results, "gyongy"):
        fl = results.pivot("gyongy_floor")
        lines += [
            "**The floor.** The daily variance swap's LSV-minus-LV residual per model: "
            + "; ".join(
                f"{r} {fl.loc[r, 'd_var']:+.3f} ± {fl.loc[r, 'd_var_stderr']:.3f} vp "
                f"(z {fl.loc[r, 'z_var']:+.1f})"
                for r in fl.index
            )
            + ".",
            "",
            "{{table:gyongy_floor}}",
            "",
            "**The verdicts**, per leg family:",
            "",
            *_verdict_counts(results),
            "",
        ]
        beyond = _beyond(results)
        if beyond:
            lines += [
                f"{len(beyond)} cell(s) exceed the floor, so the invariance is not confirmed "
                "at this precision for them:",
                "",
                *[f"- {x}" for x in beyond],
                "",
            ]
        else:
            lines += [
                "No cell exceeds its floor: no departure from the Gyongy prediction is resolved "
                "beyond the variance swap's own residual at this precision. This does not prove "
                "the invariance below that floor.",
                "",
            ]
        cells = results.pivot("gyongy")
        prev = cells[[results.axes("gyongy", r).get("indicator") == "prev" for r in cells.index]]
        sig = prev[prev["verdict"] > 0.0]
        if len(sig):
            lines += [
                f"{len(sig)} of {len(prev)} previous-close cells differ from the local vol by "
                f"more than {Z_THRESHOLD:g} paired stderr ({int((sig['d'] > 0).sum())} of "
                f"them positive; largest |d| {float(sig['d'].abs().max()):.3f} vol points): at "
                "these fixings and this calibration the LSV does not reproduce the local vol's "
                "strikes exactly, and every such difference is within its model's "
                "variance-swap floor"
                + ("" if not beyond else " except the cells listed above")
                + ".",
                "",
            ]
        comparison = quadrature_comparison(
            cells["d_stderr"].to_numpy(float),
            cells["d_se_quadrature"].to_numpy(float),
            cells["d_correlation"].to_numpy(float),
        )
        if comparison:
            lines += [comparison, ""]
        lines += [
            "{{table:gyongy}}",
            "",
            "{{table:gyongy_se}}",
            "",
            "{{figure:gyongy_check}}",
            "",
        ]
    lines += [
        "## Convexity spread",
        "",
        "{{table:corridor}}",
        "",
        "{{table:convexity}}",
        "",
        "## The knock-out variance swap",
        "",
        f"Recorded as asked: {KO_NOT_A_THEOREM} It is a survival-weighted average of the local "
        "variance over the lower spot states; it can flip for a symmetric smile or an inverted "
        "term structure, and the model dependence enters through the hitting probabilities.",
        "",
    ]
    kmv = [c for c in results.columns("ko_var") if c.startswith("ko_minus_var_")]
    piv = results.pivot("ko_var")
    holds = 0
    total = 0
    for c in kmv:
        vals = piv[c].to_numpy(float)
        errs = piv[f"{c}_stderr"].to_numpy(float)
        ok = np.isfinite(vals) & np.isfinite(errs)
        total += int(ok.sum())
        holds += int((vals[ok] > 3.0 * errs[ok]).sum())
    lines += [
        f"On this surface K_KO > K_var at 3 stderr (paired) in {holds} of {total} "
        "(model, barrier) cells.",
        "",
        "{{table:ko_strike}}",
        "",
        "{{table:ko_minus_var}}",
        "",
        "{{table:ko_probability}}",
        "",
    ]
    if _has(results, "ko_diff"):
        lines += ["{{table:ko_diff}}", "", "{{table:ko_diff_p}}", ""]
        for kind in ("nu", "rho", "theta"):
            n = len(_sweep_rows(results, "ko_diff", kind))
            lines.append(f"- strict {kind} sweep: {n} LSV model(s) in this run.")
        lines += ["", "{{figure:ko_vs_nu}}", "", "{{figure:ko_vs_rho}}", ""]
        lines += ["{{figure:ko_vs_theta}}", ""]
    lines += _store_check_lines(results)
    return "\n".join(lines)


def _store_check_lines(results: Results) -> list[str]:
    checks = [t for t in ("store_check_vol", "store_check_ratio") if _has(results, t)]
    if not checks:
        return []
    frames = {t: results.long(t, "diff") for t in checks}
    diffs = pd.concat(list(frames.values()), ignore_index=True)
    identical = int(diffs["exact"].sum())
    other = diffs[~diffs["exact"]]
    n_se = np.abs(other["value"].to_numpy(float)) / other["stderr"].to_numpy(float)
    text = (
        f"Against the store's M4c conditional set, {identical} of {len(diffs)} inline numbers "
        "are identical to the stored ones (the same computation)"
    )
    if len(other):
        text += (
            f"; the other {len(other)} differ by at most {float(np.nanmax(n_se)):.2f} times "
            "their quadrature error, which is not the exact error (the estimates share random "
            "numbers)"
        )
    shifts = []
    for t, frame in frames.items():
        unit = " vp" if t == "store_check_vol" else ""
        lv = frame[~frame["exact"] & frame["row"].str.startswith(f"{LV_LABEL} / ")]
        shifts += [
            f"{str(r['row']).split(' / ', 1)[1]} "
            f"{format_value_text(float(r['value']), float(r['stderr']))}{unit}"
            for r in records(lv)
        ]
    if shifts:
        grid = lv_grid_sentence(results)
        text += (
            f". The {len(shifts)} local-vol rows among them differ for a second reason: "
            + (grid if grid else "the local vol is stepped on another grid than the store's")
            + "; their inline-minus-store shifts are "
            + "; ".join(shifts)
        )
    lines = ["## Store check", "", text + ".", ""]
    for t in checks:
        lines += [f"{{{{table:{t}}}}}", ""]
    return lines
