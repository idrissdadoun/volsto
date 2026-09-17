"""M10 Part 3 (G3): P&L attribution for a daily backtest (``volsto.risk.attribution``, SPEC
§7.12.1).

``explain(mode="sticky_leverage")`` prices every intermediate state and every Greek on state_0's
leverage held in forward log-moneyness and books the endpoint's refit as the ``recalibration``
bucket: the builder counts no calibration but the two endpoints (and the ``"recalibrate"`` run
of the same move needs the intermediate ones, which the cache lacks); the steps and the buckets
sum to the total P&L; a pure spot move leaves the recalibration bucket at zero within its
stderr — on synthetic leverages and, when the repository cache holds them, on the real
recalibrated pair of the M7 SPX marking fit — while the spot-fixed re-anchoring of the builder's
``"sticky_leverage"`` mode would book a significant leverage move there; ``product_1`` replaces
``product.aged(dt)`` in the time step; the dry-run cost matches the real run's counters.

Synthetic leverages are written to a temporary cache (``LeverageCache.store``; nothing is
calibrated); the real pair is read with ``allow_calibrate=False``.  Monte Carlo at ≤ 2·10⁴
paths.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import numpy as np
import pytest

from volsto.calibration.cache import CacheMissError, LeverageCache, build_market
from volsto.config import (
    BergomiParams,
    CalibrationSpec,
    CurveConfig,
    MarketConfig,
    ParticleConfig,
    SimConfig,
    SSVIConfig,
    SurfacePerturbation,
    from_mapping,
)
from volsto.models.leverage import LeverageFunction
from volsto.models.lsv import LSV
from volsto.products import (
    EuropeanOption,
    KnockOutVarianceSwap,
    RealisedHistory,
    Settled,
    VarianceSwap,
    daily_schedule,
    season,
)
from volsto.risk.attribution import attribution_cost, explain
from volsto.risk.engine import BSBuilder, LSVBuilder, RiskEngine, RiskState

ROOT = Path(__file__).resolve().parents[1]
REPO_CACHE = ROOT / "cache"
SPX_BASE_KEY = "6f6302bac31c7683dcc0a6fdaac59085f37aa5650ad0256afe4994564232b615"
"""The M7 P1 marking fit on SPX 2022-12-30 (``configs/studies/m7_p1_marking/spx_ssr1_eps0.1``)."""
SPX_UP_KEY = "2a4de54d48db"
"""Prefix of its sticky-moneyness ``+1%`` delta state, calibrated by the M7 runs."""
PILLARS = (1 / 12, 0.25, 0.5)
SIM = SimConfig(n_paths=10_000, chunk_size=10_000, seed=17)


def _spec(spot: float = 100.0) -> CalibrationSpec:
    return CalibrationSpec(
        market=MarketConfig(spot, CurveConfig((1.0,), (0.02,)), CurveConfig((1.0,), (0.01,))),
        surface=SSVIConfig(PILLARS, (0.22, 0.21, 0.205), -0.7, 1.0, 0.5, max_maturity=0.5),
        model=BergomiParams.one_factor(2.0, 1.5, -0.7),
        particle=ParticleConfig(n_particles=20_000, horizon=0.5),
    )


def _leverage(spec: CalibrationSpec, scale: float = 1.0, slope: float = -0.8) -> LeverageFunction:
    """A skewed synthetic leverage ``clip(1 + slope·k, 0.4, 2.5)·(1 + 0.2 t)·scale`` on the
    spec's forward curve (a stand-in for a calibrated one: nothing is calibrated here)."""
    fc, _, _ = build_market(spec)
    t = np.linspace(0.0, 0.5, 7)
    k = np.linspace(-1.5, 1.5, 241)
    values = np.clip(1.0 + slope * k[None, :], 0.4, 2.5) * (1.0 + 0.2 * t[:, None]) * scale
    return LeverageFunction(t, k, values, fc, {"synthetic": True})


def _cache(
    tmp_path: Path, entries: list[tuple[CalibrationSpec, LeverageFunction]]
) -> LeverageCache:
    cache = LeverageCache(tmp_path / "cache")
    for spec, lev in entries:
        cache.store(spec, lev)
    return cache


def _call(state: RiskState) -> EuropeanOption:
    return EuropeanOption(100.0, 0.25, 1, build_market(state.spec)[0].rate_curve)


def _full_move(state_0: RiskState) -> RiskState:
    """Spot, rates, surface and a model parameter all move."""
    return (
        state_0.with_spot(101.0)
        .with_rate_shift(dr=0.001)
        .with_perturbation(SurfacePerturbation("parallel", {"size": 0.005}))
        .with_params(nu=1.05)
    )


def test_sticky_leverage_calibrates_only_the_endpoints(tmp_path: Path) -> None:
    state_0 = RiskState(_spec())
    state_1 = _full_move(state_0)
    lev0 = _leverage(state_0.spec)
    lev1 = _leverage(state_1.spec, scale=1.03)  # the endpoint's "refit"
    cache = _cache(tmp_path, [(state_0.spec, lev0), (state_1.spec, lev1)])
    builder = LSVBuilder(cache, state_0, allow_calibrate=False)
    engine = RiskEngine(builder, SIM)
    call = _call(state_0)
    ex = explain(engine, call, state_0, state_1, mode="sticky_leverage")
    n_pricings = engine.n_pricings
    assert [s.name for s in ex.steps] == ["spot", "rates", "surface", "params", "recalibration"]
    assert builder.cache_keys == [state_0.key, state_1.key] and ex.end_key == state_1.key
    assert builder.n_calibrations == 2 and builder.n_cache_misses == 0
    # the sums: steps telescope to the total, buckets sum to it, the residual is the Greeks'
    total = ex.total.value
    assert sum(s.actual for s in ex.steps) == pytest.approx(total, abs=1e-12)
    buckets = ex.buckets()
    assert set(buckets) == {
        "spot.delta",
        "spot.gamma",
        "rates.rho",
        "rates.repo",
        "surface.parallel_vega",
        "params.nu",
        "recalibration",
        "residual",
    }
    assert sum(buckets.values()) == pytest.approx(total, abs=1e-12)
    recal = ex.steps[-1]
    assert recal.residual == 0.0 and buckets["recalibration"] == recal.actual == ex.recalibration
    assert ex.residual == pytest.approx(sum(s.residual for s in ex.steps[:-1]), abs=1e-12)
    assert ex.price_1 == engine.price(call, state_1).mean
    assert ex.price_0 == engine.price(call, state_0).mean
    # the refit leverage is 3% higher: the bucket is its price effect, the paired difference
    p1 = engine.price(call, state_1).mean
    p1_frozen = engine.price(call, state_1, "frozen_leverage").mean
    assert recal.actual == pytest.approx(p1 - p1_frozen, abs=1e-12)
    assert recal.actual > 10 * recal.actual_stderr > 0
    frame = ex.as_frame()
    print(frame.to_string())
    print(f"budget {engine.budget()}")
    # the recalibrated attribution of the same move needs the intermediate calibrations
    with pytest.raises(CacheMissError):
        explain(
            RiskEngine(LSVBuilder(cache, state_0, allow_calibrate=False), SIM),
            call,
            state_0,
            state_1,
        )
    # the dry run projects this run's counters exactly, and the recalibrated run's cost
    cost = attribution_cost(call, state_0, state_1, SIM, mode="sticky_leverage")
    assert cost.intermediate_calibrations == 0 and cost.calibrations == 2
    assert cost.pricings == n_pricings and engine.n_pricings == n_pricings
    assert cost.steps == tuple(s.name for s in ex.steps)
    heavy = attribution_cost(call, state_0, state_1, SIM, mode="recalibrate")
    ladders = attribution_cost(
        call, state_0, state_1, SIM, mode="recalibrate", detail="ladders", pillars=PILLARS
    )
    sticky_ladders = attribution_cost(
        call, state_0, state_1, SIM, mode="sticky_leverage", detail="ladders", pillars=PILLARS
    )
    print(
        "cost per date: "
        + json.dumps([c.as_dict() for c in (cost, heavy, ladders, sticky_ladders)])
    )
    assert heavy.intermediate_calibrations > 0 and ladders.calibrations > heavy.calibrations
    assert sticky_ladders.intermediate_calibrations == 0
    assert cost.wall_clock(140.0, 1.0) == 2 * 140.0 + cost.pricings
    assert cost.wall_clock(140.0, 1.0, cached=[state_0.key]) == 140.0 + cost.pricings


def test_sticky_leverage_ladders_run_without_calibration(tmp_path: Path) -> None:
    state_0 = RiskState(_spec())
    state_1 = state_0.with_perturbation(
        SurfacePerturbation("skew_tent", {"pillars": list(PILLARS), "index": 1, "slope": 0.02})
    )
    cache = _cache(
        tmp_path,
        [(state_0.spec, _leverage(state_0.spec)), (state_1.spec, _leverage(state_1.spec, 0.99))],
    )
    builder = LSVBuilder(cache, state_0, allow_calibrate=False)
    engine = RiskEngine(builder, SimConfig(n_paths=4_000, chunk_size=4_000, seed=17))
    call = _call(state_0)
    ex = explain(
        engine, call, state_0, state_1, mode="sticky_leverage", detail="ladders", pillars=PILLARS
    )
    assert [s.name for s in ex.steps] == ["surface", "recalibration"]
    assert builder.n_calibrations == 2
    assert set(ex.steps[0].detail) == {"vega_T", "skew_T", "curvature_T"}
    assert sum(ex.buckets().values()) == pytest.approx(ex.total.value, abs=1e-12)


def test_recalibration_bucket_vanishes_on_a_pure_spot_move(tmp_path: Path) -> None:
    """A pure sticky-moneyness spot move whose stored refit is ``L₀`` held in ``k`` — what the
    particle method's homogeneity in ``k`` produces (measured on genuinely refitted leverages by
    :func:`test_recalibration_bucket_on_the_real_recalibrated_spot_pair`, not here).  The refit
    is built here without :func:`held_in_moneyness` (``L₀``'s grid values on the moved forward
    curve), so the two legs of the bucket — the cached refit and the attribution's frozen
    leverage — are the same model only if the frozen leverage is held in ``k``: the bucket is
    then exactly zero with a zero stderr (an identity, not a Monte Carlo statement).  A frozen
    leverage re-anchored in spot would book the spot-fixed difference measured below (z ≈ 100),
    which is why the attribution holds ``L₀`` in ``k``; the frozen model's leverage is also
    checked against ``L₀``'s values and against the spot re-anchoring directly."""
    state_0 = RiskState(_spec())
    state_1 = state_0.with_spot(102.0)
    lev0 = _leverage(state_0.spec)
    fc1, _, _ = build_market(state_1.spec)
    refit = LeverageFunction(lev0.times, lev0.k_grid, lev0.values, fc1, {"refit": "L0 in k"})
    cache = _cache(tmp_path, [(state_0.spec, lev0), (state_1.spec, refit)])
    builder = LSVBuilder(cache, state_0, allow_calibrate=False)
    engine = RiskEngine(builder, SIM)
    call = _call(state_0)
    ex = explain(engine, call, state_0, state_1, mode="sticky_leverage")
    assert [s.name for s in ex.steps] == ["spot", "recalibration"]
    recal = ex.steps[-1]
    # the stored refit IS L0 held in k: both legs are one model, path by path
    assert recal.actual == 0.0 and recal.actual_stderr == 0.0
    # the frozen model's leverage, independently of held_in_moneyness: L0's values on the moved
    # forward curve (L held in k), not the spot re-anchoring of the builder's sticky mode
    frozen = builder.build(state_1, "frozen_leverage")
    assert isinstance(frozen, LSV)
    np.testing.assert_array_equal(frozen.leverage.values, lev0.values)
    assert np.array_equal(frozen.leverage.times, lev0.times)
    assert float(frozen.leverage(0.0, 102.0)) == float(lev0(0.0, 100.0))  # at the money: k = 0
    reanchored = lev0.reanchored(frozen.kernel.forward_curve).values
    assert float(np.max(np.abs(reanchored - lev0.values))) > 1e-3
    sticky = builder.build(state_1, "sticky_leverage")
    assert isinstance(sticky, LSV)
    np.testing.assert_array_equal(sticky.leverage.values, reanchored)
    spot_fixed = engine.combination(
        "recal vs spot-fixed leverage",
        call,
        [(state_1, "recalibrate", 1.0), (state_1, "sticky_leverage", -1.0)],
        unit="price",
        size=0.0,
        scheme="revaluation",
    )
    print(
        f"pure +2% spot move: bucket {recal.actual:.3e} ± {recal.actual_stderr:.1e} (L held in k); "
        f"spot-fixed re-anchoring would book {spot_fixed.value:.5f} ± {spot_fixed.stderr:.1e} "
        f"(z {spot_fixed.z():.1f}); spot step {ex.steps[0].actual:.5f}, total {ex.total.value:.5f}"
    )
    assert abs(spot_fixed.z()) > 5.0
    # the recalibrated attribution of a pure spot move needs the moved state's refit only for
    # the spot step; its delta / gamma states would calibrate (not cached)
    with pytest.raises(CacheMissError):
        explain(
            RiskEngine(LSVBuilder(cache, state_0, allow_calibrate=False), SIM),
            call,
            state_0,
            state_1,
        )


def _spx_pair() -> tuple[RiskState, RiskState] | None:
    base = REPO_CACHE / SPX_BASE_KEY
    if not (base / "spec.json").is_file():
        return None
    spec = from_mapping(CalibrationSpec, json.loads((base / "spec.json").read_text()))
    state_0 = RiskState(spec)
    state_1 = state_0.with_spot(spec.market.spot * float(np.exp(0.01)))
    cache = LeverageCache(REPO_CACHE)
    if state_0.key != SPX_BASE_KEY or not state_1.key.startswith(SPX_UP_KEY):
        return None
    if not (cache.has(state_0.spec) and cache.has(state_1.spec)):
        return None
    return state_0, state_1


def test_recalibration_bucket_on_the_real_recalibrated_spot_pair() -> None:
    """The repository cache's genuinely recalibrated ``+1%`` sticky-moneyness state of the SPX
    marking fit (2F LSV, 2·10⁵ particles): the bucket is zero within 2 stderr (read-only,
    ``allow_calibrate=False``; skipped without the cache)."""
    pair = _spx_pair()
    if pair is None:
        pytest.skip("the M7 SPX delta-regime pair is not in the repository cache")
    state_0, state_1 = pair
    cache = LeverageCache(REPO_CACHE)
    builder = LSVBuilder(cache, state_0, allow_calibrate=False)
    engine = RiskEngine(builder, SimConfig(n_paths=10_000, chunk_size=10_000, seed=5))
    fc0 = build_market(state_0.spec)[0]
    call = EuropeanOption(state_0.spot, 0.25, 1, fc0.rate_curve)
    lev0 = cache.load(state_0.spec).values
    lev1 = cache.load(state_1.spec).values
    rel = float(np.max(np.abs(lev1 - lev0) / lev0))
    ex = explain(engine, call, state_0, state_1, mode="sticky_leverage")
    recal = ex.steps[-1]
    spot_fixed = engine.combination(
        "spot-fixed",
        call,
        [(state_1, "recalibrate", 1.0), (state_1, "sticky_leverage", -1.0)],
        unit="price",
        size=0.0,
        scheme="revaluation",
    )
    print(
        f"SPX 2F LSV, +1% sticky-moneyness: max |L1/L0 - 1| in k = {rel:.2e}; bucket "
        f"{recal.actual:.3e} ± {recal.actual_stderr:.1e}; spot-fixed {spot_fixed.value:.4f} ± "
        f"{spot_fixed.stderr:.1e}; call {ex.price_0:.3f} -> {ex.price_1:.3f}; "
        f"calibrations {builder.n_calibrations}"
    )
    assert builder.n_calibrations == 2
    assert abs(recal.actual) <= 2.0 * recal.actual_stderr + 1e-9 * abs(ex.price_0)
    assert rel < 1e-9


def test_product_1_replaces_the_aged_product() -> None:
    """A seasoned daily variance swap from date n to n + 1 under Black–Scholes: the time step and
    the end price use ``product_1`` (the day-(n+1) close realised), the theta ``product_theta``
    (the held close); ``product.aged`` alone would raise for the daily fixing."""
    rng = np.random.default_rng(3)
    n = 40
    closes = 100.0 * np.exp(np.cumsum(np.concatenate([[0.0], 0.012 * rng.standard_normal(n + 1)])))
    days = [
        d.astype(dt.date) for d in np.busday_offset("2022-07-01", np.arange(n + 2), roll="forward")
    ]
    hist = RealisedHistory(days[0], tuple(days), tuple(closes))
    held = RealisedHistory(days[0], tuple(days[: n + 1]), tuple(closes[: n + 1])).extended(
        days[n + 1], closes[n]
    )
    base = CalibrationSpec(
        market=MarketConfig(
            float(closes[n]), CurveConfig((1.0,), (0.02,)), CurveConfig((1.0,), (0.01,))
        ),
        surface=SSVIConfig((0.25, 0.5, 1.0), (0.2, 0.2, 0.2), 0.0, 0.0, 0.5, max_maturity=2.0),
        model=BergomiParams.one_factor(1.0, 1.0, 0.0),
    )
    state_0 = RiskState(base)
    state_1 = state_0.with_spot(float(closes[n + 1]))
    disc = build_market(base)[0].rate_curve
    swap = VarianceSwap.daily(1.0, 0.04, disc, annualisation=252.0)
    product_0 = season(swap, hist, days[n])
    product_1 = season(swap, hist, days[n + 1])
    product_theta = season(swap, held, days[n + 1])
    assert isinstance(product_0, VarianceSwap) and isinstance(product_1, VarianceSwap)
    assert isinstance(product_theta, VarianceSwap)
    assert product_theta.realised_sum_sq == product_0.realised_sum_sq  # a zero day-(n+1) return
    engine = RiskEngine(BSBuilder(state_0), SimConfig(n_paths=8_000, chunk_size=8_000, seed=9))
    dt_ = 1.0 / 252.0
    ex = explain(
        engine,
        product_0,
        state_0,
        state_1,
        dt=dt_,
        product_1=product_1,
        product_theta=product_theta,
    )
    assert [s.name for s in ex.steps] == ["spot", "time"]
    assert ex.price_1 == engine.price(product_1, state_1).mean
    time_step = ex.steps[1]
    direct = engine.paired(
        "time",
        [(product_1, state_1, "recalibrate", 1.0), (product_0, state_1, "recalibrate", -1.0)],
        unit="price",
        size=dt_,
        scheme="revaluation",
    )
    assert time_step.actual == direct.value and time_step.actual_stderr == direct.stderr
    assert sum(s.actual for s in ex.steps) == pytest.approx(ex.total.value, abs=1e-12)
    for s in ex.steps:
        print(
            f"{s.name}: actual {s.actual:.6f} ± {s.actual_stderr:.1e}, explained "
            f"{s.explained:.6f}, residual {s.residual:.2e}"
        )
    # the realised day-(n+1) variance sits in the spot step (the fixing starts from S_n)
    r = float(np.log(closes[n + 1] / closes[n]))
    assert ex.steps[0].actual == pytest.approx(float(disc.df(1.0 - n / 252)) * r * r, rel=0.05)
    assert time_step.explained < 0 and abs(time_step.residual) < 5.0 * time_step.actual_stderr + (
        0.1 * abs(time_step.explained)
    )
    # the sticky mode under Black–Scholes: the bucket is exactly zero
    sticky = explain(
        engine,
        product_0,
        state_0,
        state_1,
        dt=dt_,
        product_1=product_1,
        product_theta=product_theta,
        mode="sticky_leverage",
    )
    assert sticky.recalibration == 0.0 and sticky.total.value == ex.total.value
    # guards
    with pytest.raises(ValueError, match="product_theta"):
        explain(engine, product_0, state_0, state_1, dt=dt_, product_1=product_1)
    with pytest.raises(ValueError, match="dt > 0"):
        explain(engine, product_0, state_0, state_1, product_1=product_1)
    with pytest.raises(ValueError, match="state_0"):
        explain(engine, product_0, state_1, state_1, mode="sticky_leverage")
    with pytest.raises(ValueError, match="mode"):
        explain(engine, product_0, state_0, state_1, mode="model")


def test_ladders_on_a_repaired_essvi_day_need_more_halvings() -> None:
    """The dry run on a real repaired eSSVI day (M10 Part 0 history, git-ignored; skipped
    without it): the curvature ladder's butterfly bump is refused after the default 4 halvings
    and passes with 6 (module docstring of :mod:`volsto.risk.attribution`); the sticky run of the
    date still calibrates the endpoints only."""
    gate = ROOT / "outputs" / "essvi_gate" / "snapshots"
    d0, d1 = gate / "spx_2022-12-29.yaml", gate / "spx_2022-12-30.yaml"
    marking = ROOT / "configs" / "studies" / "m7_p1_marking" / "spx_ssr1_eps0.1.yaml"
    if not (d0.is_file() and d1.is_file()):
        pytest.skip("the repaired eSSVI history is not under outputs/essvi_gate")
    from volsto.config import load_yaml
    from volsto.market.loaders import snapshot_spec
    from volsto.market.surface import ArbitrageError

    ref = load_yaml(marking, CalibrationSpec, section="spec")
    state_0 = RiskState(snapshot_spec(ref, d0))
    state_1 = RiskState(snapshot_spec(ref, d1))
    assert state_0.spec.surface.rhos is not None and state_1.spec.surface.rhos is not None
    call = EuropeanOption(state_0.spot, 1.0, 1, build_market(state_0.spec)[0].rate_curve)
    with pytest.raises(ArbitrageError, match="curvature_tent"):
        attribution_cost(call, state_0, state_1, SIM, mode="sticky_leverage", detail="ladders")
    cost = attribution_cost(
        call, state_0, state_1, SIM, mode="sticky_leverage", detail="ladders", max_halvings=6
    )
    assert cost.intermediate_calibrations == 0 and cost.calibrations == 2
    assert cost.steps == ("spot", "rates", "surface", "recalibration")
    assert cost.detail == "ladders" and cost.mode == "sticky_leverage"


def test_a_settled_end_is_priced_as_its_cash() -> None:
    """A knock-out variance swap knocked out by the day-1 close: ``season`` returns a
    ``Settled`` for date 1, and ``explain`` prices it as its cash on the end state's curve
    (zero stderr of its own); the recalibration bucket of a settled end is zero.  A
    ``product_theta`` without ``product_1`` is refused (the verifier's reproducers)."""
    n = 10
    closes = np.concatenate([np.linspace(100.0, 108.0, n + 1), [111.0]])
    days = [
        d.astype(dt.date) for d in np.busday_offset("2022-07-01", np.arange(n + 2), roll="forward")
    ]
    hist = RealisedHistory(days[0], tuple(days), tuple(closes))
    held = RealisedHistory(days[0], tuple(days[: n + 1]), tuple(closes[: n + 1])).extended(
        days[n + 1], closes[n]
    )
    base = CalibrationSpec(
        market=MarketConfig(
            float(closes[n]), CurveConfig((1.0,), (0.02,)), CurveConfig((1.0,), (0.01,))
        ),
        surface=SSVIConfig((0.25, 0.5, 1.0), (0.2, 0.2, 0.2), 0.0, 0.0, 0.5, max_maturity=2.0),
        model=BergomiParams.one_factor(1.0, 1.0, 0.0),
    )
    state_0 = RiskState(base)
    state_1 = state_0.with_spot(float(closes[n + 1]))
    disc_1 = build_market(state_1.spec)[0].rate_curve
    ko = KnockOutVarianceSwap(daily_schedule(1.0, 252), 110.0, 0.2, disc_1)
    product_0 = season(ko, hist, days[n])
    settled = season(ko, hist, days[n + 1], discount=disc_1)
    product_theta = season(ko, held, days[n + 1])
    assert isinstance(product_0, KnockOutVarianceSwap) and isinstance(settled, Settled)
    assert isinstance(product_theta, KnockOutVarianceSwap)
    assert settled.pay_time > 0 and settled.value != 0.0
    engine = RiskEngine(BSBuilder(state_0), SimConfig(n_paths=8_000, chunk_size=8_000, seed=9))
    dt_ = 1.0 / 252.0
    for mode in ("recalibrate", "sticky_leverage"):
        ex = explain(
            engine,
            product_0,
            state_0,
            state_1,
            dt=dt_,
            product_1=settled,
            product_theta=product_theta,
            mode=mode,
        )
        assert ex.price_1 == pytest.approx(settled.value, rel=1e-14)
        time_step = next(s for s in ex.steps if s.name == "time")
        p0_at_1 = engine.price(product_0, state_1)
        assert time_step.actual == pytest.approx(settled.value - p0_at_1.mean, abs=1e-14)
        assert time_step.actual_stderr == pytest.approx(p0_at_1.stderr, rel=1e-9)
        assert sum(s.actual for s in ex.steps) == pytest.approx(ex.total.value, abs=1e-12)
        if mode == "sticky_leverage":
            recal = ex.steps[-1]
            assert recal.name == "recalibration" and recal.actual == 0.0
            assert recal.actual_stderr == 0.0
    cash = settled.as_product()
    priced = engine.price(cash, state_1)  # a constant: the stderr is the mean's round-off
    assert cash.is_seasoned and priced.stderr < 1e-15 * max(1.0, abs(priced.mean))
    # a settled held-spot product is accepted too (priced as its cash)
    held_settled = season(ko, hist, days[n + 1])
    assert isinstance(held_settled, Settled)
    ex = explain(
        engine, product_0, state_0, state_1, dt=dt_, product_1=settled, product_theta=held_settled
    )
    assert [s.name for s in ex.steps] == ["spot", "time"]
    with pytest.raises(ValueError, match="only with product_1"):
        explain(engine, product_0, state_0, state_1, dt=dt_, product_theta=product_theta)
