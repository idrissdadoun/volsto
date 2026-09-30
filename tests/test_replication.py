"""Static replications and the hedge comparison sets of the payoff study
(``volsto/hedging/replication.py``, ``volsto/hedging/comparison.py``): the strips reproduce their
payoffs, the put-call-symmetry portfolio is worth the Reiner–Rubinstein price and nothing on the
barrier under Black–Scholes (the carry power included), the Carr–Lewis strip prices the corridor
variance, the stopped log contract is worth ``L(B)`` on the barrier; the netted hedge object is
fitted on the product's alive paths; a knock-in's replication is unwound at the knock-in; the
semi-static barrier hedge beats the delta hedge end to end."""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from volsto.config import CurveConfig, MarketConfig, SimConfig
from volsto.engine.mc import MonteCarlo
from volsto.hedging import Costs, Hedger, PricingContext, Schedule, hedge_state
from volsto.hedging.comparison import comparison_strategies, delta_only
from volsto.hedging.instruments import StaticPortfolio
from volsto.hedging.pricing import union_grid
from volsto.hedging.replication import (
    barrier_replication,
    corridor_strip,
    reflection_power,
    stopped_log_strip,
    strip_for_payoff,
)
from volsto.hedging.state import NettedPortfolio
from volsto.hedging.strategies import GreekTargetStrategy, PresetContext, Target
from volsto.market.barrier_bs import bs_barrier_price
from volsto.market.bs import black_price
from volsto.market.curves import ForwardCurve
from volsto.models.bs import BlackScholes
from volsto.products.barrier import KnockInOption, KnockOutOption
from volsto.products.base import Portfolio, daily_schedule
from volsto.products.conditional_variance import DownVar, KnockOutVarianceSwap, UpVar
from volsto.products.variance import VarianceOption
from volsto.products.vko import VolKnockOutPut

FloatArray = NDArray[np.float64]

VOL = 0.2


def _market(r: float, q: float) -> ForwardCurve:
    return ForwardCurve.from_config(
        MarketConfig(100.0, CurveConfig((1.0,), (r,)), CurveConfig((1.0,), (q,)))
    )


def _ctx(fc: ForwardCurve) -> PresetContext:
    return PresetContext(fc, fc.rate_curve, 100.0, reference_vol=lambda k, t: VOL)


def _value(rep_strikes, rep_weights, rep_cps, fwd: float, tau: float, df: float) -> float:  # type: ignore[no-untyped-def]
    """Black value of a replication's options on a forward ``fwd`` with ``tau`` left."""
    k = np.asarray(rep_strikes)
    return float(
        np.sum(np.asarray(rep_weights) * black_price(fwd, k, tau, VOL, np.asarray(rep_cps), df))
    )


def _barrier(kind: str, fc: ForwardCurve, *, monitoring: str = "continuous", T: float = 1.0):  # type: ignore[no-untyped-def]
    disc = fc.rate_curve
    kw: dict = {"monitoring": monitoring}
    if monitoring == "discrete":
        kw |= {"fixing_times": np.linspace(0.0, T, round(52 * T) + 1), "strict": True}
    if kind == "dop":
        return KnockOutOption(100.0, T, -1, 90.0, "down", disc, **kw)
    if kind == "dip":
        return KnockInOption(100.0, T, -1, 90.0, "down", disc, **kw)
    return KnockOutOption(100.0, T, 1, 120.0, "up", disc, **kw)


def test_strip_for_payoff_reproduces_compact_payoffs() -> None:
    """The strip of a compact piecewise-linear payoff equals it away from the digital spreads,
    with puts below the forward and calls above."""

    def g(s: FloatArray) -> FloatArray:  # the down-and-out put's alive part, H = 90, K = 100
        return np.maximum(100.0 - s, 0.0) * (s > 90.0)

    rep = strip_for_payoff(g, [90.0, 100.0], 1.0, forward=101.0, digital_width=0.002)
    assert set(rep.cps) == {-1}
    s = np.linspace(50.0, 150.0, 20001)
    far = np.abs(s - 90.0) > 0.2
    assert np.max(np.abs(rep.payoff(s)[far] - g(s)[far])) < 1e-9
    # calls above the forward: the up-and-out call's alive part
    rep_c = strip_for_payoff(
        lambda x: np.maximum(x - 100.0, 0.0) * (x < 120.0), [100.0, 120.0], 1.0, forward=99.0
    )
    assert set(rep_c.cps) == {1}
    far = np.abs(s - 120.0) > 0.4
    assert (
        np.max(np.abs(rep_c.payoff(s)[far] - np.maximum(s - 100, 0)[far] * (s[far] < 120))) < 1e-9
    )
    with pytest.raises(ValueError, match="knots"):
        strip_for_payoff(g, [90.0], 1.0, forward=100.0)
    leg = rep.instrument(None, name="x")
    assert isinstance(leg, StaticPortfolio) and np.allclose(leg.terminal_payoff(s), rep.payoff(s))


@pytest.mark.parametrize("kind", ["dop", "uoc", "dip"])
@pytest.mark.parametrize("carry", [False, True])
def test_put_call_symmetry_replication_under_black_scholes(kind: str, carry: bool) -> None:
    """Continuous monitoring, flat 20% vol: the replication's value is the Reiner–Rubinstein
    price and it is worth nothing on the barrier (the knock-in's: the vanilla) — exactly without
    carry (``p = 1``, piecewise linear), to the strip's interpolation with carry (``r − q = 3%``:
    ``p = −0.5``); the symmetric replication misses under carry."""
    r, q = (0.03, 0.0) if carry else (0.02, 0.02)
    fc = _market(r, q)
    ctx = _ctx(fc)
    prod = _barrier(kind, fc)
    T = 1.0
    rep = barrier_replication(prod, ctx, carry=True, continuity=False, knots=24)
    p = reflection_power(float(fc.forward(T)), 100.0, T, VOL)
    assert rep.params["reflection_power"] == pytest.approx(p)
    assert p == pytest.approx(1.0 - 2.0 * (r - q) / VOL**2, rel=1e-9)
    ko_kind = "out"
    rr = float(
        bs_barrier_price(100.0, 100.0, prod.barrier, T, VOL, r, q, prod.cp, prod.direction, ko_kind)
    )
    v0 = _value(
        rep.strikes, rep.weights, rep.cps, float(fc.forward(T)), T, float(fc.rate_curve.df(T))
    )
    # measured: |v0 − RR| ≤ 5.3e-4 (the up-and-out call's digital spread at B and the chord
    # interpolation of the carry power), on the barrier ≤ 7e-4
    assert v0 == pytest.approx(rr, abs=1e-3), (kind, carry, v0, rr)
    # on the barrier at t = 0.5 (forward of the barrier level over the remaining half year)
    h = float(prod.barrier)
    fwd_h = h * float(np.exp((r - q) * 0.5))
    on_b = _value(rep.strikes, rep.weights, rep.cps, fwd_h, 0.5, float(np.exp(-r * 0.5)))
    assert abs(on_b) < 1.5e-3, (kind, carry, on_b)
    if kind == "dip":
        van = float(
            black_price(float(fc.forward(T)), 100.0, T, VOL, -1, float(fc.rate_curve.df(T)))
        )
        rr_in = float(bs_barrier_price(100.0, 100.0, h, T, VOL, r, q, -1, "down", "in"))
        assert van - v0 == pytest.approx(rr_in, abs=1e-3)
    if carry:
        sym = barrier_replication(prod, ctx, carry=False, continuity=False)
        on_sym = _value(sym.strikes, sym.weights, sym.cps, fwd_h, 0.5, float(np.exp(-r * 0.5)))
        assert abs(on_sym) > 5.0 * abs(on_b), (kind, on_sym, on_b)
    print(f"{kind} carry={carry}: replication {v0:.5f} vs RR {rr:.5f}; on the barrier {on_b:+.2e}")


def test_static_package_costs_its_net_vega() -> None:
    """The replication is traded as one structure: its half-spread is on the package's net vega,
    far below the leg-by-leg sum of the digital spreads' vegas."""
    from volsto.market.bs import black_vega

    fc = _market(0.02, 0.01)
    rep = barrier_replication(_barrier("dop", fc, monitoring="discrete"), _ctx(fc))
    leg = rep.instrument(fc.rate_curve, name="PCS replication", cost=0.25)
    leg.reference_vol = VOL
    spot = np.array([100.0])
    cost = float(leg.transaction_cost(np.array([1.0]), 0.0, spot, fc, fc.rate_curve)[0])
    f = float(fc.forward(1.0))
    vegas = np.array(
        [float(black_vega(f, k, 1.0, VOL, float(fc.rate_curve.df(1.0)))) for k in rep.strikes]
    )
    w = np.asarray(rep.weights)
    assert cost == pytest.approx(0.25 * 0.01 * abs(float(np.sum(w * vegas))), rel=1e-12)
    assert cost < 0.1 * 0.25 * 0.01 * float(np.sum(np.abs(w) * vegas))


def test_barrier_replication_continuity_and_scope() -> None:
    """Discrete monitoring moves the replicated barrier to the BGK level (down: lower, up:
    higher); the regular barriers raise."""
    fc = _market(0.02, 0.01)
    ctx = _ctx(fc)
    dop = _barrier("dop", fc, monitoring="discrete")
    uoc = _barrier("uoc", fc, monitoring="discrete")
    shift = np.exp(0.5826 * VOL * np.sqrt(1.0 / 52.0))
    assert barrier_replication(dop, ctx).params["barrier_replicated"] == pytest.approx(90 / shift)
    assert barrier_replication(uoc, ctx).params["barrier_replicated"] == pytest.approx(120 * shift)
    assert barrier_replication(dop, ctx, continuity=False).params["barrier_replicated"] == 90.0
    doc = KnockOutOption(100.0, 1.0, 1, 90.0, "down", fc.rate_curve, monitoring="continuous")
    with pytest.raises(NotImplementedError, match="reverse barriers"):
        barrier_replication(doc, ctx)


def test_corridor_strip_prices_the_corridor_variance() -> None:
    """Zero rates, flat vol: the Carr–Lewis strip of the up (down) variance above (below) the
    spot is worth ``σ² (1/T) ∫ P(S_t ≷ L) dt`` (midpoint strip, 4 sd truncation)."""
    fc = _market(0.0, 0.0)
    ctx = _ctx(fc)
    T = 1.0
    fix = daily_schedule(T)
    from scipy.stats import norm

    t = np.linspace(1e-4, T, 4001)
    d2 = (np.log(100.0 / 100.0) - 0.5 * VOL**2 * t) / (VOL * np.sqrt(t))
    p_up = np.trapezoid(norm.cdf(d2), t) / T
    for side, prob in (("up", p_up), ("down", 1.0 - p_up)):
        prod = (UpVar if side == "up" else DownVar)(
            fix, 100.0, 0.2, fc.rate_curve, convention="corridor"
        )
        rep = corridor_strip(prod, ctx, spacing=0.01)
        v = _value(rep.strikes, rep.weights, rep.cps, 100.0, T, 1.0)
        assert v == pytest.approx(VOL**2 * prob, rel=0.01), (side, v, VOL**2 * prob)


def test_stopped_log_strip_is_worth_the_log_contract_on_the_barrier() -> None:
    """Zero rates, flat vol: the knock-out variance swap's stopped log strip is worth ``L(B)/T``
    on the barrier at any remaining time (symmetric reflection) and less than the variance swap's
    ``σ²`` at inception."""
    fc = _market(0.0, 0.0)
    ctx = _ctx(fc)
    T = 1.0
    b = 105.0
    ko = KnockOutVarianceSwap(daily_schedule(T), b, 0.2, fc.rate_curve, settlement="knock_out")
    rep = stopped_log_strip(ko, ctx, spacing=0.005, width_sigmas=6.0)
    l_b = -2.0 * np.log(b / 100.0) + 2.0 * (b - 100.0) / 100.0
    for tau in (0.25, 0.5, 0.9):
        v = _value(rep.strikes, rep.weights, rep.cps, b, tau, 1.0)
        assert v == pytest.approx(l_b / T, rel=0.03, abs=2e-5), (tau, v, l_b / T)
    v0 = _value(rep.strikes, rep.weights, rep.cps, 100.0, T, 1.0)
    assert 0.0 < v0 < VOL**2


def test_netted_portfolio_is_fitted_on_the_products_alive_paths() -> None:
    """A knocked-out product netted with a vanilla leg: the plain portfolio is alive on every
    path, the netted object only where the product is."""
    fc = _market(0.02, 0.01)
    bs = BlackScholes(VOL, fc)
    dop = _barrier("dop", fc, monitoring="discrete", T=0.5)
    van = dop.vanilla()
    sim = SimConfig(n_paths=2_000, chunk_size=2_000, seed=1, dt_max=1.0 / 52.0)
    grid = union_grid([bs], [dop], np.array([0.25]), sim)
    paths = MonteCarlo(sim).simulate(bs, grid)
    idx = grid.fixing_index
    alive_ko = hedge_state(dop, paths, idx, 0.25).alive
    assert 0 < (~alive_ko).sum() < alive_ko.size
    assert hedge_state(Portfolio([dop, van], [1.0, -1.0]), paths, idx, 0.25).alive.all()
    net = NettedPortfolio([dop, van], [1.0, -1.0])
    assert np.array_equal(hedge_state(net, paths, idx, 0.25).alive, alive_ko)
    assert isinstance(net.aged(1.0 / 252.0), NettedPortfolio)


def test_unwind_on_knock_zeroes_the_leg_on_knocked_paths() -> None:
    """The ``unwind_on_knock`` static leg is held on the paths not knocked in and zero on the
    others; a leg that is not static raises."""
    fc = _market(0.02, 0.01)
    ctx = _ctx(fc)
    dip = _barrier("dip", fc)
    s = comparison_strategies(dip, ctx)["PCS static (carry, BGK)"]
    assert s.unwind_on_knock == ("PCS replication",)
    assert s.static["PCS replication"] == 1.0 and s.static["parity vanilla"] == -1.0
    n = 3
    g = {"delta": np.array([0.1, 0.2, 0.3]), "value": np.zeros(n)}
    ig = [{"delta": np.ones(n), "value": np.zeros(n)} for _ in s.instruments]

    class _State:
        names = ("knocked",)
        features = np.array([[0.0], [1.0], [0.0]])

    sol = s.solve(0.1, g, ig, [True] * len(s.instruments), state=_State())
    j = [i.name for i in s.instruments].index("PCS replication")
    assert np.allclose(sol.quantities[:, j], [1.0, 0.0, 1.0])
    k = [i.name for i in s.instruments].index("parity vanilla")
    assert np.allclose(sol.quantities[:, k], -1.0)
    with pytest.raises(ValueError, match="unwind_on_knock"):
        GreekTargetStrategy((Target("delta"),), [ctx.spot_instrument()], unwind_on_knock=("x",))


def test_comparison_sets_of_the_study_products() -> None:
    """Every study product gets the baselines plus its structural hedges, the replicating legs
    short (the knock-in's knock-out replication long, unwound at the knock-in)."""
    fc = _market(0.02, 0.01)
    ctx = _ctx(fc)
    disc = fc.rate_curve
    fix = daily_schedule(1.0)
    products = {
        "uoc": _barrier("uoc", fc, monitoring="discrete"),
        "dop": _barrier("dop", fc, monitoring="discrete"),
        "dip": _barrier("dip", fc, monitoring="discrete"),
        "put on var": VarianceOption.daily(1.0, 0.2, disc, annualisation=252.0),
        "ko var": KnockOutVarianceSwap(
            daily_schedule(0.5), 103.0, 0.2, disc, settlement="knock_out"
        ),
        "up var": UpVar(fix, 100.0, 0.2, disc, convention="corridor"),
        "down var": DownVar(fix, 100.0, 0.2, disc, convention="corridor"),
        "vko": VolKnockOutPut(100.0, 1.0, 0.3, fix, disc),
    }
    expected = {
        **{
            k: {
                "PCS static (symmetric) no delta",
                "PCS static (carry, BGK)",
                "PCS static (carry, BGK) no delta",
                "PCS static (carry, BGK) + vega",
            }
            for k in ("uoc", "dop", "dip")
        },
        "put on var": {"var + vol swap (vega, volga)"},
        "ko var": {"stopped log strip", "stopped log strip + var swap"},
        "up var": {"var swap (vega)", "corridor strip", "corridor strip + var swap"},
        "down var": {"var swap (vega)", "corridor strip", "corridor strip + var swap"},
        "vko": {"put static + delta"},
    }
    for name, prod in products.items():
        sets = comparison_strategies(prod, ctx)
        assert set(sets) == {"delta", "preset"} | expected[name], (name, set(sets))
        for label, s in sets.items():
            assert s.name == label or label == "preset", (name, label, s.name)
            for leg, qty in s.static.items():
                qty = float(qty)  # type: ignore[arg-type]
                if leg in s.unwind_on_knock or leg == "complementary corridor":
                    assert qty > 0.0, (name, label, leg)
                elif leg not in ("cap-call strip",):
                    assert qty < 0.0, (name, label, leg)
    assert delta_only(products["uoc"], ctx).target_names == ("delta",)
    # the four-target vanna-volga sets only on request
    assert "vanna-volga" in comparison_strategies(products["dop"], ctx, unstable=True)
    assert "var swap + vanna-volga" in comparison_strategies(
        products["put on var"], ctx, unstable=True
    )


SIM_E2E = SimConfig(n_paths=6_000, chunk_size=6_000, seed=11, dt_max=1.0 / 52.0)


@pytest.mark.parametrize("kind", ["dop", "dip"])
def test_semi_static_barrier_hedge_beats_the_delta_hedge(kind: str) -> None:
    """Black–Scholes pricing = world (carry 1%), weekly monitoring and rebalancing, 6·10³ paths:
    the put-call-symmetry static hedge alone (carry power, BGK barrier; unwound at the knock-out,
    for the knock-in switched to the parity vanilla at the knock-in) leaves a P&L std well below
    the delta hedge's."""
    fc = _market(0.02, 0.01)
    bs = BlackScholes(VOL, fc)
    prod = _barrier(kind, fc, monitoring="discrete", T=0.5)
    h = Hedger(
        PricingContext.from_model(bs),
        bs,
        Schedule("weekly"),
        Costs(),
        sim=SIM_E2E,
        world_paths=SIM_E2E.n_paths,
        verbose=False,
    )
    ctx = h.preset_context(prod)
    sets = comparison_strategies(prod, ctx)
    r_delta = h.run(prod, sets["delta"])
    r_pcs = h.run(prod, sets["PCS static (carry, BGK) no delta"])
    s_delta = float(np.std(r_delta.pnl_zero_cost, ddof=1))
    s_pcs = float(np.std(r_pcs.pnl_zero_cost, ddof=1))
    s_prod = float(np.std(r_delta.pnl_product, ddof=1))
    print(f"{kind}: product std {s_prod:.4f}, delta {s_delta:.4f}, PCS static {s_pcs:.4f}")
    # measured: 0.794 against 1.183 (down-and-out) and 1.557 (down-and-in; the same 0.794 as the
    # knock-out's by the in-out parity, path by path on the shared world)
    assert s_pcs < 0.75 * s_delta, (kind, s_pcs, s_delta)
    assert (
        abs(np.mean(r_pcs.pnl_zero_cost)) < 4.0 * s_pcs / np.sqrt(SIM_E2E.n_paths) + 0.02 * s_prod
    )
