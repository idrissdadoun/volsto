"""M4c: conditional / corridor / knock-out variance swaps and the volatility knock-out put
(SPEC v2 §6.1–6.2): path-wise identities and hand values, Black–Scholes fair strikes, orderings on
the negatively skewed reference surface under local vol (fast) and the calibrated LSV (slow),
Gyöngy invariance, VKO limits and the LSV-versus-LV VKO discount."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from volsto.analytics import (
    fair_strike,
    lsv_minus_lv,
    strike_differential,
    vko_analysis,
    vko_report,
)
from volsto.config import SimConfig
from volsto.engine import FixingIndex, MonteCarlo, PathSet
from volsto.market import DiscountCurve, ForwardCurve, LocalVolSurface, SSVISurface
from volsto.models import BlackScholes, LocalVol
from volsto.products import (
    ConditionalVarianceSwap,
    ConvexitySpread,
    DownVar,
    KnockOutVarianceSwap,
    UpVar,
    VarianceOption,
    VarianceSwap,
    VolKnockOutPut,
    VolSwap,
    daily_schedule,
)

ROOT = Path(__file__).resolve().parents[1]


def _paths(spots: np.ndarray, times: np.ndarray) -> PathSet:
    n, m = spots.shape
    return PathSet(
        times,
        np.log(spots),
        np.zeros((n, m)),
        np.zeros((n, m, 0)),
        np.zeros((n, m)),
        np.zeros((n, m)),
    )


def _random_paths(rng: np.random.Generator, n: int, times: np.ndarray) -> PathSet:
    steps = rng.normal(0.0, 0.012, size=(n, times.size - 1))
    spots = 100.0 * np.exp(np.concatenate([np.zeros((n, 1)), np.cumsum(steps, axis=1)], axis=1))
    return _paths(spots, times)


def test_conditional_swap_hand_values_and_validation(discount: DiscountCurve) -> None:
    times = np.array([0.0, 1 / 3, 2 / 3, 1.0])  # N = 3 returns, A = 3 -> A/N = 1
    spots = np.array([[100.0, 110.0, 90.0, 105.0]])
    ps = _paths(spots, times)
    idx = FixingIndex(times)
    r2 = np.diff(np.log(spots[0])) ** 2
    df = float(discount.df(1.0))
    k = 0.2
    kw = dict(annualisation=3.0)
    up_prev = ConditionalVarianceSwap(times, 100.0, "up", "prev", "conditional", k, discount, **kw)
    # S_0 = 100 is not > 100 (strict): I = [0, 1, 0]
    np.testing.assert_allclose(up_prev.payoff(ps, idx), [df * (r2[1] - k * k / 3)])
    up_prev_ge = ConditionalVarianceSwap(
        times, 100.0, "up", "prev", "conditional", k, discount, strict=False, **kw
    )
    np.testing.assert_allclose(up_prev_ge.payoff(ps, idx), [df * (r2[0] + r2[1] - 2 * k * k / 3)])
    up_curr = ConditionalVarianceSwap(times, 100.0, "up", "curr", "corridor", k, discount, **kw)
    np.testing.assert_allclose(up_curr.payoff(ps, idx), [df * (r2[0] + r2[2] - k * k)])
    up_both = ConditionalVarianceSwap(times, 100.0, "up", "both", "conditional", k, discount, **kw)
    np.testing.assert_allclose(up_both.payoff(ps, idx), [0.0])  # D = 0 -> zero payoff
    down_curr = DownVar(times, 100.0, k, discount, **kw)  # curr indicator: I = [0, 1, 0]
    np.testing.assert_allclose(down_curr.payoff(ps, idx), [df * (r2[1] - k * k / 3)])
    capped = ConditionalVarianceSwap(
        times, 100.0, "up", "curr", "corridor", 0.0, discount, daily_cap=0.05, **kw
    )
    np.testing.assert_allclose(
        capped.payoff(ps, idx), [df * (min(r2[0], 0.0025) + min(r2[2], 0.0025))]
    )
    st = up_curr.statistics(ps, idx)
    assert st["count"][0] == pytest.approx(2 / 3) and st["accrued"][0] == pytest.approx(
        r2[0] + r2[2]
    )
    assert "up-variance swap" in repr(up_prev) and "Corridor" in repr(up_curr)
    for bad in (
        lambda: ConditionalVarianceSwap(times, 100.0, "sideways", "prev", "corridor", k, discount),
        lambda: ConditionalVarianceSwap(times, 100.0, "up", "next", "corridor", k, discount),
        lambda: ConditionalVarianceSwap(times, 100.0, "up", "prev", "vanilla", k, discount),
        lambda: ConditionalVarianceSwap(times, -1.0, "up", "prev", "corridor", k, discount),
        lambda: UpVar(times, 100.0, k, discount, indicator="curr"),
        lambda: DownVar(times, 100.0, k, discount, indicator="prev"),
    ):
        with pytest.raises(ValueError):
            bad()


def test_complementarity_path_by_path(discount: DiscountCurve, rng: np.random.Generator) -> None:
    """up 'prev' + down 'prev' (corridor, same B, complementary inequalities) = the plain variance
    swap path by path; up 'both' + down 'both' = variance swap minus the crossing-day variance."""
    times = daily_schedule(1.0)
    ps = _random_paths(rng, 300, times)
    idx = FixingIndex(times)
    vs = VarianceSwap(times, 0.0, discount, annualisation=252.0)
    up = ConditionalVarianceSwap(times, 100.0, "up", "prev", "corridor", 0.0, discount)
    down = ConditionalVarianceSwap(
        times, 100.0, "down", "prev", "corridor", 0.0, discount, strict=False
    )
    np.testing.assert_allclose(up.payoff(ps, idx) + down.payoff(ps, idx), vs.payoff(ps, idx))
    up_b = ConditionalVarianceSwap(times, 100.0, "up", "both", "corridor", 0.0, discount)
    down_b = ConditionalVarianceSwap(
        times, 100.0, "down", "both", "corridor", 0.0, discount, strict=False
    )
    ls = ps.log_spot_at(idx.indices(times))
    above = ls > np.log(100.0)
    crossing = above[:, :-1] != above[:, 1:]
    r2 = np.diff(ls, axis=1) ** 2
    cross_var = float(discount.df(1.0)) * 252.0 / 252 * np.sum(r2 * crossing, axis=1)
    np.testing.assert_allclose(
        up_b.payoff(ps, idx) + down_b.payoff(ps, idx), vs.payoff(ps, idx) - cross_var
    )
    # convexity spread decomposition
    spread = ConvexitySpread(
        UpVar(times, 100.0, 0.2, discount), VarianceSwap(times, 0.04, discount)
    )
    parts = spread.decompose()
    np.testing.assert_allclose(spread.payoff(ps, idx), sum(p.payoff(ps, idx) for p in parts))
    with pytest.raises(ValueError):
        ConvexitySpread(
            UpVar(times, 100.0, 0.2, discount), VarianceSwap([0.0, 1.0], 0.04, discount)
        )


def test_knock_out_swap_payoffs_and_limits(
    discount: DiscountCurve, rng: np.random.Generator
) -> None:
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    spots = np.array(
        [[100.0, 105.0, 112.0, 108.0, 120.0], [115.0, 105.0, 112.0, 108.0, 120.0], [100.0] * 5]
    )
    ps = _paths(spots, times)
    idx = FixingIndex(times)
    df = float(discount.df(1.0))
    ko = KnockOutVarianceSwap(times, 110.0, 0.2, discount, annualisation=4.0)
    st = ko.statistics(ps, idx)
    np.testing.assert_allclose(st["tau"], [2.0, 0.0, 4.0])  # KO day's return accrues; S_0 breach
    np.testing.assert_allclose(st["ko"], [1.0, 1.0, 0.0])
    r2 = np.diff(np.log(spots[0])) ** 2
    np.testing.assert_allclose(
        ko.payoff(ps, idx)[:2], [df * (r2[0] + r2[1] - 0.04 * 2 / 4), 0.0], atol=1e-15
    )
    # B -> infinity recovers the plain variance swap path by path
    big = _random_paths(rng, 200, daily_schedule(1.0))
    idx_d = FixingIndex(daily_schedule(1.0))
    ko_inf = KnockOutVarianceSwap(daily_schedule(1.0), 1e9, 0.04**0.5, discount)
    vs = VarianceSwap(daily_schedule(1.0), 0.04, discount, annualisation=252.0)
    np.testing.assert_allclose(ko_inf.payoff(big, idx_d), vs.payoff(big, idx_d))
    down = KnockOutVarianceSwap(times, 106.0, 0.2, discount, direction="down", annualisation=4.0)
    np.testing.assert_allclose(down.statistics(ps, idx)["tau"], [0.0, 1.0, 0.0])  # S_0 < B
    assert "Knock-out variance swap" in repr(ko)
    with pytest.raises(NotImplementedError):
        KnockOutVarianceSwap(times, 110.0, 0.2, discount, monitoring="continuous")
    with pytest.raises(NotImplementedError):
        KnockOutVarianceSwap(times, 110.0, 0.2, discount, variant="a")


def test_knock_out_swap_settled_at_the_knock_out(discount: DiscountCurve) -> None:
    """``settlement="knock_out"`` (the desk's convention, owner 2026-09-27): the same accrued
    amount paid at the knock-out close ``t_τ`` and discounted from it — path by path the
    maturity-settled payoff times ``DF(t_τ)/DF(T)``, identical when the swap never knocks; the
    default stays at maturity; seasoning, ageing and rebinding carry the convention; other values
    raise."""
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    spots = np.array(
        [[100.0, 105.0, 112.0, 108.0, 120.0], [115.0, 105.0, 112.0, 108.0, 120.0], [100.0] * 5]
    )
    ps = _paths(spots, times)
    idx = FixingIndex(times)
    at_t = KnockOutVarianceSwap(times, 110.0, 0.2, discount, annualisation=4.0)
    at_ko = KnockOutVarianceSwap(
        times, 110.0, 0.2, discount, annualisation=4.0, settlement="knock_out"
    )
    assert at_t.settlement == "maturity"
    np.testing.assert_allclose(at_ko.pay_time(ps, idx), [0.5, 0.0, 1.0])
    ratio = np.array([discount.df(0.5), discount.df(0.0), discount.df(1.0)]) / discount.df(1.0)
    np.testing.assert_allclose(at_ko.payoff(ps, idx), at_t.payoff(ps, idx) * ratio, atol=1e-15)
    assert float(discount.df(0.5)) != float(discount.df(1.0))  # the curve discounts
    daily = KnockOutVarianceSwap(daily_schedule(1.0), 110.0, 0.2, discount, settlement="knock_out")
    assert daily.aged(0.5 / 252).settlement == "knock_out"  # type: ignore[attr-defined]
    assert at_ko.with_discount(discount).settlement == "knock_out"  # type: ignore[attr-defined]
    assert "settled at the knock-out" in repr(at_ko) and "settled" not in repr(at_t)
    with pytest.raises(ValueError, match="settlement"):
        KnockOutVarianceSwap(times, 110.0, 0.2, discount, settlement="hit")


def test_variance_option_payoffs_and_parity(
    discount: DiscountCurve, rng: np.random.Generator
) -> None:
    """The option on realised variance (the desk's put on variance): path by path the put pays
    ``DF(T) max(K² − RV, 0)`` with the variance swap's ``RV``, the call minus the put is the
    variance swap on the same strike, and on realised vol the vol swap; about half the paths end
    in the money (a non-degenerate check); ageing and rebinding keep the terms; bad terms raise."""
    times = daily_schedule(1.0)
    ps = _random_paths(rng, 400, times)
    idx = FixingIndex(times)
    k = 0.19  # the paths' daily steps of 1.2% realise about 19% vol
    put = VarianceOption(times, k, discount, annualisation=252.0)
    call = VarianceOption(times, k, discount, cp=1, annualisation=252.0)
    vs = VarianceSwap(times, k * k, discount, annualisation=252.0)
    rv = vs.realised_variance(ps, idx)
    df = float(discount.df(1.0))
    np.testing.assert_allclose(put.payoff(ps, idx), df * np.maximum(k * k - rv, 0.0), atol=1e-15)
    np.testing.assert_allclose(
        call.payoff(ps, idx) - put.payoff(ps, idx), vs.payoff(ps, idx), atol=1e-15
    )
    itm = float(np.mean(put.payoff(ps, idx) > 0.0))
    assert 0.25 < itm < 0.75, itm
    vput = VarianceOption(times, k, discount, underlying="vol", annualisation=252.0)
    vcall = VarianceOption(times, k, discount, cp=1, underlying="vol", annualisation=252.0)
    vol_swap = VolSwap(times, k, discount, annualisation=252.0)
    np.testing.assert_allclose(
        vcall.payoff(ps, idx) - vput.payoff(ps, idx), vol_swap.payoff(ps, idx), atol=1e-15
    )
    assert put.strike == pytest.approx(k * k) and vput.strike == pytest.approx(k)
    aged = put.aged(0.5 / 252)
    assert isinstance(aged, VarianceOption) and aged.cp == -1 and aged.underlying == "variance"
    assert put.with_discount(discount).strike_vol == k  # type: ignore[attr-defined]
    assert repr(put).startswith("Put on realised variance") and "Call on realised vol" in repr(
        vcall
    )
    for bad in ({"cp": 0}, {"underlying": "vega"}):
        with pytest.raises(ValueError):
            VarianceOption(times, k, discount, **bad)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        VarianceOption(times, -0.1, discount)


def test_vko_payoffs_decomposition_and_running(
    discount: DiscountCurve, rng: np.random.Generator
) -> None:
    times = daily_schedule(1.0)
    ps = _random_paths(rng, 400, times)
    idx = FixingIndex(times)
    ls = ps.log_spot_at(idx.indices(times))
    rv = 252.0 / 252 * np.sum(np.diff(ls, axis=1) ** 2, axis=1)
    put = float(discount.df(1.0)) * np.maximum(100.0 - np.exp(ls[:, -1]), 0.0)
    vko = VolKnockOutPut(100.0, 1.0, 0.20, times, discount)
    np.testing.assert_allclose(vko.payoff(ps, idx), put * (rv < 0.04))
    assert 0 < vko.statistics(ps, idx)["ko"].mean() < 1  # both outcomes occur in the sample
    parts = vko.decompose()
    assert parts is not None and len(parts) == 2
    np.testing.assert_allclose(vko.payoff(ps, idx), sum(p.payoff(ps, idx) for p in parts))
    # knock-out time: first fixing at which the accrued variance exceeds the budget (the day the
    # knock-out became certain); N + 1 when alive
    st = vko.statistics(ps, idx)
    dead = st["ko"] > 0
    assert np.all(st["ko_time"][dead] <= 252) and np.all(st["ko_time"][~dead] == 253)
    cum = np.cumsum(np.diff(ls, axis=1) ** 2, axis=1)
    first = np.argmax(cum > 0.04 * 252 / 252.0, axis=1) + 1
    np.testing.assert_allclose(st["ko_time"][dead], first[dead])
    np.testing.assert_allclose(st["itm"], (np.exp(ls[:, -1]) < 100.0).astype(float))
    huge = VolKnockOutPut(100.0, 1.0, 10.0, times, discount)
    np.testing.assert_allclose(huge.payoff(ps, idx), put)
    zero = VolKnockOutPut(100.0, 1.0, 0.0, times, discount)
    np.testing.assert_allclose(zero.payoff(ps, idx), 0.0)
    # monotone in vol_ko path by path (the indicator is monotone in the barrier)
    prices = [
        VolKnockOutPut(100.0, 1.0, h, times, discount).payoff(ps, idx) for h in (0.15, 0.2, 0.3)
    ]
    assert np.all(prices[0] <= prices[1]) and np.all(prices[1] <= prices[2])
    assert "knock-out put" in repr(vko) and "knock-in" in repr(parts[1])
    with pytest.raises(ValueError):
        VolKnockOutPut(100.0, 0.5, 0.2, times, discount)


def test_black_scholes_fair_strikes(forward_curve: ForwardCurve) -> None:
    """Flat BS σ: iid returns make every conditional and knock-out fair strike equal to σ
    (optional stopping for the KO swap), and the corridor strike σ²·E[D]/N."""
    sigma = 0.2
    model = BlackScholes(sigma, forward_curve)
    discount = forward_curve.rate_curve
    times = daily_schedule(1.0)
    sim = SimConfig(n_paths=40_000, dt_max=1.0 / 252.0, chunk_size=20_000, seed=9)
    up = UpVar(times, 100.0, sigma, discount)
    down = DownVar(times, 100.0, sigma, discount)
    ko = KnockOutVarianceSwap(times, 110.0, sigma, discount)
    for prod in (up, down, ko):
        fs = fair_strike(prod, model, sim)
        assert abs(fs.vol - sigma) < 3.5 * fs.vol_stderr, (prod, fs)
    fs_ko = fair_strike(ko, model, sim)
    assert 0.0 < fs_ko.extras["p_ko"][0] < 1.0 and fs_ko.extras["e_tau"][0] < 252.0
    corridor = UpVar(times, 100.0, sigma, discount, convention="corridor")
    fs_c = fair_strike(corridor, model, sim)
    cnt, cnt_se = fs_c.extras["count"]
    assert abs(fs_c.variance - sigma**2 * cnt) < 3.5 * np.hypot(
        fs_c.variance_stderr, sigma**2 * cnt_se
    )
    diff, se = strike_differential(
        ConvexitySpread(up, VarianceSwap(times, sigma**2, discount, annualisation=252.0)),
        model,
        sim,
    )
    assert abs(diff) < 3.5 * se  # no skew: no convexity premium
    with pytest.raises(TypeError):
        fair_strike(VarianceSwap(times, 0.04, discount), model, sim)  # type: ignore[arg-type]


def test_vko_analysis_black_scholes(forward_curve: ForwardCurve) -> None:
    """Flat BS: the barrier sweep is monotone, saturates at 1 well above σ, and the ITM-conditional
    realised vol is centred on σ (returns are iid, nearly independent of the terminal level)."""
    sigma = 0.2
    model = BlackScholes(sigma, forward_curve)
    times = daily_schedule(1.0)
    sim = SimConfig(n_paths=40_000, dt_max=1.0 / 252.0, chunk_size=20_000, seed=13)
    vko = VolKnockOutPut(100.0, 1.0, 0.30, times, forward_curve.rate_curve)
    an = vko_analysis(vko, model, sim)
    assert np.all(np.diff(an.ratios) >= 0) and an.ratios[-1] == pytest.approx(1.0)
    assert an.p_ko_itm[-1] == 0.0 and 0.4 < an.p_ko[0] < 0.6  # barrier at sigma: about half
    assert abs(an.itm_rv_quantiles[1] - sigma) < 0.01 and 0.0 < an.p_itm < 1.0
    assert an.itm_rv_quantiles[0] < an.itm_rv_quantiles[1] < an.itm_rv_quantiles[2]
    frame = an.as_frame()
    assert list(frame.columns)[:2] == ["vol_ko", "ratio"] and len(frame) == 5
    assert an.vanilla == pytest.approx(an.prices[-1]) and "VKOAnalysis" in repr(an)
    with pytest.raises(ValueError):
        vko_analysis(vko.decompose()[1], model, sim)  # type: ignore[arg-type]


def test_lv_orderings_and_vko_monotonicity(ssvi: SSVISurface, local_vol: LocalVolSurface) -> None:
    """On the negatively skewed reference surface under local vol: K_up < K_var < K_down for
    B ∈ {90, 100, 110}% and K_KO > K_var for B ∈ {105, 110, 120}%, at 3 stderr.  The KO ordering
    is a property of negative skew and a non-inverted term structure (survival-weighted local
    variance over lower spot states), not a theorem — it can flip for symmetric smiles or
    inverted term structures.  The VKO price is monotone in the vol barrier and below the vanilla.
    """
    model = LocalVol(local_vol)
    discount = ssvi.discount
    times = daily_schedule(1.0)
    sim = SimConfig(n_paths=100_000, chunk_size=50_000, seed=4)
    spot = ssvi.forward_curve.spot
    vs = VarianceSwap(times, 0.0, discount, annualisation=252.0)
    prods: list = [vs]
    for b in (0.9, 1.0, 1.1):
        prods += [UpVar(times, b * spot, 0.0, discount), DownVar(times, b * spot, 0.0, discount)]
    for b in (1.05, 1.1, 1.2):
        prods.append(KnockOutVarianceSwap(times, b * spot, 0.0, discount))
    legs = [vs] + [p.leg(n) for p in prods[1:] for n in ("accrued", "count")]
    res = MonteCarlo(sim).price_many(legs, model, keep_payoffs=True)
    pair = lambda x: 0.5 * (x[0::2] + x[1::2])  # noqa: E731
    rv = pair(np.asarray(res[0].payoffs)) / float(discount.df(1.0))
    k_var, k_var_se = rv.mean(), rv.std(ddof=1) / np.sqrt(rv.size)
    strikes = {}
    for j, p in enumerate(prods[1:]):
        a = pair(np.asarray(res[1 + 2 * j].payoffs))
        d = pair(np.asarray(res[2 + 2 * j].payoffs))
        r = a.mean() / d.mean()
        se = (a - r * d).std(ddof=1) / (np.sqrt(a.size) * d.mean())
        strikes[p] = (r, se)
    for b in (0.9, 1.0, 1.1):
        up, down = prods[1 + 2 * (0.9, 1.0, 1.1).index(b)], prods[2 + 2 * (0.9, 1.0, 1.1).index(b)]
        ku, su = strikes[up]
        kd, sd = strikes[down]
        assert ku < k_var - 3 * np.hypot(su, k_var_se), (b, ku, k_var)
        assert kd > k_var + 3 * np.hypot(sd, k_var_se), (b, kd, k_var)
    for p in prods[7:]:
        kk, sk = strikes[p]
        assert kk > k_var + 3 * np.hypot(sk, k_var_se), (p, kk, k_var)
    # VKO: monotone in the barrier, never above the vanilla
    vkos = [VolKnockOutPut(spot, 1.0, h, times, discount) for h in (0.25, 0.30, 0.35, 0.40)]
    out = MonteCarlo(sim).price_many([*vkos, vkos[0].vanilla()], model)
    prices = [r.mean for r in out[:4]]
    assert prices == sorted(prices) and prices[-1] < out[4].mean
    rep = vko_report(vkos[1], model, sim)
    assert 0.0 < rep.discount < 1.0 and rep.discount_stderr > 0 and 0.0 < rep.p_ko < 1.0


# ---------------------------------------------------------------------------------------------
# slow: calibrated LSV (cache hits after the headline run)
# ---------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_gyongy_invariance_orderings_and_vko_discount_under_lsv() -> None:
    """Corridor swaps with a single-close indicator depend only on the marginals (Gyöngy): same
    fair strike under LV and LSV for ω ∈ {1, 2, 3} within 2 stderr, equal to
    (A/N) Σ E[σ_loc²(t_{i−1}, S_{i−1}) δ · I_i] on the LV paths; K_up < K_var < K_down and
    K_KO > K_var hold under the LSV; the VKO ratios to the vanilla put are reported per model."""
    import dataclasses

    from volsto.calibration import LeverageCache
    from volsto.calibration.cache import build_market
    from volsto.config import CalibrationSpec, load_yaml
    from volsto.studies.m4 import one_factor_variants

    base = load_yaml(ROOT / "configs/studies/lsv_reference_1f.yaml", CalibrationSpec)
    _, surface, _ = build_market(base)
    cache = LeverageCache(ROOT / "cache")
    lv = LocalVol(LocalVolSurface.from_implied(surface, base.local_vol))
    discount = surface.discount
    spot = surface.forward_curve.spot
    times = daily_schedule(1.0)
    sim = dataclasses.replace(base.sim, n_paths=200_000, seed=21)
    corridor_up = ConditionalVarianceSwap(times, spot, "up", "prev", "corridor", 0.0, discount)
    corridor_down = ConditionalVarianceSwap(times, spot, "down", "curr", "corridor", 0.0, discount)
    # LV identity: (A/N) Σ σ_loc²(t_{i-1}, S_{i-1}) δ I_i on the LV paths
    grid = MonteCarlo(sim).build_grid([corridor_up], lv)
    paths = MonteCarlo(sim).simulate(lv, grid)
    idx = grid.fixing_index
    cols = idx.indices(times)
    ls = paths.log_spot_at(cols)
    var_prev = paths.variance_at(cols[:-1])
    ind_up = (ls[:, :-1] > np.log(spot)).astype(float)
    ident_up = np.mean(np.sum(var_prev * ind_up, axis=1) / 252.0)
    lv_up = fair_strike(corridor_up, lv, sim)
    assert abs(lv_up.variance - ident_up) < 3 * lv_up.variance_stderr + 0.0002, (lv_up, ident_up)
    vko = VolKnockOutPut(spot, 1.0, 0.30, times, discount)
    rep_lv = vko_report(vko, lv, sim)
    vko_ratios: dict[str, tuple[float, float]] = {}
    for name, spec in one_factor_variants(base).items():
        model, _ = cache.get_or_calibrate(spec, allow_calibrate=True)
        for prod, ref in ((corridor_up, lv_up), (corridor_down, None)):
            fs = fair_strike(prod, model, sim)
            fs_lv = ref or fair_strike(prod, lv, sim)
            gap = abs(fs.vol - fs_lv.vol)
            assert gap < 2.0 * np.hypot(fs.vol_stderr, fs_lv.vol_stderr) + 0.0002, (
                name,
                prod,
                fs,
                fs_lv,
            )
        vs = VarianceSwap(times, 0.0, discount, annualisation=252.0)
        k_var = MonteCarlo(sim).price(vs, model)
        kv = k_var.mean / float(discount.df(1.0))
        up = fair_strike(UpVar(times, spot, 0.0, discount), model, sim)
        down = fair_strike(DownVar(times, spot, 0.0, discount), model, sim)
        ko = fair_strike(KnockOutVarianceSwap(times, 1.1 * spot, 0.0, discount), model, sim)
        assert up.variance < kv < down.variance and ko.variance > kv, (name, up, kv, down, ko)
        rep = vko_report(vko, model, sim)
        assert 0.0 < rep.discount < 1.0 and 0.0 < rep.p_ko < 1.0, (name, rep)
        vko_ratios[name] = (rep.discount, rep.discount_stderr)
    print("VKO ratio to vanilla (LV):", rep_lv, "LSV:", vko_ratios)
    d, se, _, _ = lsv_minus_lv(UpVar(times, spot, 0.0, discount), model, lv, sim)
    assert np.isfinite(d) and se > 0
