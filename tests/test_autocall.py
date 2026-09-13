"""M6 Part 2: autocall and Phoenix (SPEC §6.6).  Hand values on synthetic paths (both final-date
readings, the barrier boundary, breach versus knock-in), path-by-path decomposition identities
(the interview-thread European KI identity, American ≥ European KI put, monotonicity in the KI
level, the AC → ∞ / KI → 0 bond limit), Black–Scholes closed forms (digital autocall
probability, the European KI put leg as put(B) + (K − B) cash-or-nothing put, the continuous
American KI put as the Reiner–Rubinstein down-and-in put), vol monotonicity under common random
numbers, the analytics report / LSV-minus-LV table and the forward-skew exposure under local vol
on a flat surface against the static surface sensitivity."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest

from volsto.analytics.autocall import (
    autocall_probabilities,
    autocall_report,
    expected_life,
    forward_skew_exposure,
    ki_probability,
    leg_attribution,
    lsv_minus_lv_table,
    skew_tent_perturbation,
    skew_tent_slope,
    state_model_factory,
)
from volsto.analytics.conditional_variance import pair_average
from volsto.config import SimConfig
from volsto.engine import FixingIndex, MonteCarlo, PathSet
from volsto.market import DiscountCurve, ForwardCurve, bs_price, bs_vega, norm_cdf
from volsto.market.barrier_bs import bs_barrier_price
from volsto.models import BlackScholes
from volsto.products import DigitalOption, EuropeanOption, daily_schedule
from volsto.products.autocall import (
    Autocall,
    AutocallStatistic,
    BondLeg,
    ConditionalDigital,
    CouponLeg,
    KIPutLeg,
    Phoenix,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helpers import flat_state

OBS = [1.0, 2.0, 3.0]
S0 = 100.0


def _paths(spots: np.ndarray, times: np.ndarray, variance: float = 0.0) -> PathSet:
    """Synthetic path set; with ``variance > 0`` the ``sum_sq`` accumulator carries the squared
    log-returns of every column step, so the Brownian-bridge machinery sees a full-step grid."""
    n, m = spots.shape
    ls = np.log(spots)
    dx = np.diff(ls, axis=1)
    sum_sq = np.concatenate([np.zeros((n, 1)), np.cumsum(dx * dx, axis=1)], axis=1)
    return PathSet(
        times,
        ls,
        np.full((n, m), variance),
        np.zeros((n, m, 0)),
        np.zeros((n, m)),
        sum_sq,
    )


def _random_paths(
    rng: np.random.Generator, n: int, times: np.ndarray, vol: float = 0.35
) -> PathSet:
    dt = np.diff(times)
    steps = rng.normal(-0.5 * vol**2 * dt, vol * np.sqrt(dt), size=(n, times.size - 1))
    spots = S0 * np.exp(np.concatenate([np.zeros((n, 1)), np.cumsum(steps, axis=1)], axis=1))
    return _paths(spots, times)


def _autocall(discount: DiscountCurve, **kw) -> Autocall:
    args = dict(spot_reference=S0, coupons=0.06, ki_level=0.6, ki_type="european")
    args.update(kw)
    return Autocall(OBS, discount, **args)


def _phoenix(discount: DiscountCurve, **kw) -> Autocall:
    args = dict(
        spot_reference=S0,
        coupon=0.05,
        coupon_barrier=0.7,
        memory=True,
        ki_level=0.6,
        ki_type="american",
        ki_monitoring="discrete",
    )
    args.update(kw)
    return Phoenix(OBS, discount, **args)


def _digital_prob(fc: ForwardCurve, strike: float, t: float, vol: float) -> float:
    """``N(d2)``: risk-neutral ``P(S_t ≥ strike)`` under Black–Scholes."""
    f = float(fc.forward(t))
    return float(norm_cdf((np.log(f / strike) - 0.5 * vol**2 * t) / (vol * np.sqrt(t))))


def test_autocall_hand_values_and_validation(discount: DiscountCurve) -> None:
    times = np.array([0.0, 1.0, 2.0, 3.0])
    spots = np.array(
        [
            [100.0, 105.0, 50.0, 50.0],  # autocall at date 1: 1 + 0.06
            [100.0, 95.0, 101.0, 90.0],  # autocall at date 2: 1 + 2 x 0.06
            [100.0, 90.0, 95.0, 70.0],  # alive, S_3 = 70 ≥ 60: par
            [100.0, 90.0, 80.0, 50.0],  # KI: 1 − (100 − 50)/100
            [100.0, 100.0, 50.0, 50.0],  # S = AC exactly autocalls (≥ convention)
        ]
    )
    ps, idx = _paths(spots, times), FixingIndex(times)
    df = discount.df(OBS)
    ac = _autocall(discount)
    np.testing.assert_allclose(
        ac.payoff(ps, idx), [df[0] * 1.06, df[1] * 1.12, df[2], df[2] * 0.5, df[0] * 1.06]
    )
    st = ac.statistics(ps, idx)
    np.testing.assert_allclose(st["ac_index"], [1, 2, 4, 4, 1])
    np.testing.assert_allclose(st["life"], [1, 2, 3, 3, 1])
    np.testing.assert_allclose(st["ki_hit"], [0, 0, 0, 1, 0])
    np.testing.assert_allclose(st["coupons_paid"], [0.06, 0.12, 0, 0, 0.06])
    np.testing.assert_allclose(st["redemption"], [1, 1, 1, 0.5, 1])
    np.testing.assert_allclose(st["put_loss"], [0, 0, 0, 0.5, 0])
    # explicit coupon schedule, step-down barriers, non-call period, notional
    sd = _autocall(
        discount,
        coupons=[0.05, 0.11, 0.2],
        autocall_barriers=[1.0, 0.97, 0.9],
        non_call_periods=1,
        notional=2.0,
    )
    # row 0: date 1 is non-call, S_2 = 50 < 97, S_3 = 50 < 90 and < 60: KI, put 0.5
    # row 1: S_2 = 101 ≥ 97: autocall 1 + 0.11; row 2: 95 < 97, 70 < 90, no KI: par
    # row 3: 80 < 97, 50 < 90, KI; row 4: date 1 non-call (S = 100), then 50 < 97, 50 < 90, KI
    np.testing.assert_allclose(
        sd.payoff(ps, idx),
        [2 * df[2] * 0.5, 2 * df[1] * 1.11, 2 * df[2], 2 * df[2] * 0.5, 2 * df[2] * 0.5],
    )
    np.testing.assert_allclose(sd.statistics(ps, idx)["ac_index"], [4, 2, 4, 4, 4])
    # legs sum to the payoff path by path; the European put splits into put(B) + digital
    for prod in (ac, sd):
        legs = prod.decompose()
        assert [type(x) for x in legs] == [ConditionalDigital] * 3 + [BondLeg, KIPutLeg, KIPutLeg]
        np.testing.assert_allclose(sum(x.payoff(ps, idx) for x in legs), prod.payoff(ps, idx))
    assert "growing" in repr(ac) and "European KI 0.6" in repr(ac) and "non-call" in repr(sd)
    assert "no earlier autocall" in repr(ac.decompose()[0]) and "K - B" in repr(ac.decompose()[-1])
    assert ac.maturity == 3.0 and np.array_equal(ac.pay_times, OBS)
    # the barrier boundary: S_T = B and one ulp below.  paths.spot_at = exp(log S) lands one ulp
    # below 60 for both, where ln S < ln B is false but S < B is true; every comparison is made
    # in spot space on the same array, so put = put_vanilla + put_digital holds there too
    edge = np.array([[100.0, 90.0, 80.0, 60.0], [100.0, 90.0, 80.0, np.nextafter(60.0, 0.0)]])
    ps_e = _paths(edge, times)
    legs_e = ac.leg_payoffs(ps_e, idx)
    below = ps_e.spot_at(idx[3.0]) < 60.0
    np.testing.assert_array_equal(ac.statistics(ps_e, idx)["ki_hit"], below.astype(float))
    np.testing.assert_allclose(
        legs_e["put"], legs_e["put_vanilla"] + legs_e["put_digital"], rtol=1e-14, atol=0.0
    )
    np.testing.assert_allclose(legs_e["put"], -df[2] * 0.4 * below, rtol=1e-12)
    # ki_breach versus ki_hit: an American breach followed by an autocall is a breach, not a
    # knock-in of the life; a European path autocalled before T_N never observes the level
    am = _autocall(discount, ki_type="american", ki_monitoring="discrete", ki_fixing_times=times)
    mixed = np.array([[100.0, 55.0, 105.0, 105.0], [100.0, 55.0, 95.0, 50.0]])
    st_am = am.statistics(_paths(mixed, times), idx)
    np.testing.assert_allclose(st_am["ki_breach"], [1, 1])
    np.testing.assert_allclose(st_am["ki_hit"], [0, 1])
    np.testing.assert_allclose(st_am["ac_index"], [2, 4])
    st_eu = ac.statistics(_paths(np.array([[100.0, 105.0, 50.0, 50.0]]), times), idx)
    assert st_eu["ki_breach"][0] == 0.0 and st_eu["ki_hit"][0] == 0.0
    # the repr names the monitoring schedule (span, step, content hash when non-uniform)
    a1 = am.replace(ki_fixing_times=[0.0, 0.5, 1.5, 3.0])
    a2 = am.replace(ki_fixing_times=[0.0, 0.7, 2.5, 3.0])
    assert repr(a1) != repr(a2) and "non-uniform" in repr(a1) and "sha1" in repr(a2)
    assert "on 4 discrete dates from 0y to 3y (step 1y)" in repr(am)
    # ageing shifts the schedule, with_discount rebinds legs to the new curve
    aged = ac.aged(0.5)
    np.testing.assert_allclose(aged.observation_times, [0.5, 1.5, 2.5])
    other = DiscountCurve.flat(0.05)
    leg = KIPutLeg(ac).with_discount(other)
    assert isinstance(leg, KIPutLeg) and leg.parent.discount is other and leg.discount is other
    for bad in (
        lambda: _autocall(discount, final_redemption="par"),
        lambda: _autocall(discount, final_redemption="coupon_barrier"),  # needs a Phoenix
        lambda: _phoenix(  # a European KI above CB is never the effective level
            discount,
            ki_type="european",
            ki_monitoring=None,
            ki_level=0.8,
            final_redemption="coupon_barrier",
        ),
        lambda: _autocall(discount, spot_reference=0.0),
        lambda: _autocall(discount, ki_level=1.2),
        lambda: _autocall(discount, ki_level=0.6, ki_type="bermudan"),
        lambda: _autocall(discount, ki_monitoring="discrete"),  # European KI has no monitoring
        lambda: _autocall(discount, ki_type="american"),  # needs a monitoring convention
        lambda: _autocall(discount, ki_type="american", ki_monitoring="daily"),
        lambda: _autocall(
            discount, ki_type="american", ki_monitoring="discrete", ki_fixing_times=[0.5, 2.9]
        ),
        lambda: _autocall(
            discount, ki_type="american", ki_monitoring="continuous", ki_fixing_times=[3.0]
        ),
        lambda: _autocall(discount, coupons=-0.01),
        lambda: _autocall(discount, coupons=[0.06, 0.12]),
        lambda: _autocall(discount, autocall_barriers=[1.0, 0.0, 1.0]),
        lambda: _autocall(discount, memory=True),  # memory needs a coupon barrier
        lambda: _autocall(discount, guaranteed_coupons=True, coupon_barrier=0.7),
        lambda: _autocall(discount, non_call_periods=4),
        lambda: Autocall(
            [0.0, 1.0], discount, spot_reference=S0, coupons=0.06, ki_level=0.6, ki_type="european"
        ),
        lambda: KIPutLeg(_phoenix(discount), "vanilla"),
        lambda: ConditionalDigital(ac, 4),
        lambda: AutocallStatistic(ac, "autocall_at"),
        lambda: AutocallStatistic(ac, "price"),
    ):
        with pytest.raises(ValueError):
            bad()
    with pytest.raises(ValueError):
        ac.aged(1.0)  # a fixing inside the roll window


def test_phoenix_hand_values(discount: DiscountCurve) -> None:
    """Both final-date readings on the same paths: ``final_redemption="knock_in"`` (coupon
    decision and knock-in redemption independent) and ``"coupon_barrier"`` (SPEC §6.6 literal:
    par plus the coupon when ``S_TN ≥ CB``).  They differ only on the path that breached and ends
    in ``[CB, K)`` (row 2)."""
    times = np.array([0.0, 1.0, 2.0, 3.0])
    spots = np.array(
        [
            [100.0, 65.0, 75.0, 80.0],  # no coupon, coupon + memory (0.10), coupon; no KI: par
            [100.0, 55.0, 75.0, 120.0],  # breach at date 1 (harmless): autocall at 3 with coupon
            [100.0, 55.0, 65.0, 90.0],  # KI hit; three coupons paid at date 3; redemption 0.9
            [100.0, 75.0, 75.0, 75.0],  # coupon every date, alive, no KI
            [100.0, 55.0, 65.0, 65.0],  # KI hit, ends below CB: no coupon, redemption 0.65
        ]
    )
    ps, idx = _paths(spots, times), FixingIndex(times)
    df = discount.df(OBS)
    ph = _phoenix(discount, ki_fixing_times=times)  # the KI monitored on the observation dates
    expected = [
        df[1] * 0.10 + df[2] * 1.05,
        df[1] * 0.10 + df[2] * 1.05,
        df[2] * (0.15 + 0.9),
        df[0] * 0.05 + df[1] * 0.05 + df[2] * 1.05,
        df[2] * 0.65,
    ]
    np.testing.assert_allclose(ph.payoff(ps, idx), expected)
    st = ph.statistics(ps, idx)
    np.testing.assert_allclose(st["ac_index"], [4, 3, 4, 4, 4])
    np.testing.assert_allclose(st["ki_breach"], [0, 1, 1, 0, 1])
    np.testing.assert_allclose(st["ki_hit"], [0, 0, 1, 0, 1])  # the breach then autocall: no KI
    np.testing.assert_allclose(st["coupons_paid"], [0.15, 0.15, 0.15, 0.15, 0.0])
    np.testing.assert_allclose(st["redemption"], [1, 1, 0.9, 1, 0.65])
    no_mem = _phoenix(discount, memory=False, ki_fixing_times=times)
    np.testing.assert_allclose(no_mem.payoff(ps, idx)[2], df[2] * (0.05 + 0.9))
    # the literal SPEC reading: row 2 ends at 90 ≥ CB = 70, par plus the coupons; row 4 ends
    # below CB and keeps the put loss; everything else is unchanged
    ph_cb = _phoenix(discount, ki_fixing_times=times, final_redemption="coupon_barrier")
    expected_cb = list(expected)
    expected_cb[2] = df[2] * (0.15 + 1.0)
    np.testing.assert_allclose(ph_cb.payoff(ps, idx), expected_cb)
    st_cb = ph_cb.statistics(ps, idx)
    np.testing.assert_allclose(st_cb["ki_hit"], st["ki_hit"])  # the event is the same
    np.testing.assert_allclose(st_cb["redemption"], [1, 1, 1, 1, 0.65])
    np.testing.assert_allclose(st_cb["put_loss"], [0, 0, 0, 0, 0.35])
    no_mem_cb = no_mem.replace(final_redemption="coupon_barrier")
    np.testing.assert_allclose(no_mem_cb.payoff(ps, idx)[2], df[2] * 1.05)
    assert "final put loss only below the coupon barrier" in repr(ph_cb)
    assert "final coupon and knock-in redemption independent" in repr(ph)
    # a European KI at or below CB: the gate is a no-op and the put split stays exact
    ph_eu_cb = _phoenix(
        discount, ki_type="european", ki_monitoring=None, final_redemption="coupon_barrier"
    )
    ph_eu = _phoenix(discount, ki_type="european", ki_monitoring=None)
    np.testing.assert_allclose(ph_eu_cb.payoff(ps, idx), ph_eu.payoff(ps, idx))
    legs_eu = ph_eu_cb.leg_payoffs(ps, idx)
    np.testing.assert_allclose(legs_eu["put"], legs_eu["put_vanilla"] + legs_eu["put_digital"])
    # guaranteed coupons: paid at every date while alive, autocall date included
    g = _autocall(discount, coupons=0.05, guaranteed_coupons=True)
    np.testing.assert_allclose(g.payoff(ps, idx)[1], df[0] * 0.05 + df[1] * 0.05 + df[2] * 1.05)
    np.testing.assert_allclose(g.payoff(ps, idx)[2], df[0] * 0.05 + df[1] * 0.05 + df[2] * 1.05)
    g_am = _autocall(
        discount,
        coupons=0.05,
        guaranteed_coupons=True,
        ki_type="american",
        ki_monitoring="discrete",
        ki_fixing_times=times,
    )  # row 2 knocked in at date 1 (55 < 60) and ends at 90: coupons plus 0.9
    np.testing.assert_allclose(g_am.payoff(ps, idx)[2], df[0] * 0.05 + df[1] * 0.05 + df[2] * 0.95)
    with pytest.raises(ValueError):
        g_am.replace(final_redemption="coupon_barrier")  # no coupon barrier to gate on
    for prod in (ph, no_mem, g, g_am, ph_cb, no_mem_cb, ph_eu_cb):
        legs = prod.decompose()
        assert any(isinstance(x, CouponLeg) for x in legs)
        np.testing.assert_allclose(sum(x.payoff(ps, idx) for x in legs), prod.payoff(ps, idx))
    assert "Phoenix" in repr(ph) and "(memory)" in repr(ph) and "American KI" in repr(ph)
    assert "guaranteed" in repr(g)
    # the default daily monitoring ends at the maturity and enters the fixing schedule
    daily = _phoenix(discount)
    assert daily.ki_fixing_times is not None and daily.ki_fixing_times[-1] == 3.0
    assert daily.fixing_times.size == daily_schedule(3.0).size and daily.requires_all_steps is False
    cont = _phoenix(discount, ki_monitoring="continuous")
    assert cont.requires_all_steps and cont.fixing_times.size == 3 and "continuous" in repr(cont)
    assert isinstance(KIPutLeg(cont).european_counterpart().parent, Autocall)
    assert KIPutLeg(cont).european_counterpart().parent.ki_type == "european"
    # ageing: monitoring dates inside the roll window drop out (observed at the held spot), t = 0
    # stays, the rest shift; an observation date inside the window raises
    aged = daily.aged(0.5)
    assert aged.ki_fixing_times is not None and aged.ki_fixing_times[-1] == pytest.approx(2.5)
    assert aged.ki_fixing_times.size == 757 - 126 and aged.ki_fixing_times[0] == 0.0
    assert np.all(aged.ki_fixing_times[1:] > 0) and aged.observation_times.tolist() == [
        0.5,
        1.5,
        2.5,
    ]
    # 0.5y is a monitoring date: the rolled schedule is daily again; an off-grid roll is not
    assert "non-uniform" not in repr(aged) and "non-uniform" not in repr(daily)
    assert "non-uniform" in repr(daily.aged(0.3)) and "sha1" in repr(daily.aged(0.3))
    with pytest.raises(ValueError):
        daily.aged(1.0)


def test_decomposition_identities_on_random_paths(
    discount: DiscountCurve, rng: np.random.Generator
) -> None:
    times = daily_schedule(3.0)
    ps, idx = _random_paths(rng, 600, times), FixingIndex(times)
    eu = _autocall(discount)
    am = _autocall(discount, ki_type="american", ki_monitoring="discrete")
    ph = _phoenix(discount)
    ph_eu = _phoenix(discount, ki_type="european", ki_monitoring=None)
    for prod in (eu, am, ph, ph_eu):
        legs = prod.decompose()
        np.testing.assert_allclose(
            sum(x.payoff(ps, idx) for x in legs), prod.payoff(ps, idx), rtol=1e-12, atol=1e-15
        )
        st = prod.statistics(ps, idx)
        assert 0 < st["autocalled"].mean() < 1 and 0 < st["ki_hit"].mean() < 1
    # the European put leg = put(B) + (K − B) digital put at B, path by path, and each term is
    # the survival-conditioned vanilla / digital option
    legs = eu.leg_payoffs(ps, idx)
    np.testing.assert_allclose(legs["put"], legs["put_vanilla"] + legs["put_digital"], rtol=1e-12)
    van, dig = KIPutLeg(eu).unconditional_components()
    assert isinstance(van, EuropeanOption) and isinstance(dig, DigitalOption)
    assert van.strike == 60.0 and dig.strike == 60.0 and dig.payout == 40.0 and van.cp == -1
    alive = 1.0 - eu.statistics(ps, idx)["autocalled"]
    np.testing.assert_allclose(legs["put_vanilla"], alive * van.payoff(ps, idx), rtol=1e-12)
    np.testing.assert_allclose(legs["put_digital"], alive * dig.payoff(ps, idx), rtol=1e-12)
    parts = KIPutLeg(eu).decompose()
    assert parts is not None and len(parts) == 2 and KIPutLeg(am).decompose() is None
    np.testing.assert_allclose(sum(p.payoff(ps, idx) for p in parts), legs["put"], rtol=1e-12)
    # American KI ≥ European KI put leg (the legs are short: more negative), and the American
    # counterpart machinery
    am_put = KIPutLeg(am).payoff(ps, idx)
    eu_put = KIPutLeg(am).european_counterpart().payoff(ps, idx)
    np.testing.assert_allclose(eu_put, KIPutLeg(eu).payoff(ps, idx))
    assert np.all(am_put <= eu_put + 1e-15) and am_put.sum() < eu_put.sum()
    # price monotone decreasing in the KI level, path by path
    pays = [eu.replace(ki_level=b).payoff(ps, idx) for b in (0.5, 0.6, 0.7, 0.8)]
    for lo, hi in itertools.pairwise(pays):
        assert np.all(hi <= lo + 1e-15) and hi.sum() < lo.sum()
    pays = [am.replace(ki_level=b).payoff(ps, idx) for b in (0.5, 0.6, 0.7)]
    assert np.all(pays[1] <= pays[0] + 1e-15) and np.all(pays[2] <= pays[1] + 1e-15)
    # AC → ∞ and KI → 0: a zero-coupon bond (plain autocall) or bond + coupon leg (Phoenix)
    bond = eu.replace(autocall_barriers=np.inf, ki_level=1e-9)
    np.testing.assert_allclose(bond.payoff(ps, idx), float(discount.df(3.0)))
    zcb = ph.replace(autocall_barriers=np.inf, ki_level=1e-9)
    np.testing.assert_allclose(
        zcb.payoff(ps, idx), float(discount.df(3.0)) + CouponLeg(zcb).payoff(ps, idx), rtol=1e-12
    )
    # memory: the coupons paid total c × (last date on which S ≥ CB)
    s_obs = ps.spot_at(idx.indices(OBS))
    above = s_obs >= 70.0
    last = np.where(above.any(axis=1), np.argmax(above[:, ::-1], axis=1), 3)
    last = np.where(above.any(axis=1), 3 - last, 0)
    np.testing.assert_allclose(zcb.statistics(ps, idx)["coupons_paid"], 0.05 * last)
    # statistic products
    assert np.array_equal(
        AutocallStatistic(eu, "autocall_at", 2).payoff(ps, idx),
        eu.statistics(ps, idx)["ac_index"] == 2,
    )
    assert "Statistic" in repr(AutocallStatistic(eu, "life"))


def test_black_scholes_closed_forms(forward_curve: ForwardCurve) -> None:
    """Flat BS σ = 20%: P(first autocall at T_1) = N(d₂) at AC_1 S_0; the European KI put leg =
    −[put(B) + (K − B) · DF · N(−d₂(B))]/K (the interview-thread identity against the closed
    forms); AC → ∞, KI → 0 Phoenix without memory = DF(T_N) + Σ c DF(T_i) N(d₂(CB, T_i)); the 3y
    autocall and Phoenix statistics are finite with an expected life in (0, 3]."""
    sigma = 0.2
    fc = forward_curve
    discount = fc.rate_curve
    model = BlackScholes(sigma, fc)
    sim = SimConfig(n_paths=40_000, dt_max=1.0 / 50.0, chunk_size=20_000, seed=7)
    ac = _autocall(discount)
    rep = autocall_report(ac, model, sim)
    p1 = _digital_prob(fc, S0, 1.0, sigma)
    assert abs(rep.autocall_probabilities[0] - p1) < 3 * rep.autocall_probabilities_stderr[0]
    assert np.all(np.isfinite(rep.autocall_probabilities)) and rep.price_stderr > 0
    assert rep.autocall_probabilities.sum() == pytest.approx(1.0)
    assert 0.0 < rep.expected_life <= 3.0 and 0.0 < rep.ki_probability < 1.0
    assert abs(rep.legs_total - rep.price) < 1e-12 and 0.9 < rep.price < 1.05
    assert "AutocallReport" in repr(rep) and len(rep.as_frame()) == 4
    # European KI put leg of a never-autocalling note against the closed forms
    b, k, t = 60.0, S0, 3.0
    never = ac.replace(autocall_barriers=np.inf, coupons=0.0)
    leg = KIPutLeg(never)
    van, dig = leg.unconditional_components()
    res = MonteCarlo(sim).price_many(
        [leg, KIPutLeg(never, "vanilla"), KIPutLeg(never, "digital"), van, dig], model
    )
    df_t = float(discount.df(t))
    f = float(fc.forward(t))
    d2 = (np.log(f / b) - 0.5 * sigma**2 * t) / (sigma * np.sqrt(t))
    put_b = float(bs_price(S0, b, t, sigma, 0.02, 0.01, -1))
    digital_b = (k - b) * df_t * float(norm_cdf(-d2))
    closed = -(put_b + digital_b) / k
    assert abs(res[0].mean - closed) < 3 * res[0].stderr, (res[0], closed)
    assert abs(res[1].mean + put_b / k) < 3 * res[1].stderr, (res[1], -put_b / k)
    assert abs(res[2].mean + digital_b / k) < 3 * res[2].stderr, (res[2], -digital_b / k)
    assert res[3].mean == pytest.approx(res[1].mean) and res[4].mean == pytest.approx(res[2].mean)
    # AC → ∞, KI → 0: bond plus the Phoenix coupon strip (digital probabilities, no memory)
    strip = _phoenix(discount, memory=False, autocall_barriers=np.inf, ki_level=1e-9)
    r = MonteCarlo(sim).price(strip, model)
    closed_strip = df_t + sum(
        0.05 * float(discount.df(ti)) * _digital_prob(fc, 70.0, ti, sigma) for ti in OBS
    )
    assert abs(r.mean - closed_strip) < 3 * r.stderr, (r, closed_strip)
    # the 3y Phoenix (CB 70%, memory, American KI 60% daily) prices sensibly
    ph = _phoenix(discount)
    rep_ph = autocall_report(ph, model, sim)
    assert 0.0 < rep_ph.expected_life <= 3.0 and 0.0 < rep_ph.ki_probability < 1.0
    assert np.all(np.isfinite(rep_ph.autocall_probabilities)) and 0.9 < rep_ph.price < 1.1
    assert abs(rep_ph.autocall_probabilities[0] - p1) < 3 * rep_ph.autocall_probabilities_stderr[0]
    assert rep_ph.legs["put"][0] < rep_ph.legs["put_european"][0] < 0  # American ≥ European
    # P(KI) is the same event (breach and no autocall) for both types: the daily American KI of
    # the Phoenix (same autocall barriers and dates) knocks in more often than the European one;
    # P(breach) counts the breaches on autocalled paths too
    assert rep_ph.ki_probability > rep.ki_probability
    assert rep_ph.breach_probability > rep_ph.ki_probability
    assert rep.breach_probability == rep.ki_probability  # European: T_N is the only monitoring
    assert "P(breach)" in repr(rep_ph)
    frame = rep_ph.leg_frame()
    assert set(frame["leg"]) >= {"coupon", "put", "put_european", "product"}


def test_continuous_knock_in(forward_curve: ForwardCurve) -> None:
    """Continuous American knock-in (Brownian bridge of the barrier module).  Hand values of the
    knock-in weight on synthetic full-step paths (``1 − Π(1 − p_i)`` over the life, ``1`` at or
    below the barrier, ``ki_hit = 0`` after an autocall); under flat BS σ = 20% the
    never-autocalling continuous KI put leg equals ``−DIP(K = S_0, H = 0.6 S_0, T = 3) / S_0``
    (Reiner–Rubinstein, :func:`bs_barrier_price`) within 3 stderr — the bridge is exact under
    Black–Scholes at any step; on common paths continuous ≤ daily ≤ European put payoffs path by
    path (the legs are short); ``MonteCarlo.build_grid`` records every step for the continuous
    note without ``SimConfig(record_all_steps=True)``."""
    fc = forward_curve
    discount = fc.rate_curve
    sigma = 0.2
    # hand values: never-autocalling note, variance σ² = 0.04 on unit steps
    times = np.array([0.0, 1.0, 2.0, 3.0])
    spots = np.array(
        [
            [100.0, 80.0, 90.0, 100.0],  # never at the barrier: weight in (0, 1)
            [100.0, 55.0, 90.0, 100.0],  # a recorded point below the barrier: 1
            [100.0, 60.0, 90.0, 100.0],  # at the barrier (bridge convention: knocked): 1
        ]
    )
    ps, idx = _paths(spots, times, variance=sigma**2), FixingIndex(times)
    never = _autocall(
        discount,
        autocall_barriers=np.inf,
        coupons=0.0,
        ki_type="american",
        ki_monitoring="continuous",
    )
    st = never.statistics(ps, idx)
    b = np.log(60.0)
    d = np.log(spots[0]) - b
    p = np.exp(-2.0 * d[:-1] * d[1:] / (sigma**2 * np.diff(times)))
    w0 = 1.0 - np.prod(1.0 - p)
    assert 0.0 < w0 < 1.0
    np.testing.assert_allclose(st["ki_breach"], [w0, 1.0, 1.0])
    np.testing.assert_allclose(st["ki_hit"], [w0, 1.0, 1.0])  # nothing autocalls
    np.testing.assert_allclose(st["put_loss"], [0.0, 0.0, 0.0])  # S_T = 100: no loss
    # with AC = 100% the same paths autocall at date 3 (S_T = 100, the ≥ convention): the breach
    # weight over the life is unchanged and ki_hit is zero; a note autocalling at date 1 has its
    # weight over [0, T_1] only
    ac = never.replace(autocall_barriers=1.0)
    st_ac = ac.statistics(ps, idx)
    np.testing.assert_allclose(st_ac["ac_index"], [3, 3, 3])
    np.testing.assert_allclose(st_ac["ki_breach"], [w0, 1.0, 1.0])
    np.testing.assert_allclose(st_ac["ki_hit"], [0.0, 0.0, 0.0])
    ps_ac = _paths(np.array([[100.0, 105.0, 50.0, 50.0]]), times, variance=sigma**2)
    st1 = ac.statistics(ps_ac, idx)
    d1 = np.log([100.0, 105.0]) - b
    w1 = 1.0 - (1.0 - np.exp(-2.0 * d1[0] * d1[1] / sigma**2))
    np.testing.assert_allclose(st1["ac_index"], [1])
    np.testing.assert_allclose(st1["ki_breach"], [w1])
    np.testing.assert_allclose(st1["ki_hit"], [0.0])
    assert 0.0 < w1 < 0.01
    # a coarser recording (sum_sq inconsistent with the column returns) is refused by the bridge
    coarse = PathSet(
        times,
        np.log(spots),
        np.full(spots.shape, sigma**2),
        np.zeros((*spots.shape, 0)),
        np.zeros(spots.shape),
        np.zeros(spots.shape),
    )
    with pytest.raises(ValueError):
        never.statistics(coarse, idx)
    # Monte Carlo: closed form, CRN ordering, grid recording
    model = BlackScholes(sigma, fc)
    sim = SimConfig(n_paths=40_000, dt_max=1.0 / 50.0, chunk_size=20_000, seed=7)
    cont = KIPutLeg(never)
    daily = KIPutLeg(never.replace(ki_monitoring="discrete"))
    eu = cont.european_counterpart()
    mc = MonteCarlo(sim)
    grid = mc.build_grid([cont], model)
    assert grid.record_times.size == grid.times.size  # requires_all_steps honoured
    grid_eu = mc.build_grid([eu], model)
    assert grid_eu.record_times.size == 4 < grid_eu.times.size
    res = mc.price_many([cont, daily, eu], model, keep_payoffs=True)
    dip = float(bs_barrier_price(S0, S0, 60.0, 3.0, sigma, 0.02, 0.01, -1, "down", "in"))
    closed = -dip / S0
    assert abs(res[0].mean - closed) < 3.0 * res[0].stderr, (res[0], closed)
    pc, pd_, pe = (np.asarray(r.payoffs) for r in res)
    assert np.all(pc <= pd_ + 1e-15) and np.all(pd_ <= pe + 1e-15)
    assert res[0].mean < res[1].mean < res[2].mean < 0.0
    # the full report of a continuous 3y autocall runs on the all-steps grid
    rep = autocall_report(
        _autocall(discount, ki_type="american", ki_monitoring="continuous"), model, sim
    )
    assert 0.0 < rep.ki_probability <= rep.breach_probability < 1.0
    assert abs(rep.legs_total - rep.price) < 1e-12 and np.isfinite(rep.expected_life)


def test_price_decreases_with_vol(forward_curve: ForwardCurve) -> None:
    """Short vol: a parallel vol bump lowers the price (common random numbers, 3 stderr of the
    paired difference), for the autocall and the Phoenix."""
    fc = forward_curve
    sim = SimConfig(n_paths=20_000, dt_max=1.0 / 50.0, chunk_size=20_000, seed=5)
    m20, m25 = BlackScholes(0.2, fc), BlackScholes(0.25, fc)
    mc = MonteCarlo(sim)
    for prod in (_autocall(fc.rate_curve), _phoenix(fc.rate_curve)):
        grid = mc.build_grid([prod], m20)
        draws = mc.draws_for(grid, m20)
        lo = mc.price(prod, m20, grid=grid, draws=draws, keep_payoffs=True)
        hi = mc.price(prod, m25, grid=grid, draws=draws, keep_payoffs=True)
        d = pair_average(np.asarray(hi.payoffs) - np.asarray(lo.payoffs), sim.antithetic)
        assert d.mean() < -3.0 * d.std(ddof=1) / np.sqrt(d.size), (prod, d.mean())


def test_analytics_functions_and_lsv_minus_lv_table(forward_curve: ForwardCurve) -> None:
    fc = forward_curve
    sim = SimConfig(n_paths=20_000, dt_max=1.0 / 50.0, chunk_size=20_000, seed=2)
    ac = _autocall(fc.rate_curve)
    models = {"bs20": BlackScholes(0.2, fc), "bs25": BlackScholes(0.25, fc)}
    life, life_se = expected_life(ac, models["bs20"], sim)
    assert 0.0 < life <= 3.0 and life_se > 0
    p, se = ki_probability(ac, models["bs20"], sim)
    assert 0.0 < p < 1.0 and se > 0
    probs = autocall_probabilities(ac, models["bs20"], sim)
    assert list(probs.columns) == ["date", "T", "probability", "stderr"] and len(probs) == 4
    assert probs["probability"].sum() == pytest.approx(1.0)
    attr = leg_attribution(ac, models["bs20"], sim)
    resid = float(attr.loc[attr["leg"].str.startswith("residual"), "price"].iloc[0])
    assert abs(resid) < 1e-12 and len(attr) == 5 + 3
    table = lsv_minus_lv_table(
        models,
        sim,
        [ac, _phoenix(fc.rate_curve, ki_type="european", ki_monitoring=None)],
        reference="bs20",
    )
    assert len(table) == 4 and set(table["model"]) == {"bs20", "bs25"}
    row = table[(table["model"] == "bs25") & (table["product"] == 0)].iloc[0]
    assert row["price_minus_ref"] < -3.0 * row["price_minus_ref_stderr"]
    assert row["p_ki_minus_ref"] > 3.0 * row["p_ki_minus_ref_stderr"]
    assert row["leg:put_vanilla_minus_ref"] < 0 and np.isfinite(row["expected_life_minus_ref"])
    base = table[(table["model"] == "bs20") & (table["product"] == 0)].iloc[0]
    assert base["price_minus_ref"] == 0.0 and base["leg:bond"] > 0
    assert base["price_minus_ref_stderr"] == 0.0  # a self-difference is exact
    assert row["price_minus_ref_stderr"] > 0 and "p_breach" in table.columns
    assert np.all(table["p_breach"] >= table["p_ki"])
    with pytest.raises(ValueError):
        lsv_minus_lv_table(models, sim, ac, reference="lv")
    with pytest.raises(ValueError):
        lsv_minus_lv_table(models, sim, [])


def test_forward_skew_exposure() -> None:
    """Local vol on a flat 20% surface: the skew tent at the last date moves the never-
    autocalling European KI put leg by its static surface sensitivity (put(B) at the bumped vol
    plus the (K − B) digital with the smile slope, ``∂P/∂K = DF N(−d₂) + vega ∂σ/∂K``) within
    3 stderr; tents at earlier dates leave a 3y European payoff unchanged (zero at T_N); the 3y
    autocall is short skew at its maturity date; the Black–Scholes builder (ATM vol only) sees
    no skew bump at all."""
    from volsto.risk.engine import BSBuilder, LVBuilder, surface_of

    state = flat_state()
    fc = surface_of(state).forward_curve
    discount = fc.rate_curve
    sim = SimConfig(n_paths=40_000, dt_max=1.0 / 50.0, chunk_size=20_000, seed=3)
    factory = state_model_factory(LVBuilder(state), state)
    never = _autocall(discount, autocall_barriers=np.inf, coupons=0.0)
    fse = forward_skew_exposure(never, factory, sim)
    assert list(fse["date"]) == [1, 2, 3] and np.all(fse["achieved"] == 0.01)
    assert np.allclose(fse["slope"], skew_tent_slope(0.01))
    b, k, t = 60.0, S0, 3.0
    f, df_t = float(fc.forward(t)), float(discount.df(t))

    def static_leg(surface) -> float:  # type: ignore[no-untyped-def]
        vol = float(surface.implied_vol_k(np.log(b / f), t))
        h = 1e-4
        dvol_dk = (
            float(surface.implied_vol_k(np.log((b + h) / f), t))
            - float(surface.implied_vol_k(np.log((b - h) / f), t))
        ) / (2 * h)
        d2 = (np.log(f / b) - 0.5 * vol**2 * t) / (vol * np.sqrt(t))
        digital = df_t * float(norm_cdf(-d2)) + float(bs_vega(S0, b, t, vol, 0.02, 0.01)) * dvol_dk
        return -(float(bs_price(S0, b, t, vol, 0.02, 0.01, -1)) + (k - b) * digital) / k

    bumped = state.with_perturbation(skew_tent_perturbation(tuple(OBS), 2, 0.01))
    analytic = static_leg(surface_of(bumped)) - static_leg(surface_of(state))
    row = fse.iloc[2]
    assert analytic < -0.005  # the put wing is richer: the short put leg loses
    assert abs(row["exposure"] - analytic) < 3.0 * row["exposure_stderr"], (row, analytic)
    for i in (0, 1):  # tents at 1y / 2y vanish at 3y: a European payoff at 3y is untouched
        assert abs(fse.iloc[i]["exposure"]) < 3.0 * fse.iloc[i]["exposure_stderr"]
    assert np.all(fse["base_stderr"] > 0) and np.all(fse["bumped_stderr"] > 0)
    ac = _autocall(discount)
    fse_ac = forward_skew_exposure(ac, factory, sim, with_legs=True)
    prod = fse_ac[fse_ac["leg"] == "product"]
    assert len(prod) == 3 and set(fse_ac["leg"]) == {
        "product",
        "autocall_1",
        "autocall_2",
        "autocall_3",
        "bond",
        "put_vanilla",
        "put_digital",
    }
    last = prod.iloc[2]
    assert last["exposure"] < -3.0 * last["exposure_stderr"]  # short skew at maturity
    assert np.all(np.isfinite(prod["exposure"])) and np.all(prod["exposure_stderr"] > 0)
    # the leg exposures sum to the product's (same paths)
    at3 = fse_ac[fse_ac["date"] == 3]
    legs_sum = at3[at3["leg"] != "product"]["exposure"].sum()
    assert legs_sum == pytest.approx(float(last["exposure"]), abs=1e-12)
    zero = forward_skew_exposure(ac, state_model_factory(BSBuilder(state), state), sim)
    assert np.all(zero["exposure"] == 0.0) and np.all(zero["exposure_stderr"] == 0.0)
    with pytest.raises(ValueError):
        forward_skew_exposure(ac, factory, sim, pillars=(1.0, 2.0))
    with pytest.raises(ValueError):
        forward_skew_exposure(ac, factory, sim, size=0.0)
