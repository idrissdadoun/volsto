"""The barrier study on 2007–2026 (:mod:`volsto.studies.barrier_history`): its pure functions —
dates, structures and their model-free facts, payoffs, knock rules, hedge accounting, the
bucketing and the bootstrap, the moved surface, the bulk marks, the path functionals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from volsto.market.barrier_bs import bs_barrier_price
from volsto.market.curves import ForwardCurve
from volsto.market.surface import ImpliedSurface
from volsto.studies import barrier_history as bh


class FlatSurface(ImpliedSurface):
    """One implied vol for every strike and maturity, plus a linear skew in log-moneyness."""

    def __init__(self, vol: float, spot: float, r: float, q: float, skew: float = 0.0) -> None:
        fc = ForwardCurve.flat(spot, r, q)
        super().__init__(fc, fc.rate_curve, 10.0)
        self.vol, self.skew = vol, skew

    def total_variance(self, k: Any, T: Any) -> Any:
        v = self.vol + self.skew * np.asarray(k, dtype=np.float64)
        return v * v * np.asarray(T, dtype=np.float64)


def test_dates() -> None:
    cal = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2023-12-25", "2024-12-31")]
    cal.remove("2024-01-01")  # a holiday: the week's first trading day is the Tuesday
    entries = bh.entry_dates(cal, "2024-01-01", "2024-02-29")
    assert entries[:3] == ["2024-01-02", "2024-01-08", "2024-01-15"] and len(entries) == 9
    assert bh.monthly_subset(entries) == ["2024-01-02", "2024-02-05"]
    assert bh.add_months(pd.Timestamp("2024-01-31").date(), 1).isoformat() == "2024-02-29"
    assert bh.add_months(pd.Timestamp("2023-11-30").date(), 3).isoformat() == "2024-02-29"
    assert bh.expiry_date("2024-01-02", 1, cal) == "2024-02-02"
    assert bh.expiry_date("2024-01-02", 3, cal) == "2024-04-02"
    assert bh.expiry_date("2024-01-08", 3, cal) == "2024-04-08"
    assert bh.expiry_date("2024-01-12", 2, cal) == "2024-03-12"
    assert bh.expiry_date("2024-01-06", 1, cal) == "2024-02-06"
    assert bh.expiry_date("2024-02-10", 1, cal) == "2024-03-08"  # the 10th is a Sunday
    with pytest.raises(ValueError, match="no trading day"):
        bh.expiry_date("2024-12-31", 1, cal)
    assert bh.year_fraction("2024-01-02", "2025-01-01") == 1.0
    assert bh.trading_calendar(["2024-01-04"], "2024-01-09") == [
        "2024-01-04",
        "2024-01-05",
        "2024-01-08",
        "2024-01-09",
    ]
    assert bh.period_of("2011-10-03") == "2007 to 2011-10-03"
    assert bh.period_of("2019-02-05") == "2019-02-05 to 2021-05-27"
    assert bh.period_of("2024-01-02") == "2021-05-28 on"
    assert bh.seed_of("2024-01-02") == 20240102


def test_barrier_levels() -> None:
    up = bh.barrier_levels(1, 100.0, 0.2, 0.25)
    assert up["p105"] == pytest.approx(105.0) and up["p120"] == pytest.approx(120.0)
    assert up["s1"] == pytest.approx(100.0 * np.exp(0.2 * 0.5))
    down = bh.barrier_levels(-1, 100.0, 0.2, 0.25)
    assert down["p95"] == pytest.approx(95.0) and down["s2"] == pytest.approx(
        100.0 * np.exp(-2 * 0.2 * 0.5)
    )
    assert len(up) == len(down) == 9
    with pytest.raises(ValueError, match="beyond the strike"):
        bh.structure_legs(1, 100.0, 95.0, 100.0)


@pytest.mark.parametrize("side,B", [(1, 110.0), (-1, 90.0)])
def test_structure_payoffs_against_the_knock_out(side: int, B: float) -> None:
    """Spec §3.4 facts: every fly with its short strike at the barrier pays the knock-out's
    payoff when the barrier is not finished beyond and never less; the tight limit is the
    European knock-out; the ladder identities."""
    K = 100.0
    legs = bh.structure_legs(side, K, B, K)
    s = np.linspace(60.0, 140.0, 1601)
    inside = side * (s - B) <= 0
    vanilla = np.maximum(side * (s - K), 0.0)
    b6 = bh.legs_payoff(legs["B6"], side, s)
    assert np.allclose(b6, vanilla * inside, atol=1e-12)  # the European knock-out
    w = abs(B - K)
    for n in bh.FLY_N:
        fly = bh.legs_payoff(legs[f"B5_{n}"], side, s)
        assert np.allclose(fly[inside], vanilla[inside]) and (fly >= -1e-12).all()
        assert (fly >= b6 - 1e-12).all()  # the overshoot piece is a payoff, never negative
        over = np.maximum(w - (n - 1) * side * (s - B), 0.0) * ~inside
        assert np.allclose(fly - b6, over, atol=1e-9)  # OS(n) of §6.3
    # the knock-out pays at most the European knock-out; the fly at least (never knocked or not)
    assert bh.knock_out_payoff(side, K, B - side * 5.0, knocked=False) == pytest.approx(w - 5.0)
    assert bh.knock_out_payoff(side, K, B - side * 5.0, knocked=True) == 0.0
    # ladder: spread − fly(2) = the option at B minus the option at 2B − K; fly(2) − ratio = wing
    up = bh.legs_payoff(legs["B3"], side, s) - bh.legs_payoff(legs["B5_2"], side, s)
    far = 2 * B - K
    assert np.allclose(
        up, np.maximum(side * (s - B), 0.0) - np.maximum(side * (s - far), 0.0), atol=1e-9
    )
    wing = bh.legs_payoff(legs["B5_2"], side, s) - bh.legs_payoff(legs["B4"], side, s)
    assert np.allclose(wing, np.maximum(side * (s - far), 0.0), atol=1e-9)
    # the halfway ratio is a sub-hedge payoff: below the European knock-out everywhere
    assert (bh.legs_payoff(legs["E13"], side, s) <= b6 + 1e-9).all()
    # C7 is C8 with its two far legs merged at B: same payoff at the barrier side of B
    c7, c8 = bh.legs_payoff(legs["C7"], side, s), bh.legs_payoff(legs["C8"], side, s)
    assert np.allclose(c7[inside], c8[inside], atol=1e-9)
    # the forward and the vanilla
    assert np.allclose(bh.legs_payoff(legs["FWD"], side, s), s - K)
    assert np.allclose(bh.legs_payoff(legs["VAN"], side, s), vanilla)


@pytest.mark.parametrize("side,B", [(1, 108.0), (-1, 93.0)])
@pytest.mark.parametrize("vol", [0.10, 0.30])
@pytest.mark.parametrize("T", [1.0 / 12.0, 0.5])
def test_c8_is_the_continuous_knock_out_in_black_scholes(
    side: int, B: float, vol: float, T: float
) -> None:
    """Pilot check 1: with zero carry the put-call-symmetry hedge C8 is worth zero when spot is
    at the barrier and its price is the closed-form continuous knock-out's, both sides."""
    K = 100.0
    legs = bh.structure_legs(side, K, B, K)["C8"]
    assert abs(bh.bs_legs_value(legs, side, B, vol, T)) < 1e-8 * K
    assert abs(bh.bs_legs_value(legs, side, B, vol, 0.37 * T)) < 1e-8 * K  # at any later date
    closed = float(
        bs_barrier_price(K, K, B, T, vol, 0.0, 0.0, side, "up" if side > 0 else "down", "out")
    )
    assert bh.bs_legs_value(legs, side, K, vol, T) == pytest.approx(closed, abs=1e-8 * K)


@pytest.mark.parametrize("side,B", [(1, 110.0), (-1, 90.0)])
def test_model_free_band_and_fly_ordering(side: int, B: float) -> None:
    """Brown–Hobson–Rogers: ``LB ≤ knock-out ≤ B6`` (here the Black–Scholes continuous
    knock-out on a flat surface), and the flies fall towards B6 as n rises."""
    K, vol, T = 100.0, 0.2, 0.5
    surf = FlatSurface(vol, K, 0.0, 0.0)
    legs = bh.structure_legs(side, K, B, K)
    value = {name: bh.legs_value(v, side, surf, T) for name, v in legs.items()}
    chain = [value[f"B5_{n}"] for n in (6, 4, 3, 2)]
    assert value["B6"] <= chain[0] <= chain[1] <= chain[2] <= chain[3] <= value["B3"]
    ko = float(
        bs_barrier_price(K, K, B, T, vol, 0.0, 0.0, side, "up" if side > 0 else "down", "out")
    )
    lb, at = bh.lower_bound(side, K, B, surf, T)
    assert 0.0 <= lb <= ko <= value["B6"] and min(K, B) <= at <= max(K, B)
    assert value["E13"] <= lb + 1e-12  # the halfway ratio is one point of the search
    # the centred-difference digital (half-width 0.5 % of the strike) against the exact one:
    # measured 5.6e-4 and 3.5e-4 on these two cells (w = 10 times the digital's error)
    assert value["B6"] == pytest.approx(bh.bs_legs_value(legs["B6"], side, K, vol, T), abs=1e-3)
    # premium-matched structures hit the premium
    matched = bh.matched_legs(side, K, B, ko, surf, T)
    assert "D10" in matched  # the fly is absent when no wing reaches the premium
    for name, m_legs in matched.items():
        assert bh.legs_value(m_legs, side, surf, T) == pytest.approx(ko, abs=1e-8), name
    # matched to the halfway fly's own premium, the matched fly is the halfway fly (wing at B)
    both = bh.matched_legs(side, K, B, value["E12"], surf, T)
    assert set(both) == {"D10", "D11"}
    assert sorted(leg.strike for leg in both["D11"]) == pytest.approx(
        sorted([K, 0.5 * (K + B), B]), abs=1e-6
    )
    assert bh.legs_from_json(bh.legs_to_json(matched)) == matched


def test_knock_dates_and_hedge_accounting() -> None:
    days = ["d1", "d2", "d3", "d4"]
    ohlc = pd.DataFrame(
        {
            "open": [100.0, 101.0, 104.0, 103.0],
            "high": [101.0, 105.0, 106.5, 104.0],
            "low": [99.0, 100.5, 103.0, 96.0],
            "close": [100.5, 104.0, 106.0, 97.0],
        },
        index=days,
    )
    # calls: a high at the barrier knocks the continuous one; only a close strictly above the
    # daily one
    assert bh.knock_dates(1, 105.0, ohlc, days) == ("d3", "d2")
    assert bh.knock_dates(1, 106.0, ohlc, days) == (None, "d3")  # a close at B is not a knock
    assert bh.knock_dates(1, 110.0, ohlc, days) == (None, None)
    assert bh.knock_dates(-1, 97.0, ohlc, days) == (None, "d4")
    assert bh.knock_dates(-1, 98.0, ohlc, days) == ("d4", "d4")
    assert bh.knock_dates(1, 105.0, ohlc, days[2:]) == ("d3", "d3")  # only the observed dates
    # the hedge of a forward: delta one, every interval, closed at the official close
    fwd = np.array([100.0, 102.0, 99.0])
    h = bh.hedge_pnl(np.ones(3), fwd, closing_forward=104.0)
    assert h.tolist() == [-2.0, 3.0, -5.0]
    K, payoff, premium_at_T = 95.0, 104.0 - 95.0, 100.0 - 95.0  # P0 / DF0 = F0 − K
    assert payoff - premium_at_T + h.sum() == 0.0  # spec §5.3: exact for a forward
    assert K == 95.0


def test_bucketing_and_bootstrap() -> None:
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 10.0, 3.5, np.nan])
    lab = bh.expanding_tercile(x, min_history=3)
    assert lab.tolist()[:3] == ["no history"] * 3
    assert lab.tolist()[3:] == ["high", "high", "high", "low", "high", "mid", "no history"]
    assert [bh.sd_bin(v) for v in (0.2, 0.5, 0.99, 1.0, 1.49, 1.5, 3.0)] == [
        "<0.5",
        "0.5-1",
        "0.5-1",
        "1-1.5",
        "1-1.5",
        ">1.5",
        ">1.5",
    ]
    assert bh.sd_bin(float("nan")) == "nan"
    rng = np.random.default_rng(0)
    iid = rng.normal(1.0, 2.0, size=2000)
    mean, se = bh.block_bootstrap(iid, block=1, seed=1)
    assert mean == pytest.approx(iid.mean()) and se == pytest.approx(2.0 / np.sqrt(2000), rel=0.1)
    # overlapping windows: the block standard error is larger than the i.i.d. one
    overlap = np.convolve(rng.normal(size=2012), np.ones(13), mode="valid")
    _, se_block = bh.block_bootstrap(overlap, block=13, seed=1)
    _, se_iid = bh.block_bootstrap(overlap, block=1, seed=1)
    assert se_block > 2.0 * se_iid
    assert np.isnan(bh.block_bootstrap(np.array([np.nan]), 5)[0])
    assert bh.effective_sample(1000, 7.0, 365.0) == pytest.approx(1000 * 7 / 365)
    assert bh.effective_sample(10, 30.0, 7.0) == 10.0


def test_moved_surface_keeps_the_vol_of_every_strike() -> None:
    base = FlatSurface(0.2, 100.0, 0.03, 0.01, skew=-0.3)
    up = bh.MovedSurface(base, 1.01)
    K = np.array([80.0, 100.0, 125.0])
    for T in (0.1, 1.0):
        assert np.allclose(up.implied_vol(K, T), base.implied_vol(K, T), atol=1e-14)
        assert float(up.forward(T)) == pytest.approx(1.01 * float(base.forward(T)))
        assert float(up.discount.df(T)) == float(base.discount.df(T))
    # the bad-day carry: the vols of the same expiry dates, one day later
    dt = 1.0 / 365.0
    later = bh.MovedSurface(base, 0.98, dt)
    assert np.allclose(later.implied_vol(K, 0.5), base.implied_vol(K, 0.5 + dt), atol=1e-14)
    assert float(later.forward(0.5)) == pytest.approx(0.98 * float(base.forward(0.5 + dt)))
    assert later.max_maturity == pytest.approx(10.0 - dt)
    vega = bh.ParallelVolSurface(base, 0.01)
    assert np.allclose(vega.implied_vol(K, 1.0), base.implied_vol(K, 1.0) + 0.01, atol=1e-14)


def test_bulk_values_match_the_leg_by_leg_marks() -> None:
    surf = FlatSurface(0.18, 100.0, 0.04, 0.015, skew=-0.25)
    cells = []
    Ts = []
    for side, B, T in ((1, 112.0, 0.25), (-1, 88.0, 0.5), (1, 104.0, 1.0)):
        legs = bh.structure_legs(side, 100.0, B, 100.0)
        legs.update(bh.matched_legs(side, 100.0, B, 0.3, surf, T))
        cells.append((side, legs))
        Ts.append(T)
    pieces = bh.expand_legs(cells)
    piece_T = np.asarray(Ts)[pieces["slot"] // len(bh.STRUCTURES)]
    ratios = (1.0, 1.01, 0.99)
    bulk = bh.bulk_values(pieces, piece_T, surf, ratios)
    assert bulk.shape == (3, 3, len(bh.STRUCTURES))
    for c, (side, legs) in enumerate(cells):
        for s, name in enumerate(bh.STRUCTURES):
            if name not in legs:
                assert np.isnan(bulk[:, c, s]).all()
                continue
            for r, ratio in enumerate(ratios):
                want = bh.legs_value(legs[name], side, surf, Ts[c], ratio)
                assert bulk[r, c, s] == pytest.approx(want, abs=1e-10)
    # pilot check 3: the sticky-strike bump of the vanilla is the Black delta at its strike's vol
    i = bh.STRUCTURES.index("VAN")
    F, df = float(surf.forward(0.25)), float(surf.discount.df(0.25))
    delta_f = (bulk[1, 0, i] - bulk[2, 0, i]) / (2 * bh.BUMP * F * df)
    vol = float(np.asarray(surf.implied_vol(100.0, 0.25)).reshape(()))
    d1 = (np.log(F / 100.0) + 0.5 * vol * vol * 0.25) / (vol * 0.5)
    from scipy.special import ndtr

    assert delta_f == pytest.approx(float(ndtr(d1)), abs=2e-3)
    # the forward's delta is exactly one
    j = bh.STRUCTURES.index("FWD")
    assert (bulk[1, 0, j] - bulk[2, 0, j]) / (2 * bh.BUMP * F * df) == pytest.approx(1.0)


def test_knock_out_values_on_hand_paths() -> None:
    """Four paths, two future days: the daily knock is strict on the closes, the continuous one
    reads the running extreme; the control variate is exact when the control is the payoff."""
    close = np.array([[104.0, 108.0], [111.0, 107.0], [101.0, 110.0], [96.0, 103.0]])
    logc = np.log(close)
    cont_max = np.log(np.array([[105.0, 110.0], [112.0, 112.0], [102.0, 110.0], [100.0, 104.0]]))
    paths = bh.DayPaths(
        times=np.array([1 / 365, 2 / 365]),
        close=close,
        run_max_close=np.maximum.accumulate(logc, axis=1),
        run_min_close=np.minimum.accumulate(logc, axis=1),
        run_max_cont=cont_max,
        run_min_cont=np.minimum.accumulate(logc, axis=1),
        antithetic=False,
    )
    K, B = np.array([100.0]), np.array([110.0])
    v = bh.knock_out_values(paths, 1, 1, K, B)
    # daily: path 2 closed at 111 > 110 on day 1: knocked; path 3 closes at 110: not knocked
    assert v["a1"][0] == pytest.approx((8.0 + 0.0 + 10.0 + 3.0) / 4)
    # continuous: paths 1 and 3 touch 110 (at or above), path 2 is beyond
    assert v["a2"][0] == pytest.approx(3.0 / 4)
    assert v["p1"][0] == 0.25 and v["p2"][0] == 0.75
    assert v["e"][0] == pytest.approx((8.0 + 7.0 + 10.0 + 3.0) / 4)
    # put side on the same closes
    w = bh.knock_out_values(paths, 1, -1, np.array([109.0]), np.array([100.0]))
    # payoffs 1, 2, 0 and 6; path 4 closed at 96 < 100 on day 1: knocked
    assert w["a1"][0] == pytest.approx(3.0 / 4) and w["p1"][0] == 0.25
    # control variate: with the true mean of the control the estimate moves by β × its error
    c = bh.knock_out_values(paths, 1, 1, K, B, european=np.array([6.0]))
    assert c["a1"][0] == pytest.approx(v["a1"][0] - c["a1_beta"][0] * (v["e"][0] - 6.0))
    same = bh.knock_out_values(paths, 1, 1, K, B, european=v["e"])
    assert same["a1"][0] == pytest.approx(v["a1"][0]) and same["a1_se"][0] <= v["a1_se"][0]
    fixed = bh.knock_out_values(
        paths,
        1,
        1,
        K,
        B,
        european=np.array([6.0]),
        beta={"a1": np.array([1.0]), "a2": np.array([0.0])},
    )
    assert fixed["a1"][0] == pytest.approx(v["a1"][0] - (v["e"][0] - 6.0))
    assert fixed["a2"][0] == pytest.approx(v["a2"][0])


def test_surface_readings() -> None:
    surf = FlatSurface(0.2, 100.0, 0.02, 0.0, skew=-0.4)
    m = bh.surface_metrics(surf)
    assert m["atm_3m"] == pytest.approx(0.2) and m["skew_1y"] == pytest.approx(-0.4)
    assert m["conv_6m"] == pytest.approx(0.0, abs=1e-12) and "atm_2y" in m
    c = bh.entry_conditions(surf, 0.25, 1)
    sd = 0.2 * 0.5
    assert c["skew_side"] == pytest.approx(100 * -0.4 * sd)
    assert bh.entry_conditions(surf, 0.25, -1)["skew_side"] == pytest.approx(100 * 0.4 * sd)
    assert c["convexity"] == pytest.approx(0.0, abs=1e-10) and c["term_slope"] == pytest.approx(0.0)
    assert c["fwd_vol_2nd_half"] == pytest.approx(0.2) and c["atm_skew"] == pytest.approx(-0.4)
    assert c["skew_decay"] == pytest.approx(0.0, abs=1e-12)  # the same skew at 1m and 2y


def test_position_outcome() -> None:
    """A forward struck at K, delta one: zero hedged P&L on any path, at every date (pilot
    check 2); a position knocked or sold early stops hedging there."""
    K, df0 = 95.0, 0.98
    fwd = np.array([100.0, 103.0, 99.0, 101.0])
    df = np.array([0.98, 0.985, 0.99, 0.995])
    value = df * (fwd - K)
    out = bh.position_outcome(
        value,
        np.ones(4),
        fwd,
        df,
        premium=float(value[0]),
        df0=df0,
        terminal=104.0 - K,
        closing_forward=104.0,
        n_hedge=4,
        n_life=4,
    )
    assert out["pnl_u"] == pytest.approx(4.0) and out["pnl_h"] == pytest.approx(0.0, abs=1e-12)
    assert out["hedge"] == pytest.approx(-4.0)
    assert max(abs(out[k]) for k in ("best", "worst", "run_25", "run_50", "run_75")) < 1e-12
    # an option-like position: knocked on the third snapshot, hedge closed at 98
    v = np.array([3.0, 4.0, 2.5, 0.0])
    delta = np.array([0.5, 0.6, 0.4, 0.0])
    ko = bh.position_outcome(
        v,
        delta,
        fwd,
        df,
        premium=3.0,
        df0=df0,
        terminal=0.0,
        closing_forward=98.0,
        n_hedge=2,
        n_life=4,
    )
    hedge = -0.5 * (103.0 - 100.0) - 0.6 * (98.0 - 103.0)
    assert ko["pnl_u"] == pytest.approx(-3.0 / df0) and ko["hedge"] == pytest.approx(hedge)
    assert ko["pnl_h"] == pytest.approx(-3.0 / df0 + hedge)
    run1 = 4.0 / 0.985 - 3.0 / df0 - 0.5 * 3.0
    assert ko["run_25"] == pytest.approx(run1)  # one quarter of four intervals: snapshot 1
    assert ko["run_50"] == ko["run_75"] == ko["pnl_h"]  # after the knock: the final P&L
    assert ko["best"] == pytest.approx(max(0.0, run1, ko["pnl_h"]))
    none = bh.position_outcome(
        v,
        delta,
        fwd,
        df,
        premium=3.0,
        df0=df0,
        terminal=1.0,
        closing_forward=0.0,
        n_hedge=0,
        n_life=4,
    )
    assert none["pnl_h"] == none["pnl_u"] == pytest.approx(1.0 - 3.0 / df0)
