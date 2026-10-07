"""Vanilla structures (``volsto/products/structures.py``): payoff shapes, the model-free
surface price against the engine, premium matching, the European knock-out."""

from __future__ import annotations

import numpy as np
import pytest

from volsto.config import SimConfig
from volsto.engine.mc import MonteCarlo
from volsto.market.bs import black_price
from volsto.market.curves import ForwardCurve
from volsto.models.bs import BlackScholes
from volsto.products.barrier import KnockOutOption
from volsto.products.structures import (
    VanillaStructure,
    call_fly,
    call_ratio,
    call_spread,
    european_knock_out,
    european_knock_out_surface_price,
    fly_width_for_premium,
    fly_wing_for_premium,
    put_fly,
    put_ratio,
    put_spread,
    ratio_for_premium,
    straddle,
    surface_digital,
    units_for_premium,
)


@pytest.fixture(scope="module")
def world(ssvi):  # type: ignore[no-untyped-def]
    fc = ssvi.forward_curve
    return ssvi, fc, fc.rate_curve


def test_payoff_shapes(world) -> None:  # type: ignore[no-untyped-def]
    _, _fc, dc = world
    s = np.array([80.0, 95.0, 100.0, 105.0, 110.0, 120.0])
    fly = call_fly(100.0, 105.0, 110.0, 1.0, dc)
    assert fly.leg_weights == (1.0, -2.0, 1.0)
    np.testing.assert_allclose(fly.payoff_at(s), [0, 0, 0, 5, 0, 0])
    bw = call_fly(100.0, 104.0, 110.0, 1.0, dc)  # broken wing: zero beyond 110 by construction
    np.testing.assert_allclose(bw.payoff_at([110.0, 150.0]), [0.0, 0.0], atol=1e-12)
    assert bw.max_payoff(80.0, 150.0) == pytest.approx(4.0)
    pfly = put_fly(90.0, 95.0, 100.0, 1.0, dc)
    np.testing.assert_allclose(pfly.payoff_at([80.0, 90.0, 95.0, 100.0]), [0, 0, 5, 0])
    cs = call_spread(100.0, 110.0, 1.0, dc)
    np.testing.assert_allclose(cs.payoff_at(s), [0, 0, 0, 5, 10, 10])
    ps = put_spread(90.0, 100.0, 1.0, dc)
    np.testing.assert_allclose(ps.payoff_at([80.0, 95.0, 105.0]), [10, 5, 0])
    r = call_ratio(100.0, 105.0, 2.0, 1.0, dc)
    np.testing.assert_allclose(r.payoff_at([100.0, 105.0, 110.0, 120.0]), [0, 5, 0, -10])
    pr = put_ratio(95.0, 100.0, 2.0, 1.0, dc)
    np.testing.assert_allclose(pr.payoff_at([100.0, 95.0, 90.0, 80.0]), [0, 5, 0, -10])
    st = straddle(100.0, 1.0, dc)
    np.testing.assert_allclose(st.payoff_at([90.0, 100.0, 115.0]), [10, 0, 15])
    with pytest.raises(ValueError):
        call_fly(100.0, 110.0, 105.0, 1.0, dc)
    with pytest.raises(ValueError):
        call_ratio(100.0, 110.0, 0.0, 1.0, dc)
    aged = fly.aged(0.25)
    assert isinstance(aged, VanillaStructure) and pytest.approx(0.75) == aged.T
    assert "call fly" in repr(fly)


def test_surface_price_is_the_sum_of_black_prices(world) -> None:  # type: ignore[no-untyped-def]
    surf, fc, dc = world
    T = 0.5
    fly = call_fly(100.0, 105.0, 110.0, T, dc, notional=2.0)
    expected = 0.0
    f, df = float(fc.forward(T)), float(dc.df(T))
    for k, w in zip(fly.strikes, fly.leg_weights, strict=True):
        sig = float(surf.implied_vol(k, T))
        expected += w * float(black_price(f, k, T, sig, 1, df))
    assert fly.surface_price(surf) == pytest.approx(2.0 * expected, rel=1e-10)
    assert fly.surface_vega(surf) > 0 or fly.surface_vega(surf) < 0  # finite, signed


def test_surface_price_matches_black_scholes_engine(world) -> None:  # type: ignore[no-untyped-def]
    """On a flat-vol world the model-free price is the Black–Scholes Monte Carlo price."""
    from volsto.market.surface import SSVISurface

    fc = ForwardCurve.flat(100.0, 0.02, 0.0)
    dc = fc.rate_curve
    ts = np.array([0.25, 0.5, 1.0, 2.0])
    surf = SSVISurface(ts, 0.04 * ts, 0.0, 0.0, 0.5, fc, dc)
    ratio = call_ratio(100.0, 110.0, 2.0, 0.5, dc)
    model = BlackScholes(0.2, fc)
    res = MonteCarlo(SimConfig(n_paths=40_000, chunk_size=20_000, seed=5, dt_max=0.05)).price(
        ratio, model
    )
    assert abs(res.mean - ratio.surface_price(surf)) < 3 * res.stderr


def test_premium_matching(world) -> None:  # type: ignore[no-untyped-def]
    surf, _fc, dc = world
    T = 0.5
    target = 0.4
    n = ratio_for_premium(100.0, 105.0, T, surf, target, cp=1)
    assert call_ratio(100.0, 105.0, n, T, dc).surface_price(surf) == pytest.approx(target, abs=1e-9)
    m = ratio_for_premium(95.0, 100.0, T, surf, target, cp=-1)
    assert put_ratio(95.0, 100.0, m, T, dc).surface_price(surf) == pytest.approx(target, abs=1e-9)
    # a fly with the inner strikes held is worth at least the European knock-out at the middle
    # strike: the wing solve reaches premiums above that floor only
    floor = european_knock_out_surface_price(surf, 100.0, 105.0, "up", 1, T)
    top = call_spread(100.0, 105.0, T, dc).surface_price(surf)
    assert 0 < floor < top
    mid = 0.5 * (floor + top)
    fly = fly_wing_for_premium(100.0, 105.0, T, surf, mid, dc, cp=1)
    assert fly.strikes[2] > 105.0 and fly.surface_price(surf) == pytest.approx(mid, abs=1e-7)
    pfloor = european_knock_out_surface_price(surf, 100.0, 95.0, "down", -1, T)
    ptop = put_spread(95.0, 100.0, T, dc).surface_price(surf)
    pfly = fly_wing_for_premium(95.0, 100.0, T, surf, 0.5 * (pfloor + ptop), dc, cp=-1)
    assert pfly.strikes[0] < 95.0
    assert pfly.surface_price(surf) == pytest.approx(0.5 * (pfloor + ptop), abs=1e-7)
    with pytest.raises(ValueError):
        fly_wing_for_premium(100.0, 105.0, T, surf, 0.5 * floor, dc, cp=1)
    # the width solve reaches any premium below the fly spanning [k, 2k]
    small = 0.5 * floor
    wf = fly_width_for_premium(100.0, T, surf, small, dc, cp=1)
    assert 100.0 < wf.strikes[2] < 110.0 and wf.surface_price(surf) == pytest.approx(
        small, abs=1e-8
    )
    assert wf.strikes[1] == pytest.approx(0.5 * (100.0 + wf.strikes[2]))
    wp = fly_width_for_premium(100.0, T, surf, small, dc, cp=-1)
    assert 90.0 < wp.strikes[0] < 100.0 and wp.surface_price(surf) == pytest.approx(small, abs=1e-8)
    u = units_for_premium(call_fly(100.0, 105.0, 110.0, T, dc), surf, target)
    assert u * call_fly(100.0, 105.0, 110.0, T, dc).surface_price(surf) == pytest.approx(target)


def test_european_knock_out_and_digital(world) -> None:  # type: ignore[no-untyped-def]
    surf, _fc, dc = world
    T = 0.5
    eko = european_knock_out(100.0, 110.0, "up", 1, T, dc)
    assert isinstance(eko, KnockOutOption) and eko.fixing_times.tolist() == [T]
    # the digital of a flat smile is the Black digital; on the skewed surface the centred
    # difference includes the smile slope (compare with a tighter width: converged)
    d1 = surface_digital(surf, 110.0, T, 1, rel_width=0.005)
    d2 = surface_digital(surf, 110.0, T, 1, rel_width=0.001)
    assert 0 < d1 < 1 and abs(d1 - d2) < 2e-3
    p = european_knock_out_surface_price(surf, 100.0, 110.0, "up", 1, T)
    spread = call_spread(100.0, 110.0, T, dc).surface_price(surf)
    assert 0 < p < spread
    # against the engine under local vol: the European knock-out is model-free
    from volsto.config import LocalVolConfig
    from volsto.market.dupire import LocalVolSurface
    from volsto.models.localvol import LocalVol

    lv = LocalVol(LocalVolSurface.from_implied(surf, LocalVolConfig(t_max=1.5, n_t=60, n_k=1201)))
    res = MonteCarlo(SimConfig(n_paths=40_000, chunk_size=20_000, seed=7, dt_max=1 / 52)).price(
        eko, lv
    )
    assert abs(res.mean - p) < 3 * res.stderr + 0.02
    with pytest.raises(NotImplementedError):
        european_knock_out_surface_price(surf, 100.0, 110.0, "up", -1, T)
