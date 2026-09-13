"""M4: forward-start options, forward smile analytics and the FVA (SPEC §6, §7, §10).

Black–Scholes closed forms (book eqs. 3.1–3.2: the forward-start price only depends on the
forward volatility), path-wise parity and decomposition identities, the FVA fair strike, and the
ordering forward ATM vol < vol-swap vol < variance-swap vol for ``ρ < 0``.
"""

from __future__ import annotations

import numpy as np
import pytest

from volsto.analytics import (
    forward_atm_vol,
    forward_ratio,
    forward_smile,
    forward_vol_comparison,
    put_wing_table,
)
from volsto.config import BergomiParams, SimConfig
from volsto.engine import FixingIndex, MonteCarlo, PathSet
from volsto.market import DiscountCurve, ForwardCurve, ForwardVarianceCurve, black_price
from volsto.models import BergomiSV, BlackScholes
from volsto.products import FVA, ForwardStartOption, ForwardStartStraddle, forward_start_strip


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


def test_forward_start_payoffs_parity_and_validation(discount: DiscountCurve) -> None:
    times = np.array([0.0, 0.5, 1.0])
    ps = _paths(np.array([[100.0, 110.0, 121.0], [100.0, 90.0, 99.0], [100.0, 100.0, 80.0]]), times)
    idx = FixingIndex(times)
    r = np.array([1.1, 1.1, 0.8])
    df1 = float(discount.df(1.0))
    call = ForwardStartOption(0.5, 1.0, 1.0, "call", discount)
    put = ForwardStartOption(0.5, 1.0, 1.0, "put", discount, notional=3.0)
    np.testing.assert_allclose(call.payoff(ps, idx), df1 * np.maximum(r - 1.0, 0.0))
    np.testing.assert_allclose(put.payoff(ps, idx), 3.0 * df1 * np.maximum(1.0 - r, 0.0))
    # k = 0 call pays the return itself; deferred payment discounts at the pay date
    zero = ForwardStartOption(0.5, 1.0, 0.0, 1, discount, pay_time=2.0)
    np.testing.assert_allclose(zero.payoff(ps, idx), float(discount.df(2.0)) * r)
    assert zero.maturity == 2.0 and call.tau == 0.5
    # a t1 = 0 forward-start is a vanilla on S_T/S_0
    van = ForwardStartOption(0.0, 1.0, 1.0, 1, discount)
    np.testing.assert_allclose(
        van.payoff(ps, idx), df1 * np.maximum(np.array([1.21, 0.99, 0.8]) - 1, 0)
    )
    # parity path by path and the straddle decomposition
    k = 0.95
    c, p = (ForwardStartOption(0.5, 1.0, k, cp, discount) for cp in (1, -1))
    np.testing.assert_allclose(c.payoff(ps, idx) - p.payoff(ps, idx), df1 * (r - k))
    strad = ForwardStartStraddle(0.5, 1.0, k, discount, notional=2.0)
    parts = strad.decompose()
    np.testing.assert_allclose(strad.payoff(ps, idx), sum(q.payoff(ps, idx) for q in parts))
    assert len(forward_start_strip(0.5, 1.0, [0.9, 1.0, 1.1], "put", discount)) == 3
    assert "Forward-start Call" in repr(call) and "paid at 2y" in repr(zero)
    assert "straddle" in repr(strad)
    for bad in (
        lambda: ForwardStartOption(1.0, 0.5, 1.0, 1, discount),
        lambda: ForwardStartOption(0.5, 1.0, -0.1, 1, discount),
        lambda: ForwardStartOption(0.5, 1.0, 0.0, -1, discount),
        lambda: ForwardStartOption(0.5, 1.0, 1.0, 1, discount, pay_time=0.75),
        lambda: ForwardStartStraddle(0.5, 1.0, 0.0, discount),
    ):
        with pytest.raises(ValueError):
            bad()


def test_forward_start_black_scholes(forward_curve: ForwardCurve, fast_sim: SimConfig) -> None:
    """Flat BS: price = DF(T2) Black(F_R, k, τ, σ) (eqs. 3.1–3.2 with a flat term structure);
    the forward smile is flat at σ; the three forward vol measures agree."""
    sigma = 0.2
    model = BlackScholes(sigma, forward_curve)
    discount = forward_curve.rate_curve
    t1, t2 = 0.5, 1.0
    f_r = forward_ratio(model, t1, t2)
    assert f_r == pytest.approx(float(forward_curve.forward(t2) / forward_curve.forward(t1)))
    ks = np.array([0.9, 1.0, 1.1])
    products = [ForwardStartOption(t1, t2, k, cp, discount) for k in ks for cp in (1, -1)]
    res = MonteCarlo(fast_sim).price_many(products, model)
    for prod, r in zip(products, res, strict=True):
        closed = float(discount.df(t2) * black_price(f_r, prod.strike, t2 - t1, sigma, prod.cp))
        assert r.stderr > 0 and abs(r.mean - closed) < 3.5 * r.stderr, (prod, r, closed)
    smile = forward_smile(model, t1, t2, ks, fast_sim)
    assert smile.strikes.size == 4 and np.isclose(smile.strikes, f_r).any()  # ATMF added
    assert np.all(np.abs(smile.vols - sigma) < 3.5 * smile.vol_stderr)
    assert np.all(smile.vol_stderr > 0) and np.all(np.isfinite(smile.log_moneyness))
    v, se = smile.atm
    assert abs(v - sigma) < 3.5 * se
    frame = smile.as_frame()
    assert list(frame.columns)[:3] == ["k", "log_moneyness", "cp"] and len(frame) == 4
    atm = forward_atm_vol(model, t1, t2, fast_sim)
    assert atm == pytest.approx((v, se))  # same paths, same product
    cmp = forward_vol_comparison(model, t1, t2, fast_sim)
    assert abs(cmp.atm_vol - sigma) < 3.5 * cmp.atm_stderr
    # log-Euler is exact for BS: Σ(Δ ln S)² has mean σ² τ; E sqrt(RV) ≤ sqrt(E RV) (Jensen)
    assert abs(cmp.vs_vol - sigma) < 3.5 * cmp.vs_stderr + 1e-9
    assert cmp.volswap_vol <= cmp.vs_vol and cmp.volswap_vol > sigma - 0.01
    table = put_wing_table({"bs": smile, "bs2": smile})
    assert np.allclose(table["spread"], 0.0) and len(table) == 4


def test_fva_fair_strike_is_forward_atm_vol(
    forward_curve: ForwardCurve, fast_sim: SimConfig, discount: DiscountCurve
) -> None:
    sigma = 0.2
    model = BlackScholes(sigma, forward_curve)
    t1, t2 = 0.5, 1.0
    fair = FVA(t1, t2, sigma, discount, forward_curve=forward_curve)
    rich = FVA(t1, t2, 0.25, discount, forward_curve=forward_curve)
    assert fair.moneyness == pytest.approx(forward_ratio(model, t1, t2))
    r_fair, r_rich = MonteCarlo(fast_sim).price_many([fair, rich], model)
    assert abs(r_fair.mean) < 3.5 * r_fair.stderr, r_fair
    assert r_rich.mean < -3.5 * r_rich.stderr  # paying too high a vol loses money
    # premium and decomposition (straddle + cash) path by path
    m, tau = fair.moneyness, t2 - t1
    assert fair.straddle_premium == pytest.approx(
        float(black_price(m, m, tau, sigma, 1) + black_price(m, m, tau, sigma, -1))
    )
    times = np.array([0.0, 0.5, 1.0])
    ps = _paths(np.array([[100.0, 110.0, 121.0], [100.0, 90.0, 99.0], [100.0, 100.0, 80.0]]), times)
    idx = FixingIndex(times)
    np.testing.assert_allclose(
        fair.payoff(ps, idx), sum(q.payoff(ps, idx) for q in fair.decompose())
    )
    explicit = FVA(t1, t2, sigma, discount, moneyness=1.0, notional=10.0)
    r = np.array([1.1, 1.1, 0.8])
    np.testing.assert_allclose(
        explicit.payoff(ps, idx),
        10.0 * float(discount.df(t2)) * (np.abs(r - 1.0) - explicit.straddle_premium),
    )
    assert "FVA" in repr(fair)
    with pytest.raises(ValueError):
        FVA(t1, t2, sigma, discount)  # neither forward curve nor moneyness


def test_forward_vol_ordering_negative_correlation() -> None:
    """Pure 1F Bergomi SV (ω = 3, κ = 1.5, ρ = −0.7, flat ξ₀ = 4%), 6m → 1y: forward ATM vol
    < forward vol-swap vol < forward variance-swap vol (SPEC §10 identity test)."""
    fc = ForwardCurve.flat(100.0, 0.0, 0.0)
    model = BergomiSV(BergomiParams.one_factor(3.0, 1.5, -0.7), ForwardVarianceCurve.flat(0.04), fc)
    sim = SimConfig(n_paths=100_000, dt_max=1.0 / 200.0, chunk_size=50_000, seed=11)
    cmp = forward_vol_comparison(model, 0.5, 1.0, sim)
    assert cmp.atm_vol < cmp.volswap_vol - 2.0 * np.hypot(cmp.atm_stderr, cmp.volswap_stderr), cmp
    assert cmp.volswap_vol < cmp.vs_vol, cmp
    # the flat ξ₀ pins the variance-swap level: sqrt(E RV) = 20% up to discretisation
    assert abs(cmp.vs_vol - 0.20) < 3.5 * cmp.vs_stderr + 5e-4, cmp
