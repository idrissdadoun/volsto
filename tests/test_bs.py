"""SPEC §10: put–call parity, implied-vol round trip, Greeks vs finite differences."""

from __future__ import annotations

import numpy as np
import pytest

from volsto.market import (
    black_price,
    bs_delta,
    bs_gamma,
    bs_price,
    bs_rho,
    bs_theta,
    bs_vanna,
    bs_vega,
    bs_volga,
    implied_vol,
)
from volsto.market.bs import strike_from_delta

S, R, Q = 100.0, 0.02, 0.01


def test_put_call_parity() -> None:
    K = np.linspace(50, 160, 23)[:, None]
    T = np.array([0.05, 0.5, 2.0])[None, :]
    vol = 0.25
    F = S * np.exp((R - Q) * T)
    df = np.exp(-R * T)
    c = bs_price(S, K, T, vol, R, Q, 1)
    p = bs_price(S, K, T, vol, R, Q, -1)
    np.testing.assert_allclose(c - p, df * (F - K), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(black_price(F, K, T, vol, 1, df), c, rtol=1e-14)


def test_zero_vol_and_zero_time_are_intrinsic() -> None:
    assert black_price(100.0, 90.0, 1.0, 0.0, 1) == pytest.approx(10.0)
    assert black_price(100.0, 90.0, 0.0, 0.3, -1) == 0.0


def test_implied_vol_round_trip() -> None:
    vols = np.array([0.03, 0.1, 0.2, 0.5, 1.2])[:, None, None, None]
    T = np.array([0.01, 0.25, 1.0, 5.0])[None, :, None, None]
    k = np.linspace(-1.0, 1.0, 21)[None, None, :, None]
    cp = np.array([1.0, -1.0])[None, None, None, :]
    F = 100.0
    K = F * np.exp(k)
    df = 0.97
    price = black_price(F, K, T, vols, cp, df)
    iv = implied_vol(price, F, K, T, cp, df)
    # where the time value is resolvable in double precision, the vol must be recovered
    intrinsic = df * np.maximum(cp * (F - K), 0.0)
    tv = price - intrinsic
    resolvable = tv > 1e-10 * F
    err = np.abs(iv - np.broadcast_to(vols, iv.shape))
    assert np.all(err[resolvable] < 1e-7), err[resolvable].max()
    # everywhere: repricing at the recovered vol reproduces the price to round-off
    finite = np.isfinite(iv)
    reprice = black_price(F, K, T, np.where(finite, iv, 0.0), cp, df)
    np.testing.assert_allclose(reprice[finite], price[finite], atol=1e-10 * F)


def test_implied_vol_out_of_bounds_is_nan() -> None:
    out = implied_vol([-1.0, 200.0, 5.0], 100.0, 100.0, [1.0, 1.0, 0.0], 1)
    assert np.all(np.isnan(out))
    assert implied_vol(0.0, 100.0, 120.0, 1.0, 1) == 0.0


@pytest.mark.parametrize("cp", [1, -1])
@pytest.mark.parametrize("K", [80.0, 100.0, 125.0])
def test_greeks_vs_finite_differences(cp: int, K: float) -> None:
    T, vol = 0.75, 0.3
    h = 1e-4

    def p(S_: float = S, vol_: float = vol, T_: float = T, r_: float = R) -> float:
        return float(bs_price(S_, K, T_, vol_, r_, Q, cp))

    delta_fd = (p(S + h) - p(S - h)) / (2 * h)
    h2 = 1e-2  # second differences: larger step to keep round-off below truncation error
    gamma_fd = (p(S + h2) - 2 * p() + p(S - h2)) / h2**2
    vega_fd = (p(vol_=vol + h) - p(vol_=vol - h)) / (2 * h)
    theta_fd = -(p(T_=T + h) - p(T_=T - h)) / (2 * h)
    rho_fd = (p(r_=R + h) - p(r_=R - h)) / (2 * h)
    hv = 1e-3
    vanna_fd = (
        p(S + h2, vol + hv) - p(S + h2, vol - hv) - p(S - h2, vol + hv) + p(S - h2, vol - hv)
    ) / (4 * h2 * hv)
    volga_fd = (p(vol_=vol + hv) - 2 * p() + p(vol_=vol - hv)) / hv**2

    assert float(bs_delta(S, K, T, vol, R, Q, cp)) == pytest.approx(delta_fd, rel=1e-6, abs=1e-8)
    assert float(bs_gamma(S, K, T, vol, R, Q)) == pytest.approx(gamma_fd, rel=1e-5)
    assert float(bs_vega(S, K, T, vol, R, Q)) == pytest.approx(vega_fd, rel=1e-6)
    assert float(bs_theta(S, K, T, vol, R, Q, cp)) == pytest.approx(theta_fd, rel=1e-6, abs=1e-8)
    assert float(bs_rho(S, K, T, vol, R, Q, cp)) == pytest.approx(rho_fd, rel=1e-6, abs=1e-8)
    assert float(bs_vanna(S, K, T, vol, R, Q)) == pytest.approx(vanna_fd, rel=1e-4, abs=1e-6)
    assert float(bs_volga(S, K, T, vol, R, Q)) == pytest.approx(volga_fd, rel=1e-4, abs=1e-6)


def test_strike_from_delta_round_trip() -> None:
    T, vol = 0.5, 0.22
    F = S * np.exp((R - Q) * T)
    for d in (0.25, 0.5, -0.1, -0.4):
        K = float(strike_from_delta(d, F, T, vol, Q))
        cp = 1 if d > 0 else -1
        assert float(bs_delta(S, K, T, vol, R, Q, cp)) == pytest.approx(d, abs=1e-12)
