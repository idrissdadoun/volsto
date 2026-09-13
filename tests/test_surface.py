"""SPEC §10: SSVI no-arbitrage checks, Dupire positivity, local-vol MC reprices SSVI pillars."""

from __future__ import annotations

import numpy as np
import pytest

from volsto.config import LocalVolConfig, SimConfig
from volsto.engine import MonteCarlo
from volsto.market import (
    DiscountCurve,
    ForwardCurve,
    GridSurface,
    LocalVolSurface,
    SSVISurface,
    black_vega,
    implied_vol,
)
from volsto.models import BlackScholes, LocalVol
from volsto.products import EuropeanOption


def test_ssvi_atm_and_symmetry(forward_curve: ForwardCurve, discount: DiscountCurve) -> None:
    T = np.array([0.1, 0.5, 1.0, 4.0])
    s = SSVISurface([0.5, 1.0, 2.0], [0.02, 0.045, 0.1], -0.6, 0.8, 0.5, forward_curve, discount)
    np.testing.assert_allclose(s.total_variance(0.0, T), s.theta(T))
    np.testing.assert_allclose(s.theta([0.5, 1.0, 2.0]), [0.02, 0.045, 0.1])
    assert s.theta(4.0) == pytest.approx(0.1 + 2.0 * 0.055)  # last forward variance extended
    sym = SSVISurface.flat_atm(0.2, 0.0, 1.0, 0.5, forward_curve, discount)
    k = np.linspace(-1, 1, 11)
    np.testing.assert_allclose(sym.total_variance(k, 1.0), sym.total_variance(-k, 1.0))
    flat = SSVISurface.flat_atm(0.2, 0.0, 0.0, 0.5, forward_curve, discount)
    np.testing.assert_allclose(flat.implied_vol_k(k, 2.0), 0.2)


def test_ssvi_atm_skew_matches_finite_difference(ssvi: SSVISurface) -> None:
    h = 1e-6
    for T in (0.25, 1.0, 3.0):
        fd = (ssvi.implied_vol_k(h, T) - ssvi.implied_vol_k(-h, T)) / (2 * h)
        assert float(ssvi.atm_skew(T)) == pytest.approx(float(fd), rel=1e-6)
        assert float(ssvi.atm_skew(T)) < 0  # negative skew for rho < 0


def test_ssvi_no_arbitrage_checks(forward_curve: ForwardCurve, discount: DiscountCurve) -> None:
    # butterfly: theta phi^2 (1+|rho|) = eta^2 (1+|rho|)/(1+theta) > 4 near theta -> 0 for gamma = 1/2
    with pytest.raises(ValueError, match="butterfly"):
        SSVISurface.flat_atm(0.2, -0.5, 2.0, 0.5, forward_curve, discount)
    # butterfly: theta phi (1+|rho|) >= 4 with gamma = 1 (phi = eta/(1+theta)): eta = 30, rho = 0.9
    with pytest.raises(ValueError, match="butterfly"):
        SSVISurface.flat_atm(0.2, 0.9, 30.0, 1.0, forward_curve, discount)
    # calendar: decreasing ATM total variance
    with pytest.raises(ValueError, match="calendar"):
        SSVISurface([0.5, 1.0], [0.03, 0.02], -0.5, 1.0, 0.5, forward_curve, discount)
    # bad correlation
    with pytest.raises(ValueError):
        SSVISurface([1.0], [0.04], -1.0, 1.0, 0.5, forward_curve, discount)


def test_grid_surface_interpolation(forward_curve: ForwardCurve, discount: DiscountCurve) -> None:
    mats = [0.5, 1.0, 2.0]
    F = [float(forward_curve.forward(T)) for T in mats]
    strikes = [np.array([0.8, 0.9, 1.0, 1.1, 1.2]) * f for f in F]
    vols = [np.array([0.26, 0.23, 0.20, 0.185, 0.18]) + 0.005 * i for i in range(3)]
    g = GridSurface.from_strike_vols(mats, strikes, vols, forward_curve, discount)
    # nodes reproduced
    for T, K, v in zip(mats, strikes, vols):
        np.testing.assert_allclose(g.implied_vol(K, T), v, rtol=1e-12)
    # linear in w along T at fixed k between slices
    k = 0.05
    w_mid = g.total_variance(k, 1.5)
    assert w_mid == pytest.approx(0.5 * (g.total_variance(k, 1.0) + g.total_variance(k, 2.0)))
    # flat in implied vol beyond the wings, w linear from 0 before the first slice
    assert g.implied_vol_k(2.0, 1.0) == pytest.approx(g.implied_vol_k(np.log(1.2), 1.0))
    assert g.total_variance(0.0, 0.25) == pytest.approx(0.5 * g.total_variance(0.0, 0.5))
    assert g.implied_vol_k(0.0, 4.0) == pytest.approx(g.implied_vol_k(0.0, 2.0))
    # calendar violation raises
    with pytest.raises(ValueError, match="calendar"):
        GridSurface.from_strike_vols(
            mats, strikes, [vols[0], vols[0] * 0.5, vols[2]], forward_curve, discount
        )
    # delta quotes give the same surface as the equivalent strike quotes
    from volsto.market import bs_delta

    deltas = [
        bs_delta(forward_curve.spot, K, T, v, 0.02, 0.01, np.where(f < K, 1, -1))
        for T, K, v, f in zip(mats, strikes, vols, F)
    ]
    gd = GridSurface.from_delta_vols(mats, deltas, vols, forward_curve, discount)
    for T, K, v in zip(mats, strikes, vols):
        np.testing.assert_allclose(gd.implied_vol(K, T), v, rtol=1e-8)


def test_dupire_local_vol_positive(local_vol: LocalVolSurface, ssvi: SSVISurface) -> None:
    diag = local_vol.check_positive()
    assert diag.ok, diag
    assert diag.min_denominator > 0 and diag.min_dw_dt > 0
    assert diag.n_floored == 0 and diag.n_capped == 0
    # flat surface -> flat local vol
    flat = SSVISurface.flat_atm(0.2, 0.0, 0.0, 0.5, ssvi.forward_curve, ssvi.discount)
    lv = LocalVolSurface.from_implied(flat, LocalVolConfig(t_max=2.0, n_t=20, n_k=41))
    np.testing.assert_allclose(lv.local_var, 0.04, rtol=1e-6)
    # ATM short-dated local vol ~ implied vol
    assert float(local_vol.local_vol_k(1 / 365, 0.0)) == pytest.approx(
        float(ssvi.atm_vol(1 / 365)), rel=0.02
    )


def test_local_vol_flat_equals_black_scholes(
    forward_curve: ForwardCurve, discount: DiscountCurve, fast_sim: SimConfig
) -> None:
    mc = MonteCarlo(fast_sim)
    bs = BlackScholes(0.2, forward_curve)
    lv = LocalVol(LocalVolSurface.flat(0.2, forward_curve), forward_curve)
    opt = EuropeanOption(105.0, 1.0, "call", discount)
    a = mc.price(opt, bs)
    b = mc.price(opt, lv)
    assert a.mean == pytest.approx(b.mean, abs=1e-12)


def _reprice_pillars(
    ssvi: SSVISurface, local_vol: LocalVolSurface, cfg: SimConfig, T: float, ks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mc = MonteCarlo(cfg)
    F = float(ssvi.forward(T))
    df = float(ssvi.discount.df(T))
    cps = np.where(ks >= 0, 1, -1)  # OTM options
    strikes = F * np.exp(ks)
    prods = [EuropeanOption(K, T, int(cp), ssvi.discount) for K, cp in zip(strikes, cps)]
    res = mc.price_many(prods, LocalVol(local_vol))
    prices = np.array([r.mean for r in res])
    errs = np.array([r.stderr for r in res])
    iv = implied_vol(prices, F, strikes, T, cps, df)
    iv_err = errs / black_vega(F, strikes, T, iv, df)
    return iv - ssvi.implied_vol_k(ks, T), iv_err


# Default scheme (Platen weak order 2) and default step schedule: pure local vol reprices the
# SSVI within noise at all maturities (see tests/test_scheme.py for the scheme comparison).
@pytest.mark.parametrize("T", [0.25, 1.0])
def test_local_vol_mc_reprices_ssvi(
    ssvi: SSVISurface, local_vol: LocalVolSurface, T: float
) -> None:
    ks = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    cfg = SimConfig(n_paths=100_000, chunk_size=50_000, seed=11)
    diff, err = _reprice_pillars(ssvi, local_vol, cfg, T=T, ks=ks)
    assert np.all(np.abs(diff) < np.maximum(0.001, 3 * err)), (diff * 100, err * 100)


@pytest.mark.slow
@pytest.mark.parametrize("T", [1 / 12, 0.25, 0.5, 1.0, 1.5, 2.0])
def test_local_vol_mc_reprices_ssvi_full(
    ssvi: SSVISurface, local_vol: LocalVolSurface, T: float
) -> None:
    """§4.2 accuracy for pure local vol: ≤ 0.1 vol points inside ±20% moneyness, 1m to 2y."""
    ks = np.log(np.array([0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]))
    cfg = SimConfig(n_paths=400_000, chunk_size=50_000, seed=12)
    diff, err = _reprice_pillars(ssvi, local_vol, cfg, T=T, ks=ks)
    assert np.all(np.abs(diff) < np.maximum(0.001, 3 * err)), (diff * 100, err * 100)
