"""SPEC §10: SSVI no-arbitrage checks, Dupire positivity, local-vol MC reprices SSVI pillars."""

from __future__ import annotations

import re
from itertools import pairwise

import numpy as np
import pytest
from numpy.typing import NDArray

from volsto.config import LocalVolConfig, SimConfig
from volsto.engine import MonteCarlo
from volsto.market import (
    DiscountCurve,
    ESSVISurface,
    ForwardCurve,
    GridSurface,
    LocalVolSurface,
    SSVISurface,
    black_vega,
    implied_vol,
)
from volsto.market.surface import certify_calendar, dw_dt_lower_bound, essvi_segments
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


# --------------------------------------------------------------------------------------------
# eSSVI calendar: exact ∂_T w, constructor grid, certificate, dense check (M10 Part 0)
# --------------------------------------------------------------------------------------------

_PILLARS_7 = np.array([1 / 12, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0])
_THETA_7 = np.array([0.30, 0.29, 0.28, 0.27, 0.265, 0.26, 0.255]) ** 2 * _PILLARS_7


class _UncheckedESSVI(ESSVISurface):
    """Constructor calendar check skipped, so a violating surface can be inspected."""

    def _check_calendar_numeric(self) -> None:
        return None


def _m3b_min_dw(s: ESSVISurface) -> float:
    """The constructor's slack on its pre-fix grid (``linspace ∪ {max_maturity}``, no pillars)."""
    ks = np.linspace(-1.0, 1.0, 81)
    Ts = np.unique(np.concatenate((np.linspace(s.min_maturity, s.pillars[-1], 200), [3.0])))
    return float(np.min(np.diff(s.total_variance(ks[None, :], Ts[:, None]), axis=0)))


def _brute_min_dw_dt(s: ESSVISurface, ks: NDArray[np.float64], Ts: NDArray[np.float64]) -> float:
    """Exact ``∂_T w`` on ``ks × Ts`` through :meth:`ESSVISurface.dw_dT`, both limits at every
    maturity (they coincide away from the knots)."""
    left = s.dw_dT(ks[None, :], Ts[:, None], "left")
    right = s.dw_dT(ks[None, :], Ts[:, None], "right")
    return float(min(np.min(left[1:]), np.min(right[:-1])))


def test_essvi_calendar_grid_dedup(forward_curve: ForwardCurve, discount: DiscountCurve) -> None:
    """Seven pillars ending at 3y with max_maturity 3y: the M3b grid listed T = 3 twice (a row
    with Δw ≡ 0); the grid now has no zero-length step, contains every pillar, and the
    constructor's quantity is the exact ``∂_T w`` minimum over exactly that grid (both limits at
    the knots; the left limit at ``min_maturity`` and the right one at ``max_maturity`` lie
    outside the range)."""
    s = ESSVISurface(
        _PILLARS_7, _THETA_7, [-0.7] * 7, 0.8, 0.4, forward_curve, discount, max_maturity=3.0
    )
    ks, Ts = s.calendar_grid()
    assert ks.size == ESSVISurface.CALENDAR_N_K and ks[-1] == ESSVISurface.CALENDAR_K_ABS
    assert np.all(np.diff(Ts) > 0) and Ts[0] == s.min_maturity and Ts[-1] == 3.0
    lin = np.linspace(s.min_maturity, 3.0, ESSVISurface.CALENDAR_N_T)
    np.testing.assert_array_equal(Ts, np.union1d(lin, _PILLARS_7))
    old = np.concatenate((lin, [3.0]))
    dw_old = np.diff(s.total_variance(ks[None, :], old[:, None]), axis=0)
    assert np.all(dw_old[-1] == 0.0)  # the zero-length row of the M3b grid
    dw = np.diff(s.total_variance(ks[None, :], Ts[:, None]), axis=0)
    assert s.calendar_min_dw() == float(np.min(dw))
    assert s.calendar_min_dw_dt() == pytest.approx(_brute_min_dw_dt(s, ks, Ts), rel=1e-12)
    # max_maturity beyond the last pillar: it is appended, every pillar is present
    s5 = ESSVISurface(
        _PILLARS_7[:6],
        _THETA_7[:6],
        [-0.7] * 6,
        0.8,
        0.4,
        forward_curve,
        discount,
        max_maturity=3.0,
    )
    Ts5 = s5.calendar_grid()[1]
    lin5 = np.linspace(s5.min_maturity, 2.0, ESSVISurface.CALENDAR_N_T)
    np.testing.assert_array_equal(Ts5, np.union1d(np.union1d(lin5, _PILLARS_7[:6]), [3.0]))


def test_essvi_calendar_grid_sees_pillars(
    forward_curve: ForwardCurve, discount: DiscountCurve
) -> None:
    """ρ_T falling from -0.5 (6m) to -0.682 (1y) and back to -0.5 (18m): ``w`` decreases in
    ``T`` only just below the 1y kink.  The M3b grid (no pillars) straddles the kink and passes
    (+2.7e-6); the constructor's exact derivative sees the negative left limit at 1y and
    refuses the surface, and the dense check and the certificate agree."""
    rhos = [-0.5, -0.5, -0.5, -0.682, -0.5, -0.5]
    args = (_PILLARS_7[:6], _THETA_7[:6], rhos, 0.5, 0.4, forward_curve, discount)
    u = _UncheckedESSVI(*args, max_maturity=3.0)
    assert _m3b_min_dw(u) >= 0.0
    assert u.calendar_min_dw_dt() < -ESSVISurface.CALENDAR_TOL
    with pytest.raises(ValueError, match="calendar"):
        ESSVISurface(*args, max_maturity=3.0)
    assert float(np.min(u.dw_dT(np.linspace(0.0, 1.0, 81), 1.0, "left"))) < 0.0
    assert float(np.min(u.dw_dT(np.linspace(-1.0, 1.0, 81), 1.0, "right"))) > 0.0
    dense = u.calendar_dense_check(1.0, 401)
    assert not dense.ok and dense.min_left < 0.0 and dense.min_right > 0.0
    assert dense.worst_T == 1.0 and dense.worst_k > 0.0
    assert dense.certificate.status == "violated"


def test_essvi_dw_dt_is_the_derivative(
    forward_curve: ForwardCurve, discount: DiscountCurve
) -> None:
    """The closed form of :func:`volsto.market.surface.essvi_dw_dt` is ``∂_T w``: central
    differences at random interior points, one-sided differences at every knot (whose two
    limits differ: ``θ_T`` and ``ρ_T`` kink there), and ``θ'(T) ∂_θ W`` for a single ρ."""
    rhos = [-0.3, -0.9, -0.6, 0.2, -0.75, -0.5]
    s = _UncheckedESSVI(
        _PILLARS_7[:6], _THETA_7[:6], rhos, 0.7, 0.35, forward_curve, discount, max_maturity=3.0
    )
    rng = np.random.default_rng(11)
    k = rng.uniform(-3.0, 3.0, 4000)
    T = rng.uniform(s.min_maturity, 3.0, 4000)
    knots = np.concatenate(([s.min_maturity], s.pillars, [3.0]))
    T = T[np.min(np.abs(T[:, None] - knots[None, :]), axis=1) > 1e-3]
    k = k[: T.size]
    h = 1e-6
    fd = (s.total_variance(k, T + h) - s.total_variance(k, T - h)) / (2.0 * h)
    np.testing.assert_allclose(s.dw_dT(k, T), fd, rtol=1e-6, atol=1e-8)
    np.testing.assert_array_equal(s.dw_dT(k, T, "left"), s.dw_dT(k, T, "right"))
    ks = np.linspace(-3.0, 3.0, 61)[None, :]
    p = s.pillars[:, None]
    h = 1e-8
    left = (s.total_variance(ks, p) - s.total_variance(ks, p - h)) / h
    right = (s.total_variance(ks, p + h) - s.total_variance(ks, p)) / h
    np.testing.assert_allclose(s.dw_dT(ks, p, "left"), left, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(s.dw_dT(ks, p, "right"), right, rtol=1e-5, atol=1e-6)
    assert np.max(np.abs(s.dw_dT(ks, p, "left") - s.dw_dT(ks, p, "right"))) > 0.1
    with pytest.raises(ValueError, match="side"):
        s.dw_dT(0.0, 1.0, "both")


def test_essvi_calendar_dense_check_agrees_with_grid(
    forward_curve: ForwardCurve, discount: DiscountCurve
) -> None:
    """On a single-ρ surface (calendar-free) the dense check is positive and proven; its secant
    minimum is the brute-force minimum over the same grid, its exact minimum the brute-force
    exact minimum, and its one-sided limits at the pillars equal the chain rule
    ``θ'(T±) ∂_θ W(k, θ)`` (``θ_T`` piecewise linear, ``ρ`` constant)."""
    rho, eta, gamma = -0.6, 0.5, 0.4
    s = ESSVISurface(
        _PILLARS_7[:6],
        _THETA_7[:6],
        [rho] * 6,
        eta,
        gamma,
        forward_curve,
        discount,
        max_maturity=3.0,
    )
    d = s.calendar_dense_check(3.0, 121, 400)
    assert d.ok and d.min_dw_dt > 0.0 and d.certificate.certified
    assert d.certificate.lower_bound >= 0.0
    ks = np.linspace(-3.0, 3.0, 121)
    Ts = np.union1d(np.linspace(s.min_maturity, 3.0, 400), s.pillars)
    w = s.total_variance(ks[None, :], Ts[:, None])
    assert d.min_slope == pytest.approx(float(np.min(np.diff(w, axis=0) / np.diff(Ts)[:, None])))
    assert d.min_dw_dt == pytest.approx(_brute_min_dw_dt(s, ks, Ts), rel=1e-12)

    def dW_dtheta(th: float) -> NDArray[np.float64]:
        def W(x: float) -> NDArray[np.float64]:
            pk = eta / (x**gamma * (1.0 + x) ** (1.0 - gamma)) * ks
            out = 0.5 * x * (1.0 + rho * pk + np.sqrt((pk + rho) ** 2 + 1.0 - rho * rho))
            return np.asarray(out, dtype=np.float64)

        e = 1e-6 * th
        return (W(th + e) - W(th - e)) / (2.0 * e)

    t = np.concatenate(([0.0], s.pillars))
    th = np.asarray(s.theta(t), dtype=np.float64)
    seg = np.diff(th) / np.diff(t)  # θ' on each segment; beyond the last pillar: the last one
    left = min(float(np.min(seg[i] * dW_dtheta(th[i + 1]))) for i in range(s.pillars.size))
    right = min(
        float(np.min(seg[min(i + 1, seg.size - 1)] * dW_dtheta(th[i + 1])))
        for i in range(s.pillars.size)
    )
    assert d.min_left == pytest.approx(left, rel=1e-6)
    assert d.min_right == pytest.approx(right, rel=1e-6)


def test_essvi_calendar_min_dw_dt_matches_check(
    forward_curve: ForwardCurve, discount: DiscountCurve
) -> None:
    """``calendar_min_dw_dt`` is the quantity the constructor raises on: a ρ_T that collapses
    θ_T(1+ρ_T) between 1.5y and 2y gives a negative value and a refused surface; a single ρ gives
    a positive value and no raise."""
    args = (_PILLARS_7[:6], _THETA_7[:6])
    bad = [-0.5, -0.55, -0.6, -0.6, -0.3, -0.95]
    u = _UncheckedESSVI(*args, bad, 0.5, 0.4, forward_curve, discount, max_maturity=3.0)
    assert u.calendar_min_dw_dt() < -ESSVISurface.CALENDAR_TOL
    with pytest.raises(ValueError, match="calendar"):
        ESSVISurface(*args, bad, 0.5, 0.4, forward_curve, discount, max_maturity=3.0)
    ok = ESSVISurface(*args, [-0.6] * 6, 0.5, 0.4, forward_curve, discount, max_maturity=3.0)
    assert ok.calendar_min_dw_dt() > 0.0
    assert ok.calendar_min_dw_dt(3.0, 121) > 0.0


def test_dw_dt_lower_bound_is_sound() -> None:
    """The certificate's cell bound (:func:`volsto.market.surface.dw_dt_lower_bound`) never
    exceeds the exact ``∂_T w`` anywhere in the cell: 2·10⁵ random cells of random eSSVI
    surfaces (``|ρ|`` up to 0.9999, ``γ`` in (0, 1], ``η`` up to 3, cells from 1e-6 of a
    segment to a whole one, ``k`` beyond ``±3``), 64 points per cell including the corners.
    And it converges: on shrinking cells around a point the gap to the point value falls."""
    rng = np.random.default_rng(20260916)
    worst = np.inf
    n_cells = 0
    while n_cells < 200_000:
        P = int(rng.integers(2, 8))
        pil = np.sort(rng.uniform(0.02, 3.0, P))
        if np.any(np.diff(pil) < 1e-3):
            continue
        th = np.cumsum(rng.uniform(1e-4, 0.5, P) * np.diff(np.concatenate(([0.0], pil))))
        rho = rng.uniform(-0.9999, 0.9999, P)
        g, e = float(rng.uniform(0.01, 1.0)), float(rng.uniform(0.0, 3.0))
        segs = essvi_segments(pil, th, rho, 1.0 / 365.0, max(3.0, float(pil[-1])))
        m = 2000
        seg = rng.integers(0, segs.n, m).astype(np.intp)
        a, b = segs.a[seg], segs.b[seg]
        u0 = rng.uniform(0.0, 1.0, m)
        u1 = np.minimum(1.0, u0 + rng.uniform(0.0, 1.0, m) * 10 ** rng.uniform(-6, 0, m))
        T0, T1 = a + (b - a) * u0, a + (b - a) * u1
        kc, kw = rng.uniform(-4.0, 4.0, m), 10 ** rng.uniform(-6, 0.5, m)
        k0, k1 = kc - kw * rng.uniform(0.0, 1.0, m), kc + kw * rng.uniform(0.0, 1.0, m)
        lb = dw_dt_lower_bound(segs, seg, T0, T1, k0, k1, e, g)
        su, sv = rng.uniform(0.0, 1.0, (m, 64)), rng.uniform(0.0, 1.0, (m, 64))
        su[:, :4], sv[:, :4] = [0.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]
        TT = T0[:, None] + (T1 - T0)[:, None] * su
        KK = k0[:, None] + (k1 - k0)[:, None] * sv
        D = segs.dw_dt(KK, np.broadcast_to(seg[:, None], TT.shape), TT, e, g)
        gap = (D.min(axis=1) - lb) / np.maximum(1.0, np.abs(D.min(axis=1)))
        worst = min(worst, float(gap.min()))
        n_cells += m
    assert worst >= -1e-13, worst  # rounding only
    segs = essvi_segments(
        _PILLARS_7, _THETA_7, [-0.3, -0.9, -0.6, 0.2, -0.75, -0.5, -0.6], 1.0 / 365.0, 3.0
    )
    idx = np.array([2], dtype=np.intp)
    Tc, kc0 = np.array([0.4]), np.array([1.3])
    v = float(segs.dw_dt(kc0, idx, Tc, 0.7, 0.4)[0])
    gaps = [
        v - float(dw_dt_lower_bound(segs, idx, Tc - r, Tc + r, kc0 - r, kc0 + r, 0.7, 0.4)[0])
        for r in (1e-1, 1e-2, 1e-3, 1e-4)
    ]
    assert all(x >= 0.0 for x in gaps) and all(y < 0.2 * x for x, y in pairwise(gaps)), gaps


def test_certify_calendar_outcomes(forward_curve: ForwardCurve, discount: DiscountCurve) -> None:
    """``certified`` proves the floor (its lower bound is at least the floor), ``violated``
    returns only points below ``stop_below``, lowest first and at most ``max_points``, each
    really below it, and a spent budget (a floor the bound cannot reach, or one at the sampled
    minimum) is ``inconclusive`` — never a pass."""
    ok = ESSVISurface(
        _PILLARS_7[:6],
        _THETA_7[:6],
        [-0.6] * 6,
        0.5,
        0.4,
        forward_curve,
        discount,
        max_maturity=3.0,
    )
    c = ok.calendar_certificate(3.0, 0.0)
    assert c.certified and c.lower_bound >= 0.0 and c.min_value >= c.lower_bound
    too_high = certify_calendar(
        ok.calendar_segments(),
        ok.eta,
        ok.gamma,
        k_abs=3.0,
        floor=1.0,
        stop_below=0.0,
        max_cells=200_000,
    )
    assert too_high.status == "inconclusive" and not too_high.certified
    assert np.isnan(too_high.lower_bound)
    tiny = certify_calendar(
        ok.calendar_segments(),
        ok.eta,
        ok.gamma,
        k_abs=3.0,
        floor=c.min_value * (1.0 - 1e-9),
        stop_below=0.0,
        max_cells=50_000,
    )
    assert tiny.status == "inconclusive"
    bad = _UncheckedESSVI(
        _PILLARS_7[:6],
        _THETA_7[:6],
        [-0.5, -0.55, -0.6, -0.6, -0.3, -0.95],
        0.5,
        0.4,
        forward_curve,
        discount,
        max_maturity=3.0,
    )
    segs = bad.calendar_segments()
    v = certify_calendar(
        segs, bad.eta, bad.gamma, k_abs=3.0, floor=0.0, stop_below=0.0, max_points=7
    )
    assert v.status == "violated" and 1 <= v.cut_k.size <= 7
    D = segs.dw_dt(v.cut_k, v.cut_seg, v.cut_T, bad.eta, bad.gamma)
    assert np.all(D < 0.0) and np.all(np.diff(D) >= 0.0) and D[0] == v.min_value


SSVI_W_FORMULA = re.compile(
    r"0\.5\s*\*\s*\w+\s*\*\s*\(\s*1\.0\s*\+\s*\w+\s*\*\s*\w+\s*\+\s*np\.sqrt\(\s*\(\s*\w+\s*\+\s*\w+"
    r"\s*\)\s*\*\*\s*2\s*\+\s*1\.0\s*-\s*\w+\s*\*\s*\w+\s*\)"
)
"""The (e)SSVI total variance ``0.5 * th * (1.0 + r * pk + np.sqrt((pk + r) ** 2 + 1.0 - r * r))`` as it
is written in source (any names)."""


def test_ssvi_formula_has_one_implementation() -> None:
    """The (e)SSVI total variance is written once, in
    :func:`volsto.market.surface.ssvi_total_variance`, which the surfaces and the importer's fit
    call (the third occurrence of "a fit or a check evaluates a different surface than the one
    returned" — the repair's ``θ``, the flattened eSSVI of M10 Part 3, the fit's clamped ``θ``
    until 2026-09-17 — so the invariant is enforced here).  Walks ``volsto/`` and ``scripts/``;
    independent reference formulae in ``tests/`` are allowed."""
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    # the pattern sees the formula as the fit and the surfaces wrote it before 2026-09-17
    for old in (
        "w = 0.5 * th * (1.0 + r * pk + np.sqrt((pk + r) ** 2 + 1.0 - r * r))",
        "w = 0.5 * th * (1.0 + rho * pk + np.sqrt((pk + rho) ** 2 + 1.0 - rho * rho))",
    ):
        assert SSVI_W_FORMULA.search(old), old
    found: list[tuple[str, str]] = []
    files = sorted((root / "volsto").rglob("*.py")) + sorted((root / "scripts").rglob("*.py"))
    assert len(files) > 50
    for path in files:
        text = path.read_text(encoding="utf-8")
        if not SSVI_W_FORMULA.search(text):
            continue
        rel = path.relative_to(root).as_posix()
        tree = ast.parse(text, filename=rel)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                seg = ast.get_source_segment(text, node) or ""
                inner = [
                    n
                    for n in ast.walk(node)
                    if n is not node and isinstance(n, ast.FunctionDef | ast.AsyncFunctionDef)
                ]
                for n in inner:  # attribute a nested function's formula to the nested function
                    seg = seg.replace(ast.get_source_segment(text, n) or "\0", "")
                if SSVI_W_FORMULA.search(seg):
                    found.append((rel, node.name))
    assert found == [("volsto/market/surface.py", "ssvi_total_variance")], found


def test_ssvi_surfaces_are_bit_identical_to_the_written_formula(
    forward_curve: ForwardCurve, discount: DiscountCurve
) -> None:
    """Routing :class:`SSVISurface` and :class:`ESSVISurface` through the shared helpers
    (2026-09-17) changed no number: exact equality with the formula as the classes wrote it
    (``θ`` linear from 0, the last forward variance extended, ``ρ_T`` flat outside the pillars),
    inside, between and beyond the pillars."""
    pil = np.array(_PILLARS_7[:6])
    th_p = np.array(_THETA_7[:6])
    rhos = np.array([-0.8, -0.75, -0.7, -0.66, -0.62, -0.6])
    eta, gamma = 0.9, 0.45
    rng = np.random.default_rng(20260917)
    k = rng.uniform(-3.0, 3.0, 4000)
    T = np.concatenate((rng.uniform(1e-3, 6.0, 3990), pil, [0.0 + 1e-9, 10.0, 3.0, 2.0]))

    def written(rho: NDArray[np.float64] | float) -> NDArray[np.float64]:
        t = np.concatenate(([0.0], pil))
        th_all = np.concatenate(([0.0], th_p))
        slope = (th_all[-1] - th_all[-2]) / (t[-1] - t[-2])
        th = np.where(t[-1] < T, th_all[-1] + slope * (T - t[-1]), np.interp(T, t, th_all))
        ph = eta / (th**gamma * (1.0 + th) ** (1.0 - gamma))
        pk = ph * k
        return np.asarray(
            0.5 * th * (1.0 + rho * pk + np.sqrt((pk + rho) ** 2 + 1.0 - rho * rho)),
            dtype=np.float64,
        )

    s = SSVISurface(pil, th_p, -0.7, eta, gamma, forward_curve, discount, max_maturity=10.0)
    assert np.array_equal(s.total_variance(k, T), written(-0.7))
    e = ESSVISurface(pil, th_p, rhos, eta, gamma, forward_curve, discount, max_maturity=10.0)
    assert np.array_equal(e.total_variance(k, T), written(np.interp(T, pil, rhos)))
