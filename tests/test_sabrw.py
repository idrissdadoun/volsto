"""SABRW, the desk's in-house SABR smile (``volsto/market/sabrw.py``; SPEC §15 Part 3, *Step 0 on
(e)SSVI surfaces and the desk's SABRW*): the central zone is plain SABR, the zone sums are the
integral of the equivalent local vol, the smile is continuous and C¹ at the zone edges, the ATM
triplet inverts to the fitted ``(ρ, ν)``, the fit recovers known parameters and flags what it
cannot identify.  No calibration, no Monte Carlo."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm

from volsto.market.sabrw import (
    MIN_ZONE_QUOTES,
    SabrwParams,
    SabrwZones,
    atm_triplet,
    breakeven_from_triplet,
    fit_sabrw,
    h,
    local_vol_integral,
    sabr_bbf_vol,
    sabr_smile_minimum,
    sabrw_vol,
    zone_counts,
    zones,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "hdn_sample" / "options_sample_2022H2"
P = SabrwParams(0.22, -0.8, 0.65, t_d=0.7, t_u=1.4, ex_d=0.4, ex_u=0.9)
T = 1.0


def test_central_zone_is_plain_sabr_and_the_zone_edges() -> None:
    """Inside ``(x_Td, x_Tu)`` the SABRW smile is the plain short-maturity SABR smile exactly;
    the edges are the documents' rules: ``x_Td`` at a 15-delta put with a 30 % reference vol,
    ``x_Tu`` the plain SABR smile's minimum, ``x_EXd = 4 x_Td``, ``x_EXu = 2 x_Tu``."""
    z, interior = zones(P, T)
    assert interior
    k = np.linspace(z.x_td, z.x_tu, 201)
    np.testing.assert_allclose(
        sabrw_vol(k, P, T), sabr_bbf_vol(k, P.sigma, P.rho, P.nu), rtol=0.0, atol=1e-15
    )
    assert z.x_td == pytest.approx(norm.ppf(0.15) * 0.30 * np.sqrt(T), rel=1e-12)
    assert z.x_exd == pytest.approx(4.0 * z.x_td) and z.x_exu == pytest.approx(2.0 * z.x_tu)
    xs = np.linspace(1e-4, 1.5, 150_001)
    brute = float(xs[np.argmin(sabr_bbf_vol(xs, P.sigma, P.rho, P.nu))])
    assert z.x_tu == pytest.approx(brute, abs=2e-5)
    # a smile rising from the money has no interior minimum: the edge sits at its floor, flagged
    x_flat, ok = sabr_smile_minimum(0.2, 0.3, 0.5, T)
    assert not ok and x_flat > 0.0


def test_zone_sums_are_the_local_vol_integral() -> None:
    """``I(k)``, summed over the zones between 0 and ``k``, equals the adaptive quadrature of
    ``1/g`` with ``g(x) = √(σ² + 2ρνσ h(x) + ν² h(x)²)`` in every zone, both signs of ``k``."""
    z, _ = zones(P, T)
    edges = [z.x_exd, z.x_td, z.x_tu, z.x_exu]

    def g(x: float) -> float:
        hx = float(h(np.array([x]), P, z)[0])
        return float(np.sqrt(P.sigma**2 + 2 * P.rho * P.nu * P.sigma * hx + P.nu**2 * hx**2))

    for kq in (-2.0, -1.3, -0.6, -0.2, -0.05, 0.05, 0.2, 0.5, 0.9, 1.6):
        pts = [e for e in edges if min(0.0, kq) < e < max(0.0, kq)]
        ref, _ = quad(lambda x: 1.0 / g(x), 0.0, kq, points=pts or None, epsabs=1e-14, limit=200)
        got = float(local_vol_integral(np.array([kq]), P, z)[0])
        assert got == pytest.approx(ref, rel=1e-10, abs=1e-13), kq


def test_continuous_and_c1_at_the_zone_edges() -> None:
    """``h`` is continuous, so the smile is continuous and C¹ at every zone edge (only the second
    derivative jumps, with ``h′``)."""
    z, _ = zones(P, T)
    for edge in (z.x_exd, z.x_td, z.x_tu, z.x_exu):
        e = 1e-9
        hl, hr = h(np.array([edge - e, edge + e]), P, z)
        assert abs(hr - hl) < 1e-8
        vl, v0, vr = sabrw_vol(np.array([edge - e, edge, edge + e]), P, T)
        assert abs(vr - vl) < 1e-8 and abs(v0 - vl) < 1e-8
        d = 1e-5
        v = sabrw_vol(np.array([edge - d, edge, edge + d]), P, T)
        left, right = (v[1] - v[0]) / d, (v[2] - v[1]) / d
        assert abs(right - left) < 1e-3, (edge, left, right)


def test_atm_triplet_inverts_to_the_parameters() -> None:
    """The ATM level, skew and curvature are the central zone's SABR Taylor terms (checked by
    finite differences of the smile), and the step-0 reduction ``ν = √(6 skw² + 3 atf cvx)``,
    ``ρ = 2 skw/ν`` returns the parameters exactly; a triplet outside SABR's reach raises."""
    atf, skw, cvx = atm_triplet(P)
    # the smile is k / (difference of two arcsinh values): near the money that difference carries
    # a rounding error of ~1e-13 relative, so the second difference needs a step of 1e-3 (at 1e-4
    # rounding alone moves it by ~4e-4 relative); the triplet itself is analytic
    hs, hc = 1e-4, 1e-3
    v = sabrw_vol(np.array([-hs, 0.0, hs]), P, T)
    w = sabrw_vol(np.array([-hc, 0.0, hc]), P, T)
    assert atf == P.sigma == v[1]
    assert (v[2] - v[0]) / (2 * hs) == pytest.approx(skw, rel=1e-6)
    assert (w[2] - 2 * w[1] + w[0]) / hc**2 == pytest.approx(cvx, rel=1e-4)
    nu, rho = breakeven_from_triplet(atf, skw, cvx)
    assert nu == pytest.approx(P.nu, rel=1e-12) and rho == pytest.approx(P.rho, rel=1e-12)
    with pytest.raises(ValueError, match="SABR reading"):
        breakeven_from_triplet(0.2, -0.3, -1.0)


def test_fit_recovers_known_parameters() -> None:
    """Noise-free quotes spanning every zone give back the seven parameters; nothing is held,
    at a bound or flagged."""
    z, _ = zones(P, T)
    k = np.linspace(-1.8, 1.5, 80)
    assert min(zone_counts(k, z).values()) >= MIN_ZONE_QUOTES
    fit = fit_sabrw(
        k, sabrw_vol(k, P, T), np.full(k.size, 1e-3), T, init=SabrwParams(0.2, -0.6, 1.0)
    )
    np.testing.assert_allclose(fit.params.as_array(), P.as_array(), rtol=1e-5, atol=1e-6)
    assert fit.rms_vp < 1e-5 and fit.held == () and fit.at_bound == () and fit.flags == ()
    assert fit.zones == zones(fit.params, T)[0]


def test_fit_holds_what_it_cannot_identify() -> None:
    """Quotes that never reach the extreme zones leave ``ex_d`` and ``ex_u`` held at their initial
    value and flagged; the central parameters are still recovered."""
    k = np.linspace(-0.6, 0.6, 50)
    fit = fit_sabrw(
        k, sabrw_vol(k, P, T), np.full(k.size, 1e-3), T, init=SabrwParams(0.2, -0.6, 1.0)
    )
    assert set(fit.held) >= {"ex_d"} and fit.params.ex_d == 1.0
    assert any("held" in f for f in fit.flags)
    assert fit.params.rho == pytest.approx(P.rho, abs=1e-4)
    assert fit.params.nu == pytest.approx(P.nu, rel=1e-3)


def test_validation() -> None:
    for bad in (
        dict(sigma=0.0, rho=-0.5, nu=0.5),
        dict(sigma=0.2, rho=1.0, nu=0.5),
        dict(sigma=0.2, rho=-0.5, nu=0.0),
        dict(sigma=0.2, rho=-0.5, nu=0.5, t_u=0.0),
    ):
        with pytest.raises(ValueError):
            SabrwParams(**bad)
    with pytest.raises(ValueError, match="x_exd < x_td"):
        SabrwZones(-0.1, -0.2, 0.3, 0.6)
    k = np.linspace(-0.3, 0.3, 5)
    with pytest.raises(ValueError, match="at least 7"):
        fit_sabrw(k, np.full(5, 0.2), np.ones(5), T)
    k = np.linspace(-0.3, 0.3, 9)
    with pytest.raises(ValueError, match="positive"):
        fit_sabrw(k, np.full(9, 0.2), np.zeros(9), T)


@pytest.mark.skipif(not SAMPLE.exists(), reason="HDN sample not present")
def test_fit_on_the_2022_12_30_one_year_slice() -> None:
    """The SPX 1y expiry of 2022-12-30 (importer quotes, weights the half bid–ask spread in vol,
    floored at 0.05 vp): the fitted correlation is inside (−1, 1) and not at a bound — where the
    eSSVI's ATM derivatives read −1.048 (SPEC §15 Part 3) — and the fit error is a few tenths of
    a vol point (measured −0.827, 0.23 vp)."""
    from volsto.market import import_hdn as ih

    f = ih.HdnFilters()
    ch = ih.load_day(
        SAMPLE / "day_by_date" / "2022-12-30_options.csv", "SPX", manifest=ih.load_manifest(SAMPLE)
    )
    _, pts = ih.to_grid_surface(
        ch, ih.implied_forwards(ch, max_years=f.max_years, band=f.near_atm_band), f
    )
    tbl = pts.table
    t1 = min(tbl["T"].unique(), key=lambda t: abs(t - 1.0))
    g = tbl[tbl["T"] == t1]
    w = np.maximum(
        0.5 * (g["iv_ask_used"].to_numpy(float) - g["iv_bid_used"].to_numpy(float)), 5e-4
    )
    fit = fit_sabrw(g["k"].to_numpy(float), g["iv_mid"].to_numpy(float), w, float(t1))
    print(f"\n1y SABRW: {fit.params}, rms {fit.rms_vp:.3f} vp, flags {fit.flags}")
    assert -0.9 < fit.params.rho < -0.7 and "rho" not in fit.at_bound and "nu" not in fit.at_bound
    assert fit.rms_vp < 0.5
    nu, rho = breakeven_from_triplet(*atm_triplet(fit.params))
    assert rho == pytest.approx(fit.params.rho, rel=1e-12) and nu == pytest.approx(
        fit.params.nu, rel=1e-12
    )
