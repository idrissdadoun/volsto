"""The desk's most-likely-path break-evens (``volsto/analytics/p1_mlp.py``; SPEC §15 Part 3, *The
note's engine*): the flat-smile limit is volsto's first-order sensitivity, the analytic SensiX /
SensiY (the note's eq. 57 and appendix D) are the derivatives of the closed form eq. 33, the
first-order pieces on volsto's kernel reproduce volsto's engine, and the quadrature has
converged.  No calibration, no Monte Carlo."""

from __future__ import annotations

import importlib
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from volsto.analytics.p1_mlp import MlpGrid, mlp_breakevens, mlp_setup
from volsto.analytics.reparam import BreakEvenParams, to_breakeven
from volsto.config import BergomiParams
from volsto.market.loaders import load_ssvi_surface

ROOT = Path(__file__).resolve().parents[1]
SPX = ROOT / "configs" / "surfaces" / "snapshots" / "hdn_2022H2" / "spx_2022-12-30.yaml"
#: today's marking fit on the SPX anchor (2022-12-30, SSR 1, eps 0.10; outputs/step0_stage3)
SPX_FIT = BergomiParams(
    nu=2.109, theta=0.120, k1=8.55, k2=0.2, rho12=0.650, rho_SX1=-0.873, rho_SX2=-0.937
)


class _Flat:
    """A flat smile at ``σ``: ``w = σ² t``, no skew."""

    def __init__(self, sigma: float) -> None:
        self.s = sigma

    def total_variance(self, k: Any, t: Any) -> Any:
        return self.s**2 * np.asarray(t, dtype=float) + 0.0 * np.asarray(k, dtype=float)

    def dw_dT(self, k: Any, t: Any) -> Any:
        return self.s**2 + 0.0 * np.asarray(k, dtype=float) * np.asarray(t, dtype=float)

    def atm_skew(self, t: Any) -> Any:
        return 0.0 * np.asarray(t, dtype=float)


class _Kernel:
    """The anchor with the variance-swap forward variance in place of the ATMF one where the
    closed forms read ``ξ̂`` (volsto's naked kernel), the smile unchanged."""

    def __init__(self, surface: Any, curve: Any) -> None:
        self.s, self.c, self.pillars = surface, curve, surface.pillars

    def dw_dT(self, k: Any, t: Any) -> Any:
        return np.asarray(self.c.xi0(np.asarray(t, dtype=float)), dtype=float) + 0.0 * np.asarray(
            k, dtype=float
        )

    def total_variance(self, k: Any, t: Any) -> Any:
        return self.s.total_variance(k, t)

    def atm_skew(self, t: Any) -> Any:
        return self.s.atm_skew(t)


def test_flat_smile_without_spot_correlation_is_the_first_order_sensitivity() -> None:
    """On a flat smile with ``λ = 0`` the closed form returns the market variance and SensiX /
    SensiY are ``½ ω_i A_i``, ``A_i = (1 − e^{−k_i T})/(k_i T)`` (book eq. 7.38): the note's
    extra terms vanish (``c = 0``, ``E[(u² − 1) σ²] = 0``); with ``ω = 0`` they are 0."""
    p = BreakEvenParams(8.0, 0.3, 4.0, 1.0, 0.0, 0.0, 0.3)
    for T in (0.25, 1.0, 3.0):
        s = mlp_setup(_Flat(0.2), p, T)
        v = s.implied_variance()
        assert v == pytest.approx(0.04, rel=1e-12)
        a1 = (1 - math.exp(-8.0 * T)) / (8.0 * T)
        a2 = (1 - math.exp(-0.3 * T)) / (0.3 * T)
        assert s.d_implied_variance("X") / (2 * v) == pytest.approx(0.5 * 4.0 * a1, rel=1e-9)
        assert s.d_implied_variance("Y") / (2 * v) == pytest.approx(0.5 * 1.0 * a2, rel=1e-9)
    zero = BreakEvenParams(8.0, 0.3, 0.0, 1.0, 0.0, 0.0, 0.3)
    assert mlp_setup(_Flat(0.2), zero, 1.0).d_implied_variance("X") == 0.0


def test_analytic_sensitivities_are_the_derivatives_of_eq_33() -> None:
    """On the SPX anchor at today's fitted parameters, eq. 57 (SensiX) and its appendix-D
    counterpart (SensiY) — with the exact derivatives of ``η``, ``μ``, ``γ`` — equal the central
    difference of the closed form eq. 33 in ``X0`` and ``Y0`` to 1e-6 relative; the closed form's
    ATM vol is within 1 % of the market's."""
    surf = load_ssvi_surface(SPX)
    p = to_breakeven(SPX_FIT)
    h = 1e-4
    for T in (0.25, 1.0):
        s = mlp_setup(surf, p, T)
        fd_x = (s.implied_variance(h, 0.0) - s.implied_variance(-h, 0.0)) / (2 * h)
        fd_y = (s.implied_variance(0.0, h) - s.implied_variance(0.0, -h)) / (2 * h)
        assert s.d_implied_variance("X") == pytest.approx(fd_x, rel=1e-6)
        assert s.d_implied_variance("Y") == pytest.approx(fd_y, rel=1e-6)
        assert math.sqrt(s.implied_variance()) == pytest.approx(math.sqrt(s.Q0T / T), rel=1e-2)
    with pytest.raises(ValueError, match="factor"):
        s.d_implied_variance("Z")


def test_first_order_pieces_reproduce_volsto_engine() -> None:
    """The closed forms' first-order pieces evaluated on volsto's naked kernel (the variance-swap
    forward variance) reproduce volsto's first-order engine on the anchor
    (:class:`volsto.calibration.fit_2f.P1Maps`): ``A_i = SensiX_i^{(1)}/(½ ω_i)`` and the naked
    skew ``λ·J = (1/(2 σ̂³ T²)) ∫₀ᵀ c ξ̂`` at every fitted pillar to 1e-4 relative, volsto's
    quadrature raised to 512 × 256 nodes (at its default 64 × 32 its Gauss–Legendre rule on
    ``[0, T]`` runs across the kinks of the forward variance: 2.4e-3 at 3Y)."""
    f2 = importlib.import_module("volsto.calibration.fit_2f")
    from volsto.market.varswap import xi0_curve

    surf = load_ssvi_surface(SPX)
    p = to_breakeven(SPX_FIT)
    cfg = f2.BreakEvenFitConfig(skew_eps=0.10, n_quad=512, n_inner=256)
    targets = f2.marking_targets_for(surf, cfg, ssr_target=1.0)
    xi0 = xi0_curve(surf, float(min(surf.max_maturity, max(targets.pillars))))
    prob, _ = f2._first_problem(targets, cfg, xi0)
    mm = prob.maps(p.k1)
    lam = np.array([p.lambda1, p.lambda2])
    for i, T in enumerate(prob.T):
        s = mlp_setup(_Kernel(surf, xi0), p, float(T))
        assert s.leading_sensi("X") / (0.5 * p.omega1) == pytest.approx(mm.naked.A[i, 0], rel=1e-4)
        assert s.leading_sensi("Y") / (0.5 * p.omega2) == pytest.approx(mm.naked.A[i, 1], rel=1e-4)
        cxi = float(np.sum(np.diff(s.s) * 0.5 * (s.c0[1:] + s.c0[:-1]) * s.xi_mid))
        lam_j = cxi / (2.0 * (s.Q0T / T) ** 1.5 * T**2)
        assert lam_j == pytest.approx(float(mm.naked.J[i] @ lam), rel=1e-4)


def test_quadrature_has_converged_and_validation() -> None:
    """Doubling every grid moves the sensitivities and the break-evens by less than 1e-5
    relative on the anchor (measured 1.4e-6 on SensiSpot, the slowest); bad inputs raise."""
    surf = load_ssvi_surface(SPX)
    p = to_breakeven(SPX_FIT)
    base = mlp_breakevens(surf, p, 1.0, sigma_0=0.2)
    fine = mlp_breakevens(surf, p, 1.0, sigma_0=0.2, grid=MlpGrid(8000, 96, 96))
    for a, b in (
        (base.sensi_x, fine.sensi_x),
        (base.sensi_y, fine.sensi_y),
        (base.sensi_spot, fine.sensi_spot),
        (base.vol_var, fine.vol_var),
    ):
        assert a == pytest.approx(b, rel=1e-5)
    svc, vv = base.absolute()
    assert svc == pytest.approx(base.atmf_vol * base.spot_vol_covar)
    assert vv == pytest.approx(base.atmf_vol**2 * base.vol_var)
    with pytest.raises(ValueError, match="positive"):
        mlp_setup(surf, p, 0.0)
    with pytest.raises(ValueError, match="MlpGrid"):
        MlpGrid(5, 48, 48)


def test_volsto_engine_on_atmf_kernels_is_the_notes_first_order() -> None:
    """volsto's first-order engine with ``kernel_curve="atmf"`` reads the note's ATMF forward
    variance: its ``A`` and ``λ·J`` equal the closed forms' first-order pieces on the anchor
    to 5e-4 (volsto's quadrature raised to 512 × 256 nodes; its ATMF curve interpolates the
    surface's ATMF variance on a weekly grid, which rounds the kinks at the pillars: 1.1e-4 at
    3M measured)."""
    f2 = importlib.import_module("volsto.calibration.fit_2f")

    surf = load_ssvi_surface(SPX)
    p = to_breakeven(SPX_FIT)
    cfg = f2.BreakEvenFitConfig(skew_eps=0.10, n_quad=512, n_inner=256, kernel_curve="atmf")
    targets = f2.marking_targets_for(surf, cfg, ssr_target=1.0)
    curve = f2._engine_curve(targets, None, cfg)
    prob, _ = f2._first_problem(targets, cfg, curve)
    mm = prob.maps(p.k1)
    lam = np.array([p.lambda1, p.lambda2])
    for i, T in enumerate(prob.T):
        s = mlp_setup(surf, p, float(T))
        assert s.leading_sensi("X") / (0.5 * p.omega1) == pytest.approx(mm.naked.A[i, 0], rel=5e-4)
        assert s.leading_sensi("Y") / (0.5 * p.omega2) == pytest.approx(mm.naked.A[i, 1], rel=5e-4)
        cxi = float(np.sum(np.diff(s.s) * 0.5 * (s.c0[1:] + s.c0[:-1]) * s.xi_mid))
        lam_j = cxi / (2.0 * (s.Q0T / T) ** 1.5 * T**2)
        assert lam_j == pytest.approx(float(mm.naked.J[i] @ lam), rel=5e-4)


def test_mlp_pillar_is_the_engine() -> None:
    """:class:`MlpPillar` (the fit's per-pillar evaluator, everything but ``λ`` precomputed)
    returns the engine's SensiSpot and ``SensiX/ω1``, ``SensiY/ω2`` exactly on the same grid, and
    to 1e-4 on the fit's lighter grid."""
    from volsto.analytics.p1_mlp import MlpPillar
    from volsto.calibration.fit_2f import MLP_FIT_GRID

    surf = load_ssvi_surface(SPX)
    p = to_breakeven(SPX_FIT)
    lam = np.array([p.lambda1, p.lambda2])
    for T in (0.25, 1.0, 3.0):
        be = mlp_breakevens(surf, p, T, sigma_0=0.2)
        for grid, tol in ((None, 1e-12), (MLP_FIT_GRID, 1e-4)):
            spot, gx, gy = MlpPillar(surf, T, p.k1, p.k2, sigma_0=0.2, grid=grid).evaluate(lam)
            assert spot == pytest.approx(be.sensi_spot, rel=tol)
            assert p.omega1 * gx == pytest.approx(be.sensi_x, rel=tol)
            assert p.omega2 * gy == pytest.approx(be.sensi_y, rel=tol)
