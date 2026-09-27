"""P1 marking calibration by SABR break-evens (SPEC §15 Part 3, owner's "M7 Part 3 FINAL" and
the report decisions; ``calibration/fit_2f.py``, ``calibration/stability.py``).  Tests never
calibrate a leverage: cached leverages are read with ``allow_calibrate=False`` and skipped when
absent; no wall-clock assertion.  Every SSR asserted without a Monte Carlo standard error is a
first-order value (optimiser / formula self-consistency); the calibrated LSV's numerical SSR is
reported, never asserted equal to the target.

Fast:

* machinery: ``PillarQuad`` / affine maps against the engine to 1e-12; the exact 2-D QP against
  SLSQP and the least-violation programme; ``min ν = |λ1 + λ2|/2``; the P1 break-even identities
  (affine in ``λ``, ``R^LV(Mkt)`` of the reference SSVI, reduction to the naked form when the
  market skew is the kernel's own, the pillar-interpolated historical form); ``volvar_p1`` against
  the engine's quadratic form and the owner's half-weight cross terms;
* the owner's list: the break-even formula reproduces the policy at ``ssr_target = 1``; the
  two-point constraint binds only at 1Y / 5Y (3Y on a surface quoted to 3Y, with the note);
  ``ssr_target`` and ``skew_eps`` both exercised (scalar, pair, soft mode); the binding message
  on an incompatible ``(ssr, eps)`` pair and the infeasible message with the ν box; the ν-cap
  warning logged (cap 3.5, report decision viii); ``Corr_BE = ρ_SABR`` kept by the VolVar
  rebuild; the ρ12 collapse note (decision vii); the realised LSV SSR and the stage-3 assertion
  verdict of the cached study fits reported, not asserted equal to 1 / pass;
* historical mode: recovery on the one-year order-one history under the **default** config
  (``skew_mode="auto"`` → soft all-pillar penalty at weight 10, report decision v) and the
  two-point fit reported; the rolling fit and identification flags;
* the stage-3 assertion machinery on synthetic check tables (the owner's split on the M7 Part 3
  report: the engine bias — simulated against the first-order break-evens at the fitted
  parameters — is asserted at 10%; the target miss of a binding fit is reported in the table and
  in one sentence of the failure message, never asserted; a row without a first-order value falls
  back to the target gap and is flagged); stage 3 on the cached 2F Table 8.2 LSV: the P1
  first-order SSR within 10% of the numerical LSV SSR, ``breakeven_check`` on the Table 8.2 LSV
  against ``ssr = 1`` targets (SpotVolCovar engine bias within 10% with a +177% target miss
  reported; the VolVar rows without a first-order value fall back to the target gap and fail);
  :class:`BreakEvenValidationError` carries the result; the iteration plumbing on the cached
  model; the study specs round trip.

Slow: the three-year mixing recovery under the default config (owner's tolerances; un-xfailed
per report decision v), the explicit two-point configuration as a strict xfail with the measured
ρ_SX1 miss, the real-data run.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import LinearConstraint, minimize

from volsto.analytics.bergomi import atmf_skew_order1
from volsto.analytics.breakeven import first_order_breakevens, kernels
from volsto.analytics.reparam import BreakEvenParams, to_breakeven
from volsto.calibration.fit_2f import (
    BINDING_MESSAGE,
    INFEASIBLE_MESSAGE,
    NO_FIRST_ORDER_NOTE,
    NU_CAP_WARNING,
    BreakEvenFitConfig,
    BreakEvenValidationError,
    Stage3Inputs,
    affine_maps,
    affine_maps_from_kernels,
    breakeven_check,
    fit_2f,
    fit_2f_historical,
    fit_2f_marking,
    load_fit_spec,
    pillar_quad,
    resolve_skew_mode,
    stage3_validation,
    volvar_p1,
    write_fit_spec,
)
from volsto.calibration.history import SurfaceHistory, synthetic_2f_history
from volsto.calibration.stability import PARAM_COLUMNS, flag_unidentified, rolling_fit
from volsto.calibration.targets import marking_targets
from volsto.config import BergomiParams, CalibrationSpec, SimConfig, from_mapping, load_yaml
from volsto.market.varswap import xi0_curve

fit_2f_module = importlib.import_module("volsto.calibration.fit_2f")

ROOT = Path(__file__).resolve().parents[1]
P82 = BergomiParams(1.74, 0.245, 5.35, 0.28, 0.0, -0.759, -0.487)
STUDY_DIR = ROOT / "configs" / "studies" / "m7_p1_marking"
SPX = ROOT / "configs" / "surfaces" / "snapshots" / "hdn_2022H2" / "spx_2022-12-30.yaml"
SSR_SIM = SimConfig(n_paths=40_000, dt_max=1.0 / 100.0, chunk_size=20_000, seed=7)


def _cached(spec: Any) -> Any:
    """The cached LSV of ``spec`` (never calibrated here) or a skip."""
    from volsto.calibration.cache import CacheMissError, LeverageCache

    try:
        lsv, _ = LeverageCache(ROOT / "cache").get_or_calibrate(spec, allow_calibrate=False)
    except CacheMissError as exc:
        pytest.skip(f"leverage not in the cache (tests never calibrate): {exc}")
    return lsv


def _reference_spec(kind: str):  # type: ignore[no-untyped-def]
    spec = load_yaml(ROOT / "configs" / "studies" / f"lsv_reference_{kind}.yaml", CalibrationSpec)
    return dataclasses.replace(
        spec, particle=dataclasses.replace(spec.particle, n_particles=800_000)
    )


def _spx_surface() -> Any:
    from volsto.market.loaders import load_ssvi_surface

    return load_ssvi_surface(SPX)


@pytest.fixture(scope="module")
def spx():  # type: ignore[no-untyped-def]
    return _spx_surface()


@pytest.fixture(scope="module")
def ref_fits(ssvi):  # type: ignore[no-untyped-def]
    """The owner's two pairs on the reference SSVI (defaults otherwise)."""
    return {
        (ssr, eps): fit_2f_marking(ssvi, BreakEvenFitConfig(skew_eps=eps), ssr_target=ssr)
        for ssr, eps in ((1.0, 0.10), (1.5, 0.05))
    }


@pytest.fixture(scope="module")
def spx_fits(spx):  # type: ignore[no-untyped-def]
    return {
        (ssr, eps): fit_2f_marking(spx, BreakEvenFitConfig(skew_eps=eps), ssr_target=ssr)
        for ssr, eps in ((1.0, 0.10), (1.5, 0.05))
    }


@pytest.fixture(scope="module")
def synthetic_1y():  # type: ignore[no-untyped-def]
    return synthetic_2f_history(P82, years=1.0, seed=8, atm_source="order_one")


@pytest.fixture(scope="module")
def synthetic_3y():  # type: ignore[no-untyped-def]
    return synthetic_2f_history(P82, years=3.0, seed=13, atm_source="mixing", mixing_paths=100_000)


# --------------------------------------------------------------------------------------------
# configuration and machinery
# --------------------------------------------------------------------------------------------


def test_config_validation(ssvi) -> None:  # type: ignore[no-untyped-def]
    d = BreakEvenFitConfig()
    assert d.pillars == (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0) and d.mat_min == 0.25
    assert d.smooth_breakeven and d.sabrw_power == 1.0 and d.atf_ref == 0.3
    assert (d.k2, d.k1_bounds, d.skew_mode, d.skew_pillars) == (0.2, (0.3, 20.0), "auto", (1, 5))
    assert d.skew_eps == (0.10, 0.10) and d.eps_pair == (0.10, 0.10) and d.nu_cap == 3.5
    assert d.skew_weight == 10.0 and d.radicand_floor == 0.5 and d.stage3_tolerance == 0.10
    assert d.rho12_flag == 0.9
    assert d.weights_covar == d.weights_volvar == "relative" and d.chi_bounds == (-0.99, 0.99)
    tg = marking_targets(ssvi, (0.25, 1.0, 3.0))
    assert resolve_skew_mode(d, tg)[0] == "twopoint"
    hist_like = dataclasses.replace(tg, mode="historical", skew_fn=None, atm_vol_fn=None)
    mode, note = resolve_skew_mode(d, hist_like)
    assert mode == "soft" and "auto" in note
    explicit = dataclasses.replace(d, skew_mode="twopoint")
    assert resolve_skew_mode(explicit, hist_like) == ("twopoint", "")
    assert BreakEvenFitConfig(skew_eps=(0.1, 0.02)).eps_pair == (0.1, 0.02)
    assert BreakEvenFitConfig(skew_eps=0.05).skew_eps == (0.05, 0.05)
    for bad in (
        {"k2": 0.28},  # k1_bounds[0] = 0.3 < k2 + 0.05
        {"skew_eps": -0.1},
        {"skew_eps": (0.1, 0.1, 0.1)},
        {"skew_mode": "hard"},
        {"skew_pillars": (5.0, 1.0)},
        {"weights_covar": "log"},
        {"weights_volvar": (1.0, -1.0)},
        {"term_structure": "linear"},
        {"omega_max": 6.0},  # must exceed 2 nu_cap = 7
        {"nu_cap": 0.0},
        {"atf_ref": 0.0},
        {"skew_weight": -1.0},
        {"radicand_floor": 1.0},
        {"stage3_tolerance": 0.0},
        {"rho12_flag": 1.5},
    ):
        with pytest.raises(ValueError):
            BreakEvenFitConfig(**bad)
    # the YAML round trip of a config (the study specs carry it)
    from volsto.config import to_mapping

    c = BreakEvenFitConfig(skew_eps=(0.1, 0.05), weights_covar=(1.0, 2.0, 3.0))
    assert from_mapping(BreakEvenFitConfig, to_mapping(c)) == c


def test_affine_maps_match_engine(ssvi, rng) -> None:  # type: ignore[no-untyped-def]
    """``PillarQuad`` against ``kernels`` (A, J to 1e-12), the naked maps against
    ``first_order_breakevens(sigma_hat=atf)`` to 1e-12, affinity in ``(λ1, λ2)``."""
    pillars = (0.25, 0.5, 1.0, 2.0, 3.0)
    tg = marking_targets(ssvi, pillars)
    xi0 = xi0_curve(ssvi, 3.0)
    ks = (4.0, 0.2)
    quads = [pillar_quad(xi0, T, n_quad=64, n_inner=32) for T in pillars]
    kerns = [kernels(ks, xi0, T) for T in pillars]
    m_fast = affine_maps(quads, tg.atf, *ks)
    m_ref = affine_maps_from_kernels(ks, kerns, tg.atf)
    assert np.allclose(m_fast.A, m_ref.A, rtol=0, atol=1e-12)
    assert np.allclose(m_fast.J, m_ref.J, rtol=0, atol=1e-12)
    for _ in range(3):
        lam = rng.uniform(-3.0, 1.0, size=2)
        be = BreakEvenParams(ks[0], ks[1], 4.0, 4.0, lam[0], lam[1], 0.0)
        fo = [
            first_order_breakevens(be, xi0, T, sigma_hat=float(a)) for T, a in zip(pillars, tg.atf)
        ]
        assert np.allclose(m_fast.svc(lam), [f.spot_vol_covar for f in fo], rtol=0, atol=1e-12)
        assert np.allclose(m_fast.skew(lam), [f.skew for f in fo], rtol=0, atol=1e-12)
    a, b = rng.normal(size=2), rng.normal(size=2)
    for f in (m_fast.svc, m_fast.skew):
        assert np.allclose(f(a + b), f(a) + f(b), atol=1e-12)
        assert np.allclose(f(np.zeros(2)), 0.0)


def test_exact_qp_and_least_violation(rng) -> None:  # type: ignore[no-untyped-def]
    """The candidate enumeration of the inner 2-D QP against SLSQP on random instances (the ν
    box, random slabs, both together), and the least-violation programme: an empty set becomes
    feasible exactly at the returned relaxation of the slab rows, the box kept hard."""
    qp = fit_2f_module._qp2
    box = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    n_active = 0
    for i in range(30):
        M = rng.normal(size=(5, 2))
        y = rng.normal(size=5) * (4.0 if i % 2 else 0.5)
        H, g = M.T @ M, M.T @ y
        js, jl = rng.normal(size=2), rng.normal(size=2)
        c_s, c_l, width = rng.normal(), rng.normal(), rng.uniform(0.05, 1.0)
        slabs = np.array([js, -js, jl, -jl])
        hs = np.array([c_s + width, -(c_s - width), c_l + width, -(c_l - width)])
        if i % 3 == 0:
            G, h = box, np.full(4, rng.uniform(0.2, 2.0))
        elif i % 3 == 1:
            G, h = slabs, hs
        else:
            G, h = np.vstack((box, slabs)), np.concatenate((np.full(4, 50.0), hs))
        x, active, feas = qp(H, g, G, h)
        n_active += bool(active)
        assert feas and np.all(G @ x <= h + 1e-8)

        def quad(z: Any, H: Any = H, g: Any = g) -> Any:
            return 0.5 * z @ H @ z - g @ z

        res = minimize(
            quad,
            np.zeros(2),
            method="SLSQP",
            constraints=[LinearConstraint(G, -np.inf, h)],
            options={"ftol": 1e-14, "maxiter": 500},
        )
        assert quad(x) <= quad(res.x) + 1e-8 * (1.0 + abs(quad(res.x))), (quad(x), active)
    assert n_active >= 5
    # a slab x1 + x2 in [-3.2, -3] outside the box |x1| + |x2| <= 1: infeasible; the slab rows
    # relax by 2 (to x1 + x2 <= -1), the box stays hard
    G = np.vstack((box, [[1.0, 1.0], [-1.0, -1.0]]))
    h = np.array([1.0, 1.0, 1.0, 1.0, -3.0, 3.2])
    _, _, feas = qp(np.eye(2), np.zeros(2), G, h)
    assert not feas
    relax = np.array([False] * 4 + [True, True])
    delta = fit_2f_module._min_violation(G, h, relax)
    assert delta == pytest.approx(2.0, abs=1e-9)
    x, _, feas2 = qp(np.eye(2), np.zeros(2), G, h + (delta + 1e-9) * relax)
    assert feas2 and np.all(box @ x <= 1.0 + 1e-9)


def test_nu_feasibility_polytope(rng) -> None:  # type: ignore[no-untyped-def]
    """``min ν`` over ``ω_i ≥ |λ_i|`` and ``χ`` is ``|λ1 + λ2|/2`` at ``ω_i = |λ_i|``; for
    same-sign loadings it equals ``(|λ1| + |λ2|)/2``, the box row of step 2; ``volvar_p1``'s ``ν``
    equals the reparametrisation's."""
    for i in range(6):
        lam = rng.uniform(-3.0, 3.0, size=2)
        if i < 3:
            lam = -np.abs(lam)
        bound = 0.5 * abs(float(np.sum(lam)))
        at = BreakEvenParams(2.0, 0.2, abs(lam[0]), abs(lam[1]), lam[0], lam[1], 0.0).nu
        assert at == pytest.approx(bound, rel=1e-9)
        for _ in range(200):
            om = np.abs(lam) * np.exp(rng.uniform(0.0, 2.0, size=2))
            chi = rng.uniform(-1.0, 1.0)
            be = BreakEvenParams(2.0, 0.2, om[0], om[1], lam[0], lam[1], chi)
            assert be.nu >= bound * (1 - 1e-9)
            x = np.array([om[0], om[1], chi])
            _, nu = volvar_p1(x, lam, np.ones((1, 2)), np.ones(1), np.zeros(1))
            assert nu == pytest.approx(be.nu, rel=1e-10)


def test_p1_breakeven_identities(ssvi, rng) -> None:  # type: ignore[no-untyped-def]
    """The P1 (leverage-included) break-even of the module docstring on the reference SSVI:
    ``SpotVolCovar_P1`` affine in ``λ``; ``λ = 0`` gives ``σ_0 S R^LV(Mkt)`` with the order-one
    ``R^LV(Mkt)`` 3.007 / 3.014 / 3.027 at 3M / 6M / 1Y; the Table 8.2 kernel's first-order P1 SSR
    2.641 / 2.352 / 2.082 (the numerical SSR of the cached Table 8.2 LSV is 2.48 / – / 2.14,
    ``test_stage3_machinery_on_cached_2f``); the form reduces to the naked kernel's when the
    market skew term structure is the kernel's own order-one skew (``SensiSpot → 0``); in
    historical mode (pillar-interpolated residual) ``SensiSpot`` is exactly zero when the naked
    skew equals the market skew at the pillars."""
    pillars = (0.25, 0.5, 1.0, 2.0, 3.0)
    xi0 = xi0_curve(ssvi, 4.0)
    tg = marking_targets(ssvi, pillars)
    cfg = BreakEvenFitConfig(pillars=pillars, k2=0.28, k1_bounds=(0.5, 20.0))
    prob, _ = fit_2f_module._first_problem(tg, cfg, xi0)
    assert prob.bank.source == "surface" and prob.skew_mode == "twopoint"
    mm = prob.maps(5.35)
    a, b = rng.normal(size=2), rng.normal(size=2)
    f0 = mm.svc(np.zeros(2))
    assert np.allclose(mm.svc(a + b) - f0, (mm.svc(a) - f0) + (mm.svc(b) - f0))
    r_lv = f0 / (tg.sigma_0 * tg.skew_target)
    assert np.allclose(r_lv[:3], [3.007, 3.014, 3.027], rtol=1e-3), r_lv
    be = to_breakeven(P82)
    lam82 = np.array([be.lambda1, be.lambda2])
    assert np.allclose(mm.ssr_first_order(lam82)[:3], [2.641, 2.352, 2.082], rtol=3e-3)
    assert np.allclose(mm.sensi_spot(lam82), mm.svc(lam82) - mm.svc_naked(lam82))
    own = dataclasses.replace(
        tg,
        skew_target=np.array([atmf_skew_order1(P82, xi0, T) for T in pillars]),
        skew_fn=lambda t: np.array([atmf_skew_order1(P82, xi0, float(x)) for x in t]),
    )
    po, _ = fit_2f_module._first_problem(own, cfg, xi0)
    mo = po.maps(5.35)
    assert np.allclose(mo.skew_naked(lam82), own.skew_target, rtol=2e-3)
    assert np.allclose(mo.svc(lam82), mo.svc_naked(lam82), rtol=2e-3)
    # historical (pillars) source: zero residual at the pillars -> zero leverage term
    hist_like = dataclasses.replace(own, skew_fn=None, atm_vol_fn=None)
    hist_like = dataclasses.replace(hist_like, skew_target=mo.skew_naked(lam82))
    ph, _ = fit_2f_module._first_problem(hist_like, cfg, xi0)
    assert ph.bank.source == "pillars"
    mh = ph.maps(5.35)
    assert np.allclose(mh.sensi_spot(lam82), 0.0, atol=1e-14)
    assert np.allclose(mh.svc(lam82), mh.svc_naked(lam82), atol=1e-14)


def test_volvar_p1_quadratic_form(ssvi, rng) -> None:  # type: ignore[no-untyped-def]
    """``volvar_p1`` is the engine's quadratic form: with ``SensiSpot = 0`` it equals
    ``first_order_breakevens(sigma_hat=atf).vol_var`` to 1e-12; with a spot sensitivity it is
    ``VolVar_naked + SensiSpot² + 2 SensiSpot ½ atf λ·A``.  The owner's transcription with both
    cross terms at half weight is lower by the missing half, measured here (report decision ii:
    an unresolved normalisation, not used)."""
    pillars = (0.25, 1.0, 3.0)
    tg = marking_targets(ssvi, pillars)
    xi0 = xi0_curve(ssvi, 3.0)
    ks = (6.0, 0.2)
    quads = [pillar_quad(xi0, T, n_quad=64, n_inner=32) for T in pillars]
    m = affine_maps(quads, tg.atf, *ks)
    for _ in range(3):
        lam = -rng.uniform(0.2, 2.0, size=2)
        x = np.array([abs(lam[0]) * 1.4, abs(lam[1]) * 1.3, rng.uniform(-0.8, 0.8)])
        be = BreakEvenParams(ks[0], ks[1], x[0], x[1], lam[0], lam[1], x[2])
        fo = [
            first_order_breakevens(be, xi0, T, sigma_hat=float(a)) for T, a in zip(pillars, tg.atf)
        ]
        vv0, _ = volvar_p1(x, lam, m.A, tg.atf, np.zeros(3))
        assert np.allclose(vv0, [f.vol_var for f in fo], rtol=1e-12)
        spot = rng.normal(scale=0.02, size=3)
        vv, _ = volvar_p1(x, lam, m.A, tg.atf, spot)
        cross = 2 * spot * (0.5 * tg.atf * (m.A @ lam))
        assert np.allclose(vv, vv0 + spot**2 + cross, rtol=1e-12)
        sx, sy = 0.5 * x[0] * m.A[:, 0] * tg.atf, 0.5 * x[1] * m.A[:, 1] * tg.atf
        owner = spot**2 + 0.5 * cross + sx**2 + sy**2 + be.rho_XY * sx * sy
        print("owner's half cross terms / quadratic form:", np.round(owner / vv, 4))
        assert np.allclose(vv - owner, 0.5 * cross + be.rho_XY * sx * sy)


# --------------------------------------------------------------------------------------------
# the owner's test list
# --------------------------------------------------------------------------------------------


def test_breakeven_formula_reproduces_policy(ssvi, ref_fits) -> None:  # type: ignore[no-untyped-def]
    """At ``ssr_target = 1``: the raw ``VoV_BE`` is ``½ VoV_SABR atf_3M/atf_T`` (the policy, absolute
    reading) to 1e-12 at every pillar; the fit's requested VolVar is ``VoV_BE²`` (smoothed),
    ``Corr_BE = Corr_SABR``, ``SpotVolCovar_target = Corr_BE VoV_BE``; ``ssr = 1.5`` scales the raw
    curve by 1.5 exactly."""
    r = ref_fits[(1.0, 0.10)]
    tg = r.targets
    pc = tg.policy_check()
    assert (pc["reading"] == "absolute").all()
    for s, raw in zip(tg.sabr, tg.vov_be_raw):
        assert raw == pytest.approx(0.5 * s.vov_sabr * tg.atf_anchor / s.atf, rel=1e-12)
    assert np.allclose(r.table["volvar_target_requested"], tg.vovol**2, rtol=1e-12)
    assert np.allclose(r.table["corr_target"], [s.rho_sabr for s in tg.sabr], rtol=1e-12)
    assert np.allclose(r.table["svc_target"], tg.correl_target * tg.vovol, rtol=1e-12)
    t15 = ref_fits[(1.5, 0.05)].targets
    assert np.allclose(t15.vov_be_raw, 1.5 * tg.vov_be_raw, rtol=1e-12)
    assert r.risk_regime == "sticky_strike" and r.config.skew_mode == "twopoint"


def test_twopoint_constraint_binds_only_at_1y_5y(ref_fits, spx_fits) -> None:  # type: ignore[no-untyped-def]
    """The skew constraint acts at the two points only: every active skew row names T = 1 or 5
    (3 on SPX, quoted to 3Y, with the relocation note), the naked skew is inside ``(1 ± eps)
    Skew_SABR`` there, and the free pillars are not held to ``eps`` (the short end moves by more:
    reference (1.5, 0.05) +16% / +7% at 3M / 6M; SPX (1.0, 0.10) +57% / +33%); along the whole
    ``k1`` profile no other maturity is ever constrained."""
    for fits, points in ((ref_fits, {1.0, 5.0}), (spx_fits, {1.0, 3.0})):
        for (ssr, eps), r in fits.items():
            ct = r.constraints
            assert set(ct["T"]) == points and ct["within_eps"].all(), ct
            assert np.allclose(ct["eps"], eps)
            assert np.allclose(ct["lower"], (1 + eps) * ct["skew_market"])  # negative skews
            assert np.allclose(ct["upper"], (1 - eps) * ct["skew_market"])
            for label in (*r.first.active, *r.first.profile["active"]):
                for part in [x for x in str(label).split(";") if x.startswith("skew")]:
                    assert float(part.split("T=")[1].split()[0]) in points, part
            free = r.table[~r.table["T"].isin(list(points))]
            print(r.status, ssr, eps, np.round(r.table["skew_gap_rel"], 3).tolist())
            assert (free["skew_gap_rel"].abs() > eps).any()
    assert any("applied at the nearest pillar 3y" in n for n in spx_fits[(1.0, 0.10)].first.notes)
    assert any("dropped" in n and "5.0" in n for n in spx_fits[(1.0, 0.10)].notes)


def test_ssr_target_and_skew_eps_exercised(spx, spx_fits) -> None:  # type: ignore[no-untyped-def]
    """Both user inputs move the fit: ``ssr_target`` scales ``VoV_BE`` and ``SpotVolCovar_target``
    (1.5x) and moves ``(k1, λ)``; ``skew_eps`` moves the constraint bounds, and a pair applies each
    tolerance at its point; a per-pillar ``ssr_target`` curve is accepted; the soft all-pillar
    penalty runs (no skew rows, the mean naked-skew gap shrinks as the weight grows)."""
    a, b = spx_fits[(1.0, 0.10)], spx_fits[(1.5, 0.05)]
    assert np.allclose(b.table["svc_target"] / a.table["svc_target"], 1.5, rtol=2e-3)
    assert not np.isclose(a.params.k1, b.params.k1) and a.breakeven.lambda1 != b.breakeven.lambda1
    c = fit_2f_marking(spx, BreakEvenFitConfig(skew_eps=0.05), ssr_target=1.0)
    assert np.allclose(c.table["svc_target"], a.table["svc_target"])
    assert np.allclose(c.constraints["lower"], 1.05 * c.constraints["skew_market"])
    assert c.params != a.params
    pair = fit_2f_marking(spx, BreakEvenFitConfig(skew_eps=(0.10, 0.02)), ssr_target=1.0)
    assert np.allclose(pair.constraints["eps"], [0.10, 0.02])
    assert pair.constraints["gap_rel"].abs().iloc[1] <= 0.02 + 1e-9
    curve = fit_2f_marking(spx, BreakEvenFitConfig(), ssr_target={0.25: 1.2, 3.0: 0.9})
    expected = np.interp(curve.table["T"], [0.25, 3.0], [1.2, 0.9])
    assert np.allclose(curve.targets.ssr_target, expected)
    gaps = []
    for w in (0.1, 10.0):
        cfg_soft = BreakEvenFitConfig(skew_mode="soft", skew_weight=w)
        s = fit_2f_marking(spx, cfg_soft, ssr_target=1.0)
        assert s.config.skew_mode == "soft"
        assert not any(x.startswith("skew") for x in s.first.active)
        assert (s.constraints["binding_edge"] == "").all()
        gaps.append(s.mean_skew_gap)
    assert gaps[1] < gaps[0], gaps


def test_binding_message_on_incompatible_pair(ref_fits, spx_fits, ssvi, caplog) -> None:  # type: ignore[no-untyped-def]
    """An incompatible ``(ssr_target, skew_eps)`` pair binds: on SPX ``(1.0, 0.10)`` the 1Y point
    binds at the steep edge ``(1+0.1)`` and the message names the maturity, the edge, the
    naked-vs-market skew and the achieved-vs-target SpotVolCovar per pillar (on the repaired
    eSSVI anchor of 2026-09-22 the 3Y point is free; on the plain-SSVI anchor it bound at the flat
    edge, its 2y/3y skew being steeper: −0.218 / −0.191 against −0.194 / −0.166); on the
    reference SSVI ``(1.0, 0.10)`` the SpotVolCovar the SSR asks for is out of reach within the ν
    cap 3.5 (the P1 covariance is +97% of target at 3M), both skew points bind and step 3 sits on
    the cap (``|ρ_SX1| = 1``, ρ12 +0.987: the collapse note fires); at ``(1.5, 0.05)`` the ν box
    binds (ρ = −1 / −1, ρ12 = +1) and the owner's ν-cap warning is logged and attached.  On SPX
    the constraint binds for every pair of ``ssr ∈ {1, 1.25, 1.5, 1.75, 2} × eps ∈ {0.05, 0.1, 0.2,
    0.3}`` except ``ssr 1.75`` with ``eps ≥ 0.2``: that compatible pair binds nothing, carries no
    message and meets every SpotVolCovar target within 4% (2.7 / 3.9 / 0.05 / 3.2 / 1.6 % on the
    repaired eSSVI anchor of 2026-09-22; within 2% on the plain-SSVI anchor)."""
    r = spx_fits[(1.0, 0.10)]
    assert r.status == "binding"
    msgs = [m for m in r.messages if m.startswith("skew constraint binds")]
    assert len(msgs) == 1
    assert "T=1 (lower edge: naked skew = (1+0.1) x Skew_SABR" in msgs[0]
    assert list(r.constraints["binding_edge"]) == ["(1+0.1)", ""]
    for m in msgs:
        assert "SpotVolCovar achieved vs target: 0.25y" in m and "3y" in m
    assert BINDING_MESSAGE.split("{")[0] in msgs[0]
    ref = ref_fits[(1.0, 0.10)]
    assert ref.status == "binding" and ref.second.nu_at_cap and not ref.first.box_binding
    assert (ref.constraints["binding_edge"] != "").all()
    warn = NU_CAP_WARNING.format(cap=3.5)
    assert warn == (
        "nu at cap 3.5; first-order break-even engine ~15% biased beyond ~4; raise only if "
        "stage-3 simulation validates"
    )
    assert any(m.startswith(warn) for m in ref.messages)
    assert ref.svc_rel_error[0] > 0.9 and abs(ref.params.rho_SX1) > 0.9999
    assert ref.params.nu == pytest.approx(3.5, rel=1e-4)
    assert any("collapsing" in n for n in ref.notes) and abs(ref.params.rho12) > 0.9
    ref15 = ref_fits[(1.5, 0.05)]
    assert ref15.first.box_binding and ref15.second.nu_at_cap
    assert any("collapsing" in n for n in ref15.notes)
    with caplog.at_level(logging.WARNING, logger="volsto.calibration.fit_2f"):
        fit_2f_marking(ssvi, BreakEvenFitConfig(skew_eps=0.05), ssr_target=1.5)
    assert any(warn in rec.getMessage() for rec in caplog.records)
    ok = fit_2f_marking(_spx_surface(), BreakEvenFitConfig(skew_eps=0.20), ssr_target=1.75)
    print(ok.status, ok.messages, np.round(ok.constraints["gap_rel"], 4).tolist())
    assert ok.status == "interior" and ok.messages == () and ok.first.active == ()
    assert np.all(np.abs(ok.svc_rel_error) < 0.04), ok.svc_rel_error


def test_infeasible_message(spx) -> None:  # type: ignore[no-untyped-def]
    """With a ν box too small for the market skew (``ν_cap = 0.3``: ``|λ1| + |λ2| ≤ 0.6``) no
    ``λ`` meets the two points at any ``k1``: status ``infeasible``, the least-violation fit is
    returned (finite parameters, positive relaxation) with the message naming the constraint, the
    box, the naked-vs-market skews and the SpotVolCovar per pillar, and the ν-cap warning."""
    r = fit_2f_marking(spx, BreakEvenFitConfig(nu_cap=0.3, omega_max=5.0), ssr_target=1.0)
    print(r.summary())
    assert r.status == "infeasible" and not r.first.feasible and r.first.violation > 0
    m = r.messages[0]
    assert m.startswith("infeasible: no (lambda1, lambda2) meets the two-point skew constraint")
    assert "2 nu_cap = 0.6" in m and "naked skew 1y" in m and "SpotVolCovar achieved" in m
    assert INFEASIBLE_MESSAGE.split("{")[0] in m
    assert any(x.startswith("nu at cap 0.3") for x in r.messages)
    assert all(np.isfinite(v) for v in dataclasses.asdict(r.params).values())
    assert r.params.nu <= 0.3 * (1 + 1e-6)


def test_k2_fitted_within_bounds(spx) -> None:  # type: ignore[no-untyped-def]
    """Owner's decision of 2026-09-26: ``k2`` may be tuned.  With ``k2_bounds`` the first
    minimisation fits ``k2`` on a geometric grid that includes the fixed ``k2`` and refines it,
    each candidate running the ``k1`` search above ``k2 + k1_min_gap``: the covariance objective
    never exceeds the fixed-``k2`` fit's, the fitted ``k2`` lies inside the bounds and reaches the
    parameters, and the config round-trips through the fit's YAML.  On the SPX anchor at
    ``(1.0, 0.10)`` it lands near 0.89 and the largest covariance miss falls from 14.4 % to
    about 5 % (measured 2026-09-26; SPEC §15 Part 3).  Bounds that leave no room for ``k1``, a
    decreasing or non-positive range and a grid under three points raise."""
    import yaml

    fixed = fit_2f_marking(spx, BreakEvenFitConfig(skew_eps=0.10), ssr_target=1.0)
    free_cfg = BreakEvenFitConfig(skew_eps=0.10, k2_bounds=(0.05, 1.5))
    free = fit_2f_marking(spx, free_cfg, ssr_target=1.0)
    assert fixed.first.k2 == fixed.params.k2 == free_cfg.k2 == 0.2
    assert 0.05 < free.first.k2 < 1.5 and free.params.k2 == pytest.approx(free.first.k2)
    assert free.params.k1 > free.params.k2 + free_cfg.k1_min_gap
    assert free.first.objective <= fixed.first.objective * (1.0 + 1e-9)
    assert any(n.startswith("k2 fitted") for n in free.first.notes)
    worst = {
        name: float(np.max(np.abs(r.svc_rel_error)))
        for name, r in (("fixed", fixed), ("free", free))
    }
    print(f"\nk2 {free.first.k2:.3f}: largest covariance miss {worst}")
    assert worst["free"] < 0.5 * worst["fixed"]
    doc = yaml.safe_load(free.config_yaml)
    assert from_mapping(BreakEvenFitConfig, doc["provenance"]["config"]) == free.config
    for bad in ((0.5, 0.2), (0.0, 1.0), (0.1, 19.99)):
        with pytest.raises(ValueError, match="k2_bounds"):
            BreakEvenFitConfig(k2_bounds=bad)
    with pytest.raises(ValueError, match="k2_grid"):
        BreakEvenFitConfig(k2_bounds=(0.1, 1.0), k2_grid=2)
    with pytest.raises(ValueError, match="k2_grid needs k2_bounds"):
        BreakEvenFitConfig(k2_grid=5)
    # unset, the new options leave the config's mapping as it was: the backtest hashes the resolved
    # fit config, so a field added at a behaviour-neutral default must not change it
    from volsto.config import to_mapping

    default_map = to_mapping(BreakEvenFitConfig())
    assert "k2_bounds" not in default_map and "k2_grid" not in default_map
    assert to_mapping(free_cfg)["k2_bounds"] == [0.05, 1.5]


def test_skew_band_on_more_pillars(spx) -> None:  # type: ignore[no-untyped-def]
    """Owner's decision of 2026-09-26: the band should also apply at the short end.
    ``skew_pillars`` takes any increasing maturities and ``skew_eps`` one value for all or one per
    pillar.  On the SPX anchor at ``(1.0, 0.10)`` with ``(0.25, 0.5, 1, 3)`` every point is a
    constraint (``T_1`` … ``T_4``), the fit is feasible with the model's skew inside ±10 % at all
    four (binding at 3M), and the free short end is 1M alone — at a price: the largest covariance
    miss is several times the two-point fit's (measured 71 % against 14 %, SPEC §15 Part 3).  The
    two-point default, its names, ``eps_pair`` and its mapping are unchanged; the config
    round-trips; mismatched lengths and a non-increasing list raise."""
    import yaml

    from volsto.config import to_mapping

    four = BreakEvenFitConfig(skew_eps=0.10, skew_pillars=(0.25, 0.5, 1.0, 3.0))
    assert four.skew_eps == (0.10, 0.10, 0.10, 0.10)
    with pytest.raises(ValueError, match="two-point"):
        _ = four.eps_pair
    r = fit_2f_marking(spx, four, ssr_target=1.0)
    ct = r.constraints
    assert list(ct["name"]) == ["T_1", "T_2", "T_3", "T_4"]
    np.testing.assert_allclose(ct["T"], [0.25, 0.5, 1.0, 3.0])
    assert r.first.feasible and bool(ct["within_eps"].all()) and r.status == "binding"
    assert "T=0.25" in r.messages[0]
    np.testing.assert_allclose(r.first.short_end["T"], [1.0 / 12.0])
    two = fit_2f_marking(spx, BreakEvenFitConfig(skew_eps=0.10), ssr_target=1.0)
    assert list(two.constraints["name"]) == ["T_s", "T_l"]
    worst_four = float(np.max(np.abs(r.svc_rel_error)))
    worst_two = float(np.max(np.abs(two.svc_rel_error)))
    print(f"\nlargest covariance miss: four-point band {worst_four:.3f}, two-point {worst_two:.3f}")
    assert worst_four > 2.0 * worst_two
    per_point = BreakEvenFitConfig(
        skew_eps=(0.3, 0.2, 0.1, 0.1), skew_pillars=(0.25, 0.5, 1.0, 3.0)
    )
    assert per_point.skew_eps == (0.3, 0.2, 0.1, 0.1)
    doc = yaml.safe_load(r.config_yaml)
    assert from_mapping(BreakEvenFitConfig, doc["provenance"]["config"]) == r.config
    d = BreakEvenFitConfig()
    assert d.skew_pillars == (1.0, 5.0) and d.eps_pair == (0.1, 0.1)
    assert to_mapping(d)["skew_pillars"] == [1.0, 5.0] and to_mapping(d)["skew_eps"] == [0.1, 0.1]
    with pytest.raises(ValueError, match="one non-negative value per skew pillar"):
        BreakEvenFitConfig(skew_pillars=(0.25, 1.0, 3.0), skew_eps=(0.1, 0.1))
    with pytest.raises(ValueError, match="increasing"):
        BreakEvenFitConfig(skew_pillars=(1.0, 0.5))


def test_correlation_min_eigenvalue(spx_fits, ref_fits) -> None:  # type: ignore[no-untyped-def]
    """:func:`correlation_min_eigenvalue` is the smallest eigenvalue of the correlation matrix of
    ``(S, X¹, X²)``: 1 for independent factors, the closed form ``1 − sqrt(ρ_SX1² + ρ_SX2²)`` when
    the factors are uncorrelated with each other, near 0 at the collapsed corner.  Every fit is
    noted near-singular exactly when it reads below :data:`CORRELATION_EIGEN_FLAG`; the SPX
    ``(1.0, 0.10)`` fit on the repaired eSSVI anchor is (6·10⁻⁴: ``χ`` at its −0.99 bound —
    SPEC §8.2)."""
    from volsto.calibration.fit_2f import CORRELATION_EIGEN_FLAG, correlation_min_eigenvalue

    ind = dataclasses.replace(P82, rho_SX1=0.0, rho_SX2=0.0, rho12=0.0)
    assert correlation_min_eigenvalue(ind) == pytest.approx(1.0, abs=1e-12)
    unc = dataclasses.replace(P82, rho_SX1=-0.6, rho_SX2=-0.5, rho12=0.0)
    assert correlation_min_eigenvalue(unc) == pytest.approx(1 - np.hypot(0.6, 0.5), abs=1e-12)
    near = dataclasses.replace(P82, rho_SX1=-0.99, rho_SX2=-0.99, rho12=0.98)
    assert 0.0 < correlation_min_eigenvalue(near) < 1e-2
    for r in (*spx_fits.values(), *ref_fits.values()):
        flagged = any("near-singular" in n for n in r.notes)
        assert flagged == (r.min_correlation_eigenvalue < CORRELATION_EIGEN_FLAG), r.notes
    anchor = spx_fits[(1.0, 0.10)]
    assert anchor.min_correlation_eigenvalue < 1e-3
    assert any("near-singular" in n for n in anchor.notes)


def test_correl_rho_sabr_kept(spx_fits) -> None:  # type: ignore[no-untyped-def]
    """``Corr_BE = ρ_SABR`` always: the VolVar target is ``(SpotVolCovar_P1 / Corr_BE)²``, so the
    fitted model's spot/vol correlation ``SpotVolCovar / sqrt(VolVar)`` sits on ``ρ_SABR`` (within
    1% where the VolVar fit is tight) even where the covariance target is missed, and no
    correlation is railed on SPX.  On the repaired eSSVI anchor of 2026-09-22 the ρ12 collapse
    note (decision vii) fires on neither pair (ρ12 +0.65 at ``(1.0, 0.10)``, +0.30 at
    ``(1.5, 0.05)``; on the plain-SSVI anchor it fired on ``(1.5, 0.05)`` at +0.988) — the
    collapse itself is exercised on the reference surface in
    :func:`test_binding_message_on_incompatible_pair`."""
    assert not any("collapsing" in n for n in spx_fits[(1.5, 0.05)].notes)
    assert not any("collapsing" in n for n in spx_fits[(1.0, 0.10)].notes)
    assert 0.2 < spx_fits[(1.5, 0.05)].params.rho12 < 0.4
    for r in spx_fits.values():
        t = r.table
        assert np.allclose(t["volvar_target"], (t["svc_model"] / t["corr_target"]) ** 2)
        # Corr_BE = rho_SABR is imposed through the VolVar target, so the fitted correlation
        # misses rho_SABR by half the VolVar fit's relative miss: within 1% where the VolVar fit
        # is within 2%, within 3% everywhere (on the repaired eSSVI anchor of 2026-09-22 the 2y
        # and 3y VolVar fits miss by 3-4%, the correlations by up to 2.1%; the plain-SSVI anchor
        # fitted VolVar within 2% at every pillar)
        ratio = np.sqrt(t["volvar_target"] / t["volvar_model"])
        np.testing.assert_allclose(t["corr_model"] / t["corr_target"], ratio, rtol=1e-6)
        tight = np.abs(t["volvar_model"] / t["volvar_target"] - 1) < 0.02
        assert tight.any()
        assert np.allclose(t["corr_model"][tight], t["corr_target"][tight], rtol=0.01), t[
            ["corr_model", "corr_target"]
        ]
        assert np.allclose(t["corr_model"], t["corr_target"], rtol=0.03), t[
            ["corr_model", "corr_target"]
        ]
        p = r.params
        assert max(abs(p.rho_SX1), abs(p.rho_SX2)) < 0.99 and not r.second.nu_at_cap
        assert not any("|rho_SX" in f for f in r.second.bound_flags)


def test_tables_yaml_and_fit_spec(spx_fits, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The always-reported tables (constraints, short end with 1M, per pillar), the loadable YAML
    and the study-spec round trip."""
    r = spx_fits[(1.0, 0.10)]
    assert list(r.short_end["T"])[:3] == pytest.approx([1 / 12, 0.25, 0.5])
    assert {"skew_market", "skew_naked", "gap_rel", "source"} <= set(r.short_end)
    assert {"svc_target", "svc_model", "sensi_spot", "ssr_first_order", "ssr_implied"} <= set(
        r.table
    )
    assert "two-point skew constraint" in r.summary() and "free short-end" in r.summary()
    import yaml

    doc = yaml.safe_load(r.config_yaml)
    assert from_mapping(BergomiParams, doc["model"]) == r.params
    assert from_mapping(BreakEvenFitConfig, doc["provenance"]["config"]) == r.config
    assert doc["provenance"]["status"] == "binding" and len(doc["provenance"]["messages"]) == 1
    assert doc["provenance"]["iterations"] is None and doc["provenance"]["stage3"] is None
    base = dataclasses.replace(_reference_spec("2f"), model=P82)
    path = write_fit_spec(r, base, tmp_path / "x.yaml", n_particles=1234, ssr_target=1.0, label="x")
    fs = load_fit_spec(path)
    assert fs.spec.model == r.params and fs.spec.particle.n_particles == 1234
    assert fs.config == r.config and fs.ssr_target == 1.0 and fs.breakeven == r.breakeven


def test_realised_lsv_ssr_reported_for_study_fits(fast_sim) -> None:  # type: ignore[no-untyped-def]
    """The cached study fits of ``scripts/m7_p1_marking.py`` (``configs/studies/m7_p1_marking``;
    skipped when absent): the recorded fit is reproduced by the fitter (same parameters, the cache
    key), the cached leverage is read (never calibrated), stage 3 reports the calibrated LSV's
    numerical SSR at 3M and 1Y with standard errors, the mean ``|L − 1|`` and the stage-3 assertion
    verdict at 3M — reported, not asserted equal to the SSR target / pass (the owner: the realised
    SSR is a diagnostic, about 1.4–2.0 at ``ssr_target = 1``)."""
    from volsto.calibration.cache import build_market

    specs = sorted(STUDY_DIR.glob("*_ssr*_eps*.yaml")) if STUDY_DIR.is_dir() else []
    if not specs:
        pytest.skip("study specs absent (scripts/m7_p1_marking.py)")
    sim = dataclasses.replace(fast_sim, n_paths=20_000, chunk_size=20_000)
    for path in specs:
        fs = load_fit_spec(path)
        _, surface, _ = build_market(fs.spec)
        r = fit_2f_marking(surface, fs.config, ssr_target=fs.ssr_target)
        assert r.params == fs.spec.model, (path.name, r.params, fs.spec.model)
        assert r.status == fs.fit["status"]
        lsv = _cached(fs.spec)
        rep = stage3_validation(
            r.params,
            Stage3Inputs(
                surface=surface,
                particle=fs.spec.particle,
                sim=fs.spec.sim,
                pricing_sim=sim,
                ssr_pillars=(0.25, 1.0),
                breakeven_pillars=(0.25,),
                forward_starts=(),
                model=lsv,
            ),
            r.targets,
            fit_table=r.table,
            tolerance=fs.config.stage3_tolerance,
        )
        st = rep.ssr_table
        print(
            path.name,
            r.status,
            f"|L-1| {rep.mean_abs_l_minus_1:.3f}",
            "stage-3 assertion",
            "PASS" if rep.within_tolerance else "FAIL: " + rep.check_message,
        )
        print(st.round(3).to_string(index=False))
        assert len(rep.check) == 2 and rep.within_tolerance == bool(rep.check["within"].all())
        assert not rep.recalibrated and np.all(np.isfinite(st["ssr_lsv"]))
        assert (st["ssr_lsv_se"] > 0).all() and np.isfinite(rep.mean_abs_l_minus_1)
        assert np.allclose(st["ssr_target"], fs.ssr_target)


# --------------------------------------------------------------------------------------------
# historical mode, stability, stage 3
# --------------------------------------------------------------------------------------------


def test_historical_recovery_fast(synthetic_1y) -> None:  # type: ignore[no-untyped-def]
    """Recovery on the one-year order-one synthetic history (k2 fixed at the true 0.28, MatMin 3M,
    windows 200 / 200) under the **default** config (``skew_mode="auto"`` → soft all-pillar penalty
    at weight 10): ν 1.602, θ 0.247, k1 5.217, ρ_SX1 −0.789, ρ_SX2 −0.508 (true 1.74, 0.245, 5.35,
    −0.759, −0.487) within 15% / 15% / 20% / 0.2 / 0.2; SpotVolCovar within 5% of target; standard
    errors from the target noise.  The explicit two-point configuration is reported: every
    SpotVolCovar within 2% but ν 1.422 (−18%) and k1 4.47 (−16%): the covariance targets at five
    pillars do not pin ``(k1, λ)`` when the short-end skew is free (the leverage term of the P1
    break-even absorbs it)."""
    hist = synthetic_1y.history
    base = BreakEvenFitConfig(
        pillars=tuple(map(float, hist.pillars)), k2=0.28, k1_bounds=(0.5, 20.0)
    )
    r = fit_2f_historical(hist, base, window_vol=200, window_ssr=200)
    p = r.params
    print(r.summary())
    assert r.config.skew_mode == "soft" and any("auto" in n for n in r.notes)
    assert r.targets.mode == "historical" and r.pricing_date == hist.dates[-1]
    assert r.table["T"].tolist() == [0.25, 0.5, 1.0, 2.0, 3.0]
    assert abs(p.nu - P82.nu) < 0.15 * P82.nu, p
    assert abs(p.theta - P82.theta) < 0.15 * P82.theta, p
    assert abs(p.k1 - P82.k1) < 0.20 * P82.k1, p
    assert abs(p.rho_SX1 - P82.rho_SX1) < 0.2 and abs(p.rho_SX2 - P82.rho_SX2) < 0.2, p
    assert r.first.feasible and np.all(np.abs(r.svc_rel_error) < 0.05), r.svc_rel_error
    assert np.isfinite(r.first.k1_se)
    assert np.all(np.isfinite([r.first.lambda1_se, r.first.lambda2_se]))
    assert np.allclose(r.table["volvar_target"], r.targets.vol_var)  # empirical VolVar kept
    d = fit_2f_historical(
        hist, dataclasses.replace(base, skew_mode="twopoint"), window_vol=200, window_ssr=200
    )
    print("two-point:", d.status, d.params, np.round(d.svc_rel_error, 3).tolist())
    assert np.all(np.abs(d.svc_rel_error) < 0.02)
    assert any("skew constraint T_l = 5y" in n for n in d.first.notes)


def test_rolling_fit_and_flags(synthetic_1y) -> None:  # type: ignore[no-untyped-def]
    """The rolling fit every 5 dates over dates 200–230 (windows 120 / 60, the default config: soft
    weight 10 in historical mode): seven fits with the parameter, standard-error and status
    columns; standard errors above ``max_se`` are NaN with the raw value kept;
    ``flag_unidentified`` reads the break-even columns."""
    hist = synthetic_1y.history
    cfg = BreakEvenFitConfig(
        pillars=tuple(map(float, hist.pillars)), k2=0.28, k1_bounds=(0.5, 20.0)
    )
    frame = rolling_fit(
        hist,
        cfg,
        every=5,
        window_vol=120,
        window_ssr=60,
        start=hist.dates[200],
        end=hist.dates[230],
    )
    print(frame.drop(columns=["date"]).round(4).to_string())
    assert len(frame) == 7
    for c in PARAM_COLUMNS:
        assert c in frame and f"{c}_se" in frame and f"{c}_se_raw" in frame
    assert {
        "nu",
        "theta",
        "rho_SX1",
        "rho_SX2",
        "rho12",
        "first_objective",
        "second_objective",
        "status",
        "svc_rel_error_max",
        "skew_gap_T_s",
        "skew_gap_T_l",
        "mean_skew_gap",
        "active",
        "messages",
        "bound_flags",
    } <= set(frame.columns)
    assert (frame["k2"] == 0.28).all() and set(frame["status"]) <= {"interior", "binding"}
    se = frame[[f"{c}_se" for c in PARAM_COLUMNS]].to_numpy(dtype=float)
    assert np.all(np.isnan(se) | (se <= fit_2f_module.MAX_FINITE_SE))
    flags = flag_unidentified(frame)
    assert list(flags["param"]) == list(PARAM_COLUMNS) and flags["unidentified"].dtype == bool
    with pytest.raises(ValueError):
        rolling_fit(hist, cfg, start=hist.dates[10], window_vol=120)


def _check_input(**over: float) -> pd.DataFrame:
    """A one-pillar simulated break-even table (the columns ``breakeven_check`` reads): both
    targets missed by +25% at first order (a binding fit) with a small engine bias at the solution
    (SpotVolCovar +2%, VolVar +0.8%); ``volvar_target_fit`` is the step-3 target the fit solved."""
    rec: dict[str, float] = {
        "T": 0.25,
        "svc_sim": -0.102,
        "svc_se": 0.001,
        "svc_target": -0.08,
        "svc_p1_first_order": -0.10,
        "volvar_sim": 0.0126,
        "volvar_se": 0.0001,
        "volvar_target": 0.02,
        "volvar_target_fit": 0.01,
        "volvar_p1_first_order": 0.0125,
    }
    rec.update(over)
    return pd.DataFrame([rec])


def test_breakeven_check_engine_bias_rule() -> None:
    """The owner's split of the stage-3 assertion on synthetic check tables: ``within`` is
    ``|engine_bias| <= tolerance`` (simulated against the first-order break-evens at the fitted
    parameters); ``gap_vs_target`` and ``first_order_miss`` are reported columns.  A +25% target
    miss with a 2% engine bias passes with an empty message and the miss in the table; a +20%
    engine bias fails with the message naming that row's engine bias only (the VolVar row with the
    large miss and small bias is not a failure) and one sentence listing the first-order misses
    above the tolerance as reported, not asserted; misses below the tolerance are not listed; a
    row without a finite non-zero first-order value falls back to the target gap and is flagged
    :data:`NO_FIRST_ORDER_NOTE`; ``volvar_target`` stands in for a missing
    ``volvar_target_fit``."""
    chk, ok, msg = breakeven_check(_check_input(), 0.10)
    assert ok and msg == "" and chk["within"].all() and (chk["note"] == "").all()
    assert list(chk.columns) == [
        "T",
        "quantity",
        "sim",
        "se",
        "target",
        "first_order",
        "gap_vs_target",
        "first_order_miss",
        "engine_bias",
        "within",
        "note",
    ]
    svc = chk[chk["quantity"] == "SpotVolCovar"].iloc[0]
    assert svc["first_order_miss"] == pytest.approx(0.25)
    assert svc["engine_bias"] == pytest.approx(0.02)
    assert svc["gap_vs_target"] == pytest.approx(0.275)
    vv = chk[chk["quantity"] == "VolVar"].iloc[0]
    assert vv["target"] == 0.01 and vv["first_order_miss"] == pytest.approx(0.25)
    assert vv["engine_bias"] == pytest.approx(0.008)
    # a +20% engine bias on the covariance: FAIL on that row only, the misses reported
    chk, ok, msg = breakeven_check(_check_input(svc_sim=-0.12), 0.10)
    assert not ok and list(chk["within"]) == [False, True]
    assert msg.startswith(
        "simulated break-evens outside 10% of the first-order break-evens at the fitted "
        "parameters (engine bias): SpotVolCovar at T=0.25: simulated -0.12000 +/- 0.00100 vs "
        "first-order -0.10000 at the fitted parameters (engine bias +20.0%; target -0.08000, "
        "+50.0%)"
    )
    assert "VolVar at T=0.25: simulated" not in msg
    assert msg.endswith(
        ". First-order misses of the targets above 10%, reported, not asserted (the first-order "
        "fit's miss of its target, a binding constraint when the fit says so): "
        "SpotVolCovar at T=0.25 +25.0%, VolVar at T=0.25 +25.0%."
    )
    # misses below the tolerance are not listed
    _, ok, msg = breakeven_check(
        _check_input(svc_sim=-0.12, svc_target=-0.098, volvar_target_fit=0.0123), 0.10
    )
    assert not ok and "engine bias +20.0%" in msg and "reported, not asserted" not in msg
    # the tolerance applies to the engine bias: 2% fails at 1%, 0.8% passes
    chk, ok, msg = breakeven_check(_check_input(), 0.01)
    assert not ok and list(chk["within"]) == [False, True] and "engine bias +2.0%" in msg
    # no first-order value: the target gap decides, flagged
    chk, ok, msg = breakeven_check(_check_input(svc_p1_first_order=float("nan")), 0.10)
    svc = chk[chk["quantity"] == "SpotVolCovar"].iloc[0]
    assert not ok and not svc["within"] and svc["note"] == NO_FIRST_ORDER_NOTE
    assert np.isnan(svc["engine_bias"]) and np.isnan(svc["first_order_miss"])
    assert (
        "SpotVolCovar at T=0.25: simulated -0.10200 +/- 0.00100 vs target -0.08000 (+27.5%; "
        f"{NO_FIRST_ORDER_NOTE})" in msg
    ) and msg.endswith("when the fit says so): VolVar at T=0.25 +25.0%.")
    # every failing row a fallback: the header names the target gap, not the engine bias
    assert "(no first-order value: target gap): SpotVolCovar" in msg
    assert chk[chk["quantity"] == "VolVar"]["within"].all()
    chk, ok, msg = breakeven_check(_check_input(svc_p1_first_order=0.0, svc_sim=-0.085), 0.10)
    svc = chk[chk["quantity"] == "SpotVolCovar"].iloc[0]
    assert ok and msg == "" and svc["within"] and svc["note"] == NO_FIRST_ORDER_NOTE
    # volvar_target when the fit carried no step-3 target
    chk, ok, _ = breakeven_check(_check_input(volvar_target_fit=float("nan")), 0.10)
    vv = chk[chk["quantity"] == "VolVar"].iloc[0]
    assert ok and vv["target"] == 0.02 and vv["first_order_miss"] == pytest.approx(-0.375)


def test_stage3_machinery_on_cached_2f(ssvi, fast_sim) -> None:  # type: ignore[no-untyped-def]
    """Stage 3 on the cached 2F Table 8.2 LSV (8e5 particles, never calibrated here) through the
    ``model=`` override with the Table 8.2 kernel's P1 table: the numerical LSV SSR (2.48 / 2.14 at
    3M / 1Y) against the P1 first-order SSR (2.64 / 2.08, within 10%), the SSR the targets imply
    (≈ 0.95), the naked mixing skew, the simulated break-evens with standard errors, the forward
    table at 1y-into-1y with the spot 1y skew and its ratio; "recalibrated: no".  The break-even
    assertion under the owner's split: the Table 8.2 LSV is not an ``ssr = 1`` fit, so the
    SpotVolCovar rows carry a first-order miss of +177% / +118% (reported) with an engine bias
    within 10% (−6% / +5%, within); the step-2 table carries no VolVar first-order value, so the
    VolVar rows fall back to the target gap (flagged) and fail; the message names the fallback
    rows and lists the SpotVolCovar misses as reported, not asserted; ``fit_2f`` with stage 3 on
    the cached model raises :class:`BreakEvenValidationError` carrying the result (the fitted
    parameters' first order against the Table 8.2 simulation: an engine-bias failure), and
    ``iterate_against_simulation=1`` (plumbing only: the cached model stands in for the
    calibration) records two iteration rows and the correction columns."""
    from volsto.calibration.cache import build_market

    spec = _reference_spec("2f")
    lsv = _cached(spec)
    _, surface, _ = build_market(spec)
    pillars = (0.25, 1.0)
    tg = marking_targets(surface, pillars)
    cfg = BreakEvenFitConfig(pillars=pillars, k2=0.28, k1_bounds=(0.5, 20.0))
    prob, _ = fit_2f_module._first_problem(tg, cfg, lsv.kernel.xi0)
    be = to_breakeven(P82)
    lam = np.array([be.lambda1, be.lambda2])
    table = fit_2f_module._first_table(prob, lam, prob.maps(P82.k1))
    sim = dataclasses.replace(fast_sim, n_paths=40_000)
    inputs = Stage3Inputs(
        surface=surface,
        particle=spec.particle,
        sim=spec.sim,
        pricing_sim=sim,
        ssr_pillars=pillars,
        breakeven_pillars=pillars,
        forward_starts=((1.0, 2.0),),
        model=lsv,
        mixing_paths=50_000,
    )
    rep = stage3_validation(P82, inputs, tg, fit_table=table)
    print(rep.summary())
    assert not rep.recalibrated and rep.calibration_seconds == 0.0 and rep.n_particles == 800_000
    st = rep.ssr_table
    assert (st["ssr_lsv_se"] > 0).all() and np.allclose(st["ssr_target"], 1.0)
    assert np.all(np.abs(st["ssr_first_order_p1"] / st["ssr_lsv"] - 1.0) < 0.10), st
    assert np.all((st["ssr_implied_by_targets"] > 0.9) & (st["ssr_implied_by_targets"] < 1.0))
    assert (rep.skew_table["skew_naked_se"] > 0).all() and (rep.skew_table["skew_naked"] < 0).all()
    bt = rep.breakeven_table
    assert (bt["svc_se"] > 0).all() and np.all(np.isfinite(bt[["corr_sim", "corr_target"]]))
    ft = rep.forward_table
    assert len(ft) == 1 and ft["spot_skew_90_110"].iloc[0] > 0 and ft["ratio_se"].iloc[0] > 0
    assert ft["fwd_skew_90_110"].iloc[0] > 0
    assert "recalibrated: no" in rep.summary()
    # the stage-3 assertion on a model that is not a fit of these targets
    ck = rep.check
    assert len(ck) == 4 and not rep.within_tolerance and "FAIL" in rep.summary()
    svc = ck[ck["quantity"] == "SpotVolCovar"]
    assert svc["within"].all() and (svc["engine_bias"].abs() < 0.10).all()
    assert (svc["first_order_miss"] > 1.0).all() and (svc["note"] == "").all()
    vv = ck[ck["quantity"] == "VolVar"]
    assert not vv["within"].any() and (vv["note"] == NO_FIRST_ORDER_NOTE).all()
    assert np.isnan(vv["engine_bias"]).all() and (vv["gap_vs_target"] > 1.0).all()
    assert "VolVar at T=0.25: simulated" in rep.check_message
    # every failing row is a fallback row here, so the header names the target-gap rule
    assert NO_FIRST_ORDER_NOTE in rep.check_message
    assert "(no first-order value: target gap):" in rep.check_message
    assert "when the fit says so): SpotVolCovar at T=0.25 +" in rep.check_message
    assert "SpotVolCovar at T=0.25: simulated" not in rep.check_message
    chk, ok, msg = breakeven_check(rep.breakeven_table, 5.0)  # a 500% tolerance passes
    assert ok and msg == "" and chk["within"].all()
    with pytest.raises(BreakEvenValidationError):
        stage3_validation(P82, inputs, tg, fit_table=table, assert_breakevens=True)
    light = dataclasses.replace(
        inputs,
        pricing_sim=dataclasses.replace(fast_sim, n_paths=20_000, chunk_size=20_000),
        ssr_pillars=(0.25,),
        breakeven_pillars=(0.25,),
        forward_starts=(),
        mixing_paths=20_000,
    )
    with pytest.raises(BreakEvenValidationError) as exc:
        fit_2f(tg, lsv.kernel.xi0, cfg, stage3=light)
    res = exc.value.result
    assert res is not None and res.stage3 is not None and res.stage3_passed is False
    assert not res.recalibrated and "FAIL" in res.summary()
    assert "(engine bias): SpotVolCovar at T=0.25: simulated" in str(exc.value)
    assert (res.stage3.check["note"] == "").all()  # the fit's table carries every first order
    it = fit_2f(
        tg, lsv.kernel.xi0, cfg, stage3=light, iterate_against_simulation=1, assert_stage3=False
    )
    assert it.iterations is not None and list(it.iterations["iteration"]) == [0, 1]
    assert {"max_gap_svc_vs_target", "max_engine_bias_svc", "calibration_seconds"} <= set(
        it.iterations.columns
    )
    assert "svc_correction" in it.table and "volvar_correction" in it.table
    assert any("iterated 1x against simulation" in n for n in it.notes) and not it.recalibrated
    with pytest.raises(ValueError):
        fit_2f(tg, lsv.kernel.xi0, cfg, iterate_against_simulation=1)


# --------------------------------------------------------------------------------------------
# slow
# --------------------------------------------------------------------------------------------


@pytest.mark.slow
def test_recovery_three_year_mixing(synthetic_3y) -> None:  # type: ignore[no-untyped-def]
    """SPEC recovery test on the three-year mixing history (seed 13, 10⁵ mixing paths per day,
    windows 250 / 250, k2 fixed at 0.28, MatMin 3M) under the **default** config (auto → soft
    all-pillar penalty at weight 10; report decision v): ν 1.798, θ 0.245, k1 5.783, ρ −0.716 /
    −0.461 — inside the owner's tolerances (ν, θ, k1 10%, ρ 0.05); weight 1 gives ρ_SX1 −0.686 (off
    0.073)."""
    hist = synthetic_3y.history
    cfg = BreakEvenFitConfig(
        pillars=tuple(map(float, hist.pillars)), k2=0.28, k1_bounds=(0.5, 20.0)
    )
    r = fit_2f_historical(hist, cfg, window_vol=250, window_ssr=250)
    print(r.summary())
    assert r.config.skew_mode == "soft"
    p = r.params
    assert abs(p.nu - P82.nu) < 0.10 * P82.nu, p
    assert abs(p.theta - P82.theta) < 0.10 * P82.theta, p
    assert abs(p.k1 - P82.k1) < 0.10 * P82.k1, p
    assert abs(p.rho_SX1 - P82.rho_SX1) < 0.05 and abs(p.rho_SX2 - P82.rho_SX2) < 0.05, p


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="the two-point configuration (eps 0.10 at 1Y / 5Y -> 3Y, MatMin 3M) on the three-year "
    "mixing history: nu 1.780 (+2.3%), theta 0.249, k1 5.801 (+8.4%) inside, but rho_SX1 -0.664 "
    "(off 0.095 > 0.05), rho_SX2 -0.499; the covariance targets at 3M-3Y leave the short-factor "
    "correlation unidentified when the short-end skew is free - hence soft is the historical "
    "default (report decision v)",
)
def test_recovery_three_year_twopoint(synthetic_3y) -> None:  # type: ignore[no-untyped-def]
    hist = synthetic_3y.history
    cfg = BreakEvenFitConfig(
        pillars=tuple(map(float, hist.pillars)),
        k2=0.28,
        k1_bounds=(0.5, 20.0),
        skew_mode="twopoint",
    )
    p = fit_2f_historical(hist, cfg, window_vol=250, window_ssr=250).params
    print("two-point:", p)
    assert abs(p.nu - P82.nu) < 0.10 * P82.nu, p
    assert abs(p.theta - P82.theta) < 0.10 * P82.theta, p
    assert abs(p.k1 - P82.k1) < 0.10 * P82.k1, p
    assert abs(p.rho_SX1 - P82.rho_SX1) < 0.05 and abs(p.rho_SX2 - P82.rho_SX2) < 0.05, p


@pytest.mark.slow
def test_real_data_end_to_end() -> None:
    """The 2022 H2 SPX history (plain SSVI snapshots): historical mode (pillars 3M–1Y, windows 100
    / 60, the default soft penalty) and marking mode on the last snapshot at both owner pairs;
    sanity only (finite parameters, statuses, the policy reading)."""
    path = ROOT / "outputs" / "m7" / "hdn_history_ssvi.csv"
    if not path.exists():
        pytest.skip("2022 H2 history not built (scripts/m7_hdn_history.py --no-essvi)")
    hist = SurfaceHistory(pd.read_csv(path))
    cfg = BreakEvenFitConfig(pillars=(0.25, 0.5, 1.0))
    r = fit_2f_historical(hist, cfg, window_vol=100, window_ssr=60)
    print(r.summary())
    assert all(np.isfinite(v) for v in dataclasses.asdict(r.params).values())
    assert r.params.k1 > r.params.k2 == 0.2 and r.status in ("interior", "binding", "infeasible")
    surface = _spx_surface()
    for ssr, eps in ((1.0, 0.10), (1.5, 0.05)):
        m = fit_2f_marking(surface, BreakEvenFitConfig(skew_eps=eps), ssr_target=ssr)
        print(m.summary())
        assert all(np.isfinite(v) for v in dataclasses.asdict(m.params).values())
        assert (m.targets.policy_check()["reading"] == "absolute").all()


def test_marking_targets_stencil_consistent_argument(spx) -> None:  # type: ignore[no-untyped-def]
    """The stencil-consistency argument ``skew_h`` of :func:`marking_targets_for` /
    :func:`fit_2f_marking` (M8b fix of 2026-09-16, for the recalibration rule's base fit): without
    it the M7 reading is unchanged (the SPX SSVI's analytic ATM skew, the ``h = 1e-3``
    curvature, no stencil flag); with ``skew_h = 0.05, h = 0.10`` step 0 reads the skew as the
    ``±0.05`` central difference and the curvature as the ``±0.10`` second difference of the
    surface's own implied vols (to round-off), the skew constraint's ``skew_fn`` reads the same
    stencil, the pillar records it and a flag says so."""
    from volsto.calibration.fit_2f import marking_targets_for
    from volsto.calibration.targets import SABR_CURVATURE_H

    surf = _spx_surface()
    cfg = BreakEvenFitConfig(pillars=(0.25, 1.0, 3.0), mat_min=0.0, skew_pillars=(1.0, 3.0))
    m7 = marking_targets_for(surf, cfg, ssr_target=1.0)
    for s in m7.sabr:
        assert s.skew == float(np.asarray(surf.atm_skew(s.T)))
        assert s.h == SABR_CURVATURE_H and s.skew_h is None
        assert not any("stencil-consistent" in f for f in s.flags)
    same = marking_targets_for(surf, cfg, ssr_target=1.0, h=SABR_CURVATURE_H, skew_h=None)
    assert np.array_equal(same.correl_target, m7.correl_target)
    assert np.array_equal(same.spot_vol_covar, m7.spot_vol_covar)
    st = marking_targets_for(surf, cfg, ssr_target=1.0, h=0.10, skew_h=0.05)
    rows = []
    for s, s7 in zip(st.sabr, m7.sabr, strict=True):
        T = float(s.T)
        v = np.asarray(surf.implied_vol_k(np.array([-0.10, 0.0, 0.10]), np.full(3, T)))
        w = np.asarray(surf.implied_vol_k(np.array([-0.05, 0.05]), np.full(2, T)))
        # to round-off: step 0 passes the reads through the 365-day quote conversion
        assert s.skew == pytest.approx(float((w[1] - w[0]) / (2 * 0.05)), rel=1e-13)
        assert s.curv == pytest.approx(float((v[2] - 2 * v[1] + v[0]) / (0.10 * 0.10)), rel=1e-13)
        assert s.atf == pytest.approx(float(v[1]), rel=1e-13) and s.atf == s7.atf
        assert s.h == 0.10 and s.skew_h == 0.05
        assert any("stencil-consistent" in f for f in s.flags)
        rows.append((T, s7.skew, s.skew, s7.curv, s.curv, s7.rho_sabr, s.rho_sabr))
    for r in rows:
        print(
            f"T {r[0]:g}: skew {r[1]:+.5f} -> {r[2]:+.5f}, curv {r[3]:+.4f} -> {r[4]:+.4f}, "
            f"Corr_SABR {r[5]:+.4f} -> {r[6]:+.4f}"
        )
    t = np.array([0.5, 2.0])
    assert st.skew_fn is not None and m7.skew_fn is not None
    fd = [
        float(np.diff(np.asarray(surf.implied_vol_k(np.array([-0.05, 0.05]), np.full(2, x))))[0])
        / (2 * 0.05)
        for x in t
    ]
    assert np.array_equal(st.skew_fn(t), np.array(fd))
    assert np.array_equal(m7.skew_fn(t), np.asarray(surf.atm_skew(t), dtype=np.float64))
    with pytest.raises(ValueError):
        marking_targets_for(surf, cfg, ssr_target=1.0, skew_h=0.0)
