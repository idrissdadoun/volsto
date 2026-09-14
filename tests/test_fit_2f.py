"""Soft-skew break-even fit of the two-factor model (SPEC §15 Part 3, M7 addendum and the owner's
M7 Part 3 redesign; ``calibration/fit_2f.py``, ``calibration/targets.py`` additions and
``calibration/stability.py``).  Tests never calibrate a leverage: cached leverages are read with
``allow_calibrate=False`` and skipped when absent; no wall-clock assertion.

Every SSR asserted here without a Monte Carlo standard error is a **first-order** value: those
checks are optimiser / formula self-consistency, not dynamics.  The dynamics checks are the
numerical SSRs (with standard errors) of the naked kernel, of the cached Table 8.2 LSV and of the
cached study leverages.  Numbers quoted: see the module docstring of ``fit_2f`` (defaults ``ν_cap
= 2.5``, ``ssr_measure="auto"``, ``volvar_target="achieved"``, scan targets 0–3).

Fast:

* the machinery: ``PillarQuad`` / affine maps against the engine to 1e-12; the exact 2-D QP
  against SLSQP on the ν polytope; ``min ν = |λ1 + λ2|/2``; the ``TargetSet`` term structures and
  ``with_ssr_target``; the term-structure quadrature (substitution power, convergence); the
  lsv-measure identities; the VolVar decomposition implying ``svc_model`` under both prefactors;
* (d) the floor message on the reference SSVI (lsv, weight 1, target 1: floor 1.408 at scan
  target 0 with the ν limit binding, refit there, ``Y`` = ``AttainableSSR.floor``); no message
  inside the band; the ceiling message; the unclamped variant; curve targets on the same scan
  definition (a 1e-6 perturbation does not move ``Y``); ``W`` formatting;
* the attainable band (floors 1.779 / 1.508 / 1.408 / 1.352 at weights 100 / 10 / 1 / 0.1 under
  lsv, bound flags asserted), tracking segments and ``k1`` basin jumps;
* (a) tracking at the loose weight 0.1: naked measure, interior solutions, every pillar within
  ``ssr_tol`` for 1.3 / 1.6 / 1.9 and the naked kernel's numerical SSR; lsv measure on the mean for
  2.0–2.6 with the bounds that bind there asserted;
* (b) / (c) the trade-off sweep (first order; the proxy only printed) and the calibrated study
  leverages of ``configs/studies/m7_skew_tradeoff`` (``scripts/m7_skew_tradeoff.py``; skipped when
  absent): mean ``|L − 1|`` 0.201 / 0.241 / 0.286 / 0.286 at target 1 and 0.196 / 0.205 / 0.209 /
  0.224 at target 1.5 (weights 100 / 10 / 1 / 0.1) within a three-sigma particle-noise slack of
  0.006, and the numerical LSV SSR of the loosest fit against its first-order value;
* the leverage proxy on the cached reference leverages and its validity flags;
* historical recovery on the one-year order-one synthetic Table 8.2 history with the **default**
  measure (auto → naked), the lsv measure reported;
* the rolling fit (7 dates, windows 120 / 60), the bound pattern and the identification flags;
* stage 3 on the cached 2F Table 8.2 LSV (lsv measure within 10% of the numerical LSV SSR);
* the V2 local-vol SSR path (machinery) and its forwarding through the trade-off helper.

Slow: the three-year mixing history with the default config (owner's tolerances), a strict xfail
recording that the lsv measure in historical mode misses them, and the real-data run.
"""

from __future__ import annotations

import dataclasses
import importlib
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.optimize import LinearConstraint, minimize

from volsto.analytics.bergomi import atmf_skew_order1
from volsto.analytics.breakeven import first_order_breakevens, kernels
from volsto.analytics.reparam import BreakEvenParams, to_breakeven
from volsto.analytics.smile_dynamics import ssr_numerical_many
from volsto.calibration.fit_2f import (
    CEILING_MESSAGE,
    FLOOR_MESSAGE,
    TRADEOFF_NOISE_SEED,
    BreakEvenFitConfig,
    Stage3Inputs,
    affine_maps,
    affine_maps_from_kernels,
    attainable_ssr,
    fit_2f,
    fit_2f_historical,
    fit_2f_marking,
    fit_first,
    fit_second,
    format_skew_weight,
    k1_profile,
    leverage_proxy,
    load_tradeoff_specs,
    local_vol_ssr_numerical,
    mean_abs_leverage_deviation,
    naked_kernel,
    pillar_quad,
    resolve_ssr_measure,
    skew_weight_tradeoff,
    stage3_validation,
    term_structure_bank,
    ts_substitution_power,
    write_tradeoff_spec,
)
from volsto.calibration.history import SurfaceHistory, synthetic_2f_history
from volsto.calibration.stability import PARAM_COLUMNS, flag_unidentified, rolling_fit
from volsto.calibration.targets import marking_targets, pillar_power_law_skew
from volsto.config import BergomiParams, SimConfig
from volsto.market.varswap import xi0_curve

fit_2f_module = importlib.import_module("volsto.calibration.fit_2f")

ROOT = Path(__file__).resolve().parents[1]
P82 = BergomiParams(1.74, 0.245, 5.35, 0.28, 0.0, -0.759, -0.487)
MARKING_PILLARS = (0.25, 0.5, 1.0, 2.0, 3.0)
TRADEOFF_WEIGHTS = (100.0, 10.0, 1.0, 0.1)
TRADEOFF_DIR = ROOT / "configs" / "studies" / "m7_skew_tradeoff"
SSR_SIM = SimConfig(n_paths=40_000, dt_max=1.0 / 100.0, chunk_size=20_000, seed=7)


def _cached(spec):  # type: ignore[no-untyped-def]
    """The cached LSV of ``spec`` (never calibrated here) or a skip."""
    from volsto.calibration.cache import CacheMissError, LeverageCache

    try:
        lsv, _ = LeverageCache(ROOT / "cache").get_or_calibrate(spec, allow_calibrate=False)
    except CacheMissError as exc:
        pytest.skip(f"leverage not in the cache (tests never calibrate): {exc}")
    return lsv


def _reference_spec(kind: str):  # type: ignore[no-untyped-def]
    from volsto.config import CalibrationSpec, load_yaml

    spec = load_yaml(ROOT / "configs" / "studies" / f"lsv_reference_{kind}.yaml", CalibrationSpec)
    return dataclasses.replace(
        spec, particle=dataclasses.replace(spec.particle, n_particles=800_000)
    )


@pytest.fixture(scope="module")
def synthetic_1y():  # type: ignore[no-untyped-def]
    return synthetic_2f_history(P82, years=1.0, seed=8, atm_source="order_one")


@pytest.fixture(scope="module")
def synthetic_3y():  # type: ignore[no-untyped-def]
    return synthetic_2f_history(P82, years=3.0, seed=13, atm_source="mixing", mixing_paths=100_000)


@pytest.fixture(scope="module")
def marking_fit(ssvi):  # type: ignore[no-untyped-def]
    """The default fitter (auto → lsv measure, skew weight 1, clamp on) at ssr_target 1 on the
    reference SSVI."""
    return fit_2f_marking(ssvi, BreakEvenFitConfig(pillars=MARKING_PILLARS), ssr_target=1.0)


@pytest.fixture(scope="module")
def tradeoff(ssvi):  # type: ignore[no-untyped-def]
    """The trade-off sweep at ssr_target 1 under both measures (no clamp)."""
    return {
        m: skew_weight_tradeoff(
            ssvi,
            BreakEvenFitConfig(pillars=MARKING_PILLARS, ssr_measure=m),
            weights=TRADEOFF_WEIGHTS,
            ssr_target=1.0,
        )
        for m in ("lsv", "naked")
    }


def test_config_validation(ssvi) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        BreakEvenFitConfig(k2=0.28)  # k1_bounds[0] = 0.3 < k2 + 0.05
    with pytest.raises(ValueError):
        BreakEvenFitConfig(weights_ssr="log")
    with pytest.raises(ValueError):
        BreakEvenFitConfig(sigma_hat_prefactor="mixed")
    with pytest.raises(ValueError):
        BreakEvenFitConfig(skew_weight=-1.0)
    with pytest.raises(ValueError):
        BreakEvenFitConfig(ssr_measure="pilv")
    with pytest.raises(ValueError):
        BreakEvenFitConfig(term_structure="linear")
    with pytest.raises(ValueError):
        BreakEvenFitConfig(ssr_tol=0.0)
    with pytest.raises(ValueError):
        BreakEvenFitConfig(omega_max=4.0)  # must exceed 2 nu_cap = 5
    cfg = BreakEvenFitConfig(k2=0.28, k1_bounds=(0.5, 20.0))
    assert cfg.k1_bounds[0] > cfg.k2 + cfg.k1_min_gap
    d = BreakEvenFitConfig()
    assert (d.skew_weight, d.ssr_measure, d.ssr_tol, d.term_structure) == (
        1.0,
        "auto",
        0.05,
        "atmf",
    )
    assert d.clamp_to_attainable and d.lv_ssr == "order1" and d.volvar_target == "achieved"
    assert d.nu_cap == d.nu_flag == 2.5
    # "auto" resolves on the targets: lsv with a surface skew term structure, naked without
    tg = marking_targets(ssvi, MARKING_PILLARS)
    assert resolve_ssr_measure(d, tg)[0] == "lsv"
    hist_like = dataclasses.replace(tg, skew_fn=None, atm_vol_fn=None)
    measure, note = resolve_ssr_measure(d, hist_like)
    assert measure == "naked" and "auto" in note
    assert resolve_ssr_measure(dataclasses.replace(d, ssr_measure="lsv"), hist_like) == ("lsv", "")
    assert [format_skew_weight(w) for w in (1000.0, 100.0, 1.0, 0.1, 0.012345)] == [
        "1000",
        "100",
        "1",
        "0.1",
        "0.0123",
    ]


def test_affine_maps_match_engine(ssvi, rng) -> None:  # type: ignore[no-untyped-def]
    """``PillarQuad`` against ``kernels`` (A, J to 1e-12), the maps against
    ``first_order_breakevens(sigma_hat=atf)`` to 1e-12, affinity in ``(λ1, λ2)``."""
    tg = marking_targets(ssvi, MARKING_PILLARS)
    xi0 = xi0_curve(ssvi, 3.0)
    ks = (4.0, 0.2)
    quads = [pillar_quad(xi0, T, n_quad=64, n_inner=32) for T in MARKING_PILLARS]
    kerns = [kernels(ks, xi0, T) for T in MARKING_PILLARS]
    m_fast = affine_maps(quads, tg.atf, *ks)
    m_ref = affine_maps_from_kernels(ks, kerns, tg.atf)
    assert np.allclose(m_fast.A, m_ref.A, rtol=0, atol=1e-12)
    assert np.allclose(m_fast.J, m_ref.J, rtol=0, atol=1e-12)
    for _ in range(3):
        lam = rng.uniform(-3.0, 1.0, size=2)
        be = BreakEvenParams(ks[0], ks[1], 4.0, 4.0, lam[0], lam[1], 0.0)
        fo = [
            first_order_breakevens(be, xi0, T, sigma_hat=float(a))
            for T, a in zip(MARKING_PILLARS, tg.atf)
        ]
        assert np.allclose(m_fast.svc(lam), [f.spot_vol_covar for f in fo], rtol=0, atol=1e-12)
        assert np.allclose(m_fast.skew(lam), [f.skew for f in fo], rtol=0, atol=1e-12)
    a, b = rng.normal(size=2), rng.normal(size=2)
    for f in (m_fast.svc, m_fast.skew):
        assert np.allclose(f(a + b), f(a) + f(b), atol=1e-12)
        assert np.allclose(f(2.5 * a), 2.5 * f(a), atol=1e-12)
        assert np.allclose(f(np.zeros(2)), 0.0)
    m_model = affine_maps(quads, tg.atf, *ks, prefactor="model")
    fo0 = [first_order_breakevens(be, xi0, T) for T in MARKING_PILLARS]
    assert np.allclose(m_model.skew(lam), [f.skew for f in fo0], atol=1e-12)


def test_exact_qp_matches_slsqp(rng) -> None:  # type: ignore[no-untyped-def]
    """The candidate enumeration of the inner 2-D QP against SLSQP on random instances: the
    ν-feasibility polytope ``|λ1| + |λ2| ≤ 2 ν_cap`` of the first minimisation (active and
    inactive) and random strips."""
    qp = fit_2f_module._qp2
    polytope = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
    n_active = 0
    for i in range(30):
        M = rng.normal(size=(5, 2))
        y = rng.normal(size=5) * (4.0 if i % 2 else 0.5)
        H = M.T @ M
        g = M.T @ y
        if i % 3:
            G = polytope
            h = np.full(4, rng.uniform(0.2, 2.0))
        else:
            js, jl = rng.normal(size=2), rng.normal(size=2)
            c_s, c_l = rng.normal(), rng.normal()
            width = rng.uniform(0.05, 1.0)
            G = np.array([js, -js, jl, -jl])
            h = np.array([c_s + width, -(c_s - width), c_l + width, -(c_l - width)])
        x, active, feas = qp(H, g, G, h)
        n_active += bool(active)
        assert feas
        assert np.all(G @ x <= h + 1e-8)

        def quad(z, H=H, g=g):  # type: ignore[no-untyped-def]
            return 0.5 * z @ H @ z - g @ z

        res = minimize(
            quad,
            np.zeros(2),
            method="SLSQP",
            constraints=[LinearConstraint(G, -np.inf, h)],
            options={"ftol": 1e-14, "maxiter": 500},
        )
        f_exact = 0.5 * x @ H @ x - g @ x
        f_slsqp = 0.5 * res.x @ H @ res.x - g @ res.x
        assert f_exact <= f_slsqp + 1e-8 * (1.0 + abs(f_slsqp)), (f_exact, f_slsqp, active)
    assert n_active >= 5


def test_nu_feasibility_polytope(rng) -> None:  # type: ignore[no-untyped-def]
    """``min ν`` over ``ω_i ≥ |λ_i|`` and ``χ`` is ``|λ1 + λ2|/2`` (``Cov(W_S, ω1 X1 + ω2 X2) =
    λ1 + λ2 ≤ 2ν``), attained at ``ω_i = |λ_i|``: no random admissible ``(ω1, ω2, χ)`` goes
    below it; for same-sign loadings (every fit measured) it equals ``(|λ1| + |λ2|)/2``, the
    polytope row of the first minimisation, which is conservative for opposite signs (it
    excludes cancelling loadings)."""
    for i in range(6):
        lam = rng.uniform(-3.0, 3.0, size=2)
        if i < 3:
            lam = -np.abs(lam)
        bound = 0.5 * abs(float(np.sum(lam)))
        at = BreakEvenParams(2.0, 0.2, abs(lam[0]), abs(lam[1]), lam[0], lam[1], 0.0).nu
        assert at == pytest.approx(bound, rel=1e-9)
        if i < 3:
            assert bound == pytest.approx(0.5 * float(np.sum(np.abs(lam))))
        for _ in range(400):
            om = np.abs(lam) * np.exp(rng.uniform(0.0, 2.0, size=2))
            chi = rng.uniform(-1.0, 1.0)
            nu = BreakEvenParams(2.0, 0.2, om[0], om[1], lam[0], lam[1], chi).nu
            assert nu >= bound * (1 - 1e-9)


def test_target_term_structures_and_with_ssr(ssvi) -> None:  # type: ignore[no-untyped-def]
    """``TargetSet`` additions: ``correl_target = ρ_SABR`` (marking) and the implied value in
    historical mode, the market skew on ``(0, T]`` from the surface (marking) and the power-law
    interpolation through the pillars (exponents clipped), the ATMF curve reproducing the
    surface's ATMF total variance, ``with_ssr_target`` against a fresh ``marking_targets``."""
    tg = marking_targets(ssvi, MARKING_PILLARS)
    assert tg.term_structure_source == "surface"
    assert np.allclose(tg.correl_target, [s.rho_sabr for s in tg.sabr])
    t = np.array([0.01, 0.1, 0.25, 0.7, 2.5])
    assert np.allclose(tg.market_skew(t), ssvi.atm_skew(t), rtol=1e-14)
    curve = tg.atmf_curve(3.0)
    for T in (0.25, 1.0, 3.0):
        assert curve.total_variance(T) == pytest.approx(float(ssvi.atm_vol(T)) ** 2 * T, rel=1e-9)
    for ssr in (0.7, {0.25: 1.5, 3.0: 0.8}):
        a = tg.with_ssr_target(ssr)
        b = marking_targets(ssvi, MARKING_PILLARS, ssr_target=ssr)
        for f in ("ssr_target", "spot_vol_covar", "vol_var", "vovol", "skew_target", "atf"):
            assert np.allclose(getattr(a, f), getattr(b, f), rtol=1e-13), f
    # the power law through the pillars
    p = np.array([1 / 12, 0.25, 1.0])
    s = np.array([-0.8, -0.5, -0.25])
    assert np.allclose(pillar_power_law_skew(p, s, p), s)
    g = -np.log(0.5 / 0.8) / np.log(3.0)
    assert pillar_power_law_skew(p, s, np.array([1 / 48]))[0] == pytest.approx(-0.8 * 4.0**g)
    g_hi = -np.log(0.25 / 0.5) / np.log(4.0)
    assert pillar_power_law_skew(p, s, np.array([4.0]))[0] == pytest.approx(-0.25 * 4.0**-g_hi)
    steep = pillar_power_law_skew(p, np.array([-2.0, -0.5, -0.25]), np.array([1 / 48]))
    assert steep[0] == pytest.approx(-2.0 * 4.0**0.75)  # exponent clipped to 0.75
    with pytest.raises(ValueError):
        pillar_power_law_skew(p, np.array([-0.8, 0.5, -0.2]), p)


def test_term_structure_quadrature(ssvi) -> None:  # type: ignore[no-untyped-def]
    """The ``t = T u^p`` substitution (``ts_substitution_power``): ``p = 2`` on the reference SSVI
    (short-end skew exponent 0.50), ``p = 4`` on the historical power-law extension at the
    exponent clip 0.75; at the default ``n_ts = 64`` the market integral ``I^mkt/S`` (``R^LV(Mkt)
    − 1``) is within 1e-3 of a 1024-node reference in both cases, while ``p = 2`` on the steep
    power law misses by more than 1e-2 (the singular integrand)."""
    tg = marking_targets(ssvi, MARKING_PILLARS)
    xi0 = xi0_curve(ssvi, 4.0)
    T = np.array(MARKING_PILLARS)
    steep = dataclasses.replace(
        tg,
        skew_fn=None,
        atm_vol_fn=None,
        skew_target=np.array([-1.8, -0.9, -0.55, -0.35, -0.28]),
    )
    assert ts_substitution_power(tg) == pytest.approx(2.0, abs=1e-3)
    assert ts_substitution_power(steep) == pytest.approx(4.0, abs=1e-9)

    def rlv(targets, n, power=None):  # type: ignore[no-untyped-def]
        b = term_structure_bank(
            targets, T, xi0, kind="atmf", n_ts=n, n_quad=32, n_inner=24, power=power
        )
        return b.I_market / targets.skew_target

    for targets in (tg, steep):
        err = np.max(np.abs(rlv(targets, 64) - rlv(targets, 1024)))
        print("R^LV(Mkt) quadrature error at n_ts = 64:", err)
        assert err < 1e-3
    assert np.max(np.abs(rlv(steep, 64, 2.0) - rlv(steep, 1024, 4.0))) > 1e-2


def test_lsv_measure_identities(ssvi, rng) -> None:  # type: ignore[no-untyped-def]
    """The eq. 12.52 covariance form (module docstring): the term-structure bank's ``J(t)``
    against ``PillarQuad``; both measures affine in ``λ``; ``λ = 0`` gives ``σ_0 S R^LV(Mkt)``;
    the lsv measure reduces to the naked one when the market skew term structure is the
    kernel's own order-one skew; the order-one ``R^LV(Mkt)`` of the reference SSVI (3.007 /
    3.014 / 3.027 at 3M / 6M / 1Y) and the lsv SSR of the Table 8.2 kernel (2.641 / 2.352 /
    2.082) within 0.3%; ``flat`` and ``atmf`` factors differ by less than 2%."""
    xi0 = xi0_curve(ssvi, 4.0)
    tg = marking_targets(ssvi, MARKING_PILLARS)
    cfg = BreakEvenFitConfig(pillars=MARKING_PILLARS, k2=0.28, k1_bounds=(0.5, 20.0))
    prob, notes = fit_2f_module._first_problem(tg, cfg, xi0)
    assert prob.measure == "lsv" and any("auto" in n for n in notes)
    bank = prob.bank
    Jn = bank.J_nodes(5.35)
    for i, j in ((0, 3), (2, 40), (4, 63)):
        ref = pillar_quad(xi0, float(bank.t[i, j]), n_quad=64, n_inner=32).J(5.35)
        assert Jn[i, j] == pytest.approx(ref, rel=2e-3)  # 32 x 24 against 64 x 32 nodes
    mm = prob.maps(5.35)
    be = to_breakeven(P82)
    lam82 = np.array([be.lambda1, be.lambda2])
    a, b = rng.normal(size=2), rng.normal(size=2)
    for meas in ("lsv", "naked"):
        f0 = mm.svc(np.zeros(2), meas)
        assert np.allclose(
            mm.svc(a + b, meas) - f0, (mm.svc(a, meas) - f0) + (mm.svc(b, meas) - f0)
        )
        assert np.allclose(mm.svc(2.5 * a, meas) - f0, 2.5 * (mm.svc(a, meas) - f0))
    assert np.allclose(mm.svc(np.zeros(2), "lsv"), tg.sigma_0 * tg.skew_target * prob.r_lv_market)
    assert np.allclose(prob.r_lv_market[:3], [3.007, 3.014, 3.027], rtol=1e-3)
    assert np.allclose(mm.ssr(lam82, "lsv")[:3], [2.641, 2.352, 2.082], rtol=3e-3)
    flat = BreakEvenFitConfig(**{**dataclasses.asdict(cfg), "term_structure": "flat"})
    pf, _ = fit_2f_module._first_problem(tg, flat, xi0)
    assert np.allclose(pf.maps(5.35).ssr(lam82, "lsv"), mm.ssr(lam82, "lsv"), rtol=0.02)
    # reduction: the market skew term structure replaced by the kernel's own order-one skew
    own = dataclasses.replace(
        tg,
        skew_target=np.array([atmf_skew_order1(P82, xi0, T) for T in MARKING_PILLARS]),
        skew_fn=lambda t: np.array([atmf_skew_order1(P82, xi0, float(x)) for x in t]),
    )
    po, _ = fit_2f_module._first_problem(own, cfg, xi0)
    mo = po.maps(5.35)
    assert np.allclose(mo.skew_naked(lam82), own.skew_target, rtol=2e-3)
    assert np.allclose(mo.svc(lam82, "lsv"), mo.svc(lam82, "naked"), rtol=2e-3)


def test_volvar_decomposition_consistent(ssvi, rng) -> None:  # type: ignore[no-untyped-def]
    """The second fit's VolVar decomposition implies the first fit's covariance: ``c (SensiSpot
    + Σ ρ_Si SensiX_i) = svc_model`` under both prefactors (``c = σ_0/sqrt(ξ₀(0))`` = 0.85 under
    ``"model"``) and both measures, with ``SensiSpot = (svc_lsv − svc_naked)/c``; and a fitted
    second minimisation reports ``correl_implied`` from that covariance."""
    xi0 = xi0_curve(ssvi, 3.0)
    tg = marking_targets(ssvi, MARKING_PILLARS)
    for pref in ("market", "model"):
        cfg = BreakEvenFitConfig(pillars=MARKING_PILLARS, sigma_hat_prefactor=pref)
        prob, _ = fit_2f_module._first_problem(tg, cfg, xi0)
        mm = prob.maps(6.0)
        if pref == "model":
            assert 0.8 < mm.level_ratio < 0.9
        for _ in range(3):
            lam = rng.uniform(-3.0, 0.0, size=2)
            spot = mm.sensi_spot_lv(lam)
            assert np.allclose(mm.implied_covariance(lam, spot), mm.svc_lsv(lam), rtol=1e-12)
            assert np.allclose(mm.implied_covariance(lam, None), mm.svc_naked(lam), rtol=1e-12)
        first = fit_first(tg, cfg, xi0)
        second = fit_second(tg, cfg, first)
        spot = first.maps.sensi_spot_lv(first.lam)
        assert np.allclose(second.table["sensi_spot_lv"], spot)
        assert np.allclose(
            first.maps.implied_covariance(first.lam, spot), first.table["svc_model"], rtol=1e-12
        )


def test_marking_fit_floor_message(marking_fit, ssvi) -> None:  # type: ignore[no-untyped-def]
    """(d) The default fitter at ``ssr_target = 1`` on the reference SSVI: the attainable floor
    at skew weight 1 is the lowest first-order mean SSR over the scan (1.408, at scan target 0,
    the ν limit binding — asserted, so a change of the cap is caught); the request lies below
    it, so the fit refits at that scan target and ``message`` is the owner's text exactly with
    ``Y`` = ``AttainableSSR.floor``; the clamp moves the achieved SSR towards the request, never
    away (unclamped 1.538); the VolVar target is built at ``Y``; the tracking detail is always
    present; the YAML round trips; nothing recalibrated."""
    r = marking_fit
    att = r.attainable
    assert att is not None and r.config.ssr_measure == "lsv"
    y = att.floor
    assert 1.3 < y < 1.5 and att.floor_target == 0.0
    assert any("nu limit binding" in b for b in att.floor_bounds), att.floor_bounds
    assert r.message == FLOOR_MESSAGE.format(x=1.0, y=y, w="1")
    pattern = (
        r"^ssr_target=1\.000 below attainable floor (\d\.\d{3}) at skew_weight=1; fitted at "
        r"(\d\.\d{3})\. Lower the skew weight to reach lower SSR \(naked skew will diverge "
        r"further from market, leverage will do more\)\.$"
    )
    m = re.match(pattern, r.message or "")
    assert m and m.group(1) == m.group(2) == f"{y:.3f}", r.message
    # the message fires only below the floor, and Y is what the refit achieves
    assert y > 1.0 and abs(r.ssr_achieved_mean - y) < 1e-9
    unclamped = float(
        att.scan.loc[att.scan["kind"].str.contains("request"), "ssr_achieved_mean"].iloc[0]
    )
    print("unclamped", unclamped, "floor", y)
    assert abs(r.ssr_achieved_mean - 1.0) <= abs(unclamped - 1.0)
    assert np.allclose(r.table["ssr_target"], att.floor_target)
    assert np.allclose(r.table["ssr_fitted"], y) and np.allclose(r.targets.ssr_target, y)
    assert np.allclose(r.ssr_requested, 1.0) and np.allclose(r.table["ssr_requested"], 1.0)
    assert any(d.startswith("first-order mean lsv-measure SSR") for d in r.message_details)
    assert any("clamped" in d for d in r.message_details)
    cols = {
        "svc_model",
        "ssr_achieved",
        "ssr_achieved_naked_vs_market",
        "ssr_naked_own",
        "ssr_achieved_lsv",
        "skew_naked",
        "skew_market",
        "skew_gap_rel",
        "correl_target",
        "correl_implied",
        "volvol_target",
        "volvol_model",
    }
    assert cols <= set(r.table.columns)
    assert np.allclose(r.table["correl_target"], [s.rho_sabr for s in r.targets.sabr])
    assert np.allclose(r.table["ssr_achieved"], r.table["ssr_achieved_lsv"])
    assert (r.targets.policy_check()["reading"] == "absolute").all()
    assert r.risk_regime == "sticky_strike" and not r.recalibrated and r.stage3 is None
    assert r.leverage_proxy is not None and 0.05 < r.leverage_proxy.mean_abs_l_minus_1 < 0.3
    assert r.params.k2 == 0.2 and r.params.k1 > r.params.k2
    # marking targets carry no sampling error: no standard errors, with the reason in the notes
    assert np.isnan(r.first.k1_se) and np.isnan(r.first.lambda1_se)
    assert any("no standard errors" in n for n in r.first.notes)
    doc = yaml.safe_load(r.config_yaml)
    assert BergomiParams(**doc["model"]) == r.params
    assert BreakEvenParams(**doc["breakeven"]).to_book() == r.params
    prov = doc["provenance"]
    assert prov["message"] == r.message and prov["ssr_measure"] == "lsv"
    assert prov["skew_weight"] == 1.0 and prov["ssr_requested"] == [1.0] * 5
    assert prov["attainable"]["floor"] == pytest.approx(y)
    s = r.summary()
    assert "MESSAGE: ssr_target=1.000" in s and "recalibrated: no" in s and "wall clock" in s
    first_targets = r.targets.with_ssr_target(r.table["ssr_target"].to_numpy())
    prof = k1_profile(first_targets, r.config, r.xi0, k1s=[2.0, r.first.k1, 8.0])
    assert prof["objective"].iloc[1] <= prof["objective"].min() + 1e-12
    print(r.summary())


def test_floor_ceiling_and_clamp_variants(ssvi) -> None:  # type: ignore[no-untyped-def]
    """(d) continued: no message inside the band (the tracking detail says how far the achieved
    SSR sits from the request); the ceiling message with ``Y`` = ``AttainableSSR.ceiling``
    (weight 100, target 2.2); ``clamp_to_attainable=False`` keeps the target and returns the
    unclamped variant; curve targets use the same scan (the request's shape scaled), so a
    constant target and the same target perturbed by 1e-6 give the same ``Y``; a curve below
    the floor refits at the scaled shape and ``Y`` is the reported floor."""
    base = BreakEvenFitConfig(pillars=MARKING_PILLARS)
    inside = fit_2f_marking(ssvi, base, ssr_target=1.8, proxy=False)
    att = inside.attainable
    assert inside.message is None and att is not None and att.floor < 1.8 < att.ceiling
    assert np.allclose(inside.table["ssr_target"], 1.8)
    assert inside.message_details[0].startswith("first-order mean lsv-measure SSR")
    tight = dataclasses.replace(base, skew_weight=100.0)
    above = fit_2f_marking(ssvi, tight, ssr_target=2.2, proxy=False)
    a = above.attainable
    assert a is not None
    assert above.message == CEILING_MESSAGE.format(x=2.2, y=a.ceiling, w="100")
    assert abs(above.ssr_achieved_mean - a.ceiling) < 1e-9
    raw = fit_2f_marking(
        ssvi, dataclasses.replace(base, clamp_to_attainable=False), ssr_target=1.0, proxy=False
    )
    assert raw.message is not None and "not clamped" in raw.message
    assert np.allclose(raw.table["ssr_target"], 1.0) and raw.ssr_achieved_mean > 1.0
    assert f"achieved {raw.ssr_achieved_mean:.3f}" in raw.message
    const = fit_2f_marking(ssvi, base, ssr_target=0.9, proxy=False)
    near = fit_2f_marking(ssvi, base, ssr_target={0.25: 0.9, 3.0: 0.9 + 1e-6}, proxy=False)
    assert const.attainable is not None and near.attainable is not None
    assert near.attainable.floor == pytest.approx(const.attainable.floor, abs=1e-4)
    assert near.message is not None and const.message is not None
    assert near.message[:60] == const.message[:60]
    curve = fit_2f_marking(ssvi, base, ssr_target={0.25: 0.9, 3.0: 1.1}, proxy=False)
    ca = curve.attainable
    assert ca is not None and curve.message is not None
    x = float(np.mean(curve.ssr_requested))
    assert curve.message == FLOOR_MESSAGE.format(x=x, y=ca.floor, w="1")
    assert np.allclose(ca.shape, curve.ssr_requested / x)
    assert np.allclose(curve.table["ssr_target"], ca.floor_target * ca.shape)
    assert abs(curve.ssr_achieved_mean - ca.floor) < 1e-9
    assert any(f"floor {ca.floor:.3f}" in d for d in curve.message_details)


def test_attainable_ssr_band(ssvi) -> None:  # type: ignore[no-untyped-def]
    """``attainable_ssr`` on the reference SSVI: the floor falls as the skew weight loosens (lsv
    1.779 / 1.508 / 1.408 / 1.352 at 100 / 10 / 1 / 0.1, every one with the ν limit binding —
    asserted), the naked band at weight 100 [1.379, 1.457] is interior and saturated (no bound,
    no scan edge); the lsv band at weight 1 reports a ``k1`` basin jump near target 2.9 and two
    mean tracking segments; strict (every pillar) segments lie inside the mean ones; a surface
    argument equals the target-set route."""
    xi0 = xi0_curve(ssvi, 3.0)
    tg = marking_targets(ssvi, MARKING_PILLARS)
    floors = []
    for w in TRADEOFF_WEIGHTS:
        a = attainable_ssr(tg, BreakEvenFitConfig(pillars=MARKING_PILLARS, skew_weight=w), xi0)
        floors.append(a.floor)
        assert a.ssr_measure == "lsv"
        assert a.floor == pytest.approx(a.scan["ssr_achieved_mean"].min())
        assert a.ceiling == pytest.approx(a.scan["ssr_achieved_mean"].max())
        assert any("nu limit" in b for b in a.floor_bounds), (w, a.floor_bounds)
        for lo, hi in a.strict_tracking_segments:
            assert any(l0 <= lo and hi <= h0 for l0, h0 in a.tracking_segments)
        assert {"first_tracking_target", "ssr_min_achieved"} <= set(a.pillar_band.columns)
        assert "recalibrated: no" in a.summary()
        print(a.summary().splitlines()[0])
    print("lsv floors:", np.round(floors, 3))
    assert np.all(np.diff(floors) < 0), floors
    assert 1.7 < floors[0] < 1.85 and 1.3 < floors[-1] < 1.4
    w1 = attainable_ssr(tg, BreakEvenFitConfig(pillars=MARKING_PILLARS), xi0)
    assert len(w1.basin_jumps) >= 1 and len(w1.tracking_segments) >= 2
    via_surface = attainable_ssr(ssvi, BreakEvenFitConfig(pillars=MARKING_PILLARS))
    assert via_surface.floor == pytest.approx(w1.floor, abs=1e-9)
    nk = attainable_ssr(
        tg, BreakEvenFitConfig(pillars=MARKING_PILLARS, ssr_measure="naked", skew_weight=100), xi0
    )
    assert 1.3 < nk.floor < nk.ceiling < 1.5
    assert not (nk.floor_bounds or nk.ceiling_bounds or nk.floor_at_scan_edge)
    assert not nk.ceiling_at_scan_edge


def test_ssr_tracking_loose_weight(ssvi) -> None:  # type: ignore[no-untyped-def]
    """(a) At the loose skew weight 0.1.  Naked measure: 1.3 / 1.6 / 1.9 give 1.307 / 1.593 /
    1.880 with **every pillar** within ``ssr_tol`` and interior solutions (``k1`` off its bounds,
    ν limit not binding; asserted).  These are first-order values — optimiser self-consistency;
    the dynamics check is the naked kernel's numerical SSR at 1.6 against its own first-order
    SSR (4e4 paths, within 10%).  Lsv measure with ``ν_cap = 2.5``: the mid band 1.3–1.9 is not
    tracked (the achieved SSR saturates near the floor 1.35), the mean tracks within ``ssr_tol``
    over 2.0–2.6 with the worst pillar off by up to 0.22 and the bounds that bind asserted (ν
    limit at 2.0, ``k1`` at 20 at 2.3 / 2.6)."""
    xi0 = xi0_curve(ssvi, 3.0)
    tg = marking_targets(ssvi, MARKING_PILLARS)
    cfg = BreakEvenFitConfig(pillars=MARKING_PILLARS, skew_weight=0.1, ssr_measure="naked")
    att = attainable_ssr(tg, cfg, xi0)
    assert any(lo <= 1.3 and hi >= 1.9 for lo, hi in att.strict_tracking_segments), att.summary()
    for r in (1.3, 1.6, 1.9):
        f = fit_first(tg.with_ssr_target(r), cfg, xi0)
        per = f.table["ssr_achieved"].to_numpy()
        print("naked", r, round(f.ssr_achieved_mean, 4), np.round(per, 3), f.k1, f.nu_min)
        assert abs(f.ssr_achieved_mean - r) <= cfg.ssr_tol
        assert np.max(np.abs(per - r)) <= cfg.ssr_tol
        assert not f.k1_at_bound and not f.nu_limit_binding
    lsv = dataclasses.replace(cfg, ssr_measure="lsv")
    assert abs(fit_first(tg.with_ssr_target(1.6), lsv, xi0).ssr_achieved_mean - 1.6) > 0.1
    flags = {2.0: (False, True), 2.3: (True, False), 2.6: (True, False)}
    for r, (k1_bound, nu_bind) in flags.items():
        f = fit_first(tg.with_ssr_target(r), lsv, xi0)
        per = f.table["ssr_achieved"].to_numpy()
        print("lsv", r, round(f.ssr_achieved_mean, 4), np.round(per, 3), f.k1, f.nu_min)
        assert abs(f.ssr_achieved_mean - r) <= lsv.ssr_tol
        assert np.max(np.abs(per - r)) < 0.25
        assert (f.k1_at_bound, f.nu_limit_binding) == (k1_bound, nu_bind), (r, f.k1, f.nu_min)
    res = fit_2f(tg.with_ssr_target(1.6), xi0, cfg)
    rows = ssr_numerical_many(
        naked_kernel(res, ssvi.forward_curve), [0.25, 1.0], eps=0.05, sim=SSR_SIM
    )
    own = res.table.set_index("T")["ssr_naked_own"]
    for x in rows:
        print("naked own", x.T, own[x.T], x.R, x.R_stderr)
        assert abs(own[x.T] / x.R - 1.0) < 0.10 and x.R_stderr < 0.05


def test_tight_weight_near_pure_sv(tradeoff) -> None:  # type: ignore[no-untyped-def]
    """(b) At skew weight 100 the naked skew stays within 5% of the market skew on average under
    both measures (first order); the lsv fit has the ν limit binding, the naked one is interior
    (asserted).  ``|L − 1|`` is not asserted on the first-order proxy (it does not track the
    calibrated leverage outside ν ≤ 1.74, module docstring): the proxy is printed, and the
    calibrated study leverages are checked in :func:`test_tradeoff_study_cached_leverages`."""
    for meas, df in tradeoff.items():
        tight = df.loc[df["skew_weight"] == 100.0].iloc[0]
        print(meas, tight[["mean_skew_gap", "mean_abs_L_minus_1_proxy", "proxy_valid"]].to_dict())
        assert tight["mean_skew_gap"] <= 0.05, (meas, df)
        assert tight["nu_limit_binding"] == (meas == "lsv") and not tight["k1_at_bound"]


def test_tradeoff_sweep_monotone(tradeoff) -> None:  # type: ignore[no-untyped-def]
    """(c) The trade-off sweep at ``ssr_target = 1`` over skew weights 100 / 10 / 1 / 0.1 (first
    order): looser weight → lower attainable floor (strictly), non-increasing achieved SSR and
    non-decreasing naked-skew gap under both measures; strictly monotone under the naked measure
    (1.405 / 1.334 / 1.128 / 1.020), with ties under lsv where the ν limit binds (weights 1 and
    0.1 give the same fit).  The request lies below the lsv floor at every weight, so every lsv
    row carries the unclamped message.  Wall clock and ``recalibrated`` per row."""
    for meas, df in tradeoff.items():
        print(meas)
        print(df.drop(columns=["message"]).round(4).T.to_string())
        assert list(df["skew_weight"]) == list(TRADEOFF_WEIGHTS)
        assert np.all(np.diff(df["ssr_floor"]) < 0), df["ssr_floor"]
        assert np.all(np.diff(df["ssr_achieved"]) <= 1e-6), df["ssr_achieved"]
        assert np.all(np.diff(df["mean_skew_gap"]) >= -1e-6), df["mean_skew_gap"]
        assert not df["recalibrated"].any() and (df["wall_seconds"] > 0).all()
        assert df["mean_abs_L_minus_1"].isna().all()
    naked = tradeoff["naked"]
    assert np.all(np.diff(naked["ssr_achieved"]) < 0) and np.all(
        np.diff(naked["mean_skew_gap"]) > 0
    )
    lsv = tradeoff["lsv"]
    assert lsv["nu_limit_binding"].all()
    assert lsv["message"].map(lambda m: isinstance(m, str) and "not clamped" in m).all()


def test_tradeoff_spec_round_trip(marking_fit, tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The study YAML (``spec`` + ``fit`` sections) written by :func:`write_tradeoff_spec` loads
    back: the calibration spec carries the fitted model and the particle count, the fit config
    and break-even parameters round trip."""
    base = _reference_spec("2f")
    p = write_tradeoff_spec(
        marking_fit, base, tmp_path / "w1.yaml", n_particles=200_000, ssr_target=1.0
    )
    (entry,) = load_tradeoff_specs(tmp_path)
    assert entry.path == p and entry.spec.model == marking_fit.params
    assert entry.spec.particle.n_particles == 200_000 and entry.spec.surface == base.surface
    assert entry.config == marking_fit.config and entry.skew_weight == 1.0
    assert entry.ssr_target == 1.0 and entry.breakeven == marking_fit.breakeven
    assert load_tradeoff_specs(tmp_path / "absent") == []


def test_tradeoff_study_cached_leverages() -> None:
    """(b) / (c) on calibrated leverages: ``scripts/m7_skew_tradeoff.py`` writes the fitted
    parameter sets to ``configs/studies/m7_skew_tradeoff/*.yaml`` and calibrates them into the
    cache, plus the tightest and loosest weight of each target at the second particle seed
    ``TRADEOFF_NOISE_SEED``; this test reads them with ``allow_calibrate=False`` (skip when
    absent), re-runs each fit and checks it reproduces the stored break-even parameters, then per
    ``ssr_target``: the actual mean ``|L − 1|`` does not fall as the skew weight loosens beyond
    a slack of three particle-noise standard deviations (from the two-seed differences) and is
    larger at the loosest than at the tightest weight (the numerical SSR check is
    :func:`test_tradeoff_study_numerical_ssr`).  Study values: target 1 mean ``|L − 1|`` 0.201 / 0.241 / 0.286 / 0.286, target 1.5 0.196 / 0.205
    / 0.209 / 0.224 at weights 100 / 10 / 1 / 0.1, two-seed differences 0.0015–0.0029."""
    from volsto.calibration.cache import build_market

    entries = load_tradeoff_specs(TRADEOFF_DIR)
    if len(entries) < 2:
        pytest.skip(f"no skew trade-off study specs in {TRADEOFF_DIR} (study script not run)")
    by_target: dict[float, list[tuple[float, float, object]]] = {}
    markets = {(repr(e.spec.market), repr(e.spec.surface)) for e in entries}
    assert len(markets) == 1, "the study specs share one market and surface"
    _, surface, _ = build_market(entries[0].spec)
    for e in entries:
        lsv = _cached(e.spec)
        ap = e.fit.get("anchor_power")
        refit = fit_2f_marking(
            surface,
            e.config,
            ssr_target=e.ssr_target,
            anchor_power=1.0 if ap is None else float(ap),
            proxy=False,
        )
        stored = e.breakeven
        for name in ("k1", "lambda1", "lambda2", "omega1", "omega2"):
            assert getattr(refit.breakeven, name) == pytest.approx(
                getattr(stored, name), rel=1e-3, abs=1e-4
            ), (e.path, name)
        actual, _ = mean_abs_leverage_deviation(lsv.leverage, surface)
        by_target.setdefault(e.ssr_target, []).append((e.skew_weight, actual, e))
    for target, rows in sorted(by_target.items()):
        rows.sort(key=lambda x: -x[0])  # skew-tight first
        diffs = []
        for _, base_value, e in (rows[0], rows[-1]):
            spec2 = dataclasses.replace(
                e.spec, particle=dataclasses.replace(e.spec.particle, seed=TRADEOFF_NOISE_SEED)
            )
            lsv2 = _cached(spec2)
            diffs.append(abs(mean_abs_leverage_deviation(lsv2.leverage, surface)[0] - base_value))
        slack = 3.0 * max(diffs) / np.sqrt(2.0)
        vals = np.array([v for _, v, _ in rows])
        print(f"target {target:g}: mean |L - 1| by weight", [(w, round(v, 4)) for w, v, _ in rows])
        print(f"  particle-noise slack {slack:.4f} (two-seed differences {np.round(diffs, 4)})")
        assert np.all(np.diff(vals) > -slack), (rows, slack)
        assert vals[-1] > vals[0]


def test_tradeoff_study_numerical_ssr() -> None:
    """The numerical LSV SSR (4e4 paths, 3M and 1Y) of the loosest-weight study leverage at
    ``ssr_target = 1`` (read from the cache) against its stored first-order lsv SSR: within 25%
    (the study measured −1% at 3M and −19% at 1Y: the first-order lsv SSR under-states the LSV,
    module docstring)."""
    entries = [e for e in load_tradeoff_specs(TRADEOFF_DIR) if e.ssr_target == 1.0]
    if not entries:
        pytest.skip(f"no skew trade-off study specs in {TRADEOFF_DIR} (study script not run)")
    loose = min(entries, key=lambda e: e.skew_weight)
    lsv = _cached(loose.spec)
    num = ssr_numerical_many(lsv, [0.25, 1.0], eps=0.05, sim=SSR_SIM)
    pillars = list(loose.fit["pillars"])
    for x in num:
        pred = float(loose.fit["ssr_achieved_lsv"][pillars.index(x.T)])
        print(f"T={x.T:g}: numerical LSV SSR {x.R:.3f} +- {x.R_stderr:.3f}, lsv {pred:.3f}")
        assert abs(pred / x.R - 1.0) < 0.25 and x.R_stderr < 0.05


def test_leverage_proxy_on_cached_models() -> None:
    """The convexity-adjusted leverage proxy against the cached reference leverages (8e5
    particles, read from the cache): 1F ω = 1 / 2 / 3 and 2F Table 8.2 actual mean ``|L − 1|``
    0.240 / 0.151 / 0.110 / 0.112, proxy 0.256 / 0.170 / 0.140 / 0.132; asserted within 35% and
    ranking the clearly separated levels (ω = 1 > ω = 2 > {ω = 3, 2F}); flagged invalid above a
    skew ratio of 1.2 and above ν = 2.5 (measured −53% / +82% / +91% there)."""
    from volsto.calibration.cache import build_market
    from volsto.studies.m4 import TWO_FACTOR_NAME, one_factor_variants

    s1 = _reference_spec("1f")
    specs = dict(one_factor_variants(s1, (1.0, 2.0, 3.0)))
    specs[TWO_FACTOR_NAME] = _reference_spec("2f")
    _, surface, _ = build_market(s1)
    actual, proxy = {}, {}
    for name, spec in specs.items():
        lsv = _cached(spec)
        actual[name], _ = mean_abs_leverage_deviation(lsv.leverage, surface)
        px = leverage_proxy(lsv.kernel.params, surface, lsv.kernel.xi0)
        proxy[name] = px.mean_abs_l_minus_1
        assert px.valid and len(px.table) > 1000 and px.buckets["mean_abs_L_minus_1"].notna().all()
        print(name, round(actual[name], 4), round(proxy[name], 4))
        assert abs(proxy[name] / actual[name] - 1.0) < 0.35
    a = list(actual.values())
    p = list(proxy.values())
    assert a[0] > a[1] > max(a[2], a[3]) and p[0] > p[1] > max(p[2], p[3])
    xi0 = xi0_curve(surface, 4.0)
    ratio = leverage_proxy(P82, surface, xi0, skew_ratio=1.3)
    assert not ratio.valid and any("skew ratio" in n for n in ratio.notes)
    steep = leverage_proxy(dataclasses.replace(P82, nu=3.0), surface, xi0, skew_ratio=1.0)
    assert not steep.valid and any("validated range" in n for n in steep.notes)


def test_historical_recovery_fast(synthetic_1y) -> None:  # type: ignore[no-untyped-def]
    """Recovery on the one-year order-one synthetic history, k2 fixed at the true 0.28, the
    **default** skew weight and measure (auto → naked in historical mode): ν 1.656, θ 0.240, k1
    5.34, ρ_SX1 −0.814, ρ_SX2 −0.518 (true 1.74, 0.245, 5.35, −0.759, −0.487) within 15% / 15% /
    20% / 0.2 / 0.2; ``SpotVolCovar`` within 2% of its target at every pillar (measured 1.3%
    worst); standard errors from the target noise (sandwich); the lsv measure (explicit) and k2
    = 0.2 reported with looser assertions."""
    hist = synthetic_1y.history
    cfg = BreakEvenFitConfig(
        pillars=tuple(map(float, hist.pillars)), k2=0.28, k1_bounds=(0.5, 20.0)
    )
    r = fit_2f_historical(hist, cfg, window_vol=200, window_ssr=200)
    p = r.params
    print(r.summary())
    assert r.config.ssr_measure == "naked" and any("auto" in n for n in r.notes)
    assert r.targets.mode == "historical" and r.pricing_date == hist.dates[-1]
    assert abs(p.nu - P82.nu) < 0.15 * P82.nu, p
    assert abs(p.theta - P82.theta) < 0.15 * P82.theta, p
    assert abs(p.k1 - P82.k1) < 0.20 * P82.k1, p
    assert abs(p.rho_SX1 - P82.rho_SX1) < 0.2 and abs(p.rho_SX2 - P82.rho_SX2) < 0.2, p
    assert p.k2 == 0.28 and r.first.feasible and not r.first.active and r.message is None
    assert np.isfinite(r.first.k1_se) and 0 < r.first.k1_se < 2.0
    assert np.all(np.isfinite([r.first.lambda1_se, r.first.lambda2_se]))
    assert np.all(r.targets.spot_vol_covar_se > 0) and np.all(r.targets.vol_var_se > 0)
    ratio = r.table["svc_model"] / r.table["svc_target"]
    assert np.all(np.abs(ratio - 1.0) < 0.02), ratio
    assert r.mean_skew_gap < 0.05
    lsv = fit_2f_historical(
        hist, dataclasses.replace(cfg, ssr_measure="lsv"), window_vol=200, window_ssr=200
    )
    q = lsv.params
    print("lsv measure:", q, lsv.notes, lsv.first.notes)
    assert any("power law" in n for n in lsv.first.notes)
    assert abs(q.nu - P82.nu) < 0.15 * P82.nu and abs(q.rho_SX1 - P82.rho_SX1) < 0.2
    r2 = fit_2f_historical(
        hist,
        dataclasses.replace(cfg, k2=0.2, k1_bounds=(0.3, 20.0)),
        window_vol=200,
        window_ssr=200,
    )
    print("k2 = 0.2:", r2.params)
    assert abs(r2.params.nu - P82.nu) < 0.15 * P82.nu and abs(r2.params.k1 - P82.k1) < 0.2 * P82.k1


def test_rolling_fit_and_flags(synthetic_1y) -> None:  # type: ignore[no-untyped-def]
    """The rolling fit every 5 dates over dates 200–230 (windows 120 / 60, default measure):
    seven fits, no floor / ceiling message; ``ω`` sits on its lower bounds (``|ρ_Si| = 1``) on
    exactly dates 2 and 3, where the 60-day SSR target (about 2.55) asks for more spot/vol
    covariance than the 120-day VolVar target allows (the soft-skew fit tracks the SSR; the
    hard-skew fitter of commit adb673a pinned ``λ`` from the skew instead) — the ``ω`` / ``χ``
    standard errors are NaN there; standard errors above ``max_se`` are NaN with the raw value
    kept; ``k1`` and ``λ`` always have one."""
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
        "ssr_target",
        "ssr_achieved",
        "mean_skew_gap",
        "message",
    } <= set(frame.columns)
    assert (frame["k2"] == 0.28).all()
    clamp_rate = float((frame["message"] != "").mean())
    print("floor / ceiling message rate:", clamp_rate)
    assert clamp_rate == 0.0
    binding = [i for i, f in enumerate(frame["bound_flags"]) if f]
    assert binding == [2, 3], frame["bound_flags"].tolist()
    assert all("omega1 at lower bound" in frame["bound_flags"].iloc[i] for i in binding)
    assert frame.loc[binding, ["omega1_se", "omega2_se", "chi_se"]].isna().all().all()
    se = frame[[f"{c}_se" for c in PARAM_COLUMNS]].to_numpy(dtype=float)
    assert np.all(np.isnan(se) | (se <= fit_2f_module.MAX_FINITE_SE))
    flags = flag_unidentified(frame)
    assert list(flags["param"]) == list(PARAM_COLUMNS)
    n = flags.set_index("param")["n_changes"]
    assert (n[["k1", "lambda1", "lambda2"]] == 6).all() and (
        n[["omega1", "omega2", "chi"]] == 3
    ).all()
    assert flags["unidentified"].dtype == bool
    with pytest.raises(ValueError):
        rolling_fit(hist, cfg, start=hist.dates[10], window_vol=120)


def test_stage3_machinery_on_cached_2f(ssvi, fast_sim) -> None:  # type: ignore[no-untyped-def]
    """The stage-3 report on the cached 2F Table 8.2 LSV (8e5 particles, read from the cache,
    never calibrated here) through the ``model=`` override, with the first-order SSRs of the
    Table 8.2 kernel's maps as ``fit_table``: pillars 3M and 1Y, 4e4 paths; the numerical LSV
    SSR (2.48 / 2.14) against the lsv-measure prediction (2.64 / 2.08: +6.5% / −2.8%, asserted
    within 10%) and the naked-vs-market measure (1.31 / 1.26, off by −47% / −41%)."""
    from volsto.calibration.cache import build_market

    spec = _reference_spec("2f")
    lsv = _cached(spec)
    _, surface, _ = build_market(spec)
    tg = marking_targets(surface, (0.25, 1.0))
    cfg = BreakEvenFitConfig(pillars=(0.25, 1.0), k2=0.28, k1_bounds=(0.5, 20.0))
    prob, _ = fit_2f_module._first_problem(tg, cfg, lsv.kernel.xi0)
    be = to_breakeven(P82)
    lam = np.array([be.lambda1, be.lambda2])
    table = fit_2f_module._first_table(prob, lam, prob.maps(P82.k1))
    inputs = Stage3Inputs(
        surface=surface,
        particle=spec.particle,
        sim=spec.sim,
        pricing_sim=fast_sim,
        ssr_pillars=(0.25, 1.0),
        breakeven_pillars=(0.25, 1.0),
        forward_starts=(),
        model=lsv,
        mixing_paths=50_000,
    )
    rep = stage3_validation(P82, inputs, tg, fit_table=table)
    print(rep.summary())
    assert not rep.recalibrated and rep.calibration_seconds == 0.0 and rep.n_particles == 800_000
    assert rep.n_paths == fast_sim.n_paths and rep.wall_seconds > 0
    assert 0.05 < rep.mean_abs_l_minus_1 < 0.3
    st = rep.ssr_table
    assert list(st["T"]) == [0.25, 1.0] and (st["ssr_model_se"] > 0).all()
    assert np.allclose(st["ssr_target"], 1.0)
    assert (st["ssr_model"] > 1.5).all()  # the Table 8.2 LSV is not an ssr = 1 fit
    assert np.all(np.abs(st["ssr_achieved_lsv"] / st["ssr_model"] - 1.0) < 0.10), st
    assert np.all(st["ssr_achieved_naked_vs_market"] / st["ssr_model"] < 0.7), st
    assert (rep.skew_table["skew_naked_se"] > 0).all() and (rep.skew_table["skew_naked"] < 0).all()
    assert np.allclose(rep.skew_table["skew_target"], tg.skew_target)
    assert rep.skew_table["skew_naked_first_order"].notna().all()
    be_t = rep.breakeven_table
    assert (be_t["svc_se"] > 0).all() and (be_t["volvar_se"] > 0).all()
    assert np.all(np.isfinite(be_t[["svc_sim", "volvar_sim", "svc_z", "volvar_z", "vovol_sim"]]))
    assert len(rep.forward_table) == 0 and rep.headline is None
    assert "recalibrated: no" in rep.summary()


def test_local_vol_ssr_numerical_machinery(ssvi) -> None:  # type: ignore[no-untyped-def]
    """The V2 correction path (``lv_ssr="numerical"``) on a small pure-Dupire Monte Carlo (2e4
    paths, dt 1/100 — machinery only; production uses 2e5 paths and dt ≤ 1/400): ``R^LV_num``
    with standard errors, ``r_lv_market`` replaced and the local integrals scaled by ``c(T) =
    (R^LV_num − 1)/(R^LV_o1 − 1)``; a missing ``LocalVolSSR`` raises, in the fitter and in the
    trade-off helper, which forwards it."""
    sim = SimConfig(n_paths=20_000, dt_max=1.0 / 100.0, chunk_size=20_000, seed=3)
    pillars = (0.25, 1.0)
    lv = local_vol_ssr_numerical(ssvi, pillars, sim=sim)
    print(lv)
    assert np.all((lv.R > 2.0) & (lv.R < 3.2)) and np.all(lv.R_stderr > 0)
    xi0 = xi0_curve(ssvi, 1.0)
    tg = marking_targets(ssvi, pillars)
    cfg = BreakEvenFitConfig(pillars=pillars, lv_ssr="numerical")
    with pytest.raises(ValueError):
        fit_first(tg, cfg, xi0)
    prob, notes = fit_2f_module._first_problem(tg, cfg, xi0, lv)
    assert np.allclose(prob.r_lv_market, lv.R)
    assert np.allclose(prob.lv_scale, (lv.R - 1.0) / (prob.r_lv_order1 - 1.0))
    assert any("pure-Dupire" in n for n in notes)
    f = fit_first(tg, cfg, xi0, lv_ssr=lv)
    assert np.allclose(f.maps.svc(np.zeros(2), "lsv"), tg.sigma_0 * tg.skew_target * lv.R)
    with pytest.raises(ValueError):
        skew_weight_tradeoff(ssvi, cfg, weights=(1.0,), proxy=False)
    df = skew_weight_tradeoff(ssvi, cfg, weights=(1.0,), proxy=False, lv_ssr=lv)
    assert len(df) == 1 and np.isfinite(df["ssr_floor"].iloc[0])


@pytest.mark.slow
def test_recovery_three_year_mixing(synthetic_3y) -> None:  # type: ignore[no-untyped-def]
    """SPEC recovery test on the three-year mixing history (seed 13, 10⁵ mixing paths per day,
    windows 250 / 250, k2 fixed at the true 0.28) with the **default** skew weight and measure
    (auto → naked), under both prefactor conventions: ``market`` within the owner's tolerances
    (ν, θ, k1 10%, ρ 0.05), ``model`` within 10% / 15% / 25% / 0.07; k2 = 0.2 reported."""
    hist = synthetic_3y.history
    pillars = tuple(map(float, hist.pillars))
    fits = {
        conv: fit_2f_historical(
            hist,
            BreakEvenFitConfig(
                pillars=pillars, k2=0.28, k1_bounds=(0.5, 20.0), sigma_hat_prefactor=conv
            ),
            window_vol=250,
            window_ssr=250,
        )
        for conv in ("market", "model")
    }
    for conv, r in fits.items():
        print(conv)
        print(r.summary())
        assert r.config.ssr_measure == "naked"
    p = fits["market"].params
    assert abs(p.nu - P82.nu) < 0.10 * P82.nu, p
    assert abs(p.theta - P82.theta) < 0.10 * P82.theta, p
    assert abs(p.k1 - P82.k1) < 0.10 * P82.k1, p
    assert abs(p.rho_SX1 - P82.rho_SX1) < 0.05 and abs(p.rho_SX2 - P82.rho_SX2) < 0.05, p
    assert fits["market"].message is None and not fits["market"].first.active
    q = fits["model"].params
    assert abs(q.nu - P82.nu) < 0.10 * P82.nu and abs(q.theta - P82.theta) < 0.15 * P82.theta
    assert abs(q.k1 - P82.k1) < 0.25 * P82.k1, q
    assert abs(q.rho_SX1 - P82.rho_SX1) < 0.07 and abs(q.rho_SX2 - P82.rho_SX2) < 0.07, q
    r2 = fit_2f_historical(
        hist, BreakEvenFitConfig(pillars=pillars, k2=0.2), window_vol=250, window_ssr=250
    )
    print("k2 = 0.2:", r2.params)
    assert abs(r2.params.nu - P82.nu) < 0.10 * P82.nu


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason="lsv measure in historical mode integrates a power-law extension of the pillar skews: "
    "measured nu 1.732, theta 0.270 (+10.2%), k1 7.20 (+35%), rho_SX1 -0.920 (off 0.16), rho_SX2 "
    "-0.573 (off 0.09) on the three-year history - outside the owner's tolerances; the default "
    "(auto) measure is naked there",
)
def test_recovery_three_year_lsv_historical(synthetic_3y) -> None:  # type: ignore[no-untyped-def]
    """The explicit lsv measure on the same history against the owner's tolerances (strict
    xfail: the regression stays visible)."""
    hist = synthetic_3y.history
    cfg = BreakEvenFitConfig(
        pillars=tuple(map(float, hist.pillars)), k2=0.28, k1_bounds=(0.5, 20.0), ssr_measure="lsv"
    )
    p = fit_2f_historical(hist, cfg, window_vol=250, window_ssr=250).params
    print("lsv measure:", p)
    assert abs(p.nu - P82.nu) < 0.10 * P82.nu, p
    assert abs(p.theta - P82.theta) < 0.10 * P82.theta, p
    assert abs(p.k1 - P82.k1) < 0.10 * P82.k1, p
    assert abs(p.rho_SX1 - P82.rho_SX1) < 0.05 and abs(p.rho_SX2 - P82.rho_SX2) < 0.05, p


@pytest.mark.slow
def test_real_data_end_to_end() -> None:
    """The 2022 H2 SPX history (plain SSVI snapshots): historical mode (pillars 1m–1y, windows
    100 / 60) and marking mode on the last snapshot at ``ssr_target = 1`` for skew weights 1 and
    0.1 (achieved SSR under both measures, naked-skew gap, leverage proxy, the message); sanity
    only (finite parameters, the floor lower at the looser weight)."""
    from volsto.market.loaders import load_ssvi_surface

    path = ROOT / "outputs" / "m7" / "hdn_history_ssvi.csv"
    if not path.exists():
        pytest.skip("2022 H2 history not built (scripts/m7_hdn_history.py --no-essvi)")
    hist = SurfaceHistory(pd.read_csv(path))
    cfg = BreakEvenFitConfig(pillars=(1 / 12, 0.25, 0.5, 1.0))
    r = fit_2f_historical(hist, cfg, window_vol=100, window_ssr=60)
    print(r.summary())
    p = r.params
    assert all(np.isfinite(v) for v in dataclasses.asdict(p).values())
    assert p.k1 > p.k2 == 0.2 and abs(p.rho_SX1) <= 1 and abs(p.rho_SX2) <= 1
    assert np.isfinite(r.first.objective) and np.isfinite(r.second.objective)
    snap = (
        ROOT
        / "configs"
        / "surfaces"
        / "snapshots"
        / "hdn_2022H2_ssvi"
        / f"spx_{hist.dates[-1].date()}.yaml"
    )
    if snap.exists():
        surface = load_ssvi_surface(snap)
        df = skew_weight_tradeoff(surface, cfg, weights=(1.0, 0.1), ssr_target=1.0)
        print(df.drop(columns=["message"]).round(4).T.to_string())
        assert df["ssr_floor"].iloc[1] < df["ssr_floor"].iloc[0]
        m = fit_2f_marking(surface, cfg, ssr_target=1.0)
        print(m.summary())
        assert all(np.isfinite(v) for v in dataclasses.asdict(m.params).values())
        assert (
            m.targets.mode == "marking"
            and (m.targets.policy_check()["reading"] == "absolute").all()
        )
