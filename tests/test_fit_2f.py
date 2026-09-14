"""Break-even fit of the two-factor model (SPEC §15 Part 3, M7 addendum; ``calibration/fit_2f.py``
and ``calibration/stability.py``).

Fast:

* the fitter's affine maps: ``PillarQuad`` reproduces ``kernels`` to round-off, the maps equal
  ``first_order_breakevens`` with the market ``sigma_hat`` to 1e-12, ``SpotVolCovar`` and
  ``Skew_naked`` are affine in ``(λ1, λ2)`` numerically, and the exact 2-D QP agrees with SLSQP;
* marking mode on the reference SSVI (pillars 3M–3Y, ``ssr_target = 1``, k2 = 0.2): the policy
  check reads ``absolute`` at every pillar with the lognormal mismatch factor 4.8–5.0 flagged;
  the skew guard holds at both guards but binds, k1 runs to its bound (20) and the second fit
  to its lower bounds — the target set is infeasible for a naked two-factor kernel (its SSR lies
  in [1, 2], eq. 9.21; first-order ``ssr_model`` 1.40 / 1.37 / 1.41 / 1.44 / 1.43 at 3M–3Y against
  1); the fitted naked kernel's **numerical** SSR (4e4 paths, dt 1/100) is 1.35 / 1.36 / 1.42 /
  1.48 / 1.49 (3M / 6M / 1Y / 2Y / 3Y) — the owner's ±0.2 self-consistency band is a strict
  ``xfail`` with the measured numbers (the structural floor, not a fitter defect);
* the SSR dial: at ``{0.8, 1.0, 1.2}`` the fit is bound-limited and the naked kernel's numerical
  SSR at 3M is unchanged within noise (1.353 / 1.353 / 1.370 ± 0.02: inert — the owner's
  monotonicity criterion is a strict ``xfail``); at ``ssr_target = 2`` (the short-maturity SSR of
  a diffusive smile) the solution is interior (k1 3.0, ν 2.3) and the numerical SSR rises to
  1.67 ± 0.02 at 3M; the skew guard holds at both guards throughout;
* historical mode on the one-year synthetic Table 8.2 history (order-one ATM source, seed 8,
  windows 200 / 200, k2 fixed at the true 0.28): ν 1.664, θ 0.238, k1 5.30, ρ_SX1 −0.833,
  ρ_SX2 −0.522 (true 1.74, 0.245, 5.35, −0.759, −0.487) — asserted within 15% / 15% / 20% /
  0.2 / 0.2; with k2 fixed at the default 0.2: ν 1.639, θ 0.224, k1 5.13, ρ −0.871 / −0.523;
* the stage-3 machinery on the cached 2F Table 8.2 model (no calibration: ``model=`` override),
  pillars 3M and 1Y, 4e4 paths: 15 s wall, ``recalibrated`` False, numerical SSR 2.48 / 2.14,
  naked mixing skew −0.505 / −0.283 against the market −0.696 / −0.343, simulated ``SpotVolCovar``
  −0.380 / −0.165 and ``VolVar`` 0.156 / 0.032 with standard errors;
* the rolling fit (7 dates, windows 120 / 60) and the identification flags; the YAML round trip.

Slow: the three-year mixing history (seed 13, 10⁵ mixing paths, windows 250 / 250, k2 = 0.28)
under both prefactor conventions (numbers in the test docstring after the skew-convention fix:
the SV skew is eq. 8.54 at the VS vol under both, the conventions differ in the sensitivities'
prefactor only) and the real-data end-to-end run on the 2022 H2 SPX history (historical and
marking modes, no fixed numbers; numbers in ``outputs/m7/fit_2f_hdn.md``).
"""

from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from scipy.optimize import LinearConstraint, minimize

from volsto.analytics.breakeven import first_order_breakevens, kernels
from volsto.analytics.reparam import BreakEvenParams
from volsto.analytics.smile_dynamics import ssr_numerical_many
from volsto.calibration.fit_2f import (
    BreakEvenFitConfig,
    Stage3Inputs,
    affine_maps,
    affine_maps_from_kernels,
    attainable_ssr,
    fit_2f_historical,
    fit_2f_marking,
    k1_profile,
    naked_kernel,
    pillar_quad,
    stage3_validation,
)
from volsto.calibration.history import SurfaceHistory, synthetic_2f_history
from volsto.calibration.stability import PARAM_COLUMNS, flag_unidentified, rolling_fit
from volsto.calibration.targets import marking_targets
from volsto.config import BergomiParams, SimConfig
from volsto.market.varswap import xi0_curve

fit_2f_module = importlib.import_module("volsto.calibration.fit_2f")

ROOT = Path(__file__).resolve().parents[1]
P82 = BergomiParams(1.74, 0.245, 5.35, 0.28, 0.0, -0.759, -0.487)
MARKING_PILLARS = (0.25, 0.5, 1.0, 2.0, 3.0)
SSR_SIM = SimConfig(n_paths=40_000, dt_max=1.0 / 100.0, chunk_size=20_000, seed=7)


@pytest.fixture(scope="module")
def synthetic_1y():  # type: ignore[no-untyped-def]
    return synthetic_2f_history(P82, years=1.0, seed=8, atm_source="order_one")


@pytest.fixture(scope="module")
def marking_fit(ssvi):  # type: ignore[no-untyped-def]
    cfg = BreakEvenFitConfig(pillars=MARKING_PILLARS)
    return fit_2f_marking(ssvi, cfg, ssr_target=1.0)


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        BreakEvenFitConfig(k2=0.28)  # k1_bounds[0] = 0.3 < k2 + 0.05
    with pytest.raises(ValueError):
        BreakEvenFitConfig(weights_svc="log")
    with pytest.raises(ValueError):
        BreakEvenFitConfig(sigma_hat_prefactor="mixed")
    with pytest.raises(ValueError):
        BreakEvenFitConfig(eps_skew=1.5)
    cfg = BreakEvenFitConfig(k2=0.28, k1_bounds=(0.5, 20.0))
    assert cfg.k1_bounds[0] > cfg.k2 + cfg.k1_min_gap


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
    # affinity: f(a + b) = f(a) + f(b), f(c a) = c f(a), f(0) = 0
    a, b = rng.normal(size=2), rng.normal(size=2)
    for f in (m_fast.svc, m_fast.skew):
        assert np.allclose(f(a + b), f(a) + f(b), atol=1e-12)
        assert np.allclose(f(2.5 * a), 2.5 * f(a), atol=1e-12)
        assert np.allclose(f(np.zeros(2)), 0.0)
    # the model convention: the skew is the kernel's own order-one skew
    m_model = affine_maps(quads, tg.atf, *ks, prefactor="model")
    fo0 = [first_order_breakevens(be, xi0, T) for T in MARKING_PILLARS]
    assert np.allclose(m_model.skew(lam), [f.skew for f in fo0], atol=1e-12)


def test_exact_qp_matches_slsqp(rng) -> None:  # type: ignore[no-untyped-def]
    """The candidate enumeration of the inner 2-D QP against SLSQP on random instances with
    the four guard inequalities (two strips), active and inactive."""
    qp = fit_2f_module._qp2
    for _ in range(25):
        M = rng.normal(size=(5, 2))
        y = rng.normal(size=5)
        H = M.T @ M
        g = M.T @ y
        js, jl = rng.normal(size=2), rng.normal(size=2)
        c_s, c_l = rng.normal(), rng.normal()
        width = rng.uniform(0.05, 1.0)
        G = np.array([js, -js, jl, -jl])
        h = np.array([c_s + width, -(c_s - width), c_l + width, -(c_l - width)])
        x, active, feas = qp(H, g, G, h)
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


def test_marking_self_consistency_and_policy(marking_fit, ssvi) -> None:  # type: ignore[no-untyped-def]
    """Marking mode on the reference SSVI (module docstring): policy check, skew guard, the
    documented infeasibility of ``ssr_target = 1`` (bounds and binding guards reported, never
    silent), the YAML round trip, the k1 profile."""
    r = marking_fit
    tg = r.targets
    pc = tg.policy_check()
    assert (pc["reading"] == "absolute").all(), pc
    assert np.all((pc["mismatch_lognormal"] > 4.5) & (pc["mismatch_lognormal"] < 5.1)), pc
    assert np.all(np.abs(tg.ssr_target - 1.0) < 1e-12)
    eps = r.config.eps_skew
    t = r.table.set_index("T")
    for T in r.first.guard:
        assert abs(t.loc[T, "skew_gap"]) <= eps + 1e-6, t
    # the structural floor: the first-order SSR of any naked kernel is >= 1 (eq. 9.21), so the
    # fit cannot reach 1 and says so through its diagnostics
    assert np.all(t["ssr_model"] > 1.3), t
    assert r.first.k1_at_bound and len(r.first.active) >= 1, r.first
    assert r.second.bound_flags, r.second
    assert r.first.feasible and np.isfinite(r.first.objective)
    assert r.risk_regime == "sticky_strike" and not r.recalibrated and r.stage3 is None
    assert r.params.k2 == r.config.k2 == 0.2 and r.params.k1 > r.params.k2
    # loadable YAML with provenance
    doc = yaml.safe_load(r.config_yaml)
    assert BergomiParams(**doc["model"]) == r.params
    assert BreakEvenParams(**doc["breakeven"]).to_book() == r.params
    assert doc["risk_regime"] == "sticky_strike"
    assert doc["provenance"]["mode"] == "marking" and doc["provenance"]["k2_fixed"] == 0.2
    assert doc["provenance"]["sigma_hat_prefactor"] == "market"
    assert "recalibrated: no" in r.summary() and "wall clock" in r.summary()
    prof = k1_profile(tg, r.config, r.xi0, k1s=[2.0, r.params.k1, 8.0])
    assert prof["objective"].iloc[1] <= prof["objective"].min() + 1e-12


def test_marking_naked_kernel_numerical_ssr_reported(marking_fit, ssvi) -> None:  # type: ignore[no-untyped-def]
    """The fitted naked kernel's numerical SSR (pure SV, 4e4 paths, dt 1/100) is measured and
    printed: 1.35 / 1.36 / 1.42 / 1.48 / 1.49 ± 0.02 at 3M / 6M / 1Y / 2Y / 3Y against the
    target 1 (asserted to be finite, inside [1, 2] and within 0.25 of the fit's own first-order
    ``ssr_model``); the band assertion is the xfail below."""
    rows = ssr_numerical_many(
        naked_kernel(marking_fit, ssvi.forward_curve), MARKING_PILLARS, eps=0.05, sim=SSR_SIM
    )
    R = np.array([x.R for x in rows])
    se = np.array([x.R_stderr for x in rows])
    print("naked numerical SSR:", [(x.T, round(x.R, 3), round(x.R_stderr, 3)) for x in rows])
    assert np.all(np.isfinite(R)) and np.all(se > 0) and np.all(se < 0.05)
    assert np.all((1.0 - 2 * se < R) & (2.0 + 2 * se > R))
    first_order = marking_fit.table.set_index("T").loc[list(MARKING_PILLARS), "ssr_model"]
    assert np.all(np.abs(R - first_order.to_numpy()) < 0.25), (R, first_order.to_numpy())


@pytest.mark.xfail(
    strict=True,
    reason="owner's self-consistency band: the naked kernel's numerical SSR is 1.35-1.49 across "
    "3M-3Y against ssr_target 1 (measured); a naked two-factor kernel's SSR lies in [1, 2] "
    "(eq. 9.21, -> 2 at short T), so the target is a floor it cannot reach with the market skew",
)
def test_marking_self_consistency_band(marking_fit, ssvi) -> None:  # type: ignore[no-untyped-def]
    rows = ssr_numerical_many(
        naked_kernel(marking_fit, ssvi.forward_curve), MARKING_PILLARS, eps=0.05, sim=SSR_SIM
    )
    R = np.array([x.R for x in rows])
    assert np.all(np.abs(R - 1.0) < 0.2), R


def _dial_ssr(ssvi, cfg, dials):  # type: ignore[no-untyped-def]
    out = []
    for s in dials:
        r = fit_2f_marking(ssvi, cfg, ssr_target=s)
        t = r.table.set_index("T")
        for T in r.first.guard:
            assert abs(t.loc[T, "skew_gap"]) <= cfg.eps_skew + 1e-6, (s, t)
        assert np.allclose(r.targets.ssr_target, s)
        rows = ssr_numerical_many(
            naked_kernel(r, ssvi.forward_curve), [0.25, 1.0], eps=0.05, sim=SSR_SIM
        )
        out.append((r, rows[0].R, rows[0].R_stderr, rows[1].R))
    return out


def test_ssr_dial(ssvi) -> None:  # type: ignore[no-untyped-def]
    """The dial (module docstring): at ``{0.8, 1.0, 1.2}`` the fit is bound-limited (k1 at its
    bound, guards binding) and the naked kernel's numerical SSR at 3M is unchanged within 2 se
    (inert, 1.353 / 1.353 / 1.370 ± 0.02); at ``ssr_target = 2`` the solution is interior and
    the numerical SSR rises well above the inert value (1.67 ± 0.02); the skew guard holds at
    both guards throughout."""
    cfg = BreakEvenFitConfig(pillars=MARKING_PILLARS)
    low = _dial_ssr(ssvi, cfg, (0.8, 1.0, 1.2))
    R3m = [x[1] for x in low]
    se3m = [x[2] for x in low]
    print("dial numerical SSR 3M:", np.round(R3m, 3), "1Y:", np.round([x[3] for x in low], 3))
    assert all(x[0].first.k1_at_bound for x in low)
    assert max(R3m) - min(R3m) < 2 * max(se3m) + 0.02, (R3m, se3m)
    ((r2, R2, se2, _),) = _dial_ssr(ssvi, cfg, (2.0,))
    assert not r2.first.k1_at_bound, r2.first
    assert max(R3m) + 2 * (se2 + max(se3m)) < R2, (R2, R3m)


@pytest.mark.xfail(
    strict=True,
    reason="owner's dial criterion: sweeping ssr_target in {0.8, 1.0, 1.2} must move the naked "
    "kernel's numerical SSR monotonically; measured 1.353 / 1.353 / 1.370 +- 0.02 at 3M — the "
    "fit is bound-limited there (the target SSR is below the kernel's floor), so the dial is inert",
)
def test_ssr_dial_monotone_criterion(ssvi) -> None:  # type: ignore[no-untyped-def]
    cfg = BreakEvenFitConfig(pillars=MARKING_PILLARS)
    low = _dial_ssr(ssvi, cfg, (0.8, 1.0, 1.2))
    R3m = np.array([x[1] for x in low])
    se3m = np.array([x[2] for x in low])
    d = np.diff(R3m)
    assert np.all(d > 0) and d[0] > 2 * max(se3m[:2]), (R3m, se3m)


def test_historical_recovery_fast(synthetic_1y) -> None:  # type: ignore[no-untyped-def]
    """Recovery on the one-year order-one synthetic history (module docstring numbers) with
    k2 fixed at the true 0.28; the k2 = 0.2 default is reported."""
    hist = synthetic_1y.history
    cfg = BreakEvenFitConfig(
        pillars=tuple(map(float, hist.pillars)), k2=0.28, k1_bounds=(0.5, 20.0)
    )
    r = fit_2f_historical(hist, cfg, window_vol=200, window_ssr=200)
    p = r.params
    print(r.summary())
    assert r.targets.mode == "historical" and r.pricing_date == hist.dates[-1]
    assert abs(p.nu - P82.nu) < 0.15 * P82.nu, p
    assert abs(p.theta - P82.theta) < 0.15 * P82.theta, p
    assert abs(p.k1 - P82.k1) < 0.20 * P82.k1, p
    assert abs(p.rho_SX1 - P82.rho_SX1) < 0.2 and abs(p.rho_SX2 - P82.rho_SX2) < 0.2, p
    assert p.k2 == 0.28 and r.first.feasible and not r.first.active
    assert np.isfinite(r.first.k1_se) and r.first.k1_se < 0.5
    assert np.all(np.isfinite([r.first.lambda1_se, r.first.lambda2_se]))
    # the target standard errors are carried (historical mode) and the fit reproduces svc
    assert np.all(r.targets.spot_vol_covar_se > 0) and np.all(r.targets.vol_var_se > 0)
    assert np.all(np.abs(r.table["svc_model"] / r.table["svc_target"] - 1.0) < 0.01)
    r2 = fit_2f_historical(
        hist, BreakEvenFitConfig(pillars=cfg.pillars, k2=0.2), window_vol=200, window_ssr=200
    )
    q = r2.params
    print("k2 = 0.2:", q)
    assert abs(q.nu - P82.nu) < 0.15 * P82.nu and abs(q.k1 - P82.k1) < 0.2 * P82.k1


def test_rolling_fit_and_flags(synthetic_1y) -> None:  # type: ignore[no-untyped-def]
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
    assert len(frame) == 7
    for c in PARAM_COLUMNS:
        assert c in frame and f"{c}_se" in frame
    assert {
        "nu",
        "theta",
        "rho_SX1",
        "rho_SX2",
        "rho12",
        "first_objective",
        "second_objective",
    } <= set(frame.columns)
    assert (frame["k2"] == 0.28).all()
    flags = flag_unidentified(frame)
    assert list(flags["param"]) == list(PARAM_COLUMNS)
    assert (flags["n_changes"] == 6).all() and flags["unidentified"].dtype == bool
    with pytest.raises(ValueError):
        rolling_fit(hist, cfg, start=hist.dates[10], window_vol=120)


def test_stage3_machinery_on_cached_2f(fast_sim) -> None:  # type: ignore[no-untyped-def]
    """The stage-3 report on the cached 2F Table 8.2 LSV (8e5 particles, read from the cache,
    never calibrated here) through the ``model=`` override: pillars 3M and 1Y, 4e4 paths,
    5e4 mixing paths; numbers in the module docstring."""
    from volsto.calibration.cache import LeverageCache, build_market
    from volsto.config import CalibrationSpec, load_yaml

    spec = load_yaml(ROOT / "configs" / "studies" / "lsv_reference_2f.yaml", CalibrationSpec)
    spec = dataclasses.replace(
        spec, particle=dataclasses.replace(spec.particle, n_particles=800_000)
    )
    lsv, _ = LeverageCache(ROOT / "cache").get_or_calibrate(spec, allow_calibrate=False)
    _, surface, _ = build_market(spec)
    tg = marking_targets(surface, (0.25, 1.0))
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
    rep = stage3_validation(P82, inputs, tg)
    print(rep.summary())
    assert not rep.recalibrated and rep.calibration_seconds == 0.0 and rep.n_particles == 800_000
    assert rep.n_paths == fast_sim.n_paths and rep.wall_seconds > 0
    assert 0.05 < rep.mean_abs_l_minus_1 < 0.3
    assert list(rep.ssr_table["T"]) == [0.25, 1.0] and (rep.ssr_table["ssr_model_se"] > 0).all()
    assert np.allclose(rep.ssr_table["ssr_target"], 1.0)
    assert (rep.ssr_table["ssr_model"] > 1.5).all()  # the Table 8.2 LSV is not an ssr = 1 fit
    assert (rep.skew_table["skew_naked_se"] > 0).all() and (rep.skew_table["skew_naked"] < 0).all()
    assert np.allclose(rep.skew_table["skew_target"], tg.skew_target)
    be = rep.breakeven_table
    assert (be["svc_se"] > 0).all() and (be["volvar_se"] > 0).all()
    assert np.all(np.isfinite(be[["svc_sim", "volvar_sim", "svc_z", "volvar_z", "vovol_sim"]]))
    assert len(rep.forward_table) == 0 and rep.headline is None
    assert "recalibrated: no" in rep.summary()


@pytest.mark.slow
def test_recovery_three_year_mixing() -> None:
    """SPEC recovery test on the three-year mixing history (seed 13, 10⁵ mixing paths per
        day, windows 250 / 250, k2 fixed at the true 0.28) under both prefactor conventions
    (the default ``market`` convention within the owner's tolerances ν, θ, k1 10% and ρ 0.05,
        the ``model`` one within 10% / 15% / 25% / 0.07); k2 = 0.2 reported alongside."""
    sh = synthetic_2f_history(P82, years=3.0, seed=13, atm_source="mixing", mixing_paths=100_000)
    hist = sh.history
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
    # the default (market prefactor for the sensitivities, eq. 8.54 skew at the VS vol):
    # nu 1.768, theta 0.248, k1 5.51, rho_SX1 -0.720, rho_SX2 -0.482, rho12 +0.09 measured —
    # inside the owner's tolerances (10% / 0.05); no guard binds on the true dynamics
    p = fits["market"].params
    assert abs(p.nu - P82.nu) < 0.10 * P82.nu, p
    assert abs(p.theta - P82.theta) < 0.10 * P82.theta, p
    assert abs(p.k1 - P82.k1) < 0.10 * P82.k1, p
    assert abs(p.rho_SX1 - P82.rho_SX1) < 0.05 and abs(p.rho_SX2 - P82.rho_SX2) < 0.05, p
    assert len(fits["market"].first.active) == 0, fits["market"].first
    # the "model" convention (VS-vol prefactor): nu 1.717, theta 0.212, k1 6.63, rho -0.766 /
    # -0.550 measured — looser on theta and k1
    q = fits["model"].params
    assert abs(q.nu - P82.nu) < 0.10 * P82.nu and abs(q.theta - P82.theta) < 0.15 * P82.theta
    assert abs(q.k1 - P82.k1) < 0.25 * P82.k1, q
    assert abs(q.rho_SX1 - P82.rho_SX1) < 0.07 and abs(q.rho_SX2 - P82.rho_SX2) < 0.07, q
    r2 = fit_2f_historical(
        hist,
        BreakEvenFitConfig(pillars=pillars, k2=0.2),
        window_vol=250,
        window_ssr=250,
    )
    print("k2 = 0.2:", r2.params)
    assert abs(r2.params.nu - P82.nu) < 0.10 * P82.nu


@pytest.mark.slow
def test_real_data_end_to_end() -> None:
    """The 2022 H2 SPX history (plain SSVI snapshots): historical mode (pillars 1m–1y, windows
    100 / 60) and marking mode on the last snapshot; sanity only (finite, PSD, objectives)."""
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
        m = fit_2f_marking(surface, cfg, ssr_target=1.0)
        print(m.summary())
        assert all(np.isfinite(v) for v in dataclasses.asdict(m.params).values())
        assert (
            m.targets.mode == "marking"
            and (m.targets.policy_check()["reading"] == "absolute").all()
        )


def test_attainable_ssr_band(marking_fit, ssvi) -> None:  # type: ignore[no-untyped-def]
    """The feasible SSR band per pillar (skew inside the ±10% guard, loadings on the skew's
    side): on the reference SSVI the band at an exact skew match lies above 1 at every pillar
    (1.24–1.35 at 3M, 1.03–1.43 at 3Y) and the guard corners reach below 1 only at 3Y (0.92);
    the band contains the fit's own first-order ``ssr_model`` — the ``ssr_target = 1``
    infeasibility at the short and medium pillars made explicit for the owner."""
    band = attainable_ssr(marking_fit.targets, marking_fit.config, marking_fit.xi0)
    assert list(band["T"]) == list(MARKING_PILLARS)
    assert np.all(band["ssr_at_skew_match_min"] > 1.0), band
    assert np.all(band.loc[band["T"] <= 2.0, "ssr_min"] > 1.0), band
    assert np.all(band["ssr_min"] <= band["ssr_at_skew_match_min"] + 1e-9)
    t = marking_fit.table.set_index("T")
    for _, row in band.iterrows():
        assert row["ssr_min"] - 1e-6 <= t.loc[row["T"], "ssr_model"] <= row["ssr_max"] + 1e-6, (
            row,
            t,
        )
    print(band.round(3).to_string(index=False))
