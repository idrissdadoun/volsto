"""Smile-dynamics analytics (SPEC §4.4, §7.14, §15 Part 1; Bergomi ch. 9 and §12.4).

Pure one-factor Bergomi kernels on a flat 20% VS curve (the order-one formulas are accurate at
small vol of vol; the T → 0 and T → ∞ limits of the SSR are 2 and 1 for any ν):

* the numerical SSR (state-bump estimator) matches eq. 9.21 at ν = 0.6 and tends to 2 at one
  week; the joint and factor-only bumps agree for pure SV (the ATMF vol has no spot dependence
  with ``L ≡ 1``); the estimate is ε-independent;
* the M5 regime identity: the ``"model"`` delta regime of the risk layer bumps the spot with the
  leverage held and the factors at zero, i.e. the spot-partial term of eq. 12.50 alone; the
  ATM-vol shift per unit ``Δ ln S`` of that regime is ``ssr_numerical(kind="spot").slope`` (zero
  for pure SV) while ``SSR_T · S_T`` is the slope of the joint (conditional) move — both are
  asserted, the joint one against the conditional smile after a spot move at one week;
* the vol of the ATMF vol (eq. 12.56) equals the VS vol of vol of eq. 7.39 at one month for pure
  SV, and the vol-of-vol term structure with the state-bump partials reproduces eq. 7.39;
* the eq. 12.52 decomposition: ``R = R^SV`` when the market skew is the kernel's own order-one
  skew, ``R = R^LV(Mkt)`` when ``ν → 0`` (deterministic surface), and the linear interpolation in
  between (hand check on a synthetic ATMF term structure);
* the Var(V) decomposition: pure SV Monte Carlo variance within 3 stderr of the exact discrete
  moments, leverage terms identically zero; the cached 1F LSV split (slow).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from volsto.analytics.bergomi import (
    atmf_skew_order1_flat,
    ssr_order1_flat,
    vs_vol_of_vol_flat,
)
from volsto.analytics.smile_dynamics import (
    atmf_vol_of_vol,
    conditional_smile,
    ssr_decomposition,
    ssr_numerical,
    ssr_numerical_many,
    ssr_short_horizon,
    state_bump,
    state_covariance,
    volvol_term_structure,
)
from volsto.analytics.var_decomp import var_decomposition
from volsto.config import BergomiParams, SimConfig
from volsto.market import ForwardCurve, ForwardVarianceCurve
from volsto.models import BergomiSV

ROOT = Path(__file__).resolve().parents[1]
FC0 = ForwardCurve.flat(100.0, 0.0, 0.0)
XI_FLAT = ForwardVarianceCurve.flat(0.04)
P_SMALL = BergomiParams.one_factor(0.6, 2.0, -0.7)  # small vol of vol: order one accurate
P_LARGE = BergomiParams.one_factor(1.5, 2.0, -0.7)
SIM = SimConfig(n_paths=40_000, dt_max=1.0 / 100.0, chunk_size=40_000, seed=3)


def _model(p: BergomiParams = P_SMALL) -> BergomiSV:
    return BergomiSV(p, XI_FLAT, FC0)


def test_ssr_numerical_pure_sv_matches_order_one_and_limits() -> None:
    m = _model()
    r = ssr_numerical(m, 1.0, eps=0.05, sim=SIM)
    r1 = float(ssr_order1_flat(P_SMALL, 1.0))
    assert abs(r.R - r1) < max(3 * r.R_stderr, 0.05), (r, r1)
    assert abs(r.skew - float(atmf_skew_order1_flat(P_SMALL, 1.0))) < 0.01
    assert r.slope == pytest.approx(r.R * r.skew) and r.sigma_0 == pytest.approx(0.2)
    # factors alone = joint for pure SV (no spot dependence of the ATMF vol with L ≡ 1)
    rf = ssr_numerical(m, 1.0, eps=0.05, sim=SIM, kind="factors")
    assert abs(rf.R - r.R) < 2 * np.hypot(r.R_stderr, rf.R_stderr) + 1e-3
    rs = ssr_numerical(m, 1.0, eps=0.05, sim=SIM, kind="spot")
    assert abs(rs.slope) < 3 * rs.slope_stderr + 1e-4  # the spot term vanishes
    # limits: R → 2 at short maturity (book §9.3), the Table 8.2 shoulder ≈ 1.5 at 1–5y
    r0 = ssr_numerical(m, 1.0 / 52.0, eps=0.05, sim=SIM)
    assert abs(r0.R - 2.0) < max(3 * r0.R_stderr, 0.05), r0
    p82 = BergomiParams(1.74, 0.245, 5.35, 0.28, 0.0, -0.759, -0.487)
    rows = ssr_numerical_many(BergomiSV(p82, XI_FLAT, FC0), [1.0, 3.0], eps=0.05, sim=SIM)
    assert all(1.3 < row.R < 1.7 for row in rows), rows
    # ε-independence (central scheme: O(ε²))
    r_eps = ssr_numerical(m, 1.0, eps=0.02, sim=SIM)
    assert abs(r_eps.R - r.R) < 2 * np.hypot(r.R_stderr, r_eps.R_stderr) + 0.02
    with pytest.raises(ValueError):
        ssr_numerical(m, 1.0, eps=-1.0, sim=SIM)
    with pytest.raises(ValueError):
        ssr_numerical(m, 1.0, eps=0.05, sim=SIM, kind="sideways")


def test_m5_regime_identity_and_conditional_smile() -> None:
    """Model-regime ATM-vol shift per unit Δ ln S = the spot-partial slope (0 for pure SV);
    SSR_T · S_T = the slope of the joint move = the conditional ATMF vol shift per unit spot
    move at a one-week horizon (book eq. 9.3 read over [0, t])."""
    m = _model(P_LARGE)
    sim = SimConfig(n_paths=100_000, dt_max=1.0 / 100.0, chunk_size=100_000, seed=5)
    joint = ssr_numerical(m, 1.0, eps=0.05, sim=sim)
    spot = ssr_numerical(m, 1.0, eps=0.05, sim=sim, kind="spot")
    # the risk layer's "model" regime: spot bumped, leverage held in spot, factors at zero
    up = state_bump(m, d_ln_s=0.01)
    dn = state_bump(m, d_ln_s=-0.01)
    from volsto.engine import MonteCarlo
    from volsto.market import implied_vol
    from volsto.products import EuropeanOption

    def atmf_vol(model: BergomiSV) -> float:
        F = float(model.forward_curve.forward(1.0))
        res = MonteCarlo(sim).price(EuropeanOption(F, 1.0, 1, FC0.rate_curve), model)
        return float(implied_vol(res.mean, F, F, 1.0, 1, 1.0))

    model_regime_slope = (atmf_vol(up) - atmf_vol(dn)) / 0.02
    assert abs(model_regime_slope - spot.slope) < 3 * spot.slope_stderr + 5e-4
    assert abs(spot.slope) < 1e-3  # pure SV: the ATMF vol does not depend on ln S_0
    # the joint (conditional) move carries SSR_T · S_T
    assert joint.slope == pytest.approx(joint.R * joint.skew)
    h = 0.05
    cs_up = conditional_smile(m, 1.0 / 52.0, h, 1.0, [0.0], sim=sim)
    cs_dn = conditional_smile(m, 1.0 / 52.0, -h, 1.0, [0.0], sim=sim)
    slope_cond = (cs_up.atmf_vol[0] - cs_dn.atmf_vol[0]) / (2 * h)
    se_cond = float(np.hypot(cs_up.atmf_vol[1], cs_dn.atmf_vol[1]) / (2 * h))
    assert abs(slope_cond - joint.slope) < 3 * np.hypot(se_cond, joint.slope_stderr) + 0.01, (
        slope_cond,
        se_cond,
        joint,
    )
    cov = state_covariance(m)
    assert cov.shape == (2, 2) and cov[0, 0] == pytest.approx(0.04) and cov[1, 1] == 1.0
    assert cov[0, 1] == pytest.approx(-0.7 * 0.2)


def test_atmf_vol_of_vol_and_term_structure_pure_sv() -> None:
    m = _model()
    vv = atmf_vol_of_vol(m, 1.0 / 12.0, eps=0.05, sim=SIM)
    ref = float(vs_vol_of_vol_flat(P_SMALL, 1.0 / 12.0))
    assert abs(vv.vol_of_vol - ref) < max(3 * vv.vol_of_vol_stderr, 0.01 * ref), (vv, ref)
    assert vv.vs_vol_of_vol == pytest.approx(ref)
    assert abs(vv.spot_vol_correlation - (-0.7)) < 0.05
    r = ssr_numerical(m, 1.0 / 12.0, eps=0.05, sim=SIM)
    assert abs(vv.ssr - r.R) < 3 * np.hypot(vv.ssr_stderr, r.R_stderr) + 1e-3
    ts = volvol_term_structure(m, [0.25, 1.0], eps=0.05, sim=SIM)
    for i in range(2):
        tol = 2 * ts.model_volvol_stderr[i] + 0.02 * ts.kernel_volvol[i]
        assert abs(ts.model_volvol[i] - ts.kernel_volvol[i]) < tol, ts
    assert np.allclose(ts.kernel_volvol, vs_vol_of_vol_flat(P_SMALL, [0.25, 1.0]))
    assert np.all(np.diff(ts.kernel_volvol) < 0)  # vol of vol decays with maturity
    frame = ts.to_dataframe()
    assert {"volvol_kernel_eq_7_39", "volvol_model", "spot_vol_corr"} <= set(frame.columns)


class _SyntheticSurface:
    """ATMF vol 20% flat with a prescribed ATM skew term structure (duck-typed surface)."""

    def __init__(self, skew_fn) -> None:  # type: ignore[no-untyped-def]
        self._skew = skew_fn

    def atm_vol(self, t):  # type: ignore[no-untyped-def]
        return np.full_like(np.asarray(t, dtype=float), 0.2)

    def atm_skew(self, t):  # type: ignore[no-untyped-def]
        return np.asarray(self._skew(np.asarray(t, dtype=float)), dtype=float)


def test_ssr_decomposition_hand_checks() -> None:
    """Eq. 12.52 on the flat expansion: market skew = the kernel's order-one skew gives R =
    R^SV (eq. 9.21); a scaled market skew interpolates linearly; ν → 0 gives R = R^LV(Mkt)."""
    m = _model(P_LARGE)
    T = 1.0
    own = _SyntheticSurface(lambda t: atmf_skew_order1_flat(P_LARGE, t))
    d = ssr_decomposition(m, own, T, term_structure="flat")  # type: ignore[arg-type]
    assert pytest.approx(float(ssr_order1_flat(P_LARGE, T)), rel=1e-10) == d.R
    assert d.skew_ratio == pytest.approx(1.0) and d.R_lv_market == pytest.approx(d.R_lv_sv)
    twice = _SyntheticSurface(lambda t: 2.0 * atmf_skew_order1_flat(P_LARGE, t))
    d2 = ssr_decomposition(m, twice, T, term_structure="flat")  # type: ignore[arg-type]
    assert d2.skew_ratio == pytest.approx(0.5)
    assert pytest.approx(d2.R_lv_market + 0.5 * (d2.R_sv - d2.R_lv_sv)) == d2.R
    # R^LV(Mkt) for a skew ∝ 1/√t is 1 + (1/T)∫ √(T/t) dt = 3 (book: local vol on a √t skew)
    sqrt_t = _SyntheticSurface(lambda t: -0.05 / np.sqrt(np.maximum(t, 1e-12)))
    d3 = ssr_decomposition(m, sqrt_t, T, term_structure="flat")  # type: ignore[arg-type]
    assert d3.R_lv_market == pytest.approx(3.0, abs=1e-3)
    tiny = BergomiSV(BergomiParams.one_factor(1e-4, 2.0, -0.7), XI_FLAT, FC0)
    d4 = ssr_decomposition(tiny, twice, T, term_structure="flat")  # type: ignore[arg-type]
    assert pytest.approx(d4.R_lv_market, abs=1e-3) == d4.R
    assert "SSRDecomposition" in repr(d)
    with pytest.raises(ValueError):
        ssr_decomposition(m, own, T, term_structure="sideways")  # type: ignore[arg-type]


def test_var_decomposition_pure_sv() -> None:
    m = _model(P_LARGE)
    vd = var_decomposition(m, 1.0, sim=SIM)
    assert abs(vd.var_total - vd.var_sv_discrete) < 3 * vd.var_total_stderr, vd
    assert abs(vd.var_sv_discrete - vd.var_sv_closed) < 0.02 * vd.var_sv_closed
    assert vd.var_leverage == 0.0 and vd.cov_cross == 0.0 and vd.var_sv == vd.var_total
    assert abs(vd.mean - vd.mean_sv_closed) < 3 * vd.mean_stderr + 1e-6
    assert vd.var_explained == pytest.approx(vd.var_total, rel=1e-6)  # V = V^SV exactly
    with pytest.raises(ValueError):
        var_decomposition(m, -1.0, sim=SIM)


@pytest.mark.slow
def test_ssr_short_horizon_slow() -> None:
    m = _model(P_LARGE)
    sim = SimConfig(n_paths=200_000, dt_max=1.0 / 100.0, chunk_size=200_000, seed=11)
    ref = ssr_numerical(m, 1.0, eps=0.05, sim=sim)
    sh = ssr_short_horizon(m, 1.0, sim=sim)
    assert abs(sh.R - ref.R) < 3 * np.hypot(sh.R_stderr, ref.R_stderr) + 0.05, (sh, ref)


@pytest.mark.slow
def test_lsv_ssr_decomposition_vs_numerical(ssvi) -> None:  # type: ignore[no-untyped-def]
    """The cached reference 1F LSV (ω = 3, 8·10⁵ particles): the numerical SSR against the
    eq. 12.52 decomposition on the reference surface, and the Var(V) split."""
    from volsto.calibration import LeverageCache
    from volsto.config import CalibrationSpec, load_yaml
    from volsto.studies.m4 import headline_models

    spec_1f = load_yaml(ROOT / "configs" / "studies" / "lsv_reference_1f.yaml", CalibrationSpec)
    spec_2f = load_yaml(ROOT / "configs" / "studies" / "lsv_reference_2f.yaml", CalibrationSpec)
    models, _ = headline_models(
        LeverageCache(ROOT / "cache"), spec_1f, spec_2f, n_particles=800_000
    )
    lsv = models["1F ω=3"]
    sim = SimConfig(n_paths=200_000, dt_max=1.0 / 100.0, chunk_size=100_000, seed=21)
    num = ssr_numerical(lsv, 1.0, eps=0.05, sim=sim)
    dec = ssr_decomposition(lsv, ssvi, 1.0)
    assert abs(num.R - dec.R) < max(4 * num.R_stderr, 0.15), (num, dec)
    vd = var_decomposition(lsv, 1.0, sim=sim)
    assert (
        vd.var_total > 0 and abs(vd.var_sv + vd.var_leverage + vd.cov_cross - vd.var_total) < 1e-12
    )
