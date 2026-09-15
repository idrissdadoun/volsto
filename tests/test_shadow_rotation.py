"""Shadow-rotation greek (SPEC §15 Part 3 KEEP list and report decision ix;
``risk/shadow_rotation.py``).  No calibration runs here: the rotated states of
``scripts/m7_p1_marking.py`` are read from the cache with ``allow_calibrate=False`` (skipped when
absent); no wall-clock assertion.

* the rota: 2/sqrt(T) ln(110/90) vol points of 90/110 skew (0.568 at 6M, 0.560 with the saturated
  profile), the ATM skew moved by exactly ``−size · 0.02/sqrt(max(T, 1M))`` with the ATM vol
  unchanged, the arbitrage checks passed at ±1 rota on both surfaces, the ``+h`` then ``−h`` round
  trip back to the base surface;
* the refit under the two owner policies (``sabr_linked``, ``sticky_breakeven``) and the
  implementer's variant (``sticky_breakeven_skew``): on SPX 2022-12-30 the SABR-linked targets'
  ``Skew_SABR`` moves by ∓ one rota, the sticky targets hold ``VoV_BE`` / ``Corr_BE`` at the base
  values (the variant also the two-point skew reference), the round-trip surface refits to the base
  parameters, ν rises with the rotation under both policies, an unknown policy raises;
* d(fee)/d(rota) of the M6 headline 3y autocall on the cached calibrations under both owner
  policies: usual, recalibrated and shadow with standard errors, shadow = recalibrated − usual
  path by path, the refit sets equal the study's record, nothing recalibrated; the **sign test**
  (decision ix) on the recorded study numbers: the owner expected ``|shadow_sticky| >>
  |shadow_sabr|`` with the sticky-break-even shadow negative — measured: both shadows positive
  (the recalibration offsets the usual rotation under both policies), the sticky one 1.4× the
  SABR-linked one; the test asserts what was measured and records the expectation.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from volsto.calibration.cache import CacheMissError, LeverageCache, build_market
from volsto.calibration.fit_2f import BreakEvenFitConfig, fit_2f_marking
from volsto.config import (
    BergomiParams,
    CalibrationSpec,
    MarketConfig,
    SimConfig,
    SSVIConfig,
    SurfacePerturbation,
    from_mapping,
    load_yaml,
)
from volsto.market.surface import atm_skew_numeric, perturbed_surface
from volsto.risk.engine import RiskState, surface_of
from volsto.risk.shadow_rotation import (
    RECALIBRATION_POLICIES,
    ROTA_T_MIN,
    refit_on_rotated,
    rota_skew_vol_points,
    rota_slope,
    rotation_perturbation,
    rotation_shadow_sensitivity,
    rotation_states,
)
from volsto.studies.m6 import AUTOCALL_NAME, headline_products

ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = ROOT / "configs" / "studies" / "m7_p1_marking"
SPX = ROOT / "configs" / "surfaces" / "snapshots" / "hdn_2022H2_ssvi" / "spx_2022-12-30.yaml"
OWNER_POLICIES = ("sabr_linked", "sticky_breakeven")


def _spec(surface: str) -> CalibrationSpec:
    """The study's base calibration spec (``scripts/m7_p1_marking.py::base_spec``)."""
    ref = load_yaml(ROOT / "configs" / "studies" / "lsv_reference_2f.yaml", CalibrationSpec)
    if surface == "spx":
        ref = dataclasses.replace(
            ref,
            market=load_yaml(SPX, MarketConfig, section="market"),
            surface=load_yaml(SPX, SSVIConfig, section="ssvi"),
        )
    return dataclasses.replace(ref, particle=dataclasses.replace(ref.particle, n_particles=200_000))


def _record(policy: str) -> dict | None:  # type: ignore[type-arg]
    path = STUDY_DIR / f"rotation_spx_{policy}.yaml"
    if not path.exists():
        return None
    doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    return dict(doc) if isinstance(doc, dict) else None


def test_rota_units_and_surface_round_trip(ssvi) -> None:  # type: ignore[no-untyped-def]
    lin = rota_skew_vol_points(0.5, linear=True)
    assert lin == pytest.approx(2.0 / math.sqrt(0.5) * math.log(110.0 / 90.0), rel=1e-12)
    assert lin == pytest.approx(0.5676, abs=1e-4) and rota_skew_vol_points(0.5) == pytest.approx(
        0.5600, abs=1e-4
    )
    assert rota_skew_vol_points(1.0 / 52.0) == rota_skew_vol_points(ROTA_T_MIN)  # floored
    spx = surface_of(RiskState(_spec("spx")))
    for base in (ssvi, spx):
        k = np.linspace(-0.8, 0.8, 41)
        for size in (1.0, -1.0):
            s = perturbed_surface(rotation_perturbation(size), base)  # arbitrage-checked
            for T in (0.05, 0.25, 0.5, 1.0, 2.0, 3.0):
                assert float(s.atm_vol(T)) == pytest.approx(float(base.atm_vol(T)), abs=1e-15)
                ds = float(s.atm_skew(T)) - float(atm_skew_numeric(base, T)[()])
                assert ds == pytest.approx(-size * rota_slope(T), rel=1e-5), (T, ds)
                lo, hi = np.log(0.9), np.log(1.1)
                d90 = float(s.implied_vol_k(lo, T) - s.implied_vol_k(hi, T)) - float(
                    base.implied_vol_k(lo, T) - base.implied_vol_k(hi, T)
                )
                assert 100 * d90 == pytest.approx(size * rota_skew_vol_points(T), rel=1e-12)
        both = SurfacePerturbation(
            "composite",
            {"items": [dataclasses.asdict(rotation_perturbation(x)) for x in (0.7, -0.7)]},
        )
        rt = perturbed_surface(both, base)
        for T in (0.1, 1.0, 3.0):
            assert np.allclose(rt.implied_vol_k(k, T), base.implied_vol_k(k, T), atol=1e-15)


def test_rotated_refit_under_the_policies() -> None:
    """SPX 2022-12-30, ``(ssr 1, eps 0.10)``: the ±1 rota states and refits under each policy.
    SABR-linked: the rotated targets' ``Skew_SABR`` moves by ∓ one rota at every pillar and ``Corr_BE``
    moves with it.  Sticky break-even: ``SpotVolCovar`` / ``VolVar`` / ``Corr_BE`` targets equal the
    base fit's exactly, the two-point ``Skew_SABR`` reference is the rotated surface's; the variant
    also holds that reference.  ν rises with the rotation under both owner policies (measured
    SABR-linked 1.736 / 1.940 / 2.155 and sticky 1.621 / 1.940 / 2.273 at −1 / 0 / +1 rota: the
    sticky refit moves ν more per rota); the ``+0.7`` then ``−0.7`` round-trip surface refits to the
    base parameters within 1e-4 relative; an unknown policy raises."""
    spec = _spec("spx")
    cfg = BreakEvenFitConfig(skew_eps=0.10)
    states, fits, size = rotation_states(spec, cfg, ssr_target=1.0)
    assert size == 1.0 and set(states) == {"base", "up", "down", "up_refit", "down_refit"}
    base, up, dn = fits["base"], fits["up"], fits["down"]
    assert states["base"].spec.model == base.params
    assert states["up_refit"].spec.model == up.params and states["up"].spec.model == base.params
    assert states["up_refit"].spec.perturbation == states["up"].spec.perturbation
    T = base.targets.pillars
    slope = np.array([rota_slope(t) for t in T])
    assert np.allclose(up.targets.skew_target - base.targets.skew_target, -slope, rtol=1e-4)
    assert np.allclose(dn.targets.skew_target - base.targets.skew_target, slope, rtol=1e-4)
    assert np.all(np.abs(up.targets.correl_target - base.targets.correl_target) > 1e-4)
    assert dn.params.nu < base.params.nu < up.params.nu
    print("sabr_linked nu:", dn.params.nu, base.params.nu, up.params.nu)
    rot_up = surface_of(states["up"])
    rot_dn = surface_of(states["down"])
    sticky_up = refit_on_rotated(rot_up, base, cfg, ssr_target=1.0, policy="sticky_breakeven")
    sticky_dn = refit_on_rotated(rot_dn, base, cfg, ssr_target=1.0, policy="sticky_breakeven")
    for f in (sticky_up, sticky_dn):
        for name in ("spot_vol_covar", "vol_var", "vovol", "correl_target"):
            assert np.array_equal(getattr(f.targets, name), getattr(base.targets, name)), name
        assert any("sticky_breakeven" in fl for fl in f.targets.flags)
    assert np.allclose(sticky_up.targets.skew_target, up.targets.skew_target)  # rotated reference
    assert sticky_dn.params.nu < base.params.nu < sticky_up.params.nu
    print("sticky nu:", sticky_dn.params.nu, base.params.nu, sticky_up.params.nu)
    variant = refit_on_rotated(rot_up, base, cfg, ssr_target=1.0, policy="sticky_breakeven_skew")
    assert np.array_equal(variant.targets.skew_target, base.targets.skew_target)
    assert np.array_equal(variant.targets.spot_vol_covar, base.targets.spot_vol_covar)
    assert any("sticky_breakeven_skew" in fl for fl in variant.targets.flags)
    with pytest.raises(ValueError):
        refit_on_rotated(rot_up, base, cfg, ssr_target=1.0, policy="old")
    with pytest.raises(ValueError):
        rotation_states(spec, cfg, ssr_target=1.0, policy="old")
    both = SurfacePerturbation(
        "composite",
        {"items": [dataclasses.asdict(rotation_perturbation(x)) for x in (0.7, -0.7)]},
    )
    again = fit_2f_marking(surface_of(RiskState(spec).with_perturbation(both)), cfg, ssr_target=1.0)
    for fld in dataclasses.fields(BergomiParams):
        assert getattr(again.params, fld.name) == pytest.approx(
            getattr(base.params, fld.name), rel=1e-4, abs=1e-8
        )


def test_sign_test_on_the_recorded_study() -> None:
    """Report decision ix on the recorded study numbers (``rotation_spx_<policy>.yaml``, 2·10⁵
    paths, 2·10⁵-particle calibrations; skipped when absent): shadow = recalibrated − usual for
    each policy; the usual rotation is the same under both (same base set, same rotated
    surfaces); the owner's expected ordering ``|shadow_sticky| >> |shadow_sabr|`` holds in
    magnitude (ratio about 1.4, asserted > 1) but **the sticky shadow is positive, not negative**:
    under both policies the recalibrated rotation is smaller in magnitude than the usual one (the
    refit set partly offsets the skew-up move), most so under sticky break-evens.  Asserted as
    measured; the variant (break-evens and skew reference held) is printed when recorded."""
    recs = {p: _record(p) for p in RECALIBRATION_POLICIES}
    if any(recs[p] is None for p in OWNER_POLICIES):
        pytest.skip("rotation records absent (scripts/m7_p1_marking.py)")
    out = {}
    for p, doc in recs.items():
        if doc is None:
            continue
        usual, recal, shadow = (np.array(doc[k]) for k in ("usual", "recalibrated", "shadow"))
        assert shadow[0] == pytest.approx(recal[0] - usual[0], abs=1e-12)
        assert usual[1] > 0 and recal[1] > 0 and shadow[1] > 0
        out[p] = (usual, recal, shadow)
        print(p, "usual", usual, "recalibrated", recal, "shadow", shadow)
    u_s, r_s, sh_s = out["sabr_linked"]
    u_t, r_t, sh_t = out["sticky_breakeven"]
    assert u_s[0] == pytest.approx(u_t[0], abs=1e-12)  # same usual rotation
    assert u_s[0] < 0 and abs(u_s[0]) > 5 * u_s[1]  # the note is short skew
    assert abs(r_s[0]) < abs(u_s[0]) and abs(r_t[0]) < abs(u_t[0])  # recalibration offsets
    assert sh_s[0] > 3 * sh_s[1] and sh_t[0] > 3 * sh_t[1]  # both shadows positive
    assert abs(sh_t[0]) > abs(sh_s[0])  # the owner's magnitude ordering
    assert not sh_t[0] < 0, "the owner's expected negative sticky shadow is not reproduced"


@pytest.mark.parametrize("policy", OWNER_POLICIES)
def test_shadow_rotation_on_cached_calibrations(policy: str) -> None:
    """d(fee)/d(rota) of the M6 headline 3y autocall at ``(ssr 1, eps 0.10)`` on the cached
    calibrations of ``scripts/m7_p1_marking.py`` (4·10⁴ paths here, the study's 2·10⁵ in
    ``outputs/m7/p1_marking.md``): the refit sets equal the study's record, no cache miss, nothing
    recalibrated, finite sensitivities with standard errors, shadow = recalibrated − usual path by
    path.  The numbers are printed, not asserted (Monte Carlo at the test's path count)."""
    doc = _record(policy)
    if doc is None:
        pytest.skip(f"rotation_spx_{policy}.yaml absent (scripts/m7_p1_marking.py)")
    spec = _spec("spx")
    cfg = BreakEvenFitConfig(skew_eps=float(doc["skew_eps"]))
    _, fits, _ = rotation_states(spec, cfg, ssr_target=float(doc["ssr_target"]), policy=policy)
    for name, f in fits.items():
        assert f.params == from_mapping(BergomiParams, doc["fits"][name]), name
    fc, _, _ = build_market(spec)
    product = headline_products(fc.rate_curve, spec.market.spot)[AUTOCALL_NAME]
    sim = SimConfig(n_paths=40_000, chunk_size=40_000, seed=2024)
    try:
        rep = rotation_shadow_sensitivity(
            product,
            spec,
            cfg,
            ssr_target=float(doc["ssr_target"]),
            cache=LeverageCache(ROOT / "cache"),
            pricing_sim=sim,
            size=float(doc["size"]),
            allow_calibrate=False,
            product_name=f"{AUTOCALL_NAME} on spx",
            policy=policy,
        )
    except CacheMissError as exc:
        pytest.skip(f"rotated calibrations not in the cache (tests never calibrate): {exc}")
    print(rep.summary())
    assert rep.policy == policy
    assert rep.n_cache_misses == 0 and not rep.recalibrated_any and rep.n_calibrations == 5
    for s in (rep.fee, rep.usual, rep.recalibrated, rep.shadow):
        assert np.isfinite(s.value) and s.stderr > 0
    assert rep.shadow.value == pytest.approx(rep.recalibrated.value - rep.usual.value, abs=1e-12)
    assert 0.8 < rep.fee.value < 1.1
    assert set(rep.frame()["greek"]) == {"usual", "recalibrated", "shadow"}
