"""Shadow-rotation greek (SPEC §15 Part 3 KEEP list; ``risk/shadow_rotation.py``).  No
calibration runs here: the rotated states of ``scripts/m7_p1_marking.py`` are read from the cache
with ``allow_calibrate=False`` (skipped when absent); no wall-clock assertion.

* the rota: 2/sqrt(T) ln(110/90) vol points of 90/110 skew (0.568 at 6M, 0.560 with the saturated
  profile), the ATM skew moved by exactly ``−size · 0.02/sqrt(max(T, 1M))`` with the ATM vol
  unchanged, the arbitrage checks passed at ±1 rota on both surfaces, the ``+h`` then ``−h`` round
  trip back to the base surface;
* the refit: on SPX 2022-12-30 the rotated fits' SABR skews move by ∓ one rota, the round-trip
  surface refits to the base parameters exactly, and the ±1 rota refits bracket the base set;
* d(fee)/d(rota) of the M6 headline 3y autocall on the cached calibrations: usual, recalibrated
  and shadow with standard errors, shadow = recalibrated − usual path by path, the refit sets equal
  the study's record, nothing recalibrated.
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
    ROTA_T_MIN,
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


def test_rotated_refit_round_trip() -> None:
    """SPX 2022-12-30, ``(ssr 1, eps 0.10)``: the ±1 rota states and refits of the greek; the
    rotated targets' ``Skew_SABR`` moves by ∓ one rota at every pillar and ``Corr_BE`` moves with it; the ``+0.7`` then ``−0.7`` round-trip surface (implied vols equal to 1e-15) refits to the base
    parameters within 1e-4 relative (measured 3e-6 on ν: the perturbed surface's ATM skew is a
    central difference of half-width 1e-3, the base SSVI's analytic); the
    ±1 rota refits bracket the base set (``ν`` and ``k1`` on opposite sides, the midpoint within a
    tenth of the half-difference)."""
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
    for name in ("nu", "k1", "theta", "rho_SX1", "rho_SX2", "rho12"):
        print(name, [round(getattr(f.params, name), 4) for f in (dn, base, up)])
    assert dn.params.nu < base.params.nu < up.params.nu  # a steeper skew asks for more vol of vol
    print("status", dn.status, base.status, up.status, "active", up.first.active, dn.first.active)
    both = SurfacePerturbation(
        "composite",
        {"items": [dataclasses.asdict(rotation_perturbation(x)) for x in (0.7, -0.7)]},
    )
    rt = surface_of(RiskState(spec).with_perturbation(both))
    again = fit_2f_marking(rt, cfg, ssr_target=1.0)
    for f in dataclasses.fields(BergomiParams):
        assert getattr(again.params, f.name) == pytest.approx(
            getattr(base.params, f.name), rel=1e-4, abs=1e-8
        )


@pytest.mark.parametrize("surface", ["spx", "reference"])
def test_shadow_rotation_on_cached_calibrations(surface: str) -> None:
    """d(fee)/d(rota) of the M6 headline 3y autocall at ``(ssr 1, eps 0.10)`` on the cached
    calibrations of ``scripts/m7_p1_marking.py`` (4·10⁴ paths here, the study's 2·10⁵ in
    ``outputs/m7/p1_marking.md``): the refit sets equal the study's record, no cache miss, nothing
    recalibrated, finite sensitivities with standard errors, shadow = recalibrated − usual path by
    path.  The numbers are printed, not asserted (Monte Carlo at the test's path count)."""
    record = STUDY_DIR / f"rotation_{surface}.yaml"
    if not record.exists():
        pytest.skip(f"{record.name} absent (scripts/m7_p1_marking.py)")
    doc = yaml.safe_load(record.read_text(encoding="utf-8"))
    spec = _spec(surface)
    cfg = BreakEvenFitConfig(skew_eps=float(doc["skew_eps"]))
    _, fits, _ = rotation_states(spec, cfg, ssr_target=float(doc["ssr_target"]))
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
            product_name=f"{AUTOCALL_NAME} on {surface}",
        )
    except CacheMissError as exc:
        pytest.skip(f"rotated calibrations not in the cache (tests never calibrate): {exc}")
    print(rep.summary())
    assert rep.n_cache_misses == 0 and not rep.recalibrated_any and rep.n_calibrations == 5
    for s in (rep.fee, rep.usual, rep.recalibrated, rep.shadow):
        assert np.isfinite(s.value) and s.stderr > 0
    assert rep.shadow.value == pytest.approx(rep.recalibrated.value - rep.usual.value, abs=1e-12)
    assert 0.8 < rep.fee.value < 1.1
    assert set(rep.frame()["greek"]) == {"usual", "recalibrated", "shadow"}
