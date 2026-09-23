"""Shadow-rotation greek (SPEC §15 Part 3 KEEP list and report decision ix, restated in the
owner's declared convention; ``risk/shadow_rotation.py``).  No calibration runs here: the rotated
states of ``scripts/m7_p1_marking.py`` are read from the cache with ``allow_calibrate=False``
(skipped when absent); no wall-clock assertion.

* the rota: 2/sqrt(T) ln(110/90) vol points of 90/110 skew (0.568 at 6M, 0.560 with the saturated
  profile), the ATM skew moved by exactly ``−size · 0.02/sqrt(max(T, 1M))`` with the ATM vol
  unchanged, the arbitrage checks passed at ±1 rota on both surfaces, the ``+h`` then ``−h`` round
  trip back to the base surface;
* the refit under the two owner policies (``sabr_linked``, ``sticky_breakeven``) and the
  implementer's variant (``sticky_breakeven_skew``): on SPX 2022-12-30 the SABR-linked targets'
  ``Skew_SABR`` moves by ∓ one rota, the sticky targets hold ``SpotVolCovar`` / ``Corr_BE`` at the
  base values (the variant also the two-point skew reference), the round-trip surface refits to
  the base parameters, ν rises with the rotation under both policies, an unknown policy raises;
* ``sticky_breakeven``'s definition (owner's decision of 2026-09-16): the effective held set is
  ``{spot_vol_covar, correl_target}`` — the former five-array holding and the two-array one feed
  identical fits, each of the two moves the fit on its own, the three variance arrays alone are
  inert in marking mode;
* the convention algebra on a synthetic report (no cache): ``fee = P1 − LV``, ``fee_* = P1_* −
  lv_rotation``, ``fee_shadow = recalibrated − usual`` exactly (the LV rotation cancels), every
  ``desk_pnl_*`` the exact negative of its ``fee_*`` with the same standard error, the frame rows
  and the convention line;
* d(fee)/d(rota) of the M6 headline 3y autocall on the cached calibrations under both owner
  policies: the P1 / LV / fee levels, the LV rotation, the usual, recalibrated and shadow rotations
  of the P1 price, the fee and the desk P&L with standard errors, the refit sets equal the study's
  record, nothing recalibrated; the **sign test** (decision ix in the convention) on the recorded
  study numbers: the desk-P&L shadow for the short note is **negative** under ``sticky_breakeven``
  (and under ``sabr_linked``, reported), ``|desk_pnl_shadow(sticky)| > |desk_pnl_shadow(sabr)|``
  (measured ratio about 1.4 — the deck's ordering), the fee shadow equals the P1 shadow to
  round-off, the LV rotation is recorded with its stderr and the convention string is present.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml

from volsto.calibration.cache import CacheMissError, LeverageCache, build_market
from volsto.calibration.fit_2f import BreakEvenFitConfig, fit_2f_marking
from volsto.config import (
    BergomiParams,
    CalibrationSpec,
    SimConfig,
    SurfacePerturbation,
    from_mapping,
    load_yaml,
)
from volsto.market.loaders import snapshot_spec
from volsto.market.surface import atm_skew_numeric, perturbed_surface
from volsto.risk.engine import RiskState, Sensitivity, surface_of
from volsto.risk.shadow_rotation import (
    LV_STATES,
    P1_STATES,
    RECALIBRATION_POLICIES,
    ROTA_T_MIN,
    ROTATION_CONVENTION,
    ShadowRotationReport,
    refit_on_rotated,
    rota_skew_vol_points,
    rota_slope,
    rotation_perturbation,
    rotation_shadow_sensitivity,
    rotation_states,
    shadow_quantities,
)
from volsto.studies.m6 import AUTOCALL_NAME, headline_products

ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = ROOT / "configs" / "studies" / "m7_p1_marking"
SPX = ROOT / "configs" / "surfaces" / "snapshots" / "hdn_2022H2" / "spx_2022-12-30.yaml"
OWNER_POLICIES = ("sabr_linked", "sticky_breakeven")


def _spec(surface: str) -> CalibrationSpec:
    """The study's base calibration spec (``scripts/m7_p1_marking.py::base_spec``)."""
    ref = load_yaml(ROOT / "configs" / "studies" / "lsv_reference_2f.yaml", CalibrationSpec)
    if surface == "spx":
        # the one reader of a snapshot's surface (SPEC §13.2): reading the ``ssvi`` section
        # alone flattened the eSSVI anchor to its mean rho and fitted another surface
        ref = snapshot_spec(ref, SPX)
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
                # the rotated surface's own analytic ATM skew (not on the ImpliedSurface ABC)
                ds = float(cast(Any, s).atm_skew(T)) - float(atm_skew_numeric(base, T)[()])
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
    moves with it.  Sticky break-even: ``SpotVolCovar`` / ``Corr_BE`` targets equal the base
    fit's exactly, the two-point ``Skew_SABR`` reference and the (inert) variance targets are the
    rotated surface's; the variant also holds that reference.  ν rises with the rotation under both owner policies (measured
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
    # every correlation target moves with the rota except where it sits on the -1 bound (the
    # 1y pillar of the repaired eSSVI anchor: Corr_BE is clipped there before and after)
    d_corr = up.targets.correl_target - base.targets.correl_target
    at_bound = np.isclose(np.abs(base.targets.correl_target), 1.0)
    assert np.all(np.abs(d_corr[~at_bound]) > 1e-4) and np.all(d_corr[at_bound] == 0.0)
    assert dn.params.nu < base.params.nu < up.params.nu
    print("sabr_linked nu:", dn.params.nu, base.params.nu, up.params.nu)
    rot_up = surface_of(states["up"])
    rot_dn = surface_of(states["down"])
    sticky_up = refit_on_rotated(rot_up, base, cfg, ssr_target=1.0, policy="sticky_breakeven")
    sticky_dn = refit_on_rotated(rot_dn, base, cfg, ssr_target=1.0, policy="sticky_breakeven")
    for f in (sticky_up, sticky_dn):
        for name in ("spot_vol_covar", "correl_target"):
            assert np.array_equal(getattr(f.targets, name), getattr(base.targets, name)), name
        assert any("sticky_breakeven" in fl for fl in f.targets.flags)
    assert np.allclose(sticky_up.targets.skew_target, up.targets.skew_target)  # rotated reference
    for name in ("vol_var", "vovol"):  # the variance targets are the rotated surface's
        assert np.array_equal(getattr(sticky_up.targets, name), getattr(up.targets, name)), name
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


def test_sticky_breakeven_holds_the_effective_targets() -> None:
    """``sticky_breakeven`` holds exactly the targets a marking fit reads (owner's decision of
    2026-09-16; measured on SPX 2022-12-30 at +1 rota, ``(ssr 1, eps 0.10)``): the former holding
    of five arrays (``spot_vol_covar``, ``vol_var``, ``vovol``, ``vov_be_raw``,
    ``correl_target``) and :func:`held_targets`' two feed fits with **identical** parameters,
    step-2 / step-3 objectives and table columns — all but ``volvar_target_requested``, which
    only reports ``targets.vol_var`` —; holding the three variance arrays alone leaves the
    parameters of the unheld (rotated) fit unchanged, so they are inert; holding
    ``spot_vol_covar`` alone and ``correl_target`` alone each move the fit (``spot_vol_covar``
    through step 2's ``k1``, ``correl_target`` through step 3), so the owner's shorthand "holds
    the correlation target" is incomplete and the documented set is both.  Pure fits, no
    simulation."""
    from dataclasses import replace

    from volsto.calibration.fit_2f import fit_2f, marking_targets_for
    from volsto.market.varswap import xi0_curve
    from volsto.risk.shadow_rotation import held_targets

    spec = _spec("spx")
    cfg = BreakEvenFitConfig(skew_eps=0.10)
    base = fit_2f_marking(surface_of(RiskState(spec)), cfg, ssr_target=1.0)
    rot = surface_of(RiskState(spec).with_perturbation(rotation_perturbation(1.0)))
    moved = marking_targets_for(rot, cfg, ssr_target=1.0)
    c = replace(cfg, pillars=tuple(float(t) for t in moved.pillars))
    xi0 = xi0_curve(rot, float(min(rot.max_maturity, max(moved.pillars))))
    b = base.targets

    def fit_holding(*names: str):  # type: ignore[no-untyped-def]
        return fit_2f(replace(moved, **{k: getattr(b, k).copy() for k in names}), xi0, c)

    five = fit_holding("spot_vol_covar", "vol_var", "vovol", "vov_be_raw", "correl_target")
    two = fit_2f(held_targets(moved, b, "sticky_breakeven"), xi0, c)
    assert two.params == five.params
    assert two.first.objective == five.first.objective
    assert two.second.objective == five.second.objective
    differing = [
        col
        for col in five.table
        if not np.array_equal(five.table[col].to_numpy(), two.table[col].to_numpy())
    ]
    assert differing == ["volvar_target_requested"], differing
    unheld = fit_holding()
    assert fit_holding("vol_var", "vovol", "vov_be_raw").params == unheld.params
    svc, cor = fit_holding("spot_vol_covar"), fit_holding("correl_target")
    print(
        f"unheld {unheld.params}\nsvc only {svc.params}\ncorr only {cor.params}\n"
        f"sticky {two.params}"
    )
    for p in (svc.params, cor.params):
        assert p != unheld.params and p != two.params
    assert svc.params.k1 != unheld.params.k1 and cor.params.k1 == unheld.params.k1
    with pytest.raises(ValueError):
        held_targets(replace(moved, mode="historical"), b, "sticky_breakeven")


def test_convention_algebra_on_a_synthetic_report() -> None:
    """:func:`shadow_quantities` on synthetic per-path payoffs (no cache, no pricing): the fee
    level is ``P1 − LV`` path by path, ``fee_usual = usual − lv_rotation`` and
    ``fee_recalibrated = recalibrated − lv_rotation`` in value, ``fee_shadow`` equals
    ``recalibrated − usual`` (and the P1 ``shadow``) exactly with the same standard error, every
    ``desk_pnl_*`` is the exact negative of its ``fee_*`` with the same standard error, the
    standard errors are those of the per-path combinations; the report's frame lists the nine
    quantities in order, its summary starts with the convention line and states the policy;
    missing states raise."""
    rng = np.random.default_rng(3)
    n, size = 4_000, 0.5
    base = rng.normal(0.9, 0.05, n)
    p1 = {
        "base": base,
        "up": base - 0.02 + rng.normal(0, 1e-3, n),
        "down": base + 0.02 + rng.normal(0, 1e-3, n),
        "up_refit": base - 0.01 + rng.normal(0, 1e-3, n),
        "down_refit": base + 0.01 + rng.normal(0, 1e-3, n),
    }
    lv = {
        "base": base - 0.07 + rng.normal(0, 1e-2, n),
        "up": base - 0.07 - 0.03 + rng.normal(0, 1e-2, n),
        "down": base - 0.07 + 0.03 + rng.normal(0, 1e-2, n),
    }
    assert set(p1) == set(P1_STATES) and set(lv) == set(LV_STATES)
    q = shadow_quantities(p1, lv, size=size, n_paths=n)
    c = 1.0 / (2.0 * size)

    def se(x: np.ndarray) -> float:
        return float(x.std(ddof=1) / np.sqrt(n))

    fee_paths = p1["base"] - lv["base"]
    assert q["fee"].value == pytest.approx(fee_paths.mean(), abs=1e-14)
    assert q["fee"].stderr == pytest.approx(se(fee_paths), rel=1e-12)
    assert q["p1_level"].value - q["lv_level"].value == pytest.approx(q["fee"].value, abs=1e-14)
    lv_rot = c * (lv["up"] - lv["down"])
    assert q["lv_rotation"].value == pytest.approx(lv_rot.mean(), abs=1e-14)
    assert q["lv_rotation"].stderr == pytest.approx(se(lv_rot), rel=1e-12)
    for name in ("usual", "recalibrated"):
        assert q[f"fee_{name}"].value == pytest.approx(
            q[name].value - q["lv_rotation"].value, abs=1e-14
        )
    fee_usual_paths = c * (p1["up"] - p1["down"]) - lv_rot
    assert q["fee_usual"].stderr == pytest.approx(se(fee_usual_paths), rel=1e-12)
    assert q["fee_shadow"].value == pytest.approx(
        q["fee_recalibrated"].value - q["fee_usual"].value, abs=1e-14
    )
    assert q["fee_shadow"].value == pytest.approx(
        q["recalibrated"].value - q["usual"].value, abs=1e-14
    )
    assert (
        q["fee_shadow"].value == q["shadow"].value and q["fee_shadow"].stderr == q["shadow"].stderr
    )
    shadow_paths = c * (p1["up_refit"] - p1["down_refit"] - p1["up"] + p1["down"])
    assert q["shadow"].stderr == pytest.approx(se(shadow_paths), rel=1e-12)
    for name in ("usual", "recalibrated", "shadow"):
        d, f = q[f"desk_pnl_{name}"], q[f"fee_{name}"]
        assert d.value == -f.value and d.stderr == f.stderr and d.unit == f.unit
        assert d.states == f.states and d.size == f.size and d.scheme == f.scheme
    assert q["usual"].scheme == "central" and q["fee"].scheme == "level"
    assert all(v.n_paths == n for v in q.values())
    rep = ShadowRotationReport(
        product="synthetic",
        policy="sticky_breakeven",
        size=size,
        convention=ROTATION_CONVENTION,
        p1_level=q["p1_level"],
        lv_level=q["lv_level"],
        fee=q["fee"],
        lv_rotation=q["lv_rotation"],
        usual=q["usual"],
        recalibrated=q["recalibrated"],
        shadow=q["shadow"],
        fee_usual=q["fee_usual"],
        fee_recalibrated=q["fee_recalibrated"],
        fee_shadow=q["fee_shadow"],
        desk_pnl_usual=q["desk_pnl_usual"],
        desk_pnl_recalibrated=q["desk_pnl_recalibrated"],
        desk_pnl_shadow=q["desk_pnl_shadow"],
        fits={},
        param_moves={},
        per_vol_point={0.5: rota_skew_vol_points(0.5)},
        n_calibrations=0,
        n_lv_builds=0,
        n_cache_misses=0,
        recalibrated_any=False,
        wall_seconds=0.0,
    )
    fr = rep.frame()
    assert list(fr["greek"]) == [
        "lv_rotation",
        "usual",
        "recalibrated",
        "fee_usual",
        "fee_recalibrated",
        "fee_shadow",
        "desk_pnl_usual",
        "desk_pnl_recalibrated",
        "desk_pnl_shadow",
    ]
    row = fr.set_index("greek").loc["desk_pnl_shadow"]
    assert row["per_rota"] == -q["fee_shadow"].value and row["per_rota_se"] == q["shadow"].stderr
    assert row["per_vp_90_110_0.5y"] == pytest.approx(
        -q["fee_shadow"].value / rota_skew_vol_points(0.5)
    )
    text = rep.summary()
    assert text.startswith("convention: " + ROTATION_CONVENTION)
    assert "policy sticky_breakeven" in text and "fee = P1 - LV" in text
    assert "0.56 vp" in ROTATION_CONVENTION and "SHORT position = -(d fee)" in ROTATION_CONVENTION
    assert isinstance(q["fee"], Sensitivity)
    with pytest.raises(ValueError):
        shadow_quantities({k: v for k, v in p1.items() if k != "up_refit"}, lv, size=1.0, n_paths=n)


def test_sign_test_on_the_recorded_study() -> None:
    """Report decision ix in the owner's declared convention, on the recorded study numbers
    (``rotation_spx_<policy>.yaml``, 2·10⁵ paths, 2·10⁵-particle calibrations; skipped when
    absent).  The desk-P&L shadow per +1 rota for the short note is **negative** under
    ``sticky_breakeven`` (the fee rises when the marking set follows a skew-up rotation — the
    deck's "fee increases, negative P&L") and, reported, under ``sabr_linked`` too;
    ``|desk_pnl_shadow(sticky_breakeven)| > |desk_pnl_shadow(sabr_linked)|`` (measured ratio about
    1.4, printed — the deck's ordering); ``fee_shadow`` equals the P1 shadow ``recalibrated −
    usual`` to round-off; ``fee_usual = usual − lv_rotation``; the usual P1 rotation is the same
    under both policies (same base set, same rotated surfaces); ``lv_rotation`` is recorded with a
    positive stderr; the convention string is present.  The variant (break-evens and skew
    reference held) is printed when recorded."""
    recs = {p: _record(p) for p in RECALIBRATION_POLICIES}
    if any(recs[p] is None for p in OWNER_POLICIES):
        pytest.skip("rotation records absent (scripts/m7_p1_marking.py)")
    keys = (
        "lv_rotation",
        "usual",
        "recalibrated",
        "shadow",
        "fee_usual",
        "fee_recalibrated",
        "fee_shadow",
        "desk_pnl_usual",
        "desk_pnl_recalibrated",
        "desk_pnl_shadow",
    )
    out: dict[str, dict[str, np.ndarray]] = {}
    for p, doc in recs.items():
        if doc is None:
            continue
        if "desk_pnl_shadow" not in doc:
            pytest.skip(f"rotation_spx_{p}.yaml predates the convention (re-run the script)")
        assert doc["convention"] == ROTATION_CONVENTION
        q = {k: np.array(doc[k], dtype=float) for k in keys}
        for k in ("p1_level", "lv_level", "fee"):
            v = np.array(doc[k], dtype=float)
            assert np.isfinite(v).all() and v[1] > 0, k
        assert np.array(doc["fee"])[0] == pytest.approx(
            np.array(doc["p1_level"])[0] - np.array(doc["lv_level"])[0], abs=1e-12
        )
        assert all(q[k][1] > 0 for k in keys)
        assert q["shadow"][0] == pytest.approx(q["recalibrated"][0] - q["usual"][0], abs=1e-12)
        assert q["fee_shadow"][0] == pytest.approx(q["shadow"][0], abs=1e-12)
        assert q["fee_shadow"][1] == pytest.approx(q["shadow"][1], abs=1e-12)
        assert q["fee_usual"][0] == pytest.approx(q["usual"][0] - q["lv_rotation"][0], abs=1e-12)
        assert q["fee_recalibrated"][0] == pytest.approx(
            q["recalibrated"][0] - q["lv_rotation"][0], abs=1e-12
        )
        assert q["fee_shadow"][0] == pytest.approx(
            q["fee_recalibrated"][0] - q["fee_usual"][0], abs=1e-12
        )
        for name in ("usual", "recalibrated", "shadow"):
            assert q[f"desk_pnl_{name}"][0] == -q[f"fee_{name}"][0]
            assert q[f"desk_pnl_{name}"][1] == q[f"fee_{name}"][1]
        out[p] = q
        print(
            p,
            *(f"{k} {q[k][0]:+.6f} +/- {q[k][1]:.6f}" for k in keys),
            f"desk_pnl_shadow z {q['desk_pnl_shadow'][0] / q['desk_pnl_shadow'][1]:+.1f}",
            sep="\n  ",
        )
    sabr, sticky = out["sabr_linked"], out["sticky_breakeven"]
    assert sabr["usual"][0] == pytest.approx(sticky["usual"][0], abs=1e-12)  # same usual rotation
    assert sabr["lv_rotation"][0] == pytest.approx(sticky["lv_rotation"][0], abs=1e-12)
    assert sabr["usual"][0] < 0 and abs(sabr["usual"][0]) > 5 * sabr["usual"][1]  # short skew
    d_sticky, d_sabr = sticky["desk_pnl_shadow"], sabr["desk_pnl_shadow"]
    assert d_sticky[0] < -3 * d_sticky[1], "desk-P&L shadow not negative under sticky_breakeven"
    print("sabr_linked desk-P&L shadow negative:", bool(d_sabr[0] < 0), d_sabr)
    ratio = abs(d_sticky[0]) / abs(d_sabr[0])
    print(f"|desk_pnl_shadow| sticky / sabr = {ratio:.3f}")
    assert ratio > 1.0, ratio  # the deck's ordering (measured about 1.4)


@pytest.mark.parametrize("policy", OWNER_POLICIES)
def test_shadow_rotation_on_cached_calibrations(policy: str) -> None:
    """d(fee)/d(rota) of the M6 headline 3y autocall at ``(ssr 1, eps 0.10)`` on the cached
    calibrations of ``scripts/m7_p1_marking.py`` (4·10⁴ paths here, the study's 2·10⁵ in
    ``outputs/m7/p1_marking.md``): the refit sets equal the study's record, no cache miss, nothing
    recalibrated (five leverage states, three LV builds), finite levels and sensitivities with
    standard errors in the declared convention, shadow = recalibrated − usual path by path,
    ``fee_shadow`` the P1 shadow, ``desk_pnl_*`` = −``fee_*``.  The numbers are printed, not
    asserted (Monte Carlo at the test's path count)."""
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
    assert rep.policy == policy and rep.convention == ROTATION_CONVENTION
    assert rep.n_cache_misses == 0 and not rep.recalibrated_any and rep.n_calibrations == 5
    assert rep.n_lv_builds == 3
    for s in (rep.p1_level, rep.lv_level, rep.fee, rep.lv_rotation, *(r[2] for r in rep.rows())):
        assert np.isfinite(s.value) and s.stderr > 0, s
    assert rep.shadow.value == pytest.approx(rep.recalibrated.value - rep.usual.value, abs=1e-12)
    assert rep.fee_shadow.value == rep.shadow.value and rep.fee_shadow.stderr == rep.shadow.stderr
    assert rep.fee_usual.value == pytest.approx(rep.usual.value - rep.lv_rotation.value, abs=1e-12)
    assert rep.desk_pnl_shadow.value == -rep.fee_shadow.value
    assert rep.desk_pnl_usual.value == -rep.fee_usual.value
    assert rep.desk_pnl_recalibrated.value == -rep.fee_recalibrated.value
    assert 0.8 < rep.p1_level.value < 1.1 and 0.8 < rep.lv_level.value < 1.1
    assert rep.fee.value == pytest.approx(rep.p1_level.value - rep.lv_level.value, abs=1e-12)
    assert len(rep.frame()) == 9 and "desk_pnl_shadow" in set(rep.frame()["greek"])
