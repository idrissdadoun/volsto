"""Skew-shock world of M8b study C (SPEC §8.1 "Skew-shock world"; ``volsto/hedging/worlds.py``).
Fast and calibration-free: the blends run on synthetic leverages / local vols, the LSV simulation
uses the reference 2F parameters over a flat ``ξ₀`` with constant leverages; the two cache-backed
tests read a leverage cache with ``allow_calibrate=False`` — the repo cache, asserting the miss is
raised cleanly, and a temporary cache holding stored constant leverages for the end-to-end
construction (neither calibrates).  Every Monte Carlo number is printed with its standard error."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from volsto.calibration.cache import CacheMissError, LeverageCache, build_market
from volsto.config import CalibrationSpec, SimConfig, load_yaml
from volsto.engine.mc import MonteCarlo
from volsto.hedging.worlds import (
    DAYS_PER_YEAR,
    SHOCK_DAYS,
    blend_leverage,
    blend_local_vol,
    blend_times,
    shock_state,
    shock_weights,
    shocked_surface,
    skew_90_110,
    skew_shock_world,
)
from volsto.market.curves import ForwardCurve
from volsto.market.dupire import LocalVolSurface
from volsto.market.varswap import ForwardVarianceCurve
from volsto.models.bergomi import BergomiSV
from volsto.models.leverage import LeverageFunction
from volsto.models.lsv import LSV
from volsto.products.vanilla import EuropeanOption
from volsto.risk.engine import RiskState
from volsto.risk.shadow_rotation import rota_skew_vol_points

ROOT = Path(__file__).resolve().parents[1]
SPEC_2F = ROOT / "configs" / "studies" / "lsv_reference_2f.yaml"
T0 = 0.45  # not a slice of the synthetic grids below: both boundaries must be added
WINDOW = SHOCK_DAYS / DAYS_PER_YEAR
T_END = T0 + WINDOW


@pytest.fixture(scope="module")
def fc() -> ForwardCurve:
    return ForwardCurve.flat(100.0, 0.02, 0.01)


def _constant_leverage(fc: ForwardCurve, value: float, times: np.ndarray) -> LeverageFunction:
    k = np.arange(-2.5, 2.5 + 1e-12, 0.05)
    return LeverageFunction(times, k, np.full((times.size, k.size), value), fc, {"seed": 0})


def _reference_state(n_particles: int = 800_000) -> RiskState:
    spec = load_yaml(SPEC_2F, CalibrationSpec)
    spec = dataclasses.replace(
        spec, particle=dataclasses.replace(spec.particle, n_particles=n_particles)
    )
    return RiskState(spec)


# --------------------------------------------------------------------------------------------
# (i) blend weights, boundaries, slices
# --------------------------------------------------------------------------------------------


def test_blend_leverage_is_base_before_rotated_after_linear_between(fc: ForwardCurve) -> None:
    times = np.linspace(0.0, 1.0, 11)
    base = _constant_leverage(fc, 1.0, times)
    rot = _constant_leverage(fc, 1.2, times)
    lev = blend_leverage(base, rot, t0=T0, t_end=T_END)
    # slices added at both boundaries, grid otherwise untouched
    assert lev.n_slices == times.size + 2
    assert lev.metadata["blend"]["slices_added"] == 2
    assert np.any(lev.times == T0) and np.any(lev.times == T_END)
    assert np.array_equal(lev.k_grid, base.k_grid)
    # exact at the boundaries (the added slices hold w = 0 and w = 1 exactly)
    i0, i1 = int(np.searchsorted(lev.times, T0)), int(np.searchsorted(lev.times, T_END))
    assert np.all(lev.values[i0] == 1.0) and np.all(lev.values[i1] == 1.2)
    # base before, rotated after, linear between
    probe = np.array(
        [0.0, 0.3, T0 - 1e-6, T0, T0 + 0.25 * WINDOW, T0 + 0.5 * WINDOW, T_END, 0.9, 5.0]
    )
    got = lev.rows(probe)[:, 0]
    want = 1.0 + 0.2 * shock_weights(probe, T0, T_END)
    np.testing.assert_allclose(got, want, rtol=0.0, atol=1e-12)
    print(
        "blend at t = 0.3 / t0 / t0+w/4 / t0+w/2 / t_end / 0.9:",
        np.round(lev.rows([0.3, T0, T0 + 0.25 * WINDOW, T0 + 0.5 * WINDOW, T_END, 0.9])[:, 0], 6),
    )
    # the LSV model requires the boundary slices on its grid
    kernel = BergomiSV(
        load_yaml(SPEC_2F, CalibrationSpec).model, ForwardVarianceCurve.flat(0.04), fc
    )
    req = LSV(kernel, lev).required_times()
    assert T0 in req and T_END in req


def test_blend_times_reuses_an_existing_slice() -> None:
    times = np.array([0.0, 0.5, 1.0])
    grid, n_added = blend_times(times, 0.5, 0.5 + WINDOW)
    assert n_added == 1 and grid.size == 4 and np.all(np.diff(grid) > 0)
    grid, n_added = blend_times(times, 0.5 + 1e-12, 1.0)  # within tolerance of both slices
    assert n_added == 0 and np.array_equal(grid, times)


def test_blend_leverage_rejects_mismatched_grids(fc: ForwardCurve) -> None:
    times = np.linspace(0.0, 1.0, 11)
    base = _constant_leverage(fc, 1.0, times)
    other_times = _constant_leverage(fc, 1.2, np.linspace(0.0, 1.0, 21))
    with pytest.raises(ValueError, match="slice times"):
        blend_leverage(base, other_times, t0=T0, t_end=T_END)
    k = np.arange(-2.0, 2.0 + 1e-12, 0.05)
    other_k = LeverageFunction(times, k, np.full((times.size, k.size), 1.2), fc)
    with pytest.raises(ValueError, match="k grid"):
        blend_leverage(base, other_k, t0=T0, t_end=T_END)
    with pytest.raises(ValueError, match="t0 < t_end"):
        shock_weights(times, 0.5, 0.5)


def test_blend_local_vol_is_linear_in_variance_with_boundary_slices(fc: ForwardCurve) -> None:
    t = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    k = np.linspace(-1.0, 1.0, 5)
    base = LocalVolSurface(t, k, np.full((5, 5), 0.20**2), fc)
    rot = LocalVolSurface(t, k, np.full((5, 5), 0.24**2), fc)
    lv = blend_local_vol(base, rot, t0=T0, t_end=T_END)
    assert lv.t_grid.size == 7
    # the surface interpolates the variance linearly between slices: the blend is linear in the
    # variance across the whole window (the vol is not: 0.22² = 0.0484 would need a slice there)
    mid = lv.var_at_times([T0 + 0.5 * WINDOW])[0, 0]
    assert mid == pytest.approx(0.5 * (0.04 + 0.0576), rel=1e-12)
    q = lv.var_at_times([T0 + 0.25 * WINDOW])[0, 0]
    assert q == pytest.approx(0.75 * 0.04 + 0.25 * 0.0576, rel=1e-12)
    print(
        "local variance at t0 / t0+w/4 / t0+w/2 / t_end:",
        lv.var_at_times([T0, T0 + 0.25 * WINDOW, T0 + 0.5 * WINDOW, T_END])[:, 0],
    )
    assert lv.var_at_times([0.1])[0, 0] == pytest.approx(0.04) and lv.var_at_times([0.9])[
        0, 0
    ] == pytest.approx(0.0576)


# --------------------------------------------------------------------------------------------
# (ii) the blended LSV simulates; its price sits between the two endpoints
# --------------------------------------------------------------------------------------------


def test_blended_lsv_simulates_between_the_endpoints(fc: ForwardCurve) -> None:
    params = load_yaml(SPEC_2F, CalibrationSpec).model
    kernel = BergomiSV(params, ForwardVarianceCurve.flat(0.04), fc)
    times = np.linspace(0.0, 1.5, 16)
    base = _constant_leverage(fc, 1.0, times)
    rot = _constant_leverage(fc, 1.2, times)
    world = LSV(kernel, base).bump(leverage=blend_leverage(base, rot, t0=T0, t_end=T_END))
    assert isinstance(world, LSV) and world.kernel is kernel  # the kernel is untouched
    sim = SimConfig(n_paths=20_000, chunk_size=20_000, seed=11, dt_max=1.0 / 52.0)
    call = EuropeanOption(100.0, 1.0, "call", fc.rate_curve)
    mc = MonteCarlo(sim)
    prices = {
        name: mc.price(call, m)
        for name, m in (
            ("base", LSV(kernel, base)),
            ("world", world),
            ("rotated", LSV(kernel, rot)),
        )
    }
    for name, r in prices.items():
        print(f"1y ATM call under {name}: {r.mean:.4f} +/- {r.stderr:.4f}")
        assert np.isfinite(r.mean) and r.stderr > 0
    lo, hi, mid = prices["base"], prices["rotated"], prices["world"]
    tol = 3.0 * mid.stderr
    assert lo.mean - tol < mid.mean < hi.mean + tol
    assert hi.mean - lo.mean > 5.0 * hi.stderr  # the endpoints are distinguishable


# --------------------------------------------------------------------------------------------
# (iii) the ξ₀-rescale identity: LSV(kernel_A, L) == LSV(kernel_B, L sqrt(ξ₀_A/ξ₀_B)) in law
# --------------------------------------------------------------------------------------------


def test_xi0_rescale_identity_on_two_kernels(fc: ForwardCurve) -> None:
    params = load_yaml(SPEC_2F, CalibrationSpec).model
    T = np.array([0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
    xi_a = ForwardVarianceCurve.flat(0.04)
    xi_b = ForwardVarianceCurve(T, 0.04 * T + 0.005 * T * T)  # ξ₀ ≈ 0.04 + 0.01 t, +25%/yr
    ka, kb = BergomiSV(params, xi_a, fc), BergomiSV(params, xi_b, fc)
    times = np.linspace(0.0, 1.0, 53)
    l_a = _constant_leverage(fc, 1.0, times)
    ratio = np.sqrt(np.asarray(xi_a.xi0(times)) / np.asarray(xi_b.xi0(times)))
    l_b = l_a.with_values(l_a.values * ratio[:, None])
    ma, mb = LSV(ka, l_a), LSV(kb, l_b)
    sim = SimConfig(n_paths=20_000, chunk_size=20_000, seed=5, dt_max=1.0 / 252.0)
    call = EuropeanOption(100.0, 1.0, "call", fc.rate_curve)
    mc = MonteCarlo(sim)
    ga, gb = mc.build_grid([call], ma), mc.build_grid([call], mb)
    assert np.array_equal(ga.times, gb.times)  # same grid, same draws: CRN
    ra = mc.price(call, ma, grid=ga, keep_payoffs=True)
    rb = mc.price(call, mb, grid=gb, keep_payoffs=True)
    pa, pb = np.asarray(ra.payoffs), np.asarray(rb.payoffs)
    d = 0.5 * (pa[0::2] + pa[1::2]) - 0.5 * (pb[0::2] + pb[1::2])
    diff, se = float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size))
    print(
        f"1y ATM call: kernel A {ra.mean:.4f} +/- {ra.stderr:.4f}, kernel B rescaled "
        f"{rb.mean:.4f} +/- {rb.stderr:.4f}, CRN difference {diff:.5f} +/- {se:.5f}"
    )
    # identical in law; the residual is the within-step treatment of the ratio (O(δt)): the
    # bound is 0.1% of the price, ~0.01 vp
    assert abs(diff) < 1e-3 * ra.mean


# --------------------------------------------------------------------------------------------
# (iv) the rotated surface: skew, ATM and ξ₀ moves (the docstring's numbers)
# --------------------------------------------------------------------------------------------


def test_shocked_surface_moves_skew_keeps_atm_and_moves_xi0() -> None:
    base = _reference_state()
    _, s0, k0 = build_market(base.spec)
    T = np.array([1 / 12, 0.5, 1.0, 5.0])
    for rota in (1.0, 3.0):
        s1 = shocked_surface(s0, rota)
        d_skew = skew_90_110(s1, 0.5) - skew_90_110(s0, 0.5)
        assert d_skew == pytest.approx(rota * rota_skew_vol_points(0.5), rel=1e-9)
        np.testing.assert_allclose(s1.atm_vol(T), s0.atm_vol(T), rtol=0.0, atol=1e-15)
        _, _, k1 = build_market(shock_state(base, rota).spec)
        rel = np.asarray(k1.xi0.xi0(T)) / np.asarray(k0.xi0.xi0(T)) - 1.0
        print(
            f"rota {rota:+g}: 6M 90/110 skew {skew_90_110(s0, 0.5):.3f} -> {skew_90_110(s1, 0.5):.3f} vp; "
            f"xi0 relative change at 1M/6M/1Y/5Y {np.round(rel, 4)}"
        )
        # the strip integrates the steepened put wing: ξ₀ rises, not preserved (module docstring)
        assert np.all(rel > 0.0) and np.all(rel < 0.25 * rota)
        assert np.all(np.diff(rel) < 0)  # largest at the short end


# --------------------------------------------------------------------------------------------
# (v) cache-backed: the miss is raised cleanly, nothing is calibrated
# --------------------------------------------------------------------------------------------


def test_skew_shock_world_raises_cache_miss_without_calibrating() -> None:
    base = _reference_state()
    cache = LeverageCache(ROOT / "cache")
    if not cache.has(base.spec):
        pytest.skip(f"base leverage absent from the cache (tests never calibrate): {base.key}")
    rot = shock_state(base, 1.0)
    if cache.has(rot.spec):
        world, meta = skew_shock_world(base, cache, rota=1.0, t0=1.0, allow_calibrate=False)
        assert isinstance(world, LSV) and meta["calibrated"] is False
        print({k: v for k, v in meta.items() if not k.endswith("key")})
        return
    with pytest.raises(CacheMissError):
        skew_shock_world(base, cache, rota=1.0, t0=1.0, allow_calibrate=False)
    assert not cache.has(rot.spec)  # still absent: nothing was calibrated


def test_skew_shock_world_end_to_end_on_a_temporary_cache(tmp_path: Path) -> None:
    """The full construction without any calibration: constant leverages (1.0 base, 1.1 on the
    rotated surface) stored under the two cache keys of a temporary cache; the world is an LSV
    whose leverage is 1.0 before ``t0``, the rescaled 1.1·sqrt(ξ₀_rot/ξ₀_base) after ``t_end``
    (``xi0_rescale=True``) and 1.1 exactly with ``xi0_rescale=False``; ``calibrated`` is False and
    the boundary slices are added."""
    base = _reference_state(20_000)
    rot = shock_state(base, 1.0)
    cache = LeverageCache(tmp_path / "cache")
    fc = build_market(base.spec)[0]
    times = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    cache.store(base.spec, _constant_leverage(fc, 1.0, times))
    cache.store(rot.spec, _constant_leverage(fc, 1.1, times))
    t0 = 0.9
    world, meta = skew_shock_world(base, cache, rota=1.0, t0=t0, allow_calibrate=False)
    assert isinstance(world, LSV) and meta["calibrated"] is False and meta["n_calibrations"] == 0
    t_end = meta["t_end"]
    assert t_end == pytest.approx(t0 + SHOCK_DAYS / DAYS_PER_YEAR)
    lev = world.leverage
    assert t0 in set(np.round(lev.times, 12)) and round(t_end, 12) in set(np.round(lev.times, 12))
    assert meta["slices_added"] == 2
    # before the shock: the base leverage; after it: the rotated one rescaled by the xi0 ratio
    before = lev.rows(np.array([0.5]))[0]
    assert np.allclose(before, 1.0)
    _, _, k_base = build_market(base.spec)
    _, _, k_rot = build_market(rot.spec)
    for t in (1.5, 2.0, 3.0):
        ratio = float(k_rot.xi0.xi0(np.array([t]))[0] / k_base.xi0.xi0(np.array([t]))[0])
        assert ratio > 1.0  # the rotation steepens the put wing: the VS strip rises
        assert np.allclose(lev.rows(np.array([t]))[0], 1.1 * np.sqrt(ratio), rtol=1e-12)
    raw, meta_raw = skew_shock_world(
        base, cache, rota=1.0, t0=t0, allow_calibrate=False, xi0_rescale=False
    )
    assert (
        np.allclose(raw.leverage.rows(np.array([2.0]))[0], 1.1) and meta_raw["calibrated"] is False
    )
    print({k: v for k, v in meta.items() if not str(k).endswith("key")})
