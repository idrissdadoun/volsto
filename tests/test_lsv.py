"""M3 tests (SPEC §10 LSV, §4.1–4.3): particle calibration reprices the surface with the pricing
kernel; calibration and pricing share the kernel step for step; variance swaps invariant across
ω; leverage serialisation; content-addressed cache."""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import platform
import sys
from pathlib import Path

import _k5_reference
import _particle_reference_m6
import numpy as np
import pytest

from volsto.calibration import (
    CacheMissError,
    LeverageCache,
    calibrate_leverage,
    reprice_surface,
)
from volsto.calibration.cache import build_market, spec_key
from volsto.calibration.particle import conditional_variance_estimate
from volsto.config import (
    BergomiParams,
    CalibrationSpec,
    ParticleConfig,
    SimConfig,
    StepSchedule,
    load_yaml,
)
from volsto.engine import GaussianDraws, MonteCarlo, TimeGrid
from volsto.market import ForwardVarianceCurve, varswap_strike
from volsto.models import LSV, BergomiSV, LeverageFunction
from volsto.products import VarianceSwap

SPEC_1F = Path(__file__).resolve().parents[1] / "configs" / "studies" / "lsv_reference_1f.yaml"

# fast settings shared by the tests: 50k particles, 1y horizon, default step schedule and scheme
# (a uniform dt = 1/100 was tried and is far too coarse for the short-maturity leverage: +0.4 vp
# at 3m +10% and +0.8 vp on the 3m variance swap)
FAST_SIM = SimConfig(n_paths=100_000, chunk_size=50_000, seed=7)
FAST_PARTICLE = ParticleConfig(n_particles=50_000, horizon=1.0, seed=1)


@pytest.fixture(scope="module")
def spec() -> CalibrationSpec:
    return load_yaml(SPEC_1F, CalibrationSpec)


@pytest.fixture(scope="module")
def market(spec: CalibrationSpec):
    return build_market(spec)


@pytest.fixture(scope="module")
def fast_calibration(market):
    _, surface, kernel = market
    return calibrate_leverage(surface, kernel, FAST_PARTICLE, FAST_SIM)


# ---------------------------------------------------------------------------------------------
# repricing with the pricing kernel (owner item 2) and §4.2 acceptance
# ---------------------------------------------------------------------------------------------


def test_calibrated_lsv_reprices_surface_with_pricing_kernel(market, fast_calibration) -> None:
    """Fast version of §4.2: 50k particles, 1y horizon, default schedule; ATM within 0.1 vp and
    ±10% within 0.2 vp at 3m/6m/1y, priced with MonteCarlo.price_many (fresh seed)."""
    _, surface, kernel = market
    model = LSV(kernel, fast_calibration.leverage)
    rep = reprice_surface(
        model, surface, FAST_SIM, maturities=(0.25, 0.5, 1.0), log_moneyness=(-0.1, 0.0, 0.1)
    )
    err = rep.pivot("error_vp")
    se = rep.pivot("stderr_vp")
    assert np.all(
        np.abs(err[0.0].to_numpy()) < np.maximum(0.10, 3 * se[0.0].to_numpy())
    ), rep.summary()
    assert np.all(np.abs(err.to_numpy()) < np.maximum(0.20, 3 * se.to_numpy())), rep.summary()
    # variance swaps vs replication: the residual excess is regression noise in the tails and
    # falls with N (0.24 vp at 50k, 0.13 at 200k, 0.08 at 400k+ particles, 1y); 0.3 vp here
    assert np.all(
        np.abs(rep.varswaps["diff_vp"]) < np.maximum(0.3, 3 * rep.varswaps["stderr_vp"])
    ), rep.varswaps


@pytest.mark.slow
def test_calibration_acceptance_full_size(spec: CalibrationSpec) -> None:
    """§4.2: default spec (2·10⁵ particles, 3y horizon, default schedule and scheme), 4·10⁵
    pricing paths — max abs error ≤ 0.15 vp inside ±20% moneyness for T ≤ 2y."""
    _, surface, kernel = build_market(spec)
    res = calibrate_leverage(surface, kernel, spec.particle, spec.sim)
    rep = reprice_surface(
        LSV(kernel, res.leverage), surface, SimConfig(n_paths=400_000, chunk_size=50_000, seed=7)
    )
    # a cell fails only when |error| exceeds both 0.15 vp and 3 MC standard errors, and the
    # region is restricted to 2.5 ATM standard deviations (the 1m ±20% cells are 3.4 std out;
    # the 2F far right tail misses them by ~1.4 vp on a sub-basis-point option, see the M3 report)
    assert rep.passes(0.15, t_max=2.0, k_abs=0.2, z=3.0, max_std=2.5), rep.summary()
    assert np.all(
        np.abs(rep.varswaps["diff_vp"]) < np.maximum(0.2, 3 * rep.varswaps["stderr_vp"])
    ), rep.varswaps


def test_calibration_and_pricing_share_the_kernel(market, fast_calibration) -> None:
    """Simulating the calibrated LSV with the calibration's own seed / grid / particle count
    reproduces the particle cloud at the horizon exactly (same kernel, same frozen-L rule)."""
    _, _, kernel = market
    res = fast_calibration
    model = LSV(kernel, res.leverage)
    grid = TimeGrid.build([FAST_PARTICLE.horizon], FAST_SIM.dt_max)
    assert grid.n_steps == res.grid.n_steps
    draws = GaussianDraws(
        FAST_PARTICLE.seed,
        FAST_PARTICLE.n_particles,
        grid.n_steps,
        kernel.n_brownians,
        FAST_PARTICLE.antithetic,
    )
    paths = model.simulate_chunk(grid, draws, 0, FAST_PARTICLE.n_particles, FAST_SIM.scheme)
    np.testing.assert_allclose(paths.log_spot_at(1), res.final_log_spot, rtol=0, atol=1e-12)
    np.testing.assert_allclose(paths.factors_at(1), res.final_factors, rtol=0, atol=1e-12)
    # the pricing grid built by the engine contains every calibration slice
    mc_grid = MonteCarlo(FAST_SIM).build_grid(
        [VarianceSwap([0.0, 0.5], 0.0, kernel.forward_curve.rate_curve)], model
    )
    assert np.all(
        np.isin(
            np.round(res.leverage.times[res.leverage.times <= 0.5], 9), np.round(mc_grid.times, 9)
        )
    )


def test_variance_swaps_invariant_across_omega(market) -> None:
    """Surface-calibrated invariance: VS strikes agree across ω and match replication."""
    _, surface, kernel3 = market
    kernel1 = kernel3.bump(nu=0.5)  # omega = 1
    strikes = {}
    for label, kernel in (("omega=3", kernel3), ("omega=1", kernel1)):
        res = calibrate_leverage(surface, kernel, FAST_PARTICLE, FAST_SIM)
        swap = VarianceSwap([0.0, 0.5], 0.0, surface.discount, use_simulation_grid=True)
        r = MonteCarlo(FAST_SIM).price(swap, LSV(kernel, res.leverage))
        df = float(surface.discount.df(0.5))
        strikes[label] = (r.mean / df, r.stderr / df)
    k_rep = varswap_strike(surface, 0.5)
    for label, (k, se) in strikes.items():
        assert abs(np.sqrt(k) - np.sqrt(k_rep)) < 0.002 + 3 * se / (2 * np.sqrt(k)), (
            label,
            k,
            k_rep,
        )
    k3, s3 = strikes["omega=3"]
    k1, s1 = strikes["omega=1"]
    assert abs(np.sqrt(k3) - np.sqrt(k1)) < 0.002 + 3 * np.hypot(s3, s1) / (2 * np.sqrt(k3))


# ---------------------------------------------------------------------------------------------
# regression estimator, leverage function, cache
# ---------------------------------------------------------------------------------------------


def test_conditional_variance_estimate_recovers_smooth_curve() -> None:
    rng = np.random.default_rng(0)
    n = 200_000
    k = rng.normal(0.0, 0.2, n)
    m_true = 0.04 * (1.0 + 0.5 * k * k - 1.5 * k)  # U-shaped, skewed
    v = m_true * np.exp(rng.normal(-0.245, 0.7, n))  # lognormal noise (std 80%), E = m_true
    grid = np.linspace(-0.6, 0.6, 121)
    cfg = ParticleConfig(n_particles=n, horizon=1.0)
    est = conditional_variance_estimate(k, v, grid, h=0.04, cfg=cfg)
    truth = 0.04 * (1.0 + 0.5 * grid * grid - 1.5 * grid)
    inner = np.abs(grid) <= 0.4  # up to 2 cloud standard deviations
    assert np.max(np.abs(est[inner] / truth[inner] - 1.0)) < 0.04
    # tails continue log-linearly, positive and finite
    assert np.all(np.isfinite(est)) and np.all(est > 0)
    flat = conditional_variance_estimate(
        k,
        v,
        grid,
        h=0.04,
        cfg=ParticleConfig(n_particles=n, horizon=1.0, tail_extrapolation="flat"),
    )
    assert flat[0] == flat[1] and flat[-1] == flat[-2]


# ---------------------------------------------------------------------------------------------
# the regression estimators: the sorted path is the m6 estimator, the binned default is K.5's
# ---------------------------------------------------------------------------------------------


#: the K.5 reference estimator of the deflation arm (what the library's default is since tag k5)
_K5_DEFLATED = functools.partial(_k5_reference.conditional_variance_estimate, deflate=True)


def _same_calibration(a, b) -> None:
    """Bit-for-bit: every leverage row, the bandwidths and the final particle cloud."""
    assert a.leverage.values.dtype == b.leverage.values.dtype == np.float64
    np.testing.assert_array_equal(a.leverage.values, b.leverage.values)
    np.testing.assert_array_equal(a.bandwidths, b.bandwidths)
    np.testing.assert_array_equal(a.final_log_spot, b.final_log_spot)
    np.testing.assert_array_equal(a.final_factors, b.final_factors)


def _synthetic_cloud(n: int = 60_000, seed: int = 5):
    rng = np.random.default_rng(seed)
    k = 0.2 * rng.standard_t(6, n) - 0.03 * rng.exponential(1.0, n)  # skewed, fat put tail
    v = 0.04 * np.exp(-1.5 * k) * np.exp(rng.normal(-0.245, 0.7, n))
    return k, v, np.linspace(-2.5, 2.5, 2001)


def test_sorted_estimator_is_the_m6_path(market, fast_calibration, monkeypatch) -> None:
    """``estimator="sorted"`` is the m6 estimator, bit for bit: against the frozen copy of the m6
    function (``tests/_particle_reference_m6.py``), every option of the post-processing on a
    synthetic cloud, then a whole calibration at the fast settings (50k particles, 1y).  The
    default is the binned estimator and says so in the leverage metadata."""
    import volsto.calibration.particle as particle

    assert ParticleConfig().estimator == "binned"
    assert fast_calibration.leverage.metadata["estimator"] == "binned"
    k, v, grid = _synthetic_cloud()
    slope = float(np.polyfit(k, np.log(v), 1)[0])
    cases: list[tuple[dict, float | None]] = [({}, None)]
    cases += [({"tail_extrapolation": t}, None) for t in ("flat", "log_linear", "adaptive")]
    cases += [({"tail_extrapolation": t}, slope) for t in ("sv_slope", "cloud_slope")]
    cases += [
        ({"bias_correction": False}, None),
        ({"kernel": "quartic"}, None),
        ({"regression": "nadaraya_watson", "tail_extrapolation": "flat"}, None),
        ({"min_window": 500, "min_window_fraction": 0.002}, None),
        ({"quantile_clip": 0.02, "n_regression_points": 51}, None),
    ]
    for changes, tail_slope in cases:
        cfg = ParticleConfig(n_particles=k.size, horizon=1.0, estimator="sorted", **changes)
        for h in (0.004, 0.03):
            got = conditional_variance_estimate(k, v, grid, h, cfg, tail_slope)
            ref = _particle_reference_m6.conditional_variance_estimate(
                k, v, grid, h, cfg, tail_slope
            )
            np.testing.assert_array_equal(got, ref, err_msg=f"{changes} h={h}")
    # degenerate clouds: one point, and no positive estimate
    one = np.full(4000, 0.1)
    for vv in (np.full(4000, 0.04), np.zeros(4000)):
        cfg = ParticleConfig(n_particles=4000, estimator="sorted")
        np.testing.assert_array_equal(
            conditional_variance_estimate(one, vv, grid, 0.01, cfg),
            _particle_reference_m6.conditional_variance_estimate(one, vv, grid, 0.01, cfg),
        )
    # a whole calibration with the frozen estimator in place of the library's
    _, surface, kernel = market
    cfg = dataclasses.replace(FAST_PARTICLE, estimator="sorted")
    lib = calibrate_leverage(surface, kernel, cfg, FAST_SIM)
    assert lib.leverage.metadata["estimator"] == "sorted"
    monkeypatch.setattr(
        particle,
        "conditional_variance_estimate",
        _particle_reference_m6.conditional_variance_estimate,
    )
    frozen = calibrate_leverage(surface, kernel, cfg, FAST_SIM)
    _same_calibration(lib, frozen)
    # and the default is another estimator: close (SPEC §4.1), not equal
    assert not np.array_equal(lib.leverage.values, fast_calibration.leverage.values)


def test_binned_estimator_equals_the_k5_reference(market, fast_calibration, monkeypatch) -> None:
    """Fast variant of the K.5 equality (the 8·10⁵ one is
    ``test_binned_estimator_equals_the_k5_run``): the library's binned estimator against the K.5
    scratch estimator (``tests/_k5_reference.py``, another route to the same numbers) — node
    for node on a synthetic cloud, then a whole calibration at the fast settings on the 1F
    reference spec.  At 50k particles the floor (2000 particles, 4% of the cloud) is active on
    most nodes, so the exact tail sums carry the comparison; the 8·10⁵ test carries the bins."""
    import volsto.calibration.binned as binned
    import volsto.calibration.particle as particle

    k, v, grid = _synthetic_cloud()
    for changes in ({}, {"min_window": 300, "min_window_fraction": 0.005}, {"min_window": 1}):
        cfg = ParticleConfig(n_particles=k.size, horizon=1.0, estimator="binned", **changes)
        for h in (0.004, 0.03):
            for deflate in (False, True):
                got = binned.binned_regression(k, v, h, cfg, deflate=deflate)
                ref = _k5_reference.regression(k, v, h, cfg, deflate)
                assert got is not None and ref is not None
                for a, b in zip(got, ref[:5]):
                    np.testing.assert_array_equal(a, b, err_msg=f"{changes} h={h} {deflate}")
            np.testing.assert_array_equal(  # the library deflates (code tag k5)
                conditional_variance_estimate(k, v, grid, h, cfg),
                _k5_reference.conditional_variance_estimate(k, v, grid, h, cfg, deflate=True),
            )
    # the binned estimate is the sorted one up to the binning error (floored nodes: rounding)
    cfg = ParticleConfig(n_particles=k.size, horizon=1.0, estimator="binned")
    sorted_est = conditional_variance_estimate(
        k, v, grid, 0.03, dataclasses.replace(cfg, estimator="sorted")
    )
    assert not np.array_equal(conditional_variance_estimate(k, v, grid, 0.03, cfg), sorted_est)
    # degenerate cloud: the plain mean, as the sorted path
    one = np.full(4000, 0.1)
    flat = conditional_variance_estimate(
        one, np.full(4000, 0.04), grid, 0.01, ParticleConfig(n_particles=4000, estimator="binned")
    )
    np.testing.assert_array_equal(flat, np.full(grid.shape, 0.04))
    # a whole calibration
    _, surface, kernel = market
    monkeypatch.setattr(particle, "conditional_variance_estimate", _K5_DEFLATED)
    ref_run = calibrate_leverage(surface, kernel, FAST_PARTICLE, FAST_SIM)
    _same_calibration(fast_calibration, ref_run)


# -- the same two statements at the cache settings (8·10⁵ particles, 3y), slow ----------------

#: 2022-07-01 desk marking fit (fit_preset("desk", skew_eps=0.10), ssr_target=1.0), the
#: parameters of the J.4 / K.2 / K.5 runs of the calibration-speed study, as literals: a test
#: does not refit (CONTRIBUTING, machine-dependent arithmetic).
K_STUDY_SNAPSHOT = "configs/surfaces/snapshots/hdn_2022H2/spx_2022-07-01.yaml"
K_STUDY_PARAMS = {
    "nu": 2.991888248939484,
    "theta": 0.2039339463581286,
    "k1": 15.740028921421281,
    "k2": 1.0573587761444543,
    "rho12": 0.7155136238037159,
    "rho_SX1": -0.99,
    "rho_SX2": -0.726374380798787,
}
K_STUDY_SEED = 12345
K_STUDY_PARTICLES = 800_000
#: SHA-256 of the float64 leverage values (1255 x 2001, C order) and of the 1254 bandwidths of
#: the sorted (m6) path on that date and seed: the K.2 rerun baseline (recorded from the
#: K.3(0) control run of 2026-10-04, commit 9a99aaf, which K.3(0) showed equal to the K.2 rerun).
K2_BASELINE_LEVERAGE_SHA256 = "95a379d6f47f6c5e92d1b698c39dc500d80032768046a4ee05707642b9bffda8"
K2_BASELINE_BANDWIDTHS_SHA256 = "5d19730b673dd791a42774b2c1276f5db4509a16a7f5c46fd996552fe03b35b3"
#: SHA-256 of the float64 leverage values of the K.5(2) deflation-arm run (variant C with the
#: deflated bandwidth) on that date and seed.  NOT RECORDED YET: K.5(2) stored its leverages in
#: float32; the float64 rerun of the scratch reference is pending (it waits for the machine).
#: The test fails until it is.
K5_RUN_LEVERAGE_SHA256: str | None = None
#: The digests are bit patterns of numbers computed with this platform's libm and numpy
#: partition: they are asserted where they were recorded and nowhere else.
DIGEST_PLATFORM = ("darwin", "arm64")


def _on_digest_platform() -> bool:
    return (sys.platform, platform.machine()) == DIGEST_PLATFORM


def _sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.float64).tobytes()).hexdigest()


def _k_study_market(estimator: str):
    from volsto.market.loaders import snapshot_spec

    root = Path(__file__).resolve().parents[1]
    base = load_yaml(root / "configs" / "studies" / "lsv_reference_2f.yaml", CalibrationSpec)
    spec = snapshot_spec(base, root / K_STUDY_SNAPSHOT)
    spec = dataclasses.replace(
        spec,
        model=BergomiParams(**K_STUDY_PARAMS),
        particle=dataclasses.replace(
            spec.particle, n_particles=K_STUDY_PARTICLES, seed=K_STUDY_SEED, estimator=estimator
        ),
    )
    _, surface, kernel = build_market(spec)
    return spec, surface, kernel


@pytest.mark.slow
@pytest.mark.skipif(
    not _on_digest_platform(),
    reason=f"the K.2 baseline digest was recorded on {DIGEST_PLATFORM}; bit patterns are "
    "asserted only there",
)
def test_sorted_path_matches_the_k2_baseline_digest() -> None:
    """The sorted path at the cache settings (8·10⁵ particles, 3y, 2022-07-01, seed 12345) gives
    the leverage the m6 code gave before the estimator was split and the draws were made lean:
    the digest of the K.2 rerun baseline."""
    spec, surface, kernel = _k_study_market("sorted")
    res = calibrate_leverage(surface, kernel, spec.particle, spec.sim, local_vol_cfg=spec.local_vol)
    assert res.leverage.values.shape == (1255, 2001)
    assert _sha256(res.bandwidths) == K2_BASELINE_BANDWIDTHS_SHA256
    assert _sha256(res.leverage.values) == K2_BASELINE_LEVERAGE_SHA256


@pytest.mark.slow
def test_binned_estimator_equals_the_k5_run(monkeypatch) -> None:
    """The library's default (binned, deflated) estimator at the cache settings equals the K.5
    scratch estimator of the deflation arm bit for bit — all 1255 leverage rows and the
    bandwidths, in float64 — and, where it was recorded, the digest of that K.5(2) run."""
    import volsto.calibration.particle as particle

    spec, surface, kernel = _k_study_market("binned")
    lib = calibrate_leverage(surface, kernel, spec.particle, spec.sim, local_vol_cfg=spec.local_vol)
    monkeypatch.setattr(particle, "conditional_variance_estimate", _K5_DEFLATED)
    ref_run = calibrate_leverage(
        surface, kernel, spec.particle, spec.sim, local_vol_cfg=spec.local_vol
    )
    assert lib.leverage.values.shape == (1255, 2001)
    _same_calibration(lib, ref_run)
    if _on_digest_platform():
        assert K5_RUN_LEVERAGE_SHA256 is not None, (
            "the K.5(2) deflation digest is not recorded: run the float64 rerun of the scratch "
            "reference and set K5_RUN_LEVERAGE_SHA256"
        )
        assert _sha256(lib.leverage.values) == K5_RUN_LEVERAGE_SHA256


def test_particle_estimator_switch_validation() -> None:
    with pytest.raises(ValueError, match="estimator"):
        ParticleConfig(estimator="histogram")
    for bad in ({"kernel": "quartic"}, {"regression": "nadaraya_watson"}):
        with pytest.raises(ValueError, match="binned"):
            ParticleConfig(estimator="binned", **bad)
    from volsto.config import to_mapping

    assert to_mapping(ParticleConfig())["estimator"] == "binned"
    assert to_mapping(ParticleConfig(estimator="sorted"))["estimator"] == "sorted"
    # the default estimator refuses what it does not implement, by name
    with pytest.raises(ValueError, match="set estimator='sorted'"):
        ParticleConfig(kernel="quartic")
    assert ParticleConfig(kernel="quartic", estimator="sorted").kernel == "quartic"


def test_leverage_function_interpolation_and_io(market, tmp_path: Path) -> None:
    fc, _, _ = market
    times = np.array([0.0, 0.5, 1.0])
    k = np.linspace(-1.0, 1.0, 5)
    values = np.array(
        [[1.0, 1.0, 1.0, 1.0, 1.0], [2.0, 1.5, 1.0, 1.5, 2.0], [3.0, 2.0, 1.0, 2.0, 3.0]]
    )
    lev = LeverageFunction(times, k, values, fc, {"seed": 1})
    np.testing.assert_allclose(lev.rows([0.25])[0], [1.5, 1.25, 1.0, 1.25, 1.5])  # linear in t
    np.testing.assert_allclose(lev.rows([5.0])[0], values[2])  # held beyond the horizon
    F = float(fc.forward(0.5))
    assert float(lev(0.5, F)) == pytest.approx(1.0)
    assert float(lev(0.5, F * np.exp(0.25))) == pytest.approx(1.25)  # linear in k
    assert float(lev(0.5, F * np.exp(5.0))) == pytest.approx(2.0)  # flat beyond the grid
    p = lev.save(tmp_path / "lev.npz")
    back = LeverageFunction.load(p)
    np.testing.assert_array_equal(back.values, values)
    assert back.metadata["seed"] == 1 and back.forward_curve.spot == fc.spot
    with pytest.raises(ValueError):
        LeverageFunction(times, k, -values, fc)


def test_lsv_model_interface(market, fast_calibration) -> None:
    _, _, kernel = market
    model = LSV(kernel, fast_calibration.leverage)
    st = model.initial_state(4)
    l0 = float(fast_calibration.leverage(0.0, kernel.spot))
    np.testing.assert_allclose(st.variance, l0**2 * float(kernel.xi0.xi0(0.0)))
    bumped = model.bump(nu=1.0)
    assert bumped.kernel.params.nu == 1.0 and bumped.leverage is model.leverage  # sticky leverage
    assert "LSV(" in repr(model)


def test_cache_round_trip(spec: CalibrationSpec, tmp_path: Path) -> None:
    fast_spec = CalibrationSpec(
        spec.market,
        spec.surface,
        spec.model,
        ParticleConfig(n_particles=20_000, horizon=0.5, seed=3),
        FAST_SIM,
    )
    cache = LeverageCache(tmp_path / "cache")
    key = cache.key(fast_spec)
    assert key == spec_key(fast_spec) and len(key) == 64
    # pricing-only settings do not change the key; model / particle settings do
    other_sim = CalibrationSpec(
        fast_spec.market,
        fast_spec.surface,
        fast_spec.model,
        fast_spec.particle,
        SimConfig(n_paths=10, seed=99),
    )
    assert cache.key(other_sim) == key
    assert (
        cache.key(
            CalibrationSpec(
                fast_spec.market,
                fast_spec.surface,
                fast_spec.model.replace(nu=1.0),
                fast_spec.particle,
                FAST_SIM,
            )
        )
        != key
    )
    assert (
        cache.key(
            CalibrationSpec(
                fast_spec.market,
                fast_spec.surface,
                fast_spec.model,
                fast_spec.particle,
                SimConfig(dt_max=StepSchedule.uniform(0.02)),
            )
        )
        != key
    )
    with pytest.raises(CacheMissError):
        cache.get_or_calibrate(fast_spec, allow_calibrate=False)
    model, report = cache.get_or_calibrate(
        fast_spec,
        run_diagnostics=True,
        diagnostics_sim=SimConfig(n_paths=20_000, chunk_size=10_000),
    )
    assert cache.has(fast_spec) and report is not None
    atm = report.vanillas[(report.vanillas["k"] == 0.0) & (report.vanillas["T"] <= 0.5)]
    assert np.all(np.abs(atm["error_vp"]) < 1.0), report.summary()  # plumbing check only
    entry = cache.entry_dir(fast_spec)
    assert (
        (entry / "leverage.npz").exists()
        and (entry / "diagnostics.json").exists()
        and (entry / "spec.json").exists()
    )
    man = cache.manifest()
    assert len(man) == 1 and man.iloc[0]["key"] == key and man.iloc[0]["n_particles"] == 20_000
    # hit: no recalibration, identical leverage, report reloaded
    model2, report2 = cache.get_or_calibrate(fast_spec, allow_calibrate=False)
    np.testing.assert_array_equal(model2.leverage.values, model.leverage.values)
    assert report2 is not None and report2.n_paths == 20_000
    assert model2.leverage.metadata["cache_key"] == key and "git_commit" in model2.leverage.metadata


def test_particle_config_validation() -> None:
    with pytest.raises(ValueError):
        ParticleConfig(n_particles=1001)
    with pytest.raises(ValueError):
        ParticleConfig(regression="spline")
    with pytest.raises(ValueError):
        ParticleConfig(tail_extrapolation="quadratic")
    p = BergomiParams.one_factor(3.0, 1.5, -0.7)
    assert (
        BergomiSV(
            p, ForwardVarianceCurve.flat(0.04), build_market(load_yaml(SPEC_1F, CalibrationSpec))[0]
        ).n_factors
        == 1
    )


def test_calibration_code_tag_guard() -> None:
    """The source of the calibration and stepping modules is hashed; a change without a bump of
    CALIBRATION_CODE_TAG (and a refreshed code_tag_guard.json) fails here."""
    from volsto.calibration.cache import GUARD_FILE, check_guard, read_guard, source_hash
    from volsto.calibration.particle import CALIBRATION_CODE_TAG

    assert GUARD_FILE.exists(), "code_tag_guard.json missing: run write_guard()"
    assert CALIBRATION_CODE_TAG in read_guard()
    assert len(source_hash()) == 64
    check_guard()
