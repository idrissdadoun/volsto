"""Smart (signed) gap — owner addendum at the M6 review (SPEC §6.9): mechanism tests.

* The sign rule and the sizing placeholder, the regression helper, the level factors and the
  regression grid (hand values).
* The applied shift is conservative for the seller in every tested state: the smart-gap price
  lies on the cautious side of the unshifted price (paired, common random numbers) for the
  autocall, the Phoenix, an up-and-out call and a down-and-in put; a zero-size gap reproduces the
  unshifted payoffs path by path; the fixed mode equals the M6 level shift.
* The sign is measured, never fixed per barrier: on the SAME autocall barrier the effective shift
  has opposite signs in the owner's two states — (i) mid-life, spot near the level, the full
  coupon stream ahead (continuing costs the seller more: ``ΔV < 0``, the level moves up) and
  (ii) a few days before the observation with the spot far above the level, the call near-certain
  (the paths that reach the level got there through a fast fall and carry a high variance state
  in which the continuing note is worth less than the redemption: ``ΔV > 0``, the level moves
  down) — each sign matching a direct measurement (raw conditional means of the realised cash
  flows of the paths just below the level, no regression).  A knocked-in state of the same note
  was also measured while designing the test: the live put lowers the continuing value by only
  about two points of notional here and does not flip the sign (recorded in SPEC §6.9).
* ``|shift|`` grows with ``|ΔV|``: moving the knock-in level changes the put depth at the
  autocall barrier (``|ΔV|`` and ``|shift|`` at the first date increase with the knock-in level)
  and the knock-in barrier's own jump at maturity ``1 − KI`` (its shift decreases with it).
* Barrier options: the up-and-out call's continuation value at the barrier exceeds a zero rebate
  (level moves up) and falls short of a large rebate (level moves down); the down-and-in put's
  vanilla exceeds the not-yet-knocked value (level moves up).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from helpers import flat_state

from volsto.analytics.autocall import autocall_report
from volsto.config import BergomiParams, SimConfig
from volsto.engine import MonteCarlo
from volsto.market import ForwardCurve, ForwardVarianceCurve
from volsto.models import BergomiSV, BlackScholes
from volsto.products import (
    Autocall,
    GapSpec,
    KnockInOption,
    KnockOutOption,
    LevelFactors,
    Phoenix,
    conditional_value_at_level,
    conservative_shift,
    month_grid,
)
from volsto.products.gap import gap_linear
from volsto.risk import BSBuilder, RiskEngine, risk_report

OBS = [1.0, 2.0, 3.0]
S0 = 100.0
SMART = GapSpec(mode="smart")
ZERO = GapSpec(mode="smart", params={"scale": 0.0, "floor_bp": 0.0, "cap_bp": 500.0})


def _paths(mc: MonteCarlo, model, product):  # type: ignore[no-untyped-def]
    grid = mc.build_grid([product], model)
    draws = mc.draws_for(grid, model)
    return model.simulate_chunk(grid, draws, 0, mc.cfg.n_paths, mc.cfg.scheme), grid.fixing_index


def _paired(mc: MonteCarlo, model, base, smart):  # type: ignore[no-untyped-def]
    res = mc.price_many([base, smart], model, keep_payoffs=True)
    d = np.asarray(res[1].payoffs) - np.asarray(res[0].payoffs)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(d.size)), res


def test_sign_rule_sizing_and_helpers() -> None:
    dv = np.array([0.3, -0.3, 0.0])
    size = np.array([0.02, 0.02, 0.02])
    # ΔV > 0: the knock costs more → enlarge the knocked region: up-barrier down, down-barrier up
    assert np.allclose(conservative_shift(dv, "up", size), [-0.02, 0.02, 0.0])
    assert np.allclose(conservative_shift(dv, "down", size), [0.02, -0.02, 0.0])
    with pytest.raises(ValueError):
        conservative_shift(dv, "sideways", size)
    params = {"scale": 0.2, "floor_bp": 10.0, "cap_bp": 500.0}
    assert np.allclose(gap_linear(np.array([0.0, 0.1, 1.0]), params), [0.001, 0.021, 0.05])
    spec = GapSpec(mode="smart")
    assert spec.smart and "placeholder" in repr(spec)
    assert np.allclose(spec.shift(np.array([0.1, -0.1, 5e-4]), "up"), [-0.021, 0.021, 0.0])
    # below z_min standard errors the sign is not identified: no shift
    assert np.allclose(
        spec.shift(np.array([0.1, 0.1]), "up", np.array([0.01, 0.06])), [-0.021, 0.0]
    )
    assert repr(GapSpec(mode="fixed", fixed_shift=0.01)) == "gap fixed +1.0000%"
    for bad in (
        {"mode": "auto"},
        {"function": "vegas"},
        {"apply_to": ("ac",)},
        {"degree": 0},
        {"band": 0.0},
        {"ki_grid_months": 0},
        {"min_paths": 1},
        {"report_band": 1.0},
        {"dv_min": -1.0},
        {"fixed_shift": -1.0},
    ):
        with pytest.raises(ValueError):
            GapSpec(**bad)  # type: ignore[arg-type]
    # regression helper: an exact quadratic is recovered at the level with its standard error
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.25, 0.0, 4000)
    z = rng.normal(size=4000)
    y = 1.0 + 2.0 * x - 3.0 * x * x + 0.5 * z + 1e-3 * rng.normal(size=4000)
    feats = np.column_stack([x, z])
    ev = np.column_stack([np.zeros(3), np.array([-1.0, 0.0, 1.0])])
    val, n, se = conditional_value_at_level(y, feats, ev, degree=2, band=0.3, min_paths=500)
    assert n == 4000 and np.allclose(val, [0.5, 1.0, 1.5], atol=2e-3) and np.all(se < 1e-3)
    # degenerate cloud (every path at one spot) or too few paths: no fit
    val, n, se = conditional_value_at_level(y, np.column_stack([np.zeros(4000), z]), ev, degree=2)
    assert n == 0 and np.all(np.isnan(val))
    assert conditional_value_at_level(y[:100], feats[:100], ev, degree=2)[1] == 0
    # level factors: piecewise constant from each grid date to the next
    lf = LevelFactors(fixed=1.0, grid=np.array([0.0, 1.0]), factors=np.array([[1.01, 0.98]]))
    assert np.allclose(lf.at(np.array([0.0, 0.5, 1.0, 1.5]), 1), [[1.01, 1.01, 0.98, 0.98]])
    assert np.allclose(LevelFactors(fixed=1.02).at(np.array([0.3, 0.7]), 2), 1.02)
    # regression grid: monthly dates snapped to the monitoring schedule
    grid = month_grid(np.arange(0, 253) / 252.0, 1)
    assert grid.size == 13 and grid[0] == 0.0 and grid[-1] == 1.0
    assert np.all(np.diff(grid) > 0.07)


def test_validation_on_products(discount) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(NotImplementedError):
        Autocall(
            OBS,
            discount,
            spot_reference=S0,
            coupons=0.06,
            ki_level=0.6,
            ki_type="american",
            ki_monitoring="continuous",
            gap=SMART,
        )
    with pytest.raises(TypeError):
        Autocall(OBS, discount, spot_reference=S0, coupons=0.06, ki_level=0.6, ki_type="european", gap="smart")  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        KnockOutOption(
            100.0, 1.0, "call", 120.0, "up", discount, monitoring="continuous", gap=SMART
        )
    with pytest.raises(ValueError):
        KnockOutOption(
            100.0,
            1.0,
            "call",
            120.0,
            "up",
            discount,
            monitoring="discrete",
            strict=True,
            barrier_shift=0.01,
            gap=GapSpec(mode="fixed", fixed_shift=0.02),
        )
    ko = KnockOutOption(
        100.0,
        1.0,
        "call",
        120.0,
        "up",
        discount,
        monitoring="discrete",
        strict=True,
        gap=GapSpec(mode="fixed", fixed_shift=0.02),
    )
    assert ko.barrier_shift == 0.02 and ko.barrier_eff == pytest.approx(122.4)
    plain = Autocall(
        OBS, discount, spot_reference=S0, coupons=0.06, ki_level=0.6, ki_type="european"
    )
    with pytest.raises(ValueError):
        plain.gap_report(None, None)  # type: ignore[arg-type]
    smart = plain.replace(gap=SMART)
    assert "gap smart" in repr(smart) and smart.aged(0.1).gap is SMART


def test_conservative_zero_limit_and_fixed_mode(forward_curve) -> None:  # type: ignore[no-untyped-def]
    """Every tested product: the paired smart-minus-unshifted price difference is positive (the
    seller's liability is marked higher) by several standard errors; the zero-size gap gives the
    unshifted payoffs exactly; the fixed mode equals the M6 shift of every monitored level."""
    disc = forward_curve.rate_curve
    model = BlackScholes(0.25, forward_curve)
    sim = SimConfig(n_paths=40_000, dt_max=1.0 / 52.0, chunk_size=40_000, seed=11)
    mc = MonteCarlo(sim)
    autocall = Autocall(
        OBS, disc, spot_reference=S0, coupons=0.08, ki_level=0.6, ki_type="european"
    )
    phoenix = Phoenix(
        OBS,
        disc,
        spot_reference=S0,
        coupon=0.06,
        coupon_barrier=0.7,
        memory=True,
        ki_level=0.6,
        ki_type="american",
        ki_monitoring="discrete",
        ki_fixing_times=np.arange(0, 3 * 52 + 1) / 52.0,
    )
    ko = KnockOutOption(100.0, 1.0, "call", 120.0, "up", disc, monitoring="discrete", strict=True)
    ki = KnockInOption(100.0, 1.0, "put", 80.0, "down", disc, monitoring="discrete", strict=True)
    for base in (autocall, phoenix, ko, ki):
        smart = base.replace(gap=SMART) if isinstance(base, Autocall) else _with_gap(base, SMART)
        zero = base.replace(gap=ZERO) if isinstance(base, Autocall) else _with_gap(base, ZERO)
        mean, se, res = _paired(mc, model, base, smart)
        assert mean > 3 * se, (repr(base), mean, se)
        pay0 = np.asarray(res[0].payoffs)
        res_zero = mc.price(zero, model, keep_payoffs=True)
        assert np.array_equal(np.asarray(res_zero.payoffs), pay0), repr(base)
    # fixed mode: the autocall's levels (autocall barriers and knock-in) move by the same factor
    fixed = autocall.replace(gap=GapSpec(mode="fixed", fixed_shift=0.01))
    shifted = autocall.replace(autocall_barriers=1.01, ki_level=0.6 * 1.01)
    paths, idx = _paths(mc, model, autocall)
    assert np.array_equal(fixed.payoff(paths, idx), shifted.payoff(paths, idx))
    paths, idx = _paths(mc, model, ko)
    assert np.array_equal(
        _with_gap(ko, GapSpec(mode="fixed", fixed_shift=0.01)).payoff(paths, idx),
        _rebuild(ko, barrier_shift=0.01).payoff(paths, idx),
    )


def _with_gap(product, gap):  # type: ignore[no-untyped-def]
    return _rebuild(product, gap=gap)


def _rebuild(product, **changes):  # type: ignore[no-untyped-def]
    kw = product._kwargs()
    kw.update(changes)
    return type(product)(
        product.strike,
        product.T,
        product.cp,
        product.barrier,
        product.direction,
        product.discount,
        rebate=product.rebate,
        rebate_timing=product.rebate_timing,
        notional=product.notional,
        **kw,
    )


def _direct_dv_at_first_level(product: Autocall, base: Autocall, paths, idx, width: float = 0.02):  # type: ignore[no-untyped-def]
    """Directly measured seller ΔV at the first observation date's autocall level: the called
    liability ``1 + coupon`` less the raw mean of the realised cash flows (date money) of the
    paths continuing from just below the level (no regression)."""
    t1 = float(product.observation_times[0])
    level = float(product.autocall_levels[0])
    s1 = paths.spot_at(idx[t1])
    stats = base.statistics(paths, idx)
    cont = stats["ac_index"] > 1
    band = cont & (s1 < level) & (s1 >= level * (1.0 - width))
    y = base.payoff(paths, idx) / float(np.asarray(base.df(np.array([t1])))[0])
    v_cont = float(y[band].mean())
    se = float(y[band].std(ddof=1) / np.sqrt(band.sum()))
    coupon = float(base.coupon_schedule[0])
    return (1.0 + coupon) - v_cont, se, int(band.sum())


def test_two_states_same_autocall_barrier_opposite_signs() -> None:
    """The owner's two-state test (module docstring).  One-factor Bergomi (ν = 2.5, κ = 0.3,
    ρ = −0.9, flat ξ = 0.09), Phoenix coupon 6% at CB 50% with memory, American daily knock-in
    at 50%, autocall at 100%, five annual dates, rates 1%.  The autocall barrier under test is
    the original third date's.  State (i): six months before it, spot at the level, the full
    coupon stream ahead — continuing costs the seller more than calling (``ΔV < 0``), the level
    moves up.  State (ii): ten days before it, spot 10% above the level, the call near-certain —
    the paths that do reach the level got there through a 10% fall in ten days and carry a high
    variance state (``X ≈ +0.4``: instantaneous vol about 2.3× the base), in which the continuing
    note (coupons at risk, knock-in put) is worth less than the redemption: ``ΔV > 0``, the level
    moves down.  Both signs are measured directly (raw conditional means of the realised cash
    flows of the paths just below the level, no regression) and the effective shift follows the
    measured sign in each state; the test fails if the sign is ever fixed per barrier."""
    params = BergomiParams.one_factor(2.5, 0.3, -0.9)
    xi = ForwardVarianceCurve.flat(0.09)
    smart = GapSpec(mode="smart", apply_to=("autocall",))

    def state(spot: float, age: float):  # type: ignore[no-untyped-def]
        fc = ForwardCurve.flat(spot, 0.01, 0.0)
        model = BergomiSV(params, xi, fc)
        obs = [t - age for t in [1.0, 2.0, 3.0, 4.0, 5.0] if t - age > 1e-9]
        base = Phoenix(
            obs,
            fc.rate_curve,
            spot_reference=S0,
            coupon=0.06,
            coupon_barrier=0.5,
            memory=True,
            ki_level=0.5,
            ki_type="american",
            ki_monitoring="discrete",
        )
        product = base.replace(gap=smart)
        sim = SimConfig(n_paths=200_000, dt_max=1.0 / 104.0, chunk_size=200_000, seed=5)
        mc = MonteCarlo(sim)
        paths, idx = _paths(mc, model, product)
        rep = product.gap_report(paths, idx).as_frame()
        row = rep[rep.barrier == "autocall_1"].iloc[0]
        dv_direct, se_direct, n_band = _direct_dv_at_first_level(product, base, paths, idx)
        d = product.payoff(paths, idx) - base.payoff(paths, idx)
        t1 = float(product.observation_times[0])
        s1 = paths.spot_at(idx[t1])
        x1 = paths.factors_at(idx[t1])[:, 0]
        near = np.abs(np.log(s1 / product.autocall_levels[0])) <= 0.02
        return {
            "row": row,
            "dv_direct": dv_direct,
            "se_direct": se_direct,
            "n_band": n_band,
            "diff": float(d.mean()),
            "diff_se": float(d.std(ddof=1) / np.sqrt(d.size)),
            "x_near": float(x1[near].mean()),
        }

    states = {"i": state(100.0, 2.5), "ii": state(110.0, 3.0 - 10.0 / 252.0)}
    for key, st in states.items():
        row = st["row"]
        # the direct measurement is significant; the regression agrees in sign and magnitude
        assert st["n_band"] > 1000 and abs(st["dv_direct"]) > 5 * st["se_direct"], (key, st)
        assert np.sign(row["dv_near_mean"]) == np.sign(st["dv_direct"]), (key, st)
        assert abs(row["dv_near_mean"] - st["dv_direct"]) < 0.02, (key, st)
        # the effective shift on the paths at the barrier follows the measured sign (up-barrier:
        # shift = −sign(ΔV) size) and is applied on most of them
        assert row["share_shifted_near"] > 0.5, (key, row["share_shifted_near"])
        assert np.sign(row["shift_near_mean"]) == -np.sign(st["dv_direct"]), (key, st)
        # conservative for the seller in every state
        assert st["diff"] > 3 * st["diff_se"], (key, st)
    # the SAME autocall barrier, opposite signs: continuing costs more mid-life, calling costs
    # more in the high-variance state the near-certain-call scenario reaches the level in
    assert states["i"]["dv_direct"] < -0.01 and states["ii"]["dv_direct"] > 0.01, states
    assert states["i"]["row"]["shift_near_mean"] > 0 > states["ii"]["row"]["shift_near_mean"]
    assert states["ii"]["x_near"] > states["i"]["x_near"] + 0.1, (
        states["i"]["x_near"],
        states["ii"]["x_near"],
    )


def test_shift_grows_with_dv_via_ki_level(forward_curve) -> None:  # type: ignore[no-untyped-def]
    """Black–Scholes 25%, plain autocall 8% growing coupon, European knock-in at 50 / 60 / 70%:
    the deeper the put at the autocall barrier the larger ``|ΔV|`` and ``|shift|`` there (the
    first date), while the knock-in barrier's own maturity jump ``1 − KI`` and its shift shrink."""
    disc = forward_curve.rate_curve
    model = BlackScholes(0.25, forward_curve)
    sim = SimConfig(n_paths=40_000, dt_max=1.0 / 52.0, chunk_size=40_000, seed=3)
    mc = MonteCarlo(sim)
    spec = GapSpec(mode="smart", params={"scale": 0.05, "floor_bp": 10.0, "cap_bp": 500.0})
    dv_ac: list[float] = []
    sh_ac: list[float] = []
    dv_ki: list[float] = []
    sh_ki: list[float] = []
    for ki in (0.5, 0.6, 0.7):
        p = Autocall(
            OBS, disc, spot_reference=S0, coupons=0.08, ki_level=ki, ki_type="european", gap=spec
        )
        paths, idx = _paths(mc, model, p)
        rep = p.gap_report(paths, idx).as_frame().set_index("barrier")
        dv_ac.append(float(rep.loc["autocall_1", "dv_near_mean"]))
        sh_ac.append(float(rep.loc["autocall_1", "abs_shift_near_mean"]))
        dv_ki.append(float(rep.loc["ki", "dv_near_mean"]))
        sh_ki.append(float(rep.loc["ki", "abs_shift_near_mean"]))
        assert rep.loc["ki", "dv_near_mean"] == pytest.approx(-(1.0 - ki), abs=0.02)
    assert np.all(np.diff(np.abs(dv_ac)) > 0) and np.all(np.diff(sh_ac) > 0), (dv_ac, sh_ac)
    assert np.all(np.diff(np.abs(dv_ki)) < 0) and np.all(np.diff(sh_ki) < 0), (dv_ki, sh_ki)
    assert np.allclose(sh_ki, 0.001 + 0.05 * np.abs(dv_ki), atol=1e-6)


def test_barrier_options_follow_the_measured_sign(forward_curve) -> None:  # type: ignore[no-untyped-def]
    disc = forward_curve.rate_curve
    model = BlackScholes(0.25, forward_curve)
    sim = SimConfig(n_paths=40_000, dt_max=1.0 / 52.0, chunk_size=40_000, seed=9)
    mc = MonteCarlo(sim)
    weekly = np.arange(0, 53) / 52.0
    ko = KnockOutOption(
        100.0,
        1.0,
        "call",
        120.0,
        "up",
        disc,
        monitoring="discrete",
        strict=True,
        fixing_times=weekly,
        gap=SMART,
    )
    ko_rebate = KnockOutOption(
        100.0,
        1.0,
        "call",
        120.0,
        "up",
        disc,
        monitoring="discrete",
        strict=True,
        fixing_times=weekly,
        rebate=40.0,
        rebate_timing="hit",
        gap=SMART,
    )
    ki = KnockInOption(
        100.0,
        1.0,
        "put",
        80.0,
        "down",
        disc,
        monitoring="discrete",
        strict=True,
        fixing_times=weekly,
        gap=SMART,
    )
    paths, idx = _paths(mc, model, ko)
    for product, expected_sign in ((ko, -1), (ko_rebate, +1), (ki, +1)):
        rep = product.gap_report(paths, idx).as_frame()
        late = rep[(rep.date >= 0.75) & (rep.n_fit > 0)]
        assert len(late) >= 3, repr(product)
        assert np.all(np.sign(late["dv_near_mean"]) == expected_sign), (repr(product), late)
        assert np.all(late["share_shifted_near"] > 0.9), late
        # up-barrier: shift = −sign(ΔV) size; down-barrier: +sign(ΔV) size
        direction_sign = -1 if product.direction == "up" else 1
        assert np.all(np.sign(late["shift_near_mean"]) == direction_sign * expected_sign), late
        base = _rebuild(product, gap=None)
        d = product.payoff(paths, idx) - base.payoff(paths, idx)
        assert d.mean() > 3 * d.std(ddof=1) / np.sqrt(d.size), repr(product)
    frame = ko.gap_report(paths, idx).as_frame()
    assert isinstance(frame, pd.DataFrame) and {"sign_near", "n_fit", "shift_near_mean"} <= set(
        frame
    )


def test_reports_carry_the_gap_table(forward_curve) -> None:  # type: ignore[no-untyped-def]
    """``autocall_report`` and the RiskReport product section report ΔV, its sign and the
    effective shift per barrier and date for a smart-gap product (SPEC §6.9)."""
    disc = forward_curve.rate_curve
    model = BlackScholes(0.25, forward_curve)
    sim = SimConfig(n_paths=20_000, dt_max=1.0 / 52.0, chunk_size=20_000, seed=2)
    product = Autocall(
        OBS, disc, spot_reference=S0, coupons=0.08, ki_level=0.6, ki_type="european", gap=SMART
    )
    rep = autocall_report(product, model, sim)
    assert rep.gap is not None and rep.gap.attrs["n_paths"] == 20_000
    assert list(rep.gap["barrier"]) == ["autocall_1", "autocall_2", "autocall_3", "ki"]
    assert {"dv_near_mean", "sign_near", "shift_near_mean", "n_fit"} <= set(rep.gap.columns)
    assert abs(rep.legs_total - rep.price) < 1e-12  # the legs share the two-pass evaluation
    plain = autocall_report(product.replace(gap=None), model, sim)
    assert plain.gap is None and rep.price > plain.price  # conservative for the seller
    state = flat_state(sigma=0.25)
    engine = RiskEngine(BSBuilder(state), sim)
    risk = risk_report(engine, product, state, sections=["product"])
    assert "gap" in risk.tables and len(risk.tables["gap"]) == 4
    assert risk.tables["gap"].attrs["state"] == state.label
