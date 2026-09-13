"""M6 Part 3: risk for digital structures (SPEC v2 §6.7, extending §7.10).  The residual-note
construction (explicit versus ``aged``, re-indexed coupons, knocked-in and memory histories) as a
path-by-path identity against the original note on an extended path; the single-date autocall
profile against the Black–Scholes digital and call-spread digital stencils at 1% and 0.25% (peak
at the barrier, barrier shift, convergence of the stencil to the exact digital delta); the
multi-date profiles (explicit residual notes priced end to end, the smoothed residual note, the
barrier-shift scopes, the American knock-in); the knock-in put's ``∂price/∂B`` against the closed
form and the ±5% profile row by row against the closed-form delta / gamma; the sign test under
local vol on the reference surface (short vol, short skew under the fixed-ATM rotation, long
forward through the delta, the repo delta and the equity legs' rho — the note's rho pinned
negative: the bond leg's discounting dominates); the expected-life sensitivities; the
likelihood-ratio cross-check on the digital legs with the standard-error ratios at two stencils
and the curve binding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from helpers import flat_state as _flat_state

from volsto.config import CalibrationSpec, SimConfig, load_yaml
from volsto.engine import FixingIndex, PathSet
from volsto.market import (
    DiscountCurve,
    ForwardCurve,
    bs_delta,
    bs_gamma,
    bs_price,
    norm_cdf,
    norm_pdf,
)
from volsto.products import Autocall, Phoenix
from volsto.risk import BSBuilder, LVBuilder, RiskEngine, RiskState, delta_gamma
from volsto.risk.digital_risk import (
    SmoothedAutocall,
    autocall_spot_profiles,
    callable_dates,
    default_profile_shifts,
    expected_life_sensitivity,
    ki_barrier_profile,
    ki_put_barrier_risk,
    lr_cross_check,
    residual_note,
    shifted_levels,
    structure_vega_skew,
)
from volsto.risk.engine import surface_of

ROOT = Path(__file__).resolve().parents[1]
SPEC_1F = ROOT / "configs" / "studies" / "lsv_reference_1f.yaml"
SIGMA, R, Q = 0.2, 0.02, 0.01
OBS = [1.0, 2.0, 3.0]
S0 = 100.0
DT = 1.0 / 252.0


# --------------------------------------------------------------------------------------------
# synthetic paths
# --------------------------------------------------------------------------------------------


def _paths(spots: np.ndarray, times: np.ndarray) -> PathSet:
    n, m = spots.shape
    ls = np.log(spots)
    dx = np.diff(ls, axis=1)
    sum_sq = np.concatenate([np.zeros((n, 1)), np.cumsum(dx * dx, axis=1)], axis=1)
    return PathSet(times, ls, np.zeros((n, m)), np.zeros((n, m, 0)), np.zeros((n, m)), sum_sq)


def _random_spots(
    rng: np.random.Generator, n: int, times: np.ndarray, start: float, vol: float = 0.4
) -> np.ndarray:
    dt = np.diff(times)
    steps = rng.normal(-0.5 * vol**2 * dt, vol * np.sqrt(dt), size=(n, times.size - 1))
    return start * np.exp(np.concatenate([np.zeros((n, 1)), np.cumsum(steps, axis=1)], axis=1))


def _identity_check(
    product: Autocall, date: int, spot: float, rng: np.random.Generator, n: int = 400
) -> Autocall:
    """The original note on a path flat at ``spot`` up to ``T_i − dt`` and random afterwards
    equals ``DF(T_i − dt)`` times the residual note on the truncated path, path by path."""
    note, info = residual_note(product, date, spot=spot, dt=DT)
    t0 = info["valuation_time"]
    res_times = note.fixing_times  # a t = 0 column carries the held spot at the valuation time
    res_times = np.concatenate([[0.0], res_times]) if res_times[0] > 0 else res_times
    res_spots = _random_spots(rng, n, res_times, spot)
    res_spots[:, 0] = spot
    orig_times = product.fixing_times
    orig_times = np.concatenate([[0.0], orig_times]) if orig_times[0] > 0 else orig_times
    orig_spots = np.full((n, orig_times.size), spot)
    later = orig_times > t0 + 1e-9
    # every residual fixing after 0 is an original fixing shifted by t0
    shifted = res_times[1:] + t0
    cols = np.searchsorted(orig_times, shifted)
    np.testing.assert_allclose(orig_times[cols], shifted, atol=1e-9)
    # a knocked-in residual note (European form) drops the later monitoring dates, whose spots
    # then no longer matter; an alive one keeps every later fixing
    mapped, later_set = set(cols.tolist()), set(np.flatnonzero(later).tolist())
    assert mapped <= later_set
    if info["ki_history"] != "knocked_in":
        assert mapped == later_set
    orig_spots[:, cols] = res_spots[:, 1:]
    pay_orig = product.payoff(_paths(orig_spots, orig_times), FixingIndex(orig_times))
    pay_res = note.payoff(_paths(res_spots, res_times), FixingIndex(res_times))
    df0 = float(product.df(t0))
    # cash flows of the held history (period coupons paid at the dropped dates on a flat path at
    # or above the coupon barrier) belong to the past, not to the residual note
    past = 0.0
    if product.has_coupon_leg:
        cb = product.coupon_barrier
        if product.guaranteed_coupons or (cb is not None and spot >= cb * product.spot_reference):
            obs = product.observation_times[: date - 1]
            past = product.notional * float(
                np.sum(product.coupon_schedule[: date - 1] * product.df(obs))
            )
    np.testing.assert_allclose(pay_orig - past, df0 * pay_res, rtol=1e-12, atol=1e-15)
    assert np.std(pay_res) > 0
    return note


def test_residual_note_construction(discount: DiscountCurve, rng: np.random.Generator) -> None:
    """Explicit residual notes reprice the original note with the state held (identity above) for
    a growing-coupon step-down autocall with a non-call period (dates 2, 3 explicit; date 1 through
    ``aged``), its guaranteed-coupon variant (the past coupons of the dropped dates), a memory
    Phoenix with a quarterly American knock-in (alive, missed-coupon carry and knocked-in
    histories) under both final-date readings; the call-spread split of :class:`SmoothedAutocall`;
    the shifted levels per scope; validation."""
    ac = Autocall(
        OBS,
        discount,
        spot_reference=S0,
        coupons=0.06,
        ki_level=0.6,
        ki_type="european",
        autocall_barriers=[1.0, 0.97, 0.9],
        non_call_periods=1,
    )
    n1 = _identity_check(ac, 1, 95.0, rng)
    assert residual_note(ac, 1, spot=95.0)[1]["construction"] == "aged"
    np.testing.assert_allclose(n1.observation_times, [DT, 1.0 + DT, 2.0 + DT])
    n2 = _identity_check(ac, 2, 95.0, rng)
    n3 = _identity_check(ac, 3, 95.0, rng)
    info2 = residual_note(ac, 2, spot=95.0)[1]
    assert info2["construction"] == "explicit" and info2["ki_history"] == "european"
    np.testing.assert_allclose(n2.observation_times, [DT, 1.0 + DT])
    assert n2.coupons == (0.12, 0.18) and n3.coupons == (0.18,)  # re-indexed growing coupon
    np.testing.assert_allclose(n2.autocall_barriers, [0.97, 0.9])
    assert n2.non_call_periods == 0 and n2.ki_level == 0.6 and n2.spot_reference == S0
    assert callable_dates(ac) == [2, 3] and callable_dates(n2) == [1, 2]
    # guaranteed coupons: the coupon of the dropped date 1 was paid in the past (flat schedule)
    g2 = _identity_check(ac.replace(guaranteed_coupons=True), 2, 95.0, rng)
    assert g2.guaranteed_coupons and g2.coupons == (0.06, 0.06)
    quarterly = np.linspace(0.0, 3.0, 13)
    for final in ("knock_in", "coupon_barrier"):
        ph = Phoenix(
            OBS,
            discount,
            spot_reference=S0,
            coupon=0.05,
            coupon_barrier=0.7,
            memory=True,
            ki_level=0.6,
            ki_type="american",
            ki_monitoring="discrete",
            ki_fixing_times=quarterly,
            final_redemption=final,
        )
        # alive, coupons paid: the schedule keeps t = 0 and the dates after the valuation time
        alive = _identity_check(ph, 2, 95.0, rng)
        info = residual_note(ph, 2, spot=95.0)[1]
        assert info["ki_history"] == "alive" and info["memory_carry"] == 0.0
        assert alive.ki_fixing_times is not None and alive.ki_fixing_times[0] == 0.0
        np.testing.assert_allclose(
            alive.ki_fixing_times[1:], quarterly[quarterly > 2.0 - DT] - (2.0 - DT)
        )
        assert alive.coupons == (0.05, 0.05)
        # below the coupon barrier, above the knock-in: the missed coupon is carried
        carry = _identity_check(ph, 2, 65.0, rng)
        assert carry.coupons == (0.10, 0.05) and residual_note(ph, 3, spot=65.0)[1][
            "memory_carry"
        ] == pytest.approx(0.10)
        # below the knock-in barrier: the knocked-in note is the European knock-in at 100% (or at
        # the coupon barrier under the "coupon_barrier" reading), at every date including the first
        for date in (1, 2, 3):
            knocked = _identity_check(ph, date, 55.0, rng)
            info = residual_note(ph, date, spot=55.0)[1]
            assert info["ki_history"] == "knocked_in" and info["construction"] == "explicit"
            assert knocked.ki_type == "european" and knocked.ki_fixing_times is None
            assert knocked.ki_level == (1.0 if final == "knock_in" else 0.7)
    # continuous monitoring: at or below the barrier is knocked in; a custom schedule that never
    # observed the level before the valuation time leaves the note alive
    cont = ph.replace(ki_monitoring="continuous", ki_fixing_times=None)
    assert residual_note(cont, 2, spot=60.0)[1]["ki_history"] == "knocked_in"
    assert residual_note(cont, 2, spot=60.5)[1]["ki_history"] == "alive"
    late = ph.replace(ki_fixing_times=[2.5, 3.0])
    assert residual_note(late, 2, spot=55.0)[1]["ki_history"] == "alive"
    # the smoothed note: the split identity and the call-spread weight
    times = np.array([0.0, DT, 1.0 + DT, 2.0 + DT])
    spots = np.array(
        [
            [95.0, 101.0, 50.0, 50.0],
            [95.0, 99.0, 50.0, 50.0],
            [95.0, 99.5, 120.0, 50.0],
            [95.0, 100.4, 90.0, 50.0],
        ]
    )
    ps, idx = _paths(spots, times), FixingIndex(times)
    plain = ac.replace(non_call_periods=0)
    n0 = residual_note(plain, 1, spot=95.0)[0]
    sm = SmoothedAutocall(n0, 2.0)
    np.testing.assert_allclose(sm.weight(spots[:, 1]), [1.0, 0.0, 0.25, 0.7])
    call, cont_ = sm.note_call.payoff(ps, idx), sm.note_cont.payoff(ps, idx)
    th = sm.weight(spots[:, 1])
    np.testing.assert_allclose(sm.payoff(ps, idx), th * call + (1 - th) * cont_, rtol=1e-12)
    exact = n0.payoff(ps, idx)
    ind = (spots[:, 1] >= 100.0).astype(float)
    np.testing.assert_allclose(exact, ind * call + (1 - ind) * cont_, rtol=1e-12)
    np.testing.assert_allclose(sm.payoff(ps, idx)[:2], exact[:2], rtol=1e-12)  # outside the band
    assert (
        "Call-spread smoothed" in repr(sm) and sm.fixing_times.tolist() == n0.fixing_times.tolist()
    )
    # barrier shift: monitored levels only — every autocall barrier or the given dates' only, the
    # knock-in level always; the coupon barrier never
    sh = shifted_levels(ph, 0.02)
    np.testing.assert_allclose(sh.autocall_barriers, 1.02)
    assert sh.ki_level == pytest.approx(0.612) and sh.coupon_barrier == 0.7
    assert shifted_levels(ph, 0.0) is ph
    one = shifted_levels(ac, 0.02, dates=(2,))
    np.testing.assert_allclose(one.autocall_barriers, [1.0, 0.97 * 1.02, 0.9])
    assert one.ki_level == pytest.approx(0.612) and one.coupon_barrier is None
    assert len(default_profile_shifts()) == 33 and 0.0 in default_profile_shifts()
    for bad in (
        lambda: residual_note(ac, 0, spot=95.0),
        lambda: residual_note(ac, 4, spot=95.0),
        lambda: residual_note(ac, 1, spot=95.0, dt=1.0),
        lambda: residual_note(ac, 1, spot=0.0),
        lambda: residual_note(
            ac.replace(observation_times=[1.0, 1.0 + 1 / 500, 3.0]), 2, spot=95.0
        ),
        lambda: SmoothedAutocall(residual_note(ac, 1, spot=95.0)[0], 2.0),  # non-call first date
        lambda: SmoothedAutocall(n0, 0.0),
        lambda: shifted_levels(ph, -1.5),
        lambda: shifted_levels(ph, 0.7),  # the knock-in level would exceed 100%
        lambda: shifted_levels(ac, 0.02, dates=(4,)),
        lambda: shifted_levels(ac, 0.02, dates=(1.5,)),
        lambda: default_profile_shifts(fine_step=0.02, step=0.01),
        lambda: default_profile_shifts(fine_width=0.2),
    ):
        with pytest.raises(ValueError):
            bad()


# --------------------------------------------------------------------------------------------
# Black–Scholes bed
# --------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bs_engine() -> tuple[RiskEngine, RiskState]:
    state = _flat_state(SIGMA)
    sim = SimConfig(n_paths=40_000, dt_max=1.0 / 50.0, chunk_size=20_000, seed=5)
    return RiskEngine(BSBuilder(state), sim), state


def _digital_stencil(S: float, B: float, w: float, c: float, dt: float, h: float) -> float:
    """Central log-spot stencil ``±h`` of the Black–Scholes price of ``DF [1 + c θ_w(S_dt)]``: the
    expectation of the profile's delta estimator (the log-Euler step is exact)."""
    df = float(np.exp(-R * dt))

    def price(s: float) -> float:
        if w == 0.0:
            d2 = (np.log(s / B) + (R - Q - 0.5 * SIGMA**2) * dt) / (SIGMA * np.sqrt(dt))
            return df * (1.0 + c * float(norm_cdf(d2)))
        lo = float(bs_price(s, B - 0.5 * w, dt, SIGMA, R, Q, 1))
        hi = float(bs_price(s, B + 0.5 * w, dt, SIGMA, R, Q, 1))
        return df + c / w * (lo - hi)

    su, sd = S * np.exp(h), S * np.exp(-h)
    return (price(su) - price(sd)) / (su - sd)


def _digital_exact(S: float, B: float, w: float, c: float, dt: float) -> float:
    """Exact delta of ``DF [1 + c θ_w(S_dt)]``: ``DF c φ(d₂)/(S σ √dt)`` for the digital, the
    unit-height call spread's delta for ``w > 0``."""
    if w == 0.0:
        d2 = (np.log(S / B) + (R - Q - 0.5 * SIGMA**2) * dt) / (SIGMA * np.sqrt(dt))
        return float(np.exp(-R * dt)) * c * float(norm_pdf(d2)) / (S * SIGMA * np.sqrt(dt))
    lo = float(bs_delta(S, B - 0.5 * w, dt, SIGMA, R, Q, 1))
    hi = float(bs_delta(S, B + 0.5 * w, dt, SIGMA, R, Q, 1))
    return c / w * (lo - hi)


def test_spot_profile_single_date_digital_black_scholes(bs_engine) -> None:
    """A single-date autocall (AC 100%, coupon 6%, knock-in disabled) one business day before its
    date is ``DF(dt) [1 + c 1{S ≥ B}]``: the model-regime delta profile equals the Black–Scholes
    digital stencil within 3 stderr at three spots and peaks at the barrier; the ``w = 2%``
    smoothing equals the call-spread digital stencil; at a 0.25% stencil both profiles equal their
    analytic stencils within 3 stderr while those stencils lie within 1% of the exact digital /
    call-spread delta (the 1% stencil reads 4–10% low: its own smoothing, recorded in ``attrs``);
    ``barrier_shift = +2%`` moves the peak to ``1.02 × AC``; the profile saturates at the bond and
    bond-plus-coupon values ±10% away."""
    engine, state = bs_engine
    disc = surface_of(state).discount
    one = Autocall([1.0], disc, spot_reference=S0, coupons=0.06, ki_level=1e-9, ki_type="european")
    profiles = autocall_spot_profiles(engine, one, state)
    assert list(profiles) == [1]
    f = profiles[1]
    assert len(f) == 33 and f.attrs["construction"] == "aged" and f.attrs["smoothed"] == "none"
    assert (
        f.attrs["valuation_time"] == pytest.approx(1.0 - DT) and f.attrs["monitored_barrier"] == S0
    )
    assert f.attrs["stencil_width"] == pytest.approx(S0 * (np.exp(0.01) - np.exp(-0.01)))
    assert f.attrs["barrier_shift_scope"] == "date" and f.attrs["residual_ki_level"] == 1e-9
    assert {"price", "delta", "gamma", "vega", "gamma_fd", "spot_over_barrier"} <= set(f.columns)
    df = float(disc.df(DT))
    for rel in (-0.01, 0.0, 0.01):
        row = f[np.isclose(f["spot_over_monitored"], 1.0 + rel)].iloc[0]
        ref = _digital_stencil(row["spot"], S0, 0.0, 0.06, DT, 0.01)
        assert abs(row["delta"] - ref) < 3.0 * row["delta_stderr"] + 1e-12, (row["delta"], ref)
    peak = f.loc[f["delta"].idxmax(), "spot_over_monitored"]
    assert abs(peak - 1.0) < 0.0051, peak
    assert f["delta"].max() > 10.0 * f["delta"].iloc[0] and f["gamma"].abs().max() > 0
    lo, hi = f.iloc[0], f.iloc[-1]
    assert abs(lo["price"] - df) < 3.0 * lo["price_stderr"] + 1e-9
    assert abs(hi["price"] - 1.06 * df) < 3.0 * hi["price_stderr"] + 1e-9
    # call-spread smoothing (w = 2% of S_ref = 2 spot units): the whole jump of the note at the
    # date is the digital here, so the smoothed profile is the call-spread digital
    sm = autocall_spot_profiles(engine, one, state, smoothing=0.02, with_vega=False)[1]
    assert (
        sm.attrs["smoothing_width"] == pytest.approx(2.0) and "call spread" in sm.attrs["smoothed"]
    )
    assert "Call-spread smoothed" in sm.attrs["residual_note"]
    for rel in (-0.01, 0.0, 0.01):
        row = sm[np.isclose(sm["spot_over_monitored"], 1.0 + rel)].iloc[0]
        ref = _digital_stencil(row["spot"], S0, 2.0, 0.06, DT, 0.01)
        assert abs(row["delta"] - ref) < 3.0 * row["delta_stderr"] + 1e-12, (row["delta"], ref)
        print(
            f"smoothed delta at {rel:+.2%}: {row['delta']:.5f} ± {row['delta_stderr']:.1e}, "
            f"stencil {ref:.5f}, exact call-spread derivative "
            f"{_digital_exact(row['spot'], S0, 2.0, 0.06, DT):.5f}"
        )
    assert sm["delta"].max() < f["delta"].max()  # smoothing lowers the spike
    # convergence of the stencil to the exact delta: at 0.25% the Monte Carlo delta equals the
    # analytic 0.25% stencil within 3 stderr (the estimator identity, twice the noise of the 1%
    # stencil) and that stencil lies within 1% of the exact derivative (a = h/(σ√dt) ≈ 0.2,
    # relative bias ≈ a²/6), where the 1% stencil (a ≈ 0.8) reads 4–10% low
    three = (-0.01, 0.0, 0.01)
    for w in (0.0, 2.0):
        fine = autocall_spot_profiles(
            engine, one, state, shifts=three, size=0.0025, smoothing=w / S0, with_vega=False
        )[1]
        assert fine.attrs["stencil_width"] == pytest.approx(S0 * (np.exp(0.0025) - np.exp(-0.0025)))
        for rel in three:
            row = fine[np.isclose(fine["spot_over_monitored"], 1.0 + rel)].iloc[0]
            ref = _digital_stencil(row["spot"], S0, w, 0.06, DT, 0.0025)
            exact = _digital_exact(row["spot"], S0, w, 0.06, DT)
            coarse = _digital_stencil(row["spot"], S0, w, 0.06, DT, 0.01)
            assert abs(row["delta"] - ref) < 3.0 * row["delta_stderr"] + 1e-12, (row["delta"], ref)
            assert abs(ref - exact) < 0.01 * exact, (ref, exact)
            assert exact - coarse > 0.03 * exact, (coarse, exact)
            print(
                f"w={w:g} at {rel:+.2%}: 0.25% stencil MC {row['delta']:.5f} ± "
                f"{row['delta_stderr']:.1e}, analytic {ref:.5f}, exact {exact:.5f}, 1% stencil "
                f"{coarse:.5f}"
            )
    shifted = autocall_spot_profiles(engine, one, state, barrier_shift=0.02, with_vega=False)[1]
    assert shifted.attrs["monitored_barrier"] == pytest.approx(102.0)
    assert shifted.attrs["autocall_barrier"] == S0 and shifted.attrs["barrier_shift"] == 0.02
    assert shifted.attrs["residual_autocall_barriers"] == pytest.approx((1.02,))
    peak_s = shifted.loc[shifted["delta"].idxmax(), "spot_over_barrier"]
    assert abs(peak_s - 1.02) < 0.0051 * 1.02, peak_s
    assert engine.n_calibrations == 1  # model regime: no recalibration along the profiles
    for bad in (
        lambda: autocall_spot_profiles(engine, one.replace(non_call_periods=1), state, dates=[1]),
        lambda: autocall_spot_profiles(engine, one, state, smoothing=-0.01),
        lambda: autocall_spot_profiles(engine, one, state, dates=[1.5]),
        lambda: autocall_spot_profiles(engine, one, state, dates=[1, 1]),
        lambda: autocall_spot_profiles(engine, one, state, barrier_shift_scope="first"),
        lambda: autocall_spot_profiles(engine, one, state, size=0.0),
    ):
        with pytest.raises(ValueError):
            bad()


def test_spot_profiles_multi_date_black_scholes(bs_engine) -> None:
    """The 3y note's profiles priced end to end through the explicit residual notes: at the final
    date (one day left, knock-in at 60% out of reach) the price is ``DF(dt) (1 + 3c)`` 5% above the
    barrier and ``DF(dt)`` 5% below; at date 2 the smoothed residual note (a later digital left)
    lowers the peak delta; the shift scopes move the residual note's first barrier only (``"date"``)
    or every barrier (``"all"``), the knock-in level in both; the American-knock-in Phoenix runs
    through the profile, the barrier risk (no components) and the likelihood-ratio table (no
    ``put_digital``) on a light engine."""
    engine, state = bs_engine
    disc = surface_of(state).discount
    note = Autocall(OBS, disc, spot_reference=S0, coupons=0.06, ki_level=0.6, ki_type="european")
    out = autocall_spot_profiles(
        engine, note, state, dates=[3], shifts=(-0.05, 0.05), with_vega=False
    )
    assert list(out) == [3]
    f3 = out[3]
    assert f3.attrs["construction"] == "explicit" and f3.attrs["date"] == 3
    assert f3.attrs["valuation_time"] == pytest.approx(3.0 - DT)
    assert f3.attrs["residual_autocall_barriers"] == (1.0,)
    np.testing.assert_allclose(f3["spot_over_barrier"], [0.95, 1.05])
    df = float(disc.df(DT))
    lo, hi = f3.iloc[0], f3.iloc[1]
    assert abs(lo["price"] - df) < 3.0 * lo["price_stderr"] + 1e-9, (lo["price"], df)
    assert abs(hi["price"] - 1.18 * df) < 3.0 * hi["price_stderr"] + 1e-9, (hi["price"], 1.18 * df)
    five = (-0.02, -0.01, 0.0, 0.01, 0.02)
    plain = autocall_spot_profiles(engine, note, state, dates=[2], shifts=five, with_vega=False)[2]
    sm = autocall_spot_profiles(
        engine, note, state, dates=[2], shifts=five, smoothing=0.02, with_vega=False
    )[2]
    assert plain.attrs["residual_autocall_barriers"] == (1.0, 1.0)
    assert "Call-spread smoothed" in sm.attrs["residual_note"] and "date 2" in sm.attrs["smoothed"]
    assert np.all(np.isfinite(sm[["price", "delta", "gamma"]].to_numpy()))
    assert sm["delta"].max() < plain["delta"].max() - 3.0 * float(plain["delta_stderr"].max())
    assert abs(int(plain["delta"].to_numpy().argmax()) - 2) <= 1  # the spike sits at the barrier
    for scope, barriers in (("date", (1.02, 1.0)), ("all", (1.02, 1.02))):
        sc = autocall_spot_profiles(
            engine,
            note,
            state,
            dates=[2],
            shifts=(0.0,),
            barrier_shift=0.02,
            barrier_shift_scope=scope,
            with_vega=False,
        )[2]
        assert sc.attrs["barrier_shift_scope"] == scope
        assert sc.attrs["residual_autocall_barriers"] == pytest.approx(barriers)
        assert sc.attrs["residual_ki_level"] == pytest.approx(0.612)
        assert sc.attrs["monitored_barrier"] == pytest.approx(102.0)
        assert sc["spot_over_barrier"].iloc[0] == pytest.approx(1.02)
        assert sc["spot_over_monitored"].iloc[0] == pytest.approx(1.0)
    # American knock-in (quarterly, memory Phoenix) on a light engine: keys and shapes only
    ph = Phoenix(
        OBS,
        disc,
        spot_reference=S0,
        coupon=0.05,
        coupon_barrier=0.7,
        memory=True,
        ki_level=0.6,
        ki_type="american",
        ki_monitoring="discrete",
        ki_fixing_times=np.linspace(0.0, 3.0, 13),
    )
    light = RiskEngine(
        BSBuilder(state), SimConfig(n_paths=4_000, dt_max=1.0 / 12.0, chunk_size=4_000, seed=5)
    )
    pf = autocall_spot_profiles(light, ph, state, dates=[2], shifts=(0.0,), with_vega=False)[2]
    assert pf.attrs["ki_history"] == "alive" and pf.attrs["construction"] == "explicit"
    kb = ki_put_barrier_risk(light, ph, state)
    assert set(kb) == {"dprice_dB", "dprice_dB_pct", "dpki_dB", "dpki_dB_pct"}
    assert kb["dprice_dB"].extra["ki_type"] == "american" and np.isfinite(kb["dprice_dB"].value)
    model = light.builder.build(state, "recalibrate")
    tab = lr_cross_check(ph, model, light.sim, engine=light, state=state)
    assert set(tab["leg"]) == {"product", "autocall_1", "autocall_2", "autocall_3"}
    assert len(tab) == 8 and np.all(np.isfinite(tab[["lr", "bump", "z"]].to_numpy()))


def test_ki_put_barrier_risk_black_scholes() -> None:
    """Never-autocalling 3y note, European knock-in at 60%: ``∂price/∂B`` of the put leg equals
    ``−(K − B) DF φ(d₂(B)) / (K B σ √T)`` within 3 stderr, its vanilla and digital components
    ``−DF N(−d₂)/K`` and ``−[(K − B) DF f(B) − DF N(−d₂)]/K``, ``∂P(KI)/∂B`` the density ``f(B)``;
    the per-1% scaling; the ±5% profile has 21 rows centred on the barrier and every row's delta
    equals the closed form ``[e^{−qT} N(−d₁) + (K − B) DF φ(d₂)/(S σ √T)]/K`` within 3 stderr, its
    gamma the closed form's derivative (the 1% log-stencil bias is ``O((h/σ√T)²) ≈ 1e-4``
    relative); ``spot_over_barrier`` stays contractual under a shift."""
    state = _flat_state(SIGMA)
    disc = surface_of(state).discount
    engine = RiskEngine(
        BSBuilder(state), SimConfig(n_paths=40_000, dt_max=1.0 / 26.0, chunk_size=20_000, seed=5)
    )
    never = Autocall(
        OBS,
        disc,
        spot_reference=S0,
        coupons=0.0,
        ki_level=0.6,
        ki_type="european",
        autocall_barriers=np.inf,
    )
    out = ki_put_barrier_risk(engine, never, state)
    b, k, t = 60.0, S0, 3.0
    d2 = (np.log(S0 / b) + (R - Q - 0.5 * SIGMA**2) * t) / (SIGMA * np.sqrt(t))
    df_t = float(np.exp(-R * t))
    dens = float(norm_pdf(d2)) / (b * SIGMA * np.sqrt(t))
    refs = {
        "dprice_dB": -(k - b) / k * df_t * dens,
        "dprice_dB_vanilla": -df_t * float(norm_cdf(-d2)) / k,
        "dprice_dB_digital": -((k - b) * df_t * dens - df_t * float(norm_cdf(-d2))) / k,
        "dpki_dB": dens,
    }
    for key, ref in refs.items():
        s = out[key]
        assert abs(s.value - ref) < 3.0 * s.stderr + 1e-12, (key, s, ref)
        assert s.scheme == "central" and s.size == 0.005 and s.extra["barrier"] == b
        assert out[f"{key}_pct"].value == pytest.approx(s.value * 0.01 * b)
        assert out[f"{key}_pct"].unit == "per 1% of barrier"
    assert out["dprice_dB"].value < 0 and out["dpki_dB"].value > 0
    assert out["dprice_dB"].value == pytest.approx(
        out["dprice_dB_vanilla"].value + out["dprice_dB_digital"].value, abs=1e-12
    )
    print(
        "KI put dprice/dB per 1% of barrier:",
        out["dprice_dB_pct"],
        "dP(KI)/dB per 1%:",
        out["dpki_dB_pct"],
    )
    light = RiskEngine(
        BSBuilder(state), SimConfig(n_paths=20_000, dt_max=1.0 / 12.0, chunk_size=20_000, seed=5)
    )
    prof = ki_barrier_profile(light, never, state)
    assert len(prof) == 21 and abs(prof["spot_over_barrier"].iloc[10] - 1.0) < 1e-9
    assert {"price", "delta", "gamma", "gamma_fd", "spot_over_monitored"} <= set(prof.columns)
    assert np.all(np.isfinite(prof[["price", "delta", "gamma"]].to_numpy()))
    assert prof.attrs["barrier"] == b and prof.attrs["monitored_barrier"] == b
    assert np.all(prof["delta"] > 0)  # short put: long the underlying around the barrier
    # row by row against the closed form of the never-autocalling note (module docstring of
    # digital_risk): price = DF − [put(S; B) + (K − B) DF N(−d₂)]/K
    sig_t = SIGMA * np.sqrt(t)
    worst_d = worst_g = worst_fd = 0.0
    for _, row in prof.iterrows():
        s = float(row["spot"])
        d1 = (np.log(s / b) + (R - Q + 0.5 * SIGMA**2) * t) / sig_t
        d2 = d1 - sig_t
        ref_delta = (
            np.exp(-Q * t) * float(norm_cdf(-d1))
            + (k - b) * df_t * float(norm_pdf(d2)) / (s * sig_t)
        ) / k
        ref_gamma = (
            -(
                float(bs_gamma(s, b, t, SIGMA, R, Q))
                + (k - b) * df_t * float(norm_pdf(d2)) * (1.0 + d2 / sig_t) / (s * s * sig_t)
            )
            / k
        )
        assert abs(row["delta"] - ref_delta) < 3.0 * row["delta_stderr"] + 1e-12, (s, ref_delta)
        assert abs(row["gamma"] - ref_gamma) < 3.0 * row["gamma_stderr"] + 1e-12, (s, ref_gamma)
        worst_d = max(worst_d, abs(row["delta"] - ref_delta) / row["delta_stderr"])
        worst_g = max(worst_g, abs(row["gamma"] - ref_gamma) / row["gamma_stderr"])
        if np.isfinite(row["gamma_fd"]):
            # the price profile's second difference on the 0.5% grid: the same CRN paths, the
            # digital's noise growing as h^{-3/2} from the 1% stencil (factor 2^{1.5})
            tol = 3.0 * 2.0**1.5 * row["gamma_stderr"]
            assert abs(row["gamma_fd"] - ref_gamma) < tol + 1e-12, (s, row["gamma_fd"], ref_gamma)
            worst_fd = max(worst_fd, abs(row["gamma_fd"] - ref_gamma) / tol)
    print(
        f"KI profile worst |z|: delta {worst_d:.2f}, gamma {worst_g:.2f}; gamma_fd worst "
        f"{worst_fd:.2f} of its tolerance"
    )
    shifted = ki_barrier_profile(light, never, state, barrier_shift=0.01, width=0.01, step=0.01)
    assert shifted.attrs["monitored_barrier"] == pytest.approx(60.6) and len(shifted) == 3
    assert shifted["spot_over_barrier"].iloc[1] == pytest.approx(1.01)  # contractual level
    assert shifted["spot_over_monitored"].iloc[1] == pytest.approx(1.0)  # monitored level
    with pytest.raises(ValueError):
        ki_put_barrier_risk(engine, never, state, size=1.5)


def test_expected_life_sensitivity_black_scholes(bs_engine) -> None:
    """3y annual autocall under Black–Scholes: ``E[life]`` in (0, 3]; a higher spot autocalls
    earlier (``∂E[life]/∂ln S < 0`` at 3 stderr); the vol sensitivity is finite (reported)."""
    engine, state = bs_engine
    disc = surface_of(state).discount
    note = Autocall(OBS, disc, spot_reference=S0, coupons=0.06, ki_level=0.6, ki_type="european")
    with pytest.raises(ValueError):
        expected_life_sensitivity(engine, note, state, spot_size=np.nan)
    out = expected_life_sensitivity(engine, note, state)
    life, d_s, d_v = out["life"], out["dlife_dlnS"], out["dlife_dsigma"]
    assert 0.0 < life.value <= 3.0 and life.stderr > 0 and life.unit == "years"
    assert d_s.value < -3.0 * d_s.stderr, d_s
    assert np.isfinite(d_v.value) and d_v.stderr > 0
    assert d_s.unit == "years per unit ln S" and d_v.unit == "years per vol point"
    assert (
        d_s.scheme == "central"
        and d_v.scheme == "forward"
        and d_v.extra["variant"] == "sticky_leverage"
    )
    print("expected life", life, d_s, d_v)


def test_lr_cross_check_black_scholes() -> None:
    """Black–Scholes, 3y annual autocall, 100k paths, ``first_step = 1/52``: likelihood-ratio
    delta and vega of the note and of every digital leg agree with the bump estimates within 4
    stderr at a 1% (1 vp) and a 0.2% (0.2 vp) stencil.  Standard errors: the likelihood-ratio
    error is stencil-independent while the bump's grows as ``1/√h`` on a discontinuous payoff, so
    the ratio ``lr/bump`` falls by ``≈ √(0.2)`` between the stencils; at the 1% stencil the bump is
    the less noisy estimator on these digitals (ratio > 1, the reported finding), at 0.2% the
    likelihood ratio wins on the conditional digitals.  Curve binding: a note built on another
    discount curve gives the same rows (both sides discount with the state's curve)."""
    state = _flat_state(SIGMA)
    disc = surface_of(state).discount
    sim = SimConfig(n_paths=100_000, dt_max=1.0 / 52.0, chunk_size=25_000, seed=17)
    engine = RiskEngine(BSBuilder(state), sim)
    model = engine.builder.build(state, "recalibrate")
    note = Autocall(OBS, disc, spot_reference=S0, coupons=0.06, ki_level=0.6, ki_type="european")
    table = lr_cross_check(
        note,
        model,
        sim,
        engine=engine,
        state=state,
        spot_size=(0.01, 0.002),
        vol_size=(0.01, np.float64(0.002)),
    )
    print(table.to_string())
    assert set(table["leg"]) == {"product", "autocall_1", "autocall_2", "autocall_3", "put_digital"}
    assert len(table) == 5 * 4 and np.all(table["lr_n_paths"] == 100_000)
    assert np.allclose(table["requested_size"], table["bump_size"])  # flat surface: no halving
    digital = table[table["leg"] != "product"]
    assert np.all(np.abs(digital["z"]) < 4.0), digital[["leg", "greek", "requested_size", "z"]]
    assert np.all(np.abs(table["z"]) < 4.0)
    assert np.allclose(table.loc[table["greek"] == "delta", "first_step"], 1.0 / 52.0)
    assert np.all(table["lr_stderr"] > 0) and np.all(table["bump_stderr"] > 0)
    for (leg, greek), grp in digital.groupby(["leg", "greek"]):
        wide = grp[grp["requested_size"] == 0.01].iloc[0]
        fine = grp[grp["requested_size"] == 0.002].iloc[0]
        assert wide["lr_stderr"] == fine["lr_stderr"]  # one likelihood-ratio pass
        assert (
            fine["bump_stderr"] > wide["bump_stderr"]
        )  # the bump's error grows as the stencil shrinks
        assert fine["stderr_ratio"] < wide["stderr_ratio"]
        print(
            f"{leg} {greek}: LR/bump stderr ratio {wide['stderr_ratio']:.2f} at 1%, {fine['stderr_ratio']:.2f} at 0.2%"
        )
    d_fine = digital[(digital["greek"] == "delta") & (digital["requested_size"] == 0.002)]
    # the bump's error diverges as h -> 0, so the likelihood ratio wins from some stencil on: at
    # 0.2% it already does on the conditional digitals of the later dates (measured 0.56-0.61)
    assert d_fine["stderr_ratio"].min() < 1.0, d_fine[["leg", "stderr_ratio"]]
    # curve binding: the same note built on a 5% curve is rebound to the state's curve on both
    # sides, so the rows coincide (a light engine; same draws)
    small = SimConfig(n_paths=4_000, dt_max=1.0 / 12.0, chunk_size=4_000, seed=3)
    light = RiskEngine(BSBuilder(state), small)
    model_s = light.builder.build(state, "recalibrate")
    other = note.with_discount(ForwardCurve.flat(S0, 0.05, 0.0).rate_curve)
    t_state = lr_cross_check(note, model_s, small, engine=light, state=state)
    t_other = lr_cross_check(other, model_s, small, engine=light, state=state)
    cols = ["lr", "lr_stderr", "bump", "bump_stderr"]
    np.testing.assert_allclose(t_other[cols].to_numpy(), t_state[cols].to_numpy(), rtol=1e-12)
    assert abs(float(disc.df(3.0)) - float(other.df(3.0))) > 0.05  # the curves do differ
    for bad in (
        lambda: lr_cross_check(note, model, sim, engine=engine, state=state.with_spot(101.0)),
        lambda: lr_cross_check(note, model, sim, engine=engine, state=state, spot_size=np.int64(0)),
        lambda: lr_cross_check(note, model, sim, engine=engine, state=state, vol_size=()),
        lambda: lr_cross_check(note, model, sim, engine=engine, state=state, vol_size=np.nan),
    ):
        with pytest.raises(ValueError):
            bad()


# --------------------------------------------------------------------------------------------
# local vol on the reference surface: the sign test
# --------------------------------------------------------------------------------------------


def test_sign_test_local_vol() -> None:
    """3y annual autocall (AC 100%, c = 6%, European KI 60%) under local vol on the reference
    surface, 40k paths.  Short vol: parallel vega < 0 at 3 stderr.  Long forward: model delta > 0,
    repo delta (``q`` up, forward down, discounting held) < 0 and the equity legs' rho (the note
    less its bond leg, per path) > 0, each at 3 stderr; the curve sensitivities use a 25 bp shift
    reported per bp (the digital legs' per-bp error scales as ``1/√h``).  The note's own rho is
    pinned *negative* at 3 stderr as a regression guard pending the owner's sign-off on "rho >
    0": the funded par note has a duration of about ``E[life] ≈ 1.9y`` (bond leg about −2e-4 per
    bp) that outweighs the forward exposure (equity legs about +1.2e-4 per bp).  Skew under the
    fixed-ATM rotation (put wing up, call wing down): the note is short skew at the 3y pillar and
    globally; the knock-in put leg is short skew at 3y; the 1y autocall digital is long skew at
    its own date (a digital call is a call spread); the bond leg (par on survival) is short skew
    at 3y.  The legs' sensitivities add up to the note's exactly (same paths); the total row's
    projection sum carries the parallel vega's own standard error."""
    state = RiskState(load_yaml(SPEC_1F, CalibrationSpec))
    sim = SimConfig(n_paths=40_000, dt_max=1.0 / 50.0, chunk_size=20_000, seed=11)
    engine = RiskEngine(LVBuilder(state), sim)
    disc = surface_of(state).discount
    note = Autocall(
        OBS, disc, spot_reference=state.spot, coupons=0.06, ki_level=0.6, ki_type="european"
    )
    pillars = (1.0, 2.0, 3.0)
    frame = structure_vega_skew(engine, note, state, pillars)
    legs = ["autocall_1", "autocall_2", "autocall_3", "bond", "put_vanilla", "put_digital"]
    assert list(dict.fromkeys(frame["leg"])) == ["product", *legs, "put"]
    assert (
        len(frame) == 8 * 4
        and frame.attrs["pillars"] == pillars
        and "rotation" in frame.attrs["skew_bump"]
    )
    tot = frame[frame["kind"] == "total"].set_index("leg")
    pil = frame[frame["kind"] == "pillar"]

    def at(leg: str, pillar: float, col: str) -> tuple[float, float]:
        r = pil[(pil["leg"] == leg) & (pil["pillar"] == pillar)].iloc[0]
        return float(r[col]), float(r[f"{col}_stderr"])

    # short vol
    v, se = tot.loc["product", "vega_wave"], tot.loc["product", "vega_wave_stderr"]
    assert v < -3.0 * se, (v, se)
    assert tot.loc["product", "vega_projection"] == pytest.approx(v, abs=1e-12)
    assert tot.loc["product", "vega_projection_stderr"] == se  # the telescoped sum's own error
    # additivity over the decomposition (the legs sum to the note path by path)
    for col in ("vega_wave", "skew"):
        assert tot.loc[legs, col].sum() == pytest.approx(tot.loc["product", col], abs=1e-9)
        for p in pillars:
            assert sum(at(leg, p, col)[0] for leg in legs) == pytest.approx(
                at("product", p, col)[0], abs=1e-9
            )
    assert tot.loc["put", "skew"] == pytest.approx(
        tot.loc["put_vanilla", "skew"] + tot.loc["put_digital", "skew"], abs=1e-9
    )
    # skew signs under the rotation
    s3, se3 = at("product", 3.0, "skew")
    assert s3 < -3.0 * se3, (s3, se3)  # short skew at maturity
    g, ge = tot.loc["product", "skew"], tot.loc["product", "skew_stderr"]
    assert g < -3.0 * ge, (g, ge)  # and under the global rotation
    p3, pe3 = at("put", 3.0, "skew")
    assert p3 < -3.0 * pe3, (p3, pe3)  # the knock-in put leg is short skew at 3y
    a1, ae1 = at("autocall_1", 1.0, "skew")
    assert a1 > 3.0 * ae1, (a1, ae1)  # the digital at the money is long skew at its date
    b3, be3 = at("bond", 3.0, "skew")
    assert b3 < -3.0 * be3, (b3, be3)
    for p in pillars:
        s, se_ = at("product", p, "skew")
        ach = float(pil[(pil["leg"] == "product") & (pil["pillar"] == p)]["skew_achieved"].iloc[0])
        print(f"note skew at {p:g}y: {s:.5f} +/- {se_:.1e} (achieved slope {ach:.4f})")
    print(
        "note global skew rotation:",
        g,
        "±",
        ge,
        "| legs:",
        {leg: round(float(tot.loc[leg, "skew"]), 5) for leg in legs},
    )
    # long forward: delta, repo delta and the equity legs' rho; the note's rho pinned negative
    d, _ = delta_gamma(engine, note, state, "model")
    assert d.value > 3.0 * d.stderr, d
    bp = 1e-4
    h = 25.0 * bp  # a 25 bp curve shift for clean signs on the digital legs, reported per bp
    c = bp / h

    def curve_sens(name: str, prod, **shift: float):  # type: ignore[no-untyped-def]
        st = state.with_rate_shift(**shift, label=name)
        return engine.combination(
            name,
            prod,
            [(st, "recalibrate", c), (state, "recalibrate", -c)],
            unit="per bp",
            size=h,
            scheme="forward",
        )

    repo = curve_sens("repo_delta", note, dq=h)
    rho = curve_sens("rho", note, dr=h)
    assert repo.value < -3.0 * repo.stderr, repo
    rho_legs = {leg.key: curve_sens("rho", leg, dr=h) for leg in note.decompose()}
    assert sum(s.value for s in rho_legs.values()) == pytest.approx(rho.value, abs=1e-9)
    print(
        "delta",
        d,
        "| repo delta",
        repo,
        "| rho",
        rho,
        "| rho by leg:",
        {k: (round(s.value, 7), round(s.stderr, 7)) for k, s in rho_legs.items()},
    )
    # the bond leg: the discounting of the par redemption, duration about E[life] = 1.9y
    assert rho_legs["bond"].value < -3.0 * rho_legs["bond"].stderr, rho_legs["bond"]
    # the equity legs (digitals and the short put) gain with the forward: the note less its bond
    # leg as one per-path combination (exact standard error, every pricing already cached)
    up = state.with_rate_shift(dr=h, label="rho")
    bond = note.bond_leg()
    equity = engine.paired(
        "rho[equity legs]",
        [
            (note, up, "recalibrate", c),
            (note, state, "recalibrate", -c),
            (bond, up, "recalibrate", -c),
            (bond, state, "recalibrate", c),
        ],
        unit="per bp",
        size=h,
        scheme="forward",
    )
    assert equity.value == pytest.approx(rho.value - rho_legs["bond"].value, abs=1e-9)
    assert equity.value > 3.0 * equity.stderr, equity
    # the note's own rho: the owner's "rho > 0" does not hold for the funded par note — the
    # bond leg dominates (measured −0.93e-4 +/- 0.04e-4 per bp); pinned as a regression guard
    assert rho.value < -3.0 * rho.stderr, rho
    print(
        f"rho of the equity legs (note less bond leg): {equity} | note rho "
        f"{rho.value / rho.stderr:.1f} stderr negative"
    )
