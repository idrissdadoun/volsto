from __future__ import annotations

import datetime as dt

import numpy as np
import pytest

from volsto.market import Calendar, DiscountCurve, ForwardCurve


def test_flat_curve() -> None:
    c = DiscountCurve.flat(0.03)
    T = np.array([0.0, 0.5, 1.0, 7.0])
    np.testing.assert_allclose(c.df(T), np.exp(-0.03 * T), rtol=1e-14)
    np.testing.assert_allclose(c.zero_rate(T), 0.03)
    np.testing.assert_allclose(c.instantaneous_forward(T), 0.03)


def test_piecewise_flat_forwards() -> None:
    c = DiscountCurve([1.0, 2.0], [0.02, 0.03])
    # pillars reproduced
    np.testing.assert_allclose(c.df(1.0), np.exp(-0.02), rtol=1e-14)
    np.testing.assert_allclose(c.df(2.0), np.exp(-0.06), rtol=1e-14)
    # forward between 1y and 2y = 4%, flat extrapolation of that forward
    assert c.forward_rate(1.0, 2.0) == pytest.approx(0.04)
    assert c.forward_rate(1.5, 1.75) == pytest.approx(0.04)
    assert c.forward_rate(2.0, 5.0) == pytest.approx(0.04)
    assert c.forward_rate(0.0, 0.5) == pytest.approx(0.02)
    # log-linear interpolation
    assert np.log(c.df(1.5)) == pytest.approx(-0.02 - 0.04 * 0.5)
    np.testing.assert_allclose(c.integrated(0.5, 1.5), 0.02 * 0.5 + 0.04 * 0.5)


def test_from_instantaneous_round_trip() -> None:
    c = DiscountCurve.from_instantaneous([0.5, 1.0, 3.0], [0.01, 0.02, 0.03])
    np.testing.assert_allclose(
        c.instantaneous_forward([0.1, 0.7, 2.0, 9.0]), [0.01, 0.02, 0.03, 0.03]
    )


def test_forward_curve() -> None:
    fc = ForwardCurve.flat(100.0, 0.02, 0.01)
    T = np.array([0.0, 1.0, 2.5])
    np.testing.assert_allclose(fc.forward(T), 100.0 * np.exp(0.01 * T), rtol=1e-14)
    assert fc.drift(1.0, 2.0) == pytest.approx(0.01)
    assert fc.with_spot(90.0).forward(1.0) == pytest.approx(90.0 * np.exp(0.01))


def test_forward_curve_from_forwards() -> None:
    """``ForwardCurve.from_forwards`` (SPEC §8.7): every listed forward reproduced to 1e-12, the
    carry piecewise flat between the expiries and flat beyond the last, one expiry a flat carry
    (check C8's curve)."""
    rates = DiscountCurve([0.25, 1.0, 3.0], [0.045, 0.04, 0.035])
    times = np.array([18, 46, 81, 109, 172, 263, 445, 809]) / 365.0
    carry = np.array([0.031, 0.012, 0.027, 0.018, 0.022, 0.019, 0.024, 0.021])  # uneven dividends
    for spot in (1.0, 187.35):
        forwards = spot * np.exp(carry * times)
        fc = ForwardCurve.from_forwards(spot, times, forwards, rates)
        assert fc.spot == spot and fc.rate_curve is rates
        np.testing.assert_allclose(fc.forward(times), forwards, rtol=1e-12, atol=0.0)
        # piecewise-flat carry: ln F is linear in t inside every interval between expiries (the
        # rate curve's own knots at 0.25, 1 and 3 bend it inside the intervals that hold one)
        for a, b in zip(np.concatenate(([0.0], times[:-1])), times, strict=True):
            if any(a < knot < b for knot in (0.25, 1.0, 3.0)):
                continue
            mid = float(fc.log_forward(0.5 * (a + b)))
            assert mid == pytest.approx(
                0.5 * (float(fc.log_forward(a)) + float(fc.log_forward(b))), abs=1e-13
            )
        # flat beyond the last expiry: the dividend forward of the last interval continues
        q_last = float(fc.dividend_curve.forward_rate(times[-2], times[-1]))
        assert float(fc.dividend_curve.forward_rate(times[-1], times[-1] + 2.0)) == pytest.approx(
            q_last, abs=1e-13
        )
    # one expiry: a flat carry, the forward ratio of check C8 (rate r, q = r − ln f / T)
    T, f, r = 0.25, 1.0061, 0.043
    one = ForwardCurve.from_forwards(1.0, [T], [f], DiscountCurve.flat(r))
    c8 = ForwardCurve.flat(1.0, r, r - float(np.log(f)) / T)
    t = np.array([0.01, 0.1, T, 0.4, 1.3])
    np.testing.assert_allclose(one.log_forward(t), c8.log_forward(t), rtol=0, atol=1e-15)
    bad_inputs: tuple[tuple[list[float], list[float]], ...] = (
        ([0.5, 0.25], [1.0, 1.0]),
        ([0.25], [-1.0]),
        ([0.25, 0.5], [1.0]),
        ([], []),
    )
    for bad_t, bad_f in bad_inputs:
        with pytest.raises(ValueError):
            ForwardCurve.from_forwards(1.0, bad_t, bad_f, rates)
    with pytest.raises(ValueError):
        ForwardCurve.from_forwards(0.0, [0.25], [1.0], rates)


def test_invalid_curves() -> None:
    with pytest.raises(ValueError):
        DiscountCurve([1.0, 1.0], [0.0, 0.0])
    with pytest.raises(ValueError):
        DiscountCurve([0.0], [0.0])
    with pytest.raises(ValueError):
        ForwardCurve(0.0, DiscountCurve.flat(0.0), DiscountCurve.flat(0.0))


def test_calendar_act365() -> None:
    cal = Calendar(dt.date(2026, 9, 13))
    assert cal.t(dt.date(2027, 9, 13)) == pytest.approx(365 / 365)
    assert cal.year_fraction(dt.date(2026, 1, 1), dt.date(2026, 1, 2)) == pytest.approx(1 / 365)
    assert cal.date(1.0) == dt.date(2027, 9, 13)
