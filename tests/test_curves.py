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
