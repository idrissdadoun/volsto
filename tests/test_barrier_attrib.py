"""Addendum 2 of the barrier study (:mod:`volsto.studies.barrier_attrib`): the realised vol of
a life, the vol carry and its identities, turnover, the clustered fits, the skew tilt."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from volsto.market.curves import ForwardCurve
from volsto.market.surface import ImpliedSurface
from volsto.studies import barrier_attrib as ba


class FlatSurface(ImpliedSurface):
    def __init__(self, vol: float, spot: float, r: float, q: float) -> None:
        fc = ForwardCurve.flat(spot, r, q)
        super().__init__(fc, fc.rate_curve, 10.0)
        self.vol = vol

    def total_variance(self, k: Any, T: Any) -> Any:
        return self.vol**2 * np.asarray(T, dtype=np.float64) + 0.0 * np.asarray(k)


def closes(x: float, n: int) -> pd.Series:
    """``n`` returns of alternating ±x from 100, on business days."""
    r = x * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
    days = pd.bdate_range("2020-01-01", periods=n + 1).strftime("%Y-%m-%d")
    return pd.Series(100.0 * np.exp(np.concatenate(([0.0], np.cumsum(r)))), index=days)


def test_realised_vol_of_constant_absolute_returns() -> None:
    for x, n in ((0.01, 21), (0.007, 64), (0.02, 15)):
        c = closes(x, n)
        assert ba.realised_vol(c.to_numpy()) == pytest.approx(x * np.sqrt(252.0), abs=1e-10)
        got = ba.life_realised_vol(c, str(c.index[0]), str(c.index[-1]))
        assert got == pytest.approx(x * np.sqrt(252.0), abs=1e-10)


def test_realised_vol_needs_fifteen_returns_and_a_finished_life() -> None:
    c = closes(0.01, 14)
    assert np.isnan(ba.realised_vol(c.to_numpy()))
    c = closes(0.01, 30)
    assert np.isnan(ba.life_realised_vol(c, str(c.index[0]), "2031-01-01"))
    # a life is the closes from the entry to the expiry, both included
    sub = ba.life_realised_vol(c, str(c.index[3]), str(c.index[23]))
    assert sub == pytest.approx(0.01 * np.sqrt(252.0), abs=1e-10)


def frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    c = closes(0.01, 40)
    entry, expiry = str(c.index[0]), str(c.index[30])
    cells = pd.DataFrame(
        {
            "cell": ["a", "b"],
            "entry": entry,
            "expiry": [expiry, "2031-01-01"],
            "K": [100.0, 100.0],
            "atm": [0.20, 0.20],
            "lv_a1_vega": [-12.0, -12.0],
            "vega_B5_2": [1.5, 1.5],
            "vega_FWD": [0.0, 0.0],
        }
    )
    pos = pd.DataFrame(
        {
            "cell": ["a", "a", "a", "a", "b"],
            "position": ["A1", "B5_2", "FWD", "A1", "A1"],
            "model": ["lv", "surface", "surface", "lsv", "lv"],
            "pnl_h": [0.004, -0.001, 0.0005, 0.003, np.nan],
        }
    )
    return pos, cells, c


def test_attribution_carry_and_pair_identity() -> None:
    pos, cells, c = frames()
    out = ba.attribute(pos, cells, c).set_index(["cell", "position", "model"])
    dsig = 100.0 * (0.01 * np.sqrt(252.0) - 0.20)
    ko, fly = out.loc[("a", "A1", "lv")], out.loc[("a", "B5_2", "surface")]
    assert ko["dsig"] == pytest.approx(dsig, abs=1e-10)
    assert ko["vega"] == pytest.approx(-0.12) and fly["vega"] == pytest.approx(0.015)
    assert ko["carry"] == pytest.approx(-0.12 * dsig) and fly["carry"] == pytest.approx(
        0.015 * dsig
    )
    # the pair: its carry is the difference of the legs', and so is its ex-vol P&L
    pair_carry = (fly["vega"] - ko["vega"]) * dsig
    assert pair_carry == pytest.approx(fly["carry"] - ko["carry"], abs=1e-12)
    pair_hx = (fly["pnl_h"] - ko["pnl_h"]) - pair_carry
    assert pair_hx == pytest.approx(fly["pnl_hx"] - ko["pnl_hx"], abs=1e-12)
    # a position without vega keeps its hedged P&L exactly
    assert out.loc[("a", "FWD", "surface"), "pnl_hx"] == out.loc[("a", "FWD", "surface"), "pnl_h"]
    assert float(ba.vol_carry(0.0, dsig)) == 0.0
    # the LSV row takes the local-vol vega, and says so
    lsv = out.loc[("a", "A1", "lsv")]
    assert lsv["vega"] == ko["vega"] and lsv["vega_source"] == "lv"
    # an unfinished life has no realised vol, hence no ex-vol P&L
    assert np.isnan(out.loc[("b", "A1", "lv"), "rv_life"])


def test_attribution_does_not_touch_the_input() -> None:
    pos, cells, c = frames()
    before = pos.copy()
    ba.attribute(pos, cells, c)
    pd.testing.assert_frame_equal(pos, before)


def test_turnover() -> None:
    assert ba.turnover([]) == 0.0
    assert ba.turnover([0.3, 0.3, 0.3]) == pytest.approx(0.6)
    assert ba.turnover([-0.25] * 10) == pytest.approx(0.5)
    assert ba.turnover([0.2, 0.5, 0.1]) == pytest.approx(0.2 + 0.3 + 0.4 + 0.1)
    w = [1.0, 1.1, 1.2, 0.9]
    got = ba.turnover([0.2, 0.5, 0.1], w)
    assert got == pytest.approx(0.2 * 1.0 + 0.3 * 1.1 + 0.4 * 1.2 + 0.1 * 0.9)
    with pytest.raises(ValueError):
        ba.turnover([0.2, 0.5], [1.0, 1.0])


def test_clustered_ols_recovers_the_coefficients() -> None:
    rng = np.random.default_rng(0)
    n_groups, per = 200, 5
    g = np.repeat(np.arange(n_groups), per)
    x = rng.normal(size=g.size)
    t = (rng.random(g.size) < 0.3).astype(float)
    y = 0.5 + 1.0 * x - 2.0 * t + 0.1 * rng.normal(size=g.size)
    fit = ba.clustered_ols(y, np.column_stack([np.ones(g.size), x, t]), g, block=5)
    assert fit["coef"] == pytest.approx([0.5, 1.0, -2.0], abs=0.03)
    assert fit["r2"] > 0.98 and fit["n"] == g.size and fit["n_groups"] == n_groups
    assert np.all(fit["se"] > 0) and np.all(fit["se"] < 0.05)
    # a regressor without variation is left out, the others are unchanged
    fit0 = ba.clustered_ols(y, np.column_stack([np.ones(g.size), x, np.zeros(g.size)]), g, block=5)
    assert np.isnan(fit0["coef"][2]) and np.isfinite(fit0["coef"][:2]).all()


def test_ratio_interval() -> None:
    rng = np.random.default_rng(1)
    den = 0.01 + 0.001 * rng.normal(size=300)
    num = 0.6 * den + 0.0005 * rng.normal(size=300)
    ratio, lo, hi, m_num, m_den = ba.ratio_interval(num, den, 5, floor=0.2e-4)
    assert lo < ratio < hi and lo < 0.6 < hi and hi - lo < 0.1
    assert ratio == pytest.approx(m_num / m_den)
    assert np.isnan(ba.ratio_interval(num[:10], den[:10], 5, floor=0.2e-4)[0])
    assert np.isnan(ba.ratio_interval(num, 1e-7 * den, 5, floor=0.2e-4)[0])


def test_tilt_on_a_flat_surface() -> None:
    base = FlatSurface(0.2, 100.0, 0.03, 0.01)
    T = 0.25
    F = float(base.forward(T))
    for B in (108.0, 93.0):
        tilted = ba.TiltedSurface(base, float(np.log(B / F)))
        assert float(tilted.implied_vol_k(0.0, T)) == pytest.approx(0.2, abs=1e-12)
        move = float(tilted.implied_vol(B, T)) - 0.2
        assert move == pytest.approx(0.01 if B > F else -0.01, abs=1e-12)
        mirror = F * F / B
        assert float(tilted.implied_vol(mirror, T)) - 0.2 == pytest.approx(-move, abs=1e-12)
    with pytest.raises(ValueError):
        ba.TiltedSurface(base, 0.0)
