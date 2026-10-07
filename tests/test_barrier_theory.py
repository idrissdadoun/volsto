"""Addendum 1 of the barrier study (:mod:`volsto.studies.barrier_theory`): the carry control,
the weights and the annuity, the bucket statistics and their identities, the directional SSR,
the frozen forecast of realised vol, the local-vol restart."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from volsto.market.curves import ForwardCurve
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import ImpliedSurface
from volsto.studies import barrier_history as bh
from volsto.studies import barrier_theory as bt


class FlatSurface(ImpliedSurface):
    def __init__(self, vol: float, spot: float, r: float, q: float, skew: float = 0.0) -> None:
        fc = ForwardCurve.flat(spot, r, q)
        super().__init__(fc, fc.rate_curve, 10.0)
        self.vol, self.skew = vol, skew

    def total_variance(self, k: Any, T: Any) -> Any:
        v = self.vol + self.skew * np.asarray(k, dtype=np.float64)
        return v * v * np.asarray(T, dtype=np.float64)


@pytest.mark.parametrize("side,B", [(1, 110.0), (-1, 90.0)])
def test_carry_control(side: int, B: float) -> None:
    """Addendum §4.1b: with r = q the continuous knock-out is C8 (the control is zero to
    1e-10); with carry the Reiner–Rubinstein values are the reference ones."""
    K, T, sig = 100.0, 0.5, 0.15
    for r in (0.0, 0.03):
        assert abs(bt.pi_flat(side, K, K, B, T, sig, r, r)) < 1e-10 * K
    ref = {1: 0.5859, -1: 0.6896}[side]
    assert bt.ko_rr(side, K, K, B, T, sig, 0.045, 0.015) == pytest.approx(ref, abs=5e-5)
    assert abs(bt.pi_flat(side, K, K, B, T, sig, 0.045, 0.015)) > 1e-3  # the carry term exists
    # C8 with carry reduces to the zero-carry closed form of the study
    legs = bh.structure_legs(side, K, B, K)["C8"]
    assert bt.c8_bs(side, K, K, B, T, sig, 0.0, 0.0) == pytest.approx(
        bh.bs_legs_value(legs, side, K, sig, T), abs=1e-12
    )
    r, q = bt.carry_rates(np.exp(-0.045 * T), K * np.exp(0.03 * T), K, T)
    assert (r, q) == (pytest.approx(0.045), pytest.approx(0.015))


def test_touch_weights_and_ratios() -> None:
    w = bt.touch_weights(100.0, 100.0, 110.0, 0.5, 0.2)
    assert w["c"] == pytest.approx(np.log(1.1) / (0.2 * np.sqrt(0.5)))
    assert w["pi"].sum() == pytest.approx(1.0) and (w["pi"] > 0).all()
    assert np.allclose(w["theta"], 0.5 * np.array([5 / 6, 1 / 2, 1 / 6]))
    k = np.log(1.1) / (0.2 * np.sqrt(w["theta"]))
    assert np.allclose(w["kap"], 1 - np.exp(-0.5 * k * k)) and (np.diff(w["kap"]) > 0).all()
    assert w["kapbar"] == pytest.approx(float(np.sum(w["pi"] * w["kap"])))
    assert w["om"].sum() == pytest.approx(1.0)
    assert np.allclose(bt.kap(100.0, 110.0, 0.2, w["theta"]), w["kap"])
    # a later touch is likelier for a far barrier, an early one for a near barrier
    far = bt.touch_weights(100.0, 100.0, 130.0, 0.5, 0.2)["pi"]
    near = bt.touch_weights(100.0, 100.0, 101.0, 0.5, 0.2)["pi"]
    assert far[2] > far[0] and near[0] > near[2]
    assert np.isnan(bt.guarded_ratio(1.0, 1e-5)) and bt.guarded_ratio(1.0, 4e-5) == pytest.approx(
        25_000.0
    )
    assert np.isnan(bt.guarded_ratio(np.array([1.0, 2.0]), np.array([0.0, 0.5]))[0])
    surf = FlatSurface(0.2, 100.0, 0.02, 0.0, skew=-0.4)
    n, clamped = bt.normalised_skew(surf, 0.25)
    assert n == pytest.approx(0.4 * 0.5) and not clamped
    n, clamped = bt.normalised_skew(surf, 1.0 / 365.0)
    assert n == pytest.approx(0.4 * np.sqrt(7.0 / 365.0)) and clamped


def _paths(
    n: int = 4000, days: int = 30, vol: float = 0.2, seed: int = 3
) -> tuple[bh.DayPaths, np.ndarray]:
    rng = np.random.default_rng(seed)
    times = np.arange(1, days + 1) / 365.0
    z = rng.standard_normal((n, days))
    x = np.log(100.0) + np.cumsum(vol * np.sqrt(1 / 365.0) * z - 0.5 * vol**2 / 365.0, axis=1)
    cont_max = np.maximum.accumulate(x + 0.002, axis=1)
    cont_min = np.minimum.accumulate(x - 0.002, axis=1)
    paths = bh.DayPaths(
        times=times,
        close=np.exp(x),
        run_max_close=np.maximum.accumulate(x, axis=1),
        run_min_close=np.minimum.accumulate(x, axis=1),
        run_max_cont=cont_max,
        run_min_cont=cont_min,
        antithetic=False,
    )
    return paths, times


def test_bucket_identities() -> None:
    """Addendum §11.1 and §11.2 on synthetic paths: Σ_j p_j·vB6_j (daily rule) is the mean
    tight-limit payoff minus the mean daily knock-out payoff; Σ_j p_j is the daily knock
    probability; Σ_j p_j·d_j is the share of paths finishing beyond the barrier."""
    paths, times = _paths()
    days = [f"d{i}" for i in range(times.size)]
    rows = []
    for side, B in ((1, 103.0), (-1, 97.0), (1, 106.0)):
        legs = bh.structure_legs(side, 100.0, B, 100.0)
        rows.append(
            {"expiry": days[-1], "side": side, "K": 100.0, "B": B, "legs": bh.legs_to_json(legs)}
        )
    cells = pd.DataFrame(rows)
    half = paths.close.shape[0] // 2
    acc = bt.Buckets(cells, days, times)
    for sl in (slice(0, half), slice(half, None)):  # two chunks add up
        acc(
            bh.DayPaths(
                times,
                paths.close[sl],
                paths.run_max_close[sl],
                paths.run_min_close[sl],
                paths.run_max_cont[sl],
                paths.run_min_cont[sl],
                False,
            )
        )
    col = acc.columns("m")
    for i, r in cells.iterrows():
        side, B = int(r["side"]), float(r["B"])
        p = np.array([col[f"m_d_p{j}"][i] for j in (1, 2, 3)])
        v = np.nan_to_num(np.array([col[f"m_d_vB6{j}"][i] for j in (1, 2, 3)]))
        d = np.nan_to_num(np.array([col[f"m_d_d{j}"][i] for j in (1, 2, 3)]))
        ko = bh.knock_out_values(paths, times.size - 1, side, np.array([100.0]), np.array([B]))
        assert p.sum() == pytest.approx(float(ko["p1"][0]), abs=1e-12)
        assert float(p @ v) == pytest.approx(col["m_bk_b6"][i] - col["m_bk_a1"][i], abs=1e-10)
        assert col["m_bk_a1"][i] == pytest.approx(float(ko["a1"][0]), abs=1e-10)
        assert float(p @ d) == pytest.approx(col["m_bk_beyond"][i], abs=1e-12)  # beyond ⇒ knocked
        pc = np.array([col[f"m_c_p{j}"][i] for j in (1, 2, 3)])
        assert pc.sum() == pytest.approx(float(ko["p2"][0]), abs=1e-12) and pc.sum() >= p.sum()
        tl = np.array([col[f"m_d_tl{j}"][i] for j in (1, 2, 3)])
        T = times[-1]
        ok = ~np.isnan(tl)
        assert (tl[ok] <= T * np.array([1.0, 2 / 3, 1 / 3])[ok] + 1e-12).all()
        assert (tl[ok] >= T * np.array([2 / 3, 1 / 3, 0.0])[ok] - 1e-12).all()


def test_directional_ssr_and_forecast() -> None:
    rng = np.random.default_rng(1)
    ret = rng.normal(0, 0.01, 400)
    skew = -0.8
    d_sig = np.where(ret > 0, 1.5 * skew * ret, 0.5 * skew * ret)  # vol moves 1.5x / 0.5x the skew
    up, dn = bt.directional_ssr(d_sig, ret, skew)
    assert up == pytest.approx(1.5) and dn == pytest.approx(0.5)
    assert np.isnan(bt.directional_ssr(d_sig[:30], np.abs(ret[:30]), skew)[1])  # no down-day
    # the forecast: fitted before the cut, never after
    dates = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("1990-01-02", "2010-12-31")]
    vol = 0.1 + 0.1 * (np.sin(np.arange(len(dates)) / 200.0) > 0)
    closes = pd.Series(
        100 * np.exp(np.cumsum(rng.normal(0, 1, len(dates)) * vol / np.sqrt(252))), index=dates
    )
    fit = bt.fit_rv_forecast(closes)
    assert set(fit) == set(bt.RV_HORIZONS)
    for h, f in fit.items():
        assert len(f["coef"]) == 4 and 0.0 < f["r2"] < 1.0
        last_i = dates.index(f["last"])
        assert dates[last_i + h] <= "2006-12-29"  # the future window ends inside the sample
    later = closes.copy()
    later.loc["2007-06-01":] *= 3.0
    assert bt.fit_rv_forecast(later)[21]["coef"] == fit[21]["coef"]  # frozen: no look-ahead
    assert bt.rv_forecast([0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]) == bt.RV_FLOOR
    assert bt.rv_forecast([0.04, 0.04, 0.04], [0.0, 0.5, 0.25, 0.25]) == pytest.approx(0.2)


def test_restart_model_keeps_the_local_vol_in_spot() -> None:
    """A local vol that depends on spot and time: the restarted model has the same
    σ_loc(t, S) at calendar time τ + t."""
    surf = FlatSurface(0.2, 100.0, 0.03, 0.01)
    base = LocalVolSurface.from_implied(surf, bh.LV_CONFIG)
    t, k = base.t_grid[:, None], base.k_grid[None, :]
    spot_dep = (0.2 + 0.1 * np.tanh(-k) + 0.05 * t) ** 2
    base = LocalVolSurface(base.t_grid, base.k_grid, spot_dep, base.forward_curve, base.diagnostics)
    tau, spot = 0.2, 93.0
    model = bt.restart_model(base, tau, spot)
    assert model.forward_curve.spot == spot
    for dt_, S in ((0.05, 90.0), (0.3, 105.0), (0.6, 99.0)):
        want = float(base.local_vol(tau + dt_, S))
        got = float(model.local_vol.local_vol(dt_, S))
        assert got == pytest.approx(want, abs=2e-4)
    # the forward of the restarted curve is the base forward re-based on the new spot
    f = float(base.forward_curve.forward(tau + 0.3) / base.forward_curve.forward(tau))
    assert float(model.forward_curve.forward(0.3)) == pytest.approx(spot * f)
