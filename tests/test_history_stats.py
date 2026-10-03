"""``volsto/studies/history_stats.py``: the structure payoffs, the touch rule, the window
statistics and the filtered historical simulation on synthetic series (no data files)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volsto.studies.history_stats import (
    barrier_path_statistics,
    dispersion_statistics,
    ewma_vol,
    fhs_barrier_table,
    filtered_historical_paths,
    frequency_table,
    path_windows,
    regime_bins,
    structure_payoffs,
    touched_strict,
)


def _series(n: int = 2000, seed: int = 0, vol: float = 0.2) -> pd.Series:
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, vol / np.sqrt(252), n)
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(100.0 * np.exp(np.cumsum(r)), index=idx)


def test_structure_payoffs_and_regret() -> None:
    term = np.array([0.95, 1.02, 1.08, 1.12])
    touched = np.array([False, True, True, True])
    p = structure_payoffs(term, touched, strike=1.0, barrier=1.10, direction="up")
    np.testing.assert_allclose(p["eko"], [0, 0.02, 0.08, 0.0])
    np.testing.assert_allclose(p["uoc"], [0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(p["spread"], [0, 0.02, 0.08, 0.10])
    np.testing.assert_allclose(p["fly"], [0, 0.02, 0.02, 0.0])
    np.testing.assert_allclose(p["ratio"], [0, 0.02, 0.02, -0.02])
    np.testing.assert_allclose(p["regret"], [0, 1, 1, 0])
    q = structure_payoffs(1.0 / term, touched, strike=1.0, barrier=1 / 1.10, direction="down")
    assert q["regret"].tolist() == p["regret"].tolist()
    with pytest.raises(ValueError):
        structure_payoffs(term, touched, strike=1.0, barrier=1.10, direction="up", mid=1.2)


def test_windows_touch_and_tables() -> None:
    s = _series()
    _dates, ratios = path_windows(s, 63)
    assert ratios.shape == (len(s) - 63, 64) and np.all(ratios[:, 0] == 1.0)
    t = touched_strict(ratios, 1.10, "up")
    assert t.dtype == bool and 0 < t.mean() < 1
    st = barrier_path_statistics(s, 63, barrier=1.10, direction="up")
    assert set(st.columns) >= {"terminal", "extreme", "touched", "regret", "uoc", "fly"}
    assert np.all(st["uoc"] <= st["eko"] + 1e-12) and np.all(st["eko"] <= st["spread"] + 1e-12)
    ft = frequency_table(st, 63)
    assert ft.loc[0, "n_eff"] == pytest.approx(len(st) / 63)
    reg = regime_bins(pd.Series(np.arange(len(st)), index=st.index), [0, 1000, 1e9], ["a", "b"])
    gt = frequency_table(st, 63, groups=reg)
    assert set(gt["group"]) == {"a", "b"}


def test_filtered_historical_simulation_hits_the_target_vol() -> None:
    s = _series(vol=0.3)
    r = np.log(s).diff().dropna()
    v = ewma_vol(r)
    assert abs(float(v.iloc[-500:].mean()) - 0.3) < 0.05
    paths = filtered_historical_paths(r, 63, 20_000, vol_target=0.15, seed=1)
    assert paths.shape == (20_000, 64) and np.all(paths[:, 0] == 1.0)
    realised = np.std(np.log(paths[:, -1]), ddof=1) / np.sqrt(63 / 252)
    assert abs(realised - 0.15) < 0.01
    assert abs(float(np.mean(np.log(paths[:, -1])))) < 0.003  # demeaned
    drift = filtered_historical_paths(r, 63, 20_000, vol_target=0.15, seed=1, demean=False)
    assert drift.shape == paths.shape
    tab = fhs_barrier_table(paths, barrier=1.10, direction="up")
    assert 0 < tab["touched"][0] < 1 and tab["uoc"][0] <= tab["eko"][0]
    raw = filtered_historical_paths(r, 63, 2000, vol_target=0.15, seed=1, scale=False)
    assert abs(np.std(np.log(raw[:, -1]), ddof=1) / np.sqrt(63 / 252) - 0.3) < 0.05


def test_dispersion_statistics() -> None:
    rng = np.random.default_rng(2)
    n, m = 800, 4
    z = rng.normal(size=(n, m))
    common = rng.normal(size=(n, 1))
    r = 0.01 * (0.7 * common + 0.7 * z)
    idx = pd.bdate_range("2010-01-04", periods=n)
    closes = pd.DataFrame(100.0 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=list("WXYZ"))
    st = dispersion_statistics(closes, 63)
    assert len(st) == n - 63
    assert np.all(st["dispersion"] >= st["straddle_package"] - 1e-12)
    assert 0.2 < st["corr_realised"].mean() < 0.8
    assert np.all(st["vol_basket"] <= st["vol_mean"] + 1e-9)
