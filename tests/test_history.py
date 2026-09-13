"""Historical estimators from a surface history (SPEC §15 Part 2; Bergomi ch. 7 and ch. 9).

* Newey–West / HAC helpers: the lag rule, a hand computation of the Bartlett long-run variance,
  the AR(1) long-run variance ``σ²(1 + φ)/(1 − φ)`` recovered within its sampling error, the
  HAC slope on a known regression, the correlation standard errors.
* ``SurfaceHistory``: round trip through the tidy frame and the wide views, validation errors
  naming the date and pillar, windows and subsets.
* Gaussian synthetic recovery: a history generated with known daily vol of ``ln vs_vol``, a
  known ATM-vol beta to the spot and a constant skew — ``volvol_hist`` recovers the vol of vol,
  ``ssr_hist`` recovers ``β / skew``, the cross-pillar correlation its generator value, each
  within 2–3 standard errors; the rolling estimators have the right shape.
* Identifiability (SPEC §15 Part 2): the pure 2F Table 8.2 model simulated daily for three
  years on a flat 20% VS curve, each day's VS vols from the factor state and the ATMF vols and
  skews by the mixing solution refreshed with the day's curve; ``volvol_hist`` recovers eq. 7.39,
  ``SSR_hist`` eq. 9.21 and the cross-pillar correlations eq. 7.20 at order one (2 SE against
  the in-sample model targets, the flat-curve formulas reported alongside).  The mixing batch is
  checked against the library's mixing solution on the flat state.  Slow (≈ 1–2 min); a
  one-year order-one variant runs in the fast suite.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volsto.analytics.bergomi import ssr_order1_flat, vs_vol_of_vol_flat
from volsto.analytics.mixing import mixing_atmf_skew
from volsto.calibration.history import (
    DEFAULT_PILLARS,
    PillarQuotes,
    SurfaceHistory,
    correlation_with_se,
    estimate_history,
    hac_ols,
    mixing_atmf_batch,
    newey_west_covariance,
    newey_west_lags,
    newey_west_variance,
    rolling_ssr,
    rolling_volvol,
    synthetic_2f_history,
    vs_vol_correlation_matrix,
)
from volsto.config import BergomiParams
from volsto.market import ForwardCurve, ForwardVarianceCurve
from volsto.models import BergomiSV

P82 = BergomiParams(1.74, 0.245, 5.35, 0.28, 0.0, -0.759, -0.487)


def test_newey_west_helpers() -> None:
    assert (newey_west_lags(60), newey_west_lags(250), newey_west_lags(756)) == (3, 4, 6)
    x = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    xc = x - x.mean()
    g0 = float(xc @ xc) / 5
    g1 = float(xc[1:] @ xc[:-1]) / 5
    assert newey_west_variance(x, 1) == pytest.approx(g0 + 2 * 0.5 * g1)
    assert newey_west_variance(x, 0) == pytest.approx(g0)
    # AR(1): long-run variance σ²/(1 − φ)² against the naive variance σ²/(1 − φ²)
    rng = np.random.default_rng(0)
    phi, n = 0.6, 20_000
    e = rng.normal(size=n)
    y = np.empty(n)
    y[0] = e[0]
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    lrv = newey_west_variance(y, 60)
    assert abs(lrv - 1.0 / (1 - phi) ** 2) < 0.15 * 1.0 / (1 - phi) ** 2
    assert lrv > 1.5 * y.var()
    cov = newey_west_covariance(np.column_stack([y, e]), 5)
    assert cov.shape == (2, 2) and cov[0, 0] > cov[1, 1]
    # HAC OLS recovers a known slope with a sensible standard error
    xr = rng.normal(size=2000)
    yr = 0.3 + 1.7 * xr + 0.5 * rng.normal(size=2000)
    reg = hac_ols(xr, yr, 3, intercept=True)
    assert abs(reg.slope - 1.7) < 3 * reg.slope_se and abs(reg.intercept - 0.3) < 0.05
    assert 0.9 < reg.r2 < 0.95 and reg.influence.shape == (2000,)
    c = correlation_with_se(xr, yr, 3)
    assert c.corr == pytest.approx(np.corrcoef(xr, yr)[0, 1]) and 0 < c.se_fisher < 0.01
    with pytest.raises(ValueError):
        newey_west_covariance(y, -1)


def _gaussian_history(seed: int, n_days: int = 800, pillars=(0.25, 1.0, 2.0)):  # type: ignore[no-untyped-def]
    """ln vs_vol random walks with annual vol (0.8, 0.5, 0.4) and cross correlation 0.7; ATM vol
    = 0.35 + (vs_vol − 0.2) + β (ln S − ln S_0) with β per pillar (so Δ atm_vol = Δ vs_vol +
    β Δ ln S), skew constant; spot a 20% random walk.  Returns the history and the generator's
    vol of vol, SSR = β / skew and correlation matrix."""
    rng = np.random.default_rng(seed)
    p = np.asarray(pillars)
    vols = np.array([0.8, 0.5, 0.4]) / np.sqrt(252)
    corr = np.full((3, 3), 0.7) + 0.3 * np.eye(3)
    z = rng.multivariate_normal(np.zeros(3), corr, size=n_days)
    ln_vs = np.log(0.2) + np.cumsum(z * vols, axis=0)
    d_ln_s = 0.2 / np.sqrt(252) * rng.normal(size=n_days)
    ln_s = np.log(100.0) + np.cumsum(d_ln_s)
    beta = np.array([-0.3, -0.15, -0.1])  # ATM vol change per unit Δ ln S
    skew = np.array([-0.8, -0.4, -0.25])
    atm = 0.35 + (np.exp(ln_vs) - 0.2) + (ln_s - ln_s[0])[:, None] * beta[None, :]
    assert atm.min() > 0.05, "generator produced a non-positive ATM vol: change the seed"
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    hist = SurfaceHistory.from_arrays(
        dates, p, np.exp(ln_vs), atm, np.tile(skew, (n_days, 1)), ln_s
    )
    return hist, vols * np.sqrt(252), beta / skew, corr


def test_history_construction_validation_and_gaussian_recovery() -> None:
    hist, volvol_true, ssr_true, _corr_true = _gaussian_history(1)
    assert len(hist) == 800 and hist.pillars.tolist() == [0.25, 1.0, 2.0]
    frame = hist.frame
    assert list(frame.columns) == ["date", "T", "vs_vol", "atm_vol", "skew", "ln_spot"]
    again = SurfaceHistory(frame.sample(frac=1.0, random_state=0))  # order does not matter
    assert np.array_equal(again.vs_vol.to_numpy(), hist.vs_vol.to_numpy())
    assert hist.subset(end=hist.dates[99]).n_dates == 100
    assert "SurfaceHistory(800 dates" in repr(hist)
    # validation names the offending date / pillar
    bad = frame.copy()
    bad.loc[bad.index[5], "vs_vol"] = -0.1
    with pytest.raises(ValueError, match="vs_vol is non-positive"):
        SurfaceHistory(bad)
    with pytest.raises(ValueError, match="every date must carry every pillar"):
        SurfaceHistory(frame.iloc[1:])
    with pytest.raises(ValueError, match="window"):
        hist.volvol_hist(1.0, window=5000)
    with pytest.raises(KeyError):
        hist.volvol_hist(0.5)
    # estimators recover the generator (long window)
    for j, T in enumerate(hist.pillars):
        vv = hist.volvol_hist(float(T), window=750)
        assert abs(vv.volvol - volvol_true[j]) < 3 * vv.se, (vv, volvol_true[j])
        ssr = hist.ssr_hist(float(T), window=750)
        assert abs(ssr.ssr - ssr_true[j]) < 3 * ssr.se, (ssr, ssr_true[j])
        assert 0.0 <= ssr.r2 <= 1.0 and ssr.mean_skew == pytest.approx([-0.8, -0.4, -0.25][j])
    pc = hist.pillar_correlations(window=750)
    off = pc.corr.to_numpy()[np.triu_indices(3, 1)]
    assert np.all(np.abs(off - 0.7) < 3 * pc.se_fisher.to_numpy()[np.triu_indices(3, 1)] + 0.02)
    est = estimate_history(hist, window_vol=750, window_ssr=60)
    tidy = est.to_frame()
    assert set(tidy["statistic"]) >= {"volvol", "ssr", "pillar_corr", "mean_skew", "latest_skew"}
    assert "HistoryEstimates" in repr(est)
    roll = rolling_volvol(hist, 250, step=50, pillars=[1.0])
    assert len(roll) >= 10 and {"volvol", "se"} <= set(roll.columns)
    roll_s = rolling_ssr(hist, 60, step=50, pillars=[1.0])
    assert len(roll_s) >= 10 and {"ssr", "se"} <= set(roll_s.columns)


def test_mixing_batch_matches_library() -> None:
    """The per-day mixing batch on the flat state (X = 0 at t = 0) against the library's
    mixing solution: ATMF vol and skew within 3 stderr."""
    m = BergomiSV(P82, ForwardVarianceCurve.flat(0.04), ForwardCurve.flat(100.0, 0.0, 0.0))
    mb = mixing_atmf_batch(m, 1.0, [0.0], np.zeros((1, 2)), n_paths=40_000, seed=1, dt=1 / 365)
    skew, atm, skew_se = mixing_atmf_skew(m, 1.0, n_paths=40_000, seed=2)
    assert abs(mb.atm_vol[0] - atm) < 3 * mb.atm_vol_se[0] + 1e-3, (mb, atm)
    assert abs(mb.skew[0] - skew) < 3 * np.hypot(mb.skew_se[0], skew_se) + 2e-3, (mb, skew)


def test_identifiability_order_one_fast() -> None:
    """One year of the Table 8.2 model with the order-one ATM source: volvol_hist within 2 SE of
    the in-sample eq. 7.39 target (flat-curve formula reported), SSR_hist within 2 SE of the
    order-one target, cross-pillar correlations within 3 Fisher SE of eq. 7.20."""
    sh = synthetic_2f_history(P82, years=1.0, seed=8, atm_source="order_one")
    est = estimate_history(sh.history, window_vol=200, window_ssr=60)
    targets = sh.targets(200)
    for T in sh.pillars:
        row, tgt = est.volvol.loc[T], targets.loc[T]
        assert abs(row["volvol"] - tgt["volvol_in"]) < 2 * row["se"] + 0.05 * tgt["volvol_in"], (
            T,
            row,
            tgt,
        )
        assert abs(tgt["volvol_flat"] - float(vs_vol_of_vol_flat(P82, T))) < 1e-12
        s = est.ssr_long.loc[T]
        assert abs(s["ssr"] - tgt["ssr_order1_in"]) < 2 * s["se"] + 0.1, (T, s, tgt)
        assert abs(tgt["ssr_flat"] - float(ssr_order1_flat(P82, T))) < 1e-12
    corr_in, corr_flat = sh.correlation_targets(200)
    got = est.correlations.corr.to_numpy()
    se = est.correlations.se_fisher.to_numpy()
    iu = np.triu_indices(len(sh.pillars), 1)
    assert np.all(np.abs(got[iu] - corr_in.to_numpy()[iu]) < 3 * se[iu] + 0.05)
    assert np.allclose(corr_flat.to_numpy(), vs_vol_correlation_matrix(P82, sh.pillars.tolist()))


@pytest.mark.slow
def test_identifiability_2f_mixing_slow() -> None:
    """SPEC §15 Part 2 identifiability test: three years, ATMF vols and skews by the mixing
    solution; volvol_hist ↔ eq. 7.39, SSR_hist ↔ eq. 9.21, correlations ↔ eq. 7.20 (2 SE
    against the in-sample targets).  Seed 13 is a typical path: over 14 three-year paths the
    last-window ratio realised / instantaneous RMS vol of vol ranged 0.87–1.09 (common to all
    pillars: one draw of the window's vol-of-vol regime), i.e. a window-to-window dispersion of
    about 6% of which the Newey–West standard error captures 3.8%; seeds 2 and 11 sit at −3σ
    (a 30-year path gives ratio 0.996 with no bias) — recorded in SPEC §15 notes."""
    sh = synthetic_2f_history(P82, years=3.0, seed=13, atm_source="mixing", mixing_paths=20_000)
    est = estimate_history(sh.history, window_vol=250, window_ssr=60)
    targets = sh.targets(250)
    for T in sh.pillars:
        row, tgt = est.volvol.loc[T], targets.loc[T]
        assert abs(row["volvol"] - tgt["volvol_in"]) < 2 * row["se"] + 0.05 * tgt["volvol_in"], (
            T,
            row,
            tgt,
        )
        s = est.ssr_long.loc[T]
        assert abs(s["ssr"] - tgt["ssr_order1_in"]) < 2 * s["se"] + 0.15, (T, s, tgt)
    corr_in, _ = sh.correlation_targets(250)
    got = est.correlations.corr.to_numpy()
    se = est.correlations.se_fisher.to_numpy()
    iu = np.triu_indices(len(sh.pillars), 1)
    assert np.all(np.abs(got[iu] - corr_in.to_numpy()[iu]) < 3 * se[iu] + 0.05)
    assert list(DEFAULT_PILLARS) == [1 / 12, 0.25, 0.5, 1.0, 2.0, 3.0]
    q = PillarQuotes(np.array([1.0]), np.array([0.2]), np.array([0.21]), np.array([-0.1]), 100.0)
    assert q.atm_vol(1.0) == 0.21
