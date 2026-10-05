"""Dispersion study (question 2): the identities of the notes' §2, Euler's relation for the
Palladium delta, the strip on a flat smile (C2), the copula on its marginals (C3, C4, C6), the
exact decomposition (C11), the sensitivities on flat equal-vol baskets (C12), the common-move
delta (C13), the FHS on an iid history (C14), no look-ahead in the indicators (C10), the
frozen basket through corporate actions, and the statistics for overlapping windows."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.special import ndtr

from volsto.studies import disp_copula as dc
from volsto.studies import disp_fhs as fhs
from volsto.studies import disp_indicators as di
from volsto.studies import disp_payoff as dp
from volsto.studies import disp_smile as ds
from volsto.studies import disp_stats as st
from volsto.studies import disp_universe as du


def random_baskets(seed: int, n_paths: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(2, 40))
    w = rng.random(n) + 0.05
    w /= w.sum()
    common = rng.standard_t(4, size=(n_paths, 1))
    X = np.exp(0.15 * common + 0.2 * rng.standard_t(4, size=(n_paths, n)))
    return X, w


def test_sandwich_gap_and_jensen_on_random_baskets() -> None:
    for seed in range(40):
        X, w = random_baskets(seed)
        t = dp.terminal(X, w)
        assert np.all(t["D"] >= t["SD"] - 1e-12)
        assert np.all(t["D"] <= t["SD"] + 2 * t["absRb"] + 1e-12)
        assert np.max(np.abs(dp.gap_formula(X, w) - t["G"])) < 1e-12
        assert np.all(t["D"] <= np.sqrt(t["V"]) + 1e-12)
        # floating strike (notes eq. 2.5): twice the average one-sided option
        eps = np.where(t["Rb"] >= 0, 1.0, -1.0)[:, None]
        R = X - 1.0
        assert np.allclose(2 * np.maximum(eps * (t["Rb"][:, None] - R), 0) @ w, t["D"], atol=1e-12)
        assert np.allclose(2 * np.maximum(-eps * R, 0) @ w, t["SD"], atol=1e-12)
        # squared dispersion is a vanilla payoff
        assert np.allclose((R * R) @ w - t["Rb"] ** 2, t["V"], atol=1e-12)


def test_four_name_examples_of_the_notes() -> None:
    w = np.full(4, 0.25)
    for R, want in (
        ([0.12, 0.04, -0.04, -0.12], (0.08, 0.08, 0.0)),
        ([0.30, 0.20, 0.10, 0.0], (0.0, 0.10, 0.10)),
        ([-0.05, -0.15, -0.25, -0.35], (0.0, 0.10, 0.10)),
        ([0.15] * 4, (0.0, 0.0, 0.0)),
    ):
        t = dp.terminal(1.0 + np.array(R), w)
        assert (t["SD"], t["D"], t["G"]) == pytest.approx(want, abs=1e-12)


def test_palladium_delta_satisfies_euler_and_is_the_gradient() -> None:
    rng = np.random.default_rng(3)
    n = 12
    w = rng.random(n)
    w /= w.sum()
    X = np.exp(0.2 * rng.standard_normal(n))
    sig_rel = 0.15 + 0.2 * rng.random(n)
    tau = 0.2
    delta = dp.palladium_delta(X, w, sig_rel, tau)
    value = dp.palladium_value(X, w, sig_rel, tau)
    assert float(X @ delta) == pytest.approx(value, rel=1e-12)
    for j in (0, 5, 11):
        up, dn = X.copy(), X.copy()
        up[j] *= 1 + 1e-6
        dn[j] *= 1 - 1e-6
        fd = (dp.palladium_value(up, w, sig_rel, tau) - dp.palladium_value(dn, w, sig_rel, tau)) / (
            2e-6 * X[j]
        )
        assert delta[j] == pytest.approx(fd, abs=1e-7)
    # a path: one row per day
    path = np.vstack([X, X * 1.01])
    assert np.allclose(dp.palladium_delta(path, w, sig_rel, np.array([tau, tau]))[0], delta)


def test_strip_on_a_flat_smile() -> None:
    for vol, T in ((0.3, 0.5), (0.2, 0.25), (0.45, 1.0)):
        m = ds.build_marginal(ds.flat_smile(vol, T))
        assert pytest.approx(np.exp(vol * vol * T) - 1.0, abs=1e-6) == m.M
        assert m.M > 0 and m.tail_share < 1e-3
        assert m.var_vs == pytest.approx(vol * vol, abs=1e-6)
        assert m.straddle == pytest.approx(2 * (2 * ndtr(vol * np.sqrt(T) / 2) - 1), abs=1e-12)
    # with carry: E[(X − 1)²] = Var(X) + (F/S − 1)²
    m = ds.build_marginal(ds.flat_smile(0.3, 0.5, f=1.02))
    assert pytest.approx(1.02**2 * (np.exp(0.045) - 1.0) + 0.02**2, abs=1e-6) == m.M


def test_convex_envelope() -> None:
    x = np.linspace(0, 4, 9)
    y = (x - 2.0) ** 2
    assert np.allclose(ds.convex_envelope(x, y), y)
    y2 = y.copy()
    y2[4] += 1.0  # a bump above the curve is cut off
    assert ds.convex_envelope(x, y2)[4] == pytest.approx(0.5 * (y[3] + y[5]))


def skewed_smile(T: float, atm: float, skew: float, f: float = 1.0) -> ds.TenorSmile:
    k = np.linspace(-1.0, 1.0, 41)
    vol = np.maximum(atm + skew * k + 0.15 * k * k, 0.05)
    e = ds.ExpirySmile("x", T, f, 0.0, k, vol, vol, 0)
    return ds.TenorSmile(1.0, T, f, 0.0, e, e)


def test_copula_reprices_marginals_and_the_basket_straddle() -> None:
    T, n = 0.25, 12
    rng = np.random.default_rng(5)
    smiles = [
        skewed_smile(T, 0.2 + 0.2 * rng.random(), -0.3 * rng.random(), 1.0 + 0.01 * rng.random())
        for _ in range(n)
    ]
    margs = [ds.build_marginal(s) for s in smiles]
    w = np.full(n, 1.0 / n)
    draws = dc.sobol_draws(n, seed=7)
    out = dc.price_basket(margs, None, w, draws, 0.4)
    want = np.array([m.straddle for m in margs])
    assert np.max(np.abs(out["straddle_cop"] / want - 1.0)) < 2e-3  # C3
    assert out["E_B"] == pytest.approx(float(np.mean([m.f for m in margs])), abs=2e-4)
    rho, flag = dc.calibrate_rho(dc.tables(margs), draws, w, out["Str_B"])
    assert flag == "" and rho == pytest.approx(0.4, abs=1e-6)
    assert abs(dc.basket_straddle(dc.tables(margs), draws, w, rho) - out["Str_B"]) < 1e-5  # C4
    # a target no correlation reaches is clipped and flagged
    assert dc.calibrate_rho(dc.tables(margs), draws, w, 10.0) == (0.99, "clipped high")
    assert dc.calibrate_rho(dc.tables(margs), draws, w, 0.0) == (0.0, "clipped low")


@pytest.mark.parametrize(("n", "rho"), [(20, 0.115), (20, 0.5), (30, 0.3), (30, 0.7)])
def test_flat_equal_vol_basket_against_closed_forms(n: int, rho: float) -> None:
    T, vol = 0.25, 0.3
    m = ds.build_marginal(ds.flat_smile(vol, T))
    mb = ds.build_marginal(ds.flat_smile(vol, T), 0.01)
    w = np.full(n, 1.0 / n)
    out = dc.price_basket([m] * n, [mb] * n, w, dc.sobol_draws(n, seed=1, n_points=2**15), rho)
    _, _, rel = dc.gaussian_inputs(np.full(n, vol), w, rho)
    exact = float(np.sum(w * 2 * (2 * ndtr(rel * np.sqrt(T) / 2) - 1)))
    assert out["P_D"] == pytest.approx(exact, rel=0.01)  # C6
    assert out["P_D_se"] < 0.005 * out["P_D"]
    lam = np.sqrt(rho + (1 - rho) / n) / (np.sqrt(1 - rho) * np.sqrt(1 - 1 / n))
    assert out["lambda_rho"] == pytest.approx(lam, rel=0.03)  # C12
    assert out["h_v"] == pytest.approx(np.sqrt((1 - rho) * (1 - 1 / n)), rel=0.03)
    priced = out["calls"] >= dc.MIN_CALL
    assert np.allclose(out["delta_c"][priced], out["delta_c_fd"][priced], rtol=0.01)  # C13
    # cash strikes are multiples of the forward's price, and the 0.5 call is deep in the money
    assert out["strikes"] == pytest.approx(np.array(dc.CALL_MULTIPLES) * out["P_D"])
    assert out["calls"][0] == pytest.approx(0.5 * out["P_D"], rel=0.02)


def test_decomposition_of_forward_minus_package() -> None:
    rng = np.random.default_rng(9)
    for _ in range(50):
        X, w = random_baskets(int(rng.integers(1e6)), n_paths=1)
        t = dp.terminal(X[0], w)
        lam = float(rng.uniform(0.3, 2.0))
        leg = {
            "D": float(t["D"]), "absR": float(t["absR"]), "absRb": float(t["absRb"]),
            "P_D": 0.07, "SS": 0.09, "Str_B": 0.05, "lam_theta": lam, "lam_rho": 0.8, "h_v": 0.7,
        }  # fmt: skip
        s = dp.structures(leg)
        p_g = leg["P_D"] - (leg["SS"] - leg["Str_B"])
        want = (float(t["G"]) - p_g) + (lam - 1.0) * (leg["absRb"] - leg["Str_B"])
        assert s["PF_U"] - s["PKG_theta_U"] == pytest.approx(want, abs=1e-12)  # C11
        assert s["GAP_U"] == pytest.approx(float(t["G"]) - p_g, abs=1e-12)
        assert s["REV_U"] == pytest.approx(-s["PKG_v_U"])


def test_hedge_legs_on_a_flat_path_and_a_common_move() -> None:
    n, N, T = 8, 63, 0.25
    w = np.full(n, 1.0 / n)
    vols = np.full(n, 0.3)
    _, sb, rel = dc.gaussian_inputs(vols, w, 0.5)
    flat = np.ones((N + 1, n))
    h = dp.hedge_legs(flat, w, T, vols, sb, rel)
    assert h["hedge_SS"] == 0.0 and h["hedge_BS"] == 0.0 and h["hedge_PF"] == 0.0
    # every name up 0.1 % a day: by Euler's relation the Palladium's hedge loses its own value
    # per unit of basket times the basket's move, day after day (the scaling delta, no more)
    path = np.cumprod(np.vstack([np.ones(n), np.full((N, n), 1.001)]), axis=0)
    h = dp.hedge_legs(path, w, T, vols, sb, rel)
    assert h["hedge_SS"] < 0 and h["hedge_BS"] < 0
    basket = path @ w
    tau = T * (N - np.arange(N)) / N
    want = -sum(
        dp.palladium_value(path[t], w, rel, float(tau[t])) / basket[t] * (basket[t + 1] - basket[t])
        for t in range(N)
    )
    assert h["hedge_PF"] == pytest.approx(want, rel=1e-10)
    assert abs(h["hedge_PF"]) < abs(h["hedge_SS"]) < abs(h["hedge_BS"])
    # the notional traded counts the entry and exit trades
    assert h["traded_BS"] >= 2 * 0.0 and h["traded_PF"] > 0


def test_daily_stats_of_a_one_factor_path() -> None:
    rng = np.random.default_rng(4)
    n, N = 20, 2520
    w = np.full(n, 1.0 / n)
    rho, vol = 0.4, 0.25
    r = (
        vol
        / np.sqrt(252)
        * (
            np.sqrt(rho) * rng.standard_normal((N, 1))
            + np.sqrt(1 - rho) * rng.standard_normal((N, n))
        )
    )
    path = np.cumprod(np.vstack([np.ones(n), 1 + r]), axis=0)
    s = dp.daily_stats(path, w)
    assert s["rho_pairwise"] == pytest.approx(rho, abs=0.05)
    assert s["rho_real"] == pytest.approx(rho, abs=0.08)
    assert s["vol_names_real"] == pytest.approx(vol, rel=0.05)


def test_fhs_on_an_iid_gaussian_history() -> None:
    rng = np.random.default_rng(12)
    n, days, h, rho = 10, 20_000, 63, 0.35
    vols = np.linspace(0.18, 0.40, n)
    w = np.full(n, 1.0 / n)
    r = (
        (
            np.sqrt(rho) * rng.standard_normal((days, 1))
            + np.sqrt(1 - rho) * rng.standard_normal((days, n))
        )
        * vols
        / np.sqrt(252)
    )
    X = fhs.simulate(r, vols, h, seed=1)
    s = fhs.summary(X, w)
    _, _, rel = dc.gaussian_inputs(vols, w, rho)
    T = h / 252
    exact_d = float(np.sum(w * 2 * (2 * ndtr(rel * np.sqrt(T) / 2) - 1)))
    cov = rho * np.outer(vols, vols) + (1 - rho) * np.diag(vols**2)
    exact_v = float(np.sum(w * (1 + vols**2 / 252) ** h) - w @ ((1 + cov / 252) ** h) @ w)
    assert s["E_D"] == pytest.approx(exact_d, rel=0.03)  # C14
    assert s["E_V"] == pytest.approx(exact_v, rel=0.03)
    assert s["rho_fhs"] == pytest.approx(rho, abs=0.05)
    # reproducible, and the bootstrap moves in blocks
    assert np.array_equal(fhs.simulate(r, vols, h, seed=1), X)
    idx = fhs.bootstrap_days(1000, 63, 200, 10, seed=3)
    assert np.mean(np.diff(idx, axis=1) % 1000 == 1) == pytest.approx(0.9, abs=0.02)


def test_indicators_use_nothing_from_the_entry_date_on() -> None:
    rng = np.random.default_rng(8)
    days = pd.bdate_range("2010-01-01", periods=1600).strftime("%Y-%m-%d")
    names = [f"N{i}" for i in range(6)]
    base = pd.DataFrame(0.012 * rng.standard_normal((1600, 6)), index=days, columns=names)
    entry = days[1400]
    w = np.full(6, 1 / 6)
    a = di.indicators(di.trailing(base, entry, names, 1260), w, 63, 0.25)
    changed = base.copy()
    changed.loc[changed.index >= entry] = 0.5  # the entry date and everything after it
    b = di.indicators(di.trailing(changed, entry, names, 1260), w, 63, 0.25)
    assert a.keys() == b.keys() and all(a[k] == b[k] for k in a)
    # the day before the entry does matter
    changed.loc[days[1399]] = 0.5
    c = di.indicators(di.trailing(changed, entry, names, 1260), w, 63, 0.25)
    assert c["sig_fc_bar"] != a["sig_fc_bar"]
    assert di.trailing(base, entry, names, 10).shape == (10, 6)
    assert np.array_equal(di.trailing(base, entry, names, 10)[-1], base.loc[days[1399]].to_numpy())


def test_variance_ratio_and_forecast_blend() -> None:
    rng = np.random.default_rng(2)
    z = 0.01 * rng.standard_normal((756, 25))
    w = np.full(25, 0.04)
    cs, raw = di.variance_ratio(z, w, 63)
    assert cs == pytest.approx(1.0, abs=0.15) and raw == pytest.approx(1.0, abs=0.25)
    trend = z + 0.1 * np.vstack([np.zeros((1, 25)), z[:-1]])  # positive autocorrelation
    assert di.variance_ratio(trend, w, 21)[0] > di.variance_ratio(z, w, 21)[0]
    r = np.full((252, 1), 0.01)
    assert float(di.forecast_vol(r)[0]) == pytest.approx(0.01 * np.sqrt(252))
    g = di.gaussian_forecast(np.full(3, 0.3), 0.2, np.full(3, 0.22), np.full(3, 1 / 3), 0.25, 1.2)
    assert g["ED_G1"] == pytest.approx(g["ED_G0"] * np.sqrt(1.2))
    assert g["EV_G1"] == pytest.approx(g["EV_G0"] * 1.2) and g["EabsRb_G1"] == g["EabsRb_G0"]
    assert g["EG_G0"] == pytest.approx(g["ED_G0"] - g["EabsR_G0"] + g["EabsRb_G0"])
    assert di.earnings_share("2024-01-15", "2024-02-15") == 1.0
    assert di.earnings_share("2024-03-01", "2024-03-31") == 0.0


def test_membership_and_the_frozen_basket_through_actions() -> None:
    assert len(du.members_on("2007-01-08")) == 30 and "MO" in du.members_on("2008-02-15")
    assert "BAC" in du.members_on("2008-02-19") and "MO" not in du.members_on("2008-02-19")
    assert "GOOGL" in du.members_on("2026-10-02") and "VZ" not in du.members_on("2026-10-02")
    tab = du.membership_table()
    assert len(tab[tab["end"] == ""]) == 30
    # a split and a spin-off on a synthetic panel
    days = [f"2020-01-{d:02d}" for d in range(1, 11)]
    prices = pd.DataFrame({"AAPL": 400.0, "UTX": 100.0}, index=days)
    path = du.holding_path("AAPL", "2020-08-28", "2020-09-02")
    assert path[-1] == ("2020-08-31", {"AAPL": 4.0})
    path = du.holding_path("UTX", "2020-04-01", "2020-04-30")
    assert path[-1][1] == {"RTX": 1.0, "CARR": 1.0, "OTIS": 0.5}
    # chain: old DuPont → DowDuPont → DuPont, Dow, Corteva
    final = du.holding_path("DD", "2017-08-01", "2019-12-31")[-1][1]
    assert final == pytest.approx({"DOW": 1.282 / 3, "DD": 1.282 / 3, "CTVA": 1.282 / 3})
    x, event, carried = du.holding_values(prices, "AAPL", days[0], days)
    assert np.allclose(x, 1.0) and not event and carried == 0


def test_holding_values_through_a_spin_off_with_a_late_first_print() -> None:
    days = ["2021-11-02", "2021-11-03", "2021-11-05", "2021-11-08", "2021-11-09"]
    prices = pd.DataFrame(
        {"IBM": [126.0, 127.0, 123.0, 124.0, 125.0], "KD": [np.nan, np.nan, np.nan, 25.0, 26.0]},
        index=days,
    )
    x, event, carried = du.holding_values(prices, "IBM", days[0], days)
    assert event
    assert x[1] == pytest.approx(127 / 126)
    assert x[2] == pytest.approx(127 / 126) and carried >= 1  # Kyndryl has not printed yet
    assert x[3] == pytest.approx((124 + 0.2 * 25) / 126)
    assert x[4] == pytest.approx((125 + 0.2 * 26) / 126)


def test_statistics_for_overlapping_windows() -> None:
    rng = np.random.default_rng(1)
    e = rng.standard_normal(3000)
    overlapping = np.convolve(e, np.ones(13), mode="valid")  # 13-week sums of iid weeks
    _, se, n = st.mean_se(overlapping, 12)
    naive = overlapping.std(ddof=1) / np.sqrt(n)
    assert se == pytest.approx(naive * np.sqrt(13), rel=0.25)
    x = np.convolve(rng.standard_normal(3000), np.ones(13), mode="valid")
    fit = st.ols(overlapping + 0.5 * x, np.column_stack([np.ones(x.size), x]), 12)
    assert fit["coef"][1] == pytest.approx(0.5, abs=0.1) and fit["se"][1] > 0
    cuts = st.tercile_cuts(np.arange(300.0))
    lab = st.tercile(pd.Series([0.0, 150.0, 299.0, np.nan]), cuts)
    assert list(lab[:3]) == ["low", "mid", "high"] and pd.isna(lab[3])
    q = st.bh_qvalues([0.001, 0.02, 0.04, 0.5, np.nan])
    assert q[0] == pytest.approx(0.004) and np.isnan(q[4]) and np.all(np.diff(q[:4]) >= 0)
    r, lo, hi = st.bootstrap_ratio(np.full(100, 2.0), np.full(100, 4.0), 13)
    assert (r, lo, hi) == pytest.approx((0.5, 0.5, 0.5))
    pct = st.expanding_percentile(pd.Series(np.arange(200.0)), min_obs=104)
    assert np.isnan(pct[103]) and pct[150] == 1.0


def test_subset_mean_keeps_the_rows_at_their_place_in_time() -> None:
    rng = np.random.default_rng(6)
    y = np.convolve(rng.standard_normal(2000), np.ones(13), mode="valid")
    mask = (np.arange(y.size) // 50) % 3 == 0  # blocks of 50 weeks, one in three
    mean, se, n = st.subset_mean_se(y, mask, 12)
    assert mean == pytest.approx(float(y[mask].mean())) and n == int(mask.sum())
    assert se > float(
        y[mask].std(ddof=1) / np.sqrt(n)
    )  # overlapping sums: wider than the naive error
    whole = st.subset_mean_se(y, np.ones(y.size, dtype=bool), 12)
    assert whole[:2] == pytest.approx(st.mean_se(y, 12)[:2])
    d = st.describe_subset(y, mask, 12)
    assert d["mean"] == pytest.approx(mean) and d["t"] == pytest.approx(mean / se)


def test_an_expiry_without_strikes_near_the_money_is_not_a_smile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An expiry listed with strikes far from the money only (the vendor's rows of UNH's
    November 2017 expiry on 2017-08-28: strikes 45 to 65 for a share at 195, vols of 160 %)
    must not set the at-the-money vol of the tenor it brackets."""
    from volsto.market.bs import black_price

    spot = 100.0

    def rows(expiry: str, T: float, strikes: np.ndarray, vol: float) -> pd.DataFrame:
        call = black_price(spot, strikes, T, vol, 1.0)
        put = black_price(spot, strikes, T, vol, -1.0)
        return pd.DataFrame(
            {
                "expirDate": expiry, "strike": strikes, "stkPx": spot, "cValue": call,
                "pValue": put, "smoothSmvVol": vol, "iRate": 0.0, "residualRateData": 0.0,
            }
        )  # fmt: skip

    near = rows("2017-10-20", 53 / 365.0, np.arange(80.0, 121.0, 5.0), 0.20)
    far = rows("2017-11-17", 81 / 365.0, np.arange(25.0, 36.0, 2.5), 1.60)
    late = rows("2017-12-15", 109 / 365.0, np.arange(80.0, 121.0, 5.0), 0.20)
    chain = pd.concat([near, far, late], ignore_index=True)
    smiles = ds.expiry_smiles(chain, "2017-08-28", "parity", spot=spot)
    assert [e.expiry for e in smiles] == ["2017-10-20", "2017-12-15"]
    atm = ds.build_marginal(ds.smile_at(smiles, spot, 0.25)).atm_vol
    assert atm == pytest.approx(0.20, abs=2e-3)
    # without the guards the far expiry brackets three months and the vol is nowhere near 20 %
    monkeypatch.setattr(ds, "EXPIRY_GUARDS", False)
    loose = ds.expiry_smiles(chain, "2017-08-28", "parity", spot=spot)
    assert len(loose) == 3
    assert ds.build_marginal(ds.smile_at(loose, spot, 0.25)).atm_vol > 0.5
    monkeypatch.setattr(ds, "EXPIRY_GUARDS", True)
    # a ticker with one-sided expiries only keeps them (nothing better that day)
    alone = ds.expiry_smiles(far, "2017-08-28", "parity", spot=spot)
    assert [e.expiry for e in alone] == ["2017-11-17"]
    assert ds.money_reach(np.array([-0.4, -0.1, 0.05, 0.3])) == (-0.1, 0.05)
    assert ds.two_sided(np.array([-0.1, 0.0, 0.2]))
    assert not ds.two_sided(np.array([-1.4, -1.2, -1.1]))
    # an expiry at ten times the vol of its neighbours (XOM, September 2017 expiry in March
    # 2017) is dropped; a term structure that halves from one end to the other is kept
    wild = pd.concat(
        [
            rows("2017-10-20", 53 / 365.0, np.arange(80.0, 121.0, 5.0), 0.20),
            rows("2017-11-17", 81 / 365.0, np.arange(80.0, 121.0, 5.0), 1.90),
            rows("2017-12-15", 109 / 365.0, np.arange(80.0, 121.0, 5.0), 0.21),
            rows("2018-01-19", 144 / 365.0, np.arange(80.0, 121.0, 5.0), 0.22),
        ],
        ignore_index=True,
    )
    kept = ds.expiry_smiles(wild, "2017-08-28", "parity", spot=spot)
    assert [e.expiry for e in kept] == ["2017-10-20", "2017-12-15", "2018-01-19"]
    assert ds.term_ratio(np.array([0.8, 0.7, 0.6, 0.5, 0.4])) == pytest.approx(
        [0.8 / 0.55, 0.7 / 0.55, 0.6 / 0.6, 0.5 / 0.65, 0.4 / 0.65]
    )
    sloped = [
        ds.ExpirySmile(str(i), 0.1 * (i + 1), 1.0, 0.0, np.array([-0.1, 0.0, 0.1]), np.full(3, v), np.full(3, v), 0)
        for i, v in enumerate([0.8, 0.7, 0.6, 0.5, 0.4])
    ]  # fmt: skip
    assert len(ds.consistent_vols(sloped)) == 5
