"""The comparison report of the local correlation sweeps (``scripts/lcm_report.py``, SPEC §8.7,
the owner's decisions and checks of 2026-10-09): the regression of check (d), the columns the
report derives for the tables that do not carry them, the rows of the summaries with and without
the flagged dates, and the report and the comparison of two passes end to end.

Synthetic frames only: nothing of the study's data is read (the study's tables are written into
``tmp_path``).  Every test is fast.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

SCRIPTS = str(Path(__file__).resolve().parents[1] / "scripts")
if SCRIPTS not in sys.path:
    sys.path.insert(0, SCRIPTS)

import lcm_price as lp  # noqa: E402
import lcm_report as rep  # noqa: E402

# ---------------------------------------------------------------------------------------------
# the regression of check (d)
# ---------------------------------------------------------------------------------------------


def test_ols_recovers_known_coefficients() -> None:
    """Without noise ``ols`` returns the coefficients the data were built with, also when one
    regressor has the scale of a variance (10⁻⁴) beside one of order 0.1, with ``R² = 1`` and
    residuals at rounding level; with noise of known variance the estimates sit within four
    classical standard errors of the truth and ``s²`` near the variance."""
    rng = np.random.default_rng(20261009)
    n = 200
    clip = rng.uniform(0.0, 0.4, n)
    basket = 1e-4 * rng.standard_normal(n)
    truth = np.array([0.02, 0.28, -7.0])
    y = truth[0] + truth[1] * clip + truth[2] * basket
    fit = rep.ols(y, np.column_stack([clip, basket]), ["clip", "basket"])
    print("exact fit:", fit["coef"], "R2", fit["r2"], "s2", fit["s2"])
    assert fit["names"] == ["intercept", "clip", "basket"]
    assert fit["n"] == n and fit["k"] == 3
    np.testing.assert_allclose(fit["coef"], truth, rtol=1e-9, atol=1e-12)
    assert abs(fit["r2"] - 1.0) < 1e-12
    assert fit["s2"] < 1e-24

    sd = 0.01
    noisy = y + sd * rng.standard_normal(n)
    fit = rep.ols(noisy, np.column_stack([clip, basket]), ["clip", "basket"])
    z = (fit["coef"] - truth) / fit["se"]
    print("noisy fit:", fit["coef"], "se", fit["se"], "z", z, "s", math.sqrt(fit["s2"]))
    assert np.all(np.abs(z) < 4.0)
    assert 0.8 * sd < math.sqrt(fit["s2"]) < 1.2 * sd
    assert 0.0 < fit["r2"] < 1.0
    # the two one-regressor fits of the report are the same function with one column
    one = rep.ols(y - truth[2] * basket, clip, ["clip"])
    np.testing.assert_allclose(one["coef"], truth[:2], rtol=1e-9, atol=1e-12)


def test_ols_hand_example() -> None:
    """Four points worked by hand: ``x = (0, 1, 2, 3)``, ``y = (1, 3, 2, 6)``.

    ``x̄ = 1.5``, ``ȳ = 3``, ``S_xx = 5``, ``S_xy = 7``: slope 1.4, intercept 0.9; residuals
    ``(0.1, 0.7, −1.7, 0.9)``, ``e'e = 4.2``, ``s² = 4.2/2 = 2.1``, ``S_yy = 14``, ``R² = 0.7``.
    Classical: ``var(slope) = s²/S_xx = 0.42``, ``var(intercept) = s²(1/n + x̄²/S_xx) = 1.47``.
    HC1 with ``n/(n − k) = 2`` and the weights of each estimate
    (slope ``(x − x̄)/S_xx = (−0.3, −0.1, 0.1, 0.3)``, intercept ``1/n − x̄(x − x̄)/S_xx =
    (0.7, 0.4, 0.1, −0.2)``): ``var(slope) = 2·Σ w²e² = 2·0.1076 = 0.2152``,
    ``var(intercept) = 2·0.1446 = 0.2892``."""
    fit = rep.ols(np.array([1.0, 3.0, 2.0, 6.0]), np.array([0.0, 1.0, 2.0, 3.0]), ["x"])
    print({k: v for k, v in fit.items() if k != "names"})
    np.testing.assert_allclose(fit["coef"], [0.9, 1.4], rtol=1e-12)
    np.testing.assert_allclose(fit["se"], [math.sqrt(1.47), math.sqrt(0.42)], rtol=1e-12)
    np.testing.assert_allclose(fit["se_hc1"], [math.sqrt(0.2892), math.sqrt(0.2152)], rtol=1e-12)
    np.testing.assert_allclose(fit["t"], [0.9 / math.sqrt(1.47), 1.4 / math.sqrt(0.42)], rtol=1e-12)
    np.testing.assert_allclose(
        fit["t_hc1"], [0.9 / math.sqrt(0.2892), 1.4 / math.sqrt(0.2152)], rtol=1e-12
    )
    assert abs(fit["r2"] - 0.7) < 1e-12
    assert abs(fit["s2"] - 2.1) < 1e-12
    assert (fit["n"], fit["k"]) == (4, 2)


def test_ols_hc1_matches_the_direct_formula() -> None:
    """On a small heteroskedastic example with two regressors, the classical and the HC1
    covariances of ``ols`` equal the textbook formulas written out directly:
    ``s²(Z'Z)⁻¹`` and ``n/(n − k)·(Z'Z)⁻¹ (Σ e_i² z_i z_i') (Z'Z)⁻¹``, the sum taken
    observation by observation; with one regressor the HC1 variance of the slope equals the
    closed form ``n/(n − 2)·Σ (x_i − x̄)² e_i² / S_xx²``."""
    rng = np.random.default_rng(7)
    n = 12
    x = np.column_stack([rng.uniform(0.0, 1.0, n), rng.standard_normal(n)])
    y = 0.3 + 1.5 * x[:, 0] - 0.4 * x[:, 1] + (0.05 + 0.5 * x[:, 0]) * rng.standard_normal(n)
    fit = rep.ols(y, x, ["a", "b"])
    z = np.column_stack([np.ones(n), x])
    zz_inv = np.linalg.inv(z.T @ z)
    coef = zz_inv @ z.T @ y
    e = y - z @ coef
    k = 3
    classical = float(e @ e) / (n - k) * zz_inv
    meat = np.zeros((k, k))
    for i in range(n):
        meat += e[i] ** 2 * np.outer(z[i], z[i])
    hc1 = n / (n - k) * zz_inv @ meat @ zz_inv
    print("coef", fit["coef"], "se", fit["se"], "HC1", fit["se_hc1"], "direct HC1", np.sqrt(np.diag(hc1)))  # fmt: skip
    np.testing.assert_allclose(fit["coef"], coef, rtol=1e-10)
    np.testing.assert_allclose(fit["se"], np.sqrt(np.diag(classical)), rtol=1e-10)
    np.testing.assert_allclose(fit["se_hc1"], np.sqrt(np.diag(hc1)), rtol=1e-10)
    np.testing.assert_allclose(fit["t_hc1"], coef / np.sqrt(np.diag(hc1)), rtol=1e-10)
    assert abs(fit["r2"] - (1.0 - float(e @ e) / float(np.sum((y - y.mean()) ** 2)))) < 1e-12
    # the two covariances differ on heteroskedastic data: the test would not pass on a mix-up
    assert not np.allclose(fit["se"], fit["se_hc1"], rtol=0.02)

    x1 = x[:, 0]
    one = rep.ols(y, x1, ["a"])
    sxx = float(np.sum((x1 - x1.mean()) ** 2))
    slope = float(np.sum((x1 - x1.mean()) * (y - y.mean()))) / sxx
    res = y - (y.mean() - slope * x1.mean()) - slope * x1
    closed = n / (n - 2) * float(np.sum((x1 - x1.mean()) ** 2 * res**2)) / sxx**2
    assert abs(one["coef"][1] - slope) < 1e-12
    assert abs(one["se_hc1"][1] - math.sqrt(closed)) < 1e-12
    assert abs(one["se"][1] - math.sqrt(float(res @ res) / (n - 2) / sxx)) < 1e-12


def test_ols_refuses_a_degenerate_design_and_regress_reports_none() -> None:
    """``ols`` raises when there are no more observations than coefficients or when a regressor
    does not vary; ``regress`` drops the rows with a missing value and returns ``None`` in
    those cases, so that a partial table gives n/a and not an error."""
    with pytest.raises(ValueError, match="observations"):
        rep.ols(np.array([1.0, 2.0]), np.array([0.0, 1.0]), ["x"])
    with pytest.raises(ValueError, match="rank"):
        rep.ols(np.arange(6.0), np.ones(6), ["x"])
    with pytest.raises(ValueError, match="names"):
        rep.ols(np.arange(6.0), np.arange(6.0), ["x", "y"])
    frame = pd.DataFrame(
        {
            "y": [1.0, 3.0, 2.0, 6.0, np.nan, 9.0],
            "x": [0.0, 1.0, 2.0, 3.0, 4.0, np.inf],
            "c": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }
    )
    fit = rep.regress(frame, "y", ["x"])
    assert fit is not None and fit["n"] == 4  # the NaN and the infinite rows are dropped
    np.testing.assert_allclose(fit["coef"], [0.9, 1.4], rtol=1e-12)
    assert rep.regress(frame, "y", ["c"]) is None
    assert rep.regress(frame.iloc[:2], "y", ["x"]) is None
    assert rep.regress(frame, "y", ["absent"]) is None
    text = rep.regression_table([("fit", fit), ("none", None)], {"x": "the regressor"})
    print(text)
    assert "| fit | intercept | +0.9000 |" in text and "the regressor | +1.4000" in text
    assert "0.700 | 4 |" in text and "| none | n/a" in text


# ---------------------------------------------------------------------------------------------
# the derived columns
# ---------------------------------------------------------------------------------------------


def base_rows() -> pd.DataFrame:
    """Five rows with the base columns only (a table written before 2026-10-09): three priced,
    one priced at the 1 % boundary of the clip flags, one failed."""
    nan = float("nan")
    return pd.DataFrame(
        {
            "tenor": ["3m"] * 5,
            "status": ["ok", "check", "ok", "ok", "failed"],
            "ED_lc": [0.097, 0.050, 0.110, 0.080, nan],
            "ED_lc_se": [0.0006, 0.0003, 0.0007, 0.0005, nan],
            "ED_cc": [0.100, 0.052, 0.108, 0.081, nan],
            "ED_cc_se": [0.0007, 0.0004, 0.0008, 0.0006, nan],
            "P_D_copula": [0.1, 0.05, 0.1, 0.08, nan],
            "EV_copula": [0.0174, 0.0044, 0.0180, 0.0110, nan],
            "EQV": [0.0160, 0.0040, 0.0190, 0.0100, nan],
            "clip_low_inner_max": [0.0, 0.02, 0.0100, 0.010001, nan],
            "clip_high_inner_max": [0.15, 0.003, 0.0100, 0.0, nan],
            "clip_inner_max": [0.15, 0.02, 0.0100, 0.010001, nan],
            "E_Rbar2_lc": [0.0088, 0.0021, 0.0101, 0.0050, nan],
            "E_Rbar2_lc_se": [4e-5, 1e-5, 5e-5, 2e-5, nan],
            "M_B_listed": [0.0092, 0.0020, 0.0100, 0.0050, nan],
            "ratio": [0.97, 0.9615, 1.0185, 0.9877, nan],
        },
        index=pd.to_datetime(
            ["2024-01-02", "2024-02-05", "2024-03-04", "2024-04-01", "2024-05-06"]
        ),
    )


def test_derived_columns() -> None:
    """``derive`` on a table without the columns of 2026-10-09: ``lc_over_copula = ED_lc/P_D``
    and ``cc_over_copula = ED_cc/P_D`` with the numerators' errors over ``P_D``;
    ``listed_fwd_ratio = √(EQV/EV_copula)`` and ``listed_fwd = P_D`` times it; the clip flags
    are strict at 1 % (a mass of exactly 0.01 is not flagged, 0.010001 is) and ``flag_clip`` is
    either; ``flag_unscreened`` is ``False`` without ``n_names_unscreened`` and
    ``n_names_unscreened > 0`` with it; the basket part of the ``E[V]`` split is
    ``E_Rbar2_lc − M_B_listed``, also over ``M_B_listed``; a failed row has NaN quantities and
    no flag; a value the table already carries is kept and only its gaps are filled; model S's
    ratio exists on its converged dates only."""
    rows = base_rows()
    out = rep.derive(rows)
    print(out[[*rep.DERIVED, "EV_basket_part", "EV_basket_rel", "EV_basket_rel_se"]].T)
    priced = rows["status"] != "failed"
    p = rows.loc[priced]
    np.testing.assert_allclose(out.loc[priced, "lc_over_copula"], p["ED_lc"] / p["P_D_copula"])
    np.testing.assert_allclose(out.loc[priced, "cc_over_copula"], p["ED_cc"] / p["P_D_copula"])
    np.testing.assert_allclose(
        out.loc[priced, "lc_over_copula_se"], p["ED_lc_se"] / p["P_D_copula"]
    )
    np.testing.assert_allclose(
        out.loc[priced, "cc_over_copula_se"], p["ED_cc_se"] / p["P_D_copula"]
    )
    np.testing.assert_allclose(out.loc[priced, "listed_fwd_ratio"], np.sqrt(p["EQV"] / p["EV_copula"]))  # fmt: skip
    np.testing.assert_allclose(
        out.loc[priced, "listed_fwd"], p["P_D_copula"] * np.sqrt(p["EQV"] / p["EV_copula"])
    )
    assert abs(out["lc_over_copula"].iloc[0] - 0.97) < 1e-12
    assert abs(out["listed_fwd_ratio"].iloc[0] - math.sqrt(0.016 / 0.0174)) < 1e-12
    assert out["flag_clip_low"].tolist() == [False, True, False, True, False]
    assert out["flag_clip_high"].tolist() == [True, False, False, False, False]
    assert out["flag_clip"].tolist() == [True, True, False, True, False]
    assert out["flag_unscreened"].tolist() == [False] * 5
    assert out["indicative"].tolist() == [False] * 5
    for name in ("flag_clip_low", "flag_clip_high", "flag_clip", "flag_unscreened", "indicative"):
        assert out[name].dtype == bool
    np.testing.assert_allclose(
        out.loc[priced, "EV_basket_part"], p["E_Rbar2_lc"] - p["M_B_listed"], atol=1e-18
    )
    np.testing.assert_allclose(
        out.loc[priced, "EV_basket_rel"], p["E_Rbar2_lc"] / p["M_B_listed"] - 1.0, atol=1e-12
    )
    np.testing.assert_allclose(
        out.loc[priced, "EV_basket_rel_se"], p["E_Rbar2_lc_se"] / p["M_B_listed"]
    )
    failed = out.loc[~priced].iloc[0]
    for name in ("lc_over_copula", "cc_over_copula", "listed_fwd_ratio", "listed_fwd", "EV_basket_rel"):  # fmt: skip
        assert math.isnan(failed[name])
    assert "S_ratio" not in out.columns and "gap_to_S" not in out.columns

    # with the fallback's column, and with columns the pricer already wrote
    rows["n_names_unscreened"] = [0.0, 2.0, 0.0, 1.0, np.nan]
    rows["lc_over_copula"] = [9.0, np.nan, 9.5, np.nan, np.nan]
    rows["flag_clip_high"] = pd.Series([False, True, None, None, None], index=rows.index, dtype=object)  # fmt: skip
    rows["tenor"] = "24m"
    out = rep.derive(rows)
    assert out["flag_unscreened"].tolist() == [False, True, False, True, False]
    assert out["lc_over_copula"].iloc[0] == 9.0 and out["lc_over_copula"].iloc[2] == 9.5
    assert abs(out["lc_over_copula"].iloc[1] - 1.0) < 1e-12
    assert abs(out["lc_over_copula"].iloc[3] - 1.0) < 1e-12
    assert out["flag_clip_high"].tolist() == [False, True, False, False, False]
    assert out["flag_clip"].tolist() == [False, True, False, True, False]
    assert out["indicative"].tolist() == [True] * 5

    # the study's entry fills a failed row; model S on its converged dates
    rows["P_D_cop"] = [0.1, 0.05, 0.1, 0.08, 0.2]
    rows["EV_cop"] = [0.0174, 0.0044, 0.0180, 0.0110, 0.05]
    rows["EQV_cop"] = [0.0160, 0.0040, 0.0190, 0.0100, 0.045]
    rows["P_D_S"] = [0.094, 0.049, 0.09, 0.08, 0.19]
    rows["converged"] = pd.Series([True, True, False, None, True], index=rows.index, dtype=object)
    for t in rep.TAGS:
        rows[f"C_S_{t}"] = 0.5
        rows[f"C_{t}_cop"] = [1.0, 0.0, 1.0, 1.0, 1.0]
    out = rep.derive(rows)
    assert abs(out["listed_fwd_ratio"].iloc[4] - math.sqrt(0.045 / 0.05)) < 1e-12
    np.testing.assert_allclose(out["S_ratio"].iloc[[0, 1, 4]], [0.94, 0.98, 0.95])
    assert out["S_ratio"].iloc[[2, 3]].isna().all()
    assert abs(out["gap_to_S"].iloc[0] - (0.97 - 0.94)) < 1e-12
    assert math.isnan(out["gap_to_S"].iloc[2]) and math.isnan(out["gap_to_S"].iloc[4])
    assert out["S_C_100"].iloc[0] == 0.5 and math.isnan(out["S_C_100"].iloc[1])


def test_derived_columns_agree_with_the_pricer() -> None:
    """The report's ``derive`` and the pricer's ``derived_columns`` (the function that writes the
    columns into the rows from 2026-10-09) give the same values on the same base columns, row
    by row, and the constants they share are equal: the report of an old table and the columns
    of a new one cannot drift apart."""
    assert tuple(lp.DERIVED_COLUMNS) == rep.DERIVED
    assert rep.CLIP_FLAG_MASS == lp.CLIP_FLAG_MASS
    assert tuple(lp.INDICATIVE_TENORS) == rep.INDICATIVE_TENORS
    assert tuple(name for name, _ in rep.GATES) == tuple(lp.GATING_CHECKS)
    assert rep.DIAGNOSTIC[0] == "check_names"
    for tenor, unscreened in (("3m", None), ("24m", [0.0, 2.0, 0.0, 1.0, np.nan])):
        rows = base_rows()
        rows["tenor"] = tenor
        if unscreened is not None:
            rows["n_names_unscreened"] = unscreened
        out = rep.derive(rows)
        for i in range(len(rows)):
            row: dict[str, Any] = rows.iloc[i].to_dict()
            want = lp.derived_columns(row)
            for name in rep.DERIVED:
                got = out[name].iloc[i]
                if isinstance(want[name], bool):
                    assert bool(got) is want[name], (tenor, i, name)
                elif math.isnan(want[name]):
                    assert math.isnan(got), (tenor, i, name)
                else:
                    assert abs(got - want[name]) <= 1e-14 * abs(want[name]), (tenor, i, name)


# ---------------------------------------------------------------------------------------------
# the rows of the summaries
# ---------------------------------------------------------------------------------------------


def flagged_frame() -> pd.DataFrame:
    """Ten priced rows: rows 2 and 7 carry a name kept unscreened, rows 0, 2, 3 and 4 a clip
    flag (0 and 2 on the low side); the flagged rows have their own level of ``ratio``."""
    index = pd.date_range("2024-01-01", periods=10, freq="MS")
    unscreened = np.array([0, 0, 1, 0, 0, 0, 0, 1, 0, 0], dtype=bool)
    low = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=bool)
    high = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    ratio = 0.96 + 0.001 * np.arange(10) + 0.5 * unscreened + 0.1 * (low | high)
    return pd.DataFrame(
        {
            "ratio": ratio,
            "ratio_se": 0.0002 + 0.0001 * np.arange(10),
            "flag_unscreened": unscreened,
            "flag_clip_low": low,
            "flag_clip_high": high,
            "flag_clip": low | high,
        },
        index=index,
    )


def test_subsets_drop_exactly_the_flagged_rows() -> None:
    """At 3m the summaries are on all priced dates and on the same without the dates on which a
    name is kept unscreened: the second set is the first minus exactly the flagged rows, and the
    clip flags remove nothing; without a flagged date there is one set.  At 12m and 24m the
    reported summary leaves out exactly the rows with ``flag_clip`` (with and without the
    unscreened dates), and the summaries on all priced rows and on the rows without the low-side
    flag are labelled for information only.  ``summary`` on a set equals the statistics of the
    rows of that set taken by hand."""
    ok = flagged_frame()
    subs = rep.subsets(ok, "3m")
    for s in subs:
        print("3m:", s.label, int(s.mask.sum()), rep.summary(ok.loc[s.mask, "ratio"]))
    assert len(subs) == 2 and not any(s.information for s in subs)
    assert subs[0].mask.all() and "all priced dates" in subs[0].label
    assert subs[1].label == "without the 2 dates on which a name is kept unscreened"
    kept = ok.index[subs[1].mask]
    assert list(kept) == [d for d, f in ok["flag_unscreened"].items() if not f]
    assert set(ok.index) - set(kept) == set(ok.index[[2, 7]])
    with_all = rep.summary(ok.loc[subs[0].mask, "ratio"], ok.loc[subs[0].mask, "ratio_se"])
    without = rep.summary(ok.loc[subs[1].mask, "ratio"], ok.loc[subs[1].mask, "ratio_se"])
    by_hand = ok["ratio"].to_numpy()[[0, 1, 3, 4, 5, 6, 8, 9]]
    assert (with_all["n"], without["n"]) == (10, 8)
    assert abs(without["mean"] - by_hand.mean()) < 1e-15
    assert abs(without["median"] - np.median(by_hand)) < 1e-15
    assert abs(without["max"] - by_hand.max()) < 1e-15
    assert abs(without["se of the mean"] - by_hand.std(ddof=1) / math.sqrt(8)) < 1e-15
    assert abs(without["median MC se"] - np.median(ok["ratio_se"].to_numpy()[[0, 1, 3, 4, 5, 6, 8, 9]])) < 1e-15  # fmt: skip
    assert abs(with_all["mean"] - ok["ratio"].mean()) < 1e-15
    assert with_all["mean"] - without["mean"] > 0.05  # the flagged rows moved the mean
    # one flagged date: the label is in the singular
    one = ok.copy()
    one["flag_unscreened"] = [False] * 9 + [True]
    assert rep.subsets(one, "3m")[1].label == "without the 1 date on which a name is kept unscreened"  # fmt: skip
    # no flagged date: one set, nothing to drop
    none = ok.copy()
    none["flag_unscreened"] = False
    assert [s.label for s in rep.subsets(none, "3m")] == ["all priced dates"]

    for tenor in ("12m", "24m"):
        subs = rep.subsets(ok, tenor)
        for s in subs:
            print(f"{tenor}:", s.label, int(s.mask.sum()), rep.summary(ok.loc[s.mask, "ratio"]))
        assert [s.information for s in subs] == [False, False, True, True]
        assert list(ok.index[subs[0].mask]) == list(ok.index[[1, 5, 6, 7, 8, 9]])
        assert list(ok.index[subs[1].mask]) == list(ok.index[[1, 5, 6, 8, 9]])
        assert subs[2].mask.all()
        assert list(ok.index[subs[3].mask]) == list(ok.index[[1, 3, 4, 5, 6, 7, 8, 9]])
        assert "the reported summary" in subs[0].label
        assert "without the 1 date on which a name is kept unscreened" in subs[1].label
        assert all(s.label.startswith("for information only") for s in subs[2:])
        assert "4 clip-flagged" in subs[2].label and "2 removed" in subs[3].label
        reported = rep.summary(ok.loc[subs[0].mask, "ratio"])
        assert reported["n"] == 6
        assert abs(reported["mean"] - ok["ratio"].to_numpy()[[1, 5, 6, 7, 8, 9]].mean()) < 1e-15
        # an unscreened date that is also clip-flagged is already out: no second set for it
        only_clipped = ok.copy()
        only_clipped["flag_unscreened"] = [False, False, True] + [False] * 7
        assert [s.information for s in rep.subsets(only_clipped, tenor)] == [False, True, True]


# ---------------------------------------------------------------------------------------------
# the report and the comparison end to end, on synthetic tables
# ---------------------------------------------------------------------------------------------

GAP = (0.01, 0.25, -0.1)  # the planted intercept and slopes of LC/CC - P_D_S/P_D


def study_tables(folder: Path, tenor: str, dates: list[str], rng: np.random.Generator) -> pd.DataFrame:  # fmt: skip
    """A synthetic study: ``entries_<tenor>.parquet`` with baskets B1 and B2 and, at 3m,
    ``model_s_3m.parquet`` (the second date not converged).  Returns the B1 entry by date."""
    n = len(dates)
    b1 = pd.DataFrame(
        {
            "date": dates,
            "basket": "B1",
            "P_D": rng.uniform(0.05, 0.15, n),
            "P_D_se": 1e-5,
            "EV": rng.uniform(0.005, 0.03, n),
            "kappa_cop": rng.uniform(0.72, 0.78, n),
        }
    )
    b1["EQV"] = b1["EV"] * rng.uniform(0.8, 1.0, n)
    for t in rep.TAGS:
        b1[f"C_{t}"] = rng.uniform(0.001, 0.05, n)
    b2 = b1.assign(basket="B2", P_D=9.0)
    pd.concat([b1, b2]).to_parquet(folder / f"entries_{tenor}.parquet")
    if tenor == "3m":
        p_d_s = b1["P_D"] * rng.uniform(0.9, 1.0, n)
        s = pd.DataFrame({"date": dates, "converged": [i != 1 for i in range(n)], "P_D_S": p_d_s})
        for t in rep.TAGS:
            s[f"C_S_{t}"] = rng.uniform(0.001, 0.05, n)
        s.to_parquet(folder / "model_s_3m.parquet")
    return b1.set_index("date")


def sweep_table(
    entries: pd.DataFrame,
    tenor: str,
    rng: np.random.Generator,
    commit: str,
    failed: dict[str, str],
    unscreened: dict[str, str] | None = None,
    model_s: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """A synthetic sweep table on the dates of ``entries``: the base columns of the report,
    ``failed`` dates with their reasons (the numbers absent), and ``LC/CC`` built from model S,
    the clipped mass and the basket part with the coefficients ``GAP`` when ``model_s`` is
    given.  ``unscreened`` (date to names) adds the fallback's columns."""
    dates = list(entries.index)
    n = len(dates)
    rows = pd.DataFrame({"date": dates})
    rep_joined = set(rep.JOINED)
    for c in rep.BASE:
        if c not in rep_joined and c not in ("status", "reason", "git_commit"):
            rows[c] = rng.uniform(0.5, 1.5, n)
    rows["tenor"], rows["git_commit"], rows["status"], rows["reason"] = tenor, commit, "ok", ""
    rows["n_particles"], rows["n_paths"], rows["threads"] = 200000.0, 200000.0, 4.0
    rows["P_D_copula"] = entries["P_D"].to_numpy()
    rows["EV_copula"] = entries["EV"].to_numpy()
    rows["EQV"] = entries["EQV"].to_numpy()
    rows["ED_cc"] = rows["P_D_copula"] * rng.uniform(0.97, 1.03, n)
    rows["clip_low_inner_max"] = rng.uniform(0.0, 0.004, n)
    rows["clip_high_inner_max"] = rng.uniform(0.0, 0.3, n)
    rows["clip_inner_max"] = rows[["clip_low_inner_max", "clip_high_inner_max"]].max(axis=1)
    rows["M_B_listed"] = rng.uniform(0.004, 0.02, n)
    rows["E_Rbar2_lc"] = rows["M_B_listed"] * (1.0 + rng.uniform(-0.15, 0.05, n))
    rows["E_Rbar2_lc_se"] = 1e-5
    rows["ratio"] = rng.uniform(0.94, 1.0, n)
    if model_s is not None:
        s_ratio = model_s["P_D_S"].to_numpy() / entries["P_D"].to_numpy()
        rel = rows["E_Rbar2_lc"] / rows["M_B_listed"] - 1.0
        rows["ratio"] = s_ratio + GAP[0] + GAP[1] * rows["clip_inner_max"] + GAP[2] * rel
    rows["ED_lc"] = rows["ED_cc"] * rows["ratio"]
    rows["ED_lc_se"], rows["ED_cc_se"], rows["ratio_se"] = 5e-5, 6e-5, 2e-4
    for c in ("check_no_nan", "check_forward", "check_index", "check_names", "cache_hit", "index_extrapolated", "align_flagged"):  # fmt: skip
        rows[c] = pd.Series([True] * n, dtype=object)
    rows.loc[3, "check_names"], rows.loc[3, "status"] = False, "check"
    if n > 4:
        rows.loc[4, "check_forward"], rows.loc[4, "status"] = False, "check"
    rows["seconds_total"] = rng.uniform(60.0, 120.0, n)
    if unscreened is not None:
        rows["n_names_unscreened"] = [1.0 if d in unscreened else 0.0 for d in dates]
        rows["names_unscreened"] = [unscreened.get(d, "") for d in dates]
        rows["calendar_repair"] = pd.Series([True] * n, dtype=object)
        rows["n_dropped_calendar"] = rng.integers(0, 40, n).astype(float)
    numbers = [c for c in rows.columns if c not in ("date", "tenor", "git_commit", "status", "reason")]  # fmt: skip
    for d, reason in failed.items():
        i = dates.index(d)
        rows.loc[i, "status"], rows.loc[i, "reason"] = "failed", reason
        for c in numbers:
            rows.loc[i, c] = None
    return rows


def tables_of(report: str) -> list[tuple[str, str]]:
    """The tables of a markdown report as (the caption line before it, its header line)."""
    lines = report.splitlines()
    out = []
    for i, line in enumerate(lines):
        if line.startswith("|") and i + 1 < len(lines) and lines[i + 1].startswith("|---"):
            caption = next(x for x in reversed(lines[:i]) if x.strip())
            out.append((caption, line))
    return out


def test_build_and_compare_on_synthetic_tables(tmp_path: Path) -> None:
    """The report of a pass with the fallback's flags and of an older pass without the new
    columns, and the comparison of the two in both directions, on synthetic tables.

    * the failed dates are listed, every one with its reason, and stay in the by-date file;
    * the flagged dates are named, and every summary table exists for all priced dates and for
      the dates without the flagged ones, the second with exactly that many dates fewer;
    * the forward's table puts LC/copula and CC/copula directly after model S/copula;
    * the regression of check (d) recovers the coefficients ``LC/CC − P_D_S/P_D`` was built
      with, through the same ``load``, ``complete`` and ``derive`` the report uses;
    * the names' 2 % check is worded as a diagnostic and is not among the gates;
    * the study-only line gives the means of ``√(EQV/EV)`` and ``P_D_S/P_D`` on the converged
      dates of the study's own tables;
    * the comparison runs between two tables with different columns and different commits and
      is named ``report_<stem>_vs_<other>.md``; a table with no priced date still gives a
      report."""
    rng = np.random.default_rng(12345)
    study, out = tmp_path / "study", tmp_path / "outputs" / "dispersion_lc"
    study.mkdir()
    out.mkdir(parents=True)
    dates = [f"{d:%Y-%m-%d}" for d in pd.date_range("2020-01-06", periods=24, freq="28D")]
    entries = study_tables(study, "3m", dates, rng)
    model_s = pd.read_parquet(study / "model_s_3m.parquet").set_index("date")
    flagged = {dates[5]: "GM", dates[9]: "GM,UNH"}
    new_failed = {dates[20]: "ValueError: the index: no listed expiry passes the screen"}
    new = sweep_table(entries, "3m", rng, "aaaaaaa", new_failed, flagged, model_s)
    old_failed = {
        dates[5]: "ValueError: GM: no listed expiry passes the screen",
        dates[9]: "ValueError: GM: no listed expiry passes the screen",
        dates[20]: "ValueError: the index: no listed expiry passes the screen",
    }
    old = sweep_table(entries, "3m", rng, "bbbbbbb", old_failed, model_s=model_s)
    old = old.drop(columns=["names_mc_over_listed", "clip_high_max"], errors="ignore")
    new.to_parquet(out / "lcm_3m_dev_new.parquet")
    old.to_parquet(out / "lcm_3m_dev_old.parquet")

    path = rep.build("3m", "development", out, "new", study)
    assert path == out / "report_3m_dev_new.md"
    text = path.read_text()
    print(text[:6000])
    # failed and flagged dates
    assert f"| {dates[20]} | ValueError: the index: no listed expiry passes the screen |" in text
    assert "**1 date(s) failed**" in text
    assert f"on 2 of 23 priced dates**: {dates[5]} (GM), {dates[9]} (GM,UNH)." in text
    # every table is given for both sets of rows; the second has two dates fewer
    captions = [c for c, header in tables_of(text) if not header.startswith("| date |")]
    with_all = [c for c in captions if "all priced dates: 23 dates" in c]
    without = [c for c in captions if "without the 2 dates on which a name is kept unscreened: 21 dates" in c]  # fmt: skip
    assert len(with_all) == len(without) >= 9
    assert len(with_all) + len(without) == len(captions)
    # like for like: LC/copula and CC/copula directly beside model S/copula
    block = text[text.index("**The forward") :].splitlines()
    assert block[4].startswith("| model S / copula (P_D_S / P_D) | 22 |")
    assert block[5].startswith("| **E_LC[D] / copula P_D** | 23 |")
    assert block[6].startswith("| **E_CC[D] / copula P_D** | 23 |")
    assert block[7].startswith("| listed-variance forward / copula P_D")
    assert block[8].startswith("| E_LC[D] / E_CC[D] (paired) | 23 |")
    # the diagnostic is not a gate
    gates = text[text.index("## 6. Gates") : text.index("## 7. Timings")]
    assert "Reported diagnostic, not a gate (owner's decision 3 of 2026-10-09)" in gates
    assert "within 2 % of the listed strips on 22 of 23 dates" in gates
    assert "| E[B_T] within 3 standard errors of its forward | 22 of 23 | " + dates[4] in gates
    assert "| Σ w E[R_i²] within 2 %" not in gates
    assert "Status `check` is carried by 2 of these dates in the table; 1 of them" in gates
    # the regression of check (d), through the report's own loading
    frame = rep.derive(rep.complete(rep.load("3m", "development", out, "new", study)[0])[0])
    ok = frame[frame["status"] != "failed"]
    fit = rep.regress(ok, "gap_to_S", ["clip_inner_max", "EV_basket_rel"])
    assert fit is not None
    print("planted", GAP, "recovered", fit["coef"], "n", fit["n"], "R2", fit["r2"])
    assert fit["n"] == 22  # 24 dates, one failed, one without a converged model S
    np.testing.assert_allclose(fit["coef"], GAP, atol=1e-10)
    assert abs(fit["r2"] - 1.0) < 1e-10
    assert f"| both regressors | intercept | {GAP[0]:+.4f} |" in text
    assert f"|  | clipped mass inside ±2.5 sd | {GAP[1]:+.4f} |" in text
    assert f"|  | basket part / M_B^listed | {GAP[2]:+.4f} |" in text
    assert "| 1.000 | 22 |" in text and "| 1.000 | 20 |" in text
    # the study-only line
    line = rep.study_line(study, "3m")
    assert line is not None
    conv = model_s["converged"].to_numpy()
    assert line["n"] == 23 and line["n_model_s"] == 24
    assert abs(line["mean_listed"] - np.sqrt(entries["EQV"] / entries["EV"]).to_numpy()[conv].mean()) < 1e-14  # fmt: skip
    assert abs(line["mean_s"] - (model_s["P_D_S"] / entries["P_D"]).to_numpy()[conv].mean()) < 1e-14  # fmt: skip
    assert line["last"] == dates[-1]
    assert f"the mean of `√(EQV/EV)` is {line['mean_listed']:.4f}" in text
    # the by-date file: every date, the failed one included
    csv = pd.read_csv(out / "report_3m_dev_new_by_date.csv")
    assert list(csv["date"]) == dates
    assert csv.loc[20, "status"] == "failed" and "no listed expiry" in csv.loc[20, "reason"]
    assert math.isnan(csv.loc[20, "lc_over_cc"]) and not bool(csv.loc[20, "in_summary"])
    assert csv.loc[5, "flags"].startswith("unscreened(GM)")
    assert csv.loc[9, "flags"].startswith("unscreened(GM,UNH)")
    assert math.isnan(csv.loc[1, "model_s_over_copula"])  # model S not converged there
    row = new.set_index("date").loc[dates[0]]
    assert abs(csv.loc[0, "lc_over_copula"] - row["ED_lc"] / row["P_D_copula"]) < 1e-7
    assert abs(csv.loc[0, "cc_over_copula"] - row["ED_cc"] / row["P_D_copula"]) < 1e-7
    assert abs(csv.loc[0, "listed_fwd_ratio"] - math.sqrt(row["EQV"] / row["EV_copula"])) < 1e-7
    assert csv.loc[4, "gates_failed"] == "check_forward"
    assert str(csv.loc[3, "names_within_2pct"]) == "False"
    for name in ("clip_low_inner_max", "clip_high_inner_max", "EV_basket_part", "kappa_lc", "kappa_cc", "kappa_copula"):  # fmt: skip
        assert name in csv.columns

    # the older table: no fallback column, two base columns absent
    text_old = rep.build("3m", "development", out, "old", study).read_text()
    assert "**3 date(s) failed**" in text_old
    for d, reason in old_failed.items():
        assert f"| {d} | {reason} |" in text_old
    assert "no date is flagged" in text_old
    assert "on which a name is kept unscreened" not in text_old
    assert "Columns the table does not carry, reported as n/a: `clip_high_max`." in text_old

    # the comparison, both ways, between tables of different columns and commits
    one = rep.compare("3m", "development", out, "new", "old", study)
    two = rep.compare("3m", "development", out, "old", "new", study)
    assert one == out / "report_3m_dev_new_vs_old.md"
    assert two == out / "report_3m_dev_old_vs_new.md"
    cmp = one.read_text()
    print(cmp)
    assert "commit(s) aaaaaaa" in cmp and "commit(s) bbbbbbb" in cmp
    assert "21 dates are priced by both" in cmp
    assert f"Priced by the pass `new` only (2): {dates[5]}, {dates[9]}." in cmp
    assert f"{dates[5]}: ValueError: GM: no listed expiry passes the screen" in cmp
    assert "Priced by the pass `old` only (0): none." in cmp
    both = [d for d in dates if d not in old_failed]
    a, b = new.set_index("date").loc[both], old.set_index("date").loc[both]
    want = float((a["ED_lc"].astype(float) / b["ED_lc"].astype(float) - 1.0).mean())
    back = float((b["ED_lc"].astype(float) / a["ED_lc"].astype(float) - 1.0).mean())
    assert f"| E_LC[D], relative | 21 | {want:.4f} |" in cmp
    assert f"| E_LC[D], relative | 21 | {back:.4f} |" in two.read_text()
    diff = float((a["ratio"].astype(float) - b["ratio"].astype(float)).mean())
    assert f"{diff:.4f}" != f"{-diff:.4f}"  # the two directions are told apart
    assert f"| LC/CC of the forward, difference | 21 | {diff:.4f} |" in cmp
    assert f"| LC/CC of the forward, difference | 21 | {-diff:.4f} |" in two.read_text()
    assert "names' second moment over listed strips - 1: the pass `old` | 0 | n/a" in cmp

    # a table that does not exist: the comparison says so through FileNotFoundError
    with pytest.raises(FileNotFoundError):
        rep.compare("3m", "development", out, "new", "", study)
    # a table without a priced date still gives a report and a by-date file
    nothing = sweep_table(entries, "3m", rng, "ccccccc", dict.fromkeys(dates, "RuntimeError: stopped"))  # fmt: skip
    nothing[["date", "tenor", "git_commit", "status", "reason"]].to_parquet(out / "lcm_3m_dev_none.parquet")  # fmt: skip
    empty = rep.build("3m", "development", out, "none", study).read_text()
    assert "**24 date(s) failed**" in empty and "No date of the table is priced" in empty
    assert len(pd.read_csv(out / "report_3m_dev_none_by_date.csv")) == 24


def test_long_tenor_summaries_leave_the_clip_flagged_rows_out(tmp_path: Path) -> None:
    """At 24m (owner's decision 5): the reported summaries are on the rows without a clip flag,
    the flagged dates are counted and listed with their clipped masses and stay in the by-date
    file, the summaries on all priced rows and without the low-side flag only are printed and
    labelled for information, there is no model S (no regression), and every table of the
    report is captioned "indicative (four dates)"."""
    rng = np.random.default_rng(2024)
    study, out = tmp_path / "study", tmp_path / "outputs" / "dispersion_lc"
    study.mkdir()
    out.mkdir(parents=True)
    dates = ["2017-04-03", "2019-09-03", "2024-08-05", "2026-10-02"]
    entries = study_tables(study, "24m", dates, rng)
    rows = sweep_table(entries, "24m", rng, "ddddddd", {})
    rows["clip_low_inner_max"] = [0.0, 0.03, 0.002, 0.0]
    rows["clip_high_inner_max"] = [0.004, 0.0, 0.2, 0.009]
    rows["clip_inner_max"] = rows[["clip_low_inner_max", "clip_high_inner_max"]].max(axis=1)
    rows.to_parquet(out / "lcm_24m_dev.parquet")
    text = rep.build("24m", "development", out, "", study).read_text()
    print(text[:5000])
    assert text.splitlines()[0].endswith("indicative (four dates)")
    assert "**Clip flag (decision 5, `flag_clip`): 2 of 4 priced dates**" in text
    assert "(`flag_clip_low`, 1 date) or at the cap (`flag_clip_high`, 1 date)" in text
    assert "| 2019-09-03 | 0.0300 | 0.0000 | low |" in text
    assert "| 2024-08-05 | 0.0020 | 0.2000 | cap |" in text
    assert "No model S at 24m" in text
    tables = tables_of(text)
    summaries = [c for c, header in tables if header.startswith("| quantity |") or header.startswith("| gate |")]  # fmt: skip
    assert len(summaries) >= 15
    assert all("indicative (four dates)" in c for c in summaries)
    reported = [c for c in summaries if "priced dates without a clip flag (the reported summary): 2 dates" in c]  # fmt: skip
    everything = [c for c in summaries if "for information only: all priced dates, the 2 clip-flagged included: 4 dates" in c]  # fmt: skip
    no_low = [c for c in summaries if "for information only: priced dates without the low-side clip flag (1 removed" in c]  # fmt: skip
    assert len(reported) == len(everything) == len(no_low) == len(summaries) // 3
    # the reported forward is the mean of the two unflagged dates
    by = rows.set_index("date")
    want = float(by.loc[["2017-04-03", "2026-10-02"], "ratio"].mean())
    block = text[text.index("**The forward") :].splitlines()
    assert "the reported summary" in block[0]
    assert block[4].startswith("| **E_LC[D] / copula P_D** | 2 |")
    assert block[7].startswith(f"| E_LC[D] / E_CC[D] (paired) | 2 | {want:.4f} |")
    csv = pd.read_csv(out / "report_24m_dev_by_date.csv")
    assert list(csv["in_summary"]) == [True, False, False, True]
    assert list(csv["flags"]) == ["indicative", "clip_low;indicative", "clip_high;indicative", "indicative"]  # fmt: skip
    assert list(csv["status"]) == ["ok", "ok", "ok", "check"]
