"""M8b raw-slice discriminator (SPEC §15 Part 2, "Raw-slice discriminator (M8b)"): the local
quadratic recovers a synthetic slice's ATM vol / skew exactly and the pillar interpolation is
linear in ``T`` (skew) and in total variance (ATM vol); ``raw_pillar_frame`` on the first three
sample days builds a valid ``SurfaceHistory`` frame; the verdict rule on synthetic tables.  Fast,
no leverage cache; the sample-data test skips when the HDN sample is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volsto.calibration.history import SurfaceHistory, hdn_available_dates
from volsto.calibration.raw_history import (
    RAW_PILLARS,
    SliceFit,
    fit_slice_quadratic,
    interpolate_pillars,
    raw_pillar_frame,
    verdict_from_table,
)

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "hdn_sample" / "options_sample_2022H2"
SSVI_HISTORY = ROOT / "outputs" / "m7" / "hdn_history_ssvi.csv"


def test_local_quadratic_and_pillar_interpolation() -> None:
    k = np.linspace(-0.1, 0.1, 11)
    a, b, c = 0.21, -0.45, 1.3
    fa, fb, fc = fit_slice_quadratic(k, a + b * k + c * k * k)
    assert (fa, fb, fc) == pytest.approx((a, b, c), abs=1e-12)
    with pytest.raises(ValueError):
        fit_slice_quadratic(k[:4], k[:4])
    # two slices at 0.2y / 0.6y: the 0.5y pillar is linear in T for the skew, linear in a² T
    # for the ATM vol; 0.1y and 1y lie outside the bracket
    fits = [SliceFit(0.2, 0.20, -0.5, 0.0, 10), SliceFit(0.6, 0.25, -0.3, 0.0, 10)]
    atm, skew = interpolate_pillars(fits, (0.1, 0.5, 1.0))
    assert np.isnan(atm[[0, 2]]).all() and np.isnan(skew[[0, 2]]).all()
    assert skew[1] == pytest.approx(-0.5 + (-0.3 + 0.5) * 0.75)
    w = 0.2**2 * 0.2 + (0.25**2 * 0.6 - 0.2**2 * 0.2) * 0.75
    assert atm[1] == pytest.approx(np.sqrt(w / 0.5))
    # an exact slice maturity returns the slice's own values
    atm_e, skew_e = interpolate_pillars(fits, (0.6,))
    assert (atm_e[0], skew_e[0]) == pytest.approx((0.25, -0.3))


@pytest.mark.skipif(not SAMPLE.exists(), reason="HDN sample not present")
def test_raw_pillar_frame_first_three_days() -> None:
    dates = hdn_available_dates(SAMPLE)[:3]
    ssvi = SSVI_HISTORY if SSVI_HISTORY.exists() else None
    frame, diag = raw_pillar_frame(SAMPLE, dates, ssvi_history=ssvi)
    assert list(diag["date"]) == dates and diag["kept"].all()
    assert (diag["n_slices"] >= 10).all()
    assert len(frame) == 4 * 3 and set(frame["date"]) == set(dates)
    assert np.isfinite(frame[["vs_vol", "atm_vol", "skew", "ln_spot"]].to_numpy()).all()
    assert (frame.groupby("date")["ln_spot"].nunique() == 1).all()
    hist = SurfaceHistory(frame)
    assert hist.n_dates == 3 and hist.pillars.tolist() == pytest.approx(list(RAW_PILLARS))
    # SPX 2022 H2: ATM vols 15-40%, negative skews per unit log-moneyness, decaying in T
    assert ((hist.atm_vol > 0.15) & (hist.atm_vol < 0.40)).all().all()
    assert (hist.skew < 0).all().all()
    assert (hist.skew.iloc[:, 0] < hist.skew.iloc[:, -1]).all()
    if ssvi is not None:
        ref = pd.read_csv(ssvi)
        ref = ref[ref["date"].isin(dates) & ref["T"].round(9).isin(np.round(RAW_PILLARS, 9))]
        ref = ref.sort_values(["date", "T"]).reset_index(drop=True)
        assert frame.sort_values(["date", "T"])["vs_vol"].to_numpy() == pytest.approx(
            ref["vs_vol"].to_numpy()
        )
        assert frame.attrs["vs_vol_from_ssvi"] == 12


def _table(ssr_raw: list[float], ssr_ssvi: list[float], se: float = 0.08) -> pd.DataFrame:
    T = list(RAW_PILLARS)
    raw = np.asarray(ssr_raw)
    sv = np.asarray(ssr_ssvi)
    se_diff = np.sqrt(2.0) * se
    return pd.DataFrame(
        {
            "T": T,
            "window": 60,
            "ssr_raw": raw,
            "se_raw": se,
            "ssr_ssvi": sv,
            "se_ssvi": se,
            "diff": raw - sv,
            "se_diff": se_diff,
            "z": (raw - sv) / se_diff,
        }
    )


def test_verdict_logic() -> None:
    agree = verdict_from_table(_table([0.80, 0.82, 0.81, 0.83], [0.81, 0.82, 0.82, 0.84]))
    assert agree.verdict == "real"
    disagree = verdict_from_table(_table([1.05, 0.82, 0.81, 0.83], [0.81, 0.82, 0.82, 0.84]))
    assert disagree.verdict == "surface artefact" and "disagree" in disagree.reason
    assert "T=0.0833" in disagree.reason
    not_below = verdict_from_table(_table([0.80, 0.82, 0.81, 0.90], [0.81, 0.82, 0.82, 0.92]))
    assert not_below.verdict == "surface artefact" and "not below 1" in not_below.reason
    # the gate reads the short window only: a long-window disagreement does not count
    long = _table([1.3] * 4, [0.8] * 4)
    long["window"] = 100
    tab = pd.concat([_table([0.80] * 4, [0.81] * 4), long], ignore_index=True)
    assert verdict_from_table(tab, 60).verdict == "real"
