"""M3b tests (SPEC §13): HistoricalData.net importer on the sample day 2022-09-15 (SPX).

Skipped when ``./data/hdn_sample/options_sample_2022H2`` is absent (the sample is licensed and
git-ignored).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volsto.config import MarketConfig, SSVIConfig, load_yaml
from volsto.market import ESSVISurface, SSVISurface, implied_vol
from volsto.market.import_hdn import (
    HDN_COLUMNS,
    HdnFilters,
    implied_forwards,
    import_day,
    load_day,
    load_manifest,
    main,
    write_snapshot,
)
from volsto.market.loaders import load_ssvi_surface

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "hdn_sample" / "options_sample_2022H2"
DAY = "2022-09-15"
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="HDN sample not present")


@pytest.fixture(scope="module")
def chain() -> pd.DataFrame:
    return load_day(
        SAMPLE / "day_by_date" / f"{DAY}_options.csv", "SPX", manifest=load_manifest(SAMPLE)
    )


@pytest.fixture(scope="module")
def pipeline():
    return import_day(SAMPLE, DAY, "SPX")


def test_load_day_roots_quotes_and_time_convention(chain: pd.DataFrame) -> None:
    assert set(chain["root"].unique()) <= {"SPX", "SPXW"}
    assert (chain["bid"] > 0).all() and (chain["ask"] > 0).all()
    assert chain.attrs["quote_date"] == DAY and chain.attrs["spot"] > 3000
    # AM settlement loses one calendar day (README §5.2); PM does not
    am = chain[chain["settlement_time"] == "AM"].iloc[0]
    days_am = (pd.Timestamp(am["expiration"]) - pd.Timestamp(DAY)).days
    assert am["T"] == pytest.approx((days_am - 1) / 365.0)
    pm = chain[chain["settlement_time"] == "PM"].iloc[0]
    days_pm = (pd.Timestamp(pm["expiration"]) - pd.Timestamp(DAY)).days
    assert pm["T"] == pytest.approx(days_pm / 365.0)
    # rate from the manifest curve, flat below the 1m tenor, ~2.8-4% in Sept 2022
    assert 0.02 < chain["r"].min() < chain["r"].max() < 0.05
    # SPX (AM) and SPXW (PM) share expiration dates but are separate slices
    assert chain.groupby("expiration")["expiry"].nunique().max() == 2


def test_implied_forward_agrees_with_vendor_parity_forward(chain: pd.DataFrame) -> None:
    """Regression forward vs the vendor's closest-strike parity forward (README §5.4) within a few
    bp up to 1y; the residual grows with T because the vendor discounts at the Treasury rate while
    the regression uses the market-implied discount factor."""
    fwds = implied_forwards(chain, max_years=2.0)
    checked = 0
    for fe in fwds.values():
        if fe.T > 1.0 or fe.n_pairs < 5:
            continue
        assert abs(fe.forward / fe.vendor_style_forward - 1.0) < 3e-4, fe  # 3 bp
        assert 0.0 < fe.forward_stderr < 0.5, fe
        checked += 1
    assert checked >= 20
    # implied discount within 1.5% (absolute rate within ~1.5%) of the Treasury discount at 1y
    one_y = min(fwds.values(), key=lambda fe: abs(fe.T - 1.0))
    assert (
        abs(one_y.implied_rate - float(chain.loc[chain["expiry"] == one_y.expiry, "r"].iloc[0]))
        < 0.015
    )


def test_mid_implied_vols_reproduce_vendor_iv_on_flag0_rows(chain: pd.DataFrame) -> None:
    """Our Black-76 inversion with the implied forward reproduces the vendor's mid iv (iv_flag 0,
    European SPX) to a few tenths of a vol point near the money for 1m-1y."""
    fwds = implied_forwards(chain, max_years=2.0)
    diffs = []
    for e, fe in fwds.items():
        if not 1 / 12 <= fe.T <= 1.0:
            continue
        g = chain[
            (chain["expiry"] == e)
            & (chain["iv_flag"] == 0)
            & (np.abs(np.log(chain["strike"] / fe.forward)) < 0.1)
        ]
        if g.empty:
            continue
        iv = implied_vol(
            g["mid"].to_numpy(float),
            fe.forward,
            g["strike"].to_numpy(float),
            fe.T,
            g["cp"].to_numpy(int),
            fe.discount,
        )
        diffs.append(np.abs(iv - g["iv"].to_numpy(float)))
    d = np.concatenate(diffs)
    assert np.median(d) < 0.001 and np.quantile(d, 0.9) < 0.005, (np.median(d), np.quantile(d, 0.9))


def test_grid_surface_and_ssvi_fit(pipeline) -> None:
    _cfg, fit, points, _chain = pipeline
    assert points.table["expiry"].nunique() >= 25 and len(points.table) > 3000
    assert set(points.dropped) >= {"butterfly", "calendar", "spread", "min_bid"}
    assert isinstance(fit.surface, SSVISurface)
    fit.surface.check_no_arbitrage()  # raises on violation
    # SSVI: rms ≤ 0.3 vp inside ±20% from 3m to 2y (single power law cannot follow the 1-2m
    # weeklies of this high-vol day: reported, not asserted)
    assert fit.rms_error(2.0, 0.2, 0.25) < 0.30, fit.rms_error(2.0, 0.2, 0.25)
    assert fit.max_error(2.0, 0.2, 0.5) < 1.0
    assert -0.9 < fit.params["rho"] < -0.3 and 0.05 < fit.params["gamma"] <= 1.0
    # ATM term structure at the pillar tenors, in a plausible range for Sept 2022
    assert all(0.15 < v < 0.40 for v in fit.params["atm_vols"])


def test_essvi_fit_improves_near_money(pipeline) -> None:
    _cfg, fit, _points, _chain = pipeline
    _, fit_e, _, _ = import_day(SAMPLE, DAY, "SPX", essvi=True)
    assert isinstance(fit_e.surface, ESSVISurface)
    fit_e.surface.check_no_arbitrage()
    assert fit_e.rms_error(2.0, 0.2, 0.25) <= fit.rms_error(2.0, 0.2, 0.25) + 1e-9
    assert fit_e.rms_error(2.0, 0.2, 0.5) < 0.20
    assert len(fit_e.params["rho"]) == len(fit_e.params["atm_maturities"])


def test_snapshot_config_round_trip(pipeline, tmp_path: Path) -> None:
    cfg, fit, points, chain = pipeline
    p = write_snapshot(cfg, tmp_path / "spx.yaml")
    surface = load_ssvi_surface(p)
    market = load_yaml(p, MarketConfig, section="market")
    ssvi = load_yaml(p, SSVIConfig, section="ssvi")
    assert market.spot == pytest.approx(chain.attrs["spot"])
    assert ssvi.rho == pytest.approx(fit.params["rho"])
    np.testing.assert_allclose(surface.atm_vol(1.0), fit.surface.atm_vol(1.0), rtol=1e-9)
    prov = cfg["provenance"]
    assert prov["vendor"] == "historicaldata.net" and prov["file_sha256"] == prov["manifest_sha256"]
    assert prov["filters"]["min_bid"] == HdnFilters().min_bid and prov["n_points"] == len(
        points.table
    )


def test_essvi_snapshot_round_trip(tmp_path: Path) -> None:
    cfg, fit, _, _ = import_day(SAMPLE, DAY, "SPX", essvi=True)
    p = write_snapshot(cfg, tmp_path / "spx_essvi.yaml")
    surface = load_ssvi_surface(p)
    assert isinstance(surface, ESSVISurface)
    np.testing.assert_allclose(
        surface.implied_vol_k([-0.1, 0.0, 0.1], 1.0),
        fit.surface.implied_vol_k([-0.1, 0.0, 0.1], 1.0),
        rtol=1e-9,
    )
    assert cfg["essvi"]["rhos"] == fit.params["rho"]


def test_cli_writes_config(tmp_path: Path) -> None:
    rc = main(
        [
            "--vendor",
            "hdn",
            "--date",
            DAY,
            "--underlying",
            "SPX",
            "--root",
            str(SAMPLE),
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = tmp_path / f"spx_{DAY}.yaml"
    assert out.exists()
    assert load_ssvi_surface(out).atm_vol(0.5) > 0.1


def test_yfinance_capture_layout(tmp_path: Path) -> None:
    """The capture script's row builder emits the 34-column layout the importer reads."""
    spec = importlib.util.spec_from_file_location(
        "capture_yfinance", ROOT / "scripts" / "capture_yfinance.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["capture_yfinance"] = mod
    spec.loader.exec_module(mod)
    calls = pd.DataFrame(
        {
            "contractSymbol": ["SPXW230120C04000000", "SPX230120C04100000"],
            "strike": [4000.0, 4100.0],
            "bid": [10.0, 5.0],
            "ask": [10.5, 5.5],
            "volume": [1, 2],
            "openInterest": [10, 20],
            "lastPrice": [10.2, 5.1],
        }
    )
    rows = mod.build_rows(
        calls,
        underlying="SPX",
        expiration="2023-01-20",
        quote_date="2022-12-30",
        spot=3850.0,
        opt_type="call",
    )
    assert list(rows.columns) == list(HDN_COLUMNS)
    assert rows["settlement_time"].tolist() == ["PM", "AM"]  # SPXW weekly vs SPX third Friday
    assert (rows["iv_flag"] == 7).all()
    path = mod.write_day(rows, tmp_path, "2022-12-30", [4.1, 4.4, 4.7, 4.4, 4.0, 3.9, 4.0])
    man = load_manifest(tmp_path)
    assert man["rates"]["2022-12-30"][0] == 4.1 and man["files"][0]["rows"] == 2
    day = load_day(path, "SPX", manifest=man)
    assert len(day) == 2 and set(day["root"]) == {"SPX", "SPXW"} and (day["T"] > 0).all()
