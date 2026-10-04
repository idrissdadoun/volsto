"""M11 Part 4 tests (SPEC §18.6): the canonical chain and the ORATS importer for SPX.

The plumbing runs on the synthetic ORATS-format fixture (``tests/_orats_fixture.py``: Black-76
prices on a known forward, discount and smile, with the SPX and SPXW roots sharing the third
Fridays) converted into a temporary store.  The test on the real sample day skips when
``data/orats_sample/`` is absent and asserts the structure and the tolerances the owner agreed on
2026-10-04 (the named constants below, each with its measured value); every other measured
number is printed and reported in SPEC §18.6.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import _orats_fixture as fx
import numpy as np
import pandas as pd
import pytest
import yaml

from volsto.data import orats, raw, roots, store
from volsto.data.roots import DataError
from volsto.market import chain as chainmod
from volsto.market import import_hdn as ih
from volsto.market import import_orats as io
from volsto.market.loaders import load_sabrw_fits, load_ssvi_surface
from volsto.market.store import StoreMissing

SAMPLE = roots.REPO_ROOT / "data" / "orats_sample" / "ORATS_SMV_Strikes_20240103.zip"
HDN_SAMPLE = roots.REPO_ROOT / "data" / "hdn_sample" / "options_sample_2022H2"
DAY = dt.date(2024, 1, 3)

# Tolerances asserted on the real sample day 2024-01-03 (owner's decision F, 2026-10-04), each
# with the value measured that day beside it (SPEC §18.6).  Everything else is printed.
#: |parity forward / (stkPx · exp(iRate · T)) − 1| over the slices up to one year, in bp:
FORWARD_CROSSCHECK_MEDIAN_BP = 3.0  # measured: median 1.05 to 6 months, 1.7 from 6 to 12
FORWARD_CROSSCHECK_MAX_BP = 15.0  # measured: 10.0 to 6 months, 10.7 from 6 to 12
#: eSSVI RMS residual inside |k| <= 0.2, in vol points:
ESSVI_RMS_3M_2Y_VP = 0.25  # measured 0.188
ESSVI_RMS_6M_2Y_VP = 0.20  # measured 0.154
ISO = DAY.isoformat()


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    """A temporary store holding the synthetic day, and its close history."""
    raw_dir = tmp_path / "raw" / "orats"
    fx.write_day(raw_dir, DAY)
    fx.write_calendar(tmp_path / "SPX.csv", [DAY])  # close 4700.0, the fixture's spot
    raw.write_manifest(
        raw_dir, raw.verify_raw(raw_dir, calendar=tmp_path / "SPX.csv", workers=1).manifest()
    )
    assert store.convert(raw_dir, tmp_path / "store" / "orats", workers=1).clean
    return tmp_path


def load(root: Path) -> pd.DataFrame:
    return io.load_day(ISO, "SPX", store=root / "store", closes=root / "SPX.csv")


def test_loader_returns_the_canonical_chain(root: Path) -> None:
    chain = load(root)
    assert chainmod.validate_chain(chain) is chain
    frame = fx.day_frame(DAY)
    spx = frame[frame["ticker"] == "SPX"]
    # one row per contract: the call and the put of a vendor row melt apart, and a contract is
    # kept only when its own bid and ask are both positive
    calls = spx[(spx["cBidPx"] > 0) & (spx["cAskPx"] > 0)]
    puts = spx[(spx["pBidPx"] > 0) & (spx["pAskPx"] > 0)]
    assert int((chain["cp"] == 1).sum()) == len(calls) and int((chain["cp"] == -1).sum()) == len(
        puts
    )
    assert len(puts) < len(spx), "the fixture has zero put bids: the filter is exercised"
    assert set(chain[chain["cp"] == 1]["contract"]) == set(calls["cOpra"])
    assert set(chain[chain["cp"] == -1]["contract"]) == set(puts["pOpra"])
    # the root comes from the OPRA symbol; slices are (root, expiration)
    assert set(chain["root"]) == {"SPX", "SPXW"}
    assert (chain["root"] == chain["contract"].map(orats.opra_root)).all()
    assert chain["expiry"].nunique() == sum(1 for t, *_ in fx.SLICES if t == "SPX")
    assert {"SPX|2024-01-19", "SPXW|2024-01-19"} <= set(chain["expiry"])
    # decision 1: calendar days / 365, one day less for the AM root; yte is a cross-check
    T = chain.groupby("expiry")["T"].first()
    assert T["SPXW|2024-01-19"] == 16 / 365 and T["SPX|2024-01-19"] == 15 / 365
    assert T["SPX|2024-12-20"] == (dt.date(2024, 12, 20) - DAY).days / 365 - 1 / 365
    pm = chain[chain["root"] == "SPXW"]
    assert np.abs(pm["yte"] - pm["T"]).max() < 5e-6  # the vendor's rounding to five decimals
    am = chain[chain["root"] == "SPX"]
    assert np.allclose(am["yte"] - am["T"], 1 / 365, atol=5e-6)
    # decision 2: the vendor's one-sided vols are never inputs; its mid vol is a cross-check
    assert chain["iv_bid"].isna().all() and chain["iv_ask"].isna().all()
    assert chain["iv"].notna().all()
    # decision 4: the close is the official one, never spot_px; decision 6: the prior is iRate
    assert (chain["underlying_close"] == 4700.0).all() and "spot_px" not in chain.columns
    assert (chain["r"] == fx.RATE).all() and np.allclose(chain["df"], np.exp(-fx.RATE * chain["T"]))
    a = chain.attrs
    assert (a["quote_date"], a["underlying"], a["vendor"]) == (ISO, "SPX", "orats")
    assert a["spot"] == 4700.0 and a["spot_async_check"] is False
    assert a["file"] == fx.FILE_NAME.format(DAY) and a["schema_version"] == 1
    assert a["raw_sha256"] == raw.sha256_file(root / "raw" / "orats" / a["file"])
    assert a["rate_tenors"] == sorted(set(chain["T"])) and set(a["rate_zeros"]) == {fx.RATE}
    assert (chain["quote_date"] == ISO).all() and chain["mid"].gt(0).all()


def test_loader_fails_loudly(root: Path) -> None:
    with pytest.raises(StoreMissing, match="volsto-data"):
        io.load_day("2024-01-04", "SPX", store=root / "store", closes=root / "SPX.csv")
    with pytest.raises(DataError, match="has no close for 2024-01-03"):
        fx.write_calendar(root / "other.csv", [dt.date(2024, 1, 2)])
        io.load_day(ISO, "SPX", store=root / "store", closes=root / "other.csv")
    with pytest.raises(DataError, match="no close history"):
        io.load_day(ISO, "SPX", store=root / "store", closes=root / "absent.csv")
    # decision 8: a ticker row without its OPRA symbol (the converter refuses SPX; XSP shows
    # the loader's own check) and a root outside the known ones
    raw_dir = root / "raw" / "orats"

    def edit(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.loc[df.index[df["ticker"] == "XSP"][0], "pOpra"] = ""
        aapl = df["ticker"] == "AAPL"
        df.loc[aapl, "cOpra"] = df.loc[aapl, "cOpra"].str.replace("AAPL", "AAPL1", n=1)
        return df

    fx.write_day(raw_dir, DAY, mutate=edit)
    raw.write_manifest(
        raw_dir, raw.verify_raw(raw_dir, calendar=root / "SPX.csv", workers=1).manifest()
    )
    assert store.convert(raw_dir, root / "store" / "orats", workers=1).converted == [ISO]
    fx.write_calendar(root / "XSP.csv", [DAY])
    with pytest.raises(DataError, match=r"1 of .* XSP rows without an OPRA symbol"):
        io.load_day(ISO, "XSP", store=root / "store", closes=root / "XSP.csv")
    with pytest.raises(DataError, match=r"OPRA roots \['AAPL1'\], outside the known \['AAPL'\]"):
        io.load_day(ISO, "AAPL", store=root / "store", closes=root / "XSP.csv")
    with pytest.raises(ValueError, match="columns missing"):
        chainmod.validate_chain(pd.DataFrame({"contract": ["x"]}))


def test_parity_recovers_the_fixture_market(root: Path) -> None:
    """The vendor-independent steps on an ORATS chain: the parity regression recovers the
    forward and discount factor the fixture priced with, per (root, expiry) slice — the AM
    slice with its own, shorter T — and the stkPx cross-check is in basis points."""
    chain = load(root)
    fwds = ih.implied_forwards(chain, max_years=3.0, band=0.10)
    assert set(fwds) == set(chain["expiry"])
    worst_f = worst_df = 0.0
    for fe in fwds.values():
        want_f = 4700.0 * np.exp((fx.RATE - fx.CARRY) * fe.T)
        worst_f = max(worst_f, abs(fe.forward / want_f - 1.0))
        worst_df = max(worst_df, abs(fe.discount / np.exp(-fx.RATE * fe.T) - 1.0))
    print(
        f"\nfixture: forward off by at most {1e4 * worst_f:.3f} bp, discount {1e4 * worst_df:.3f} bp"
    )
    # prices are rounded to cents on a 4700 index: the regression is exact to a fraction of a bp
    assert worst_f < 1e-5 and worst_df < 1e-4
    am, pm = fwds["SPX|2024-01-19"], fwds["SPXW|2024-01-19"]
    assert am.T == 15 / 365 and pm.T == 16 / 365 and am.forward < pm.forward
    xc = io.forward_crosscheck(chain, fwds)
    assert list(xc["expiry"]) == sorted(fwds, key=lambda k: fwds[k].T)
    assert {"bp", "bp_yte", "forward_se_bp", "yte_minus_T_days", "n_pairs"} <= set(xc.columns)
    pm_rows = xc[xc["expiry"].str.startswith("SPXW")]
    assert pm_rows["bp_yte"].abs().max() < 0.2  # the fixture's stkPx is F exp(-iRate yte)
    assert np.allclose(xc[xc["expiry"].str.startswith("SPX|")]["yte_minus_T_days"], 1.0, atol=2e-3)


def test_import_day_writes_a_snapshot_under_the_store(
    root: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv(roots.ENV_STORE, str(root / "store"))
    monkeypatch.setattr(io, "HISTORY_DIR", root)
    cfg, _fit, _points, chain = io.import_day(ISO, "SPX")
    prov = cfg["provenance"]
    assert prov["vendor"] == "orats" and prov["file_sha256"] == chain.attrs["raw_sha256"]
    assert prov["store_schema_version"] == 1 and "not verified" in prov["snapshot_time"]
    assert prov["importer_tag"] == ih.IMPORTER_TAG and prov["roots"] == ["SPX", "SPXW"]
    assert prov["close_source"] == "SPX.csv" and "iRate" in prov["prior_rate_curve"]
    assert list(prov)[:5] == ["vendor", "product", "file", "file_sha256", "store_schema_version"]
    # the market: the option-implied spot (the fixture's 4700) with the close kept beside it;
    # the offset is a measurement, never the asynchrony warning
    assert cfg["market"]["close"] == 4700.0
    assert abs(cfg["market"]["spot"] / 4700.0 - 1.0) < 1e-4
    assert prov["spot"]["asynchronous"] is False
    # the funding curve the options imply is the fixture's flat rate
    assert np.allclose(cfg["market"]["rate_curve"]["rates"], fx.RATE, atol=2e-3)
    assert cfg["essvi"]["rhos"] and "sabrw" in cfg
    # the CLI writes under the store, never under configs/
    assert io.snapshots_dir() == root / "store" / "orats" / "snapshots"
    with caplog.at_level("INFO"):
        assert ih.main(["--vendor", "orats", "--date", ISO, "--underlying", "SPX"]) == 0
    out = root / "store" / "orats" / "snapshots" / f"spx_{ISO}.yaml"
    assert out.is_file() and f"wrote {out}" in caplog.text
    assert not any("asynchrony" in r.getMessage() for r in caplog.records)
    doc = yaml.safe_load(out.read_text())
    assert doc["provenance"]["vendor"] == "orats"
    surface = load_ssvi_surface(out)
    assert load_sabrw_fits(out) is not None
    # the fitted surface reproduces the fixture's smile near the money (synthetic data: the
    # smile is smooth, so the fit is tight; printed)
    T = 0.5
    err = 100.0 * abs(float(surface.implied_vol_k(0.0, T)) - float(fx.smile(np.array([0.0]), T)[0]))
    print(f"\nfixture ATM vol at 6m: fitted against true, {err:.3f} vol points")
    assert err < 0.5
    with pytest.raises(StoreMissing):
        ih.main(["--vendor", "orats", "--date", "2024-01-04", "--underlying", "SPX"])


def test_hdn_loader_returns_the_canonical_chain() -> None:
    """The HDN loader produces the same frame (the one definition, volsto.market.chain)."""
    if not HDN_SAMPLE.exists():
        pytest.skip("the HDN sample is absent (data/hdn_sample/options_sample_2022H2)")
    chain = ih.load_day(HDN_SAMPLE / "day_by_date" / "2022-09-15_options.csv", "SPX")
    assert chainmod.validate_chain(chain) is chain
    assert "spot_async_check" not in chain.attrs  # the HDN check stays on (16:00 against 16:15)


def test_orats_sample_day_imports(tmp_path: Path) -> None:
    """The real sample day (2024-01-03) through the store and the importer: structure only;
    the measured numbers are printed and reported in SPEC §18.6."""
    if not SAMPLE.exists():
        pytest.skip("the ORATS one-day sample is absent (data/orats_sample)")
    closes = roots.REPO_ROOT / "data" / "history" / "SPX.csv"
    if not closes.exists():
        pytest.skip("the SPX close history is absent (data/history/SPX.csv)")
    raw_dir = tmp_path / "raw" / "orats"
    raw_dir.mkdir(parents=True)
    (raw_dir / SAMPLE.name).symlink_to(SAMPLE)
    raw.write_manifest(raw_dir, raw.verify_raw(raw_dir, workers=1).manifest())
    assert store.convert(raw_dir, tmp_path / "store" / "orats", workers=1).clean
    cfg, fit, points, chain = io.import_day("2024-01-03", "SPX", store=tmp_path / "store")
    assert set(chain["root"]) == {"SPX", "SPXW"} and chain.attrs["raw_sha256"].startswith(
        "e04a3731"
    )
    assert cfg["provenance"]["vendor"] == "orats" and cfg["provenance"]["fit"]["essvi"] is True
    assert cfg["provenance"]["spot"]["asynchronous"] is False
    # the forward cross-check up to one year (decision 5)
    xc = io.forward_crosscheck(chain, ih.implied_forwards(chain, max_years=3.0, band=0.10))
    year = xc[xc["T"] <= 1.0]["bp"].abs()
    assert len(year) >= 50
    assert year.median() <= FORWARD_CROSSCHECK_MEDIAN_BP, year.median()
    assert year.max() <= FORWARD_CROSSCHECK_MAX_BP, year.max()
    # the eSSVI residuals on the regions SPEC §13 reports
    rms_3m, rms_6m = fit.rms_error(2.0, 0.2, 0.25), fit.rms_error(2.0, 0.2, 0.5)
    assert rms_3m <= ESSVI_RMS_3M_2Y_VP and rms_6m <= ESSVI_RMS_6M_2Y_VP, (rms_3m, rms_6m)
    # the calendar certificate is proven: no fallback, the full margin on the full Dupire range
    cal = cfg["provenance"]["fit"]
    assert cal["calendar_fallback"] is None
    assert cal["calendar_floor"] == ih.DEFAULT_CALENDAR_REPAIR.margin
    assert cal["calendar_k_abs"] == ih.DEFAULT_CALENDAR_REPAIR.k_max
    assert cal["calendar_lower_bound"] >= cal["calendar_floor"]
    print(
        f"\nforward cross-check up to 1y: median {year.median():.2f} bp, max {year.max():.2f} bp "
        f"({len(year)} slices); eSSVI RMS 3m-2y {rms_3m:.3f}, 6m-2y {rms_6m:.3f} vp; calendar "
        f"lower bound {cal['calendar_lower_bound']:.6e}"
    )
    sp = points.spot
    print(
        f"\nORATS 2024-01-03: {len(points.table)} points, {points.table['expiry'].nunique()} "
        f"expiries; implied spot {sp.spot:.2f} against the close {sp.close:.2f} "
        f"({sp.offset_bp:+.1f} bp, {sp.z:.1f} se); eSSVI RMS 3m-2y |k|<=0.2 "
        f"{fit.rms_error(2.0, 0.2, 0.25):.3f} vp"
    )
