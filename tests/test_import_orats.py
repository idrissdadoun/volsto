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

from volsto.data import orats, raw, roots, spx_settlement, store
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
    # a row without any OPRA symbol on an underlying that has no settlement rule (only SPX
    # has one, owner's instruction 2026-10-04) and a root outside the known ones
    raw_dir = root / "raw" / "orats"

    def edit(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.loc[df.index[df["ticker"] == "XSP"][0], ["cOpra", "pOpra"]] = ""
        aapl = df["ticker"] == "AAPL"
        df.loc[aapl, "cOpra"] = df.loc[aapl, "cOpra"].str.replace("AAPL", "AAPL1", n=1)
        return df

    fx.write_day(raw_dir, DAY, mutate=edit)
    raw.write_manifest(
        raw_dir, raw.verify_raw(raw_dir, calendar=root / "SPX.csv", workers=1).manifest()
    )
    assert store.convert(raw_dir, root / "store" / "orats", workers=1).converted == [ISO]
    fx.write_calendar(root / "XSP.csv", [DAY])
    with pytest.raises(DataError, match=r"1 of .* XSP rows without an OPRA symbol, and no"):
        io.load_day(ISO, "XSP", store=root / "store", closes=root / "XSP.csv")
    with pytest.raises(DataError, match=r"OPRA roots \['AAPL1'\], outside the known \['AAPL'\]"):
        io.load_day(ISO, "AAPL", store=root / "store", closes=root / "XSP.csv")
    with pytest.raises(ValueError, match="columns missing"):
        chainmod.validate_chain(pd.DataFrame({"contract": ["x"]}))


def _store_of(tmp: Path, mutate: object) -> Path:
    """A store holding the fixture day edited by ``mutate`` (an older layout)."""
    raw_dir = tmp / "raw" / "orats"
    fx.write_day(raw_dir / "2024", DAY, mutate=mutate)  # type: ignore[arg-type]
    fx.write_calendar(tmp / "SPX.csv", [DAY])
    raw.write_manifest(
        raw_dir, raw.verify_raw(raw_dir, calendar=tmp / "SPX.csv", workers=1).manifest()
    )
    assert not store.convert(raw_dir, tmp / "store" / "orats", workers=1).failed
    return tmp


def _saturday(df: pd.DataFrame, rows: pd.Series) -> pd.Series:
    """``expirDate`` with the standard expiries of ``rows`` dated on the Saturday, as the
    files before 2015 date them."""
    third = {fx.us_date(d): fx.us_date(d + dt.timedelta(days=1)) for d in fx.THIRD_FRIDAYS}
    return df["expirDate"].where(~rows, df["expirDate"].map(lambda x: third.get(x, x)))


def test_settlement_by_period_reproduces_the_opra_chain(root: Path, tmp_path: Path) -> None:
    """Owner's instruction 2026-10-04.  The same day told three ways gives the same chain
    (contracts, roots, maturities, quotes): with OPRA symbols (2021-05-28 on), and without
    them with the PM third-Friday series under ticker SPXPM and the standard expiries dated
    on the Saturday (2011-10-04 to 2017).  With only the PM series under ticker SPX and no
    SPXPM (2019-02-05 to 2021-05-27 for the near monthlies) the settlement of each monthly
    comes from the open-interest table of the store: PM gives the PM part of the chain, a
    ``DROP`` leaves the expiry out, a missing table or entry fails loudly.  A day on which
    nothing tells the two series apart is excluded: :class:`AmbiguousSettlement`."""
    want = load(root)
    cols = [c for c in want.columns if c not in ("yte",)]

    def spxpm_period(df: pd.DataFrame) -> pd.DataFrame:
        spx = df["ticker"] == "SPX"
        pm = df["cOpra"].map(orats.opra_root) == "SPXW"
        third = df["expirDate"].isin([fx.us_date(d) for d in fx.THIRD_FRIDAYS])
        df = df.assign(
            ticker=df["ticker"].where(~(spx & pm & third), orats.SPX_PM_TICKER),
            expirDate=_saturday(df, spx),
        )
        return df.drop(columns=["cOpra", "pOpra"])

    got = io.load_day(
        ISO, "SPX", store=_store_of(tmp_path / "a", spxpm_period) / "store", closes=root / "SPX.csv"
    )
    pd.testing.assert_frame_equal(got[cols], want[cols])
    assert got.attrs["schema_version"] == 2 and set(got.attrs["settled_by"]) == {
        "ticker",
        "calendar",
    }
    assert want.attrs["settled_by"] == {"opra": int((fx.day_frame(DAY)["ticker"] == "SPX").sum())}
    assert "settlement" not in io.source_provenance(want)
    assert (
        io.source_provenance(got)["settlement"]["vendor_rows_by_source"] == got.attrs["settled_by"]
    )

    def pm_only(df: pd.DataFrame) -> pd.DataFrame:
        am = (df["ticker"] == "SPX") & (df["cOpra"].map(orats.opra_root) == "SPX")
        return df[~am].drop(columns=["cOpra", "pOpra"])

    where = _store_of(tmp_path / "b", pm_only)
    with pytest.raises(DataError, match="no SPX settlement table"):
        io.load_day(ISO, "SPX", store=where / "store", closes=root / "SPX.csv")
    table = spx_settlement.table_path(where / "store" / "orats")
    monthly = [d.isoformat() for d in fx.THIRD_FRIDAYS]
    rows = [(ISO, monthly[0], "PM", "test"), (ISO, monthly[1], "PM", "test")]
    pd.DataFrame(rows, columns=["date", "expiry", "settlement", "evidence"]).to_csv(
        table, index=False
    )
    got = io.load_day(ISO, "SPX", store=where / "store", closes=root / "SPX.csv")
    pm_part = want[want["root"] == "SPXW"].reset_index(drop=True)
    pd.testing.assert_frame_equal(got[cols], pm_part[cols])
    assert set(got["root"]) == {"SPXW"} and set(got.attrs["settled_by"]) == {"calendar"}
    assert got.attrs["dropped_expiries"] == [] and not got.attrs["settlement_uncertain"]
    rows[1] = (ISO, monthly[1], "DROP", "test")
    pd.DataFrame(rows, columns=["date", "expiry", "settlement", "evidence"]).to_csv(
        table, index=False
    )
    got = io.load_day(ISO, "SPX", store=where / "store", closes=root / "SPX.csv")
    assert got.attrs["dropped_expiries"] == [monthly[1]]
    assert monthly[1] not in set(got["expiration"]) and monthly[0] in set(got["expiration"])
    assert io.source_provenance(got)["settlement"]["dropped_expiries"] == [monthly[1]]
    pd.DataFrame(rows[:1], columns=["date", "expiry", "settlement", "evidence"]).to_csv(
        table, index=False
    )
    with pytest.raises(DataError, match="settlement table has no entry"):
        io.load_day(ISO, "SPX", store=where / "store", closes=root / "SPX.csv")

    def no_marker(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=["cOpra", "pOpra"])  # both series under SPX, no symbol

    where = _store_of(tmp_path / "c", no_marker)
    with pytest.raises(io.AmbiguousSettlement, match=r"repeat a \(root, expiry, strike\)"):
        io.load_day(ISO, "SPX", store=where / "store", closes=root / "SPX.csv")


def test_settlement_rule_and_canonical_expiry() -> None:
    d = dt.date
    # the Saturday of an old standard expiry is its Friday; a Good Friday its Thursday
    assert orats.canonical_expiry(d(2012, 3, 17)) == (d(2012, 3, 16), True)
    assert orats.canonical_expiry(d(2016, 6, 17)) == (d(2016, 6, 17), True)
    assert orats.good_friday(2014) == d(2014, 4, 18) and orats.good_friday(2019) == d(2019, 4, 19)
    assert orats.canonical_expiry(d(2014, 4, 19)) == (d(2014, 4, 17), True)  # Saturday dated
    assert orats.canonical_expiry(d(2019, 4, 18)) == (d(2019, 4, 18), True)  # Thursday dated
    assert orats.canonical_expiry(d(2019, 4, 19)) == (d(2019, 4, 18), True)  # Friday dated
    assert orats.canonical_expiry(d(2012, 12, 31)) == (d(2012, 12, 31), False)  # a quarterly
    assert orats.canonical_expiry(d(2012, 3, 23)) == (d(2012, 3, 23), False)  # a weekly
    rule = orats.settlement_without_opra
    monthly, weekly = d(2012, 4, 21), d(2012, 3, 23)
    assert rule("SPXPM", monthly, d(2012, 3, 16), spxpm_listed=True) == "PM"
    assert rule("SPX", monthly, d(2012, 3, 16), spxpm_listed=True) == "AM"
    assert rule("SPX", weekly, d(2012, 3, 16), spxpm_listed=True) == "PM"
    assert rule("SPX", d(2007, 3, 17), d(2007, 1, 3), spxpm_listed=False) == "AM"  # before SPXPM
    assert rule("SPX", d(2011, 12, 17), d(2011, 10, 3), spxpm_listed=False) == "AM"
    # after SPXPM: what the open-interest table says, per (trade date, expiry)
    table = {("2019-01-03", "2019-03-15"): "AM", ("2019-03-01", "2019-03-15"): "PM"}
    assert rule("SPX", d(2019, 3, 15), d(2019, 1, 3), spxpm_listed=False, table=table) == "AM"
    assert rule("SPX", d(2019, 3, 15), d(2019, 3, 1), spxpm_listed=False, table=table) == "PM"
    assert rule("SPX", d(2019, 3, 15), d(2019, 3, 15), spxpm_listed=False) == "DROP"  # stale
    with pytest.raises(ValueError, match="needs the open-interest table"):
        rule("SPX", d(2019, 3, 15), d(2019, 1, 3), spxpm_listed=False)
    with pytest.raises(KeyError):
        rule("SPX", d(2019, 6, 21), d(2019, 1, 3), spxpm_listed=False, table=table)
    # the end-of-week expirations were AM-settled before 2010-12-01; the quarterlies PM
    assert rule("SPX", d(2007, 1, 5), d(2007, 1, 3), spxpm_listed=False) == "AM"
    assert rule("SPX", d(2010, 12, 3), d(2010, 11, 30), spxpm_listed=False) == "AM"
    assert rule("SPX", d(2010, 12, 10), d(2010, 12, 1), spxpm_listed=False) == "PM"
    assert rule("SPX", d(2010, 12, 31), d(2010, 11, 1), spxpm_listed=False) == "PM"  # Friday,
    assert orats.is_quarter_end(d(2010, 12, 31)) and orats.is_quarter_end(d(2009, 6, 30))  # Q end
    assert not orats.is_quarter_end(d(2010, 12, 24)) and not orats.is_quarter_end(d(2010, 1, 29))
    assert rule("SPX", d(2009, 6, 30), d(2009, 6, 1), spxpm_listed=False) == "PM"
    assert orats.settlement_uncertain(d(2010, 12, 15)) and not orats.settlement_uncertain(
        d(2011, 1, 3)
    )
    with pytest.raises(ValueError, match="not an SPX-family ticker"):
        rule("XSP", monthly, d(2012, 3, 16), spxpm_listed=False)
    assert orats.osi_symbol("SPXW", d(2012, 3, 16), -1, 1402.5) == "SPXW120316P01402500"
    assert orats.opra_root(orats.osi_symbol("SPX", d(2012, 4, 20), 1, 1400.0)) == "SPX"


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


def _oi(series: dict[str, list[float]], dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(series, index=dates)


def test_settlement_table_from_open_interest() -> None:
    """:func:`volsto.data.spx_settlement.classify` on synthetic open interest: a switch, an
    expiry born on the PM series, a relapse day, a day that cannot be told, growth."""
    nan = float("nan")
    dates = [f"2019-02-{d:02d}" for d in (1, 4, 5, 6, 7, 8, 11, 12)]
    table = spx_settlement.classify(
        _oi(
            {
                # the AM monthly switches on the third date and stays PM
                "2019-03-15": [3.0e6, 3.1e6, 1.5e5, 1.6e5, 9.0e5, 1.7e5, 1.7e5, 1.8e5],
                # listed after the first switch, never switches, expiry inside the horizon
                "2019-06-21": [nan, nan, nan, 100.0, 5.0e4, 900.0, 4.0e3, 3.0e4],
                # a far expiry: stays AM, with a one-day hole on the last-but-one date
                "2019-12-20": [1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 1.0e6, 2.0e3, 1.0e6],
            },
            dates,
        )
    )
    got = {(r.date, r.expiry): r.settlement for r in table.itertuples()}
    assert got[("2019-02-01", "2019-03-15")] == "AM" and got[("2019-02-04", "2019-03-15")] == "AM"
    assert all(got[(d, "2019-03-15")] == "PM" for d in ("2019-02-05", "2019-02-06", "2019-02-08"))
    # 02-07: both PM-state expiries jump up and come back the next day: a relapse to AM
    assert got[("2019-02-07", "2019-03-15")] == "AM" and got[("2019-02-07", "2019-06-21")] == "AM"
    assert got[("2019-02-06", "2019-06-21")] == "PM" and got[("2019-02-12", "2019-06-21")] == "PM"
    assert got[("2019-02-07", "2019-12-20")] == "AM"  # untouched by the relapse
    assert got[("2019-02-11", "2019-12-20")] == "DROP"  # an AM series collapsing for one day
    assert got[("2019-02-12", "2019-12-20")] == "AM"
    info = spx_settlement.summary(table)
    assert len(info["switches"]) == 1 and "2019-02-05" in info["switches"][0]
    assert info["relapse_days"] == ["2019-02-07"] and info["drop_days"] == ["2019-02-11"]
    assert ("2019-02-01", "2019-06-21") not in got  # not listed that day


def test_settlement_table_of_the_archive() -> None:
    """The table built from the archive (skips without the store): the switches from
    2019-02-05, the six relapse days, the days that cannot be told."""
    path = spx_settlement.table_path(roots.DataRoots.resolve().store_dir("orats"))
    if not path.exists():
        pytest.skip(f"{path} absent (scripts/orats_spx_settlement.py)")
    table = pd.read_csv(path, dtype={"date": str, "expiry": str}).fillna({"evidence": ""})
    info = spx_settlement.summary(table)
    assert (table["date"].min(), table["date"].max()) == ("2017-05-10", "2021-05-27")
    assert len(info["switches"]) == 29
    assert min(s.split("switch on ")[1][:10] for s in info["switches"]) == "2019-02-05"
    assert info["relapse_days"] == [
        "2019-05-14",
        "2020-06-04",
        "2020-06-23",
        "2020-07-06",
        "2020-07-15",
        "2021-04-13",
    ]
    assert info["drop_days"] == ["2018-12-10", "2020-03-09", "2021-02-25"]
    before = table[table["date"] < "2019-02-05"]
    assert set(before["settlement"]) <= {"AM", "DROP"}  # the AM monthly only until the switch
