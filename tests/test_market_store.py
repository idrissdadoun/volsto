"""M11 Part 3 tests (SPEC §18.5): the read API over the vendor store, on the synthetic
ORATS-format fixture (``tests/_orats_fixture.py``); one test reads the real sample and skips
when it is absent."""

from __future__ import annotations

import datetime as dt
import shlex
from pathlib import Path

import _orats_fixture as fx
import pandas as pd
import pytest

from volsto.data import cli, extract, orats, raw, roots, store
from volsto.data.roots import DataError
from volsto.market import store as api

SAMPLE = roots.REPO_ROOT / "data" / "orats_sample" / "ORATS_SMV_Strikes_20240103.zip"
DAYS = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4), dt.date(2024, 1, 5)]


@pytest.fixture()
def root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A store root holding the first three synthetic days (the fourth is verified in raw
    only) and the SPX extract; the roots are given by the environment."""
    raw_dir = tmp_path / "raw" / "orats"
    for day in DAYS[:3]:
        fx.write_day(raw_dir, day)
    fx.write_calendar(tmp_path / "calendar.csv", DAYS)
    raw.write_manifest(
        raw_dir, raw.verify_raw(raw_dir, calendar=tmp_path / "calendar.csv", workers=1).manifest()
    )
    store_dir = tmp_path / "store" / "orats"
    assert store.convert(raw_dir, store_dir, workers=1).clean
    extract.extract(store_dir, "SPX")
    fx.write_day(raw_dir, DAYS[3])
    raw.write_manifest(
        raw_dir, raw.verify_raw(raw_dir, calendar=tmp_path / "calendar.csv", workers=1).manifest()
    )
    monkeypatch.setenv(roots.ENV_RAW, str(tmp_path / "raw"))
    monkeypatch.setenv(roots.ENV_STORE, str(tmp_path / "store"))
    return tmp_path


def snapshot_of(path: Path) -> dict[str, tuple[int, int]]:
    return {str(p): (p.stat().st_size, p.stat().st_mtime_ns) for p in sorted(path.rglob("*"))}


def test_available_dates_and_load_chain(root: Path) -> None:
    before = snapshot_of(root)
    assert api.available_dates("orats") == ["2024-01-02", "2024-01-03", "2024-01-04"]
    assert api.available_dates("orats", store=root / "elsewhere") == []
    chain = api.load_chain("orats", "2024-01-03", "spx")
    want = fx.day_frame(DAYS[1])
    want = want[want["ticker"] == "SPX"]
    assert tuple(chain.columns) == orats.schema_columns(1)  # vendor-native, nothing derived
    assert len(chain) == len(want) and set(chain["ticker"]) == {"SPX"}
    assert sorted(chain["cOpra"]) == sorted(want["cOpra"])
    assert set(chain["trade_date"]) == {DAYS[1]} and chain["divRate"].dtype == float
    key = chain[["expirDate", "strike", "cOpra"]]
    assert key.equals(key.sort_values(["expirDate", "strike", "cOpra"]).reset_index(drop=True))
    # the third Friday holds both roots at one (expiry, strike)
    jan = chain[chain["expirDate"] == dt.date(2024, 1, 19)]
    assert set(jan["cOpra"].map(orats.opra_root)) == {"SPX", "SPXW"}
    # provenance for the importer
    entry = store.read_manifest(root / "store" / "orats")["files"]["2024-01-03"]
    assert chain.attrs["raw_sha256"] == entry["raw_sha256"] and chain.attrs["schema_version"] == 1
    assert chain.attrs["raw_file"] == "ORATS_SMV_Strikes_20240103.zip"
    assert (chain.attrs["vendor"], chain.attrs["ticker"]) == ("orats", "SPX")
    assert api.chain_entry("orats", DAYS[1])["rows"] == entry["rows"]
    # a date object and a column selection
    few = api.load_chain("orats", DAYS[1], "AAPL", columns=["strike", "cBidPx", "pAskPx"])
    assert list(few.columns) == ["strike", "cBidPx", "pAskPx"] and len(few) > 0
    with pytest.raises(DataError, match=r"unknown columns \['mid'\]"):
        api.load_chain("orats", DAYS[1], "AAPL", columns=["strike", "mid"])
    with pytest.raises(DataError, match="unknown vendor 'hdn'"):
        api.available_dates("hdn")
    assert snapshot_of(root) == before, "the read API wrote something"


def test_missing_date_and_ticker_name_the_command(
    root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # verified in raw, not converted: the exact convert command
    with pytest.raises(api.StoreMissing) as info:
        api.load_chain("orats", "2024-01-05", "SPX")
    assert info.value.command == "volsto-data convert --vendor orats"
    assert str(info.value).endswith(": volsto-data convert --vendor orats")
    # the printed command runs as printed and produces the date
    assert cli.main([*shlex.split(info.value.command)[1:], "--workers", "1"]) == 0
    capsys.readouterr()
    assert len(api.load_chain("orats", "2024-01-05", "SPX")) > 0
    # no raw file either: fetch, verify-raw, convert
    with pytest.raises(api.StoreMissing) as info:
        api.load_chain("orats", "2023-06-01", "SPX")
    cmd = info.value.command or ""
    assert cmd.startswith("volsto-data fetch --vendor orats --profile <aws profile>")
    assert cmd.endswith(
        "volsto-data verify-raw --vendor orats && volsto-data convert --vendor orats"
    )
    # a ticker the vendor's file does not list: no command, and the message says so
    with pytest.raises(api.StoreMissing, match="no row for ticker 'NOPE'") as info:
        api.load_chain("orats", "2024-01-03", "NOPE")
    assert info.value.command is None and "no command produces it" in str(info.value)
    # a file the manifest names but the disk lost
    store.day_path(root / "store" / "orats", "2024-01-02").unlink()
    with pytest.raises(api.StoreMissing) as info:
        api.load_chain("orats", "2024-01-02", "SPX")
    assert info.value.command == "volsto-data verify --vendor orats"
    with pytest.raises(ValueError):
        api.load_chain("orats", "1/3/2024", "SPX")
    # an explicit store: the command carries it
    other = root / "other store"
    with pytest.raises(api.StoreMissing) as info:
        api.load_chain("orats", "2024-01-03", "SPX", store=other)
    assert f"--store {shlex.quote(str(other))} convert --vendor orats" in (info.value.command or "")


def test_load_range_reads_the_extract(root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    before = snapshot_of(root)
    rng = api.load_range("orats", "SPX", "2024-01-03", dt.date(2024, 1, 4))
    assert tuple(rng.columns) == orats.schema_columns(1)
    assert sorted(set(rng["trade_date"])) == DAYS[1:3] and set(rng["ticker"]) == {"SPX"}
    parts = [api.load_chain("orats", d, "SPX") for d in DAYS[1:3]]
    pd.testing.assert_frame_equal(rng, pd.concat(parts, ignore_index=True), check_exact=True)
    assert rng.attrs["store_digest"] == store.store_digest(
        store.read_manifest(root / "store" / "orats")
    )
    whole = api.load_range("orats", "SPX", "2000-01-01", "2030-01-01", ["trade_date", "strike"])
    assert (
        list(whole.columns) == ["trade_date", "strike"]
        and sorted(set(whole["trade_date"])) == DAYS[:3]
    )
    assert snapshot_of(root) == before, "the read API wrote something"
    # no extract for the ticker: the exact extract command, which runs as printed
    with pytest.raises(api.StoreMissing) as info:
        api.load_range("orats", "aapl", "2024-01-02", "2024-01-04")
    assert info.value.command == "volsto-data extract --vendor orats --tickers AAPL"
    assert cli.main(shlex.split(info.value.command)[1:]) == 0
    capsys.readouterr()
    assert set(api.load_range("orats", "AAPL", "2024-01-02", "2024-01-04")["ticker"]) == {"AAPL"}
    # the store moved on: the SPX extract is stale and is not read
    assert cli.main(["convert", "--vendor", "orats", "--workers", "1"]) == 0
    capsys.readouterr()
    with pytest.raises(api.StoreMissing, match="another state of the store") as info:
        api.load_range("orats", "SPX", "2024-01-03", "2024-01-04")
    assert info.value.command == "volsto-data extract --vendor orats --tickers SPX"
    assert cli.main(shlex.split(info.value.command)[1:]) == 0
    capsys.readouterr()
    assert sorted(
        set(api.load_range("orats", "SPX", "2024-01-05", "2024-01-05")["trade_date"])
    ) == [DAYS[3]]
    # a range the extract does not reach, and a reversed one
    with pytest.raises(api.StoreMissing, match="its extract covers 2024-01-02 to 2024-01-05"):
        api.load_range("orats", "SPX", "2023-01-01", "2023-12-31")
    with pytest.raises(DataError, match="is after end"):
        api.load_range("orats", "SPX", "2024-01-04", "2024-01-03")


def test_read_api_does_not_import_the_writers() -> None:
    """Nothing here downloads, converts or fits: the module's own imports hold no fetch, no
    importer and no calibration."""
    import ast

    tree = ast.parse(Path(api.__file__).read_text())
    imported = {
        (n.module or "") + "." + a.name
        for n in ast.walk(tree)
        if isinstance(n, ast.ImportFrom)
        for a in n.names
    } | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    banned = ("fetch", "import_hdn", "calibration", "duckdb", "subprocess")
    assert not [name for name in imported if any(b in name for b in banned)], imported
    called = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert not called & {
        "convert",
        "extract",
        "verify_raw",
        "write_manifest",
        "write_parquet",
    }, called


def test_read_api_on_the_sample(tmp_path: Path) -> None:
    """The real sample day through the store and the read API (Part 0's SPX counts)."""
    if not SAMPLE.exists():
        pytest.skip("the ORATS one-day sample is absent (data/orats_sample)")
    raw_dir = tmp_path / "raw" / "orats"
    raw_dir.mkdir(parents=True)
    (raw_dir / SAMPLE.name).symlink_to(SAMPLE)
    raw.write_manifest(raw_dir, raw.verify_raw(raw_dir, workers=1).manifest())
    assert store.convert(raw_dir, tmp_path / "store" / "orats", workers=1).clean
    extract.extract(tmp_path / "store" / "orats", "SPX")
    spx = api.load_chain("orats", "2024-01-03", "SPX", store=tmp_path / "store")
    assert len(spx) == 10_519 and spx.attrs["raw_sha256"].startswith("e04a3731")
    roots_ = spx["cOpra"].map(orats.opra_root).value_counts().to_dict()
    assert roots_ == {"SPXW": 7241, "SPX": 3278}
    assert int(spx.duplicated(["expirDate", "strike"]).sum()) == 1_590
    rng = api.load_range("orats", "SPX", "2024-01-03", "2024-01-03", store=tmp_path / "store")
    pd.testing.assert_frame_equal(rng, spx, check_exact=True)
