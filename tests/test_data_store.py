"""M11 Part 2b tests (SPEC §18.4): ``volsto-data convert``, ``verify``, ``extract`` and ``sql``.

Everything runs on the synthetic ORATS-format fixture (``tests/_orats_fixture.py``) in a
temporary directory; the one test on the real sample skips when ``data/orats_sample/`` is
absent, and the DuckDB test skips when DuckDB is not installed (it belongs to the ``data``
extra).  No size or time is asserted: both are printed.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

import _orats_fixture as fx
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

from volsto.data import cli, extract, orats, raw, roots, sql, store
from volsto.data.roots import DataError

SAMPLE = roots.REPO_ROOT / "data" / "orats_sample" / "ORATS_SMV_Strikes_20240103.zip"
DAYS = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4)]
ISO = [d.isoformat() for d in DAYS]


@pytest.fixture()
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    """``(raw dir, store dir)``: three synthetic days, verified (the raw manifest written)."""
    raw_dir = tmp_path / "raw" / "orats"
    for day in DAYS:
        fx.write_day(raw_dir, day)
    fx.write_calendar(tmp_path / "calendar.csv", DAYS)
    reverify(raw_dir)
    return raw_dir, tmp_path / "store" / "orats"


def reverify(raw_dir: Path) -> raw.RawReport:
    rep = raw.verify_raw(raw_dir, calendar=raw_dir.parents[1] / "calendar.csv", workers=1)
    raw.write_manifest(raw_dir, rep.manifest())
    return rep


def expected(raw_dir: Path, day: dt.date) -> pd.DataFrame:
    """What the store must hold for ``day``: the CSV inside the raw zip (the text is the
    vendor's value — the fixture writes eight significant digits), typed and sorted."""
    df = pd.read_csv(
        raw_dir / fx.FILE_NAME.format(day),
        dtype={"ticker": str, "cOpra": str, "pOpra": str, "divRate": float},
        keep_default_na=False,
        na_values={"spot_px": [""]},
        float_precision="round_trip",
    )
    for col in ("expirDate", "trade_date"):
        df[col] = pd.to_datetime(df[col], format=orats.DATE_FORMAT).dt.date
    return df.sort_values(list(store.SORT_KEY), kind="stable").reset_index(drop=True)


# ---------------------------------------------------------------------------------------------
# convert
# ---------------------------------------------------------------------------------------------


def test_convert_is_a_faithful_typed_sorted_copy(dirs: tuple[Path, Path]) -> None:
    raw_dir, store_dir = dirs
    rep = store.convert(raw_dir, store_dir, workers=1)
    assert rep.clean and rep.converted == ISO and rep.skipped == 0
    assert sorted(p.relative_to(store_dir).as_posix() for p in store_dir.rglob("*.parquet")) == [
        f"strikes/year=2024/{d}.parquet" for d in ISO
    ]
    assert not list(store_dir.rglob(".*tmp"))
    man = store.read_manifest(store_dir)
    assert man["layout"] == store.LAYOUT and sorted(man["files"]) == ISO
    raw_sha = {e["trade_date"]: e["sha256"] for e in raw.read_manifest(raw_dir)["files"]}  # type: ignore[index]
    for day, iso in zip(DAYS, ISO):
        path = store.day_path(store_dir, iso)
        entry = man["files"][iso]
        want = expected(raw_dir, day)
        # the manifest binds the file to its raw sha256, row count and schema version
        assert entry["raw_sha256"] == raw_sha[iso] and entry["schema_version"] == 1
        assert entry["rows"] == len(want) and entry["sha256"] == raw.sha256_file(path)
        assert entry["copra_duplicates"] == 0 and entry["copra_empty"] == 0
        tbl = pq.read_table(path)
        # every vendor column, original names, declared types, nothing derived
        assert tbl.schema.remove_metadata() == store.arrow_schema(1)
        assert tuple(tbl.column_names) == orats.schema_columns(1)
        assert tbl.schema.field("divRate").type == pa.float64()
        assert tbl.schema.field("expirDate").type == pa.date32()
        got = tbl.to_pandas()
        pd.testing.assert_frame_equal(got, want, check_dtype=False, check_exact=True)
        # sorted by the key, and the third Friday keeps both roots at one (expiry, strike)
        key = got[list(store.SORT_KEY)]
        assert key.equals(
            key.sort_values(list(store.SORT_KEY), kind="stable").reset_index(drop=True)
        )
        spx = got[got["ticker"] == "SPX"]
        jan = spx[spx["expirDate"] == dt.date(2024, 1, 19)]
        assert set(jan["cOpra"].map(orats.opra_root)) == {"SPX", "SPXW"}
        assert jan.duplicated(["ticker", "expirDate", "strike"]).sum() == len(jan) // 2
        # the layout the owner chose, and the provenance inside the file
        md = pq.read_metadata(path)
        col = md.row_group(0).column(3)
        assert col.compression == "ZSTD" and not any("DICTIONARY" in e for e in col.encodings)
        meta = {k.decode(): v.decode() for k, v in md.metadata.items()}
        assert meta["volsto.raw_sha256"] == raw_sha[iso] and meta["volsto.schema_version"] == "1"
        assert meta["volsto.trade_date"] == iso and meta["volsto.vendor"] == "orats"
    assert store.available_dates(store_dir) == ISO
    assert "clean" in store.format_convert(rep) and "peak memory" in store.format_convert(rep)


def test_convert_row_groups_prune_on_ticker(
    dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir, store_dir = dirs
    monkeypatch.setitem(store.LAYOUT, "row_group_size", 64)
    store.convert(raw_dir, store_dir, workers=1)
    md = pq.read_metadata(store.day_path(store_dir, ISO[0]))
    assert md.num_row_groups > 3
    holding = [
        i
        for i in range(md.num_row_groups)
        if md.row_group(i).column(0).statistics.min
        <= "AAPL"
        <= md.row_group(i).column(0).statistics.max
    ]
    assert 0 < len(holding) < md.num_row_groups  # the ticker's rows sit in a few groups only


def test_convert_is_idempotent_and_follows_the_raw_layer(dirs: tuple[Path, Path]) -> None:
    raw_dir, store_dir = dirs
    first = store.convert(raw_dir, store_dir, workers=2)
    assert first.clean and sorted(first.converted) == ISO
    before = {iso: store.day_path(store_dir, iso).stat().st_mtime_ns for iso in ISO}
    again = store.convert(raw_dir, store_dir, workers=2)
    assert again.converted == [] and again.skipped == 3 and again.clean
    assert before == {iso: store.day_path(store_dir, iso).stat().st_mtime_ns for iso in ISO}
    # a raw file replaced without verify-raw: refused, nothing written
    fx.write_day(raw_dir, DAYS[1], mutate=lambda d: d.iloc[:-5])
    with pytest.raises(DataError, match="run volsto-data verify-raw"):
        store.convert(raw_dir, store_dir, workers=1)
    reverify(raw_dir)
    rep = store.convert(raw_dir, store_dir, workers=1)
    assert rep.converted == [ISO[1]] and rep.skipped == 2
    assert (
        pq.read_metadata(store.day_path(store_dir, ISO[1])).num_rows
        == len(fx.day_frame(DAYS[1])) - 5
    )
    # a Parquet file removed is written again; --force rewrites everything
    store.day_path(store_dir, ISO[0]).unlink()
    assert store.convert(raw_dir, store_dir, workers=1).converted == [ISO[0]]
    assert sorted(store.convert(raw_dir, store_dir, workers=1, force=True).converted) == ISO
    # a file written under another layout is converted again
    man = store.read_manifest(store_dir)
    man["files"][ISO[2]]["layout_version"] = 0
    store.write_manifest(store_dir, man)
    assert store.convert(raw_dir, store_dir, workers=1).converted == [ISO[2]]
    with pytest.raises(DataError, match="no raw manifest"):
        store.convert(raw_dir.parent / "none", store_dir)


def test_convert_is_atomic(dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch) -> None:
    raw_dir, store_dir = dirs
    real = pq.write_table
    calls = {"n": 0}

    def dying(table: pa.Table, where: Path, **kw: object) -> None:
        calls["n"] += 1
        real(table, where, **kw)  # the temporary file is complete on disk ...
        if calls["n"] == 2:
            raise KeyboardInterrupt  # ... and the process dies before the rename

    monkeypatch.setattr(pq, "write_table", dying)
    with pytest.raises(KeyboardInterrupt):
        store.convert(raw_dir, store_dir, workers=1)
    on_disk = sorted(p.name for p in store_dir.rglob("*") if p.is_file())
    assert on_disk == [f"{ISO[0]}.parquet"], on_disk  # no partial file, no temporary left
    monkeypatch.setattr(pq, "write_table", real)
    rep = store.convert(raw_dir, store_dir, workers=1)  # resumes: no manifest was written
    assert sorted(rep.converted) == ISO and rep.clean
    assert store.verify_store(raw_dir, store_dir, workers=1).clean


def test_convert_fails_loudly(dirs: tuple[Path, Path]) -> None:
    raw_dir, store_dir = dirs

    def drift(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=["cMidIv"]).assign(newCol=1.0)

    def blank_spx(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.loc[df["ticker"] == "SPX", "cOpra"] = ""
        return df

    def iso_dates(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(expirDate="2024-01-19")

    fx.write_day(raw_dir, DAYS[0], mutate=drift)
    fx.write_day(raw_dir, DAYS[1], mutate=blank_spx)
    fx.write_day(raw_dir, DAYS[2], mutate=iso_dates)
    fx.write_day(raw_dir, dt.date(2024, 1, 5), mutate=lambda d: d.assign(trade_date="1/9/2024"))
    fx.write_day(raw_dir, dt.date(2024, 1, 8), mutate=lambda d: d.assign(cOi="many"))
    fx.write_day(raw_dir, dt.date(2024, 1, 9))
    reverify(raw_dir)
    rep = store.convert(raw_dir, store_dir, workers=1)
    # owner's instruction 2026-10-04: SPX rows without an OPRA symbol are converted
    assert sorted(rep.converted) == [ISO[1], "2024-01-09"] and not rep.clean
    blank = store.read_manifest(store_dir)["files"][ISO[1]]
    assert blank["copra_empty"] > 0 and blank["key_duplicates"] > 0  # SPX and SPXW share strikes
    why = {fx.FILE_NAME.format(d): rep.failed[fx.FILE_NAME.format(d)] for d in (DAYS[0], DAYS[2])}
    assert "schema drift" in why[fx.FILE_NAME.format(DAYS[0])]
    assert "missing ['cMidIv']" in why[fx.FILE_NAME.format(DAYS[0])]
    assert "unexpected ['newCol']" in why[fx.FILE_NAME.format(DAYS[0])]
    assert "ArrowInvalid" in why[fx.FILE_NAME.format(DAYS[2])]  # a date not in %m/%d/%Y
    assert "trade_date column" in rep.failed["ORATS_SMV_Strikes_20240105.zip"]
    assert "ArrowInvalid" in rep.failed["ORATS_SMV_Strikes_20240108.zip"]  # an int that is not
    assert sorted(store.read_manifest(store_dir)["files"]) == [ISO[1], "2024-01-09"]
    assert sorted(p.name for p in store_dir.rglob("*.parquet")) == [
        f"{ISO[1]}.parquet",
        "2024-01-09.parquet",
    ]
    text = store.format_convert(rep)
    assert text.count("FAILED") == 4 and "NOT CLEAN" in text
    # a raw file edited after verify-raw but with size and mtime kept: the hash catches it
    victim = raw_dir / fx.FILE_NAME.format(dt.date(2024, 1, 9))
    st = victim.stat()
    data = bytearray(victim.read_bytes())
    data[-1] ^= 0xFF
    victim.write_bytes(bytes(data))
    import os

    os.utime(victim, ns=(st.st_atime_ns, st.st_mtime_ns))
    rep = store.convert(raw_dir, store_dir, workers=1, force=True)
    assert "changed since verify-raw" in rep.failed[victim.name]
    # a failed file is vouched for by nothing
    assert sorted(store.read_manifest(store_dir)["files"]) == [ISO[1]]


def test_convert_reports_copra_violations_and_duplicate_dates(dirs: tuple[Path, Path]) -> None:
    raw_dir, store_dir = dirs

    def repeated(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        aapl = df.index[df["ticker"] == "AAPL"]
        df.loc[aapl[1], "cOpra"] = df.loc[aapl[0], "cOpra"]
        df.loc[aapl[2], "cOpra"] = ""
        return df

    fx.write_day(raw_dir, DAYS[0], mutate=repeated)
    fx.write_day(raw_dir / "again", DAYS[2])  # two raw files for one date
    reverify(raw_dir)
    rep = store.convert(raw_dir, store_dir, workers=1)
    assert sorted(rep.converted) == ISO[:2]  # the violating file is still a faithful copy
    assert "1 duplicate cOpra symbols" in rep.violations[ISO[0]] and not rep.clean
    entry = store.read_manifest(store_dir)["files"][ISO[0]]
    assert (entry["copra_duplicates"], entry["copra_empty"]) == (1, 1)
    assert sum("2 raw files for 2024-01-04" in v for v in rep.failed.values()) == 2
    later = store.convert(raw_dir, store_dir, workers=1)
    assert later.converted == [] and ISO[0] in later.violations  # still reported, never silent


def test_convert_checks_free_space_and_the_volume(
    dirs: tuple[Path, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw_dir, store_dir = dirs
    monkeypatch.setattr(roots, "free_bytes", lambda path: 1024)
    with pytest.raises(DataError, match="not enough free space for converting 3 files"):
        store.convert(raw_dir, store_dir, workers=1)
    assert not list(store_dir.rglob("*.parquet"))
    # a store on a volume that is not mounted is refused, not created on the internal disk
    volumes = tmp_path / "Volumes"
    (volumes / "Mounted").mkdir(parents=True)
    assert roots.ensure_dir(volumes / "Mounted" / "store", volumes_dir=volumes).is_dir()
    with pytest.raises(DataError, match="is not mounted"):
        roots.ensure_dir(volumes / "Unplugged" / "store" / "orats", volumes_dir=volumes)
    assert not (volumes / "Unplugged").exists()


# ---------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------


def test_verify_store(dirs: tuple[Path, Path]) -> None:
    raw_dir, store_dir = dirs
    with pytest.raises(DataError, match="no store manifest"):
        store.verify_store(raw_dir, store_dir)
    store.convert(raw_dir, store_dir, workers=1)
    rep = store.verify_store(raw_dir, store_dir, workers=2)
    assert rep.clean and rep.checked == 3
    assert rep.rows == sum(len(fx.day_frame(d)) for d in DAYS)
    assert "faithful copy" in store.format_verify(rep)

    # one value changed, one made null, one row dropped, one file orphaned, one missing
    def rewrite(iso: str, change: object) -> None:
        path = store.day_path(store_dir, iso)
        tbl = pq.read_table(path)
        pq.write_table(change(tbl), path)  # type: ignore[operator]

    def bump(tbl: pa.Table) -> pa.Table:
        col = tbl["cBidPx"].to_pylist()
        col[7] += 0.01
        return tbl.set_column(tbl.column_names.index("cBidPx"), "cBidPx", pa.array(col))

    def null_one(tbl: pa.Table) -> pa.Table:
        col = tbl["stkPx"].to_pylist()
        col[3] = None
        return tbl.set_column(tbl.column_names.index("stkPx"), "stkPx", pa.array(col, pa.float64()))

    rewrite(ISO[0], bump)
    rewrite(ISO[1], null_one)
    rewrite(ISO[2], lambda t: t.slice(1))
    orphan = store_dir / "strikes" / "year=2023" / "2023-12-29.parquet"
    orphan.parent.mkdir()
    orphan.write_bytes(store.day_path(store_dir, ISO[0]).read_bytes())
    fx.write_day(raw_dir, dt.date(2024, 1, 5))
    reverify(raw_dir)
    rep = store.verify_store(raw_dir, store_dir, workers=1)
    assert any(p.startswith("cBidPx: sum raw") for p in rep.problems[ISO[0]])
    assert any(p.startswith("stkPx: null count raw 0, Parquet 1") for p in rep.problems[ISO[1]])
    assert any(p.startswith("rows: raw") for p in rep.problems[ISO[2]])
    assert all("sha256 is not the manifest's" in rep.problems[i][0] for i in ISO)
    assert rep.orphans == ["strikes/year=2023/2023-12-29.parquet"]
    assert rep.not_converted == ["2024-01-05"] and not rep.clean
    text = store.format_verify(rep)
    assert "NOT CLEAN" in text and "raw dates not converted (1)" in text
    # two rows swapped: counts, nulls and sums agree, the tables do not
    store.convert(raw_dir, store_dir, workers=1, force=True)
    orphan.unlink()
    path = store.day_path(store_dir, ISO[0])
    tbl = pq.read_table(path)
    order = list(range(tbl.num_rows))
    order[0], order[1] = order[1], order[0]
    pq.write_table(tbl.take(order), path)
    rep = store.verify_store(raw_dir, store_dir, workers=1)
    assert rep.problems[ISO[0]][-1] == (
        "row counts, null counts and sums agree but the tables differ"
    )


# ---------------------------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------------------------


def test_extract_per_ticker(dirs: tuple[Path, Path]) -> None:
    raw_dir, store_dir = dirs
    with pytest.raises(DataError, match="is empty"):
        extract.extract(store_dir, "SPX")
    store.convert(raw_dir, store_dir, workers=1)
    rep = extract.extract(store_dir, "spx, AAPL,spx,NOPE")
    assert list(rep.tickers) == ["SPX", "AAPL"] and rep.missing == ["NOPE"]
    assert not extract.ticker_path(store_dir, "NOPE").exists()
    assert not list((store_dir / "by_ticker").glob(".*tmp"))
    for ticker in ("SPX", "AAPL"):
        frames = [expected(raw_dir, d) for d in DAYS]
        want = pd.concat([f[f["ticker"] == ticker] for f in frames], ignore_index=True)
        path = extract.ticker_path(store_dir, ticker)
        tbl = pq.read_table(path)
        assert tbl.schema.remove_metadata() == store.arrow_schema(1)
        pd.testing.assert_frame_equal(tbl.to_pandas(), want, check_dtype=False, check_exact=True)
        md = pq.read_metadata(path)
        assert md.num_row_groups == len(DAYS)  # one row group per trade date
        entry = rep.tickers[ticker]
        assert (entry["rows"], entry["n_dates"]) == (len(want), 3)
        assert (entry["first"], entry["last"]) == (ISO[0], ISO[-1])
        assert entry["sha256"] == raw.sha256_file(path)
        assert entry["store_digest"] == store.store_digest(store.read_manifest(store_dir))
        assert extract.extract_state(store_dir, ticker) == "current"
    # a date range prunes on the trade_date statistics
    one = pq.read_table(
        extract.ticker_path(store_dir, "SPX"), filters=[("trade_date", "=", DAYS[1])]
    )
    assert pc.unique(one["trade_date"]).to_pylist() == [DAYS[1]]
    # the store changes: the extract is stale until it is rebuilt
    assert extract.extract_state(store_dir, "XSP") == "absent"
    fx.write_day(raw_dir, dt.date(2024, 1, 5))
    fx.write_calendar(raw_dir.parents[1] / "calendar.csv", [*DAYS, dt.date(2024, 1, 5)])
    reverify(raw_dir)
    store.convert(raw_dir, store_dir, workers=1)
    assert extract.extract_state(store_dir, "SPX") == "stale"
    again = extract.extract(store_dir, ["SPX"])
    assert (
        again.tickers["SPX"]["n_dates"] == 4
        and extract.extract_state(store_dir, "SPX") == "current"
    )
    assert extract.extract_state(store_dir, "AAPL") == "stale"
    with pytest.raises(DataError, match="not a ticker"):
        extract.parse_tickers("SPX,../x")
    with pytest.raises(DataError, match="no ticker"):
        extract.parse_tickers(" , ")
    assert again.tickers["SPX"]["schema_versions"] == [1]


# ---------------------------------------------------------------------------------------------
# sql and the CLI
# ---------------------------------------------------------------------------------------------


def test_sql_over_the_store(dirs: tuple[Path, Path]) -> None:
    pytest.importorskip("duckdb", reason="DuckDB is in the data extra only")
    raw_dir, store_dir = dirs
    with pytest.raises(DataError, match="run volsto-data convert first"):
        sql.run_sql(store_dir, "select 1")
    store.convert(raw_dir, store_dir, workers=1)
    out = sql.run_sql(
        store_dir,
        "select ticker, count(*) as n, min(year) as y from strikes group by all order by ticker",
    )
    want = pd.concat([fx.day_frame(d) for d in DAYS]).groupby("ticker").size()
    assert dict(zip(out["ticker"], out["n"])) == want.to_dict() and set(out["y"]) == {2024}
    with pytest.raises(DataError, match="DuckDB"):
        sql.run_sql(store_dir, "select nope from strikes")


def test_core_does_not_import_duckdb() -> None:
    """DuckDB is optional: importing the data layer, its CLI and the library never loads it."""
    code = (
        "import sys, volsto, volsto.data.cli, volsto.data.store, volsto.data.extract, "
        "volsto.data.sql, volsto.market; assert 'duckdb' not in sys.modules, 'duckdb imported'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_cli_convert_verify_extract_sql(
    dirs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    raw_dir, store_dir = dirs
    base = ["--raw", str(raw_dir.parent), "--store", str(store_dir.parent)]
    assert cli.main([*base, "verify", "--vendor", "orats"]) == 2
    assert "no store manifest" in capsys.readouterr().out
    assert cli.main([*base, "convert", "--vendor", "orats", "--workers", "2"]) == 0
    out = capsys.readouterr().out
    assert "converted 3 files" in out and "clean" in out and "peak memory" in out
    assert cli.main([*base, "convert", "--vendor", "orats"]) == 0
    assert "converted 0 files" in capsys.readouterr().out
    assert cli.main([*base, "verify", "--vendor", "orats", "--workers", "1"]) == 0
    assert "faithful copy" in capsys.readouterr().out
    assert cli.main([*base, "extract", "--tickers", "SPX,XSP"]) == 0
    assert "SPX:" in capsys.readouterr().out
    assert cli.main([*base, "extract", "--tickers", "NOPE"]) == 1
    assert "FAILED NOPE" in capsys.readouterr().out
    assert cli.main([*base, "status"]) == 0
    out = capsys.readouterr().out
    assert "3 days, 2024-01-02 .. 2024-01-04" in out and "extract SPX" in out and "current" in out
    try:
        import duckdb  # noqa: F401
    except ImportError:
        assert cli.main([*base, "sql", "select 1"]) == 2
        return
    assert cli.main([*base, "sql", "--max-rows", "1", "select distinct ticker from strikes"]) == 0
    assert "more rows" in capsys.readouterr().out
    # a failing file makes convert exit 1 and names it
    fx.write_day(raw_dir, DAYS[0], mutate=lambda d: d.drop(columns=["pOpra"]))
    reverify(raw_dir)
    assert cli.main([*base, "convert", "--vendor", "orats", "--workers", "1"]) == 1
    assert "FAILED ORATS_SMV_Strikes_20240102.zip: schema drift" in capsys.readouterr().out
    assert json.loads((store_dir / store.MANIFEST_NAME).read_text())["files"].keys() == {
        ISO[1],
        ISO[2],
    }


def test_convert_the_sample(tmp_path: Path) -> None:
    """The real sample day through convert and verify: the Part 0 counts, a faithful copy.
    Size, time and memory are printed, not asserted."""
    if not SAMPLE.exists():
        pytest.skip("the ORATS one-day sample is absent (data/orats_sample)")
    raw_dir = tmp_path / "raw" / "orats"
    raw_dir.mkdir(parents=True)
    (raw_dir / SAMPLE.name).symlink_to(SAMPLE)
    raw.write_manifest(raw_dir, raw.verify_raw(raw_dir, workers=1).manifest())
    store_dir = tmp_path / "store" / "orats"
    rep = store.convert(raw_dir, store_dir, workers=1)
    print("\n" + store.format_convert(rep))
    assert rep.clean and rep.converted == ["2024-01-03"] and rep.rows == 716_822
    entry = store.read_manifest(store_dir)["files"]["2024-01-03"]
    assert entry["raw_sha256"].startswith("e04a3731") and entry["copra_duplicates"] == 0
    tbl = pq.read_table(store.day_path(store_dir, "2024-01-03"), filters=[("ticker", "=", "SPX")])
    assert tbl.num_rows == 10_519
    assert tbl.group_by(["ticker", "expirDate", "strike"]).aggregate([]).num_rows == 10_519 - 1_590
    check = store.verify_store(raw_dir, store_dir, workers=1)
    print(store.format_verify(check))
    assert check.clean


def test_older_layout_becomes_nulls_and_the_key_falls_back(dirs: tuple[Path, Path]) -> None:
    """Owner's instruction 2026-10-04: the 37-column layout (no OPRA columns) is schema
    version 2; its absent columns are null in the store, which keeps one schema; the uniqueness
    key is ``cOpra`` when present, otherwise (ticker, expirDate, strike); the extract spans
    both layouts."""
    raw_dir, store_dir = dirs

    def old(df: pd.DataFrame) -> pd.DataFrame:
        # the old files: no OPRA columns, the PM third-Friday series under its own ticker
        pm = df["cOpra"].map(orats.opra_root) == "SPXW"
        df = df.assign(ticker=df["ticker"].where(~((df["ticker"] == "SPX") & pm), "SPXPM"))
        return df.drop(columns=["cOpra", "pOpra"])

    def old_repeated(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=["cOpra", "pOpra"])  # SPX and SPXW rows now share their keys
        return df

    for day in DAYS:  # the fixture's files move into the year folder
        (raw_dir / fx.FILE_NAME.format(day)).unlink()
    fx.write_day(raw_dir / "2024", DAYS[0], mutate=old)
    fx.write_day(raw_dir / "2024", DAYS[1], mutate=old_repeated)
    fx.write_day(raw_dir / "2024", DAYS[2])
    assert orats.schema_version_of(orats.schema_columns(2)) == 2
    assert not reverify(raw_dir).findings()
    rep = store.convert(raw_dir, store_dir, workers=1)
    assert sorted(rep.converted) == ISO and not rep.failed
    files = store.read_manifest(store_dir)["files"]
    assert [files[d]["schema_version"] for d in ISO] == [2, 2, 1]
    frame = fx.day_frame(DAYS[0])
    assert files[ISO[0]]["copra_empty"] == len(frame) and files[ISO[0]]["key_duplicates"] == 0
    shared = frame[frame["ticker"] == "SPX"].duplicated(["expirDate", "strike"]).sum()
    assert files[ISO[1]]["key_duplicates"] == shared > 0
    assert (
        list(rep.violations) == [ISO[1]] and "without a symbol repeating" in rep.violations[ISO[1]]
    )
    got = pq.read_table(store.day_path(store_dir, ISO[0]))
    assert got.schema.remove_metadata() == store.arrow_schema()
    assert got["cOpra"].null_count == got["pOpra"].null_count == got.num_rows == len(frame)
    assert got["cMidIv"].null_count == 0
    assert store.verify_store(raw_dir, store_dir, workers=1).clean
    out = extract.extract(store_dir, ["SPX", "SPXPM"])
    assert out.tickers["SPX"]["n_dates"] == 3 and out.tickers["SPX"]["schema_versions"] == [1, 2]
    assert out.tickers["SPXPM"]["n_dates"] == 1
    spx = pq.read_table(extract.ticker_path(store_dir, "SPX"))
    assert spx.schema.remove_metadata() == store.arrow_schema()
    assert 0 < spx["cOpra"].null_count < spx.num_rows
