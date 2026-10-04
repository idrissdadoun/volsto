"""M11 Part 1 tests (SPEC §18): data roots, ``verify-raw``, ``fetch`` and the ``volsto-data``
CLI.

The plumbing runs on the synthetic ORATS-format fixture of ``tests/_orats_fixture.py`` (built
in a temporary directory; its third Fridays carry the SPX and SPXW roots on one expiry date).
The tests that read the real one-day sample skip when ``data/orats_sample/`` is absent (the
sample is the vendor's and git-ignored).  ``fetch`` is run against a stand-in ``aws``
executable: no test touches the network or a credential.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import _orats_fixture as fx
import pandas as pd
import pytest

from volsto.data import cli, orats, raw
from volsto.data import fetch as fetchmod
from volsto.data.roots import (
    DATA_DIR,
    DEFAULT_RAW_ROOT,
    DEFAULT_STORE_ROOT,
    ENV_RAW,
    ENV_STORE,
    REPO_ROOT,
    DataError,
    DataRoots,
    require_free_space,
)

SAMPLE = REPO_ROOT / "data" / "orats_sample" / "ORATS_SMV_Strikes_20240103.zip"
SAMPLE_SHA256 = "e04a37310b0f453fedc03a000c55adfc517a119279150173916ebdc5fb514570"
DAYS = [dt.date(2024, 1, 2), dt.date(2024, 1, 3), dt.date(2024, 1, 4), dt.date(2024, 1, 5)]


def need_sample() -> Path:
    if not SAMPLE.exists():
        pytest.skip(f"the ORATS one-day sample is absent ({SAMPLE.relative_to(REPO_ROOT)})")
    return SAMPLE


@pytest.fixture()
def raw_dir(tmp_path: Path) -> Path:
    """Four consecutive synthetic trading days under ``<tmp>/raw/orats`` and their calendar."""
    d = tmp_path / "raw" / "orats"
    for day in DAYS:
        fx.write_day(d, day)
    fx.write_calendar(tmp_path / "calendar.csv", [dt.date(2023, 12, 29), *DAYS])
    return d


def verify(raw_dir: Path, **kwargs: object) -> raw.RawReport:
    kwargs.setdefault("calendar", raw_dir.parents[1] / "calendar.csv")
    kwargs.setdefault("workers", 1)
    return raw.verify_raw(raw_dir, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------------------------
# roots and git hygiene
# ---------------------------------------------------------------------------------------------


def test_roots_precedence(tmp_path: Path) -> None:
    r = DataRoots.resolve(env={})
    assert (r.raw, r.store) == (DEFAULT_RAW_ROOT, DEFAULT_STORE_ROOT)
    assert (r.raw_source, r.store_source) == ("default", "default")
    assert DATA_DIR in DEFAULT_RAW_ROOT.parents and DATA_DIR in DEFAULT_STORE_ROOT.parents
    env = {ENV_RAW: "/Volumes/ext/raw", ENV_STORE: str(tmp_path / "store")}
    r = DataRoots.resolve(env=env)
    assert (r.raw, r.raw_source) == (Path("/Volumes/ext/raw"), "env")
    assert r.store == tmp_path / "store"
    r = DataRoots.resolve(raw="rel/raw", env=env)
    assert (r.raw, r.raw_source, r.store_source) == (Path("rel/raw"), "flag", "env")
    assert r.raw_dir("orats") == Path("rel/raw/orats")
    with pytest.raises(DataError, match="blank"):
        DataRoots.resolve(env={ENV_RAW: " "})
    with pytest.raises(DataError, match="blank"):
        DataRoots.resolve(store="")


def test_no_tracked_file_under_the_data_roots() -> None:
    """Vendor data never enters git: nothing tracked under ``data/`` or under either resolved
    root, and no tracked zip or Parquet anywhere but the test goldens."""
    git = shutil.which("git")
    if git is None or not (REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    out = subprocess.run(
        [git, "ls-files", "-z"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    )
    tracked = [f for f in out.stdout.split("\0") if f]
    assert tracked
    roots = DataRoots.resolve()
    prefixes = ["data/"]
    for root in (DEFAULT_RAW_ROOT, DEFAULT_STORE_ROOT, roots.raw.resolve(), roots.store.resolve()):
        if REPO_ROOT in root.parents:
            prefixes.append(root.relative_to(REPO_ROOT).as_posix() + "/")
    under = [f for f in tracked if any(f.startswith(p) for p in prefixes)]
    assert under == [], f"tracked files under the data roots: {under[:10]}"
    bulk = [f for f in tracked if f.endswith((".parquet", ".zip")) and not f.startswith("tests/")]
    assert bulk == [], f"tracked vendor-like files: {bulk[:10]}"
    ignored = subprocess.run(
        [git, "check-ignore", "-q", "data/raw/orats/x.zip"], cwd=REPO_ROOT, check=False
    )
    assert ignored.returncode == 0, "data/ is no longer git-ignored"


def test_require_free_space(tmp_path: Path) -> None:
    free = require_free_space(tmp_path / "not" / "yet", 1, what="a test", margin_bytes=0)
    assert free > 0
    with pytest.raises(DataError, match="not enough free space for a test"):
        require_free_space(tmp_path, free * 2, what="a test")


# ---------------------------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------------------------


def test_schema_is_declared() -> None:
    cols = orats.schema_columns(1)
    assert len(cols) == 39 and len(set(cols)) == 39
    types = dict(orats.SCHEMA_V1)
    assert types["divRate"] == "float64"
    assert {c for c, t in types.items() if t == "date"} == {"expirDate", "trade_date"}
    assert {c for c, t in types.items() if t == "string"} == {"ticker", "cOpra", "pOpra"}
    assert {c for c, t in types.items() if t == "int64"} == {"cVolu", "cOi", "pVolu", "pOi"}
    assert orats.schema_version_of(cols) == 1
    assert orats.schema_version_of(cols[:-1]) is None
    drift = orats.describe_drift([c for c in cols if c != "cOpra"] + ["extra"])
    assert "missing ['cOpra']" in drift and "unexpected ['extra']" in drift
    assert "another order" in orats.describe_drift([cols[1], cols[0], *cols[2:]])
    assert orats.parse_date("1/3/2024") == dt.date(2024, 1, 3)
    with pytest.raises(ValueError):
        orats.parse_date("2024-01-03")
    assert orats.file_trade_date("ORATS_SMV_Strikes_20240103.zip") == dt.date(2024, 1, 3)
    assert orats.file_trade_date("other_20240103.zip") is None
    assert orats.opra_root("SPXW240119C04700000") == "SPXW"
    assert orats.opra_root("A240119C00055000") == "A"


def test_fixture_third_friday_shares_roots() -> None:
    df = fx.day_frame(DAYS[1])
    assert tuple(df.columns) == orats.schema_columns(1)
    dup = df[df.duplicated(["ticker", "expirDate", "strike"], keep=False)]
    assert set(dup["ticker"]) == {"SPX"}
    assert set(dup["expirDate"]) == {fx.us_date(d) for d in fx.THIRD_FRIDAYS}
    roots = dup["cOpra"].map(orats.opra_root)
    assert set(roots) == {"SPX", "SPXW"}
    assert (
        not df.assign(root=df["cOpra"].map(orats.opra_root))
        .duplicated(["root", "expirDate", "strike"])
        .any()
    )
    assert df["cOpra"].is_unique and df["pOpra"].is_unique
    # the AM root prices one day shorter: its straddle is cheaper at the same strike
    jan = dup[dup["expirDate"] == "1/19/2024"].assign(root=roots)
    atm = jan[jan["strike"] == 4700.0].set_index("root")
    straddle = atm["cValue"] + atm["pValue"]
    assert straddle["SPX"] < straddle["SPXW"]


# ---------------------------------------------------------------------------------------------
# verify-raw
# ---------------------------------------------------------------------------------------------


def test_verify_raw_clean(raw_dir: Path) -> None:
    rep = verify(raw_dir)
    assert rep.findings() == {}
    assert [f.trade_date for f in rep.files] == [d.isoformat() for d in DAYS]
    f = rep.files[1]
    frame = fx.day_frame(DAYS[1])
    assert f.rows == len(frame) and f.columns == orats.schema_columns(1)
    assert f.schema_version == 1 and f.content_trade_dates == ("1/3/2024",)
    assert f.opra_columns_present and (f.opra_c_empty, f.opra_p_empty) == (0, 0)
    assert f.spx_rows == int((frame["ticker"] == "SPX").sum()) and f.spx_opra_empty == 0
    assert f.sha256 == raw.sha256_file(raw_dir / f.file) and f.bytes == os.path.getsize(
        raw_dir / f.file
    )
    (year,) = rep.opra_coverage_by_year()
    assert (year["year"], year["files"], year["opra_ok_spx"]) == ("2024", 4, 4)
    text = raw.format_report(rep)
    assert "clean" in text and "FINDING" not in text


def test_verify_raw_manifest_and_reuse(raw_dir: Path) -> None:
    rep = verify(raw_dir)
    path = raw.write_manifest(raw_dir, rep.manifest())
    man = json.loads(path.read_text())
    assert man["n_files"] == 4 and man["first"] == "2024-01-02" and man["last"] == "2024-01-05"
    assert [e["sha256"] for e in man["files"]] == [f.sha256 for f in rep.files]
    assert not list(raw_dir.glob(".*tmp"))
    again = verify(raw_dir)
    assert again.reused == 4 and again.files == rep.files
    # a file that changed is read again; --rehash reads everything
    time.sleep(0.01)
    fx.write_day(raw_dir, DAYS[0], mutate=lambda d: d.iloc[:-3])
    changed = verify(raw_dir)
    assert changed.reused == 3 and changed.files[0].rows == (rep.files[0].rows or 0) - 3
    assert changed.files[0].sha256 != rep.files[0].sha256
    assert verify(raw_dir, rehash=True).reused == 0
    # the zips were only read
    assert sorted(p.name for p in raw_dir.iterdir()) == sorted(
        [raw.MANIFEST_NAME, *(fx.FILE_NAME.format(d) for d in DAYS)]
    )


def test_verify_raw_checks_a_second_copy(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The manifest travels inside the raw directory, is never read as a raw file, and a copy
    is checked against it by content (size and mtime preserved, so nothing may be reused)."""
    cal = tmp_path / "calendar.csv"
    raw.write_manifest(raw_dir, verify(raw_dir).manifest())
    assert verify(raw_dir).findings() == {}, "the manifest in the directory is not a finding"
    copy = tmp_path / "copy" / "orats"
    shutil.copytree(raw_dir, copy, copy_function=shutil.copy2)
    own = copy / raw.MANIFEST_NAME
    rep = raw.verify_raw(copy, calendar=cal, workers=1, against=own)
    assert rep.findings() == {} and rep.reused == 0 and rep.reference == own
    assert "identical to the reference manifest" in raw.format_report(rep)
    # one flipped byte with size and mtime kept, one file missing, one extra
    victim = copy / fx.FILE_NAME.format(DAYS[1])
    st = victim.stat()
    data = bytearray(victim.read_bytes())
    data[-1] ^= 0xFF  # the zip's trailing comment-length byte: size unchanged
    victim.write_bytes(bytes(data))
    os.utime(victim, ns=(st.st_atime_ns, st.st_mtime_ns))
    (copy / fx.FILE_NAME.format(DAYS[3])).unlink()
    rep = raw.verify_raw(copy, calendar=cal, workers=1, against=raw_dir / raw.MANIFEST_NAME)
    assert set(rep.reference_diff) == {fx.FILE_NAME.format(DAYS[1]), fx.FILE_NAME.format(DAYS[3])}
    assert "sha256" in rep.reference_diff[fx.FILE_NAME.format(DAYS[1])]
    assert "absent here" in rep.reference_diff[fx.FILE_NAME.format(DAYS[3])]
    assert rep.findings()["files differing from the reference manifest"] == 2
    with pytest.raises(DataError, match="no such manifest"):
        raw.verify_raw(copy, calendar=cal, against=tmp_path / "none.json")
    argv = ["--raw", str(copy.parent), "verify-raw", "--vendor", "orats", "--workers", "1"]
    argv += ["--calendar", str(cal), "--against", str(raw_dir / raw.MANIFEST_NAME)]
    assert cli.main(argv) == 1
    assert "FINDING files differing from the reference manifest (2)" in capsys.readouterr().out


def test_verify_raw_parallel_matches_serial(raw_dir: Path) -> None:
    assert verify(raw_dir, workers=3).files == verify(raw_dir, workers=1).files


def test_verify_raw_calendar_findings(raw_dir: Path, tmp_path: Path) -> None:
    (raw_dir / fx.FILE_NAME.format(DAYS[2])).unlink()  # a missing trading day
    fx.write_day(raw_dir, dt.date(2024, 1, 6))  # a Saturday: beyond this calendar
    fx.write_day(raw_dir / "again", DAYS[0])  # the same date twice (sub-directory)
    fx.write_day(raw_dir, DAYS[1], name="strikes_20240103.zip")  # the pattern does not match
    rep = verify(raw_dir)
    assert rep.missing == ["2024-01-04"]
    assert rep.beyond_calendar == ["2024-01-06"]
    assert list(rep.duplicated) == ["2024-01-02"] and len(rep.duplicated["2024-01-02"]) == 2
    assert rep.unmatched == ["strikes_20240103.zip"]
    cal = fx.write_calendar(tmp_path / "cal2.csv", [*DAYS, dt.date(2024, 1, 8)])
    rep2 = verify(raw_dir, calendar=cal)
    assert rep2.unexpected == ["2024-01-06"] and rep2.beyond_calendar == []
    found = rep.findings()
    assert found["missing trading days"] == 1 and found["dates with more than one file"] == 1
    text = raw.format_report(rep)
    assert "NOT CLEAN" in text and "2024-01-04" in text and "strikes_20240103.zip" in text
    with pytest.raises(DataError, match="calendar"):
        verify(raw_dir, calendar=tmp_path / "absent.csv")


def test_verify_raw_content_findings(raw_dir: Path) -> None:
    def no_opra(df: pd.DataFrame) -> pd.DataFrame:
        return df.drop(columns=["cOpra", "pOpra"])

    def blank_spx(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df.loc[(df["ticker"] == "SPX") & (df.index % 2 == 0), "pOpra"] = ""
        return df

    def wrong_date(df: pd.DataFrame) -> pd.DataFrame:
        return df.assign(trade_date="1/9/2024")

    def renamed(df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={"spot_px": "spotPx"}).assign(newCol=1)

    fx.write_day(raw_dir, DAYS[0], mutate=no_opra)
    fx.write_day(raw_dir, DAYS[1], mutate=blank_spx)
    fx.write_day(raw_dir, DAYS[2], mutate=wrong_date)
    fx.write_day(raw_dir, DAYS[3], mutate=renamed)
    (raw_dir / "ORATS_SMV_Strikes_20231229.zip").write_bytes(b"not a zip")
    rep = verify(raw_dir)
    names = {d: fx.FILE_NAME.format(d) for d in DAYS}
    assert "columns absent: ['cOpra', 'pOpra']" in rep.opra_missing[names[DAYS[0]]]
    assert "SPX rows without an OPRA symbol" in rep.opra_missing[names[DAYS[1]]]
    assert set(rep.opra_missing) == {names[DAYS[0]], names[DAYS[1]]}
    assert "file name says 2024-01-04, the trade_date column 2024-01-09" in (
        rep.date_mismatch[names[DAYS[2]]]
    )
    assert set(rep.drift) == {names[DAYS[0]], names[DAYS[3]]}
    assert "missing ['spot_px']" in rep.drift[names[DAYS[3]]]
    assert "unexpected ['spotPx', 'newCol']" in rep.drift[names[DAYS[3]]]
    assert list(rep.unreadable) == ["ORATS_SMV_Strikes_20231229.zip"]
    (y24,) = rep.opra_coverage_by_year()  # the unreadable 2023 file is in no year
    assert (y24["files"], y24["opra_columns"], y24["opra_ok_spx"]) == (4, 3, 2)
    assert y24["p_empty"] == y24["spx_opra_empty"] > 0 and y24["c_empty"] == 0
    assert set(rep.findings()) == {
        "unreadable files",
        "trade_date mismatches",
        "schema drift",
        "files without OPRA symbols on SPX",
    }
    with pytest.raises(DataError, match="does not exist"):
        raw.verify_raw(raw_dir / "nope")
    with pytest.raises(DataError, match="regular expression"):
        verify(raw_dir, pattern="(")


def test_verify_raw_on_the_sample() -> None:
    """The Part 0 facts of the real sample (2024-01-03), re-measured by the scanner."""
    sample = need_sample()
    f = raw.scan_zip(sample, sample.name, orats.DEFAULT_FILE_PATTERN)
    assert f.error is None
    assert (f.bytes, f.sha256) == (66_808_802, SAMPLE_SHA256)
    assert (f.csv_bytes, f.rows) == (222_935_113, 716_822)
    assert f.columns == orats.schema_columns(1) and f.schema_version == 1
    assert f.trade_date == "2024-01-03" and f.content_trade_dates == ("1/3/2024",)
    assert f.opra_columns_present and (f.opra_c_empty, f.opra_p_empty) == (0, 0)
    assert (f.spx_rows, f.spx_opra_empty) == (10_519, 0)
    rep = raw.verify_raw(sample.parent, workers=1, rehash=True)
    assert rep.findings() == {} and rep.files == [f]


# ---------------------------------------------------------------------------------------------
# fetch (against a stand-in aws executable)
# ---------------------------------------------------------------------------------------------

FAKE_AWS = """#!/bin/sh
echo "$@" >> "{log}"
if [ "$2" = "ls" ]; then
  [ -n "$FAKE_AWS_FAIL_LS" ] && exit 255
  echo "2026-10-10 01:00:00   66808802 smv/ORATS_SMV_Strikes_20240102.zip"
  echo "2026-10-10 01:00:01   66808802 smv/ORATS_SMV_Strikes_20240103.zip"
  echo ""
  echo "Total Objects: 2"
  echo "   Total Size: ${{FAKE_AWS_TOTAL:-133617604}}"
  exit 0
fi
case "$*" in
  *--dryrun*) echo "(dryrun) download: s3://b/smv/a.zip to $4/a.zip"; exit 0;;
esac
mkdir -p "$4" && echo data > "$4/ORATS_SMV_Strikes_20240102.zip"
exit ${{FAKE_AWS_SYNC_EXIT:-0}}
"""


@pytest.fixture()
def fake_aws(tmp_path: Path) -> tuple[str, Path]:
    log = tmp_path / "aws_calls.log"
    exe = tmp_path / "bin" / "aws"
    exe.parent.mkdir()
    exe.write_text(FAKE_AWS.format(log=log))
    exe.chmod(0o755)
    return str(exe), log


def test_fetch_dry_run_then_sync(
    tmp_path: Path, fake_aws: tuple[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    aws, log = fake_aws
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "s3cr3t-never-printed")
    dest = tmp_path / "raw" / "orats"
    out: list[str] = []
    kw = {"profile": "orats", "bucket": "the-bucket", "prefix": "/smv/", "aws_bin": aws}
    assert fetchmod.fetch(dest, dry_run=True, out=out, **kw) == 0  # type: ignore[arg-type]
    assert not dest.exists(), "a dry run writes nothing"
    assert any("2 objects, 133,617,604 bytes" in ln for ln in out)
    assert any("1 objects would be downloaded" in ln for ln in out)
    assert fetchmod.fetch(dest, include=["*.zip"], out=out, **kw) == 0  # type: ignore[arg-type]
    assert (dest / "ORATS_SMV_Strikes_20240102.zip").exists()
    calls = log.read_text().splitlines()
    assert calls[0] == "s3 ls s3://the-bucket/smv/ --recursive --summarize --profile orats"
    assert calls[1].endswith("--profile orats --no-progress --dryrun")
    assert calls[2] == calls[0], "the sync lists first"
    assert calls[3] == (
        f"s3 sync s3://the-bucket/smv/ {dest} --profile orats --no-progress "
        "--exclude * --include *.zip"
    )
    assert all("--delete" not in c for c in calls)
    assert "s3cr3t" not in "\n".join(out) and "s3cr3t" not in log.read_text()


def test_fetch_refusals(
    tmp_path: Path, fake_aws: tuple[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    aws, log = fake_aws
    dest = tmp_path / "raw" / "orats"
    kw = {"profile": "orats", "bucket": "b", "prefix": "p", "aws_bin": aws}
    monkeypatch.setenv("FAKE_AWS_TOTAL", str(10**18))  # an exabyte: short of space
    with pytest.raises(DataError, match="not enough free space for the download"):
        fetchmod.fetch(dest, **kw)  # type: ignore[arg-type]
    assert not dest.exists() and "sync" not in log.read_text()
    out: list[str] = []
    assert fetchmod.fetch(dest, dry_run=True, out=out, **kw) == 0  # type: ignore[arg-type]
    assert any("WARNING: the sync would be refused" in ln for ln in out)
    monkeypatch.delenv("FAKE_AWS_TOTAL")
    monkeypatch.setenv("FAKE_AWS_FAIL_LS", "1")
    with pytest.raises(DataError, match=r"aws s3 ls .* failed"):
        fetchmod.fetch(dest, **kw)  # type: ignore[arg-type]
    monkeypatch.delenv("FAKE_AWS_FAIL_LS")
    for bad in ("--delete", "--profile=other", "--dryrun"):
        with pytest.raises(DataError, match="refused"):
            fetchmod.fetch(dest, extra=[bad], **kw)  # type: ignore[arg-type]
    with pytest.raises(DataError, match="not found on PATH"):
        fetchmod.fetch(dest, **{**kw, "aws_bin": str(tmp_path / "no-aws")})  # type: ignore[arg-type]
    with pytest.raises(DataError, match="bucket name alone"):
        fetchmod.fetch(dest, **{**kw, "bucket": "b/sub"})  # type: ignore[arg-type]
    with pytest.raises(DataError, match="--profile is required"):
        fetchmod.fetch(dest, **{**kw, "profile": " "})  # type: ignore[arg-type]
    monkeypatch.setenv("FAKE_AWS_SYNC_EXIT", "1")
    out = []
    assert fetchmod.fetch(dest, out=out, **kw) == 1  # type: ignore[arg-type]
    assert any("run the same command again to resume" in ln for ln in out)
    with pytest.raises(DataError, match="Total Objects"):
        fetchmod.parse_listing("garbage")
    assert fetchmod.s3_uri("s3://b/", "") == "s3://b/"


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def test_cli_status_and_verify_raw(
    raw_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    base = ["--raw", str(raw_dir.parent), "--store", str(tmp_path / "store")]
    cal = ["--calendar", str(tmp_path / "calendar.csv"), "--workers", "1"]
    assert cli.main([*base, "status"]) == 0
    out = capsys.readouterr().out
    assert "4 raw zips" in out and "volsto-data verify-raw --vendor orats" in out
    assert cli.main([*base, "verify-raw", "--vendor", "orats", *cal]) == 0
    out = capsys.readouterr().out
    assert "clean" in out and "OPRA coverage per year" in out and "wall clock" in out
    assert (raw_dir / raw.MANIFEST_NAME).exists()
    assert cli.main([*base, "status"]) == 0
    assert "findings none" in capsys.readouterr().out
    (raw_dir / fx.FILE_NAME.format(DAYS[1])).unlink()
    assert cli.main([*base, "status"]) == 0
    assert "the manifest is stale" in capsys.readouterr().out
    assert cli.main([*base, "verify-raw", "--vendor", "orats", *cal]) == 1
    assert "FINDING missing trading days (1): 2024-01-03" in capsys.readouterr().out
    assert cli.main(["--raw", str(tmp_path / "none"), "verify-raw", "--vendor", "orats"]) == 2
    assert "does not exist" in capsys.readouterr().out


def test_cli_fetch(
    tmp_path: Path,
    fake_aws: tuple[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    aws, log = fake_aws
    monkeypatch.setenv("PATH", f"{Path(aws).parent}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv(ENV_RAW, str(tmp_path / "raw"))
    argv = ["fetch", "--vendor", "orats", "--profile", "p", "--bucket", "b", "--prefix", "x"]
    assert cli.main([*argv, "--dry-run", "--", "--request-payer", "requester"]) == 0
    assert "nothing written" in capsys.readouterr().out
    assert log.read_text().splitlines()[0].endswith("--profile p --request-payer requester")
    assert cli.main(argv) == 0
    assert (tmp_path / "raw" / "orats" / "ORATS_SMV_Strikes_20240102.zip").exists()
    assert cli.main([*argv, "--", "--delete"]) == 2
