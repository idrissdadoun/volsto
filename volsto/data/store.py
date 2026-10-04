"""The typed Parquet store: ``volsto-data convert`` and ``verify`` (SPEC §18.4, M11 Part 2b).

**A faithful typed copy, nothing derived.**  Each raw zip's CSV is read straight from the zip
(never unzipped to disk) and written as one Parquet file per trading day,

    ``<store>/<vendor>/strikes/year=YYYY/YYYY-MM-DD.parquet``

with every vendor column under its original name, typed from the schema declared in
:mod:`volsto.data.orats` (nothing is inferred from a file; dates are parsed with the explicit
format and fail loudly).  Rows are sorted by :data:`SORT_KEY` — (ticker, expirDate, strike,
cOpra) — so a ticker filter prunes row groups on their statistics.

**Layout** (owner's choice from the Part 2a table, 2026-10-03; :data:`LAYOUT`): zstd level 9,
dictionary encoding off, row groups of 131,072 rows, float64 kept exactly, the full universe.
Measured on the sample day: 76.4 MB, 1.14 × the zip.

**Loud failures** — a file is not converted, its reason is listed and the command exits 1 —
on: schema drift (the header is not a known schema version; the difference is named, and a
version is added only after the owner confirms); a date or number that does not parse; a
``trade_date`` column that disagrees with the file name; ``cOpra`` or ``pOpra`` empty on an
``SPX`` row (owner's decision 8: the OPRA root is the only AM/PM marker); a raw file whose
sha256 is not the raw manifest's (run ``verify-raw`` again).  ``cOpra`` is checked for
uniqueness within each file and violations are reported (the file is still converted: the
store is a copy).

**Idempotent and atomic.**  The store manifest ``<store>/<vendor>/store_manifest.json`` binds
each Parquet file to its raw file's sha256, its row count, its schema version and its own
sha256; a date whose entry matches the raw manifest and whose file is on disk is skipped.  Each
file is written to a temporary name in its own directory (the same volume, wherever the store
is mounted) and renamed; the manifest is replaced atomically, every
:data:`MANIFEST_EVERY` files and at the end, so an interrupted run resumes.  Workers are
bounded (:data:`DEFAULT_WORKERS`); the peak resident memory of a worker is reported.

``verify`` re-reads every raw file and its Parquet file and compares them: the raw sha256
against the manifest, the row count, and per column the null count and the sum (integers and
floats summed over identically ordered rows in one chunk — so the float sums are equal bit
for bit — dates as day numbers, strings as their UTF-8 lengths), then the whole tables for
equality.  Raw files without a Parquet file and Parquet
files without a manifest entry are listed.  Never silent: exit 1 on any finding.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import resource
import sys
import time
import zipfile
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

from volsto.data import orats
from volsto.data import raw as rawmod
from volsto.data.roots import DataError, ensure_dir, fmt_bytes, require_free_space

STRIKES_DIR: Final = "strikes"
MANIFEST_NAME: Final = "store_manifest.json"
MANIFEST_VERSION: Final = 1
SORT_KEY: Final[tuple[str, ...]] = ("ticker", "expirDate", "strike", "cOpra")
#: The Parquet layout (owner's choice, 2026-10-03).  A file written under another layout is
#: converted again.
LAYOUT: Final[dict[str, Any]] = {
    "version": 1,
    "compression": "zstd",
    "compression_level": 9,
    "dictionary": False,
    "row_group_size": 131_072,
    "sort": list(SORT_KEY),
}
#: Parquet bytes per raw zip byte, measured on the sample day (76.4 MB / 66.8 MB); used only
#: for the free-space estimate before a bulk conversion.
STORE_TO_ZIP_RATIO: Final[float] = 1.14
#: Conversion processes (bounded; each holds one day: about 1.2 GB at the sample's size).
DEFAULT_WORKERS: Final[int] = 8
#: The manifest is rewritten after this many converted files (and at the end).
MANIFEST_EVERY: Final[int] = 25
ARROW_TYPES: Final[dict[str, pa.DataType]] = {
    "string": pa.string(),
    "int64": pa.int64(),
    "float64": pa.float64(),
    "date": pa.date32(),
}
_CSV_BLOCK = 1 << 24


class ConvertError(DataError):
    """One raw file that cannot be converted; the message names the reason."""


def arrow_schema(version: int) -> pa.Schema:
    """The store schema of an ORATS schema version: the declared columns, in file order."""
    return pa.schema([(name, ARROW_TYPES[kind]) for name, kind in orats.SCHEMA_VERSIONS[version]])


def day_path(store_dir: Path, date: str) -> Path:
    """``<store>/<vendor>/strikes/year=YYYY/YYYY-MM-DD.parquet``."""
    return store_dir / STRIKES_DIR / f"year={date[:4]}" / f"{date}.parquet"


def read_raw_table(zip_path: Path) -> tuple[pa.Table, int]:
    """The CSV of a raw zip as a typed table in file order, and its schema version.  Raises
    :class:`ConvertError` on schema drift, on a zip without exactly one member and on any value
    that does not parse under the declared types."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            if len(infos) != 1:
                raise ConvertError(f"expected one member, found {len(infos)}")
            member = infos[0].filename
            columns = rawmod.read_header(zf, member)
            version = orats.schema_version_of(columns)
            if version is None:
                raise ConvertError("schema drift " + orats.describe_drift(columns))
            declared = orats.SCHEMA_VERSIONS[version]
            read_types = {
                name: pa.string() if kind == "date" else ARROW_TYPES[kind]
                for name, kind in declared
            }
            with zf.open(member) as fh:
                tbl = pcsv.read_csv(
                    fh,
                    read_options=pcsv.ReadOptions(block_size=_CSV_BLOCK),
                    convert_options=pcsv.ConvertOptions(
                        column_types=read_types, strings_can_be_null=False
                    ),
                )
        for name, kind in declared:
            if kind == "date":
                parsed = pc.strptime(tbl[name], format=orats.DATE_FORMAT, unit="s")
                i = tbl.column_names.index(name)
                tbl = tbl.set_column(i, name, pc.cast(parsed, pa.date32()))
        return tbl.cast(arrow_schema(version)), version
    except (zipfile.BadZipFile, pa.ArrowException, OSError, UnicodeDecodeError) as exc:
        raise ConvertError(f"{type(exc).__name__}: {exc}") from exc


def sort_table(tbl: pa.Table) -> pa.Table:
    """Rows in the store's order (:data:`SORT_KEY`; the sort is stable)."""
    return tbl.sort_by([(name, "ascending") for name in SORT_KEY]).combine_chunks()


def check_table(tbl: pa.Table, date: str) -> dict[str, Any]:
    """The per-file checks of the module docstring on a raw table; raises :class:`ConvertError`
    on a ``trade_date`` or OPRA failure and returns the ``cOpra`` uniqueness counts."""
    days = pc.unique(tbl["trade_date"]).to_pylist()
    if [d.isoformat() for d in days] != [date]:
        got = sorted(d.isoformat() for d in days)[:5]
        raise ConvertError(f"the file name says {date}, the trade_date column {got}")
    c_empty = pc.equal(pc.utf8_trim_whitespace(tbl["cOpra"]), "")
    p_empty = pc.equal(pc.utf8_trim_whitespace(tbl["pOpra"]), "")
    spx = pc.equal(tbl["ticker"], orats.OPRA_REQUIRED_TICKER)
    spx_empty = int(pc.sum(pc.and_(spx, pc.or_(c_empty, p_empty))).as_py() or 0)
    if spx_empty:
        raise ConvertError(
            f"{spx_empty} of {int(pc.sum(spx).as_py() or 0)} SPX rows without an OPRA symbol "
            "(the OPRA root is the only AM/PM marker; no fallback is implemented)"
        )
    n_empty = int(pc.sum(c_empty).as_py() or 0)
    symbols = tbl["cOpra"].filter(pc.invert(c_empty))
    duplicates = len(symbols) - int(pc.count_distinct(symbols).as_py())
    examples: list[str] = []
    if duplicates:
        counts = pc.value_counts(symbols)
        many = counts.filter(pc.greater(counts.field("counts"), 1))
        examples = many.field("values").to_pylist()[:5]
    return {"copra_empty": n_empty, "copra_duplicates": duplicates, "copra_examples": examples}


def write_parquet(tbl: pa.Table, dest: Path, metadata: dict[str, Any]) -> None:
    """Write ``tbl`` to ``dest`` in the store layout, atomically (a temporary file in the same
    directory, synced, then renamed)."""
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    meta = {f"volsto.{k}".encode(): str(v).encode() for k, v in metadata.items()}
    try:
        pq.write_table(
            tbl.replace_schema_metadata(meta),
            tmp,
            compression=LAYOUT["compression"],
            compression_level=LAYOUT["compression_level"],
            use_dictionary=LAYOUT["dictionary"],
            row_group_size=LAYOUT["row_group_size"],
        )
        with tmp.open("rb") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, dest)
    finally:
        tmp.unlink(missing_ok=True)


def peak_rss_bytes(who: int = resource.RUSAGE_SELF) -> int:
    """Peak resident set size (``ru_maxrss`` is bytes on macOS, kilobytes on Linux)."""
    peak = resource.getrusage(who).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)


def convert_one(job: tuple[str, str, str, str, str]) -> dict[str, Any]:
    """Convert one raw zip; returns the manifest entry, or ``{"error": reason}``.  Never
    raises on a bad file (the caller lists it)."""
    zip_path, rel, date, raw_sha, dest = job
    t0 = time.perf_counter()
    try:
        sha = rawmod.sha256_file(Path(zip_path))
        if sha != raw_sha:
            raise ConvertError(
                f"sha256 {sha[:12]}… is not the raw manifest's {raw_sha[:12]}…: the raw file "
                "changed since verify-raw"
            )
        tbl, version = read_raw_table(Path(zip_path))
        checks = check_table(tbl, date)
        out = Path(dest)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_parquet(
            sort_table(tbl),
            out,
            {
                "vendor": orats.VENDOR,
                "trade_date": date,
                "raw_file": rel,
                "raw_sha256": sha,
                "schema_version": version,
                "layout_version": LAYOUT["version"],
            },
        )
        return {
            "date": date,
            "file": out.name,
            "raw_file": rel,
            "raw_sha256": sha,
            "schema_version": version,
            "rows": tbl.num_rows,
            "bytes": out.stat().st_size,
            "sha256": rawmod.sha256_file(out),
            **checks,
            "layout_version": LAYOUT["version"],
            "converted_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "seconds": round(time.perf_counter() - t0, 3),
            "peak_rss": peak_rss_bytes(),
        }
    except DataError as exc:
        return {"date": date, "raw_file": rel, "error": str(exc)}


# --------------------------------------------------------------------------------------------
# manifest
# --------------------------------------------------------------------------------------------


def read_manifest(store_dir: Path) -> dict[str, Any]:
    """The store manifest (an empty one when the store has none)."""
    p = store_dir / MANIFEST_NAME
    if not p.exists():
        return {"version": MANIFEST_VERSION, "vendor": orats.VENDOR, "layout": LAYOUT, "files": {}}
    data: dict[str, Any] = json.loads(p.read_text())
    if data.get("version") != MANIFEST_VERSION:
        raise DataError(f"{p}: store manifest version {data.get('version')!r}, expected 1")
    return data


def write_manifest(store_dir: Path, manifest: dict[str, Any]) -> Path:
    p = store_dir / MANIFEST_NAME
    tmp = store_dir / f".{MANIFEST_NAME}.{os.getpid()}.tmp"
    manifest = dict(manifest, updated_utc=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"))
    manifest["files"] = dict(sorted(manifest["files"].items()))
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    return p


def store_digest(manifest: dict[str, Any]) -> str:
    """What derived data (the per-ticker extracts) is bound to: a SHA-256 over every date's raw
    sha256, row count, schema version and Parquet sha256, in date order."""
    rows = [
        [d, e["raw_sha256"], e["rows"], e["schema_version"], e["sha256"]]
        for d, e in sorted(manifest["files"].items())
    ]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def available_dates(store_dir: Path) -> list[str]:
    """ISO trade dates the store manifest holds, ascending."""
    return sorted(read_manifest(store_dir)["files"])


# --------------------------------------------------------------------------------------------
# convert
# --------------------------------------------------------------------------------------------


@dataclass
class ConvertReport:
    store_dir: Path
    converted: list[str] = field(default_factory=list)
    skipped: int = 0
    failed: dict[str, str] = field(default_factory=dict)  # raw file → reason
    violations: dict[str, str] = field(default_factory=dict)  # date → cOpra uniqueness
    rows: int = 0
    bytes_written: int = 0
    raw_bytes: int = 0
    wall_s: float = 0.0
    worker_seconds: float = 0.0
    peak_rss: int = 0
    workers: int = 1

    @property
    def clean(self) -> bool:
        return not self.failed and not self.violations


def raw_inputs(raw_dir: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """``(date → raw manifest entry, raw file → refusal)`` of a verified raw directory.  Raises
    :class:`DataError` when there is no raw manifest or it no longer describes the directory
    (``verify-raw`` hashes the files; ``convert`` checks each hash again as it reads)."""
    man = rawmod.read_manifest(raw_dir) if raw_dir.is_dir() else None
    if man is None:
        raise DataError(
            f"no raw manifest in {raw_dir}: run volsto-data verify-raw --vendor {orats.VENDOR}"
        )
    on_disk = {
        p.relative_to(raw_dir).as_posix(): p.stat()
        for p in raw_dir.rglob("*.zip")
        if not p.name.startswith(".")
    }
    listed = {e["file"]: e for e in man["files"]}
    stale = sorted(
        f
        for f in on_disk.keys() | listed.keys()
        if f not in on_disk
        or f not in listed
        or (on_disk[f].st_size, on_disk[f].st_mtime_ns)
        != (listed[f]["bytes"], listed[f]["mtime_ns"])
    )
    if stale:
        raise DataError(
            f"the raw manifest no longer describes {raw_dir} ({len(stale)} files added, removed "
            f"or changed, e.g. {stale[0]}): run volsto-data verify-raw --vendor {orats.VENDOR}"
        )
    by_date: dict[str, list[dict[str, Any]]] = {}
    refused: dict[str, str] = {}
    for e in man["files"]:
        if e["trade_date"] is None:
            refused[e["file"]] = "the file name does not match the pattern (no trade date)"
        elif e.get("error"):
            refused[e["file"]] = f"unreadable: {e['error']}"
        else:
            by_date.setdefault(e["trade_date"], []).append(e)
    inputs: dict[str, dict[str, Any]] = {}
    for date, entries in by_date.items():
        if len(entries) > 1:
            for e in entries:
                refused[e["file"]] = f"{len(entries)} raw files for {date}"
        else:
            inputs[date] = entries[0]
    return inputs, refused


def up_to_date(store_dir: Path, date: str, entry: dict[str, Any] | None, raw_sha: str) -> bool:
    """Whether ``date`` needs no conversion: its manifest entry names this raw sha256 and this
    layout, and its Parquet file is on disk with the recorded size."""
    if entry is None or entry["raw_sha256"] != raw_sha:
        return False
    if entry.get("layout_version") != LAYOUT["version"]:
        return False
    p = day_path(store_dir, date)
    return p.is_file() and p.stat().st_size == entry["bytes"]


def convert(
    raw_dir: Path,
    store_dir: Path,
    *,
    workers: int = DEFAULT_WORKERS,
    force: bool = False,
    progress: bool = True,
) -> ConvertReport:
    """Convert every verified raw file that is not up to date (module docstring)."""
    if workers < 1:
        raise DataError("workers must be at least 1")
    t0 = time.perf_counter()
    inputs, refused = raw_inputs(raw_dir)
    ensure_dir(store_dir)
    manifest = read_manifest(store_dir)
    manifest["layout"] = LAYOUT
    files: dict[str, Any] = manifest["files"]
    rep = ConvertReport(store_dir, failed=dict(refused), workers=workers)
    jobs = []
    for date in sorted(inputs):
        e = inputs[date]
        if not force and up_to_date(store_dir, date, files.get(date), e["sha256"]):
            rep.skipped += 1
            continue
        dest = day_path(store_dir, date)
        jobs.append((str(raw_dir / e["file"]), e["file"], date, e["sha256"], str(dest)))
        rep.raw_bytes += int(e["bytes"])
    if jobs:
        need = int(rep.raw_bytes * STORE_TO_ZIP_RATIO)
        require_free_space(store_dir, need, what=f"converting {len(jobs)} files")

    def take(result: dict[str, Any]) -> None:
        if "error" in result:
            rep.failed[result["raw_file"]] = result["error"]
            files.pop(result["date"], None)  # a stale entry must not vouch for a failed file
            return
        date = result.pop("date")
        rep.worker_seconds += result.pop("seconds")
        rep.peak_rss = max(rep.peak_rss, result.pop("peak_rss"))
        files[date] = result
        rep.converted.append(date)
        rep.rows += result["rows"]
        rep.bytes_written += result["bytes"]
        if result["copra_duplicates"]:
            rep.violations[date] = (
                f"{result['copra_duplicates']} duplicate cOpra symbols, e.g. "
                f"{result['copra_examples']}"
            )

    done = 0
    if workers == 1 or len(jobs) <= 1:
        for job in jobs:
            take(convert_one(job))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            for fut in as_completed([pool.submit(convert_one, job) for job in jobs]):
                take(fut.result())
                done += 1
                if done % MANIFEST_EVERY == 0:
                    write_manifest(store_dir, manifest)
                    if progress:
                        print(
                            f"  {done}/{len(jobs)} files, {time.perf_counter() - t0:.0f} s",
                            flush=True,
                        )
    # violations of files converted earlier stay visible
    for date, entry in files.items():
        if entry.get("copra_duplicates") and date not in rep.violations:
            rep.violations[date] = f"{entry['copra_duplicates']} duplicate cOpra symbols"
    write_manifest(store_dir, manifest)
    rep.wall_s = time.perf_counter() - t0
    return rep


def format_convert(rep: ConvertReport) -> str:
    out = [
        f"store {rep.store_dir}",
        f"  converted {len(rep.converted)} files ({rep.rows:,} rows, {fmt_bytes(rep.raw_bytes)} "
        f"of zips -> {fmt_bytes(rep.bytes_written)} of Parquet), {rep.skipped} up to date, "
        f"{len(rep.failed)} failed",
        f"  wall clock {rep.wall_s:.1f} s on {rep.workers} workers ({rep.worker_seconds:.1f} s "
        f"of worker time); peak memory of one worker {fmt_bytes(rep.peak_rss)}",
    ]
    for name, why in list(rep.failed.items())[:40]:
        out.append(f"  FAILED {name}: {why}")
    if len(rep.failed) > 40:
        out.append(f"  … and {len(rep.failed) - 40} more failures")
    for date, why in list(rep.violations.items())[:40]:
        out.append(f"  FINDING {date}: {why}")
    out.append("  clean" if rep.clean else "  NOT CLEAN")
    return "\n".join(out)


# --------------------------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------------------------


def column_summary(tbl: pa.Table) -> dict[str, tuple[int, Any]]:
    """Per column ``(null count, sum)``: integers and floats summed as stored, dates as their
    day numbers, strings as their UTF-8 lengths."""
    out: dict[str, tuple[int, Any]] = {}
    for name in tbl.column_names:
        col = tbl[name]
        if pa.types.is_string(col.type):
            total = pc.sum(pc.utf8_length(col)).as_py()
        elif pa.types.is_date32(col.type):
            total = pc.sum(pc.cast(col, pa.int32())).as_py()
        else:
            total = pc.sum(col).as_py()
        out[name] = (col.null_count, total)
    return out


def verify_one(job: tuple[str, str, str, dict[str, Any]]) -> list[str]:
    """Problems of one date (empty when the Parquet file is a faithful copy of its raw file)."""
    zip_path, parquet_path, _date, entry = job
    problems: list[str] = []
    try:
        if not Path(parquet_path).is_file():
            return ["the Parquet file is missing"]
        if rawmod.sha256_file(Path(zip_path)) != entry["raw_sha256"]:
            return ["the raw file's sha256 is not the one the store was built from"]
        if rawmod.sha256_file(Path(parquet_path)) != entry["sha256"]:
            problems.append("the Parquet file's sha256 is not the manifest's")
        raw, version = read_raw_table(Path(zip_path))
        raw = sort_table(raw)
        # one chunk on both sides: a float sum depends on how the rows are grouped
        got = pq.read_table(parquet_path).combine_chunks()
        if version != entry["schema_version"]:
            problems.append(f"schema version {entry['schema_version']} stored, {version} raw")
        if got.schema.remove_metadata() != arrow_schema(version):
            problems.append("the Parquet schema is not the declared schema")
            return problems
        if got.num_rows != raw.num_rows or got.num_rows != entry["rows"]:
            problems.append(
                f"rows: raw {raw.num_rows}, Parquet {got.num_rows}, manifest {entry['rows']}"
            )
            return problems
        want_sum, got_sum = column_summary(raw), column_summary(got)
        differing = len(problems)
        for name in raw.column_names:
            if want_sum[name][0] != got_sum[name][0]:
                problems.append(
                    f"{name}: null count raw {want_sum[name][0]}, Parquet {got_sum[name][0]}"
                )
            elif want_sum[name][1] != got_sum[name][1]:
                problems.append(f"{name}: sum raw {want_sum[name][1]}, Parquet {got_sum[name][1]}")
        differing = len(problems) - differing
        if not differing and not got.equals(raw):
            problems.append("row counts, null counts and sums agree but the tables differ")
    except DataError as exc:
        problems.append(str(exc))
    return problems


@dataclass
class VerifyReport:
    store_dir: Path
    checked: int = 0
    rows: int = 0
    problems: dict[str, list[str]] = field(default_factory=dict)  # date → problems
    not_converted: list[str] = field(default_factory=list)  # raw dates without an entry
    no_raw: list[str] = field(default_factory=list)  # entries whose raw file is gone
    orphans: list[str] = field(default_factory=list)  # Parquet files without an entry
    wall_s: float = 0.0

    @property
    def clean(self) -> bool:
        return not (self.problems or self.not_converted or self.no_raw or self.orphans)


def verify_store(raw_dir: Path, store_dir: Path, *, workers: int = DEFAULT_WORKERS) -> VerifyReport:
    """Compare every store file with its raw file (module docstring)."""
    t0 = time.perf_counter()
    if not (store_dir / MANIFEST_NAME).exists():
        raise DataError(
            f"no store manifest in {store_dir}: run volsto-data convert --vendor {orats.VENDOR}"
        )
    inputs, _ = raw_inputs(raw_dir)
    files = read_manifest(store_dir)["files"]
    rep = VerifyReport(store_dir)
    rep.not_converted = sorted(set(inputs) - set(files))
    rep.no_raw = sorted(set(files) - set(inputs))
    known = {day_path(store_dir, d) for d in files}
    strikes = store_dir / STRIKES_DIR
    rep.orphans = sorted(
        p.relative_to(store_dir).as_posix()
        for p in strikes.rglob("*.parquet")
        if p not in known and not p.name.startswith(".")
    )
    jobs = [
        (str(raw_dir / inputs[d]["file"]), str(day_path(store_dir, d)), d, files[d])
        for d in sorted(set(files) & set(inputs))
    ]
    if workers == 1 or len(jobs) <= 1:
        results = [verify_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            results = list(pool.map(verify_one, jobs, chunksize=1))
    for job, problems in zip(jobs, results):
        rep.checked += 1
        rep.rows += int(job[3]["rows"])
        if problems:
            rep.problems[job[2]] = problems
    rep.wall_s = time.perf_counter() - t0
    return rep


def _some(items: Sequence[str]) -> str:
    return ", ".join(items[:20]) + (f" … and {len(items) - 20} more" if len(items) > 20 else "")


def format_verify(rep: VerifyReport) -> str:
    out = [
        f"store {rep.store_dir}",
        f"  {rep.checked} files compared with their raw files ({rep.rows:,} rows): row counts, "
        f"per-column null counts and sums, then full equality; wall clock {rep.wall_s:.1f} s",
    ]
    if rep.not_converted:
        out.append(f"  FINDING raw dates not converted ({len(rep.not_converted)}): "
                   f"{_some(rep.not_converted)}")  # fmt: skip
    if rep.no_raw:
        out.append(f"  FINDING store dates without a raw file ({len(rep.no_raw)}): "
                   f"{_some(rep.no_raw)}")  # fmt: skip
    if rep.orphans:
        out.append(f"  FINDING Parquet files without a manifest entry ({len(rep.orphans)}): "
                   f"{_some(rep.orphans)}")  # fmt: skip
    for date, problems in list(rep.problems.items())[:40]:
        out.append(f"  FINDING {date}: " + "; ".join(problems[:6]))
    out.append(
        "  clean: every Parquet file is a faithful copy of its raw file"
        if rep.clean
        else "  NOT CLEAN"
    )
    return "\n".join(out)
