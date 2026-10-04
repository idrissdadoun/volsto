"""``volsto-data verify-raw``: the raw manifest and the trading-calendar check (SPEC §18, M11
Part 1).

The raw layer is the vendor's zips exactly as delivered: this module only reads them.  Each
zip is hashed and its CSV member is streamed from inside the zip (nothing is unzipped to
disk), and one entry per file goes to the raw manifest ``<raw>/<vendor>/raw_manifest.json``:

``file`` (path relative to the vendor's raw directory), ``bytes``, ``mtime_ns``, ``sha256``
(of the zip), ``trade_date`` (from the file name), ``member`` and ``csv_bytes``, ``columns``
(the header), ``schema_version`` (``None`` on drift), ``rows``, and ``opra``: whether both
OPRA columns exist, how many rows leave ``cOpra`` / ``pOpra`` empty, and the same on the
``SPX`` rows.

Checks, none of which passes silently (:func:`findings` lists every one and the CLI exits 1
when any is present):

* **calendar** — every trading day from the first file to the last has exactly one file.  The
  trading calendar is the ``date`` column of ``data/history/SPX.csv``
  (:data:`DEFAULT_CALENDAR`).  Missing dates, unexpected dates (a file on a day the calendar
  does not list), dates with more than one file and files dated beyond the calendar's last
  date are listed.
* **names and content** — a zip the pattern does not match; a zip that cannot be read or does
  not hold exactly one member; a ``trade_date`` column that is absent, holds more than one
  value or disagrees with the file name.
* **schema drift** — a header that is not a known schema version, with the difference named.
* **OPRA dependency** — a file whose ``cOpra`` or ``pOpra`` is missing or empty on an ``SPX``
  row (decision 8), and the coverage per year (:func:`opra_coverage_by_year`).

An entry whose size and modification time are unchanged is reused from the existing manifest
(``reused``); ``rehash=True`` reads every file again.  The manifest is replaced atomically.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import re
import zipfile
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pcsv

from volsto.data import orats
from volsto.data.roots import REPO_ROOT, DataError

MANIFEST_NAME = "raw_manifest.json"
MANIFEST_VERSION = 1
#: The trading calendar: the dates of the SPX close history (owner's instruction, M11 Part 1).
DEFAULT_CALENDAR = REPO_ROOT / "data" / "history" / "SPX.csv"
#: Processes hashing and scanning files (bounded; each streams one zip in 16 MiB blocks).
DEFAULT_WORKERS: int = 8
_CSV_BLOCK = 1 << 24
_HASH_BLOCK = 1 << 20


@dataclass(frozen=True)
class RawFile:
    """One raw zip as scanned (module docstring).  ``error`` is set when the zip could not be
    read as one CSV member; the content fields are then empty."""

    file: str
    bytes: int
    mtime_ns: int
    sha256: str
    trade_date: str | None  # ISO, from the file name; None when the name does not match
    member: str | None = None
    csv_bytes: int | None = None
    columns: tuple[str, ...] = ()
    schema_version: int | None = None
    rows: int | None = None
    content_trade_dates: tuple[str, ...] = ()  # distinct ``trade_date`` texts in the CSV
    opra_columns_present: bool = False
    opra_c_empty: int | None = None
    opra_p_empty: int | None = None
    spx_rows: int | None = None
    spx_opra_empty: int | None = None  # SPX rows with either symbol empty
    error: str | None = None

    @property
    def opra_ok_for_spx(self) -> bool:
        """Both OPRA columns exist and no ``SPX`` row leaves either empty."""
        return self.opra_columns_present and self.spx_opra_empty == 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(_HASH_BLOCK):
            h.update(block)
    return h.hexdigest()


def read_header(zf: zipfile.ZipFile, member: str) -> tuple[str, ...]:
    """The CSV header of ``member`` (first line; the files are unquoted)."""
    with zf.open(member) as fh:
        line = fh.readline().decode("utf-8").rstrip("\r\n")
    return tuple(line.split(","))


def scan_zip(path: Path, rel: str, pattern: str) -> RawFile:
    """Hash ``path`` and stream its CSV member: header, row count, ``trade_date`` values and
    OPRA-column population.  Never raises on a bad file: the reason goes to ``error``."""
    st = path.stat()
    date = orats.file_trade_date(path.name, pattern)
    base: dict[str, Any] = {
        "file": rel,
        "bytes": st.st_size,
        "mtime_ns": st.st_mtime_ns,
        "sha256": sha256_file(path),
        "trade_date": date.isoformat() if date else None,
    }
    try:
        with zipfile.ZipFile(path) as zf:
            infos = [i for i in zf.infolist() if not i.is_dir()]
            if len(infos) != 1:
                names = [i.filename for i in infos]
                return RawFile(**base, error=f"expected one member, found {len(infos)}: {names}")
            info = infos[0]
            columns = read_header(zf, info.filename)
            base.update(
                member=info.filename,
                csv_bytes=info.file_size,
                columns=columns,
                schema_version=orats.schema_version_of(columns),
            )
            want = [c for c in ("ticker", *orats.OPRA_COLUMNS, "trade_date") if c in columns]
            has_opra = all(c in columns for c in orats.OPRA_COLUMNS)
            rows = c_empty = p_empty = spx_rows = spx_empty = 0
            dates: set[str] = set()
            with zf.open(info.filename) as fh:
                reader = pcsv.open_csv(
                    fh,
                    read_options=pcsv.ReadOptions(block_size=_CSV_BLOCK),
                    convert_options=pcsv.ConvertOptions(
                        include_columns=want,
                        column_types={c: pa.string() for c in want},
                        strings_can_be_null=False,
                    ),
                )
                for batch in reader:
                    rows += batch.num_rows
                    if "trade_date" in want:
                        dates.update(pc.unique(batch.column("trade_date")).to_pylist())
                    if not has_opra:
                        if "ticker" in want:
                            spx_rows += _count(
                                pc.equal(batch.column("ticker"), orats.OPRA_REQUIRED_TICKER)
                            )
                        continue
                    ce = pc.equal(pc.utf8_trim_whitespace(batch.column("cOpra")), "")
                    pe = pc.equal(pc.utf8_trim_whitespace(batch.column("pOpra")), "")
                    c_empty += _count(ce)
                    p_empty += _count(pe)
                    if "ticker" in want:
                        spx = pc.equal(batch.column("ticker"), orats.OPRA_REQUIRED_TICKER)
                        spx_rows += _count(spx)
                        spx_empty += _count(pc.and_(spx, pc.or_(ce, pe)))
    except (zipfile.BadZipFile, pa.ArrowException, OSError, UnicodeDecodeError) as exc:
        return RawFile(**base, error=f"{type(exc).__name__}: {exc}")
    return RawFile(
        **base,
        rows=rows,
        content_trade_dates=tuple(sorted(dates)),
        opra_columns_present=has_opra,
        opra_c_empty=c_empty if has_opra else None,
        opra_p_empty=p_empty if has_opra else None,
        spx_rows=spx_rows if "ticker" in want else None,
        spx_opra_empty=spx_empty if has_opra and "ticker" in want else None,
    )


def _count(mask: Any) -> int:
    return int(pc.sum(mask).as_py() or 0)


def _scan_job(job: tuple[str, str, str]) -> RawFile:
    path, rel, pattern = job
    return scan_zip(Path(path), rel, pattern)


def load_calendar(path: Path) -> list[_dt.date]:
    """The sorted trading dates of a ``date,close`` CSV (ISO dates in the first column)."""
    if not path.exists():
        raise DataError(
            f"trading calendar {path} not found (verify-raw checks the files against the dates "
            "of the SPX close history; pass --calendar)"
        )
    lines = path.read_text().splitlines()
    if not lines or lines[0].split(",")[0].strip().lower() != "date":
        raise DataError(f"{path}: expected a CSV whose first column is 'date'")
    dates = sorted({_dt.date.fromisoformat(ln.split(",")[0].strip()) for ln in lines[1:] if ln})
    if not dates:
        raise DataError(f"{path}: no dates")
    return dates


@dataclass
class RawReport:
    """Everything ``verify-raw`` found (module docstring)."""

    vendor: str
    raw_dir: Path
    pattern: str
    calendar_path: Path
    calendar_last: str
    files: list[RawFile]
    reused: int
    unmatched: list[str] = field(default_factory=list)  # zips the pattern does not match
    missing: list[str] = field(default_factory=list)
    unexpected: list[str] = field(default_factory=list)
    duplicated: dict[str, list[str]] = field(default_factory=dict)
    beyond_calendar: list[str] = field(default_factory=list)
    unreadable: dict[str, str] = field(default_factory=dict)
    date_mismatch: dict[str, str] = field(default_factory=dict)
    drift: dict[str, str] = field(default_factory=dict)
    opra_missing: dict[str, str] = field(default_factory=dict)

    @property
    def first(self) -> str | None:
        ds = [f.trade_date for f in self.files if f.trade_date]
        return min(ds) if ds else None

    @property
    def last(self) -> str | None:
        ds = [f.trade_date for f in self.files if f.trade_date]
        return max(ds) if ds else None

    def findings(self) -> dict[str, int]:
        """Count per kind of finding; empty when the raw layer is clean."""
        kinds: dict[str, int] = {
            "no files": 0 if self.files else 1,
            "files the pattern does not match": len(self.unmatched),
            "missing trading days": len(self.missing),
            "unexpected dates": len(self.unexpected),
            "dates with more than one file": len(self.duplicated),
            "files beyond the calendar": len(self.beyond_calendar),
            "unreadable files": len(self.unreadable),
            "trade_date mismatches": len(self.date_mismatch),
            "schema drift": len(self.drift),
            "files without OPRA symbols on SPX": len(self.opra_missing),
        }
        return {k: v for k, v in kinds.items() if v}

    def opra_coverage_by_year(self) -> list[dict[str, Any]]:
        """Per year of trade date: files, files whose OPRA columns exist, files usable for SPX,
        rows, rows with an empty ``cOpra`` / ``pOpra``, SPX rows and SPX rows without a symbol."""
        years: dict[str, dict[str, Any]] = {}
        for f in self.files:
            if f.trade_date is None or f.error is not None:
                continue
            y = years.setdefault(
                f.trade_date[:4],
                {
                    "year": f.trade_date[:4],
                    "files": 0,
                    "opra_columns": 0,
                    "opra_ok_spx": 0,
                    "rows": 0,
                    "c_empty": 0,
                    "p_empty": 0,
                    "spx_rows": 0,
                    "spx_opra_empty": 0,
                },
            )
            y["files"] += 1
            y["opra_columns"] += int(f.opra_columns_present)
            y["opra_ok_spx"] += int(f.opra_ok_for_spx)
            y["rows"] += f.rows or 0
            y["c_empty"] += f.opra_c_empty or 0
            y["p_empty"] += f.opra_p_empty or 0
            y["spx_rows"] += f.spx_rows or 0
            y["spx_opra_empty"] += f.spx_opra_empty or 0
        return [years[k] for k in sorted(years)]

    def manifest(self) -> dict[str, Any]:
        return {
            "version": MANIFEST_VERSION,
            "vendor": self.vendor,
            "pattern": self.pattern,
            "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "calendar": {"path": str(self.calendar_path), "last": self.calendar_last},
            "first": self.first,
            "last": self.last,
            "n_files": len(self.files),
            "total_bytes": sum(f.bytes for f in self.files),
            "findings": {
                "unmatched": self.unmatched,
                "missing": self.missing,
                "unexpected": self.unexpected,
                "duplicated": self.duplicated,
                "beyond_calendar": self.beyond_calendar,
                "unreadable": self.unreadable,
                "date_mismatch": self.date_mismatch,
                "drift": self.drift,
                "opra_missing": self.opra_missing,
            },
            "opra_coverage_by_year": self.opra_coverage_by_year(),
            "files": [asdict(f) for f in self.files],
        }


def read_manifest(raw_dir: Path) -> dict[str, Any] | None:
    """The stored raw manifest of a vendor's raw directory, ``None`` when there is none."""
    p = raw_dir / MANIFEST_NAME
    if not p.exists():
        return None
    data: dict[str, Any] = json.loads(p.read_text())
    if data.get("version") != MANIFEST_VERSION:
        raise DataError(f"{p}: manifest version {data.get('version')!r}, expected 1")
    return data


def _entry_from_manifest(d: dict[str, Any]) -> RawFile:
    d = dict(d)
    d["columns"] = tuple(d["columns"])
    d["content_trade_dates"] = tuple(d["content_trade_dates"])
    return RawFile(**d)


def write_manifest(raw_dir: Path, manifest: dict[str, Any]) -> Path:
    """Replace ``raw_manifest.json`` atomically (temporary file in the same directory)."""
    p = raw_dir / MANIFEST_NAME
    tmp = raw_dir / f".{MANIFEST_NAME}.{os.getpid()}.tmp"
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
    return p


def verify_raw(
    raw_dir: Path,
    *,
    vendor: str = orats.VENDOR,
    pattern: str = orats.DEFAULT_FILE_PATTERN,
    calendar: Path = DEFAULT_CALENDAR,
    workers: int = DEFAULT_WORKERS,
    rehash: bool = False,
) -> RawReport:
    """Scan every ``*.zip`` below ``raw_dir`` (recursively: a sync keeps the vendor's prefix
    layout) and run the checks of the module docstring.  Does not write: the caller stores
    :meth:`RawReport.manifest` with :func:`write_manifest`."""
    if not raw_dir.is_dir():
        raise DataError(f"raw directory {raw_dir} does not exist (run volsto-data fetch first)")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise DataError(f"--pattern {pattern!r} is not a regular expression: {exc}") from exc
    if workers < 1:
        raise DataError("workers must be at least 1")
    cal = load_calendar(calendar)
    paths = sorted(p for p in raw_dir.rglob("*.zip") if not p.name.startswith("."))
    previous: dict[str, RawFile] = {}
    if not rehash:
        old = read_manifest(raw_dir)
        if old is not None and old.get("pattern") == pattern:
            previous = {e["file"]: _entry_from_manifest(e) for e in old["files"]}
    files: list[RawFile] = []
    jobs: list[tuple[str, str, str]] = []
    for p in paths:
        rel = p.relative_to(raw_dir).as_posix()
        st = p.stat()
        prev = previous.get(rel)
        if prev is not None and (prev.bytes, prev.mtime_ns) == (st.st_size, st.st_mtime_ns):
            files.append(prev)
        else:
            jobs.append((str(p), rel, pattern))
    reused = len(files)
    if len(jobs) <= 1 or workers == 1:
        files.extend(_scan_job(j) for j in jobs)
    else:
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as pool:
            files.extend(pool.map(_scan_job, jobs, chunksize=1))
    files.sort(key=lambda f: f.file)
    rep = RawReport(vendor, raw_dir, pattern, calendar, cal[-1].isoformat(), files, reused)
    _check(rep, cal)
    return rep


def _check(rep: RawReport, cal: Sequence[_dt.date]) -> None:
    by_date: dict[str, list[str]] = {}
    for f in rep.files:
        if f.trade_date is None:
            rep.unmatched.append(f.file)
        else:
            by_date.setdefault(f.trade_date, []).append(f.file)
        if f.error is not None:
            rep.unreadable[f.file] = f.error
            continue
        if f.schema_version is None:
            rep.drift[f.file] = orats.describe_drift(f.columns)
        if f.trade_date is not None:
            problem = _date_problem(f)
            if problem:
                rep.date_mismatch[f.file] = problem
        if not f.opra_ok_for_spx:
            if not f.opra_columns_present:
                absent = [c for c in orats.OPRA_COLUMNS if c not in f.columns]
                rep.opra_missing[f.file] = f"columns absent: {absent}"
            else:
                rep.opra_missing[f.file] = (
                    f"{f.spx_opra_empty} of {f.spx_rows} SPX rows without an OPRA symbol"
                )
    rep.duplicated = {d: fs for d, fs in sorted(by_date.items()) if len(fs) > 1}
    if not by_date:
        return
    first = _dt.date.fromisoformat(min(by_date))
    last = _dt.date.fromisoformat(max(by_date))
    cal_set = set(cal)
    expected = [d for d in cal if first <= d <= last]
    rep.missing = [d.isoformat() for d in expected if d.isoformat() not in by_date]
    for d in sorted(by_date):
        day = _dt.date.fromisoformat(d)
        if day > cal[-1]:
            rep.beyond_calendar.append(d)
        elif day not in cal_set:
            rep.unexpected.append(d)


def _date_problem(f: RawFile) -> str | None:
    if "trade_date" not in f.columns:
        return "no trade_date column"
    if len(f.content_trade_dates) != 1:
        return f"{len(f.content_trade_dates)} distinct trade_date values: " + ", ".join(
            f.content_trade_dates[:5]
        )
    text = f.content_trade_dates[0]
    try:
        inside = orats.parse_date(text).isoformat()
    except ValueError:
        return f"trade_date {text!r} is not {orats.DATE_FORMAT}"
    if inside != f.trade_date:
        return f"the file name says {f.trade_date}, the trade_date column {inside}"
    return None


def _ranges(dates: Sequence[str]) -> str:
    """ISO dates as a compact list (at most 40 shown)."""
    shown = list(dates[:40])
    more = f" … and {len(dates) - 40} more" if len(dates) > 40 else ""
    return ", ".join(shown) + more


def format_report(rep: RawReport) -> str:
    """The text ``verify-raw`` prints: totals, every finding, OPRA coverage per year."""
    total = sum(f.bytes for f in rep.files)
    rows = sum(f.rows or 0 for f in rep.files)
    versions = Counter(f.schema_version for f in rep.files if f.error is None)
    out = [
        f"raw {rep.vendor}: {rep.raw_dir}",
        f"  files {len(rep.files)} ({rep.reused} reused from the manifest, "
        f"{len(rep.files) - rep.reused} hashed), {total:,} bytes, {rows:,} rows",
        f"  trade dates {rep.first} .. {rep.last}; calendar {rep.calendar_path} "
        f"(to {rep.calendar_last})",
        "  schema versions: "
        + (", ".join(f"v{v}: {n}" if v else f"unknown: {n}" for v, n in versions.items()) or "-"),
    ]
    listed: list[tuple[str, Sequence[str]]] = [
        ("missing trading days", rep.missing),
        ("unexpected dates (not in the calendar)", rep.unexpected),
        ("files dated beyond the calendar (extend the calendar file)", rep.beyond_calendar),
        ("files the pattern does not match", rep.unmatched),
    ]
    for title, items in listed:
        if items:
            out.append(f"  FINDING {title} ({len(items)}): {_ranges(items)}")
    mapped: list[tuple[str, dict[str, Any]]] = [
        ("dates with more than one file", rep.duplicated),
        ("unreadable files", rep.unreadable),
        ("trade_date mismatches", rep.date_mismatch),
        ("schema drift", rep.drift),
        ("files without OPRA symbols on SPX", rep.opra_missing),
    ]
    for title, items_d in mapped:
        if items_d:
            out.append(f"  FINDING {title} ({len(items_d)}):")
            for k, v in list(items_d.items())[:40]:
                out.append(f"    {k}: {v}")
            if len(items_d) > 40:
                out.append(f"    … and {len(items_d) - 40} more (all in the manifest)")
    out.append("  OPRA coverage per year (files | with OPRA columns | usable for SPX | rows |")
    out.append("    rows with empty cOpra | empty pOpra | SPX rows | SPX rows without a symbol):")
    for y in rep.opra_coverage_by_year():
        out.append(
            f"    {y['year']}: {y['files']} | {y['opra_columns']} | {y['opra_ok_spx']} | "
            f"{y['rows']:,} | {y['c_empty']:,} | {y['p_empty']:,} | {y['spx_rows']:,} | "
            f"{y['spx_opra_empty']:,}"
        )
    found = rep.findings()
    if found:
        out.append("  NOT CLEAN: " + "; ".join(f"{v} x {k}" for k, v in found.items()))
    else:
        out.append("  clean: one file per trading day, known schema, OPRA symbols on every SPX row")
    return "\n".join(out)
