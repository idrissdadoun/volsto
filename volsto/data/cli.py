"""``volsto-data``: the vendor data layer's command line (SPEC §18, M11).

``status``      the two roots, where each came from, free space, raw files and manifest.
``fetch``       ``aws s3 sync`` of a vendor archive into ``<raw>/<vendor>/``
                (:mod:`volsto.data.fetch`; ``--dry-run`` lists first; no credentials here).
``verify-raw``  hash and scan every raw zip (year folders included; other files ignored),
                write the raw manifest, check the trading calendar and the schema, report
                the OPRA columns (:mod:`volsto.data.raw`); exit 1 on any finding.
``convert``     raw zips → one typed Parquet file per trading day (:mod:`volsto.data.store`).
``verify``      every Parquet file against its raw file: rows, null counts, sums, equality.
``extract``     one Parquet file per ticker across all dates (:mod:`volsto.data.extract`).
``sql``         a DuckDB query over the store (optional; the ``data`` extra).

Exit codes: 0 clean, 1 a check found something, 2 a refusal (:class:`DataError`).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from volsto.data import extract as extractmod
from volsto.data import orats
from volsto.data import raw as rawmod
from volsto.data import sql as sqlmod
from volsto.data import store as storemod
from volsto.data.fetch import fetch
from volsto.data.roots import (
    ENV_RAW,
    ENV_STORE,
    DataError,
    DataRoots,
    fmt_bytes,
    free_bytes,
)

log = logging.getLogger(__name__)
VENDORS: tuple[str, ...] = (orats.VENDOR,)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="volsto-data",
        description="Vendor data layer: fetch, verify and convert vendor option data",
    )
    ap.add_argument("--raw", default=None, help=f"raw root (env {ENV_RAW}; default data/raw)")
    ap.add_argument(
        "--store", default=None, help=f"store root (env {ENV_STORE}; default data/store)"
    )
    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("status", help="roots, free space, raw files and manifests")

    fp = sub.add_parser("fetch", help="download a vendor archive with aws s3 sync")
    fp.add_argument("--vendor", required=True, choices=VENDORS)
    fp.add_argument("--profile", required=True, help="name of the AWS profile you configured")
    fp.add_argument("--bucket", required=True)
    fp.add_argument("--prefix", required=True, help="key prefix inside the bucket ('' for none)")
    fp.add_argument(
        "--include",
        action="append",
        default=[],
        help="sync only keys matching this glob (repeatable), e.g. '*.zip'",
    )
    fp.add_argument("--dry-run", action="store_true", help="list and count; write nothing")
    fp.add_argument("aws_args", nargs="*", help="extra AWS CLI arguments, after --")

    vp = sub.add_parser("verify-raw", help="raw manifest and trading-calendar check")
    vp.add_argument("--vendor", required=True, choices=VENDORS)
    vp.add_argument(
        "--pattern",
        default=orats.DEFAULT_FILE_PATTERN,
        help="regular expression of a raw file name; group 1 is the trade date YYYYMMDD",
    )
    vp.add_argument(
        "--calendar",
        default=str(rawmod.DEFAULT_CALENDAR),
        help="date,close CSV whose dates are the trading calendar",
    )
    vp.add_argument("--workers", type=int, default=rawmod.DEFAULT_WORKERS)
    vp.add_argument(
        "--rehash",
        action="store_true",
        help="read every file again (default: reuse manifest entries of unchanged size and mtime)",
    )
    vp.add_argument(
        "--against",
        default=None,
        help="a raw manifest to compare with by name and sha256 (checks a second copy; "
        "implies --rehash)",
    )
    cp = sub.add_parser("convert", help="raw zips -> one typed Parquet file per trading day")
    cp.add_argument("--vendor", required=True, choices=VENDORS)
    cp.add_argument("--workers", type=int, default=storemod.DEFAULT_WORKERS)
    cp.add_argument("--force", action="store_true", help="convert again files that are up to date")

    yp = sub.add_parser("verify", help="compare every Parquet file with its raw file")
    yp.add_argument("--vendor", required=True, choices=VENDORS)
    yp.add_argument("--workers", type=int, default=storemod.DEFAULT_WORKERS)

    ep = sub.add_parser("extract", help="one Parquet file per ticker across all dates")
    ep.add_argument("--vendor", default=orats.VENDOR, choices=VENDORS)
    ep.add_argument("--tickers", required=True, help="comma-separated, e.g. SPX,AAPL")

    qp = sub.add_parser("sql", help="a DuckDB query over the store (view: strikes)")
    qp.add_argument("--vendor", default=orats.VENDOR, choices=VENDORS)
    qp.add_argument("--max-rows", type=int, default=sqlmod.DEFAULT_MAX_ROWS)
    qp.add_argument("query")
    return ap


def cmd_status(roots: DataRoots) -> int:
    for name, root, source, var in (
        ("raw", roots.raw, roots.raw_source, ENV_RAW),
        ("store", roots.store, roots.store_source, ENV_STORE),
    ):
        state = "exists" if root.is_dir() else "absent"
        print(f"{name:5s} {root} ({source}; {var}) — {state}, {fmt_bytes(free_bytes(root))} free")
    for vendor in VENDORS:
        raw_dir = roots.raw_dir(vendor)
        zips = sorted(raw_dir.rglob("*.zip")) if raw_dir.is_dir() else []
        size = sum(p.stat().st_size for p in zips)
        print(f"{vendor}: {len(zips)} raw zips, {fmt_bytes(size)} in {raw_dir}")
        man = rawmod.read_manifest(raw_dir) if raw_dir.is_dir() else None
        if man is None:
            print(f"  no raw manifest: volsto-data verify-raw --vendor {vendor}")
        else:
            found = {k: len(v) for k, v in man["findings"].items() if v}
            print(
                f"  raw manifest of {man['created_utc']}: {man['n_files']} files, "
                f"{man['first']} .. {man['last']}, findings {found or 'none'}"
            )
            if man["n_files"] != len(zips):
                print(f"  the manifest is stale: volsto-data verify-raw --vendor {vendor}")
        store_dir = roots.store_dir(vendor)
        if not (store_dir / storemod.MANIFEST_NAME).exists():
            print(f"  store {store_dir}: no manifest (volsto-data convert --vendor {vendor})")
            continue
        files = storemod.read_manifest(store_dir)["files"]
        size = sum(e["bytes"] for e in files.values())
        span = f"{min(files)} .. {max(files)}" if files else "-"
        print(f"  store {store_dir}: {len(files)} days, {span}, {fmt_bytes(size)}")
        raw_dates = {e["trade_date"] for e in (man or {}).get("files", []) if e["trade_date"]}
        if raw_dates - set(files):
            print(
                f"    {len(raw_dates - set(files))} raw dates not converted: "
                f"volsto-data convert --vendor {vendor}"
            )
        for ticker, e in extractmod.read_manifest(store_dir)["tickers"].items():
            state = extractmod.extract_state(store_dir, ticker)
            hint = "" if state == "current" else f": volsto-data extract --tickers {ticker}"
            print(f"    extract {ticker}: {e['rows']:,} rows, {e['n_dates']} dates, {state}{hint}")
    return 0


def cmd_verify_raw(roots: DataRoots, args: argparse.Namespace) -> int:
    raw_dir = roots.raw_dir(args.vendor)
    t0 = time.perf_counter()
    rep = rawmod.verify_raw(
        raw_dir,
        vendor=args.vendor,
        pattern=args.pattern,
        calendar=Path(args.calendar),
        workers=args.workers,
        rehash=args.rehash,
        against=Path(args.against) if args.against else None,
    )
    path = rawmod.write_manifest(raw_dir, rep.manifest())
    print(rawmod.format_report(rep))
    print(f"  manifest {path}; wall clock {time.perf_counter() - t0:.1f} s")
    return 1 if rep.findings() else 0


def cmd_convert(roots: DataRoots, args: argparse.Namespace) -> int:
    rep = storemod.convert(
        roots.raw_dir(args.vendor),
        roots.store_dir(args.vendor),
        workers=args.workers,
        force=args.force,
    )
    print(storemod.format_convert(rep))
    return 0 if rep.clean else 1


def cmd_verify(roots: DataRoots, args: argparse.Namespace) -> int:
    rep = storemod.verify_store(
        roots.raw_dir(args.vendor), roots.store_dir(args.vendor), workers=args.workers
    )
    print(storemod.format_verify(rep))
    return 0 if rep.clean else 1


def cmd_extract(roots: DataRoots, args: argparse.Namespace) -> int:
    rep = extractmod.extract(roots.store_dir(args.vendor), args.tickers)
    for ticker, e in rep.tickers.items():
        print(
            f"{ticker}: {e['rows']:,} rows on {e['n_dates']} dates ({e['first']} .. {e['last']}), "
            f"{fmt_bytes(e['bytes'])} -> {extractmod.ticker_path(rep.store_dir, ticker)}"
        )
    for ticker in rep.missing:
        print(f"FAILED {ticker}: on no date of the store (no file written)")
    print(f"  {rep.dates} store dates read; wall clock {rep.wall_s:.1f} s")
    return 1 if rep.missing else 0


def cmd_sql(roots: DataRoots, args: argparse.Namespace) -> int:
    frame = sqlmod.run_sql(roots.store_dir(args.vendor), args.query)
    print(frame.head(args.max_rows).to_string(index=False))
    if len(frame) > args.max_rows:
        print(f"… {len(frame) - args.max_rows} more rows (--max-rows)")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        roots = DataRoots.resolve(raw=args.raw, store=args.store)
        if args.command == "status":
            return cmd_status(roots)
        if args.command == "fetch":
            return fetch(
                roots.raw_dir(args.vendor),
                profile=args.profile,
                bucket=args.bucket,
                prefix=args.prefix,
                include=args.include,
                dry_run=args.dry_run,
                extra=args.aws_args,
            )
        if args.command == "verify-raw":
            return cmd_verify_raw(roots, args)
        if args.command == "convert":
            return cmd_convert(roots, args)
        if args.command == "verify":
            return cmd_verify(roots, args)
        if args.command == "extract":
            return cmd_extract(roots, args)
        if args.command == "sql":
            return cmd_sql(roots, args)
    except DataError as exc:
        print(f"volsto-data: {exc}")
        return 2
    raise AssertionError(args.command)  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
