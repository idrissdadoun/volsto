"""``volsto-data``: the vendor data layer's command line (SPEC §18, M11).

``status``      the two roots, where each came from, free space, raw files and manifest.
``fetch``       ``aws s3 sync`` of a vendor archive into ``<raw>/<vendor>/``
                (:mod:`volsto.data.fetch`; ``--dry-run`` lists first; no credentials here).
``verify-raw``  hash and scan every raw zip, write the raw manifest, check the trading
                calendar, schema and OPRA columns (:mod:`volsto.data.raw`); exit 1 on any
                finding.

Exit codes: 0 clean, 1 a check found something, 2 a refusal (:class:`DataError`).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from volsto.data import orats
from volsto.data import raw as rawmod
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
        print(f"  store {store_dir}: {'exists' if store_dir.is_dir() else 'absent'}")
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
    )
    path = rawmod.write_manifest(raw_dir, rep.manifest())
    print(rawmod.format_report(rep))
    print(f"  manifest {path}; wall clock {time.perf_counter() - t0:.1f} s")
    return 1 if rep.findings() else 0


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
    except DataError as exc:
        print(f"volsto-data: {exc}")
        return 2
    raise AssertionError(args.command)  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
