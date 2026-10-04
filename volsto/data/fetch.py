"""``volsto-data fetch``: download a vendor archive with ``aws s3 sync`` (SPEC §18, M11 Part 1).

The code never sees a credential: the owner configures an AWS profile (``aws configure
--profile <name>``) and ``fetch`` passes only the profile's *name* to the AWS CLI, which reads
the credentials itself.  Nothing is read from or written to ``~/.aws``, and the environment is
never printed.  Bucket, prefix and file naming are parameters (they are only known after the
purchase).

Two modes:

* ``--dry-run`` — list the bucket prefix (``aws s3 ls --recursive --summarize``: object count
  and total bytes), compare the total with the free space of the raw volume, then run the
  sync with ``--dryrun`` and count what it would download.  Writes nothing.
* the sync itself — the same listing first; **refused when the volume of the raw root is short**
  of the bytes still to download plus :data:`~volsto.data.roots.FREE_SPACE_MARGIN_BYTES`; then
  ``aws s3 sync s3://<bucket>/<prefix> <raw>/<vendor>/``.  ``sync`` skips objects already
  present with the same size and a local modification time not older than the object's, so an
  interrupted download is resumed by running the same command again.  ``--delete`` is never
  passed and is refused as an extra argument: the raw layer is never modified.

Extra AWS arguments (for example ``--request-payer requester``) go after ``--``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from volsto.data.roots import DataError, fmt_bytes, free_bytes, require_free_space

AWS_INSTALL_HINT = "install the AWS CLI (brew install awscli) and configure a profile"
_FORBIDDEN_EXTRA = ("--delete", "--profile", "--dryrun")
_TOTAL_OBJECTS = re.compile(r"^\s*Total Objects:\s*(\d+)\s*$", re.MULTILINE)
_TOTAL_SIZE = re.compile(r"^\s*Total Size:\s*(\d+)\s*$", re.MULTILINE)


@dataclass(frozen=True)
class Listing:
    """``aws s3 ls --recursive --summarize`` of the source."""

    objects: int
    total_bytes: int
    lines: tuple[str, ...]  # the object lines (date, time, size, key)


def s3_uri(bucket: str, prefix: str) -> str:
    """``s3://bucket/prefix/`` (a prefix names a directory of the bucket; blank is the root)."""
    b = bucket.strip().removeprefix("s3://").strip("/")
    if not b or "/" in b:
        raise DataError(f"--bucket {bucket!r}: give the bucket name alone (the path is --prefix)")
    p = prefix.strip().strip("/")
    return f"s3://{b}/{p}/" if p else f"s3://{b}/"


def find_aws(aws_bin: str = "aws") -> str:
    found = shutil.which(aws_bin)
    if found is None:
        raise DataError(f"{aws_bin!r} not found on PATH: {AWS_INSTALL_HINT}")
    return found


def check_extra(extra: Sequence[str]) -> list[str]:
    for arg in extra:
        if arg.split("=")[0] in _FORBIDDEN_EXTRA:
            raise DataError(
                f"extra AWS argument {arg!r} is refused (fetch sets the profile and the dry run "
                "itself, and never deletes: the raw layer is never modified)"
            )
    return list(extra)


def list_argv(aws: str, uri: str, profile: str, extra: Sequence[str]) -> list[str]:
    return [aws, "s3", "ls", uri, "--recursive", "--summarize", "--profile", profile, *extra]


def sync_argv(
    aws: str,
    uri: str,
    dest: Path,
    profile: str,
    *,
    include: Sequence[str] = (),
    dry_run: bool = False,
    extra: Sequence[str] = (),
) -> list[str]:
    argv = [aws, "s3", "sync", uri, str(dest), "--profile", profile, "--no-progress"]
    if include:
        argv += ["--exclude", "*"]
        for pat in include:
            argv += ["--include", pat]
    if dry_run:
        argv.append("--dryrun")
    return [*argv, *extra]


def parse_listing(text: str) -> Listing:
    mo, ms = _TOTAL_OBJECTS.search(text), _TOTAL_SIZE.search(text)
    if mo is None or ms is None:
        raise DataError(
            "could not read 'Total Objects' / 'Total Size' from the aws s3 ls output; "
            "the listing is not trusted and nothing was downloaded"
        )
    lines = tuple(
        ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("Total ")
    )
    return Listing(int(mo.group(1)), int(ms.group(1)), lines)


def list_source(aws: str, uri: str, profile: str, extra: Sequence[str] = ()) -> Listing:
    """Run the listing; raises :class:`DataError` when the AWS CLI fails (its own message —
    which never contains the secret — goes to stderr untouched)."""
    proc = subprocess.run(list_argv(aws, uri, profile, extra), stdout=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise DataError(
            f"aws s3 ls {uri} failed (exit {proc.returncode}); check the profile "
            f"{profile!r}, the bucket and the prefix"
        )
    return parse_listing(proc.stdout)


def local_bytes(dest: Path) -> int:
    """Bytes already under ``dest`` (regular files, recursively)."""
    if not dest.is_dir():
        return 0
    return sum(p.stat().st_size for p in dest.rglob("*") if p.is_file())


def fetch(
    dest: Path,
    *,
    profile: str,
    bucket: str,
    prefix: str,
    include: Sequence[str] = (),
    dry_run: bool = False,
    extra: Sequence[str] = (),
    aws_bin: str = "aws",
    out: list[str] | None = None,
) -> int:
    """Run the dry run or the sync into ``dest`` (module docstring); returns the AWS CLI's exit
    code.  Progress lines are printed, and appended to ``out`` when given."""

    def say(line: str) -> None:
        print(line, flush=True)
        if out is not None:
            out.append(line)

    if not profile.strip():
        raise DataError("--profile is required: the name of the AWS profile you configured")
    extra = check_extra(extra)
    aws = find_aws(aws_bin)
    uri = s3_uri(bucket, prefix)
    listing = list_source(aws, uri, profile, extra)
    present = local_bytes(dest)
    needed = max(listing.total_bytes - present, 0)
    say(
        f"source {uri}: {listing.objects} objects, {listing.total_bytes:,} bytes "
        f"({fmt_bytes(listing.total_bytes)})"
    )
    for ln in listing.lines[:3]:
        say(f"  {ln}")
    if len(listing.lines) > 6:
        say(f"  … {len(listing.lines) - 6} more …")
    for ln in listing.lines[max(len(listing.lines) - 3, 3) :]:
        say(f"  {ln}")
    say(
        f"destination {dest}: {fmt_bytes(present)} present, about {fmt_bytes(needed)} to "
        f"download, {fmt_bytes(free_bytes(dest))} free"
    )
    if dry_run:
        argv = sync_argv(aws, uri, dest, profile, include=include, dry_run=True, extra=extra)
        say("dry run: " + " ".join(argv))
        proc = subprocess.run(argv, stdout=subprocess.PIPE, text=True)
        n = sum(1 for ln in proc.stdout.splitlines() if "download:" in ln)
        say(f"dry run: {n} objects would be downloaded (exit {proc.returncode}); nothing written")
        try:
            require_free_space(dest, needed, what="the download")
        except DataError as exc:
            say(f"WARNING: the sync would be refused — {exc}")
        return proc.returncode
    require_free_space(dest, needed, what="the download")
    dest.mkdir(parents=True, exist_ok=True)
    argv = sync_argv(aws, uri, dest, profile, include=include, extra=extra)
    say("sync: " + " ".join(argv))
    code = subprocess.run(argv).returncode
    after = local_bytes(dest)
    say(f"sync exit {code}: {fmt_bytes(after)} under {dest}")
    if code != 0:
        say("the sync did not finish: run the same command again to resume")
    return code
