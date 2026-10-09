"""PM results package of 2026-10-09: freeze it.

    python scripts/pm_freeze.py [--check]

Writes ``MANIFEST.sha256`` (one line per file of the package: its SHA-256 and its path; the open
files of ``pm_common.FROZEN_OPEN`` and the folders ``later/`` and ``1y/`` are not in it), then
the marker ``FROZEN`` (time, commit, number of files) that makes every builder refuse to write a
frozen file (``pm_common.guard_frozen``), and one line in ``STATUS.md``.  ``--check`` recomputes
the hashes and reports any file that changed, is missing or is new since the freeze.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_common as pc

SKIP_DIRS = ("later", "1y")
SKIP_FILES = (*pc.FROZEN_OPEN, "MANIFEST.sha256", "FROZEN")


def files() -> list[Path]:
    out = []
    for path in sorted(pc.PM.rglob("*")):
        rel = path.relative_to(pc.PM)
        if (
            path.is_file()
            and rel.parts[0] not in SKIP_DIRS
            and rel.name not in SKIP_FILES
            and not rel.name.endswith(".tmp")
        ):
            out.append(path)
    return out


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    manifest = pc.PM / "MANIFEST.sha256"
    if args.check:
        stored = dict(reversed(line.split("  ", 1)) for line in manifest.read_text().splitlines())
        now = {str(p.relative_to(pc.PM)): digest(p) for p in files()}
        changed = sorted(k for k in stored if k in now and now[k] != stored[k])
        missing = sorted(k for k in stored if k not in now)
        new = sorted(k for k in now if k not in stored)
        print(
            f"{len(stored)} files in the manifest: {len(changed)} changed, {len(missing)} missing, {len(new)} new"
        )
        for label, items in (("changed", changed), ("missing", missing), ("new", new)):
            for k in items:
                print(f"  {label}: {k}")
        return 1 if changed or missing else 0
    if (pc.PM / "FROZEN").exists():
        print("already frozen:", (pc.PM / "FROZEN").read_text().strip())
        return 1
    todo = files()
    manifest.write_text("".join(f"{digest(p)}  {p.relative_to(pc.PM)}\n" for p in todo))
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=pc.ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    (pc.PM / "FROZEN").write_text(
        f"frozen {stamp} New York time; builders at commit {commit}; {len(todo)} files in MANIFEST.sha256\n"
    )
    pc.status(
        f"FROZEN: {len(todo)} files listed with their SHA-256 in MANIFEST.sha256 (builders at commit {commit}). From now on: later/ for anything new, ERRATA.md for a correction; 1y/ keeps its own deadline of 13:00."
    )
    print(f"frozen: {len(todo)} files; commit {commit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
