"""PM results package of 2026-10-09: tidy ``STATUS.md`` to one timestamped line per step.

    python scripts/pm_status_tidy.py [--base <pm_update or pm_update/1y>] [--dry-run]

The builders append one line per run, so a step rerun while it was being built appears several
times.  This keeps, for each group of lines that differ only by their time and their digits (the
same step reported again), the LAST one, in place; every other line is kept as it is.  The
removed lines are written to ``STATUS_removed.md`` beside the file, so nothing is lost.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_common as pc

LINE = re.compile(r"^- (\d\d:\d\d) (.*)$")


def tidy(text: str) -> tuple[str, list[str]]:
    lines = text.splitlines()
    keys = []
    for line in lines:
        m = LINE.match(line)
        keys.append(None if m is None else re.sub(r"\d+", "#", m.group(2))[:160])
    last = {key: i for i, key in enumerate(keys) if key is not None}
    kept, removed = [], []
    for i, (line, key) in enumerate(zip(lines, keys, strict=True)):
        (kept if key is None or last[key] == i else removed).append(line)
    return "\n".join(kept) + "\n", removed


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--base", default=str(pc.PM))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    path = Path(args.base) / "STATUS.md"
    text, removed = tidy(path.read_text())
    print(f"{path}: {len(removed)} repeated lines to remove, {len(text.splitlines())} kept")
    if not args.dry_run and removed:
        with (path.parent / "STATUS_removed.md").open("a") as fh:
            fh.write("\n".join(removed) + "\n")
        path.write_text(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
