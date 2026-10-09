"""PM results package of 2026-10-09: assemble ``NUMBERS.md`` and ``numbers.json`` from the parts
the builders wrote (``scripts/pm_common.py``: one ``parts/<name>.json`` of records and one
``parts/<name>.md`` of tables per builder).

    python scripts/pm_assemble.py [--base <pm_update or pm_update/1y>] [--title ...]

The sections come in the owner's order; a part that is not there yet is written as PENDING with
the file it waits for — never as a number.  ``numbers.json`` is the concatenation of the parts'
records (ids unique across the package) with the package's own header.  The script only reads
``parts/`` and writes the two files: it changes no number.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_common as pc

log = logging.getLogger("pm_assemble")

#: (part name, heading) in the order of the owner's brief.
SECTIONS = (
    ("R_reader_notes", "How to read this package, and what not to quote"),
    ("A_today", "A. Today (2026-10-02), 3m, production budget"),
    (
        "A_check_e",
        "A9. Check (e): why the cross-dependent volatility moves the forward less than the wing estimate",
    ),
    ("B_reference", "B. The three reference dates, 3m, production budget"),
    ("C_history", "C. History, 3m, development budget, decisions 1-2 on"),
    ("C_check_c", "C (continued). Check (c)"),
    ("C_arith", "C (continued). The report's arithmetic at the LC price"),
    ("D_stratified", "D. Stratified cross-dependent volatility (check f)"),
    ("F_figures", "F. Figures"),
    ("V_validation", "V. Validation status and caveats"),
    ("Z_findings", "Findings"),
)


def head_commit() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        cwd=pc.ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip()


def assemble(base: Path, title: str, intro: str) -> tuple[int, list[str]]:
    parts = base / "parts"
    md: list[str] = [f"# {title}", "", intro, ""]
    records: list[dict[str, Any]] = []
    pending: list[str] = []
    for name, heading in SECTIONS:
        text, data = parts / f"{name}.md", parts / f"{name}.json"
        if not text.exists():
            pending.append(name)
            md += [
                f"## {heading}",
                "",
                f"**PENDING** — this section is not written yet (it waits for `parts/{name}.md`).",
                "",
            ]
            continue
        body = text.read_text().rstrip()
        # a part that opens with its own section heading keeps it
        md += ([] if body.startswith("## ") else [f"## {heading}", ""]) + [body, ""]
        if data.exists():
            records += json.loads(data.read_text())
    ids = [r["id"] for r in records]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"record ids repeated across parts: {dup[:8]}")
    stamp = time.strftime("%Y-%m-%d %H:%M")
    md += ["---", "", f"Assembled {stamp} (New York) by `scripts/pm_assemble.py` at commit {head_commit()}; {len(records)} records in `numbers.json`."
           + (f" Pending sections: {', '.join(pending)}." if pending else ""), ""]  # fmt: skip
    doc = {"title": title, "assembled": stamp, "assembler_commit": head_commit(), "pending": pending,
           "budgets": pc.BUDGETS, "n_records": len(records), "records": records}  # fmt: skip
    for path, content in (
        (base / "NUMBERS.md", "\n".join(md)),
        (base / "numbers.json", json.dumps(doc, indent=1)),
    ):
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(content)
        tmp.replace(path)
    return len(records), pending


INTRO = (
    "Frozen results package for the PM report of 2026-10-09 (local correlation model, milestone M12, branch "
    "`local-correlation`). **How to read it.** `a ± b`: value and its Monte Carlo standard error on antithetic pair "
    "means (ratios: delta method; LC/CC is paired on common paths). Numbers from the dispersion study's own tables "
    "(the copula's `P_D`, model S's `P_D_S`, their calls, `EV`, `EQV`) are quoted as they are and carry no error of "
    "ours. Budgets: production = 8·10⁵ particles and 8·10⁵ pricing paths (the constant-correlation companion fitted on "
    "4·10⁵ paths); development = 2·10⁵ and 2·10⁵ (companion on 10⁵). M12 defaults since commit d4faa74 (owner's "
    "decisions of 2026-10-09): the calendar repair of the names' and of the index's crossing slices on, a name with no "
    "expiry passing the quote screen kept unscreened and flagged. LC = the calibrated local correlation model; CC = "
    "its constant-correlation companion; copula = the study's model (`P_D`); model S = the study's skewed model "
    "(`P_D_S`). Every number is also in `numbers.json` with its definition, budget, commit and source file; the tables "
    "are in `tables/` and each figure in `figures/` has its CSV beside it. `STATUS.md` has the timeline; anything "
    "produced after the freeze is in `later/`; a correction to a frozen number would be in `ERRATA.md`."
)


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--base", default=str(pc.PM))
    ap.add_argument("--title", default="Local correlation results for the PM report — NUMBERS")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    n, pending = assemble(Path(args.base), args.title, INTRO)
    log.info(
        "NUMBERS.md and numbers.json written in %s: %d records; pending: %s",
        args.base,
        n,
        pending or "none",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
