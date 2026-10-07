"""Dated snapshot of what is small and irreplaceable (BARRIER_STUDY_ADDENDUM §13.3).

    python scripts/barrier_backup.py

Copies into ``~/volsto_interview_backup/<YYYYmmdd_HHMM>/`` the report PDFs and tables, PROGRESS,
the cell / position / feature / touch / bucket tables, and writes ``cells.csv.gz``,
``positions.csv.gz`` and ``tables.xlsx`` (one sheet per report table) so that the results can be
read without Python.  Skipped, with a message, when less than 50 GB of disk is free.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "interview"
RES = OUT / "barrier_results"
BACKUP = Path.home() / "volsto_interview_backup"


def main() -> None:
    free = shutil.disk_usage(Path.home()).free / 1e9
    if free < 50:
        print(f"snapshot skipped: {free:.0f} GB free")
        return
    dest = BACKUP / time.strftime("%Y%m%d_%H%M")
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("PROGRESS.md", "gate.json", "BARRIER_STUDY_SPEC.md", "BARRIER_STUDY_ADDENDUM.md"):
        if (OUT / name).exists():
            shutil.copy2(OUT / name, dest / name)
    for rep in sorted(OUT.glob("report*")):
        if not rep.is_dir():
            continue
        (dest / rep.name).mkdir(exist_ok=True)
        for f in ("report.pdf", "report.md"):
            if (rep / f).exists():
                shutil.copy2(rep / f, dest / rep.name / f)
        if (rep / "tables").is_dir():
            shutil.copytree(rep / "tables", dest / rep.name / "tables", dirs_exist_ok=True)
    for f in sorted(RES.glob("*.parquet")):
        if f.name.startswith(("cells", "positions", "features", "touches", "buckets_lv", "quotes")):
            shutil.copy2(f, dest / f.name)
    for name in ("cells", "positions"):
        p = RES / f"{name}.parquet"
        if p.exists():
            pd.read_parquet(p).to_csv(dest / f"{name}.csv.gz", index=False, compression="gzip")
    tables = OUT / "report" / "tables"
    if tables.is_dir():
        try:
            with pd.ExcelWriter(dest / "tables.xlsx", engine="openpyxl") as xl:
                for f in sorted(tables.glob("*.csv")):
                    pd.read_csv(f).to_excel(xl, sheet_name=f.stem[:31], index=False)
        except ImportError:
            print("no Excel writer installed: tables.xlsx skipped")
    size = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file()) / 1e6
    print(f"snapshot {dest}: {size:.0f} MB")


if __name__ == "__main__":
    main()
