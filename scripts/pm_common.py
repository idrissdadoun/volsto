"""The PM results package of 2026-10-09 (``outputs/dispersion_lc/pm_update``): what its builders
share — the record every number is written as, the parts a section is made of, and the tables
and figures with the CSV behind each.

A *record* is one number: ``id`` (unique, dotted, e.g. ``A.ED.lc``), ``section``, ``date``,
``tenor``, ``quantity`` (a label a reader understands), ``value``, ``se`` (its standard error;
``None`` when the number is not a Monte Carlo estimate here — say why in ``notes``), ``unit``,
``definition`` (the formula in words), ``budget`` (particles / paths, or "study table"),
``commit`` (of the code that produced the number), ``source`` (the file it is read from), ``n``
(dates behind a summary) and ``notes``.

A *part* is one builder's output for a section: ``parts/<name>.json`` (its records) and
``parts/<name>.md`` (its tables as Markdown); ``scripts/pm_assemble.py`` concatenates the parts
into ``NUMBERS.md`` and ``numbers.json``.  Nothing here writes outside
``outputs/dispersion_lc/pm_update``.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
#: The package's folder in the worktree that holds ``outputs/dispersion_lc``.
PM = Path("/Users/idrissdadoun/Code/volsto-lc/outputs/dispersion_lc/pm_update")
LC_OUT = PM.parent
STUDY = ROOT / "outputs" / "dispersion"
REFERENCE_DATES = ("2026-10-02", "2019-09-03", "2017-04-03", "2008-07-07")
MULT_TAGS = ("050", "075", "100", "125", "150", "200")
BUDGETS = {
    "production": "8e5 particles / 8e5 paths (constant-correlation fit on 4e5 paths)",
    "development": "2e5 particles / 2e5 paths (constant-correlation fit on 1e5 paths)",
    "study": "the dispersion study's own table (not computed here)",
}


def record(
    id: str,
    section: str,
    quantity: str,
    value: float | None,
    se: float | None = None,
    *,
    date: str = "",
    tenor: str = "3m",
    unit: str = "",
    definition: str = "",
    budget: str = "",
    commit: str = "",
    source: str = "",
    n: int | None = None,
    notes: str = "",
) -> dict[str, Any]:
    """One number of the package (module docstring).  Non-finite values become ``None``."""

    def clean(x: float | None) -> float | None:
        if x is None:
            return None
        x = float(x)
        return x if math.isfinite(x) else None

    return {
        "id": id, "section": section, "date": date, "tenor": tenor, "quantity": quantity,
        "value": clean(value), "se": clean(se), "unit": unit, "definition": definition,
        "budget": budget, "commit": commit, "source": source, "n": n, "notes": notes,
    }  # fmt: skip


def ratio_se(a: float, a_se: float, b: float, b_se: float = 0.0, corr: float = 0.0) -> float:
    """Delta-method standard error of ``a / b`` from the two standard errors and the correlation
    of the two estimates: ``|a/b|·√((s_a/a)² + (s_b/b)² − 2·corr·s_a·s_b/(a·b))``.  Use it when
    the per-path samples are not at hand (a paired ratio on common paths has its own, smaller
    error from the samples: keep that one)."""
    if not (b and a):
        return float("nan")
    var = (a_se / a) ** 2 + (b_se / b) ** 2 - 2.0 * corr * a_se * b_se / (a * b)
    return abs(a / b) * math.sqrt(max(var, 0.0))


def pm(value: float | None, se: float | None = None, digits: int = 4) -> str:
    """``value ± se`` for a table cell (``n/a`` when missing)."""
    if value is None or not math.isfinite(value):
        return "n/a"
    if se is None or not math.isfinite(se):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ± {se:.{digits}f}"


def _atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def write_part(
    name: str, records: Sequence[dict[str, Any]], markdown: str, base: Path = PM
) -> None:
    """``parts/<name>.json`` and ``parts/<name>.md`` (atomic).  Record ids must be unique."""
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"duplicate record ids in part {name}: {dup[:5]}")
    _atomic(base / "parts" / f"{name}.json", json.dumps(list(records), indent=1))
    _atomic(base / "parts" / f"{name}.md", markdown.rstrip() + "\n")


def save_table(frame: pd.DataFrame, name: str, base: Path = PM) -> Path:
    """``tables/<name>.csv`` (atomic), floats with 10 significant digits."""
    path = base / "tables" / f"{name}.csv"
    _atomic(path, frame.to_csv(index=False, float_format="%.10g"))
    return path


def save_figure(fig: Any, name: str, frame: pd.DataFrame, base: Path = PM) -> Path:
    """``figures/<name>.pdf`` (vector) with ``figures/<name>.csv``, the data behind it."""
    path = base / "figures" / f"{name}.pdf"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    _atomic(base / "figures" / f"{name}.csv", frame.to_csv(index=False, float_format="%.10g"))
    return path


def status(line: str, base: Path = PM) -> None:
    """Append one timestamped line to ``STATUS.md``."""
    with (base / "STATUS.md").open("a") as fh:
        fh.write(f"- {time.strftime('%H:%M')} {line}\n")
