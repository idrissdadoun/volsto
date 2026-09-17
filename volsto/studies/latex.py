"""LaTeX and markdown rendering of study results (SPEC §10.1, M10 Part 1): the number formatter,
the booktabs tables, the two documents of an output directory (``study.md``, ``study.tex``), and
the LaTeX check.

**Rounding rule** (the owner's "significant digits driven by the stderr"; :func:`round_parts`):
the stderr is rounded to **2 significant digits when its leading digit is 1 or 2, else 1**; the
value is rounded to the same decimal place; half-up on the shortest decimal representation of
the float (what a reader of ``repr`` sees).  A stderr whose rounding carries into the next power
of ten (``0.096`` → ``0.1``) now leads with a 1 and is shown with 2 digits (``0.10``).  A rounded
zero is unsigned.  When the leading decimal exponent of the larger of the two rounded numbers
exceeds :data:`SCI_EXPONENT_LIMIT` (6) in magnitude, both are printed with that shared exponent,
``$(1.235 \\pm 0.025)\\times 10^{-9}$``.  The decimal precision is sized from the exponents, so
no finite double raises.  A zero stderr (a Monte Carlo number with no sampling noise) shows the
value to :data:`ZERO_STDERR_SIG_DIGITS` significant digits and ``\\pm 0``.  NaN renders ``--``.
:func:`format_value` renders ``$v \\pm s$`` (plain math; siunitx is not required);
:func:`format_exact` renders an exact value to the column's significant ``digits`` (integers
below 10^15 in full).  Units go in the column header (``fwd ATM vol [vol pts]``;
:data:`~volsto.studies.results.DIMENSIONLESS` is not printed).

**Tables.** :func:`latex_table` returns a ``table`` float — caption, ``\\label{tab:<name>}``, a
booktabs ``tabular`` inside an ``adjustbox`` (``max width=\\linewidth``) — or, for a direct call
on a spec above :data:`LONGTABLE_MIN_ROWS` rows, a ``longtable`` (caption and label inside, the
header repeated on every page, "continued" footers; not scalable).  :func:`split_table` is the
one splitter: by a row-group key (an ``axes`` coordinate or a function of the row label), then
into numbered parts of at most ``max_rows`` rows (``<name>-<group>-partK``, captions ``(key =
group)`` and ``(part K of N)``).  The runner renders every table through :func:`expand_tables`
with :data:`LONGTABLE_MIN_ROWS`, so each file it writes is a float scaled to the text width — a
long *and* wide table neither overflows nor is truncated — and ``study.md`` / ``study.tex``
place the parts where the narrative names the table.  Every text is escaped
(:func:`latex_escape`, which also maps the Greek letters and math symbols of
:data:`UNICODE_MATH` to ``\\ensuremath``; a character the font lacks fails the LaTeX check
instead of vanishing).  A cell the results do not carry renders ``--``; a row or column the spec
names and the results table lacks raises ``KeyError``.  :func:`markdown_table` is the same table
for ``study.md`` (``value ± stderr``).

**Documents.** The narrative is markdown.  Explicit math is written ``\\(...\\)`` and passes
through; a ``$`` is a literal dollar sign (escaped).  A line holding only ``{{table:<name>}}``
or ``{{figure:<name>}}`` places that table / figure there; the tables and figures the narrative
does not place follow it, in declaration order.  :func:`study_markdown` writes the title, the
question (first, one sentence; a fast-mode study gets :data:`FAST_MODE_BANNER` right under it),
the narrative with the tables inlined and the figures linked
(``figures/<name>.png``), and the provenance block; :func:`study_tex` writes the master document
(``\\input{tables/<name>.tex}``, ``\\includegraphics{figures/<name>.pdf}``, a
``\\FloatBarrier`` after every float so no study exceeds LaTeX's float queue), converting the
narrative with :func:`markdown_to_latex` (headings, paragraphs, lists, block quotes, fenced code
— a block containing ``\\end{verbatim}`` is refused —, pipe tables, ``**bold**``,
``*emphasis*``, ```code```, links).  The preamble compiles under XeTeX (Tectonic, the project's
engine) and pdfTeX (``iftex``).

**The LaTeX check** (:func:`compile_latex`): Tectonic ``-X compile --keep-logs``; the check fails
when the engine fails, the PDF is missing, or the log (:func:`parse_latex_log`) shows a TeX
error, a missing character, a float too large for the page, an undefined reference, or an
overfull box beyond :data:`OVERFULL_TOLERANCE_PT`.

Checked by ``tests/test_study_runner.py`` (``test_format_value_rounding``,
``test_format_value_never_raises``, ``test_latex_table_and_escape``, ``test_split_table``,
``test_latex_compiles``, ``test_latex_check_catches``, the render round trip).
"""

from __future__ import annotations

import dataclasses
import math
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping, Sequence
from decimal import ROUND_HALF_UP, Context, Decimal, localcontext
from pathlib import Path
from typing import Any

from volsto.studies.results import (
    DIMENSIONLESS,
    Column,
    FigureSpec,
    Results,
    TableSpec,
)

__all__ = [
    "Column",
    "FigureSpec",
    "TableSpec",
    "compile_latex",
    "expand_tables",
    "find_tectonic",
    "format_exact",
    "format_exact_text",
    "format_value",
    "format_value_text",
    "latex_escape",
    "latex_table",
    "markdown_table",
    "markdown_to_latex",
    "parse_latex_log",
    "round_pair",
    "round_parts",
    "split_table",
    "study_markdown",
    "study_tex",
]

#: Significant digits of a value whose stderr is exactly zero.
ZERO_STDERR_SIG_DIGITS = 6
#: What a NaN (or absent) number renders as, in both formats.
MISSING = "--"
#: Longest provenance value printed in ``study.tex`` (longer ones are shortened in the middle;
#: ``study.md`` and ``manifest.json`` keep them whole).
PROVENANCE_MAX_CHARS = 64
#: The banner a fast-mode study carries right under its question (``study.md`` and
#: ``study.tex``), placed by the runner from the manifest's ``mode``.
FAST_MODE_BANNER = "reduced paths and toy inputs: plumbing check, not results."
FAST_MODE_TITLE = "FAST MODE"
#: Width of a figure in ``study.tex``.
FIGURE_WIDTH = r"0.85\linewidth"
#: Decimal precision floor and the guard digits added to the size a quantisation needs.
DECIMAL_MIN_PREC = 28
DECIMAL_GUARD_DIGITS = 6
#: A Monte Carlo number whose leading decimal exponent exceeds this in magnitude is printed as
#: ``(v ± s) × 10^n`` with the shared exponent ``n``.
SCI_EXPONENT_LIMIT = 6
#: Tables with more rows than this are ``longtable`` environments (they break across pages); the
#: shorter ones are ``table`` floats scaled to the text width.
LONGTABLE_MIN_ROWS = 25
#: Overfull boxes up to this size are tolerated by the LaTeX check (a few points is invisible).
OVERFULL_TOLERANCE_PT = 3.0

#: Unicode the project writes in labels and captions → LaTeX (inside ``\ensuremath``).
UNICODE_MATH: dict[str, str] = {
    "α": r"\alpha",  # noqa: RUF001
    "β": r"\beta",
    "γ": r"\gamma",  # noqa: RUF001
    "δ": r"\delta",
    "ε": r"\varepsilon",
    "η": r"\eta",
    "θ": r"\theta",
    "κ": r"\kappa",
    "λ": r"\lambda",
    "μ": r"\mu",
    "ν": r"\nu",  # noqa: RUF001
    "ξ": r"\xi",
    "π": r"\pi",
    "ρ": r"\rho",  # noqa: RUF001
    "σ": r"\sigma",  # noqa: RUF001
    "τ": r"\tau",
    "φ": r"\phi",
    "χ": r"\chi",
    "ψ": r"\psi",
    "ω": r"\omega",
    "Γ": r"\Gamma",
    "Δ": r"\Delta",
    "Θ": r"\Theta",
    "Λ": r"\Lambda",
    "Ξ": r"\Xi",
    "Π": r"\Pi",
    "Σ": r"\Sigma",
    "Φ": r"\Phi",
    "Ψ": r"\Psi",
    "Ω": r"\Omega",
    "−": "-",  # noqa: RUF001
    "±": r"\pm",
    "×": r"\times",  # noqa: RUF001
    "·": r"\cdot",
    "≤": r"\leq",
    "≥": r"\geq",
    "≈": r"\approx",
    "≠": r"\neq",
    "→": r"\to",
    "←": r"\leftarrow",
    "∞": r"\infty",
    "√": r"\surd",
    "∫": r"\int",
    "∂": r"\partial",
    "∑": r"\sum",
    "⁰": "^{0}",
    "¹": "^{1}",
    "²": "^{2}",
    "³": "^{3}",
    "⁴": "^{4}",
    "⁵": "^{5}",
    "⁶": "^{6}",
    "⁷": "^{7}",
    "⁸": "^{8}",
    "⁹": "^{9}",
    "⁻": "^{-}",
    "₀": "_{0}",
    "₁": "_{1}",
    "₂": "_{2}",
    "₃": "_{3}",
    "₄": "_{4}",
    "°": r"^{\circ}",
    "µ": r"\mu",
    "∆": r"\Delta",
    "∓": r"\mp",
    "∇": r"\nabla",
    "∈": r"\in",
    "∉": r"\notin",
    "≡": r"\equiv",
    "∝": r"\propto",
    "∼": r"\sim",  # noqa: RUF001
    "≲": r"\lesssim",
    "≳": r"\gtrsim",
    "≪": r"\ll",
    "≫": r"\gg",
    "⇒": r"\Rightarrow",
    "⇐": r"\Leftarrow",
    "⇔": r"\Leftrightarrow",
    "↑": r"\uparrow",
    "↓": r"\downarrow",
    "⋅": r"\cdot",
    "′": r"\prime",  # noqa: RUF001
    "∧": r"\wedge",
    "∨": r"\vee",  # noqa: RUF001
    "✓": r"\checkmark",
    "✔": r"\checkmark",
}
_LATEX_SPECIAL: dict[str, str] = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "<": r"\textless{}",
    ">": r"\textgreater{}",
    "|": r"\textbar{}",
    "—": "---",
    "–": "--",  # noqa: RUF001
    "…": r"\ldots{}",
    " ": "~",  # noqa: RUF001
}


# --------------------------------------------------------------------------------------------
# numbers
# --------------------------------------------------------------------------------------------


def _dec(x: float) -> Decimal:
    return Decimal(repr(float(x)))


def _quantize(d: Decimal, places: int) -> Decimal:
    """``d`` rounded half-up to ``places`` decimals, with a context wide enough for any double
    at any place (the precision is sized from the exponents, so no finite input raises)."""
    need = (d.adjusted() + places if not d.is_zero() else 0) + DECIMAL_GUARD_DIGITS
    with localcontext(Context(prec=max(need, DECIMAL_MIN_PREC), Emin=-999999, Emax=999999)):
        q = d.quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)
    return q.copy_abs() if q.is_zero() else q


def _fixed(d: Decimal, places: int) -> str:
    with localcontext(Context(prec=max(len(d.as_tuple().digits) + 5, DECIMAL_MIN_PREC))):
        return f"{d:.{max(places, 0)}f}"


def stderr_places(stderr: float) -> int:
    """The decimal place (digits after the point; negative = tens, hundreds…) the rounding rule
    gives a positive finite ``stderr``."""
    if not (math.isfinite(stderr) and stderr > 0):
        raise ValueError(f"stderr must be positive and finite, got {stderr}")
    d = _dec(stderr)
    e = d.adjusted()
    lead = int(d.scaleb(-e, Context(Emin=-999999, Emax=999999)))
    places = (2 if lead in (1, 2) else 1) - 1 - e
    rounded = _quantize(d, places)
    if rounded.adjusted() > e:  # carried into the next power of ten: now leads with a 1
        places = 1 - rounded.adjusted()
    return places


def _sig(value: float, digits: int) -> str:
    """``value`` to ``digits`` significant digits in plain notation when short, else ``e``."""
    return f"{value:.{digits}g}"


def _check_mc(value: float, stderr: float) -> None:
    if math.isinf(value):
        raise ValueError(f"infinite value {value}")
    if not math.isfinite(stderr) or stderr < 0:
        raise ValueError(
            f"a Monte Carlo value needs a finite non-negative stderr, got {stderr} "
            "(format an exact value with format_exact)"
        )


def round_parts(value: float, stderr: float) -> tuple[str, str, int | None]:
    """``(value, stderr, exponent)`` under the rounding rule: when the leading decimal exponent
    of the larger of the two rounded numbers exceeds :data:`SCI_EXPONENT_LIMIT` in magnitude the
    two strings are scaled by ``10^-exponent`` (the shared exponent), else ``exponent`` is
    ``None``.  NaN value → ``("--", "", None)``; a zero stderr → the value to
    :data:`ZERO_STDERR_SIG_DIGITS` significant digits and ``"0"``."""
    if math.isnan(value):
        return MISSING, "", None
    _check_mc(value, stderr)
    if stderr == 0.0:
        return _sig(value, ZERO_STDERR_SIG_DIGITS), "0", None
    places = stderr_places(stderr)
    s = _quantize(_dec(stderr), places)
    v = _quantize(_dec(value), places)
    ref = v if (not v.is_zero() and abs(v) >= s) else s
    n = ref.adjusted()
    if abs(n) > SCI_EXPONENT_LIMIT:
        p = places + n
        big = Context(Emin=-999999, Emax=999999)
        return _fixed(v.scaleb(-n, big), p), _fixed(s.scaleb(-n, big), p), n
    return _fixed(v, places), _fixed(s, places), None


def round_pair(value: float, stderr: float) -> tuple[str, str]:
    """``(value, stderr)`` as unscaled strings under the rounding rule (module docstring); NaN
    value → ``("--", "")``.  A non-finite stderr with a finite value raises ``ValueError`` (an
    exact value goes through :func:`format_exact`)."""
    if math.isnan(value):
        return MISSING, ""
    _check_mc(value, stderr)
    if stderr == 0.0:
        return _sig(value, ZERO_STDERR_SIG_DIGITS), "0"
    places = stderr_places(stderr)
    return _fixed(_quantize(_dec(value), places), places), _fixed(
        _quantize(_dec(stderr), places), places
    )


def _math_number(text: str) -> str:
    """A number string in math mode (``1.2e-05`` → ``1.2\\times 10^{-5}``)."""
    m = re.fullmatch(r"(-?[0-9.]+)e([+-]?\d+)", text)
    if m:
        return rf"{m.group(1)}\times 10^{{{int(m.group(2))}}}"
    return text


def format_value(
    value: float, stderr: float, *, unit_in_header: bool = True, unit: str = ""
) -> str:
    """``$v \\pm s$`` under the rounding rule (``$(v \\pm s)\\times 10^{n}$`` beyond
    :data:`SCI_EXPONENT_LIMIT`); ``--`` for NaN.  With ``unit_in_header=False`` the (escaped)
    ``unit`` follows the number in the cell.  Never raises for a finite value and a finite
    non-negative stderr."""
    v, s, n = round_parts(value, stderr)
    if v == MISSING:
        return MISSING
    out = rf"${_math_number(v)} \pm {s}$" if n is None else rf"$({v} \pm {s})\times 10^{{{n}}}$"
    if not unit_in_header and unit and unit != DIMENSIONLESS:
        out += "~" + latex_escape(unit)
    return out


def format_value_text(value: float, stderr: float) -> str:
    """``v ± s`` (markdown; ``(v ± s)×10^n`` beyond :data:`SCI_EXPONENT_LIMIT`) under the
    rounding rule; ``--`` for NaN."""
    v, s, n = round_parts(value, stderr)
    if v == MISSING:
        return MISSING
    return f"{v} ± {s}" if n is None else f"({v} ± {s})\N{MULTIPLICATION SIGN}10^{n}"


def _exact_str(value: float, digits: int) -> str:
    if math.isnan(value):
        return MISSING
    if math.isinf(value):
        raise ValueError(f"infinite value {value}")
    if float(value).is_integer() and abs(value) < 1e15:
        return str(int(value))
    return _sig(value, digits)


def format_exact(value: float, digits: int) -> str:
    """An exact value to ``digits`` significant digits (an integer value in full), in math
    mode; ``--`` for NaN."""
    text = _exact_str(value, digits)
    return MISSING if text == MISSING else f"${_math_number(text)}$"


def format_exact_text(value: float, digits: int) -> str:
    """:func:`format_exact` for markdown."""
    return _exact_str(value, digits)


# --------------------------------------------------------------------------------------------
# escaping
# --------------------------------------------------------------------------------------------


def latex_escape(text: str) -> str:
    """Text-mode LaTeX for ``text``: the special characters escaped, the Greek letters and math
    symbols of :data:`UNICODE_MATH` wrapped in ``\\ensuremath``, newlines as spaces."""
    out: list[str] = []
    for ch in str(text).replace("\r", "").replace("\n", " "):
        if ch in _LATEX_SPECIAL:
            out.append(_LATEX_SPECIAL[ch])
        elif ch in UNICODE_MATH:
            out.append(rf"\ensuremath{{{UNICODE_MATH[ch]}}}")
        else:
            out.append(ch)
    return "".join(out)


def markdown_escape(text: str) -> str:
    """A markdown table cell: pipes escaped, newlines as spaces."""
    return str(text).replace("\n", " ").replace("|", r"\|")


# --------------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------------


def column_unit(spec: TableSpec, col: Column, results: Results) -> str:
    """The header unit of ``col``: its own ``unit``, else the results' unit of the column."""
    return col.unit or results.unit(spec.table, col.key)


def header_text(col: Column, unit: str) -> str:
    """``header [unit]`` (no bracket for an empty or dimensionless unit)."""
    return f"{col.header} [{unit}]" if unit and unit != DIMENSIONLESS else col.header


def _cells(spec: TableSpec, results: Results) -> tuple[list[str], dict[tuple[str, str], Any]]:
    """Row order and the ``(row, column) → record`` map of a spec, validated."""
    long = results.long(spec.table)
    have_rows = list(dict.fromkeys(long["row"]))
    have_cols = set(long["column"])
    rows = list(spec.rows) if spec.rows is not None else have_rows
    unknown_rows = [r for r in rows if r not in set(have_rows)]
    if unknown_rows:
        raise KeyError(f"table {spec.name!r}: rows {unknown_rows} not in results {spec.table!r}")
    unknown_cols = [c.key for c in spec.columns if c.key not in have_cols]
    if unknown_cols:
        raise KeyError(
            f"table {spec.name!r}: columns {unknown_cols} not in results {spec.table!r} "
            f"(have {sorted(have_cols)})"
        )
    records = {(str(r["row"]), str(r["column"])): r for r in long.to_dict("records")}
    return rows, records


def _cell(rec: Mapping[str, Any] | None, col: Column, *, tex: bool) -> str:
    if rec is None:
        return MISSING
    value, stderr = float(rec["value"]), float(rec["stderr"])
    if bool(rec["exact"]):
        return format_exact(value, col.digits) if tex else format_exact_text(value, col.digits)
    return format_value(value, stderr) if tex else format_value_text(value, stderr)


def latex_table(spec: TableSpec, results: Results) -> str:
    """The booktabs table of ``spec`` (module docstring): a ``table`` float scaled to the text
    width, or a ``longtable`` (header repeated on every page, not scaled) when it has more than
    :data:`LONGTABLE_MIN_ROWS` rows."""
    rows, records = _cells(spec, results)
    header = " & ".join(
        [latex_escape(spec.row_header)]
        + [latex_escape(header_text(c, column_unit(spec, c, results))) for c in spec.columns]
    )
    body = []
    for r in rows:
        cells = [latex_escape(r)] + [
            _cell(records.get((r, c.key)), c, tex=True) for c in spec.columns
        ]
        body.append(" & ".join(cells) + r" \\")
    colspec = f"l{'r' * len(spec.columns)}"
    caption = latex_escape(spec.caption)
    lines = [
        f"% table {spec.name}: generated by volsto.studies.latex from results.parquet "
        f"(results table {latex_escape(spec.table)}); do not edit",
    ]
    if len(rows) > LONGTABLE_MIN_ROWS:
        n = len(spec.columns) + 1
        lines += [
            r"\begingroup",
            r"\small",
            rf"\begin{{longtable}}{{{colspec}}}",
            rf"\caption{{{caption}}}\label{{tab:{spec.name}}} \\",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            r"\endfirsthead",
            rf"\caption[]{{{caption} (continued)}} \\",
            r"\toprule",
            header + r" \\",
            r"\midrule",
            r"\endhead",
            r"\midrule",
            rf"\multicolumn{{{n}}}{{r}}{{\emph{{continued on the next page}}}} \\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
            *body,
            r"\end{longtable}",
            r"\endgroup",
            "",
        ]
        return "\n".join(lines)
    lines += [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{tab:{spec.name}}}",
        r"\begin{adjustbox}{max width=\linewidth}",
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *body,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{table}",
        "",
    ]
    return "\n".join(lines)


def markdown_table(spec: TableSpec, results: Results) -> str:
    """The markdown version of :func:`latex_table`: a bold caption line then a pipe table."""
    rows, records = _cells(spec, results)
    header = [markdown_escape(spec.row_header or " ")] + [
        markdown_escape(header_text(c, column_unit(spec, c, results))) for c in spec.columns
    ]
    lines = [
        f"**Table `{spec.name}`.** {markdown_escape(spec.caption)}",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "|".join([":--"] + ["--:"] * len(spec.columns)) + "|",
    ]
    for r in rows:
        cells = [markdown_escape(r)] + [
            _cell(records.get((r, c.key)), c, tex=False) for c in spec.columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _group_label(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "none"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_") or "none"


def _unique(name: str, used: set[str]) -> str:
    out, i = name, 1
    while out in used:
        i += 1
        out = f"{name}_{i}"
    used.add(out)
    return out


def split_table(
    spec: TableSpec,
    results: Results,
    max_rows: int = LONGTABLE_MIN_ROWS,
    by: str | Callable[[str], Any] | None = None,
) -> list[TableSpec]:
    """The one table splitter: ``spec`` as a list of specs of at most ``max_rows`` rows each.

    With ``by`` the rows are first grouped — ``by`` is an ``axes`` key of the results rows (the
    group is that coordinate) or a function of the row label — in order of first appearance,
    each group a table ``<name>-<group>`` captioned ``(<by> = <group>)``; then every group (or the
    whole table) longer than ``max_rows`` is cut into ``<name>-partK`` tables captioned ``(part K
    of N)``.  Names stay unique and valid.  A table that needs no split is returned unchanged
    (``[spec]``).  The runner applies it with :data:`LONGTABLE_MIN_ROWS` to every table it
    renders, so each part is a width-scaled float (a long *and* wide table does not overflow);
    a module splits by its own key first when it wants grouped tables."""
    if max_rows < 1:
        raise ValueError(f"max_rows must be >= 1, got {max_rows}")
    rows = list(spec.rows) if spec.rows is not None else results.rows(spec.table)
    groups: list[tuple[str | None, list[str]]]
    if by is None:
        groups = [(None, rows)]
    else:
        key_name = by if isinstance(by, str) else ""
        key: Callable[[str], Any] = (
            (lambda row: results.axes(spec.table, row).get(key_name)) if isinstance(by, str) else by
        )
        grouped: dict[str, list[str]] = {}
        for r in rows:
            grouped.setdefault(_group_label(key(r)), []).append(r)
        groups = list(grouped.items())
    out: list[TableSpec] = []
    used: set[str] = set()
    for label, group_rows in groups:
        if label is None:
            base_name, base_caption = spec.name, spec.caption
        else:
            base_name = _unique(f"{spec.name}-{_slug(label)}", used)
            what = f"{key_name} = {label}" if key_name else label
            base_caption = f"{spec.caption} ({what})"
        chunks = [group_rows[i : i + max_rows] for i in range(0, len(group_rows), max_rows)]
        chunks = chunks or [[]]
        for j, chunk in enumerate(chunks, 1):
            if len(chunks) == 1:
                name, caption = base_name, base_caption
            else:
                name = _unique(f"{base_name}-part{j}", used)
                caption = f"{base_caption} (part {j} of {len(chunks)})"
            out.append(dataclasses.replace(spec, name=name, caption=caption, rows=tuple(chunk)))
    if by is None and len(out) == 1:
        return [spec]
    return out


def expand_tables(
    tables: Sequence[TableSpec], results: Results, max_rows: int = LONGTABLE_MIN_ROWS
) -> tuple[list[TableSpec], dict[str, list[str]]]:
    """:func:`split_table` over ``tables``: the flat list of the parts and, per declared table,
    the names of its parts (in order)."""
    flat: list[TableSpec] = []
    parts: dict[str, list[str]] = {}
    for spec in tables:
        pieces = split_table(spec, results, max_rows)
        parts[spec.name] = [p.name for p in pieces]
        flat.extend(pieces)
    _check_unique([p.name for p in flat], "table (after splitting)")
    return flat, parts


# --------------------------------------------------------------------------------------------
# markdown → LaTeX (the narrative)
# --------------------------------------------------------------------------------------------

PLACEHOLDER = re.compile(r"\{\{\s*(table|figure)\s*:\s*([A-Za-z0-9_-]+)\s*\}\}")
_INLINE = re.compile(
    r"(?P<math>\\\(.+?\\\))"
    r"|(?P<code>`[^`\n]+`)"
    r"|(?P<bold>\*\*(?:(?!\*\*).)+\*\*)"
    r"|(?P<emph>\*[^*\n]+\*)"
    r"|(?P<link>\[[^\]\n]+\]\([^)\s]+\))"
)
_HEADINGS = {1: "section", 2: "subsection", 3: "subsubsection", 4: "paragraph"}
VERBATIM_END = "\\end{verbatim}"


def inline_to_latex(text: str) -> str:
    """Inline markdown → LaTeX: explicit math ``\\(...\\)`` passes through verbatim; code,
    bold, emphasis and links are converted; everything else is escaped — a ``$`` is a literal
    dollar sign, never math."""
    out: list[str] = []
    pos = 0
    for m in _INLINE.finditer(text):
        out.append(latex_escape(text[pos : m.start()]))
        tok = m.group(0)
        if m.group("math"):
            out.append(tok)
        elif m.group("code"):
            out.append(rf"\texttt{{{latex_escape(tok[1:-1])}}}")
        elif m.group("bold"):
            out.append(rf"\textbf{{{inline_to_latex(tok[2:-2])}}}")
        elif m.group("emph"):
            out.append(rf"\emph{{{inline_to_latex(tok[1:-1])}}}")
        else:
            label, _, url = tok[1:-1].partition("](")
            safe_url = url.replace("\\", "/").replace("%", r"\%").replace("#", r"\#")
            out.append(rf"\href{{{safe_url}}}{{{inline_to_latex(label)}}}")
        pos = m.end()
    out.append(latex_escape(text[pos:]))
    return "".join(out)


def _pipe_cells(line: str) -> list[str]:
    body = line.strip()
    body = body[1:] if body.startswith("|") else body
    body = body[:-1] if body.endswith("|") else body
    return [c.strip() for c in re.split(r"(?<!\\)\|", body)]


def _pipe_table(lines: Sequence[str]) -> list[str]:
    rows = [_pipe_cells(line) for line in lines]
    if len(rows) >= 2 and all(re.fullmatch(r":?-+:?", c) for c in rows[1]):
        head, body = rows[0], rows[2:]
    else:
        head, body = None, rows
    ncol = max(len(r) for r in rows)
    out = [r"\begin{center}", r"\begin{adjustbox}{max width=\linewidth}"]
    out.append(rf"\begin{{tabular}}{{{'l' * ncol}}}")
    out.append(r"\toprule")
    if head is not None:
        out.append(" & ".join(inline_to_latex(c.replace(r"\|", "|")) for c in head) + r" \\")
        out.append(r"\midrule")
    for r in body:
        cells = [inline_to_latex(c.replace(r"\|", "|")) for c in r] + [""] * (ncol - len(r))
        out.append(" & ".join(cells) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{center}"]
    return out


def markdown_to_latex(text: str) -> str:
    """Block markdown → LaTeX (module docstring).  Placeholders are not handled here (see
    :func:`narrative_blocks`)."""
    lines = text.replace("\r", "").split("\n")
    out: list[str] = []
    para: list[str] = []
    i = 0

    def flush() -> None:
        if para:
            out.append(inline_to_latex(" ".join(s.strip() for s in para)))
            out.append("")
            para.clear()

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            flush()
            i += 1
            continue
        if stripped.startswith("```"):
            flush()
            j = i + 1
            code: list[str] = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                if VERBATIM_END in lines[j]:
                    raise ValueError(
                        f"a fenced code block of the narrative contains {VERBATIM_END!r}, which "
                        "would end the LaTeX verbatim environment: remove or rewrite that line"
                    )
                code.append(lines[j])
                j += 1
            out += [r"\begin{verbatim}", *code, r"\end{verbatim}", ""]
            i = j + 1
            continue
        heading = re.match(r"^(#{1,4})\s+(.*)$", stripped)
        if heading:
            flush()
            cmd = _HEADINGS[len(heading.group(1))]
            out += [rf"\{cmd}*{{{inline_to_latex(heading.group(2).strip())}}}", ""]
            i += 1
            continue
        if stripped.startswith("|"):
            flush()
            block: list[str] = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                block.append(lines[i])
                i += 1
            out += [*_pipe_table(block), ""]
            continue
        item = re.match(r"^\s*([-*+]|\d+[.)])\s+(.*)$", line)
        if item:
            flush()
            env = "enumerate" if item.group(1)[0].isdigit() else "itemize"
            items: list[str] = []
            while i < len(lines):
                m = re.match(r"^\s*([-*+]|\d+[.)])\s+(.*)$", lines[i])
                if m:
                    items.append(m.group(2))
                elif lines[i].strip() and items and lines[i].startswith((" ", "\t")):
                    items[-1] += " " + lines[i].strip()
                else:
                    break
                i += 1
            out.append(rf"\begin{{{env}}}")
            out += [rf"\item {inline_to_latex(it)}" for it in items]
            out += [rf"\end{{{env}}}", ""]
            continue
        if stripped.startswith(">"):
            flush()
            quote: list[str] = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip()[1:].strip())
                i += 1
            out += [r"\begin{quote}", inline_to_latex(" ".join(quote)), r"\end{quote}", ""]
            continue
        para.append(line)
        i += 1
    flush()
    return "\n".join(out).rstrip() + "\n"


def narrative_blocks(
    narrative: str, tables: Sequence[str], figures: Sequence[str]
) -> list[tuple[str, str]]:
    """The narrative split into ``("markdown", text)``, ``("table", name)`` and
    ``("figure", name)`` blocks; the tables and figures it does not place are appended in
    declaration order.  A placeholder must stand on its own line and name a declared table /
    figure, at most once."""
    known = {"table": list(tables), "figure": list(figures)}
    blocks: list[tuple[str, str]] = []
    placed: set[tuple[str, str]] = set()
    buf: list[str] = []
    for line in narrative.replace("\r", "").split("\n"):
        m = PLACEHOLDER.fullmatch(line.strip())
        if m is None:
            if PLACEHOLDER.search(line):
                raise ValueError(f"a placeholder must stand on its own line: {line!r}")
            buf.append(line)
            continue
        kind, name = m.group(1), m.group(2)
        if name not in known[kind]:
            raise ValueError(f"narrative places unknown {kind} {name!r}; declared {known[kind]}")
        if (kind, name) in placed:
            raise ValueError(f"narrative places {kind} {name!r} twice")
        placed.add((kind, name))
        if "\n".join(buf).strip():
            blocks.append(("markdown", "\n".join(buf).strip("\n")))
        buf = []
        blocks.append((kind, name))
    if "\n".join(buf).strip():
        blocks.append(("markdown", "\n".join(buf).strip("\n")))
    for kind in ("table", "figure"):
        blocks += [(kind, n) for n in known[kind] if (kind, n) not in placed]
    return blocks


# --------------------------------------------------------------------------------------------
# documents
# --------------------------------------------------------------------------------------------

TEX_PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage{iftex}
\ifPDFTeX
  \usepackage[T1]{fontenc}
  \usepackage[utf8]{inputenc}
  \usepackage{lmodern}
\else
  \usepackage{fontspec}
\fi
\usepackage[margin=2.5cm]{geometry}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{graphicx}
\usepackage{adjustbox}
\usepackage{placeins}
\usepackage[hidelinks]{hyperref}
\emergencystretch=2em
"""


def _check_unique(names: Sequence[str], kind: str) -> None:
    dup = sorted({n for n in names if list(names).count(n) > 1})
    if dup:
        raise ValueError(f"duplicate {kind} names {dup}")


def study_markdown(
    *,
    title: str,
    question: str,
    narrative: str,
    tables: Sequence[TableSpec],
    figures: Sequence[FigureSpec],
    results: Results,
    provenance: Sequence[tuple[str, str]],
    max_rows: int = LONGTABLE_MIN_ROWS,
    mode: str = "full",
) -> str:
    """``study.md``: title, the question first (with the :data:`FAST_MODE_BANNER` right under it
    when ``mode`` is ``"fast"``), the narrative with the tables inlined (split by
    :func:`split_table` exactly as the LaTeX tables are) and the figures linked, then the
    provenance block."""
    _check_unique([t.name for t in tables], "table")
    _check_unique([f.name for f in figures], "figure")
    flat, table_parts = expand_tables(tables, results, max_rows)
    by_part = {t.name: t for t in flat}
    by_figure = {f.name: f for f in figures}
    parts = [f"# {title}", "", f"**Question.** {question.strip()}", ""]
    if mode == "fast":
        parts += [f"> **{FAST_MODE_TITLE}** — {FAST_MODE_BANNER}", ""]
    for kind, payload in narrative_blocks(
        narrative, [t.name for t in tables], [f.name for f in figures]
    ):
        if kind == "markdown":
            parts += [payload.strip("\n"), ""]
        elif kind == "table":
            parts += [markdown_table(by_part[name], results) for name in table_parts[payload]]
        else:
            fig = by_figure[payload]
            parts += [
                f"![{markdown_escape(fig.caption)}](figures/{fig.name}.png)",
                "",
                f"**Figure `{fig.name}`.** {fig.caption} " f"([PDF](figures/{fig.name}.pdf))",
                "",
            ]
    parts += ["## Provenance", "", "| | |", "|:--|:--|"]
    parts += [f"| {markdown_escape(k)} | {markdown_escape(v)} |" for k, v in provenance]
    return "\n".join(parts).rstrip() + "\n"


def _middle_truncate(text: str, limit: int | None = None) -> str:
    """``text`` shortened in the middle to ``limit`` (default :data:`PROVENANCE_MAX_CHARS`)
    characters when it is a single token (a long path or hash in a PDF table)."""
    n = PROVENANCE_MAX_CHARS if limit is None else limit
    if len(text) <= n or any(ch.isspace() for ch in text):
        return text  # prose wraps inside the scaled table; only paths and hashes are cut
    keep = n - 3
    return text[: keep // 3] + "..." + text[-(keep - keep // 3) :]


def study_tex(
    *,
    title: str,
    question: str,
    narrative: str,
    tables: Sequence[TableSpec],
    figures: Sequence[FigureSpec],
    provenance: Sequence[tuple[str, str]],
    date: str = "",
    results: Results | None = None,
    max_rows: int = LONGTABLE_MIN_ROWS,
    mode: str = "full",
) -> str:
    """``study.tex``: the master document (module docstring), with the boxed
    :data:`FAST_MODE_BANNER` right under the question when ``mode`` is ``"fast"``.  The tables are
    ``\\input``-ed from ``tables/<part>.tex`` — with ``results``, every table is split by
    :func:`split_table` (the parts the runner writes); without, one file per table — so this
    text depends on the specs and the row counts, not on the numbers."""
    _check_unique([t.name for t in tables], "table")
    if results is not None:
        table_parts = expand_tables(tables, results, max_rows)[1]
    else:
        table_parts = {t.name: [t.name] for t in tables}
    _check_unique([f.name for f in figures], "figure")
    by_figure = {f.name: f for f in figures}
    body: list[str] = [
        TEX_PREAMBLE.rstrip("\n"),
        rf"\title{{{latex_escape(title)}}}",
        rf"\date{{{latex_escape(date)}}}",
        r"\begin{document}",
        r"\maketitle",
        "",
        rf"\noindent\textbf{{Question.}} {inline_to_latex(question.strip())}",
        "",
    ]
    if mode == "fast":
        body += [
            r"\medskip",
            r"\noindent\fbox{\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}{"
            rf"\textbf{{{FAST_MODE_TITLE}}} --- {latex_escape(FAST_MODE_BANNER)}}}}}",
            "",
        ]
    for kind, payload in narrative_blocks(
        narrative, [t.name for t in tables], [f.name for f in figures]
    ):
        if kind == "markdown":
            body.append(markdown_to_latex(payload))
        elif kind == "table":
            for name in table_parts[payload]:
                body += [rf"\input{{tables/{name}.tex}}", r"\FloatBarrier", ""]
        else:
            fig = by_figure[payload]
            body += [
                r"\begin{figure}[htbp]",
                r"\centering",
                rf"\includegraphics[width={FIGURE_WIDTH}]{{figures/{fig.name}.pdf}}",
                rf"\caption{{{latex_escape(fig.caption)}}}",
                rf"\label{{fig:{fig.name}}}",
                r"\end{figure}",
                r"\FloatBarrier",
                "",
            ]
    body += [
        r"\section*{Provenance}",
        r"\begin{small}",
        r"\begin{adjustbox}{max width=\linewidth}",
        r"\begin{tabular}{@{}ll@{}}",
        r"\toprule",
    ]
    body += [
        rf"{latex_escape(k)} & \texttt{{{latex_escape(_middle_truncate(v))}}} \\"
        for k, v in provenance
    ]
    body += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{adjustbox}",
        r"\end{small}",
        "",
        r"\end{document}",
        "",
    ]
    return "\n".join(body)


# --------------------------------------------------------------------------------------------
# the LaTeX check (Tectonic)
# --------------------------------------------------------------------------------------------

#: The LaTeX engine: ``$VOLSTO_TECTONIC``, else ``tectonic`` on PATH, else the Homebrew path.
TECTONIC_ENV = "VOLSTO_TECTONIC"
TECTONIC_FALLBACK = Path("/opt/homebrew/bin/tectonic")
#: Seconds before a LaTeX compile is abandoned (the first compile on a machine downloads the
#: engine's package bundle).
LATEX_TIMEOUT_S = 600.0
TECTONIC_MISSING = (
    "tectonic not found (install with `brew install tectonic`, or set VOLSTO_TECTONIC to the "
    "executable; its first compile on a new machine downloads the TeX package bundle and needs "
    "network access)"
)
_OVERFULL = re.compile(r"^Overfull \\[hv]box \(([0-9.]+)pt too (?:wide|high)\)")
#: Log lines that fail the check whatever their context (TeX errors, dropped glyphs, a float
#: taller than a page — which LaTeX truncates silently — and unresolved references).
LOG_FAILURES: tuple[re.Pattern[str], ...] = (
    re.compile(r"^! "),
    re.compile(r"^Missing character: There is no"),
    re.compile(r"Float too large for page"),
    re.compile(r"There were undefined references"),
    re.compile(r"Reference `[^']*' on page .* undefined"),
    re.compile(r"Citation `[^']*' on page .* undefined"),
)


def find_tectonic() -> str | None:
    """The Tectonic executable (:data:`TECTONIC_ENV`, PATH, the Homebrew path) or ``None``."""
    env = os.environ.get(TECTONIC_ENV)
    if env:
        return env if Path(env).exists() else None
    found = shutil.which("tectonic")
    if found:
        return found
    return str(TECTONIC_FALLBACK) if TECTONIC_FALLBACK.exists() else None


def parse_latex_log(text: str, overfull_tolerance_pt: float = OVERFULL_TOLERANCE_PT) -> list[str]:
    """The problems of a TeX log that fail the LaTeX check (module docstring of the runner):
    every :data:`LOG_FAILURES` line and every overfull box larger than
    ``overfull_tolerance_pt``; in log order, duplicates removed."""
    problems: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        m = _OVERFULL.match(line)
        if m is not None:
            if float(m.group(1)) > overfull_tolerance_pt:
                problems.append(line)
            continue
        if any(p.search(line) for p in LOG_FAILURES):
            problems.append(line)
    return list(dict.fromkeys(problems))


def compile_latex(directory: Path, tex_name: str = "study.tex") -> dict[str, Any]:
    """Compile ``directory/tex_name`` with Tectonic (``-X compile --keep-logs``) and check its
    log (:func:`parse_latex_log`).  Returns ``{engine, status: ok | failed | skipped, seconds,
    pdf, log, problems, reason}``: ``failed`` when the engine fails, the PDF is missing, or the
    log shows a problem (the PDF is kept for inspection)."""
    exe = find_tectonic()
    if exe is None:
        return {
            "engine": "tectonic",
            "status": "skipped",
            "seconds": 0.0,
            "pdf": None,
            "log": None,
            "problems": [],
            "reason": TECTONIC_MISSING,
        }
    t0 = time.perf_counter()
    stem = Path(tex_name).stem
    try:
        proc = subprocess.run(
            [exe, "-X", "compile", "--keep-logs", tex_name],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=LATEX_TIMEOUT_S,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "engine": exe,
            "status": "failed",
            "seconds": time.perf_counter() - t0,
            "pdf": None,
            "log": None,
            "problems": [],
            "reason": f"{type(exc).__name__}: {exc}",
        }
    pdf = directory / f"{stem}.pdf"
    log_path = directory / f"{stem}.log"
    log_text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    problems = parse_latex_log(log_text)
    tail = "\n".join((proc.stdout + proc.stderr).strip().splitlines()[-25:])
    reason = ""
    if proc.returncode != 0 or not pdf.is_file():
        reason = f"tectonic exit {proc.returncode}:\n{tail}"
    elif problems:
        shown = problems[:20]
        more = f"\n  ... and {len(problems) - 20} more" if len(problems) > 20 else ""
        reason = f"{len(problems)} problem(s) in {log_path.name}:\n  " + "\n  ".join(shown) + more
    return {
        "engine": exe,
        "status": "failed" if reason else "ok",
        "seconds": time.perf_counter() - t0,
        "pdf": pdf.name if pdf.is_file() else None,
        "log": log_path.name if log_path.is_file() else None,
        "problems": problems,
        "reason": reason,
    }
