"""PM results package of 2026-10-09, sections A and B: today (2026-10-02) and the three other
reference dates (2019-09-03, 2017-04-03, 2008-07-07) at 3m, production budget, the owner's
decisions on.

    python scripts/pm_ab.py [--rows-dir <dir>] [--risk <json>] [--cdv <json>] [--out <dir>]

Reads (never writes there): the production rows ``outputs/dispersion_lc/rows/3m_production/
<date>.json`` (one per date), today's risk ``pm_update/parts/A_risk_today_raw.json``, the
cross-dependent scan ``outputs/dispersion_lc/cdv/pm/cdv_scan_2026-10-02_3m.json``, the study's
``entries_3m.parquet`` (basket B1) and ``model_s_3m.parquet``, the parametric reference's
fixtures ``tests/golden/lcm_reference/<tag>.json`` and, for check (a), the old-default rows
``rows/3m_production_norepair`` and the log of the independent rebuild of the reference.

Writes one part per section — ``parts/A_today.{json,md}``, ``parts/B_reference.{json,md}`` —
and the CSV behind every table (``tables/A_*.csv``, ``tables/B_*.csv``).  Pure reading and
arithmetic: no number is estimated.  An input that has not arrived is written as ``pending:
<the file it waits for>``; rerun the script when it arrives (idempotent).

Standard errors.  A model's own number carries its Monte Carlo error from its file.  LC/CC, the
call ratios LC/CC and ``ED_wing``/``ED_eqv`` over CC carry the row's paired errors.  A ratio to
the copula is ``pm_common.ratio_se`` of the numerator's error and the copula's (``P_D_se``,
``C_se_<m>``), taken independent.  The study's tables give no error for model S nor for ``EV``:
those cells have none.

``--rows-dir`` other than the default is for testing the code path on stand-in rows: it is
refused unless ``--out`` points away from the package.
"""

# ruff: noqa: RUF001, E501 — table labels and definitions: typographic signs, long lines
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_common as pc

log = logging.getLogger("pm_ab")

ROOT = pc.ROOT
TODAY = pc.REFERENCE_DATES[0]
OTHERS = pc.REFERENCE_DATES[1:]
TENOR = "3m"
ROWS_DIR = pc.LC_OUT / "rows" / "3m_production"
OLD_ROWS_DIR = pc.LC_OUT / "rows" / "3m_production_norepair"
DEV_ROWS_DIR = pc.LC_OUT / "rows" / "3m_development_repair"
RISK_FILE = pc.PM / "parts" / "A_risk_today_raw.json"
CDV_FILE = pc.LC_OUT / "cdv" / "pm" / f"cdv_scan_{TODAY}_{TENOR}.json"
ENTRIES = pc.STUDY / "entries_3m.parquet"
MODEL_S = pc.STUDY / "model_s_3m.parquet"
FIXTURES = ROOT / "tests" / "golden" / "lcm_reference"
REF_TAGS = ("today", "typical", "steep", "typical_alt")
CHECK_A_LOG = Path(
    "/private/tmp/claude-501/-Users-idrissdadoun-Code-volsto/75dd7f23-d74c-43a9-b99e-61c23722b22c/scratchpad/r4/check-a/s4_ratio_by_tag.log"
)
REF_BUDGET = "reference implementation fixture"
REF_COMMIT = "n/a (stand-alone reference implementation, run of 2026-10-07)"
STUDY_COMMIT = "n/a (study table)"
CLIP_FLAG = 0.01
CDV_BETAS = (3.0, 6.0)
NAN = float("nan")


def rel(path: Path | str) -> str:
    """The path relative to the worktree when it is inside it."""
    p = Path(os.path.abspath(path))  # not resolve(): outputs/dispersion is a link to the study
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def num(doc: Mapping[str, Any] | None, key: str) -> float:
    """A finite float of ``doc[key]`` or NaN."""
    if doc is None:
        return NAN
    try:
        x = float(doc[key])
    except (KeyError, TypeError, ValueError):
        return NAN
    return x if math.isfinite(x) else NAN


def ok(x: float | None) -> bool:
    return x is not None and math.isfinite(x)


def div(a: float, b: float) -> float:
    return a / b if ok(a) and ok(b) and b != 0.0 else NAN


@dataclass
class Item:
    """One number of a table: what the cell shows and what the record says."""

    key: str
    label: str
    value: float | None = None
    se: float | None = None
    unit: str = ""
    definition: str = ""
    budget: str = ""
    commit: str = ""
    source: str = ""
    notes: str = ""
    digits: int = 4
    pending: str = ""  # the file the number waits for
    mark: str = ""  # a suffix of the cell (e.g. model S not converged)
    na: str = ""  # why there is no value, when it is not pending

    def cell(self) -> str:
        if self.pending:
            return f"pending: {self.pending}"
        if not ok(self.value):
            return f"n/a ({self.na})" if self.na else "n/a"
        return pc.pm(self.value, self.se, self.digits) + self.mark

    def record(self, prefix: str, section: str, date: str) -> dict[str, Any]:
        notes = [self.notes] if self.notes else []
        if self.pending:
            notes.append(f"pending: waits for {self.pending}")
        elif not ok(self.value) and self.na:
            notes.append(f"n/a: {self.na}")
        return pc.record(
            prefix + self.key, section, self.label, None if self.pending else self.value,
            None if self.pending else self.se, date=date, tenor=TENOR, unit=self.unit,
            definition=self.definition, budget=self.budget, commit=self.commit, source=self.source,
            notes="; ".join(notes),
        )  # fmt: skip


@dataclass
class Part:
    """A section being built: its records, its Markdown and the CSV behind each table."""

    section: str
    records: list[dict[str, Any]] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    tables: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    pending: list[str] = field(default_factory=list)

    def add(self, table: str, prefix: str, date: str, items: Iterable[Item]) -> None:
        for it in items:
            rec = it.record(prefix, self.section, date)
            self.records.append(rec)
            self.tables.setdefault(table, []).append({**rec, "cell": it.cell()})
            if it.pending:
                self.pending.append(f"{rec['id']} <- {it.pending}")

    def text(self, *lines: str) -> None:
        self.lines.extend(lines)

    def table(self, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
        self.lines.append("| " + " | ".join(headers) + " |")
        self.lines.append("|" + "|".join(["---"] * len(headers)) + "|")
        for row in rows:
            self.lines.append("| " + " | ".join(str(c).replace("|", "/") for c in row) + " |")
        self.lines.append("")

    def save(self, name: str, base: Path) -> None:
        cols = ["id", "date", "quantity", "value", "se", "unit", "cell", "definition", "budget", "commit", "source", "notes"]  # fmt: skip
        for table, rows in self.tables.items():
            pc.save_table(pd.DataFrame(rows)[cols], table, base)
        pc.write_part(name, self.records, "\n".join(self.lines), base)


# --------------------------------------------------------------------------- inputs


def load_json(path: Path) -> tuple[dict[str, Any] | None, str]:
    """``(document, "")`` or ``(None, why)``: a file that is absent or half written is pending."""
    if not path.exists():
        return None, rel(path)
    try:
        doc = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None, f"{rel(path)} (unreadable: being written?)"
    if not isinstance(doc, dict):
        return None, f"{rel(path)} (not a JSON object)"
    return doc, ""


def short(n: float) -> str:
    """``8e5`` for 800000 (the package's way of writing a budget)."""
    if not ok(n) or n <= 0:
        return "?"
    exp = math.floor(math.log10(n))
    mant = n / 10**exp
    return f"{mant:g}e{exp}" if exp >= 3 else f"{n:g}"


def budget_of(n_particles: float, n_paths: float, companion: float | None = NAN) -> str:
    """The package's budget label when the counts are a known budget's (the companion's paths
    too, when the file states them), the counts otherwise.  ``companion=None``: a run without a
    constant-correlation fit, always written as its counts."""
    if not (ok(n_particles) and ok(n_paths)):
        return "budget not stated in the file"
    if companion is None:
        return f"{short(n_particles)} particles / {short(n_paths)} paths"
    known = {
        (800_000.0, 800_000.0, 400_000.0): pc.BUDGETS["production"],
        (200_000.0, 200_000.0, 100_000.0): pc.BUDGETS["development"],
    }
    for (n, m, c), label in known.items():
        if (n_particles, n_paths) == (n, m) and (not ok(companion) or companion == c):
            return label
    tail = f" (constant-correlation fit on {short(companion)} paths)" if ok(companion) else ""
    return f"{short(n_particles)} particles / {short(n_paths)} paths{tail}"


@dataclass
class Inputs:
    date: str
    row: dict[str, Any] | None
    row_path: Path
    row_pending: str
    entry: dict[str, Any] | None
    entry_na: str
    model_s: dict[str, Any] | None
    model_s_na: str
    ref: dict[str, Any] | None
    ref_tag: str

    @property
    def priced(self) -> bool:
        return self.row is not None and ok(num(self.row, "ED_lc"))

    @property
    def row_na(self) -> str:
        if self.row is None or self.priced:
            return ""
        return f"row {self.row.get('status', '?')}: {self.row.get('reason', '')}".strip()

    @property
    def row_budget(self) -> str:
        r = self.row
        return (
            budget_of(num(r, "n_particles"), num(r, "n_paths"), num(r, "companion_paths"))
            if r
            else pc.BUDGETS["production"]
        )

    @property
    def row_commit(self) -> str:
        return str(self.row.get("git_commit", "")) if self.row else ""

    @property
    def s_mark(self) -> str:
        return (
            " [model S not converged]"
            if self.model_s and not bool(self.model_s.get("converged"))
            else ""
        )


def study_row(frame: pd.DataFrame, date: str, what: str) -> tuple[dict[str, Any] | None, str]:
    """The one row of ``frame`` for ``date`` (the caller has filtered the basket): exactly one,
    or none with the reason."""
    sel = frame[frame["date"].astype(str).str[:10] == date]
    if len(sel) != 1:
        return None, f"{what}: {len(sel)} rows for {date}"
    return dict(sel.iloc[0]), ""


def load_inputs(rows_dir: Path) -> dict[str, Inputs]:
    entries = pd.read_parquet(ENTRIES)
    b1 = entries[entries["basket"] == "B1"]
    model_s = pd.read_parquet(MODEL_S)
    refs: dict[str, tuple[str, dict[str, Any]]] = {}
    for tag in REF_TAGS:
        doc = json.loads((FIXTURES / f"{tag}.json").read_text())
        refs[str(doc["date"])] = (tag, doc)
    out = {}
    for date in pc.REFERENCE_DATES:
        path = rows_dir / f"{date}.json"
        row, why = load_json(path)
        if row is not None and str(row.get("date")) != date:
            row, why = None, f"{rel(path)} (its date is {row.get('date')})"
        entry, e_na = study_row(b1, date, "entries_3m.parquet, basket B1")
        ms, s_na = study_row(model_s, date, "model_s_3m.parquet")
        tag, ref = refs.get(date, ("", None))
        out[date] = Inputs(date, row, path, why, entry, e_na, ms, s_na, ref, tag)
        log.info(
            "%s: row %s; entry %s; model S %s; reference tag %s",
            date,
            "pending" if row is None else f"{row.get('status')} at {row.get('git_commit')}",
            "found" if entry else e_na,
            "missing" if ms is None else ("converged" if ms.get("converged") else "NOT converged"),
            tag or "none",
        )
    return out


# --------------------------------------------------------------------------- the core set

D_ED = "E[D], D = Σ w_i |R_i − R̄| at the horizon, in units of notional"
D_LISTED = "√(EQV/EV): the copula's κ on the listed dispersion variance over P_D (EQV and EV of the study's entry)"
D_WING = "ED_wing = κ_LC·√(Σ w E_LC[R_i²] − M_B^listed)"
D_EQV = "ED_eqv = κ_LC·√EQV"
D_KAPPA = "κ = E[D]/√E[V], V = Σ w_i (R_i − R̄)²"
D_CLIP = "largest share of particles, over the calibration slices, whose λ is clipped"
D_IDX = "index implied vol of the model minus the target's at the horizon, vol points"
D_DELTA = "percent of the model's own E[D] per +1 % on every spot (arithmetic bump of 1 %, central)"


def core_items(x: Inputs) -> dict[str, Item]:
    """The core set of a date, keyed as the record ids' suffixes (module docstring for the
    standard errors)."""
    r, e, s = x.row if x.priced else None, x.entry, x.model_s
    row_file = rel(x.row_path)
    row_kw: dict[str, Any] = {
        "budget": x.row_budget,
        "commit": x.row_commit,
        "source": row_file,
        "pending": x.row_pending if x.row is None else "",
        "na": x.row_na,
    }
    ent_kw: dict[str, Any] = {
        "budget": pc.BUDGETS["study"],
        "commit": STUDY_COMMIT,
        "source": rel(ENTRIES) + " (basket B1)",
        "na": x.entry_na,
    }
    s_kw: dict[str, Any] = {
        "budget": pc.BUDGETS["study"],
        "commit": STUDY_COMMIT,
        "source": rel(MODEL_S),
        "na": x.model_s_na,
        "mark": x.s_mark,
        "notes": "no standard error in the study's table"
        + ("; model S NOT converged on this date" if x.s_mark else ""),
    }
    mix_kw = {
        **row_kw,
        "budget": f"{x.row_budget}; denominator: {pc.BUDGETS['study']}",
        "source": f"{row_file}; {rel(ENTRIES)} (basket B1)",
    }
    if e is None and not mix_kw["pending"]:
        mix_kw["na"] = x.entry_na
    p_d, p_d_se, ev, eqv = (num(e, k) for k in ("P_D", "P_D_se", "EV", "EQV"))
    eqv_row = num(r, "EQV")
    out: dict[str, Item] = {}

    def put(key: str, label: str, value: float, se: float | None, kw: Mapping[str, Any], **more: Any) -> None:  # fmt: skip
        args = {**kw, **more}
        out[key] = Item(key, label, value, se, **args)

    def over_copula(a: float, a_se: float, b: float, b_se: float) -> tuple[float, float]:
        return div(a, b), pc.ratio_se(a, a_se, b, b_se) if ok(a) and ok(b) else NAN

    # --- the forward
    put("ED.lc", "E_LC[D]", num(r, "ED_lc"), num(r, "ED_lc_se"), row_kw, unit="notional", definition=D_ED + ", calibrated local correlation", digits=6)  # fmt: skip
    put("ED.cc", "E_CC[D]", num(r, "ED_cc"), num(r, "ED_cc_se"), row_kw, unit="notional", definition=D_ED + ", constant-correlation companion", digits=6)  # fmt: skip
    put("ED.copula", "P_D (copula)", p_d, p_d_se, ent_kw, unit="notional", definition=D_ED + ", the study's copula", digits=6)  # fmt: skip
    put("ED.model_s", "P_D_S (model S)", num(s, "P_D_S"), None, s_kw, unit="notional", definition=D_ED + ", the study's skewed model", digits=6)  # fmt: skip
    put("ED.wing", "ED_wing", num(r, "ED_wing"), num(r, "ED_wing_se"), row_kw, unit="notional", definition=D_WING, digits=6)  # fmt: skip
    put("ED.eqv", "ED_eqv", num(r, "ED_eqv"), num(r, "ED_eqv_se"), row_kw, unit="notional", definition=D_EQV, digits=6)  # fmt: skip
    # --- the ratios
    put("ratio.lc_over_cc", "LC/CC", num(r, "ratio"), num(r, "ratio_se"), row_kw, definition="E_LC[D]/E_CC[D], paired on the pricing paths (the row's ratio and ratio_se)", digits=5)  # fmt: skip
    for tag, name in (("lc", "LC"), ("cc", "CC")):
        v, se = over_copula(num(r, f"ED_{tag}"), num(r, f"ED_{tag}_se"), p_d, p_d_se)
        put(f"ratio.{tag}_over_copula", f"{name}/copula", v, se, mix_kw, definition=f"E_{name}[D]/P_D; delta method on ED_{tag}_se and P_D_se, independent", digits=5)  # fmt: skip
    put("ratio.s_over_copula", "S/copula", div(num(s, "P_D_S"), p_d), None, s_kw, definition="P_D_S/P_D", digits=5,
        source=f"{rel(MODEL_S)}; {rel(ENTRIES)} (basket B1)", na=x.model_s_na or x.entry_na)  # fmt: skip
    put("ratio.listed_fwd", "listed-variance forward √(EQV/EV)", math.sqrt(eqv / ev) if ok(eqv) and ok(ev) and ev > 0 and eqv >= 0 else NAN, None, ent_kw,
        definition=D_LISTED, digits=5, notes="no Monte Carlo number of this model enters; the entry gives no standard error for EV")  # fmt: skip
    for tag in ("wing", "eqv"):
        put(f"ratio.{tag}_over_cc", f"ED_{tag}/CC", num(r, f"ED_{tag}_ratio"), num(r, f"ED_{tag}_ratio_se"), row_kw,
            definition=f"{D_WING if tag == 'wing' else D_EQV}, over E_CC[D] (the row's ED_{tag}_ratio and its error)", digits=5)  # fmt: skip
        v, se = over_copula(num(r, f"ED_{tag}"), num(r, f"ED_{tag}_se"), p_d, p_d_se)
        put(f"ratio.{tag}_over_copula", f"ED_{tag}/copula", v, se, mix_kw,
            definition=f"{D_WING if tag == 'wing' else D_EQV}, over P_D; delta method on ED_{tag}_se and P_D_se, independent", digits=5)  # fmt: skip
    # --- kappa and E[V]/EQV
    put("kappa.lc", "κ under LC", num(r, "kappa_lc"), num(r, "kappa_lc_se"), row_kw, definition=D_KAPPA + " (the row's kappa_lc)")  # fmt: skip
    put("kappa.cc", "κ under CC", num(r, "kappa_cc"), num(r, "kappa_cc_se"), row_kw, definition=D_KAPPA + " (the row's kappa_cc)")  # fmt: skip
    put("kappa.copula", "κ under the copula", num(e, "kappa_cop"), None, ent_kw, definition="kappa_cop = P_D/√EV of the entry", notes="the entry gives no standard error for EV")  # fmt: skip
    ev_s = num(s, "EV_S")
    put("kappa.model_s", "κ under model S", num(s, "P_D_S") / math.sqrt(ev_s) if ok(ev_s) and ev_s > 0 else NAN, None, s_kw, definition="P_D_S/√EV_S")  # fmt: skip
    put("EV_over_EQV.lc", "E[V]/EQV under LC", num(r, "EV_over_EQV"), num(r, "EV_over_EQV_se"), row_kw, definition="EV_lc/EQV (the row's EV_over_EQV; error EV_lc_se/EQV)")  # fmt: skip
    put("EV_over_EQV.cc", "E[V]/EQV under CC", div(num(r, "EV_cc"), eqv_row), div(num(r, "EV_cc_se"), eqv_row), row_kw, definition="EV_cc/EQV; error EV_cc_se/EQV")  # fmt: skip
    put("EV_over_EQV.copula", "E[V]/EQV under the copula", div(ev, eqv), None, ent_kw, definition="EV/EQV of the entry", notes="the entry gives no standard error for EV")  # fmt: skip
    put("EV_over_EQV.model_s", "E[V]/EQV under model S", div(ev_s, eqv), None, s_kw, definition="EV_S/EQV (EQV of the entry)",
        source=f"{rel(MODEL_S)}; {rel(ENTRIES)} (basket B1)", na=x.model_s_na or x.entry_na)  # fmt: skip
    put("EV_split.single_over_EQV", "LC, single-name part / EQV", div(num(r, "EV_single_part"), eqv_row), div(num(r, "EV_single_part_se"), eqv_row), row_kw,
        definition="(Σ w E_LC[R_i²] − Σ w M_i^listed)/EQV (the row's EV_single_part over EQV)")  # fmt: skip
    put("EV_split.basket_over_EQV", "LC, basket part / EQV", div(num(r, "EV_basket_part"), eqv_row), div(num(r, "EV_basket_part_se"), eqv_row), row_kw,
        definition="(E_LC[R̄²] − M_B^listed)/EQV (the row's EV_basket_part over EQV); E_LC[V]/EQV − 1 = single-name part − basket part")  # fmt: skip
    put("EV_split.single", "LC, single-name part", num(r, "EV_single_part"), num(r, "EV_single_part_se"), row_kw, definition="Σ w E_LC[R_i²] − Σ w M_i^listed", digits=6)  # fmt: skip
    put("EV_split.basket", "LC, basket part", num(r, "EV_basket_part"), num(r, "EV_basket_part_se"), row_kw, definition="E_LC[R̄²] − M_B^listed", digits=6)  # fmt: skip
    m_b = num(r, "M_B_listed")
    put("Rbar2_over_listed.lc", "E_LC[R̄²]/M_B^listed", div(num(r, "E_Rbar2_lc"), m_b), div(num(r, "E_Rbar2_lc_se"), m_b), row_kw,
        definition="the basket's second moment under LC over its listed value (the row's E_Rbar2_lc / M_B_listed)")  # fmt: skip
    # --- the calls
    for m in pc.MULT_TAGS:
        k = num(e, f"K_{m}")
        put(f"call.K_{m}.strike", f"K_{m}", k, None, ent_kw, unit="notional", definition=f"{int(m) / 100:.2f} × P_D of the entry", digits=6, notes="a strike: no standard error")  # fmt: skip
        d_call = f"E[(D − K_{m})⁺], in units of notional"
        c_cop, c_cop_se = num(e, f"C_{m}"), num(e, f"C_se_{m}")
        put(f"call.K_{m}.lc", f"call K_{m}, LC", num(r, f"C_{m}_lc"), num(r, f"C_{m}_lc_se"), row_kw, unit="notional", definition=d_call + ", LC", digits=7)  # fmt: skip
        put(f"call.K_{m}.cc", f"call K_{m}, CC", num(r, f"C_{m}_cc"), num(r, f"C_{m}_cc_se"), row_kw, unit="notional", definition=d_call + ", CC", digits=7)  # fmt: skip
        put(f"call.K_{m}.copula", f"call K_{m}, copula", c_cop, c_cop_se, ent_kw, unit="notional", definition=d_call + ", copula", digits=7)  # fmt: skip
        put(f"call.K_{m}.model_s", f"call K_{m}, model S", num(s, f"C_S_{m}"), None, s_kw, unit="notional", definition=d_call + ", model S", digits=7)  # fmt: skip
        zero = "the copula's call is zero" if ok(c_cop) and c_cop <= 0.0 else ""
        for tag, name in (("lc", "LC"), ("cc", "CC")):
            v, se = (NAN, NAN) if zero else over_copula(num(r, f"C_{m}_{tag}"), num(r, f"C_{m}_{tag}_se"), c_cop, c_cop_se)  # fmt: skip
            kw = {**mix_kw, "na": zero} if zero and not mix_kw["pending"] else mix_kw
            put(f"call.K_{m}.{tag}_over_copula", f"call K_{m}, {name}/copula", v, se, kw,
                definition=f"the {name} call over the copula's; delta method on the two standard errors, independent")  # fmt: skip
        put(f"call.K_{m}.s_over_copula", f"call K_{m}, S/copula", NAN if zero else div(num(s, f"C_S_{m}"), c_cop), None, s_kw,
            definition="model S's call over the copula's", source=f"{rel(MODEL_S)}; {rel(ENTRIES)} (basket B1)", na=zero or x.model_s_na or x.entry_na)  # fmt: skip
        put(f"call.K_{m}.lc_over_cc", f"call K_{m}, LC/CC", num(r, f"C_{m}_ratio"), num(r, f"C_{m}_ratio_se"), row_kw,
            definition=f"the LC call over the CC call, paired (the row's C_{m}_ratio and its error)")  # fmt: skip
    # --- the index smile at the horizon and the clipped mass
    for key, col, where in (("atm", "idx_err_atm", "at the money"), ("90", "idx_err_90", "the 90 % strike"),
                            ("m15", "idx_err_m15", "−1.5 sd"), ("m20", "idx_err_m20", "−2 sd"), ("m25", "idx_err_m25", "−2.5 sd")):  # fmt: skip
        put(f"idx_err.{key}", f"index error, {where}", num(r, col), num(r, f"{col}_se"), row_kw, unit="vol points", definition=f"{D_IDX}; {where} (sd = at-the-money vol × √T)", digits=3)  # fmt: skip
    for key, col, where in (
        ("inner_low", "clip_low_inner_max", "inside ±2.5 sd, at λ = 0"),
        ("inner_high", "clip_high_inner_max", "inside ±2.5 sd, at the cap"),
        ("inner_total", "clip_inner_max", "inside ±2.5 sd, both sides"),
        ("cloud_low", "clip_low_max", "whole cloud, at λ = 0"),
        ("cloud_high", "clip_high_max", "whole cloud, at the cap"),
    ):
        v = num(r, col)
        put(f"clip.{key}", f"clipped mass, {where}", 100 * v if ok(v) else NAN, None, row_kw, unit="% of particles",
            definition=f"{D_CLIP}; {where} (the row's {col} × 100)", digits=3, notes="a calibration diagnostic: no standard error")  # fmt: skip
    # --- the deltas of the row (sticky strike)
    put("delta.lc_ss", "Δ sticky strike, LC", num(r, "delta_fwd_lc"), num(r, "delta_fwd_lc_se"), row_kw, unit="% of E[D] per +1 %", definition=D_DELTA + "; LC, sticky strike", digits=3)  # fmt: skip
    put("delta.cc_ss", "Δ sticky strike, CC", num(r, "delta_fwd_cc"), num(r, "delta_fwd_cc_se"), row_kw, unit="% of E[D] per +1 %", definition=D_DELTA + "; CC, sticky strike", digits=3)  # fmt: skip
    put("delta.homogeneity", "homogeneity", num(r, "delta_homogeneity"), None, row_kw, unit="% of E[D] per +1 %",
        definition="the sticky-moneyness delta, exactly 1 in the row (a common move at sticky moneyness scales D)", digits=3, notes="exact, not measured in the row")  # fmt: skip
    put("delta.skew_channel", "skew channel", num(r, "delta_skew_channel"), num(r, "delta_skew_channel_se"), row_kw, unit="% of E[D] per +1 %", definition="Δ^CC_ss − 1", digits=3)  # fmt: skip
    put("delta.correlation_channel", "correlation channel", num(r, "delta_correlation_channel"), num(r, "delta_correlation_channel_se"), row_kw,
        unit="% of E[D] per +1 %", definition="Δ^LC_ss − Δ^CC_ss, paired", digits=3)  # fmt: skip
    put("delta.C100_lc", "Δ sticky strike of the call K_100, LC", num(r, "delta_C100_lc"), num(r, "delta_C100_lc_se"), row_kw, unit="% of the call per +1 %", definition="percent of the model's own call at K_100 per +1 % on every spot; LC", digits=3)  # fmt: skip
    put("delta.C100_cc", "Δ sticky strike of the call K_100, CC", num(r, "delta_C100_cc"), num(r, "delta_C100_cc_se"), row_kw, unit="% of the call per +1 %", definition="percent of the model's own call at K_100 per +1 % on every spot; CC", digits=3)  # fmt: skip
    return out


def ref_items(x: Inputs) -> dict[str, Item]:
    """The parametric reference's numbers of a date (its own world: two-parameter λ, no carry)."""
    out: dict[str, Item] = {}
    f = x.ref
    kw: dict[str, Any] = {
        "budget": REF_BUDGET,
        "commit": REF_COMMIT,
        "source": rel(FIXTURES / f"{x.ref_tag}.json") if f else "",
        "na": "" if f else "no fixture for this date",
        "notes": f"tag {x.ref_tag}; {f['n_paths']} paths" if f else "",
    }
    a = f["anchors"] if f else {}

    def pair(key: str) -> tuple[float, float]:
        v = a.get(key)
        return (float(v[0]), float(v[1])) if v else (NAN, NAN)

    def put(key: str, label: str, vs: tuple[float, float], **more: Any) -> None:
        out[key] = Item(key, label, vs[0], vs[1], **{**kw, **more})

    put("ref.ED_lc", "reference E_LC[D]", pair("mom.lc.ED"), unit="notional", definition=D_ED + ", the reference's local correlation", digits=6)  # fmt: skip
    put("ref.ED_cc", "reference E_CC[D]", pair("mom.cc.ED"), unit="notional", definition=D_ED + ", the reference's constant correlation", digits=6)  # fmt: skip
    put("ref.lc_over_cc", "reference LC/CC", pair("forward_ratio"), definition="the reference's E_LC[D]/E_CC[D], paired", digits=5)  # fmt: skip
    put("ref.delta_lc", "reference Δ sticky strike, LC", pair("delta.lc"), unit="% of E[D] per +1 %", definition="the reference's sticky-strike delta: % change of E[D] per +1 % on every spot, central; LC", digits=3)  # fmt: skip
    put("ref.delta_cc", "reference Δ sticky strike, CC", pair("delta.cc"), unit="% of E[D] per +1 %", definition="the reference's sticky-strike delta: % change of E[D] per +1 % on every spot, central; CC", digits=3)  # fmt: skip
    for mult in ("0.75", "1.00", "1.25", "1.50"):
        c = (a.get("calls") or {}).get(mult)
        vs = (float(c["ratio"][0]), float(c["ratio"][1])) if c else (NAN, NAN)
        put(f"ref.call_{mult.replace('.', '')}.lc_over_cc", f"reference call at {mult} × its E_CC[D], LC/CC", vs,
            definition=f"the reference's LC call over its CC call at the strike {mult} × its own E_CC[D] (not the study's K), paired")  # fmt: skip
    return out


# --------------------------------------------------------------------------- shared pieces

FLAG_COLUMNS = (
    "date", "status", "reason", "git_commit", "budget", "calendar_repair", "n_names_unscreened", "names_unscreened",
    "n_dropped", "n_dropped_calendar", "n_dropped_calendar_index", "n_names_extrapolated", "index_extrapolated",
    "flag_unscreened", "flag_clip_low", "flag_clip_high", "sanity checks failing", "model_s_converged", "spec_key", "source",
)  # fmt: skip


def flags_row(x: Inputs) -> dict[str, Any]:
    """What the row says about itself: status, specification and flags (the flags recomputed
    from the base columns when the row does not carry them)."""
    r = x.row
    if r is None:
        return {"date": x.date, "status": f"pending: {x.row_pending}", "model_s_converged": None if x.model_s is None else bool(x.model_s.get("converged"))}  # fmt: skip
    low, high = num(r, "clip_low_inner_max"), num(r, "clip_high_inner_max")
    failing = [k for k, v in r.items() if k.startswith("check_") and v is False]
    return {
        "date": x.date, "status": r.get("status"), "reason": r.get("reason", ""), "git_commit": r.get("git_commit"),
        "budget": x.row_budget, "calendar_repair": r.get("calendar_repair"),
        "n_names_unscreened": r.get("n_names_unscreened"), "names_unscreened": r.get("names_unscreened", ""),
        "n_dropped": r.get("n_dropped"), "n_dropped_calendar": r.get("n_dropped_calendar"),
        "n_dropped_calendar_index": r.get("n_dropped_calendar_index"),
        "n_names_extrapolated": r.get("n_names_extrapolated"), "index_extrapolated": r.get("index_extrapolated"),
        "flag_unscreened": bool(num(r, "n_names_unscreened") > 0), "flag_clip_low": bool(low > CLIP_FLAG) if ok(low) else None,
        "flag_clip_high": bool(high > CLIP_FLAG) if ok(high) else None, "sanity checks failing": ", ".join(failing),
        "model_s_converged": None if x.model_s is None else bool(x.model_s.get("converged")),
        "spec_key": str(r.get("spec_key", ""))[:12], "source": rel(x.row_path),
    }  # fmt: skip


def show(v: Any) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "not in the row"
    if isinstance(v, bool):
        return "yes" if v else "no"
    return str(v) if str(v) != "" else "—"


def flags_table(part: Part, inputs: Sequence[Inputs], name: str, base: Path) -> None:
    rows = [flags_row(x) for x in inputs]
    frame = pd.DataFrame(rows).reindex(columns=list(FLAG_COLUMNS))
    pc.save_table(frame, name, base)
    labels = (
        ("status", "row status"), ("reason", "reason"), ("git_commit", "commit"), ("budget", "budget"),
        ("calendar_repair", "calendar repair of the names' slices on"),
        ("n_names_unscreened", "names kept unscreened (flagged date when > 0)"), ("names_unscreened", "which"),
        ("n_dropped", "quotes or slices dropped by the screen (n_dropped)"),
        ("n_dropped_calendar", "slices dropped by the calendar repair, names"),
        ("n_dropped_calendar_index", "slices dropped by the calendar repair, index"),
        ("n_names_extrapolated", "names priced beyond their last listed expiry"),
        ("index_extrapolated", "index target extrapolated"),
        ("flag_clip_low", "clipped mass at λ = 0 inside ±2.5 sd above 1 %"),
        ("flag_clip_high", "clipped mass at the cap inside ±2.5 sd above 1 %"),
        ("sanity checks failing", "sanity checks failing"), ("model_s_converged", "model S converged (study)"),
        ("spec_key", "specification key (first 12)"),
    )  # fmt: skip
    body = []
    for key, label in labels:
        cells = []
        for row in rows:
            if str(row.get("status", "")).startswith("pending") and key not in (
                "status",
                "model_s_converged",
            ):
                cells.append("pending")
            else:
                cells.append(show(row.get(key)))
        body.append([label, *cells])
    part.table(["", *[x.date for x in inputs]], body)
    part.text(
        f"What each row says about itself (`tables/{name}.csv`); the clip flags are recomputed from the row's clipped masses (strictly above 1 %). "
        f"Source: {rel(inputs[0].row_path.parent)}/<date>.json; model S: {rel(MODEL_S)}.",
        "",
    )


def consistency(inputs: Sequence[Inputs]) -> list[str]:
    """Facts checked on the inputs: each line says what was compared and the largest gap."""
    lines = []
    for x in inputs:
        r, e = x.row, x.entry
        if not x.priced or e is None or r is None:
            continue
        if x.row_budget != pc.BUDGETS["production"]:
            lines.append(
                f"- {x.date}: NOTE the row's budget is {x.row_budget}, not the production budget."
            )
        if r.get("calendar_repair") is not True:
            lines.append(
                f"- {x.date}: NOTE the row does not say the calendar repair is on (calendar_repair = {r.get('calendar_repair')})."
            )
        if "n_names_unscreened" not in r:
            lines.append(
                f"- {x.date}: NOTE the row has no n_names_unscreened (a row written before the owner's decision 2)."
            )
        if str(r.get("status")) != "ok":
            lines.append(
                f"- {x.date}: NOTE the row's status is {r.get('status')}: {r.get('reason', '')}."
            )
        gaps = {
            "P_D_copula vs P_D": abs(num(r, "P_D_copula") - num(e, "P_D")),
            "EV_copula vs EV": abs(num(r, "EV_copula") - num(e, "EV")),
            "EQV": abs(num(r, "EQV") - num(e, "EQV")),
            "T": abs(num(r, "T") - num(e, "T")),
            "rho_cop": abs(num(r, "rho_cop") - num(e, "rho_cop")),
            "K_<m>": max(abs(num(r, f"K_{m}") - num(e, f"K_{m}")) for m in pc.MULT_TAGS),
            "ED_lc/ED_cc vs ratio": abs(num(r, "ED_lc") / num(r, "ED_cc") - num(r, "ratio")),
            "EV_lc/EQV vs EV_over_EQV": abs(
                num(r, "EV_lc") / num(r, "EQV") - num(r, "EV_over_EQV")
            ),
        }
        for key in ("lc_over_copula", "cc_over_copula", "listed_fwd_ratio"):
            if key in r:
                mine = math.sqrt(num(e, "EQV") / num(e, "EV")) if key == "listed_fwd_ratio" else num(r, f"ED_{key[:2]}") / num(e, "P_D")  # fmt: skip
                gaps[f"the row's {key} vs recomputed"] = abs(num(r, key) - mine)
        worst = max(gaps.values())
        bad = {k: v for k, v in gaps.items() if not (v <= 1e-9)}
        if bad:
            lines.append(f"- {x.date}: MISMATCH between the row and the study's entry (B1): " + ", ".join(f"{k} {v:.3g}" for k, v in bad.items()) + ".")  # fmt: skip
            log.warning("%s: row and entry disagree: %s", x.date, bad)
        else:
            lines.append(f"- {x.date}: the row's T, P_D_copula, EV_copula, EQV, rho_cop and strikes equal the entry's (B1); ED_lc/ED_cc equals the row's ratio"
                         f"{'; the derived columns equal the recomputed ones' if 'lc_over_copula' in r else ''} (largest gap {worst:.1e}).")  # fmt: skip
    return lines


# --------------------------------------------------------------------------- section A


def one_line(kind: str, x: Inputs) -> str:
    """The line under a table: budget, commit and source of our row, and the study's files."""
    commit = x.row_commit or "pending"
    return (
        f"{kind} LC and CC: {x.row_budget}, commit {commit}, `{rel(x.row_path)}`. "
        f"Copula: `{rel(ENTRIES)}`, basket B1 (one row per date and basket); model S: `{rel(MODEL_S)}` (no standard errors in that table"
        f"{'; NOT converged on ' + x.date if x.s_mark else ''})."
    )


def risk_items(risk: dict[str, Any] | None, pending: str, row: Inputs) -> tuple[dict[str, Item], list[Item], list[str]]:  # fmt: skip
    """Today's risk: the deltas, the −3 sd index errors, the vegas in both units and the
    model-risk variants.  A piece absent from the file is pending."""
    out: dict[str, Item] = {}
    notes: list[str] = []
    src = rel(RISK_FILE) if pending == "" else pending
    budget = (
        budget_of(num(risk, "n_particles"), num(risk, "n_paths"), num(risk, "companion_paths"))
        if risk
        else pc.BUDGETS["production"]
    )
    if risk and not ok(num(risk, "n_particles")):
        budget = str(risk.get("budget", budget))
        budget = pc.BUDGETS.get(budget, budget)
    kw: dict[str, Any] = {"budget": budget, "commit": str(risk.get("commit", "")) if risk else "", "source": rel(RISK_FILE)}  # fmt: skip

    def piece(name: str) -> str:
        return (
            pending
            if risk is None
            else ("" if name in risk else f"{src} (piece {name!r} not in the file yet)")
        )

    def put(key: str, label: str, vs: Sequence[float] | None, wait: str, **more: Any) -> None:
        v, se = (float(vs[0]), float(vs[1])) if vs is not None and len(vs) >= 2 else (NAN, NAN)
        out[key] = Item(key, label, v, se, pending=wait, **{**kw, **more})

    d = (risk or {}).get("deltas") or {}
    wait = piece("deltas")
    for key, label, definition in (
        ("lc_ss", "Δ sticky strike, LC", "LC, sticky strike"),
        ("cc_ss", "Δ sticky strike, CC", "CC, sticky strike"),
        ("lc_sm", "Δ sticky moneyness, LC", "LC, sticky moneyness"),
        ("cc_sm", "Δ sticky moneyness, CC", "CC, sticky moneyness"),
        ("homogeneity", "homogeneity Δ^LC_sm", "the decomposition's homogeneity term: Δ^LC_sm"),
        ("skew_channel", "skew channel Δ^CC_ss − Δ^CC_sm", "skew channel: Δ^CC_ss − Δ^CC_sm"),
        ("correlation_channel", "correlation channel (Δ^LC_ss − Δ^LC_sm) − (Δ^CC_ss − Δ^CC_sm)", "correlation channel: (Δ^LC_ss − Δ^LC_sm) − (Δ^CC_ss − Δ^CC_sm)"),
        ("level_term", "level term Δ^LC_sm − Δ^CC_sm", "level term: Δ^LC_sm − Δ^CC_sm"),
    ):  # fmt: skip
        put(f"delta.risk.{key}", label, d.get(key), wait, unit="% of E[D] per +1 %", definition=f"{D_DELTA}; {definition}", digits=4,
            na="" if key in d or wait else "not in the file")  # fmt: skip
    wait = piece("index_errors_T")
    errs = {str(q.get("strike")): q for q in (risk or {}).get("index_errors_T") or []}
    for strike in ("-3.5", "-3.0", "-2.5"):
        q = errs.get(strike)
        put(f"idx_err.risk.m{strike[1:].replace('.', '')}", f"index error, −{strike[1:]} sd", None if q is None else (q["error_vp"], q["stderr_vp"]), wait,
            unit="vol points", definition=f"{D_IDX}; {strike} sd (today's risk run)", digits=3, na="" if q or wait else "not in the file")  # fmt: skip
    # the vegas: value = change of E_LC[D] in units of notional per vol point (per +1 rota)
    wait = piece("vegas")
    base = ((risk or {}).get("vegas_base") or (risk or {}).get("base") or {}).get("ED_lc")
    vegas = {str(v.get("name")): v for v in (risk or {}).get("vegas") or []}
    for key, name, per in (
        ("names_recalibrated", "vega[all names] recalibrated", "vol point on every name's surface, λ recalibrated to the unchanged index smile"),
        ("names_held", "vega[all names] held", "vol point on every name's surface, λ held"),
        ("index", "index vega", "vol point on the index target"),
        ("skew_rotation", "index skew vega (rotation)", "+1 rota of the index smile"),
        ("skew_put90", "index skew vega (put90)", "vol point at the 90 % strike of the index (tent)"),
    ):  # fmt: skip
        v = vegas.get(name)
        na = "" if v or wait else "not in the file"
        val, se = (float(v["value"]), float(v["stderr"])) if v else (NAN, NAN)
        unit = str(v.get("unit", "")) if v else ""
        put(f"vega.{key}.pct_notional", f"{name}, % of notional", (100 * val, 100 * se), wait, unit=f"% of notional {unit}".strip(),
            definition=f"100 × the change of E_LC[D] (units of notional) per {per}; paired on the pricing seed", na=na)  # fmt: skip
        if base and ok(val):
            pct = (
                100 * val / float(base[0]),
                100 * pc.ratio_se(val, se, float(base[0]), float(base[1])),
            )
        else:
            pct = (NAN, NAN)
        put(f"vega.{key}.pct_price", f"{name}, % of the price", pct, wait, unit=f"% of E_LC[D] {unit}".strip(),
            definition=f"100 × the change of E_LC[D] per {per}, over the run's base E_LC[D]; delta method on the two errors", digits=3,
            na=na or ("" if base else "the file has no base E_LC[D]"))  # fmt: skip
    # the model-risk range
    wait = piece("model_risk")
    mr = (risk or {}).get("model_risk") or {}
    rows = mr.get("rows") or []
    labels = list((risk or {}).get("model_risk_labels") or [r["label"] for r in rows])
    base_mr = ((risk or {}).get("model_risk_base") or (risk or {}).get("base") or {}).get("ED_lc")
    done = {str(r["label"]): r for r in rows}
    variants: list[Item] = []
    for i, label in enumerate(labels):
        r = done.get(label)
        w = wait or ("" if r else f"{src} (variant {label!r} not in the file yet)")
        slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        v = (float(r["price"]), float(r["stderr"])) if r else (NAN, NAN)
        dv = (float(r["minus_base"]), float(r["minus_base_se"])) if r else (NAN, NAN)
        variants.append(Item(f"model_risk.{i}_{slug}.price", f"{label}: E[D]", v[0], v[1], pending=w, unit="notional", digits=6,
                             definition=f"E[D] under the variant {label} (another R_low or family, recalibrated)", **kw))  # fmt: skip
        variants.append(Item(f"model_risk.{i}_{slug}.minus_base", f"{label}: minus base", dv[0], dv[1], pending=w, unit="notional", digits=6,
                             definition="the variant's E[D] minus the base E_LC[D], paired on the pricing seed", **kw))  # fmt: skip
        pct = (
            (100 * dv[0] / float(base_mr[0]), 100 * dv[1] / float(base_mr[0]))
            if base_mr and r
            else (NAN, NAN)
        )
        variants.append(Item(f"model_risk.{i}_{slug}.minus_base_pct", f"{label}: minus base, % of the price", pct[0], pct[1], pending=w, unit="% of E_LC[D]", digits=3,
                             definition="100 × (variant − base)/base E_LC[D]; the paired error of the difference over the base", **kw))  # fmt: skip
    missing = [lab for lab in labels if lab not in done]
    complete = bool(rows) and not missing
    n_txt = f"{len(rows)} of {len(labels)} variants" if labels else "no variant"
    wait_range = wait or (
        ""
        if complete
        else f"{src} ({n_txt} in the file: the range waits for {', '.join(missing) or 'the variants'})"
    )
    for key, label in (("low", "model-risk range, low"), ("high", "model-risk range, high")):
        out[f"model_risk.{key}"] = Item(f"model_risk.{key}", label, num(mr, key), None, pending=wait_range, unit="notional", digits=6,
                                        definition=f"the {'lowest' if key == 'low' else 'highest'} E[D] over the variants ({n_txt})",
                                        notes="an extreme over the variants: no standard error of its own", **kw)  # fmt: skip
    if base_mr:
        out["model_risk.base"] = Item("model_risk.base", "base E_LC[D] of the risk run", float(base_mr[0]), float(base_mr[1]), unit="notional", digits=6,
                                      definition=D_ED + ", the risk run's base", **kw)  # fmt: skip
    if risk is not None and row.priced and row.row is not None:
        if risk.get("spec_key") and risk.get("spec_key") != row.row.get("spec_key"):
            notes.append(f"- MISMATCH: the risk file's specification key {str(risk.get('spec_key'))[:12]} is not the row's {str(row.row.get('spec_key'))[:12]}.")  # fmt: skip
        b = (risk.get("base") or {}).get("ED_lc")
        if b:
            notes.append(f"- The risk run's base E_LC[D] {float(b[0]):.6f} ± {float(b[1]):.6f} against the row's {num(row.row, 'ED_lc'):.6f} ± {num(row.row, 'ED_lc_se'):.6f} "
                         f"(difference {float(b[0]) - num(row.row, 'ED_lc'):+.2e}); risk commit {risk.get('commit')}, row commit {row.row_commit}.")  # fmt: skip
        q = errs.get("-2.5")
        if q:
            notes.append(f"- Index error at −2.5 sd: risk run {float(q['error_vp']):+.3f} ± {float(q['stderr_vp']):.3f}, row {num(row.row, 'idx_err_m25'):+.3f} ± {num(row.row, 'idx_err_m25_se'):.3f} vol points.")  # fmt: skip
    return out, variants, notes


def cdv_block(part: Part, cdv: dict[str, Any] | None, pending: str, x: Inputs, risk_out: Mapping[str, Item], core: Mapping[str, Item]) -> None:  # fmt: skip
    """The cross-dependent scan at β = 3 and β = 6 next to the local correlation row (β = 0)."""
    part.text("### A8. Cross-dependent volatility (CDV) at β = 3 and β = 6 — check (e)", "")
    rec = (cdv or {}).get("record") or {}
    budget = budget_of(num(rec, "n_particles"), num(rec, "n_paths"), None) if cdv else ""
    if cdv and cdv.get("budget"):
        budget = f"{cdv['budget']} budget: {budget}"
    kw: dict[str, Any] = {
        "budget": budget,
        "commit": str(rec.get("git_commit", "")),
        "source": rel(CDV_FILE),
    }
    rows = {float(r["beta"]): r for r in (cdv or {}).get("rows") or []}
    betas = sorted(set(rows) | set(CDV_BETAS)) if cdv else list(CDV_BETAS)
    attributions = {float(a["beta"]): a for a in (cdv or {}).get("attribution_from_beta_0") or []}
    e = x.entry
    p_d, p_d_se = num(e, "P_D"), num(e, "P_D_se")
    ed_cc_file = num(cdv, "E_CC_D_of_the_M12_row")
    # whose E_CC[D] the scan divided by: looked up among the rows at hand, for its standard error
    cc_se, cc_src = NAN, "the scan's file gives no error for its E_CC[D]: numerator's error only"
    candidates = [
        x.row_path,
        DEV_ROWS_DIR / f"{x.date}.json",
        pc.LC_OUT / "rows" / "3m_development" / f"{x.date}.json",
    ]
    for cand in candidates:
        doc, _ = load_json(cand)
        if doc is not None and ok(ed_cc_file) and abs(num(doc, "ED_cc") - ed_cc_file) <= 1e-12:
            cc_se, cc_src = (
                num(doc, "ED_cc_se"),
                f"E_CC[D] of {rel(cand)} (commit {doc.get('git_commit')}), unpaired",
            )
            break
    spec = (
        ("ED", "E[D]", "notional", 6, D_ED + ", cross-dependent model at this β"),
        ("ED_over_cc", "E[D]/E_CC[D]", "", 5, "E[D] over the E_CC[D] the scan read"),
        ("ED_over_copula", "E[D]/P_D (copula)", "", 5, "E[D] over the copula's P_D; delta method, independent"),
        ("ED_over_beta0", "E[D]/E[D](β = 0)", "", 5, "exp of the scan's paired log-change of E[D] from β = 0 (attribution_from_beta_0.dln_ED)"),
        ("kappa", "κ", "", 4, D_KAPPA),
        ("EV_over_EQV", "E[V]/EQV", "", 4, "E[V]/EQV (error E[V]'s over EQV)"),
        ("names_part", "single-name part", "", 6, "Σ w E[R_i²] − Σ w M_i^listed"),
        ("basket_part", "basket part", "", 6, "E[R̄²] − M_B^listed; E[V] − EQV = single-name part − basket part"),
        ("E_Rbar2_over_listed", "E[R̄²]/M_B^listed", "", 4, "the basket's second moment over its listed value"),
        ("clip_low_inner", "clipped mass inside ±2.5 sd, at λ = 0", "% of particles", 3, D_CLIP + "; inside ±2.5 sd, at λ = 0"),
        ("clip_high_inner", "clipped mass inside ±2.5 sd, at the cap", "% of particles", 3, D_CLIP + "; inside ±2.5 sd, at the cap"),
        ("clip_low", "clipped mass, whole cloud, at λ = 0", "% of particles", 3, D_CLIP + "; whole cloud, at λ = 0"),
        ("clip_high", "clipped mass, whole cloud, at the cap", "% of particles", 3, D_CLIP + "; whole cloud, at the cap"),
        ("idx_-2.5", "index error, −2.5 sd", "vol points", 3, D_IDX + "; −2.5 sd"),
        ("idx_-3.0", "index error, −3.0 sd", "vol points", 3, D_IDX + "; −3.0 sd"),
        ("idx_-3.5", "index error, −3.5 sd", "vol points", 3, D_IDX + "; −3.5 sd"),
        ("idx_+0.0", "index error, at the money", "vol points", 3, D_IDX + "; at the money"),
        ("idx_90%", "index error, 90 % strike", "vol points", 3, D_IDX + "; the 90 % strike"),
    )  # fmt: skip
    base_col = {
        "ED": core["ED.lc"], "ED_over_cc": core["ratio.lc_over_cc"], "ED_over_copula": core["ratio.lc_over_copula"],
        "kappa": core["kappa.lc"], "EV_over_EQV": core["EV_over_EQV.lc"], "names_part": core["EV_split.single"],
        "basket_part": core["EV_split.basket"], "E_Rbar2_over_listed": core["Rbar2_over_listed.lc"],
        "clip_low_inner": core["clip.inner_low"], "clip_high_inner": core["clip.inner_high"], "clip_low": core["clip.cloud_low"],
        "clip_high": core["clip.cloud_high"], "idx_-2.5": core["idx_err.m25"], "idx_-3.0": risk_out["idx_err.risk.m30"],
        "idx_-3.5": risk_out["idx_err.risk.m35"], "idx_+0.0": core["idx_err.atm"], "idx_90%": core["idx_err.90"],
    }  # fmt: skip
    body: list[list[str]] = []
    cells: dict[tuple[str, float], str] = {}
    own_ratio = True
    g_max = sorted({num(q, "g_max") for q in rows.values() if ok(num(q, "g_max"))})
    for beta in betas:
        r = rows.get(beta)
        wait = (
            pending
            if cdv is None
            else ("" if r else f"{rel(CDV_FILE)} (no β = {beta:g} row in the file)")
        )
        items = []
        for key, label, unit, digits, definition in spec:
            v: float
            se: float | None
            na, note, wait_cc = "", "", ""
            if key == "ED_over_cc":
                v, se = num(r, "ED_over_cc"), NAN
                if ok(v):
                    se = pc.ratio_se(num(r, "ED"), num(r, "ED_se"), ed_cc_file, cc_se if ok(cc_se) else 0.0)  # fmt: skip
                    definition = f"{definition}: {cc_src}; delta method"
                elif r is not None and x.row is None:
                    # the scan read no E_CC[D]: the ratio is to section A's row, not arrived yet
                    wait_cc = x.row_pending
                elif r is not None and x.priced:
                    cc, cc_err = num(x.row, "ED_cc"), num(x.row, "ED_cc_se")
                    v = div(num(r, "ED"), cc)
                    se = pc.ratio_se(num(r, "ED"), num(r, "ED_se"), cc, cc_err)
                    definition = f"E[D] over E_CC[D] of section A's row {rel(x.row_path)} (the scan's file holds no E_CC[D]); delta method on the two errors, unpaired"
                    own_ratio = False
            elif key == "ED_over_copula":
                v = div(num(r, "ED"), p_d)
                se = pc.ratio_se(num(r, "ED"), num(r, "ED_se"), p_d, p_d_se) if ok(v) else NAN
            elif key == "ED_over_beta0":
                a, r0 = attributions.get(beta), rows.get(0.0)
                if ok(num(a, "dln_ED")):
                    v = math.exp(num(a, "dln_ED"))
                    se = v * num(a, "dln_ED_se") if ok(num(a, "dln_ED_se")) else None
                elif beta == 0.0 and r:
                    v, se, definition = 1.0, None, "the β = 0 row itself"
                    note = "1 by definition"
                elif r0 is not None and ok(num(r, "ED")):
                    v = div(num(r, "ED"), num(r0, "ED"))
                    se = pc.ratio_se(num(r, "ED"), num(r, "ED_se"), num(r0, "ED"), num(r0, "ED_se"))
                    definition = "E[D] at this β over E[D] of the scan's β = 0 row; delta method on the two errors, unpaired (the file has no paired attribution)"
                else:
                    v, se = NAN, None
                    na = "no β = 0 row in the scan"
            elif key.startswith("clip"):
                v, se = 100 * num(r, key), None
                note = "a calibration diagnostic: no standard error"
            else:
                v, se = num(r, key), num(r, f"{key}_se")
            if not ok(v) and not na:
                na = "not in the file"
            it = Item(f"cdv.beta_{beta:g}.{key.replace('idx_', 'idx_err_').replace('%', 'pct').replace('+', 'p').replace('-', 'm').replace('.', '')}",
                      f"CDV β = {beta:g}: {label}", v, se, unit=unit, definition=definition, digits=digits, pending=wait or wait_cc, na=na, notes=note, **kw)  # fmt: skip
            items.append(it)
            cells[(key, beta)] = it.cell()
        part.add("A_cdv", "A.", x.date, items)
    shown = [k for k in spec if k[0] not in ("idx_+0.0", "idx_90%")]
    for key, label, unit, _digits, _definition in shown:
        ref = base_col.get(key)
        body.append([f"{label}{' (' + unit + ')' if unit else ''}", ref.cell() if ref else "—", *[cells[(key, b)] for b in betas]])  # fmt: skip
    part.table(
        [
            "quantity",
            "LC (section A's row; −3 and −3.5 sd from the risk run)",
            *[f"CDV β = {b:g}" for b in betas],
        ],
        body,
    )
    if cdv is None:
        part.text(
            f"Pending: {pending}. The budget (production or development) is the one the file will state.",
            "",
        )
    else:
        part.text(
            f"Cross-dependent volatility prototype (`scripts/cdv_scan.py` of the volsto-cdv worktree), {budget}, commit {rec.get('git_commit')}, `{rel(CDV_FILE)}`; "
            f"g_max = {', '.join(f'{g:g}' for g in g_max) or 'not stated'}; β = 0 is the local correlation model. "
            f"E[D]/E_CC[D] is {'the ratio the scan wrote (' + cc_src + ')' if own_ratio else 'over E_CC[D] of the row of section A, unpaired (the scan file holds no E_CC[D])'}; E[D]/E[D](β = 0) is "
            f"{'paired on the scan pricing paths (attribution_from_beta_0)' if attributions else 'unpaired (the file has no paired attribution)'}. "
            "The split follows the row's convention: E[V] − EQV = single-name part − basket part. "
            f"M12 specification key of the scan: {str(rec.get('spec_key', 'not stated'))[:12]}"
            f"{'' if x.row is None else (' (the same as the row of section A)' if rec.get('spec_key') == x.row.get('spec_key') else ' (NOT the key of the row of section A, ' + str(x.row.get('spec_key', ''))[:12] + ')')}. "
            f"The LC column is section A's production row ({x.row_budget}, commit {x.row_commit or 'pending'}).",
            "",
        )


def build_a(inputs: Mapping[str, Inputs], risk: dict[str, Any] | None, risk_pending: str, cdv: dict[str, Any] | None,
            cdv_pending: str, base: Path, rows_dir: Path) -> Part:  # fmt: skip
    x = inputs[TODAY]
    part = Part("A")
    c = core_items(x)
    r_out, variants, risk_notes = risk_items(risk, risk_pending, x)
    # no section heading here: scripts/pm_assemble.py writes "## A. Today (2026-10-02), 3m, production budget"
    if rows_dir.resolve() != ROWS_DIR.resolve():
        part.text(
            f"**STAND-IN ROWS ({rel(rows_dir)}): a test of the code path, not the package's numbers.**",
            "",
        )
    part.text(
        "Cells are value ± standard error; `pending: <file>` marks a number whose input has not arrived.",
        "",
    )
    flags_table(part, [x], "A_flags", base)

    def cell(key: str) -> str:
        return c[key].cell()

    # A1
    part.text("### A1. E[D] under the four models", "")
    part.add(
        "A_forward", "A.", TODAY, [c[k] for k in ("ED.lc", "ED.cc", "ED.copula", "ED.model_s")]
    )
    part.table(["model", "E[D] (units of notional)"],
               [["LC (calibrated local correlation)", cell("ED.lc")], ["CC (constant-correlation companion)", cell("ED.cc")],
                ["copula (P_D)", cell("ED.copula")], ["model S (P_D_S)", cell("ED.model_s")]])  # fmt: skip
    part.text(
        f"D = Σ w_i |R_i − R̄| at the horizon, basket B1 (price weights). {one_line('', x).strip()}",
        "",
    )
    # A2
    part.text("### A2. Ratios", "")
    keys = ("ratio.lc_over_cc", "ratio.lc_over_copula", "ratio.cc_over_copula", "ratio.s_over_copula", "ratio.listed_fwd",
            "ratio.wing_over_cc", "ratio.wing_over_copula", "ratio.eqv_over_cc", "ratio.eqv_over_copula", "ED.wing", "ED.eqv")  # fmt: skip
    part.add("A_ratios", "A.", TODAY, [c[k] for k in keys])
    part.table(["ratio", "value"], [[c[k].label, cell(k)] for k in keys[:5]])
    part.table(["estimate", "level (units of notional)", "over CC", "over the copula"],
               [["ED_wing", cell("ED.wing"), cell("ratio.wing_over_cc"), cell("ratio.wing_over_copula")],
                ["ED_eqv", cell("ED.eqv"), cell("ratio.eqv_over_cc"), cell("ratio.eqv_over_copula")]])  # fmt: skip
    part.text(
        "LC/CC = E_LC[D]/E_CC[D], paired on the pricing paths; LC/copula = E_LC[D]/P_D, CC/copula = E_CC[D]/P_D, S/copula = P_D_S/P_D "
        "(errors over the copula: delta method on the numerator's error and P_D_se, independent; S/copula has none, model S's table has no errors). "
        f"Listed-variance forward = {D_LISTED}: no Monte Carlo number of this model, no error. {D_WING}; {D_EQV}; over CC with the row's own errors. "
        + one_line("", x).strip(),
        "",
    )
    # A3
    part.text("### A3. κ and E[V]/EQV", "")
    keys = ("kappa.lc", "kappa.cc", "kappa.copula", "kappa.model_s", "EV_over_EQV.lc", "EV_over_EQV.cc", "EV_over_EQV.copula", "EV_over_EQV.model_s",
            "EV_split.single_over_EQV", "EV_split.basket_over_EQV", "EV_split.single", "EV_split.basket", "Rbar2_over_listed.lc")  # fmt: skip
    part.add("A_kappa_ev", "A.", TODAY, [c[k] for k in keys])
    part.table(["", "LC", "CC", "copula", "model S"],
               [["κ", cell("kappa.lc"), cell("kappa.cc"), cell("kappa.copula"), cell("kappa.model_s")],
                ["E[V]/EQV", cell("EV_over_EQV.lc"), cell("EV_over_EQV.cc"), cell("EV_over_EQV.copula"), cell("EV_over_EQV.model_s")]])  # fmt: skip
    part.table(["LC's E[V] against EQV", "value", "over EQV"],
               [["single-name part: Σ w E_LC[R_i²] − Σ w M_i^listed", cell("EV_split.single"), cell("EV_split.single_over_EQV")],
                ["basket part: E_LC[R̄²] − M_B^listed", cell("EV_split.basket"), cell("EV_split.basket_over_EQV")],
                ["E_LC[R̄²]/M_B^listed", cell("Rbar2_over_listed.lc"), "—"]])  # fmt: skip
    part.text(
        f"{D_KAPPA}; under the copula kappa_cop = P_D/√EV, under model S P_D_S/√EV_S (no errors in the study's tables). "
        "E[V]/EQV = EV_lc/EQV, EV_cc/EQV, EV/EQV, EV_S/EQV with EQV the listed value (single-name strips minus the index strip). "
        "The split: E_LC[V] − EQV = single-name part − basket part. " + one_line("", x).strip(),
        "",
    )
    # A4
    part.text("### A4. Calls at K_050 … K_200", "")
    call_keys = (
        "strike",
        "lc",
        "cc",
        "copula",
        "model_s",
        "lc_over_copula",
        "cc_over_copula",
        "s_over_copula",
        "lc_over_cc",
    )
    part.add(
        "A_calls", "A.", TODAY, [c[f"call.K_{m}.{k}"] for m in pc.MULT_TAGS for k in call_keys]
    )
    part.table(["strike", "K", "LC", "CC", "copula", "model S"],
               [[f"K_{m}", *[cell(f"call.K_{m}.{k}") for k in call_keys[:5]]] for m in pc.MULT_TAGS])  # fmt: skip
    part.table(["strike", "LC/copula", "CC/copula", "S/copula", "LC/CC (paired)"],
               [[f"K_{m}", *[cell(f"call.K_{m}.{k}") for k in call_keys[5:]]] for m in pc.MULT_TAGS])  # fmt: skip
    part.text(
        "Call = E[(D − K)⁺] in units of notional, K_m = m % of the copula's P_D. Ratios to the copula: delta method on the two standard errors, independent "
        "(S/copula has none); a ratio is n/a when the copula's call is zero. "
        + one_line("", x).strip(),
        "",
    )
    # A5
    part.text("### A5. Index smile errors at the horizon and the clipped mass", "")
    idx = ("idx_err.atm", "idx_err.90", "idx_err.m15", "idx_err.m20", "idx_err.m25")
    part.add("A_index_clip", "A.", TODAY, [c[k] for k in idx])
    part.add(
        "A_index_clip",
        "A.",
        TODAY,
        [r_out["idx_err.risk.m30"], r_out["idx_err.risk.m35"], r_out["idx_err.risk.m25"]],
    )
    part.table(["strike", "model − target (vol points)", "source"],
               [*[[c[k].label.replace("index error, ", ""), cell(k), "row"] for k in idx],
                ["−3 sd", r_out["idx_err.risk.m30"].cell(), "risk run"]])  # fmt: skip
    clip = (
        "clip.inner_low",
        "clip.inner_high",
        "clip.inner_total",
        "clip.cloud_low",
        "clip.cloud_high",
    )
    part.add("A_index_clip", "A.", TODAY, [c[k] for k in clip])
    part.table(
        ["clipped mass", "% of particles"],
        [[c[k].label.replace("clipped mass, ", ""), cell(k)] for k in clip],
    )
    part.text(
        f"Index error = {D_IDX}; sd = at-the-money vol × √T. ATM, 90 % and −1.5 to −2.5 sd from the row ({x.row_budget}, commit {x.row_commit or 'pending'}, `{rel(x.row_path)}`); "
        f"−3 sd from today's risk run (`{rel(RISK_FILE)}`{', commit ' + str(risk.get('commit')) if risk else ''}). "
        f"Clipped mass = {D_CLIP} (at λ = 0, at the cap), inside ±2.5 sd and on the whole cloud: a calibration diagnostic, no standard error.",
        "",
    )
    # A6
    part.text("### A6. Common deltas", "")
    row_keys = (
        "delta.lc_ss",
        "delta.cc_ss",
        "delta.homogeneity",
        "delta.skew_channel",
        "delta.correlation_channel",
        "delta.C100_lc",
        "delta.C100_cc",
    )
    risk_keys = ("delta.risk.lc_ss", "delta.risk.cc_ss", "delta.risk.lc_sm", "delta.risk.cc_sm", "delta.risk.homogeneity", "delta.risk.skew_channel",
                 "delta.risk.correlation_channel", "delta.risk.level_term")  # fmt: skip
    part.add("A_deltas", "A.", TODAY, [c[k] for k in row_keys])
    part.add("A_deltas", "A.", TODAY, [r_out[k] for k in risk_keys])
    part.table(["delta of the forward (% of E[D] per +1 %)", "LC", "CC", "source"],
               [["sticky strike", cell("delta.lc_ss"), cell("delta.cc_ss"), "row"],
                ["sticky strike", r_out["delta.risk.lc_ss"].cell(), r_out["delta.risk.cc_ss"].cell(), "risk run"],
                ["sticky moneyness", r_out["delta.risk.lc_sm"].cell(), r_out["delta.risk.cc_sm"].cell(), "risk run"]])  # fmt: skip
    part.table(["decomposition of Δ^LC sticky strike", "row (homogeneity exactly 1)", "risk run (sticky-moneyness deltas measured)"],
               [["homogeneity", cell("delta.homogeneity"), r_out["delta.risk.homogeneity"].cell()],
                ["skew channel", cell("delta.skew_channel"), r_out["delta.risk.skew_channel"].cell()],
                ["correlation channel", cell("delta.correlation_channel"), r_out["delta.risk.correlation_channel"].cell()],
                ["level term Δ^LC_sm − Δ^CC_sm", "—", r_out["delta.risk.level_term"].cell()]])  # fmt: skip
    part.table(["delta of the call at K_100 (% of the call per +1 %)", "LC", "CC", "source"],
               [["sticky strike", cell("delta.C100_lc"), cell("delta.C100_cc"), "row"]])  # fmt: skip
    part.text(
        f"Elasticities: {D_DELTA}. Row: homogeneity = 1 exactly, skew channel = Δ^CC_ss − 1, correlation channel = Δ^LC_ss − Δ^CC_ss (paired); Δ^LC_ss is their sum. "
        "Risk run: homogeneity = Δ^LC_sm, skew channel = Δ^CC_ss − Δ^CC_sm, correlation channel = (Δ^LC_ss − Δ^LC_sm) − (Δ^CC_ss − Δ^CC_sm); Δ^LC_ss is their sum. "
        f"Row: {x.row_budget}, commit {x.row_commit or 'pending'}, `{rel(x.row_path)}`; risk run: `{rel(RISK_FILE)}`"
        f"{', ' + r_out['delta.risk.lc_ss'].budget + ', commit ' + str(risk.get('commit')) if risk else ''}.",
        "",
    )
    # A7
    part.text("### A7. Vegas and the model-risk range", "")
    vkeys = ("names_recalibrated", "names_held", "index", "skew_rotation", "skew_put90")
    part.add(
        "A_vegas",
        "A.",
        TODAY,
        [r_out[f"vega.{k}.{u}"] for k in vkeys for u in ("pct_price", "pct_notional")],
    )
    names = {"names_recalibrated": "all names +1 vol point, λ recalibrated", "names_held": "all names +1 vol point, λ held",
             "index": "index +1 vol point", "skew_rotation": "index skew, rotation (per +1 rota)", "skew_put90": "index skew, put90 (per vol point at the 90 % strike)"}  # fmt: skip
    part.table(["vega of E_LC[D]", "% of the price", "% of notional"],
               [[names[k], r_out[f"vega.{k}.pct_price"].cell(), r_out[f"vega.{k}.pct_notional"].cell()] for k in vkeys])  # fmt: skip
    part.text(
        "Each vega is the change of E_LC[D] per vol point (per +1 rota for the rotation), paired on the pricing seed: % of notional = 100 × the change in units of notional; "
        "% of the price = 100 × the change over the run's base E_LC[D] (delta method). "
        f"`{rel(RISK_FILE)}`{', ' + r_out['vega.index.pct_price'].budget + ', commit ' + str(risk.get('commit')) if risk else ''}. Per-name vegas are not reported.",
        "",
    )
    range_items = [r_out[k] for k in ("model_risk.low", "model_risk.high") if k in r_out]
    if "model_risk.base" in r_out:
        range_items.append(r_out["model_risk.base"])
    part.add("A_model_risk", "A.", TODAY, range_items)
    part.add("A_model_risk", "A.", TODAY, variants)
    body = [["base (the calibrated model)", r_out["model_risk.base"].cell() if "model_risk.base" in r_out else f"pending: {risk_pending}" if risk is None else "n/a", "—", "—"]]  # fmt: skip
    for i in range(0, len(variants), 3):
        body.append(
            [
                variants[i].label.replace(": E[D]", ""),
                variants[i].cell(),
                variants[i + 1].cell(),
                variants[i + 2].cell(),
            ]
        )
    body.append(["range, low", r_out["model_risk.low"].cell(), "—", "—"])
    body.append(["range, high", r_out["model_risk.high"].cell(), "—", "—"])
    part.table(
        [
            "variant (R_low / family, recalibrated)",
            "E[D]",
            "minus base",
            "minus base, % of the price",
        ],
        body,
    )
    part.text(
        "Model-risk range = the lowest and the highest E[D] over the variants (another R_low or family, each recalibrated to the same index smile); "
        "the differences to the base are paired on the pricing seed. "
        f"`{rel(RISK_FILE)}`{', ' + r_out['model_risk.low'].budget + ', commit ' + str(risk.get('commit')) if risk else ''}.",
        "",
    )
    # A8
    cdv_block(part, cdv, cdv_pending, x, r_out, c)
    checks = consistency([x]) + risk_notes
    if checks:
        part.text("### A. Consistency of the inputs", "", *checks, "")
    return part


# --------------------------------------------------------------------------- section B


def parse_rebuild(path: Path) -> dict[str, dict[str, float]]:
    """The independent rebuild of the reference on the library, by date: its LC/CC and error,
    the reference's and the z-score, from the verification agent's log."""
    out: dict[str, dict[str, float]] = {}
    if not path.exists():
        return out
    pat = re.compile(
        r"tag (\S+) date (\S+) .*?n_paths (\d+).*?LC/CC ([\d.]+) \(([\d.]+)\) ref ([\d.]+) \(([\d.]+)\); z ([-+\d.]+)"
    )
    for line in path.read_text().splitlines():
        m = pat.search(line)
        if m:
            out[m.group(2)] = {"n_paths": float(m.group(3)), "ratio": float(m.group(4)), "se": float(m.group(5)),
                               "ref": float(m.group(6)), "ref_se": float(m.group(7)), "z": float(m.group(8))}  # fmt: skip
    return out


def check_a(part: Part, inputs: Mapping[str, Inputs], rebuild_log: Path) -> None:
    """Check (a): the labels of the reference dates."""
    part.text("### B3. Check (a): the reference-date labels", "")
    a_date, b_date = "2017-04-03", "2019-09-03"
    old: dict[str, dict[str, Any]] = {}
    for d in (a_date, b_date):
        doc, _ = load_json(OLD_ROWS_DIR / f"{d}.json")
        if doc is not None:
            old[d] = doc
    rebuild = parse_rebuild(rebuild_log)
    items: list[tuple[str, Item]] = []
    body = []
    labels_ok = []
    for d in (b_date, a_date):
        x = inputs[d]
        o = old.get(d)
        ref = ref_items(x)["ref.lc_over_cc"]
        it_old = Item("check_a.lc_over_cc.old_defaults", "LC/CC, old defaults (the number of the check)", num(o, "ratio"), num(o, "ratio_se"), digits=5,
                      definition="E_LC[D]/E_CC[D], paired; production row on the old defaults (no repair, no fallback)",
                      budget=budget_of(num(o, "n_particles"), num(o, "n_paths"), num(o, "companion_paths")) if o else "",
                      commit=str(o.get("git_commit", "")) if o else "", source=rel(OLD_ROWS_DIR / f"{d}.json"),
                      pending="" if o else rel(OLD_ROWS_DIR / f"{d}.json"))  # fmt: skip
        new = core_items(x)["ratio.lc_over_cc"]
        it_new = Item("check_a.lc_over_cc.decisions_on", "LC/CC, decisions on (section B's row)", new.value, new.se, digits=5, definition=new.definition,
                      budget=new.budget, commit=new.commit, source=new.source, pending=new.pending, na=new.na)  # fmt: skip
        it_ref = Item("check_a.lc_over_cc.reference", f"parametric reference LC/CC (tag {x.ref_tag})", ref.value, ref.se, digits=5, definition=ref.definition,
                      budget=ref.budget, commit=ref.commit, source=ref.source, notes=ref.notes)  # fmt: skip
        rb = rebuild.get(d)
        it_rb = Item("check_a.lc_over_cc.rebuild", "the reference's world rebuilt on the library, LC/CC", rb["ratio"] if rb else NAN, rb["se"] if rb else NAN, digits=5,
                     definition="E_LC[D]/E_CC[D] of the fixture's world rebuilt with the library (tests/_lcm_reference.py), paired; an independent verification run",
                     budget=f"{rb['n_paths']:g} paths, seed 2024" if rb else "", commit="d4faa74 worktree (verification run of 2026-10-09)", source=str(rebuild_log),
                     pending="" if rb else str(rebuild_log), notes=f"z against the fixture {rb['z']:+.2f}" if rb else "")  # fmt: skip
        f, e = x.ref, x.entry
        t_gap = abs(float(f["T"]) - num(e, "T")) if f and e else NAN
        sig_gap = abs(float(f["targets"]["sigB"]) - num(e, "sig_B_DJX")) if f and e else NAN
        row_gap = max(abs(num(o, k) - num(e, j)) for k, j in (("T", "T"), ("P_D_copula", "P_D"), ("EQV", "EQV"), ("rho_cop", "rho_cop"))) if o and e else NAN  # fmt: skip
        labels_ok.append(
            bool(
                f and str(f["date"]) == d and t_gap <= 1e-9 and sig_gap <= 1e-9 and row_gap <= 1e-9
            )
        )
        it_gap_f = Item("check_a.label_gap.fixture", "fixture vs the study's entry of the date: largest gap of T and the index ATM vol", max(t_gap, sig_gap) if ok(t_gap) and ok(sig_gap) else NAN, None,
                        digits=12, definition="max(|T_fixture − T_entry|, |sigB_fixture − sig_B_DJX_entry|), entry of basket B1", budget="exact comparison", commit=STUDY_COMMIT,
                        source=f"{ref.source}; {rel(ENTRIES)}")  # fmt: skip
        it_gap_r = Item("check_a.label_gap.row", "old-default row vs the study's entry of the date: largest gap of T, P_D, EQV, rho_cop", row_gap, None, digits=12,
                        definition="max over T, P_D_copula/P_D, EQV, rho_cop of |row − entry|, entry of basket B1", budget="exact comparison",
                        commit=str(o.get("git_commit", "")) if o else "", source=f"{rel(OLD_ROWS_DIR / f'{d}.json')}; {rel(ENTRIES)}",
                        pending="" if o else rel(OLD_ROWS_DIR / f"{d}.json"))  # fmt: skip
        for it in (it_old, it_new, it_ref, it_rb, it_gap_f, it_gap_r):
            items.append((d, it))
        body.append([d, x.ref_tag, it_old.cell(), it_new.cell(), it_ref.cell(), it_rb.cell()])
    for d, it in items:
        part.add("B_check_a", f"B.{d}.", d, [it])
    part.table(["date", "reference tag", "M12 LC/CC, old defaults (the 0.94535 of the check)", "M12 LC/CC, decisions on (section B)", "parametric reference LC/CC", "reference world rebuilt on the library"], body)  # fmt: skip
    o_a, o_b = old.get(a_date), old.get(b_date)
    r_a, r_b = (
        ref_items(inputs[a_date])["ref.lc_over_cc"],
        ref_items(inputs[b_date])["ref.lc_over_cc"],
    )
    if (
        o_a
        and o_b
        and ok(r_a.value)
        and ok(r_b.value)
        and r_a.value is not None
        and r_b.value is not None
    ):
        m_a, m_b = num(o_a, "ratio"), num(o_b, "ratio")
        if all(labels_ok):
            head = "**Answer: the labels are right; the agreement within 0.0001 is a coincidence between two different dates in two different models.**"
            match = (
                f"Each M12 row's T, P_D, EQV and rho_cop, and each fixture's date, T and index at-the-money vol, equal those of the study's B1 entry of its own date "
                f"(tag typical = {b_date}, tag steep = {a_date})."
            )
        else:
            head = "**Answer: NOT confirmed by this script — a row or a fixture does not match the study's B1 entry of its date (`tables/B_check_a.csv`).**"
            match = "The comparison of T, P_D, EQV, rho_cop (rows) and of the date, T and index at-the-money vol (fixtures) with the study's B1 entries shows a gap."
        sentences = [
            f"{head} M12's {a_date} on the old defaults, {m_a:.5f} ± {num(o_a, 'ratio_se'):.5f}, sits next to the reference's {b_date}, {r_b.value:.5f} ± {r_b.se:.5f}, "
            f"but a swap of labels would also need M12's {b_date}, {m_b:.5f} ± {num(o_b, 'ratio_se'):.5f}, to match the reference's {a_date}, {r_a.value:.5f} ± {r_a.se:.5f}: "
            f"they differ by {abs(m_b - r_a.value):.5f}.",
            match,
        ]
        if a_date in rebuild and b_date in rebuild:
            sentences.append(
                f"The reference's own worlds rebuilt on the library reproduce each tag's value — {b_date}: {rebuild[b_date]['ratio']:.5f} ± {rebuild[b_date]['se']:.5f} "
                f"(z {rebuild[b_date]['z']:+.2f}), {a_date}: {rebuild[a_date]['ratio']:.5f} ± {rebuild[a_date]['se']:.5f} (z {rebuild[a_date]['z']:+.2f}) — "
                f"so on a given date the two columns differ by the model (the reference's world: two-parameter local vols and λ, no carry): M12 − reference is "
                f"{m_b - r_b.value:+.5f} on {b_date} and {m_a - r_a.value:+.5f} on {a_date}."
            )
        else:
            sentences.append(
                f"Pending: the rebuild of the reference's worlds on the library ({rebuild_log})."
            )
        n_a, n_b = (
            core_items(inputs[a_date])["ratio.lc_over_cc"],
            core_items(inputs[b_date])["ratio.lc_over_cc"],
        )
        if ok(n_a.value) and ok(n_b.value):
            sentences.append(
                f"With the decisions on (section B's rows) LC/CC is {n_a.cell()} on {a_date} and {n_b.cell()} on {b_date}."
            )
        part.text(" ".join(sentences), "")
    else:
        part.text(
            f"Pending: the old-default rows `{rel(OLD_ROWS_DIR)}/<date>.json` or the fixtures are missing.",
            "",
        )
    part.text(
        f"Old defaults: `{rel(OLD_ROWS_DIR)}/<date>.json` (the production pass stopped at 07:45, no repair, no fallback); reference: `{rel(FIXTURES)}/<tag>.json` "
        f"({REF_BUDGET}); rebuild: an independent verification run with `tests/_lcm_reference.py` (log `{rebuild_log}`); numbers in `tables/B_check_a.csv`.",
        "",
    )


def build_b(inputs: Mapping[str, Inputs], base: Path, rows_dir: Path, rebuild_log: Path) -> Part:
    part = Part("B")
    dates = [*OTHERS, TODAY]
    xs = [inputs[d] for d in dates]
    cores = {d: core_items(inputs[d]) for d in dates}
    refs = {d: ref_items(inputs[d]) for d in dates}
    # no section heading here: scripts/pm_assemble.py writes "## B. The three reference dates, ..."
    if rows_dir.resolve() != ROWS_DIR.resolve():
        part.text(
            f"**STAND-IN ROWS ({rel(rows_dir)}): a test of the code path, not the package's numbers.**",
            "",
        )
    part.text(
        f"Cells are value ± standard error; `pending: <file>` marks a number whose input has not arrived. {TODAY} (section A) is repeated for reference.",
        "",
    )
    flags_table(part, xs, "B_flags", base)
    head = ["quantity", *[f"{d}{' (today)' if d == TODAY else ''}" for d in dates]]

    def rows_for(keys: Sequence[tuple[str, str]], table: str) -> list[list[str]]:
        body = []
        for key, label in keys:
            src = refs if key.startswith("ref.") else cores
            for d in dates:
                part.add(table, f"B.{d}.", d, [src[d][key]])
            body.append([label or src[dates[0]][key].label, *[src[d][key].cell() for d in dates]])
        return body

    part.text("### B1. The core set", "")
    core_keys = (
        ("ED.lc", "E_LC[D]"), ("ED.cc", "E_CC[D]"), ("ED.copula", "P_D (copula)"), ("ED.model_s", "P_D_S (model S)"),
        ("ratio.lc_over_cc", "LC/CC"), ("ref.lc_over_cc", "parametric reference LC/CC"), ("ratio.lc_over_copula", "LC/copula"),
        ("ratio.cc_over_copula", "CC/copula"), ("ratio.s_over_copula", "model S/copula"), ("ratio.listed_fwd", "listed-variance forward √(EQV/EV)"),
        ("ratio.wing_over_cc", "ED_wing/CC"), ("ratio.wing_over_copula", "ED_wing/copula"), ("ratio.eqv_over_cc", "ED_eqv/CC"),
        ("ratio.eqv_over_copula", "ED_eqv/copula"),
        ("kappa.lc", "κ, LC"), ("kappa.cc", "κ, CC"), ("kappa.copula", "κ, copula"), ("kappa.model_s", "κ, model S"),
        ("EV_over_EQV.lc", "E[V]/EQV, LC"), ("EV_over_EQV.cc", "E[V]/EQV, CC"), ("EV_over_EQV.copula", "E[V]/EQV, copula"),
        ("EV_over_EQV.model_s", "E[V]/EQV, model S"), ("EV_split.single_over_EQV", "LC split: single-name part / EQV"),
        ("EV_split.basket_over_EQV", "LC split: basket part / EQV"), ("Rbar2_over_listed.lc", "E_LC[R̄²]/M_B^listed"),
        ("clip.inner_low", "clipped mass inside ±2.5 sd, at λ = 0 (%)"), ("clip.inner_high", "clipped mass inside ±2.5 sd, at the cap (%)"),
        ("clip.inner_total", "clipped mass inside ±2.5 sd, both sides (%)"), ("clip.cloud_low", "clipped mass, whole cloud, at λ = 0 (%)"),
        ("clip.cloud_high", "clipped mass, whole cloud, at the cap (%)"),
        ("idx_err.atm", "index error at the money (vol points)"), ("idx_err.90", "index error at the 90 % strike (vol points)"),
        ("idx_err.m25", "index error at −2.5 sd (vol points)"),
        ("delta.lc_ss", "Δ sticky strike, LC (% of E[D] per +1 %)"), ("ref.delta_lc", "parametric reference Δ sticky strike, LC"),
        ("delta.cc_ss", "Δ sticky strike, CC"), ("ref.delta_cc", "parametric reference Δ sticky strike, CC"),
        ("delta.homogeneity", "decomposition: homogeneity (exact)"), ("delta.skew_channel", "decomposition: skew channel Δ^CC_ss − 1"),
        ("delta.correlation_channel", "decomposition: correlation channel Δ^LC_ss − Δ^CC_ss"),
        ("ref.ED_lc", "parametric reference E_LC[D]"), ("ref.ED_cc", "parametric reference E_CC[D]"),
    )  # fmt: skip
    part.table(head, rows_for(core_keys, "B_core"))
    x0 = xs[0]
    part.text(
        "Definitions as in section A: LC/CC paired (the row's ratio_se); ratios to the copula by the delta method on the numerator's error and P_D_se, independent; "
        f"listed-variance forward = {D_LISTED}; {D_WING}; {D_EQV}; {D_KAPPA}; E_LC[V] − EQV = single-name part − basket part; "
        f"clipped mass = {D_CLIP}; deltas = {D_DELTA}, sticky strike. "
        f"LC and CC: {x0.row_budget}, commit {', '.join(sorted({x.row_commit for x in xs if x.row_commit})) or 'pending'}, `{rel(rows_dir)}/<date>.json`. "
        f"Copula: `{rel(ENTRIES)}`, basket B1; model S: `{rel(MODEL_S)}` (no standard errors; NOT converged on "
        f"{', '.join(x.date for x in xs if x.s_mark) or 'none of these dates'}). "
        f"Parametric reference: `{rel(FIXTURES)}/<tag>.json` ({REF_BUDGET}; a separate implementation with a two-parameter λ and no carry, in its own world: "
        f"tags {', '.join(f'{x.ref_tag} = {x.date}' for x in xs)}).",
        "",
    )
    part.text("### B2. Calls", "")
    call_keys: list[tuple[str, str]] = []
    for m in pc.MULT_TAGS:
        call_keys += [(f"call.K_{m}.strike", f"K_{m}"), (f"call.K_{m}.lc", f"call K_{m}, LC"), (f"call.K_{m}.cc", f"call K_{m}, CC"),
                      (f"call.K_{m}.copula", f"call K_{m}, copula"), (f"call.K_{m}.model_s", f"call K_{m}, model S"),
                      (f"call.K_{m}.lc_over_copula", f"K_{m}: LC/copula"), (f"call.K_{m}.cc_over_copula", f"K_{m}: CC/copula"),
                      (f"call.K_{m}.s_over_copula", f"K_{m}: model S/copula"), (f"call.K_{m}.lc_over_cc", f"K_{m}: LC/CC (paired)")]  # fmt: skip
    call_keys += [
        (
            f"ref.call_{t}.lc_over_cc",
            f"parametric reference LC/CC at {t[0]}.{t[1:]} × its own E_CC[D]",
        )
        for t in ("075", "100", "125", "150")
    ]
    part.table(head, rows_for(call_keys, "B_calls"))
    part.text(
        "Call = E[(D − K)⁺] in units of notional, K_m = m % of the copula's P_D; ratios to the copula by the delta method on the two errors, independent "
        "(n/a when the copula's call is zero; model S has no errors). The parametric reference's call ratios are at multiples of its own E_CC[D], not at the study's strikes. "
        "Budgets, commits and sources as under B1.",
        "",
    )
    check_a(part, inputs, rebuild_log)
    checks = consistency(xs)
    if checks:
        part.text("### B. Consistency of the inputs", "", *checks, "")
    return part


# --------------------------------------------------------------------------- main


def main(argv: Sequence[str] | None = None) -> int:
    global RISK_FILE, CDV_FILE  # the sources named in the tables are the files read
    default_risk, default_cdv = RISK_FILE, CDV_FILE
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument(
        "--rows-dir", default=str(ROWS_DIR), help="the production rows, one JSON per date"
    )
    ap.add_argument(
        "--risk", default=str(default_risk), help="today's risk (parts/A_risk_today_raw.json)"
    )
    ap.add_argument("--cdv", default=str(default_cdv), help="the cross-dependent scan of today")
    ap.add_argument(
        "--check-a-log",
        default=str(CHECK_A_LOG),
        help="the log of the rebuild of the reference (check a)",
    )
    ap.add_argument(
        "--out", default=str(pc.PM), help="the package's folder (another one for tests)"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
    )
    rows_dir, base = Path(args.rows_dir), Path(args.out)
    stand_in = [
        name
        for name, given, default in (("--rows-dir", rows_dir, ROWS_DIR), ("--risk", Path(args.risk), default_risk), ("--cdv", Path(args.cdv), default_cdv))
        if given.resolve() != default.resolve()
    ]  # fmt: skip
    if stand_in and base.resolve() == pc.PM.resolve():
        raise SystemExit(
            f"{', '.join(stand_in)} not the package's input: give --out a test folder (stand-in inputs never go into the package)"
        )
    RISK_FILE, CDV_FILE = Path(args.risk), Path(args.cdv)
    inputs = load_inputs(rows_dir)
    risk, risk_pending = load_json(RISK_FILE)
    if risk is not None and (str(risk.get("date")) != TODAY or str(risk.get("tenor")) != TENOR):
        risk, risk_pending = (
            None,
            f"{rel(RISK_FILE)} (it is for {risk.get('date')} {risk.get('tenor')})",
        )
    cdv, cdv_pending = load_json(CDV_FILE)
    if cdv is not None and (str(cdv.get("date")) != TODAY or str(cdv.get("tenor")) != TENOR):
        cdv, cdv_pending = None, f"{rel(CDV_FILE)} (it is for {cdv.get('date')} {cdv.get('tenor')})"
    log.info(
        "risk: %s; cdv: %s",
        "found" if risk else f"pending ({risk_pending})",
        "found" if cdv else f"pending ({cdv_pending})",
    )
    part_a = build_a(inputs, risk, risk_pending, cdv, cdv_pending, base, rows_dir)
    part_a.save("A_today", base)
    part_b = build_b(inputs, base, rows_dir, Path(args.check_a_log))
    part_b.save("B_reference", base)
    for name, part in (("A_today", part_a), ("B_reference", part_b)):
        waits = sorted({p.split(" <- ")[1] for p in part.pending})
        log.info(
            "%s: %d records, %d pending; waits for: %s",
            name,
            len(part.records),
            len(part.pending),
            "; ".join(waits) or "nothing",
        )
    rows_in = [d for d in pc.REFERENCE_DATES if inputs[d].row is not None]
    waits = sorted(
        {
            Path(p.split(" <- ")[1].split(" (")[0]).name
            for part in (part_a, part_b)
            for p in part.pending
        }
    )
    pc.status(
        f"A/B (scripts/pm_ab.py): parts/A_today and parts/B_reference written, tables A_*.csv and B_*.csv; production rows read: {', '.join(rows_in) or 'none'}; "
        f"risk {'read' if risk else 'pending'}; CDV {'read' if cdv else 'pending'}; pending cells: A {len(part_a.pending)}, B {len(part_b.pending)}"
        f"{' (waits for ' + '; '.join(waits) + ')' if waits else ''}.",
        base,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
