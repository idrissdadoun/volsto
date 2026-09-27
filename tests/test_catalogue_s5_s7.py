"""Catalogue studies S5–S7 (SPEC §10.2, owner's M10 Part 2; ``volsto/studies/catalogue/
s5_marking.py``, ``s6_shadow_rotation.py``, ``s7_hedging.py``).

The owner's test for the catalogue: **every catalogue study runs on a small grid in CI-fast mode
and produces tables and figures.**  Nothing here calibrates: S6 and S7 read the synthetic M7 /
M8b outputs the sanctioned fixture ``tests/conftest.py::toy_build`` writes
(``tests/_synthetic_store.py::make_synthetic_outputs``); S5 reads the toy marking store of the
second sanctioned fixture ``tests/conftest.py::toy_marking_build`` (``configs/grids/
toy_marking.yaml``) and fits its binding map inline (a parameter fit, not a leverage calibration
— the cache directory is asserted unchanged).

What is asserted:

* S6 / S7 fast: exit 0; every declared table (``tables/*.tex``) and figure (``.pdf`` + ``.png``)
  written; ``study.md`` opens with the one-sentence question; ``manifest.json`` says
  ``recalibrated: false`` and lists the artefacts read; the LaTeX document compiles (when
  Tectonic is installed); ``render`` rebuilds the tables and ``study.md`` byte-identically from
  ``results.parquet``; ``rerun`` moves nothing;
* S6: the table's state (17 of 18 configured rows, the absent row listed); the ratios equal to
  ``table_C``'s own; every row classed at 2 stderr (:func:`classify`, its boundaries pinned
  case by case) and the claim's per-rota counts equal to those classes, the smallest / largest
  ratio carried with its own stderr; the claim section excludes the contaminated
  ``sabr_linked`` column, covers +1 and +2 / +3 separately, says it is drawn on a table whose
  rows have lost their result files, and never speaks of a "cost" or a "lower bound"; its two
  guards tripped by a synthetic table (a "within" ratio below 1 blocks the magnitude sentence; a
  simulated P&L with |z| <= 2 is listed as sign-undecided, not as a gain or a loss); the refits,
  guarded fallbacks and capped targets per column equal the table's sums and the reading quotes
  them; the simulated policy ordering at every rota equals sticky / sabr of the table's P&Ls, its
  sides (:func:`ordering_side`, pinned case by case) and the disagreements with the static side
  are consistent and listed; a table
  written with the **pre-2026-09-16 stem names** (``recal_se`` / ``total_se`` / ``static_se``,
  no ratio / nonlinearity stderr, no ``refits_fallback`` / ``refits_capped``) giving the same
  numbers, the refit flags read as -1 and the fallback / cap counts reported as unrecorded
  (no reading printed);
* S7: the reserve, the gated world with the verdict's reason, the owner's study-D headline
  quoted verbatim and checked, a table-A twin written under the old ``mean_se`` stem paired, the
  spread of the implied MV deltas carried with its conservative stderr (the two extreme rows in
  quadrature); a missing table → exit 2 with ``scripts/m8b.py --study D --resume`` and no output
  directory;
* S5 fast (``toy_marking_build``): the binding map reproduces the stored fits, realised SSR
  with stderr, the forward window beyond the 1y horizon flagged, the placeholder warning, the
  recorded M7 fits labelled, |L-1| exact with its note and drawn without error bars, the cache
  and the store unchanged; with the full config's two 1y-long windows every forward-skew
  column (the spot skew included) is keyed by its window, so the build does not collide;
* S5 Greeks (the synthetic store's risk rows, real ``skew_tent[..]`` naming): the stored
  fractions of notional x 100 in ``% notional <store unit>`` with their stderr; the forward
  90/110 skew's stderr the sum of the two strikes' errors (0.10 vol pts there, not the
  quadrature's 0.0707); an unknown Greek name → exit 1; a product without stored risk refused;
* S5 requirements (no build needed): on an empty store the fast and the full configs exit 2 and
  print one ``volsto-precompute --grid <the configured grid> ... --only <ids> --resume`` line
  (4 toy ids / the 36 default-grid marking ids); on a store whose run record names **another**
  grid (the repository's ``outputs/store`` is built from ``placeholder_cached.yaml``) the line
  still names the configured grid and a note names the recorded one;
* every S5–S7 config loads strictly against its module; the helpers never record a Monte Carlo
  value without its stderr;
* derived numbers (review of M10): every z-score of S5–S7 carries stderr 1 (not exact); S6's
  ratios of runs sharing the world seed carry :func:`volsto.studies.m8b.ratio_stderr_bound`
  (pinned against the exact first-order formula over every correlation, including the
  opposite-sign case where the independent figure understates), its simulated / static ratio is
  labelled "not a bound", no study text calls an error "conservative"; S5's cost of a mark
  carries the shared ``QUADRATURE_NOTE``; S7's ranks are exact with their note and each carries
  its "decided at 2 se" flag (recomputed from the table's values: the two ``value +/- 2 se``
  intervals of consecutive ranks; the headline set against every farther ranked regime, with a
  synthetic table where that differs from the boundary pair), its spread carries the sum of the two extreme errors and is
  shown over its noise floor ``d_k`` x the mean error (a Monte Carlo ratio); S7's
  gate reads the repaired eSSVI history's verdict and names it, falls back to the M8b file (and
  says so) only when that file is absent, and prints the repaired run's command when neither
  exists; S5 and the precompute share one marking-fit helper;
* exact rows: every exact row of every S5–S7 run is declared in its module's ``EXACT_KINDS``
  (``_common.unclassified_exact_rows``), each rank with its flag column;
* LaTeX: asserted when Tectonic ran; a skipped check is reported as a pytest skip at the end of
  the test instead of passing silently.

Wall clocks are printed, never asserted.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from _synthetic_store import SYNTHETIC_C_MISSING, SYNTHETIC_VERDICT

from volsto.studies import latex, m8b, runner
from volsto.studies.catalogue import s5_marking, s6_shadow_rotation, s7_hedging
from volsto.studies.catalogue._common import (
    QUADRATURE_NOTE,
    parse_exact_kinds,
    unclassified_exact_rows,
)
from volsto.studies.results import Column, Results, ResultsBuilder, TableSpec
from volsto.viewers.store import ResultsStore

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "configs" / "studies" / "catalogue"
CONFIGS = {
    "s5": ("s5.yaml", "s5_fast.yaml"),
    "s6": ("s6.yaml", "s6_fast.yaml"),
    "s7": ("s7.yaml", "s7_fast.yaml"),
}
MODULES = {"s5": s5_marking, "s6": s6_shadow_rotation, "s7": s7_hedging}
S6_TABLES = {
    "table_c_state",
    "table_c_missing",
    "m7_greek",
    "static_greek",
    "m7_vs_static",
    "pnl_autocall_3y",
    "pnl_cliquet_1y",
    "agreement_autocall_3y",
    "agreement_cliquet_1y",
    "claim_columns",
    "claim",
    "ordering",
    "ordering_sim",
    "ordering_sides",
}
S6_FIGURES = {"pnl_vs_static", "ratio", "nonlinearity", "m7_greek"}
S7_TABLES = {
    "sources",
    "a_ranking",
    "b_reserve",
    "b_matrix",
    "b_skipped",
    "gate",
    "gate_pillars",
    "d_ranking",
    "d_common",
    "d_summary",
}
S7_FIGURES = {"a_std", "b_reserve", "d_distance"}
#: S5's per-surface tables on the toy marking grid (one surface: the placeholder).
S5_TABLES = {
    "setup",
    "binding_map_placeholder",
    "marks_placeholder",
    "forward_skew_placeholder",
    "prices_placeholder",
    "mark_cost_placeholder",
    "m7_fits",
}
S5_FIGURES = {
    "binding_map",
    "realised_ssr",
    "leverage_deviation",
    "fwd_spot_ratio",
    "mark_cost_placeholder",
}
#: The toy marking grid of the S5 requirement tests (written by the test, never computed).
TOY_MARKING_GRID = """\
name: toy_marking_test
reference_spec: configs/studies/lsv_reference_2f.yaml
surfaces:
  - {name: placeholder, kind: placeholder, one_factor: false, two_factor: false, marking: true}
particle: {n_particles: 20000, horizon: 1.0}
marking: {ssr_target: [1.0, 1.5], skew_eps: [0.05, 0.10], fit: m7}
include_degenerate: false
products: {m4: true, m6: false, conditional: false, cliquet_maturities: [1.0]}
pricing: {n_paths: 4000, seed: 2024}
risk: {tier: none, products: ["autocall 3y", "cliquet 1y"], light_pillars: [0.25, 1.0, 3.0],
       fwd_var_buckets: 20}
"""


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def _run(
    config: str,
    out: Path,
    capsys: pytest.CaptureFixture[str],
    *flags: str,
) -> tuple[int, str, float]:
    t0 = time.perf_counter()
    code = runner.main(["run", str(CATALOGUE / config), "--out", str(out), *flags])
    wall = time.perf_counter() - t0
    captured = capsys.readouterr()
    print(f"\n{config}: exit {code} in {wall:.2f} s")
    return code, captured.out + captured.err, wall


def _files(root: Path) -> dict[str, tuple[int, int]]:
    """``relpath -> (size, mtime_ns)`` of every file under ``root``."""
    if not root.exists():
        return {}
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _check_outputs(
    out: Path, tables: set[str], figures: set[str], question: str
) -> tuple[Results, dict[str, Any]]:
    """The run's files, the question first, the manifest's provenance, the LaTeX check."""
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["recalibrated"] is False
    assert set(manifest["tables"]) == tables, sorted(manifest["tables"])
    assert set(manifest["figures"]) == figures, sorted(manifest["figures"])
    for t in tables:
        assert (out / "tables" / f"{t}.tex").is_file(), t
    for f in figures:
        assert (out / "figures" / f"{f}.pdf").is_file(), f
        assert (out / "figures" / f"{f}.png").is_file(), f
    md = (out / "study.md").read_text()
    lines = md.splitlines()
    assert lines[0].startswith("# ")
    # the question is the first statement of the document (a runner banner may precede it)
    first = next(i for i, ln in enumerate(lines) if ln.startswith("**Question.**"))
    assert first <= 6 and lines[first] == f"**Question.** {question}"
    assert question.count("?") == 1 and question.endswith("?"), question
    assert "| recalibrated | no |" in md
    assert (out / "study.tex").is_file()
    latex_check = manifest["latex"]
    print(f"latex: {latex_check['status']} {latex_check.get('reason', '')[:200]}")
    if latex_check["status"] != "skipped":  # a skip is reported by _latex_ok_or_skip, last
        assert latex_check["status"] == "ok", latex_check
        assert (out / "study.pdf").is_file()
    results = Results.read(out / "results.parquet")
    frame = results.frame
    mc = frame[~frame["exact"] & frame["value"].notna()]
    assert mc["stderr"].notna().all() and (mc["stderr"] >= 0).all()
    _check_exact_kinds(results, question)
    return results, manifest


def _check_exact_kinds(results: Results, question: str) -> None:
    """Every exact row is declared in the study's ``EXACT_KINDS`` (ranks with their flag)."""
    module = next(m for m in MODULES.values() if question == m.QUESTION)
    bad = unclassified_exact_rows(results, module.EXACT_KINDS)
    listed = bad.drop_duplicates(["table", "column"])
    assert bad.empty, "undeclared exact rows: " + "; ".join(
        f"{t}.{c}" for t, c in zip(listed["table"], listed["column"], strict=True)
    )


def _latex_ok_or_skip(manifest: dict[str, Any]) -> None:
    """The last step of a test that ran the LaTeX check: without Tectonic the check was skipped
    by the runner (exit 0) — say so as a pytest skip instead of passing silently."""
    status = manifest["latex"]["status"]
    if status == "skipped":
        pytest.skip(runner.TECTONIC_MISSING)
    assert status == "ok", manifest["latex"]
    assert runner.find_tectonic() is not None


def _check_render_and_rerun(out: Path, tmp: Path) -> None:
    """``render`` rebuilds the tables and ``study.md`` byte-identically; ``rerun`` moves nothing."""
    before = {p.name: p.read_bytes() for p in (out / "tables").glob("*.tex")}
    md = (out / "study.md").read_bytes()
    shutil.rmtree(out / "tables")
    shutil.rmtree(out / "figures")
    runner.render_study(out)
    after = {p.name: p.read_bytes() for p in (out / "tables").glob("*.tex")}
    assert after == before
    assert (out / "study.md").read_bytes() == md
    outcome = runner.rerun_study(out, out_dir=tmp)
    moved = outcome.changed
    assert outcome.exit_code == runner.EXIT_OK, moved.to_string()


def _copy_outputs(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def _roots(toy: Any) -> list[str]:
    return ["--store", str(toy.store_root), "--cache", str(toy.cache_root)]


# --------------------------------------------------------------------------------------------
# configs and helpers
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("study", sorted(CONFIGS))
def test_configs_load_strictly(study: str) -> None:
    """Every S5–S7 config loads against its module: the declared params, the module's own
    checks, one-line questions."""
    module = runner.load_runner_module(MODULES[study].__name__)
    for name in CONFIGS[study]:
        cfg = runner.load_study_config(CATALOGUE / name)
        assert cfg.runner == module.__name__
        runner.validate_params(module, cfg.params)
        assert runner.study_question(cfg, module) == module.QUESTION
    assert runner.load_study_config(CATALOGUE / CONFIGS[study][1]).mode == "fast"
    assert runner.load_study_config(CATALOGUE / CONFIGS[study][0]).mode == "full"


@pytest.mark.parametrize("module", [s5_marking, s6_shadow_rotation, s7_hedging])
def test_helpers_never_record_a_value_without_its_stderr(module: Any) -> None:
    b = ResultsBuilder()
    kw: dict[str, Any] = {"unit": "1", "source": "computed", "axes": {}}
    module.add_mc(b, "t", "ok", "x", 1.5, 0.1, **kw)
    module.add_mc(b, "t", "no_se", "x", 1.5, math.nan, **kw)
    module.add_mc(b, "t", "inf", "x", math.inf, 0.1, **kw)
    res = b.build()
    assert res.value("t", "ok", "x") == (1.5, 0.1)
    for row in ("no_se", "inf"):
        v, se = res.value("t", row, "x")
        assert math.isnan(v) and math.isnan(se)
        assert res.record("t", row, "x")["note"]


@pytest.mark.parametrize("module", [s5_marking, s6_shadow_rotation, s7_hedging])
def test_long_tables_are_split_below_the_longtable_threshold(module: Any) -> None:
    """A table with more rows than ``latex.LONGTABLE_MIN_ROWS`` would be an unscaled longtable
    (a wide one overflows the page): the modules split it into parts, in row order."""
    n = latex.LONGTABLE_MIN_ROWS
    b = ResultsBuilder()
    for i in range(n + 3):
        b.add_exact("t", f"r{i:02d}", "x", float(i), unit="", source="computed")
        b.add_exact("u", f"r{i:02d}", "x", float(i), unit="", source="computed")
    res = b.build()
    specs = [
        TableSpec("t", "long", "t", (Column("x", "x"),)),
        TableSpec("u", "short", "u", (Column("x", "x"),), rows=("r00", "r01")),
    ]
    out = module.fit_specs(specs, res)
    assert [s.name for s in out] == ["t_1", "t_2", "u"]
    assert out[0].rows is not None and len(out[0].rows) == n
    assert out[1].rows == tuple(f"r{i:02d}" for i in range(n, n + 3))
    assert out[0].caption == "long (part 1 of 2)"


def test_s5_units_of_the_store() -> None:
    """Vols to vol points; the M6 cells (fractions under the '% notional' label) x100 with a
    note; the M4 % notional columns unchanged."""
    assert s5_marking.display("atm_vol", "vol")[:2] == (100.0, "vol pts")
    scale, unit, note = s5_marking.display("autocall 3y:price", "% notional")
    assert (scale, unit) == (100.0, "% notional") and "fraction" in note
    assert s5_marking.display("cliquet_1y", "% notional") == (1.0, "% notional", "")
    assert s5_marking.display("kovar_110_p_ko", "probability")[1] == "1"


# --------------------------------------------------------------------------------------------
# S6
# --------------------------------------------------------------------------------------------


def test_s6_fast_on_the_toy_outputs(
    toy_build: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    toy = toy_build.require()
    out = tmp_path / "s6"
    code, text, _ = _run(
        "s6_fast.yaml", out, capsys, "--outputs", str(toy.outputs_root), *_roots(toy)
    )
    assert code == 0, text
    res, manifest = _check_outputs(out, S6_TABLES, S6_FIGURES, s6_shadow_rotation.QUESTION)
    arts = {a["path"] for a in manifest["artefacts"]}
    assert {"m8b/m8b_table_C.csv", "m7/p1_marking_shadow_rotation.csv"} <= arts
    assert "m8b/C/static_autocall_3y__sabr_linked.json" in arts
    assert manifest["records"]["table_c"]["missing"] == [list(SYNTHETIC_C_MISSING)]
    # the table's state: 17 of 18 rows, the absent one listed
    state = res.pivot("table_c_state").iloc[0]
    assert (state["rows"], state["rows_expected"], state["rows_missing"]) == (17, 18, 1)
    product, rota, policy = SYNTHETIC_C_MISSING
    assert res.rows("table_c_missing") == [f"{product} | {policy} | +{rota:g}"]
    # the restricted claim: which column is clean, how many of its rows lost their result file
    assert res.value("claim_columns", "sabr_linked", "clean")[0] == 0.0
    assert res.value("claim_columns", "sticky_breakeven", "clean")[0] == 1.0
    # 2 products x 3 rotas, none with a task result under m8b/C in the synthetic outputs
    assert res.value("claim_columns", "sticky_breakeven", "rows_without_result_file")[0] == 6.0
    md = (out / "study.md").read_text()
    claim = md[md.index("## The restricted claim") : md.index("## First-order agreement")]
    assert "The **sabr_linked** column is reported but **excluded**" in claim
    assert "**sticky_breakeven, +1 rota**" in claim and "at +2:" in claim and "at +3:" in claim
    assert "This is the claim of the table as read" in claim
    assert "6 of the column's 6 rows have no task result" in claim
    assert "lower bound" not in md
    assert "recalibration cost" not in md and "simulated cost" not in md
    assert "1 configured row(s) are missing" in md and "modified 20" in md
    assert "desk P&L per +1 rota for a SHORT position" in md  # the declared convention
    # the ratios equal table_C's own (same definitions, recomputed from the table's values); the
    # class of every row follows its ratio and stderr, and the claim counts are those classes
    table = pd.read_csv(toy.outputs_root / "m8b" / "m8b_table_C.csv")
    tol = res.value("setup", "study C", "first_order_tolerance")[0]
    classes: dict[tuple[str, float], list[str]] = {}
    for r in table.to_dict("records"):
        row = f"{r['product']} | {r['recalibration']} | +{r['rota']:g}"
        v, se = res.value("agreement", row, "ratio")
        assert v == pytest.approx(r["ratio"], rel=1e-12), row
        assert se == pytest.approx(r["ratio_se"], rel=1e-12), row
        cls = s6_shadow_rotation.classify(v, se, tol)
        rec = res.record("agreement", row, "class_2se")
        assert (rec["value"], rec["note"]) == (s6_shadow_rotation.CLASS_CODES[cls], cls), row
        if r["recalibration"] != "none":
            classes.setdefault((r["recalibration"], r["rota"]), []).append(cls)
        if r["rota"] != 1.0:
            v, se = res.value("agreement", row, "nonlinearity")
            assert v == pytest.approx(r["nonlinearity"], rel=1e-12, abs=1e-14), row
    for (policy, rota), got in classes.items():
        row = f"{policy} | +{rota:g}"
        assert res.value("claim", row, "pairs")[0] == len(got), row
        for cls in s6_shadow_rotation.CLASS_CODES:
            assert res.value("claim", row, cls)[0] == got.count(cls), (row, cls)
        lo, lo_se = res.value("claim", row, "min_ratio")  # a Monte Carlo value with its stderr
        assert math.isfinite(lo) and lo_se > 0 and not res.is_exact("claim", row, "min_ratio")
    # the synthetic clean column is larger than the greek at +1 on both products: the magnitude
    # sentence is printed, every decided product has |simulated| >= |static| at 2 stderr
    assert res.value("claim", "sticky_breakeven | +1", "above")[0] == 2.0
    assert res.value("claim", "sticky_breakeven | +1", "decided_static_not_larger")[0] == 2.0
    assert "on every product the comparison decides, |simulated| >= |static|" in claim
    # the refits per column and the reading, computed from the table's counts
    recal = table[table["recalibration"] != "none"]
    for key, g in recal.groupby("recalibration"):
        col_policy = str(key)
        n, fb, cap = (int(g[c].sum()) for c in ("n_refits", "refits_fallback", "refits_capped"))
        assert res.value("claim_columns", col_policy, "refits")[0] == n
        assert res.value("claim_columns", col_policy, "refits_fallback")[0] == fb
        assert res.value("claim_columns", col_policy, "refits_capped")[0] == cap
        assert res.value("claim_columns", col_policy, "rows_counts_unrecorded")[0] == 0
        assert f"- **{col_policy}**: {n} refits over its {len(g)} rows; {fb} of the {n} " in claim
        assert f"and {cap} ({cap / n:.0%}) had their correlation target capped" in claim
        if col_policy == "sabr_linked":
            assert 0 < fb < n and cap > 0  # the synthetic counts are not trivial
            assert (
                f"so on {fb} of sabr_linked's {n} recorded refits ({fb / n:.0%}) the two policies "
                "share their correlation target" in claim
            )
    # the simulated policy ordering at every rota beside the static one
    sides = res.pivot("ordering_sides")
    sim_pivot = res.pivot("ordering_sim")
    disagree = []
    for product in res.rows("ordering_sim"):
        static_side = sides.loc[product, "static"]
        for rota in (1.0, 2.0, 3.0):
            a = table[
                (table["product"] == product)
                & (table["rota"] == rota)
                & (table["recalibration"] == "sabr_linked")
            ]
            c = table[
                (table["product"] == product)
                & (table["rota"] == rota)
                & (table["recalibration"] == "sticky_breakeven")
            ]
            label = f"+{rota:g}"
            if a.empty or c.empty:  # the synthetic table's absent row: no ratio at that rota
                assert pd.isna(sim_pivot.at[product, f"ratio@{label}"])
                continue
            va, vc = a.iloc[0], c.iloc[0]
            v, se = res.value("ordering_sim", product, f"ratio@{label}")
            assert v == pytest.approx(vc["recal_pnl_desk"] / va["recal_pnl_desk"], rel=1e-12)
            assert se == pytest.approx(
                m8b.ratio_stderr_bound(
                    vc["recal_pnl_desk"],
                    vc["recal_pnl_desk_se"],
                    va["recal_pnl_desk"],
                    va["recal_pnl_desk_se"],
                ),
                rel=1e-12,
            )
            side = s6_shadow_rotation.ordering_side(v, se)
            assert sides.loc[product, f"sim@{label}"] == side
            flag = static_side != 0 and side != 0 and side != static_side
            assert sides.loc[product, f"disagree@{label}"] == float(flag)
            if flag:
                disagree.append(f"{product} at {label}")
    assert disagree  # the synthetic contaminated column is of the opposite sign
    ordering = md[md.index("## Policy ordering") :]
    assert f"they differ on {len(disagree)} ({', '.join(disagree)})" in ordering
    # z-scores carry their sampling sd 1; ratios of same-seed runs carry the bound
    for rec in _records_of(res, "agreement", "z"):
        if math.isfinite(rec["value"]):
            assert not rec["exact"] and rec["stderr"] == 1.0 and "sampling sd" in rec["note"]
    for rec in _records_of(res, "agreement", "nonlinearity"):
        if math.isfinite(rec["value"]):
            assert rec["note"] == s6_shadow_rotation.BOUND_NOTE
    for rec in _records_of(res, "agreement", "ratio"):
        if math.isfinite(rec["value"]):
            assert rec["note"] == s6_shadow_rotation.AGREEMENT_SE_NOTE
    assert "conservative" not in md
    _check_render_and_rerun(out, tmp_path / "s6_rerun")
    _latex_ok_or_skip(manifest)


@pytest.mark.parametrize(
    ("ratio", "stderr", "expected"),
    [
        (1.19, 0.42, "undecided"),  # the Phoenix at +1: inside the band on the point only
        (1.20, 0.04, "within"),
        (2.56, 0.26, "above"),
        (0.38, 0.06, "below"),
        (-7.8, 1.6, "below"),  # the opposite sign
        (1.29, 0.01, "undecided"),  # the interval reaches past 1.3
        (math.nan, 0.1, "undecided"),
        (1.0, math.nan, "undecided"),
    ],
)
def test_s6_classifies_at_two_stderr(ratio: float, stderr: float, expected: str) -> None:
    assert s6_shadow_rotation.classify(ratio, stderr, 0.30) == expected


@pytest.mark.parametrize(
    ("ratio", "stderr", "expected"),
    [
        (1.28, 0.05, 1.0),  # the KO var's static ordering: sticky larger
        (0.72, 0.02, -1.0),  # its simulated +1 ordering: sticky smaller, same sign
        (-1.45, 0.12, -2.0),  # opposite signs
        (1.37, 0.21, 0.0),  # the autocall's static ordering: undecided at 2 se
        (0.9, 1.9, 0.0),
        (math.nan, 0.1, 0.0),
    ],
)
def test_s6_ordering_sides(ratio: float, stderr: float, expected: float) -> None:
    assert s6_shadow_rotation.ordering_side(ratio, stderr) == expected


def _records_of(res: Results, table: str, column: str) -> list[dict[str, Any]]:
    long = res.long(table, column)
    return [{str(k): v for k, v in r.items()} for r in long.to_dict("records")]


@pytest.mark.parametrize(
    ("num", "num_se", "den", "den_se"), [(1.0, 0.1, -2.0, 0.2), (1.0, 0.1, 2.0, 0.2)]
)
def test_ratio_stderr_bound(num: float, num_se: float, den: float, den_se: float) -> None:
    """The first-order stderr of num/den for correlation rho is
    |r| sqrt(a² + b² − 2 rho a b s) with a = se_n/n, b = se_d/d (absolute), s = sign(n d):
    the bound is its maximum over rho ∈ [−1, 1]; the independent figure understates it for
    opposite signs and a positive correlation (and for same signs and a negative one)."""
    r = num / den
    a, b = num_se / abs(num), den_se / abs(den)
    sign = math.copysign(1.0, num * den)
    exact = [
        abs(r) * math.sqrt(max(a * a + b * b - 2.0 * rho * a * b * sign, 0.0))
        for rho in [i / 20.0 for i in range(-20, 21)]
    ]
    bound = m8b.ratio_stderr_bound(num, num_se, den, den_se)
    independent = m8b.ratio_stderr(num, num_se, den, den_se)
    assert bound == pytest.approx(max(exact), rel=1e-12)
    assert all(e <= bound + 1e-15 for e in exact)
    assert independent <= bound <= math.sqrt(2.0) * independent + 1e-15
    rho = 0.8 if sign < 0 else -0.8  # the correlation that makes the independent figure too small
    understated = abs(r) * math.sqrt(a * a + b * b - 2.0 * rho * a * b * sign)
    assert independent < understated <= bound
    assert math.isnan(m8b.ratio_stderr_bound(0.0, 0.1, 1.0, 0.1))


def _set_row(
    df: pd.DataFrame, product: str, rota: float, policy: str, **values: float
) -> pd.DataFrame:
    hit = (df["product"] == product) & (df["rota"] == rota) & (df["recalibration"] == policy)
    assert int(hit.sum()) == 1, (product, rota, policy)
    for col, v in values.items():
        df.loc[hit, col] = v
    return df


def test_s6_claim_guards_trip_on_a_synthetic_table(
    toy_build: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The two guards of the claim, tripped by a synthetic table C (the claim is redrawn from
    whatever table C holds): (1) a product classed *within* the band with a ratio below 1
    (0.80 ± 0.009: the static greek larger by 25%) blocks the sentence "|simulated| >=
    |static| on every decided product" and the claim is stated class by class; (2) a simulated
    P&L whose sign is not significant (-0.01 ± 0.02, z -0.5) is neither a gain nor a loss and is
    listed as sign-undecided."""
    toy = toy_build.require()
    outputs = _copy_outputs(toy.outputs_root, tmp_path / "outputs")
    path = outputs / "m8b" / "m8b_table_C.csv"
    df = pd.read_csv(path)
    pol = "sticky_breakeven"
    # autocall +1: static -0.074 (se 0.0005), simulated 0.8 x static (se 0.0005) -> within, < 1
    df = _set_row(
        df,
        "autocall 3y",
        1.0,
        pol,
        static_prediction=-0.074,
        static_prediction_se=0.0005,
        recal_pnl_desk=-0.0592,
        recal_pnl_desk_se=0.0005,
    )
    # cliquet +1: simulated twice the static -> above
    df = _set_row(
        df,
        "cliquet 1y",
        1.0,
        pol,
        static_prediction=-0.102,
        static_prediction_se=0.0005,
        recal_pnl_desk=-0.204,
        recal_pnl_desk_se=0.002,
    )
    # cliquet +2: a simulated P&L of undecided sign
    df = _set_row(df, "cliquet 1y", 2.0, pol, recal_pnl_desk=-0.01, recal_pnl_desk_se=0.02)
    df.to_csv(path, index=False)
    out = tmp_path / "s6"
    code, text, _ = _run("s6_fast.yaml", out, capsys, "--outputs", str(outputs), "--no-latex-check")
    assert code == 0, text
    res = Results.read(out / "results.parquet")
    one, two = f"{pol} | +1", f"{pol} | +2"
    v, se = res.value("agreement", f"autocall 3y | {pol} | +1", "ratio")
    assert v == pytest.approx(0.8) and s6_shadow_rotation.classify(v, se, 0.3) == "within"
    assert v + 2 * se < 1.0
    assert res.value("claim", one, "within")[0] == 1.0
    assert res.value("claim", one, "above")[0] == 1.0
    assert res.value("claim", one, "decided_static_not_larger")[0] == 1.0  # the cliquet only
    assert res.value("claim", two, "sign_undecided")[0] == 1.0
    assert res.record("claim", two, "sign_undecided")["note"] == "cliquet 1y"
    assert "cliquet 1y" not in res.record("claim", two, "desk_losses")["note"]
    md = (out / "study.md").read_text()
    claim = md[md.index("## The restricted claim") : md.index("## First-order agreement")]
    assert "|simulated| >= |static| at 2 stderr (same sign)" not in claim
    assert "no single direction holds for the decided products" in claim
    assert "1 where the static greek may be the larger" in claim
    assert "simulated sign undecided for cliquet 1y" in claim


def test_s6_reads_a_table_written_before_the_exact_twins(
    toy_build: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Study C's table as written before 2026-09-16 (``recal_se`` / ``total_se`` /
    ``static_se``, no ratio or nonlinearity stderr, no fallback / cap counts): the same P&L,
    ratios and nonlinearities; the refit flags read as -1 (not recorded)."""
    toy = toy_build.require()
    new = tmp_path / "new"
    code, text, _ = _run(
        "s6_fast.yaml", new, capsys, "--outputs", str(toy.outputs_root), "--no-latex-check"
    )
    assert code == 0, text
    outputs = _copy_outputs(toy.outputs_root, tmp_path / "outputs_old")
    path = outputs / "m8b" / "m8b_table_C.csv"
    df = pd.read_csv(path)
    df = df.rename(
        columns={
            "recal_pnl_desk_se": "recal_se",
            "total_pnl_desk_se": "total_se",
            "static_prediction_se": "static_se",
        }
    ).drop(columns=["ratio_se", "nonlinearity_se", "refits_fallback", "refits_capped"])
    df.to_csv(path, index=False)
    old = tmp_path / "old"
    code, text, _ = _run("s6_fast.yaml", old, capsys, "--outputs", str(outputs), "--no-latex-check")
    assert code == 0, text
    a, b = Results.read(new / "results.parquet"), Results.read(old / "results.parquet")
    for table in ("pnl", "claim", "ordering_sim", "ordering_sides"):
        pd.testing.assert_frame_equal(a.pivot(table), b.pivot(table))
    # the fallback / cap counts are absent from the old table: the rows say so, the rest agrees
    ca, cb = a.pivot("claim_columns"), b.pivot("claim_columns")
    counts = (
        "refits_recorded",
        "refits_fallback",
        "refits_capped",
        "fallback_share",
        "capped_share",
        "rows_counts_unrecorded",
    )
    keep = [c for c in ca.columns if not c.startswith(counts)]
    pd.testing.assert_frame_equal(ca[keep], cb[keep])
    assert (cb["rows_counts_unrecorded"] == cb["rows_ok"]).all()
    assert (cb["refits_recorded"] == 0).all() and (ca["rows_counts_unrecorded"] == 0).all()
    md_old = (old / "study.md").read_text()
    assert "do not record the fallback / cap counts" in md_old
    assert "Reading: on a fallback refit" not in md_old
    pa, pb = a.pivot("agreement"), b.pivot("agreement")
    same = [c for c in pa.columns if not c.startswith(("refits_fallback", "refits_capped"))]
    pd.testing.assert_frame_equal(pa[same], pb[same])
    assert (pb["refits_fallback"] == -1.0).all() and (pb["refits_capped"] == -1.0).all()


# --------------------------------------------------------------------------------------------
# S7
# --------------------------------------------------------------------------------------------


def test_s7_fast_on_the_toy_outputs(
    toy_build: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    toy = toy_build.require()
    out = tmp_path / "s7"
    code, text, _ = _run(
        "s7_fast.yaml", out, capsys, "--outputs", str(toy.outputs_root), *_roots(toy)
    )
    assert code == 0, text
    res, manifest = _check_outputs(out, S7_TABLES, S7_FIGURES, s7_hedging.QUESTION)
    arts = {a["path"] for a in manifest["artefacts"]}
    assert {f"m8b/m8b_table_{x}.csv" for x in "ABD"} | {s7_hedging.VERDICT_REPAIRED} <= arts
    assert "m8b/discriminator_verdict.json" not in arts  # the fallback is not read
    md = (out / "study.md").read_text()
    assert m8b.STUDY_D_HEADLINE in md
    assert SYNTHETIC_VERDICT["reason"] in md and "**Skipped:** world(s) historical" in md
    # the gate and the skipped rows
    assert res.value("gate", "discriminator", "world_ii_enabled")[0] == 0.0
    assert res.value("gate", "discriminator", "reason_matches_table_b")[0] == 1.0
    assert len(res.rows("b_skipped")) == 2
    # the reserve relative to the same world: zero on the same rows, by construction
    same = [r for r in res.rows("b_reserve") if r.startswith("same |")]
    assert same and all(res.value("b_reserve", r, "leakage_vs_same")[0] == 0.0 for r in same)
    # the headline checked on each product; the old table-A stem paired
    for product in res.rows("d_summary"):
        assert res.value("d_summary", product, "headline_holds")[0] == 1.0, product
    v, se = res.value("a_ranking", "2F | cliquet 1y | delta only", "desk_mean")
    assert math.isfinite(v) and se == pytest.approx(0.05)
    # the gate read the repaired history's verdict and says so; its z carries stderr 1
    assert res.value("gate", "discriminator", "repaired_history")[0] == 1.0
    assert "uses the repaired eSSVI history" in md and "eSSVI repaired" in md
    repaired = json.loads((toy.outputs_root / s7_hedging.VERDICT_REPAIRED).read_text())
    for rec in repaired["table"]:
        row = f"T={rec['T']:.4g}y window={rec['window']:g}d"
        assert res.value("gate_pillars", row, "ssr_fitted") == pytest.approx(
            (rec["ssr_ssvi"], rec["se_ssvi"])
        )
        z, z_se = res.value("gate_pillars", row, "z")
        assert z == pytest.approx(rec["z"]) and z_se == 1.0
        assert not res.is_exact("gate_pillars", row, "z")
    for rec in _records_of(res, "d_ranking", "mv_z"):
        if math.isfinite(rec["value"]):
            assert not rec["exact"] and rec["stderr"] == 1.0
    for col in ("std_rank", "distance_rank"):
        for rec in _records_of(res, "d_ranking", col):
            assert rec["exact"] and rec["note"] == s7_hedging.RANK_NOTE
    _check_rank_flags(res, toy.outputs_root / "m8b")
    # the spread of the implied MV deltas: the sum of the two extreme rows' stderrs (a bound
    # whatever their correlation), equal to the table's own spread; its noise floor d_k x mean se
    table_d = pd.read_csv(toy.outputs_root / "m8b" / "m8b_table_D.csv")
    for product in res.rows("d_common"):
        rows = table_d[(table_d["product"] == product) & (table_d["regime"] != "min_variance")]
        rows = rows.sort_values("mv_delta_implied")
        lo, hi = rows.iloc[0], rows.iloc[-1]
        v, se = res.value("d_common", product, "mv_common_spread")
        assert not res.is_exact("d_common", product, "mv_common_spread")
        assert v == pytest.approx(hi["mv_delta_implied"] - lo["mv_delta_implied"], rel=1e-12)
        assert se == pytest.approx(hi["mv_delta_implied_se"] + lo["mv_delta_implied_se"])
        assert res.value("d_common", product, "spread_matches_rows")[0] == 1.0
        # the spread over its noise floor d_k x mean se: a Monte Carlo ratio (the floor fixed)
        floor = s7_hedging.RANGE_FACTORS[len(rows)] * float(rows["mv_delta_implied_se"].mean())
        ratio, ratio_se = res.value("d_common", product, "spread_over_noise")
        assert not res.is_exact("d_common", product, "spread_over_noise")
        assert ratio == pytest.approx(v / floor, rel=1e-12)
        assert ratio_se == pytest.approx(se / floor, rel=1e-12)
        assert f"floor {floor:.4g}" in res.record("d_common", product, "spread_over_noise")["note"]
    assert "conservative" not in md
    _check_render_and_rerun(out, tmp_path / "s7_rerun")
    _latex_ok_or_skip(manifest)


def _decided(v: float, se: float, v_next: float, se_next: float) -> float:
    """The reference rule: the upper end of the row's 2-se interval lies below the lower end of
    the next-ranked row's (the ranks ascend with the value)."""
    return 1.0 if v + 2 * se < v_next - 2 * se_next else 0.0


def _check_rank_flags(res: Results, m8b_dir: Path) -> None:
    """Every rank's "decided at 2 se" flag, recomputed from the source tables: 1 / 0 against the
    next-ranked row of its group, NaN for the last or an unranked row; both values occur."""
    table_a = pd.read_csv(m8b_dir / "m8b_table_A.csv")
    table_d = pd.read_csv(m8b_dir / "m8b_table_D.csv")
    table_d["abs_distance"] = table_d["distance_to_mv"].abs()
    cases = (
        (table_a, ["pricing", "product"], "rank", "std", "std_se", "a_ranking", "rank"),
        (table_d, ["product"], "std_rank", "std", "std_se", "d_ranking", "std_rank"),
        (
            table_d,
            ["product"],
            "distance_rank",
            "abs_distance",
            "distance_to_mv_se",
            "d_ranking",
            "distance_rank",
        ),
    )
    headline = runner.load_study_config(CATALOGUE / "s7_fast.yaml").params["headline_regimes"]
    seen: set[float] = set()
    for frame, keys, rank_col, v_col, se_col, table, col in cases:
        for _, g in frame.groupby(keys):
            by_rank = {int(r[rank_col]): r for _, r in g.iterrows() if r[rank_col] >= 1}
            for _, r in g.iterrows():
                if table == "a_ranking":
                    row = f"{r['pricing']} | {r['product']} | {r['strategy']}"
                else:
                    row = f"{r['product']} | {r['regime']}"
                flag = res.value(table, row, f"{col}_decided")[0]
                assert res.is_exact(table, row, f"{col}_decided")
                nxt = by_rank.get(int(r[rank_col]) + 1) if r[rank_col] >= 1 else None
                if nxt is None:
                    assert math.isnan(flag), (table, row, col)
                    continue
                expected = _decided(r[v_col], r[se_col], nxt[v_col], nxt[se_col])
                assert flag == expected, (table, row, col)
                seen.add(flag)
        if table == "d_ranking" and col == "distance_rank":
            for product, g in frame.groupby("product"):
                model = g[g["regime"] == "model"].iloc[0]
                assert res.value("d_summary", str(product), "model_rank_decided")[0] == (
                    pytest.approx(
                        res.value("d_ranking", f"{product} | model", "distance_rank_decided")[0],
                        nan_ok=True,
                    )
                )
                assert res.value("d_summary", str(product), "model_rank")[0] == float(
                    model["distance_rank"]
                )
                # the headline set against every farther ranked regime
                k = len(headline)
                ranked = g[g["distance_rank"] >= 1]
                inside = ranked[ranked["distance_rank"] <= k]
                outside = ranked[ranked["distance_rank"] > k]
                top = float((inside["abs_distance"] + 2 * inside["distance_to_mv_se"]).max())
                bottom = float((outside["abs_distance"] - 2 * outside["distance_to_mv_se"]).min())
                assert res.value("d_summary", str(product), "headline_decided")[0] == (
                    1.0 if top < bottom else 0.0
                )
    assert seen == {0.0, 1.0}, seen


@pytest.mark.parametrize(
    ("entries", "expected"),
    [
        # separated at 2 se: 1.0 + 2*0.1 = 1.2 < 2.0 - 2*0.1 = 1.8
        ([("a", 1, 1.0, 0.1), ("b", 2, 2.0, 0.1)], {"a": 1.0, "b": math.nan}),
        # overlapping: 1.0 + 0.4 = 1.4 >= 1.5 - 0.4 = 1.1
        ([("a", 1, 1.0, 0.2), ("b", 2, 1.5, 0.2)], {"a": 0.0, "b": math.nan}),
        # the boundary (difference = 2 (se + se_next)) is not decided
        ([("a", 1, 1.0, 0.25), ("b", 2, 2.0, 0.25)], {"a": 0.0, "b": math.nan}),
        # unranked (-1), the reference row (0), a gap in the ranks, a missing stderr
        (
            [
                ("mv", 0, 0.0, 0.1),
                ("x", -1, 5.0, 0.1),
                ("a", 1, 1.0, math.nan),
                ("b", 2, 3.0, 0.1),
                ("c", 4, 9.0, 0.1),
            ],
            {"mv": math.nan, "x": math.nan, "a": math.nan, "b": math.nan, "c": math.nan},
        ),
        # the ranks, not the input order, decide the pairs
        (
            [("c", 3, 3.0, 0.1), ("a", 1, 1.0, 0.4), ("b", 2, 2.0, 0.1)],
            {"a": 0.0, "b": 1.0, "c": math.nan},
        ),
    ],
)
def test_s7_decided_flags(
    entries: list[tuple[str, float, float, float]], expected: dict[str, float]
) -> None:
    got = s7_hedging.decided_flags(entries)
    assert got.keys() == expected.keys()
    for k, v in expected.items():
        assert got[k] == pytest.approx(v, nan_ok=True), k


@pytest.mark.parametrize(
    ("inside", "outside", "expected"),
    [
        ([(0.05, 0.001), (0.06, 0.001)], [(0.08, 0.001), (0.09, 0.001)], 1.0),
        # a member with a wide error reaches past the nearest farther regime
        ([(0.05, 0.030), (0.06, 0.001)], [(0.08, 0.001), (0.09, 0.001)], 0.0),
        # a farther regime with a wide error reaches into the set
        ([(0.05, 0.001), (0.06, 0.001)], [(0.08, 0.001), (0.12, 0.030)], 0.0),
        # the boundary (top == bottom) is not decided
        ([(0.05, 0.005)], [(0.07, 0.005)], 0.0),
        ([(0.05, 0.001)], [], math.nan),
        ([], [(0.05, 0.001)], math.nan),
        ([(0.05, math.nan)], [(0.08, 0.001)], math.nan),
    ],
)
def test_s7_set_decided(
    inside: list[tuple[float, float]], outside: list[tuple[float, float]], expected: float
) -> None:
    assert s7_hedging.set_decided(inside, outside) == pytest.approx(expected, nan_ok=True)


def _table_d_row(
    regime: str, dist: float, dist_se: float, rank: int, std_rank: int
) -> dict[str, Any]:
    mv = regime == "min_variance"
    return {
        "product": "autocall 3y",
        "regime": regime,
        "status": "ok",
        "unit": "% of notional",
        "std": 8.0 + std_rank,
        "std_stderr": 0.01,
        "std_rank": std_rank,
        "mean_delta": 0.3,
        "mean_delta_stderr": 0.001,
        "lambda_star": 1.0,
        "lambda_star_stderr": 0.001,
        "std_at_lambda": 1.7,
        "std_at_lambda_stderr": 0.01,
        "mv_delta_implied": 0.30,
        "mv_delta_implied_stderr": 0.001,
        "distance_to_mv": dist,
        "distance_to_mv_stderr": dist_se,
        "distance_rank": rank,
        "mv_valid": True if mv else None,
        "mv_z": 0.1 if mv else None,
        "mv_note": "",
        "mv_common": 0.3,
        "mv_common_stderr": 0.001,
        "mv_common_spread": 0.0,
        "mv_common_rows": 4,
    }


@pytest.mark.parametrize(("strike_se", "expected"), [(0.030, 0.0), (0.001, 1.0)])
def test_s7_headline_decided_tests_the_whole_closest_set(strike_se: float, expected: float) -> None:
    """The boundary row (sticky_skew, rank 2) is ahead of the next (model) at 2 stderr in both
    cases; with sticky_strike's wide error (rank 1) the closest set is nevertheless not decided —
    the flag follows the set, not the boundary pair."""
    df = pd.DataFrame(
        [
            _table_d_row("sticky_strike", 0.050, strike_se, 1, 2),
            _table_d_row("sticky_skew", 0.060, 0.001, 2, 3),
            _table_d_row("model", 0.080, 0.001, 3, 4),
            _table_d_row("sticky_moneyness", 0.090, 0.001, 4, 5),
            _table_d_row("min_variance", 0.0, 0.001, 0, 1),
        ]
    )
    b = ResultsBuilder()
    s7_hedging._table_d(b, df, ["sticky_strike", "sticky_skew"], "computed")
    res = b.build()
    assert res.value("d_ranking", "autocall 3y | sticky_skew", "distance_rank_decided")[0] == 1.0
    assert res.value("d_summary", "autocall 3y", "headline_holds")[0] == 1.0
    assert res.value("d_summary", "autocall 3y", "headline_decided")[0] == expected
    assert unclassified_exact_rows(res, s7_hedging.EXACT_KINDS).empty


def test_exact_kinds_are_declared() -> None:
    for module in MODULES.values():
        kinds = parse_exact_kinds(module.EXACT_KINDS)
        assert kinds
    ranks = [k for k in parse_exact_kinds(s7_hedging.EXACT_KINDS) if k.kind == "rank"]
    assert {(k.table, k.column, k.decided_flag) for k in ranks} == {
        ("a_ranking", "rank", "rank_decided"),
        ("d_ranking", "std_rank", "std_rank_decided"),
        ("d_ranking", "distance_rank", "distance_rank_decided"),
        ("d_summary", "model_rank", "model_rank_decided"),
    }


def test_s7_gate_falls_back_to_the_m8b_verdict(
    toy_build: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Without the repaired history's verdict the gate reads the M8b file (the documented
    fallback) and study.md says the table uses the pre-repair history; without either, exit 2
    with the repaired run's command."""
    toy = toy_build.require()
    outputs = _copy_outputs(toy.outputs_root, tmp_path / "outputs")
    (outputs / s7_hedging.VERDICT_REPAIRED).unlink()
    out = tmp_path / "s7"
    code, text, _ = _run("s7_fast.yaml", out, capsys, "--outputs", str(outputs), "--no-latex-check")
    assert code == 0, text
    res = Results.read(out / "results.parquet")
    manifest = json.loads((out / "manifest.json").read_text())
    assert s7_hedging.VERDICT_FALLBACK in {a["path"] for a in manifest["artefacts"]}
    rec = res.record("gate", "discriminator", "repaired_history")
    assert rec["value"] == 0.0 and "fallback" in rec["note"]
    assert s7_hedging.FALLBACK_FITTED_HISTORY in rec["note"]
    fallback = json.loads((outputs / s7_hedging.VERDICT_FALLBACK).read_text())
    first = fallback["table"][0]
    row = f"T={first['T']:.4g}y window={first['window']:g}d"
    assert res.value("gate_pillars", row, "ssr_fitted")[0] == pytest.approx(first["ssr_ssvi"])
    md = (out / "study.md").read_text()
    assert "uses the pre-repair history" in md
    (outputs / s7_hedging.VERDICT_FALLBACK).unlink()
    code, text, _ = _run("s7_fast.yaml", tmp_path / "s7_none", capsys, "--outputs", str(outputs))
    assert code == runner.EXIT_MISSING, text
    assert s7_hedging.VERDICT_REPAIRED_COMMAND in text


def test_s7_missing_table_prints_its_command(
    toy_build: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    toy = toy_build.require()
    outputs = _copy_outputs(toy.outputs_root, tmp_path / "outputs")
    (outputs / "m8b" / "m8b_table_D.csv").unlink()
    out = tmp_path / "s7"
    code, text, _ = _run("s7_fast.yaml", out, capsys, "--outputs", str(outputs))
    assert code == runner.EXIT_MISSING, text
    assert ".venv/bin/python scripts/m8b.py --study D --resume" in text
    assert "m8b/m8b_table_D.csv" in text
    assert not out.exists()


# --------------------------------------------------------------------------------------------
# S5
# --------------------------------------------------------------------------------------------


def _toy_marking_grid(tmp: Path) -> Path:
    p = tmp / "toy_marking_test.yaml"
    p.write_text(TOY_MARKING_GRID)
    return p


def _only_ids(text: str) -> list[str]:
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("volsto-precompute"))
    tokens = line.split()
    i = tokens.index("--only")
    ids: list[str] = []
    for t in tokens[i + 1 :]:
        if t.startswith("--"):
            break
        ids.append(t)
    return ids


def _grid_of(text: str) -> str:
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("volsto-precompute"))
    tokens = line.split()
    return tokens[tokens.index("--grid") + 1]


def test_s5_fast_missing_points_print_the_precompute_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """On an empty store the four toy marks are missing: one precompute line naming the
    configured grid, exit 2, nothing computed (no fixture needed)."""
    grid = _toy_marking_grid(tmp_path)
    out = tmp_path / "s5"
    code, text, _ = _run(
        "s5_fast.yaml",
        out,
        capsys,
        "--grid",
        str(grid),
        "--store",
        str(tmp_path / "empty_store"),
        "--cache",
        str(tmp_path / "empty_cache"),
    )
    assert code == runner.EXIT_MISSING, text
    assert sorted(_only_ids(text)) == sorted(
        f"marking:placeholder:ssr{s}:eps{e}" for s in ("1", "1.5") for e in ("0.05", "0.1")
    )
    assert Path(_grid_of(text)).name == grid.name
    assert not out.exists() and not (tmp_path / "empty_store").exists()
    assert not (tmp_path / "empty_cache").exists()


def test_s5_full_config_lists_the_default_grid_marks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The full config on an empty store: the 36 marking points of the default grid (3 SPX
    snapshots x 4 ssr x 3 eps) in one line naming configs/grids/default.yaml."""
    out = tmp_path / "s5"
    code, text, _ = _run(
        "s5.yaml",
        out,
        capsys,
        "--store",
        str(tmp_path / "empty_store"),
        "--cache",
        str(tmp_path / "empty_cache"),
        "--outputs",
        str(tmp_path / "empty_outputs"),
    )
    assert code == runner.EXIT_MISSING, text
    ids = _only_ids(text)
    assert len(ids) == 36 and all(i.startswith("marking:spx_2022-") for i in ids)
    assert _grid_of(text).endswith("configs/grids/default.yaml")
    assert "scripts/m7_p1_marking.py" in text  # the recorded M7 fits are absent there too


def test_s5_missing_points_name_the_configured_grid_on_a_mixed_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A store whose run record names another grid (as ``outputs/store`` does): the line names
    the configured grid — the one holding the marking ids — and a note names the recorded one.
    (Until the runner's fix of 2026-09-16 the line quoted the recorded grid, a command that
    ``volsto-precompute`` rejects with "unknown point id(s)".)"""
    store = tmp_path / "store"
    ResultsStore(store).write_run(
        {"grid_path": "configs/grids/placeholder_cached.yaml", "shard": "1/1", "argv": []}
    )
    grid = _toy_marking_grid(tmp_path)
    code, text, _ = _run(
        "s5_fast.yaml",
        tmp_path / "s5",
        capsys,
        "--grid",
        str(grid),
        "--store",
        str(store),
        "--cache",
        str(tmp_path / "cache"),
    )
    assert code == runner.EXIT_MISSING, text
    assert Path(_grid_of(text)).name == grid.name, _grid_of(text)
    assert "placeholder_cached.yaml" in text  # the note naming the recorded grid


def test_s5_fast_on_the_toy_marking_build(
    toy_marking_build: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    toy = toy_marking_build.require()
    outputs = Path(toy.outputs_root)  # the synthetic outputs beside the build (the M7 fits)
    assert (outputs / "m7" / "p1_marking_fits.csv").is_file()
    cache_before = _files(Path(toy.cache_root))
    store_before = _files(Path(toy.store_root))
    out = tmp_path / "s5"
    flags: Sequence[str] = (
        "--grid",
        str(toy.grid_path),
        "--store",
        str(toy.store_root),
        "--cache",
        str(toy.cache_root),
        "--outputs",
        str(outputs),
    )
    code, text, _ = _run("s5_fast.yaml", out, capsys, *flags)
    assert code == 0, text
    res, manifest = _check_outputs(out, S5_TABLES, S5_FIGURES, s5_marking.QUESTION)
    assert _files(Path(toy.cache_root)) == cache_before, "the cache changed: something calibrated"
    assert _files(Path(toy.store_root)) == store_before
    assert len(manifest["store_points"]) == 4
    assert manifest["particles"] == [20_000]
    # the binding map fitted inline reproduces the stored fits at the grid points
    bm = res.pivot("binding_map")
    grid_rows = bm[bm["grid_point"] == 1.0]
    assert len(grid_rows) == 2
    assert (grid_rows["store_param_diff"] < 1e-8).all(), grid_rows["store_param_diff"]
    assert (grid_rows["store_status_match"] == 1.0).all()
    # realised SSR with its stderr; the 1y -> 2y window beyond the 1y horizon, flagged
    marks = res.pivot("marks")
    assert len(marks) == 4 and marks["ssr_lsv@1y_stderr"].gt(0).all()
    fwd = res.pivot("forward_skew")
    assert (fwd["beyond_horizon@1y-into-1y"] == 1.0).all()
    # the cost of the reference mark is not listed; the others are
    assert "placeholder | ssr 1 | eps 0.1" not in res.rows("mark_cost")
    assert len(res.rows("mark_cost")) == 3
    md = (out / "study.md").read_text()
    assert s5_marking.PLACEHOLDER_WARNING in md and "**Fast mode**" in md
    assert "not a leverage calibration" in md
    assert "As recorded by `scripts/m7_p1_marking.py`" in md
    # |L-1| is exact with the orchestrator's note, and its figure draws no error bar
    rec = res.record("marks", "placeholder | ssr 1 | eps 0.1", "mean_abs_L_minus_1")
    assert rec["exact"] and rec["note"] == s5_marking.L_NOTE
    assert "particle-seed noise is not estimated upstream" in s5_marking.L_NOTE
    fig = s5_marking._draw_leverage(res)
    bars = [c for ax in fig.axes for c in ax.containers if hasattr(c, "has_yerr")]
    assert bars and not any(c.has_yerr for c in bars)
    # the cost of a mark: quadrature labelled as such; its z-score carries stderr 1
    for rec in _records_of(res, "mark_cost", "atm_vol"):
        assert rec["note"].endswith(QUADRATURE_NOTE)
    for rec in _records_of(res, "mark_cost_z", "atm_vol"):
        assert not rec["exact"] and rec["stderr"] == 1.0
    assert "conservative" not in md
    walls = manifest["records"]["binding_map_fit_seconds"]
    print(f"inline fits: {walls}")
    _check_render_and_rerun(out, tmp_path / "s5_rerun")
    assert _files(Path(toy.cache_root)) == cache_before
    _latex_ok_or_skip(manifest)


def test_s5_two_forward_windows_of_equal_length(
    toy_marking_build: Any, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The shipped full config's windows [[1, 2], [2, 3]] are both 1y long: every forward-skew
    column, the spot skew included, is keyed by its window, so the build does not collide."""
    toy = toy_marking_build.require()
    out = tmp_path / "s5_windows"
    full = runner.load_study_config(CATALOGUE / "s5.yaml").params["forward_windows"]
    windows = [[float(a), float(b)] for a, b in full]
    assert len(windows) == 2 and len({b - a for a, b in windows}) == 1
    code, text, _ = _run(
        "s5_fast.yaml",
        out,
        capsys,
        "--grid",
        str(toy.grid_path),
        "--store",
        str(toy.store_root),
        "--cache",
        str(toy.cache_root),
        "--outputs",
        str(toy.outputs_root),
        "--set",
        f"forward_windows={json.dumps(windows)}",
        "--no-latex-check",
    )
    assert code == 0, text
    res = Results.read(out / "results.parquet")
    _check_exact_kinds(res, s5_marking.QUESTION)
    labels = [s5_marking.window_label(a, b) for a, b in windows]
    fwd = res.pivot("forward_skew")
    assert len(fwd) == 4
    for w in labels:
        for stem in ("spot_skew", "fwd_skew", "ratio", "beyond_horizon"):
            assert f"{stem}@{w}" in fwd.columns, (stem, w, sorted(fwd.columns))
    long = res.long("forward_skew")
    spot = long[long["column"].str.startswith("spot_skew@")]
    assert not spot.duplicated(["row", "column"]).any()
    assert sorted(set(spot["column"])) == sorted(f"spot_skew@{w}" for w in labels)
    # the same tenor on the same snapshot: one spot skew, recorded under each window
    a, b = (f"spot_skew@{w}" for w in labels)
    assert (fwd[a] == fwd[b]).all()


def test_s5_and_the_precompute_share_one_marking_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """S5's binding map and the precompute's marking points go through one helper,
    :func:`volsto.viewers.grid.marking_fit`: S5 imports it, and ``resolve_marking`` calls it."""
    from volsto.viewers import grid as grid_mod

    assert vars(s5_marking)["marking_fit"] is grid_mod.marking_fit

    class CalledError(Exception):
        pass

    def stub(surface: Any, ssr_target: float, skew_eps: float, **kw: Any) -> Any:
        raise CalledError((ssr_target, skew_eps, kw))

    monkeypatch.setattr(grid_mod, "marking_fit", stub)
    g = grid_mod.load_grid(ROOT / "configs" / "grids" / "toy_marking.yaml")
    point = next(p for p in grid_mod.enumerate_points(g) if p.mode == "marking")
    with pytest.raises(CalledError) as info:
        grid_mod.resolve_marking(point, surface=None)  # type: ignore[arg-type]
    assert info.value.args[0] == (
        point.axes["ssr_target"],
        point.axes["skew_eps"],
        {"fit": "m7", "snapshot": None},
    )


def _synthetic_marking_store(tmp: Path) -> tuple[Path, Path]:
    """The synthetic store (one SPX 2022-09-15 marking point with light-tier risk rows) and its
    grid as a YAML — written, never computed."""
    import yaml
    from _synthetic_store import make_synthetic_store, synthetic_grid

    from volsto.viewers.grid import grid_mapping

    grid = tmp / "synthetic.yaml"
    grid.write_text(yaml.safe_dump(grid_mapping(synthetic_grid()), sort_keys=False))
    make_synthetic_store(tmp / "store")
    return grid, tmp / "store"


SYNTHETIC_GREEKS = ("delta[model]", "skew_tent[1y]")


def _s5_synthetic_flags(grid: Path, store: Path, outputs: Path, names: Sequence[str]) -> list[str]:
    quoted = ", ".join(f'"{n}"' for n in names)
    return [
        "--grid",
        str(grid),
        "--store",
        str(store),
        "--cache",
        str(store.parent / "cache"),
        "--outputs",
        str(outputs),
        "--set",
        "surfaces=[spx_2022-09-15]",
        "--set",
        "binding_map.ssr_target=[1.0]",
        "--set",
        "products=[cliquet_1y, atm_vol]",
        "--set",
        "greeks.products=[autocall 3y, cliquet 1y]",
        "--set",
        f"greeks.names=[{quoted}]",
    ]


def test_s5_greeks_on_the_synthetic_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The Greek path on the synthetic store's risk rows (the risk report's names): the stored
    fractions of notional x 100 in '% notional <store unit>', each with its stderr; a Greek name
    no stored point carries is a config error (exit 1); a product the store has no risk rows for
    is refused when the params load."""
    from _synthetic_store import make_synthetic_outputs

    from volsto.config import ConfigError
    from volsto.viewers.store import StoreReader

    grid, store = _synthetic_marking_store(tmp_path)
    outputs = tmp_path / "outputs"
    make_synthetic_outputs(outputs)
    out = tmp_path / "s5"
    flags = _s5_synthetic_flags(grid, store, outputs, SYNTHETIC_GREEKS)
    code, text, _ = _run("s5_fast.yaml", out, capsys, *flags)
    assert code == 0, text
    res = Results.read(out / "results.parquet")
    risk = StoreReader(store).risk()
    pid = "marking:spx_2022-09-15:ssr1:eps0.1"
    row = "spx_2022-09-15 | ssr 1 | eps 0.1"
    for product in ("autocall 3y", "cliquet 1y"):
        table = f"greeks_{s5_marking.slug(product)}"
        assert (out / "tables" / f"{table}_spx_2022_09_15.tex").is_file()
        for name in SYNTHETIC_GREEKS:
            stored = risk[
                (risk["point_id"] == pid) & (risk["product"] == product) & (risk["name"] == name)
            ].iloc[0]
            v, se = res.value(table, row, name)
            assert v == pytest.approx(100.0 * stored["value"], rel=1e-12)
            assert se == pytest.approx(100.0 * stored["value_stderr"], rel=1e-12)
            rec = res.record(table, row, name)
            assert rec["unit"] == f"% notional {stored['unit']}" and not rec["exact"]
            assert rec["note"] == s5_marking.RISK_NOTE
    md = (out / "study.md").read_text()
    assert "[% notional per vol point of 90/110 skew]" in md
    # the forward 90/110 skew's stderr is the SUM of the two strikes' errors (a bound for any
    # correlation): the synthetic smile has 0.0005 at each strike -> 0.10 vol pts, not 0.0707
    smile = StoreReader(store).forward_smile(pid)
    win = smile[(smile["t1"] == 1.0) & (smile["t2"] == 2.0)]
    lo = win[win["strike_moneyness"] == 0.9].iloc[0]
    hi = win[win["strike_moneyness"] == 1.1].iloc[0]
    v, se = res.value("forward_skew", row, "fwd_skew@1y-into-1y")
    assert v == pytest.approx(100.0 * (lo["iv"] - hi["iv"]), rel=1e-12)
    assert se == pytest.approx(0.10, rel=1e-12)
    assert se == pytest.approx(100.0 * (lo["iv_stderr"] + hi["iv_stderr"]), rel=1e-12)
    assert res.record("forward_skew", row, "fwd_skew@1y-into-1y")["note"] == (
        s5_marking.FWD_SKEW_SE_NOTE
    )
    # a Greek no stored point carries (the old synthetic name): a config error, exit 1
    bad = tmp_path / "s5_bad"
    code, text, _ = _run(
        "s5_fast.yaml",
        bad,
        capsys,
        *_s5_synthetic_flags(grid, store, outputs, ("skew_T[1y]",)),
    )
    assert code == runner.EXIT_FAILED and "skew_T[1y]" in text, text
    params = dict(runner.load_study_config(CATALOGUE / "s5_fast.yaml").params)
    params["greeks"] = {"products": ["phoenix 3y"], "names": ["delta[model]"]}
    with pytest.raises(ConfigError, match="phoenix 3y"):
        s5_marking.validate_params(params)


# --------------------------------------------------------------------------------------------
# the repository's own outputs (artefact reads only)
# --------------------------------------------------------------------------------------------


@pytest.mark.skipif(
    not (ROOT / "outputs" / "m8b" / "m8b_table_C.csv").exists()
    or not (ROOT / "outputs" / "m7" / "p1_marking_shadow_rotation.csv").exists(),
    reason="outputs/m7 or outputs/m8b absent",
)
@pytest.mark.parametrize("config", ["s6.yaml", "s7.yaml"])
def test_full_configs_on_the_repository_outputs(
    config: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """S6 and S7 on whatever the repository's outputs hold today (study C is being re-run: the
    table's state is printed, not asserted)."""
    out = tmp_path / config.removesuffix(".yaml")
    code, text, _ = _run(config, out, capsys, "--no-latex-check")
    assert code == 0, text
    res = Results.read(out / "results.parquet")
    if config == "s6.yaml":
        state = res.pivot("table_c_state").iloc[0]
        print(state.to_string())
        assert state["rows_expected"] == 45
    else:
        print(res.pivot("d_summary").to_string())
        assert set(res.rows("d_summary")) == {"autocall 3y", "vanilla 1y atm"}
    assert os.path.getsize(out / "study.md") > 0
