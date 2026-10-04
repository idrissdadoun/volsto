"""Study runner (SPEC §10.1, M10 Part 1): ``volsto.studies.runner`` / ``results`` / ``latex`` /
``style``, the calibration guard ``volsto.calibration.guard`` and the ``volsto-study`` CLI.

**Nothing here calibrates.**  The example study (``tests/_example_study.py``) prices European
calls under Black–Scholes by Monte Carlo (2·10⁴ paths, one exact step) beside the closed form.
Its ``attack`` modes try to calibrate or to write the cache by every route the runner
verification found (``scratchpad/runner_verify``: an alias bound at import, a dict and a
``functools.partial``, a thread pool, a spawned process pool, a caught refusal, a thread acting
during render, a study writing the configured cache root); the direct calls pass ``None``
arguments, so a failing guard raises on the arguments instead of calibrating, and the one real
route uses a 5000-particle, 0.1y spec.  The toy-store test reads the one sanctioned toy build of
``tests/conftest.py::toy_build``.

What is asserted:

* run → every output file, every contract manifest key, ``recalibrated`` false as observed,
  the roots not created, ``argv`` masked, no staging directory left;
* rerun from the stored manifest (the YAML on disk modified) → nothing moved, exit 0; a stored
  value perturbed by 3 stderr → moved, exit 1; ``--nse nan`` → exit 1;
* missing point / leverage / artefact / off-grid id → runnable commands (one executed with
  ``--dry-run``), exit 2, ``compute`` never called, no directory; a point requirement with
  ``grid: null`` → exit 1; a leverage miss inside ``compute`` → exit 2 and no directory;
* the guard refuses every route, nests, spans threads, reaches spawned children through
  ``VOLSTO_FORBID_CALIBRATION`` and restores the environment; a caught refusal fails the run
  (``calibration_refusals``); a child that escaped the prohibition leaves a marker and makes
  ``recalibrated`` true (exit 1); a cache write by a concurrent process, a thread or the study
  itself is "cache changed during the run, not attributed" (a warning, exit 0);
* the particle kernel is reached only through ``calibrate_leverage``, whose first statement is
  the guard (an AST walk over ``volsto/``);
* ``ctx.cache`` hits (direct and through ``LSVBuilder``) are recorded with ``n_particles``;
* render after deleting ``tables/`` and ``figures/`` rebuilds byte-identical ``.tex`` files and
  ``study.md`` and the same figure set; with the LaTeX check it recompiles ``study.pdf``, without
  it a stale PDF is removed;
* the strict YAML (duplicates, ``1e-3``, ``yes``, empty ``--set``, ``nan``);
* ``format_value`` rounding, the shared-exponent form, no exception for any finite double;
* the fast-mode banner under the question (study.md and study.tex, full mode without it, render
  byte-identical) and the mode in the provenance block and in ``volsto-study list``;
* ``Results`` invariants; ``diff_results`` moved / changed / added / removed, the zero-stderr
  tolerance, ``nse`` validation;
* the LaTeX check: the full document compiles, a 60-row 12-column table split into three scaled
  parts by ``split_table``; a 40-row longtable with a literal ``$``, explicit ``\\(...\\)`` math and
  mapped glyphs passes; a dropped glyph, a float taller than the page and an overfull box fail;
  a verbatim terminator in the narrative is refused; ``split_table`` by parts, groups, callables;
* one-line errors (the traceback only with ``-v``), the toy store's requirement check.

Wall clocks are printed, never asserted.
"""

from __future__ import annotations

import ast
import functools
import json
import logging
import math
import os
import random
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import _example_study
import numpy as np
import pandas as pd
import pytest
import yaml

from volsto.calibration import cache as cache_mod
from volsto.calibration import guard
from volsto.calibration.cache import CacheMissError, LeverageCache, spec_key
from volsto.calibration.guard import CalibrationForbiddenError
from volsto.config import ConfigError
from volsto.studies import latex, runner, style
from volsto.studies.results import (
    DIMENSIONLESS,
    Column,
    Results,
    ResultsBuilder,
    ResultsError,
    TableSpec,
    changed,
    diff_results,
)
from volsto.studies.runner import (
    ConfigOverrides,
    MissingRequirements,
    ReadOnlyLeverageCache,
    StudyConfig,
    StudyContext,
)
from volsto.viewers.grid import enumerate_points, load_grid

ROOT = Path(__file__).resolve().parents[1]
TOY_GRID = ROOT / "configs" / "grids" / "toy.yaml"
#: Every key the M10 contract lists for manifest.json.
CONTRACT_MANIFEST_KEYS = (
    "study",
    "runner",
    "config",
    "config_path",
    "config_sha256",
    "git_commit",
    "git_dirty",
    "code_version",
    "calibration_code_tag",
    "grid",
    "cache_keys",
    "store_points",
    "artefacts",
    "particles",
    "seeds",
    "n_paths",
    "mode",
    "wall_clock",
    "recalibrated",
    "host",
    "python",
    "numpy",
    "numba",
    "created_utc",
    "argv",
    "results_sha256",
)
EXAMPLE_PARAMS: dict[str, Any] = {
    "vol": 0.2,
    "maturity": 1.0,
    "strikes": [80, 100, 120],
    "n_paths": 20_000,
    "requires": [],
}
FAKE_POINT = "lv:" + "d" * 64
FAST_BANNER_MD = "> **FAST MODE** — reduced paths and toy inputs: plumbing check, not results."
FAST_BANNER_TEX = (
    r"\noindent\fbox{\parbox{\dimexpr\linewidth-2\fboxsep-2\fboxrule\relax}{"
    r"\textbf{FAST MODE} --- reduced paths and toy inputs: plumbing check, not results.}}"
)
FAKE_KEY = "0" * 64
FAKE_ARTEFACT = "outputs/m8b/m8b_table_D.csv"


def example_mapping(**changes: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "example_bs_vanilla",
        "runner": "_example_study",
        "grid": None,
        "store": "outputs/store",
        "cache": "cache",
        "outputs": "outputs",
        "mode": "fast",
        "seeds": {"pricing": 2024},
        "params": json.loads(json.dumps(EXAMPLE_PARAMS)),
    }
    data.update(changes)
    return data


def write_config(path: Path, data: dict[str, Any]) -> Path:
    path.write_text(yaml.safe_dump(data, sort_keys=False))
    return path


def roots_argv(base: Path) -> list[str]:
    return [
        "--store",
        str(base / "store"),
        "--cache",
        str(base / "cache"),
        "--outputs",
        str(base / "outputs"),
    ]


def staging_leftovers(parent: Path) -> list[str]:
    return [p.name for p in parent.iterdir() if runner.STAGING_MARK in p.name]


def assert_guard_clear() -> None:
    assert guard.depth() == 0
    assert guard.ENV_VAR not in os.environ


@pytest.fixture(scope="module")
def example_run(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One CLI run of the example study (no LaTeX check), shared by the module's tests; tests
    that modify the output work on a copy."""
    base = tmp_path_factory.mktemp("example")
    cfg = write_config(base / "example.yaml", example_mapping())
    out = base / "out"
    argv = ["run", str(cfg), "--out", str(out), *roots_argv(base), "--no-latex-check"]
    t0 = time.perf_counter()
    code = runner.main(argv)
    wall = time.perf_counter() - t0
    print(f"example run: exit {code}, wall clock {wall:.2f} s")
    return {"base": base, "config": cfg, "out": out, "code": code, "wall": wall}


def copy_output(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("rerun"))
    return dst


# --------------------------------------------------------------------------------------------
# run / rerun / render
# --------------------------------------------------------------------------------------------


def test_run_writes_outputs_and_manifest(example_run: dict[str, Any]) -> None:
    assert example_run["code"] == 0
    out, base = example_run["out"], example_run["base"]
    for rel in (
        "results.parquet",
        "tables/calls.tex",
        "figures/mc_vs_bs.pdf",
        "figures/mc_vs_bs.png",
        "study.md",
        "study.tex",
        "manifest.json",
    ):
        assert (out / rel).is_file(), rel
    assert not (out / "study.pdf").exists()  # --no-latex-check
    assert staging_leftovers(base) == []
    # the study read nothing: the roots were not created
    for d in ("store", "cache", "outputs"):
        assert not (base / d).exists(), d
    m = json.loads((out / "manifest.json").read_text())
    missing = [k for k in CONTRACT_MANIFEST_KEYS if k not in m]
    assert missing == []
    assert m["recalibrated"] is False and m["calibration_refusals"] == 0
    assert m["calibrations_started"] == [] and m["cache_changed_during_run"] is False
    obs = m["cache_observation"]
    assert set(obs["stages"]) == {"before", "after_compute", "after_render", "final"}
    assert obs["new_keys"] == [] and obs["rewritten_keys"] == [] and obs["manifest_rows_added"] == 0
    assert obs["attribution"] == ""
    assert m["study"] == "example_bs_vanilla" and m["runner"] == "_example_study"
    assert set(m["wall_clock"]) == {"requirements", "compute", "render", "total"}
    assert all(math.isfinite(v) and v >= 0 for v in m["wall_clock"].values())
    assert m["n_paths"] == 20_000 and m["seeds"] == {"pricing": 2024} and m["mode"] == "fast"
    assert m["particles"] == [] and m["cache_keys"] == [] and m["store_points"] == []
    assert m["grid"] is None and m["artefacts"] == []
    assert m["calibration_code_tag"] == cache_mod.CALIBRATION_CODE_TAG
    assert m["code_version"] == cache_mod.code_version()
    assert isinstance(m["git_commit"], str) and len(m["git_commit"]) == 40
    assert isinstance(m["git_dirty"], bool)
    assert m["tables"] == ["calls"] and m["table_specs"] == ["calls"]
    assert m["figures"] == ["mc_vs_bs"]
    # the stored config is the resolved one: the root overrides, not the YAML's paths
    cfg = StudyConfig.from_mapping(m["config"])
    assert cfg.path("store") == (base / "store").resolve()
    assert m["config_sha256"] == cfg.sha256()
    assert m["results_sha256"] == runner.file_sha256(out / "results.parquet")
    # argv: roots masked, other paths masked when outside the repository
    argv = m["argv"]
    assert argv[:2] == ["volsto-study", "run"]
    assert argv[argv.index("--store") + 1] == "<store>"
    assert argv[argv.index("--cache") + 1] == "<cache>"
    assert argv[argv.index("--outputs") + 1] == "<outputs>"
    assert argv[argv.index("--out") + 1] == "<out>"
    assert not any(str(base) in a for a in argv)
    # results: 3 strikes x 4 columns + the budget row; Monte Carlo rows carry a stderr
    res = Results.read(out / "results.parquet")
    assert len(res) == 13 and m["n_results"] == 13
    mc = res.frame[~res.frame["exact"]]
    assert len(mc) == 6 and np.isfinite(mc["stderr"]).all() and (mc["stderr"] > 0).all()
    for row in res.rows("calls"):
        v, se = res.value("calls", row, "mc_price")
        bs, _ = res.value("calls", row, "bs_price")
        print(f"{row}: MC {v:.4f} ± {se:.4f}, BS {bs:.4f}, z {(v - bs) / se:+.2f}")
    # study.md: title, then the question first; the provenance block at the end
    md = (out / "study.md").read_text().splitlines()
    assert md[0] == f"# {_example_study.TITLE}"
    assert md[2] == f"**Question.** {_example_study.QUESTION}"
    assert md[4] == FAST_BANNER_MD  # fast mode, right under the question
    text = "\n".join(md)
    assert text.count("FAST MODE") == 1 and "| mode | fast |" in text
    assert "| recalibrated | no |" in text and "| cache changed during the run | no |" in text
    assert "± " in text
    assert text.index("**Table `calls`.**") < text.index("- one exact log-Euler step")
    tex = (out / "study.tex").read_text()
    tex_lines = tex.splitlines()
    q = next(i for i, ln in enumerate(tex_lines) if ln.startswith(r"\noindent\textbf{Question.}"))
    assert tex_lines[q + 1 : q + 4] == ["", r"\medskip", FAST_BANNER_TEX]
    assert r"\input{tables/calls.tex}" in tex
    assert r"\includegraphics[width=0.85\linewidth]{figures/mc_vs_bs.pdf}" in tex
    assert r"\(\sigma\)" in tex and r"a \$5 note" in tex  # explicit math kept, $ literal
    assert_guard_clear()
    print(f"example run wall clock {example_run['wall']:.2f} s; recalibrated: no")


def test_rerun_from_manifest_nothing_moved(
    example_run: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = copy_output(example_run["out"], tmp_path / "out")
    # the YAML on disk now says something else: rerun must not read it
    write_config(
        example_run["config"],
        example_mapping(params={**EXAMPLE_PARAMS, "n_paths": 4000}, seeds={"pricing": 7}),
    )
    rerun_out = tmp_path / "rerun"
    t0 = time.perf_counter()
    code = runner.main(["rerun", str(src), "--out", str(rerun_out)])
    print(f"rerun wall clock {time.perf_counter() - t0:.2f} s")
    assert code == 0
    diff = pd.read_csv(rerun_out / "diff.csv")
    assert len(diff) == 13 and set(diff["status"]) == {"unchanged"}
    # identical seeds: the Monte Carlo numbers are reproduced exactly, not just within 2 stderr
    assert float(np.nanmax(np.abs(diff["delta"]))) == 0.0
    md = (rerun_out / "diff.md").read_text()
    assert "**Verdict:** nothing moved." in md
    assert "**Verdict:** nothing moved." in capsys.readouterr().out
    m = json.loads((rerun_out / "manifest.json").read_text())
    assert m["recalibrated"] is False and m["n_paths"] == 20_000
    src_manifest = json.loads((src / "manifest.json").read_text())
    assert m["rerun_of"] and m["config_sha256"] == src_manifest["config_sha256"]
    # the default rerun directory lives under the output directory, which a later run keeps
    code = runner.main(["rerun", str(src)])
    assert code == 0
    assert len(list((src / "rerun").iterdir())) == 1
    (src / "notes.txt").write_text("mine")
    cfg = write_config(tmp_path / "again.yaml", example_mapping())
    assert runner.main(["run", str(cfg), "--out", str(src), *roots_argv(tmp_path)]) == 0
    assert (src / "notes.txt").read_text() == "mine" and len(list((src / "rerun").iterdir())) == 1
    assert (src / "study.pdf").is_file() or runner.find_tectonic() is None
    # --nse must be finite and positive (exit 1, not the missing-requirement 2)
    for bad in ("nan", "inf", "0", "-1"):
        assert runner.main(["rerun", str(src), "--nse", bad]) == 1
    assert "--nse must be finite and positive" in capsys.readouterr().err
    with pytest.raises(SystemExit) as usage:  # a usage error exits 1 (2 means missing inputs)
        runner.main(["rerun", str(src), "--set", "x=1"])
    assert usage.value.code == 1


def test_rerun_detects_a_moved_number(
    example_run: dict[str, Any], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    src = copy_output(example_run["out"], tmp_path / "out")
    df = pd.read_parquet(src / "results.parquet")
    hit = (df["table"] == "calls") & (df["row"] == "K=100") & (df["column"] == "mc_price")
    assert hit.sum() == 1
    se = float(df.loc[hit, "stderr"].iloc[0])
    df.loc[hit, "value"] = df.loc[hit, "value"] + 3.0 * se
    df.to_parquet(src / "results.parquet", index=False)
    code = runner.main(["rerun", str(src), "--out", str(tmp_path / "rerun")])
    assert code == 1
    diff = pd.read_csv(tmp_path / "rerun" / "diff.csv")
    moved = diff[diff["status"] != "unchanged"]
    assert len(moved) == 1
    row = moved.iloc[0]
    assert (row["table"], row["row"], row["column"], row["status"]) == (
        "calls",
        "K=100",
        "mc_price",
        "moved",
    )
    assert row["n_stderr"] == pytest.approx(3.0, rel=1e-9)
    out = capsys.readouterr().out
    assert "1 number(s) changed" in out and "| calls | K=100 | mc_price | moved |" in out


def test_render_rebuilds_byte_identical(
    example_run: dict[str, Any], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    src = copy_output(example_run["out"], tmp_path / "out")
    before = {
        p.relative_to(src).as_posix(): p.read_bytes()
        for p in [*src.glob("tables/*.tex"), src / "study.tex", src / "study.md"]
    }
    figures = sorted(p.name for p in (src / "figures").iterdir())
    manifest = (src / "manifest.json").read_bytes()
    shutil.rmtree(src / "tables")
    shutil.rmtree(src / "figures")
    (src / "study.tex").unlink()
    (src / "study.md").unlink()
    (src / "figures").mkdir()
    (src / "figures" / "stale.png").write_bytes(b"x")  # a figure no spec declares
    (src / "study.pdf").write_bytes(b"%PDF stale")
    with caplog.at_level(logging.WARNING, logger="volsto.studies.runner"):
        assert runner.main(["render", str(src), "--no-latex-check"]) == 0
    assert "removed the stale study.pdf" in caplog.text
    assert not (src / "study.pdf").exists()
    after = {k: (src / k).read_bytes() for k in before}
    assert after == before
    assert sorted(p.name for p in (src / "figures").iterdir()) == figures
    assert (src / "manifest.json").read_bytes() == manifest
    assert_guard_clear()


def test_config_path_repo_relative_and_overrides(tmp_path: Path) -> None:
    example = ROOT / "tests" / "_example_study.yaml"
    cfg = runner.load_study_config(example)
    assert cfg.name == "example_bs_vanilla" and cfg.path("store") == ROOT / "outputs" / "store"
    assert runner.display_path(example) == "tests/_example_study.yaml"
    over = runner.load_study_config(
        example,
        ConfigOverrides(
            store=str(tmp_path / "s"),
            cache="cache",  # cwd-relative
            fast=True,
            sets=("n_paths=4000", "strikes=[90, 110]", " vol = 1e-1"),
        ),
    )
    assert over.store == str((tmp_path / "s").resolve())
    assert over.cache == runner.display_path(Path.cwd() / "cache")
    assert over.params["n_paths"] == 4000 and over.params["strikes"] == [90, 110]
    assert over.params["vol"] == 0.1 and isinstance(over.params["vol"], float)
    assert over.mode == "fast"
    nested = runner.apply_overrides(
        StudyConfig.from_mapping(example_mapping(params={"a": {"b": 1}})),
        ConfigOverrides(sets=("a.b=2.5",)),
    )
    assert nested.params == {"a": {"b": 2.5}}
    with pytest.raises(ConfigError, match="not an existing mapping"):
        runner.apply_overrides(nested, ConfigOverrides(sets=("x.y=1",)))
    with pytest.raises(ConfigError, match="unknown keys"):
        StudyConfig.from_mapping({**example_mapping(), "extra": 1})
    with pytest.raises(ConfigError, match="missing required keys"):
        StudyConfig.from_mapping({k: v for k, v in example_mapping().items() if k != "seeds"})
    with pytest.raises(ConfigError, match="mode"):
        StudyConfig.from_mapping(example_mapping(mode="slow"))
    with pytest.raises(ConfigError, match="seeds"):
        StudyConfig.from_mapping(example_mapping(seeds={"pricing": -1}))
    with pytest.raises(ConfigError, match="seeds"):
        StudyConfig.from_mapping(example_mapping(seeds={"pricing": 2**70}))
    module = runner.load_runner_module("_example_study")
    with pytest.raises(ConfigError, match="missing required params"):
        runner.validate_params(module, {k: 1 for k in list(EXAMPLE_PARAMS)[:-1]})
    with pytest.raises(ConfigError, match="unknown params"):
        runner.validate_params(module, {**EXAMPLE_PARAMS, "n_path": 1})
    with pytest.raises(ConfigError, match="strikes must not be empty"):
        runner.validate_params(module, {**EXAMPLE_PARAMS, "strikes": []})
    with pytest.raises(ConfigError, match="cannot be imported"):
        runner.load_runner_module("volsto.studies.catalogue.no_such_study")
    with pytest.raises(ConfigError, match="lacks"):
        runner.load_runner_module("os")
    ctx = StudyContext(cfg, out_dir=tmp_path)
    assert ctx.seed("pricing") == 2024
    with pytest.raises(ConfigError, match="no seed 'hedging'"):
        ctx.seed("hedging")


def test_strict_yaml(tmp_path: Path) -> None:
    """The verifier's p_cfg.py: duplicates, YAML 1.2 floats and booleans, empty --set, nan."""
    dup = tmp_path / "dup.yaml"
    text = (ROOT / "tests" / "_example_study.yaml").read_text()
    dup.write_text(text + "name: other\n")
    with pytest.raises(ConfigError, match="duplicate key 'name'"):
        runner.load_study_config(dup)
    nested_dup = tmp_path / "dup2.yaml"
    nested_dup.write_text(text.replace("  vol: 0.2\n", "  vol: 0.2\n  vol: 0.3\n"))
    with pytest.raises(ConfigError, match="duplicate key 'vol'"):
        runner.load_study_config(nested_dup)
    floats = tmp_path / "floats.yaml"
    floats.write_text(
        text.replace("  vol: 0.2\n", "  vol: 2e-1\n").replace("n_paths: 20000", "n_paths: 2e4")
    )
    cfg = runner.load_study_config(floats)
    assert cfg.params["vol"] == 0.2 and isinstance(cfg.params["vol"], float)
    assert cfg.params["n_paths"] == 20000.0 and isinstance(cfg.params["n_paths"], float)
    base = runner.load_study_config(ROOT / "tests" / "_example_study.yaml")

    def setp(*sets: str) -> dict[str, Any]:
        return dict(runner.apply_overrides(base, ConfigOverrides(sets=sets)).params)

    assert setp("vol=1e-3")["vol"] == 0.001
    assert setp("vol=.5")["vol"] == 0.5
    assert setp("vol=yes")["vol"] == "yes"  # YAML 1.2: not a boolean
    assert setp("vol=true")["vol"] is True
    assert setp("vol=null")["vol"] is None
    for bad, match in (
        ("n_paths=", "empty value"),
        ("n_paths= ", "empty value"),
        ("=3", "expected key.sub=value"),
        ("n_paths", "expected key.sub=value"),
        ("strikes=[1,2", "while parsing"),
        ("vol=.nan", "non-finite"),
        ("vol=nan", "non-finite"),
        ("vol=-inf", "non-finite"),
        ("vol=2024-01-01", "not plain YAML data"),
        ("strikes.0=5", "not an existing mapping"),
        ("a..b=1", "empty key segment"),
    ):
        with pytest.raises(ConfigError, match=match):
            setp(bad)
    for change, match in (
        ({"question": ""}, "question"),
        ({"question": "two\nlines"}, "question"),
        ({"name": "../evil"}, "name"),
        ({"grid": ""}, "grid"),
        ({"params": []}, "params"),
        ({"mode": "FAST"}, "mode"),
    ):
        data = base.to_mapping()
        data.update(change)
        with pytest.raises(ConfigError, match=match):
            StudyConfig.from_mapping(data)


def test_fast_mode_banner_and_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The runner (not the modules) marks a fast-mode study under its question in study.md and
    study.tex; a full-mode study carries no banner; the provenance block and ``list`` show the
    mode; render keeps the banner byte-identically (``test_render_rebuilds_byte_identical`` runs
    on the fast example)."""
    params = {**EXAMPLE_PARAMS, "n_paths": 2000}
    full = write_config(tmp_path / "full.yaml", example_mapping(mode="full", params=params))
    fast = write_config(tmp_path / "fast.yaml", example_mapping(params=params))
    for cfg, name in ((full, "full"), (fast, "fast")):
        out = tmp_path / f"out_{name}"
        argv = ["run", str(cfg), "--out", str(out), *roots_argv(tmp_path), "--no-latex-check"]
        assert runner.main(argv) == 0
        md = (out / "study.md").read_text()
        tex = (out / "study.tex").read_text()
        assert f"| mode | {name} |" in md
        assert (FAST_BANNER_MD in md) is (name == "fast")
        assert (FAST_BANNER_TEX in tex) is (name == "fast")
        assert ("FAST MODE" in md + tex) is (name == "fast")
        before = [(out / f).read_bytes() for f in ("study.md", "study.tex")]
        assert runner.main(["render", str(out), "--no-latex-check"]) == 0
        assert [(out / f).read_bytes() for f in ("study.md", "study.tex")] == before
    # --fast on a full config sets the mode (and so the banner)
    out = tmp_path / "out_forced"
    argv = [
        "run",
        str(full),
        "--fast",
        "--out",
        str(out),
        *roots_argv(tmp_path),
        "--no-latex-check",
    ]
    assert runner.main(argv) == 0
    assert FAST_BANNER_MD in (out / "study.md").read_text()
    capsys.readouterr()
    catalogue = tmp_path / "catalogue"
    catalogue.mkdir()
    shutil.copy(full, catalogue / "a_full.yaml")
    shutil.copy(fast, catalogue / "b_fast.yaml")
    monkeypatch.setattr(runner, "CATALOGUE_DIR", catalogue)
    assert runner.main(["list"]) == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 2
    assert "a_full.yaml: example_bs_vanilla [mode full] — " in lines[0]
    assert "b_fast.yaml: example_bs_vanilla [mode fast] — " in lines[1]


def test_list_catalogue(tmp_path: Path) -> None:
    write_config(tmp_path / "good.yaml", example_mapping(question="Is this one line?"))
    write_config(tmp_path / "bad.yaml", {**example_mapping(), "typo": 1})
    rows = {Path(r["path"]).name: r for r in runner.list_catalogue(tmp_path)}
    assert rows["good.yaml"]["question"] == "Is this one line?" and not rows["good.yaml"]["error"]
    assert "unknown keys ['typo']" in rows["bad.yaml"]["error"]


# --------------------------------------------------------------------------------------------
# requirements
# --------------------------------------------------------------------------------------------


def _toy_points() -> tuple[str, str]:
    points = enumerate_points(load_grid(TOY_GRID))
    lv = next(p for p in points if p.mode == "lv")
    cal = next(p for p in points if p.cache_key is not None)
    assert cal.cache_key == cal.id
    return lv.id, cal.id


@pytest.mark.parametrize(
    "kinds",
    [
        ("point",),
        ("leverage",),
        ("artefact",),
        ("offgrid",),
        ("point", "leverage", "artefact", "offgrid"),
    ],
)
def test_missing_requirements_exit_2(
    kinds: tuple[str, ...],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    lv_id, key = _toy_points()
    ids = {"point": lv_id, "leverage": key, "artefact": FAKE_ARTEFACT, "offgrid": FAKE_POINT}
    kind_of = {"point": "point", "leverage": "leverage", "artefact": "artefact", "offgrid": "point"}
    requires = [{"kind": kind_of[k], "id": ids[k], "what": f"the {k}"} for k in kinds]
    cfg = write_config(
        tmp_path / "ex.yaml",
        example_mapping(grid=str(TOY_GRID), params={**EXAMPLE_PARAMS, "requires": requires}),
    )

    def boom(ctx: StudyContext) -> Results:
        raise AssertionError("compute must not run when a requirement is missing")

    monkeypatch.setattr(_example_study, "compute", boom)
    out = tmp_path / "out"
    code = runner.main(["run", str(cfg), "--out", str(out), *roots_argv(tmp_path)])
    assert code == 2
    assert not out.exists() and staging_leftovers(tmp_path) == []
    for d in ("store", "cache", "outputs"):
        assert not (tmp_path / d).exists(), d
    printed = capsys.readouterr().out
    cwd = runner.cwd_path
    prefix = (
        f"volsto-precompute --grid {cwd(TOY_GRID)} --store {cwd(tmp_path / 'store')} "
        f"--cache {cwd(tmp_path / 'cache')}"
    )
    expected: list[str] = []
    inside = [ids[k] for k in kinds if k in ("point", "leverage")]
    if inside:
        expected.append(f"{prefix} --only {' '.join(inside)} --resume")
    if "offgrid" in kinds:
        expected.append(
            f"# {FAKE_POINT} is not a point of {cwd(TOY_GRID)}: add it to that grid YAML first, "
            f"then run: {prefix} --only {FAKE_POINT} --resume"
        )
    if "artefact" in kinds:
        expected.append(_example_study.ARTEFACT_COMMAND)
    lines = printed.splitlines()
    start = lines.index("produce them with:")
    assert [ln.strip() for ln in lines[start + 1 :]] == expected
    assert lines[0] == f"{len(kinds)} requirement(s) missing:"
    for k in kinds:
        assert f"  - {kind_of[k]} {ids[k]}: the {k}" in lines
    assert "<grid>" not in printed
    if len(kinds) > 1:  # the printed line runs as printed, from this directory
        exe = Path(sys.executable).with_name("volsto-precompute")
        argv = shlex.split(expected[0])
        t0 = time.perf_counter()
        proc = subprocess.run(
            [str(exe), *argv[1:], "--dry-run"],
            capture_output=True,
            text=True,
            timeout=300,
            env={**os.environ, "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS", "2")},
        )
        print(
            f"dry run of the printed line: rc {proc.returncode}, {time.perf_counter() - t0:.1f} s"
        )
        assert proc.returncode == 0, proc.stdout[-1500:] + proc.stderr[-1500:]
        assert "dry run: nothing computed" in proc.stdout
        assert not (tmp_path / "store").exists()


def test_requirement_commands_without_grid_and_recorded_grid_note(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # a point requirement needs a grid: a config error (exit 1), nothing computed or written
    requires = [{"kind": "point", "id": FAKE_POINT, "what": "p"}]
    cfg = write_config(
        tmp_path / "p.yaml", example_mapping(params={**EXAMPLE_PARAMS, "requires": requires})
    )
    assert runner.main(["run", str(cfg), "--out", str(tmp_path / "o1"), *roots_argv(tmp_path)]) == 1
    err = capsys.readouterr().err
    assert "needs a grid" in err and not (tmp_path / "o1").exists()
    # a missing leverage without a grid: a note, no runnable-looking line, exit 2
    requires = [{"kind": "leverage", "id": FAKE_KEY, "what": "lev"}]
    cfg = write_config(
        tmp_path / "l.yaml", example_mapping(params={**EXAMPLE_PARAMS, "requires": requires})
    )
    assert runner.main(["run", str(cfg), "--out", str(tmp_path / "o2"), *roots_argv(tmp_path)]) == 2
    out = capsys.readouterr().out
    assert f"# leverage {FAKE_KEY} (lev): the study has no grid (grid: null)" in out
    assert not any(ln.strip().startswith("volsto-precompute") for ln in out.splitlines())
    assert "<grid>" not in out
    # a store built from another grid: the configured grid is kept and a note says so
    store = tmp_path / "store_other"
    (store / "results" / "runs").mkdir(parents=True)
    (store / "results" / "runs" / "r.json").write_text(
        json.dumps({"grid_path": "configs/grids/default.yaml"})
    )
    cfg_ctx = runner.apply_overrides(
        StudyConfig.from_mapping(example_mapping()),
        ConfigOverrides(grid=str(TOY_GRID), store=str(store), cache=str(tmp_path / "c")),
    )
    ctx = StudyContext(cfg_ctx, out_dir=tmp_path / "o3")
    lv_id, _ = _toy_points()
    cmds = ctx.commands_for([ctx.point_requirement(lv_id, "lv")])
    assert cmds[0].startswith(f"volsto-precompute --grid {runner.cwd_path(TOY_GRID)} ")
    assert cmds[1].startswith(
        f"# note: the store {runner.cwd_path(store)} was built from grid "
        "configs/grids/default.yaml"
    )


def test_artefact_requirement_present(tmp_path: Path) -> None:
    requires = [{"kind": "artefact", "id": FAKE_ARTEFACT, "what": "table D"}]
    art = tmp_path / "outputs" / "m8b" / "m8b_table_D.csv"
    art.parent.mkdir(parents=True)
    art.write_text("a,b\n1,2\n")
    cfg = StudyConfig.from_mapping(
        example_mapping(
            params={**EXAMPLE_PARAMS, "requires": requires, "n_paths": 2000},
            outputs=str(tmp_path / "outputs"),
            store=str(tmp_path / "store"),
            cache=str(tmp_path / "cache"),
        )
    )
    run = runner.run_study(cfg, out_dir=tmp_path / "out", latex_check=False)
    assert run.exit_code == 0
    assert run.manifest["artefacts"] == [{"path": FAKE_ARTEFACT, "sha256": runner.file_sha256(art)}]
    ctx = StudyContext(cfg, out_dir=tmp_path)
    assert ctx.artefact("m8b/m8b_table_D.csv") == art
    with pytest.raises(ConfigError, match="relative path"):
        ctx.artefact("../escape.csv")


def test_toy_store_requirements(toy_build: Any, tmp_path: Path) -> None:
    toy = toy_build.require()
    cache_before = {
        p.relative_to(toy.cache_root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in toy.cache_root.rglob("*")
        if p.is_file()
    }
    cfg = runner.apply_overrides(
        StudyConfig.from_mapping(example_mapping()),
        ConfigOverrides(
            grid=str(toy.grid_path),
            store=str(toy.store_root),
            cache=str(toy.cache_root),
            outputs=str(toy.outputs_root),
        ),
    )
    assert cfg.grid == "configs/grids/toy.yaml"  # repo-relative
    ctx = StudyContext(cfg, out_dir=tmp_path / "out")
    points = ctx.grid_points()
    calibrated = [p for p in points if p.cache_key is not None]
    assert len(points) == 4 and len(calibrated) == 3
    present = [ctx.point_requirement(p.id, p.label) for p in points]
    present += [ctx.leverage_requirement(p.spec, p.label) for p in calibrated]
    assert ctx.missing(present) == []
    fake = [
        ctx.point_requirement(FAKE_POINT, "made-up LV point"),
        ctx.leverage_requirement(FAKE_KEY, "made-up leverage"),
    ]
    assert ctx.missing([*present, *fake]) == fake
    commands = ctx.commands_for(fake)
    cwd = runner.cwd_path
    prefix = (
        f"volsto-precompute --grid {cwd(toy.grid_path)} --store {cwd(toy.store_root)} "
        f"--cache {cwd(toy.cache_root)}"
    )
    print("\n".join(commands))
    assert commands == [
        f"# {fid} is not a point of {cwd(toy.grid_path)}: add it to that grid YAML first, "
        f"then run: {prefix} --only {fid} --resume"
        for fid in (FAKE_POINT, FAKE_KEY)
    ]  # the store recorded the same grid: no note
    real_missing = [
        runner.Requirement("point", calibrated[0].id, "a", ""),
        runner.Requirement("leverage", calibrated[1].cache_key or "", "b", ""),
    ]
    assert ctx.commands_for(real_missing) == [
        f"{prefix} --only {calibrated[0].id} {calibrated[1].id} --resume"
    ]
    with pytest.raises(MissingRequirements):
        ctx.require([*present, *fake])
    ctx.require(present)
    assert list(ctx.store_points) == [p.id for p in points]
    assert {k["key"] for k in ctx.cache_keys.values()} == {p.cache_key for p in calibrated}
    assert {k["n_particles"] for k in ctx.cache_keys.values()} == {20_000}
    # a toy leverage loads without calibrating; the store accessors record what they return
    fresh = StudyContext(cfg, out_dir=tmp_path / "out2")
    with runner.calibration_forbidden():
        model = fresh.leverage(calibrated[0].spec, "toy 1F")
        products = fresh.store().products(calibrated[1].id)
    assert model is not None
    assert fresh.cache_keys[calibrated[0].id]["what"] == "toy 1F"
    assert not products.empty
    assert list(fresh.store_points) == [calibrated[1].id]
    keys = fresh.store_point_keys()
    assert [(k["key"], k["n_particles"]) for k in keys] == [(calibrated[1].cache_key, 20_000)]
    cache_after = {
        p.relative_to(toy.cache_root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in toy.cache_root.rglob("*")
        if p.is_file()
    }
    assert cache_after == cache_before
    assert_guard_clear()


# --------------------------------------------------------------------------------------------
# the no-calibration guarantee
# --------------------------------------------------------------------------------------------


def _attack_run(tmp_path: Path, attack: str, **changes: Any) -> tuple[int, Path]:
    cfg = write_config(
        tmp_path / f"{attack}.yaml",
        example_mapping(params={**EXAMPLE_PARAMS, "n_paths": 2000, "attack": attack}, **changes),
    )
    out = tmp_path / f"out_{attack}"
    code = runner.main(
        ["run", str(cfg), "--out", str(out), *roots_argv(tmp_path), "--no-latex-check"]
    )
    return code, out


@pytest.mark.parametrize("attack", ["bypass_cache", "process_pool", "thread_pool"])
def test_guard_refuses_calibration_routes(
    attack: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A study calibrating through its own ``LeverageCache``, a thread pool, or a spawned process
    pool (the child inherits ``VOLSTO_FORBID_CALIBRATION``) is refused: exit 1, one error line,
    nothing written."""
    t0 = time.perf_counter()
    code, out = _attack_run(tmp_path, attack)
    print(f"{attack}: exit {code}, {time.perf_counter() - t0:.1f} s")
    err = capsys.readouterr().err
    assert code == 1
    assert "error: CalibrationForbiddenError: leverage calibration refused" in err
    if attack == "process_pool":
        assert f"{guard.ENV_VAR}='1' is set" in err  # refused in the child, by the variable
    assert "Traceback" not in err
    assert not out.exists() and staging_leftovers(tmp_path) == []
    assert not (tmp_path / "cache").exists()
    assert_guard_clear()


def test_guard_counts_a_caught_refusal_and_a_late_thread(tmp_path: Path) -> None:
    code, out = _attack_run(tmp_path, "swallow")
    assert code == 1
    m = json.loads((out / "manifest.json").read_text())
    assert m["calibration_refusals"] == 1 and m["recalibrated"] is False
    code, out = _attack_run(tmp_path, "late_calibrate_in_render")
    assert _example_study.LATE["log"] == ["late calibrate: CalibrationForbiddenError"]
    assert code == 1
    assert json.loads((out / "manifest.json").read_text())["calibration_refusals"] == 1
    assert_guard_clear()


WRITER_SCRIPT = """
import sys, time
from pathlib import Path
sys.path.insert(0, {tests!r})
import _example_study as ex
from volsto.calibration.cache import LeverageCache
directory, root = Path(sys.argv[1]), sys.argv[2]
t0 = time.monotonic()
while not (directory / "go").exists():
    if time.monotonic() - t0 > 120:
        sys.exit(3)
    time.sleep(0.05)
spec = ex.tiny_spec(seed_shift=31)
LeverageCache(root).store(spec, ex.flat_leverage(spec))
(directory / "done").write_text("done")
"""


def test_cache_changes_are_reported_not_blamed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """Growth of the cache root during a run — another process writing (the study-C refit), a
    thread, the study's own ``LeverageCache.store`` — is recorded as "cache changed during the
    run, not attributed" with a warning; it does not make ``recalibrated`` true nor fail."""
    writer_dir = tmp_path / "writer"
    writer_dir.mkdir()
    monkeypatch.setenv(_example_study.WRITER_ENV, str(writer_dir))
    script = WRITER_SCRIPT.format(tests=str(ROOT / "tests"))
    writer = subprocess.Popen(
        [sys.executable, "-c", script, str(writer_dir), str(tmp_path / "cache")],
        env={k: v for k, v in os.environ.items() if k not in (guard.ENV_VAR, guard.MARKER_ENV)},
    )
    try:
        with caplog.at_level(logging.WARNING, logger="volsto.studies.runner"):
            code, out = _attack_run(tmp_path, "wait_for_writer")
    finally:
        assert writer.wait(timeout=300) == 0
    assert code == 0
    assert "changed during the run" in caplog.text and "not attributed" in caplog.text
    m = json.loads((out / "manifest.json").read_text())
    key = spec_key(_example_study.tiny_spec(seed_shift=31))
    assert m["recalibrated"] is False and m["calibrations_started"] == []
    assert m["cache_changed_during_run"] is True
    obs = m["cache_observation"]
    assert obs["new_keys"] == [key] and obs["new_keys_complete"] == [key]
    assert obs["manifest_rows_added"] == 1 and obs["attribution"].startswith("not attributed")
    assert (
        obs["stages"]["before"]["entries"] == 0 and obs["stages"]["after_compute"]["entries"] == 1
    )
    md = (out / "study.md").read_text()
    assert "| recalibrated | no |" in md
    assert (
        "| cache changed during the run | 1 new, 0 rewritten entries, 1 manifest rows; "
        f"not attributed ({key[:12]}) |"
    ) in md
    # the study's own write into the configured root, and a thread writing during render
    shutil.rmtree(tmp_path / "cache")
    code, out = _attack_run(tmp_path, "write_root")
    m = json.loads((out / "manifest.json").read_text())
    assert code == 0 and m["recalibrated"] is False and m["cache_changed_during_run"] is True
    assert m["cache_observation"]["stages"]["after_compute"]["entries"] == 1
    shutil.rmtree(tmp_path / "cache")
    code, out = _attack_run(tmp_path, "late_write_in_render")
    assert _example_study.LATE["log"] == ["late write: done"]
    m = json.loads((out / "manifest.json").read_text())
    stages = m["cache_observation"]["stages"]
    assert code == 0 and m["cache_changed_during_run"] is True
    assert stages["after_compute"]["entries"] == 0 and stages["after_render"]["entries"] == 1
    assert_guard_clear()


def test_a_calibration_that_escaped_the_prohibition_is_recalibrated(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A child process whose environment dropped VOLSTO_FORBID_CALIBRATION (but kept the run's
    marker directory) starts a calibration: the marker makes ``recalibrated`` true, exit 1."""
    with caplog.at_level(logging.ERROR, logger="volsto.studies.runner"):
        code, out = _attack_run(tmp_path, "escaped_child")
    assert code == 1
    assert "RECALIBRATED" in caplog.text
    m = json.loads((out / "manifest.json").read_text())
    assert m["recalibrated"] is True and m["cache_changed_during_run"] is False
    started = m["calibrations_started"]
    assert len(started) == 1 and started[0]["pid"] != os.getpid()
    assert started[0]["kind"] == "started" and started[0]["what"] == "calibrate_leverage"
    assert "| recalibrated | YES: 1 calibration(s) started (pid" in (out / "study.md").read_text()
    assert_guard_clear()


def test_ctx_cache_hits_are_recorded_and_misses_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _example_study.tiny_spec()
    LeverageCache(tmp_path / "cache").store(spec, _example_study.flat_leverage(spec))
    code, out = _attack_run(tmp_path, "ctx_cache_hit")
    assert code == 0
    m = json.loads((out / "manifest.json").read_text())
    assert m["recalibrated"] is False
    assert [(k["key"], k["n_particles"]) for k in m["cache_keys"]] == [(spec_key(spec), 5000)]
    assert m["cache_keys"][0]["what"].startswith("ctx.cache (nu=1.5 theta=0")
    assert m["particles"] == [5000]
    # an undeclared miss inside compute: exit 2, the note, no output directory
    capsys.readouterr()
    code, out = _attack_run(tmp_path, "undeclared_miss")
    assert code == 2
    printed = capsys.readouterr().out
    missed = spec_key(_example_study.tiny_spec(seed_shift=23))
    assert f"  - leverage {missed}: undeclared tiny leverage" in printed
    assert "the study has no grid" in printed
    assert not out.exists() and staging_leftovers(tmp_path) == []
    assert_guard_clear()


def test_guard_every_reference_nesting_threads_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from volsto.calibration import particle

    monkeypatch.delenv(guard.ENV_VAR, raising=False)
    held = {"dict": particle.calibrate_leverage}
    partial = functools.partial(particle.calibrate_leverage, None)
    refs = (
        lambda: _example_study.direct_calibration(),  # alias bound at import
        lambda: held["dict"](None, None, None, None),
        lambda: partial(None, None, None),
    )
    before = guard.refusals()
    for _ in range(3):  # entered repeatedly, nested
        with runner.calibration_forbidden():
            assert os.environ[guard.ENV_VAR] == "1"
            with runner.calibration_forbidden():
                for ref in refs:
                    with pytest.raises(CalibrationForbiddenError):
                        ref()
            for ref in refs:
                with pytest.raises(CalibrationForbiddenError):
                    ref()
        assert_guard_clear()
        for ref in refs:  # outside the guard the call proceeds (and fails on its arguments)
            with pytest.raises((AttributeError, TypeError)):
                ref()
    assert guard.refusals() - before == 18
    # a child started inside the block inherits the prohibition
    code = (
        "from volsto.calibration.particle import calibrate_leverage\n"
        "try:\n calibrate_leverage(None, None, None, None)\n"
        "except Exception as e: print(type(e).__name__)\n"
    )
    with runner.calibration_forbidden():
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
        )
    assert proc.stdout.strip() == "CalibrationForbiddenError", proc.stderr[-1000:]
    # two threads: B stays guarded after A leaves
    a_in, a_out, seen = threading.Event(), threading.Event(), {}

    def a() -> None:
        with runner.calibration_forbidden():
            a_in.set()
            a_out.wait(30)

    def b() -> None:
        a_in.wait(30)
        with runner.calibration_forbidden():
            a_out.set()
            time.sleep(0.2)
            seen["after_a"] = guard.calibration_is_forbidden()

    ta, tb = threading.Thread(target=a), threading.Thread(target=b)
    ta.start()
    tb.start()
    ta.join(30)
    tb.join(30)
    assert seen["after_a"] is True
    assert_guard_clear()
    # a variable the user exported survives a block and forbids on its own
    monkeypatch.setenv(guard.ENV_VAR, "1")
    with runner.calibration_forbidden():
        pass
    assert os.environ[guard.ENV_VAR] == "1"
    with pytest.raises(CalibrationForbiddenError, match="VOLSTO_FORBID_CALIBRATION"):
        refs[1]()
    monkeypatch.setenv(guard.ENV_VAR, "0")
    assert not guard.calibration_is_forbidden()


def _names(node: ast.AST) -> set[str]:
    out: set[str] = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name):
            out.add(n.id)
        elif isinstance(n, ast.Attribute):
            out.add(n.attr)
    return out


def test_particle_kernel_reached_only_through_calibrate_leverage() -> None:
    """The walking test of the guard's premise: the particle calibration runs only inside
    ``calibrate_leverage``, whose first statement is the guard check."""
    pkg = ROOT / "volsto"
    particle_path = pkg / "calibration" / "particle.py"
    tree = ast.parse(particle_path.read_text())
    funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    cal = funcs["calibrate_leverage"]
    body = (
        cal.body[1:]
        if isinstance(cal.body[0], ast.Expr) and isinstance(cal.body[0].value, ast.Constant)
        else cal.body
    )
    first = body[0]
    assert (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Call)
        and isinstance(first.value.func, ast.Name)
        and first.value.func.id == "check_calibration_allowed"
    ), "the guard check must be the first statement of calibrate_leverage"
    # kernel_regression is a pure regression (reused by risk/profiles.py): it steps nothing
    reusable = {"kernel_regression"}
    assert "step_lsv_block" not in _names(funcs["kernel_regression"])
    helpers = set(funcs) - {"calibrate_leverage"} - reusable
    # _sorted_regression and _finish_estimate are the two stages conditional_variance_estimate
    # was split into (binned-estimator change): pure regressions, checked below like the others
    assert helpers == {
        "conditional_variance_estimate",
        "_sorted_regression",
        "_finish_estimate",
        "leverage_grid_config",
    }
    # inside particle.py: the particle step and the helpers are used by calibrate_leverage only
    for name, fn in funcs.items():
        if name == "calibrate_leverage":
            continue
        used = _names(fn)
        assert "step_lsv_block" not in used, name
        assert not (used & helpers) or name in helpers, (name, used & helpers)
    assert "step_lsv_block" in _names(cal)
    # elsewhere in volsto/: only these names come from particle.py; the particle step is used by
    # the pricing module and particle.py only; the helpers are not referenced at all
    allowed_imports = {"calibrate_leverage", "CALIBRATION_CODE_TAG", "CalibrationResult"} | reusable
    step_users = set()
    for path in sorted(pkg.rglob("*.py")):
        if path == particle_path:
            continue
        mod = ast.parse(path.read_text())
        rel = path.relative_to(ROOT).as_posix()
        for n in ast.walk(mod):
            if isinstance(n, ast.ImportFrom) and n.module == "volsto.calibration.particle":
                bad = {a.name for a in n.names} - allowed_imports
                assert not bad, (rel, bad)
        used = _names(mod)
        assert not (used & helpers), (rel, used & helpers)
        if "step_lsv_block" in used:
            step_users.add(rel)
    assert step_users == {"volsto/models/lsv.py"}
    # the guard module imports nothing from volsto (no cycle with particle.py)
    guard_tree = ast.parse((pkg / "calibration" / "guard.py").read_text())
    for n in ast.walk(guard_tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            mods = [n.module or ""] if isinstance(n, ast.ImportFrom) else [a.name for a in n.names]
            assert not any(m.startswith("volsto") for m in mods), mods


def test_read_only_cache_refuses_to_write(tmp_path: Path) -> None:
    spec = _example_study.tiny_spec()
    root = tmp_path / "cache"
    hits: list[str] = []
    ro = ReadOnlyLeverageCache(root, on_hit=lambda key, s: hits.append(key))
    with runner.calibration_forbidden(), pytest.raises(CacheMissError):
        ro.get_or_calibrate(spec, allow_calibrate=True)  # forced to False
    for call in (
        lambda: ro.store(spec, None, None),  # type: ignore[arg-type]
        lambda: ro.calibrate(spec),
        lambda: ro.put(spec),
        lambda: ro.write_report(spec, None),  # type: ignore[arg-type]
        lambda: ro.merge_manifest(pd.DataFrame({"key": ["x"]})),
        lambda: ro._update_manifest(lambda df: df),
    ):
        with pytest.raises(CalibrationForbiddenError, match="never calibrates"):
            call()
    assert not root.exists() and hits == []
    assert isinstance(ro, LeverageCache)


# --------------------------------------------------------------------------------------------
# errors
# --------------------------------------------------------------------------------------------

BAD_MODULE = """
from volsto.studies.results import ResultsBuilder
TITLE = "bad"
QUESTION = "bad?"
REQUIRED_PARAMS = ()
def requirements(ctx):
    return []
def compute(ctx):
    b = ResultsBuilder()
    b.add("t", "r", "c", 1.0, None, unit="u", source="computed")
    return b.build()
def tables(r):
    return []
def figures(r):
    return []
def narrative(r):
    return "```\\n\\\\end{verbatim}\\n```"
"""


def test_unexpected_errors_are_one_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    (tmp_path / "bad_results_study.py").write_text(BAD_MODULE)
    monkeypatch.syspath_prepend(str(tmp_path))
    cfg = write_config(
        tmp_path / "bad.yaml", example_mapping(runner="bad_results_study", params={})
    )
    argv = ["run", str(cfg), "--out", str(tmp_path / "out"), *roots_argv(tmp_path)]
    with caplog.at_level(logging.INFO, logger="volsto"):
        assert runner.main(argv) == 1
    err = capsys.readouterr().err
    assert [ln for ln in err.splitlines() if ln.strip()] == [
        "error: ResultsError: 1 results invariant violation(s):"
    ]
    assert not any(r.exc_info for r in caplog.records)
    assert "volsto-study run: exit 1" in caplog.text and "recalibrated: no" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="volsto"):
        assert runner.main([*argv, "-v"]) == 1
    assert any(r.exc_info for r in caplog.records)  # the traceback with -v
    assert not (tmp_path / "out").exists() and staging_leftovers(tmp_path) == []


# --------------------------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "stderr", "expected"),
    [
        (21.4437, 0.0312, r"$21.44 \pm 0.03$"),  # leading 3: one digit
        (21.4437, 0.0152, r"$21.444 \pm 0.015$"),  # leading 1: two digits
        (1.2345, 0.0251, r"$1.235 \pm 0.025$"),  # leading 2: two digits, half up
        (0.8393, 0.00917, r"$0.839 \pm 0.009$"),  # leading 9: one digit
        (0.123, 0.096, r"$0.12 \pm 0.10$"),  # carries to 0.1: leads with 1, two digits
        (0.96, 0.96, r"$1.0 \pm 1.0$"),
        (5.0, 0.00995, r"$5.000 \pm 0.010$"),
        (1234.5, 35.0, r"$1230 \pm 40$"),  # the tens place
        (98765.0, 1234.0, r"$98800 \pm 1200$"),
        (-0.0004, 0.003, r"$0.000 \pm 0.003$"),  # a rounded zero is unsigned
        (-1.23456, 0.0456, r"$-1.23 \pm 0.05$"),
        (5.0, 0.0, r"$5 \pm 0$"),
        (1.23456e-9, 2.5e-11, r"$(1.235 \pm 0.025)\times 10^{-9}$"),  # shared exponent
        (1e12, 3e9, r"$(1.000 \pm 0.003)\times 10^{12}$"),
        (1e20, 1e18, r"$(1.000 \pm 0.010)\times 10^{20}$"),
        (3.2e6, 0.5e6, r"$3200000 \pm 500000$"),  # |exponent| 6: plain
        (float("nan"), 0.1, "--"),
        (float("nan"), float("nan"), "--"),
    ],
)
def test_format_value_rounding(value: float, stderr: float, expected: str) -> None:
    assert latex.format_value(value, stderr) == expected
    text = latex.format_value_text(value, stderr)
    plain = (
        expected.strip("$")
        .replace(r" \pm ", " ± ")
        .replace(r")\times 10^{", ")\N{MULTIPLICATION SIGN}10^")
    )
    assert text == (plain[:-1] if plain.endswith("}") else plain)


def test_format_value_never_raises() -> None:
    """The verifier's p_round.py sweep, widened: any finite value with any finite non-negative
    stderr formats (the decimal precision is sized from the exponents)."""
    rng = random.Random(7)
    extremes = [0.0, 5e-324, 1e-310, 1e-300, 1e-20, 1.0, 1e20, 1e300, 1.7976931348623157e308]
    cases = [(v, s) for v in extremes for s in extremes] + [
        (-v, s) for v in extremes for s in extremes
    ]
    for _ in range(3000):
        v = rng.choice((-1, 1)) * 10 ** rng.uniform(-320, 308)
        s = 10 ** rng.uniform(-323, 308)
        cases.append((v, s))
    for v, s in cases:
        out = latex.format_value(v, s)
        assert out.startswith("$") and out.endswith("$"), (v, s, out)
        latex.format_value_text(v, s)
    assert latex.format_value(1e300, 1e-300).startswith(r"$(1.000")


def test_format_value_errors_and_exact() -> None:
    with pytest.raises(ValueError, match="format_exact"):
        latex.format_value(1.0, float("nan"))
    with pytest.raises(ValueError):
        latex.format_value(1.0, -0.1)
    with pytest.raises(ValueError):
        latex.format_value(float("inf"), 0.1)
    assert latex.format_value(1.0, 0.25, unit_in_header=False, unit="vol pts") == (
        r"$1.00 \pm 0.25$~vol pts"
    )
    assert latex.format_value(1.0, 0.25, unit_in_header=False, unit=DIMENSIONLESS) == (
        r"$1.00 \pm 0.25$"
    )
    assert latex.format_exact(0.245, 4) == "$0.245$"
    assert latex.format_exact(21.456789, 4) == "$21.46$"
    assert latex.format_exact(800000.0, 4) == "$800000$"
    assert latex.format_exact(-3.0, 4) == "$-3$"
    assert latex.format_exact(1.5e-5, 3) == r"$1.5\times 10^{-5}$"
    assert latex.format_exact(1e100, 4) == r"$1\times 10^{100}$"
    assert latex.format_exact(float("nan"), 4) == "--"
    assert latex.format_exact_text(0.2456, 2) == "0.25"
    assert latex.stderr_places(0.0312) == 2 and latex.stderr_places(35.0) == -1


def _small_results() -> Results:
    b = ResultsBuilder()
    b.add("t", "1F ω=1 & 50%_x", "vol", 21.44, 0.03, unit="vol pts", source="store:abc")
    b.add("t", "1F ω=1 & 50%_x", "p", 0.25, 0.012, unit=DIMENSIONLESS, source="computed")
    b.add("t", "LV", "vol", 22.0, 0.05, unit="vol pts", source="cache:k;computed")
    b.add_exact("t", "LV", "nu", 0.0, unit="", source="computed", axes={"nu": 0.0})
    return b.build()


def test_latex_table_and_escape() -> None:
    res = _small_results()
    spec = TableSpec(
        name="headline",
        caption="Forward vol & P(KO) at ρ = −0.7 ($x$) ✓ ≲ ⇒",  # noqa: RUF001
        table="t",
        columns=(
            Column("vol", "fwd ATM vol"),
            Column("p", "P(KO)"),
            Column("nu", "\N{GREEK SMALL LETTER NU}", digits=3),
        ),
        row_header="model",
    )
    tex = latex.latex_table(spec, res)
    assert r"\label{tab:headline}" in tex and r"\toprule" in tex and r"\bottomrule" in tex
    assert (
        r"\caption{Forward vol \& P(KO) at \ensuremath{\rho} = \ensuremath{-}0.7 (\$x\$) "
        r"\ensuremath{\checkmark} \ensuremath{\lesssim} \ensuremath{\Rightarrow}}"
    ) in tex
    assert r"model & fwd ATM vol [vol pts] & P(KO) & \ensuremath{\nu} \\" in tex
    assert (
        r"1F \ensuremath{\omega}=1 \& 50\%\_x & $21.44 \pm 0.03$ & $0.250 \pm 0.012$ & -- \\" in tex
    )
    assert r"LV & $22.00 \pm 0.05$ & -- & $0$ \\" in tex
    assert r"\begin{tabular}{lrrr}" in tex and "longtable" not in tex
    md = latex.markdown_table(spec, res)
    assert "| 1F ω=1 & 50%_x | 21.44 ± 0.03 | 0.250 ± 0.012 | -- |" in md
    assert "| model | fwd ATM vol [vol pts] | P(KO) | ν |" in md  # noqa: RUF001
    with pytest.raises(KeyError, match="rows"):
        latex.latex_table(TableSpec("x", "c", "t", spec.columns, rows=("nope",)), res)
    with pytest.raises(KeyError, match="columns"):
        latex.latex_table(TableSpec("x", "c", "t", (Column("nope", "n"),)), res)
    with pytest.raises(ValueError, match="must match"):
        TableSpec("bad name", "c", "t", spec.columns)
    assert (
        latex.latex_escape("a\\b~c^d") == r"a\textbackslash{}b\textasciitilde{}c\textasciicircum{}d"
    )


def test_split_table() -> None:
    b = ResultsBuilder()
    for i in range(7):
        b.add("t", f"r{i}", "v", float(i), 0.1, unit="u", source="computed", axes={"g": i // 3})
    b.add("t", "orphan", "v", 9.0, 0.1, unit="u", source="computed")
    res = b.build()
    spec = TableSpec("t", "cap", "t", (Column("v", "v"),))
    assert latex.split_table(spec, res, 10) == [spec]
    parts = latex.split_table(spec, res, 3)
    assert [p.name for p in parts] == ["t-part1", "t-part2", "t-part3"]
    assert [p.rows for p in parts] == [("r0", "r1", "r2"), ("r3", "r4", "r5"), ("r6", "orphan")]
    assert [p.caption for p in parts] == [
        "cap (part 1 of 3)",
        "cap (part 2 of 3)",
        "cap (part 3 of 3)",
    ]
    grouped = latex.split_table(spec, res, 2, by="g")
    assert [p.name for p in grouped] == [
        "t-0-part1",
        "t-0-part2",
        "t-1-part1",
        "t-1-part2",
        "t-2",
        "t-none",
    ]
    assert grouped[0].caption == "cap (g = 0) (part 1 of 2)" and grouped[4].rows == ("r6",)
    assert grouped[5].caption == "cap (g = none)" and grouped[5].rows == ("orphan",)
    by_fn = latex.split_table(spec, res, 99, by=lambda row: "odd" if row[-1] in "13579" else "even")
    assert [(p.name, p.caption) for p in by_fn] == [
        ("t-even", "cap (even)"),
        ("t-odd", "cap (odd)"),
    ]
    ordered = latex.split_table(TableSpec("t", "c", "t", spec.columns, rows=("r6", "r0")), res, 1)
    assert [p.rows for p in ordered] == [("r6",), ("r0",)]
    with pytest.raises(ValueError, match="max_rows"):
        latex.split_table(spec, res, 0)
    flat, names = latex.expand_tables([spec], res, 3)
    assert names == {"t": ["t-part1", "t-part2", "t-part3"]} and len(flat) == 3
    md = latex.study_markdown(
        title="T", question="Q?", narrative="{{table:t}}", tables=[spec], figures=[],
        results=res, provenance=[], max_rows=3,
    )  # fmt: skip
    assert md.index("`t-part1`") < md.index("`t-part3`")


def test_narrative_placeholders_and_markdown() -> None:
    blocks = latex.narrative_blocks("intro\n{{table:a}}\nmiddle\n", ["a", "b"], ["f"])
    assert blocks == [
        ("markdown", "intro"),
        ("table", "a"),
        ("markdown", "middle"),
        ("table", "b"),
        ("figure", "f"),
    ]
    with pytest.raises(ValueError, match="unknown table"):
        latex.narrative_blocks("{{table:zz}}", ["a"], [])
    with pytest.raises(ValueError, match="own line"):
        latex.narrative_blocks("see {{table:a}} here", ["a"], [])
    with pytest.raises(ValueError, match="twice"):
        latex.narrative_blocks("{{table:a}}\n{{table:a}}", ["a"], [])
    body = latex.markdown_to_latex(
        "# Head\n\nSome **bold** and *it* with `co_de` and \\(\\nu^2\\) 5%, a lone $ and $x$.\n\n"
        "- a\n- b\n\n1. one\n\n> quoted\n\n```\nraw_{x}\n```\n\n[link](https://example.org/a_b)\n"
    )
    assert r"\section*{Head}" in body
    assert (
        r"Some \textbf{bold} and \emph{it} with \texttt{co\_de} and \(\nu^2\) 5\%, "
        r"a lone \$ and \$x\$."
    ) in body
    assert "\\begin{itemize}\n\\item a\n\\item b\n\\end{itemize}" in body
    assert r"\begin{enumerate}" in body and r"\begin{quote}" in body
    assert "\\begin{verbatim}\nraw_{x}\n\\end{verbatim}" in body
    assert r"\href{https://example.org/a_b}{link}" in body
    with pytest.raises(ValueError, match="end\\{verbatim\\}"):
        latex.markdown_to_latex("```\nsneaky \\end{verbatim} \\input{/etc/passwd}\n```\n")


# --------------------------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------------------------


def test_results_invariants(tmp_path: Path) -> None:
    def build(*rows: dict[str, Any]) -> Results:
        b = ResultsBuilder()
        for r in rows:
            r = dict(r)
            b.add(r.pop("table", "t"), r.pop("row", "r"), r.pop("column", "c"), **r)
        return b.build()

    ok = {"value": 1.0, "stderr": 0.1, "unit": "u", "source": "computed"}
    with pytest.raises(ResultsError, match="without a finite stderr"):
        build({**ok, "stderr": None})
    with pytest.raises(ResultsError, match="without a finite stderr"):
        build({**ok, "stderr": float("nan")})
    with pytest.raises(ResultsError, match="without a finite stderr"):
        build({**ok, "stderr": float("inf")})
    with pytest.raises(ResultsError, match="negative stderr"):
        build({**ok, "stderr": -0.1})
    with pytest.raises(ResultsError, match="infinite value"):
        build({**ok, "value": float("inf")})
    with pytest.raises(ResultsError, match="duplicated"):
        build(ok, ok)
    with pytest.raises(ResultsError, match="without a unit"):
        build({**ok, "unit": ""})
    with pytest.raises(ResultsError, match="source"):
        build({**ok, "source": "somewhere"})
    with pytest.raises(ResultsError, match="source"):
        build({**ok, "source": "store:"})
    with pytest.raises(ResultsError, match="exact value with stderr"):
        build({**ok, "exact": True})
    with pytest.raises(ResultsError, match="empty row"):
        build({**ok, "row": ""})
    with pytest.raises(ResultsError, match="missing number has no stderr"):
        build({**ok, "value": float("nan")})
    with pytest.raises(ResultsError, match="scalar"):
        build({**ok, "axes": {"nu": [1, 2]}})
    with pytest.raises(ResultsError):
        build({**ok, "axes": {"nu": float("nan")}})
    # every violation is listed at once
    with pytest.raises(ResultsError, match="3 results invariant violation"):
        build(
            {**ok, "stderr": None}, {**ok, "row": "b", "unit": ""}, {**ok, "row": "c", "source": ""}
        )
    # allowed: an exact value (stderr 0 / None stored as NaN), a missing Monte Carlo number
    res = build(
        ok,
        {**ok, "column": "e", "stderr": 0.0, "exact": True, "unit": ""},
        {**ok, "column": "n", "value": float("nan"), "stderr": float("nan")},
        {**ok, "row": "r2", "stderr": 0.0},
        {**ok, "row": "r2", "column": "e", "stderr": None, "exact": True, "axes": {"nu": 1}},
    )
    assert math.isnan(res.value("t", "r", "e")[1]) and res.is_exact("t", "r", "e")
    assert res.value("t", "r", "c") == (1.0, 0.1)
    assert res.axes("t", "r2") == {"nu": 1}
    piv = res.pivot("t")
    assert list(piv.columns) == ["c", "c_stderr", "e", "e_stderr", "n", "n_stderr"]
    assert list(piv.index) == ["r", "r2"] and math.isnan(piv.loc["r2", "n"])
    assert list(res.pivot("t", value=False).columns) == ["c_stderr", "e_stderr", "n_stderr"]
    with pytest.raises(KeyError):
        res.value("t", "r", "zz")
    path = res.write(tmp_path / "r.parquet")
    back = Results.read(path)
    pd.testing.assert_frame_equal(back.frame, res.frame)
    # a tampered file fails the invariants on read
    df = pd.read_parquet(path)
    df.loc[0, "stderr"] = np.nan
    df.to_parquet(path, index=False)
    with pytest.raises(ResultsError):
        Results.read(path)
    assert len(ResultsBuilder().build()) == 0


def test_diff_results_statuses() -> None:
    """Statuses, the ``max`` rule (not quadrature), the verifier's p_diff.py cases."""

    def res(rows: list[tuple[str, float, float | None, bool, str]]) -> Results:
        b = ResultsBuilder()
        for col, v, se, exact, unit in rows:
            b.add("t", "r", col, v, se, exact=exact, unit=unit, source="computed")
        return b.build()

    old = res(
        [
            ("same", 1.0, 0.1, False, "u"),
            ("max_rule", 1.0, 0.1, False, "u"),  # quadrature would call 0.25 unchanged
            ("within", 1.0, 0.1, False, "u"),
            ("boundary", 1.0, 0.1, False, "u"),  # 1.2 - 1.0 < 0.2 in floating point
            ("exact_same", 2.0, None, True, ""),
            ("exact_moved", 2.0, None, True, ""),
            ("exact_zero_to_tiny", 0.0, None, True, ""),
            ("to_nan", 1.0, 0.1, False, "u"),
            ("nan_both", float("nan"), float("nan"), False, "u"),
            ("zero_se_same", 1.0, 0.0, False, "u"),
            ("zero_se_noise", 1.0, 0.0, False, "u"),
            ("zero_se_moved", 1.0, 0.0, False, "u"),
            ("unit_change", 21.4, 0.1, False, "vol pts"),
            ("exact_to_mc", 2.0, None, True, "price"),
            ("mc_to_exact", 2.0, 0.1, False, "price"),
            ("gone", 1.0, 0.1, False, "u"),
        ]
    )
    new = res(
        [
            ("same", 1.0, 0.1, False, "u"),
            ("max_rule", 1.25, 0.1, False, "u"),
            ("within", 1.15, 0.05, False, "u"),
            ("boundary", 1.2, 0.1, False, "u"),
            ("exact_same", 2.0 * (1 + 1e-14), None, True, ""),
            ("exact_moved", 2.0 * (1 + 1e-9), None, True, ""),
            ("exact_zero_to_tiny", 1e-300, None, True, ""),
            ("to_nan", float("nan"), float("nan"), False, "u"),
            ("nan_both", float("nan"), float("nan"), False, "u"),
            ("zero_se_same", 1.0, 0.0, False, "u"),
            ("zero_se_noise", 1.0 + 1e-15, 0.0, False, "u"),
            ("zero_se_moved", 1.0 + 1e-6, 0.0, False, "u"),
            ("unit_change", 21.4, 0.1, False, "%"),
            ("exact_to_mc", 2.0, 0.1, False, "price"),
            ("mc_to_exact", 2.0, None, True, "price"),
            ("new", 3.0, 0.1, False, "u"),
        ]
    )
    d = diff_results(old, new).set_index("column")
    expected = {
        "same": "unchanged",
        "max_rule": "moved",
        "within": "unchanged",
        "boundary": "unchanged",
        "exact_same": "unchanged",
        "exact_moved": "moved",
        "exact_zero_to_tiny": "moved",
        "to_nan": "moved",
        "nan_both": "unchanged",
        "zero_se_same": "unchanged",
        "zero_se_noise": "unchanged",
        "zero_se_moved": "moved",
        "unit_change": "changed",
        "exact_to_mc": "changed",
        "mc_to_exact": "changed",
        "gone": "removed",
        "new": "added",
    }
    assert d["status"].to_dict() == expected
    assert d.loc["max_rule", "n_stderr"] == pytest.approx(2.5)
    assert d.loc["unit_change", "why"] == "unit 'vol pts' -> '%'"
    assert d.loc["exact_to_mc", "why"] == "exact -> Monte Carlo"
    assert set(changed(diff_results(old, new))["column"]) == {
        k for k, v in expected.items() if v != "unchanged"
    }
    assert diff_results(old, new, nse=3.0).set_index("column").loc["max_rule", "status"] == (
        "unchanged"
    )
    assert changed(diff_results(old, old)).empty
    for bad in (float("nan"), float("inf"), 0.0, -1.0):
        with pytest.raises(ValueError, match="finite and positive"):
            diff_results(old, new, nse=bad)


# --------------------------------------------------------------------------------------------
# style, LaTeX, packaging
# --------------------------------------------------------------------------------------------


def test_mc_errorbar_refuses_missing_stderr() -> None:
    _, ax = style.new_figure()
    with pytest.raises(ValueError, match="without a finite stderr"):
        style.mc_errorbar(ax, [1, 2], [1.0, 2.0], [0.1, float("nan")])
    with pytest.raises(ValueError, match="without a finite stderr"):
        style.mc_errorbar(ax, [1, 2], [1.0, 2.0], [0.1, -1.0])
    # exact points and missing values are fine
    style.mc_errorbar(
        ax, [1, 2, 3], [1.0, 2.0, np.nan], [np.nan, 0.1, np.nan], exact=[True, False, False]
    )
    with pytest.raises(ValueError, match="palette has 8 hues"):
        style.series_color(8)
    assert len(style.PALETTE) == len(style.MARKERS) == 8


def _require_tectonic() -> str:
    exe = runner.find_tectonic()
    if exe is None:
        pytest.skip(runner.TECTONIC_MISSING)
    return exe


def test_latex_compiles(tmp_path: Path) -> None:
    """``volsto-study run`` with the LaTeX check: ``study.tex`` compiles cleanly and the PDF
    exists; ``render`` recompiles it."""
    exe = _require_tectonic()
    cfg = write_config(
        tmp_path / "ex.yaml", example_mapping(params={**EXAMPLE_PARAMS, "extra_rows": 60})
    )
    out = tmp_path / "out"
    t0 = time.perf_counter()
    code = runner.main(["run", str(cfg), "--out", str(out), *roots_argv(tmp_path)])
    print(f"run with the LaTeX check: {time.perf_counter() - t0:.2f} s ({exe})")
    m = json.loads((out / "manifest.json").read_text())
    assert m["latex"]["status"] == "ok", m["latex"]["reason"]
    assert m["latex"]["problems"] == [] and m["latex"]["log"] == "study.log"
    assert code == 0
    # the 60-row, 12-column table is split into three scaled floats (no overflow)
    parts = ["wide-part1", "wide-part2", "wide-part3"]
    assert m["tables"] == [*parts, "calls"] and m["table_specs"] == ["wide", "calls"]
    tex = (out / "study.tex").read_text()
    assert [ln for ln in tex.splitlines() if ln.startswith(r"\input{tables/wide")] == [
        rf"\input{{tables/{p}.tex}}" for p in parts
    ]
    part1 = (out / "tables" / "wide-part1.tex").read_text()
    assert "(part 1 of 3)" in part1 and r"\begin{adjustbox}" in part1 and "longtable" not in part1
    assert part1.count(r"\\") == 26  # header + 25 rows
    assert (
        "**Table `wide-part3`.** A long and wide table (part 3 of 3)"
        in (out / "study.md").read_text()
    )
    pdf = out / "study.pdf"
    assert pdf.is_file() and pdf.read_bytes().startswith(b"%PDF")
    print(f"tectonic compile {m['latex']['seconds']:.2f} s, study.pdf {pdf.stat().st_size} bytes")
    pdf.unlink()
    assert runner.main(["render", str(out)]) == 0
    assert pdf.is_file()


def test_latex_check_catches(tmp_path: Path) -> None:
    """A long table (longtable), a literal ``$``, explicit math and mapped glyphs compile
    cleanly; a dropped glyph, a float taller than the page and an overfull box fail."""
    assert latex.parse_latex_log(
        "\n".join(
            [
                "Overfull \\hbox (2.5pt too wide) in paragraph at lines 1--2",
                "Overfull \\hbox (15.2pt too wide) in alignment at lines 3--9",
                "Missing character: There is no \u2603 (U+2603) in font [lmroman10-regular]:ma",
                "LaTeX Warning: Float too large for page by 1434.19853pt on input line 164.",
                "LaTeX Warning: There were undefined references.",
                "! Undefined control sequence.",
                "Underfull \\hbox (badness 10000) in paragraph at lines 74--97",
            ]
        )
    ) == [
        "Overfull \\hbox (15.2pt too wide) in alignment at lines 3--9",
        "Missing character: There is no \u2603 (U+2603) in font [lmroman10-regular]:ma",
        "LaTeX Warning: Float too large for page by 1434.19853pt on input line 164.",
        "LaTeX Warning: There were undefined references.",
        "! Undefined control sequence.",
    ]
    _require_tectonic()
    b = ResultsBuilder()
    n_rows = 40
    for i in range(n_rows):
        b.add("long", f"row_{i:03d}", "v", 1.0 + i, 0.05, unit="vol pts", source="computed")
        b.add(
            "long",
            f"row_{i:03d}",
            "w",
            1e-9 * (i + 1),
            2.5e-11,
            unit=DIMENSIONLESS,
            source="computed",
        )
    res = b.build()
    spec = TableSpec(
        "long", "A long table with ✓ ≲ ⇒ and 100%", "long", (Column("v", "v"), Column("w", "w"))
    )
    good = tmp_path / "good"
    (good / "tables").mkdir(parents=True)
    table_tex = latex.latex_table(spec, res)
    assert r"\begin{longtable}{lrr}" in table_tex and r"\endhead" in table_tex
    assert table_tex.count(r" \pm ") == 2 * n_rows
    (good / "tables" / "long.tex").write_text(table_tex)
    narrative = "A lone $ sign, $5, \\(\\sigma_{\\rm atm}\\), ✓ ≲ ⇒ µ ∆ and 50% of a_b."
    (good / "study.tex").write_text(
        latex.study_tex(
            title="Long $ table",
            question="Does a long table compile?",
            narrative=narrative,
            tables=[spec],
            figures=[],
            provenance=[("config", "/very/long/" + "x" * 200 + "/cfg.yaml"), ("a", "b c")],
        )
    )
    check = latex.compile_latex(good)
    assert check["status"] == "ok", check["reason"]
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "study.tex").write_text(
        latex.TEX_PREAMBLE
        + "\\begin{document}\n"
        + "Snow \u2603 here.\n\n"
        + "\\noindent\\rule{20cm}{1pt}\n\n"
        + "\\begin{figure}[htbp]\\rule{1pt}{40cm}\\caption{tall}\\end{figure}\n"
        + "\\end{document}\n"
    )
    check = latex.compile_latex(bad)
    print(check["reason"])
    assert check["status"] == "failed"
    kinds = " ".join(check["problems"])
    assert "Missing character" in kinds and "Float too large" in kinds and "Overfull" in kinds


def test_runner_imports_without_matplotlib() -> None:
    code = (
        "import sys; sys.modules['matplotlib'] = None\n"
        "import volsto.studies.runner as r, volsto.studies.style as s\n"
        "assert r.main(['list']) in (0, 1)\n"
        "print('ok')\n"
    )
    env = {**os.environ, "NUMBA_NUM_THREADS": os.environ.get("NUMBA_NUM_THREADS", "2")}
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().endswith("ok")


def test_console_script_help() -> None:
    exe = Path(sys.executable).with_name("volsto-study")
    if not exe.exists():
        pytest.skip(
            "volsto-study is not installed in this environment: "
            "uv pip install --python .venv/bin/python --no-deps -e ."
        )
    proc = subprocess.run([str(exe), "--help"], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0
    for cmd in ("run", "rerun", "render", "list"):
        assert cmd in proc.stdout
