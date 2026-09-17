"""Study runner (SPEC §10.1, M10 Part 1; console script ``volsto-study``): a study is a YAML
config plus a runner module, and ``volsto-study run <config>`` turns it into a self-contained
output directory whose tables and figures regenerate from its ``results.parquet``.

Output directory (default ``<repo>/outputs/studies/<name>``, or ``--out``)::

    results.parquet      every number with its stderr (volsto.studies.results.Results)
    tables/<name>.tex    booktabs table floats scaled to the text width; a table longer than
                         25 rows is split into <name>-partK files (latex.split_table)
    figures/<name>.pdf   house style (volsto.studies.style), and <name>.png
    study.md             title, the question first, the narrative with the tables inlined and
                         the figures linked, the provenance block
    study.tex            master document \\input-ing every table, \\includegraphics-ing every figure
    study.pdf, study.log study.tex compiled by Tectonic and its log (the LaTeX check)
    manifest.json        provenance (below)
    profile.txt          cProfile of compute() under --profile

A run is written into a temporary sibling directory (``.<name>.staging-<id>``, which is also
``ctx.out_dir`` during ``compute``) and moved into place at the end: a run that fails — a
missing requirement found inside ``compute``, an exception — leaves no output directory and does
not touch an earlier one.  Moving into an existing directory replaces the files the runner owns
(:data:`OWNED_OUTPUTS`) and keeps the rest (``rerun/``); a stale ``study.pdf`` of an earlier run is
removed, and said so, when the LaTeX check is off.

**Config** (:class:`StudyConfig`, strict YAML: an unknown, missing or duplicated key is an error;
``question`` is the only optional key and defaults to the module's ``QUESTION``)::

    name: s1_forward_vol                      # [A-Za-z0-9_-]+
    runner: volsto.studies.catalogue.s1_forward_vol
    question: "How do forward vol and cliquet prices move with vol-of-vol at a fixed smile?"
    grid: configs/grids/placeholder_cached.yaml   # or null
    store: outputs/store
    cache: cache
    outputs: outputs
    mode: full                                # full | fast
    seeds: {pricing: 2024}
    params: {...}                             # validated against the module (below)

The YAML is read by :class:`StrictLoader`: duplicate keys raise; floats follow YAML 1.2
(``1e-3`` and ``2e4`` are floats, not strings); booleans are only ``true`` / ``false`` (``yes``,
``on`` stay strings); a non-finite number, and in ``params`` a string spelling one (``nan``,
``inf``), is an error.  Paths in the YAML resolve against the repository root (:data:`REPO_ROOT`,
the directory holding ``pyproject.toml``), never the cwd.  CLI overrides ``--grid --store
--cache --outputs --out`` are resolved against the **cwd** (the shell convention of
``volsto-precompute`` and ``volsto-viewer``) and stored in the resolved config repo-relative when
they lie inside the repository, absolute otherwise.  ``--fast`` sets ``mode: fast``; ``--set
key.sub=value`` sets a params entry to the value parsed by the same loader (an empty value is an
error; an intermediate key must exist and be a mapping).

**Runner-module protocol** (duck-typed, checked at load by :func:`load_runner_module`)::

    TITLE: str
    QUESTION: str                              # the config's `question` overrides it
    REQUIRED_PARAMS: tuple[str, ...]
    OPTIONAL_PARAMS: tuple[str, ...]           # optional attribute (see below)
    def validate_params(params) -> None        # optional attribute, deeper checks
    def requirements(ctx: StudyContext) -> list[Requirement]   # cheap; no pricing
    def compute(ctx: StudyContext) -> Results
    def tables(results: Results) -> list[TableSpec]            # from results ONLY
    def figures(results: Results) -> list[FigureSpec]          # from results ONLY
    def narrative(results: Results) -> str                     # markdown, from results ONLY

A params key missing from ``REQUIRED_PARAMS`` is an error; a params key that is in neither
``REQUIRED_PARAMS`` nor ``OPTIONAL_PARAMS`` is an error too (strict loading: a typo is never
silently ignored).  The narrative is markdown: explicit math is written ``\\(...\\)``, a ``$`` is a
literal dollar sign; a line holding only ``{{table:<name>}}`` / ``{{figure:<name>}}`` places a
table or a figure, the rest follow the narrative (:mod:`volsto.studies.latex`).

**No calibration: enforced in one place; what happened, observed and separated.**

* *Enforced.*  :func:`volsto.calibration.particle.calibrate_leverage` — the only function that
  runs the particle calibration — calls :func:`volsto.calibration.guard.check_calibration_allowed`
  as its first statement.  ``run`` / ``render`` hold :func:`calibration_forbidden` (re-exported
  from :mod:`volsto.calibration.guard`) from the requirements check to the end of the LaTeX
  check: a process flag (a depth counter under a lock: nesting- and thread-safe, and it covers
  every thread of the process) plus the environment variable ``VOLSTO_FORBID_CALIBRATION``
  (inherited by any child process started meanwhile — a ``ProcessPoolExecutor``, the
  ``volsto-precompute --workers`` pool).  However the function object is reached — an alias bound
  at import, a dict or ``functools.partial``, a thread pool, a spawned process, a thread that
  outlives ``compute`` but calibrates before the run ends — it raises
  :class:`CalibrationForbiddenError` and the run exits 1.  ``ctx.cache`` is a
  :class:`ReadOnlyLeverageCache` besides: ``allow_calibrate`` is forced to ``False``, the writers
  (``store``, ``write_report``, ``merge_manifest``, the manifest update, and the ``calibrate`` /
  ``put`` names) raise, a miss becomes :class:`MissingRequirements` with the precompute line.
* *This study calibrated* (``recalibrated``).  The run passes a per-run marker directory to the
  guard (``VOLSTO_CALIBRATION_MARKERS``, inherited by children with the prohibition): every call
  of ``calibrate_leverage`` in this process **or a child** leaves a ``refused`` or ``started``
  marker (pid, host, time).  ``recalibrated`` is true when a calibration *started* — only
  possible for a process that escaped the prohibition (a child whose environment dropped the
  flag but kept the marker variable) — and the run then fails loudly (exit 1) with
  ``calibrations_started`` in the manifest.  A refusal, even one the study caught, also fails the
  run (``calibration_refusals``).
* *The cache changed during the run* (``cache_changed_during_run``, not a failure).  The
  configured cache root is snapshotted (:func:`snapshot_cache`: the entry directories holding a
  ``leverage.npz`` with its size and mtime, and the manifest row count) before the requirements,
  after ``compute``, after the tables and figures, and after the LaTeX check.  New or rewritten
  entries and added manifest rows are recorded in ``cache_observation`` and in the provenance
  block and logged as a warning, **not attributed**: the leverage cache records no writer
  identity (no pid or host in ``spec.json``, the leverage metadata or the manifest), and another
  process — ``volsto-precompute``, the study-C refit — may legitimately be writing into the same
  root.  A study's own ``LeverageCache.store`` into the configured root (a write, not a
  calibration) is reported the same way.
* *Residual.*  A thread or process that calibrates or writes after the run has ended, or writes
  into another cache root, is neither refused nor observed; several runs in one process share the
  outermost run's marker directory.

**Requirements.**  ``requirements(ctx)`` returns :class:`Requirement` records of kind
``point`` (a store point id; it needs a configured grid — a point requirement with ``grid: null``
is a config error), ``leverage`` (a cache key) or ``artefact`` (a path under the ``outputs`` root;
a leading ``outputs/`` — the default root's name, as in the contract's
``outputs/m8b/m8b_table_D.csv`` — is read as that root).  ``run`` checks them all before
computing anything; on any miss it prints every command and exits **2** without creating the
output directory.  The printed commands are runnable as printed:

* the point and leverage ids that are points of the **configured** grid are one line
  ``volsto-precompute --grid G --store S --cache C --only <id> [<id> ...] --resume``, every path
  relative to the current directory when it lies below it and absolute otherwise;
* an id that is not a point of the configured grid gets a ``#`` note (add it to the grid YAML,
  then the same line with that id);
* when the store's run records name a different grid than the configured one, a ``#`` note says
  so (the configured grid is kept — the ids were checked against it);
* a missing leverage with ``grid: null`` gets a ``#`` note (no line can produce it without a grid);
* an artefact prints the requirement's own command.

A miss met inside ``compute`` — ``ctx.leverage``, or ``ctx.cache`` reached through e.g.
``LSVBuilder(ctx.cache, ...)`` — is reported the same way (exit 2).

**manifest.json** — ``study``, ``runner``, ``runner_sha256``, ``title``, ``question``,
``config`` (the full resolved mapping), ``config_path`` (repo-relative), ``config_sha256``
(SHA-256 of the canonical JSON of the resolved config — what ran, overrides included),
``config_file_sha256`` (the YAML's bytes), ``git_commit`` (full), ``git_dirty`` (``git status
--porcelain`` over :data:`DIRTY_PATHS` non-empty, untracked files included; ``null`` without git),
``code_version`` (:func:`volsto.calibration.cache.code_version`), ``calibration_code_tag``,
``grid`` (``{path, sha256, name}`` or ``null``), ``cache_keys`` (``[{key, what, n_particles}]``:
every leverage read through ``ctx.cache`` — ``ctx.leverage`` or any hit of
``ctx.cache.get_or_calibrate``, ``LSVBuilder(ctx.cache, ...)`` included —, the leverage
requirements, and the leverages behind the store points read), ``store_points`` (ids read through
``ctx.store()`` — per-point accessors record their id, whole-table reads every id they return —
plus the point requirements), ``artefacts`` (``[{path, sha256}]``), ``particles`` (sorted distinct
``n_particles``), ``seeds``, ``n_paths`` (``ctx.record("n_paths", ...)``, else ``null``),
``records`` (every other ``ctx.record``), ``mode``, ``tables`` (the table files written, after
:func:`volsto.studies.latex.split_table`), ``table_specs`` (the declared tables), ``figures``,
``n_results``,
``wall_clock`` (``{requirements, compute, render, total}`` seconds; render = tables and figures —
the two text documents are written after the clock stops, so ``render`` rebuilds them
byte-identically), ``recalibrated``, ``calibrations_started``, ``calibration_refusals``,
``cache_changed_during_run`` and ``cache_observation`` (above), ``host``,
``python``, ``numpy``, ``numba``, ``numba_threads``, ``pandas``, ``matplotlib``, ``created_utc``,
``argv`` (``--store`` / ``--cache`` masked by :func:`volsto.viewers.precompute.provenance_argv`,
``--grid`` repo-relative, the other paths repo-relative or masked), ``results_sha256``, ``latex``
(the LaTeX check), ``profile``, and for a rerun ``rerun_of``.

**The LaTeX check** (on by default for ``run`` and ``render``; :func:`volsto.studies.latex.
compile_latex`): Tectonic compiles ``study.tex`` with ``--keep-logs`` and the check fails —
exit 1, the outputs still written, the reason in the manifest and the log — when the engine
fails, the PDF is missing, or ``study.log`` shows a TeX error, a missing character (a glyph the
font lacks is otherwise silently dropped), a float too large for the page (otherwise silently
truncated), an unresolved reference, or an overfull box beyond
:data:`volsto.studies.latex.OVERFULL_TOLERANCE_PT`.  A missing engine skips the check with the
install line.  ``--no-latex-check`` removes a stale ``study.pdf`` / ``study.log`` and says so.

**Commands** (every one ends with a log line giving its exit status, its wall clock,
``recalibrated`` and whether the cache changed during the run)::

    volsto-study run <config.yaml> [--out DIR] [--fast] [--grid G] [--store S] [--cache C]
                     [--outputs O] [--set k=v ...] [--profile] [--no-latex-check] [-v]
        exit 0; 2 on missing requirements (commands printed, nothing written); 1 on a failed
        LaTeX check, a calibration started or refused during the run, a config error or any
        other exception (one error line; the traceback with -v or --profile); a cache changed
        by someone else is a warning
    volsto-study rerun <output_dir> [--out DIR] [--grid G] [--store S] [--cache C] [--outputs O]
                       [--nse 2.0] [-v]
        runs manifest["config"] (never the YAML on disk) into DIR (default
        <output_dir>/rerun/<utc>), writes diff.csv + diff.md there; exit 0 when nothing changed,
        1 when a number moved by more than nse stderr, changed unit or kind, was added or
        removed (or the run itself failed as above), 2 on missing requirements.  The path flags
        relocate a moved store / cache / outputs (nothing else of the stored config can change);
        --nse must be finite and positive.
    volsto-study render <output_dir> [--no-latex-check] [-v]
        tables, figures, study.md, study.tex (and study.pdf) from results.parquet (the module's
        tables / figures / narrative see only the results) and the stored manifest, which is not
        modified
    volsto-study list
        the catalogue configs (configs/studies/catalogue/*.yaml) with their mode and question

A command-line usage error exits 1 (argparse's usual 2 is reserved for missing requirements).

**Deviations from the M10 contract** (recorded as the contract asks):

1. ``LeverageCache`` has no ``calibrate`` / ``put`` methods; its writer is ``store``.  The
   wrapper refuses ``store``, ``write_report``, ``merge_manifest`` and the manifest update, and
   defines ``calibrate`` / ``put`` only to refuse them.  The structural guarantee is the guard at
   the entry of ``calibrate_leverage`` (above), not the wrapper.
2. Additions: ``OPTIONAL_PARAMS`` / ``validate_params`` (optional module attributes), the
   narrative placeholders, the ``rerun`` relocation flags and ``--nse``, ``render
   --no-latex-check``, ``-v``, the ``ctx.*_requirement`` / ``ctx.artefact`` / ``ctx.missing``
   helpers, the manifest keys beyond the contract's list.
3. ``rerun`` exits 1 when a number changed unit or kind, was added or removed, as well as when
   one moved (a changed set of numbers is a changed conclusion).
4. A non-exact row with a NaN value and a NaN stderr is allowed (a missing number, rendered
   ``--``); a finite Monte Carlo value still requires a finite stderr.
5. :class:`Column`, :class:`TableSpec`, :class:`FigureSpec` are defined in
   :mod:`volsto.studies.results` (no matplotlib dependency) and re-exported here and by
   :mod:`volsto.studies.latex` / :mod:`volsto.studies.style`.  ``study_tex`` / ``study_markdown``
   take keyword arguments (title, question, narrative, specs, provenance).
6. An artefact id is relative to the configured ``outputs`` root; the contract's example
   ``outputs/m8b/...`` is accepted by reading a leading ``outputs/`` as that root.
7. ``recalibrated`` is observed (not a constant): a calibration started in this process or a
   child fails the run; a cache that changed during the run is reported, not attributed, and
   does not fail it.
8. The precompute line names the configured grid (the read API's
   :func:`~volsto.viewers.api.precompute_command` names the store's recorded grid): the ids are
   checked against the configured grid, and a different recorded grid is reported in a note.

Checked by ``tests/test_study_runner.py`` (run → files and manifest; rerun from the manifest with
nothing moved; a stored value perturbed by 3 stderr → moved, exit 1; missing point / leverage /
artefact → runnable commands and exit 2 before compute, no directory; the guard against every
route; the cache observation; the read-only cache records its hits; render byte-identical; the
toy store's requirement check; the strict YAML; the LaTeX check on long tables, dropped glyphs and
literal dollars; one-line errors).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import cProfile
import hashlib
import importlib
import io
import json
import logging
import math
import os
import platform
import pstats
import re
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import Any, NoReturn

import numpy as np
import pandas as pd
import yaml

from volsto.calibration import guard
from volsto.calibration.cache import (
    LEVERAGE_NAME,
    CacheMissError,
    LeverageCache,
    atomic_write,
    code_version,
    has_complete_leverage,
    spec_key,
)
from volsto.calibration.cache import MANIFEST_NAME as CACHE_MANIFEST_NAME
from volsto.calibration.diagnostics import CalibrationReport
from volsto.calibration.guard import CalibrationForbiddenError, calibration_forbidden
from volsto.calibration.particle import CALIBRATION_CODE_TAG
from volsto.config import CalibrationSpec, ConfigError, SimConfig
from volsto.models.leverage import LeverageFunction
from volsto.models.lsv import LSV
from volsto.studies import latex
from volsto.studies.latex import TECTONIC_MISSING, compile_latex, find_tectonic
from volsto.studies.results import (
    DEFAULT_NSE,
    Column,
    FigureSpec,
    Results,
    TableSpec,
    changed,
    diff_results,
)
from volsto.viewers.grid import REPO_ROOT, GridPoint, GridSpec, enumerate_points, load_grid
from volsto.viewers.precompute import provenance_argv
from volsto.viewers.store import StoreReader, utc_now

__all__ = [
    "TECTONIC_MISSING",
    "CalibrationForbiddenError",
    "Column",
    "FigureSpec",
    "MissingRequirements",
    "ReadOnlyLeverageCache",
    "Requirement",
    "Results",
    "StrictLoader",
    "StudyCacheMissError",
    "StudyConfig",
    "StudyContext",
    "StudyRun",
    "TableSpec",
    "calibration_forbidden",
    "compile_latex",
    "find_tectonic",
    "load_runner_module",
    "load_study_config",
    "main",
    "render_study",
    "rerun_study",
    "run_study",
    "snapshot_cache",
]

log = logging.getLogger(__name__)

#: Study modes: ``full`` (the study's numbers) and ``fast`` (CI plumbing on small budgets).
STUDY_MODES: tuple[str, ...] = ("full", "fast")
#: Keys of a study YAML; all required except :data:`OPTIONAL_CONFIG_KEYS`.
CONFIG_KEYS: tuple[str, ...] = (
    "name",
    "runner",
    "question",
    "grid",
    "store",
    "cache",
    "outputs",
    "mode",
    "seeds",
    "params",
)
OPTIONAL_CONFIG_KEYS: frozenset[str] = frozenset({"question"})
#: The path-valued config keys (``grid`` may be null).
PATH_KEYS: tuple[str, ...] = ("grid", "store", "cache", "outputs")
#: Attributes a runner module must define (callables after the first three).
MODULE_ATTRIBUTES: tuple[str, ...] = (
    "TITLE",
    "QUESTION",
    "REQUIRED_PARAMS",
    "requirements",
    "compute",
    "tables",
    "figures",
    "narrative",
)
REQUIREMENT_KINDS: tuple[str, ...] = ("point", "leverage", "artefact")
#: Largest seed accepted (the engine's PCG64 streams take 64-bit seeds).
MAX_SEED = 2**64 - 1
#: Strings rejected as ``params`` values: they spell a non-finite number that ``float()`` would
#: silently accept.
NONFINITE_STRINGS: frozenset[str] = frozenset(
    {"nan", "+nan", "-nan", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity"}
)
#: Default output root of ``run`` (``<root>/<name>``), repository-relative.
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "studies"
#: Catalogue configs listed by ``volsto-study list``.
CATALOGUE_DIR = REPO_ROOT / "configs" / "studies" / "catalogue"
#: The prefix of an artefact id that names the default outputs root (read as the configured one).
OUTPUTS_PREFIX = "outputs/"
#: Paths whose uncommitted changes (untracked files included) make ``git_dirty`` true.
DIRTY_PATHS: tuple[str, ...] = ("volsto", "configs", "scripts", "tests", "pyproject.toml")
#: File names of an output directory.
RESULTS_NAME = "results.parquet"
MANIFEST_NAME = "manifest.json"
STUDY_MD = "study.md"
STUDY_TEX = "study.tex"
STUDY_PDF = "study.pdf"
STUDY_LOG = "study.log"
PROFILE_NAME = "profile.txt"
TABLES_DIR = "tables"
FIGURES_DIR = "figures"
RERUN_DIR = "rerun"
DIFF_CSV = "diff.csv"
DIFF_MD = "diff.md"
#: What a run replaces in an existing output directory (everything else there is kept).
OWNED_OUTPUTS: tuple[str, ...] = (
    RESULTS_NAME,
    MANIFEST_NAME,
    STUDY_MD,
    STUDY_TEX,
    STUDY_PDF,
    STUDY_LOG,
    PROFILE_NAME,
    TABLES_DIR,
    FIGURES_DIR,
    DIFF_CSV,
    DIFF_MD,
)
STAGING_MARK = ".staging-"
#: Exit codes.
EXIT_OK = 0
EXIT_FAILED = 1
EXIT_MOVED = 1
EXIT_MISSING = 2

_NAME = re.compile(r"[A-Za-z0-9_-]+")
_MODULE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*")


# --------------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------------


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), allow_nan=False)


def display_path(p: str | Path) -> str:
    """``p`` repo-relative when it lies inside :data:`REPO_ROOT`, else absolute."""
    resolved = Path(p).expanduser().resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def cwd_path(p: str | Path) -> str:
    """``p`` for a command the user runs from the current directory: relative to the cwd when it
    lies below it, else absolute (both resolved)."""
    resolved = Path(p).expanduser().resolve()
    try:
        rel = resolved.relative_to(Path.cwd().resolve())
    except ValueError:
        return str(resolved)
    return str(rel) if str(rel) != "." else "."


def resolve_repo_path(p: str | Path) -> Path:
    """A config path: absolute as given, else under :data:`REPO_ROOT`."""
    q = Path(p).expanduser()
    return q if q.is_absolute() else REPO_ROOT / q


def git_state() -> tuple[str | None, bool | None]:
    """``(full commit, dirty)``; ``None`` for what git cannot tell."""

    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout if out.returncode == 0 else None

    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain", "--", *DIRTY_PATHS)
    return (head.strip() or None) if head else None, (
        None if status is None else bool(status.strip())
    )


def _json_plain(value: Any, path: str) -> Any:
    """``value`` checked to be plain JSON data (the params of a study)."""
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        if value.strip().lower() in NONFINITE_STRINGS:
            raise ConfigError(f"{path}: {value!r} spells a non-finite number")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ConfigError(f"{path}: non-finite number {value!r}")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_plain(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for k, v in value.items():
            if not isinstance(k, str):
                raise ConfigError(f"{path}: key {k!r} must be a string")
            out[k] = _json_plain(v, f"{path}.{k}")
        return out
    raise ConfigError(f"{path}: {value!r} ({type(value).__name__}) is not plain YAML data")


# --------------------------------------------------------------------------------------------
# strict YAML
# --------------------------------------------------------------------------------------------

_BOOL_TAG = "tag:yaml.org,2002:bool"
_FLOAT_TAG = "tag:yaml.org,2002:float"
#: YAML 1.2 core-schema floats (the exponent form without a dot included).
_FLOAT_12 = re.compile(
    r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+
    |[-+]?\.[0-9_]+(?:[eE][-+]?[0-9]+)?
    |[-+]?\.(?:inf|Inf|INF)
    |\.(?:nan|NaN|NAN))$""",
    re.X,
)
_BOOL_12 = re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$")


class StrictLoader(yaml.SafeLoader):
    """``yaml.SafeLoader`` with duplicate keys refused, YAML 1.2 floats (``1e-3``) and YAML 1.2
    booleans (``true`` / ``false`` only)."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            try:
                duplicate = key in seen
            except TypeError:
                continue  # an unhashable key: the base constructor reports it
            if duplicate:
                raise yaml.constructor.ConstructorError(
                    "while constructing a mapping",
                    node.start_mark,
                    f"found duplicate key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


StrictLoader.yaml_implicit_resolvers = {
    ch: [(tag, rx) for tag, rx in resolvers if tag not in (_BOOL_TAG, _FLOAT_TAG)]
    for ch, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
StrictLoader.add_implicit_resolver(_BOOL_TAG, _BOOL_12, list("tTfF"))
StrictLoader.add_implicit_resolver(_FLOAT_TAG, _FLOAT_12, list("-+0123456789."))


def load_yaml_strict(text: str, source: str) -> Any:
    try:
        return yaml.load(text, Loader=StrictLoader)  # a SafeLoader subclass
    except yaml.YAMLError as exc:
        raise ConfigError(f"{source}: {exc}") from exc


# --------------------------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StudyConfig:
    """A study YAML, strictly validated (module docstring).  Paths are kept as given (relative
    = to :data:`REPO_ROOT`); :meth:`path` resolves them."""

    name: str
    runner: str
    grid: str | None
    store: str
    cache: str
    outputs: str
    mode: str
    seeds: Mapping[str, int]
    params: Mapping[str, Any]
    question: str | None = None

    def __post_init__(self) -> None:
        for key in ("name", "runner", "mode"):
            if not isinstance(getattr(self, key), str):
                raise ConfigError(f"{key}: expected a string, got {getattr(self, key)!r}")
        if not _NAME.fullmatch(self.name):
            raise ConfigError(f"study name {self.name!r} must match [A-Za-z0-9_-]+")
        if not _MODULE.fullmatch(self.runner):
            raise ConfigError(f"runner {self.runner!r} is not a dotted module name")
        if self.mode not in STUDY_MODES:
            raise ConfigError(f"mode {self.mode!r} must be one of {STUDY_MODES}")
        for key in PATH_KEYS:
            v = getattr(self, key)
            if key == "grid" and v is None:
                continue
            if not isinstance(v, str) or not v:
                raise ConfigError(f"{key}: expected a path string, got {v!r}")
        if not isinstance(self.seeds, Mapping):
            raise ConfigError(f"seeds must be a mapping, got {self.seeds!r}")
        for k, s in self.seeds.items():
            if (
                not isinstance(k, str)
                or isinstance(s, bool)
                or not isinstance(s, int)
                or not 0 <= s <= MAX_SEED
            ):
                raise ConfigError(f"seeds.{k}: expected an integer in [0, 2**64), got {s!r}")
        if not isinstance(self.params, Mapping):
            raise ConfigError(f"params must be a mapping, got {self.params!r}")
        object.__setattr__(self, "seeds", dict(self.seeds))
        object.__setattr__(self, "params", _json_plain(copy.deepcopy(dict(self.params)), "params"))
        if self.question is not None:
            q = self.question
            if not isinstance(q, str) or not q.strip() or "\n" in q.strip():
                raise ConfigError(f"question must be one non-empty line, got {q!r}")

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any], *, source: str = "study config") -> StudyConfig:
        if not isinstance(data, Mapping):
            raise ConfigError(f"{source}: top level must be a mapping")
        unknown = sorted(str(k) for k in set(data) - set(CONFIG_KEYS))
        if unknown:
            raise ConfigError(f"{source}: unknown keys {unknown}; known {list(CONFIG_KEYS)}")
        missing = [k for k in CONFIG_KEYS if k not in data and k not in OPTIONAL_CONFIG_KEYS]
        if missing:
            raise ConfigError(f"{source}: missing required keys {missing}")
        try:
            return cls(**{k: data[k] for k in CONFIG_KEYS if k in data})
        except ConfigError as exc:
            raise ConfigError(f"{source}: {exc}") from exc

    def to_mapping(self) -> dict[str, Any]:
        """The resolved mapping the manifest stores (and ``rerun`` runs)."""
        return {
            "name": self.name,
            "runner": self.runner,
            "question": self.question,
            "grid": self.grid,
            "store": self.store,
            "cache": self.cache,
            "outputs": self.outputs,
            "mode": self.mode,
            "seeds": dict(self.seeds),
            "params": copy.deepcopy(dict(self.params)),
        }

    def sha256(self) -> str:
        return hashlib.sha256(canonical_json(self.to_mapping()).encode()).hexdigest()

    def path(self, key: str) -> Path | None:
        """The resolved path of ``grid`` / ``store`` / ``cache`` / ``outputs``."""
        if key not in PATH_KEYS:
            raise KeyError(key)
        v = getattr(self, key)
        return None if v is None else resolve_repo_path(v)


@dataclass(frozen=True)
class ConfigOverrides:
    """CLI overrides: path flags (cwd-relative), ``--fast`` and ``--set``."""

    grid: str | None = None
    store: str | None = None
    cache: str | None = None
    outputs: str | None = None
    fast: bool = False
    sets: tuple[str, ...] = ()


def _set_param(params: dict[str, Any], assignment: str) -> None:
    key, sep, raw = assignment.partition("=")
    if not sep or not key.strip():
        raise ConfigError(f"--set {assignment!r}: expected key.sub=value")
    if not raw.strip():
        raise ConfigError(f"--set {assignment!r}: empty value (write null explicitly)")
    parts = [p.strip() for p in key.strip().split(".")]
    if not all(parts):
        raise ConfigError(f"--set {assignment!r}: empty key segment")
    node: Any = params
    for i, part in enumerate(parts[:-1]):
        if not isinstance(node, dict) or part not in node or not isinstance(node[part], dict):
            where = ".".join(parts[: i + 1])
            raise ConfigError(f"--set {assignment!r}: params.{where} is not an existing mapping")
        node = node[part]
    node[parts[-1]] = load_yaml_strict(raw, f"--set {assignment!r}")


def apply_overrides(config: StudyConfig, overrides: ConfigOverrides | None) -> StudyConfig:
    """``config`` with the CLI overrides applied (module docstring)."""
    if overrides is None:
        return config
    data = config.to_mapping()
    for key in PATH_KEYS:
        v = getattr(overrides, key)
        if v is not None:
            data[key] = display_path(Path(v).expanduser().absolute())
    if overrides.fast:
        data["mode"] = "fast"
    for s in overrides.sets:
        _set_param(data["params"], s)
    return StudyConfig.from_mapping(data, source="study config with overrides")


def load_study_config(path: str | Path, overrides: ConfigOverrides | None = None) -> StudyConfig:
    """Strict YAML (:class:`StrictLoader`) → :class:`StudyConfig`, with the CLI ``overrides``
    applied."""
    p = Path(path)
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"{p}: {exc}") from exc
    raw = load_yaml_strict(text, str(p))
    if raw is None:
        raise ConfigError(f"{p}: empty file")
    return apply_overrides(StudyConfig.from_mapping(raw, source=str(p)), overrides)


# --------------------------------------------------------------------------------------------
# runner modules
# --------------------------------------------------------------------------------------------


def load_runner_module(name: str) -> ModuleType:
    """Import ``name`` and check the protocol attributes (module docstring)."""
    try:
        module = importlib.import_module(name)
    except ImportError as exc:
        raise ConfigError(f"runner module {name!r} cannot be imported: {exc}") from exc
    missing = [a for a in MODULE_ATTRIBUTES if not hasattr(module, a)]
    if missing:
        raise ConfigError(f"runner module {name!r} lacks {missing}")
    for a in ("TITLE", "QUESTION"):
        v = getattr(module, a)
        if not isinstance(v, str) or not v.strip():
            raise ConfigError(f"runner module {name!r}: {a} must be a non-empty string")
    for a in ("REQUIRED_PARAMS", "OPTIONAL_PARAMS"):
        v = getattr(module, a, ())
        if not isinstance(v, (tuple, list)) or not all(isinstance(x, str) for x in v):
            raise ConfigError(f"runner module {name!r}: {a} must be a tuple of strings")
    for a in MODULE_ATTRIBUTES[3:]:
        if not callable(getattr(module, a)):
            raise ConfigError(f"runner module {name!r}: {a} is not callable")
    return module


def validate_params(module: ModuleType, params: Mapping[str, Any]) -> None:
    """Missing required and undeclared params are errors; then the module's own check."""
    required = tuple(module.REQUIRED_PARAMS)
    optional = tuple(getattr(module, "OPTIONAL_PARAMS", ()))
    missing = [k for k in required if k not in params]
    if missing:
        raise ConfigError(f"{module.__name__}: missing required params {missing}")
    unknown = sorted(set(params) - set(required) - set(optional))
    if unknown:
        raise ConfigError(
            f"{module.__name__}: unknown params {unknown}; required {list(required)}, "
            f"optional {list(optional)}"
        )
    check = getattr(module, "validate_params", None)
    if callable(check):
        check(params)


def study_question(config: StudyConfig, module: ModuleType) -> str:
    q = (config.question if config.question is not None else str(module.QUESTION)).strip()
    if not q or "\n" in q:
        raise ConfigError(f"study {config.name!r}: the question must be one non-empty line")
    return q


# --------------------------------------------------------------------------------------------
# requirements and the read-only cache
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Requirement:
    """Something ``compute`` needs that the runner does not produce (module docstring)."""

    kind: str
    id: str
    what: str
    command: str

    def __post_init__(self) -> None:
        if self.kind not in REQUIREMENT_KINDS:
            raise ValueError(f"requirement kind {self.kind!r} must be one of {REQUIREMENT_KINDS}")
        if not self.id:
            raise ValueError("a requirement needs an id")


class MissingRequirements(Exception):  # noqa: N818 — the contract's name
    """Requirements absent from the store / cache / outputs; ``commands`` produce them."""

    def __init__(self, missing: Sequence[Requirement], commands: Sequence[str]) -> None:
        self.missing = list(missing)
        self.commands = list(commands)
        super().__init__(format_missing(self.missing, self.commands))


def format_missing(missing: Sequence[Requirement], commands: Sequence[str]) -> str:
    lines = [f"{len(missing)} requirement(s) missing:"]
    lines += [f"  - {r.kind} {r.id}: {r.what}" for r in missing]
    lines.append("produce them with:")
    lines += [f"  {c}" for c in commands]
    return "\n".join(lines)


class StudyCacheMissError(CacheMissError):
    """A miss of the study's read-only cache (the runner reports it as a missing leverage)."""

    def __init__(self, key: str, spec: CalibrationSpec) -> None:
        super().__init__(f"no calibrated leverage for key {key}")
        self.key = key
        self.spec = spec


def model_label(spec: CalibrationSpec) -> str:
    m = spec.model
    return (
        f"nu={m.nu:g} theta={m.theta:g} k1={m.k1:g} k2={m.k2:g} rho12={m.rho12:g} "
        f"rho_SX1={m.rho_SX1:g} rho_SX2={m.rho_SX2:g}"
    )


class ReadOnlyLeverageCache(LeverageCache):
    """The only cache a study sees: a hit is reported to ``on_hit(key, spec)`` (the manifest's
    ``cache_keys``), a miss raises :class:`StudyCacheMissError`, every writer raises
    :class:`CalibrationForbiddenError` (module docstring)."""

    def __init__(
        self,
        root: str | Path,
        on_hit: Callable[[str, CalibrationSpec], None] | None = None,
    ) -> None:
        super().__init__(root)
        self._on_hit = on_hit

    def get_or_calibrate(
        self,
        spec: CalibrationSpec,
        *,
        allow_calibrate: bool = True,
        run_diagnostics: bool = False,
        diagnostics_sim: SimConfig | None = None,
    ) -> tuple[LSV, CalibrationReport | None]:
        """:meth:`LeverageCache.get_or_calibrate` with ``allow_calibrate=False`` whatever the
        caller passed; the hit is recorded."""
        key = self.key(spec)
        try:
            out = super().get_or_calibrate(
                spec, allow_calibrate=False, run_diagnostics=False, diagnostics_sim=None
            )
        except CacheMissError as exc:
            raise StudyCacheMissError(key, spec) from exc
        if self._on_hit is not None:
            self._on_hit(key, spec)
        return out

    def _refuse(self, what: str) -> NoReturn:
        raise CalibrationForbiddenError(
            f"a study never calibrates or writes the leverage cache ({what} refused on "
            f"{self.root}); run volsto-precompute for the missing leverage"
        )

    def store(
        self,
        spec: CalibrationSpec,
        leverage: LeverageFunction,
        report: CalibrationReport | None = None,
    ) -> NoReturn:
        self._refuse("store")

    def write_report(self, spec: CalibrationSpec, report: CalibrationReport) -> NoReturn:
        self._refuse("write_report")

    def merge_manifest(self, rows: pd.DataFrame) -> NoReturn:
        self._refuse("merge_manifest")

    def _update_manifest(self, update: Any) -> NoReturn:
        self._refuse("manifest update")

    def calibrate(self, *args: object, **kwargs: object) -> NoReturn:
        self._refuse("calibrate")

    def put(self, *args: object, **kwargs: object) -> NoReturn:
        self._refuse("put")


class RecordingStoreReader(StoreReader):
    """A :class:`~volsto.viewers.store.StoreReader` that records the point ids it returns
    (module docstring, ``store_points``)."""

    def __init__(self, root: str | Path, sink: dict[str, None]) -> None:
        super().__init__(root)
        self._sink = sink

    def point(self, point_id: str) -> dict[str, Any]:
        rec = super().point(point_id)
        self._sink.setdefault(point_id, None)
        return rec

    def _filtered(self, name: str, point_id: str | None) -> pd.DataFrame:
        df = super()._filtered(name, point_id)
        if point_id is not None:
            self._sink.setdefault(point_id, None)
        elif not df.empty and "point_id" in df.columns:
            for pid in dict.fromkeys(str(p) for p in df["point_id"]):
                self._sink.setdefault(pid, None)
        return df


def _entry_particles(cache_root: Path, key: str) -> int | None:
    """``n_particles`` of a cache entry from its ``spec.json`` (``None`` when unreadable)."""
    try:
        data = json.loads((cache_root / key / "spec.json").read_text())
        return int(data["particle"]["n_particles"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def artefact_path(outputs_root: Path, rel: str) -> Path:
    """An artefact id under the outputs root (a leading ``outputs/`` names that root)."""
    r = rel[len(OUTPUTS_PREFIX) :] if rel.startswith(OUTPUTS_PREFIX) else rel
    p = Path(r)
    if p.is_absolute() or ".." in p.parts:
        raise ConfigError(f"artefact {rel!r} must be a relative path under the outputs root")
    return outputs_root / p


# --------------------------------------------------------------------------------------------
# cache observation
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class CacheSnapshot:
    """What :func:`snapshot_cache` saw: ``entries`` maps each entry directory holding a
    ``leverage.npz`` to that file's ``(size, mtime_ns)``; ``manifest_rows`` is the row count of
    the cache manifest (0 without one)."""

    root: str
    entries: Mapping[str, tuple[int, int]]
    manifest_rows: int

    def summary(self) -> dict[str, int]:
        return {"entries": len(self.entries), "manifest_rows": self.manifest_rows}


def snapshot_cache(root: Path) -> CacheSnapshot:
    """A cheap snapshot of a leverage cache root (directory listing and ``stat``; the manifest's
    parquet footer) — nothing is created when the root is missing."""
    entries: dict[str, tuple[int, int]] = {}
    rows = 0
    if root.is_dir():
        for d in root.iterdir():
            npz = d / LEVERAGE_NAME
            try:
                st = npz.stat()
            except OSError:
                continue
            entries[d.name] = (int(st.st_size), int(st.st_mtime_ns))
        manifest = root / CACHE_MANIFEST_NAME
        if manifest.is_file():
            import pyarrow.parquet as pq

            try:
                rows = int(pq.read_metadata(manifest).num_rows)
            except (OSError, ValueError):
                rows = -1
    return CacheSnapshot(str(root), entries, rows)


def cache_growth(before: CacheSnapshot, after: CacheSnapshot) -> dict[str, Any]:
    """What changed between two snapshots: new keys (with their completeness), rewritten keys
    and manifest rows added; ``grew`` is the verdict."""
    new = sorted(k for k in after.entries if k not in before.entries)
    rewritten = sorted(
        k for k in after.entries if k in before.entries and after.entries[k] != before.entries[k]
    )
    added_rows = max(after.manifest_rows - before.manifest_rows, 0)
    root = Path(after.root)
    return {
        "new_keys": new,
        "new_keys_complete": [k for k in new if has_complete_leverage(root, k)],
        "rewritten_keys": rewritten,
        "manifest_rows_added": added_rows,
        "grew": bool(new or rewritten or added_rows),
    }


class CacheObserver:
    """The snapshots of one run (module docstring): ``mark(stage)`` snapshots, ``growth`` is the
    cumulative change since the first snapshot."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.first = snapshot_cache(root)
        self.stages: dict[str, dict[str, int]] = {"before": self.first.summary()}
        self.growth: dict[str, Any] = cache_growth(self.first, self.first)

    def mark(self, stage: str) -> dict[str, Any]:
        snap = snapshot_cache(self.root)
        self.stages[stage] = snap.summary()
        self.growth = cache_growth(self.first, snap)
        return self.growth

    @property
    def changed(self) -> bool:
        return bool(self.growth["grew"])

    def record(self) -> dict[str, Any]:
        """The manifest's ``cache_observation``."""
        return {
            "root": display_path(self.root),
            "stages": dict(self.stages),
            **{k: v for k, v in self.growth.items() if k != "grew"},
            "attribution": UNATTRIBUTED if self.changed else "",
        }

    def describe(self) -> str:
        g = self.growth
        parts = []
        if g["new_keys"]:
            parts.append(f"new entries {[k[:16] for k in g['new_keys']]}")
        if g["rewritten_keys"]:
            parts.append(f"rewritten entries {[k[:16] for k in g['rewritten_keys']]}")
        if g["manifest_rows_added"]:
            parts.append(f"{g['manifest_rows_added']} manifest row(s) added")
        return "; ".join(parts) or "no change"


#: What the runner can say about who changed the cache: the leverage cache records no writer
#: identity (no pid / host in ``spec.json``, the leverage metadata or the manifest columns).
UNATTRIBUTED = (
    "not attributed: the leverage cache records no writer identity (another process — a "
    "volsto-precompute, a study C refit — may be writing into the same root)"
)
#: Manifest keys the provenance block shows and a late observation can change.
WATCH_KEYS: tuple[str, ...] = (
    "recalibrated",
    "calibrations_started",
    "calibration_refusals",
    "cache_changed_during_run",
    "cache_observation",
)


class RunWatch:
    """What a run observes about calibration (module docstring): the guard's marker files of
    this process and its children in a per-run temporary directory (``recalibrated`` = a
    calibration started; ``calibration_refusals``) and the cache snapshots (``cache changed
    during the run``, not attributed)."""

    def __init__(self, cache_root: Path) -> None:
        self.observer = CacheObserver(cache_root)
        self.marker_dir = Path(tempfile.mkdtemp(prefix="volsto-study-markers-"))
        self.started: list[dict[str, Any]] = []
        self.refused: list[dict[str, Any]] = []

    def forbid(self) -> contextlib.AbstractContextManager[None]:
        """The calibration prohibition of the run, with its marker directory."""
        return calibration_forbidden(self.marker_dir)

    def _read(self) -> None:
        if self.marker_dir.is_dir():
            markers = guard.read_markers(self.marker_dir)
            self.started, self.refused = markers[guard.STARTED], markers[guard.REFUSED]

    def mark(self, stage: str) -> None:
        self.observer.mark(stage)
        self._read()

    @property
    def recalibrated(self) -> bool:
        return bool(self.started)

    @property
    def cache_changed(self) -> bool:
        return self.observer.changed

    def fields(self) -> dict[str, Any]:
        """The manifest entries (:data:`WATCH_KEYS`)."""
        return {
            "recalibrated": self.recalibrated,
            "calibrations_started": list(self.started),
            "calibration_refusals": len(self.refused),
            "cache_changed_during_run": self.cache_changed,
            "cache_observation": self.observer.record(),
        }

    def close(self) -> None:
        self._read()
        shutil.rmtree(self.marker_dir, ignore_errors=True)

    def report(self, name: str) -> None:
        if self.recalibrated:
            who = sorted({f"pid {r.get('pid')} on {r.get('host')}" for r in self.started})
            log.error(
                "%s: RECALIBRATED — %d leverage calibration(s) started during the run (%s): a "
                "process escaped the prohibition; recalibrated: true is recorded and the run fails",
                name,
                len(self.started),
                ", ".join(who),
            )
        if self.refused:
            log.error(
                "%s: %d leverage calibration(s) were attempted and refused during the run (in this "
                "process or a child; caught by the study if the run went on); the run fails",
                name,
                len(self.refused),
            )
        if self.cache_changed:
            log.warning(
                "%s: the leverage cache %s changed during the run (%s), %s",
                name,
                self.observer.root,
                self.observer.describe(),
                UNATTRIBUTED,
            )

    def summary(self) -> dict[str, Any]:
        return {
            "recalibrated": self.recalibrated,
            "refusals": len(self.refused),
            "cache_changed": self.cache_changed,
            "cache": self.observer.record(),
        }

    @property
    def failed(self) -> bool:
        return self.recalibrated or bool(self.refused)


# --------------------------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------------------------

NO_GRID_NOTE = (
    "# {kind} {id} ({what}): the study has no grid (grid: null), so no volsto-precompute line "
    "can produce it; set `grid:` to a grid holding this point / spec (extra_points) and rerun"
)


class StudyContext:
    """What ``requirements`` and ``compute`` may use (module docstring); it records what was
    read for the manifest."""

    def __init__(self, config: StudyConfig, *, out_dir: Path) -> None:
        self.config = config
        self.out_dir = out_dir
        self.repo_root = REPO_ROOT
        self.log = logging.getLogger(f"volsto.studies.{config.name}")
        self._params: Mapping[str, Any] = MappingProxyType(copy.deepcopy(dict(config.params)))
        self._cache: ReadOnlyLeverageCache | None = None
        self._grid: GridSpec | None = None
        self._grid_loaded = False
        self._points: list[GridPoint] | None = None
        self._store_ids: set[str] | None = None
        self.cache_keys: dict[str, dict[str, Any]] = {}
        self.store_points: dict[str, None] = {}
        self.artefacts: dict[str, str] = {}
        self.records: dict[str, Any] = {}

    # -- configuration -----------------------------------------------------------------------

    @property
    def params(self) -> Mapping[str, Any]:
        return self._params

    @property
    def mode(self) -> str:
        return self.config.mode

    @property
    def grid_path(self) -> Path | None:
        return self.config.path("grid")

    @property
    def store_root(self) -> Path:
        p = self.config.path("store")
        assert p is not None
        return p

    @property
    def cache_root(self) -> Path:
        p = self.config.path("cache")
        assert p is not None
        return p

    @property
    def outputs_root(self) -> Path:
        p = self.config.path("outputs")
        assert p is not None
        return p

    def seed(self, name: str) -> int:
        """The config's seed ``name`` (a missing seed is an error, never a default)."""
        if name not in self.config.seeds:
            raise ConfigError(
                f"study {self.config.name!r}: no seed {name!r} in the config "
                f"(seeds: {sorted(self.config.seeds)})"
            )
        return int(self.config.seeds[name])

    def record(self, key: str, value: Any) -> None:
        """An extra manifest entry (``n_paths`` becomes the manifest's ``n_paths``)."""
        self.records[key] = value

    # -- cache -------------------------------------------------------------------------------

    @property
    def cache(self) -> ReadOnlyLeverageCache:
        """The read-only cache; every hit through it is recorded in ``cache_keys``."""
        if self._cache is None:
            self._cache = ReadOnlyLeverageCache(self.cache_root, on_hit=self._cache_hit)
        return self._cache

    def _cache_hit(self, key: str, spec: CalibrationSpec) -> None:
        self._note_key(key, f"ctx.cache ({model_label(spec)})", spec.particle.n_particles)

    def _note_key(
        self, key: str, what: str, n_particles: int | None, *, override: bool = False
    ) -> None:
        entry = {"key": key, "what": what, "n_particles": n_particles}
        if override or key not in self.cache_keys:
            self.cache_keys[key] = entry

    def leverage(self, spec: CalibrationSpec, what: str) -> LSV:
        """The cached LSV of ``spec`` (a miss raises :class:`MissingRequirements` naming the
        precompute line); the key is recorded in the manifest under ``what``."""
        try:
            model, _ = self.cache.get_or_calibrate(spec, allow_calibrate=False)
        except CacheMissError as exc:
            req = self.leverage_requirement(spec, what)
            raise MissingRequirements([req], self.commands_for([req])) from exc
        self._note_key(spec_key(spec), what, spec.particle.n_particles, override=True)
        return model

    # -- grid and store ----------------------------------------------------------------------

    def grid(self) -> GridSpec | None:
        if not self._grid_loaded:
            path = self.grid_path
            self._grid = None if path is None else load_grid(path)
            self._grid_loaded = True
        return self._grid

    def grid_points(self) -> list[GridPoint]:
        """The configured grid's points (``volsto.viewers.grid.enumerate_points``)."""
        if self._points is None:
            g = self.grid()
            if g is None:
                raise ConfigError(f"study {self.config.name!r} has no grid (grid: null)")
            self._points = enumerate_points(g)
        return list(self._points)

    def store(self) -> StoreReader:
        """A reader of the configured store that records the points it returns."""
        return RecordingStoreReader(self.store_root, self.store_points)

    def artefact(self, rel: str) -> Path:
        """The path of an existing artefact (recorded with its SHA-256); a missing one raises
        ``FileNotFoundError`` — declare it in ``requirements`` to get the command printed."""
        p = artefact_path(self.outputs_root, rel)
        if not p.is_file():
            raise FileNotFoundError(f"artefact {rel!r} not found at {p}")
        self.artefacts.setdefault(rel, file_sha256(p))
        return p

    # -- requirements ------------------------------------------------------------------------

    def precompute_line(self, ids: Sequence[str]) -> str:
        """The runnable ``volsto-precompute`` line producing ``ids`` from the **configured**
        grid, every path relative to the current directory (or absolute outside it); the
        ``--only ... --resume`` form of :func:`volsto.viewers.api.precompute_command`."""
        if self.grid_path is None:
            raise ConfigError(
                f"study {self.config.name!r} has no grid (grid: null): no precompute line exists"
            )
        return (
            f"volsto-precompute --grid {cwd_path(self.grid_path)} "
            f"--store {cwd_path(self.store_root)} --cache {cwd_path(self.cache_root)} "
            f"--only {' '.join(ids)} --resume"
        )

    def point_requirement(self, point_id: str, what: str) -> Requirement:
        """A store point of the configured grid (``grid: null`` is a config error)."""
        if self.grid_path is None:
            raise ConfigError(
                f"study {self.config.name!r}: a point requirement ({point_id}) needs a grid "
                "(grid: null); set `grid:` to the grid the store was built from"
            )
        return Requirement("point", point_id, what, self.precompute_line([point_id]))

    def leverage_requirement(self, spec: CalibrationSpec | str, what: str) -> Requirement:
        key = spec if isinstance(spec, str) else spec_key(spec)
        if self.grid_path is None:
            command = NO_GRID_NOTE.format(kind="leverage", id=key, what=what)
        else:
            command = self.precompute_line([self._point_id_of(key)])
        return Requirement("leverage", key, what, command)

    def artefact_requirement(self, path: str, what: str, command: str) -> Requirement:
        return Requirement("artefact", path, what, command)

    def _point_id_of(self, key: str) -> str:
        """The grid point computing leverage ``key`` (its id; 1F / 2F ids are the key)."""
        if self.grid_path is None:
            return key
        for p in self.grid_points():
            if p.cache_key == key:
                return p.id
        return key

    def _stored_ids(self) -> set[str]:
        if self._store_ids is None:
            df = StoreReader(self.store_root).points()
            self._store_ids = set() if df.empty else {str(i) for i in df["point_id"]}
        return self._store_ids

    def is_present(self, req: Requirement) -> bool:
        if req.kind == "point":
            return req.id in self._stored_ids()
        if req.kind == "leverage":
            return has_complete_leverage(self.cache_root, req.id)
        return artefact_path(self.outputs_root, req.id).is_file()

    def missing(self, reqs: Sequence[Requirement]) -> list[Requirement]:
        """The requirements not satisfied by the configured store / cache / outputs."""
        return [r for r in reqs if not self.is_present(r)]

    def _recorded_grid_note(self) -> str | None:
        """A note when the store's run records name another grid than the configured one."""
        if self.grid_path is None:
            return None
        recorded = StoreReader(self.store_root).manifest().get("grid_path")
        configured = display_path(self.grid_path)
        if isinstance(recorded, str) and recorded and recorded != configured:
            return (
                f"# note: the store {cwd_path(self.store_root)} was built from grid {recorded} "
                f"(its run records); the ids above are checked against the configured grid "
                f"{configured}, which the line names"
            )
        return None

    def commands_for(self, missing: Sequence[Requirement]) -> list[str]:
        """The commands to print (module docstring)."""
        commands: list[str] = []
        pre = [r for r in missing if r.kind in ("point", "leverage")]
        if pre and self.grid_path is None:
            commands += [NO_GRID_NOTE.format(kind=r.kind, id=r.id, what=r.what) for r in pre]
        elif pre:
            grid_ids = {p.id for p in self.grid_points()}
            ids = list(
                dict.fromkeys(r.id if r.kind == "point" else self._point_id_of(r.id) for r in pre)
            )
            inside = [i for i in ids if i in grid_ids]
            outside = [i for i in ids if i not in grid_ids]
            if inside:
                commands.append(self.precompute_line(inside))
            grid_name = cwd_path(self.grid_path) if self.grid_path is not None else "<none>"
            for i in outside:
                commands.append(
                    f"# {i} is not a point of {grid_name}: add it to that grid YAML first, "
                    f"then run: {self.precompute_line([i])}"
                )
            note = self._recorded_grid_note()
            if note is not None:
                commands.append(note)
        for r in missing:
            if r.kind == "artefact" and r.command not in commands:
                commands.append(r.command)
        return commands

    def require(self, reqs: Sequence[Requirement]) -> None:
        """Record the present requirements; raise :class:`MissingRequirements` if any is
        missing."""
        for r in reqs:
            if not isinstance(r, Requirement):
                raise TypeError(f"requirements() must return Requirement records, got {r!r}")
            if r.kind == "point" and self.grid_path is None:
                raise ConfigError(
                    f"study {self.config.name!r}: a point requirement ({r.id}) needs a grid "
                    "(grid: null)"
                )
        missing = self.missing(reqs)
        if missing:
            raise MissingRequirements(missing, self.commands_for(missing))
        for r in reqs:
            if r.kind == "point":
                self.store_points.setdefault(r.id, None)
            elif r.kind == "leverage":
                self._note_key(r.id, r.what, _entry_particles(self.cache_root, r.id))
            else:
                self.artefacts.setdefault(r.id, file_sha256(artefact_path(self.outputs_root, r.id)))

    def store_point_keys(self) -> list[dict[str, Any]]:
        """``cache_keys`` entries of the leverages behind the store points read."""
        if not self.store_points:
            return []
        df = StoreReader(self.store_root).points()
        if df.empty:
            return []
        out: list[dict[str, Any]] = []
        rows = df[df["point_id"].astype(str).isin(set(self.store_points))]
        for rec in rows.to_dict("records"):
            key = rec.get("cache_key")
            if not isinstance(key, str) or not key:
                continue
            n = rec.get("n_particles")
            n_particles = None if n is None or pd.isna(n) else int(n)
            out.append(
                {"key": key, "what": f"store point {rec.get('label')}", "n_particles": n_particles}
            )
        return out


# --------------------------------------------------------------------------------------------
# render
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Rendered:
    """What :func:`render_outputs` produced: the declared tables and figures, the table files
    written (the parts of :func:`volsto.studies.latex.split_table`), the wall clock."""

    tables: tuple[TableSpec, ...]
    figures: tuple[FigureSpec, ...]
    seconds: float
    table_files: tuple[str, ...] = ()


def _clean_stale(directory: Path, keep: set[str], suffixes: tuple[str, ...]) -> None:
    if not directory.is_dir():
        return
    for p in directory.iterdir():
        if p.is_file() and p.suffix in suffixes and p.name not in keep:
            p.unlink()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write(path, lambda p: p.write_text(text, encoding="utf-8"))


def render_outputs(module: ModuleType, results: Results, out_dir: Path) -> Rendered:
    """``tables/*.tex`` and ``figures/*.{pdf,png}`` from ``results`` alone (stale files of
    earlier renders removed)."""
    t0 = time.perf_counter()
    tables = tuple(module.tables(results))
    figures = tuple(module.figures(results))
    for spec in tables:
        if not isinstance(spec, TableSpec):
            raise TypeError(f"tables() must return TableSpec, got {spec!r}")
    for fspec in figures:
        if not isinstance(fspec, FigureSpec):
            raise TypeError(f"figures() must return FigureSpec, got {fspec!r}")
    for kind, names in (("table", [t.name for t in tables]), ("figure", [f.name for f in figures])):
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate {kind} names in {module.__name__}: {names}")
    tdir, fdir = out_dir / TABLES_DIR, out_dir / FIGURES_DIR
    parts, _ = latex.expand_tables(tables, results)
    for spec in parts:
        _write_text(tdir / f"{spec.name}.tex", latex.latex_table(spec, results))
    if figures:
        from volsto.studies import style

        for fspec in figures:
            fig = fspec.draw(results)
            style.save_figure(fig, fdir, fspec.name)
    _clean_stale(tdir, {f"{t.name}.tex" for t in parts}, (".tex",))
    _clean_stale(
        fdir, {f"{f.name}{s}" for f in figures for s in (".pdf", ".png")}, (".pdf", ".png")
    )
    return Rendered(tables, figures, time.perf_counter() - t0, tuple(p.name for p in parts))


def provenance_items(manifest: Mapping[str, Any]) -> list[tuple[str, str]]:
    """The provenance block of ``study.md`` / ``study.tex`` (from the stored manifest only)."""
    wc = manifest["wall_clock"]
    keys = manifest.get("cache_keys") or []
    shown = ", ".join(str(k["key"])[:12] for k in keys[:6]) + (" ..." if len(keys) > 6 else "")
    seeds = ", ".join(f"{k}={v}" for k, v in sorted(manifest.get("seeds", {}).items()))
    grid = manifest.get("grid")
    commit = manifest.get("git_commit") or "unknown"
    dirty = manifest.get("git_dirty")
    arts = manifest.get("artefacts") or []
    obs = manifest.get("cache_observation") or {}
    started = manifest.get("calibrations_started") or []
    recal = "no"
    if manifest["recalibrated"] is not False:
        pids = sorted({str(r.get("pid")) for r in started})
        recal = f"YES: {len(started)} calibration(s) started (pid {', '.join(pids)})"
    changed_keys = list(obs.get("new_keys", [])) + list(obs.get("rewritten_keys", []))
    cache_changed = "no"
    if manifest.get("cache_changed_during_run"):
        cache_changed = (
            f"{len(obs.get('new_keys', []))} new, {len(obs.get('rewritten_keys', []))} rewritten "
            f"entries, {obs.get('manifest_rows_added', 0)} manifest rows; not attributed"
            + (f" ({', '.join(k[:12] for k in changed_keys[:6])})" if changed_keys else "")
        )
    items = [
        ("study", str(manifest["study"])),
        ("runner", str(manifest["runner"])),
        ("config", f"{manifest.get('config_path') or '(resolved mapping)'}"),
        ("config sha256", str(manifest["config_sha256"])[:16]),
        ("mode", str(manifest["mode"])),
        (
            "wall clock",
            f"total {wc['total']:.2f} s (requirements {wc['requirements']:.2f} s, "
            f"compute {wc['compute']:.2f} s, render {wc['render']:.2f} s)",
        ),
        ("recalibrated", recal),
        ("cache changed during the run", cache_changed),
        (
            "git commit",
            f"{str(commit)[:12]}"
            + (" (uncommitted changes)" if dirty else "" if dirty is False else " (state unknown)"),
        ),
        ("calibration code tag", str(manifest["calibration_code_tag"])),
        ("particles", ", ".join(str(p) for p in manifest.get("particles", [])) or "none"),
        ("seeds", seeds or "none"),
        ("paths", "not recorded" if manifest.get("n_paths") is None else str(manifest["n_paths"])),
        ("leverage cache keys", f"{len(keys)}" + (f": {shown}" if keys else "")),
        ("store points", str(len(manifest.get("store_points") or []))),
        ("artefacts", ", ".join(str(a["path"]) for a in arts) or "none"),
        ("grid", "none" if not grid else f"{grid['name']} ({grid['path']})"),
        ("created (UTC)", str(manifest["created_utc"])),
        ("results sha256", str(manifest["results_sha256"])[:16]),
    ]
    if manifest.get("calibration_refusals"):
        items.append(("calibration refusals", str(manifest["calibration_refusals"])))
    if manifest.get("rerun_of"):
        items.append(("rerun of", str(manifest["rerun_of"])))
    return items


def write_documents(
    module: ModuleType,
    results: Results,
    out_dir: Path,
    manifest: Mapping[str, Any],
    rendered: Rendered,
) -> list[Path]:
    """``study.md`` and ``study.tex`` (title, question and mode from the manifest: a fast-mode
    study carries :data:`volsto.studies.latex.FAST_MODE_BANNER` under its question, placed here
    once for every module)."""
    narrative = str(module.narrative(results))
    prov = provenance_items(manifest)
    md = latex.study_markdown(
        title=str(manifest["title"]),
        question=str(manifest["question"]),
        narrative=narrative,
        tables=rendered.tables,
        figures=rendered.figures,
        results=results,
        provenance=prov,
        mode=str(manifest["mode"]),
    )
    tex = latex.study_tex(
        title=str(manifest["title"]),
        question=str(manifest["question"]),
        narrative=narrative,
        tables=rendered.tables,
        figures=rendered.figures,
        provenance=prov,
        date=str(manifest["created_utc"])[:10],
        results=results,
        mode=str(manifest["mode"]),
    )
    paths = [out_dir / STUDY_MD, out_dir / STUDY_TEX]
    _write_text(paths[0], md)
    _write_text(paths[1], tex)
    return paths


def remove_stale_pdf(out_dir: Path) -> list[str]:
    """Remove ``study.pdf`` / ``study.log`` (the LaTeX check is off, so they would describe an
    older ``study.tex``); the names removed."""
    removed = []
    for name in (STUDY_PDF, STUDY_LOG):
        p = out_dir / name
        if p.exists():
            p.unlink()
            removed.append(name)
    if removed:
        log.warning(
            "%s: removed the stale %s of an earlier run (the LaTeX check is off)",
            out_dir,
            " and ".join(removed),
        )
    return removed


def _latex_step(out_dir: Path, name: str) -> tuple[dict[str, Any], int]:
    check = compile_latex(out_dir, STUDY_TEX)
    if check["status"] == "failed":
        log.error("%s: the LaTeX check failed: %s", name, check["reason"])
        return check, EXIT_FAILED
    if check["status"] == "skipped":
        log.warning("%s: LaTeX check skipped: %s", name, check["reason"])
    else:
        log.info("%s: study.tex compiled and checked in %.1f s", name, check["seconds"])
    return check, EXIT_OK


# --------------------------------------------------------------------------------------------
# run / rerun / render
# --------------------------------------------------------------------------------------------


@dataclass
class StudyRun:
    """A finished run: the directory, the results, the manifest written, the exit status."""

    out_dir: Path
    results: Results
    manifest: dict[str, Any]
    exit_code: int = EXIT_OK
    rendered: Rendered | None = field(default=None, repr=False)


def _versions() -> dict[str, Any]:
    import numba

    try:
        import matplotlib

        mpl: str | None = matplotlib.__version__
    except ImportError:
        mpl = None
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "numba": numba.__version__,
        "numba_threads": int(numba.get_num_threads()),  # type: ignore[no-untyped-call]
        "pandas": pd.__version__,
        "matplotlib": mpl,
    }


def study_argv(argv: Sequence[str], grid: Path | None) -> list[str]:
    """``argv`` for the manifest: ``--store`` / ``--cache`` masked and ``--grid`` repo-relative
    (:func:`volsto.viewers.precompute.provenance_argv`); ``--outputs`` / ``--out`` and
    positional paths repo-relative, or masked when outside the repository."""
    root = REPO_ROOT.resolve()

    def rel(value: str, mask: str) -> str:
        try:
            return str(Path(value).expanduser().resolve().relative_to(root))
        except (ValueError, OSError):
            return mask

    marks = {"--outputs": "<outputs>", "--out": "<out>"}
    verbatim = {"--set", "--nse", "--store", "--cache", "--grid"}  # values kept for now
    out: list[str] = []
    pending: str | None = None
    keep_next = False
    for a in argv:
        if keep_next:
            out.append(a)
            keep_next = False
            continue
        if pending is not None:
            out.append(rel(a, pending))
            pending = None
            continue
        flag, eq, value = a.partition("=")
        if flag in marks:
            out.append(f"{flag}={rel(value, marks[flag])}" if eq else a)
            pending = None if eq else marks[flag]
            continue
        if flag in verbatim:
            out.append(a)
            keep_next = not eq
            continue
        if not a.startswith("-") and ("/" in a or a.endswith((".yaml", ".yml"))):
            out.append(rel(a, "<path>"))
            continue
        out.append(a)
    grid_display = "<grid>" if grid is None else display_path(grid)
    if grid_display.startswith("/"):
        grid_display = grid.name if grid is not None else "<grid>"
    return provenance_argv(out, grid_display)


def build_manifest(
    ctx: StudyContext,
    module: ModuleType,
    *,
    question: str,
    config_path: str | None,
    config_file_sha256: str | None,
    argv: Sequence[str],
    results: Results,
    results_path: Path,
    rendered: Rendered,
    wall_clock: Mapping[str, float],
    created_utc: str,
    watch: RunWatch,
    rerun_of: str | None = None,
) -> dict[str, Any]:
    """The manifest of a run (module docstring)."""
    config = ctx.config
    grid = ctx.grid()
    grid_path = ctx.grid_path
    for entry in ctx.store_point_keys():
        ctx._note_key(entry["key"], entry["what"], entry["n_particles"])
    keys = list(ctx.cache_keys.values())
    particles = sorted({int(k["n_particles"]) for k in keys if k["n_particles"] is not None})
    commit, dirty = git_state()
    records = dict(ctx.records)
    n_paths = records.pop("n_paths", None)
    runner_file = getattr(module, "__file__", None)
    manifest: dict[str, Any] = {
        "study": config.name,
        "runner": config.runner,
        "runner_sha256": file_sha256(runner_file) if runner_file else None,
        "title": str(module.TITLE),
        "question": question,
        "config": config.to_mapping(),
        "config_path": config_path,
        "config_sha256": config.sha256(),
        "config_file_sha256": config_file_sha256,
        "git_commit": commit,
        "git_dirty": dirty,
        "code_version": code_version(),
        "calibration_code_tag": CALIBRATION_CODE_TAG,
        "grid": (
            None
            if grid is None or grid_path is None
            else {
                "path": display_path(grid_path),
                "sha256": file_sha256(grid_path),
                "name": grid.name,
            }
        ),
        "cache_keys": keys,
        "store_points": sorted(ctx.store_points),
        "artefacts": [{"path": p, "sha256": s} for p, s in ctx.artefacts.items()],
        "particles": particles,
        "seeds": dict(config.seeds),
        "n_paths": n_paths,
        "records": records,
        "mode": config.mode,
        "tables": list(rendered.table_files),
        "table_specs": [t.name for t in rendered.tables],
        "figures": [f.name for f in rendered.figures],
        "n_results": len(results),
        "wall_clock": {k: float(v) for k, v in wall_clock.items()},
        **watch.fields(),
        "host": socket.gethostname(),
        **_versions(),
        "created_utc": created_utc,
        "argv": study_argv(argv, grid_path),
        "results_sha256": file_sha256(results_path),
        "latex": None,
        "profile": None,
    }
    if rerun_of is not None:
        manifest["rerun_of"] = rerun_of
    return manifest


def write_manifest(out_dir: Path, manifest: Mapping[str, Any]) -> Path:
    """Write ``manifest.json`` atomically (``recalibrated`` must be the observed boolean)."""
    if not isinstance(manifest["recalibrated"], bool):
        raise TypeError("manifest['recalibrated'] must be the observed boolean")
    text = json.dumps(manifest, indent=1, sort_keys=False, default=str) + "\n"
    path = out_dir / MANIFEST_NAME
    _write_text(path, text)
    return path


def read_manifest(output_dir: str | Path) -> dict[str, Any]:
    path = Path(output_dir) / MANIFEST_NAME
    if not path.is_file():
        raise ConfigError(f"{output_dir}: no {MANIFEST_NAME} (not a study output directory)")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: not a JSON object")
    return data


def _staging_dir(out: Path) -> Path:
    return out.parent / f".{out.name}{STAGING_MARK}{os.getpid()}-{secrets.token_hex(4)}"


def _promote(staging: Path, out: Path) -> None:
    """Move a finished staging directory into place (module docstring)."""
    if not out.exists():
        staging.rename(out)
        return
    for name in OWNED_OUTPUTS:
        p = out / name
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        elif p.exists() or p.is_symlink():
            p.unlink()
    for entry in list(staging.iterdir()):
        dest = out / entry.name
        if dest.is_dir() and not dest.is_symlink():
            shutil.rmtree(dest)
        elif dest.exists():
            dest.unlink()
        entry.rename(dest)
    staging.rmdir()


def run_study(
    config: StudyConfig,
    *,
    out_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    argv: Sequence[str] = (),
    profile: bool = False,
    latex_check: bool = True,
    rerun_of: str | None = None,
    config_path_display: str | None = None,
    observation: dict[str, Any] | None = None,
) -> StudyRun:
    """Run a study (module docstring).  Raises :class:`MissingRequirements` before ``compute``
    when a declared requirement is missing (and from ``compute`` for an undeclared one);
    ``observation`` (optional) receives ``recalibrated`` and the growth even when the run
    raises."""
    t0 = time.perf_counter()
    obs = observation if observation is not None else {}
    module = load_runner_module(config.runner)
    validate_params(module, config.params)
    question = study_question(config, module)
    out = (Path(out_dir) if out_dir is not None else DEFAULT_OUT_ROOT / config.name).expanduser()
    out = out.absolute()
    staging = _staging_dir(out)
    watch = RunWatch(Path(config.path("cache") or ""))
    obs.update(watch.summary())
    ctx = StudyContext(config, out_dir=staging)
    profiler = cProfile.Profile() if profile else None
    exit_code = EXIT_OK
    try:
        with watch.forbid():
            reqs = list(module.requirements(ctx))
            ctx.require(reqs)
            t1 = time.perf_counter()
            log.info(
                "%s: %d requirement(s) present; computing (mode %s)",
                config.name,
                len(reqs),
                config.mode,
            )
            staging.mkdir(parents=True)
            if profiler is not None:
                profiler.enable()
            try:
                results = module.compute(ctx)
            except StudyCacheMissError as exc:
                req = ctx.leverage_requirement(
                    exc.spec, f"read through ctx.cache by compute ({model_label(exc.spec)})"
                )
                raise MissingRequirements([req], ctx.commands_for([req])) from exc
            finally:
                if profiler is not None:
                    profiler.disable()
            watch.mark("after_compute")
            if not isinstance(results, Results):
                raise TypeError(
                    f"{config.runner}.compute must return Results, got {type(results)!r}"
                )
            results_path = results.write(staging / RESULTS_NAME)
            t2 = time.perf_counter()
            rendered = render_outputs(module, results, staging)
            t3 = time.perf_counter()
            watch.mark("after_render")
            wall = {
                "requirements": t1 - t0,
                "compute": t2 - t1,
                "render": t3 - t2,
                "total": t3 - t0,
            }
            cfg_file_sha = None
            if config_path is not None:
                cfg_file_sha = file_sha256(config_path)
                config_path_display = display_path(config_path)
            manifest = build_manifest(
                ctx,
                module,
                question=question,
                config_path=config_path_display,
                config_file_sha256=cfg_file_sha,
                argv=argv,
                results=results,
                results_path=results_path,
                rendered=rendered,
                wall_clock=wall,
                created_utc=utc_now(),
                watch=watch,
                rerun_of=rerun_of,
            )
            if profiler is not None:
                buf = io.StringIO()
                pstats.Stats(profiler, stream=buf).sort_stats("cumulative").print_stats(50)
                _write_text(staging / PROFILE_NAME, buf.getvalue())
                manifest["profile"] = PROFILE_NAME
            write_manifest(staging, manifest)
            write_documents(module, results, staging, manifest, rendered)
            if latex_check:
                manifest["latex"], exit_code = _latex_step(staging, config.name)
            watch.mark("final")
            fields = watch.fields()
            late = any(manifest[k] != fields[k] for k in WATCH_KEYS if k != "cache_observation")
            manifest.update(fields)
            if late:
                # an observation made after the documents: rewrite them (and the PDF)
                write_documents(module, results, staging, manifest, rendered)
                if latex_check:
                    manifest["latex"], exit_code = _latex_step(staging, config.name)
            write_manifest(staging, manifest)
        if not latex_check and out.exists():
            remove_stale_pdf(out)
        _promote(staging, out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        watch.mark("final")
        raise
    finally:
        watch.close()
        obs.update(watch.summary())
        watch.report(config.name)
    if watch.failed:
        exit_code = EXIT_FAILED
    log.info(
        "%s: %d numbers, %d tables, %d figures -> %s; wall clock %.2f s (requirements %.2f, "
        "compute %.2f, render %.2f); recalibrated: %s; cache changed during the run: %s",
        config.name,
        len(results),
        len(rendered.tables),
        len(rendered.figures),
        out,
        wall["total"],
        wall["requirements"],
        wall["compute"],
        wall["render"],
        "YES" if watch.recalibrated else "no",
        "yes (not attributed)" if watch.cache_changed else "no",
    )
    return StudyRun(out, results, manifest, exit_code, rendered)


def render_study(
    output_dir: str | Path,
    *,
    latex_check: bool = True,
    observation: dict[str, Any] | None = None,
) -> tuple[Rendered, int]:
    """Rebuild tables, figures, ``study.md`` and ``study.tex`` from ``results.parquet`` and the
    stored manifest (which is not modified), and recompile ``study.pdf`` when the LaTeX check is
    on (a stale PDF is removed otherwise).  Module code runs under the calibration guard and the
    cache observation.  Returns the rendered specs and the exit status."""
    t0 = time.perf_counter()
    obs = observation if observation is not None else {}
    out = Path(output_dir)
    manifest = read_manifest(out)
    results_path = out / RESULTS_NAME
    if file_sha256(results_path) != manifest.get("results_sha256"):
        log.warning("%s: results.parquet differs from the manifest's results_sha256", out)
    results = Results.read(results_path)
    module = load_runner_module(str(manifest["runner"]))
    config = StudyConfig.from_mapping(manifest["config"], source=f"{out / MANIFEST_NAME}:config")
    watch = RunWatch(Path(config.path("cache") or ""))
    code = EXIT_OK
    try:
        with watch.forbid():
            rendered = render_outputs(module, results, out)
            write_documents(module, results, out, manifest, rendered)
            if latex_check:
                _, code = _latex_step(out, str(manifest["study"]))
            else:
                remove_stale_pdf(out)
    finally:
        watch.mark("final")
        watch.close()
        obs.update(watch.summary())
        watch.report(str(manifest["study"]))
    if watch.failed:
        code = EXIT_FAILED
    log.info(
        "%s: rendered %d tables, %d figures from %s in %.2f s; recalibrated: %s",
        manifest["study"],
        len(rendered.tables),
        len(rendered.figures),
        results_path,
        time.perf_counter() - t0,
        "YES" if watch.recalibrated else "no",
    )
    return rendered, code


@dataclass
class RerunOutcome:
    """A rerun: the new run, the full diff, the changed rows and the exit status."""

    run: StudyRun
    diff: pd.DataFrame
    changed: pd.DataFrame
    exit_code: int


def _fmt_num(value: float, stderr: float, exact: bool) -> str:
    if math.isnan(value):
        return latex.MISSING
    if exact or math.isnan(stderr):
        return f"{value:.12g}"
    return f"{value:.6g} ± {stderr:.2g}"


def diff_markdown(
    diff: pd.DataFrame,
    *,
    study: str,
    original: str,
    rerun: str,
    nse: float,
    commits: tuple[str, str],
) -> str:
    counts = {
        s: int((diff["status"] == s).sum())
        for s in ("unchanged", "moved", "changed", "added", "removed")
    }
    ch = changed(diff)
    lines = [
        f"# Rerun of `{study}`",
        "",
        f"- original: `{original}` (commit {commits[0][:12]})",
        f"- rerun: `{rerun}` (commit {commits[1][:12]})",
        f"- rule: a Monte Carlo number moved when |new - old| > {nse:g} x max(stderr_old, "
        "stderr_new) (same seeds, so the errors are not combined in quadrature); an exact number, "
        "or one with a zero stderr on both sides, when it changed by more than 1e-12 relative; "
        "a changed unit or a switch between exact and Monte Carlo is 'changed'",
        f"- {len(diff)} numbers compared: {counts['unchanged']} unchanged, "
        f"{counts['moved']} moved, {counts['changed']} changed, {counts['added']} added, "
        f"{counts['removed']} removed",
        "",
        "**Verdict:** "
        + ("nothing moved." if ch.empty else f"{len(ch)} number(s) changed (listed below)."),
        "",
    ]
    if not ch.empty:
        lines += [
            "| table | row | column | status | old | new | delta | n stderr | why |",
            "|:--|:--|:--|:--|--:|--:|--:|--:|:--|",
        ]
        for r in ch.to_dict("records"):
            old = _fmt_num(r["old_value"], r["old_stderr"], bool(r["exact"]))
            new = _fmt_num(r["new_value"], r["new_stderr"], bool(r["exact"]))
            delta = "" if math.isnan(r["delta"]) else f"{r['delta']:.4g}"
            nse_ = "" if math.isnan(r["n_stderr"]) else f"{r['n_stderr']:.2f}"
            cells = [str(r["table"]), str(r["row"]), str(r["column"]), str(r["status"]), old, new]
            cells += [delta, nse_, str(r["why"])]
            lines.append("| " + " | ".join(latex.markdown_escape(c) for c in cells) + " |")
    return "\n".join(lines) + "\n"


def rerun_study(
    output_dir: str | Path,
    *,
    out_dir: str | Path | None = None,
    overrides: ConfigOverrides | None = None,
    nse: float = DEFAULT_NSE,
    argv: Sequence[str] = (),
    observation: dict[str, Any] | None = None,
) -> RerunOutcome:
    """Re-execute the stored config of ``output_dir`` and diff the numbers (module
    docstring).  Only the path overrides are honoured."""
    if not (math.isfinite(nse) and nse > 0):
        raise ConfigError(f"--nse must be finite and positive, got {nse}")
    src = Path(output_dir).expanduser().absolute()
    manifest = read_manifest(src)
    if overrides is not None and (overrides.fast or overrides.sets):
        raise ConfigError("rerun runs the stored config: only path overrides are allowed")
    config = apply_overrides(
        StudyConfig.from_mapping(manifest["config"], source=f"{src / MANIFEST_NAME}:config"),
        overrides,
    )
    if config.sha256() != manifest.get("config_sha256") and overrides is None:
        raise ConfigError(f"{src}: the stored config does not match its config_sha256")
    old_path = src / RESULTS_NAME
    if file_sha256(old_path) != manifest.get("results_sha256"):
        log.warning("%s: results.parquet differs from the manifest's results_sha256", src)
    old = Results.read(old_path)
    if out_dir is not None:
        new_out = Path(out_dir)
    else:
        stamp = utc_now().replace("-", "").replace(":", "").replace("+0000", "Z")
        new_out = src / RERUN_DIR / stamp
        n = 1
        while new_out.exists():
            n += 1
            new_out = src / RERUN_DIR / f"{stamp}_{n}"
    run = run_study(
        config,
        out_dir=new_out,
        argv=argv,
        latex_check=False,
        rerun_of=display_path(src),
        config_path_display=manifest.get("config_path"),
        observation=observation,
    )
    diff = diff_results(old, run.results, nse)
    ch = changed(diff)
    diff.to_csv(run.out_dir / DIFF_CSV, index=False)
    _write_text(
        run.out_dir / DIFF_MD,
        diff_markdown(
            diff,
            study=config.name,
            original=display_path(src),
            rerun=display_path(run.out_dir),
            nse=nse,
            commits=(
                str(manifest.get("git_commit") or "unknown"),
                str(run.manifest.get("git_commit") or "unknown"),
            ),
        ),
    )
    code = EXIT_OK if ch.empty and run.exit_code == EXIT_OK else EXIT_MOVED
    log.info(
        "%s: rerun compared %d numbers, %d changed (> %g stderr, unit / kind, added / removed) "
        "-> %s; wall clock %.2f s",
        config.name,
        len(diff),
        len(ch),
        nse,
        run.out_dir,
        run.manifest["wall_clock"]["total"],
    )
    return RerunOutcome(run, diff, ch, code)


def list_catalogue(directory: Path = CATALOGUE_DIR) -> list[dict[str, Any]]:
    """One row per catalogue YAML: path, name, mode, runner, question or the load error."""
    rows: list[dict[str, Any]] = []
    for p in sorted(directory.glob("*.yaml")):
        row: dict[str, Any] = {"path": display_path(p), "name": None, "mode": None, "error": ""}
        try:
            cfg = load_study_config(p)
            row.update(name=cfg.name, mode=cfg.mode, runner=cfg.runner)
            row["question"] = study_question(cfg, load_runner_module(cfg.runner))
        except (ConfigError, ImportError) as exc:
            row["error"] = str(exc)
        rows.append(row)
    return rows


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def _add_path_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--grid", default=None, help="volsto-precompute grid YAML (cwd-relative)")
    p.add_argument("--store", default=None, help="results store root (cwd-relative)")
    p.add_argument("--cache", default=None, help="leverage cache root (cwd-relative)")
    p.add_argument("--outputs", default=None, help="root of the M7 / M8b artefacts")
    p.add_argument("--out", default=None, help="output directory")


def _add_verbose(p: argparse.ArgumentParser) -> None:
    p.add_argument("-v", "--verbose", action="store_true", help="print tracebacks of errors")


class _Parser(argparse.ArgumentParser):
    """argparse with usage errors exiting :data:`EXIT_FAILED` (1): status 2 is reserved for
    missing requirements."""

    def error(self, message: str) -> NoReturn:
        self.print_usage(sys.stderr)
        self.exit(EXIT_FAILED, f"{self.prog}: error: {message}\n")


def build_parser() -> argparse.ArgumentParser:
    parser = _Parser(
        prog="volsto-study",
        description=(
            "Run a study (YAML config + runner module) into a self-contained output directory; "
            "reads the leverage cache and the results store, never calibrates."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a study config")
    run.add_argument("config", help="study YAML")
    _add_path_flags(run)
    run.add_argument("--fast", action="store_true", help="mode: fast (CI plumbing)")
    run.add_argument(
        "--set",
        dest="sets",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="set params.KEY (dotted) to the YAML-parsed VALUE; repeatable",
    )
    run.add_argument("--profile", action="store_true", help="cProfile compute() to profile.txt")
    run.add_argument(
        "--no-latex-check", action="store_true", help="do not compile study.tex with tectonic"
    )
    _add_verbose(run)
    rerun = sub.add_parser("rerun", help="re-execute a stored study and diff the numbers")
    rerun.add_argument("output_dir", help="a study output directory")
    _add_path_flags(rerun)
    rerun.add_argument(
        "--nse",
        type=float,
        default=DEFAULT_NSE,
        help="moved threshold in standard errors (finite, > 0)",
    )
    _add_verbose(rerun)
    render = sub.add_parser("render", help="rebuild tables / figures / documents from results")
    render.add_argument("output_dir", help="a study output directory")
    render.add_argument("--no-latex-check", action="store_true", help="do not recompile study.pdf")
    _add_verbose(render)
    sub.add_parser("list", help="the catalogue configs and their questions")
    return parser


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {text[0] if text else ''}"


def main(argv: Sequence[str] | None = None) -> int:
    """``volsto-study`` (module docstring)."""
    args_list = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(args_list)
    if not logging.getLogger().handlers:
        # the volsto loggers at INFO, everything else (matplotlib, fontTools) at WARNING
        logging.basicConfig(
            level=logging.WARNING,
            stream=sys.stderr,
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )
        logging.getLogger("volsto").setLevel(logging.INFO)
    t0 = time.perf_counter()
    record_argv = ["volsto-study", *args_list]
    observation: dict[str, Any] = {}
    verbose = bool(getattr(args, "verbose", False) or getattr(args, "profile", False))
    try:
        if args.command == "run":
            overrides = ConfigOverrides(
                grid=args.grid,
                store=args.store,
                cache=args.cache,
                outputs=args.outputs,
                fast=args.fast,
                sets=tuple(args.sets),
            )
            config = load_study_config(args.config, overrides)
            result = run_study(
                config,
                out_dir=args.out,
                config_path=args.config,
                argv=record_argv,
                profile=args.profile,
                latex_check=not args.no_latex_check,
                observation=observation,
            )
            print(f"{config.name}: wrote {result.out_dir}")
            code = result.exit_code
        elif args.command == "rerun":
            overrides_r = None
            if any(getattr(args, k) is not None for k in PATH_KEYS):
                overrides_r = ConfigOverrides(
                    grid=args.grid, store=args.store, cache=args.cache, outputs=args.outputs
                )
            outcome = rerun_study(
                args.output_dir,
                out_dir=args.out,
                overrides=overrides_r,
                nse=args.nse,
                argv=record_argv,
                observation=observation,
            )
            print((outcome.run.out_dir / DIFF_MD).read_text(encoding="utf-8"))
            code = outcome.exit_code
        elif args.command == "render":
            _, code = render_study(
                args.output_dir, latex_check=not args.no_latex_check, observation=observation
            )
            print(f"rendered {args.output_dir}")
        else:
            rows = list_catalogue(CATALOGUE_DIR)
            if not rows:
                print(f"no study configs under {display_path(CATALOGUE_DIR)}")
            for r in rows:
                if r["error"]:
                    print(f"{r['path']}: ERROR {r['error']}")
                else:
                    print(f"{r['path']}: {r['name']} [mode {r['mode']}] — {r['question']}")
            code = EXIT_FAILED if any(r["error"] for r in rows) else EXIT_OK
    except MissingRequirements as exc:
        print(str(exc))
        code = EXIT_MISSING
    except Exception as exc:
        print(f"error: {_first_line(exc)}", file=sys.stderr)
        if verbose:
            log.exception("volsto-study %s failed", args.command)
        code = EXIT_FAILED
    log.info(
        "volsto-study %s: exit %d, wall clock %.2f s; recalibrated: %s; cache changed during the "
        "run: %s",
        args.command,
        code,
        time.perf_counter() - t0,
        "YES" if observation.get("recalibrated") else "no",
        "yes (not attributed)" if observation.get("cache_changed") else "no",
    )
    return code


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
