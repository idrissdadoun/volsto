"""M9 S4 — integration of the viewers layer (SPEC §9.2 "Tests and measured"): every page
rendered headless against the **synthetic** store (``tests/_synthetic_store.py``) and against
the **toy** store of ``configs/grids/toy.yaml``; the read API's stderr-twin schema on the toy
store; relocatability end to end (toy store + cache + outputs copied from directory A to B —
and A renamed away while B renders when the run is not under xdist — every page renders
identically); ``volsto-viewer --check`` and ``volsto-precompute --dry-run`` run from the
command line.

**The toy calibration runs exactly once per pytest run, in the tests/conftest.py session
fixture** ``toy_build`` (the one sanctioned calibrating site: 3 leverage calibrations at 2·10⁴
particles over a 1y horizon, 4000 pricing paths, no risk, shards 1/2 and 2/2 through
``volsto.viewers.precompute.main`` into a temporary directory, never the repository cache);
this file only **consumes** it, as does ``tests/test_precompute.py``.  Under ``pytest -n auto``
the workers share that one build.  When the build exceeds its budget or fails, the toy half of
every test skips with the reason and the wall clock; nothing here asserts on wall clock.  No
test here calibrates anything: the pages and the API read the toy cache by file, every LSV
point's leverage is read through ``api.get_leverage`` after each render, and the toy cache's
files are asserted unchanged (size and mtime) after every render.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from _synthetic_store import make_synthetic_outputs, make_synthetic_store
from conftest import ToyBuild
from pandas.testing import assert_frame_equal

from volsto.models.leverage import LeverageFunction
from volsto.viewers import api, app
from volsto.viewers.config import ENV_CONFIG_FILE, ENV_VARS, ViewerConfig
from volsto.viewers.pages import PAGES, page_paths
from volsto.viewers.store import TABLES, StoreReader, mc_columns_without_stderr

ROOT = Path(__file__).resolve().parents[1]
TOY_GRID = ROOT / "configs" / "grids" / "toy.yaml"
DEFAULT_GRID = ROOT / "configs" / "grids" / "default.yaml"
REPO_CACHE = ROOT / "cache"
PAGE_TITLES: dict[str, str] = {f: t for f, t, _ in PAGES}
PAGE_NAMES = [p.name for p in page_paths()]
#: AppTest per-page timeout (page 1 builds the SSVI + Dupire surface on a cold process).
RENDER_TIMEOUT_S = 300
#: ``(store, page)`` pairs that cannot show a figure by construction — asserted on their
#: notices instead (see ``test_every_page_renders``).
NO_FIGURE_BY_CONSTRUCTION: frozenset[tuple[str, str]] = frozenset({("toy", "6_product_grid.py")})


# --------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StoreEnv:
    """A configured read-only environment: the roots, where they live, how they were made."""

    name: str
    cfg: ViewerConfig
    base: Path
    wall_s: float
    skip_reason: str = ""


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> StoreEnv:
    base = tmp_path_factory.mktemp("s4_synthetic")
    t0 = time.perf_counter()
    make_synthetic_store(base / "store")
    make_synthetic_outputs(base / "outputs")
    wall = time.perf_counter() - t0
    cfg = ViewerConfig(
        cache_root=base / "cache_empty",  # never created: nothing calibrates
        store_root=base / "store",
        outputs_root=base / "outputs",
    )
    return StoreEnv("synthetic", cfg, base, wall)


@pytest.fixture(scope="session")
def toy(toy_build: ToyBuild) -> StoreEnv:
    """The toy store of the conftest session fixture (module docstring) as a :class:`StoreEnv`;
    ``skip_reason`` carries the fixture's (budget / failure) reason."""
    cfg = ViewerConfig(
        cache_root=toy_build.cache_root,
        store_root=toy_build.store_root,
        outputs_root=toy_build.outputs_root,
        grid_path=toy_build.grid_path,
    )
    return StoreEnv("toy", cfg, toy_build.base, toy_build.total_wall_s, toy_build.skip_reason)


@pytest.fixture
def env(request: pytest.FixtureRequest) -> StoreEnv:
    """``synthetic`` or ``toy`` by parameter name; the toy half skips with its reason."""
    store = request.getfixturevalue(str(request.param))
    assert isinstance(store, StoreEnv)
    if store.skip_reason:
        pytest.skip(store.skip_reason)
    return store


def _toy(toy: StoreEnv) -> StoreEnv:
    if toy.skip_reason:
        pytest.skip(toy.skip_reason)
    return toy


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def _render(page: str, cfg: ViewerConfig, monkeypatch: pytest.MonkeyPatch) -> Any:
    """AppTest of ``page`` with ``cfg`` in the environment (what the app exports)."""
    from streamlit.testing.v1 import AppTest

    for var, value in cfg.to_env().items():
        monkeypatch.setenv(var, value)
    monkeypatch.setenv(ENV_CONFIG_FILE, str(cfg.store_root.parent / "no_viewer.yaml"))
    path = next(p for p in page_paths() if p.name == page)
    return AppTest.from_file(str(path), default_timeout=RENDER_TIMEOUT_S).run()


def _file_snapshot(root: Path) -> dict[str, tuple[int, float]]:
    """``relative path → (size, mtime)`` of every file below ``root`` (calibration detector)."""
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def _frames(at: Any) -> list[pd.DataFrame]:
    return [d.value for d in at.dataframe]


def _chart_specs(at: Any) -> list[str]:
    return [str(c.proto.spec) for c in at.get("plotly_chart")]


def _relocated(text: str, a: Path, b: Path) -> str:
    return text.replace(str(a), str(b))


def _cli(
    args: list[str], *, env_extra: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if k not in set(ENV_VARS.values())}
    env[ENV_CONFIG_FILE] = str(ROOT / "no_such_viewer.yaml")
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, "-m", *args],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
        timeout=900,
        check=False,
    )


# --------------------------------------------------------------------------------------------
# (1) every page renders against both stores
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("env", ["synthetic", "toy"], indirect=True)
@pytest.mark.parametrize("page", PAGE_NAMES)
def test_every_page_renders(env: StoreEnv, page: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Headless render: no exception, the title, at least one dataframe and one figure, an Excel
    export button; on the toy store the cache files are unchanged afterwards — size and mtime of
    every file (nothing calibrated) — and the leverage of every LSV point is then read by file
    through ``api.get_leverage`` (finite values, the point's cache key).  Page 6 on the toy
    store cannot draw a figure by construction (a heatmap needs two grid axes with ≥ 2 values —
    the toy grid is 2 × 1 × 1 — and the toy risk tier is ``none``): it must say so and print the
    ``--risk`` precompute command instead."""
    before = _file_snapshot(env.cfg.cache_root) if env.cfg.cache_root.exists() else {}
    t0 = time.perf_counter()
    at = _render(page, env.cfg, monkeypatch)
    wall = time.perf_counter() - t0
    n_df, n_fig = len(at.dataframe), len(at.get("plotly_chart"))
    print(f"\n[{env.name}] {page}: {wall:.2f} s, {n_df} tables, {n_fig} figures")
    assert not at.exception, [e.value for e in at.exception]
    assert [t.value for t in at.title] == [PAGE_TITLES[page]]
    assert n_df >= 1, "at least one dataframe per page"
    labels = [b.label for b in at.get("download_button")]
    assert any(lbl.startswith("Excel:") for lbl in labels), labels
    if (env.name, page) in NO_FIGURE_BY_CONSTRUCTION:
        infos = " ".join(i.value for i in at.info)
        assert "two grid axes" in infos, infos
        codes = [c.value for c in at.code]
        assert any(c.startswith("volsto-precompute") and "--risk" in c for c in codes), codes
    else:
        assert n_fig >= 1, "at least one figure per page"
        assert any(lbl.startswith(("PNG:", "SVG:", "HTML:")) for lbl in labels), labels
    captions = " ".join(c.value for c in at.caption)
    if env.name == "toy":
        assert "4 points" in captions and "grid 'toy'" in captions and "20000 particles" in captions
        assert _file_snapshot(env.cfg.cache_root) == before, "the cache changed during a render"
        grid = api.list_grid(env.cfg)
        lsv = grid[grid["mode"] != "lv"]
        assert len(lsv) == 3 and lsv["has_leverage"].all()
        for pid, key in zip(lsv["id"], lsv["cache_key"]):
            lev = api.get_leverage(env.cfg, pid)
            assert lev.cache_key == key and np.isfinite(lev.leverage.values).all()
    else:
        assert "3 points" in captions and "code tag synthetic" in captions
        assert not env.cfg.cache_root.exists()


# --------------------------------------------------------------------------------------------
# (2) stderr twins on the toy store
# --------------------------------------------------------------------------------------------


def test_toy_api_stderr_schema(toy: StoreEnv) -> None:
    """Every numeric field the API returns on the toy store carries its ``_stderr`` twin (the
    declared exact columns aside); the store tables pass ``mc_columns_without_stderr``; the
    leverage of each LSV point is read by file (values finite, the cache key, the particle
    count 20 000 and the 1y horizon in the npz metadata);
    a risk request on this ``risk: none`` store raises ``MissingPoint`` naming ``--risk light``."""
    cfg = _toy(toy).cfg
    grid = api.list_grid(cfg)
    assert len(grid) == 4 and sorted(grid["mode"]) == [
        "lv",
        "one_factor",
        "one_factor",
        "two_factor",
    ]
    assert api.columns_without_stderr(grid) == []
    assert api.columns_without_stderr(api.list_surfaces(cfg)) == []
    assert api.columns_without_stderr(api.get_products(cfg)) == []
    assert (grid["n_particles"] == 20_000).all()
    for pid in grid["id"]:
        rec = api.get_point(cfg, pid)
        for name, df in rec.tables().items():
            assert api.columns_without_stderr(df) == [], (pid, name)
        assert set(rec.products["key"]) == {"atm_vol", "vs_vol", "volswap_vol", "cliquet_1y"}
        assert (rec.products["value_stderr"] > 0).all()
        assert sorted(set(rec.ssr["T"])) == [0.25, 0.5, 1.0]
        if rec.mode == "lv":
            assert rec.varv.empty
            with pytest.raises(api.MissingPoint):
                api.get_leverage(cfg, pid)
        else:
            assert set(rec.varv["T"]) == {1.0}
            lev = api.get_leverage(cfg, pid)
            assert lev.cache_key == rec.diagnostics["cache_key"]
            assert np.isfinite(lev.leverage.values).all() and lev.leverage.values.shape[0] > 1
            # written by the particle calibration into the npz metadata (particle.py)
            assert lev.metadata["n_particles"] == 20_000 and lev.metadata["horizon"] == 1.0
            assert lev.metadata["cache_key"] == lev.cache_key
            assert api.columns_without_stderr(lev.error_map) == []
            if lev.error_map.empty:
                assert lev.report is None  # the cache entry holds no diagnostics.json
            table = lev.leverage_table(0.5)
            assert table.index.name == "t" and table.columns.name == "k" and not table.empty
        with pytest.raises(api.MissingPoint) as ei:
            api.get_risk(cfg, pid, "cliquet 1y")
        assert "--risk light" in ei.value.command and "toy.yaml" in ei.value.command
    reader = StoreReader(cfg.store_root)
    for name in TABLES:
        df = reader.store.table(name)
        assert mc_columns_without_stderr(df) == [], (name, mc_columns_without_stderr(df))
    assert api.store_summary(cfg).n_points == 4 and api.store_summary(cfg).cache_exists


# --------------------------------------------------------------------------------------------
# (3) relocatability end to end
# --------------------------------------------------------------------------------------------


def _string_cells(df: pd.DataFrame) -> pd.Series:
    """Every string-typed cell of ``df`` as one flat series (the path scan of the tables)."""
    obj = df.select_dtypes(include=["object", "string"])
    return pd.Series(obj.astype(str).to_numpy().ravel(), dtype=str)


def _without_argv(x: Any) -> Any:
    """Drop the run records' ``argv`` (the precompute command line kept as provenance in
    ``results/runs/*.json`` and in the merged manifest's run list: it names the ``--store`` /
    ``--cache`` the run was given, and nothing reads it back)."""
    if isinstance(x, dict):
        return {k: _without_argv(v) for k, v in x.items() if k != "argv"}
    if isinstance(x, list):
        return [_without_argv(v) for v in x]
    return x


def _assert_nothing_names(b_base: Path, a_base: Path) -> None:
    """Nothing under ``b_base`` names ``a_base``: every JSON of the store (argv aside), every
    string cell of every store table, every cache ``spec.json`` and the leverage ``npz``
    metadata."""
    needle = str(a_base)
    for p in (b_base / "store" / "results").rglob("*.json"):
        assert needle not in json.dumps(_without_argv(json.loads(p.read_text()))), p
    reader = StoreReader(b_base / "store")
    for name in TABLES:
        cells = _string_cells(reader.store.table(name))
        assert not cells.str.contains(needle, regex=False).any(), (name, cells.tolist())
    for key in os.listdir(b_base / "cache"):
        spec = b_base / "cache" / key / "spec.json"
        if spec.exists():
            assert needle not in spec.read_text(), spec
        npz = b_base / "cache" / key / "leverage.npz"
        if npz.exists():
            meta = json.dumps(LeverageFunction.load(npz).metadata, default=str)
            assert needle not in meta, npz


@dataclass
class _Rendered:
    """What one page rendered: dataframes, figure specifications, captions."""

    frames: list[pd.DataFrame]
    charts: list[str]
    captions: list[str]


def _render_all(cfg: ViewerConfig, monkeypatch: pytest.MonkeyPatch) -> dict[str, _Rendered]:
    out: dict[str, _Rendered] = {}
    for page in PAGE_NAMES:
        at = _render(page, cfg, monkeypatch)
        assert not at.exception, (page, [e.value for e in at.exception])
        out[page] = _Rendered(_frames(at), _chart_specs(at), [c.value for c in at.caption])
    return out


def test_relocatable_end_to_end(
    toy: StoreEnv, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Toy store + toy cache + outputs copied from directory A to directory B: nothing under B
    names A (every store JSON, every string cell of every store table, the cache specs and the
    npz metadata); the API frames, the leverage values and every page's dataframes and figure
    specifications are identical up to the root path in a provenance caption.  Outside xdist
    (no ``PYTEST_XDIST_WORKER``) A is renamed away while the B half runs — a hidden absolute
    reference into A would then fail, not silently pass — and restored afterwards, byte-identical
    (a rename); under xdist A is shared with the other workers and stays, which is why the path
    scan covers every table's string cells and not only the JSON files."""
    a = _toy(toy)
    b_base = tmp_path / "B"
    t0 = time.perf_counter()
    shutil.copytree(a.base, b_base)
    cfg_b = ViewerConfig(
        cache_root=b_base / "cache",
        store_root=b_base / "store",
        outputs_root=b_base / "outputs",
        grid_path=a.cfg.grid_path,
    )
    print(f"\ncopied A → B in {time.perf_counter() - t0:.2f} s")
    _assert_nothing_names(b_base, a.base)

    # the A half first (API records, leverage values, every page)
    grid_a = api.list_grid(a.cfg)
    records_a = {pid: api.get_point(a.cfg, pid) for pid in grid_a["id"]}
    lev_a = {
        pid: api.get_leverage(a.cfg, pid).leverage
        for pid, mode in zip(grid_a["id"], grid_a["mode"])
        if mode != "lv"
    }
    pages_a = _render_all(a.cfg, monkeypatch)

    # the B half — with A out of the way when this process owns it
    hide_a = not os.environ.get("PYTEST_XDIST_WORKER")
    aside = a.base.with_name(a.base.name + ".aside")
    snapshot_a = _file_snapshot(a.base)
    if hide_a:
        os.rename(a.base, aside)
    try:
        assert a.base.exists() != hide_a
        grid_b = api.list_grid(cfg_b)
        assert_frame_equal(grid_a, grid_b)
        for pid, ra in records_a.items():
            rb = api.get_point(cfg_b, pid)
            for name, df in ra.tables().items():
                assert_frame_equal(rb.tables()[name], df)
            assert ra.params == rb.params and ra.provenance == rb.provenance
        for pid, la in lev_a.items():
            lb = api.get_leverage(cfg_b, pid).leverage
            assert np.array_equal(la.values, lb.values) and np.array_equal(la.times, lb.times)
        pages_b = _render_all(cfg_b, monkeypatch)
    finally:
        if hide_a:
            os.rename(aside, a.base)
    assert _file_snapshot(a.base) == snapshot_a, "A changed across the B half"

    # page level: every dataframe and figure identical
    n_frames = n_charts = 0
    for page in PAGE_NAMES:
        ra_, rb_ = pages_a[page], pages_b[page]
        assert len(ra_.frames) == len(rb_.frames), page
        for i, (da, db) in enumerate(zip(ra_.frames, rb_.frames)):
            assert_frame_equal(da, db, obj=f"{page} dataframe {i}")
        assert len(ra_.charts) == len(rb_.charts), page
        for i, (sa, sb) in enumerate(zip(ra_.charts, rb_.charts)):
            assert _relocated(sa, a.base, b_base) == sb, (page, i)
        assert [_relocated(c, a.base, b_base) for c in ra_.captions] == rb_.captions, page
        n_frames += len(ra_.frames)
        n_charts += len(ra_.charts)
    print(
        f"relocatability: {n_frames} dataframes and {n_charts} figures identical over "
        f"{len(PAGE_NAMES)} pages (A {'renamed away' if hide_a else 'kept, shared under xdist'} "
        "for the B half)"
    )
    assert n_frames >= 8 and n_charts >= 7


# --------------------------------------------------------------------------------------------
# (4) the two CLIs
# --------------------------------------------------------------------------------------------


def test_check_pages_in_process(synthetic: StoreEnv, tmp_path: Path) -> None:
    """``check_pages``: every page ok on the synthetic store (counts filled, environment
    restored); a page that raises gives ``ok = False`` with the exception text, exit code 1 and a
    ``FAIL`` line in the report."""
    env_before = {k: os.environ.get(k) for k in ENV_VARS.values()}
    results = app.check_pages(synthetic.cfg, timeout=RENDER_TIMEOUT_S)
    assert {k: os.environ.get(k) for k in ENV_VARS.values()} == env_before
    assert [r.page for r in results] == PAGE_NAMES
    assert all(r.ok and r.n_dataframes >= 1 and r.n_figures >= 1 for r in results), results
    assert app.exit_code(results) == 0
    report = app.format_checks(results)
    assert report.count(" ok ") == 8 and "all pages rendered" in report
    print("\n" + report)
    broken = tmp_path / "9_broken.py"
    broken.write_text('"""A page that raises."""\nraise RuntimeError("boom from the page")\n')
    bad = app.check_pages(synthetic.cfg, pages=[(broken, "Broken", "")], timeout=60)
    assert len(bad) == 1 and not bad[0].ok and "boom from the page" in bad[0].exceptions[0]
    assert app.exit_code(bad) == 1 and app.exit_code([]) == 1
    assert "FAIL" in app.format_checks(bad) and "1 page(s) FAILED" in app.format_checks(bad)
    help_text = app.app_parser().format_help()
    assert "--check" in help_text and "--check-timeout" in help_text


def test_viewer_check_cli(toy: StoreEnv, tmp_path: Path) -> None:
    """``python -m volsto.viewers.app --check`` against the toy store: exit 0, one ``ok`` line per
    page; against an empty triple: exit 0 and the store reported empty (the pages print the
    precompute command) — and **nothing is created**, not the store root, not the cache, not the
    outputs (a read path never touches the filesystem); the toy cache is untouched."""
    t = _toy(toy)
    before = _file_snapshot(t.cfg.cache_root)
    t0 = time.perf_counter()
    out = _cli(
        [
            "volsto.viewers.app",
            "--check",
            "--store",
            str(t.cfg.store_root),
            "--cache",
            str(t.cfg.cache_root),
            "--outputs",
            str(t.cfg.outputs_root),
            "--grid",
            str(t.cfg.grid_path),
        ]
    )
    wall = time.perf_counter() - t0
    print(f"\nvolsto-viewer --check (toy): rc {out.returncode} in {wall:.1f} s\n{out.stdout}")
    assert out.returncode == 0, out.stdout + out.stderr
    for page in PAGE_NAMES:
        assert any(
            line.startswith(page) and " ok " in line for line in out.stdout.splitlines()
        ), page
    assert "check: 8 pages, all pages rendered" in out.stdout
    assert "store: 4 points" in out.stdout and "grid 'toy'" in out.stdout
    assert _file_snapshot(t.cfg.cache_root) == before
    empty = _cli(
        [
            "volsto.viewers.app",
            "--check",
            "--store",
            str(tmp_path / "empty_store"),
            "--cache",
            str(tmp_path / "empty_cache"),
            "--outputs",
            str(tmp_path / "empty_outputs"),
        ]
    )
    assert empty.returncode == 0, empty.stdout + empty.stderr
    assert "store: EMPTY" in empty.stdout and "all pages rendered" in empty.stdout
    for name in ("empty_store", "empty_cache", "empty_outputs"):
        assert not (tmp_path / name).exists(), f"--check created {name}"
    assert sorted(os.listdir(tmp_path)) == [], os.listdir(tmp_path)


def test_precompute_dry_run_cli(toy: StoreEnv, tmp_path: Path) -> None:
    """``python -m volsto.viewers.precompute --dry-run``: on the toy grid against the built toy
    store and cache every point is a cache hit (``0 misses``) and nothing is written; on the
    default grid against the repository cache (the projection reads its manifest) and a
    ``tmp_path`` store — never the repository's ``outputs/store`` — the 131-point projection
    prints in about a second (the numbers quoted in SPEC §9.2), with one row per tier and the
    light tier twice (the configured forward-variance bucket count and the alternative one), and
    the store root is not even created."""
    t = _toy(toy)
    manifest = t.cfg.store_root / "results" / "manifest.json"
    before_manifest = manifest.read_text()
    before_files = _file_snapshot(t.cfg.store_root)
    out = _cli(
        [
            "volsto.viewers.precompute",
            "--grid",
            str(TOY_GRID),
            "--store",
            str(t.cfg.store_root),
            "--cache",
            str(t.cfg.cache_root),
            "--dry-run",
        ]
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "4 points (0 misses)" in out.stdout and "dry run: nothing computed" in out.stdout
    assert "cache manifest median of 3 entries at 20000 particles" in out.stdout
    assert manifest.read_text() == before_manifest
    assert _file_snapshot(t.cfg.store_root) == before_files
    t0 = time.perf_counter()
    full = _cli(
        [
            "volsto.viewers.precompute",
            "--grid",
            str(DEFAULT_GRID),
            "--store",
            str(tmp_path / "store"),
            "--dry-run",
        ]
    )
    wall = time.perf_counter() - t0
    assert full.returncode == 0, full.stdout + full.stderr
    assert not (tmp_path / "store").exists()
    assert "131 points" in full.stdout and "dry run: nothing computed" in full.stdout
    tiers = [
        ln
        for ln in full.stdout.splitlines()
        if ln.lstrip(" *").startswith(("none", "light", "full"))
    ]
    # four rows since the light tier is projected at both fwd-var bucket counts (the configured
    # 20-bucket M5 ladder and the coarse ladder over the light pillars), one of them starred
    assert len(tiers) == 4, full.stdout
    assert sum(ln.startswith(" *") for ln in tiers) == 1, tiers
    light = [ln for ln in tiers if ln.lstrip(" *").startswith("light")]
    assert len(light) == 2 and "20-bucket" in light[0] and "3-bucket" in light[1], light
    print(f"\ndefault-grid dry run from the CLI in {wall:.1f} s:\n" + "\n".join(tiers))
    print(f"toy build wall (session fixture): {t.wall_s:.1f} s")
