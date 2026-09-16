"""M9 Parts 2–3 — the read API, the viewer configuration, the shared components, the app
skeleton and the placeholder pages (SPEC §9.2 "Read API and app").

Everything here runs against the **synthetic** store of ``tests/_synthetic_store.py`` (three
points, two surfaces, every table, a synthetic M8b run and the M7 tables) with the leverage
cache pointed at an empty temporary directory: no test calibrates, no test simulates, and
``test_api_never_calibrates`` asserts the cache directory stays absent and that the API module
does not even import ``get_or_calibrate``.  A second configuration points at a cache holding
**one** entry written from arrays (``make_leverage_entry``: ``leverage.npz`` +
``diagnostics.json``, still no calibration), which is what makes ``get_leverage``'s success
path — the calibration error map and its standard errors — part of the stderr walk.  The tests
that touch the repository's real ``outputs/m7`` and ``outputs/m8b`` files (walking their stderr
schema) skip when those git-ignored files are absent.  Wall clocks are printed (``-s``), never
asserted.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from _synthetic_store import (
    GRID_PATH,
    HEDGING_RUN_ID,
    MARKING_KEY,
    SNAPSHOT_NAME,
    make_leverage_entry,
    make_synthetic_outputs,
    make_synthetic_store,
)
from pandas.testing import assert_frame_equal

from volsto.config import LocalVolConfig
from volsto.viewers import api
from volsto.viewers.config import ENV_VARS, ViewerConfig, build_parser
from volsto.viewers.pages import PAGES, page_paths

ROOT = Path(__file__).resolve().parents[1]
API_SOURCE = (ROOT / "volsto" / "viewers" / "api.py").read_text()


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    base = tmp_path_factory.mktemp("viewer")
    t0 = time.perf_counter()
    info = make_synthetic_store(base / "store")
    make_synthetic_outputs(base / "outputs")
    print(f"\nsynthetic store + outputs written in {time.perf_counter() - t0:.2f} s under {base}")
    cfg = ViewerConfig(
        cache_root=base / "cache_empty",  # never created: nothing calibrates
        store_root=base / "store",
        outputs_root=base / "outputs",
    )
    # a second configuration whose cache holds ONE entry written from arrays (no calibration,
    # no Monte Carlo): the only way to walk get_leverage's success path — the error map
    cfg_cache = ViewerConfig(
        cache_root=base / "cache", store_root=base / "store", outputs_root=base / "outputs"
    )
    grid = api.list_grid(cfg_cache)
    one = grid[grid["mode"] == "one_factor"].iloc[0]
    one_id = str(one["id"])
    make_leverage_entry(cfg_cache.cache_root, str(one["cache_key"]))
    return {"cfg": cfg, "cfg_cache": cfg_cache, "one_factor": one_id, "base": base, **info}


def _cfg(synthetic: dict[str, object]) -> ViewerConfig:
    cfg = synthetic["cfg"]
    assert isinstance(cfg, ViewerConfig)
    return cfg


def _cfg_cache(synthetic: dict[str, object]) -> ViewerConfig:
    cfg = synthetic["cfg_cache"]
    assert isinstance(cfg, ViewerConfig)
    return cfg


def _ids(synthetic: dict[str, object]) -> list[str]:
    ids = synthetic["point_ids"]
    assert isinstance(ids, list)
    return [str(i) for i in ids]


# --------------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------------


def test_config_precedence(tmp_path: Path) -> None:
    """Defaults ← YAML ← environment ← CLI flags; unknown YAML keys are rejected; the
    environment export reproduces the configuration."""
    d = ViewerConfig()
    assert d.cache_root == ROOT / "cache" and d.store_root == ROOT / "outputs" / "store"
    y = tmp_path / "viewer.yaml"
    y.write_text("store_root: from_yaml\ncache_root: yaml_cache\n")
    c1 = ViewerConfig.from_sources(yaml_path=y, env={})
    assert c1.store_root == Path("from_yaml") and c1.cache_root == Path("yaml_cache")
    c2 = ViewerConfig.from_sources(yaml_path=y, env={"VOLSTO_STORE": "from_env"})
    assert c2.store_root == Path("from_env") and c2.cache_root == Path("yaml_cache")
    c3, ns = ViewerConfig.from_args(
        ["--store", "from_cli", "--config", str(y), "--port", "9000"], env={"VOLSTO_STORE": "x"}
    )
    assert c3.store_root == Path("from_cli") and ns.port == 9000
    assert c3.cache_root == Path("yaml_cache")
    c4 = ViewerConfig.from_env(env=c3.to_env())
    assert c4 == c3
    assert set(c3.to_env()) == set(ENV_VARS.values())
    bad = tmp_path / "bad.yaml"
    bad.write_text("results: nope\n")
    with pytest.raises(ValueError, match="unknown viewer settings"):
        ViewerConfig.from_sources(yaml_path=bad, env={})
    help_text = build_parser().format_help()
    for flag in ("--cache", "--store", "--outputs", "--snapshots", "--grid", "--port"):
        assert flag in help_text


# --------------------------------------------------------------------------------------------
# the store side of the API
# --------------------------------------------------------------------------------------------


def test_list_grid_synthetic(synthetic: dict[str, object]) -> None:
    cfg = _cfg(synthetic)
    g = api.list_grid(cfg)
    assert list(g.columns) == list(api.GRID_COLUMNS)
    assert len(g) == 3 and set(g["id"]) == set(_ids(synthetic))
    assert set(g["mode"]) == {"lv", "one_factor", "marking"}
    assert set(g["surface"]) == {"placeholder", SNAPSHOT_NAME}
    assert not g["has_leverage"].any()  # the cache is empty: a file check, no calibration
    assert g["has_products"].all() and g["has_risk"].all() and g["has_ssr"].all()
    assert g.loc[g["mode"] == "lv", "has_varv"].eq(False).all()
    lv = g[g["mode"] == "lv"].iloc[0]
    assert np.isnan(lv["nu"]) and lv["id"].startswith("lv:")
    one = g[g["mode"] == "one_factor"].iloc[0]
    assert one["nu"] == 0.5 and one["axis_rho"] == -0.7 and one["axis_kappa"] == 1.5
    mk = g[g["mode"] == "marking"].iloc[0]
    assert mk["status"] == "binding" and mk["ssr_target"] == 1.0 and mk["skew_eps"] == 0.1
    assert int(g["n_particles"].iloc[0]) == 800_000
    # an empty store gives the same columns and no rows
    empty = api.list_grid(ViewerConfig(store_root=Path(str(synthetic["base"])) / "nowhere"))
    assert list(empty.columns) == list(api.GRID_COLUMNS) and empty.empty


def test_get_point_records(synthetic: dict[str, object]) -> None:
    cfg = _cfg(synthetic)
    for pid in _ids(synthetic):
        rec = api.get_point(cfg, pid)
        assert rec.id == pid and rec.label == synthetic["labels"][pid]  # type: ignore[index]
        for name, df in rec.tables().items():
            assert "point_id" not in df.columns, name
        assert not rec.products.empty and not rec.forward_smile.empty
        assert set(rec.forward_smile[["t1", "t2"]].drop_duplicates().itertuples(index=False)) == {
            (1.0, 2.0),
            (2.0, 3.0),
        }
        # the store's beyond_horizon flag reaches the pages unchanged (S1 F14): a boolean
        # column of both forward tables, false on this grid (3y horizon, windows to 3y)
        for name in ("forward_smile", "forward_vols"):
            col = rec.tables()[name]["beyond_horizon"]
            assert col.dtype == bool and not col.any(), name
        assert rec.provenance["code_tag"] == "synthetic"
        assert rec.diagnostics["n_particles"] == 800_000
        if rec.mode == "lv":
            assert rec.params == {} and rec.varv.empty and rec.diagnostics["cache_key"] == ""
        else:
            assert set(rec.params) == set(api.MODEL_PARAMS) and not rec.varv.empty
            assert rec.diagnostics["cache_key"]
        if rec.mode == "marking":
            assert rec.fit["fit_status"] == "binding"
            assert rec.ssr["ssr_naked_first_order"].notna().all()
            assert rec.ssr["ssr_target"].eq(1.0).all()
    mk = api.get_point(cfg, MARKING_KEY)
    assert mk.params["nu"] == 1.9


def test_every_numeric_field_has_stderr(synthetic: dict[str, object]) -> None:
    """Walk every API return on the synthetic store: no Monte Carlo column without its
    ``_stderr`` twin (exact fields are the declared ones).  Included: the leverage error map of
    :func:`api.get_leverage`'s **success** path (a cache entry written from arrays — no
    calibration) and the closed-form tables of :class:`api.SurfaceRecord`, which are exact only
    within their own table (``api.TABLE_EXACT``)."""
    cfg = _cfg(synthetic)
    assert api.columns_without_stderr(api.list_grid(cfg)) == []
    assert api.columns_without_stderr(api.list_surfaces(cfg)) == []
    assert api.columns_without_stderr(api.get_products(cfg)) == []
    regimes = {"model", "sticky_strike", "sticky_moneyness", "sticky_skew", "sticky_local_vol"}
    for pid in _ids(synthetic):
        rec = api.get_point(cfg, pid)
        for name, df in rec.tables().items():
            assert api.columns_without_stderr(df) == [], (pid, name)
            if name == "products":
                assert {"value", "value_stderr"} <= set(df.columns)
        for product in ("autocall 3y", "cliquet 1y"):
            risk = api.get_risk(cfg, pid, product)
            assert list(risk.columns) == list(api.RISK_COLUMNS)
            assert api.columns_without_stderr(risk) == []
            # the RiskReport groups the precompute stores: gamma is its own section
            assert set(risk["section"]) == {"delta", "gamma", "fwd_var", "skew"}
            for kind in ("delta", "gamma"):
                rows = risk[risk["section"] == kind]
                assert set(rows["name"]) == {f"{kind}[{r}]" for r in regimes}
                assert set(rows["regime"]) == regimes
    # the forward smile's ``iv`` is a Monte Carlo column and IS walked (it is exact only in the
    # closed-form surface tables, api.TABLE_EXACT): dropping its twin is caught
    smile = api.get_point(cfg, _ids(synthetic)[0]).forward_smile
    assert api.columns_without_stderr(smile.drop(columns=["iv_stderr"])) == ["iv"]
    surf = api.get_surface(cfg, "placeholder")
    ivt = surf.implied_vol_table([0.5, 1.0], [-0.1, 0.0, 0.1])
    assert api.columns_without_stderr(ivt, exact=api.TABLE_EXACT["surface_implied_vol"]) == []
    lvt = surf.local_vol_table([0.5, 1.0], [-0.1, 0.0, 0.1])
    assert api.columns_without_stderr(lvt, exact=api.TABLE_EXACT["surface_local_vol"]) == []
    atm = surf.atm_table([0.5, 1.0])
    assert api.columns_without_stderr(atm, exact=api.TABLE_EXACT["surface_atm"]) == []
    # without the per-table scope the same closed-form columns are walked like any MC column
    assert api.columns_without_stderr(ivt) == ["iv", "forward"]
    # the leverage of the point whose cache entry the fixture wrote (still no calibration)
    lev = api.get_leverage(_cfg_cache(synthetic), str(synthetic["one_factor"]))
    assert lev.report is not None and not lev.error_map.empty
    assert {"error_vp_stderr", "price_stderr", "model_vol_stderr"} <= set(lev.error_map.columns)
    assert api.columns_without_stderr(lev.error_map) == []
    assert lev.error_map["model_vol_stderr"].eq(lev.error_map["error_vp_stderr"] / 100.0).all()
    runs = api.list_hedging_runs(cfg)
    assert list(runs.columns) == list(api.HEDGING_RUN_COLUMNS) and len(runs) == 1
    assert api.columns_without_stderr(runs) == []
    run = api.get_hedging_run(cfg, HEDGING_RUN_ID)
    for name, df in run.tables().items():
        unpaired = api.UNPAIRED_UPSTREAM["hedging_regimes"] if name == "regimes" else frozenset()
        assert api.columns_without_stderr(df, unpaired=unpaired) == [], name
    assert {"mean", "mean_stderr"} <= set(run.regimes.columns)
    assert api.columns_without_stderr(api.get_hedging_table(cfg, "A")) == []
    marking = api.get_marking(cfg)
    for name, df in marking.tables().items():
        assert api.columns_without_stderr(df, unpaired=api.marking_unpaired(name, df)) == [], name
    assert "ssr_lsv@1y_stderr" in marking.fits.columns
    # the stage-3 forward / spot ratio is a Monte Carlo number, not an exact field: it is walked
    # above and it is paired, per window (review:S2 F4)
    for window in ("1y-into-1y", "2y-into-1y"):
        assert {f"fwd/spot@{window}", f"fwd/spot@{window}_stderr"} <= set(marking.fits.columns)
        assert not api.is_exact_column(f"fwd/spot@{window}")
    dropped = marking.fits.drop(columns=["fwd/spot@1y-into-1y_stderr"])
    assert api.columns_without_stderr(dropped, unpaired=api.marking_unpaired("fits", dropped)) == [
        "fwd/spot@1y-into-1y"
    ]
    assert "per_vp_90_110_0.5y_stderr" in marking.shadow_rotation.columns
    sr = marking.shadow_rotation.iloc[0]
    assert sr["per_vp_90_110_0.5y_stderr"] == pytest.approx(sr["per_rota_stderr"] / 0.56)
    assert marking.convention.startswith("rota +1")


def test_missing_point_carries_command(synthetic: dict[str, object]) -> None:
    """The command names the point (``--only <id>``) and the grid the **store** was built with
    (its run record's ``grid_path``, not the viewer's configured grid: review:S2 F2); the
    missing-risk command is the per-point line without ``--resume`` (review:S2 F3)."""
    cfg = _cfg(synthetic)
    with pytest.raises(api.MissingPoint) as ei:
        api.get_point(cfg, "not-a-point")
    exc = ei.value
    assert exc.key == "not-a-point"
    assert exc.command == (
        f"volsto-precompute --grid {GRID_PATH} "
        f"--store {cfg.relative_to_cwd(cfg.store_root)} "
        f"--cache {cfg.relative_to_cwd(cfg.cache_root)} --only not-a-point --resume"
    )
    assert str(cfg.grid_path.name) not in exc.command  # the store's grid wins over the default
    assert "not-a-point" in str(exc)
    pid = _ids(synthetic)[0]
    with pytest.raises(api.MissingPoint) as ei2:
        api.get_risk(cfg, pid, "phoenix 3y")  # the point exists, the product has no risk rows
    assert ei2.value.command == (
        f"volsto-precompute --grid {GRID_PATH} "
        f"--store {cfg.relative_to_cwd(cfg.store_root)} "
        f"--cache {cfg.relative_to_cwd(cfg.cache_root)} --only {pid} --resume --risk light"
    )
    # --resume is what makes it the risk-only refresh of precompute.pending_steps: the stored
    # pricing and analytics of that point are reused and only the risk step runs
    assert "--resume" in ei2.value.command
    assert "phoenix 3y" in ei2.value.what and "stored" in ei2.value.what
    with pytest.raises(api.MissingPoint):
        api.get_risk(cfg, "nope", "autocall 3y")
    with pytest.raises(api.MissingArtefact) as ei3:
        api.get_hedging_run(cfg, "B__nothing")
    assert ei3.value.command == "scripts/m8b.py --study B"
    with pytest.raises(api.MissingArtefact) as ei4:
        api.get_hedging_pnl(cfg, HEDGING_RUN_ID)  # the synthetic run has no .pkl
    assert ei4.value.command == "scripts/m8b.py --study A"
    no_m7 = ViewerConfig(
        store_root=cfg.store_root, outputs_root=Path(str(synthetic["base"])) / "no_outputs"
    )
    with pytest.raises(api.MissingArtefact) as ei5:
        api.get_marking(no_m7)
    assert ei5.value.command == "scripts/m7_p1_marking.py"


def test_precompute_command_grid_and_flags(synthetic: dict[str, object], tmp_path: Path) -> None:
    """:func:`api.precompute_command` quotes the store's own grid when it has one and falls back
    to the configured grid on an empty store; ``--only`` / ``--resume`` / ``--risk`` are the
    contract of :mod:`volsto.viewers.precompute` (review:S2 F2, F3)."""
    cfg = _cfg(synthetic)
    assert api.grid_path_of(cfg) == GRID_PATH
    # a store copy whose run record names another grid: the command follows the store
    other = tmp_path / "store_toy"
    shutil.copytree(cfg.store_root, other)
    run = next((other / "results" / "runs").glob("*.json"))
    doc = json.loads(run.read_text())
    doc["grid_path"] = "configs/grids/toy.yaml"
    run.write_text(json.dumps(doc))
    (other / "results" / "manifest.json").unlink()  # rebuilt from the run records on read
    cfg_toy = ViewerConfig(
        cache_root=cfg.cache_root, store_root=other, outputs_root=cfg.outputs_root
    )
    assert api.grid_path_of(cfg_toy) == "configs/grids/toy.yaml"
    with pytest.raises(api.MissingPoint) as ei:
        api.get_point(cfg_toy, "nope")
    assert "--grid configs/grids/toy.yaml" in ei.value.command and "--only nope" in ei.value.command
    # an empty store has no run record: the configured grid
    empty = ViewerConfig(store_root=tmp_path / "empty", grid_path=Path("configs/grids/toy.yaml"))
    assert api.grid_path_of(empty) == "configs/grids/toy.yaml"
    assert api.precompute_command(empty).endswith("--resume")
    assert " --only " not in api.precompute_command(empty)
    assert api.precompute_command(empty, risk="light").endswith("--resume --risk light")
    assert not (tmp_path / "empty").exists()  # reading a missing store creates nothing


def test_unknown_surface_asks_for_a_grid_entry(
    synthetic: dict[str, object], tmp_path: Path
) -> None:
    """An unknown surface is not a missing computation: the command says to add it to the grid
    YAML first (review:S2 F10).  An off-grid snapshot that exists both at the top level of
    ``snapshots_root`` and in a history sub-directory resolves to the top-level file — the one
    :func:`api.list_surfaces` offers — whatever the order the filesystem walks them in."""
    cfg = _cfg(synthetic)
    with pytest.raises(api.MissingArtefact) as ei:
        api.get_surface(cfg, "no-such-surface")
    assert ei.value.command.startswith(f"add the surface to {GRID_PATH}")
    assert "volsto-precompute --grid" in ei.value.command
    assert "no-such-surface" in ei.value.what
    top_file = sorted(cfg.snapshots_root.glob("*.yaml"))[0]
    snaps = tmp_path / "snapshots"
    (snaps / "history").mkdir(parents=True)
    shutil.copy(top_file, snaps / "twice.yaml")
    (snaps / "history" / "twice.yaml").write_text("not: a surface\n")  # would fail if chosen
    cfg_snaps = ViewerConfig(
        cache_root=cfg.cache_root,
        store_root=cfg.store_root,
        outputs_root=cfg.outputs_root,
        snapshots_root=snaps,
    )
    rec = api.get_surface(cfg_snaps, "twice")
    assert rec.kind == "snapshot" and rec.path.endswith("twice.yaml")
    assert "history" not in rec.path and rec.spot > 0


def test_api_never_calibrates(synthetic: dict[str, object]) -> None:
    """The leverage of a stored point is read by file only: with an empty cache root the API
    raises MissingPoint (command included) and never creates the cache directory; the module
    does not import the calibrating entry point."""
    cfg = _cfg(synthetic)
    import ast

    tree = ast.parse(API_SOURCE)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
    assert "get_or_calibrate" not in attrs | names
    assert "LeverageCache" not in imported | names  # addressed by file, never constructed
    assert "MonteCarlo" not in imported | names
    one = next(p for p in _ids(synthetic) if not p.startswith(("lv:", "marking:")))
    with pytest.raises(api.MissingPoint) as ei:
        api.get_leverage(cfg, one)
    assert ei.value.key == one and "leverage" in ei.value.what and "--resume" in ei.value.command
    lv = next(p for p in _ids(synthetic) if p.startswith("lv:"))
    with pytest.raises(api.MissingPoint):
        api.get_leverage(cfg, lv)  # the LV point has no leverage by construction
    assert not cfg.cache_root.exists()


def test_relocatable_store(synthetic: dict[str, object], tmp_path: Path) -> None:
    """Move the store to another directory: every API frame is identical."""
    cfg = _cfg(synthetic)
    before = {pid: api.get_point(cfg, pid) for pid in _ids(synthetic)}
    grid_before = api.list_grid(cfg)
    moved = tmp_path / "moved" / "store"
    moved.parent.mkdir()
    shutil.copytree(cfg.store_root, moved)
    cfg_b = ViewerConfig(cache_root=cfg.cache_root, store_root=moved, outputs_root=cfg.outputs_root)
    assert_frame_equal(api.list_grid(cfg_b), grid_before)
    for pid, rec in before.items():
        rec_b = api.get_point(cfg_b, pid)
        for name, df in rec.tables().items():
            assert_frame_equal(rec_b.tables()[name], df)
        assert rec_b.params == rec.params
        pd.testing.assert_series_equal(pd.Series(rec_b.diagnostics), pd.Series(rec.diagnostics))
        assert_frame_equal(
            api.get_risk(cfg_b, pid, "cliquet 1y"), api.get_risk(cfg, pid, "cliquet 1y")
        )
    assert_frame_equal(api.list_surfaces(cfg_b), api.list_surfaces(cfg))
    assert api.store_summary(cfg_b).n_points == 3


# --------------------------------------------------------------------------------------------
# surfaces (pure construction)
# --------------------------------------------------------------------------------------------


def test_get_surface_builds_without_cache(synthetic: dict[str, object]) -> None:
    """Placeholder and snapshot surfaces and their Dupire grids are built from the configs
    (the grid mapping of the store manifest); the empty cache root is never touched."""
    cfg = _cfg(synthetic)
    surfaces = api.list_surfaces(cfg)
    assert list(surfaces.columns) == ["name", "kind", "path", "source", "n_points"]
    by_name = surfaces.set_index("name")
    assert by_name.loc["placeholder", "kind"] == "placeholder"
    assert by_name.loc["placeholder", "n_points"] == 2
    assert by_name.loc[SNAPSHOT_NAME, "kind"] == "snapshot"
    assert by_name.loc[SNAPSHOT_NAME, "source"] == "store manifest"
    assert "spx_2022-12-02" in by_name.index  # a snapshot file not on this grid
    t0 = time.perf_counter()
    ph = api.get_surface(cfg, "placeholder")
    t1 = time.perf_counter()
    sp = api.get_surface(cfg, SNAPSHOT_NAME)
    t2 = time.perf_counter()
    print(f"\nget_surface: placeholder {t1 - t0:.2f} s, {SNAPSHOT_NAME} {t2 - t1:.2f} s")
    assert ph.kind == "placeholder" and ph.spot == 100.0 and ph.path == ""
    assert sp.kind == "snapshot" and sp.path.endswith(f"{SNAPSHOT_NAME}.yaml")
    assert sp.spot == pytest.approx(3901.35)
    lv_cfg = ph.spec.local_vol or LocalVolConfig()  # the reference spec uses the defaults
    assert ph.local_vol.local_var.shape == (lv_cfg.n_t, lv_cfg.n_k) == (400, 2401)
    assert np.all(ph.local_vol.local_var > 0)
    iv = ph.implied_vol_table([0.25, 1.0, 2.0], [-0.2, -0.1, 0.0, 0.1])
    assert list(iv.columns) == ["T", "k", "iv", "forward"] and len(iv) == 12
    assert (iv["iv"] > 0.05).all() and (iv["iv"] < 1.0).all()
    atm = ph.atm_table([0.25, 1.0, 3.0])
    assert atm["atm_vol"].iloc[1] == pytest.approx(0.20, abs=1e-6)  # the reference SSVI at 1y
    assert (atm["atm_skew"] < 0).all()
    lv = sp.local_vol_table([0.5, 1.0], [-0.1, 0.0, 0.1])
    assert list(lv.columns) == ["t", "k", "local_vol"] and (lv["local_vol"] > 0).all()
    # a snapshot file not on the grid builds from snapshots_root
    other = api.get_surface(cfg, "spx_2022-12-02")
    assert other.kind == "snapshot" and other.spot > 0
    with pytest.raises(api.MissingArtefact):
        api.get_surface(cfg, "no-such-surface")
    assert not cfg.cache_root.exists()


# --------------------------------------------------------------------------------------------
# hedging / marking records and the product catalogue
# --------------------------------------------------------------------------------------------


def test_hedging_and_marking_records(synthetic: dict[str, object]) -> None:
    cfg = _cfg(synthetic)
    run = api.get_hedging_run(cfg, HEDGING_RUN_ID)
    assert run.meta["product"] == "cliquet 1y" and run.meta["unit"] == "% of notional"
    stats = set(run.distribution["statistic"])
    assert {"value_0", "mean", "std", "q01", "q50", "q99", "zero_cost_mean", "recal_total"} <= stats
    assert len(run.regimes) == 3 and len(run.attribution) == 3 and not run.has_paths
    assert run.recalibration.empty
    runs = api.list_hedging_runs(cfg)
    assert runs.iloc[0]["run_id"] == HEDGING_RUN_ID and not runs.iloc[0]["has_paths"]
    m = api.get_marking(cfg)
    assert set(m.fits["status"]) == {"binding"}
    assert set(m.shadow_rotation["greek"]) >= {"usual", "recalibrated", "fee_shadow"}
    assert len(m.soft_skew) == 4 and len(m.skew_tradeoff) == 4
    assert any(name.startswith("rotation_spx_") for name in m.fit_specs)
    cat = api.get_products(cfg)
    assert {"name", "class", "unit", "fields", "in_store", "n_points"} <= set(cat.columns)
    assert cat.set_index("name").loc["Autocall", "in_store"]
    assert not cat.set_index("name").loc["Knock-in option", "in_store"]
    for cls in cat["class"]:
        module, _, name = cls.rpartition(".")
        assert hasattr(__import__(module, fromlist=[name]), name), cls


def test_stderr_walk_requires_the_twin_whatever_the_values() -> None:
    """A Monte Carlo column whose estimates all failed (all NaN) still needs its twin — the
    walk no longer skips an empty column (review:S2 F9); a text column read from a CSV that
    holds no value at all is typed ``object`` by :func:`api.read_study_csv`, not walked."""
    df = pd.DataFrame({"pnl": [float("nan"), float("nan")], "T": [1.0, 2.0]})
    assert api.columns_without_stderr(df) == ["pnl"]
    df["pnl_stderr"] = [0.1, 0.2]
    assert api.columns_without_stderr(df) == []


def test_read_study_csv_types_the_empty_text_columns(tmp_path: Path) -> None:
    """A runner's unfilled text column (``reason``) is read as ``object``, so it is not a
    numeric column without a stderr; a numeric column keeps its dtype and its pairing."""
    p = tmp_path / "t.csv"
    p.write_text("product,value,value_se,reason\nautocall,1.5,0.1,\ncliquet,2.5,0.2,\n")
    df = api.read_study_csv(p)
    assert df["reason"].dtype == object and df["value_stderr"].dtype == float
    assert api.columns_without_stderr(df) == []


def test_reattach_stderr_names_on_the_three_desk_conventions() -> None:
    """The M8b naming conventions (review:S2 F1): ``desk_<x>`` + ``<x>_se`` (tables A / D),
    ``<x>_desk`` + ``<x>_desk_se`` (table B) and ``<stem>_pnl_desk`` + ``<stem>_se`` (table C);
    an ambiguous stem is left unpaired rather than attached to the wrong column."""
    a = api.reattach_stderr_names(
        api.normalise_stderr_names(pd.DataFrame({"desk_mean": [1.0], "mean_se": [0.1]}))
    )
    assert list(a.columns) == ["desk_mean", "desk_mean_stderr"]
    b = api.reattach_stderr_names(
        api.normalise_stderr_names(pd.DataFrame({"leakage_desk": [1.0], "leakage_desk_se": [0.1]}))
    )
    assert list(b.columns) == ["leakage_desk", "leakage_desk_stderr"]
    c = api.reattach_stderr_names(
        api.normalise_stderr_names(
            pd.DataFrame({"recal_pnl_desk": [1.0], "recal_se": [0.1], "recal_by_date_desk": ["x"]})
        )
    )
    assert list(c.columns) == ["recal_pnl_desk", "recal_pnl_desk_stderr", "recal_by_date_desk"]
    assert api.columns_without_stderr(c) == []
    ambiguous = api.reattach_stderr_names(
        api.normalise_stderr_names(pd.DataFrame({"x_one": [1.0], "x_two": [2.0], "x_se": [0.1]}))
    )
    assert api.columns_without_stderr(ambiguous) == ["x_one", "x_two"]


@pytest.mark.skipif(
    not (ROOT / "outputs" / "m7" / "p1_marking_fits.csv").exists(), reason="outputs/m7 absent"
)
def test_real_marking_tables_schema() -> None:
    """The repository's M7 tables pass the stderr walk with the declared unpaired columns; the
    paired stage-3 means ``ssr_lsv_num_<T>`` are **not** exempted by the unpaired prefixes
    (review:S2 F5) and the particle-method summaries of the fits table are declared as recorded
    upstream without a standard error, not as exact (review:PC F10)."""
    m = api.get_marking(ViewerConfig())
    for name, df in m.tables().items():
        bad = api.columns_without_stderr(df, unpaired=api.marking_unpaired(name, df))
        assert bad == [], (name, bad)
    assert not m.fits.empty and not m.shadow_rotation.empty
    if not m.soft_skew.empty:
        unpaired = api.marking_unpaired("soft_skew", m.soft_skew)
        assert "ssr_lsv_num_1" in m.soft_skew.columns
        assert "ssr_lsv_num_1" not in unpaired and "ssr_lsv_num_1_stderr" in m.soft_skew.columns
        assert "ssr_lsv_1" in unpaired  # the unpaired stage-3 mean itself
    for col in ("mean_abs_L_minus_1", "svc_err_max", "corr_model_mean"):
        assert col in m.fits.columns
        assert not api.is_exact_column(col)
        assert col in api.UNPAIRED_UPSTREAM["marking_fits"]


@pytest.mark.skipif(
    not list((ROOT / "outputs" / "m8b").glob("*/*.json")), reason="outputs/m8b absent"
)
def test_real_hedging_runs_schema() -> None:
    """The repository's M8b task results list, load and pass the stderr walk; a run with a
    ``.pkl`` yields exactly one P&L sample per world path.  The summary tables A … D pass the
    walk under every desk-naming convention (review:S2 F1); a table a study wrote without an
    ``ok`` run is empty, which is tolerated — an *unpaired column* is not."""
    cfg = ViewerConfig()
    t0 = time.perf_counter()
    runs = api.list_hedging_runs(cfg)
    print(f"\nlist_hedging_runs: {len(runs)} runs in {time.perf_counter() - t0:.2f} s")
    assert not runs.empty and api.columns_without_stderr(runs) == []
    first = runs.iloc[0]["run_id"]
    rec = api.get_hedging_run(cfg, first)
    for name, df in rec.tables().items():
        unpaired = api.UNPAIRED_UPSTREAM["hedging_regimes"] if name == "regimes" else frozenset()
        assert api.columns_without_stderr(df, unpaired=unpaired) == [], (first, name)
    with_paths = runs[runs["has_paths"]]
    if not with_paths.empty:
        row = with_paths.iloc[0]
        pnl = api.get_hedging_pnl(cfg, row["run_id"])
        assert {"pnl_total", "pnl_product", "pnl_hedges", "costs"} <= set(pnl.columns)
        # one row per simulated world path: the samples are the run's paths, none dropped
        assert len(pnl) == int(row["n_paths_world"]), (row["run_id"], len(pnl))
    for name in ("A", "B", "C", "D"):
        if not (ROOT / "outputs" / "m8b" / f"m8b_table_{name}.csv").exists():
            continue
        table = api.get_hedging_table(cfg, name)
        assert api.columns_without_stderr(table) == [], (name, list(table.columns))
        print(f"m8b table {name}: {len(table)} rows, {len(table.columns)} columns")
    if (ROOT / "outputs" / "m8b" / "discriminator.csv").exists():
        disc = api.get_hedging_table(cfg, "discriminator")
        bad = api.columns_without_stderr(
            disc, unpaired=api.UNPAIRED_UPSTREAM["hedging_discriminator"]
        )
        assert bad == [], bad
        assert {"ssr_raw", "ssr_raw_stderr", "diff", "diff_stderr"} <= set(disc.columns)


@pytest.mark.skipif(
    not list((ROOT / "outputs" / "m8b").glob("C/static_*.json")), reason="outputs/m8b/C absent"
)
def test_hedging_runs_ignore_the_runners_own_caches() -> None:
    """``<outputs>/m8b/C/static_<product>__<policy>.json`` is the study-C static-prediction cache,
    not a task result: :func:`api.list_hedging_runs` must not try to load it (it has none of the
    TaskResult fields).  Found while reproducing review:S2 F1."""
    cfg = ViewerConfig()
    runs = api.list_hedging_runs(cfg)
    assert not runs.empty
    assert not any(str(r).startswith("static_") for r in runs["run_id"])
    assert set(runs["study"]) <= {"A", "B", "C", "D"}


# --------------------------------------------------------------------------------------------
# components, app skeleton and pages
# --------------------------------------------------------------------------------------------


def test_export_helpers() -> None:
    """Excel bytes round-trip through openpyxl; figure bytes for SVG / PNG (kaleido) and HTML."""
    import plotly.graph_objects as go

    from volsto.viewers import components

    df = pd.DataFrame({"T": [0.5, 1.0], "value": [1.5, 2.25], "value_stderr": [0.1, 0.2]})
    data = components.excel_bytes({"one": df, "two: [bad]/name?": df}, "x")
    import io

    back = pd.read_excel(io.BytesIO(data), sheet_name=None)
    assert set(back) == {"one", "two_ _bad__name_"}
    assert_frame_equal(back["one"], df)
    fig = go.Figure()
    components.stderr_bar(
        fig,
        df["T"].tolist(),
        df["value"].tolist(),
        df["value_stderr"].tolist(),
        name="v",
        unit="vp",
    )
    assert fig.data[0].error_y.array == (0.1, 0.2)
    assert components.hover_stderr([1.0], [0.1], "vp") == ["1 ± 0.1 vp"]
    assert (
        components.figure_bytes(fig, "html")[:15].lower().startswith(b"<html")
        or b"plotly" in components.figure_bytes(fig, "html")[:2000]
    )
    t0 = time.perf_counter()
    svg = components.figure_bytes(fig, "svg")
    png = components.figure_bytes(fig, "png")
    print(f"\nkaleido: svg {len(svg)} B, png {len(png)} B in {time.perf_counter() - t0:.2f} s")
    assert svg.lstrip().startswith(b"<svg") and png[:4] == b"\x89PNG"
    assert components.safe_filename("fwd ATM vol 1y→2y / x") == "fwd_ATM_vol_1y_2y_x"


def test_two_tables_with_the_same_name_render(tmp_path: Path) -> None:
    """Two tables and two figures with the *same* display name on one page: the default widget
    keys are made unique within the script run, so the page renders instead of raising
    ``StreamlitDuplicateElementKey`` (review:S2 F11); an explicit key still wins."""
    from streamlit.testing.v1 import AppTest

    script = tmp_path / "dup.py"
    script.write_text(
        "import pandas as pd\n"
        "from volsto.viewers import components as c\n"
        "df = pd.DataFrame({'T': [1.0], 'value': [2.0], 'value_stderr': [0.1]})\n"
        "import plotly.graph_objects as go\n"
        "for _ in range(2):\n"
        "    c.table_with_export(df, 'SSR')\n"
        "    c.figure_with_download(go.Figure(), 'SSR')\n"
        "c.table_with_export(df, 'SSR', key='explicit_ssr')\n"
    )
    at = AppTest.from_file(str(script), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.dataframe) == 3
    buttons = list(at.download_button)
    assert [b.label for b in buttons].count("Excel: SSR") == 3
    assert buttons[-1].key == "explicit_ssr"
    # a second run of the same script produces the same keys (the counter is per run)
    keys = [b.key for b in buttons]
    again = AppTest.from_file(str(script), default_timeout=120).run()
    assert [b.key for b in again.download_button] == keys
    assert len(set(keys)) == len(keys), keys


def test_pages_registry() -> None:
    """Eight pages in the owner's order, files present, titles unique, each exposing render()."""
    from volsto.viewers import app

    assert len(PAGES) == 8
    assert [t for _, t, _ in PAGES] == [
        "Surface & model",
        "Forward smile",
        "Forward skew",
        "Smile dynamics",
        "Model risk",
        "Product grid",
        "Marking / calibration",
        "Hedging",
    ]
    for path in page_paths():
        assert path.exists(), path
        src = path.read_text()
        assert "def render(cfg: ViewerConfig) -> None" in src and "run_if_streamlit(render)" in src
        assert "get_or_calibrate" not in src and "MonteCarlo" not in src
    assert [p for p, _, _ in app.page_list()] == page_paths()
    cmd = app.streamlit_command(ViewerConfig(), port=8765, headless=True)
    assert cmd[:4] == [sys.executable, "-m", "streamlit", "run"] and "--" in cmd
    assert cmd[cmd.index("--") + 1 :][:2] == ["--cache", str(ROOT / "cache")]


def test_app_help() -> None:
    out = subprocess.run(
        [sys.executable, "-m", "volsto.viewers.app", "--help"],
        capture_output=True,
        text=True,
        cwd=ROOT,
        timeout=120,
        check=False,
    )
    assert out.returncode == 0, out.stderr
    for flag in (
        "--cache",
        "--store",
        "--outputs",
        "--snapshots",
        "--grid",
        "--port",
        "--headless",
    ):
        assert flag in out.stdout


@pytest.mark.parametrize("page", [p.name for p in page_paths()])
def test_pages_render(
    synthetic: dict[str, object], page: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every page renders headless (streamlit AppTest) against the synthetic store: no
    exception, the title set, the store provenance line shown."""
    from streamlit.testing.v1 import AppTest

    cfg = _cfg(synthetic)
    for var, value in cfg.to_env().items():
        monkeypatch.setenv(var, value)
    monkeypatch.setenv("VOLSTO_VIEWER_CONFIG", str(Path(str(synthetic["base"])) / "no_viewer.yaml"))
    path = next(p for p in page_paths() if p.name == page)
    t0 = time.perf_counter()
    at = AppTest.from_file(str(path), default_timeout=120).run()
    print(f"\n{page}: rendered in {time.perf_counter() - t0:.2f} s")
    assert not at.exception, [e.value for e in at.exception]
    titles = [t.value for t in at.title]
    assert titles == [dict((f, t) for f, t, _ in PAGES)[page]]
    captions = " ".join(c.value for c in at.caption)
    assert "3 points" in captions and "code tag synthetic" in captions
    assert not cfg.cache_root.exists()
    assert os.environ["VOLSTO_STORE"] == str(cfg.store_root)


def test_page_header_on_empty_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty store: the header says so and prints the precompute command; no exception."""
    from streamlit.testing.v1 import AppTest

    cfg = ViewerConfig(
        cache_root=tmp_path / "c", store_root=tmp_path / "s", outputs_root=tmp_path / "o"
    )
    for var, value in cfg.to_env().items():
        monkeypatch.setenv(var, value)
    monkeypatch.setenv("VOLSTO_VIEWER_CONFIG", str(tmp_path / "none.yaml"))
    at = AppTest.from_file(str(page_paths()[4]), default_timeout=120).run()
    assert not at.exception
    codes = [c.value for c in at.code]
    assert any(c.startswith("volsto-precompute --grid") for c in codes)
    assert not (tmp_path / "c").exists()
