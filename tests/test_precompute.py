"""M9 Part 1 — the precompute CLI, the grid and the results store (SPEC §9.2).

**The toy calibration runs exactly once per pytest run, in the tests/conftest.py session
fixture** ``toy_build`` (the one sanctioned calibrating site of the test suite, owner's M9
rule): ``volsto-precompute`` on the 3-point toy grid (``configs/grids/toy.yaml``: placeholder
surface, 1F ν ∈ {0.25, 0.5} × ρ −0.7 × κ 1.5 and Table 8.2 at 2·10⁴ particles over a 1y
horizon, 4000 pricing paths, no risk) into a **temporary** leverage cache — never the
repository cache — as ``--shard 1/2`` then ``2/2``.  ``test_toy_precompute_end_to_end``
**consumes** that fixture for the calibrating-pass assertions (4 store rows, 3 calibrated + the
LV point, the per-shard cache manifests, the three ``calibrating`` log records) and performs
live only the non-calibrating steps on a private copy: ``--resume`` on both shards (nothing
recomputed, per-point manifest entries identical), the ``--diagnostics`` backfill of a cache
entry whose report was removed (a repricing, no calibration), then store and cache moved
together and read back unchanged (relocatability).  The fixture directory stays byte-identical.

Every other test here reads YAML, a store or a synthetic frame and calibrates nothing: the grid
counts and shards, the default-grid ``--dry-run`` projection (the repository cache manifest is
read and unchanged), the risk plan and the cost model of the configured ladder, ``--only`` and
its errors, the tier-aware ``--resume`` rule (:func:`~volsto.viewers.precompute.pending_steps`),
the per-point failure handling (a monkeypatched ``process_point``), the provenance strings
(relative grid path, path-free ``argv``) and the store's stderr invariant on a synthetic marking
row.  The measured-cost report (``volsto-precompute report``) and ``--cost-from`` are checked on
a synthetic run-record set (a current record and an old-format one whose missing fields must be
reported absent, never filled in) and on the toy fixture's store, which they only read (output
to a temporary directory; the fixture stays byte-identical).  Wall clock is reported, never
asserted.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from conftest import TOY_SHARDS, ToyBuild

from volsto.calibration.cache import LeverageCache
from volsto.risk.ladders import default_buckets
from volsto.viewers import precompute
from volsto.viewers.grid import (
    DEGENERATE_ONE_FACTOR,
    M5_FWD_VAR_BUCKETS,
    MODES,
    GridPoint,
    RiskSettings,
    count_by,
    enumerate_points,
    load_grid,
    parse_shard,
    resolve_axis_point,
    shard,
)
from volsto.viewers.store import (
    TABLES,
    PointResult,
    ResultsStore,
    StoreReader,
    mc_columns_without_stderr,
)

ROOT = Path(__file__).resolve().parents[1]
TOY_GRID = ROOT / "configs" / "grids" / "toy.yaml"
DEFAULT_GRID = ROOT / "configs" / "grids" / "default.yaml"
REPO_CACHE = ROOT / "cache"


# --------------------------------------------------------------------------------------------
# grid (no computation)
# --------------------------------------------------------------------------------------------


def test_default_grid_counts_and_shards() -> None:
    """Default grid: 105 one-factor combinations on the placeholder = 90 LSV points + the LV
    point (ν = 0 collapses), Table 8.2, 12 marking fits per SPX snapshot, one LV point per
    surface; the degenerate ω = 1, 2, 3 points are present; the shards partition the points
    exactly once each and the enumeration is deterministic."""
    grid = load_grid(DEFAULT_GRID)
    assert grid.particle.n_particles == 800_000
    assert grid.one_factor is not None and grid.one_factor.n_combinations == 105
    points = enumerate_points(grid)
    counts = count_by(points)
    assert counts[("placeholder", "one_factor")] == 90
    assert counts[("placeholder", "two_factor")] == 1
    assert counts[("placeholder", "lv")] == 1
    for s in ("spx_2022-12-30", "spx_2022-09-15", "spx_2022-12-02"):
        assert counts[(s, "marking")] == 12
        assert counts[(s, "lv")] == 1
    assert len(points) == 131
    ids = [p.id for p in points]
    assert len(set(ids)) == len(ids)
    assert [p.id for p in enumerate_points(grid)] == ids  # deterministic
    for nu, rho, kappa in DEGENERATE_ONE_FACTOR:
        p = resolve_axis_point(points, "placeholder", nu, rho, kappa)
        assert p is not None and p.mode == "one_factor" and p.spec.model.omega == 2 * nu
    lv = resolve_axis_point(points, "placeholder", 0.0, -0.9, 4.0)
    assert lv is not None and lv.mode == "lv" and lv.id.startswith("lv:")
    # modes appear in the canonical order within a surface
    for surface in {p.surface for p in points}:
        modes = [p.mode for p in points if p.surface == surface]
        assert modes == sorted(modes, key=MODES.index)
    for n in (1, 2, 3, 7):
        parts = [shard(points, i, n) for i in range(1, n + 1)]
        assert sorted(q.id for part in parts for q in part) == sorted(ids)
        assert parts[0] == points[0::n]
    with pytest.raises(ValueError):
        shard(points, 0, 2)
    with pytest.raises(ValueError):
        parse_shard("3/2")
    assert parse_shard("2/5") == (2, 5)
    marking = [p for p in points if p.mode == "marking"]
    assert all(not p.resolved and p.cache_key is None for p in marking)
    assert marking[0].id == "marking:spx_2022-12-30:ssr0.75:eps0.05"
    # the named fit (SPEC §15 Part 3): the default grid marks with the desk's fit, which reads
    # each snapshot's SABRW fits; a placeholder has no quotes, so a desk grid refuses it
    assert grid.marking is not None and grid.marking.fit == "desk"
    assert all(p.fit == "desk" and p.snapshot is not None for p in marking)
    assert marking[0].snapshot == "configs/surfaces/snapshots/hdn_2022H2/spx_2022-12-30.yaml"
    with pytest.raises(ValueError, match=r"marking\.fit"):
        dataclasses.replace(grid.marking, fit="house")
    bare = dataclasses.replace(grid.surfaces[0], marking=True)
    with pytest.raises(ValueError, match="no quotes"):
        dataclasses.replace(grid, surfaces=(bare, *grid.surfaces[1:]))
    toy = load_grid(ROOT / "configs" / "grids" / "toy_marking.yaml")
    assert toy.marking is not None and toy.marking.fit == "m7"


def _cache_keys(root: Path) -> set[str]:
    m = LeverageCache(root).manifest()
    return set() if m.empty else {str(k) for k in m["key"]}


def test_dry_run_projects_without_computing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--dry-run`` on the default grid against the repository cache: prints the resolved store
    and cache roots, the counts (105 one-factor combinations), the per-point cost lines
    (overhead, calibration from the cache manifest's measured wall times, diagnostics repricing)
    and one row per tier — the light tier twice, at the configured 3-bucket fwd-var ladder (the
    owner's grid decision) and at the 20-bucket alternative, so the owner can choose.  It writes nothing into the store and
    **calibrates nothing**: the repository cache manifest is byte-identical afterwards."""
    if not (REPO_CACHE / "manifest.parquet").is_file():
        pytest.skip(f"no leverage-cache manifest at {REPO_CACHE / 'manifest.parquet'}")
    store = tmp_path / "store"
    keys_before = _cache_keys(REPO_CACHE)
    manifest_before = (REPO_CACHE / "manifest.parquet").read_bytes()
    rc = precompute.main(
        [
            "--grid",
            str(DEFAULT_GRID),
            "--store",
            str(store),
            "--cache",
            str(REPO_CACHE),
            "--dry-run",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "105 combinations" in out and "90 LSV calibrations" in out
    assert "131 points" in out
    assert f"store {store.resolve()}" in out and f"cache {REPO_CACHE.resolve()}" in out
    for tier in ("none", "light", "full"):
        assert f"{tier:<6} 131 points" in out  # " *light  131 points" marks the requested tier
    assert out.count("light  131 points") == 2  # the configured ladder and the alternative one
    assert "20-bucket fwd-var ladder" in out and "3-bucket fwd-var ladder" in out
    assert "overhead: " in out and "diagnostics cost: " in out
    assert "dry run: nothing computed" in out
    manifest = LeverageCache(REPO_CACHE).manifest()
    at_8e5 = manifest[(manifest["n_particles"] == 800_000) & (manifest["horizon"] == 3.0)]
    if not at_8e5.empty:
        assert "cache manifest median" in out
    assert not store.exists()  # a dry run creates nothing, not even the store root
    assert _cache_keys(REPO_CACHE) == keys_before  # nothing calibrated, nothing stored
    assert (REPO_CACHE / "manifest.parquet").read_bytes() == manifest_before
    print("\n" + out)


# --------------------------------------------------------------------------------------------
# the toy precompute (built once in tests/conftest.py::toy_build) — consumed here
# --------------------------------------------------------------------------------------------


def _read_all(root: Path) -> dict[str, pd.DataFrame]:
    r = StoreReader(root)
    return {t: r.store.table(t) for t in TABLES}


def _file_snapshot(root: Path) -> dict[str, tuple[int, float]]:
    """``relative path → (size, mtime)`` of every file below ``root``."""
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_toy_precompute_end_to_end(
    toy_build: ToyBuild,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The toy calibration of the conftest fixture (module docstring), asserted from what it
    recorded and from its store: shards 1/2 and 2/2 into a temporary cache, 3 calibrations
    (one ``calibrating`` record in shard 1/2, two in 2/2), stderr on every MC column, the LV
    point present; then, live on a private copy: resume on both shards without recomputation
    (identical manifest entries, hits only) and relocatability (store + cache moved together,
    read unchanged, resume against the moved pair).  The fixture directory is untouched."""
    toy = toy_build.require()
    store_a, cache_a = toy.store_root, toy.cache_root
    fixture_before = _file_snapshot(toy.base)

    # -- the calibrating pass, as recorded by the fixture -------------------------------------
    assert toy.return_codes == (0, 0)
    for s in TOY_SHARDS:
        assert "projected wall clock" in toy.stdout[s] and "ETA" in toy.stdout[s], s
    # shard 1/2 = points[0::2] = the LV point + 1F ν = 0.5; shard 2/2 = 1F ν = 0.25 + Table 8.2;
    # each cache miss logs one "calibrating" record in the precompute and one in the cache
    assert [len(toy.calibrating_messages(s)) for s in TOY_SHARDS] == [1, 2]
    assert len(toy.calibrating_messages()) == 3, toy.calibrating_messages()
    assert all("calibrating at 20000 particles" in m for m in toy.calibrating_messages())
    cache_records = toy.calibrating_messages(logger="volsto.calibration.cache")
    assert [len(toy.calibrating_messages(s, "volsto.calibration.cache")) for s in TOY_SHARDS] == [
        1,
        2,
    ] and len(cache_records) == 3
    assert [len(toy.manifests[s]) for s in TOY_SHARDS] == [1, 3]
    # the done line states what it recalibrated (owner rule): 1 miss in shard 1/2, 2 in 2/2
    assert "1 calibrated / 0 cache hits / 1 without leverage" in toy.stdout["1/2"]
    assert "2 calibrated / 0 cache hits / 0 without leverage" in toy.stdout["2/2"]

    reader = StoreReader(store_a)
    points = reader.points()
    assert len(points) == 4, points[["label", "mode"]]
    assert sorted(points["mode"]) == ["lv", "one_factor", "one_factor", "two_factor"]
    calibrated = points[points["mode"] != "lv"]
    assert calibrated["calibrated_this_run"].all()
    assert np.isfinite(calibrated["calibration_seconds"]).all()
    assert (calibrated["n_particles"] == 20_000).all()
    cache = LeverageCache(cache_a)
    man = cache.manifest()
    # every calibration ran its diagnostics (run_diagnostics at the point's pricing SimConfig),
    # so the report is in the cache and the store's summary is finite — not NaN
    for key in calibrated["cache_key"]:
        assert (cache.root / str(key) / "diagnostics.json").exists(), key
    assert np.isfinite(calibrated["max_abs_error_vp"]).all()
    assert np.isfinite(calibrated["max_z"]).all()
    assert (calibrated["wall_diagnostics_s"] > 0).all()
    assert not calibrated["diagnostics_backfilled"].any()
    assert len(man) == 3 and set(man["key"]) == set(calibrated["cache_key"])
    assert {r["key"] for r in toy.manifests["2/2"]} == set(man["key"])
    assert (man["n_particles"] == 20_000).all() and (man["horizon"] == 1.0).all()
    lv = points[points["mode"] == "lv"].iloc[0]
    assert lv["point_id"].startswith("lv:") and lv["cache_key"] == ""
    # tables: every point priced, both forward windows, SSR within the 1y horizon, Var(V) for LSV
    products = reader.products()
    assert set(products["point_id"]) == set(points["point_id"])
    assert set(products["key"]) == {"atm_vol", "vs_vol", "volswap_vol", "cliquet_1y"}
    fv = reader.forward_vols()
    assert sorted(set(zip(fv["t1"], fv["t2"]))) == [(1.0, 2.0), (2.0, 3.0)]
    assert len(fv) == 8
    smile = reader.forward_smile()
    assert len(smile) == 4 * 2 * 8  # 7 strikes + ATMF per window
    # both windows are kept whatever the horizon; beyond it (the toy's is 1y) the last leverage
    # slice is held constant, which the stored flag states for the pages
    for df in (fv, smile):
        assert df["beyond_horizon"].dtype == bool
        assert set(zip(df["t2"], df["beyond_horizon"])) == {(2.0, True), (3.0, True)}
    ssr = reader.ssr()
    assert sorted(set(ssr["T"])) == [0.25, 0.5, 1.0] and len(ssr) == 12
    varv = reader.varv()
    assert set(varv["point_id"]) == set(calibrated["point_id"])
    assert set(varv["T"]) == {1.0}
    assert reader.risk().empty
    fixture_tables = _read_all(store_a)
    for name, df in fixture_tables.items():
        assert mc_columns_without_stderr(df) == [], (name, mc_columns_without_stderr(df))
    for c in ("iv", "price"):
        assert (smile[f"{c}_stderr"] > 0).all()
    assert (products["value_stderr"] > 0).all()
    manifest_before = json.dumps(reader.manifest()["points"], sort_keys=True)
    walls = points.set_index("label")[
        ["wall_seconds", "wall_calibration_s", "wall_pricing_s", "wall_analytics_s"]
    ]

    # -- resume on a private copy (the fixture stays intact): nothing recomputed, hits only ----
    shutil.copytree(toy.base, tmp_path / "A")  # store + cache + outputs together
    store_c, cache_c = tmp_path / "A" / "store", tmp_path / "A" / "cache"
    base = ["--grid", str(TOY_GRID), "--cache", str(cache_c), "--store", str(store_c)]
    caplog.clear()
    capsys.readouterr()
    with caplog.at_level(logging.INFO):
        assert precompute.main([*base, "--shard", "1/2", "--resume"]) == 0
        assert precompute.main([*base, "--shard", "2/2", "--resume"]) == 0
    messages = [r.getMessage() for r in caplog.records]
    assert not any("calibrating" in m or "cache miss" in m for m in messages), messages
    assert sum("resume: skipping" in m for m in messages) == 4
    out = capsys.readouterr().out
    assert out.count("resume: 2 points already done, 0 to compute") == 2
    assert json.dumps(StoreReader(store_c).manifest()["points"], sort_keys=True) == manifest_before
    assert len(LeverageCache(cache_c).manifest()) == 3
    after = _read_all(store_c)
    for name, df in fixture_tables.items():
        pd.testing.assert_frame_equal(df, after[name])
    assert len(StoreReader(store_c).manifest()["runs"]) == 4  # 2 builds + 2 resumes

    # -- relocatability: move store and cache together, read unchanged -----------------------
    shutil.move(str(tmp_path / "A"), str(tmp_path / "B"))
    store_b, cache_b = tmp_path / "B" / "store", tmp_path / "B" / "cache"
    moved = _read_all(store_b)
    for name in TABLES:
        pd.testing.assert_frame_equal(fixture_tables[name], moved[name])
    moved_cache = LeverageCache(cache_b)
    for key in calibrated["cache_key"]:
        assert (moved_cache.root / key / "leverage.npz").exists()
    # the moved manifest names the points and nothing absolute — argv included: the run record
    # replaces the --store / --cache values with <store> / <cache> and the grid with its
    # repository-relative path, so no machine path survives anywhere in it
    manifest_b = StoreReader(store_b).manifest()
    text = json.dumps(manifest_b, default=str)
    assert str(tmp_path) not in text
    assert str(toy.base) not in text and str(toy.root) not in text
    runs = manifest_b["runs"]
    assert {r["grid_path"] for r in runs} == {"configs/grids/toy.yaml"}
    for r in runs:
        assert "<store>" in r["argv"] and "<cache>" in r["argv"]
        assert "configs/grids/toy.yaml" in r["argv"]
        assert r["n_calibrated"] + r["n_cache_hits"] <= len(r["points_computed"])
        assert r["failed"] == []
    # a resume against the moved pair still finds everything done
    caplog.clear()
    with caplog.at_level(logging.INFO):
        rc = precompute.main(
            ["--grid", str(TOY_GRID), "--cache", str(cache_b), "--store", str(store_b), "--resume"]
        )
    assert rc == 0
    assert sum("resume: skipping" in r.getMessage() for r in caplog.records) == 4
    capsys.readouterr()
    assert _file_snapshot(toy.base) == fixture_before, "the fixture directory changed"

    print(
        f"\ntoy precompute (conftest fixture, built by {toy.built_by}): 4 points (3 calibrations "
        f"at 2e4 particles, 1y horizon, 4000 paths) in {toy.total_wall_s:.1f} s wall — shard 1/2 "
        f"{toy.wall_s['1/2']:.1f} s, shard 2/2 {toy.wall_s['2/2']:.1f} s (sequential)"
    )
    print(walls.round(2).to_string())
    print("calibration wall times (cache manifest, s):", man["wall_time"].round(2).tolist())


def test_diagnostics_backfill_reprices_without_calibrating(
    toy_build: ToyBuild,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--resume --diagnostics`` on a private copy of the fixture whose cache lost one entry's
    ``diagnostics.json``: that point alone is refreshed, by repricing the pillar surface at the
    point's pricing SimConfig (:func:`~volsto.calibration.diagnostics.reprice_surface`) — **no
    calibration** — the report lands back in the cache, the store's ``max_abs_error_vp`` /
    ``max_z`` are finite again and the row records the refresh; the other points are untouched
    and a plain ``--resume`` sees nothing to do."""
    toy = toy_build.require()
    shutil.copytree(toy.base, tmp_path / "D")
    store_d, cache_d = tmp_path / "D" / "store", tmp_path / "D" / "cache"
    before = StoreReader(store_d).points().set_index("point_id")
    lsv = before[before["mode"] != "lv"].iloc[0]
    point_id, key = str(lsv.name), str(lsv["cache_key"])
    (cache_d / key / "diagnostics.json").unlink()

    base = ["--grid", str(TOY_GRID), "--store", str(store_d), "--cache", str(cache_d)]
    assert precompute.main([*base, "--resume"]) == 0  # the report is not part of the plain rule
    assert "resume: 4 points already done, 0 to compute" in capsys.readouterr().out
    # the projection charges the repricing of that one hit — and of no other point
    assert precompute.main([*base, "--dry-run", "--diagnostics"]) == 0
    assert "1 backfills" in capsys.readouterr().out
    caplog.clear()
    with caplog.at_level(logging.INFO):
        assert precompute.main([*base, "--resume", "--diagnostics"]) == 0
    messages = [r.getMessage() for r in caplog.records]
    assert not any("calibrating" in m or "cache miss" in m for m in messages), messages
    assert sum("backfilled" in m for m in messages) >= 1
    out = capsys.readouterr().out
    assert "resume: 3 points already done, 1 to compute (1 of them refreshed" in out
    assert "1 diagnostics backfilled" in out

    assert (cache_d / key / "diagnostics.json").exists()
    after = StoreReader(store_d).points().set_index("point_id")
    row = after.loc[point_id]
    assert bool(row["diagnostics_backfilled"]) and row["updated_steps"] == "diagnostics"
    assert np.isfinite(row["max_abs_error_vp"]) and np.isfinite(row["max_z"])
    assert row["wall_diagnostics_s"] > 0
    # the run record states what this run did: no calibration, one hit, one backfill
    record = ResultsStore(store_d).runs()[-1]
    assert (record["n_calibrated"], record["n_cache_hits"]) == (0, 1)
    assert record["n_diagnostics_backfilled"] == 1 and record["diagnostics"] is True
    assert [w["steps"] for w in record["points_computed"]] == [["diagnostics"]]
    untouched = [i for i in after.index if i != point_id]
    pd.testing.assert_frame_equal(before.loc[untouched], after.loc[untouched][before.columns])
    print(
        f"\ndiagnostics backfill (toy, {int(before['pricing_n_paths'].iloc[0])} paths): "
        f"{row['wall_diagnostics_s']:.1f} s, max |error| {row['max_abs_error_vp']:.3f} vp, "
        f"max z {row['max_z']:.2f}"
    )


def test_store_reader_on_missing_store(tmp_path: Path) -> None:
    """An empty store reads as empty frames and a missing point raises KeyError — and reading
    creates nothing: neither :class:`ResultsStore` nor :class:`StoreReader` makes a directory
    (the writers do), so a viewer or a dry run pointed at a missing root leaves it missing."""
    missing = tmp_path / "empty"
    r = StoreReader(missing)
    assert r.points().empty and r.products().empty and r.risk().empty
    assert r.manifest()["points"] == {}
    with pytest.raises(KeyError):
        r.point("nope")
    store = ResultsStore(missing)
    assert store.point_ids() == [] and store.runs() == [] and not store.has_point("x")
    assert not missing.exists()


def test_store_compaction_roundtrip(tmp_path: Path) -> None:
    """``compact()`` writes merged tables; reading with and without the per-point files gives
    the same frames (the compacted rows fill in for removed point directories)."""
    from volsto.viewers.store import PointResult

    store = ResultsStore(tmp_path / "s")
    for i in range(3):
        row = {"label": f"p{i}", "surface": "x", "mode": "lv", "status": "ok"}
        tables = {
            "products": pd.DataFrame(
                [
                    {
                        "product": "a",
                        "quantity": "v",
                        "key": "a",
                        "value": float(i),
                        "value_stderr": 0.1,
                        "unit": "",
                    }
                ]
            )
        }
        store.write_point(PointResult(f"id{i}", row, tables, {"label": f"p{i}"}))
    full = store.table("products")
    assert len(full) == 3
    store.compact()
    shutil.rmtree(store.point_dir("id1"))
    merged = store.table("products")
    assert sorted(merged["point_id"]) == ["id0", "id1", "id2"]
    assert sorted(store.point_ids()) == ["id0", "id2"]


# --------------------------------------------------------------------------------------------
# cost model, selection, resume and failures (no computation)
# --------------------------------------------------------------------------------------------


def test_risk_plan_counts_follow_the_grid_settings() -> None:
    """The light tier's budget is derived from the configured ladder, not hard-coded: with the
    M5 ladder (``fwd_var_buckets: 20``) one LSV point costs 34 leverage recalibrations and 72
    CRN pricings, with the coarse ladder on the three light pillars 17 and 38 — the counting
    rule of :func:`~volsto.viewers.precompute.risk_plan` (1 base + 2 x 4 recalibrating delta
    regimes + buckets + 1 + pillars + 1 calibrations; per product 1 + 2 x 5 + buckets + 1 +
    pillars + 1 pricings), measured against a counting pass of the RiskEngine plan on a
    non-simulating engine (reported in the stream's measurements, too slow for the suite).  The
    full tier is the measured engine constant, which no grid setting changes."""
    # the default grid runs the light tier on the coarse 3-bucket ladder (owner's decision of
    # 2026-09-16: the grid is for cross-model comparison); 20 stays the library default
    coarse = load_grid(DEFAULT_GRID).risk
    assert (coarse.tier, coarse.fwd_var_buckets, coarse.light_pillars) == (
        "light",
        3,
        (0.25, 1.0, 3.0),
    )
    assert M5_FWD_VAR_BUCKETS == len(default_buckets()) == 20
    m5 = dataclasses.replace(coarse, fwd_var_buckets=M5_FWD_VAR_BUCKETS)
    assert precompute.risk_plan("none", m5) == (0, 0)
    assert precompute.risk_plan("light", m5) == (34, 72)
    assert precompute.risk_plan("light", m5, 3) == (17, 38)
    assert precompute.risk_plan("full", m5) == (
        precompute.RISK_FULL_CALIBRATIONS,
        precompute.RISK_FULL_PRICINGS,
    )
    assert precompute.alternative_buckets(m5) == 3
    # the coarse ladder is the tent over the light pillars; the M5 one is the library's
    assert precompute.light_buckets(m5) == default_buckets()
    assert precompute.light_buckets(coarse) == ((0.0, 0.25), (0.25, 1.0), (1.0, 3.0))
    assert precompute.risk_plan("light", coarse) == (17, 38)
    assert precompute.alternative_buckets(coarse) == 20
    risk = m5  # the pillar-count check below starts from the M5 ladder
    # one more pillar moves both counts by one calibration and one pricing per product
    four = dataclasses.replace(risk, light_pillars=(0.25, 1.0, 2.0, 3.0))
    assert precompute.risk_plan("light", four) == (35, 74)
    with pytest.raises(ValueError):
        precompute.risk_plan("bogus", risk)
    with pytest.raises(ValueError):  # the grid rejects any other bucket count
        RiskSettings(fwd_var_buckets=7)


def test_cost_model_charges_the_overhead_and_the_diagnostics(tmp_path: Path) -> None:
    """Every point pays the measured per-point overhead (market build + leverage load / Dupire);
    a cache miss pays the calibration **and** its diagnostics repricing, a hit neither unless
    ``--diagnostics`` backfills it; a refresh pays the overhead plus its own steps only."""
    grid = load_grid(TOY_GRID)
    cost = precompute.default_cost_model(LeverageCache(tmp_path / "cache"), grid)
    assert cost.calibration_s == precompute.CALIBRATION_S_FALLBACK  # empty cache manifest
    assert "empty cache manifest" in cost.calibration_source
    assert cost.overhead_s == precompute.OVERHEAD_S_FALLBACK
    scale = grid.pricing.n_paths / precompute.FALLBACK_N_PATHS
    assert cost.diagnostics_s == pytest.approx(precompute.DIAGNOSTICS_S_FALLBACK * scale)
    base = cost.overhead_s + cost.pricing_s + cost.analytics_s
    assert cost.point_s("none", calibrates=True, miss=False) == pytest.approx(base)
    assert cost.point_s("none", calibrates=True, miss=True) == pytest.approx(
        base + cost.calibration_s + cost.diagnostics_s
    )
    assert cost.point_s("none", calibrates=True, miss=False, backfill=True) == pytest.approx(
        base + cost.diagnostics_s
    )
    # a miss already pays the diagnostics: --diagnostics does not charge it twice
    assert cost.point_s("none", calibrates=True, miss=True, backfill=True) == pytest.approx(
        base + cost.calibration_s + cost.diagnostics_s
    )
    assert cost.steps_s("light", (), calibrates=True, miss=False) == 0.0
    assert cost.steps_s("light", ("risk",), calibrates=True, miss=False) == pytest.approx(
        cost.overhead_s + cost.risk_s("light", True)
    )
    assert cost.steps_s("none", ("diagnostics",), calibrates=True, miss=False) == pytest.approx(
        cost.overhead_s + cost.diagnostics_s
    )
    # the light row of the projection is priced at the configured count, the alternative beside it
    points = enumerate_points(grid)
    table = precompute.projection_table(points, LeverageCache(tmp_path / "cache"), cost, grid.risk)
    assert list(table["tier"]) == ["none", "light", "light", "full"]
    assert list(table["fwd_var_buckets"]) == [0, 20, 3, 0]
    assert list(table["risk_calibrations"]) == [0, 34, 17, precompute.RISK_FULL_CALIBRATIONS]
    assert (table["points"] == len(points)).all()
    assert (table["cache_misses"] == 3).all()  # the empty cache: the LV point calibrates nothing


def test_only_selects_points_and_rejects_unknown_ids(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--only ID [ID ...]`` filters the shard by the stable point ids (the leverage-cache key,
    ``lv:<hash>``, ``marking:<label>``); an id outside the grid is an error listing the nearest
    ids, an id of another shard an error naming that, and neither computes anything."""
    points = enumerate_points(load_grid(TOY_GRID))
    lv = next(p for p in points if p.mode == "lv")
    lsv = next(p for p in points if p.mode == "one_factor")
    base = [
        "--grid",
        str(TOY_GRID),
        "--store",
        str(tmp_path / "store"),
        "--cache",
        str(tmp_path / "cache"),
        "--dry-run",
    ]
    assert precompute.main([*base, "--only", lv.id, lsv.id]) == 0
    out = capsys.readouterr().out
    assert f"only: 2 of the shard's {len(points)} points selected" in out
    assert "dry run: nothing computed" in out

    assert precompute.main([*base, "--only", "lv:deadbeef"]) == 2
    err = capsys.readouterr().err
    assert "unknown point id" in err and "nearest" in err and lv.id[:8] in err

    shard_2 = {q.id for q in shard(points, 2, 2)}
    off = next(q for q in points if q.id not in shard_2)
    assert precompute.main([*base, "--shard", "2/2", "--only", off.id]) == 2
    assert "belong to another shard" in capsys.readouterr().err
    assert not (tmp_path / "store").exists()


def test_pending_steps_is_tier_aware(toy_build: ToyBuild, tmp_path: Path) -> None:
    """The ``--resume`` rule compares the stored risk tier with the requested one: the toy store
    holds every point at tier ``none``, so a resume at ``none`` skips them all and a resume at
    ``light`` / ``full`` asks for the risk step only (never ``all``: the pricing is kept).  A
    point absent from the store is ``("all",)``; with ``--diagnostics`` nothing is pending
    because every calibration stored its report.  Reading a missing store creates nothing."""
    toy = toy_build.require()
    store, cache = ResultsStore(toy.store_root), LeverageCache(toy.cache_root)
    empty = ResultsStore(tmp_path / "empty")
    points = enumerate_points(load_grid(TOY_GRID))
    assert len(points) == 4
    for point in points:
        assert precompute.pending_steps(point, store, cache, "none") == ()
        assert precompute.is_done(point, store, cache, "none")
        assert precompute.pending_steps(point, store, cache, "light") == ("risk",)
        assert precompute.pending_steps(point, store, cache, "full") == ("risk",)
        assert not precompute.is_done(point, store, cache, "light")
        assert precompute.pending_steps(point, store, cache, "none", diagnostics=True) == ()
        assert precompute.pending_steps(point, empty, cache, "none") == ("all",)
    assert not (tmp_path / "empty").exists()


def test_resume_below_the_requested_tier_plans_a_risk_refresh(
    toy_build: ToyBuild, capsys: pytest.CaptureFixture[str]
) -> None:
    """The CLI's resume plan on the toy store (read-only, ``--dry-run``): at the grid's tier
    (none) everything is done; at ``--risk light`` the four stored points are refreshed for the
    risk step only — what the read API's missing-risk command (``--only <id> --resume --risk
    light``) relies on."""
    toy = toy_build.require()
    base = [
        "--grid",
        str(TOY_GRID),
        "--store",
        str(toy.store_root),
        "--cache",
        str(toy.cache_root),
        "--resume",
        "--dry-run",
    ]
    assert precompute.main(base) == 0
    assert "resume: 4 points already done, 0 to compute" in capsys.readouterr().out
    assert precompute.main([*base, "--risk", "light"]) == 0
    out = capsys.readouterr().out
    assert "resume: 0 points already done, 4 to compute (4 of them refreshed" in out
    point = enumerate_points(load_grid(TOY_GRID))[0]
    assert precompute.main([*base, "--risk", "light", "--only", point.id]) == 0
    out = capsys.readouterr().out
    assert "only: 1 of the shard's 4 points selected" in out
    assert "resume: 0 points already done, 1 to compute (1 of them refreshed" in out


def test_failed_point_is_recorded_and_the_run_exits_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A point that raises does not tear the shard down: the traceback is logged, the run record
    keeps a ``failed`` entry per point (id, label, error, traceback) and is written, the other
    points continue and the CLI exits 1 (a later ``--resume`` picks the failures up).  The
    compute step is monkeypatched to raise, so nothing is calibrated or priced."""

    def boom(
        point: GridPoint,
        grid: Any,
        cache: Any,
        tier: str,
        store: Any,
        steps: Any,
        *,
        diagnostics: bool = False,
    ) -> dict[str, Any]:
        raise RuntimeError(f"boom on {point.label}")

    monkeypatch.setattr(precompute, "process_point", boom)
    store_root = tmp_path / "store"
    rc = precompute.main(
        [
            "--grid",
            str(TOY_GRID),
            "--store",
            str(store_root),
            "--cache",
            str(tmp_path / "cache"),
        ]
    )
    out = capsys.readouterr().out
    assert rc == 1
    assert "done: 0 points computed (0 calibrated / 0 cache hits" in out
    assert "4 failed" in out and "failed points" in out
    runs = ResultsStore(store_root).runs()
    assert len(runs) == 1
    record = runs[0]
    assert record["points_computed"] == [] and record["n_calibrated"] == 0
    failed = record["failed"]
    assert len(failed) == 4
    assert {f["label"] for f in failed} == {p.label for p in enumerate_points(load_grid(TOY_GRID))}
    for f in failed:
        assert f["error"].startswith("RuntimeError: boom on ")
        assert "Traceback (most recent call last)" in f["traceback"]
    assert ResultsStore(store_root).point_ids() == []  # nothing was written for a failed point


def test_run_record_stores_no_absolute_path() -> None:
    """Provenance strings (F5 / F12): the grid path is stored relative to the repository root
    (a grid outside it keeps its basename), the ``argv`` has its path values replaced, and the
    store / cache defaults are resolved against the repository root like the grid, so the CLI
    finds the same pair from any working directory."""
    assert precompute.grid_path_for_record(TOY_GRID) == "configs/grids/toy.yaml"
    assert precompute.grid_path_for_record(Path("/tmp/elsewhere/vm_grid.yaml")) == "vm_grid.yaml"
    argv = [
        "--grid",
        str(TOY_GRID),
        "--store",
        "/mnt/vm/store",
        "--cache=/mnt/vm/cache",
        "--shard",
        "1/4",
        "--resume",
    ]
    assert precompute.provenance_argv(argv, "configs/grids/toy.yaml") == [
        "--grid",
        "configs/grids/toy.yaml",
        "--store",
        "<store>",
        "--cache=<cache>",
        "--shard",
        "1/4",
        "--resume",
    ]
    assert precompute.DEFAULT_STORE == ROOT / "outputs" / "store"
    assert precompute.DEFAULT_CACHE == ROOT / "cache"
    assert precompute.DEFAULT_GRID == DEFAULT_GRID


def test_mc_columns_without_stderr_on_a_marking_row() -> None:
    """The store's stderr invariant on a marking point's row (the mode the toy grid has none of):
    the ``skew_gap_<T>`` naked-skew gaps and the ``fit_*`` summaries are deterministic outputs of
    the P1 fit and carry no ``_stderr`` twin, while a genuine Monte Carlo column without its twin
    is still caught."""
    row: dict[str, Any] = {
        "point_id": "marking:spx_2022-12-30:ssr0.75:eps0.05",
        "label": "spx_2022-12-30 marking ssr0.75 eps0.05",
        "surface": "spx_2022-12-30",
        "mode": "marking",
        "status": "binding",
        "nu": 1.93,
        "theta": 0.245,
        "k1": 5.35,
        "k2": 0.2,
        "rho12": 0.0,
        "rho_SX1": -0.72,
        "rho_SX2": -0.49,
        "axis_nu": float("nan"),
        "ssr_target": 0.75,
        "skew_eps": 0.05,
        "n_particles": 800_000,
        "horizon": 3.0,
        "calibration_seconds": 141.2,
        "mean_abs_L_minus_1": 0.031,
        "max_abs_error_vp": 0.28,
        "max_z": 1.6,
        "fit_status": "binding",
        "fit_messages": "nu at cap",
        "fit_mean_skew_gap": 0.0121,
        "fit_wall_seconds": 2.1,
        "skew_gap_T_s@1y": 0.0043,
        "skew_gap_T_l@3y": -0.0025,
        "wall_seconds": 620.0,
        "pricing_n_paths": 400_000,
        "pricing_seed": 2024,
    }
    df = pd.DataFrame([row])
    assert mc_columns_without_stderr(df) == []
    df["ssr_lsv"] = 1.42  # a stage-3 Monte Carlo number without its twin: still flagged
    assert mc_columns_without_stderr(df) == ["ssr_lsv"]
    df["ssr_lsv_stderr"] = 0.019
    assert mc_columns_without_stderr(df) == []


# --------------------------------------------------------------------------------------------
# measured cost: the run-record report and --cost-from (read-only, nothing calibrates)
# --------------------------------------------------------------------------------------------

_NEW_RUN = "2026-09-16T100000Z_1of2_1.json"
_OLD_RUN = "2026-09-15T160350Z_1of1_2.json"


def _entry(pid: str, mode: str, walls: dict[str, float], total: float, **kw: Any) -> dict[str, Any]:
    base = dict.fromkeys(precompute.WALL_STEPS, 0.0)
    base.update(walls)
    return {"point_id": pid, "label": pid[:8], "mode": mode, **base, "total": total, **kw}


def _entry_row(points: pd.DataFrame, run: str, point_id: str) -> pd.Series:
    """The one flattened entry of ``point_id`` in ``run``."""
    sub = points[(points["run"] == run) & (points["point_id"] == point_id)]
    assert len(sub) == 1, (run, point_id, len(sub))
    return sub.iloc[0]


def _synthetic_cost_store(root: Path) -> None:
    """Two run records — a current one (2 workers x 3 threads, a calibrated 1F point, a cache
    hit, the LV point and a risk-only refresh) and an old-format one (the M9 S1 layout of
    ``outputs/store``: no mode, steps, calibrated, cache_hit, threads or peak RSS) — plus the
    store rows the report reads the particle pass and the modes from."""
    store = ResultsStore(root)
    rows = {
        "a" * 64: ("one_factor", 100.0),
        "b" * 64: ("one_factor", 90.0),
        "c" * 64: ("two_factor", 120.0),
    }
    for pid, (mode, cal) in rows.items():
        store.write_point(PointResult(pid, {"mode": mode, "calibration_seconds": cal}, {}, {}))
    new = {
        "host": "vm-1",
        "shard": "1/2",
        "workers": 2,
        "threads_per_worker": 3,
        "cpu_count": 6,
        "wall_seconds": 7200.0,
        "points_selected": 6,
        "points_skipped": 1,
        "n_calibrated": 1,
        "n_cache_hits": 3,
        "failed": [{"point_id": "x", "label": "x", "error": "E", "traceback": ""}],
        "n_particles": 800_000,
        "pricing": {"n_paths": 400_000, "seed": 2024},
        "peak_rss_bytes": 2**30,
        "resume": True,
        "points_computed": [
            _entry(
                "a" * 64,
                "one_factor",
                {"calibration": 110.0, "diagnostics": 20.0, "pricing": 150.0, "analytics": 100.0},
                383.0,
                steps=["all"],
                calibrated=True,
                cache_hit=False,
                threads=3,
                peak_rss_bytes=3 * 2**30,
            ),
            _entry(
                "b" * 64,
                "one_factor",
                {"calibration": 4.0, "pricing": 170.0, "analytics": 110.0},
                285.0,
                steps=["all"],
                calibrated=False,
                cache_hit=True,
                threads=3,
                peak_rss_bytes=2 * 2**30,
            ),
            _entry(
                "lv:" + "d" * 64,
                "lv",
                {"calibration": 2.0, "pricing": 90.0, "analytics": 30.0},
                122.5,
                steps=["all"],
                calibrated=False,
                cache_hit=False,
                threads=3,
                peak_rss_bytes=2**30,
            ),
            _entry(
                "c" * 64,
                "two_factor",
                {"calibration": 3.0, "risk": 600.0},
                603.0,
                steps=["risk"],
                calibrated=False,
                cache_hit=True,
                threads=3,
                peak_rss_bytes=2**30,
            ),
        ],
    }
    old = {  # the S1 record layout: walls and totals only
        "host": "laptop",
        "shard": "1/1",
        "workers": 1,
        "wall_seconds": 1447.0,
        "n_particles": 800_000,
        "pricing": {"n_paths": 400_000, "seed": 2024},
        "points_in_shard": 2,
        "points_computed": [
            {"point_id": "b" * 64, "label": "hit", "calibration": 5.8, "diagnostics": 0.0,
             "pricing": 175.8, "analytics": 123.0, "risk": 0.0, "total": 304.7},
            {"point_id": "lv:" + "e" * 64, "label": "lv", "calibration": 3.3,
             "pricing": 118.6, "analytics": 43.2, "risk": 0.0, "total": 165.2},
        ],
    }  # fmt: skip
    store.runs_dir.mkdir(parents=True, exist_ok=True)
    (store.runs_dir / _NEW_RUN).write_text(json.dumps(new))
    (store.runs_dir / _OLD_RUN).write_text(json.dumps(old))


def test_cost_report_on_synthetic_records(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``volsto-precompute report`` on a synthetic record set: per run the wall clock, host,
    workers, threads, points computed / skipped / failed, calibrated vs cache hits and the hit
    rate, core-hours = wall x workers x threads and the peak RSS; per entry the particle pass
    split from the overhead (the store row's ``calibration_seconds``), core-seconds = seconds x
    threads; per (kind, mode, how) and step the median and p90.  The old-format record's missing
    fields are **listed and left absent** — no hit rate, no core-seconds, the calibration step
    unsplit, the mode taken from the store row or the id prefix — and ``--assume-threads``
    labels a stated thread count as assumed.  Nothing is computed."""
    root = tmp_path / "store"
    _synthetic_cost_store(root)
    rec = precompute.load_cost_records(root)
    runs = rec.runs.set_index("run")
    new, old = runs.loc[_NEW_RUN], runs.loc[_OLD_RUN]
    assert (new["host"], new["workers"], new["threads_per_worker"]) == ("vm-1", 2, 3.0)
    assert new["core_hours"] == pytest.approx(7200.0 * 2 * 3 / 3600.0)
    assert (new["points_computed"], new["points_skipped"], new["points_failed"]) == (4, 1, 1)
    assert (new["n_calibrated"], new["n_cache_hits"]) == (1, 3)
    assert new["cache_hit_rate"] == pytest.approx(3 / 4)
    assert new["peak_rss_gib"] == pytest.approx(3.0)
    assert old[["n_calibrated", "n_cache_hits"]].isna().all()  # absent, not zero
    assert np.isnan(old["cache_hit_rate"]) and np.isnan(old["core_hours"])
    assert np.isnan(old["threads_per_worker"]) and old["threads_source"] == "absent"
    assert old[["points_skipped", "points_failed"]].isna().all()
    assert np.isnan(old["peak_rss_gib"])
    absent = rec.absent[_OLD_RUN]
    for field in ("threads_per_worker", "n_calibrated", "n_cache_hits", "peak_rss_bytes"):
        assert field in absent
    for field in ("calibrated", "cache_hit", "mode", "steps", "threads"):
        assert f"points_computed[].{field}" in absent
    assert rec.absent[_NEW_RUN] == []

    pts = rec.points
    cal = _entry_row(pts, _NEW_RUN, "a" * 64)
    assert (cal["how"], cal["kind"], bool(cal["calibration_split"])) == ("calibrated", "full", True)
    assert cal["calibration"] == pytest.approx(100.0)  # the particle pass of the store row
    # the rest of the step (10 s) + the untimed remainder 383 - 380 = 3 s
    assert cal["overhead"] == pytest.approx(13.0)
    assert cal["core_total"] == pytest.approx(383.0 * 3)
    hit = _entry_row(pts, _NEW_RUN, "b" * 64)
    assert (hit["how"], hit["calibration"], hit["overhead"]) == ("cache hit", 0.0, 5.0)
    lv = _entry_row(pts, _NEW_RUN, "lv:" + "d" * 64)
    assert (lv["how"], lv["overhead"]) == ("no leverage", 2.5)
    refresh = _entry_row(pts, _NEW_RUN, "c" * 64)
    assert (refresh["kind"], refresh["mode"], refresh["risk"]) == (
        "refresh risk",
        "two_factor",
        600.0,
    )
    old_hit = _entry_row(pts, _OLD_RUN, "b" * 64)
    assert (old_hit["how"], old_hit["kind"], old_hit["mode"]) == (
        "unknown",
        "unknown",
        "one_factor",
    )
    assert old_hit["mode_source"] == "store" and not bool(old_hit["calibration_split"])
    assert old_hit["calibration"] == pytest.approx(5.8)  # kept whole: no flag to split on
    assert np.isnan(old_hit["core_total"]) and old_hit["threads_source"] == "absent"
    old_lv = _entry_row(pts, _OLD_RUN, "lv:" + "e" * 64)
    assert (old_lv["mode"], old_lv["mode_source"]) == ("lv", "id")
    # its diagnostics wall is absent: reported absent (NaN), never 0 — and so is the untimed
    # remainder the overhead would absorb it into
    assert np.isnan(old_lv["diagnostics"]) and np.isnan(old_lv["overhead"])
    assert old_lv["pricing"] == pytest.approx(118.6)
    assert "points_computed[].diagnostics" in absent

    summary = precompute.cost_summary(rec.points)
    one = summary[(summary["kind"] == "full") & (summary["mode"] == "one_factor")]
    pricing = one[(one["how"] == "calibrated") & (one["step"] == "pricing")].iloc[0]
    assert (pricing["n"], pricing["median_s"], pricing["median_core_s"]) == (1, 150.0, 450.0)
    unknown = summary[(summary["kind"] == "unknown") & (summary["step"] == "total")]
    assert set(unknown["mode"]) == {"one_factor", "lv"}
    old_lv_diag = summary[
        (summary["kind"] == "unknown")
        & (summary["mode"] == "lv")
        & (summary["step"] == "diagnostics")
    ].iloc[0]
    assert old_lv_diag["n"] == 0 and np.isnan(old_lv_diag["median_s"])
    assert unknown["median_core_s"].isna().all() and unknown["core_hours"].isna().all()
    # median and p90 over several entries: the linear-interpolation percentile
    many = pd.concat([rec.points] * 3, ignore_index=True)
    many.loc[many.index[: len(rec.points)], "pricing"] += 10.0
    s3 = precompute.cost_summary(many)
    row = s3[(s3["kind"] == "full") & (s3["how"] == "cache hit") & (s3["step"] == "pricing")]
    assert row["median_s"].iloc[0] == pytest.approx(170.0)
    assert row["p90_s"].iloc[0] == pytest.approx(np.percentile([180.0, 170.0, 170.0], 90))

    out_dir = tmp_path / "report"
    assert precompute.main(["report", "--store", str(root), "--out", str(out_dir)]) == 0
    text = capsys.readouterr().out
    assert "| vm-1 | 1/2 | 2 | 3 |" in text and "0.750" in text
    assert "## Fields absent from the records" in text and "points_computed[].calibrated" in text
    assert "over the 1 runs that record both (1 do not)" in text
    assert (
        "| unknown | lv | unknown | diagnostics | 0 | absent | absent | absent | absent |" in text
    )
    assert "Report wall clock" in text and "nothing recalibrated" in text
    for name in ("cost_report.md", "runs.csv", "points.csv", "summary.csv"):
        assert (out_dir / name).exists()
    assert len(pd.read_csv(out_dir / "points.csv")) == 6
    assert (
        precompute.main(["report", "--store", str(root), "--no-write", "--assume-threads", "12"])
        == 0
    )
    text = capsys.readouterr().out
    assert "12 (assumed)" in text and "written:" not in text
    assert not (root / precompute.COST_REPORT_DIRNAME).exists()
    assert precompute.main(["report", "--store", str(tmp_path / "missing")]) == 2
    assert "no results store" in capsys.readouterr().err
    assert not (tmp_path / "missing").exists()


def test_thread_factor_reads_the_measured_ratios() -> None:
    """The thread rescaling of ``--cost-from``: identity at equal counts, the measured ratio at
    a measured count, inverse pairs, interpolation in ``1/n`` between measured counts, the
    fitted Amdahl tail beyond them, serial steps unchanged."""
    table = precompute.MEASURED_THREAD_RATIO
    assert set(table) == set(precompute.THREAD_PARALLEL_FRACTION)
    for step, ratios in table.items():
        assert ratios[1] == 1.0
        assert precompute.thread_factor(step, 4, 4) == 1.0
        assert precompute.thread_factor(step, 1, 2) == pytest.approx(ratios[2])
        assert precompute.thread_factor(step, 12, 1) == pytest.approx(1.0 / ratios[12])
        for n_from, n_to in ((1, 12), (2, 3), (5, 48)):
            assert precompute.thread_factor(step, n_from, n_to) * precompute.thread_factor(
                step, n_to, n_from
            ) == pytest.approx(1.0)
        # 3 threads: between the 2- and 4-thread ratios, linear in 1/n
        w = (1 / 3 - 1 / 4) / (1 / 2 - 1 / 4)
        assert precompute.thread_factor(step, 1, 3) == pytest.approx(
            w * ratios[2] + (1 - w) * ratios[4]
        )
        # beyond 12: the fitted Amdahl tail, continuous at 12 and bounded by the serial part
        p = precompute.THREAD_PARALLEL_FRACTION[step]
        assert precompute.thread_factor(step, 1, 24) == pytest.approx(
            ratios[12] - p * (1 / 12 - 1 / 24)
        )
        assert precompute.thread_factor(step, 1, 10**6) <= ratios[12] + 1e-12
    # the second thread gains 8 % on the particle pass (measured), not Amdahl's 16 %
    assert precompute.thread_factor("calibration", 1, 2) == pytest.approx(0.9182)
    assert precompute.thread_factor("overhead", 1, 48) == 1.0
    with pytest.raises(ValueError):
        precompute.thread_factor("pricing", 0, 1)
    with pytest.raises(KeyError):
        precompute.thread_factor("bogus", 1, 2)
    with pytest.raises(KeyError):
        precompute.thread_factor("bogus", 2, 2)


def test_risk_cost_charges_states_misses_and_mode() -> None:
    """``CostModel.risk_s``: each bumped state pays its mode's budget-independent cost (the work
    outside the engine + the state's model build), an LSV point pays ``states − 1`` particle
    passes (its base state is its own leverage), and each pricing its mode's risk pricing — with
    the measured constants this reproduces the production light-tier risk steps measured on the
    laptop (1F 4283.0 s, LV 943.5 s; 17 states, 38 pricings, 1 thread)."""
    outside = precompute.RISK_STATE_OVERHEAD_S
    build, ratio = precompute.RISK_STATE_BUILD_S, precompute.RISK_PRICING_RATIO
    cost = precompute.CostModel(
        overhead_s=6.0,
        calibration_s=3045.5 / 16,  # the 16 risk calibrations of that run (cache manifest)
        calibration_source="test",
        diagnostics_s=36.7,
        pricing_s=265.403,
        analytics_s=196.6,
        risk_pricing_s=ratio["lsv"] * 265.403,
        pricing_source="test",
        risk_calibrations={"none": 0, "light": 17},
        risk_pricings={"none": 0, "light": 38},
        risk_state_s=outside + build["lsv"],
        risk_source="test",
        by_mode={
            "lv": {"risk_pricing": ratio["lv"] * 187.491, "risk_state": outside + build["lv"]}
        },
    )
    assert cost.risk_s("none", True) == 0.0
    assert cost.risk_s("light", True, "one_factor") == pytest.approx(
        17 * (outside + build["lsv"]) + 3045.5 + 38 * ratio["lsv"] * 265.403
    )
    assert cost.risk_s("light", True, "one_factor") == pytest.approx(4283.0, rel=2e-3)
    assert cost.risk_s("light", False, "lv") == pytest.approx(943.5, rel=2e-3)
    # an LSV point that does not calibrate (none does; the guard of the formula) pays no pass
    assert cost.risk_s("light", False, "one_factor") == pytest.approx(
        cost.risk_s("light", True, "one_factor") - 3045.5
    )
    # the default model charges the measured constants
    grid = load_grid(DEFAULT_GRID)
    default = precompute.default_cost_model(LeverageCache(Path("/nonexistent/cache")), grid)
    assert default.risk_state_s == pytest.approx(outside + build["lsv"])
    assert default.by_mode["lv"]["risk_state"] == pytest.approx(outside + build["lv"])
    assert default.risk_pricing_s == pytest.approx(ratio["lsv"] * default.pricing_s)
    # the LV ratio is to the LV pricing step; the fallback pricing is an LSV one
    assert default.by_mode["lv"]["risk_pricing"] == pytest.approx(
        ratio["lv"] * precompute.LV_TO_LSV_PRICING_RATIO * default.pricing_s
    )
    assert pytest.approx(187.5 / 265.4) == precompute.LV_TO_LSV_PRICING_RATIO
    assert not Path("/nonexistent/cache").exists()


def _synthetic_risk_store(root: Path) -> None:
    """Probe C as a synthetic store (the laptop measurement, 1 thread, 10⁵ particles, 5·10⁴
    paths, light tier on the 3-bucket ladder): a calibrated 1F point and the LV point with their
    run-record walls and their engine budgets."""
    store = ResultsStore(root)
    lsv_id, lv_id = "f" * 64, "lv:" + "g" * 64
    store.write_point(
        PointResult(
            lsv_id,
            {"mode": "one_factor", "calibration_seconds": 26.0},
            {},
            {"risk_budget": {"cache_misses": 16.0, "pricings": 38.0, "recalibrations": 17.0,
                             "wall_clock_s": 642.78}},
        )
    )  # fmt: skip
    store.write_point(
        PointResult(
            lv_id,
            {"mode": "lv", "calibration_seconds": float("nan")},
            {},
            {"risk_budget": {"cache_misses": 17.0, "pricings": 38.0, "recalibrations": 17.0,
                             "wall_clock_s": 123.07}},
        )
    )  # fmt: skip
    rec = {
        "host": "laptop",
        "shard": "1/1",
        "workers": 1,
        "threads_per_worker": 1,
        "wall_seconds": 1068.0,
        "n_particles": 100_000,
        "pricing": {"n_paths": 50_000, "seed": 2024},
        "points_computed": [
            _entry(
                lsv_id,
                "one_factor",
                {"calibration": 32.08, "diagnostics": 4.79, "pricing": 34.30,
                 "analytics": 25.09, "risk": 726.30},
                822.59,
                steps=["all"], calibrated=True, cache_hit=False, threads=1,
                peak_rss_bytes=2_299_854_848,
            ),
            _entry(
                lv_id,
                "lv",
                {"calibration": 2.83, "pricing": 23.20, "analytics": 10.97, "risk": 206.69},
                243.74,
                steps=["all"], calibrated=False, cache_hit=False, threads=1,
                peak_rss_bytes=2_299_854_848,
            ),
        ],
    }  # fmt: skip
    store.runs_dir.mkdir(parents=True, exist_ok=True)
    (store.runs_dir / _NEW_RUN).write_text(json.dumps(rec))


def test_cost_from_risk_budgets_by_mode(tmp_path: Path) -> None:
    """``--cost-from`` on a store with an LSV and an LV risk budget: the LSV risk pricing
    subtracts its cache misses x the particle pass, the LV one does **not** (its "misses" are
    Dupire rebuilds; subtracting them clipped the LV budget to 0 and halved the median), the
    per-state overhead is the step wall outside the engine, and both pricings rescale with the
    paths and the threads."""
    root = tmp_path / "risk_store"
    _synthetic_risk_store(root)
    rec = precompute.load_cost_records(root)
    budgets = precompute.risk_budget_costs(ResultsStore(root), rec.points, 26.0)
    build = precompute.RISK_STATE_BUILD_S
    lsv_pp = (642.78 - 16 * 26.0 - 17 * build["lsv"]) / 38
    lv_pp = (123.07 - 17 * build["lv"]) / 38
    assert budgets.lsv_per_pricing == [pytest.approx(lsv_pp)]
    assert budgets.lv_per_pricing == [pytest.approx(lv_pp)]
    assert sorted(budgets.state_s) == [
        pytest.approx((726.30 - 642.78) / 17),
        pytest.approx((206.69 - 123.07) / 17),
    ]
    assert budgets.notes == []

    grid = load_grid(DEFAULT_GRID)  # light tier, 3-bucket ladder: 17 states, 38 pricings
    assert precompute.risk_plan("light", grid.risk) == (17, 38)
    probe = dataclasses.replace(
        grid,
        particle=dataclasses.replace(grid.particle, n_particles=100_000),
        pricing=dataclasses.replace(grid.pricing, n_paths=50_000),
    )
    fallback = precompute.default_cost_model(LeverageCache(tmp_path / "cache"), probe)
    same, lines = precompute.cost_model_from_store(root, probe, fallback, target_threads=1)
    state = float(np.median(budgets.state_s))
    assert same.risk_state_s == pytest.approx(state + build["lsv"])
    assert same.by_mode["lv"]["risk_state"] == pytest.approx(state + build["lv"])
    assert same.risk_pricing_s == pytest.approx(lsv_pp)
    assert same.by_mode["lv"]["risk_pricing"] == pytest.approx(lv_pp)
    # the measured risk steps come back at the source budget and thread count
    assert same.risk_s("light", True, "one_factor") == pytest.approx(726.30, rel=1e-3)
    assert same.risk_s("light", False, "lv") == pytest.approx(206.69, rel=1e-3)
    assert any("risk pricing (LV)" in line and "stored LV risk budgets" in line for line in lines)
    assert any("risk state" in line and "RISK_STATE_BUILD_S" in line for line in lines)

    # production budget (x8 paths), 2 threads per worker: pricings rescale, state overhead not
    two, _ = precompute.cost_model_from_store(root, grid, fallback, target_threads=2)
    tf = precompute.thread_factor("risk", 1, 2)
    assert two.risk_pricing_s == pytest.approx(lsv_pp * 8 * tf)
    assert two.by_mode["lv"]["risk_pricing"] == pytest.approx(lv_pp * 8 * tf)
    assert two.risk_state_s == pytest.approx(state + build["lsv"])  # not rescaled

    # an LV-only source: the LSV risk pricing falls back to the measured ratio, named
    ResultsStore(root).point_dir("f" * 64).rename(tmp_path / "moved")
    lv_only, lines = precompute.cost_model_from_store(root, probe, fallback, target_threads=1)
    assert lv_only.by_mode["lv"]["risk_pricing"] == pytest.approx(lv_pp)
    assert lv_only.risk_pricing_s == pytest.approx(
        precompute.RISK_PRICING_RATIO["lsv"] * lv_only.pricing_s
    )
    assert any("no LSV risk budget in the source" in line for line in lines)

    # an LSV-only source (probe D's ``--only`` store): the LV risk pricing falls back to the LV
    # ratio applied to the LV pricing step, itself the LSV pricing x LV_TO_LSV_PRICING_RATIO
    lsv_root = tmp_path / "lsv_only"
    _synthetic_risk_store(lsv_root)
    lsv_store = ResultsStore(lsv_root)
    lsv_store.point_dir("lv:" + "g" * 64).rename(tmp_path / "moved_lv")
    run_file = lsv_store.runs_dir / _NEW_RUN
    run = json.loads(run_file.read_text())
    run["points_computed"] = [e for e in run["points_computed"] if e["mode"] != "lv"]
    run_file.write_text(json.dumps(run))
    lsv_only, lines = precompute.cost_model_from_store(lsv_root, probe, fallback, target_threads=1)
    assert "pricing" not in lsv_only.by_mode.get("lv", {})
    assert lsv_only.risk_pricing_s == pytest.approx(lsv_pp)
    assert lsv_only.by_mode["lv"]["risk_pricing"] == pytest.approx(
        precompute.RISK_PRICING_RATIO["lv"]
        * precompute.LV_TO_LSV_PRICING_RATIO
        * lsv_only.pricing_s
    )
    assert any("no LV risk budget in the source" in line for line in lines)


def test_cost_report_on_the_toy_store(toy_build: ToyBuild, tmp_path: Path) -> None:
    """The report on the fixture's store (read only; the output goes to a temporary directory):
    the two shard runs, 3 calibrated points and no cache hit (hit rate 0), the LV point without a
    leverage, the thread count and peak RSS recorded by the precompute itself, the particle pass
    equal to the store rows' ``calibration_seconds`` — and the fixture directory unchanged."""
    toy = toy_build.require()
    before = _file_snapshot(toy.base)
    rec = precompute.load_cost_records(toy.store_root)
    assert len(rec.runs) == len(TOY_SHARDS)
    assert int(rec.runs["points_computed"].sum()) == 4
    assert int(rec.runs["n_calibrated"].sum()) == 3 and int(rec.runs["n_cache_hits"].sum()) == 0
    assert (rec.runs["threads_source"] == "record").all()
    assert (rec.runs["peak_rss_gib"] > 0).all()
    assert all(fields == [] for fields in rec.absent.values()), rec.absent
    pts = rec.points
    assert sorted(pts["how"]) == ["calibrated"] * 3 + ["no leverage"]
    assert set(pts["mode"]) == {"lv", "one_factor", "two_factor"}
    assert (pts["mode_source"] == "record").all() and (pts["kind"] == "full").all()
    rows = StoreReader(toy.store_root).points().set_index("point_id")
    for _, e in pts[pts["how"] == "calibrated"].iterrows():
        assert e["calibration"] == pytest.approx(rows.loc[e["point_id"], "calibration_seconds"])
        assert e["overhead"] >= 0 and e["total"] == pytest.approx(
            sum(e[s] for s in precompute.COST_STEPS[:-1]), rel=1e-9
        )
    assert np.isfinite(pts["core_total"]).all()
    out = tmp_path / "toy_report"
    assert precompute.report_main(["--store", str(toy.store_root), "--out", str(out)]) == 0
    assert (out / "cost_report.md").read_text().count("| full | one_factor | calibrated |") >= 5
    assert _file_snapshot(toy.base) == before
    print("\n" + (out / "cost_report.md").read_text())


def test_cost_from_builds_the_cost_model_from_the_toy_store(
    toy_build: ToyBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--dry-run --cost-from`` on the toy store: the calibration cost is the median particle
    pass of its calibrated points (rescaled linearly in particles), pricing / analytics the
    median of its LSV points (linearly in paths) with per-mode values, the risk pricing the
    documented ratio (the toy ran no risk), the thread rescaling the Amdahl factor, and the
    source of every number is printed.  A source without a thread count is not rescaled and says
    so.  Nothing is computed or written."""
    toy = toy_build.require()
    before = _file_snapshot(toy.base)
    grid = load_grid(TOY_GRID)
    fallback = precompute.default_cost_model(LeverageCache(tmp_path / "cache"), grid)
    rec = precompute.load_cost_records(toy.store_root)
    threads = int(rec.points["threads"].iloc[0])
    cal = rec.points[rec.points["how"] == "calibrated"]
    lsv = rec.points[rec.points["mode"] != "lv"]
    cost, lines = precompute.cost_model_from_store(
        toy.store_root, grid, fallback, target_threads=threads
    )
    assert cost.calibration_s == pytest.approx(float(np.median(cal["calibration"])))
    assert cost.diagnostics_s == pytest.approx(float(np.median(cal["diagnostics"])))
    assert cost.pricing_s == pytest.approx(float(np.median(lsv["pricing"])))
    assert cost.analytics_s == pytest.approx(float(np.median(lsv["analytics"])))
    assert cost.overhead_s == pytest.approx(float(np.median(rec.points["overhead"])))
    assert cost.risk_pricing_s == pytest.approx(
        precompute.RISK_PRICING_RATIO["lsv"] * cost.pricing_s
    )
    assert cost.risk_state_s == pytest.approx(
        precompute.RISK_STATE_OVERHEAD_S + precompute.RISK_STATE_BUILD_S["lsv"]
    )
    lv = rec.points[rec.points["mode"] == "lv"].iloc[0]
    assert cost.by_mode["lv"]["pricing"] == pytest.approx(lv["pricing"])
    assert set(cost.by_mode) == {"lv", "one_factor", "two_factor"}
    assert any("no LSV risk budget in the source" in line for line in lines)
    assert cost.by_mode["lv"]["risk_pricing"] == pytest.approx(
        precompute.RISK_PRICING_RATIO["lv"] * cost.by_mode["lv"]["pricing"]
    )
    # the LV point is charged its own measured pricing, an LSV point the LSV median
    assert cost.point_s("none", calibrates=False, miss=False, mode="lv") == pytest.approx(
        cost.by_mode["lv"]["overhead"] + lv["pricing"] + lv["analytics"]
    )

    # rescaled: twice the particles and paths, a different thread count
    bigger = dataclasses.replace(
        grid,
        particle=dataclasses.replace(grid.particle, n_particles=2 * grid.particle.n_particles),
        pricing=dataclasses.replace(grid.pricing, n_paths=2 * grid.pricing.n_paths),
    )
    target = threads + 3
    big, _ = precompute.cost_model_from_store(
        toy.store_root, bigger, fallback, target_threads=target
    )
    assert big.calibration_s == pytest.approx(
        2 * cost.calibration_s * precompute.thread_factor("calibration", threads, target)
    )
    assert big.pricing_s == pytest.approx(
        2 * cost.pricing_s * precompute.thread_factor("pricing", threads, target)
    )
    assert big.overhead_s == pytest.approx(cost.overhead_s)  # serial, budget-independent

    base = [
        "--grid",
        str(TOY_GRID),
        "--store",
        str(tmp_path / "store"),
        "--cache",
        str(tmp_path / "cache"),
        "--dry-run",
    ]
    assert precompute.main([*base, "--cost-from", str(toy.store_root), "--workers", "2"]) == 0
    out = capsys.readouterr().out
    assert "cost model from the measured store" in out
    assert "--cost-from: median particle pass" in out
    assert "memory: measured peak RSS" in out and "x 2 workers" in out
    assert "parallel (shard 1/1, 4 points to compute): 2 worker(s)" in out
    assert "dry run: nothing computed" in out
    # on a shard, the parallel line is the shard's, not the whole grid's
    assert precompute.main([*base, "--cost-from", str(toy.store_root), "--shard", "2/2"]) == 0
    out = capsys.readouterr().out
    line = next(ln for ln in out.splitlines() if ln.startswith("parallel ("))
    shard_line = next(ln for ln in out.splitlines() if ln.startswith("this shard (2/2)"))
    assert "shard 2/2, 2 points to compute" in line
    shard_h = float(shard_line.split("projected ")[1].split(" process-hours")[0])
    assert f"({shard_h:.2f} process-hours" in line
    assert precompute.main([*base, "--cost-from", str(tmp_path / "nowhere")]) == 2
    assert "no run records" in capsys.readouterr().err

    # a source whose records carry no thread count: no rescaling, said so
    old = tmp_path / "old_store"
    _synthetic_cost_store(old)
    (old / "results" / "runs" / _NEW_RUN).unlink()
    _, lines = precompute.cost_model_from_store(old, grid, fallback, target_threads=4)
    assert any("thread count ABSENT" in line for line in lines)
    assert any("calibration" in line and "kept" in line for line in lines)  # nothing to split
    _, lines = precompute.cost_model_from_store(
        old, grid, fallback, target_threads=4, source_threads=12
    )
    assert any("assumed by --cost-from-threads" in line for line in lines)
    assert not (tmp_path / "store").exists()
    assert _file_snapshot(toy.base) == before
