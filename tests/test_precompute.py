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
row.  Wall clock is reported, never asserted.
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


def _cache_keys(root: Path) -> set[str]:
    m = LeverageCache(root).manifest()
    return set() if m.empty else {str(k) for k in m["key"]}


def test_dry_run_projects_without_computing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--dry-run`` on the default grid against the repository cache: prints the resolved store
    and cache roots, the counts (105 one-factor combinations), the per-point cost lines
    (overhead, calibration from the cache manifest's measured wall times, diagnostics repricing)
    and one row per tier — the light tier twice, at the configured 20-bucket fwd-var ladder and
    at the 3-bucket alternative, so the owner can choose.  It writes nothing into the store and
    **calibrates nothing**: the repository cache manifest is byte-identical afterwards."""
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
    risk = load_grid(DEFAULT_GRID).risk
    assert (risk.tier, risk.fwd_var_buckets, risk.light_pillars) == ("light", 20, (0.25, 1.0, 3.0))
    assert precompute.risk_plan("none", risk) == (0, 0)
    assert precompute.risk_plan("light", risk) == (34, 72)
    assert precompute.risk_plan("light", risk, 3) == (17, 38)
    assert precompute.risk_plan("full", risk) == (
        precompute.RISK_FULL_CALIBRATIONS,
        precompute.RISK_FULL_PRICINGS,
    )
    assert precompute.alternative_buckets(risk) == 3
    # the coarse ladder is the tent over the light pillars; the M5 one is the library's
    assert precompute.light_buckets(risk) == default_buckets()
    coarse = dataclasses.replace(risk, fwd_var_buckets=3)
    assert precompute.light_buckets(coarse) == ((0.0, 0.25), (0.25, 1.0), (1.0, 3.0))
    assert precompute.risk_plan("light", coarse) == (17, 38)
    assert precompute.alternative_buckets(coarse) == 20
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
