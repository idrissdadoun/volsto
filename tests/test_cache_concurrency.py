"""Concurrent and interrupted writers of the leverage cache (``volsto/calibration/cache.py``).

A ``volsto-precompute --workers N`` pool, or N ``--shard i/n`` processes on one cache, append
manifest rows concurrently.  The manifest is a single parquet file rewritten on every append, so
without a lock two writers lose each other's rows, and without an atomic rename a reader can see
a torn file.  This test runs many writer processes (synthetic rows, no calibration, no Monte
Carlo) alongside reader processes and checks that

* every row every writer appended is in the final manifest, exactly once;
* no read by any reader or writer failed, and no reader ever saw the row count go down;
* re-appending an existing key still replaces its row (the manifest's one-row-per-key rule);
* the on-disk format is unchanged: a plain parquet file at ``<root>/manifest.parquet``, with no
  temporary file left behind.

A spot instance can kill a process in the middle of any cache write, and ``--resume`` trusts
``LeverageCache.has``.  The second half checks the invariant "every cache file is published by
``atomic_write``, so a kill leaves the old file or the new one, never a torn one that ``has``
reports":

* a walking test records every ``atomic_write`` and checks that it accounts for every file a
  store / report / manifest write leaves in the cache, each with the ordinary umask mode;
* a child process killed (SIGKILL) while its leverage is half written leaves no entry (fresh key)
  or the old entry intact (existing key), plus a leftover temporary that nothing reads;
* a save that raises half-way leaves no temporary and the same entry state;
* a torn ``leverage.npz`` written before writes were atomic is a miss, not a hit.

Nothing is calibrated: the leverages are constant synthetic ones.  Wall clock is printed, never
asserted.
"""

from __future__ import annotations

import dataclasses
import logging
import multiprocessing as mp
import os
import signal
import stat
import time
import traceback
import zipfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest

import volsto.calibration.cache as cache_mod
from volsto.calibration.cache import (
    DIAGNOSTICS_NAME,
    LEVERAGE_NAME,
    MANIFEST_LOCK_NAME,
    MANIFEST_NAME,
    SPEC_NAME,
    LeverageCache,
)
from volsto.calibration.diagnostics import CalibrationReport
from volsto.config import CalibrationSpec, load_yaml
from volsto.market.curves import ForwardCurve
from volsto.models.leverage import LeverageFunction
from volsto.viewers import precompute

ROOT = Path(__file__).resolve().parents[1]
SPEC_1F = ROOT / "configs" / "studies" / "lsv_reference_1f.yaml"

#: Writer processes (a 24-48 worker VM pool is the case being guarded; 12 already loses rows
#: reliably without the lock on a laptop) and rows each writer appends.
N_WRITERS = 12
ROWS_PER_WRITER = 30
#: Reader processes polling the manifest while the writers run.
N_READERS = 3
#: Upper bound on the whole exercise before the test fails rather than hangs.
JOIN_TIMEOUT_S = 300.0


def _synthetic(key_seed: int, nu: float) -> tuple[CalibrationSpec, LeverageFunction]:
    """Duck-typed spec and leverage carrying exactly the fields the manifest row reads."""
    spec = SimpleNamespace(
        model=SimpleNamespace(
            nu=nu, theta=0.0, k1=1.0, k2=0.0, rho12=0.0, rho_SX1=-0.7, rho_SX2=0.0
        ),
        particle=SimpleNamespace(horizon=1.0, n_particles=1000, seed=key_seed),
        surface=SimpleNamespace(rho=-0.5, eta=1.0, gamma=0.5),
    )
    leverage = SimpleNamespace(
        metadata={
            "created_utc": "2026-09-16T00:00:00+00:00",
            "git_commit": "test",
            "wall_time": 1.0,
        }
    )
    return cast(CalibrationSpec, spec), cast(LeverageFunction, leverage)


#: At most this many failure messages are sent back per process (the count is always sent).
MAX_MESSAGES = 5


def _writer(root: str, writer: int, start: Any, results: Any) -> None:
    cache = LeverageCache(root)
    start.wait()
    failures: list[str] = []
    for i in range(ROWS_PER_WRITER):
        spec, lev = _synthetic(writer * 1000 + i, nu=float(writer))
        try:
            cache._append_manifest(f"w{writer:02d}-{i:03d}", spec, lev, None)
        except Exception:  # every failure is reported by the parent
            failures.append(f"writer {writer} row {i}: {traceback.format_exc(limit=1)}")
    # exactly one message per process, sent last: the parent drains the queue before joining
    results.put(("writer", len(failures), failures[:MAX_MESSAGES], 0))


def _reader(root: str, reader: int, start: Any, stop: Any, results: Any) -> None:
    cache = LeverageCache(root)
    start.wait()
    failures: list[str] = []
    n_failed = 0
    last = 0
    reads = 0
    while not stop.is_set():
        try:
            n = len(cache.manifest())
        except Exception:  # a torn read is exactly what the test looks for
            n_failed += 1
            failures.append(f"reader {reader}: {traceback.format_exc(limit=1)}")
            continue
        reads += 1
        if n < last:
            n_failed += 1
            failures.append(f"reader {reader}: row count went down {last} -> {n}")
        last = n
    results.put(("reader", n_failed, failures[:MAX_MESSAGES], reads))


def test_concurrent_manifest_writers_lose_no_row(tmp_path: Path) -> None:
    root = tmp_path / "cache"  # not created: the first writer creates it
    ctx = mp.get_context("spawn")
    start = ctx.Event()
    stop = ctx.Event()
    results = ctx.Queue()
    writers = [
        ctx.Process(target=_writer, args=(str(root), w, start, results)) for w in range(N_WRITERS)
    ]
    readers = [
        ctx.Process(target=_reader, args=(str(root), r, start, stop, results))
        for r in range(N_READERS)
    ]
    procs = writers + readers
    for p in procs:
        p.start()
    t0 = time.perf_counter()
    start.set()
    got: list[tuple[str, int, list[str], int]] = []
    try:
        while sum(1 for g in got if g[0] == "writer") < N_WRITERS:
            got.append(results.get(timeout=JOIN_TIMEOUT_S))
        stop.set()
        while len(got) < len(procs):
            got.append(results.get(timeout=JOIN_TIMEOUT_S))
    finally:
        stop.set()
        for p in procs:
            p.join(JOIN_TIMEOUT_S)
            if p.is_alive():
                p.kill()
    wall = time.perf_counter() - t0
    assert all(p.exitcode == 0 for p in procs), [p.exitcode for p in procs]

    n_failed = sum(g[1] for g in got)
    messages = [m for g in got for m in g[2]]
    reads = sum(g[3] for g in got)
    print(
        f"{N_WRITERS} writers x {ROWS_PER_WRITER} rows, {N_READERS} readers ({reads} reads): "
        f"{n_failed} failed operation(s), wall {wall:.2f} s"
    )
    assert n_failed == 0, "\n".join(messages)

    manifest = pd.read_parquet(root / MANIFEST_NAME)
    expected = {f"w{w:02d}-{i:03d}" for w in range(N_WRITERS) for i in range(ROWS_PER_WRITER)}
    assert len(manifest) == len(expected), f"{len(expected) - len(manifest)} row(s) lost"
    assert set(manifest["key"]) == expected
    assert manifest["key"].is_unique
    # each row carries its own writer's values (no row was mixed up with another)
    for key, nu in zip(manifest["key"], manifest["nu"], strict=True):
        assert float(nu) == float(int(str(key)[1:3]))
    # nothing but the manifest and its (empty) lock file is left in the root
    leftovers = sorted(p.name for p in root.iterdir() if p.name != MANIFEST_NAME)
    assert leftovers == [MANIFEST_LOCK_NAME], leftovers


def test_reappending_a_key_replaces_its_row(tmp_path: Path) -> None:
    cache = LeverageCache(tmp_path / "cache")
    for nu in (1.0, 2.0):
        spec, lev = _synthetic(1, nu=nu)
        cache._append_manifest("same", spec, lev, None)
    spec, lev = _synthetic(2, nu=3.0)
    cache._append_manifest("other", spec, lev, None)
    m = cache.manifest()
    assert list(m["key"]) == ["same", "other"]
    assert list(m["nu"]) == [2.0, 3.0]
    assert list(m["max_abs_error_vp"]) == [None, None]


# --------------------------------------------------------------------------------------------
# interrupted writes: a kill leaves the old entry or no entry, never a torn one
# --------------------------------------------------------------------------------------------

#: Particle count of the synthetic specs (only a cache-key ingredient here: nothing calibrates).
SYNTHETIC_PARTICLES = 1_000
#: Seconds the parent waits for the child to reach the middle of its leverage write.
READY_TIMEOUT_S = 120.0


def _spec(seed: int = 7) -> CalibrationSpec:
    spec = load_yaml(SPEC_1F, CalibrationSpec)
    return dataclasses.replace(
        spec,
        particle=dataclasses.replace(spec.particle, n_particles=SYNTHETIC_PARTICLES, seed=seed),
    )


def _leverage(spec: CalibrationSpec, value: float) -> LeverageFunction:
    """A constant leverage on a grid large enough that its archive takes several blocks."""
    fc = ForwardCurve.from_config(spec.market)
    times = np.linspace(0.0, 3.0, 61)
    k = np.linspace(-2.5, 2.5, 401)
    rng = np.random.default_rng(int(value * 1000))
    values = value + 1e-3 * rng.random((times.size, k.size))  # incompressible enough
    return LeverageFunction(times, k, values, fc, {"seed": 0, "wall_time": 1.0})


def _report() -> CalibrationReport:
    vanillas = pd.DataFrame(
        {
            "T": [0.5, 1.0],
            "k": [0.0, 0.1],
            "error_vp": [0.05, -0.08],
            "stderr_vp": [0.02, 0.03],
            "target_vol": [0.2, 0.21],
        }
    )
    return CalibrationReport(vanillas, pd.DataFrame({"T": [1.0]}), 1000, 1, 0.1)


def _umask() -> int:
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


def _files(root: Path) -> set[Path]:
    return {p for p in root.rglob("*") if p.is_file() and p.name != MANIFEST_LOCK_NAME}


def test_every_cache_write_is_atomic_with_the_umask_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Walk every write path — ``store`` with a report, ``store`` over an existing key without
    one, ``write_report``, the precompute's ``store_report`` and ``merge_manifest`` — and check
    that ``atomic_write`` published every file the cache holds afterwards, that no temporary is
    left, that a stale report does not survive a new leverage, and that every file has the mode a
    plain ``open`` gives (``0666 & ~umask``; ``mkstemp`` would give 0600)."""
    published: set[Path] = set()
    real = cache_mod.atomic_write

    def _spy(dest: str | Path, write: Any) -> Path:
        out = real(dest, write)
        published.add(Path(dest))
        return out

    monkeypatch.setattr(cache_mod, "atomic_write", _spy)
    root = tmp_path / "cache"
    cache = LeverageCache(root)
    spec, other = _spec(1), _spec(2)
    d = cache.store(spec, _leverage(spec, 1.0), _report())
    assert {p.name for p in d.iterdir()} == {LEVERAGE_NAME, SPEC_NAME, DIAGNOSTICS_NAME}
    cache.store(other, _leverage(other, 1.1))
    precompute.store_report(cache, other, cache.load(other), _report())
    cache.store(spec, _leverage(spec, 1.2))  # a new leverage: the old report is dropped
    assert not (d / DIAGNOSTICS_NAME).exists()
    cache.write_report(spec, _report())
    foreign = cache.manifest().assign(key=lambda m: m["key"] + "-vm")
    assert cache.merge_manifest(foreign) == (2, 2)
    assert cache.merge_manifest(foreign) == (4, 0)

    files = _files(root)
    assert files == published, sorted(str(p.relative_to(root)) for p in files ^ published)
    assert cache.temporaries() == []
    expected_mode = 0o666 & ~_umask()
    modes = {str(p.relative_to(root)): stat.S_IMODE(p.stat().st_mode) for p in files}
    assert set(modes.values()) == {expected_mode}, modes
    assert np.allclose(cache.load(spec).values, _leverage(spec, 1.2).values)
    assert cache.has(spec) and cache.has(other) and cache.unlisted_keys() == []
    print(f"{len(files)} files, all published atomically, mode {expected_mode:o}")


def _killed_writer(root: str, value: float, ready: Any) -> None:
    """Store a leverage whose save stops half-way (the archive truncated to half its size, as a
    kill during ``np.savez_compressed`` leaves it), then wait to be killed."""
    real_save = LeverageFunction.save

    def _half_save(self: LeverageFunction, path: str | Path) -> Path:
        out = real_save(self, path)
        os.truncate(out, out.stat().st_size // 2)
        ready.set()
        time.sleep(3600.0)
        return out

    LeverageFunction.save = _half_save  # type: ignore[method-assign]
    spec = _spec()
    LeverageCache(root).store(spec, _leverage(spec, value))


@pytest.mark.parametrize("existing", [False, True], ids=["fresh-key", "existing-key"])
def test_a_kill_mid_save_leaves_the_old_entry_or_none(tmp_path: Path, existing: bool) -> None:
    root = tmp_path / "cache"
    cache = LeverageCache(root)
    spec = _spec()
    if existing:
        cache.store(spec, _leverage(spec, 1.0))
    ctx = mp.get_context("spawn")
    ready = ctx.Event()
    child = ctx.Process(target=_killed_writer, args=(str(root), 2.0, ready))
    t0 = time.perf_counter()
    child.start()
    try:
        assert ready.wait(READY_TIMEOUT_S), "the child never reached its leverage write"
        assert child.pid is not None
        os.kill(child.pid, signal.SIGKILL)
    finally:
        child.join(READY_TIMEOUT_S)
        if child.is_alive():
            child.kill()
    assert child.exitcode == -signal.SIGKILL
    wall = time.perf_counter() - t0

    leftovers = cache.temporaries()
    assert len(leftovers) == 1 and leftovers[0].parent == cache.entry_dir(spec), leftovers
    assert not zipfile.is_zipfile(leftovers[0])  # the torn bytes exist, under a temporary name
    assert cache.has(spec) is existing
    assert (cache.entry_dir(spec) / LEVERAGE_NAME).exists() is existing
    manifest = cache.manifest()
    assert (manifest.empty and not existing) or list(manifest["key"]) == [cache.key(spec)]
    assert cache.unlisted_keys() == []
    if existing:  # the old leverage, whole
        assert np.allclose(cache.load(spec).values, _leverage(spec, 1.0).values)

    # --resume: the leftover does not count, and the next store publishes over it cleanly
    cache.store(spec, _leverage(spec, 2.0))
    assert cache.has(spec)
    assert np.allclose(cache.load(spec).values, _leverage(spec, 2.0).values)
    assert cache.temporaries() == leftovers  # untouched: deleting it is the operator's call
    print(f"killed mid-save ({'existing' if existing else 'fresh'} key), wall {wall:.2f} s")


@pytest.mark.parametrize("existing", [False, True], ids=["fresh-key", "existing-key"])
def test_a_save_raising_mid_write_leaves_no_temporary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    cache = LeverageCache(tmp_path / "cache")
    spec = _spec()
    if existing:
        cache.store(spec, _leverage(spec, 1.0))
    real_save = LeverageFunction.save

    def _failing_save(self: LeverageFunction, path: str | Path) -> Path:
        out = real_save(self, path)
        os.truncate(out, out.stat().st_size // 2)
        raise OSError("disk full (simulated)")

    monkeypatch.setattr(LeverageFunction, "save", _failing_save)
    with pytest.raises(OSError, match="simulated"):
        cache.store(spec, _leverage(spec, 2.0))
    monkeypatch.setattr(LeverageFunction, "save", real_save)
    assert cache.temporaries() == []
    assert cache.has(spec) is existing
    assert len(cache.manifest()) == int(existing)
    if existing:
        assert np.allclose(cache.load(spec).values, _leverage(spec, 1.0).values)


def test_a_torn_leverage_from_before_atomic_writes_is_a_miss(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """An entry whose ``leverage.npz`` was cut short in place (the pre-fix save) is not a hit for
    ``has`` / ``has_key`` (the ``--resume`` test) and is not listed by ``unlisted_keys``."""
    cache = LeverageCache(tmp_path / "cache")
    spec = _spec()
    cache.store(spec, _leverage(spec, 1.0))
    path = cache.entry_dir(spec) / LEVERAGE_NAME
    os.truncate(path, path.stat().st_size // 2)
    with caplog.at_level(logging.WARNING, logger=cache_mod.__name__):
        assert not cache.has(spec)
        assert not cache.has_key(cache.key(spec))
    assert "not a complete archive" in caplog.text
    assert cache.unlisted_keys() == []
    path.write_bytes(b"")  # an empty file (killed right after open) is a miss too
    assert not cache.has(spec)
    cache.store(spec, _leverage(spec, 1.5))  # the recalibration replaces it
    assert cache.has(spec)


def test_unlisted_keys_names_an_entry_whose_manifest_row_was_lost(tmp_path: Path) -> None:
    """A kill between the leverage (the commit point) and the manifest row leaves a complete
    entry without a row: a hit, and listed by ``unlisted_keys`` for the arrival check."""
    cache = LeverageCache(tmp_path / "cache")
    kept, lost = _spec(1), _spec(2)
    cache.store(kept, _leverage(kept, 1.0))
    cache.store(lost, _leverage(lost, 1.1))
    m = cache.manifest()
    m[m["key"] == cache.key(kept)].to_parquet(cache.root / MANIFEST_NAME, index=False)
    assert cache.has(lost)
    assert cache.unlisted_keys() == [cache.key(lost)]
