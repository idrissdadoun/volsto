"""Shared fixtures: reference surface, flat curves, fast simulation settings — and the toy
precompute of the viewers layer (M9).

**The toy calibration runs exactly once per pytest run, in the tests/conftest.py session
fixture** :func:`toy_build`: ``volsto-precompute`` on ``configs/grids/toy.yaml`` (3 leverage
calibrations at 2·10⁴ particles over a 1y horizon, 4000 pricing paths, no risk, plus the LV
point) as shards 1/2 then 2/2 into a **temporary** directory — never the repository cache.  It
is the one sanctioned calibrating site of the test suite (owner's M9 rule): every other test,
every viewer and the read API only read a store and a cache.  ``tests/test_precompute.py``
(the calibrating-pass assertions and the live, non-calibrating ``--resume`` / relocation steps
on a private copy) and ``tests/test_viewers_app.py`` (every page rendered headless, the API's
stderr schema, relocatability, the two CLIs) consume it; the fixture directory stays intact.
Under ``pytest -n auto`` the workers share one build (controller base temp + an atomically
created lock directory naming the building process, ``tests/_locks.py`` — the one lock
implementation of the session fixtures, shared with ``tests/_backtest_build.py`` — +
``done.json`` marker, stdlib only).  Wall
clocks are recorded and printed, never asserted and never used to skip.  **A build that raised,
exited non-zero or died fails every test that requires it** (:meth:`ToyBuild.require`); the
toy build needs nothing optional (the placeholder surface is in the repository), so nothing in
it skips.

**The toy marking build** (M10 Part 2, for the S5 marking study): :func:`toy_marking_build` runs
``volsto-precompute`` on ``configs/grids/toy_marking.yaml`` (the P1 marking fits at
``ssr_target {1.0, 1.5} × skew_eps {0.05, 0.10}`` on the placeholder surface, at most 4 leverage
calibrations at 2·10⁴ particles over a 1y horizon — an infeasible fit is stored without one — the
LV point, M4 products at 4000 paths, no risk) as the single shard ``1/1`` into its own temporary
directory, once per pytest run, with the same sharing, recording and failure rules and the same
interface as :func:`toy_build` (``.require()``, ``.store_root``, ``.cache_root``,
``.outputs_root``, ``.grid_path``; the synthetic M7 / M8b outputs beside it).  It is the second
sanctioned calibrating site of the suite, and like the first it calibrates only when a test asks
for it (``tests/test_catalogue_s1_s4.py::test_toy_marking_build`` and the S5 tests).
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import _locks
import numpy as np
import pytest

from volsto.config import SimConfig
from volsto.market import DiscountCurve, ForwardCurve, LocalVolSurface, SSVISurface
from volsto.market.loaders import load_ssvi_surface

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_SURFACE = ROOT / "configs" / "surfaces" / "reference_ssvi.yaml"


@pytest.fixture(scope="session")
def forward_curve() -> ForwardCurve:
    return ForwardCurve.flat(100.0, 0.02, 0.01)


@pytest.fixture(scope="session")
def discount(forward_curve: ForwardCurve) -> DiscountCurve:
    return forward_curve.rate_curve


@pytest.fixture(scope="session")
def ssvi() -> SSVISurface:
    return load_ssvi_surface(REFERENCE_SURFACE)


@pytest.fixture(scope="session")
def local_vol(ssvi: SSVISurface) -> LocalVolSurface:
    return LocalVolSurface.from_implied(ssvi)


@pytest.fixture
def fast_sim() -> SimConfig:
    return SimConfig(n_paths=40_000, dt_max=1.0 / 100.0, chunk_size=20_000, antithetic=True, seed=7)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(123)


# --------------------------------------------------------------------------------------------
# the toy precompute — the ONE calibrating site of the test suite (M9, SPEC §9.2)
# --------------------------------------------------------------------------------------------

TOY_GRID = ROOT / "configs" / "grids" / "toy.yaml"
#: The shards the fixture runs, in order (``volsto-precompute --shard``).
TOY_SHARDS: tuple[str, ...] = ("1/2", "2/2")
#: The toy marking grid (M10 Part 2) and its single shard.
TOY_MARKING_GRID = ROOT / "configs" / "grids" / "toy_marking.yaml"
TOY_MARKING_SHARDS: tuple[str, ...] = ("1/1",)


@dataclass(frozen=True)
class ToyBuild:
    """What the session fixture :func:`toy_build` produced: the roots (store, cache and the
    synthetic M7 / M8b outputs side by side under ``base``) and what the calibrating pass
    recorded per shard — return code, wall clock, captured stdout, captured ``volsto`` log
    messages (``"<logger>: <message>"``) and the leverage-cache manifest after the shard.
    ``error`` is set when the build raised, was interrupted or its builder died; tests call
    :meth:`require`, which **fails** them on any such problem or a non-zero exit (never a
    skip: the build has no optional input)."""

    root: Path
    base: Path
    grid_path: Path
    return_codes: tuple[int, ...]
    wall_s: dict[str, float]
    stdout: dict[str, str]
    log_messages: dict[str, list[str]]
    manifests: dict[str, list[dict[str, Any]]]
    built_by: str
    error: str = ""
    #: the ``--shard`` values the build ran, in order
    shards: tuple[str, ...] = TOY_SHARDS

    @property
    def store_root(self) -> Path:
        return self.base / "store"

    @property
    def cache_root(self) -> Path:
        return self.base / "cache"

    @property
    def outputs_root(self) -> Path:
        return self.base / "outputs"

    @property
    def total_wall_s(self) -> float:
        return float(sum(self.wall_s.values())) if self.wall_s else float("nan")

    @property
    def failure_reason(self) -> str:
        """Why the build is unusable (``""`` when it is usable): an exception, an interrupted
        or dead builder, or a non-zero exit of any shard."""
        if self.error:
            return f"toy build failed: {self.error}"
        if tuple(self.return_codes) != (0,) * len(self.shards):
            return f"toy precompute returned {list(self.return_codes)}"
        return ""

    def require(self) -> ToyBuild:
        """**Fail** the calling test with the reason when the build is unusable."""
        if self.failure_reason:
            pytest.fail(self.failure_reason, pytrace=False)
        return self

    def calibrating_messages(
        self, shard: str | None = None, logger: str = "volsto.viewers.precompute"
    ) -> list[str]:
        """The ``"calibrating"`` log records of ``logger`` (default: the precompute's own
        "leverage cache miss … calibrating at N particles" line, one per miss; the cache logs a
        second one under ``volsto.calibration.cache``) for ``shard`` or all shards."""
        shards = self.shards if shard is None else (shard,)
        return [
            m
            for s in shards
            for m in self.log_messages.get(s, [])
            if m.startswith(f"{logger}: ") and "calibrating" in m
        ]


class _ListHandler(logging.Handler):
    """Collects ``"<logger>: <message>"`` of every record of the ``volsto`` loggers."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == "volsto" or record.name.startswith("volsto."):
            self.messages.append(f"{record.name}: {record.getMessage()}")


def _shared_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A directory every xdist worker of this pytest run sees: the controller's base temporary
    directory (each worker's own base temp is ``<controller base>/popen-gwN``)."""
    base = tmp_path_factory.getbasetemp()
    return base.parent if os.environ.get("PYTEST_XDIST_WORKER") else base


def _run_toy_precompute(
    root: Path, grid: Path = TOY_GRID, shards: tuple[str, ...] = TOY_SHARDS
) -> dict[str, Any]:
    """The sanctioned toy calibration: ``volsto.viewers.precompute.main`` on ``grid`` (the toy
    grid by default) as the ``shards`` in order (1/2 then 2/2 by default) into
    ``root/A/{store,cache}`` (in process, so the log records are captured), then the synthetic
    M7 / M8b outputs beside them (``tests/_synthetic_store.py``, no computation) — everything the
    pages read."""
    from _synthetic_store import make_synthetic_outputs

    from volsto.calibration.cache import LeverageCache
    from volsto.viewers import precompute

    base = root / "A"
    store, cache = base / "store", base / "cache"
    argv = ["--grid", str(grid), "--store", str(store), "--cache", str(cache)]
    info: dict[str, Any] = {
        "return_codes": [],
        "wall_s": {},
        "stdout": {},
        "log_messages": {},
        "manifests": {},
    }
    root_logger = logging.getLogger()
    handler = _ListHandler()
    old_level = root_logger.level
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)  # the "calibrating" / "resume" records are INFO
    try:
        for s in shards:
            handler.messages = []
            buf = io.StringIO()
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(buf):
                rc = precompute.main([*argv, "--shard", s])
            info["wall_s"][s] = time.perf_counter() - t0
            info["return_codes"].append(int(rc))
            info["stdout"][s] = buf.getvalue()
            info["log_messages"][s] = list(handler.messages)
            man = LeverageCache(cache).manifest()
            info["manifests"][s] = json.loads(man.to_json(orient="records"))
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(old_level)
    make_synthetic_outputs(base / "outputs")
    return info


def _session_build(
    tmp_path_factory: pytest.TempPathFactory,
    dirname: str,
    grid: Path,
    shards: tuple[str, ...],
) -> ToyBuild:
    """One shared build of ``grid`` under ``<controller base temp>/<dirname>`` (the sharing,
    locking and recording of :func:`toy_build`)."""
    root = _shared_root(tmp_path_factory) / dirname
    root.mkdir(exist_ok=True)
    info = _locks.shared_build(root, dirname, lambda: _run_toy_precompute(root, grid, shards))
    return ToyBuild(
        root=root,
        base=root / "A",
        grid_path=grid,
        return_codes=tuple(int(rc) for rc in info.get("return_codes", [])),
        wall_s={str(k): float(v) for k, v in info.get("wall_s", {}).items()},
        stdout={str(k): str(v) for k, v in info.get("stdout", {}).items()},
        log_messages={str(k): list(v) for k, v in info.get("log_messages", {}).items()},
        manifests={str(k): list(v) for k, v in info.get("manifests", {}).items()},
        built_by=str(info.get("built_by", "")),
        error=str(info.get("error", "")),
        shards=shards,
    )


#: Environment switches that change what volsto code does; a test starts with none of them
#: (and sets one itself when it tests it): a developer's or an agent's shell must not leak in.
ISOLATED_ENV: tuple[str, ...] = (
    "VOLSTO_BACKTEST_REQUIRE_PATHS",
    "VOLSTO_FORBID_CALIBRATION",
    "VOLSTO_CALIBRATION_MARKERS",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Clear :data:`ISOLATED_ENV` for every test (restored afterwards by ``monkeypatch``)."""
    for name in ISOLATED_ENV:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def toy_build(tmp_path_factory: pytest.TempPathFactory) -> ToyBuild:
    """The toy store + cache + outputs, built **exactly once per pytest run** (module
    docstring).  Under ``pytest -n auto`` the workers share the build under the controller's
    base temporary directory: the first worker to take the lock (``tests/_locks.py``: a
    directory holding its pid, renamed into place atomically) builds and writes ``done.json``
    (atomically), the others wait for it while its builder is alive.  The fixture
    never fails by itself — a failed build is returned with its reason so every consumer fails
    uniformly through :meth:`ToyBuild.require`.  Consumers must leave the directory intact (copy
    it before writing anything)."""
    return _session_build(tmp_path_factory, "toy_precompute", TOY_GRID, TOY_SHARDS)


@pytest.fixture(scope="session")
def toy_marking_build(tmp_path_factory: pytest.TempPathFactory) -> ToyBuild:
    """The toy **marking** store + cache + outputs (module docstring): ``configs/grids/
    toy_marking.yaml`` built once per pytest run as shard ``1/1``, shared across xdist workers
    like :func:`toy_build`, with the same :class:`ToyBuild` interface (``.require()``,
    ``.store_root``, ``.cache_root``, ``.outputs_root``, ``.grid_path``).  Consumers must leave
    the directory intact."""
    return _session_build(
        tmp_path_factory, "toy_marking_precompute", TOY_MARKING_GRID, TOY_MARKING_SHARDS
    )
