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
Under ``pytest -n auto`` the workers share one build (controller base temp + ``os.mkdir`` lock
+ ``done.json`` marker, stdlib only).  Wall clocks are recorded, never asserted.
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
#: Budget of the toy build (the owner's M9 brief: the toy half of a test skips with a reason
#: when the build takes more than 3 min); reported, never asserted.
TOY_BUDGET_S = 180.0
#: How long an xdist worker waits for another worker's toy build before skipping.
TOY_WAIT_S = 600.0


@dataclass(frozen=True)
class ToyBuild:
    """What the session fixture :func:`toy_build` produced: the roots (store, cache and the
    synthetic M7 / M8b outputs side by side under ``base``) and what the calibrating pass
    recorded per shard — return code, wall clock, captured stdout, captured ``volsto`` log
    messages (``"<logger>: <message>"``) and the leverage-cache manifest after the shard.
    ``error`` is set when the build raised; tests call :meth:`require` to skip on any problem
    with the reason."""

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
    def skip_reason(self) -> str:
        if self.error:
            return f"toy build failed: {self.error}"
        if tuple(self.return_codes) != (0,) * len(TOY_SHARDS):
            return f"toy precompute returned {list(self.return_codes)}"
        if self.total_wall_s > TOY_BUDGET_S:
            return f"toy build took {self.total_wall_s:.0f} s > {TOY_BUDGET_S:.0f} s budget"
        return ""

    def require(self) -> ToyBuild:
        """Skip the calling test with the reason when the build is unusable."""
        if self.skip_reason:
            pytest.skip(self.skip_reason)
        return self

    def calibrating_messages(
        self, shard: str | None = None, logger: str = "volsto.viewers.precompute"
    ) -> list[str]:
        """The ``"calibrating"`` log records of ``logger`` (default: the precompute's own
        "leverage cache miss … calibrating at N particles" line, one per miss; the cache logs a
        second one under ``volsto.calibration.cache``) for ``shard`` or all shards."""
        shards = TOY_SHARDS if shard is None else (shard,)
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


def _run_toy_precompute(root: Path) -> dict[str, Any]:
    """The sanctioned toy calibration: ``volsto.viewers.precompute.main`` on the toy grid as
    shards 1/2 then 2/2 into ``root/A/{store,cache}`` (in process, so the log records are
    captured), then the synthetic M7 / M8b outputs beside them (``tests/_synthetic_store.py``,
    no computation) — everything the pages read."""
    from _synthetic_store import make_synthetic_outputs

    from volsto.calibration.cache import LeverageCache
    from volsto.viewers import precompute

    base = root / "A"
    store, cache = base / "store", base / "cache"
    argv = ["--grid", str(TOY_GRID), "--store", str(store), "--cache", str(cache)]
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
        for s in TOY_SHARDS:
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


@pytest.fixture(scope="session")
def toy_build(tmp_path_factory: pytest.TempPathFactory) -> ToyBuild:
    """The toy store + cache + outputs, built **exactly once per pytest run** (module
    docstring).  Under ``pytest -n auto`` the workers share the build under the controller's
    base temporary directory: the first worker to ``os.mkdir`` the lock builds and writes
    ``done.json`` (atomically), the others wait for it up to :data:`TOY_WAIT_S`.  The fixture
    never skips by itself — a failed or slow build is returned with its reason so every consumer
    skips uniformly through :meth:`ToyBuild.require`.  Consumers must leave the directory
    intact (copy it before writing anything)."""
    root = _shared_root(tmp_path_factory) / "toy_precompute"
    root.mkdir(exist_ok=True)
    done, lock = root / "done.json", root / "lock"
    if not done.exists():
        try:
            os.mkdir(lock)  # atomic: exactly one process builds
        except FileExistsError:
            t0 = time.perf_counter()
            while not done.exists() and time.perf_counter() - t0 < TOY_WAIT_S:
                time.sleep(0.5)
        else:
            info: dict[str, Any] = {"error": "toy build interrupted"}
            try:
                info = _run_toy_precompute(root)
            except Exception as exc:  # the toy half skips with the reason, the rest runs
                info = {"error": f"{type(exc).__name__}: {exc}"}
            finally:
                info.setdefault(
                    "built_by", f"{os.environ.get('PYTEST_XDIST_WORKER', 'main')}:{os.getpid()}"
                )
                tmp = root / "done.json.tmp"
                tmp.write_text(json.dumps(info))
                os.replace(tmp, done)
                os.rmdir(lock)
    if not done.exists():
        info = {"error": f"another worker's toy build did not finish within {TOY_WAIT_S:.0f} s"}
    else:
        info = json.loads(done.read_text())
    return ToyBuild(
        root=root,
        base=root / "A",
        grid_path=TOY_GRID,
        return_codes=tuple(int(rc) for rc in info.get("return_codes", [])),
        wall_s={str(k): float(v) for k, v in info.get("wall_s", {}).items()},
        stdout={str(k): str(v) for k, v in info.get("stdout", {}).items()},
        log_messages={str(k): list(v) for k, v in info.get("log_messages", {}).items()},
        manifests={str(k): list(v) for k, v in info.get("manifests", {}).items()},
        built_by=str(info.get("built_by", "")),
        error=str(info.get("error", "")),
    )
