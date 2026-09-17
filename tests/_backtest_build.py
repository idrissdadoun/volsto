"""The toy rolling-backtest build (M10 Part 3) — a sanctioned calibrating session fixture.

:func:`toy_backtest_build` runs ``volsto-backtest run configs/backtest/hdn_2022h2_toy.yaml``
**with calibration** on the 5 HDN dates of that config (2022-07-27 … 2022-08-02: 5 leverage
calibrations at 2·10⁴ particles over a 1y horizon, plus one for 2022-07-28 marked from a copy of
its vendor day file whose SPX close is moved by 0.5 % (the integrity walking test), half-year term sheets, 2000 pricing paths,
the sticky-leverage attribution at parallel detail) into a pytest temporary directory — never
the repository cache or outputs — once per pytest run, unsharded.  It is written so that the
orchestrator can move it into ``tests/conftest.py`` verbatim (it imports nothing from
``conftest``; the sharing helpers are private copies of the ``toy_build`` ones).  Consumers
import it (``from _backtest_build import toy_backtest_build  # noqa: F401``: the tests directory
is on ``sys.path``, the repository root is not, so ``tests._backtest_build`` does not import
under ``.venv/bin/pytest``).

Layout of the build (``BacktestBuild.base``)::

    outputs/backtest/hdn_2022h2_toy/             the per-date store (backtest.json, probe.json,
                                                 dates/)
    outputs/backtest/hdn_2022h2_toy/snapshots/   the 5 imported snapshots (inside the store, as
                                                 the toy config keeps them: stage 2 finds them
                                                 with the store)
    cache/                                       the 5 toy leverages
    shifted/{hdn,snapshots,cache}/               a data root whose 2022-07-28 day file has its
                                                 SPX close x 1.005, and that state's leverage
                                                 (a 6th calibration, same size)

Under ``pytest -n auto`` the workers share one build (controller base temp + ``os.mkdir`` lock +
``done.json`` marker, the ``toy_build`` scheme).  The wall clock, the captured stdout and the
``volsto`` log records are recorded, never asserted.  The fixture never skips by itself: a
missing HDN sample, a failed build or a non-zero exit is returned with its reason; through
:meth:`BacktestBuild.require` a consumer skips on the absent sample and FAILS on a failed
build.  Consumers must leave the directory intact (:meth:`BacktestBuild.copy` first).
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

BACKTEST_ROOT = Path(__file__).resolve().parents[1]
#: The toy backtest config the fixture runs.
TOY_BACKTEST_CONFIG = BACKTEST_ROOT / "configs" / "backtest" / "hdn_2022h2_toy.yaml"
#: The HDN sample the config reads (git-ignored; the fixture reports its absence).
TOY_BACKTEST_DATA = BACKTEST_ROOT / "data" / "hdn_sample" / "options_sample_2022H2" / "day_by_date"
#: The five dates of the toy config.
TOY_BACKTEST_DATES: tuple[str, ...] = (
    "2022-07-27",
    "2022-07-28",
    "2022-07-29",
    "2022-08-01",
    "2022-08-02",
)
#: The store's location under the build's outputs root (the study's ``params.store``).
TOY_BACKTEST_STORE = "backtest/hdn_2022h2_toy"
#: The date whose close the integrity walking test moves, and the move: the fixture also
#: calibrates that date's leverage with its snapshot's spot times this factor (one more
#: calibration at the same size), so the test can recompute the date into another state.
TOY_SHIFTED_DATE = "2022-07-28"
TOY_SHIFTED_SPOT = 1.005
#: How long an xdist worker waits for another worker's build before reporting it unusable.
TOY_BACKTEST_WAIT_S = 1800.0


@dataclass(frozen=True)
class BacktestBuild:
    """What :func:`toy_backtest_build` produced (module docstring)."""

    root: Path
    base: Path
    config_path: Path
    return_code: int
    wall_s: float
    stdout: str
    log_messages: tuple[str, ...]
    built_by: str
    error: str = ""
    absent: str = ""
    dates: tuple[str, ...] = TOY_BACKTEST_DATES
    info: dict[str, Any] = field(default_factory=dict, compare=False)

    @property
    def outputs_root(self) -> Path:
        return self.base / "outputs"

    @property
    def store_root(self) -> Path:
        return self.outputs_root / TOY_BACKTEST_STORE

    @property
    def cache_root(self) -> Path:
        return self.base / "cache"

    @property
    def snapshots_root(self) -> Path:
        return self.store_root / "snapshots"

    @property
    def shifted_root(self) -> Path:
        """``shifted/{hdn,snapshots,cache}``: a data root whose :data:`TOY_SHIFTED_DATE` day
        file has its SPX close moved by :data:`TOY_SHIFTED_SPOT`, and the leverage of the state
        marked from it (:func:`_calibrate_shifted`)."""
        return self.base / "shifted"

    @property
    def shifted_key(self) -> str:
        return str(self.info.get("shifted_key", ""))

    def path_args(self, *, out: Path | None = None) -> list[str]:
        """``--out --cache --snapshots`` of this build (``out`` overrides the store)."""
        return [
            "--out",
            str(out if out is not None else self.store_root),
            "--cache",
            str(self.cache_root),
            "--snapshots",
            str(self.snapshots_root),
        ]

    @property
    def skip_reason(self) -> str:
        if self.absent:
            return self.absent
        if self.error:
            return f"toy backtest build failed: {self.error}"
        if self.return_code != 0:
            return f"toy backtest run returned {self.return_code}"
        return ""

    def require(self) -> BacktestBuild:
        """Skip the calling test when the HDN sample is absent (the only acceptable reason);
        **fail** it when the build itself failed or exited non-zero."""
        if self.absent:
            pytest.skip(self.absent)
        if self.skip_reason:
            pytest.fail(self.skip_reason)
        return self

    def copy(self, dest: Path) -> BacktestBuild:
        """A private copy of the build under ``dest`` (the consumer may write into it)."""
        shutil.copytree(self.base, dest)
        return dataclasses.replace(self, base=dest)


class _BacktestLogHandler(logging.Handler):
    """Collects ``"<logger>: <message>"`` of every record of the ``volsto`` loggers."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.name == "volsto" or record.name.startswith("volsto."):
            self.messages.append(f"{record.name}: {record.getMessage()}")


def _backtest_shared_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The controller's base temporary directory (shared by every xdist worker)."""
    base = tmp_path_factory.getbasetemp()
    return base.parent if os.environ.get("PYTEST_XDIST_WORKER") else base


def _backtest_absent() -> str:
    missing = [
        d for d in TOY_BACKTEST_DATES if not (TOY_BACKTEST_DATA / f"{d}_options.csv").is_file()
    ]
    if missing:
        return (
            f"HDN sample absent: {TOY_BACKTEST_DATA} lacks {missing} (git-ignored; unzip "
            "options_sample_2022H2.zip into data/hdn_sample/)"
        )
    return ""


def _run_toy_backtest(root: Path) -> dict[str, Any]:
    """The sanctioned calibration: ``volsto.studies.backtest.main(["run", <toy config>, ...])``
    in process (so the log records are captured) into ``root/A``."""
    from volsto.studies import backtest

    base = root / "A"
    store = base / "outputs" / TOY_BACKTEST_STORE
    argv = [
        "run",
        str(TOY_BACKTEST_CONFIG),
        "--out",
        str(store),
        "--cache",
        str(base / "cache"),
        "--snapshots",
        str(store / "snapshots"),
    ]
    root_logger = logging.getLogger()
    handler = _BacktestLogHandler()
    old_level = root_logger.level
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO)
    buf = io.StringIO()
    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(buf):
            rc = backtest.main(argv)
    finally:
        root_logger.removeHandler(handler)
        root_logger.setLevel(old_level)
    wall = time.perf_counter() - t0
    shifted: dict[str, Any] = {}
    if rc == 0:
        shifted = _calibrate_shifted(base)
    calibrated = []
    for d in TOY_BACKTEST_DATES:
        pointer = store / "dates" / d / "CURRENT"
        if pointer.is_file():
            attempt = json.loads(pointer.read_text())["attempt"]
            done = store / "dates" / d / "attempts" / attempt / "done.json"
            calibrated += json.loads(done.read_text())["calibrated"]
    print(f"toy backtest build: exit {rc}, wall clock {wall:.1f} s, calibrated {calibrated}")
    return {
        **shifted,
        "return_code": int(rc),
        "wall_s": wall,
        "stdout": buf.getvalue(),
        "log_messages": list(handler.messages),
        "calibrated": calibrated,
    }


def shift_vendor_day(src: Path, dest: Path, factor: float) -> None:
    """A copy of an HDN day file whose SPX / SPXW rows have their ``underlying_close`` (the
    importer's spot) times ``factor``; every other byte is kept."""
    lines = src.read_text(encoding="utf-8").splitlines(keepends=True)
    header = lines[0].rstrip("\r\n").split(",")
    col = header.index("underlying_close")
    out = [lines[0]]
    for line in lines[1:]:
        if '"' in line:
            raise ValueError(f"{src.name}: quoted fields are not supported")
        if line.startswith("SPX"):
            end = line[len(line.rstrip("\r\n")) :]
            fields = line.rstrip("\r\n").split(",")
            fields[col] = repr(float(fields[col]) * factor)
            line = ",".join(fields) + end
        out.append(line)
    dest.write_text("".join(out), encoding="utf-8")


def _calibrate_shifted(base: Path) -> dict[str, Any]:
    """The fixture's second calibration, for the integrity walking test: a private data root
    ``base/shifted/hdn`` (links to the sample's day files of the toy dates and a copy of its
    manifest) whose :data:`TOY_SHIFTED_DATE` day file has its SPX close times
    :data:`TOY_SHIFTED_SPOT` (:func:`shift_vendor_day`); that date is imported from it, marked,
    and its leverage calibrated into ``base/shifted/cache`` at the toy size."""
    from volsto.studies import backtest

    root = base / "shifted"
    days = root / "hdn" / "day_by_date"
    days.mkdir(parents=True)
    shutil.copyfile(TOY_BACKTEST_DATA / "manifest.json", days / "manifest.json")
    for d in TOY_BACKTEST_DATES:
        name = f"{d}_options.csv"
        if d == TOY_SHIFTED_DATE:
            shift_vendor_day(TOY_BACKTEST_DATA / name, days / name, TOY_SHIFTED_SPOT)
        else:
            (days / name).symlink_to(TOY_BACKTEST_DATA / name)
    raw = backtest.load_backtest_config(TOY_BACKTEST_CONFIG).to_mapping()
    raw["data"]["root"] = str(root / "hdn")
    cfg = backtest.BacktestConfig.from_mapping(raw, source=str(TOY_BACKTEST_CONFIG)).with_paths(
        out=root / "out", cache=root / "cache", snapshots=root / "snapshots"
    )
    run = backtest.BacktestRun(cfg, allow_calibrate=True)
    st = run.state(TOY_SHIFTED_DATE)
    calibrated, seconds = run.ensure_leverage(st)
    print(f"toy backtest build: shifted {TOY_SHIFTED_DATE} leverage in {seconds:.1f} s")
    return {"shifted_key": st.key, "shifted_spot": st.spot, "shifted_calibrated": calibrated}


@pytest.fixture(scope="session")
def toy_backtest_build(tmp_path_factory: pytest.TempPathFactory) -> BacktestBuild:
    """The toy backtest store + cache + snapshots, built **once per pytest run** with
    calibration (module docstring).  Consumers call ``.require()`` and copy before writing."""
    root = _backtest_shared_root(tmp_path_factory) / "toy_backtest"
    root.mkdir(exist_ok=True)
    absent = _backtest_absent()
    done, lock = root / "done.json", root / "lock"
    info: dict[str, Any]
    if absent:
        info = {}
    else:
        if not done.exists():
            try:
                os.mkdir(lock)  # atomic: exactly one process builds
            except FileExistsError:
                t0 = time.perf_counter()
                while not done.exists() and time.perf_counter() - t0 < TOY_BACKTEST_WAIT_S:
                    time.sleep(0.5)
            else:
                info = {"error": "toy backtest build interrupted"}
                try:
                    info = _run_toy_backtest(root)
                except Exception as exc:  # the consumers skip with the reason
                    info = {"error": f"{type(exc).__name__}: {exc}"}
                finally:
                    info.setdefault(
                        "built_by",
                        f"{os.environ.get('PYTEST_XDIST_WORKER', 'main')}:{os.getpid()}",
                    )
                    tmp = root / "done.json.tmp"
                    tmp.write_text(json.dumps(info))
                    os.replace(tmp, done)
                    os.rmdir(lock)
        if done.exists():
            info = json.loads(done.read_text())
        else:
            info = {
                "error": f"another worker's backtest build did not finish in {TOY_BACKTEST_WAIT_S:.0f} s"
            }
    return BacktestBuild(
        root=root,
        base=root / "A",
        config_path=TOY_BACKTEST_CONFIG,
        return_code=int(info.get("return_code", -1)),
        wall_s=float(info.get("wall_s", float("nan"))),
        stdout=str(info.get("stdout", "")),
        log_messages=tuple(info.get("log_messages", ())),
        built_by=str(info.get("built_by", "")),
        error=str(info.get("error", "")),
        absent=absent,
        info=info,
    )
