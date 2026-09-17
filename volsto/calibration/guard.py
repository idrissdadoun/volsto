"""Process-wide calibration guard (SPEC §10.1, M10 Part 1): the **one place** that decides whether
a leverage calibration may run, and the one place that counts the calibrations a run attempted.

:func:`volsto.calibration.particle.calibrate_leverage` — the only function that runs the particle
calibration (every ``LeverageCache`` miss, every recalibrating ``RiskEngine`` / ``LSVBuilder``,
every direct call ends there; ``tests/test_study_runner.py::test_particle_kernel_reached_only_
through_calibrate_leverage`` walks ``volsto/`` for it) — calls :func:`check_calibration_allowed`
as its first statement.  Calibration is forbidden while

* the process flag is set: a :func:`calibration_forbidden` block is active in any thread of the
  process (a depth counter under a lock, so nested and concurrent blocks compose), or
* the environment variable :data:`ENV_VAR` (``VOLSTO_FORBID_CALIBRATION``) holds anything but
  ``""`` / ``"0"``.  :func:`calibration_forbidden` sets it for its duration, so a child process
  started inside the block (``multiprocessing`` spawn or fork, a ``ProcessPoolExecutor``, a
  subprocess) inherits the prohibition; a user can also export it to protect a whole shell.

Because the check sits inside the function body, it holds however the function object is
reached: an alias bound at import time by any module, a dict or ``functools.partial`` holding it,
a thread pool, a thread that outlives the study's ``compute`` while the block is still active.

**Counting.**  Every call is counted in the process — refused (:func:`refusals`) or allowed, i.e.
a calibration started (:func:`calibrations`) — and, when :data:`MARKER_ENV`
(``VOLSTO_CALIBRATION_MARKERS``, set by ``calibration_forbidden(marker_dir=...)`` and inherited
by children like :data:`ENV_VAR`) names a directory, also as a marker file there
(``refused-<pid>-<id>.json`` / ``started-<pid>-<id>.json`` with the pid, host and time), so a run
learns what its own process **and its children** did (:func:`read_markers`).  A started
calibration while a marker directory is set can only come from a process that escaped the
prohibition (a child whose environment dropped :data:`ENV_VAR` but kept the marker variable);
it is what the study runner reports as ``recalibrated``.

What it cannot see: a thread or process that calibrates after the block has ended (its markers,
if any, land in a directory the run has already read or removed), and a child process started
before the block whose environment was fixed earlier.  Several blocks in one process share the
outermost block's marker directory.

The module imports nothing from ``volsto`` (no import cycle with ``particle``).  Checked by
``tests/test_study_runner.py`` (the guard tests: early alias, dict, partial, thread pool, spawned
process pool, a thread calibrating during render, nesting and threads, the environment variables,
a child that escaped the prohibition).
"""

from __future__ import annotations

import contextlib
import json
import os
import secrets
import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

#: Environment variable forbidding calibration in this process and its children.
ENV_VAR = "VOLSTO_FORBID_CALIBRATION"
#: Values of :data:`ENV_VAR` that do not forbid.
ENV_OFF: frozenset[str] = frozenset({"", "0"})
#: Environment variable naming the directory where every call leaves a marker file.
MARKER_ENV = "VOLSTO_CALIBRATION_MARKERS"
#: Marker kinds (file-name prefixes).
REFUSED = "refused"
STARTED = "started"

_lock = threading.Lock()
_depth = 0
_saved_env: dict[str, str | None] = {}
_refusals = 0
_calibrations = 0


class CalibrationForbiddenError(RuntimeError):
    """A leverage calibration was attempted where it is forbidden (a running study, or
    :data:`ENV_VAR` set)."""


def calibration_is_forbidden() -> bool:
    """Whether :func:`check_calibration_allowed` would raise now."""
    return _depth > 0 or os.environ.get(ENV_VAR, "").strip() not in ENV_OFF


def _mark(kind: str, what: str) -> None:
    """A marker file in the :data:`MARKER_ENV` directory (best effort: a directory that is gone
    — the run has ended — is ignored)."""
    directory = os.environ.get(MARKER_ENV, "").strip()
    if not directory:
        return
    record: dict[str, Any] = {
        "kind": kind,
        "what": what,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "time": time.time(),
    }
    name = f"{kind}-{os.getpid()}-{secrets.token_hex(6)}.json"
    with contextlib.suppress(OSError):
        (Path(directory) / name).write_text(json.dumps(record), encoding="utf-8")


def check_calibration_allowed(what: str = "calibrate_leverage") -> None:
    """Raise :class:`CalibrationForbiddenError` when calibration is forbidden; count the call
    either way (module docstring)."""
    global _refusals, _calibrations
    if calibration_is_forbidden():
        with _lock:
            _refusals += 1
        _mark(REFUSED, what)
        why = (
            "a study is running in this process"
            if _depth > 0
            else f"{ENV_VAR}={os.environ.get(ENV_VAR)!r} is set (inherited from a study run?)"
        )
        raise CalibrationForbiddenError(
            f"leverage calibration refused ({what}): {why}. Studies never calibrate: produce the "
            "leverage with volsto-precompute and read it through the cache."
        )
    with _lock:
        _calibrations += 1
    _mark(STARTED, what)


def _set_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


@contextlib.contextmanager
def calibration_forbidden(marker_dir: str | os.PathLike[str] | None = None) -> Iterator[None]:
    """Forbid calibration in this process (all threads) and in the children it starts, for the
    duration of the block; nesting-safe and thread-safe (depth counter under a lock).  The
    outermost block sets :data:`ENV_VAR` and, with ``marker_dir``, :data:`MARKER_ENV`; both are
    restored to their previous values when it exits (a nested block's ``marker_dir`` is
    ignored)."""
    global _depth
    with _lock:
        if _depth == 0:
            _saved_env.clear()
            _saved_env[ENV_VAR] = os.environ.get(ENV_VAR)
            os.environ[ENV_VAR] = "1"
            if marker_dir is not None:
                _saved_env[MARKER_ENV] = os.environ.get(MARKER_ENV)
                os.environ[MARKER_ENV] = str(Path(marker_dir))
        _depth += 1
    try:
        yield
    finally:
        with _lock:
            _depth -= 1
            if _depth == 0:
                for name, value in _saved_env.items():
                    _set_env(name, value)
                _saved_env.clear()


def depth() -> int:
    """The number of active :func:`calibration_forbidden` blocks (diagnostics and tests)."""
    return _depth


def refusals() -> int:
    """How many calibrations this process has refused so far (compare two readings)."""
    return _refusals


def calibrations() -> int:
    """How many calibrations this process has started so far (compare two readings)."""
    return _calibrations


def read_markers(marker_dir: str | os.PathLike[str]) -> dict[str, list[dict[str, Any]]]:
    """The marker records of a directory, by kind (``refused`` / ``started``)."""
    out: dict[str, list[dict[str, Any]]] = {REFUSED: [], STARTED: []}
    directory = Path(marker_dir)
    if not directory.is_dir():
        return out
    for p in sorted(directory.glob("*.json")):
        try:
            record = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            record = {"kind": p.name.split("-", 1)[0], "unreadable": p.name}
        kind = str(record.get("kind", ""))
        if kind in out:
            out[kind].append(record)
    return out
