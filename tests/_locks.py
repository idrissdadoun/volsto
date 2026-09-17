"""The cross-process build lock of the session fixtures (``pytest -n``).

The scheme is ``tests/conftest.py``'s ``toy_build`` one, moved here so that every calibrating
session fixture shares it (``tests/conftest.py`` and ``tests/_backtest_build.py`` both import it).  One process takes the lock
by renaming a directory holding its pid into place (atomic, and the lock is never removed: one
build per pytest run); the others wait while that process is alive, however long the build
takes — no wall-clock limit — and then read the build's ``done.json`` (its absence is the
builder's death, reported with its pid).  :func:`shared_build` is the whole sequence; the
session fixtures of ``tests/conftest.py`` and ``tests/_backtest_build.py`` both call it.  A pid
recycled by another process while its builder is dead would keep a waiter waiting (the pid
check cannot tell them apart); a pytest run is short enough for that to be theoretical.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: How often a waiting worker looks for the build's result.
POLL_S = 0.5


def take_lock(lock: Path) -> bool:
    """Atomically create ``lock`` holding this process's pid: a directory is renamed into place,
    which fails when another process's (non-empty, never removed) lock is already there."""
    staging = lock.with_name(f"{lock.name}.{os.getpid()}")
    staging.mkdir()
    (staging / "pid").write_text(str(os.getpid()))
    try:
        os.rename(staging, lock)
    except OSError:
        (staging / "pid").unlink()
        staging.rmdir()
        return False
    return True


def owner(lock: Path) -> str:
    """``pid <n>`` of the lock's holder (``pid unknown`` when unreadable)."""
    with contextlib.suppress(OSError):
        return f"pid {(lock / 'pid').read_text().strip()}"
    return "pid unknown"


def alive(lock: Path) -> bool:
    """Whether the process named in ``lock`` still runs."""
    try:
        pid = int((lock / "pid").read_text())
    except (OSError, ValueError):
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for(done: Path, lock: Path) -> None:
    """Wait for another process's build: as long as its builder is alive, however long it
    takes (no wall-clock limit; the result, or its absence, decides)."""
    while not done.exists() and alive(lock):
        time.sleep(POLL_S)


def shared_build(root: Path, label: str, build: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """One build per pytest run under ``root`` shared by every worker: the process that takes
    the lock runs ``build()`` (an exception is recorded as the build's error) and writes
    ``done.json`` atomically, leaving the lock in place; any other process waits while that
    process is alive.  Returns the recorded build, or an error naming the builder when it died
    without a result."""
    done, lock = root / "done.json", root / "lock"
    if not done.exists():
        if take_lock(lock):
            info: dict[str, Any] = {"error": f"{label} build interrupted"}
            try:
                info = build()
            except Exception as exc:  # recorded: every consumer fails with the reason
                info = {"error": f"{type(exc).__name__}: {exc}"}
            finally:
                info.setdefault(
                    "built_by", f"{os.environ.get('PYTEST_XDIST_WORKER', 'main')}:{os.getpid()}"
                )
                tmp = root / "done.json.tmp"
                tmp.write_text(json.dumps(info))
                os.replace(tmp, done)  # the lock stays: one build per pytest run
        else:
            wait_for(done, lock)
    if done.exists():
        return dict(json.loads(done.read_text()))
    return {"error": f"the process building {label} ({owner(lock)}) died without a result"}
