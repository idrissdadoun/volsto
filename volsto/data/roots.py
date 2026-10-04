"""The two data roots of the vendor data layer and the free-space check (SPEC §18, M11 Part 1).

``raw``   vendor files exactly as delivered (zips), never modified, never unzipped to disk; may
          sit on an external volume.  Environment ``VOLSTO_DATA_RAW``, flag ``--raw``.
``store`` the typed Parquet store derived from ``raw`` (rebuildable).  Environment
          ``VOLSTO_DATA_STORE``, flag ``--store``.

Resolution follows the viewers' ``VOLSTO_*`` pattern (:mod:`volsto.viewers.config`), later
sources winning: the documented defaults below (under the repository's git-ignored ``data/``),
the environment variables, the CLI flags.  A relative path given by the user is kept relative
to the current working directory.  Each vendor owns one sub-directory of each root
(``<raw>/orats/``, ``<store>/orats/``).

Vendor data never enters git: both defaults lie under ``data/`` (ignored) and
``tests/test_data_layer.py::test_no_tracked_file_under_the_data_roots`` fails if a tracked file
sits under ``data/`` or under either resolved root.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import volsto

REPO_ROOT = Path(volsto.__file__).resolve().parents[1]
#: Everything under this directory is git-ignored (vendor data is licensed, local only).
DATA_DIR = REPO_ROOT / "data"
#: Vendor files as delivered.
DEFAULT_RAW_ROOT = DATA_DIR / "raw"
#: Typed Parquet store.
DEFAULT_STORE_ROOT = DATA_DIR / "store"
ENV_RAW = "VOLSTO_DATA_RAW"
ENV_STORE = "VOLSTO_DATA_STORE"
#: Free space kept on a volume after any bulk write (10 GiB: macOS degrades on a full system
#: volume, and the estimate of a bulk write is never exact).
FREE_SPACE_MARGIN_BYTES: int = 10 * 1024**3


class DataError(RuntimeError):
    """A refusal or a failed check of the data layer; the message says what to do."""


@dataclass(frozen=True)
class DataRoots:
    """The resolved roots and where each came from (``default``, ``env`` or ``flag``)."""

    raw: Path
    store: Path
    raw_source: str
    store_source: str

    @classmethod
    def resolve(
        cls,
        *,
        raw: str | Path | None = None,
        store: str | Path | None = None,
        env: Mapping[str, str] | None = None,
    ) -> DataRoots:
        env = os.environ if env is None else env

        def pick(flag: str | Path | None, var: str, default: Path) -> tuple[Path, str]:
            if flag is not None:
                if not str(flag).strip():
                    raise DataError("a blank path was given for a data root")
                return Path(flag), "flag"
            val = env.get(var)
            if val is not None:
                if not val.strip():
                    raise DataError(f"{var} is set but blank")
                return Path(val), "env"
            return default, "default"

        r, rs = pick(raw, ENV_RAW, DEFAULT_RAW_ROOT)
        s, ss = pick(store, ENV_STORE, DEFAULT_STORE_ROOT)
        return cls(r, s, rs, ss)

    def raw_dir(self, vendor: str) -> Path:
        return self.raw / vendor

    def store_dir(self, vendor: str) -> Path:
        return self.store / vendor


#: Where macOS mounts external volumes.  A path below it whose volume directory is absent is an
#: unmounted disk: creating it would silently write to the internal disk instead.
VOLUMES_DIR = Path("/Volumes")


def ensure_dir(path: Path, *, volumes_dir: Path = VOLUMES_DIR) -> Path:
    """Create ``path`` (and parents) and return it — refusing when it lies on a volume that is
    not mounted (``/Volumes/<name>`` absent).  Raw and store may both sit on external volumes
    (owner's decision 2026-10-03): nothing assumes the internal disk, and nothing may fall back
    to it silently."""
    p = Path(os.path.abspath(path))
    try:
        rel = p.relative_to(volumes_dir)
    except ValueError:
        rel = None
    if rel is not None and rel.parts and not (volumes_dir / rel.parts[0]).is_dir():
        raise DataError(
            f"{path}: the volume {volumes_dir / rel.parts[0]} is not mounted (refusing to "
            "create the directory on the internal disk)"
        )
    p.mkdir(parents=True, exist_ok=True)
    return p


def existing_ancestor(path: Path) -> Path:
    """``path`` or its nearest existing ancestor (the volume a write to ``path`` lands on)."""
    p = path.resolve()
    while not p.exists():
        if p.parent == p:
            break
        p = p.parent
    return p


def free_bytes(path: Path) -> int:
    """Free bytes on the volume holding ``path`` (which need not exist yet)."""
    return int(shutil.disk_usage(existing_ancestor(path)).free)


def require_free_space(
    path: Path, needed_bytes: int, *, what: str, margin_bytes: int = FREE_SPACE_MARGIN_BYTES
) -> int:
    """Refuse (raise :class:`DataError`) unless the volume of ``path`` holds ``needed_bytes``
    plus ``margin_bytes``; returns the free bytes.  Every bulk write of the data layer calls
    this before writing."""
    free = free_bytes(path)
    if free < needed_bytes + margin_bytes:
        raise DataError(
            f"not enough free space for {what}: {fmt_bytes(needed_bytes)} needed plus a "
            f"{fmt_bytes(margin_bytes)} margin, {fmt_bytes(free)} free on the volume of {path}"
        )
    return free


def fmt_bytes(n: int | float) -> str:
    """``n`` bytes in binary units, one decimal."""
    x = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(x) < 1024.0:
            return f"{x:.0f} {unit}" if unit == "B" else f"{x:.1f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TiB"
