""" "The same surface": the one comparison of two fitted surfaces across machines
(CONTRIBUTING.md, "Machine-dependent arithmetic"; owner's decision 2026-10-03).

A number that came out of LAPACK or an iterative solver is not reproducible to the last bit on
another machine: numpy and scipy link the operating system's Accelerate on macOS, so the same
pinned wheels give different last bits on another macOS or chip.  Measured on 2026-10-03
(SPEC §13.3): re-importing the 256 tracked HDN snapshots on another Mac moved the fitted
``(ρ, η, γ)`` by up to 1e-7 and the fitted surfaces by at most 7.8e-7 vol points.

So a fresh import is never compared with a stored snapshot by its bytes or by its parameters.
It is compared here, in the unit that matters: the largest difference of implied vol, in vol
points, on the fixed grid :data:`SURFACE_GRID_T` × :data:`SURFACE_GRID_K`, against the one
tolerance :data:`SURFACE_TOL_VP`.  Where a stored snapshot must also be *the import of the same
day* (the backtest's binding), :func:`same_snapshot` adds :func:`structure_diff`: every
non-float leaf outside the ``sabrw`` section and the timestamp — keys, counts, strings, flags —
must be equal, floats being left to the surface comparison (the stored SABRW fits are data,
read back and never re-derived, so they are not compared).  Users: the tests that compare a
committed snapshot with a fresh import (``tests/test_snapshot_portability.py`` walks every
tracked snapshot) and the
backtest migration's binding rule (``volsto.studies.backtest._bind_migrated_snapshots``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import yaml
from numpy.typing import NDArray

from volsto.market.loaders import load_ssvi_surface
from volsto.market.surface import ImpliedSurface

FloatArray = NDArray[np.float64]

SURFACE_TOL_VP: Final[float] = 1e-4
"""Two surfaces within this many vol points on the grid are the same surface (owner's decision
2026-10-03; the measured cross-machine maximum is 7.8e-7 vp over 256 snapshots)."""
SURFACE_GRID_T: Final[tuple[float, ...]] = (1.0 / 12.0, 0.25, 0.5, 1.0, 2.0, 3.0)
"""Maturities of the comparison grid, in years."""
SURFACE_GRID_K: Final[tuple[float, ...]] = tuple(float(k) for k in np.linspace(-0.5, 0.3, 33))
"""Log-moneyness points of the comparison grid."""


def surface_grid_vols(surface: ImpliedSurface) -> FloatArray:
    """Implied vols of ``surface`` on the comparison grid, shape ``(len(T), len(k))``."""
    k = np.asarray(SURFACE_GRID_K)
    return np.array(
        [
            np.sqrt(np.asarray(surface.total_variance(k, T), dtype=np.float64) / T)
            for T in SURFACE_GRID_T
        ]
    )


def surface_diff_vp(a: ImpliedSurface, b: ImpliedSurface) -> float:
    """Largest ``|σ_a − σ_b|`` on the comparison grid, in vol points; ``inf`` when either
    surface is not finite somewhere on the grid."""
    diff = 100.0 * np.abs(surface_grid_vols(a) - surface_grid_vols(b))
    return float(diff.max()) if np.all(np.isfinite(diff)) else float("inf")


def same_surface(a: ImpliedSurface, b: ImpliedSurface, tol_vp: float = SURFACE_TOL_VP) -> bool:
    """Whether ``a`` and ``b`` agree within ``tol_vp`` vol points on the comparison grid."""
    return surface_diff_vp(a, b) <= tol_vp


def snapshot_diff_vp(a: str | Path, b: str | Path) -> float:
    """:func:`surface_diff_vp` of the surfaces of two snapshot files (``market`` + ``ssvi``
    sections, :func:`volsto.market.loaders.load_ssvi_surface`)."""
    return surface_diff_vp(load_ssvi_surface(a), load_ssvi_surface(b))


def same_snapshot_surface(a: str | Path, b: str | Path, tol_vp: float = SURFACE_TOL_VP) -> bool:
    """Whether two snapshot files hold the same surface within ``tol_vp`` vol points."""
    return snapshot_diff_vp(a, b) <= tol_vp


#: Parts of a snapshot document left out of :func:`structure_diff`: the stored SABRW fits are
#: data (never re-derived), the timestamp is the time of the import.
STRUCTURE_SKIPPED: Final[tuple[tuple[str, ...], ...]] = (("sabrw",), ("provenance", "created_utc"))


def structure_diff(a: Any, b: Any, path: tuple[str, ...] = ()) -> str | None:
    """The first place two snapshot documents differ in anything but a float: a key, a length,
    a type, an integer, a string or a flag (``None`` when there is none).  Float leaves are not
    compared (:func:`surface_diff_vp` judges them); :data:`STRUCTURE_SKIPPED` is left out."""
    if path in STRUCTURE_SKIPPED:
        return None
    where = ".".join(path) or "<root>"
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        if list(a) != list(b):
            keys = sorted(set(map(str, a)) ^ set(map(str, b)))
            return f"{where}: keys differ ({keys or 'order'})"
        for key in a:
            found = structure_diff(a[key], b[key], (*path, str(key)))
            if found:
                return found
        return None
    if isinstance(a, Sequence) and isinstance(b, Sequence) and not isinstance(a, str | bytes):
        if isinstance(b, str | bytes) or len(a) != len(b):
            return f"{where}: lengths differ"
        for i, (x, y) in enumerate(zip(a, b)):
            found = structure_diff(x, y, (*path, str(i)))
            if found:
                return found
        return None
    if isinstance(a, float) and isinstance(b, float):
        return None
    if type(a) is not type(b) or a != b:
        return f"{where}: {a!r} != {b!r}"
    return None


def snapshot_difference(a: str | Path, b: str | Path, tol_vp: float = SURFACE_TOL_VP) -> str | None:
    """Why two snapshot files are not the same import (``None`` when they are): the first
    structural difference (:func:`structure_diff`), else a surface difference beyond
    ``tol_vp`` vol points."""
    doc_a = yaml.safe_load(Path(a).read_text(encoding="utf-8"))
    doc_b = yaml.safe_load(Path(b).read_text(encoding="utf-8"))
    found = structure_diff(doc_a, doc_b)
    if found:
        return found
    diff = snapshot_diff_vp(a, b)
    if not diff <= tol_vp:
        return f"the surfaces differ by {diff:.3g} vol points (tolerance {tol_vp:g})"
    return None


def same_snapshot(a: str | Path, b: str | Path, tol_vp: float = SURFACE_TOL_VP) -> bool:
    """Whether two snapshot files are the same import: equal in everything but floats, and the
    same surface within ``tol_vp`` vol points."""
    return snapshot_difference(a, b, tol_vp) is None
