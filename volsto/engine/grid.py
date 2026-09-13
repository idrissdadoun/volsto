"""Simulation time grid: union of product fixings and discretisation steps (SPEC §5).

``TimeGrid.build(fixing_times, dt_max, calibration_grid=None)`` returns the sorted union of
``{0}``, the fixing times and any extra times (e.g. leverage calibration slices), each interval
subdivided evenly so that every step is ``≤ dt_max``.  Path containers record the state at the
*columns* ``[0] + fixing times``; :class:`FixingIndex` maps a fixing time to its column.
Checked by ``tests/test_engine.py``.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

_TOL = 1e-9


def _clean_times(ts: ArrayLike) -> FloatArray:
    a = np.atleast_1d(np.asarray(ts, dtype=np.float64)).ravel()
    if a.size == 0:
        return a
    if np.any(a < -_TOL) or not np.all(np.isfinite(a)):
        raise ValueError("times must be finite and non-negative")
    a = np.sort(np.maximum(a, 0.0))
    keep = np.concatenate(([True], np.diff(a) > _TOL))
    return a[keep]


class FixingIndex:
    """Map from fixing times (years) to record columns, tolerant to 1e-9 rounding."""

    def __init__(self, times: FloatArray) -> None:
        self._times = np.asarray(times, dtype=np.float64)
        if np.any(np.diff(self._times) <= 0):
            raise ValueError("record times must be strictly increasing")

    @property
    def times(self) -> FloatArray:
        return self._times

    def __len__(self) -> int:
        return int(self._times.size)

    def __contains__(self, t: object) -> bool:
        if not isinstance(t, (int, float, np.floating)):
            return False
        i = int(np.searchsorted(self._times, float(t)))
        for j in (i - 1, i):
            if 0 <= j < self._times.size and abs(self._times[j] - float(t)) <= _TOL:
                return True
        return False

    def __getitem__(self, t: float) -> int:
        i = int(np.searchsorted(self._times, float(t)))
        for j in (i - 1, i):
            if 0 <= j < self._times.size and abs(self._times[j] - float(t)) <= _TOL:
                return j
        raise KeyError(f"time {t} is not a recorded fixing")

    def indices(self, ts: Iterable[float] | FloatArray) -> IntArray:
        return np.array(
            [self[float(t)] for t in np.atleast_1d(np.asarray(ts, dtype=np.float64))],
            dtype=np.int64,
        )

    def __repr__(self) -> str:
        return f"FixingIndex(n={len(self)}, times={self._times.round(6).tolist()})"


class TimeGrid:
    """Sorted simulation times ``t_0 = 0 < t_1 < … < t_n`` with record columns.

    Attributes:
        times: ``(n_steps + 1,)`` grid times.
        dts: ``(n_steps,)`` step sizes.
        record_times: column times of the path container (``0`` plus all fixing times).
        record_steps: index into ``times`` of each record column.
        step_record: ``(n_steps,)`` column written after each step, or ``-1``.
    """

    def __init__(self, times: FloatArray, record_times: FloatArray) -> None:
        times = np.asarray(times, dtype=np.float64)
        record_times = np.asarray(record_times, dtype=np.float64)
        if times.size < 2 or times[0] != 0.0 or np.any(np.diff(times) <= 0):
            raise ValueError("times must start at 0 and be strictly increasing with ≥ 2 points")
        if record_times.size == 0 or record_times[0] != 0.0:
            raise ValueError("record_times must start with 0")
        self.times = times
        self.dts = np.diff(times)
        self.record_times = record_times
        self.fixing_index = FixingIndex(record_times)
        pos = np.searchsorted(times, record_times)
        if np.any(pos >= times.size) or np.any(
            np.abs(times[np.minimum(pos, times.size - 1)] - record_times) > _TOL
        ):
            raise ValueError("every record time must be a grid time")
        self.record_steps = pos.astype(np.int64)
        step_record = np.full(self.dts.size, -1, dtype=np.int64)
        step_record[self.record_steps[1:] - 1] = np.arange(1, record_times.size)
        self.step_record = step_record

    @classmethod
    def build(
        cls,
        fixing_times: ArrayLike,
        dt_max: float,
        calibration_grid: ArrayLike | None = None,
        extra_times: ArrayLike | None = None,
    ) -> TimeGrid:
        """Union of ``{0}``, fixings, calibration slices and extras, refined to ``dt ≤ dt_max``."""
        if dt_max <= 0:
            raise ValueError("dt_max must be positive")
        fix = _clean_times(fixing_times)
        if fix.size == 0 or fix[-1] <= 0:
            raise ValueError("at least one positive fixing time is required")
        record = _clean_times(np.concatenate(([0.0], fix)))
        anchors = [record]
        for extra in (calibration_grid, extra_times):
            if extra is not None:
                e = _clean_times(extra)
                anchors.append(e[e <= record[-1]])
        knots = _clean_times(np.concatenate(anchors))
        pieces = [np.array([0.0])]
        for a, b in itertools.pairwise(knots):
            n_sub = max(1, math.ceil((b - a) / dt_max * (1.0 - 1e-9)))
            pieces.append(np.linspace(a, b, n_sub + 1)[1:])
        times = np.concatenate(pieces)
        # snap to record times exactly (avoid 1e-16 drift from linspace)
        idx = np.searchsorted(times, record)
        times[idx] = record
        return cls(times, record)

    @property
    def n_steps(self) -> int:
        return int(self.dts.size)

    @property
    def n_records(self) -> int:
        return int(self.record_times.size)

    @property
    def horizon(self) -> float:
        return float(self.times[-1])

    @property
    def dt_max(self) -> float:
        return float(self.dts.max())

    def __repr__(self) -> str:
        return (
            f"TimeGrid(n_steps={self.n_steps}, horizon={self.horizon:.6g}, "
            f"dt_max={self.dt_max:.4g}, n_records={self.n_records})"
        )
