"""Leverage function ``L(t, S)`` of the LSV model (SPEC §3.5, §4.3).

Stored on a ``(t, k)`` grid with ``k = ln(S / F(t))`` — a log-spot grid shifted by the forward,
so that the ±6-standard-deviation window stays centred — with linear interpolation in ``k`` and
flat extrapolation, linear interpolation in ``t`` between slices, the first slice below ``t_0``
and the last slice held constant beyond the calibration horizon.  Serialises to ``.npz`` with its
provenance metadata (JSON).  Checked by ``tests/test_lsv.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.market.curves import DiscountCurve, ForwardCurve

FloatArray = NDArray[np.float64]


class LeverageFunction:
    """``L(t, S)`` on a ``(t, k)`` grid; ``values`` is ``(n_t, n_k)``."""

    def __init__(
        self,
        times: ArrayLike,
        k_grid: ArrayLike,
        values: ArrayLike,
        forward_curve: ForwardCurve,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        t = np.asarray(times, dtype=np.float64).ravel()
        k = np.asarray(k_grid, dtype=np.float64).ravel()
        v = np.asarray(values, dtype=np.float64)
        if t.size < 1 or np.any(np.diff(t) <= 0) or t[0] < 0:
            raise ValueError("times must be non-negative and strictly increasing")
        if k.size < 2 or not np.allclose(np.diff(k), k[1] - k[0], rtol=1e-8, atol=1e-14):
            raise ValueError("k_grid must be uniform with at least two points")
        if v.shape != (t.size, k.size):
            raise ValueError(f"values must have shape {(t.size, k.size)}")
        if not np.all(np.isfinite(v)) or np.any(v <= 0):
            raise ValueError("leverage values must be positive and finite")
        self.times = t
        self.k_grid = k
        self.values = np.ascontiguousarray(v)
        self.forward_curve = forward_curve
        self.metadata: dict[str, Any] = dict(metadata or {})

    # -- grid ----------------------------------------------------------------------------------

    @property
    def k0(self) -> float:
        return float(self.k_grid[0])

    @property
    def dk(self) -> float:
        return float(self.k_grid[1] - self.k_grid[0])

    @property
    def horizon(self) -> float:
        return float(self.times[-1])

    @property
    def n_slices(self) -> int:
        return int(self.times.size)

    # -- evaluation ------------------------------------------------------------------------------

    def rows(self, times: ArrayLike) -> FloatArray:
        """``L(t_j, k)`` rows for each ``t_j``: linear in ``t`` between slices, flat outside."""
        t = np.atleast_1d(np.asarray(times, dtype=np.float64))
        tg = self.times
        if tg.size == 1:
            return np.repeat(self.values, t.size, axis=0)
        i = np.clip(np.searchsorted(tg, t, side="right") - 1, 0, tg.size - 2)
        w = np.clip((t - tg[i]) / (tg[i + 1] - tg[i]), 0.0, 1.0)
        out = (1.0 - w)[:, None] * self.values[i] + w[:, None] * self.values[i + 1]
        return np.asarray(out, dtype=np.float64)

    def l2_rows(self, times: ArrayLike) -> FloatArray:
        """``L²`` rows (what the kernels consume)."""
        return np.asarray(self.rows(times) ** 2, dtype=np.float64)

    def step_tables(self, t_nodes: ArrayLike) -> tuple[FloatArray, FloatArray, FloatArray]:
        """``(lev_a, lev_b, lev_rec)`` ``L²`` tables for the steps between ``t_nodes`` under the
        frozen-leverage rule: ``lev_a = lev_b`` = the slice at the step start (every lookup inside
        the step, weak order-2 supporting values included), ``lev_rec`` = the slice at the step
        end (recorded instantaneous variance).  The particle calibration uses the same tables.
        """
        rows = self.l2_rows(t_nodes)
        start = np.ascontiguousarray(rows[:-1])
        return start, start, np.ascontiguousarray(rows[1:])

    def __call__(self, t: ArrayLike, S: ArrayLike) -> FloatArray:
        """``L(t, S)`` with ``k = ln S − ln F(t)``, flat in ``k`` outside the grid."""
        t_, S_ = np.broadcast_arrays(
            np.asarray(t, dtype=np.float64), np.asarray(S, dtype=np.float64)
        )
        k = np.log(S_) - np.asarray(self.forward_curve.log_forward(t_))
        rows = self.rows(t_.ravel())
        kk = k.ravel()
        x = np.clip((kk - self.k0) / self.dk, 0.0, self.k_grid.size - 1.0)
        i = np.minimum(x.astype(np.int64), self.k_grid.size - 2)
        w = x - i
        out = (1.0 - w) * rows[np.arange(kk.size), i] + w * rows[np.arange(kk.size), i + 1]
        return np.asarray(out.reshape(t_.shape), dtype=np.float64)

    def with_values(self, values: ArrayLike, **metadata: Any) -> LeverageFunction:
        return LeverageFunction(
            self.times, self.k_grid, values, self.forward_curve, {**self.metadata, **metadata}
        )

    @staticmethod
    def average(a: LeverageFunction, b: LeverageFunction) -> LeverageFunction:
        """Slice-wise average of two calibrations on the same grid (second-pass smoothing)."""
        if not (np.array_equal(a.times, b.times) and np.array_equal(a.k_grid, b.k_grid)):
            raise ValueError("leverage functions must share the same grid")
        return a.with_values(0.5 * (a.values + b.values), averaged_with=b.metadata.get("seed"))

    # -- serialisation ---------------------------------------------------------------------------

    def save(self, path: str | Path) -> Path:
        """Write ``.npz`` with grid, values, curves and JSON metadata."""
        p = Path(path)
        fc = self.forward_curve
        np.savez_compressed(
            p,
            times=self.times,
            k_grid=self.k_grid,
            values=self.values,
            spot=np.array(fc.spot),
            rate_times=fc.rate_curve.times,
            rate_zeros=fc.rate_curve.zero_rates,
            div_times=fc.dividend_curve.times,
            div_zeros=fc.dividend_curve.zero_rates,
            metadata=np.array(json.dumps(self.metadata, sort_keys=True, default=str)),
        )
        return p if p.suffix == ".npz" else p.with_suffix(p.suffix + ".npz")

    @classmethod
    def load(cls, path: str | Path) -> LeverageFunction:
        with np.load(path, allow_pickle=False) as z:
            fc = ForwardCurve(
                float(z["spot"]),
                DiscountCurve(z["rate_times"], z["rate_zeros"]),
                DiscountCurve(z["div_times"], z["div_zeros"]),
            )
            meta = json.loads(str(z["metadata"]))
            return cls(z["times"], z["k_grid"], z["values"], fc, meta)

    def __repr__(self) -> str:
        return (
            f"LeverageFunction(n_slices={self.n_slices}, horizon={self.horizon:.4g}, "
            f"k∈[{self.k_grid[0]:.3g}, {self.k_grid[-1]:.3g}] x{self.k_grid.size}, "
            f"L∈[{self.values.min():.3g}, {self.values.max():.3g}])"
        )
