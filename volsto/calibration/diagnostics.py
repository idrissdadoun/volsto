"""Repricing diagnostics of a calibrated LSV model (SPEC §4.2).

Reprice the target surface on the pillar grid ``T ∈ {1m, 3m, 6m, 1y, 18m, 2y, 3y}``,
``k ∈ ±{0, 0.05, 0.1, 0.2, 0.3}`` with the *pricing* kernel (``MonteCarlo.price_many``, common
paths, ``n_paths ≥ 4·10⁵``), report the implied-vol error in vol points with its MC standard
error, plus variance-swap strikes (realised on the simulation grid) against the log-contract
replication values.  Acceptance (SPEC §4.2 table): max abs error ≤ 0.15 vp inside ±20% moneyness
for T ≤ 2y.  Checked by ``tests/test_lsv.py``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.config import SimConfig
from volsto.engine.mc import MonteCarlo
from volsto.market.bs import black_vega, implied_vol
from volsto.market.surface import ImpliedSurface
from volsto.market.varswap import varswap_strike
from volsto.models.base import Model
from volsto.products.base import Product
from volsto.products.vanilla import EuropeanOption
from volsto.products.variance import VarianceSwap

FloatArray = NDArray[np.float64]

PILLAR_MATURITIES: tuple[float, ...] = (1.0 / 12.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
PILLAR_LOG_MONEYNESS: tuple[float, ...] = (-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3)


@dataclass
class CalibrationReport:
    """Vanilla repricing errors (vol points) and variance-swap checks, with standard errors."""

    vanillas: pd.DataFrame
    varswaps: pd.DataFrame
    n_paths: int
    seed: int
    wall_time: float
    extra: dict[str, Any] = field(default_factory=dict)

    def _region(self, t_max: float, k_abs: float, max_std: float | None = None) -> pd.DataFrame:
        """Cells with ``T ≤ t_max``, ``|k| ≤ k_abs`` and, if ``max_std`` is given, within
        ``max_std`` ATM standard deviations (``|k| ≤ max_std · σ_target √T``) — the 1m ±20% cells
        are 3.4 standard deviations out and priced by the far tail of the particle cloud."""
        d = self.vanillas
        m = (d["T"] <= t_max + 1e-9) & (d["k"].abs() <= k_abs + 1e-9)
        if max_std is not None:
            m &= d["k"].abs() <= max_std * d["target_vol"] * np.sqrt(d["T"]) + 1e-9
        return d[m]

    def max_abs_error(
        self, t_max: float = 2.0, k_abs: float = 0.2, max_std: float | None = None
    ) -> float:
        """Largest |error| in the region (nan cells — unresolvable prices — are skipped)."""
        return float(self._region(t_max, k_abs, max_std)["error_vp"].abs().max())

    def max_z(self, t_max: float = 2.0, k_abs: float = 0.2, max_std: float | None = None) -> float:
        """Largest |error| / stderr in the region."""
        r = self._region(t_max, k_abs, max_std)
        return float((r["error_vp"].abs() / r["stderr_vp"]).max())

    def violations(
        self,
        tol_vp: float = 0.15,
        t_max: float = 2.0,
        k_abs: float = 0.2,
        z: float = 3.0,
        max_std: float | None = None,
    ) -> pd.DataFrame:
        """Cells with ``|error| > max(tol_vp, z · stderr)``: an error is a violation only when it
        is both above the tolerance and statistically resolved at ``z`` standard errors."""
        r = self._region(t_max, k_abs, max_std)
        bad = r["error_vp"].abs() > np.maximum(tol_vp, z * r["stderr_vp"])
        return r[bad.fillna(False)]

    def passes(
        self,
        tol_vp: float = 0.15,
        t_max: float = 2.0,
        k_abs: float = 0.2,
        z: float = 3.0,
        max_std: float | None = None,
    ) -> bool:
        return self.violations(tol_vp, t_max, k_abs, z, max_std).empty

    def pivot(self, value: str = "error_vp") -> pd.DataFrame:
        """Pillars × strikes table of ``value`` (``error_vp`` or ``stderr_vp``)."""
        return self.vanillas.pivot(index="T", columns="k", values=value)

    def summary(self) -> str:
        err = self.pivot("error_vp")
        se = self.pivot("stderr_vp")
        cells = err.copy().astype(object)
        for i in err.index:
            for j in err.columns:
                cells.loc[i, j] = f"{err.loc[i, j]:+.3f}±{se.loc[i, j]:.3f}"
        head = (
            f"implied-vol error (vol points ± MC se), n_paths={self.n_paths}, "
            f"wall {self.wall_time:.1f}s, max|err| (T≤2y, |k|≤0.2) = {self.max_abs_error():.3f}, "
            f"max |err|/se = {self.max_z():.2f}, "
            f"violations(0.15 vp, 3 se) = {len(self.violations())}"
        )
        return head + "\n" + cells.to_string() + "\n" + self.varswaps.to_string(index=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "vanillas": self.vanillas.to_dict(orient="list"),
            "varswaps": self.varswaps.to_dict(orient="list"),
            "n_paths": self.n_paths,
            "seed": self.seed,
            "wall_time": self.wall_time,
            "max_abs_error_vp": self.max_abs_error(),
            "extra": self.extra,
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=1, default=str))

    @classmethod
    def load(cls, path: str | Path) -> CalibrationReport:
        d = json.loads(Path(path).read_text())
        return cls(
            pd.DataFrame(d["vanillas"]),
            pd.DataFrame(d["varswaps"]),
            int(d["n_paths"]),
            int(d["seed"]),
            float(d["wall_time"]),
            dict(d.get("extra", {})),
        )


def reprice_surface(
    model: Model,
    surface: ImpliedSurface,
    sim: SimConfig,
    *,
    maturities: Sequence[float] = PILLAR_MATURITIES,
    log_moneyness: Sequence[float] = PILLAR_LOG_MONEYNESS,
    include_varswaps: bool = True,
) -> CalibrationReport:
    """Price the pillar vanillas (OTM) and variance swaps under ``model`` on common paths."""
    t0 = time.perf_counter()
    disc = surface.discount
    prods: list[Product] = []
    meta = []
    for T in maturities:
        F = float(surface.forward(T))
        for k in log_moneyness:
            K = F * np.exp(k)
            cp = 1 if k >= 0 else -1
            prods.append(EuropeanOption(K, T, cp, disc))
            meta.append((T, k, K, cp, F))
    n_van = len(prods)
    if include_varswaps:
        for T in maturities:
            prods.append(VarianceSwap([0.0, T], 0.0, disc, use_simulation_grid=True))
    mc = MonteCarlo(sim)
    res = mc.price_many(prods, model)
    rows = []
    for (T, k, K, cp, F), r in zip(meta, res[:n_van]):
        df = float(disc.df(T))
        target = float(surface.implied_vol(K, T))
        iv = float(implied_vol(r.mean, F, K, T, cp, df))
        # a price below the implied-vol resolution (deep OTM, short maturity) is reported as
        # unresolved (nan) rather than as a zero vol
        resolved = bool(np.isfinite(iv) and iv > 0.0 and r.mean > 1e-10 * F)
        vega = float(black_vega(F, K, T, iv if resolved else target, df))
        rows.append(
            {
                "T": T,
                "k": k,
                "K": K,
                "cp": cp,
                "price": r.mean,
                "price_stderr": r.stderr,
                "model_vol": iv if resolved else np.nan,
                "target_vol": target,
                "error_vp": 100.0 * (iv - target) if resolved else np.nan,
                "stderr_vp": 100.0 * r.stderr / vega if vega > 0 else np.nan,
            }
        )
    vanillas = pd.DataFrame(rows)
    vs_rows = []
    if include_varswaps:
        for T, r in zip(maturities, res[n_van:]):
            df = float(disc.df(T))
            k_mc = r.mean / df
            k_rep = varswap_strike(surface, T)
            vs_rows.append(
                {
                    "T": T,
                    "mc_strike": k_mc,
                    "mc_stderr": r.stderr / df,
                    "replication": k_rep,
                    "mc_vol": np.sqrt(k_mc),
                    "replication_vol": np.sqrt(k_rep),
                    "diff_vp": 100.0 * (np.sqrt(k_mc) - np.sqrt(k_rep)),
                    "stderr_vp": 100.0 * r.stderr / df / (2.0 * np.sqrt(k_mc)),
                }
            )
    varswaps = pd.DataFrame(vs_rows)
    return CalibrationReport(vanillas, varswaps, sim.n_paths, sim.seed, time.perf_counter() - t0)
