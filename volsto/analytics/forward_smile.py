"""Forward-smile analytics (SPEC §7 / viewer 2; book §3.1 "The forward smile").

The forward smile ``σ̂_k^{T1T2}`` of a model is obtained by pricing forward-start options
``(S_T2/S_T1 − k)⁺`` over a moneyness grid and implying a Black volatility through
``P(k) = DF(T2) · Black(F_R, k, τ, σ̂_k)`` with the model's own forward ratio
``F_R = F(T2)/F(T1)`` and ``τ = T2 − T1`` (Bergomi §3.1, p. 104; the book stresses it is an
*average over future smiles*, "invariably more convex" than a spot-starting smile, and not a
market observable).  Out-of-the-money options are used for the inversion (calls for
``k ≥ F_R``, puts below) and the volatility standard error is the price standard error divided
by the Black vega.

:func:`forward_vol_comparison` prices, on the same paths, the three forward volatility
measures compared in the study: the forward ATM-forward implied volatility (a forward-start
straddle's Black volatility, the fair FVA strike), the forward variance-swap volatility
``sqrt(E[RV])`` and the forward volatility-swap volatility ``E[sqrt(RV)]``.  With ``ρ < 0``
one has ATM forward vol < vol-swap vol < variance-swap vol (Jensen for the last inequality;
``tests/test_forward_start.py::test_forward_vol_ordering_negative_correlation``).
:func:`put_wing_table` lays out forward smiles of several models side by side with the spread
across models per strike — the study's put-wing invariance diagnostic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from volsto.engine.mc import MonteCarlo, PriceResult
from volsto.market.bs import black_vega, implied_vol
from volsto.products.forward_start import ForwardStartOption
from volsto.products.variance import VarianceSwap, VolSwap

if TYPE_CHECKING:
    from volsto.config import SimConfig
    from volsto.engine.grid import TimeGrid
    from volsto.engine.rng import GaussianDraws
    from volsto.models.base import Model

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def forward_ratio(model: Model, t1: float, t2: float) -> float:
    """``F(T2)/F(T1) = E[S_T2/S_T1]`` under deterministic rates (the ATM-forward strike)."""
    fc = model.forward_curve
    return float(fc.forward(t2) / fc.forward(t1))


def _check_dates(t1: float, t2: float) -> None:
    if t1 < 0 or t2 <= t1:
        raise ValueError("need 0 ≤ t1 < t2")


@dataclass(frozen=True)
class ForwardSmile:
    """Forward smile ``T1 → T2`` on a moneyness grid (strikes relative to ``S_T1``)."""

    t1: float
    t2: float
    forward_ratio: float
    strikes: FloatArray
    vols: FloatArray
    vol_stderr: FloatArray
    prices: FloatArray
    price_stderr: FloatArray
    cps: IntArray
    n_paths: int

    @property
    def tau(self) -> float:
        return self.t2 - self.t1

    @property
    def log_moneyness(self) -> FloatArray:
        """``ln(k / F_R)``."""
        return np.log(self.strikes / self.forward_ratio)

    def vol_at(self, k: float) -> tuple[float, float]:
        """Volatility and standard error at moneyness ``k`` (linear interpolation in ``k``)."""
        if k < self.strikes[0] - 1e-12 or k > self.strikes[-1] + 1e-12:
            raise ValueError("k outside the smile's strike range")
        return (
            float(np.interp(k, self.strikes, self.vols)),
            float(np.interp(k, self.strikes, self.vol_stderr)),
        )

    @property
    def atm(self) -> tuple[float, float]:
        """Forward ATM-forward volatility (at ``k = F_R``) and its standard error."""
        return self.vol_at(self.forward_ratio)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "k": self.strikes,
                "log_moneyness": self.log_moneyness,
                "cp": self.cps,
                "price": self.prices,
                "price_stderr": self.price_stderr,
                "vol": self.vols,
                "vol_stderr": self.vol_stderr,
            }
        )

    def __repr__(self) -> str:
        v, se = self.atm
        return (
            f"ForwardSmile({self.t1:g}y → {self.t2:g}y, {self.strikes.size} strikes, "
            f"ATMF vol {v:.4%} ± {se:.2%}, n_paths={self.n_paths})"
        )


def _invert(
    results: Sequence[PriceResult],
    strikes: FloatArray,
    cps: IntArray,
    f_r: float,
    tau: float,
    df: float,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    prices = np.array([r.mean for r in results])
    errs = np.array([r.stderr for r in results])
    vols = implied_vol(prices, f_r, strikes, tau, cps, df)
    vega = black_vega(f_r, strikes, tau, np.where(np.isfinite(vols), vols, 0.0), df)
    with np.errstate(divide="ignore", invalid="ignore"):
        vol_se = np.where(vega > 0, errs / vega, np.nan)
    return prices, errs, vols, vol_se


def forward_smile_from_prices(
    t1: float,
    t2: float,
    f_r: float,
    strikes: ArrayLike,
    cps: ArrayLike,
    results: Sequence[PriceResult],
    df: float,
    n_paths: int,
) -> ForwardSmile:
    """Assemble a :class:`ForwardSmile` from already-priced forward-start options (used when a
    study prices the strip together with other products on one path set)."""
    ks = np.asarray(strikes, dtype=np.float64)
    cp = np.asarray(cps, dtype=np.int64)
    if ks.shape != cp.shape or len(results) != ks.size:
        raise ValueError("strikes, cps and results must have the same length")
    prices, errs, vols, vol_se = _invert(results, ks, cp, f_r, t2 - t1, df)
    return ForwardSmile(float(t1), float(t2), f_r, ks, vols, vol_se, prices, errs, cp, n_paths)


def forward_smile(
    model: Model,
    t1: float,
    t2: float,
    strikes: ArrayLike,
    sim: SimConfig,
    *,
    include_atm: bool = True,
    grid: TimeGrid | None = None,
    draws: GaussianDraws | None = None,
) -> ForwardSmile:
    """Forward smile ``T1 → T2`` from out-of-the-money forward-start options on ``strikes``
    (moneyness, relative to ``S_T1``); the ATM-forward strike ``F_R`` is added unless
    ``include_atm=False``.  All options are priced on the same paths.  Strikes whose price is
    outside the Black bounds (deep wings at small ``n_paths``) get ``nan`` volatilities."""
    _check_dates(t1, t2)
    ks = np.unique(np.asarray(strikes, dtype=np.float64))
    if ks.size == 0 or np.any(ks <= 0):
        raise ValueError("strikes must be positive")
    f_r = forward_ratio(model, t1, t2)
    if include_atm and not np.any(np.abs(ks - f_r) < 1e-12):
        ks = np.sort(np.append(ks, f_r))
    cps = np.where(ks >= f_r, 1, -1).astype(np.int64)
    discount = model.forward_curve.rate_curve
    products = [ForwardStartOption(t1, t2, float(k), int(cp), discount) for k, cp in zip(ks, cps)]
    results = MonteCarlo(sim).price_many(products, model, grid=grid, draws=draws)
    return forward_smile_from_prices(
        t1, t2, f_r, ks, cps, results, float(discount.df(t2)), sim.n_paths
    )


def forward_atm_vol(
    model: Model,
    t1: float,
    t2: float,
    sim: SimConfig,
    *,
    grid: TimeGrid | None = None,
    draws: GaussianDraws | None = None,
) -> tuple[float, float]:
    """Forward ATM-forward implied volatility ``T1 → T2`` and its standard error (the fair
    strike of the :class:`~volsto.products.variance.FVA`)."""
    smile = forward_smile(
        model, t1, t2, [forward_ratio(model, t1, t2)], sim, grid=grid, draws=draws
    )
    return float(smile.vols[0]), float(smile.vol_stderr[0])


@dataclass(frozen=True)
class ForwardVolComparison:
    """Forward ATM-forward vol, forward variance-swap vol and forward vol-swap vol, one path
    set; all in volatility units with standard errors (delta method for ``sqrt(E[RV])``)."""

    t1: float
    t2: float
    atm_vol: float
    atm_stderr: float
    vs_vol: float
    vs_stderr: float
    volswap_vol: float
    volswap_stderr: float
    n_paths: int

    def as_dict(self) -> dict[str, float]:
        return {
            "atm_vol": self.atm_vol,
            "atm_stderr": self.atm_stderr,
            "vs_vol": self.vs_vol,
            "vs_stderr": self.vs_stderr,
            "volswap_vol": self.volswap_vol,
            "volswap_stderr": self.volswap_stderr,
        }

    def __repr__(self) -> str:
        return (
            f"ForwardVolComparison({self.t1:g}y → {self.t2:g}y: ATMF {self.atm_vol:.4%} ± "
            f"{self.atm_stderr:.2%}, VS {self.vs_vol:.4%} ± {self.vs_stderr:.2%}, vol swap "
            f"{self.volswap_vol:.4%} ± {self.volswap_stderr:.2%}, n_paths={self.n_paths})"
        )


def forward_vol_comparison(
    model: Model,
    t1: float,
    t2: float,
    sim: SimConfig,
    *,
    per_year: int = 252,
    vs_on_simulation_grid: bool = True,
    grid: TimeGrid | None = None,
    draws: GaussianDraws | None = None,
) -> ForwardVolComparison:
    """The three forward volatility measures on the same paths.

    The variance swap uses the simulation grid's ``Σ(Δ ln S)²`` by default (its fair strike is
    then the continuous-monitoring value up to the scheme's own error, the convention of the
    calibration diagnostics); the vol swap uses daily fixings (``per_year``) as SPEC §6 requires
    realised quantities to be computed on fixing dates.
    """
    _check_dates(t1, t2)
    fc = model.forward_curve
    discount = fc.rate_curve
    f_r = forward_ratio(model, t1, t2)
    tau = t2 - t1
    if vs_on_simulation_grid:
        vs = VarianceSwap([t1, t2], 0.0, discount, use_simulation_grid=True)
    else:
        vs = VarianceSwap.daily(t2, 0.0, discount, start=t1, per_year=per_year)
    products = [
        ForwardStartOption(t1, t2, f_r, 1, discount),
        vs,
        VolSwap.daily(t2, 0.0, discount, start=t1, per_year=per_year),
    ]
    res = MonteCarlo(sim).price_many(products, model, grid=grid, draws=draws)
    df = float(discount.df(t2))
    _, _, vols, vol_se = _invert(res[:1], np.array([f_r]), np.array([1]), f_r, tau, df)
    k_var, se_var = res[1].mean / df, res[1].stderr / df
    vs_vol = float(np.sqrt(k_var))
    return ForwardVolComparison(
        float(t1),
        float(t2),
        float(vols[0]),
        float(vol_se[0]),
        vs_vol,
        float(se_var / (2.0 * vs_vol)),
        res[2].mean / df,
        res[2].stderr / df,
        sim.n_paths,
    )


def put_wing_table(smiles: Mapping[str, ForwardSmile]) -> pd.DataFrame:
    """Forward smiles of several models side by side on their common strikes.

    Columns: one ``vol`` / ``se`` pair per model, ``spread`` (max − min across models) and
    ``spread_z`` (spread over the root-sum-square of the models' standard errors).  Strikes at
    which ``spread_z`` stays small while it is large elsewhere are the model-invariant part of
    the forward smile (the study's put-wing invariance).
    """
    if not smiles:
        raise ValueError("no smiles")
    items = list(smiles.items())
    ks = items[0][1].strikes
    for _, s in items[1:]:
        ks = np.intersect1d(ks, s.strikes)
    if ks.size == 0:
        raise ValueError("smiles share no strike")
    data: dict[str, FloatArray] = {"k": ks}
    vols = []
    ses = []
    for name, s in items:
        sel = np.searchsorted(s.strikes, ks)
        data[f"{name}_vol"] = s.vols[sel]
        data[f"{name}_se"] = s.vol_stderr[sel]
        vols.append(s.vols[sel])
        ses.append(s.vol_stderr[sel])
    v = np.vstack(vols)
    e = np.vstack(ses)
    spread = np.nanmax(v, axis=0) - np.nanmin(v, axis=0)
    data["spread"] = spread
    with np.errstate(divide="ignore", invalid="ignore"):
        data["spread_z"] = spread / np.sqrt(np.nansum(e * e, axis=0))
    return pd.DataFrame(data)
