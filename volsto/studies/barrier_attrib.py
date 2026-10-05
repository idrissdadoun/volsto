"""Addendum 2 of the barrier study: attribution of the hedged P&L, turnover, clustered fits.

Post-processing only — nothing here prices.  The hedged P&L of a position is read as
``vol carry + the rest``: the carry is the entry vega times the realised vol premium of the
trade's life (:func:`life_realised_vol`, :func:`attribute`); the rest (``pnl_hx``) is what the
framework's question bears on.  :func:`turnover` is the trading a daily delta hedge did;
:func:`clustered_ols` and :func:`ratio_interval` are the block-bootstrap statistics over entry
dates the tables use.  :class:`TiltedSurface` is the skew bump of the optional sensitivity.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.market.surface import ImpliedSurface

FloatArray = NDArray[np.float64]

#: Fewest close-to-close returns for a realised vol of the life (addendum 2 §2.1).
MIN_RETURNS: Final = 15
#: Hedging costs, fraction of the notional traded (addendum 2 §4): futures-like, conservative.
COST_RATES: Final[dict[str, float]] = {"0.5bp": 0.5e-4, "2bp": 2.0e-4}


def realised_vol(closes: Any, min_returns: int = MIN_RETURNS) -> float:
    """Annualised close-to-close realised vol of ``closes`` (consecutive official closes):
    ``sqrt(252 × mean of the squared log returns)``, no mean removed.  NaN below
    ``min_returns`` returns."""
    c = np.asarray(closes, dtype=np.float64)
    if c.size < min_returns + 1:
        return float("nan")
    r = np.diff(np.log(c))
    return float(np.sqrt(252.0 * np.mean(r * r)))


def life_realised_vol(
    close: pd.Series, entry: str, expiry: str, min_returns: int = MIN_RETURNS
) -> float:
    """:func:`realised_vol` of the closes from the entry date to the expiry date, both included
    (``close`` indexed by ISO date, ascending).  NaN if the expiry is beyond the history."""
    if expiry > str(close.index[-1]):
        return float("nan")
    return realised_vol(close.loc[entry:expiry].to_numpy(), min_returns)


def vega_column(position: str) -> str:
    """The cell column holding the entry vega of ``position`` (index points per vol point)."""
    if position == "A1":
        return "lv_a1_vega"
    if position == "A2":
        return "lv_a2_vega"
    return f"vega_{position}"


def vol_carry(vega: Any, dsig: Any) -> Any:
    """First-order vol carry of a delta-hedged position: entry vega (per vol point) times the
    realised-minus-implied vol of its life (vol points)."""
    return np.asarray(vega, dtype=np.float64) * np.asarray(dsig, dtype=np.float64)


def attribute(pos: pd.DataFrame, cells: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """One row per row of ``pos`` (``cell, position, model, pnl_h``): ``sig_imp`` (the entry
    at-the-money vol of the cell), ``rv_life``, ``dsig`` (vol points), ``vega`` (fraction of
    spot per vol point), ``carry`` and ``pnl_hx = pnl_h − carry`` (fractions of spot).

    ``cells`` carries ``cell, entry, expiry, K, atm`` and the raw vega columns
    (:func:`vega_column`).  Rows of model ``lsv`` take the local-vol vega of the same
    knock-out (``vega_source`` = ``lv``): no LSV vega is stored."""
    c = cells.set_index("cell")
    life = c[["entry", "expiry"]].drop_duplicates()
    rv = {
        (e, x): life_realised_vol(close, e, x) for e, x in life.itertuples(index=False, name=None)
    }
    out = pos[["cell", "position", "model", "pnl_h"]].copy()
    k = c["K"].reindex(out["cell"]).to_numpy(float)
    out["sig_imp"] = c["atm"].reindex(out["cell"]).to_numpy(float)
    out["rv_life"] = [
        rv[key]
        for key in zip(
            c["entry"].reindex(out["cell"]), c["expiry"].reindex(out["cell"]), strict=True
        )
    ]
    out["dsig"] = 100.0 * (out["rv_life"] - out["sig_imp"])
    vega = np.full(len(out), np.nan)
    for name in out["position"].unique():
        col = vega_column(str(name))
        if col not in c:
            continue
        sel = (out["position"] == name).to_numpy()
        vega[sel] = c[col].reindex(out.loc[sel, "cell"]).to_numpy(float)
    out["vega"] = vega / k
    out["vega_source"] = np.where(out["model"] == "lsv", "lv", out["model"])
    out["carry"] = vol_carry(out["vega"], out["dsig"])
    out["pnl_hx"] = out["pnl_h"] - out["carry"]
    return out


def turnover(delta: Any, weights: Any | None = None) -> float:
    """Trading of a delta hedge held over the snapshots of ``delta`` (entry first):
    ``|δ_0| + Σ|δ_t − δ_{t−1}| + |δ_last|``, the last term the close-out.  With ``weights``
    (one per snapshot, then one for the close-out) each trade is weighted — the forward over
    the entry spot gives the notional traded.  Zero for an empty hedge."""
    d = np.asarray(delta, dtype=np.float64)
    if d.size == 0:
        return 0.0
    trades = np.abs(np.diff(np.concatenate(([0.0], d, [0.0]))))
    if weights is None:
        return float(trades.sum())
    w = np.asarray(weights, dtype=np.float64)
    if w.size != trades.size:
        raise ValueError(f"{trades.size} trades but {w.size} weights")
    return float((trades * w).sum())


def block_indices(n: int, block: int, n_resamples: int, seed: int) -> NDArray[np.int64]:
    """Circular-block-bootstrap index draws, ``(n_resamples, n)``."""
    block = max(1, min(int(block), n))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_resamples, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(n_resamples, -1)[:, :n]


def clustered_ols(
    y: Any, X: Any, groups: Any, block: int, n_resamples: int = 1000, seed: int = 5
) -> dict[str, Any]:
    """Least squares of ``y`` on the columns of ``X`` with a circular block bootstrap over the
    groups (entry dates, in time order; a block is ``block`` consecutive groups, all the rows
    of a drawn group are kept).  Returns ``coef``, ``se``, ``r2``, ``n`` (rows) and
    ``n_groups``.  A column with no variation gets a NaN coefficient and is left out of the
    fit."""
    y = np.asarray(y, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    g = np.asarray(groups)
    ok = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y, X, g = y[ok], X[ok], g[ok]
    p = X.shape[1]
    nan = np.full(p, np.nan)
    if y.size <= p + 1:
        return {"coef": nan, "se": nan.copy(), "r2": float("nan"), "n": int(y.size), "n_groups": 0}
    # a column is kept when it varies, or when it is the (first) constant
    spread = X.max(axis=0) - X.min(axis=0)
    keep = spread > 0
    const = np.flatnonzero(~keep & (np.abs(X).max(axis=0) > 0))
    if const.size:
        keep[const[0]] = True
    Xk = X[:, keep]
    names, inv = np.unique(g, return_inverse=True)
    q = Xk.shape[1]
    xtx = np.zeros((names.size, q, q))
    xty = np.zeros((names.size, q))
    np.add.at(xtx, inv, Xk[:, :, None] * Xk[:, None, :])
    np.add.at(xty, inv, Xk * y[:, None])
    beta = np.linalg.lstsq(xtx.sum(axis=0), xty.sum(axis=0), rcond=None)[0]
    res = y - Xk @ beta
    var = float(y.var())
    r2 = 1.0 - float(res.var()) / var if var > 0 else float("nan")
    idx = block_indices(names.size, block, n_resamples, seed)
    a = xtx[idx].sum(axis=1)
    b = xty[idx].sum(axis=1)
    draws = np.full((n_resamples, q), np.nan)
    for i in range(n_resamples):
        try:
            draws[i] = np.linalg.solve(a[i], b[i])
        except np.linalg.LinAlgError:
            continue
    coef, se = nan.copy(), nan.copy()
    coef[keep] = beta
    se[keep] = np.nanstd(draws, axis=0, ddof=1)
    return {"coef": coef, "se": se, "r2": r2, "n": int(y.size), "n_groups": int(names.size)}


def ratio_interval(
    num: Any,
    den: Any,
    block: int,
    *,
    floor: float,
    n_resamples: int = 2000,
    seed: int = 9,
    min_obs: int = 30,
) -> tuple[float, float, float, float, float]:
    """``mean(num) / mean(den)`` over time-ordered observations (one per entry date) with its
    95 % circular-block-bootstrap interval: ``(ratio, low, high, mean num, mean den)``.  A
    denominator below ``floor`` in absolute value gives NaN (for the estimate and for the draws
    concerned); fewer than ``min_obs`` observations give NaN throughout."""
    arr = np.column_stack([np.asarray(num, dtype=np.float64), np.asarray(den, dtype=np.float64)])
    arr = arr[np.isfinite(arr).all(axis=1)]
    if len(arr) < min_obs:
        return (float("nan"),) * 5
    m_num, m_den = float(arr[:, 0].mean()), float(arr[:, 1].mean())
    ratio = m_num / m_den if abs(m_den) >= floor else float("nan")
    mu = arr[block_indices(len(arr), block, n_resamples, seed)].mean(axis=1)
    draws = np.where(np.abs(mu[:, 1]) >= floor, mu[:, 0] / mu[:, 1], np.nan)
    if np.isfinite(draws).sum() < 0.5 * n_resamples:
        return ratio, float("nan"), float("nan"), m_num, m_den
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return ratio, float(lo), float(hi), m_num, m_den


class TiltedSurface(ImpliedSurface):
    """``base`` with a skew tilt (addendum 2 §6): the implied vol at log-moneyness ``k`` (against
    the forward of the same maturity) moves by ``size × k / |k_barrier|`` — ``size`` at the
    barrier's log-moneyness on the call side, minus ``size`` at the mirror strike, nothing at
    the money."""

    def __init__(self, base: ImpliedSurface, k_barrier: float, size: float = 0.01) -> None:
        if k_barrier == 0.0:
            raise ValueError("the tilt needs a barrier away from the forward")
        super().__init__(base.forward_curve, base.discount, base.max_maturity)
        self.base, self.slope = base, float(size) / abs(float(k_barrier))

    def total_variance(self, k: Any, T: Any) -> Any:
        k = np.asarray(k, dtype=np.float64)
        T = np.asarray(T, dtype=np.float64)
        w = np.asarray(self.base.total_variance(k, T), dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            vol = np.sqrt(np.where(T > 0, w / np.where(T > 0, T, 1.0), 0.0))
        vol = np.maximum(vol + self.slope * k, 1e-4)
        return vol * vol * T
