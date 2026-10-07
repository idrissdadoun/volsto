"""Statistical (real-world) metrics from the cached daily histories (``data/history/``,
written by ``scripts/fetch_history.py``): the layer of the barrier-versus-vanilla and
dispersion frameworks that no pricing model supplies.

**Barrier path statistics** (:func:`barrier_path_statistics`).  For every entry date and a
horizon of ``n`` trading days, the realised path ``S_{t+j}/S_t``: whether the daily closes
touched the barrier (strict, the library's convention), the terminal level, and the undiscounted
payoffs per unit of the initial spot of the structures the study compares — up-and-out call
(daily), its European knock-out, the call spread, the call fly (``K, mid, B``), the 1×2 call
ratio (``K, mid``) — or their put mirrors for a down barrier.  The *regret* indicator is
``touched and K < S_T < B``: the paths on which the barrier threw away a payoff the vanilla
structure keeps.  Overlapping windows: the effective sample is about ``n_dates / n``, and the
standard errors here use that count (:func:`frequency_table`, block standard errors).

**Filtered historical simulation** (:func:`filtered_historical_paths`).  Daily returns
re-scaled by the ratio of a target volatility to their trailing EWMA volatility
(``λ = 0.94``), resampled in blocks of ``n`` consecutive days: the path shapes of history
(clustering, trending, gaps) at today's vol level — a model of paths with no distributional
assumption, the P-measure counterpart of the Q-measure barrier prices.

**Dispersion statistics** (:func:`dispersion_statistics`).  For a basket of names and a horizon:
the realised performances ``r_i``, the basket's ``r_B``, the dispersion ``D = Σ w_i |r_i −
r_B|``, the straddle package ``Σ w_i |r_i| − |r_B|``, the window's realised pairwise mean
correlation, mean single-name vol and basket vol — the historical joint distribution the
dispersion framework calibrates its expectations on, joined to the Cboe implied-correlation
index when the file is there.

Nothing here prices: no model, no surface.  Checked by ``tests/test_history_stats.py``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[2]
HISTORY_DIR = ROOT / "data" / "history"
TRADING_DAYS = 252
EWMA_LAMBDA = 0.94


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------


def load_series(name: str, directory: Path = HISTORY_DIR) -> pd.Series:
    """``close`` of ``<directory>/<name>.csv`` indexed by date."""
    path = directory / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run scripts/fetch_history.py")
    df = pd.read_csv(path, parse_dates=["date"])
    s = df.set_index("date")["close"].astype(float).sort_index()
    s.name = name
    return s


def load_closes(names: Sequence[str], directory: Path = HISTORY_DIR) -> pd.DataFrame:
    """The named series aligned on their common dates (inner join)."""
    frames = [load_series(n, directory) for n in names]
    return pd.concat(frames, axis=1, join="inner").dropna()


def history_manifest(directory: Path = HISTORY_DIR) -> dict[str, Any]:
    path = directory / "manifest.json"
    return dict(json.loads(path.read_text())) if path.exists() else {}


def log_returns(closes: pd.Series) -> pd.Series:
    """Daily log returns of a close series."""
    x = pd.Series(np.log(closes.to_numpy(dtype=np.float64)), index=closes.index)
    return x.diff().dropna()


def realised_vol(closes: pd.Series, window: int) -> pd.Series:
    """Trailing annualised close-to-close vol over ``window`` days (sample std, zero mean)."""
    r = pd.Series(np.log(closes.to_numpy(dtype=np.float64)), index=closes.index).diff()
    out = np.sqrt((r * r).rolling(window).mean() * TRADING_DAYS)
    return pd.Series(out, index=closes.index)


def ewma_vol(returns: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """RiskMetrics EWMA annualised vol of daily log returns (``σ²_t = λ σ²_{t−1} + (1−λ) r²_t``,
    seeded with the first 60 days' variance)."""
    r = returns.to_numpy(dtype=np.float64)
    out = np.empty_like(r)
    v = float(np.mean(r[:60] ** 2)) if r.size >= 60 else float(np.mean(r * r))
    for i, x in enumerate(r):
        v = lam * v + (1.0 - lam) * x * x
        out[i] = v
    return pd.Series(np.sqrt(out * TRADING_DAYS), index=returns.index)


# --------------------------------------------------------------------------------------------
# barrier path statistics
# --------------------------------------------------------------------------------------------


def structure_payoffs(
    terminal: FloatArray,
    touched: NDArray[np.bool_],
    *,
    strike: float,
    barrier: float,
    direction: str,
    mid: float | None = None,
) -> dict[str, FloatArray]:
    """Undiscounted payoffs per unit of the initial spot, in moneyness terms (``terminal = S_T /
    S_0``): for ``direction="up"`` the call family on ``(K, B)`` with ``K < B``; for ``"down"``
    the put family on ``(H, K)`` with ``H < K`` (``barrier = H``).  ``mid`` defaults to the
    midpoint of strike and barrier."""
    s = np.asarray(terminal, dtype=np.float64)
    k, b = float(strike), float(barrier)
    m = 0.5 * (k + b) if mid is None else float(mid)
    out: dict[str, FloatArray] = {}
    if direction == "up":
        if not k < m < b:
            raise ValueError("need strike < mid < barrier for an up barrier")
        vanilla = np.maximum(s - k, 0.0)
        alive_T = s < b
        out["eko"] = vanilla * alive_T
        out["uoc"] = out["eko"] * (~touched)
        out["spread"] = vanilla - np.maximum(s - b, 0.0)
        out["fly"] = (
            vanilla
            - (b - k) / (b - m) * np.maximum(s - m, 0.0)
            + (m - k) / (b - m) * np.maximum(s - b, 0.0)
        )
        out["ratio"] = vanilla - 2.0 * np.maximum(s - m, 0.0)
        out["regret"] = (touched & alive_T & (s > k)).astype(np.float64)
    elif direction == "down":
        if not b < m < k:
            raise ValueError("need barrier < mid < strike for a down barrier")
        vanilla = np.maximum(k - s, 0.0)
        alive_T = s > b
        out["eko"] = vanilla * alive_T
        out["uoc"] = out["eko"] * (~touched)
        out["spread"] = vanilla - np.maximum(b - s, 0.0)
        out["fly"] = (
            vanilla
            - (k - b) / (m - b) * np.maximum(m - s, 0.0)
            + (k - m) / (m - b) * np.maximum(b - s, 0.0)
        )
        out["ratio"] = vanilla - 2.0 * np.maximum(m - s, 0.0)
        out["regret"] = (touched & alive_T & (s < k)).astype(np.float64)
    else:
        raise ValueError("direction must be 'up' or 'down'")
    out["touched"] = touched.astype(np.float64)
    return out


def path_windows(closes: pd.Series, horizon: int) -> tuple[pd.DatetimeIndex, FloatArray]:
    """Every window of ``horizon`` trading days as ``S_{t+j}/S_t``, ``j = 0..horizon``:
    ``(n_dates, horizon + 1)`` with the entry dates."""
    s = closes.to_numpy(dtype=np.float64)
    n = s.size - horizon
    if n <= 0:
        raise ValueError("the history is shorter than the horizon")
    idx = np.arange(n)[:, None] + np.arange(horizon + 1)[None, :]
    ratios = s[idx] / s[:n, None]
    return pd.DatetimeIndex(closes.index[:n]), np.asarray(ratios, dtype=np.float64)


def touched_strict(ratios: FloatArray, barrier: float, direction: str) -> NDArray[np.bool_]:
    """Strict daily-close breach on the monitoring dates ``j = 1..horizon`` (the entry close
    itself is at the spot and cannot breach; the library's ``strict=True`` knocks on ``>`` /
    ``<``)."""
    inner = ratios[:, 1:]
    if direction == "up":
        return np.asarray(np.any(inner > barrier, axis=1))
    return np.asarray(np.any(inner < barrier, axis=1))


def barrier_path_statistics(
    closes: pd.Series,
    horizon: int,
    *,
    barrier: float,
    direction: str,
    strike: float = 1.0,
    mid: float | None = None,
) -> pd.DataFrame:
    """Per entry date (module docstring): ``terminal``, ``extreme`` (max for up, min for down),
    ``touched``, ``regret`` and the structures' payoffs."""
    dates, ratios = path_windows(closes, horizon)
    touched = touched_strict(ratios, barrier, direction)
    pay = structure_payoffs(
        ratios[:, -1], touched, strike=strike, barrier=barrier, direction=direction, mid=mid
    )
    extreme = ratios[:, 1:].max(axis=1) if direction == "up" else ratios[:, 1:].min(axis=1)
    out = pd.DataFrame({"terminal": ratios[:, -1], "extreme": extreme, **pay}, index=dates)
    out.index.name = "date"
    return out


def frequency_table(
    stats: pd.DataFrame,
    horizon: int,
    columns: Sequence[str] = ("touched", "regret", "uoc", "eko", "spread", "fly", "ratio"),
    groups: pd.Series | None = None,
) -> pd.DataFrame:
    """Means of the columns with block standard errors (``std / sqrt(n / horizon)``: the windows
    overlap, about ``n / horizon`` are independent), overall or per group of ``groups``
    (aligned on the index)."""
    rows = []
    frame = stats if groups is None else stats.join(groups.rename("_group"), how="inner")
    keys = [None] if groups is None else list(pd.unique(frame["_group"].dropna()))
    for key in keys:
        sub = frame if key is None else frame[frame["_group"] == key]
        n = len(sub)
        n_eff = max(n / horizon, 1.0)
        row: dict[str, Any] = {"group": "all" if key is None else key, "n": n, "n_eff": n_eff}
        for c in columns:
            x = sub[c].to_numpy(dtype=np.float64)
            row[c] = float(np.mean(x)) if n else np.nan
            row[f"{c}_stderr"] = float(np.std(x, ddof=1) / np.sqrt(n_eff)) if n > 1 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# filtered historical simulation
# --------------------------------------------------------------------------------------------


def filtered_historical_paths(
    returns: pd.Series,
    horizon: int,
    n_paths: int,
    *,
    vol_target: float,
    seed: int = 0,
    lam: float = EWMA_LAMBDA,
    scale: bool = True,
    demean: bool = True,
) -> FloatArray:
    """``(n_paths, horizon + 1)`` ratios ``S_j / S_0`` from blocks of ``horizon`` consecutive
    daily log returns.  With ``scale`` each day's return is standardised by the trailing EWMA
    vol known before the day, ``z_t = r_t / σ_EWMA(t−1)``, the standardised series is rescaled
    to unit variance (the EWMA lags, so the raw ``z`` has a variance above one on real data)
    and multiplied by ``vol_target / √252``; without it the raw returns are used (a plain block
    bootstrap).  With ``demean`` the sample mean of the (scaled) returns is removed: the
    statistical layer then differs from the risk-neutral one by path shape and vol only, not
    by the equity drift (the drift-included variant is the owner's bull view, reported apart).
    Block starts are uniform over the history (``seed``)."""
    r = returns.to_numpy(dtype=np.float64)
    if scale:
        v = ewma_vol(returns, lam).to_numpy(dtype=np.float64) / np.sqrt(TRADING_DAYS)
        v_prev = np.concatenate(([v[0]], v[:-1]))
        z = r / np.maximum(v_prev, 1e-8)
        z = z / float(np.std(z, ddof=1))
        r = z * float(vol_target) / np.sqrt(TRADING_DAYS)
    if demean:
        r = r - float(np.mean(r))
    n = r.size - horizon
    if n <= 0:
        raise ValueError("the history is shorter than the horizon")
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n + 1, size=n_paths)
    idx = starts[:, None] + np.arange(horizon)[None, :]
    cum = np.cumsum(r[idx], axis=1)
    return np.asarray(np.exp(np.column_stack((np.zeros(n_paths), cum))), dtype=np.float64)


def fhs_barrier_table(
    paths: FloatArray,
    *,
    barrier: float,
    direction: str,
    strike: float = 1.0,
    mid: float | None = None,
) -> dict[str, tuple[float, float]]:
    """Mean and standard error of the structures' payoffs and of the touch / regret
    frequencies on simulated paths (independent samples)."""
    touched = touched_strict(paths, barrier, direction)
    pay = structure_payoffs(
        paths[:, -1], touched, strike=strike, barrier=barrier, direction=direction, mid=mid
    )
    n = paths.shape[0]
    return {k: (float(np.mean(v)), float(np.std(v, ddof=1) / np.sqrt(n))) for k, v in pay.items()}


# --------------------------------------------------------------------------------------------
# dispersion statistics
# --------------------------------------------------------------------------------------------


def dispersion_statistics(
    closes: pd.DataFrame,
    horizon: int,
    weights: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Per entry date (module docstring): ``dispersion`` ``D``, ``straddle_package``
    ``Σ w|r_i| − |r_B|``, ``basket_abs`` ``|r_B|``, ``singles_abs`` ``Σ w|r_i|``, ``r_basket``,
    ``corr_realised`` (window pairwise mean correlation of daily log returns), ``vol_mean``,
    ``vol_basket`` (annualised), ``corr_implied_realised`` (``σ_B²`` against the names' vols —
    the realised analogue of the index-implied correlation)."""
    p = closes.to_numpy(dtype=np.float64)
    n_dates, n_names = p.shape
    w = np.ones(n_names) / n_names if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (n_names,):
        raise ValueError("one weight per name")
    n = n_dates - horizon
    if n <= 0:
        raise ValueError("the history is shorter than the horizon")
    perf = p[horizon:] / p[:n] - 1.0  # (n, names): the window's performances
    rb = perf @ w
    disp = np.abs(perf - rb[:, None]) @ w
    singles = np.abs(perf) @ w
    lr = np.diff(np.log(p), axis=0)  # daily log returns (n_dates − 1, names)
    basket_level = (p / p[0]) @ w
    corr = np.empty(n)
    vol_mean = np.empty(n)
    vol_basket = np.empty(n)
    corr_ir = np.empty(n)
    iu = np.triu_indices(n_names, 1)
    for t in range(n):
        block = lr[t : t + horizon]
        c = np.corrcoef(block.T)
        corr[t] = float(np.mean(c[iu]))
        vols = block.std(axis=0, ddof=1) * np.sqrt(TRADING_DAYS)
        vol_mean[t] = float(np.sum(w * vols))
        lb = np.diff(np.log(basket_level[t : t + horizon + 1]))
        vb = float(lb.std(ddof=1) * np.sqrt(TRADING_DAYS))
        vol_basket[t] = vb
        own = float(np.sum(w * w * vols * vols))
        cross = float(np.sum(np.outer(w * vols, w * vols))) - own
        corr_ir[t] = (vb * vb - own) / cross if cross > 0 else np.nan
    out = pd.DataFrame(
        {
            "r_basket": rb,
            "dispersion": disp,
            "singles_abs": singles,
            "basket_abs": np.abs(rb),
            "straddle_package": singles - np.abs(rb),
            "corr_realised": corr,
            "corr_implied_realised": corr_ir,
            "vol_mean": vol_mean,
            "vol_basket": vol_basket,
        },
        index=pd.DatetimeIndex(closes.index[:n], name="date"),
    )
    return out


def regime_bins(series: pd.Series, edges: Sequence[float], labels: Sequence[str]) -> pd.Series:
    """``pd.cut`` of a regime variable into named bins (aligned with the series' index)."""
    return pd.Series(
        pd.cut(series, bins=list(edges), labels=list(labels), include_lowest=True).astype(str),
        index=series.index,
    )


__all__ = [
    "EWMA_LAMBDA",
    "HISTORY_DIR",
    "TRADING_DAYS",
    "barrier_path_statistics",
    "dispersion_statistics",
    "ewma_vol",
    "fhs_barrier_table",
    "filtered_historical_paths",
    "frequency_table",
    "history_manifest",
    "load_closes",
    "load_series",
    "log_returns",
    "path_windows",
    "realised_vol",
    "regime_bins",
    "structure_payoffs",
    "touched_strict",
]
