"""Dispersion study: indicators and Gaussian forecasts at entry (spec §6, §7.1).

Every function takes the daily returns **strictly before** the entry date (rows in time
order, the most recent last) and the weights of the basket frozen at entry; nothing here sees
the entry date or after (:func:`trailing` is the only slicer, and the unit test of check C10
feeds it a series whose values change from the entry date on).  Forecast weights are fixed a
priori: ``σ²_fc = 0.5·RV²_1m + 0.3·RV²_3m + 0.2·RV²_12m``.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

BLEND: Final = ((21, 0.5), (63, 0.3), (252, 0.2))
#: Earnings seasons (month, day) → (month, day), both included.
SEASONS: Final = (((1, 15), (2, 15)), ((4, 15), (5, 15)), ((7, 15), (8, 15)), ((10, 15), (11, 15)))


def trailing(returns: pd.DataFrame, entry: str, names: list[str], n_rows: int) -> FloatArray:
    """The last ``n_rows`` daily returns of ``names`` strictly before ``entry`` (fewer if the
    panel starts later): ``rows × names``, NaN where a name has no return."""
    i = int(np.searchsorted(returns.index.to_numpy(str), entry, side="left"))
    cols = [c if c in returns.columns else None for c in names]
    block = returns.iloc[max(0, i - n_rows) : i]
    out = np.full((len(block), len(names)), np.nan)
    for j, c in enumerate(cols):
        if c is not None:
            out[:, j] = block[c].to_numpy(float)
    return out


def usable(
    hist: FloatArray, w: FloatArray, n_rows: int, min_share: float = 0.9
) -> tuple[FloatArray, FloatArray]:
    """The last ``n_rows`` rows of ``hist`` restricted to the names with at least
    ``min_share`` of them observed, missing returns set to zero, and the weights renormalised
    over those names (a name with a short history is dropped from the statistic)."""
    block = hist[-n_rows:]
    if block.shape[0] < n_rows:
        return np.empty((0, 0)), np.empty(0)
    ok = np.isfinite(block).mean(axis=0) >= min_share
    if ok.sum() < 2:
        return np.empty((0, 0)), np.empty(0)
    ww = w[ok] / w[ok].sum()
    return np.nan_to_num(block[:, ok]), ww


def realised_vol(r: FloatArray, n: int) -> FloatArray:
    """Annualised realised vol of the last ``n`` rows (columns are series)."""
    return np.sqrt(252.0 * np.mean(r[-n:] ** 2, axis=0))


def forecast_vol(r: FloatArray) -> FloatArray:
    """``σ_fc`` of each column from its last 252 rows (the fixed blend)."""
    return np.sqrt(sum(wt * realised_vol(r, n) ** 2 for n, wt in BLEND))


def cboe(basket_vol: float, vols: FloatArray, w: FloatArray) -> float:
    own = float(np.sum((w * vols) ** 2))
    cross = float(np.sum(w * vols)) ** 2 - own
    return (basket_vol**2 - own) / cross if cross > 0 else float("nan")


def variance_ratio(z: FloatArray, w: FloatArray, h: int) -> tuple[float, float]:
    """``(VR_cs, VR_raw)`` of the relative returns ``z`` (``L × n``) at horizon ``h`` (spec §6,
    Q3): overlapping ``h``-day sums, the demeaned one with the Lo–MacKinlay small-sample
    factors, the raw one without."""
    L = z.shape[0]
    if 2 * h >= L:
        return float("nan"), float("nan")
    zt = z - z.mean(axis=0)

    def sums(x: FloatArray) -> FloatArray:
        c = np.concatenate([np.zeros((1, x.shape[1])), np.cumsum(x, axis=0)])
        return np.asarray(c[h:] - c[:-h], dtype=np.float64)

    num_cs = float(np.sum(sums(zt) ** 2 @ w)) / ((L - h + 1) * (1.0 - h / L))
    den_cs = h * float(np.sum(zt**2 @ w)) / (L - 1)
    num_raw = float(np.sum(sums(z) ** 2 @ w)) / (L - h + 1)
    den_raw = h * float(np.sum(z**2 @ w)) / L
    return num_cs / den_cs, num_raw / den_raw


def beta_dispersion(r: FloatArray, rb: FloatArray, w: FloatArray) -> float:
    """``Σ w_i |β_i − β̄|`` with betas on the basket over the rows given."""
    var = float(np.var(rb))
    if var <= 0:
        return float("nan")
    beta = ((r - r.mean(axis=0)) * (rb - rb.mean())[:, None]).mean(axis=0) / var
    return float(np.sum(w * np.abs(beta - np.sum(w * beta))))


def csad_slopes(r: FloatArray, rb: FloatArray, w: FloatArray) -> tuple[float, float]:
    """``(γ_1, γ_2)`` of ``CSAD_t = α + γ_1|r_B| + γ_2 r_B²``."""
    csad = np.abs(r - rb[:, None]) @ w
    A = np.column_stack([np.ones(rb.size), np.abs(rb), rb * rb])
    g = np.linalg.lstsq(A, csad, rcond=None)[0]
    return float(g[1]), float(g[2])


def variability(z: FloatArray, w: FloatArray, block: int = 21) -> float:
    """Coefficient of variation of the realised cross-sectional vol over the non-overlapping
    ``block``-day blocks of ``z``."""
    n = z.shape[0] // block
    if n < 6:
        return float("nan")
    csv = (z[-n * block :] ** 2 @ w).reshape(n, block).sum(axis=1)
    vol = np.sqrt(csv)
    return float(vol.std(ddof=1) / vol.mean())


def earnings_share(entry: str, expiry: str) -> float:
    """Share of the calendar days of ``[entry, expiry]`` inside the earnings seasons."""
    days = pd.date_range(entry, expiry, freq="D")
    md = days.month * 100 + days.day
    inside = np.zeros(len(days), dtype=bool)
    for (m0, d0), (m1, d1) in SEASONS:
        inside |= (md >= m0 * 100 + d0) & (md <= m1 * 100 + d1)
    return float(inside.mean())


def gaussian_forecast(
    sig: FloatArray, sig_b: float, sig_rel: FloatArray, w: FloatArray, T: float, vr: float
) -> dict[str, float]:
    """Expected payoffs under the two Gaussian variants of spec §7.1: G0 (no persistence) and
    G1 (persistence ``vr``; G1 falls back to G0 where ``vr`` is not available)."""
    c = np.sqrt(2.0 * T / np.pi)
    out: dict[str, float] = {}
    for tag, v in (("G0", 1.0), ("G1", vr if np.isfinite(vr) and vr > 0 else 1.0)):
        e_d = c * float(np.sum(w * sig_rel)) * np.sqrt(v)
        e_abs = c * float(np.sum(w * np.sqrt(sig**2 + (v - 1.0) * sig_rel**2)))
        e_b = c * sig_b
        out[f"ED_{tag}"] = e_d
        out[f"EabsR_{tag}"] = e_abs
        out[f"EabsRb_{tag}"] = e_b
        out[f"EV_{tag}"] = T * float(np.sum(w * sig_rel**2)) * v
        out[f"EG_{tag}"] = e_d - e_abs + e_b
    return out


def indicators(hist: FloatArray, w: FloatArray, h: int, T: float) -> dict[str, float]:
    """The realised-side indicators of one basket at one entry from ``hist``, the trailing
    returns of its names (at least 1,260 rows wanted; what a short history cannot give is NaN).
    ``h`` is the tenor in trading days, ``T`` in years."""
    out: dict[str, float] = {}
    r1, w1 = usable(hist, w, 252)
    if r1.size == 0:
        return out
    rb = r1 @ w1
    z = r1 - rb[:, None]
    sig = forecast_vol(r1)
    sig_b = float(forecast_vol(rb[:, None])[0])
    sig_rel = forecast_vol(z)
    out["n_names_1y"] = float(len(w1))
    out["sig_fc_bar"] = float(np.sum(w1 * sig))
    out["sig_B_fc"] = sig_b
    out["sig_rel_fc_bar"] = float(np.sum(w1 * sig_rel))
    out["rho_fc"] = cboe(sig_b, sig, w1)
    for label, n in (("1m", 21), ("3m", 63)):
        out[f"rho_real_{label}"] = cboe(
            float(realised_vol(rb[:, None], n)[0]), realised_vol(r1, n), w1
        )
    out["ret_B_3m"] = float(np.prod(1.0 + rb[-63:]) - 1.0)
    out["ret_B_12m"] = float(np.prod(1.0 + rb) - 1.0)
    out["beta_disp"] = beta_dispersion(r1, rb, w1)
    out["csad_g1"], out["csad_g2"] = csad_slopes(r1, rb, w1)
    ret3 = np.prod(1.0 + r1[-63:], axis=0) - 1.0
    out["rotation"] = float(np.sum(w1 * (ret3 - np.sum(w1 * ret3)) ** 2))
    zz = z[-63:]
    m2 = (zz**2) @ w1
    out["xs_kurtosis"] = float(np.mean((zz**4 @ w1) / np.maximum(m2 * m2, 1e-30)))
    # variance ratio and variability over three years (five for the long tenors, at least three)
    want = 756 if h <= 63 else 1260
    rL, wL = usable(hist, w, want)
    if rL.size == 0 and want > 756:
        rL, wL = usable(hist, w, 756)
    vr = float("nan")
    if rL.size:
        zL = rL - (rL @ wL)[:, None]
        vr, raw = variance_ratio(zL, wL, h)
        out["VR_cs"], out["VR_raw"] = vr, raw
    r3, w3 = usable(hist, w, 756)
    if r3.size:
        out["variability"] = variability(r3 - (r3 @ w3)[:, None], w3)
    out.update(gaussian_forecast(sig, sig_b, sig_rel, w1, T, vr))
    return out
