"""Dispersion study: payoffs, structures, deltas and hedged P&L (spec §4, §5; notes §2, App. B).

Everything is per unit of notional and in performance units: ``X_i = S_{i,T}/S_{i,0}`` of the
basket frozen at entry, ``B = Σ w_i X_i``, ``R = X − 1``.  :func:`terminal` returns the payoffs
at expiry and the identities the study checks on every window (the sandwich
``SD ≤ D ≤ SD + 2|R̄|``, the gap ``G = D − SD``); :func:`hedge_legs` the daily delta hedges of
the three hedged legs (single-name straddles and basket straddle at their entry vols, the
Palladium forward with the homogeneous delta of the notes' Appendix B); :func:`daily_stats`
the realised statistics inside a window; :func:`structures` the P&L of every structure of the
spec from the legs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.special import ndtr

FloatArray = NDArray[np.float64]


def terminal(X: FloatArray, w: FloatArray) -> dict[str, Any]:
    """Payoffs at expiry of one scenario (``X`` of shape ``n``) or many (``N × n``)."""
    X = np.asarray(X, dtype=np.float64)
    B = X @ w
    R = X - 1.0
    Rb = B - 1.0
    Z = X - np.expand_dims(B, -1)
    D = np.abs(Z) @ w
    absR = np.abs(R) @ w
    V = (Z * Z) @ w
    SD = absR - np.abs(Rb)
    return {
        "B": B, "Rb": Rb, "D": D, "SD": SD, "G": D - SD, "V": V, "absR": absR,
        "absRb": np.abs(Rb), "D_rel": D / B,
    }  # fmt: skip


def gap_formula(X: FloatArray, w: FloatArray) -> FloatArray:
    """The gap by the notes' eq. (2.6): ``2 Σ w_i (|R̄| − (ε R_i)⁺)⁺`` with ``ε = sgn R̄``."""
    R = np.asarray(X, dtype=np.float64) - 1.0
    Rb = R @ w
    eps = np.where(Rb >= 0, 1.0, -1.0)
    inner = np.maximum(np.expand_dims(eps, -1) * R, 0.0)
    gap = 2.0 * np.maximum(np.expand_dims(np.abs(Rb), -1) - inner, 0.0) @ w
    return np.asarray(gap, dtype=np.float64)


def cross_section(X: FloatArray, w: FloatArray) -> dict[str, float]:
    """Shape of one realised cross-section: ``D/√V``, the weighted kurtosis of ``R_i − R̄`` and
    the number of names on each side of zero and of the basket."""
    t = terminal(X, w)
    Z = X - t["B"]
    m2 = float(np.sum(w * Z * Z))
    m4 = float(np.sum(w * Z**4))
    return {
        "kappa": float(t["D"] / np.sqrt(t["V"])) if t["V"] > 0 else float("nan"),
        "kurtosis": m4 / (m2 * m2) if m2 > 0 else float("nan"),
        "n_up": int(np.sum(X > 1.0)),
        "n_down": int(np.sum(X < 1.0)),
        "n_above_basket": int(np.sum(Z > 0)),
    }


def straddle_delta(x: FloatArray, vol: FloatArray, tau: FloatArray) -> FloatArray:
    """Delta, in units of the underlying's performance, of a straddle struck at 1 on a
    performance ``x`` (Black–Scholes at ``vol``, no carry): ``2Φ(d1) − 1``."""
    a = np.maximum(vol * np.sqrt(np.maximum(tau, 1e-12)), 1e-9)
    return 2.0 * ndtr(np.log(x) / a + 0.5 * a) - 1.0


def palladium_value(X: FloatArray, w: FloatArray, sig_rel: FloatArray, tau: float) -> float:
    """``Π = B Σ w_i g(X_i/B)``, ``g`` the Black–Scholes straddle on the relative performance
    struck at 1 with total vol ``a_i = σ_{i−B}√τ`` (notes App. B)."""
    B = float(X @ w)
    a = np.maximum(sig_rel * np.sqrt(max(tau, 1e-12)), 1e-9)
    y = X / B
    d1 = np.log(y) / a + 0.5 * a
    g = y * (2.0 * ndtr(d1) - 1.0) - (2.0 * ndtr(d1 - a) - 1.0)
    return float(B * np.sum(w * g))


def palladium_delta(
    X: FloatArray, w: FloatArray, sig_rel: FloatArray, tau: FloatArray
) -> FloatArray:
    """``∂Π/∂X_j = w_j [Σ_i w_i (1 − 2Φ(d_{2,i})) + 2Φ(d_{1,j}) − 1]`` for one state (``X`` of
    shape ``n``, scalar ``tau``) or a path (``t × n`` with ``tau`` of shape ``t``)."""
    X = np.asarray(X, dtype=np.float64)
    B = X @ w
    tau_ = np.expand_dims(np.asarray(tau, dtype=np.float64), -1)
    a = np.maximum(sig_rel * np.sqrt(np.maximum(tau_, 1e-12)), 1e-9)
    d1 = np.log(X / np.expand_dims(B, -1)) / a + 0.5 * a
    common = (1.0 - 2.0 * ndtr(d1 - a)) @ w
    return w * (np.expand_dims(common, -1) + 2.0 * ndtr(d1) - 1.0)


def hedge_legs(
    path: FloatArray,
    w: FloatArray,
    T: float,
    vol_names: FloatArray,
    vol_basket: float,
    sig_rel: FloatArray,
    n_total: int | None = None,
) -> dict[str, float]:
    """Daily delta hedges over one window.  ``path`` is ``(N + 1) × n``, the value of each
    frozen holding per unit of its entry price (row 0 is ones).  Hedges are rebalanced at each
    of the first ``N`` closes and held to the next; price changes only (no carry).  With
    ``n_total`` the path is the first ``N`` days of a window of ``n_total`` days of maturity
    ``T`` (the hedges accrued when the position is unwound before expiry).

    Returns the hedge P&L of the single-name strip (``Σ w_i`` of each straddle's hedge), of the
    basket straddle (hedged with the frozen basket) and of the Palladium forward, and the
    notional traded by each (``Σ_t |Δδ_t|`` with the entry and exit trades)."""
    path = np.asarray(path, dtype=np.float64)
    n_steps = path.shape[0] - 1
    life = n_steps if n_total is None else int(n_total)
    tau = T * (life - np.arange(n_steps)) / life
    x = path[:-1]
    dx = np.diff(path, axis=0)
    basket = path @ w
    d_names = straddle_delta(x, vol_names[None, :], tau[:, None])
    d_basket = straddle_delta(basket[:-1], np.asarray(vol_basket, dtype=np.float64), tau)
    d_pall = palladium_delta(x, w, sig_rel, tau)

    def traded(delta: FloatArray, scale: FloatArray) -> float:
        full = np.concatenate(
            [np.zeros((1, *delta.shape[1:])), delta, np.zeros((1, *delta.shape[1:]))]
        )
        return float(np.sum(np.abs(np.diff(full, axis=0)) * scale))

    return {
        "hedge_SS": float(-np.sum(d_names * dx * w[None, :])),
        "hedge_BS": float(-np.sum(d_basket * np.diff(basket))),
        "hedge_PF": float(-np.sum(d_pall * dx)),
        "traded_SS": traded(d_names, w[None, :]),
        "traded_BS": traded(d_basket, np.ones(())),
        "traded_PF": traded(d_pall, np.ones((1, path.shape[1]))),
    }


def daily_stats(path: FloatArray, w: FloatArray) -> dict[str, float]:
    """Realised statistics inside one window (``path`` as in :func:`hedge_legs`): realised
    vols and the Cboe-formula correlation on them, the average pairwise correlation, the daily
    cross-sectional variance with the frozen basket's drifting weights, the window's variance
    ratio ``V_T / Σ_t CSV_t``, the CSAD regression and the realised variance dispersion."""
    path = np.asarray(path, dtype=np.float64)
    n_steps = path.shape[0] - 1
    basket = path @ w
    r = path[1:] / path[:-1] - 1.0
    rb = basket[1:] / basket[:-1] - 1.0
    drift_w = w[None, :] * path[:-1] / basket[:-1, None]
    z = r - rb[:, None]
    csv = np.sum(drift_w * z * z, axis=1)
    csad = np.sum(drift_w * np.abs(z), axis=1)
    lr, lrb = np.log(path[1:] / path[:-1]), np.log(basket[1:] / basket[:-1])
    rv_names = 252.0 / n_steps * np.sum(lr * lr, axis=0)
    rv_basket = 252.0 / n_steps * float(np.sum(lrb * lrb))
    vols = np.sqrt(rv_names)
    own = float(np.sum((w * vols) ** 2))
    cross = float(np.sum(w * vols)) ** 2 - own
    live = vols > 0
    corr = np.asarray(np.corrcoef(r[:, live].T) if live.sum() > 1 else np.ones((1, 1)))
    iu = np.triu_indices(int(corr.shape[0]), 1)
    term = terminal(path[-1], w)
    A = np.column_stack([np.ones(n_steps), np.abs(rb), rb * rb])
    gamma = np.linalg.lstsq(A, csad, rcond=None)[0]
    return {
        "rho_real": (rv_basket - own) / cross if cross > 0 else float("nan"),
        "rho_pairwise": float(np.nanmean(corr[iu])) if iu[0].size else float("nan"),
        "vol_names_real": float(np.sum(w * vols)),
        "vol_basket_real": float(np.sqrt(rv_basket)),
        "csv_annual": float(252.0 * csv.mean()),
        "sum_csv": float(csv.sum()),
        "vr_window": float(term["V"] / csv.sum()) if csv.sum() > 0 else float("nan"),
        "csad_g0": float(gamma[0]),
        "csad_g1": float(gamma[1]),
        "csad_g2": float(gamma[2]),
        "rv_dispersion": float(np.sum(w * rv_names) - rv_basket),
    }


def structures(leg: dict[str, float]) -> dict[str, float]:
    """P&L of every structure of spec §4 from the legs of one window.

    ``leg`` holds the payoffs (``D``, ``absR``, ``absRb``), the undiscounted prices (``P_D``,
    ``SS``, ``Str_B``), the hedge P&Ls (``hedge_PF``, ``hedge_SS``, ``hedge_BS``; NaN when not
    computed) and the ratios ``lam_theta``, ``lam_rho``, ``h_v``.  Suffix ``_U`` is held to
    expiry unhedged, ``_H`` delta-hedged daily (a structure's hedged P&L combines its hedged
    legs)."""
    pf_u = leg["D"] - leg["P_D"]
    ss_u = leg["absR"] - leg["SS"]
    bs_u = leg["absRb"] - leg["Str_B"]
    out = {"PF_U": pf_u, "SS_U": ss_u, "BS_U": bs_u}
    pf_h = pf_u + leg.get("hedge_PF", np.nan)
    ss_h = ss_u + leg.get("hedge_SS", np.nan)
    bs_h = bs_u + leg.get("hedge_BS", np.nan)
    out.update(PF_H=pf_h, SS_H=ss_h, BS_H=bs_h)
    for tag, pf, ss, bs in (("U", pf_u, ss_u, bs_u), ("H", pf_h, ss_h, bs_h)):
        out[f"PKG_v_{tag}"] = ss - bs
        out[f"PKG_theta_{tag}"] = ss - leg["lam_theta"] * bs
        out[f"PKG_rho_{tag}"] = ss - leg["lam_rho"] * bs
        out[f"GAP_{tag}"] = pf - (ss - bs)
        out[f"GAP_rho_{tag}"] = pf - (ss - leg["lam_rho"] * bs)
        out[f"PF_v_{tag}"] = pf - leg["h_v"] * ss
        out[f"REV_{tag}"] = -(ss - bs)
    return out
