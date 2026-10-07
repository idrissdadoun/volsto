"""Dispersion study: the Gaussian copula on exact marginals (spec §3.2–§3.4).

European payoffs on the basket depend on the joint law at expiry only.  The study's market
model joins the marginals of :mod:`volsto.studies.disp_smile` with one correlation:
``L_i = √ρ·M + √(1−ρ)·E_i``, ``X_i = Q_i(Φ(L_i))``, with ``2^17`` scrambled Sobol points kept
fixed across correlations, bumps and strikes of a date (:class:`Draws`), and ``Q_i∘Φ``
tabulated on a uniform latent grid and read by index arithmetic (:func:`simulate`).
:func:`calibrate_rho` finds the correlation that reprices a basket straddle;
:func:`price_basket` returns every price, sensitivity and check of one basket on one date.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.special import ndtri
from scipy.stats import qmc

from volsto.studies.disp_smile import Z_MAX, Z_POINTS, Marginal

FloatArray = NDArray[np.float64]

N_POINTS: Final = 2**17
BATCHES: Final = 16
RHO_BOUNDS: Final = (0.0, 0.99)
#: Palladium call strikes as multiples of the forward's price at the mark.
CALL_MULTIPLES: Final = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
#: Smallest call price that counts as priced.
MIN_CALL: Final = 0.0005
#: Buckets of the basket's performance for the conditional profile of relative dispersion.
PROFILE_EDGES: Final = (-0.10, -0.03, 0.03, 0.10)
PROFILE_LABELS: Final = ("below -10%", "-10 to -3%", "-3 to 3%", "3 to 10%", "above 10%")


@dataclass(frozen=True)
class Draws:
    """Standard normal draws of a date: the common factor ``m`` (``N``) and the specific ones
    ``e`` (``N × n``)."""

    m: FloatArray
    e: FloatArray


def sobol_draws(n_names: int, seed: int, n_points: int = N_POINTS) -> Draws:
    """``n_points`` scrambled Sobol points in dimension ``n_names + 1``, mapped to normals."""
    u = qmc.Sobol(d=n_names + 1, scramble=True, seed=seed).random(n_points)
    z = ndtri(np.clip(u, 1e-12, 1 - 1e-12))
    return Draws(np.ascontiguousarray(z[:, 0]), np.ascontiguousarray(z[:, 1:]))


def tables(marginals: list[Marginal]) -> FloatArray:
    """The quantile tables of the names, one row each (``n × Z_POINTS``)."""
    return np.ascontiguousarray(np.stack([m.table() for m in marginals]))


def simulate(tab: FloatArray, draws: Draws, rho: float) -> FloatArray:
    """Performances ``X`` (``N × n``, in units of the spots) at correlation ``rho``."""
    n = tab.shape[0]
    latent = np.sqrt(rho) * draws.m[:, None] + np.sqrt(1.0 - rho) * draws.e[:, :n]
    pos = (latent + Z_MAX) * ((Z_POINTS - 1) / (2.0 * Z_MAX))
    np.clip(pos, 0.0, Z_POINTS - 1.000001, out=pos)
    i0 = pos.astype(np.int64)
    frac = pos - i0
    i0 += np.arange(n, dtype=np.int64)[None, :] * Z_POINTS
    flat = tab.ravel()
    lo = flat[i0]
    return np.asarray(lo + frac * (flat[i0 + 1] - lo), dtype=np.float64)


def batch_se(values: FloatArray) -> float:
    """Standard error of the mean of ``values`` by batch means (:data:`BATCHES` batches)."""
    means = values[: values.size // BATCHES * BATCHES].reshape(BATCHES, -1).mean(axis=1)
    return float(means.std(ddof=1) / np.sqrt(BATCHES))


def payoffs(X: FloatArray, w: FloatArray) -> dict[str, FloatArray]:
    """Scenario payoffs of the frozen basket: ``B``, ``Rb``, ``D``, ``SD``, ``V``, ``absR``
    (``Σ w_i|R_i|``)."""
    B = X @ w
    D = np.abs(X - B[:, None]) @ w
    absR = np.abs(X - 1.0) @ w
    V = ((X - B[:, None]) ** 2) @ w
    Rb = B - 1.0
    return {"B": B, "Rb": Rb, "D": D, "SD": absR - np.abs(Rb), "V": V, "absR": absR}


def basket_straddle(tab: FloatArray, draws: Draws, w: FloatArray, rho: float) -> float:
    return float(np.mean(np.abs(simulate(tab, draws, rho) @ w - 1.0)))


def calibrate_rho(
    tab: FloatArray, draws: Draws, w: FloatArray, target: float, xtol: float = 1e-9
) -> tuple[float, str]:
    """The correlation in :data:`RHO_BOUNDS` at which the copula's basket straddle equals
    ``target`` (same random numbers throughout): ``(rho, flag)`` with flag ``""``,
    ``"clipped low"`` or ``"clipped high"`` when no correlation in the range does."""
    lo, hi = RHO_BOUNDS

    def gap(rho: float) -> float:
        return basket_straddle(tab, draws, w, rho) - target

    g_lo = gap(lo)
    if g_lo >= 0:
        return lo, "clipped low"
    g_hi = gap(hi)
    if g_hi <= 0:
        return hi, "clipped high"
    return float(brentq(gap, lo, hi, xtol=xtol, rtol=1e-12)), ""


def calibrate_rho_moment(
    tab: FloatArray, draws: Draws, w: FloatArray, target_m: float
) -> tuple[float, str]:
    """As :func:`calibrate_rho` on the basket's second moment ``E[R̄²]`` (``ρ_V``)."""
    lo, hi = RHO_BOUNDS

    def gap(rho: float) -> float:
        return float(np.mean((simulate(tab, draws, rho) @ w - 1.0) ** 2)) - target_m

    if gap(lo) >= 0:
        return lo, "clipped low"
    if gap(hi) <= 0:
        return hi, "clipped high"
    return float(brentq(gap, lo, hi, xtol=1e-7)), ""


def _legs(X: FloatArray, w: FloatArray, strikes: FloatArray | None) -> dict[str, Any]:
    p = payoffs(X, w)
    out: dict[str, Any] = {
        "P_D": float(p["D"].mean()),
        "Str_B": float(np.abs(p["Rb"]).mean()),
        "SS": float(p["absR"].mean()),
        "EV": float(p["V"].mean()),
        "M_B": float((p["Rb"] ** 2).mean()),
        "_p": p,
    }
    if strikes is not None:
        out["calls"] = np.array([np.maximum(p["D"] - k, 0.0).mean() for k in strikes])
    return out


def price_basket(
    marginals: list[Marginal],
    bumped: list[Marginal] | None,
    w: FloatArray,
    draws: Draws,
    rho_mark: float,
    *,
    extras: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Prices, sensitivities and checks of one basket at ``rho_mark`` (spec §3.3–§3.4).

    ``bumped`` are the marginals with every single-name smile one vol point higher (``None``
    skips the vol sensitivities).  ``extras`` maps a label to another correlation at which
    the forward, the calls (same cash strikes) and the basket straddle are also reported
    (the ``±0.05`` and ``ρ_90`` marks of the monthly subset).  Everything is undiscounted and
    per unit of notional; correlation sensitivities per ``+0.01``, vol sensitivities per vol
    point."""
    tab = tables(marginals)
    n = len(marginals)
    w = np.asarray(w, dtype=np.float64)
    X = simulate(tab, draws, rho_mark)
    base = _legs(X, w, None)
    p = base.pop("_p")
    D = p["D"]
    strikes = np.array(CALL_MULTIPLES) * base["P_D"]
    calls = np.array([np.maximum(D - k, 0.0).mean() for k in strikes])
    out: dict[str, Any] = {
        "rho_mark": rho_mark,
        **{k: v for k, v in base.items()},
        "P_D_se": batch_se(D),
        "sd_D": float(D.std()),
        "strikes": strikes,
        "calls": calls,
        "calls_se": np.array([batch_se(np.maximum(D - k, 0.0)) for k in strikes]),
        "kappa_cop": base["P_D"] / np.sqrt(base["EV"]),
        "E_B": float(p["B"].mean()),
    }
    # the copula against the marginals it was built from (check C3)
    single = np.abs(X - 1.0)
    out["straddle_cop"] = single.mean(axis=0)
    out["straddle_se"] = np.array([batch_se(single[:, i]) for i in range(n)])
    # common-move delta of each call (Euler) and its finite-difference check (C13)
    out["delta_c"] = np.array([np.mean(D * (k < D)) for k in strikes])
    out["delta_c_fd"] = np.array(
        [
            (np.maximum(1.01 * D - k, 0.0).mean() - np.maximum(0.99 * D - k, 0.0).mean()) / 0.02
            for k in strikes
        ]
    )
    # conditional profile of relative dispersion
    rel = D / p["B"]
    out["E_DB"] = float(rel.mean())
    bucket = np.digitize(p["Rb"], PROFILE_EDGES)
    out["profile"] = np.array(
        [rel[bucket == j].mean() / out["E_DB"] if (bucket == j).any() else np.nan for j in range(5)]
    )
    out["profile_share"] = np.array([(bucket == j).mean() for j in range(5)])
    # the regressions of the static replica on the model's scenarios (T16)
    for name, cols in (
        ("rep_abs", [p["absR"], np.abs(p["Rb"])]),
        ("rep_sq", [(X - 1.0) ** 2 @ w, p["Rb"] ** 2]),
    ):
        A = np.column_stack([np.ones(D.size), *cols])
        coef = np.linalg.lstsq(A, D, rcond=None)[0]
        res = D - A @ coef
        out[name] = np.array([*coef, 1.0 - res.var() / D.var()])
    # correlation sensitivities: central differences at ±0.01, one-sided at the bounds
    up, dn = min(rho_mark + 0.01, RHO_BOUNDS[1]), max(rho_mark - 0.01, RHO_BOUNDS[0])
    if up > dn:
        hi = _legs(simulate(tab, draws, up), w, strikes)
        lo = _legs(simulate(tab, draws, dn), w, strikes)
        scale = 0.01 / (up - dn)
        out["dPD_drho"] = (hi["P_D"] - lo["P_D"]) * scale
        out["dStrB_drho"] = (hi["Str_B"] - lo["Str_B"]) * scale
        out["dcalls_drho"] = (hi["calls"] - lo["calls"]) * scale
        out["dEV_drho"] = (hi["EV"] - lo["EV"]) * scale
    else:
        out.update(
            dPD_drho=np.nan,
            dStrB_drho=np.nan,
            dcalls_drho=np.full(len(strikes), np.nan),
            dEV_drho=np.nan,
        )
    out["lambda_rho"] = -out["dPD_drho"] / out["dStrB_drho"] if out["dStrB_drho"] != 0.0 else np.nan
    # single-name vol sensitivity: every smile one vol point up, same correlation
    if bumped is not None:
        b = _legs(simulate(tables(bumped), draws, rho_mark), w, strikes)
        out["dPD_dvol"] = b["P_D"] - base["P_D"]
        out["dStrB_dvol"] = b["Str_B"] - base["Str_B"]
        out["dcalls_dvol"] = b["calls"] - calls
        out["dSS_dvol"] = b["SS"] - base["SS"]
        vega = float(np.sum(w * np.array([m.vega for m in marginals])))
        out["vega_SS"] = vega
        out["h_v"] = out["dPD_dvol"] / vega
    for label, rho in (extras or {}).items():
        e = _legs(simulate(tab, draws, float(np.clip(rho, *RHO_BOUNDS))), w, strikes)
        out[f"P_D@{label}"] = e["P_D"]
        out[f"Str_B@{label}"] = e["Str_B"]
        out[f"calls@{label}"] = e["calls"]
    return out


def gaussian_inputs(
    vols: FloatArray, w: FloatArray, rho: float
) -> tuple[FloatArray, float, FloatArray]:
    """Equicorrelation Gaussian set-up: ``(covariance of each name with the basket, basket vol,
    relative vols σ_{i−B})`` from the vols, the weights and one correlation."""
    ws = w * vols
    cov_ib = vols * (rho * ws.sum() + (1.0 - rho) * ws)
    var_b = float(np.sum(w * cov_ib))
    rel = np.sqrt(np.maximum(vols**2 + var_b - 2.0 * cov_ib, 0.0))
    return cov_ib, float(np.sqrt(var_b)), rel


def cboe_correlation(basket_vol: float, vols: FloatArray, w: FloatArray) -> float:
    """``(σ_B² − Σ w_i²σ_i²) / Σ_{i≠j} w_i w_j σ_i σ_j`` (not clipped)."""
    own = float(np.sum((w * vols) ** 2))
    cross = float(np.sum(w * vols)) ** 2 - own
    return (basket_vol**2 - own) / cross if cross > 0 else float("nan")
