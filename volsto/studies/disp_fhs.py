"""Dispersion study: filtered historical simulation (spec §7.2).

Forecast distribution of the performances over the window from the trailing daily returns of
the entry basket: each name's returns are standardised by its own daily EWMA vol (returns
strictly before the day), demeaned and rescaled to unit variance over the sample, multiplied by
today's forecast vol, and resampled jointly — whole days, in blocks (stationary bootstrap) — so
that the cross-section of each day, fat tails, earnings jumps and a few days of persistence
are kept, and the vol level is today's.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from volsto.studies import disp_payoff as dp

FloatArray = NDArray[np.float64]

DECAY: Final = 0.97
SEED_DAYS: Final = 21
N_PATHS: Final = 4000
MEAN_BLOCK: Final = 10
MIN_DAYS: Final = 252


def standardise(hist: FloatArray, demean: bool = True) -> FloatArray:
    """Returns over their daily EWMA vol (decay :data:`DECAY`, from the returns strictly before
    each day, seeded with the first :data:`SEED_DAYS` rows, which are then dropped), demeaned
    and rescaled to zero mean and unit variance per name (``demean=False``: left as they
    are)."""
    r = np.asarray(hist, dtype=np.float64)
    var = np.mean(r[:SEED_DAYS] ** 2, axis=0)
    z = np.empty((r.shape[0] - SEED_DAYS, r.shape[1]))
    for t in range(SEED_DAYS, r.shape[0]):
        z[t - SEED_DAYS] = r[t] / np.sqrt(np.maximum(var, 1e-12))
        var = DECAY * var + (1.0 - DECAY) * r[t] ** 2
    if demean:
        z = (z - z.mean(axis=0)) / z.std(axis=0)
    return z


def bootstrap_days(
    n_days: int, h: int, n_paths: int, mean_block: int, seed: int
) -> NDArray[np.int64]:
    """Stationary-bootstrap day indices, ``n_paths × h``: each path starts on a random day and,
    at each step, moves to the next day (circular) or, with probability ``1/mean_block``,
    jumps to a random one."""
    rng = np.random.default_rng(seed)
    jump = rng.random((n_paths, h)) < 1.0 / mean_block
    jump[:, 0] = True
    fresh = rng.integers(0, n_days, size=(n_paths, h))
    idx = np.empty((n_paths, h), dtype=np.int64)
    cur = fresh[:, 0]
    for s in range(h):
        cur = np.where(jump[:, s], fresh[:, s], (cur + 1) % n_days)
        idx[:, s] = cur
    return idx


def simulate(
    hist: FloatArray,
    sig_fc: FloatArray,
    h: int,
    seed: int,
    *,
    n_paths: int = N_PATHS,
    mean_block: int = MEAN_BLOCK,
    demean: bool = True,
) -> FloatArray:
    """Performances ``X`` (``n_paths × n``) after ``h`` days.  ``hist`` is the trailing sample
    (rows = days on which every name has a return, most recent last; at least
    :data:`MIN_DAYS` plus the seed rows), ``sig_fc`` today's annualised forecast vols."""
    z = standardise(hist, demean)
    if z.shape[0] < MIN_DAYS:
        raise ValueError(f"{z.shape[0]} days in the sample, {MIN_DAYS} needed")
    daily = z * (np.asarray(sig_fc, dtype=np.float64) / np.sqrt(252.0))[None, :]
    logs = np.log1p(np.maximum(daily, -0.99))
    idx = bootstrap_days(z.shape[0], h, n_paths, mean_block, seed)
    out = np.empty((n_paths, z.shape[1]))
    step = 500
    for a in range(0, n_paths, step):
        out[a : a + step] = np.exp(logs[idx[a : a + step]].sum(axis=1))
    return out


def summary(X: FloatArray, w: FloatArray, strikes: FloatArray | None = None) -> dict[str, Any]:
    """Means and standard deviations of the forecast payoffs: ``D``, ``SD``, ``G``, ``V``,
    ``Σ w_i|R_i|``, ``|R̄|``, ``D/B``, the calls at ``strikes``; ``κ_fc`` and the correlation
    by the Cboe formula on the paths' terminal covariance."""
    t = dp.terminal(X, w)
    out: dict[str, Any] = {}
    for k in ("D", "SD", "G", "V", "absR", "absRb", "D_rel", "Rb"):
        out[f"E_{k}"] = float(np.mean(t[k]))
        out[f"sd_{k}"] = float(np.std(t[k]))
    out["kappa_fc"] = out["E_D"] / np.sqrt(out["E_V"]) if out["E_V"] > 0 else float("nan")
    vols = X.std(axis=0)
    own = float(np.sum((w * vols) ** 2))
    cross = float(np.sum(w * vols)) ** 2 - own
    out["rho_fhs"] = (float(np.var(t["B"])) - own) / cross if cross > 0 else float("nan")
    if strikes is not None:
        pay = np.maximum(t["D"][:, None] - np.asarray(strikes)[None, :], 0.0)
        out["E_calls"] = pay.mean(axis=0)
        out["sd_calls"] = pay.std(axis=0)
        out["_calls"] = pay
    out["_t"] = t
    return out
