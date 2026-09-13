"""Identity from the study (SPEC §10): ATM implied variance equals the dollar-gamma-weighted
average of realised (local) variance along the path,

    σ̂² · E[∫₀ᵀ Γ_BS(t, S_t; σ̂) S_t² dt] = E[∫₀ᵀ Γ_BS(t, S_t; σ̂) S_t² σ_loc²(t, S_t) dt],

which is the expectation of the delta-hedging P&L of an option marked at its implied vol
(Dupire / Gatheral "most likely path").  Verified to ≤ 0.1 vol point (slow) under local vol.
"""

from __future__ import annotations

import numpy as np
import pytest

from volsto.config import SimConfig
from volsto.engine import GaussianDraws, TimeGrid
from volsto.market import LocalVolSurface, SSVISurface, norm_pdf
from volsto.models import LocalVol
from volsto.products import daily_schedule


def _dollar_gamma_identity(
    ssvi: SSVISurface, local_vol: LocalVolSurface, n_paths: int, seed: int
) -> tuple[float, float]:
    T = 1.0
    F = float(ssvi.forward(T))
    K = F
    sig = float(ssvi.implied_vol(K, T))
    fixings = daily_schedule(T, per_year=252)
    cfg = SimConfig(n_paths=n_paths, dt_max=1.0 / 252.0, chunk_size=10_000, seed=seed)
    grid = TimeGrid.build(fixings, cfg.dt_max)
    model = LocalVol(local_vol)
    draws = GaussianDraws(seed, n_paths, grid.n_steps, model.n_brownians, cfg.antithetic)
    cols = np.arange(grid.n_records - 1)  # left Riemann sum; the T-endpoint gamma is singular
    t = grid.record_times[cols]
    dt = np.diff(grid.record_times)
    tau = T - t
    q = float(-ssvi.forward_curve.dividend_curve.log_df(T) / T)
    drift = np.asarray(ssvi.forward_curve.drift(t, T))
    nums, dens = [], []
    for p0 in range(0, n_paths, cfg.chunk_size):
        paths = model.simulate_chunk(grid, draws, p0, min(p0 + cfg.chunk_size, n_paths), cfg.scheme)
        S = paths.spot_at(cols)
        var_loc = paths.variance_at(cols)
        d1 = (np.log(S) + drift - np.log(K)) / (sig * np.sqrt(tau)) + 0.5 * sig * np.sqrt(tau)
        dollar_gamma = np.exp(-q * tau) * norm_pdf(d1) * S / (sig * np.sqrt(tau))  # S² Γ
        nums.append(np.sum(dollar_gamma * var_loc * dt, axis=1))
        dens.append(np.sum(dollar_gamma * dt, axis=1))
    num = np.concatenate(nums)
    den = np.concatenate(dens)
    # ratio of means; delta-method stderr in vol terms (antithetic pairs averaged)
    num = 0.5 * (num[0::2] + num[1::2])
    den = 0.5 * (den[0::2] + den[1::2])
    ratio = num.mean() / den.mean()
    resid = num - ratio * den
    err_var = resid.std(ddof=1) / np.sqrt(resid.size) / den.mean()
    return float(np.sqrt(ratio)), float(err_var / (2 * np.sqrt(ratio)))


def test_atm_implied_variance_is_dollar_gamma_weighted_realised_variance(
    ssvi: SSVISurface, local_vol: LocalVolSurface
) -> None:
    vol_dg, err = _dollar_gamma_identity(ssvi, local_vol, n_paths=20_000, seed=21)
    sig = float(ssvi.implied_vol(ssvi.forward(1.0), 1.0))
    assert abs(vol_dg - sig) < max(0.0025, 3 * err), (vol_dg, sig, err)


@pytest.mark.slow
def test_dollar_gamma_identity_full(ssvi: SSVISurface, local_vol: LocalVolSurface) -> None:
    vol_dg, err = _dollar_gamma_identity(ssvi, local_vol, n_paths=200_000, seed=22)
    sig = float(ssvi.implied_vol(ssvi.forward(1.0), 1.0))
    assert abs(vol_dg - sig) < 0.001, (vol_dg, sig, err)
