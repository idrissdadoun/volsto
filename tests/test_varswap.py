"""SPEC §10: replication strike = E[realised variance] under local vol; ξ₀ integrates to the strip."""

from __future__ import annotations

import numpy as np
import pytest

from volsto.config import SimConfig
from volsto.engine import MonteCarlo
from volsto.market import (
    DiscountCurve,
    ForwardCurve,
    ForwardVarianceCurve,
    LocalVolSurface,
    SSVISurface,
    varswap_strike,
    xi0_curve,
)
from volsto.models import LocalVol
from volsto.products import VarianceSwap


def test_flat_surface_strike_is_variance(
    forward_curve: ForwardCurve, discount: DiscountCurve
) -> None:
    flat = SSVISurface.flat_atm(0.2, 0.0, 0.0, 0.5, forward_curve, discount)
    for T in (0.05, 0.5, 3.0):
        assert varswap_strike(flat, T) == pytest.approx(0.04, rel=1e-9)
    xi = xi0_curve(flat, 3.0)
    np.testing.assert_allclose(xi.xi0(np.linspace(0.01, 3.0, 50)), 0.04, rtol=1e-7)


def test_skewed_surface_strike_above_atm(ssvi: SSVISurface) -> None:
    for T in (0.25, 1.0, 2.0):
        assert varswap_strike(ssvi, T) > float(ssvi.atm_vol(T)) ** 2


def test_xi0_integrates_back_to_strip(ssvi: SSVISurface) -> None:
    xi = xi0_curve(ssvi, 3.0)
    T = np.linspace(0.05, 3.0, 40)
    assert np.all(xi.xi0(T) > 0)
    # exact at nodes
    for Tn in xi.maturities[::7]:
        assert float(xi.total_variance(Tn)) == pytest.approx(
            Tn * varswap_strike(ssvi, float(Tn)), rel=1e-10
        )
    # close between nodes
    for Tm in (0.3, 0.77, 1.9, 2.55):
        assert float(xi.varswap_strike(Tm)) == pytest.approx(varswap_strike(ssvi, Tm), rel=2e-4)
    # forward strike identity
    k1, k2 = varswap_strike(ssvi, 1.0), varswap_strike(ssvi, 2.0)
    assert float(xi.forward_varswap_strike(1.0, 2.0)) == pytest.approx(2 * k2 - k1, rel=1e-9)
    assert float(xi.integral(0.0, 2.0)) == pytest.approx(2 * k2, rel=1e-9)


def test_forward_variance_curve_validation() -> None:
    with pytest.raises(ValueError, match="increasing"):
        ForwardVarianceCurve([0.5, 1.0], [0.03, 0.02])
    flat = ForwardVarianceCurve.flat(0.05)
    np.testing.assert_allclose(flat.xi0([0.1, 2.0, 9.0]), 0.05)


def test_replication_strike_equals_expected_realised_variance(
    ssvi: SSVISurface, local_vol: LocalVolSurface
) -> None:
    """Fair strike from log-contract replication vs MC E[RV] under Dupire local vol (daily fixings)."""
    T = 0.5
    cfg = SimConfig(n_paths=60_000, dt_max=1.0 / 252.0, chunk_size=30_000, seed=5)
    mc = MonteCarlo(cfg)
    swap = VarianceSwap.daily(T, 0.0, ssvi.discount, per_year=252)
    swap_grid = VarianceSwap(swap.fixing_times, 0.0, ssvi.discount, use_simulation_grid=True)
    res, res_grid = mc.price_many([swap, swap_grid], LocalVol(local_vol))
    df = float(ssvi.discount.df(T))
    k_mc = res.mean / df
    k_rep = varswap_strike(ssvi, T)
    # fixings coincide with simulation steps here, so both accumulators must agree exactly
    assert res_grid.mean == pytest.approx(res.mean, rel=1e-12)
    vol_mc, vol_rep = np.sqrt(k_mc), np.sqrt(k_rep)
    vol_err = res.stderr / df / (2 * vol_mc)
    assert abs(vol_mc - vol_rep) < max(0.0015, 3 * vol_err), (vol_mc, vol_rep, vol_err)


@pytest.mark.slow
def test_replication_strike_full(ssvi: SSVISurface, local_vol: LocalVolSurface) -> None:
    T = 1.0
    cfg = SimConfig(n_paths=400_000, dt_max=1.0 / 365.0, chunk_size=50_000, seed=6)
    swap = VarianceSwap.daily(T, 0.0, ssvi.discount, per_year=365)
    res = MonteCarlo(cfg).price(swap, LocalVol(local_vol))
    df = float(ssvi.discount.df(T))
    vol_mc, vol_rep = np.sqrt(res.mean / df), np.sqrt(varswap_strike(ssvi, T))
    vol_err = res.stderr / df / (2 * vol_mc)
    assert abs(vol_mc - vol_rep) < max(0.0007, 3 * vol_err), (vol_mc, vol_rep, vol_err)
