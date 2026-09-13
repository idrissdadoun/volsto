"""VIX futures and options in the two-factor model — a check on the vol-of-vol dynamics (SPEC
§15 Part 5; Bergomi §7.7).

The VIX future expiring at ``T`` settles on the 30-day variance-swap volatility observed at
``T`` (eqs. 7.84–7.85, with the VS/log-contract distinction ignored as in the book):

    ``VIX_T² = (1/Δ) ∫_T^{T+Δ} ξ_T^u du``,  ``Δ = 30/365``,

and in the two-factor model ``ξ_T^u = ξ_0^u exp(ω x_T^u − ½ ω² χ(T, u))`` is an explicit function
of the Gaussian factor state ``(X¹_T, X²_T)`` (eqs. 7.30–7.35, :meth:`BergomiSV.forward_variance`),
so every VIX instrument is a two-dimensional Gaussian integral: :func:`vix_quadrature` prices
the future ``F_T = E[VIX_T]`` (a future is driftless, eq. 7.86), the calls and puts on it
``E[(VIX_T − K)⁺]`` (undiscounted forward prices) and the implied Black volatility of each strike
by tensor Gauss–Hermite quadrature on the factor covariance :meth:`BergomiSV.factor_covariance`
(the book's §7.7.2 numerical route; ``n_hermite`` points per factor after a Cholesky rotation).
``E[VIX_T²] = (1/Δ) ∫_T^{T+Δ} ξ_0^u du`` exactly (forward variances are martingales), so the
future lies below the root of the forward 1m VS variance by the convexity of the square root
— the "VIX² futures vs forward VS variance up to the convexity term" check — and the implied
vol of vol at short horizon tends to the instantaneous lognormal vol of the forward-starting
one-month VS volatility (eq. 7.39 machinery, :func:`~volsto.analytics.bergomi.
forward_vs_vol_of_vol_flat` for a flat curve).  :func:`vix_monte_carlo` prices the same
instruments on simulated factor paths of the library kernel (standard errors), the cross-check
of the quadrature.  The model is lognormal in the instantaneous forward variances, so it
under-produces the upward skew of VIX smiles (book §7.7.4): as a stage-1 target only the
futures and the ATM VIX volatility are meaningful (SPEC §15 Part 3), never the wings.
Checked by ``tests/test_vix.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from volsto.analytics.bergomi import forward_vs_vol_of_vol_flat
from volsto.config import SimConfig
from volsto.engine import GaussianDraws, TimeGrid
from volsto.market import black_vega, bs_price, implied_vol
from volsto.models.bergomi import BergomiSV

FloatArray = NDArray[np.float64]

#: the VIX window, 30 calendar days
VIX_DELTA = 30.0 / 365.0


@dataclass(frozen=True)
class VIXQuotes:
    """VIX instruments of one expiry: the future, the strikes with their forward call and put
    prices and Black implied volatilities, the ATM implied vol of vol, the forward 1m VS
    variance ``E[VIX²]`` and the convexity gap ``sqrt(E[VIX²]) − F``; ``stderr`` arrays are
    filled by the Monte Carlo variant (zeros for the quadrature)."""

    T: float
    delta: float
    method: str
    future: float
    future_stderr: float
    strikes: FloatArray
    calls: FloatArray
    calls_stderr: FloatArray
    puts: FloatArray
    implied_vols: FloatArray
    implied_vols_stderr: FloatArray
    atm_vol_of_vol: float
    atm_vol_of_vol_stderr: float
    forward_vs_variance: float
    n_points: int

    @property
    def convexity_gap(self) -> float:
        """``sqrt(E[VIX²]) − E[VIX]`` (positive: Jensen on the square root)."""
        return float(np.sqrt(self.forward_vs_variance) - self.future)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "strike": self.strikes,
                "moneyness": self.strikes / self.future,
                "call": self.calls,
                "call_stderr": self.calls_stderr,
                "put": self.puts,
                "implied_vol": self.implied_vols,
                "implied_vol_stderr": self.implied_vols_stderr,
            }
        )

    def __repr__(self) -> str:
        return (
            f"VIXQuotes(T={self.T:g}, {self.method}: future {self.future:.4f} ± "
            f"{self.future_stderr:.4f}, sqrt(E[VIX²]) {np.sqrt(self.forward_vs_variance):.4f}, "
            f"ATM vol of vol {self.atm_vol_of_vol:.4f} ± {self.atm_vol_of_vol_stderr:.4f}, "
            f"{self.strikes.size} strikes)"
        )


def forward_vs_variance(
    model: BergomiSV, T: float, delta: float = VIX_DELTA, n_quad: int = 32
) -> float:
    """``E[VIX_T²] = (1/Δ) ∫_T^{T+Δ} ξ_0^u du`` from the model's initial curve."""
    x, w = np.polynomial.legendre.leggauss(n_quad)
    u = T + 0.5 * delta * (x + 1.0)
    return float(np.asarray(model.xi0.xi0(u), dtype=np.float64) @ w * 0.5)


def _vix_from_factors(
    model: BergomiSV, T: float, factors: FloatArray, delta: float, n_quad: int
) -> FloatArray:
    """``VIX_T`` per factor state ``(n, nf)``."""
    return np.asarray(
        np.sqrt(model.vs_variance(T, factors, T, T + delta, n_quad)), dtype=np.float64
    )


def _quotes(
    T: float,
    delta: float,
    method: str,
    vix: FloatArray,
    weights: FloatArray,
    strikes: ArrayLike | None,
    fwd_var: float,
    n_points: int,
    *,
    antithetic: bool | None,
) -> VIXQuotes:
    """Assemble the quotes from VIX samples / nodes with probability weights (sum to 1).
    ``antithetic=None``: quadrature nodes (no standard errors); otherwise Monte Carlo samples
    whose payoffs are averaged over antithetic pairs before the standard error."""
    w = np.asarray(weights, dtype=np.float64)
    future = float(vix @ w)
    if strikes is None:
        ks = future * np.array([0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5])
    else:
        ks = np.atleast_1d(np.asarray(strikes, dtype=np.float64))
        if np.any(ks <= 0):
            raise ValueError("strikes must be positive")
    call_pay = np.maximum(vix[:, None] - ks[None, :], 0.0)
    put_pay = np.maximum(ks[None, :] - vix[:, None], 0.0)
    atm_pay = np.maximum(vix - future, 0.0)
    calls = w @ call_pay
    puts = w @ put_pay
    otm = np.where(ks >= future, calls, puts)
    cp = np.where(ks >= future, 1, -1)
    iv = np.asarray(implied_vol(otm, future, ks, T, cp, 1.0), dtype=np.float64)
    atm_c = float(w @ atm_pay)
    atm_iv = float(implied_vol(atm_c, future, future, T, 1, 1.0))
    if antithetic is None:
        se_f, se_atm = 0.0, 0.0
        se_c = np.zeros(ks.size)
        se_iv = np.zeros(ks.size)
    else:

        def se(samples: FloatArray) -> FloatArray:
            x = 0.5 * (samples[0::2] + samples[1::2]) if antithetic else samples
            return np.asarray(x.std(axis=0, ddof=1) / np.sqrt(x.shape[0]), dtype=np.float64)

        se_f = float(se(vix[:, None])[0])
        se_c = se(call_pay)
        vega = np.asarray(black_vega(future, ks, T, iv, 1.0), dtype=np.float64)
        se_iv = np.where(vega > 0, se_c / np.maximum(vega, 1e-300), np.nan)
        se_atm = float(se(atm_pay[:, None])[0]) / float(black_vega(future, future, T, atm_iv, 1.0))
    return VIXQuotes(
        float(T),
        float(delta),
        method,
        future,
        se_f,
        ks,
        np.asarray(calls, dtype=np.float64),
        np.asarray(se_c, dtype=np.float64),
        np.asarray(puts, dtype=np.float64),
        iv,
        np.asarray(se_iv, dtype=np.float64),
        atm_iv,
        se_atm,
        fwd_var,
        int(n_points),
    )


def vix_quadrature(
    model: BergomiSV,
    T: float,
    *,
    strikes: ArrayLike | None = None,
    delta: float = VIX_DELTA,
    n_hermite: int = 60,
    n_quad: int = 32,
) -> VIXQuotes:
    """VIX future, calls / puts and implied vols of expiry ``T`` by tensor Gauss–Hermite
    quadrature over the factor state ``X_T ~ N(x₀ e^{−kT}, Σ_T)`` (``Σ_T`` =
    :meth:`BergomiSV.factor_covariance`), ``n_hermite`` nodes per factor; ``n_quad``
    Gauss–Legendre points for the 30-day integral of ``ξ_T^u``.  Default strikes: 70–150% of the
    future.  Cost: ``n_hermite^nf`` evaluations of the forward-variance curve (3600 for the
    two-factor model at 60 points); the kinked option payoffs converge algebraically in the node
    count (0.4% on the ATM implied vol of vol between 20 and 60 nodes), the future geometrically.
    """
    if T <= 0 or delta <= 0 or n_hermite < 4:
        raise ValueError("T and delta must be positive, n_hermite >= 4")
    nf = model.n_factors
    x, w = np.polynomial.hermite_e.hermegauss(n_hermite)  # weight exp(−x²/2)
    w = w / np.sqrt(2.0 * np.pi)
    grids = np.meshgrid(*([x] * nf), indexing="ij")
    z = np.column_stack([g.ravel() for g in grids])  # (n_hermite^nf, nf) standard normals
    wts = np.prod(
        np.column_stack([np.meshgrid(*([w] * nf), indexing="ij")[i].ravel() for i in range(nf)]),
        axis=1,
    )
    cov = model.factor_covariance(T)
    chol = np.linalg.cholesky(cov)
    mean = model.x0 * np.exp(-model.ks * T)
    factors = mean[None, :] + z @ chol.T
    vix = _vix_from_factors(model, T, factors, delta, n_quad)
    return _quotes(
        T,
        delta,
        "quadrature",
        vix,
        wts,
        strikes,
        forward_vs_variance(model, T, delta, n_quad),
        int(z.shape[0]),
        antithetic=None,
    )


def vix_monte_carlo(
    model: BergomiSV,
    T: float,
    *,
    sim: SimConfig,
    strikes: ArrayLike | None = None,
    delta: float = VIX_DELTA,
    n_quad: int = 32,
) -> VIXQuotes:
    """The same quotes from simulated factor paths of the library kernel at ``T`` (the exact
    OU transition of :meth:`BergomiSV.simulate_chunk`; the spot is irrelevant), with standard
    errors from antithetic pair averages."""
    if T <= 0 or delta <= 0:
        raise ValueError("T and delta must be positive")
    grid = TimeGrid.build([T], sim.dt_max)
    col = grid.fixing_index[T]
    draws = GaussianDraws(sim.seed, sim.n_paths, grid.n_steps, model.n_brownians, sim.antithetic)
    vix = np.empty(sim.n_paths)
    for p0, p1 in sim.chunk_ranges(grid.n_records, model.n_factors):
        paths = model.simulate_chunk(grid, draws, p0, p1, sim.scheme)
        vix[p0:p1] = _vix_from_factors(model, T, paths.factors_at(col), delta, n_quad)
    n = vix.size
    return _quotes(
        T,
        delta,
        "monte_carlo",
        vix,
        np.full(n, 1.0 / n),
        strikes,
        forward_vs_variance(model, T, delta, n_quad),
        int(sim.n_paths),
        antithetic=bool(sim.antithetic),
    )


def vix_term_structure(
    model: BergomiSV,
    Ts: ArrayLike,
    *,
    delta: float = VIX_DELTA,
    n_hermite: int = 60,
) -> pd.DataFrame:
    """Futures, ``sqrt(E[VIX²])``, convexity gap and ATM implied vol of vol per expiry by
    quadrature, next to the flat-curve instantaneous vol of the forward-starting 30-day VS
    volatility (:func:`~volsto.analytics.bergomi.forward_vs_vol_of_vol_flat`, the ``T → 0``
    limit of the implied vol of vol)."""
    rows = []
    for T in np.atleast_1d(np.asarray(Ts, dtype=np.float64)):
        q = vix_quadrature(model, float(T), delta=delta, n_hermite=n_hermite)
        rows.append(
            {
                "T": float(T),
                "future": q.future,
                "sqrt_forward_vs_variance": float(np.sqrt(q.forward_vs_variance)),
                "convexity_gap": q.convexity_gap,
                "atm_vol_of_vol": q.atm_vol_of_vol,
                "instantaneous_vol_of_vol_flat": float(
                    forward_vs_vol_of_vol_flat(model.params, float(T), float(T) + delta)
                ),
            }
        )
    return pd.DataFrame(rows)


def black_call(F: float, K: ArrayLike, T: float, sigma: ArrayLike) -> FloatArray:
    """Undiscounted Black call on a future (a helper for tests)."""
    return np.asarray(bs_price(F, K, T, sigma, 0.0, 0.0, 1), dtype=np.float64)


__all__ = [
    "VIX_DELTA",
    "VIXQuotes",
    "black_call",
    "forward_vs_variance",
    "vix_monte_carlo",
    "vix_quadrature",
    "vix_term_structure",
]
