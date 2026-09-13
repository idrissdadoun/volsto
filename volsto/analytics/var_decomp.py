"""Variance of the integrated variance ``V = ∫₀ᵀ V_u du`` — the variance-swap payoff's variance —
and its stochastic-volatility / leverage decomposition (SPEC §7.14, §15 Part 1).

Pure SV (``V_u = ξ_u^u``, lognormal forward variances, book eqs. 7.19–7.20 machinery on the
diagonal): ``Var(V) = ∫∫ ξ_0^u ξ_0^v (exp(ω² Cov(x_u^u, x_v^v)) − 1) du dv`` is
:func:`volsto.analytics.bergomi.var_integrated_variance`; :func:`sv_integrated_variance_moments`
returns it with the mean ``∫₀ᵀ ξ_0`` and :func:`sv_discrete_moments` gives the *exact* moments of
the kernel's discrete accumulator (trapezoid in the SV variance under the second-order SV step,
left point under the frozen-variance step) on a given grid, so a Monte Carlo check separates the
time-discretisation bias from the sampling error.

LSV (``V_u = L²(u, S_u) ξ_u^u``): with the kernel simulated on the same draws the factor paths
are identical (the factor step is exact and the spot never feeds back into the factors), so
``V^SV = ∫₀ᵀ ξ_u^u du`` is available path by path.  The split reported by
:func:`var_decomposition` is

    ``Var(V) = Var(V^SV) + Var(V − V^SV) + 2 Cov(V^SV, V − V^SV)``

(SV part, leverage part, cross term — an identity) together with the law-of-total-variance split
with respect to the factor-path functional ``V^SV``: ``Var(V) = Var(E[V | V^SV]) + E[Var(V |
V^SV)]``, the conditional expectation by a polynomial regression of ``V`` on ``V^SV``.  With
``L ≡ 1`` every leverage term vanishes and the SV numbers are reproduced (tested).  Optionally
the variance of the discretely sampled realised variance ``Σ ln²(S_{t_i}/S_{t_{i−1}})`` is
reported, whose excess over ``Var(V)`` is the sampling term ``≈ 2 E[Σ (∫_i V)²]``.

Standard errors: batch means over pair-aligned batches (:func:`volsto.engine.stats.batch_means`).
Checked by ``tests/test_smile_dynamics.py::test_var_decomposition_pure_sv`` and
``test_var_decomposition_lsv``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from volsto.analytics.bergomi import cov_x_diag, var_integrated_variance
from volsto.analytics.smile_dynamics import kernel_of
from volsto.config import BergomiParams, SimConfig
from volsto.engine.grid import TimeGrid
from volsto.engine.rng import GaussianDraws
from volsto.engine.stats import batch_means
from volsto.market.varswap import ForwardVarianceCurve
from volsto.models.base import Model
from volsto.models.bergomi import BergomiSV
from volsto.products.base import daily_schedule

FloatArray = NDArray[np.float64]

RULES = ("trapezoid", "left")


def sv_integrated_variance_moments(
    p: BergomiParams, xi0: ForwardVarianceCurve, T: float, *, n_quad: int = 48
) -> tuple[float, float]:
    """``(E[V], Var(V))`` of ``V = ∫₀ᵀ ξ_u^u du`` in the pure SV model: ``E[V] = ∫₀ᵀ ξ_0^u du``
    (martingale forward variances) and ``Var(V)`` by
    :func:`volsto.analytics.bergomi.var_integrated_variance`."""
    if T <= 0:
        raise ValueError("T must be positive")
    return float(xi0.total_variance(T)), var_integrated_variance(p, xi0, 0.0, T, n_quad)


def sv_discrete_moments(kernel: BergomiSV, times: FloatArray, *, rule: str) -> tuple[float, float]:
    """Exact ``(E, Var)`` of the discrete accumulator ``Σ_m c_m ξ_{t_m}^{t_m}`` of the pure SV
    kernel on the grid ``times`` (``t_0 = 0``): ``rule="trapezoid"`` — ``c_0 = Δ_0/2``, ``c_m =
    (Δ_{m−1} + Δ_m)/2``, ``c_N = Δ_{N−1}/2`` (the ``sv_order2`` accumulator ``Σ Δ (ξ_n +
    ξ_{n+1})/2``); ``rule="left"`` — ``c_m = Δ_m`` on the left nodes (the frozen-variance step).
    ``E = Σ c_m ξ_0^{t_m}``, ``Var = Σ_ml c_m c_l ξ_0^{t_m} ξ_0^{t_l} (e^{ω² Cov(x_m, x_l)} − 1)``
    with ``Cov`` from :func:`volsto.analytics.bergomi.cov_x_diag` (initial factors at zero)."""
    if rule not in RULES:
        raise ValueError(f"rule must be one of {RULES}")
    if np.any(kernel.x0 != 0.0):
        raise ValueError("discrete moments assume the factors start at zero")
    t = np.asarray(times, dtype=np.float64)
    if t.size < 2 or t[0] != 0.0 or np.any(np.diff(t) <= 0):
        raise ValueError("times must start at 0 and be strictly increasing")
    d = np.diff(t)
    if rule == "trapezoid":
        c = np.concatenate(([0.5 * d[0]], 0.5 * (d[:-1] + d[1:]), [0.5 * d[-1]]))
        nodes = t
    else:
        c = d
        nodes = t[:-1]
    xi = np.asarray(kernel.xi0.xi0(nodes), dtype=np.float64)
    a = c * xi
    w = kernel.params.omega
    cov = cov_x_diag(kernel.params, nodes[:, None], nodes[None, :])
    return float(a.sum()), float(a @ np.expm1(w * w * cov) @ a)


@dataclass(frozen=True)
class VarianceDecomposition:
    """Var(V) of the integrated variance to ``T`` and its SV / leverage split (see module)."""

    T: float
    n_paths: int
    n_batches: int
    degree: int
    mean: float
    mean_stderr: float
    var_total: float
    var_total_stderr: float
    var_sv: float
    var_sv_stderr: float
    var_leverage: float
    var_leverage_stderr: float
    cov_cross: float
    cov_cross_stderr: float
    var_explained: float
    var_explained_stderr: float
    var_residual: float
    r2: float
    mean_sv_closed: float
    var_sv_closed: float
    var_sv_discrete: float
    mean_sv_discrete: float
    var_realised: float | None
    var_realised_stderr: float | None
    fixings_per_year: int | None

    @property
    def leverage_share(self) -> float:
        """``[Var(V) − Var(V^SV)] / Var(V)``: the part of the payoff variance the leverage adds
        (negative when the leverage damps the SV variance)."""
        return (self.var_total - self.var_sv) / self.var_total

    def to_dict(self) -> dict[str, float | int | None]:
        return {
            "T": self.T,
            "mean": self.mean,
            "mean_stderr": self.mean_stderr,
            "var_total": self.var_total,
            "var_total_stderr": self.var_total_stderr,
            "var_sv": self.var_sv,
            "var_sv_stderr": self.var_sv_stderr,
            "var_leverage": self.var_leverage,
            "var_leverage_stderr": self.var_leverage_stderr,
            "cov_cross": self.cov_cross,
            "cov_cross_stderr": self.cov_cross_stderr,
            "var_explained": self.var_explained,
            "var_explained_stderr": self.var_explained_stderr,
            "var_residual": self.var_residual,
            "r2": self.r2,
            "leverage_share": self.leverage_share,
            "mean_sv_closed": self.mean_sv_closed,
            "var_sv_closed": self.var_sv_closed,
            "mean_sv_discrete": self.mean_sv_discrete,
            "var_sv_discrete": self.var_sv_discrete,
            "var_realised": self.var_realised,
            "var_realised_stderr": self.var_realised_stderr,
            "fixings_per_year": self.fixings_per_year,
            "n_paths": self.n_paths,
        }

    def __repr__(self) -> str:
        rv = (
            f", Var(RV) = {self.var_realised:.4g} ± {self.var_realised_stderr:.2g}"
            if self.var_realised is not None and self.var_realised_stderr is not None
            else ""
        )
        return (
            f"VarianceDecomposition(T={self.T:g}: E[V] = {self.mean:.5g} ± {self.mean_stderr:.2g}, "
            f"Var(V) = {self.var_total:.4g} ± {self.var_total_stderr:.2g} = SV {self.var_sv:.4g} "
            f"+ leverage {self.var_leverage:.4g} + cross {self.cov_cross:.4g}; explained by V_SV "
            f"{self.var_explained:.4g} (r2 {self.r2:.4f}); closed form SV "
            f"{self.var_sv_closed:.4g}, discrete {self.var_sv_discrete:.4g}{rv})"
        )


def _explained_variance(v: FloatArray, vsv: FloatArray, degree: int) -> float:
    z = (vsv - vsv.mean()) / (vsv.std() if vsv.std() > 0 else 1.0)
    x = np.vander(z, degree + 1, increasing=True)
    beta, *_ = np.linalg.lstsq(x, v, rcond=None)
    return float(np.var(x @ beta, ddof=1))


def var_decomposition(
    model: Model,
    T: float,
    *,
    sim: SimConfig,
    degree: int = 3,
    n_batches: int = 20,
    fixings_per_year: int | None = None,
) -> VarianceDecomposition:
    """Var(V) decomposition of ``V = ∫₀ᵀ V_u du`` by simulation (see the module docstring).

    ``model`` is a :class:`~volsto.models.bergomi.BergomiSV` (then ``V^SV = V`` and the leverage
    terms vanish identically) or an :class:`~volsto.models.lsv.LSV` (its kernel is simulated on
    the same draws for ``V^SV``).  ``degree`` is the polynomial degree of the regression of ``V``
    on ``V^SV``; ``fixings_per_year`` adds the variance of the realised variance sampled on that
    daily schedule.  Closed forms: the continuous SV moments and the exact discrete moments of
    the accumulator on the simulation grid (``rule`` chosen from ``sim.scheme.sv_order2``).
    """
    if T <= 0:
        raise ValueError("T must be positive")
    if degree < 1 or n_batches < 2:
        raise ValueError("need degree >= 1 and n_batches >= 2")
    kernel = kernel_of(model)
    fixings: FloatArray = np.array([T])
    if fixings_per_year is not None:
        if fixings_per_year <= 0:
            raise ValueError("fixings_per_year must be positive")
        fixings = daily_schedule(T, fixings_per_year)
    grid = TimeGrid.build(fixings, sim.dt_max, calibration_grid=model.required_times())
    idx = grid.fixing_index
    col_T = idx[T]
    draws = GaussianDraws(sim.seed, sim.n_paths, grid.n_steps, model.n_brownians, sim.antithetic)
    n = sim.n_paths
    v = np.empty(n)
    vsv = np.empty(n)
    rv = np.empty(n) if fixings_per_year is not None else None
    cols = idx.indices(fixings) if fixings_per_year is not None else None
    same = isinstance(model, BergomiSV)
    for p0, p1 in sim.chunk_ranges(grid.n_records, model.n_factors):
        paths = model.simulate_chunk(grid, draws, p0, p1, sim.scheme)
        v[p0:p1] = paths.integrated_variance(0, col_T)
        if same:
            vsv[p0:p1] = v[p0:p1]
        else:
            kp = kernel.simulate_chunk(grid, draws, p0, p1, sim.scheme)
            vsv[p0:p1] = kp.integrated_variance(0, col_T)
        if rv is not None and cols is not None:
            rv[p0:p1] = paths.realised_variance_fixings(cols)
    anti = sim.antithetic
    mean = batch_means(lambda a: float(np.mean(a)), v, n_batches=n_batches, pair_aligned=anti)
    var_t = batch_means(
        lambda a: float(np.var(a, ddof=1)), v, n_batches=n_batches, pair_aligned=anti
    )
    var_sv = batch_means(
        lambda a: float(np.var(a, ddof=1)), vsv, n_batches=n_batches, pair_aligned=anti
    )
    lev = v - vsv
    var_lev = batch_means(
        lambda a: float(np.var(a, ddof=1)), lev, n_batches=n_batches, pair_aligned=anti
    )
    cross = batch_means(
        lambda a, b: 2.0 * float(np.cov(a, b, ddof=1)[0, 1]),
        vsv,
        lev,
        n_batches=n_batches,
        pair_aligned=anti,
    )
    expl = batch_means(
        lambda a, b: _explained_variance(a, b, degree),
        v,
        vsv,
        n_batches=n_batches,
        pair_aligned=anti,
    )
    mean_c, var_c = sv_integrated_variance_moments(kernel.params, kernel.xi0, T)
    rule = "trapezoid" if sim.scheme.sv_order2 else "left"
    mean_d, var_d = sv_discrete_moments(
        kernel, grid.times[: grid.record_steps[col_T] + 1], rule=rule
    )
    var_rv = var_rv_se = None
    if rv is not None:
        est = batch_means(
            lambda a: float(np.var(a, ddof=1)), rv, n_batches=n_batches, pair_aligned=anti
        )
        var_rv, var_rv_se = est.value, est.stderr
    return VarianceDecomposition(
        float(T),
        int(n),
        int(n_batches),
        int(degree),
        mean.value,
        mean.stderr,
        var_t.value,
        var_t.stderr,
        var_sv.value,
        var_sv.stderr,
        var_lev.value,
        var_lev.stderr,
        cross.value,
        cross.stderr,
        expl.value,
        expl.stderr,
        var_t.value - expl.value,
        expl.value / var_t.value if var_t.value > 0 else 1.0,
        mean_c,
        var_c,
        var_d,
        mean_d,
        var_rv,
        var_rv_se,
        fixings_per_year,
    )


__all__ = [
    "RULES",
    "VarianceDecomposition",
    "sv_discrete_moments",
    "sv_integrated_variance_moments",
    "var_decomposition",
]
