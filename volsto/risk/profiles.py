"""Spot-shift profiles and the cliquet gamma profile (SPEC v2 §7.8).

:func:`spot_profile` — ``S₀`` shifts on a grid (−30% to +30% in 2.5% steps by default) with the
price and, under ``"model"`` dynamics (leverage held in spot, factors at zero), the delta and
gamma (central three-point at each shift, all three legs in ``"model"`` mode) and the
sticky-leverage vega (parallel surface bump, ``L`` held); no recalibration is triggered along
the profile.  Other regimes reprice each shift through the regime's own state (recalibrated,
cached) and give the price only.  ``gamma_fd`` is the second difference of the price profile.

:func:`cliquet_gamma_profile` — the conditional value of an additive cliquet at an intermediate
fixing ``t_j`` as a function of the accumulated sum ``A_j`` (local-linear Gaussian-kernel
regression of the payoff forward-valued to ``t_j`` on ``A_j`` over the simulated paths, the
calibration's :func:`~volsto.calibration.particle.kernel_regression`), its second difference in
``A_j`` (the gamma profile of the study), and the Bachelier cross-check of the study: with the
remaining capped returns independent under Black–Scholes, the remaining sum has mean ``μ = Σ
E[clip(r_i)]`` and variance ``s² = Σ Var[clip(r_i)]`` (exact per-leg moments by Gauss–Hermite
quadrature) and, treating it as Gaussian, ``E[max(GF, A + Σ)] = GF + (A + μ − GF) N(d) + s φ(d)``,
``d = (A + μ − GF)/s``.  For the study cliquet at the 6m fixing ``s ≈ 0.11`` against an
accumulated-sum range of about ``−0.35`` to ``+0.12``, so the floor is in play over the whole
profile and the Gaussian step is never exact: the remaining sum is bounded above (capped legs)
with an open left tail, and the Gaussian of equal variance *over*prices the floor — measured
under Black–Scholes the Bachelier value exceeds the exact conditional value by up to 0.18% of
notional, i.e. by 20–25% of the conditional value around ``A_j ≈ 0`` (where that value is about
1% of notional) and by 4% of the value at the top of the range, and by nothing at the bottom
where both vanish.  The exact reference under Black–Scholes is :func:`bs_cliquet_value_mc`
(independent legs simulated directly); the regression is checked against it and the Bachelier
deviation is reported.  Checked by ``tests/test_risk_profiles.py``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.calibration.particle import kernel_regression
from volsto.config import SimConfig, SurfacePerturbation
from volsto.engine.mc import MonteCarlo
from volsto.market.bs import norm_cdf, norm_pdf
from volsto.market.curves import ForwardCurve
from volsto.models.base import Model
from volsto.products.base import Product
from volsto.products.cliquet import AdditiveCliquet
from volsto.risk.engine import RiskEngine, RiskState, Sensitivity
from volsto.risk.greeks import _spot_state

FloatArray = NDArray[np.float64]

DEFAULT_SHIFTS: tuple[float, ...] = tuple(
    float(x) for x in np.round(np.arange(-0.30, 0.3001, 0.025), 4)
)


def _model_delta_gamma(
    engine: RiskEngine, product: Product, st: RiskState, size: float
) -> tuple[Sensitivity, Sensitivity]:
    """Central delta / gamma with every leg under ``"model"`` dynamics."""
    up, _ = _spot_state(st, "model", size)
    dn, _ = _spot_state(st, "model", -size)
    s0, s_up, s_dn = st.spot, up.spot, dn.spot
    delta = engine.combination(
        f"delta[model @ {s0:.4g}]",
        product,
        [(up, "model", 1.0 / (s_up - s_dn)), (dn, "model", -1.0 / (s_up - s_dn))],
        unit="per unit spot",
        size=size,
        scheme="central",
    )
    c_up = 2.0 / ((s_up - s0) * (s_up - s_dn))
    c_dn = 2.0 / ((s0 - s_dn) * (s_up - s_dn))
    gamma = engine.combination(
        f"gamma[model @ {s0:.4g}]",
        product,
        [(up, "model", c_up), (st, "model", -(c_up + c_dn)), (dn, "model", c_dn)],
        unit="per unit spot squared",
        size=size,
        scheme="central",
    )
    return delta, gamma


def _sticky_vega(engine: RiskEngine, product: Product, st: RiskState, size: float) -> Sensitivity:
    """Parallel vega with ``L`` held: both legs in ``"sticky_leverage"`` mode."""
    bumped, achieved = engine.perturbed_state(
        st, lambda s: SurfacePerturbation("parallel", {"size": s}), size, label="vega"
    )
    c = 0.01 / achieved
    return engine.combination(
        f"vega[sticky @ {st.spot:.4g}]",
        product,
        [(bumped, "sticky_leverage", c), (st, "sticky_leverage", -c)],
        unit="per vol point",
        size=achieved,
        scheme="forward",
    )


def spot_profile(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    shifts: Sequence[float] = DEFAULT_SHIFTS,
    *,
    regime: str = "model",
    size: float = 0.01,
    with_vega: bool = True,
) -> pd.DataFrame:
    """Price along a grid of relative spot shifts; under ``regime="model"`` also the model
    delta / gamma and the sticky-leverage vega at each shift.  ``gamma_fd`` is the second
    difference of the price column (``nan`` at the ends)."""
    rows = []
    for m in shifts:
        h = float(np.log1p(m))
        if m != 0:
            shifted, mode = _spot_state(state, regime, h)
        else:
            shifted, mode = state, "recalibrate"
        price = engine.price(product, shifted, mode)
        row: dict[str, float] = {
            "shift": float(m),
            "spot": shifted.spot,
            "price": price.mean,
            "price_stderr": price.stderr,
        }
        if regime == "model":
            d, g = _model_delta_gamma(engine, product, shifted, size)
            row.update(delta=d.value, delta_stderr=d.stderr, gamma=g.value, gamma_stderr=g.stderr)
            if with_vega:
                v = _sticky_vega(engine, product, shifted, size)
                row.update(vega=v.value, vega_stderr=v.stderr)
        rows.append(row)
    frame = pd.DataFrame(rows)
    s = frame["spot"].to_numpy(dtype=np.float64)
    p = frame["price"].to_numpy(dtype=np.float64)
    fd = np.full(p.size, np.nan)
    if p.size >= 3:
        left = (p[1:-1] - p[:-2]) / (s[1:-1] - s[:-2])
        right = (p[2:] - p[1:-1]) / (s[2:] - s[1:-1])
        fd[1:-1] = 2.0 * (right - left) / (s[2:] - s[:-2])
    frame["gamma_fd"] = fd
    return frame


def _clipped_lognormal_moments(
    mean_log: float, std_log: float, lo: float, hi: float, n: int = 96
) -> tuple[float, float]:
    """Mean and variance of ``clip(e^Z − 1, lo, hi)`` for ``Z ~ N(mean_log, std_log²)``
    (probabilists' Gauss–Hermite quadrature)."""
    x, w = np.polynomial.hermite_e.hermegauss(n)
    r = np.clip(np.expm1(mean_log + std_log * x), lo, hi)
    norm = float(np.sqrt(2.0 * np.pi))
    m = float(np.sum(w * r) / norm)
    m2 = float(np.sum(w * r * r) / norm)
    return m, max(m2 - m * m, 0.0)


def bachelier_cliquet_value(
    cliquet: AdditiveCliquet,
    accumulated: FloatArray,
    t_j: float,
    sigma: float,
    forward_curve: ForwardCurve,
) -> FloatArray:
    """The study's cross-check: value at ``t_j`` (in the product's notional) given the
    accumulated sum, the remaining capped returns i.i.d. under Black–Scholes at ``sigma`` and
    their sum Gaussian; global floor only."""
    if np.isfinite(cliquet.global_cap):
        raise NotImplementedError("the Bachelier cross-check covers a global floor only")
    fixings = cliquet.fixing_times
    mu = var = 0.0
    for a, b in pairwise(fixings):
        if a < t_j - 1e-12:
            continue
        tau = float(b - a)
        drift = float(forward_curve.drift(a, b))
        m, v = _clipped_lognormal_moments(
            drift - 0.5 * sigma**2 * tau,
            sigma * float(np.sqrt(tau)),
            float(cliquet.local_floor),
            float(cliquet.local_cap),
        )
        mu += m
        var += v
    acc = np.asarray(accumulated, dtype=np.float64)
    df = float(cliquet.df(cliquet.maturity)) / float(cliquet.df(t_j))
    gf = float(cliquet.global_floor)
    if not np.isfinite(gf):
        return np.asarray(cliquet.notional * df * (acc + mu), dtype=np.float64)
    s = float(np.sqrt(var))
    d = (acc + mu - gf) / s
    val = gf + (acc + mu - gf) * norm_cdf(d) + s * norm_pdf(d)
    return np.asarray(cliquet.notional * df * val, dtype=np.float64)


def bs_cliquet_value_mc(
    cliquet: AdditiveCliquet,
    accumulated: FloatArray,
    t_j: float,
    sigma: float,
    forward_curve: ForwardCurve,
    *,
    n_samples: int = 400_000,
    seed: int = 0,
) -> tuple[FloatArray, FloatArray]:
    """Exact Black–Scholes reference for the conditional value at ``t_j``: the remaining capped
    returns are independent of ``A_j``, so ``V(A) = E[clip(A + Σ, GF, GC)]`` with ``Σ`` simulated
    directly (antithetic lognormal legs).  Returns the value and its standard error per grid
    point (in the product's notional, forward-valued to ``t_j``)."""
    fixings = cliquet.fixing_times
    rng = np.random.default_rng(seed)
    half = n_samples // 2
    total = np.zeros(2 * half)
    for a, b in pairwise(fixings):
        if a < t_j - 1e-12:
            continue
        tau = float(b - a)
        z = rng.standard_normal(half)
        z = np.concatenate([z, -z])
        r = np.expm1(
            float(forward_curve.drift(a, b)) - 0.5 * sigma**2 * tau + sigma * np.sqrt(tau) * z
        )
        total += np.clip(r, float(cliquet.local_floor), float(cliquet.local_cap))
    df = cliquet.notional * float(cliquet.df(cliquet.maturity)) / float(cliquet.df(t_j))
    acc = np.asarray(accumulated, dtype=np.float64)
    pay = np.clip(
        acc[:, None] + total[None, :], float(cliquet.global_floor), float(cliquet.global_cap)
    )
    pairs = 0.5 * (pay[:, :half] + pay[:, half:])
    return (
        np.asarray(df * pairs.mean(axis=1), dtype=np.float64),
        np.asarray(df * pairs.std(axis=1, ddof=1) / np.sqrt(half), dtype=np.float64),
    )


@dataclass(frozen=True)
class CliquetGammaProfile:
    """Conditional value and gamma of a cliquet across the accumulated sum at ``t_j``."""

    t_j: float
    accumulated: FloatArray
    value: FloatArray
    gamma: FloatArray
    bachelier: FloatArray | None
    n_paths: int
    bandwidth: float
    value_raw: FloatArray

    def as_frame(self) -> pd.DataFrame:
        data = {
            "accumulated": self.accumulated,
            "value": self.value,
            "value_raw": self.value_raw,
            "gamma": self.gamma,
        }
        if self.bachelier is not None:
            data["bachelier"] = self.bachelier
            data["deviation"] = self.value - self.bachelier
        return pd.DataFrame(data)


def cliquet_gamma_profile(
    cliquet: AdditiveCliquet,
    model: Model,
    sim: SimConfig,
    t_j: float,
    *,
    n_grid: int = 31,
    bandwidth: float | None = None,
    bs_sigma: float | None = None,
) -> CliquetGammaProfile:
    """Regression of the payoff forward-valued to ``t_j`` on the accumulated sum ``A_j`` over
    the paths; the grid spans the 1–99% quantiles of ``A_j``; the default bandwidth is twice the
    grid spacing (the second difference needs a smooth estimate).  The local-linear estimate
    carries the smoothing bias ``½h²m″`` (+0.1% of notional on the study cliquet at h = 0.03,
    where ``m″`` — the gamma profile — reaches 3–4 per unit²): ``value`` is corrected with the
    stencil second difference, ``value_raw`` is the uncorrected estimate, and the outermost
    ``round(h/da)`` points on each side carry no correction (``gamma`` is ``nan`` there).
    ``bs_sigma`` adds the Bachelier cross-check."""
    fixings = cliquet.fixing_times
    hits = np.flatnonzero(np.isclose(fixings, t_j))
    if hits.size == 0:
        raise ValueError("t_j must be a fixing date of the cliquet")
    j = int(hits[0])
    mc = MonteCarlo(sim)
    grid = mc.build_grid([cliquet], model)
    draws = mc.draws_for(grid, model)
    idx = grid.fixing_index
    acc_parts: list[FloatArray] = []
    pay_parts: list[FloatArray] = []
    for p0, p1 in sim.chunk_ranges(grid.n_records, model.n_factors):
        paths = model.simulate_chunk(grid, draws, p0, p1, sim.scheme)
        loc = cliquet.local_returns(paths, idx)
        acc_parts.append(np.sum(loc[:, :j], axis=1))
        pay_parts.append(cliquet.payoff(paths, idx) / float(cliquet.df(t_j)))
    acc = np.concatenate(acc_parts)
    pay = np.concatenate(pay_parts)
    lo, hi = np.percentile(acc, [1.0, 99.0])
    a_grid = np.linspace(lo, hi, n_grid)
    da = float(a_grid[1] - a_grid[0])
    h = float(bandwidth) if bandwidth is not None else 2.0 * da
    order = np.argsort(acc, kind="stable")
    m, _, _ = kernel_regression(
        np.ascontiguousarray(acc[order], dtype=np.float64),
        np.ascontiguousarray(pay[order], dtype=np.float64),
        np.ascontiguousarray(a_grid, dtype=np.float64),
        h,
        True,
        True,
        200,
    )
    raw = np.asarray(m, dtype=np.float64)
    # second difference on a bandwidth-wide stencil (as the calibration's m″ estimate) and the
    # ½h²m″ local-linear smoothing-bias correction (the conditional value is convex in A_j)
    st = max(1, round(h / da))
    gamma = np.full(n_grid, np.nan)
    gamma[st:-st] = (raw[2 * st :] - 2.0 * raw[st:-st] + raw[: -2 * st]) / (st * da) ** 2
    value = raw.copy()
    value[st:-st] -= 0.5 * h * h * gamma[st:-st]
    bach = None
    if bs_sigma is not None:
        bach = bachelier_cliquet_value(cliquet, a_grid, t_j, bs_sigma, model.forward_curve)
    return CliquetGammaProfile(t_j, a_grid, value, gamma, bach, int(acc.size), h, raw)
