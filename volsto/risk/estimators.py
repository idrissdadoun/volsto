"""Precision estimators (SPEC v2 §7.11): likelihood-ratio delta and vega, conditional Greeks by
regression, and the control variate on the difference.

**Likelihood ratio.**  The spot normal of each step is ``z_{i,0}`` (spot Brownian first in the
kernels' Cholesky factor, ``δW^S = √δ_i z_{i,0}``).  Only the first transition depends on ``S₀``
in the discretised model, so the exact score of the scheme is ``z_{0,0} / (S₀ σ₀ √δ₀)`` with
``σ₀² = v₀`` the initial instantaneous variance (leverage included) — variance ``∝ 1/δ₀``, i.e.
useless on the 1/1460 first step of the default schedule.  :func:`lr_delta` therefore
aggregates the spot normals over ``[0, τ]`` (``first_step``, default one week): ``Z̄ = Σ_{t_i<τ}
√δ_i z_{i,0} / √τ`` and weight ``Z̄ / (S₀ σ₀ √τ)`` — the score of the constant-coefficient
approximation of the dynamics over ``[0, τ]``, exact under Black–Scholes and for ``τ = δ₀``,
with an ``O(τ)`` bias under a local / stochastic vol from the variation of ``σ`` over the week
that the bump cross-check bounds.  :func:`lr_vega` is the per-step score of a parallel shift of
the instantaneous vol, ``Σ_i 0.01 [(z_{i,0}² − 1)/σ_i − √δ_i z_{i,0}]`` with ``σ_i`` the
recorded vol at the start of step ``i`` (a record-all-steps grid): exact under Black–Scholes,
the frozen-coefficient approximation of the sticky-leverage parallel vega otherwise (the shift
of the surface moves ``ξ₀`` and ``L``, not each ``σ_i`` by 0.01 exactly).  Both carry the
payoff's discontinuities without differencing them; :func:`compare` reports the z-score against
the bump estimate and flags ``|z| > 3``.

**Conditional Greeks** (:func:`conditional_greeks`): the base paths and their CRN bumps
(``S₀ e^{±h}`` under the ``"model"`` regime; an optional vol-bumped model) are simulated on one
grid that records ``t``; the discounted payoff and the per-path bump differences are regressed
on a polynomial basis (degree 3 by default, cross terms) in ``(ln S_t, X¹_t, X²_t)`` — a
quadratic cannot follow the sigmoid of a vanilla delta across the spot range (RMSE 0.05 against
the Black–Scholes delta at 6m, 0.02 with the cubic).  The conditional
delta is ``E[pay₊ − pay₋ | state_t] / (S_t (e^h − e^{−h}))``: exact for spot-homogeneous
dynamics (Black–Scholes), the tangent-process approximation for the LSV whose leverage is held
in spot.  This is the hedger's engine of M8.

**Control variate on the difference** (:func:`cv_difference`): for vanilla-like products the
per-path bump difference under the model is regressed on the same difference under
Black–Scholes at the market implied vol of the strike, simulated on the same grid with the
same normals (column 0 of the draws is shared by every model); the Black–Scholes difference has
a known expectation (the analytic Greek), so ``D̄ − β (C̄ − E[C])`` is unbiased and the variance
reduction ``Var(D)/Var(D − βC)`` is reported.

Checked by ``tests/test_risk_estimators.py``.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from volsto.config import SimConfig
from volsto.engine.grid import TimeGrid
from volsto.engine.mc import MonteCarlo
from volsto.market.bs import bs_delta, bs_gamma, bs_vega
from volsto.market.curves import ForwardCurve
from volsto.models.base import Model
from volsto.models.bs import BlackScholes
from volsto.products.base import Product
from volsto.products.vanilla import EuropeanOption
from volsto.risk.engine import (
    RiskEngine,
    RiskState,
    Sensitivity,
    model_regime_spot_bump,
    surface_of,
)
from volsto.risk.greeks import _spot_state

FloatArray = NDArray[np.float64]


def _pairs(x: FloatArray, antithetic: bool) -> FloatArray:
    return 0.5 * (x[0::2] + x[1::2]) if antithetic else x


def _sensitivity(
    name: str, vals: FloatArray, unit: str, size: float, scheme: str, **extra: object
) -> Sensitivity:
    return Sensitivity(
        name,
        float(vals.mean()),
        float(vals.std(ddof=1) / np.sqrt(vals.size)),
        unit,
        size,
        scheme,
        (),
        int(vals.size),
        dict(extra),
    )


def _initial_vol(model: Model) -> float:
    v0 = model.instantaneous_variance(model.initial_state(2))
    return float(np.sqrt(float(np.mean(v0))))


# --------------------------------------------------------------------------------------------
# likelihood ratio
# --------------------------------------------------------------------------------------------


def lr_delta(
    product: Product, model: Model, sim: SimConfig, *, first_step: float | None = 1.0 / 52.0
) -> Sensitivity:
    """Likelihood-ratio delta (per unit spot) with the spot normals aggregated over
    ``[0, first_step]`` (``None``: the exact first-step score of the scheme)."""
    mc = MonteCarlo(sim)
    grid = mc.build_grid([product], model)
    draws = mc.draws_for(grid, model)
    idx = grid.fixing_index
    if first_step is None:
        k = 1
    else:
        k = max(1, int(np.searchsorted(grid.times, first_step * (1.0 - 1e-9), side="left")))
    tau = float(grid.times[k])
    sq = np.sqrt(grid.dts[:k])
    sigma0 = _initial_vol(model)
    s0 = model.spot
    parts: list[FloatArray] = []
    for p0, p1 in sim.chunk_ranges(grid.n_records, model.n_factors):
        paths = model.simulate_chunk(grid, draws, p0, p1, sim.scheme)
        pay = product.payoff(paths, idx)
        z = draws.block(0, k, p0, p1)[:, :, 0]
        zbar = z @ sq / np.sqrt(tau)
        parts.append(pay * zbar / (s0 * sigma0 * np.sqrt(tau)))
    vals = _pairs(np.concatenate(parts), sim.antithetic)
    return _sensitivity(
        "delta[LR]", vals, "per unit spot", tau, "likelihood-ratio", first_step=tau, n_steps=k
    )


def lr_vega(product: Product, model: Model, sim: SimConfig) -> Sensitivity:
    """Likelihood-ratio vega (per vol point of a parallel shift of the instantaneous vol) on a
    record-all-steps grid."""
    grid = TimeGrid.build(
        product.fixing_times,
        sim.dt_max,
        calibration_grid=model.required_times(),
        record_all_steps=True,
    )
    mc = MonteCarlo(sim)
    draws = mc.draws_for(grid, model)
    idx = grid.fixing_index
    sq = np.sqrt(grid.dts)
    parts: list[FloatArray] = []
    for p0, p1 in sim.chunk_ranges(grid.n_records, model.n_factors):
        paths = model.simulate_chunk(grid, draws, p0, p1, sim.scheme)
        pay = product.payoff(paths, idx)
        z = draws.block(0, grid.n_steps, p0, p1)[:, :, 0]
        sigma = np.sqrt(np.maximum(paths.variance[:, :-1], 1e-16))
        score = np.sum((z * z - 1.0) / sigma - sq[None, :] * z, axis=1)
        parts.append(0.01 * pay * score)
    vals = _pairs(np.concatenate(parts), sim.antithetic)
    return _sensitivity("vega[LR]", vals, "per vol point", 0.01, "likelihood-ratio")


def compare(lr: Sensitivity, bump: Sensitivity, threshold: float = 3.0) -> dict[str, float | bool]:
    """z-score of ``lr − bump`` on the combined standard error and the disagreement flag."""
    se = float(np.hypot(lr.stderr, bump.stderr))
    z = (lr.value - bump.value) / se if se > 0 else float("nan")
    return {
        "lr": lr.value,
        "lr_stderr": lr.stderr,
        "bump": bump.value,
        "bump_stderr": bump.stderr,
        "z": z,
        "flag": bool(abs(z) > threshold),
    }


# --------------------------------------------------------------------------------------------
# conditional Greeks
# --------------------------------------------------------------------------------------------


def _basis(features: FloatArray, degree: int) -> FloatArray:
    """``[1, x_j, x_j x_k, …]`` up to ``degree`` (with repetition) on standardised features."""
    n, d = features.shape
    cols = [np.ones(n)]
    for deg in range(1, degree + 1):
        for combo in itertools.combinations_with_replacement(range(d), deg):
            cols.append(np.prod(features[:, list(combo)], axis=1))
    return np.column_stack(cols)


@dataclass
class ConditionalGreeks:
    """Regression-based conditional Greeks at ``t`` as functions of ``(ln S_t, X_t)``."""

    t: float
    degree: int
    mean: FloatArray
    scale: FloatArray
    coefficients: dict[str, FloatArray]
    r2: dict[str, float]
    features: FloatArray = field(repr=False)
    targets: dict[str, FloatArray] = field(repr=False)
    targets_raw: dict[str, FloatArray] = field(repr=False)

    def predict(self, kind: str, ln_s: FloatArray, factors: FloatArray | None = None) -> FloatArray:
        """``kind`` in ``coefficients``; ``factors`` ``(n, n_factors)`` (omitted when the model has
        none)."""
        ln_s = np.atleast_1d(np.asarray(ln_s, dtype=np.float64))
        if factors is None:
            raw = ln_s[:, None]
        else:
            raw = np.column_stack([ln_s, np.asarray(factors, dtype=np.float64)])
        x = (raw - self.mean) / self.scale
        return np.asarray(_basis(x, self.degree) @ self.coefficients[kind], dtype=np.float64)

    def fitted(self, kind: str) -> FloatArray:
        return np.asarray(_basis(self.features, self.degree) @ self.coefficients[kind])

    def unconditional(self, kind: str) -> tuple[float, float]:
        """Mean and standard error of the raw ``S₀``-bump target (the ``t = 0`` Greek; the
        conditional targets are normalised by ``S_t/S₀`` and their mean is ``E[Greek_t]``)."""
        y = self.targets_raw[kind]
        return float(y.mean()), float(y.std(ddof=1) / np.sqrt(y.size))


def conditional_greeks(
    product: Product,
    model: Model,
    sim: SimConfig,
    t: float,
    *,
    spot_size: float = 0.01,
    vol_model: Model | None = None,
    vol_size: float = 0.01,
    degree: int = 3,
) -> ConditionalGreeks:
    """Value, delta, gamma (and vega when ``vol_model`` — the sticky-leverage bumped model — is
    given) at ``t`` by regression of the per-path CRN quantities on the basis in the state at
    ``t``; the regression is on the non-antithetic per-path samples."""
    if t <= 0 or t >= product.maturity:
        raise ValueError("t must lie strictly inside (0, maturity)")
    fixings = np.unique(np.concatenate([product.fixing_times, [t]]))
    grid = TimeGrid.build(fixings, sim.dt_max, calibration_grid=model.required_times())
    idx = grid.fixing_index
    col_t = idx[t]
    mc = MonteCarlo(sim)
    draws = mc.draws_for(grid, model)
    s0 = model.spot
    up = model_regime_spot_bump(model, s0 * float(np.exp(spot_size)))
    dn = model_regime_spot_bump(model, s0 * float(np.exp(-spot_size)))
    s_up, s_dn = up.spot, dn.spot
    feats: list[FloatArray] = []
    tg: dict[str, list[FloatArray]] = {"value": [], "delta": [], "gamma": []}
    raw_tg: dict[str, list[FloatArray]] = {"value": [], "delta": [], "gamma": []}
    if vol_model is not None:
        tg["vega"] = []
        raw_tg["vega"] = []
    for p0, p1 in sim.chunk_ranges(grid.n_records, model.n_factors):
        base = model.simulate_chunk(grid, draws, p0, p1, sim.scheme)
        pb = product.payoff(base, idx)
        pu = product.payoff(up.simulate_chunk(grid, draws, p0, p1, sim.scheme), idx)
        pd_ = product.payoff(dn.simulate_chunk(grid, draws, p0, p1, sim.scheme), idx)
        s_t = np.exp(base.log_spot_at(col_t))
        ratio = s_t / s0
        feats.append(np.column_stack([base.log_spot_at(col_t), base.factors_at(col_t)]))
        d_raw = (pu - pd_) / (s_up - s_dn)
        c_up = 2.0 / ((s_up - s0) * (s_up - s_dn))
        c_dn = 2.0 / ((s0 - s_dn) * (s_up - s_dn))
        g_raw = c_up * pu - (c_up + c_dn) * pb + c_dn * pd_
        raw_tg["value"].append(pb)
        raw_tg["delta"].append(d_raw)
        raw_tg["gamma"].append(g_raw)
        tg["value"].append(pb)
        tg["delta"].append(d_raw / ratio)
        tg["gamma"].append(g_raw / ratio**2)
        if vol_model is not None:
            pv = product.payoff(vol_model.simulate_chunk(grid, draws, p0, p1, sim.scheme), idx)
            v_raw = (pv - pb) * 0.01 / vol_size
            raw_tg["vega"].append(v_raw)
            tg["vega"].append(v_raw)
    raw = np.concatenate(feats)
    mean = raw.mean(axis=0)
    scale = np.where(raw.std(axis=0) > 0, raw.std(axis=0), 1.0)
    x = _basis((raw - mean) / scale, degree)
    coefficients: dict[str, FloatArray] = {}
    r2: dict[str, float] = {}
    targets: dict[str, FloatArray] = {}
    for kind, parts in tg.items():
        y = np.concatenate(parts)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        resid = y - x @ beta
        coefficients[kind] = np.asarray(beta)
        var_y = float(y.var())
        r2[kind] = 1.0 - float(resid.var()) / var_y if var_y > 0 else 1.0
        targets[kind] = y
    return ConditionalGreeks(
        t,
        degree,
        mean,
        scale,
        coefficients,
        r2,
        (raw - mean) / scale,
        targets,
        {k: np.concatenate(v) for k, v in raw_tg.items()},
    )


# --------------------------------------------------------------------------------------------
# control variate on the difference
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ControlledSensitivity:
    """A bump sensitivity with a Black–Scholes control on the per-path difference."""

    raw: Sensitivity
    controlled: Sensitivity
    beta: float
    control_mean: float
    control_value: float
    variance_reduction: float


def cv_difference(
    engine: RiskEngine,
    product: EuropeanOption,
    state: RiskState,
    kind: str = "delta",
    size: float = 0.01,
    regime: str = "model",
) -> ControlledSensitivity:
    """``kind`` in ``("delta", "gamma", "vega")`` for a European option under the engine's
    model: the CRN bump difference per path, controlled by the Black–Scholes difference at the
    market implied vol of ``(K, T)`` on the same grid and normals, with the analytic Greek as
    the control's expectation (vega: sticky-leverage variant, control shifted by the same vol
    step).  ``regime`` selects the spot move of delta / gamma (``"model"`` or any §7.2 regime):
    under local vol the ``"model"`` move holds the local vol fixed in spot, so the up path runs
    at a lower vol than the down path and the pathwise difference *anti*-correlates with the
    Black–Scholes one on the deep in-the-money paths (β < 0, variance reduction ≈ 1.1 on the
    reference surface); the ``"sticky_moneyness"`` move keeps the smile in moneyness and is the
    Black–Scholes-like case the control was designed for."""
    if not isinstance(product, EuropeanOption):
        raise TypeError("the Black-Scholes control covers European options")
    surface = surface_of(state)
    fc = surface.forward_curve
    vol = float(surface.implied_vol(product.strike, product.T))
    r_q = fc.rate_curve, fc.dividend_curve
    K, T, cp = product.strike, product.T, product.cp
    S = state.spot

    def bs_model(spot: float, sig: float) -> Model:
        return BlackScholes(sig, ForwardCurve(spot, *r_q))

    base_model = engine.builder.build(state, "recalibrate")
    bound = product.with_discount(fc.rate_curve)
    mc = MonteCarlo(engine.sim)
    grid = mc.build_grid([bound], base_model)

    def pay(model: Model) -> FloatArray:
        res = mc.price(bound, model, grid=grid, keep_payoffs=True)
        return _pairs(np.asarray(res.payoffs), engine.sim.antithetic)

    if kind in ("delta", "gamma"):
        up_s, mode_up = _spot_state(state, regime, size)
        dn_s, mode_dn = _spot_state(state, regime, -size)
        s_up, s_dn = up_s.spot, dn_s.spot
        p_up = engine.priced(product, up_s, mode_up).payoffs
        p_dn = engine.priced(product, dn_s, mode_dn).payoffs
        c_up_p, c_dn_p = pay(bs_model(s_up, vol)), pay(bs_model(s_dn, vol))
        if kind == "delta":
            d = (p_up - p_dn) / (s_up - s_dn)
            c = (c_up_p - c_dn_p) / (s_up - s_dn)
            value = float(bs_delta(S, K, T, vol, *_rates(fc, T), cp))
            unit, name = "per unit spot", "delta"
        else:
            p0 = engine.priced(product, state, "recalibrate").payoffs
            c0 = pay(bs_model(S, vol))
            a, b = 2.0 / ((s_up - S) * (s_up - s_dn)), 2.0 / ((S - s_dn) * (s_up - s_dn))
            d = a * p_up - (a + b) * p0 + b * p_dn
            c = a * c_up_p - (a + b) * c0 + b * c_dn_p
            value = float(bs_gamma(S, K, T, vol, *_rates(fc, T)))
            unit, name = "per unit spot squared", "gamma"
    elif kind == "vega":
        from volsto.config import SurfacePerturbation

        vb, achieved = engine.perturbed_state(
            state, lambda s: SurfacePerturbation("parallel", {"size": s}), size, label="vega"
        )
        p_b = engine.priced(product, vb, "sticky_leverage").payoffs
        p0 = engine.priced(product, state, "recalibrate").payoffs
        d = (p_b - p0) * 0.01 / achieved
        c = (pay(bs_model(S, vol + achieved)) - pay(bs_model(S, vol))) * 0.01 / achieved
        value = float(bs_vega(S, K, T, vol, *_rates(fc, T))) * 0.01
        unit, name = "per vol point", "vega"
    else:
        raise ValueError("kind must be 'delta', 'gamma' or 'vega'")
    var_c = float(c.var(ddof=1))
    beta = float(np.cov(d, c, ddof=1)[0, 1] / var_c) if var_c > 0 else 0.0
    resid = d - beta * (c - value)
    raw = _sensitivity(f"{name}[raw]", d, unit, size, "central", regime=regime)
    ctrl = _sensitivity(f"{name}[cv]", resid, unit, size, "central", beta=beta, regime=regime)
    return ControlledSensitivity(
        raw,
        ctrl,
        beta,
        float(c.mean()),
        value,
        float(d.var(ddof=1) / resid.var(ddof=1)) if resid.var(ddof=1) > 0 else float("inf"),
    )


def _rates(fc: ForwardCurve, T: float) -> tuple[float, float]:
    """Flat-equivalent continuous rates to ``T`` for the Black–Scholes formulas."""
    r = -float(np.log(fc.rate_curve.df(T))) / T
    q = -float(np.log(fc.dividend_curve.df(T))) / T
    return r, q


__all__ = [
    "ConditionalGreeks",
    "ControlledSensitivity",
    "compare",
    "conditional_greeks",
    "cv_difference",
    "lr_delta",
    "lr_vega",
]
