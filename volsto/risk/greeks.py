"""Delta regimes, vega variants, theta split and cross-Greeks (SPEC v2 §7.2, §7.3, §7.7).

Delta regimes (smile in ``k = ln(K/S)``, ``Δ = ln(S0_new/S0_old)``, ``s_T`` the ATM skew of the
surface): ``"model"`` bumps the spot with the leverage held fixed in spot and the factors at
zero — the hedger's delta; ``"sticky_strike"`` ``σ_new(k, T) = σ_old(k + Δ, T)``;
``"sticky_moneyness"`` ``σ_new(k, T) = σ_old(k, T)``; ``"sticky_skew"`` ``σ_old(k, T) + s_T Δ``;
``"sticky_local_vol"`` ``σ_old(k, T) + 2 s_T Δ`` (Derman's sticky-implied-tree regime, SSR = 2,
reference only).  All but ``"model"`` rebuild the surface, recalibrate the leverage (cached) and
reprice under CRN.  Delta and gamma are the central three-point differences in log-spot
(``1%`` by default): ``δ = (P₊ − P₋)/(S₊ − S₋)``, ``Γ = [(P₊ − P₀)/(S₊ − S₀) − (P₀ − P₋)/(S₀ −
S₋)] / (½(S₊ − S₋))``.  In a flat Black–Scholes world every regime coincides (tested).

Vega: parallel +1 vol point at every ``(K, T)`` (forward difference); ``"recalibrated"``
(default) refits the leverage, ``"sticky_leverage"`` holds ``L`` and moves only ``ξ₀`` with the
bumped strip; the difference is the leverage vega, a study quantity.

Theta (one business day, ``1/252``): the product aged by ``dt`` (fixings moved earlier; a fixing
inside the roll window raises) priced with the surface held in ``(K, absolute expiry)`` — the
``"roll"`` perturbation — factors at zero, forward rolled; split into pure decay (surface held in
time-to-maturity, zero rates and dividends), carry (the rates/dividends part of the decay) and
roll-down (the surface roll), see :func:`theta`.  BS identity ``θ + ½σ²S²Γ + rS δ − rP = 0``
(no dividends) is the test.

Cross-Greeks (§7.7): vanna as ``∂vega/∂ln S`` and as ``∂delta/∂σ`` (equal only in BS), volga,
charm and veta (one-business-day rolls of delta and vega), rho (+1 bp rates), repo delta (+1 bp
in q), and the cross terms ``∂delta/∂ρ_SX1``, ``∂delta/∂ρ_SX2``, ``∂delta/∂ν`` (recalibrated).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from volsto.config import SurfacePerturbation
from volsto.market.surface import ArbitrageError
from volsto.products.base import Product
from volsto.risk.engine import RiskEngine, RiskState, Sensitivity, default_params_bump

REGIMES = ("model", "sticky_strike", "sticky_moneyness", "sticky_skew", "sticky_local_vol")
VEGA_VARIANTS = ("recalibrated", "sticky_leverage")
BUSINESS_DAY = 1.0 / 252.0
#: the sticky-skew / sticky-local-vol shifts evaluate the ATM skew at max(T, SKEW_T_MIN): the
#: first vega pillar (1m); below it the SSVI skew diverges like 1/sqrt(T)
SKEW_T_MIN = 1.0 / 12.0


def _spot_state(state: RiskState, regime: str, h: float) -> tuple[RiskState, str]:
    """The state after a log-spot move ``h`` under a regime and the builder mode to price it."""
    spot = state.spot * float(np.exp(h))
    moved = state.with_spot(spot, label=f"{regime} S={spot:.4f}")
    if regime == "model":
        return moved, "model"
    if regime == "sticky_moneyness":
        return moved, "recalibrate"
    if regime == "sticky_strike":
        pert = SurfacePerturbation("shift_k", {"delta": h})
    elif regime == "sticky_skew":
        pert = SurfacePerturbation("atm_shift", {"delta": h, "factor": 1.0, "t_min": SKEW_T_MIN})
    elif regime == "sticky_local_vol":
        pert = SurfacePerturbation("atm_shift", {"delta": h, "factor": 2.0, "t_min": SKEW_T_MIN})
    else:
        raise ValueError(f"regime must be one of {REGIMES}")
    return moved.with_perturbation(pert, label=f"{regime} S={spot:.4f}"), "recalibrate"


def spot_state(state: RiskState, regime: str, h: float) -> tuple[RiskState, str]:
    """Public name of :func:`_spot_state` (the state after a log-spot move ``h`` under a §7.2
    regime and the builder mode that prices it), for the hedging engine and the studies."""
    return _spot_state(state, regime, h)


def delta_gamma(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    regime: str = "model",
    size: float = 0.01,
) -> tuple[Sensitivity, Sensitivity]:
    """Central three-point delta and gamma (per unit spot) under a delta regime."""
    if regime not in REGIMES:
        raise ValueError(f"regime must be one of {REGIMES}")
    up, mode_up = _spot_state(state, regime, size)
    dn, mode_dn = _spot_state(state, regime, -size)
    s0, s_up, s_dn = state.spot, up.spot, dn.spot
    base_mode = "recalibrate"
    delta = engine.combination(
        f"delta[{regime}]",
        product,
        [(up, mode_up, 1.0 / (s_up - s_dn)), (dn, mode_dn, -1.0 / (s_up - s_dn))],
        unit="per unit spot",
        size=size,
        scheme="central",
        extra={"regime": regime},
    )
    hp, hm = s_up - s0, s0 - s_dn
    scale = 2.0 / (s_up - s_dn)
    gamma = engine.combination(
        f"gamma[{regime}]",
        product,
        [
            (up, mode_up, scale / hp),
            (state, base_mode, -scale * (1.0 / hp + 1.0 / hm)),
            (dn, mode_dn, scale / hm),
        ],
        unit="per unit spot²",
        size=size,
        scheme="central",
        extra={"regime": regime},
    )
    return delta, gamma


def delta_table(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    size: float = 0.01,
    regimes: tuple[str, ...] = REGIMES,
) -> pd.DataFrame:
    """All regimes side by side (SPEC v2 §7.2)."""
    rows = []
    for regime in regimes:
        d, g = delta_gamma(engine, product, state, regime, size)
        rows.append(
            {
                "regime": regime,
                "delta": d.value,
                "delta_stderr": d.stderr,
                "gamma": g.value,
                "gamma_stderr": g.stderr,
            }
        )
    return pd.DataFrame(rows)


def vega(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    variant: str = "recalibrated",
    size: float = 0.01,
) -> Sensitivity:
    """Parallel vega per vol point (forward difference) under a leverage variant."""
    if variant not in VEGA_VARIANTS:
        raise ValueError(f"variant must be one of {VEGA_VARIANTS}")
    bumped, achieved = engine.perturbed_state(
        state, lambda s: SurfacePerturbation("parallel", {"size": s}), size, label="vega"
    )
    mode = "recalibrate" if variant == "recalibrated" else "sticky_leverage"
    per_vp = 0.01 / achieved
    return engine.combination(
        f"vega[{variant}]",
        product,
        [(bumped, mode, per_vp), (state, "recalibrate", -per_vp)],
        unit="per vol point",
        size=achieved,
        scheme="forward",
        extra={"variant": variant, "requested": size},
    )


@dataclass(frozen=True)
class ThetaReport:
    """Total theta and its split; all per year (multiply by ``dt`` for the one-day P&L)."""

    total: Sensitivity
    decay: Sensitivity
    carry: Sensitivity
    roll_down: Sensitivity
    dt: float

    def as_dict(self) -> dict[str, float]:
        return {
            "theta": self.total.value,
            "theta_stderr": self.total.stderr,
            "decay": self.decay.value,
            "carry": self.carry.value,
            "roll_down": self.roll_down.value,
        }


def theta(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    dt: float = BUSINESS_DAY,
    *,
    aged: Product | None = None,
) -> ThetaReport:
    """Theta per year from a one-business-day roll (SPEC v2 §7.3).

    ``total = [P(aged, rolled surface) − P(base)] / dt`` where the rolled surface holds implied
    vols per ``(K, absolute expiry)`` (the ``"roll"`` perturbation), ``decay = [P(aged, surface
    held in residual maturity) − P(base)] / dt`` with the rates and dividends set to zero (pure
    time decay), ``carry`` = the same difference with the actual curves minus ``decay`` (the
    rates/dividends part), ``roll_down = total − decay − carry``.  Aged products live on their own
    grid, so the errors of the two prices add in quadrature (no CRN).

    ``aged`` (M10 Part 3) is the product seen ``dt`` later with the state held, default
    ``product.aged(dt)`` — which raises for a fixing inside the roll window; a daily-fixed
    product passes its seasoned form with the window's fixings at the held spot
    (:func:`volsto.products.seasoning.season` on the history extended by the held close).
    """
    if aged is None:
        aged = product.aged(dt)
    rolled = state.with_perturbation(SurfacePerturbation("roll", {"dt": dt}), label="rolled")
    zero = state.with_zero_rates()
    total = engine.paired(
        "theta",
        [(aged, rolled, "recalibrate", 1.0 / dt), (product, state, "recalibrate", -1.0 / dt)],
        unit="per year",
        size=dt,
        scheme="forward",
    )
    decay = engine.paired(
        "theta.decay",
        [(aged, zero, "recalibrate", 1.0 / dt), (product, zero, "recalibrate", -1.0 / dt)],
        unit="per year",
        size=dt,
        scheme="forward",
    )
    held = engine.paired(
        "theta.held",
        [(aged, state, "recalibrate", 1.0 / dt), (product, state, "recalibrate", -1.0 / dt)],
        unit="per year",
        size=dt,
        scheme="forward",
    )
    carry = Sensitivity(
        "theta.carry",
        held.value - decay.value,
        float(np.hypot(held.stderr, decay.stderr)),
        "per year",
        dt,
        "forward",
        (state.label,),
        engine.sim.n_paths,
    )
    roll_down = Sensitivity(
        "theta.roll_down",
        total.value - held.value,
        float(np.hypot(total.stderr, held.stderr)),
        "per year",
        dt,
        "forward",
        (state.label,),
        engine.sim.n_paths,
    )
    return ThetaReport(total, decay, carry, roll_down, dt)


def rate_sensitivities(
    engine: RiskEngine, product: Product, state: RiskState, rate_bp: float = 1e-4
) -> dict[str, Sensitivity]:
    """``rho`` and ``repo_delta``: forward differences for a parallel ``+rate_bp`` shift of the
    rate and of the dividend curve (per 1 bp by default), recalibrated states — the rate items of
    :func:`cross_greeks`, on their own for the P&L attribution (the rates step needs nothing
    else)."""
    out: dict[str, Sensitivity] = {}
    out["rho"] = engine.combination(
        "rho",
        product,
        [
            (state.with_rate_shift(dr=rate_bp, label="r+1bp"), "recalibrate", 1.0),
            (state, "recalibrate", -1.0),
        ],
        unit="per bp of rates",
        size=rate_bp,
        scheme="forward",
    )
    out["repo_delta"] = engine.combination(
        "repo_delta",
        product,
        [
            (state.with_rate_shift(dq=rate_bp, label="q+1bp"), "recalibrate", 1.0),
            (state, "recalibrate", -1.0),
        ],
        unit="per bp of repo",
        size=rate_bp,
        scheme="forward",
    )
    return out


def cross_greeks(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    *,
    spot_size: float = 0.01,
    vol_size: float = 0.01,
    dt: float = BUSINESS_DAY,
    rate_bp: float = 1e-4,
    params: tuple[str, ...] = ("rho_SX1", "rho_SX2", "nu"),
) -> dict[str, Sensitivity]:
    """Vanna (both forms), volga, charm, veta, rho, repo delta and the parameter cross terms of
    the delta (SPEC v2 §7.7), under the ``"model"`` regime for the spot moves."""
    out: dict[str, Sensitivity] = {}
    par = lambda s: SurfacePerturbation("parallel", {"size": s})  # noqa: E731
    up_s, _ = _spot_state(state, "model", spot_size)
    dn_s, _ = _spot_state(state, "model", -spot_size)
    # vanna 1: ∂vega/∂ln S — recalibrated vega at the moved spots (the moved base states are the
    # "model" regime; the vega-bumped states recalibrate at the moved spot)
    vp_up, a1 = engine.perturbed_state(up_s, par, vol_size, label="vega@S+")
    vp_dn, a2 = engine.perturbed_state(dn_s, par, vol_size, label="vega@S-")
    c_up, c_dn = 0.01 / a1, 0.01 / a2
    out["vanna_dvega_dlnS"] = engine.combination(
        "vanna[dvega/dlnS]",
        product,
        [
            (vp_up, "recalibrate", c_up / (2 * spot_size)),
            (up_s, "model", -c_up / (2 * spot_size)),
            (vp_dn, "recalibrate", -c_dn / (2 * spot_size)),
            (dn_s, "model", c_dn / (2 * spot_size)),
        ],
        unit="per vol point per unit ln S",
        size=spot_size,
        scheme="central",
    )
    # vanna 2: ∂delta/∂σ — model delta at the vega-bumped surface minus the base delta
    vb, a = engine.perturbed_state(state, par, vol_size, label="vega")
    vb_up, _ = _spot_state(vb, "model", spot_size)
    vb_dn, _ = _spot_state(vb, "model", -spot_size)
    ds = up_s.spot - dn_s.spot
    per_vp = 0.01 / a
    out["vanna_ddelta_dsigma"] = engine.combination(
        "vanna[ddelta/dsigma]",
        product,
        [
            (vb_up, "model", per_vp / ds),
            (vb_dn, "model", -per_vp / ds),
            (up_s, "model", -per_vp / ds),
            (dn_s, "model", per_vp / ds),
        ],
        unit="per unit spot per vol point",
        size=vol_size,
        scheme="forward",
    )
    # volga: three-point second difference in the parallel vol bump at the achieved sizes
    # 0, a, a3 (f'' ≈ 2[(P(a3) − P(a))/(a3 − a) − (P(a) − P(0))/a]/a3), per (vol point)²
    vb2, a3 = engine.perturbed_state(state, par, 2 * vol_size, label="vega x2")
    if a3 <= a:
        raise ArbitrageError("volga: the double vol bump could not exceed the single one")
    c2 = 2.0 / ((a3 - a) * a3) * 0.01**2
    c1 = -2.0 * (1.0 / ((a3 - a) * a3) + 1.0 / (a * a3)) * 0.01**2
    c0 = 2.0 / (a * a3) * 0.01**2
    out["volga"] = engine.combination(
        "volga",
        product,
        [(vb2, "recalibrate", c2), (vb, "recalibrate", c1), (state, "recalibrate", c0)],
        unit="per vol point²",
        size=vol_size,
        scheme="central",
        extra={"achieved": (a, a3)},
    )
    # charm and veta: one-business-day roll of delta and vega (aged product on its own grid)
    aged = product.aged(dt)
    rolled = state.with_perturbation(SurfacePerturbation("roll", {"dt": dt}), label="rolled")
    r_up, _ = _spot_state(rolled, "model", spot_size)
    r_dn, _ = _spot_state(rolled, "model", -spot_size)
    d_rolled = engine.combination(
        "delta@rolled",
        aged,
        [(r_up, "model", 1.0 / ds), (r_dn, "model", -1.0 / ds)],
        unit="per unit spot",
        size=spot_size,
        scheme="central",
    )
    d_base = engine.combination(
        "delta@base",
        product,
        [(up_s, "model", 1.0 / ds), (dn_s, "model", -1.0 / ds)],
        unit="per unit spot",
        size=spot_size,
        scheme="central",
    )
    out["charm"] = Sensitivity(
        "charm",
        (d_rolled.value - d_base.value) / dt,
        float(np.hypot(d_rolled.stderr, d_base.stderr)) / dt,
        "delta per year",
        dt,
        "forward",
        (state.label, rolled.label),
        engine.sim.n_paths,
    )
    rv, a4 = engine.perturbed_state(rolled, par, vol_size, label="vega@rolled")
    v_rolled = engine.combination(
        "vega@rolled",
        aged,
        [(rv, "recalibrate", 0.01 / a4), (rolled, "recalibrate", -0.01 / a4)],
        unit="per vol point",
        size=a4,
        scheme="forward",
    )
    v_base = engine.combination(
        "vega@base",
        product,
        [(vb, "recalibrate", per_vp), (state, "recalibrate", -per_vp)],
        unit="per vol point",
        size=a,
        scheme="forward",
    )
    out["veta"] = Sensitivity(
        "veta",
        (v_rolled.value - v_base.value) / dt,
        float(np.hypot(v_rolled.stderr, v_base.stderr)) / dt,
        "vega per year",
        dt,
        "forward",
        (state.label, rolled.label),
        engine.sim.n_paths,
    )
    # rho and repo delta (per 1 bp)
    out.update(rate_sensitivities(engine, product, state, rate_bp))
    # cross terms: model delta as a function of the vol-sto parameters (recalibrated)
    for name in params:
        h = default_params_bump(name, state.spec.model)
        base_val = float(getattr(state.spec.model, name))
        p_up = state.with_params(label=f"{name}+", **{name: base_val + h})
        p_dn = state.with_params(label=f"{name}-", **{name: base_val - h})
        deltas = []
        for ps in (p_up, p_dn):
            su, _ = _spot_state(ps, "model", spot_size)
            sd, _ = _spot_state(ps, "model", -spot_size)
            deltas.append(
                engine.combination(
                    f"delta@{ps.label}",
                    product,
                    [(su, "model", 1.0 / ds), (sd, "model", -1.0 / ds)],
                    unit="per unit spot",
                    size=spot_size,
                    scheme="central",
                )
            )
        out[f"ddelta_d{name}"] = Sensitivity(
            f"ddelta/d{name}",
            (deltas[0].value - deltas[1].value) / (2 * h),
            float(np.hypot(deltas[0].stderr, deltas[1].stderr)) / (2 * h),
            f"delta per unit {name}",
            h,
            "central",
            (p_up.label, p_dn.label),
            engine.sim.n_paths,
        )
    return out
