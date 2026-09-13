"""P&L attribution (SPEC v2 §7.12): ``explain(engine, product, state_0, state_1)``.

Sequential CRN revaluation from ``state_0`` to ``state_1`` in the order

1. **spot** — ``state_0`` with ``state_1``'s spot (the surface configuration re-evaluated at the
   new spot, i.e. the ``"sticky_moneyness"`` move of the state's own surface); explained by the
   sticky-moneyness delta and gamma of ``state_0``: ``Δ·δS + ½Γ·δS²``; the remainder is the
   third-order (speed) term, reported;
2. **rates** — ``state_1``'s rate and dividend curves; explained by rho and the repo delta
   (``+1 bp`` sensitivities) times the parallel shifts;
3. **surface** — ``state_1``'s surface configuration and perturbation; explained by the parallel
   vega times the average ATM change over the pillars (``detail="parallel"``) or, with
   ``detail="ladders"``, by the vega-T tents times the per-pillar ATM changes, the skew ladder
   times the 90/110 skew changes and the curvature ladder times the butterfly changes;
4. **params** — ``state_1``'s model parameters; explained by the recalibrated parameter
   sensitivities times the changes;
5. **factors** — ``state_1``'s initial factor state (actual only: no first-order Greek is kept
   for ``x₀``);
6. **time** — the product aged by ``dt`` at ``state_1``; explained by the held-surface theta
   (decay + carry; the roll-down sits in the surface step because ``state_1``'s surface is the
   observed end surface) times ``dt``.

Every actual step is a paired CRN difference with its standard error; the residual of a step is
``actual − explained`` and the total residual ``actual P&L − Σ explained``.  Greeks are taken at
``state_0`` (start-of-day risk), so cross terms between steps land in the residuals.  Checked by
``tests/test_risk_attribution.py`` (pure spot move: residual at the third-order level, reported;
pure parallel vol move: residual within 2 stderr).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.market.surface import ImpliedSurface
from volsto.products.base import Product
from volsto.risk.engine import RiskEngine, RiskState, Sensitivity, surface_of
from volsto.risk.greeks import cross_greeks, delta_gamma, theta
from volsto.risk.greeks import vega as parallel_vega
from volsto.risk.ladders import K90, K110, PILLARS, curvature_T, skew_T, vega_T
from volsto.risk.volsto_sens import PARAMS, parameter_sensitivity

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class Step:
    name: str
    actual: float
    actual_stderr: float
    explained: float
    detail: dict[str, float]

    @property
    def residual(self) -> float:
        return self.actual - self.explained


@dataclass(frozen=True)
class Explain:
    steps: tuple[Step, ...]
    total: Sensitivity
    price_0: float
    price_1: float

    @property
    def explained(self) -> float:
        return float(sum(s.explained for s in self.steps))

    @property
    def residual(self) -> float:
        return self.total.value - self.explained

    def as_frame(self) -> pd.DataFrame:
        rows = [
            {
                "step": s.name,
                "actual": s.actual,
                "actual_stderr": s.actual_stderr,
                "explained": s.explained,
                "residual": s.residual,
            }
            for s in self.steps
        ]
        rows.append(
            {
                "step": "total",
                "actual": self.total.value,
                "actual_stderr": self.total.stderr,
                "explained": self.explained,
                "residual": self.residual,
            }
        )
        return pd.DataFrame(rows)


def _actual(
    engine: RiskEngine, product: Product, a: RiskState, b: RiskState, name: str
) -> Sensitivity:
    return engine.combination(
        name,
        product,
        [(b, "recalibrate", 1.0), (a, "recalibrate", -1.0)],
        unit="price",
        size=0.0,
        scheme="revaluation",
    )


def _skew_90_110(surface: ImpliedSurface, ps: FloatArray) -> FloatArray:
    k90, k110 = np.full(ps.size, K90), np.full(ps.size, K110)
    return np.asarray(surface.implied_vol_k(k90, ps)) - np.asarray(surface.implied_vol_k(k110, ps))


def _butterfly_90_110(surface: ImpliedSurface, ps: FloatArray) -> FloatArray:
    k90, k110 = np.full(ps.size, K90), np.full(ps.size, K110)
    wings = 0.5 * (
        np.asarray(surface.implied_vol_k(k90, ps)) + np.asarray(surface.implied_vol_k(k110, ps))
    )
    return np.asarray(wings - np.asarray(surface.atm_vol(ps)), dtype=np.float64)


def _mean_rate(state: RiskState) -> tuple[float, float]:
    m = state.spec.market
    return float(np.mean(m.rate_curve.rates)), float(np.mean(m.dividend_curve.rates))


def explain(
    engine: RiskEngine,
    product: Product,
    state_0: RiskState,
    state_1: RiskState,
    *,
    dt: float = 0.0,
    detail: str = "parallel",
    pillars: tuple[float, ...] = PILLARS,
    size: float = 0.01,
) -> Explain:
    """Sequential attribution of ``P(state_1, aged by dt) − P(state_0)`` (module docstring)."""
    if detail not in ("parallel", "ladders"):
        raise ValueError("detail must be 'parallel' or 'ladders'")
    steps: list[Step] = []
    surf0, surf1 = surface_of(state_0), surface_of(state_1)

    # 1. spot
    s_a = state_0.with_spot(state_1.spot, label="explain:spot")
    if state_1.spot != state_0.spot:
        act = _actual(engine, product, state_0, s_a, "explain.spot")
        d, g = delta_gamma(engine, product, state_0, "sticky_moneyness", size)
        ds = state_1.spot - state_0.spot
        det = {"delta": d.value * ds, "gamma": 0.5 * g.value * ds * ds}
        steps.append(Step("spot", act.value, act.stderr, det["delta"] + det["gamma"], det))

    # 2. rates
    m1 = state_1.spec.market
    s_b = RiskState(
        dataclasses.replace(
            s_a.spec,
            market=dataclasses.replace(
                s_a.spec.market, rate_curve=m1.rate_curve, dividend_curve=m1.dividend_curve
            ),
        ),
        s_a.x0,
        "explain:rates",
    )
    if s_b.key != s_a.key:
        act = _actual(engine, product, s_a, s_b, "explain.rates")
        cg = cross_greeks(engine, product, state_0, spot_size=size, vol_size=size, params=())
        r0, q0 = _mean_rate(state_0)
        r1, q1 = _mean_rate(state_1)
        det = {
            "rho": cg["rho"].value * (r1 - r0) / 1e-4,
            "repo": cg["repo_delta"].value * (q1 - q0) / 1e-4,
        }
        steps.append(Step("rates", act.value, act.stderr, det["rho"] + det["repo"], det))

    # 3. surface
    s_c = RiskState(
        dataclasses.replace(
            s_b.spec, surface=state_1.spec.surface, perturbation=state_1.spec.perturbation
        ),
        s_b.x0,
        "explain:surface",
    )
    if s_c.key != s_b.key:
        act = _actual(engine, product, s_b, s_c, "explain.surface")
        ps = np.asarray(pillars, dtype=np.float64)
        d_atm = np.asarray(surf1.atm_vol(ps)) - np.asarray(surf0.atm_vol(ps))
        det = {}
        if detail == "parallel":
            v = parallel_vega(engine, product, state_0, "recalibrated", size)
            det["parallel_vega"] = v.value * float(d_atm.mean()) / 0.01
        else:
            vt = vega_T(engine, product, state_0, pillars, size, with_tents=True)
            det["vega_T"] = float(sum(t.value * dv / 0.01 for t, dv in zip(vt.tents, d_atm)))
            d_skew = _skew_90_110(surf1, ps) - _skew_90_110(surf0, ps)
            d_fly = _butterfly_90_110(surf1, ps) - _butterfly_90_110(surf0, ps)
            sk = skew_T(engine, product, state_0, pillars, size)
            cv = curvature_T(engine, product, state_0, pillars, size)
            det["skew_T"] = float(sum(e.value * x / 0.01 for e, x in zip(sk.entries, d_skew)))
            det["curvature_T"] = float(sum(e.value * x / 0.01 for e, x in zip(cv.entries, d_fly)))
        steps.append(Step("surface", act.value, act.stderr, float(sum(det.values())), det))

    # 4. parameters
    s_d = RiskState(
        dataclasses.replace(s_c.spec, model=state_1.spec.model), s_c.x0, "explain:params"
    )
    if s_d.key != s_c.key:
        act = _actual(engine, product, s_c, s_d, "explain.params")
        det = {}
        for name in PARAMS:
            dp = float(getattr(state_1.spec.model, name)) - float(getattr(state_0.spec.model, name))
            if dp != 0.0:
                det[name] = parameter_sensitivity(engine, product, state_0, name).value * dp
        steps.append(Step("params", act.value, act.stderr, float(sum(det.values())), det))

    # 5. factor state
    s_e = RiskState(s_d.spec, state_1.x0, "explain:factors")
    if s_e.x0 != s_d.x0:
        act = _actual(engine, product, s_d, s_e, "explain.factors")
        steps.append(Step("factors", act.value, act.stderr, 0.0, {}))

    # 6. time
    if dt > 0:
        aged = product.aged(dt)
        act = engine.paired(
            "explain.time",
            [(aged, s_e, "recalibrate", 1.0), (product, s_e, "recalibrate", -1.0)],
            unit="price",
            size=dt,
            scheme="revaluation",
        )
        th = theta(engine, product, state_0, dt)
        det = {"decay": th.decay.value * dt, "carry": th.carry.value * dt}
        steps.append(Step("time", act.value, act.stderr, det["decay"] + det["carry"], det))
        end_price = engine.price(aged, s_e).mean
    else:
        end_price = engine.price(product, s_e).mean
    p0 = engine.price(product, state_0).mean
    total = Sensitivity(
        "pnl",
        end_price - p0,
        float(np.sqrt(sum(s.actual_stderr**2 for s in steps))),
        "price",
        0.0,
        "revaluation",
        (state_0.label, state_1.label),
        engine.sim.n_paths,
    )
    return Explain(tuple(steps), total, p0, end_price)


__all__ = ["Explain", "Step", "explain"]
