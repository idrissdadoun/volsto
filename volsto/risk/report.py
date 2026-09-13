"""RiskReport (SPEC v2 §7.13): one object per product holding every sensitivity with its
standard error, the bump specification (size, scheme, states), the cache keys of every
recalibration and the budget (recalibrations, cache misses, pricings, wall clock), with
``to_dataframe()`` / ``to_excel()``.

Contents of a full report: price; delta and gamma under the five regimes; parallel vega
(recalibrated and sticky-leverage); theta split; cross-Greeks (§7.7); vega-T waves,
projections and tents; forward-variance ladder (recalibrated and sticky-leverage) with its
parallel bump; skew and curvature ladders; the seven parameter sensitivities (ν also sticky);
and the product-specific risks that apply (fixing risk for forward starts and cliquets,
barrier sensitivities and ``∂P(KO)/∂ln S`` for knock-out variance swaps and vol knock-out
puts, the realised-variance exposure profile for realised-variance products).

Two statements the desk must keep in mind when reading the vega term structure:

(i) **wave projections equal single-pillar bumps only to first order** — the projection at
``T_j`` is ``wave_j − wave_{j−1}``, and a single tent differs from it by the cross term of the
recalibration and of the vega convexity across the pillars bumped together (tested within 2
stderr on the reference surface);

(ii) **an expiry between two pillars shows vega in both neighbouring projections**, in
proportion to its distance from each, because the bump lives at the pillars and the
interpolation ramps it linearly in between: a 9-month option carries vega on the 6m and 1y
projections when 9m is not a pillar.

Budget: a full report of the default configuration is on the order of 80–100 recalibrations
(9 pillars × 3 ladders + 9 waves, 20 forward-variance buckets, 4 regimes × 2 spots, 7 × 2
parameters, vega and cross-Greek states); ``budget`` reports the count and the wall clock — the
viewer precompute budget.  Checked by ``tests/test_risk_report.py``.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from volsto.products.base import Product
from volsto.products.cliquet import AdditiveCliquet
from volsto.products.conditional_variance import KnockOutVarianceSwap, RealisedVarianceSchedule
from volsto.products.forward_start import ForwardStartOption, ForwardStartStraddle
from volsto.products.variance import VarianceSwap
from volsto.products.vko import VolKnockOutPut
from volsto.risk.engine import RiskEngine, RiskState, Sensitivity
from volsto.risk.greeks import REGIMES, cross_greeks, delta_gamma, theta, vega
from volsto.risk.ladders import (
    PILLARS,
    curvature_T,
    default_buckets,
    fwd_var_ladder,
    skew_T,
    vega_T,
)
from volsto.risk.product_risk import (
    barrier_sensitivity,
    fixing_risk,
    ko_probability_delta,
    realised_variance_exposure,
)
from volsto.risk.volsto_sens import PARAMS, parameter_sensitivities

SECTIONS: tuple[str, ...] = (
    "delta",
    "vega",
    "theta",
    "cross",
    "vega_T",
    "fwd_var",
    "fwd_var_sticky",
    "skew",
    "curvature",
    "params",
    "product",
)


@dataclass
class RiskReport:
    """See the module docstring."""

    product: str
    state: str
    price: float
    price_stderr: float
    rows: list[dict[str, Any]] = field(default_factory=list)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    budget: dict[str, float] = field(default_factory=dict)
    cache_keys: list[str] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)

    def add(self, group: str, s: Sensitivity, **more: Any) -> None:
        self.rows.append(
            {
                "group": group,
                "name": s.name,
                "value": s.value,
                "stderr": s.stderr,
                "unit": s.unit,
                "size": s.size,
                "scheme": s.scheme,
                "states": " | ".join(s.states),
                **{k: v for k, v in s.extra.items() if isinstance(v, int | float | str)},
                **more,
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.rows)
        head = ["group", "name", "value", "stderr", "unit", "size", "scheme", "states"]
        rest = [c for c in frame.columns if c not in head]
        return frame[head + rest]

    def to_excel(self, path: str | Path) -> Path:
        """Sheets: ``sensitivities`` (flat), ``budget``, and one per table."""
        path = Path(path)
        with pd.ExcelWriter(path) as xw:
            self.to_dataframe().to_excel(xw, sheet_name="sensitivities", index=False)
            pd.DataFrame(
                [
                    {
                        "product": self.product,
                        "state": self.state,
                        "price": self.price,
                        "price_stderr": self.price_stderr,
                        **self.budget,
                    }
                ]
            ).to_excel(xw, sheet_name="budget", index=False)
            for name, table in self.tables.items():
                table.to_excel(xw, sheet_name=name[:31], index=False)
            pd.DataFrame({"cache_key": self.cache_keys}).to_excel(
                xw, sheet_name="cache_keys", index=False
            )
        return path

    def summary(self) -> str:
        b = self.budget
        return (
            f"RiskReport[{self.product}] price {self.price:.6g} ± {self.price_stderr:.2g}; "
            f"{len(self.rows)} sensitivities; {b.get('recalibrations', 0):.0f} recalibrations "
            f"({b.get('cache_misses', 0):.0f} cache misses), {b.get('pricings', 0):.0f} pricings, "
            f"{b.get('wall_clock_s', 0):.0f} s wall clock"
        )


def risk_report(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    *,
    sections: Sequence[str] = SECTIONS,
    pillars: Sequence[float] = PILLARS,
    buckets: Sequence[tuple[float, float]] | None = None,
    regimes: Sequence[str] = REGIMES,
    params: Sequence[str] = PARAMS,
    size: float = 0.01,
) -> RiskReport:
    """Build the report; ``sections`` selects the blocks (all by default)."""
    unknown = set(sections) - set(SECTIONS)
    if unknown:
        raise ValueError(f"unknown sections {sorted(unknown)}")
    t0 = time.perf_counter()
    n_cal0 = engine.n_calibrations
    p = engine.price(product, state)
    rep = RiskReport(repr(product), state.label, p.mean, p.stderr)
    rep.settings = {
        "pillars": list(pillars),
        "buckets": list(buckets) if buckets is not None else list(default_buckets()),
        "regimes": list(regimes),
        "params": list(params),
        "bump": size,
        "n_paths": engine.sim.n_paths,
        "seed": engine.sim.seed,
    }
    bk = tuple(buckets) if buckets is not None else default_buckets()

    if "delta" in sections:
        rows = []
        for regime in regimes:
            d, g = delta_gamma(engine, product, state, regime, size)
            rep.add("delta", d, regime=regime)
            rep.add("gamma", g, regime=regime)
            rows.append(
                {
                    "regime": regime,
                    "delta": d.value,
                    "delta_stderr": d.stderr,
                    "gamma": g.value,
                    "gamma_stderr": g.stderr,
                }
            )
        rep.tables["delta_regimes"] = pd.DataFrame(rows)
    if "vega" in sections:
        for variant in ("recalibrated", "sticky_leverage"):
            rep.add("vega", vega(engine, product, state, variant, size), variant=variant)
    if "theta" in sections:
        th = theta(engine, product, state)
        for s in (th.total, th.decay, th.carry, th.roll_down):
            rep.add("theta", s)
    if "cross" in sections:
        for key, s in cross_greeks(engine, product, state, spot_size=size, vol_size=size).items():
            rep.add("cross", s, key=key)
    if "vega_T" in sections:
        vt = vega_T(engine, product, state, pillars, size, with_tents=True)
        for w, pr, t in zip(vt.waves, vt.projections, vt.tents, strict=True):
            rep.add("vega_T", w)
            rep.add("vega_T", pr)
            rep.add("vega_T", t)
        rep.tables["vega_T"] = vt.as_frame()
    for sec, variant in (("fwd_var", "recalibrated"), ("fwd_var_sticky", "sticky_leverage")):
        if sec in sections:
            lad = fwd_var_ladder(engine, product, state, bk, size, variant)
            for e in lad.entries:
                rep.add(sec, e, variant=variant)
            assert lad.total is not None and lad.parallel is not None
            rep.add(sec, lad.total, variant=variant)
            rep.add(sec, lad.parallel, variant=variant)
            rep.tables[sec] = lad.as_frame()
    for sec, fn in (("skew", skew_T), ("curvature", curvature_T)):
        if sec in sections:
            lad = fn(engine, product, state, pillars, size)
            for e in lad.entries:
                rep.add(sec, e)
            assert lad.total is not None and lad.parallel is not None
            rep.add(sec, lad.total)
            rep.add(sec, lad.parallel)
            rep.tables[sec] = lad.as_frame()
    if "params" in sections:
        table = parameter_sensitivities(engine, product, state, params)
        rep.tables["params"] = table
        for _, r in table.iterrows():
            rep.rows.append(
                {
                    "group": "params",
                    "name": f"d/d{r['param']}[recalibrated]",
                    "value": r["value"],
                    "stderr": r["stderr"],
                    "unit": f"per unit {r['param']}",
                    "size": r["bump"],
                    "scheme": r["scheme"],
                    "states": "",
                }
            )
            if pd.notna(r["value_sticky"]):
                rep.rows.append(
                    {
                        "group": "params",
                        "name": f"d/d{r['param']}[sticky_leverage]",
                        "value": r["value_sticky"],
                        "stderr": r["stderr_sticky"],
                        "unit": f"per unit {r['param']}",
                        "size": r["bump"],
                        "scheme": r["scheme"],
                        "states": "",
                    }
                )
    if "product" in sections:
        if isinstance(product, ForwardStartOption | ForwardStartStraddle | AdditiveCliquet):
            rep.tables["fixing_risk"] = fixing_risk(engine, product, state, size=size)
        if isinstance(product, KnockOutVarianceSwap | VolKnockOutPut):
            for key, s in barrier_sensitivity(engine, product, state).items():
                rep.add("barrier", s, key=key)
            rep.add("barrier", ko_probability_delta(engine, product, state, size))
        if isinstance(product, RealisedVarianceSchedule | VarianceSwap):
            model = engine.builder.build(state, "recalibrate")
            rep.tables["realised_variance_exposure"] = realised_variance_exposure(
                product, model, engine.sim
            )
    rep.budget = {
        **engine.budget(),
        "report_recalibrations": float(engine.n_calibrations - n_cal0),
        "report_wall_clock_s": time.perf_counter() - t0,
    }
    rep.cache_keys = list(engine.builder.cache_keys)
    return rep


__all__ = ["SECTIONS", "RiskReport", "risk_report"]
