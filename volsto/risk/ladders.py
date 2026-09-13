"""Term-structure ladders (SPEC v2 §7.4–7.6): vega by maturity in the desk "wave" convention,
the forward-variance vega ladder, and skew / curvature risk by pillar.

Pillars default to the surface's expiry pillars (1m, 2m, 3m, 6m, 9m, 1y, 18m, 2y, 3y) with
``tent_i(T)`` linear 0 → 1 → 0 over ``(T_{i−1}, T_i, T_{i+1})``, flat beyond the ends
(:func:`volsto.market.surface.tent`).  Every bump is an additive perturbation layer on the
surface; bumps that fail the arbitrage checks are halved and retried and the achieved size is
reported (the sensitivities are per unit of the *requested* size definition).

* :func:`vega_T` — pillar bump ``σ → σ + 0.01·tent_i(T)`` after checking the calendar
  condition on every ramp (``(σ + 0.01)² T_i ≤ σ² T_{i+1}``, i.e. ``T_{i+1}/T_i ≥ 1.10`` at σ = 20%;
  raise for pillar sets that violate it); wave ``j`` bumps every pillar with ``T_i ≤ T_j``
  together (the sum of tents ``1..j``), wave ``n`` is the parallel vega; the projection at
  ``T_j`` is ``wave_j − wave_{j−1}``, so projections sum to the parallel vega by construction.
  Projections equal single-pillar bumps only to first order (recalibration and vega convexity
  add a cross term) and an expiry between two pillars shows vega in both neighbouring
  projections in proportion to its distance from each, because the bump lives at the pillars
  and the interpolation ramps it.
* :func:`fwd_var_ladder` — buckets monthly to 1y and quarterly to 3y; bump ``ξ₀^T → (1+ε)ξ₀^T``
  on the bucket, propagated to the surface as ``dW(T) = ε ∫_{bucket ∩ [0,T]} ξ₀`` added to
  ``w(k, T)`` for every ``k`` (a parallel shift in total variance per maturity, skew preserved in
  total-variance terms); ``ε`` sized so the bucket's forward VS vol *measured on the log-contract
  strip* rises by 1 vp (:func:`bucket_epsilon` — the ``ξ₀``-scaling value overshoots by the smile
  factor, 1.17 on the reference surface, M5 finding).  Variants ``"recalibrated"`` (leverage
  refit) and ``"sticky_leverage"`` (``L`` held, ``ξ₀`` moved).  For a plain variance swap the
  ladder is model-independent: :func:`varswap_bucket_sensitivity`.
* :func:`skew_T` / :func:`curvature_T` — ``σ → σ + s·κ(k)·tent_i(T)`` with ``κ(k) = k_cap
  tanh(k/k_cap)`` (a rotation around the money saturating in the wings), ``s`` sized so
  ``σ(ln 0.9) − σ(ln 1.1)`` at ``T_i`` rises by 1 vp, and ``σ → σ + c·κ(k)²·tent_i(T)`` with ``c``
  sized so ``[σ(ln 0.9) + σ(ln 1.1)]/2 − σ(0)`` rises by 1 vp; units per vol point of 90/110 skew /
  butterfly at the pillar.

Checked by ``tests/test_risk_ladders.py`` (under local vol on the reference surface: projections
against single tents within 2 stderr, wave_n against the parallel vega, the variance-swap ladder
against its analytic sensitivity and its sum against the parallel bump, ATM vanilla with zero
skew vega and a 90% put with positive skew vega, a 90/110 risk reversal with zero curvature vega
and a strangle with positive curvature vega, the skew ladder summing to a global rotation).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from volsto.config import SurfacePerturbation
from volsto.market.surface import ImpliedSurface, perturbed_surface, saturated_k
from volsto.market.varswap import varswap_strike, xi0_curve
from volsto.products.base import Product
from volsto.products.variance import VarianceSwap
from volsto.risk.engine import RiskEngine, RiskState, Sensitivity, surface_of

PILLARS: tuple[float, ...] = (1 / 12, 2 / 12, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
K90, K110 = float(np.log(0.9)), float(np.log(1.1))
K_CAP = 0.5


def check_pillars_calendar(
    pillars: Sequence[float], sigma: float = 0.20, size: float = 0.01
) -> None:
    """A pillar bump of ``size`` at σ must not break the calendar condition on its ramps:
    ``(σ + size)² T_i ≤ σ² T_{i+1}`` (SPEC v2 §7.4)."""
    ps = np.asarray(pillars, dtype=np.float64)
    if ps.size < 2 or np.any(np.diff(ps) <= 0):
        raise ValueError("pillars must be strictly increasing (at least two)")
    ratio = ((sigma + size) / sigma) ** 2
    bad = np.flatnonzero(ps[1:] / ps[:-1] < ratio - 1e-12)
    if bad.size:
        raise ValueError(
            f"pillars {ps[bad].tolist()} → {ps[bad + 1].tolist()} violate the calendar condition "
            f"for a {size:.2%} bump at sigma = {sigma:.0%} (need T_{{i+1}}/T_i >= {ratio:.3f})"
        )


def _wave_maker(ps: tuple[float, ...], j: int) -> Callable[[float], SurfacePerturbation]:
    def make(s: float) -> SurfacePerturbation:
        items = [
            {"kind": "tent", "params": {"pillars": ps, "index": i, "size": s}} for i in range(j + 1)
        ]
        return SurfacePerturbation("composite", {"items": items})

    return make


def _bucket_maker(t_lo: float, t_hi: float) -> Callable[[float], SurfacePerturbation]:
    def make(eps: float) -> SurfacePerturbation:
        return SurfacePerturbation("total_variance", {"eps": eps, "t_lo": t_lo, "t_hi": t_hi})

    return make


def _tent_maker(
    kind: str, ps: tuple[float, ...], i: int, coef_name: str, k_cap: float | None
) -> Callable[[float], SurfacePerturbation]:
    def make(c: float) -> SurfacePerturbation:
        params: dict[str, object] = {"pillars": ps, "index": i, coef_name: c}
        if k_cap is not None:
            params["k_cap"] = k_cap
        return SurfacePerturbation(kind, params)

    return make


@dataclass(frozen=True)
class Ladder:
    """A per-pillar (or per-bucket) ladder with its sum and the matching global bump."""

    kind: str
    variant: str
    labels: tuple[str, ...]
    entries: tuple[Sensitivity, ...]
    total: Sensitivity | None = None
    parallel: Sensitivity | None = None
    extra: dict[str, object] = field(default_factory=dict)

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "label": self.labels,
                "value": [e.value for e in self.entries],
                "stderr": [e.stderr for e in self.entries],
                "achieved_size": [e.size for e in self.entries],
            }
        )

    def __repr__(self) -> str:
        rows = ", ".join(
            f"{lab}: {e.value:.4g}±{e.stderr:.2g}" for lab, e in zip(self.labels, self.entries)
        )
        return f"Ladder({self.kind}/{self.variant}: {rows})"


def _mode(variant: str) -> str:
    if variant == "recalibrated":
        return "recalibrate"
    if variant == "sticky_leverage":
        return "sticky_leverage"
    raise ValueError("variant must be 'recalibrated' or 'sticky_leverage'")


def _bump_sensitivity(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    make: Callable[[float], SurfacePerturbation],
    size: float,
    *,
    name: str,
    unit: str,
    mode: str,
    per_unit: float,
) -> Sensitivity:
    """``[P(bumped) − P(base)] · per_unit · (size / achieved)``: the forward difference scaled to
    one requested unit whatever halving the arbitrage checks imposed."""
    bumped, achieved = engine.perturbed_state(state, make, size, label=name)
    c = per_unit * size / achieved
    return engine.combination(
        name,
        product,
        [(bumped, mode, c), (state, "recalibrate", -c)],
        unit=unit,
        size=achieved,
        scheme="forward",
        extra={"requested": size},
    )


@dataclass(frozen=True)
class VegaTermStructure:
    """Cumulative waves, projections per pillar and the single-pillar tents (SPEC v2 §7.4)."""

    pillars: tuple[float, ...]
    waves: tuple[Sensitivity, ...]
    projections: tuple[Sensitivity, ...]
    tents: tuple[Sensitivity, ...]
    parallel: Sensitivity
    variant: str

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "pillar": self.pillars,
                "wave": [w.value for w in self.waves],
                "wave_stderr": [w.stderr for w in self.waves],
                "projection": [p.value for p in self.projections],
                "projection_stderr": [p.stderr for p in self.projections],
                "tent": [t.value for t in self.tents],
                "tent_stderr": [t.stderr for t in self.tents],
            }
        )


def vega_T(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    pillars: Sequence[float] = PILLARS,
    size: float = 0.01,
    variant: str = "recalibrated",
    *,
    with_tents: bool = True,
) -> VegaTermStructure:
    """Vega by maturity in the desk "wave" convention; units per vol point."""
    ps = tuple(float(p) for p in pillars)
    check_pillars_calendar(ps, float(surface_of(state).atm_vol(1.0)), size)
    mode = _mode(variant)
    per_vp = 0.01 / size

    waves: list[Sensitivity] = []
    wave_states: list[RiskState] = []
    for j in range(len(ps)):
        make = _wave_maker(ps, j)
        bumped, achieved = engine.perturbed_state(state, make, size, label=f"wave{j + 1}")
        c = per_vp * size / achieved
        waves.append(
            engine.combination(
                f"vega_wave[{ps[j]:g}y]",
                product,
                [(bumped, mode, c), (state, "recalibrate", -c)],
                unit="per vol point",
                size=achieved,
                scheme="forward",
            )
        )
        wave_states.append(bumped)
    projections: list[Sensitivity] = []
    for j in range(len(ps)):
        terms = [(wave_states[j], mode, per_vp)]
        if j > 0:
            terms.append((wave_states[j - 1], mode, -per_vp))
        else:
            terms.append((state, "recalibrate", -per_vp))
        projections.append(
            engine.combination(
                f"vega_projection[{ps[j]:g}y]",
                product,
                terms,
                unit="per vol point",
                size=size,
                scheme="forward",
            )
        )
    tents: list[Sensitivity] = []
    if with_tents:
        for i in range(len(ps)):
            tents.append(
                _bump_sensitivity(
                    engine,
                    product,
                    state,
                    _tent_maker("tent", ps, i, "size", None),
                    size,
                    name=f"vega_tent[{ps[i]:g}y]",
                    unit="per vol point",
                    mode=mode,
                    per_unit=per_vp,
                )
            )
    return VegaTermStructure(ps, tuple(waves), tuple(projections), tuple(tents), waves[-1], variant)


def default_buckets() -> tuple[tuple[float, float], ...]:
    """Monthly to 1y, quarterly to 3y (20 buckets)."""
    edges = [i / 12 for i in range(13)] + [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    return tuple((edges[i], edges[i + 1]) for i in range(len(edges) - 1))


def _bucket_perturbation(eps: float, t_lo: float, t_hi: float) -> SurfacePerturbation:
    return SurfacePerturbation("total_variance", {"eps": eps, "t_lo": t_lo, "t_hi": t_hi})


def bucket_forward_variance(surface: ImpliedSurface, t_lo: float, t_hi: float) -> float:
    """Forward variance-swap variance of the bucket from the log-contract strip:
    ``[K(t_hi) t_hi − K(t_lo) t_lo] / (t_hi − t_lo)``."""
    k_hi = float(varswap_strike(surface, t_hi)) * t_hi
    k_lo = float(varswap_strike(surface, t_lo)) * t_lo if t_lo > 0 else 0.0
    return (k_hi - k_lo) / (t_hi - t_lo)


def bucket_epsilon(surface: ImpliedSurface, t_lo: float, t_hi: float, size: float) -> float:
    """``ε`` of the ``"total_variance"`` bump such that the bucket's forward VS vol (log-contract
    strip) rises by ``size``.  The ``ξ₀``-scaling start value ``[(σ_b + size)² − σ_b²]/σ_b²`` is
    exact for a flat smile only: a parallel shift of total variance across strikes moves the log
    contract by a smile-dependent factor (1.17 on the reference surface at 1y, where ``K_var`` =
    6.0% against an ATM variance of 4.0%), so ε is refined by two secant steps on the measured
    strip move (the response is linear to well below the MC resolution)."""
    var0 = bucket_forward_variance(surface, t_lo, t_hi)
    target = (float(np.sqrt(var0)) + size) ** 2 - var0
    xi0 = xi0_curve(surface, min(surface.max_maturity, max(t_hi + 1.0, 5.0)))
    mean_xi = float(xi0.integral(t_lo, t_hi)) / (t_hi - t_lo)
    eps = target / mean_xi
    move = 0.0
    for _ in range(2):
        bumped = perturbed_surface(_bucket_perturbation(eps, t_lo, t_hi), surface)
        move = bucket_forward_variance(bumped, t_lo, t_hi) - var0
        eps *= target / move
    return float(eps)


def varswap_bucket_sensitivity(
    vs: VarianceSwap, surface: ImpliedSurface, t_lo: float, t_hi: float, size: float = 0.01
) -> float:
    """Model-independent ladder entry of a plain variance swap (Gyöngy: every model calibrated to
    the surface prices it at the log-contract strip): ``N · DF(T) · [K_bumped(T) − K(T)]`` for a
    spot-start swap, the strip difference over ``[start, T]`` for a forward-start one.  For a
    bucket inside ``[start, T]`` this is ``N · DF(T) · [(σ_b + size)² − σ_b²] · (t_hi − t_lo)/(T −
    start)`` by the sizing of :func:`bucket_epsilon`."""
    T = vs.maturity
    if min(T, t_hi) <= max(t_lo, vs.start):
        return 0.0
    eps = bucket_epsilon(surface, t_lo, t_hi, size)
    bumped = perturbed_surface(_bucket_perturbation(eps, t_lo, t_hi), surface)
    d_total = (float(varswap_strike(bumped, T)) - float(varswap_strike(surface, T))) * T
    if vs.start > 0:
        d_total -= (
            float(varswap_strike(bumped, vs.start)) - float(varswap_strike(surface, vs.start))
        ) * vs.start
    return float(vs.notional * float(vs.df(T)) * d_total / (T - vs.start))


def fwd_var_ladder(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    buckets: Sequence[tuple[float, float]] | None = None,
    size: float = 0.01,
    variant: str = "recalibrated",
) -> Ladder:
    """Forward-variance vega ladder; entries per vol point of the bucket's forward VS vol."""
    bk = tuple(buckets) if buckets is not None else default_buckets()
    mode = _mode(variant)
    surface = surface_of(state)
    entries: list[Sensitivity] = []
    labels: list[str] = []
    eps_list: list[float] = []
    for t_lo, t_hi in bk:
        eps = bucket_epsilon(surface, t_lo, t_hi, size)
        eps_list.append(eps)
        make = _bucket_maker(t_lo, t_hi)
        entries.append(
            _bump_sensitivity(
                engine,
                product,
                state,
                make,
                eps,
                name=f"fwd_var[{t_lo:g}-{t_hi:g}y]",
                unit="per vol point of bucket fwd VS vol",
                mode=mode,
                per_unit=1.0,
            )
        )
        labels.append(f"{t_lo:g}-{t_hi:g}y")
    total = Sensitivity(
        "fwd_var_ladder_sum",
        float(sum(e.value for e in entries)),
        float(np.sqrt(sum(e.stderr**2 for e in entries))),
        "per vol point",
        size,
        "forward",
    )
    # the parallel forward-variance bump: every bucket at once, each at its own ε
    items = [
        {"kind": "total_variance", "params": {"eps": e, "t_lo": lo, "t_hi": hi}}
        for e, (lo, hi) in zip(eps_list, bk)
    ]
    parallel = _bump_sensitivity(
        engine,
        product,
        state,
        lambda s: SurfacePerturbation("composite", {"items": items}),
        1.0,
        name="fwd_var_parallel",
        unit="per vol point",
        mode=mode,
        per_unit=1.0,
    )
    return Ladder(
        "fwd_var", variant, tuple(labels), tuple(entries), total, parallel, {"eps": eps_list}
    )


def skew_slope(size: float = 0.01, k_cap: float = K_CAP) -> float:
    """``s`` such that ``σ(ln 0.9) − σ(ln 1.1)`` rises by ``size`` under ``s·κ(k)``."""
    return size / float(saturated_k(K90, k_cap) - saturated_k(K110, k_cap))


def curvature_coefficient(size: float = 0.01, k_cap: float = K_CAP) -> float:
    """``c`` such that ``[σ(ln 0.9) + σ(ln 1.1)]/2 − σ(0)`` rises by ``size`` under ``c·κ(k)²``."""
    return size / float(0.5 * (saturated_k(K90, k_cap) ** 2 + saturated_k(K110, k_cap) ** 2))


def _tent_ladder(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    pillars: Sequence[float],
    size: float,
    variant: str,
    kind: str,
    coef_name: str,
    coef: float,
    unit: str,
    k_cap: float,
) -> Ladder:
    ps = tuple(float(p) for p in pillars)
    mode = _mode(variant)
    entries: list[Sensitivity] = []
    for i in range(len(ps)):
        make = _tent_maker(kind, ps, i, coef_name, k_cap)
        entries.append(
            _bump_sensitivity(
                engine,
                product,
                state,
                make,
                coef,
                name=f"{kind}[{ps[i]:g}y]",
                unit=unit,
                mode=mode,
                per_unit=0.01 / size,
            )
        )
    total = Sensitivity(
        f"{kind}_sum",
        float(sum(e.value for e in entries)),
        float(np.sqrt(sum(e.stderr**2 for e in entries))),
        unit,
        size,
        "forward",
    )
    items = [
        {"kind": kind, "params": {"pillars": ps, "index": i, coef_name: coef, "k_cap": k_cap}}
        for i in range(len(ps))
    ]
    global_bump = _bump_sensitivity(
        engine,
        product,
        state,
        lambda c: SurfacePerturbation(
            "composite",
            {
                "items": [
                    {"kind": kind, "params": {**it["params"], coef_name: c}} for it in items  # type: ignore[dict-item]
                ]
            },
        ),
        coef,
        name=f"{kind}_global",
        unit=unit,
        mode=mode,
        per_unit=0.01 / size,
    )
    return Ladder(kind, variant, tuple(f"{p:g}y" for p in ps), tuple(entries), total, global_bump)


def skew_T(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    pillars: Sequence[float] = PILLARS,
    size: float = 0.01,
    variant: str = "recalibrated",
    k_cap: float = K_CAP,
) -> Ladder:
    """Skew risk per pillar: per vol point of 90/110 skew at the pillar."""
    return _tent_ladder(
        engine,
        product,
        state,
        pillars,
        size,
        variant,
        "skew_tent",
        "slope",
        skew_slope(size, k_cap),
        "per vol point of 90/110 skew",
        k_cap,
    )


def curvature_T(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    pillars: Sequence[float] = PILLARS,
    size: float = 0.01,
    variant: str = "recalibrated",
    k_cap: float = K_CAP,
) -> Ladder:
    """Curvature risk per pillar: per vol point of the 90/110 butterfly at the pillar."""
    return _tent_ladder(
        engine,
        product,
        state,
        pillars,
        size,
        variant,
        "curvature_tent",
        "curv",
        curvature_coefficient(size, k_cap),
        "per vol point of 90/110 butterfly",
        k_cap,
    )


def fwd_var_convexity(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    buckets: Sequence[tuple[float, float]] | None = None,
    size: float = 0.01,
    variant: str = "recalibrated",
) -> Ladder:
    """Second-order forward-variance ladder (SPEC v2 §7.5, on request): the diagonal convexity
    per bucket from the three-point second difference on the bucket bump at the achieved sizes
    ``0, a, a₂`` (``a₂`` the double bump), per (vol point of bucket forward VS vol)²; off-diagonal
    cross terms are not computed."""
    bk = tuple(buckets) if buckets is not None else default_buckets()
    mode = _mode(variant)
    surface = surface_of(state)
    entries: list[Sensitivity] = []
    labels: list[str] = []
    for t_lo, t_hi in bk:
        eps1 = bucket_epsilon(surface, t_lo, t_hi, size)
        eps2 = bucket_epsilon(surface, t_lo, t_hi, 2.0 * size)
        make = _bucket_maker(t_lo, t_hi)
        s1, a1 = engine.perturbed_state(state, make, eps1, label=f"fwd_var[{t_lo:g}-{t_hi:g}y]")
        s2, a2 = engine.perturbed_state(state, make, eps2, label=f"fwd_var2[{t_lo:g}-{t_hi:g}y]")
        # achieved sizes in vol points of the bucket forward VS vol
        v1 = size * a1 / eps1
        v2 = 2.0 * size * a2 / eps2
        if v2 <= v1:
            raise ValueError("the double bucket bump could not exceed the single one")
        c2 = 2.0 / ((v2 - v1) * v2) * 0.01**2
        c1 = -2.0 * (1.0 / ((v2 - v1) * v2) + 1.0 / (v1 * v2)) * 0.01**2
        c0 = 2.0 / (v1 * v2) * 0.01**2
        entries.append(
            engine.combination(
                f"fwd_var_convexity[{t_lo:g}-{t_hi:g}y]",
                product,
                [(s2, mode, c2), (s1, mode, c1), (state, "recalibrate", c0)],
                unit="per (vol point of bucket fwd VS vol)^2",
                size=v1,
                scheme="three-point",
                extra={"sizes": (v1, v2)},
            )
        )
        labels.append(f"{t_lo:g}-{t_hi:g}y")
    return Ladder("fwd_var_convexity", variant, tuple(labels), tuple(entries))
