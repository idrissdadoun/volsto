"""The hedge comparison sets of the payoff study (SPEC §8 Part 3; the owner's request of
2026-09-27: "propose the best possible ways to hedge and compare different proposed hedges").

:func:`comparison_strategies` returns, per product, named :class:`GreekTargetStrategy` objects run
side by side by the study on the same worlds (daily rebalancing; the hedger reports every run with
and without transaction costs).  Every set starts from the two baselines — ``"delta"`` (the spot
on the model delta) and ``"preset"`` (:func:`~volsto.hedging.strategies.default_strategy`) — and
adds the product's structural hedges:

======================================  ================================================
product                                 strategies beyond the baselines
======================================  ================================================
knock-out / knock-in barrier option     ``"PCS static (symmetric) no delta"`` — the
(reverse barriers of the study)         put-call-symmetry replication without carry or
                                        continuity correction (Carr–Bowie), held alone;
                                        ``"PCS static (carry, BGK) no delta"`` — with the
                                        carry power and the BGK barrier; ``"PCS static
                                        (carry, BGK)"`` — + the spot on the net delta;
                                        ``"PCS static (carry, BGK) + vega"`` — + the ATM
                                        vanilla on the vega
option on realised variance             ``"var + vol swap (vega, volga)"`` (the vol-of-vol
                                        overlay: the var–vol swap spread carries the
                                        convexity — the product's main residual once the
                                        variance level is hedged)
up / down (corridor) variance swap      ``"var swap (vega)"``; ``"corridor strip"`` (Carr–
                                        Lewis ``2/K²`` options on the corridor + delta);
                                        ``"corridor strip + var swap"`` (the residual vega)
knock-out variance swap                 ``"stopped log strip"`` (the symmetric-reflection
                                        replication + delta, unwound at the knock-out);
                                        ``"stopped log strip + var swap"``
VKO put                                 ``"put static + delta"`` (short the underlying put,
                                        the vol-knock-in part on the delta only)
======================================  ================================================

``unstable=True`` adds the four-target vanna-volga sets — ``"vanna-volga"`` on the barriers
(delta, vega, vanna, volga with the 25Δ put, ATM call and 25Δ call of the expiry) and ``"var swap
+ vanna-volga"`` on the option on variance — which are off by default: with the regressed vanna
and volga the solve takes noise-driven quantities (measured under Black–Scholes, daily
rebalancing, 10⁴ paths: P&L std 13.6 and 19.3 against the unhedged 1.6 and 4.3 on the
down-and-out put and the up-and-out call; 796 bp against 26 bp on the put on variance) — a
statement about the estimator, not the hedge, which a desk runs on closed-form or PDE Greeks.

Measured under Black–Scholes (pricing = world, carry 1%, daily monitoring and rebalancing, 10⁴
paths; P&L std): the static replication alone beats every dynamic overlay on the barriers — 0.43
(carry, BGK) and 0.51 (symmetric) against 0.80 with the spot on the net delta, 1.05 with the vega
added, 0.72 for the delta alone and 1.38 for the preset on the down-and-out put (up-and-out call:
0.57, 0.81, 1.36, 1.69, 1.33, 4.82 — the preset is worse than unhedged, 4.26): the net delta of
product and replication is a regression estimate of a quantity the replication zeroes.  The
variance strips need the pricing paths: the corridor strip's P&L std fell from 43 to 24 bp between
5·10³ and 2·10⁴ pricing paths (delta 30, the variance swap on the vega 11), its net delta carrying
the strip's ``2/K²`` gamma.

The static legs follow the hedger's convention (:mod:`volsto.hedging.strategies`): the product is
held long and a leg replicating a piece of it is held short.  A knock-in's knock-out replication
is held long against the short parity vanilla and unwound at the knock-in
(``GreekTargetStrategy.unwind_on_knock``).  The delta-regime and recalibration variants are
hedger settings, not strategies: the study crosses them with these sets.

Checked by ``tests/test_replication.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from volsto.hedging.instruments import HedgeInstrument, VolSwap
from volsto.hedging.replication import (
    barrier_replication,
    corridor_strip,
    stopped_log_strip,
)
from volsto.hedging.strategies import (
    GreekTargetStrategy,
    PresetContext,
    Target,
    default_strategy,
)
from volsto.products.base import Product

#: the delta of the vanna-volga wings
WING_DELTA = 0.25


def delta_only(product: Product, ctx: PresetContext) -> GreekTargetStrategy:
    """The spot on the model delta (the baseline of every set)."""
    return GreekTargetStrategy((Target("delta"),), [ctx.spot_instrument()], name="delta")


def wing_strikes(ctx: PresetContext, T: float, delta: float = WING_DELTA) -> tuple[float, float]:
    """The ``delta``-put and ``delta``-call strikes of the expiry at the ATM reference vol
    (forward delta, Black)."""
    from scipy.stats import norm

    f = float(ctx.forward_curve.forward(T))
    sd = ctx.atm_vol(T) * np.sqrt(T)
    z = float(norm.ppf(1.0 - delta))
    return f * float(np.exp(-z * sd + 0.5 * sd * sd)), f * float(np.exp(z * sd + 0.5 * sd * sd))


def vanna_volga_strategy(
    product: Product, ctx: PresetContext, T: float | None = None
) -> GreekTargetStrategy:
    """Delta, vega, vanna and volga with the 25Δ put, the ATM call and the 25Δ call of the
    product's expiry (four targets, four instruments: an exact per-path solve)."""
    T = float(product.maturity if T is None else T)
    f = float(ctx.forward_curve.forward(T))
    kp, kc = wing_strikes(ctx, T)
    inst: list[HedgeInstrument] = [
        ctx.spot_instrument(),
        ctx.vanilla(kp, T, -1, "25d put"),
        ctx.vanilla(f, T, 1, "atm call"),
        ctx.vanilla(kc, T, 1, "25d call"),
    ]
    targets = (Target("delta"), Target("vega"), Target("vanna"), Target("volga"))
    return GreekTargetStrategy(targets, inst, name="vanna-volga")


def barrier_static_strategy(
    product: Any,
    ctx: PresetContext,
    *,
    carry: bool = True,
    continuity: bool = True,
    vega: bool = False,
    spot: bool = True,
    name: str = "",
) -> GreekTargetStrategy:
    """The put-call-symmetry semi-static hedge (module docstring): the knock-out replication held
    short (a knock-out) or long against the short parity vanilla and unwound at the knock-in (a
    knock-in), the spot on the residual delta (``spot=False``: the static legs alone, the delta
    target only reported); ``vega`` adds the ATM vanilla on the vega."""
    from volsto.products.barrier import KnockInOption

    rep = barrier_replication(product, ctx, carry=carry, continuity=continuity)
    T = float(product.T)
    n = float(product.notional)
    leg = rep.instrument(ctx.discount, name="PCS replication", cost=ctx.cost_vol_points)
    inst: list[HedgeInstrument] = [ctx.spot_instrument(), leg] if spot else [leg]
    static: dict[str, float | Callable[[float], float]] = {}
    unwind: tuple[str, ...] = ()
    if isinstance(product, KnockInOption):
        van = ctx.vanilla(float(product.strike), T, int(product.cp), "parity vanilla")
        inst.append(van)
        # knock-in = vanilla − knock-out: short the vanilla, long the knock-out's replication
        # until the knock-in, then the vanilla alone
        static["parity vanilla"] = -n
        static["PCS replication"] = n
        unwind = ("PCS replication",)
    else:
        static["PCS replication"] = -n
    targets = [Target("delta")]
    if vega:
        inst.append(ctx.vanilla(float(ctx.forward_curve.forward(T)), T, 1, "atm vanilla"))
        targets.append(Target("vega"))
    label = name or (
        "PCS static"
        + (" (carry, BGK)" if carry and continuity else " (symmetric)" if not carry else "")
        + (" + vega" if vega else "")
        + ("" if spot else " no delta")
    )
    s = GreekTargetStrategy(tuple(targets), inst, static, name=label, unwind_on_knock=unwind)
    s.notes.extend(rep.notes)
    return s


def variance_option_overlays(product: Any, ctx: PresetContext) -> dict[str, GreekTargetStrategy]:
    """The vol-of-vol overlays of an option on realised variance (module docstring)."""
    T = float(product.maturity)
    start = float(product.start)
    k_vol = ctx.atm_vol(T)
    var_swap = ctx.varswap(start, T, k_vol, name="var swap")
    vol_swap = VolSwap(
        start_time=start,
        end=T,
        strike_vol=k_vol,
        discount=ctx.discount,
        cost=ctx.cost_vol_points,
        name="vol swap",
    )
    kp, kc = wing_strikes(ctx, T)
    out = {
        "var + vol swap (vega, volga)": GreekTargetStrategy(
            (Target("delta"), Target("vega"), Target("volga")),
            [ctx.spot_instrument(), var_swap, vol_swap],
            name="var + vol swap (vega, volga)",
        ),
        "var swap + vanna-volga": GreekTargetStrategy(
            (Target("delta"), Target("vega"), Target("vanna"), Target("volga")),
            [
                ctx.spot_instrument(),
                ctx.varswap(start, T, k_vol, name="var swap"),
                ctx.vanilla(kp, T, -1, "25d put"),
                ctx.vanilla(kc, T, 1, "25d call"),
            ],
            name="var swap + vanna-volga",
        ),
    }
    return out


def _strip_strategy(
    rep_leg: HedgeInstrument, product: Any, ctx: PresetContext, *, vega: bool, name: str
) -> GreekTargetStrategy:
    T = float(product.maturity)
    inst: list[HedgeInstrument] = [ctx.spot_instrument(), rep_leg]
    targets = [Target("delta")]
    if vega:
        inst.append(ctx.varswap(0.0, T, ctx.atm_vol(T), name="var swap"))
        targets.append(Target("vega"))
    # the product is long its float leg: the hedge is short the replicating options
    static: dict[str, float | Callable[[float], float]] = {rep_leg.name: -float(product.notional)}
    s = GreekTargetStrategy(tuple(targets), inst, static, name=name)
    s.notes.extend(rep_leg.notes)
    return s


def comparison_strategies(
    product: Product, ctx: PresetContext, *, unstable: bool = False
) -> dict[str, GreekTargetStrategy]:
    """The named strategies of ``product``'s comparison set (module docstring); ``unstable``
    adds the four-target vanna-volga sets."""
    from volsto.products.barrier import _BarrierOption
    from volsto.products.conditional_variance import (
        ConditionalVarianceSwap,
        KnockOutVarianceSwap,
    )
    from volsto.products.variance import VarianceOption
    from volsto.products.vko import VolKnockOutPut

    out: dict[str, GreekTargetStrategy] = {
        "delta": delta_only(product, ctx),
        "preset": default_strategy(product, ctx),
    }
    if isinstance(product, _BarrierOption):
        for carry, spot, vega in (
            (False, False, False),
            (True, False, False),
            (True, True, False),
            (True, True, True),
        ):
            s = barrier_static_strategy(
                product, ctx, carry=carry, continuity=carry, spot=spot, vega=vega
            )
            out[s.name] = s
        if unstable:
            out["vanna-volga"] = vanna_volga_strategy(product, ctx, float(product.T))
    elif isinstance(product, VarianceOption):
        overlays = variance_option_overlays(product, ctx)
        out["var + vol swap (vega, volga)"] = overlays["var + vol swap (vega, volga)"]
        if unstable:
            out["var swap + vanna-volga"] = overlays["var swap + vanna-volga"]
    elif isinstance(product, ConditionalVarianceSwap):
        T = float(product.maturity)
        out["var swap (vega)"] = GreekTargetStrategy(
            (Target("delta"), Target("vega")),
            [ctx.spot_instrument(), ctx.varswap(0.0, T, ctx.atm_vol(T), name="var swap")],
            name="var swap (vega)",
        )
        if product.convention == "corridor":
            rep = corridor_strip(product, ctx)
            for vega in (False, True):
                leg = rep.instrument(ctx.discount, name="corridor strip", cost=ctx.cost_vol_points)
                nm = "corridor strip" + (" + var swap" if vega else "")
                out[nm] = _strip_strategy(leg, product, ctx, vega=vega, name=nm)
    elif isinstance(product, KnockOutVarianceSwap):
        rep = stopped_log_strip(product, ctx)
        for vega in (False, True):
            leg = rep.instrument(ctx.discount, name="stopped log strip", cost=ctx.cost_vol_points)
            nm = "stopped log strip" + (" + var swap" if vega else "")
            out[nm] = _strip_strategy(leg, product, ctx, vega=vega, name=nm)
    elif isinstance(product, VolKnockOutPut):
        T = float(product.T)
        put = ctx.vanilla(float(product.strike), T, -1, "underlying put")
        out["put static + delta"] = GreekTargetStrategy(
            (Target("delta"),),
            [ctx.spot_instrument(), put],
            {"underlying put": -float(product.notional)},
            name="put static + delta",
        )
    return out


__all__ = [
    "WING_DELTA",
    "barrier_static_strategy",
    "comparison_strategies",
    "delta_only",
    "vanna_volga_strategy",
    "variance_option_overlays",
    "wing_strikes",
]
