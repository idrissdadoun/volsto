"""Static replications by same-expiry vanillas (SPEC §8 Part 3; the payoff study of 2026-09-27):
the legs a semi-static hedge holds from inception and unwinds when the product terminates.

**Compact payoffs as option strips** (:func:`strip_for_payoff`).  A payoff ``F`` that vanishes
outside ``[x_0, x_m]`` and is linear between the knots ``x_0 < … < x_m`` (jumps allowed at the
knots) is exactly ``Σ_i Δs_i (S − x_i)⁺ + Σ_i J_i 1{S > x_i}`` with ``Δs_i`` the slope change
and ``J_i`` the jump at ``x_i`` — or, equivalently, ``Σ_i Δs_i (x_i − S)⁺ − Σ_i J_i 1{S < x_i}``.
Calls are used when the support lies above the forward, puts when below (the options a desk
would trade; the payoff is identical either way).  A smooth segment is sampled on ``n`` interior
knots (chord interpolation); a digital is a centred call / put spread of width ``digital_width``
times its level.

**Barrier options: put-call symmetry** (:func:`barrier_replication`; Carr, Ellis & Gupta 1998,
Carr & Lee 2009).  A knock-out option pays ``g(S_T)`` — its vanilla restricted to the alive side
of the barrier ``H`` — when the barrier was never touched.  Under a driftless, smile-symmetric
diffusion the value of any ``ψ(S_T)`` at the hitting time equals that of ``(S_T/H) ψ(H²/S_T)``;
with carry ``μ`` and a flat vol ``σ`` (Black–Scholes) the reflection carries the power
``p = 1 − 2μ/σ²``: ``E_τ[ψ(S_T)] = E_τ[(S_T/H)^p ψ(H²/S_T)]``.  Hence
``KO = E[g(S_T)] − E[h(S_T)]`` with the reflected payoff ``h(S) = (S/H)^p g(H²/S)`` on the
knocked side: the static portfolio ``g − h`` is worth zero on the barrier and is unwound there.
For the reverse barriers of the study (``K`` on the alive side):

* down-and-out put, ``H < K``: ``g = P(K) − P(H) − (K − H) 1{S<H}``, ``h(S) = (S/H)^p (K −
  H²/S)`` on ``[H²/K, H)`` — with ``p = 1`` the ``K/H`` calls struck at ``H²/K``;
* up-and-out call, ``B > K``: ``g = C(K) − C(B) − (B − K) 1{S>B}``, ``h(S) = (S/B)^p (B²/S −
  K)`` on ``(B, B²/K]``;
* a knock-in is ``vanilla − knock-out``: the parity vanilla plus the knock-out's replication with
  the opposite sign, the latter unwound at the knock-in (the hedge then holds the vanilla).

Model risk of the replication (the study measures it, the builder only states it): skew breaks
the symmetry (the residual at the hit is the smile's asymmetry at the barrier, first order in
the risk reversal), the carry correction is exact only under Black–Scholes (``σ`` is the implied
vol at the barrier), and discrete monitoring overshoots the barrier — ``continuity=True`` moves
the replication's barrier to the Broadie–Glasserman–Kou level ``H e^{∓0.5826 σ √Δt}`` of the
monitoring step for the reflection ``h`` (the terminal part ``g`` keeps the monitored level, the
last observation being exact), so the portfolio is worth ``−E[g_H − g_{H'}]``, a small amount,
at the corrected level.

**Corridor variance** (:func:`corridor_strip`; Carr & Lewis 2004).  The float leg of a corridor
variance swap ``(1/T) ∫ 1{S_t ∈ C} σ_t² dt`` equals ``(1/T)[f(S_T) − f(S_0) − ∫ f'(S_t) dS_t]``
for ``f'' = (2/S²) 1{S ∈ C}``: the options ``2 dK/K²`` on the corridor's strikes (calls above the
level for an up variance, puts below it for a down variance) plus a delta that trades only
inside the corridor — which the hedger's delta target supplies.

**Knock-out variance swap: the stopped log contract** (:func:`stopped_log_strip`).  Before the
knock-out the swap's float leg is the variance swap's, ``(1/T)[L(S_{τ∧T}) − ∫_0^{τ∧T} L'(S_t)
dS_t]`` with ``L(S) = −2 ln(S/S_0) + 2(S − S_0)/S_0`` (the log contract minus its forward, the
``2/K²`` strip); at the knock-out the static part is the known ``L(B)``.  With put-call symmetry
(``p = 1``) the terminal payoff ``Φ = L`` on the alive side and ``Φ(S) = L(B)(1 + S/B) − (S/B)
L(B²/S)`` beyond ``B`` is worth ``L(B)`` on the barrier; ``Φ`` is ``C¹`` at ``B`` and
``Φ''(S) = −2/(B S)`` beyond it: the ``2 dK/K²`` strip on the alive side and the options
``−2 dK/(K B)`` beyond the barrier (short).  The fixed leg ``−K² τ/N`` (a rebate linear in the
knock-out time) has no static replication: its delta is the hedger's, its vega is left to a
variance swap on the ``vega`` target.

Checked by ``tests/test_replication.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volsto.hedging.instruments import StaticPortfolio

FloatArray = NDArray[np.float64]

#: the Broadie–Glasserman–Kou continuity constant ``−ζ(1/2)/√(2π)``
BGK_BETA = 0.5826
#: default width of the replication's digitals (fraction of the level)
REPLICATION_DIGITAL_WIDTH = 0.005
#: default number of interior knots per smooth segment of a reflected payoff
REPLICATION_KNOTS = 8
#: default truncation of the variance strips (standard deviations of ``ln S_T``)
STRIP_WIDTH_SIGMAS = 4.0
#: default strike spacing of the variance strips (fraction of the forward)
STRIP_SPACING = 0.025


@dataclass
class Replication:
    """A static replication: the options (``StaticPortfolio`` legs, per unit notional of the
    product) and what it assumed."""

    strikes: tuple[float, ...]
    weights: tuple[float, ...]
    cps: tuple[int, ...]
    maturity: float
    notes: list[str] = field(default_factory=list)
    params: dict[str, float] = field(default_factory=dict)

    def instrument(self, discount: Any, *, name: str, cost: float = 0.0) -> StaticPortfolio:
        leg = StaticPortfolio(
            name=name,
            cost=cost,
            strikes=self.strikes,
            weights=self.weights,
            cps=self.cps,
            maturity=self.maturity,
            discount=discount,
        )
        leg.notes.extend(self.notes)
        return leg

    def payoff(self, spot: FloatArray) -> FloatArray:
        s = np.asarray(spot, dtype=np.float64)[..., None]
        k = np.asarray(self.strikes)
        c = np.asarray(self.cps, dtype=np.float64)
        return np.asarray(
            np.sum(np.asarray(self.weights) * np.maximum(c * (s - k), 0.0), axis=-1),
            dtype=np.float64,
        )

    def scaled(self, factor: float) -> Replication:
        return Replication(
            self.strikes,
            tuple(factor * w for w in self.weights),
            self.cps,
            self.maturity,
            list(self.notes),
            dict(self.params),
        )

    def __add__(self, other: Replication) -> Replication:
        if abs(self.maturity - other.maturity) > 1e-12:
            raise ValueError("replications of different expiries cannot be added")
        book: dict[tuple[float, int], float] = {}
        for src in (self, other):
            for k, w, c in zip(src.strikes, src.weights, src.cps, strict=True):
                book[(k, c)] = book.get((k, c), 0.0) + w
        return _from_book(book, self.maturity, self.notes + other.notes, {**self.params})


def _from_book(
    book: dict[tuple[float, int], float],
    maturity: float,
    notes: list[str],
    params: dict[str, float],
) -> Replication:
    items = sorted((kc, w) for kc, w in book.items() if abs(w) > 1e-14)
    return Replication(
        tuple(kc[0] for kc, _ in items),
        tuple(w for _, w in items),
        tuple(kc[1] for kc, _ in items),
        maturity,
        notes,
        params,
    )


def strip_for_payoff(
    f: Callable[[FloatArray], FloatArray],
    knots: Sequence[float] | FloatArray,
    maturity: float,
    *,
    forward: float,
    digital_width: float = REPLICATION_DIGITAL_WIDTH,
) -> Replication:
    """The options of a compact piecewise-linear payoff (module docstring): ``f`` vectorised,
    zero outside ``[min(knots), max(knots)]``, linear between consecutive knots, possibly
    discontinuous at a knot (its one-sided limits are extrapolated from ``1e-7`` and ``2e-7``
    relative away)."""
    x = np.unique(np.asarray(knots, dtype=np.float64))
    if x.size < 2 or np.any(x <= 0):
        raise ValueError("need at least two positive knots")
    eps = 1e-7 * x
    # one-sided limits, extrapolated linearly to the knot (exact on a linear segment)
    left = 2.0 * np.asarray(f(x - eps), dtype=np.float64) - np.asarray(f(x - 2 * eps))
    right = 2.0 * np.asarray(f(x + eps), dtype=np.float64) - np.asarray(f(x + 2 * eps))
    left[0] = 0.0  # outside the support
    right[-1] = 0.0
    slope = np.zeros(x.size + 1)  # slope[i] on (x_{i-1}, x_i); 0 outside
    slope[1:-1] = (left[1:] - right[:-1]) / np.diff(x)
    ds = np.diff(slope)  # slope change at each knot
    jump = right - left
    above = float(x[0]) >= forward
    book: dict[tuple[float, int], float] = {}

    def add(k: float, c: int, w: float) -> None:
        if abs(w) > 1e-14:
            book[(float(k), c)] = book.get((float(k), c), 0.0) + float(w)

    cp = 1 if above else -1
    for xi, dsi, ji in zip(x, ds, jump, strict=True):
        add(xi, cp, dsi)
        if abs(ji) > 1e-14:
            # calls: + J 1{S > x};  puts: − J 1{S < x}, a centred spread of width w
            w = digital_width * xi
            lo, hi = xi - 0.5 * w, xi + 0.5 * w
            if above:
                add(lo, 1, ji / w)
                add(hi, 1, -ji / w)
            else:
                add(hi, -1, -ji / w)
                add(lo, -1, ji / w)
    return _from_book(book, maturity, [], {})


def reflection_power(forward_T: float, spot: float, maturity: float, vol: float) -> float:
    """``p = 1 − 2μ/σ²`` with the average carry ``μ = ln(F(T)/S_0)/T`` (module docstring)."""
    mu = float(np.log(forward_T / spot) / maturity)
    return 1.0 - 2.0 * mu / (vol * vol)


def barrier_replication(
    product: Any,
    ctx: Any,
    *,
    carry: bool = True,
    continuity: bool = True,
    knots: int = REPLICATION_KNOTS,
    digital_width: float = REPLICATION_DIGITAL_WIDTH,
) -> Replication:
    """The put-call-symmetry replication ``g − h`` of the knock-out option with ``product``'s
    strike, barrier, side and payoff (a knock-in's knock-out twin), per unit notional (module
    docstring).  ``ctx`` is a :class:`~volsto.hedging.strategies.PresetContext` (forward, spot,
    the reference vol at the barrier).  Implemented for the reverse barriers, strike on the
    alive side (down-and-out put ``H < K``, up-and-out call ``B > K`` and their knock-ins), whose
    ``g`` and ``h`` have compact supports; other combinations raise."""
    T = float(product.T)
    b0 = float(product.barrier_eff)
    up = product.direction == "up"
    k = float(product.strike)
    cp = int(product.cp)
    if not ((up and cp > 0 and b0 > k) or (not up and cp < 0 and b0 < k)):
        raise NotImplementedError(
            "the put-call-symmetry replication is implemented for the reverse barriers "
            "(up-and-out call B > K, down-and-out put H < K, and their knock-ins)"
        )
    fwd = float(ctx.forward_curve.forward(T))
    vol = ctx.ref(b0, T)
    notes: list[str] = []
    if vol is None:
        vol = 0.2
        notes.append("reflection vol: no reference surface, 20% assumed")
    b = b0
    if continuity and getattr(product, "monitoring", "discrete") == "discrete":
        sched = np.asarray(product.schedule, dtype=np.float64)
        dt = float(np.mean(np.diff(sched))) if sched.size > 1 else T
        b = b0 * float(np.exp((1.0 if up else -1.0) * BGK_BETA * vol * np.sqrt(dt)))
        notes.append(
            f"barrier moved to the continuity-corrected {b:.6g} (BGK, step {dt:.5g}y, "
            f"vol {vol:.4g})"
        )
    p = reflection_power(fwd, float(ctx.spot), T, vol) if carry else 1.0
    if carry:
        notes.append(
            f"reflection power p = {p:.4g} (carry {np.log(fwd / ctx.spot) / T:.4g}, vol {vol:.4g})"
        )

    def alive_part(level: float) -> Callable[[FloatArray], FloatArray]:
        def g(s: FloatArray) -> FloatArray:
            s = np.asarray(s, dtype=np.float64)
            alive = s < level if up else s > level
            return np.asarray(np.maximum(cp * (s - k), 0.0) * alive, dtype=np.float64)

        return g

    g_b = alive_part(b)

    def h(s: FloatArray) -> FloatArray:
        s = np.asarray(s, dtype=np.float64)
        knocked = s > b if up else s < b
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            val = (s / b) ** p * g_b(b * b / s)
        return np.asarray(np.where(knocked, val, 0.0), dtype=np.float64)

    # the terminal payoff keeps the monitored level (the last observation is exact: measured, a
    # g truncated at the corrected level paid K − S_T on the paths ending between the two levels,
    # which the product had knocked out); only the reflection uses the corrected level
    rep_g = strip_for_payoff(alive_part(b0), [k, b0], T, forward=fwd, digital_width=digital_width)
    lo, hi = (b, b * b / k) if up else (b * b / k, b)
    rep_h = strip_for_payoff(
        h, np.linspace(lo, hi, knots + 2), T, forward=fwd, digital_width=digital_width
    )
    rep = rep_g + rep_h.scaled(-1.0)
    rep.notes = notes
    rep.params = {"barrier_replicated": b, "reflection_power": p, "reflection_vol": vol}
    return rep


def _strip_grid(lo: float, hi: float, spacing: float) -> FloatArray:
    n = max(int(np.ceil((hi - lo) / spacing)), 1)
    return np.linspace(lo, hi, n + 1)


def _weights_midpoint(
    edges: FloatArray, density: Callable[[FloatArray], FloatArray]
) -> tuple[FloatArray, FloatArray]:
    mid = 0.5 * (edges[1:] + edges[:-1])
    return mid, np.asarray(density(mid) * np.diff(edges), dtype=np.float64)


def corridor_strip(
    product: Any,
    ctx: Any,
    *,
    width_sigmas: float = STRIP_WIDTH_SIGMAS,
    spacing: float = STRIP_SPACING,
) -> Replication:
    """The Carr–Lewis options of an up / down (corridor) variance swap per unit variance notional
    (module docstring): ``(2/K²) dK / T`` on the corridor's strikes, truncated at
    ``width_sigmas`` standard deviations of ``ln S_T`` (midpoint rule, spacing ``spacing`` of the
    forward)."""
    T = float(product.maturity)
    fwd = float(ctx.forward_curve.forward(T))
    vol = ctx.atm_vol(T)
    level = float(product.barrier)
    sd = vol * np.sqrt(T)
    if product.side == "up":
        edges = _strip_grid(level, fwd * float(np.exp(width_sigmas * sd)), spacing * fwd)
        cp = 1
    else:
        edges = _strip_grid(fwd * float(np.exp(-width_sigmas * sd)), level, spacing * fwd)
        cp = -1
    mid, w = _weights_midpoint(edges, lambda k: 2.0 / (k * k))
    notes = [
        f"corridor strip: {mid.size} {'calls' if cp > 0 else 'puts'} from {edges[0]:.4g} to "
        f"{edges[-1]:.4g} (2/K^2 dK / T, truncated at {width_sigmas:g} sd)"
    ]
    return Replication(
        tuple(float(k) for k in mid),
        tuple(float(x) / T for x in w),
        tuple(cp for _ in mid),
        T,
        notes,
        {"level": level},
    )


def stopped_log_strip(
    product: Any,
    ctx: Any,
    *,
    width_sigmas: float = STRIP_WIDTH_SIGMAS,
    spacing: float = STRIP_SPACING,
) -> Replication:
    """The stopped log contract of a knock-out variance swap per unit variance notional (module
    docstring): ``2 dK/K²`` (puts below the forward, calls above) on the alive side of the
    barrier and ``−2 dK/(K B)`` beyond it, ``/ T``; symmetric reflection ``p = 1`` (noted)."""
    T = float(product.maturity)
    fwd = float(ctx.forward_curve.forward(T))
    vol = ctx.atm_vol(T)
    b = float(product.barrier)
    sd = vol * np.sqrt(T)
    lo = fwd * float(np.exp(-width_sigmas * sd))
    hi = fwd * float(np.exp(width_sigmas * sd))
    book: dict[tuple[float, int], float] = {}
    if product.direction == "up":
        alive_edges = _strip_grid(lo, b, spacing * fwd)
        beyond_edges = _strip_grid(b, max(hi, b * (1.0 + 4 * spacing)), spacing * fwd)
    else:
        alive_edges = _strip_grid(b, hi, spacing * fwd)
        beyond_edges = _strip_grid(min(lo, b * (1.0 - 4 * spacing)), b, spacing * fwd)
    mid_a, w_a = _weights_midpoint(alive_edges, lambda k: 2.0 / (k * k))
    mid_b, w_b = _weights_midpoint(beyond_edges, lambda k: -2.0 / (k * b))
    for k, w in zip(mid_a, w_a, strict=True):
        c = -1 if k < fwd else 1
        book[(float(k), c)] = book.get((float(k), c), 0.0) + float(w) / T
    cb = 1 if product.direction == "up" else -1
    for k, w in zip(mid_b, w_b, strict=True):
        book[(float(k), cb)] = book.get((float(k), cb), 0.0) + float(w) / T
    notes = [
        f"stopped log contract: {mid_a.size} options on the alive side, {mid_b.size} beyond the "
        f"barrier {b:.4g} (symmetric reflection p = 1; the fixed leg's rebate is not replicated)"
    ]
    return _from_book(book, T, notes, {"barrier": b})


__all__ = [
    "BGK_BETA",
    "REPLICATION_DIGITAL_WIDTH",
    "REPLICATION_KNOTS",
    "STRIP_SPACING",
    "STRIP_WIDTH_SIGMAS",
    "Replication",
    "barrier_replication",
    "corridor_strip",
    "reflection_power",
    "stopped_log_strip",
    "strip_for_payoff",
]
