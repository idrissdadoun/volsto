"""Black–Scholes proxies for the §7.11 control variate on the difference in the hedging engine
(SPEC §8.1, owner decision (b) at the M8 acceptance: "use the control variate on the difference
for the tent-bump target").

A hedge object with a proxy is a portfolio of legs whose conditional value under a flat-vol
Black–Scholes model is known in closed form given the object's hedge state at ``t``.  For each
such object and each surface-driven model bump (vega / volga, forward-variance buckets, the
``skew_T`` / ``curvature_T`` tents) :class:`~volsto.hedging.pricing.ConditionalPricer` prices a
Black–Scholes **shadow** on the pricing draws' spot normals — one per proxy vol, with the pricing
model's forward curve — and forms the per-path control ``c_i`` as the leg payoffs on the shadow
futures spliced onto the pricing path's history at ``t`` (the hybrid construction of the
targets), scaled exactly like the target.  Its conditional expectation given the state is the
analytic value difference of :meth:`ProxyLeg.conditional_value`; the controlled target ``y_i − β
(c_i − E[c_i | state_i])`` keeps the regression unbiased whatever ``β`` (``β = Cov(y, c)/Var(c)``
on the alive paths at the date).

**The shadow is exact by construction.**  :class:`ShadowBrownian` accumulates the spot Brownian
motion ``W`` of the pricing draws (``GaussianDraws`` with the pricing seed, Brownian column 0 —
the spot's in every model of the library) at the record columns, and the shadow at vol ``σ``
spliced at ``t`` is the Black–Scholes law on those increments, ``ln S_T = ln S_t^base +
ln F(T)/F(t) − ½σ²(T − t) + σ (W_T − W_t)`` — the same numbers
:class:`~volsto.models.bs.BlackScholes` produces step by step (its kernel is exact for a flat
vol; equality to round-off is checked by ``test_shadow_brownian_matches_black_scholes_kernel``),
without one full path set per proxy vol: one ``(n_paths, n_cols)`` array serves every leg, vol
and date, and the conditional law given ``S_t`` is exactly lognormal whatever the step sizes, so
the analytic expectation is exact.

Proxies in this version (state-free objects only):

==================================  ========================================================
object                              proxy
==================================  ========================================================
``EuropeanOption``                  Black at the surface's implied vol of ``(K, T)``
``DigitalOption``                   cash-or-nothing Black, same vol
``ForwardStartOption``,             **before ``T1`` only**: the unconditional forward-start
``ForwardStartStraddle``            Black price on the ratio at the surface vol of ``(ln m,
                                    T2)`` — a proxy vol; the expectation is exact by
                                    construction and constant across paths; after ``T1`` no
                                    control (the proxy would need the history held, ``S_T1``)
``Portfolio``                       every leg with a proxy (the forward risk reversal, the
                                    forward-start straddle, the ``T1``-struck vanilla family,
                                    call spreads and risk reversals of vanillas)
anything else                       no control (noted): cliquets, autocalls, barriers,
                                    variance products, accumulated-sum options, the FVA
==================================  ========================================================

The proxy vol of a bumped shadow is ``base vol + [bumped surface vol − base surface vol]`` at the
leg's ``(k, T)`` (or ``+`` the flat shift of a bare Black–Scholes context); a Black–Scholes
pricing model's own ``vol`` is the base vol (the shadow is then the pricing model itself and the
controlled vega target is the analytic one to round-off).  A zero-strike forward-start call
(``k = 0``, the cliquet decomposition's forward-return leg) pays ``R`` itself, whose value
``F(T2)/F(T1)`` is vol-independent: its proxy vol is read at the money.

Checked by ``tests/test_hedging.py`` (``test_control_variate_vega_vanilla_lv``,
``test_control_variate_forward_risk_reversal_skew_tent``, ``test_no_control_for_autocall``,
``test_shadow_brownian_matches_black_scholes_kernel``).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volsto.engine.grid import TimeGrid
from volsto.engine.paths import PathSet
from volsto.engine.rng import GaussianDraws
from volsto.market.bs import black_price, norm_cdf
from volsto.products.base import Portfolio, Product
from volsto.products.forward_start import ForwardStartOption, ForwardStartStraddle
from volsto.products.vanilla import DigitalOption, EuropeanOption

FloatArray = NDArray[np.float64]
_TOL = 1e-9
#: shortest maturity at which a proxy vol is read from the surface (the surface's own floor in
#: :meth:`~volsto.hedging.hedger.PricingContext.reference_vol`)
MIN_VOL_MATURITY = 1.0 / 365.0
#: floor of a shadow vol (a bumped-down proxy vol cannot cross zero; the bare Black–Scholes
#: context floors its own bumped model at the same level in ``PricingContext.bump``)
MIN_SHADOW_VOL = 1e-4
PROXY_KINDS = ("vanilla", "digital", "forward_start")
#: steps per block when accumulating the Brownian (the models' own ``step_block``)
_STEP_BLOCK = 64


@dataclass(frozen=True)
class ProxyLeg:
    """One Black–Scholes shadow leg of a hedge object (module docstring): the leg ``product``
    evaluated on the shadow paths, its ``weight`` in the object, where its proxy vol is read
    (``strike`` — the strike of a vanilla, the moneyness of a forward start — and ``T``), the
    date ``start`` before which the control is available (``None``: always) and its ``kind``."""

    product: Product
    weight: float
    strike: float
    T: float
    kind: str
    start: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in PROXY_KINDS:
            raise ValueError(f"kind must be one of {PROXY_KINDS}")

    def active(self, t: float) -> bool:
        """Whether the proxy holds at ``t`` (a forward start: before its ``T1`` fixing)."""
        return self.start is None or t < self.start - _TOL

    def vol(self, surface: Any) -> float:
        """The proxy vol read from ``surface`` at the leg's ``(k, T)`` (``T`` clipped to the
        surface's range like the hedger's reference vols)."""
        T = min(max(self.T, MIN_VOL_MATURITY), float(surface.max_maturity))
        if self.kind == "forward_start":
            k = float(np.log(self.strike)) if self.strike > 0 else 0.0
        else:
            k = float(np.asarray(surface.log_moneyness(self.strike, T)).ravel()[0])
        return float(np.asarray(surface.implied_vol_k(k, T)).ravel()[0])

    def conditional_value(self, ln_s_t: FloatArray, t: float, vol: float, fc: Any) -> FloatArray:
        """``E[discounted leg payoff | S_t]`` under Black–Scholes at ``vol`` started from ``S_t``
        with the forward curve ``fc`` (time-0 money: the leg's own discount curve), exact for
        the shadow of :class:`ShadowBrownian`; a leg whose fixings all lie at or before ``t``
        is settled and returns its realised payoff (the shadow reproduces it)."""
        p = self.product
        n = ln_s_t.size
        if self.kind == "forward_start":
            assert isinstance(p, ForwardStartOption)
            f_ratio = float(fc.forward(p.T2) / fc.forward(p.T1))
            value = float(p.df(p.pay_time)) * float(
                black_price(f_ratio, p.strike, p.tau, vol, p.cp)
            )
            return np.full(n, p.notional * value)
        assert isinstance(p, EuropeanOption | DigitalOption)
        tau = max(p.T - t, 0.0)
        f_t = np.exp(ln_s_t) * float(fc.forward(p.T) / fc.forward(t))
        df = float(p.df(p.T))
        if self.kind == "vanilla":
            return np.asarray(
                p.notional * df * black_price(f_t, p.strike, tau, vol, p.cp), dtype=np.float64
            )
        assert isinstance(p, DigitalOption)
        s = vol * np.sqrt(tau)
        if s <= 0.0:
            hit = (p.cp * (f_t - p.strike) > 0.0).astype(np.float64)
        else:
            d2 = (np.log(f_t / p.strike) - 0.5 * s * s) / s
            hit = norm_cdf(p.cp * d2)
        return np.asarray(p.notional * p.payout * df * hit, dtype=np.float64)

    @property
    def columns(self) -> FloatArray:
        """The leg's fixing times (the columns the shadow has to fill)."""
        return np.asarray(self.product.fixing_times, dtype=np.float64)


def proxy_legs(product: Product, weight: float = 1.0) -> tuple[ProxyLeg, ...] | None:
    """The Black–Scholes proxy legs of ``product`` (module docstring), ``None`` when it has no
    proxy (a portfolio needs one for every leg)."""
    if isinstance(product, EuropeanOption):
        return (ProxyLeg(product, weight, product.strike, product.T, "vanilla"),)
    if isinstance(product, DigitalOption):
        return (ProxyLeg(product, weight, product.strike, product.T, "digital"),)
    if isinstance(product, ForwardStartOption):
        return (ProxyLeg(product, weight, product.strike, product.T2, "forward_start", product.T1),)
    if isinstance(product, ForwardStartStraddle):
        legs = product.decompose()
        return tuple(
            ProxyLeg(leg, weight, product.strike, product.T2, "forward_start", product.T1)
            for leg in legs
        )
    if isinstance(product, Portfolio):
        out: list[ProxyLeg] = []
        for w, leg in zip(product.weights, product.legs, strict=True):
            sub = proxy_legs(leg, weight * float(w))
            if sub is None:
                return None
            out.extend(sub)
        return tuple(out)
    return None


def proxy_note(product: Product, t: float | None = None) -> str:
    """The note for an object without a control at ``t``: the class (no proxy at all) or the
    forward start past its ``T1`` (proxy needs the history held)."""
    legs = proxy_legs(product)
    name = type(product).__name__
    if legs is None:
        return f"no Black-Scholes control for {name}: raw bump targets"
    started = sorted({leg.start for leg in legs if leg.start is not None and t is not None})
    return (
        f"no Black-Scholes control for {name} after T1 = {started[0]:g} (the proxy would need "
        f"the history held, S_T1): raw bump targets from T1 on"
        if started
        else f"no Black-Scholes control for {name}: raw bump targets"
    )


@dataclass
class ShadowBrownian:
    """The spot Brownian motion of the pricing draws at the record columns, ``w`` of shape
    ``(n_paths, n_cols)`` (``w[:, 0] = 0``), and the shadow Black–Scholes payoffs built on it
    (module docstring)."""

    w: FloatArray
    times: FloatArray

    @staticmethod
    def from_draws(
        seed: int, n_paths: int, grid: TimeGrid, antithetic: bool, chunks: Sequence[tuple[int, int]]
    ) -> ShadowBrownian:
        """``W_t = Σ √Δt_s z_s`` over the steps up to each record column, ``z`` the first
        Brownian column of ``GaussianDraws(seed, n_paths, n_steps, 1, antithetic)`` — the same
        normals every pricing model reads for its spot."""
        draws = GaussianDraws(seed, n_paths, grid.n_steps, 1, antithetic)
        w = np.zeros((n_paths, grid.n_records))
        sq_dt = np.sqrt(grid.dts)
        for p0, p1 in chunks:
            running = np.zeros(p1 - p0)
            for s0 in range(0, grid.n_steps, _STEP_BLOCK):
                s1 = min(s0 + _STEP_BLOCK, grid.n_steps)
                z = draws.block(s0, s1, p0, p1)[:, :, 0]
                cum = running[:, None] + np.cumsum(sq_dt[s0:s1][None, :] * z, axis=1)
                rec = grid.step_record[s0:s1]
                for j in np.nonzero(rec >= 0)[0]:
                    w[p0:p1, rec[j]] = cum[:, j]
                running = cum[:, -1]
        return ShadowBrownian(w, np.asarray(grid.record_times, dtype=np.float64))

    def shadow_paths(
        self, base: PathSet, idx: Any, col: int, vol: float, fc: Any, fixings: FloatArray
    ) -> PathSet:
        """The base paths with the columns of ``fixings`` after ``col`` (the splice at ``t``)
        replaced by the Black–Scholes shadow at ``vol`` from ``S_t^base``: ``ln S_T = ln S_t +
        ln F(T)/F(t) − ½ vol² (T − t) + vol (W_T − W_t)``; a fixing at or before ``t`` keeps the
        base value (the history held).  Only ``log_spot`` is rewritten — the proxies read
        nothing else."""
        ls = base.log_spot.copy()
        t = float(self.times[col])
        ln_f_t = float(np.log(fc.forward(t)))
        for T in np.unique(fixings):
            c = idx[float(T)]
            if c <= col:
                continue
            tau = float(T) - t
            drift = float(np.log(fc.forward(float(T)))) - ln_f_t - 0.5 * vol * vol * tau
            ls[:, c] = base.log_spot[:, col] + drift + vol * (self.w[:, c] - self.w[:, col])
        return PathSet(base.times, ls, base.variance, base.factors, base.int_var, base.sum_sq)

    def leg_payoff(
        self, leg: ProxyLeg, base: PathSet, idx: Any, col: int, vol: float, fc: Any
    ) -> FloatArray:
        """``weight × leg payoff`` on the shadow paths of ``vol`` spliced at ``col``."""
        paths = self.shadow_paths(base, idx, col, vol, fc, leg.columns)
        return np.asarray(leg.weight * leg.product.payoff(paths, idx), dtype=np.float64)


__all__ = [
    "MIN_SHADOW_VOL",
    "MIN_VOL_MATURITY",
    "PROXY_KINDS",
    "ProxyLeg",
    "ShadowBrownian",
    "proxy_legs",
    "proxy_note",
]
