"""Fair strikes and reports for the conditional-variance products and the VKO put (v2 §6.1–6.2).

Fair strikes are ratios of expectations of per-path statistics: ``K² = E[accrued] / E[count]``
with ``accrued = (A/N) Σ r_i² I_i`` and ``count = D/N`` (conditional convention), ``accrued =
(A/N) Σ_{i≤τ} r_i²`` and ``count = τ/N`` (knock-out swap), or ``K² = E[accrued]`` (corridor).  The
standard error of a ratio ``R = ā/d̄`` is the delta-method one, ``std(a − R d) / (√n d̄)`` on the
antithetic-pair-averaged samples; all legs are priced on one path set through
:meth:`~volsto.engine.mc.MonteCarlo.price_many` with the per-path payoffs kept.

:func:`lsv_minus_lv` is the study quantity of §6.1: the LSV fair strike minus the pure local-vol
fair strike on the same surface and seed, as a function of the model parameters.
:func:`vko_report` returns the VKO price, the ratio to the vanilla put (the "VKO discount") and
``P(knock-out)`` with standard errors.  Checked by ``tests/test_conditional_variance.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from volsto.engine.mc import MonteCarlo
from volsto.products.conditional_variance import (
    ConditionalVarianceSwap,
    ConvexitySpread,
    KnockOutVarianceSwap,
    RealisedVarianceSchedule,
)
from volsto.products.vko import VolKnockOutPut

if TYPE_CHECKING:
    from volsto.config import SimConfig
    from volsto.engine.grid import TimeGrid
    from volsto.engine.rng import GaussianDraws
    from volsto.models.base import Model

FloatArray = NDArray[np.float64]


def pair_average(x: FloatArray, antithetic: bool) -> FloatArray:
    """Antithetic pairs averaged (adjacent paths), the independent samples behind the errors."""
    return 0.5 * (x[0::2] + x[1::2]) if antithetic else x


_pairs = pair_average


def mean_and_stderr(x: FloatArray) -> tuple[float, float]:
    n = x.size
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(n)) if n > 1 else float("nan")


def ratio_of_means(a: FloatArray, d: FloatArray) -> tuple[float, float]:
    """``ā/d̄`` with the delta-method standard error ``std(a − R d) / (√n d̄)``."""
    dbar = float(d.mean())
    if dbar <= 0:
        return float("nan"), float("nan")
    r = float(a.mean()) / dbar
    resid = a - r * d
    return r, float(resid.std(ddof=1) / (np.sqrt(a.size) * dbar))


@dataclass(frozen=True)
class FairStrike:
    """Fair strike in annualised variance with its standard error; ``vol`` is ``sqrt``."""

    variance: float
    variance_stderr: float
    convention: str
    n_paths: int
    extras: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def vol(self) -> float:
        return float(np.sqrt(self.variance))

    @property
    def vol_stderr(self) -> float:
        return float(self.variance_stderr / (2.0 * np.sqrt(self.variance)))

    def __repr__(self) -> str:
        ex = ", ".join(f"{k}={v:.4g}±{s:.2g}" for k, (v, s) in self.extras.items())
        return (
            f"FairStrike({self.vol:.4%} ± {self.vol_stderr:.3%} vol, {self.convention}, "
            f"n_paths={self.n_paths}{', ' + ex if ex else ''})"
        )


def fair_strike(
    product: RealisedVarianceSchedule,
    model: Model,
    sim: SimConfig,
    *,
    grid: TimeGrid | None = None,
    draws: GaussianDraws | None = None,
) -> FairStrike:
    """Model fair strike of a conditional / corridor / knock-out variance swap."""
    names = ["accrued", "count"]
    if isinstance(product, KnockOutVarianceSwap):
        names += ["ko", "tau"]
        convention = "knock-out"
    elif isinstance(product, ConditionalVarianceSwap):
        convention = product.convention
    else:
        raise TypeError("fair_strike expects a conditional or knock-out variance swap")
    res = MonteCarlo(sim).price_many(
        [product.leg(n) for n in names], model, grid=grid, draws=draws, keep_payoffs=True
    )
    arrays = {n: _pairs(np.asarray(r.payoffs), sim.antithetic) for n, r in zip(names, res)}
    extras: dict[str, tuple[float, float]] = {"count": mean_and_stderr(arrays["count"])}
    if convention == "corridor":
        k2, se = mean_and_stderr(arrays["accrued"])
    else:
        k2, se = ratio_of_means(arrays["accrued"], arrays["count"])
    if convention == "knock-out":
        extras["p_ko"] = mean_and_stderr(arrays["ko"])
        extras["e_tau"] = mean_and_stderr(arrays["tau"])
    return FairStrike(k2, se, convention, sim.n_paths, extras)


def strike_differential(
    spread: ConvexitySpread, model: Model, sim: SimConfig
) -> tuple[float, float]:
    """``K_up² − K_var²`` of a convexity spread with a joint delta-method standard error (all
    legs on one path set)."""
    up = spread.upvar
    vs = spread.varswap.floating_leg()
    res = MonteCarlo(sim).price_many(
        [up.leg("accrued"), up.leg("count"), vs], model, keep_payoffs=True
    )
    a = _pairs(np.asarray(res[0].payoffs), sim.antithetic)
    d = _pairs(np.asarray(res[1].payoffs), sim.antithetic)
    rv = _pairs(np.asarray(res[2].payoffs), sim.antithetic) / (
        float(vs.df(vs.maturity)) * vs.notional
    )
    if up.convention == "corridor":
        g = a - rv
        return float(g.mean()), float(g.std(ddof=1) / np.sqrt(g.size))
    dbar = float(d.mean())
    r = float(a.mean()) / dbar
    g = a / dbar - r * d / dbar - rv
    return float(r - rv.mean()), float(g.std(ddof=1) / np.sqrt(g.size))


def lsv_minus_lv(
    product: RealisedVarianceSchedule, lsv_model: Model, lv_model: Model, sim: SimConfig
) -> tuple[float, float, FairStrike, FairStrike]:
    """The study quantity: LSV fair vol minus pure-LV fair vol (same seed), with the standard
    error of the difference (independent path sets: root sum of squares)."""
    fs_lsv = fair_strike(product, lsv_model, sim)
    fs_lv = fair_strike(product, lv_model, sim)
    return (
        fs_lsv.vol - fs_lv.vol,
        float(np.hypot(fs_lsv.vol_stderr, fs_lv.vol_stderr)),
        fs_lsv,
        fs_lv,
    )


@dataclass(frozen=True)
class VKOReport:
    """VKO price, vanilla put price, their ratio (the VKO discount) and ``P(KO)``, one path set."""

    price: float
    price_stderr: float
    vanilla: float
    vanilla_stderr: float
    discount: float
    discount_stderr: float
    p_ko: float
    p_ko_stderr: float
    n_paths: int

    def __repr__(self) -> str:
        return (
            f"VKOReport(price {self.price:.5g} ± {self.price_stderr:.2g}, vanilla "
            f"{self.vanilla:.5g} ± {self.vanilla_stderr:.2g}, discount {self.discount:.4f} ± "
            f"{self.discount_stderr:.4f}, P(KO) {self.p_ko:.4f} ± {self.p_ko_stderr:.4f}, "
            f"n_paths={self.n_paths})"
        )


def vko_report(vko: VolKnockOutPut, model: Model, sim: SimConfig) -> VKOReport:
    """Price the VKO put, its vanilla and the knock-out indicator on one path set."""
    if vko.knock_in:
        raise ValueError("vko_report expects the knock-out put")
    res = MonteCarlo(sim).price_many([vko, vko.vanilla(), vko.leg("ko")], model, keep_payoffs=True)
    p = _pairs(np.asarray(res[0].payoffs), sim.antithetic)
    v = _pairs(np.asarray(res[1].payoffs), sim.antithetic)
    ko = _pairs(np.asarray(res[2].payoffs), sim.antithetic)
    vbar = float(v.mean())
    ratio = float(p.mean()) / vbar
    g = p / vbar - ratio * v / vbar
    p_ko, p_ko_se = mean_and_stderr(ko)
    return VKOReport(
        res[0].mean,
        res[0].stderr,
        res[1].mean,
        res[1].stderr,
        ratio,
        float(g.std(ddof=1) / np.sqrt(g.size)),
        p_ko,
        p_ko_se,
        sim.n_paths,
    )
