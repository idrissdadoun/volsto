"""Autocall analytics (SPEC §6.6, M6 Part 2): expected life, autocall and knock-in probabilities,
price attribution by leg, the LSV-minus-LV table and the forward-skew exposure.

Every quantity is a Monte Carlo mean with its standard error.  All statistics of one product
under one model come from a single path set (:meth:`~volsto.engine.mc.MonteCarlo.price_many`
with the per-path payoffs kept; antithetic pairs are averaged before the error, as in
:mod:`volsto.analytics.conditional_variance`).  Probabilities are frequencies of the per-path
indicators — risk-neutral, undiscounted — ``1{first autocall at date i}``, ``ki_hit`` (the
knock-in event that drives the put leg: level breached at a monitoring date of the life *and*
no autocall, the same definition for the European and the American type) reported as ``P(KI)``,
and ``ki_breach`` (the level breached at a monitoring date ``≤`` the termination date, whether
or not the note then autocalled) reported as ``P(breach)``; under Black–Scholes ``P(autocall at
date 1) = N(d₂)`` at the strike ``AC_1 S_ref`` (``tests/test_autocall.py::
test_black_scholes_closed_forms``).  The expected life is ``E[T_{τ}]`` with ``τ`` the first
autocall date (``T_N`` when none).

:func:`leg_attribution` prices the legs of ``decompose()`` on the product's paths (the residual
against the product is reported and is zero up to rounding: the legs sum path by path); the
American knock-in put is accompanied by its European counterpart (SPEC §6.6).
:func:`lsv_minus_lv_table` lays the report out over a mapping of models (one row per model and
product) with the difference to a ``reference`` model (the pure local vol of the study) and the
root-sum-square error of the difference (the models are simulated with the same seed but on
their own grids, so the paths are not paired).

:func:`forward_skew_exposure` is the price change per vol point of 90/110 skew at each observation
date under the ``skew_tent`` perturbation of SPEC §7.6 / :mod:`volsto.risk.ladders`: ``σ → σ +
s κ(k) tent_i(T)`` with ``κ(k) = k_cap tanh(k / k_cap)``, tents on the observation dates and ``s``
sized so ``σ(ln 0.9) − σ(ln 1.1)`` at ``T_i`` rises by ``size`` (:func:`skew_tent_slope`, the
``skew_slope`` of the ladders).  It is written against a *model factory* — a callable returning
the priced model for a perturbation layer (``None`` for the base) — so the analytics need no
risk-engine wiring; :func:`state_model_factory` builds one from a
:class:`~volsto.risk.engine.RiskState` and a model builder (recalibrated LSV, local vol or
Black–Scholes).  Base and bumped prices share the grid and the Gaussian draws (common random
numbers) and the standard error is that of the per-path difference; a perturbation failing the
surface's arbitrage checks is halved and retried (``max_halvings``), the achieved size is
reported and the exposure is per requested unit (``tests/test_autocall.py::
test_forward_skew_exposure``).


Sign conventions worth keeping (measured at M6 under local vol on the reference surface, 3y
annual autocall, owner-confirmed): the funded par note has a **negative rho** (−9.4e-5 ± 0.4e-5
per bp: the discounting of the par redemption, duration ≈ E[life], outweighs the equity legs'
+1.2e-4); "long forward" holds for the delta (+0.0070 per unit spot), the repo delta (negative,
−5.9e-5 per bp) and the equity legs, not for the note's rho.  Do not "fix" the rho sign.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.analytics.conditional_variance import mean_and_stderr, pair_average
from volsto.config import SurfacePerturbation
from volsto.engine.mc import MonteCarlo
from volsto.market.surface import ArbitrageError, saturated_k
from volsto.products.autocall import Autocall, AutocallStatistic, KIPutLeg

if TYPE_CHECKING:
    from volsto.config import SimConfig
    from volsto.engine.grid import TimeGrid
    from volsto.engine.rng import GaussianDraws
    from volsto.models.base import Model
    from volsto.products.base import Product
    from volsto.risk.engine import ModelBuilder, RiskState

FloatArray = NDArray[np.float64]
ModelFactory = Callable[[SurfacePerturbation | None], "Model"]
log = logging.getLogger(__name__)

K90, K110 = float(np.log(0.9)), float(np.log(1.1))
K_CAP = 0.5


@dataclass(frozen=True)
class AutocallReport:
    """Price, expected life, autocall probabilities per date (last entry: never autocalled),
    knock-in probability ``P(KI) = E[ki_hit]``, breach probability ``P(breach) = E[ki_breach]``
    (module docstring) and leg prices of one autocall under one model, one path set.
    ``legs`` maps the leg repr-independent names (``autocall_i``, ``bond``, ``coupon``,
    ``put`` / ``put_vanilla`` / ``put_digital``, ``put_european`` for an American knock-in) to
    ``(price, stderr)``."""

    product: Autocall
    price: float
    price_stderr: float
    expected_life: float
    expected_life_stderr: float
    autocall_probabilities: FloatArray
    autocall_probabilities_stderr: FloatArray
    ki_probability: float
    ki_probability_stderr: float
    breach_probability: float
    breach_probability_stderr: float
    legs: dict[str, tuple[float, float]]
    n_paths: int

    @property
    def legs_total(self) -> float:
        """Sum of the decomposition legs (excludes the European counterpart of an American put)."""
        return float(sum(v for k, (v, _) in self.legs.items() if k != "put_european"))

    def as_frame(self) -> pd.DataFrame:
        """One row per observation date plus a ``"none"`` row: ``P(first autocall at date)``."""
        n = self.product.n_dates
        return pd.DataFrame(
            {
                "date": [*range(1, n + 1), "none"],
                "T": [*self.product.observation_times.tolist(), np.nan],
                "probability": self.autocall_probabilities,
                "stderr": self.autocall_probabilities_stderr,
            }
        )

    def leg_frame(self) -> pd.DataFrame:
        rows = [{"leg": k, "price": v, "stderr": s} for k, (v, s) in self.legs.items()]
        rows.append({"leg": "product", "price": self.price, "stderr": self.price_stderr})
        rows.append(
            {
                "leg": "residual (legs - product)",
                "price": self.legs_total - self.price,
                "stderr": np.nan,
            }
        )
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"AutocallReport(price {self.price:.5g} ± {self.price_stderr:.2g}, life "
            f"{self.expected_life:.4g} ± {self.expected_life_stderr:.2g}y, P(KI) "
            f"{self.ki_probability:.4f} ± {self.ki_probability_stderr:.4f}, P(breach) "
            f"{self.breach_probability:.4f} ± {self.breach_probability_stderr:.4f}, P(AC) by "
            f"date {np.round(self.autocall_probabilities[:-1], 4).tolist()}, "
            f"n_paths={self.n_paths})"
        )


def _leg_name(leg: Product) -> str:
    from volsto.products.autocall import _AutocallLeg

    assert isinstance(leg, _AutocallLeg)
    return leg.key


def autocall_report(
    product: Autocall,
    model: Model,
    sim: SimConfig,
    *,
    grid: TimeGrid | None = None,
    draws: GaussianDraws | None = None,
) -> AutocallReport:
    """Everything on one path set: product, legs, life, autocall dates and knock-in indicator."""
    n = product.n_dates
    legs = product.decompose()
    extra: list[Product] = []
    if product.ki_type != "european":
        extra.append(KIPutLeg(product).european_counterpart())
    stats: list[Product] = [
        AutocallStatistic(product, "life"),
        AutocallStatistic(product, "ki_hit"),
        AutocallStatistic(product, "ki_breach"),
        *[AutocallStatistic(product, "autocall_at", i) for i in range(1, n + 2)],
    ]
    products: list[Product] = [product, *legs, *extra, *stats]
    res = MonteCarlo(sim).price_many(products, model, grid=grid, draws=draws, keep_payoffs=True)
    pairs = [pair_average(np.asarray(r.payoffs), sim.antithetic) for r in res]
    k = 1 + len(legs)
    leg_prices = {_leg_name(leg): (r.mean, r.stderr) for leg, r in zip(legs, res[1:k])}
    if extra:
        leg_prices["put_european"] = (res[k].mean, res[k].stderr)
    k += len(extra)
    life, life_se = mean_and_stderr(pairs[k])
    p_ki, p_ki_se = mean_and_stderr(pairs[k + 1])
    p_breach, p_breach_se = mean_and_stderr(pairs[k + 2])
    probs = [mean_and_stderr(p) for p in pairs[k + 3 :]]
    return AutocallReport(
        product=product,
        price=res[0].mean,
        price_stderr=res[0].stderr,
        expected_life=life,
        expected_life_stderr=life_se,
        autocall_probabilities=np.array([p for p, _ in probs]),
        autocall_probabilities_stderr=np.array([s for _, s in probs]),
        ki_probability=p_ki,
        ki_probability_stderr=p_ki_se,
        breach_probability=p_breach,
        breach_probability_stderr=p_breach_se,
        legs=leg_prices,
        n_paths=sim.n_paths,
    )


def expected_life(product: Autocall, model: Model, sim: SimConfig) -> tuple[float, float]:
    """``E[T_τ]`` (years) with its standard error, ``τ`` the first autocall date or ``N``."""
    rep = autocall_report(product, model, sim)
    return rep.expected_life, rep.expected_life_stderr


def autocall_probabilities(product: Autocall, model: Model, sim: SimConfig) -> pd.DataFrame:
    """``P(first autocall at T_i)`` per date with standard errors; the last row (``date =
    "none"``) is the probability of reaching maturity without autocall."""
    return autocall_report(product, model, sim).as_frame()


def ki_probability(product: Autocall, model: Model, sim: SimConfig) -> tuple[float, float]:
    """``P(KI) = E[ki_hit]``: the knock-in level breached at a monitoring date of the life *and*
    the note not autocalled — the event that drives the put leg, the same definition for the
    European and the American type (the expected weight for the continuous variant).  The
    breach probability regardless of a later autocall is
    :attr:`AutocallReport.breach_probability`."""
    rep = autocall_report(product, model, sim)
    return rep.ki_probability, rep.ki_probability_stderr


def leg_attribution(product: Autocall, model: Model, sim: SimConfig) -> pd.DataFrame:
    """Price per leg of ``decompose()`` with standard errors on the product's own paths, the
    product price and the residual (zero up to rounding); an American knock-in put also shows
    its European counterpart (``put_european``)."""
    return autocall_report(product, model, sim).leg_frame()


def lsv_minus_lv_table(
    models: Mapping[str, Model],
    sim: SimConfig,
    products: Autocall | Sequence[Autocall],
    *,
    reference: str | None = None,
) -> pd.DataFrame:
    """One row per (model, product): price, expected life, ``P(KI)``, ``P(breach)``,
    ``P(no autocall)`` and every leg price with standard errors; with ``reference`` (a key of
    ``models``, the pure local vol of the study) the columns ``<x>_minus_ref`` /
    ``<x>_minus_ref_stderr`` give the difference to that model (root-sum-square errors: the path
    sets are independent; the reference's own row is an exact zero with a zero error)."""
    prods = [products] if isinstance(products, Autocall) else list(products)
    if not prods:
        raise ValueError("no products")
    if reference is not None and reference not in models:
        raise ValueError(f"reference {reference!r} is not one of the models")
    reports: dict[tuple[str, int], AutocallReport] = {}
    for name, model in models.items():
        for j, prod in enumerate(prods):
            reports[(name, j)] = autocall_report(prod, model, sim)

    def cells_of(rep: AutocallReport) -> dict[str, tuple[float, float]]:
        cells = {
            "price": (rep.price, rep.price_stderr),
            "expected_life": (rep.expected_life, rep.expected_life_stderr),
            "p_ki": (rep.ki_probability, rep.ki_probability_stderr),
            "p_breach": (rep.breach_probability, rep.breach_probability_stderr),
            "p_no_autocall": (
                float(rep.autocall_probabilities[-1]),
                float(rep.autocall_probabilities_stderr[-1]),
            ),
        }
        cells.update({f"leg:{k}": v for k, v in rep.legs.items()})
        return cells

    rows = []
    for (name, j), rep in reports.items():
        row: dict[str, object] = {"model": name, "product": j, "term_sheet": repr(rep.product)}
        ref_cells = cells_of(reports[(reference, j)]) if reference is not None else None
        for key, (v, s) in cells_of(rep).items():
            row[key] = v
            row[f"{key}_stderr"] = s
            if ref_cells is not None:
                if name == reference:  # a self-difference is exact
                    row[f"{key}_minus_ref"] = 0.0
                    row[f"{key}_minus_ref_stderr"] = 0.0
                else:
                    rv, rs = ref_cells.get(key, (np.nan, np.nan))
                    row[f"{key}_minus_ref"] = v - rv
                    row[f"{key}_minus_ref_stderr"] = float(np.hypot(s, rs))
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# forward-skew exposure
# ---------------------------------------------------------------------------------------------


def skew_tent_slope(size: float = 0.01, k_cap: float = K_CAP) -> float:
    """``s`` such that ``σ(ln 0.9) − σ(ln 1.1)`` rises by ``size`` under ``s κ(k)`` (negative: the
    put wing rises); identical to :func:`volsto.risk.ladders.skew_slope`."""
    return float(size / float(saturated_k(K90, k_cap) - saturated_k(K110, k_cap)))


def skew_tent_perturbation(
    pillars: Sequence[float], index: int, size: float = 0.01, k_cap: float = K_CAP
) -> SurfacePerturbation:
    """The ``skew_tent`` layer centred at ``pillars[index]`` sized to ``size`` of 90/110 skew."""
    return SurfacePerturbation(
        "skew_tent",
        {
            "pillars": tuple(float(p) for p in pillars),
            "index": int(index),
            "slope": skew_tent_slope(size, k_cap),
            "k_cap": float(k_cap),
        },
    )


def state_model_factory(
    builder: ModelBuilder, state: RiskState, mode: str = "recalibrate"
) -> ModelFactory:
    """A model factory for :func:`forward_skew_exposure` from a risk state and a builder
    (:class:`~volsto.risk.engine.LSVBuilder` / ``LVBuilder`` / ``BSBuilder``): the perturbation is
    composed onto the state's surface layer and the model rebuilt under ``mode``."""

    def make(pert: SurfacePerturbation | None) -> Model:
        st = state if pert is None else state.with_perturbation(pert)
        return builder.build(st, mode)

    return make


def forward_skew_exposure(
    product: Autocall,
    model_for: ModelFactory,
    sim: SimConfig,
    *,
    size: float = 0.01,
    k_cap: float = K_CAP,
    pillars: Sequence[float] | None = None,
    max_halvings: int = 4,
    with_legs: bool = False,
) -> pd.DataFrame:
    """Price change per vol point of 90/110 skew at each observation date (module docstring).

    Columns: ``date``, ``T``, ``requested`` / ``achieved`` (skew size after any halving),
    ``slope`` (the achieved ``s``), ``base`` / ``base_stderr`` and ``bumped`` /
    ``bumped_stderr`` (the two Monte Carlo prices with their own errors), ``exposure`` (scaled
    to one requested unit: ``(bumped − base) · 0.01 / achieved``, per vol point) and
    ``exposure_stderr`` (per-path difference, common random numbers); with ``with_legs`` a
    ``leg`` column adds one row per leg of ``decompose()`` (``leg = "product"`` for the note).
    ``pillars`` default to the observation dates (tents ``T_{i−1} → T_i → T_{i+1}``).
    """
    if size <= 0 or k_cap <= 0:
        raise ValueError("size and k_cap must be positive")
    ps = (
        tuple(float(t) for t in product.observation_times)
        if pillars is None
        else tuple(float(p) for p in pillars)
    )
    if len(ps) != product.n_dates or np.any(np.diff(ps) <= 0):
        raise ValueError("pillars must be increasing with one entry per observation date")
    base_model = model_for(None)
    prods: list[Product] = [product]
    names = ["product"]
    if with_legs:
        legs = product.decompose()
        prods += legs
        names += [_leg_name(leg) for leg in legs]
    mc = MonteCarlo(sim)
    grid = mc.build_grid(prods, base_model)
    draws = mc.draws_for(grid, base_model)
    base = mc.price_many(prods, base_model, grid=grid, draws=draws, keep_payoffs=True)
    base_pairs = [pair_average(np.asarray(r.payoffs), sim.antithetic) for r in base]
    rows = []
    for i, t_i in enumerate(ps):
        achieved = float(size)
        bumped_model: Model | None = None
        last: Exception | None = None
        for _ in range(max_halvings + 1):
            try:
                bumped_model = model_for(skew_tent_perturbation(ps, i, achieved, k_cap))
                break
            except ArbitrageError as exc:
                last = exc
                achieved *= 0.5
        if bumped_model is None:
            raise ArbitrageError(
                f"skew tent at {t_i:g}y fails the arbitrage checks after {max_halvings} "
                f"halvings: {last}"
            )
        if achieved != size:
            log.warning("skew tent at %gy halved to %.3g of %.3g", t_i, achieved, size)
        bumped = mc.price_many(prods, bumped_model, grid=grid, draws=draws, keep_payoffs=True)
        scale = 0.01 / achieved
        for name, rb, pb, rbump in zip(names, base, base_pairs, bumped):
            diff = scale * (pair_average(np.asarray(rbump.payoffs), sim.antithetic) - pb)
            rows.append(
                {
                    "date": i + 1,
                    "T": t_i,
                    "leg": name,
                    "requested": size,
                    "achieved": achieved,
                    "slope": skew_tent_slope(achieved, k_cap),
                    "base": rb.mean,
                    "base_stderr": rb.stderr,
                    "bumped": rbump.mean,
                    "bumped_stderr": rbump.stderr,
                    "exposure": float(diff.mean()),
                    "exposure_stderr": float(diff.std(ddof=1) / np.sqrt(diff.size)),
                }
            )
    frame = pd.DataFrame(rows)
    return frame if with_legs else frame.drop(columns="leg")
