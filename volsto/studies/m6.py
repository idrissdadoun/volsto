"""M6 headline rows (SPEC §9.1 "Headline additions (M6)", §6.6; owner's words at M6):

    "Add to the headline table: a 3y annual autocall (AC 100%, c = 6% p.a., European KI 60%) and
    a 3y Phoenix (CB 70%, memory, American KI 60% daily), price, expected life, P(KI), for
    ω = 0/1/2/3 in 1F and the Table 8.2 2F set, 8e5 particles."

Term sheets (:func:`headline_products`; levels in % of the spot the note is struck on,
``spot_reference = spot``, notional 1 so every price is a fraction of notional):

* ``"autocall 3y"`` — annual observation dates 1y, 2y, 3y; autocall barrier 100% at every date;
  growing coupon ``c_i = i × 6%`` paid with the redemption at the autocall date (the ``coupons =
  0.06`` float convention of :class:`~volsto.products.autocall.Autocall`); European knock-in at
  60% (``S_{T_N} < 0.6 S_0``); put strike 100% geared 1:1.
* ``"phoenix 3y"`` — the same dates and autocall barrier; coupon barrier 70% with memory;
  American knock-in at 60% monitored daily (252 fixings per year, the library's daily
  convention, ``t = 0`` included, 757 dates); ``final_redemption = "knock_in"`` — the coupon
  decision at maturity and the knock-in redemption are independent, the market-standard reading
  flagged as an owner decision in the SPEC §6.6 notes.  **The owner gave no Phoenix coupon
  rate: the runner uses 6% per period** (``PHOENIX_COUPON``), the same headline rate as the
  autocall's — an owner decision to confirm.  M6 Part 2 priced the same headline Phoenix at
  **5%** per period (``tests/test_autocall.py::_phoenix``, ``coupon=0.05``; the SPEC §6.8
  measured line "0.9921 ± 0.0008" is that 5% note), so its number and this table's Phoenix
  row differ by the coupon leg, not by the model: on the fast test bed (BS 20%, 20k paths,
  dt 1/50, seed 2024, the same paths) 0.9918 ± 0.0011 at 5% versus 1.0080 ± 0.0011 at 6%,
  the put leg −0.0515 ± 0.0009 and ``P(KI)`` 0.1356 ± 0.0022 identical.  The integrator
  records the owner's rate once and aligns both places.

:func:`run_m6_headline` prices both notes for every model on **one path set per model** (the
grid is the union of the two notes' fixings — the Phoenix's daily schedule, which contains the
annual dates, so it is exactly the Phoenix's own grid — and the Gaussian draws are shared, so the
two rows of a model see the same Brownian paths) through
:func:`volsto.analytics.autocall.autocall_report`, with the same seed for every model.  The
models run on their own grids (the leverage slices differ), so **LSV minus LV differences are
not paired**: the ``<x>_minus_ref`` columns carry the root-sum-square error of two independent
estimates, the layout of :func:`volsto.analytics.autocall.lsv_minus_lv_table`.  Every number
carries its standard error.  The table holds, per (model, product): price, expected life
``E[T_τ]``, ``P(KI) = E[ki_hit]`` (breach and no autocall, the event driving the put leg),
``P(breach)``, ``P(no autocall)``, ``P(first autocall at T_i)`` per date and every leg of
``decompose()`` (the American knock-in put with its European counterpart).  Under Black–Scholes
``P(first autocall at T_1) = N(d₂)`` at ``AC_1 S_0`` (``tests/test_m6_headline.py::
test_first_autocall_probability_is_the_digital``); the fast tests run the two Black–Scholes
models of ``tests/test_m6_headline.py``, the production table comes from
``scripts/m6_headline.py`` (400k paths, seed 2024, 8·10⁵-particle calibrations).

:func:`regression_keys` lists the ``(model, key)`` pairs a future baseline (the M6 counterpart of
``tests/test_m4_regression.py``) would pin and :func:`baseline_values` reads them off a result.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from volsto.analytics.autocall import AutocallReport, autocall_report
from volsto.config import SimConfig
from volsto.engine.mc import MonteCarlo
from volsto.market.curves import DiscountCurve
from volsto.models.base import Model
from volsto.products.autocall import Autocall, Phoenix
from volsto.products.base import daily_schedule
from volsto.studies.m4 import HEADLINE_OMEGAS, LV_NAME, TWO_FACTOR_NAME, one_factor_name

log = logging.getLogger(__name__)

#: Annual observation dates (years); the last one is the maturity.
OBSERVATION_TIMES: tuple[float, ...] = (1.0, 2.0, 3.0)
#: Autocall barrier at every date (fraction of spot).
AUTOCALL_BARRIER = 1.0
#: Autocall coupon rate ``c`` per annum: ``c_i = i × c`` paid at the autocall date.
AUTOCALL_COUPON = 0.06
#: Phoenix coupon per period — NOT given by the owner (module docstring): 6% per period.  M6
#: Part 2 (``tests/test_autocall.py::_phoenix``, the SPEC §6.8 "0.9921 ± 0.0008" line) used 5%:
#: the two Phoenix prices differ by the coupon leg (0.9918 vs 1.0080 ± 0.0011 on the test bed,
#: same paths), not by the model.  Owner decision; the integrator aligns both places.
PHOENIX_COUPON = 0.06
#: Phoenix coupon barrier (fraction of spot), memory on.
COUPON_BARRIER = 0.70
#: Knock-in level of both notes (fraction of spot).
KI_LEVEL = 0.60
#: Daily knock-in monitoring of the Phoenix: fixings per year (SPEC §6.1 ``A = 252``).
KI_PER_YEAR = 252
AUTOCALL_NAME = "autocall 3y"
PHOENIX_NAME = "phoenix 3y"
#: The headline model set of :func:`volsto.studies.m4.headline_models`, in table order.
HEADLINE_MODEL_NAMES: tuple[str, ...] = (
    LV_NAME,
    *(one_factor_name(w) for w in HEADLINE_OMEGAS),
    TWO_FACTOR_NAME,
)
#: Scalar cells of a row (each with ``_stderr`` and, with a reference, ``_minus_ref`` columns).
SCALAR_CELLS: tuple[str, ...] = ("price", "expected_life", "p_ki", "p_breach", "p_no_autocall")
#: Bookkeeping columns of a row (no ``_stderr`` companion): the labels, ``n_paths``, ``seed``
#: (SPEC §11: every artefact records its seed) and the wall clock.
ROW_META_COLUMNS: tuple[str, ...] = ("model", "product", "term_sheet", "n_paths", "seed", "wall_s")
#: Legs a regression baseline pins (those present for the product's knock-in type).
REGRESSION_LEGS: tuple[str, ...] = ("put", "put_vanilla", "put_digital", "put_european", "coupon")


def headline_products(
    discount: DiscountCurve, spot: float, *, per_year: int = KI_PER_YEAR
) -> dict[str, Autocall]:
    """The two headline term sheets struck on ``spot`` (module docstring), discounted on
    ``discount``: ``{"autocall 3y": Autocall, "phoenix 3y": Autocall}``.  ``per_year`` is the
    Phoenix's daily knock-in monitoring frequency (252: the library's daily convention).
    Checked by ``tests/test_m6_headline.py::test_headline_term_sheets``."""
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("spot must be positive")
    if int(per_year) != per_year or per_year <= 0:
        raise ValueError("per_year must be a positive integer")
    t_n = OBSERVATION_TIMES[-1]
    autocall = Autocall(
        OBSERVATION_TIMES,
        discount,
        spot_reference=float(spot),
        coupons=AUTOCALL_COUPON,
        ki_level=KI_LEVEL,
        ki_type="european",
        autocall_barriers=AUTOCALL_BARRIER,
        final_redemption="knock_in",
        notional=1.0,
    )
    phoenix = Phoenix(
        OBSERVATION_TIMES,
        discount,
        spot_reference=float(spot),
        coupon=PHOENIX_COUPON,
        coupon_barrier=COUPON_BARRIER,
        memory=True,
        ki_level=KI_LEVEL,
        ki_type="american",
        ki_monitoring="discrete",
        ki_fixing_times=daily_schedule(t_n, int(per_year)),
        autocall_barriers=AUTOCALL_BARRIER,
        final_redemption="knock_in",
        notional=1.0,
    )
    return {AUTOCALL_NAME: autocall, PHOENIX_NAME: phoenix}


def _leg_keys(product: Autocall) -> list[str]:
    """The leg names :func:`autocall_report` reports for ``product`` (mirrors ``decompose()`` plus
    the European counterpart of an American knock-in put)."""
    keys = [f"autocall_{i}" for i in range(1, product.n_dates + 1)] + ["bond"]
    if product.has_coupon_leg:
        keys.append("coupon")
    keys += ["put_vanilla", "put_digital"] if product.ki_type == "european" else ["put"]
    if product.ki_type != "european":
        keys.append("put_european")
    return keys


def regression_columns(product: Autocall) -> list[str]:
    """Table columns of ``product`` a baseline pins: the scalar cells, ``p_autocall_i`` per date
    and the knock-in put / coupon legs present for its knock-in type.  Checked by
    ``tests/test_m6_headline.py::test_regression_keys`` (against :func:`baseline_values`) and
    ``test_table_shape_and_errors`` (every pinned cell finite with a positive stderr)."""
    cols = list(SCALAR_CELLS) + [f"p_autocall_{i}" for i in range(1, product.n_dates + 1)]
    cols += [f"leg:{k}" for k in _leg_keys(product) if k in REGRESSION_LEGS]
    return cols


def regression_keys(
    products: Mapping[str, Autocall] | None = None,
    models: Sequence[str] = HEADLINE_MODEL_NAMES,
) -> list[tuple[str, str]]:
    """``(model, key)`` pairs a regression baseline would pin, ``key = "<product>:<column>"``
    over :func:`regression_columns`, in table order (model-major).  ``products`` default to the
    headline term sheets (only their structure matters here, so a unit spot and a zero curve are
    used to build them).  Checked by ``tests/test_m6_headline.py::test_regression_keys``."""
    prods = headline_products(DiscountCurve.flat(0.0), 1.0) if products is None else products
    return [
        (str(m), f"{pname}:{col}")
        for m in models
        for pname, prod in prods.items()
        for col in regression_columns(prod)
    ]


def _cells(rep: AutocallReport) -> dict[str, tuple[float, float]]:
    cells: dict[str, tuple[float, float]] = {
        "price": (rep.price, rep.price_stderr),
        "expected_life": (rep.expected_life, rep.expected_life_stderr),
        "p_ki": (rep.ki_probability, rep.ki_probability_stderr),
        "p_breach": (rep.breach_probability, rep.breach_probability_stderr),
        "p_no_autocall": (
            float(rep.autocall_probabilities[-1]),
            float(rep.autocall_probabilities_stderr[-1]),
        ),
    }
    for i in range(rep.product.n_dates):
        cells[f"p_autocall_{i + 1}"] = (
            float(rep.autocall_probabilities[i]),
            float(rep.autocall_probabilities_stderr[i]),
        )
    cells.update({f"leg:{k}": v for k, v in rep.legs.items()})
    return cells


@dataclass(frozen=True)
class M6HeadlineResult:
    """``table``: one row per (model, product) — ``model``, ``product``, ``term_sheet``, then
    every cell as ``<x>`` / ``<x>_stderr``: the scalar cells of :data:`SCALAR_CELLS` (``price``,
    ``expected_life``, ``p_ki``, ``p_breach``, ``p_no_autocall``), ``p_autocall_i`` per
    observation date (``P(first autocall at T_i)``) and ``leg:<name>`` for every leg of the
    report (``leg:autocall_i``, ``leg:bond``, ``leg:coupon``; ``leg:put`` / ``leg:put_european``
    for an American knock-in, ``leg:put_vanilla`` / ``leg:put_digital`` for a European one; a
    product's absent legs are NaN), with ``<x>_minus_ref`` / ``<x>_minus_ref_stderr`` against
    ``reference`` when there is one (root-sum-square errors, the reference's own rows an exact
    zero); the :data:`ROW_META_COLUMNS` ``n_paths``, ``seed`` and ``wall_s`` (the report's wall
    clock, simulation included) close the row.  The legs and the probabilities are two views of
    the same paths: ``leg:bond = DF(T_N) P(no AC)``, ``leg:autocall_i = DF(T_i) (1 + c_i^AC)
    P(AC at i)`` (``1 + c_i^AC`` the :attr:`~volsto.products.autocall.ConditionalDigital.payout`,
    ``1`` for a Phoenix) and the legs sum to the price, stderr included.  ``reports`` keeps the
    underlying :class:`AutocallReport` per (model, product).  Layout checked by
    ``tests/test_m6_headline.py::test_table_shape_and_errors``, the leg / probability identities
    by ``test_legs_tie_to_the_probabilities``, the markdown by
    ``test_markdown_renders_the_rows``."""

    table: pd.DataFrame
    reports: dict[tuple[str, str], AutocallReport]
    reference: str | None
    n_paths: int
    seed: int

    @property
    def model_names(self) -> list[str]:
        return list(dict.fromkeys(m for m, _ in self.reports))

    @property
    def product_names(self) -> list[str]:
        return list(dict.fromkeys(p for _, p in self.reports))

    def to_markdown(self) -> str:
        """One block per product: the term sheet, then a row per model with every number as
        ``value ± stderr`` (prices and legs in % of notional, life in years, probabilities as
        such) and the difference to the reference model when there is one; the footer states the
        paths, the seed and that the differences are not paired.  Checked by
        ``tests/test_m6_headline.py::test_markdown_renders_the_rows``."""
        t = self.table
        blocks: list[str] = []
        for pname in self.product_names:
            sub = t[t["product"] == pname]
            obs = self.reports[(self.model_names[0], pname)].product.observation_times
            n = int(obs.size)
            leg_cols = [
                c
                for c in t.columns
                if c.startswith("leg:")
                and not c.endswith("_stderr")
                and "_minus_ref" not in c
                and c[4:] in REGRESSION_LEGS
                and bool(np.all(np.isfinite(sub[c].to_numpy(dtype=float))))
            ]
            head = ["price (% notional)", "E[life] (y)", "P(KI)", "P(breach)", "P(no AC)"]
            head += [f"P(AC at {obs[i]:g}y)" for i in range(n)]
            head += [f"{c[4:]} leg (% notional)" for c in leg_cols]
            ref = self.reference
            if ref is not None:
                head += [f"price - {ref} (% notional)", f"E[life] - {ref} (y)", f"P(KI) - {ref}"]
            head.append("wall (s)")
            lines = [
                f"**{pname}**: {sub.iloc[0]['term_sheet']}",
                "",
                "| model | " + " | ".join(head) + " |",
                "|---" * (len(head) + 1) + "|",
            ]
            for _, r in sub.iterrows():
                cells = [f"{100 * r['price']:.3f} ± {100 * r['price_stderr']:.3f}"]
                cells.append(f"{r['expected_life']:.3f} ± {r['expected_life_stderr']:.3f}")
                cells += [f"{r[k]:.4f} ± {r[k + '_stderr']:.4f}" for k in ("p_ki", "p_breach")]
                cells.append(f"{r['p_no_autocall']:.4f} ± {r['p_no_autocall_stderr']:.4f}")
                cells += [
                    f"{r[f'p_autocall_{i + 1}']:.4f} ± {r[f'p_autocall_{i + 1}_stderr']:.4f}"
                    for i in range(n)
                ]
                cells += [f"{100 * r[c]:.3f} ± {100 * r[c + '_stderr']:.3f}" for c in leg_cols]
                if ref is not None:
                    cells.append(
                        f"{100 * r['price_minus_ref']:.3f} ± "
                        f"{100 * r['price_minus_ref_stderr']:.3f}"
                    )
                    cells.append(
                        f"{r['expected_life_minus_ref']:.3f} ± "
                        f"{r['expected_life_minus_ref_stderr']:.3f}"
                    )
                    cells.append(f"{r['p_ki_minus_ref']:.4f} ± {r['p_ki_minus_ref_stderr']:.4f}")
                cells.append(f"{r['wall_s']:.0f}")
                lines.append(f"| {r['model']} | " + " | ".join(cells) + " |")
            blocks.append("\n".join(lines))
        note = (
            f"{self.n_paths} paths per model (seed {self.seed}), one path set per model shared by "
            "the two notes; "
            "the models run on their own grids, so the differences to the reference are "
            "root-sum-square errors of independent estimates, not paired."
        )
        return "\n\n".join([*blocks, note])


def run_m6_headline(
    models: Mapping[str, Model],
    sim: SimConfig,
    products: Mapping[str, Autocall] | None = None,
    *,
    reference: str | None = None,
    allow_spot_mismatch: bool = False,
) -> M6HeadlineResult:
    """Price the M6 headline notes for every model (module docstring).

    ``products`` default to :func:`headline_products` struck on each model's own spot and
    discounted on its own rate curve (the headline models share the market).  An explicit
    mapping is priced as given for every model; every level of a note (autocall, coupon,
    knock-in) is a fraction of its ``spot_reference``, so a note struck away from a model's spot
    changes meaning from model to model and raises ``ValueError`` (before anything is simulated)
    unless ``allow_spot_mismatch=True``, which prices it as given and logs the mismatch at
    WARNING.  ``reference`` names the model the ``_minus_ref`` columns compare to (the pure
    local vol of the study in ``scripts/m6_headline.py``); ``None`` adds no difference columns.
    Checked by ``tests/test_m6_headline.py::test_table_shape_and_errors``,
    ``test_phoenix_row_reproduces_the_analytics_report``,
    ``test_expected_life_and_autocall_probabilities``, ``test_legs_tie_to_the_probabilities``,
    ``test_spot_mismatch_raises_unless_allowed`` and ``test_validation``.
    """
    if not models:
        raise ValueError("no models")
    if reference is not None and reference not in models:
        raise ValueError(f"reference {reference!r} is not one of the models")
    if products is not None and not products:
        raise ValueError("no products")
    per_model: dict[str, dict[str, Autocall]] = {
        name: (
            headline_products(model.forward_curve.rate_curve, model.spot)
            if products is None
            else dict(products)
        )
        for name, model in models.items()
    }
    for name, model in models.items():
        for pname, prod in per_model[name].items():
            if abs(prod.spot_reference - model.spot) > 1e-9 * model.spot:
                msg = (
                    f"{name} / {pname}: spot_reference {prod.spot_reference:g} differs from the "
                    f"model spot {model.spot:g} (every level is a fraction of spot_reference)"
                )
                if not allow_spot_mismatch:
                    raise ValueError(msg + "; pass allow_spot_mismatch=True to price it as given")
                log.warning(msg)
    mc = MonteCarlo(sim)
    reports: dict[tuple[str, str], AutocallReport] = {}
    walls: dict[tuple[str, str], float] = {}
    for name, model in models.items():
        prods = per_model[name]
        # one path set per model: the union grid of both notes and shared Gaussian draws
        grid = mc.build_grid(list(prods.values()), model)
        draws = mc.draws_for(grid, model)
        for pname, prod in prods.items():
            t0 = time.perf_counter()
            rep = autocall_report(prod, model, sim, grid=grid, draws=draws)
            walls[(name, pname)] = time.perf_counter() - t0
            reports[(name, pname)] = rep
            log.info("%s / %s in %.0f s: %r", name, pname, walls[(name, pname)], rep)
    rows: list[dict[str, object]] = []
    for (name, pname), rep in reports.items():
        row: dict[str, object] = {"model": name, "product": pname, "term_sheet": repr(rep.product)}
        ref_cells = _cells(reports[(reference, pname)]) if reference is not None else None
        for key, (v, s) in _cells(rep).items():
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
        row["n_paths"] = sim.n_paths
        row["seed"] = sim.seed
        row["wall_s"] = walls[(name, pname)]
        rows.append(row)
    return M6HeadlineResult(pd.DataFrame(rows), reports, reference, sim.n_paths, sim.seed)


def baseline_values(result: M6HeadlineResult) -> dict[tuple[str, str], tuple[float, float]]:
    """``{(model, "<product>:<column>"): (value, stderr)}`` over :func:`regression_columns` of
    every row — what a regression baseline records (:func:`regression_keys` lists the keys).
    Checked by ``tests/test_m6_headline.py::test_regression_keys``."""
    out: dict[tuple[str, str], tuple[float, float]] = {}
    for (name, pname), rep in result.reports.items():
        row = result.table[(result.table["model"] == name) & (result.table["product"] == pname)]
        r = row.iloc[0]
        for col in regression_columns(rep.product):
            out[(name, f"{pname}:{col}")] = (float(r[col]), float(r[f"{col}_stderr"]))
    return out
