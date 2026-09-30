"""The payoff study's stage 1 (``volsto/studies/payoff.py``): the book's products and
conventions, the hedge task list, the model-risk and rotation-greek tables.  No calibration, no
pricing of the anchor (the parts run in ``scripts/payoff_study.py``)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volsto.market.curves import ForwardCurve
from volsto.products.barrier import KnockInOption, KnockOutOption
from volsto.products.conditional_variance import ConditionalVarianceSwap, KnockOutVarianceSwap
from volsto.products.variance import VarianceOption
from volsto.products.vko import VolKnockOutPut
from volsto.studies.payoff import (
    HEDGE_REGIMES,
    LV_WORLD_STRATEGIES,
    REGIME_PRODUCTS,
    REPRESENTATIVES,
    HedgeTask,
    hedge_tasks,
    model_risk_table,
    payoff_book,
    rotation_greek,
)


def _book() -> dict:  # type: ignore[type-arg]
    fc = ForwardCurve.flat(4000.0, 0.04, 0.015)
    return payoff_book(4000.0, fc.rate_curve, lambda T: 0.2 + 0.01 * T)


def test_book_products_and_conventions() -> None:
    book = _book()
    assert len(book) == 26 and set(REPRESENTATIVES) <= set(book)
    fams = {e.family for e in book.values()}
    assert fams == {"uoc", "dop", "dip", "put on var", "ko var", "up var", "down var", "vko"}
    uoc = book["uoc 6m 110"].product
    assert isinstance(uoc, KnockOutOption) and uoc.direction == "up" and uoc.cp == 1
    assert uoc.barrier == pytest.approx(4400.0) and uoc.strike == 4000.0
    assert uoc.monitoring == "discrete" and uoc.strict and uoc.notional == pytest.approx(1 / 4000)
    dip = book["dip 1y 80"].product
    assert isinstance(dip, KnockInOption) and dip.barrier == pytest.approx(3200.0)
    pv = book["put on var 1y 80"]
    assert isinstance(pv.product, VarianceOption) and pv.product.cp == -1
    assert pv.strike_vol == pytest.approx(0.8 * 0.21)
    assert pv.product.notional == pytest.approx(1 / (2 * pv.strike_vol))
    ko = book["ko var 3m 101"]
    assert isinstance(ko.product, KnockOutVarianceSwap) and ko.product.settlement == "knock_out"
    assert ko.product.barrier == pytest.approx(4040.0)
    assert ko.strike_vol == pytest.approx(0.2025)
    for side in ("up", "down"):
        cv = book[f"{side} var 1y"].product
        assert isinstance(cv, ConditionalVarianceSwap) and cv.convention == "corridor"
        assert cv.side == side and cv.barrier == 4000.0
    assert isinstance(book["vko put 12m"].product, VolKnockOutPut)
    assert {e.unit for e in book.values()} == {"% of spot", "vol pts (vega notional 1)"}


def test_hedge_task_list() -> None:
    names = {
        p: ["delta", "preset", "PCS static (carry, BGK) no delta", "x"] for p in REPRESENTATIVES
    }
    tasks = hedge_tasks(names)
    per = len(REPRESENTATIVES)
    lv = sum(1 for s in names[REPRESENTATIVES[0]] if s in LV_WORLD_STRATEGIES)
    assert len(tasks) == per * 4 + per * lv + len(REGIME_PRODUCTS) * len(HEDGE_REGIMES)
    assert len({t.key for t in tasks}) == len(tasks)
    assert HedgeTask("up var 1y", "corridor strip + var swap").key.isascii()
    assert all(t.world in ("2F", "LV") for t in tasks)


def test_model_risk_and_rotation_tables() -> None:
    rows = []
    for m, v in (("LV", 1.0), ("1F", 1.3), ("2F", 1.1), ("BS (ATM vol)", 0.9)):
        rows.append({"product": "p", "model": m, "unit": "u", "value": v, "stderr": 0.01})
    mr = model_risk_table(pd.DataFrame(rows))
    r = mr.iloc[0]
    assert r["spread"] == pytest.approx(0.3) and r["2F_minus_LV"] == pytest.approx(0.1)
    assert r["spread_stderr"] == pytest.approx(np.hypot(0.01, 0.01))
    rot = pd.DataFrame(
        [
            {"product": "p", "unit": "u", "rota": k, "value": v, "stderr": 0.02}
            for k, v in ((-2.0, 0.6), (-1.0, 0.8), (0.0, 1.0), (1.0, 1.2), (2.0, 1.5))
        ]
    )
    g = rotation_greek(rot).iloc[0]
    assert g["rotation_greek"] == pytest.approx(0.2) and g["convexity"] == pytest.approx(0.1)
