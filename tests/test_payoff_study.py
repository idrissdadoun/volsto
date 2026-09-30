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


def _synthetic_stage1(root) -> None:  # type: ignore[no-untyped-def]
    """Stage-1 tables in the formats scripts/payoff_study.py writes (two products per unit)."""
    d = root / "payoff"
    d.mkdir(parents=True)
    prods = [
        ("uoc 6m 110", "uoc", "% of spot"),
        ("ko var 6m 103", "ko var", "vol pts (vega notional 1)"),
    ]
    rows = []
    for name, fam, unit in prods:
        for m, v in (("LV", 1.0), ("1F", 1.2), ("2F", 1.1), ("BS (ATM vol)", 0.8)):
            rows.append(
                {
                    "product": name,
                    "family": fam,
                    "unit": unit,
                    "value": v,
                    "stderr": 0.01,
                    "n_paths": 100_000,
                    "wall_s": 1.0,
                    "model": m,
                }
            )
    prices = pd.DataFrame(rows)
    prices.to_csv(d / "prices.csv", index=False)
    model_risk_table(prices).to_csv(d / "model_risk.csv", index=False)
    g = []
    for name, fam, unit in prods:
        g += [
            {
                "product": name,
                "family": fam,
                "unit": unit,
                "scale": 100.0,
                "group": "delta",
                "name": "delta",
                "value": 0.3,
                "stderr": 0.01,
                "regime": "model",
            },
            {
                "product": name,
                "family": fam,
                "unit": unit,
                "scale": 100.0,
                "group": "barrier",
                "name": "dprice_dB_pct",
                "value": -0.02,
                "stderr": 0.001,
                "regime": "",
            },
        ]
    g.append(
        {
            "product": "ko var 6m 103",
            "family": "ko var",
            "unit": "vol pts (vega notional 1)",
            "scale": 1.0,
            "group": "product",
            "name": "dP(KO)/dlnS",
            "value": 1.5,
            "stderr": 0.05,
            "regime": "",
        }
    )
    pd.DataFrame(g).to_csv(d / "greeks.csv", index=False)
    dl = []
    for (ssr, eps), shift in (((1.0, 0.10), 0.0), ((1.5, 0.10), 0.05), ((1.0, 0.05), -0.02)):
        for name, fam, unit in prods:
            dl.append(
                {
                    "product": name,
                    "family": fam,
                    "unit": unit,
                    "value": 1.1 + shift,
                    "stderr": 0.01,
                    "ssr_target": ssr,
                    "skew_eps": eps,
                    "status": "interior",
                }
            )
    pd.DataFrame(dl).to_csv(d / "dials.csv", index=False)
    pd.DataFrame(
        [
            {
                "product": n,
                "unit": u,
                "rotation_greek": 0.1,
                "stderr_bound": 0.01,
                "convexity": 0.02,
            }
            for n, _, u in prods
        ]
    ).to_csv(d / "rotation_greek.csv", index=False)
    pd.DataFrame([{"product": n, "rota": 0.0} for n, _, _ in prods]).to_csv(
        d / "rotation.csv", index=False
    )
    h = []
    for name, _, unit in prods:
        for strat, world, std in (
            ("delta", "2F", 0.7),
            ("preset", "2F", 1.3),
            ("delta", "LV", 0.8),
        ):
            row = {
                "product": name,
                "strategy": strat,
                "world": world,
                "regime": "model",
                "unit": unit,
                "product_std": 1.6,
                "std_ratio_zc": std / 1.6,
                "costs_mean": 0.05,
                "wall_s": 10.0,
            }
            for stat, v in (
                ("mean", 0.01),
                ("std", std),
                ("q01", -2.0),
                ("es01", -2.5),
                ("es05", -1.8),
            ):
                for tag in ("zc", "tc"):
                    row[f"{stat}_{tag}"] = v
                    row[f"{stat}_{tag}_stderr"] = 0.02
            h.append(row)
    pd.DataFrame(h).to_csv(d / "hedge.csv", index=False)


def test_report_renders_from_the_stage1_tables(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """compute / tables / figures / narrative on synthetic stage-1 tables: the mixed units are
    split into per-unit tables, the knock-out probability has its own table, every table spec
    reads existing columns, and the narrative names the best hedge."""
    from types import SimpleNamespace

    from volsto.studies import payoff_report as rep

    _synthetic_stage1(tmp_path)
    ctx = SimpleNamespace(params={"dir": "payoff"}, outputs_root=tmp_path)
    res = rep.compute(ctx)  # type: ignore[arg-type]
    have = set(res.tables())
    assert {"prices_pct", "prices_vp", "model_risk_pct", "ko_probability", "dials_vp"} <= have
    assert {"hedge_uoc", "hedge_ko_var", "rotation_pct", "greeks_ko_var"} <= have
    for spec in rep.tables(res):
        for col in spec.columns:
            assert col.key in res.columns(spec.table), (spec.name, col.key)
            res.unit(spec.table, col.key)  # one unit per column
    for fig in rep.figures(res):
        fig.draw(res)
    text = rep.narrative(res)
    assert "Lowest hedged P&L std without costs: **delta**" in text
    assert "{{table:hedge_uoc}}" in text and "{{figure:hedge_std}}" in text
    v, _ = res.value("dials_pct", "uoc 6m 110", "ssr 1.5 eps 0.1")
    assert v == pytest.approx(0.05)
