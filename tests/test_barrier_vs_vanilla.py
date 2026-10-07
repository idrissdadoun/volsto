"""The barrier-versus-vanilla study's stage 1 (``volsto/studies/barrier_vs_vanilla.py``): the
book, the decomposition and decision tables on synthetic frames, the regret indicator against
the knock-outs path by path.  No calibration, no anchor pricing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volsto.config import SimConfig
from volsto.engine.mc import MonteCarlo
from volsto.market.curves import ForwardCurve
from volsto.models.bs import BlackScholes
from volsto.products.barrier import KnockOutOption
from volsto.products.structures import VanillaStructure
from volsto.studies.barrier_vs_vanilla import (
    DOWN_BARRIERS,
    MATURITIES,
    REPRESENTATIVES,
    UP_BARRIERS,
    decision_table,
    decomposition_table,
    hedge_tasks,
    study_book,
    surface_rows,
    touch_products,
)


def test_book() -> None:
    fc = ForwardCurve.flat(4000.0, 0.04, 0.015)
    book = study_book(4000.0, fc.rate_curve)
    n_lvl = len(UP_BARRIERS) + len(DOWN_BARRIERS)
    assert len(book) == len(MATURITIES) * n_lvl * 6
    assert set(REPRESENTATIVES) <= set(book)
    uoc = book["uoc 6m 110"].product
    assert isinstance(uoc, KnockOutOption) and uoc.monitoring == "discrete" and uoc.strict
    cont = book["uoc 6m 110 cont"].product
    assert isinstance(cont, KnockOutOption) and cont.monitoring == "continuous"
    eko = book["eko 6m 110"].product
    assert isinstance(eko, KnockOutOption) and eko.fixing_times.tolist() == [0.5]
    fly = book["fly 6m 110"].structure
    assert isinstance(fly, VanillaStructure) and fly.strikes == (4000.0, 4200.0, 4400.0)
    assert fly.leg_weights == (1.0, -2.0, 1.0) and fly.notional == pytest.approx(1 / 4000)
    pr = book["pratio 6m 90"].structure
    assert pr.strikes == (3800.0, 4000.0) and pr.cps == (-1, -1) and pr.leg_weights == (-2.0, 1.0)
    assert {e.family for e in book.values()} == {"barrier", "vanilla"}
    assert len(hedge_tasks()) == 16


def test_regret_indicator_is_the_knock_outs_complement(ssvi) -> None:  # type: ignore[no-untyped-def]
    """``EKO − UOC`` in payoff terms is the regret paths' vanilla payoff; the indicator counts
    exactly those paths (and the one-touch the touched ones)."""
    fc = ssvi.forward_curve
    spot, dc = fc.spot, fc.rate_curve
    book = study_book(spot, dc)
    touch, regret = touch_products(spot, dc, 0.25, 1.10, "up")
    prods = [book["uoc 3m 110"].product, book["eko 3m 110"].product, touch, regret]
    model = BlackScholes(0.25, fc)
    res = MonteCarlo(SimConfig(n_paths=4000, chunk_size=4000, seed=2, dt_max=1 / 252)).price_many(
        prods, model, keep_payoffs=True
    )
    uoc, eko, pt, pr = (np.asarray(r.payoffs) for r in res)
    gap = eko - uoc  # the vanilla payoff on the regret paths (discounted, notional 1/spot)
    assert np.all((gap > 0) == (pr > 0))
    assert np.all(pr <= pt + 1e-12)
    rows = surface_rows([book["eko 3m 110"], book["fly 3m 110"]], ssvi)
    assert [r["product"] for r in rows] == ["eko 3m 110", "fly 3m 110"] and rows[0]["value"] > 0


def _frames(ssvi):  # type: ignore[no-untyped-def]
    fc = ssvi.forward_curve
    spot, dc = fc.spot, fc.rate_curve
    book = study_book(spot, dc)
    rows = surface_rows(list(book.values()), ssvi, model="surface")
    # synthetic model prices: the knock-outs at 40% of the European value, continuous at 35%
    for e in book.values():
        if e.family != "barrier" or e.role == "eko":
            continue
        eko_name = (
            f"eko {e.tag} {round(100 * e.level)}"
            if e.side == "up"
            else f"ekp {e.tag} {round(100 * e.level)}"
        )
        ek = next(r["value"] for r in rows if r["product"] == eko_name)
        share = 0.35 if e.role == "uoc cont" else 0.4
        for m, bump in (("LV", 0.9), ("1F", 1.05), ("2F", 1.0), ("BS (ATM vol)", 0.95)):
            if e.role == "uoc cont" and m == "BS (ATM vol)":
                continue
            rows.append(
                {
                    "product": e.name,
                    "family": "barrier",
                    "tag": e.tag,
                    "T": e.T,
                    "side": e.side,
                    "level": e.level,
                    "role": e.role,
                    "value": share * ek * bump,
                    "stderr": 0.01,
                    "n_paths": 1,
                    "wall_s": 0.0,
                    "model": m,
                }
            )
        if e.role == "uoc":
            for kind, val in (("p_touch", 0.3), ("p_regret", 0.1)):
                rows.append(
                    {
                        "product": e.name,
                        "family": "probability",
                        "tag": e.tag,
                        "T": e.T,
                        "side": e.side,
                        "level": e.level,
                        "role": kind,
                        "value": val,
                        "stderr": 0.0,
                        "n_paths": 1,
                        "wall_s": 0.0,
                        "model": "2F",
                    }
                )
    return pd.DataFrame(rows), book, spot, dc


def test_decomposition_and_decision_tables(ssvi) -> None:  # type: ignore[no-untyped-def]
    anchor, _book, spot, dc = _frames(ssvi)
    dec = decomposition_table(anchor, ssvi, spot, dc)
    assert len(dec) == len(MATURITIES) * (len(UP_BARRIERS) + len(DOWN_BARRIERS))
    r = dec[(dec["tag"] == "6m") & (dec["side"] == "up") & (dec["level"] == 1.10)].iloc[0]
    assert r["regret_share"] == pytest.approx(0.6)
    assert r["continuous_discount"] == pytest.approx(r["2F"] - r["2F_cont"])
    assert r["model_spread"] == pytest.approx(0.15 * r["EKO_surface"] * 0.4, rel=1e-6)
    assert np.isfinite(r["ratio_matched"]) and np.isfinite(r["fly_matched_top"])
    assert 1.0 < r["fly_matched_top"] < 1.10
    assert r["fly_units_for_premium"] == pytest.approx(r["2F"] / r["fly"])
    # a synthetic P-measure layer: payoffs equal to the model's share → indifferent region
    hist_rows = []
    for _, d in dec.iterrows():
        lvl, side = d["level"], d["side"]
        base = {
            "tag": d["tag"],
            "T": d["T"],
            "side": side,
            "level": lvl,
            "horizon_days": 63,
            "group": "all",
            "regime": "all",
        }
        hist_rows.append(
            {**base, "layer": "2F mark (Q)", "touched": 0.3, "regret": 0.1, "uoc": d["2F"] / 100}
        )
        hist_rows.append(
            {
                **base,
                "layer": "FHS at implied ATM",
                "touched": 0.35,
                "regret": 0.12,
                "uoc": d["2F"] / 100,
                "fly": d["fly"] / 100,
                "spread": d["spread"] / 100,
                "ratio": d["ratio_1x2"] / 100,
            }
        )
    hist = pd.DataFrame(hist_rows)
    dt = decision_table(dec, hist)
    assert len(dt) == len(dec)
    row = dt.iloc[0]
    assert row["uoc_payoff_to_premium"] == pytest.approx(1.0)
    assert row["fly_payoff_to_premium"] == pytest.approx(1.0)
    assert row["touch_gap_P_minus_Q"] == pytest.approx(0.05)
    assert row["verdict_vs_fly"] in {"barrier", "fly", "indifferent (inside the model band)"}
