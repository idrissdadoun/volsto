"""M6 headline rows (``volsto.studies.m6``, SPEC §9.1 headline additions) on the Black–Scholes
bed: the two term sheets, the table layout (every number with its standard error), the vol
ordering of price and P(KI) at 3 stderr, the expected life, the American-daily versus European
knock-in ordering, P(first autocall at T_1) = N(d₂), the leg cells tied to the probability
cells (bond = DF P(no AC), autocall_i = DF payout P(AC at i), legs sum to the price), the
Phoenix row reproducing :func:`autocall_report` bit for bit on the shared path set, the
spot-reference guard, the regression keys, the markdown, the runtime budget of the daily Phoenix
at 20k paths and ``scripts/m6_headline.py`` end to end on monkeypatched Black–Scholes models."""

from __future__ import annotations

import importlib.util
import json
import logging
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest

from volsto.analytics.autocall import autocall_report
from volsto.calibration import LeverageCache
from volsto.config import CalibrationSpec, SimConfig
from volsto.market import DiscountCurve, ForwardCurve, bs_digital_price
from volsto.models import BlackScholes
from volsto.products.autocall import Autocall, ConditionalDigital
from volsto.studies.m4 import LV_NAME
from volsto.studies.m6 import (
    AUTOCALL_COUPON,
    AUTOCALL_NAME,
    COUPON_BARRIER,
    HEADLINE_MODEL_NAMES,
    KI_LEVEL,
    OBSERVATION_TIMES,
    PHOENIX_COUPON,
    PHOENIX_NAME,
    ROW_META_COLUMNS,
    SCALAR_CELLS,
    M6HeadlineResult,
    baseline_values,
    headline_products,
    regression_columns,
    regression_keys,
    run_m6_headline,
)

ROOT = Path(__file__).resolve().parents[1]
VOLS = {"bs20": 0.20, "bs25": 0.25}
SIM = SimConfig(n_paths=20_000, dt_max=1.0 / 50.0, chunk_size=20_000, seed=2024)
SMALL = SimConfig(n_paths=2_000, dt_max=1.0 / 12.0, chunk_size=2_000, seed=3)
PRODUCTS = (AUTOCALL_NAME, PHOENIX_NAME)


@pytest.fixture(scope="module")
def models(forward_curve: ForwardCurve) -> dict[str, BlackScholes]:
    return {name: BlackScholes(vol, forward_curve) for name, vol in VOLS.items()}


@pytest.fixture(scope="module")
def result(models: dict[str, BlackScholes]) -> M6HeadlineResult:
    return run_m6_headline(models, SIM, reference="bs20")


def _row(result: M6HeadlineResult, model: str, product: str) -> pd.Series:
    t = result.table
    return t[(t["model"] == model) & (t["product"] == product)].iloc[0]


def test_headline_term_sheets(discount: DiscountCurve) -> None:
    prods = headline_products(discount, 100.0)
    assert list(prods) == [AUTOCALL_NAME, PHOENIX_NAME]
    ac, ph = prods[AUTOCALL_NAME], prods[PHOENIX_NAME]
    for p in (ac, ph):
        assert isinstance(p, Autocall)
        np.testing.assert_array_equal(p.observation_times, OBSERVATION_TIMES)
        np.testing.assert_array_equal(p.autocall_barriers, 1.0)
        assert p.spot_reference == 100.0 and p.notional == 1.0 and p.ki_level == KI_LEVEL == 0.6
        assert p.final_redemption == "knock_in" and p.non_call_periods == 0
    # autocall: growing coupon c_i = i x 6% at the autocall date, European KI 60% at maturity
    assert not ac.is_phoenix and ac.coupons == AUTOCALL_COUPON == 0.06
    np.testing.assert_allclose(ac.coupon_schedule, [0.06, 0.12, 0.18])
    assert ac.ki_type == "european" and ac.ki_fixing_times is None
    # Phoenix: CB 70% with memory, 6% per period (owner decision pending), American KI daily
    assert ph.is_phoenix and ph.memory and ph.coupon_barrier == COUPON_BARRIER == 0.7
    np.testing.assert_allclose(ph.coupon_schedule, [PHOENIX_COUPON] * 3)
    assert ph.ki_type == "american" and ph.ki_monitoring == "discrete"
    assert ph.ki_fixing_times is not None and ph.ki_fixing_times.size == 3 * 252 + 1
    assert ph.ki_fixing_times[0] == 0.0 and ph.ki_fixing_times[-1] == 3.0
    assert "Phoenix" in repr(ph) and "growing" in repr(ac)
    weekly = headline_products(discount, 100.0, per_year=52)[PHOENIX_NAME]
    assert weekly.ki_fixing_times is not None and weekly.ki_fixing_times.size == 3 * 52 + 1
    with pytest.raises(ValueError):
        headline_products(discount, -1.0)
    with pytest.raises(ValueError):
        headline_products(discount, 100.0, per_year=0)


def test_table_shape_and_errors(result: M6HeadlineResult) -> None:
    """One row per (model, product); every number carries a finite positive standard error, the
    reference's own differences are exact zeros and the others carry root-sum-square errors."""
    t = result.table
    assert len(t) == 4 and result.n_paths == SIM.n_paths and result.seed == SIM.seed
    assert list(t["model"]) == ["bs20", "bs20", "bs25", "bs25"]
    assert list(t["product"]) == [*PRODUCTS, *PRODUCTS]
    assert set(ROW_META_COLUMNS) <= set(t.columns)
    values = [
        c
        for c in t.columns
        if c not in ROW_META_COLUMNS and not c.endswith("_stderr") and not c.endswith("_minus_ref")
    ]
    assert set(SCALAR_CELLS) <= set(values)
    assert {"p_autocall_1", "p_autocall_2", "p_autocall_3", "leg:bond"} <= set(values)
    for c in values:
        assert f"{c}_stderr" in t.columns and f"{c}_minus_ref" in t.columns
        assert f"{c}_minus_ref_stderr" in t.columns
    for model, product in zip(t["model"], t["product"], strict=True):
        r = _row(result, model, product)
        for c in regression_columns(result.reports[(model, product)].product):
            assert np.isfinite(r[c]) and np.isfinite(r[f"{c}_stderr"]) and r[f"{c}_stderr"] > 0
            if model == "bs20":
                assert r[f"{c}_minus_ref"] == 0.0 and r[f"{c}_minus_ref_stderr"] == 0.0
            else:
                ref = _row(result, "bs20", product)
                assert r[f"{c}_minus_ref"] == pytest.approx(r[c] - ref[c])
                assert r[f"{c}_minus_ref_stderr"] == pytest.approx(
                    float(np.hypot(r[f"{c}_stderr"], ref[f"{c}_stderr"]))
                )
    # legs of the other knock-in type are absent from a product's rows
    assert np.isnan(_row(result, "bs20", AUTOCALL_NAME)["leg:put"])
    assert np.isnan(_row(result, "bs20", PHOENIX_NAME)["leg:put_vanilla"])
    assert (t["n_paths"] == SIM.n_paths).all() and (t["wall_s"] > 0).all()
    assert (t["seed"] == SIM.seed).all()


def test_legs_tie_to_the_probabilities(result: M6HeadlineResult, discount: DiscountCurve) -> None:
    """The leg cells and the probability cells are two views of the same paths, so they tie
    exactly (values and stderr, both ``std(ddof=1)/sqrt(n)`` of the same pair-averaged samples):
    the legs sum to the price, ``leg:bond = DF(T_N) P(no AC)`` and ``leg:autocall_i = DF(T_i)
    payout_i P(AC at i)`` with ``payout_i = 1 + c_i`` (growing coupon) for the autocall and
    ``1`` for the Phoenix — a mis-sliced ``p_autocall_i`` or a leg mapped to the wrong column
    would break one of them."""
    t = result.table
    for (model, product), rep in result.reports.items():
        r = _row(result, model, product)
        prod = rep.product
        legs = [
            c
            for c in t.columns
            if c.startswith("leg:")
            and not c.endswith("_stderr")
            and "_minus_ref" not in c
            and c != "leg:put_european"
            and np.isfinite(r[c])
        ]
        assert len(legs) == len(prod.decompose())
        assert sum(r[c] for c in legs) == pytest.approx(r["price"], abs=1e-12), (model, product)
        assert rep.legs_total == pytest.approx(rep.price, abs=1e-12)
        df_n = float(discount.df(prod.maturity_date))
        assert r["leg:bond"] == pytest.approx(df_n * r["p_no_autocall"], abs=1e-12)
        assert r["leg:bond_stderr"] == pytest.approx(df_n * r["p_no_autocall_stderr"], abs=1e-12)
        for i in range(1, prod.n_dates + 1):
            payout = ConditionalDigital(prod, i).payout
            expected_payout = 1.0 if product == PHOENIX_NAME else 1.0 + i * AUTOCALL_COUPON
            assert payout == pytest.approx(expected_payout)
            scale = float(discount.df(prod.observation_times[i - 1])) * payout
            assert r[f"leg:autocall_{i}"] == pytest.approx(
                scale * r[f"p_autocall_{i}"], abs=1e-12
            ), (model, product, i)
            assert r[f"leg:autocall_{i}_stderr"] == pytest.approx(
                scale * r[f"p_autocall_{i}_stderr"], abs=1e-12
            ), (model, product, i)


def test_vol_ordering_at_three_stderr(result: M6HeadlineResult) -> None:
    """Short vol, short the knock-in put: the 25% model has the lower price and the higher P(KI)
    for both notes (3 root-sum-square stderr of the two independent estimates)."""
    for product in PRODUCTS:
        r = _row(result, "bs25", product)
        assert r["price_minus_ref"] < -3.0 * r["price_minus_ref_stderr"], (product, r["price"])
        assert r["p_ki_minus_ref"] > 3.0 * r["p_ki_minus_ref_stderr"], (product, r["p_ki"])


def test_expected_life_and_autocall_probabilities(result: M6HeadlineResult) -> None:
    """``E[T_τ]`` lies strictly inside ``(0, 3)`` and the autocall-date probabilities plus
    ``P(no autocall)`` sum to one exactly (they partition the paths)."""
    for (model, product), rep in result.reports.items():
        r = _row(result, model, product)
        assert 0.0 < r["expected_life"] < 3.0 and r["expected_life_stderr"] > 0
        total = sum(r[f"p_autocall_{i}"] for i in (1, 2, 3)) + r["p_no_autocall"]
        assert total == pytest.approx(1.0, abs=1e-12)
        assert rep.autocall_probabilities.sum() == pytest.approx(1.0, abs=1e-12)
        assert 0.0 < r["p_ki"] <= r["p_breach"] < 1.0
    # the two notes share dates, autocall barrier and path set: identical autocall index per
    # path, hence identical expected life and autocall-date probabilities, exactly
    for model in VOLS:
        ac, ph = _row(result, model, AUTOCALL_NAME), _row(result, model, PHOENIX_NAME)
        for c in ("expected_life", "p_no_autocall", "p_autocall_1", "p_autocall_2"):
            assert ac[c] == ph[c] and ac[f"{c}_stderr"] == ph[f"{c}_stderr"], (model, c)


def test_american_daily_knock_in_dominates_european(result: M6HeadlineResult) -> None:
    """Same dates, autocall barrier and knock-in level: the Phoenix's daily American knock-in
    knocks in at least as often as the autocall's European one (3 stderr, root-sum-square; the
    two notes share the path set, so the paired error is smaller still)."""
    for model in VOLS:
        ph, ac = _row(result, model, PHOENIX_NAME), _row(result, model, AUTOCALL_NAME)
        se = float(np.hypot(ph["p_ki_stderr"], ac["p_ki_stderr"]))
        assert ph["p_ki"] - ac["p_ki"] > 3.0 * se, (model, ph["p_ki"], ac["p_ki"], se)
        # the European note breaches only at T_N: P(breach) = P(KI) exactly
        assert ac["p_breach"] == ac["p_ki"] and ph["p_breach"] >= ph["p_ki"]


def test_first_autocall_probability_is_the_digital(
    result: M6HeadlineResult, forward_curve: ForwardCurve
) -> None:
    """Black–Scholes: ``P(first autocall at T_1) = N(d₂)`` at ``AC_1 S_0 = S_0`` — the undiscounted
    cash-or-nothing call of :func:`bs_digital_price` — within 3 stderr for both notes and both
    vols (the notes share the autocall event at the first date)."""
    df1 = float(forward_curve.rate_curve.df(1.0))
    for model, vol in VOLS.items():
        n_d2 = float(bs_digital_price(100.0, 100.0, 1.0, vol, 0.02, 0.01, 1)) / df1
        for product in PRODUCTS:
            r = _row(result, model, product)
            assert abs(r["p_autocall_1"] - n_d2) < 3.0 * r["p_autocall_1_stderr"], (
                model,
                product,
                r["p_autocall_1"],
                n_d2,
            )


def test_phoenix_row_reproduces_the_analytics_report(
    result: M6HeadlineResult, models: dict[str, BlackScholes]
) -> None:
    """The shared grid of the two notes is the Phoenix's own grid (its daily schedule contains
    the annual dates), so the Phoenix row equals ``autocall_report`` on its own paths exactly;
    the autocall row is the same note priced on that daily-record grid."""
    rep = autocall_report(result.reports[("bs20", PHOENIX_NAME)].product, models["bs20"], SIM)
    r = _row(result, "bs20", PHOENIX_NAME)
    assert r["price"] == rep.price and r["price_stderr"] == rep.price_stderr
    assert r["p_ki"] == rep.ki_probability and r["expected_life"] == rep.expected_life
    assert (
        r["leg:put"] == rep.legs["put"][0] and r["leg:put_european"] == rep.legs["put_european"][0]
    )
    assert r["leg:coupon"] == rep.legs["coupon"][0] and r["leg:put"] < r["leg:put_european"] < 0


def test_regression_keys(result: M6HeadlineResult) -> None:
    """The keys a baseline would pin cover exactly what :func:`baseline_values` reads off a
    result; the default set spans the five headline models."""
    prods = {p: result.reports[("bs20", p)].product for p in PRODUCTS}
    keys = regression_keys(prods, models=list(VOLS))
    values = baseline_values(result)
    assert set(keys) == set(values) and len(keys) == len(values)
    assert ("bs25", f"{AUTOCALL_NAME}:leg:put_digital") in values
    assert ("bs25", f"{PHOENIX_NAME}:leg:put_european") in values
    assert ("bs25", f"{PHOENIX_NAME}:leg:coupon") in values
    for (model, key), (v, s) in values.items():
        assert np.isfinite(v) and s > 0 and model in VOLS and ":" in key
    default = regression_keys()
    assert {m for m, _ in default} == set(HEADLINE_MODEL_NAMES) and default[0][0] == LV_NAME
    assert len(default) == len(HEADLINE_MODEL_NAMES) * len(keys) // len(VOLS)


def test_markdown_renders_the_rows(
    result: M6HeadlineResult, models: dict[str, BlackScholes]
) -> None:
    """Both blocks render every row with ``±`` errors and the reference differences; without a
    reference (an explicit product mapping, a small run) there are no difference columns."""
    md = result.to_markdown()
    for product in PRODUCTS:
        assert f"**{product}**" in md
    assert md.count("| bs20 |") == 2 and md.count("| bs25 |") == 2 and "±" in md
    assert "price - bs20 (% notional)" in md and "P(AC at 1y)" in md and "put_european leg" in md
    assert "not paired" in md
    plain = run_m6_headline(
        {"bs20": models["bs20"]},
        SimConfig(n_paths=2_000, dt_max=1.0 / 12.0, chunk_size=2_000, seed=1),
        {AUTOCALL_NAME: result.reports[("bs20", AUTOCALL_NAME)].product},
    )
    assert plain.reference is None and "price_minus_ref" not in plain.table.columns
    assert "| bs20 |" in plain.to_markdown() and "P(KI) - " not in plain.to_markdown()


def test_validation(models: dict[str, BlackScholes]) -> None:
    with pytest.raises(ValueError):
        run_m6_headline({}, SIM)
    with pytest.raises(ValueError):
        run_m6_headline(models, SIM, reference="lv")
    with pytest.raises(ValueError):
        run_m6_headline(models, SIM, {})


def test_spot_mismatch_raises_unless_allowed(
    models: dict[str, BlackScholes],
    discount: DiscountCurve,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Every level of a note is a fraction of ``spot_reference``: an explicit product struck away
    from a model's spot is refused (no silent change of meaning per model) unless the caller
    opts in with ``allow_spot_mismatch=True``, which prices it as given and logs a WARNING; a
    note struck on the model's own spot passes silently."""
    bs20 = {"bs20": models["bs20"]}
    off = {AUTOCALL_NAME: headline_products(discount, 105.0)[AUTOCALL_NAME]}
    with pytest.raises(ValueError, match="spot_reference 105"):
        run_m6_headline(bs20, SMALL, off)
    with caplog.at_level(logging.WARNING, logger="volsto.studies.m6"):
        res = run_m6_headline(bs20, SMALL, off, allow_spot_mismatch=True)
    assert len(res.table) == 1 and res.table.iloc[0]["seed"] == SMALL.seed
    assert res.reports[("bs20", AUTOCALL_NAME)].product.spot_reference == 105.0
    assert any("spot_reference 105" in rec.getMessage() for rec in caplog.records)
    caplog.clear()
    on = {AUTOCALL_NAME: headline_products(discount, 100.0)[AUTOCALL_NAME]}
    with caplog.at_level(logging.WARNING, logger="volsto.studies.m6"):
        run_m6_headline(bs20, SMALL, on)
    assert not caplog.records


def _load_script() -> ModuleType:
    """``scripts/m6_headline.py`` as a module (``scripts`` is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "m6_headline_script", ROOT / "scripts" / "m6_headline.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_script_main_writes_the_artefacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    forward_curve: ForwardCurve,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``scripts/m6_headline.py`` end to end with :func:`headline_models` monkeypatched to two
    Black–Scholes models (no cache, no calibration): the repo's 1F spec is loaded, the sim is
    overridden by ``--n-paths`` / ``--seed``, the table is priced against ``LV_NAME`` and
    ``headline.csv``, ``headline.md`` and ``headline.meta.json`` (n_paths, seed, particles, spec
    paths and keys, code version) land under ``--out``."""
    script = _load_script()
    seen: dict[str, Any] = {}

    def fake_headline_models(
        cache: LeverageCache,
        base: CalibrationSpec,
        spec_2f: CalibrationSpec | None,
        *,
        n_particles: int | None = None,
    ) -> tuple[dict[str, BlackScholes], None]:
        seen.update(cache=cache, base=base, spec_2f=spec_2f, n_particles=n_particles)
        return {
            LV_NAME: BlackScholes(0.20, forward_curve),
            "bs25": BlackScholes(0.25, forward_curve),
        }, None

    monkeypatch.setattr(script, "headline_models", fake_headline_models)
    out = tmp_path / "out"
    argv = ["--n-paths", "2000", "--seed", "11", "--no-2f", "--out", str(out)]
    argv += ["--cache", str(tmp_path / "cache")]
    assert script.main(argv) == 0
    assert isinstance(seen["base"], CalibrationSpec) and seen["spec_2f"] is None
    assert seen["n_particles"] == 800_000 and seen["cache"].root == tmp_path / "cache"
    table = pd.read_csv(out / "headline.csv")
    assert len(table) == 4 and set(table["model"]) == {LV_NAME, "bs25"}
    assert set(table["product"]) == set(PRODUCTS)
    assert {"price", "price_stderr", "price_minus_ref", "price_minus_ref_stderr"} <= set(
        table.columns
    )
    assert {"p_ki_minus_ref", "expected_life_minus_ref", *ROW_META_COLUMNS} <= set(table.columns)
    assert (table["n_paths"] == 2000).all() and (table["seed"] == 11).all()
    assert (table.loc[table["model"] == LV_NAME, "price_minus_ref"] == 0.0).all()
    assert (table["price_stderr"] > 0).all()
    md = (out / "headline.md").read_text()
    assert f"| {LV_NAME} |" in md and "| bs25 |" in md and "±" in md and "seed 11" in md
    assert md.strip() in capsys.readouterr().out
    meta = json.loads((out / "headline.meta.json").read_text())
    assert meta["n_paths"] == 2000 and meta["seed"] == 11 and meta["n_particles"] == 800_000
    assert meta["reference"] == LV_NAME and meta["models"] == [LV_NAME, "bs25"]
    assert meta["products"] == list(PRODUCTS)
    assert meta["spec_1f"].endswith("lsv_reference_1f.yaml") and meta["spec_2f"] is None
    assert len(meta["spec_key_1f"]) == 64 and meta["spec_key_2f"] is None
    assert isinstance(meta["code_version"], str) and meta["code_version"]
    assert meta["sim"]["n_paths"] == 2000 and meta["sim"]["seed"] == 11


def test_daily_phoenix_runtime_budget(result: M6HeadlineResult) -> None:
    """The daily Phoenix at 20k paths must price in under ~30 s (integrator's budget; measured
    about 2 s under Black–Scholes on the development machine)."""
    wall = result.table.set_index(["model", "product"])["wall_s"]
    assert wall[("bs20", PHOENIX_NAME)] < 30.0, dict(wall)
