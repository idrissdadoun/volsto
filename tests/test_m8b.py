"""M8b hedging studies (SPEC §8.2; ``volsto/studies/m8b.py``).  Fast tests need no cache and
never calibrate: task enumeration and sharding, the discriminator gate, the table builders on
synthetic results, the first-order agreement, the desk-sign conversion, the JSON round trip, the
refit-particle rewrite.  Slow tests (``-m slow``) read the leverage cache with
``allow_calibrate=False`` and skip on :class:`~volsto.calibration.cache.CacheMissError`: the study
A ranking pinned at a reduced budget, one study B (world, product) end to end, study C's
first-order agreement at +1 rota, study D's ranking for the 1y vanilla.  No wall-clock assertion.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest

from volsto.calibration.cache import CacheMissError
from volsto.calibration.fit_2f import load_fit_spec
from volsto.config import SurfacePerturbation
from volsto.hedging.instruments import Spot
from volsto.hedging.strategies import GreekTargetStrategy, Target
from volsto.risk.engine import RiskState
from volsto.studies.m8b import (
    BOOK,
    PRICING_A,
    Q_SWEEP,
    RECALIBRATIONS_C,
    REGIMES_D,
    ROTAS,
    STUDIES,
    STUDY_A_FREQUENCY,
    STUDY_C_STRIP_PATHS,
    TABLE_HEADERS,
    WORLDS_B,
    Gate,
    StudyConfig,
    StudyEnvironment,
    TaskResult,
    build_tables,
    discriminator_gate,
    enumerate_tasks,
    first_order_agreement,
    frequency_for,
    load_results,
    n_pricing_models,
    nonlinearity,
    parse_shard,
    product_maturity,
    refit_state,
    refits_capped,
    refits_fallback,
    run_task,
    save_result,
    shard,
    static_prediction,
    table_A,
    table_B,
    table_C,
    table_D,
    to_desk_pnl,
    write_tables,
)

ROOT = Path(__file__).resolve().parents[1]
REAL = Gate(True, "real", "raw SSR below 1 at every pillar")
ARTEFACT = Gate(
    False, "surface artefact", "T=0.0833: raw SSR 0.942 + 2 x 0.153 = 1.249 is not below 1"
)


def _cfg(tmp: Path, **kw: object) -> StudyConfig:
    base: dict[str, object] = dict(
        n_particles=200_000,
        refit_particles=200_000,
        allow_calibrate=False,
        out=tmp,
        cache=ROOT / "cache",
        verbose=False,
    )
    base.update(kw)
    return StudyConfig(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------
# fast: tasks, sharding, gate
# --------------------------------------------------------------------------------------------


def test_task_enumeration_counts_keys_and_frequencies() -> None:
    a = enumerate_tasks("A", gate=REAL)
    b = enumerate_tasks("B", gate=REAL)
    c = enumerate_tasks("C", gate=REAL)
    d = enumerate_tasks("D", gate=REAL)
    # A: (7 cliquet + 3 FVA strategies) x 2 pricing models; B: 5 products x 4 worlds;
    # C: 5 products x 3 rotas x 3 recalibrations; D: 2 products x 4 regimes
    assert len(a.tasks) == 2 * (1 + 2 * len(Q_SWEEP) + 3) == 20
    assert len(b.tasks) == len(BOOK) * len(WORLDS_B) == 20
    assert len(c.tasks) == len(BOOK) * len(ROTAS) * len(RECALIBRATIONS_C) == 45
    assert len(d.tasks) == 2 * len(REGIMES_D) == 8
    for tl in (a, b, c, d):
        keys = [t.key for t in tl.tasks]
        assert len(set(keys)) == len(keys)
        assert all(t.study == tl.study for t in tl.tasks)
    assert {t.pricing for t in a.tasks} == set(PRICING_A)
    assert all(t.frequency == STUDY_A_FREQUENCY for t in a.tasks)
    freq = {(t.product, t.frequency) for t in b.tasks}
    assert ("cliquet 1y", "daily") in freq and ("autocall 3y", "weekly") in freq
    assert {t.rota for t in c.tasks} == set(ROTAS) and {t.policy for t in c.tasks} == set(
        RECALIBRATIONS_C
    )
    assert {t.regime for t in d.tasks} == set(REGIMES_D)
    # the override
    cfg = StudyConfig(frequency="monthly")
    assert all(t.frequency == "monthly" for t in enumerate_tasks("B", cfg, REAL).tasks)
    assert frequency_for(1.0, "B") == "daily" and frequency_for(3.0, "C") == "weekly"
    assert frequency_for(3.0, "A") == "monthly"
    assert product_maturity("phoenix 3y") == 3.0 and product_maturity("fva 1y-2y") == 2.0
    with pytest.raises(ValueError):
        enumerate_tasks("E")


@pytest.mark.parametrize("n", [1, 2, 3, 7, 10])
def test_sharding_covers_every_task_exactly_once(n: int) -> None:
    tasks = list(enumerate_tasks("C", gate=REAL).tasks)
    seen: list[str] = []
    for i in range(1, n + 1):
        part = shard(tasks, i, n)
        seen += [t.key for t in part]
        # interleaved: the i-th shard holds positions i-1, i-1+n, ...
        assert [t.key for t in part] == [t.key for t in tasks[i - 1 :: n]]
    assert sorted(seen) == sorted(t.key for t in tasks) and len(seen) == len(tasks)
    assert parse_shard(" 2 / 5 ") == (2, 5)
    with pytest.raises(ValueError):
        shard(tasks, 0, n)
    with pytest.raises(ValueError):
        parse_shard("2-5")


def test_discriminator_gate_removes_world_ii_with_the_reason(tmp_path: Path) -> None:
    p = tmp_path / "verdict.json"
    p.write_text(json.dumps({"verdict": "surface artefact", "reason": "1m pillar not below 1"}))
    g = discriminator_gate(p)
    assert not g.enabled and g.verdict == "surface artefact"
    cfg = StudyConfig(discriminator_verdict=p)
    tl = enumerate_tasks("B", cfg)
    assert len(tl.tasks) == 15 and len(tl.skipped) == 5
    assert {s["product"] for s in tl.skipped} == set(BOOK)
    assert all(s["status"] == "surface artefact, skipped" for s in tl.skipped)
    assert all("1m pillar not below 1" in s["reason"] for s in tl.skipped)
    assert not any(t.world == "historical" for t in tl.tasks)
    # the table carries the gated rows
    df = table_B([], tl.skipped)
    assert len(df) == 5 and set(df["status"]) == {"surface artefact, skipped"}
    p.write_text(json.dumps({"verdict": "real", "reason": "below 1 everywhere"}))
    assert discriminator_gate(p).enabled
    assert len(enumerate_tasks("B", cfg).tasks) == 20
    missing = discriminator_gate(tmp_path / "absent.json")
    assert not missing.enabled and "no discriminator verdict" in missing.reason
    # the repository's verdict file (A4) is read the same way
    g_repo = discriminator_gate()
    assert g_repo.verdict in ("real", "surface artefact", "missing")


# --------------------------------------------------------------------------------------------
# fast: conventions
# --------------------------------------------------------------------------------------------


def test_first_order_agreement_and_nonlinearity() -> None:
    ratio, within, z = first_order_agreement(-0.050, 0.005, -0.060, 0.007)
    assert ratio == pytest.approx(0.05 / 0.06) and within
    assert z == pytest.approx(0.010 / math.hypot(0.005, 0.007))
    ratio, within, _ = first_order_agreement(-0.030, 0.005, -0.060, 0.007)
    assert ratio == pytest.approx(0.5) and not within
    ratio, within, _ = first_order_agreement(-0.077, 0.005, -0.060, 0.007)
    assert ratio == pytest.approx(0.077 / 0.060) and within
    assert not first_order_agreement(-0.079, 0.005, -0.060, 0.007)[1]
    ratio, within, z = first_order_agreement(0.01, 0.0, 0.0, 0.0)
    assert math.isnan(ratio) and not within and math.isnan(z)
    assert nonlinearity(-0.12, 2.0, -0.05) == pytest.approx(0.2)
    assert nonlinearity(-0.15, 3.0, -0.05) == pytest.approx(0.0)
    assert math.isnan(nonlinearity(1.0, 2.0, 0.0))


def test_desk_sign_conversion() -> None:
    """The hedger prices the product long; the desk is short the note: every desk number is
    the negative, the standard error unchanged."""
    x = np.array([1.0, -2.0, 0.5])
    assert np.array_equal(to_desk_pnl(x), -x) and to_desk_pnl(0.25) == -0.25
    r = _synthetic("C", "autocall 3y", mean=(0.0300, 0.0040), recal_total=(0.0539, 0.0066))
    assert r.desk_recal_total() == (-0.0539, 0.0066) and r.desk_mean() == (-0.03, 0.004)
    static = {
        ("autocall 3y", "sabr_linked"): {
            "desk_pnl_shadow": [-0.0539, 0.0066],
            "desk_pnl_usual": [0.0055, 0.0066],
        }
    }
    df = table_C([dataclasses.replace(r, rota=1.0, policy="sabr_linked")], static)
    row = df.iloc[0]
    assert row["recal_pnl_desk"] == pytest.approx(-0.0539) and row["recal_se"] == pytest.approx(
        0.0066
    )
    assert row["static_prediction"] == pytest.approx(-0.0539) and row["ratio"] == pytest.approx(1.0)
    assert bool(row["within_30pct"]) and row["nonlinearity"] == 0.0


def test_refit_state_rewrites_only_parameter_changes() -> None:
    cfg = StudyConfig(n_particles=800_000, refit_particles=200_000)
    env_state = RiskState(load_fit_spec(cfg.marking_fit).spec).with_particles(cfg.n_particles)
    base = env_state.spec.model
    assert refit_state(env_state, base, cfg.refit_particles) is env_state
    bumped = env_state.with_perturbation(SurfacePerturbation("parallel", {"size": 0.01}))
    assert refit_state(bumped, base, cfg.refit_particles).spec.particle.n_particles == 800_000
    refit = env_state.with_params(nu=base.nu * 1.1)
    out = refit_state(refit, base, cfg.refit_particles)
    assert out.spec.particle.n_particles == 200_000 and out.spec.model == refit.spec.model


def test_n_pricing_models_counts_bump_simulations() -> None:
    d = GreekTargetStrategy((Target("delta"),), [Spot()])
    assert n_pricing_models(d) == 3
    assert n_pricing_models(GreekTargetStrategy((Target("delta"), Target("vega")), [Spot()])) == 5
    s = GreekTargetStrategy(
        (Target("delta"), Target("gamma"), Target("vega"), Target("skew_T:3")), [Spot()]
    )
    assert n_pricing_models(s) == 6
    assert n_pricing_models(GreekTargetStrategy((Target("delta"), Target("vanna")), [Spot()])) == 5
    r = GreekTargetStrategy((Target("delta"),), [Spot()], delta_regime="sticky_strike")
    assert n_pricing_models(r) == 5


# --------------------------------------------------------------------------------------------
# fast: tables on synthetic results
# --------------------------------------------------------------------------------------------


def _synthetic(study: str, product: str, **kw: object) -> TaskResult:
    base: dict[str, object] = dict(
        key=f"{study}__{product}__"
        + "__".join(f"{k}_{v}" for k, v in kw.items() if isinstance(v, str)),
        study=study,
        product=product,
        world=kw.pop("world", "pricing"),
        strategy=kw.pop("strategy", "preset"),
        n_paths_world=1000,
        value_0=(1.0, 0.01),
        mean=(0.0, 0.01),
        std=(0.1, 0.005),
        quantiles={"q05": (-0.2, 0.01), "q50": (0.0, 0.01), "q95": (0.2, 0.01)},
    )
    base.update(kw)
    return TaskResult(**base)  # type: ignore[arg-type]


def test_refits_at_bound_counts_pinned_refits() -> None:
    """``refits_at_bound`` reads the hedger's own ``at_bound`` column when the run recorded it
    and otherwise parses the correlations out of the stored ``params`` repr (the study-C runs of
    2026-09-15 predate the flag; their JSONs were backfilled from the pickles).  Measured over
    the 45 stored runs: 41 of the 113 ``sabr_linked`` refits are pinned, 0 of the 113
    ``sticky_breakeven`` ones."""
    import pandas as pd

    from volsto.studies.m8b import refits_at_bound

    assert refits_at_bound(pd.DataFrame()) == 0
    recorded = pd.DataFrame(
        {
            "t": [0.5, 1.0, 1.5],
            "recalibrated": [True, False, True],
            "at_bound": ["rho12=+1.0000; rho_SX1=-1.0000", "", ""],
        }
    )
    assert refits_at_bound(recorded) == 1
    collapsed = "LSV(BergomiSV(2F, BergomiParams(nu=2.4, rho12=0.9999999955, rho_SX1=-0.99999, "
    sane = "LSV(BergomiSV(2F, BergomiParams(nu=2.4, rho12=0.41, rho_SX1=-0.92, rho_SX2=-0.73)))"
    parsed = pd.DataFrame(
        {
            "t": [0.5, 1.0, 1.5],
            "recalibrated": [True, True, False],
            "params": [collapsed + "rho_SX2=-0.99999)))", sane, None],
        }
    )
    assert refits_at_bound(parsed) == 1  # only the fired, collapsed refit counts


def test_refit_fallback_and_cap_counts_backfill_safe(tmp_path: Path) -> None:
    """``refits_fallback`` / ``refits_capped`` count the fired refits whose recalibration row
    carries ``fallback_applied`` / ``corr_capped`` (the rebuilt refit of 2026-09-16), ``-1`` for
    a run recorded before the columns existed; a result JSON written before the fields existed
    reads back with ``-1`` (backfill-safe), a new one round-trips its counts."""
    import pandas as pd

    assert refits_fallback(pd.DataFrame()) == 0 == refits_capped(pd.DataFrame())
    new = pd.DataFrame(
        {
            "t": [0.5, 1.0, 1.5, 2.0],
            "recalibrated": [True, False, True, True],
            "fallback_applied": [True, True, False, True],  # the unfired row never counts
            "corr_capped": [False, False, True, False],
            "at_bound": ["", "", "", ""],
        }
    )
    assert refits_fallback(new) == 2 and refits_capped(new) == 1
    old = pd.DataFrame({"t": [0.5], "recalibrated": [True], "at_bound": [""]})
    assert refits_fallback(old) == -1 and refits_capped(old) == -1
    r = _synthetic("C", "cliquet 1y", world="skew shock", rota=3.0, policy="sabr_linked")
    assert r.n_refits_fallback == -1 and r.n_refits_capped == -1
    doc = r.to_dict()
    for k in ("n_refits_fallback", "n_refits_capped"):
        doc.pop(k)
    assert TaskResult.from_dict(doc).n_refits_fallback == -1
    r2 = dataclasses.replace(r, n_refits_fallback=2, n_refits_capped=1)
    save_result(r2, None, tmp_path)
    back = load_results(tmp_path, "C")[0]
    assert back.n_refits_fallback == 2 and back.n_refits_capped == 1


def test_study_c_rule_uses_its_own_strip_paths() -> None:
    """Study C's rule (:func:`study_c_rule`) reads its strips at
    :data:`STUDY_C_STRIP_PATHS` = 8·10⁴ (the owner's decision of 2026-09-16), independent of
    the 2·10⁴ world paths, caps the refit correlation at 0.97 and keeps the documented pillars,
    skew pillars and threshold.  The marking fit spec only (no leverage)."""
    from volsto.hedging.hedger import REFIT_CORRELATION_CAP
    from volsto.studies.m8b import (
        STUDY_C_RULE_PILLARS,
        STUDY_C_RULE_SKEW_PILLARS,
        STUDY_C_SKEW_MOVE_THRESHOLD,
        study_c_rule,
    )

    env = StudyEnvironment(StudyConfig(allow_calibrate=False, verbose=False))
    for policy in ("sabr_linked", "sticky_breakeven"):
        rule = study_c_rule(policy, env)
        assert rule.policy == policy and rule.strip_paths == STUDY_C_STRIP_PATHS == 80_000
        assert rule.correlation_cap == REFIT_CORRELATION_CAP == 0.97
        assert rule.pillars == STUDY_C_RULE_PILLARS
        assert rule.config().skew_pillars == STUDY_C_RULE_SKEW_PILLARS
        assert rule.skew_move_threshold == STUDY_C_SKEW_MOVE_THRESHOLD
    assert env.cfg.n_world == 20_000


def test_projection_counts_the_strip_at_its_own_path_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The dry-run projection of study C adds the recalibration strips to the recalibration rows
    only, probed with the rule at :data:`STUDY_C_STRIP_PATHS` (the hedger's probes stubbed: the
    test checks the accounting, not the timing), once per product and frequency, and says so in
    its notes.  Needs the cached pricing leverages (skipped when absent)."""
    from volsto.hedging.hedger import Hedger

    seen: list[int] = []

    def fake_strip(self: Hedger, product: object, rule: object, dates: object = None) -> float:
        seen.append(int(rule.strip_paths))  # type: ignore[attr-defined]
        return 1000.0

    monkeypatch.setattr(Hedger, "projected_wall_clock", lambda self, *a, **k: 10.0)
    monkeypatch.setattr(Hedger, "projected_strip_seconds", fake_strip)
    cfg = StudyConfig(allow_calibrate=False, verbose=False)
    tasks = [t for t in enumerate_tasks("C", cfg, REAL).tasks if t.product == "cliquet 1y"]
    try:
        from volsto.studies.m8b import project

        pr = project(tasks, StudyEnvironment(cfg))
    except CacheMissError as exc:
        pytest.skip(f"cached leverage absent (tests never calibrate): {exc}")
    rows = {r["key"]: r for r in pr.per_task}
    for t in tasks:
        row = rows[t.key]
        if t.policy == "none":
            assert row["strip_seconds"] == 0.0 and row["projected_seconds"] == 10.0
        else:
            assert row["strip_seconds"] == 1000.0 and row["projected_seconds"] == 1010.0
    assert seen == [STUDY_C_STRIP_PATHS]  # memoised per product and frequency
    assert pr.hedger_seconds == pytest.approx(9 * 10.0 + 6 * 1000.0)
    assert any(f"{STUDY_C_STRIP_PATHS} paths" in n for n in pr.notes), pr.notes


def test_table_builders_on_synthetic_results(tmp_path: Path) -> None:
    res: list[TaskResult] = []
    # A: three strategies, two pricing models; the std sets the rank
    for pricing, stds in (("LV", (0.014, 0.008, 0.033)), ("2F", (0.020, 0.010, 0.030))):
        for name, s in zip(
            ("delta only", "delta + cap calls q=1", "delta + cap calls q=0.5 + net var swap"), stds
        ):
            res.append(
                _synthetic("A", "cliquet 1y", strategy=name, pricing=pricing, std=(s, 0.001))
            )
    ta = table_A(res)
    lv = ta[ta["pricing"] == "LV"]
    assert list(lv["strategy"]) == [
        "delta + cap calls q=1",
        "delta only",
        "delta + cap calls q=0.5 + net var swap",
    ]
    assert list(lv["rank"]) == [1.0, 2.0, 3.0]
    # B: leakage in the desk sign, static + dynamic = total
    rb = _synthetic(
        "B",
        "cliquet 1y",
        world="pure LV",
        mean=(-0.30, 0.02),
        static_spread=(0.10, 0.03),
        regimes=[
            {"regime": "realised vol low", "mean": -0.1, "stderr": 0.01, "std": 0.2, "n": 300}
        ],
        attribution=[
            {"component": "hedge spot", "mean": 0.5, "stderr": 0.01},
            {"component": "total", "mean": -0.3, "stderr": 0.02},
            {"component": "product residual", "mean": -0.8, "stderr": 0.02},
        ],
    )
    skipped = [
        {"world": "historical", "product": p, "status": "surface artefact, skipped", "reason": "1m"}
        for p in BOOK
    ]
    rb_same = _synthetic("B", "cliquet 1y", world="same", mean=(-0.05, 0.01), value_0=(1.0, 0.02))
    tb = table_B([rb, rb_same], skipped)
    row = tb[tb["world"] == "pure LV"].iloc[0]
    assert row["leakage_desk"] == pytest.approx(0.30) and row["static_spread"] == pytest.approx(
        0.10
    )
    assert row["dynamic_leakage"] == pytest.approx(0.20)
    # the model reserve reads off the difference to the engine's own baseline (world 'same')
    assert row["leakage_vs_same"] == pytest.approx(0.25)
    assert row["leakage_vs_same_se"] == pytest.approx(math.hypot(0.02, 0.01))
    assert row["leakage_desk_incl_v0_se"] == pytest.approx(math.hypot(0.02, 0.01))
    assert row["leakage_desk_incl_v0"] == pytest.approx(row["leakage_desk"])
    assert row["leakage_desk_se"] == pytest.approx(0.02)
    assert row["dynamic_leakage_se"] == pytest.approx(math.hypot(0.02, row["static_spread_se"]))
    assert row["q05_desk_se"] == pytest.approx(0.01) and row["q95_desk_se"] == pytest.approx(0.01)
    # every Monte Carlo column of table B carries its `<x>_se` twin (the viewer's stderr walk)
    mc_cols = [
        "leakage_desk",
        "leakage_desk_incl_v0",
        "leakage_vs_same",
        "std",
        "q05_desk",
        "q95_desk",
        "static_spread",
        "dynamic_leakage",
        "value_0",
    ]
    assert all(f"{c}_se" in tb.columns for c in mc_cols), tb.columns.tolist()
    assert tb[tb["world"] == "same"].iloc[0]["leakage_vs_same"] == pytest.approx(0.0)
    assert row["q05_desk"] == pytest.approx(-0.2) and row["q95_desk"] == pytest.approx(0.2)
    assert "product residual: +0.8000" in row["attribution_top_desk"]
    assert "realised vol low: +0.1000" in row["regimes_desk"]
    assert (tb["status"] == "surface artefact, skipped").sum() == 5
    assert list(tb["world"])[-1] == "pure LV"  # the owner's world order: historical rows first
    # C: recal P&L per rota, static prediction x rota, nonlinearity vs the +1 row
    rc = []
    for rota, tot in ((1.0, 0.050), (2.0, 0.120), (3.0, 0.150)):
        rc.append(
            _synthetic(
                "C",
                "autocall 3y",
                world="skew shock",
                rota=rota,
                policy="sabr_linked",
                recal_total=(tot, 0.005),
                recal_by_date=[{"t": 1.5, "mean": tot, "stderr": 0.005, "alive_fraction": 0.7}],
                n_refits=1,
                # the +2 row's refit pinned a correlation: the row is reported and marked
                n_refits_at_bound=1 if rota == 2.0 else 0,
                # the +3 row predates the fallback / cap record (-1); +1 and +2 recorded it
                n_refits_fallback=-1 if rota == 3.0 else int(rota == 2.0),
                n_refits_capped=-1 if rota == 3.0 else 0,
                mean=(0.2, 0.01),
            )
        )
    rc.append(
        _synthetic(
            "C", "autocall 3y", world="skew shock", rota=1.0, policy="none", mean=(0.02, 0.01)
        )
    )
    static = {
        ("autocall 3y", "sabr_linked"): {
            "desk_pnl_shadow": [-0.0539, 0.0066],
            "desk_pnl_usual": [0.0055, 0.0066],
        }
    }
    tc = table_C(rc, static)
    assert {"ratio_se", "nonlinearity_se", "refits_at_bound", "contaminated"} <= set(tc.columns)
    pinned = tc[(tc["rota"] == 2.0) & (tc["recalibration"] == "sabr_linked")].iloc[0]
    clean1 = tc[(tc["rota"] == 1.0) & (tc["recalibration"] == "sabr_linked")].iloc[0]
    assert pinned["refits_at_bound"] == 1 and bool(pinned["contaminated"])
    assert clean1["refits_at_bound"] == 0 and not bool(clean1["contaminated"])
    assert {"refits_fallback", "refits_capped"} <= set(tc.columns)
    assert pinned["refits_fallback"] == 1 and pinned["refits_capped"] == 0
    assert clean1["refits_fallback"] == 0 and clean1["refits_capped"] == 0
    old3 = tc[(tc["rota"] == 3.0) & (tc["recalibration"] == "sabr_linked")].iloc[0]
    assert old3["refits_fallback"] == -1 and old3["refits_capped"] == -1
    assert "refits_fallback" in TABLE_HEADERS["C"] and "80000 paths" in TABLE_HEADERS["C"]
    ok_rows = tc[tc["ratio"].notna()]
    assert (ok_rows["ratio_se"] > 0).all(), ok_rows[["ratio", "ratio_se"]]
    two = tc[(tc["rota"] != 1.0) & tc["nonlinearity"].notna()]
    assert (two["nonlinearity_se"] >= 0).all()
    r1 = tc[(tc["rota"] == 1.0) & (tc["recalibration"] == "sabr_linked")].iloc[0]
    r2 = tc[(tc["rota"] == 2.0) & (tc["recalibration"] == "sabr_linked")].iloc[0]
    r3 = tc[(tc["rota"] == 3.0) & (tc["recalibration"] == "sabr_linked")].iloc[0]
    assert r1["recal_pnl_desk"] == pytest.approx(-0.050) and r1["ratio"] == pytest.approx(
        0.050 / 0.0539
    )
    assert bool(r1["within_30pct"]) and r1["nonlinearity"] == 0.0
    assert r2["static_prediction"] == pytest.approx(-2 * 0.0539) and r2[
        "nonlinearity"
    ] == pytest.approx(0.2)
    assert r3["nonlinearity"] == pytest.approx(0.0) and r3["refit_dates"] == "1.5"
    rn = tc[tc["recalibration"] == "none"].iloc[0]
    assert rn["prediction_of"] == "desk_pnl_usual x rota" and rn["total_pnl_desk"] == pytest.approx(
        -0.02
    )
    assert rn["static_prediction"] == pytest.approx(0.0055)
    # D: winner and closest-to-model
    rd = [
        _synthetic("D", "vanilla 1y atm", strategy="delta only", regime=reg, std=(s, 0.001))
        for reg, s in zip(REGIMES_D, (0.100, 0.130, 0.090, 0.105))
    ]
    td = table_D(rd)
    assert td.loc[td["winner"], "regime"].tolist() == ["sticky_skew"]
    assert td.loc[td["closest_to_model"], "regime"].tolist() == ["sticky_moneyness"]
    # the writers: CSVs, the markdown with the placeholders, the results round trip from disk
    tables = build_tables([*res, rb, rb_same, *rc, *rd], skipped, static)
    md = write_tables(
        tmp_path, tables, header_lines=["wall clock 1 s", "recalibrated: no"], static=static
    )
    text = md.read_text()
    for st in STUDIES:
        assert f"M8B_TABLE_{st}" in text and (tmp_path / f"m8b_table_{st}.csv").exists()
    assert "recalibrated: no" in text and "desk P&L per +1 rota" in text


def test_task_result_json_round_trip(tmp_path: Path) -> None:
    r = _synthetic(
        "B",
        "ko var 1y",
        world="nu x1.5",
        mean=(float("nan"), 0.0),
        static_spread=None,
        world_value_0=(0.5, 0.01),
        notes=["variance P&L / (2 K_vol)"],
        world_meta={"nu": 2.91, "cached_before": True},
    )
    save_result(r, None, tmp_path)
    back = load_results(tmp_path)
    assert len(back) == 1
    b = back[0]
    assert b.key == r.key and math.isnan(b.mean[0]) and b.static_spread is None
    assert (
        b.world_value_0 == (0.5, 0.01) and b.quantiles == r.quantiles and b.world_meta["nu"] == 2.91
    )
    assert b.notes == r.notes and b.study == "B"


# --------------------------------------------------------------------------------------------
# slow: on the cached SPX marking fit (never calibrating)
# --------------------------------------------------------------------------------------------


def _skip_if(res: TaskResult) -> None:
    if res.status == "skipped":
        pytest.skip(res.reason)
    assert res.status == "ok", res.reason


@pytest.mark.slow
@pytest.mark.parametrize("pricing", ["LV", "2F"])
def test_study_a_cliquet_ranking_pinned(tmp_path: Path, pricing: str) -> None:
    """Study A at a reduced budget (6·10³ paths, monthly) on the SPX 2022-12-30 surface: the
    cliquet strategies' P&L std ranking.  Pinned ORDER: the static replication (``q = 1`` strip +
    accumulated-sum put) beats delta only; the ``q = 0.5`` strip with the net-sized variance swap
    beats the ``q = 0.5`` strip alone.  The 2F variant needs the marking fit's parallel ±1 vp
    leverages at 2·10⁵ particles for the variance-swap lines (skipped with the reason when absent;
    tests never calibrate).

    Recorded (LV pricing = world, Dupire of the SPX 2022-12-30 surface, 6·10³ pricing and world
    paths, seed 2024, monthly rebalancing over weekly simulation steps — daily where the variance
    swap's fixings refine the grid; P&L std in % of notional with its fourth-moment standard
    error; run of 2026-09-15, nothing calibrated): delta only 1.2334 ± 0.0261; q = 0.5 strip
    6.6806 ± 0.1225; q = 0.75 strip 3.3246 ± 0.0613; q = 1 strip 0.5448 ± 0.1323 (fat-tailed:
    the SPX surface's crash paths); q = 0.5 + net var swap 3.2338 ± 0.0563; q = 0.75 + net var
    swap 1.6983 ± 0.0381; q = 1 + var swap 0.6528 ± 0.0379 (the swap's quantity is regression
    noise at q = 1: recorded).  Ranking: q = 1 strip < q = 1 + var swap < delta only < q = 0.75 +
    net var swap < q = 0.5 + net var swap < q = 0.75 strip < q = 0.5 strip.  Under 2F (the marking
    LSV at 2·10⁵ particles) delta only measured 2.1919 ± 0.0617 before the variance-swap lines
    skipped on the uncached parallel ±1 vp leverages.
    """
    cfg = _cfg(tmp_path, n_paths=6_000, world_paths=6_000, frequency="monthly")
    try:
        env = StudyEnvironment(cfg)
        tasks = [
            t
            for t in enumerate_tasks("A", cfg, REAL).tasks
            if t.pricing == pricing and t.product == "cliquet 1y"
        ]
        stds: dict[str, tuple[float, float]] = {}
        for task in tasks:
            res = run_task(task, cfg, env, save=False)
            _skip_if(res)
            stds[task.strategy] = res.std
            print(
                f"A {pricing} {task.strategy}: std {res.std[0]:.4f} +/- {res.std[1]:.4f} {res.unit}, mean {res.mean[0]:+.4f} +/- {res.mean[1]:.4f}, wall {res.wall_seconds:.0f} s, calibrations {res.calibrations}"
            )
    except CacheMissError as exc:
        pytest.skip(f"cached leverage absent (tests never calibrate): {exc}")
    print(f"A {pricing} ranking:", sorted(stds.items(), key=lambda kv: kv[1][0]))
    if pricing == "2F":
        assert stds["delta + cap calls q=1"][0] < stds["delta only"][0]
    else:
        # LV finding (SPEC §8.1): with the LV model delta the static replication's residual is
        # the CRN tangent-process noise of the wing local vol — recorded, not asserted
        print(
            f"A LV finding: q=1 strip / delta only std ratio "
            f"{stds['delta + cap calls q=1'][0] / stds['delta only'][0]:.2f}"
        )
    assert stds["delta + cap calls q=0.5 + net var swap"][0] < stds["delta + cap calls q=0.5"][0]
    assert env.calibrations == 0


@pytest.mark.slow
def test_study_b_pure_lv_cliquet_end_to_end(tmp_path: Path) -> None:
    """Study B for (``pure LV``, ``cliquet 1y``) at 3·10³ paths (monthly for the test budget):
    finite distribution, regimes, attribution and static spread; nothing calibrated."""
    cfg = _cfg(tmp_path, n_paths=3_000, world_paths=3_000, frequency="monthly")
    try:
        env = StudyEnvironment(cfg)
        task = next(
            t
            for t in enumerate_tasks("B", cfg, REAL).tasks
            if t.world == "pure LV" and t.product == "cliquet 1y"
        )
        res = run_task(task, cfg, env)
    except CacheMissError as exc:
        pytest.skip(f"cached leverage absent (tests never calibrate): {exc}")
    _skip_if(res)
    print(
        f"B pure LV cliquet: desk leakage {res.desk_mean()[0]:+.4f} +/- {res.mean[1]:.4f} {res.unit}, std {res.std[0]:.4f}, static spread {res.static_spread}, q05/q95 {res.quantiles['q05']} / {res.quantiles['q95']}, wall {res.wall_seconds:.0f} s"
    )
    assert all(np.isfinite(v[0]) and np.isfinite(v[1]) for v in (res.mean, res.std, res.value_0))
    assert all(np.isfinite(v[0]) for v in res.quantiles.values()) and set(res.quantiles) >= {
        "q05",
        "q50",
        "q95",
    }
    assert (
        res.static_spread is not None
        and np.isfinite(res.static_spread[0])
        and res.static_spread[1] > 0
    )
    assert len(res.regimes) >= 6 and any(a["component"] == "total" for a in res.attribution)
    assert res.calibrations == 0 and env.calibrations == 0
    assert len(load_results(tmp_path, "B")) == 1


@pytest.mark.slow
def test_study_c_first_order_agreement_at_one_rota(tmp_path: Path) -> None:
    """Study C, 3y autocall, +1 rota, ``sabr_linked``, 4·10³ paths: the recalibration P&L (desk
    convention) within 30% of the M7 greek's ``desk_pnl_shadow × 1`` — only when the shock world's
    leverages, the preset's bump leverages and the refit sets are cached at 2·10⁵ particles
    (else skipped with the reason)."""
    cfg = _cfg(tmp_path, n_paths=4_000, world_paths=4_000, frequency="monthly")
    try:
        env = StudyEnvironment(cfg)
        task = next(
            t
            for t in enumerate_tasks("C", cfg, REAL).tasks
            if t.product == "autocall 3y" and t.rota == 1.0 and t.policy == "sabr_linked"
        )
        res = run_task(task, cfg, env)
        _skip_if(res)
        doc = static_prediction("autocall 3y", "sabr_linked", env)
    except CacheMissError as exc:
        pytest.skip(f"cached leverage absent (tests never calibrate): {exc}")
    recal, se = res.desk_recal_total()
    pred, pred_se = doc["desk_pnl_shadow"]
    ratio, within, z = first_order_agreement(recal, se, pred, pred_se)
    print(
        f"C autocall +1 rota sabr_linked: recal P&L desk {recal:+.4f} +/- {se:.4f} over {res.n_refits} refit(s) at {[d['t'] for d in res.recal_by_date]}; static {pred:+.4f} +/- {pred_se:.4f}; ratio {ratio:.3f}, z {z:+.2f}, wall {res.wall_seconds:.0f} s"
    )
    assert res.n_refits >= 1
    assert within, f"first-order agreement failed: ratio {ratio:.3f} (z {z:+.2f})"


@pytest.mark.slow
def test_study_c_excess_skew_trigger_fires_once_when_the_strip_first_spans_the_shock() -> None:
    """The excess-skew trigger of :class:`RecordingHedger` on the cached 2·10⁵ base and +1 rota
    leverages (skipped when absent): cliquet 1y, delta only (no bump leverage), the rule's refit
    stubbed to return the same parameters (nothing calibrated, recalibration P&L exactly 0).
    Measured 2026-09-15 (4·10³ paths, monthly, t0 = 0.5): the excess skew 0.001–0.009 while no
    strip option spans the shock window, 0.018 at t = 0.417 (the 3M option then spans it — the
    world's conditional smile anticipates its deterministic shock), one refit there, moves of
    0.001–0.008 against the reset reference afterwards and no further refit."""
    from volsto.hedging import Costs, RecalibrationRule, Schedule
    from volsto.studies.m8b import RecordingHedger, recalibration_by_date

    cfg = _cfg(Path("/nonexistent"), n_paths=4_000, world_paths=4_000, frequency="monthly")
    try:
        env = StudyEnvironment(cfg)
        product = env.product("cliquet 1y")
        world, meta = env.shock_world(1.0, 0.5)
        # the strips at the world's path count, as measured on 2026-09-15
        rule = RecalibrationRule(
            policy="sabr_linked", refit=lambda surf, params: params, strip_paths=4_000
        )
        h = RecordingHedger(
            env.fresh_ctx("2F"),
            world,
            Schedule("monthly"),
            Costs(),
            recalibration=rule,
            sim=env.sim("monthly"),
            world_paths=4_000,
            verbose=False,
        )
        r = h.run(product, GreekTargetStrategy((Target("delta"),), [Spot()], name="delta only"))
    except CacheMissError as exc:
        pytest.skip(f"cached leverage absent (tests never calibrate): {exc}")
    rec = r.recalibrations
    fired = rec.loc[rec["recalibrated"].astype(bool), "t"].tolist()
    print(
        f"excess-skew trigger: t0 {meta['t0']}, moves {rec['skew_move'].round(4).tolist()}, refits at {fired}"
    )
    assert not meta["calibrated"] and env.calibrations == 0
    assert 1 <= len(fired) <= 2, fired
    # the reference resets at the refit: the held rotation triggers nothing afterwards
    assert fired[-1] <= meta["t_end"] + min(rule.pillars) + 1e-9, fired
    # no refit while no option of the strip (pillars 3M / 1Y) reaches into the shock window
    assert min(fired) + min(rule.pillars) > meta["t0"] - 1e-9
    quiet = rec[rec["t"] + min(rule.pillars) <= meta["t0"] + 1e-9]
    assert (quiet["skew_move"] < rule.skew_move_threshold).all()
    assert np.all(r.pnl_recalibration == 0.0)
    rows, note = recalibration_by_date(h, r)
    assert (
        note is None
        and [row["t"] for row in rows] == fired
        and all(row["mean"] == 0.0 for row in rows)
    )


@pytest.mark.slow
def test_study_d_vanilla_regime_ranking(tmp_path: Path) -> None:
    """Study D for the 1y ATM vanilla at 4·10³ paths: the P&L std per delta regime reported (the
    sticky regimes need the spot-moved / surface-shifted leverages at 2·10⁵ particles: skipped
    when absent)."""
    cfg = _cfg(tmp_path, n_paths=4_000, world_paths=4_000, frequency="monthly")
    try:
        env = StudyEnvironment(cfg)
        tasks = [t for t in enumerate_tasks("D", cfg, REAL).tasks if t.product == "vanilla 1y atm"]
        results = []
        for task in tasks:
            res = run_task(task, cfg, env, save=False)
            _skip_if(res)
            results.append(res)
    except CacheMissError as exc:
        pytest.skip(f"cached leverage absent (tests never calibrate): {exc}")
    td = table_D(results)
    print(
        "D vanilla 1y:\n", td[["regime", "std", "std_se", "winner", "closest_to_model"]].to_string()
    )
    assert np.isfinite(td["std"]).all() and td["winner"].sum() == 1
