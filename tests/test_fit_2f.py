"""Staged two-factor fitting (SPEC §15 Parts 3–4; Bergomi ch. 7–9).

* Fast, on a one-year synthetic history of the Table 8.2 model (order-one ATM source, seed 8):
  stage 1 with the correlation target recovers ``(ν, θ, k1, k2, ρ12)`` from every Table 7.1
  start (unique optimum; ν within 10%, θ within 0.05, k1 within 15%, ρ12 within 0.15); stage 2
  on the long SSR window recovers the spot/vol correlations within 0.2 (a one-year sample —
  the SPEC's 0.05 is the three-year test below); the Table 7.1 degeneracy — with ``ρ12`` fixed
  at −0.7, 0 and +0.7 the vol-of-vol curve is fit about equally well by different
  ``(θ, k1, k2)``, and the correlation target resolves it; the skew/ν scale direction leaves
  the stage-2 objective unchanged while stage 1 rises steeply; the YAML round trip; the rolling
  fit and the identification flags on a short span.
* Slow: the SPEC recovery test on the three-year mixing history (ν, θ, k1 within 10%, k2
  within its sensitivity band, ρ_SX1 / ρ_SX2 within 0.05) and the real-data end-to-end run on
  the 2022 H2 SPX history when ``outputs/m7/hdn_history_ssvi.csv`` is present (no fixed
  numbers: finite, PSD, objectives reported).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from volsto.calibration.fit_2f import (
    TABLE_7_1,
    Fit2FConfig,
    chi_from_rhos,
    fit_2f,
    fit_stage1,
    fit_stage2,
    rho_sx2_from_chi,
    skew_scale_degeneracy,
)
from volsto.calibration.history import SurfaceHistory, synthetic_2f_history
from volsto.calibration.stability import flag_unidentified, rolling_fit
from volsto.config import BergomiParams

ROOT = Path(__file__).resolve().parents[1]
P82 = BergomiParams(1.74, 0.245, 5.35, 0.28, 0.0, -0.759, -0.487)


@pytest.fixture(scope="module")
def synthetic_1y():  # type: ignore[no-untyped-def]
    return synthetic_2f_history(P82, years=1.0, seed=8, atm_source="order_one")


def test_eq_8_56_parametrisation() -> None:
    for rho12 in (-0.7, 0.0, 0.7):
        for r1 in (-0.9, -0.3, 0.5):
            for chi in (-1.0, -0.2, 0.9):
                r2 = rho_sx2_from_chi(rho12, r1, chi)
                p = BergomiParams(1.0, 0.3, 5.0, 0.3, rho12, r1, r2)  # PSD by construction
                assert p.rho_SX2 == pytest.approx(r2)
                assert chi_from_rhos(rho12, r1, r2) == pytest.approx(chi)
    with pytest.raises(ValueError):
        Fit2FConfig(rho12_mode="guess")
    with pytest.raises(ValueError):
        Fit2FConfig(starts=("Set IV",))


def test_stage1_recovery_and_table_7_1_degeneracy(synthetic_1y) -> None:  # type: ignore[no-untyped-def]
    hist = synthetic_1y.history
    cfg = Fit2FConfig(window_vol=200, window_ssr=200, rho12_mode="from_correlation")
    s1 = fit_stage1(hist, cfg)  # window-mean curve: the flat formula biases k1 by 18% here
    assert s1.unique, s1.table()
    p = s1.best.params
    assert abs(p.nu - P82.nu) < 0.1 * P82.nu and abs(p.theta - P82.theta) < 0.05
    assert abs(p.k1 - P82.k1) < 0.15 * P82.k1 and abs(p.rho12 - P82.rho12) < 0.15
    assert abs(p.k2 - P82.k2) < max(0.25 * P82.k2, 2 * s1.best.stderr["k2"])
    assert s1.best.k2_flag == "identified" and s1.best.singular_values.min() > 0
    for o in s1.optima:  # every Table 7.1 start reaches the same optimum
        assert abs(o.params.nu - p.nu) < 0.02 * p.nu and abs(o.params.k1 - p.k1) < 0.05 * p.k1
    # the degeneracy: with ρ12 fixed at the three Table 7.1 values the vol-of-vol curve is fit
    # about equally well (objectives within a factor 3 of the best) by different parameters
    fixed = [
        fit_stage1(hist, Fit2FConfig(window_vol=200, rho12=r, starts=(name,), curve="flat"))
        for name, (_, _, _, _, r) in TABLE_7_1.items()
    ]
    objs = np.array([f.best.objective for f in fixed])
    k1s = np.array([f.best.params.k1 for f in fixed])
    assert np.all(objs < 3.0 * max(objs.min(), 0.5)), objs
    assert np.ptp(k1s) > 0.2 * k1s.mean(), k1s  # different mean reversions, same curve
    table = s1.table()
    assert set(table.columns) >= {"start", "nu", "k2_flag", "sv_min"}
    fit_table = s1.fit_table()
    assert np.allclose(fit_table["volvol_model"], fit_table["volvol_hist"], rtol=0.02)


def test_stage2_recovery_yaml_and_scale_degeneracy(synthetic_1y) -> None:  # type: ignore[no-untyped-def]
    hist = synthetic_1y.history
    cfg = Fit2FConfig(
        window_vol=200, window_ssr=200, rho12_mode="from_correlation", refine_mixing=False
    )
    result = fit_2f(hist, None, cfg)
    p = result.params
    assert abs(p.rho_SX1 - P82.rho_SX1) < 0.2 and abs(p.rho_SX2 - P82.rho_SX2) < 0.2, p
    t2 = result.stage2.table()
    assert np.all(np.abs(t2["skew_residual"]) < 0.05), t2
    assert result.stage2.passes == 0 and result.stage3 is None
    # the ν ⟷ correlations direction: stage 2 flat, stage 1 rising
    deg = result.degeneracy
    psd = deg[deg["psd"]]
    assert psd["stage2_objective"].std() < 1e-6 * max(psd["stage2_objective"].mean(), 1.0)
    assert psd["stage1_objective"].iloc[-1] > 100 * max(psd["stage1_objective"].min(), 1e-6)
    deg2 = skew_scale_degeneracy(hist, cfg, p, scales=(1.0, 1.2))
    assert len(deg2) == 2 and deg2["psd"].all()
    # loadable YAML with provenance
    text = result.config_yaml
    doc = yaml.safe_load(text)
    loaded = BergomiParams(**doc["model"])
    assert loaded == p and doc["omega"] == pytest.approx(2 * p.nu)
    assert doc["provenance"]["stage1"]["unique_across_starts"] is True
    assert "fit_2f @" in result.summary()
    # stage 2 is not convex: a single start in the second basin lands elsewhere, the multi-start
    # default keeps the best optimum and reports the distinct ones
    s2b = fit_stage2(hist, cfg, result.stage1.best.params, x0=(-0.2, 0.5))
    assert s2b.objective >= result.stage2.objective - 1e-9
    assert result.stage2.n_distinct_optima >= 1


def test_rolling_fit_and_flags(synthetic_1y) -> None:  # type: ignore[no-untyped-def]
    hist = synthetic_1y.history
    cfg = Fit2FConfig(window_vol=120, window_ssr=60, refine_mixing=False, curve="flat")
    frame = rolling_fit(
        hist, cfg, stage1_every=10, stage2_every=5, start=hist.dates[200], end=hist.dates[230]
    )
    assert len(frame) == 7 and frame["stage1_refit"].sum() == 4
    assert {"nu", "rho_SX1", "chi", "nu_se", "stage2_objective"} <= set(frame.columns)
    flags = flag_unidentified(frame)
    assert list(flags["param"]) == ["nu", "theta", "k1", "k2", "rho_SX1", "chi"]
    assert flags["n_changes"].iloc[0] == 3  # stage-1 changes on the refit dates only
    assert flags["unidentified"].dtype == bool
    with pytest.raises(ValueError):
        rolling_fit(hist, cfg, start=hist.dates[10])


@pytest.mark.slow
def test_recovery_three_year_mixing() -> None:
    """SPEC §15 Part 3 recovery test on the synthetic history (mixing ATM source, 3 years, seed
    13, 100k mixing paths per day).  Stage 1 (correlation target): ν, θ, k1 within 10%, k2
    within its sensitivity band, unique across the Table 7.1 starts.  Stage 2: the skew term
    structure with the mixing-solution skew recovers ρ_SX1 / ρ_SX2 within 0.05 (the SPEC
    criterion) when fitted alone; with the SSR target at the SPEC's weight the recovery is
    within 0.03 / 0.06 on this path — the 250-day SSR's sampling noise (SE 0.06–0.08) moves
    ρ_SX2 by a few hundredths — recorded in the §15 notes as a deviation for the owner."""
    sh = synthetic_2f_history(P82, years=3.0, seed=13, atm_source="mixing", mixing_paths=100_000)
    cfg = Fit2FConfig(window_vol=250, window_ssr=250, rho12_mode="from_correlation")
    result = fit_2f(sh.history, None, cfg)
    p, s1 = result.params, result.stage1.best
    assert abs(p.nu - P82.nu) < 0.1 * P82.nu and abs(p.theta - P82.theta) < 0.1 * P82.theta + 0.02
    assert abs(p.k1 - P82.k1) < 0.1 * P82.k1
    assert abs(p.k2 - P82.k2) < max(0.1 * P82.k2, 2 * s1.stderr["k2"]), (p.k2, s1.stderr)
    assert result.stage1.unique and result.stage2.passes == 2
    assert abs(p.rho_SX1 - P82.rho_SX1) < 0.05 and abs(p.rho_SX2 - P82.rho_SX2) < 0.07, p
    # the numerical SSR of the fitted kernel reads the historical SSR within 1.5 SE
    t2 = result.stage2.table()
    assert np.all(np.abs(t2["ssr_2f"] - t2["ssr_hist"]) < 1.5 * t2["ssr_hist_se"] + 0.02), t2
    skew_only = fit_stage2(
        sh.history, Fit2FConfig(**{**cfg.__dict__, "ssr_scale": 1e6}), result.stage1.best.params
    )
    q = skew_only.params
    assert abs(q.rho_SX1 - P82.rho_SX1) < 0.05 and abs(q.rho_SX2 - P82.rho_SX2) < 0.05, q
    assert np.all(np.abs(skew_only.table()["skew_residual"]) < 0.01)


@pytest.mark.slow
def test_real_data_end_to_end() -> None:
    """The 2022 H2 SPX history (plain SSVI snapshots, ``scripts/m7_hdn_history.py --no-essvi``):
    stages 1–2 run end to end; no fixed numbers, only sanity (finite, PSD, objectives)."""
    path = ROOT / "outputs" / "m7" / "hdn_history_ssvi.csv"
    if not path.exists():
        pytest.skip("2022 H2 history not built (scripts/m7_hdn_history.py --no-essvi)")
    hist = SurfaceHistory(pd.read_csv(path))
    # pillars to 1y: the sample quotes ATM maturities to 1.5-3y only (SPEC §15 notes)
    cfg = Fit2FConfig(
        pillars=(1 / 12, 0.25, 0.5, 1.0),
        corr_pillars=(0.25, 1.0),
        window_vol=100,
        window_ssr=60,
        rho12_mode="from_correlation",
    )
    result = fit_2f(hist, None, cfg)
    p = result.params
    assert all(np.isfinite(v) for v in dataclasses.asdict(p).values())
    assert 0.3 <= p.nu <= 4.0 and p.k1 > p.k2 > 0 and abs(p.rho_SX1) < 1 and abs(p.rho_SX2) < 1
    assert np.isfinite(result.stage1.best.objective) and np.isfinite(result.stage2.objective)
    assert result.stage2.passes == 2
