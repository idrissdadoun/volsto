"""M3b tests (SPEC §13): HistoricalData.net importer on the sample day 2022-09-15 (SPX).

Skipped when ``./data/hdn_sample/options_sample_2022H2`` is absent (the sample is licensed and
git-ignored).
"""

from __future__ import annotations

import importlib.util
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from volsto.config import MarketConfig, SSVIConfig, load_yaml
from volsto.market import ESSVISurface, SSVISurface, implied_vol, import_hdn
from volsto.market.import_hdn import (
    DEFAULT_CALENDAR_REPAIR,
    HDN_COLUMNS,
    HdnFilters,
    SSVIFit,
    SurfacePoints,
    _eta_max,
    _eta_theta_range,
    _surface_theta,
    calendar_constraint_grid,
    implied_forwards,
    import_day,
    load_day,
    load_manifest,
    main,
    write_snapshot,
)
from volsto.market.loaders import load_ssvi_surface

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "hdn_sample" / "options_sample_2022H2"
DAY = "2022-09-15"
FAILING_DAY = "2022-12-21"
"""One of the 15 days whose unrepaired eSSVI fails the calendar check (M10 Part 0)."""
Pipeline = tuple[dict[str, Any], SSVIFit, SurfacePoints, pd.DataFrame]
pytestmark = pytest.mark.skipif(not SAMPLE.exists(), reason="HDN sample not present")


@pytest.fixture(scope="module")
def chain() -> pd.DataFrame:
    return load_day(
        SAMPLE / "day_by_date" / f"{DAY}_options.csv", "SPX", manifest=load_manifest(SAMPLE)
    )


@pytest.fixture(scope="module")
def pipeline() -> Pipeline:
    return import_day(SAMPLE, DAY, "SPX")


def test_load_day_roots_quotes_and_time_convention(chain: pd.DataFrame) -> None:
    assert set(chain["root"].unique()) <= {"SPX", "SPXW"}
    assert (chain["bid"] > 0).all() and (chain["ask"] > 0).all()
    assert chain.attrs["quote_date"] == DAY and chain.attrs["spot"] > 3000
    # AM settlement loses one calendar day (README §5.2); PM does not
    am = chain[chain["settlement_time"] == "AM"].iloc[0]
    days_am = (pd.Timestamp(am["expiration"]) - pd.Timestamp(DAY)).days
    assert am["T"] == pytest.approx((days_am - 1) / 365.0)
    pm = chain[chain["settlement_time"] == "PM"].iloc[0]
    days_pm = (pd.Timestamp(pm["expiration"]) - pd.Timestamp(DAY)).days
    assert pm["T"] == pytest.approx(days_pm / 365.0)
    # rate from the manifest curve, flat below the 1m tenor, ~2.8-4% in Sept 2022
    assert 0.02 < chain["r"].min() < chain["r"].max() < 0.05
    # SPX (AM) and SPXW (PM) share expiration dates but are separate slices
    assert chain.groupby("expiration")["expiry"].nunique().max() == 2


def test_implied_forward_agrees_with_vendor_parity_forward(chain: pd.DataFrame) -> None:
    """Regression forward vs the vendor's closest-strike parity forward (README §5.4) within a few
    bp up to 1y; the residual grows with T because the vendor discounts at the Treasury rate while
    the regression uses the market-implied discount factor."""
    fwds = implied_forwards(chain, max_years=2.0)
    checked = 0
    for fe in fwds.values():
        if fe.T > 1.0 or fe.n_pairs < 5:
            continue
        assert abs(fe.forward / fe.vendor_style_forward - 1.0) < 3e-4, fe  # 3 bp
        assert 0.0 < fe.forward_stderr < 0.5, fe
        checked += 1
    assert checked >= 20
    # implied discount within 1.5% (absolute rate within ~1.5%) of the Treasury discount at 1y
    one_y = min(fwds.values(), key=lambda fe: abs(fe.T - 1.0))
    assert (
        abs(one_y.implied_rate - float(chain.loc[chain["expiry"] == one_y.expiry, "r"].iloc[0]))
        < 0.015
    )


def test_mid_implied_vols_reproduce_vendor_iv_on_flag0_rows(chain: pd.DataFrame) -> None:
    """Our Black-76 inversion with the implied forward reproduces the vendor's mid iv (iv_flag 0,
    European SPX) to a few tenths of a vol point near the money for 1m-1y."""
    fwds = implied_forwards(chain, max_years=2.0)
    diffs = []
    for e, fe in fwds.items():
        if not 1 / 12 <= fe.T <= 1.0:
            continue
        g = chain[
            (chain["expiry"] == e)
            & (chain["iv_flag"] == 0)
            & (np.abs(np.log(chain["strike"] / fe.forward)) < 0.1)
        ]
        if g.empty:
            continue
        iv = implied_vol(
            g["mid"].to_numpy(float),
            fe.forward,
            g["strike"].to_numpy(float),
            fe.T,
            g["cp"].to_numpy(int),
            fe.discount,
        )
        diffs.append(np.abs(iv - g["iv"].to_numpy(float)))
    d = np.concatenate(diffs)
    assert np.median(d) < 0.001 and np.quantile(d, 0.9) < 0.005, (np.median(d), np.quantile(d, 0.9))


def test_grid_surface_and_ssvi_fit(pipeline: Pipeline) -> None:
    _cfg, fit, points, _chain = pipeline
    assert points.table["expiry"].nunique() >= 25 and len(points.table) > 3000
    assert set(points.dropped) >= {"butterfly", "calendar", "spread", "min_bid"}
    assert isinstance(fit.surface, ESSVISurface)  # imported surfaces default to eSSVI
    fit.surface.check_no_arbitrage()  # raises on violation
    # SSVI: rms ≤ 0.3 vp inside ±20% from 3m to 2y (single power law cannot follow the 1-2m
    # weeklies of this high-vol day: reported, not asserted)
    assert fit.rms_error(2.0, 0.2, 0.25) < 0.30, fit.rms_error(2.0, 0.2, 0.25)
    assert fit.max_error(2.0, 0.2, 0.5) < 1.0
    assert all(-0.9 < r < -0.3 for r in fit.params["rho"]) and 0.05 < fit.params["gamma"] <= 1.0
    # ATM term structure at the pillar tenors, in a plausible range for Sept 2022
    assert all(0.15 < v < 0.40 for v in fit.params["atm_vols"])


def test_essvi_fit_improves_near_money(pipeline: Pipeline) -> None:
    _cfg, fit_e, _points, _chain = pipeline
    _, fit, _, _ = import_day(SAMPLE, DAY, "SPX", essvi=False)  # plain SSVI, single rho
    assert isinstance(fit.surface, SSVISurface) and isinstance(fit_e.surface, ESSVISurface)
    fit.surface.check_no_arbitrage()
    assert -0.9 < fit.params["rho"] < -0.3
    assert fit_e.rms_error(2.0, 0.2, 0.25) <= fit.rms_error(2.0, 0.2, 0.25) + 1e-9
    assert fit_e.rms_error(2.0, 0.2, 0.5) < 0.20
    assert len(fit_e.params["rho"]) == len(fit_e.params["atm_maturities"])


def test_snapshot_config_round_trip(pipeline: Pipeline, tmp_path: Path) -> None:
    cfg, fit, points, chain = pipeline
    p = write_snapshot(cfg, tmp_path / "spx.yaml")
    surface = load_ssvi_surface(p)
    market = load_yaml(p, MarketConfig, section="market")
    ssvi = load_yaml(p, SSVIConfig, section="ssvi")
    assert market.spot == pytest.approx(chain.attrs["spot"])
    assert isinstance(surface, ESSVISurface)
    assert ssvi.rho == pytest.approx(float(np.mean(fit.params["rho"])))
    np.testing.assert_allclose(surface.atm_vol(1.0), fit.surface.atm_vol(1.0), rtol=1e-9)
    prov = cfg["provenance"]
    assert prov["vendor"] == "historicaldata.net" and prov["file_sha256"] == prov["manifest_sha256"]
    assert prov["filters"]["min_bid"] == HdnFilters().min_bid and prov["n_points"] == len(
        points.table
    )


def test_essvi_snapshot_round_trip(tmp_path: Path) -> None:
    cfg, fit, _, _ = import_day(SAMPLE, DAY, "SPX")
    p = write_snapshot(cfg, tmp_path / "spx_essvi.yaml")
    surface = load_ssvi_surface(p)
    assert isinstance(surface, ESSVISurface)
    np.testing.assert_allclose(
        surface.implied_vol_k([-0.1, 0.0, 0.1], 1.0),
        fit.surface.implied_vol_k([-0.1, 0.0, 0.1], 1.0),
        rtol=1e-9,
    )
    assert cfg["essvi"]["rhos"] == fit.params["rho"]


def test_cli_writes_config(tmp_path: Path) -> None:
    rc = main(
        [
            "--vendor",
            "hdn",
            "--date",
            DAY,
            "--underlying",
            "SPX",
            "--root",
            str(SAMPLE),
            "--out",
            str(tmp_path),
        ]
    )
    assert rc == 0
    out = tmp_path / f"spx_{DAY}.yaml"
    assert out.exists()
    assert load_ssvi_surface(out).atm_vol(0.5) > 0.1


def test_yfinance_capture_layout(tmp_path: Path) -> None:
    """The capture script's row builder emits the 34-column layout the importer reads."""
    spec = importlib.util.spec_from_file_location(
        "capture_yfinance", ROOT / "scripts" / "capture_yfinance.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["capture_yfinance"] = mod
    spec.loader.exec_module(mod)
    calls = pd.DataFrame(
        {
            "contractSymbol": ["SPXW230120C04000000", "SPX230120C04100000"],
            "strike": [4000.0, 4100.0],
            "bid": [10.0, 5.0],
            "ask": [10.5, 5.5],
            "volume": [1, 2],
            "openInterest": [10, 20],
            "lastPrice": [10.2, 5.1],
        }
    )
    rows = mod.build_rows(
        calls,
        underlying="SPX",
        expiration="2023-01-20",
        quote_date="2022-12-30",
        spot=3850.0,
        opt_type="call",
    )
    assert list(rows.columns) == list(HDN_COLUMNS)
    assert rows["settlement_time"].tolist() == ["PM", "AM"]  # SPXW weekly vs SPX third Friday
    assert (rows["iv_flag"] == 7).all()
    path = mod.write_day(rows, tmp_path, "2022-12-30", [4.1, 4.4, 4.7, 4.4, 4.0, 3.9, 4.0])
    man = load_manifest(tmp_path)
    assert man["rates"]["2022-12-30"][0] == 4.1 and man["files"][0]["rows"] == 2
    day = load_day(path, "SPX", manifest=man)
    assert len(day) == 2 and set(day["root"]) == {"SPX", "SPXW"} and (day["T"] > 0).all()


# --------------------------------------------------------------------------------------------
# eSSVI calendar repair (M10 Part 0)
# --------------------------------------------------------------------------------------------


class _UncheckedESSVI(ESSVISurface):
    def _check_calendar_numeric(self) -> None:
        return None


@pytest.fixture(scope="module")
def failing_day() -> Pipeline:
    return import_day(SAMPLE, FAILING_DAY, "SPX")


def test_feasible_day_is_bit_identical(pipeline: Pipeline) -> None:
    """2022-09-15: the certificate proves ``∂_T w ≥ margin`` on ±3 for the unrepaired fit, so
    the repair is skipped and the fit equals the pre-M10 fit (``calendar_repair=None``) bit for
    bit."""
    _cfg, fit, _points, _chain = pipeline
    cfg = DEFAULT_CALENDAR_REPAIR
    assert fit.params["calendar_repaired"] is False and fit.params["calendar_stages"] == 0
    assert fit.params["calendar_fallback"] is None and fit.params["calendar_cost_delta"] == 0.0
    assert fit.params["calendar_min_dw_dt"] >= cfg.margin
    assert fit.params["calendar_floor"] == cfg.margin and fit.params["calendar_k_abs"] == 3.0
    assert fit.params["calendar_lower_bound"] >= cfg.margin
    _, old, _, _ = import_day(SAMPLE, DAY, "SPX", calendar_repair=None)
    assert "calendar_repaired" not in old.params
    for key in ("rho", "eta", "gamma", "atm_vols", "atm_maturities", "cost"):
        assert fit.params[key] == old.params[key], key
    assert isinstance(old.surface, ESSVISurface)
    assert repr(fit.surface) == repr(old.surface)
    pd.testing.assert_frame_equal(fit.points, old.points)


def test_essvi_repair_on_failing_day(
    failing_day: Pipeline, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2022-12-21: the unrepaired eSSVI is refused by the calendar check; the repaired one
    imports with the invariant PROVEN (exact ``∂_T w ≥ margin`` on ±3, every ``T``, both knot
    limits — re-proven here on the returned surface and seen on the dense check).

    Its fit cost is checked against a bracket that is well motivated but NOT guaranteed, because
    both ends assume optimality and ``least_squares`` only returns a local minimiser:

    * above the unrepaired cost — the unrepaired fit minimises the same residual without the
      constraint (``calendar_cost_delta > 0``);
    * at most the plain-SSVI cost — the SSVI fit is a point of the repair's feasible set (same
      ``θ_T``; one ``ρ`` at every pillar; its ``η`` below the repair's cap, its exact ``∂_T w``
      above ``margin + headroom`` on the constraint grid and its invariant proven, all three
      checked here), and the constrained optimum costs no more than any feasible point; a
      penalty iterate that globally minimised its penalised objective would cost no more than
      that optimum.

    (Measured: cost 85.87 -> 90.18 against SSVI 121.36; RMS inside ±20%, 3m-3y 0.2797 ->
    0.2989 against SSVI 0.3256 — the RMS is not the fitted objective, so it is not bounded.)"""
    with pytest.raises(ValueError, match="calendar"):
        import_day(SAMPLE, FAILING_DAY, "SPX", calendar_repair=None)
    _cfg, fit, _points, _chain = failing_day
    cfg = DEFAULT_CALENDAR_REPAIR
    assert isinstance(fit.surface, ESSVISurface)
    assert fit.params["calendar_repaired"] is True and fit.params["calendar_fallback"] is None
    assert fit.params["calendar_min_dw_dt_before"] < 0.0
    assert fit.params["calendar_min_dw_dt"] >= cfg.margin + cfg.headroom - cfg.tol
    assert fit.params["calendar_lower_bound"] >= cfg.margin
    assert fit.params["calendar_floor"] == cfg.margin and fit.params["calendar_k_abs"] == 3.0
    assert fit.surface.calendar_certificate(3.0, cfg.margin).certified
    assert fit.surface.calendar_min_dw_dt(3.0, 121) >= cfg.margin
    assert fit.surface.calendar_min_dw_dt() == pytest.approx(
        fit.params["calendar_min_dw_dt_1"], rel=1e-9
    )
    dense = fit.surface.calendar_dense_check()
    assert dense.ok and dense.min_dw_dt >= cfg.margin, dense
    assert fit.params["calendar_cost_delta"] > 0.0

    _, plain, _, _ = import_day(SAMPLE, FAILING_DAY, "SPX", essvi=False)
    pil = np.asarray(plain.params["atm_maturities"])
    th = np.asarray(plain.params["atm_vols"]) ** 2 * pil
    np.testing.assert_array_equal(th, np.asarray(fit.params["atm_vols"]) ** 2 * pil)
    lo, hi = _eta_theta_range(th)
    ends = _surface_theta(pil, th, np.array([1.0 / 365.0, fit.surface.max_maturity]))
    cap = 0.999 * _eta_max(
        abs(plain.params["rho"]), plain.params["gamma"], min(lo, ends[0]), max(hi, ends[1])
    )
    assert plain.params["eta"] <= cap
    as_essvi = ESSVISurface(
        pil,
        th,
        [plain.params["rho"]] * pil.size,
        plain.params["eta"],
        plain.params["gamma"],
        plain.surface.forward_curve,
        plain.surface.discount,
        max_maturity=fit.surface.max_maturity,
    )
    ks, seg, ts = calendar_constraint_grid(pil, fit.surface.max_maturity, cfg)
    D = as_essvi.calendar_segments().dw_dt(
        ks[None, :], seg[:, None], ts[:, None], as_essvi.eta, as_essvi.gamma
    )
    assert np.min(D) >= cfg.margin + cfg.headroom
    assert as_essvi.calendar_certificate(3.0, cfg.margin).certified
    assert fit.params["cost"] <= plain.params["cost"], (fit.params["cost"], plain.params["cost"])

    monkeypatch.setattr(import_hdn, "ESSVISurface", _UncheckedESSVI)
    _, base, _, _ = import_day(SAMPLE, FAILING_DAY, "SPX", calendar_repair=None)
    assert isinstance(base.surface, ESSVISurface)
    assert base.surface.calendar_min_dw_dt() < -ESSVISurface.CALENDAR_TOL
    assert not base.surface.calendar_dense_check().ok
    assert base.params["cost"] < fit.params["cost"]


def test_repaired_snapshot_round_trip(failing_day: Pipeline, tmp_path: Path) -> None:
    cfg, fit, _points, _chain = failing_day
    p = write_snapshot(cfg, tmp_path / "spx_repaired.yaml")
    surface = load_ssvi_surface(p)
    assert isinstance(surface, ESSVISurface)
    assert cfg["essvi"]["rhos"] == fit.params["rho"]
    np.testing.assert_allclose(
        surface.implied_vol_k([-0.3, 0.0, 0.3], [0.5, 2.0, 3.0]),
        fit.surface.implied_vol_k([-0.3, 0.0, 0.3], [0.5, 2.0, 3.0]),
        rtol=1e-12,
    )
    prov = cfg["provenance"]["fit"]
    for key in (
        "calendar_repaired",
        "calendar_stages",
        "calendar_min_dw_dt",
        "calendar_lower_bound",
        "calendar_fallback",
    ):
        assert prov[key] == fit.params[key], key


def test_repair_failure_falls_back_to_ssvi() -> None:
    """No penalty stage allowed: the escalation ends in plain SSVI for the day, recorded."""
    cfg = replace(DEFAULT_CALENDAR_REPAIR, max_stages=0)
    _, fit, _, _ = import_day(SAMPLE, FAILING_DAY, "SPX", calendar_repair=cfg)
    assert type(fit.surface) is SSVISurface
    assert fit.params["calendar_fallback"] == "ssvi" and fit.params["essvi"] is False
    assert fit.params["calendar_min_dw_dt_before"] < 0.0
    assert fit.params["calendar_min_dw_dt"] is None
    assert fit.params["calendar_floor"] is None and fit.params["calendar_lower_bound"] is None
    _, plain, _, _ = import_day(SAMPLE, FAILING_DAY, "SPX", essvi=False)
    assert fit.params["rho"] == plain.params["rho"] and fit.params["eta"] == plain.params["eta"]


def test_cli_no_calendar_repair(tmp_path: Path) -> None:
    rc = main(
        ["--date", DAY, "--root", str(SAMPLE), "--out", str(tmp_path), "--no-calendar-repair"]
    )
    assert rc == 0
    doc = load_ssvi_surface(tmp_path / f"spx_{DAY}.yaml")
    assert isinstance(doc, ESSVISurface)


def _gate_script() -> Any:
    """``scripts/essvi_calendar_gate.py`` as a module (its per-day chain and summary are reused,
    not duplicated)."""
    spec = importlib.util.spec_from_file_location(
        "essvi_calendar_gate", ROOT / "scripts" / "essvi_calendar_gate.py"
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["essvi_calendar_gate"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.slow
def test_part0_gate_on_the_127_day_sample() -> None:
    """M10 Part 0's gate (owner: "eSSVI passes on >= 95% of days, and the 2y-3y skew is no longer
    extrapolated on passing days") on the whole 2022 H2 sample, through the gate script's own
    per-day chain (:func:`run_day`: the unrepaired, repaired and plain-SSVI fits of one day) and
    summary (:func:`summarise`); single core, no Monte Carlo, nothing written.

    Asserted: the repaired eSSVI passes on at least ``GATE_PASS_FRACTION`` of the days with no
    fallback; every day's surface carries the importer's proof (floor 1e-4 on ``|k| ≤ 3``) and
    re-proves it independently (certificate at the margin on ``±3``); the repaired / bit-equal
    split and the pillar counts SPEC §13.1 records.  The owner's second criterion cannot be met by
    any repair (SPEC §13.1, disagreement 3): on 118 of the 127 days the last fitted pillar is at
    or below 2y (P = 5 / 6 / 7 pillars on 1 / 117 / 9 days, one day quotes at or beyond 3y), so
    the 3y skew is flat-forward-variance extrapolation whatever the parametrisation.  The test
    pins that measured count instead of asserting the criterion, and asserts what the repair
    delivers instead: on every passing day the 2y/3y skew is the eSSVI ``ρ_T``'s, not the
    single-``ρ`` fallback's (the medians of SPEC §13.1)."""
    gate = _gate_script()
    from volsto.calibration.history import hdn_available_dates

    dates = hdn_available_dates(SAMPLE)
    assert len(dates) == 127
    t0 = time.perf_counter()
    df = pd.DataFrame([gate.run_day(SAMPLE, d, with_ssvi=True) for d in dates])
    wall = time.perf_counter() - t0
    s = gate.summarise(df, wall, True)
    passing, pillars, skew = s["passing"], s["pillars"], s["skew"]
    print(
        f"Part 0 gate: {passing['repaired']}/{len(df)} repaired eSSVI pass (unrepaired "
        f"{passing['base']}), {passing['n_repaired']} repaired, fallbacks {passing['fallbacks']}, "
        f"min proven bound {passing['min_proven_lower_bound']:.4e}; P {pillars['P_distribution']}; "
        f"wall clock {wall:.1f} s (recalibrated: no)"
    )
    # the owner's threshold, with no fallback
    assert passing["repaired_fraction"] >= gate.GATE_PASS_FRACTION and passing["gate_met"]
    assert df["fallback"].isna().all() and passing["fallbacks"] == {}
    assert df["rep_essvi"].all() and passing["repaired"] == len(df)
    # the invariant, proven on every day (the importer's proof and an independent certificate)
    assert (df["proven_floor"] == gate.MARGIN).all() and (df["proven_k_abs"] == gate.K_WIDE).all()
    assert (df["proven_lower_bound"] >= gate.MARGIN).all()
    assert (df["rep_certm_k3"] == "certified").all()
    assert (df["rep_certm_k3_lb"] >= gate.MARGIN).all()
    repaired = df[df["repaired"]]
    assert len(repaired) == 25 and (repaired["rep_certm_k3"] == "certified").all()
    assert passing["n_identity_bit_equal"] == 102  # the other days are returned bit-equal
    assert (df["rep_dense_min_dw_dt"] >= gate.MARGIN - 1e-12).all()
    # the second criterion: measured, not met (SPEC §13.1)
    last = df["last_pillar"]
    assert int((last <= 2.0 + 1e-9).sum()) == 118
    assert pillars["P_distribution"] == {"5": 1, "6": 117, "7": 9}
    assert pillars["days_quote_at_or_beyond_3y"] == 1
    # what the repair delivers instead: the eSSVI rho_T's 2y/3y skew, not the SSVI fallback's
    for tenor, rep, ssvi in (("2y", -0.1966, -0.2240), ("3y", -0.1626, -0.1910)):
        assert skew["rep"][tenor] == pytest.approx(rep, abs=1e-4)
        assert skew["ssvi"][tenor] == pytest.approx(ssvi, abs=1e-4)
    assert skew["median_gap_vs_ssvi"]["rep_on_passing_days_n"] == len(df)
