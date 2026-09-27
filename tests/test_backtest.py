"""Rolling backtest (SPEC §10.3, owner's M10 Part 3; ``volsto/studies/backtest.py``).

The owner's test: **the backtest runs 5 dates end to end with --shard and --resume**.  The only
calibrating site is the session fixture ``tests/_backtest_build.py::toy_backtest_build`` (the
toy config's 5 HDN dates, 2·10⁴ particles over a 1y horizon, calibrated once per pytest run);
every other test works on a **copy** of that build and runs ``volsto-backtest run
--no-calibrate`` (the calibration guard held) or reads the stored rows.  Without the HDN sample
the fixture-based tests skip with the reason.

What is asserted:

* config: strict loading (unknown / missing keys, kind-specific keys, the trading grid, stage 3
  and the recalibrate mode refused), the hash leaves out the paths, the data root and the end
  date; the 3y term sheets are the M6 headline products exactly; contiguous shards;
* the ladders attribution: the start-of-period Greeks read back from the memo reproduce every
  bucket of ``explain`` from the surface moves (Black–Scholes builder, no Monte Carlo cost), and
  cost no extra pricing;
* the build: 5 dates done, one calibration each (printed, never asserted on time);
* shards 1/2 and 2/2 under ``--no-calibrate`` give exactly the unsharded rows and fits; a second
  ``--resume`` computes nothing and touches nothing; the cache is unchanged;
* a changed config hash is refused (exit 2) and ``--no-calibrate`` with a missing leverage exits
  2 with the ``volsto-backtest run`` line;
* the projection prints before any work (and ``dry-run`` writes no date);
* stage 2 renders tables and figures from the stored rows through the runner with
  ``recalibrated: false``, rebuilds them from ``results.parquet`` alone, and a missing date is a
  requirement whose command is the ``volsto-backtest run ... --only-dates`` line;
* the attribution buckets (Greeks + recalibration + residual) plus the cash flows sum to the MTM
  P&L; the previous value the attribution prices is the stored one; no extra pricing;
* a fixed trade's value on its inception date is the fresh product's price under the same seed;
* the VKO row carries its realised state (returns, realised vol from the stored closes, the
  knock-out flag, the budget);
* the fitted-parameter frame feeds ``flag_unidentified`` (the marking fits carry no standard
  error: every date is counted without one); the historical-mode flags and the realised SSR on a
  synthetic 30-date history; the realised SSR is NaN with its reason before ``window + 1``
  dates;
* integrity: the **walking test** — every recorded input (the snapshot, every file of the
  current attempt including fields of done.json outside the record, an unexpected file, the
  CURRENT pointer, the leverage, the base spec, the manifest rates, the calendar, the vendor day
  file of the previous date) mutated in turn makes the date and every later date unconfirmed,
  and stage 2 reports exactly them; an edited snapshot is re-imported; contiguous shards run in
  reverse order (the later block pending until the earlier one is stored); an incomplete date
  computed alone waits, then turns stale and is recomputed next to the kept attempt;
* storage: both refusals read the verdicts (a wrong cache; N13, a previous date without stored
  results whose leverage the next date's P&L used); the **crash matrix** — an exception, a
  KeyboardInterrupt and a simulated kill at every step of the attempt and pointer paths, plus
  four real kills — never loses a confirmed result, CURRENT always names a complete attempt,
  and ``run`` adopts or recomputes; two processes commit the same date while ``gc`` runs and a
  lock-free reader always verifies; the round-2 store (``tests/golden``) is migrated in place,
  bit-identical, and a migration interrupted at each step completes on the next open;
* the fourth pass's principles, each on the verifier's case: P1 verdicts read once under churn
  (race_toctou); P2 a failure never displaces results (G3); P3 gc refuses another config (G2),
  keeps the attempts of a date that is not done and skips symbolic links (L1); P4 refusals come
  before any pointer move (G1); P5 snapshots bound only when a fresh import reproduces them
  (M3); P6 exactly ``.DS_Store`` and ``._<listed name>`` ignored (F1); P7 a flat date without
  done.json quarantined (M1) and migration cleanup crashes finished (k = 124..129); P8 unknown
  pointer versions (V1) and foreign leftovers (N4); the code version informational;
* the P&L and residual standard errors of ``explain`` are paired (no extra pricing) and match
  the seed dispersion.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import hashlib
import io
import json
import math
import os
import re
import shlex
import shutil
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import _backtest_build
import numpy as np
import pandas as pd
import pytest
import yaml
from _backtest_build import BacktestBuild, toy_backtest_build  # noqa: F401

from volsto.calibration import guard
from volsto.calibration.fit_2f import BreakEvenFitConfig, fit_preset
from volsto.calibration.history import DEFAULT_PILLARS
from volsto.calibration.stability import PARAM_COLUMNS, flag_unidentified
from volsto.config import CalibrationSpec, ConfigError, CurveConfig, MarketConfig, SimConfig
from volsto.config import load_yaml as load_config_yaml
from volsto.market.curves import DiscountCurve
from volsto.products.base import Product
from volsto.products.seasoning import RealisedHistory, replay
from volsto.products.variance import VarianceSwap
from volsto.risk.attribution import explain
from volsto.risk.engine import BSBuilder, LSVBuilder, RiskEngine, RiskState, product_key, surface_of
from volsto.studies import backtest as bt
from volsto.studies import runner
from volsto.studies.m6 import headline_products
from volsto.studies.results import Results, ResultsBuilder

ROOT = Path(__file__).resolve().parents[1]
POC_CONFIG = ROOT / "configs" / "backtest" / "hdn_2022h2.yaml"
FULL_CONFIG = ROOT / "configs" / "backtest" / "hdn_2022h2_full.yaml"
ESSVI_SNAPSHOT = ROOT / "configs" / "surfaces" / "snapshots" / "spx_2022-09-15.yaml"
#: ``spec_key`` of configs/studies/lsv_reference_2f.yaml recorded on 2026-09-16 at commit 0ca0897
#: (tests/test_surface_config.py::RECORDED_SPEC_KEYS)
RECORDED_REFERENCE_2F_KEY = "6a007884b6de9bd0b7dad2540e77ad5f647268783d76ea2586a2a426609d937c"
TOY_CONFIG = ROOT / "configs" / "backtest" / "hdn_2022h2_toy.yaml"
STUDY_FAST = ROOT / "configs" / "studies" / "catalogue" / "backtest_2022h2_fast.yaml"
STUDY_FULL = ROOT / "configs" / "studies" / "catalogue" / "backtest_2022h2.yaml"
REFERENCE_SPEC = ROOT / "configs" / "studies" / "lsv_reference_2f.yaml"
#: Row columns that legitimately differ between two runs of the same date (timings).
VOLATILE = ("seconds",)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def raw_config(path: Path = TOY_CONFIG) -> dict[str, Any]:
    return dict(yaml.safe_load(path.read_text()))


def write_config(tmp_path: Path, data: dict[str, Any], name: str = "cfg.yaml") -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data, sort_keys=False))
    return p


def current(store: Path, date: str, name: str = "") -> Path:
    """The current attempt of a date (``CURRENT``), or one of its files."""
    ptr = json.loads((store / "dates" / date / "CURRENT").read_text())
    path = store / "dates" / date / "attempts" / ptr["attempt"]
    return path / name if name else path


def current_json(store: Path, date: str, name: str) -> dict[str, Any]:
    return dict(json.loads(current(store, date, name).read_text()))


def stored_dates(store: Path) -> list[str]:
    return sorted(p.parent.name for p in (store / "dates").glob("*/CURRENT"))


def store_rows(store: Path) -> pd.DataFrame:
    frames = [
        pd.read_parquet(current(store, d, "rows.parquet"))
        for d in stored_dates(store)
        if current(store, d, "rows.parquet").is_file()
    ]
    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["date", "trade_id"]).reset_index(drop=True)


def store_fits(store: Path) -> dict[str, dict[str, Any]]:
    return {
        d: current_json(store, d, "fit.json")
        for d in stored_dates(store)
        if current(store, d, "fit.json").is_file()
    }


def cfg_at(build: BacktestBuild, *, out: Path | None = None, cfg: Path = TOY_CONFIG) -> Any:
    return bt.load_backtest_config(cfg).with_paths(
        out=out if out is not None else build.store_root,
        cache=build.cache_root,
        snapshots=build.snapshots_root,
    )


def tree_state(root: Path) -> dict[str, tuple[int, int]]:
    return {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def run_cli(argv: list[str], capsys: pytest.CaptureFixture[str]) -> tuple[int, str]:
    code = bt.main(argv)
    out = capsys.readouterr()
    return code, out.out + out.err


def assert_same_rows(a: pd.DataFrame, b: pd.DataFrame) -> None:
    cols = [c for c in bt.ROW_COLUMNS if c not in VOLATILE]
    pd.testing.assert_frame_equal(a[cols], b[cols], check_exact=True)


@pytest.fixture
def build_copy(toy_backtest_build: BacktestBuild, tmp_path: Path) -> BacktestBuild:  # noqa: F811
    return toy_backtest_build.require().copy(tmp_path / "build")


# --------------------------------------------------------------------------------------------
# config and book (no calibration, no Monte Carlo)
# --------------------------------------------------------------------------------------------


def test_configs_load_and_hash_rules(tmp_path: Path) -> None:
    poc = bt.load_backtest_config(POC_CONFIG)
    full = bt.load_backtest_config(FULL_CONFIG)
    toy = bt.load_backtest_config(TOY_CONFIG)
    assert poc.n_particles == 800_000 and poc.horizon == 3.0
    # the default is the cheapest configuration delivering every listed output; the rich one
    # differs only by its ladder pillars, its rolling marks and its paths
    assert poc.attribution["detail"] == "ladders" and poc.attribution["max_halvings"] == 6
    assert poc.attribution["pillars"] == [0.25, 0.5, 1.0, 2.0, 3.0]
    assert poc.rolling_mark == "inception" and poc.n_paths == 20_000
    assert full.rolling_mark == "daily" and len(full.attribution["pillars"]) == 9
    a, b = poc.content(), full.content()
    for m in (a, b):
        m.pop("name")
        m["attribution"].pop("pillars")
        m["book"]["rolling"].pop("mark")
    assert a == b
    assert full.path("out") != poc.path("out")
    for cfg_ in (poc, full, toy):
        text = Path(ROOT / cfg_.source).read_text()
        assert "1/(2 K_vol)" in text and "vs_strip" in text
    assert poc.section("marking")["stage3"] is False
    assert [t.id for t in poc.fixed] == [
        "autocall_3y",
        "phoenix_3y",
        "cliquet_1y",
        "vko_put_12m",
        "ko_var_1y",
        "var_swap_1y",
    ]
    assert toy.n_particles == 20_000 and toy.horizon <= 1.0
    assert max(t.maturity for t in (*toy.fixed, *toy.rolling)) <= toy.horizon
    # the hash leaves out the paths, the data root and the end date; keeps the start
    data = raw_config()
    moved = dict(data, paths={"out": "/x", "cache": "/y", "snapshots": "/z"})
    moved["data"] = dict(data["data"], root="/elsewhere")
    moved["dates"] = dict(data["dates"], end="2022-12-30")
    assert bt.BacktestConfig.from_mapping(moved).content_hash() == toy.content_hash()
    shifted = dict(data, dates=dict(data["dates"], start="2022-07-28"))
    assert bt.BacktestConfig.from_mapping(shifted).content_hash() != toy.content_hash()
    more_paths = dict(data, pricing=dict(data["pricing"], n_paths=2002))
    assert bt.BacktestConfig.from_mapping(more_paths).content_hash() != toy.content_hash()
    # R1: the base spec enters by content — a copy elsewhere hashes alike, an edit does not
    copy = tmp_path / "base.yaml"
    copy.write_text(REFERENCE_SPEC.read_text())
    moved_base = dict(data, calibration=dict(data["calibration"], base_spec=str(copy)))
    assert bt.BacktestConfig.from_mapping(moved_base).content_hash() == toy.content_hash()
    copy.write_text(
        REFERENCE_SPEC.read_text().replace("bandwidth_factor: 1.5", "bandwidth_factor: 1.2")
    )
    assert bt.BacktestConfig.from_mapping(moved_base).content_hash() != toy.content_hash()
    fit_changed = dict(
        data, stability=dict(data["stability"], fit=dict(data["stability"]["fit"], k2=0.25))
    )
    assert bt.BacktestConfig.from_mapping(fit_changed).content_hash() != toy.content_hash()
    # the marking fit is named (SPEC §15 Part 3): the toy keeps the M7 fit, the runs the desk's;
    # the M7 fit hashes as before the key existed (named or not), so no stored store moves
    unnamed = dict(data, marking={k: v for k, v in data["marking"].items() if k != "fit"})
    assert data["marking"]["fit"] == "m7" and "fit" not in toy.to_mapping()["marking"]
    assert bt.BacktestConfig.from_mapping(unnamed).content_hash() == toy.content_hash()
    desk = bt.BacktestConfig.from_mapping(dict(data, marking=dict(data["marking"], fit="desk")))
    assert desk.content_hash() != toy.content_hash()
    assert desk.content()["marking"]["resolved"]["step0"] == "sabrw"
    assert toy.fit_config() == BreakEvenFitConfig(skew_eps=0.10)
    for cfg_ in (poc, full):
        assert cfg_.fit_config() == fit_preset("desk", skew_eps=0.10)
    for cfg_ in (poc, full, toy):
        assert cfg_.missing_close == "fail"
        assert cfg_.stability_fit_config().skew_mode == "soft"
    # the study configs point at the two backtest configs and share their seed
    for study, cfg in ((STUDY_FULL, poc), (STUDY_FAST, toy)):
        sc = runner.load_study_config(study)
        assert sc.runner == "volsto.studies.backtest" and sc.grid is None
        assert bt.load_backtest_config(ROOT / sc.params["backtest"]).name == cfg.name
        assert sc.seeds["pricing"] == cfg.section("pricing")["seed"]
    assert runner.load_study_config(STUDY_FAST).mode == "fast"
    # the full study's store is the PoC config's paths.out under the outputs root
    assert (ROOT / "outputs" / runner.load_study_config(STUDY_FULL).params["store"]) == poc.path(
        "out"
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d.update(extra=1), "unknown keys"),
        (lambda d: d.pop("ssr"), "missing keys"),
        (lambda d: d["marking"].update(stage3=True), "stage3"),
        (lambda d: d["marking"].update(fit="house"), "marking.fit"),
        (lambda d: d["attribution"].update(mode="recalibrate"), "sticky_leverage"),
        (lambda d: d["book"]["fixed"][0].update(maturity=0.3), "trading days"),
        (lambda d: d["book"]["fixed"][0].update(observations=4), "do not divide"),
        (lambda d: d["book"]["fixed"][2].update(vol_ko=0.3), "unknown keys"),
        (lambda d: d["book"]["fixed"][4].update(strike="atm"), "expected a number"),
        (lambda d: d["book"]["fixed"][0].update(maturity=2.0), "beyond the calibration horizon"),
        (lambda d: d["book"]["rolling"].update(every="none"), "every: none"),
        (lambda d: d["ssr"].update(window=10), "at least 20"),
        (lambda d: d["dates"].update(start="2022-13-01"), "YYYY-MM-DD"),
        (lambda d: d["surface"].update(essvi=False), "calendar_repair"),
        (lambda d: d["data"].pop("missing_close"), "missing keys"),
        (lambda d: d["data"].update(missing_close="interpolate"), "must be one of"),
        (lambda d: d["stability"].pop("fit"), "missing keys"),
        (lambda d: d["stability"]["fit"].pop("k2"), "no silent default"),
        (lambda d: d["stability"]["fit"].update(k3=1.0), "unknown BreakEvenFitConfig keys"),
        (lambda d: d["stability"]["fit"].update(skew_mode="sideways"), "skew_mode"),
    ],
)
def test_config_is_strict(mutate: Any, message: str) -> None:
    data = raw_config()
    mutate(data)
    with pytest.raises(ConfigError, match=message):
        bt.BacktestConfig.from_mapping(data)


def test_duplicate_yaml_key_is_refused(tmp_path: Path) -> None:
    text = TOY_CONFIG.read_text().replace("name: hdn_2022h2_toy", "name: a\nname: b")
    p = tmp_path / "dup.yaml"
    p.write_text(text)
    with pytest.raises(ConfigError, match="duplicate key"):
        bt.load_backtest_config(p)


def test_three_year_term_sheets_are_the_m6_headline_products() -> None:
    poc = bt.load_backtest_config(POC_CONFIG)
    spec = load_config_yaml(REFERENCE_SPEC, CalibrationSpec)
    state = RiskState(spec)
    surface = surface_of(state)
    disc = surface.forward_curve.rate_curve
    head = headline_products(disc, 100.0)
    by_kind = {t.kind: t for t in poc.fixed}
    auto = bt.build_product(by_kind["autocall"], 100.0, surface, disc).product
    phoe = bt.build_product(by_kind["phoenix"], 100.0, surface, disc).product
    assert product_key(auto) == product_key(head["autocall 3y"])
    assert product_key(phoe) == product_key(head["phoenix 3y"])
    # the variance products are struck on the surface's strip with vega notional 1
    var = bt.build_product(by_kind["var_swap"], 100.0, surface, disc)
    assert isinstance(var.product, VarianceSwap)
    assert var.product.strike == pytest.approx(var.strike**2, rel=1e-14)
    assert var.product.notional == pytest.approx(1.0 / (2.0 * var.strike), rel=1e-14)


def test_only_dates_takes_dates_and_ranges() -> None:
    cal = ["2022-07-27", "2022-07-28", "2022-07-29", "2022-08-01", "2022-08-02"]
    only = bt._split_dates(["2022-07-28..2022-07-31,2022-08-02", "2022-07-27"])
    assert only == ["2022-07-28..2022-07-31", "2022-08-02", "2022-07-27"]
    assert bt.expand_dates(cal, only) == ["2022-07-27", "2022-07-28", "2022-07-29", "2022-08-02"]
    assert bt.expand_dates(cal, ["2022-07-01..2022-12-31"]) == cal
    with pytest.raises(ConfigError, match="no calendar date"):
        bt.expand_dates(cal, ["2022-07-30..2022-07-31"])
    with pytest.raises(ConfigError, match="not calendar dates"):
        bt.expand_dates(cal, ["2022-07-30"])
    with pytest.raises(ConfigError, match="ends before it starts"):
        bt._split_dates(["2022-08-02..2022-07-27"])
    with pytest.raises(ConfigError, match="YYYY-MM-DD"):
        bt._split_dates(["2022-07-27..soon"])


def test_desk_marking_reads_the_snapshot_sabrw_fits(
    build_copy: BacktestBuild, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``marking.fit: desk`` (SPEC §15 Part 3): a date's marking fit gets the desk preset at the
    config's ``skew_eps`` and step 0's source built from the SABRW fits of the snapshot bytes the
    date verified — the fits the loader reads from that file, at the surface's ATM level; a
    parse without the section fails the date instead of falling back to the surface."""
    from volsto.market.loaders import load_step0_source

    b = build_copy
    raw = cfg_at(b).to_mapping()
    raw["marking"]["fit"] = "desk"
    cfg = bt.BacktestConfig.from_mapping(raw, source=str(TOY_CONFIG)).with_paths(
        out=b.store_root, cache=b.cache_root, snapshots=b.snapshots_root
    )
    seen: dict[str, Any] = {}

    class FitCalledError(Exception):
        pass

    def fake(surface: Any, c: Any, *, ssr_target: float, step0: Any = None) -> Any:
        seen.update(surface=surface, cfg=c, step0=step0, ssr=ssr_target)
        raise FitCalledError

    monkeypatch.setattr(bt, "fit_2f_marking", fake)
    date = _backtest_build.TOY_BACKTEST_DATES[0]
    run = bt.BacktestRun(cfg, allow_calibrate=False)
    with pytest.raises(FitCalledError):
        run._mark(date)
    assert seen["cfg"] == fit_preset("desk", skew_eps=0.10) and seen["ssr"] == 1.0
    want = load_step0_source(run.snapshot_path(date), seen["surface"])
    assert seen["step0"].label == want.label
    for T in (0.25, 1.0, 2.0):
        assert seen["step0"].triplet(T) == want.triplet(T)
    # a bound snapshot cannot lose its section (an edited file is unbound and re-imported, and
    # the importer always writes it); the guard behind it fails the date, never falls back
    monkeypatch.setattr(bt, "sabrw_fits_from_config", lambda raw: None)
    with pytest.raises(bt.DateFailure, match="no sabrw section"):
        bt.BacktestRun(cfg, allow_calibrate=False)._mark(date)


def test_shards_split_the_only_dates_selection(tmp_path: Path) -> None:
    """``--shard i/n`` cuts the ``--only-dates`` selection (before ``--resume``), not the
    whole calendar: two shards of a sub-window are its two halves."""
    data = raw_config()
    data["dates"] = {"start": "2022-07-01", "end": "2022-08-31"}
    data["paths"] = {k: str(tmp_path / k) for k in ("out", "cache", "snapshots")}
    cfg = bt.BacktestConfig.from_mapping(data)
    absent = _backtest_build._backtest_absent()
    if absent:
        pytest.skip(absent)
    run = bt.BacktestRun(cfg, allow_calibrate=False)
    assert len(run.calendar) >= 30, run.calendar
    only = bt._split_dates(["2022-07-05..2022-07-15"])
    window = bt.selection_of(run, only)
    assert window[0] == "2022-07-05" and window[-1] == "2022-07-15"
    parts = [
        bt.select_dates(run, shard=(i, 2), only=only, resume=True, limit=None)[0] for i in (1, 2)
    ]
    assert parts[0] + parts[1] == window
    assert abs(len(parts[0]) - len(parts[1])) <= 1
    assert bt.select_dates(run, shard=(1, 2), only=only, resume=False, limit=2)[0] == window[:2]


def test_xi0_memo_is_exact_and_keys_are_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ξ₀ memo of ``build_market`` on a perturbed eSSVI state: the curve with the memo on is
    bit-identical to the one computed with it off; a parameter or particle bump reuses the strip
    (the same object); a spot, rate or perturbation change does not; the LRU bound holds; the
    leverage-cache keys do not move."""
    from volsto.calibration import cache as cache_mod
    from volsto.config import SurfacePerturbation
    from volsto.market.loaders import snapshot_spec

    memo = cache_mod._Xi0Memo(size=3)
    monkeypatch.setattr(cache_mod, "XI0_MEMO", memo)
    base = load_config_yaml(REFERENCE_SPEC, CalibrationSpec)
    assert cache_mod.spec_key(base) == RECORDED_REFERENCE_2F_KEY
    spec = snapshot_spec(base, ESSVI_SNAPSHOT)
    assert spec.surface.rhos is not None  # eSSVI
    spec = dataclasses.replace(
        spec,
        perturbation=SurfacePerturbation(
            "skew_tent", {"pillars": [0.25, 0.5, 1.0], "index": 1, "slope": 0.01}
        ),
    )
    key_before = cache_mod.spec_key(spec)
    grid = np.linspace(0.0, 6.0, 6001)

    def curve(s: CalibrationSpec) -> Any:
        return cache_mod.build_market(s)[2].xi0

    memo.enabled = False
    off = curve(spec)
    assert memo.info()["entries"] == 0
    memo.enabled = True
    on = curve(spec)
    for f in ("maturities",):
        assert np.array_equal(getattr(off, f), getattr(on, f))
    assert np.array_equal(off.xi0(grid), on.xi0(grid))
    assert np.array_equal(off.total_variance(grid), on.total_variance(grid))
    assert memo.info() == {"enabled": True, "size": 3, "entries": 1, "hits": 0, "misses": 1}
    # the model parameters and the particle count do not enter the strip: a hit, same object
    bumped = dataclasses.replace(spec, model=spec.model.replace(nu=spec.model.nu * 1.05))
    more = dataclasses.replace(
        spec, particle=dataclasses.replace(spec.particle, n_particles=spec.particle.n_particles * 2)
    )
    assert curve(bumped) is on and curve(more) is on
    assert memo.info()["hits"] == 2
    # the spot, the rates and the perturbation do: misses, each equal to a fresh strip
    moved = dataclasses.replace(
        spec, market=dataclasses.replace(spec.market, spot=spec.market.spot * 1.01)
    )
    rates = RiskState(spec).with_rate_shift(dr=1e-4).spec
    other = dataclasses.replace(spec, perturbation=None)
    for s in (moved, rates, other):
        c = curve(s)
        assert c is not on
        memo.enabled = False
        fresh = curve(s)
        memo.enabled = True
        assert np.array_equal(c.xi0(grid), fresh.xi0(grid))
    info = memo.info()
    assert info["misses"] == 4 and info["entries"] == 3  # the LRU bound evicted the first
    t_max = min(spec.surface.max_maturity, max(spec.particle.horizon + 1.0, 5.0))
    assert memo.key(spec, t_max) not in memo._curves
    assert memo.key(moved, t_max) in memo._curves
    # keys are untouched by the memo; a payload that is not plain JSON data is not memoised
    assert cache_mod.spec_key(spec) == key_before
    assert cache_mod.spec_key(base) == RECORDED_REFERENCE_2F_KEY
    odd = dataclasses.replace(spec, perturbation=SurfacePerturbation("parallel", {"size": {1, 2}}))
    assert memo.key(odd, 3.0) is None
    # D11: a float32 surface config computes as its float64 twin (equal spec keys, one
    # computation, one entry); a non-string payload key or a float32 in the payload is not memoised
    v32 = tuple(np.float32(v) for v in spec.surface.atm_vols)
    a32 = dataclasses.replace(spec, surface=dataclasses.replace(spec.surface, atm_vols=v32))
    a64 = dataclasses.replace(
        spec, surface=dataclasses.replace(spec.surface, atm_vols=tuple(float(v) for v in v32))
    )
    assert cache_mod.spec_key(a32) == cache_mod.spec_key(a64)
    assert memo.key(a32, t_max) == memo.key(a64, t_max)
    memo.enabled = False
    c32, c64 = curve(a32), curve(a64)
    memo.enabled = True
    assert np.array_equal(c32.xi0(grid), c64.xi0(grid))
    assert np.array_equal(curve(a32).xi0(grid), c32.xi0(grid))
    for params in ({"size": 0.01, "meta": {1: "x"}}, {"size": np.float32(0.01)}):
        weird = dataclasses.replace(spec, perturbation=SurfacePerturbation("parallel", params))
        assert memo.key(weird, t_max) is None
    plain = dataclasses.replace(
        spec, perturbation=SurfacePerturbation("parallel", {"size": 0.01, "meta": {"1": "x"}})
    )
    assert memo.key(plain, t_max) is not None


def test_risk_package_exports_the_attribution_engines() -> None:
    """``from volsto.risk import *`` works (a stray ``"["`` in ``__all__`` broke it before M10
    Part 3) and the attribution's engines are public with their old names kept."""
    import volsto.risk as risk
    from volsto.risk import attribution

    missing = [n for n in risk.__all__ if not hasattr(risk, n)]
    assert missing == []
    namespace: dict[str, Any] = {}
    exec("from volsto.risk import *", namespace)
    assert namespace["FrozenLeverageEngine"] is attribution.FrozenLeverageEngine
    assert namespace["DryRunEngine"] is attribution.DryRunEngine
    assert namespace["DryRunBuilder"] is attribution.DryRunBuilder
    assert attribution._FrozenLeverageEngine is attribution.FrozenLeverageEngine
    assert attribution._DryRunEngine is attribution.DryRunEngine
    assert bt.FrozenLeverageEngine is attribution.FrozenLeverageEngine


def test_contiguous_shards_and_rolling_inceptions() -> None:
    dates = [f"2022-07-{d:02d}" for d in range(1, 30)] + ["2022-08-01", "2022-08-02"]
    for n in (1, 2, 3, 4, 8):
        blocks = [bt.shard_block(dates, i, n) for i in range(1, n + 1)]
        assert [d for b in blocks for d in b] == dates  # union, in order, disjoint
        assert max(map(len, blocks)) - min(map(len, blocks)) <= 1
    with pytest.raises(ValueError, match="1 <= i <= n"):
        bt.shard_block(dates, 3, 2)
    toy = bt.load_backtest_config(TOY_CONFIG)
    assert bt.rolling_inceptions(toy, dates) == ["2022-07-01", "2022-08-01"]


def test_start_greeks_reproduce_the_ladder_buckets_without_pricing() -> None:
    """On a Black–Scholes builder (every mode accepted, no leverage): the Greeks read back after
    ``explain`` cost no pricing, and each Greek × its move reproduces the attribution bucket."""
    spec0 = load_config_yaml(REFERENCE_SPEC, CalibrationSpec)
    s = spec0.surface
    surf1 = dataclasses.replace(
        s, atm_vols=tuple(v + 0.004 * (1 + i % 2) for i, v in enumerate(s.atm_vols)), rho=-0.66
    )
    market1 = MarketConfig(101.0, CurveConfig((1.0,), (0.0206,)), CurveConfig((1.0,), (0.0097,)))
    spec1 = dataclasses.replace(
        spec0, market=market1, surface=surf1, model=spec0.model.replace(nu=1.8, k1=5.5)
    )
    s0, s1 = RiskState(spec0, None, "d0"), RiskState(spec1, None, "d1")
    d0, d1 = dt.date(2022, 7, 27), dt.date(2022, 7, 28)
    disc = DiscountCurve.flat(0.02)
    product = VarianceSwap.daily(0.5, 0.04, disc, notional=2.5)
    hist = RealisedHistory(d0, (d0, d1), (100.0, 101.0))
    p1 = replay(product, hist, d1).result
    pt = replay(product, RealisedHistory(d0, (d0,), (100.0,)).extended(d1, 100.0), d1).result
    pillars = (0.25, 0.5, 1.0)
    prev = SimpleNamespace(state=s0, spot=100.0, spec=spec0)
    cur = SimpleNamespace(state=s1, spot=101.0, spec=spec1)
    sim = SimConfig(n_paths=2000, chunk_size=2000, dt_max=1.0 / 52.0, seed=7)
    for detail in ("ladders", "parallel"):
        engine = RiskEngine(BSBuilder(s0, 0.5), sim, max_halvings=6)
        ex = explain(
            engine,
            product,
            s0,
            s1,
            dt=bt.TRADING_DT,
            detail=detail,
            pillars=pillars,
            size=0.01,
            mode="sticky_leverage",
            product_1=p1,
            product_theta=pt,
        )
        n0 = engine.n_pricings
        g, err, ladders = bt.start_greeks(engine, ex, product, pt, prev, cur, pillars, 0.01, detail)
        assert engine.n_pricings == n0, "the Greeks must come from the memo"
        buckets = ex.buckets()
        assert sum(buckets.values()) == pytest.approx(ex.total.value, abs=1e-14)
        assert g["delta"].value * 1.0 == pytest.approx(buckets["spot.delta"], rel=1e-12, abs=1e-15)
        assert 0.5 * g["gamma"].value == pytest.approx(buckets["spot.gamma"], rel=1e-12, abs=1e-15)
        assert g["rho"].value == buckets["rates.rho"] and g["repo"].value == buckets["rates.repo"]
        assert g["theta_decay"].value * bt.TRADING_DT == pytest.approx(buckets["time.decay"])
        for name in ("nu", "k1"):
            dp = getattr(spec1.model, name) - getattr(spec0.model, name)
            assert g[f"d_{name}"].value * dp == pytest.approx(buckets[f"params.{name}"], abs=1e-15)
        moves = bt.ladder_moves(surface_of(s0), surface_of(s1), pillars)
        if detail == "ladders":
            for name, key in (
                ("surface.vega_T", "atm"),
                ("surface.skew_T", "skew"),
                ("surface.curvature_T", "fly"),
            ):
                lad = ladders[name]
                recon = sum(v * m / 0.01 for v, m in zip(lad["value"], moves[key], strict=True))
                assert recon == pytest.approx(buckets[name], rel=1e-12, abs=1e-15)
                assert lad["move"] == pytest.approx(list(moves[key]))
        else:
            recon = g["vega"].value * float(np.mean(moves["atm"])) / 0.01
            assert recon == pytest.approx(buckets["surface.parallel_vega"], rel=1e-12)
        assert err and all(math.isfinite(v) and v >= 0 for v in err.values())


def test_rates_bucket_follows_a_non_parallel_curve_move() -> None:
    """D1: a curve move whose mean zero rate does not change (short end up, long end down, on
    another pillar grid) moves a 3-month call; the pre-M10 mean-shift reading explains nothing,
    the directional sensitivity explains the step to its convexity."""
    spec0 = load_config_yaml(REFERENCE_SPEC, CalibrationSpec)
    m0 = MarketConfig(
        100.0,
        CurveConfig((0.25, 0.5, 1.0, 2.0), (0.02, 0.02, 0.02, 0.02)),
        CurveConfig((0.25, 1.0), (0.01, 0.01)),
    )
    m1 = MarketConfig(
        100.0,
        CurveConfig((0.1, 0.5, 1.0, 3.0), (0.05, 0.03, 0.01, -0.01)),  # mean 0.02: unchanged
        CurveConfig((0.08, 0.25, 1.0), (0.04, 0.0, -0.01)),  # mean 0.01: unchanged
    )
    s0 = RiskState(dataclasses.replace(spec0, market=m0), None, "d0")
    s1 = RiskState(dataclasses.replace(spec0, market=m1), None, "d1")
    assert np.mean(m1.rate_curve.rates) == pytest.approx(np.mean(m0.rate_curve.rates))
    assert np.mean(m1.dividend_curve.rates) == pytest.approx(np.mean(m0.dividend_curve.rates))
    from volsto.products.vanilla import EuropeanOption
    from volsto.risk.attribution import blend_curves, curve_move_sensitivities
    from volsto.risk.greeks import rate_sensitivities

    # the blend is exact: h = 0 reproduces curve 0 and h = 1 curve 1 on every time
    grid = np.linspace(0.0, 5.0, 501)
    for c0, c1 in ((m0.rate_curve, m1.rate_curve), (m0.dividend_curve, m1.dividend_curve)):
        for h, ref in ((0.0, c0), (1.0, c1)):
            got = DiscountCurve.from_config(blend_curves(c0, c1, h)).log_df(grid)
            np.testing.assert_allclose(got, DiscountCurve.from_config(ref).log_df(grid), atol=1e-15)
    product = EuropeanOption(100.0, 0.25, "call", DiscountCurve.flat(0.02))
    sim = SimConfig(n_paths=20_000, chunk_size=20_000, dt_max=1.0 / 52.0, seed=7)
    engine = RiskEngine(BSBuilder(s0, 0.25), sim)
    ex = explain(engine, product, s0, s1, mode="sticky_leverage")
    step = next(x for x in ex.steps if x.name == "rates")
    old = rate_sensitivities(engine, product, s0)
    mean_shift = old["rho"].value * 0.0 / 1e-4 + old["repo_delta"].value * 0.0 / 1e-4
    new = curve_move_sensitivities(engine, product, s0, s1)
    assert abs(step.actual) > 20 * step.actual_stderr + 1e-3  # a real move
    assert mean_shift == 0.0  # the pre-M10 explanation
    assert step.explained == new["rho"].value + new["repo"].value
    assert abs(step.actual - step.explained) < 0.05 * abs(step.actual)
    assert abs(step.actual - step.explained) < abs(step.actual - mean_shift) / 10


def test_parameter_jitter_is_not_a_move() -> None:
    """D8: a fit on a bound jitters by ~4e-16; that is no parameter move (no sensitivity, two
    pricings saved), while a real move is explained."""
    from volsto.risk.attribution import attribution_cost, param_moved

    assert not param_moved(3.5, 3.5 + 4.4e-16) and param_moved(3.5, 3.5 + 1e-9)
    assert not param_moved(0.0, 5e-13) and param_moved(0.0, 2e-12)
    spec0 = load_config_yaml(REFERENCE_SPEC, CalibrationSpec)
    s0 = RiskState(spec0, None, "d0")
    jitter = RiskState(
        dataclasses.replace(spec0, model=spec0.model.replace(nu=spec0.model.nu + 4.4e-16)),
        None,
        "j",
    )
    real = RiskState(dataclasses.replace(spec0, model=spec0.model.replace(nu=1.8)), None, "r")
    sim = SimConfig(n_paths=2000, chunk_size=2000, dt_max=1.0 / 52.0, seed=7)
    product = VarianceSwap.daily(0.5, 0.04, DiscountCurve.flat(0.02))
    engine = RiskEngine(BSBuilder(s0, 0.5), sim)
    ex = explain(engine, product, s0, jitter, mode="sticky_leverage")
    assert "params.nu" not in ex.buckets()
    c_jit = attribution_cost(product, s0, jitter, sim, mode="sticky_leverage")
    c_real = attribution_cost(product, s0, real, sim, mode="sticky_leverage")
    assert c_real.pricings - c_jit.pricings == 2


def test_explain_total_and_residual_stderrs_are_paired() -> None:
    """Round-2 (5): the P&L's standard error is the direct paired one of V(d) - V(d-1) — both
    prices kept, no extra pricing — and the residual's is the per-path residual's; across 24
    seeds both match the dispersion of the estimates (a 3-month call on a Black-Scholes builder,
    a day of spot, curve, surface and parameter moves)."""
    from volsto.products.vanilla import EuropeanOption

    spec0 = load_config_yaml(REFERENCE_SPEC, CalibrationSpec)
    market1 = MarketConfig(101.0, CurveConfig((1.0,), (0.0206,)), CurveConfig((1.0,), (0.0097,)))
    surf1 = dataclasses.replace(
        spec0.surface, atm_vols=tuple(v + 0.004 for v in spec0.surface.atm_vols)
    )
    spec1 = dataclasses.replace(
        spec0, market=market1, surface=surf1, model=spec0.model.replace(nu=1.8)
    )
    s0, s1 = RiskState(spec0, None, "d0"), RiskState(spec1, None, "d1")
    call = EuropeanOption(100.0, 0.5, "call", DiscountCurve.flat(0.02))
    totals, residuals, total_se, resid_se, rss = [], [], [], [], []
    for seed in range(24):
        sim = SimConfig(n_paths=2000, chunk_size=2000, dt_max=1.0 / 52.0, seed=100 + seed)
        engine = RiskEngine(BSBuilder(s0, 0.5), sim)
        ex = explain(engine, call, s0, s1, dt=bt.TRADING_DT, mode="sticky_leverage")
        n0 = engine.n_pricings
        direct = engine.paired(
            "pnl",
            [(call.aged(bt.TRADING_DT), s1, "recalibrate", 1.0), (call, s0, "recalibrate", -1.0)],
            unit="price",
            size=0.0,
            scheme="revaluation",
        )
        assert engine.n_pricings == n0, "both endpoint prices are in the memo"
        assert ex.total.stderr == direct.stderr
        assert ex.total.value == pytest.approx(direct.value, abs=1e-14)
        assert ex.residual_paired and math.isfinite(ex.residual_stderr)
        assert ex.as_frame()["residual_stderr"].iloc[-1] == ex.residual_stderr
        # the root-sum-square reading of the residual (total and buckets) is far too wide here
        assert ex.residual_stderr < 0.1 * ex.total.stderr
        totals.append(ex.total.value)
        residuals.append(ex.residual)
        total_se.append(ex.total.stderr)
        resid_se.append(ex.residual_stderr)
        rss.append(math.sqrt(sum(s.actual_stderr**2 for s in ex.steps)))
    sd_total, sd_resid = float(np.std(totals, ddof=1)), float(np.std(residuals, ddof=1))
    print(
        f"paired total stderr {np.mean(total_se):.3g} (seed dispersion {sd_total:.3g}); RSS of the "
        f"steps {np.mean(rss):.3g}; residual stderr {np.mean(resid_se):.3g} (dispersion "
        f"{sd_resid:.3g})"
    )
    # 24 seeds: the sample standard deviation is within ~[0.56, 1.44] of the truth at 3 sigma
    assert 0.55 < sd_total / np.mean(total_se) < 1.5
    assert 0.55 < sd_resid / np.mean(resid_se) < 1.5


def test_short_notes_pay_per_annum_coupons() -> None:
    """D9: the 6 % coupons are per annum — a half-year note with two observations pays
    1.5 % and 3 % (autocall, growing) or 1.5 % per quarter (Phoenix); annual notes keep the M6
    float coupon (their product keys are the headline's, tested above)."""
    spec = load_config_yaml(REFERENCE_SPEC, CalibrationSpec)
    surface = surface_of(RiskState(spec))
    disc = surface.forward_curve.rate_curve
    ac = bt.build_product(bt.TradeSpec("a", "autocall", 0.5, observations=2), 100.0, surface, disc)
    ph = bt.build_product(bt.TradeSpec("p", "phoenix", 0.5, observations=2), 100.0, surface, disc)
    np.testing.assert_allclose(ac.product.coupon_schedule, [0.015, 0.03], rtol=1e-15)
    np.testing.assert_allclose(ph.product.coupon_schedule, [0.015, 0.015], rtol=1e-15)
    a3 = bt.build_product(bt.TradeSpec("a", "autocall", 3.0, observations=3), 100.0, surface, disc)
    p3 = bt.build_product(bt.TradeSpec("p", "phoenix", 3.0, observations=3), 100.0, surface, disc)
    np.testing.assert_allclose(a3.product.coupon_schedule, [0.06, 0.12, 0.18], rtol=1e-15)
    np.testing.assert_allclose(p3.product.coupon_schedule, [0.06, 0.06, 0.06], rtol=1e-15)
    a6 = bt.build_product(bt.TradeSpec("a", "autocall", 3.0, observations=6), 100.0, surface, disc)
    np.testing.assert_allclose(a6.product.coupon_schedule, 0.06 * np.arange(1, 7) / 2, rtol=1e-14)


def _rows_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = {c: np.nan for c in bt.ROW_COLUMNS}
        for b_ in bt.BUCKETS:
            row[bt.bucket_column(b_)] = 0.0
            row[bt.bucket_column(b_) + "_stderr"] = 0.0
        row.update(unit="% notional", scale=100.0, reason="", pnl_note="", realised_json="{}")
        row.update(rec)
        rows.append(row)
    return pd.DataFrame(rows, columns=list(bt.ROW_COLUMNS))


def test_stage2_tables_with_twin_term_sheets_and_desk_sign() -> None:
    """D2: two trades with the same book, kind, maturity and inception (a KO-variance barrier
    pair) have their own rows everywhere; D10: the monthly table carries every bucket and a
    zero check; (a): P&L is presented in the desk sign."""
    common = dict(book="fixed", kind="ko_var", inception="2022-10-27", maturity=0.5, strike=0.3)
    recs = []
    for tid in ("fixed:ko_var_lo", "fixed:ko_var_6m"):
        recs.append(
            dict(
                common,
                date="2022-10-27",
                trade_id=tid,
                status="inception",
                age=0,
                value=0.01,
                value_stderr=0.001,
                pnl_method="none",
            )
        )
        recs.append(
            dict(
                common,
                date="2022-10-28",
                trade_id=tid,
                status="live",
                age=1,
                value=0.02,
                value_stderr=0.001,
                pnl=0.012,
                pnl_stderr=0.002,
                pnl_method="attributed",
                price_0=0.01,
                **{
                    bt.bucket_column("spot.delta"): 0.004,
                    bt.bucket_column("rates.repo"): 0.003,
                    bt.bucket_column("time.decay"): -0.001,
                    bt.bucket_column("cash_flows"): 0.002,
                    bt.bucket_column("residual"): 0.004,
                },
            )
        )
    rows = _rows_frame(recs)
    b = ResultsBuilder()
    bt._trade_results(b, rows)
    bt._pnl_results(b, rows)
    res = b.build()
    assert len(res.rows("inception")) == 2
    v, _ = res.value("pnl_trade", "fixed:ko_var_lo", "pnl")
    assert v == pytest.approx(-100.0 * 0.012)
    book_row = bt.book_row("% notional")  # the unit _rows_frame gives both
    book, _ = res.value("pnl_trade", book_row, "pnl")
    assert book == pytest.approx(-100.0 * 0.024)
    assert res.record("pnl_trade", book_row, "pnl")["unit"] == "% notional"
    month = "2022-10 (% notional)"
    cols = res.columns("pnl_month")
    assert [g for g, _ in bt.BUCKET_GROUPS] == [c for c in cols if c not in ("total", "check")]
    parts = sum(res.value("pnl_month", month, g)[0] for g, _ in bt.BUCKET_GROUPS)
    assert parts == pytest.approx(res.value("pnl_month", month, "total")[0], abs=1e-12)
    assert res.value("pnl_month", month, "check")[0] == pytest.approx(0.0, abs=1e-12)
    specs = {t.name: t for t in bt.tables(res)}
    assert specs["pnl_trade_pct_notional"].columns[0].header == "desk P&L"
    assert specs["pnl_trade_pct_notional"].columns[0].unit == "% notional"
    # one date each: the cumulative errors are the rows' own; the book's is labelled
    assert res.record("pnl_trade", "fixed:ko_var_lo", "pnl")["note"] == ""
    assert bt.SE_RSS_TRADES in res.record("pnl_trade", book_row, "pnl")["note"]


def test_scaled_calibration_charge(tmp_path: Path) -> None:
    """D6: the calibration charge scales with particles x steps (and never goes below the
    small-calibration floor), prefers the manifest at the exact size, and a --no-calibrate
    projection charges none."""
    from volsto.calibration.cache import LeverageCache

    base = load_config_yaml(REFERENCE_SPEC, CalibrationSpec)
    empty = LeverageCache(tmp_path / "empty")
    s1, why = bt.scaled_calibration_s(empty, base, 800_000, 3.0)
    assert s1 == pytest.approx(bt.CALIBRATION_REFERENCE_S) and "reference" in why
    s2, _ = bt.scaled_calibration_s(empty, base, 400_000, 1.0)
    ratio = (400_000 * bt.calibration_steps(base, 1.0)) / (
        800_000 * bt.calibration_steps(base, 3.0)
    )
    assert s2 == pytest.approx(bt.CALIBRATION_REFERENCE_S * ratio)
    assert bt.scaled_calibration_s(empty, base, 20_000, 1.0)[0] == bt.CALIBRATION_MIN_S
    root = tmp_path / "cache"
    root.mkdir()
    pd.DataFrame(
        {
            "key": ["a", "b", "c"],
            "n_particles": [200_000, 200_000, 20_000],
            "horizon": [3.0, 3.0, 1.0],
            "wall_time": [30.0, 40.0, 5.0],
            "code_tag": ["m6", "m6", "m6"],
        }
    ).to_parquet(root / "manifest.parquet")
    cache = LeverageCache(root)
    assert bt.scaled_calibration_s(cache, base, 20_000, 1.0)[0] == 5.0
    s3, why3 = bt.scaled_calibration_s(cache, base, 800_000, 3.0)
    assert s3 == pytest.approx(35.0 * 4.0) and "scaled" in why3


def test_fixture_failure_is_a_failure_not_a_skip(tmp_path: Path) -> None:
    """(d): only an absent HDN sample skips; a failed build fails its consumers."""
    kw: dict[str, Any] = dict(
        root=tmp_path,
        base=tmp_path,
        config_path=TOY_CONFIG,
        wall_s=0.0,
        stdout="",
        log_messages=(),
        built_by="test",
    )
    with pytest.raises(pytest.fail.Exception, match="boom"):
        BacktestBuild(return_code=-1, error="boom", **kw).require()
    with pytest.raises(pytest.fail.Exception, match="returned 1"):
        BacktestBuild(return_code=1, **kw).require()
    with pytest.raises(pytest.skip.Exception, match="absent"):
        BacktestBuild(return_code=-1, absent="HDN sample absent", **kw).require()
    ok = BacktestBuild(return_code=0, **kw)
    assert ok.require() is ok


def test_synthetic_history_gives_realised_ssr_and_historical_flags() -> None:
    """30 synthetic dates: the realised SSR is NaN with its reason on the first ``window`` dates
    and finite after; the historical-mode rolling fit runs and ``flag_unidentified`` reads it."""
    toy = bt.load_backtest_config(TOY_CONFIG)
    rng = np.random.default_rng(11)
    n = 30
    dates = list(pd.bdate_range("2022-07-01", periods=n).strftime("%Y-%m-%d"))
    dls = 0.015 * rng.standard_normal(n - 1)
    ls = 8.3 + np.concatenate([[0.0], np.cumsum(dls)])
    ps = np.asarray(DEFAULT_PILLARS)
    skew = -0.3 / np.sqrt(np.maximum(ps, 0.25))
    fits: dict[str, dict[str, Any]] = {}
    atm = 0.25 + 0.0 * ps
    for i, d in enumerate(dates):
        if i:
            atm = atm + 0.8 * skew * dls[i - 1] + 0.002 * rng.standard_normal(ps.size)
        fits[d] = {
            "history": {
                "T": ps.tolist(),
                "vs_vol": (atm + 0.02).tolist(),
                "atm_vol": atm.tolist(),
                "skew": skew.tolist(),
                "ln_spot": float(ls[i]),
            },
            "pillars": [0.25, 0.5, 1.0],
            "ssr_first_order": [1.1, 1.0, 0.95],
        }
    b = ResultsBuilder()
    bt._ssr_results(b, toy, dates, fits)
    bt._stability_hist(b, toy, dates, fits)
    res = b.build()
    window = int(toy.section("ssr")["window"])
    for i, d in enumerate(dates):
        rec = res.record("ssr_daily", f"{d}|0.25", "realised")
        if i < window:
            assert math.isnan(rec["value"]) and "needs 21 dates" in rec["note"]
        else:
            assert math.isfinite(rec["value"]) and rec["stderr"] > 0
            assert 0.3 < rec["value"] < 1.5  # the synthetic slope is 0.8 x skew
    v, se = res.value("ssr", "0.25y", "realised_last")
    assert math.isfinite(v) and se > 0
    assert res.value("ssr", "1y", "first_order_median")[0] == pytest.approx(0.95)
    assert res.rows("stability_hist") == list(PARAM_COLUMNS)
    assert (
        "historical-mode rolling fit"
        in res.record("stability_hist", "k1", "share_beyond_se")["note"]
    )
    assert len(res.rows("stability_daily")) == n - window


# --------------------------------------------------------------------------------------------
# the build (the one calibrating fixture of this module)
# --------------------------------------------------------------------------------------------


def test_toy_backtest_build(toy_backtest_build: BacktestBuild) -> None:  # noqa: F811
    build = toy_backtest_build.require()
    print(
        f"toy backtest build: {build.wall_s:.1f} s by {build.built_by}; calibrated "
        f"{build.info.get('calibrated')}"
    )
    assert sorted(build.info["calibrated"]) == list(build.dates)  # one calibration per date
    for d in build.dates:
        done = current_json(build.store_root, d, "done.json")
        assert done["status"] == "ok" and done["config_hash"]
        assert len(list((build.store_root / "dates" / d / "attempts").iterdir())) == 1
    assert build.info.get("shifted_key") and build.info.get("shifted_calibrated") is True
    assert "PROJECTED WALL CLOCK" in build.stdout
    assert "recalibrated: yes" in build.stdout
    projected = [ln for ln in build.stdout.splitlines() if "TOTAL (one process)" in ln]
    print(f"D6 toy projection {projected}; measured {build.wall_s:.1f} s")
    rows = store_rows(build.store_root)
    assert set(rows["date"]) == set(build.dates)
    toy = bt.load_backtest_config(TOY_CONFIG)
    n_fixed = len(toy.fixed)
    counts = rows.groupby("date").size().to_dict()
    # the fixed book on every date; the rolling var swap from 07-27, a second one from 08-01
    assert counts == {
        "2022-07-27": n_fixed + 1,
        "2022-07-28": n_fixed + 1,
        "2022-07-29": n_fixed + 1,
        "2022-08-01": n_fixed + 2,
        "2022-08-02": n_fixed + 2,
    }
    assert set(rows.loc[rows["date"] == "2022-07-27", "status"]) == {"inception"}
    assert (rows["value_stderr"] > 0).all()


# --------------------------------------------------------------------------------------------
# shards, resume, refusals, projection
# --------------------------------------------------------------------------------------------


def test_shards_union_equals_the_unsharded_run_and_resume_computes_nothing(
    build_copy: BacktestBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    b = build_copy
    out = tmp_path / "sharded"
    cache_before = tree_state(b.cache_root)
    cfg = bt.load_backtest_config(TOY_CONFIG).with_paths(
        out=out, cache=b.cache_root, snapshots=b.snapshots_root
    )
    for shard in ("2/2", "1/2"):  # the later block first: its dates wait for the earlier one
        code, text = run_cli(
            ["run", str(TOY_CONFIG), "--no-calibrate", "--shard", shard, *b.path_args(out=out)],
            capsys,
        )
        print(text[-400:])
        assert code == 0, text
        assert "recalibrated: no" in text
        if shard == "2/2":
            ledger = bt.Ledger.of(cfg)
            late = [ledger.verdict(d) for d in b.dates[3:]]
            assert [v.status for v in late] == ["pending", "pending"], late
            assert late[0].waits == (b.dates[0], b.dates[2]), late[0]
            code, text = run_cli(
                [
                    "run",
                    str(TOY_CONFIG),
                    "--no-calibrate",
                    "--resume",
                    "--shard",
                    "2/2",
                    *b.path_args(out=out),
                ],
                capsys,
            )
            assert code == 0 and "nothing to compute" in text, text
            # D7: pending dates are not called done
            assert "resume: 0 date(s) done and still matching, 2 pending" in text, text
            assert f"they wait for {b.dates[0]}, {b.dates[2]}, not in this selection" in text
    assert [bt.Ledger.of(cfg).verdict(d).status for d in b.dates] == ["done"] * 5
    assert tree_state(b.cache_root) == cache_before, "--no-calibrate wrote the cache"
    assert_same_rows(store_rows(out), store_rows(b.store_root))
    fits_a, fits_b = store_fits(out), store_fits(b.store_root)
    assert sorted(fits_a) == list(b.dates)
    for d in b.dates:
        for k in ("params", "breakeven", "history", "cache_key", "spot", "snapshot_sha256"):
            assert fits_a[d][k] == fits_b[d][k], (d, k)
        assert fits_a[d]["calibrated"] is False
    # a second run with --resume computes nothing and writes nothing
    before = tree_state(out)
    for extra in ([], ["--shard", "2/2"]):
        code, text = run_cli(
            ["run", str(TOY_CONFIG), "--no-calibrate", "--resume", *extra, *b.path_args(out=out)],
            capsys,
        )
        assert code == 0, text
        assert "nothing to compute" in text and "PROJECTED" not in text
    assert tree_state(out) == before


def test_changed_config_hash_is_refused(
    build_copy: BacktestBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    b = build_copy
    data = raw_config()
    data["pricing"]["n_paths"] = 2002
    cfg = write_config(tmp_path, data)
    before = tree_state(b.store_root)
    code, text = run_cli(["run", str(cfg), "--resume", "--no-calibrate", *b.path_args()], capsys)
    assert code == bt.EXIT_REFUSED, text
    assert "REFUSED" in text and "config hash" in text
    assert tree_state(b.store_root) == before
    # the status command reports the dates as stale under the new config
    code, text = run_cli(["status", str(cfg), *b.path_args()], capsys)
    assert code == 1 and "stale 5" in text


def test_missing_leverage_under_no_calibrate_prints_the_command(
    build_copy: BacktestBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    b = build_copy
    first = b.dates[0]
    key = current_json(b.store_root, first, "fit.json")["cache_key"]
    shutil.rmtree(b.cache_root / key)
    out = tmp_path / "fresh"
    refusals = guard.refusals()
    code, text = run_cli(
        ["run", str(TOY_CONFIG), "--no-calibrate", "--only-dates", first, *b.path_args(out=out)],
        capsys,
    )
    assert code == bt.EXIT_REFUSED, text
    assert "not in the cache" in text
    assert "volsto-backtest run" in text and "hdn_2022h2_toy.yaml" in text
    assert f"--cache {b.cache_root}" in text or "--cache " in text
    assert f"--only-dates {first}" in text
    assert guard.refusals() == refusals, "the missing leverage is found before any calibration"
    failure = current_json(out, first, "done.json")
    assert failure["status"] == "failed" and key[:16] in failure["error"]
    assert current(out, first).name.startswith("failed-")
    v = bt.Ledger.of(cfg_at(b, out=out)).verdict(first)
    assert v.status == "failed" and v.doc is not None and v.doc["error"] == failure["error"], v


def hdn_root(
    tmp_path: Path,
    dates: Sequence[str],
    *,
    corrupt: Sequence[str] = (),
    manifest_edit: Any = None,
) -> Path:
    """A private HDN data root: links to the sample's day files for ``dates`` (``corrupt`` ones
    replaced by junk) and a copy of its manifest (``manifest_edit(dict)`` applied)."""
    real = ROOT / "data" / "hdn_sample" / "options_sample_2022H2" / "day_by_date"
    root = tmp_path / "hdn" / "day_by_date"
    root.mkdir(parents=True)
    manifest = json.loads((real / "manifest.json").read_text())
    if manifest_edit is not None:
        manifest_edit(manifest)
    (root / "manifest.json").write_text(json.dumps(manifest))
    for d in dates:
        name = f"{d}_options.csv"
        if d in corrupt:
            (root / name).write_text("not,a,chain\n")
        else:
            (root / name).symlink_to(real / name)
    return root.parent


def test_missing_close_fail_names_the_fix_and_later_dates_proceed(
    build_copy: BacktestBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D7, ``data.missing_close: fail``: a corrupt day file fails its date with the fix in the
    message; the next date is marked and valued where it can be — the trades struck on the
    corrupt date are unpriced with that message and its P&L is missing — so it is committed
    ``incomplete``; the run exits 1 and lists both."""
    b = build_copy
    data = raw_config()
    data["data"]["root"] = str(hdn_root(tmp_path, b.dates[:2], corrupt=b.dates[:1]))
    cfg = write_config(tmp_path, data)
    outputs = tmp_path / "outputs"
    out = outputs / "failing"
    args = [*b.path_args(out=out)[:4], "--snapshots", str(out / "snapshots")]
    code, text = run_cli(["run", str(cfg), "--no-calibrate", *args], capsys)
    assert code == bt.EXIT_FAILED, text
    assert "0 date(s) ok, 1 incomplete, 1 failed" in text
    first = current_json(out, b.dates[0], "done.json")
    assert first["status"] == "failed"
    assert "import failed" in first["error"] and "exists, so the date is failed" in first["error"]
    assert "rerun with --resume" in first["error"]
    second = current_json(out, b.dates[1], "done.json")
    assert second["status"] == "incomplete" and "command" in second
    rows = pd.read_parquet(current(out, b.dates[1], "rows.parquet"))
    assert len(rows) == len(bt.load_backtest_config(cfg).fixed) + 1
    assert set(rows["status"]) == {"unpriced"}
    assert rows["reason"].str.contains("close of 2022-07-27 is unavailable").all()
    assert rows["value"].isna().all() and rows["pnl"].isna().all()
    code, text = run_cli(["status", str(cfg), "--out", str(out), *args[2:]], capsys)
    assert code == 1 and "failed 1" in text and "incomplete 1" in text, text
    # stage 2 reports both dates (LOW: a failed date counts its live trades, not 0 rows)
    # (no LaTeX check: the reasons quote the pytest temporary paths, which overflow a line)
    run = run_stage2(
        b, tmp_path / "study", outputs=outputs, backtest=cfg, store="failing", latex_check=False
    )
    assert run.exit_code == 0
    n_live = len(rows)
    assert run.results.value("pnl_missing", b.dates[0], "rows_without_pnl")[0] == n_live
    assert run.results.value("pnl_missing", b.dates[0], "failed")[0] == 1.0
    assert run.results.value("pnl_missing", b.dates[1], "rows_without_pnl")[0] == len(rows)
    md = (run.out_dir / "study.md").read_text()
    assert f"- {b.dates[0]}: failed, no rows stored ({n_live} live trade(s)" in md
    assert f"- {b.dates[1]}: {len(rows)} row(s) without P&L" in md
    assert "Every stored date carries its P&L." not in md


def test_missing_close_skip_date_drops_the_day_and_records_the_gap(
    build_copy: BacktestBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D7, ``data.missing_close: skip_date``: a day the vendor manifest lists but whose file is
    absent is dropped from the calendar before anything is computed (``done.json`` status
    skipped, its cause recorded); the next date is attributed from the date before the gap, its
    trades are one trading day old and list the gap.  Round 5 F: the skip stops applying when
    the day file comes back (its cause is re-derived)."""
    b = build_copy
    days = list(b.dates[:3])
    data = raw_config()
    root = hdn_root(
        tmp_path,
        [days[0], days[2]],
        manifest_edit=lambda m: m.update(trading_days_missing=[days[1]]),
    )
    data["data"]["root"] = str(root)
    data["data"]["missing_close"] = "skip_date"
    cfg = write_config(tmp_path, data)
    # LOW: the dry run finds the skip too, so its shard blocks are the run's
    dry_out = tmp_path / "dry"
    dry_args = ["--out", str(dry_out), *b.path_args()[2:4], "--snapshots", str(tmp_path / "ds")]
    code, text = run_cli(["dry-run", str(cfg), "--shard", "2/2", *dry_args], capsys)
    assert code == 0, text
    assert f"dropped 1 date(s) from the calendar: {days[1]}" in text and "not recorded" in text
    assert "1 of 2 calendar dates to compute" in text, text
    assert not (dry_out / "dates").exists()
    out = tmp_path / "skipping"
    code, text = run_cli(["run", str(cfg), "--no-calibrate", *b.path_args(out=out)], capsys)
    assert code == 0, text
    assert f"dropped 1 date(s) from the calendar: {days[1]}" in text
    assert "2 date(s) ok" in text
    skipped = current_json(out, days[1], "done.json")
    assert skipped["status"] == "skipped" and bt.SKIP_CAUSE in skipped["error"]
    assert skipped["record"]["cause"] == bt.SKIP_CAUSE
    # LOW: a date's gaps are the skips before it
    assert current_json(out, days[0], "done.json")["gaps"] == []
    assert current_json(out, days[2], "done.json")["gaps"] == [days[1]]
    rows = pd.read_parquet(current(out, days[2], "rows.parquet"))
    live = rows[rows["book"] == "fixed"]
    assert (live["age"] == 1).all() and (live["pnl_method"] == "attributed").all()
    assert (live["gaps"] == json.dumps([days[1]])).all()
    ref = store_rows(b.store_root)
    ref = ref[(ref["date"] == days[0]) & (ref["book"] == "fixed")].set_index("trade_id")
    got = live.set_index("trade_id")["price_0"]
    assert (got == ref.loc[got.index, "value"]).all()
    code, text = run_cli(["status", str(cfg), *b.path_args(out=out)], capsys)
    assert code == 0 and "skipped 1" in text and "done 2" in text, text
    code, text = run_cli(
        ["run", str(cfg), "--no-calibrate", "--resume", *b.path_args(out=out)], capsys
    )
    assert code == 0 and "nothing to compute" in text
    real = ROOT / "data" / "hdn_sample" / "options_sample_2022H2" / "day_by_date"
    name = f"{days[1]}_options.csv"
    (root / "day_by_date" / name).symlink_to(real / name)
    paths = dict(out=out, cache=b.cache_root, snapshots=b.snapshots_root)
    ledger = bt.Ledger.of(bt.load_backtest_config(cfg).with_paths(**paths))
    assert ledger.skips() == {} and days[1] in ledger.calendar()


def test_incomplete_date_waits_for_its_dependencies_then_is_recomputed(
    build_copy: BacktestBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D3 (R2): computed alone (``--only-dates``) while the previous date's leverage is missing
    under --no-calibrate, a date is committed ``incomplete`` (valuations stored, P&L missing,
    the reason and the command, exit 2).  While the dates it depends on are not stored it is
    ``pending``; once they are (the previous date with its leverage) it is stale — its record
    says the previous leverage was missing — and --resume recomputes exactly it, to the rows of
    the complete build."""
    b = build_copy
    first, prev, d = b.dates[0], b.dates[1], b.dates[2]
    key = current_json(b.store_root, prev, "fit.json")["cache_key"]
    stash = tmp_path / "stash"
    shutil.move(str(b.cache_root / key), stash)
    out = tmp_path / "window"
    code, text = run_cli(
        ["run", str(TOY_CONFIG), "--no-calibrate", "--only-dates", d, *b.path_args(out=out)],
        capsys,
    )
    assert code == bt.EXIT_REFUSED, text
    assert "INCOMPLETE" in text and f"previous date {prev} is unavailable" in text
    done = current_json(out, d, "done.json")
    assert done["status"] == "incomplete" and done["leverage_missing"] is True
    assert f"--only-dates {d}" in done["command"] and done["previous_cache_key"] is None
    rows = pd.read_parquet(current(out, d, "rows.parquet"))
    assert rows["value"].notna().all() and rows["pnl"].isna().all()
    assert rows["pnl_note"].str.contains("no P&L").all()
    cfg = cfg_at(b, out=out)
    v = bt.Ledger.of(cfg).verdict(d)
    assert v.status == "pending" and v.waits == (first, prev), v
    assert v.doc is not None and v.doc["status"] == "incomplete"
    shutil.move(str(stash), b.cache_root / key)
    code, text = run_cli(
        [
            "run",
            str(TOY_CONFIG),
            "--no-calibrate",
            "--only-dates",
            first,
            prev,
            *b.path_args(out=out),
        ],
        capsys,
    )
    assert code == 0, text
    v = bt.Ledger.of(cfg).verdict(d)
    assert v.status == "stale" and f"the leverage of its previous date {prev} is now present" in (
        v.reason
    ), v
    code, text = run_cli(
        [
            "run",
            str(TOY_CONFIG),
            "--no-calibrate",
            "--resume",
            "--only-dates",
            f"{first}..{d}",
            *b.path_args(out=out),
        ],
        capsys,
    )
    assert code == 0 and "2 skipped (resume)" in text and "1 date(s) ok" in text, text
    assert current_json(out, d, "done.json")["status"] == "ok"
    # the incomplete attempt stays on disk next to the current one
    labels = sorted(a.name.split("-")[0] for a in (out / "dates" / d / "attempts").iterdir())
    assert labels == ["incomplete", "ok"], labels
    ref = store_rows(b.store_root)
    assert_same_rows(
        store_rows(out), ref[ref["date"].isin([first, prev, d])].reset_index(drop=True)
    )


def test_changed_inputs_make_dates_stale(
    build_copy: BacktestBuild,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D4: R3 — a data root without 2022-07-28 (same config hash, same day files otherwise)
    makes that date's successors stale (their calendar prefix changed) and --resume would
    recompute exactly them; R3b — a manifest whose rates for 2022-07-29 moved by +50 bp makes
    that date and its successors stale, and the snapshot is re-imported with the new rates;
    R4 — a stored snapshot imported under another importer tag (its bytes intact) makes its
    date stale with the reason naming both tags, is no longer bound, and every date after it
    is stale through it (SPEC §13.1, 2026-09-22)."""
    b = build_copy
    toy = cfg_at(b)
    ledger0 = bt.Ledger.of(toy)
    assert [ledger0.verdict(d).status for d in b.dates] == ["done"] * 5
    # R3: a gap in the calendar
    gap = [d for d in b.dates if d != "2022-07-28"]
    data = raw_config()
    data["data"]["root"] = str(hdn_root(tmp_path / "gap", gap))
    cfg_gap = bt.BacktestConfig.from_mapping(data).with_paths(
        out=b.store_root, cache=b.cache_root, snapshots=b.snapshots_root
    )
    assert cfg_gap.content_hash() == toy.content_hash()
    run_gap = bt.BacktestRun(cfg_gap, allow_calibrate=False)
    status = {d: run_gap.ledger.verdict(d) for d in gap}
    assert (status["2022-07-27"].status, status["2022-07-27"].reason) == ("done", "")
    for d in gap[1:]:
        assert status[d].status == "stale" and "inputs changed" in status[d].reason, status[d]
    assert status[gap[1]].reason.startswith("its inputs changed")
    assert status[gap[2]].reason.startswith(f"depends on stale {gap[1]}: its inputs changed")
    todo, kept = bt.select_dates(run_gap, shard=None, only=None, resume=True, limit=None)
    assert todo == gap[1:] and kept == ["2022-07-27"]

    # R3b: the rates of one date moved in the manifest
    def shift(m: dict[str, Any]) -> None:
        m["rates"]["2022-07-29"] = [
            r + 0.5 if r is not None else r for r in m["rates"]["2022-07-29"]
        ]

    data = raw_config()
    data["data"]["root"] = str(hdn_root(tmp_path / "rates", b.dates, manifest_edit=shift))
    cfg_r = bt.BacktestConfig.from_mapping(data).with_paths(
        out=b.store_root, cache=b.cache_root, snapshots=b.snapshots_root
    )
    run_r = bt.BacktestRun(cfg_r, allow_calibrate=False)
    got = [run_r.ledger.verdict(d).status for d in b.dates]
    assert got == ["done", "done", "stale", "stale", "stale"]
    snap = run_r.snapshot_path("2022-07-29")
    old_snap = yaml.safe_load(snap.read_text())
    old_rates = old_snap["provenance"]["rate_curve_percent"]["zeros"]
    assert not bt.snapshot_bound(
        cfg_r, snap, run_r.inputs.file_sha("2022-07-29"), run_r.inputs.manifest_sha("2022-07-29")
    )
    _, _, imported, _ = run_r.ensure_snapshot("2022-07-29")
    path = run_r.snapshot_path("2022-07-29")
    new_snap = yaml.safe_load(path.read_text())
    new_rates = new_snap["provenance"]["rate_curve_percent"]["zeros"]
    assert imported and new_rates[0] == pytest.approx(old_rates[0] + 0.005, abs=2e-3)
    assert new_rates != old_rates
    # the market's rate curve is the option-implied funding curve: the manifest move reaches
    # the provenance (and the digest), not the discounting (SPEC §13.1, 2026-09-22)
    assert new_snap["market"]["rate_curve"] == old_snap["market"]["rate_curve"]
    assert bt.snapshot_bound(
        cfg_r, path, run_r.inputs.file_sha("2022-07-29"), run_r.inputs.manifest_sha("2022-07-29")
    )
    # an unchanged manifest entry of another date keeps that date's snapshot
    assert bt.snapshot_bound(
        cfg_r,
        run_r.snapshot_path("2022-07-28"),
        run_r.inputs.file_sha("2022-07-28"),
        run_r.inputs.manifest_sha("2022-07-28"),
    )
    # R4: the importer that wrote a stored snapshot is not the current one (R3b re-imported
    # 2022-07-29 into the shared snapshots directory, so under ``toy`` that date and its
    # successors are already stale; the first date is still done)
    first = b.dates[0]
    ledger1 = bt.Ledger.of(toy)
    assert ledger1.verdict(first).status == "done"
    snap = ledger1.snapshot_path(first)
    assert ledger1.read_snapshot(first).importer_tag == bt.IMPORTER_TAG
    monkeypatch.setattr(bt, "IMPORTER_TAG", "2099-01-01")
    ledger2 = bt.Ledger.of(toy)
    got = [ledger2.verdict(d).status for d in b.dates]
    assert got == ["stale"] * len(b.dates), got
    reason = ledger2.verdict(first).reason
    assert reason == bt.importer_tag_reason(ledger2.read_snapshot(first).importer_tag)
    assert "imported by importer 2026-" in reason and "(current 2099-01-01)" in reason
    assert f"depends on stale {first}: {reason}" == ledger2.verdict(b.dates[1]).reason
    inputs = ledger2.inputs
    assert not bt.snapshot_bound(toy, snap, inputs.file_sha(first), inputs.manifest_sha(first))
    monkeypatch.undo()
    assert bt.snapshot_bound(toy, snap, inputs.file_sha(first), inputs.manifest_sha(first))
    assert bt.Ledger.of(toy).verdict(first).status == "done"


# --------------------------------------------------------------------------------------------
# the integrity invariant: the walking test
# --------------------------------------------------------------------------------------------

#: The date the walking test mutates (its previous date is the fixture's shifted date).
WALK_DATE = "2022-07-29"


def test_integrity_walking_test(
    build_copy: BacktestBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A date is done iff its ``CURRENT`` pointer verifies (every file of the attempt it names
    hashes as recorded) and :func:`dependency_record` recomputes the attempt's record.  Every
    recorded input is mutated in turn on the build; the mutated date and every later date turn
    unconfirmed and stage 2 reports exactly them with their commands (never a crash, never a
    silent render); undoing the mutation confirms them again.  An edited snapshot is re-imported
    by the next run (D3).  Last, the vendor day file of the previous date changes (its SPX close
    x 1.005): the previous date and its successors turn stale; recomputing the previous date
    and then the date restores the chain one date at a time."""
    b = build_copy
    dates = list(b.dates)
    later = dates[dates.index(WALK_DATE) :]
    day = b.store_root / "dates" / WALK_DATE
    att = current(b.store_root, WALK_DATE)
    snap = b.snapshots_root / f"spx_{WALK_DATE}.yaml"
    key = current_json(b.store_root, WALK_DATE, "fit.json")["cache_key"]
    toy = cfg_at(b)
    matrix: list[tuple[str, dict[str, str], str]] = []

    def verdicts(cfg: Any) -> dict[str, bt.Verdict]:
        ledger = bt.Ledger.of(cfg)
        return {d: ledger.verdict(d) for d in ledger.vendor_calendar}

    def walk(
        label: str,
        expected: dict[str, str],
        reason: str,
        *,
        cfg: Any = toy,
        cfg_file: Path | None = None,
    ) -> None:
        got = verdicts(cfg)
        unconfirmed = {d: v.status for d, v in got.items() if v.status != "done"}
        assert unconfirmed == expected, (label, {d: (v.status, v.reason) for d, v in got.items()})
        root = next(d for d in got if d in expected)
        assert reason in got[root].reason, (label, got[root].reason)
        for d in expected:
            if expected[d] == "stale" and d != root:
                assert got[d].reason.startswith(f"depends on stale {root}: "), (label, d)
        with pytest.raises(runner.MissingRequirements) as info:
            run_stage2(b, tmp_path / f"study{len(matrix)}", backtest=cfg_file, latex_check=False)
        reported = {Path(r.id).parent.name for r in info.value.missing}
        assert reported == set(expected), (label, reported)
        assert info.value.commands and all(
            c.startswith("volsto-backtest run") for c in info.value.commands
        )
        assert not (tmp_path / f"study{len(matrix)}").exists()
        matrix.append((label, expected, got[root].reason))

    def confirmed() -> None:
        assert {d: v.status for d, v in verdicts(toy).items()} == dict.fromkeys(dates, "done")

    def swap(path: Path, data: bytes | None) -> bytes:
        old = path.read_bytes()
        path.chmod(0o644)
        if data is None:
            path.unlink()
        else:
            path.write_bytes(data)
        return old

    stale_later = dict.fromkeys(later, "stale")
    confirmed()
    # the snapshot: edited (the next run re-imports it: D3), then removed
    original = snap.read_bytes()
    text = original.decode()
    spot = yaml.safe_load(text)["market"]["spot"]
    snap.write_text(text.replace(f"  spot: {spot}\n", f"  spot: {spot * 1.001!r}\n", 1))
    walk("snapshot edited (spot x 1.001)", stale_later, "its snapshot differs")
    code, text = run_cli(
        ["run", str(TOY_CONFIG), "--no-calibrate", "--resume", *b.path_args()], capsys
    )
    assert code == 0 and "0 date(s) ok" in text and "3 confirmed during the run" in text, text
    assert yaml.safe_load(snap.read_text())["market"]["spot"] == spot, "D3: re-imported"
    confirmed()
    snap.unlink()
    walk("snapshot removed", stale_later, "its snapshot is missing")
    snap.write_bytes(original)
    confirmed()
    # every file of the current attempt: edited, removed; an unexpected file; the pointer
    fit = json.loads((att / "fit.json").read_text())
    fit["params"]["nu"] += 1e-3
    old = swap(att / "fit.json", json.dumps(fit).encode())
    walk("fit.json edited (nu + 1e-3)", stale_later, "its fit.json was modified")
    swap(att / "fit.json", None)
    walk("fit.json removed", stale_later, "its fit.json is missing")
    swap_back = old
    (att / "fit.json").write_bytes(swap_back)
    confirmed()
    frame = pd.read_parquet(att / "rows.parquet")
    buf = tmp_path / "short.parquet"
    frame.iloc[1:].to_parquet(buf, index=False)
    old = swap(att / "rows.parquet", buf.read_bytes())
    walk(
        "rows.parquet with one row dropped",
        stale_later,
        f"its rows.parquet was modified ({len(frame) - 1} row(s), {len(frame)} recorded)",
    )
    swap(att / "rows.parquet", None)
    walk("rows.parquet removed", stale_later, "its rows.parquet is missing")
    (att / "rows.parquet").write_bytes(old)
    confirmed()
    done = json.loads((att / "done.json").read_text())
    done["wall_s"] = 1.0  # a field outside the record (D4)
    old = swap(att / "done.json", json.dumps(done).encode())
    walk("done.json edited outside its record (wall_s)", stale_later, "done.json was modified")
    (att / "done.json").write_bytes(old)
    (att / "extra.txt").write_text("x")
    walk("an unexpected file in the attempt", stale_later, "unexpected file extra.txt")
    (att / "extra.txt").unlink()
    confirmed()
    pointer = (day / "CURRENT").read_bytes()
    (day / "CURRENT").unlink()
    walk(
        "CURRENT removed",
        {WALK_DATE: "missing", **dict.fromkeys(later[1:], "pending")},
        "no CURRENT (1 published attempt(s)",
    )
    ptr = json.loads(pointer)
    (day / "CURRENT").write_text(json.dumps({**ptr, "attempt": "ok-" + "0" * 32}))
    walk("CURRENT names an attempt that does not exist", stale_later, "is missing")
    (day / "CURRENT").write_bytes(pointer)
    confirmed()
    # the leverage
    shutil.move(str(b.cache_root / key), tmp_path / "stash")
    walk("leverage removed from the cache", stale_later, "not in the configured cache")
    shutil.move(str(tmp_path / "stash"), b.cache_root / key)
    confirmed()
    # the base spec (content), the rates entry, the calendar
    spec = yaml.safe_load(REFERENCE_SPEC.read_text())
    spec["particle"]["bandwidth_factor"] = 1.3
    spec_file = tmp_path / "spec.yaml"
    spec_file.write_text(yaml.safe_dump(spec))
    data = raw_config()
    data["calibration"]["base_spec"] = str(spec_file)
    cfg_file = write_config(tmp_path, data, "spec_cfg.yaml")
    walk(
        "base spec changed (bandwidth_factor 1.3)",
        dict.fromkeys(dates, "stale"),
        "stored under another config hash",
        cfg=cfg_at(b, cfg=cfg_file),
        cfg_file=cfg_file,
    )

    def shift(m: dict[str, Any]) -> None:
        m["rates"][WALK_DATE] = [r + 0.5 if r is not None else r for r in m["rates"][WALK_DATE]]

    gap_root = hdn_root(tmp_path / "gap", [d for d in dates if d != "2022-07-28"])
    for label, root in (
        (
            "manifest rates of the date + 50 bp",
            hdn_root(tmp_path / "rates", dates, manifest_edit=shift),
        ),
        ("calendar without the previous date", gap_root),
    ):
        data = raw_config()
        data["data"]["root"] = str(root)
        cfg_file = write_config(tmp_path, data, f"{len(matrix)}.yaml")
        walk(
            label, stale_later, "its inputs changed", cfg=cfg_at(b, cfg=cfg_file), cfg_file=cfg_file
        )
    confirmed()
    # the vendor data of the previous date (its SPX close x 1.005): recompute it, then the date
    prev = _backtest_build.TOY_SHIFTED_DATE
    assert dates.index(prev) == dates.index(WALK_DATE) - 1 and b.shifted_key
    data = raw_config()
    data["data"]["root"] = str(b.shifted_root / "hdn")
    cfg_file = write_config(tmp_path, data, "vendor.yaml")
    moved = cfg_at(b, cfg=cfg_file)
    walk(
        "previous date's day file: SPX close x 1.005",
        dict.fromkeys(dates[1:], "stale"),
        "its inputs changed",
        cfg=moved,
        cfg_file=cfg_file,
    )
    shutil.copytree(b.shifted_root / "cache" / b.shifted_key, b.cache_root / b.shifted_key)
    code, text = run_cli(
        ["run", str(cfg_file), "--no-calibrate", "--only-dates", prev, *b.path_args()], capsys
    )
    assert code == 0 and "1 date(s) ok" in text, text
    assert yaml.safe_load((b.snapshots_root / f"spx_{prev}.yaml").read_text())["market"][
        "spot"
    ] == pytest.approx(b.info["shifted_spot"], rel=1e-15)
    assert current_json(b.store_root, prev, "fit.json")["cache_key"] == b.shifted_key
    walk(
        "previous date recomputed from the new day file",
        stale_later,
        "its inputs changed",
        cfg=moved,
        cfg_file=cfg_file,
    )
    code, text = run_cli(
        ["run", str(cfg_file), "--no-calibrate", "--resume", "--limit", "1", *b.path_args()],
        capsys,
    )
    assert code == 0 and "1 date(s) ok" in text, text
    rows = store_rows(b.store_root).set_index(["date", "trade_id"])
    walk_rows = rows.loc[WALK_DATE]
    for trade, r in walk_rows[walk_rows["price_0"].notna()].iterrows():
        assert r["price_0"] == rows.loc[(prev, trade), "value"], trade  # the chain holds again
    walk(
        "the date recomputed: its successors still stale",
        dict.fromkeys(later[1:], "stale"),
        "its inputs changed",
        cfg=moved,
        cfg_file=cfg_file,
    )
    # the replaced attempts are still on disk (never deleted by run)
    assert len(list((b.store_root / "dates" / prev / "attempts").iterdir())) == 2
    print("\nintegrity walking test (mutation -> unconfirmed dates -> first reason):")
    for label, expected, why in matrix:
        print(f"  {label:<52} {sorted(expected.items())} :: {why}")


# --------------------------------------------------------------------------------------------
# storage: refusals, crashes, concurrency, migration
# --------------------------------------------------------------------------------------------


def test_refusals_read_the_verdicts(
    build_copy: BacktestBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Under ``--no-calibrate`` a wrong ``--cache`` is refused before anything is touched; so is
    a previous date whose leverage is missing — with its own stored results, or (N13) without
    them when the stored P&L of the next date used it (its key from ``previous_cache_key``, or
    by marking it for a store of the flat layout).  Explicitly recomputing such a date anyway
    never moves its ``CURRENT`` off the better attempt (the pointer rule)."""
    b = build_copy
    state = tree_state(b.base)
    empty = tmp_path / "empty_cache"
    wrong = [
        "--out",
        str(b.store_root),
        "--cache",
        str(empty),
        "--snapshots",
        str(b.snapshots_root),
    ]
    code, text = run_cli(
        ["run", str(TOY_CONFIG), "--resume", "--no-calibrate", "--only-dates", b.dates[0], *wrong],
        capsys,
    )
    assert code == bt.EXIT_REFUSED and "its stored results" in text and "PROJECTED" not in text
    assert tree_state(b.base) == state and not empty.exists()
    # N13: 07-28 removed (no stored results) and its leverage missing; 07-29 used it
    k28 = current_json(b.store_root, "2022-07-28", "fit.json")["cache_key"]
    shutil.move(str(b.cache_root / k28), tmp_path / "k28")
    shutil.move(str(b.store_root / "dates" / "2022-07-28"), tmp_path / "d28")
    rows29 = current(b.store_root, "2022-07-29", "rows.parquet").read_bytes()
    state = tree_state(b.base)
    for extra in (["--resume"], ["--only-dates", "2022-07-29"]):
        code, text = run_cli(
            ["run", str(TOY_CONFIG), "--no-calibrate", *extra, *b.path_args()], capsys
        )
        assert code == bt.EXIT_REFUSED, text
        assert f"2022-07-28 ({k28[:12]}, used by the stored P&L of 2022-07-29" in text, text
        assert tree_state(b.base) == state
    # the flat-layout record has no previous_cache_key: the previous date is marked instead
    run = bt.BacktestRun(cfg_at(b), allow_calibrate=False)
    v29 = run.ledger.verdict("2022-07-29")
    assert v29.status == "pending" and v29.stored is not None
    legacy = dict(v29.stored)
    legacy.pop("previous_cache_key")
    run.ledger._verdicts["2022-07-29"] = dataclasses.replace(v29, stored=legacy)
    assert "2022-07-28" in bt.leverage_refusals(run, ["2022-07-29"])
    # recomputing 07-29 anyway (the refusal bypassed): the incomplete attempt is only recorded
    run2 = bt.BacktestRun(cfg_at(b), allow_calibrate=False)
    with guard.calibration_forbidden():
        run2.resolve_closes("2022-07-29")
        out = run2.run_date("2022-07-29")
    assert out.status == "incomplete" and out.current is False
    assert current(b.store_root, "2022-07-29", "rows.parquet").read_bytes() == rows29
    assert bt.Ledger.of(cfg_at(b)).verdict("2022-07-29").status == "pending"
    shutil.move(str(tmp_path / "k28"), b.cache_root / k28)
    shutil.move(str(tmp_path / "d28"), b.store_root / "dates" / "2022-07-28")
    assert {v.status for v in bt.Ledger.of(cfg_at(b)).verdicts().values()} == {"done"}
    # the hash refusal also reads the verdicts, the previous date included
    data = raw_config()
    data["pricing"]["n_paths"] = 2002
    cfg = write_config(tmp_path, data)
    shutil.rmtree(b.store_root / "dates" / "2022-08-02")
    header = b.store_root / "backtest.json"
    head = json.loads(header.read_text())
    head["config_hash"] = bt.load_backtest_config(cfg).content_hash()
    header.write_text(json.dumps(head))
    code, text = run_cli(
        ["run", str(cfg), "--no-calibrate", "--only-dates", "2022-08-02", *b.path_args()], capsys
    )
    assert (
        code == bt.EXIT_REFUSED and "before them hold results computed under another config" in text
    )


def test_verdicts_read_once_under_churn(
    build_copy: BacktestBuild, monkeypatch: pytest.MonkeyPatch
) -> None:
    """P1 (the verifier's race_toctou / race_gc): while a reader judges 07-29 (which judges
    07-28), another writer recommits 07-28 and removes the attempt the reader just read — right
    after the reader read the attempt's files, or right after it read ``CURRENT``.  The verdicts
    stay right (07-28 done with its state link, 07-29 done): the reader parses the bytes it
    verified and reads again an attempt that vanished while ``CURRENT`` moved.  P2: a failure of
    07-29 recorded during that churn leaves ``CURRENT`` on the done attempt.  P3: ``gc`` (which
    holds every lock, so nothing interleaves) removes only superseded attempts."""
    b = build_copy
    cfg = cfg_at(b)
    q = bt.BacktestRun(cfg, allow_calibrate=False)
    held: set[str] = set()
    real_lock = bt.BacktestStore.lock

    @contextlib.contextmanager
    def lock(self: bt.BacktestStore, date: str) -> Any:
        with real_lock(self, date) as h:
            held.add(date)
            try:
                yield h
            finally:
                held.discard(date)

    monkeypatch.setattr(bt.BacktestStore, "lock", lock)
    state: dict[str, Any] = {"busy": False, "n": 0, "at": "attempt", "left": 0}

    def churn(what: str, date: str) -> None:
        if (
            date != "2022-07-28"
            or what != state["at"]
            or state["busy"]
            or state["left"] <= 0
            or "2022-07-28" in held
        ):
            return
        state["busy"] = True
        try:
            old = q.store.pointer("2022-07-28")["attempt"]  # type: ignore[index]
            recommit(q, "2022-07-28", f"churn-{state['n']}")
            with real_lock(q.store, "2022-07-28") as h:
                if q.store.pointer("2022-07-28")["attempt"] != old:  # type: ignore[index]
                    q.store.remove_attempt(h, q.store.attempt("2022-07-28", old))
            state["n"] += 1
            state["left"] -= 1
            q.ledger.invalidate()
        finally:
            state["busy"] = False

    monkeypatch.setattr(bt, "READ_HOOK", churn)
    # after the attempt read: one churn per pass (the verdict is memoised); after the pointer
    # read: 8 churns in a row (more than the 5 retries of round 3)
    for at, times in (("attempt", 1), ("attempt", 1), ("pointer", 8)):
        state.update(at=at, left=times)
        ledger = bt.Ledger.of(cfg)
        v28, v29 = ledger.verdict("2022-07-28"), ledger.verdict("2022-07-29")
        assert state["left"] == 0, (at, state)
        assert v28.status == "done" and v28.leverage and v28.link != bt.UNAVAILABLE, (at, v28)
        assert v29.status == "done", (at, v29.reason)
    # P2: a failed 07-29 recorded while 07-28 churns
    before = current(b.store_root, "2022-07-29").name
    state.update(at="attempt", left=5)
    note = bt.BacktestRun(cfg, allow_calibrate=False)._write_failure(
        "2022-07-29", "2022-07-29: simulated failure", 1.0, [], {}
    )
    assert current(b.store_root, "2022-07-29").name == before
    assert "recorded as failed-" in note and "CURRENT keeps" in note, note
    # P3: gc under every lock
    state.update(at="attempt", left=5)
    counts = bt.collect_garbage(bt.BacktestStore(b.store_root), bt.Ledger.of(cfg))
    monkeypatch.setattr(bt, "READ_HOOK", None)
    assert counts["attempts"] >= 1, counts
    assert [p.name for p in (b.store_root / "dates" / "2022-07-29" / "attempts").iterdir()] == [
        before
    ]
    assert {v.status for v in bt.Ledger.of(cfg).verdicts().values()} == {"done"}
    print(f"churn: {state['n']} recommits + removals during the reads; gc {counts}")


def test_failures_never_displace_results_and_gc_is_conservative(
    build_copy: BacktestBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """P2 (G3): 08-02's vendor file is broken, so its results are stale, and its recompute
    fails: ``CURRENT`` keeps the results (the failure is recorded beside them).  P3: ``gc``
    keeps every attempt of a date that is not done, removes the failure once the date is done
    again, refuses a config that is not the store's (G2) and skips a symbolic link (L1)."""
    b = build_copy
    data = raw_config()
    data["data"]["root"] = str(hdn_root(tmp_path, b.dates, corrupt=["2022-08-02"]))
    broken = write_config(tmp_path, data, "broken.yaml")
    results = current(b.store_root, "2022-08-02").name
    code, text = run_cli(
        ["run", str(broken), "--no-calibrate", "--only-dates", "2022-08-02", *b.path_args()],
        capsys,
    )
    assert code == bt.EXIT_FAILED and "KEPT 2022-08-02" in text, text
    assert current(b.store_root, "2022-08-02").name == results
    v = bt.Ledger.of(cfg_at(b, cfg=broken)).verdict("2022-08-02")
    assert v.status == "stale" and "inputs changed" in v.reason, v
    assert "a later attempt failed" in v.note and "later" not in v.reason, v
    code, text = run_cli(["gc", str(broken), *b.path_args()], capsys)
    assert code == 0 and "kept 1 attempt(s)" in text, text
    assert len(list((b.store_root / "dates" / "2022-08-02" / "attempts").iterdir())) == 2
    code, text = run_cli(["gc", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == 0 and "removed 1 non-current attempt(s)" in text, text
    assert [p.name for p in (b.store_root / "dates" / "2022-08-02" / "attempts").iterdir()] == [
        results
    ]
    # G2: another config's gc is refused before anything is touched
    data = raw_config()
    data["pricing"]["n_paths"] = 2002
    other = write_config(tmp_path, data, "other.yaml")
    state = tree_state(b.store_root)
    code, text = run_cli(["gc", str(other), *b.path_args()], capsys)
    assert code == bt.EXIT_REFUSED and "gc only runs with the store's own config" in text
    assert tree_state(b.store_root) == state
    # L1: a symbolic link among the attempts is skipped, its target untouched
    target = tmp_path / "elsewhere" / ("ok-" + "0" * 32)
    shutil.copytree(current(b.store_root, "2022-07-27"), target)
    link = b.store_root / "dates" / "2022-07-28" / "attempts" / target.name
    link.symlink_to(target)
    code, text = run_cli(["gc", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == 0 and "skipped 1 symbolic link(s)" in text, text
    assert link.is_symlink() and (target / "done.json").is_file()


def test_refusals_come_before_any_pointer_move(
    build_copy: BacktestBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """P4 (G1): after a ``--force`` run of config B (the header now names B), a ``run --resume``
    of config A is refused before any adoption or pointer move — A's published attempt, which
    A's adoption would take, stays non-current."""
    b = build_copy
    data = raw_config()
    data["pricing"]["n_paths"] = 2002
    other = write_config(tmp_path, data, "b.yaml")
    code, text = run_cli(
        [
            "run",
            str(other),
            "--no-calibrate",
            "--force",
            "--only-dates",
            b.dates[0],
            *b.path_args(),
        ],
        capsys,
    )
    assert code == 0, text
    assert len(list((b.store_root / "dates" / b.dates[0] / "attempts").iterdir())) == 2
    state = tree_state(b.store_root)
    code, text = run_cli(
        ["run", str(TOY_CONFIG), "--no-calibrate", "--resume", *b.path_args()], capsys
    )
    assert code == bt.EXIT_REFUSED and "was built from config hash" in text, text
    assert "adopted" not in text
    assert tree_state(b.store_root) == state


class SimulatedKill(BaseException):
    """A kill as the storage sees it: nothing handles it (no ``except``), only ``finally``."""


def recommit(run: bt.BacktestRun, date: str, tag: str) -> tuple[Any, Any, bool]:
    """Commit the date's current outcome again (a new attempt: same rows and fit, another
    ``created_utc``) through the run's commit path — a recompute without pricing."""
    v = run.ledger.verdict(date)
    rows = pd.read_parquet(v.files / "rows.parquet")
    fit = json.loads((v.files / "fit.json").read_text())
    done = json.loads((v.files / "done.json").read_text())
    used = bool(done["record"]["previous"] and done["record"]["previous"]["leverage"])
    done.pop("record")
    done["created_utc"] = f"{done['created_utc']}#{tag}"
    return run._commit(date, rows, fit, done, prev_used=used)


def pointer_state(store: Path, date: str) -> tuple[str, bool]:
    """``(attempt id, whether CURRENT names a complete attempt)``."""
    ptr = json.loads((store / "dates" / date / "CURRENT").read_text())
    att = bt.BacktestStore(store).attempt(date, ptr["attempt"])
    return ptr["attempt"], att.verify(ptr["files"]) == ""


CRASH_MODES: dict[str, BaseException] = {
    "exception": OSError("injected"),
    "interrupt": KeyboardInterrupt(),
    "kill": SimulatedKill(),
}


def test_storage_crash_matrix(
    build_copy: BacktestBuild,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failure — an exception, a KeyboardInterrupt, or a kill (nothing but ``finally``
    blocks run) — injected at every step of the attempt and pointer paths
    (:data:`~volsto.studies.backtest.CRASH_POINTS`), in two scenarios: (A) recommitting a done
    date (the pointer may move to the equal new attempt) and (B) recommitting a date whose
    current attempt was tampered with (the pointer must move).  After every crash: ``CURRENT``
    names a complete attempt; no confirmed result is lost (A: every date stays done); the
    verdicts are right (B: the date and its successors stale until the pointer moved); then
    ``status``, ``run --resume`` (which adopts a published attempt or recomputes) and stage 2
    leave every date done.  Four real kills (``os._exit`` in a subprocess) check the
    simulation, and ``gc`` removes the leftovers of the dead writers only."""
    b = build_copy
    cfg = cfg_at(b)
    run = bt.BacktestRun(cfg, allow_calibrate=False)
    results: list[tuple[str, str, str, str]] = []
    first = b.dates[0]

    def inject(point: str, mode: str) -> None:
        def hook(name: str) -> None:
            if name == point:
                raise CRASH_MODES[mode]

        monkeypatch.setattr(bt, "CRASH_HOOK", hook)

    def all_done(ledger_cfg: Any = cfg) -> dict[str, str]:
        return {d: v.status for d, v in bt.Ledger.of(ledger_cfg).verdicts().items()}

    points = bt.CRASH_POINTS
    with guard.calibration_forbidden():
        # scenario A: 07-29 is done; its recommit may move the pointer to the equal attempt
        for mode in CRASH_MODES:
            for point in points:
                before, _ = pointer_state(b.store_root, "2022-07-29")
                inject(point, mode)
                with pytest.raises((OSError, KeyboardInterrupt, SimulatedKill)):
                    recommit(run, "2022-07-29", f"A-{mode}-{point}")
                monkeypatch.setattr(bt, "CRASH_HOOK", None)
                run.ledger.invalidate()
                after, complete = pointer_state(b.store_root, "2022-07-29")
                assert complete, (mode, point)
                moved = after != before
                assert moved == (points.index(point) >= points.index("pointer.replaced")), (
                    mode,
                    point,
                )
                assert set(all_done().values()) == {"done"}, (mode, point, all_done())
                results.append(("A", mode, point, "moved" if moved else "kept"))
        # scenario B: 07-27's current attempt tampered with (stale): the recommit must move
        for mode in CRASH_MODES:
            for point in points:
                att = current(b.store_root, first)
                assert all_done()[first] == "done"
                rows = pd.read_parquet(att / "rows.parquet")  # the pristine outcome
                fit = json.loads((att / "fit.json").read_text())
                done = json.loads((att / "done.json").read_text())
                done.pop("record")
                done["created_utc"] = f"{done['created_utc']}#B-{mode}-{point}"
                (att / "done.json").write_bytes((att / "done.json").read_bytes() + b" ")
                run.ledger.invalidate()
                assert all_done()[first] == "stale"
                inject(point, mode)
                with pytest.raises((OSError, KeyboardInterrupt, SimulatedKill)):
                    run._commit(first, rows, fit, done)
                monkeypatch.setattr(bt, "CRASH_HOOK", None)
                run.ledger.invalidate()
                after_id, complete = pointer_state(b.store_root, first)
                assert complete is (after_id != att.name), (mode, point)  # tampered = incomplete
                moved = after_id != att.name
                assert moved == (points.index(point) >= points.index("pointer.replaced"))
                published = points.index(point) >= points.index("publish.renamed")
                statuses = all_done()
                if moved:
                    assert set(statuses.values()) == {"done"}, (mode, point, statuses)
                else:
                    assert set(statuses.values()) == {"stale"}, (mode, point, statuses)
                recompute = not published and point == points[0]
                if not (moved or published or recompute):
                    # the same state as the first pre-publish point: restore it without pricing
                    restore = dict(done, created_utc=f"{done['created_utc']}-restored")
                    assert run._commit(first, rows, fit, restore)[2]
                code, text = run_cli(
                    ["run", str(TOY_CONFIG), "--no-calibrate", "--resume", *b.path_args()], capsys
                )
                assert code == 0, (mode, point, text)
                if published and not moved:
                    assert "adopted a published attempt for 1 date(s)" in text, text
                if recompute:
                    assert "1 date(s) ok" in text and "4 confirmed during the run" in text, text
                else:
                    assert "nothing to compute" in text, (mode, point, text)
                code, text = run_cli(["status", str(TOY_CONFIG), *b.path_args()], capsys)
                assert code == 0 and "done 5" in text, (mode, point, text)
                run.ledger.invalidate()
                how = "moved" if moved else "adopted" if published else "recomputed"
                results.append(
                    ("B", mode, point, how if how != "recomputed" or recompute else "restored")
                )
    staging = [
        p for d in b.dates for p in (b.store_root / "dates" / d / "attempts").glob(".staging-*")
    ]
    assert staging, "the simulated kills leave staging directories"
    # stage 2 after the matrix
    study = run_stage2(b, tmp_path / "study", latex_check=False)
    assert study.exit_code == 0
    # real kills in a subprocess (os._exit: no finally at all), scenario A on 07-27
    script = tmp_path / "kill.py"
    script.write_text(
        "import os, sys\n"
        "sys.path.insert(0, sys.argv[4])\n"
        "import test_backtest as tb\n"
        "from volsto.studies import backtest as bt\n"
        "cfg = bt.load_backtest_config(sys.argv[1])\n"
        "run = bt.BacktestRun(cfg, allow_calibrate=False)\n"
        "def hook(name):\n"
        "    if name == sys.argv[2]:\n"
        "        os._exit(137)\n"
        "bt.CRASH_HOOK = hook\n"
        "tb.recommit(run, sys.argv[3], 'kill-' + sys.argv[2])\n"
        "os._exit(0)\n"
    )
    cfg_file = tmp_path / "abs.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg.to_mapping()))
    import subprocess
    import sys

    for point in ("stage.done", "publish.renamed", "pointer.written", "pointer.replaced"):
        before, _ = pointer_state(b.store_root, first)
        proc = subprocess.run(
            [sys.executable, str(script), str(cfg_file), point, first, str(Path(__file__).parent)],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert proc.returncode == 137, proc.stderr[-2000:]
        after, complete = pointer_state(b.store_root, first)
        assert complete and (after != before) == (point == "pointer.replaced"), point
        assert set(all_done().values()) == {"done"}, point
        results.append(("kill", "os._exit", point, "moved" if after != before else "kept"))
    left = bt.BacktestStore(b.store_root).leftovers()
    assert left["staging"] and left["pointer"], left
    code, text = run_cli(["gc", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == 0, text
    left = bt.BacktestStore(b.store_root).leftovers()
    assert not left["pointer"]
    # the in-process kills' staging directories belong to this (live) process: kept.  gc's
    # ownership test is the pid in the name (os.kill(pid, 0)): a killed child's pid reused by
    # another live process keeps that child's staging directory too (seen once under a
    # saturated machine, 2026-09-22) — so what is asserted is gc's rule, dead owners removed and
    # live owners kept, not that only this pid survives
    assert left["staging"], left
    for p in left["staging"]:
        pid = int(p.name[len(bt.STAGING_PREFIX) :].split("-")[1])
        assert pid == os.getpid() or bt._staging_owner_alive(p), p
        assert bt._staging_owner_alive(p), p
    for d in b.dates:
        assert len(list((b.store_root / "dates" / d / "attempts").glob("[a-z]*"))) == 1
    assert set(all_done().values()) == {"done"}
    print("\nstorage crash matrix (scenario, mode, point -> pointer):")
    for row in results:
        print("  " + " ".join(f"{x:<18}" for x in row))


def _race_worker(args: tuple[str, str, str, int, str]) -> str:
    role, cfg_file, date, n, tests_dir = args
    import sys

    sys.path.insert(0, tests_dir)
    import test_backtest as tb

    from volsto.studies import backtest as bt_

    cfg = bt_.load_backtest_config(cfg_file)
    run = bt_.BacktestRun(cfg, allow_calibrate=False)
    for i in range(n):
        if role == "gc":
            bt_.collect_garbage(run.store, run.ledger)
        else:
            tb.recommit(run, date, f"{role}-{i}")
        run.ledger.invalidate()
    return role


def test_concurrent_commits_and_gc(build_copy: BacktestBuild, tmp_path: Path) -> None:
    """Two processes recommit the same date while a third runs ``gc`` and this one reads: every
    read sees ``CURRENT`` naming a complete attempt of a done date; afterwards the date is done
    and ``gc`` leaves exactly the current attempt."""
    import multiprocessing as mp

    b = build_copy
    cfg = cfg_at(b)
    cfg_file = tmp_path / "abs.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg.to_mapping()))
    date = b.dates[0]
    tests_dir = str(Path(__file__).parent)
    jobs = [
        ("w1", str(cfg_file), date, 12, tests_dir),
        ("w2", str(cfg_file), date, 12, tests_dir),
        ("gc", str(cfg_file), date, 12, tests_dir),
    ]
    ctx = mp.get_context("spawn")
    reads = 0
    with ctx.Pool(3) as pool:
        pending = pool.map_async(_race_worker, jobs)
        while not pending.ready():
            ledger = bt.Ledger.of(cfg)
            att, doc, why = ledger.verify_current(date)  # lock-free, as status and stage 2 read
            assert not why and att is not None and doc is not None, why
            assert ledger.verdict(date).status == "done"
            reads += 1
        assert sorted(pending.get(timeout=600)) == ["gc", "w1", "w2"]
    print(f"concurrent commits: {reads} consistent reads during the race")
    assert reads > 0
    assert {v.status for v in bt.Ledger.of(cfg).verdicts().values()} == {"done"}
    bt.collect_garbage(bt.BacktestStore(b.store_root), bt.Ledger.of(cfg))
    assert len(list((b.store_root / "dates" / date / "attempts").glob("[a-z]*"))) == 1


GOLDEN_R2 = ROOT / "tests" / "golden" / "backtest_store_r2.tar.gz"


def flat_store(build: BacktestBuild) -> dict[str, bytes]:
    """Replace the build's store by the round-2 golden (the flat layout written by the
    volsto-backtest of 2026-09-16 on the same 5 toy dates) and unbind the snapshots; returns the
    golden's files."""
    import tarfile

    shutil.rmtree(build.store_root / "dates")
    for rec in build.snapshots_root.glob("*.import.json"):
        rec.unlink()
    files: dict[str, bytes] = {}
    with tarfile.open(GOLDEN_R2) as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            assert not member.name.startswith(("/", "..")) and ".." not in member.name
            data = tar.extractfile(member).read()  # type: ignore[union-attr]
            dest = build.store_root / member.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            files[member.name] = data
    return files


def test_round2_store_is_migrated_in_place(
    build_copy: BacktestBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A store written by the flat layout (the round-2 golden) is readable: stage 2 asks for the
    migration; ``status`` migrates it in place — every file moves bit-identical into
    ``attempts/legacy-ok-*``, ``CURRENT`` is written from their hashes, the migration is
    recorded, the snapshots whose results verify are bound — and every date is done; ``run
    --resume`` computes nothing and stage 2 renders."""
    b = build_copy
    files = flat_store(b)
    with pytest.raises(runner.MissingRequirements) as info:
        run_stage2(b, tmp_path / "before", latex_check=False)
    assert info.value.commands[0].startswith("volsto-backtest migrate")
    code, text = run_cli(["status", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == 0 and "migrated 5 date(s)" in text and "done 5" in text, text
    for d in b.dates:
        assert sorted(p.name for p in (b.store_root / "dates" / d).iterdir()) == [
            "CURRENT",
            "attempts",
        ]
        att = current(b.store_root, d)
        assert att.name.startswith("legacy-ok-")
        ptr = json.loads((b.store_root / "dates" / d / "CURRENT").read_text())
        for name in ("rows.parquet", "fit.json", "done.json"):
            data = files[f"dates/{d}/{name}"]
            assert (att / name).read_bytes() == data
            assert ptr["files"][name] == hashlib.sha256(data).hexdigest()
    log_ = json.loads((b.store_root / "migrations.json").read_text())["migrations"]
    assert len(log_) == 1 and sorted(log_[0]["dates"]) == list(b.dates)
    # P5: bound because a fresh import of each day file reproduced the snapshot
    assert sorted(log_[0]["snapshots_bound"]) == list(b.dates)
    assert log_[0]["snapshots_unbound"] == {}
    assert len(list(b.snapshots_root.glob("*.import.json"))) == len(b.dates)
    code, text = run_cli(
        ["run", str(TOY_CONFIG), "--no-calibrate", "--resume", *b.path_args()], capsys
    )
    assert code == 0 and "nothing to compute" in text, text
    after = run_stage2(b, tmp_path / "after", latex_check=False)
    assert after.exit_code == 0
    # a round-2 store has no cumulative error: every aggregate is labelled a root sum of squares
    res = after.results
    for row, note in zip(res.long("pnl_trade", "pnl")["row"], res.long("pnl_trade", "pnl")["note"]):
        one_day = res.value("pnl_trade", row, "n_days")[0] == 1.0
        assert (note == "") if one_day else (bt.SE_RSS_DATES in note), (row, note)
    assert "predates the stored cumulative P&L error" in (after.out_dir / "study.md").read_text()
    code, text = run_cli(["migrate", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == 0 and "migrated 0 date(s)" in text  # idempotent


def n3f_leftover(build: BacktestBuild) -> None:
    """The flat-layout leftovers the verifier's N3f / crash_ops migrate scenario starts from:
    07-29 only in ``.staging`` (a flat commit interrupted between its renames) and a failure
    recorded next to 08-01's results."""
    staging = build.store_root / ".staging"
    staging.mkdir()
    os.rename(build.store_root / "dates" / "2022-07-29", staging / "2022-07-29.old-x1")
    failure = {"date": "2022-08-01", "status": "failed", "error": "simulated", "created_utc": "z"}
    (build.store_root / "dates" / "2022-08-01" / "failure.json").write_text(json.dumps(failure))


def test_migration_survives_crashes_and_keeps_flat_leftovers(
    build_copy: BacktestBuild, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A migration interrupted at each of its steps (:data:`MIGRATION_CRASH_POINTS`: exception,
    KeyboardInterrupt, kill) loses no file and completes when the store is opened again — the
    leftovers marked imported included (P7, the verifier's k = 124..129).  The leftovers of the
    flat layout are kept: results moved aside by an interrupted flat commit (N3f) become the
    date's current attempt, and a failure recorded next to results a non-current attempt."""
    b = build_copy
    cfg = cfg_at(b)
    # the snapshot binding (a fresh import per date) is tested on its own below
    monkeypatch.setattr(bt, "_bind_migrated_snapshots", lambda ledger, dates: ([], {}))
    for mode in CRASH_MODES:
        for point in bt.MIGRATION_CRASH_POINTS:
            files = flat_store(b)
            n3f_leftover(b)

            def hook(name: str, point: str = point, mode: str = mode) -> None:
                if name == point:
                    raise CRASH_MODES[mode]

            monkeypatch.setattr(bt, "CRASH_HOOK", hook)
            with pytest.raises((OSError, KeyboardInterrupt, SimulatedKill)):
                bt.BacktestRun(cfg, allow_calibrate=False, migrate=True)
            monkeypatch.setattr(bt, "CRASH_HOOK", None)
            store = bt.BacktestStore(b.store_root)
            for name, data in files.items():  # every golden file is somewhere, intact
                if not name.startswith("dates/"):
                    continue
                _, d, f = name.split("/")
                where = [store.date_dir(d) / f] + [a.path / f for a in store.attempts(d)]
                where += list((b.store_root / ".staging").glob(f"{d}.*/{f}"))
                assert any(p.is_file() and p.read_bytes() == data for p in where), (
                    mode,
                    point,
                    name,
                )
            reopened = bt.BacktestRun(cfg, allow_calibrate=False, migrate=True)
            assert {v.status for v in reopened.ledger.verdicts().values()} == {"done"}, (
                mode,
                point,
            )
            assert not any(store.legacy_files(d) for d in b.dates)
            assert not (b.store_root / ".staging").exists(), (mode, point)
    # the last store: 07-29 restored from the leftover, 08-01's failure recorded beside
    code, text = run_cli(["status", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == 0 and "done 5" in text, text
    labels = sorted(
        p.name.rsplit("-", 1)[0]
        for p in (b.store_root / "dates" / "2022-08-01" / "attempts").iterdir()
    )
    assert labels == ["legacy-failed", "legacy-ok"], labels
    assert current(b.store_root, "2022-08-01").name.startswith("legacy-ok-")
    v = bt.Ledger.of(cfg).verdict("2022-08-01")
    assert "a later attempt failed: simulated" in v.note and v.reason == "", v


def test_migration_quarantines_foreign_leftovers_and_binds_snapshots_to_the_vendor(
    build_copy: BacktestBuild, capsys: pytest.CaptureFixture[str]
) -> None:
    """P7 (M1): a flat date without done.json is moved intact to ``quarantine/`` (recorded), never
    unlinked.  P8 (N4): a ``.staging`` leftover of another config never becomes current.  P5
    (M3): a snapshot is bound only when a fresh import of its day file reproduces it; an edited
    snapshot whose results verify stays unbound (recorded), and the next run re-imports it."""
    b = build_copy
    cfg = cfg_at(b)
    files = flat_store(b)
    # M1: 07-28 lost its done.json
    (b.store_root / "dates" / "2022-07-28" / "done.json").unlink()
    # N4: 07-29 exists only as a leftover of another config
    staging = b.store_root / ".staging"
    staging.mkdir()
    foreign = staging / "2022-07-29.new-foreign"
    os.rename(b.store_root / "dates" / "2022-07-29", foreign)
    doc = json.loads((foreign / "done.json").read_text())
    doc["config_hash"] = doc["record"]["config_hash"] = "f" * 64
    (foreign / "done.json").write_text(json.dumps(doc))
    # M3: 07-27's snapshot edited (a provenance count the spec does not read, so its leverage
    # key still follows), and its record says it was computed from the edited snapshot
    snap = b.snapshots_root / "spx_2022-07-27.yaml"
    text = snap.read_text()
    n_points = yaml.safe_load(text)["provenance"]["n_points"]
    snap.write_text(text.replace(f"  n_points: {n_points}\n", f"  n_points: {n_points + 1}\n", 1))
    d27 = b.store_root / "dates" / "2022-07-27" / "done.json"
    doc = json.loads(d27.read_text())
    doc["record"]["snapshot"] = bt.snapshot_digest(snap)
    d27.write_text(json.dumps(doc))
    _, text = run_cli(["status", str(TOY_CONFIG), *b.path_args()], capsys)
    assert "migrated 5 date(s)" in text, text
    log_ = json.loads((b.store_root / "migrations.json").read_text())["migrations"][-1]
    # M1: quarantined, bit-identical, recorded
    q = log_["dates"]["2022-07-28"]["quarantined"]
    for name in ("rows.parquet", "fit.json"):
        assert (b.store_root / q / name).read_bytes() == files[f"dates/2022-07-28/{name}"]
    assert not (b.store_root / "dates" / "2022-07-28" / "attempts").exists()
    # N4: published, never current
    assert not (b.store_root / "dates" / "2022-07-29" / "CURRENT").exists()
    assert [
        p.name.rsplit("-", 1)[0]
        for p in (b.store_root / "dates" / "2022-07-29" / "attempts").iterdir()
    ] == ["legacy-ok"]
    verdicts = bt.Ledger.of(cfg).verdicts()
    assert verdicts["2022-07-28"].status == "missing" and verdicts["2022-07-29"].status == "missing"
    # M3: 07-27 verifies (its record names the edited snapshot) but stays unbound
    assert verdicts["2022-07-27"].status == "done"
    assert "fresh import" in log_["snapshots_unbound"]["2022-07-27"], log_
    assert not bt.import_record_path(snap).exists()
    assert "2022-07-27" not in log_["snapshots_bound"]
    run = bt.BacktestRun(cfg, allow_calibrate=False)
    run.ensure_snapshot("2022-07-27")
    assert yaml.safe_load(snap.read_text())["provenance"]["n_points"] == n_points  # re-imported
    assert bt.Ledger.of(cfg).verdict("2022-07-27").status == "stale"


def test_metadata_files_versions_and_foreign_pointers(build_copy: BacktestBuild) -> None:
    """P6 (F1): exactly ``.DS_Store`` and the AppleDouble ``._<name>`` of a listed file are
    ignored (``._other`` is not).  P8 (V1): a pointer of an unknown version is unusable.  The
    code version recorded in each attempt is informational: not part of the record."""
    b = build_copy
    cfg = cfg_at(b)
    a29, a01 = current(b.store_root, "2022-07-29"), current(b.store_root, "2022-08-01")
    (a29 / ".DS_Store").write_bytes(b"\x00\x00\x00\x01Bud1")
    (a01 / "._rows.parquet").write_bytes(b"\x00\x05\x16\x07")
    assert {v.status for v in bt.Ledger.of(cfg).verdicts().values()} == {"done"}
    (a01 / "._other.txt").write_bytes(b"\x00")
    v = bt.Ledger.of(cfg).verdict("2022-08-01")
    assert v.status == "stale" and "unexpected file ._other.txt" in v.reason, v
    (a01 / "._other.txt").unlink()
    ptr_path = b.store_root / "dates" / "2022-07-29" / "CURRENT"
    ptr = json.loads(ptr_path.read_text())
    ptr_path.write_text(json.dumps({**ptr, "version": 99}))
    v = bt.Ledger.of(cfg).verdict("2022-07-29")
    assert v.status == "stale" and "pointer version 99" in v.reason, v
    ptr_path.write_text(json.dumps(ptr))
    doc = current_json(b.store_root, "2022-07-29", "done.json")
    assert doc["volsto_version"]["package"] and "git_commit" in doc["volsto_version"]
    assert "volsto_version" not in doc["record"] and "code_version" not in doc["record"]
    # another code version writing the same outcome is still done
    run = bt.BacktestRun(cfg, allow_calibrate=False)
    v = run.ledger.verdict("2022-07-29")
    rows = v.verified.frame()
    fit = v.verified.json("fit.json")
    done = dict(v.doc or {})
    used = bool(done.pop("record")["previous"]["leverage"])
    done["volsto_version"] = {"package": "0.0.0", "git_commit": "0" * 40, "git_dirty": True}
    done["created_utc"] += "#other-code"
    _, verdict, is_current = run._commit("2022-07-29", rows, fit, done, prev_used=used)
    assert verdict.status == "done" and is_current


def other_config(tmp_path: Path, name: str = "other.yaml") -> Path:
    """The toy config under another hash (one more pricing path)."""
    data = raw_config()
    data["pricing"]["n_paths"] = 2002
    return write_config(tmp_path, data, name)


def full_tree(root: Path) -> dict[str, str]:
    """Every entry under ``root`` (directories included) with the SHA-256 of each file."""
    return {
        str(p.relative_to(root)): (
            hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else "<dir>"
        )
        for p in sorted(root.rglob("*"))
    }


def test_round5_refusals_come_before_any_write(
    build_copy: BacktestBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Round 5 A (findings 1, 14, 18, 19): on a store still in the flat layout nothing is
    written — no header rebinding, no migration, no probe.json, no snapshot — before a refusal:
    another config's ``run`` (exit 2, with the ``--force`` and ``--out`` commands) and
    ``migrate`` (exit 2, not a silent 0), its ``dry-run`` (only a note), a wrong ``--cache``
    under ``--no-calibrate`` and a flat date computed under another config (both refusals read
    the flat files, each with a runnable command)."""
    b = build_copy
    flat_store(b)
    other = other_config(tmp_path)
    before = full_tree(b.store_root)
    code, text = run_cli(["run", str(other), "--no-calibrate", "--resume", *b.path_args()], capsys)
    assert code == bt.EXIT_REFUSED and "was built from config hash" in text, text
    assert full_tree(b.store_root) == before
    code, text = run_cli(["migrate", str(other), *b.path_args()], capsys)
    assert code == bt.EXIT_REFUSED and "the store's own config" in text, text
    assert full_tree(b.store_root) == before
    code, text = run_cli(["dry-run", str(other), *b.path_args()], capsys)
    assert code == 0 and "a run would be refused" in text, text
    assert full_tree(b.store_root) == before  # F14: no probe.json, no re-imported snapshot
    # F18: a wrong cache under --no-calibrate, evaluated on the flat results
    empty = tmp_path / "empty_cache"
    empty.mkdir()
    args = ["--out", str(b.store_root), "--cache", str(empty), "--snapshots", str(b.snapshots_root)]
    code, text = run_cli(["run", str(TOY_CONFIG), "--no-calibrate", "--resume", *args], capsys)
    assert code == bt.EXIT_REFUSED and "lacks the leverage" in text, text
    assert "Recalibrate them with: volsto-backtest run" in text and "--only-dates" in text, text
    assert full_tree(b.store_root) == before
    # a flat date computed under another config: refused (flat-aware), with the --force line
    done = b.store_root / "dates" / b.dates[2] / "done.json"
    doc = json.loads(done.read_text())
    doc["config_hash"] = "f" * 64
    done.write_text(json.dumps(doc))
    before = full_tree(b.store_root)
    code, text = run_cli(
        ["run", str(TOY_CONFIG), "--no-calibrate", "--only-dates", b.dates[3], *b.path_args()],
        capsys,
    )
    assert code == bt.EXIT_REFUSED and "another config hash" in text, text
    line = re.search(r"under this config with: (volsto-backtest run [^—]*--force)", text)
    assert line is not None and f"--only-dates {b.dates[3]}" in line.group(1), text
    assert full_tree(b.store_root) == before
    # the printed command proceeds: it migrates, then recomputes next to the foreign date
    argv = shlex.split(line.group(1))[1:]
    code, text = run_cli([*argv, "--no-calibrate"], capsys)
    assert code == 0 and "migrated 5 date(s)" in text, text


def test_round5_migration_is_journalled_resumable_and_rule_bound(
    build_copy: BacktestBuild,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 5 C (findings 5, 6, 7, 8): a migration interrupted after its first date keeps an
    ``in_progress`` journal entry with that date's report; the next open finishes it — every
    date reported, the snapshots bound (an interruption during the binding resumes too) and one
    completed entry.  A torn flat ``done.json`` does not block the migration: its bytes go into
    an attempt that the pointer rule never makes current (the date is missing, the others done);
    a flat date of another config is published, never current.  A write-protected flat store is
    a clear error (exit 1), not a traceback."""
    b = build_copy
    cfg = cfg_at(b)
    flat_store(b)
    torn = b.store_root / "dates" / b.dates[1] / "done.json"
    torn.write_bytes(torn.read_bytes()[: len(torn.read_bytes()) // 2])
    foreign = b.store_root / "dates" / b.dates[4] / "done.json"
    doc = json.loads(foreign.read_text())
    doc["config_hash"] = doc["record"]["config_hash"] = "e" * 64
    foreign.write_text(json.dumps(doc))
    calls = {"reported": 0}

    def hook(name: str) -> None:
        if name == "migrate.reported":
            calls["reported"] += 1
            raise SimulatedKill()

    monkeypatch.setattr(bt, "CRASH_HOOK", hook)
    with pytest.raises(SimulatedKill):
        bt.BacktestRun(cfg, allow_calibrate=False, migrate=True)
    journal = json.loads((b.store_root / "migrations.json").read_text())["migrations"]
    assert len(journal) == 1 and journal[0]["state"] == "in_progress", journal
    assert list(journal[0]["dates"]) == [b.dates[0]] and len(journal[0]["todo"]) == 5
    # interrupted again, in the snapshot binding (after two dates were bound)
    real_bind = bt.bind_snapshot
    bound: list[str] = []

    def bind_twice(*a: Any, **k: Any) -> None:
        if len(bound) == 2:
            raise SimulatedKill()
        real_bind(*a, **k)
        bound.append(str(a[1]))

    monkeypatch.setattr(bt, "CRASH_HOOK", None)
    monkeypatch.setattr(bt, "bind_snapshot", bind_twice)
    with pytest.raises(SimulatedKill):
        bt.BacktestRun(cfg, allow_calibrate=False, migrate=True)
    monkeypatch.setattr(bt, "bind_snapshot", real_bind)
    journal = json.loads((b.store_root / "migrations.json").read_text())["migrations"]
    assert len(journal) == 1 and journal[0]["state"] == "in_progress"
    assert sorted(journal[0]["dates"]) == list(b.dates)
    run = bt.BacktestRun(cfg, allow_calibrate=False, migrate=True)
    journal = json.loads((b.store_root / "migrations.json").read_text())["migrations"]
    assert len(journal) == 1 and journal[0]["state"] == "complete", journal
    assert sorted(journal[0]["dates"]) == list(b.dates)
    ok = [b.dates[0], b.dates[2], b.dates[3]]
    assert sorted(journal[0]["snapshots_bound"]) == ok, journal[0]
    verdicts = run.ledger.verdicts()
    assert {d: v.status for d, v in verdicts.items()} == {
        b.dates[0]: "done",
        b.dates[1]: "missing",  # its torn outcome is published, never current
        b.dates[2]: "pending",  # they wait for it
        b.dates[3]: "pending",
        b.dates[4]: "missing",  # another config's outcome: published, never current
    }, verdicts
    labels = [a.id.rsplit("-", 1)[0] for a in run.store.attempts(b.dates[1])]
    assert labels == ["legacy-unknown"], labels
    assert not (b.store_root / "dates" / b.dates[4] / "CURRENT").exists()
    assert [a.id.rsplit("-", 1)[0] for a in run.store.attempts(b.dates[4])] == ["legacy-ok"]
    assert not any(run.store.legacy_files(d) for d in b.dates)
    # F8: a write-protected flat store
    b2 = build_copy.copy(tmp_path / "ro")
    flat_store(b2)
    for p in [b2.store_root, *b2.store_root.rglob("*")]:
        if p.is_dir():
            p.chmod(0o555)
    try:
        code, text = run_cli(["status", str(TOY_CONFIG), *b2.path_args()], capsys)
        assert code == bt.EXIT_FAILED and "not writable" in text, text
        assert "Traceback" not in text and "PermissionError" not in text
    finally:
        for p in [b2.store_root, *b2.store_root.rglob("*")]:
            if p.is_dir():
                p.chmod(0o755)


def _migrate_worker(args: tuple[str, str]) -> int:
    cfg_file, tests_dir = args
    import sys

    sys.path.insert(0, tests_dir)
    from volsto.studies import backtest as bt_

    run = bt_.BacktestRun(bt_.load_backtest_config(cfg_file), allow_calibrate=False, migrate=True)
    return len(run.migration)


def test_round5_concurrent_migrations_are_serialised(
    build_copy: BacktestBuild, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 5 C (finding 4): three processes opening one flat store at once — the store lock
    serialises them: one migrates every date, the others find nothing to do; one journal entry,
    every date done, no flat file left."""
    import multiprocessing as mp

    b = build_copy
    cfg = cfg_at(b)
    flat_store(b)
    n3f_leftover(b)
    cfg_file = tmp_path / "abs.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg.to_mapping()))
    tests_dir = str(Path(__file__).parent)
    with mp.get_context("spawn").Pool(3) as pool:
        counts = pool.map(_migrate_worker, [(str(cfg_file), tests_dir)] * 3)
    assert sorted(counts) == [0, 0, 5], counts
    journal = json.loads((b.store_root / "migrations.json").read_text())["migrations"]
    assert [e["state"] for e in journal] == ["complete"]
    assert {v.status for v in bt.Ledger.of(cfg).verdicts().values()} == {"done"}
    assert not (b.store_root / ".staging").exists()


def test_round5_unsettled_foreign_pointers_metadata_and_the_lock(
    build_copy: BacktestBuild,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 5 D (findings 2, 3, 13, 22): retry exhaustion on a dependency is ``unsettled`` and
    never moves a pointer (a done date keeps its attempt against a new incomplete one); a
    ``CURRENT`` of a newer pointer version is refused by ``run``, ``gc``, adoption and the
    pointer writer, byte for byte untouched; the metadata ignore applies to regular files only;
    the date lock is a ``flock`` on the date directory — no lock file, a second holder of the
    same directory waits, and a holder of a swapped path never writes the other directory."""
    import fcntl

    b = build_copy
    cfg = cfg_at(b)
    run = bt.BacktestRun(cfg, allow_calibrate=False)
    d_prev, d = b.dates[2], b.dates[3]
    kept = current(b.store_root, d).name
    real = bt.Ledger._judge_once

    def churn(self: bt.Ledger, date: str) -> bt.Verdict:
        if date == d_prev:
            raise bt._Reread(date)
        return real(self, date)

    monkeypatch.setattr(bt.Ledger, "_judge_once", churn)
    monkeypatch.setattr(bt, "_reread_pause", lambda k: None)
    run.ledger.invalidate()
    assert run.ledger.verdict(d_prev).status == "unsettled"
    v = run.ledger.verdict(d)
    assert v.status == "unsettled" and d_prev in v.reason, v
    rows = pd.read_parquet(current(b.store_root, d, "rows.parquet"))
    fit = current_json(b.store_root, d, "fit.json")
    done = current_json(b.store_root, d, "done.json")
    done.pop("record")
    done.update(status="incomplete", incomplete_reasons=["simulated"], created_utc="z#r5")
    _, _, is_current = run._commit(d, rows, fit, done, prev_used=True)
    assert not is_current and current(b.store_root, d).name == kept
    monkeypatch.setattr(bt.Ledger, "_judge_once", real)
    run.ledger.invalidate()
    assert run.ledger.verdict(d).status == "done"
    # F13: a CURRENT of pointer version 99
    ptr_path = b.store_root / "dates" / b.dates[1] / "CURRENT"
    ptr = json.loads(ptr_path.read_text())
    ptr_path.write_text(json.dumps({**ptr, "version": 99}))
    foreign_bytes = ptr_path.read_bytes()
    state = full_tree(b.store_root)
    v = bt.Ledger.of(cfg).verdict(b.dates[1])
    assert v.status == "stale" and v.foreign and "pointer version 99" in v.reason, v
    code, text = run_cli(["run", str(TOY_CONFIG), "--no-calibrate", *b.path_args()], capsys)
    assert code == bt.EXIT_REFUSED and "another pointer version" in text, text
    code, text = run_cli(["gc", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == bt.EXIT_REFUSED and "another pointer version" in text, text
    assert full_tree(b.store_root) == state
    with pytest.raises(bt.RefusedError):
        bt.adopt(run.store, bt.Ledger.of(cfg), [b.dates[1]])
    att = run.store.attempts(b.dates[1])[0]
    with run.store.lock(b.dates[1]) as held, pytest.raises(bt.RefusedError):
        run.store.set_pointer(held, att, "ok", att.listing())
    assert ptr_path.read_bytes() == foreign_bytes
    with pytest.raises(runner.MissingRequirements) as info:
        run_stage2(b, tmp_path / "foreign", latex_check=False)
    assert any("use the volsto-backtest that wrote it" in c for c in info.value.commands)
    ptr_path.write_text(json.dumps(ptr))
    # F22: directories named like metadata are never ignored
    a29 = current(b.store_root, b.dates[2])
    (a29 / ".DS_Store").mkdir()
    v = bt.Ledger.of(cfg).verdict(b.dates[2])
    assert v.status == "stale" and ".DS_Store/" in v.reason, v
    (a29 / ".DS_Store").rmdir()
    (a29 / "._rows.parquet").mkdir()
    assert bt.Ledger.of(cfg).verdict(b.dates[2]).status == "stale"
    (a29 / "._rows.parquet").rmdir()
    (a29 / ".DS_Store").write_bytes(b"\x00")
    assert bt.Ledger.of(cfg).verdict(b.dates[2]).status == "done"
    # F2: no lock file; the lock is the directory's
    assert sorted(p.name for p in (b.store_root / "dates" / d).iterdir()) == ["CURRENT", "attempts"]
    date_dir = b.store_root / "dates" / d
    with run.store.lock(d) as held:
        fd = os.open(date_dir, os.O_RDONLY)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)
        # a swapped path: the holder keeps mutating ITS directory (moved), never the new one
        moved = date_dir.with_name(d + ".moved")
        os.rename(date_dir, moved)
        date_dir.mkdir()
        with run.store.lock(d) as other:  # another inode: not excluded, and not shared
            assert other.names() == []
            held.unlink(bt.CURRENT_NAME)
            assert not (moved / bt.CURRENT_NAME).exists() and other.names() == []
    shutil.rmtree(date_dir)
    os.rename(moved, date_dir)


def test_round5_records_name_what_was_used(
    build_copy: BacktestBuild,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 5 E (findings 12, 15): a new record (version 2) carries the leverage's numeric
    content digest; new numbers under the same key make the date and the later dates stale,
    new metadata does not.  The snapshot digest recorded is that of the bytes the run parsed: a
    snapshot edited between the marking and the commit leaves the new attempt stale (it names
    the parsed bytes), never done; a leverage whose numbers change while the date uses it fails
    the date."""
    b = build_copy
    cfg = cfg_at(b)
    d = b.dates[1]
    doc = current_json(b.store_root, d, "done.json")
    rec = doc["record"]
    key = rec["leverage"]["key"]
    npz = b.cache_root / key / "leverage.npz"
    assert rec["version"] == 2 and rec["leverage"]["content"] == bt.leverage_content_digest(npz)
    original = npz.read_bytes()

    def rewrite(values_factor: float, meta: str = "") -> None:
        with np.load(io.BytesIO(original), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["values"] = arrays["values"] * values_factor
        if meta:
            arrays["metadata"] = np.array(json.dumps({"note": meta}))
        with open(npz, "wb") as fh:
            np.savez_compressed(fh, **arrays)

    rewrite(1.0, meta="recalibrated, same numbers")
    assert {v.status for v in bt.Ledger.of(cfg).verdicts().values()} == {"done"}
    rewrite(1.0 + 1e-9)
    verdicts = bt.Ledger.of(cfg).verdicts()
    assert verdicts[b.dates[0]].status == "done"
    assert "leverage" in verdicts[d].reason and verdicts[d].status == "stale", verdicts[d]
    assert all(verdicts[e].status == "stale" for e in b.dates[2:])
    npz.write_bytes(original)
    assert {v.status for v in bt.Ledger.of(cfg).verdicts().values()} == {"done"}
    # F15: the snapshot changes after the date parsed it
    snap = b.snapshots_root / f"spx_{b.dates[4]}.yaml"
    parsed = bt.snapshot_digest(snap)
    real_commit = bt.BacktestRun._commit

    def edit_then_commit(self: bt.BacktestRun, date: str, *a: Any, **k: Any) -> Any:
        if date == b.dates[4] and k.get("used") is not None:
            text = snap.read_text()
            n = yaml.safe_load(text)["provenance"]["n_points"]
            snap.write_text(text.replace(f"  n_points: {n}\n", f"  n_points: {n + 1}\n", 1))
        return real_commit(self, date, *a, **k)

    monkeypatch.setattr(bt.BacktestRun, "_commit", edit_then_commit)
    run = bt.BacktestRun(cfg, allow_calibrate=False)
    with guard.calibration_forbidden():
        out = run.run_date(b.dates[4])
    attempt = run.store.attempt(b.dates[4], out.attempt)
    assert json.loads((attempt.path / "done.json").read_text())["record"]["snapshot"] == parsed
    assert bt.snapshot_digest(snap) != parsed
    assert run.ledger.candidate(b.dates[4], attempt).status == "stale"
    monkeypatch.setattr(bt.BacktestRun, "_commit", real_commit)
    # the leverage numbers change while the date is being priced
    real_ensure = bt.BacktestRun.ensure_leverage

    def ensure_then_change(self: bt.BacktestRun, st: Any) -> tuple[bool, float]:
        got = real_ensure(self, st)
        if st.date == b.dates[3]:
            p = b.cache_root / st.key / "leverage.npz"
            with np.load(p, allow_pickle=False) as z:
                arrays = {k: z[k] for k in z.files}
            arrays["values"] = arrays["values"] * (1.0 + 1e-9)
            with open(p, "wb") as fh:
                np.savez_compressed(fh, **arrays)
        return got

    monkeypatch.setattr(bt.BacktestRun, "ensure_leverage", ensure_then_change)
    run = bt.BacktestRun(cfg, allow_calibrate=False)
    with guard.calibration_forbidden():
        out = run.run_date(b.dates[3])
    assert out.status == "failed" and "changed while it was used" in out.error, out


def test_round5_skip_date_needs_an_absent_day_file(
    build_copy: BacktestBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Round 5 F (finding 9): under ``skip_date`` an unreadable day file that exists fails its
    date (exit 1, retried by ``--resume`` once the file is fixed) — never a calendar gap (the
    absent-file skip and its re-derived cause: the skip_date test above)."""
    b = build_copy
    days = list(b.dates[:3])
    data = raw_config()
    root = hdn_root(tmp_path, days, corrupt=[days[1]])
    data["data"]["root"] = str(root)
    data["data"]["missing_close"] = "skip_date"
    cfg = write_config(tmp_path, data)
    out = tmp_path / "transient"
    code, text = run_cli(["run", str(cfg), "--no-calibrate", *b.path_args(out=out)], capsys)
    assert code == bt.EXIT_FAILED and "dropped" not in text, text
    assert current_json(out, days[1], "done.json")["status"] == "failed"
    assert "exists, so the date is failed, not skipped" in text, text
    day = root / "day_by_date" / f"{days[1]}_options.csv"
    day.unlink()
    day.symlink_to(
        ROOT / "data" / "hdn_sample" / "options_sample_2022H2" / "day_by_date" / day.name
    )
    code, text = run_cli(
        ["run", str(cfg), "--no-calibrate", "--resume", *b.path_args(out=out)], capsys
    )
    assert code == 0 and "2 date(s) ok" in text, text
    code, text = run_cli(["status", str(cfg), *b.path_args(out=out)], capsys)
    assert code == 0 and "done 3" in text and "skipped" not in text, text


def test_projection_prints_before_any_work(
    build_copy: BacktestBuild,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    b = build_copy
    seen: list[str] = []
    original = bt.BacktestRun.run_date

    def spy(self: bt.BacktestRun, date: str) -> bt.DateOutcome:
        seen.append(capsys.readouterr().out)
        return original(self, date)

    monkeypatch.setattr(bt.BacktestRun, "run_date", spy)
    out = tmp_path / "one"
    code = bt.main(
        [
            "run",
            str(TOY_CONFIG),
            "--no-calibrate",
            "--only-dates",
            b.dates[0],
            *b.path_args(out=out),
        ]
    )
    assert code == 0
    assert seen and "PROJECTED WALL CLOCK" in seen[0], seen
    assert "TOTAL (one process)" in seen[0] and "leverage calibration" in seen[0]
    # D6: --no-calibrate charges no calibration
    assert "--no-calibrate: a missing leverage fails its date" in seen[0]
    # dry-run: the projection with the 1 / 4 / 8 shard lines, and no date written
    code, text = run_cli(
        ["dry-run", str(TOY_CONFIG), "--shard", "2/2", *b.path_args(out=tmp_path / "dry")], capsys
    )
    assert code == 0, text
    for k in ("1 shard(s)", "2 shard(s)", "4 shard(s)", "8 shard(s)", "model builds", "strips"):
        assert k in text, k
    assert "dry run: nothing computed" in text
    assert not (tmp_path / "dry" / "dates").exists()


# --------------------------------------------------------------------------------------------
# stage 2: the study through the runner
# --------------------------------------------------------------------------------------------


def run_stage2(
    build: BacktestBuild,
    out: Path,
    *,
    outputs: Path | None = None,
    backtest: Path | None = None,
    store: str | None = None,
    **kwargs: Any,
) -> runner.StudyRun:
    """The fast study through the runner on the build (or on ``backtest`` / ``store`` under
    ``outputs``), with the build's cache."""
    sets = tuple(f"{k}={v}" for k, v in (("backtest", backtest), ("store", store)) if v is not None)
    overrides = runner.ConfigOverrides(
        outputs=str(outputs if outputs is not None else build.outputs_root),
        cache=str(build.cache_root),
        store=str(build.base / "st"),
        sets=sets,
    )
    config = runner.load_study_config(STUDY_FAST, overrides)
    return runner.run_study(config, out_dir=out, config_path=STUDY_FAST, **kwargs)


def test_stage2_renders_from_the_stored_rows_without_calibrating(
    build_copy: BacktestBuild, tmp_path: Path
) -> None:
    b = build_copy
    cache_before = tree_state(b.cache_root)
    store_before = tree_state(b.store_root)
    run = run_stage2(b, tmp_path / "study")
    m = run.manifest
    assert run.exit_code == 0, m.get("latex")
    assert m["recalibrated"] is False and m["mode"] == "fast"
    assert m["calibration_refusals"] == 0
    assert tree_state(b.cache_root) == cache_before
    assert tree_state(b.store_root) == store_before
    assert set(m["particles"]) == {20_000}
    fits = store_fits(b.store_root)
    assert {k["key"] for k in m["cache_keys"]} == {f["cache_key"] for f in fits.values()}
    assert len(m["artefacts"]) == len(b.dates)
    assert (
        m["records"]["backtest_config_hash"] == bt.load_backtest_config(TOY_CONFIG).content_hash()
    )
    out = run.out_dir
    for t in (
        "trades_pct_notional",
        "inception_vol_pts",
        "pnl_trade_pct_notional",
        "pnl_trade_pct_spot",
        "pnl_trade_vol_pts",
        "pnl_month_vol_pts",
        "pnl_bucket",
        "ssr",
        "vko",
        "params_flags",
    ):
        assert t in m["tables"] and (out / "tables" / f"{t}.tex").is_file(), t
    for f in ("book_pnl", "trade_pnl", "explained", "params", "ssr", "vko"):
        assert (out / "figures" / f"{f}.pdf").is_file() and (out / "figures" / f"{f}.png").is_file()
    text = (out / "study.md").read_text()
    assert "proof of concept" in text and "too short for a conclusion" in text
    assert "needs only a config change" in text and "Fast mode" in text
    assert "not observable in the window" in text
    assert "variance notional 1/(2 K_vol)" in text and "strike 0 in variance units" in text
    assert "inception surface's log-contract strike" in text
    assert "marked on every date after inception" in text  # the toy marks its rolling book daily
    # D5: the window is described from the data, without a regime adjective
    assert "bear market" not in text and "2022 H2 sample" not in text
    closes = [store_fits(b.store_root)[d]["close"] for d in b.dates]
    assert f"went from {closes[0]:.2f} to {closes[-1]:.2f}" in text
    assert "realised volatility of" in text and "SPX day files under" in text
    # (c) the stability fit is stated; (a) the desk sign is stated and used; D10 checks
    assert "stability.fit" in text and "skew_mode soft" in text
    assert "desk P&L = -(V(d) - V(d-1)" in text and "Every stored date carries its P&L." in text
    assert "MTM" not in text and all("MTM" not in f.caption for f in bt.figures(run.results))
    tex = (out / "tables" / "pnl_trade_pct_notional.tex").read_text()
    assert "desk P\\&L [\\% notional]" in tex and "$-0 " not in tex
    for month in run.results.rows("pnl_month"):
        assert abs(run.results.value("pnl_month", month, "check")[0]) < 1e-10
    assert not re.search(r"\bnan\b", text, flags=re.IGNORECASE)
    # each unit's book desk P&L is minus the sum of the stored (holder) daily P&L of its trades
    rows = store_rows(b.store_root)
    for unit in ("% notional", "vol pts (vega notional 1)"):
        fixed = rows[(rows["book"] == "fixed") & rows["pnl"].notna() & (rows["unit"] == unit)]
        v, se = run.results.value("pnl_trade", bt.book_row(unit), "pnl")
        assert v == pytest.approx(-100.0 * fixed["pnl"].sum(), rel=1e-12)
        rss = 100.0 * math.sqrt((fixed["pnl_stderr"] ** 2).sum())
        assert se == pytest.approx(rss, rel=1e-12)
    # tables, figures and study.md are functions of results.parquet alone
    copy = tmp_path / "render"
    shutil.copytree(out, copy)
    shutil.rmtree(copy / "tables")
    shutil.rmtree(copy / "figures")
    _, code = runner.render_study(copy, latex_check=False)
    assert code == 0
    for t in m["tables"]:
        assert (copy / "tables" / f"{t}.tex").read_bytes() == (
            out / "tables" / f"{t}.tex"
        ).read_bytes()
    assert (copy / "study.md").read_text() == text


def test_stage2_missing_date_is_a_requirement_with_the_backtest_command(
    build_copy: BacktestBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    b = build_copy
    gone = b.dates[2]
    (b.store_root / "dates" / gone / "CURRENT").unlink()
    code = runner.main(
        [
            "run",
            str(STUDY_FAST),
            "--out",
            str(tmp_path / "study"),
            "--outputs",
            str(b.outputs_root),
            "--cache",
            str(b.cache_root),
            "--no-latex-check",
        ]
    )
    text = capsys.readouterr()
    printed = text.out + text.err
    assert code == runner.EXIT_MISSING, printed
    assert "volsto-backtest run" in printed and f"--only-dates {gone}" in printed
    assert "--resume" in printed and "--out" in printed
    assert not (tmp_path / "study").exists()


# --------------------------------------------------------------------------------------------
# what the rows say
# --------------------------------------------------------------------------------------------


def test_attribution_buckets_sum_to_the_mtm_pnl(
    toy_backtest_build: BacktestBuild,  # noqa: F811
) -> None:
    b = toy_backtest_build.require()
    rows = store_rows(b.store_root)
    att = rows[rows["pnl_method"] == "attributed"]
    toy = bt.load_backtest_config(TOY_CONFIG)
    assert len(att) == len(toy.fixed) * (len(b.dates) - 1)
    greek_cols = [bt.bucket_column(x) for x in bt.GREEK_BUCKETS]
    total = (
        att[greek_cols].sum(axis=1)
        + att[bt.bucket_column("recalibration")]
        + att[bt.bucket_column("residual")]
        + att[bt.bucket_column("cash_flows")]
        + att[bt.bucket_column("settlement")]
        + att[bt.bucket_column("unattributed")]
    )
    np.testing.assert_allclose(total, att["pnl"], rtol=0, atol=1e-14)
    # the steps telescope to V(d) - V(d-1); the explained part of the recalibration step is its actual
    steps = att[[f"s_{s}" for s in bt.STEPS]].fillna(0.0).sum(axis=1)
    np.testing.assert_allclose(steps, att["value"] - att["price_0"], rtol=0, atol=1e-13)
    np.testing.assert_allclose(
        att["s_recalibration"], att[bt.bucket_column("recalibration")], atol=0
    )
    # the start-of-period Greeks cost no pricing; the cumulative P&L one per trade from the
    # second day after inception (the day after is a memo hit)
    assert (att["extra_pricings"] == (att["age"] >= 2).astype(float)).all()
    assert (att["pnl_stderr"] > 0).all()
    for c in [*greek_cols, bt.bucket_column("recalibration"), bt.bucket_column("residual")]:
        assert np.isfinite(att[c + "_stderr"]).all() and (att[c + "_stderr"] >= 0).all(), c
    for g in ("delta", "gamma", "vega", "rho", "repo", "theta", "theta_decay", "theta_carry"):
        assert np.isfinite(att[f"g_{g}"]).all() and (att[f"g_{g}_stderr"] >= 0).all(), g
    # the previous value the attribution prices is the stored one (one seed, one model)
    by = rows.set_index(["trade_id", "date"])
    for _, r in rows[rows["price_0"].notna()].iterrows():
        prev = b.dates[b.dates.index(r["date"]) - 1]
        assert r["price_0"] == by.loc[(r["trade_id"], prev), "value"], (r["trade_id"], r["date"])
    # the paired P&L of the rolling trade equals the fixed twin's attributed P&L
    twin = rows[rows["trade_id"] == "fixed:var_swap_6m"].set_index("date")["pnl"]
    roll = rows[rows["trade_id"] == "rolling:2022-07-27:var_swap_6m"].set_index("date")
    assert (roll["pnl_method"].iloc[1:] == "paired").all()
    np.testing.assert_allclose(roll["pnl"].iloc[1:], twin.iloc[1:], rtol=1e-12, atol=1e-15)


def test_inception_value_is_the_fresh_price_under_the_same_seed(
    build_copy: BacktestBuild,
) -> None:
    b = build_copy
    toy = bt.load_backtest_config(TOY_CONFIG).with_paths(
        out=b.store_root, cache=b.cache_root, snapshots=b.snapshots_root
    )
    rows = store_rows(b.store_root)
    with guard.calibration_forbidden():
        run = bt.BacktestRun(toy, allow_calibrate=False)
        checked = 0
        for inception in ("2022-07-27", "2022-08-01"):
            st = run.state(inception)
            engine = RiskEngine(LSVBuilder(run.cache, st.state, allow_calibrate=False), run.sim)
            first = rows[(rows["date"] == inception) & (rows["inception"] == inception)]
            for _, r in first.iterrows():
                spec = next(
                    t for t in (*toy.fixed, *toy.rolling) if r["trade_id"].endswith(f":{t.id}")
                )
                # struck at the close and seasoned with the inception fixing (the close), as the
                # run prices it; the model starts at the option-implied spot (SPEC §13.1)
                built = bt.build_product(spec, st.close, st.surface, st.discount).product
                fresh = replay(
                    built,
                    run.history(inception, inception),
                    dt.date.fromisoformat(inception),
                    discount=st.discount,
                ).result
                assert isinstance(fresh, Product)
                price = engine.price(fresh, st.state)
                assert float(price.mean) == r["value"], r["trade_id"]
                assert float(price.stderr) == r["value_stderr"]
                checked += 1
    assert checked == len(toy.fixed) + 2


def test_vko_row_carries_the_realised_state(
    toy_backtest_build: BacktestBuild,  # noqa: F811
) -> None:
    b = toy_backtest_build.require()
    rows = store_rows(b.store_root)
    vko = rows[rows["kind"] == "vko_put"].sort_values("date")
    assert len(vko) == len(b.dates)
    closes = []
    for d in b.dates:
        snap = yaml.safe_load((b.snapshots_root / f"spx_{d}.yaml").read_text())
        closes.append(float(snap["market"]["close"]))  # the fixing level, not the implied spot
    r2 = np.diff(np.log(closes)) ** 2
    for i, (_, r) in enumerate(vko.iterrows()):
        state = json.loads(r["realised_json"])
        assert r["vol_ko"] == 0.30 and r["strike"] == pytest.approx(closes[0], rel=1e-15)
        if i == 0:  # the trade-date close is fixing 0: nothing realised yet
            assert r["status"] == "inception" and state["realised_returns"] == 0
            assert state["realised_sum_sq"] == 0.0 and math.isnan(r["realised_vol"])
            continue
        assert state["realised_returns"] == i == int(r["realised_returns"])
        assert state["realised_sum_sq"] == pytest.approx(float(r2[:i].sum()), rel=1e-12)
        assert r["realised_vol"] == pytest.approx(math.sqrt(252.0 * r2[:i].sum() / i), rel=1e-12)
        assert state["knocked_out"] is False and r["knocked_out"] == 0.0
        assert state["variance_budget"] > state["realised_sum_sq"]
        assert r["status"] == "live" and r["value"] > 0


def test_fitted_parameter_frame_feeds_flag_unidentified(
    toy_backtest_build: BacktestBuild,  # noqa: F811
) -> None:
    b = toy_backtest_build.require()
    fits = store_fits(b.store_root)
    assert sorted(fits) == list(b.dates)
    frame = pd.DataFrame(
        [
            {
                **{k: f["breakeven"][k] for k in PARAM_COLUMNS},
                **{f"{k}_se": np.nan if f["se"][k] is None else f["se"][k] for k in PARAM_COLUMNS},
            }
            for f in fits.values()
        ]
    )
    flags = flag_unidentified(frame)
    assert list(flags["param"]) == list(PARAM_COLUMNS)
    # the finding: marking fits carry no standard error, so no parameter can be flagged
    assert (flags["n_without_se"] == len(b.dates)).all()
    assert flags["share_beyond_se"].isna().all() and not flags["unidentified"].any()
    assert (flags["n_changes"] == 0).all()
    for f in fits.values():
        assert f["status"] in ("interior", "binding")
        assert set(f["params"]) == {"nu", "theta", "k1", "k2", "rho12", "rho_SX1", "rho_SX2"}
        assert f["calendar"]["calendar_fallback"] is None
        assert f["surface_type"] == "ESSVISurface"
        assert len(f["history"]["vs_vol"]) == len(DEFAULT_PILLARS)


def test_realised_ssr_is_nan_with_a_reason_before_enough_history(
    build_copy: BacktestBuild, tmp_path: Path
) -> None:
    run = run_stage2(build_copy, tmp_path / "study", latex_check=False)
    res: Results = run.results
    for d in build_copy.dates:
        for T in ("0.25", "1"):
            rec = res.record("ssr_daily", f"{d}|{T}", "realised")
            assert math.isnan(rec["value"]) and math.isnan(rec["stderr"])
            assert "needs 21 dates (20 increments)" in rec["note"]
    assert "fewer than 21 dates" in (run.out_dir / "study.md").read_text()
    rec = res.record("stability_hist", "k1", "share_beyond_se")
    assert math.isnan(rec["value"]) and "needs 22 dates" in rec["note"]


def test_round5_stage2_commands_notes_and_pins(
    build_copy: BacktestBuild,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 5 B and G (findings 10, 11, 17): after another config recomputed one date, stage 2
    prints ONE ``run`` line with ``--force`` for that date and every date depending on it — and
    that line runs; a "later attempt failed" note comes from verified bytes only and never
    changes a classification; the manifest pins the ``CURRENT`` bytes ``compute()`` read, not
    those ``requirements()`` saw."""
    b = build_copy
    other = other_config(tmp_path)
    code, text = run_cli(
        [
            "run",
            str(other),
            "--no-calibrate",
            "--force",
            "--only-dates",
            b.dates[2],
            *b.path_args(),
        ],
        capsys,
    )
    assert code == 0, text
    with pytest.raises(runner.MissingRequirements) as info:
        run_stage2(b, tmp_path / "mixed", latex_check=False)
    runs = [c for c in info.value.commands if c.startswith("volsto-backtest run")]
    assert len(runs) == 1 and runs[0].endswith("--force"), info.value.commands
    assert f"--only-dates {' '.join(b.dates[2:])} " in runs[0], runs[0]
    # it runs: here the pointer rule adopts 07-29's earlier attempt of this config (the others
    # then verify again), so nothing is recomputed
    code, text = run_cli([*shlex.split(runs[0])[1:], "--no-calibrate"], capsys)
    assert code == 0 and f"adopted a published attempt for 1 date(s): {b.dates[2]}" in text, text
    assert run_stage2(b, tmp_path / "healed", latex_check=False).exit_code == 0
    # F11: a failed attempt beside the results — its note only when its files verify
    cfg = cfg_at(b)
    run = bt.BacktestRun(cfg, allow_calibrate=False)
    run._write_failure(b.dates[1], "simulated failure", 0.0, [], {})
    v = bt.Ledger.of(cfg).verdict(b.dates[1])
    assert v.status == "done" and "later attempt failed" in v.note and v.reason == "", v
    failed = next(a for a in run.store.attempts(b.dates[1]) if a.id.startswith("failed-"))
    doc = json.loads((failed.path / "done.json").read_text())
    doc["error"] = "tampered"
    (failed.path / "done.json").write_text(json.dumps(doc))
    v = bt.Ledger.of(cfg).verdict(b.dates[1])
    assert v.status == "done" and v.note == "", v
    # F17: CURRENT changes after requirements() hashed it and before compute() reads it (the
    # runner pins what requirements() saw unless compute() assigns what it read)
    ptr = b.store_root / "dates" / b.dates[4] / "CURRENT"
    before = hashlib.sha256(ptr.read_bytes()).hexdigest()
    real = bt.compute

    def recommit_then_compute(ctx: Any) -> Any:
        recommit(bt.BacktestRun(cfg, allow_calibrate=False), b.dates[4], "between")
        return real(ctx)

    monkeypatch.setattr(bt, "compute", recommit_then_compute)
    res = run_stage2(b, tmp_path / "pinned", latex_check=False)
    manifest = json.loads((res.out_dir / "manifest.json").read_text())
    pinned = {a["path"]: a["sha256"] for a in manifest["artefacts"]}
    after = hashlib.sha256(ptr.read_bytes()).hexdigest()
    key = f"{_backtest_build.TOY_BACKTEST_STORE}/dates/{b.dates[4]}/CURRENT"
    assert after != before and pinned[key] == after, (before, after, pinned[key])


def test_round5_status_gc_help_and_the_path_guard(
    build_copy: BacktestBuild,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 5 H (findings 16, 20, 21): ``status`` hides done dates without ``-v`` and lists
    the staging directories of other hosts with a hint; ``gc`` removes a dead local writer's and
    keeps (and counts) another host's; every subcommand and argument has a help text, the exit
    codes included; ``VOLSTO_BACKTEST_REQUIRE_PATHS=1`` refuses a command without the path
    overrides before touching anything."""
    b = build_copy
    code, text = run_cli(["status", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == 0 and "done 5" in text and "listed with -v" in text, text
    assert b.dates[0] not in text.split("\n", 1)[1], text
    code, text = run_cli(["status", str(TOY_CONFIG), "-v", *b.path_args()], capsys)
    assert all(d in text for d in b.dates), text
    attempts = b.store_root / "dates" / b.dates[0] / "attempts"
    foreign = attempts / ".staging-otherhost-4242-abcd"
    foreign.mkdir()
    dead = attempts / f"{bt.STAGING_PREFIX}{bt._host_tag()}-999999-dead"
    dead.mkdir()
    code, text = run_cli(["status", str(TOY_CONFIG), *b.path_args()], capsys)
    assert "1 staging directory(ies) of writers on other hosts (otherhost)" in text, text
    assert "gc here keeps them" in text and "1 staging directory(ies) of finished" in text, text
    code, text = run_cli(["gc", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == 0 and "1 staging directory(ies) of finished writers on this host" in text
    assert "1 staging directory(ies) of running writers or of other hosts" in text, text
    assert foreign.is_dir() and not dead.exists()
    # F21: help texts
    parser = bt.build_parser()
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    assert set(sub.choices) == {"run", "dry-run", "status", "migrate", "gc"}
    for name, q in sub.choices.items():
        assert q.description and "exit codes" in (q.epilog or ""), name
        for action in q._actions:
            if not isinstance(action, argparse._HelpAction):
                assert action.help, (name, action.dest)
    assert "done dates only with -v" in sub.choices["status"].description
    assert all(
        bt.REQUIRE_PATHS_ENV in (q.epilog or "") for q in sub.choices.values()
    ), "the path guard is documented in every --help"
    # F16: the opt-in path guard, blank overrides included (follow-up 0)
    work = tmp_path / "cwd"
    work.mkdir()
    monkeypatch.chdir(work)
    with pytest.raises(ConfigError, match="empty path"):
        bt.load_backtest_config(TOY_CONFIG).with_paths(out="  ")
    blank = ["--out", "", "--cache", str(b.cache_root), "--snapshots", str(b.snapshots_root)]
    code, text = run_cli(["run", str(TOY_CONFIG), "--no-calibrate", *blank], capsys)
    assert code == bt.EXIT_FAILED and "empty path" in text, text
    monkeypatch.setenv(bt.REQUIRE_PATHS_ENV, "1")
    code, text = run_cli(["status", str(TOY_CONFIG), "--out", str(b.store_root)], capsys)
    assert code == bt.EXIT_REFUSED and "--cache, --snapshots" in text, text
    code, text = run_cli(["run", str(TOY_CONFIG), "--no-calibrate", *blank], capsys)
    assert code == bt.EXIT_REFUSED and "pass --out" in text, text
    assert list(work.iterdir()) == []
    code, text = run_cli(["status", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == 0, text


def test_round5_cumulative_pnl_units_and_labels(
    build_copy: BacktestBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Round 5 P1 and P2 (and the review's unit, explained and flag findings): every attributed
    row stores V(d) - V(inception) + flows with its direct paired stderr at one extra pricing
    (none the day after inception, a memo hit); it telescopes to the sum of the daily P&L; the
    study takes it as the trade's cumulative error and labels every other aggregate as a root sum
    of squares of correlated errors; no number is in a mixed unit (the books add only trades of
    one unit, per-trade rows carry their own unit, tables and figures split by unit); the daily
    explained sum is a Monte Carlo value with its error; an unflaggable parameter's flag is
    NaN; the projection charges the cumulative pricing."""
    b = build_copy
    rows = store_rows(b.store_root)
    att = rows[rows["pnl_method"] == "attributed"]
    assert len(att) and att["cum_pnl"].notna().all() and att["cum_pnl_stderr"].gt(0).all()
    for col in (bt.group_column(g) for g in ("spot", "greeks", "explained")):
        assert att[col].notna().all(), col
    assert (att["residual_paired"] == 1.0).all()
    for tid, g in att.groupby("trade_id"):
        g = g.sort_values("date")
        first = g.iloc[0]
        assert first["age"] == 1 and first["extra_pricings"] == 0.0, tid
        assert first["cum_pnl"] == pytest.approx(first["pnl"], abs=1e-12)
        assert first["cum_pnl_stderr"] == pytest.approx(first["pnl_stderr"], rel=1e-12)
        assert (g["extra_pricings"].iloc[1:] == 1.0).all(), tid
        last = g.iloc[-1]
        daily = rows[(rows["trade_id"] == tid) & rows["pnl"].notna()]
        assert last["cum_pnl"] == pytest.approx(daily["pnl"].sum(), abs=1e-10), tid
        rss = math.sqrt(float((daily["pnl_stderr"] ** 2).sum()))
        print(f"{tid}: cumulative stderr paired {last['cum_pnl_stderr']:.3g}, RSS {rss:.3g}")
    study = run_stage2(b, tmp_path / "study", latex_check=False)
    res = study.results
    kinds = rows.groupby("trade_id")["kind"].first()
    units = {str(tid): bt.TRADE_UNITS[str(k)][0] for tid, k in kinds.items()}
    frame = res.frame
    assert "x100" not in set(frame["unit"]) and not (frame["row"] == "book (fixed)").any()
    for table in ("pnl_trade", "trades"):
        long = res.long(table)
        for row, unit in zip(long["row"], long["unit"]):
            if row in units and unit:
                assert unit == units[row], (table, row, unit)
    pnl = res.long("pnl_trade", "pnl")
    for row, note in zip(pnl["row"], pnl["note"]):
        if row.startswith("fixed:"):
            assert bt.SE_PAIRED_CUMULATIVE in note, (row, note)
        elif row.startswith("book ("):
            assert bt.SE_RSS_DATES in note, (row, note)
    books = [r for r in res.rows("pnl_trade") if r.startswith("book (")]
    assert set(books) == {bt.book_row("% notional"), bt.book_row("vol pts (vega notional 1)")}
    for row in books:
        v, _ = res.value("pnl_trade", row, "pnl")
        unit = row[len("book (fixed, ") : -1]
        members = [t for t, u in units.items() if u == unit and t.startswith("fixed:")]
        assert v == pytest.approx(sum(res.value("pnl_trade", t, "pnl")[0] for t in members))
    bucket = res.long("pnl_bucket")
    assert all(bt.SE_RSS_DATES in n for n in bucket["note"]), set(bucket["note"])
    explained = res.long("daily_trade", "explained")
    assert not explained["exact"].any() and explained["stderr"].gt(0).all()
    rolling = [r for r in res.rows("pnl_trade") if r.startswith("rolling:")]
    assert rolling
    for r in rolling:  # nothing attributed: a missing number with its reason, not 0 +/- 0
        rec = res.record("pnl_trade", r, "explained")
        assert math.isnan(rec["value"]) and "not attributed" in rec["note"], rec
    flags = res.long("params_flags", "unidentified")
    assert flags["value"].isna().all() and flags["note"].str.contains("cannot flag").all()
    names = [t.name for t in bt.tables(res)]
    assert {"pnl_trade_pct_notional", "pnl_trade_pct_spot", "pnl_trade_vol_pts"} <= set(names)
    assert "pnl_trade" not in names and "trades" not in names
    md = (study.out_dir / "study.md").read_text()
    assert "not the error of the cumulative P&L" in md and "no total adds two units" in md
    assert "calendar-repaired" in md and "one-line" not in md
    # an independent recomputation of one stored cumulative error: the last date of the
    # autocall, its value and the fresh trade at inception paired on a new engine
    tid = "fixed:autocall_6m"
    last = att[att["trade_id"] == tid].sort_values("date").iloc[-1]
    assert last["age"] >= 2
    fresh = bt.BacktestRun(cfg_at(b), allow_calibrate=False)
    trade = next(t for t in fresh.trades if t.trade_id == tid)
    date = str(last["date"])
    with guard.calibration_forbidden():
        st, inc = fresh.state(date), fresh.state(trade.inception)
        built = fresh.built(trade)
        p_d = replay(
            built.product,
            fresh.history(trade.inception, date),
            dt.date.fromisoformat(date),
            discount=st.discount,
        ).result
        p_inc = replay(
            built.product,
            fresh.history(trade.inception, trade.inception),
            dt.date.fromisoformat(trade.inception),
            discount=inc.discount,
        ).result
        engine = RiskEngine(LSVBuilder(fresh.cache, st.state, allow_calibrate=False), fresh.sim)
        pair = engine.paired(
            "cum_pnl",
            [(p_d, st.state, "recalibrate", 1.0), (p_inc, inc.state, "recalibrate", -1.0)],
            unit="price",
            size=0.0,
            scheme="revaluation",
        )
    assert last["cum_pnl_stderr"] == pytest.approx(pair.stderr, rel=1e-9)
    assert last["cum_pnl"] - last["flows_cum"] == pytest.approx(pair.value, rel=1e-9, abs=1e-12)
    # the stage-2 aggregation rules on their own
    one = att[att["trade_id"] == att["trade_id"].iloc[0]].sort_values("date")
    agg = bt.aggregate(one, "pnl")
    assert agg.note == bt.SE_PAIRED_CUMULATIVE and agg.stderr == one["cum_pnl_stderr"].iloc[-1]
    gap = bt.aggregate(one.iloc[[0, -1]], "pnl") if len(one) > 2 else agg
    assert len(one) <= 2 or gap.note == bt.SE_RSS_DATES
    old = one.drop(columns=["cum_pnl_stderr"])
    assert bt.aggregate(old, "pnl").note == bt.SE_RSS_DATES
    row = one.iloc[[-1]]
    assert bt.aggregate(row, bt.GREEK_BUCKETS).stderr == row[bt.group_column("greeks")].iloc[0]
    bare = row.drop(columns=[bt.group_column(g) for g in bt.PAIRED_GROUPS])
    assert bt.aggregate(bare, bt.GREEK_BUCKETS).note == bt.SE_RSS_BUCKETS
    # projection: the cumulative pricing is charged
    code, text = run_cli(["dry-run", str(TOY_CONFIG), *b.path_args(out=tmp_path / "dry")], capsys)
    assert code == 0 and "cumulative P&L pricings" in text, text


def test_explain_bucket_stderrs_are_paired_and_the_fallback_is_flagged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Review (ladder and group stderrs; the explain fallback; follow-up 10 and 29): on a
    9-month call with ladders on four pillars (Black-Scholes builder, 24 seeds; the 9-month
    expiry sits between the 6m and 1y pillars, both moved), the paired errors of the vega ladder
    bucket and of the rates and time groups match the seed dispersion of those sums; the vega
    ladder's (correlated pillars) and the rates group's differ from the root sums of squares of
    their items, and the carry's own error is the paired one (far below the time group's); with
    one Greek without a per-path combination, the
    residual's error is exactly the root sum of squares of the total's, every item's, the time
    buckets' and the (non-zero) recalibration step's, and the fallback is flagged."""
    from volsto.models.bs import BlackScholes
    from volsto.products.vanilla import EuropeanOption
    from volsto.risk import attribution as attr
    from volsto.risk.ladders import vega_T

    spec0 = load_config_yaml(REFERENCE_SPEC, CalibrationSpec)
    market1 = MarketConfig(
        101.0, CurveConfig((0.5, 2.0), (0.0206, 0.024)), CurveConfig((1.0,), (0.0097,))
    )
    surf1 = dataclasses.replace(
        spec0.surface,
        atm_vols=tuple(v + 0.004 * (1 + i) for i, v in enumerate(spec0.surface.atm_vols)),
    )
    spec1 = dataclasses.replace(spec0, market=market1, surface=surf1)
    s0, s1 = RiskState(spec0, None, "d0"), RiskState(spec1, None, "d1")
    call = EuropeanOption(100.0, 0.75, "call", DiscountCurve.flat(0.02))
    pillars = (0.25, 0.5, 1.0, 2.0)
    ps = np.asarray(pillars)
    d_atm = np.asarray(surface_of(s1).atm_vol(ps)) - np.asarray(surface_of(s0).atm_vol(ps))
    assert np.count_nonzero(d_atm[1:3]) == 2
    keys = ("surface.vega_T", "rates", "time")
    sums: dict[str, list[float]] = {k: [] for k in keys}
    ses: dict[str, list[float]] = {k: [] for k in keys}
    rss: dict[str, list[float]] = {k: [] for k in keys}
    for seed in range(24):
        sim = SimConfig(n_paths=2000, chunk_size=2000, dt_max=1.0 / 52.0, seed=300 + seed)
        engine = RiskEngine(BSBuilder(s0, 0.75), sim)  # its vol reads the 6m and 1y pillars
        ex = explain(
            engine,
            call,
            s0,
            s1,
            dt=bt.TRADING_DT,
            detail="ladders",
            pillars=pillars,
            mode="sticky_leverage",
        )
        buckets, se = ex.buckets(), ex.bucket_stderrs
        assert not ex.unpaired and ex.residual_paired
        sums["surface.vega_T"].append(buckets["surface.vega_T"])
        ses["surface.vega_T"].append(se["surface.vega_T"])
        tents = vega_T(
            attr.FrozenLeverageEngine(engine, s0), call, s0, pillars, 0.01, with_tents=True
        ).tents
        rss["surface.vega_T"].append(
            math.sqrt(sum((dv / 0.01 * t.stderr) ** 2 for t, dv in zip(tents, d_atm)))
        )
        sums["rates"].append(buckets["rates.rho"] + buckets["rates.repo"])
        ses["rates"].append(ex.group_stderrs["rates"])
        rss["rates"].append(math.hypot(se["rates.rho"], se["rates.repo"]))
        sums["time"].append(buckets["time.decay"] + buckets["time.carry"])
        ses["time"].append(ex.group_stderrs["time"])
        rss["time"].append(math.hypot(se["time.decay"], se["time.carry"]))
        # the carry is paired (theta): an RSS carry would be at least the held difference's error
        assert se["time.carry"] < 0.5 * ex.group_stderrs["time"], (se, ex.group_stderrs)
    for k in keys:
        ratio = float(np.std(sums[k], ddof=1)) / float(np.mean(ses[k]))
        vs_rss = float(np.mean(ses[k])) / float(np.mean(rss[k]))
        print(f"{k}: dispersion / paired stderr {ratio:.2f}; paired / RSS {vs_rss:.2f}")
        assert 0.65 < ratio < 1.4, (k, ratio)
        if k != "time":  # decay + carry: the carry's paired error is tiny, so the two agree
            assert abs(vs_rss - 1.0) > 0.05, (k, vs_rss)  # an RSS would give 1

    # the fallback: one Greek without a path, and a leverage refit that moves the price
    class NoisyRefit(BSBuilder):
        def build(self, state: RiskState, mode: str) -> Any:
            model = super().build(state, mode)
            if mode == attr.FROZEN_MODE:
                return BlackScholes(1.02 * model.vol, model.forward_curve)
            return model

    real = attr._PathRecorder.path_of

    def forgetful(self: Any, sens: Any) -> Any:
        return None if sens.name.startswith("delta") else real(self, sens)

    monkeypatch.setattr(attr._PathRecorder, "path_of", forgetful)
    sim = SimConfig(n_paths=2000, chunk_size=2000, dt_max=1.0 / 52.0, seed=1)
    ex = explain(
        RiskEngine(NoisyRefit(s0, 0.5), sim), call, s0, s1, dt=bt.TRADING_DT, mode="sticky_leverage"
    )
    assert not ex.residual_paired and "spot.delta" in ex.unpaired, ex.unpaired
    recal = next(s for s in ex.steps if s.name == "recalibration")
    assert recal.actual_stderr > 0
    # parallel detail: one item per bucket, so each bucket's error is its item's |move| x stderr
    items = sum(v**2 for v in ex.bucket_stderrs.values())
    expected = ex.total.stderr**2 + items + recal.actual_stderr**2
    assert ex.residual_stderr**2 == pytest.approx(expected, rel=1e-9)
    assert bool(ex.as_frame()["residual_paired"].iloc[-1]) is False


def test_theta_carry_and_roll_down_are_paired_at_no_extra_pricing() -> None:
    """Follow-up 10: theta's carry and roll-down are paired combinations of the prices theta
    already has — five distinct pricings in all — so their errors include the covariance: the
    carry's error is well below the root sum of squares of the held and zero-rate differences
    it combines, and the roll-down's well below that of the total and held ones."""
    from volsto.products.vanilla import EuropeanOption
    from volsto.risk.greeks import theta

    spec0 = load_config_yaml(REFERENCE_SPEC, CalibrationSpec)
    s0 = RiskState(spec0, None, "d0")
    call = EuropeanOption(100.0, 0.75, "call", DiscountCurve.flat(0.02))
    sim = SimConfig(n_paths=4000, chunk_size=4000, dt_max=1.0 / 52.0, seed=11)
    engine = RiskEngine(BSBuilder(s0, 0.5), sim)
    th = theta(engine, call, s0, bt.TRADING_DT)
    assert engine.n_pricings == 5
    aged = call.aged(bt.TRADING_DT)
    held = engine.paired(
        "held",
        [
            (aged, s0, "recalibrate", 1 / bt.TRADING_DT),
            (call, s0, "recalibrate", -1 / bt.TRADING_DT),
        ],
        unit="per year",
        size=bt.TRADING_DT,
        scheme="forward",
    )
    assert engine.n_pricings == 5 and th.decay.stderr > 0
    assert th.carry.value == pytest.approx(held.value - th.decay.value, rel=1e-9)
    print(
        f"carry stderr {th.carry.stderr:.3g} against RSS {math.hypot(held.stderr, th.decay.stderr):.3g}"
    )
    assert th.carry.stderr < 0.5 * math.hypot(held.stderr, th.decay.stderr)
    rolled_rss = math.hypot(th.total.stderr, held.stderr)
    assert th.roll_down.stderr < 0.1 * rolled_rss, (th.roll_down, rolled_rss)
    assert th.total.value == pytest.approx(
        th.decay.value + th.carry.value + th.roll_down.value, rel=1e-9
    )


def test_session_build_lock_waits_for_a_live_builder_only(tmp_path: Path) -> None:
    """Follow-up 33: the lock of ``tests/_locks.py`` is taken once, and ``wait_for`` returns at
    once when the result exists or the holder is dead (waiting on a live holder is covered by
    ``tests/test_catalogue_s1_s4.py``)."""
    import _locks

    lock, done = tmp_path / "lock", tmp_path / "done.json"
    assert _locks.take_lock(lock) and not _locks.take_lock(lock)
    assert _locks.alive(lock) and _locks.owner(lock) == f"pid {os.getpid()}"
    done.write_text("{}")
    _locks.wait_for(done, lock)  # the result is there: no wait
    dead = tmp_path / "dead"
    dead.mkdir()
    (dead / "pid").write_text("999999")
    assert not _locks.alive(dead)
    _locks.wait_for(tmp_path / "never.json", dead)  # a dead builder: no wait either


def test_followup_headerless_and_torn_journal_refusals(
    build_copy: BacktestBuild,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Follow-up 2, 4, 8: a flat store without a header whose dates another config computed is
    refused by that config's ``status`` / ``migrate`` (nothing written), and its dry run notes
    the refusal and writes no probe.json; an unreadable ``migrations.json`` is kept byte for byte
    — a store with nothing flat says so and goes on, a store to migrate is refused until the
    file is moved aside, and the migration then records every date, the ones an unrecorded
    migration had moved included; a migration interrupted under one config is refused to
    another."""
    b = build_copy
    other = other_config(tmp_path)
    flat_store(b)
    (b.store_root / "backtest.json").unlink()
    (b.store_root / "probe.json").unlink(missing_ok=True)
    before = full_tree(b.store_root)
    for command in ("status", "migrate"):
        code, text = run_cli([command, str(other), *b.path_args()], capsys)
        assert code == bt.EXIT_REFUSED and "has no header" in text, (command, text)
        assert full_tree(b.store_root) == before
    toy_hash = bt.load_backtest_config(TOY_CONFIG).content_hash()
    assert f"under config hash {toy_hash[:12]}" in text  # the dates' own config is named
    v = bt.Ledger.of(cfg_at(b, cfg=other)).verdict(b.dates[0])
    assert v.flat and f"config hash {toy_hash[:12]}" in v.reason, v
    # round 6: run refuses too, before writing its header, even with nothing to compute
    code, text = run_cli(["run", str(other), "--limit", "0", *b.path_args()], capsys)
    assert code == bt.EXIT_REFUSED and "has no header" in text, text
    assert full_tree(b.store_root) == before
    # a config whose calendar does not overlap the store's dates: every stored date is judged
    data = raw_config()
    data["dates"] = {"start": "2022-08-15", "end": "2022-08-19"}
    later = write_config(tmp_path, data, "later.yaml")
    for argv in (["status"], ["migrate"], ["run", "--limit", "0"]):
        code, text = run_cli([argv[0], str(later), *argv[1:], *b.path_args()], capsys)
        assert code == bt.EXIT_REFUSED and "has no header" in text, (argv, text)
        assert full_tree(b.store_root) == before
    code, text = run_cli(["dry-run", str(other), *b.path_args()], capsys)
    assert code == 0 and "a run would be refused" in text, text
    assert full_tree(b.store_root) == before  # F8: no probe.json
    # a torn journal next to flat dates: refused, kept
    journal = b.store_root / "migrations.json"
    journal.write_bytes(b'{"migrations": [{"state": "in_prog')
    before = full_tree(b.store_root)
    code, text = run_cli(["status", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == bt.EXIT_REFUSED and "not a readable migration journal" in text, text
    assert full_tree(b.store_root) == before
    code, text = run_cli(["run", str(TOY_CONFIG), "--limit", "0", *b.path_args()], capsys)
    assert code == bt.EXIT_REFUSED and "not a readable migration journal" in text, text
    assert full_tree(b.store_root) == before  # no header written either
    # an interrupted migration (two dates moved), its journal then lost
    cfg = cfg_at(b)
    journal.unlink()
    calls = {"n": 0}

    def hook(name: str) -> None:
        if name == "migrate.reported":
            calls["n"] += 1
            if calls["n"] == 2:
                raise SimulatedKill()

    monkeypatch.setattr(bt, "CRASH_HOOK", hook)
    with pytest.raises(SimulatedKill):
        bt.BacktestRun(cfg, allow_calibrate=False, migrate=True)
    monkeypatch.setattr(bt, "CRASH_HOOK", None)
    torn = b'{"migrations": [{"state": "in_prog'
    journal.write_bytes(torn)
    code, text = run_cli(["migrate", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == bt.EXIT_REFUSED and "move it aside" in text and journal.read_bytes() == torn
    aside = b.store_root / "quarantine" / "migrations.torn.json"
    aside.parent.mkdir(exist_ok=True)
    os.replace(journal, aside)
    code, text = run_cli(["status", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == 0 and "done 5" in text, text
    entry = json.loads(journal.read_text())["migrations"][-1]
    assert entry["state"] == "complete" and entry["config_hash"] == cfg.content_hash()
    assert sorted(entry["dates"]) == list(b.dates), entry  # the two moved before included
    assert sum(bool(e.get("rebuilt")) for e in entry["dates"].values()) == 2
    assert sorted(entry["snapshots_bound"]) == list(b.dates)
    # nothing flat, a torn journal: status says so and changes nothing
    journal.write_bytes(torn)
    before = full_tree(b.store_root)
    code, text = run_cli(["status", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == 0 and "not a readable migration journal" in text, text
    assert full_tree(b.store_root) == before
    # an interrupted migration of one config is not finished by another
    journal.write_text(
        json.dumps({"migrations": [{"state": "in_progress", "config_hash": "c" * 64}]})
    )
    before = full_tree(b.store_root)
    code, text = run_cli(["migrate", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == bt.EXIT_REFUSED and "started under config hash cccccccccccc" in text, text
    assert full_tree(b.store_root) == before
    # with a header of this config, another config is sent to the journal's config
    bt.bind_header(bt.load_backtest_config(TOY_CONFIG), bt.BacktestStore(b.store_root))
    before = full_tree(b.store_root)
    code, text = run_cli(["run", str(other), "--limit", "0", *b.path_args()], capsys)
    assert code == bt.EXIT_REFUSED, text
    code, text = run_cli(["migrate", str(other), *b.path_args()], capsys)
    assert code == bt.EXIT_REFUSED and "config hash cccccccccccc, which started" in text, text
    assert full_tree(b.store_root) == before


def test_followup_writer_guards_and_runnable_refusals(
    build_copy: BacktestBuild,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Follow-up 1, 5, 6, 7: a commit refuses a foreign CURRENT before publishing anything; a
    date directory moved while its lock is held fails the commit (never reported current); gc
    on a store it cannot write is a clear error (exit 1), nothing removed; every printed refusal
    line parses as a command (the leverage line keeps ``--force``; the header refusal names its
    two commands); a foreign CURRENT leaves stage 2 with that refusal as its only command."""
    b = build_copy
    cfg = cfg_at(b)
    run = bt.BacktestRun(cfg, allow_calibrate=False)
    d = b.dates[3]
    # F7: foreign CURRENT, nothing published
    v = run.ledger.verdict(d)
    rows, fit, done = v.verified.frame(), v.verified.json("fit.json"), dict(v.doc or {})
    done.pop("record")
    done["created_utc"] += "#foreign"
    ptr_path = b.store_root / "dates" / d / "CURRENT"
    ptr = json.loads(ptr_path.read_text())
    ptr_path.write_text(json.dumps({**ptr, "version": 2}))
    listing = sorted(p.name for p in (b.store_root / "dates" / d / "attempts").iterdir())
    with pytest.raises(bt.RefusedError, match="pointer version 2"):
        run._commit(d, rows, fit, done, prev_used=True)
    assert sorted(p.name for p in (b.store_root / "dates" / d / "attempts").iterdir()) == listing
    with pytest.raises(runner.MissingRequirements) as info:
        run_stage2(b, tmp_path / "foreign", latex_check=False)
    assert not any(c.startswith("volsto-backtest") for c in info.value.commands), info.value
    assert any("another pointer version" in c for c in info.value.commands)
    ptr_path.write_text(json.dumps(ptr))
    # F1: the date directory is moved while the commit holds its lock
    date_dir = b.store_root / "dates" / d
    moved = date_dir.with_name(d + ".moved")

    def swap(name: str) -> None:
        if name == "pointer.decided":
            os.rename(date_dir, moved)
            shutil.copytree(moved, date_dir)

    monkeypatch.setattr(bt, "CRASH_HOOK", swap)
    with pytest.raises(bt.RefusedError, match="moved or replaced while it was locked"):
        recommit(bt.BacktestRun(cfg, allow_calibrate=False), d, "swapped")
    monkeypatch.setattr(bt, "CRASH_HOOK", None)
    shutil.rmtree(moved)
    assert bt.Ledger.of(cfg).verdict(d).status == "done"
    # round 6: dates/ made unsearchable while the lock is held: refused, not a traceback
    dates_root = b.store_root / "dates"

    def lock_out(name: str) -> None:
        if name == "pointer.decided":
            dates_root.chmod(0o600)

    monkeypatch.setattr(bt, "CRASH_HOOK", lock_out)
    try:
        with pytest.raises(bt.RefusedError, match="moved or replaced"):
            recommit(bt.BacktestRun(cfg, allow_calibrate=False), d, "unsearchable")
    finally:
        dates_root.chmod(0o755)
        monkeypatch.setattr(bt, "CRASH_HOOK", None)
    # F5: gc on a store whose attempts cannot be written
    extra = recommit(bt.BacktestRun(cfg, allow_calibrate=False), b.dates[1], "extra")
    assert extra[2]
    attempts = b.store_root / "dates" / b.dates[1] / "attempts"
    before = full_tree(b.store_root)
    attempts.chmod(0o555)
    try:
        code, text = run_cli(["gc", str(TOY_CONFIG), *b.path_args()], capsys)
    finally:
        attempts.chmod(0o755)
    assert code == bt.EXIT_FAILED and "not writable" in text and "Traceback" not in text, text
    assert full_tree(b.store_root) == before
    # F6: the printed lines parse; the leverage line keeps --force
    parser = bt.build_parser()
    other = other_config(tmp_path)
    empty = tmp_path / "empty_cache"
    empty.mkdir()
    args = ["--out", str(b.store_root), "--cache", str(empty), "--snapshots", str(b.snapshots_root)]
    code, text = run_cli(["run", str(other), "--no-calibrate", "--force", *args], capsys)
    assert code == bt.EXIT_REFUSED and "lacks the leverage" in text, text
    line = re.search(r"Recalibrate them with: (volsto-backtest run .*?) —", text)
    assert line is not None and line.group(1).endswith("--force"), text
    parsed = parser.parse_args(shlex.split(line.group(1))[1:])
    assert parsed.force and parsed.only_dates
    code, text = run_cli(["run", str(other), "--no-calibrate", *b.path_args()], capsys)
    assert code == bt.EXIT_REFUSED and "was built from config hash" in text, text
    force_line = re.search(r"under this config with: (volsto-backtest run .*?--force)", text)
    elsewhere = re.search(
        r"write elsewhere \(replace NEW_STORE\): (volsto-backtest run .*? --resume)", text
    )
    assert force_line is not None and elsewhere is not None, text
    assert parser.parse_args(shlex.split(force_line.group(1))[1:]).force
    argv = shlex.split(elsewhere.group(1))[1:]
    parsed = parser.parse_args(argv)
    assert argv.count("--out") == 1 and parsed.out == bt.NEW_STORE
    assert parsed.snapshots == f"{bt.NEW_STORE}/snapshots"


def test_followup_version_1_records_are_flagged_never_silent(
    build_copy: BacktestBuild,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Follow-up 3 (the orchestrator's decision): a version-1 date with results stays done but
    carries ``legacy_unverified`` — even when its leverage's numbers change under the same key
    (a version-2 date is stale then) — and ``status``, study.md and the manifest name it."""
    b = build_copy
    flat_store(b)  # the round-2 golden: version-1 records
    cfg = cfg_at(b)
    run = bt.BacktestRun(cfg, allow_calibrate=False, migrate=True)
    verdicts = run.ledger.verdicts()
    assert {v.status for v in verdicts.values()} == {"done"}
    assert all(v.legacy_unverified == bt.LEGACY_UNVERIFIED for v in verdicts.values())
    key = verdicts[b.dates[1]].doc["record"]["leverage"]["key"]  # type: ignore[index]
    npz = b.cache_root / key / "leverage.npz"
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["values"] = arrays["values"] * (1.0 + 1e-9)
    with open(npz, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    v = bt.Ledger.of(cfg).verdict(b.dates[1])
    assert v.status == "done" and v.legacy_unverified, v  # documented: not verifiable at v1
    code, text = run_cli(["status", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == 0 and f"5 date(s) with results carry a {bt.LEGACY_UNVERIFIED}" in text, text
    assert "--resume" not in text.split(bt.LEGACY_UNVERIFIED, 1)[1].split("\n", 1)[0]
    res = run_stage2(b, tmp_path / "legacy", latex_check=False)
    manifest = json.loads((res.out_dir / "manifest.json").read_text())
    flag = manifest["records"]["backtest_legacy_unverified"]
    assert flag["count"] == 5 and flag["dates"] == list(b.dates), flag
    md = (res.out_dir / "study.md").read_text()
    assert f"5 date(s) carry a {bt.LEGACY_UNVERIFIED}" in md
    # a date recomputed (version 2) loses the flag
    code, text = run_cli(
        ["run", str(TOY_CONFIG), "--no-calibrate", "--only-dates", b.dates[4], *b.path_args()],
        capsys,
    )
    assert code == 0, text
    v = bt.Ledger.of(cfg).verdict(b.dates[4])
    assert v.status == "done" and v.legacy_unverified == "", v
    v2_attempt = current(b.store_root, b.dates[4]).name
    # round 6: the version-2 date checks the numbers of the version-1 state it uses
    key3 = bt.Ledger.of(cfg).verdict(b.dates[3]).doc["record"]["leverage"]["key"]  # type: ignore[index]
    npz3 = b.cache_root / key3 / "leverage.npz"
    with np.load(npz3, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    arrays["values"] = arrays["values"] * (1.0 + 1e-9)
    with open(npz3, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    verdicts = bt.Ledger.of(cfg).verdicts()
    assert verdicts[b.dates[3]].status == "done" and verdicts[b.dates[3]].legacy_unverified
    assert verdicts[b.dates[4]].status == "stale", verdicts[b.dates[4]]
    # ... and its version-1 attempt (done by its weaker check) never becomes current again
    store = bt.BacktestStore(b.store_root)
    assert bt.adopt(store, bt.Ledger.of(cfg), [b.dates[4]]) == []
    code, text = run_cli(["gc", str(TOY_CONFIG), *b.path_args()], capsys)
    assert code == 0, text
    assert current(b.store_root, b.dates[4]).name == v2_attempt
    labels = sorted(a.id.rsplit("-", 1)[0] for a in store.attempts(b.dates[4]))
    assert labels == ["legacy-ok", "ok"], labels
    assert bt.Ledger.of(cfg).verdict(b.dates[4]).status == "stale"


def test_every_printed_command_runs_as_printed(
    toy_backtest_build: BacktestBuild,  # noqa: F811
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The printed-command invariant (third occurrence of a printed command that does not run
    as printed): every ``volsto-backtest`` line the module prints goes through
    :func:`~volsto.studies.backtest.command_line`.  The walking test records every such line
    while it drives each path that prints one — the header, config-hash and leverage refusals
    and their "elsewhere" lines, a date failed for a missing leverage, the ``status`` hints (gc,
    other hosts, the version-1 re-record), the verdict hints (migrate, adopt), stage 2's
    requirements and missing-date commands — on stores whose paths hold spaces, under
    ``VOLSTO_BACKTEST_REQUIRE_PATHS=1``.  Each line must split back (``shlex``), parse, name an
    existing config and carry all three paths; each is then executed as printed (``--limit 0``
    for a run line) and does what it says.  The "elsewhere" line of a config with other
    surface settings is run on one date: the refused store is left byte for byte."""
    printed: list[str] = []
    real_line = bt.command_line

    def record(argv: Sequence[str]) -> str:
        line = real_line(argv)
        printed.append(line)
        return line

    monkeypatch.setattr(bt, "command_line", record)
    monkeypatch.setenv(bt.REQUIRE_PATHS_ENV, "1")
    toy = toy_backtest_build.require()
    a = toy.copy(tmp_path / "a store" / "build")
    parser = bt.build_parser()
    (tmp_path / "cfg dir").mkdir()
    other = other_config(tmp_path / "cfg dir")
    data = raw_config()
    data["surface"]["calendar_repair"] = False  # other snapshots: a re-import would overwrite
    unrepaired = write_config(tmp_path / "cfg dir", data, "unrepaired.yaml")
    empty = tmp_path / "empty cache"
    empty.mkdir()

    def lines_since(n: int) -> list[str]:
        return list(dict.fromkeys(printed[n:]))

    def parsed(line: str, new_store: Path | None = None) -> list[str]:
        argv = shlex.split(line)
        assert argv[0] == "volsto-backtest", line
        argv = argv[1:]
        if new_store is not None:
            argv = [x.replace(bt.NEW_STORE, str(new_store)) for x in argv]
        args = parser.parse_args(argv)
        assert bt.missing_path_flags(args) == [], line
        assert Path(args.config).is_file(), line
        assert " " in args.out or bt.NEW_STORE in line, line  # the paths survived quoting
        return argv

    # 1. the refusals and their lines
    n = len(printed)
    code, text = run_cli(["run", str(other), "--no-calibrate", *a.path_args()], capsys)
    assert code == bt.EXIT_REFUSED, text
    header_lines = lines_since(n)
    assert len(header_lines) == 2 and all(line in text for line in header_lines), text
    n = len(printed)
    args = ["--out", str(a.store_root), "--cache", str(empty), "--snapshots", str(a.snapshots_root)]
    code, text = run_cli(["run", str(TOY_CONFIG), "--no-calibrate", "--resume", *args], capsys)
    assert code == bt.EXIT_REFUSED and "lacks the leverage" in text, text
    leverage_lines = lines_since(n)
    fresh = tmp_path / "fresh store"
    n = len(printed)
    fresh_args = ["--out", str(fresh), "--cache", str(empty), "--snapshots", str(a.snapshots_root)]
    code, text = run_cli(
        ["run", str(TOY_CONFIG), "--no-calibrate", "--only-dates", a.dates[0], *fresh_args],
        capsys,
    )
    assert code == bt.EXIT_REFUSED and "produce it with" in text, text
    missing_lines = lines_since(n)
    stored = current_json(fresh, a.dates[0], "done.json")["command"]
    assert stored in missing_lines
    # 2. status hints
    attempts = a.store_root / "dates" / a.dates[0] / "attempts"
    (attempts / ".staging-otherhost-4242-abcd").mkdir()
    (attempts / f"{bt.STAGING_PREFIX}{bt._host_tag()}-999999-dead").mkdir()
    n = len(printed)
    code, text = run_cli(["status", str(TOY_CONFIG), *a.path_args()], capsys)
    assert code == 0, text
    status_lines = lines_since(n)
    assert status_lines and all(line in text for line in status_lines), text
    # 3. verdict hints and stage 2
    cfg_a = cfg_at(a)
    ptr = a.store_root / "dates" / a.dates[2] / "CURRENT"
    ptr_bytes = ptr.read_bytes()
    ptr.unlink()
    n = len(printed)
    v = bt.Ledger.of(cfg_a).verdict(a.dates[2])
    assert v.status == "missing" and printed[-1] in v.reason, v
    with pytest.raises(runner.MissingRequirements) as info:
        run_stage2(a, tmp_path / "study 1", latex_check=False)
    stage2_lines = lines_since(n)
    assert all(c in stage2_lines for c in info.value.commands if not c.startswith("#"))
    ptr.write_bytes(ptr_bytes)
    n = len(printed)
    assert run_stage2(a, tmp_path / "study 2", latex_check=False).exit_code == 0
    requirement_lines = lines_since(n)
    assert len(requirement_lines) >= len(a.dates)
    # 4. every line parses, then runs as printed
    everything = (
        header_lines
        + leverage_lines
        + missing_lines
        + status_lines
        + stage2_lines
        + requirement_lines
    )
    new_store = tmp_path / "new store"
    forced = [line for line in everything if "--force" in shlex.split(line)]
    for line in everything:
        argv = parsed(line, new_store)
        if line in forced:
            continue  # below: it rebinds the store to another config
        extra = ["--limit", "0"] if argv[0] == "run" else []
        code, text = run_cli([*argv, *extra], capsys)
        assert code in ((0, 1) if argv[0] == "status" else (0,)), (line, text)
        if argv[0] == "run" and bt.NEW_STORE in line:
            assert json.loads((new_store / "backtest.json").read_text())["config_hash"] == (
                bt.load_backtest_config(other).content_hash()
            )
    # 5. the elsewhere line of a config whose snapshots differ leaves the refused store alone
    n = len(printed)
    code, text = run_cli(["run", str(unrepaired), "--no-calibrate", *a.path_args()], capsys)
    assert code == bt.EXIT_REFUSED, text
    elsewhere = [line for line in lines_since(n) if bt.NEW_STORE in line]
    assert len(elsewhere) == 1
    before = full_tree(a.store_root)
    target = tmp_path / "unrepaired store"
    argv = parsed(elsewhere[0], target)
    code, text = run_cli([*argv, "--limit", "1", "--no-calibrate"], capsys)
    assert code in (0, bt.EXIT_FAILED, bt.EXIT_REFUSED), text
    assert full_tree(a.store_root) == before
    assert (target / "snapshots" / f"spx_{a.dates[0]}.yaml").is_file()
    # 6. the forced lines accept the store for their config
    for line in forced:
        argv = parsed(line, new_store)
        code, text = run_cli([*argv, "--limit", "0"], capsys)
        assert code == 0, (line, text)
        hashed = bt.load_backtest_config(Path(argv[1])).content_hash()
        store = json.loads((Path(parser.parse_args(argv).out) / "backtest.json").read_text())
        assert store["config_hash"] == hashed
    # 7. a flat store: the migrate hint and the version-1 re-record line
    f = toy.copy(tmp_path / "flat store" / "build")
    flat_store(f)
    cfg_f = cfg_at(f)
    n = len(printed)
    v = bt.Ledger.of(cfg_f).verdict(f.dates[0])
    assert v.flat and printed[-1] in v.reason, v
    with pytest.raises(runner.MissingRequirements) as info:
        run_stage2(f, tmp_path / "study 3", latex_check=False)
    migrate_lines = [line for line in lines_since(n) if shlex.split(line)[1] == "migrate"]
    assert migrate_lines and migrate_lines[0] in info.value.commands
    code, text = run_cli(parsed(migrate_lines[0]), capsys)
    assert code == 0 and "migrated 5 date(s)" in text, text
    n = len(printed)
    code, text = run_cli(["status", str(TOY_CONFIG), *f.path_args()], capsys)
    legacy = [line for line in lines_since(n) if "--only-dates" in line]
    assert len(legacy) == 1 and "--resume" not in shlex.split(legacy[0]), text
    code, text = run_cli([*parsed(legacy[0]), "--limit", "0"], capsys)
    assert code == 0, text
    print(f"printed commands checked: {len(set(printed))}")


def test_backtest_fixture_goes_through_the_shared_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Round 6: ``toy_backtest_build`` builds through ``tests/_locks.py::shared_build`` — a
    recorded result is returned without building, a dead builder's lock fails the consumers
    naming its pid, and a build that raises is recorded as the error."""
    import _locks

    monkeypatch.delenv("PYTEST_XDIST_WORKER", raising=False)
    monkeypatch.setattr(
        _backtest_build, "_run_toy_backtest", lambda root: pytest.fail("must not build")
    )
    monkeypatch.setattr(_backtest_build, "_backtest_absent", lambda: "")

    def build(base: Path) -> BacktestBuild:
        return _backtest_build.shared_toy_backtest(base)

    root = tmp_path / "done" / "toy_backtest"
    root.mkdir(parents=True)
    assert _locks.take_lock(root / "lock")
    (root / "done.json").write_text(json.dumps({"return_code": 0, "built_by": "other"}))
    got = build(tmp_path / "done")
    assert got.require() is got and got.built_by == "other"
    root = tmp_path / "dead" / "toy_backtest"
    root.mkdir(parents=True)
    assert _locks.take_lock(root / "lock")
    (root / "lock" / "pid").write_text("999999")
    with pytest.raises(pytest.fail.Exception, match=r"pid 999999.*died without a result"):
        build(tmp_path / "dead").require()

    def boom(root: Path) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(_backtest_build, "_run_toy_backtest", boom)
    got = build(tmp_path / "raises")
    with pytest.raises(pytest.fail.Exception, match="RuntimeError: boom"):
        got.require()
    assert (tmp_path / "raises" / "toy_backtest" / "lock").is_dir()  # kept: one build per run
