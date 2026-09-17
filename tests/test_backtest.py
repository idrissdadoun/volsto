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

import contextlib
import dataclasses
import datetime as dt
import hashlib
import json
import math
import os
import re
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
from volsto.calibration.history import DEFAULT_PILLARS
from volsto.calibration.stability import PARAM_COLUMNS, flag_unidentified
from volsto.config import CalibrationSpec, ConfigError, CurveConfig, MarketConfig, SimConfig
from volsto.config import load_yaml as load_config_yaml
from volsto.market.curves import DiscountCurve
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


def test_shards_split_the_only_dates_selection(tmp_path: Path) -> None:
    """``--shard i/n`` cuts the ``--only-dates`` selection (before ``--resume``), not the
    whole calendar: two shards of a sub-window are its two halves."""
    data = raw_config()
    data["dates"] = {"start": "2022-07-01", "end": "2022-08-31"}
    data["paths"] = {k: str(tmp_path / k) for k in ("out", "cache", "snapshots")}
    cfg = bt.BacktestConfig.from_mapping(data)
    run = bt.BacktestRun(cfg, allow_calibrate=False)
    if len(run.calendar) < 30:
        pytest.skip(f"HDN sample absent: {len(run.calendar)} calendar dates")
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
    book, _ = res.value("pnl_trade", bt.BOOK_ROW, "pnl")
    assert book == pytest.approx(-100.0 * 0.024)
    cols = res.columns("pnl_month")
    assert [g for g, _ in bt.BUCKET_GROUPS] == [c for c in cols if c not in ("total", "check")]
    parts = sum(res.value("pnl_month", "2022-10", g)[0] for g, _ in bt.BUCKET_GROUPS)
    assert parts == pytest.approx(res.value("pnl_month", "2022-10", "total")[0], abs=1e-12)
    assert res.value("pnl_month", "2022-10", "check")[0] == pytest.approx(0.0, abs=1e-12)
    specs = {t.name: t for t in bt.tables(res)}
    assert specs["pnl_trade"].columns[0].header == "desk P&L"


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
    code, text = run_cli(["status", str(cfg), "--out", str(b.store_root)], capsys)
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
    assert "import failed" in first["error"] and "missing_close: skip_date" in first["error"]
    assert "restore" in first["error"]
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
    """D7, ``data.missing_close: skip_date``: a corrupt day is dropped from the calendar before
    anything is computed (``done.json`` status skipped); the next date is attributed from the
    date before the gap, its trades are one trading day old and list the gap."""
    b = build_copy
    days = list(b.dates[:3])
    data = raw_config()
    data["data"]["root"] = str(hdn_root(tmp_path, days, corrupt=[days[1]]))
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
    assert skipped["status"] == "skipped" and "import failed" in skipped["error"]
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
    build_copy: BacktestBuild, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """D4: R3 — a data root without 2022-07-28 (same config hash, same day files otherwise)
    makes that date's successors stale (their calendar prefix changed) and --resume would
    recompute exactly them; R3b — a manifest whose rates for 2022-07-29 moved by +50 bp makes
    that date and its successors stale, and the snapshot is re-imported with the new rates."""
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
    old_rates = yaml.safe_load(snap.read_text())["market"]["rate_curve"]["rates"]
    assert not bt.snapshot_bound(
        cfg_r, snap, run_r.inputs.file_sha("2022-07-29"), run_r.inputs.manifest_sha("2022-07-29")
    )
    path, seconds = run_r.ensure_snapshot("2022-07-29")
    new_rates = yaml.safe_load(path.read_text())["market"]["rate_curve"]["rates"]
    assert seconds > 0 and new_rates[0] == pytest.approx(old_rates[0] + 0.005, abs=2e-3)
    assert new_rates != old_rates
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
    assert code == bt.EXIT_REFUSED and "before them were computed under another config" in text


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
        with real_lock(self, date):
            held.add(date)
            try:
                yield
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
            with real_lock(q.store, "2022-07-28"):
                if q.store.pointer("2022-07-28")["attempt"] != old:  # type: ignore[index]
                    q.store.remove_attempt(q.store.attempt("2022-07-28", old))
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
    assert "a later attempt failed" in v.reason, v
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
    return run._commit(date, rows, fit, done, prev_used=used)  # type: ignore[no-any-return]


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
    # the in-process kills' staging directories belong to this (live) process: kept
    assert left["staging"] and all(f"-{os.getpid()}-" in p.name for p in left["staging"])
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
            ".lock",
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
    assert run_stage2(b, tmp_path / "after", latex_check=False).exit_code == 0
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
    assert "a later attempt failed: simulated" in v.reason, v


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
    for t in ("trades", "inception", "pnl_trade", "pnl_bucket", "ssr", "vko", "params_flags"):
        assert t in m["tables"] and (out / "tables" / f"{t}.tex").is_file(), t
    for f in ("book_pnl", "trade_pnl", "explained", "params", "ssr", "vko"):
        assert (out / "figures" / f"{f}.pdf").is_file() and (out / "figures" / f"{f}.png").is_file()
    text = (out / "study.md").read_text()
    assert "proof of concept" in text and "too short for a conclusion" in text
    assert "one-line change" in text and "Fast mode" in text
    assert "not observable in the window" in text
    assert "variance notional 1/(2 K_vol)" in text and "strike 0 in variance units" in text
    assert "inception surface's log-contract strike" in text
    assert "marked on every date after inception" in text  # the toy marks its rolling book daily
    # D5: the window is described from the data, without a regime adjective
    assert "bear market" not in text and "2022 H2 sample" not in text
    closes = [store_fits(b.store_root)[d]["spot"] for d in b.dates]
    assert f"went from {closes[0]:.2f} to {closes[-1]:.2f}" in text
    assert "realised volatility of" in text and "SPX day files under" in text
    # (c) the stability fit is stated; (a) the desk sign is stated and used; D10 checks
    assert "stability.fit" in text and "skew_mode soft" in text
    assert "desk P&L = -(V(d) - V(d-1)" in text and "Every stored date carries its P&L." in text
    assert "MTM" not in text and all("MTM" not in f.caption for f in bt.figures(run.results))
    tex = (out / "tables" / "pnl_trade.tex").read_text()
    assert "desk P\\&L" in tex and "$-0 " not in tex
    for month in run.results.rows("pnl_month"):
        assert abs(run.results.value("pnl_month", month, "check")[0]) < 1e-10
    assert not re.search(r"\bnan\b", text, flags=re.IGNORECASE)
    # the book's desk P&L in the results is minus the sum of the stored (holder) daily P&L
    rows = store_rows(b.store_root)
    fixed = rows[(rows["book"] == "fixed") & rows["pnl"].notna()]
    v, se = run.results.value("pnl_trade", bt.BOOK_ROW, "pnl")
    assert v == pytest.approx(-100.0 * fixed["pnl"].sum(), rel=1e-12)
    assert se == pytest.approx(100.0 * math.sqrt((fixed["pnl_stderr"] ** 2).sum()), rel=1e-12)
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
    assert (att["extra_pricings"] == 0).all()
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
                fresh = bt.build_product(spec, st.spot, st.surface, st.discount).product
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
        closes.append(float(snap["market"]["spot"]))
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
