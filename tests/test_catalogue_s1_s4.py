"""Study catalogue S1–S4 (SPEC §10.2, owner's M10 Part 2) and the walking test of every catalogue
study's CI-fast config.

**Nothing here calibrates.**  The fast configs run against the sanctioned session builds of
``tests/conftest.py`` — ``toy_build`` (toy grid) and ``toy_marking_build`` (toy marking grid) —
chosen by the grid each fast config names; every run goes through
:func:`volsto.studies.runner.run_study`, whose process-wide guard makes a calibration attempt a
:class:`~volsto.studies.runner.CalibrationForbiddenError`.

What is asserted:

* **the walking test** (``test_catalogue_fast_config_runs_without_calibration``, one case per
  ``configs/studies/catalogue/*_fast.yaml``, S5–S7 included; configs of the rolling backtest,
  ``runner: volsto.studies.backtest``, are excluded by module and walked by
  ``tests/test_backtest.py``): the run exits 0 (the LaTeX check
  compiles when Tectonic is installed), the manifest says ``recalibrated: false`` and records
  only leverages of the build's cache at its particle count, the build's cache is unchanged —
  the same manifest rows and the same files by size and mtime —, at least one table and one
  figure are written, ``study.md`` starts with the title and the one-sentence question, and
  ``volsto-study render`` rebuilds byte-identical ``.tex`` tables and the same figure set from
  ``results.parquet`` alone; then the study's own checks for S1–S4 (below);
* S1: every toy model read from the store, the headline / smile / LSV-minus-LV (with the
  z lower bounds) / cliquet-ladder / regression tables present, the budgets reported as different from the baseline's; a point
  removed from a store copy is priced from its cached leverage (``price_missing``) or, with
  ``price_missing: false``, is a missing requirement whose command names the config's grid and
  the point; the baseline file of ``tests/test_m4_regression.py`` is read as data; every
  number whose product ends beyond the 1y toy horizon carries the note, in every table (the
  forward-vol rows by their flag, smile and wing cells by their axis, headline and regression
  cells of the 1y -> 2y keys by that window's flag, cliquets by maturity), and no other does;
* S2: ratios in (0, 1] and increasing in the barrier, ordered ITM percentiles, the realised-
  outcome tables with the maturity after the last close, both sign readings scored only where
  resolved, the paired stderr of every ratio difference below its quadrature error, no
  ``± 0.0`` in study.md, the ``pairing`` table and the local-vol grid sentence; the quantile estimator against numpy's inverted-CDF percentile, its
  bootstrap error against the asymptotic one and the bootstrap's pairing across path sets; the
  realised figures against a hand computation;
* S3: ``K_up(100%) < K_var < K_down(100%)`` for every toy model; one Gyöngy cell per model ×
  leg × barrier whose verdict is recomputed from its z and tolerance, the per-leg counts in
  study.md equal to the table's, no fixed "zero line" sentence; the ``pairing`` table and the
  local-vol grid sentence; the monitoring-order table; the owner's sentence recorded verbatim;
  the store-check paragraph (the toy store has no conditional numbers) on hand-made results;
* S4: the legs of ``decompose()`` add up to the price (the Phoenix's European counterpart of its
  American put is reported beside them), the differences paired and labelled so, the paired
  stderr below the quadrature one, the 1y toy horizon disclosed on every number of an LSV
  model and of a flagged forward window, in any table (and on no other number), in the
  ``horizon`` table and study.md, deterministic zero exposures exact, the LSV ``skew_T``
  reported missing with its ``--risk light`` line and the tents' mismatch; the like-for-like
  sentence on hand-made pillar sets;
* the pairing helpers: ``paired_difference`` against the per-sample difference, the quadrature
  error and the correlation; the quadrature-versus-paired sentence from its numbers; ``ratio_estimate`` against ``ratio_of_means``; ``shell_block``
  pastes back into the same command; on the repository store (skipped
  when absent) the stored M6 fractions are converted to % of notional and noted;
* the requirement lines of ``s1_grid.yaml`` and ``s3.yaml`` against the repository store and
  cache (skipped when absent): exit 2 and the **config's** grid in the printed command;
* ``toy_marking_build`` builds (the S5 fixture): the LV point and the four marking points, one
  calibration per feasible fit, the build time printed.

Wall clocks are printed, never asserted.
"""

from __future__ import annotations

import contextlib
import io
import math
import re
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from volsto.calibration.cache import LeverageCache
from volsto.studies import runner
from volsto.studies.catalogue import s1_forward_vol, s2_vko, s3_conditional_variance, s4_autocall
from volsto.studies.catalogue._common import (
    LV_LABEL,
    PAIRED_NOTE,
    Estimate,
    axis,
    lv_grid_sentence,
    mean_estimate,
    paired_difference,
    quadrature_comparison,
    ratio_estimate,
    shell_block,
)
from volsto.studies.results import Results, ResultsBuilder
from volsto.studies.runner import ConfigOverrides, StudyRun

ROOT = Path(__file__).resolve().parents[1]
CATALOGUE = ROOT / "configs" / "studies" / "catalogue"
#: Runner modules whose fast configs this walker leaves alone: the rolling backtest is a study
#: module too, but it needs its own date-range fixture and is walked by tests/test_backtest.py.
WALKED_ELSEWHERE: frozenset[str] = frozenset({"volsto.studies.backtest"})
FAST_CONFIGS = sorted(
    p
    for p in CATALOGUE.glob("*_fast.yaml")
    if runner.load_study_config(p).runner not in WALKED_ELSEWHERE
)
#: The session build each fast config runs on, by the grid it names (``None``: an artefact-only
#: study, run on the toy build's synthetic M7 / M8b outputs).
FIXTURE_FOR_GRID: dict[str | None, str] = {
    "configs/grids/toy.yaml": "toy_build",
    "configs/grids/toy_marking.yaml": "toy_marking_build",
    None: "toy_build",
}
HISTORY_REL = Path("essvi_gate") / "hdn_history_repaired.csv"
TOY_PARTICLES = 20_000
REPO_STORE = ROOT / "outputs" / "store"
REPO_CACHE = ROOT / "cache"
#: The phrase of every beyond-the-calibration-horizon note (S1 and S4).
BEYOND = "calibration horizon"


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def write_synthetic_history(path: Path, n_days: int = 127, seed: int = 7) -> pd.DataFrame:
    """A surface history in the schema of ``outputs/essvi_gate/hdn_history_repaired.csv``
    (``date, T, vs_vol, atm_vol, skew, ln_spot``): business days from 2022-07-01, a seeded
    25%-vol log-spot path, two pillars."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-07-01", periods=n_days)
    steps = 0.25 / math.sqrt(252.0) * rng.standard_normal(n_days - 1)
    ls = 8.25 + np.concatenate([[0.0], np.cumsum(steps)])
    rows = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "T": T,
            "vs_vol": vs,
            "atm_vol": atm,
            "skew": -0.3,
            "ln_spot": float(x),
        }
        for d, x in zip(dates, ls, strict=True)
        for T, vs, atm in ((0.25, 0.27, 0.25), (1.0, 0.28, 0.24))
    ]
    frame = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame


def outputs_copy(build: Any, tmp_path: Path) -> Path:
    """A private copy of the build's outputs root plus the synthetic surface history (the build
    directory stays untouched)."""
    dst = tmp_path / "outputs"
    shutil.copytree(build.outputs_root, dst)
    write_synthetic_history(dst / HISTORY_REL)
    return dst


def cache_snapshot(root: Path) -> tuple[int, dict[str, tuple[int, int]]]:
    """``(manifest rows, {relative file: (size, mtime_ns)})`` of a leverage cache."""
    rows = len(LeverageCache(root).manifest())
    files = {
        str(p.relative_to(root)): (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }
    return rows, files


def run_config(
    config_path: Path,
    build: Any,
    tmp_path: Path,
    *,
    store: Path | None = None,
    outputs: Path | None = None,
    sets: tuple[str, ...] = (),
    latex_check: bool = True,
) -> StudyRun:
    grid_null = runner.load_study_config(config_path).grid is None
    overrides = ConfigOverrides(
        grid=None if grid_null else str(build.grid_path),
        store=str(store if store is not None else build.store_root),
        cache=str(build.cache_root),
        outputs=str(outputs if outputs is not None else outputs_copy(build, tmp_path)),
        sets=sets,
    )
    config = runner.load_study_config(config_path, overrides)
    t0 = time.perf_counter()
    run = runner.run_study(
        config, out_dir=tmp_path / "out", config_path=config_path, latex_check=latex_check
    )
    print(
        f"{config.name}: {len(run.results)} numbers in {time.perf_counter() - t0:.1f} s "
        f"(compute {run.manifest['wall_clock']['compute']:.1f} s); latex "
        f"{(run.manifest.get('latex') or {}).get('status')}"
    )
    return run


def fast_build(config_path: Path, request: pytest.FixtureRequest) -> Any:
    grid = runner.load_study_config(config_path).grid
    if grid not in FIXTURE_FOR_GRID:
        pytest.fail(
            f"{config_path.name}: a fast config must run on a sanctioned toy grid "
            f"{[g for g in FIXTURE_FOR_GRID if g]} or on none, not {grid!r}"
        )
    return request.getfixturevalue(FIXTURE_FOR_GRID[grid]).require()


def values(results: Results, table: str, column: str) -> dict[str, tuple[float, float]]:
    return {r: results.value(table, r, column) for r in results.rows(table)}


def noted(frame: pd.DataFrame) -> pd.Series:
    return frame["note"].fillna("").str.contains(BEYOND, regex=False)


def assert_noted(frame: pd.DataFrame, flagged: pd.Series, where: str) -> None:
    """Every flagged record carries the beyond-horizon note and no other record does."""
    bad = frame[noted(frame) != flagged]
    assert bad.empty, (where, bad[["row", "column", "note"]].head().to_dict("records"))


def coordinates(axes: dict[str, Any]) -> str:
    """A model's own coordinates, as a key (the study-specific ones dropped)."""
    own = {k: v for k, v in axes.items() if k not in ("baseline_model", "key")}
    return repr(sorted(own.items()))


def check_s1_beyond(r: Results) -> None:
    """Every S1 number whose product ends beyond the calibration horizon says so, in any table:
    the forward-vol rows by their flag, the smile / wing cells by their axis, the headline and
    regression cells of the 1y -> 2y keys by that window's flag, the cliquets by maturity."""
    fv = r.pivot("forward_vols")
    flags = fv["beyond_horizon"]
    assert (flags == 1.0).any()  # the toy horizon is 1y
    long = r.long("forward_vols")
    assert_noted(long, long["row"].map(flags).eq(1.0), "forward_vols")
    window = s1_forward_vol.window_label(*s1_forward_vol.WINDOWS[0])
    horizon = r.pivot("provenance")["horizon"]
    fwd_keys = {k for k, _ in s1_forward_vol.FWD_KEYS}
    label_of = {coordinates(r.axes("provenance", m)): m for m in r.rows("provenance")}
    covered = set()
    for t in r.tables():
        long = r.long(t)
        if t == "forward_vols" or long.empty:
            continue
        ax = axis(long, "beyond_horizon")
        if ax.notna().any():
            assert ax.notna().all(), t
            assert_noted(long, ax.eq(1.0), t)
        elif t in ("headline", "cliquet_ladder"):
            win = long["row"].map(lambda m: flags[f"{m} | {window}"])
            clq = long["column"].str.extract(r"^(?:cliquet_|ratio_)([\d.]+)y$")[0].astype(float)
            beyond_clq = clq > long["row"].map(horizon) + 1e-9
            if t == "cliquet_ladder":  # a ratio also carries its reference's note
                lv_beyond = clq > float(horizon[LV_LABEL]) + 1e-9
                beyond_clq |= long["column"].str.startswith("ratio_") & lv_beyond
            want = (long["column"].str.startswith("fwd_") & win.eq(1.0)) | beyond_clq
            assert_noted(long, want, t)
        elif t.startswith("regression_") and t != "regression_meta":
            # the rows are named after the baseline's models: back to the labels by coordinates
            key = long["row"].str.split(" / ").str[1]
            model = pd.Series(
                [label_of[coordinates(r.axes(t, row))] for row in long["row"]], index=long.index
            )
            win = model.map(lambda m: flags[f"{m} | {window}"])
            clq = key.str.extract(r"^cliquet_([\d.]+)y$")[0].astype(float)
            want = (key.isin(fwd_keys) & win.eq(1.0)) | (clq > model.map(horizon) + 1e-9)
            assert_noted(long, want, t)
        else:
            assert not noted(long).any(), t
            continue
        covered.add(t)
    assert {"headline", "smile_1y2y", "wing_vs_lv_1y2y", "wing_z_1y2y", "wing_lv_1y2y"} <= covered


def check_s4_beyond(r: Results) -> None:
    """Every S4 number of a model whose notes run beyond its horizon, and every number of a
    forward window ending beyond it, says so, in any table; no other number does."""
    hz = r.pivot("horizon")
    models = set(hz.index[hz["notes_beyond"] == 1.0])
    assert models and LV_LABEL not in models
    flags = r.pivot("fwd_skew")["beyond_horizon"]
    assert (flags == 1.0).any()
    covered = set()
    for t in r.tables():
        long = r.long(t)
        in_window = long["row"].map(flags).eq(1.0)
        want = in_window if t == "fwd_skew" else long["row"].isin(models)
        assert_noted(long, want, t)
        if want.any():
            covered.add(t)
    assert {
        "autocall",
        "phoenix",
        "legs_autocall",
        "legs_phoenix",
        "diff_autocall",
        "diff_phoenix",
        "se_autocall",
        "se_phoenix",
        "horizon",
        "fwd_skew",
    } <= covered, covered


def check_lv_grid(run: StudyRun) -> None:
    """S2 / S3: the pairing table and, when the local vol steps on the reference LSV's grid,
    the disclosure in study.md."""
    r = run.results
    pairing = r.pivot("pairing")
    assert LV_LABEL in pairing.index and (pairing["same_grid"] == 1.0).all()
    lv = pairing.loc[LV_LABEL]
    lsv_steps = pairing.drop(index=LV_LABEL)["steps"]
    assert (lsv_steps == lv["steps"]).all()
    text = (run.out_dir / "study.md").read_text()
    if lv["steps"] != lv["own_steps"]:
        sentence = lv_grid_sentence(r)
        assert f"({lv['steps']:.0f} steps to the maturity against the {lv['own_steps']:.0f}" in (
            sentence
        )
        assert sentence in text
    assert "The quadrature error is " in text


# --------------------------------------------------------------------------------------------
# per-study checks of the fast runs (called by the walking test)
# --------------------------------------------------------------------------------------------


def check_s1(run: StudyRun) -> None:
    r = run.results
    for t in (
        "headline",
        "forward_vols",
        "smile_1y2y",
        "wing_vs_lv_1y2y",
        "wing_z_1y2y",
        "cliquet_ladder",
        "provenance",
        "regression_meta",
    ):
        assert t in r.tables(), t
    prov = r.pivot("provenance")
    assert len(prov) == 4 and (prov["from_store"] == 1.0).all()
    assert set(r.frame.loc[r.frame["table"] == "headline", "source"].str[:6]) == {"store:"}
    # the toy budgets are not the baseline's: the study says so
    assert (r.pivot("regression_meta")["match"] == 0.0).all()
    assert "The budgets differ from the baseline's" in (run.out_dir / "study.md").read_text()
    assert len(run.manifest["store_points"]) == 4 and run.manifest["particles"] == [TOY_PARTICLES]
    # the local vol is the reference of the ladder
    assert r.value("cliquet_ladder", "LV (ω=0)", "ratio_1y")[0] == 1.0
    check_s1_beyond(r)
    assert (
        "(model, window) forward-vol rows end beyond the calibration horizon"
        in (run.out_dir / "study.md").read_text()
    )


def check_s2(run: StudyRun) -> None:
    r = run.results
    ratios = r.pivot("vko_ratio")
    cols = [c for c in ratios.columns if c.startswith("ratio_") and not c.endswith("_stderr")]
    v = ratios[cols].to_numpy(float)
    assert np.all((v > 0) & (v <= 1)) and np.all(np.diff(v, axis=1) > 0)
    itm = r.pivot("itm_rv")
    assert np.all(itm["p10"] < itm["p50"]) and np.all(itm["p50"] < itm["p90"])
    assert np.all(itm["p50_stderr"] > 0)
    summary = {rec["row"]: rec["value"] for rec in r.long("realised_summary").to_dict("records")}
    assert summary["maturity date"] > summary["last close"] > summary["start date"]
    assert "the payoff is not observable" in (run.out_dir / "study.md").read_text()
    assert [a["path"] for a in run.manifest["artefacts"]] == [str(HISTORY_REL)]
    mech = r.pivot("mechanism")
    for pred, agree in (("predicted_p", "agrees_p"), ("predicted_w", "agrees_w")):
        assert set(mech[pred]) <= {-1.0, 0.0, 1.0}
        resolved = (mech[pred] != 0.0) & (mech["significant"] == 1.0)
        assert np.isfinite(mech.loc[resolved, agree]).all()
        assert not np.isfinite(mech.loc[~resolved, agree]).any()
    ses = r.pivot("lsv_minus_lv_se")
    paired = ses[
        [c for c in ses.columns if c.startswith("se_paired_") and not c.endswith("_stderr")]
    ].to_numpy(float)
    quad = ses[
        [c for c in ses.columns if c.startswith("se_quadrature_") and not c.endswith("_stderr")]
    ].to_numpy(float)
    assert np.all(paired < quad)  # positively correlated on the shared random numbers
    assert " ± 0.0 " not in (run.out_dir / "study.md").read_text()
    check_lv_grid(run)


def check_s3(run: StudyRun) -> None:
    r = run.results
    ud = r.pivot("updown")
    assert len(ud) == 4
    assert np.all(ud["up_100"] < ud["var"]) and np.all(ud["var"] < ud["down_100"])
    assert len(r.rows("gyongy_floor")) == 3
    cells = r.pivot("gyongy")
    assert len(cells) == 3 * 3 * 3  # LSV models x legs x barriers
    assert set(cells["verdict"]) <= {0.0, 1.0, 2.0}
    np.testing.assert_allclose(cells["z"], cells["d"] / cells["d_stderr"], rtol=1e-12)
    assert np.all(cells["tolerance"] > 0)
    # the verdict is a function of z and the tolerance, never a fixed sentence
    zero = cells["z"].abs() <= 2.0
    within = cells["d"].abs() <= cells["tolerance"]
    expected = np.where(zero, 0.0, np.where(within, 1.0, 2.0))
    np.testing.assert_array_equal(cells["verdict"].to_numpy(), expected)
    text = (run.out_dir / "study.md").read_text()
    assert "the zero line." not in text and "O(dt)" in text
    for leg in ("corridor up", "conditional up", "conditional down"):
        n = [
            int((cells.loc[[i for i in cells.index if f"| {leg} " in i], "verdict"] == k).sum())
            for k in (0.0, 1.0, 2.0)
        ]
        assert (
            f"{n[0]} consistent with zero, {n[1]} significant but within the floor, "
            f"{n[2]} beyond the floor" in text
        )
    # paired differences: the quadrature error is reported beside the paired one
    assert np.all(cells["d_se_quadrature"] > 0) and np.all(cells["d_correlation"] > 0)
    check_lv_grid(run)
    order = r.pivot("monitoring_order")
    ratios = order["ratio"].to_numpy(float)
    assert np.isfinite(ratios).sum() == 4 and np.all(order["D"] > 0)
    p_ko = r.pivot("ko_var")["p_ko_110"].to_numpy(float)
    assert np.all((p_ko > 0) & (p_ko < 1))
    from volsto.studies.catalogue.s3_conditional_variance import KO_NOT_A_THEOREM

    assert KO_NOT_A_THEOREM in (run.out_dir / "study.md").read_text()
    assert (
        KO_NOT_A_THEOREM
        == "K_KO > K_var holds under negative skew and a non-inverted term structure but is not "
        "a theorem."
    )


def check_s4(run: StudyRun) -> None:
    r = run.results
    for slug in ("autocall", "phoenix"):
        prices = values(r, slug, "price")
        assert len(prices) == 4
        legs = r.pivot(f"legs_{slug}")
        # the American knock-in's European counterpart is reported beside the legs, not in them
        leg_cols = [
            c for c in legs.columns if not c.endswith("_stderr") and c != "leg:put_european"
        ]
        for model, (price, _) in prices.items():
            assert abs(legs.loc[model, leg_cols].sum() - price) < 1e-8, (slug, model)
        p_ki = r.pivot(slug)["p_ki"].to_numpy(float)
        assert np.all((p_ki > 0) & (p_ki < 1))
    assert "fwd_skew_lv" in r.tables() and "fwd_skew" in r.tables()
    missing = r.record("setup", "models without stored skew_T", "value")
    assert missing["value"] == 4.0 and "--risk light" in str(missing["note"])
    assert set(r.frame.loc[r.frame["table"] == "autocall", "source"].str[:6]) <= {
        "cache:",
        "comput",
    }
    # every note priced here: the differences are paired and say so
    diff = r.long("diff_autocall")
    assert set(diff["note"].str[: len(PAIRED_NOTE)]) == {PAIRED_NOTE}
    se = r.pivot("se_autocall")
    assert np.all(se["price_paired"] < se["price_quadrature"])
    # the toy leverage stops at 1y: the 3y notes and the forward windows are flagged
    hz = r.pivot("horizon")
    assert hz.loc["LV (ω=0)", "notes_beyond"] == 0.0
    assert (hz.drop(index="LV (ω=0)")["notes_beyond"] == 1.0).all()
    assert all(
        "beyond the leverage's 1y calibration horizon" in n
        for n in r.long("autocall").query("row != 'LV (ω=0)'")["note"]
    )
    fs = r.pivot("fwd_skew")
    lsv_rows = [i for i in fs.index if not i.startswith("LV")]
    assert (fs.loc[lsv_rows, "beyond_horizon"] == 1.0).all()
    assert (fs.drop(index=lsv_rows)["beyond_horizon"] == 0.0).all()
    text = (run.out_dir / "study.md").read_text()
    assert "3 model(s) price the 3y notes beyond their leverage's calibration horizon" in text
    # deterministic zeros are exact, never "0 ± 0"
    exp = r.long("fwd_skew_lv")
    zeros = exp[exp["value"] == 0.0]
    assert len(zeros) > 0 and zeros["exact"].all()
    assert re.search(r"(?<![\d.])0 ± 0(?![\d.])", text) is None
    check_s4_beyond(r)
    # the missing LSV skew_T: the store's tents would not sit on the observation dates
    assert "the two are not like-for-like" in text


STUDY_CHECKS: dict[str, Callable[[StudyRun], None]] = {
    "s1_forward_vol_fast": check_s1,
    "s2_vko_fast": check_s2,
    "s3_conditional_variance_fast": check_s3,
    "s4_autocall_fast": check_s4,
}


# --------------------------------------------------------------------------------------------
# the walking test
# --------------------------------------------------------------------------------------------


def test_every_catalogue_study_has_a_fast_config() -> None:
    names = {p.stem.removesuffix("_fast") for p in FAST_CONFIGS}
    assert {"s1", "s2", "s3", "s4"} <= names, names
    for stem in ("s1", "s2", "s3", "s4"):
        assert (CATALOGUE / f"{stem}.yaml").is_file()


@pytest.mark.parametrize("config_path", FAST_CONFIGS, ids=[p.stem for p in FAST_CONFIGS])
def test_catalogue_fast_config_runs_without_calibration(
    config_path: Path, request: pytest.FixtureRequest, tmp_path: Path
) -> None:
    build = fast_build(config_path, request)
    config = runner.load_study_config(config_path)
    assert config.mode == "fast"
    before = cache_snapshot(build.cache_root)
    run = run_config(config_path, build, tmp_path)
    assert run.exit_code == 0, run.manifest.get("latex")
    m = run.manifest
    assert m["recalibrated"] is False and m["mode"] == "fast"
    assert cache_snapshot(build.cache_root) == before, "the toy cache changed"
    toy_keys = set(LeverageCache(build.cache_root).manifest()["key"])
    assert {k["key"] for k in m["cache_keys"]} <= toy_keys
    assert set(m["particles"]) <= {TOY_PARTICLES}
    assert m["tables"] and m["figures"]
    out = run.out_dir
    for t in m["tables"]:
        assert (out / "tables" / f"{t}.tex").is_file()
    for f in m["figures"]:
        assert (out / "figures" / f"{f}.pdf").is_file() and (out / "figures" / f"{f}.png").is_file()
    lines = (out / "study.md").read_text().splitlines()
    module = runner.load_runner_module(config.runner)
    assert lines[0] == f"# {module.TITLE}"
    assert lines[2] == f"**Question.** {runner.study_question(config, module)}"
    assert "\n" not in runner.study_question(config, module)
    # tables and figures are functions of results.parquet alone
    copy = tmp_path / "render"
    shutil.copytree(out, copy)
    shutil.rmtree(copy / "tables")
    shutil.rmtree(copy / "figures")
    runner.render_study(copy)
    for t in m["tables"]:
        assert (copy / "tables" / f"{t}.tex").read_bytes() == (
            out / "tables" / f"{t}.tex"
        ).read_bytes()
    assert sorted(p.name for p in (copy / "figures").iterdir()) == sorted(
        p.name for p in (out / "figures").iterdir()
    )
    assert (copy / "study.md").read_text() == (out / "study.md").read_text()
    check = STUDY_CHECKS.get(config.name)
    if check is not None:
        check(run)


# --------------------------------------------------------------------------------------------
# S1
# --------------------------------------------------------------------------------------------


def test_s1_prices_a_point_missing_from_the_store(toy_build: Any, tmp_path: Path) -> None:
    toy = toy_build.require()
    store = tmp_path / "store"
    shutil.copytree(toy.store_root, store)
    from volsto.viewers.store import ResultsStore, StoreReader

    points = StoreReader(store).points()
    gone = points[points["label"] == "placeholder 1F nu=0.5 rho=-0.7 kappa=1.5"]
    point_id = str(gone["point_id"].iloc[0])
    shutil.rmtree(ResultsStore(store).point_dir(point_id))
    config = CATALOGUE / "s1_fast.yaml"
    before = cache_snapshot(toy.cache_root)
    outputs = outputs_copy(toy, tmp_path)

    run = run_config(config, toy, tmp_path / "a", store=store, outputs=outputs)
    prov = run.results.pivot("provenance")
    label = "1F ω=1 ρ=-0.7 κ=1.5"  # noqa: RUF001
    assert prov.loc[label, "from_store"] == 0.0 and prov["from_store"].sum() == 3.0
    src = run.results.record("headline", label, "fwd_atm_vol")["source"]
    assert src == f"cache:{point_id}"
    assert run.manifest["records"]["store_models"] == 3
    assert cache_snapshot(toy.cache_root) == before

    # store-only: the missing point is a requirement, nothing is computed
    buf = io.StringIO()
    argv = [
        "run",
        str(config),
        "--grid",
        str(toy.grid_path),
        "--store",
        str(store),
        "--cache",
        str(toy.cache_root),
        "--outputs",
        str(outputs),
        "--out",
        str(tmp_path / "b"),
        "--set",
        "price_missing=false",
    ]
    with contextlib.redirect_stdout(buf):
        code = runner.main(argv)
    text = buf.getvalue()
    print(text)
    assert code == runner.EXIT_MISSING
    assert f"--only {point_id} --resume" in text and "toy.yaml" in text
    assert not (tmp_path / "b").exists()


def test_s1_reads_the_m4_baseline_file_as_data() -> None:
    base = s1_forward_vol.read_baseline(
        {"file": "tests/test_m4_regression.py", "variable": "PLACEHOLDER_BASELINES"}
    )
    assert set(base["values"]) == {"LV (ω=0)", "1F ω=1", "1F ω=2", "1F ω=3", "2F Table 8.2"}
    assert all(len(v) == 16 for v in base["values"].values())
    assert base["constants"] == {
        "HEADLINE_N_PARTICLES": 800_000,
        "HEADLINE_N_PATHS": 400_000,
        "HEADLINE_SEED": 2024,
    }
    assert s1_forward_vol.key_group("cliquet_2y")[3] == 0.02
    assert s1_forward_vol.key_group("vko_ratio_20")[3] == 0.002
    assert s1_forward_vol.key_group("atm_vol")[2:] == (100.0, 0.02)


# --------------------------------------------------------------------------------------------
# S2
# --------------------------------------------------------------------------------------------


def test_s2_itm_quantiles_and_bootstrap() -> None:
    rng = np.random.default_rng(3)
    n = 40_000
    sigma = rng.lognormal(np.log(0.25), 0.3, n)
    itm = rng.random(n) < 0.4
    point, reps = s2_vko.itm_quantiles(sigma, itm, True, 5, 200)
    sub = sigma[itm]
    for k, q in s2_vko.QUANTILES:
        assert point[k] == np.percentile(sub, 100 * q, method="inverted_cdf")
    # the median's asymptotic error (independent members here)
    m = sub.size
    f = np.exp(-0.5 * ((np.log(point["p50"]) - np.log(0.25)) / 0.3) ** 2) / (
        point["p50"] * 0.3 * math.sqrt(2 * math.pi)
    )
    asymptotic = math.sqrt(0.25 / m) / f
    se = float(np.std(reps[:, 1], ddof=1))
    assert 0.7 < se / asymptotic < 1.4, (se, asymptotic)
    # the same seed resamples the same pairs: a path set compared with itself has zero spread,
    # and a perturbed copy is paired replicate by replicate
    _, again = s2_vko.itm_quantiles(sigma, itm, True, 5, 200)
    np.testing.assert_array_equal(reps, again)
    _, shifted = s2_vko.itm_quantiles(sigma * 1.01, itm, True, 5, 200)
    diff = shifted[:, 1] - reps[:, 1]
    assert float(np.std(diff, ddof=1)) < 0.2 * se


def test_paired_difference_against_the_quadrature_error() -> None:
    rng = np.random.default_rng(11)
    n = 50_000
    common = rng.standard_normal(n)
    a = common + 0.3 * rng.standard_normal(n)
    b = 0.8 * common + 0.3 * rng.standard_normal(n) + 0.1
    d = paired_difference(mean_estimate(a), mean_estimate(b), same_grid=True)
    assert d.value == pytest.approx(float(a.mean() - b.mean()), rel=1e-12)
    assert d.stderr == pytest.approx(float((a - b).std(ddof=1) / math.sqrt(n)), rel=1e-12)
    assert d.stderr_quadrature == pytest.approx(
        math.hypot(a.std(ddof=1), b.std(ddof=1)) / math.sqrt(n), rel=1e-12
    )
    assert d.correlation == pytest.approx(float(np.corrcoef(a, b)[0, 1]), rel=1e-9)
    assert d.stderr < 0.5 * d.stderr_quadrature and d.z == d.value / d.stderr
    # the ratio estimator's influence gives the delta-method stderr of ratio_of_means
    from volsto.analytics.conditional_variance import ratio_of_means

    num = 2.0 + a
    den = 3.0 + 0.1 * b
    r = ratio_estimate(num, den)
    assert r.pair == pytest.approx(ratio_of_means(num, den), rel=1e-12)
    # an estimate minus itself is exactly zero with zero paired error
    e = Estimate(1.0, a - a.mean())
    zero = paired_difference(e, e, same_grid=True)
    assert zero.value == 0.0 and zero.stderr == 0.0


def test_quadrature_comparison_from_the_numbers() -> None:
    paired = np.array([1.0, 2.0, 4.0, np.nan])
    quad = np.array([1.5, 1.8, 6.0, 1.0])
    corr = np.array([0.4, -0.1, 0.5, 0.2])
    text = quadrature_comparison(paired, quad, corr)
    assert text.startswith("The quadrature error is 0.90 to 1.50 times the paired stderr")
    assert "larger one in 2 of 3 differences" in text and "smaller one in 1," in text
    assert "correlation runs from -0.10 to +0.50" in text
    only_larger = quadrature_comparison(paired[:1], quad[:1], corr[:1])
    assert "smaller" not in only_larger and "1 of 1 differences" in only_larger
    assert quadrature_comparison(paired[3:], quad[3:], corr[3:]) == ""


def _pairing_results(b: ResultsBuilder, steps: float, own: float) -> None:
    for row, st, ow in ((LV_LABEL, steps, own), ("1F ω=1", steps, steps)):
        for col, v in (("steps", st), ("own_steps", ow), ("same_grid", 1.0)):
            b.add_exact("pairing", row, col, v, unit="", source="computed")


def test_s3_store_check_discloses_the_local_vol_grid() -> None:
    """The toy store has no conditional numbers, so the S3 store-check paragraph is checked on
    hand-made results: identical rows are counted, the local-vol shifts are listed with the
    grid sentence, and the quadrature error is not called a bound."""
    b = ResultsBuilder()
    _pairing_results(b, 886.0, 756.0)
    table = "store_check_vol"
    for row, d, se in (("LV (ω=0) / upvar_100", 0.006, 0.003), ("1F ω=1 / upvar_100", 0.0, 0.0)):
        if se:
            b.add(table, row, "diff", d, se, unit="vol pts", source="computed")
        else:
            b.add_exact(table, row, "diff", 0.0, unit="vol pts", source="computed")
    b.add(
        "store_check_ratio",
        "LV (ω=0) / kovar_110_p_ko",
        "diff",
        0.001,
        0.0005,
        unit="1",
        source="computed",
    )
    r = b.build()
    lines = s3_conditional_variance._store_check_lines(r)
    text = "\n".join(lines)
    assert lines[0] == "## Store check"
    assert "1 of 3 inline numbers are identical" in text
    assert "The 2 local-vol rows among them differ for a second reason" in text
    assert "(886 steps to the maturity against the 756 it would use alone" in text
    assert "upvar_100 0.006 ± 0.003 vp" in text and "kovar_110_p_ko" in text
    assert "bound" not in text
    assert "{{table:store_check_vol}}" in lines and "{{table:store_check_ratio}}" in lines
    # a local vol on its own grid: no grid sentence
    b2 = ResultsBuilder()
    _pairing_results(b2, 756.0, 756.0)
    assert lv_grid_sentence(b2.build()) == ""


def test_s4_skew_tents_like_for_like() -> None:
    def results(store: tuple[float, ...], dates: tuple[float, ...]) -> Results:
        b = ResultsBuilder()
        for T in store:
            b.add("skew_store", "1F ω=1", f"skew_T_{T:g}y", 0.1, 0.01, unit="%", source="store:abc")
        for T in dates:
            b.add(
                "fwd_skew_lv",
                "autocall / product",
                f"exposure_{T:g}y",
                0.1,
                0.01,
                unit="%",
                source="computed",
            )
        return b.build()

    text = s4_autocall._like_for_like(results((0.25, 1.0, 3.0), (1.0, 2.0, 3.0)))
    assert "not like-for-like" in text
    assert "risk pillars (0.25y, 1y, 3y)" in text and "observation dates (1y, 2y, 3y)" in text
    assert text.endswith("common neighbours: none.")
    shared = s4_autocall._like_for_like(results((0.5, 1.0, 2.0, 3.0), (1.0, 2.0, 3.0)))
    assert shared.endswith("common neighbours: 2y, 3y.")
    same = s4_autocall._like_for_like(results((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)))
    assert "comparable tent by tent" in same


def test_shell_block_round_trip() -> None:
    import subprocess

    cmd = (
        "volsto-precompute --grid configs/grids/toy.yaml --store /private/var/folders/aa/"
        + "x" * 150
        + "/store --cache cache --only abc def --resume --risk light"
    )
    block = shell_block(cmd)
    lines = block.splitlines()
    assert lines[0] == lines[-1] == "```"
    assert all(len(line) <= 68 for line in lines)
    body = "\n".join(lines[1:-1])
    joined = subprocess.run(
        ["sh", "-c", f"set -- {body}\nprintf '%s ' \"$@\""],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert joined == cmd


def test_s2_realised_outcome_by_hand(tmp_path: Path) -> None:
    frame = write_synthetic_history(tmp_path / "h.csv", n_days=40)
    terms = {"strike": 1.0, "maturity": 1.0, "per_year": 252, "vol_ko": [0.2, 0.3]}
    out = s2_vko.realised_outcome(frame, "2022-07-05", 1.0, terms)
    ls = frame.drop_duplicates("date")["ln_spot"].to_numpy()[2:]  # 07-01 and 07-04 dropped
    r = np.diff(ls)
    assert out["n_obs"] == r.size == 37 and out["n_life"] == 252
    assert out["sum_sq"] == pytest.approx(float(np.sum(r * r)), rel=1e-12)
    assert out["rv_to_date"] == pytest.approx(math.sqrt(252 * np.sum(r * r) / r.size))
    assert out["spot_return"] == pytest.approx(math.exp(ls[-1] - ls[0]) - 1.0)
    assert out["first"] == pd.Timestamp("2022-07-05") and out["atm_mark"] == 0.24
    assert out["maturity"] > out["last"]


# --------------------------------------------------------------------------------------------
# S4 on the repository store (read only; skipped without it)
# --------------------------------------------------------------------------------------------


def _repository_ready() -> bool:
    from volsto.viewers.store import StoreReader

    if not (REPO_STORE.is_dir() and REPO_CACHE.is_dir()):
        return False
    df = StoreReader(REPO_STORE).products()
    return not df.empty and bool((df["key"] == "autocall 3y:price").sum() >= 5)


@pytest.mark.skipif(not _repository_ready(), reason="repository store without the M6 cells")
def test_s4_reads_the_stored_m6_fractions_as_percent(tmp_path: Path) -> None:
    config = runner.load_study_config(
        CATALOGUE / "s4.yaml", ConfigOverrides(sets=("lv_exposure=null",))
    )
    before = cache_snapshot(REPO_CACHE)
    run = runner.run_study(config, out_dir=tmp_path / "s4", latex_check=False)
    r = run.results
    price, se = r.value("autocall", "LV (ω=0)", "price")
    assert 90.0 < price < 100.0 and 0 < se < 0.1  # % of notional, not a fraction
    rec = r.record("legs_phoenix", "LV (ω=0)", "leg:put")
    assert rec["unit"] == "% notional" and "x 100" in rec["note"]
    assert set(r.frame["source"].str[:6]) <= {"store:", "comput"}
    assert "fwd_skew_lv" not in r.tables() and len(r.rows("fwd_skew")) == 10
    assert cache_snapshot(REPO_CACHE) == before


# --------------------------------------------------------------------------------------------
# requirement lines on the repository roots (no computation; skipped without them)
# --------------------------------------------------------------------------------------------


@pytest.mark.skipif(not _repository_ready(), reason="repository store / cache absent")
@pytest.mark.parametrize(
    ("config_name", "grid"),
    [("s1_grid.yaml", "configs/grids/default.yaml"), ("s3.yaml", "configs/grids/s3_ko_var.yaml")],
)
def test_full_configs_print_their_own_grid(
    config_name: str, grid: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    buf = io.StringIO()
    before = cache_snapshot(REPO_CACHE)
    with contextlib.redirect_stdout(buf):
        code = runner.main(
            ["run", str(CATALOGUE / config_name), "--out", str(tmp_path / "x"), "--no-latex-check"]
        )
    text = buf.getvalue()
    print(text)
    assert code == runner.EXIT_MISSING
    lines = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("volsto-precompute")]
    assert len(lines) == 1, lines
    assert lines[0].startswith(
        f"volsto-precompute --grid {grid} --store outputs/store --cache cache"
    )
    assert lines[0].endswith("--resume")
    assert not (tmp_path / "x").exists()
    assert cache_snapshot(REPO_CACHE) == before


# --------------------------------------------------------------------------------------------
# the toy marking build (the S5 fixture)
# --------------------------------------------------------------------------------------------


def test_toy_marking_build(toy_marking_build: Any) -> None:
    from volsto.viewers.store import StoreReader

    build = toy_marking_build
    print(
        f"toy marking build: {build.total_wall_s:.1f} s by {build.built_by} "
        f"(shards {build.shards}, wall {build.wall_s}); error {build.error!r}"
    )
    build.require()
    assert build.shards == ("1/1",) and build.grid_path.name == "toy_marking.yaml"
    points = StoreReader(build.store_root).points()
    assert sorted(points["mode"]) == ["lv", "marking", "marking", "marking", "marking"]
    marking = points[points["mode"] == "marking"]
    feasible = marking[marking["status"] != "infeasible"]
    print(marking[["label", "status", "nu", "theta", "rho_SX1", "cache_key"]].to_string())
    assert len(build.calibrating_messages()) == len(feasible)
    keys = set(LeverageCache(build.cache_root).manifest()["key"])
    assert set(feasible["cache_key"]) == keys
    assert (build.outputs_root / "m7").is_dir()
