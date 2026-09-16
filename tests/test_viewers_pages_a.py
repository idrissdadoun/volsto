"""Pages 1–3 of the viewer (M9 Part 2, task PA) rendered headless with ``streamlit.testing``'s
``AppTest`` against the synthetic store of :mod:`tests._synthetic_store` — no calibration, no
Monte Carlo — plus unit checks of the page helpers in ``volsto/viewers/pages/_common.py``.

* page 1 with a synthetic leverage entry written **by file** into a temporary cache (a
  ``LeverageFunction`` + ``CalibrationReport`` built from arrays, no calibration): the surface,
  local-vol, leverage and error-map figures render, the leverage caption states the particle
  count / seed and that no per-cell stderr is stored; and without the cache entry the page prints
  the ``volsto-precompute`` command and the empty cache root is never created;
* page 2: smiles, triple and put-wing sections, tables with Excel buttons, the put-wing quote,
  the bar traces' ``± stderr`` text hover-only (``textposition = "none"``); and a store copy
  whose stored ``t1`` is perturbed by 1e-10 — the page's window filters match with ``np.isclose``
  like ``_common``'s, so both windows still render and the perturbed ``t1`` reaches the table;
* page 3: fits (with the ``misfit`` column), term structures on the canonical window labels,
  ratio figures with the deck band, the M7 ratio section (the fixture adds the ``fwd/spot@…``
  columns to the synthetic fits CSV) and the snapshot surface (marking point) selected through
  the widget; a fits CSV with a ratio column lacking its stderr twin and a NaN row: the column
  is named in an ``st.info`` and not plotted, the NaN row is dropped from the figure and kept in
  the table; and a fits CSV without ``skew_eps`` — a column the M7 ratio figure is keyed on: the
  section names it in an ``st.info``, plots nothing and still shows the table (no KeyError);
* the stderr contract of the three pages, each trigger on a pathological copy of the synthetic
  store / cache (:func:`_store_copy` / :func:`_cache_copy`): a NaN stderr beside a value (the
  row is left out of the figure, counted in a caption and kept as written in the table — the
  page-1 error map and z map blank the same cell, so the two figures agree) and the stderr
  column gone from the artefact altogether (an ``st.info`` naming the column and the producing
  command, no figure, the table still exported — a ``KeyError`` mid-render before the guard);
* every page test pins **every** table to one Excel button and **every** figure to one PNG + one
  SVG button (or one HTML button when the image engine is unavailable) — equalities, not bounds;
* helpers: the weighted quadratic fit (exact recovery, covariance stderr, too few strikes), the
  misfit flag on a deliberately non-quadratic smile, the put-wing spread against
  :func:`volsto.analytics.forward_smile.put_wing_table`, the forward / spot ratio arithmetic,
  the spot 90/110 skew convention of ``fit_2f``, the window labels, the store stamp (per-point
  manifests and the cache root invalidate the caches, not only the merged manifest);
* an AST check that none of the four modules imports a calibrating or simulating symbol, nor
  anything from ``volsto.calibration`` / ``volsto.engine`` / ``volsto.risk`` / ``volsto.studies``.

Wall clock a few seconds per page (the surface build dominates).
"""

from __future__ import annotations

import ast
import base64
import json
import math
import os
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from _synthetic_store import make_synthetic_outputs, make_synthetic_store

from volsto.analytics.forward_smile import ForwardSmile, put_wing_table
from volsto.calibration.diagnostics import CalibrationReport
from volsto.market.curves import ForwardCurve
from volsto.models.leverage import LeverageFunction
from volsto.viewers import api
from volsto.viewers.components import image_engine_ok
from volsto.viewers.config import ViewerConfig
from volsto.viewers.pages import PAGES_DIR
from volsto.viewers.pages import _common as common

ROOT = Path(__file__).resolve().parents[1]
PAGE_FILES = ("1_surface_model.py", "2_forward_smile.py", "3_forward_skew.py")
#: Symbols a viewer page must never import (the owner's review lens: MonteCarlo, calibrate,
#: get_or_calibrate, simulate, risk_report, run_headline) — matched as substrings of every
#: imported module / name, so ``calibrate`` also catches ``calibrate_leverage`` and ``fit_2f``
#: the calibration module.
FORBIDDEN = (
    "LeverageCache",
    "get_or_calibrate",
    "MonteCarlo",
    "fit_2f",
    "calibrate",
    "simulate",
    "risk_report",
    "run_headline",
)
#: Packages a viewer page must not import from at all (calibration, simulation, risk, studies).
FORBIDDEN_PACKAGES = ("volsto.calibration", "volsto.engine", "volsto.risk", "volsto.studies")


# --------------------------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------------------------


def _write_leverage_entry(cache_root: Path, key: str) -> None:
    """A synthetic cache entry (leverage.npz + diagnostics.json) — arrays only."""
    entry = cache_root / key
    entry.mkdir(parents=True)
    times = np.linspace(0.05, 3.0, 24)
    k_grid = np.linspace(-2.0, 2.0, 41)
    values = 1.0 + 0.15 * np.sin(k_grid)[None, :] * np.exp(-times)[:, None]
    lev = LeverageFunction(
        times,
        k_grid,
        values,
        ForwardCurve.flat(100.0, 0.02, 0.0),
        {"wall_time": 2.2, "code_tag": "synthetic", "git_commit": "0000000"},
    )
    lev.save(entry / "leverage.npz")
    rows = []
    for T in (1 / 12, 0.25, 0.5, 1.0, 2.0, 3.0):
        for k in (-0.2, -0.1, 0.0, 0.1, 0.2):
            err = 0.05 * k * T
            rows.append(
                {
                    "T": T,
                    "k": k,
                    "K": 100.0 * math.exp(k),
                    "cp": 1 if k >= 0 else -1,
                    "price": 5.0,
                    "price_stderr": 0.01,
                    "model_vol": 0.2 + err / 100,
                    "target_vol": 0.2,
                    "error_vp": err,
                    "stderr_vp": 0.02,
                }
            )
    CalibrationReport(pd.DataFrame(rows), pd.DataFrame(), 400_000, 2024, 130.0).save(
        entry / "diagnostics.json"
    )


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    base = tmp_path_factory.mktemp("pages_a")
    t0 = time.perf_counter()
    info = make_synthetic_store(base / "store")
    outs = make_synthetic_outputs(base / "outputs")
    # the M7 runner's fwd/spot ratio columns (stage 3) on the synthetic fits table
    fits = pd.read_csv(outs["marking_fits"])
    fits["fwd/spot@1y-into-1y"] = 1.41
    fits["fwd/spot_se@1y-into-1y"] = 0.05
    fits["fwd/spot@2y-into-1y"] = 1.22
    fits["fwd/spot_se@2y-into-1y"] = 0.06
    fits.to_csv(outs["marking_fits"], index=False)
    cfg_empty = ViewerConfig(
        cache_root=base / "cache_empty", store_root=base / "store", outputs_root=base / "outputs"
    )
    cfg_cache = ViewerConfig(
        cache_root=base / "cache", store_root=base / "store", outputs_root=base / "outputs"
    )
    grid = api.list_grid(cfg_cache)
    one = grid[grid["mode"] == "one_factor"].iloc[0]
    _write_leverage_entry(cfg_cache.cache_root, str(one["cache_key"]))
    print(f"\nsynthetic store, outputs and cache entry in {time.perf_counter() - t0:.2f} s")
    return {"cfg_empty": cfg_empty, "cfg_cache": cfg_cache, "base": base, **info}


def _run_page(name: str, cfg: ViewerConfig, monkeypatch: pytest.MonkeyPatch, base: Path) -> Any:
    from streamlit.testing.v1 import AppTest

    for var, value in cfg.to_env().items():
        monkeypatch.setenv(var, value)
    monkeypatch.setenv("VOLSTO_VIEWER_CONFIG", str(base / "no_viewer.yaml"))
    t0 = time.perf_counter()
    at = AppTest.from_file(str(PAGES_DIR / name), default_timeout=180).run()
    print(f"\n{name}: rendered in {time.perf_counter() - t0:.2f} s")
    assert not at.exception, [e.value for e in at.exception]
    return at


def _charts(at: Any) -> int:
    return len(at.get("plotly_chart"))


def _downloads(at: Any) -> list[str]:
    return [str(b.label) for b in at.download_button]


def _fig_specs(at: Any) -> list[dict[str, Any]]:
    """The plotly figure JSON of every chart on the page (``data`` traces and ``layout``)."""
    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]


def _bin_array(value: Any) -> np.ndarray:
    """A plotly array attribute of a chart spec as floats (plotly encodes ``z`` /
    ``customdata`` as base64 binary with a shape string)."""
    if isinstance(value, dict):
        flat = np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"]))
        return flat.reshape([int(n) for n in str(value["shape"]).split(",")])
    return np.asarray(value, dtype=float)


def _z_array(trace: dict[str, Any]) -> np.ndarray:
    """The ``z`` matrix of a plotly heatmap trace."""
    return _bin_array(trace["z"])


def _assert_exports(at: Any) -> None:
    """Every table has exactly one Excel button; every figure exactly one PNG and one SVG
    download (or exactly one HTML download each when the image engine is unavailable)."""
    labels = _downloads(at)
    n_tables, n_figs = len(at.dataframe), _charts(at)
    assert n_tables > 0 and n_figs > 0
    assert sum(lab.startswith("Excel: ") for lab in labels) == n_tables, labels
    n_img = sum(lab.startswith(("PNG: ", "SVG: ")) for lab in labels)
    n_html = sum(lab.startswith("HTML: ") for lab in labels)
    if image_engine_ok():
        assert (n_img, n_html) == (2 * n_figs, 0), (n_figs, labels)
    else:
        assert (n_img, n_html) == (0, n_figs), (n_figs, labels)


# --------------------------------------------------------------------------------------------
# pathological copies: the stderr triggers the pages must refuse to plot
# --------------------------------------------------------------------------------------------


def _store_copy(
    base: Path,
    name: str,
    table: str,
    edit: Callable[[pd.DataFrame], pd.DataFrame],
    *,
    only: str = "",
) -> Path:
    """A private copy of the synthetic store under ``base / name`` with ``edit`` applied to the
    ``table`` parquet of every point (or of the points whose id starts with ``only``): the
    reviewer's triggers — a NaN stderr beside a value, the stderr column gone."""
    store = base / name
    if not store.exists():
        shutil.copytree(base / "store", store)
        for path in sorted(store.glob(f"results/points/*/{table}.parquet")):
            if only and not path.parent.name.startswith(only):
                continue
            edit(pd.read_parquet(path)).to_parquet(path, index=False)
    return store


def _cache_copy(base: Path, name: str, edit: Callable[[pd.DataFrame], pd.DataFrame]) -> Path:
    """A private copy of the synthetic leverage cache under ``base / name`` with ``edit``
    applied to the ``vanillas`` table of every entry's ``diagnostics.json``."""
    cache = base / name
    if not cache.exists():
        shutil.copytree(base / "cache", cache)
        for path in sorted(cache.glob("*/diagnostics.json")):
            rep = CalibrationReport.load(path)
            CalibrationReport(
                edit(rep.vanillas), rep.varswaps, rep.n_paths, rep.seed, rep.wall_time, rep.extra
            ).save(path)
    return cache


def _cfg_with(base: Path, *, store: Path | None = None, cache: Path | None = None) -> ViewerConfig:
    return ViewerConfig(
        cache_root=cache or base / "cache_empty",
        store_root=store or base / "store",
        outputs_root=base / "outputs",
    )


# --------------------------------------------------------------------------------------------
# page 1
# --------------------------------------------------------------------------------------------


def test_page1_renders_with_cache(
    synthetic: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Surface heatmap + slices, local vol, leverage heatmap, error map and z map render from
    the synthetic cache entry; every table has an Excel button, every figure a download."""
    at = _run_page("1_surface_model.py", synthetic["cfg_cache"], monkeypatch, synthetic["base"])
    assert [t.value for t in at.title] == ["Surface & model"]
    assert _charts(at) == 6  # iv heatmap, slices, local vol, leverage, error map, z map
    assert len(at.dataframe) == 6
    _assert_exports(at)
    captions = " ".join(c.value for c in at.caption)
    assert "cache key" in captions and "no standard error" in captions
    # the leverage is a particle MC estimate without a stored per-cell stderr: said on the page
    # with the entry's particle count / seed (the synthetic entry records neither: 'not recorded'
    # for the seed, the point's n_particles for the count — never a silent number)
    assert "no per-cell stderr" in captions and "particle" in captions
    assert "800000 particles, seed not recorded" in captions
    assert not any(c.value.startswith("volsto-precompute") for c in at.code)


def test_page1_missing_leverage(synthetic: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the cache entry the page prints the precompute command, never calibrates and
    never creates the cache root."""
    cfg = synthetic["cfg_empty"]
    at = _run_page("1_surface_model.py", cfg, monkeypatch, synthetic["base"])
    codes = [c.value for c in at.code]
    assert any(c.startswith("volsto-precompute --grid") for c in codes)
    assert _charts(at) == 3  # the surface sections still render (exact values)
    _assert_exports(at)
    assert not cfg.cache_root.exists()


def test_page1_error_map_nan_stderr(
    synthetic: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagnostics entry with one NaN ``stderr_vp``: the cell is blank in the error map *and*
    in the z map (the two figures agree on which cells exist — before the guard the hover read
    'x ± nan (z=nan)'), the cell is counted in a caption and kept as written in the error
    table."""
    base = Path(str(synthetic["base"]))

    def nan_one(v: pd.DataFrame) -> pd.DataFrame:
        out = v.copy()
        out.loc[0, "stderr_vp"] = np.nan
        return out

    cfg = _cfg_with(base, cache=_cache_copy(base, "cache_nan_se", nan_one))
    at = _run_page("1_surface_model.py", cfg, monkeypatch, base)
    assert _charts(at) == 6
    specs = {f["layout"]["title"]["text"]: f for f in _fig_specs(at)}
    emap = specs["Model - target implied vol (vp) with MC stderr and z"]["data"][0]
    zmap = specs["z = error / stderr"]["data"][0]
    blank = np.isnan(_z_array(emap))
    assert int(blank.sum()) == 1, _z_array(emap)
    np.testing.assert_array_equal(blank, np.isnan(_z_array(zmap)))  # the two maps agree
    # the hover's ± stderr and z come from customdata: finite wherever a cell is coloured
    custom = _bin_array(emap["customdata"])
    drawn = ~blank
    assert np.isfinite(custom[..., 0][drawn]).all() and np.isfinite(custom[..., 1][drawn]).all()
    assert np.isnan(custom[..., 0][blank]).all()
    captions = " ".join(c.value for c in at.caption)
    assert "1 cell(s) with a NaN error or stderr are left out" in captions
    tab = next(d.value for d in at.dataframe if "error_vp_stderr" in d.value.columns)
    assert len(tab) == 30 and int(tab["error_vp_stderr"].isna().sum()) == 1
    _assert_exports(at)


def test_page1_error_map_without_stderr_column(
    synthetic: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A diagnostics entry written without ``stderr_vp`` at all (no ``error_vp_stderr`` from the
    read API): an ``st.info`` names the column and the precompute command, the error map and the
    z map are not drawn (a KeyError mid-render before the guard) and the error table is still
    exported."""
    base = Path(str(synthetic["base"]))
    cfg = _cfg_with(
        base,
        cache=_cache_copy(base, "cache_no_se", lambda v: v.drop(columns=["stderr_vp"])),
    )
    at = _run_page("1_surface_model.py", cfg, monkeypatch, base)
    infos = [i.value for i in at.info]
    hit = [i for i in infos if "error_vp_stderr" in i]
    assert len(hit) == 1 and "volsto-precompute" in hit[0], infos
    titles = [f["layout"]["title"]["text"] for f in _fig_specs(at)]
    assert "z = error / stderr" not in titles and _charts(at) == 4
    assert any(lab.endswith("error table") for lab in _downloads(at))
    tab = next(d.value for d in at.dataframe if "error_vp" in d.value.columns)
    assert "error_vp_stderr" not in tab.columns and len(tab) == 30
    captions = " ".join(c.value for c in at.caption)
    assert "no max z" in captions  # the report's max z needs the per-cell stderr: never invented
    _assert_exports(at)


# --------------------------------------------------------------------------------------------
# page 2
# --------------------------------------------------------------------------------------------


def test_page2_renders(synthetic: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Smiles (two windows), the triple (two windows) and the put-wing figure with the quote."""
    cfg = synthetic["cfg_empty"]
    at = _run_page("2_forward_smile.py", cfg, monkeypatch, synthetic["base"])
    assert [t.value for t in at.title] == ["Forward smile"]
    assert _charts(at) == 5  # two smiles, two triples, the put-wing spread
    assert len(at.dataframe) == 3
    _assert_exports(at)
    md = " ".join(m.value for m in at.markdown)
    assert "Put wing" in md and "vp" in md
    # the bars' '± stderr' strings are hover-only: plotly's default textposition 'auto' would
    # print them on every bar
    bars = [tr for f in _fig_specs(at) for tr in f["data"] if tr.get("type") == "bar"]
    assert len(bars) == 5  # 2 models x 2 windows + the spread bars
    assert all(tr["textposition"] == "none" for tr in bars), bars
    assert all("±" in tr["text"][0] and "%{text}" in tr["hovertemplate"] for tr in bars)
    assert not cfg.cache_root.exists()
    # the snapshot surface holds the marking point only: no LV overlay, still renders
    at.selectbox(key="p2_surface").select("spx_2022-09-15").run()
    assert not at.exception, [e.value for e in at.exception]
    infos = " ".join(i.value for i in at.info)
    assert "LV reference is not overlaid" in infos or "at least two points" in infos


def test_page2_window_isclose(synthetic: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """A store copy whose stored ``t1`` is ``1.0 + 1e-10`` (and ``2.0 + 1e-10``): the window
    filters of the smiles and the triple match with ``np.isclose`` like ``_common``'s helpers,
    so both windows still render (five figures, no 'No stored … smile' notice) and the smiles
    table shows the perturbed ``t1`` as stored."""
    base = synthetic["base"]
    store = base / "store_t1"
    if not store.exists():
        shutil.copytree(base / "store", store)
        for table in ("forward_smile", "forward_vols"):
            for path in store.glob(f"results/points/*/{table}.parquet"):
                df = pd.read_parquet(path)
                df["t1"] = df["t1"] + 1e-10
                df.to_parquet(path, index=False)
    cfg = ViewerConfig(
        cache_root=base / "cache_empty", store_root=store, outputs_root=base / "outputs"
    )
    at = _run_page("2_forward_smile.py", cfg, monkeypatch, base)
    assert not any(i.value.startswith("No stored") for i in at.info), [i.value for i in at.info]
    assert _charts(at) == 5  # two smiles, two triples, the put-wing spread
    _assert_exports(at)
    titles = [f["layout"]["title"]["text"] for f in _fig_specs(at)]
    assert titles[:4] == [f"Forward smile {w}" for _, _, w in common.WINDOWS] + [
        f"Forward vol triple {w}" for _, _, w in common.WINDOWS
    ]
    smiles = next(d.value for d in at.dataframe if "strike_moneyness" in d.value.columns)
    assert not (smiles["t1"] == 1.0).any() and np.isclose(smiles["t1"], 1.0).any()
    assert not cfg.cache_root.exists()


def test_page2_smile_nan_stderr(synthetic: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """The LV point's 0.9 strike of the 1y→1y window stored with a NaN ``iv_stderr``: the strike
    is not a marker of the smile (before the guard it drew an invisible error bar), it is
    counted in a caption and kept as stored in the smiles table, and the put-wing spread — a
    hypot over the two extreme models — has no 0.9 row at all rather than a NaN error bar."""
    base = Path(str(synthetic["base"]))

    def nan_09(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        hit = np.isclose(out["t1"], 1.0) & np.isclose(out["strike_moneyness"], 0.9)
        assert int(hit.sum()) == 1
        out.loc[hit, "iv_stderr"] = np.nan
        return out

    store = _store_copy(base, "store_nan_iv_se", "forward_smile", nan_09, only="lv")
    cfg = _cfg_with(base, store=store)
    at = _run_page("2_forward_smile.py", cfg, monkeypatch, base)
    assert _charts(at) == 5
    specs = {f["layout"]["title"]["text"]: f for f in _fig_specs(at)}
    lv = next(tr for tr in specs["Forward smile 1y→1y"]["data"] if "LV" in str(tr["name"]))
    assert 0.9 not in list(_bin_array(lv["x"])) and len(_bin_array(lv["x"])) == 6
    other = next(tr for tr in specs["Forward smile 2y→1y"]["data"] if tr["name"] == lv["name"])
    assert len(_bin_array(other["x"])) == 7  # the other window is untouched
    captions = " ".join(c.value for c in at.caption)
    assert "1 stored strike(s) with a NaN vol or stderr" in captions
    smiles = next(d.value for d in at.dataframe if "strike_moneyness" in d.value.columns)
    assert int(smiles["iv_stderr"].isna().sum()) == 1  # kept as stored
    wing = next(d.value for d in at.dataframe if "spread_z" in d.value.columns)
    assert 0.9 not in [round(float(k), 6) for k in wing["strike_moneyness"]]
    assert bool(np.isfinite(wing["spread_stderr"]).all())
    assert "left out of the spread" in captions
    _assert_exports(at)


def test_page2_without_iv_stderr_column(
    synthetic: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store written without ``iv_stderr`` at all: one ``st.info`` per smile window and one
    for the put wing names the column and the precompute command (a KeyError mid-render before
    the guard), no smile or spread figure is drawn, the triple still renders and the smiles
    table is still exported."""
    base = Path(str(synthetic["base"]))
    store = _store_copy(
        base, "store_no_iv_se", "forward_smile", lambda df: df.drop(columns=["iv_stderr"])
    )
    cfg = _cfg_with(base, store=store)
    at = _run_page("2_forward_smile.py", cfg, monkeypatch, base)
    infos = [i.value for i in at.info]
    hits = [i for i in infos if "iv_stderr" in i]
    assert len(hits) == 3 and all("volsto-precompute" in i for i in hits), infos
    titles = [f["layout"]["title"]["text"] for f in _fig_specs(at)]
    assert not any(t.startswith(("Forward smile", "Spread of the forward smile")) for t in titles)
    assert _charts(at) == 2  # the two triples (their own stderr columns are intact)
    smiles = next(d.value for d in at.dataframe if "strike_moneyness" in d.value.columns)
    assert "iv_stderr" not in smiles.columns and len(smiles) > 0
    _assert_exports(at)


def test_page2_triple_stderr_guard(
    synthetic: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The forward vol triple: a NaN ``fwd_volswap_stderr`` costs that one bar (not the point's
    other two, and never an invisible error bar) and is counted in a caption; a table written
    without the column names it in an ``st.info`` with the precompute command and costs the two
    triple figures, not the render."""
    base = Path(str(synthetic["base"]))

    def nan_volswap(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.loc[np.isclose(out["t1"], 1.0), "fwd_volswap_stderr"] = np.nan
        return out

    cfg = _cfg_with(
        base, store=_store_copy(base, "store_nan_vs_se", "forward_vols", nan_volswap, only="lv")
    )
    at = _run_page("2_forward_smile.py", cfg, monkeypatch, base)
    specs = {f["layout"]["title"]["text"]: f for f in _fig_specs(at)}
    lv = next(tr for tr in specs["Forward vol triple 1y→1y"]["data"] if "LV" in str(tr["name"]))
    assert list(lv["x"]) == ["fwd ATM vol", "fwd VS"]  # the vol swap bar is gone
    assert all(math.isfinite(v) for v in _bin_array(lv["error_y"]["array"]))
    captions = " ".join(c.value for c in at.caption)
    assert "1 stored value(s) of the triple" in captions
    _assert_exports(at)

    cfg = _cfg_with(
        base,
        store=_store_copy(
            base,
            "store_no_vs_se",
            "forward_vols",
            lambda df: df.drop(columns=["fwd_volswap_stderr"]),
        ),
    )
    at = _run_page("2_forward_smile.py", cfg, monkeypatch, base)
    infos = [i.value for i in at.info]
    hits = [i for i in infos if "fwd_volswap_stderr" in i]
    assert len(hits) == 2 and all("volsto-precompute" in i for i in hits), infos
    titles = [f["layout"]["title"]["text"] for f in _fig_specs(at)]
    assert not any(t.startswith("Forward vol triple") for t in titles)
    triple = next(d.value for d in at.dataframe if "fwd_volswap" in d.value.columns)
    assert "fwd_volswap_stderr" not in triple.columns and len(triple) > 0
    _assert_exports(at)


# --------------------------------------------------------------------------------------------
# page 3
# --------------------------------------------------------------------------------------------


def test_page3_renders(synthetic: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """Fits table, two term structures, skew / curvature vs axis, the ratio figure with the
    band and the M7 ratio section; then the snapshot surface (marking mode)."""
    cfg = synthetic["cfg_empty"]
    at = _run_page("3_forward_skew.py", cfg, monkeypatch, synthetic["base"])
    assert [t.value for t in at.title] == ["Forward skew"]
    captions = " ".join(c.value for c in at.caption)
    assert "Method: weighted least squares" in captions
    assert f"chi2/dof > {common.QUADRATIC_MISFIT_CHI2:g} are flagged misfit" in captions
    assert "1.3-1.5 band" in captions
    assert _charts(at) == 6  # 2 term structures, skew + curvature vs axis, ratio, M7 ratio
    assert len(at.dataframe) == 3
    _assert_exports(at)
    labels = _downloads(at)
    assert any(lab == "Excel: forward skew fits" for lab in labels)
    assert any(lab == "Excel: m7 fwd spot ratio" for lab in labels)
    # one naming of the windows on the whole page: the term-structure x labels are the
    # canonical WINDOWS labels, the same strings as the fits table and the window selector
    window_labels = [w for _, _, w in common.WINDOWS]
    specs = _fig_specs(at)
    ts = [f for f in specs if "term structure" in f["layout"]["title"]["text"]]
    assert len(ts) == 2 and all(tr["x"] == window_labels for f in ts for tr in f["data"])
    fits_tab = next(d.value for d in at.dataframe if "chi2_dof" in d.value.columns)
    assert sorted(set(fits_tab["window"])) == sorted(window_labels)
    assert at.selectbox(key="p3_window").options == window_labels
    # the synthetic smiles are exact quadratics: no misfit, and the hover carries chi2/dof
    assert "misfit" in fits_tab.columns and not fits_tab["misfit"].any()
    assert all("chi2/dof" in tr["text"][0] for f in ts for tr in f["data"])
    assert not any("misfit" in w.value for w in at.warning)
    # the M7 section: both ratio columns carry their stderr twin → plotted with real error bars
    m7 = next(f for f in specs if f["layout"]["title"]["text"] == "M7 fits: fwd/spot ratio")
    assert len(m7["data"]) == 2 and all(tr["error_y"]["array"][0] > 0 for tr in m7["data"])
    at.selectbox(key="p3_surface").select("spx_2022-09-15").run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.selectbox(key="p3_mode").value == "marking"
    assert _charts(at) == 7  # incl. the marking ratio-vs-ssr_target figure
    _assert_exports(at)
    assert not cfg.cache_root.exists()


def test_page3_m7_ratio_without_stderr(
    synthetic: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fits CSV whose ``fwd/spot@2y-into-1y`` has no stderr twin and whose first
    ``fwd/spot@1y-into-1y`` row is NaN: the twin-less column is named in an ``st.info`` (with
    the producing stage) and not plotted, the NaN row is left out of the figure — no zero-filled
    error bar — and both stay in the table as written."""
    base = synthetic["base"]
    outputs = base / "outputs_no_se"
    if not outputs.exists():
        shutil.copytree(base / "outputs", outputs)
    fits_path = outputs / "m7" / "p1_marking_fits.csv"
    fits = pd.read_csv(fits_path)
    fits = pd.concat([fits, fits.assign(ssr_target=fits["ssr_target"] + 0.5)], ignore_index=True)
    fits.loc[0, "fwd/spot@1y-into-1y"] = np.nan
    fits = fits.drop(columns=["fwd/spot_se@2y-into-1y"])
    fits.to_csv(fits_path, index=False)
    cfg = ViewerConfig(
        cache_root=base / "cache_empty", store_root=base / "store", outputs_root=outputs
    )
    at = _run_page("3_forward_skew.py", cfg, monkeypatch, base)
    infos = [i.value for i in at.info]
    hit = [i for i in infos if i.startswith("fwd/spot@2y-into-1y is in ")]
    assert len(hit) == 1 and "scripts/m7_p1_marking.py stage 3" in hit[0], infos
    assert "fwd/spot_se@2y-into-1y" in hit[0]
    m7 = next(
        f for f in _fig_specs(at) if f["layout"]["title"]["text"] == "M7 fits: fwd/spot ratio"
    )
    assert len(m7["data"]) == 1  # the 1y-into-1y column only
    tr = m7["data"][0]
    assert tr["y"] == [1.41] and tr["error_y"]["array"] == [0.05]  # the NaN row is not drawn
    assert tr["x"] == [1.5]
    captions = " ".join(c.value for c in at.caption)
    assert "1 value(s) with a NaN ratio or stderr" in captions
    tab = next(d.value for d in at.dataframe if "fwd/spot@2y-into-1y" in d.value.columns)
    assert "fwd/spot@2y-into-1y_stderr" not in tab.columns
    assert len(tab) == 2 and math.isnan(float(tab["fwd/spot@1y-into-1y"].iloc[0]))
    _assert_exports(at)
    assert not cfg.cache_root.exists()


def test_page3_m7_ratio_missing_key_column(
    synthetic: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fits CSV without ``skew_eps`` (one of the columns keying the M7 ratio figure): the
    section prints an ``st.info`` naming the missing column and the producing script, draws no
    M7 ratio figure and still shows the ratio table — no KeyError."""
    base = synthetic["base"]
    outputs = base / "outputs_no_eps"
    if not outputs.exists():
        shutil.copytree(base / "outputs", outputs)
        fits_path = outputs / "m7" / "p1_marking_fits.csv"
        pd.read_csv(fits_path).drop(columns=["skew_eps"]).to_csv(fits_path, index=False)
    cfg = ViewerConfig(
        cache_root=base / "cache_empty", store_root=base / "store", outputs_root=outputs
    )
    at = _run_page("3_forward_skew.py", cfg, monkeypatch, base)
    infos = [i.value for i in at.info]
    hit = [i for i in infos if "skew_eps" in i and "p1_marking_fits.csv" in i]
    assert len(hit) == 1 and "scripts/m7_p1_marking.py" in hit[0], infos
    titles = [f["layout"]["title"]["text"] for f in _fig_specs(at)]
    assert "M7 fits: fwd/spot ratio" not in titles and _charts(at) == 5
    assert any(lab == "Excel: m7 fwd spot ratio" for lab in _downloads(at))
    tab = next(d.value for d in at.dataframe if "fwd/spot@1y-into-1y" in d.value.columns)
    assert "skew_eps" not in tab.columns and {"surface", "ssr_target"} <= set(tab.columns)
    _assert_exports(at)
    assert not cfg.cache_root.exists()


def test_page3_ratio_nan_stderr(synthetic: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    """The LV point's 1.1 strike of the ``1y→1y`` window with a NaN ``iv_stderr``: its
    forward / spot ratio stderr is a hypot over the two strikes, NaN as soon as one of them is —
    the point is not a marker of the ratio figure (a bar with an invisible error bar before the
    guard), it is counted in a caption and it is not a row of the ratio table either."""
    base = Path(str(synthetic["base"]))

    def nan_11(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        hit = np.isclose(out["t1"], 1.0) & np.isclose(out["strike_moneyness"], 1.1)
        assert int(hit.sum()) == 1
        out.loc[hit, "iv_stderr"] = np.nan
        return out

    cfg = _cfg_with(
        base, store=_store_copy(base, "store_nan_iv_11", "forward_smile", nan_11, only="lv")
    )
    at = _run_page("3_forward_skew.py", cfg, monkeypatch, base)
    ratio = next(
        f
        for f in _fig_specs(at)
        if f["layout"]["title"]["text"].startswith("Forward 1y→1y / spot 90/110")
    )
    drawn = [x for tr in ratio["data"] for x in tr["x"]]
    assert drawn and not any("LV" in str(x) for x in drawn), drawn
    assert all(math.isfinite(v) for tr in ratio["data"] for v in _bin_array(tr["error_y"]["array"]))
    captions = " ".join(c.value for c in at.caption)
    assert "have no 0.9 / 1.1 strike pair with both a vol and its stderr" in captions
    tab = next(d.value for d in at.dataframe if "ratio_fwd_to_spot" in d.value.columns)
    assert not any("LV" in str(lab) for lab in tab["label"])
    assert bool(np.isfinite(tab["ratio_fwd_to_spot_stderr"]).all())
    _assert_exports(at)


def test_page3_without_iv_stderr_column(
    synthetic: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store written without ``iv_stderr``: the weighted quadratic has no weights and the
    ratio has no stderr, so the fit section and the ratio section each name the column with the
    precompute command and draw nothing (a KeyError mid-render before the guard); the M7 ratio
    section, which reads the outputs and not the store, still renders."""
    base = Path(str(synthetic["base"]))
    store = _store_copy(
        base, "store_no_iv_se", "forward_smile", lambda df: df.drop(columns=["iv_stderr"])
    )
    cfg = _cfg_with(base, store=store)
    at = _run_page("3_forward_skew.py", cfg, monkeypatch, base)
    infos = [i.value for i in at.info]
    hits = [i for i in infos if "iv_stderr" in i]
    assert len(hits) == 2 and all("volsto-precompute" in i for i in hits), infos
    titles = [f["layout"]["title"]["text"] for f in _fig_specs(at)]
    assert titles == ["M7 fits: fwd/spot ratio"]
    _assert_exports(at)


# --------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------


def test_quadratic_fit() -> None:
    """Exact quadratic recovered; covariance stderr of the slope for equal weights and a
    symmetric design equals se / sqrt(Σx²); fewer than three strikes → None."""
    x = np.array([-0.2, -0.1, 0.0, 0.1, 0.2])
    y = 0.2 - 0.3 * x + 0.8 * x * x
    se = np.full(5, 0.001)
    fit = common.quadratic_fit(x.tolist(), y.tolist(), se.tolist())
    assert fit is not None
    assert fit.level == pytest.approx(0.2, abs=1e-12)
    assert fit.skew == pytest.approx(-0.3, abs=1e-12)
    assert fit.curvature == pytest.approx(1.6, abs=1e-12)
    assert fit.skew_stderr == pytest.approx(0.001 / math.sqrt(float(np.sum(x * x))), rel=1e-9)
    assert fit.n == 5 and fit.chi2_dof == pytest.approx(0.0, abs=1e-12)
    assert common.quadratic_fit(x[:2].tolist(), y[:2].tolist(), se[:2].tolist()) is None
    assert common.quadratic_fit(x.tolist(), y.tolist(), [0.0] * 5) is None
    d = fit.as_dict()
    assert {k for k in d if k.endswith("_stderr")} == {
        "fwd_atm_level_stderr",
        "fwd_skew_stderr",
        "fwd_curvature_stderr",
    }
    assert d["misfit"] is False


def test_quadratic_misfit_flag() -> None:
    """A deliberately non-quadratic smile — a kinked put wing (a 3 vp jump between 0.9 and 0.8
    on an otherwise flat smile) with a 0.05 vp stderr — has chi2/dof far above
    QUADRATIC_MISFIT_CHI2 and is flagged, through :class:`SmileFit` and through
    :func:`fit_smiles`; the exact quadratic of the same strikes is not; with exactly three
    strikes (zero degrees of freedom) chi2/dof is NaN and the flag is off."""
    ks = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    x = np.log(ks)
    se = 0.0005
    kinked = np.array([0.26, 0.23, 0.23, 0.23, 0.23])
    fit = common.quadratic_fit(x.tolist(), kinked.tolist(), [se] * 5)
    assert fit is not None
    assert fit.chi2_dof > 10 * common.QUADRATIC_MISFIT_CHI2 and fit.misfit
    smooth = 0.2 - 0.3 * x + 0.8 * x * x
    ok = common.quadratic_fit(x.tolist(), smooth.tolist(), [se] * 5)
    assert ok is not None and ok.chi2_dof < 1e-9 and not ok.misfit
    three = common.quadratic_fit(x[:3].tolist(), kinked[:3].tolist(), [se] * 3)
    assert three is not None and math.isnan(three.chi2_dof) and not three.misfit
    long = _long({"kinked": kinked, "smooth": smooth}, ks, se)
    fits, unfitted, absent = common.fit_smiles(long)
    assert unfitted == 0 and absent == []
    fits = fits.set_index("label")
    assert bool(fits.loc["kinked", "misfit"]) and not bool(fits.loc["smooth", "misfit"])
    assert fits["misfit"].dtype == bool and list(fits["window"]) == ["1y→1y", "1y→1y"]
    assert api.columns_without_stderr(fits.drop(columns=["t1", "t2"])) == ["n_strikes", "chi2_dof"]


def _long(vols: dict[str, np.ndarray], ks: np.ndarray, se: float) -> pd.DataFrame:
    rows = []
    for lab, v in vols.items():
        for k, iv in zip(ks, v):
            rows.append(
                {
                    "point": lab,
                    "label": lab,
                    "mode": "one_factor",
                    "t1": 1.0,
                    "t2": 2.0,
                    "strike_moneyness": float(k),
                    "log_moneyness": math.log(float(k)),
                    "iv": float(iv),
                    "iv_stderr": se,
                    "price": 1.0,
                    "price_stderr": 0.01,
                }
            )
    return pd.DataFrame(rows)


def test_common_helpers() -> None:
    """wing_spread matches put_wing_table (spread, spread_z); forward_spot_skew_ratio follows
    the M7 arithmetic."""
    ks = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    vols = {
        "a": np.array([0.30, 0.25, 0.20, 0.18, 0.17]),
        "b": np.array([0.30, 0.25, 0.21, 0.20, 0.20]),
    }
    se = 0.001
    long = _long(vols, ks, se)
    tab, dropped, absent = common.wing_spread(long, 1.0, 2.0)
    assert dropped == 0 and absent == []
    smiles = {
        lab: ForwardSmile(
            1.0, 2.0, 1.0, ks, v, np.full(5, se), np.ones(5), np.ones(5), np.ones(5, int), 1
        )
        for lab, v in vols.items()
    }
    ref = put_wing_table(smiles)
    np.testing.assert_allclose(tab["spread"], ref["spread"])
    np.testing.assert_allclose(tab["spread_z"], ref["spread_z"])
    np.testing.assert_allclose(tab["spread_stderr"], math.hypot(se, se))
    assert list(tab["put_wing"]) == [True, True, False, False, False]
    assert tab["spread"].iloc[:2].max() == 0.0 and tab["spread"].iloc[2] == pytest.approx(0.01)
    assert common.wing_spread(long[long["label"] == "a"], 1.0, 2.0)[0].empty
    ratio, dropped, absent = common.forward_spot_skew_ratio(long, 1.0, 2.0, 0.05)
    assert dropped == 0 and absent == []
    r = ratio.set_index("label").loc["a"]
    assert r["fwd_skew_90_110"] == pytest.approx(0.07)
    assert r["fwd_skew_90_110_stderr"] == pytest.approx(math.hypot(se, se))
    assert r["ratio_fwd_to_spot"] == pytest.approx(1.4)
    assert r["ratio_fwd_to_spot_stderr"] == pytest.approx(math.hypot(se, se) / 0.05)
    fits, _, _ = common.fit_smiles(long)
    assert len(fits) == 2 and api.columns_without_stderr(fits.drop(columns=["t1", "t2"])) == [
        "n_strikes",
        "chi2_dof",
    ]


def test_fits_on_synthetic_store(synthetic: dict[str, Any]) -> None:
    """Through the store path: the synthetic smiles are exactly 0.215 - 0.006 ω - 0.25 x + 0.8 x²
    in log-moneyness, so every fitted skew is -0.25 and every curvature 1.6 (stderr from the
    constant 5e-4 iv stderr), and the 1y→1y forward 90/110 skew follows from the same polynomial."""
    cfg = synthetic["cfg_empty"]
    grid = api.list_grid(cfg)
    long, missing = common.smile_long(cfg, grid, list(grid["id"]))
    assert not missing and long["label"].nunique() == 3
    fits, unfitted, absent = common.fit_smiles(long)
    assert unfitted == 0 and absent == []
    assert len(fits) == 6  # 3 points x 2 windows
    np.testing.assert_allclose(fits["fwd_skew"], -0.25, atol=1e-9)
    np.testing.assert_allclose(fits["fwd_curvature"], 1.6, atol=1e-9)
    assert (fits["fwd_skew_stderr"] > 0).all() and (fits["chi2_dof"] < 1e-12).all()
    x9, x11 = math.log(0.9), math.log(1.1)
    fs = -0.25 * (x9 - x11) + 0.8 * (x9 * x9 - x11 * x11)
    ratios, _, _ = common.forward_spot_skew_ratio(long, 1.0, 2.0, 0.05)
    np.testing.assert_allclose(ratios["fwd_skew_90_110"], fs, atol=1e-9)
    np.testing.assert_allclose(ratios["ratio_fwd_to_spot"], fs / 0.05, atol=1e-9)
    np.testing.assert_allclose(ratios["ratio_fwd_to_spot_stderr"], math.hypot(5e-4, 5e-4) / 0.05)
    assert not cfg.cache_root.exists()


def test_spot_skew_convention(synthetic: dict[str, Any]) -> None:
    """The page's spot 90/110 skew equals fit_2f.spot_skew_90_110 on the placeholder."""
    from volsto.calibration.fit_2f import spot_skew_90_110

    rec = api.get_surface(synthetic["cfg_empty"], "placeholder")
    ours = common.spot_skew_90_110(rec, 1.0)
    assert ours == pytest.approx(spot_skew_90_110(rec.surface, 1.0), rel=1e-12)
    assert ours > 0


def test_window_label() -> None:
    """The canonical labels for the two stored windows (float-tolerant), the ``start → tenor``
    fallback for any other window (labelled, never dropped)."""
    assert common.window_label(1.0, 2.0) == "1y→1y"
    assert common.window_label(2.0, 3.0) == "2y→1y"
    assert common.window_label(1.0 + 1e-12, 2.0 - 1e-12) == "1y→1y"
    assert common.window_label(0.5, 1.5) == "0.5y→1y"
    assert common.window_label(3.0, 5.0) == "3y→2y"
    assert [w for _, _, w in common.WINDOWS] == ["1y→1y", "2y→1y"]


def test_store_stamp(tmp_path: Path) -> None:
    """The cache-invalidation stamp is the newest mtime over the merged manifest, the per-point
    manifests and the cache root (mtimes set explicitly with ``os.utime`` — no wall clock):
    a point written by a running shard, or a cache entry rsync'd in, moves it; an empty store
    stamps 0.0."""
    store, cache = tmp_path / "store", tmp_path / "cache"
    cfg = ViewerConfig(cache_root=cache, store_root=store, outputs_root=tmp_path / "out")
    assert common.store_stamp(cfg) == 0.0
    results = store / "results"
    results.mkdir(parents=True)
    merged = results / "manifest.json"
    merged.write_text("{}")
    os.utime(merged, (1_000.0, 1_000.0))
    assert common.store_stamp(cfg) == 1_000.0
    # a per-point manifest newer than the merged one (a shard still running / a crashed run)
    pt = results / "points" / "p1"
    pt.mkdir(parents=True)
    (pt / "manifest.json").write_text("{}")
    os.utime(pt / "manifest.json", (2_000.0, 2_000.0))
    os.utime(results / "points", (1_500.0, 1_500.0))  # the points dir itself does not count
    assert common.store_stamp(cfg) == 2_000.0
    # the cache root (an entry directory added flips has_leverage inside the cached list_grid)
    cache.mkdir()
    os.utime(cache, (3_000.0, 3_000.0))
    assert common.store_stamp(cfg) == 3_000.0
    # an older per-point manifest changes nothing; the max is what counts
    p2 = results / "points" / "p2"
    p2.mkdir()
    (p2 / "manifest.json").write_text("{}")
    os.utime(p2 / "manifest.json", (500.0, 500.0))
    assert common.store_stamp(cfg) == 3_000.0
    # a store without a merged manifest at all (rsync'd points only) still stamps its points
    merged.unlink()
    os.utime(cache, (100.0, 100.0))
    assert common.store_stamp(cfg) == 2_000.0


def test_pages_never_calibrate() -> None:
    """Pages 1–3 and _common import no calibrating / simulating symbol (AST): none of
    :data:`FORBIDDEN` as a substring of any imported module or name, and no ``from … import``
    whose module lies in :data:`FORBIDDEN_PACKAGES`."""
    for name in (*PAGE_FILES, "_common.py"):
        tree = ast.parse((PAGES_DIR / name).read_text())
        names: set[str] = set()
        modules: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                modules.add(str(node.module))
                names.add(str(node.module))
                names.update(a.name for a in node.names)
            elif isinstance(node, ast.Import):
                modules.update(a.name for a in node.names)
                names.update(a.name for a in node.names)
        bad = [n for n in names if any(f in n for f in FORBIDDEN)]
        assert not bad, (name, bad)
        bad_mod = [
            m for m in modules if any(m == p or m.startswith(p + ".") for p in FORBIDDEN_PACKAGES)
        ]
        assert not bad_mod, (name, bad_mod)
        assert modules, name  # the walk saw the imports
