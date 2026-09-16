"""M9 Part 2 — pages 4 (smile dynamics), 5 (model risk) and 6 (product grid) rendered headless
with ``streamlit.testing.v1.AppTest`` against the synthetic store of ``tests/_synthetic_store.py``
(extended here with three more placeholder 1F points so the product-grid heatmap has a 2 × 2
``ν × ρ`` face), the pure page helpers (the model-risk table, the heatmap pivot and its missing
cells, the risk sections under both namings of the regime rows, the term-sheet matching and the
product constructors), the empty-store path (the precompute command is printed, nothing
computed), a partial heatmap face on a store copy (the missing cells are named with the
precompute command), the real RiskReport naming on a store copy (``delta[<regime>]`` /
``gamma[<regime>]`` in two sections, built through the precompute's row writer, no computation)
and the read count of the risk section (one cached read per store state).

Six further store copies (:func:`_edit_tables` / :func:`_edit_products`) close the stderr
contract on the three pages' figures, each with both triggers: a NaN stderr beside a value — the
SSR pillar, the Var(V) term, the model price, the heatmap cell and the risk row are left out of
the figure, counted in a caption and kept as written in the exported table, the value heatmap and
the stderr heatmap blank the same cell, and the LV line beside the face says so instead of
printing ``nan ± nan`` — and the stderr column gone from the stored table altogether: an
``st.info`` names the column and the producing command, the figure goes and the render does not
(a ``KeyError`` mid-render before the guard).  A NaN stderr on the *LV* point is its own case:
``sqrt(se² + se_LV²)`` is then NaN for every LSV row, so page 5 drops the whole difference figure
rather than drawing bars without error bars.

No test calibrates or simulates: the leverage cache is pointed at an empty directory that must
stay absent.  The one slow test (``test_model_risk_matches_m4_baselines``) reads the **real**
store and cache of the repository and compares the model-risk page's numbers for the cached
8·10⁵ placeholder models with ``tests/test_m4_regression.py::PLACEHOLDER_BASELINES``; it skips
unless those points are already in the store — it never runs the precompute.  Wall clocks are
printed (``-s``), never asserted.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import math
import shutil
import time
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd
import pytest
from _synthetic_store import MARKING_KEY, make_synthetic_outputs, make_synthetic_store

from volsto.market.curves import DiscountCurve
from volsto.products.base import daily_schedule
from volsto.viewers import api
from volsto.viewers.config import ViewerConfig
from volsto.viewers.pages import PAGES, page_paths
from volsto.viewers.pages._common import (
    MODE_LABELS,
    difference_stderr,
    model_risk_table,
    products_on_surface,
)
from volsto.viewers.store import PointResult, ResultsStore, StoreReader

ROOT = Path(__file__).resolve().parents[1]
MY_PAGES = ("4_smile_dynamics.py", "5_model_risk.py", "6_product_grid.py")
#: Extra placeholder 1F points written by the fixture (ν, ρ, κ): with the synthetic (0.5, −0.7,
#: 1.5) point they form the 2 × 2 ν × ρ face at κ = 1.5 of the heatmap.
EXTRA_ONE_FACTOR: tuple[tuple[float, float, float], ...] = (
    (0.5, -0.5, 1.5),
    (1.0, -0.7, 1.5),
    (1.0, -0.5, 1.5),
)
#: Store labels of the cached placeholder models ↔ the M4 baseline model names.
BASELINE_LABELS: dict[str, str] = {
    "placeholder LV (ω=0)": "LV (ω=0)",
    "placeholder 1F nu=0.5 rho=-0.7 kappa=1.5": "1F ω=1",
    "placeholder 1F nu=1 rho=-0.7 kappa=1.5": "1F ω=2",
    "placeholder 1F nu=1.5 rho=-0.7 kappa=1.5": "1F ω=3",
    "placeholder 2F Table 8.2": "2F Table 8.2",
}
BASELINE_N_PARTICLES, BASELINE_N_PATHS, BASELINE_SEED = 800_000, 400_000, 2024


def _add_extra_points(root: Path, base_id: str) -> list[str]:
    """Copy the synthetic 1F point's tables to :data:`EXTRA_ONE_FACTOR` points (values scaled
    by ν so the heatmap face is not flat); test infrastructure, not a page (pages never read
    parquet)."""
    reader = StoreReader(root)
    store = ResultsStore(root)
    row = reader.point(base_id)
    ids = []
    for nu, rho, kappa in EXTRA_ONE_FACTOR:
        pid = f"synthetic:1f:nu{nu:g}:rho{rho:g}:kappa{kappa:g}"
        r = dict(row)
        r.update(
            {
                "label": f"placeholder 1F nu={nu:g} rho={rho:g} kappa={kappa:g}",
                "axis_nu": nu,
                "axis_rho": rho,
                "axis_kappa": kappa,
                "nu": nu,
                "rho_SX1": rho,
                "k1": kappa,
                "k2": kappa,
                "cache_key": "",
            }
        )
        r.pop("point_id", None)
        tables: dict[str, pd.DataFrame] = {}
        for name in ("products", "forward_smile", "forward_vols", "ssr", "varv", "risk"):
            df = getattr(reader, name)(base_id).drop(columns=["point_id"])
            if name == "products":
                df = df.assign(value=df["value"] * (1.0 + 0.1 * nu - 0.05 * rho))
            tables[name] = df
        store.write_point(PointResult(pid, r, tables, {"label": r["label"], "mode": "one_factor"}))
        ids.append(pid)
    return ids


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    base = tmp_path_factory.mktemp("pages_b")
    t0 = time.perf_counter()
    info = make_synthetic_store(base / "store")
    make_synthetic_outputs(base / "outputs")
    ids = [str(i) for i in info["point_ids"]]
    one_factor = next(i for i in ids if i != MARKING_KEY and not i.startswith("lv:"))
    extra = _add_extra_points(base / "store", one_factor)
    ResultsStore(base / "store").refresh_manifest()  # the cached grid stamp sees the new points
    print(f"\nsynthetic store (+{len(extra)} extra 1F points) in {time.perf_counter() - t0:.2f} s")
    cfg = ViewerConfig(
        cache_root=base / "cache_empty", store_root=base / "store", outputs_root=base / "outputs"
    )
    return {"cfg": cfg, "base": base, "point_ids": ids + extra, "one_factor": one_factor}


def _cfg(synthetic: dict[str, object]) -> ViewerConfig:
    cfg = synthetic["cfg"]
    assert isinstance(cfg, ViewerConfig)
    return cfg


def _load_page(name: str) -> ModuleType:
    """Import a page file by path (its name starts with a digit); the guard renders nothing."""
    path = next(p for p in page_paths() if p.name == name)
    spec = importlib.util.spec_from_file_location(f"page_{name[0]}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _set_env(cfg: ViewerConfig, base: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var, value in cfg.to_env().items():
        monkeypatch.setenv(var, value)
    monkeypatch.setenv("VOLSTO_VIEWER_CONFIG", str(base / "no_viewer.yaml"))


def _store_copy(synthetic: dict[str, object], dest: Path) -> ViewerConfig:
    """A private copy of the module fixture's store and outputs under ``dest`` (a test mutates
    it; the fixture stays intact), with the cache root still an absent directory."""
    src = Path(str(synthetic["base"]))
    shutil.copytree(src / "store", dest / "store")
    shutil.copytree(src / "outputs", dest / "outputs")
    return ViewerConfig(
        cache_root=dest / "cache_empty", store_root=dest / "store", outputs_root=dest / "outputs"
    )


def _rewrite_point_tables(root: Path, point_id: str, edit: dict[str, Any]) -> None:
    """Rewrite a point of a store copy with ``edit`` (table name → replacement frame without
    ``point_id``); test infrastructure, not a page."""
    reader, store = StoreReader(root), ResultsStore(root)
    row = reader.point(point_id)
    row.pop("point_id", None)
    tables = {
        name: getattr(reader, name)(point_id).drop(columns=["point_id"])
        for name in ("products", "forward_smile", "forward_vols", "ssr", "varv", "risk")
    }
    tables.update(edit)
    manifest = json.loads((store.point_dir(point_id) / "manifest.json").read_text())
    manifest.pop("point_id", None)
    store.write_point(PointResult(point_id, row, tables, manifest))


def _render(page: str, timeout: int = 180) -> Any:
    from streamlit.testing.v1 import AppTest

    path = next(p for p in page_paths() if p.name == page)
    return AppTest.from_file(str(path), default_timeout=timeout).run()


# --------------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("page", MY_PAGES)
def test_page_renders(
    synthetic: dict[str, object], page: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each page renders headless against the synthetic store: no exception, the title, the
    provenance line, figures and tables present, and the page-specific notices."""
    from streamlit.testing.v1 import AppTest

    cfg = _cfg(synthetic)
    _set_env(cfg, Path(str(synthetic["base"])), monkeypatch)
    path = next(p for p in page_paths() if p.name == page)
    t0 = time.perf_counter()
    at = AppTest.from_file(str(path), default_timeout=180).run()
    print(f"\n{page}: rendered in {time.perf_counter() - t0:.2f} s")
    assert not at.exception, [e.value for e in at.exception]
    assert [t.value for t in at.title] == [dict((f, t) for f, t, _ in PAGES)[page]]
    captions = " ".join(c.value for c in at.caption)
    assert "6 points" in captions and "code tag synthetic" in captions
    infos = " ".join(i.value for i in at.info)
    downloads = [b.label for b in at.download_button]
    assert any(lbl.startswith("Excel:") for lbl in downloads), downloads
    assert len(at.get("plotly_chart")) >= 1
    if page == "4_smile_dynamics.py":
        assert "conditional smile" in infos.lower() and "not precomputed" in infos.lower()
        assert "--analytics conditional" in infos
    if page == "5_model_risk.py":
        assert "LSV minus LV" in captions
        assert len(at.get("plotly_chart")) >= 2  # prices and the LSV − LV chart
    if page == "6_product_grid.py":
        assert "needs two grid axes" not in infos  # the 2 × 2 ν × ρ face renders
        assert not at.warning, [w.value for w in at.warning]  # the face is complete: no hole
        assert any("not precomputed at this tier" in i for i in infos.split("`"))
        # both regime sections render as bar charts with a download (F1: the gamma rows are found
        # whether the store names them 'gamma[<regime>]' under section 'gamma' — the RiskReport
        # naming — or 'gamma [<regime>]' under section 'delta' — the earlier synthetic naming)
        for kind in ("delta", "gamma"):
            assert any(f": {kind}_" in lbl for lbl in downloads), (kind, downloads)
            assert f"{kind} regimes: not precomputed" not in infos
        # the editor's default term sheet is the 3y headline autocall: its repr is shown and its
        # stored price per model is served from the store (no pricing run)
        assert any(c.value.startswith("Autocall") for c in at.code), [c.value for c in at.code]
        assert any(lbl == "Excel: cache_price_autocall 3y" for lbl in downloads), downloads
        assert any("no pricing run" in s.value for s in at.success)
    assert not cfg.cache_root.exists()


@pytest.mark.parametrize("page", MY_PAGES)
def test_page_on_empty_store(page: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An empty store: the precompute command is printed, nothing raised, no cache created."""
    from streamlit.testing.v1 import AppTest

    cfg = ViewerConfig(
        cache_root=tmp_path / "c", store_root=tmp_path / "s", outputs_root=tmp_path / "o"
    )
    _set_env(cfg, tmp_path, monkeypatch)
    path = next(p for p in page_paths() if p.name == page)
    at = AppTest.from_file(str(path), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert any(c.value.startswith("volsto-precompute --grid") for c in at.code)
    assert not (tmp_path / "c").exists()


# --------------------------------------------------------------------------------------------
# page helpers
# --------------------------------------------------------------------------------------------


def test_model_risk_table_equals_store(synthetic: dict[str, object]) -> None:
    """The model-risk page's numbers are the store's ``products`` rows unchanged, and the
    ``LSV − LV`` column is the exact difference with stderr ``sqrt(se² + se_LV²)``."""
    cfg = _cfg(synthetic)
    grid = api.list_grid(cfg)
    prods = products_on_surface(cfg, grid, "placeholder")
    assert set(prods["id"]) == set(grid.loc[grid["surface"] == "placeholder", "id"])
    lv_id = str(grid.loc[grid["mode"] == "lv", "id"].iloc[0])
    lv = api.get_point(cfg, lv_id).products.set_index("key")
    pairs = prods[["product", "quantity"]].drop_duplicates().itertuples(index=False)
    n = 0
    for product, quantity in pairs:
        for sort_by in ("grid", "price", "label"):
            table, unplottable, absent = model_risk_table(
                prods, str(product), str(quantity), sort_by=sort_by
            )
            assert unplottable == 0 and absent == []
            assert len(table) == 5  # LV + 4 one-factor points on the placeholder
            if sort_by == "price":
                assert table["value"].is_monotonic_increasing
            for _, r in table.iterrows():
                stored = api.get_point(cfg, str(r["id"])).products
                cell = stored[(stored["product"] == product) & (stored["quantity"] == quantity)]
                assert len(cell) == 1
                assert float(r["value"]) == float(cell["value"].iloc[0])
                assert float(r["value_stderr"]) == float(cell["value_stderr"].iloc[0])
                key = str(cell["key"].iloc[0])
                assert r["minus_lv"] == float(cell["value"].iloc[0]) - float(lv["value"][key])
                assert r["minus_lv_stderr"] == difference_stderr(
                    float(cell["value_stderr"].iloc[0]), float(lv["value_stderr"][key])
                )
                n += 1
    print(f"\nmodel-risk table checked on {n} cells")
    lv_rows, _, _ = model_risk_table(prods, "cliquet 1y", "price")
    assert (lv_rows.loc[lv_rows["family"] == MODE_LABELS["lv"], "minus_lv"] == 0.0).all()
    # the snapshot has no LV point in the synthetic store: the difference column is empty
    snap = products_on_surface(cfg, grid, "spx_2022-09-15")
    t, _, _ = model_risk_table(snap, "cliquet 1y", "price")
    assert len(t) == 1 and math.isnan(float(t["minus_lv"].iloc[0]))
    assert not cfg.cache_root.exists()


def test_heatmap_pivot(synthetic: dict[str, object]) -> None:
    """The product-grid pivot reproduces the store's values on the 2 × 2 ν × ρ face and refuses
    an ambiguous cell."""
    page6 = _load_page("6_product_grid.py")
    cfg = _cfg(synthetic)
    prods = products_on_surface(cfg, api.list_grid(cfg), "placeholder")
    v, s = page6.heatmap_pivot(prods, "cliquet 1y", "price", "axis_nu", "axis_rho")
    assert v.shape == (2, 2) and list(v.columns) == [0.5, 1.0] and list(v.index) == [-0.7, -0.5]
    for nu in (0.5, 1.0):
        for rho in (-0.7, -0.5):
            cell = prods[
                (prods["product"] == "cliquet 1y")
                & (prods["quantity"] == "price")
                & (prods["axis_nu"] == nu)
                & (prods["axis_rho"] == rho)
            ]
            assert float(v.loc[rho, nu]) == float(cell["value"].iloc[0])
            assert float(s.loc[rho, nu]) == float(cell["value_stderr"].iloc[0])
    dup = pd.concat([prods, prods], ignore_index=True)
    with pytest.raises(ValueError, match="share"):
        page6.heatmap_pivot(dup, "cliquet 1y", "price", "axis_nu", "axis_rho")
    fig = page6.heatmap_figure(v, s, "t", "axis_nu", "axis_rho")
    assert "±" in str(fig.data[0].text[0][0])
    # a hole in the face (no point at ν = 1, ρ = −0.5): named as an (x, y) pair, the rest intact
    assert page6.missing_cells(v) == []
    hole = ~((prods["axis_nu"] == 1.0) & (prods["axis_rho"] == -0.5))
    v2, s2 = page6.heatmap_pivot(prods[hole], "cliquet 1y", "price", "axis_nu", "axis_rho")
    assert v2.shape == (2, 2) and page6.missing_cells(v2) == [(1.0, -0.5)]
    assert math.isnan(float(v2.loc[-0.5, 1.0])) and math.isnan(float(s2.loc[-0.5, 1.0]))
    assert float(v2.loc[-0.7, 1.0]) == float(v.loc[-0.7, 1.0])


def test_heatmap_partial_face_names_missing_cells(
    synthetic: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A face with a hole (a store copy without the ν = 1, ρ = −0.5 point, as after a ``--limit``
    run or while a shard is running): the page plots what is stored, lists the missing cell with
    the precompute command and keeps it out of the exported table."""
    cfg = _store_copy(synthetic, tmp_path)
    store = ResultsStore(cfg.store_root)
    gone = "synthetic:1f:nu1:rho-0.5:kappa1.5"
    shutil.rmtree(store.point_dir(gone))
    store.refresh_manifest()
    _set_env(cfg, tmp_path, monkeypatch)
    at = _render("6_product_grid.py")
    assert not at.exception, [e.value for e in at.exception]
    warnings = [w.value for w in at.warning]
    hole = [w for w in warnings if "heatmap cells" in w]
    assert len(hole) == 1, warnings
    # the page's default product is the first stored one (the 1y→2y forward ATM vol)
    product, quantity = "fwd ATM vol 1y→2y", "vol"
    assert f"{product!r} {quantity}" in hole[0] and "= (1, -0.5) —" in hole[0], hole[0]
    assert "(nu (omega = 2 nu), rho)" in hole[0] and "absent from the exported table" in hole[0]
    assert any(c.value == api.precompute_command(cfg) for c in at.code), [c.value for c in at.code]
    infos = " ".join(i.value for i in at.info)
    assert "needs two grid axes" not in infos  # the face still has two axes with two values
    downloads = [b.label for b in at.download_button]
    assert any(f": heatmap_{product}_{quantity}" in lbl for lbl in downloads), downloads
    assert any(lbl == f"Excel: heatmap_placeholder_{product}_{quantity}" for lbl in downloads)
    captions = " ".join(c.value for c in at.caption)
    assert "1 missing cell(s) of the face are not rows of this table" in captions
    # the exported rows are the store's remaining points only (3 cells + the LV row off the axes)
    prods = products_on_surface(cfg, api.list_grid(cfg), "placeholder")
    sel = prods[(prods["product"] == product) & (prods["quantity"] == quantity)]
    assert gone not in set(sel["id"]) and len(sel[sel["axis_nu"].notna()]) == 3
    page6 = _load_page("6_product_grid.py")
    v, _ = page6.heatmap_pivot(sel, product, quantity, "axis_nu", "axis_rho")
    assert page6.missing_cells(v) == [(1.0, -0.5)]
    assert not cfg.cache_root.exists()


def test_term_sheet_matching_and_constructors() -> None:
    """The editor's headline term sheets map onto the store's product names, a custom one onto
    ``None``; every class builds a Product (pure construction: repr and fixings, no pricing)."""
    page6 = _load_page("6_product_grid.py")
    m = page6.store_product_name
    assert (
        m("cliquet", {"T": 1.0, "ppy": 12, "cap": 0.02, "local_floor": None, "global_floor": 0.0})
        == "cliquet 1y"
    )
    assert (
        m("cliquet", {"T": 2.0, "ppy": 12, "cap": 0.02, "local_floor": None, "global_floor": 0.0})
        == "cliquet 2y"
    )
    assert (
        m("cliquet", {"T": 1.0, "ppy": 12, "cap": 0.03, "local_floor": None, "global_floor": 0.0})
        is None
    )
    assert (
        m("VKO put", {"strike": 1.0, "T": 1.0, "vol_ko": 0.30, "knock_in": False})
        == "VKO 12m 100% put @30%"
    )
    assert m("VKO put", {"strike": 1.0, "T": 1.0, "vol_ko": 0.30, "knock_in": True}) is None
    assert m("KO var", {"barrier": 1.1, "T": 1.0, "strike_vol": 0.0}) == "KO var 1y B=110%"
    assert m("up-var", {"barrier": 1.0, "T": 1.0, "strike_vol": 0.0}) == "up-var 1y B=100%"
    assert m("down-var", {"barrier": 1.0, "T": 1.0, "strike_vol": 0.0}) == "down-var 1y B=100%"
    assert m("autocall", {"T": 3, "coupon": 0.06, "ac": 1.0, "ki": 0.6}) == "autocall 3y"
    assert m("autocall", {"T": 5, "coupon": 0.06, "ac": 1.0, "ki": 0.6}) is None
    assert (
        m("Phoenix", {"T": 3, "coupon": 0.06, "ac": 1.0, "ki": 0.6, "cb": 0.7, "memory": True})
        == "phoenix 3y"
    )
    assert m("barrier KO/KI", {}) is None
    discount, spot = DiscountCurve.flat(0.02), 100.0
    forms: dict[str, dict[str, object]] = {
        "autocall": {"T": 3, "coupon": 0.06, "ac": 1.0, "ki": 0.6},
        "Phoenix": {"T": 3, "coupon": 0.06, "ac": 1.0, "ki": 0.6, "cb": 0.7, "memory": True},
        "cliquet": {"T": 1.0, "ppy": 12, "cap": 0.02, "local_floor": None, "global_floor": 0.0},
        "VKO put": {"strike": 1.0, "T": 1.0, "vol_ko": 0.30, "knock_in": False},
        "KO var": {"barrier": 1.1, "T": 1.0, "strike_vol": 0.0},
        "up-var": {"barrier": 1.0, "T": 1.0, "strike_vol": 0.0},
        "down-var": {"barrier": 1.0, "T": 1.0, "strike_vol": 0.0},
        "barrier KO/KI": {
            "knock": "out",
            "cp": "call",
            "strike": 1.0,
            "T": 1.0,
            "barrier": 1.2,
            "direction": "up",
            "monitoring": "continuous",
            "rebate": 0.0,
        },
    }
    daily = {T: daily_schedule(T, 252).size for T in (1.0, 3.0)}  # the t = 0 fixing included
    expect = {
        "autocall": 3,
        "Phoenix": daily[3.0],
        "cliquet": 12,
        "VKO put": daily[1.0],
        "KO var": daily[1.0],
        "up-var": daily[1.0],
        "down-var": daily[1.0],
        "barrier KO/KI": 1,
    }
    for kind, f in forms.items():
        product = page6.build_product(kind, f, spot, discount)
        ft = np.asarray(product.fixing_times)
        assert ft.size in (expect[kind], expect[kind] + 1), (kind, ft.size)  # ± t = 0
        assert math.isclose(float(ft[-1]), float(f["T"]))  # type: ignore[arg-type]
        assert repr(product)
    ki = page6.build_product(
        "barrier KO/KI",
        {**forms["barrier KO/KI"], "knock": "in", "monitoring": "discrete"},
        spot,
        discount,
    )
    assert np.asarray(ki.fixing_times).size == daily[1.0]


def _real_naming(risk: pd.DataFrame) -> pd.DataFrame:
    """The synthetic store's risk rows renamed to the RiskReport convention: ``delta[<regime>]``
    under group ``delta`` and ``gamma[<regime>]`` under group ``gamma`` (the store's ``group``
    column is what the API renames ``section``)."""
    out = risk.copy()
    regimes = out["regime"].astype(str) != ""
    kind = out["name"].str.extract(r"^(delta|gamma)")[0]
    out.loc[regimes, "group"] = kind[regimes]
    out.loc[regimes, "name"] = kind[regimes] + "[" + out.loc[regimes, "regime"].astype(str) + "]"
    out.loc[regimes, "key"] = out.loc[regimes, "group"] + ":" + out.loc[regimes, "name"]
    return out


def test_risk_sections_under_both_namings(synthetic: dict[str, object]) -> None:
    """The regime sections are keyed on the store's ``section`` column: rows written by the
    precompute's row writer (``volsto.viewers.precompute._risk_rows`` on RiskReport-shaped rows —
    read-only, no engine) land in two sections ``delta`` / ``gamma`` and both render; the
    synthetic store's earlier naming (``gamma [<regime>]`` under ``delta``) renders the same."""
    from volsto.viewers.precompute import _risk_rows

    page6 = _load_page("6_product_grid.py")
    regimes = ("model", "sticky_strike", "sticky_moneyness", "sticky_skew", "sticky_local_vol")
    rows: list[dict[str, Any]] = []
    for regime in regimes:  # what risk_report() adds: rep.add('delta', d, regime=…), ('gamma', g)
        rows.append(
            {
                "group": "delta",
                "name": f"delta[{regime}]",
                "value": 0.3,
                "stderr": 0.01,
                "unit": "per spot",
                "regime": regime,
            }
        )
        rows.append(
            {
                "group": "gamma",
                "name": f"gamma[{regime}]",
                "value": -0.01,
                "stderr": 0.001,
                "unit": "per spot²",
                "regime": regime,
            }
        )
    rows.append(
        {"group": "fwd_var", "name": "fwd-var vega [0-0.25y]", "value": 0.02, "stderr": 0.001}
    )
    rows.append({"group": "skew", "name": "skew_T [1y]", "value": -0.05, "stderr": 0.002})
    real = _risk_rows(rows, "autocall 3y", "light").rename(columns={"group": "section"})
    assert sorted(real["section"].unique()) == ["delta", "fwd_var", "gamma", "skew"]
    assert (real[real["section"] == "delta"]["name"].str.startswith("gamma")).sum() == 0
    cfg = _cfg(synthetic)
    stored = api.get_risk(cfg, str(synthetic["one_factor"]), "autocall 3y")
    # the synthetic fixture writes the RiskReport naming today: gamma rows under section 'gamma'
    assert set(stored["section"]) == {"delta", "gamma", "fwd_var", "skew"}
    # the earlier naming the helper still has to read: gamma rows under 'delta', named 'gamma […]'
    legacy = stored.copy()
    legacy["name"] = legacy["name"].str.replace("gamma[", "gamma [", regex=False)
    legacy.loc[legacy["section"] == "gamma", "section"] = "delta"
    assert set(legacy["section"]) == {"delta", "fwd_var", "skew"}
    for frame, label in (
        (real, "RiskReport naming"),
        (stored, "stored naming"),
        (legacy, "earlier synthetic naming"),
    ):
        for kind in ("delta", "gamma"):
            part = page6.risk_section_rows(frame, kind)
            assert len(part) == 5 and set(part["regime"]) == set(regimes), (label, kind)
            assert part["name"].str.startswith(kind).all(), (label, kind)
        assert len(page6.risk_section_rows(frame, "fwd_var")) >= 1
        assert page6.risk_section_rows(frame, "curvature").empty
    # the API-shaped frame keeps only the API's columns: the helper needs none beyond them
    assert {"section", "name", "regime"} <= set(api.RISK_COLUMNS)


def test_page6_renders_real_risk_naming(
    synthetic: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Page 6 against a store copy whose risk rows carry the RiskReport naming (two sections):
    delta and gamma bars both present with downloads, neither reported as not precomputed."""
    cfg = _store_copy(synthetic, tmp_path)
    reader = StoreReader(cfg.store_root)
    n = 0
    for pid in ResultsStore(cfg.store_root).point_ids():
        risk = reader.risk(pid)
        if risk.empty:
            continue
        _rewrite_point_tables(
            cfg.store_root, pid, {"risk": _real_naming(risk.drop(columns=["point_id"]))}
        )
        n += 1
    assert n >= 1
    ResultsStore(cfg.store_root).refresh_manifest()
    pid = str(synthetic["one_factor"])
    stored = api.get_risk(cfg, pid, "autocall 3y")
    assert set(stored["section"]) == {"delta", "fwd_var", "gamma", "skew"}
    assert set(stored.loc[stored["section"] == "gamma", "name"]) == {
        f"gamma[{r}]"
        for r in ("model", "sticky_strike", "sticky_moneyness", "sticky_skew", "sticky_local_vol")
    }
    _set_env(cfg, tmp_path, monkeypatch)
    at = _render("6_product_grid.py")
    assert not at.exception, [e.value for e in at.exception]
    downloads = [b.label for b in at.download_button]
    infos = " ".join(i.value for i in at.info)
    for kind in ("delta", "gamma"):
        assert any(f": {kind}_" in lbl for lbl in downloads), (kind, downloads)
        assert f"{kind} regimes: not precomputed" not in infos
    assert "curvature_T: not precomputed" in infos and "vega-T waves: not precomputed" in infos
    assert not cfg.cache_root.exists()


def test_risk_section_reads_store_once(
    synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The risk section reads the point's risk rows through one cached accessor (F4): the
    ``StoreReader.risk`` reads of a first render are bounded by one discovery read plus the
    API's two per stored product, and a rerun (widget interaction) reads the risk table not at
    all; the total ``ResultsStore.table`` calls are printed as the diagnostic."""
    cfg = _cfg(synthetic)
    _set_env(cfg, Path(str(synthetic["base"])), monkeypatch)
    calls = {"table": 0, "risk_table": 0, "reader_risk": 0}
    orig_table, orig_risk = ResultsStore.table, StoreReader.risk

    def counting_table(self: ResultsStore, name: str) -> pd.DataFrame:
        calls["table"] += 1
        calls["risk_table"] += name == "risk"
        return orig_table(self, name)

    def counting_risk(
        self: StoreReader, point_id: str | None = None, product: str | None = None
    ) -> pd.DataFrame:
        calls["reader_risk"] += 1
        return orig_risk(self, point_id, product)

    monkeypatch.setattr(ResultsStore, "table", counting_table)
    monkeypatch.setattr(StoreReader, "risk", counting_risk)
    at = _render("6_product_grid.py")
    assert not at.exception, [e.value for e in at.exception]
    first = dict(calls)
    n_products = len(set(StoreReader(cfg.store_root).risk(str(synthetic["one_factor"]))["product"]))
    calls.update(table=0, risk_table=0, reader_risk=0)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    rerun = dict(calls)
    print(f"\nstore reads: first render {first}, rerun {rerun} ({n_products} risk products)")
    assert n_products == 2
    # first render: ≤ 1 discovery read + get_risk's one risk read per product (the accessor may
    # already be warm from another test of this process, hence ≤ rather than ==)
    assert first["reader_risk"] <= 1 + n_products, first
    assert rerun["reader_risk"] == 0 and rerun["risk_table"] == 0, rerun
    assert not cfg.cache_root.exists()


# --------------------------------------------------------------------------------------------
# pathological copies: the stderr triggers pages 4-6 must refuse to plot
# --------------------------------------------------------------------------------------------


def _edit_tables(
    root: Path, table: str, edit: Callable[[pd.DataFrame], pd.DataFrame], *, only: str = ""
) -> None:
    """Apply ``edit`` to the ``table`` parquet of every point of a store copy (or of the points
    whose directory starts with ``only``) and refresh the manifest: the reviewer's triggers —
    a NaN stderr beside a value, the stderr column gone."""
    for path in sorted(root.glob(f"results/points/*/{table}.parquet")):
        if only and not path.parent.name.startswith(only):
            continue
        edit(pd.read_parquet(path)).to_parquet(path, index=False)
    ResultsStore(root).refresh_manifest()


def _nan_column(column: str, rows: int = 1) -> Callable[[pd.DataFrame], pd.DataFrame]:
    def edit(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out.loc[out.index[:rows], column] = math.nan
        return out

    return edit


def _drop_column(column: str) -> Callable[[pd.DataFrame], pd.DataFrame]:
    return lambda df: df.drop(columns=[column])


def _titles(at: Any) -> list[str]:
    return [json.loads(c.proto.spec)["layout"]["title"]["text"] for c in at.get("plotly_chart")]


def _bin_array(value: Any) -> np.ndarray:
    """A plotly array attribute of a chart spec as floats (plotly encodes numeric arrays as
    base64 binary with a shape string)."""
    if isinstance(value, dict):
        flat = np.frombuffer(base64.b64decode(value["bdata"]), dtype=np.dtype(value["dtype"]))
        return flat.reshape([int(n) for n in str(value["shape"]).split(",")])
    return np.asarray(value, dtype=float)


def test_page4_ssr_nan_stderr(
    synthetic: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored SSR pillar with a NaN ``ssr_lsv_stderr``: the pillar is not a marker of the SSR
    term structure (an invisible error bar and a '1.4 ± nan' hover before the guard), it is
    counted in a caption and kept as NaN in the exported SSR table."""
    cfg = _store_copy(synthetic, tmp_path)
    _edit_tables(cfg.store_root, "ssr", _nan_column("ssr_lsv_stderr"))
    _set_env(cfg, tmp_path, monkeypatch)
    at = _render("4_smile_dynamics.py")
    assert not at.exception, [e.value for e in at.exception]
    ssr = next(
        json.loads(c.proto.spec)
        for c in at.get("plotly_chart")
        if json.loads(c.proto.spec)["layout"]["title"]["text"].startswith("SSR term structure")
    )
    tab = next(d.value for d in at.dataframe if "ssr_lsv_stderr" in d.value.columns)
    drawn = ssr["data"][0]
    assert len(_bin_array(drawn["x"])) == len(tab) - 1  # the NaN pillar is not a marker
    assert np.isfinite(_bin_array(drawn["error_y"]["array"])).all()
    captions = " ".join(c.value for c in at.caption)
    assert "1 pillar(s) with a NaN SSR or stderr" in captions
    assert int(tab["ssr_lsv_stderr"].isna().sum()) == 1  # kept as NaN in the table
    assert not cfg.cache_root.exists()


def test_page4_without_stderr_columns(
    synthetic: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store whose ``ssr`` table has no ``ssr_lsv_stderr`` and whose ``varv`` table has no
    ``value_stderr``: each figure is replaced by an ``st.info`` naming the column and the
    precompute command (a KeyError mid-render before the guard) and both tables are still
    exported."""
    cfg = _store_copy(synthetic, tmp_path)
    _edit_tables(cfg.store_root, "ssr", _drop_column("ssr_lsv_stderr"))
    _edit_tables(cfg.store_root, "varv", _drop_column("value_stderr"))
    _set_env(cfg, tmp_path, monkeypatch)
    at = _render("4_smile_dynamics.py")
    assert not at.exception, [e.value for e in at.exception]
    infos = [i.value for i in at.info]
    assert [i for i in infos if "ssr_lsv_stderr" in i and "volsto-precompute" in i], infos
    assert [i for i in infos if "value_stderr" in i and "volsto-precompute" in i], infos
    assert not any(t.startswith(("SSR term structure", "Var(V)")) for t in _titles(at))
    labels = [b.label for b in at.download_button]
    assert any(lbl.startswith("Excel: ssr_") for lbl in labels), labels
    assert any(lbl.startswith("Excel: varv_") for lbl in labels), labels
    assert not cfg.cache_root.exists()


def test_page4_varv_nan_stderr(
    synthetic: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Monte Carlo Var(V) row with a NaN ``value_stderr``: neither the stacked bar nor the
    ``var_total`` diamond draws it, the row is counted in a caption and kept in the table."""
    cfg = _store_copy(synthetic, tmp_path)
    page4 = _load_page("4_smile_dynamics.py")
    before = api.get_point(cfg, str(synthetic["one_factor"])).varv
    fig0, _, _ = page4.varv_figure(before, "x")
    assert fig0 is not None
    n_drawn = sum(len(tr.x) for tr in fig0.data)

    def nan_first_var_sv(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        mc = out.index[(out["kind"] == "mc") & (out["term"] == "var_sv")]
        assert len(mc) >= 1
        out.loc[mc[:1], "value_stderr"] = math.nan
        return out

    _edit_tables(cfg.store_root, "varv", nan_first_var_sv)
    after = api.get_point(cfg, str(synthetic["one_factor"])).varv
    fig, dropped, absent = page4.varv_figure(after, "x")
    assert (dropped, absent) == (1, [])
    assert fig is not None and sum(len(tr.x) for tr in fig.data) == n_drawn - 1
    assert all(np.isfinite(np.asarray(tr.error_y.array, dtype=float)).all() for tr in fig.data)
    _set_env(cfg, tmp_path, monkeypatch)
    at = _render("4_smile_dynamics.py")
    assert not at.exception, [e.value for e in at.exception]
    captions = " ".join(c.value for c in at.caption)
    assert "1 Monte Carlo term(s) with a NaN value or stderr" in captions
    assert not cfg.cache_root.exists()


def test_page4_figures(synthetic: dict[str, object]) -> None:
    """The SSR figure carries the naked first-order line and the ssr_target line for the
    marking point only; the Var(V) figure stacks the three MC terms plus var_total."""
    page4 = _load_page("4_smile_dynamics.py")
    cfg = _cfg(synthetic)
    one = api.get_point(cfg, str(synthetic["one_factor"]))
    fig = page4.ssr_figure(one.ssr, one.label)
    assert len(fig.data) == 1 and fig.data[0].error_y.array is not None
    assert not fig.layout.shapes
    mk = api.get_point(cfg, MARKING_KEY)
    fig = page4.ssr_figure(mk.ssr, mk.label)
    assert [d.name for d in fig.data] == ["LSV numerical (MC)", "naked first order (exact)"]
    assert len(fig.layout.shapes) == 1  # the ssr_target line
    vf, dropped, absent = page4.varv_figure(one.varv, one.label)
    assert vf is not None and dropped == 0 and absent == []
    assert [d.name for d in vf.data] == ["var_sv", "var_leverage", "cov_cross", "var_total"]
    assert all(d.error_y.array is not None for d in vf.data)


def _edit_products(
    cfg: ViewerConfig, pid: str, edit: Callable[[pd.DataFrame], pd.DataFrame]
) -> None:
    """Rewrite one point's ``products`` table of a store copy and refresh the manifest."""
    prods = StoreReader(cfg.store_root).products(pid).drop(columns=["point_id"])
    _rewrite_point_tables(cfg.store_root, pid, {"products": edit(prods)})
    ResultsStore(cfg.store_root).refresh_manifest()


def test_page5_price_nan_stderr(
    synthetic: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One model's stored price with a NaN ``value_stderr``: it is not a bar of the price figure
    (a NaN ``error_y`` and a '0.2082 ± nan vol' hover before the guard), it is counted in a
    caption and kept as stored in the model-risk table."""
    cfg = _store_copy(synthetic, tmp_path)
    pid = str(synthetic["one_factor"])
    _edit_products(cfg, pid, _nan_column("value_stderr"))
    _set_env(cfg, tmp_path, monkeypatch)
    at = _render("5_model_risk.py")
    assert not at.exception, [e.value for e in at.exception]
    product = str(at.selectbox(key="p5_product").value)
    quantity = str(at.selectbox(key="p5_quantity").value)
    table = next(d.value for d in at.dataframe if "minus_lv" in d.value.columns)
    price = next(
        json.loads(c.proto.spec)
        for c in at.get("plotly_chart")
        if json.loads(c.proto.spec)["layout"]["title"]["text"].startswith(f"{product} — {quantity}")
    )
    drawn = [x for tr in price["data"] for x in tr["x"]]
    nan_rows = table[table["value_stderr"].isna()]
    assert len(nan_rows) == 1 and len(drawn) == len(table) - 1
    assert str(nan_rows["label"].iloc[0]) not in drawn
    assert all(np.isfinite(_bin_array(tr["error_y"]["array"])).all() for tr in price["data"])
    captions = " ".join(c.value for c in at.caption)
    assert "1 model(s) with a NaN price or stderr are left out of the bars" in captions
    assert not cfg.cache_root.exists()


def test_page5_lv_nan_stderr_drops_the_difference_figure(
    synthetic: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LV point's price with a NaN ``value_stderr``: ``sqrt(se² + se_LV²)`` is NaN for
    *every* LSV row, so the difference figure is not drawn at all (before the guard every bar
    lost its error bar and none said so) — an ``st.info`` names it, and the NaN column stays in
    the exported table."""
    cfg = _store_copy(synthetic, tmp_path)
    grid = api.list_grid(cfg)
    lv_id = str(grid.loc[grid["mode"] == "lv", "id"].iloc[0])
    _edit_products(cfg, lv_id, lambda df: df.assign(value_stderr=math.nan))
    _set_env(cfg, tmp_path, monkeypatch)
    at = _render("5_model_risk.py")
    assert not at.exception, [e.value for e in at.exception]
    infos = [i.value for i in at.info]
    hit = [i for i in infos if "LSV minus LV" in i]
    assert len(hit) == 1 and "sqrt(se² + se_LV²)" in hit[0] and "volsto-precompute" in hit[0]
    assert not any("minus LV" in t for t in _titles(at))
    table = next(d.value for d in at.dataframe if "minus_lv_stderr" in d.value.columns)
    assert bool(table["minus_lv_stderr"].isna().all()) and table["minus_lv"].notna().any()
    assert not cfg.cache_root.exists()


def test_page5_without_stderr_column(
    synthetic: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store whose ``products`` table has no ``value_stderr``: an ``st.info`` names the column
    and the precompute command (a KeyError mid-render before the guard), no price figure is
    drawn and the table is still exported with the column as NaN."""
    cfg = _store_copy(synthetic, tmp_path)
    for pid in ResultsStore(cfg.store_root).point_ids():
        if not StoreReader(cfg.store_root).products(pid).empty:
            _edit_products(cfg, pid, _drop_column("value_stderr"))
    _set_env(cfg, tmp_path, monkeypatch)
    at = _render("5_model_risk.py")
    assert not at.exception, [e.value for e in at.exception]
    infos = [i.value for i in at.info]
    assert [i for i in infos if "value_stderr" in i and "volsto-precompute" in i], infos
    assert not at.get("plotly_chart")
    table = next(d.value for d in at.dataframe if "minus_lv" in d.value.columns)
    assert bool(table["value_stderr"].isna().all()) and len(table) >= 2
    assert not cfg.cache_root.exists()


def test_page6_heatmap_nan_stderr_cell(
    synthetic: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A stored cell with a price but a NaN ``value_stderr``: the cell is blank in the value map
    *and* in the stderr map (before the guard the value map read 'value ± nan' while its
    companion rendered it blank — the two figures disagreed), it is counted in the table's
    caption and kept as a row of the exported table."""
    cfg = _store_copy(synthetic, tmp_path)
    product, quantity = "fwd ATM vol 1y→2y", "vol"

    def nan_cell(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        hit = (out["product"] == product) & (out["quantity"] == quantity)
        assert int(hit.sum()) == 1
        out.loc[hit, "value_stderr"] = math.nan
        return out

    _edit_products(cfg, "synthetic:1f:nu1:rho-0.5:kappa1.5", nan_cell)
    _set_env(cfg, tmp_path, monkeypatch)
    at = _render("6_product_grid.py")
    assert not at.exception, [e.value for e in at.exception]
    specs = {
        json.loads(c.proto.spec)["layout"]["title"]["text"]: json.loads(c.proto.spec)
        for c in at.get("plotly_chart")
    }
    value_map = specs[f"{product} — {quantity}"]["data"][0]
    stderr_map = specs[f"{product} — {quantity} stderr"]["data"][0]
    blank = np.isnan(_bin_array(value_map["z"]))
    assert int(blank.sum()) == 1
    np.testing.assert_array_equal(blank, np.isnan(_bin_array(stderr_map["z"])))
    assert not any("nan" in str(t) for row in value_map["text"] for t in row)
    assert [t for row in value_map["text"] for t in row].count("") == 1
    captions = " ".join(c.value for c in at.caption)
    assert "1 stored cell(s) with a price but no finite stderr" in captions
    table = next(
        d.value for d in at.dataframe if {"value_stderr", "axis_nu"} <= set(d.value.columns)
    )
    assert int(table["value_stderr"].isna().sum()) == 1  # kept as written
    assert not cfg.cache_root.exists()


def test_page6_lv_caption_without_finite_stderr(
    synthetic: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The LV point's stored price with a NaN value and stderr: the caption beside the face says
    so and names the precompute command instead of printing 'nan ± nan'."""
    cfg = _store_copy(synthetic, tmp_path)
    grid = api.list_grid(cfg)
    lv_id = str(grid.loc[grid["mode"] == "lv", "id"].iloc[0])
    _edit_products(cfg, lv_id, lambda df: df.assign(value=math.nan, value_stderr=math.nan))
    _set_env(cfg, tmp_path, monkeypatch)
    at = _render("6_product_grid.py")
    assert not at.exception, [e.value for e in at.exception]
    captions = [c.value for c in at.caption]
    lv_caption = [c for c in captions if c.startswith("nu = 0 is the surface's LV point")]
    assert len(lv_caption) == 1, captions
    assert "±" not in lv_caption[0] and "nan" not in lv_caption[0]
    assert "no finite value and stderr" in lv_caption[0]
    assert "volsto-precompute" in lv_caption[0]
    assert not cfg.cache_root.exists()


def test_page6_risk_bars_stderr_guard(
    synthetic: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The RiskReport bars: a risk row with a NaN ``value_stderr`` is not a bar and is counted in
    a caption; a risk table written without the column names it in an ``st.info`` with the
    ``--risk full`` command (a KeyError mid-render before the guard) and the risk table is still
    exported."""
    cfg = _store_copy(synthetic, tmp_path)
    _edit_tables(cfg.store_root, "risk", _nan_column("value_stderr"))
    _set_env(cfg, tmp_path, monkeypatch)
    at = _render("6_product_grid.py")
    assert not at.exception, [e.value for e in at.exception]
    captions = " ".join(c.value for c in at.caption)
    assert "1 row(s) with a NaN value or stderr are left out of these bars" in captions
    risk = next(d.value for d in at.dataframe if "section" in d.value.columns)
    assert int(risk["value_stderr"].isna().sum()) == 1

    cfg2 = _store_copy(synthetic, tmp_path / "no_se")
    _edit_tables(cfg2.store_root, "risk", _drop_column("value_stderr"))
    _set_env(cfg2, tmp_path / "no_se", monkeypatch)
    at = _render("6_product_grid.py")
    assert not at.exception, [e.value for e in at.exception]
    infos = [i.value for i in at.info]
    hits = [i for i in infos if "value_stderr" in i and "--risk full" in i]
    assert hits, infos
    downloads = [b.label for b in at.download_button]
    assert any(lbl.startswith("Excel: risk_") for lbl in downloads), downloads
    assert not any(": delta_" in lbl for lbl in downloads), downloads
    assert not cfg2.cache_root.exists()


# --------------------------------------------------------------------------------------------
# the real store vs the M4 baselines (slow; never computes)
# --------------------------------------------------------------------------------------------


#: Significant digits of the recorded baseline literals: values ``0.214576`` (six), stderrs
#: ``2.64e-05`` (three); the store carries full precision.
BASELINE_VALUE_DIGITS, BASELINE_STDERR_DIGITS = 6, 3


def _half_unit(value: float, digits: int) -> float:
    """Half a unit of the last of ``digits`` significant digits of ``value`` (the diagnostic
    scale of the worst-ratio print, not the pass criterion)."""
    return 0.5 * 10.0 ** (math.floor(math.log10(abs(value))) - (digits - 1))


@pytest.mark.slow
def test_model_risk_matches_m4_baselines() -> None:
    """The model-risk page's numbers for the cached 8·10⁵ placeholder models (LV, 1F ω = 1, 2, 3,
    2F Table 8.2) round to ``PLACEHOLDER_BASELINES`` (400k paths, seed 2024) exactly: the baseline
    literal **is** the six-significant-digit rounding of the page's value (``f"{v:.6g}"``) and the
    three-significant-digit rounding of its stderr — a literal whose trailing zeros were dropped
    (``0.3475``) gets no wider tolerance than one written in full.  Skips unless the repository
    store already holds those points with their leverage in the cache — the test never computes.
    The worst ``|diff| / half-unit`` is printed as a diagnostic."""
    from test_m4_regression import PLACEHOLDER_BASELINES

    cfg = ViewerConfig()
    grid = api.list_grid(cfg)
    have = grid[grid["label"].isin(BASELINE_LABELS)].set_index("label")
    missing = sorted(set(BASELINE_LABELS) - set(have.index))
    if missing:
        pytest.skip(f"store {cfg.store_root} lacks the placeholder baseline points {missing}")
    bad_particles = have[have["n_particles"] != BASELINE_N_PARTICLES]
    if not bad_particles.empty:
        pytest.skip(f"points not at {BASELINE_N_PARTICLES} particles: {list(bad_particles.index)}")
    no_lev = have[(have["mode"] != "lv") & ~have["has_leverage"]]
    if not no_lev.empty:
        pytest.skip(f"leverage missing in {cfg.cache_root} for {list(no_lev.index)}")
    prods = products_on_surface(cfg, grid, "placeholder")
    worst = 0.0
    n = 0
    for label, model in BASELINE_LABELS.items():
        rec = api.get_point(cfg, str(have.loc[label, "id"]))
        assert rec.provenance.get("pricing_n_paths") == BASELINE_N_PATHS, rec.provenance
        assert rec.provenance.get("pricing_seed") == BASELINE_SEED, rec.provenance
        stored = rec.products.set_index("key")
        for key, (value, se_base) in PLACEHOLDER_BASELINES[model].items():
            assert key in stored.index, (model, key, "not in the store's products")
            product, quantity = str(stored.loc[key, "product"]), str(stored.loc[key, "quantity"])
            table = model_risk_table(prods, product, quantity)[0].set_index("label")
            page_value = float(table["value"][label])
            page_se = float(table["value_stderr"][label])
            assert page_value == float(stored["value"][key])  # the page shows the store
            worst = max(
                worst,
                abs(page_value - value) / _half_unit(value, BASELINE_VALUE_DIGITS),
                abs(page_se - se_base) / _half_unit(se_base, BASELINE_STDERR_DIGITS),
            )
            assert float(f"{page_value:.{BASELINE_VALUE_DIGITS}g}") == value, (
                model,
                key,
                page_value,
                value,
            )
            assert float(f"{page_se:.{BASELINE_STDERR_DIGITS}g}") == se_base, (
                model,
                key,
                page_se,
                se_base,
            )
            n += 1
    print(
        f"\n{n} baseline cells matched exactly at 6 / 3 significant digits; worst |diff| / half-unit = {worst:.3f}"
    )
