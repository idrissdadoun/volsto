"""M9 Part 2 — pages 7 (Marking / calibration) and 8 (Hedging) rendered headless with the
Streamlit ``AppTest`` against the synthetic store and outputs of ``tests/_synthetic_store.py``
(no calibration, no Monte Carlo, the leverage cache pointed at a directory that must stay
absent).  On top of the synthetic outputs this file writes — through the M8b table builders
themselves, so the layouts are the real ones — a study-C comparison table
(:func:`volsto.studies.m8b.table_C` from two synthetic study-C ``TaskResult`` s and a static
prediction), a study-B table (:func:`~volsto.studies.m8b.table_B`), the study-C task JSONs and a
small :class:`~volsto.hedging.hedger.HedgeResult` pickle beside the synthetic study-A run so the
histogram and the per-date recalibration sections render; a second copy of the outputs with a
sparse (surface, ssr_target, skew_eps) fits list exercises the dependent fit selectors.

Three further copies carry the stderr pathologies the pages must refuse to plot: a fits CSV whose
``ssr_lsv@1y`` has lost its ``ssr_lsv_se@1y`` twin and gained an all-NaN ``ssr_lsv@2y`` pillar
(the twin-less pillar named in an ``st.info`` and absent from both figure and table, the NaN
pillar counted in a caption and kept as NaN in the table), one whose every ``ssr_lsv_se@T`` twin
is absent (a notice per pillar, no realised-SSR figure and no realised-SSR table) and a study-B
table whose 'pure LV' row has a leakage but a NaN ``leakage_desk_se`` (left out of the bars,
counted in a caption, kept in the table).

Six further copies (:func:`_store_copy` / :func:`_outputs_copy`) close the same rule on the rest
of the two pages' figures — the store point's SSR term structure, the shadow-rotation bars and
the study-A ranking — each with both triggers: a NaN stderr beside a value (the row is left out
of the figure, counted in a caption and kept as written in the table) and the stderr column
missing from the table altogether (an ``st.info`` naming the column and the producing command, no
figure, the table still exported — a ``KeyError`` mid-render before the guard).

One test walks the eight read-only pages' declared ``(value, stderr)`` pairs (their ``MC_PAIRS``) against
what the read API actually returns for the synthetic artefacts and for the repository's real
store, cache and outputs: the API's stderr pairing was rewritten under the pages once
(``reattach_stderr_names`` renamed study C's ``recal_se`` twin to ``recal_pnl_desk_stderr`` while
page 8 still declared ``recal_stderr``), so a rename now fails a test instead of a page.

The pages against the repository's real ``outputs/m7`` (the sparse M7 fit pairs, the 150-column
binding maps) and ``outputs/m8b`` (the runs with their ``.pkl``, the study tables) run when those
git-ignored files exist; those tests assert what the pages show, never which runs another process
has written.  Wall clocks are printed (``-s``), never asserted.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import json
import math
import pickle
import re
import shutil
import time
import warnings
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest
from _synthetic_store import HEDGING_RUN_ID, make_synthetic_outputs, make_synthetic_store

from volsto.calibration.diagnostics import CalibrationReport
from volsto.hedging.hedger import HedgeResult
from volsto.market.curves import ForwardCurve
from volsto.models.leverage import LeverageFunction
from volsto.risk.shadow_rotation import ROTATION_CONVENTION
from volsto.studies.m8b import TaskResult, save_result, table_B, table_C
from volsto.viewers.config import ViewerConfig
from volsto.viewers.pages import _common, page_paths

ROOT = Path(__file__).resolve().parents[1]
PAGE_7 = next(p for p in page_paths() if p.name == "7_marking.py")
PAGE_8 = next(p for p in page_paths() if p.name == "8_hedging.py")
C_RUN_ID = "C__cliquet_1y__rotated__delta_only__rota_1__sabr_linked"
#: The desk-toggle annotations: ``mean <value> ± <stderr>`` and ``qNN -> desk qMM <value> ± <se>``.
_MEAN = re.compile(r"^mean (\S+) ± (\S+)$")
_DESK_Q = re.compile(r"^q(\d\d) -> desk q(\d\d) (\S+) ± (\S+)$")
_PERF = pd.errors.PerformanceWarning


def _load_page(path: Path) -> ModuleType:
    """Import a ``<n>_<name>.py`` page module (the ``run_if_streamlit`` guard renders nothing
    outside a Streamlit script run)."""
    spec = importlib.util.spec_from_file_location(f"page_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hedge_result(rng: np.random.Generator, n: int = 400) -> HedgeResult:
    """A small HedgeResult with the arrays the API's per-path reader uses."""
    prod = rng.normal(0.0, 0.05, n)
    hedges = np.column_stack([-0.9 * prod + rng.normal(0, 0.01, n), rng.normal(0, 0.005, n)])
    empty = pd.DataFrame()
    return HedgeResult(
        product="cliquet 1y",
        strategy="delta only",
        dates=np.linspace(1 / 12, 1.0, 12),
        instruments=("spot", "cap-call strip"),
        targets=("delta",),
        pnl_product=prod,
        pnl_hedges=hedges,
        costs=np.zeros(n),
        pnl_recalibration=rng.normal(-0.001, 0.002, n),
        termination=np.full(n, np.nan),
        value_0=0.0155,
        value_0_stderr=0.00004,
        by_date=empty,
        quantities=empty,
        residual=empty,
        world_paths=None,  # type: ignore[arg-type]  # dropped by save_result too
        pricing_notes=(),
        recalibrations=pd.DataFrame({"t": [0.5], "recalibrated": [True]}),
        budget={"wall_seconds": 3.4},
    )


def _c_task(rota: float, policy: str, seed: int) -> TaskResult:
    rng = np.random.default_rng(seed)
    return TaskResult(
        key=f"C__cliquet_1y__rotated__delta_only__rota_{rota:g}__{policy}",
        study="C",
        product="cliquet 1y",
        world="rotated",
        strategy="delta only",
        rota=rota,
        policy=policy,
        frequency="monthly",
        n_paths_pricing=20_000,
        n_paths_world=20_000,
        n_particles=800_000,
        n_dates=12,
        value_0=(1.55, 0.004),
        mean=(-0.08 * rota + rng.normal(0, 0.002), 0.05),
        std=(7.1, 0.04),
        quantiles={"q05": (-11.0, 0.2), "q50": (0.1, 0.05), "q95": (12.0, 0.2)},
        zero_cost_mean=(-0.08 * rota, 0.05),
        regimes=[{"regime": "realised vol low", "n": 100, "mean": 0.1, "stderr": 0.02}],
        attribution=[{"component": "recalibration", "mean": -0.054 * rota, "stderr": 0.004}],
        recal_total=(-0.054 * rota, 0.004 * rota),
        recal_by_date=[
            {"t": 0.5, "mean": -0.054 * rota, "stderr": 0.004 * rota, "alive_fraction": 1.0}
        ],
        n_refits=1,
        world_meta={"skew_90_110_6m_base_vp": 5.6, "skew_90_110_6m_rotated_vp": 5.6 + 0.56 * rota},
    )


def _b_task(world: str, mean: float) -> TaskResult:
    return TaskResult(
        key=f"B__cliquet_1y__{world.replace(' ', '_')}__preset",
        study="B",
        product="cliquet 1y",
        world=world,
        strategy="cliquet preset",
        n_paths_world=20_000,
        n_dates=252,
        value_0=(1.53, 0.022),
        mean=(mean, 0.0076),
        std=(1.07, 0.1),
        quantiles={"q05": (-1.58, 0.026), "q95": (0.25, 0.008)},
        zero_cost_mean=(mean, 0.0076),
        world_value_0=(1.54, 0.022),
        static_spread=(-0.017, 0.031),
    )


def write_extra_m8b(outputs: Path) -> None:
    """The study-C / study-B artefacts on top of :func:`make_synthetic_outputs`."""
    m8b = outputs / "m8b"
    rng = np.random.default_rng(11)
    with (m8b / "A" / f"{HEDGING_RUN_ID}.pkl").open("wb") as fh:
        pickle.dump(_hedge_result(rng), fh)
    c_tasks = [
        _c_task(1.0, "sabr_linked", 1),
        _c_task(2.0, "sabr_linked", 2),
        _c_task(1.0, "none", 3),
    ]
    save_result(c_tasks[0], _hedge_result(rng), m8b)
    for t in c_tasks[1:]:
        save_result(t, None, m8b)
    static = {
        ("cliquet 1y", "sabr_linked"): {
            "desk_pnl_shadow": [0.060, 0.005],
            "desk_pnl_usual": [0.010, 0.002],
        }
    }
    table_C(c_tasks, static).to_csv(m8b / "m8b_table_C.csv", index=False)
    b_tasks = [_b_task("same", -0.56), _b_task("pure LV", -0.71)]
    skipped = [
        {
            "world": "historical",
            "product": "cliquet 1y",
            "status": "surface artefact, skipped",
            "reason": "gate",
        }
    ]
    table_B(b_tasks, skipped).to_csv(m8b / "m8b_table_B.csv", index=False)


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    base = tmp_path_factory.mktemp("pages_c")
    t0 = time.perf_counter()
    info = make_synthetic_store(base / "store")
    make_synthetic_outputs(base / "outputs")
    write_extra_m8b(base / "outputs")
    print(
        f"\nsynthetic store + outputs (+ study C / B) written in {time.perf_counter() - t0:.2f} s"
    )
    cfg = ViewerConfig(
        cache_root=base / "cache_empty", store_root=base / "store", outputs_root=base / "outputs"
    )
    return {"cfg": cfg, "base": base, **info}


def _run(cfg: ViewerConfig, path: Path, monkeypatch: pytest.MonkeyPatch, base: Path):  # type: ignore[no-untyped-def]
    """Render ``path`` headless; pandas' PerformanceWarning (a fragmented frame being extended
    column by column) is an error — the warning filters are process-wide, so the script thread's
    warnings are caught here."""
    from streamlit.testing.v1 import AppTest

    for var, value in cfg.to_env().items():
        monkeypatch.setenv(var, value)
    monkeypatch.setenv("VOLSTO_VIEWER_CONFIG", str(base / "no_viewer.yaml"))
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", _PERF)
        at = AppTest.from_file(str(path), default_timeout=300).run()
    print(f"\n{path.name}: rendered in {time.perf_counter() - t0:.2f} s")
    assert not at.exception, [e.value for e in at.exception]
    perf = [str(w.message)[:60] for w in caught if issubclass(w.category, _PERF)]
    assert not perf, perf
    assert not cfg.cache_root.exists()
    return at


def _histogram(at) -> tuple[str, list[str]]:  # type: ignore[no-untyped-def]
    """Title and annotation texts of the page-8 histogram (the first plotly figure)."""
    spec = json.loads(at.get("plotly_chart")[0].proto.spec)
    layout = spec["layout"]
    return str(layout["title"]["text"]), [str(a["text"]) for a in layout.get("annotations", [])]


def _figure_titles(at) -> list[str]:  # type: ignore[no-untyped-def]
    """Titles of every plotly figure on the page, in render order."""
    return [
        str(json.loads(el.proto.spec)["layout"].get("title", {}).get("text", ""))
        for el in at.get("plotly_chart")
    ]


def _figure(at, title: str) -> dict[str, object]:  # type: ignore[no-untyped-def]
    """The plotly JSON spec (``data`` traces and ``layout``) of the one figure titled ``title``."""
    specs = [json.loads(el.proto.spec) for el in at.get("plotly_chart")]
    hits = [f for f in specs if f["layout"].get("title", {}).get("text", "") == title]
    assert len(hits) == 1, (title, [f["layout"].get("title", {}).get("text") for f in specs])
    spec: dict[str, object] = hits[0]
    return spec


def _outputs_copy(
    cfg0: ViewerConfig, base: Path, name: str, edit: Callable[[Path], None]
) -> ViewerConfig:
    """A copy of the synthetic outputs under ``base / name``, ``edit`` applied to it once (the
    pathological table), and the config reading it with the fixture's store and absent cache."""
    root = base / name
    if not root.exists():
        shutil.copytree(cfg0.outputs_root, root)
        edit(root)
    return ViewerConfig(cache_root=cfg0.cache_root, store_root=cfg0.store_root, outputs_root=root)


def _store_copy(
    cfg0: ViewerConfig, base: Path, name: str, edit: Callable[[Path], None]
) -> ViewerConfig:
    """The same for the synthetic store: a copy under ``base / name`` with ``edit`` applied to
    it once, read with the fixture's outputs."""
    root = base / name
    if not root.exists():
        shutil.copytree(cfg0.store_root, root)
        edit(root)
    return ViewerConfig(cache_root=cfg0.cache_root, store_root=root, outputs_root=cfg0.outputs_root)


def _marking_ssr(store_root: Path) -> Path:
    """The ``ssr`` parquet of the synthetic store's one marking point."""
    hits = sorted(store_root.glob("results/points/marking__*/ssr.parquet"))
    assert len(hits) == 1, hits
    return hits[0]


# --------------------------------------------------------------------------------------------
# page 7
# --------------------------------------------------------------------------------------------


def test_page_7_renders(synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    """Fits, store marking point, binding maps, flag maps and the shadow-rotation table all
    render; the rotation convention is printed verbatim; every table has its Excel button."""
    cfg, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg, ViewerConfig) and isinstance(base, Path)
    at = _run(cfg, PAGE_7, monkeypatch, base)
    assert [t.value for t in at.title] == ["Marking / calibration"]
    captions = " ".join(c.value for c in at.caption)
    assert "3 points" in captions and "code tag synthetic" in captions
    assert ROTATION_CONVENTION in [c.value for c in at.code]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["status"] == "binding" and metrics["stage-3 verdict"] == "FAIL"
    assert at.selectbox(key="m7_surface").value == "spx"
    # the mean |L - 1| metric and the raw fits table carry the no-stderr note (a particle-method
    # summary the API declares exact — shown as such, never bare)
    note = "leverage-function summary recorded upstream without a standard error"
    assert sum(c.value.startswith(note) for c in at.caption) == 2
    subheaders = [s.value for s in at.subheader]
    assert subheaders == [
        "Marking fits (M7 Part 3)",
        "Store marking points (volsto-precompute)",
        "Binding maps",
        "Shadow rotation",
    ]
    n_tables = len(at.dataframe)
    excel_buttons = [b for b in at.get("download_button") if b.label.startswith("Excel:")]
    assert n_tables >= 10 and len(excel_buttons) == n_tables, (n_tables, len(excel_buttons))
    labels = [b.label for b in at.get("download_button")]
    assert any(lab.startswith("PNG:") or lab.startswith("HTML:") for lab in labels)
    # the shadow-rotation table carries value ± stderr pairs and the per-vp scaled stderr
    rot = next(d.value for d in at.dataframe if "per_vp_90_110_0.5y_stderr" in d.value.columns)
    assert set(rot["greek"]) >= {
        "lv_rotation",
        "usual",
        "recalibrated",
        "fee_shadow",
        "desk_pnl_shadow",
    }
    assert np.allclose(rot["per_vp_90_110_0.5y_stderr"], rot["per_rota_stderr"] / 0.56)
    # no warning: every requested point exists in the synthetic tables
    assert not at.warning and not at.error


def test_page_7_store_point_slider_snaps(
    synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The store section's snap sliders show the single grid value and the store point's SSR
    table (ssr_lsv ± stderr, the naked first-order SSR and the ssr_target)."""
    cfg, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg, ViewerConfig) and isinstance(base, Path)
    at = _run(cfg, PAGE_7, monkeypatch, base)
    captions = [c.value for c in at.caption]
    assert any(c.startswith("ssr_target (store): 1") for c in captions)
    assert any(c.startswith("skew_eps (store): 0.1") for c in captions)
    ssr = next(d.value for d in at.dataframe if "ssr_naked_first_order" in d.value.columns)
    assert {"T", "ssr_lsv", "ssr_lsv_stderr", "ssr_target"} <= set(ssr.columns)
    assert (ssr["ssr_target"] == 1.0).all()


def test_page_7_dependent_sliders(
    synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A sparse fits list — spx (1.0, 0.05), (1.0, 0.10), (1.5, 0.05) and reference (1.0, 0.10):
    the skew_eps slider offers only the values fitted for the chosen ssr_target, so the default and
    every move land on a fitted pair (metrics shown, no missing-fit warning); the surface opens on
    spx although 'reference' sorts first."""
    cfg0, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg0, ViewerConfig) and isinstance(base, Path)
    sparse = base / "outputs_sparse"
    if not sparse.exists():
        shutil.copytree(cfg0.outputs_root, sparse)
        fits = pd.read_csv(sparse / "m7" / "p1_marking_fits.csv")
        rows = [fits.iloc[0].copy() for _ in range(4)]
        rows[1]["skew_eps"], rows[1]["status"] = 0.05, "interior"
        rows[2]["ssr_target"], rows[2]["skew_eps"] = 1.5, 0.05
        rows[3]["surface"] = "reference"
        pd.DataFrame(rows).to_csv(sparse / "m7" / "p1_marking_fits.csv", index=False)
    cfg = ViewerConfig(cache_root=cfg0.cache_root, store_root=cfg0.store_root, outputs_root=sparse)
    at = _run(cfg, PAGE_7, monkeypatch, base)
    assert list(at.selectbox(key="m7_surface").options) == ["reference", "spx"]
    assert at.selectbox(key="m7_surface").value == "spx"
    # AppTest exposes the slider positions as their formatted labels
    assert list(at.select_slider(key="m7_ssr").options) == ["1.0", "1.5"]
    assert list(at.select_slider(key="m7_eps").options) == ["0.05", "0.1"]
    assert {m.label: m.value for m in at.metric}["status"] == "interior"  # (1.0, 0.05)
    assert not at.warning
    at.select_slider(key="m7_ssr").set_value(1.5).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not any(s.key == "m7_eps" for s in at.select_slider)  # one fitted eps: a caption
    assert any(c.value == "skew_eps: 0.05 (single grid value)" for c in at.caption)
    assert {m.label: m.value for m in at.metric}["status"] == "binding"
    assert not at.warning
    at.selectbox(key="m7_surface").select("reference").run()
    assert not at.exception, [e.value for e in at.exception]
    captions = [c.value for c in at.caption]
    assert "ssr_target: 1.0 (single grid value)" in captions
    assert "skew_eps: 0.1 (single grid value)" in captions
    assert {m.label: m.value for m in at.metric}["status"] == "binding" and not at.warning
    assert not cfg.cache_root.exists()


def test_page_7_ssr_without_stderr_twin(
    synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fits CSV whose ``ssr_lsv@1y`` has no ``ssr_lsv_se@1y`` twin and whose added ``ssr_lsv@2y``
    pillar is NaN (with a stderr): the twin-less pillar is named in an ``st.info`` with the
    missing twin and the producing stage and is not in the realised-SSR figure or its table (never
    a NaN error bar); the NaN pillar is left out of the figure, counted in a caption and kept as
    NaN in the table; the figure holds the 0.25y pillar alone with its real error bar."""
    cfg0, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg0, ViewerConfig) and isinstance(base, Path)
    outputs = base / "outputs_ssr_no_se"
    if not outputs.exists():
        shutil.copytree(cfg0.outputs_root, outputs)
        fits_path = outputs / "m7" / "p1_marking_fits.csv"
        fits = pd.read_csv(fits_path).drop(columns=["ssr_lsv_se@1y"])
        fits["ssr_lsv@2y"] = np.nan
        fits["ssr_lsv_se@2y"] = 0.01
        fits.to_csv(fits_path, index=False)
    cfg = ViewerConfig(cache_root=cfg0.cache_root, store_root=cfg0.store_root, outputs_root=outputs)
    at = _run(cfg, PAGE_7, monkeypatch, base)
    infos = [i.value for i in at.info]
    hit = [i for i in infos if i.startswith("ssr_lsv@1y is in ")]
    assert len(hit) == 1, infos
    assert "ssr_lsv_se@1y" in hit[0] and "scripts/m7_p1_marking.py stage 3" in hit[0]
    fig = _figure(at, "realised LSV SSR by pillar (stage 3) vs the ssr_target input")
    data = fig["data"]
    assert isinstance(data, list) and len(data) == 1
    tr = data[0]
    assert tr["x"] == ["0.25y"] and tr["y"] == [1.333] and tr["error_y"]["array"] == [0.0147]
    captions = [c.value for c in at.caption]
    assert any(c.startswith("1 pillar(s) with a NaN realised SSR or stderr") for c in captions)
    tab = next(d.value for d in at.dataframe if "ssr_lsv_stderr" in d.value.columns)
    assert tab["T"].tolist() == ["0.25y", "2y"]  # the twin-less 1y pillar is not padded in
    assert (
        math.isnan(float(tab["ssr_lsv"].iloc[1])) and float(tab["ssr_lsv_stderr"].iloc[1]) == 0.01
    )
    assert (tab["ssr_target"] == 1.0).all()
    n_tables = len(at.dataframe)
    excel_buttons = [b for b in at.get("download_button") if b.label.startswith("Excel:")]
    assert len(excel_buttons) == n_tables
    assert not at.warning and not at.error


def test_page_7_ssr_no_pillar_plottable(
    synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fits CSV whose every ``ssr_lsv_se@T`` twin is absent: one notice per pillar, no
    realised-SSR figure and no realised-SSR table; the page otherwise renders."""
    cfg0, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg0, ViewerConfig) and isinstance(base, Path)
    outputs = base / "outputs_ssr_no_se_at_all"
    if not outputs.exists():
        shutil.copytree(cfg0.outputs_root, outputs)
        fits_path = outputs / "m7" / "p1_marking_fits.csv"
        fits = pd.read_csv(fits_path)
        fits.drop(columns=[c for c in fits.columns if c.startswith("ssr_lsv_se@")]).to_csv(
            fits_path, index=False
        )
    cfg = ViewerConfig(cache_root=cfg0.cache_root, store_root=cfg0.store_root, outputs_root=outputs)
    at = _run(cfg, PAGE_7, monkeypatch, base)
    infos = [i.value for i in at.info if i.value.startswith("ssr_lsv@")]
    assert sorted(i.split(" ")[0] for i in infos) == ["ssr_lsv@0.25y", "ssr_lsv@1y"], infos
    assert "realised LSV SSR by pillar (stage 3) vs the ssr_target input" not in _figure_titles(at)
    assert not any(b.label.startswith("Excel: realised SSR") for b in at.get("download_button"))
    assert not at.warning and not at.error


def test_page_7_store_point_ssr_nan_stderr(
    synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store copy whose marking point's ``ssr`` table carries a NaN ``ssr_lsv_stderr`` on the
    first pillar: that pillar is left out of the store-point SSR figure (which would otherwise
    draw a bar with an invisible error bar), counted in a caption and kept as NaN in the table;
    every error bar still drawn is finite."""
    cfg0, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg0, ViewerConfig) and isinstance(base, Path)

    def edit(root: Path) -> None:
        path = _marking_ssr(root)
        df = pd.read_parquet(path)
        df.loc[0, "ssr_lsv_stderr"] = np.nan
        df.to_parquet(path, index=False)

    cfg = _store_copy(cfg0, base, "store_ssr_nan_se", edit)
    at = _run(cfg, PAGE_7, monkeypatch, base)
    data = _figure(at, "SSR term structure of the store point")["data"]
    assert isinstance(data, list)
    lsv = next(tr for tr in data if tr["name"] == "LSV numerical")
    assert len(lsv["x"]) == 4 and 0.25 not in lsv["x"]
    assert all(math.isfinite(float(v)) for v in lsv["error_y"]["array"])
    captions = [c.value for c in at.caption]
    assert any(
        c.startswith("1 pillar(s) with a NaN stored SSR or stderr") for c in captions
    ), captions
    tab = next(d.value for d in at.dataframe if "skew_lsv_stderr" in d.value.columns)
    assert len(tab) == 5 and math.isnan(float(tab["ssr_lsv_stderr"].iloc[0]))
    assert not at.warning and not at.error


def test_page_7_store_point_ssr_without_stderr_column(
    synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A store copy with no ``ssr_lsv_stderr`` column at all (dropped from every point, so the
    merge-on-read cannot fill it from a sibling: the read raised a KeyError mid-render before the
    guard): the page renders, names the column and the ``volsto-precompute`` line in an
    ``st.info``, draws no store-point SSR figure and still exports the table."""
    cfg0, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg0, ViewerConfig) and isinstance(base, Path)

    def edit(root: Path) -> None:
        paths = sorted(root.glob("results/points/*/ssr.parquet"))
        assert paths
        for path in paths:  # the merged table takes the union of the per-point columns
            pd.read_parquet(path).drop(columns=["ssr_lsv_stderr"]).to_parquet(path, index=False)

    cfg = _store_copy(cfg0, base, "store_ssr_no_se_column", edit)
    at = _run(cfg, PAGE_7, monkeypatch, base)
    hit = [i.value for i in at.info if "ssr_lsv_stderr" in i.value]
    assert len(hit) == 1 and "volsto-precompute" in hit[0], [i.value for i in at.info]
    assert "SSR term structure of the store point" not in _figure_titles(at)
    assert any(b.label.startswith("Excel: store SSR table") for b in at.get("download_button"))
    assert not at.warning and not at.error


def test_page_7_shadow_rotation_nan_stderr(
    synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``p1_marking_shadow_rotation.csv`` with a NaN ``per_rota_se`` on its first row: that greek
    is left out of the rotation bars (``dropna(subset=['per_rota'])`` alone kept it and drew a
    bar with no error bar), counted in a caption and kept as written in the table."""
    cfg0, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg0, ViewerConfig) and isinstance(base, Path)

    def edit(root: Path) -> None:
        path = root / "m7" / "p1_marking_shadow_rotation.csv"
        rot = pd.read_csv(path)
        rot.loc[0, "per_rota_se"] = np.nan
        rot.to_csv(path, index=False)

    cfg = _outputs_copy(cfg0, base, "outputs_rota_nan_se", edit)
    at = _run(cfg, PAGE_7, monkeypatch, base)
    data = _figure(at, "P&L per +1 rota by greek and policy (± stderr)")["data"]
    assert isinstance(data, list) and len(data) == 1
    assert "lv_rotation" not in data[0]["x"] and len(data[0]["x"]) == 5
    assert all(math.isfinite(float(v)) for v in data[0]["error_y"]["array"])
    captions = [c.value for c in at.caption]
    assert any(
        c.startswith("1 greek/policy row(s) with a NaN per-rota P&L or stderr") for c in captions
    ), captions
    tab = next(d.value for d in at.dataframe if "per_rota_stderr" in d.value.columns)
    assert len(tab) == 6
    assert math.isnan(float(tab.loc[tab["greek"] == "lv_rotation", "per_rota_stderr"].iloc[0]))
    assert not at.warning and not at.error


def test_page_7_shadow_rotation_without_stderr_column(
    synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``p1_marking_shadow_rotation.csv`` written without ``per_rota_se`` (a KeyError mid-render
    before the guard): the page renders, names the column and ``scripts/m7_p1_marking.py``, draws
    no rotation figure and still exports the table."""
    cfg0, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg0, ViewerConfig) and isinstance(base, Path)

    def edit(root: Path) -> None:
        path = root / "m7" / "p1_marking_shadow_rotation.csv"
        pd.read_csv(path).drop(columns=["per_rota_se"]).to_csv(path, index=False)

    cfg = _outputs_copy(cfg0, base, "outputs_rota_no_se_column", edit)
    at = _run(cfg, PAGE_7, monkeypatch, base)
    hit = [i.value for i in at.info if "per_rota_stderr" in i.value]
    assert len(hit) == 1 and "scripts/m7_p1_marking.py" in hit[0], [i.value for i in at.info]
    assert "P&L per +1 rota by greek and policy (± stderr)" not in _figure_titles(at)
    assert any(b.label == "Excel: shadow rotation table" for b in at.get("download_button"))
    assert not at.warning and not at.error


# --------------------------------------------------------------------------------------------
# page 8
# --------------------------------------------------------------------------------------------


def test_page_8_renders(synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs list, the first run's histogram (from the pickle), distribution / regimes /
    attribution tables, the study-C comparison with the ratio stderr, the study-A ranking and
    the study-B leakage table."""
    cfg, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg, ViewerConfig) and isinstance(base, Path)
    at = _run(cfg, PAGE_8, monkeypatch, base)
    assert [t.value for t in at.title] == ["Hedging"]
    assert at.selectbox(key="run_id").value == HEDGING_RUN_ID
    assert at.selectbox(key="hist_component").value == "pnl_total"
    subheaders = [s.value for s in at.subheader]
    assert subheaders[:2] == ["Stored runs", "One run"] and len(subheaders) == 5
    runs = at.dataframe[0].value
    assert set(runs["study"]) == {"A", "C"} and len(runs) == 4
    c = next(d.value for d in at.dataframe if "ratio_stderr" in d.value.columns)
    # the read API re-attaches table_C's ``recal_se`` / ``static_se`` to their value column
    # (api.reattach_stderr_names), so the page's twin names are the value names + "_stderr"
    assert len(c) == 3 and {
        "recal_pnl_desk",
        "recal_pnl_desk_stderr",
        "static_prediction",
        "static_prediction_stderr",
    } <= set(c.columns)
    row = c[(c["rota"] == 1.0) & (c["recalibration"] == "sabr_linked")].iloc[0]
    assert math.isclose(row["recal_pnl_desk"], 0.054, rel_tol=1e-9)  # desk sign of −0.054
    assert math.isclose(row["static_prediction"], 0.060, rel_tol=1e-9)
    assert math.isclose(row["ratio"], 0.9, rel_tol=1e-9)
    expected_se = 0.9 * math.sqrt((0.004 / 0.054) ** 2 + (0.005 / 0.060) ** 2)
    assert math.isclose(row["ratio_stderr"], expected_se, rel_tol=1e-9)
    a = next(d.value for d in at.dataframe if "rank" in d.value.columns)
    assert {"std", "std_stderr", "desk_mean", "desk_mean_stderr"} <= set(a.columns)
    # the numeric study-B table pairs the desk leakage with its own stderr (table_B's
    # ``leakage_desk_se``); its ± display copy in the expander joins the pair into one text column
    b = next(
        d.value
        for d in at.dataframe
        if {"leakage_desk", "leakage_desk_stderr"} <= set(d.value.columns)
    )
    assert {"world", "status"} <= set(b.columns) and len(b) == 3
    shown = next(
        d.value
        for d in at.dataframe
        if "leakage_desk" in d.value.columns and "leakage_desk_stderr" not in d.value.columns
    )
    assert shown["leakage_desk"].astype(str).str.contains("±").sum() == 2  # two ok rows
    assert _figure_titles(at)[-1] == "desk leakage by world and product (± stderr)"
    n_tables = len(at.dataframe)
    excel_buttons = [x for x in at.get("download_button") if x.label.startswith("Excel:")]
    assert n_tables >= 7 and len(excel_buttons) == n_tables, (n_tables, len(excel_buttons))
    assert any(x.label == "Excel: study B (value ± stderr)" for x in excel_buttons)
    assert not at.warning and not at.error


def test_page_8_study_b_nan_stderr(
    synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A study-B table whose 'pure LV' row has a desk leakage but a NaN ``leakage_desk_se``: the
    leakage bars hold the 'same' world alone (with its real error bar), a caption counts the one
    row left out, and the numeric table still holds all three rows with the NaN as written."""
    cfg0, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg0, ViewerConfig) and isinstance(base, Path)
    outputs = base / "outputs_b_nan_se"
    if not outputs.exists():
        shutil.copytree(cfg0.outputs_root, outputs)
        b_path = outputs / "m8b" / "m8b_table_B.csv"
        b = pd.read_csv(b_path)
        b.loc[b["world"] == "pure LV", "leakage_desk_se"] = np.nan
        b.to_csv(b_path, index=False)
    cfg = ViewerConfig(cache_root=cfg0.cache_root, store_root=cfg0.store_root, outputs_root=outputs)
    at = _run(cfg, PAGE_8, monkeypatch, base)
    fig = _figure(at, "desk leakage by world and product (± stderr)")
    data = fig["data"]
    assert isinstance(data, list) and [tr["name"] for tr in data] == ["same"]
    assert data[0]["x"] == ["cliquet 1y"] and data[0]["error_y"]["array"] == [0.0076]
    captions = [c.value for c in at.caption]
    assert any(
        c.startswith("1 row(s) with a desk leakage but a NaN leakage_desk_stderr") for c in captions
    ), captions
    b = next(
        d.value
        for d in at.dataframe
        if {"leakage_desk", "leakage_desk_stderr"} <= set(d.value.columns)
    )
    assert len(b) == 3
    row = b[b["world"] == "pure LV"].iloc[0]
    assert math.isclose(float(row["leakage_desk"]), -0.71 - 1.53, rel_tol=1e-9) or pd.notna(
        row["leakage_desk"]
    )
    assert math.isnan(float(row["leakage_desk_stderr"]))
    assert not at.warning and not at.error


def test_page_8_study_a_without_stderr_column(
    synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``m8b_table_A.csv`` written without ``std_se``: the ranking bars read ``std_stderr`` and
    raised a KeyError mid-render before the guard.  Now the page renders, names the column with
    ``scripts/m8b.py --study A``, draws no ranking figure and still exports the table."""
    cfg0, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg0, ViewerConfig) and isinstance(base, Path)

    def edit(root: Path) -> None:
        path = root / "m8b" / "m8b_table_A.csv"
        pd.read_csv(path).drop(columns=["std_se"]).to_csv(path, index=False)

    cfg = _outputs_copy(cfg0, base, "outputs_a_no_se_column", edit)
    at = _run(cfg, PAGE_8, monkeypatch, base)
    hit = [i.value for i in at.info if "std_stderr" in i.value]
    assert len(hit) == 1, [i.value for i in at.info]
    assert "m8b_table_A.csv" in hit[0] and "scripts/m8b.py --study A" in hit[0]
    assert "P&L std by strategy (± stderr), ranked" not in _figure_titles(at)
    assert any(b.label == "Excel: study A table" for b in at.get("download_button"))
    assert not at.warning and not at.error


def test_page_8_study_a_nan_stderr(
    synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """``m8b_table_A.csv`` with a second strategy whose ``std_se`` is NaN: the ranking bars hold
    the paired strategy alone (with its real error bar), a caption counts the row left out, and
    the table still holds both rows as written."""
    cfg0, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg0, ViewerConfig) and isinstance(base, Path)

    def edit(root: Path) -> None:
        path = root / "m8b" / "m8b_table_A.csv"
        a = pd.read_csv(path)
        extra = a.iloc[[0]].copy()
        extra["strategy"] = "delta + vega"
        extra["std"] = 5.5
        extra["std_se"] = np.nan
        extra["rank"] = 2
        pd.concat([a, extra], ignore_index=True).to_csv(path, index=False)

    cfg = _outputs_copy(cfg0, base, "outputs_a_nan_se", edit)
    at = _run(cfg, PAGE_8, monkeypatch, base)
    data = _figure(at, "P&L std by strategy (± stderr), ranked")["data"]
    assert isinstance(data, list) and len(data) == 1
    assert data[0]["x"] == ["delta only"] and data[0]["error_y"]["array"] == [0.04]
    captions = [c.value for c in at.caption]
    assert any(c.startswith("1 row(s) with a NaN std or std_stderr") for c in captions), captions
    a = next(d.value for d in at.dataframe if "rank" in d.value.columns)
    assert len(a) == 2
    assert math.isnan(float(a.loc[a["strategy"] == "delta + vega", "std_stderr"].iloc[0]))
    assert not at.warning and not at.error


def test_page_8_desk_toggle_and_c_run(
    synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The desk-sign toggle flips the histogram: the title says '(desk sign)', the mean
    annotation is the negative of the hedger-sign one with the same stderr, every stored quantile
    is annotated with its desk name (the hedger's q05 is the desk's q95); selecting the study-C run
    shows its per-date recalibration P&L table (t, mean, stderr, alive_fraction)."""
    cfg, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg, ViewerConfig) and isinstance(base, Path)
    at = _run(cfg, PAGE_8, monkeypatch, base)
    title_h, ann_h = _histogram(at)
    assert title_h.endswith("(hedger sign)")
    mean_h = _MEAN.match(ann_h[0])
    assert mean_h is not None and float(mean_h.group(1)) < 0  # the synthetic A run loses money
    at.checkbox(key="desk_sign").check().run()
    assert not at.exception, [e.value for e in at.exception]
    title_d, ann_d = _histogram(at)
    assert title_d.endswith("(desk sign)") and len(ann_d) == len(ann_h)
    mean_d = _MEAN.match(ann_d[0])
    assert mean_d is not None
    assert float(mean_d.group(1)) == -float(mean_h.group(1))
    assert mean_d.group(2) == mean_h.group(2)  # stderr unchanged by the sign flip
    quantiles = [_DESK_Q.match(a) for a in ann_d[1:]]
    assert quantiles and all(q is not None for q in quantiles)
    assert all(int(q.group(1)) + int(q.group(2)) == 100 for q in quantiles if q is not None)
    assert any(a.startswith("q05 -> desk q95 ") for a in ann_d)
    # each desk quantile is the negative of the hedger one, same stderr
    for h, d in zip(ann_h[1:], ann_d[1:]):
        hv, hse = h.split(" ")[1], h.split(" ")[3]
        dv, dse = d.split(" ")[4], d.split(" ")[6]
        assert float(dv) == -float(hv) and dse == hse, (h, d)
    at.selectbox(key="run_study").select("C").run()
    assert not at.exception
    at.selectbox(key="run_id").select(C_RUN_ID).run()
    assert not at.exception, [e.value for e in at.exception]
    recal = next(d.value for d in at.dataframe if "alive_fraction" in d.value.columns)
    assert list(recal["t"]) == [0.5] and "stderr" in recal.columns
    assert not cfg.cache_root.exists()


@pytest.mark.parametrize("page", [PAGE_7, PAGE_8])
def test_pages_without_outputs(
    synthetic: dict[str, object], page: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No M7 / M8b files: the pages name the producing scripts and compute nothing."""
    cfg0, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg0, ViewerConfig) and isinstance(base, Path)
    cfg = ViewerConfig(
        cache_root=cfg0.cache_root, store_root=cfg0.store_root, outputs_root=base / "no_outputs"
    )
    at = _run(cfg, page, monkeypatch, base)
    codes = " ".join(c.value for c in at.code)
    assert len(at.warning) >= 1
    if page is PAGE_7:
        assert "scripts/m7_p1_marking.py" in codes
        assert any("Store marking points" in s.value for s in at.subheader)  # store part renders
    else:
        assert "scripts/m8b.py --study A" in codes and "--study C" in codes


# --------------------------------------------------------------------------------------------
# the pages' hard-coded stderr twin names against the read API
# --------------------------------------------------------------------------------------------

#: ``MC_PAIRS`` keys that need an artefact the repository does not always hold (the study-D
#: table is git-ignored): unresolved, they are printed, not asserted.
OPTIONAL_FRAMES = {"hedging_table.D"}


def _leverage_entry(cfg: ViewerConfig, cache_root: Path) -> ViewerConfig:
    """A synthetic leverage cache entry (``leverage.npz`` + ``diagnostics.json``) for the first
    calibrated point of the store, so page 1's ``leverage.error_map`` pair resolves here too
    (the repository's own cache entries carry the leverage without a diagnostics table)."""
    from volsto.viewers import api

    grid = api.list_grid(cfg)
    key = str(grid.loc[grid["cache_key"].astype(str) != "", "cache_key"].iloc[0])
    entry = cache_root / key
    entry.mkdir(parents=True, exist_ok=True)
    times, ks = np.linspace(0.05, 3.0, 8), np.linspace(-1.0, 1.0, 9)
    LeverageFunction(
        times,
        ks,
        np.ones((times.size, ks.size)),
        ForwardCurve.flat(100.0, 0.02, 0.0),
        {"code_tag": "synthetic", "git_commit": "0000000"},
    ).save(entry / "leverage.npz")
    rows = [
        {
            "T": T,
            "k": k,
            "cp": 1,
            "price": 5.0,
            "price_stderr": 0.01,
            "model_vol": 0.2,
            "target_vol": 0.2,
            "error_vp": 0.01,
            "stderr_vp": 0.02,
        }
        for T in (0.5, 1.0)
        for k in (-0.1, 0.0, 0.1)
    ]
    CalibrationReport(pd.DataFrame(rows), pd.DataFrame(), 400_000, 2024, 1.0).save(
        entry / "diagnostics.json"
    )
    return ViewerConfig(
        cache_root=cache_root, store_root=cfg.store_root, outputs_root=cfg.outputs_root
    )


def _api_frames(cfg: ViewerConfig) -> dict[str, list[pd.DataFrame]]:
    """Every non-empty read-API frame a page's :data:`MC_PAIRS` can name, keyed
    ``<accessor>.<table>`` — what the API *actually* returns for this configuration."""
    from volsto.viewers import api
    from volsto.viewers.store import StoreReader

    out: dict[str, list[pd.DataFrame]] = {}

    def add(key: str, df: pd.DataFrame | None) -> None:
        if df is not None and not df.empty:
            out.setdefault(key, []).append(df)

    grid = api.list_grid(cfg)
    for pid in [str(i) for i in grid.get("id", pd.Series(dtype=str))]:
        rec = api.get_point(cfg, pid)
        for name, df in rec.tables().items():
            add(f"point.{name}", df)
        with contextlib.suppress(api.MissingPoint, api.MissingArtefact):
            add("leverage.error_map", api.get_leverage(cfg, pid).error_map)
        rows = StoreReader(cfg.store_root).risk(pid)
        for product in dict.fromkeys(str(p) for p in rows.get("product", pd.Series(dtype=str))):
            add("point.risk", api.get_risk(cfg, pid, product))
    try:
        mk = api.get_marking(cfg)
    except api.MissingArtefact:
        pass
    else:
        for name, df in mk.tables().items():
            add(f"marking.{name}", df)
    for table in ("A", "B", "C", "D"):
        with contextlib.suppress(api.MissingArtefact):
            add(f"hedging_table.{table}", api.get_hedging_table(cfg, table))
    runs = api.list_hedging_runs(cfg)
    for run_id in [str(r) for r in runs.get("run_id", pd.Series(dtype=str))]:
        run = api.get_hedging_run(cfg, run_id)
        add("hedging_run.distribution", run.distribution)
        add("hedging_run.regimes", run.regimes)
        add("hedging_run.attribution", run.attribution)
    return out


def test_page_twin_names_match_api(synthetic: dict[str, object]) -> None:
    """Every ``<value>_stderr`` name the eight read-only pages hard-code is the name the read API emits.

    The pages declare their ``(value, stderr)`` pairs per API frame (``MC_PAIRS``); this walks
    them against what the API returns for the synthetic store / outputs and — when the
    repository holds them — for the real ``outputs/m7``, ``outputs/m8b``, results store and
    leverage cache.  The API's stderr pairing was rewritten once under the pages
    (``reattach_stderr_names`` renamed study C's ``recal_stderr`` to ``recal_pnl_desk_stderr``
    and page 8 kept reading the old name until it failed here): a rename now fails a test
    instead of a page."""
    cfg = synthetic["cfg"]
    base = synthetic["base"]
    assert isinstance(cfg, ViewerConfig) and isinstance(base, Path)
    sources = {"synthetic": _api_frames(_leverage_entry(cfg, base / "cache_walk"))}
    if (ROOT / "outputs").is_dir():
        sources["repository"] = _api_frames(ViewerConfig())
    pages = {path.name: _load_page(path) for path in page_paths()}
    assert len(pages) == 9
    # page 9 (What-if) computes its own frames and reads none from the API: no pairs to walk
    assert not hasattr(pages["9_what_if.py"], "MC_PAIRS")
    del pages["9_what_if.py"]
    unresolved: list[tuple[str, str]] = []
    checked = 0
    for name, mod in sorted(pages.items()):
        pairs = getattr(mod, "MC_PAIRS", None)
        assert isinstance(pairs, dict) and pairs, f"{name} declares no MC_PAIRS"
        for key, declared in pairs.items():
            found = [(src, f) for src, frames in sources.items() for f in frames.get(key, [])]
            if not found:
                unresolved.append((name, key))
                continue
            for value, stderr in declared:
                seen = False
                for src, frame in found:
                    if value not in frame.columns:
                        continue
                    seen = True
                    assert stderr in frame.columns, (
                        f"{name}: {key} of the {src} artefacts pairs {value!r} with something "
                        f"other than {stderr!r} — the API returns {sorted(frame.columns)}"
                    )
                    checked += 1
                assert seen, f"{name}: no {src} {key} frame carries the value column {value!r}"
    print(f"\ntwin names walked: {checked} (value, stderr) pairs over {sorted(sources)}")
    assert {key for _, key in unresolved} <= OPTIONAL_FRAMES, unresolved
    if unresolved:
        print(f"not resolvable here (git-ignored artefacts): {unresolved}")


@pytest.mark.skipif(
    not (ROOT / "outputs" / "m7" / "p1_marking_fits.csv").is_file(), reason="real outputs/m7 absent"
)
def test_page_7_real_outputs(synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    """The repository's real M7 marking results — a sparse list of (ssr_target, skew_eps) pairs
    per surface, the 150-column binding maps: the page opens on the spx fit with its metrics and no
    missing-fit warning, the skew_eps selector follows the chosen ssr_target on every surface, and
    the binding maps render without pandas' PerformanceWarning (asserted by ``_run``)."""
    cfg0, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg0, ViewerConfig) and isinstance(base, Path)
    cfg = ViewerConfig(
        cache_root=cfg0.cache_root, store_root=cfg0.store_root, outputs_root=ROOT / "outputs"
    )
    fits = pd.read_csv(ROOT / "outputs" / "m7" / "p1_marking_fits.csv")
    at = _run(cfg, PAGE_7, monkeypatch, base)
    surfaces = sorted(fits["surface"].unique())
    assert at.selectbox(key="m7_surface").value == ("spx" if "spx" in surfaces else surfaces[0])
    metrics = {m.label: m.value for m in at.metric}
    print(f"real m7 fits: {len(fits)} on {surfaces}; default fit metrics {metrics}")
    assert {"status", "mean |L - 1|", "stage-3 verdict"} <= set(metrics)
    assert not at.warning, [w.value for w in at.warning]
    n_tables = len(at.dataframe)
    excel_buttons = [b for b in at.get("download_button") if b.label.startswith("Excel:")]
    assert len(excel_buttons) == n_tables
    for surface in surfaces:
        at.selectbox(key="m7_surface").select(surface).run()
        assert not at.exception, [e.value for e in at.exception]
        sub = fits[fits["surface"] == surface]
        for ssr in sorted(sub["ssr_target"].unique()):
            if any(s.key == "m7_ssr" for s in at.select_slider):
                at.select_slider(key="m7_ssr").set_value(ssr).run()
                assert not at.exception, [e.value for e in at.exception]
            fitted = sorted(sub.loc[sub["ssr_target"] == ssr, "skew_eps"].unique())
            if len(fitted) == 1:
                assert any(
                    c.value == f"skew_eps: {fitted[0]} (single grid value)" for c in at.caption
                ), (surface, ssr, fitted)
            else:
                assert list(at.select_slider(key="m7_eps").options) == [str(v) for v in fitted]
            assert {"status", "mean |L - 1|", "stage-3 verdict"} <= {m.label for m in at.metric}
            assert not at.warning, (surface, ssr, [w.value for w in at.warning])
    assert not cfg.cache_root.exists()


@pytest.mark.skipif(not (ROOT / "outputs" / "m8b" / "A").is_dir(), reason="real outputs/m8b absent")
def test_page_8_real_outputs(synthetic: dict[str, object], monkeypatch: pytest.MonkeyPatch) -> None:
    """The repository's real M8b results (per-path pickles, the study tables): the page renders,
    and a run that has its .pkl — chosen through the selectors, whatever another process has
    written since — shows the histogram of its per-path P&L; skipped when no run has paths."""
    cfg0, base = synthetic["cfg"], synthetic["base"]
    assert isinstance(cfg0, ViewerConfig) and isinstance(base, Path)
    cfg = ViewerConfig(
        cache_root=cfg0.cache_root, store_root=cfg0.store_root, outputs_root=ROOT / "outputs"
    )
    at = _run(cfg, PAGE_8, monkeypatch, base)
    runs = at.dataframe[0].value
    with_paths = runs[runs["has_paths"].astype(bool)]
    print(f"real runs: {len(runs)} ({len(with_paths)} with paths)")
    if with_paths.empty:
        pytest.skip("no real run has its per-path .pkl yet")
    study, run_id = str(with_paths["study"].iloc[0]), str(with_paths["run_id"].iloc[0])
    at.selectbox(key="run_study").select(study).run()
    assert not at.exception, [e.value for e in at.exception]
    at.selectbox(key="run_id").select(run_id).run()
    assert not at.exception, [e.value for e in at.exception]
    title, annotations = _histogram(at)
    assert title.startswith(f"pnl_total of {run_id} — ") and "world paths" in title
    assert annotations and _MEAN.match(annotations[0]) is not None
    assert not any("per-path" in w.value for w in at.warning), [w.value for w in at.warning]
    infos = " ".join(i.value for i in at.info)
    assert "m8b_table_C.csv is empty" in infos or any(
        "ratio_stderr" in d.value.columns for d in at.dataframe
    )
    # the study-B leakage chart is drawn from the paired ``leakage_desk_stderr`` (table_B's
    # ``leakage_desk_se``); a table without the pair says so instead
    b = next((d.value for d in at.dataframe if "leakage_desk" in d.value.columns), None)
    if b is not None:
        paired = "leakage_desk_stderr" in b.columns
        print(f"real table B: {len(b)} rows, leakage stderr paired = {paired}")
        assert ("desk leakage by world and product (± stderr)" in _figure_titles(at)) == paired
        assert any("regenerate with scripts/m8b.py --study B" in c.value for c in at.caption) == (
            not paired
        )
    n_tables = len(at.dataframe)
    excel_buttons = [b for b in at.get("download_button") if b.label.startswith("Excel:")]
    assert len(excel_buttons) == n_tables
    assert not cfg.cache_root.exists()


# --------------------------------------------------------------------------------------------
# helpers and the standing rule
# --------------------------------------------------------------------------------------------


def test_page_helpers() -> None:
    """The pages' helpers: the ± formatting and the bar trace with error bars are ``_common``'s
    (no page-local copies), called with the pages' 4 significant digits; the page-specific
    delta-method ratio stderr, desk quantile name, exact fit command, status heatmap codes, and
    the ``@T`` column parser in both modes — paired (a pillar enters the frame only with its
    stderr twin, the twin-less ones come back by name, never padded) and ``paired=False`` (an
    exact first-order quantity: value only, nothing unpaired) — with the ``_se@T`` file spelling
    of the twin a notice must name."""
    page8 = _load_page(PAGE_8)
    page7 = _load_page(PAGE_7)
    assert page8.fmt_pm is _common.fmt_pm and page8.pm_display is _common.pm_display
    assert page7.pm_display is _common.pm_display
    assert page7.bar_stderr_trace is page8.bar_stderr_trace is _common.bar_stderr_trace
    for mod in (page7, page8):
        assert not hasattr(mod, "pm_frame") and not hasattr(mod, "bar_error_trace")
        assert mod.PM_DIGITS == 4
    assert _common.fmt_pm(1.23456, 0.0123, page8.PM_DIGITS) == "1.235 ± 0.012"
    assert _common.fmt_pm(math.nan, 0.1) == "nan" and _common.fmt_pm(2.0, math.nan) == "2 (no se)"
    df = pd.DataFrame({"a": [1.0, 2.0], "a_stderr": [0.1, math.nan], "b": ["x", "y"]})
    pm = _common.pm_display(df, digits=page7.PM_DIGITS)
    assert list(pm.columns) == ["a", "b"] and pm["a"].tolist() == ["1 ± 0.1", "2 (no se)"]
    bar = _common.bar_stderr_trace(
        ["u", "v"], pd.Series([1.0, -2.0]), pd.Series([0.1, 0.2]), name="t", unit="vp"
    )
    assert bar.error_y.array == (0.1, 0.2) and bar.text[1] == "-2 ± 0.2 vp"
    assert math.isclose(page8.ratio_stderr(2.0, 0.2, 4.0, 0.4), 0.5 * math.sqrt(0.01 + 0.01))
    assert math.isnan(page8.ratio_stderr(1.0, 0.1, 0.0, 0.1))
    assert [page8.desk_quantile(q) for q in ("q01", "q05", "q50", "q95", "q99")] == [
        "q99",
        "q95",
        "q50",
        "q05",
        "q01",
    ]
    with pytest.raises(ValueError):
        page8.desk_quantile("mean")
    assert (
        page7.fit_command("reference", 1.0, 0.05)
        == "scripts/m7_p1_marking.py --surfaces reference --pairs 1:0.05"
    )
    assert (
        page7.fit_command("spx", 1.5, 0.1)
        == "scripts/m7_p1_marking.py --surfaces spx --pairs 1.5:0.1"
    )
    assert page7.DEFAULT_SURFACE == "spx"
    row = pd.Series(
        {
            "ssr_lsv@0.25y": 1.3,
            "ssr_lsv@0.25y_stderr": 0.01,
            "ssr_lsv@1y": 1.1,
            "short_gap@0.25y": 0.02,
            "x": 0.0,
        }
    )
    # a pillar without its stderr twin is returned by name, never padded with a NaN stderr
    tab, unpaired = page7.at_columns(row, "ssr_lsv")
    assert tab["T"].tolist() == ["0.25y"] and tab["ssr_lsv_stderr"].tolist() == [0.01]
    assert unpaired == ["ssr_lsv@1y"]
    assert page7.file_stderr_name("ssr_lsv@1y") == "ssr_lsv_se@1y"
    exact, none = page7.at_columns(row, "short_gap", paired=False)
    assert list(exact.columns) == ["T", "short_gap"] and exact["T"].tolist() == ["0.25y"]
    assert none == []
    empty, unpaired = page7.at_columns(row, "fwd/spot")
    assert empty.empty and unpaired == []
    fig = page7.status_heatmap(
        pd.DataFrame(
            {"ssr_target": [1.0, 1.5], "skew_eps": [0.1, 0.1], "status": ["binding", "infeasible"]}
        ),
        x="skew_eps",
        y="ssr_target",
        status="status",
        title="t",
    )
    assert fig.data[0].z[0][0] == 1 and fig.data[0].z[1][0] == 2


def test_pages_never_calibrate_or_simulate() -> None:
    """Neither page imports the cache, the calibrator or a Monte Carlo engine; they call the
    read API only."""
    for path in (PAGE_7, PAGE_8):
        tree = ast.parse(path.read_text())
        imported = {
            n.module or "" for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module
        } | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any(
            m.startswith(("volsto.calibration", "volsto.engine", "volsto.models")) for m in imported
        ), (path, imported)
        src = path.read_text()
        assert (
            "get_or_calibrate" not in src and "MonteCarlo" not in src and "LeverageCache" not in src
        )
        assert "read_parquet" not in src and "StoreReader" not in src
