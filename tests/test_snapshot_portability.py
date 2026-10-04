"""Machine-dependent arithmetic (CONTRIBUTING.md; SPEC §13.3; owner's decisions 2026-10-03).

**Invariant.** Nothing depends on the bit pattern of a number that came out of LAPACK or an
iterative solver: identities (digests, keys, bindings) are computed from inputs and stored
bytes, never from a recomputation on the current machine, and a fresh fit is compared with a
stored one by :mod:`volsto.market.compare` — the same surface within ``SURFACE_TOL_VP`` vol
points on the fixed grid — never by its bytes or its parameters.

The walking tests:

* every tracked snapshot under ``configs/surfaces/snapshots`` (256) is re-imported from the
  HDN sample and is the same surface as the committed file (one test per file, so
  ``pytest -n auto`` spreads them; skipped when the sample is absent);
* every snapshot shipped in the golden backtest store is the one its record names (stored
  bytes against stored bytes, no sample needed) and a fresh import reproduces its surface;
* the stored SABRW fits are data: only the importers (HDN, ORATS) and the script that
  back-fills the section may fit them (:func:`fit_sabrw` moves by up to 3.7 vol points under a few ulps of input
  noise), so every other module of ``volsto/`` and ``scripts/`` reads them back.
"""

from __future__ import annotations

import ast
import io
import json
import logging
import tarfile
from pathlib import Path

import pytest
import yaml

from volsto.market import compare, import_hdn
from volsto.market.loaders import load_ssvi_surface
from volsto.studies.backtest import snapshot_bytes_digest

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS = ROOT / "configs" / "surfaces" / "snapshots"
HDN_SAMPLE = ROOT / "data" / "hdn_sample" / "options_sample_2022H2"
GOLDEN_R2 = ROOT / "tests" / "golden" / "backtest_store_r2.tar.gz"
TRACKED = sorted(SNAPSHOTS.rglob("spx_*.yaml"))
#: The only modules that may fit SABRW smiles (an importer writes them into the snapshot it
#: produces; the script back-fills the section of snapshots imported before it existed).
SABRW_WRITERS = (
    "volsto/market/sabrw.py",
    "volsto/market/import_hdn.py",
    "volsto/market/import_orats.py",  # the second importer: it writes the section too
    "scripts/add_sabrw_sections.py",
)


def need_sample() -> Path:
    if not HDN_SAMPLE.exists():
        pytest.skip("the HDN sample is absent (data/hdn_sample/options_sample_2022H2)")
    return HDN_SAMPLE


def fresh_import(tmp_path: Path, doc: dict, date: str) -> Path:  # type: ignore[type-arg]
    """Import ``date`` again with the stored snapshot's own surface settings."""
    fit = doc["provenance"]["fit"]
    repaired = any(str(k).startswith("calendar_") for k in fit)
    logging.disable(logging.WARNING)
    try:
        cfg, *_ = import_hdn.import_day(
            need_sample(),
            date,
            doc["provenance"]["underlying"],
            essvi=bool(fit["essvi"]),
            calendar_repair=import_hdn.DEFAULT_CALENDAR_REPAIR if repaired else None,
        )
    finally:
        logging.disable(logging.NOTSET)
    return import_hdn.write_snapshot(cfg, tmp_path / "fresh.yaml")


def test_the_tolerance_is_the_agreed_one() -> None:
    assert compare.SURFACE_TOL_VP == 1e-4
    assert len(compare.SURFACE_GRID_T) == 6 and len(compare.SURFACE_GRID_K) == 33
    assert len(TRACKED) == 256


@pytest.mark.parametrize("path", TRACKED, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_tracked_snapshot_is_reproduced_within_tolerance(path: Path, tmp_path: Path) -> None:
    """A fresh import of the day is the same surface as the committed snapshot."""
    doc = yaml.safe_load(path.read_text())
    fresh = fresh_import(tmp_path, doc, doc["provenance"]["quote_date"])
    diff = compare.snapshot_diff_vp(path, fresh)
    assert diff <= compare.SURFACE_TOL_VP, f"{path.name}: {diff:.3g} vol points"
    # the inputs and the selection are the same: only fitted floats may differ
    new = yaml.safe_load(fresh.read_text())["provenance"]
    old = doc["provenance"]
    for key in ("file_sha256", "n_points", "n_expiries", "dropped", "filters"):
        assert old[key] == new[key], (path.name, key)
    assert list(old["forwards"]) == list(new["forwards"]), path.name


def test_comparator_detects_a_moved_surface(tmp_path: Path) -> None:
    path = SNAPSHOTS / "hdn_2022H2" / "spx_2022-12-30.yaml"
    doc = yaml.safe_load(path.read_text())
    assert compare.same_snapshot(path, path)
    assert compare.same_surface(load_ssvi_surface(path), load_ssvi_surface(path))
    moved = dict(doc, ssvi=dict(doc["ssvi"], eta=doc["ssvi"]["eta"] * (1.0 + 1e-4)))
    p_moved = tmp_path / "moved.yaml"
    p_moved.write_text(yaml.safe_dump(moved, sort_keys=False))
    assert compare.snapshot_diff_vp(path, p_moved) > compare.SURFACE_TOL_VP
    assert "vol points" in (compare.snapshot_difference(path, p_moved) or "")
    # a float in the last digits is the same surface; a count is not the same import
    near = dict(doc, ssvi=dict(doc["ssvi"], eta=doc["ssvi"]["eta"] * (1.0 + 1e-9)))
    p_near = tmp_path / "near.yaml"
    p_near.write_text(yaml.safe_dump(near, sort_keys=False))
    assert compare.same_snapshot(path, p_near)
    prov = dict(doc["provenance"], n_points=doc["provenance"]["n_points"] + 1)
    p_count = tmp_path / "count.yaml"
    p_count.write_text(yaml.safe_dump(dict(doc, provenance=prov), sort_keys=False))
    assert compare.same_snapshot_surface(path, p_count)
    assert "provenance.n_points" in (compare.snapshot_difference(path, p_count) or "")
    # the stored SABRW fits and the timestamp are not compared
    sab = dict(doc["sabrw"], fits=doc["sabrw"]["fits"][:-1])
    prov = dict(doc["provenance"], created_utc="2000-01-01T00:00:00+00:00")
    p_sab = tmp_path / "sabrw.yaml"
    p_sab.write_text(yaml.safe_dump(dict(doc, sabrw=sab, provenance=prov), sort_keys=False))
    assert compare.same_snapshot(path, p_sab)


def golden_members() -> dict[str, bytes]:
    with tarfile.open(GOLDEN_R2) as tar:
        return {
            m.name: tar.extractfile(m).read()  # type: ignore[union-attr]
            for m in tar.getmembers()
            if m.isfile()
        }


def test_golden_store_records_name_its_stored_snapshots() -> None:
    """Stored bytes against stored bytes: each date's record names the digest of the snapshot
    shipped beside it (no import, no sample, any machine)."""
    files = golden_members()
    dates = sorted({n.split("/")[1] for n in files if n.startswith("dates/")})
    assert len(dates) == 5
    for d in dates:
        record = json.loads(files[f"dates/{d}/done.json"])["record"]
        assert record["snapshot"] == snapshot_bytes_digest(files[f"snapshots/spx_{d}.yaml"]), d


def test_golden_store_snapshots_are_reproduced_within_tolerance(tmp_path: Path) -> None:
    """A fresh import reproduces each stored golden snapshot's surface (the bytes need not)."""
    need_sample()
    for name, data in sorted(golden_members().items()):
        if not name.startswith("snapshots/"):
            continue
        stored = tmp_path / Path(name).name
        stored.write_bytes(data)
        doc = yaml.safe_load(io.BytesIO(data))
        fresh = fresh_import(tmp_path, doc, doc["provenance"]["quote_date"])
        diff = compare.snapshot_diff_vp(stored, fresh)
        assert diff <= compare.SURFACE_TOL_VP, f"{name}: {diff:.3g} vol points"


def test_stored_sabrw_fits_are_never_fitted_again() -> None:
    """Walk ``volsto/`` and ``scripts/``: a call of ``fit_sabrw`` or ``sabrw_fits`` outside the
    writers is a second derivation of numbers that are stored data."""
    offenders = []
    for base in ("volsto", "scripts"):
        for py in sorted((ROOT / base).rglob("*.py")):
            rel = py.relative_to(ROOT).as_posix()
            if rel in SABRW_WRITERS:
                continue
            for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
                if isinstance(node, ast.Call):
                    f = node.func
                    name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                    if name in ("fit_sabrw", "sabrw_fits"):
                        offenders.append(f"{rel}:{node.lineno} {name}")
    assert offenders == [], offenders
