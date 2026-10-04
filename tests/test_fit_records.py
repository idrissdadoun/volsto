"""Stored fit records (SPEC §13.4; owner's decision 5): the key, the store, reading instead of
fitting, the code-tag guard and the migration of fits stored before the records existed.

Nothing here touches the repository's ``cache/``: every store and cache is a temporary
directory.  One real marking fit runs (the reference surface, a few seconds); the backtest's
side is in ``tests/test_backtest.py`` (``test_marking_reads_the_fit_record`` and
``test_fit_records_migrate_from_a_backtest_store``).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from volsto.calibration import fit_records as fr
from volsto.calibration.cache import LeverageCache, build_market, spec_key
from volsto.calibration.fit_2f import FitResult, fit_2f_marking, fit_preset, load_fit_spec
from volsto.config import BergomiParams, SurfacePerturbation
from volsto.market.loaders import load_sabrw_fits

ROOT = Path(__file__).resolve().parents[1]
STUDY_DIR = ROOT / "configs" / "studies" / "m7_p1_marking"
REFERENCE = STUDY_DIR / "reference_ssr1_eps0.1.yaml"
SPX = STUDY_DIR / "spx_ssr1_eps0.1.yaml"
SNAPSHOT = ROOT / "configs" / "surfaces" / "snapshots" / "hdn_2022H2" / "spx_2022-12-30.yaml"


@pytest.fixture(scope="module")
def fitted() -> tuple[Any, FitResult]:
    """The reference study spec and its marking fit, made once."""
    fs = load_fit_spec(REFERENCE)
    _, surface, _ = build_market(fs.spec)
    r = fit_2f_marking(surface, fs.config, ssr_target=fs.ssr_target, step0=fs.step0_source(surface))
    return fs, r


def test_fit_key_is_the_fit_inputs_and_nothing_else() -> None:
    fs = load_fit_spec(SPX)
    cfg, ssr = fit_preset("desk", skew_eps=0.10), fs.ssr_target  # step 0 reads the SABRW fits
    assert cfg.step0 is not None
    fits = load_sabrw_fits(SNAPSHOT)
    assert fits is not None
    key = fr.fit_key(fr.fit_inputs(fs.spec, cfg, ssr, fits))
    assert len(key) == 64 and key == fr.fit_key(fr.fit_inputs(fs.spec, cfg, ssr, list(fits)))

    def other(**changes: Any) -> str:
        spec = dataclasses.replace(fs.spec, **{k: v for k, v in changes.items() if k != "cfg"})
        return fr.fit_key(fr.fit_inputs(spec, changes.get("cfg", cfg), ssr, fits))

    # the model, the particle and the scheme settings play no part: a refit moves none of them
    moved = dataclasses.replace(fs.spec.model, nu=fs.spec.model.nu * (1 + 1e-8))
    assert other(model=moved) == key
    assert other(particle=dataclasses.replace(fs.spec.particle, n_particles=123456)) == key
    assert other(sim=dataclasses.replace(fs.spec.sim, seed=99)) == key
    # everything the fit reads does
    market = dataclasses.replace(fs.spec.market, spot=fs.spec.market.spot * 1.0001)
    assert other(market=market) != key
    surface = dataclasses.replace(fs.spec.surface, eta=fs.spec.surface.eta * (1 + 1e-12))
    assert other(surface=surface) != key
    assert other(perturbation=SurfacePerturbation("parallel", {"size": 0.01})) != key
    assert other(cfg=dataclasses.replace(cfg, k2=cfg.k2 * 1.01)) != key
    assert fr.fit_key(fr.fit_inputs(fs.spec, cfg, ssr + 0.25, fits)) != key
    assert fr.fit_key(fr.fit_inputs(fs.spec, cfg, ssr, fits[:-1])) != key
    assert fr.fit_key(fr.fit_inputs(fs.spec, cfg, ssr, fits), code_tag="other") != key
    with pytest.raises(ValueError, match="reads the snapshot's SABRW fits"):
        fr.fit_inputs(fs.spec, cfg, ssr, None)
    # a fit whose step 0 reads the surface does not depend on the snapshot's fits
    surface_cfg = fit_preset("m7", skew_eps=0.10)
    assert surface_cfg.step0 is None
    assert fr.fit_inputs(fs.spec, surface_cfg, ssr, fits) == fr.fit_inputs(
        fs.spec, surface_cfg, ssr, None
    )
    digests = fr.input_digests(fr.fit_inputs(fs.spec, cfg, ssr, fits))
    assert sorted(digests) == ["fit", "market", "perturbation", "sabrw", "ssr_target", "surface"]


def test_records_are_immutable_and_atomic(tmp_path: Path, fitted: tuple[Any, FitResult]) -> None:
    fs, r = fitted
    records = fr.FitRecords.of_cache(tmp_path / "cache")
    assert records.root == tmp_path / "cache" / "fits" and records.keys() == []
    inputs = fr.fit_inputs(fs.spec, fs.config, fs.ssr_target, None)
    key = fr.fit_key(inputs)
    assert records.get(key) is None
    summary = fr.fit_summary(r)
    doc = records.put(key, inputs, summary, origin="test")
    assert doc["key"] == key and doc["fit_code_tag"] == fr.FIT_CODE_TAG and doc["origin"] == "test"
    assert doc["inputs"] == fr.input_digests(inputs)
    assert records.path(key) == records.root / key[:2] / f"{key}.json"
    assert not list(records.root.rglob(".*tmp")) and records.keys() == [key]
    back = fr.RecordedFit.from_summary(key, records.get(key)["fit"], "record")  # type: ignore[index]
    assert back.params == r.params and back.status == r.status  # bit for bit through JSON
    assert back.messages == tuple(r.messages) and back.result is None
    # the first writer wins: a second put never replaces the record
    other = json.loads(json.dumps(summary))
    other["params"]["nu"] *= 1 + 1e-8
    kept = records.put(key, inputs, other, origin="a later machine")
    assert kept["fit"]["params"] == summary["params"] and kept["origin"] == "test"
    # a record that cannot be trusted is never treated as absent (and refitted over)
    records.path(key).write_text(json.dumps({"version": 1, "key": "0" * 64}))
    with pytest.raises(ValueError, match="not the fit record"):
        records.get(key)
    records.path(key).write_text("{ torn")
    with pytest.raises(json.JSONDecodeError):
        records.get(key)


def test_a_recorded_fit_is_read_never_fitted_again(
    tmp_path: Path, fitted: tuple[Any, FitResult]
) -> None:
    """The invariant: on any machine that holds the record the parameters — hence the leverage
    key — are the recorded ones, whatever a refit there would give."""
    fs, r = fitted
    records = fr.FitRecords.of_cache(tmp_path / "cache")
    calls = {"n": 0}

    def this_machine() -> FitResult:
        calls["n"] += 1
        return r

    def another_machine() -> FitResult:  # the same fit in other last bits (SPEC §13.3)
        calls["n"] += 1
        moved = dataclasses.replace(
            r.params, nu=r.params.nu * (1 + 3e-9), theta=r.params.theta * (1 - 2e-9)
        )
        return dataclasses.replace(r, params=moved)

    kw: dict[str, Any] = {"ssr_target": fs.ssr_target, "sabrw_fits": None, "records": records}
    first = fr.recorded_marking_fit(fs.spec, fs.config, fit=this_machine, **kw)
    assert (first.source, calls["n"]) == ("fit", 1) and first.result is r
    assert first.params == r.params
    assert json.dumps(first.summary) == json.dumps(fr.fit_summary(r))  # NaN errors included
    again = fr.recorded_marking_fit(fs.spec, fs.config, fit=another_machine, **kw)
    assert (again.source, calls["n"]) == ("record", 1), "a recorded fit was fitted again"
    assert again.params == first.params and again.key == first.key and again.result is None
    key_of = lambda p: spec_key(dataclasses.replace(fs.spec, model=p))  # noqa: E731
    assert key_of(again.params) == key_of(first.params)
    # without the record the other machine's parameters give another leverage key: the defect
    assert key_of(another_machine().params) != key_of(first.params)
    # records=None always fits and stores nothing; other inputs are another record
    free = fr.recorded_marking_fit(
        fs.spec,
        fs.config,
        ssr_target=fs.ssr_target,
        sabrw_fits=None,
        records=None,
        fit=this_machine,
    )
    assert free.source == "fit" and len(records.keys()) == 1
    kw["ssr_target"] = fs.ssr_target + 0.5
    assert fr.recorded_marking_fit(fs.spec, fs.config, fit=this_machine, **kw).source == "fit"
    assert len(records.keys()) == 2
    # a writer that lost the race reads the winner's parameters
    lost = fr.FitRecords.of_cache(tmp_path / "race")
    inputs = fr.fit_inputs(fs.spec, fs.config, fs.ssr_target, None)

    def racing() -> FitResult:
        lost.put(fr.fit_key(inputs), inputs, fr.fit_summary(r), origin="the winner")
        return another_machine()

    kw.update(ssr_target=fs.ssr_target, records=lost)
    got = fr.recorded_marking_fit(fs.spec, fs.config, fit=racing, **kw)
    assert got.source == "record" and got.params == r.params


def test_fit_code_tag_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """The fit sources changed without a bump of FIT_CODE_TAG (or a re-recorded hash)."""
    fr.check_fit_guard()
    assert fr.read_fit_guard()[fr.FIT_CODE_TAG] == fr.fit_source_hash()
    for rel in fr.FIT_GUARDED_MODULES:
        assert (ROOT / rel).is_file(), rel
    monkeypatch.setattr(fr, "FIT_CODE_TAG", "not-recorded")
    with pytest.raises(AssertionError, match="was not bumped"):
        fr.check_fit_guard()


def test_migrate_fit_specs_on_a_temporary_cache(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The study fit specs become records under the key of their inputs, holding the stored
    parameters; nothing is fitted; a leverage is reported found only when the temporary cache
    holds it; a second run writes nothing; a record is never replaced."""
    specs = tmp_path / "specs"
    shutil.copytree(STUDY_DIR, specs)
    cache = tmp_path / "cache"
    n = len(list(specs.glob("*.yaml")))
    dry = fr.migrate_fit_specs(specs, cache, dry_run=True)
    assert len(dry.written) + len(dry.skipped) == n and not (cache / "fits").exists()
    rep = fr.migrate_fit_specs(specs, cache)
    assert rep.written == dry.written and len(rep.written) >= 5 and not rep.differs
    assert set(rep.skipped) == {p.name for p in specs.glob("rotation_*.yaml")}  # not fit specs
    assert len(rep.leverage_miss) == len(rep.written) and rep.leverage_hit == []
    records = fr.FitRecords.of_cache(cache)
    for name in rep.written:
        fs = load_fit_spec(specs / name)
        fits = load_sabrw_fits(ROOT / fs.snapshot) if fs.config.step0 is not None else None
        doc = records.get(fr.fit_key(fr.fit_inputs(fs.spec, fs.config, fs.ssr_target, fits)))
        assert doc is not None and doc["origin"] == f"migration: fit spec {name}"
        got = fr.RecordedFit.from_summary(doc["key"], doc["fit"], "record")
        assert got.params == fs.spec.model and got.status == fs.fit["status"]
        # reading the record gives the leverage key the spec was calibrated under
        assert spec_key(dataclasses.replace(fs.spec, model=got.params)) == spec_key(fs.spec)

    def never() -> FitResult:
        raise AssertionError("a migrated fit was fitted again")

    fs = load_fit_spec(specs / rep.written[0])
    fits = load_sabrw_fits(ROOT / fs.snapshot) if fs.config.step0 is not None else None
    read = fr.recorded_marking_fit(
        fs.spec, fs.config, ssr_target=fs.ssr_target, sabrw_fits=fits, records=records, fit=never
    )
    assert read.source == "record" and read.params == fs.spec.model
    again = fr.migrate_fit_specs(specs, cache)
    assert again.written == [] and sorted(again.present) == sorted(rep.written)
    # a leverage in the (temporary) cache is found
    target = LeverageCache(cache).entry_dir(fs.spec)
    target.mkdir(parents=True)
    import numpy as np

    np.savez(target / "leverage.npz", x=np.zeros(1))
    assert len(fr.migrate_fit_specs(specs, cache).leverage_hit) == 1
    # a spec edited afterwards differs from its record: reported, the record kept
    text = (specs / rep.written[0]).read_text()
    nu = fs.spec.model.nu
    (specs / rep.written[0]).write_text(text.replace(f"nu: {nu!r}", f"nu: {nu * 1.5!r}", 1))
    assert load_fit_spec(specs / rep.written[0]).spec.model.nu != nu
    changed = fr.migrate_fit_specs(specs, cache)
    assert list(changed.differs) == [rep.written[0]]
    assert fr.RecordedFit.from_summary("k", records.get(read.key)["fit"], "record").params.nu == nu  # type: ignore[index]
    # the CLI: status, migrate, exit 1 when a record differs
    assert fr.main(["status", "--cache", str(cache)]) == 0
    assert f"{len(rep.written)} fit records" in capsys.readouterr().out
    assert fr.main(["migrate", "--cache", str(cache), "--fit-specs", str(specs)]) == 1
    assert "DIFFERS" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        fr.main(["migrate", "--cache", str(cache)])
    assert isinstance(fs.spec.model, BergomiParams)


#: Modules of ``volsto/`` that call the marking fit directly, and why they may.
FIT_CALLERS = {
    "volsto/studies/backtest.py": "inside the record closure of BacktestRun._mark",
}
#: Sites whose fitted parameters still reach a leverage key without a record.  Wiring them is
#: an open question for the owner (SPEC §13.4): they are in-run refits on bumped surfaces or
#: consumers of the full FitResult, which a record does not hold.
NOT_YET_RECORDED = {
    "volsto/viewers/grid.py": "marking_fit: precompute marking points, S5, the payoff study",
    "volsto/hedging/hedger.py": "RecalibrationRule: refits on moved surfaces during a hedge",
    "volsto/risk/shadow_rotation.py": "the rotation states' refits",
}


def test_every_marking_fit_site_is_declared() -> None:
    """Walk ``volsto/``: a new direct caller of the marking fit must either go through the
    records or be declared here with its reason — it cannot appear silently."""
    import ast

    found: dict[str, int] = {}
    for py in sorted((ROOT / "volsto").rglob("*.py")):
        rel = py.relative_to(ROOT).as_posix()
        for node in ast.walk(ast.parse(py.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Call):
                f = node.func
                name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
                if name == "fit_2f_marking":
                    found[rel] = found.get(rel, 0) + 1
    assert set(found) == set(FIT_CALLERS) | set(NOT_YET_RECORDED), found
    # the backtest's only call is the closure handed to recorded_marking_fit
    src = (ROOT / "volsto/studies/backtest.py").read_text()
    assert found["volsto/studies/backtest.py"] == 1
    assert "fit=lambda: fit_2f_marking(surface, cfg, ssr_target=ssr, step0=step0)," in src
