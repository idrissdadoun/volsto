"""The vendor source (SPEC §18.9, M11 Part 5): the invariant's walking test and the contract
every registered source meets.

**Invariant.**  Nothing outside a vendor source knows where a vendor's days live, how a day
becomes a chain, what its checksums are or where its prior rate curve comes from.

The contract tests need the HDN sample and skip when ``data/hdn_sample/`` is absent; the walk
needs nothing.
"""

from __future__ import annotations

import ast
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from volsto.calibration import history, raw_history
from volsto.market import import_hdn as ih
from volsto.market import vendor
from volsto.studies import backtest as bt

ROOT = Path(__file__).resolve().parents[1]
HDN_SAMPLE = ROOT / "data" / "hdn_sample" / "options_sample_2022H2"
TOY_CONFIG = ROOT / "configs" / "backtest" / "hdn_2022h2_toy.yaml"

#: The modules that may know a vendor's layout and call its loaders.
VENDOR_MODULES = ("volsto/market/import_hdn.py", "volsto/market/vendor.py")
#: Declared exceptions, each with its reason.
DECLARED = {
    "scripts/capture_yfinance.py": "writes today's chain IN the HDN layout: a producer of the "
    "vendor's format, not a reader of a vendor's data",
}
LAYOUT_LITERALS = ("day_by_date", "_options.csv")
VENDOR_LOADERS = ("load_day", "load_manifest", "import_day", "rate_curve")
#: Where a chain must come from: what the importer's vendor-independent steps read.
CHAIN_COLUMNS = (
    "expiry", "T", "cp", "strike", "bid", "ask", "mid", "r", "df", "underlying_close",
    "iv_bid", "iv_ask", "iv", "iv_flag",
)  # fmt: skip
CHAIN_ATTRS = ("quote_date", "underlying", "file", "rate_tenors", "rate_zeros", "spot")


def need_sample() -> Path:
    if not HDN_SAMPLE.exists():
        pytest.skip("the HDN sample is absent (data/hdn_sample/options_sample_2022H2)")
    return HDN_SAMPLE


def test_no_site_outside_the_vendor_modules_knows_a_vendor_layout() -> None:
    """Walk ``volsto/`` and ``scripts/``: a vendor's layout literal, or a call of the HDN
    loaders through the importer module, outside the vendor modules is a site that knows a
    vendor's files — the class of defect this invariant closes."""
    offenders: list[str] = []
    for base in ("volsto", "scripts"):
        for py in sorted((ROOT / base).rglob("*.py")):
            rel = py.relative_to(ROOT).as_posix()
            if rel in VENDOR_MODULES or rel in DECLARED:
                continue
            tree = ast.parse(py.read_text(encoding="utf-8"))
            docstrings = {
                id(n.body[0].value)
                for n in ast.walk(tree)
                if isinstance(n, ast.Module | ast.FunctionDef | ast.ClassDef | ast.AsyncFunctionDef)
                and n.body
                and isinstance(n.body[0], ast.Expr)
                and isinstance(n.body[0].value, ast.Constant)
            }
            hdn_names: set[str] = set()  # names bound to the HDN importer or its loaders
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module == "volsto.market":
                    hdn_names |= {a.asname or a.name for a in node.names if a.name == "import_hdn"}
                if isinstance(node, ast.ImportFrom) and node.module == "volsto.market.import_hdn":
                    for a in node.names:
                        if a.name in VENDOR_LOADERS:
                            offenders.append(f"{rel}:{node.lineno} imports {a.name}")
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and id(node) not in docstrings
                    and any(lit in node.value for lit in LAYOUT_LITERALS)
                ):
                    offenders.append(f"{rel}:{node.lineno} layout literal {node.value[:40]!r}")
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in VENDOR_LOADERS
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id in hdn_names
                ):
                    offenders.append(f"{rel}:{node.lineno} calls {node.func.attr}")
    assert offenders == [], offenders
    for rel in (*VENDOR_MODULES, *DECLARED):
        assert (ROOT / rel).is_file(), rel


def test_registry_and_the_backtest_choice() -> None:
    assert set(vendor.SOURCES) == {"hdn"} and tuple(vendor.SOURCES) == bt.VENDORS
    assert isinstance(vendor.vendor_source("hdn", "/nowhere"), vendor.HdnSource)
    with pytest.raises(vendor.VendorError, match="unknown vendor 'nope'"):
        vendor.vendor_source("nope", "/nowhere")
    cfg = bt.load_backtest_config(TOY_CONFIG)
    a, b = cfg.data_source(), cfg.data_source()
    assert isinstance(a, vendor.HdnSource) and a is not b and a.location == cfg.data_root
    assert bt.ABSENT == vendor.ABSENT


@pytest.fixture()
def source_copy(tmp_path: Path) -> vendor.HdnSource:
    """A private HDN source: three day files linked from the sample, the manifest copied."""
    sample = need_sample()
    days = tmp_path / "hdn" / "day_by_date"
    days.mkdir(parents=True)
    for date in vendor.HdnSource(sample).available_dates()[:3]:
        (days / f"{date}_options.csv").symlink_to(sample / "day_by_date" / f"{date}_options.csv")
    manifest = vendor.HdnSource(sample).manifest_path()
    shutil.copy(manifest, days / "manifest.json")
    return vendor.HdnSource(tmp_path / "hdn")


@pytest.mark.parametrize("name", sorted(vendor.SOURCES))
def test_source_contract(name: str, source_copy: vendor.HdnSource, tmp_path: Path) -> None:
    """Every registered source: the calendar, the chain, the digests, the rate curve and the
    import agree with each other."""
    assert name == "hdn"  # the fixture of the one registered source; a new source adds its own
    src = vendor.vendor_source(name, source_copy.location)
    dates = src.available_dates()
    assert dates == sorted(set(dates)) and len(dates) == 3 and all(len(d) == 10 for d in dates)
    assert isinstance(src.missing_dates(), list)
    d = dates[0]
    # the day's own digest is the file's SHA-256; a date without a file is ABSENT
    assert src.day_digest(d) == hashlib.sha256(src.day_file(d).read_bytes()).hexdigest()
    assert src.day_digest("1999-01-04") == vendor.ABSENT
    # the chain is what the importer's steps read
    chain = src.load_chain(d, "SPX")
    assert set(CHAIN_COLUMNS) <= set(chain.columns) and set(CHAIN_ATTRS) <= set(chain.attrs)
    assert chain.attrs["quote_date"] == d and chain.attrs["file"] == src.day_file(d).name
    assert ((chain["bid"] > 0) & (chain["ask"] > 0) & (chain["T"] > 0)).all()
    # the prior rate curve is the one the chain carries
    tenors, zeros = src.prior_rate_curve(d)
    assert np.array_equal(tenors, chain.attrs["rate_tenors"])
    assert np.array_equal(zeros, chain.attrs["rate_zeros"])
    # the import names the source's digest of the day
    cfg, _fit, points, _chain = src.import_day(d, "SPX")
    assert cfg["provenance"]["file_sha256"] == src.day_digest(d)
    assert cfg["provenance"]["quote_date"] == d and len(points.table) > 0
    # the entry digest changes when, and only when, what the import of that date reads changes
    entry = json.dumps(src.import_entry(d), sort_keys=True)
    other = json.dumps(src.import_entry(dates[1]), sort_keys=True)
    path = src.manifest_path()  # type: ignore[attr-defined]
    doc = json.loads(path.read_text())
    doc["rates"][dates[1]] = [r if r is None else r + 0.01 for r in doc["rates"][dates[1]]]
    doc["files"].append({"name": "2099-01-02_options.csv", "sha256": "0" * 64})
    path.write_text(json.dumps(doc))
    fresh = vendor.vendor_source(name, source_copy.location)
    assert json.dumps(fresh.import_entry(d), sort_keys=True) == entry
    assert json.dumps(fresh.import_entry(dates[1]), sort_keys=True) != other
    assert json.dumps(src.import_entry(dates[1]), sort_keys=True) == other  # read once per source
    # an unreadable manifest: loud for the entry, silent (empty) for the missing-day list
    path.write_text("{ torn")
    broken = vendor.vendor_source(name, source_copy.location)
    with pytest.raises(vendor.VendorError, match="unreadable vendor manifest"):
        broken.import_entry(d)
    assert broken.missing_dates() == []
    path.unlink()
    with pytest.raises(vendor.VendorError, match=r"no manifest\.json under"):
        vendor.vendor_source(name, source_copy.location).import_entry(d)


def test_consumers_read_through_the_source(source_copy: vendor.HdnSource, tmp_path: Path) -> None:
    """The history, the raw history and the backtest's inputs give what the source gives."""
    root = source_copy.location
    dates = source_copy.available_dates()
    assert history.hdn_available_dates(root) == dates
    # the backtest: calendar, per-date checksums and the import go through cfg.data_source()
    raw = bt.load_backtest_config(TOY_CONFIG).to_mapping()
    raw["data"]["root"] = str(root)
    raw["dates"] = {"start": dates[0], "end": dates[-1]}
    cfg = bt.BacktestConfig.from_mapping(raw, source=str(TOY_CONFIG))
    assert bt.calendar(cfg) == dates
    index = bt.InputIndex(cfg, dates)
    for d in dates:
        assert index.file_sha(d) == source_copy.day_digest(d)
        want = hashlib.sha256(
            bt._canonical(bt._jsonable(source_copy.import_entry(d))).encode()
        ).hexdigest()
        assert index.manifest_sha(d) == want
        assert bt.source_file(cfg, d) == source_copy.day_file(d)
    # a missing day the vendor lists enters the calendar; a vanished manifest is a DateFailure
    path = source_copy.manifest_path()
    doc = json.loads(path.read_text())
    doc["trading_days_missing"] = [*doc.get("trading_days_missing", []), "2022-07-04"]
    path.write_text(json.dumps(doc))
    raw["dates"] = {"start": "2022-07-01", "end": dates[-1]}
    assert "2022-07-04" in bt.calendar(bt.BacktestConfig.from_mapping(raw, source=str(TOY_CONFIG)))
    assert bt.InputIndex(cfg, dates).file_sha("2022-07-04") == bt.ABSENT
    # the raw history and the history builder accept a source or the HDN directory
    fits_a, spot_a, _ = raw_history.raw_day(root, dates[0])
    fits_b, spot_b, _ = raw_history.raw_day(source_copy, dates[0])
    assert spot_a == spot_b and [f.T for f in fits_a] == [f.T for f in fits_b]
    hist, failures = history.build_history(source_copy, dates[:2], tmp_path / "snaps")
    assert failures == {} and hist.n_dates == 2
    direct, *_ = ih.import_day(root, dates[0], "SPX")
    stored = (tmp_path / "snaps" / f"spx_{dates[0]}.yaml").read_text()
    assert f"file_sha256: {direct['provenance']['file_sha256']}" in stored
    path.unlink()
    with pytest.raises(bt.DateFailure, match=r"no manifest\.json under"):
        bt.InputIndex(cfg, dates).manifest_sha(dates[0])
