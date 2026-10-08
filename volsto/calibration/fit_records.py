"""Stored fit records: a marking fit is a record keyed by its inputs (SPEC §13.4; owner's
decision 5 of 2026-10-03, CONTRIBUTING.md "Machine-dependent arithmetic").

**Why.**  The leverage cache key hashes the fitted model parameters exactly, and a marking fit
is reproduced on another machine to 1e-8 only (SPEC §13.3): a leverage calibrated on one
machine was a cache miss after a refit on another.  There is no rounding inside keys.  Instead
the fitted parameters are **data**: the first fit of given inputs writes a record, and every
later run — on any machine that holds the record — reads the parameters back and never fits
again, so the leverage key is the same everywhere.

**Key** (:func:`fit_key`): the SHA-256 of what the fit reads and nothing else — the market and
surface configuration (an eSSVI's pillar ``rhos`` included), the perturbation layer, the
snapshot's stored SABRW fits when step 0 reads them (:func:`stored_sabrw_fit`: the fields the
snapshot holds, **not** the zone edges a loaded fit recomputes — those are machine-dependent in
their last bits and the marking fit never reads them), the resolved fit configuration, the SSR
target — and the fit code tag :data:`FIT_CODE_TAG`.  It is the content of the snapshot the fit
depends on (not the digest of its bytes: provenance text does not move a fit), so the backtest
and any other pipeline fitting the same surface share one record.

**Canonical encoding** (what is hashed; :func:`canonical`).  The inputs are first brought to
plain values by :func:`volsto.config.to_mapping` — a dataclass becomes a mapping of its fields,
a tuple or an array a list, a numpy scalar a Python number — from the *loaded* objects, so a
snapshot's bytes never reach the key: its key order, comments, number formatting (``0.1`` or
``1.0e-1``; ``3`` where the field is a float) and provenance play no part.  The mapping
``{"fit_code_tag": tag, "inputs": inputs}`` is then written as JSON with the keys sorted, the
separators ``,`` and ``:`` and no other whitespace, every float as Python's shortest
round-trip ``repr`` of its IEEE-754 double (two floats have the same text if and only if they
are the same double, so a one-ulp change of any number the fit reads is another key), integers
as integers, ``None`` as ``null``, and the SHA-256 is taken of its UTF-8 bytes.  Checked by
``tests/test_fit_records.py::test_key_ignores_the_bytes_and_sees_every_ulp``.

**Store** (:class:`FitRecords`): ``<root>/<key[:2]>/<key>.json`` — by convention
``<leverage cache>/fits``, so the records travel with the leverages they key.  A record holds the
key, its inputs' digests, the fit code tag, where and when it was written, and the fit's summary
(:func:`fit_summary`: parameters, break-even parameters, status, messages, standard errors, the
per-pillar SSR and skews, objectives).  Records are immutable: :meth:`FitRecords.put` writes
atomically and never replaces an existing record (the first writer wins and its record is
returned), so two machines racing on one cache agree afterwards.

**Reading** (:func:`recorded_marking_fit`): the record when it exists, else the fit — then
recorded.  The result is a :class:`RecordedFit` (parameters, status, messages, summary, and
``source`` = ``"record"`` or ``"fit"``); ``result`` holds the full
:class:`~volsto.calibration.fit_2f.FitResult` only when the fit ran in this process.

**Code tag.**  :data:`FIT_CODE_TAG` is bumped whenever a change moves any marking fit; the
sources in :data:`FIT_GUARDED_MODULES` are hashed against it (``fit_guard.json``,
``tests/test_fit_records.py::test_fit_code_tag_guard``), like the calibration and importer tags.

**Migration** (:func:`migrate_backtest_store`, :func:`migrate_fit_specs`; ``volsto-fit-records
migrate``): fits made before the records existed are already stored — in a backtest store's
``fit.json`` beside its snapshot, in the study fit specs — and are written as records under the
key of their inputs, so that the leverages calibrated from them are found again on any machine.
Nothing is fitted and nothing is calibrated; a record is written only when its leverage key is
re-derived from the stored parameters (reported as a cache hit or miss); existing records are
never replaced.  ``status`` lists a store's records.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as _dt
import hashlib
import json
import logging
import os
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

import volsto
from volsto.calibration.fit_2f import BreakEvenFitConfig, FitResult, load_fit_spec
from volsto.calibration.stability import PARAM_COLUMNS
from volsto.config import BergomiParams, CalibrationSpec, to_mapping
from volsto.market.sabrw import PARAM_NAMES

log = logging.getLogger(__name__)

FIT_CODE_TAG: Final[str] = "2026-10-04"
"""The marking fit's numerics tag, part of every fit key.  Bump it whenever a change moves any
marking fit (then every fit is made again and recorded under the new tag; the old records and
the leverages keyed by their parameters stay on disk, unreachable)."""
FIT_GUARDED_MODULES: Final[tuple[str, ...]] = (
    "volsto/calibration/fit_2f.py",
    "volsto/calibration/targets.py",
    "volsto/analytics/breakeven.py",
    "volsto/analytics/reparam.py",
    "volsto/market/sabrw.py",
)
FIT_GUARD_FILE = Path(__file__).resolve().parent / "fit_guard.json"
RECORD_VERSION: Final[int] = 1
FITS_DIR: Final[str] = "fits"
"""Sub-directory of a leverage cache that holds its fit records."""
MODEL_FIELDS: Final[tuple[str, ...]] = tuple(f.name for f in dataclasses.fields(BergomiParams))


# --------------------------------------------------------------------------------------------
# key
# --------------------------------------------------------------------------------------------


def canonical(payload: Any) -> str:
    """The canonical JSON text of ``payload`` (module docstring, *Canonical encoding*)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


_canonical = canonical


def stored_sabrw_fit(fit: Any) -> dict[str, Any]:
    """One SABRW fit as the snapshot stores it (:func:`volsto.market.import_hdn.sabrw_section`):
    ``T``, ``n``, the seven parameters, the two fit errors, the held and at-bound slopes and the
    flags — and nothing else.  **Not the zones**: a loaded :class:`~volsto.market.sabrw.SabrwFit`
    carries zone edges recomputed at load time (``x_Tu`` by a root solver, ``x_Td`` through
    ``norm.ppf``), which differ in their last bits between machines for one snapshot file
    (measured on Linux: up to 10 ulp on 352 of 458 stored fits), and which the marking fit
    never reads.  The key is made of what the file holds."""
    return {
        "T": float(fit.T),
        "n": int(fit.n),
        "params": {k: float(v) for k, v in zip(PARAM_NAMES, fit.params.as_array(), strict=True)},
        "rms_vp": float(fit.rms_vp),
        "weighted_rms": float(fit.weighted_rms),
        "held": list(fit.held),
        "at_bound": list(fit.at_bound),
        "flags": list(fit.flags),
    }


def fit_inputs(
    spec: CalibrationSpec,
    cfg: BreakEvenFitConfig,
    ssr_target: float,
    sabrw_fits: Sequence[Any] | None,
) -> dict[str, Any]:
    """What a marking fit reads, as a JSON-able mapping: the market and surface of ``spec`` with
    its perturbation layer (the model, particle and scheme settings play no part), the resolved
    fit configuration, the SSR target and — when the configuration's step 0 reads them — the
    snapshot's stored SABRW fits (:func:`stored_sabrw_fit`: what the file holds, never the zones
    recomputed at load time; ``None`` otherwise, whatever the snapshot holds)."""
    reads_fits = cfg.step0 is not None
    if reads_fits and sabrw_fits is None:
        raise ValueError("the fit's step 0 reads the snapshot's SABRW fits: none given")
    return {
        "market": to_mapping(spec.market),
        "surface": to_mapping(spec.surface),
        "perturbation": None if spec.perturbation is None else to_mapping(spec.perturbation),
        "fit": to_mapping(cfg),
        "ssr_target": float(ssr_target),
        "sabrw": [stored_sabrw_fit(f) for f in sabrw_fits or ()] if reads_fits else None,
    }


def fit_key(inputs: Mapping[str, Any], code_tag: str = FIT_CODE_TAG) -> str:
    """The record key of :func:`fit_inputs` under ``code_tag``."""
    return hashlib.sha256(
        _canonical({"inputs": inputs, "fit_code_tag": code_tag}).encode()
    ).hexdigest()


def input_digests(inputs: Mapping[str, Any]) -> dict[str, str]:
    """One SHA-256 per part of the inputs (stored in the record: says which part differs when
    two keys do)."""
    return {
        name: hashlib.sha256(_canonical(value).encode()).hexdigest()[:16]
        for name, value in inputs.items()
    }


# --------------------------------------------------------------------------------------------
# summary and record
# --------------------------------------------------------------------------------------------


def fit_summary(r: FitResult) -> dict[str, Any]:
    """The JSON summary of a marking fit (parameters, status, messages, the per-pillar SSR) —
    what a fit record stores and what the backtest writes to ``fit.json``."""
    p, b, f, s = r.params, r.breakeven, r.first, r.second
    se = {
        "k1": f.k1_se,
        "lambda1": f.lambda1_se,
        "lambda2": f.lambda2_se,
        "omega1": s.stderr.get("omega1", float("nan")),
        "omega2": s.stderr.get("omega2", float("nan")),
        "chi": s.stderr.get("chi", float("nan")),
    }
    tbl = r.table
    return {
        "status": r.status,
        "messages": list(r.messages),
        "notes": [*r.notes, *f.notes, *s.notes],
        "params": {k: float(getattr(p, k)) for k in MODEL_FIELDS},
        "breakeven": {k: float(getattr(b, k)) for k in PARAM_COLUMNS},
        "se": se,
        "pillars": [float(x) for x in tbl["T"]],
        "ssr_first_order": [float(x) for x in tbl["ssr_first_order"]],
        "ssr_target": [float(x) for x in r.targets.ssr_target],
        "skew_market": [float(x) for x in tbl["skew_market"]],
        "skew_naked": [float(x) for x in tbl["skew_naked"]],
        "active": list(f.active),
        "bound_flags": list(s.bound_flags),
        "k1_at_bound": bool(f.k1_at_bound),
        "first_objective": float(f.objective),
        "second_objective": float(s.objective),
        "wall_seconds": float(r.wall_seconds),
    }


@dataclass(frozen=True)
class RecordedFit:
    """A marking fit as a pipeline consumes it: the parameters (the leverage key follows from
    them), the status and messages, and the summary.  ``source`` says whether it was read
    (``"record"``) or fitted in this process (``"fit"``, then ``result`` is the full fit)."""

    key: str
    params: BergomiParams
    status: str
    messages: tuple[str, ...]
    summary: dict[str, Any]
    source: str
    result: FitResult | None = None

    @classmethod
    def from_summary(
        cls, key: str, summary: Mapping[str, Any], source: str, result: FitResult | None = None
    ) -> RecordedFit:
        params = BergomiParams(**{k: float(summary["params"][k]) for k in MODEL_FIELDS})
        return cls(
            key,
            params,
            str(summary["status"]),
            tuple(str(m) for m in summary["messages"]),
            dict(summary),
            source,
            result,
        )


class FitRecords:
    """The record store of one directory (module docstring)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @classmethod
    def of_cache(cls, cache_root: str | Path) -> FitRecords:
        """The records that travel with a leverage cache: ``<cache>/fits``."""
        return cls(Path(cache_root) / FITS_DIR)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        """The stored record of ``key`` (``None`` when there is none); a record that cannot be
        read or names another key raises — it is never treated as absent and refitted over."""
        p = self.path(key)
        try:
            text = p.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        doc: dict[str, Any] = json.loads(text)
        if doc.get("version") != RECORD_VERSION or doc.get("key") != key:
            raise ValueError(
                f"{p}: not the fit record of {key[:16]} (version {doc.get('version')})"
            )
        return doc

    def put(
        self,
        key: str,
        inputs: Mapping[str, Any],
        summary: Mapping[str, Any],
        *,
        origin: str,
        code_tag: str = FIT_CODE_TAG,
    ) -> dict[str, Any]:
        """Write the record of ``key`` unless one exists; returns the stored record (the
        existing one when there is one: the first writer wins).  ``code_tag`` is the numerics
        tag of the fit the record holds: the marking fit's by default, the SVI slice fit's
        (:data:`volsto.market.svi_slices.SVI_FIT_CODE_TAG`) for the records of SPEC §8.7."""
        doc = {
            "version": RECORD_VERSION,
            "key": key,
            "fit_code_tag": code_tag,
            "inputs": input_digests(inputs),
            "origin": origin,
            "host": socket.gethostname(),
            "code_version": volsto.__version__,
            "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            "fit": dict(summary),
        }
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=1, sort_keys=True)
                fh.write("\n")
                fh.flush()
                os.fsync(fh.fileno())
            with contextlib.suppress(FileExistsError):
                os.link(tmp, p)  # atomic, and fails when a record is already there
        finally:
            tmp.unlink(missing_ok=True)
        stored = self.get(key)
        assert stored is not None
        return stored

    def keys(self) -> list[str]:
        if not self.root.is_dir():
            return []
        return sorted(p.stem for p in self.root.glob("??/*.json") if not p.name.startswith("."))


def recorded_marking_fit(
    spec: CalibrationSpec,
    cfg: BreakEvenFitConfig,
    *,
    ssr_target: float,
    sabrw_fits: Sequence[Any] | None,
    records: FitRecords | None,
    fit: Callable[[], FitResult],
    origin: str = "fit",
) -> RecordedFit:
    """The marking fit of these inputs: read from ``records`` when its record exists, else made
    by ``fit()`` (the caller's closure over the surface it built from ``spec``) and recorded.
    ``records=None`` always fits and stores nothing."""
    inputs = fit_inputs(spec, cfg, ssr_target, sabrw_fits)
    key = fit_key(inputs)
    if records is not None:
        doc = records.get(key)
        if doc is not None:
            return RecordedFit.from_summary(key, doc["fit"], "record")
    result = fit()
    summary = fit_summary(result)
    if records is not None:
        stored = records.put(key, inputs, summary, origin=origin)
        if stored["fit"]["params"] != summary["params"]:  # another writer got there first
            return RecordedFit.from_summary(key, stored["fit"], "record")
    return RecordedFit.from_summary(key, summary, "fit", result)


# --------------------------------------------------------------------------------------------
# code tag guard
# --------------------------------------------------------------------------------------------


def fit_source_hash() -> str:
    """SHA-256 of the concatenated source of :data:`FIT_GUARDED_MODULES` (line endings
    normalised) — the construction of the calibration and importer guards."""
    root = Path(volsto.__file__).resolve().parents[1]
    h = hashlib.sha256()
    for rel in FIT_GUARDED_MODULES:
        text = (root / rel).read_text(encoding="utf-8").replace("\r\n", "\n")
        h.update(rel.encode())
        h.update(b"\0")
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def read_fit_guard() -> dict[str, str]:
    if FIT_GUARD_FILE.exists():
        return {str(k): str(v) for k, v in json.loads(FIT_GUARD_FILE.read_text()).items()}
    return {}


def write_fit_guard() -> Path:
    """Record the current source hash under the current tag (after bumping the tag, or after a
    change proven not to move any fit — say which in the commit)."""
    guard = read_fit_guard()
    guard[FIT_CODE_TAG] = fit_source_hash()
    FIT_GUARD_FILE.write_text(json.dumps(guard, indent=1, sort_keys=True) + "\n")
    return FIT_GUARD_FILE


def check_fit_guard() -> None:
    """Raise if the guarded sources changed without a bump of :data:`FIT_CODE_TAG` (or a
    re-recorded hash)."""
    stored = read_fit_guard().get(FIT_CODE_TAG)
    current = fit_source_hash()
    if stored != current:
        raise AssertionError(
            f"fit sources changed but FIT_CODE_TAG ({FIT_CODE_TAG!r}) was not bumped: stored "
            f"{str(stored)[:12]}…, current {current[:12]}… — bump the tag and run "
            "write_fit_guard()"
        )


# --------------------------------------------------------------------------------------------
# migration
# --------------------------------------------------------------------------------------------


@dataclass
class MigrationReport:
    """What a migration found: per source item ``written`` (a new record), ``present`` (a record
    was already there; ``differs`` when its parameters are not the stored fit's), ``skipped``
    with the reason; and for each record whether the leverage keyed by its parameters is in the
    cache."""

    written: list[str] = dataclasses.field(default_factory=list)
    present: list[str] = dataclasses.field(default_factory=list)
    differs: dict[str, str] = dataclasses.field(default_factory=dict)
    skipped: dict[str, str] = dataclasses.field(default_factory=dict)
    leverage_hit: list[str] = dataclasses.field(default_factory=list)
    leverage_miss: list[str] = dataclasses.field(default_factory=list)

    def lines(self) -> list[str]:
        out = [
            f"records written {len(self.written)}, already present {len(self.present)}, "
            f"skipped {len(self.skipped)}; leverages found in the cache "
            f"{len(self.leverage_hit)}, not found {len(self.leverage_miss)}"
        ]
        out += [f"  DIFFERS {k}: {v}" for k, v in self.differs.items()]
        out += [f"  skipped {k}: {v}" for k, v in self.skipped.items()]
        out += [f"  no leverage in the cache for {k}" for k in self.leverage_miss]
        return out


def _record_stored_fit(
    name: str,
    spec: CalibrationSpec,
    cfg: BreakEvenFitConfig,
    ssr_target: float,
    sabrw_fits: Sequence[Any] | None,
    summary: Mapping[str, Any],
    records: FitRecords,
    cache_root: Path,
    rep: MigrationReport,
    origin: str,
    dry_run: bool,
) -> None:
    from volsto.calibration.cache import has_complete_leverage, spec_key

    inputs = fit_inputs(spec, cfg, ssr_target, sabrw_fits)
    key = fit_key(inputs)
    params = BergomiParams(**{k: float(summary["params"][k]) for k in MODEL_FIELDS})
    lev = spec_key(dataclasses.replace(spec, model=params))
    (rep.leverage_hit if has_complete_leverage(cache_root, lev) else rep.leverage_miss).append(
        f"{name} (leverage {lev[:16]})"
    )
    existing = records.get(key)
    if existing is not None:
        rep.present.append(name)
        if existing["fit"]["params"] != dict(summary["params"]):
            rep.differs[name] = (
                f"record {key[:16]} holds other parameters than the stored fit (kept: records "
                "are never replaced)"
            )
        return
    if not dry_run:
        records.put(key, inputs, summary, origin=origin)
    rep.written.append(name)


def migrate_fit_specs(
    spec_dir: Path, cache_root: Path, *, dry_run: bool = False
) -> MigrationReport:
    """One record per study fit spec (``*.yaml`` with ``spec`` and ``fit`` sections,
    :class:`~volsto.calibration.fit_2f.FitSpec`): the entry's stored parameters under the key of
    its inputs.  Entries that are not fit specs are skipped with the reason."""
    from volsto.market.loaders import load_sabrw_fits

    rep = MigrationReport()
    records = FitRecords.of_cache(cache_root)
    repo = Path(volsto.__file__).resolve().parents[1]
    for path in sorted(Path(spec_dir).glob("*.yaml")):
        try:
            fs = load_fit_spec(path)
            cfg = fs.config
            fits = None
            if cfg.step0 is not None:
                if fs.snapshot is None:
                    raise ValueError(
                        "step 0 reads a snapshot's SABRW fits and the entry names none"
                    )
                fits = load_sabrw_fits(repo / fs.snapshot)
                if fits is None:
                    raise ValueError(f"{fs.snapshot} has no sabrw section")
            summary = {
                "status": str(fs.fit["status"]),
                "messages": [str(m) for m in fs.fit.get("messages", [])],
                "params": {k: float(getattr(fs.spec.model, k)) for k in MODEL_FIELDS},
                "breakeven": {k: float(v) for k, v in fs.fit["breakeven"].items()},
                "migrated_from": str(path),
            }
        except Exception as exc:  # not a fit spec, or one whose inputs cannot be re-derived
            rep.skipped[path.name] = f"{type(exc).__name__}: {exc}"
            continue
        _record_stored_fit(
            path.name, fs.spec, cfg, fs.ssr_target, fits, summary, records, Path(cache_root),
            rep, f"migration: fit spec {path.name}", dry_run,
        )  # fmt: skip
    return rep


def migrate_backtest_store(
    config_path: Path,
    store_root: Path,
    snapshots_root: Path,
    cache_root: Path,
    *,
    dry_run: bool = False,
) -> MigrationReport:
    """One record per date of a backtest store whose current attempt holds a ``fit.json``: its
    stored parameters under the key of the date's snapshot, the config's marking fit and SSR
    target.  Reads the store; never writes to it."""
    from volsto.market.loaders import sabrw_fits_from_config, snapshot_spec
    from volsto.studies import backtest as bt

    rep = MigrationReport()
    records = FitRecords.of_cache(cache_root)
    cfg = bt.load_backtest_config(config_path)
    fit_cfg = cfg.fit_config()
    ssr = float(cfg.section("marking")["ssr_target"])
    base = cfg.base_spec()
    for date_dir in sorted((Path(store_root) / "dates").glob("*")):
        date = date_dir.name
        try:
            pointer = json.loads((date_dir / "CURRENT").read_text())
            fit_path = date_dir / "attempts" / pointer["attempt"] / "fit.json"
            summary = json.loads(fit_path.read_text())
            snap = Path(snapshots_root) / f"{cfg.underlying.lower()}_{date}.yaml"
            raw = yaml.safe_load(snap.read_text(encoding="utf-8"))
            spec = snapshot_spec(base, snap)
            fits = sabrw_fits_from_config(raw) if fit_cfg.step0 is not None else None
            if "params" not in summary or "status" not in summary:
                raise ValueError("fit.json holds no fitted parameters (a skipped or failed date)")
            if fit_cfg.step0 is not None and fits is None:
                raise ValueError(
                    "its snapshot has no sabrw section (imported before the importer stored "
                    "the fits): the fit's inputs cannot be re-derived; re-import and recompute "
                    "the date"
                )
            fit_inputs(spec, fit_cfg, ssr, fits)
        except Exception as exc:
            rep.skipped[date] = f"{type(exc).__name__}: {exc}"
            continue
        _record_stored_fit(
            date, spec, fit_cfg, ssr, fits, summary, records, Path(cache_root), rep,
            f"migration: backtest store {store_root} date {date}", dry_run,
        )  # fmt: skip
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="volsto-fit-records",
        description="Stored marking-fit records (the parameters a leverage key is built from)",
    )
    sub = ap.add_subparsers(dest="command", required=True)
    st = sub.add_parser("status", help="list the records of a cache")
    st.add_argument("--cache", required=True)
    mg = sub.add_parser(
        "migrate", help="write records for fits stored before the records existed (never fits)"
    )
    mg.add_argument(
        "--cache", required=True, help="leverage cache root (records go to <cache>/fits)"
    )
    mg.add_argument("--fit-specs", default=None, help="directory of study fit specs (*.yaml)")
    mg.add_argument("--backtest-config", default=None)
    mg.add_argument("--backtest-store", default=None)
    mg.add_argument("--backtest-snapshots", default=None)
    mg.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    records = FitRecords.of_cache(args.cache)
    if args.command == "status":
        keys = records.keys()
        print(f"{len(keys)} fit records in {records.root} (fit code tag {FIT_CODE_TAG})")
        for key in keys:
            doc = records.get(key)
            assert doc is not None
            print(f"  {key[:16]} {doc['fit']['status']:10s} {doc['created_utc']} {doc['origin']}")
        return 0
    reports: list[tuple[str, MigrationReport]] = []
    if args.fit_specs:
        reports.append(
            (
                "fit specs",
                migrate_fit_specs(Path(args.fit_specs), Path(args.cache), dry_run=args.dry_run),
            )
        )
    bt_args = (args.backtest_config, args.backtest_store, args.backtest_snapshots)
    if any(bt_args):
        if not all(bt_args):
            ap.error("--backtest-config, --backtest-store and --backtest-snapshots go together")
        rep_bt = migrate_backtest_store(
            Path(bt_args[0]),
            Path(bt_args[1]),
            Path(bt_args[2]),
            Path(args.cache),
            dry_run=args.dry_run,
        )
        reports.append(("backtest store", rep_bt))
    if not reports:
        ap.error("nothing to migrate: give --fit-specs and/or the three --backtest-* options")
    bad = False
    for name, rep in reports:
        print(f"{name}{' (dry run)' if args.dry_run else ''}:")
        for line in rep.lines():
            print(f"  {line}")
        bad = bad or bool(rep.differs)
    return 1 if bad else 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
