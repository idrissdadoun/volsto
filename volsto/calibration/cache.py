"""Content-addressed cache of calibrated leverage functions (SPEC §4.3).

Key = SHA-256 of the canonical JSON of :meth:`~volsto.config.CalibrationSpec.key_payload` plus
the calibration code tag (:data:`~volsto.calibration.particle.CALIBRATION_CODE_TAG`, bumped when
the calibration numerics change).  Each entry directory holds ``leverage.npz`` (with provenance
metadata: git commit, config hash, seeds), ``spec.json`` and optionally ``diagnostics.json``;
``manifest.parquet`` at the cache root has one row per entry.  ``get_or_calibrate`` is the only
entry point studies and viewers use; a viewer passes ``allow_calibrate=False`` and reports what is
missing.

**Every file this module writes goes through** :func:`atomic_write`: a temporary file in the
destination's directory (named ``.<name>.<random>.tmp<suffix>``, :data:`TEMP_MARKER`), created
with the ordinary ``0666 & ~umask`` mode, fsync'd, then ``os.replace``-d over the destination and
the directory fsync'd.  A process killed at any instant therefore leaves the previous file or the
new one, never a torn one, plus at most a leftover temporary, which no reader looks at
(:meth:`LeverageCache.temporaries` lists them).  ``leverage.npz`` is the entry's commit point: it
is written after ``spec.json`` and before ``diagnostics.json`` and the manifest row, and
:meth:`LeverageCache.has` also rejects a ``leverage.npz`` that is not a complete zip archive (a
torn file written before writes were atomic).  A kill between the leverage and the manifest row
leaves a complete entry whose row is missing (:meth:`LeverageCache.unlisted_keys`); a kill before
``diagnostics.json`` leaves an entry without a report, which ``volsto-precompute --resume
--diagnostics`` refreshes.  Checked by ``tests/test_cache_concurrency.py``.
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import fcntl
import hashlib
import json
import logging
import os
import secrets
import subprocess
import threading
import zipfile
from collections import OrderedDict
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import volsto
from volsto.calibration.diagnostics import CalibrationReport, reprice_surface
from volsto.calibration.particle import CALIBRATION_CODE_TAG, calibrate_leverage
from volsto.config import (
    CalibrationSpec,
    CurveConfig,
    MarketConfig,
    SimConfig,
    from_mapping,
    to_mapping,
)
from volsto.market.curves import ForwardCurve
from volsto.market.surface import ImpliedSurface, perturbed_surface, surface_from_config
from volsto.market.varswap import ForwardVarianceCurve, xi0_curve
from volsto.models.bergomi import BergomiSV
from volsto.models.leverage import LeverageFunction
from volsto.models.lsv import LSV

log = logging.getLogger(__name__)


#: The cache manifest: one row per entry, rewritten atomically under :data:`MANIFEST_LOCK_NAME`.
MANIFEST_NAME = "manifest.parquet"
#: Advisory-lock file (POSIX ``flock``) serialising manifest writers; empty, never removed.
MANIFEST_LOCK_NAME = "manifest.parquet.lock"
#: The entry's commit point (written last of the entry's own files, before the manifest row).
LEVERAGE_NAME = "leverage.npz"
#: The entry's full spec (:func:`~volsto.config.to_mapping`), for humans; nothing reads it back.
SPEC_NAME = "spec.json"
#: The entry's calibration report (:class:`CalibrationReport`), optional.
DIAGNOSTICS_NAME = "diagnostics.json"
#: Every temporary file :func:`atomic_write` creates is named ``.<name>.<16 hex>.tmp<suffix>``:
#: a leading dot and this marker, so ``find -name '.*.tmp*'`` / ``rsync --exclude '.*.tmp*'``
#: match all of them and none of the cache's real files.  The destination's suffix is kept at the
#: end because ``numpy.savez_compressed`` appends ``.npz`` to a path without it.
TEMP_MARKER = ".tmp"
#: Glob matching every leftover temporary (see :data:`TEMP_MARKER`).
TEMP_GLOB = ".*" + TEMP_MARKER + "*"
#: Mode passed to ``os.open`` for a temporary: the process umask applies, exactly as for a plain
#: ``open(path, "w")`` (``tempfile.mkstemp`` would force 0600).
TEMP_FILE_MODE = 0o666


def _temp_path(dest: Path) -> tuple[int, Path]:
    """Create (``O_EXCL``) and open a fresh temporary beside ``dest``; return ``(fd, path)``."""
    while True:
        tmp = dest.with_name(f".{dest.name}.{secrets.token_hex(8)}{TEMP_MARKER}{dest.suffix}")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, TEMP_FILE_MODE)
        except FileExistsError:  # 64 random bits: practically never
            continue
        return fd, tmp


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write(dest: str | Path, write: Callable[[Path], object]) -> Path:
    """Publish ``dest`` atomically: ``write(tmp)`` fills a temporary beside it (by path), which is
    then fsync'd and renamed over ``dest``.  On any exception (``KeyboardInterrupt`` included)
    the temporary is removed and ``dest`` is untouched; a SIGKILL leaves ``dest`` untouched plus
    the temporary.  The one implementation of the cache's write rule (module docstring)."""
    path = Path(dest)
    fd, tmp = _temp_path(path)
    try:
        os.close(fd)
        write(tmp)
        with open(tmp, "rb+") as fh:
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    _fsync_dir(path.parent)
    return path


def _is_complete_npz(path: Path) -> bool:
    """``path`` exists and is a complete zip archive: the central directory opens and every
    member's CRC checks (``ZipFile.testzip``).  ``zipfile.is_zipfile`` alone only searches the
    file's last ~65 KB for the end-of-central-directory signature, so a file cut short by a kill
    whose remaining bytes happen to contain that signature would pass it (reproduced by the M10
    verification with an incompressible leverage array).  The CRC pass reads the whole file —
    about 10-20 ms for an 18 MB leverage entry."""
    try:
        if not path.is_file():
            return False
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except (OSError, zipfile.BadZipFile, EOFError, ValueError):
        return False


def has_complete_leverage(root: str | Path, key: str) -> bool:
    """Whether the cache under ``root`` holds a complete ``leverage.npz`` for ``key`` — the one
    completeness test, shared by :meth:`LeverageCache.has_key` and the viewers' read API.  A read,
    with no side effect on a missing root.  Leftover temporaries never count; a torn
    ``leverage.npz`` (a pre-atomic write cut by a kill) is a miss, logged, so a ``--resume``
    recalibrates it instead of loading it."""
    path = Path(root) / key / LEVERAGE_NAME
    if not path.exists():
        return False
    if _is_complete_npz(path):
        return True
    log.warning("leverage cache entry %s: %s is not a complete archive; a miss", key[:12], path)
    return False


#: Code tags whose entries were written before ``ParticleConfig.estimator`` was part of every
#: record: the field did not exist up to ``m6``, and under ``m6`` it was left out of ``spec.json``
#: while unset.  An entry under one of these tags that names no estimator was computed by the
#: sorted estimator, the only one there was (:func:`record_estimator`).
PRE_ESTIMATOR_CODE_TAGS: frozenset[str] = frozenset({"m3.2", "m4b", "m6"})


def record_estimator(particle: Mapping[str, Any], code_tag: str | None, *, where: str = "") -> str:
    """The regression estimator a stored entry was computed with — the one rule for reading it.

    The stored ``particle`` mapping names it (every entry since code tag ``k5``, and an ``m6``
    entry written with the switch set): that name.  It names none and the entry's ``code_tag``
    is in :data:`PRE_ESTIMATOR_CODE_TAGS`: ``"sorted"``.  Anything else is refused — a record
    under a tag that knows the field must carry it, and the config default (``"binned"``) is what
    a *new* calibration uses, not what an old one did."""
    name = particle.get("estimator")
    if name is not None:
        return str(name)
    if code_tag in PRE_ESTIMATOR_CODE_TAGS:
        return "sorted"
    raise ValueError(
        f"{where or 'stored spec'}: no estimator in the record and code tag {code_tag!r} is not "
        f"one of {sorted(PRE_ESTIMATOR_CODE_TAGS)}, whose entries are the sorted estimator's"
    )


def stored_code_tag(leverage_path: str | Path) -> str | None:
    """The ``code_tag`` in the metadata of a stored ``leverage.npz`` (``None`` when it has none);
    only the metadata member of the archive is read."""
    with np.load(leverage_path, allow_pickle=False) as z:
        tag = json.loads(str(z["metadata"])).get("code_tag")
    return None if tag is None else str(tag)


class CacheMissError(KeyError):
    """The requested calibration is not in the cache."""


def code_version() -> str:
    """Short git commit of the working tree when available, else the package version."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(volsto.__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return f"volsto-{volsto.__version__}"


def spec_key(spec: CalibrationSpec, code_tag: str = CALIBRATION_CODE_TAG) -> str:
    payload = {"spec": spec.key_payload(), "code_tag": code_tag}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()


#: Largest number of ξ₀ curves :func:`build_market` keeps (least recently used evicted first).
#: A ladders attribution date touches about 62 states, a backtest keeps two dates alive.
XI0_MEMO_SIZE = 256


class _Xi0Memo:
    """The process-local, size-bounded, exact memo of the ξ₀ strip of :func:`build_market`.

    **What it keys on.**  The strip ``xi0_curve(surface, t_max)`` is a deterministic function of
    the surface object, which :func:`build_market` builds from ``spec.market`` (spot and the
    two curves: the forward and the discount curve), ``spec.surface`` and
    ``spec.perturbation`` alone, and of ``t_max`` (from the surface's ``max_maturity`` and
    ``spec.particle.horizon``).  The key is the SHA-256 of the canonical JSON of exactly those
    four items (:func:`volsto.config.to_mapping`; floats by ``repr``, which round-trips), so a
    hit returns the curve the same computation would return, bit for bit.  The model
    parameters, the simulation schedule and scheme, the particle settings other than the
    horizon and the local-vol grid do not enter the strip and are not in the key: a parameter
    or particle-count bump reuses its state's strip.  A payload that is not plain JSON data is
    not memoised (computed every time), and an ``int`` against a ``float`` spelling of the same
    number is a miss, never a wrong hit.  The market and surface numbers are made Python floats
    first (:func:`canonical_market_surface`, also applied by :func:`build_market` before it
    computes anything), so a ``numpy.float32`` config and its float64 twin — equal under
    :func:`spec_key` — are one computation and one entry; a perturbation payload with a
    non-string mapping key or a non-float64 float is not memoised.

    **Why it is safe to share.**  :class:`~volsto.market.varswap.ForwardVarianceCurve` has no
    mutator and nothing in ``volsto`` writes its arrays (``BergomiSV.bump`` builds a new curve).
    Cache keys are untouched (:func:`spec_key` does not read the memo).  Checked by
    ``tests/test_backtest.py::test_xi0_memo_is_exact_and_keys_are_unchanged`` (memo on / off on
    a perturbed eSSVI state, bit for bit; recorded keys unchanged; LRU bound)."""

    def __init__(self, size: int) -> None:
        self.size = int(size)
        self.enabled = True
        self.hits = 0
        self.misses = 0
        self._curves: OrderedDict[str, ForwardVarianceCurve] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def key(spec: CalibrationSpec, t_max: float) -> str | None:
        """The memo key of ``spec``'s strip, or ``None`` (not memoised) when the perturbation
        payload is not strict plain data (:func:`_plain_payload`).  The market and the surface
        enter through :func:`canonical_market_surface` (Python floats), which is also what
        :func:`build_market` computes from, so equal keys mean equal computations."""
        spec = canonical_market_surface(spec)
        pert = None if spec.perturbation is None else _plain_payload(spec.perturbation)
        if spec.perturbation is not None and pert is _NOT_PLAIN:
            return None
        payload = {
            "market": to_mapping(spec.market),
            "surface": to_mapping(spec.surface),
            "perturbation": pert,
            "t_max": float(t_max),
        }
        try:
            blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        except (TypeError, ValueError):
            return None
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(
        self, spec: CalibrationSpec, surface: ImpliedSurface, t_max: float
    ) -> ForwardVarianceCurve:
        key = self.key(spec, t_max) if self.enabled and self.size > 0 else None
        if key is not None:
            with self._lock:
                hit = self._curves.get(key)
                if hit is not None:
                    self._curves.move_to_end(key)
                    self.hits += 1
                    return hit
        curve = xi0_curve(surface, t_max)
        if key is not None:
            with self._lock:
                self.misses += 1
                self._curves[key] = curve
                self._curves.move_to_end(key)
                while len(self._curves) > self.size:
                    self._curves.popitem(last=False)
        return curve

    def clear(self) -> None:
        with self._lock:
            self._curves.clear()
            self.hits = self.misses = 0

    def info(self) -> dict[str, int | bool]:
        with self._lock:
            return {
                "enabled": self.enabled,
                "size": self.size,
                "entries": len(self._curves),
                "hits": self.hits,
                "misses": self.misses,
            }


_NOT_PLAIN: Any = object()


def _plain_payload(obj: Any) -> Any:
    """``obj`` as strict plain data for a memo key, or :data:`_NOT_PLAIN`: dataclasses by their
    fields, mappings with **string** keys only (``{1: …}`` and ``{"1": …}`` would otherwise hash
    alike), lists and tuples, ``str`` / ``bool`` / ``int`` / ``None``, and floats that are
    float64 (a Python ``float`` or ``numpy.float64``) — any other number type (``numpy.float32``
    and friends, whose arithmetic differs) is not plain."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        out: dict[str, Any] = {}
        for f in dataclasses.fields(obj):
            v = _plain_payload(getattr(obj, f.name))
            if v is _NOT_PLAIN:
                return _NOT_PLAIN
            out[f.name] = v
        return out
    if isinstance(obj, Mapping):
        items: dict[str, Any] = {}
        for k, v in obj.items():
            if not isinstance(k, str):
                return _NOT_PLAIN
            pv = _plain_payload(v)
            if pv is _NOT_PLAIN:
                return _NOT_PLAIN
            items[k] = pv
        return items
    if isinstance(obj, (list, tuple)):
        seq = [_plain_payload(v) for v in obj]
        return _NOT_PLAIN if any(v is _NOT_PLAIN for v in seq) else seq
    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (float, np.float64)):
        return float(obj)
    if type(obj) is int:  # a Python int (numpy integers are not plain)
        return int(obj)
    return _NOT_PLAIN


def canonical_market_surface(spec: CalibrationSpec) -> CalibrationSpec:
    """``spec`` with every number of its market and surface configs as a Python float (a
    ``numpy.float32`` or an ``int`` would otherwise change the surface's arithmetic while
    :func:`spec_key` — which reads them through ``to_mapping`` — does not see the difference).
    Configs read from YAML are unchanged value for value, so their computations are unchanged."""
    s, m = spec.surface, spec.market

    def floats(xs: Any) -> tuple[float, ...]:
        return tuple(float(x) for x in xs)

    surface = dataclasses.replace(
        s,
        atm_maturities=floats(s.atm_maturities),
        atm_vols=floats(s.atm_vols),
        rho=float(s.rho),
        eta=float(s.eta),
        gamma=float(s.gamma),
        max_maturity=float(s.max_maturity),
        rhos=None if s.rhos is None else floats(s.rhos),
    )
    market = MarketConfig(
        float(m.spot),
        CurveConfig(floats(m.rate_curve.times), floats(m.rate_curve.rates)),
        CurveConfig(floats(m.dividend_curve.times), floats(m.dividend_curve.rates)),
    )
    return dataclasses.replace(spec, surface=surface, market=market)


XI0_MEMO = _Xi0Memo(XI0_MEMO_SIZE)
"""The ξ₀ memo of :func:`build_market` (``XI0_MEMO.enabled = False`` switches it off;
``XI0_MEMO.info()`` / ``clear()``)."""


def build_market(spec: CalibrationSpec) -> tuple[ForwardCurve, ImpliedSurface, BergomiSV]:
    """Forward curve, (possibly perturbed) SSVI or eSSVI surface
    (:func:`~volsto.market.surface.surface_from_config`) and pure SV kernel (ξ₀ from the
    variance-swap strip of that surface, reused from :data:`XI0_MEMO` when the same market,
    surface, perturbation and strip horizon were stripped before in this process)."""
    spec = canonical_market_surface(spec)
    fc = ForwardCurve.from_config(spec.market)
    base = surface_from_config(spec.surface, fc, fc.rate_curve)
    surface = perturbed_surface(spec.perturbation, base)
    t_max = min(surface.max_maturity, max(spec.particle.horizon + 1.0, 5.0))
    xi0 = XI0_MEMO.get(spec, surface, t_max)
    return fc, surface, BergomiSV(spec.model, xi0, fc)


class LeverageCache:
    """Directory-backed cache; see module docstring."""

    def __init__(self, root: str | Path) -> None:
        #: the cache root is **not** created here: a reader (the viewers' read API, a
        #: ``--dry-run`` projection, a test that asserts nothing was calibrated) must leave a
        #: missing root missing.  The writers create it — :meth:`store` through the entry
        #: directory's ``parents=True`` and :meth:`_record` before the manifest.
        self.root = Path(root)

    # -- addressing ----------------------------------------------------------------------------

    def key(self, spec: CalibrationSpec) -> str:
        return spec_key(spec)

    def entry_dir(self, spec: CalibrationSpec) -> Path:
        return self.root / self.key(spec)

    def has(self, spec: CalibrationSpec) -> bool:
        """A committed entry for ``spec`` (:meth:`has_key`)."""
        return self.has_key(self.key(spec))

    def has_key(self, key: str) -> bool:
        """The entry ``key`` holds a complete ``leverage.npz`` (:func:`has_complete_leverage`)."""
        return has_complete_leverage(self.root, key)

    def temporaries(self) -> list[Path]:
        """Leftover temporaries of interrupted writes (root and entry directories), sorted.
        Harmless; delete them when no writer is running."""
        if not self.root.is_dir():
            return []
        return sorted([*self.root.glob(TEMP_GLOB), *self.root.glob(f"*/{TEMP_GLOB}")])

    def unlisted_keys(self) -> list[str]:
        """Keys with a committed leverage and no manifest row (a kill between the two writes)."""
        if not self.root.is_dir():
            return []
        m = self.manifest()
        listed = set() if m.empty else {str(k) for k in m["key"]}
        return sorted(
            d.name
            for d in self.root.iterdir()
            if d.is_dir() and d.name not in listed and self.has_key(d.name)
        )

    # -- read / write --------------------------------------------------------------------------

    def load(self, spec: CalibrationSpec) -> LeverageFunction:
        p = self.entry_dir(spec) / LEVERAGE_NAME
        if not p.exists():
            raise CacheMissError(self.key(spec))
        return LeverageFunction.load(p)

    def load_spec(self, key: str) -> CalibrationSpec:
        """The spec the entry ``key`` was computed from, as a record of how it was computed.

        ``spec.json`` is read as written, except that an entry under a code tag of
        :data:`PRE_ESTIMATOR_CODE_TAGS` that names no estimator reads ``estimator="sorted"``
        (:func:`record_estimator`): loading its mapping as a config would give the current
        default, ``"binned"``, which is not how it was computed.  Nothing is rewritten.  The key
        of such an entry is not reproduced by ``spec_key`` of the record under any tag (its
        payload had no such field): the key of an entry is its directory name."""
        d = self.root / key
        if not self.has_key(key):
            raise CacheMissError(key)
        path = d / SPEC_NAME
        if not path.is_file():
            raise ValueError(f"{path}: the entry has no stored spec")
        data = json.loads(path.read_text())
        particle = dict(data.get("particle") or {})
        particle["estimator"] = record_estimator(
            particle, stored_code_tag(d / LEVERAGE_NAME), where=str(path)
        )
        return from_mapping(CalibrationSpec, {**data, "particle": particle}, path=str(path))

    def load_report(self, spec: CalibrationSpec) -> CalibrationReport | None:
        p = self.entry_dir(spec) / DIAGNOSTICS_NAME
        return CalibrationReport.load(p) if p.exists() else None

    def write_report(self, spec: CalibrationSpec, report: CalibrationReport) -> Path:
        """Write ``diagnostics.json`` of an entry atomically (the entry must exist)."""
        return self._write_report(self.entry_dir(spec), report)

    @staticmethod
    def _write_report(d: Path, report: CalibrationReport) -> Path:
        text = json.dumps(report.to_dict(), indent=1, default=str)
        return atomic_write(d / DIAGNOSTICS_NAME, lambda tmp: tmp.write_text(text))

    def store(
        self,
        spec: CalibrationSpec,
        leverage: LeverageFunction,
        report: CalibrationReport | None = None,
    ) -> Path:
        key = self.key(spec)
        d = self.root / key
        d.mkdir(parents=True, exist_ok=True)
        leverage.metadata.update(
            {
                "cache_key": key,
                "git_commit": code_version(),
                "code_tag": CALIBRATION_CODE_TAG,
                "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
                "spec": spec.key_payload(),
            }
        )
        spec_text = json.dumps(to_mapping(spec), indent=1, default=str)
        atomic_write(d / SPEC_NAME, lambda tmp: tmp.write_text(spec_text))
        # a report left from an earlier leverage of this key does not describe the new one
        (d / DIAGNOSTICS_NAME).unlink(missing_ok=True)
        atomic_write(d / LEVERAGE_NAME, leverage.save)  # the commit point
        if report is not None:
            self._write_report(d, report)
        self._append_manifest(key, spec, leverage, report)
        return d

    def _append_manifest(
        self,
        key: str,
        spec: CalibrationSpec,
        leverage: LeverageFunction,
        report: CalibrationReport | None,
    ) -> None:
        m = spec.model
        row: dict[str, Any] = {
            "key": key,
            "created_utc": leverage.metadata.get("created_utc"),
            "git_commit": leverage.metadata.get("git_commit"),
            "code_tag": CALIBRATION_CODE_TAG,
            "horizon": spec.particle.horizon,
            "n_particles": spec.particle.n_particles,
            "seed": spec.particle.seed,
            "nu": m.nu,
            "theta": m.theta,
            "k1": m.k1,
            "k2": m.k2,
            "rho12": m.rho12,
            "rho_SX1": m.rho_SX1,
            "rho_SX2": m.rho_SX2,
            "surface_rho": spec.surface.rho,
            "surface_eta": spec.surface.eta,
            "surface_gamma": spec.surface.gamma,
            "wall_time": float(leverage.metadata.get("wall_time", float("nan"))),
            "max_abs_error_vp": report.max_abs_error() if report is not None else None,
        }
        # Concurrent writers (a ``--workers N`` pool, or N ``--shard i/n`` processes sharing one
        # cache) serialise the read-modify-write on an advisory lock held on a sibling file, and
        # publish the new manifest by an atomic rename, so a reader never sees a torn parquet
        # and no writer's row is lost.  The on-disk format is unchanged.
        row_df = pd.DataFrame([row])
        self._update_manifest(
            lambda old: (
                row_df
                if old.empty
                else pd.concat([old[old["key"] != key], row_df], ignore_index=True)
            )
        )

    def _update_manifest(self, update: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        """``update(current manifest)`` under the lock, published by :func:`atomic_write`."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / MANIFEST_NAME
        with open(self.root / MANIFEST_LOCK_NAME, "a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                old = pd.read_parquet(path) if path.exists() else pd.DataFrame({"key": []})
                new = update(old)
                atomic_write(path, lambda tmp: new.to_parquet(tmp, index=False))
                return new
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def merge_manifest(self, rows: pd.DataFrame) -> tuple[int, int]:
        """Add the rows of another copy of this cache's manifest whose key this one lacks (the
        runbook's merge of a VM's manifest), under the lock and atomically.  Returns
        ``(rows before, rows added)``."""
        counts: list[int] = []

        def _merge(old: pd.DataFrame) -> pd.DataFrame:
            add = rows[~rows["key"].isin(old["key"])]
            counts.extend([len(old), len(add)])
            if old.empty:
                return add.reset_index(drop=True)
            return pd.concat([old, add], ignore_index=True)

        self._update_manifest(_merge)
        return counts[0], counts[1]

    def manifest(self) -> pd.DataFrame:
        path = self.root / MANIFEST_NAME
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    # -- entry point ---------------------------------------------------------------------------

    def get_or_calibrate(
        self,
        spec: CalibrationSpec,
        *,
        allow_calibrate: bool = True,
        run_diagnostics: bool = False,
        diagnostics_sim: SimConfig | None = None,
    ) -> tuple[LSV, CalibrationReport | None]:
        """Return the LSV model for ``spec``, calibrating and storing on a miss.

        Viewers call with ``allow_calibrate=False`` (calibration never runs silently there) and
        get :class:`CacheMissError` naming the key.
        """
        _, surface, kernel = build_market(spec)
        if self.has(spec):
            log.info("leverage cache hit %s", self.key(spec)[:12])
            return LSV(kernel, self.load(spec)), self.load_report(spec)
        if not allow_calibrate:
            raise CacheMissError(f"no calibrated leverage for key {self.key(spec)}")
        log.info("leverage cache miss %s: calibrating", self.key(spec)[:12])
        result = calibrate_leverage(
            surface, kernel, spec.particle, spec.sim, local_vol_cfg=spec.local_vol
        )
        model = LSV(kernel, result.leverage)
        report = None
        if run_diagnostics:
            report = reprice_surface(model, surface, diagnostics_sim or spec.sim)
        self.store(spec, result.leverage, report)
        return model, report


# --------------------------------------------------------------------------------------------
# code-tag guard (owner request before M4)
# --------------------------------------------------------------------------------------------

#: Modules whose source defines the calibration numerics, the normal draws the particles are
#: stepped with (a calibrated leverage is a function of them) and the shared stepping routine.
GUARDED_MODULES: tuple[str, ...] = (
    "volsto/calibration/particle.py",
    "volsto/calibration/binned.py",
    "volsto/engine/rng.py",
    "volsto/models/leverage.py",
    "volsto/models/lsv.py",
    "volsto/models/bergomi.py",
    "volsto/models/localvol.py",
)
GUARD_FILE = Path(__file__).resolve().parent / "code_tag_guard.json"


def source_hash() -> str:
    """SHA-256 of the concatenated source of :data:`GUARDED_MODULES` (line endings normalised)."""
    root = Path(volsto.__file__).resolve().parents[1]
    h = hashlib.sha256()
    for rel in GUARDED_MODULES:
        text = (root / rel).read_text(encoding="utf-8").replace("\r\n", "\n")
        h.update(rel.encode())
        h.update(b"\0")
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def read_guard() -> dict[str, str]:
    """Stored ``{code_tag: source_hash}``; a missing file is an empty mapping."""
    if GUARD_FILE.exists():
        data = json.loads(GUARD_FILE.read_text())
        return {str(k): str(v) for k, v in data.items()}
    return {}


def write_guard() -> Path:
    """Record the current source hash under the current code tag (run after bumping the tag)."""
    guard = read_guard()
    guard[CALIBRATION_CODE_TAG] = source_hash()
    GUARD_FILE.write_text(json.dumps(guard, indent=1, sort_keys=True) + "\n")
    return GUARD_FILE


def check_guard() -> None:
    """Raise if the guarded sources changed without bumping :data:`CALIBRATION_CODE_TAG`.

    ``tests/test_lsv.py::test_calibration_code_tag_guard`` calls this.  To accept a change:
    bump ``CALIBRATION_CODE_TAG`` in ``volsto/calibration/particle.py`` and run
    ``python -c "from volsto.calibration.cache import write_guard; write_guard()"``.
    """
    guard = read_guard()
    current = source_hash()
    stored = guard.get(CALIBRATION_CODE_TAG)
    if stored is None:
        raise AssertionError(
            f"no stored source hash for CALIBRATION_CODE_TAG={CALIBRATION_CODE_TAG!r}; "
            "run write_guard() after bumping the tag"
        )
    if stored != current:
        raise AssertionError(
            "calibration / stepping sources changed but CALIBRATION_CODE_TAG "
            f"({CALIBRATION_CODE_TAG!r}) was not bumped: stored {stored[:12]}…, "
            f"current {current[:12]}… — bump the tag and run write_guard()"
        )
