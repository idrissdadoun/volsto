"""Vendor sources: the one place that knows a vendor's files (SPEC §18.9, M11 Part 5).

**Invariant.**  Nothing outside a vendor source knows where a vendor's days live, how a day
becomes a chain, what its checksums are or where its prior rate curve comes from.  The
backtest (``volsto/studies/backtest.py``), the surface history
(``volsto/calibration/history.py``) and the raw history
(``volsto/calibration/raw_history.py``) each hard-coded the HistoricalData.net layout — the
third occurrence — and now read through a :class:`VendorSource`:

* the **calendar** — :meth:`~VendorSource.available_dates` (the days it holds) and
  :meth:`~VendorSource.missing_dates` (trading days the vendor itself lists as absent);
* **day loading** — :meth:`~VendorSource.load_chain` (the canonical chain of
  the importer's loader: the frame its vendor-independent steps read) and
  :meth:`~VendorSource.import_day` (the whole import of a date into a snapshot document);
* the **checksums** — :meth:`~VendorSource.day_digest` (the SHA-256 of the day's own file,
  :data:`ABSENT` when there is none) and :meth:`~VendorSource.import_entry` (whatever else the
  import of that date reads, as a JSON-able value the caller hashes);
* the **rate curve** — :meth:`~VendorSource.prior_rate_curve` (the vendor's prior curve of a
  date: reported beside the option-implied funding curve, never the discounting curve).

:class:`HdnSource` is HistoricalData.net (a directory with ``day_by_date/<date>_options.csv``
and a ``manifest.json``); it wraps :mod:`volsto.market.import_hdn` unchanged, so every digest,
chain and snapshot is what the hard-coded sites produced (proven on the same machine, SPEC
§18.9).  :class:`OratsSource` is ORATS, read from the Parquet store through
:mod:`volsto.market.import_orats`.  :data:`SOURCES` is the registry the backtest's
``data.vendor`` chooses from;
:func:`vendor_source` builds one.  A source is cheap and reads lazily; it caches the vendor's
manifest for its own lifetime, so a caller that must see a changed manifest builds a new one.

``tests/test_vendor_source.py`` walks ``volsto/`` and ``scripts/`` for the vendor's layout and
loaders outside the vendor modules, and runs one contract test over every registered source.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.data import raw as raw_layer
from volsto.data import store as store_layer
from volsto.data.roots import DataError, DataRoots
from volsto.market import import_hdn as ih
from volsto.market import import_orats

ABSENT: Final = "absent"
"""The digest of a calendar date whose day file does not exist."""


class VendorError(RuntimeError):
    """A vendor input that is missing or unreadable; the message names it."""


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class VendorSource(ABC):
    """One vendor's data at one location (module docstring)."""

    name: ClassVar[str]

    @property
    @abstractmethod
    def location(self) -> Path:
        """Where the data lives (for messages)."""

    @abstractmethod
    def available_dates(self) -> list[str]:
        """ISO trade dates the source holds, ascending."""

    @abstractmethod
    def missing_dates(self) -> list[str]:
        """Trading days the vendor lists as absent (an expected day without a file must not
        silently shift a fixing grid).  Empty when the vendor says nothing — never raises."""

    @abstractmethod
    def day_file(self, date: str) -> Path:
        """The file the chain of ``date`` is read from (it need not exist)."""

    def day_digest(self, date: str) -> str:
        """SHA-256 of :meth:`day_file`, :data:`ABSENT` when it does not exist."""
        src = self.day_file(date)
        return file_sha256(src) if src.is_file() else ABSENT

    @abstractmethod
    def import_entry(self, date: str) -> Any:
        """What the import of ``date`` reads besides its day file, as a JSON-able value (the
        caller hashes it: a change makes the date stale).  Raises :class:`VendorError` when it
        cannot be read."""

    @abstractmethod
    def load_chain(self, date: str, underlying: str = "SPX") -> pd.DataFrame:
        """The canonical chain of ``date`` (the frame the importer's steps 2–5 read)."""

    @abstractmethod
    def prior_rate_curve(self, date: str) -> tuple[NDArray[Any], NDArray[Any]]:
        """``(tenors, zero rates)`` of the vendor's prior curve for ``date``."""

    @abstractmethod
    def import_day(
        self,
        date: str,
        underlying: str = "SPX",
        *,
        filters: ih.HdnFilters | None = None,
        essvi: bool = True,
        calendar_repair: ih.CalendarRepairConfig | None = ih.DEFAULT_CALENDAR_REPAIR,
    ) -> tuple[dict[str, Any], ih.SSVIFit, ih.SurfacePoints, pd.DataFrame]:
        """The whole import of ``date``: ``(snapshot document, fit, points, chain)``."""


class HdnSource(VendorSource):
    """HistoricalData.net: ``<root>/day_by_date/<date>_options.csv`` and a ``manifest.json``
    (under ``day_by_date/`` or at the root — the importer's search order) holding the Treasury
    curve of each date, the files' checksums and the trading days the vendor lacks."""

    name: ClassVar[str] = "hdn"
    DAY_DIR: ClassVar[str] = "day_by_date"
    DAY_SUFFIX: ClassVar[str] = "_options.csv"
    MANIFEST: ClassVar[str] = "manifest.json"

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self._manifest: dict[str, Any] | None = None

    @property
    def location(self) -> Path:
        return self.root

    def available_dates(self) -> list[str]:
        files = sorted(self.root.joinpath(self.DAY_DIR).glob(f"*{self.DAY_SUFFIX}"))
        return [f.name[:10] for f in files]

    def manifest_path(self) -> Path:
        for cand in (self.root / self.DAY_DIR / self.MANIFEST, self.root / self.MANIFEST):
            if cand.is_file():
                return cand
        raise VendorError(f"no manifest.json under {self.root}")

    def manifest(self) -> dict[str, Any]:
        """The vendor's manifest, read once per source."""
        if self._manifest is None:
            try:
                data = json.loads(self.manifest_path().read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise VendorError(f"unreadable vendor manifest: {exc}") from exc
            self._manifest = dict(data) if isinstance(data, dict) else {}
        return self._manifest

    def missing_dates(self) -> list[str]:
        try:
            listed = json.loads(self.manifest_path().read_text(encoding="utf-8")).get(
                "trading_days_missing", []
            )
        except (VendorError, OSError, ValueError, AttributeError):
            return []
        return [str(d) for d in listed if isinstance(d, str)]

    def day_file(self, date: str) -> Path:
        return self.root / self.DAY_DIR / f"{date}{self.DAY_SUFFIX}"

    def import_entry(self, date: str) -> Any:
        """What the importer reads of the manifest for ``date``: the product, the date's rate
        curve and the ``files`` entry of its day file (a manifest that gains a new day leaves
        older dates valid)."""
        m = self.manifest()
        name = self.day_file(date).name
        return {
            "product": m.get("product"),
            "rates": (m.get("rates") or {}).get(date),
            "file": next((f for f in m.get("files", []) if f.get("name") == name), None),
        }

    def load_chain(self, date: str, underlying: str = "SPX") -> pd.DataFrame:
        return ih.load_day(self.day_file(date), underlying, manifest=self.manifest())

    def prior_rate_curve(self, date: str) -> tuple[NDArray[Any], NDArray[Any]]:
        return ih.rate_curve(self.manifest(), date)

    def import_day(
        self,
        date: str,
        underlying: str = "SPX",
        *,
        filters: ih.HdnFilters | None = None,
        essvi: bool = True,
        calendar_repair: ih.CalendarRepairConfig | None = ih.DEFAULT_CALENDAR_REPAIR,
    ) -> tuple[dict[str, Any], ih.SSVIFit, ih.SurfacePoints, pd.DataFrame]:
        return ih.import_day(
            self.root,
            date,
            underlying,
            filters=filters,
            essvi=essvi,
            calendar_repair=calendar_repair,
        )


class OratsSource(VendorSource):
    """ORATS, read from the Parquet store (:mod:`volsto.data.store`) through the importer of
    :mod:`volsto.market.import_orats`.  ``location`` is the store *root* (what
    ``VOLSTO_DATA_STORE`` names); the vendor's files are under ``<root>/orats``.

    * calendar — the store manifest's dates; the missing days are those ``volsto-data
      verify-raw`` found against the trading calendar (the raw manifest's ``findings.missing``),
      none when there is no raw manifest;
    * day file — the store's Parquet file of the date; the **day digest is the raw zip's
      sha256** the store manifest binds the file to (what the importer writes as
      ``file_sha256``), :data:`ABSENT` when the Parquet file is not on disk;
    * import entry — the store manifest's entry of the date (raw file, raw sha256, schema
      version, rows, Parquet sha256, layout version) and the official close the import reads
      for each index underlying (:data:`volsto.market.import_hdn.INDEX_ROOTS`): a re-converted
      day, another schema version or a corrected close makes the date stale;
    * rate curve — the file's ``iRate`` at the slices of the index chain (a prior: SPEC §18.6).
    """

    name: ClassVar[str] = "orats"

    def __init__(self, root: str | Path, *, history_dir: str | Path | None = None) -> None:
        self.root = Path(root)
        self.history_dir = None if history_dir is None else Path(history_dir)
        self._manifest: dict[str, Any] | None = None

    @property
    def location(self) -> Path:
        return self.root

    @property
    def store_dir(self) -> Path:
        return self.root / self.name

    def _closes(self, underlying: str) -> Path | None:
        if self.history_dir is None:
            return None  # the importer's default: data/history/<underlying>.csv
        return self.history_dir / f"{underlying.upper()}.csv"

    def manifest(self) -> dict[str, Any]:
        """The store manifest's ``files``, read once per source."""
        if self._manifest is None:
            try:
                self._manifest = dict(store_layer.read_manifest(self.store_dir)["files"])
            except (OSError, ValueError, KeyError, DataError) as exc:
                raise VendorError(f"unreadable store manifest: {exc}") from exc
        return self._manifest

    def available_dates(self) -> list[str]:
        try:
            return sorted(self.manifest())
        except VendorError:
            return []

    def missing_dates(self) -> list[str]:
        try:
            raw_dir = DataRoots.resolve().raw_dir(self.name)
            man = raw_layer.read_manifest(raw_dir) if raw_dir.is_dir() else None
            listed = [] if man is None else man["findings"].get("missing", [])
        except (OSError, ValueError, KeyError, DataError):
            return []
        return [str(d) for d in listed if isinstance(d, str)]

    def day_file(self, date: str) -> Path:
        return store_layer.day_path(self.store_dir, date)

    def day_digest(self, date: str) -> str:
        entry = self.manifest().get(date) if self.day_file(date).is_file() else None
        return ABSENT if entry is None else str(entry["raw_sha256"])

    def import_entry(self, date: str) -> Any:
        entry = self.manifest().get(date)
        closes: dict[str, float | None] = {}
        for underlying in ih.INDEX_ROOTS:
            try:
                closes[underlying] = import_orats.official_close(
                    underlying, date, self._closes(underlying)
                )
            except DataError:
                closes[underlying] = None
        keys = ("raw_file", "raw_sha256", "schema_version", "rows", "sha256", "layout_version")
        return {
            "store": None if entry is None else {k: entry.get(k) for k in keys},
            "closes": closes,
        }

    def load_chain(self, date: str, underlying: str = "SPX") -> pd.DataFrame:
        return import_orats.load_day(
            date, underlying, store=self.root, closes=self._closes(underlying)
        )

    def prior_rate_curve(self, date: str) -> tuple[NDArray[Any], NDArray[Any]]:
        chain = self.load_chain(date, next(iter(ih.INDEX_ROOTS)))
        return np.asarray(chain.attrs["rate_tenors"]), np.asarray(chain.attrs["rate_zeros"])

    def import_day(
        self,
        date: str,
        underlying: str = "SPX",
        *,
        filters: ih.HdnFilters | None = None,
        essvi: bool = True,
        calendar_repair: ih.CalendarRepairConfig | None = ih.DEFAULT_CALENDAR_REPAIR,
    ) -> tuple[dict[str, Any], ih.SSVIFit, ih.SurfacePoints, pd.DataFrame]:
        return import_orats.import_day(
            date,
            underlying,
            filters=filters,
            essvi=essvi,
            calendar_repair=calendar_repair,
            store=self.root,
            closes=self._closes(underlying),
        )


SOURCES: Final[dict[str, Callable[[Path], VendorSource]]] = {
    HdnSource.name: HdnSource,
    OratsSource.name: OratsSource,
}
"""Registered vendor sources by name (the backtest's ``data.vendor``)."""


def vendor_source(name: str, location: str | Path) -> VendorSource:
    """A fresh source of vendor ``name`` at ``location``."""
    if name not in SOURCES:
        raise VendorError(f"unknown vendor {name!r}; registered: {sorted(SOURCES)}")
    return SOURCES[name](Path(location))
