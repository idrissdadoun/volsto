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
§18.7).  :data:`SOURCES` is the registry the backtest's ``data.vendor`` chooses from;
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

import pandas as pd
from numpy.typing import NDArray

from volsto.market import import_hdn as ih

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


SOURCES: Final[dict[str, Callable[[Path], VendorSource]]] = {HdnSource.name: HdnSource}
"""Registered vendor sources by name (the backtest's ``data.vendor``)."""


def vendor_source(name: str, location: str | Path) -> VendorSource:
    """A fresh source of vendor ``name`` at ``location``."""
    if name not in SOURCES:
        raise VendorError(f"unknown vendor {name!r}; registered: {sorted(SOURCES)}")
    return SOURCES[name](Path(location))
