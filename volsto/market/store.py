"""Read API over the vendor Parquet store (SPEC §18.5, M11 Part 3).

Three functions, on :mod:`pyarrow.dataset`, returning the vendor's own columns under their own
names and types (nothing renamed, nothing derived):

* :func:`available_dates` — the ISO trade dates the store holds;
* :func:`load_chain` — one ticker's rows of one trade date, from that day's file (row groups
  pruned on the ticker statistics);
* :func:`load_range` — one ticker's rows from ``start`` to ``end`` (both included), from the
  per-ticker extract (row groups pruned on the trade date).

**Nothing here downloads, converts, extracts or fits**, and nothing is written.  What is
missing raises :class:`StoreMissing`, whose message ends with the exact ``volsto-data`` command
that produces it (also in its ``command`` attribute): ``convert`` for a date whose raw file is
verified, the ``fetch`` / ``verify-raw`` / ``convert`` sequence for one whose raw file is not
there, ``extract`` for a ticker whose extract is absent or was built from another state of the
store, ``verify`` for a file the manifest names but the disk does not hold.  A ticker with no
row on a date the store holds cannot be produced by any command and says so.

The store is found through :class:`volsto.data.roots.DataRoots` (``VOLSTO_DATA_STORE``, else
the documented default); ``store=`` overrides it, and the printed commands then carry
``--store``.  The frame's ``attrs`` carry the provenance the importer records: vendor, ticker,
the store file, and for :func:`load_chain` the raw file, its sha256 and the schema version.
"""

from __future__ import annotations

import datetime as _dt
import shlex
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as pads

from volsto.data import extract as extractmod
from volsto.data import orats
from volsto.data import raw as rawmod
from volsto.data import store as storemod
from volsto.data.roots import DataError, DataRoots

VENDORS: tuple[str, ...] = (orats.VENDOR,)
DateLike = str | _dt.date


class StoreMissing(DataError):  # noqa: N818 — what is missing, with the command that makes it
    """A date, a ticker or an extract the store does not hold.  ``command`` is the
    ``volsto-data`` line that produces it (``None`` when no command can)."""

    def __init__(self, message: str, command: str | None = None) -> None:
        super().__init__(f"{message}: {command}" if command else message)
        self.command = command


def _iso(day: DateLike) -> str:
    if isinstance(day, _dt.datetime):
        return day.date().isoformat()
    if isinstance(day, _dt.date):
        return day.isoformat()
    return _dt.date.fromisoformat(str(day)).isoformat()


def _dirs(vendor: str, store: str | Path | None) -> tuple[Path, Path, str]:
    """``(store dir, raw dir, root flags for a printed command)`` of ``vendor``."""
    if vendor not in VENDORS:
        raise DataError(f"unknown vendor {vendor!r}; known: {list(VENDORS)}")
    roots = DataRoots.resolve(store=store)
    flags = f"--store {shlex.quote(str(roots.store))} " if store is not None else ""
    return roots.store_dir(vendor), roots.raw_dir(vendor), flags


def _columns(columns: Sequence[str] | None, version: int) -> list[str] | None:
    if columns is None:
        return None
    known = orats.schema_columns(version)
    unknown = [c for c in columns if c not in known]
    if unknown:
        raise DataError(f"unknown columns {unknown}; the vendor's columns are {list(known)}")
    return list(columns)


def available_dates(vendor: str, *, store: str | Path | None = None) -> list[str]:
    """ISO trade dates the store holds for ``vendor``, ascending (empty when there is no
    store yet)."""
    store_dir, _, _ = _dirs(vendor, store)
    return storemod.available_dates(store_dir)


def _missing_date(
    vendor: str, date: str, store_dir: Path, raw_dir: Path, flags: str
) -> StoreMissing:
    convert = f"volsto-data {flags}convert --vendor {vendor}"
    try:
        man = rawmod.read_manifest(raw_dir) if raw_dir.is_dir() else None
    except DataError:
        man = None
    in_raw = man is not None and any(e["trade_date"] == date for e in man["files"])
    if in_raw:
        return StoreMissing(
            f"{vendor} {date} is verified in raw but not in the store {store_dir}", convert
        )
    return StoreMissing(
        f"{vendor} {date} is not in the store {store_dir} and has no verified raw file",
        f"volsto-data fetch --vendor {vendor} --profile <aws profile> --bucket <bucket> "
        f"--prefix <prefix> && volsto-data verify-raw --vendor {vendor} && {convert}",
    )


def chain_entry(vendor: str, date: DateLike, *, store: str | Path | None = None) -> dict[str, Any]:
    """The store manifest's entry of ``date`` (raw file and sha256, schema version, rows);
    raises :class:`StoreMissing` like :func:`load_chain`."""
    store_dir, raw_dir, flags = _dirs(vendor, store)
    iso = _iso(date)
    entry = storemod.read_manifest(store_dir)["files"].get(iso)
    if entry is None:
        raise _missing_date(vendor, iso, store_dir, raw_dir, flags)
    if not storemod.day_path(store_dir, iso).is_file():
        raise StoreMissing(
            f"{storemod.day_path(store_dir, iso)} is in the store manifest but not on disk",
            f"volsto-data {flags}verify --vendor {vendor}",
        )
    return dict(entry)


def load_chain(
    vendor: str,
    date: DateLike,
    ticker: str,
    *,
    columns: Sequence[str] | None = None,
    store: str | Path | None = None,
) -> pd.DataFrame:
    """``ticker``'s rows of trade date ``date`` (ISO text or a date), vendor-native columns
    (``columns`` selects some), in the store's order (expirDate, strike, cOpra)."""
    store_dir, _, _ = _dirs(vendor, store)
    iso = _iso(date)
    entry = chain_entry(vendor, iso, store=store)
    path = storemod.day_path(store_dir, iso)
    name = ticker.strip().upper()
    table = pads.dataset(path, format="parquet").to_table(
        columns=_columns(columns, int(entry["schema_version"])),
        filter=pads.field("ticker") == name,
    )
    if table.num_rows == 0:
        raise StoreMissing(
            f"{vendor} {iso} has no row for ticker {name!r} ({entry['rows']:,} rows that day): "
            "the vendor's file of that day does not list it, so no command produces it"
        )
    frame: pd.DataFrame = table.replace_schema_metadata(None).to_pandas()
    frame.attrs.update(
        {
            "vendor": vendor,
            "ticker": name,
            "trade_date": iso,
            "store_file": str(path),
            "raw_file": entry["raw_file"],
            "raw_sha256": entry["raw_sha256"],
            "schema_version": int(entry["schema_version"]),
        }
    )
    return frame


def load_range(
    vendor: str,
    ticker: str,
    start: DateLike,
    end: DateLike,
    columns: Sequence[str] | None = None,
    *,
    store: str | Path | None = None,
) -> pd.DataFrame:
    """``ticker``'s rows on every stored trade date from ``start`` to ``end`` (both included),
    vendor-native columns, in (trade_date, expirDate, strike, cOpra) order, read from the
    per-ticker extract."""
    store_dir, _, flags = _dirs(vendor, store)
    name = ticker.strip().upper()
    lo, hi = _dt.date.fromisoformat(_iso(start)), _dt.date.fromisoformat(_iso(end))
    if lo > hi:
        raise DataError(f"start {lo} is after end {hi}")
    command = f"volsto-data {flags}extract --vendor {vendor} --tickers {name}"
    state = extractmod.extract_state(store_dir, name)
    if state == "absent":
        raise StoreMissing(f"no per-ticker extract of {name} in {store_dir}", command)
    if state == "stale":
        raise StoreMissing(
            f"the per-ticker extract of {name} was built from another state of the store", command
        )
    entry = extractmod.read_manifest(store_dir)["tickers"][name]
    path = extractmod.ticker_path(store_dir, name)
    day = pads.field("trade_date")
    table = pads.dataset(path, format="parquet").to_table(
        columns=_columns(columns, int(entry["schema_version"])),
        filter=(day >= lo) & (day <= hi),
    )
    if table.num_rows == 0:
        raise StoreMissing(
            f"{name} has no row from {lo} to {hi}: its extract covers {entry['first']} to "
            f"{entry['last']} ({entry['n_dates']} dates); a date outside it must be in the "
            f"store first (volsto-data {flags}convert --vendor {vendor}), then",
            command,
        )
    frame: pd.DataFrame = table.replace_schema_metadata(None).to_pandas()
    frame.attrs.update(
        {
            "vendor": vendor,
            "ticker": name,
            "start": lo.isoformat(),
            "end": hi.isoformat(),
            "store_file": str(path),
            "store_digest": entry["store_digest"],
            "schema_version": int(entry["schema_version"]),
        }
    )
    return frame
