"""``volsto-data extract``: one Parquet file per ticker across all dates (SPEC §18.4, M11
Part 2b), for backtests that read one underlying's whole history.

Derived from the store and rebuildable from it: ``<store>/<vendor>/by_ticker/<TICKER>.parquet``
holds every store row of the ticker, the vendor's columns unchanged, in (trade_date, expirDate,
strike, cOpra) order, **one row group per trade date** (so a date range prunes on the
``trade_date`` statistics), in the store's compression.  ``by_ticker/manifest.json`` binds each
extract to the store it was built from (:func:`volsto.data.store.store_digest`), with its row
count, dates and sha256; an extract whose store digest is not the current one is stale
(:func:`extract_state`) and is rebuilt by the same command.  Files are written to a temporary
name in their directory and renamed.

A ticker that appears on no date fails loudly (no empty file is written).  Every store file
has the store's one schema, so dates of different raw layouts concatenate; the manifest lists
the layouts met (``schema_versions``).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import pyarrow.compute as pc
import pyarrow.parquet as pq

from volsto.data import raw as rawmod
from volsto.data import store as storemod
from volsto.data.roots import DataError, ensure_dir, require_free_space

BY_TICKER_DIR: Final = "by_ticker"
MANIFEST_NAME: Final = "manifest.json"
MANIFEST_VERSION: Final = 1
_TICKER = re.compile(r"[A-Z0-9][A-Z0-9_.]*")


def ticker_path(store_dir: Path, ticker: str) -> Path:
    return store_dir / BY_TICKER_DIR / f"{ticker}.parquet"


def parse_tickers(text: str | Sequence[str]) -> list[str]:
    """``"SPX,AAPL"`` → ``["SPX", "AAPL"]`` (upper-cased, de-duplicated, order kept)."""
    items = text.split(",") if isinstance(text, str) else list(text)
    out: list[str] = []
    for item in items:
        t = item.strip().upper()
        if not t:
            continue
        if not _TICKER.fullmatch(t):
            raise DataError(f"{item!r} is not a ticker")
        if t not in out:
            out.append(t)
    if not out:
        raise DataError("no ticker given")
    return out


def read_manifest(store_dir: Path) -> dict[str, Any]:
    p = store_dir / BY_TICKER_DIR / MANIFEST_NAME
    if not p.exists():
        return {"version": MANIFEST_VERSION, "tickers": {}}
    data: dict[str, Any] = json.loads(p.read_text())
    if data.get("version") != MANIFEST_VERSION:
        raise DataError(f"{p}: extract manifest version {data.get('version')!r}, expected 1")
    return data


def extract_state(store_dir: Path, ticker: str) -> str:
    """``"current"``, ``"stale"`` (built from another state of the store) or ``"absent"``."""
    entry = read_manifest(store_dir)["tickers"].get(ticker)
    if entry is None or not ticker_path(store_dir, ticker).is_file():
        return "absent"
    current = storemod.store_digest(storemod.read_manifest(store_dir))
    return "current" if entry["store_digest"] == current else "stale"


@dataclass
class ExtractReport:
    store_dir: Path
    tickers: dict[str, dict[str, Any]] = field(default_factory=dict)
    missing: list[str] = field(default_factory=list)  # tickers on no date
    dates: int = 0
    wall_s: float = 0.0


def extract(store_dir: Path, tickers: Sequence[str]) -> ExtractReport:
    """Build (or rebuild) the per-ticker files of ``tickers`` from the store."""
    t0 = time.perf_counter()
    wanted = parse_tickers(tickers)
    manifest = storemod.read_manifest(store_dir)
    files = manifest["files"]
    if not files:
        raise DataError(
            f"the store {store_dir} is empty: run volsto-data convert --vendor orats first"
        )
    digest = storemod.store_digest(manifest)
    schema = storemod.arrow_schema()
    out_dir = ensure_dir(store_dir / BY_TICKER_DIR)
    # an extract is at most the whole store (one ticker never is); refuse if even that is short
    require_free_space(
        out_dir, sum(e["bytes"] for e in files.values()) // 10, what="the per-ticker extracts"
    )
    tmp = {t: out_dir / f".{t}.parquet.{os.getpid()}.tmp" for t in wanted}
    writers: dict[str, pq.ParquetWriter] = {}
    stats: dict[str, dict[str, Any]] = {
        t: {"rows": 0, "dates": [], "versions": set()} for t in wanted
    }
    layout = storemod.LAYOUT
    try:
        for date in sorted(files):
            path = storemod.day_path(store_dir, date)
            if not path.is_file():
                raise DataError(
                    f"{path} is in the store manifest but not on disk: run volsto-data verify"
                )
            day = pq.read_table(path, filters=[("ticker", "in", wanted)])
            if day.num_rows == 0:
                continue
            day = day.replace_schema_metadata(None).cast(schema)
            names = day["ticker"]
            for t in wanted:
                part = day.filter(pc.equal(names, t))
                if part.num_rows == 0:
                    continue
                if t not in writers:
                    writers[t] = pq.ParquetWriter(
                        tmp[t],
                        schema,
                        compression=layout["compression"],
                        compression_level=layout["compression_level"],
                        use_dictionary=layout["dictionary"],
                    )
                writers[t].write_table(part, row_group_size=max(part.num_rows, 1))
                stats[t]["rows"] += part.num_rows
                stats[t]["dates"].append(date)
                stats[t]["versions"].add(int(files[date]["schema_version"]))
        for w in writers.values():
            w.close()
        rep = ExtractReport(store_dir, dates=len(files))
        man = read_manifest(store_dir)
        for t in wanted:
            if t not in writers:
                rep.missing.append(t)
                continue
            dest = ticker_path(store_dir, t)
            with tmp[t].open("rb") as fh:
                os.fsync(fh.fileno())
            os.replace(tmp[t], dest)
            man["tickers"][t] = {
                "file": dest.name,
                "rows": stats[t]["rows"],
                "n_dates": len(stats[t]["dates"]),
                "first": stats[t]["dates"][0],
                "last": stats[t]["dates"][-1],
                "bytes": dest.stat().st_size,
                "sha256": rawmod.sha256_file(dest),
                "schema_versions": sorted(stats[t]["versions"]),
                "store_digest": digest,
                "built_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
            }
            rep.tickers[t] = man["tickers"][t]
        _write_manifest(store_dir, man)
    finally:
        for w in writers.values():
            if w.is_open:
                w.close()
        for p in tmp.values():
            p.unlink(missing_ok=True)
    rep.wall_s = time.perf_counter() - t0
    return rep


def _write_manifest(store_dir: Path, manifest: dict[str, Any]) -> None:
    p = store_dir / BY_TICKER_DIR / MANIFEST_NAME
    tmp = p.with_name(f".{p.name}.{os.getpid()}.tmp")
    manifest["tickers"] = dict(sorted(manifest["tickers"].items()))
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)
