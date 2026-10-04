"""ORATS "Near End-of-Day" importer for an index (SPEC §18.6, milestone M11 Part 4).

Only the loader is ORATS-specific: :func:`load_day` reads one ticker's rows of one trade date
from the Parquet store (:func:`volsto.market.store.load_chain`; nothing here downloads or
converts) and returns the canonical chain of :mod:`volsto.market.chain` — the frame the HDN
loader returns too.  Everything after it is the vendor-independent pipeline of
:mod:`volsto.market.import_hdn`: :func:`~volsto.market.import_hdn.implied_forwards`,
:func:`~volsto.market.import_hdn.to_grid_surface`, :func:`~volsto.market.import_hdn.fit_ssvi`,
:func:`~volsto.market.import_hdn.sabrw_fits`, :func:`~volsto.market.import_hdn.snapshot_document`.

**From the vendor's rows to the chain.**  An ORATS row is one (OPRA root, expiry, strike) with
the call and the put side by side; the two melt apart into one row per contract.  The root is
taken from the contract's own OPRA symbol (:func:`volsto.data.orats.opra_root`), slices are
grouped by (root, expiration) as the HDN loader does — ticker ``SPX`` merges the roots ``SPX``
(AM-settled) and ``SPXW`` (PM-settled), which share the third-Friday expiry dates — and a
contract is kept only when its own bid and ask are both positive.

**The owner's decisions (2026-10-03), as implemented here.**

1. *Time to expiry* comes from (trade date, expiry date, OPRA root) with the HDN rule
   (:func:`~volsto.market.import_hdn.time_to_expiry`): calendar days / 365, one day less for
   the AM root (:data:`AM_ROOTS`).  The vendor's ``yte`` is carried as a cross-check only.
2. *Spread filter*: the bid and ask prices are inverted with our implied forward and discount
   factor.  The chain's ``iv_bid`` / ``iv_ask`` are NaN, so the vendor's one-sided vols
   (``cBidIv`` … ``pAskIv``) are never inputs; its mid vols go to ``iv`` as a cross-check.
   (The HDN path prefers the vendor's one-sided vols where present — the two vendors' filters
   differ in that, SPEC §18.6.)
3. *Schema*: declared in code (:mod:`volsto.data.orats`); the store is typed from it and this
   loader reads the store, so nothing is inferred from a file.
4. *Spot*: never ``spot_px`` (not read).  The pricing spot is the option-implied spot the
   pipeline computes; ``underlying_close`` is the official close of
   ``data/history/<underlying>.csv`` (:func:`official_close`), the fixing reference only.  The
   chain sets ``spot_async_check`` false: implied spot minus close is reported as a
   measurement, never as a data problem (the snapshot precedes the close).
5. *``stkPx``* is a cross-check only: :func:`forward_crosscheck` compares our parity forward
   with ``stkPx · exp(iRate · T)`` per expiry, in basis points.
6. *Prior rate curve*: the file's ``iRate`` — a step function of the day's Treasury
   constant-maturity yields — at each slice; the chain's ``rate_tenors`` / ``rate_zeros`` are
   those values at the slices' maturities.  It is a prior and a cross-check: the option-implied
   funding curve supersedes it exactly as for HDN.

*OPRA dependency* (decision 8): a row of the ticker without a ``cOpra`` or ``pOpra`` symbol
fails loudly, and so does a root outside the underlying's known roots; the fallback (AM/PM from
the expiry calendar) is a plan only (SPEC §18.6).  XSP and single stocks are out of scope
(decision 9; SPEC §18.8 reports what breaks on three of them).

**Provenance** (:func:`import_day`): vendor ``orats``, the raw file and its sha256, the store
schema version, the close's source, and the statement that the snapshot time is the vendor's
claim, not verified (the file has no timestamp).  ORATS-derived snapshots are written under the
store (:func:`snapshots_dir`), never under ``configs/``.

CLI: ``volsto-import --vendor orats --date 2024-01-03 --underlying SPX``.  Checked by
``tests/test_import_orats.py``.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from volsto.data import orats
from volsto.data.roots import REPO_ROOT, DataError, DataRoots
from volsto.market import import_hdn as ih
from volsto.market import store as vendor_store
from volsto.market.chain import validate_chain

VENDOR: Final = orats.VENDOR
PRODUCT: Final = "ORATS Near End-of-Day strikes (SMV)"
AM_ROOTS: Final[tuple[str, ...]] = ("SPX",)
"""OPRA roots settled on the morning of the expiry date (one day less in ``T``)."""
HISTORY_DIR: Final = REPO_ROOT / "data" / "history"
"""Official closes (``<underlying>.csv``, ``date,close``): the fixing reference."""
SNAPSHOT_TIME_NOTE: Final = (
    "the vendor's claim (about 14 minutes before the close); not verified: the file has no "
    "timestamp"
)
PRIOR_RATE_NOTE: Final = (
    "the file's iRate at each slice (a step function of the day's Treasury constant-maturity "
    "yields); a prior and a cross-check, superseded by the option-implied funding curve"
)
#: Vendor columns carried on the chain as cross-checks (never inputs).
CROSSCHECK_COLUMNS: Final[tuple[str, ...]] = (
    "yte",
    "stkPx",
    "iRate",
    "smoothSmvVol",
    "residualRateData",
)


def official_close(underlying: str, date: str, closes: str | Path | None = None) -> float:
    """The official close of ``underlying`` on ``date`` from ``closes`` (default
    ``data/history/<underlying>.csv``); a missing file or date fails loudly."""
    path = Path(closes) if closes is not None else HISTORY_DIR / f"{underlying.upper()}.csv"
    if not path.exists():
        raise DataError(f"no close history {path} for {underlying} (the fixing reference)")
    frame = pd.read_csv(path, dtype={"date": str})
    row = frame.loc[frame["date"] == date, "close"]
    if row.empty:
        raise DataError(f"{path} has no close for {date}: refresh the close history")
    return float(row.iloc[0])


def snapshots_dir(store: str | Path | None = None) -> Path:
    """Where ORATS-derived snapshots go: ``<store>/orats/snapshots`` (vendor-derived data
    stays under the store, never under ``configs/``)."""
    return DataRoots.resolve(store=store).store_dir(VENDOR) / "snapshots"


def _side(frame: pd.DataFrame, cp: int) -> pd.DataFrame:
    """The call (``cp`` = 1) or put side of the vendor rows as contract rows."""
    c = "c" if cp == 1 else "p"
    out = pd.DataFrame(
        {
            "contract": frame[f"{c}Opra"].str.strip(),
            "expiration": frame["expirDate"].map(_dt.date.isoformat),
            "quote_date": frame["trade_date"].map(_dt.date.isoformat),
            "cp": cp,
            "strike": frame["strike"].astype(float),
            "bid": frame[f"{c}BidPx"].astype(float),
            "ask": frame[f"{c}AskPx"].astype(float),
            "iv": frame[f"{c}MidIv"].astype(float),
            "volume": frame[f"{c}Volu"],
            "open_interest": frame[f"{c}Oi"],
        }
    )
    for name in CROSSCHECK_COLUMNS:
        out[name] = frame[name].to_numpy()
    return out


def load_day(
    date: str | _dt.date,
    underlying: str = "SPX",
    *,
    store: str | Path | None = None,
    closes: str | Path | None = None,
) -> pd.DataFrame:
    """One trading day's canonical chain for ``underlying`` from the ORATS store (module
    docstring).  Raises :class:`volsto.market.store.StoreMissing` (with the ``volsto-data``
    command) when the store does not hold the day, and :class:`DataError` on a missing OPRA
    symbol, an unknown root or a missing close."""
    name = underlying.upper()
    frame = vendor_store.load_chain(VENDOR, date, name, store=store)
    iso = str(frame.attrs["trade_date"])
    empty = (frame["cOpra"].str.strip() == "") | (frame["pOpra"].str.strip() == "")
    if empty.any():
        raise DataError(
            f"orats {iso}: {int(empty.sum())} of {len(frame)} {name} rows without an OPRA "
            "symbol — the root is the only AM/PM marker and no fallback is implemented"
        )
    chain = pd.concat([_side(frame, 1), _side(frame, -1)], ignore_index=True)
    chain["root"] = chain["contract"].map(orats.opra_root)
    roots = ih.INDEX_ROOTS.get(name, (name,))
    unknown = sorted(set(chain["root"]) - set(roots))
    if unknown:
        raise DataError(
            f"orats {iso}: {name} rows carry the OPRA roots {unknown}, outside the known "
            f"{list(roots)}: their settlement is not known to the importer"
        )
    chain = chain[(chain["bid"] > 0) & (chain["ask"] > 0)].copy()
    chain["expiry"] = chain["root"] + "|" + chain["expiration"]
    chain["mid"] = 0.5 * (chain["bid"] + chain["ask"])
    settlement = pd.Series(np.where(chain["root"].isin(AM_ROOTS), "AM", "PM"), index=chain.index)
    chain["T"] = ih.time_to_expiry(chain["quote_date"], chain["expiration"], settlement)
    chain = chain[chain["T"] > 0].copy()
    if chain.empty:
        raise DataError(f"orats {iso}: no two-sided {name} contract with a positive maturity")
    chain["r"] = chain["iRate"].astype(float)
    chain["df"] = np.exp(-chain["r"] * chain["T"])
    close = official_close(name, iso, closes)
    chain["underlying_close"] = close
    chain["iv_bid"] = np.nan  # decision 2: the bid and ask prices are inverted, never the
    chain["iv_ask"] = np.nan  # vendor's one-sided vols
    chain["iv_flag"] = np.nan
    chain = chain.sort_values(["T", "root", "strike", "cp"]).reset_index(drop=True)
    prior = chain.groupby("T")["r"].first()
    chain.attrs.update(
        {
            "quote_date": iso,
            "underlying": name,
            "file": frame.attrs["raw_file"],
            "rate_tenors": [float(t) for t in prior.index],
            "rate_zeros": [float(r) for r in prior.to_numpy()],
            "spot": close,
            "vendor": VENDOR,
            "spot_async_check": False,
            "raw_sha256": frame.attrs["raw_sha256"],
            "schema_version": int(frame.attrs["schema_version"]),
            "closes": str(Path(closes) if closes is not None else HISTORY_DIR / f"{name}.csv"),
        }
    )
    return validate_chain(chain)


def source_provenance(chain: pd.DataFrame) -> dict[str, Any]:
    """The vendor's entries of a snapshot's provenance (module docstring)."""
    return {
        "vendor": VENDOR,
        "product": PRODUCT,
        "file": chain.attrs["file"],
        "file_sha256": chain.attrs["raw_sha256"],
        "store_schema_version": chain.attrs["schema_version"],
        "snapshot_time": SNAPSHOT_TIME_NOTE,
        "close_source": Path(chain.attrs["closes"]).name,
        "prior_rate_curve": PRIOR_RATE_NOTE,
    }


def import_day(
    date: str | _dt.date,
    underlying: str = "SPX",
    *,
    filters: ih.HdnFilters | None = None,
    essvi: bool = True,
    calendar_repair: ih.CalendarRepairConfig | None = ih.DEFAULT_CALENDAR_REPAIR,
    store: str | Path | None = None,
    closes: str | Path | None = None,
) -> tuple[dict[str, Any], ih.SSVIFit, ih.SurfacePoints, pd.DataFrame]:
    """The whole pipeline for one ORATS day; returns ``(config, fit, points, chain)`` like
    :func:`volsto.market.import_hdn.import_day`."""
    f = filters or ih.HdnFilters()
    chain = load_day(date, underlying, store=store, closes=closes)
    fwds = ih.implied_forwards(chain, max_years=f.max_years, band=f.near_atm_band)
    grid, points = ih.to_grid_surface(chain, fwds, f)
    fit = ih.fit_ssvi(grid, points, filters=f, essvi=essvi, calendar_repair=calendar_repair)
    t_max = float(points.table["T"].max())
    fits = (
        ih.sabrw_fits(points, t_min=ih.SABRW_T_MIN, t_max=t_max) if t_max > ih.SABRW_T_MIN else ()
    )
    cfg = ih.snapshot_document(chain, fit, points, f, source=source_provenance(chain), sabrw=fits)
    return cfg, fit, points, chain


def forward_crosscheck(
    chain: pd.DataFrame, forwards: dict[str, ih.ForwardEstimate]
) -> pd.DataFrame:
    """Decision 5: per slice, our parity forward against the vendor's ``stkPx`` carried at its
    own ``iRate`` — ``bp`` = 1e4 (F / (stkPx · exp(iRate · T)) − 1) with our ``T``, and
    ``bp_yte`` the same with the vendor's ``yte`` — with the forward's standard error in bp and
    the vendor's ``yte`` against our ``T`` in days.  A cross-check, never an input."""
    rows = []
    for key, fe in sorted(forwards.items(), key=lambda kv: kv[1].T):
        g = chain[chain["expiry"] == key]
        stk, rate, yte = (float(g[c].iloc[0]) for c in ("stkPx", "iRate", "yte"))
        rows.append(
            {
                "expiry": key,
                "T": fe.T,
                "days": round(fe.T * 365.0),
                "n_pairs": fe.n_pairs,
                "forward": fe.forward,
                "forward_se_bp": 1e4 * fe.forward_stderr / fe.forward,
                "stkPx": stk,
                "iRate": rate,
                "bp": 1e4 * (fe.forward / (stk * np.exp(rate * fe.T)) - 1.0),
                "bp_yte": 1e4 * (fe.forward / (stk * np.exp(rate * yte)) - 1.0),
                "yte_minus_T_days": 365.0 * (yte - fe.T),
                "implied_rate": fe.implied_rate,
            }
        )
    return pd.DataFrame(rows)
