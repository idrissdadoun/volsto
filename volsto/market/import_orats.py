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

**SPX settlement by period** (owner's instruction, 2026-10-04; SPEC §18.10), from the data of
the day (:func:`settle`).  A row with OPRA symbols takes its root from them (``SPX`` AM,
``SPXW`` PM), and a root outside the underlying's known roots fails loudly.  A row without
symbols is settled by :func:`volsto.data.orats.settlement_without_opra`: ticker ``SPXPM`` is PM;
ticker ``SPX`` on a monthly expiry is AM while ``SPXPM`` rows exist that day or before
:data:`volsto.data.orats.SPXPM_FIRST_DATE`, PM otherwise.  Such a row is given the root of its
settlement (``SPX`` for AM, ``SPXW`` for PM) and contract names built from it
(:func:`volsto.data.orats.osi_symbol`).  The expiry is the last trading day
(:func:`volsto.data.orats.canonical_expiry`: the old files date a standard expiry on the
Saturday).  A day on which two rows claim the same (root, expiry, strike) cannot be settled
and raises :class:`AmbiguousSettlement`: it is excluded and listed by the caller.  XSP and
single stocks are out of scope (decision 9; SPEC §18.8 reports what breaks on three of them).

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

from volsto.data import orats, spx_settlement
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
#: Tickers whose rows make an underlying's chain (the first is the underlying's own).
FAMILY: Final[dict[str, tuple[str, ...]]] = {"SPX": ("SPX", orats.SPX_PM_TICKER)}
#: The root a row without OPRA symbols is given, by settlement.
ROOT_OF_SETTLEMENT: Final[dict[str, str]] = {"AM": "SPX", "PM": "SPXW"}
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


class AmbiguousSettlement(DataError):  # noqa: N818 — a day to exclude and list
    """A day whose rows cannot be told AM from PM: two rows claim one (root, expiry, strike)."""


def _symbol(col: pd.Series) -> pd.Series:
    """An OPRA column stripped, the empty string where the store holds null or nothing."""
    return col.fillna("").astype(str).str.strip()


def family_rows(
    date: str | _dt.date, name: str, *, store: str | Path | None = None
) -> pd.DataFrame:
    """The store rows of ``name``'s tickers (:data:`FAMILY`) on ``date``; a family ticker the
    day does not list is simply absent."""
    frame = vendor_store.load_chain(VENDOR, date, name, store=store)
    parts = [frame]
    for ticker in FAMILY.get(name, (name,))[1:]:
        try:
            parts.append(vendor_store.load_chain(VENDOR, date, ticker, store=store))
        except vendor_store.StoreMissing as exc:
            if exc.command is not None:  # the day itself is missing: not ours to hide
                raise
    out = pd.concat(parts, ignore_index=True) if len(parts) > 1 else frame
    out.attrs.update(frame.attrs)
    return out


def settle(frame: pd.DataFrame, name: str, *, store: str | Path | None = None) -> pd.DataFrame:
    """``frame`` (vendor rows of ``name``'s tickers on one day) with ``root``, ``settlement``
    (``AM``/``PM``), ``settled_by`` (``opra``, ``ticker`` or ``calendar``), ``expiration``
    (ISO, the last trading day) and the two contract names ``c_contract`` / ``p_contract``
    (module docstring).  Rows the settlement table cannot tell (``DROP``) are left out and
    listed in ``attrs["dropped_expiries"]``; ``attrs["settlement_uncertain"]`` flags December
    2010.  Raises :class:`DataError` on an unknown root, on symbols of two roots in one row or
    on a missing settlement table, :class:`AmbiguousSettlement` on a repeated (root, expiry,
    strike)."""
    iso = str(frame.attrs["trade_date"])
    trade = _dt.date.fromisoformat(iso)
    out = frame.copy()
    out.attrs.update(frame.attrs)
    c_sym, p_sym = _symbol(out["cOpra"]), _symbol(out["pOpra"])
    c_root = c_sym.map(lambda x: orats.opra_root(x) if x else "")
    p_root = p_sym.map(lambda x: orats.opra_root(x) if x else "")
    known = ih.INDEX_ROOTS.get(name, (name,))
    unknown = sorted((set(c_root) | set(p_root)) - {""} - set(known))
    if unknown:
        raise DataError(
            f"orats {iso}: {name} rows carry the OPRA roots {unknown}, outside the known "
            f"{list(known)}: their settlement is not known to the importer"
        )
    clash = (c_root != "") & (p_root != "") & (c_root != p_root)
    if clash.any():
        raise DataError(
            f"orats {iso}: {int(clash.sum())} {name} rows whose call and put roots differ"
        )
    sym_root = c_root.where(c_root != "", p_root)
    canon = out["expirDate"].map(lambda d: orats.canonical_expiry(d)[0])
    bare = sym_root == ""
    bare_mask = bare.to_numpy()
    settlement = np.where(sym_root.isin(AM_ROOTS), "AM", "PM").astype(object)
    settled_by = np.full(len(out), "opra", dtype=object)
    if bare_mask.any():
        if name not in FAMILY:
            raise DataError(
                f"orats {iso}: {int(bare_mask.sum())} of {len(out)} {name} rows without an "
                "OPRA symbol, and no settlement rule for this underlying"
            )
        twice = (
            out[bare]
            .assign(_canon=canon[bare])
            .duplicated(["ticker", "_canon", "strike"], keep=False)
        )
        if twice.any():
            ex = out[bare].loc[twice, ["ticker", "expirDate", "strike"]].head(3).to_dict("records")
            raise AmbiguousSettlement(
                f"orats {iso}: {int(twice.sum())} {name} rows without a symbol repeat a "
                f"(root, expiry, strike) and nothing tells them apart, e.g. {ex}"
            )
        tickers = out["ticker"].to_numpy()
        listed = bool((tickers == orats.SPX_PM_TICKER).any())
        table = None
        if not listed and trade >= orats.SPXPM_FIRST_DATE:
            table = spx_settlement.load_table(DataRoots.resolve(store=store).store_dir(VENDOR))
        try:
            settlement[bare_mask] = [
                orats.settlement_without_opra(t, e, trade, spxpm_listed=listed, table=table)
                for t, e in zip(tickers[bare_mask], out["expirDate"].to_numpy()[bare_mask])
            ]
        except KeyError as exc:
            raise DataError(
                f"orats {iso}: the SPX settlement table has no entry for {exc.args[0]}: "
                "python scripts/orats_spx_settlement.py"
            ) from exc
        settled_by[bare_mask] = np.where(
            tickers[bare_mask] == orats.SPX_PM_TICKER, "ticker", "calendar"
        )
    keep = settlement != spx_settlement.DROP
    dropped = sorted({d.isoformat() for d in canon[~keep]})
    if not keep.all():
        out, canon, sym_root = out[keep], canon[keep], sym_root[keep]
        c_sym, p_sym, bare = c_sym[keep], p_sym[keep], bare[keep]
        settlement, settled_by = settlement[keep], settled_by[keep]
    out.attrs.update(frame.attrs)
    out.attrs["dropped_expiries"] = dropped
    out.attrs["settlement_uncertain"] = orats.settlement_uncertain(trade)
    by_rule = pd.Series(settlement, index=out.index).map(ROOT_OF_SETTLEMENT)
    out["root"] = sym_root.where(~bare, by_rule)
    out["settlement"] = settlement
    out["settled_by"] = settled_by
    out["expiration"] = canon.map(_dt.date.isoformat)
    dup = out.duplicated(["root", "expiration", "strike"], keep=False)
    if dup.any():
        ex = out.loc[dup, ["ticker", "root", "expiration", "strike"]].head(3).to_dict("records")
        raise AmbiguousSettlement(
            f"orats {iso}: {int(dup.sum())} {name} rows repeat a (root, expiry, strike) and "
            f"nothing tells them apart, e.g. {ex}"
        )
    for c, cp, sym in (("c", 1, c_sym), ("p", -1, p_sym)):
        built = [
            orats.osi_symbol(r, e, cp, float(k))
            for r, e, k in zip(out["root"], canon, out["strike"])
        ]
        out[f"{c}_contract"] = sym.where(sym != "", pd.Series(built, index=out.index))
    return out


def _side(frame: pd.DataFrame, cp: int) -> pd.DataFrame:
    """The call (``cp`` = 1) or put side of the settled vendor rows as contract rows."""
    c = "c" if cp == 1 else "p"
    out = pd.DataFrame(
        {
            "contract": frame[f"{c}_contract"],
            "root": frame["root"],
            "settlement": frame["settlement"],
            "expiration": frame["expiration"],
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
    command) when the store does not hold the day, :class:`AmbiguousSettlement` on a day that
    cannot be settled, and :class:`DataError` on an unknown root or a missing close."""
    name = underlying.upper()
    frame = settle(family_rows(date, name, store=store), name, store=store)
    iso = str(frame.attrs["trade_date"])
    settled_by = {str(k): int(v) for k, v in frame["settled_by"].value_counts().items()}
    chain = pd.concat([_side(frame, 1), _side(frame, -1)], ignore_index=True)
    chain = chain[(chain["bid"] > 0) & (chain["ask"] > 0)].copy()
    chain["expiry"] = chain["root"] + "|" + chain["expiration"]
    chain["mid"] = 0.5 * (chain["bid"] + chain["ask"])
    chain["T"] = ih.time_to_expiry(chain["quote_date"], chain["expiration"], chain["settlement"])
    chain = chain.drop(columns="settlement")
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
            "settled_by": settled_by,
            "dropped_expiries": list(frame.attrs.get("dropped_expiries", [])),
            "settlement_uncertain": bool(frame.attrs.get("settlement_uncertain", False)),
            "closes": str(Path(closes) if closes is not None else HISTORY_DIR / f"{name}.csv"),
        }
    )
    return validate_chain(chain)


def source_provenance(chain: pd.DataFrame) -> dict[str, Any]:
    """The vendor's entries of a snapshot's provenance (module docstring)."""
    out = {
        "vendor": VENDOR,
        "product": PRODUCT,
        "file": chain.attrs["file"],
        "file_sha256": chain.attrs["raw_sha256"],
        "store_schema_version": chain.attrs["schema_version"],
        "snapshot_time": SNAPSHOT_TIME_NOTE,
        "close_source": Path(chain.attrs["closes"]).name,
        "prior_rate_curve": PRIOR_RATE_NOTE,
    }
    settled_by = dict(chain.attrs.get("settled_by", {}))
    if set(settled_by) - {"opra"}:  # a day settled, in part, without OPRA symbols
        out["settlement"] = {
            "rule": "OPRA root where present; ticker SPXPM is PM; ticker SPX on a monthly "
            "expiry is AM while SPXPM is listed that day or before "
            f"{orats.SPXPM_FIRST_DATE.isoformat()}, then as the open-interest table says; "
            f"non-monthly Fridays before {orats.WEEKLY_PM_FROM.isoformat()} are AM",
            "vendor_rows_by_source": settled_by,
            "dropped_expiries": list(chain.attrs.get("dropped_expiries", [])),
            "uncertain": bool(chain.attrs.get("settlement_uncertain", False)),
        }
    return out


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
