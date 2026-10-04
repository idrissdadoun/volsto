"""AM or PM settlement of the SPX monthly series where the ORATS file cannot say (SPEC §18.10).

From the day the ``SPXPM`` ticker disappears to the day the OPRA symbols appear (2017-05-10 to
2021-05-27 in the archive) a third-Friday row under ticker ``SPX`` is the AM-settled monthly
or the PM-settled third-Friday series, with nothing in the row to tell.  Their open interest
differs by an order of magnitude, so the change of series shows as a jump of an expiry's total
open interest from one trade date to the next.  :func:`classify` turns the open interest per
(trade date, monthly expiry) into a settlement per (trade date, monthly expiry), with the
evidence beside it:

* every expiry starts on the AM series — unless it is first listed after the first switch of
  the archive and never switches although its switch would have been seen (its expiry is
  within :data:`SWITCH_LEAD_DAYS` of the last date of the table): it is then the PM series
  from its first day (the vendor lists the PM series of a month about 180 days ahead, and the
  AM series of the months listed earlier switch about 150 days ahead);
* a **switch**: the open interest falls below a third of the previous day's (which was at
  least :data:`LEVEL`) and stays there — PM from that day on;
* a **one-day excursion**: a jump by more than :data:`FACTOR` that is undone the next day.
  When every excursion of the day is a PM-state expiry jumping up, the file went back to the
  AM series for that day (a *relapse*): those expiries are AM that day.  Any other pattern
  cannot be told: the affected expiries are ``DROP`` that day (the importer leaves them out of
  the chain and flags the day).

The table is derived from vendor data and lives under the store
(``<store>/orats/spx_settlement.csv``, :func:`table_path`), never in git; it is built by
``scripts/orats_spx_settlement.py``.  :func:`load_table` reads it for the importer.
"""

from __future__ import annotations

import datetime as _dt
from functools import lru_cache
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from volsto.data.roots import DataError

TABLE_NAME: Final = "spx_settlement.csv"
#: A jump of an expiry's open interest by more than this factor is an event.
FACTOR: Final[float] = 3.0
#: The high side of an event must be at least this many contracts (new listings growing
#: from nothing are not events).
LEVEL: Final[float] = 20_000.0
#: A monthly switches to the PM series about this many calendar days before its expiry (151 on
#: every switch of the archive after the first batch): an expiry further than this from the
#: table's last date may still switch.
SWITCH_LEAD_DAYS: Final[int] = 150
AM: Final = "AM"
PM: Final = "PM"
DROP: Final = "DROP"
COLUMNS: Final[tuple[str, ...]] = ("date", "expiry", "settlement", "evidence", "oi", "oi_ref")


def table_path(store_dir: Path) -> Path:
    return store_dir / TABLE_NAME


def _event(prev: float, now: float) -> str:
    """``"down"``, ``"up"`` or ``""``: ``now`` against ``prev`` (:data:`FACTOR`, :data:`LEVEL`)."""
    if prev >= LEVEL and now < prev / FACTOR:
        return "down"
    if now >= LEVEL and now > prev * FACTOR:
        return "up"
    return ""


def classify(oi: pd.DataFrame) -> pd.DataFrame:
    """The settlement table of ``oi`` — a frame indexed by ISO trade date (ascending), one
    column per monthly expiry (ISO), total call + put open interest under ticker ``SPX``, NaN
    where the expiry is not in the day's file.  Returns :data:`COLUMNS`, one row per listed
    (date, expiry): ``settlement`` in ``AM`` / ``PM`` / ``DROP`` and the ``evidence`` (module
    docstring), with the day's open interest and the reference level it was compared with."""
    dates = list(oi.index)
    # pass 1: raw events per expiry against a reference level that excursions do not move
    raw: dict[str, list[tuple[int, str, bool, float]]] = {}  # expiry → (i, kind, excursion, ref)
    for expiry in oi.columns:
        x = oi[expiry].to_numpy(dtype=np.float64)
        listed = np.flatnonzero(~np.isnan(x))
        events: list[tuple[int, str, bool, float]] = []
        ref: float | None = None
        for pos, at in enumerate(listed):
            now = float(x[at])
            if ref is None:
                ref = now
                continue
            kind = _event(ref, now)
            if not kind:
                ref = now
                continue
            nxt = float(x[listed[pos + 1]]) if pos + 1 < len(listed) else float("nan")
            undone = not np.isnan(nxt) and not _event(ref, nxt) and bool(_event(now, nxt))
            events.append((int(at), kind, undone, ref))
            if not undone:
                ref = now
        raw[str(expiry)] = events
    switches = [
        i for ev in raw.values() for i, kind, undone, _ in ev if kind == "down" and not undone
    ]
    first_switch = min(switches) if switches else len(dates)
    # pass 2: the state of each expiry on each date, then the day-level reading of excursions
    state: dict[tuple[int, str], tuple[str, str, float]] = {}  # → (state, evidence, ref)
    excursions: dict[int, list[tuple[str, str, str]]] = {}  # i → (expiry, kind, state)
    for expiry, events in raw.items():
        x = oi[expiry].to_numpy(dtype=np.float64)
        listed = np.flatnonzero(~np.isnan(x))
        if listed.size == 0:
            continue
        first = int(listed[0])
        to_expiry = (_dt.date.fromisoformat(expiry) - _dt.date.fromisoformat(dates[-1])).days
        switches_later = any(kind == "down" and not undone for _, kind, undone, _ in events)
        born_pm = first >= first_switch and not switches_later and to_expiry < SWITCH_LEAD_DAYS
        current = PM if born_pm else AM
        evidence = (
            "first listed after the archive's first switch; never switches" if born_pm else ""
        )
        by_index = {i: (kind, undone, ref) for i, kind, undone, ref in events}
        ref_now = float(x[first])
        for j in listed:
            i = int(j)
            if i in by_index:
                kind, undone, ref = by_index[i]
                if undone:
                    excursions.setdefault(i, []).append((expiry, kind, current))
                    state[(i, expiry)] = (current, f"one-day {kind}", ref)
                    continue
                if kind == "down" and current == AM:
                    current = PM
                    evidence = f"switch on {dates[i]}: {ref:,.0f} -> {x[i]:,.0f}"
                # a lasting rise, or a lasting fall of a PM series: growth or decay, no change
            else:
                ref_now = float(x[i])
            state[(i, expiry)] = (current, evidence, by_index[i][2] if i in by_index else ref_now)
    rows = []
    for (i, expiry), (current, evidence, ref) in state.items():
        settlement = current
        if i in excursions and evidence.startswith("one-day"):
            day = excursions[i]
            relapse = all(kind == "up" and st == PM for _, kind, st in day)
            if relapse:
                settlement, evidence = AM, f"relapse day ({len(day)} expiries jump up, undone)"
            else:
                settlement, evidence = DROP, f"cannot tell: {evidence}, mixed with others"
        rows.append((dates[i], expiry, settlement, evidence, float(oi[expiry].iloc[i]), ref))
    out = pd.DataFrame(rows, columns=list(COLUMNS))
    return out.sort_values(["date", "expiry"]).reset_index(drop=True)


def summary(table: pd.DataFrame) -> dict[str, list[str]]:
    """The switches (one per expiry), the relapse days and the days with a ``DROP``."""
    sw = table[table["evidence"].str.startswith("switch on")]
    first = sw.drop_duplicates("expiry")
    return {
        "switches": [f"{r.expiry}: {r.evidence}" for r in first.itertuples()],
        "relapse_days": sorted(set(table.loc[table["evidence"].str.startswith("relapse"), "date"])),
        "drop_days": sorted(set(table.loc[table["settlement"] == DROP, "date"])),
    }


@lru_cache(maxsize=4)
def _load(path: str, mtime_ns: int) -> dict[tuple[str, str], str]:
    frame = pd.read_csv(path, dtype=str)
    missing = [c for c in ("date", "expiry", "settlement") if c not in frame.columns]
    if missing:
        raise DataError(f"{path}: columns missing {missing}")
    return {
        (d, e): s
        for d, e, s in zip(frame["date"], frame["expiry"], frame["settlement"], strict=True)
    }


def load_table(store_dir: Path) -> dict[tuple[str, str], str]:
    """``(ISO trade date, ISO canonical expiry) → AM / PM / DROP``; raises :class:`DataError`
    with the command that builds the table when it is absent."""
    path = table_path(store_dir)
    if not path.exists():
        raise DataError(f"no SPX settlement table {path}: python scripts/orats_spx_settlement.py")
    return _load(str(path), path.stat().st_mtime_ns)
