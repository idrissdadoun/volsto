"""Dispersion study (question 2): the Dow's point-in-time members and the frozen basket.

The main basket of the study is the Dow Jones Industrial Average as it stood on each entry
date (:func:`members_on`), frozen at entry: a Palladium's basket does not change.  What one
entry share of a name became by a later date — through splits, spin-offs, renames and mergers
— is tracked by :func:`holding_values` from the corporate actions of :data:`ACTIONS`, each a
mapping "one parent share becomes these shares" on its ex-date, with the spun-off shares kept
and valued at their own prices (spec §1.3).

Sources: Wikipedia, "Historical components of the Dow Jones Industrial Average" (membership;
fetched 2026-10-05), company notices for the ratios; every ratio is checked against the ORATS
prices around the ex-date by ``scripts/disp_setup.py`` (the jump of the holding's value across
the ex-date must be an ordinary daily return).
"""

# ruff: noqa: E501 — the table of corporate actions keeps one action per line
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

#: The thirty members on 2007-01-01 (ORATS spellings of that date).
MEMBERS_2007: Final[tuple[str, ...]] = (
    "MMM", "AA", "MO", "AXP", "AIG", "T", "BA", "CAT", "C", "KO", "DD", "XOM", "GE", "GM", "HPQ",
    "HD", "HON", "INTC", "IBM", "JNJ", "JPM", "MCD", "MRK", "MSFT", "PFE", "PG", "UTX", "VZ",
    "WMT", "DIS",
)  # fmt: skip

#: Membership changes: (first trading day of the new composition, added, removed).  A rename or
#: a merger that keeps the seat is a pair (new ticker in, old ticker out).
CHANGES: Final[tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...]] = (
    ("2008-02-19", ("BAC", "CVX"), ("MO", "HON")),
    ("2008-09-22", ("KFT",), ("AIG",)),
    ("2009-06-08", ("CSCO", "TRV"), ("C", "GM")),
    ("2012-09-24", ("UNH",), ("KFT",)),
    ("2013-09-23", ("GS", "NKE", "V"), ("AA", "BAC", "HPQ")),
    ("2015-03-19", ("AAPL",), ("T",)),
    ("2017-09-01", ("DWDP",), ("DD",)),
    ("2018-06-26", ("WBA",), ("GE",)),
    ("2019-04-02", ("DOW",), ("DWDP",)),
    ("2020-04-06", ("RTX",), ("UTX",)),
    ("2020-08-31", ("CRM", "AMGN", "HON"), ("XOM", "PFE", "RTX")),
    ("2024-02-26", ("AMZN",), ("WBA",)),
    ("2024-11-08", ("NVDA", "SHW"), ("INTC", "DOW")),
    ("2026-06-29", ("GOOGL",), ("VZ",)),
)

#: The ten large caps of basket B3 (chosen in 2026: survivorship).
B3_NAMES: Final[tuple[str, ...]] = (
    "AAPL", "MSFT", "AMZN", "NVDA", "JPM", "XOM", "JNJ", "PG", "HD", "UNH",
)  # fmt: skip


@dataclass(frozen=True)
class Action:
    """On ``ex_date`` one share of ``parent`` becomes ``becomes`` (ticker → shares).  A split
    maps the parent to itself, a spin-off keeps the parent and adds the child, a rename or a
    merger maps it to the new ticker."""

    ex_date: str
    parent: str
    becomes: tuple[tuple[str, float], ...]
    kind: str
    source: str


def _a(ex: str, parent: str, kind: str, source: str, **becomes: float) -> Action:
    return Action(ex, parent, tuple(becomes.items()), kind, source)


#: Corporate actions of every ticker that was ever a member, 2007 to 2026 (ex-dates as in the
#: ORATS prices).  Voluntary exchange offers (PG/Folgers 2008, PFE/Zoetis 2013, GE/Synchrony
#: 2015, PG/Coty 2016, JNJ/Kenvue 2023) change nothing for a holder who does not tender.
ACTIONS: Final[tuple[Action, ...]] = (
    # splits and spin-offs outside a ticker's membership: they enter trailing statistics (a
    # member's history before it joined), long windows after it left, and basket B3
    _a("2007-04-03", "NKE", "split", "Nike 2-for-1", NKE=2.0),
    _a(
        "2007-09-12",
        "NVDA",
        "split",
        "Nvidia 3-for-2 (no store row of NVDA on 2007-09-11)",
        NVDA=1.5,
    ),
    _a("2012-12-26", "NKE", "split", "Nike 2-for-1", NKE=2.0),
    _a("2013-04-18", "CRM", "split", "Salesforce 4-for-1", CRM=4.0),
    _a("2014-06-09", "AAPL", "split", "Apple 7-for-1", AAPL=7.0),
    _a(
        "2015-11-02",
        "HPQ",
        "spin-off",
        "Hewlett-Packard: 1 Hewlett Packard Enterprise per share",
        HPQ=1.0,
        HPE=1.0,
    ),
    _a("2021-04-01", "SHW", "split", "Sherwin-Williams 3-for-1", SHW=3.0),
    _a("2021-07-20", "NVDA", "split", "Nvidia 4-for-1", NVDA=4.0),
    _a("2021-08-02", "GE", "reverse split", "GE 1-for-8", GE=0.125),
    _a("2022-06-06", "AMZN", "split", "Amazon 20-for-1", AMZN=20.0),
    _a("2022-07-18", "GOOGL", "split", "Alphabet 20-for-1", GOOGL=20.0),
    _a("2024-06-10", "NVDA", "split", "Nvidia 10-for-1", NVDA=10.0),
    _a(
        "2007-04-03",
        "MO",
        "spin-off",
        "Altria notice: 0.692024 Kraft per share",
        MO=1.0,
        KFT=0.692024,
    ),
    _a(
        "2008-04-01",
        "MO",
        "spin-off",
        "Altria notice: 1 Philip Morris International per share",
        MO=1.0,
        PM=1.0,
    ),
    _a("2009-07-02", "AIG", "reverse split", "AIG 1-for-20", AIG=0.05),
    _a(
        "2010-07-06",
        "VZ",
        "spin-off",
        "Verizon notice: 1 Frontier per 4.165977 shares",
        VZ=1.0,
        FTR=0.240040,
    ),
    _a("2011-05-10", "C", "reverse split", "Citigroup 1-for-10", C=0.1),
    _a("2012-08-13", "KO", "split", "Coca-Cola 2-for-1", KO=2.0),
    _a(
        "2012-10-03",
        "KFT",
        "spin-off and rename",
        "Kraft Foods renamed Mondelez; 1 Kraft Foods Group per 3 shares",
        MDLZ=1.0,
        KRFT=1.0 / 3.0,
    ),
    _a("2015-03-19", "V", "split", "Visa 4-for-1", V=4.0),
    _a("2015-07-01", "DD", "spin-off", "DuPont notice: 1 Chemours per 5 shares", DD=1.0, CC=0.2),
    _a("2015-12-24", "NKE", "split", "Nike 2-for-1", NKE=2.0),
    _a("2017-09-01", "DD", "merger", "DowDuPont: 1.282 shares per DuPont share", DWDP=1.282),
    _a(
        "2019-02-27", "GE", "spin-off", "GE notice: 0.005371 Wabtec per share", GE=1.0, WAB=0.005371
    ),
    _a(
        "2019-04-03",
        "DWDP",
        "spin-off",
        "DowDuPont notice: 1 Dow Inc. per 3 shares",
        DWDP=1.0,
        DOW=1.0 / 3.0,
    ),
    _a(
        "2019-06-04",
        "DWDP",
        "spin-off, reverse split and rename",
        "DowDuPont: 1 Corteva per 3 shares, 1-for-3 reverse split, renamed DuPont",
        DD=1.0 / 3.0,
        CTVA=1.0 / 3.0,
    ),
    _a(
        "2020-04-03",
        "UTX",
        "spin-offs and rename",
        "United Technologies: 1 Carrier and 0.5 Otis per share; renamed Raytheon Technologies",
        RTX=1.0,
        CARR=1.0,
        OTIS=0.5,
    ),
    _a("2020-08-31", "AAPL", "split", "Apple 4-for-1", AAPL=4.0),
    _a(
        "2020-11-18",
        "PFE",
        "spin-off",
        "Pfizer notice: 0.124079 Viatris per share",
        PFE=1.0,
        VTRS=0.124079,
    ),
    _a("2021-06-04", "MRK", "spin-off", "Merck notice: 1 Organon per 10 shares", MRK=1.0, OGN=0.1),
    _a("2021-11-05", "IBM", "spin-off", "IBM notice: 1 Kyndryl per 5 shares", IBM=1.0, KD=0.2),
    _a("2024-02-26", "WMT", "split", "Walmart 3-for-1", WMT=3.0),
    _a("2024-04-01", "MMM", "spin-off", "3M notice: 1 Solventum per 4 shares", MMM=1.0, SOLV=0.25),
    _a(
        "2025-10-30",
        "HON",
        "spin-off",
        "Honeywell notice: 1 Solstice Advanced Materials per 4 shares",
        HON=1.0,
        SOLS=0.25,
    ),
    _a(
        "2026-06-29",
        "HON",
        "spin-off and reverse split",
        "Honeywell press release of June 2026: 1 Honeywell Aerospace per 2 shares and a 1-for-2 reverse split",
        HON=0.5,
        HONA=0.5,
    ),
)

#: Ex-dates are those of the ORATS prices: where the store has no row of the parent on the
#: official ex-date (MO 2007-04-02 and 2008-03-31, AIG 2009-07-01, VZ 2010-07-02, C 2011-05-09,
#: KFT 2012-10-02, GE 2019-02-26, DWDP 2019-04-02 and 2019-06-03, IBM 2021-11-04) the action
#: is dated on the parent's next row.
#:
#: Prices the study does not use: the "GM" rows from November 2010 are a different company (the
#: old General Motors left the store on 2009-06-01 and keeps its last price); Walgreens was
#: taken private on 2025-08-28 and its rows from that day are a cash deliverable.
PRICE_CUTS: Final[dict[str, str]] = {"GM": "2009-06-02", "WBA": "2025-08-28"}
#: A DJX price above this is the Dow itself (2018-01-09 to 2018-01-26 in the store): divided by 100.
DJX_MAX: Final = 5000.0
#: Days on which a ticker has no ORATS row yet and its closing price is taken from yfinance
#: (spun-off shares before their options are listed; RTX in April 2020).
PRICE_FILLS: Final[dict[str, tuple[str, str]]] = {
    "RTX": ("2020-04-03", "2020-04-24"),
    "CARR": ("2020-04-03", "2020-05-08"),
    "OTIS": ("2020-04-03", "2020-05-06"),
    "CTVA": ("2019-06-03", "2019-07-25"),
    "VTRS": ("2020-11-17", "2020-12-10"),
    "OGN": ("2021-06-03", "2021-07-14"),
    "KD": ("2021-11-04", "2021-11-16"),
    "SOLV": ("2024-04-01", "2024-04-05"),
    "SOLS": ("2025-10-30", "2025-12-16"),
    "HONA": ("2026-06-29", "2026-07-08"),
}
#: Predecessor of a ticker for trailing statistics: (ticker before, first day of the new one).
PREDECESSOR: Final[dict[str, tuple[str, str]]] = {
    "DWDP": ("DD", "2017-09-01"),
    "DOW": ("DWDP", "2019-04-02"),
    "RTX": ("UTX", "2020-04-03"),
    "TRV": ("STA", "2007-02-27"),
    "WBA": ("WAG", "2014-12-31"),
    "GOOGL": ("GOOG", "2014-04-03"),
}

#: Every ticker whose daily price the study reads: members, successors and spun-off shares,
#: the listed baskets and the S&P 500.
CHILDREN: Final[tuple[str, ...]] = tuple(
    sorted(
        {t for a in ACTIONS for t, _ in a.becomes} | {"PM", "MDLZ", "KRFT", "STA", "WAG", "GOOG"}
    )
)
BASKET_TICKERS: Final[tuple[str, ...]] = ("DJX", "DIA")


def all_members() -> list[str]:
    """Every ticker that was a member at some date from 2007 (ORATS spelling of its dates)."""
    out = list(MEMBERS_2007)
    for _, added, _ in CHANGES:
        out += [t for t in added if t not in out]
    return out


def price_tickers() -> list[str]:
    return sorted(
        set(all_members()) | set(CHILDREN) | set(B3_NAMES) | set(BASKET_TICKERS) | {"SPX"}
    )


def members_on(date: str) -> list[str]:
    """The thirty members on ``date`` (ISO), in a fixed order."""
    names = list(MEMBERS_2007)
    for day, added, removed in CHANGES:
        if day > date:
            break
        for t in removed:
            names.remove(t)
        names += list(added)
    if len(names) != 30 or len(set(names)) != 30:
        raise AssertionError(f"{len(names)} members on {date}")
    return names


def membership_table(last: str = "2026-10-02") -> pd.DataFrame:
    """One row per seat spell: ``ticker, start, end`` (both included; ``end`` empty if still a
    member on ``last``)."""
    start = {t: "2007-01-01" for t in MEMBERS_2007}
    rows = []
    for day, added, removed in CHANGES:
        prev = (pd.Timestamp(day) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        for t in removed:
            rows.append({"ticker": t, "start": start.pop(t), "end": prev})
        for t in added:
            start[t] = day
    rows += [{"ticker": t, "start": s, "end": ""} for t, s in start.items()]
    return pd.DataFrame(rows).sort_values(["start", "ticker"]).reset_index(drop=True)


def actions_frame() -> pd.DataFrame:
    rows = []
    for a in ACTIONS:
        for child, ratio in a.becomes:
            rows.append(
                {
                    "ex_date": a.ex_date,
                    "parent": a.parent,
                    "action": a.kind,
                    "ratio": ratio,
                    "child": child,
                    "source": a.source,
                }
            )
    return pd.DataFrame(rows)


def holding_path(ticker: str, entry: str, last: str) -> list[tuple[str, dict[str, float]]]:
    """The composition of the holding that started as one share of ``ticker`` on ``entry``:
    ``[(from date, {ticker: shares}), …]``, the first from ``entry``, one more for each action
    with ``entry < ex_date <= last`` that touches it (actions chain: DD → DWDP → DD, DOW,
    CTVA)."""
    holding = {ticker: 1.0}
    out = [(entry, dict(holding))]
    for a in sorted(ACTIONS, key=lambda x: x.ex_date):
        if not (entry < a.ex_date <= last) or a.parent not in holding:
            continue
        shares = holding.pop(a.parent)
        for child, ratio in a.becomes:
            holding[child] = holding.get(child, 0.0) + shares * ratio
        out.append((a.ex_date, dict(holding)))
    return out


def holding_values(
    prices: pd.DataFrame, ticker: str, entry: str, dates: list[str]
) -> tuple[FloatArray, bool, int]:
    """Value on each of ``dates`` (ascending, from ``entry``) of the holding that started as one
    share of ``ticker`` on ``entry``, per unit of its entry price: ``(X, event, carried)``.

    ``prices`` is the daily price panel (index ISO dates, one column per ticker, NaN where the
    ticker has no row).  A missing price is carried from the last available one (a delisted
    name keeps its last price); on a day when a component has never had a price yet (a spun-off
    share before its first print) the holding keeps its previous value.  ``event`` says an
    action fell inside the dates; ``carried`` counts the days valued with a carried price or
    value."""
    path = holding_path(ticker, entry, dates[-1])
    filled = prices.ffill()
    rows = prices.index.get_indexer(pd.Index(dates))
    tickers = sorted({t for _, h in path for t in h})
    raw = {
        t: prices[t].to_numpy(float)[rows] if t in prices.columns else np.full(len(dates), np.nan)
        for t in tickers
    }
    fill = {
        t: filled[t].to_numpy(float)[rows] if t in filled.columns else np.full(len(dates), np.nan)
        for t in tickers
    }
    p0 = float(fill[ticker][0])
    if not np.isfinite(p0) or p0 <= 0:
        raise ValueError(f"{ticker} has no price on {entry}")
    out = np.empty(len(dates))
    carried = 0
    last_value = p0
    starts = [d for d, _ in path]
    for j, d in enumerate(dates):
        k = int(np.searchsorted(starts, d, side="right")) - 1
        value, ok, stale = 0.0, True, False
        for t, shares in path[k][1].items():
            px = float(fill[t][j])
            if not np.isfinite(px):
                ok = False
                break
            stale = stale or not np.isfinite(float(raw[t][j]))
            value += shares * px
        if not ok:
            value, stale = last_value, True
        carried += int(stale)
        out[j] = value / p0
        last_value = value
    return out, len(path) > 1, carried


def clean_prices(raw: pd.DataFrame, fills: pd.DataFrame | None = None) -> pd.DataFrame:
    """The study's price panel from the raw ORATS one: the rows of :data:`PRICE_CUTS` removed,
    and the days of :data:`PRICE_FILLS` taken from ``fills`` (closing prices, same layout)
    where ORATS has no row."""
    out = raw.copy()
    if (
        "DJX" in out
    ):  # the store shows the full index level instead of a hundredth of it on a few days
        out["DJX"] = out["DJX"].where(out["DJX"] < DJX_MAX, out["DJX"] / 100.0)
    for t, start in PRICE_CUTS.items():
        if t in out:
            out.loc[out.index >= start, t] = np.nan
    if fills is not None:
        for t, (lo, hi) in PRICE_FILLS.items():
            if t not in fills:
                continue
            if t not in out:
                out[t] = np.nan
            sel = (out.index >= lo) & (out.index <= hi) & out[t].isna()
            out.loc[sel, t] = fills[t].reindex(out.index)[sel]
    return out


def history_returns(prices: pd.DataFrame, ticker: str) -> pd.Series:
    """:func:`adjusted_returns` of ``ticker``, continued backwards by its predecessors'
    (:data:`PREDECESSOR`): the series the trailing statistics of an entry-date member use."""
    r = adjusted_returns(prices, ticker)
    t = ticker
    while t in PREDECESSOR:
        before, first = PREDECESSOR[t]
        if before not in prices:
            break
        rb = adjusted_returns(prices, before)
        r = r.where(r.index > first, rb)
        t = before
    return r


def adjusted_returns(prices: pd.DataFrame, ticker: str) -> pd.Series:
    """Daily returns of holding ``ticker`` with every action applied on its ex-date (the
    spun-off shares counted at their price of that day, or at their first price within five
    days, then left out): the series the trailing statistics use.  NaN where the return would
    cross a gap of more than five trading days or the ticker has no price."""
    p = prices[ticker].to_numpy(float)
    n = p.size
    pos = np.arange(n)
    seen = np.where(np.isfinite(p), pos, -1)
    last_seen = np.concatenate([[-1], np.maximum.accumulate(seen)[:-1]])
    prev = np.where(last_seen >= 0, p[np.maximum(last_seen, 0)], np.nan)
    r = p / prev - 1.0
    r[(pos - last_seen) > 5] = np.nan
    index = list(prices.index)
    where = {d: i for i, d in enumerate(index)}
    for a in ACTIONS:
        if a.parent != ticker or a.ex_date not in where:
            continue
        i = where[a.ex_date]
        if i == 0 or not np.isfinite(prev[i]):
            continue
        value = 0.0
        for child, ratio in a.becomes:
            px = float("nan")
            if child in prices.columns:
                c = prices[child].to_numpy(float)
                # a spun-off share may print a few days late; the parent itself must print
                window = c[i : i + 1] if child == ticker else c[i : i + 6]
                window = window[np.isfinite(window)]
                px = float(window[0]) if window.size else float("nan")
            value += ratio * px
        r[i] = value / prev[i] - 1.0 if np.isfinite(value) else np.nan
    return pd.Series(r, index=prices.index)
