"""The ORATS "Near End-of-Day" strikes file (SPEC §18, M11): naming, schema, date format.

One zipped CSV per trading day, every US listed option, one row per (OPRA root, expiry,
strike) with the call and the put side by side.  Comma-delimited, LF, no quoting, a header row.

**The schema is declared here, column by column; nothing is inferred from a file** (owner's
decision 3, 2026-10-03).  :data:`SCHEMA_VERSIONS` maps a schema version to its ordered
``(column, type)`` pairs; a file whose header is not exactly one of them is schema drift, named
column by column (:func:`describe_drift`), and a new version is added only after the owner
confirms it.  Version 1 is the 39-column layout of the free sample (trade date 2024-01-03): the
36 columns ORATS publishes, in their published order, plus ``cOpra``, ``pOpra`` (OPRA symbols)
and ``cMidIv``.  Version 2 is the older 37-column layout of the archive (from 2007): version 1
without the two OPRA columns, the others in the same order (owner's instruction, 2026-10-04:
the older layouts are accepted).  **The store has one schema**, :data:`STORE_SCHEMA` (version
1's columns): a column a file's layout does not have is null in the store.

Types: ``string``, ``int64``, ``float64`` and ``date`` (``M/D/YYYY`` without zero padding,
parsed with the explicit format :data:`DATE_FORMAT`; anything else fails loudly).  ``divRate``
is ``float64`` although the sample only holds zeros.

**AM or PM settlement of SPX** changes with the period (the vendor's statement, checked on
the data by ``scripts/orats_census.py``; SPEC §18.10).  Where the OPRA symbols exist, ticker
``SPX`` merges the roots ``SPX`` (AM-settled) and ``SPXW`` (PM-settled), which share the
third-Friday expiry dates, and the root (:func:`opra_root`) tells them apart.  Before that the
PM third-Friday series is the ticker ``SPXPM`` (:data:`SPX_PM_TICKER`) and ticker ``SPX`` is AM
on the monthly expiry dates and PM on the others; later the AM monthly is absent and ticker
``SPX`` is PM throughout.  :func:`settlement_without_opra` is that rule for a row without a
symbol; :func:`canonical_expiry` maps the Saturday dates of the old standard expiries to the
last trading day and says whether a date is a monthly expiry.  A day without OPRA symbols is
no longer a failure (owner's instruction, 2026-10-04; it was under decision 8).
"""

from __future__ import annotations

import datetime as _dt
import re
from collections.abc import Sequence

VENDOR = "orats"
#: Default raw file name; group 1 is the trade date ``YYYYMMDD``.  A parameter of ``fetch``
#: and ``verify-raw`` (``--pattern``): the archive's naming is only known after purchase.
DEFAULT_FILE_PATTERN = r"ORATS_SMV_Strikes_(\d{8})\.zip"
#: ``expirDate`` and ``trade_date`` text format (no zero padding in the files; ``strptime``
#: accepts both).
DATE_FORMAT = "%m/%d/%Y"
OPRA_COLUMNS: tuple[str, str] = ("cOpra", "pOpra")
#: The ticker whose OPRA-symbol population the raw manifest counts.
OPRA_REQUIRED_TICKER = "SPX"
#: The ticker of the PM-settled third-Friday SPX series where the file has no OPRA symbols.
SPX_PM_TICKER = "SPXPM"
#: First trade date of the ``SPXPM`` ticker in the archive (the vendor's statement; the census
#: of ``scripts/orats_census.py`` reports the date the data shows).  Before it ticker ``SPX``
#: holds only the AM monthly series.
SPXPM_FIRST_DATE = _dt.date(2011, 10, 4)
#: An OSI symbol is the root followed by ``YYMMDD``, ``C``/``P`` and eight strike digits.
OSI_TAIL = 15

SCHEMA_V1: tuple[tuple[str, str], ...] = (
    ("ticker", "string"),
    ("cOpra", "string"),
    ("pOpra", "string"),
    ("stkPx", "float64"),
    ("expirDate", "date"),
    ("yte", "float64"),
    ("strike", "float64"),
    ("cVolu", "int64"),
    ("cOi", "int64"),
    ("pVolu", "int64"),
    ("pOi", "int64"),
    ("cBidPx", "float64"),
    ("cValue", "float64"),
    ("cAskPx", "float64"),
    ("pBidPx", "float64"),
    ("pValue", "float64"),
    ("pAskPx", "float64"),
    ("cBidIv", "float64"),
    ("cMidIv", "float64"),
    ("cAskIv", "float64"),
    ("smoothSmvVol", "float64"),
    ("pBidIv", "float64"),
    ("pMidIv", "float64"),
    ("pAskIv", "float64"),
    ("iRate", "float64"),
    ("divRate", "float64"),
    ("residualRateData", "float64"),
    ("delta", "float64"),
    ("gamma", "float64"),
    ("theta", "float64"),
    ("vega", "float64"),
    ("rho", "float64"),
    ("phi", "float64"),
    ("driftlessTheta", "float64"),
    ("extVol", "float64"),
    ("extCTheo", "float64"),
    ("extPTheo", "float64"),
    ("spot_px", "float64"),
    ("trade_date", "date"),
)
#: The older layout: version 1 without the OPRA columns.
SCHEMA_V2: tuple[tuple[str, str], ...] = tuple(
    (name, kind) for name, kind in SCHEMA_V1 if name not in OPRA_COLUMNS
)
#: Known schema versions.  Add one only after the owner confirms the drift it answers.
SCHEMA_VERSIONS: dict[int, tuple[tuple[str, str], ...]] = {1: SCHEMA_V1, 2: SCHEMA_V2}
#: The store's one schema: every known layout is a subset of it, in its order.
STORE_SCHEMA: tuple[tuple[str, str], ...] = SCHEMA_V1
STORE_COLUMNS: tuple[str, ...] = tuple(name for name, _ in STORE_SCHEMA)


def schema_columns(version: int) -> tuple[str, ...]:
    return tuple(name for name, _ in SCHEMA_VERSIONS[version])


def schema_version_of(columns: Sequence[str]) -> int | None:
    """The version whose ordered column list equals ``columns``; ``None`` on drift."""
    cols = tuple(columns)
    for version in SCHEMA_VERSIONS:
        if schema_columns(version) == cols:
            return version
    return None


def describe_drift(columns: Sequence[str], version: int = 1) -> str:
    """How ``columns`` differ from schema ``version``: missing, unexpected, reordered."""
    want = schema_columns(version)
    missing = [c for c in want if c not in columns]
    extra = [c for c in columns if c not in want]
    parts = []
    if missing:
        parts.append(f"missing {missing}")
    if extra:
        parts.append(f"unexpected {extra}")
    if not missing and not extra and tuple(columns) != want:
        moved = [c for c, w in zip(columns, want) if c != w]
        parts.append(f"same columns in another order (first difference at {moved[0]!r})")
    if len(set(columns)) != len(columns):
        parts.append("duplicate column names")
    return f"against schema v{version}: " + ("; ".join(parts) if parts else "no difference")


def parse_date(text: str) -> _dt.date:
    """``M/D/YYYY`` → date; raises ``ValueError`` naming the text otherwise."""
    return _dt.datetime.strptime(text, DATE_FORMAT).date()


def file_trade_date(name: str, pattern: str = DEFAULT_FILE_PATTERN) -> _dt.date | None:
    """The trade date a raw file name states (group 1 of ``pattern``, ``YYYYMMDD``); ``None``
    when the name does not match."""
    m = re.fullmatch(pattern, name)
    if m is None:
        return None
    return _dt.datetime.strptime(m.group(1), "%Y%m%d").date()


def opra_root(symbol: str) -> str:
    """OSI root of an OPRA symbol: the symbol without its trailing 15 characters, stripped
    (the sample's symbols are unpadded, 16–20 characters)."""
    return symbol[:-OSI_TAIL].strip()


def good_friday(year: int) -> _dt.date:
    """Good Friday of ``year`` (Gregorian Easter, the anonymous algorithm, minus two days):
    the only exchange holiday that fell on a third Friday before 2026."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    m = (32 + 2 * e + 2 * i - h - k) % 7
    n = (a + 11 * h + 19 * m) // 433
    month = (h + m - 7 * n + 90) // 25
    day = (h + m - 7 * n + 33 * month + 19) % 32
    return _dt.date(year, month, day) - _dt.timedelta(days=2)


def third_friday(year: int, month: int) -> _dt.date:
    first = _dt.date(year, month, 1)
    return first + _dt.timedelta(days=(4 - first.weekday()) % 7 + 14)


def canonical_expiry(expiry: _dt.date) -> tuple[_dt.date, bool]:
    """``(last trading day, is a monthly expiry)`` of a file's ``expirDate``.

    The standard (third-Friday) expiries are dated on the Saturday after the third Friday in
    the older files and on the Friday itself later; a Saturday date is moved to the Friday,
    and a Friday that is Good Friday to the Thursday before it.  The date is a monthly expiry
    when it is the third Friday of its month, the Saturday after it, or the Thursday before a
    third Friday that is Good Friday.  Any other date is returned unchanged."""
    day = expiry - _dt.timedelta(days=1) if expiry.weekday() == 5 else expiry
    friday = third_friday(day.year, day.month)
    holiday = good_friday(day.year)
    if day == friday:
        return (day - _dt.timedelta(days=1) if day == holiday else day), True
    if friday == holiday and expiry == friday - _dt.timedelta(days=1):
        return expiry, True
    return day, False


def settlement_without_opra(
    ticker: str, expiry: _dt.date, trade_date: _dt.date, *, spxpm_listed: bool
) -> str:
    """``"AM"`` or ``"PM"`` of an SPX-family row that has no OPRA symbol (owner's rule,
    2026-10-04): ticker ``SPXPM`` is PM; ticker ``SPX`` on a monthly expiry
    (:func:`canonical_expiry`) is AM while ``SPXPM`` rows exist that day
    (``spxpm_listed``) or before :data:`SPXPM_FIRST_DATE`, PM otherwise."""
    if ticker == SPX_PM_TICKER:
        return "PM"
    if ticker != OPRA_REQUIRED_TICKER:
        raise ValueError(f"{ticker!r} is not an SPX-family ticker")
    _, monthly = canonical_expiry(expiry)
    if monthly and (spxpm_listed or trade_date < SPXPM_FIRST_DATE):
        return "AM"
    return "PM"


def osi_symbol(root: str, expiry: _dt.date, cp: int, strike: float) -> str:
    """The OSI symbol of a contract (root, ``YYMMDD``, ``C``/``P``, the strike in thousandths
    on eight digits), unpadded like the vendor's: what a row without symbols is named by."""
    return f"{root}{expiry:%y%m%d}{'C' if cp == 1 else 'P'}{round(strike * 1000):08d}"
