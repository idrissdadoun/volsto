"""The ORATS "Near End-of-Day" strikes file (SPEC §18, M11): naming, schema, date format.

One zipped CSV per trading day, every US listed option, one row per (OPRA root, expiry,
strike) with the call and the put side by side.  Comma-delimited, LF, no quoting, a header row.

**The schema is declared here, column by column; nothing is inferred from a file** (owner's
decision 3, 2026-10-03).  :data:`SCHEMA_VERSIONS` maps a schema version to its ordered
``(column, type)`` pairs; a file whose header is not exactly one of them is schema drift, named
column by column (:func:`describe_drift`), and a new version is added only after the owner
confirms it.  Version 1 is the 39-column layout of the free sample (trade date 2024-01-03): the
36 columns ORATS publishes, in their published order, plus ``cOpra``, ``pOpra`` (OPRA symbols)
and ``cMidIv``.

Types: ``string``, ``int64``, ``float64`` and ``date`` (``M/D/YYYY`` without zero padding,
parsed with the explicit format :data:`DATE_FORMAT`; anything else fails loudly).  ``divRate``
is ``float64`` although the sample only holds zeros.

The OPRA root (:func:`opra_root`) is the only AM/PM marker in the file: ticker ``SPX`` merges
the roots ``SPX`` (AM-settled) and ``SPXW`` (PM-settled), which share the third-Friday expiry
dates.  A file whose ``cOpra`` or ``pOpra`` is missing or empty on an ``SPX`` row is unusable
for the index importer and is reported loudly (owner's decision 8).
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
#: The ticker whose OPRA symbols must be populated (decision 8).
OPRA_REQUIRED_TICKER = "SPX"
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
#: Known schema versions.  Add one only after the owner confirms the drift it answers.
SCHEMA_VERSIONS: dict[int, tuple[tuple[str, str], ...]] = {1: SCHEMA_V1}


def schema_columns(version: int) -> tuple[str, ...]:
    return tuple(name for name, _ in SCHEMA_VERSIONS[version])


def schema_version_of(columns: Sequence[str]) -> int | None:
    """The version whose ordered column list equals ``columns``; ``None`` on drift."""
    cols = tuple(columns)
    for version in SCHEMA_VERSIONS:
        if schema_columns(version) == cols:
            return version
    return None


def describe_drift(columns: Sequence[str], version: int = max(SCHEMA_VERSIONS)) -> str:
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
