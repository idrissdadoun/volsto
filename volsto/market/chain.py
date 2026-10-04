"""The canonical option chain: the one frame every vendor loader produces and the importer's
vendor-independent steps consume (SPEC §18.6, M11 Part 4).

``volsto.market.import_hdn.load_day`` (HistoricalData.net) and
``volsto.market.import_orats.load_day`` (ORATS, from the Parquet store) both return this
frame; :func:`~volsto.market.import_hdn.implied_forwards`,
:func:`~volsto.market.import_hdn.to_grid_surface`, :func:`~volsto.market.import_hdn.fit_ssvi`,
:func:`~volsto.market.import_hdn.sabrw_fits` and the snapshot writer read nothing else.

**One row per contract** (a call or a put at one strike of one slice), only contracts whose
own bid and ask are both positive and whose time to expiry is positive.

Columns (:data:`CHAIN_COLUMNS`), units in brackets:

``contract``          OSI symbol of the contract (text)
``root``              OPRA root (text): the settlement marker for an index (SPX AM, SPXW PM)
``expiration``        expiry date, ISO text
``expiry``            slice key ``"<root>|<expiration>"`` — slices are grouped by (root,
                      expiration), because two roots can share an expiry date with different T
``quote_date``        trade date, ISO text
``cp``                +1 call, −1 put
``strike``            strike [index points]
``bid``, ``ask``, ``mid``   prices [index points]; ``mid`` = ½(bid + ask)
``T``                 time to expiry [years]: calendar days / 365, one day less for AM
                      settlement (:func:`volsto.market.import_hdn.time_to_expiry`)
``r``                 the vendor's prior rate at ``T`` [per year, continuous]: a cross-check and
                      a fallback only — payoffs are discounted on the option-implied funding
                      curve
``df``                ``exp(−r T)``
``underlying_close``  the official close of the underlying on the trade date [index points]:
                      the fixing reference and the centre of the near-the-money band, never
                      the pricing spot (that is the option-implied spot)
``iv_bid``, ``iv_ask``   the vendor's one-sided implied vols where the loader trusts them
                      for the spread filter, else NaN — then the bid and ask prices are
                      inverted with our implied forward and discount factor
``iv``, ``iv_flag``   the vendor's mid implied vol and its quality flag: cross-checks only,
                      copied into the retained-quote table (NaN where the vendor has none)

``attrs`` (:data:`CHAIN_ATTRS`): ``quote_date``; ``underlying``; ``file`` (the vendor file the
day came from); ``rate_tenors``, ``rate_zeros`` (the vendor's prior rate curve, reported
beside the funding curve); ``spot`` (the official close).  Optional: ``vendor`` and
``spot_async_check`` (default true: an implied spot far from the close is logged as a quote
asynchrony — the HDN case, a 16:00 close against 16:15 quotes; a loader whose snapshot
precedes the close sets it false and the offset is a measurement, not a warning).

A loader may add columns (vendor cross-checks); the vendor-independent steps ignore them.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pandas as pd

CHAIN_COLUMNS: Final[tuple[str, ...]] = (
    "contract",
    "root",
    "expiration",
    "expiry",
    "quote_date",
    "cp",
    "strike",
    "bid",
    "ask",
    "mid",
    "T",
    "r",
    "df",
    "underlying_close",
    "iv_bid",
    "iv_ask",
    "iv",
    "iv_flag",
)
CHAIN_ATTRS: Final[tuple[str, ...]] = (
    "quote_date",
    "underlying",
    "file",
    "rate_tenors",
    "rate_zeros",
    "spot",
)


def validate_chain(chain: pd.DataFrame) -> pd.DataFrame:
    """Raise ``ValueError`` unless ``chain`` is a canonical chain (module docstring); returns
    it.  Checked: the columns and attrs are present, one quote date, both quotes positive,
    ``cp`` is ±1, ``T`` is positive and single-valued per slice, ``mid`` is the quote mid, and
    no contract appears twice."""
    missing = [c for c in CHAIN_COLUMNS if c not in chain.columns]
    if missing:
        raise ValueError(f"not a canonical chain: columns missing {missing}")
    absent = [a for a in CHAIN_ATTRS if a not in chain.attrs]
    if absent:
        raise ValueError(f"not a canonical chain: attrs missing {absent}")
    if chain.empty:
        raise ValueError("not a canonical chain: no rows")
    if chain["quote_date"].nunique() != 1:
        raise ValueError("not a canonical chain: more than one quote date")
    if not ((chain["bid"] > 0) & (chain["ask"] > 0)).all():
        raise ValueError("not a canonical chain: a contract without both quotes")
    if not chain["cp"].isin([1, -1]).all():
        raise ValueError("not a canonical chain: cp must be +1 or -1")
    if not (chain["T"] > 0).all():
        raise ValueError("not a canonical chain: non-positive time to expiry")
    if (chain.groupby("expiry")["T"].nunique() != 1).any():
        raise ValueError("not a canonical chain: a slice with more than one T")
    if not np.array_equal(chain["mid"].to_numpy(), 0.5 * (chain["bid"] + chain["ask"]).to_numpy()):
        raise ValueError("not a canonical chain: mid is not the quote mid")
    if not (chain["expiry"] == chain["root"] + "|" + chain["expiration"]).all():
        raise ValueError("not a canonical chain: expiry is not '<root>|<expiration>'")
    if chain["contract"].duplicated().any():
        raise ValueError("not a canonical chain: a contract appears twice")
    return chain
