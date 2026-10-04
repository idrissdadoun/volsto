"""Vendor data layer (SPEC §18, milestone M11): the raw files exactly as a vendor delivers
them, the typed Parquet store derived from them, and the ``volsto-data`` CLI that fetches,
verifies and converts.  Nothing here fits, calibrates or prices; nothing here handles
credentials (``fetch`` takes the name of an AWS profile the owner configured).

* :mod:`volsto.data.roots` — the two documented roots (``VOLSTO_DATA_RAW``,
  ``VOLSTO_DATA_STORE``) and the free-space check every bulk write goes through.
* :mod:`volsto.data.orats` — the ORATS "Near End-of-Day" strikes file: naming, the schema
  declared column by column, date format.
* :mod:`volsto.data.raw` — ``verify-raw``: the raw manifest and the trading-calendar check.
* :mod:`volsto.data.fetch` — ``fetch``: a wrapper around ``aws s3 sync``.
* :mod:`volsto.data.cli` — the ``volsto-data`` entry point.
"""
