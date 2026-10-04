"""``volsto-data sql``: an optional DuckDB convenience over the store (SPEC §18.4, M11 Part 2b).

``duckdb`` belongs to the ``data`` extra only and is imported here, inside the function: the
library core (and every other ``volsto-data`` command) never imports it
(``tests/test_data_store.py::test_core_does_not_import_duckdb``).  The query sees one view,
``strikes``: every day file of the store, the vendor's columns plus ``year`` from the
directory names.  Read-only: an in-memory database, nothing is written to the store.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from volsto.data import store as storemod
from volsto.data.roots import DataError

#: Rows printed by default (``--max-rows``).
DEFAULT_MAX_ROWS: int = 100


def run_sql(store_dir: Path, query: str) -> Any:
    """Run ``query`` against the ``strikes`` view; returns a pandas DataFrame."""
    try:
        import duckdb
    except ImportError as exc:
        raise DataError(
            'volsto-data sql needs DuckDB: pip install -e ".[data]" (the data extra)'
        ) from exc
    strikes = store_dir / storemod.STRIKES_DIR
    if not any(strikes.rglob("*.parquet")):
        raise DataError(f"no Parquet file under {strikes}: run volsto-data convert first")
    glob = str(strikes / "*" / "[0-9]*.parquet").replace("'", "''")
    con = duckdb.connect(":memory:")
    try:
        con.execute(
            f"create view strikes as select * from read_parquet('{glob}', hive_partitioning = true)"
        )
        return con.execute(query).fetch_df()
    except duckdb.Error as exc:
        raise DataError(f"DuckDB: {exc}") from exc
    finally:
        con.close()
