"""Regenerate ``tests/golden/backtest_store_r2.tar.gz``: the toy backtest store in the **flat
layout** of the round-2 ``volsto-backtest`` (2026-09-16; ``dates/<d>/{rows.parquet, fit.json,
done.json}`` and ``backtest.json``, version-1 dependency records) — the legacy store the
migration tests read (``tests/test_backtest.py::flat_store``).

The golden must be computed by the current importer and calibration code, so it is regenerated
whenever those change (first: the fit's θ fix of 2026-09-17, SPEC §13.1).  Since 2026-10-03 it
carries the snapshots its records name (``snapshots/spx_<date>.yaml``, the store's own
location for them), so the snapshot binding is verified against stored bytes on any machine.
**Its leverage keys are still those of the machine that built it** (a key hashes the fitted
parameters, which another machine reproduces to 1e-8 only, and a toy leverage is 8.6 MB — too
large to ship): on another machine the migration tests fail on the leverage until fitted
parameters are stored records (CONTRIBUTING.md, "Machine-dependent arithmetic"); regenerate the
golden there.  Last regenerated on the owner's Mac (M5 Pro, macOS 26.5.1), 2026-10-03.  Source: an
attempts-layout toy build written by the sanctioned fixture (``tests/_backtest_build.py``, e.g.
the last pytest run's ``<tmp>/pytest-<n>/toy_backtest/A/outputs/backtest/hdn_2022h2_toy``).
Nothing is calibrated here.

For each date the current attempt's files are written as round 2 wrote them, the previous
golden being the schema: ``rows.parquet`` without the columns added since (the cumulative P&L
and its stderr, the paired group stderrs, ``residual_paired`` — the migration test asserts a
round-2 store carries no cumulative error), ``fit.json`` without the keys added since (``close``, timing keys),
``done.json`` without the top-level keys added since (``previous_cache_key``,
``previous_date``, ``volsto_version``) and with a version-1 record (no leverage content
digest; the file digests recomputed; version-1 state links from the library's own
:func:`~volsto.studies.backtest.state_link`, checked by reproducing the stored version-2 links
and file digests from the same inputs).  Columns, keys and file names are asserted against the
previous golden.

Usage::

    .venv/bin/python tests/golden/make_backtest_store_r2.py <attempts-layout toy store>
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path
from typing import Any

import pandas as pd

from volsto.studies.backtest import snapshot_bytes_digest, state_link, state_params

GOLDEN = Path(__file__).resolve().parent / "backtest_store_r2.tar.gz"
SNAPSHOTS = "snapshots/"
DROPPED_TOP_LEVEL = ("previous_cache_key", "previous_date", "volsto_version")


def _members(tar: tarfile.TarFile) -> dict[str, bytes]:
    out: dict[str, bytes] = {}
    for m in tar.getmembers():
        if m.isfile():
            out[m.name] = tar.extractfile(m).read()  # type: ignore[union-attr]
    return out


def main(argv: list[str]) -> int:
    store = Path(argv[1])
    with tarfile.open(GOLDEN) as tar:
        old = _members(tar)
    dates = sorted(p.parent.name for p in store.glob("dates/*/CURRENT"))
    assert dates == sorted({n.split("/")[1] for n in old if n.startswith("dates/")}), dates
    new: dict[str, bytes] = {"backtest.json": (store / "backtest.json").read_bytes()}
    old_bt = json.loads(old["backtest.json"])
    assert sorted(json.loads(new["backtest.json"])) == sorted(old_bt), "backtest.json keys"
    # state of every date (from the attempts layout), for the version-1 links
    state: dict[str, tuple[str, dict[str, float], str, str]] = {}
    docs: dict[str, dict[str, Any]] = {}
    for d in dates:
        attempt = json.loads((store / "dates" / d / "CURRENT").read_text())["attempt"]
        adir = store / "dates" / d / "attempts" / attempt
        doc = json.loads((adir / "done.json").read_text())
        assert doc["status"] == "ok" and doc["record"]["version"] == 2, (d, doc["status"])
        fit_bytes = (adir / "fit.json").read_bytes()
        rows_bytes = (adir / "rows.parquet").read_bytes()
        fit = json.loads(fit_bytes)
        lev = doc["record"]["leverage"]
        state[d] = (
            doc["record"]["snapshot"],
            state_params(fit["params"]),
            lev["key"],
            lev["content"],
        )
        # the stored digests are those of the attempt's files (checked before they are recomputed)
        assert doc["record"]["fit"] == hashlib.sha256(fit_bytes).hexdigest(), d
        assert doc["record"]["rows"]["sha256"] == hashlib.sha256(rows_bytes).hexdigest(), d
        # rows: the round-2 columns only
        old_rows = pd.read_parquet(io.BytesIO(old[f"dates/{d}/rows.parquet"]))
        rows = pd.read_parquet(io.BytesIO(rows_bytes))
        assert set(old_rows) <= set(rows), sorted(str(c) for c in set(old_rows) - set(rows))
        rows = rows[list(old_rows.columns)]
        assert list(rows.dtypes.astype(str)) == list(old_rows.dtypes.astype(str)), d
        buf_rows = io.BytesIO()
        rows.to_parquet(buf_rows, index=False)
        new[f"dates/{d}/rows.parquet"] = buf_rows.getvalue()
        # fit: the round-2 keys only
        old_fit = json.loads(old[f"dates/{d}/fit.json"])
        assert set(old_fit) <= set(fit), (d, sorted(set(old_fit) - set(fit)))
        fit = {k: fit[k] for k in old_fit}  # the round-2 keys only (e.g. no ``close``)
        for k, v in fit.items():
            if isinstance(v, dict) and isinstance(old_fit[k], dict):
                assert set(old_fit[k]) <= set(v), (d, k)
                fit[k] = {kk: vv for kk, vv in v.items() if kk in old_fit[k]}
        new[f"dates/{d}/fit.json"] = (json.dumps(fit, indent=2, sort_keys=True) + "\n").encode()
        docs[d] = doc
    for d in dates:
        doc = docs[d]
        rec = dict(doc["record"])
        links_v1: dict[str, str] = {}
        for prev, stored_v2 in rec["links"].items():
            snap, params, key, content = state[prev]
            assert state_link(prev, snap, params, key, content, version=2) == stored_v2, (d, prev)
            links_v1[prev] = state_link(prev, snap, params, key, version=1)
        rec["links"] = links_v1
        rec["version"] = 1
        rec["fit"] = hashlib.sha256(new[f"dates/{d}/fit.json"]).hexdigest()
        rec["rows"] = {
            "n": len(pd.read_parquet(io.BytesIO(new[f"dates/{d}/rows.parquet"]))),
            "sha256": hashlib.sha256(new[f"dates/{d}/rows.parquet"]).hexdigest(),
        }
        assert rec["rows"]["n"] == doc["record"]["rows"]["n"] == doc["n_rows"], d
        rec["leverage"] = {k: v for k, v in rec["leverage"].items() if k != "content"}
        flat = {k: v for k, v in doc.items() if k not in DROPPED_TOP_LEVEL}
        flat["record"] = rec
        old_doc = json.loads(old[f"dates/{d}/done.json"])
        assert sorted(flat) == sorted(old_doc), (d, sorted(set(flat) ^ set(old_doc)))
        assert sorted(rec) == sorted(old_doc["record"]), d
        assert sorted(rec["leverage"]) == sorted(old_doc["record"]["leverage"]), d
        new[f"dates/{d}/done.json"] = (json.dumps(flat, indent=2, sort_keys=True) + "\n").encode()
    assert sorted(new) == sorted(n for n in old if not n.startswith(SNAPSHOTS)), sorted(
        set(new) ^ set(old)
    )
    # the snapshots the records name, byte for byte (2026-10-03): the migration tests verify the
    # stored records against these stored bytes, never against an import on the current machine
    for d in dates:
        snap = store / "snapshots" / f"spx_{d}.yaml"
        data = snap.read_bytes()
        assert snapshot_bytes_digest(data) == docs[d]["record"]["snapshot"], d
        new[f"{SNAPSHOTS}spx_{d}.yaml"] = data
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz", compresslevel=9) as tar:
        for name in sorted(new):
            info = tarfile.TarInfo(name)
            info.size = len(new[name])
            info.mtime = 0
            tar.addfile(info, io.BytesIO(new[name]))
    GOLDEN.write_bytes(buf.getvalue())
    print(f"wrote {GOLDEN} ({len(new)} files, {len(buf.getvalue())} bytes) from {store}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
