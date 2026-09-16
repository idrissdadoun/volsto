"""Results store of the viewer precompute (SPEC §9.2, M9 Part 1): parquet tables under
``<root>/results/`` plus a JSON manifest, written per grid point and merged on read.

Layout (everything relative to ``root``; no absolute path is ever stored, so a store computed on
a VM is rsync'd next to its leverage cache and read unchanged — ``tests/test_precompute.py::
test_toy_precompute_end_to_end`` moves one)::

    root/results/points/<point dir>/<table>.parquet   # points, products, forward_smile,
                                                       # forward_vols, ssr, varv, risk
    root/results/points/<point dir>/manifest.json      # the point's manifest entry
    root/results/runs/<utc>_<shard>.json               # one record per CLI run
    root/results/manifest.json                         # merged view (rewritten by each run)
    root/results/<table>.parquet                       # optional compaction (compact())

**Append safety.** Concurrent shards (``--shard i/n`` on several processes or machines) never
touch the same file: every point writes its own directory (each file written to a temporary
name and renamed, so a reader never sees a half-written parquet) and its own manifest entry;
the merged tables are built on read (:meth:`ResultsStore.table` concatenates the per-point files,
then the rows of a compacted table whose point directory is gone).  This per-point-files +
merge-on-read design is the simplest one that cannot corrupt under concurrency; ``compact()`` is
an optional acceleration for big stores and keeps the per-point files unless told otherwise.

Tables (every Monte Carlo quantity has a ``<name>_stderr`` twin; ``point_id`` joins them):

* ``points`` — one row per grid point: ``point_id, label, surface, mode, status, nu, theta, k1,
  k2, rho12, rho_SX1, rho_SX2`` (NaN for the LV point), the axis values (``axis_nu, axis_rho,
  axis_kappa, ssr_target, skew_eps``), ``cache_key, n_particles, horizon, calibrated_this_run,
  calibration_seconds`` (the leverage's measured wall time, from the cache on a hit),
  ``mean_abs_L_minus_1`` (the M7 convention :func:`~volsto.calibration.fit_2f.
  mean_abs_leverage_deviation`: ``|k| ≤ 2`` ATM sd per slice), ``max_abs_error_vp, max_z``
  (from the cache's :class:`~volsto.calibration.diagnostics.CalibrationReport` when present,
  else NaN),
  ``wall_seconds`` (total, split in ``wall_calibration_s, wall_diagnostics_s, wall_pricing_s,
  wall_analytics_s, wall_risk_s``), ``risk_tier, pricing_n_paths, pricing_seed, git_commit,
  code_tag, created_utc`` (first write), ``updated_utc`` / ``updated_steps`` when a later
  ``--resume`` refreshed the risk or diagnostics step only (:func:`volsto.viewers.precompute.
  refresh_point`), and for marking points ``fit_status, fit_messages, fit_mean_skew_gap,
  fit_wall_seconds`` and the ``skew_gap_*`` columns (deterministic fit outputs, no stderr:
  :data:`NON_MC_PREFIXES`).
* ``products`` — ``point_id, product, quantity, key, value, value_stderr, unit``: the M4 headline
  columns (``key`` = the :func:`volsto.studies.m4.run_headline` column, the model-risk page
  matches the M4 baselines on it) and the M6 note cells (``key = "<product>:<column>"``, the
  :func:`volsto.studies.m6.baseline_values` key).
* ``forward_smile`` — ``point_id, t1, t2, beyond_horizon, strike_moneyness, log_moneyness, cp,
  iv, iv_stderr, price, price_stderr`` at 1y → 2y and 2y → 3y on the headline strikes;
  ``beyond_horizon`` (bool) flags a window whose ``t2`` exceeds the calibration horizon — priced
  on the last leverage slice held constant (``volsto.models.leverage``), so the pages must flag
  it (the toy grid's 1y horizon: the 2y → 3y window).
* ``forward_vols`` — ``point_id, t1, t2, beyond_horizon, fwd_atm_vol, fwd_vs, fwd_volswap`` (+
  ``_stderr``).
* ``ssr`` — ``point_id, T, ssr_lsv, ssr_lsv_stderr, skew_lsv, skew_lsv_stderr, slope,
  slope_stderr, atmf_vol, atmf_vol_stderr, eps, ssr_naked_first_order`` (the fit's first-order
  P1 SSR at the pillar for marking points, NaN elsewhere), ``ssr_target`` (marking points).
* ``varv`` — ``point_id, T, term, value, value_stderr, kind`` (``kind`` ``mc`` or
  ``closed_form`` — closed forms are exact, stderr 0).
* ``risk`` — ``point_id, product, tier, group, name, value, value_stderr, unit, size, scheme,
  states, T, bucket, regime, variant, key`` from the :class:`~volsto.risk.report.RiskReport`.

Checked by ``tests/test_precompute.py`` (``test_toy_precompute_end_to_end``,
``test_mc_columns_without_stderr_on_marking_row``, ``test_store_reader_on_missing_store``).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from volsto.calibration.cache import atomic_write

TABLES: tuple[str, ...] = (
    "points",
    "products",
    "forward_smile",
    "forward_vols",
    "ssr",
    "varv",
    "risk",
)
#: Numeric columns that are not Monte Carlo estimates and carry no ``_stderr`` twin.
NON_MC_COLUMNS: frozenset[str] = frozenset(
    {
        "t1",
        "t2",
        "T",
        "strike_moneyness",
        "log_moneyness",
        "cp",
        "size",
        "eps",
        "ssr_target",
        "ssr_naked_first_order",
        "bucket_lo",
        "bucket_hi",
        "n_particles",
        "horizon",
        "nu",
        "theta",
        "k1",
        "k2",
        "rho12",
        "rho_SX1",
        "rho_SX2",
        "axis_nu",
        "axis_rho",
        "axis_kappa",
        "skew_eps",
        "calibration_seconds",
        "mean_abs_L_minus_1",
        "max_abs_error_vp",
        "max_z",
        "wall_seconds",
        "wall_calibration_s",
        "wall_diagnostics_s",
        "wall_pricing_s",
        "wall_analytics_s",
        "wall_risk_s",
        "pricing_n_paths",
        "pricing_seed",
        "fit_mean_skew_gap",
        "fit_wall_seconds",
    }
)
#: Prefixes of numeric columns that are deterministic outputs of the P1 marking fit (no Monte
#: Carlo, no ``_stderr`` twin): the ``skew_gap_<name>@<T>y`` naked-skew gaps at the constraint
#: pillars (dynamic names, :func:`volsto.viewers.grid.marking_summary`) and the ``fit_*``
#: summary columns.
NON_MC_PREFIXES: tuple[str, ...] = ("skew_gap_", "fit_")

_SAFE = re.compile(r"[^A-Za-z0-9_.=+-]")


def point_dirname(point_id: str) -> str:
    """Filesystem-safe directory name of a point id (``:`` → ``__``, anything else odd → ``_``)."""
    return _SAFE.sub("_", point_id.replace(":", "__"))


def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds")


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write ``df`` as parquet through the cache's atomic writer (temporary beside the target,
    fsync, rename, directory fsync) — the one write rule of the project's on-disk stores."""
    atomic_write(path, lambda p: df.to_parquet(p, index=False))


def _write_json_atomic(data: Mapping[str, Any], path: Path) -> None:
    """Write ``data`` to ``path`` through the cache's atomic writer (a temporary beside the target
    created with the umask's mode, fsync, rename, directory fsync): a reader never sees a torn
    file and the file keeps the usual 0644 mode, which ``mkstemp``'s 0600 did not."""
    text = json.dumps(data, indent=1, sort_keys=True, default=str)
    atomic_write(path, lambda p: p.write_text(text, encoding="utf-8"))


@dataclass
class PointResult:
    """Everything one grid point writes: its ``points`` row, the per-table frames (a missing table
    is an empty frame) and its manifest entry."""

    point_id: str
    row: dict[str, Any]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)


class ResultsStore:
    """Directory-backed results store (module docstring).  Constructing one creates nothing on
    disk — the read paths (``StoreReader``, the read API, ``volsto-viewer --check``,
    ``volsto-precompute --dry-run``) leave a missing root missing; ``write_point``,
    ``write_run``, ``refresh_manifest`` and ``compact`` create the directories they write to
    (checked by ``tests/test_viewers_app.py::test_viewer_check_cli`` on an empty triple)."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.results = self.root / "results"
        self.points_dir = self.results / "points"
        self.runs_dir = self.results / "runs"
        # No directory is created here: a reader (``StoreReader``, the read API, ``--check``,
        # ``--dry-run``) must leave the filesystem untouched; the writers create what they need.

    # -- addressing ----------------------------------------------------------------------------

    def point_dir(self, point_id: str) -> Path:
        return self.points_dir / point_dirname(point_id)

    def has_point(self, point_id: str) -> bool:
        return (self.point_dir(point_id) / "points.parquet").exists()

    def point_ids(self) -> list[str]:
        """Ids of the points present (from their ``points.parquet`` rows), sorted."""
        ids: list[str] = []
        for d in sorted(self.points_dir.iterdir()) if self.points_dir.exists() else []:
            p = d / "points.parquet"
            if p.exists():
                ids.extend(str(x) for x in pd.read_parquet(p, columns=["point_id"])["point_id"])
        return sorted(ids)

    # -- write ---------------------------------------------------------------------------------

    def write_point(self, result: PointResult) -> Path:
        """Write a point's tables (atomically, one parquet per table) and its manifest entry;
        an existing point directory is overwritten table by table."""
        d = self.point_dir(result.point_id)
        d.mkdir(parents=True, exist_ok=True)
        row = dict(result.row)
        row["point_id"] = result.point_id
        _write_atomic(pd.DataFrame([row]), d / "points.parquet")
        for name in TABLES[1:]:
            df = result.tables.get(name)
            path = d / f"{name}.parquet"
            if df is None or df.empty:
                if path.exists():
                    path.unlink()
                continue
            out = df.copy()
            out.insert(0, "point_id", result.point_id)
            _write_atomic(out, path)
        entry = {"point_id": result.point_id, **result.manifest}
        _write_json_atomic(entry, d / "manifest.json")
        return d

    def read_point(self, point_id: str) -> PointResult:
        """The stored :class:`PointResult` of a point (its row without ``point_id``, its tables
        without the ``point_id`` column, its manifest entry) — what a partial refresh
        (:func:`volsto.viewers.precompute.refresh_point`) rewrites; ``KeyError`` when absent."""
        d = self.point_dir(point_id)
        if not (d / "points.parquet").exists():
            raise KeyError(point_id)
        row = {str(k): v for k, v in pd.read_parquet(d / "points.parquet").iloc[0].items()}
        row.pop("point_id", None)
        tables: dict[str, pd.DataFrame] = {}
        for name in TABLES[1:]:
            path = d / f"{name}.parquet"
            if path.exists():
                tables[name] = pd.read_parquet(path).drop(columns=["point_id"])
        manifest: dict[str, Any] = {}
        if (d / "manifest.json").exists():
            manifest = json.loads((d / "manifest.json").read_text())
            manifest.pop("point_id", None)
        return PointResult(point_id, row, tables, manifest)

    def write_run(self, record: Mapping[str, Any]) -> Path:
        """Record one CLI run (``results/runs/<utc>_<shard>_<pid>.json``) and refresh the merged
        ``results/manifest.json``.  The stamp has second resolution, so a second run of the same
        shard in the same second (a ``--resume`` right after the build) takes a ``-NN`` suffix
        instead of overwriting the earlier record; the suffix is zero-padded so the records keep
        their chronological order under a plain name sort past the ninth."""
        stamp = utc_now().replace(":", "").replace("+0000", "Z")
        shard = str(record.get("shard", "1/1")).replace("/", "of")
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{stamp}_{shard}_{os.getpid()}"
        path = self.runs_dir / f"{stem}.json"
        n = 1
        while path.exists():
            path = self.runs_dir / f"{stem}-{n:02d}.json"
            n += 1
        _write_json_atomic(dict(record), path)
        self.refresh_manifest()
        return path

    def refresh_manifest(self) -> Path:
        """Rewrite ``results/manifest.json`` from the per-point entries and the run records."""
        self.results.mkdir(parents=True, exist_ok=True)
        path = self.results / "manifest.json"
        _write_json_atomic(self.manifest(), path)
        return path

    def compact(self, *, remove_point_files: bool = False) -> dict[str, Path]:
        """Merge the per-point files into ``results/<table>.parquet`` (optional acceleration;
        the per-point files stay unless ``remove_point_files``)."""
        out: dict[str, Path] = {}
        self.results.mkdir(parents=True, exist_ok=True)
        for name in TABLES:
            df = self.table(name)
            path = self.results / f"{name}.parquet"
            _write_atomic(df, path)
            out[name] = path
        if remove_point_files and self.points_dir.exists():
            for d in self.points_dir.iterdir():
                for name in TABLES:
                    p = d / f"{name}.parquet"
                    if p.exists():
                        p.unlink()
        return out

    # -- read ----------------------------------------------------------------------------------

    def _point_frames(self, name: str) -> Iterable[pd.DataFrame]:
        if not self.points_dir.exists():
            return
        for d in sorted(self.points_dir.iterdir()):
            p = d / f"{name}.parquet"
            if p.exists():
                yield pd.read_parquet(p)

    def table(self, name: str) -> pd.DataFrame:
        """The merged table ``name``: per-point files first, then the compacted rows of points
        whose directory is gone (empty frame when nothing is stored)."""
        if name not in TABLES:
            raise KeyError(f"unknown table {name!r}; tables are {TABLES}")
        frames = list(self._point_frames(name))
        compacted = self.results / f"{name}.parquet"
        if compacted.exists():
            c = pd.read_parquet(compacted)
            present = set(self.point_ids()) if frames else set()
            if not c.empty:
                frames.append(c[~c["point_id"].isin(present)])
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)

    def point_manifests(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        if not self.points_dir.exists():
            return out
        for d in sorted(self.points_dir.iterdir()):
            p = d / "manifest.json"
            if p.exists():
                entry = json.loads(p.read_text())
                out[str(entry["point_id"])] = entry
        return out

    def runs(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for p in sorted(self.runs_dir.glob("*.json")) if self.runs_dir.exists() else []:
            out.append(json.loads(p.read_text()))
        return out

    def manifest(self) -> dict[str, Any]:
        """``{"grid", "code_tag", "git_commit", "n_particles", "pricing", "risk_tier", "runs",
        "points": {id: entry}}`` — the grid header from the latest run record, the points from
        their own entries."""
        runs = self.runs()
        head: dict[str, Any] = {}
        if runs:
            last = runs[-1]
            head = {
                k: last.get(k)
                for k in (
                    "grid",
                    "grid_path",
                    "code_tag",
                    "git_commit",
                    "n_particles",
                    "pricing",
                    "risk_tier",
                )
            }
        return {**head, "runs": runs, "points": self.point_manifests()}


class StoreReader:
    """Typed read-only accessors over a :class:`ResultsStore` (the read API of M9 Part 3 sits on
    these); every method returns a DataFrame or a plain dict, never a path."""

    def __init__(self, root: str | Path) -> None:
        self.store = ResultsStore(root)

    def points(self) -> pd.DataFrame:
        return self.store.table("points")

    def point(self, point_id: str) -> dict[str, Any]:
        df = self.points()
        if df.empty or point_id not in set(df["point_id"]):
            raise KeyError(point_id)
        rec = df[df["point_id"] == point_id].iloc[0].to_dict()
        return {str(k): v for k, v in rec.items()}

    def _filtered(self, name: str, point_id: str | None) -> pd.DataFrame:
        df = self.store.table(name)
        if point_id is None or df.empty:
            return df
        return df[df["point_id"] == point_id].reset_index(drop=True)

    def products(self, point_id: str | None = None) -> pd.DataFrame:
        return self._filtered("products", point_id)

    def forward_smile(self, point_id: str | None = None) -> pd.DataFrame:
        return self._filtered("forward_smile", point_id)

    def forward_vols(self, point_id: str | None = None) -> pd.DataFrame:
        return self._filtered("forward_vols", point_id)

    def ssr(self, point_id: str | None = None) -> pd.DataFrame:
        return self._filtered("ssr", point_id)

    def varv(self, point_id: str | None = None) -> pd.DataFrame:
        return self._filtered("varv", point_id)

    def risk(self, point_id: str | None = None, product: str | None = None) -> pd.DataFrame:
        df = self._filtered("risk", point_id)
        if product is not None and not df.empty:
            df = df[df["product"] == product].reset_index(drop=True)
        return df

    def manifest(self) -> dict[str, Any]:
        return self.store.manifest()


def mc_columns_without_stderr(df: pd.DataFrame) -> list[str]:
    """Numeric columns of a store table that are Monte Carlo quantities (not in
    :data:`NON_MC_COLUMNS`, not under a :data:`NON_MC_PREFIXES` prefix, not themselves a
    ``_stderr``, not boolean) lacking a ``<name>_stderr`` twin — the ``value`` / ``value_stderr``
    pair of the long tables counts as paired."""
    bad: list[str] = []
    for c in df.columns:
        name = str(c)
        if not pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]):
            continue
        if name.endswith("_stderr") or name in NON_MC_COLUMNS or name == "point_id":
            continue
        if name.startswith(NON_MC_PREFIXES):
            continue
        if f"{name}_stderr" not in df.columns:
            bad.append(name)
    return bad
