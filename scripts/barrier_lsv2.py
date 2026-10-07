"""LSV passes of the barrier study after addendum 1 (BARRIER_STUDY_ADDENDUM §2, §3, §5.2).

The particle estimator comes from ``outputs/interview/gate.json`` (written by
``scripts/barrier_gate.py``): ``"binned"`` or ``null`` (the library's default path).  Every
output file records it.

    python scripts/barrier_lsv2.py entries [--monthly] [--workers 10]
    python scripts/barrier_lsv2.py daily --start 2021-06-01 --end 2026-10-02 [--workers 14]

``entries``  per entry date → ``barrier_results/lsv2/<date>.parquet``: the ``ssr12`` mark on
             every date, ``ssr10`` and ``ssr15`` on the monthly subset (the first entry of each
             month); knock-out premiums with standard errors, knock probabilities, the
             forward-start smile, and the bucket statistics of §5.2 under each rule.
``daily``    per trading day → ``barrier_results/lsv2_daily/<date>.parquet``: ``ssr12`` marks
             and bumped marks of every live cell (the pilot's pass with the gate's estimator).

Resumable (a date whose file exists is skipped); files are written to a temporary name and
renamed.  New store: the pre-addendum ``lsv_entries/`` and ``lsv_daily/`` are never written.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barrier_history as run  # noqa: E402

from volsto.studies import barrier_history as bh  # noqa: E402

GATE = bh.OUT / "gate.json"
LSV2 = bh.RESULTS / "lsv2"
LSV2_DAILY = bh.RESULTS / "lsv2_daily"
PARTICLES = 100_000
HORIZON = 1.03


def gate_estimator() -> str | None:
    """The estimator the gate selected (``None`` = the library's default path); ``None`` too
    when there is no gate file (the sorted path is the fallback of addendum §2.5)."""
    if not GATE.exists():
        return None
    value = json.loads(GATE.read_text()).get("estimator")
    return str(value) if value else None


def lsv_model(
    surface: Any,
    params: dict[str, float],
    *,
    seed: int,
    estimator: str | None,
    n_particles: int = PARTICLES,
    min_window: int = 0,
) -> Any:
    """The LSV model of ``params``, its leverage calibrated on ``surface`` with the particle
    method (``estimator`` passed to ``ParticleConfig`` only when set: the default path is the
    library's own).  Nothing is persisted."""
    from volsto.calibration.particle import calibrate_leverage
    from volsto.config import BergomiParams, ParticleConfig, SimConfig
    from volsto.market.varswap import xi0_curve
    from volsto.models.bergomi import BergomiSV
    from volsto.models.lsv import LSV

    extra: dict[str, Any] = {"estimator": estimator} if estimator else {}
    fc = surface.forward_curve
    kernel = BergomiSV(
        BergomiParams(**params), xi0_curve(surface, min(surface.max_maturity, 5.0)), fc
    )
    particle = ParticleConfig(
        n_particles=n_particles, horizon=HORIZON, min_window=min_window, **extra
    )
    result = calibrate_leverage(
        surface, kernel, particle, SimConfig(n_paths=n_particles, seed=seed)
    )
    return LSV(kernel, result.leverage)


def lsv_marks(
    surf: Any,
    params: dict[str, float],
    cells: pd.DataFrame,
    days: list[str],
    times: np.ndarray,
    values: np.ndarray,
    df: np.ndarray,
    *,
    seed: int,
    estimator: str | None,
) -> tuple[pd.DataFrame, list[float]]:
    """Marks and bumped marks of ``cells`` under the LSV (the pilot's ``lsv_marks`` with the
    estimator switch) and the seconds of the three calibrations."""
    i6 = bh.STRUCTURES.index("B6")
    out = pd.DataFrame(index=cells.index)
    base = None
    seconds = []
    for r_i, (tag, ratio) in enumerate(
        zip(("", "_up", "_dn"), (1.0, 1.0 + bh.BUMP, 1.0 - bh.BUMP), strict=True)
    ):
        s = surf if ratio == 1.0 else bh.MovedSurface(surf, ratio)
        t0 = time.perf_counter()
        model = lsv_model(s, params, seed=seed, estimator=estimator)
        seconds.append(time.perf_counter() - t0)
        ko = run._ko_table(model, times, seed, cells, days, values[r_i][:, i6], df, base)
        if base is None:
            base = ko
            for k in ("a1_se", "a2_se", "p1", "p2"):
                out[f"lsv_{k}"] = ko[k].to_numpy()
        out[f"lsv_a1{tag}"] = ko["a1"].to_numpy()
        out[f"lsv_a2{tag}"] = ko["a2"].to_numpy()
    return out, seconds


# --------------------------------------------------------------------------------------------
# entries
# --------------------------------------------------------------------------------------------


def _finite(frame: pd.DataFrame, columns: list[str]) -> bool:
    cols = [c for c in columns if c in frame]
    return bool(np.isfinite(frame[cols].to_numpy(float)).all()) if cols else True


def _entry_compute(
    entry: str, monthly: bool, estimator: str | None
) -> tuple[pd.DataFrame, dict[str, float]]:
    """The LSV entry marks of one date with ``estimator`` (raises on any failure)."""
    from volsto.studies import barrier_theory as bt

    ctx = run.context()
    ent = pd.read_parquet(run.ENTRIES / f"{entry}.parquet")
    market = bh.load_day(entry, ctx["calendar"], ctx["ohlc"])
    if market.carried:
        raise LookupError("carried surface: no desk fit on this date")
    surf = market.surface
    days, times = bh.future_times(entry, ctx["calendar"])
    seed = bh.seed_of(entry)
    df0 = ent["DF0"].to_numpy(float)
    e0 = ent["P_B6"].to_numpy(float)
    res = ent[["entry", "months", "side", "barrier"]].copy()
    res["monthly_subset"] = monthly
    marks = ("ssr12", "ssr10", "ssr15") if monthly else ("ssr12",)
    timing: dict[str, float] = {}
    for mark in marks:
        fit = run.desk_fit(entry, mark)
        res[f"{mark}_status"] = fit["status"]
        if not run.fit_sound(fit):
            continue
        t1 = time.perf_counter()
        model = lsv_model(surf, fit["params"], seed=seed, estimator=estimator)
        timing[f"{mark}_calib"] = time.perf_counter() - t1
        fwd_start = run.ForwardStart(ent, days, times)
        buckets = bt.Buckets(ent, days, times) if mark == "ssr12" else None

        def on_chunk(
            paths: bh.DayPaths,
            fwd_start: run.ForwardStart = fwd_start,
            buckets: Any = buckets,
        ) -> None:
            fwd_start(paths)
            if buckets is not None:
                buckets(paths)

        t1 = time.perf_counter()
        ko = run._ko_table(model, times, seed, ent, days, e0, df0, on_chunk=on_chunk)
        timing[f"{mark}_price"] = time.perf_counter() - t1
        for k in ("a1", "a1_se", "a2", "a2_se", "p1", "p2", "a1_raw", "a2_raw"):
            res[f"{mark}_{k}"] = ko[k].to_numpy()
        for key, col in fwd_start.columns(ent, mark).items():
            res[key] = col
        if buckets is not None:
            for key, col in buckets.columns("ssr12").items():
                res[key] = col
    return res, timing


def entry_job(job: tuple[str, bool]) -> dict[str, Any]:
    """One entry date.  With the binned estimator, a date that raises or returns a non-finite
    knock-out mark is recomputed on the default path and recorded ``sorted (fallback)``; if
    that fails too the error is kept (addendum 1b)."""
    entry, monthly = job
    out = LSV2 / f"{entry}.parquet"
    t0 = time.perf_counter()
    estimator = gate_estimator()
    label = estimator or "sorted"
    fallback = ""
    try:
        try:
            res, timing = _entry_compute(entry, monthly, estimator)
            marks = [f"{m}_{k}" for m in ("ssr12", "ssr10", "ssr15") for k in ("a1", "a2")]
            if estimator and not _finite(res, marks):
                raise FloatingPointError("non-finite knock-out mark")
        except LookupError:
            raise
        except Exception as first:
            if not estimator:
                raise
            fallback = f"{type(first).__name__}: {first}"[:200]
            res, timing = _entry_compute(entry, monthly, None)
            label = "sorted (fallback)"
        res["estimator"] = label
        LSV2.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        res.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        return {"date": entry, "seconds": time.perf_counter() - t0, "fallback": fallback, **timing}
    except Exception as exc:
        return {
            "date": entry,
            "error": f"{type(exc).__name__}: {exc}",
            "fallback": fallback,
            "trace": traceback.format_exc()[-800:],
            "seconds": time.perf_counter() - t0,
        }


#: Seconds to wait for one result of a pool before it is declared dead (addendum 1b).
RESULT_TIMEOUT_S = 30 * 60.0
MAX_RESTARTS = 3


def run_pool(job: Any, jobs: list[Any], workers: int, done: Any) -> list[dict[str, Any]]:
    """``job`` over ``jobs`` on a pool, each result read with a timeout of 30 minutes.  On a
    timeout the pool is terminated and a new one started on the jobs whose output is still
    missing (``done(job)`` false); after three restarts the process exits non-zero so that the
    chain moves on.  A dead worker must not hang the night."""
    import multiprocessing

    results: list[dict[str, Any]] = []
    todo = list(jobs)
    restarts = 0
    while todo:
        pool = Pool(workers)
        it = pool.imap_unordered(job, todo)
        timed_out = False
        for _ in range(len(todo)):
            try:
                results.append(it.next(timeout=RESULT_TIMEOUT_S))
            except multiprocessing.TimeoutError:
                timed_out = True
                break
        if not timed_out:
            pool.close()
            pool.join()
            break
        pool.terminate()
        pool.join()
        restarts += 1
        finished = {r["date"] for r in results}
        todo = [
            j for j in todo if not done(j) and (j[0] if isinstance(j, tuple) else j) not in finished
        ]
        print(
            f"TIMEOUT: no result for {RESULT_TIMEOUT_S / 60:.0f} minutes; pool terminated; "
            f"restart {restarts} of {MAX_RESTARTS} on {len(todo)} dates still missing",
            flush=True,
        )
        if restarts > MAX_RESTARTS:
            print(f"giving up after {MAX_RESTARTS} restarts: {len(todo)} dates missing", flush=True)
            _summary(results)
            sys.exit(1)
    return results


def _summary(res: list[dict[str, Any]]) -> None:
    bad = [r for r in res if "error" in r]
    fb = [r for r in res if r.get("fallback") and "error" not in r]
    print(f"  fallbacks to the default path: {len(fb)}; errors kept: {len(bad)}")
    for r in sorted(fb, key=lambda r: r["date"])[:30]:
        print(f"  fallback {r['date']}: {r['fallback']}")
    for r in sorted(bad, key=lambda r: r["date"])[:30]:
        print(f"  {r['date']}: {r['error']}\n{r.get('trace', '')}")


def run_entries(args: argparse.Namespace) -> None:
    entries = sorted(p.stem for p in run.ENTRIES.glob("*.parquet"))
    monthly = set(bh.monthly_subset(entries))
    jobs = [(e, e in monthly) for e in entries if not (LSV2 / f"{e}.parquet").exists()]
    if args.monthly:
        jobs = [j for j in jobs if j[1]]
    if args.limit:
        jobs = jobs[: args.limit]
    t0 = time.perf_counter()
    res = run_pool(entry_job, jobs, args.workers, lambda j: (LSV2 / f"{j[0]}.parquet").exists())
    bad = [r for r in res if "error" in r]
    secs = [r["seconds"] for r in res if "error" not in r]
    print(
        f"lsv2 entries ({gate_estimator() or 'sorted'}): {len(jobs)} run, {len(bad)} errors, "
        f"{time.perf_counter() - t0:.0f} s wall, {np.mean(secs) if secs else 0:.0f} s per entry"
    )
    for k in sorted({k for r in res for k in r if k.endswith(("_calib", "_price"))}):
        print(f"  {k}: {np.mean([r[k] for r in res if k in r]):.1f} s")
    _summary(res)


# --------------------------------------------------------------------------------------------
# daily
# --------------------------------------------------------------------------------------------


def daily_job(date: str) -> dict[str, Any]:
    """One trading day; the same fallback to the default path as :func:`entry_job`."""
    out = LSV2_DAILY / f"{date}.parquet"
    t0 = time.perf_counter()
    fallback = ""
    try:
        estimator = gate_estimator()
        label = estimator or "sorted"
        ctx = run.context()
        trades = run.load_trades()
        live = trades[(trades["entry"] <= date) & (trades["expiry"] > date)].reset_index(drop=True)
        if live.empty:
            return {"date": date, "cells": 0, "seconds": 0.0}
        params, remark, carried = run.week_parameters(date)
        if params is None:
            return {"date": date, "error": "no sound desk fit yet"}
        market = bh.load_day(date, ctx["calendar"], ctx["ohlc"])
        days, times = bh.future_times(date, ctx["calendar"])
        surf = market.surface
        T = np.array([bh.year_fraction(date, e) for e in live["expiry"]])
        df = np.array([market.df(t) for t in T])
        values = run._live_values(live, T, surf)

        def compute(est: str | None) -> pd.DataFrame:
            marks, _ = lsv_marks(
                surf, params, live, days, times, values, df, seed=bh.seed_of(date), estimator=est
            )
            return marks

        try:
            marks = compute(estimator)
            if estimator and not _finite(
                marks, [c for c in marks.columns if c.startswith(("lsv_a1", "lsv_a2"))]
            ):
                raise FloatingPointError("non-finite knock-out mark")
        except Exception as first:
            if not estimator:
                raise
            fallback = f"{type(first).__name__}: {first}"[:200]
            marks = compute(None)
            label = "sorted (fallback)"
        res = pd.DataFrame({"date": date, "cell": live["cell"]})
        for c in marks.columns:
            res[c] = marks[c].to_numpy(np.float32)
        res["remark_date"] = remark
        res["is_remark_day"] = remark == date
        res["carried_parameters"] = carried
        res["estimator"] = label
        LSV2_DAILY.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        res.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        return {
            "date": date,
            "cells": len(live),
            "seconds": time.perf_counter() - t0,
            "fallback": fallback,
        }
    except Exception as exc:
        return {
            "date": date,
            "error": f"{type(exc).__name__}: {exc}",
            "fallback": fallback,
            "trace": traceback.format_exc()[-800:],
            "seconds": time.perf_counter() - t0,
        }


def run_daily(args: argparse.Namespace) -> None:
    days = run.study_days(args.start, args.end)
    jobs = [d for d in days if not (LSV2_DAILY / f"{d}.parquet").exists()]
    t0 = time.perf_counter()
    res = run_pool(daily_job, jobs, args.workers, lambda d: (LSV2_DAILY / f"{d}.parquet").exists())
    bad = [r for r in res if "error" in r]
    secs = [r["seconds"] for r in res if "error" not in r and r.get("cells")]
    print(
        f"lsv2 daily ({gate_estimator() or 'sorted'}): {len(jobs)} run of {len(days)}, "
        f"{len(bad)} errors, {time.perf_counter() - t0:.0f} s wall, "
        f"{np.mean(secs) if secs else 0:.0f} s per day"
    )
    _summary(res)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("command", choices=["entries", "daily"])
    ap.add_argument("--start", default="2007-01-03")
    ap.add_argument("--end", default=run.LAST_DATA_DAY)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--monthly", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    {"entries": run_entries, "daily": run_daily}[args.command](args)


if __name__ == "__main__":
    main()
