"""The estimator gate of the barrier study (BARRIER_STUDY_ADDENDUM §2).

Merges ``feature/binned-estimator``, checks that no test newly fails, runs the machinery of
pilot check 9 with the estimator unset and ``"binned"`` (the 3 check-9 dates and 3 stressed
ones — 2008-10-06, 2018-02-05, 2020-03-02 — × 4 seeds, mark ``ssr12``, 100,000 particles,
``min_window=0``, every cell of the date's entry; addendum 1b), and writes its decision to
``outputs/interview/gate.json``: ``{"estimator": "binned", …}`` when (a), (b) and (c) all pass,
``{"estimator": null, "reason": …}`` otherwise — also when the merge conflicts in code, a test
newly fails (the merge is then reverted), the gate errors, or it runs past 60 minutes.  Binned
is selected only if (a), (b) and (c) pass on all six dates.

    python scripts/barrier_gate.py [--workers 12]

It never edits a test or a tolerance.  Run it with a clean working tree.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "interview"
GATE = OUT / "gate.json"
BRANCH = "feature/binned-estimator"
#: The three calm check-9 dates and three stressed ones (addendum 1b: the binned-minus-sorted
#: scatter grows as the particle count falls, and the pilot dates are all calm).
DATES = ("2024-01-02", "2024-02-12", "2024-03-25", "2008-10-06", "2018-02-05", "2020-03-02")
SEEDS = (0, 1, 2, 3)
ESTIMATORS: tuple[str | None, ...] = (None, "binned")
TESTS = (
    "tests/test_lsv.py",
    "tests/test_engine.py",
    "tests/test_barrier_history.py",
    "tests/test_barrier_theory.py",
    "tests/test_import_orats.py",
)
DEADLINE_S = 60 * 60.0  # addendum 1b
#: Limits of (b) are multiplied, cell by cell, by max(1, at-the-money vol at entry / 15 %).
VOL_REFERENCE = 0.15
MAX_PRICE_BP = 0.6
MAX_DELTA = 0.003
MAX_TIME_RATIO = 0.8
PY = str(ROOT / ".venv" / "bin" / "python")


def sh(*args: str, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=check)


def failing_tests() -> set[str]:
    """The ids of the tests that fail or error among :data:`TESTS` (fast ones only)."""
    files = [t for t in TESTS if (ROOT / t).exists()]
    proc = sh(PY, "-m", "pytest", "-n", "8", "-m", "not slow", "-q", "-p", "no:cacheprovider",
              "-rfE", "--tb=no", *files)  # fmt: skip
    return {
        line.split(" ", 1)[1].split(" - ")[0].strip()
        for line in proc.stdout.splitlines()
        if line.startswith(("FAILED ", "ERROR "))
    }


def merge() -> tuple[bool, str]:
    """Merge the estimator branch (``--no-ff``).  ``(merged, note)``; a conflict in a code or
    test file aborts; a conflict in ``SPEC.md`` or a doc only keeps both sides."""
    if sh("git", "merge-base", "--is-ancestor", BRANCH, "HEAD").returncode == 0:
        return True, "already merged"
    proc = sh("git", "merge", "--no-ff", "--no-edit", BRANCH)
    if proc.returncode == 0:
        return True, "merged cleanly"
    conflicted = sh("git", "diff", "--name-only", "--diff-filter=U").stdout.split()
    docs = [f for f in conflicted if f == "SPEC.md" or f.endswith(".md") or f.startswith("docs/")]
    if not conflicted or set(conflicted) != set(docs):
        sh("git", "merge", "--abort")
        return False, f"conflict in code or tests: {conflicted or proc.stderr[-300:]}"
    for f in docs:  # keep both sides: drop the markers
        path = ROOT / f
        kept = [
            line
            for line in path.read_text().splitlines(keepends=True)
            if not line.startswith(("<<<<<<< ", "=======", ">>>>>>> "))
        ]
        path.write_text("".join(kept))
        sh("git", "add", f)
    sh("git", "commit", "--no-edit")
    return True, f"merged; both sides kept in {docs}"


def gate_job(job: tuple[str, int, str | None]) -> Any:
    """One (date, seed, estimator): the marks and bumped marks of the date's entry cells."""
    date, seed, estimator = job
    import pandas as pd

    sys.path.insert(0, str(ROOT / "scripts"))
    import barrier_history as run
    import barrier_lsv2 as lsv2

    from volsto.studies import barrier_history as bh

    ctx = run.context()
    ent = pd.read_parquet(run.ENTRIES / f"{date}.parquet")
    market = bh.load_day(date, ctx["calendar"], ctx["ohlc"])
    days, times = bh.future_times(date, ctx["calendar"])
    T = ent["T"].to_numpy(float)
    marks, seconds = lsv2.lsv_marks(
        market.surface,
        run.desk_fit(date, "ssr12")["params"],
        ent,
        days,
        times,
        run._live_values(ent, T, market.surface),
        ent["DF0"].to_numpy(float),
        seed=bh.seed_of(date) + 1000 * seed,
        estimator=estimator,
    )
    out = ent[["entry", "months", "side", "barrier", "K", "F0", "DF0", "atm"]].copy()
    for c in marks.columns:
        out[c] = marks[c].to_numpy()
    out["seed"] = seed
    out["estimator"] = estimator or "sorted"
    out["calibration_seconds"] = float(sum(seconds)) / 3.0
    return out


def timing() -> dict[str, float]:
    """Seconds per calibration, each estimator, this process alone (one date)."""
    import pandas as pd

    sys.path.insert(0, str(ROOT / "scripts"))
    import barrier_history as run
    import barrier_lsv2 as lsv2

    from volsto.studies import barrier_history as bh

    ctx = run.context()
    date = DATES[0]
    market = bh.load_day(date, ctx["calendar"], ctx["ohlc"])
    params = run.desk_fit(date, "ssr12")["params"]
    _ = pd
    out = {}
    for est in ESTIMATORS:
        lsv2.lsv_model(market.surface, params, seed=1, estimator=est, n_particles=20_000)  # warm
        t0 = time.perf_counter()
        lsv2.lsv_model(market.surface, params, seed=1, estimator=est)
        out[est or "sorted"] = time.perf_counter() - t0
    return out


def evaluate(frame: Any) -> dict[str, Any]:
    """Criteria (a) and (b) per date and quantity.  (a): every cell, |binned − sorted| of the
    seed means within 2 standard errors or 1 % (check 9's criterion).  (b): every same-seed
    paired difference at most 0.6 bp of spot (prices) or 0.003 (deltas), times
    ``max(1, atm / 15 %)`` of the cell.  Also the mean signed paired difference and its
    standard error."""
    import numpy as np

    x = frame.copy()
    e2 = 0.02
    for k in ("a1", "a2"):
        x[k] = x[f"lsv_{k}"] / x["K"]
        x[f"{k}_delta"] = (x[f"lsv_{k}_up"] - x[f"lsv_{k}_dn"]) / (e2 * x["F0"] * x["DF0"])
    key = ["entry", "months", "side", "barrier"]
    table = []
    ok_a = ok_b = True
    for date, xd in x.groupby("entry"):
        for q in ("a1", "a2", "a1_delta", "a2_delta"):
            g = (
                xd.groupby([*key, "estimator"])[q]
                .agg(["mean", "std", "count"])
                .unstack("estimator")
            )
            a, b = g["mean"]["binned"], g["mean"]["sorted"]
            se = np.sqrt(
                g["std"]["binned"] ** 2 / g["count"]["binned"]
                + g["std"]["sorted"] ** 2 / g["count"]["sorted"]
            )
            d = a - b
            z = (d / se.where(se > 0)).abs()
            rel = d.abs() / b.abs().where(b.abs() > 0)
            passed = (z <= 2) | (rel <= 0.01) | (d == 0)
            paired = xd.pivot_table(index=[*key, "seed"], columns="estimator", values=q)
            diff = paired["binned"] - paired["sorted"]
            atm = xd.groupby([*key, "seed"])["atm"].first().reindex(diff.index)
            factor = np.maximum(1.0, atm / VOL_REFERENCE)
            is_price = not q.endswith("delta")
            scale = 1e4 if is_price else 1.0
            limit = MAX_PRICE_BP if is_price else MAX_DELTA
            over = scale * diff.abs() > limit * factor
            a_ok, b_ok = bool(passed.all()), bool(not over.any())
            ok_a &= a_ok
            ok_b &= b_ok
            table.append(
                {
                    "date": str(date),
                    "quantity": q,
                    "cells": len(d),
                    "(a) outside 2 se or 1 %": int((~passed).sum()),
                    "(a) pass": a_ok,
                    "paired |diff| median": float(scale * diff.abs().median()),
                    "paired |diff| max": float(scale * diff.abs().max()),
                    "max |diff| / scaled limit": float(
                        (scale * diff.abs() / (limit * factor)).max()
                    ),
                    "mean signed paired diff": float(scale * diff.mean()),
                    "its standard error": float(scale * diff.std(ddof=1) / np.sqrt(len(diff))),
                    "unit": "bp of spot" if is_price else "delta",
                    "(b) limit before scaling": limit,
                    "atm scaling max": float(factor.max()),
                    "(b) over the limit": int(over.sum()),
                    "(b) pass": b_ok,
                }
            )
    return {"table": table, "a": ok_a, "b": ok_b}


def decide(estimator: str | None, **info: Any) -> None:
    head = sh("git", "rev-parse", "--short", "HEAD").stdout.strip()
    doc = {
        "estimator": estimator,
        "head": head,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        **info,
    }
    GATE.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(json.dumps(doc, indent=1, default=str))
    if estimator == "binned":  # keep the sorted results aside; the LSV phase recomputes
        for name in ("lsv_entries", "lsv_daily"):
            src = OUT / "barrier_results" / name
            if src.exists() and not (src.parent / f"{name}_sorted").exists():
                shutil.move(str(src), str(src.parent / f"{name}_sorted"))


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    t0 = time.perf_counter()
    try:
        if (
            sh("git", "diff", "--quiet").returncode
            or sh("git", "diff", "--cached", "--quiet").returncode
        ):
            return decide(None, reason="the working tree has uncommitted changes to tracked files")
        before = failing_tests()
        merged, note = merge()
        if not merged:
            return decide(None, reason=f"merge failed: {note}", failing_before=sorted(before))
        after = failing_tests()
        new = sorted(after - before)
        if new:
            if note != "already merged":
                sh("git", "revert", "-m", "1", "--no-edit", "HEAD")
            return decide(None, reason=f"tests newly failing after the merge: {new}", merge=note,
                          failing_before=sorted(before), failing_after=sorted(after))  # fmt: skip
        import pandas as pd

        jobs = [(d, s, e) for d in DATES for s in SEEDS for e in ESTIMATORS]
        parts = []
        with Pool(args.workers) as pool:
            it = pool.imap_unordered(gate_job, jobs)
            for _ in jobs:
                left = DEADLINE_S - (time.perf_counter() - t0)
                if left <= 0:
                    raise TimeoutError("gate run beyond 60 minutes")
                parts.append(it.next(timeout=left))
        frame = pd.concat(parts, ignore_index=True)
        frame.to_parquet(OUT / "gate_runs.parquet", index=False)
        ev = evaluate(frame)
        secs = timing()
        ratio = secs["binned"] / secs["sorted"]
        c_ok = bool(ratio <= MAX_TIME_RATIO)
        info = {
            "merge": note,
            "failing_before": sorted(before),
            "failing_after": sorted(after),
            "table": ev["table"],
            "a_pass": ev["a"],
            "b_pass": ev["b"],
            "c_pass": c_ok,
            "seconds_per_calibration": secs,
            "time_ratio": ratio,
            "seconds_per_calibration_in_pool": frame.groupby("estimator")["calibration_seconds"]
            .mean()
            .to_dict(),
            "gate_seconds": time.perf_counter() - t0,
        }
        if ev["a"] and ev["b"] and c_ok:
            return decide("binned", **info)
        failed = [k for k, v in (("a", ev["a"]), ("b", ev["b"]), ("c", c_ok)) if not v]
        return decide(None, reason=f"criteria not met: {failed}", **info)
    except Exception as exc:
        return decide(None, reason=f"gate error: {type(exc).__name__}: {exc}",
                      trace=traceback.format_exc()[-1500:])  # fmt: skip


if __name__ == "__main__":
    main()
