"""Local correlation model (SPEC §8.7, M12 part LC7): the sweep over the dispersion study's
dates — one process of ``scripts/lcm_price.py`` per date, resumable, one row per date.

    python scripts/disp_lcm.py --tenor 3m|12m|24m --dates monthly|today|reference|<d1,d2,...>
        [--budget production|development] [--risk none|deltas|full] [--workers N|auto]
        [--root <dir>] [--config <yaml>] [--limit N] [--varswap] [--no-report] [--no-retry]

Dates.  ``monthly``: the converged dates of the study's model S table
(``outputs/dispersion/model_s_3m.parquet``, 218 dates) and today, 219 in all — at 12m and 24m
those of them with an entry of that tenor.  ``reference``: the four dates of the reference
implementation.  The order of a pass is today and the reference dates first, then every 8th
date, then the gaps by halving (4, 2, 1): a pass stopped early covers the whole period.

Rows.  Each date's row is ``<root>/outputs/dispersion_lc/rows/<tenor>_<budget>/<date>.json``
(what ``lcm_price.py`` writes); the pass's table is ``lcm_<tenor>.parquet`` at the production
budget and ``lcm_<tenor>_dev.parquet`` at the development one, rewritten after every date.
Resume: a date is skipped when its row was written by the same git commit with the same
configuration digest (the YAML, the tenor and the budget — with the read-only study data, the
same specification key), did not fail, and carries at least the risk asked for.  A date that
fails gets a row with ``status = "failed"`` and its reason, the pass goes on, and the failed
dates are run once more at the end with the same seeds.

Workers.  One process per date with ``NUMBA_NUM_THREADS = cores // workers`` and one BLAS
thread.  ``--workers auto`` times the first two dates one after the other on every core and the
next two side by side on half the cores each, and keeps the faster arrangement.

Log.  One line per date with its status, its seconds, the elapsed time and the estimated time
left, on stdout and in ``<root>/outputs/dispersion_lc/logs/<tenor>_<budget>.log``; each date's
own output is in ``logs/<tenor>_<budget>/<date>.log``.  The pass ends by running the comparison
report (``scripts/lcm_report.py``).  Nothing is written into the study's ``outputs/dispersion``.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lcm_price as lp

from volsto.calibration.cache import code_version
from volsto.studies import disp_data as dd

log = logging.getLogger("disp_lcm")

TODAY = "2026-10-02"
REFERENCE_DATES = ("2026-10-02", "2019-09-03", "2017-04-03", "2008-07-07")
RISK_RANK = {"none": 0, "deltas": 1, "full": 2}
NESTED = ("index_errors", "full_risk")


def study_dates(tenor: str) -> list[str]:
    """The converged model S dates and today, with an entry at ``tenor``, in time order."""
    table = pd.read_parquet(dd.OUT / "model_s_3m.parquet")
    dates = {str(d)[:10] for d in table.loc[table["converged"], "date"]} | {TODAY}
    have = {p.stem for p in (dd.OUT / "entries" / tenor).glob("*.pkl")}
    return sorted(dates & have)


def interleaved(dates: Sequence[str], first: Sequence[str] = REFERENCE_DATES) -> list[str]:
    """``first`` (those in ``dates``), then every 8th date, then the gaps by halving."""
    rest = [d for d in dates if d not in first]
    order = [d for d in first if d in dates]
    seen = set(order)
    for start, step in ((0, 8), (4, 8), (2, 4), (1, 2)):
        for d in rest[start::step]:
            if d not in seen:
                order.append(d)
                seen.add(d)
    return order


class Sweep:
    """A pass: its folders, the rows it has, the dates left."""

    def __init__(self, tenor: str, budget: str, risk: str, cfg: dict[str, Any], root: Path) -> None:
        self.tenor, self.budget, self.risk, self.cfg = tenor, budget, risk, cfg
        self.out = lp.out_root(cfg, root)
        self.root = root
        self.tag = f"{tenor}_{budget}"
        self.rows = self.out / "rows" / self.tag
        self.logs = self.out / "logs" / self.tag
        self.rows.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.commit = code_version()
        self.digest = lp.config_digest(cfg, tenor, budget)
        self.table = self.out / (f"lcm_{tenor}.parquet" if budget == "production" else f"lcm_{tenor}_dev.parquet")  # fmt: skip

    def row(self, date: str) -> dict[str, Any] | None:
        path = self.rows / f"{date}.json"
        if not path.exists():
            return None
        try:
            return dict(json.loads(path.read_text()))
        except (OSError, ValueError):
            return None

    def done(self, date: str) -> bool:
        r = self.row(date)
        return bool(
            r is not None
            and r.get("status") in ("ok", "check")
            and r.get("git_commit") == self.commit
            and r.get("config_digest") == self.digest
            and RISK_RANK[r.get("risk", "none")] >= RISK_RANK[self.risk]
        )

    def command(self, date: str, varswap: bool, config: str) -> list[str]:
        cmd = [
            sys.executable, str(Path(__file__).resolve().parent / "lcm_price.py"),
            "--date", date, "--tenor", self.tenor, "--budget", self.budget, "--risk", self.risk,
            "--config", config, "--root", str(self.root),
            "--row-out", str(self.rows / f"{date}.json"),
        ]  # fmt: skip
        return [*cmd, "--varswap"] if varswap else cmd

    def write_table(self) -> int:
        """The rows on disk as one table (the nested entries left to the JSON rows)."""
        rows = []
        for path in sorted(self.rows.glob("*.json")):
            try:
                r = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            rows.append({k: v for k, v in r.items() if k not in NESTED})
        if not rows:
            return 0
        frame = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
        tmp = self.table.with_suffix(".parquet.tmp")
        frame.to_parquet(tmp, index=False)
        tmp.replace(self.table)
        return len(frame)


def run_dates(
    sweep: Sweep, dates: Sequence[str], workers: int, threads: int, args: argparse.Namespace,
    t_start: float, n_total: int, n_before: int,
) -> list[tuple[str, str, float]]:  # fmt: skip
    """Run ``dates`` on ``workers`` processes; one log line per date.  Returns ``(date, status,
    seconds)`` in completion order."""
    env = {**os.environ, "NUMBA_NUM_THREADS": str(threads)}
    for v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[v] = "1"
    queue = list(dates)
    running: dict[str, tuple[subprocess.Popen[bytes], float, Any]] = {}
    finished: list[tuple[str, str, float]] = []
    while queue or running:
        while queue and len(running) < workers:
            d = queue.pop(0)
            fh = (sweep.logs / f"{d}.log").open("wb")
            varswap = bool(args.varswap and d in REFERENCE_DATES)
            proc = subprocess.Popen(sweep.command(d, varswap, args.config), stdout=fh, stderr=subprocess.STDOUT, env=env)  # fmt: skip
            running[d] = (proc, time.perf_counter(), fh)
        time.sleep(1.0)
        for d, (proc, t0, fh) in list(running.items()):
            if proc.poll() is None:
                continue
            fh.close()
            del running[d]
            seconds = time.perf_counter() - t0
            r = sweep.row(d)
            if r is None:  # the process died before its row: the row is written here
                r = {"date": d, "tenor": sweep.tenor, "budget": sweep.budget, "risk": sweep.risk,
                     "git_commit": sweep.commit, "config_digest": sweep.digest, "status": "failed",
                     "reason": f"process exited with code {proc.returncode} and no row"}  # fmt: skip
                (sweep.rows / f"{d}.json").write_text(json.dumps(r, indent=1))
            finished.append((d, r["status"], seconds))
            n_rows = sweep.write_table()
            n_done = n_before + len(finished)
            elapsed = time.perf_counter() - t_start
            rate = elapsed / max(len(finished), 1)  # this call's dates only: the pass's pace now
            left = (n_total - n_done) * rate
            if r["status"] == "failed":
                detail = f"FAILED: {r['reason'][:160]}"
            else:
                detail = (
                    f"E_LC[D] {r['ED_lc']:.6f} ({r['ED_lc_se']:.6f}), LC/CC {r['ratio']:.5f} ({r['ratio_se']:.5f}), "
                    f"clip inside {r['clip_inner_max']:.4f}, idx ATM {r['idx_err_atm']:+.3f} vp"
                    + (f"; {r['reason']}" if r["reason"] else "")
                )
            log.info(
                "%s %s [%d/%d] %s %.0f s (workers %d x %d threads); elapsed %s, left about %s; table %d rows; %s",
                time.strftime("%H:%M:%S"), d, n_done, n_total, r["status"], seconds, workers, threads,
                hms(elapsed), hms(left), n_rows, detail,
            )  # fmt: skip
    return finished


def hms(seconds: float) -> str:
    s = int(max(seconds, 0))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m{s % 60:02d}s"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--tenor", default="3m", choices=("3m", "12m", "24m"))
    ap.add_argument("--dates", default="monthly", help="monthly | today | reference | d1,d2,...")
    ap.add_argument("--budget", default="production", choices=("production", "development"))
    ap.add_argument("--risk", default=None, choices=("none", "deltas", "full"))
    ap.add_argument("--workers", default="1", help="a number, or auto")
    ap.add_argument("--config", default=str(lp.CONFIG))
    ap.add_argument("--root", default=None, help="the worktree that holds outputs/dispersion_lc")
    ap.add_argument(
        "--limit", type=int, default=None, help="at most this many dates (in the pass's order)"
    )
    ap.add_argument(
        "--varswap", action="store_true", help="the basket variance swap on the reference dates"
    )
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--no-retry", action="store_true")
    args = ap.parse_args(argv)
    cfg = lp.load_config(args.config)
    risk = args.risk or cfg["risk"]
    root = Path(args.root).resolve() if args.root else lp.ROOT
    sweep = Sweep(args.tenor, args.budget, risk, cfg, root)
    logging.basicConfig(
        level=logging.INFO, format="%(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(sweep.out / "logs" / f"{sweep.tag}.log")],
    )  # fmt: skip
    if args.dates == "monthly":
        dates = interleaved(study_dates(args.tenor))
    elif args.dates == "today":
        dates = [TODAY]
    elif args.dates == "reference":
        dates = list(REFERENCE_DATES)
    else:
        dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    if args.limit is not None:
        dates = dates[: args.limit]
    todo = [d for d in dates if not sweep.done(d)]
    cores = os.cpu_count() or 1
    log.info(
        "=== %s pass %s (%s budget, risk %s): %d dates, %d to run; commit %s, config %s; %d cores; rows %s",
        time.strftime("%Y-%m-%d %H:%M:%S"), args.tenor, args.budget, risk, len(dates), len(todo),
        sweep.commit, sweep.digest, cores, sweep.rows,
    )  # fmt: skip
    t_start = time.perf_counter()
    n_total, n_before = len(dates), len(dates) - len(todo)
    finished: list[tuple[str, str, float]] = []
    if args.workers == "auto" and len(todo) >= 6:
        one = run_dates(sweep, todo[:2], 1, cores, args, t_start, n_total, n_before)
        t_one = sum(s for _, _, s in one) / 2
        t0 = time.perf_counter()
        two = run_dates(sweep, todo[2:4], 2, cores // 2, args, t_start, n_total, n_before + 2)
        t_two = (time.perf_counter() - t0) / 2
        workers = 2 if t_two < t_one else 1
        log.info("workers: %.0f s per date one at a time, %.0f s per date two side by side -> %d", t_one, t_two, workers)  # fmt: skip
        finished += one + two
        todo = todo[4:]
    else:
        workers = 1 if args.workers == "auto" else int(args.workers)
    threads = max(cores // workers, 1)
    finished += run_dates(sweep, todo, workers, threads, args, t_start, n_total, n_before + len(finished))  # fmt: skip
    failed = [d for d, status, _ in finished if status == "failed"]
    if failed and not args.no_retry:
        log.info(
            "retrying the %d failed dates once, same seeds: %s", len(failed), ", ".join(failed)
        )
        again = run_dates(sweep, failed, workers, threads, args, t_start, n_total, n_total - len(failed))  # fmt: skip
        for d, status, _ in again:
            path = sweep.rows / f"{d}.json"
            r = json.loads(path.read_text())
            r["retried"] = True
            path.write_text(json.dumps(r, indent=1))
            log.info("retry of %s: %s", d, status)
        sweep.write_table()
    n_rows = sweep.write_table()
    statuses = pd.Series([sweep.row(d)["status"] if sweep.row(d) else "missing" for d in dates]).value_counts()  # type: ignore[index]
    log.info(
        "=== %s pass %s (%s) finished in %s: %d rows in %s; statuses %s",
        time.strftime("%Y-%m-%d %H:%M:%S"), args.tenor, args.budget, hms(time.perf_counter() - t_start),
        n_rows, sweep.table, statuses.to_dict(),
    )  # fmt: skip
    if not args.no_report:
        report = Path(__file__).resolve().parent / "lcm_report.py"
        cmd = [sys.executable, str(report), "--tenor", args.tenor, "--budget", args.budget, "--root", str(root)]  # fmt: skip
        code = subprocess.run(cmd, check=False).returncode
        log.info("report: exit code %d", code)
    return 0


if __name__ == "__main__":
    sys.exit(main())
