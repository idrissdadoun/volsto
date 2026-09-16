"""M8b hedging studies — the CLI of :mod:`volsto.studies.m8b` (SPEC §8.2; the owner's "M8b —
hedging studies": compute-heavy, print the projected wall clock per study and shard across the
VM; 2·10⁴ paths default, weekly for 3y products; reference = the SPX 2022-12-30 marking fit).

Flow: the projected wall clock per study is printed BEFORE anything runs (the hedger's probe per
task plus the leverage calibrations the tasks need up front, checked against the cache); the
shard's tasks run under the standing rules (per task: key, status, wall clock, calibrations
performed, cache hits — a task whose leverage is not cached is skipped with the reason under
``--no-calibrate``; otherwise the script calibrates and SAYS so); the study C static predictions
(the M7 rotation greek per product and policy) are computed unless ``--skip-static``; the tables
are then rebuilt from EVERY result present under ``--out`` (so shards merge) into
``<out>/m8b_table_<study>.csv`` and ``<out>/m8b.md`` (with the wall clock and whether anything
was recalibrated).

Usage::

    .venv/bin/python scripts/m8b.py --dry-run --study all
    caffeinate -i .venv/bin/python scripts/m8b.py --study B --shard 1/4
    caffeinate -i .venv/bin/python scripts/m8b.py --study C --resume

Smoke (plumbing only, no calibration; tasks whose leverage is absent are skipped)::

    .venv/bin/python scripts/m8b.py --study A --n-paths 2000 --world-paths 2000 \\
        --n-particles 200000 --refit-particles 200000 --no-calibrate --frequency monthly \\
        --out /tmp/m8b_smoke --only A__cliquet_1y__pricing__delta_only__pricing_LV
"""

from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

from volsto.calibration.cache import CacheMissError
from volsto.studies.m8b import (
    BOOK,
    DEFAULT_CACHE,
    DEFAULT_N_PARTICLES,
    DEFAULT_N_PATHS,
    DEFAULT_OUT,
    DEFAULT_REFIT_PARTICLES,
    DEFAULT_ROTATION_PARTICLES,
    DEFAULT_ROTATION_PATHS,
    RECALIBRATIONS_C,
    STUDIES,
    StudyConfig,
    StudyEnvironment,
    Task,
    TaskList,
    build_tables,
    discriminator_gate,
    enumerate_tasks,
    load_results,
    load_static_predictions,
    parse_shard,
    project,
    result_paths,
    run_task,
    shard,
    static_prediction,
    write_tables,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--study", default="all", help="A|B|C|D|all (comma-separated allowed)")
    ap.add_argument(
        "--shard", default="1/1", help="i/n: every n-th task of the ordered list from i"
    )
    ap.add_argument("--n-paths", type=int, default=DEFAULT_N_PATHS)
    ap.add_argument("--n-particles", type=int, default=DEFAULT_N_PARTICLES)
    ap.add_argument("--refit-particles", type=int, default=DEFAULT_REFIT_PARTICLES)
    ap.add_argument("--world-paths", type=int, default=None)
    ap.add_argument("--frequency", default=None, help="override: daily|weekly|monthly")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    ap.add_argument(
        "--dry-run", action="store_true", help="print the task counts and projections only"
    )
    ap.add_argument("--resume", action="store_true", help="skip tasks with an existing result file")
    ap.add_argument(
        "--no-calibrate", action="store_true", help="never calibrate: skip on a cache miss"
    )
    ap.add_argument("--stream-bumps", action="store_true")
    ap.add_argument("--no-control-variate", action="store_true")
    ap.add_argument("--rotation-paths", type=int, default=DEFAULT_ROTATION_PATHS)
    ap.add_argument("--rotation-particles", type=int, default=DEFAULT_ROTATION_PARTICLES)
    ap.add_argument(
        "--skip-static", action="store_true", help="do not compute the study C static greeks"
    )
    ap.add_argument(
        "--only", default=None, help="comma-separated task keys (or key substrings) to run"
    )
    ap.add_argument("--quiet", action="store_true")
    return ap.parse_args()


def selected_studies(text: str) -> list[str]:
    if text.strip().lower() == "all":
        return list(STUDIES)
    out = [s.strip().upper() for s in text.split(",") if s.strip()]
    bad = [s for s in out if s not in STUDIES]
    if bad:
        raise SystemExit(f"unknown study {bad}; choose from {STUDIES}")
    return out


def print_projection(studies: list[str], lists: dict[str, TaskList], cfg: StudyConfig) -> None:
    """The projection never calibrates: it probes with ``allow_calibrate=False`` and reports the
    base leverage as a needed calibration when it is absent."""
    env = StudyEnvironment(dataclasses.replace(cfg, allow_calibrate=False))
    total = 0.0
    for st in studies:
        tl = lists[st]
        try:
            pr = project(list(tl.tasks), env)
        except CacheMissError as exc:
            print(
                f"[m8b] study {st}: {len(tl.tasks)} tasks; projection unavailable — {exc} "
                f"(the run would calibrate it first)",
                flush=True,
            )
            continue
        print("[m8b] " + pr.summary(), flush=True)
        for row in pr.per_task:
            print(f"[m8b]     {row['key']}: projected {row['projected_seconds']:.0f} s", flush=True)
        for key, lab in pr.missing_keys.items():
            print(f"[m8b]     calibration needed: {lab} ({key[:12]})", flush=True)
        if tl.skipped:
            print(
                f"[m8b]     {len(tl.skipped)} row(s) gated out: {tl.skipped[0]['status']} — "
                f"{tl.skipped[0]['reason']}",
                flush=True,
            )
        total += pr.total_seconds
    print(
        f"[m8b] projected total for the selection: {total:.0f} s ({total / 3600:.2f} h)", flush=True
    )


def main() -> None:
    args = parse_args()
    t_start = time.perf_counter()
    cfg = StudyConfig(
        n_paths=int(args.n_paths),
        n_particles=int(args.n_particles),
        refit_particles=int(args.refit_particles),
        world_paths=None if args.world_paths is None else int(args.world_paths),
        frequency=args.frequency,
        out=Path(args.out),
        cache=Path(args.cache),
        allow_calibrate=not args.no_calibrate,
        stream_bumps=bool(args.stream_bumps),
        control_variate=not args.no_control_variate,
        rotation_paths=int(args.rotation_paths),
        rotation_particles=int(args.rotation_particles),
        verbose=not args.quiet,
    )
    studies = selected_studies(args.study)
    i, n = parse_shard(args.shard)
    gate = discriminator_gate(cfg.discriminator_verdict)
    print(
        f"[m8b] config: {cfg.n_paths} pricing paths, {cfg.n_world} world paths, "
        f"{cfg.n_particles} particles ({cfg.refit_particles} for refits), seed {cfg.seed}, "
        f"frequency {cfg.frequency or 'rule (daily <= 1y, weekly beyond; A monthly)'}, "
        f"calibrate {'yes' if cfg.allow_calibrate else 'NO (skip on a cache miss)'}, "
        f"stream_bumps {cfg.stream_bumps}, control_variate {cfg.control_variate}, out {cfg.out}",
        flush=True,
    )
    print(f"[m8b] discriminator gate for world (ii): {gate.verdict!r} — {gate.reason}", flush=True)
    lists = {st: enumerate_tasks(st, cfg, gate) for st in STUDIES}
    for st in studies:
        print(
            f"[m8b] study {st}: {len(lists[st].tasks)} tasks ({len(lists[st].skipped)} gated out)"
        )
    print_projection(studies, lists, cfg)
    if args.dry_run:
        return

    env = StudyEnvironment(cfg)
    only = [s.strip() for s in args.only.split(",")] if args.only else None
    run_rows: list[tuple[str, str, float, int, int]] = []
    for st in studies:
        tasks: list[Task] = shard(lists[st].tasks, i, n)
        if only is not None:
            tasks = [t for t in tasks if any(o == t.key or o in t.key for o in only)]
        print(f"[m8b] study {st}: shard {i}/{n} carries {len(tasks)} task(s)", flush=True)
        for task in tasks:
            jp, _ = result_paths(cfg.out, task)
            if args.resume and jp.exists():
                print(f"[m8b]   resume: {task.key} exists, skipped", flush=True)
                continue
            print(f"[m8b]   run {task.label}", flush=True)
            try:
                res = run_task(task, cfg, env)
            except Exception as exc:
                print(f"[m8b]   FAILED {task.key}: {type(exc).__name__}: {exc}", flush=True)
                run_rows.append((task.key, f"failed: {type(exc).__name__}", 0.0, 0, 0))
                continue
            print(
                f"[m8b]   {res.status:8s} {task.key}: wall {res.wall_seconds:.0f} s, "
                f"{res.calibrations} calibration(s), {res.cache_hits} cache hit(s)"
                + (f"; {res.reason}" if res.reason else "")
                + (
                    f"; mean {res.mean[0]:+.4f} +/- {res.mean[1]:.4f}, std {res.std[0]:.4f} "
                    f"{res.unit}"
                    if res.ok
                    else ""
                ),
                flush=True,
            )
            run_rows.append(
                (task.key, res.status, res.wall_seconds, res.calibrations, res.cache_hits)
            )

    static = load_static_predictions(cfg.out)
    if "C" in studies and not args.skip_static:
        for product in BOOK:
            for policy in RECALIBRATIONS_C:
                if policy == "none":
                    continue
                try:
                    doc = static_prediction(product, policy, env)
                except CacheMissError as exc:
                    print(f"[m8b]   static greek {product} / {policy}: skipped — {exc}", flush=True)
                    continue
                static[(product, policy)] = doc
                print(
                    f"[m8b]   static greek {product} / {policy}: desk_pnl_shadow "
                    f"{doc['desk_pnl_shadow'][0]:+.4f} +/- {doc['desk_pnl_shadow'][1]:.4f} "
                    f"{doc['unit']} per rota ({doc['wall_seconds']:.0f} s, recalibrated: "
                    f"{'yes' if doc['recalibrated'] else 'no'})",
                    flush=True,
                )

    results = load_results(cfg.out)
    tables = build_tables(results, lists["B"].skipped, static)
    wall = time.perf_counter() - t_start
    n_cal = env.calibrations
    header = [
        f"wall clock of this invocation: {wall:.0f} s ({wall / 3600:.2f} h); studies {studies}, "
        f"shard {i}/{n}; {len(run_rows)} task(s) run now, {len(results)} result(s) in {cfg.out}",
        f"recalibrated: {'yes, ' + str(n_cal) + ' leverage calibration(s)' if n_cal else 'no'}"
        + (f" — {'; '.join(env.notes)}" if env.notes else ""),
        f"budget: {cfg.n_paths} pricing paths, {cfg.n_world} world paths, {cfg.n_particles} "
        f"particles ({cfg.refit_particles} for refits), seed {cfg.seed}, frequency "
        f"{cfg.frequency or 'rule (daily <= 1y, weekly beyond; A monthly)'}, stream_bumps "
        f"{cfg.stream_bumps}, control_variate {cfg.control_variate}",
        f"pricing model: {cfg.marking_fit.name} at {cfg.n_particles} particles; world (ii) gate: "
        f"{gate.verdict!r} — {gate.reason}",
        f"static greeks of study C: {cfg.rotation_paths} paths, {cfg.rotation_particles} particles "
        f"(the M7 rotation states)",
    ]
    md = write_tables(cfg.out, tables, header_lines=header, static=static)
    for st in studies:
        df = tables[st]
        print(f"[m8b] table {st}: {len(df)} row(s) -> {cfg.out / f'm8b_table_{st}.csv'}")
    print(f"[m8b] report: {md}; wall clock {wall:.0f} s; recalibrated: {'yes' if n_cal else 'no'}")


if __name__ == "__main__":
    main()
