"""Run the passes of the barrier study on 2007–2026 (:mod:`volsto.studies.barrier_history`).

    python scripts/barrier_history.py days    --start 2007-01-03 --end 2026-10-02
    python scripts/barrier_history.py entries --start ... --end ... [--monthly]
    python scripts/barrier_history.py daily   --start ... --end ...

``days``     import every trading day (surface; SABRW fits on the first day of each week) into
             ``outputs/interview/cache/days``.
``entries``  on each entry date: the cells, the structures, the entry premiums off the surface,
             the local-vol knock-out premiums, the lower bound, vega and theta
             → ``barrier_results/entries/<date>.parquet``.
``daily``    on each trading day: the marks and bumped marks of every live cell
             → ``barrier_results/daily/<date>.parquet``.

Every pass is resumable (a date whose file exists is skipped) and runs one process per date,
one thread each.  Nothing here is vendor data in git: the outputs are under ``outputs/``.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
from multiprocessing import Pool
from typing import Any

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from volsto.studies import barrier_history as bh  # noqa: E402

ENTRIES = bh.RESULTS / "entries"
DAILY = bh.RESULTS / "daily"
LAST_DATA_DAY = "2026-10-02"

_ctx: dict[str, Any] = {}


def context() -> dict[str, Any]:
    """Per-process: the OHLC history and the trading calendar (extended into the future)."""
    if "ohlc" not in _ctx:
        ohlc = bh.load_ohlc()
        known = [d for d in ohlc.index if d >= "2006-12-01"]
        _ctx["ohlc"] = ohlc
        _ctx["calendar"] = bh.trading_calendar(known, "2027-12-31")
        _ctx["known"] = known
    return _ctx


def study_days(start: str, end: str) -> list[str]:
    return [d for d in context()["known"] if start <= d <= end]


def all_entries(start: str, end: str) -> list[str]:
    cal = context()["known"]
    return bh.entry_dates(cal, max(start, "2007-01-01"), end)


# --------------------------------------------------------------------------------------------
# days
# --------------------------------------------------------------------------------------------


def _day_job(job: tuple[str, bool]) -> dict[str, Any]:
    date, sabrw = job
    return bh.import_day(date, sabrw=sabrw)


def run_days(args: argparse.Namespace) -> None:
    days = study_days(args.start, args.end)
    remark = set(bh.entry_dates(days, args.start, args.end))
    jobs = []
    for d in days:
        info = bh.day_info(d)
        done = info is not None and (info.get("sabrw") or d not in remark or not info.get("built"))
        if done and not args.force:
            continue
        jobs.append((d, d in remark))
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        infos = list(pool.imap_unordered(_day_job, jobs, chunksize=4))
    built = sum(1 for i in infos if i["built"])
    print(
        f"days: {len(jobs)} imported ({built} built) of {len(days)} "
        f"in {time.perf_counter() - t0:.0f} s"
    )
    for i in sorted(infos, key=lambda i: i["date"]):
        if not i["built"]:
            print(f"  not built {i['date']}: {i.get('error')}")


# --------------------------------------------------------------------------------------------
# entries
# --------------------------------------------------------------------------------------------


def _ko_table(
    model: Any,
    times: np.ndarray,
    seed: int,
    cells: pd.DataFrame,
    days: list[str],
    european: np.ndarray,
    df: np.ndarray,
    betas: pd.DataFrame | None = None,
    *,
    n_paths: int | None = None,
    on_chunk: Any = None,
) -> pd.DataFrame:
    """Discounted knock-out values of ``cells`` (per unit underlying) under ``model``: one path
    set on the future trading days ``times``, simulated and reduced chunk by chunk, every cell
    a functional of it, controlled by the European knock-out whose discounted surface value is
    ``european`` (``df`` the discount factor of each cell); ``betas`` reuses the base set's
    coefficients."""
    cols = ["a1", "a1_se", "a2", "a2_se", "p1", "p2", "a1_beta", "a2_beta"]
    cols += ["a1_raw", "a1_raw_se", "a2_raw", "a2_raw_se"]
    out = pd.DataFrame(index=cells.index, columns=cols, dtype=np.float64)
    index = {d: j for j, d in enumerate(days)}
    groups = []
    for (expiry, side), g in cells.groupby(["expiry", "side"]):
        pos = cells.index.get_indexer(g.index)
        groups.append(
            (index[expiry], int(side), g["K"].to_numpy(float), g["B"].to_numpy(float), g.index, pos)
        )
    totals: list[dict[str, np.ndarray] | None] = [None] * len(groups)
    for paths in bh.simulate_day_chunks(model, times, n_paths=n_paths or bh.LV_PATHS, seed=seed):
        if on_chunk is not None:
            on_chunk(paths)
        for i, (j, side, K, B, _, _) in enumerate(groups):
            sums = bh.knock_out_sums(paths, j, side, K, B)
            prev = totals[i]
            totals[i] = sums if prev is None else {k: prev[k] + sums[k] for k in sums}
    for (_, _, _, _, idx, pos), sums in zip(groups, totals, strict=True):
        assert sums is not None
        beta = None
        if betas is not None:
            beta = {k: betas.loc[idx, f"{k}_beta"].to_numpy(float) for k in ("a1", "a2")}
        v = bh.knock_out_from_sums(sums, european[pos] / df[pos], beta)
        for k in ("a1", "a1_se", "a2", "a2_se"):
            out.loc[idx, k] = df[pos] * v[k]
        for k in ("p1", "p2", "a1_beta", "a2_beta"):
            out.loc[idx, k] = v[k]
        raw = bh.knock_out_from_sums(sums)  # without the control: pilot check 5
        for k in ("a1", "a1_se", "a2", "a2_se"):
            out.loc[idx, k.replace("a1", "a1_raw").replace("a2", "a2_raw")] = df[pos] * raw[k]
    return out


class ForwardStart:
    """Accumulates the forward-start smile ``T/2 → T`` of each maturity of an entry over the
    chunks of a path set (spec §7.1.4)."""

    def __init__(self, cells: pd.DataFrame, days: list[str], times: np.ndarray) -> None:
        index = {d: j for j, d in enumerate(days)}
        self.legs: dict[int, tuple[int, int, float]] = {}
        for months, g in cells.groupby("months"):
            j_end = index[g["expiry"].iloc[0]]
            j_half = int(np.argmin(np.abs(times - 0.5 * times[j_end])))
            if j_half < j_end:
                self.legs[int(months)] = (j_half, j_end, float(times[j_end] - times[j_half]))
        self.sums: dict[int, np.ndarray] = {}

    def __call__(self, paths: bh.DayPaths) -> None:
        for months, (j_half, j_end, _) in self.legs.items():
            part = bh.forward_start_sums(paths, j_half, j_end)
            self.sums[months] = part if months not in self.sums else self.sums[months] + part

    def columns(self, cells: pd.DataFrame, prefix: str) -> dict[str, np.ndarray]:
        vols = {m: bh.forward_start_vols(self.sums[m], self.legs[m][2]) for m in self.sums}
        keys = ["fvol_0.95", "fvol_1", "fvol_1.05", "fskew"]
        return {
            f"{prefix}_{k}": np.array(
                [vols.get(int(m), {}).get(k, np.nan) for m in cells["months"]]
            )
            for k in keys
        }


def entry_job(entry: str) -> dict[str, Any]:
    out = ENTRIES / f"{entry}.parquet"
    t0 = time.perf_counter()
    try:
        ctx = context()
        market = bh.load_day(entry, ctx["calendar"], ctx["ohlc"])
        ok = market.tolerances_ok()
        cells = bh.cells_of(entry, market, ctx["calendar"])
        days, times = bh.future_times(entry, ctx["calendar"])
        seed = bh.seed_of(entry)
        surf = market.surface
        i6 = bh.STRUCTURES.index("B6")
        Ts = cells["T"].to_numpy(float)
        df0 = np.array([market.df(T) for T in Ts])
        plain = [
            (int(r.side), bh.structure_legs(int(r.side), r.K, r.B, r.K)) for r in cells.itertuples()
        ]
        pp = bh.expand_legs(plain)
        pT = Ts[pp["slot"] // len(bh.STRUCTURES)]
        aged_T = np.maximum(pT - 1.0 / 365.0, 1e-6)
        e0 = bh.bulk_values(pp, pT, surf)[0][:, i6]
        e_vega = bh.bulk_values(pp, pT, surf, vol_shift=0.01)[0][:, i6]
        e_theta = bh.bulk_values(pp, aged_T, surf)[0][:, i6]
        fwd_start = ForwardStart(cells, days, times)
        ko = _ko_table(
            bh.local_vol_model(surf), times, seed, cells, days, e0, df0, on_chunk=fwd_start
        )
        vega_surf = bh.ParallelVolSurface(surf, 0.01)
        ko_vega = _ko_table(
            bh.local_vol_model(vega_surf), times, seed, cells, days, e_vega, df0, ko
        )
        # theta: the same model, every remaining observation one calendar day closer
        aged_times = np.maximum(times - 1.0 / 365.0, 1e-6)
        df_aged = np.array([market.df(max(T - 1.0 / 365.0, 1e-6)) for T in Ts])
        ko_theta = _ko_table(
            bh.local_vol_model(surf), aged_times, seed, cells, days, e_theta, df_aged, ko
        )
        legs_all = []
        lbs = []
        for (side, legs), row, a1 in zip(plain, cells.itertuples(), ko["a1"], strict=True):
            legs.update(bh.matched_legs(side, row.K, row.B, float(a1), surf, row.T))
            legs_all.append((side, legs))
            lbs.append(bh.lower_bound(side, row.K, row.B, surf, row.T))
        pieces = bh.expand_legs(legs_all)
        piece_cell = pieces["slot"] // len(bh.STRUCTURES)
        piece_T = Ts[piece_cell]
        v0 = bh.bulk_values(pieces, piece_T, surf)[0]
        v_vega = bh.bulk_values(pieces, piece_T, surf, vol_shift=0.01)[0]
        v_theta = bh.bulk_values(pieces, np.maximum(piece_T - 1.0 / 365.0, 1e-6), surf)[0]
        res = cells.copy()
        res["S0"] = market.spot
        res["F0"] = [market.forward(T) for T in cells["T"]]
        res["DF0"] = [market.df(T) for T in cells["T"]]
        res["tolerances_ok"] = ok
        res["carried"] = market.carried
        for k in ("a1", "a1_se", "a2", "a2_se", "p1", "p2", "a2_raw", "a2_raw_se"):
            res[f"lv_{k}"] = ko[k].to_numpy()
        res["lv_a1_vega"] = (ko_vega["a1"] - ko["a1"]).to_numpy()
        res["lv_a2_vega"] = (ko_vega["a2"] - ko["a2"]).to_numpy()
        res["lv_a1_theta"] = (ko_theta["a1"] - ko["a1"]).to_numpy()
        res["lv_a2_theta"] = (ko_theta["a2"] - ko["a2"]).to_numpy()
        for key, col in fwd_start.columns(cells, "lv").items():
            res[key] = col
        res["LB"] = [x[0] for x in lbs]
        res["LB_strike"] = [x[1] for x in lbs]
        for s, name in enumerate(bh.STRUCTURES):
            res[f"P_{name}"] = v0[:, s]
            res[f"vega_{name}"] = v_vega[:, s] - v0[:, s]
            res[f"theta_{name}"] = v_theta[:, s] - v0[:, s]
        res["legs"] = [bh.legs_to_json(legs) for _, legs in legs_all]
        for key in ("fwd_median_bp", "fwd_max_bp", "rms_3m_2y", "rms_6m_2y"):
            res[key] = market.info.get(key, np.nan)
        res["settlement_uncertain"] = bool(market.info.get("settlement_uncertain", False))
        cond = [bh.entry_conditions(surf, r.T, int(r.side)) for r in cells.itertuples()]
        for key in cond[0]:
            res[key] = [c[key] for c in cond]
        ENTRIES.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        res.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        return {"date": entry, "ok": ok, "seconds": time.perf_counter() - t0}
    except Exception as exc:
        return {
            "date": entry,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-600:],
            "seconds": time.perf_counter() - t0,
        }


def run_entries(args: argparse.Namespace) -> None:
    entries = all_entries(args.start, args.end)
    if args.monthly:
        entries = bh.monthly_subset(entries)
    jobs = [e for e in entries if args.force or not (ENTRIES / f"{e}.parquet").exists()]
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        res = list(pool.imap_unordered(entry_job, jobs))
    bad = [r for r in res if "error" in r]
    print(
        f"entries: {len(jobs)} run of {len(entries)}, {len(bad)} errors, "
        f"{time.perf_counter() - t0:.0f} s wall, "
        f"{np.mean([r['seconds'] for r in res]) if res else 0:.1f} s per entry"
    )
    for r in sorted(bad, key=lambda r: r["date"]):
        print(f"  {r['date']}: {r['error']}")


# --------------------------------------------------------------------------------------------
# daily
# --------------------------------------------------------------------------------------------


def load_trades() -> pd.DataFrame:
    """Every cell of every entry file, with ``cell`` = ``entry|months|side|barrier``."""
    if "trades" not in _ctx:
        parts = [pd.read_parquet(p) for p in sorted(ENTRIES.glob("*.parquet"))]
        t = pd.concat(parts, ignore_index=True)
        t["cell"] = (
            t["entry"]
            + "|"
            + t["months"].astype(str)
            + "|"
            + t["side"].astype(str)
            + "|"
            + t["barrier"]
        )
        _ctx["trades"] = t
    return _ctx["trades"]


def daily_job(date: str) -> dict[str, Any]:
    out = DAILY / f"{date}.parquet"
    t0 = time.perf_counter()
    try:
        ctx = context()
        trades = load_trades()
        live = trades[(trades["entry"] <= date) & (trades["expiry"] > date)].reset_index(drop=True)
        if live.empty:
            return {"date": date, "cells": 0, "seconds": 0.0}
        market = bh.load_day(date, ctx["calendar"], ctx["ohlc"])
        days, times = bh.future_times(date, ctx["calendar"])
        seed = bh.seed_of(date)
        surf = market.surface
        res = pd.DataFrame({"date": date, "cell": live["cell"]})
        T = np.array([bh.year_fraction(date, e) for e in live["expiry"]])
        res["S"] = market.spot
        res["F"] = [market.forward(t) for t in T]
        res["DF"] = [market.df(t) for t in T]
        res["carried"] = market.carried
        ratios = (1.0, 1.0 + bh.BUMP, 1.0 - bh.BUMP)
        legs_all = [
            (int(side), bh.legs_from_json(text))
            for side, text in zip(live["side"], live["legs"], strict=True)
        ]
        pieces = bh.expand_legs(legs_all)
        piece_T = T[pieces["slot"] // len(bh.STRUCTURES)]
        v = bh.bulk_values(pieces, piece_T, surf, ratios)
        i6 = bh.STRUCTURES.index("B6")
        df = res["DF"].to_numpy(float)
        timing = {}
        base_ko = None
        for r_i, (tag, ratio) in enumerate(zip(("", "_up", "_dn"), ratios, strict=True)):
            s = surf if ratio == 1.0 else bh.MovedSurface(surf, ratio)
            t1 = time.perf_counter()
            ko = _ko_table(
                bh.local_vol_model(s), times, seed, live, days, v[r_i][:, i6], df, base_ko
            )
            timing[f"ko{tag}"] = time.perf_counter() - t1
            if base_ko is None:
                base_ko = ko
            res[f"a1{tag}"] = ko["a1"].to_numpy()
            res[f"a2{tag}"] = ko["a2"].to_numpy()
            if ratio == 1.0:
                for k in ("a1_se", "a2_se", "p1", "p2"):
                    res[k] = ko[k].to_numpy()
        for s_i, name in enumerate(bh.STRUCTURES):
            res[name] = v[0][:, s_i]
            res[f"{name}_up"] = v[1][:, s_i]
            res[f"{name}_dn"] = v[2][:, s_i]
        float_cols = [c for c in res.columns if res[c].dtype == np.float64]
        res[float_cols] = res[float_cols].astype(np.float32)
        res[["S", "F", "DF"]] = np.column_stack(
            [
                np.full(len(T), market.spot),
                [market.forward(t) for t in T],
                [market.df(t) for t in T],
            ]
        )
        DAILY.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        res.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        return {
            "date": date,
            "cells": len(live),
            "carried": market.carried,
            "seconds": time.perf_counter() - t0,
            **timing,
        }
    except Exception as exc:
        return {
            "date": date,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-800:],
            "seconds": time.perf_counter() - t0,
        }


def run_daily(args: argparse.Namespace) -> None:
    days = study_days(args.start, args.end)
    jobs = [d for d in days if args.force or not (DAILY / f"{d}.parquet").exists()]
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        res = list(pool.imap_unordered(daily_job, jobs))
    bad = [r for r in res if "error" in r]
    secs = [r["seconds"] for r in res if "error" not in r and r.get("cells")]
    print(
        f"daily: {len(jobs)} run of {len(days)}, {len(bad)} errors, "
        f"{time.perf_counter() - t0:.0f} s wall, {np.mean(secs) if secs else 0:.1f} s per day, "
        f"{sum(1 for r in res if r.get('carried'))} carried days"
    )
    for r in sorted(bad, key=lambda r: r["date"])[:40]:
        print(f"  {r['date']}: {r['error']}\n{r.get('trace', '')}")


# --------------------------------------------------------------------------------------------
# LSV: the desk marks on entry dates
# --------------------------------------------------------------------------------------------

LSV_ENTRIES = bh.RESULTS / "lsv_entries"
FITS = bh.OUT / "cache" / "fits"
#: mark → (SSR target, skew eps): the desk fit at each (spec §4.1, §4.2; contradiction 2)
MARKS: dict[str, tuple[float, float]] = {
    "ssr10": (1.0, 0.10),
    "ssr12": (1.2, 0.10),
    "ssr15": (1.5, 0.05),
}
LSV_HORIZON = 1.03
LSV_PARTICLES = 100_000


def desk_fit(date: str, mark: str) -> dict[str, Any]:
    """The desk marking fit of ``date`` at ``mark`` (cached as JSON under the study's cache):
    parameters, status, messages.  Step 0 reads the date's SABRW fits."""
    import dataclasses
    import json

    from volsto.calibration.fit_2f import fit_2f_marking, fit_preset
    from volsto.market.loaders import load_ssvi_surface, load_step0_source

    path = FITS / f"{date}_{mark}.json"
    if path.exists():
        out: dict[str, Any] = json.loads(path.read_text())
        return out
    ssr, eps = MARKS[mark]
    snap = bh.DAY_CACHE / f"{date}.yaml"
    t0 = time.perf_counter()
    try:
        surf = load_ssvi_surface(snap)
        cfg = fit_preset("desk", skew_eps=eps)
        fit = fit_2f_marking(surf, cfg, ssr_target=ssr, step0=load_step0_source(snap, surf))
        out = {
            "date": date,
            "mark": mark,
            "status": fit.status,
            "messages": list(fit.messages)[:4],
            "params": dataclasses.asdict(fit.params),
        }
    except Exception as exc:
        out = {
            "date": date,
            "mark": mark,
            "status": "error",
            "messages": [f"{type(exc).__name__}: {exc}"[:300]],
        }
    out["seconds"] = round(time.perf_counter() - t0, 2)
    FITS.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")
    return out


def fit_sound(fit: dict[str, Any]) -> bool:
    """A fit whose parameters can mark: any status but ``infeasible`` or an error."""
    return fit.get("status") in ("interior", "binding") and "params" in fit


def lsv_model(
    surface: Any, params: dict[str, float], *, n_particles: int, seed: int, min_window: int = 0
) -> Any:
    """The LSV model of ``params`` with its leverage calibrated on ``surface`` (the particle
    method at ``n_particles``, the regression floor ``min_window`` particles and 1 % of N),
    horizon just past one year.  Nothing is persisted."""
    from volsto.calibration.particle import calibrate_leverage
    from volsto.config import BergomiParams, ParticleConfig, SimConfig
    from volsto.market.varswap import xi0_curve
    from volsto.models.bergomi import BergomiSV
    from volsto.models.lsv import LSV

    fc = surface.forward_curve
    xi0 = xi0_curve(surface, min(surface.max_maturity, 5.0))
    kernel = BergomiSV(BergomiParams(**params), xi0, fc)
    particle = ParticleConfig(n_particles=n_particles, horizon=LSV_HORIZON, min_window=min_window)
    result = calibrate_leverage(
        surface, kernel, particle, SimConfig(n_paths=n_particles, seed=seed)
    )
    return LSV(kernel, result.leverage)


def lsv_entry_job(entry: str) -> dict[str, Any]:
    out = LSV_ENTRIES / f"{entry}.parquet"
    t0 = time.perf_counter()
    try:
        ctx = context()
        ent = pd.read_parquet(ENTRIES / f"{entry}.parquet")
        market = bh.load_day(entry, ctx["calendar"], ctx["ohlc"])
        if market.carried:
            return {"date": entry, "error": "carried surface: no desk fit on this date"}
        surf = market.surface
        days, times = bh.future_times(entry, ctx["calendar"])
        seed = bh.seed_of(entry)
        df0 = ent["DF0"].to_numpy(float)
        e0 = ent["P_B6"].to_numpy(float)
        res = ent[["entry", "months", "side", "barrier"]].copy()
        timing: dict[str, float] = {}
        for mark in MARKS:
            fit = desk_fit(entry, mark)
            res[f"{mark}_status"] = fit["status"]
            if not fit_sound(fit):
                continue
            t1 = time.perf_counter()
            model = lsv_model(surf, fit["params"], n_particles=LSV_PARTICLES, seed=seed)
            timing[f"{mark}_calib"] = time.perf_counter() - t1
            fwd_start = ForwardStart(ent, days, times)
            checks: dict[tuple[str, int], np.ndarray] = {}

            def on_chunk(paths: bh.DayPaths, fwd_start: ForwardStart = fwd_start,
                         checks: dict[tuple[str, int], np.ndarray] = checks) -> None:  # fmt: skip
                fwd_start(paths)
                if mark != "ssr12":  # noqa: B023 — called within this iteration
                    return
                for (expiry, side), g in ent.groupby(["expiry", "side"]):
                    K, B = g["K"].to_numpy(float), g["B"].to_numpy(float)
                    strikes = np.concatenate([K[:1], B, 2 * B - K])  # K, each B, each far strike
                    part = bh.vanilla_sums(paths, days.index(expiry), int(side), strikes)
                    key = (expiry, int(side))
                    checks[key] = part if key not in checks else checks[key] + part

            t1 = time.perf_counter()
            ko = _ko_table(model, times, seed, ent, days, e0, df0, on_chunk=on_chunk)
            timing[f"{mark}_price"] = time.perf_counter() - t1
            for k in ("a1", "a1_se", "a2", "a2_se", "p1", "p2"):
                res[f"{mark}_{k}"] = ko[k].to_numpy()
            for key, col in fwd_start.columns(ent, mark).items():
                res[key] = col
            if mark == "ssr12":  # pilot check 4: the LSV reprices the vanillas the study uses
                names = ("K", "B", "far")
                cols = {
                    f"ssr12_van_{n}_{q}": np.full(len(ent), np.nan)
                    for n in names
                    for q in ("diff", "se", "vega")
                }
                for (expiry, side), g in ent.groupby(["expiry", "side"]):
                    n, sx, sxx = checks[(expiry, int(side))]
                    mean = sx / n
                    se = np.sqrt(np.maximum(sxx / n - mean * mean, 0.0) / (n - 1))
                    K, B = g["K"].to_numpy(float), g["B"].to_numpy(float)
                    strikes = np.concatenate([K[:1], B, 2 * B - K])
                    T = float(g["T"].iloc[0])
                    F = market.forward(T)
                    vol = np.asarray(surf.implied_vol(strikes, T), dtype=float)
                    exact = bh.black(
                        np.asarray(F), strikes, np.asarray(T), vol, int(side), np.asarray(1.0)
                    )
                    s_ = vol * np.sqrt(T)
                    d1 = (np.log(F / strikes) + 0.5 * s_ * s_) / s_
                    vega = F * np.sqrt(T) * np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi)
                    nb = len(g)
                    pos = ent.index.get_indexer(g.index)
                    pick = {
                        "K": np.zeros(nb, int),
                        "B": 1 + np.arange(nb),
                        "far": 1 + nb + np.arange(nb),
                    }
                    for nm, idx in pick.items():
                        # undiscounted, per unit of spot; vega per vol point
                        cols[f"ssr12_van_{nm}_diff"][pos] = (mean - exact)[idx] / K
                        cols[f"ssr12_van_{nm}_se"][pos] = se[idx] / K
                        cols[f"ssr12_van_{nm}_vega"][pos] = 0.01 * vega[idx] / K
                for name, col in cols.items():
                    res[name] = col
        LSV_ENTRIES.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        res.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        return {"date": entry, "seconds": time.perf_counter() - t0, **timing}
    except Exception as exc:
        return {
            "date": entry,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-800:],
            "seconds": time.perf_counter() - t0,
        }


def run_lsv(args: argparse.Namespace) -> None:
    entries = sorted(p.stem for p in ENTRIES.glob("*.parquet") if args.start <= p.stem <= args.end)
    if args.monthly:
        entries = bh.monthly_subset(entries)
    jobs = [e for e in entries if args.force or not (LSV_ENTRIES / f"{e}.parquet").exists()]
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        res = list(pool.imap_unordered(lsv_entry_job, jobs))
    bad = [r for r in res if "error" in r]
    secs = [r["seconds"] for r in res if "error" not in r]
    print(
        f"lsv: {len(jobs)} run of {len(entries)}, {len(bad)} errors, "
        f"{time.perf_counter() - t0:.0f} s wall, {np.mean(secs) if secs else 0:.0f} s per entry"
    )
    keys = sorted({k for r in res for k in r if k.endswith(("_calib", "_price"))})
    for k in keys:
        print(f"  {k}: {np.mean([r[k] for r in res if k in r]):.1f} s")
    for r in sorted(bad, key=lambda r: r["date"])[:30]:
        print(f"  {r['date']}: {r['error']}\n{r.get('trace', '')}")


# --------------------------------------------------------------------------------------------
# LSV: daily marks and sticky-strike deltas (SSR 1.2), and pilot check 9
# --------------------------------------------------------------------------------------------

LSV_DAILY = bh.RESULTS / "lsv_daily"


def week_parameters(date: str) -> tuple[dict[str, float] | None, str, bool]:
    """``(parameters, the re-mark date they come from, carried)`` for ``date``: the desk fit
    (SSR 1.2) of the first trading day of its week; when that fit is not sound, the last sound
    one before it (``carried``); ``None`` before the first sound fit."""
    known = context()["known"]
    remarks = bh.entry_dates(known, "2007-01-01", date)
    carried = False
    for remark in reversed(remarks[-60:]):
        info = bh.day_info(remark)
        if info and info.get("built") and info.get("sabrw"):
            fit = desk_fit(remark, "ssr12")
            if fit_sound(fit):
                return fit["params"], remark, carried
        carried = True
    return None, "", True


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
    n_particles: int,
    min_window: int,
) -> pd.DataFrame:
    """Marks and bumped marks of ``cells`` under the LSV of ``params``: the leverage is
    recalibrated on the surface and on each bumped surface with the same seed, the trades
    repriced on the same random numbers (spec §5.2 item 5).  ``values`` is the bulk value
    array of the cells (ratios 1, 1+e, 1−e) for the European control."""
    i6 = bh.STRUCTURES.index("B6")
    out = pd.DataFrame(index=cells.index)
    base = None
    for r_i, (tag, ratio) in enumerate(
        zip(("", "_up", "_dn"), (1.0, 1.0 + bh.BUMP, 1.0 - bh.BUMP), strict=True)
    ):
        s = surf if ratio == 1.0 else bh.MovedSurface(surf, ratio)
        model = lsv_model(s, params, n_particles=n_particles, seed=seed, min_window=min_window)
        ko = _ko_table(model, times, seed, cells, days, values[r_i][:, i6], df, base)
        if base is None:
            base = ko
            for k in ("a1_se", "a2_se", "p1", "p2"):
                out[f"lsv_{k}"] = ko[k].to_numpy()
        out[f"lsv_a1{tag}"] = ko["a1"].to_numpy()
        out[f"lsv_a2{tag}"] = ko["a2"].to_numpy()
    return out


def _live_values(live: pd.DataFrame, T: np.ndarray, surf: Any) -> np.ndarray:
    legs_all = [
        (int(side), {"B6": bh.legs_from_json(text)["B6"]})
        for side, text in zip(live["side"], live["legs"], strict=True)
    ]
    pieces = bh.expand_legs(legs_all)
    piece_T = T[pieces["slot"] // len(bh.STRUCTURES)]
    return bh.bulk_values(pieces, piece_T, surf, (1.0, 1.0 + bh.BUMP, 1.0 - bh.BUMP))


def lsv_daily_job(date: str) -> dict[str, Any]:
    out = LSV_DAILY / f"{date}.parquet"
    t0 = time.perf_counter()
    try:
        ctx = context()
        trades = load_trades()
        live = trades[(trades["entry"] <= date) & (trades["expiry"] > date)].reset_index(drop=True)
        if live.empty:
            return {"date": date, "cells": 0, "seconds": 0.0}
        params, remark, carried = week_parameters(date)
        if params is None:
            return {"date": date, "error": "no sound desk fit yet"}
        market = bh.load_day(date, ctx["calendar"], ctx["ohlc"])
        days, times = bh.future_times(date, ctx["calendar"])
        surf = market.surface
        T = np.array([bh.year_fraction(date, e) for e in live["expiry"]])
        df = np.array([market.df(t) for t in T])
        marks = lsv_marks(
            surf,
            params,
            live,
            days,
            times,
            _live_values(live, T, surf),
            df,
            seed=bh.seed_of(date),
            n_particles=LSV_PARTICLES,
            min_window=0,
        )
        res = pd.DataFrame({"date": date, "cell": live["cell"]})
        for c in marks.columns:
            res[c] = marks[c].to_numpy(np.float32)
        res["remark_date"] = remark
        res["is_remark_day"] = remark == date
        res["carried_parameters"] = carried
        LSV_DAILY.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        res.to_parquet(tmp, index=False)
        os.replace(tmp, out)
        return {"date": date, "cells": len(live), "seconds": time.perf_counter() - t0}
    except Exception as exc:
        return {
            "date": date,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-800:],
            "seconds": time.perf_counter() - t0,
        }


def run_lsv_daily(args: argparse.Namespace) -> None:
    days = study_days(args.start, args.end)
    jobs = [d for d in days if args.force or not (LSV_DAILY / f"{d}.parquet").exists()]
    t0 = time.perf_counter()
    with Pool(args.workers) as pool:
        res = list(pool.imap_unordered(lsv_daily_job, jobs))
    bad = [r for r in res if "error" in r]
    secs = [r["seconds"] for r in res if "error" not in r and r.get("cells")]
    print(
        f"lsv-daily: {len(jobs)} run of {len(days)}, {len(bad)} errors, "
        f"{time.perf_counter() - t0:.0f} s wall, {np.mean(secs) if secs else 0:.0f} s per day"
    )
    for r in sorted(bad, key=lambda r: r["date"])[:30]:
        print(f"  {r['date']}: {r['error']}\n{r.get('trace', '')}")


CHECK9_CONFIGS: dict[str, tuple[int, int]] = {
    "100k_floor1pct": (100_000, 0),
    "200k_library": (200_000, 2000),
}


def check9_job(job: tuple[str, int, str]) -> pd.DataFrame:
    date, seed, config = job
    ctx = context()
    n, floor = CHECK9_CONFIGS[config]
    ent = pd.read_parquet(ENTRIES / f"{date}.parquet")
    market = bh.load_day(date, ctx["calendar"], ctx["ohlc"])
    days, times = bh.future_times(date, ctx["calendar"])
    T = ent["T"].to_numpy(float)
    marks = lsv_marks(
        market.surface,
        desk_fit(date, "ssr12")["params"],
        ent,
        days,
        times,
        _live_values(ent, T, market.surface),
        ent["DF0"].to_numpy(float),
        seed=bh.seed_of(date) + 1000 * seed,
        n_particles=n,
        min_window=floor,
    )
    out = ent[["entry", "months", "side", "barrier", "K", "F0", "DF0"]].copy()
    for c in marks.columns:
        out[c] = marks[c].to_numpy()
    out["seed"] = seed
    out["config"] = config
    return out


def run_check9(args: argparse.Namespace) -> None:
    jobs = [(d, s, c) for d in args.dates for s in range(8) for c in CHECK9_CONFIGS]
    with Pool(args.workers) as pool:
        parts = list(pool.imap_unordered(check9_job, jobs))
    pd.concat(parts, ignore_index=True).to_parquet(bh.OUT / "pilot_check9.parquet", index=False)
    print(f"check9: {len(jobs)} runs written")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("command", choices=["days", "entries", "daily", "lsv", "lsv-daily", "check9"])
    ap.add_argument("--dates", nargs="*", default=[])
    ap.add_argument("--start", default="2007-01-03")
    ap.add_argument("--end", default=LAST_DATA_DAY)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--monthly", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    commands = {
        "days": run_days,
        "entries": run_entries,
        "daily": run_daily,
        "lsv": run_lsv,
        "lsv-daily": run_lsv_daily,
        "check9": run_check9,
    }
    commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
