"""Conditions at entry for the barrier study (BARRIER_STUDY_SPEC §7, §8.2).

Per entry date, from what is known before the entry's snapshot (closes up to the previous
trading day, the day cache's surface metrics up to the previous day): realised vols, implied
minus realised, trend and variance ratios, the realised skew stickiness ratio, vol of vol,
overnight share, tail-day count, up/down and range vols, drawdown, the Cboe indices; per cell,
the barrier against the trailing one-year extreme and the *historical gap* of rule 2 — the
expected (structure payoff minus knock-out payoff) on filtered historical paths drawn from SPX
returns strictly before the entry, at the entry's at-the-money vol, demeaned.

    python scripts/barrier_features.py [--workers 12]

Writes ``barrier_results/features.parquet`` (one row per cell).  Aggregates of public index
data and of the study's own surfaces: no vendor quote.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from volsto.studies.history_stats import filtered_historical_paths  # noqa: E402

TRADING_DAYS = {1: 21, 3: 63, 6: 126, 12: 252}
INDICES = ("VIX", "VIX3M", "VVIX", "SKEW", "COR1M", "COR3M")
FHS_PATHS = 20_000

_cache: dict[str, Any] = {}


def day_metrics() -> pd.DataFrame:
    """The day cache's surface readings, indexed by date (built days only)."""
    if "metrics" not in _cache:
        rows = []
        for p in sorted(bh.DAY_CACHE.glob("*.json")):
            j = json.loads(p.read_text())
            if j.get("built"):
                rows.append({k: v for k, v in j.items() if isinstance(v, (int, float, str))})
        _cache["metrics"] = pd.DataFrame(rows).set_index("date").sort_index()
    return _cache["metrics"]


def index_closes() -> pd.DataFrame:
    if "indices" not in _cache:
        cols = {}
        for name in INDICES:
            p = bh.HISTORY / f"{name}.csv"
            if p.exists():
                cols[name] = pd.read_csv(p, dtype={"date": str}).set_index("date")["close"]
        _cache["indices"] = pd.DataFrame(cols).sort_index()
    return _cache["indices"]


def date_features(entry: str) -> dict[str, float]:
    """Everything of the entry date that does not depend on the cell (module docstring)."""
    ctx = run.context()
    ohlc: pd.DataFrame = ctx["ohlc"]
    i = ohlc.index.get_loc(entry)
    past = ohlc.iloc[max(i - 520, 0) : i]  # strictly before the entry
    c = past["close"].to_numpy()
    r = np.diff(np.log(c))
    out: dict[str, float] = {}
    for months, n in TRADING_DAYS.items():
        out[f"rv_{months}m"] = (
            float(np.std(r[-n:], ddof=1) * np.sqrt(252.0)) if r.size >= n else np.nan
        )
    out["ret_3m"] = float(c[-1] / c[-64] - 1.0) if c.size > 64 else np.nan
    out["above_200d"] = float(c[-1] > c[-200:].mean()) if c.size >= 200 else np.nan
    year = r[-252:]
    if year.size >= 252:
        v1 = year.var(ddof=1)
        for q in (5, 10):
            rq = np.convolve(year, np.ones(q), mode="valid")
            out[f"var_ratio_{q}"] = float(rq.var(ddof=1) / (q * v1))
    w = past.iloc[-64:]
    ro = np.log(w["open"].to_numpy()[1:] / w["close"].to_numpy()[:-1])
    rd = np.log(w["close"].to_numpy()[1:] / w["open"].to_numpy()[1:])
    out["overnight_share"] = float((ro**2).sum() / ((ro**2).sum() + (rd**2).sum()))
    r63 = r[-63:]
    out["up_down_vol"] = float(
        np.sqrt(np.mean(r63[r63 > 0] ** 2)) / np.sqrt(np.mean(r63[r63 < 0] ** 2))
    )
    hl = np.log(w["high"].to_numpy()[1:] / w["low"].to_numpy()[1:])
    parkinson = np.sqrt(np.mean(hl**2) / (4.0 * np.log(2.0)) * 252.0)
    out["range_over_close_vol"] = float(parkinson / (np.std(r63, ddof=1) * np.sqrt(252.0)))
    # days beyond 2.5 trailing standard deviations over the last 3 months
    if r.size >= 126:
        sd = pd.Series(r).rolling(63).std(ddof=1).shift(1).to_numpy()
        out["tail_days_3m"] = float(np.sum(np.abs(r[-63:]) > 2.5 * sd[-63:]))
    out["drawdown_1y"] = float(c[-1] / past["high"].to_numpy()[-252:].max() - 1.0)
    out["high_1y"] = float(past["high"].to_numpy()[-252:].max())
    out["low_1y"] = float(past["low"].to_numpy()[-252:].min())
    idx = index_closes()
    before = idx.loc[:entry].iloc[:-1] if entry in idx.index else idx.loc[:entry]
    if len(before):
        last = before.iloc[-1]
        for name in idx.columns:
            out[name.lower()] = float(last[name])
    # the realised skew stickiness ratio and the vol dynamics, from the surfaces before entry
    m = day_metrics()
    hist = m.loc[:entry].iloc[:-1] if entry in m.index else m.loc[:entry]
    for window, tag in ((63, "3m"), (126, "6m")):
        h = hist.iloc[-(window + 1) :]
        if len(h) < window // 2:
            continue
        ret = np.diff(np.log(h["spot"].to_numpy()))
        for tenor in ("1m", "3m"):
            dv = np.diff(h[f"atm_{tenor}"].to_numpy())
            ok = np.isfinite(dv) & np.isfinite(ret)
            if ok.sum() < 20:
                continue
            beta = float(np.cov(dv[ok], ret[ok])[0, 1] / np.var(ret[ok], ddof=1))
            skew = float(h[f"skew_{tenor}"].mean())
            out[f"ssr_{tenor}_{tag}"] = beta / skew if skew != 0 else np.nan
    h = hist.iloc[-64:]
    if len(h) >= 40:
        lv = np.diff(np.log(h["atm_3m"].to_numpy()))
        ret = np.diff(np.log(h["spot"].to_numpy()))
        out["vol_of_vol_3m"] = float(np.std(lv, ddof=1) * np.sqrt(252.0))
        out["corr_vol_spot_3m"] = float(np.corrcoef(lv, ret)[0, 1])
    return out


def historical_gaps(entry: str, cells: pd.DataFrame) -> pd.DataFrame:
    """Rule 2's historical side: per cell, the mean undiscounted payoff on filtered historical
    paths (SPX returns strictly before the entry, EWMA-filtered, rescaled to the entry's
    at-the-money vol of the maturity, demeaned; daily strict knock) of the knock-out, the fly
    (n = 2), the tight limit and the ratio — in fractions of the entry spot."""
    ohlc: pd.DataFrame = run.context()["ohlc"]
    closes = ohlc["close"].loc[:entry].iloc[:-1]
    returns = np.log(closes).diff().dropna()
    out = pd.DataFrame(index=cells.index, columns=["h_A1", "h_B5_2", "h_B6", "h_B4"], dtype=float)
    for months, g in cells.groupby("months"):
        n = TRADING_DAYS[int(months)]
        paths = filtered_historical_paths(
            returns,
            n,
            FHS_PATHS,
            vol_target=float(g["atm"].iloc[0]),
            seed=bh.seed_of(entry) + int(months),
        )
        hi, lo, last = paths.max(axis=1), paths.min(axis=1), paths[:, -1]
        for r in g.itertuples():
            side, b = int(r.side), float(r.B / r.K)
            knocked = hi > b if side > 0 else lo < b
            legs = bh.structure_legs(side, 1.0, b, 1.0)
            ko = np.where(knocked, 0.0, np.maximum(side * (last - 1.0), 0.0))
            out.loc[r.Index, "h_A1"] = ko.mean()
            for name in ("B5_2", "B6", "B4"):
                out.loc[r.Index, f"h_{name}"] = bh.legs_payoff(legs[name], side, last).mean()
    return out


def feature_job(entry: str) -> pd.DataFrame:
    cells = pd.read_parquet(run.ENTRIES / f"{entry}.parquet").drop(columns=["legs"])
    feats = date_features(entry)
    for k, v in feats.items():
        cells[k] = v
    gaps = historical_gaps(entry, cells)
    for col in gaps.columns:
        cells[col] = gaps[col].to_numpy()
    cells["cell"] = (
        cells["entry"]
        + "|"
        + cells["months"].astype(str)
        + "|"
        + cells["side"].astype(str)
        + "|"
        + cells["barrier"]
    )
    keep = ["cell", *feats.keys(), *gaps.columns]
    return cells[keep]


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    entries = sorted(p.stem for p in run.ENTRIES.glob("*.parquet"))
    with Pool(args.workers) as pool:
        parts = list(pool.imap_unordered(feature_job, entries, chunksize=4))
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(bh.RESULTS / "features.parquet", index=False)
    print(f"features: {len(out)} cells of {len(entries)} entries, {out.shape[1] - 1} columns")


if __name__ == "__main__":
    main()
