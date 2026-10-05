"""Entry quantities of addendum 1 (BARRIER_STUDY_ADDENDUM §5.1, §7): post-processing, no pricing.

Per cell: the implied vol at the barrier, the touch weights and the annuity, the carry control,
the forward-skew premium of the local-vol knock-out over C8 (continuous and daily), the skew it
implies, the normalised spot skew at the three touch tenors, the roll and the stationary-smile
knock-out price (§4, §5.1); the directional skew-stickiness ratios, the frozen forecast of
realised vol and the vol-premium ratio, the historical gaps at the forecast vol and the richness
of each piece of the ladder, the upside skew (§7).  Everything at an entry uses data strictly
before it.  The LSV counterparts (``Pi_c[ssr12]``, ``n_eff[ssr12]``, ``dn_lsv``) are derived in
the report from these columns and the LSV marks.

    python scripts/barrier_features2.py [--workers 4] [--entries D1 D2 …]

Writes ``barrier_results/features2/<entry>.parquet``, ``barrier_results/features2.parquet`` and
``barrier_results/rv_forecast.json`` (the frozen coefficients).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
from volsto.studies import barrier_theory as bt  # noqa: E402
from volsto.studies.history_stats import filtered_historical_paths  # noqa: E402

OUT = bh.RESULTS / "features2"
FORECAST = bh.RESULTS / "rv_forecast.json"
HORIZON = {1: 21, 3: 63, 6: 126, 12: 252}
FHS_PATHS = 20_000
TENORS = {"1m": 1 / 12, "2m": 2 / 12, "3m": 0.25, "6m": 0.5, "12m": 1.0}
PIECES = {  # piece → (long, short)
    "TR": ("B6", "A1"),
    "FLYKO": ("B5_2", "A1"),
    "OS2": ("B5_2", "B6"),
    "UP": ("B3", "B5_2"),
    "W": ("B5_2", "B4"),
}

_cache: dict[str, Any] = {}


def forecast() -> dict[int, dict[str, Any]]:
    """The frozen coefficients (fitted on closes to 2006-12-29), written once."""
    if "fc" not in _cache:
        if FORECAST.exists():
            _cache["fc"] = {int(k): v for k, v in json.loads(FORECAST.read_text()).items()}
        else:
            closes = run.context()["ohlc"]["close"]
            _cache["fc"] = bt.fit_rv_forecast(closes)
    return _cache["fc"]


def rv_table() -> pd.DataFrame:
    if "rv" not in _cache:
        _cache["rv"] = bt.realised_variance_table(run.context()["ohlc"]["close"])
    return _cache["rv"]


def day_metrics() -> pd.DataFrame:
    if "metrics" not in _cache:
        rows = []
        for p in sorted(bh.DAY_CACHE.glob("*.json")):
            j = json.loads(p.read_text())
            if j.get("built"):
                rows.append({k: v for k, v in j.items() if isinstance(v, (int, float, str))})
        _cache["metrics"] = pd.DataFrame(rows).set_index("date").sort_index()
    return _cache["metrics"]


def date_features(entry: str) -> dict[str, float]:
    """The entry date's indicators of §7.1 and §7.2 (data strictly before the entry)."""
    out: dict[str, float] = {}
    m = day_metrics()
    hist = m.loc[:entry].iloc[:-1] if entry in m.index else m.loc[:entry]
    for window, tag in ((63, "3m"), (126, "6m")):
        h = hist.iloc[-(window + 1) :]
        if len(h) < 41:
            continue
        ret = np.diff(np.log(h["spot"].to_numpy(float)))
        for tenor in ("1m", "3m"):
            d_sig = np.diff(h[f"atm_{tenor}"].to_numpy(float))
            up, dn = bt.directional_ssr(d_sig, ret, float(h[f"skew_{tenor}"].mean()))
            out[f"ssr_up_{tenor}_{tag}"] = up
            out[f"ssr_dn_{tenor}_{tag}"] = dn
    rv = rv_table()
    before = rv.loc[:entry].iloc[:-1]
    if len(before):
        last = before.iloc[-1]
        trailing = [float(last[f"rv_{w}"]) for w in bt.RV_WINDOWS]
        if np.isfinite(trailing).all():
            for months, h in HORIZON.items():
                out[f"rv_fc_{months}m"] = bt.rv_forecast(trailing, forecast()[h]["coef"])
    return out


def historical(entry: str, cells: pd.DataFrame, vol_of: dict[int, float], tag: str) -> pd.DataFrame:
    """Mean undiscounted payoffs on filtered historical paths at the given vol per maturity
    (the method of ``scripts/barrier_features.py``, same seeds), for the knock-out and the
    structures of the pieces: columns ``<tag>_{A1,B3,B4,B5_2,B6}``."""
    closes = run.context()["ohlc"]["close"].loc[:entry].iloc[:-1]
    returns = np.log(closes).diff().dropna()
    names = ("B3", "B4", "B5_2", "B6")
    out = pd.DataFrame(index=cells.index, columns=[f"{tag}_A1", *[f"{tag}_{n}" for n in names]], dtype=float)
    for months, g in cells.groupby("months"):
        vol = vol_of.get(int(months), float("nan"))
        if not np.isfinite(vol):
            continue
        paths = filtered_historical_paths(
            returns, HORIZON[int(months)], FHS_PATHS, vol_target=vol, seed=bh.seed_of(entry) + int(months)
        )
        hi, lo, last = paths.max(axis=1), paths.min(axis=1), paths[:, -1]
        for r in g.itertuples():
            side, b = int(r.side), float(r.B / r.K)
            knocked = hi > b if side > 0 else lo < b
            legs = bh.structure_legs(side, 1.0, b, 1.0)
            out.loc[r.Index, f"{tag}_A1"] = np.where(knocked, 0.0, np.maximum(side * (last - 1.0), 0.0)).mean()
            for n in names:
                out.loc[r.Index, f"{tag}_{n}"] = bh.legs_payoff(legs[n], side, last).mean()
    return out


def feature_job(entry: str) -> dict[str, Any]:
    out_path = OUT / f"{entry}.parquet"
    try:
        ctx = run.context()
        ent = pd.read_parquet(run.ENTRIES / f"{entry}.parquet").drop(columns=["legs"])
        market = bh.load_day(entry, ctx["calendar"], ctx["ohlc"])
        surf = market.surface
        feats = date_features(entry)
        rows = []
        for r in ent.itertuples():
            side, K, B, T = int(r.side), float(r.K), float(r.B), float(r.T)
            w = abs(B - K) / K
            sig_b = float(np.asarray(surf.implied_vol(B, T)).reshape(()))
            tw = bt.touch_weights(K, K, B, T, sig_b)
            rr, qq = bt.carry_rates(float(r.DF0), float(r.F0), K, T)
            flat = bt.pi_flat(side, K, K, B, T, sig_b, rr, qq) / K
            pi_c = side * (r.lv_a2 - r.P_C8) / K
            pi_d = side * (r.lv_a1 - r.P_C8) / K
            A = bt.ANNUITY_CONSTANT * w * float(r.lv_p2) * tw["kapbar"]
            n_theta = [bt.normalised_skew(surf, float(t)) for t in tw["theta"]]
            n_spot = float(np.sum(tw["om"] * np.array([n for n, _ in n_theta])))
            n_eff = bt.guarded_ratio(pi_c - flat, A)
            roll = n_spot - n_eff
            n_T, _ = bt.normalised_skew(surf, T)
            atm = float(r.atm)
            far = 2 * B - K
            rec: dict[str, Any] = {
                "entry": entry,
                "months": r.months,
                "side": side,
                "barrier": r.barrier,
                "sig_B": sig_b,
                "c_touch": tw["c"],
                "kapbar": tw["kapbar"],
                "A": A,
                "r_T": rr,
                "q_T": qq,
                "Pi_flat": flat,
                "Pi_c_lv": pi_c,
                "Pi_d_lv": pi_d,
                "Pi_skew_lv": pi_c - flat,
                "n_eff_lv": n_eff,
                "n_spot": n_spot,
                "roll": roll,
                "roll_value": A * roll,
                "ko_stat": r.lv_a2 / K + side * A * roll,
                "n_T": n_T,
                "upside_skew": float(np.asarray(surf.implied_vol(far, T)).reshape(())) - atm
                if far > 0
                else np.nan,
            }
            for j in range(3):
                rec[f"pi_{j + 1}"] = tw["pi"][j]
                rec[f"kap_{j + 1}"] = tw["kap"][j]
                rec[f"om_{j + 1}"] = tw["om"][j]
                rec[f"n_theta_{j + 1}"] = n_theta[j][0]
                rec[f"n_clamped_{j + 1}"] = n_theta[j][1]
            rows.append(rec)
        res = pd.DataFrame(rows)
        for name, theta in TENORS.items():
            res[f"n_{name}"] = bt.normalised_skew(surf, theta)[0] if theta <= surf.max_maturity else np.nan
        for k, v in feats.items():
            res[k] = v
        atm_of = {int(m): float(g["atm"].iloc[0]) for m, g in ent.groupby("months")}
        fc_of = {m: feats.get(f"rv_fc_{m}m", float("nan")) for m in HORIZON}
        res["rv_fc"] = [fc_of[int(m)] for m in ent["months"]]
        res["vrp_ratio"] = ent["atm"].to_numpy(float) / res["rv_fc"]
        h1 = historical(entry, ent, atm_of, "h")  # at the entry's implied vol (h_B3 is new)
        h2 = historical(entry, ent, fc_of, "h2")  # at the forecast vol
        for frame in (h1[["h_B3"]], h2):
            for col in frame.columns:
                res[col] = frame[col].to_numpy()
        # richness of each piece: priced gap over discounted historical gap (local-vol knock-out)
        prem = {"A1": ent["lv_a1"] / ent["K"], **{n: ent[f"P_{n}"] / ent["K"] for n in ("B3", "B4", "B5_2", "B6")}}
        hist = {
            "h": {"A1": h1["h_A1"], "B3": h1["h_B3"], "B4": h1["h_B4"], "B5_2": h1["h_B5_2"], "B6": h1["h_B6"]},
            "h2": {n: h2[f"h2_{n}"] for n in ("A1", "B3", "B4", "B5_2", "B6")},
        }
        for piece, (long_, short_) in PIECES.items():
            priced = (prem[long_] - prem[short_]).to_numpy(float)
            res[f"priced_{piece}"] = priced
            for tag, hh in hist.items():
                gap = ent["DF0"].to_numpy(float) * (hh[long_] - hh[short_]).to_numpy(float)
                res[f"rich_{piece}_{tag}"] = bt.guarded_ratio(priced, gap)
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        res.to_parquet(tmp, index=False)
        os.replace(tmp, out_path)
        return {"date": entry}
    except Exception as exc:
        return {"date": entry, "error": f"{type(exc).__name__}: {exc}",
                "trace": traceback.format_exc()[-800:]}  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--entries", nargs="*", default=[])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if not FORECAST.exists():
        fc = bt.fit_rv_forecast(run.context()["ohlc"]["close"])
        FORECAST.write_text(json.dumps(fc, indent=1) + "\n")
    entries = args.entries or sorted(p.stem for p in run.ENTRIES.glob("*.parquet"))
    jobs = [e for e in entries if args.force or not (OUT / f"{e}.parquet").exists()]
    if args.workers == 1:
        res = [feature_job(e) for e in jobs]
    else:
        with Pool(args.workers) as pool:
            res = list(pool.imap_unordered(feature_job, jobs, chunksize=2))
    bad = [r for r in res if "error" in r]
    print(f"features2: {len(jobs)} run, {len(bad)} errors")
    for r in sorted(bad, key=lambda r: r["date"])[:10]:
        print(f"  {r['date']}: {r['error']}\n{r.get('trace', '')}")
    files = sorted(OUT.glob("*.parquet"))
    if files and not args.entries:
        pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).to_parquet(
            bh.RESULTS / "features2.parquet", index=False
        )
        print(f"features2.parquet: {len(files)} entries")


if __name__ == "__main__":
    main()
