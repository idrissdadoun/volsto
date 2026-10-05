"""Dispersion study: the filtered-historical-simulation forecasts at every entry (spec §7.2).

    python scripts/disp_fhs_run.py --tenor 3m [--workers 4] [--block 10] [--no-demean]

For each entry and basket: the trailing 756 trading days (plus 21 to seed the EWMA) on which
every entry-basket name has a return, strictly before the entry date; 4,000 paths of the
window by stationary bootstrap of whole days; the mean and standard deviation of the payoff
of every unhedged structure; forecast edge = mean payoff − price, forecast Sharpe = edge over
the standard deviation.  Writes ``fhs_<tenor>[_tag].parquet``.  A date with fewer than 252
common days is skipped and counted.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import os
from multiprocessing import Pool
from typing import Any

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from volsto.studies import disp_data as dd  # noqa: E402
from volsto.studies import disp_fhs as fhs  # noqa: E402
from volsto.studies import disp_indicators as di  # noqa: E402

MULT = ("050", "075", "100", "125", "150", "200")
_state: dict[str, Any] = {}


def init(tenor: str, block: int, demean: bool) -> None:
    _state["returns"] = pd.read_parquet(dd.OUT / "returns.parquet")
    _state["legs"] = {
        d: g for d, g in pd.read_parquet(dd.OUT / f"legs_{tenor}.parquet").groupby("date")
    }
    _state["block"], _state["demean"], _state["h"] = block, demean, dd.TENORS[tenor]


def job(e: dict[str, Any]) -> dict[str, Any]:
    g = _state["legs"][e["date"]]
    names = list(g["ticker"])
    w = g["w_B1"].to_numpy(float) if e["basket"] == "B1" else np.full(len(names), 1.0 / len(names))
    hist = di.trailing(_state["returns"], e["date"], names, 756 + fhs.SEED_DAYS)
    hist = hist[np.isfinite(hist).all(axis=1)]
    row: dict[str, Any] = {
        "date": e["date"],
        "basket": e["basket"],
        "tenor": e["tenor"],
        "fhs_days": int(hist.shape[0]),
    }
    if hist.shape[0] < fhs.MIN_DAYS + fhs.SEED_DAYS:
        return row
    sig_fc = di.forecast_vol(hist[-252:])
    seed = int(e["date"].replace("-", "")) * 10 + (1 if e["basket"] == "B1" else 2)
    X = fhs.simulate(
        hist, sig_fc, _state["h"], seed, mean_block=_state["block"], demean=_state["demean"]
    )
    strikes = np.array([e[f"K_{m}"] for m in MULT])
    s = fhs.summary(X, w, strikes)
    t, calls = s.pop("_t"), s.pop("_calls")
    for k in (
        "E_D",
        "E_V",
        "E_G",
        "E_SD",
        "E_absR",
        "E_absRb",
        "E_D_rel",
        "kappa_fc",
        "rho_fhs",
        "sd_D",
        "sd_G",
    ):
        row[f"fhs_{k}"] = s[k]
    ss, bs, pf = t["absR"] - e["SS_mkt"], t["absRb"] - e["Str_B_mkt"], t["D"] - e["P_D"]
    pnl = {
        "PF": pf, "SS": ss, "BS": bs, "PKG_v": ss - bs, "PKG_theta": ss - e["lam_theta"] * bs,
        "PKG_rho": ss - e["lambda_rho"] * bs, "GAP": pf - (ss - bs), "GAP_rho": pf - (ss - e["lambda_rho"] * bs),
        "PF_v": pf - e["h_v"] * ss, "REV": -(ss - bs),
        "PC_100_S": calls[:, 2] - e["C_100"] - e["delta_c_100"] * t["Rb"],
    }  # fmt: skip
    for j, m in enumerate(MULT):
        pnl[f"PC_{m}"] = calls[:, j] - e[f"C_{m}"]
    for k, v in pnl.items():
        row[f"fedge_{k}"] = float(np.mean(v))
        row[f"fsd_{k}"] = float(np.std(v))
        row[f"fsr_{k}"] = float(np.mean(v) / np.std(v)) if np.std(v) > 0 else np.nan
    row["GP_F"] = e["P_G"] / s["E_G"] if s["E_G"] > 0 else np.nan
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--tenor", default="3m", choices=list(dd.TENORS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--block", type=int, default=fhs.MEAN_BLOCK)
    ap.add_argument("--no-demean", action="store_true")
    ap.add_argument("--baskets", nargs="*", default=["B1", "B2"])
    args = ap.parse_args()
    entries = pd.read_parquet(dd.OUT / f"entries_{args.tenor}.parquet")
    entries = entries[entries["basket"].isin(args.baskets)]
    jobs = entries.to_dict("records")
    workers = max(1, min(args.workers, dd.cpu_budget()))
    with Pool(
        workers, initializer=init, initargs=(args.tenor, args.block, not args.no_demean)
    ) as pool:
        rows = pool.map(job, jobs, chunksize=8)
    out = pd.DataFrame(rows)
    tag = ("" if args.block == fhs.MEAN_BLOCK else f"_block{args.block}") + (
        "_undemeaned" if args.no_demean else ""
    )
    out.to_parquet(dd.OUT / f"fhs_{args.tenor}{tag}.parquet", index=False)
    done = out["fhs_E_D"].notna() if "fhs_E_D" in out else pd.Series(False, index=out.index)
    print(
        f"fhs {args.tenor}{tag}: {len(out)} basket-dates, {int(done.sum())} simulated, {int((~done).sum())} skipped (fewer than 252 common days); first date {out.loc[done, 'date'].min() if done.any() else '-'}"
    )


if __name__ == "__main__":
    main()
