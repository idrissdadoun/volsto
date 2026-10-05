"""Dispersion study: smiles, copula prices and sensitivities at every entry date (spec §2–§3).

    python scripts/disp_entries.py --tenor 3m [--workers 5] [--vendor] [--dates D1 D2 …] [--today]

For each entry date: the chains of the thirty members, DJX and DIA from the ORATS store; the
smiles (forward rule of check C1); the marginals at the tenor; the correlation ``ρ_cop`` that
reprices the DJX straddle; the Palladium forward, calls and sensitivities of the price-weighted
basket B1 and of the equally weighted B2 at that mark; implied correlations and dispersions.
One pickle per date under ``outputs/dispersion/entries/<tenor>[_vendor]/``; ``--vendor`` uses
``smoothSmvVol`` for every leg (the sensitivity of §1.1).  The worker count is capped by the
CPU budget of spec §0.3.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import ndtr, ndtri  # noqa: E402

from volsto.studies import disp_copula as dc  # noqa: E402
from volsto.studies import disp_data as dd  # noqa: E402
from volsto.studies import disp_smile as ds  # noqa: E402
from volsto.studies import disp_universe as du  # noqa: E402

BASKETS = ("DJX", "DIA")
TERMS = {"1m": 21, "3m": 63, "12m": 252}


#: "parity" (the study's rule, PROGRESS_Q2 section 2) or "dividends" (rule (c) of spec §1.1,
#: kept as a sensitivity: projected dividends for single names and DIA where a history exists).
RULE = "parity"


def forward_rule(ticker: str, date: str) -> tuple[str, Any]:
    if RULE == "parity" or ticker == "DJX":
        return "parity", None
    div = dd.projected_dividends(ticker, date)
    return ("dividends", div) if div is not None else ("parity", None)


def smiles_of(
    ticker: str, date: str, chain: pd.DataFrame | None
) -> tuple[list[ds.ExpirySmile], str, str]:
    """``(expiry smiles, the day they are from, the rule)``; the previous trading days (at most
    five) are tried when the day has no usable smile."""
    days = dd.trading_days()
    i = days.index(date)
    for back in range(0, 6):
        d = days[i - back]
        ch = chain if back == 0 else dd.read_chains(d, [ticker]).get(ticker)
        if ch is None or ch.empty:
            continue
        rule, div = forward_rule(ticker, d)
        px = float(dd.prices().at[d, ticker]) if ticker in dd.prices().columns else float("nan")
        sm = ds.expiry_smiles(ch, d, rule, dividends=div, spot=px if np.isfinite(px) else None)
        if sm:
            return sm, d, rule
    return [], "", ""


def half_spread(chain: pd.DataFrame | None, spot: float, date: str, T: float) -> float:
    """Half the bid-ask of the straddle at the listed expiry nearest ``T`` and strike nearest
    the spot, in units of the spot (NaN when not quoted)."""
    if chain is None or chain.empty:
        return float("nan")
    exps = chain["expirDate"].unique()
    days = np.array([(pd.Timestamp(e) - pd.Timestamp(date)).days for e in exps])
    e = exps[int(np.argmin(np.abs(days - 365.0 * T)))]
    g = chain[chain["expirDate"] == e]
    r = g.iloc[int(np.argmin(np.abs(g["strike"].to_numpy(float) - spot)))]
    hs = 0.5 * ((r["cAskPx"] - r["cBidPx"]) + (r["pAskPx"] - r["pBidPx"]))
    return float(hs / spot) if np.isfinite(hs) and hs >= 0 else float("nan")


def straddle_vol(straddle: float, T: float) -> float:
    """The lognormal vol of an at-the-money-forward straddle worth ``straddle``."""
    return float(2.0 * ndtri(min(0.5 * (0.5 * straddle + 1.0), 1 - 1e-12)) / np.sqrt(T))


_Z = np.linspace(-ds.Z_MAX, ds.Z_MAX, ds.Z_POINTS)
_PDF = np.exp(-0.5 * _Z * _Z) / np.sqrt(2.0 * np.pi)


def table_straddle(m: ds.Marginal) -> float:
    """The at-the-money-spot straddle of the marginal's own quantile table (quadrature)."""
    return float(np.trapezoid(np.abs(m.table() - 1.0) * _PDF, _Z))


def put_at(m: ds.Marginal, strike: float) -> float:
    return float(np.interp(strike, m.strike, m.call) - (m.f - strike))


def entry_job(args: tuple[str, str, int, bool, str, bool, str]) -> dict[str, Any]:
    global RULE
    date, label, n_days, vendor, out_dir, monthly, RULE = args
    out_path = Path(out_dir) / f"{date}.pkl"
    t0 = time.time()
    try:
        names = du.members_on(date)
        chains = dd.read_chains(date, names + list(BASKETS))
        panel = dd.prices().ffill()
        smiles: dict[str, list[ds.ExpirySmile]] = {}
        smile_day: dict[str, str] = {}
        rules: dict[str, str] = {}
        for t in names + list(BASKETS):
            sm, d, rule = smiles_of(t, date, chains.get(t))
            if sm:
                smiles[t], smile_day[t], rules[t] = sm, d, rule
        missing = [t for t in names if t not in smiles]
        carried = [t for t in names if t in smiles and smile_day[t] != date]
        if missing or len(carried) > 2:
            return {"date": date, "skipped": f"no usable smile for {missing}; carried {carried}"}
        spots = np.array([float(panel.at[date, t]) for t in names])
        if not np.all(np.isfinite(spots) & (spots > 0)):
            return {"date": date, "skipped": "a member has no price"}
        w1 = spots / spots.sum()
        w2 = np.full(len(names), 1.0 / len(names))
        T = dd.maturity(date, n_days)
        tenor = {t: ds.smile_at(smiles[t], float(panel.at[date, t]), T, vendor) for t in smiles}
        margs = [ds.build_marginal(tenor[t]) for t in names]
        # a marginal whose own straddle is more than 1 % from the smile's (non-standard rows on
        # a corporate-action day) is rebuilt from the vendor's smoothed vols, and flagged
        fallback = []
        for j, t in enumerate(names):
            if abs(table_straddle(margs[j]) / margs[j].straddle - 1.0) > 0.01 and not vendor:
                tenor[t] = ds.smile_at(smiles[t], float(panel.at[date, t]), T, True)
                margs[j] = ds.build_marginal(tenor[t])
                fallback.append(t)
        bumped = [ds.build_marginal(tenor[t], 0.01) for t in names]
        res: dict[str, Any] = {
            "date": date, "tenor": label, "n_days": n_days, "T": T, "expiry": dd.expiry_of(date, n_days),
            "names": names, "spots": spots, "w_B1": w1, "vendor": vendor, "monthly": monthly,
            "carried_smiles": carried, "vendor_fallback": fallback, "rules": [rules[t] for t in names],
            "strikes_dropped": int(sum(e.n_dropped for t in names for e in smiles[t])),
            "strikes_kept": int(sum(len(e.k) for t in names for e in smiles[t])),
            "rate": float(np.mean([m.rate for m in margs])),
        }  # fmt: skip
        vols = np.array([m.atm_vol for m in margs])
        leg = {
            "f": np.array([m.f for m in margs]), "atm_vol": vols,
            "straddle": np.array([m.straddle for m in margs]),
            "vega": np.array([m.vega for m in margs]), "M": np.array([m.M for m in margs]),
            "var_vs": np.array([m.var_vs for m in margs]),
            "vol_90": np.array([m.vol_90 for m in margs]), "vol_110": np.array([m.vol_110 for m in margs]),
            "skew": np.array([m.skew for m in margs]), "tail_share": np.array([m.tail_share for m in margs]),
            "bracket_lo": [tenor[t].lo.expiry for t in names], "bracket_hi": [tenor[t].hi.expiry for t in names],
            "half_spread": np.array([half_spread(chains.get(t), s, date, T) for t, s in zip(names, spots, strict=True)]),
        }  # fmt: skip
        res["legs"] = leg
        # the listed baskets
        bk: dict[str, dict[str, Any]] = {}
        for b in BASKETS:
            if b not in tenor:
                continue
            m = ds.build_marginal(tenor[b])
            lo, hi = tenor[b].lo, tenor[b].hi
            bk[b] = {
                "straddle": m.straddle, "M": m.M, "atm_vol": m.atm_vol, "var_vs": m.var_vs, "vol_90": m.vol_90,
                "f": m.f, "put_90": put_at(m, 0.9), "bracketed": bool(lo.T <= T <= hi.T and lo is not hi),
                "tail_share": m.tail_share, "smile_day": smile_day[b], "n_expiries": len(smiles[b]),
                "half_spread": half_spread(chains.get(b), float(panel.at[date, b]), date, T),
                "vega": m.vega, "skew": m.skew,
            }  # fmt: skip
        res["baskets"] = bk
        primary = "DJX" if "DJX" in bk else ("DIA" if "DIA" in bk else "")
        if not primary:
            return {"date": date, "skipped": "no listed basket smile"}
        res["primary"] = primary
        draws = dc.sobol_draws(len(names), seed=int(date.replace("-", "")) * 1000 + n_days)
        tab = dc.tables(margs)
        target = bk[primary]["straddle"]
        rho_cop, flag = dc.calibrate_rho(tab, draws, w1, target)
        res["rho_cop"], res["rho_cop_flag"] = rho_cop, flag
        res["c4_error"] = (
            abs(dc.basket_straddle(tab, draws, w1, rho_cop) - target) if not flag else float("nan")
        )
        if "DIA" in bk and primary == "DJX":
            res["rho_cop_DIA"], res["rho_cop_DIA_flag"] = dc.calibrate_rho(
                tab, draws, w1, bk["DIA"]["straddle"], xtol=1e-6
            )
        sig_b = bk[primary]["atm_vol"]
        rho_atm = dc.cboe_correlation(sig_b, vols, w1)
        rho_90 = dc.cboe_correlation(bk[primary]["vol_90"], leg["vol_90"], w1)
        extras = None
        if monthly:
            extras = {
                "up5": rho_cop + 0.05,
                "dn5": rho_cop - 0.05,
                "rho90": float(np.clip(rho_90, 0.0, 0.99)),
            }
            res["rho_V"], res["rho_V_flag"] = dc.calibrate_rho_moment(
                tab, draws, w1, bk[primary]["M"]
            )
        for tag, w in (("B1", w1), ("B2", w2)):
            o = dc.price_basket(
                margs, bumped, w, draws, rho_cop, extras=extras if tag == "B1" else None
            )
            clip = float(np.clip(rho_cop, -0.05, 0.99))
            _, _, sig_rel = dc.gaussian_inputs(vols, w, clip)
            o["sig_rel"] = sig_rel
            o["gauss_41"] = float(np.sqrt(2.0 * T / np.pi) * np.sum(w * sig_rel))
            o["gauss_42"] = float(np.sum(w * 2.0 * (2.0 * ndtr(0.5 * sig_rel * np.sqrt(T)) - 1.0)))
            o["SS_mkt"] = float(np.sum(w * leg["straddle"]))
            o["EQV_strip_names"] = float(np.sum(w * leg["M"]))
            o["sig_bar"] = float(np.sum(w * vols))
            o["half_spread_SS"] = float(np.nansum(w * leg["half_spread"]))
            if tag == "B1":
                o["Str_B_mkt"] = target
                o["sig_B"] = sig_b
                o["EQV"] = o["EQV_strip_names"] - bk[primary]["M"]
            else:  # no listed price: the basket straddle and its vol are the model's
                o["Str_B_mkt"] = o["Str_B"]
                o["sig_B"] = straddle_vol(o["Str_B"], T)
                o["EQV"] = o["EV"]
            o["lam_theta"] = o["sig_bar"] / o["sig_B"]
            o["P_G"] = o["P_D"] - (o["SS_mkt"] - o["Str_B_mkt"])
            o["kappa_Q"] = o["P_D"] / np.sqrt(o["EQV"]) if o["EQV"] > 0 else np.nan
            lower, upper = o["SS_mkt"] - o["Str_B_mkt"], np.sqrt(max(o["EQV"], 0.0))
            o["band_lo"], o["band_hi"] = lower, upper
            o["band_pos"] = (o["P_D"] - lower) / (upper - lower) if upper > lower else np.nan
            res[tag] = o
        res.update(
            rho_atm=rho_atm, rho_90=rho_90,
            ID2=float(np.sum(w1 * vols**2) - sig_b**2),
            ID2_VS=float(np.sum(w1 * leg["var_vs"]) - bk[primary]["var_vs"]),
        )  # fmt: skip
        # term structure of the implied correlation, the front event premium, ID_VS at 30 days
        term: dict[str, Any] = {}
        for tl, nd in TERMS.items():
            Tt = nd / 252.0
            v = np.array(
                [ds.smile_at(smiles[t], 1.0, Tt, vendor).vol(0.0) for t in names], dtype=float
            )
            vb = float(ds.smile_at(smiles[primary], 1.0, Tt, vendor).vol(0.0))
            term[f"rho_atm_{tl}"] = dc.cboe_correlation(vb, v, w1)
            term[f"sig_bar_{tl}"] = float(np.sum(w1 * v))
            term[f"sig_B_{tl}"] = vb
        T30 = 30.0 / 365.0
        vs30 = np.array(
            [
                ds.build_marginal(
                    ds.smile_at(smiles[t], float(panel.at[date, t]), T30, vendor)
                ).var_vs
                for t in names
            ]
        )
        vsb30 = ds.build_marginal(
            ds.smile_at(smiles[primary], float(panel.at[date, primary]), T30, vendor)
        ).var_vs
        term["ID2_VS_30d"] = float(np.sum(w1 * vs30) - vsb30)
        res["term"] = term
        res["seconds"] = time.time() - t0
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_suffix(".tmp")
        with tmp.open("wb") as fh:
            pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, out_path)
        return {"date": date, "seconds": res["seconds"], "rho_cop": rho_cop, "flag": flag}
    except Exception as exc:
        return {
            "date": date,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-900:],
        }


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--tenor", default="3m", choices=list(dd.TENORS))
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--vendor", action="store_true")
    ap.add_argument("--dates", nargs="*", default=[])
    ap.add_argument(
        "--today", action="store_true", help="also price the last day of the store (no outcome)"
    )
    ap.add_argument("--monthly-only", action="store_true")
    ap.add_argument("--rule", default="parity", choices=["parity", "dividends"])
    args = ap.parse_args()
    n_days = dd.TENORS[args.tenor]
    out_dir = (
        dd.OUT
        / "entries"
        / (
            args.tenor
            + ("_vendor" if args.vendor else "")
            + ("_dividends" if args.rule == "dividends" else "")
        )
    )
    entries = dd.entry_dates(n_days)
    monthly = dd.monthly_subset(entries)
    todo = args.dates or (sorted(monthly) if args.monthly_only else entries)
    if args.today and dd.LAST_DAY not in todo:
        todo = [*todo, dd.LAST_DAY]
    jobs = [
        (
            d,
            args.tenor,
            n_days,
            args.vendor,
            str(out_dir),
            d in monthly or d == dd.LAST_DAY,
            args.rule,
        )
        for d in todo
        if not (out_dir / f"{d}.pkl").exists()
    ]
    workers = max(1, min(args.workers, dd.cpu_budget()))
    t0 = time.time()
    if workers == 1 or len(jobs) <= 1:
        res = [entry_job(j) for j in jobs]
    else:
        with Pool(workers) as pool:
            res = list(pool.imap_unordered(entry_job, jobs, chunksize=2))
    bad = [r for r in res if "error" in r]
    skipped = [r for r in res if "skipped" in r]
    print(
        f"entries {args.tenor}{' vendor' if args.vendor else ''}: {len(jobs)} dates, {len(bad)} errors, {len(skipped)} skipped, {time.time() - t0:.0f} s on {workers} processes"
    )
    for r in sorted(skipped, key=lambda r: r["date"]):
        print(f"  skipped {r['date']}: {r['skipped']}")
    for r in sorted(bad, key=lambda r: r["date"])[:8]:
        print(f"  ERROR {r['date']}: {r['error']}\n{r.get('trace', '')}")
    flags = [r for r in res if r.get("flag")]
    print(
        f"  rho_cop clipped on {len(flags)} dates: {[(r['date'], r['flag']) for r in flags][:12]}"
    )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
