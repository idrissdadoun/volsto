"""Dispersion study: basket B3, ten large caps at equal weights (spec §1.2, §3.6).

    python scripts/disp_b3.py [--tenor 3m] [--workers 5]

AAPL, MSFT, AMZN, NVDA, JPM, XOM, JNJ, PG, HD, UNH, chosen in 2026 (survivorship: every
caption says so).  The basket has no listed options: its straddle is the copula's at the Dow's
``ρ_cop`` of the same date, and it uses the Dow's correlation indicators.  One pickle per date
under ``entries/<tenor>_B3/``, in the layout of ``disp_entries.py`` (key ``B3``), for
``disp_outcomes.py --suffix _B3`` and ``disp_indicators.py --suffix _B3``.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import os
import pickle
import sys
import traceback
from multiprocessing import Pool
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import numpy as np  # noqa: E402
from scipy.special import ndtr  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_entries as de  # noqa: E402

from volsto.studies import disp_copula as dc  # noqa: E402
from volsto.studies import disp_data as dd  # noqa: E402
from volsto.studies import disp_smile as ds  # noqa: E402
from volsto.studies import disp_universe as du  # noqa: E402


def job(args: tuple[str, str]) -> dict[str, Any]:
    path, out_dir = args
    with open(path, "rb") as fh:
        r = pickle.load(fh)
    date = r["date"]
    try:
        names = list(du.B3_NAMES)
        got = de.marginals_for(date, names, r["T"])
        if got is None:
            return {"date": date, "skipped": "a name has no usable smile"}
        margs = [got[t][0] for t in names]
        bumped = [ds.build_marginal(got[t][1], 0.01) for t in names]
        panel = dd.prices().ffill()
        spots = np.array([float(panel.at[date, t]) for t in names])
        w = np.full(len(names), 1.0 / len(names))
        draws = dc.sobol_draws(len(names), seed=int(date.replace("-", "")) * 1000 + r["n_days"] + 3)
        rho = r["rho_cop"]
        o = dc.price_basket(margs, bumped, w, draws, rho)
        vols = np.array([m.atm_vol for m in margs])
        _, _, sig_rel = dc.gaussian_inputs(vols, w, float(np.clip(rho, -0.05, 0.99)))
        T = r["T"]
        o["sig_rel"] = sig_rel
        o["gauss_41"] = float(np.sqrt(2.0 * T / np.pi) * np.sum(w * sig_rel))
        o["gauss_42"] = float(np.sum(w * 2.0 * (2.0 * ndtr(0.5 * sig_rel * np.sqrt(T)) - 1.0)))
        leg = {
            "f": np.array([m.f for m in margs]), "atm_vol": vols, "straddle": np.array([m.straddle for m in margs]),
            "vega": np.array([m.vega for m in margs]), "M": np.array([m.M for m in margs]), "var_vs": np.array([m.var_vs for m in margs]),
            "vol_90": np.array([m.vol_90 for m in margs]), "vol_110": np.array([m.vol_110 for m in margs]),
            "skew": np.array([m.skew for m in margs]), "tail_share": np.array([m.tail_share for m in margs]),
            "bracket_lo": [got[t][1].lo.expiry for t in names], "bracket_hi": [got[t][1].hi.expiry for t in names],
            "half_spread": np.full(len(names), np.nan),
        }  # fmt: skip
        o["SS_mkt"] = float(np.sum(w * leg["straddle"]))
        o["EQV_strip_names"] = float(np.sum(w * leg["M"]))
        o["sig_bar"] = float(np.sum(w * vols))
        o["half_spread_SS"] = float("nan")
        o["Str_B_mkt"] = o["Str_B"]  # no listed price: the model's
        o["sig_B"] = de.straddle_vol(o["Str_B"], T)
        o["EQV"] = o["EV"]
        o["lam_theta"] = o["sig_bar"] / o["sig_B"]
        o["P_G"] = o["P_D"] - (o["SS_mkt"] - o["Str_B_mkt"])
        o["kappa_Q"] = o["P_D"] / np.sqrt(o["EQV"])
        lower, upper = o["SS_mkt"] - o["Str_B_mkt"], float(np.sqrt(o["EQV"]))
        o["band_lo"], o["band_hi"] = lower, upper
        o["band_pos"] = (o["P_D"] - lower) / (upper - lower) if upper > lower else np.nan
        res = {k: r[k] for k in ("date", "tenor", "n_days", "T", "expiry", "vendor", "monthly", "rate", "baskets", "primary", "rho_cop", "rho_cop_flag",
                                 "c4_error", "rho_atm", "rho_90", "ID2", "ID2_VS", "term") if k in r}  # fmt: skip
        res.update(
            names=names,
            spots=spots,
            w_B1=w,
            carried_smiles=[],
            vendor_fallback=[],
            rules=["parity"] * len(names),
            legs=leg,
            B3=o,
        )
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with (Path(out_dir) / f"{date}.pkl").open("wb") as fh:
            pickle.dump(res, fh, protocol=pickle.HIGHEST_PROTOCOL)
        return {"date": date}
    except Exception as exc:
        return {
            "date": date,
            "error": f"{type(exc).__name__}: {exc}",
            "trace": traceback.format_exc()[-700:],
        }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--tenor", default="3m")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    out_dir = dd.OUT / "entries" / f"{args.tenor}_B3"
    jobs = [
        (str(p), str(out_dir))
        for p in sorted((dd.OUT / "entries" / args.tenor).glob("*.pkl"))
        if not (out_dir / p.name).exists()
    ]
    with Pool(max(1, min(args.workers, dd.cpu_budget()))) as pool:
        res = list(pool.imap_unordered(job, jobs, chunksize=4))
    bad = [r for r in res if "error" in r]
    skipped = [r for r in res if "skipped" in r]
    print(f"B3 {args.tenor}: {len(jobs)} dates, {len(bad)} errors, {len(skipped)} skipped")
    for r in bad[:5]:
        print(r)
    import subprocess

    for script in ("disp_outcomes.py", "disp_indicators.py"):
        subprocess.run(
            [
                sys.executable,
                "-W",
                "ignore",
                str(Path(__file__).resolve().parent / script),
                "--tenor",
                args.tenor,
                "--suffix",
                "_B3",
            ],
            check=False,
        )


if __name__ == "__main__":
    main()
