"""Dispersion study: the consolidated deliverables (spec §13).

    python scripts/disp_deliver.py

``entries.parquet`` (one row per basket × tenor × date: prices, marks, sensitivities, band,
κ_Q, the indicators with their expanding percentiles and in-sample terciles, the Gaussian and
FHS forecasts), ``outcomes.parquet`` (payoffs, P&Ls in every unit, decomposition terms) and
``daily_stats.parquet`` (the realised statistics inside each window) from the per-tenor files.
"""

# ruff: noqa: E501, RUF001
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_tables as tb
import disp_tables2 as t2

from volsto.studies import disp_data as dd
from volsto.studies import disp_stats as st

DAILY = [
    "rho_real",
    "rho_pairwise",
    "vol_names_real",
    "vol_basket_real",
    "csv_annual",
    "sum_csv",
    "vr_window",
    "w_csad_g0",
    "w_csad_g1",
    "w_csad_g2",
    "rv_dispersion",
]


def main() -> None:
    frames = []
    for tenor in dd.TENORS:
        for basket, suffix in (("B1", ""), ("B2", ""), ("B3", "_B3")):
            if not (dd.OUT / f"outcomes_{tenor}{suffix}.parquet").exists():
                continue
            try:
                d = tb.load(tenor, basket, suffix)
            except Exception as exc:
                print(f"{basket} {tenor}: {exc}")
                continue
            if d.empty:
                continue
            if basket in ("B1", "B2") and (dd.OUT / f"fhs_{tenor}.parquet").exists():
                d = t2.add_fhs(d, tenor, basket)
            for col in tb.INDICATORS.values():
                if col in d:
                    d[f"pct_{col}"] = st.expanding_percentile(d[col])
                    d[f"tercile_{col}"] = st.tercile(d[col], tb.is_cuts(d, col))
            # P&L in the risk units of spec §5.4
            for col in tb.STRUCTS.values():
                if col not in d:
                    continue
                prem = tb.premium_of(d, col)
                if prem is not None:
                    d[f"{col}_per_premium"] = d[col] / prem.where(prem > 0)
                rho_s, vol_s = tb.sensitivities(d, col)
                for tag, sens in (("corr_pts", rho_s.abs()), ("vol_pts", vol_s)):
                    ok = sens.abs() >= 0.25 * sens.abs().median()
                    d[f"{col}_{tag}"] = (d[col] / sens).where(ok)
            frames.append(d)
            print(f"{basket} {tenor}: {len(d)} dates, {int(d['has_outcome'].sum())} windows")
    full = pd.concat(frames, ignore_index=True)
    key = ["date", "basket", "tenor"]
    pnl = [
        c
        for c in full.columns
        if c.endswith(("_U", "_H", "_S"))
        or "_per_premium" in c
        or "_corr_pts" in c
        or "_vol_pts" in c
        or c.startswith(("PC_pay_", "dec_"))
    ]
    payoff = ["Rb", "D", "SD", "G", "V", "absR", "absRb", "D_rel", "kappa", "kurtosis", "n_up", "n_down", "n_above_basket", "VD", "VD_vega",
              "hedge_SS", "hedge_BS", "hedge_PF", "traded_SS", "traded_BS", "traded_PF", "basket_event", "corporate_action", "membership_change",
              "carried_days", "ret_DJX", "ret_DIA", "track_DJX", "track_DIA", "sandwich_violation", "gap_formula_error", "c11_error"]  # fmt: skip
    out_cols = [
        c for c in [*key, "expiry", "IS", "has_outcome", *payoff, *pnl] if c in full.columns
    ]
    full[out_cols].to_parquet(dd.OUT / "outcomes.parquet", index=False)
    full[[c for c in [*key, "expiry", *DAILY] if c in full.columns]].to_parquet(
        dd.OUT / "daily_stats.parquet", index=False
    )
    drop = set(payoff) | set(pnl) | set(DAILY)
    full[[c for c in full.columns if c not in drop]].to_parquet(
        dd.OUT / "entries.parquet", index=False
    )
    print(
        f"entries.parquet {len(full)} rows × {len([c for c in full.columns if c not in drop])} columns; outcomes.parquet {len(out_cols)} columns; daily_stats.parquet"
    )


if __name__ == "__main__":
    main()
