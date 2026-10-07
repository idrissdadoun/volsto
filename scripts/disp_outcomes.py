"""Dispersion study: realised outcomes of every window (spec §5) and the assembled tables.

    python scripts/disp_outcomes.py --tenor 3m [--suffix _vendor]

Reads the entry pickles of ``scripts/disp_entries.py`` and the price panel, and writes
``outputs/dispersion/entries_<tenor>.parquet`` (one row per basket and date: prices, marks,
sensitivities, band, implied correlations) and ``outcomes_<tenor>.parquet`` (payoffs at expiry,
daily statistics inside the window, hedge P&Ls, the P&L of every structure, the exact
decomposition terms, the tracking of DJX and DIA).  Single process; no pricing.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import pickle
from typing import Any

import numpy as np
import pandas as pd

from volsto.studies import disp_copula as dc
from volsto.studies import disp_data as dd
from volsto.studies import disp_payoff as dp
from volsto.studies import disp_universe as du

MULT = ("050", "075", "100", "125", "150", "200")


def window_path(
    panel: pd.DataFrame, filled: pd.DataFrame, names: list[str], days: list[str]
) -> tuple[np.ndarray, bool, int]:
    """``(N + 1) × n`` values of the frozen holdings per unit of their entry prices, whether a
    corporate action fell inside, and the number of (name, day) valued with a carried price."""
    entry, last = days[0], days[-1]
    base = filled.loc[days, names].to_numpy(float)
    path = base / base[0]
    carried = int(panel.loc[days, names].isna().to_numpy().sum())
    event = False
    for j, t in enumerate(names):
        if len(du.holding_path(t, entry, last)) > 1:
            x, _, c = du.holding_values(panel, t, entry, days)
            path[:, j] = x
            carried += c
            event = True
    return path, event, carried


def entry_row(r: dict[str, Any], tag: str) -> dict[str, Any]:
    o = r[tag]
    bk = r["baskets"]
    row: dict[str, Any] = {
        "date": r["date"], "basket": tag, "tenor": r["tenor"], "T": r["T"], "expiry": r["expiry"],
        "monthly": r["monthly"], "rate": r["rate"], "primary": r["primary"],
        "rho_cop": r["rho_cop"], "rho_cop_flag": r["rho_cop_flag"], "c4_error": r["c4_error"],
        "rho_cop_DIA": r.get("rho_cop_DIA", np.nan), "rho_atm": r["rho_atm"], "rho_90": r["rho_90"],
        "rho_V": r.get("rho_V", np.nan),
        "ID": 100.0 * np.sqrt(max(r["ID2"], 0.0)), "ID2": r["ID2"],
        "ID_VS": 100.0 * np.sqrt(max(r["ID2_VS"], 0.0)), "ID2_VS": r["ID2_VS"],
        "ID_VS_30d": 100.0 * np.sqrt(max(r["term"]["ID2_VS_30d"], 0.0)),
        "n_carried_smiles": len(r["carried_smiles"]), "n_vendor_fallback": len(r.get("vendor_fallback", [])),
        "strikes_dropped": r.get("strikes_dropped", 0), "strikes_kept": r.get("strikes_kept", 0),
        "n_parity": int(sum(x == "parity" for x in r["rules"])),
        "sig_B_DJX": bk.get("DJX", {}).get("atm_vol", np.nan), "sig_B_DIA": bk.get("DIA", {}).get("atm_vol", np.nan),
        "Str_DJX": bk.get("DJX", {}).get("straddle", np.nan), "Str_DIA": bk.get("DIA", {}).get("straddle", np.nan),
        "DJX_bracketed": bk.get("DJX", {}).get("bracketed", False), "DIA_bracketed": bk.get("DIA", {}).get("bracketed", False),
        "put90_B": bk.get(r["primary"], {}).get("put_90", np.nan), "vol90_B": bk.get(r["primary"], {}).get("vol_90", np.nan),
        "half_spread_B": bk.get(r["primary"], {}).get("half_spread", np.nan),
        "f_B": bk.get(r["primary"], {}).get("f", np.nan),
        "tail_share_max": float(max(np.max(r["legs"]["tail_share"]), bk.get(r["primary"], {}).get("tail_share", 0.0))),
        **{k: v for k, v in r["term"].items()},
    }  # fmt: skip
    for k in ("P_D", "P_D_se", "sd_D", "Str_B", "Str_B_mkt", "SS", "SS_mkt", "EV", "EQV", "kappa_cop", "kappa_Q",
              "band_lo", "band_hi", "band_pos", "P_G", "E_B", "E_DB", "dPD_drho", "dStrB_drho", "dEV_drho", "lambda_rho",
              "dPD_dvol", "dStrB_dvol", "dSS_dvol", "vega_SS", "h_v", "lam_theta", "sig_B", "sig_bar", "gauss_41",
              "gauss_42", "half_spread_SS", "M_B"):  # fmt: skip
        row[k] = float(o.get(k, np.nan))
    for j, m in enumerate(MULT):
        row[f"K_{m}"] = float(o["strikes"][j])
        row[f"C_{m}"] = float(o["calls"][j])
        row[f"C_se_{m}"] = float(o["calls_se"][j])
        row[f"dC_drho_{m}"] = float(o["dcalls_drho"][j])
        row[f"dC_dvol_{m}"] = float(o["dcalls_dvol"][j])
        row[f"delta_c_{m}"] = float(o["delta_c"][j])
        row[f"delta_c_fd_{m}"] = float(o["delta_c_fd"][j])
    for j in range(len(dc.PROFILE_LABELS)):
        row[f"profile_{j}"] = float(o["profile"][j])
        row[f"profile_share_{j}"] = float(o["profile_share"][j])
    for name in ("rep_abs", "rep_sq"):
        for j, c in enumerate(("c", "a", "b", "r2")):
            row[f"{name}_{c}"] = float(o[name][j])
    for lab in ("up5", "dn5", "rho90"):
        if f"P_D@{lab}" in o:
            row[f"P_D_{lab}"] = float(o[f"P_D@{lab}"])
            row[f"C_100_{lab}"] = float(o[f"calls@{lab}"][2])
            row[f"C_150_{lab}"] = float(o[f"calls@{lab}"][4])
    # check C3: the copula against each single-name straddle
    rel = o["straddle_cop"] / r["legs"]["straddle"] - 1.0
    z = (o["straddle_cop"] - r["legs"]["straddle"]) / np.maximum(o["straddle_se"], 1e-12)
    row["c3_max_rel"], row["c3_max_z"] = float(np.max(np.abs(rel))), float(np.max(np.abs(z)))
    return row


def outcome_row(
    r: dict[str, Any],
    tag: str,
    path: np.ndarray,
    event: bool,
    carried: int,
    track: dict[str, float],
    changed: bool,
) -> dict[str, Any]:
    o = r[tag]
    w = r["w_B1"] if tag == "B1" else np.full(len(r["names"]), 1.0 / len(r["names"]))
    t = dp.terminal(path[-1], w)
    cs = dp.cross_section(path[-1], w)
    st = dp.daily_stats(path, w)
    hd = dp.hedge_legs(path, w, r["T"], r["legs"]["atm_vol"], float(o["sig_B"]), o["sig_rel"])
    D, absR, absRb = float(t["D"]), float(t["absR"]), float(t["absRb"])
    leg = {
        "D": D, "absR": absR, "absRb": absRb, "P_D": o["P_D"], "SS": o["SS_mkt"], "Str_B": o["Str_B_mkt"],
        "lam_theta": o["lam_theta"], "lam_rho": o["lambda_rho"], "h_v": o["h_v"], **hd,
    }  # fmt: skip
    row: dict[str, Any] = {
        "date": r["date"], "basket": tag, "tenor": r["tenor"],
        "Rb": float(t["Rb"]), "D": D, "SD": float(t["SD"]), "G": float(t["G"]), "V": float(t["V"]),
        "absR": absR, "absRb": absRb, "D_rel": float(t["D_rel"]), **cs, **st, **hd,
        "basket_event": bool(event or changed), "corporate_action": bool(event), "membership_change": bool(changed),
        "carried_days": carried, **track,
        # members without a single price move in the window (delisted the next day, or no row
        # in the store for the whole window): not a tradable basket
        "stuck_names": int(np.sum(np.all(np.diff(path, axis=0) == 0.0, axis=0))),
        "gap_formula_error": float(abs(dp.gap_formula(path[-1], w) - t["G"])),
        "sandwich_violation": float(max(t["SD"] - t["D"], t["D"] - t["SD"] - 2 * absRb, 0.0)),
    }  # fmt: skip
    row.update(dp.structures(leg))
    for j, m in enumerate(MULT):
        pay = max(D - float(o["strikes"][j]), 0.0)
        row[f"PC_pay_{m}"] = pay
        row[f"PC_{m}_U"] = pay - float(o["calls"][j])
    row["PC_100_S"] = row["PC_100_U"] - float(o["delta_c"][2]) * float(
        t["Rb"]
    )  # static basket hedge
    # variance dispersion (B1: the strike is the log-contract implied dispersion)
    row["VD"] = st["rv_dispersion"] - r["ID2_VS"] if tag == "B1" else np.nan
    row["VD_vega"] = (
        row["VD"] / (2.0 * np.sqrt(r["ID2_VS"])) if tag == "B1" and r["ID2_VS"] > 0 else np.nan
    )
    # the exact decomposition of (PF − PKG_λ): the gap plus (λ − 1) basket straddles
    gap_pnl = float(t["G"]) - o["P_G"]
    bs_pnl = absRb - o["Str_B_mkt"]
    row["dec_gap"] = gap_pnl
    row["dec_theta_bs"] = (o["lam_theta"] - 1.0) * bs_pnl
    row["dec_rho_bs"] = (o["lambda_rho"] - 1.0) * bs_pnl
    row["c11_error"] = float(
        max(
            abs((row["PF_U"] - row["PKG_theta_U"]) - (gap_pnl + row["dec_theta_bs"])),
            abs((row["PF_U"] - row["PKG_rho_U"]) - (gap_pnl + row["dec_rho_bs"])),
            abs(row["GAP_U"] - gap_pnl),
        )
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--tenor", default="3m", choices=list(dd.TENORS))
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()
    n_days = dd.TENORS[args.tenor]
    src = dd.OUT / "entries" / (args.tenor + args.suffix)
    files = sorted(src.glob("*.pkl"))
    panel = dd.prices()
    filled = panel.ffill()
    days_all = dd.trading_days()
    pos = {d: i for i, d in enumerate(days_all)}
    change_days = [c[0] for c in du.CHANGES]
    e_rows, o_rows, legs = [], [], []
    for p in files:
        with p.open("rb") as fh:
            r = pickle.load(fh)
        tags = [t for t in ("B1", "B2", "B3") if t in r]
        for tag in tags:
            e_rows.append(entry_row(r, tag))
        for j, t in enumerate(r["names"]):
            legs.append(
                {
                    "date": r["date"], "ticker": t, "w_B1": r["w_B1"][j], "spot": r["spots"][j],
                    **{k: float(r["legs"][k][j]) for k in ("f", "atm_vol", "straddle", "vega", "M", "var_vs", "vol_90", "vol_110", "skew", "half_spread")},
                    **{f"sig_rel_{tag}": float(r[tag]["sig_rel"][j]) for tag in tags},
                    "rule": r["rules"][j], "bracket_lo": r["legs"]["bracket_lo"][j], "bracket_hi": r["legs"]["bracket_hi"][j],
                }
            )  # fmt: skip
        i0 = pos[r["date"]]
        if i0 + n_days >= len(days_all):
            continue  # no outcome yet (the last day of the store)
        days = days_all[i0 : i0 + n_days + 1]
        path, event, carried = window_path(panel, filled, r["names"], days)
        changed = "B1" in tags and any(days[0] < c <= days[-1] for c in change_days)
        track = {}
        for b in ("DJX", "DIA"):
            a, z = filled.at[days[0], b], filled.at[days[-1], b]
            track[f"ret_{b}"] = float(z / a - 1.0)
        for tag in tags:
            o_rows.append(outcome_row(r, tag, path, event, carried, track, changed))
    entries = pd.DataFrame(e_rows)
    outcomes = pd.DataFrame(o_rows)
    outcomes["track_DJX"] = outcomes["ret_DJX"] - outcomes["Rb"]
    outcomes["track_DIA"] = outcomes["ret_DIA"] - outcomes["Rb"]
    entries.to_parquet(dd.OUT / f"entries_{args.tenor}{args.suffix}.parquet", index=False)
    outcomes.to_parquet(dd.OUT / f"outcomes_{args.tenor}{args.suffix}.parquet", index=False)
    pd.DataFrame(legs).to_parquet(dd.OUT / f"legs_{args.tenor}{args.suffix}.parquet", index=False)
    b1 = outcomes[outcomes["basket"] == outcomes["basket"].iloc[0]]
    print(f"{args.tenor}{args.suffix}: {len(entries) // 2} entry dates, {len(b1)} windows with an outcome; "
          f"basket events {int(b1['basket_event'].sum())}; max sandwich violation {b1['sandwich_violation'].max():.1e}; "
          f"max gap-formula error {b1['gap_formula_error'].max():.1e}; max C11 error {b1['c11_error'].max():.1e}")  # fmt: skip


if __name__ == "__main__":
    main()
