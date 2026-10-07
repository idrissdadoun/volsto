"""Dispersion study: the analysis frame and the tables of spec §8 (used by ``disp_report.py``).

:func:`load` merges, for one tenor and one basket, the entry prices, the outcomes and the
indicators, and adds what needs the three together (trailing realised ``κ_P``, the regime
indicators, in-sample terciles, risk units).  Each ``t*`` function returns one table as a
DataFrame, P&L in % of notional.  No pricing here.
"""

# ruff: noqa: E501, RUF001
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import ndtr

from volsto.studies import disp_data as dd
from volsto.studies import disp_stats as st

LAG = {"1m": 4, "3m": 12, "6m": 25, "12m": 50, "24m": 100}
STEP = {"1m": 5, "3m": 13, "6m": 26, "12m": 51, "24m": 101}
MULT = ("050", "075", "100", "125", "150", "200")
PERIODS = (("2007-09", "2007", "2009"), ("2010-16", "2010", "2016"), ("2017-19", "2017", "2019"),
           ("2020", "2020", "2020"), ("2021-23", "2021", "2023"), ("2024-26", "2024", "2026"))  # fmt: skip
EVENTS = ("2008-10-10", "2011-08-08", "2015-08-24", "2018-02-05", "2018-12-24", "2020-03-16",
          "2022-06-13", "2024-08-05", "2025-04-08", "2026-03-16")  # fmt: skip
#: Structures of the decision table: label → P&L column.
STRUCTS = {
    "PF U": "PF_U", "PF H": "PF_H", "PC(0.5) U": "PC_050_U", "PC(0.75) U": "PC_075_U", "PC(1) U": "PC_100_U",
    "PC(1) static hedge": "PC_100_S", "PC(1.25) U": "PC_125_U", "PC(1.5) U": "PC_150_U", "PC(2) U": "PC_200_U",
    "SS U": "SS_U", "SS H": "SS_H", "BS U": "BS_U", "BS H": "BS_H",
    "PKG_v U": "PKG_v_U", "PKG_v H": "PKG_v_H", "PKG_theta U": "PKG_theta_U", "PKG_theta H": "PKG_theta_H",
    "PKG_rho U": "PKG_rho_U", "PKG_rho H": "PKG_rho_H", "GAP U": "GAP_U", "GAP H": "GAP_H",
    "GAP_rho U": "GAP_rho_U", "GAP_rho H": "GAP_rho_H", "PF_v U": "PF_v_U", "PF_v H": "PF_v_H",
    "REV U": "REV_U", "REV H": "REV_H", "VD (vega notional)": "VD_vega",
}  # fmt: skip
MAIN = (
    "PF U",
    "PF H",
    "PC(1) static hedge",
    "SS H",
    "BS U",
    "PKG_v U",
    "PKG_v H",
    "PKG_theta H",
    "GAP U",
    "GAP_rho U",
    "PF_v U",
    "VD (vega notional)",
)
#: Indicators of spec §6 used in the conditional tables: label → column.
INDICATORS = {
    "CRP": "CRP", "CRP (rho_cop)": "CRP_cop", "single-name vol premium": "vol_premium",
    "basket vol premium": "basket_vol_premium", "correlation skew": "corr_skew", "correlation term 1m-12m": "corr_term",
    "rho_mark": "rho_mark", "GP_G": "GP_G", "forecast basket vol": "sig_B_fc", "|basket return 3m|": "abs_ret_B_3m",
    "|basket return 12m|": "abs_ret_B_12m", "beta dispersion": "beta_disp", "CSAD gamma_1": "csad_g1",
    "VR_cs": "VR_cs", "rotation": "rotation", "variability": "variability", "kappa_Q / kappa_P": "kappa_ratio",
    "dispersion variance premium": "disp_var_premium", "earnings share": "earnings_share",
    "front event premium": "front_premium", "trailing xs kurtosis": "xs_kurtosis",
    "sigma_SPX 1m": "sigma_spx_1m", "|d rho_ATM| 21d": "d_rho_atm", "|d ln sigma_SPX| 21d": "d_sigma_spx",
    "vol of vol SPX": "volvol_spx",
}  # fmt: skip


def pct(x: float, digits: int = 3) -> str:
    return "–" if not np.isfinite(x) else f"{100 * x:.{digits}f}"


def pm(mean: float, se: float, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "–"
    return (
        f"{100 * mean:.{digits}f} ± {100 * se:.{digits}f}"
        if np.isfinite(se)
        else f"{100 * mean:.{digits}f}"
    )


def num(x: float, digits: int = 3) -> str:
    return "–" if not np.isfinite(x) else f"{x:.{digits}f}"


def load(
    tenor: str = "3m", basket: str = "B1", suffix: str = "", root: Path | None = None
) -> pd.DataFrame:
    """One run's entries, outcomes and indicators, one row per entry date of ``basket``
    (``root``: another folder than the study's output, for the copy kept of an earlier run)."""
    root = dd.OUT if root is None else root
    e = pd.read_parquet(root / f"entries_{tenor}{suffix}.parquet")
    o = pd.read_parquet(root / f"outcomes_{tenor}{suffix}.parquet")
    o = o.rename(columns={"csad_g0": "w_csad_g0", "csad_g1": "w_csad_g1", "csad_g2": "w_csad_g2"})
    d = e.merge(o, on=["date", "basket", "tenor"], how="left")
    ind_path = root / f"indicators_{tenor}{suffix}.parquet"
    if ind_path.exists():
        d = d.merge(pd.read_parquet(ind_path), on=["date", "basket", "tenor"], how="left")
    d = d[d["basket"] == basket].sort_values("date").reset_index(drop=True)
    # an entry on which a member has no price move over the whole window (GM on its last day of
    # listing, 2009-06-01; KFT's month without a row in the store at one month) is not a
    # tradable basket: left out, like the dates without a usable smile
    stuck = (
        d["stuck_names"].fillna(0) > 0 if "stuck_names" in d else pd.Series(False, index=d.index)
    )
    dropped = list(d.loc[stuck, "date"])
    d = d[~stuck].reset_index(drop=True)
    d.attrs["stuck_dates"] = dropped
    d["year"] = d["date"].str[:4]
    d["IS"] = d["date"] <= dd.IS_END
    d["sample"] = np.where(d["IS"], "IS", "OOS")
    d["period"] = ""
    for label, lo, hi in PERIODS:
        d.loc[(d["year"] >= lo) & (d["year"] <= hi), "period"] = label
    d["has_outcome"] = d["D"].notna()
    # the strips' squared dispersion is usable when it is positive and within a factor 3 of the
    # copula's own (a member on its last day of listing can carry a vol of several hundred
    # percent, extrapolated flat to the tenor: GM on 2009-06-01 at 6 and 12 months)
    d["strip_ok"] = d["EQV"].gt(0) & d["EQV"].between(d["EV"] / 3.0, d["EV"] * 3.0)
    # realised counterparts
    d["rho_gap_atm"] = d["rho_atm"] - d["rho_real"]
    d["rho_gap_cop"] = d["rho_cop"] - d["rho_real"]
    d["vol_ratio_real"] = d["vol_names_real"] / d["sig_bar"] - 1.0
    d["bs_pnl"] = d["absRb"] - d["Str_B_mkt"]
    for c in ("ret_B_3m", "ret_B_12m"):
        if c in d:
            d["abs_" + c] = d[c].abs()
    # trailing realised kappa_P: windows of the tenor that expired before the entry, over three years
    kp = np.full(len(d), np.nan)
    dates = d["date"].to_numpy(str)
    expiry = d["expiry"].fillna("9999").to_numpy(str)
    D, V = d["D"].to_numpy(float), d["V"].to_numpy(float)
    for i in range(len(d)):
        lo = (pd.Timestamp(str(dates[i])) - pd.Timedelta(days=3 * 365)).strftime("%Y-%m-%d")
        sel = (expiry < dates[i]) & (dates >= lo) & np.isfinite(D)
        if sel.sum() >= 26:
            kp[i] = D[sel].mean() / np.sqrt(V[sel].mean())
    d["kappa_P_trailing"] = kp
    d["kappa_ratio"] = d["kappa_Q"] / d["kappa_P_trailing"]
    d["d_rho_atm"] = (
        d["rho_atm"] - d["rho_atm"].shift(4)
    ).abs()  # four weekly entries: about 21 trading days
    spx = dd.OUT / "spx_vol.parquet"
    if spx.exists():
        s = pd.read_parquet(spx)["sigma_spx_1m"]
        prev = s.shift(1)  # strictly before the entry date
        d["sigma_spx_1m"] = d["date"].map(prev)
        d["d_sigma_spx"] = d["date"].map(np.log(prev / prev.shift(21)).abs())
        d["volvol_spx"] = d["date"].map(
            np.log(s / s.shift(1)).rolling(252).std().shift(1) * np.sqrt(252.0)
        )
    # premiums and sensitivities of each structure, for the risk units of spec §5.4
    d["PC_100_S_premium"] = d["C_100"]
    return d


def premium_of(d: pd.DataFrame, col: str) -> pd.Series | None:
    """The premium a P&L column is divided by ("per unit of premium" is defined for these)."""
    base = col.rsplit("_", 1)[0]
    if base == "PF":
        return d["P_D"]
    if base.startswith("PC_"):
        m = base[3:6]
        return d[f"C_{m}"].where(d[f"C_{m}"] >= 0.0005)
    return {
        "SS": d["SS_mkt"],
        "BS": d["Str_B_mkt"],
        "PKG_v": d["SS_mkt"] - d["Str_B_mkt"],
        "GAP": d["P_G"],
    }.get(base)


def sensitivities(d: pd.DataFrame, col: str) -> tuple[pd.Series, pd.Series]:
    """``(∂P/∂ρ per +0.01, ∂P/∂σ per vol point)`` of the structure behind a P&L column."""
    base = col.rsplit("_", 1)[0]
    zero = pd.Series(0.0, index=d.index)
    pf_r, bs_r = d["dPD_drho"], d["dStrB_drho"]
    pf_v, ss_v, bs_v = d["dPD_dvol"], d["vega_SS"], d["dStrB_dvol"]
    if base.startswith("PC_"):
        m = base[3:6]
        return d[f"dC_drho_{m}"], d[f"dC_dvol_{m}"]
    table = {
        "PF": (pf_r, pf_v), "SS": (zero, ss_v), "BS": (bs_r, bs_v),
        "PKG_v": (-bs_r, ss_v - bs_v), "PKG_theta": (-d["lam_theta"] * bs_r, ss_v - d["lam_theta"] * bs_v),
        "PKG_rho": (-d["lambda_rho"] * bs_r, ss_v - d["lambda_rho"] * bs_v),
        "GAP": (pf_r + bs_r, pf_v - ss_v + bs_v), "GAP_rho": (pf_r + d["lambda_rho"] * bs_r, pf_v - ss_v + d["lambda_rho"] * bs_v),
        "PF_v": (pf_r, pf_v - d["h_v"] * ss_v), "REV": (bs_r, bs_v - ss_v),
    }  # fmt: skip
    return table.get(base, (zero * np.nan, zero * np.nan))


def risk_units(d: pd.DataFrame, col: str) -> dict[str, float]:
    """Mean P&L per unit of premium, in correlation points and in vol points (spec §5.4)."""
    y = d[col]
    out: dict[str, float] = {}
    prem = premium_of(d, col)
    if prem is not None:
        ok = prem.notna() & y.notna() & (prem > 0)
        out["per premium"] = float((y[ok] / prem[ok]).mean()) if ok.any() else np.nan
        out["sum/sum premium"] = float(y[ok].sum() / prem[ok].sum()) if ok.any() else np.nan
    for tag, sens in zip(("corr pts", "vol pts"), sensitivities(d, col), strict=True):
        a = sens.abs()
        med = float(a.median()) if a.notna().any() else np.nan
        ok = y.notna() & a.notna() & (a >= 0.25 * med) & (med > 0)
        denom = a if tag == "corr pts" else sens
        out[tag] = float((y[ok] / denom[ok]).mean()) if ok.any() else np.nan
        out[f"sum/sum {tag}"] = float(y[ok].sum() / a[ok].sum()) if ok.any() else np.nan
        out[f"n {tag}"] = float(ok.sum())
    return out


def nonoverlapping(d: pd.DataFrame, tenor: str) -> pd.DataFrame:
    return d.iloc[:: STEP[tenor]]


def is_cuts(d: pd.DataFrame, col: str) -> tuple[float, float]:
    return st.tercile_cuts(d.loc[d["IS"] & d["has_outcome"], col])


# ------------------------------------------------------------------------------------------------
# tables
# ------------------------------------------------------------------------------------------------


def t1(d: pd.DataFrame) -> pd.DataFrame:
    o = d[d["has_outcome"]]
    return pd.DataFrame(
        [
            {
                "windows": len(o),
                "sandwich violations above 1e-12": int((o["sandwich_violation"] > 1e-12).sum()),
                "largest violation": f"{o['sandwich_violation'].max():.1e}",
                "largest |gap formula − (D − SD)|": f"{o['gap_formula_error'].max():.1e}",
                "largest decomposition error (C11)": f"{o['c11_error'].max():.1e}",
                "D ≤ √V violations": int((o["D"] > np.sqrt(o["V"]) + 1e-12).sum()),
            }
        ]
    )


def t2(d: pd.DataFrame) -> pd.DataFrame:
    o = d[d["has_outcome"]].copy()
    edges = [0, 0.02, 0.05, 0.10, 0.20, np.inf]
    labels = ["0–2 %", "2–5 %", "5–10 %", "10–20 %", "> 20 %"]
    o["bucket"] = pd.cut(o["absRb"], edges, labels=labels, right=False)
    rows = []
    for b in labels:
        g = o[o["bucket"] == b]
        rows.append({"|R̄|": b, "windows": len(g), "mean D": pct(g["D"].mean()), "mean SD": pct(g["SD"].mean()),
                     "mean G": pct(g["G"].mean()), "G / D": num(g["G"].sum() / g["D"].sum(), 2) if len(g) else "–"})  # fmt: skip
    g = o
    rows.append({"|R̄|": "all", "windows": len(g), "mean D": pct(g["D"].mean()), "mean SD": pct(g["SD"].mean()),
                 "mean G": pct(g["G"].mean()), "G / D": num(g["G"].sum() / g["D"].sum(), 2)})  # fmt: skip
    return pd.DataFrame(rows)


def t3_premium(d: pd.DataFrame, tenor: str) -> pd.DataFrame:
    o = d[d["has_outcome"]]
    lag = LAG[tenor]
    rows = []

    def row(label: str, g: pd.DataFrame) -> dict[str, Any]:
        a, sa, n = st.mean_se(g["rho_gap_atm"], lag)
        c, sc, _ = st.mean_se(g["rho_gap_cop"], lag)
        return {"group": label, "windows": n, "ρ_ATM": num(g["rho_atm"].mean()), "ρ_cop": num(g["rho_cop"].mean()),
                "ρ_real": num(g["rho_real"].mean()), "ρ_ATM − ρ_real": f"{a:.3f} ± {sa:.3f}", "ρ_cop − ρ_real": f"{c:.3f} ± {sc:.3f}"}  # fmt: skip

    rows.append(row("all", o))
    rows.append(row("IS", o[o["IS"]]))
    rows.append(row("OOS", o[~o["IS"]]))
    if "sigma_spx_1m" in o:
        lab = st.tercile(o["sigma_spx_1m"], is_cuts(d, "sigma_spx_1m"))
        for k in ("low", "mid", "high"):
            rows.append(row(f"SPX vol {k}", o[lab == k]))
    for y, g in o.groupby("year"):
        rows.append(row(str(y), g))
    return pd.DataFrame(rows)


def t3_pnl(d: pd.DataFrame, tenor: str) -> pd.DataFrame:
    o = d[d["has_outcome"]]
    rows = []
    for label, col in STRUCTS.items():
        if col not in o:
            continue
        s = st.describe(o[col], LAG[tenor])
        rows.append({"structure": label, "mean": pct(s["mean"]), "t": num(s["t"], 2), "sd": pct(s["sd"]), "skew": num(s["skew"], 2),
                     "5 % quantile": pct(s["q05"]), "worst": pct(s["worst"]), "hit rate": num(s["hit"], 2), "n": int(s["n"])})  # fmt: skip
    return pd.DataFrame(rows)


def richness(g: pd.DataFrame) -> dict[str, float]:
    """The pooled split of eq. (10.1) on the windows of ``g`` (exact identities)."""
    mP, mD = g["P_D"].mean(), g["D"].mean()
    mQ, mV, mC = g["EQV"].mean(), g["V"].mean(), g["EV"].mean()
    kq, kp, kc = mP / np.sqrt(mQ), mD / np.sqrt(mV), mP / np.sqrt(mC)
    return {
        "richness": mP / mD, "variance factor": np.sqrt(mQ / mV), "convexity factor": kq / kp,
        "kappa_Q": kq, "kappa_P": kp, "kappa_cop": kc, "kappa_cop/kappa_P": kc / kp, "sqrt(EcopV/EQV)": np.sqrt(mC / mQ),
    }  # fmt: skip


def t4(d: pd.DataFrame) -> pd.DataFrame:
    o = d[d["has_outcome"] & d["strip_ok"]]
    rows = []
    groups: list[tuple[str, pd.DataFrame]] = [("all", o), ("IS", o[o["IS"]]), ("OOS", o[~o["IS"]])]
    groups += [(label, o[o["period"] == label]) for label, _, _ in PERIODS]
    groups += [(str(y), g) for y, g in o.groupby("year")]
    for label, g in groups:
        if len(g) < 5:
            continue
        r = richness(g)
        rows.append({"group": label, "windows": len(g), "mean P_D": pct(g["P_D"].mean()), "mean D": pct(g["D"].mean()),
                     "P_D / D": num(r["richness"]), "= variance √(E^Q V / V)": num(r["variance factor"]),
                     "× convexity κ_Q/κ_P": num(r["convexity factor"]), "κ_Q": num(r["kappa_Q"]), "κ_P": num(r["kappa_P"]),
                     "κ_cop/κ_P": num(r["kappa_cop/kappa_P"]), "√(E^cop V / E^Q V)": num(r["sqrt(EcopV/EQV)"])})  # fmt: skip
    return pd.DataFrame(rows)


def t7(d: pd.DataFrame, tenor: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    o = d[d["has_outcome"]].copy()
    rows = []
    out: dict[str, Any] = {}
    for label, g in (("all", o), ("IS", o[o["IS"]]), ("OOS", o[~o["IS"]])):
        bd = g["beta_disp"] if "beta_disp" in g else pd.Series(np.nan, index=g.index)
        X = np.column_stack([np.ones(len(g)), g["absRb"], bd, g["absRb"] * bd])
        fit = st.ols(g["GAP_U"], X, LAG[tenor])
        rows.append({"sample": label, "n": fit["n"], "const": pm(fit["coef"][0], fit["se"][0]),
                     "|R̄|": f"{fit['coef'][1]:.3f} ± {fit['se'][1]:.3f}", "βd": f"{fit['coef'][2]:.3f} ± {fit['se'][2]:.3f}",
                     "|R̄| × βd": f"{fit['coef'][3]:.2f} ± {fit['se'][3]:.2f}", "t (interaction)": num(fit["t"][3], 2), "R²": num(fit["r2"], 2)})  # fmt: skip
        out[label] = fit
    g1 = o["w_csad_g1"]
    rows.append({"sample": "realised CSAD γ_1 in the windows", "n": int(g1.notna().sum()), "const": f"mean {g1.mean():.3f}",
                 "|R̄|": f"median {g1.median():.3f}", "βd": f"share > 0: {(g1 > 0).mean():.2f}", "|R̄| × βd": "", "t (interaction)": "", "R²": ""})  # fmt: skip
    return pd.DataFrame(rows), out


def t8(d: pd.DataFrame, tenor: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    o = d[d["has_outcome"]]
    cut = o["Rb"].quantile(0.10)
    rows = []
    for label, g in (
        ("worst decile of basket returns", o[o["Rb"] <= cut]),
        (
            "the same, without basket_event windows",
            o[(o["Rb"] <= cut) & ~o["basket_event"].astype(bool)],
        ),
        ("all windows", o),
    ):
        row: dict[str, Any] = {
            "windows": label,
            "n": len(g),
            "mean R̄": pct(g["Rb"].mean(), 1),
            "ρ_real − ρ_mark": num((g["rho_real"] - g["rho_cop"]).mean(), 2),
        }
        for s in ("PF_H", "PKG_v_H", "PKG_theta_H", "BS_U", "PF_U", "PKG_v_U", "PKG_theta_U"):
            m, se, _ = st.mean_se(g[s], LAG[tenor])
            row[s.replace("_", " ")] = pm(m, se, 2)
        rows.append(row)
    ev = []
    for day in EVENTS:
        g = o[(o["date"] <= day) & (o["expiry"] >= day)]
        if day == "2026-03-16":  # any day of March 2026
            g = o[(o["date"] <= "2026-03-31") & (o["expiry"] >= "2026-03-01")]
        row = {"event": day if day != "2026-03-16" else "March 2026", "windows": len(g), "with basket_event": int(g["basket_event"].sum()),
               "mean R̄": pct(g["Rb"].mean(), 1) if len(g) else "–"}  # fmt: skip
        for s in (
            "PF_U",
            "PF_H",
            "PC_100_S",
            "SS_U",
            "BS_U",
            "PKG_v_U",
            "PKG_v_H",
            "PKG_theta_U",
            "PKG_theta_H",
            "GAP_U",
            "PF_v_U",
            "REV_H",
        ):
            row[s.replace("_", " ")] = pct(g[s].mean(), 2) if len(g) else "–"
        ev.append(row)
    return pd.DataFrame(rows), pd.DataFrame(ev)


def t12(d: pd.DataFrame) -> pd.DataFrame:
    o = d[d["has_outcome"]]
    rows = []
    X = np.column_stack([np.ones(len(o)), o["Rb"], o["Rb"] ** 2])
    for label, col in STRUCTS.items():
        if col not in o or o[col].notna().sum() < 30:
            continue
        y = o[col].to_numpy(float)
        ok = np.isfinite(y)
        coef = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
        res = y[ok] - X[ok] @ coef
        rows.append(
            {
                "structure": label,
                "R² on R̄ and R̄²": num(1 - res.var() / y[ok].var(), 2),
                "slope on R̄": num(coef[1], 3),
                "slope on R̄²": num(coef[2], 2),
            }
        )
    return pd.DataFrame(rows)


def t13(d: pd.DataFrame, tenor: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(overall, by period, long)``: the decision table.  ``long`` has every structure by
    sample, period, basket_event filter and tercile of every indicator."""
    o = d[d["has_outcome"]]
    lag = LAG[tenor]
    overall, long = [], []
    for label, col in STRUCTS.items():
        if col not in o or o[col].notna().sum() < 10:
            continue
        s = st.describe(o[col], lag)
        ru = risk_units(o, col)
        overall.append({
            "structure": label, "mean": pct(s["mean"]), "t": num(s["t"], 2), "sd": pct(s["sd"]), "mean/sd": num(s["mean/sd"], 2),
            "hit": num(s["hit"], 2), "5 %": pct(s["q05"]), "worst": pct(s["worst"]),
            "per premium": num(ru.get("per premium", np.nan), 3), "Σ/Σ premium": num(ru.get("sum/sum premium", np.nan), 3),
            "corr pts": num(ru.get("corr pts", np.nan), 2), "Σ/Σ corr": num(ru.get("sum/sum corr pts", np.nan), 2),
            "vol pts": num(ru.get("vol pts", np.nan), 2), "Σ/Σ vol": num(ru.get("sum/sum vol pts", np.nan), 2), "n": int(s["n"]),
        })  # fmt: skip
        splits: list[tuple[str, str, pd.DataFrame]] = [("sample", "all", o), ("sample", "IS", o[o["IS"]]), ("sample", "OOS", o[~o["IS"]]),
                                                       ("no basket_event", "all", o[~o["basket_event"].astype(bool)])]  # fmt: skip
        splits += [("period", lab, o[o["period"] == lab]) for lab, _, _ in PERIODS]
        for split, group, g in splits:
            s2 = st.describe(g[col], lag)
            long.append({"structure": label, "split": split, "group": group, **s2})
        for ilabel, icol in INDICATORS.items():
            if icol not in o or o[icol].notna().sum() < 60:
                continue
            for scope, cuts, frames in (("IS terciles", is_cuts(d, icol), (("IS", o[o["IS"]]), ("OOS", o[~o["IS"]]))),
                                        ("full terciles", st.tercile_cuts(o[icol]), (("full", o),))):  # fmt: skip
                for sample, g in frames:
                    lab = st.tercile(g[icol], cuts)
                    for k in ("low", "mid", "high"):
                        s2 = st.describe_subset(g[col], (lab == k).to_numpy(), lag)
                        long.append(
                            {
                                "structure": label,
                                "split": f"{ilabel} ({scope})",
                                "group": f"{sample} {k}",
                                **s2,
                            }
                        )
    by_period = []
    for label in MAIN:
        col = STRUCTS[label]
        row: dict[str, Any] = {"structure": label}
        for lab, _, _ in PERIODS:
            m, se, _ = st.mean_se(o.loc[o["period"] == lab, col], lag)
            row[lab] = pm(m, se, 2)
        m, se, _ = st.mean_se(o.loc[~o["basket_event"].astype(bool), col], lag)
        row["no basket_event"] = pm(m, se, 2)
        m, se, _ = st.mean_se(nonoverlapping(o, tenor)[col], 0)
        row["non-overlapping"] = pm(m, se, 2)
        by_period.append(row)
    return pd.DataFrame(overall), pd.DataFrame(by_period), pd.DataFrame(long)


def tercile_contrast(
    d: pd.DataFrame,
    icol: str,
    y: pd.Series,
    tenor: str,
    sample: str,
    cuts: tuple[float, float] | None = None,
) -> dict[str, float]:
    """Mean of ``y`` in the top minus the bottom tercile of ``icol`` (in-sample cut points),
    with its Hansen–Hodrick standard error from the regression on the three tercile dummies."""
    o = d[d["has_outcome"]]
    cuts = cuts or is_cuts(d, icol)
    g = o[o["IS"]] if sample == "IS" else (o[~o["IS"]] if sample == "OOS" else o)
    lab = st.tercile(g[icol], cuts)
    yy = y.reindex(g.index)
    keep = lab.notna() & yy.notna()
    X = np.column_stack([(lab[keep] == k).to_numpy(float) for k in ("low", "mid", "high")])
    out = {
        "low": np.nan,
        "mid": np.nan,
        "high": np.nan,
        "diff": np.nan,
        "se": np.nan,
        "t": np.nan,
        "n": float(keep.sum()),
    }
    if keep.sum() < 30 or X.sum(axis=0).min() < 5:
        return out
    yv = yy[keep].to_numpy(float)
    lag = LAG[tenor]
    xtx_inv = np.linalg.pinv(X.T @ X)
    coef = xtx_inv @ X.T @ yv
    u = X * (yv - X @ coef)[:, None]
    cov = xtx_inv @ st._long_run(u, lag, False) @ xtx_inv
    var = cov[2, 2] + cov[0, 0] - 2 * cov[0, 2]
    if var <= 0:
        cov = xtx_inv @ st._long_run(u, 2 * lag, True) @ xtx_inv
        var = cov[2, 2] + cov[0, 0] - 2 * cov[0, 2]
    se = float(np.sqrt(max(var, 0.0)))
    out.update(low=float(coef[0]), mid=float(coef[1]), high=float(coef[2]), diff=float(coef[2] - coef[0]), se=se,
               t=float((coef[2] - coef[0]) / se) if se > 0 else np.nan)  # fmt: skip
    return out


def verdict(sign: int, is_: dict[str, float], oos: dict[str, float]) -> str:
    """Spec §8: confirmed / consistent / contradicted / inconclusive / not testable."""
    a, t, b = is_["diff"], is_["t"], oos["diff"]
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(t)):
        return "not testable"
    right_is, right_oos = np.sign(a) == sign, np.sign(b) == sign
    if right_is and abs(t) >= 2 and right_oos:
        return "confirmed"
    if right_is and right_oos:
        return "consistent"
    if (not right_is and abs(t) >= 2) or (not right_is and not right_oos):
        return "contradicted"
    return "inconclusive"


def primary_tests(d: pd.DataFrame, tenor: str) -> pd.DataFrame:
    o = d
    prem_pf = o["P_D"]
    prem_c = o["C_100"].where(o["C_100"] >= 0.0005)
    tests = [
        ("Q1", "CRP → PKG_theta H", "CRP", o["PKG_theta_H"], +1),
        ("Q2", "GP_G → GAP", "GP_G", o["GAP_U"], -1),
        ("Q3", "VR_cs → PF U − PF H", "VR_cs", o["PF_U"] - o["PF_H"], +1),
        (
            "Q4",
            "variability → PC(1) static hedge − PF, per premium",
            "variability",
            o["PC_100_S"] / prem_c - o["PF_U"] / prem_pf,
            +1,
        ),
        (
            "Q5",
            "single-name vol premium → PKG_v H − PKG_theta H",
            "vol_premium",
            o["PKG_v_H"] - o["PKG_theta_H"],
            -1,
        ),
    ]
    rows = []
    for q, label, icol, y, sign in tests:
        if icol not in d:
            rows.append({"test": q, "statistic": label, "verdict": "not testable"})
            continue
        a = tercile_contrast(d, icol, y, tenor, "IS")
        b = tercile_contrast(d, icol, y, tenor, "OOS")
        scale = 1.0 if q == "Q4" else 100.0
        rows.append({
            "test": q, "statistic": label, "expected sign": "+" if sign > 0 else "−",
            "IS top − bottom": f"{scale * a['diff']:.3f} ± {scale * a['se']:.3f}", "IS t": num(a["t"], 2), "IS n": int(a["n"]),
            "OOS top − bottom": f"{scale * b['diff']:.3f} ± {scale * b['se']:.3f}", "OOS t": num(b["t"], 2), "OOS n": int(b["n"]),
            "OOS minimum detectable effect": num(2.8 * scale * b["se"], 3), "verdict": verdict(sign, a, b),
            "unit": "ratio of premium" if q == "Q4" else "% of notional",
        })  # fmt: skip
    return pd.DataFrame(rows)


def t16(d: pd.DataFrame, tenor: str) -> pd.DataFrame:
    o = d[d["has_outcome"]]
    rows = []
    sumsq = o["V"] + o["Rb"] ** 2
    for label, cols in (
        ("straddles: D on [1, Σw|R_i|, |R̄|]", [o["absR"], o["absRb"]]),
        ("squares: D on [1, Σw R_i², R̄²]", [sumsq, o["Rb"] ** 2]),
    ):
        for sample, sel in (("all", o["D"].notna()), ("IS", o["IS"]), ("OOS", ~o["IS"])):
            X = np.column_stack([np.ones(int(sel.sum())), *[c[sel] for c in cols]])
            fit = st.ols(o.loc[sel, "D"], X, LAG[tenor])
            rows.append({"regression": label, "sample": f"realised, {sample}", "const": pct(fit["coef"][0]), "a": f"{fit['coef'][1]:.3f} ± {fit['se'][1]:.3f}",
                         "b": f"{fit['coef'][2]:.3f} ± {fit['se'][2]:.3f}", "−b / a": num(-fit["coef"][2] / fit["coef"][1], 2), "R²": num(fit["r2"], 2), "n": fit["n"]})  # fmt: skip
    # the model's own replica: the copula scenarios of 12 dates spread over the sample
    pick = d.iloc[np.linspace(0, len(d) - 1, 12).round().astype(int)]
    for label, pre in (
        ("straddles: D on [1, Σw|R_i|, |R̄|]", "rep_abs"),
        ("squares: D on [1, Σw R_i², R̄²]", "rep_sq"),
    ):
        a, b = pick[f"{pre}_a"].mean(), pick[f"{pre}_b"].mean()
        rows.append({"regression": label, "sample": "copula scenarios, 12 dates (mean)", "const": pct(pick[f"{pre}_c"].mean()), "a": num(a), "b": num(b),
                     "−b / a": num(-b / a, 2), "R²": num(pick[f"{pre}_r2"].mean(), 2), "n": 12})  # fmt: skip
    return pd.DataFrame(rows)


def t17(d: pd.DataFrame, tenor: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    o = d[d["has_outcome"]]
    lag = LAG[tenor]
    head = []
    for col, label in (
        ("GAP_U", "GAP = G − P_G"),
        ("GAP_rho_U", "GAP_ρ (correlation-neutral)"),
        ("GAP_H", "GAP hedged"),
    ):
        for sample, g in (
            ("all", o),
            ("IS", o[o["IS"]]),
            ("OOS", o[~o["IS"]]),
            ("non-overlapping", nonoverlapping(o, tenor)),
        ):
            s = st.describe(g[col], lag if sample != "non-overlapping" else 0)
            head.append({"structure": label, "sample": sample, "mean": pct(s["mean"]), "t": num(s["t"], 2), "sd": pct(s["sd"]), "hit": num(s["hit"], 2),
                         "5 %": pct(s["q05"]), "worst": pct(s["worst"]), "mean P_G": pct(g["P_G"].mean()), "mean G": pct(g["G"].mean()), "n": int(s["n"])})  # fmt: skip
    cond = []
    for ilabel in (
        "GP_G",
        "basket vol premium",
        "forecast basket vol",
        "|basket return 3m|",
        "beta dispersion",
        "VR_cs",
        "rho_mark",
        "CRP",
    ):
        icol = INDICATORS[ilabel]
        if icol not in o:
            continue
        for col in ("GAP_U", "GAP_rho_U"):
            row: dict[str, Any] = {
                "indicator": ilabel,
                "structure": col.replace("_U", "").replace("_", " "),
            }
            for sample in ("IS", "OOS"):
                c = tercile_contrast(d, icol, o[col], tenor, sample)
                row[f"{sample} low"], row[f"{sample} mid"], row[f"{sample} high"] = (
                    pct(c["low"]),
                    pct(c["mid"]),
                    pct(c["high"]),
                )
                row[f"{sample} high − low"] = pm(c["diff"], c["se"])
                row[f"{sample} t"] = num(c["t"], 2)
            cond.append(row)
    post = []
    edges = [0, 0.02, 0.05, 0.10, 0.20, np.inf]
    labels = ["0–2 %", "2–5 %", "5–10 %", "10–20 %", "> 20 %"]
    b = pd.cut(o["absRb"], edges, labels=labels, right=False)
    for k in labels:
        g = o[b == k]
        post.append(
            {
                "ex post": f"|R̄| {k}",
                "n": len(g),
                "GAP": pct(g["GAP_U"].mean()),
                "GAP_ρ": pct(g["GAP_rho_U"].mean()),
                "PF U": pct(g["PF_U"].mean()),
                "PKG_v U": pct(g["PKG_v_U"].mean()),
            }
        )
    lab = st.tercile(o["rho_real"] - o["rho_cop"], st.tercile_cuts(o["rho_real"] - o["rho_cop"]))
    for k in ("low", "mid", "high"):
        g = o[lab == k]
        post.append(
            {
                "ex post": f"ρ_real − ρ_mark {k}",
                "n": len(g),
                "GAP": pct(g["GAP_U"].mean()),
                "GAP_ρ": pct(g["GAP_rho_U"].mean()),
                "PF U": pct(g["PF_U"].mean()),
                "PKG_v U": pct(g["PKG_v_U"].mean()),
            }
        )
    for sample, g in (("all", o), ("IS", o[o["IS"]]), ("OOS", o[~o["IS"]])):
        a, sa, _ = st.mean_se(g["PF_U"] - g["PKG_theta_U"], lag)
        g1, s1, _ = st.mean_se(g["dec_gap"], lag)
        g2, s2, _ = st.mean_se(g["dec_theta_bs"], lag)
        post.append(
            {
                "ex post": f"PF − PKG_θ, {sample}: {pm(a, sa)} = (G − P_G) {pm(g1, s1)} + (λ_θ − 1)(|R̄| − Str_B) {pm(g2, s2)}",
                "n": len(g),
                "GAP": "",
                "GAP_ρ": "",
                "PF U": "",
                "PKG_v U": "",
            }
        )
    return pd.DataFrame(head), pd.DataFrame(cond), pd.DataFrame(post)


def t18(
    d: pd.DataFrame, tenor: str, variants: tuple[str, ...] = ("G0", "G1")
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibration of the forecast edges: realised P&L = a + b·edge, pooled, IS and OOS; and
    the deciles of realised P&L against the edge (for the figure)."""
    o = d[d["has_outcome"]]
    names = {"PF": "PF_U", "SS": "SS_U", "BS": "BS_U", "PKG_v": "PKG_v_U", "PKG_theta": "PKG_theta_U", "PKG_rho": "PKG_rho_U",
             "GAP": "GAP_U", "GAP_rho": "GAP_rho_U", "PF_v": "PF_v_U"}  # fmt: skip
    rows, dec = [], []
    for v in variants:
        for s, col in names.items():
            ecol = f"edge_{s}_{v}"
            if ecol not in o:
                continue
            for sample, g in (("IS", o[o["IS"]]), ("OOS", o[~o["IS"]])):
                X = np.column_stack([np.ones(len(g)), g[ecol]])
                fit = st.ols(g[col], X, LAG[tenor])
                rows.append({"forecast": v, "structure": s, "sample": sample, "a": pm(fit["coef"][0], fit["se"][0]), "b": f"{fit['coef'][1]:.2f} ± {fit['se'][1]:.2f}",
                             "t(b)": num(fit["t"][1], 2), "t(b − 1)": num((fit["coef"][1] - 1) / fit["se"][1], 2) if fit["se"][1] > 0 else "–",
                             "R²": num(fit["r2"], 2), "mean edge": pct(g[ecol].mean()), "mean P&L": pct(g[col].mean()), "n": fit["n"]})  # fmt: skip
            g = o[[ecol, col]].dropna()
            if len(g) >= 50:
                q = pd.qcut(g[ecol], 10, labels=False, duplicates="drop")
                for k, gg in g.groupby(q):
                    dec.append(
                        {
                            "forecast": v,
                            "structure": s,
                            "decile": int(k) + 1,
                            "edge": gg[ecol].mean(),
                            "pnl": gg[col].mean(),
                        }
                    )
    return pd.DataFrame(rows), pd.DataFrame(dec)


def t19(d: pd.DataFrame, tenor: str) -> pd.DataFrame:
    o = d[d["has_outcome"]].copy()
    o["f_rho"] = o["rho_real"] - o["rho_cop"]
    o["f_vol"] = o["vol_ratio_real"]
    o["f_bs"] = o["bs_pnl"]
    o["f_vr"] = o["vr_window"] - 1.0
    factors = ["f_rho", "f_vol", "f_bs", "f_vr"]
    rows = []
    for label, col in STRUCTS.items():
        if col not in o or o[col].notna().sum() < 60:
            continue
        g = o[[col, *factors]].dropna()
        X = np.column_stack([np.ones(len(g)), *[g[f] for f in factors]])
        fit = st.ols(g[col], X, LAG[tenor])
        row: dict[str, Any] = {"structure": label, "n": fit["n"], "R²": num(fit["r2"], 2)}
        for j, (_f, name) in enumerate(
            zip(
                factors,
                ("ρ_real − ρ_mark", "σ̄_real/σ̄_imp − 1", "|R̄| − Str_B", "VR_window − 1"),
                strict=True,
            )
        ):
            exact = (
                fit["r2"] > 1.0 - 1e-9
            )  # the structure is the factor itself (the basket straddle)
            row[name] = (
                f"{100 * fit['coef'][j + 1]:.2f} ({'identity' if exact else format(fit['t'][j + 1], '.1f')})"
            )
            others = [k for k in range(len(factors)) if k != j]
            Xr = np.column_stack([np.ones(len(g)), *[g[factors[k]] for k in others]])
            res = (
                g[col].to_numpy(float)
                - Xr @ np.linalg.lstsq(Xr, g[col].to_numpy(float), rcond=None)[0]
            )
            r2r = 1 - res.var() / g[col].var(ddof=0)
            row[f"partial R² {name}"] = num(fit["r2"] - r2r, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def checks(d: pd.DataFrame) -> dict[str, Any]:
    """The checks of spec §10 that the frame can answer (C3–C7, C11, C13)."""
    o = d[d["has_outcome"]]
    priced = np.column_stack([d[f"C_{m}"] >= 0.0005 for m in MULT])
    rel = np.column_stack([(d[f"delta_c_{m}"] / d[f"delta_c_fd_{m}"] - 1.0).abs() for m in MULT])
    c13 = float(np.nanmax(np.where(priced, rel, np.nan)))
    big = o[o["track_DJX"].abs() > 0.005]
    return {
        "C3": {
            "max relative error of a single-name straddle": float(d["c3_max_rel"].max()),
            "median over dates of the max": float(d["c3_max_rel"].median()),
            "share of dates with every straddle within 3 standard errors": float(
                (d["c3_max_z"] <= 3).mean()
            ),
            "pass": bool((d["c3_max_z"] <= 3).all()),
        },
        "C4": {
            "max |copula basket straddle − listed|": float(d["c4_error"].max()),
            "dates clipped": int((d["rho_cop_flag"] != "").sum()),
            "pass": bool(d["c4_error"].max() <= 1e-5),
        },
        "C5": {
            "windows": len(o),
            "violations": int((o["sandwich_violation"] > 1e-12).sum()),
            "pass": bool((o["sandwich_violation"] <= 1e-12).all()),
        },
        "C6": {
            "max standard error / price of the forward": float((d["P_D_se"] / d["P_D"]).max()),
            "pass": bool((d["P_D_se"] / d["P_D"]).max() < 0.005),
        },
        "C7": {
            "tracking DJX − frozen basket: mean": float(o["track_DJX"].mean()),
            "sd": float(o["track_DJX"].std()),
            "5 % / 95 %": [
                float(o["track_DJX"].quantile(0.05)),
                float(o["track_DJX"].quantile(0.95)),
            ],
            "windows above 0.5 % in absolute value": len(big),
            "of which with a basket event": int(big["basket_event"].sum()),
            "tracking DIA − frozen basket: mean": float(o["track_DIA"].mean()),
            "sd DIA": float(o["track_DIA"].std()),
        },
        "C11": {
            "max error": float(o["c11_error"].max()),
            "pass": bool(o["c11_error"].max() <= 1e-12),
        },
        "C13": {
            "max relative difference Euler against central difference, strikes priced": c13,
            "pass": bool(c13 <= 0.01),
        },
    }


def gaussian_call(F: float, sd: float, K: float) -> float:
    dd_ = (F - K) / sd
    return float((F - K) * ndtr(dd_) + sd * np.exp(-0.5 * dd_ * dd_) / np.sqrt(2 * np.pi))
