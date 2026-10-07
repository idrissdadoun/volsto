"""Dispersion study: the tables of the second phase (spec §8: T5, T6, T9, T10, T11, T14, T15,
the FHS part of T18 with rule R2, the out-of-sample replica of T16) on the frame of
``disp_tables.load``.  The rules R1 and R2 are pre-registered in the spec; nothing here is
fitted on out-of-sample outcomes.
"""

# ruff: noqa: E501, RUF001
from __future__ import annotations

from typing import Any

import disp_tables as tb
import numpy as np
import pandas as pd

from volsto.studies import disp_copula as dc
from volsto.studies import disp_data as dd
from volsto.studies import disp_stats as st

ALWAYS = {"PF U": "PF_U", "PF H": "PF_H", "PC(1) static hedge": "PC_100_S", "PKG_v U": "PKG_v_U", "PKG_v H": "PKG_v_H",
          "PKG_theta H": "PKG_theta_H", "BS U": "BS_U", "SS U": "SS_U", "VD": "VD_vega"}  # fmt: skip
R2_CANDIDATES = {
    "PF": "PF_U",
    "PC_100_S": "PC_100_S",
    "PKG_v": "PKG_v_U",
    "PKG_theta": "PKG_theta_U",
    "PKG_rho": "PKG_rho_U",
    "PF_v": "PF_v_U",
}


def add_fhs(d: pd.DataFrame, tenor: str, basket: str = "B1") -> pd.DataFrame:
    path = dd.OUT / f"fhs_{tenor}.parquet"
    if not path.exists():
        return d
    f = pd.read_parquet(path)
    f = f[f["basket"] == basket].drop(columns=["basket", "tenor"])
    return d.merge(f, on="date", how="left")


#: Strikes of the call leaf of R1 (spec v2.3, T14), furthest first; the pricing floor is the
#: study's (``disp_copula.MIN_CALL``, 0.05 % of notional: T5's "dates priced", Q4).
CALL_LEAF_STRIKES = ("150", "125", "100")


def prepare(d: pd.DataFrame, tenor: str, basket: str = "B1") -> pd.DataFrame:
    """The frame of the second phase: the FHS columns and the statically hedged calls at 1.25
    and 1.5 times the forward's price (``PC_125_S``, ``PC_150_S``), built as ``PC_100_S`` is in
    ``disp_outcomes`` (call P&L minus the common-move delta times the basket's return)."""
    x = add_fhs(d, tenor, basket)
    if x is d:
        x = d.copy()
    for m in ("125", "150"):
        if f"PC_{m}_S" not in x and {f"PC_{m}_U", f"delta_c_{m}", "Rb"} <= set(x.columns):
            x[f"PC_{m}_S"] = x[f"PC_{m}_U"] - x[f"delta_c_{m}"] * x["Rb"]
    return x


def call_leaf(d: pd.DataFrame) -> str:
    """The call leaf of R1 at this tenor (spec v2.3): the statically hedged call at the furthest
    strike among 1.5, 1.25 and 1.0 times the forward's price whose price is at or above the
    study's floor on every entry date with an outcome, so that the leaf is one structure at a
    tenor (``PC(1)`` at 1m, 3m and 6m; ``PC(1.25)`` at 12m and 24m)."""
    o = d[d["has_outcome"]]
    for m in CALL_LEAF_STRIKES:
        if f"PC_{m}_S" in d and len(o) and bool((o[f"C_{m}"] >= dc.MIN_CALL).all()):
            return f"PC_{m}_S"
    return "PC_100_S"


def priced_shares(d: pd.DataFrame) -> dict[str, float]:
    """Share of the entry dates with an outcome on which the call at each strike of
    :data:`CALL_LEAF_STRIKES` is at or above the pricing floor."""
    o = d[d["has_outcome"]]
    return {
        m: float((o[f"C_{m}"] >= dc.MIN_CALL).mean()) if len(o) else float("nan")
        for m in CALL_LEAF_STRIKES
    }


def priced_text(d: pd.DataFrame) -> str:
    sh = priced_shares(d)
    return ", ".join(f"{int(m) / 100:g}× on {100 * v:.1f} %" for m, v in sh.items())


def expanding_cuts(values: pd.Series, min_obs: int = 104) -> pd.DataFrame:
    """Tercile cut points of each entry from the values of the entries strictly before it (at
    least ``min_obs`` of them: two years of weekly entries); NaN before that."""
    v = values.to_numpy(float)
    lo = np.full(v.size, np.nan)
    hi = np.full(v.size, np.nan)
    for i in range(v.size):
        past = v[:i]
        past = past[np.isfinite(past)]
        if past.size >= min_obs:
            lo[i], hi[i] = np.quantile(past, [1.0 / 3.0, 2.0 / 3.0])
    return pd.DataFrame({"lo": lo, "hi": hi}, index=values.index)


def tercile_labels(d: pd.DataFrame, col: str, cuts: str) -> pd.Series:
    """``low`` / ``mid`` / ``high`` of an indicator on the in-sample cut points (``cuts =
    "is"``) or on the expanding-window ones (``"expanding"``); NaN where the indicator or the
    history is missing."""
    if cuts == "is":
        return st.tercile(d[col], tb.is_cuts(d, col))
    c = expanding_cuts(d[col])
    v = d[col]
    out = pd.Series(
        np.where(v <= c["lo"], "low", np.where(v > c["hi"], "high", "mid")),
        index=d.index,
        dtype=object,
    )
    out[v.isna() | c["lo"].isna()] = np.nan
    return out


def t5(d: pd.DataFrame, tenor: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    o = d[d["has_outcome"]]
    blk = tb.STEP[tenor]
    rows = []
    r, lo, hi = st.bootstrap_ratio(o["D"], o["P_D"], blk)
    rows.append({"strike": "forward (K = 0)", "Σ payoff / Σ price": f"{r:.3f} [{lo:.3f}, {hi:.3f}]", "IS": tb.num(o.loc[o["IS"], "D"].sum() / o.loc[o["IS"], "P_D"].sum()),
                 "OOS": tb.num(o.loc[~o["IS"], "D"].sum() / o.loc[~o["IS"], "P_D"].sum()), "hit rate": tb.num((o["D"] > o["P_D"]).mean(), 2), "median payoff / price": tb.num((o["D"] / o["P_D"]).median(), 2),
                 "mean price": tb.pct(o["P_D"].mean()), "dates priced": len(o)})  # fmt: skip
    for m in tb.MULT:
        pay, price = o[f"PC_pay_{m}"], o[f"C_{m}"]
        priced = price >= 0.0005
        r, lo, hi = st.bootstrap_ratio(pay, price, blk)
        rows.append({"strike": f"{int(m) / 100:g} × forward price", "Σ payoff / Σ price": f"{r:.3f} [{lo:.3f}, {hi:.3f}]",
                     "IS": tb.num(pay[o["IS"]].sum() / price[o["IS"]].sum()), "OOS": tb.num(pay[~o["IS"]].sum() / price[~o["IS"]].sum()),
                     "hit rate": tb.num((pay > price).mean(), 2), "median payoff / price": tb.num((pay[priced] / price[priced]).median(), 2) if priced.any() else "–",
                     "mean price": tb.pct(price.mean()), "dates priced": int(priced.sum())})  # fmt: skip
    s = st.describe(o["PC_100_S"], tb.LAG[tenor])
    u = st.describe(o["PC_100_U"], tb.LAG[tenor])
    rows.append({"strike": "PC(1): mean P&L without / with the static basket hedge", "Σ payoff / Σ price": f"{tb.pct(u['mean'])} (t {u['t']:.2f}) / {tb.pct(s['mean'])} (t {s['t']:.2f})",
                 "IS": "", "OOS": "", "hit rate": f"sd {tb.pct(u['sd'])} / {tb.pct(s['sd'])}", "median payoff / price": "", "mean price": "", "dates priced": len(o)})  # fmt: skip
    cond = []
    for ilabel in ("variability", "|d rho_ATM| 21d", "|d ln sigma_SPX| 21d", "vol of vol SPX"):
        icol = tb.INDICATORS[ilabel]
        if icol not in o:
            continue
        cuts = tb.is_cuts(d, icol)
        for sample, g in (("IS", o[o["IS"]]), ("OOS", o[~o["IS"]])):
            lab = st.tercile(g[icol], cuts)
            for k in ("low", "mid", "high"):
                gg = g[lab == k]
                row: dict[str, Any] = {
                    "indicator": ilabel,
                    "sample": sample,
                    "tercile": k,
                    "n": len(gg),
                }
                row["forward"] = tb.num(gg["D"].sum() / gg["P_D"].sum(), 2) if len(gg) else "–"
                for m in ("075", "100", "125", "150"):
                    row[f"{int(m) / 100:g}"] = (
                        tb.num(gg[f"PC_pay_{m}"].sum() / gg[f"C_{m}"].sum(), 2)
                        if len(gg) and gg[f"C_{m}"].sum() > 0
                        else "–"
                    )
                row["PC(1) static hedge, per premium"] = tb.num(
                    (gg["PC_100_S"] / gg["C_100"].where(gg["C_100"] >= 0.0005)).mean(), 2
                )
                cond.append(row)
    return pd.DataFrame(rows), pd.DataFrame(cond)


def t6(d: pd.DataFrame, tenor: str) -> pd.DataFrame:
    o = d[d["has_outcome"]]
    lag = tb.LAG[tenor]
    rows = []
    for label, y in (
        ("PF U − PF H", o["PF_U"] - o["PF_H"]),
        ("PF U − PKG_theta H", o["PF_U"] - o["PKG_theta_H"]),
    ):
        row: dict[str, Any] = {"quantity": label}
        for sample in ("IS", "OOS"):
            c = tb.tercile_contrast(d, "VR_cs", y, tenor, sample)
            row[f"{sample} low"], row[f"{sample} mid"], row[f"{sample} high"] = (
                tb.pct(c["low"]),
                tb.pct(c["mid"]),
                tb.pct(c["high"]),
            )
            row[f"{sample} high − low"] = tb.pm(c["diff"], c["se"])
        rows.append(row)
    groups = [("all", o), ("IS", o[o["IS"]]), ("OOS", o[~o["IS"]])] + [
        (lab, o[o["period"] == lab]) for lab, _, _ in tb.PERIODS
    ]
    row = {"quantity": "realised VR of the windows (V_T / Σ CSV_t): mean ± s.e."}
    txt = []
    for label, g in groups:
        m, se, _ = st.mean_se(g["vr_window"], lag)
        txt.append(f"{label} {m:.2f} ± {se:.2f}")
    row["IS low"] = "; ".join(txt)
    rows.append(row)
    for sample, g in (("IS", o[o["IS"]]), ("OOS", o[~o["IS"]]), ("all", o)):
        fit = st.ols(g["vr_window"], np.column_stack([np.ones(len(g)), g["VR_cs"]]), lag)
        rows.append(
            {
                "quantity": f"realised window VR = a + b × trailing VR_cs, {sample}",
                "IS low": f"a {fit['coef'][0]:.2f} ± {fit['se'][0]:.2f}; b {fit['coef'][1]:.2f} ± {fit['se'][1]:.2f} (t {fit['t'][1]:.1f}); R² {fit['r2']:.3f}; n {fit['n']}",
            }
        )
    return pd.DataFrame(rows).fillna("")


def t9(d: pd.DataFrame, tenor: str) -> pd.DataFrame:
    o = d[d["has_outcome"]]
    rows = []
    for label, col in tb.STRUCTS.items():
        if col not in o or o[col].notna().sum() < 60:
            continue
        row: dict[str, Any] = {"structure": label}
        for sample, g in (("all", o), ("IS", o[o["IS"]]), ("OOS", o[~o["IS"]])):
            fit = st.ols(
                g[col], np.column_stack([np.ones(len(g)), g["vol_ratio_real"]]), tb.LAG[tenor]
            )
            row[f"slope {sample}"] = f"{100 * fit['coef'][1]:.2f} ± {100 * fit['se'][1]:.2f}"
            row[f"t {sample}"] = tb.num(fit["t"][1], 1)
            row[f"R² {sample}"] = tb.num(fit["r2"], 2)
        rows.append(row)
    return pd.DataFrame(rows)


def t10(d: pd.DataFrame, tenor: str) -> pd.DataFrame:
    o = d[d["has_outcome"]]
    rows = []
    for ilabel, icol, scope in (("earnings share", "earnings_share", "ex ante"), ("front event premium", "front_premium", "ex ante"),
                                ("trailing xs kurtosis", "xs_kurtosis", "ex ante"), ("the window's own kurtosis (check: mechanical)", "kurtosis", "ex post")):  # fmt: skip
        if icol not in o:
            continue
        cuts = tb.is_cuts(d, icol) if scope == "ex ante" else st.tercile_cuts(o[icol])
        for sample, g in (("IS", o[o["IS"]]), ("OOS", o[~o["IS"]])):
            lab = st.tercile(g[icol], cuts)
            row: dict[str, Any] = {"sort": ilabel, "sample": sample}
            for k in ("low", "mid", "high"):
                gg = g[lab == k]
                row[f"D/√V pooled, {k}"] = (
                    tb.num(gg["D"].mean() / np.sqrt(gg["V"].mean())) if len(gg) else "–"
                )
                row[f"n {k}"] = len(gg)
            row["mean of D/√V, low / high"] = (
                f"{g.loc[lab == 'low', 'kappa'].mean():.3f} / {g.loc[lab == 'high', 'kappa'].mean():.3f}"
            )
            c = tb.tercile_contrast(d, icol, o["kappa"], tenor, sample, cuts)
            row["high − low (mean D/√V)"] = f"{c['diff']:.3f} ± {c['se']:.3f}"
            rows.append(row)
    return pd.DataFrame(rows)


def t11(d: pd.DataFrame, tenor: str) -> pd.DataFrame:
    o = d[d["has_outcome"]].copy()
    edges = [-np.inf, -0.10, -0.03, 0.03, 0.10, np.inf]
    labels = ["below −10 %", "−10 to −3 %", "±3 %", "3 to 10 %", "above 10 %"]
    o["bucket"] = pd.cut(o["Rb"], edges, labels=labels)
    o["rel"] = o["D_rel"] / o["E_DB"]
    o["rel_vol"] = o["rel"] / (o["vol_names_real"] / o["sig_bar"])
    base, base_v = o["rel"].mean(), o["rel_vol"].mean()
    rows = []
    for j, k in enumerate(labels):
        g = o[o["bucket"] == k]
        m, se, _ = st.mean_se(g["rel"], tb.LAG[tenor])
        mv, sev, _ = st.mean_se(g["rel_vol"], tb.LAG[tenor])
        rows.append({"basket performance": k, "windows": len(g), "realised D/B over its forward": f"{m:.3f} ± {se:.3f}", "relative to all windows": tb.num(m / base, 2),
                     "divided by realised / implied vol": f"{mv:.3f} ± {sev:.3f}", "relative to all (vol-adjusted)": tb.num(mv / base_v, 2),
                     "copula: E[D/B | bucket] / E[D/B]": tb.num(d[f"profile_{j}"].mean(), 2), "copula: probability of the bucket": tb.num(d[f"profile_share_{j}"].mean(), 3),
                     "realised frequency": tb.num(len(g) / len(o), 3)})  # fmt: skip
    return pd.DataFrame(rows)


def risk_scale(d: pd.DataFrame, col: str) -> pd.Series:
    """Per entry: the standard deviation of ``col`` over the windows that **expired** in the
    trailing three years (NaN until 52 such windows exist: unit notional is then used)."""
    dates = d["date"].to_numpy(str)
    expiry = d["expiry"].fillna("9999").to_numpy(str)
    y = d[col].to_numpy(float)
    out = np.full(len(d), np.nan)
    for i in range(len(d)):
        lo = (pd.Timestamp(str(dates[i])) - pd.Timedelta(days=3 * 365)).strftime("%Y-%m-%d")
        sel = (expiry < dates[i]) & (dates >= lo) & np.isfinite(y)
        if sel.sum() >= 52:
            out[i] = y[sel].std(ddof=1)
    return pd.Series(out, index=d.index)


def rule_r1(d: pd.DataFrame, root: str = "absolute", cuts: str = "is") -> pd.Series:
    """The leaf of rule R1 at each entry (spec T14, v2.3): a structure column, or NaN when an
    indicator on the path is NaN (no position).

    ``root``: ``"absolute"`` sends ``CRP < 0`` (implied correlation below its forecast) to the
    reverse trade, as the spec v2.3 prescribes; ``"tercile"`` is the earlier bottom-tercile
    root, kept as a sensitivity.  ``cuts``: ``"is"`` labels the terciles on the in-sample cut
    points (R1: out of sample a real-time rule, in sample not), ``"expanding"`` on the cut points
    of the entries before each date with at least two years of history (R1-rt, the in-sample
    version).  The call leaf is :func:`call_leaf`."""
    leaf_call = call_leaf(d)
    lab = {c: tercile_labels(d, c, cuts) for c in ("CRP", "GP_G", "variability", "vol_premium")}
    # on the expanding terciles the whole tree starts once two years of history exist: a rule
    # whose only possible leaf is the root's would be another rule
    ready = (
        expanding_cuts(d["CRP"])["lo"].notna()
        if cuts == "expanding"
        else pd.Series(True, index=d.index)
    )
    out = pd.Series(np.nan, index=d.index, dtype=object)
    for i in d.index:
        if not ready[i]:
            continue
        crp_v = d.at[i, "CRP"]
        crp, gp, var, vp, vr = (
            lab["CRP"][i],
            lab["GP_G"][i],
            lab["variability"][i],
            lab["vol_premium"][i],
            d.at[i, "VR_cs"],
        )
        if root == "absolute":
            if not np.isfinite(crp_v):
                continue
            if crp_v < 0:
                out[i] = "REV_H"
                continue
        else:
            if pd.isna(crp):
                continue
            if crp == "low":
                out[i] = "REV_H"
                continue
        if pd.isna(gp) or not np.isfinite(vr):
            continue
        if gp == "low" or vr > 1.1:
            if pd.isna(var):
                continue
            if var == "high":
                out[i] = leaf_call
                continue
            if pd.isna(vp):
                continue
            base = "PF_v" if vp == "high" else "PF"
            out[i] = f"{base}_{'H' if vr < 0.9 else 'U'}"
        else:
            if pd.isna(vp):
                continue
            out[i] = "PKG_theta_H" if vp == "high" else "PKG_v_H"
    return out


#: The rules of T14 and the samples in which each is a real-time rule: R1 (in-sample
#: terciles) out of sample only; R1-rt (expanding terciles) and R2 (FHS forecasts) in both.
RULES = {
    "R1-rt": ("absolute", "expanding", ("IS", "OOS")),
    "R1": ("absolute", "is", ("OOS",)),
    "R1-rt, tercile root (sensitivity)": ("tercile", "expanding", ("IS", "OOS")),
    "R1, tercile root (sensitivity)": ("tercile", "is", ("OOS",)),
}


def rules(d: pd.DataFrame, with_r2: bool) -> dict[str, pd.Series]:
    """Every rule's pick at each entry: the R1 variants of :data:`RULES` and R2 (NaN picks
    where a rule is not computed)."""
    out = {name: rule_r1(d, root, cuts) for name, (root, cuts, _) in RULES.items()}
    out["R2"] = rule_r2(d) if with_r2 else pd.Series(np.nan, index=d.index, dtype=object)
    return out


def samples_of(name: str) -> tuple[str, ...]:
    return RULES[name][2] if name in RULES else ("IS", "OOS")


def episodes(mask: pd.Series, step: int) -> int:
    """Number of non-overlapping windows among the dates of ``mask`` (weekly entries; a window
    spans ``step`` entries): the leaf's number of independent episodes, which its t does not
    see when the dates cluster."""
    idx = np.flatnonzero(mask.to_numpy(bool))
    count, last = 0, -np.inf
    for i in idx:
        if i >= last + step:
            count += 1
            last = i
    return count


def leaf_stats(y: pd.Series, mask: pd.Series, lag: int) -> dict[str, float]:
    """Mean P&L of a leaf (the rows of ``mask``) with a time-aware standard error: the
    Newey–West (Bartlett) long-run variance at ``2·lag`` of the leaf's residuals at their place
    in the weekly series.  A leaf's dates cluster in a few episodes; on such a subset the
    equal-weight Hansen–Hodrick kernel of ``disp_stats.subset_mean_se`` can sum negative
    autocovariances to nearly nothing (a t of −10^9 on one 24m leaf), so T14 uses the Bartlett
    kernel, which is positive semi-definite."""
    v = y.to_numpy(float)
    m = mask.to_numpy(bool) & np.isfinite(v)
    n = int(m.sum())
    out = st.describe(v[m], 0)
    if n < 3:
        return out
    mean = float(v[m].mean())
    u = np.where(m, v - mean, 0.0)[:, None]
    var = float(st._long_run(u, 2 * lag, True)[0, 0]) / n**2
    # a long-run variance below the iid one is a small-sample artefact on a clustered subset
    # (the residuals of a cluster sum to zero): the standard error is floored at sd/√n
    iid = float(np.var(v[m], ddof=1)) / n
    se = float(np.sqrt(max(var, iid, 0.0)))
    out["t"] = mean / se if se > 0 else float("nan")
    return out


def rule_r2(d: pd.DataFrame) -> pd.Series:
    """The pick of rule R2 at each entry (spec T18): the unhedged candidate with the highest
    positive FHS forecast Sharpe; else the reverse trade if its forecast Sharpe is positive;
    else no position."""
    out = pd.Series(np.nan, index=d.index, dtype=object)
    if "fsr_PF" not in d:
        return out
    sr = pd.DataFrame({col: d[f"fsr_{k}"] for k, col in R2_CANDIDATES.items()})
    best = sr.fillna(-np.inf).idxmax(axis=1)
    top = sr.max(axis=1, skipna=True)
    for i in d.index:
        if not np.isfinite(top[i]):
            continue
        if top[i] > 0:
            out[i] = best[i]
        elif np.isfinite(d.at[i, "fsr_REV"]) and d.at[i, "fsr_REV"] > 0:
            out[i] = "REV_U"
        else:
            out[i] = "none"
    return out


def stream(d: pd.DataFrame, pick: pd.Series, scaled: bool) -> pd.Series:
    """P&L per entry of a rule's picks (NaN where it has no position); ``scaled`` divides each
    position by its trailing risk (unit notional until the risk exists)."""
    names = [c for c in pick.dropna().unique() if c != "none"]
    missing = [c for c in names if c not in d.columns]
    if missing:
        raise KeyError(f"picks name columns the frame does not have: {missing} (use prepare())")
    cols = names
    out = pd.Series(np.nan, index=d.index)
    scales = {c: risk_scale(d, c) for c in cols} if scaled else {}
    for c in cols:
        sel = pick == c
        y = d.loc[sel, c]
        if scaled:
            s = scales[c][sel]
            y = y / s.where(s > 0, 1.0).fillna(1.0)
        out[sel] = y
    out[pick == "none"] = 0.0
    return out


def stream_stats(y: pd.Series, d: pd.DataFrame, tenor: str, scale: float = 100.0) -> dict[str, str]:
    s = st.describe(y, tb.LAG[tenor])
    no = y.iloc[:: tb.STEP[tenor]].dropna()
    cum = no.cumsum()
    dd_ = float((cum - cum.cummax()).min()) if len(cum) else np.nan
    return {"mean": f"{scale * s['mean']:.3f}", "t": tb.num(s["t"], 2), "sd": f"{scale * s['sd']:.3f}", "mean/sd": tb.num(s["mean/sd"], 2), "5 %": f"{scale * s['q05']:.3f}",
            "worst": f"{scale * s['worst']:.3f}", "max drawdown (non-overlapping)": f"{scale * dd_:.2f}", "positions": int(s["n"])}  # fmt: skip


def t14(
    d: pd.DataFrame, tenor: str, with_r2: bool
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.Series]]:
    """T14: the rules against the always-strategies (rows), the leaves, and the picks.  A rule
    appears only in the samples where it is a real-time rule (:data:`RULES`)."""
    o = d[d["has_outcome"]]
    picks = rules(d, with_r2)
    rule_names = [k for k, v in picks.items() if v.notna().any()]
    rows = []
    for scaled, unit in (
        (False, "unit notional, % of notional"),
        (True, "constant risk, in trailing standard deviations"),
    ):
        sc = 1.0 if scaled else 100.0
        strategies: list[tuple[str, pd.Series]] = [
            (name, stream(d, picks[name], scaled)) for name in rule_names
        ]
        for label, col in ALWAYS.items():
            if not scaled and label == "VD":
                continue
            strategies.append((f"always {label}", stream(d, pd.Series(col, index=d.index), scaled)))
        for name, y in strategies:
            for sample, sel in (("IS", o["IS"]), ("OOS", ~o["IS"])):
                if sample not in samples_of(name):
                    continue
                yy = y.reindex(o.index)[sel]
                rows.append(
                    {
                        "strategy": name,
                        "units": unit,
                        "sample": sample,
                        **stream_stats(yy, o[sel], tenor, sc),
                    }
                )
            if name in rule_names:
                for other in ("PF_U", "PKG_theta_H"):
                    base = stream(d, pd.Series(other, index=d.index), scaled)
                    for sample, sel in (("IS", o["IS"]), ("OOS", ~o["IS"])):
                        if sample not in samples_of(name):
                            continue
                        diff = (y - base).reindex(o.index)[sel]
                        s = st.describe(diff, tb.LAG[tenor])
                        rows.append({"strategy": f"{name} minus always {other.replace('_', ' ')}", "units": unit, "sample": sample, "mean": f"{sc * s['mean']:.3f}", "t": tb.num(s["t"], 2),
                                     "sd": f"{sc * s['sd']:.3f}", "mean/sd": tb.num(s["mean/sd"], 2), "5 %": "", "worst": "", "max drawdown (non-overlapping)": "", "positions": int(s["n"])})  # fmt: skip
    leaves = []
    for name in rule_names:
        pick = picks[name]
        p = pick.reindex(o.index)
        missing = (
            "no position (fewer than two years of history, or an indicator is missing)"
            if "rt" in name
            else "no position (an indicator is missing)"
        )
        for sample, sel in (("IS", o["IS"]), ("OOS", ~o["IS"])):
            if sample not in samples_of(name):
                continue
            n = int(sel.sum())
            leaves.append(
                {
                    "rule": name,
                    "sample": sample,
                    "leaf": missing,
                    "share of dates": tb.num(p[sel].isna().mean(), 2),
                    "mean P&L": "",
                    "t": "",
                    "n": int(p[sel].isna().sum()),
                    "episodes": "",
                }
            )
            for leaf in sorted(p[sel].dropna().unique()):
                m = sel & (p == leaf)
                if leaf == "none":
                    leaves.append(
                        {
                            "rule": name,
                            "sample": sample,
                            "leaf": "no position (no positive forecast Sharpe)",
                            "share of dates": tb.num(m.sum() / n, 2),
                            "mean P&L": "",
                            "t": "",
                            "n": int(m.sum()),
                            "episodes": "",
                        }
                    )
                    continue
                s = leaf_stats(o[leaf], m, tb.LAG[tenor])
                ep = episodes(m, tb.STEP[tenor])
                leaves.append(
                    {
                        "rule": name,
                        "sample": sample,
                        "leaf": leaf.replace("_", " "),
                        "share of dates": tb.num(m.sum() / n, 2),
                        "mean P&L": tb.pct(s["mean"]),
                        "t": tb.num(s["t"], 2) if ep >= 3 else "–",
                        "n": int(m.sum()),
                        "episodes": ep,
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(leaves), picks


def t15(d: pd.DataFrame, picks: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    last = d.index[-1]
    row = d.loc[last]
    ind = []
    for label, col in tb.INDICATORS.items():
        if col not in d or not np.isfinite(row[col]):
            continue
        pctile = st.expanding_percentile(d[col]).iloc[-1]
        cuts = tb.is_cuts(d, col)
        ind.append({"indicator": label, "value": f"{row[col]:.4g}", "percentile (expanding)": tb.num(100 * pctile, 0), "IS tercile": st.tercile(pd.Series([row[col]]), cuts).iloc[0],
                    "IS cut points": f"{cuts[0]:.3g} / {cuts[1]:.3g}"})  # fmt: skip
    px = []
    for label, col, scale in (("ρ_cop (the mark)", "rho_cop", 1), ("ρ_ATM", "rho_atm", 1), ("ρ_90", "rho_90", 1), ("ρ_ATM 1m / 12m", None, 1), ("implied dispersion ID (ATM), vol points", "ID", 1),
                              ("ID_VS (log contracts)", "ID_VS", 1), ("ID_VS at 30 days", "ID_VS_30d", 1), ("Palladium forward P_D, %", "P_D", 100), ("single-name strip SS, %", "SS_mkt", 100),
                              ("basket straddle Str_B, %", "Str_B_mkt", 100), ("gap price P_G, %", "P_G", 100), ("√E^Q[V], %", None, 100), ("κ_Q", "kappa_Q", 1), ("κ_Q^cop", "kappa_cop", 1),
                              ("position in the band", "band_pos", 1), ("call at the forward price, %", "C_100", 100), ("call at 1.5 ×, %", "C_150", 100),
                              ("∂P_D/∂ρ per +0.01, %", "dPD_drho", 100), ("∂Str_B/∂ρ per +0.01, %", "dStrB_drho", 100), ("∂P_D/∂σ per vol point, %", "dPD_dvol", 100),
                              ("λ_ρ", "lambda_rho", 1), ("λ_θ", "lam_theta", 1), ("h_v", "h_v", 1), ("Δ_c of the call at the forward", "delta_c_100", 1)):  # fmt: skip
        if col is None:
            v = (
                f"{row['rho_atm_1m']:.3f} / {row['rho_atm_12m']:.3f}"
                if label.startswith("ρ_ATM 1m")
                else f"{100 * np.sqrt(max(row['EQV'], 0)):.3f}"
            )
        else:
            v = f"{scale * row[col]:.3f}"
        px.append({"quantity": label, "value": v})
    for s in ("PF", "SS", "BS", "PKG_v", "PKG_theta", "PKG_rho", "GAP", "GAP_rho", "PF_v"):
        e = {
            "quantity": f"forecast edge of {s.replace('_', ' ')}, % of notional: G0 / G1 / FHS (forecast Sharpe)"
        }
        g0, g1 = row.get(f"edge_{s}_G0", np.nan), row.get(f"edge_{s}_G1", np.nan)
        f, sr = row.get(f"fedge_{s}", np.nan), row.get(f"fsr_{s}", np.nan)
        e["value"] = f"{tb.pct(g0)} / {tb.pct(g1)} / {tb.pct(f)} ({tb.num(sr, 2)})"
        px.append(e)
    for name in ("R1", "R1-rt", "R2"):
        pick = picks.get(name)
        if pick is None or pick.notna().sum() == 0:
            value = "not available"
        elif pd.notna(pick.iloc[-1]):
            value = str(pick.iloc[-1]).replace("_", " ")
        else:
            value = "no position (an indicator is missing)"
        px.append({"quantity": f"{name} selects", "value": value})
    return pd.DataFrame(ind), pd.DataFrame(px)


def t18_fhs(d: pd.DataFrame, tenor: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    o = d[d["has_outcome"]]
    names = {"PF": "PF_U", "SS": "SS_U", "BS": "BS_U", "PKG_v": "PKG_v_U", "PKG_theta": "PKG_theta_U", "PKG_rho": "PKG_rho_U", "GAP": "GAP_U",
             "GAP_rho": "GAP_rho_U", "PF_v": "PF_v_U", "PC_100": "PC_100_U", "PC_100_S": "PC_100_S", "PC_150": "PC_150_U"}  # fmt: skip
    rows, dec = [], []
    for s, col in names.items():
        ecol = f"fedge_{s}"
        if ecol not in o:
            continue
        for sample, g in (("IS", o[o["IS"]]), ("OOS", o[~o["IS"]])):
            fit = st.ols(g[col], np.column_stack([np.ones(len(g)), g[ecol]]), tb.LAG[tenor])
            rows.append({"forecast": "FHS", "structure": s, "sample": sample, "a": tb.pm(fit["coef"][0], fit["se"][0]), "b": f"{fit['coef'][1]:.2f} ± {fit['se'][1]:.2f}",
                         "t(b)": tb.num(fit["t"][1], 2), "t(b − 1)": tb.num((fit["coef"][1] - 1) / fit["se"][1], 2) if fit["se"][1] > 0 else "–", "R²": tb.num(fit["r2"], 2),
                         "mean edge": tb.pct(g[ecol].mean()), "mean P&L": tb.pct(g[col].mean()), "n": fit["n"]})  # fmt: skip
        g = o[[ecol, col]].dropna()
        if len(g) >= 50:
            q = pd.qcut(g[ecol], 10, labels=False, duplicates="drop")
            for k, gg in g.groupby(q):
                dec.append(
                    {
                        "forecast": "FHS",
                        "structure": s,
                        "decile": int(k) + 1,
                        "edge": gg[ecol].mean(),
                        "pnl": gg[col].mean(),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(dec)


def replica(d: pd.DataFrame, tenor: str) -> tuple[pd.DataFrame, pd.Series]:
    """T16, phase 3: the static replica ``c + a Σw|R_i| − b|R̄|`` estimated on the windows that
    expired by the end of the in-sample period, applied out of sample; and the P&L of the
    Palladium forward minus the replica."""
    o = d[d["has_outcome"]]
    fit_on = o[o["expiry"] <= dd.IS_END]
    X = np.column_stack([np.ones(len(fit_on)), fit_on["absR"], fit_on["absRb"]])
    c, a, b = np.linalg.lstsq(X, fit_on["D"].to_numpy(float), rcond=None)[0]
    rep_pay = c + a * d["absR"] + b * d["absRb"]
    rep_price = c + a * d["SS_mkt"] + b * d["Str_B_mkt"]
    pnl = (d["D"] - d["P_D"]) - (rep_pay - rep_price)
    rows = []
    for sample, sel in (
        ("fit sample (expired by 2016)", o["expiry"] <= dd.IS_END),
        ("OOS", ~o["IS"]),
    ):
        g = o[sel]
        err = (g["D"] - rep_pay.reindex(g.index)).to_numpy(float)
        s = st.describe(pnl.reindex(g.index), tb.LAG[tenor])
        rows.append({"sample": sample, "replica": f"{100 * c:.3f} % + {a:.3f} Σw|R_i| − {-b:.3f} |R̄|", "tracking error of the payoff (s.d.)": tb.pct(float(np.std(err))),
                     "R² of the payoff": tb.num(1 - np.var(err) / np.var(g["D"]), 2), "mean replica price": tb.pct(rep_price.reindex(g.index).mean()), "mean P_D": tb.pct(g["P_D"].mean()),
                     "PF − REP: mean P&L": tb.pct(s["mean"]), "t": tb.num(s["t"], 2), "sd": tb.pct(s["sd"]), "n": int(s["n"])})  # fmt: skip
    return pd.DataFrame(rows), pnl


def net_of_costs(d: pd.DataFrame, margin: str = "2 correlation points") -> pd.DataFrame:
    """The frame with every P&L column net of the costs of spec §11: half the bid-ask of each
    straddle leg (the vendor's quotes at the nearest listed strike and expiry), 1 bp of the
    notional traded by each daily hedge, and a dealer margin on the Palladium — two correlation
    points (``2·|∂P/∂ρ|``) or 5 % of the forward's price."""
    x = d.copy()
    hs_ss, hs_bs = x["half_spread_SS"].fillna(0.0), x["half_spread_B"].fillna(0.0)
    if margin == "2 correlation points":
        m_pf = 2.0 * x["dPD_drho"].abs()
        m_c = {m: 2.0 * x[f"dC_drho_{m}"].abs() for m in tb.MULT}
    else:
        m_pf = 0.05 * x["P_D"]
        m_c = {m: 0.05 * x[f"C_{m}"] for m in tb.MULT}
    hedge = {"PF": 1e-4 * x["traded_PF"], "SS": 1e-4 * x["traded_SS"], "BS": 1e-4 * x["traded_BS"]}
    for tag in ("U", "H"):
        h = {k: (v if tag == "H" else 0.0) for k, v in hedge.items()}
        pf, ss, bs = m_pf + h["PF"], hs_ss + h["SS"], hs_bs + h["BS"]
        x[f"PF_{tag}"] -= pf
        x[f"SS_{tag}"] -= ss
        x[f"BS_{tag}"] -= bs
        x[f"PKG_v_{tag}"] -= ss + bs
        x[f"PKG_theta_{tag}"] -= ss + x["lam_theta"].abs() * bs
        x[f"PKG_rho_{tag}"] -= ss + x["lambda_rho"].abs() * bs
        x[f"GAP_{tag}"] -= pf + ss + bs
        x[f"GAP_rho_{tag}"] -= pf + ss + x["lambda_rho"].abs() * bs
        x[f"PF_v_{tag}"] -= pf + x["h_v"].abs() * ss
        x[f"REV_{tag}"] -= ss + bs
    for m in tb.MULT:
        x[f"PC_{m}_U"] -= m_c[m]
    for m in CALL_LEAF_STRIKES:
        if f"PC_{m}_S" in x:
            x[f"PC_{m}_S"] -= m_c[m]
    return x


def break_even(d: pd.DataFrame, tenor: str, picks: dict[str, pd.Series]) -> pd.DataFrame:
    """The dealer margin at which the mean P&L (or the advantage over always-PKG_theta H) is
    zero: in correlation points (mean P&L over mean ``|∂P/∂ρ|``) and in % of the premium, with
    block-bootstrap 95 % intervals.  A negative margin: the structure lost money at mid."""
    o = d[d["has_outcome"]]
    blk = tb.STEP[tenor]
    rows = []

    def row(
        label: str, pnl: pd.Series, sens: pd.Series, prem: pd.Series, sel: pd.Series | None = None
    ) -> None:
        if sel is not None:
            pnl, sens, prem = pnl[sel], sens[sel], prem[sel]
        a, lo, hi = st.bootstrap_ratio(pnl, sens.abs(), blk)
        b, lo2, hi2 = st.bootstrap_ratio(pnl, prem, blk)
        rows.append({"structure": label, "break-even margin, correlation points": f"{a:.2f} [{lo:.2f}, {hi:.2f}]",
                     "break-even margin, % of premium": f"{100 * b:.1f} [{100 * lo2:.1f}, {100 * hi2:.1f}]", "n": int(pnl.notna().sum())})  # fmt: skip

    for sample, sel in (("all", o["D"].notna()), ("IS", o["IS"]), ("OOS", ~o["IS"])):
        row(f"PF U, {sample}", o["PF_U"], o["dPD_drho"], o["P_D"], sel)
        row(f"PC(1) U, {sample}", o["PC_100_U"], o["dC_drho_100"], o["C_100"], sel)
        row(f"GAP, {sample}", o["GAP_U"], o["dPD_drho"], o["P_D"], sel)
    base = o["PKG_theta_H"]
    for name in ("R1-rt", "R1", "R2"):
        pick = picks.get(name)
        if pick is None:
            continue
        p = pick.reindex(o.index)
        if p.notna().sum() == 0:
            continue
        y = stream(d, pick, False).reindex(o.index)
        # the leaves that buy a Palladium: the forward's sensitivity and premium for the forward
        # leaves, the call's own for a call leaf (the base net_of_costs uses)
        forward_leaf = p.isin(["PF_U", "PF_H", "PF_v_U", "PF_v_H"])
        sens = o["dPD_drho"].abs().where(forward_leaf, 0.0)
        prem = o["P_D"].where(forward_leaf, 0.0)
        for m in CALL_LEAF_STRIKES:
            call_leaf_m = p == f"PC_{m}_S"
            sens = sens.where(~call_leaf_m, o[f"dC_drho_{m}"].abs())
            prem = prem.where(~call_leaf_m, o[f"C_{m}"])
        for sample, sel in (("IS", o["IS"]), ("OOS", ~o["IS"])):
            if sample not in samples_of(name):
                continue
            m = sel & y.notna()
            if sens[m].sum() > 0:
                row(
                    f"{name} minus always PKG_theta H, {sample} (margin on its Palladium leaves)",
                    (y - base)[m],
                    sens[m],
                    prem[m],
                )
    return pd.DataFrame(rows)


def model_s_tables(d: pd.DataFrame, tenor: str) -> dict[str, pd.DataFrame]:
    """Phase 4 (spec §3.7): the skew-consistent copula against the base one on the monthly
    subset — prices, the richness split, the calls and the conditional profile."""
    path = dd.OUT / f"model_s_{tenor}.parquet"
    if not path.exists():
        return {}
    ms = pd.read_parquet(path)
    x = d.merge(ms, on="date", how="inner")
    x = x[x["converged"].astype(bool)]
    o = x[x["has_outcome"]]
    out: dict[str, pd.DataFrame] = {}
    rows = []
    groups = [("all", ms), ("IS", ms[ms["date"] <= dd.IS_END]), ("OOS", ms[ms["date"] > dd.IS_END])]
    for label, g in groups:
        gx = x[x["date"].isin(g["date"])]
        row: dict[str, Any] = {
            "sample": label, "monthly dates": len(g), "converged": int(g["converged"].sum()), "slope at a bound": int((g["slope_flag"] != "").sum()),
            "median c": tb.num(g["c"].median()), "median s": tb.num(g["s"].median()),
            "90 % put: market / Gaussian copula": tb.num((g["put90_mkt"] / g["put90_gaussian"]).median(), 2),
            "P_D^S / P_D": tb.num(gx["P_D_S"].sum() / gx["P_D"].sum()), "median of the ratio": tb.num((gx["P_D_S"] / gx["P_D"]).median()),
        }  # fmt: skip
        for m in ("075", "100", "125", "150"):
            row[f"C^S / C at {int(m) / 100:g}"] = tb.num(
                gx[f"C_S_{m}"].sum() / gx[f"C_{m}"].sum(), 2
            )
        row["√(E^S[V] / E^Q[V])"] = tb.num(float(np.sqrt(gx["EV_S"].sum() / gx["EQV"].sum())))
        row["C3 under S, max"] = f"{g['c3_S'].max():.4f}"
        rows.append(row)
    out["model_S"] = pd.DataFrame(rows)
    rows = []
    for label, g in (("all", o), ("IS", o[o["IS"]]), ("OOS", o[~o["IS"]])):
        g = g[g["strip_ok"]]
        if len(g) < 12:
            continue
        r0 = tb.richness(g)
        g2 = g.assign(P_D=g["P_D_S"], EV=g["EV_S"])
        r1 = tb.richness(g2)
        rows.append({"sample": label, "windows (monthly)": len(g), "base: P_D / D": tb.num(r0["richness"]), "base: convexity κ_Q/κ_P": tb.num(r0["convexity factor"]),
                     "model S: P_D / D": tb.num(r1["richness"]), "model S: convexity κ_Q/κ_P": tb.num(r1["convexity factor"]), "variance factor (same)": tb.num(r0["variance factor"]),
                     "κ_Q base / S": f"{r0['kappa_Q']:.3f} / {r1['kappa_Q']:.3f}", "κ_P": tb.num(r0["kappa_P"])})  # fmt: skip
    out["T4_model_S"] = pd.DataFrame(rows)
    rows = []
    blk = max(1, tb.STEP[tenor] // 4)  # monthly entries: about three months of windows per block
    for m in tb.MULT[:5]:
        row = {"strike": f"{int(m) / 100:g} × forward price"}
        for label, g in (("IS", o[o["IS"]]), ("OOS", o[~o["IS"]]), ("all", o)):
            a, lo, hi = st.bootstrap_ratio(g[f"PC_pay_{m}"], g[f"C_{m}"], blk)
            b, lo2, hi2 = st.bootstrap_ratio(g[f"PC_pay_{m}"], g[f"C_S_{m}"], blk)
            row[f"base, {label}"] = (
                f"{a:.2f} [{lo:.2f}, {hi:.2f}]" if label == "all" else tb.num(a, 2)
            )
            row[f"model S, {label}"] = (
                f"{b:.2f} [{lo2:.2f}, {hi2:.2f}]" if label == "all" else tb.num(b, 2)
            )
        rows.append(row)
    row = {"strike": "forward (K = 0)"}
    for label, g in (("IS", o[o["IS"]]), ("OOS", o[~o["IS"]]), ("all", o)):
        row[f"base, {label}"] = tb.num(g["D"].sum() / g["P_D"].sum(), 2)
        row[f"model S, {label}"] = tb.num(g["D"].sum() / g["P_D_S"].sum(), 2)
    out["T5_model_S"] = pd.DataFrame([row, *rows])
    rows = []
    labels = ["below −10 %", "−10 to −3 %", "±3 %", "3 to 10 %", "above 10 %"]
    edges = [-np.inf, -0.10, -0.03, 0.03, 0.10, np.inf]
    oo = o.assign(
        bucket=pd.cut(o["Rb"], edges, labels=labels),
        rel=o["D_rel"] / o["E_DB"],
        rel_s=o["D_rel"] / o["E_DB_S"],
    )
    for j, k in enumerate(labels):
        g = oo[oo["bucket"] == k]
        rows.append({"basket performance": k, "windows (monthly)": len(g), "realised D/B over the base forward": tb.num(g["rel"].mean()),
                     "over model S's forward": tb.num(g["rel_s"].mean()), "base copula profile": tb.num(x[f"profile_{j}"].mean(), 2),
                     "model S profile": tb.num(x[f"profile_S_{j}"].mean(), 2), "probability, base / S": f"{x[f'profile_share_{j}'].mean():.3f} / {x[f'profile_share_S_{j}'].mean():.3f}",
                     "realised frequency": tb.num(len(g) / max(len(oo), 1), 3)})  # fmt: skip
    out["T11_model_S"] = pd.DataFrame(rows)
    return out


def marks_table(d: pd.DataFrame, tenor: str) -> pd.DataFrame:
    """Phase 4 (spec §5.5): the unwind P&L at one third and two thirds of the life against the
    P&L at expiry, monthly subset."""
    path = dd.OUT / f"marks_{tenor}.parquet"
    if not path.exists():
        return pd.DataFrame()
    mk = pd.read_parquet(path)
    if "error" in mk:
        mk = mk[mk["error"].isna()]
    o = d[d["has_outcome"]].set_index("date")
    rows = []
    for label, col in (("PF U", "PF_U"), ("PF H", "PF_H"), ("PC(1) U", "PC_100_U"), ("SS U", "SS_U"), ("BS U", "BS_U"), ("PKG_v U", "PKG_v_U"),
                       ("PKG_v H", "PKG_v_H"), ("PKG_theta U", "PKG_theta_U"), ("PKG_theta H", "PKG_theta_H"), ("GAP U", "GAP_U"), ("PF_v U", "PF_v_U")):  # fmt: skip
        row: dict[str, Any] = {"structure": label}
        dates = sorted(set(mk["date"]) & set(o.index))
        final = o.loc[dates, col]
        for stage in ("1/3", "2/3"):
            g = (
                mk[(mk["stage"] == stage) & mk["date"].isin(dates)]
                .set_index("date")[col]
                .reindex(dates)
            )
            row[f"mean at {stage}"] = tb.pct(g.mean(), 2)
            row[f"sd at {stage}"] = tb.pct(g.std(), 2)
            row[f"5 % at {stage}"] = tb.pct(g.quantile(0.05), 2)
            row[f"corr. with expiry, {stage}"] = tb.num(
                float(np.corrcoef(g.fillna(g.mean()), final)[0, 1]), 2
            )
        row["mean at expiry"] = tb.pct(final.mean(), 2)
        row["sd at expiry"] = tb.pct(final.std(), 2)
        row["5 % at expiry"] = tb.pct(final.quantile(0.05), 2)
        row["entries"] = len(dates)
        rows.append(row)
    return pd.DataFrame(rows)


def contrasts(d: pd.DataFrame, tenor: str) -> pd.DataFrame:
    """Every conditional cell outside the five primary tests: top minus bottom in-sample
    tercile of each indicator for the main structures, with its Hansen–Hodrick t and the
    Benjamini–Hochberg q-value over all the cells of the table (spec §8)."""
    from scipy.special import ndtr

    o = d[d["has_outcome"]]
    rows = []
    for ilabel, icol in tb.INDICATORS.items():
        if icol not in o or o[icol].notna().sum() < 120:
            continue
        for label in tb.MAIN:
            col = tb.STRUCTS[label]
            for sample in ("IS", "OOS"):
                c = tb.tercile_contrast(d, icol, o[col], tenor, sample)
                rows.append({"indicator": ilabel, "structure": label, "sample": sample, "low": c["low"], "mid": c["mid"], "high": c["high"],
                             "high − low": c["diff"], "se": c["se"], "t": c["t"], "n": int(c["n"])})  # fmt: skip
    out = pd.DataFrame(rows)
    if len(out):
        p = 2.0 * (1.0 - ndtr(out["t"].abs().to_numpy(float)))
        out["p"] = p
        out["q (Benjamini–Hochberg)"] = st.bh_qvalues(p)
        scale = np.where(out["structure"].str.startswith("VD"), 1.0, 100.0)
        for c in ("low", "mid", "high", "high − low", "se"):
            out[c] = (out[c] * scale).round(3)
        out[["t", "p", "q (Benjamini–Hochberg)"]] = out[["t", "p", "q (Benjamini–Hochberg)"]].round(
            3
        )
    return out


def fhs_robustness(d: pd.DataFrame, tenor: str) -> pd.DataFrame:
    """The FHS forecasts under the two robustness settings of spec §7.2 (mean block 21 days;
    undemeaned standardised returns) beside the base one: mean forecast and its calibration
    slope for the forward, the theta-neutral package and the gap."""
    o = d[d["has_outcome"]]
    rows = []
    for tag, label in (
        ("", "base: mean block 10 days, demeaned"),
        ("_block21", "mean block 21 days"),
        ("_undemeaned", "undemeaned"),
    ):
        path = dd.OUT / f"fhs_{tenor}{tag}.parquet"
        if not path.exists():
            continue
        f = pd.read_parquet(path)
        f = f[f["basket"] == "B1"].set_index("date")
        row: dict[str, Any] = {"FHS setting": label, "dates": int(f["fhs_E_D"].notna().sum()), "mean E[D]": tb.pct(f["fhs_E_D"].mean()),
                               "mean E[V]^½": tb.pct(float(np.sqrt(f["fhs_E_V"].mean()))), "mean κ_fc": tb.num(f["fhs_kappa_fc"].mean()), "mean E[G]": tb.pct(f["fhs_E_G"].mean())}  # fmt: skip
        for s, col in (("PF", "PF_U"), ("PKG_theta", "PKG_theta_U"), ("GAP", "GAP_U")):
            e = o["date"].map(f[f"fedge_{s}"])
            fit = st.ols(o[col], np.column_stack([np.ones(len(o)), e]), tb.LAG[tenor])
            row[f"mean edge {s}"] = tb.pct(float(e.mean()))
            row[f"slope b, {s}"] = f"{fit['coef'][1]:.2f} ± {fit['se'][1]:.2f}"
        rows.append(row)
    return pd.DataFrame(rows)
