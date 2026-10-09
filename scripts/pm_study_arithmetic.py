"""PM results package of 2026-10-09, section C (third bullet): the dispersion study's arithmetic
redone at the local-correlation (LC) price.

    python scripts/pm_study_arithmetic.py [--lc-table lcm_3m_dev_repair.parquet]

Three results of the study's report are recomputed with the LC price in place of the copula's, by
the method the study used for its skew-consistent model ("model S"):

    P&L at the model's price = P&L at the copula's price + (copula price - model price),

per trade, in % of notional.  A *trade* is one entry date of the study at three months on basket
B1 at unit notional (``scripts/disp_tables.py::load``: the entry merged with its outcome, the
entry with a member stuck over the window left out).  The sample is the monthly subset on which
model S converged (``scripts/disp_tables2.py::model_s_tables``) intersected with the dates the LC
pass priced.

* the gap (Palladium forward minus the vega-neutral package) held to expiry (``GAP_U``) and
  delta-hedged (``GAP_H``): mean P&L per trade; the hedge is the study's in every column;
* payout per 1 of premium: ``sum(payoff) / sum(price)`` of the forward (``D`` over the price) and
  of the call struck at the copula forward's price (``PC_pay_100`` over the call's price), with
  the study's interval (``volsto.studies.disp_stats.bootstrap_ratio``: circular blocks, 2,000
  resamples, seed 11; blocks of 3 monthly entries as in ``model_s_tables``);
* the price-over-payoff split (``scripts/disp_tables.py::richness``, pooled means over trades):
  ``mean price / mean payoff = sqrt(EQV / V) x sqrt(E_model[V] / EQV) x kappa_model / kappa_real``
  with ``kappa_model = mean price / sqrt(mean E_model[V])``, ``kappa_real = mean D / sqrt(mean V)``.

The study's own printed numbers are recomputed first from the study's tables, on the study's own
samples; a bullet whose printed numbers are not reproduced gets no LC number (written pending).

Writes ``parts/C_arith.json`` and ``.md`` and ``tables/C_arith_*.csv`` of the package.  Reads the
study's tables only (nothing is written outside the package).
"""

# ruff: noqa: E501, RUF001
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_common as pc

from volsto.studies import disp_data as dd
from volsto.studies import disp_stats as st

log = logging.getLogger("pm_study_arithmetic")

PART = "C_arith"
SECTION = "C"
BASKET = "B1"
TENOR = "3m"
TODAY = "2026-10-02"
#: Blocks of the study's bootstraps: one tenor of weekly entries (``disp_tables.STEP["3m"]``) and
#: its monthly counterpart (``disp_tables2.model_s_tables``: ``STEP // 4``).
BLOCK_WEEKLY = 13
BLOCK_MONTHLY = BLOCK_WEEKLY // 4
#: Hansen-Hodrick lags: the study's 12 for weekly entries (``disp_tables.LAG["3m"]``); 2 for
#: monthly entries (three-month windows entered one month apart overlap at lags 1 and 2).
LAG_WEEKLY = 12
LAG_MONTHLY = 2
N_RESAMPLES = 2000
SEED_RATIO = 11  # disp_stats.bootstrap_ratio
SEED_SPLIT = 5  # disp_report.py, kq_ci (the interval of D4)
MODELS = ("copula", "model_s", "lc")
MODEL_LABEL = {"copula": "copula", "model_s": "model S", "lc": "LC"}
#: Price columns of each model: forward, E[V], calls by strike multiple.
FWD = {"copula": "P_D", "model_s": "P_D_S", "lc": "ED_lc"}
EVCOL = {"copula": "EV", "model_s": "EV_S", "lc": "EV_lc"}
CALL = {"copula": "C_{m}", "model_s": "C_S_{m}", "lc": "C_{m}_lc"}
STRIKES = ("050", "075", "100", "125", "150")  # the rows of the study's T5_model_S
FACTORS = (
    ("price_over_payoff", "price over payoff"),
    ("listed_part", "listed-option part"),
    ("model_part", "model part"),
    ("dispersion_model_over_listed", "dispersion priced, model over listed"),
    ("kappa_model_over_realised", "kappa, model over realised"),
)

SRC_STUDY = "outputs/dispersion/entries_3m.parquet + outcomes_3m.parquet + model_s_3m.parquet"
SRC_T17 = "outputs/dispersion/report/tables/T17_gap.csv; final report sec. 5.2 (Q2_final_sources.zip q2/f5_results.tex l.66)"
SRC_NOTE = "final report sec. 5.3, box 'What this means for the headline' (Q2_final_sources.zip q2/f5_results.tex l.131)"
SRC_T5 = "outputs/dispersion/report/tables/T5_calls.csv (report_q2.md, table T5)"
SRC_T5S = "outputs/dispersion/report/tables/T5_model_S.csv (report_q2.md, table T5_model_S)"
SRC_SPLIT = "final report sec. 5.3, table under eq. (split) (Q2_final_sources.zip q2/f5_results.tex l.114-118); report_q2.md tables T4, T4_model_S"
SRC_D4 = "outputs/dispersion/report/report_q2.md, table verdict_q2, row D4"

#: The study's printed numbers: (key, what, source, printed text, value, digits).  Keys are matched
#: to this script's values in :func:`reproduction`.
PRINTED: tuple[tuple[str, str, str, str, float, int], ...] = (
    ("gap.held.copula.weekly", "held gap, 1,010 weekly trades, copula price: mean", SRC_T17, "-0.335", -0.335, 3),
    ("gap.held.copula.weekly.se", "its Hansen-Hodrick s.e. (12 lags)", SRC_T17, "0.214", 0.214, 3),
    ("gap.hedged.copula.weekly", "hedged gap, 1,010 weekly trades, copula price: mean", SRC_T17, "-0.514", -0.514, 3),
    ("gap.hedged.copula.weekly.se", "its Hansen-Hodrick s.e. (12 lags)", SRC_T17, "0.088", 0.088, 3),
    ("gap.held.copula.study_monthly", "held gap, monthly subset, copula price", SRC_NOTE, "-0.34", -0.34, 2),
    ("gap.held.model_s.study_monthly", "held gap, monthly subset, model S price", SRC_NOTE, "+0.05", 0.05, 2),
    ("gap.hedged.copula.study_monthly", "hedged gap, monthly subset, copula price", SRC_NOTE, "-0.5", -0.5, 1),
    ("gap.hedged.model_s.study_monthly", "hedged gap, monthly subset, model S price", SRC_NOTE, "about -0.1", -0.1, 1),
    ("payout.forward.copula.weekly", "forward, 1,010 weekly trades, copula price", SRC_T5, "0.996", 0.996, 3),
    ("payout.forward.copula.weekly.lo", "its interval, lower", SRC_T5, "0.958", 0.958, 3),
    ("payout.forward.copula.weekly.hi", "its interval, upper", SRC_T5, "1.038", 1.038, 3),
    ("payout.call_100.copula.weekly", "call at the forward's price, 1,010 weekly trades, copula price", SRC_T5, "1.091", 1.091, 3),
    ("payout.call_100.copula.weekly.lo", "its interval, lower", SRC_T5, "0.822", 0.822, 3),
    ("payout.call_100.copula.weekly.hi", "its interval, upper", SRC_T5, "1.392", 1.392, 3),
    ("payout.forward.copula.study_monthly", "forward, monthly subset, copula price", SRC_T5S, "1.00", 1.00, 2),
    ("payout.forward.model_s.study_monthly", "forward, monthly subset, model S price", SRC_T5S, "1.06", 1.06, 2),
    ("payout.call_100.copula.study_monthly", "call at the forward's price, monthly subset, copula price", SRC_T5S, "1.16", 1.16, 2),
    ("payout.call_100.copula.study_monthly.lo", "its interval, lower", SRC_T5S, "0.87", 0.87, 2),
    ("payout.call_100.copula.study_monthly.hi", "its interval, upper", SRC_T5S, "1.51", 1.51, 2),
    ("payout.call_100.model_s.study_monthly", "call at the forward's price, monthly subset, model S price", SRC_T5S, "1.58", 1.58, 2),
    ("payout.call_100.model_s.study_monthly.lo", "its interval, lower", SRC_T5S, "1.16", 1.16, 2),
    ("payout.call_100.model_s.study_monthly.hi", "its interval, upper", SRC_T5S, "2.04", 2.04, 2),
    ("payout.call_050.copula.study_monthly", "call at 0.5 x the forward's price, monthly subset, copula price", SRC_T5S, "1.00", 1.00, 2),
    ("payout.call_050.model_s.study_monthly", "call at 0.5 x, monthly subset, model S price", SRC_T5S, "1.12", 1.12, 2),
    ("payout.call_075.copula.study_monthly", "call at 0.75 x, monthly subset, copula price", SRC_T5S, "1.02", 1.02, 2),
    ("payout.call_075.model_s.study_monthly", "call at 0.75 x, monthly subset, model S price", SRC_T5S, "1.23", 1.23, 2),
    ("payout.call_125.copula.study_monthly", "call at 1.25 x, monthly subset, copula price", SRC_T5S, "1.21", 1.21, 2),
    ("payout.call_125.model_s.study_monthly", "call at 1.25 x, monthly subset, model S price", SRC_T5S, "2.02", 2.02, 2),
    ("payout.call_150.copula.study_monthly", "call at 1.5 x, monthly subset, copula price", SRC_T5S, "1.36", 1.36, 2),
    ("payout.call_150.model_s.study_monthly", "call at 1.5 x, monthly subset, model S price", SRC_T5S, "2.84", 2.84, 2),
    *(
        (f"split.{f}.copula.{s}", f"split, {label}, copula: {name}", SRC_SPLIT, f"{v:.3f}", v, 3)
        for s, label, values in (
            ("weekly", "all 1,010 trades", (1.004, 0.973, 1.032, 1.051, 0.982)),
            ("weekly_is", "entries 2007-2016", (1.038, 0.996, 1.042, 1.054, 0.989)),
            ("weekly_oos", "entries 2017-2026", (0.976, 0.956, 1.021, 1.049, 0.973)),
            ("study_monthly", "monthly subset", (0.999, 0.964, 1.036, 1.049, 0.988)),
        )
        for (f, name), v in zip(FACTORS, values, strict=True)
    ),
    *(
        (f"split.{f}.model_s.study_monthly", f"split, monthly subset, model S: {name}", SRC_SPLIT, f"{v:.3f}", v, 3)
        for (f, name), v in zip(FACTORS, (0.946, 0.964, 0.982, 1.012, 0.970), strict=True)
    ),
    ("split.model_part.copula.weekly_is.lo", "D4: model part 2007-2016, interval, lower", SRC_D4, "1.008", 1.008, 3),
    ("split.model_part.copula.weekly_is.hi", "D4: model part 2007-2016, interval, upper", SRC_D4, "1.073", 1.073, 3),
    ("split.model_part.copula.weekly_oos.lo", "D4: model part 2017-2026, interval, lower", SRC_D4, "1.004", 1.004, 3),
    ("split.model_part.copula.weekly_oos.hi", "D4: model part 2017-2026, interval, upper", SRC_D4, "1.040", 1.040, 3),
)  # fmt: skip


# --------------------------------------------------------------------------------- the samples
def study_trades() -> pd.DataFrame:
    """The study's three-month trades on basket B1, one row per entry date, as
    ``scripts/disp_tables.py::load`` builds them (entries left-merged with outcomes, the entry
    with a stuck member left out), with the flags the study's tables select on."""
    e = pd.read_parquet(pc.STUDY / f"entries_{TENOR}.parquet")
    o = pd.read_parquet(pc.STUDY / f"outcomes_{TENOR}.parquet")
    e, o = e[e["basket"] == BASKET], o[o["basket"] == BASKET]
    if e["date"].duplicated().any() or o["date"].duplicated().any():
        raise ValueError(f"entries or outcomes: more than one {BASKET} row on a date")
    d = e.merge(o, on=["date", "basket", "tenor"], how="left")
    d = d.sort_values("date").reset_index(drop=True)
    stuck = d["stuck_names"].fillna(0) > 0
    d.attrs["stuck_dates"] = list(d.loc[stuck, "date"])
    d = d[~stuck].reset_index(drop=True)
    d["IS"] = d["date"] <= dd.IS_END
    d["has_outcome"] = d["D"].notna()
    d["strip_ok"] = d["EQV"].gt(0) & d["EQV"].between(d["EV"] / 3.0, d["EV"] * 3.0)
    h = d[d["has_outcome"]]
    # the identities the arithmetic rests on, checked on the study's own columns
    checks = {
        "PF_U = D - P_D": (h["PF_U"] - (h["D"] - h["P_D"])).abs().max(),
        "GAP_U = G - P_G": (h["GAP_U"] - (h["G"] - h["P_G"])).abs().max(),
        "K_100 = P_D": (h["K_100"] - h["P_D"]).abs().max(),
        "PC_pay_100 = max(D - K_100, 0)": (h["PC_pay_100"] - np.maximum(h["D"] - h["K_100"], 0.0)).abs().max(),
    }  # fmt: skip
    bad = {k: float(v) for k, v in checks.items() if not v < 1e-12}
    if bad:
        raise ValueError(f"the study's columns do not satisfy {bad}")
    return d


def samples(lc_table: Path) -> dict[str, Any]:
    """The samples of the part and how they are counted."""
    d = study_trades()
    ms = pd.read_parquet(pc.STUDY / f"model_s_{TENOR}.parquet")
    lc = pd.read_parquet(lc_table)
    if ms["date"].duplicated().any() or lc["date"].duplicated().any():
        raise ValueError("model S or LC table: more than one row on a date")
    conv = ms[ms["converged"].astype(bool)]
    need = ["ED_lc", "ED_lc_se", "EV_lc", "EV_lc_se", *(f"C_{m}_lc" for m in STRIKES), *(f"C_{m}_lc_se" for m in STRIKES)]  # fmt: skip
    priced = lc[(lc["status"] != "failed") & np.isfinite(lc[need].astype(float)).all(axis=1)]
    weekly = d[d["has_outcome"]].reset_index(drop=True)
    # the study's monthly subset: model_s_tables, x = d.merge(ms), converged, has_outcome
    monthly = d.merge(conv, on="date", how="inner", suffixes=("", "_ms"))
    monthly = monthly[monthly["has_outcome"]].reset_index(drop=True)
    lc_cols = ["date", "status", "n_names_unscreened", "names_unscreened", "git_commit", "n_particles", "n_paths", "companion_paths",
               "P_D_copula", "EV_copula", "EQV", "K_100", *need]  # fmt: skip
    inter = monthly.merge(priced[lc_cols], on="date", how="inner", suffixes=("", "_lcrow"))
    inter = inter.sort_values("date").reset_index(drop=True)
    # the LC rows' copies of the study's columns are the study's
    for a, b in (("P_D", "P_D_copula"), ("EV", "EV_copula"), ("EQV", "EQV_lcrow"), ("K_100", "K_100_lcrow")):  # fmt: skip
        gap = float((inter[a] - inter[b]).abs().max())
        if not gap < 1e-12:
            raise ValueError(f"LC rows: {b} differs from the study's {a} by up to {gap:.3g}")
    inter["flagged"] = inter["n_names_unscreened"].fillna(0) > 0
    date_inter = sorted(set(priced["date"]) & set(conv["date"]))
    if TODAY in set(inter["date"]) or TODAY in set(monthly["date"]):
        raise ValueError(f"{TODAY} must not be in the sample (no outcome yet)")
    commits = sorted(set(inter["git_commit"].dropna().astype(str)))
    budget = inter[["n_particles", "n_paths", "companion_paths"]].drop_duplicates()
    if len(budget) != 1 or tuple(budget.iloc[0]) != (2e5, 2e5, 1e5):
        raise ValueError(
            f"the LC rows are not all at the development budget: {budget.to_dict('records')}"
        )
    return {
        "weekly": weekly,
        "weekly_is": weekly[weekly["IS"]].reset_index(drop=True),
        "weekly_oos": weekly[~weekly["IS"]].reset_index(drop=True),
        "study_monthly": monthly,
        "intersection": inter,
        "intersection_unflagged": inter[~inter["flagged"]].reset_index(drop=True),
        "meta": {
            "lc_rows": len(lc), "lc_priced": len(priced), "lc_failed": lc.loc[~lc["date"].isin(priced["date"]), ["date", "status", "reason"]].to_dict("records"),
            "lc_priced_today": TODAY in set(priced["date"]), "ms_rows": len(ms), "ms_converged": len(conv), "ms_converged_today": TODAY in set(conv["date"]),
            "date_intersection": len(date_inter), "dates_without_trade": sorted(set(date_inter) - set(inter["date"])),
            "stuck_dates": d.attrs["stuck_dates"], "study_monthly_not_priced": sorted(set(monthly["date"]) - set(inter["date"])),
            "flagged": inter.loc[inter["flagged"], ["date", "names_unscreened"]].to_dict("records"),
            "status_counts": inter["status"].value_counts().to_dict(), "commit": ", ".join(commits), "lc_table": lc_table.name,
        },
    }  # fmt: skip


# ------------------------------------------------------------------------------ the statistics
def block_index(n: int, block: int, seed: int) -> np.ndarray:
    """The resampled row indices of the study's circular block bootstrap (the lines of
    ``disp_stats.bootstrap_ratio`` and of ``disp_report``'s ``kq_ci``)."""
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(N_RESAMPLES, int(np.ceil(n / block))))
    return ((starts[:, :, None] + np.arange(block)[None, None, :]) % n).reshape(N_RESAMPLES, -1)[:, :n]  # fmt: skip


def finite(*columns: Any) -> np.ndarray:
    a = np.column_stack([np.asarray(c, dtype=np.float64) for c in columns])
    if not np.isfinite(a).all():
        raise ValueError(
            "a non-finite value in a sample (the study's bootstrap would drop the row)"
        )
    return a


def boot_ratio(num: Any, den: Any, block: int) -> dict[str, float]:
    """``sum(num) / sum(den)`` with the study's 95 % interval (checked against the study's own
    function) and the standard deviation of the same resamples."""
    a = finite(num, den)
    n = len(a)
    s = a[block_index(n, max(1, min(block, n)), SEED_RATIO)].sum(axis=1)
    draws = s[:, 0] / s[:, 1]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    out = {"value": float(a[:, 0].sum() / a[:, 1].sum()), "lo": float(lo), "hi": float(hi), "se": float(draws.std(ddof=1))}  # fmt: skip
    ref = st.bootstrap_ratio(num, den, block)
    if not np.allclose([out["value"], out["lo"], out["hi"]], ref, rtol=0, atol=1e-13):
        raise ValueError("boot_ratio does not reproduce volsto.studies.disp_stats.bootstrap_ratio")
    return out


def boot_mean(x: Any, block: int) -> dict[str, float]:
    """Mean with the block-bootstrap standard error and 95 % interval of the mean, on the
    resamples of :func:`boot_ratio`."""
    a = finite(x)[:, 0]
    n = len(a)
    draws = a[block_index(n, max(1, min(block, n)), SEED_RATIO)].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"value": float(a.mean()), "lo": float(lo), "hi": float(hi), "se": float(draws.std(ddof=1))}  # fmt: skip


def split_of(mP: Any, mEV: Any, mQ: Any, mD: Any, mV: Any) -> dict[str, Any]:
    """The pooled split from the five means (``disp_tables.richness``; scalars or arrays)."""
    kappa_model, kappa_real = mP / np.sqrt(mEV), mD / np.sqrt(mV)
    return {
        "price_over_payoff": mP / mD,
        "listed_part": np.sqrt(mQ / mV),
        "model_part": (mP / np.sqrt(mQ)) / kappa_real,
        "dispersion_model_over_listed": np.sqrt(mEV / mQ),
        "kappa_model_over_realised": kappa_model / kappa_real,
        "kappa_model": kappa_model,
        "kappa_realised": kappa_real,
    }


def split(g: pd.DataFrame, model: str, block: int) -> dict[str, dict[str, float]]:
    """The split on the trades of ``g`` at ``model``'s price, each factor with the standard
    deviation and the 95 % interval of the study's D4 bootstrap (``kq_ci``: seed 5, block not
    clipped)."""
    g = g[g["strip_ok"]]
    a = finite(g[FWD[model]], g[EVCOL[model]], g["EQV"], g["D"], g["V"])
    point = split_of(*a.mean(axis=0))
    mu = a[block_index(len(a), block, SEED_SPLIT)].mean(axis=1)
    draws = split_of(*mu.T)
    out = {}
    for k, v in point.items():
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        out[k] = {"value": float(v), "lo": float(lo), "hi": float(hi), "se": float(np.std(draws[k], ddof=1))}  # fmt: skip
    out["n"] = {"value": float(len(a))}
    out["mean_price"] = {"value": float(a[:, 0].mean())}
    out["mean_payoff"] = {"value": float(a[:, 3].mean())}
    return out


def mc_mean(se: pd.Series) -> tuple[float, float]:
    """Monte Carlo error of a mean over dates of per-date estimates with standard errors ``se``:
    with independent dates, and its bound whatever the dependence between dates (every date is
    priced on the same seeds): the mean of the standard errors."""
    s = np.asarray(se, dtype=np.float64)
    return float(np.sqrt((s**2).sum()) / len(s)), float(s.mean())


# ------------------------------------------------------------------------------------ the rows
def gap_rows(S: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for sample, block, lag in (
        ("weekly", BLOCK_WEEKLY, LAG_WEEKLY),
        ("study_monthly", BLOCK_MONTHLY, LAG_MONTHLY),
        ("intersection", BLOCK_MONTHLY, LAG_MONTHLY),
        ("intersection_unflagged", BLOCK_MONTHLY, LAG_MONTHLY),
    ):
        g = S[sample]
        for model in MODELS:
            if FWD[model] not in g:
                continue
            shift = g["P_D"] - g[FWD[model]]  # copula price - model price, per trade
            mc = mc_mean(g["ED_lc_se"]) if model == "lc" else (0.0, 0.0)
            # Monte Carlo error of the mean entry price and of the mean price difference: the
            # copula's from the study's P_D_se, LC's from ED_lc_se; model S's table has none
            nan = (float("nan"), float("nan"))
            price_mc = {"copula": mc_mean(g["P_D_se"]), "model_s": nan}.get(model, mc)
            shift_mc = mc_mean(np.hypot(g["ED_lc_se"], g["P_D_se"])) if model == "lc" else nan
            if model == "lc":
                shift_mc = (shift_mc[0], float((g["ED_lc_se"] + g["P_D_se"]).mean()))
            for structure, col in (("held", "GAP_U"), ("hedged", "GAP_H")):
                pnl = g[col] + shift
                b = boot_mean(pnl, block)
                _, hh, n = st.mean_se(pnl, lag)
                rows.append({
                    "sample": sample, "n_trades": n, "structure": structure, "model": model,
                    "mean_pnl_pct": 100 * b["value"], "se_boot_pct": 100 * b["se"], "ci95_lo_pct": 100 * b["lo"], "ci95_hi_pct": 100 * b["hi"],
                    "se_hansen_hodrick_pct": 100 * hh, "hh_lags": lag, "boot_block": block,
                    "mc_se_indep_pct": 100 * mc[0], "mc_se_bound_pct": 100 * mc[1],
                    "mean_gap_payoff_pct": 100 * g["G"].mean(), "mean_gap_price_pct": 100 * (g["P_G"] - shift).mean(),
                    "mean_forward_price_pct": 100 * g[FWD[model]].mean(),
                    "forward_price_mc_indep_pct": 100 * price_mc[0], "forward_price_mc_bound_pct": 100 * price_mc[1],
                    "mean_copula_minus_model_price_pct": 100 * shift.mean(),
                    "copula_minus_model_mc_indep_pct": 100 * shift_mc[0], "copula_minus_model_mc_bound_pct": 100 * shift_mc[1],
                    "hit_rate": float((pnl > 0).mean()),
                })  # fmt: skip
    return pd.DataFrame(rows)


def payout_rows(S: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for sample, block in (
        ("weekly", BLOCK_WEEKLY),
        ("study_monthly", BLOCK_MONTHLY),
        ("intersection", BLOCK_MONTHLY),
        ("intersection_unflagged", BLOCK_MONTHLY),
    ):
        g = S[sample]
        products = [("forward", "D", FWD, "ED_lc_se")]
        products += [(f"call_{m}", f"PC_pay_{m}", {k: v.format(m=m) for k, v in CALL.items()}, f"C_{m}_lc_se") for m in STRIKES]  # fmt: skip
        for product, pay, price, se_col in products:
            for model in MODELS:
                if price[model] not in g:
                    continue
                b = boot_ratio(g[pay], g[price[model]], block)
                total = float(g[price[model]].sum())
                se = g[se_col].to_numpy(float) if model == "lc" else np.zeros(len(g))
                rows.append({
                    "sample": sample, "n_trades": len(g), "product": product, "model": model,
                    "payout_per_1_of_premium": b["value"], "ci95_lo": b["lo"], "ci95_hi": b["hi"], "se_boot": b["se"], "boot_block": block,
                    "mc_se_indep": b["value"] * float(np.sqrt((se**2).sum())) / total, "mc_se_bound": b["value"] * float(se.sum()) / total,
                    "mean_payoff_pct": 100 * g[pay].mean(), "mean_premium_pct": 100 * g[price[model]].mean(),
                    "mean_premium_over_copula": float(g[price[model]].sum() / g[price["copula"]].sum()),
                })  # fmt: skip
    return pd.DataFrame(rows)


def split_rows(S: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for sample, block in (
        ("weekly", BLOCK_WEEKLY),
        ("weekly_is", BLOCK_WEEKLY),
        ("weekly_oos", BLOCK_WEEKLY),
        ("study_monthly", BLOCK_MONTHLY),
        ("intersection", BLOCK_MONTHLY),
        ("intersection_unflagged", BLOCK_MONTHLY),
    ):
        g = S[sample]
        for model in MODELS:
            if FWD[model] not in g:
                continue
            r = split(g, model, block)
            row: dict[str, Any] = {"sample": sample, "n_trades": int(r["n"]["value"]), "model": model, "boot_block": block,
                                   "mean_price_pct": 100 * r["mean_price"]["value"], "mean_payoff_pct": 100 * r["mean_payoff"]["value"]}  # fmt: skip
            for k in (*(f for f, _ in FACTORS), "kappa_model", "kappa_realised"):
                row[k] = r[k]["value"]
                row[f"{k}_se_boot"], row[f"{k}_ci95_lo"], row[f"{k}_ci95_hi"] = r[k]["se"], r[k]["lo"], r[k]["hi"]  # fmt: skip
            # Monte Carlo error of the LC row: relative errors of the mean price and of the mean
            # E[V] (independent dates; and the bound for any dependence, also between the two)
            rel = {"indep": (0.0, 0.0), "bound": (0.0, 0.0)}
            if model == "lc":
                gs = g[g["strip_ok"]]
                p, v = mc_mean(gs["ED_lc_se"]), mc_mean(gs["EV_lc_se"])
                mP, mEV = gs["ED_lc"].mean(), gs["EV_lc"].mean()
                rel = {"indep": (p[0] / mP, v[0] / mEV), "bound": (p[1] / mP, v[1] / mEV)}
            for kind, (rp, rv) in rel.items():
                row[f"price_over_payoff_mc_{kind}"] = row["price_over_payoff"] * rp
                row[f"listed_part_mc_{kind}"] = 0.0
                row[f"model_part_mc_{kind}"] = row["model_part"] * rp
                row[f"dispersion_model_over_listed_mc_{kind}"] = row["dispersion_model_over_listed"] * 0.5 * rv  # fmt: skip
                both = float(np.hypot(rp, 0.5 * rv)) if kind == "indep" else rp + 0.5 * rv
                row[f"kappa_model_over_realised_mc_{kind}"] = row["kappa_model_over_realised"] * both  # fmt: skip
                row[f"kappa_model_mc_{kind}"] = row["kappa_model"] * both
                row[f"kappa_realised_mc_{kind}"] = 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def reproduction(gap: pd.DataFrame, payout: pd.DataFrame, sp: pd.DataFrame) -> pd.DataFrame:
    """The study's printed numbers against this script's, on the study's own samples."""
    mine: dict[str, float] = {}
    for r in gap.to_dict("records"):
        key = f"gap.{r['structure']}.{r['model']}.{r['sample']}"
        mine[key], mine[key + ".se"] = r["mean_pnl_pct"], r["se_hansen_hodrick_pct"]
    for r in payout.to_dict("records"):
        key = f"payout.{r['product']}.{r['model']}.{r['sample']}"
        mine[key], mine[key + ".lo"], mine[key + ".hi"] = r["payout_per_1_of_premium"], r["ci95_lo"], r["ci95_hi"]  # fmt: skip
    for r in sp.to_dict("records"):
        for f, _ in FACTORS:
            key = f"split.{f}.{r['model']}.{r['sample']}"
            mine[key], mine[key + ".lo"], mine[key + ".hi"] = r[f], r[f"{f}_ci95_lo"], r[f"{f}_ci95_hi"]  # fmt: skip
    rows = []
    for key, what, source, text, value, digits in PRINTED:
        v = mine[key]
        rows.append({"key": key, "bullet": key.split(".")[0], "quantity": what, "study_printed": text, "this_script": v,
                     "digits": digits, "match": abs(round(v, digits) - value) < 0.5 * 10.0**-digits * 1e-6, "source": source})  # fmt: skip
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- records and the Markdown
def f3(x: Any, digits: int = 3) -> str:
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{digits}f}"


def sig(x: Any) -> str:
    """Two significant digits in fixed notation (for the small Monte Carlo errors)."""
    if x is None or not np.isfinite(x):
        return "n/a"
    if x == 0:
        return "0"
    return f"{x:.{max(0, 1 - int(np.floor(np.log10(abs(x)))))}f}"


def interval(lo: float, hi: float, digits: int = 3) -> str:
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


SAMPLE_LABEL = {
    "weekly": "the study's 1,010 weekly trades",
    "weekly_is": "the study's weekly trades, entries 2007-2016",
    "weekly_oos": "the study's weekly trades, entries 2017-2026",
    "study_monthly": "the study's monthly subset (model S converged)",
    "intersection": "the intersection (LC priced and model S converged)",
    "intersection_unflagged": "the intersection without the flagged dates",
}
DEF_GAP = (
    "mean over trades of the study's P&L of the gap (Palladium forward minus the vega-neutral package; {how}) "
    "at the copula's price + (copula forward price - {model} forward price) of the entry date, % of notional"
)
DEF_PAYOUT = "sum over trades of the realised payoff of {product} over the sum of its {model} price (the study's 'sum payoff / sum price')"
DEF_SPLIT = {
    "price_over_payoff": "mean {model} forward price over mean realised D, pooled over trades",
    "listed_part": "sqrt(mean EQV / mean realised V): the listed options' squared dispersion against the realised one (no model)",
    "model_part": "(mean {model} forward price / sqrt(mean EQV)) / kappa realised = price over payoff / listed-option part",
    "dispersion_model_over_listed": "sqrt(mean E[V] of {model} / mean EQV)",
    "kappa_model_over_realised": "(mean {model} forward price / sqrt(mean E[V] of {model})) / (mean D / sqrt(mean V))",
    "kappa_model": "mean {model} forward price / sqrt(mean E[V] of {model}), pooled over trades",
    "kappa_realised": "mean realised D / sqrt(mean realised V), pooled over trades",
}


def build(S: dict[str, Any]) -> tuple[list[dict[str, Any]], str, dict[str, pd.DataFrame]]:
    meta = S["meta"]
    gap, payout, sp = gap_rows(S), payout_rows(S), split_rows(S)
    rep = reproduction(gap, payout, sp)
    ok = {b: bool(rep.loc[rep["bullet"] == b, "match"].all()) for b in ("gap", "payout", "split")}
    for b, good in ok.items():
        if not good:
            log.error("bullet %s: a printed number of the study is not reproduced; its LC numbers are written pending", b)  # fmt: skip
    ok["none"] = True  # the counts and the reproduction rows depend on no bullet
    src_lc = f"outputs/dispersion_lc/{meta['lc_table']} + {SRC_STUDY}"
    records: list[dict[str, Any]] = []

    def add(id_: str, quantity: str, value: float | None, se: float | None, *, model: str, bullet: str, table: str, n: int,
            unit: str, definition: str, notes: str = "") -> None:  # fmt: skip
        lc = model == "lc"
        if lc and not ok[bullet]:
            value, se, notes = None, None, "pending: a printed number of the study behind this bullet is not reproduced (tables/C_arith_reproduction.csv)"  # fmt: skip
        records.append(pc.record(
            f"C.arith.{id_}", SECTION, quantity, value, se, tenor=TENOR, unit=unit, definition=definition,
            budget=pc.BUDGETS["development" if lc else "study"], commit=meta["commit"] if lc else "the study's tables (no commit column)",
            source=f"tables/{table}.csv <- {src_lc if lc else SRC_STUDY}", n=n, notes=notes,
        ))  # fmt: skip

    # -- samples
    counts = [
        ("lc_rows", "LC variant development pass: entry dates", meta["lc_rows"]),
        ("lc_priced", "LC variant development pass: priced dates", meta["lc_priced"]),
        ("model_s_converged", "model S: converged dates", meta["ms_converged"]),
        ("date_intersection", "priced LC dates that are converged model S dates", meta["date_intersection"]),
        ("trades.study_monthly", "trades of the study's monthly subset", len(S["study_monthly"])),
        ("trades.intersection", "trades of the intersection", len(S["intersection"])),
        ("trades.intersection_flagged", "of which flagged (a name kept unscreened)", int(S["intersection"]["flagged"].sum())),
        ("trades.intersection_unflagged", "trades of the intersection without the flagged dates", len(S["intersection_unflagged"])),
    ]  # fmt: skip
    for id_, quantity, value in counts:
        add(f"sample.{id_}", quantity, value, None, model="lc" if "lc" in id_ or "intersection" in id_ else "copula", bullet="none", table="C_arith_sample",
            n=int(value), unit="count", definition="a count of entry dates (one trade per entry date)", notes="a count, no standard error")  # fmt: skip

    # -- reproduction
    for r in rep.to_dict("records"):
        add(f"repro.{r['key']}", f"reproduction: {r['quantity']}", r["this_script"], None, model="copula", bullet="none", table="C_arith_reproduction",
            n=len(S["weekly" if "weekly" in r["key"] else "study_monthly"]), unit="% of notional" if r["bullet"] == "gap" else "ratio",
            definition="the study's printed number recomputed by this script from the study's tables on the study's own sample",
            notes=f"study printed {r['study_printed']} ({r['source']}); match to the printed digits: {r['match']}")  # fmt: skip

    # -- the gap
    for r in gap.to_dict("records"):
        if r["sample"] == "weekly":
            continue
        how = (
            "held to expiry"
            if r["structure"] == "held"
            else "delta-hedged daily with the study's hedge"
        )
        mc = f"; Monte Carlo error of the LC prices in this mean: {sig(r['mc_se_indep_pct'])} with independent dates, at most {sig(r['mc_se_bound_pct'])} (mean of ED_lc_se)" if r["model"] == "lc" else ""  # fmt: skip
        base = f"gap.{r['structure']}.{r['model']}.{r['sample']}"
        common = dict(model=r["model"], bullet="gap", table="C_arith_gap", n=r["n_trades"], unit="% of notional",
                      definition=DEF_GAP.format(how=how, model=MODEL_LABEL[r["model"]]))  # fmt: skip
        add(base, f"{r['structure']} gap at the {MODEL_LABEL[r['model']]} price, mean P&L per trade, {SAMPLE_LABEL[r['sample']]}", r["mean_pnl_pct"], r["se_hansen_hodrick_pct"], **common,
            notes=f"se = sampling s.e. over trades: Hansen-Hodrick (the study's s.e. of a mean P&L, disp_stats.mean_se) with {r['hh_lags']} lags for monthly entries of three-month trades; "
                  f"block-bootstrap s.e. of the mean (circular blocks of {r['boot_block']} monthly entries, {N_RESAMPLES} resamples, seed {SEED_RATIO}): {r['se_boot_pct']:.3f}, "
                  f"95 % interval {interval(r['ci95_lo_pct'], r['ci95_hi_pct'])}; the study printed no s.e. for the gap on the monthly subset; share of trades with a positive P&L {r['hit_rate']:.3f}{mc}")  # fmt: skip
        if r["model"] == "lc":
            add(base + ".mc_bound", f"Monte Carlo error bound of the {r['structure']} gap at the LC price, {SAMPLE_LABEL[r['sample']]}", r["mc_se_bound_pct"], None, **common,
                notes=f"mean over trades of ED_lc_se, valid whatever the dependence between dates (same seeds on every date); with independent dates {sig(r['mc_se_indep_pct'])}; not itself an estimate with a standard error")  # fmt: skip
        if r["structure"] == "held" and r["model"] != "copula":
            add(f"gap.price_shift.{r['model']}.{r['sample']}", f"copula forward price minus {MODEL_LABEL[r['model']]} forward price, mean per trade, {SAMPLE_LABEL[r['sample']]}",
                r["mean_copula_minus_model_price_pct"], r["copula_minus_model_mc_indep_pct"] if r["model"] == "lc" else None, model=r["model"], bullet="gap", table="C_arith_gap", n=r["n_trades"], unit="% of notional",
                definition=f"mean over trades of P_D - {FWD[r['model']]}: what each P&L of the Palladium forward and of the gap gains at the {MODEL_LABEL[r['model']]} price",
                notes=(f"se = Monte Carlo error of the mean with independent dates (ED_lc_se and the study's P_D_se); at most {sig(r['copula_minus_model_mc_bound_pct'])} whatever the dependence between dates; a mean of entry prices, no sampling s.e. attached"
                       if r["model"] == "lc" else "model S's table carries no standard error; a mean of entry prices, no sampling s.e. attached"))  # fmt: skip
        if r["structure"] == "held":
            add(f"gap.forward_price.{r['model']}.{r['sample']}", f"mean {MODEL_LABEL[r['model']]} forward price, {SAMPLE_LABEL[r['sample']]}", r["mean_forward_price_pct"], None if r["model"] == "model_s" else r["forward_price_mc_indep_pct"],
                model=r["model"], bullet="gap", table="C_arith_gap", n=r["n_trades"], unit="% of notional", definition=f"mean over trades of {FWD[r['model']]}",
                notes=("model S's table carries no standard error" if r["model"] == "model_s" else
                       f"se = Monte Carlo error of the mean with independent dates ({'ED_lc_se' if r['model'] == 'lc' else 'the study P_D_se'}); at most {sig(r['forward_price_mc_bound_pct'])} whatever the dependence between dates")
                + "; a mean of entry prices, no sampling s.e. attached")  # fmt: skip

    # -- payout per 1 of premium
    for r in payout.to_dict("records"):
        if r["sample"] == "weekly":
            continue
        name = "the Palladium forward" if r["product"] == "forward" else f"the Palladium call struck at {int(r['product'][-3:]) / 100:g} x the copula forward's price"  # fmt: skip
        mc = f"; Monte Carlo error of the LC prices in this ratio: {sig(r['mc_se_indep'])} with independent dates, at most {sig(r['mc_se_bound'])}" if r["model"] == "lc" else ""  # fmt: skip
        base = f"payout.{r['product']}.{r['model']}.{r['sample']}"
        common = dict(model=r["model"], bullet="payout", table="C_arith_payout", n=r["n_trades"], unit="ratio",
                      definition=DEF_PAYOUT.format(product=name, model=MODEL_LABEL[r["model"]]))  # fmt: skip
        add(base, f"payout per 1 of premium of {name} at the {MODEL_LABEL[r['model']]} price, {SAMPLE_LABEL[r['sample']]}", r["payout_per_1_of_premium"], r["se_boot"], **common,
            notes=f"the study's 95 % interval {interval(r['ci95_lo'], r['ci95_hi'])} (disp_stats.bootstrap_ratio: circular blocks of {r['boot_block']} monthly entries, {N_RESAMPLES} resamples, seed {SEED_RATIO}); "
                  f"se = standard deviation of the same resamples; mean premium {r['mean_premium_pct']:.3f} % of notional{mc}")  # fmt: skip
        for end in ("lo", "hi"):
            add(f"{base}.ci95_{end}", f"95 % interval ({'lower' if end == 'lo' else 'upper'}) of the payout per 1 of premium of {name} at the {MODEL_LABEL[r['model']]} price, {SAMPLE_LABEL[r['sample']]}",
                r[f"ci95_{end}"], None, **common, notes="a percentile of the study's block bootstrap; not itself an estimate with a standard error")  # fmt: skip
        if r["model"] == "lc":
            add(base + ".mc_bound", f"Monte Carlo error bound of the payout per 1 of premium of {name} at the LC price, {SAMPLE_LABEL[r['sample']]}", r["mc_se_bound"], None, **common,
                notes=f"ratio x sum of the per-date standard errors over the sum of prices, valid whatever the dependence between dates; with independent dates {sig(r['mc_se_indep'])}")  # fmt: skip

    # -- the split
    for r in sp.to_dict("records"):
        if r["sample"].startswith("weekly"):
            continue
        for k, label in (
            *FACTORS,
            ("kappa_model", "kappa of the model"),
            ("kappa_realised", "kappa realised"),
        ):
            if k in ("listed_part", "kappa_realised") and r["model"] != "copula":
                continue  # no model in it: one record per sample
            mc = f"; Monte Carlo error of the LC prices: {sig(r[f'{k}_mc_indep'])} with independent dates, at most {sig(r[f'{k}_mc_bound'])}" if r["model"] == "lc" else ""  # fmt: skip
            base = f"split.{k}.{r['model']}.{r['sample']}"
            common = dict(model=r["model"], bullet="split", table="C_arith_split", n=r["n_trades"], unit="ratio", definition=DEF_SPLIT[k].format(model=MODEL_LABEL[r["model"]]))  # fmt: skip
            add(base, f"price-over-payoff split, {label}, {MODEL_LABEL[r['model']]}, {SAMPLE_LABEL[r['sample']]}", r[k], r[f"{k}_se_boot"], **common,
                notes=f"se = sampling s.e. over trades: the bootstrap of the study's D4 (circular blocks, {N_RESAMPLES} resamples, seed {SEED_SPLIT}) with blocks of {r['boot_block']} monthly entries, "
                      f"95 % interval {interval(r[f'{k}_ci95_lo'], r[f'{k}_ci95_hi'])}; the study printed no interval on the monthly subset{mc}")  # fmt: skip
            if r["model"] == "lc":
                add(base + ".mc_bound", f"Monte Carlo error bound of the split's {label}, LC, {SAMPLE_LABEL[r['sample']]}", r[f"{k}_mc_bound"], None, **common,
                    notes=f"from the means of ED_lc_se and EV_lc_se, valid whatever the dependence between dates and between the two estimates; with independent dates and estimates {sig(r[f'{k}_mc_indep'])}")  # fmt: skip

    md = markdown(S, gap, payout, sp, rep, ok)
    sample_table = pd.DataFrame([{"id": i, "quantity": q, "value": v} for i, q, v in counts])
    return records, md, {"C_arith_sample": sample_table, "C_arith_reproduction": rep, "C_arith_gap": gap, "C_arith_payout": payout, "C_arith_split": sp}  # fmt: skip


def markdown(S: dict[str, Any], gap: pd.DataFrame, payout: pd.DataFrame, sp: pd.DataFrame, rep: pd.DataFrame, ok: dict[str, bool]) -> str:  # fmt: skip
    meta = S["meta"]
    n_i, n_u, n_m = (
        len(S["intersection"]),
        len(S["intersection_unflagged"]),
        len(S["study_monthly"]),
    )
    out: list[str] = [
        "### The gap, the payout per 1 of premium and the price-over-payoff split at the LC price (3m, the monthly subset)",
        "",
        f"Source: `outputs/dispersion_lc/{meta['lc_table']}` (commit {meta['commit']}, {pc.BUDGETS['development']}), joined on the entry date with the study's `entries_3m.parquet` and `outcomes_3m.parquet` "
        "(basket B1: one row per date, checked) and `model_s_3m.parquet`. Method, the study's for model S: P&L at the model's price = P&L at the copula's price + (copula price − model price), per trade, in % of notional.",
        "",
        "Files: `tables/C_arith_sample.csv`, `C_arith_reproduction.csv`, `C_arith_gap.csv`, `C_arith_payout.csv`, `C_arith_split.csv` (long tables with every standard error, interval and Monte Carlo error), `C_arith_trades.csv` (one row per trade of the intersection).",
        "",
    ]  # fmt: skip
    pending = "pending (a printed number of the study behind this bullet is not reproduced)"
    not_priced = f"not defined ({len(meta['study_monthly_not_priced'])} of the {n_m} trades not priced by the LC pass)"

    def cell(frame: pd.DataFrame, value: str, se: str | None, bullet: str, **sel: Any) -> str:
        q = frame
        for k, v in sel.items():
            q = q[q[k] == v]
        if q.empty:
            return not_priced if sel.get("model") == "lc" else "n/a"
        if sel.get("model") == "lc" and not ok[bullet]:
            return pending
        r = q.iloc[0]
        if se is None and value.endswith(("_bound_pct", "_bound")):
            return sig(float(r[value]))
        return pc.pm(float(r[value]), float(r[se]) if se else None, 3)

    # samples
    flagged = ", ".join(f"{r['date']} ({r['names_unscreened']})" for r in meta["flagged"])
    failed = ", ".join(str(r["date"]) for r in meta["lc_failed"])
    out += [
        "**Table C_arith_sample.** The trades.",
        "",
        "| sample | entry dates | trades |",
        "|---|---|---|",
        f"| LC variant development pass (`{meta['lc_table']}`, commit {meta['commit']}, {pc.BUDGETS['development']}): priced | {meta['lc_priced']} of {meta['lc_rows']} | |",
        f"| model S: converged | {meta['ms_converged']} of {meta['ms_rows']} | |",
        f"| priced LC dates that are converged model S dates | {meta['date_intersection']} | {n_i} |",
        f"| the same without the flagged dates (a name kept unscreened) | | {n_u} |",
        f"| the study's monthly subset (model S converged, the study's own sample) | | {n_m} |",
        f"| the study's weekly sample | | {len(S['weekly'])} |",
        "",
        f"A trade is one entry date of the study at 3m on basket B1 at unit notional with its outcome (the gap, the forward and the call of that date); the LC pass failed on {failed}; "
        f"the intersection is the study's monthly subset less {', '.join(meta['study_monthly_not_priced'])} (not priced by the LC pass); "
        f"{', '.join(meta['dates_without_trade']) or 'no date'} is priced by both models but is not a trade of the study (a member with no price move over the window, `stuck_names`: left out by the study); "
        f"the {len(meta['flagged'])} flagged trades are {flagged}; "
        f"LC row status on the {n_i} trades: {', '.join(f'{k} {v}' for k, v in sorted(meta['status_counts'].items()))}. "
        f"{TODAY} is not in any sample: it has no outcome yet and model S did not converge on it "
        f"(LC priced it: {meta['lc_priced_today']}; model S converged: {meta['ms_converged_today']}).",
        "",
    ]  # fmt: skip

    # reproduction
    out += [
        "**Table C_arith_reproduction.** The study's printed numbers recomputed by this script from the study's tables, on the study's own samples.",
        "",
        "| quantity | study printed | this script | match to the printed digits |",
        "|---|---|---|---|",
    ]  # fmt: skip
    out += [f"| {r['quantity']} | {r['study_printed']} | {r['this_script']:.4f} | {'yes' if r['match'] else 'NO'} |" for r in rep.to_dict("records")]  # fmt: skip
    out += [
        "",
        f"{int(rep['match'].sum())} of {len(rep)} printed numbers reproduced; gap P&L in % of notional, the others ratios; sources of the printed numbers in `tables/C_arith_reproduction.csv` "
        "(the study's `report_q2.md` tables T4, T4_model_S, T5, T5_model_S, T17_gap, verdict row D4, and the final report's sections 5.2-5.3).",
        "",
    ]  # fmt: skip

    # the gap
    out += [
        "**Table C_arith_gap.** The gap held and hedged: mean P&L per trade, % of notional, ± Hansen–Hodrick standard error over trades.",
        "",
        "| structure | sample | trades | at the copula's price | at model S's price | at the LC price | LC: Monte Carlo error, at most | study printed (copula / model S) |",
        "|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    printed_gap = {
        ("held", "study_monthly"): "-0.34 / +0.05",
        ("hedged", "study_monthly"): "-0.5 / about -0.1",
    }
    for structure in ("held", "hedged"):
        for sample in ("study_monthly", "intersection", "intersection_unflagged"):
            sel = dict(structure=structure, sample=sample)
            mcb = cell(gap, "mc_se_bound_pct", None, "gap", model="lc", **sel)
            out.append(
                f"| {structure} gap | {SAMPLE_LABEL[sample]} | {len(S[sample])} | {cell(gap, 'mean_pnl_pct', 'se_hansen_hodrick_pct', 'gap', model='copula', **sel)} | "
                f"{cell(gap, 'mean_pnl_pct', 'se_hansen_hodrick_pct', 'gap', model='model_s', **sel)} | {cell(gap, 'mean_pnl_pct', 'se_hansen_hodrick_pct', 'gap', model='lc', **sel)} | "
                f"{'' if mcb == not_priced else mcb} | {printed_gap.get((structure, sample), '')} |"
            )  # fmt: skip
    for sample in ("intersection", "intersection_unflagged"):
        sel = dict(structure="held", sample=sample)
        out.append(
            f"| mean forward price | {SAMPLE_LABEL[sample]} | {len(S[sample])} | {cell(gap, 'mean_forward_price_pct', None, 'gap', model='copula', **sel)} | "
            f"{cell(gap, 'mean_forward_price_pct', None, 'gap', model='model_s', **sel)} | {cell(gap, 'mean_forward_price_pct', None, 'gap', model='lc', **sel)} | {cell(gap, 'forward_price_mc_bound_pct', None, 'gap', model='lc', **sel)} | |"
        )  # fmt: skip
        out.append(
            f"| copula price minus model price | {SAMPLE_LABEL[sample]} | {len(S[sample])} | | {cell(gap, 'mean_copula_minus_model_price_pct', None, 'gap', model='model_s', **sel)} | "
            f"{cell(gap, 'mean_copula_minus_model_price_pct', None, 'gap', model='lc', **sel)} | {cell(gap, 'copula_minus_model_mc_bound_pct', None, 'gap', model='lc', **sel)} | |"
        )  # fmt: skip
    out += [
        "",
        "Gap = Palladium forward minus the vega-neutral package; P&L at a model's price = the study's P&L at the copula's price (`GAP_U` held to expiry, `GAP_H` delta-hedged daily with the study's hedge) "
        f"+ (copula forward price − model forward price) of the entry date, per trade; ± = the study's standard error of a mean P&L (Hansen–Hodrick, `disp_stats.mean_se`) with {LAG_MONTHLY} lags for monthly entries "
        f"(the study printed none for the gap on the monthly subset; the block-bootstrap one, circular blocks of {BLOCK_MONTHLY} monthly entries, {N_RESAMPLES} resamples, seed {SEED_RATIO}, is in the CSV with its 95 % interval); "
        "the two price rows are means of entry prices and carry no sampling error; the Monte Carlo error is that of the LC prices in the mean (the mean over trades of `ED_lc_se`, plus the study's `P_D_se` in the price difference: "
        "a bound whatever the dependence between dates; with independent dates it is in the CSV).",
        "",
    ]  # fmt: skip

    # payout
    out += [
        "**Table C_arith_payout.** Payout per 1 of premium (Σ payoff / Σ price) with the study's 95 % interval.",
        "",
        "| product | sample | trades | copula | model S | LC | LC: Monte Carlo error, at most | mean premium, % of notional (copula / S / LC) | study printed (copula / model S) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    printed_pay = {
        "forward": "1.00 / 1.06", "call_100": "1.16 [0.87, 1.51] / 1.58 [1.16, 2.04]", "call_050": "1.00 [0.92, 1.09] / 1.12 [1.02, 1.22]",
        "call_075": "1.02 [0.88, 1.19] / 1.23 [1.05, 1.43]", "call_125": "1.21 [0.62, 1.94] / 2.02 [1.02, 3.24]", "call_150": "1.36 [0.35, 2.99] / 2.84 [0.71, 6.13]",
    }  # fmt: skip
    plabel = {"forward": "forward", "call_100": "call at the forward's price", "call_050": "call at 0.5 ×", "call_075": "call at 0.75 ×", "call_125": "call at 1.25 ×", "call_150": "call at 1.5 ×"}  # fmt: skip

    def pay_cell(product: str, sample: str, model: str) -> str:
        q = payout[(payout["product"] == product) & (payout["sample"] == sample) & (payout["model"] == model)]  # fmt: skip
        if q.empty:
            return not_priced if model == "lc" else "n/a"
        if model == "lc" and not ok["payout"]:
            return pending
        r = q.iloc[0]
        return f"{r['payout_per_1_of_premium']:.3f} {interval(r['ci95_lo'], r['ci95_hi'])}"

    def pay_rows(products: tuple[str, ...]) -> list[str]:
        lines = []
        for product in products:
            for sample in ("study_monthly", "intersection", "intersection_unflagged"):
                q = payout[(payout["product"] == product) & (payout["sample"] == sample)].set_index("model")  # fmt: skip
                prem = " / ".join(f"{q.loc[m, 'mean_premium_pct']:.3f}" if m in q.index else "n/a" for m in MODELS)  # fmt: skip
                mcb = sig(q.loc["lc", "mc_se_bound"]) if "lc" in q.index and ok["payout"] else ""
                lines.append(
                    f"| {plabel[product]} | {SAMPLE_LABEL[sample]} | {len(S[sample])} | {pay_cell(product, sample, 'copula')} | {pay_cell(product, sample, 'model_s')} | "
                    f"{pay_cell(product, sample, 'lc')} | {mcb} | {prem} | {printed_pay[product] if sample == 'study_monthly' else ''} |"
                )  # fmt: skip
        return lines

    out += pay_rows(("forward", "call_100"))
    out += [
        "",
        "Σ over trades of the realised payoff (`D` for the forward; `max(D − K, 0)` for the call, `K` = the copula forward's price of the entry date, in cash, the same for the three models) over Σ of the model's price "
        f"(`P_D`, `P_D_S`, `ED_lc`; `C_100`, `C_S_100`, `C_100_lc`); interval = the study's (`disp_stats.bootstrap_ratio`: circular blocks of {BLOCK_MONTHLY} monthly entries, {N_RESAMPLES} resamples, seed {SEED_RATIO}, 2.5th–97.5th percentile); "
        "the study's T5_model_S prints no interval for the forward on the monthly subset (the same bootstrap is applied here); the Monte Carlo error is that of the LC prices in the ratio (ratio × Σ standard errors / Σ prices, a bound).",
        "",
        "**Table C_arith_payout (other strikes).** The other strikes of the study's table T5_model_S, same definitions.",
        "",
        "| product | sample | trades | copula | model S | LC | LC: Monte Carlo error, at most | mean premium, % of notional (copula / S / LC) | study printed (copula / model S) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    out += pay_rows(("call_050", "call_075", "call_125", "call_150"))
    out += [
        "",
        "Strikes are multiples of the copula forward's price of the entry date, fixed in cash.",
        "",
    ]

    # the split
    out += [
        "**Table C_arith_split.** The price-over-payoff split: price over payoff = listed-option part × model part; model part = (dispersion priced, model over listed) × (κ, model over realised).",
        "",
        "| row | trades | price over payoff | = listed-option part | × model part | model part = dispersion priced, model over listed | × κ, model over realised | κ of the model | κ realised | study printed |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    printed_split = {
        ("weekly", "copula"): "1.004 = 0.973 × 1.032; 1.051 × 0.982", ("weekly_is", "copula"): "1.038 = 0.996 × 1.042; 1.054 × 0.989",
        ("weekly_oos", "copula"): "0.976 = 0.956 × 1.021; 1.049 × 0.973", ("study_monthly", "copula"): "0.999 = 0.964 × 1.036; 1.049 × 0.988",
        ("study_monthly", "model_s"): "0.946 = 0.964 × 0.982; 1.012 × 0.970",
    }  # fmt: skip
    for r in sp.to_dict("records"):
        lcrow = r["model"] == "lc"
        if lcrow and not ok["split"]:
            cells = [pending] * 6 + [f3(r["kappa_realised"])]
        else:
            show_se = not r["sample"].startswith("weekly")
            cells = [pc.pm(r[k], r[f"{k}_se_boot"] if show_se else None, 3) for k in (*(f for f, _ in FACTORS), "kappa_model", "kappa_realised")]  # fmt: skip
        out.append(f"| {SAMPLE_LABEL[r['sample']]}, {MODEL_LABEL[r['model']]} | {r['n_trades']} | " + " | ".join(cells) + f" | {printed_split.get((r['sample'], r['model']), '')} |")  # fmt: skip
    out += [
        "",
        "All ratios pooled over the trades (`disp_tables.richness`): price over payoff = mean model forward price / mean realised D; listed-option part = √(mean EQV / mean realised V), EQV the listed options' E[V] (the same for every model); "
        "dispersion priced, model over listed = √(mean model E[V] / mean EQV) (`EV`, `EV_S`, `EV_lc`); κ of the model = mean price / √(mean model E[V]); κ realised = mean D / √(mean V); "
        f"± on the monthly rows = standard deviation over the resamples of the study's D4 bootstrap (circular blocks of {BLOCK_MONTHLY} monthly entries, {N_RESAMPLES} resamples, seed {SEED_SPLIT}); the study printed no interval on these rows (95 % intervals in the CSV).",
        "",
        "**Table C_arith_split (Monte Carlo error of the LC rows).** The error of the LC prices in each factor: with independent dates / at most.",
        "",
        "| row | price over payoff | model part | dispersion priced, model over listed | κ, model over realised | κ of the model |",
        "|---|---|---|---|---|---|",
    ]  # fmt: skip
    for r in sp[sp["model"] == "lc"].to_dict("records"):
        if not ok["split"]:
            out.append(f"| {SAMPLE_LABEL[r['sample']]}, LC | " + " | ".join([pending] * 5) + " |")
            continue
        cells = [f"{sig(r[f'{k}_mc_indep'])} / {sig(r[f'{k}_mc_bound'])}" for k in ("price_over_payoff", "model_part", "dispersion_model_over_listed", "kappa_model_over_realised", "kappa_model")]  # fmt: skip
        out.append(f"| {SAMPLE_LABEL[r['sample']]}, LC | " + " | ".join(cells) + " |")
    out += [
        "",
        "Delta method on the mean of `ED_lc` and the mean of `EV_lc` with the rows' standard errors (`ED_lc_se`, `EV_lc_se`): first number with independent dates and independent estimates, "
        "second the bound whatever their dependence (means of the standard errors; every date is priced on the same seeds); the listed-option part and κ realised carry no Monte Carlo error.",
        "",
    ]  # fmt: skip
    return "\n".join(out)


def trades_table(S: dict[str, Any]) -> pd.DataFrame:
    """One row per trade of the intersection: what every number of the part is computed from."""
    g = S["intersection"]
    t = g[["date", "expiry", "IS", "flagged", "names_unscreened", "status", "git_commit"]].copy()
    for c in ("P_D", "P_D_S", "ED_lc", "ED_lc_se", "EV", "EV_S", "EV_lc", "EV_lc_se", "EQV", "D", "V", "G", "P_G", "GAP_U", "GAP_H", "K_100",
              "PC_pay_100", "C_100", "C_S_100", "C_100_lc", "C_100_lc_se"):  # fmt: skip
        t[c] = g[c].to_numpy(float)
    for model in ("model_s", "lc"):
        shift = g["P_D"] - g[FWD[model]]
        t[f"GAP_U_at_{model}"] = (g["GAP_U"] + shift).to_numpy(float)
        t[f"GAP_H_at_{model}"] = (g["GAP_H"] + shift).to_numpy(float)
    return t


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])  # fmt: skip
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--lc-table", default="lcm_3m_dev_repair.parquet", help="the LC table in outputs/dispersion_lc (default: the variant development pass)")  # fmt: skip
    args = ap.parse_args()
    S = samples(pc.LC_OUT / args.lc_table)
    records, md, tables = build(S)
    tables["C_arith_trades"] = trades_table(S)
    for name, frame in tables.items():
        log.info("%s", pc.save_table(frame, name))
    part = pc.PM / "parts" / f"{PART}.json"
    before = part.read_text() if part.exists() else None
    pc.write_part(PART, records, md)
    rep = tables["C_arith_reproduction"]
    meta = S["meta"]
    log.info("%s", md)
    log.info("records: %d; reproduction: %d of %d printed numbers matched", len(records), int(rep["match"].sum()), len(rep))  # fmt: skip
    if part.read_text() == before:
        log.info("the part is unchanged: no new line in STATUS.md")
        return
    pc.status(
        f"C_arith written (parts/C_arith.json and .md, tables/C_arith_*.csv): the study's arithmetic at the LC price on {len(S['intersection'])} trades "
        f"({len(S['intersection_unflagged'])} without the flagged dates), LC rows of commit {meta['commit']} ({meta['lc_table']}); "
        f"{int(rep['match'].sum())} of {len(rep)} printed numbers of the study reproduced."
    )  # fmt: skip


if __name__ == "__main__":
    main()
