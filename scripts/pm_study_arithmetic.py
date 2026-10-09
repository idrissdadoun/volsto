"""PM results package of 2026-10-09, section C (third bullet): the dispersion study's arithmetic
redone at the local-correlation (LC) price.

    python scripts/pm_study_arithmetic.py [--lc-table lcm_3m_dev_repair.parquet] [--base DIR]

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

Added after the verification of 2026-10-09: the status of the LC rows under the current rule of
``scripts/lcm_price.py`` (``row_status``) with what fails; the rows without the dates that fail
the index gate; the split inside and outside the names' 2 % diagnostic and the second moments
behind LC's E[V] (:func:`moment_rows`); the t-ratio, the 3-lag Hansen-Hodrick standard error and
the range of the bootstrap bounds over 40 seeds; the call's strike against each model's own
forward.  ``--base DIR`` writes the part and its tables to another folder (a dry run).

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
import lcm_price as lp  # GATING_CHECKS and row_status: the current rule of a row's status
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
#: One more lag on the monthly samples, printed next to the 2-lag standard error (some pairs of
#: trades three entries apart overlap by a few days).
LAG_MONTHLY_MORE = 3
#: Bootstrap seeds over which the range of an interval's bounds is reported.
SEEDS_RANGE = tuple(range(1, 41))
#: The names' diagnostic of ``lcm_price`` (``check_names``): sum_w E_LC[R_i^2] within 2 % of the
#: listed strips.  Not a gate (owner's decision 3 of 2026-10-09).
NAMES_TOL = 0.02
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
SRC_SPLIT = "final report sec. 5.3, table under eq. (split) (Q2_final_sources.zip q2/f5_results.tex l.113-118); report_q2.md tables T4, T4_model_S"
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
    ("gap.hedged.copula.study_monthly", "hedged gap, copula price; printed to one digit, the study's sentence does not state the sample", SRC_NOTE, "-0.5", -0.5, 1),
    ("gap.hedged.model_s.study_monthly", "hedged gap, model S price; printed to one digit, the study's sentence does not state the sample", SRC_NOTE, "about -0.1", -0.1, 1),
    ("payout.forward.copula.weekly", "forward, 1,010 weekly trades, copula price", SRC_T5, "0.996", 0.996, 3),
    ("payout.forward.copula.weekly.lo", "its interval, lower", SRC_T5, "0.958", 0.958, 3),
    ("payout.forward.copula.weekly.hi", "its interval, upper", SRC_T5, "1.038", 1.038, 3),
    ("payout.call_100.copula.weekly", "call struck at the copula's forward price, 1,010 weekly trades, copula price", SRC_T5, "1.091", 1.091, 3),
    ("payout.call_100.copula.weekly.lo", "its interval, lower", SRC_T5, "0.822", 0.822, 3),
    ("payout.call_100.copula.weekly.hi", "its interval, upper", SRC_T5, "1.392", 1.392, 3),
    ("payout.forward.copula.study_monthly", "forward, monthly subset, copula price", SRC_T5S, "1.00", 1.00, 2),
    ("payout.forward.model_s.study_monthly", "forward, monthly subset, model S price", SRC_T5S, "1.06", 1.06, 2),
    ("payout.call_100.copula.study_monthly", "call struck at the copula's forward price, monthly subset, copula price", SRC_T5S, "1.16", 1.16, 2),
    ("payout.call_100.copula.study_monthly.lo", "its interval, lower", SRC_T5S, "0.87", 0.87, 2),
    ("payout.call_100.copula.study_monthly.hi", "its interval, upper", SRC_T5S, "1.51", 1.51, 2),
    ("payout.call_100.model_s.study_monthly", "call struck at the copula's forward price, monthly subset, model S price", SRC_T5S, "1.58", 1.58, 2),
    ("payout.call_100.model_s.study_monthly.lo", "its interval, lower", SRC_T5S, "1.16", 1.16, 2),
    ("payout.call_100.model_s.study_monthly.hi", "its interval, upper", SRC_T5S, "2.04", 2.04, 2),
    ("payout.call_050.copula.study_monthly", "call struck at 0.5 x the copula's forward price, monthly subset, copula price", SRC_T5S, "1.00", 1.00, 2),
    ("payout.call_050.model_s.study_monthly", "call struck at 0.5 x the copula's forward price, monthly subset, model S price", SRC_T5S, "1.12", 1.12, 2),
    ("payout.call_075.copula.study_monthly", "call struck at 0.75 x the copula's forward price, monthly subset, copula price", SRC_T5S, "1.02", 1.02, 2),
    ("payout.call_075.model_s.study_monthly", "call struck at 0.75 x the copula's forward price, monthly subset, model S price", SRC_T5S, "1.23", 1.23, 2),
    ("payout.call_125.copula.study_monthly", "call struck at 1.25 x the copula's forward price, monthly subset, copula price", SRC_T5S, "1.21", 1.21, 2),
    ("payout.call_125.model_s.study_monthly", "call struck at 1.25 x the copula's forward price, monthly subset, model S price", SRC_T5S, "2.02", 2.02, 2),
    ("payout.call_150.copula.study_monthly", "call struck at 1.5 x the copula's forward price, monthly subset, copula price", SRC_T5S, "1.36", 1.36, 2),
    ("payout.call_150.model_s.study_monthly", "call struck at 1.5 x the copula's forward price, monthly subset, model S price", SRC_T5S, "2.84", 2.84, 2),
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
#: Printed to one digit in a sentence that does not state the sample (``SRC_NOTE``): the two
#: readings are the monthly subset and the weekly sample.
UNSTATED = ("gap.hedged.copula.study_monthly", "gap.hedged.model_s.study_monthly")


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
    lc_cols = ["date", "status", "reason", "n_names_unscreened", "names_unscreened", "git_commit", "n_particles", "n_paths", "companion_paths",
               "P_D_copula", "EV_copula", "EQV", "K_100", *need, *lp.GATING_CHECKS, "check_names", "wing_binds", "idx_err_atm", "idx_err_atm_se",
               "idx_err_90", "idx_err_90_se", "sum_w_ER2_lc", "sum_w_ER2_lc_se", "sum_w_M", "E_Rbar2_lc", "E_Rbar2_lc_se", "M_B_listed"]  # fmt: skip
    inter = monthly.merge(priced[lc_cols], on="date", how="inner", suffixes=("", "_lcrow"))
    inter = inter.sort_values("date").reset_index(drop=True)
    # the LC rows' copies of the study's columns are the study's
    for a, b in (("P_D", "P_D_copula"), ("EV", "EV_copula"), ("EQV", "EQV_lcrow"), ("K_100", "K_100_lcrow")):  # fmt: skip
        gap = float((inter[a] - inter[b]).abs().max())
        if not gap < 1e-12:
            raise ValueError(f"LC rows: {b} differs from the study's {a} by up to {gap:.3g}")
    inter["flagged"] = inter["n_names_unscreened"].fillna(0) > 0
    date_inter = sorted(set(priced["date"]) & set(conv["date"]))
    on_dates = priced[priced["date"].isin(date_inter)]
    # the status under the current rule of scripts/lcm_price.py (the stored one may predate the
    # owner's decision 3 of 2026-10-09: the names' 2 % check is a diagnostic, not a gate)
    current = [lp.row_status({k: bool(r[k]) for k in lp.GATING_CHECKS}) for r in inter[list(lp.GATING_CHECKS)].to_dict("records")]  # fmt: skip
    inter["status_current"], inter["reason_current"] = [c[0] for c in current], [c[1] for c in current]  # fmt: skip
    inter["index_gate_ok"] = inter["check_index"].astype(bool)
    inter["names_dev"] = inter["sum_w_ER2_lc"] / inter["sum_w_M"] - 1.0
    inter["names_in_2pct"] = inter["names_dev"].abs() <= NAMES_TOL
    if not (inter["names_in_2pct"] == inter["check_names"].astype(bool)).all():
        raise ValueError("the names' 2 % diagnostic recomputed from sum_w_ER2_lc / sum_w_M differs from check_names")  # fmt: skip
    # the second-moment identities the decomposition of E[V] rests on
    for a, b, c in (("EV_lc", "sum_w_ER2_lc", "E_Rbar2_lc"), ("EQV", "sum_w_M", "M_B_listed")):
        gap = float((inter[a] - (inter[b] - inter[c])).abs().max())
        if not gap < 1e-12:
            raise ValueError(f"LC rows: {a} differs from {b} - {c} by up to {gap:.3g}")
    # the non-finite columns behind a failed check_no_nan: the float columns the check reads
    # (lcm_price: every float of the row but align_*, C_200*, Cfwd*, profile_*)
    read = [c for c in lc.columns if lc[c].dtype.kind == "f" and not c.startswith(("align_", "C_200", "Cfwd", "profile_"))]  # fmt: skip
    rows_nan = lc[lc["date"].isin(inter.loc[~inter["check_no_nan"].astype(bool), "date"])]
    no_nan_failed = [{"date": r["date"], "columns": [c for c in read if not np.isfinite(r[c])]} for r in rows_nan.sort_values("date").to_dict("records")]  # fmt: skip
    only_gate = inter[list(lp.GATING_CHECKS)].astype(bool)
    SAMPLE_LABEL["intersection_index_ok"] = f"the intersection without the {int((~inter['index_gate_ok']).sum())} dates that fail the index gate"  # fmt: skip
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
        "intersection_index_ok": inter[inter["index_gate_ok"]].reset_index(drop=True),
        "intersection_names_in": inter[inter["names_in_2pct"]].reset_index(drop=True),
        "intersection_names_out": inter[~inter["names_in_2pct"]].reset_index(drop=True),
        "meta": {
            "date_intersection_unflagged": int((on_dates["n_names_unscreened"].fillna(0) == 0).sum()),
            "status_current_counts": inter["status_current"].value_counts().to_dict(),
            "no_nan_failed": no_nan_failed, "no_nan_only": int((~only_gate["check_no_nan"] & only_gate["check_forward"] & only_gate["check_index"]).sum()),
            "forward_failed": int((~only_gate["check_forward"]).sum()),
            "index_failed": inter.loc[~inter["index_gate_ok"], ["date", "wing_binds", "idx_err_atm", "idx_err_atm_se", "idx_err_90", "idx_err_90_se", "check_no_nan", "check_forward", "flagged"]].to_dict("records"),
            "names_outside": int((~inter["names_in_2pct"]).sum()), "names_outside_and_gate": int((~inter["names_in_2pct"] & (inter["status_current"] == "check")).sum()),
            "stored_reasons": inter.loc[inter["status"] != "ok", "reason"].value_counts().to_dict(),
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


def boot_ratio(num: Any, den: Any, block: int, seed: int = SEED_RATIO) -> dict[str, float]:
    """``sum(num) / sum(den)`` with the study's 95 % interval (checked against the study's own
    function at the study's seed) and the standard deviation of the same resamples."""
    a = finite(num, den)
    n = len(a)
    s = a[block_index(n, max(1, min(block, n)), seed)].sum(axis=1)
    draws = s[:, 0] / s[:, 1]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    out = {"value": float(a[:, 0].sum() / a[:, 1].sum()), "lo": float(lo), "hi": float(hi), "se": float(draws.std(ddof=1))}  # fmt: skip
    if seed == SEED_RATIO:
        ref = st.bootstrap_ratio(num, den, block)
        if not np.allclose([out["value"], out["lo"], out["hi"]], ref, rtol=0, atol=1e-13):
            raise ValueError("boot_ratio does not reproduce volsto.studies.disp_stats.bootstrap_ratio")  # fmt: skip
    return out


def boot_mean(x: Any, block: int, seed: int = SEED_RATIO) -> dict[str, float]:
    """Mean with the block-bootstrap standard error and 95 % interval of the mean, on the
    resamples of :func:`boot_ratio`."""
    a = finite(x)[:, 0]
    n = len(a)
    draws = a[block_index(n, max(1, min(block, n)), seed)].mean(axis=1)
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"value": float(a.mean()), "lo": float(lo), "hi": float(hi), "se": float(draws.std(ddof=1))}  # fmt: skip


def seed_range(boot: Any) -> dict[str, float]:
    """The range of the two bounds of a bootstrap interval over ``SEEDS_RANGE`` (``boot`` maps a
    seed to a result of :func:`boot_ratio` or :func:`boot_mean`): how many digits of a bound
    are stable."""
    b = [boot(seed) for seed in SEEDS_RANGE]
    lo, hi = [x["lo"] for x in b], [x["hi"] for x in b]
    return {"lo_min": min(lo), "lo_max": max(lo), "hi_min": min(hi), "hi_max": max(hi)}


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
    index = block_index(len(a), block, SEED_SPLIT)
    mu = a[index].mean(axis=1)
    draws = split_of(*mu.T)
    # the model part less the copula's on the same trades and the same resamples
    c = finite(g[FWD["copula"]], g[EVCOL["copula"]], g["EQV"], g["D"], g["V"])
    point["model_part_minus_copula"] = point["model_part"] - split_of(*c.mean(axis=0))["model_part"]  # fmt: skip
    draws["model_part_minus_copula"] = draws["model_part"] - split_of(*c[index].mean(axis=1).T)["model_part"]  # fmt: skip
    out = {}
    for k, v in point.items():
        lo, hi = np.percentile(draws[k], [2.5, 97.5])
        out[k] = {"value": float(v), "lo": float(lo), "hi": float(hi), "se": float(np.std(draws[k], ddof=1))}  # fmt: skip
    out["n"] = {"value": float(len(a))}
    out["mean_price"] = {"value": float(a[:, 0].mean())}
    out["mean_payoff"] = {"value": float(a[:, 3].mean())}
    return out


#: The second moments behind E[V] = sum_w E[R_i^2] - E[Rbar^2]: (names, basket) of the listed
#: strips and of each model; the copula's and model S's names' moment is E[V] + E[Rbar^2].
MOMENTS = ("sum_w_M", "M_B_listed", "sum_w_ER2_lc", "E_Rbar2_lc", "EV", "M_B", "EV_S", "M_B_S")


def moments_of(nL: Any, bL: Any, n_lc: Any, b_lc: Any, ev_c: Any, b_c: Any, ev_s: Any, b_s: Any) -> dict[str, Any]:  # fmt: skip
    """The decomposition of mean E_LC[V] - mean EQV from the eight means of ``MOMENTS`` (scalars
    or arrays): ``(names_LC - names_listed) + (basket_listed - basket_LC)``."""
    eqv, excess = nL - bL, (n_lc - b_lc) - (nL - bL)
    return {
        "excess_ev_over_eqv": excess / eqv,
        "share_names": (n_lc - nL) / excess,
        "share_basket": (bL - b_lc) / excess,
        "names_ratio_lc": n_lc / nL,
        "basket_ratio_lc": b_lc / bL,
        "names_ratio_copula": (ev_c + b_c) / nL,
        "basket_ratio_copula": b_c / bL,
        "names_ratio_model_s": (ev_s + b_s) / nL,
        "basket_ratio_model_s": b_s / bL,
        "dispersion_lc_over_listed": np.sqrt((n_lc - b_lc) / eqv),
        "dispersion_lc_names_at_listed": np.sqrt((nL - b_lc) / eqv),
    }


def moment_rows(S: dict[str, Any]) -> pd.DataFrame:
    """One row per sample of the intersection: the decomposition of :func:`moments_of`, each
    number with the standard deviation and 95 % interval of the D4 bootstrap of :func:`split`
    and, where an LC Monte Carlo moment enters alone, its Monte Carlo bound."""
    rows = []
    for sample in ("intersection", "intersection_unflagged", "intersection_index_ok", "intersection_names_in", "intersection_names_out"):  # fmt: skip
        g = S[sample][S[sample]["strip_ok"]]
        a = finite(*(g[c] for c in MOMENTS))
        point = moments_of(*a.mean(axis=0))
        draws = moments_of(*a[block_index(len(a), BLOCK_MONTHLY, SEED_SPLIT)].mean(axis=1).T)
        row: dict[str, Any] = {"sample": sample, "n_trades": len(a), "boot_block": BLOCK_MONTHLY, "mean_EV_lc": float(g["EV_lc"].mean()), "mean_EQV": float(g["EQV"].mean()),
                               "mean_names_listed": float(a[:, 0].mean()), "mean_basket_listed": float(a[:, 1].mean()), "mean_names_lc": float(a[:, 2].mean()), "mean_basket_lc": float(a[:, 3].mean())}  # fmt: skip
        for k, v in point.items():
            lo, hi = np.percentile(draws[k], [2.5, 97.5])
            row[k], row[f"{k}_se_boot"], row[f"{k}_ci95_lo"], row[f"{k}_ci95_hi"] = float(v), float(np.std(draws[k], ddof=1)), float(lo), float(hi)  # fmt: skip
        row["names_ratio_lc_mc_indep"], row["names_ratio_lc_mc_bound"] = (x / a[:, 0].mean() for x in mc_mean(g["sum_w_ER2_lc_se"]))  # fmt: skip
        row["basket_ratio_lc_mc_indep"], row["basket_ratio_lc_mc_bound"] = (x / a[:, 1].mean() for x in mc_mean(g["E_Rbar2_lc_se"]))  # fmt: skip
        rows.append(row)
    return pd.DataFrame(rows)


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
        ("intersection_index_ok", BLOCK_MONTHLY, LAG_MONTHLY),
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
                monthly = lag == LAG_MONTHLY
                hh_more = st.mean_se(pnl, LAG_MONTHLY_MORE)[1] if monthly else float("nan")
                sr = seed_range(lambda seed, pnl=pnl, block=block: boot_mean(pnl, block, seed)) if monthly else dict.fromkeys(("lo_min", "lo_max", "hi_min", "hi_max"), float("nan"))  # fmt: skip
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
                    "se_hansen_hodrick_lag3_pct": 100 * hh_more, "t_ratio": b["value"] / hh, "t_ratio_lag3": b["value"] / hh_more,
                    "ci95_lo_pct_seed_min": 100 * sr["lo_min"], "ci95_lo_pct_seed_max": 100 * sr["lo_max"],
                    "ci95_hi_pct_seed_min": 100 * sr["hi_min"], "ci95_hi_pct_seed_max": 100 * sr["hi_max"],
                })  # fmt: skip
    return pd.DataFrame(rows)


def payout_rows(S: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for sample, block in (
        ("weekly", BLOCK_WEEKLY),
        ("study_monthly", BLOCK_MONTHLY),
        ("intersection", BLOCK_MONTHLY),
        ("intersection_unflagged", BLOCK_MONTHLY),
        ("intersection_index_ok", BLOCK_MONTHLY),
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
                nan4 = dict.fromkeys(("lo_min", "lo_max", "hi_min", "hi_max"), float("nan"))
                ranged = sample != "weekly" and product in ("forward", "call_100")
                sr = seed_range(lambda seed, n=g[pay], d=g[price[model]], block=block: boot_ratio(n, d, block, seed)) if ranged else nan4  # fmt: skip
                # the call's strike against the model's own forward price (pooled over trades)
                own = boot_ratio(g[f"K_{product[-3:]}"], g[FWD[model]], block) if product != "forward" else dict.fromkeys(("value", "se", "lo", "hi"), float("nan"))  # fmt: skip
                fwd_mc = mc_mean(g["ED_lc_se"])[1] / float(g["ED_lc"].mean()) if model == "lc" else 0.0  # fmt: skip
                rows.append({
                    "sample": sample, "n_trades": len(g), "product": product, "model": model,
                    "payout_per_1_of_premium": b["value"], "ci95_lo": b["lo"], "ci95_hi": b["hi"], "se_boot": b["se"], "boot_block": block,
                    "mc_se_indep": b["value"] * float(np.sqrt((se**2).sum())) / total, "mc_se_bound": b["value"] * float(se.sum()) / total,
                    "mean_payoff_pct": 100 * g[pay].mean(), "mean_premium_pct": 100 * g[price[model]].mean(),
                    "mean_premium_over_copula": float(g[price[model]].sum() / g[price["copula"]].sum()),
                    "ci95_lo_seed_min": sr["lo_min"], "ci95_lo_seed_max": sr["lo_max"], "ci95_hi_seed_min": sr["hi_min"], "ci95_hi_seed_max": sr["hi_max"],
                    "strike_over_own_forward": own["value"], "strike_over_own_forward_se_boot": own["se"], "strike_over_own_forward_mc_bound": own["value"] * fwd_mc,
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
        ("intersection_index_ok", BLOCK_MONTHLY),
        ("intersection_names_in", BLOCK_MONTHLY),
        ("intersection_names_out", BLOCK_MONTHLY),
    ):
        g = S[sample]
        for model in MODELS:
            if FWD[model] not in g:
                continue
            r = split(g, model, block)
            row: dict[str, Any] = {"sample": sample, "n_trades": int(r["n"]["value"]), "model": model, "boot_block": block,
                                   "mean_price_pct": 100 * r["mean_price"]["value"], "mean_payoff_pct": 100 * r["mean_payoff"]["value"]}  # fmt: skip
            for k in (*(f for f, _ in FACTORS), "kappa_model", "kappa_realised", "model_part_minus_copula"):  # fmt: skip
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
            # the difference of two model parts: LC's error and the copula's (the study's P_D_se)
            row["model_part_minus_copula_mc_bound"] = 0.0
            if model == "lc":
                gs = g[g["strip_ok"]]
                cop = split(g, "copula", block)["model_part"]["value"]
                row["model_part_minus_copula_mc_bound"] = row["model_part_mc_bound"] + cop * mc_mean(gs["P_D_se"])[1] / gs["P_D"].mean()  # fmt: skip
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
    # the second reading of the numbers printed without a stated sample: the weekly sample.  Model
    # S has no weekly price: the copula's weekly mean plus the monthly subset's mean price
    # difference (what the study's sentence adds to the copula's number)
    q = gap[(gap["sample"] == "study_monthly") & (gap["model"] == "model_s") & (gap["structure"] == "hedged")]  # fmt: skip
    shift_s = float(q["mean_copula_minus_model_price_pct"].iloc[0])
    other = {
        "gap.hedged.copula.study_monthly": (mine["gap.hedged.copula.weekly"], "the 1,010 weekly trades"),
        "gap.hedged.model_s.study_monthly": (mine["gap.hedged.copula.weekly"] + shift_s, f"the copula's mean on the 1,010 weekly trades ({mine['gap.hedged.copula.weekly']:.4f}) "
                                             f"+ the mean copula minus model S forward price of the 217 monthly trades ({shift_s:.4f}); model S has no price on the weekly trades"),
    }  # fmt: skip
    if set(other) != set(UNSTATED):
        raise ValueError("the second readings do not cover UNSTATED")
    rows = []
    for key, what, source, text, value, digits in PRINTED:
        v = mine[key]
        tol = 0.5 * 10.0**-digits * 1e-6
        v2, what2 = other.get(key, (float("nan"), ""))
        rows.append({"key": key, "bullet": key.split(".")[0], "quantity": what, "study_printed": text, "this_script": v,
                     "digits": digits, "match": abs(round(v, digits) - value) < tol, "source": source, "sample_stated": key not in other,
                     "this_script_sample": sample_of_key(key), "other_reading": v2, "other_reading_what": what2,
                     "other_reading_match": bool(abs(round(v2, digits) - value) < tol) if key in other else None})  # fmt: skip
    return pd.DataFrame(rows)


def sample_of_key(key: str) -> str:
    """The sample of a reproduction key (``...<model>.<sample>[.lo|.hi|.se]``)."""
    return next(p for p in key.split(".") if p in ("weekly", "weekly_is", "weekly_oos", "study_monthly"))  # fmt: skip


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
    "intersection_index_ok": "the intersection without the dates that fail the index gate",  # the count is set in samples()
    "intersection_names_in": "the intersection, trades inside the names' 2 % diagnostic",
    "intersection_names_out": "the intersection, trades outside the names' 2 % diagnostic",
}
#: What a sample of the intersection adds to the notes and the source of a copula or model S
#: record: the prices are the study's, the sample is the LC table's.
NOTE_LC_SAMPLE = "the sample is defined by the LC table (its priced dates and, where it applies, its unscreened flag and its checks)"
MODEL_FREE = (
    "model-free: no model price in it, the same for the three models (the id keeps 'copula')"
)
DEF_MOMENT = {
    "excess_ev_over_eqv": "mean EV_lc / mean EQV - 1, with EV_lc = sum_w E_LC[R_i^2] - E_LC[Rbar^2] and EQV = sum_w M_i - M_B (the listed strips), pooled over trades",
    "share_names": "(mean sum_w E_LC[R_i^2] - mean sum_w M_i) / (mean EV_lc - mean EQV): the share of the excess that is the names' second moment under the LC Monte Carlo above the names' listed strips",
    "share_basket": "(mean M_B listed - mean E_LC[Rbar^2]) / (mean EV_lc - mean EQV): the share of the excess that is the basket's second moment under LC below the listed index strip; 1 - the names' share",
    "names_ratio_lc": "mean sum_w E_LC[R_i^2] / mean sum_w M_i (sum_w_ER2_lc over sum_w_M), pooled over trades",
    "basket_ratio_lc": "mean E_LC[Rbar^2] / mean M_B listed (E_Rbar2_lc over M_B_listed), pooled over trades",
    "names_ratio_copula": "mean (EV + M_B of the study's entries: the copula's sum_w E[R_i^2]) / mean sum_w M_i, pooled over trades",
    "basket_ratio_copula": "mean M_B of the study's entries (the copula's E[Rbar^2]) / mean M_B listed, pooled over trades",
    "names_ratio_model_s": "mean (EV_S + M_B_S: model S's sum_w E[R_i^2]) / mean sum_w M_i, pooled over trades",
    "basket_ratio_model_s": "mean M_B_S (model S's E[Rbar^2]) / mean M_B listed, pooled over trades",
    "dispersion_lc_over_listed": "sqrt(mean EV_lc / mean EQV): the split's 'dispersion priced, model over listed' of LC",
    "dispersion_lc_names_at_listed": "sqrt((mean sum_w M_i - mean E_LC[Rbar^2]) / mean EQV): LC's 'dispersion priced, model over listed' with the names' second moment set to the listed strips",
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
            unit: str, definition: str, notes: str = "", sample: str = "") -> None:  # fmt: skip
        lc = model == "lc"
        if lc and not ok[bullet]:
            value, se, notes = None, None, "pending: a printed number of the study behind this bullet is not reproduced (tables/C_arith_reproduction.csv)"  # fmt: skip
        # a copula or model S number on a sample of the intersection: the study's prices on a
        # sample the LC table defines
        on_lc_sample = not lc and sample.startswith("intersection")
        if on_lc_sample:
            notes = f"{notes}; {NOTE_LC_SAMPLE}" if notes else NOTE_LC_SAMPLE
        records.append(pc.record(
            f"C.arith.{id_}", SECTION, quantity, value, se, tenor=TENOR, unit=unit, definition=definition,
            budget=pc.BUDGETS["development" if lc else "study"], commit=meta["commit"] if lc else "the study's tables (no commit column)",
            source=f"tables/{table}.csv <- {src_lc if lc or on_lc_sample else SRC_STUDY}", n=n, notes=notes,
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
        ("date_intersection_unflagged", "priced LC dates that are converged model S dates, without the flagged dates", meta["date_intersection_unflagged"]),
        ("trades.intersection.status_stored_ok", "trades of the intersection with stored status ok (the rule before decision 3)", meta["status_counts"].get("ok", 0)),
        ("trades.intersection.status_stored_check", "trades of the intersection with stored status check (the rule before decision 3)", meta["status_counts"].get("check", 0)),
        ("trades.intersection.status_current_ok", "trades of the intersection with status ok under the current rule (lcm_price.row_status)", meta["status_current_counts"].get("ok", 0)),
        ("trades.intersection.status_current_check", "trades of the intersection with status check under the current rule (lcm_price.row_status)", meta["status_current_counts"].get("check", 0)),
        ("trades.intersection.check_no_nan_failed", "trades of the intersection that fail check_no_nan", len(meta["no_nan_failed"])),
        ("trades.intersection.check_forward_failed", "trades of the intersection that fail check_forward", meta["forward_failed"]),
        ("trades.intersection.check_index_failed", "trades of the intersection that fail check_index (the index gate)", len(meta["index_failed"])),
        ("trades.intersection.names_outside_2pct", "trades of the intersection with the names' diagnostic outside 2 % (not a gate)", meta["names_outside"]),
        ("trades.intersection_index_ok", "trades of the intersection without the dates that fail the index gate", len(S["intersection_index_ok"])),
        ("trades.intersection_names_in", "trades of the intersection inside the names' 2 % diagnostic", len(S["intersection_names_in"])),
        ("trades.intersection_names_out", "trades of the intersection outside the names' 2 % diagnostic", len(S["intersection_names_out"])),
    ]  # fmt: skip
    for id_, quantity, value in counts:
        add(f"sample.{id_}", quantity, value, None, model="lc" if "lc" in id_ or "intersection" in id_ else "copula", bullet="none", table="C_arith_sample",
            n=int(value), unit="count", definition="a count of entry dates (one trade per entry date)", notes="a count, no standard error")  # fmt: skip
    for r in meta["index_failed"]:
        for k, what in (("idx_err_90", "at the 90 % strike"), ("idx_err_atm", "at the money")):
            add(f"sample.index_gate.{k}.{r['date']}", f"index smile error {what} on {r['date']}, a date that FAILS the index gate", r[k], r[f"{k}_se"], model="lc", bullet="none", table="C_arith_trades",
                n=1, unit="vol points", definition=f"LC Monte Carlo implied volatility of the index minus the listed one {what} at the tenor ({k} of the LC table); the gate: both within 0.15 vol points unless the wing binds",
                notes=f"se = the Monte Carlo standard error of the row ({k}_se); wing_binds = {bool(r['wing_binds'])}; the date is in every summary of the part except the rows 'without the dates that fail the index gate'")  # fmt: skip

    # -- reproduction
    for r in rep.to_dict("records"):
        unstated = not r["sample_stated"]
        add(f"repro.{r['key']}", f"reproduction: {r['quantity']}" + ("; value on the 217 trades of the monthly subset" if unstated else ""), r["this_script"], None, model="copula", bullet="none", table="C_arith_reproduction",
            n=len(S[r["this_script_sample"]]), unit="% of notional" if r["bullet"] == "gap" else "ratio",
            definition="the study's printed number recomputed by this script from the study's tables on the study's own sample",
            notes=f"study printed {r['study_printed']} ({r['source']}); match to the printed digits: {r['match']}"
                  + (f"; the study's sentence does not state the sample: the other reading, {r['other_reading_what']}, is {r['other_reading']:.4f} (match to the printed digit: {r['other_reading_match']})" if unstated else ""))  # fmt: skip
        if unstated:
            add(f"repro.{r['key']}.other_reading", f"reproduction: {r['quantity']}; value on the other reading, {r['other_reading_what']}", r["other_reading"], None, model="copula", bullet="none",
                table="C_arith_reproduction", n=len(S["weekly"]), unit="% of notional", definition="the study's printed number recomputed by this script from the study's tables on the weekly sample (the second reading of a sentence that states no sample)",
                notes=f"study printed {r['study_printed']} ({r['source']}); match to the printed digit: {r['other_reading_match']}; no standard error attached here (the weekly mean's Hansen-Hodrick s.e. is the record C.arith.repro.gap.hedged.copula.weekly.se)")  # fmt: skip

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
        common = dict(model=r["model"], bullet="gap", table="C_arith_gap", n=r["n_trades"], unit="% of notional", sample=r["sample"],
                      definition=DEF_GAP.format(how=how, model=MODEL_LABEL[r["model"]]))  # fmt: skip
        where = f"the {r['structure']} gap at the {MODEL_LABEL[r['model']]} price, {SAMPLE_LABEL[r['sample']]}"
        add(base, f"{r['structure']} gap at the {MODEL_LABEL[r['model']]} price, mean P&L per trade, {SAMPLE_LABEL[r['sample']]}", r["mean_pnl_pct"], r["se_hansen_hodrick_pct"], **common,
            notes=f"se = sampling s.e. over trades: Hansen-Hodrick (the study's s.e. of a mean P&L, disp_stats.mean_se) with {r['hh_lags']} lags for monthly entries of three-month trades; "
                  f"block-bootstrap s.e. of the mean (circular blocks of {r['boot_block']} monthly entries, {N_RESAMPLES} resamples, seed {SEED_RATIO}): {r['se_boot_pct']:.3f}, "
                  f"95 % interval {interval(r['ci95_lo_pct'], r['ci95_hi_pct'])}; the study printed no s.e. for the gap on the monthly subset; share of trades with a positive P&L {r['hit_rate']:.3f}{mc}")  # fmt: skip
        for end in ("lo", "hi"):
            add(f"{base}.ci95_{end}", f"95 % block-bootstrap interval ({'lower' if end == 'lo' else 'upper'}) of the mean P&L of {where}", r[f"ci95_{end}_pct"], None,
                **{**common, "definition": f"the {'2.5th' if end == 'lo' else '97.5th'} percentile of the block-bootstrap distribution of the " + common["definition"]},
                notes=f"a percentile of the block bootstrap of the mean (circular blocks of {r['boot_block']} monthly entries, {N_RESAMPLES} resamples, seed {SEED_RATIO}); not itself an estimate with a standard error; "
                      f"over the seeds {SEEDS_RANGE[0]} to {SEEDS_RANGE[-1]} this bound runs from {r[f'ci95_{end}_pct_seed_min']:.3f} to {r[f'ci95_{end}_pct_seed_max']:.3f}")  # fmt: skip
        add(f"{base}.se_hh_lag{LAG_MONTHLY_MORE}", f"Hansen-Hodrick standard error with {LAG_MONTHLY_MORE} lags of the mean P&L of {where}", r["se_hansen_hodrick_lag3_pct"], None,
            **{**common, "definition": f"Hansen-Hodrick standard error (disp_stats.mean_se, {LAG_MONTHLY_MORE} lags) of the " + common["definition"]},
            notes=f"a standard error (disp_stats.mean_se with {LAG_MONTHLY_MORE} lags instead of {r['hh_lags']}), no standard error of its own; some pairs of trades three monthly entries apart overlap by a few days")  # fmt: skip
        add(f"{base}.t_ratio", f"t-ratio (mean over its Hansen-Hodrick s.e., {r['hh_lags']} lags) of the mean P&L of {where}", r["t_ratio"], None,
            **{**common, "unit": "ratio", "definition": f"the mean divided by its Hansen-Hodrick standard error ({r['hh_lags']} lags); the mean: " + common["definition"]},
            notes=f"mean / s.e.; no standard error of its own; with {LAG_MONTHLY_MORE} lags {r['t_ratio_lag3']:.2f}")  # fmt: skip
        if r["model"] == "lc":
            add(base + ".mc_bound", f"Monte Carlo error bound of the {r['structure']} gap at the LC price, {SAMPLE_LABEL[r['sample']]}", r["mc_se_bound_pct"], None, **common,
                notes=f"mean over trades of ED_lc_se, valid whatever the dependence between dates (same seeds on every date); with independent dates {sig(r['mc_se_indep_pct'])}; not itself an estimate with a standard error")  # fmt: skip
        if r["structure"] == "held" and r["model"] != "copula":
            add(f"gap.price_shift.{r['model']}.{r['sample']}", f"copula forward price minus {MODEL_LABEL[r['model']]} forward price, mean per trade, {SAMPLE_LABEL[r['sample']]}",
                r["mean_copula_minus_model_price_pct"], r["copula_minus_model_mc_bound_pct"] if r["model"] == "lc" else None, model=r["model"], bullet="gap", table="C_arith_gap", n=r["n_trades"], unit="% of notional", sample=r["sample"],
                definition=f"mean over trades of P_D - {FWD[r['model']]}: what each P&L of the Palladium forward and of the gap gains at the {MODEL_LABEL[r['model']]} price",
                notes=(f"se = the Monte Carlo bound of the mean (mean over trades of ED_lc_se + the study's P_D_se), valid whatever the dependence between dates (every date is priced on the same seeds); with independent dates it would be {sig(r['copula_minus_model_mc_indep_pct'])} "
                       "(not supported: the dates share their seeds); a mean of entry prices, no sampling s.e. attached"
                       if r["model"] == "lc" else "model S's table carries no standard error; a mean of entry prices, no sampling s.e. attached"))  # fmt: skip
        if r["structure"] == "held":
            se_price = {"model_s": None, "copula": r["forward_price_mc_indep_pct"], "lc": r["forward_price_mc_bound_pct"]}[r["model"]]  # fmt: skip
            note_price = {
                "model_s": "model S's table carries no standard error",
                "copula": f"se = Monte Carlo error of the mean with independent dates (the study P_D_se); at most {sig(r['forward_price_mc_bound_pct'])} whatever the dependence between dates",
                "lc": f"se = the Monte Carlo bound of the mean (mean over trades of ED_lc_se), valid whatever the dependence between dates (every date is priced on the same seeds); with independent dates it would be {sig(r['forward_price_mc_indep_pct'])} (not supported: the dates share their seeds)",
            }[r["model"]]  # fmt: skip
            add(f"gap.forward_price.{r['model']}.{r['sample']}", f"mean {MODEL_LABEL[r['model']]} forward price, {SAMPLE_LABEL[r['sample']]}", r["mean_forward_price_pct"], se_price,
                model=r["model"], bullet="gap", table="C_arith_gap", n=r["n_trades"], unit="% of notional", sample=r["sample"], definition=f"mean over trades of {FWD[r['model']]}",
                notes=note_price + "; a mean of entry prices, no sampling s.e. attached")  # fmt: skip

    # -- payout per 1 of premium
    for r in payout.to_dict("records"):
        if r["sample"] == "weekly":
            continue
        name = "the Palladium forward" if r["product"] == "forward" else f"the Palladium call struck at {int(r['product'][-3:]) / 100:g} x the copula forward's price"  # fmt: skip
        mc = f"; Monte Carlo error of the LC prices in this ratio: {sig(r['mc_se_indep'])} with independent dates, at most {sig(r['mc_se_bound'])}" if r["model"] == "lc" else ""  # fmt: skip
        base = f"payout.{r['product']}.{r['model']}.{r['sample']}"
        common = dict(model=r["model"], bullet="payout", table="C_arith_payout", n=r["n_trades"], unit="ratio", sample=r["sample"],
                      definition=DEF_PAYOUT.format(product=name, model=MODEL_LABEL[r["model"]]))  # fmt: skip
        add(base, f"payout per 1 of premium of {name} at the {MODEL_LABEL[r['model']]} price, {SAMPLE_LABEL[r['sample']]}", r["payout_per_1_of_premium"], r["se_boot"], **common,
            notes=f"the study's 95 % interval {interval(r['ci95_lo'], r['ci95_hi'])} (disp_stats.bootstrap_ratio: circular blocks of {r['boot_block']} monthly entries, {N_RESAMPLES} resamples, seed {SEED_RATIO}); "
                  f"se = standard deviation of the same resamples; mean premium {r['mean_premium_pct']:.3f} % of notional{mc}")  # fmt: skip
        for end in ("lo", "hi"):
            add(f"{base}.ci95_{end}", f"95 % interval ({'lower' if end == 'lo' else 'upper'}) of the payout per 1 of premium of {name} at the {MODEL_LABEL[r['model']]} price, {SAMPLE_LABEL[r['sample']]}",
                r[f"ci95_{end}"], None, **common, notes="a percentile of the study's block bootstrap; not itself an estimate with a standard error"
                + (f"; over the seeds {SEEDS_RANGE[0]} to {SEEDS_RANGE[-1]} (the study's is {SEED_RATIO}) this bound runs from {r[f'ci95_{end}_seed_min']:.3f} to {r[f'ci95_{end}_seed_max']:.3f}" if np.isfinite(r[f"ci95_{end}_seed_min"]) else ""))  # fmt: skip
        if r["model"] == "lc":
            add(base + ".mc_bound", f"Monte Carlo error bound of the payout per 1 of premium of {name} at the LC price, {SAMPLE_LABEL[r['sample']]}", r["mc_se_bound"], None, **common,
                notes=f"ratio x sum of the per-date standard errors over the sum of prices, valid whatever the dependence between dates; with independent dates {sig(r['mc_se_indep'])}")  # fmt: skip
        if r["product"] == "call_100" and r["model"] != "copula":
            add(f"payout.strike_over_own_forward.{r['model']}.{r['sample']}", f"strike of the call (the copula's forward price) above the {MODEL_LABEL[r['model']]} forward price, {SAMPLE_LABEL[r['sample']]}",
                100 * (r["strike_over_own_forward"] - 1.0), 100 * r["strike_over_own_forward_se_boot"],
                **{**common, "unit": "%", "definition": f"100 x (sum over trades of K_100 / sum over trades of {FWD[r['model']]} - 1): K_100 = the copula's forward price P_D of the entry date, the cash strike of the call for the three models"},
                notes=f"se = sampling s.e. over trades (standard deviation of the block bootstrap of the ratio: circular blocks of {r['boot_block']} monthly entries, {N_RESAMPLES} resamples, seed {SEED_RATIO})"
                      + (f"; Monte Carlo error of the LC forward prices in it, at most {sig(100 * r['strike_over_own_forward_mc_bound'])}" if r["model"] == "lc" else "; model S's table carries no standard error")
                      + "; the package carries no call struck at the model's own forward")  # fmt: skip

    # -- the split
    for r in sp.to_dict("records"):
        weekly = r["sample"].startswith("weekly")
        sub = r["sample"] in ("intersection_names_in", "intersection_names_out")
        for k, label in (
            *FACTORS,
            ("kappa_model", "kappa of the model"),
            ("kappa_realised", "kappa realised"),
            ("model_part_minus_copula", "model part minus the copula's model part"),
        ):
            if weekly and k not in ("kappa_model", "kappa_realised"):
                continue  # the weekly factors are reproduction records; the two kappas had none
            if k in ("listed_part", "kappa_realised") and r["model"] != "copula":
                continue  # no model in it: one record per sample
            if k == "model_part_minus_copula" and r["model"] == "copula":
                continue
            free = k in ("listed_part", "kappa_realised")
            mc = f"; Monte Carlo error of the LC prices: {sig(r[f'{k}_mc_indep'])} with independent dates, at most {sig(r[f'{k}_mc_bound'])}" if r["model"] == "lc" and k != "model_part_minus_copula" else ""  # fmt: skip
            if r["model"] == "lc" and k == "model_part_minus_copula":
                mc = f"; Monte Carlo error of the LC prices and of the study's copula prices (P_D_se), at most {sig(r[f'{k}_mc_bound'])}"  # fmt: skip
            base = f"split.{k}.{r['model']}.{r['sample']}"
            definition = "the model part of {model} minus the model part of the copula on the same trades" if k == "model_part_minus_copula" else DEF_SPLIT[k]  # fmt: skip
            common = dict(model=r["model"], bullet="split", table="C_arith_split", n=r["n_trades"], unit="ratio", sample=r["sample"], definition=definition.format(model=MODEL_LABEL[r["model"]]))  # fmt: skip
            add(base, f"price-over-payoff split, {label}, {MODEL_FREE.split(':')[0] if free else MODEL_LABEL[r['model']]}, {SAMPLE_LABEL[r['sample']]}", r[k], r[f"{k}_se_boot"], **common,
                notes=f"se = sampling s.e. over trades: the bootstrap of the study's D4 (circular blocks, {N_RESAMPLES} resamples, seed {SEED_SPLIT}) with blocks of {r['boot_block']} {'weekly' if weekly else 'monthly'} entries"
                      + (" (blocks of consecutive trades of the sub-sample, which are not consecutive months)" if sub else "")
                      + f", 95 % interval {interval(r[f'{k}_ci95_lo'], r[f'{k}_ci95_hi'])}{'' if weekly else '; the study printed no interval on the monthly subset'}{mc}"
                      + (f"; {MODEL_FREE}" if free else "")
                      + ("; price over the root of the model's own E[V], as in the final report (its sidenote prints 0.729 and 0.743 on the 1,010 trades); not the kappa_Q of the study's T4_model_S (price over the root of EQV)" if k == "kappa_model" else ""))  # fmt: skip
            if k == "model_part_minus_copula":
                continue
            if r["model"] == "lc":
                add(base + ".mc_bound", f"Monte Carlo error bound of the split's {label}, LC, {SAMPLE_LABEL[r['sample']]}", r[f"{k}_mc_bound"], None, **common,
                    notes=f"from the means of ED_lc_se and EV_lc_se, valid whatever the dependence between dates and between the two estimates; with independent dates and estimates {sig(r[f'{k}_mc_indep'])}")  # fmt: skip

    # -- the second moments behind LC's E[V] against the listed strips
    mom = moment_rows(S)
    for r in mom.to_dict("records"):
        for k, definition in DEF_MOMENT.items():
            model = next((m for m in ("copula", "model_s") if k.endswith(m)), "lc")
            mcb = r.get(f"{k}_mc_bound")
            add(f"split.second_moment.{k}.{r['sample']}", f"second moments behind E[V]: {k.replace('_', ' ')}, {SAMPLE_LABEL[r['sample']]}", r[k], r[f"{k}_se_boot"], model=model, bullet="split",
                table="C_arith_second_moments", n=r["n_trades"], unit="ratio", sample=r["sample"], definition=definition,
                notes=f"se = sampling s.e. over trades: the bootstrap of the study's D4 (circular blocks of {r['boot_block']} consecutive trades of the sample, {N_RESAMPLES} resamples, seed {SEED_SPLIT}), 95 % interval {interval(r[f'{k}_ci95_lo'], r[f'{k}_ci95_hi'])}"
                      + (f"; Monte Carlo error of the LC second moment in it: {sig(r[f'{k}_mc_indep'])} with independent dates, at most {sig(mcb)} (mean of the rows' standard errors over the mean listed moment)" if mcb is not None else
                         "; Monte Carlo error of the LC moments not propagated to this number (that of the two LC ratios is in their records)" if model == "lc" else "; the study's tables carry no standard error of these moments"))  # fmt: skip

    md = markdown(S, gap, payout, sp, rep, ok, mom)
    sample_table = pd.DataFrame([{"id": i, "quantity": q, "value": v} for i, q, v in counts])
    return records, md, {"C_arith_sample": sample_table, "C_arith_reproduction": rep, "C_arith_gap": gap, "C_arith_payout": payout, "C_arith_split": sp,
                         "C_arith_second_moments": mom}  # fmt: skip


def markdown(S: dict[str, Any], gap: pd.DataFrame, payout: pd.DataFrame, sp: pd.DataFrame, rep: pd.DataFrame, ok: dict[str, bool], mom: pd.DataFrame) -> str:  # fmt: skip
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
        "Files: `tables/C_arith_sample.csv`, `C_arith_reproduction.csv`, `C_arith_gap.csv`, `C_arith_payout.csv`, `C_arith_split.csv`, `C_arith_second_moments.csv` (long tables with every standard error, interval and Monte Carlo error), "
        "`C_arith_trades.csv` (one row per trade of the intersection, with its checks).",
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
    cur = meta["status_current_counts"]
    stored_reasons = "; ".join(f"'{k}' on {v}" for k, v in meta["stored_reasons"].items())
    nan_cols = ", ".join(f"`{c}`" for c in sorted({c for r in meta["no_nan_failed"] for c in r["columns"]}))  # fmt: skip
    idx_failed = "; ".join(
        f"{r['date']} ({r['idx_err_90']:+.2f} ± {r['idx_err_90_se']:.2f} vol points at the 90 % strike, {r['idx_err_atm']:+.2f} ± {r['idx_err_atm_se']:.2f} at the money)" for r in meta["index_failed"]
    )  # fmt: skip
    out += [
        "**Table C_arith_sample.** The trades.",
        "",
        "| sample | entry dates | trades |",
        "|---|---|---|",
        f"| LC variant development pass (`{meta['lc_table']}`, commit {meta['commit']}, {pc.BUDGETS['development']}): priced | {meta['lc_priced']} of {meta['lc_rows']} | |",
        f"| model S: converged | {meta['ms_converged']} of {meta['ms_rows']} | |",
        f"| priced LC dates that are converged model S dates | {meta['date_intersection']} | {n_i} |",
        f"| the same without the flagged dates (a name kept unscreened) | | {n_u} |",
        f"| the same without the dates that fail the index gate (flagged dates kept) | | {len(S['intersection_index_ok'])} |",
        f"| the same, inside / outside the names' 2 % diagnostic (not a gate) | | {len(S['intersection_names_in'])} / {len(S['intersection_names_out'])} |",
        f"| the study's monthly subset (model S converged, the study's own sample) | | {n_m} |",
        f"| the study's weekly sample | | {len(S['weekly'])} |",
        "",
        f"A trade is one entry date of the study at 3m on basket B1 at unit notional with its outcome (the gap, the forward and the call of that date); the LC pass failed on {failed}; "
        f"the intersection is the study's monthly subset less {', '.join(meta['study_monthly_not_priced'])} (not priced by the LC pass); "
        f"{', '.join(meta['dates_without_trade']) or 'no date'} is priced by both models but is not a trade of the study (a member with no price move over the window, `stuck_names`: left out by the study); "
        f"the {len(meta['flagged'])} flagged trades are {flagged}. "
        f"{TODAY} is not in any sample: it has no outcome yet and model S did not converge on it "
        f"(LC priced it: {meta['lc_priced_today']}; model S converged: {meta['ms_converged_today']}).",
        "",
        f"Against part C_history: section C's history summaries use {meta['date_intersection']} dates on the same intersection of dates (they keep {', '.join(meta['dates_without_trade']) or 'no other date'}); "
        f"this part has {n_i} trades because the study drops that date (a stuck member); without the flagged dates the two counts are {meta['date_intersection_unflagged']} dates and {n_u} trades.",
        "",
        f"Status of the LC rows on the {n_i} trades. Stored in the table: {', '.join(f'{k} {v}' for k, v in sorted(meta['status_counts'].items()))} "
        f"(stored reasons: {stored_reasons}). The stored status was written under the rule before the owner's decision 3 of 2026-10-09 "
        "(the names' 2 % check is a diagnostic, not a gate). Under the current rule of `scripts/lcm_price.py` (`row_status`: `check` when one of `check_no_nan`, `check_forward`, `check_index` fails), "
        f"recomputed here from the rows' check columns: {', '.join(f'{k} {v}' for k, v in sorted(cur.items(), reverse=True))}. Of the {cur.get('check', 0)} `check` trades: "
        f"{meta['no_nan_only']} fail `check_no_nan` and no other gate ({', '.join(r['date'] for r in meta['no_nan_failed'])}); the only non-finite column the check reads on these rows is {nan_cols} "
        "(the standard error of the index smile error at +2.5 standard deviations, a diagnostic column); no price column is among them, and every price used in this part is finite. "
        f"{len(meta['index_failed'])} FAIL the index gate `check_index` (the index smile within 0.15 vol points at the money and at the 90 % strike, unless the wing binds), index smile error LC minus listed: {idx_failed}. "
        f"`check_forward` fails on {meta['forward_failed']} trades. The dates that fail the index gate are in every summary of this part; tables C_arith_gap, C_arith_payout and C_arith_split each carry the rows "
        f"\"{SAMPLE_LABEL['intersection_index_ok']}\" ({len(S['intersection_index_ok'])} trades). "
        f"Separately, the names' 2 % diagnostic (`check_names`: |Σ w E_LC[R_i²] / Σ w M_i − 1| ≤ 2 %, the names' second moment under the LC Monte Carlo against the listed strips) is outside 2 % on "
        f"{meta['names_outside']} of the {n_i} trades ({meta['names_outside_and_gate']} of them also `check` under the current rule); it is not a gate.",
        "",
    ]  # fmt: skip

    # reproduction
    out += [
        "**Table C_arith_reproduction.** The study's printed numbers recomputed by this script from the study's tables, on the study's own samples.",
        "",
        "| quantity | study printed | this script | match to the printed digits |",
        "|---|---|---|---|",
    ]  # fmt: skip
    for r in rep.to_dict("records"):
        if r["sample_stated"]:
            out.append(f"| {r['quantity']} | {r['study_printed']} | {r['this_script']:.4f} | {'yes' if r['match'] else 'NO'} |")  # fmt: skip
            continue
        both = "consistent with both readings" if r["match"] and r["other_reading_match"] else "NO (one of the two readings)"  # fmt: skip
        out.append(f"| {r['quantity']} | {r['study_printed']} | {r['this_script']:.4f} on the 217 trades of the monthly subset; {r['other_reading']:.4f} on the other reading ({r['other_reading_what']}) | {both} |")  # fmt: skip
    stated = rep[rep["sample_stated"]]
    loose = rep[~rep["sample_stated"]]
    loose_ok = int((loose["match"] & loose["other_reading_match"].astype(bool)).sum())
    out += [
        "",
        f"{int(stated['match'].sum())} of {len(rep)} printed numbers reproduced to the printed digits on a stated sample; {len(loose)} printed to one digit in a sentence that does not state the sample "
        f"(final report sec. 5.3, `f5_results.tex` l.131: \"the hedged gap about -0.1 instead of -0.5\"), {loose_ok} of them consistent with both readings (the 217 trades of the monthly subset; the 1,010 weekly trades): "
        "one printed digit does not tell the two readings apart. Gap P&L in % of notional, the others ratios; sources of the printed numbers in `tables/C_arith_reproduction.csv` "
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
        (
            "hedged",
            "study_monthly",
        ): "-0.5 / about -0.1 (one digit; sample not stated by the study)",
    }
    for structure in ("held", "hedged"):
        for sample in ("study_monthly", "intersection", "intersection_unflagged", "intersection_index_ok"):  # fmt: skip
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
        "a bound whatever the dependence between dates, which is the standard error the records carry for the LC mean price and the price difference: the dates share their seeds, so the independent-dates figure, in the CSV, is not supported).",
        "",
    ]  # fmt: skip
    if ok["gap"]:
        edge = []
        for sample in ("intersection", "intersection_unflagged"):
            r = gap[(gap["sample"] == sample) & (gap["model"] == "lc") & (gap["structure"] == "hedged")].iloc[0]  # fmt: skip
            edge.append(
                f"on the {r['n_trades']} trades ({SAMPLE_LABEL[sample]}) {r['mean_pnl_pct']:.3f} ± {r['se_hansen_hodrick_pct']:.3f}, t-ratio {r['t_ratio']:.2f}, block-bootstrap 95 % interval {interval(r['ci95_lo_pct'], r['ci95_hi_pct'])} "
                f"(over the bootstrap seeds {SEEDS_RANGE[0]} to {SEEDS_RANGE[-1]} the upper bound runs from {r['ci95_hi_pct_seed_min']:+.3f} to {r['ci95_hi_pct_seed_max']:+.3f}); "
                f"with {LAG_MONTHLY_MORE} lags the standard error is {r['se_hansen_hodrick_lag3_pct']:.3f} and the t-ratio {r['t_ratio_lag3']:.2f}"
            )  # fmt: skip
        lag3 = gap[(gap["sample"] == "intersection") & (gap["model"] == "lc")].set_index("structure")  # fmt: skip
        e214 = gap[(gap["sample"] == "intersection") & (gap["model"] == "lc") & (gap["structure"] == "hedged")].iloc[0]  # fmt: skip
        e207 = gap[(gap["sample"] == "intersection_unflagged") & (gap["model"] == "lc") & (gap["structure"] == "hedged")].iloc[0]  # fmt: skip
        zero_in = "includes 0" if e214["ci95_lo_pct"] <= 0 <= e214["ci95_hi_pct"] else "excludes 0"
        zero_u = "includes 0" if e207["ci95_lo_pct"] <= 0 <= e207["ci95_hi_pct"] else "excludes 0"
        out += [
            "The hedged gap at the LC price, edge of significance: " + "; ".join(edge) + f". On the {e214['n_trades']} trades the interval {zero_in} and the mean is {abs(e214['t_ratio']):.1f} standard errors from 0; "
            f"on the {e207['n_trades']} trades the interval {zero_u}. The result sits at the edge of significance: "
            f"the {e214['n_trades']} trades do not support a claim about the sign of the hedged gap at the LC price; on the {e207['n_trades']} trades the mean is {abs(e207['t_ratio']):.1f} standard errors from 0 and the interval's upper bound is within {abs(e207['ci95_hi_pct_seed_max']):.2f} to {abs(e207['ci95_hi_pct_seed_min']):.2f} of 0 across seeds. "
            f"Lags: some pairs of trades three monthly entries apart overlap by a few days, so the {LAG_MONTHLY}-lag standard errors are the study's convention, not an upper value; "
            f"with {LAG_MONTHLY_MORE} lags, at the LC price on the {n_i} trades, the hedged gap's is {lag3.loc['hedged', 'se_hansen_hodrick_lag3_pct']:.3f} (against {lag3.loc['hedged', 'se_hansen_hodrick_pct']:.3f}) "
            f"and the held gap's {lag3.loc['held', 'se_hansen_hodrick_lag3_pct']:.3f} (against {lag3.loc['held', 'se_hansen_hodrick_pct']:.3f}); column `se_hansen_hodrick_lag3_pct` of the CSV for every row.",
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
    plabel = {"forward": "forward", "call_100": "call struck at the copula's forward price", "call_050": "call struck at 0.5 × the copula's forward price", "call_075": "call struck at 0.75 × the copula's forward price",
              "call_125": "call struck at 1.25 × the copula's forward price", "call_150": "call struck at 1.5 × the copula's forward price"}  # fmt: skip

    def pay_cell(product: str, sample: str, model: str) -> str:
        q = payout[(payout["product"] == product) & (payout["sample"] == sample) & (payout["model"] == model)]  # fmt: skip
        if q.empty:
            return not_priced if model == "lc" else "n/a"
        if model == "lc" and not ok["payout"]:
            return pending
        r = q.iloc[0]
        return f"{r['payout_per_1_of_premium']:.3f} {interval(r['ci95_lo'], r['ci95_hi'])}"

    def pay_rows(products: tuple[str, ...], extra: tuple[str, ...] = ()) -> list[str]:
        lines = []
        for product in products:
            for sample in ("study_monthly", "intersection", "intersection_unflagged", *extra):
                q = payout[(payout["product"] == product) & (payout["sample"] == sample)].set_index("model")  # fmt: skip
                prem = " / ".join(f"{q.loc[m, 'mean_premium_pct']:.3f}" if m in q.index else "n/a" for m in MODELS)  # fmt: skip
                mcb = sig(q.loc["lc", "mc_se_bound"]) if "lc" in q.index and ok["payout"] else ""
                lines.append(
                    f"| {plabel[product]} | {SAMPLE_LABEL[sample]} | {len(S[sample])} | {pay_cell(product, sample, 'copula')} | {pay_cell(product, sample, 'model_s')} | "
                    f"{pay_cell(product, sample, 'lc')} | {mcb} | {prem} | {printed_pay[product] if sample == 'study_monthly' else ''} |"
                )  # fmt: skip
        return lines

    out += pay_rows(("forward", "call_100"), ("intersection_index_ok",))
    out += [
        "",
        "Σ over trades of the realised payoff (`D` for the forward; `max(D − K, 0)` for the call, `K` = the copula forward's price of the entry date, in cash, the same for the three models) over Σ of the model's price "
        f"(`P_D`, `P_D_S`, `ED_lc`; `C_100`, `C_S_100`, `C_100_lc`); interval = the study's (`disp_stats.bootstrap_ratio`: circular blocks of {BLOCK_MONTHLY} monthly entries, {N_RESAMPLES} resamples, seed {SEED_RATIO}, 2.5th–97.5th percentile); "
        "the study's T5_model_S prints no interval for the forward on the monthly subset (the same bootstrap is applied here); the Monte Carlo error is that of the LC prices in the ratio (ratio × Σ standard errors / Σ prices, a bound).",
        "",
    ]  # fmt: skip
    own = payout[(payout["product"] == "call_100") & (payout["sample"] == "intersection")].set_index("model")  # fmt: skip
    if ok["payout"]:
        f214 = payout[(payout["product"] == "forward") & (payout["sample"] == "intersection") & (payout["model"] == "lc")].iloc[0]  # fmt: skip
        f207 = payout[(payout["product"] == "forward") & (payout["sample"] == "intersection_unflagged") & (payout["model"] == "lc")].iloc[0]  # fmt: skip
        out += [
            f"The strike of the call. The call is struck at the copula's forward price for the three models (the study's convention for model S). On the {n_i} trades this strike is "
            f"{100 * (own.loc['lc', 'strike_over_own_forward'] - 1):.1f} % above the LC forward price and {100 * (own.loc['model_s', 'strike_over_own_forward'] - 1):.1f} % above model S's "
            "(Σ strikes / Σ of the model's own forward prices − 1): under LC and model S the call is out of the money at the model's own forward. The package carries no call struck at the LC forward.",
            "",
            f"The LC forward payout, edge of significance: its interval includes 1 on the {f214['n_trades']} trades ({interval(f214['ci95_lo'], f214['ci95_hi'])}), and its lower bound on the {f207['n_trades']} trades "
            f"({f207['ci95_lo']:.3f}) is within bootstrap noise of 1: over the bootstrap seeds {SEEDS_RANGE[0]} to {SEEDS_RANGE[-1]} (the study's seed is {SEED_RATIO}) that lower bound runs from "
            f"{f207['ci95_lo_seed_min']:.3f} to {f207['ci95_lo_seed_max']:.3f} (on the {f214['n_trades']} trades from {f214['ci95_lo_seed_min']:.3f} to {f214['ci95_lo_seed_max']:.3f}); "
            "the third decimal of the bounds is not stable across bootstrap seeds. Neither result supports a claim about a sign or \"above 1\".",
            "",
        ]  # fmt: skip
    out += [
        "**Table C_arith_payout (other strikes).** The other strikes of the study's table T5_model_S, same definitions.",
        "",
        "| product | sample | trades | copula | model S | LC | LC: Monte Carlo error, at most | mean premium, % of notional (copula / S / LC) | study printed (copula / model S) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    out += pay_rows(("call_050", "call_075", "call_125", "call_150"))
    out += [
        "",
        "Strikes are multiples of the copula forward's price of the entry date, fixed in cash, for the three models; the rows without the dates that fail the index gate are in the CSV.",
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
    subsamples = ("intersection_names_in", "intersection_names_out")
    for r in sp.to_dict("records"):
        if r["sample"] in subsamples:
            continue  # the table by the names' diagnostic, below
        lcrow = r["model"] == "lc"
        if lcrow and not ok["split"]:
            cells = [pending] * 6 + [f3(r["kappa_realised"], 4)]
        else:
            show_se = not r["sample"].startswith("weekly")
            cells = [pc.pm(r[k], r[f"{k}_se_boot"] if show_se else None, 4) for k in (*(f for f, _ in FACTORS), "kappa_model", "kappa_realised")]  # fmt: skip
        out.append(f"| {SAMPLE_LABEL[r['sample']]}, {MODEL_LABEL[r['model']]} | {r['n_trades']} | " + " | ".join(cells) + f" | {printed_split.get((r['sample'], r['model']), '')} |")  # fmt: skip
    out += [
        "",
        "All ratios pooled over the trades (`disp_tables.richness`): price over payoff = mean model forward price / mean realised D; listed-option part = √(mean EQV / mean realised V), EQV the listed options' E[V] (the same for every model); "
        "dispersion priced, model over listed = √(mean model E[V] / mean EQV) (`EV`, `EV_S`, `EV_lc`); κ of the model = mean price / √(mean model E[V]); κ realised = mean D / √(mean V); "
        f"± on the monthly rows = standard deviation over the resamples of the study's D4 bootstrap (circular blocks of {BLOCK_MONTHLY} monthly entries, {N_RESAMPLES} resamples, seed {SEED_SPLIT}); the study printed no interval on these rows (95 % intervals in the CSV). "
        "The factors are printed with 4 decimals so that the printed factors multiply to the printed product (the study printed 3). The listed-option part and κ realised are model-free: no model price is in them and they are the same for the three models on a sample. "
        "κ of the model is the price over the root of the model's own E[V], as in the final report (its sidenote prints 0.729 and 0.743 on the 1,010 trades); it differs by definition from the κ_Q of the study's table T4_model_S, "
        "which is the price over the root of EQV.",
        "",
    ]  # fmt: skip

    # the split by the names' diagnostic, and the second moments behind LC's E[V]
    out += [
        "**Table C_arith_split (by the names' 2 % diagnostic).** The split on the trades inside and outside the names' 2 % diagnostic (not a gate), copula and LC.",
        "",
        "| row | trades | price over payoff | = listed-option part | × model part | model part = dispersion priced, model over listed | × κ, model over realised | model part minus the copula's |",
        "|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    by = sp.set_index(["sample", "model"])
    for sample in ("intersection", *subsamples):
        for model in ("copula", "lc"):
            r = by.loc[(sample, model)]
            if model == "lc" and not ok["split"]:
                cells = [pending] * 6
            else:
                cells = [pc.pm(r[k], r[f"{k}_se_boot"], 4) for k, _ in FACTORS]
                cells.append("" if model == "copula" else f"{r['model_part_minus_copula']:+.4f} ± {r['model_part_minus_copula_se_boot']:.4f}")  # fmt: skip
            out.append(f"| {SAMPLE_LABEL[sample]}, {MODEL_LABEL[model]} | {int(r['n_trades'])} | " + " | ".join(cells) + " |")  # fmt: skip
    out += [
        "",
        f"Inside = |Σ w E_LC[R_i²] / Σ w M_i − 1| ≤ 2 % on the entry date (`check_names` of the LC table), outside = the others; ± = the same bootstrap on the trades of the sub-sample in date order "
        f"(blocks of {BLOCK_MONTHLY} consecutive trades of the sub-sample, which are not consecutive months); model part minus the copula's = the LC model part less the copula's on the same trades and the same resamples "
        "(model S's rows on these sub-samples are in the CSV).",
        "",
        "**Table C_arith_second_moments.** Mean E_LC[V] against mean EQV. E[V] = Σ w E[R_i²] − E[R̄²], so mean E_LC[V] − mean EQV = (names' second moment under the LC Monte Carlo − names' listed strips) + (listed index strip − basket's second moment under LC).",
        "",
        "| sample | trades | mean E_LC[V] / mean EQV − 1 | of which the names above their listed strips | of which the basket below the listed index strip | names, LC / listed | basket, LC / listed | names, copula / listed | names, model S / listed | LC dispersion priced over listed | the same with the names at the listed strips |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]  # fmt: skip
    mo = mom.set_index("sample")
    for sample in mo.index:
        r = mo.loc[sample]
        if not ok["split"]:
            out.append(f"| {SAMPLE_LABEL[sample]} | {int(r['n_trades'])} | " + " | ".join([pending] * 9) + " |")  # fmt: skip
            continue
        cells = [f"{100 * r['excess_ev_over_eqv']:.2f} % ± {100 * r['excess_ev_over_eqv_se_boot']:.2f}", f"{100 * r['share_names']:.1f} % ± {100 * r['share_names_se_boot']:.1f}",
                 f"{100 * r['share_basket']:.1f} % ± {100 * r['share_basket_se_boot']:.1f}"]  # fmt: skip
        cells += [pc.pm(r[k], r[f"{k}_se_boot"], 4) for k in ("names_ratio_lc", "basket_ratio_lc", "names_ratio_copula", "names_ratio_model_s", "dispersion_lc_over_listed", "dispersion_lc_names_at_listed")]  # fmt: skip
        out.append(f"| {SAMPLE_LABEL[sample]} | {int(r['n_trades'])} | " + " | ".join(cells) + " |")
    out += [
        "",
        "Pooled ratios of means over the trades: names = Σ w E[R_i²] of the model (`sum_w_ER2_lc`; `EV + M_B` of the study's entries for the copula, `EV_S + M_B_S` for model S) over Σ w M_i of the listed strips (`sum_w_M`); "
        "basket = E[R̄²] of the model (`E_Rbar2_lc`) over the listed index strip (`M_B_listed`); ± = the bootstrap of the split table on the same trades (sampling over trades); "
        f"the Monte Carlo error of the LC names' ratio on the {n_i} trades is {sig(mo.loc['intersection', 'names_ratio_lc_mc_indep'])} with independent dates and at most {sig(mo.loc['intersection', 'names_ratio_lc_mc_bound'])} whatever their dependence, "
        f"and of the LC basket's ratio {sig(mo.loc['intersection', 'basket_ratio_lc_mc_indep'])} and at most {sig(mo.loc['intersection', 'basket_ratio_lc_mc_bound'])} "
        "(from the rows' standard errors over the mean listed moment; `tables/C_arith_second_moments.csv`).",
        "",
    ]  # fmt: skip
    if ok["split"]:
        a, lc_rows = mo.loc["intersection"], {smp: by.loc[(smp, "lc")] for smp in ("intersection", *subsamples)}  # fmt: skip
        pair = {smp: f"{r['dispersion_model_over_listed']:.4f} × {r['kappa_model_over_realised']:.4f}" for smp, r in lc_rows.items()}  # fmt: skip
        out += [
            f"Reading of the LC split. On the {n_i} trades {100 * a['share_names']:.0f} % of the excess of mean E_LC[V] over mean EQV is the names' second moment under the LC Monte Carlo above the names' listed strips "
            f"(pooled ratio {a['names_ratio_lc']:.3f}; the copula's is {a['names_ratio_copula']:.3f}) and {100 * a['share_basket']:.0f} % the basket's second moment under LC below the listed index strip (pooled ratio {a['basket_ratio_lc']:.3f}). "
            f"About half of what puts LC's factor \"dispersion priced, model over listed\" ({a['dispersion_lc_over_listed']:.4f}) above 1 therefore comes from the Monte Carlo second moment of the names exceeding the listed strips, "
            f"not from correlation (the names' second moment does not depend on correlation); with the names at the listed strips the factor is {a['dispersion_lc_names_at_listed']:.4f}. "
            f"The two sub-factors of the LC model part are not stable across the sub-samples: {pair['intersection']} on the {int(lc_rows['intersection']['n_trades'])} trades, {pair[subsamples[0]]} on the {int(lc_rows[subsamples[0]]['n_trades'])} inside the names' diagnostic, "
            f"{pair[subsamples[1]]} on the {int(lc_rows[subsamples[1]]['n_trades'])} outside it. Their product, the model part, is stable against the copula's: LC model part minus the copula's is "
            f"{lc_rows['intersection']['model_part_minus_copula']:+.4f} on the {int(lc_rows['intersection']['n_trades'])}, {lc_rows[subsamples[0]]['model_part_minus_copula']:+.4f} inside and {lc_rows[subsamples[1]]['model_part_minus_copula']:+.4f} outside. "
            "The two LC sub-factors are not to be quoted as \"LC prices that much more dispersion than listed because of its correlation\"; the model part is the number that holds on the sub-samples.",
            "",
        ]  # fmt: skip
    out += [
        "**Table C_arith_split (Monte Carlo error of the LC rows).** The error of the LC prices in each factor: with independent dates / at most.",
        "",
        "| row | price over payoff | model part | dispersion priced, model over listed | κ, model over realised | κ of the model |",
        "|---|---|---|---|---|---|",
    ]  # fmt: skip
    for r in sp[(sp["model"] == "lc") & ~sp["sample"].isin(subsamples)].to_dict("records"):
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
    t = g[["date", "expiry", "IS", "flagged", "names_unscreened", "status", "reason", "status_current", "reason_current", *lp.GATING_CHECKS, "check_names",
           "index_gate_ok", "names_in_2pct", "names_dev", "wing_binds", "git_commit"]].copy()  # fmt: skip
    for c in ("P_D", "P_D_S", "ED_lc", "ED_lc_se", "EV", "EV_S", "EV_lc", "EV_lc_se", "EQV", "D", "V", "G", "P_G", "GAP_U", "GAP_H", "K_100",
              "PC_pay_100", "C_100", "C_S_100", "C_100_lc", "C_100_lc_se", "idx_err_atm", "idx_err_atm_se", "idx_err_90", "idx_err_90_se",
              "sum_w_ER2_lc", "sum_w_ER2_lc_se", "sum_w_M", "E_Rbar2_lc", "E_Rbar2_lc_se", "M_B_listed", "M_B", "M_B_S"):  # fmt: skip
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
    ap.add_argument("--base", default=str(pc.PM), help="the folder written to (default: the package; any other folder is a dry run and appends no line to STATUS.md)")  # fmt: skip
    args = ap.parse_args()
    base = Path(args.base)
    S = samples(pc.LC_OUT / args.lc_table)
    records, md, tables = build(S)
    tables["C_arith_trades"] = trades_table(S)
    for name, frame in tables.items():
        log.info("%s", pc.save_table(frame, name, base))
    part = base / "parts" / f"{PART}.json"
    before = part.read_text() if part.exists() else None
    pc.write_part(PART, records, md, base)
    rep = tables["C_arith_reproduction"]
    meta = S["meta"]
    log.info("%s", md)
    log.info("records: %d; reproduction: %d of %d printed numbers matched", len(records), int(rep["match"].sum()), len(rep))  # fmt: skip
    if base != pc.PM:
        log.info("dry run in %s: no line in STATUS.md", base)
        return
    if part.read_text() == before:
        log.info("the part is unchanged: no new line in STATUS.md")
        return
    stated = rep[rep["sample_stated"]]
    pc.status(
        f"C_arith revised after verification (parts/C_arith.json and .md, tables/C_arith_*.csv): the study's arithmetic at the LC price on {len(S['intersection'])} trades "
        f"({len(S['intersection_unflagged'])} without the flagged dates, {len(S['intersection_index_ok'])} without the dates that fail the index gate), LC rows of commit {meta['commit']} ({meta['lc_table']}); "
        f"{int(stated['match'].sum())} of {len(rep)} printed numbers of the study reproduced on a stated sample, {len(rep) - len(stated)} printed to one digit without a stated sample."
    )  # fmt: skip


if __name__ == "__main__":
    main()
