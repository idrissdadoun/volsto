"""Section C of the PM results package of 2026-10-09 (``outputs/dispersion_lc/pm_update``): the
3m history of the local correlation model at the development budget with the owner's decisions
1-2 on, the answers to checks (b) and (d), and figures F1 and F2.

Reads (read-only) ``outputs/dispersion_lc/lcm_3m_dev_repair.parquet`` (one row per date),
``outputs/dispersion/entries_3m.parquet`` (basket B1, one row per date: checked) and
``outputs/dispersion/model_s_3m.parquet``.  Writes, through ``scripts/pm_common.py`` only:

- ``tables/C_history_by_date.csv``: one row per date (failed dates stay, their numbers empty);
- ``tables/C_history_summaries.csv``: long format ``quantity, label, sample, statistic, value,
  se, se_kind, mc_se, n``;
- ``tables/C_history_samples.csv`` (the samples and the tercile bounds) and
  ``tables/C_history_regression.csv`` (check (d));
- ``figures/F1_forward_over_copula.pdf/.csv`` and ``figures/F2_calls_over_copula_by_strike.pdf/.csv``;
- ``parts/C_history.json`` and ``parts/C_history.md``.

Conventions.  A date is *priced* when its status is not ``failed``; every sample is a subset of
the priced dates.  A date is *flagged* when a name is kept unscreened (``n_names_unscreened >
0``, decision 2); every summary is given with and without the flagged dates.  Model S numbers
are used on its converged dates only, as ``scripts/disp_tables2.py::model_s_tables`` does
(``x = x[x["converged"]]``), and the pooled ratios mirror that function: the sum over the dates
of the numerator over the sum of the denominator (``gx["P_D_S"].sum() / gx["P_D"].sum()``,
``gx[f"C_S_{m}"].sum() / gx[f"C_{m}"].sum()``).

Standard errors.  Per date: the Monte Carlo error (the row's for LC/CC, which is paired; the
delta method with the copula's own ``P_D_se`` / ``C_se_<m>`` for a ratio to the copula, the two
errors independent).  A mean over dates: ``se`` is the standard error across dates (sd/√n, no
serial-correlation adjustment); ``mc_se`` is an upper bound of its Monte Carlo error with the
dates fixed — the dates share the particle and pricing seeds, so the per-date errors are added
linearly (the mean of the per-date errors), not in quadrature.  A pooled ratio ``R = Σa/Σb``:
``se`` is the across-dates linearisation ``√(n/(n−1)·Σ(a_i − R·b_i)²)/Σb``; ``mc_se`` is the
same upper bound, ``Σ sd(a_i − R·b_i)/Σb``.  The study's own numbers (model S, the copula's κ
and E[V], the listed-variance forward) carry no Monte Carlo error here.

Check (d) is an ordinary least squares fit with an intercept, classical and HC1 standard errors
(numpy only, QR); when the notes of the independent check are on disk its results are compared
to the digit.

Run: ``.venv/bin/python scripts/pm_history.py`` (idempotent; rerun when the table changes).
"""

# ruff: noqa: RUF001, E501 — report prose: typographic signs and long table lines

from __future__ import annotations

import argparse
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pm_common as pc

LOG = logging.getLogger("pm_history")

SECTION = "C"
PART = "C_history"
ROWS = pc.LC_OUT / "lcm_3m_dev_repair.parquet"
ENTRIES = pc.STUDY / "entries_3m.parquet"
MODEL_S = pc.STUDY / "model_s_3m.parquet"
#: The independent check of (d) (another agent's notes; compared when present).
CHECK_D_JSON = Path(
    "/private/tmp/claude-501/-Users-idrissdadoun-Code-volsto/75dd7f23-d74c-43a9-b99e-61c23722b22c/scratchpad/r4/check-d/check_d_results.json"
)
#: The stopped production pass at the old defaults: today's row is shown beside check (b).
PRODUCTION_OLD = pc.LC_OUT / "lcm_3m_norepair.parquet"
CALL_TAGS = ("075", "100", "125", "150")
#: ``scripts/lcm_price.py::CLIP_FLAG_MASS`` (copied: that file's neighbours are being edited).
CLIP_FLAG_MASS = 0.01
TODAY = pc.REFERENCE_DATES[0]
#: The owner's arithmetic for today in check (b).
OWNER_B = {"lc_over_copula": 0.9693, "cc_over_copula": 0.9992}
BASE_SAMPLES = (
    ("all", "all priced dates"),
    ("h1", "2007–2016"),
    ("h2", "2017–2026"),
    ("clip_low", "clipped mass: low tercile"),
    ("clip_mid", "clipped mass: middle tercile"),
    ("clip_high", "clipped mass: high tercile"),
    ("S", "∩ model S converged"),
)
STUDY_NOTE = "the study's numbers on this table's dates: no Monte Carlo standard error here"


@dataclass(frozen=True)
class Quantity:
    """One per-date quantity: its column, label and definition, the column of its per-date
    standard error, and, for a ratio, the numerator and denominator of its pooled form."""

    key: str
    label: str
    definition: str
    se: str | None = None
    num: str | None = None
    den: str | None = None
    num_se: str | None = None
    den_se: str | None = None
    paired: bool = False
    study: bool = False
    digits: int = 4
    in_md: bool = True


def truth(flag: pd.Series) -> pd.Series:
    """A flag column as booleans (empty on a failed date: False)."""
    return flag.map(lambda x: x is True or x == 1.0).astype(bool)


def mult(tag: str) -> str:
    return f"{int(tag) / 100:g}"


def quantities() -> list[Quantity]:
    cop = "P_D the copula's forward of `entries_3m.parquet` (basket B1)"
    out = [
        Quantity("lc_over_cc", "LC/CC", "E_LC[D] / E_CC[D], paired on common paths (the row's `ratio` and `ratio_se`)", "lc_over_cc_se", "ED_lc", "ED_cc", "ED_lc_se", "ED_cc_se", paired=True),
        Quantity("lc_over_copula", "LC/copula", f"E_LC[D] / P_D, {cop}; se: delta method with `ED_lc_se` and the copula's `P_D_se`, independent", "lc_over_copula_se", "ED_lc", "P_D", "ED_lc_se", "P_D_se"),
        Quantity("cc_over_copula", "CC/copula", f"E_CC[D] / P_D, {cop}; se: delta method with `ED_cc_se` and the copula's `P_D_se`, independent", "cc_over_copula_se", "ED_cc", "P_D", "ED_cc_se", "P_D_se"),
        Quantity("s_over_copula", "S/copula", "P_D_S / P_D on the dates where model S converged (`model_s_3m.parquet`)", num="P_D_S", den="P_D", study=True),
        Quantity("listed_fwd_ratio", "listed-variance forward / copula", "√(EQV / EV), EV the copula's E[V] of the entry: κ_cop·√EQV over P_D", num="listed_fwd", den="P_D", study=True),
        Quantity("ED_wing_over_copula", "ED_wing / copula", "κ_LC·√(Σw E_LC[R_i²] − M_B^listed) / P_D; se: delta method with `ED_wing_se` and `P_D_se`", "ED_wing_over_copula_se", "ED_wing", "P_D", "ED_wing_se", "P_D_se"),
        Quantity("ED_eqv_over_copula", "ED_eqv / copula", "κ_LC·√EQV / P_D; se: delta method with `ED_eqv_se` and `P_D_se`", "ED_eqv_over_copula_se", "ED_eqv", "P_D", "ED_eqv_se", "P_D_se"),
        Quantity("kappa_lc", "κ, LC", "E_LC[D] / √E_LC[V]", "kappa_lc_se"),
        Quantity("kappa_cc", "κ, CC", "E_CC[D] / √E_CC[V]", "kappa_cc_se"),
        Quantity("kappa_cop", "κ, copula", "P_D / √EV (`kappa_cop` of the entry)", study=True),
        Quantity("kappa_S", "κ, model S", "P_D_S / √EV_S on the dates where model S converged", study=True),
        Quantity("EV_over_EQV_lc", "E[V]/EQV, LC", "EV_lc / EQV, EQV the listed-option value of E[V]; se: `EV_lc_se` / EQV", "EV_over_EQV_lc_se", "EV_lc", "EQV", "EV_lc_se"),
        Quantity("EV_over_EQV_cc", "E[V]/EQV, CC", "EV_cc / EQV; se: `EV_cc_se` / EQV", "EV_over_EQV_cc_se", "EV_cc", "EQV", "EV_cc_se"),
        Quantity("EV_over_EQV_cop", "E[V]/EQV, copula", "EV / EQV of the entry", num="EV_cop", den="EQV", study=True),
        Quantity("EV_over_EQV_S", "E[V]/EQV, model S", "EV_S / EQV on the dates where model S converged", num="EV_S", den="EQV", study=True),
        Quantity("EV_single_part", "E[V] split, single-name part", "Σw E_LC[R_i²] − Σw M_i^listed (EV_lc − EQV = single-name part − basket part)", "EV_single_part_se", digits=6),
        Quantity("EV_basket_part", "E[V] split, basket part", "E_LC[R̄²] − M_B^listed", "EV_basket_part_se", digits=6),
        Quantity("EV_basket_part_rel_MB", "basket part / M_B^listed", "(E_LC[R̄²] − M_B^listed) / M_B^listed; se: `EV_basket_part_se` / M_B^listed", "EV_basket_part_rel_MB_se"),
        Quantity("clip_inner_max", "clipped mass inside ±2.5 sd", "the probability mass inside ±2.5 sd on which the correlation multiplier λ is clipped, the largest slice (`clip_inner_max`)"),
        Quantity("clip_low_inner_max", "clipped mass at λ = 0", "the same at the lower bound λ = 0 (`clip_low_inner_max`)"),
        Quantity("clip_high_inner_max", "clipped mass at the cap", "the same at the cap of λ (`clip_high_inner_max`)"),
    ]  # fmt: skip
    for m in CALL_TAGS:
        k = f"K = {mult(m)} × P_D (the study's cash strike `K_{m}`, the same under every model)"
        out.append(Quantity(f"C_lc_over_copula_{m}", f"call {mult(m)}×: LC/copula", f"E_LC[(D − K)⁺] over the copula's `C_{m}`, {k}; se: delta method with `C_{m}_lc_se` and the copula's `C_se_{m}`", f"C_lc_over_copula_{m}_se", f"C_{m}_lc", f"C_{m}_cop", f"C_{m}_lc_se", f"C_{m}_cop_se"))  # fmt: skip
    for m in CALL_TAGS:
        out.append(Quantity(f"C_S_over_copula_{m}", f"call {mult(m)}×: S/copula", f"model S's `C_S_{m}` over the copula's `C_{m}`, same cash strike, on the dates where model S converged", num=f"C_{m}_S", den=f"C_{m}_cop", study=True))  # fmt: skip
    for m in CALL_TAGS:
        out.append(Quantity(f"C_cc_over_copula_{m}", f"call {mult(m)}×: CC/copula", f"E_CC[(D − K)⁺] over the copula's `C_{m}`, same cash strike; se: delta method", f"C_cc_over_copula_{m}_se", f"C_{m}_cc", f"C_{m}_cop", f"C_{m}_cc_se", f"C_{m}_cop_se", in_md=False))  # fmt: skip
    out.append(Quantity("y_check_d", "LC/CC − S/copula", "the regressand of check (d): `lc_over_cc` − `s_over_copula`, on the dates where model S converged; no standard error per date here (model S has none)"))  # fmt: skip
    return out


# ----------------------------------------------------------------------------- the per-date table
def ratio_and_se(
    a: pd.Series, a_se: pd.Series | None, b: pd.Series, b_se: pd.Series | None
) -> tuple[pd.Series, pd.Series]:
    """``a / b`` per date (empty where the denominator is not positive) with the delta-method
    standard error of two independent estimates (``pm_common.ratio_se``)."""
    safe = b.where(b > 0.0)
    ratio = a / safe
    zeros = pd.Series(0.0, index=a.index)
    sa = zeros if a_se is None else a_se
    sb = zeros if b_se is None else b_se
    se = []
    for x, xs, y, ys in zip(a, sa, safe, sb, strict=True):
        if not (np.isfinite(x) and np.isfinite(y) and np.isfinite(xs) and np.isfinite(ys)):
            se.append(np.nan)
        elif x == 0.0:
            se.append(xs / y)
        else:
            se.append(pc.ratio_se(float(x), float(xs), float(y), float(ys)))
    return ratio, pd.Series(se, index=a.index)


def load(rows_path: Path) -> pd.DataFrame:
    """The sweep's rows joined on the date with the study's entry (basket B1) and model S."""
    rows = pd.read_parquet(rows_path)
    if not rows["date"].is_unique:
        raise ValueError(f"{rows_path.name}: several rows on a date")
    entries = pd.read_parquet(ENTRIES)
    b1 = entries[entries["basket"] == "B1"]
    if not b1["date"].is_unique:
        raise ValueError("entries_3m: several B1 rows on a date")
    keep = {"P_D": "P_D", "P_D_se": "P_D_se", "EV": "EV_cop", "EQV": "EQV_cop", "kappa_cop": "kappa_cop", "monthly": "monthly"}  # fmt: skip
    for m in CALL_TAGS:
        keep |= {f"K_{m}": f"K_{m}_cop", f"C_{m}": f"C_{m}_cop", f"C_se_{m}": f"C_{m}_cop_se"}
    b1 = b1[["date", *keep]].rename(columns=keep)
    s = pd.read_parquet(MODEL_S)
    if not s["date"].is_unique:
        raise ValueError("model_s_3m: several rows on a date")
    s_keep = {"converged": "model_s_converged", "P_D_S": "P_D_S", "EV_S": "EV_S"}
    s_keep |= {f"C_S_{m}": f"C_{m}_S" for m in CALL_TAGS}
    s = s[["date", *s_keep]].rename(columns=s_keep)
    frame = rows.merge(b1, on="date", how="left", validate="one_to_one")
    frame = frame.merge(s, on="date", how="left", validate="one_to_one", indicator="model_s_row")
    frame["model_s_row"] = frame["model_s_row"] == "both"
    if frame["P_D"].isna().any():
        raise ValueError(
            f"dates without a B1 entry: {frame.loc[frame['P_D'].isna(), 'date'].tolist()}"
        )
    frame = frame.sort_values("date").reset_index(drop=True)
    priced = (frame["status"] != "failed") & frame["ED_lc"].notna()
    # the copula's numbers of the row are the entry's: a row priced against another entry is refused
    for ours, theirs in (("P_D_copula", "P_D"), ("EV_copula", "EV_cop"), ("EQV", "EQV_cop"), *((f"K_{m}", f"K_{m}_cop") for m in CALL_TAGS)):  # fmt: skip
        gap = (frame.loc[priced, ours] - frame.loc[priced, theirs]).abs() / frame.loc[
            priced, theirs
        ].abs()
        if not (gap <= 1e-12).all():
            raise ValueError(
                f"{ours} of the rows differs from the entry's {theirs} (max relative {gap.max():.3g})"
            )
    frame["priced"] = priced
    return frame


def by_date(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per date with the flags, every quantity of section C and its standard error."""
    f = frame
    priced = f["priced"]
    cols: dict[str, Any] = {
        "date": f["date"],
        "status": f["status"],
        "reason": f["reason"].fillna(""),
        "T": f["T"],
    }
    d: Any = cols  # filled as a dict, then one frame (no fragmentation)
    year = pd.to_datetime(f["date"]).dt.year
    d["half"] = np.where(year <= 2016, "2007-2016", "2017-2026")
    converged = f["model_s_converged"].fillna(False).astype(bool)
    d["flag_unscreened"] = f["n_names_unscreened"] > 0
    d["names_unscreened"] = f["names_unscreened"].fillna("")
    d["n_names_unscreened"] = f["n_names_unscreened"]
    d["flag_clip_low"] = f["clip_low_inner_max"] > CLIP_FLAG_MASS
    d["flag_clip_high"] = f["clip_high_inner_max"] > CLIP_FLAG_MASS
    d["flag_clip"] = d["flag_clip_low"] | d["flag_clip_high"]
    d["model_s_converged"] = converged
    d["n_names_extrapolated"] = f["n_names_extrapolated"]
    d["index_extrapolated"] = f["index_extrapolated"]
    for c in ("n_particles", "n_paths", "companion_paths", "git_commit"):
        d[c] = f[c]
    for c in ("ED_lc", "ED_lc_se", "ED_cc", "ED_cc_se", "P_D", "P_D_se"):
        d[c] = f[c]
    use_s = priced & converged
    d["P_D_S"] = f["P_D_S"].where(use_s)
    d["lc_over_cc"], d["lc_over_cc_se"] = f["ratio"], f["ratio_se"]
    d["lc_over_copula"], d["lc_over_copula_se"] = ratio_and_se(
        f["ED_lc"], f["ED_lc_se"], f["P_D"], f["P_D_se"]
    )
    d["cc_over_copula"], d["cc_over_copula_se"] = ratio_and_se(
        f["ED_cc"], f["ED_cc_se"], f["P_D"], f["P_D_se"]
    )
    d["s_over_copula"] = d["P_D_S"] / f["P_D"].where(f["P_D"] > 0.0)
    ok_var = (f["EQV_cop"] > 0.0) & (f["EV_cop"] > 0.0)
    d["listed_fwd_ratio"] = np.sqrt((f["EQV_cop"] / f["EV_cop"]).where(ok_var))
    d["listed_fwd"] = f["P_D"] * d["listed_fwd_ratio"]
    for name in ("ED_wing", "ED_eqv"):
        d[name], d[f"{name}_se"] = f[name], f[f"{name}_se"]
        d[f"{name}_over_copula"], d[f"{name}_over_copula_se"] = ratio_and_se(
            f[name], f[f"{name}_se"], f["P_D"], f["P_D_se"]
        )
        d[f"{name}_over_cc"], d[f"{name}_over_cc_se"] = f[f"{name}_ratio"], f[f"{name}_ratio_se"]
    for c in ("kappa_lc", "kappa_lc_se", "kappa_cc", "kappa_cc_se", "kappa_cop"):
        d[c] = f[c]
    d["EV_S"] = f["EV_S"].where(use_s)
    d["kappa_S"] = d["P_D_S"] / np.sqrt(d["EV_S"].where(d["EV_S"] > 0.0))
    for c in ("EV_lc", "EV_lc_se", "EV_cc", "EV_cc_se", "EV_cop"):
        d[c] = f[c]
    d["EQV"], d["M_B_listed"] = f["EQV"], f["M_B_listed"]
    eqv = f["EQV"].where(f["EQV"] > 0.0)
    d["EV_over_EQV_lc"], d["EV_over_EQV_lc_se"] = f["EV_lc"] / eqv, f["EV_lc_se"] / eqv
    d["EV_over_EQV_cc"], d["EV_over_EQV_cc_se"] = f["EV_cc"] / eqv, f["EV_cc_se"] / eqv
    d["EV_over_EQV_cop"] = f["EV_cop"] / eqv
    d["EV_over_EQV_S"] = d["EV_S"] / eqv
    for c in ("EV_single_part", "EV_single_part_se", "EV_basket_part", "EV_basket_part_se"):
        d[c] = f[c]
    m_b = f["M_B_listed"].where(f["M_B_listed"] > 0.0)
    d["EV_basket_part_rel_MB"], d["EV_basket_part_rel_MB_se"] = (
        f["EV_basket_part"] / m_b,
        f["EV_basket_part_se"] / m_b,
    )
    for c in ("clip_inner_max", "clip_low_inner_max", "clip_high_inner_max"):
        d[c] = f[c]
    for m in CALL_TAGS:
        d[f"K_{m}"] = f[f"K_{m}_cop"]
        for c in (
            f"C_{m}_cop",
            f"C_{m}_cop_se",
            f"C_{m}_lc",
            f"C_{m}_lc_se",
            f"C_{m}_cc",
            f"C_{m}_cc_se",
        ):
            d[c] = f[c]
        d[f"C_{m}_S"] = f[f"C_{m}_S"].where(use_s)
        for tag in ("lc", "cc"):
            d[f"C_{tag}_over_copula_{m}"], d[f"C_{tag}_over_copula_{m}_se"] = ratio_and_se(f[f"C_{m}_{tag}"], f[f"C_{m}_{tag}_se"], f[f"C_{m}_cop"], f[f"C_{m}_cop_se"])  # fmt: skip
        d[f"C_S_over_copula_{m}"] = d[f"C_{m}_S"] / f[f"C_{m}_cop"].where(f[f"C_{m}_cop"] > 0.0)
    d["y_check_d"] = d["lc_over_cc"] - d["s_over_copula"]
    d = pd.DataFrame(cols)
    # the identities the definitions rest on, on the priced dates
    checks = {
        "EV_over_EQV of the row = EV_lc/EQV": (f["EV_over_EQV"] - d["EV_over_EQV_lc"]).abs(),
        "kappa_cop = P_D/sqrt(EV)": (f["kappa_cop"] - f["P_D"] / np.sqrt(f["EV_cop"])).abs(),
        "EV_lc - EQV = single part - basket part": (f["EV_lc"] - f["EQV"] - f["EV_single_part"] + f["EV_basket_part"]).abs(),
        "ratio = ED_lc/ED_cc": (f["ratio"] - f["ED_lc"] / f["ED_cc"]).abs(),
    }  # fmt: skip
    for name, gap in checks.items():
        worst = float(gap[priced].max())
        if not worst <= 1e-10:
            raise ValueError(f"identity broken: {name} (max gap {worst:.3g})")
    # failed dates stay with their reason; every number is empty there
    numeric = [c for c in d.columns if c not in ("date", "status", "reason", "half", "names_unscreened", "git_commit", "model_s_converged")]  # fmt: skip
    flags = ["flag_unscreened", "flag_clip_low", "flag_clip_high", "flag_clip", "model_s_converged", "index_extrapolated"]  # fmt: skip
    d[flags] = d[flags].astype(object)
    d.loc[~priced, numeric] = np.nan
    d["priced"] = priced.to_numpy()
    return d


def production_old_defaults(p_d: float, p_d_se: float) -> dict[str, Any] | None:
    """Today's row of the stopped production pass at the old defaults (``lcm_3m_norepair.parquet``),
    for check (b): LC/CC, LC/copula and CC/copula with their Monte Carlo errors, the budget read
    from the row.  ``None`` when the file or the row is missing, or priced against another entry."""
    if not PRODUCTION_OLD.exists():
        return None
    rows = pd.read_parquet(PRODUCTION_OLD)
    rows = rows[(rows["date"] == TODAY) & (rows["status"] != "failed") & rows["ED_lc"].notna()]
    if len(rows) != 1:
        return None
    r = rows.iloc[0]
    if abs(float(r["P_D_copula"]) - p_d) > 1e-12 * p_d:
        LOG.warning(
            "%s: the row of %s is priced against another entry; left out",
            PRODUCTION_OLD.name,
            TODAY,
        )
        return None
    sizes = (float(r["n_particles"]), float(r["n_paths"]), float(r["companion_paths"]))
    budget = pc.BUDGETS["production"] if sizes == (8e5, 8e5, 4e5) else f"{sizes[0]:g} particles / {sizes[1]:g} paths (constant-correlation fit on {sizes[2]:g} paths)"  # fmt: skip
    lc, lc_se, cc, cc_se = (float(r[c]) for c in ("ED_lc", "ED_lc_se", "ED_cc", "ED_cc_se"))
    values = {
        "lc_over_cc": (float(r["ratio"]), float(r["ratio_se"])),
        "lc_over_copula": (lc / p_d, pc.ratio_se(lc, lc_se, p_d, p_d_se)),
        "cc_over_copula": (cc / p_d, pc.ratio_se(cc, cc_se, p_d, p_d_se)),
    }
    return {
        "values": values,
        "budget": budget,
        "commit": str(r["git_commit"]),
        "status": str(r["status"]),
    }


# ----------------------------------------------------------------------------- samples and summaries
def samples(d: pd.DataFrame) -> tuple[dict[str, pd.Series], pd.DataFrame]:
    """The samples (boolean masks over the per-date table) and the table that describes them.
    The terciles of the clipped mass are those of ``clip_inner_max`` on all priced dates (low:
    at or below the 1/3 quantile; high: above the 2/3 quantile); the samples without the
    flagged dates keep the same bounds."""
    priced = d["priced"].astype(bool)
    clip = d["clip_inner_max"].astype(float)
    lo, hi = (float(x) for x in clip[priced].quantile([1.0 / 3.0, 2.0 / 3.0]))
    year = pd.to_datetime(d["date"]).dt.year
    flagged = priced & truth(d["flag_unscreened"])
    converged = priced & truth(d["model_s_converged"])
    base = {
        "all": (priced, "status not failed", np.nan, np.nan),
        "h1": (priced & (year <= 2016), "entry date in 2007–2016", np.nan, np.nan),
        "h2": (priced & (year >= 2017), "entry date in 2017–2026", np.nan, np.nan),
        "clip_low": (priced & (clip <= lo), "clip_inner_max at or below its 1/3 quantile on all priced dates", float(clip[priced].min()), lo),
        "clip_mid": (priced & (clip > lo) & (clip <= hi), "clip_inner_max above the 1/3 and at or below the 2/3 quantile", lo, hi),
        "clip_high": (priced & (clip > hi), "clip_inner_max above its 2/3 quantile", hi, float(clip[priced].max())),
        "S": (converged, "priced and model S converged on the date", np.nan, np.nan),
    }  # fmt: skip
    masks: dict[str, pd.Series] = {}
    rows = []
    for key, (mask, text, low, high) in base.items():
        masks[key] = mask
        masks[f"{key}_unflagged"] = mask & ~flagged
        for name, suffix in (
            (key, ""),
            (f"{key}_unflagged", "; without the dates on which a name is kept unscreened"),
        ):
            rows.append({"sample": name, "n": int(masks[name].sum()), "definition": text + suffix, "clip_inner_max_from": low, "clip_inner_max_to": high})  # fmt: skip
    tercile = np.select(
        [masks["clip_low"], masks["clip_mid"], masks["clip_high"]],
        ["low", "mid", "high"],
        default="",
    )
    d["clip_tercile"] = tercile
    return masks, pd.DataFrame(rows)


def summarise(d: pd.DataFrame, mask: pd.Series, q: Quantity) -> dict[str, dict[str, float]]:
    """``statistic -> {value, se, mc_se, n}`` of one quantity on one sample (module docstring)."""
    sub = d.loc[mask]
    v = sub[q.key].astype(float)
    ok = v.notna()
    v = v[ok]
    n = len(v)
    out: dict[str, dict[str, float]] = {}
    if n == 0:
        return out
    mc = np.nan
    if q.se is not None:
        s = sub.loc[ok, q.se].astype(float)
        mc = float(s.mean()) if s.notna().all() else np.nan
    sem = float(v.std(ddof=1) / math.sqrt(n)) if n > 1 else np.nan
    out["mean"] = {"value": float(v.mean()), "se": sem, "mc_se": mc, "n": n}
    quart = v.quantile([0.25, 0.5, 0.75]).to_numpy()
    for name, value in (("q25", quart[0]), ("median", quart[1]), ("q75", quart[2]), ("min", v.min()), ("max", v.max())):  # fmt: skip
        out[name] = {"value": float(value), "se": np.nan, "mc_se": np.nan, "n": n}
    if q.num is None or q.den is None:
        return out
    a, b = sub.loc[ok, q.num].astype(float), sub.loc[ok, q.den].astype(float)
    if a.isna().any() or b.isna().any():
        raise ValueError(f"{q.key}: a pooled term is missing on a date where the ratio is not")
    total = float(b.sum())
    pooled = float(a.sum()) / total
    resid = a - pooled * b
    se_dates = math.sqrt(n / (n - 1) * float((resid**2).sum())) / abs(total) if n > 1 else np.nan
    mc = np.nan
    if q.num_se is not None:
        sa = sub.loc[ok, q.num_se].astype(float)
        sb = sub.loc[ok, q.den_se].astype(float) if q.den_se is not None else 0.0 * sa
        if q.paired:
            # the covariance of the paired estimates, from the row's standard error of their ratio
            r = a / b
            cov = (sa**2 + r**2 * sb**2 - (sub.loc[ok, q.se].astype(float) * b) ** 2) / (2.0 * r)
            var = sa**2 + pooled**2 * sb**2 - 2.0 * pooled * cov
        else:
            var = sa**2 + pooled**2 * sb**2
        mc = float(np.sqrt(var.clip(lower=0.0)).sum()) / abs(total) if var.notna().all() else np.nan
    out["pooled"] = {"value": pooled, "se": se_dates, "mc_se": mc, "n": n}
    return out


SE_KIND = {
    "mean": "across dates: sd/√n",
    "pooled": "across dates: √(n/(n−1)·Σ(a_i − R·b_i)²)/Σb",
}


def all_summaries(
    d: pd.DataFrame, masks: dict[str, pd.Series], qs: list[Quantity]
) -> tuple[dict[tuple[str, str], dict[str, dict[str, float]]], pd.DataFrame]:
    table: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    rows = []
    for q in qs:
        for name, mask in masks.items():
            stats = summarise(d, mask, q)
            table[(q.key, name)] = stats
            for stat, cell in stats.items():
                rows.append({"quantity": q.key, "label": q.label, "sample": name, "statistic": stat, "value": cell["value"], "se": cell["se"], "se_kind": SE_KIND.get(stat, ""), "mc_se": cell["mc_se"], "n": cell["n"]})  # fmt: skip
    return table, pd.DataFrame(rows)


# ----------------------------------------------------------------------------- check (d)
def ols(y: Any, x: Any, names: list[str]) -> dict[str, Any]:
    """Ordinary least squares of ``y`` on an intercept and the columns of ``x`` (QR): the
    coefficients, the classical standard errors ``s²(X'X)⁻¹`` with ``s² = e'e/(n − k)``, the
    HC1 ones ``n/(n − k)·(X'X)⁻¹[Σ e_i² x_i x_i'](X'X)⁻¹``, their t ratios and the centred R²."""
    yv = np.asarray(y, dtype=float).ravel()
    xv = np.asarray(x, dtype=float).reshape(len(yv), -1)
    if not (np.isfinite(yv).all() and np.isfinite(xv).all()):
        raise ValueError("non-finite value in the regression sample")
    n = len(yv)
    design = np.column_stack([np.ones(n), xv])
    k = design.shape[1]
    if n <= k:
        raise ValueError("not enough dates for the regression")
    qm, rm = np.linalg.qr(design)
    coef = np.linalg.solve(rm, qm.T @ yv)
    r_inv = np.linalg.inv(rm)
    xtx_inv = r_inv @ r_inv.T
    resid = yv - design @ coef
    rss = float(resid @ resid)
    tss = float(((yv - yv.mean()) ** 2).sum())
    cov = rss / (n - k) * xtx_inv
    xe = design * resid[:, None]
    cov_hc1 = n / (n - k) * xtx_inv @ (xe.T @ xe) @ xtx_inv
    se, se_hc1 = np.sqrt(np.diag(cov)), np.sqrt(np.diag(cov_hc1))
    r2 = 1.0 - rss / tss
    return {
        "names": ["const", *names], "coef": coef, "se": se, "se_hc1": se_hc1, "t": coef / se, "t_hc1": coef / se_hc1,
        "r2": r2, "r2_adj": 1.0 - (1.0 - r2) * (n - 1) / (n - k), "n": n,
    }  # fmt: skip


FITS = (
    ("rel", "clipped mass + basket part / M_B^listed", ["clip_inner_max", "EV_basket_part_rel_MB"]),
    ("raw", "clipped mass + basket part (raw)", ["clip_inner_max", "EV_basket_part"]),
    ("clip", "clipped mass alone", ["clip_inner_max"]),
    ("bp_rel", "basket part / M_B^listed alone", ["EV_basket_part_rel_MB"]),
    ("bp_raw", "basket part (raw) alone", ["EV_basket_part"]),
)


def regressions(
    d: pd.DataFrame, masks: dict[str, pd.Series]
) -> tuple[dict[tuple[str, str], dict[str, Any]], pd.DataFrame]:
    fits: dict[tuple[str, str], dict[str, Any]] = {}
    rows = []
    for sample in ("S", "S_unflagged"):
        sub = d.loc[masks[sample]]
        for key, label, regs in FITS:
            fit = ols(sub["y_check_d"], sub[regs], regs)
            fits[(sample, key)] = fit
            for j, term in enumerate(fit["names"]):
                rows.append({
                    "sample": sample, "fit": key, "fit_label": label, "term": term, "coef": fit["coef"][j], "se": fit["se"][j], "t": fit["t"][j],
                    "se_hc1": fit["se_hc1"][j], "t_hc1": fit["t_hc1"][j], "r2": fit["r2"], "r2_adj": fit["r2_adj"], "n": fit["n"],
                })  # fmt: skip
    return fits, pd.DataFrame(rows)


def compare_check_d(
    fits: dict[tuple[str, str], dict[str, Any]], rows_name: str
) -> tuple[str, dict[str, float]]:
    """One sentence, and its counts: the fits against the independent check's JSON, to its 6
    significant digits."""
    if not CHECK_D_JSON.exists():
        return "The independent check's results file is not on disk: not compared in this run.", {}
    results = json.loads(CHECK_D_JSON.read_text())
    block = next(
        (b for b in results.values() if isinstance(b, dict) and b.get("table") == rows_name), None
    )
    if block is None:
        return (
            f"The independent check has no block for `{rows_name}`: not compared in this run.",
            {},
        )
    pairs = [
        (("S", "raw"), block["main"]["raw"], ("coef", "se", "se_hc1", "t", "t_hc1")),
        (("S", "rel"), block["main"]["/ M_B_listed"], ("coef", "se", "se_hc1", "t", "t_hc1")),
        (
            ("S_unflagged", "raw"),
            block["robust"]["raw | without unscreened names"],
            ("coef", "se_hc1"),
        ),
    ]
    total, agree, worst, off = 0, 0, 0.0, []
    for key, theirs, fields in pairs:
        mine = fits[key]
        cells = [
            (f"{key[0]}.{key[1]}.{f}[{j}]", mine[f][j], theirs[f][j])
            for f in fields
            for j in range(len(mine["coef"]))
        ]
        cells += [
            (f"{key[0]}.{key[1]}.r2", mine["r2"], theirs["r2"]),
            (f"{key[0]}.{key[1]}.n", mine["n"], theirs["n"]),
        ]
        for name, a, b in cells:
            total += 1
            same = float(f"{float(a):.6g}") == float(b)
            agree += same
            worst = max(worst, abs(float(a) - float(b)) / max(abs(float(b)), 1e-300))
            if not same:
                off.append(name)
    text = (
        f"Against the independent check (`{CHECK_D_JSON.parent.name}/check_d_results.json`, its own `ols`; the two-regressor fits raw and relative on all dates and the raw one without the flagged dates): "
        f"{agree} of {total} numbers (coefficients, classical and HC1 standard errors, t, R², n) agree to the 6 significant digits it prints; largest relative difference {worst:.1e}."
    )
    if off:
        text += f" Not to the digit: {', '.join(off)}."
    return text, {"compared": total, "agree": agree, "worst_relative_difference": worst}


# ----------------------------------------------------------------------------- the part
class Book:
    """Formats a number for the Markdown and files its record at the same time, so that every
    number of the part has one (an id asked twice must carry the same value)."""

    def __init__(self, commit: str, source: str) -> None:
        self.records: dict[str, dict[str, Any]] = {}
        self.commit = commit
        self.source = source

    def num(
        self, id: str, quantity: str, value: float | None, se: float | None = None, *, digits: int = 4, spec: str | None = None,
        show_se: bool = True, study: bool = False, **kwargs: Any,
    ) -> str:  # fmt: skip
        kwargs.setdefault("budget", pc.BUDGETS["study"] if study else pc.BUDGETS["development"])
        kwargs.setdefault("commit", "" if study else self.commit)
        kwargs.setdefault("source", self.source)
        if study and not kwargs.get("notes"):
            kwargs["notes"] = STUDY_NOTE
        if se is None and not kwargs.get("notes"):
            kwargs["notes"] = (
                "no standard error: a count, a statistic of a fit or a stated value, not a Monte Carlo estimate"
            )
        rec = pc.record(id, SECTION, quantity, value, se, **kwargs)
        old = self.records.get(id)
        if old is not None and (old["value"], old["se"]) != (rec["value"], rec["se"]):
            raise ValueError(f"record {id} asked twice with different values")
        self.records[id] = rec
        if rec["value"] is None:
            return "n/a"
        if spec is not None:
            return format(rec["value"], spec)
        return pc.pm(rec["value"], rec["se"] if show_se else None, digits)


def md_table(headers: list[str], rows: list[list[str]], align: str = "") -> str:
    """A Markdown table; ``align`` gives ``l`` or ``r`` per column (default: first left, rest right)."""
    align = align or "l" + "r" * (len(headers) - 1)
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(":--" if a == "l" else "--:" for a in align) + "|",
    ]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def build_markdown(
    d: pd.DataFrame, masks: dict[str, pd.Series], sample_table: pd.DataFrame, qs: list[Quantity],
    table: dict[tuple[str, str], dict[str, dict[str, float]]], fits: dict[tuple[str, str], dict[str, Any]], agreement: tuple[str, dict[str, float]],
    book: Book, rows_path: Path,
) -> str:  # fmt: skip
    priced = d["priced"].astype(bool)
    md_qs = [q for q in qs if q.in_md]
    n_of = dict(zip(sample_table["sample"], sample_table["n"], strict=True))
    src = f"outputs/dispersion_lc/{rows_path.name} + outputs/dispersion/entries_3m.parquet (B1) + outputs/dispersion/model_s_3m.parquet → tables/C_history_summaries.csv"

    def stat_cell(q: Quantity, sample: str, stat: str, with_se: bool = True) -> str:
        cell = table[(q.key, sample)].get(stat)
        if cell is None:
            return "n/a"
        notes = STUDY_NOTE if q.study else ""
        if stat in ("mean", "pooled"):
            kind = (
                "mean over the dates of the sample; se across dates (sd/√n)"
                if stat == "mean"
                else "pooled: Σ numerator / Σ denominator over the dates of the sample; se across dates (linearised)"
            )
            if np.isfinite(cell["mc_se"]):
                notes = f"Monte Carlo standard error with the dates fixed, upper bound (seeds shared across dates): {cell['mc_se']:.2g}"
            definition = f"{q.definition}. {kind}"
        else:
            definition = f"{q.definition}. {stat} over the dates of the sample"
            notes = (
                notes + "; " if notes else ""
            ) + "an order statistic across dates: no standard error given"
        return book.num(
            f"C.hist.{q.key}.{stat}.{sample}", f"{q.label}: {stat}, {sample}", cell["value"], cell["se"] if stat in ("mean", "pooled") else None,
            digits=q.digits, show_se=with_se, study=q.study, definition=definition, n=cell["n"], notes=notes, source=src,
        )  # fmt: skip

    def n_cell(sample: str) -> str:
        return book.num(f"C.hist.n_dates.{sample}", f"dates in the sample {sample}", n_of[sample], spec=".0f", definition=str(sample_table.set_index("sample").loc[sample, "definition"]), n=int(n_of[sample]), unit="dates", source=src)  # fmt: skip

    commit = book.commit
    by_key = {q.key: q for q in qs}
    statuses = d["status"].value_counts().to_dict()
    lines: list[str] = []
    add = lines.append
    add("## C. History, 3m, development budget, decisions 1–2 on")
    add("")

    def count(tag: str, what: str, value: int) -> str:
        return book.num(f"C.hist.count.{tag}", what, value, spec=".0f", unit="dates", definition=what, n=int(value), source=f"{book.source} → tables/C_history_by_date.csv", notes="a count of dates")  # fmt: skip

    add(
        f"Source: `outputs/dispersion_lc/{rows_path.name}` (one row per date; {count('rows', 'dates of the table', len(d))} dates {d['date'].min()} to {d['date'].max()}: "
        f"{count('priced', 'priced dates (status not failed)', int(priced.sum()))} priced — {count('ok', 'dates with status ok', statuses.get('ok', 0))} `ok`, "
        f"{count('check', 'dates with status check (priced, a sanity check not passed)', statuses.get('check', 0))} `check` (priced, a sanity check not passed: the reason is in the per-date CSV) — "
        f"and {count('failed', 'dates with status failed', statuses.get('failed', 0))} failed), "
        f"joined on the date with `outputs/dispersion/entries_3m.parquet` (basket B1: one row per date, checked) and `outputs/dispersion/model_s_3m.parquet`. "
        f"Budget: {pc.BUDGETS['development']}; commit of the rows {commit}; calendar repair of the names' slices and the unscreened fallback on (decisions 1–2). "
        f"The row of {TODAY} in this table is at the development budget."
    )
    add("")
    add(
        "Files: `tables/C_history_by_date.csv` (per date), `tables/C_history_summaries.csv` (long: quantity, sample, statistic, value, se, mc_se, n), "
        "`tables/C_history_samples.csv`, `tables/C_history_regression.csv`, `figures/F1_forward_over_copula.pdf/.csv`, `figures/F2_calls_over_copula_by_strike.pdf/.csv`."
    )
    add("")
    # ---------------------------------------------------------------- C.0 samples
    add("### C.0 Samples")
    add("")
    rows = []
    desc = sample_table.set_index("sample")
    for key, label in BASE_SAMPLES:
        rows.append(
            [label, n_cell(key), n_cell(f"{key}_unflagged"), str(desc.loc[key, "definition"])]
        )
    add(md_table(["sample", "n", "n without flagged dates", "definition"], rows, "lrrl"))
    add("")
    lo, hi = float(desc.loc["clip_low", "clip_inner_max_to"]), float(
        desc.loc["clip_high", "clip_inner_max_from"]
    )
    lo_s = book.num(
        "C.hist.clip_tercile_bound.low",
        "clipped mass inside ±2.5 sd: 1/3 quantile on all priced dates",
        lo,
        definition="the 1/3 quantile of clip_inner_max over all priced dates (linear interpolation)",
        n=int(n_of["all"]),
        source=src,
    )
    hi_s = book.num(
        "C.hist.clip_tercile_bound.high",
        "clipped mass inside ±2.5 sd: 2/3 quantile on all priced dates",
        hi,
        definition="the 2/3 quantile of clip_inner_max over all priced dates (linear interpolation)",
        n=int(n_of["all"]),
        source=src,
    )
    add(
        f"A date is priced when its status is not `failed`; flagged when a name is kept unscreened (`n_names_unscreened > 0`). Terciles of the clipped mass inside ±2.5 sd (`clip_inner_max`) on all priced dates: "
        f"low ≤ {lo_s} < middle ≤ {hi_s} < high; the samples without flagged dates keep these bounds. Model S numbers are used on its converged dates only (the study's convention)."
    )
    add("")
    failed = d[d["status"] == "failed"]
    add(
        md_table(
            ["failed date", "reason"],
            [[r["date"], r["reason"]] for _, r in failed.iterrows()],
            "ll",
        )
        if len(failed)
        else "No failed date."
    )
    add("")
    add("Failed dates: no number of the run; they stay in the per-date CSV with the reason.")
    add("")
    flagged = d[priced & truth(d["flag_unscreened"])]
    if len(flagged):
        add(md_table(["flagged date", "name kept unscreened", "status", "LC/CC", "LC/copula"], [
            [r["date"], r["names_unscreened"], r["status"],
             book.num(f"C.hist.flagged.{r['date']}.lc_over_cc", "LC/CC on a flagged date", r["lc_over_cc"], r["lc_over_cc_se"], date=r["date"], definition=qs[0].definition, source="tables/C_history_by_date.csv"),
             book.num(f"C.hist.flagged.{r['date']}.lc_over_copula", "LC/copula on a flagged date", r["lc_over_copula"], r["lc_over_copula_se"], date=r["date"], definition=qs[1].definition, source="tables/C_history_by_date.csv")]
            for _, r in flagged.iterrows()
        ], "lllrr"))  # fmt: skip
        add("")
        add(
            "Flagged dates (decision 2): a name with no expiry passing the quote screen is kept unscreened; ± is the Monte Carlo standard error of the date."
        )
    else:
        add("No flagged date.")
    add("")
    extra_names = d[priced & (d["n_names_extrapolated"].astype(float) > 0)]
    extra_index = d[priced & truth(d["index_extrapolated"])]
    s_not = d[priced & ~truth(d["model_s_converged"])]
    names_list = ", ".join(
        f"{r.date} ({book.num(f'C.hist.n_names_extrapolated.{r.date}', 'names priced beyond their last listed expiry', r.n_names_extrapolated, spec='.0f', date=r.date, unit='names', definition='`n_names_extrapolated` of the row', source=book.source, notes='a count of names')})"
        for r in extra_names.itertuples()
    )  # fmt: skip
    low_n, high_n = (
        int(truth(d.loc[priced, c]).sum()) for c in ("flag_clip_low", "flag_clip_high")
    )
    add(
        f"Other flags of the per-date CSV: `n_names_extrapolated > 0` on {count('names_extrapolated', 'priced dates with n_names_extrapolated > 0', len(extra_names))} priced dates ({names_list or 'none'}); "
        f"`index_extrapolated` on {count('index_extrapolated', 'priced dates with index_extrapolated', len(extra_index))} ({', '.join(extra_index['date']) or 'none'}); "
        f"`flag_clip_low` (clipped mass at λ = 0 above {CLIP_FLAG_MASS:g}) on {count('flag_clip_low', f'priced dates with clip_low_inner_max > {CLIP_FLAG_MASS:g}', low_n)}, "
        f"`flag_clip_high` (at the cap, above {CLIP_FLAG_MASS:g}) on {count('flag_clip_high', f'priced dates with clip_high_inner_max > {CLIP_FLAG_MASS:g}', high_n)}. "
        f"Model S converged on {count('model_s_converged', 'dates of the table on which model S converged', int(truth(d['model_s_converged']).sum()))} of the table's {len(d)} dates "
        f"and on {int(n_of['S'])} of the {int(n_of['all'])} priced ones; not on {', '.join(s_not['date']) or 'none'} (no model S number there)."
    )
    add("")
    # ---------------------------------------------------------------- C.1 definitions
    add("### C.1 Definitions")
    add("")
    add(
        md_table(
            ["quantity", "definition"],
            [
                [
                    q.label,
                    q.definition + ("; no standard error (the study's numbers)" if q.study else ""),
                ]
                for q in md_qs
            ],
            "ll",
        )
    )
    add("")
    add(
        "LC = the calibrated local correlation model, CC = its constant-correlation companion, copula = the study's model (P_D), model S = the study's skewed model; "
        "D = Σ w_i |R_i − R̄|, V = Σ w_i (R_i − R̄)². In the tables below a mean's ± is the standard error across dates (sd/√n, no serial-correlation adjustment) and a pooled ratio's ± is its across-dates linearisation; "
        "the Monte Carlo error with the dates fixed is the column `mc_se` of the summaries CSV (an upper bound: the dates share the seeds)."
    )
    add("")
    # ---------------------------------------------------------------- C.2 full statistics
    for tag, sample, title in (
        ("a", "all", "all priced dates"),
        ("b", "all_unflagged", "without the flagged dates"),
    ):
        add(f"### C.2{tag} Full statistics, {title} (n = {n_cell(sample)})")
        add("")
        rows = []
        for q in md_qs:
            n_q = table[(q.key, sample)].get("mean", {}).get("n", 0)
            rows.append([q.label, f"{n_q:d}", stat_cell(q, sample, "mean"), *(stat_cell(q, sample, s) for s in ("q25", "median", "q75", "min", "max")), stat_cell(q, sample, "pooled") if q.num else ""])  # fmt: skip
        add(
            md_table(
                ["quantity", "n", "mean ± se", "q25", "median", "q75", "min", "max", "pooled ± se"],
                rows,
            )
        )
        add("")
        add(
            "Per-date quantities over the dates of the sample; pooled = Σ numerator / Σ denominator over the same dates; n is smaller for model S (converged dates only)."
        )
        add("")
        rows = []
        sub = d.loc[masks[sample]].set_index("date")
        for key in (
            "lc_over_cc",
            "lc_over_copula",
            "cc_over_copula",
            "s_over_copula",
            "kappa_lc",
            "EV_over_EQV_lc",
        ):
            q = by_key[key]
            v = sub[key].astype(float).dropna()
            cells = []
            for stat, at in (("min", v.idxmin()), ("max", v.idxmax())):
                flag = ", flagged" if bool(sub.loc[at, "flag_unscreened"]) else ""
                cells.append(
                    f"{stat_cell(q, sample, stat)} on {at} (`{sub.loc[at, 'status']}`{flag})"
                )
            rows.append([q.label, *cells])
        add(md_table(["quantity", "min: date (status)", "max: date (status)"], rows, "lll"))
        add("")
        add("The dates of the extremes of the table above.")
        add("")
    # ---------------------------------------------------------------- C.3-C.5 by sample
    for num, stat_title, maker, sentence, only_ratios in (
        ("C.3", "Mean ± se by sample", lambda q, s: stat_cell(q, s, "mean"), "Mean over the dates of each sample ± the standard error across dates.", False),
        ("C.4", "Pooled ratio by sample", lambda q, s: stat_cell(q, s, "pooled"), "Σ numerator / Σ denominator over the dates of each sample (the convention of the study's model S table, `scripts/disp_tables2.py::model_s_tables`) ± the across-dates linearised standard error.", True),
        ("C.5", "Median [q25, q75] by sample", lambda q, s: f"{stat_cell(q, s, 'median')} [{stat_cell(q, s, 'q25')}, {stat_cell(q, s, 'q75')}]", "Median and quartiles over the dates of each sample.", False),
    ):  # fmt: skip
        for tag, suffix, title in (
            ("a", "", "with all priced dates"),
            ("b", "_unflagged", "without the flagged dates"),
        ):
            add(f"### {num}{tag} {stat_title}, {title}")
            add("")
            keys = [f"{key}{suffix}" for key, _ in BASE_SAMPLES]
            rows = [["n dates", *(n_cell(k) for k in keys)]]
            for q in md_qs:
                if only_ratios and not q.num:
                    continue
                rows.append([q.label, *(maker(q, k) for k in keys)])
            add(md_table(["quantity", *(label for _, label in BASE_SAMPLES)], rows))
            add("")
            add(sentence + " Model S rows: its converged dates within the sample.")
            add("")
    # ---------------------------------------------------------------- check (b)
    add("### C.6 Check (b): LC/copula and CC/copula")
    add("")
    today = d[d["date"] == TODAY]
    keys_b = ("lc_over_cc", "lc_over_copula", "cc_over_copula")
    if len(today) == 1 and bool(today["priced"].iloc[0]):
        t = today.iloc[0]
        runs = [("today_dev", f"development budget, decisions 1–2 on (`{rows_path.name}`, commit {commit}, status `{t['status']}`)", {k: (float(t[k]), float(t[f"{k}_se"])) for k in keys_b}, {"source": f"{book.source} (row of {TODAY})", "notes": f"status of the row: {t['status']} ({t['reason']})"})]  # fmt: skip
        old = production_old_defaults(float(t["P_D"]), float(t["P_D_se"]))
        if old is not None:
            runs.append(("today_production_old_defaults", f"production budget, old defaults: no repair, no fallback (`{PRODUCTION_OLD.name}`, commit {old['commit']}, status `{old['status']}`)", old["values"], {"budget": old["budget"], "commit": old["commit"], "source": f"outputs/dispersion_lc/{PRODUCTION_OLD.name}", "notes": "the stopped production pass at the old defaults (no calendar repair, no unscreened fallback)"}))  # fmt: skip
        rows = []
        for tag, label, values, extra in runs:
            rows.append([label, *(book.num(f"C.b.{tag}.{k}", f"{by_key[k].label}, {TODAY}: {label}", values[k][0], values[k][1], digits=6, date=TODAY, definition=by_key[k].definition, **{"source": "tables/C_history_by_date.csv", **extra}) for k in keys_b)])  # fmt: skip
        rows.append(["the owner's arithmetic", "", *(book.num(f"C.b.owner.{k}", f"{by_key[k].label}, {TODAY}: the owner's arithmetic", OWNER_B[k], date=TODAY, definition="the owner's arithmetic of check (b), as written in the request (4 decimals)", budget="the owner's arithmetic (not computed here)", commit="", source="the owner's request, check (b)") for k in keys_b[1:])])  # fmt: skip
        for tag, label, values, extra in runs:
            cells = []
            for k in keys_b[1:]:
                diff, se = values[k][0] - OWNER_B[k], values[k][1]
                kw = {"source": "tables/C_history_by_date.csv", **extra, "notes": ""}
                diff_s = book.num(f"C.b.{tag}.{k}.minus_owner", f"{by_key[k].label}, {TODAY}: {label}, minus the owner's arithmetic", diff, se, digits=6, date=TODAY, definition=f"{by_key[k].label} of the run minus the owner's arithmetic; se: the Monte Carlo error of the run's value", **kw)  # fmt: skip
                z = book.num(f"C.b.{tag}.{k}.minus_owner_in_se", f"{by_key[k].label}, {TODAY}: {label}, that difference in standard errors", diff / se, spec="+.1f", date=TODAY, definition="the difference to the owner's arithmetic over the Monte Carlo standard error of the run's value", **kw)  # fmt: skip
                cells.append(f"{diff_s} ({z} se)")
            rows.append([f"minus the owner's arithmetic: {label.split(' (')[0]}", "", *cells])
        add(md_table([f"{TODAY}", "LC/CC", "LC/copula", "CC/copula"], rows))
        add("")
        today_sentence = f"on {TODAY} at the development budget (decisions 1–2 on) LC/copula = {rows[0][2]} and CC/copula = {rows[0][3]}"
        if old is not None:
            today_sentence += (
                f"; the production row at the old defaults gives {rows[1][2]} and {rows[1][3]}"
            )
        today_sentence += f"; the owner's arithmetic is {OWNER_B['lc_over_copula']:.4f} and {OWNER_B['cc_over_copula']:.4f}"
        s_today = (
            ""
            if bool(t["model_s_converged"])
            else f" Model S did not converge on {TODAY}: no S/copula for today."
        )
        add(
            f"± is the Monte Carlo standard error of the date (delta method with the copula's `P_D_se` for the ratios to the copula). Today's row of this section's table is at the development budget ({pc.BUDGETS['development']}), "
            f"status `{t['status']}` ({t['reason']}); the owner's arithmetic is given to 4 decimals."
            + s_today
        )
    else:
        add(f"Pending: the row of {TODAY} is not priced in `{rows_path.name}`; waits for that row.")
        today_sentence = f"the row of {TODAY} is pending"
    add("")
    rows = []
    for sample, label in (
        ("all", "all priced dates"),
        ("all_unflagged", "without flagged dates"),
        ("S", "∩ model S converged"),
        ("S_unflagged", "∩ model S converged, without flagged dates"),
    ):
        for key in ("lc_over_cc", "lc_over_copula", "cc_over_copula", "s_over_copula"):
            q = by_key[key]
            n_q = table[(q.key, sample)].get("mean", {}).get("n", 0)
            rows.append([label, q.label, f"{n_q:d}", stat_cell(q, sample, "mean"), stat_cell(q, sample, "q25"), stat_cell(q, sample, "median"), stat_cell(q, sample, "q75"), stat_cell(q, sample, "pooled")])  # fmt: skip
    add(
        md_table(
            ["sample", "quantity", "n", "mean ± se", "q25", "median", "q75", "pooled ± se"],
            rows,
            "llrrrrrr",
        )
    )
    add("")
    add(
        "Like for like across dates (the by-date values are the columns `lc_over_copula`, `cc_over_copula`, `s_over_copula` and `lc_over_cc` of `tables/C_history_by_date.csv`); ± across dates."
    )
    add("")

    def spread(key: str, sample: str) -> str:
        q = by_key[key]
        return f"{stat_cell(q, sample, 'mean')} (quartiles {stat_cell(q, sample, 'q25')}, {stat_cell(q, sample, 'median')}, {stat_cell(q, sample, 'q75')})"

    add(
        f"(b) as measured: {today_sentence}. Across the {int(n_of['all'])} priced dates the mean of LC/copula is {spread('lc_over_copula', 'all')} and of CC/copula {spread('cc_over_copula', 'all')}; "
        f"without the flagged dates (n = {int(n_of['all_unflagged'])}) {spread('lc_over_copula', 'all_unflagged')} and {spread('cc_over_copula', 'all_unflagged')}. "
        f"On the {int(n_of['S'])} dates where model S converged: S/copula {spread('s_over_copula', 'S')}, LC/CC {spread('lc_over_cc', 'S')}, LC/copula {spread('lc_over_copula', 'S')}, CC/copula {spread('cc_over_copula', 'S')}."
    )
    add("")
    # ---------------------------------------------------------------- check (d)
    add(
        "### C.7 Check (d): LC/CC − S/copula on the clipped mass and the basket part of the E[V] split"
    )
    add("")
    reg_src = "tables/C_history_regression.csv"
    for tag, sample, title in (
        ("a", "S", "all dates of the intersection"),
        ("b", "S_unflagged", "without the flagged dates"),
    ):
        add(f"**C.7{tag} {title}**")
        add("")
        rows = []
        for key, label, _regs in FITS:
            fit = fits[(sample, key)]
            for j, term in enumerate(fit["names"]):
                base = f"C.d.{sample}.{key}.{term}"
                what = f"check (d), {label}, {sample}: {term}"
                defn = f"OLS with an intercept of y = LC/CC − P_D_S/P_D on {label}; n = {fit['n']} dates"
                coef = book.num(f"{base}.coef", f"{what}, coefficient", fit["coef"][j], fit["se"][j], spec="+.5g", definition=defn + "; se: classical", n=fit["n"], source=reg_src, notes=f"HC1 standard error {fit['se_hc1'][j]:.4g}")  # fmt: skip
                se = f"{fit['se'][j]:.4g}"
                t = book.num(f"{base}.t", f"{what}, t (classical)", fit["t"][j], spec="+.2f", definition=defn + "; coefficient over its classical standard error", n=fit["n"], source=reg_src)  # fmt: skip
                hc = book.num(f"{base}.se_hc1", f"{what}, HC1 standard error", fit["se_hc1"][j], spec=".4g", definition=defn + "; HC1 (heteroskedasticity-consistent, n/(n − k)) standard error of the coefficient", n=fit["n"], source=reg_src)  # fmt: skip
                t_hc = book.num(f"{base}.t_hc1", f"{what}, t (HC1)", fit["t_hc1"][j], spec="+.2f", definition=defn + "; coefficient over its HC1 standard error", n=fit["n"], source=reg_src)  # fmt: skip
                if j == 0:
                    r2 = book.num(f"C.d.{sample}.{key}.r2", f"check (d), {label}, {sample}: R²", fit["r2"], spec=".4f", definition=defn + "; centred R²", n=fit["n"], source=reg_src)  # fmt: skip
                    n_s = book.num(f"C.d.{sample}.{key}.n", f"check (d), {label}, {sample}: n", fit["n"], spec=".0f", definition=defn, n=fit["n"], unit="dates", source=reg_src)  # fmt: skip
                else:
                    r2, n_s = "", ""
                rows.append([label if j == 0 else "", term, coef, se, t, hc, t_hc, r2, n_s])
        add(
            md_table(
                ["fit", "term", "coefficient", "se classical", "t", "se HC1", "t HC1", "R²", "n"],
                rows,
                "llrrrrrrr",
            )
        )
        add("")
        add(
            "OLS with an intercept of y = LC/CC − P_D_S/P_D across the dates where model S converged; `clip_inner_max` = clipped mass inside ±2.5 sd, "
            "`EV_basket_part` = E_LC[R̄²] − M_B^listed, `EV_basket_part_rel_MB` = that over M_B^listed; the regressors are the dates' Monte Carlo estimates."
        )
        add("")
    add(agreement[0])
    for tag, value in agreement[1].items():
        book.num(f"C.d.independent_check.{tag}", f"check (d) against the independent check: {tag.replace('_', ' ')}", value, definition="the fits of this section compared with the independent check's results file, number by number, at its 6 significant digits", source=str(CHECK_D_JSON), notes="a count or a difference of the comparison, not a Monte Carlo estimate")  # fmt: skip
    add("")
    # measured statements
    main, clip_only, bp_only = fits[("S", "rel")], fits[("S", "clip")], fits[("S", "bp_rel")]
    y_q = by_key["y_check_d"]
    add(
        f"(d) as measured, n = {main['n']}: mean y = {stat_cell(y_q, 'S', 'mean')} (median {stat_cell(y_q, 'S', 'median')}). "
        f"With both regressors the coefficient on the clipped mass is {main['coef'][1]:+.4f} (classical se {main['se'][1]:.4f}, t {main['t'][1]:+.2f}; HC1 se {main['se_hc1'][1]:.4f}, t {main['t_hc1'][1]:+.2f}) "
        f"and on the basket part over M_B^listed {main['coef'][2]:+.4f} (classical se {main['se'][2]:.4f}, t {main['t'][2]:+.2f}; HC1 se {main['se_hc1'][2]:.4f}, t {main['t_hc1'][2]:+.2f}), "
        f"intercept {main['coef'][0]:+.4f} (HC1 se {main['se_hc1'][0]:.4f}), R² {main['r2']:.4f}. "
        f"The clipped mass alone: R² {clip_only['r2']:.4f}; the basket part over M_B^listed alone: R² {bp_only['r2']:.4f}. "
        f"Without the flagged dates (n = {fits[('S_unflagged', 'rel')]['n']}): {fits[('S_unflagged', 'rel')]['coef'][1]:+.4f} and {fits[('S_unflagged', 'rel')]['coef'][2]:+.4f}, R² {fits[('S_unflagged', 'rel')]['r2']:.4f}."
    )
    add("")
    # ---------------------------------------------------------------- figures
    add("### Figures F1 and F2")
    add("")
    add(
        "- `figures/F1_forward_over_copula.pdf` (data: `figures/F1_forward_over_copula.csv`): by entry date, LC/copula, S/copula (converged dates) and the listed-variance forward over the copula √(EQV/EV); priced dates only (a failed date is a gap)."
    )
    add(
        f"- `figures/F2_calls_over_copula_by_strike.pdf` (data: `figures/F2_calls_over_copula_by_strike.csv`): calls over the copula's at 0.75, 1, 1.25 and 1.5 × the forward, LC and model S, mean and interquartile range over the same {int(n_of['S'])} dates (∩ model S converged); "
        "the CSV also carries LC on all priced dates, the medians and the pooled ratios."
    )
    return "\n".join(lines)


# ----------------------------------------------------------------------------- figures
def figure_f1(d: pd.DataFrame) -> None:
    cols = ["date", "status", "flag_unscreened", "model_s_converged", "lc_over_copula", "lc_over_copula_se", "s_over_copula", "listed_fwd_ratio"]  # fmt: skip
    frame = d[cols].copy()
    x = pd.to_datetime(frame["date"])
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.axhline(1.0, color="0.7", linewidth=0.6)
    ax.plot(x, frame["lc_over_copula"].astype(float), color="C0", linewidth=1.0, label="LC")
    ax.plot(x, frame["s_over_copula"].astype(float), color="C1", linewidth=1.0, label="model S")
    ax.plot(x, frame["listed_fwd_ratio"].astype(float), color="C2", linewidth=1.0, linestyle="--", label="listed-variance forward")  # fmt: skip
    ax.set_xlabel("entry date")
    ax.set_ylabel("forward / copula forward")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    pc.save_figure(fig, "F1_forward_over_copula", frame)
    plt.close(fig)


def figure_f2(table: dict[tuple[str, str], dict[str, dict[str, float]]]) -> None:
    rows = []
    for model, stem in (
        ("LC", "C_lc_over_copula"),
        ("model S", "C_S_over_copula"),
        ("CC", "C_cc_over_copula"),
    ):
        for sample in ("S", "all", "S_unflagged", "all_unflagged"):
            for m in CALL_TAGS:
                stats = table[(f"{stem}_{m}", sample)]
                if not stats:
                    continue
                rows.append({
                    "strike_multiple": int(m) / 100.0, "model": model, "sample": sample, "n": stats["mean"]["n"], "mean": stats["mean"]["value"],
                    "mean_se_across_dates": stats["mean"]["se"], "q25": stats["q25"]["value"], "median": stats["median"]["value"], "q75": stats["q75"]["value"],
                    "pooled": stats["pooled"]["value"], "plotted": sample == "S" and model != "CC",
                })  # fmt: skip
    frame = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    ax.axhline(1.0, color="0.7", linewidth=0.6)
    for model, color, marker, shift in (("LC", "C0", "o", -0.012), ("model S", "C1", "s", 0.012)):
        g = frame[frame["plotted"] & (frame["model"] == model)].sort_values("strike_multiple")
        at = g["strike_multiple"] + shift  # the two models side by side at each strike
        ax.vlines(at, g["q25"], g["q75"], color=color, linewidth=4.0, alpha=0.35)
        ax.plot(at, g["mean"], color=color, marker=marker, markersize=4, linewidth=1.0, label=f"{model}: mean (bar: interquartile range)")  # fmt: skip
    ax.set_xticks([int(m) / 100.0 for m in CALL_TAGS])
    ax.set_xlabel("strike / forward")
    ax.set_ylabel("call / copula call")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    pc.save_figure(fig, "F2_calls_over_copula_by_strike", frame)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--rows",
        type=Path,
        default=ROWS,
        help="the 3m development table (default: lcm_3m_dev_repair.parquet)",
    )
    parser.add_argument(
        "--no-status",
        action="store_true",
        help="do not append a line to STATUS.md (a rerun that changes nothing)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    frame = load(args.rows)
    priced = frame["priced"]
    budget = frame.loc[priced, ["n_particles", "n_paths", "companion_paths"]].drop_duplicates()
    if budget.to_numpy().tolist() != [[2e5, 2e5, 1e5]]:
        raise ValueError(
            f"the priced rows are not all at the development budget: {budget.to_dict('records')}"
        )
    commits = sorted(frame.loc[priced, "git_commit"].dropna().unique())
    commit = ", ".join(commits)
    d = by_date(frame)
    masks, sample_table = samples(d)
    qs = quantities()
    table, summary = all_summaries(d, masks, qs)
    fits, regression = regressions(d, masks)
    agreement = compare_check_d(fits, args.rows.name)
    LOG.info("%s", agreement[0])

    first = ["date", "status", "reason", "T", "half", "clip_tercile"]
    out = d[[*first, *(c for c in d.columns if c not in first and c != "priced")]]
    pc.save_table(out, "C_history_by_date")
    pc.save_table(summary, "C_history_summaries")
    pc.save_table(sample_table, "C_history_samples")
    pc.save_table(regression, "C_history_regression")
    figure_f1(d)
    figure_f2(table)
    book = Book(commit, f"outputs/dispersion_lc/{args.rows.name}")
    markdown = build_markdown(d, masks, sample_table, qs, table, fits, agreement, book, args.rows)
    pc.write_part(PART, list(book.records.values()), markdown)
    n_priced, n_failed = int(priced.sum()), int((~priced).sum())
    line = (
        f"C_history: parts/{PART}.md/.json ({len(book.records)} records), tables/C_history_by_date.csv, C_history_summaries.csv, C_history_samples.csv, "
        f"C_history_regression.csv, figures F1_forward_over_copula and F2_calls_over_copula_by_strike (pdf + csv) from {args.rows.name} "
        f"({n_priced} priced, {n_failed} failed, commit {commit}, development budget)"
    )
    if not args.no_status:
        pc.status(line)
    LOG.info(
        "section C written: %d dates, %d priced, %d records", len(d), n_priced, len(book.records)
    )


if __name__ == "__main__":
    main()
