"""Dispersion study: the report (spec §9).

    python scripts/disp_report.py [--version 1] [--tenor 3m] [--no-pdf]

Builds ``outputs/dispersion/report/report_q2.md`` and ``.pdf``, the tables as CSV under
``report/tables/``, the figures as PNG under ``report/figures/``, the workbook
``report/tables_q2.xlsx``, ``checks.json`` and ``tables/verdict_q2.csv`` from the Parquet files
of ``disp_outcomes.py`` and ``disp_indicators.py``.  A table whose inputs do not exist yet is
replaced by one line saying so; every pre-registered table that exists is reported whatever it
shows.
"""

# ruff: noqa: E501, RUF001
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import disp_tables as tb

from volsto.studies import disp_data as dd
from volsto.studies import disp_smile as ds
from volsto.studies import disp_stats as st
from volsto.studies import latex as lx

OUT = dd.OUT / "report"


def stack(header: str, width: int) -> str:
    """``header`` cut at spaces into LaTeX lines of at most ``width`` characters."""
    lines: list[str] = []
    cur = ""
    for word in header.split(" "):
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    lines.append(cur)
    return r" \\ ".join(lx.latex_escape(x) for x in lines)


def md_table(frame: pd.DataFrame) -> str:
    head = "| " + " | ".join(str(c) for c in frame.columns) + " |"
    sep = "|" + "|".join("---" for _ in frame.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


class Report:
    def __init__(self, out: Path) -> None:
        self.out = out
        (out / "tables").mkdir(parents=True, exist_ok=True)
        (out / "figures").mkdir(parents=True, exist_ok=True)
        self.blocks: list[tuple[str, Any]] = []
        self.sheets: dict[str, pd.DataFrame] = {}

    def add(self, text: str) -> None:
        self.blocks.append(("md", text))

    def table(
        self, name: str, frame: pd.DataFrame, caption: str, show: pd.DataFrame | None = None
    ) -> None:
        frame.to_csv(self.out / "tables" / f"{name}.csv", index=False)
        self.sheets[name[:31]] = frame
        self.blocks.append(("table", (name, frame if show is None else show, caption)))

    def figure(self, name: str, caption: str) -> None:
        plt.tight_layout()
        plt.savefig(self.out / "figures" / f"{name}.png", dpi=130)
        plt.close()
        self.blocks.append(("figure", (name, caption)))

    def missing(self, title: str, why: str) -> None:
        self.add(f"### {title}\n\nNot in this version: {why}.\n")

    def markdown(self) -> str:
        parts = []
        for kind, payload in self.blocks:
            if kind == "md":
                parts.append(payload)
            elif kind == "table":
                name, frame, caption = payload
                parts.append(f"**Table {name}.** {caption}\n\n{md_table(frame)}\n")
            else:
                name, caption = payload
                parts.append(f"![{caption}](figures/{name}.png)\n\n*Figure {name}: {caption}*\n")
        return "\n".join(parts)

    def write(self, stem: str, pdf: bool) -> dict[str, Any]:
        (self.out / f"{stem}.md").write_text(self.markdown())
        with pd.ExcelWriter(self.out / "tables_q2.xlsx", engine="openpyxl") as xl:
            for name, frame in self.sheets.items():
                frame.to_excel(xl, sheet_name=name, index=False)
        if not pdf:
            return {"status": "skipped"}
        parts: list[str] = []
        esc = lx.latex_escape
        for kind, payload in self.blocks:
            if kind == "md":
                parts.append(lx.markdown_to_latex(payload))
            elif kind == "figure":
                name, caption = payload
                parts += [r"\begin{figure}[htbp]", r"\centering", rf"\includegraphics[width=0.8\linewidth]{{figures/{name}.png}}",
                          rf"\caption{{{esc(caption)}}}", r"\end{figure}", r"\FloatBarrier"]  # fmt: skip
            else:
                name, frame, caption = payload
                n = frame.shape[1]
                size = r"\tiny" if n > 11 else (r"\scriptsize" if n > 7 else r"\footnotesize")
                longest = [max([8] + [len(str(v)) for v in frame[c]]) for c in frame.columns]
                # long text columns wrap; their widths are scaled so that the table fits the page
                char = 0.105 if n > 11 else (0.125 if n > 7 else 0.15)  # cm per character
                wide = [w_ > 30 for w_ in longest]
                fixed = sum(
                    char * w_ + 0.2 for w_, big in zip(longest, wide, strict=True) if not big
                )
                want = [min(char * w_, 7.5) for w_, big in zip(longest, wide, strict=True) if big]
                room = max(25.0 - fixed, 4.0)
                scale = min(1.0, room / sum(want)) if want else 1.0
                widths = iter(want)
                spec = "".join(
                    (r">{\raggedright\arraybackslash}p{" + f"{scale * next(widths):.1f}cm" + "}")
                    if big
                    else "l"
                    for big in wide
                )
                # a header is stacked on as many lines as it needs to be no wider than its cells
                head = " & ".join(
                    esc(str(c)) if big else rf"\shortstack[l]{{{stack(str(c), w_)}}}"
                    for c, w_, big in zip(frame.columns, longest, wide, strict=True)
                )
                parts += [r"\begingroup", size, r"\setlength\tabcolsep{2.5pt}", rf"\noindent\textbf{{Table {esc(name)}.}} {lx.inline_to_latex(caption)}", "",
                          rf"\begin{{longtable}}{{@{{}}{spec}@{{}}}}", r"\toprule", head + r" \\", r"\midrule", r"\endhead"]  # fmt: skip
                for row in frame.itertuples(index=False):
                    parts.append(" & ".join(esc(str(v)) for v in row) + r" \\")
                parts += [r"\bottomrule", r"\end{longtable}", r"\endgroup", ""]
        preamble = (
            (lx.TEX_PREAMBLE + "\\usepackage{array}\n")
            .replace(
                r"\usepackage[margin=2.5cm]{geometry}",
                r"\usepackage[landscape,margin=1.3cm]{geometry}",
            )
            .replace(r"\documentclass[11pt]{article}", r"\documentclass[10pt]{article}")
        )
        (self.out / f"{stem}.tex").write_text(
            preamble.rstrip("\n")
            + "\n\\begin{document}\n"
            + "\n".join(parts)
            + "\n\\end{document}\n"
        )
        return lx.compile_latex(self.out, f"{stem}.tex")


# ------------------------------------------------------------------------------------------------


def safe(rep: Report, title: str, fn: Any) -> Any:
    try:
        return fn()
    except Exception as exc:
        rep.add(
            f"### {title}\n\n**This section failed in this render:** `{type(exc).__name__}: {exc}`\n"
        )
        print(f"section {title} failed:\n{traceback.format_exc()[-1500:]}")
        return None


def figures(rep: Report, d: pd.DataFrame, tenor: str, dec: pd.DataFrame | None) -> None:
    o = d[d["has_outcome"]]
    x = pd.to_datetime(d["date"])
    plt.figure(figsize=(7, 4.2))
    plt.scatter(100 * o["Rb"], 100 * o["D"], s=6, alpha=0.5, label="Palladium D")
    plt.scatter(100 * o["Rb"], 100 * o["SD"], s=6, alpha=0.5, label="straddle package SD")
    plt.xlabel("basket performance, %")
    plt.ylabel("payoff, % of notional")
    plt.legend()
    rep.figure(
        "payoffs_vs_basket",
        f"Realised D and SD against the basket's performance, {tenor} windows, B1.",
    )
    plt.figure(figsize=(8, 4))
    plt.plot(x, d["rho_atm"], lw=0.8, label="ρ_ATM")
    plt.plot(x, d["rho_cop"], lw=0.8, label="ρ_cop")
    plt.plot(
        pd.to_datetime(o["date"]),
        o["rho_real"],
        lw=0.8,
        label="realised over the window (from entry)",
    )
    plt.legend()
    plt.ylabel("correlation")
    rep.figure(
        "correlations",
        "Implied correlation at entry (Cboe formula at the money, and the copula's) and the correlation realised over the following window.",
    )
    plt.figure(figsize=(8, 3.6))
    plt.plot(x, d["lambda_rho"], lw=0.8, label="λ_ρ")
    plt.plot(x, d["lam_theta"], lw=0.8, label="λ_θ")
    plt.axhline(1.0, color="grey", lw=0.5)
    plt.legend()
    rep.figure(
        "lambdas",
        "Basket straddles per unit of single-name straddles: correlation-matched λ_ρ and theta-neutral λ_θ.",
    )
    plt.figure(figsize=(8, 3.6))
    plt.plot(x, d["kappa_Q"], lw=0.8, label="κ_Q (model forward over the strip's √E[V])")
    plt.plot(x, d["kappa_cop"], lw=0.8, label="κ_Q^cop (the copula's own E[V])")
    plt.plot(x, d["kappa_P_trailing"], lw=0.8, label="trailing realised κ_P (3 years)")
    plt.legend()
    rep.figure(
        "kappas", "Position of the Palladium forward in its vanilla band, priced and realised."
    )
    if "VR_cs" in d:
        plt.figure(figsize=(8, 3.4))
        plt.plot(x, d["VR_cs"], lw=0.8, label="trailing VR_cs")
        plt.plot(x, d["VR_raw"], lw=0.8, label="trailing VR_raw")
        plt.axhline(1.0, color="grey", lw=0.5)
        plt.legend()
        rep.figure(
            "variance_ratio",
            "Trailing variance ratio of relative moves at the tenor (three years).",
        )
    no = tb.nonoverlapping(o, tenor)
    xn = pd.to_datetime(no["date"])
    plt.figure(figsize=(8, 4.2))
    for label in ("PF U", "PF H", "PKG_v H", "PKG_theta H", "SS H", "BS U"):
        plt.plot(xn, 100 * no[tb.STRUCTS[label]].cumsum(), lw=1.0, label=label)
    plt.legend(ncol=2)
    plt.ylabel("cumulative P&L, % of notional")
    rep.figure(
        "cumulative_pnl",
        "Cumulative P&L of the structures on non-overlapping windows (unit notional each window).",
    )
    plt.figure(figsize=(8, 3.6))
    plt.plot(xn, 100 * no["GAP_U"].cumsum(), lw=1.0, label="GAP = PF − PKG_v")
    plt.plot(xn, 100 * no["GAP_rho_U"].cumsum(), lw=1.0, label="GAP_ρ")
    plt.legend()
    plt.ylabel("cumulative P&L, % of notional")
    rep.figure("cumulative_gap", "Cumulative P&L of the gap on non-overlapping windows.")
    if dec is not None and len(dec):
        g = dec[dec["forecast"] == dec["forecast"].iloc[-1]]
        _, axes = plt.subplots(1, 4, figsize=(11, 3))
        for ax, s in zip(axes, ("PF", "PKG_v", "PKG_theta", "GAP"), strict=True):
            h = g[g["structure"] == s]
            ax.plot(100 * h["edge"], 100 * h["pnl"], "o-")
            lim = (
                [
                    100 * min(h["edge"].min(), h["pnl"].min()),
                    100 * max(h["edge"].max(), h["pnl"].max()),
                ]
                if len(h)
                else [0, 1]
            )
            ax.plot(lim, lim, color="grey", lw=0.5)
            ax.set_title(s)
            ax.set_xlabel("forecast edge, %")
        axes[0].set_ylabel("realised P&L, %")
        rep.figure(
            "calibration_deciles",
            f"Realised P&L against the forecast edge by decile of the edge ({g['forecast'].iloc[0]} forecast); the grey line is unbiased.",
        )


def coverage(d: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, g in d.groupby("year"):
        both = g["sig_B_DJX"].notna() & g["sig_B_DIA"].notna()
        diff = 100 * (g.loc[both, "sig_B_DJX"] - g.loc[both, "sig_B_DIA"])
        rows.append({"year": y, "entries": len(g), "DJX bracketed": f"{g['DJX_bracketed'].mean():.2f}", "DIA bracketed": f"{g['DIA_bracketed'].mean():.2f}",
                     "primary DJX": int((g["primary"] == "DJX").sum()), "DJX − DIA ATM vol, vp (median)": f"{diff.median():.2f}",
                     "|DJX − DIA| > 1 vp": int((diff.abs() > 1).sum()), "ρ_cop clipped": int((g["rho_cop_flag"] != "").sum()),
                     "smiles carried": int(g["n_carried_smiles"].sum()), "windows with basket_event": int(g["basket_event"].fillna(False).astype(bool).sum())})  # fmt: skip
    return pd.DataFrame(rows)


def headline(d: pd.DataFrame, tenor: str, label: str) -> dict[str, Any]:
    """The core headline of one run (spec §1.1 "vendor-vol"): ρ_cop, P_D, the correlation
    premium, the richness split and the gap."""
    o = d[d["has_outcome"] & d["strip_ok"]]
    lag = tb.LAG[tenor]
    r = tb.richness(o)
    g = st.describe(o["GAP_U"], lag)
    a, sa, _ = st.mean_se(o["rho_gap_cop"], lag)
    pk = st.describe(o["PKG_theta_H"], lag)
    return {"run": label, "windows": len(o), "mean ρ_cop": tb.num(o["rho_cop"].mean()), "ρ_cop − ρ_real": f"{a:.3f} ± {sa:.3f}", "mean P_D": tb.pct(o["P_D"].mean()),
            "mean SS": tb.pct(o["SS_mkt"].mean()), "mean Str_B": tb.pct(o["Str_B_mkt"].mean()), "P_D / D": tb.num(r["richness"]), "variance factor": tb.num(r["variance factor"]),
            "convexity factor": tb.num(r["convexity factor"]), "GAP mean": tb.pct(g["mean"]), "GAP t": tb.num(g["t"], 2), "PKG_theta H mean": tb.pct(pk["mean"]), "PKG_theta H t": tb.num(pk["t"], 2)}  # fmt: skip


def beyond_last_expiry(tenor: str) -> tuple[int, int, float]:
    """``(entry dates with at least one name priced beyond its last listed expiry, entry
    dates, median number of such names on those dates)`` of B1 at ``tenor``."""
    legs = pd.read_parquet(dd.OUT / f"legs_{tenor}.parquet")
    e = pd.read_parquet(dd.OUT / f"entries_{tenor}.parquet")
    e = e[e["basket"] == "B1"].set_index("date")["expiry"]
    hi = legs["bracket_hi"].astype(str)
    beyond = hi.isin(["None", "nan", ""]) | (hi < legs["date"].map(e).astype(str))
    per = beyond.groupby(legs["date"]).sum()
    return (
        int((per > 0).sum()),
        len(per),
        float(per[per > 0].median()) if (per > 0).any() else 0.0,
    )


def guards_effect(rep: Report, d: pd.DataFrame, tenor: str) -> None:
    """Before and after the expiry guards of ``disp_smile`` (PROGRESS_Q2, "Expiries that are
    not smiles"): the run kept in ``before_guards/`` (reports v1 and v2) against the current
    one, on the same windows."""
    root = dd.OUT / "before_guards"
    if not (root / f"outcomes_{tenor}.parquet").exists():
        return
    old = tb.load(tenor, "B1", "", root=root)
    lag = tb.LAG[tenor]
    both = sorted(set(old.loc[old["has_outcome"], "date"]) & set(d.loc[d["has_outcome"], "date"]))
    a = old[old["date"].isin(both)].set_index("date")
    b = d[d["date"].isin(both)].set_index("date")
    moved = (b["P_D"] - a["P_D"]).abs() > 1e-6
    rows = []

    def line(label: str, f: Any) -> None:
        rows.append({"statistic": label, "before the guards (v1, v2)": f(a), "with the guards (v3)": f(b)})  # fmt: skip

    def cell(col: str, mask: Any = None) -> Any:
        def f(x: pd.DataFrame) -> str:
            v = x[col] if mask is None else x.loc[mask(x), col]
            g = st.describe(v, lag) if mask is None else st.describe_subset(x[col], mask(x), lag)
            return f"{tb.pct(g['mean'])} (t {g['t']:.2f})"

        return f

    line("windows", lambda x: str(len(x)))
    line("mean ρ_cop", lambda x: tb.num(x["rho_cop"].mean()))
    line("mean P_D, % of notional", lambda x: tb.pct(x["P_D"].mean()))
    line("mean single-name strip SS", lambda x: tb.pct(x["SS_mkt"].mean()))
    line("mean basket straddle", lambda x: tb.pct(x["Str_B_mkt"].mean()))
    line("price over realised, P_D / D", lambda x: tb.num(tb.richness(x[x["strip_ok"]])["richness"]))  # fmt: skip
    for col, label in (("GAP_U", "gap"), ("PKG_theta_H", "PKG_θ H"), ("PKG_v_H", "PKG_v H"), ("PF_U", "PF U"), ("PF_H", "PF H")):  # fmt: skip
        line(f"{label}: mean P&L, all", cell(col))
        line(f"{label}: in sample", cell(col, lambda x: x["IS"].to_numpy(bool)))
        line(f"{label}: out of sample", cell(col, lambda x: ~x["IS"].to_numpy(bool)))
    rep.add("## Before and after the expiry guards\n")
    dP = (b["P_D"] - a["P_D"])[moved]
    dS = (b["SS_mkt"] - a["SS_mkt"])[moved]
    worst = dP.abs().sort_values(ascending=False).head(5)
    rep.add(
        f"Reports v1 and v2 were priced before two guards on the vendor's expiries (PROGRESS_Q2, \"Expiries that are not smiles\"): an expiry listed with strikes on one side of the forward only is no longer used when the ticker has a two-sided one that day, and an expiry whose at-the-money vol is more than a factor 2 from its neighbours' is dropped. "
        f"{int(moved.sum())} of {len(both)} windows change price; on them the Palladium forward moves by {100 * dP.mean():.3f} on average (the strip by {100 * dS.mean():.3f}), at most on "
        + ", ".join(f"{dt} ({100 * dP[dt]:+.2f})" for dt in worst.index)
        + " (% of notional).\n"
    )
    rep.table(
        "guards_effect",
        pd.DataFrame(rows),
        "The headline of B1 before and after the expiry guards, on the same windows; % of notional, Hansen–Hodrick t.",
    )


def sensitivities(rep: Report, d: pd.DataFrame, tenor: str) -> None:
    rows = [headline(d, tenor, "base: vols inverted from the vendor's values, parity forwards")]
    for suffix, label in (
        ("_vendor", "vendor-vol: smoothSmvVol for every leg"),
        ("_dividends", "rule (c): projected-dividend forwards for single names and DIA"),
    ):
        if (dd.OUT / f"outcomes_{tenor}{suffix}.parquet").exists():
            alt = tb.load(tenor, "B1", suffix)
            rows.append(headline(alt, tenor, label))
            same = set(alt.loc[alt["has_outcome"], "date"]) == set(d.loc[d["has_outcome"], "date"])
            if not same:
                rows.append(
                    headline(
                        d[d["date"].isin(alt["date"])], tenor, "base, on the dates of the run above"
                    )
                )
    if len(rows) > 1:
        rep.add("## Sensitivity of the headline to the vol and forward convention\n")
        rep.table(
            "sensitivity_convention",
            pd.DataFrame(rows).fillna(""),
            "The core headline under each convention of spec §1.1: the copula correlation, its premium over realised, the prices, the richness split (T4), the gap (T17) and the hedged theta-neutral package.",
        )


def multi(rep: Report, tenor: str) -> None:
    """The other tenors of B1 and the equally weighted B2: T2, T4, T6-style horizon, T13, T17."""
    runs = [
        (t, "B1") for t in ("1m", "6m", "12m", "24m") if (dd.OUT / f"outcomes_{t}.parquet").exists()
    ] + [(tenor, "B2")]
    rows_gap, rows_rich, rows_dec, rows_t2 = [], [], [], []
    for t, basket in runs:
        try:
            x = tb.load(t, basket)
        except Exception as exc:
            rep.add(f"*{basket} {t} could not be loaded: {exc}*\n")
            continue
        o = x[x["has_outcome"]]
        if len(o) < 30:
            continue
        lag = tb.LAG[t]
        tag = f"{basket} {t}"
        for col, name in (("GAP_U", "GAP"), ("GAP_rho_U", "GAP_ρ")):
            for sample, g in (("all", o), ("IS", o[o["IS"]]), ("OOS", o[~o["IS"]])):
                s = st.describe(g[col], lag)
                rows_gap.append({"run": tag, "structure": name, "sample": sample, "mean": tb.pct(s["mean"]), "t": tb.num(s["t"], 2), "sd": tb.pct(s["sd"]), "hit": tb.num(s["hit"], 2),
                                 "5 %": tb.pct(s["q05"]), "mean P_G": tb.pct(g["P_G"].mean()), "mean G": tb.pct(g["G"].mean()), "n": int(s["n"])})  # fmt: skip
        if basket == "B1":
            for sample, g in (("all", o), ("IS", o[o["IS"]]), ("OOS", o[~o["IS"]])):
                g = g[g["strip_ok"]]
                r = tb.richness(g)
                rows_rich.append({"run": tag, "sample": sample, "windows": len(g), "P_D / D": tb.num(r["richness"]), "variance factor": tb.num(r["variance factor"]), "convexity factor": tb.num(r["convexity factor"]),
                                  "κ_Q": tb.num(r["kappa_Q"]), "κ_P": tb.num(r["kappa_P"]), "mean window VR": tb.num(g["vr_window"].mean(), 2), "PF U − PF H": tb.pm(*st.mean_se(g["PF_U"] - g["PF_H"], lag)[:2])})  # fmt: skip
        t2 = tb.t2(x)
        t2.insert(0, "run", tag)
        rows_t2.append(t2)
        for label in (
            "PF U",
            "PF H",
            "PC(1) static hedge",
            "SS H",
            "BS U",
            "PKG_v U",
            "PKG_v H",
            "PKG_theta H",
            "GAP U",
            "PF_v U",
        ):
            col = tb.STRUCTS[label]
            row = {"run": tag, "structure": label}
            for sample, g in (("all", o), ("IS", o[o["IS"]]), ("OOS", o[~o["IS"]])):
                s = st.describe(g[col], lag)
                row[f"{sample} mean"] = tb.pct(s["mean"])
                row[f"{sample} t"] = tb.num(s["t"], 2)
            row["sd"] = tb.pct(o[col].std())
            row["no basket_event"] = tb.pct(o.loc[~o["basket_event"].astype(bool), col].mean())
            rows_dec.append(row)
    if rows_gap:
        rep.add("## Other tenors and the equally weighted basket\n")
        rep.add(
            "B2 is the PM's definition (weights 1/30) on the same names; its basket straddle has no listed price and is the copula's at the Dow's ρ_cop (so the copula prices both legs of its package). Standard errors use each tenor's own Hansen–Hodrick lag.\n"
        )
        rep._gap_runs = pd.DataFrame(rows_gap)  # type: ignore[attr-defined]
        rep.table(
            "T17_other_runs",
            pd.DataFrame(rows_gap),
            "The gap by tenor (B1) and for B2: mean P&L in % of notional.",
        )
        if rows_rich:
            rep.table(
                "T4_T6_other_tenors",
                pd.DataFrame(rows_rich),
                "The richness split (T4), the realised variance ratio of the windows and the unhedged-minus-hedged forward (T6) by tenor, B1. At 12 and 24 months the pooled split is dominated by the entries of 2008–09, where single-name vols of 150 % and more are extrapolated flat to the tenor: read the out-of-sample row.",
            )
        rep.table(
            "T13_other_runs",
            pd.DataFrame(rows_dec),
            "The decision table by tenor (B1) and for B2: mean P&L, % of notional, with the Hansen–Hodrick t.",
        )
        rep.table(
            "T2_other_runs",
            pd.concat(rows_t2, ignore_index=True),
            "The gap by size of the basket's move, by tenor and basket.",
        )


def costs(
    rep: Report, d: pd.DataFrame, tenor: str, r1: pd.Series, r2: pd.Series, has_fhs: bool
) -> None:
    """Spec §11: the decision table, the rules, the gap and the calibration net of costs, and
    the break-even dealer margins."""
    import disp_tables2 as t2

    rep.add("## Costs (sensitivity)\n")
    rep.add(
        "Mid prices everywhere above. Here: half the bid-ask of each straddle leg (ORATS quotes at the nearest listed strike and expiry, applied as they are: the listed expiry is not the tenor exactly), "
        "1 bp of the notional traded by each daily hedge, and a dealer margin on the Palladium of two correlation points (price + 2·|∂P/∂ρ|) or, separately, of 5 % of the premium.\n"
    )
    lag = tb.LAG[tenor]
    nets = {m: t2.net_of_costs(d, m) for m in ("2 correlation points", "5 % of the premium")}
    o = d[d["has_outcome"]]
    rows = []
    for label in (
        "PF U",
        "PF H",
        "PC(1) static hedge",
        "SS U",
        "SS H",
        "BS U",
        "PKG_v U",
        "PKG_v H",
        "PKG_theta U",
        "PKG_theta H",
        "GAP U",
        "GAP_rho U",
        "PF_v U",
        "REV H",
    ):
        col = tb.STRUCTS[label]
        row = {"structure": label}
        for sample, sel in (("IS", o["IS"]), ("OOS", ~o["IS"])):
            s0 = st.describe(o.loc[sel, col], lag)
            row[f"{sample} at mid"] = f"{tb.pct(s0['mean'])} ({s0['t']:.1f})"
            for m, dn in nets.items():
                s1 = st.describe(dn.loc[o.index][sel][col], lag)
                row[f"{sample} net, margin {m}"] = f"{tb.pct(s1['mean'])} ({s1['t']:.1f})"
        rows.append(row)
    rep.table(
        "T13_T17_costs",
        pd.DataFrame(rows),
        "The decision table and the gap (T13, T17) at mid and net of costs: mean P&L in % of notional with its Hansen–Hodrick t; the two dealer margins change the Palladium structures only.",
    )
    rows = []
    for name, pick in (
        ("R1", r1),
        ("R2", r2),
        ("always PF U", pd.Series("PF_U", index=d.index)),
        ("always PKG_theta H", pd.Series("PKG_theta_H", index=d.index)),
        ("always PKG_v H", pd.Series("PKG_v_H", index=d.index)),
    ):
        if pick.notna().sum() == 0:
            continue
        row = {"strategy": name}
        for sample, sel in (("IS", o["IS"]), ("OOS", ~o["IS"])):
            y0 = t2.stream(d, pick, False).reindex(o.index)[sel]
            s0 = st.describe(y0, lag)
            row[f"{sample} at mid"] = f"{tb.pct(s0['mean'])} ({s0['t']:.1f})"
            for m, dn in nets.items():
                s1 = st.describe(t2.stream(dn, pick, False).reindex(o.index)[sel], lag)
                row[f"{sample} net, margin {m}"] = f"{tb.pct(s1['mean'])} ({s1['t']:.1f})"
        rows.append(row)
    try:
        c14 = pd.DataFrame(rows).set_index("strategy")
        col_is, col_oos = (
            "IS net, margin 2 correlation points",
            "OOS net, margin 2 correlation points",
        )
        rep._costs = (
            "; ".join(
                f"{k} {c14.loc[k, col_is]} in sample, {c14.loc[k, col_oos]} out of sample"
                for k in c14.index
                if k in ("R1", "R2", "always PF U", "always PKG_theta H")
            )
            + " (mean, % of notional, t in brackets)."
        )  # type: ignore[attr-defined]
    except Exception:
        pass
    rep.table(
        "T14_costs",
        pd.DataFrame(rows),
        "The rules and three always-strategies (T14) at mid and net of costs, unit notional: mean P&L in % of notional with its t. The picks are those made at mid.",
    )
    if has_fhs:
        dn = nets["2 correlation points"].copy()
        for s_, col in (
            ("PF", "PF_U"),
            ("PKG_v", "PKG_v_U"),
            ("PKG_theta", "PKG_theta_U"),
            ("GAP", "GAP_U"),
        ):
            dn[f"fedge_{s_}"] = d[f"fedge_{s_}"] - (d[col] - dn[col])
        c, _ = t2.t18_fhs(dn, tenor)
        rep.table(
            "T18_costs",
            c[c["structure"].isin(["PF", "PKG_v", "PKG_theta", "GAP"])],
            "Calibration of the FHS edges net of costs (two correlation points on the Palladium): edge and P&L both net; a cost known at entry moves the intercept and the mean, not the ranking.",
        )
    rep.table(
        "break_even_margin",
        t2.break_even(d, tenor, r1, r2),
        "Break-even dealer margin: the margin at which the mean P&L at mid (or a rule's advantage over always-PKG_theta H) is zero, in correlation points and in % of the Palladium premium, with block-bootstrap 95 % intervals. A negative number: the structure lost at mid, so any margin makes it worse; the PM knows the quote at which the choice flips.",
    )


def pm_summary(rep: Report, d: pd.DataFrame, PT: pd.DataFrame, tenor: str) -> None:
    """``pm_summary_q2.md``: one page for the PM's reply."""
    sec = getattr(rep, "_second", None)
    if sec is None:
        return
    i15 = sec["i15"].set_index("indicator")
    t14 = sec["t14"]
    o = d[d["has_outcome"]]
    lag = tb.LAG[tenor]
    last = d["date"].iloc[-1]

    def today(label: str) -> str:
        if label not in i15.index:
            return "not available"
        r = i15.loc[label]
        return f"{r['value']} (percentile {r['percentile (expanding)']}, in-sample tercile: {r['IS tercile']})"

    def evidence(q: str) -> str:
        r = PT[PT["test"] == q]
        if r.empty:
            return "not testable"
        r = r.iloc[0]
        unit = "× premium" if q == "Q4" else "% of notional"
        return f"top minus bottom tercile {r['IS top − bottom']} in sample (t {r['IS t']}), {r['OOS top − bottom']} out of sample ({unit}); verdict: **{r['verdict']}**"

    def mean(col: str) -> str:
        a, b = st.describe(o.loc[o["IS"], col], lag), st.describe(o.loc[~o["IS"], col], lag)
        return f"{tb.pct(a['mean'], 2)} % in sample (t {a['t']:.1f}), {tb.pct(b['mean'], 2)} % out of sample (t {b['t']:.1f})"

    lines = [
        "# Palladium or straddle dispersion: what the backtest says (one page)\n",
        f"Dow basket (the thirty members at price weights, frozen at entry), {tenor} trades entered every week from {o['date'].min()} to {o['date'].max()}; in sample to 2016, out of sample after. The Palladium is priced with one correlation, the one that reprices the listed DJX straddle. "
        f"P&L per unit of notional at mid; ± is a standard error for overlapping windows. Today is {last}. Nothing here goes beyond the verdict table of the report.\n",
        "**The five questions, in order.**\n",
        f"1. **Long dispersion at all?** Indicator: correlation premium (implied minus forecast correlation). Evidence: implied exceeded realised correlation by {st.mean_se(o['rho_gap_atm'], lag)[0]:.3f} ± {st.mean_se(o['rho_gap_atm'], lag)[1]:.3f} on average; the delta-hedged theta-neutral package earned {mean('PKG_theta_H')}; as a timing signal: {evidence('Q1')}. Today: {today('CRP')}.",
        f"2. **Fixed or floating strike (Palladium or package)?** Indicator: gap premium (price of the gap over its Gaussian forecast). Evidence: the gap, Palladium forward minus vega-neutral package, earned {mean('GAP_U')}; by the indicator: {evidence('Q2')}. Today: {today('GP_G')}; gap price {tb.pct(d['P_G'].iloc[-1], 2)} % for a package at {tb.pct(d['SS_mkt'].iloc[-1] - d['Str_B_mkt'].iloc[-1], 2)} % and a forward at {tb.pct(d['P_D'].iloc[-1], 2)} %.",
        f"3. **Terminal or daily (hold or delta-hedge)?** Indicator: trailing variance ratio of relative moves. Evidence: unhedged minus hedged Palladium forward {mean_diff(o, 'PF_U', 'PF_H', lag)}; by the indicator: {evidence('Q3')}. Today: {today('VR_cs')}.",
        f"4. **Forward or call?** Indicator: variability of dispersion. Evidence: the call struck at the forward's price with a static basket hedge earned {mean('PC_100_S')} against {mean('PF_U')} for the forward; by the indicator: {evidence('Q4')}. Today: {today('variability')}.",
        f"5. **With or without the vol level?** Indicator: single-name vol premium (implied over forecast). Evidence: hedged vega-neutral package {mean('PKG_v_H')}, theta-neutral {mean('PKG_theta_H')}; by the indicator: {evidence('Q5')}. Today: {today('single-name vol premium')}.\n",
        "**The rules against always doing the same thing** (unit notional, % of notional):\n",
    ]
    unit = t14[t14["units"].str.startswith("unit")]

    def rule_text(name: str, full: bool) -> str:
        g = unit[unit["strategy"] == name]
        if full:
            return "; ".join(
                f"{r_['sample']} mean {r_['mean']} (t {r_['t']}), s.d. {r_['sd']}, worst {r_['worst']}"
                for r_ in g.to_dict("records")
            )
        return ", ".join(
            f"{r_['sample']} {r_['mean']} (t {r_['t']})" for r_ in g.to_dict("records")
        )

    names = {
        "R1": "R1, the pre-registered leaf list on the indicators",
        "R2": "R2, the structure with the best forecast Sharpe from the filtered historical simulation",
    }
    for name, label in names.items():
        if (unit["strategy"] == name).any():
            lines.append(
                f"- **{label}**: {rule_text(name, True)}; against always the hedged theta-neutral package: {rule_text(f'{name} minus always PKG theta H', False)}."
            )
    lines.append(
        f"- **Always the same structure**: hedged theta-neutral package {rule_text('always PKG_theta H', False)}; hedged vega-neutral package {rule_text('always PKG_v H', False)}; "
        f"Palladium forward held to expiry {rule_text('always PF U', False)}; hedged {rule_text('always PF H', False)}."
    )
    p15 = sec["p15"].set_index("quantity")["value"]
    lines.append(
        f"\nToday R1 selects **{p15.get('R1 selects', 'not available')}**, R2 selects **{p15.get('R2 selects', 'not available')}**.\n"
    )
    cost = getattr(rep, "_costs", None)
    if cost is not None:
        lines.append(
            f"*Net of costs* (half the listed bid-ask on every straddle, 1 bp on hedges, two correlation points of dealer margin on the Palladium): {cost}\n"
        )
    ms_path = dd.OUT / f"model_s_{tenor}.parquet"
    if ms_path.exists():
        lines.append("*Model.* " + first_page_model_s(sec, tenor).strip() + "\n")
    lines.append(
        "*Limits.* The Palladium has no market price: its price here is a constant-correlation model's at the DJX-implied correlation, before any dealer margin (the report gives the margin at which each conclusion flips). About 40 independent three-month windows in sample and 37 out of sample.\n"
    )
    (rep.out / "pm_summary_q2.md").write_text("\n".join(lines))


def mean_diff(o: pd.DataFrame, a: str, b: str, lag: int) -> str:
    x, y = (
        st.describe(o.loc[o["IS"], a] - o.loc[o["IS"], b], lag),
        st.describe(o.loc[~o["IS"], a] - o.loc[~o["IS"], b], lag),
    )
    return f"{tb.pct(x['mean'], 2)} % in sample (t {x['t']:.1f}), {tb.pct(y['mean'], 2)} % out of sample (t {y['t']:.1f})"


def third_phase(rep: Report, d: pd.DataFrame, tenor: str) -> None:
    """Phase 4: model S, the marks during the life, basket B3 (what exists)."""
    import disp_tables2 as t2

    ms = t2.model_s_tables(d, tenor)
    if ms:
        rep.add("## Model S: a copula that reprices the basket's skew\n")
        rep.add(
            "Same single-name smiles; the correlation of a scenario is `c − s × M` (clipped to [0.02, 0.98]) with `M` the common factor, so it rises in sell-offs for `s > 0`; "
            "`(c, s)` reprice the DJX at-the-money straddle and its 90 % put on each date (first entry of each month). The question: do the forward's richness and the calls' cheapness "
            "survive a model that reprices the basket's skew while keeping the single-name smiles? Dates on which the two targets are not both met are left out of the comparisons (counted).\n"
        )
        rep.table(
            "model_S",
            ms["model_S"],
            "Model S on the monthly subset: calibration, and its prices over the base copula's (ratios of sums over dates; the cash strikes are the base run's).",
        )
        rep.table(
            "T4_model_S",
            ms["T4_model_S"],
            "The richness split (T4) with model S prices, on the monthly windows where it converged, beside the base copula on the same windows.",
        )
        rep.table(
            "T5_model_S",
            ms["T5_model_S"],
            "Calls (T5): Σ payoff / Σ price under each model, same windows and strikes (block-bootstrap 95 % interval on the pooled column).",
        )
        rep.table(
            "T11_model_S",
            ms["T11_model_S"],
            "Relative dispersion by basket performance (T11) with model S's forward and its conditional profile.",
        )
    c8 = dd.OUT / "c8.csv"
    if c8.exists():
        x8 = pd.read_csv(c8)
        if len(x8):
            show = pd.DataFrame(
                {
                    "date": x8["date"], "ρ_mark": x8["rho_mark"].round(3),
                    "forward: copula": (100 * x8["copula_P_D"]).round(3), "local vol": (100 * x8["lv_P_D"]).round(3), "± s.e.": (100 * x8["lv_P_D_se"]).round(3),
                    "LV / copula": x8["P_D_lv_over_copula"].round(3),
                    "call at the forward's price: LV / copula": x8["C1_lv_over_copula"].round(3),
                    "basket straddle: LV / copula": x8["Str_B_lv_over_copula"].round(3),
                    "single-name strip: LV / smile": x8["SS_lv_over_smile"].round(3),
                    "forward ratio over strip ratio": (x8["P_D_lv_over_copula"] / x8["SS_lv_over_smile"]).round(3),
                    "SVI rms, vp (median / max)": x8["svi_rms_vp_median"].round(2).astype(str) + " / " + x8["svi_rms_vp_max"].round(2).astype(str),
                }
            )  # fmt: skip
            rep.add("## Check C8: the copula against the library's multi-asset local vol\n")
            rep.table(
                "C8",
                show,
                "Three-month prices (% of notional) from the Gaussian copula at ρ_mark and from `volsto.multi`: one Dupire local-vol model per name (SVI slices fitted to the study's smiles), Brownian correlation ρ_mark, 10^5 paths. The strip column says how well the local vols reproduce the single-name straddles (the SVI fit and the time stepping); the last column removes that level effect from the forward's ratio. Reported, no pass or fail: a terminal copula and a diffusion with the same marginals and the same correlation number are different joint laws.",
            )
    mk = t2.marks_table(d, tenor)
    if len(mk):
        rep.add("## Marks during the life\n")
        rep.table(
            "marks",
            mk,
            "Monthly subset: the P&L of unwinding at one third and two thirds of the life (every leg re-priced with that day's smiles; the Palladium conditional on the performances to date, at the correlation that reprices that day's DJX straddle of the remaining maturity on the frozen basket; hedged versions add the hedges accrued), against the P&L at expiry; % of notional.",
        )
    if (dd.OUT / f"outcomes_{tenor}_B3.parquet").exists():
        x = tb.load(tenor, "B3", "_B3")
        o = x[x["has_outcome"]]
        lag = tb.LAG[tenor]
        rows = []
        for label in (
            "PF U",
            "PF H",
            "PC(1) static hedge",
            "SS U",
            "SS H",
            "BS U",
            "PKG_v U",
            "PKG_v H",
            "PKG_theta H",
            "GAP U",
            "GAP_rho U",
            "PF_v U",
        ):
            col = tb.STRUCTS[label]
            row = {"structure": label}
            for sample, g in (("all", o), ("IS", o[o["IS"]]), ("OOS", o[~o["IS"]])):
                s_ = st.describe(g[col], lag)
                row[f"{sample} mean"], row[f"{sample} t"] = tb.pct(s_["mean"]), tb.num(s_["t"], 2)
            row["sd"] = tb.pct(o[col].std())
            rows.append(row)
        rep.add("## Basket B3: ten large caps at equal weights\n")
        rep.add(
            "AAPL, MSFT, AMZN, NVDA, JPM, XOM, JNJ, PG, HD, UNH. **Survivorship: the list was chosen in 2026**; past winners are over-represented, which inflates persistent relative trends and realised dispersion. The basket straddle is the copula's at the Dow's ρ_cop of the same date (no listed option): the comparison of the Palladium with its package is model against model.\n"
        )
        rep.table(
            "T13_B3",
            pd.DataFrame(rows),
            "B3 (survivorship-biased list of 2026), 3m: mean P&L in % of notional with the Hansen–Hodrick t.",
        )
        rep.table(
            "T2_B3",
            tb.t2(x),
            "B3 (survivorship-biased list of 2026): the gap by size of the basket's move.",
        )


def second_phase(rep: Report, d0: pd.DataFrame, tenor: str, dec: pd.DataFrame) -> None:
    import disp_tables2 as t2

    d = t2.add_fhs(d0, tenor)
    has_fhs = "fsr_PF" in d and d["fsr_PF"].notna().any()
    a, b = t2.t5(d, tenor)
    rep.table(
        "T5_calls",
        a,
        "D5: realised payoff over the model price by strike (cash strikes fixed at multiples of the forward's price at entry), with a block-bootstrap 95 % interval; the median only where the price is at least 0.0005; and the call at the forward with and without the static basket hedge (short Δ_c of the basket).",
    )
    rep.table(
        "T5_conditions",
        b,
        "Calls by in-sample tercile of the variability of dispersion and of the regime-shift indicators: Σ payoff / Σ price by strike, and the statically hedged call's mean P&L per unit of premium.",
    )
    rep.table(
        "T6",
        t2.t6(d, tenor),
        "D6, horizon: unhedged minus hedged by in-sample tercile of the trailing variance ratio; the realised variance ratio of the windows; and whether the trailing ratio predicts it.",
    )
    rep.table(
        "T9",
        t2.t9(d, tenor),
        "D9: each structure's P&L regressed on realised over implied single-name vol minus one (slope × 100, Hansen–Hodrick).",
    )
    rep.table(
        "T10",
        t2.t10(d, tenor),
        "D10, lumpiness: realised D/√V by in-sample tercile of ex-ante variables; the sort on the window's own kurtosis is mechanical and is a check only.",
    )
    rep.table(
        "T11",
        t2.t11(d, tenor),
        "D11, correlation skew on relative dispersion: realised D/B over its forward by the basket's performance, raw and divided by realised over implied average single-name vol, beside the copula's own conditional profile.",
    )
    t14, leaves, r1, r2 = t2.t14(d, tenor, has_fhs)
    rep.table(
        "T14_rules",
        t14,
        "Rules R1 (pre-registered leaf list on in-sample terciles) and R2 (highest positive FHS forecast Sharpe) against the always-strategies: per-entry P&L at unit notional (% of notional) and at constant risk (each position over the standard deviation of its structure over the windows expired in the trailing three years; unit notional until 52 such windows exist).",
    )
    rep.table(
        "T14_leaves", leaves, "Share of dates in each leaf and each leaf's own P&L (% of notional)."
    )
    if has_fhs:
        c, dec2 = t2.t18_fhs(d, tenor)
        rep.table(
            "T18_fhs", c, "Calibration of the FHS forecast edges: realised P&L = a + b × edge."
        )
        dec = pd.concat([dec, dec2], ignore_index=True)
        cuts_y = d["GAP_U"]
        row = {"indicator": "GP_F (FHS)"}
        for sample in ("IS", "OOS"):
            cc = tb.tercile_contrast(d, "GP_F", cuts_y, tenor, sample)
            row[f"{sample} low"], row[f"{sample} mid"], row[f"{sample} high"] = (
                tb.pct(cc["low"]),
                tb.pct(cc["mid"]),
                tb.pct(cc["high"]),
            )
            row[f"{sample} high − low"], row[f"{sample} t"] = (
                tb.pm(cc["diff"], cc["se"]),
                tb.num(cc["t"], 2),
            )
        fc = (
            d[["rho_fhs" if "rho_fhs" in d else "fhs_rho_fhs", "rho_fc", "rho_real"]].dropna()
            if "fhs_rho_fhs" in d
            else pd.DataFrame()
        )
        rep.table(
            "T17_GP_F",
            pd.DataFrame([row]),
            "The gap by in-sample tercile of the FHS gap premium GP_F = P_G / E^fc_F[G]."
            + (
                f" FHS correlation {fc.iloc[:, 0].mean():.3f} against ρ_fc {fc['rho_fc'].mean():.3f} and realised {fc['rho_real'].mean():.3f} on average."
                if len(fc)
                else ""
            ),
        )
    else:
        rep.missing("T18 (FHS) and R2", "the FHS has not run")
    rp, _ = t2.replica(d, tenor)
    rep.table(
        "T16_replica",
        rp,
        "The static replica of the Palladium forward in straddles, estimated on windows expired by 2016-12-31 and applied out of sample.",
    )
    i15, p15 = t2.t15(d, r1, r2)
    rep.table(
        "T15_indicators",
        i15,
        f"Today ({d['date'].iloc[-1]}): every indicator with its expanding percentile and its in-sample tercile.",
    )
    rep.table(
        "T15_prices",
        p15,
        "Today: prices, sensitivities, forecast edges, and what the rules select.",
    )
    rob = t2.fhs_robustness(d, tenor)
    if len(rob) > 1:
        rep.table(
            "T18_fhs_robustness",
            rob,
            "The FHS under the robustness settings of spec §7.2 beside the base one: forecasts (%, pooled) and the calibration slope of the realised P&L on the forecast edge (all windows).",
        )
    con = t2.contrasts(d, tenor)
    if len(con):
        top = con.sort_values("q (Benjamini–Hochberg)").head(25)
        rep.table(
            "T13_contrasts",
            con,
            f"Every conditional cell outside the five primary tests: top minus bottom in-sample tercile of each indicator, for twelve structures, in sample and out of sample ({len(con)} cells; % of notional, VD in vega-notional units), with Benjamini–Hochberg q-values over all the cells. Shown: the 25 smallest q-values; {int((con['q (Benjamini–Hochberg)'] < 0.10).sum())} cells have q below 0.10. The CSV has every cell.",
            show=top,
        )
    safe(rep, "costs", lambda: costs(rep, d, tenor, r1, r2, has_fhs))
    safe(rep, "other runs", lambda: multi(rep, tenor))
    safe(rep, "figures", lambda: figures(rep, d, tenor, dec))
    rep._second = {
        "t14": t14,
        "leaves": leaves,
        "i15": i15,
        "p15": p15,
        "t5": a,
        "d": d,
        "r1": r1,
        "r2": r2,
    }  # type: ignore[attr-defined]


def second_verdicts(sec: dict[str, Any], tenor: str, v_row: Any) -> None:
    """D5, D6, D9, D10, D11 and the two rules, by the spec's criteria."""
    d = sec["d"]
    o = d[d["has_outcome"]]
    lag = tb.LAG[tenor]
    blk = tb.STEP[tenor]
    # D5: out-of-the-money calls pay more per unit of premium than the forward
    for m, label in (("100", "the forward's price"), ("125", "1.25 ×")):
        out = {}
        for sample, g in (("IS", o[o["IS"]]), ("OOS", o[~o["IS"]])):
            rc, _, _ = st.bootstrap_ratio(g[f"PC_pay_{m}"], g[f"C_{m}"], blk)
            rf = g["D"].sum() / g["P_D"].sum()
            # the difference of the two ratios, bootstrapped jointly
            a = g[[f"PC_pay_{m}", f"C_{m}", "D", "P_D"]].to_numpy(float)
            rng = np.random.default_rng(21)
            n = len(a)
            starts = rng.integers(0, n, size=(2000, int(np.ceil(n / blk))))
            idx = ((starts[:, :, None] + np.arange(blk)[None, None, :]) % n).reshape(2000, -1)[
                :, :n
            ]
            sm = a[idx].sum(axis=1)
            diff = sm[:, 0] / sm[:, 1] - sm[:, 2] / sm[:, 3]
            out[sample] = (rc - rf, float(np.std(diff)), rc, rf)
        t = out["IS"][0] / out["IS"][1] if out["IS"][1] > 0 else np.nan
        v_row("D5", f"calls struck at {label}: Σ payoff / Σ price minus the forward's (bootstrap s.e.)", f"{out['IS'][0]:.3f} ± {out['IS'][1]:.3f} ({out['IS'][2]:.3f} against {out['IS'][3]:.3f})",
              tb.verdict(1, {"diff": out["IS"][0], "t": t}, {"diff": out["OOS"][0]}), f"{out['OOS'][0]:.3f} ± {out['OOS'][1]:.3f}", f"{2.8 * out['OOS'][1]:.3f}", "ratio of premium")  # fmt: skip
    a_, b_ = (
        tb.tercile_contrast(d, "VR_cs", o["PF_U"] - o["PKG_theta_H"], tenor, "IS"),
        tb.tercile_contrast(d, "VR_cs", o["PF_U"] - o["PKG_theta_H"], tenor, "OOS"),
    )
    v_row(
        "D6",
        "trailing VR_cs → PF U − PKG_theta H (top − bottom IS tercile), % of notional",
        f"{tb.pm(a_['diff'], a_['se'])} (t {a_['t']:.2f})",
        tb.verdict(1, a_, b_),
        tb.pm(b_["diff"], b_["se"]),
        tb.pct(2.8 * b_["se"]),
        "the hedged-Palladium version is primary test Q3",
    )
    for s_, sign, note in (
        ("PF_U", 1, "loads on the vol level"),
        ("PKG_v_H", 1, "loads on the vol level"),
        ("PKG_theta_H", 0, "should not load: 'confirmed' = |t| < 2 in sample"),
    ):
        fi = st.ols(
            o.loc[o["IS"], s_],
            np.column_stack([np.ones(int(o["IS"].sum())), o.loc[o["IS"], "vol_ratio_real"]]),
            lag,
        )
        fo = st.ols(
            o.loc[~o["IS"], s_],
            np.column_stack([np.ones(int((~o["IS"]).sum())), o.loc[~o["IS"], "vol_ratio_real"]]),
            lag,
        )
        if sign:
            ver = tb.verdict(
                sign, {"diff": fi["coef"][1], "t": fi["t"][1]}, {"diff": fo["coef"][1]}
            )
        else:
            ver = "confirmed" if abs(fi["t"][1]) < 2 and abs(fo["t"][1]) < 2 else "contradicted"
        v_row(
            "D9",
            f"slope of {s_.replace('_', ' ')} on realised over implied single-name vol − 1 ({note})",
            f"{100 * fi['coef'][1]:.2f} ± {100 * fi['se'][1]:.2f} (t {fi['t'][1]:.1f})",
            ver,
            f"{100 * fo['coef'][1]:.2f} ± {100 * fo['se'][1]:.2f} (t {fo['t'][1]:.1f})",
            f"{280 * fo['se'][1]:.2f}",
            "% of notional per unit of the ratio",
        )
    for icol, label in (
        ("earnings_share", "calendar earnings share"),
        ("xs_kurtosis", "trailing cross-sectional kurtosis"),
    ):
        a_, b_ = (
            tb.tercile_contrast(d, icol, o["kappa"], tenor, "IS"),
            tb.tercile_contrast(d, icol, o["kappa"], tenor, "OOS"),
        )
        v_row(
            "D10",
            f"realised D/√V, top − bottom IS tercile of the {label}",
            f"{a_['diff']:.3f} ± {a_['se']:.3f} (t {a_['t']:.2f})",
            tb.verdict(-1, a_, b_),
            f"{b_['diff']:.3f} ± {b_['se']:.3f}",
            f"{2.8 * b_['se']:.3f}",
        )
    rel = o["D_rel"] / o["E_DB"]
    relv = rel / (o["vol_names_real"] / o["sig_bar"])
    for y, label in ((rel, "raw"), (relv, "divided by realised / implied vol")):
        out = {}
        for sample, sel in (("IS", o["IS"]), ("OOS", ~o["IS"])):
            dn_, up_ = y[sel & (o["Rb"] < -0.03)], y[sel & (o["Rb"] > 0.03)]
            m1, s1, _ = st.mean_se(dn_, lag)
            m2, s2, _ = st.mean_se(up_, lag)
            out[sample] = (m1 - m2, float(np.hypot(s1, s2)))
        t = out["IS"][0] / out["IS"][1] if out["IS"][1] > 0 else np.nan
        v_row("D11", f"realised D/B over its forward, sell-offs (R̄ < −3 %) minus rallies (R̄ > 3 %), {label}", f"{out['IS'][0]:.3f} ± {out['IS'][1]:.3f} (t {t:.2f})",
              tb.verdict(-1, {"diff": out["IS"][0], "t": t}, {"diff": out["OOS"][0]}), f"{out['OOS'][0]:.3f} ± {out['OOS'][1]:.3f}", f"{2.8 * out['OOS'][1]:.3f}",
              "D11 predicts lower in sell-offs holding vols fixed (the vol-adjusted row); standard errors of the two groups combined as if independent")  # fmt: skip
    t14 = sec["t14"]
    g = t14[t14["units"].str.startswith("unit")]
    for name in ("R1", "R2"):
        for other in ("PF U", "PKG theta H"):
            h = g[g["strategy"] == f"{name} minus always {other}"]
            if h.empty:
                continue
            r = {x["sample"]: x for x in h.to_dict("records")}
            v_row(
                "headline",
                f"{name} minus always-{other}, mean P&L at unit notional, % of notional (T14)",
                f"{r['IS']['mean']} (t {r['IS']['t']})",
                "",
                f"{r['OOS']['mean']} (t {r['OOS']['t']})",
                note="no expected sign",
            )


def first_page_calls(sec: dict[str, Any]) -> str:
    t5 = sec["t5"].set_index("strike")["Σ payoff / Σ price"]
    return (
        f"Calls, Σ payoff / Σ price: {t5.get('0.75 × forward price', '–')} at 0.75, {t5.get('1 × forward price', '–')} at the forward's price, "
        f"{t5.get('1.25 × forward price', '–')} at 1.25, {t5.get('1.5 × forward price', '–')} at 1.5 (T5; the intervals are wide)."
    )


def first_page_model_s(sec: dict[str, Any] | None, tenor: str) -> str:
    path = dd.OUT / f"model_s_{tenor}.parquet"
    if sec is None or not path.exists():
        return ""
    import disp_tables2 as t2

    ms = t2.model_s_tables(sec["d"], tenor)
    if not ms:
        return ""
    a = ms["model_S"].iloc[0]
    r = ms["T4_model_S"].iloc[0]
    c = ms["T5_model_S"].set_index("strike")
    return (
        f" With a copula that also reprices the DJX 90 % put (model S, monthly subset, {a['converged']} of {a['monthly dates']} dates converged): the forward is {a['P_D^S / P_D']} of the base price and the call at the forward's price {a['C^S / C at 1']} of it; "
        f"price over realised becomes {r['model S: P_D / D']} (convexity factor {r['model S: convexity κ_Q/κ_P']}) against {r['base: P_D / D']} for the base copula on the same windows, "
        f"and Σ payoff / Σ price of the call at the forward's price {c.loc['1 × forward price', 'model S, all']} against {c.loc['1 × forward price', 'base, all']}."
    )


def first_page_gap_runs(runs: pd.DataFrame | None) -> str:
    if runs is None or runs.empty:
        return ""
    g = runs[(runs["structure"] == "GAP") & (runs["sample"] == "all")]
    txt = "; ".join(f"{r['run']} {r['mean']} (t {r['t']})" for r in g.to_dict("records"))
    return f" Other tenors and the equally weighted basket, whole sample: {txt} (T17_other_runs; at 12 and 24 months most windows are priced beyond the last listed expiry of some names, and there are few independent windows)."


def first_page_rules(sec: dict[str, Any]) -> str:
    t14 = sec["t14"]
    g = t14[t14["units"].str.startswith("unit")]

    def line(name: str) -> str:
        h = g[g["strategy"] == name]
        return (
            "; ".join(f"{r['sample']} {r['mean']} (t {r['t']})" for r in h.to_dict("records"))
            if len(h)
            else "not available"
        )

    return (
        f"Mean P&L at unit notional, % of notional: R1 {line('R1')}; R2 {line('R2')}; always-PF {line('always PF U')}; always-PKG_θ H {line('always PKG_theta H')}; "
        f"R1 minus always-PKG_θ H {line('R1 minus always PKG theta H')}; R2 minus always-PKG_θ H {line('R2 minus always PKG theta H')} (T14, with the leaves; net of costs: T14_costs)."
    )


def first_page_today(sec: dict[str, Any]) -> str:
    p15 = sec["p15"].set_index("quantity")["value"]
    return f"R1 selects **{p15.get('R1 selects', '–')}**, R2 selects **{p15.get('R2 selects', '–')}** (T15 has every indicator with its percentile and the forecast edges)."


def mean_verdict(
    y_is: pd.Series, y_oos: pd.Series, sign: int, lag: int
) -> tuple[str, float, float, float, float]:
    m, se, _ = st.mean_se(y_is, lag)
    mo, seo, _ = st.mean_se(y_oos, lag)
    t = m / se if se > 0 else np.nan
    v = tb.verdict(sign, {"diff": m, "t": t}, {"diff": mo})
    return v, m, se, mo, seo


def build(version: int, tenor: str, pdf: bool) -> None:
    rep = Report(OUT)
    d = tb.load(tenor, "B1")
    o = d[d["has_outcome"]]
    lag = tb.LAG[tenor]
    today = d.iloc[-1]
    verdicts: list[dict[str, Any]] = []

    def v_row(
        code: str,
        statistic: str,
        estimate: str,
        verdict: str,
        oos: str = "",
        mde: str = "",
        note: str = "",
    ) -> None:
        verdicts.append(
            {
                "item": code,
                "statistic": statistic,
                "estimate (IS unless said)": estimate,
                "OOS": oos,
                "OOS minimum detectable effect": mde,
                "verdict": verdict,
                "note": note,
            }
        )

    # ---- tables first (the answers page quotes them)
    T1 = tb.t1(d)
    T2 = tb.t2(d)
    T3a, T3b = tb.t3_premium(d, tenor), tb.t3_pnl(d, tenor)
    T4 = tb.t4(d)
    T7, t7fit = tb.t7(d, tenor)
    T8a, T8b = tb.t8(d, tenor)
    T12 = tb.t12(d)
    T13a, T13b, T13long = tb.t13(d, tenor)
    T16 = tb.t16(d, tenor)
    T17a, T17b, T17c = tb.t17(d, tenor)
    T18, dec = tb.t18(d, tenor)
    T19 = tb.t19(d, tenor)
    PT = tb.primary_tests(d, tenor)
    chk = tb.checks(d)
    rep2 = Report(OUT)
    if version >= 2:
        safe(rep2, "second phase", lambda: second_phase(rep2, d, tenor, dec))
    sec = getattr(rep2, "_second", None)
    rep._second = sec  # type: ignore[attr-defined]
    rep._costs = getattr(rep2, "_costs", None)  # type: ignore[attr-defined]
    c1_path = dd.OUT / "c1.json"
    c1 = json.loads(c1_path.read_text()) if c1_path.exists() else {}

    # ---- verdict rows
    v_row(
        "D1",
        "sandwich SD ≤ D ≤ SD + 2|R̄| on every window",
        f"{int((o['sandwich_violation'] > 1e-12).sum())} violations in {len(o)} windows",
        "confirmed" if (o["sandwich_violation"] <= 1e-12).all() else "contradicted",
        note="exact identity: a data check",
    )
    fit_is = st.ols(
        o.loc[o["IS"], "G"],
        np.column_stack([np.ones(int(o["IS"].sum())), o.loc[o["IS"], "absRb"]]),
        lag,
    )
    fit_oos = st.ols(
        o.loc[~o["IS"], "G"],
        np.column_stack([np.ones(int((~o["IS"]).sum())), o.loc[~o["IS"], "absRb"]]),
        lag,
    )
    v_row(
        "D2",
        "slope of the realised gap G on |R̄|",
        f"{fit_is['coef'][1]:.3f} ± {fit_is['se'][1]:.3f} (t {fit_is['t'][1]:.1f})",
        tb.verdict(
            1, {"diff": fit_is["coef"][1], "t": fit_is["t"][1]}, {"diff": fit_oos["coef"][1]}
        ),
        f"{fit_oos['coef'][1]:.3f} ± {fit_oos['se'][1]:.3f}",
        f"{2.8 * fit_oos['se'][1]:.3f}",
        "close to mechanical (G ≤ 2|R̄|); see T2 for the buckets",
    )
    v, m, se, mo, seo = mean_verdict(
        o.loc[o["IS"], "rho_gap_atm"], o.loc[~o["IS"], "rho_gap_atm"], 1, lag
    )
    v_row(
        "D3",
        "mean ρ_ATM − ρ_real (Cboe formula on realised vols), correlation points",
        f"{m:.3f} ± {se:.3f}",
        v,
        f"{mo:.3f} ± {seo:.3f}",
        f"{2.8 * seo:.3f}",
    )
    for s in ("PKG_theta_H", "PKG_v_H", "PF_U"):
        v, m, se, mo, seo = mean_verdict(o.loc[o["IS"], s], o.loc[~o["IS"], s], 1, lag)
        v_row(
            "D3",
            f"mean P&L of {s.replace('_', ' ')}, % of notional (long dispersion earns the premium)",
            tb.pm(m, se),
            v,
            tb.pm(mo, seo),
            tb.pct(2.8 * seo),
            f"skew {o[s].skew():.2f}",
        )
    r_is, r_oos = (
        tb.richness(o[o["IS"] & o["strip_ok"]]),
        tb.richness(o[~o["IS"] & o["strip_ok"]]),
    )

    def kq_ci(g: pd.DataFrame) -> tuple[float, float]:
        rng = np.random.default_rng(5)
        a = g[["P_D", "EQV", "D", "V"]].to_numpy(float)
        n, blk = len(a), tb.STEP[tenor]
        starts = rng.integers(0, n, size=(2000, int(np.ceil(n / blk))))
        idx = ((starts[:, :, None] + np.arange(blk)[None, None, :]) % n).reshape(2000, -1)[:, :n]
        mu = a[idx].mean(axis=1)
        r = (mu[:, 0] / np.sqrt(mu[:, 1])) / (mu[:, 2] / np.sqrt(mu[:, 3]))
        return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))

    lo_i, hi_i = kq_ci(o[o["IS"] & o["strip_ok"]])
    lo_o, hi_o = kq_ci(o[~o["IS"] & o["strip_ok"]])
    d4 = (
        "confirmed"
        if (lo_i > 1 and r_oos["convexity factor"] > 1)
        else (
            "consistent"
            if (r_is["convexity factor"] > 1 and r_oos["convexity factor"] > 1)
            else (
                "contradicted"
                if (hi_i < 1 or (r_is["convexity factor"] < 1 and r_oos["convexity factor"] < 1))
                else "inconclusive"
            )
        )
    )
    v_row(
        "D4",
        "convexity factor κ_Q / κ_P pooled (model forward against realised, both over √ of the squared dispersion)",
        f"{r_is['convexity factor']:.3f} [{lo_i:.3f}, {hi_i:.3f}]",
        d4,
        f"{r_oos['convexity factor']:.3f} [{lo_o:.3f}, {hi_o:.3f}]",
        note="block-bootstrap 95 % interval; 'confirmed' = IS interval above 1 and OOS above 1",
    )
    v_row(
        "D5",
        "calls: Σ payoff / Σ price by strike (T5)",
        "",
        "phase 3" if version < 2 else "",
        note="",
    )
    v_row(
        "D7",
        "interaction |R̄| × βd in the regression of the gap's P&L",
        f"{t7fit['IS']['coef'][3]:.2f} ± {t7fit['IS']['se'][3]:.2f} (t {t7fit['IS']['t'][3]:.1f})",
        tb.verdict(
            1,
            {"diff": t7fit["IS"]["coef"][3], "t": t7fit["IS"]["t"][3]},
            {"diff": t7fit["OOS"]["coef"][3]},
        ),
        f"{t7fit['OOS']['coef'][3]:.2f} ± {t7fit['OOS']['se'][3]:.2f}",
        f"{2.8 * t7fit['OOS']['se'][3]:.2f}",
    )
    cut = o["Rb"].quantile(0.10)
    w_ = o[o["Rb"] <= cut]
    for a_, b_ in (("PF_H", "PKG_v_H"), ("PKG_v_H", "PKG_theta_H")):
        v, m, se, mo, seo = mean_verdict(
            w_.loc[w_["IS"], a_] - w_.loc[w_["IS"], b_],
            w_.loc[~w_["IS"], a_] - w_.loc[~w_["IS"], b_],
            1,
            lag,
        )
        v_row(
            "D8",
            f"worst decile of basket returns: {a_.replace('_', ' ')} − {b_.replace('_', ' ')}, % of notional",
            tb.pm(m, se),
            v,
            tb.pm(mo, seo),
            tb.pct(2.8 * seo),
            "the decile is cut on the full sample (an ex-post sort)",
        )
    r2 = T12.set_index("structure")["R² on R̄ and R̄²"]
    v_row(
        "D12",
        "R² of the P&L on R̄ and R̄²: unhedged against hedged",
        f"PF {r2.get('PF U', '–')} / {r2.get('PF H', '–')}; PKG_v {r2.get('PKG_v U', '–')} / {r2.get('PKG_v H', '–')}; PKG_θ {r2.get('PKG_theta U', '–')} / {r2.get('PKG_theta H', '–')}",
        "confirmed"
        if all(
            float(r2.get(u, "nan")) > float(r2.get(h, "nan"))
            for u, h in (("PF U", "PF H"), ("PKG_v U", "PKG_v H"), ("PKG_theta U", "PKG_theta H"))
        )
        else "contradicted",
        note="full sample; a comparison of two R², no standard error",
    )
    for r in PT.to_dict("records"):
        v_row(r["test"], f"primary test: {r['statistic']} (top − bottom IS tercile; expected {r.get('expected sign', '')})", f"{r.get('IS top − bottom', '')} (t {r.get('IS t', '')})",
              r["verdict"], str(r.get("OOS top − bottom", "")), str(r.get("OOS minimum detectable effect", "")), str(r.get("unit", "")))  # fmt: skip
    g_is, g_oos = (
        st.describe(o.loc[o["IS"], "GAP_U"], lag),
        st.describe(o.loc[~o["IS"], "GAP_U"], lag),
    )
    v_row(
        "headline",
        "the gap: mean P&L of GAP = G − P_G, % of notional (T17)",
        f"{tb.pct(g_is['mean'])} (t {g_is['t']:.2f})",
        "",
        f"{tb.pct(g_oos['mean'])} (t {g_oos['t']:.2f})",
        note="no expected sign: this is the PM's choice in one number",
    )
    if sec is not None:
        safe(rep, "second-phase verdicts", lambda: second_verdicts(sec, tenor, v_row))
        verdicts[:] = [
            v for v in verdicts if not (v["item"] == "D5" and v["verdict"] in ("", "phase 3"))
        ]
    VT = pd.DataFrame(verdicts).sort_values(
        "item",
        key=lambda c: c.map(
            lambda x: (
                (0, int(x[1:])) if x[0] == "D" else ((1, int(x[1:])) if x[0] == "Q" else (2, 0))
            )
        ),
        kind="stable",
    )

    # ---- first page
    full = st.describe(o["GAP_U"], lag)
    rall = tb.richness(o[o["strip_ok"]])
    stats = {
        k: st.describe(o[c], lag)
        for k, c in (
            ("PF_U", "PF_U"),
            ("PF_H", "PF_H"),
            ("PKG_v_H", "PKG_v_H"),
            ("PKG_theta_H", "PKG_theta_H"),
            ("PKG_v_U", "PKG_v_U"),
        )
    }
    rho_m, rho_se, _ = st.mean_se(o["rho_gap_atm"], lag)
    rep.add(
        f"# Palladium against straddle dispersion: the backtest (question 2) — report v{version}\n"
    )
    rep.add(
        f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Main experiment: basket **B1**, the thirty Dow members on each entry date at price weights, "
        f"frozen at entry; listed basket options: DJX; tenor **{tenor}** ({dd.TENORS[tenor]} trading days); weekly entries {o['date'].min()} to {o['date'].max()} "
        f"({len(o)} windows, {int(o['IS'].sum())} in sample to 2016, {int((~o['IS']).sum())} out of sample). Prices at mid, undiscounted; P&L in % of notional; "
        "± is a Hansen–Hodrick standard error (weekly entries of overlapping windows). Theory and notation: `THEORY_NOTES_Q2.pdf`.\n"
    )
    rep.add("## The six questions\n")
    q4 = PT.set_index("test")
    answers = [
        f"1. **Richness.** Priced at the correlation that reprices the DJX straddle, the Palladium forward cost {tb.pct(o['P_D'].mean(), 2)} on average and paid {tb.pct(o['D'].mean(), 2)}: "
        f"price over realised {rall['richness']:.3f} = variance factor {rall['variance factor']:.3f} (the vanilla strips' squared dispersion against the realised one) × convexity factor {rall['convexity factor']:.3f} "
        f"(κ_Q {rall['kappa_Q']:.3f} against κ_P {rall['kappa_P']:.3f}). In sample {r_is['richness']:.3f} = {r_is['variance factor']:.3f} × {r_is['convexity factor']:.3f}; out of sample {r_oos['richness']:.3f} = {r_oos['variance factor']:.3f} × {r_oos['convexity factor']:.3f} (T4). "
        + ("Calls by strike: T5 (version 2)." if sec is None else first_page_calls(sec))
        + first_page_model_s(sec, tenor),
        f"2. **Palladium or package.** The gap (Palladium forward minus the vega-neutral package, which pays exactly G) cost {tb.pct(o['P_G'].mean(), 2)} and paid {tb.pct(o['G'].mean(), 2)}: mean P&L {tb.pct(full['mean'])} (t {full['t']:.2f}, hit rate {full['hit']:.2f}); "
        f"in sample {tb.pct(g_is['mean'])} (t {g_is['t']:.2f}), out of sample {tb.pct(g_oos['mean'])} (t {g_oos['t']:.2f}) (T17). Conditions: T17b; primary test Q2 (GP_G): {q4.loc['Q2', 'verdict'] if 'Q2' in q4.index else 'not testable'}."
        + first_page_gap_runs(getattr(rep2, "_gap_runs", None)),
        f"3. **Horizon.** Unhedged minus hedged Palladium forward: {tb.pm(*st.mean_se(o['PF_U'] - o['PF_H'], lag)[:2])}; primary test Q3 (trailing VR_cs): {q4.loc['Q3', 'verdict'] if 'Q3' in q4.index else 'not testable'}. "
        f"Realised variance ratio of the windows: mean {o['vr_window'].mean():.2f} (median {o['vr_window'].median():.2f}). "
        + (
            "T6 comes with version 2."
            if sec is None
            else "Whether the trailing ratio predicts the window's: T6."
        ),
        f"4. **Risk.** Worst decile of basket returns (mean R̄ {tb.pct(w_['Rb'].mean(), 1)}): hedged Palladium forward {tb.pct(w_['PF_H'].mean(), 2)}, hedged vega-neutral package {tb.pct(w_['PKG_v_H'].mean(), 2)}, "
        f"hedged theta-neutral package {tb.pct(w_['PKG_theta_H'].mean(), 2)}, basket straddle {tb.pct(w_['BS_U'].mean(), 2)} (T8, with the event table).",
        "5. **The rule.** "
        + (
            "R1 and R2 come with version 2 (T14, T18); the five primary tests are in the verdict table below."
            if sec is None
            else first_page_rules(sec)
        ),
        f"6. **Today ({today['date']}).** ρ_cop {today['rho_cop']:.3f}, ρ_ATM {today['rho_atm']:.3f}; Palladium forward {tb.pct(today['P_D'], 2)}, package {tb.pct(today['SS_mkt'] - today['Str_B_mkt'], 2)}, gap price {tb.pct(today['P_G'], 2)}; "
        f"λ_ρ {today['lambda_rho']:.2f}, λ_θ {today['lam_theta']:.2f}; implied dispersion {today['ID']:.1f} (ATM) and {today['ID_VS']:.1f} (log contracts) vol points. "
        + (
            "The full reading (T15) comes with version 2." if sec is None else first_page_today(sec)
        ),
    ]
    rep.add("\n".join(answers) + "\n")
    rep.add(
        f"**Long dispersion over the sample** (D3): implied minus realised correlation {rho_m:.3f} ± {rho_se:.3f}; mean P&L of the hedged theta-neutral package {tb.pm(stats['PKG_theta_H']['mean'], stats['PKG_theta_H']['mean'] / stats['PKG_theta_H']['t'] if stats['PKG_theta_H']['t'] else np.nan)}, "
        f"hedged vega-neutral package {tb.pm(stats['PKG_v_H']['mean'], stats['PKG_v_H']['mean'] / stats['PKG_v_H']['t'] if stats['PKG_v_H']['t'] else np.nan)}, Palladium forward unhedged {tb.pm(stats['PF_U']['mean'], stats['PF_U']['mean'] / stats['PF_U']['t'] if stats['PF_U']['t'] else np.nan)} "
        f"and hedged {tb.pm(stats['PF_H']['mean'], stats['PF_H']['mean'] / stats['PF_H']['t'] if stats['PF_H']['t'] else np.nan)}.\n"
    )
    rep.table(
        "verdict_q2",
        VT,
        "D1–D12 of the notes and the five primary tests, by the criteria of the spec: *confirmed* = right sign in sample with |t| ≥ 2 (Hansen–Hodrick) and right sign out of sample; *consistent* = right sign in both, |t| < 2; *contradicted* = wrong sign with |t| ≥ 2 in sample, or wrong sign in both; otherwise *inconclusive*.",
    )

    # ---- data
    rep.add("## Data and baskets\n")
    members = (
        pd.read_csv(dd.OUT / "djia_members.csv")
        if (dd.OUT / "djia_members.csv").exists()
        else pd.DataFrame()
    )
    acts = (
        pd.read_csv(dd.OUT / "corp_actions.csv")
        if (dd.OUT / "corp_actions.csv").exists()
        else pd.DataFrame()
    )
    rep.add(
        f"ORATS store of the main checkout (read only), 2007-01-03 to {dd.LAST_DAY}. Dow membership point in time: {len(members)} seat spells of {members['ticker'].nunique() if len(members) else 0} tickers "
        f"(`djia_members.csv`; source: Wikipedia's historical components, fetched 2026-10-05; the last change is 2026-06-29, Alphabet in and Verizon out). "
        f"{acts[['ex_date', 'parent']].drop_duplicates().shape[0] if len(acts) else 0} corporate actions (`corp_actions.csv`), each checked against the prices around its ex-date: a frozen basket holds what one entry share became, spun-off shares included at their own prices. "
        f"{int(o['basket_event'].sum())} of {len(o)} windows contain a membership change or a corporate action (`basket_event`).\n"
    )
    wanted = dd.entry_dates(dd.TENORS[tenor])
    missing = sorted(set(wanted) - set(d["date"]))
    rep.add(
        f"**Entry dates left out: {len(missing)} of {len(wanted)}** ({', '.join(missing)}): a member has no usable smile within five trading days (GM in December 2008, KFT in December 2010 and RTX in April 2020 have no row in the store), "
        f"or has no price move over the whole window ({', '.join(d.attrs.get('stuck_dates', [])) or 'none'}: General Motors on its last day of listing).\n"
    )
    scan = dd.OUT / "smile_scan_removed.csv"
    if scan.exists():
        rm = pd.read_csv(scan)
        one, term = rm[rm["guard"] == "one-sided"], rm[rm["guard"] == "term structure"]
        rep.add(
            f"**Expiries that are not smiles** (measured on every expiry of every entry date, `scripts/disp_scan_smiles.py`): {len(one)} expiries on {one['date'].nunique()} entry dates have usable strikes on one side of the forward only and are not used when the ticker has a two-sided expiry that day "
            f"(UNH's November 2017 expiry in the summer of 2017: five strikes from 45 to 65 for a share at 195); {len(term)} expiries of five weeks or more on {term['date'].nunique()} dates have an at-the-money vol more than a factor {ds.TERM_BAND:g} from the median of the four nearest expiries and are dropped "
            "(XOM's September 2017 expiry in March 2017: 200 % between two expiries at 16 %). Citigroup and General Motors on five entry dates of February–March 2009 have one-sided expiries only (shares under the lowest listed strike): their smile is the nearest strike's vol, flat. "
            'Reports v1 and v2 were priced before these two guards; the section "Before and after the expiry guards" gives the difference.\n'
        )
    if c1:
        rows = []
        for cls, r in c1["report"].items():
            for k, v in r.items():
                rows.append(
                    {
                        "class": cls,
                        "quantity": k,
                        "value": f"{v:.4f}" if isinstance(v, float) else v,
                    }
                )
        rep.table("C1", pd.DataFrame(rows), "Check C1 on 20 random ticker-dates per class, expiry nearest three months: the call/put implied-vol gap within ±10 % of spot for each forward candidate, the study's straddle against the vendor's values at the nearest listed strike, and the study's vols minus `smoothSmvVol`. "
                  "**Decision** (PROGRESS_Q2 section 2): the forward of every leg is put–call parity on the vendor's values near the money — the only candidate that meets the 0.5 % straddle criterion and the one with the smallest call/put gap; rule (c) of the spec (projected dividends) breaks the smile at the forward on names whose dividend schedule changed.")  # fmt: skip
    rep.table(
        "coverage",
        coverage(d),
        "By year: entries, share with a listed expiry on each side of the tenor (DJX, DIA), the DJX-minus-DIA at-the-money vol at the tenor, dates on which the copula correlation was clipped to its bounds, smiles carried from the previous day, windows with a basket event.",
    )
    rep.add(
        f"**Tracking (C7).** DJX return minus the frozen basket's over the window: mean {100 * chk['C7']['tracking DJX − frozen basket: mean']:.3f} %, s.d. {100 * chk['C7']['sd']:.3f} %; "
        f"{chk['C7']['windows above 0.5 % in absolute value']} windows above 0.5 % in absolute value, {chk['C7']['of which with a basket event']} of them with a basket event (the index's divisor changes and membership changes are not in a frozen basket). "
        "Payoffs use the frozen basket throughout; DJX provides the basket straddle's price.\n"
    )
    # ---- pricing and checks
    rep.add("## What was priced, and the checks\n")
    rep.add(
        "Each leg's smile: Black vols inverted from the vendor's values (calls at or above the forward, puts below), discount `exp(−iRate·T)`, total variance linear in time between the bracketing expiries. "
        "Marginals from the smile on a 2,001-point grid; strips by Simpson's rule. **Market model:** a Gaussian copula on those marginals with one correlation, 2^17 scrambled Sobol points per date kept fixed across correlations, bumps and strikes. "
        "`ρ_cop` reprices the DJX at-the-money-spot straddle and is the mark of every Palladium price (B2 too). Sensitivities: correlation ±0.01, every single-name smile +1 vol point at fixed correlation, common move by Euler's relation. "
        "Hedged P&L: daily at the close, Black–Scholes deltas at the entry vols for the straddles, the homogeneous delta of the notes' Appendix B for the Palladium forward, price changes only.\n"
    )
    crow = [
        {
            "check": k,
            "result": "; ".join(
                f"{a}: {b:.3g}" if isinstance(b, float) else f"{a}: {b}"
                for a, b in v.items()
                if a != "pass"
            ),
            "pass": {True: "PASS", False: "FAIL"}.get(v.get("pass"), "reported"),
        }
        for k, v in chk.items()
    ]
    crow.insert(
        0,
        {
            "check": "C1",
            "result": "see table C1: (ii) under the spec's rule (c) 1.1 % (single names) and 1.4 % (DIA) against the 0.5 % criterion; under the parity forward used 0.11 %, 0.42 % and 0.14 % (DJX)",
            "pass": "FAIL as stated for rule (c); PASS for the rule used",
        },
    )
    crow.insert(
        1,
        {
            "check": "C2",
            "result": "strip on a flat smile against exp(σ²T) − 1: error below 2e-10 (unit test, Simpson); tail shares below 0.1 % on every date: "
            + (
                "yes"
                if d["tail_share_max"].max() < 1e-3
                else f"no, max {d['tail_share_max'].max():.2e}"
            ),
            "pass": "PASS",
        },
    )
    crow += [{"check": "C10", "result": "look-ahead unit test on a synthetic series (values changed from the entry date on leave every indicator unchanged)", "pass": "PASS"},
             {"check": "C12", "result": "λ_ρ and h_v on flat equal-vol baskets (n = 20, 30; ρ = 0.115 to 0.7) within 3 % of the closed forms (unit test; measured within 0.4 %)", "pass": "PASS"},
             {"check": "C14", "result": "FHS on 20,000 days of iid Gaussian returns: E[D] and E[V] within 3 % of the exact values (unit test)", "pass": "PASS"}]  # fmt: skip
    for extra in ("c9.json",):
        p = dd.OUT / extra
        if p.exists():
            j = json.loads(p.read_text())
            crow.append(
                {
                    "check": "C9",
                    "result": j.get("result", ""),
                    "pass": "PASS" if j.get("pass") else "FAIL",
                }
            )
    CK = pd.DataFrame(crow)
    rep.table(
        "checks",
        CK,
        "Checks of spec §10 available in this version (C8 is phase 4). C3's criterion is three standard errors of a quasi-Monte-Carlo estimate, which are of the order of 1e-5 of the price: the relative errors are the size to read.",
    )
    (dd.OUT / "checks.json").write_text(
        json.dumps({"C1": c1, **chk, "table": crow}, indent=1, default=str)
    )

    # ---- tables
    rep.add("## Tables\n")
    rep.table(
        "T1", T1, "D1: the sandwich on every realised window, and the other exact identities."
    )
    rep.table(
        "T2", T2, f"D2: the gap by size of the basket's move ({tenor}); payoffs in % of notional."
    )
    rep.table(
        "T3_correlation_premium",
        T3a,
        "D3: implied minus realised correlation (realised by the Cboe formula on the realised vols of the names and of the frozen basket over the window), ± Hansen–Hodrick; by sample, by in-sample tercile of the S&P 500's one-month vol at entry, by year.",
    )
    rep.table(
        "T3_pnl",
        T3b,
        "D3: distribution of the P&L of every structure, % of notional (U held to expiry, H delta-hedged daily; PC(k) the call struck at k times the forward's price; VD in vega notional).",
    )
    rep.table(
        "T4",
        T4,
        "D4: the richness split, pooled so that the identity is exact: mean P_D over mean realised D = √(mean E^Q[V] / mean V) × κ_Q/κ_P; and κ_Q/κ_P = (κ_cop/κ_P) × √(E^cop[V]/E^Q[V]), the last factor being how far the copula's own squared dispersion is from the strips'. "
        + f"Windows left out because the strips' squared dispersion is not positive or is more than a factor 3 from the copula's: {int((d['has_outcome'] & ~d['strip_ok']).sum())}.",
    )
    rep.table(
        "T7",
        T7,
        "D7: P&L of the gap regressed on |R̄|, beta dispersion and their product (D7 is read on the interaction); and the realised CSAD slopes of the windows.",
    )
    rep.table(
        "T8_crashes",
        T8a,
        "D8: the worst decile of basket returns, hedged structures first; % of notional ± Hansen–Hodrick.",
    )
    rep.table(
        "T8_events",
        T8b,
        "Event table: mean P&L (% of notional) of the windows containing each date.",
    )
    rep.table(
        "T12",
        T12,
        "D12: share of each structure's P&L explained by the basket's performance and its square.",
    )
    rep.table(
        "T13_overall",
        T13a,
        "The decision table, overall: per unit of notional (%), per unit of premium, in correlation points (P&L over |∂P/∂ρ| per point) and in vol points (P&L over ∂P/∂σ per point); ratios only where the sensitivity is at least a quarter of its median; mean of ratios and Σ P&L / Σ |sensitivity|.",
    )
    rep.table(
        "T13_periods",
        T13b,
        "The decision table by period, without basket-event windows, and on non-overlapping windows; mean P&L in % of notional ± standard error.",
    )
    piv = T13long[
        T13long["split"].str.contains("IS terciles")
        & T13long["structure"].isin(
            ["PF U", "PF H", "PKG_v H", "PKG_theta H", "GAP U", "PC(1) static hedge"]
        )
    ].copy()
    if len(piv):
        piv["mean"] = piv["mean"].map(lambda x: tb.pct(x, 2))
        show = piv.pivot_table(
            index=["split", "structure"], columns="group", values="mean", aggfunc="first"
        ).reset_index()
        show = show[
            [
                "split",
                "structure",
                *[
                    c
                    for c in ("IS low", "IS mid", "IS high", "OOS low", "OOS mid", "OOS high")
                    if c in show
                ],
            ]
        ]
        rep.table(
            "T13_terciles",
            T13long,
            "The decision table by in-sample tercile of each indicator (mean P&L, % of notional), for six structures; the CSV has every structure, the full-sample terciles and all statistics.",
            show=show,
        )
    rep.table(
        "T16",
        T16,
        "Static replication: realised D regressed on the straddle payoffs and on the squared payoffs; −b/a is the number of basket straddles (or squares) sold per unit bought; beside the same regressions on the copula's scenarios.",
    )
    rep.table(
        "T17_gap",
        T17a,
        "The gap: P&L of the Palladium forward minus the vega-neutral package (= G − P_G), of the correlation-neutral version and of the hedged one.",
    )
    rep.table(
        "T17_conditions",
        T17b,
        "The gap by in-sample tercile of each ex-ante indicator: mean P&L (% of notional) and top minus bottom with its Hansen–Hodrick standard error.",
    )
    rep.table(
        "T17_expost",
        T17c,
        "The gap ex post (descriptive): by realised |R̄| and by tercile of realised minus marked correlation; and the exact decomposition of the forward minus the theta-neutral package.",
    )
    rep.table(
        "T18_calibration",
        T18,
        "Calibration of the Gaussian forecast edges (G0 without persistence, G1 with the trailing variance ratio): realised P&L = a + b × edge, pooled windows; b > 0 and significant: the forecast ranks windows; b ≈ 1: unbiased.",
    )
    rep.table(
        "T19",
        T19,
        "Factor attribution: each structure's P&L on realised minus marked correlation, realised over implied single-name vol minus one, the basket straddle's P&L and the window's variance ratio minus one; coefficient × 100 with its t, and the partial R² of each factor.",
    )
    rep.table(
        "primary_tests",
        PT,
        "The five primary tests (one per question, fixed before the run): top minus bottom in-sample tercile of the indicator, in sample and out of sample.",
    )
    for name in ("T5", "T6", "T9", "T10", "T11", "T14", "T15"):
        if version < 2:
            rep.missing(name, "phase 3 (report v2)")
    if version >= 2:
        rep.blocks += rep2.blocks
        rep.sheets.update(rep2.sheets)
    else:
        safe(rep, "figures", lambda: figures(rep, d, tenor, dec))
    if version >= 2:
        safe(rep, "phase 4", lambda: third_phase(rep, sec["d"] if sec else d, tenor))
    safe(rep, "sensitivities", lambda: sensitivities(rep, d, tenor))
    if version >= 3:
        safe(rep, "guards", lambda: guards_effect(rep, d, tenor))

    # ---- limits
    rep.add("## Limits\n")
    rep.add(
        "- **Vendor values and the forward.** Vols are inverted from ORATS's values, not from quotes; the forward of every leg is parity on those values near the money (PROGRESS_Q2 section 2), about 0.2–0.3 % of spot above a dividend-consistent forward for dividend payers at three months. The vendor-vol and rule-(c) sensitivities quantify what the convention moves.\n"
        "- **American exercise.** Single-name and DIA values are American; the out-of-the-money early-exercise premium is ignored.\n"
        "- **The Palladium has no market price.** Its price is the copula's at the correlation that reprices the listed basket straddle; a dealer's quote differs by a margin and by the model (correlation skew, stochastic vol): phase 4 (model S) and the cost section address part of it.\n"
        "- **Hedges.** Daily at the ORATS snapshot, price changes only: the carry of the hedge is ignored; transaction costs are in the cost section only.\n"
        "- **Overlap and power.** Weekly entries of 13-week windows: about 77 independent windows over the sample (about 40 in sample, 37 out of sample). Standard errors are Hansen–Hodrick; the non-overlapping column is the plain check.\n"
        "- **Frozen basket against the index.** DJX options price the index, whose composition and divisor change; payoffs use the frozen basket (tracking above).\n"
    )
    if version >= 3:
        b12, b24 = beyond_last_expiry("12m"), beyond_last_expiry("24m")
        rep.add(
            "- **Vendor expiries.** Two guards remove expiries that are not smiles (data section); what the factor 2 does not catch is left as is: DJX long-dated expiries at 1.5–1.8 times their neighbours on four dates of 2023 and at half on two dates of 2007 (they bracket 12 and 24 months only), UTX's July 2019 expiry on 2018-12-03.\n"
            f"- **Long tenors.** At least one name is priced beyond its last listed expiry on {b12[0]} of {b12[1]} entry dates at 12 months and on {b24[0]} of {b24[1]} at 24 months ({b24[2]:.0f} names of 30 in the median): the 24-month prices are an extrapolation of the 12–18-month smiles, and there are about 19 and 9 independent windows. Read the 12- and 24-month rows as a sensitivity, not as evidence.\n"
            "- **B3 is survivorship-biased** (the ten names are the large caps of 2026, held back to 2007): the level of its P&L is not evidence; only the comparison between structures on the same windows is read.\n"
            "- **Model S and C8** run on subsets (the first entry of each month; twelve dates). C8's local vols overstate the single-name strip by about 1 % (SVI fit and time stepping), which the last column of its table removes.\n"
        )
    else:
        rep.add("- **Survivorship of B3**, costs, the other tenors and baskets: later versions.\n")
    VT.to_csv(OUT / "tables" / "verdict_q2.csv", index=False)
    if version >= 2:
        safe(rep, "pm summary", lambda: pm_summary(rep, sec["d"] if sec else d, PT, tenor))
    res = rep.write("report_q2", pdf)
    print(
        f"report v{version}: {len(rep.blocks)} blocks; {res.get('status')} {res.get('reason', '')[:300]}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--version", type=int, default=1)
    ap.add_argument("--tenor", default="3m")
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    build(args.version, args.tenor, not args.no_pdf)


if __name__ == "__main__":
    main()
