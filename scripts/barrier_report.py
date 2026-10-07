"""Tables, figures and the PDF of the barrier study on 2007–2026 (BARRIER_STUDY_SPEC §12).

    python scripts/barrier_report.py [--out outputs/interview/report]

Reads ``barrier_results/{cells,positions,features}.parquet`` and ``lsv_entries/`` (what exists),
writes ``<out>/tables/*.csv``, ``<out>/figures/*.pdf``, ``<out>/report.md``, ``report.tex`` and,
with tectonic, ``report.pdf``.  Every table states its sample; amounts are in % of the entry
spot unless said otherwise; standard errors are circular-block-bootstrap ones with the block
equal to the maturity in entries (spec §9).
"""

# ruff: noqa: RUF001, E501 — report prose: typographic signs and long caption lines
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from volsto.studies import barrier_history as bh
from volsto.studies import latex as lx

SD = ["s0.25", "s0.5", "s1", "s1.5", "s2"]
PCT = {1: ["p105", "p110", "p115", "p120"], -1: ["p95", "p90", "p85", "p80"]}
SIDE = {1: "call", -1: "put"}
BLOCK = {1: 5, 3: 13, 6: 26, 12: 52}
DAYS = {1: 30.4, 3: 91.3, 6: 182.6, 12: 365.25}
HALVES = (("2007–2016", "2007-01-01", "2016-12-31"), ("2017–2026", "2017-01-01", "2026-12-31"))
MAIN = ["A1", "A2", "B3", "B4", "B5_2", "B5_6", "B6", "E13", "D10", "D11"]
NAMES = {
    "A1": "KO daily",
    "A2": "KO cont.",
    "B3": "spread",
    "B4": "ratio 1x2",
    "B5_2": "fly n=2",
    "B5_3": "fly n=3",
    "B5_4": "fly n=4",
    "B5_6": "fly n=6",
    "B6": "tight limit",
    "C7": "spread+dig",
    "C8": "static hedge",
    "D10": "matched ratio",
    "D11": "matched fly",
    "E12": "halfway fly",
    "E13": "halfway ratio",
    "VAN": "vanilla",
}
PRIMARY = {
    "dist_sd": "distance to the barrier (sd)",
    "iv_minus_rv": "implied minus realised vol",
    "skew_side": "skew on the trade's side",
    "fskew_diff": "forward skew, LSV minus local vol",
    "dig_touch_lv": "digital at the barrier over knock probability (local vol)",
    "ssr_3m_3m": "realised SSR (3m vol, trailing 3m)",
    "ret_3m": "3-month return",
}


EXPLORATORY = {
    "band_pos_lv": "position in the model-free band (local vol)",
    "band_pos_lsv": "position in the model-free band (LSV 1.2)",
    "ko_over_spread": "knock-out premium over spread premium",
    "atm": "at-the-money vol",
    "convexity": "convexity",
    "term_slope": "term slope (1y − 3m)",
    "fwd_vol_2nd_half": "forward vol, second half of the life",
    "corr_vol_spot_3m": "realised correlation of vol changes and returns",
    "vol_of_vol_3m": "realised vol of vol (3m)",
    "vvix": "VVIX",
    "skew_decay": "decay of the skew with maturity",
    "oi_share_barrier": "open interest within 1 % of the barrier",
    "gex_usd_per_pct": "gamma-exposure proxy",
    "barrier_vs_1y": "barrier against the trailing 1y extreme",
    "cor3m": "COR3M",
    "hs_B": "bid-ask half-width at the barrier strike (vol points)",
    "overnight_share": "overnight share of variance",
    "tail_days_3m": "days beyond 2.5 sd (3m)",
    "up_down_vol": "upside over downside realised vol",
    "range_over_close_vol": "range-based over close-to-close vol",
    "drawdown_1y": "drawdown from the 1y high",
    "var_ratio_5": "variance ratio, 5 days",
    "above_200d": "spot above its 200-day average",
    "fomc_in_life": "Fed meetings in the life",
}


# --------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(cells, positions)``: one row per cell with entry data, realised path, conditions and
    LSV marks; one row per (cell, position)."""
    cells = pd.read_parquet(bh.RESULTS / "cells.parquet")
    pos = pd.read_parquet(bh.RESULTS / "positions.parquet")
    feat = bh.RESULTS / "features.parquet"
    if feat.exists():
        cells = cells.merge(pd.read_parquet(feat), on="cell", how="left")
    chain_files = sorted((bh.RESULTS / "chain").glob("*.parquet"))
    if chain_files:
        chain = pd.concat([pd.read_parquet(p) for p in chain_files], ignore_index=True)
        cells = cells.merge(chain, on=["entry", "months", "side", "barrier"], how="left")
    fomc = bh.HISTORY / "fomc_dates.csv"
    if fomc.exists():
        f_ = pd.read_csv(fomc, dtype=str)
        meetings = np.array(sorted(f_.loc[f_["kind"] == "meeting", "date"]))
        lo = np.searchsorted(meetings, cells["entry"].to_numpy(str), side="right")
        hi = np.searchsorted(meetings, cells["expiry"].to_numpy(str), side="right")
        cells["fomc_in_life"] = (hi - lo).astype(float)
        cells["knock_on_fomc"] = cells["tau1"].isin(set(meetings)) if "tau1" in cells else False
    lsv_files = sorted((bh.RESULTS / "lsv2").glob("*.parquet")) or sorted(
        (bh.RESULTS / "lsv_entries").glob("*.parquet")
    )  # the addendum's store when it exists
    if lsv_files:
        lsv = pd.concat([pd.read_parquet(p) for p in lsv_files], ignore_index=True)
        lsv = lsv[[c for c in lsv.columns if "_van_" not in c]]
        cells = cells.merge(lsv, on=["entry", "months", "side", "barrier"], how="left")
    c = cells
    c["year"] = c["entry"].str[:4]
    c["half"] = np.where(c["entry"] <= "2016-12-31", HALVES[0][0], HALVES[1][0])
    c["period"] = c["entry"].map(bh.period_of)
    c["built"] = ~c["carried"].astype(bool)
    c["strict"] = c["tolerances_ok"].astype(bool) & c["built"]
    c["w"] = (c["B"] - c["K"]).abs()
    sd = c["atm"] * np.sqrt(c["T"])
    c["dist_sd"] = np.log(c["B"] / c["K"]).abs() / sd
    c["dist_sd_fwd"] = np.log(c["B"] / c["F0"]).abs() / sd
    c["sd_bin"] = c["dist_sd"].map(bh.sd_bin)
    for col in list(c.columns):
        if (
            col.startswith(("lv_a", "P_", "LB", "ssr1"))
            and c[col].dtype.kind == "f"
            and (col.endswith(("_a1", "_a2", "_a1_se", "_a2_se")) or col.startswith(("P_", "LB")))
            and col != "LB_strike"
        ):
            c[col] = c[col] / c["K"]  # fractions of the entry spot
    for col in [x for x in c.columns if x.startswith(("vega_", "theta_", "lv_a1_", "lv_a2_"))]:
        if col.endswith(("vega", "theta")) or col.startswith(("vega_", "theta_")):
            c[col] = c[col] / c["K"]
    c["w_rel"] = c["w"] / c["K"]
    if "rv_1m" in c:
        rv = np.select(
            [c["months"] == m for m in (1, 3, 6, 12)], [c[f"rv_{m}m"] for m in (1, 3, 6, 12)]
        )
        c["iv_minus_rv"] = 100.0 * (c["atm"] - rv)
    dig = (c["P_B3"] - c["P_B6"]) / (c["w_rel"] * c["DF0"])
    c["dig_touch_lv"] = dig / c["lv_p1"].where(c["lv_p1"] > 1e-4)
    c["band_pos_lv"] = (c["lv_a1"] - c["LB"]) / (c["P_B6"] - c["LB"]).where(
        c["P_B6"] - c["LB"] > 2e-5
    )
    c["ko_over_spread"] = c["lv_a1"] / c["P_B3"]
    c["regret_share"] = (c["P_B6"] - c["lv_a1"]) / c["P_B6"].where(c["P_B6"] > 2e-5)
    if "ssr12_a1" in c:
        c["fskew_diff"] = c["ssr12_fskew"] - c["lv_fskew"]
        c["lsv_gap_share"] = (c["ssr12_a1"] - c["lv_a1"]) / c["lv_a1"].where(c["lv_a1"] > 2e-4)
        c["dig_touch_lsv"] = dig / c["ssr12_p1"].where(c["ssr12_p1"] > 1e-4)
        c["band_pos_lsv"] = (c["ssr12_a1"] - c["LB"]) / (c["P_B6"] - c["LB"]).where(
            c["P_B6"] - c["LB"] > 2e-5
        )
    if "high_1y" in c:
        c["barrier_vs_1y"] = np.where(c["side"] > 0, c["B"] / c["high_1y"], c["B"] / c["low_1y"])
    return c, pos


def wide(pos: pd.DataFrame, value: str) -> pd.DataFrame:
    """cell × position table of ``value``."""
    return pos.pivot_table(index="cell", columns="position", values=value, aggfunc="first")


def stat(x: pd.Series, months: int) -> tuple[float, float, int, float]:
    """Mean, block standard error, n and effective sample of a time-ordered series of trades of
    one maturity (weekly entries)."""
    v = x.to_numpy(dtype=float)
    v = v[~np.isnan(v)]
    mean, se = bh.block_bootstrap(v, BLOCK[months], seed=7)
    return mean, se, int(v.size), bh.effective_sample(int(v.size), 7.0, DAYS[months])


def fmt(mean: float, se: float, scale: float = 100.0, digits: int = 3) -> str:
    if not np.isfinite(mean):
        return "–"
    if not np.isfinite(se):
        return f"{scale * mean:.{digits}f}"
    return f"{scale * mean:.{digits}f} ± {scale * se:.{digits}f}"


def md_table(frame: pd.DataFrame, index: bool = True) -> str:
    f = frame.reset_index() if index else frame
    head = "| " + " | ".join(str(c) for c in f.columns) + " |"
    sep = "|" + "|".join("---" for _ in f.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in f.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


# --------------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------------


class Report:
    def __init__(self, out: Path, sample: str = "strict") -> None:
        self.out = out
        self.sample = sample
        (out / "tables").mkdir(parents=True, exist_ok=True)
        (out / "figures").mkdir(parents=True, exist_ok=True)
        self.cells_all, pos = load()
        is_lsv = pos["model"] == "lsv"
        self.pos, self.pos_lsv = pos[~is_lsv].copy(), pos[is_lsv].copy()
        self.c = self.cells_all[self.cells_all[sample]].copy()
        self.fin = self.c[self.c["finished"].fillna(False).astype(bool)].copy()
        self.pnl_u = wide(self.pos, "pnl_u")
        self.pnl_h = wide(self.pos, "pnl_h") if "pnl_h" in self.pos else None
        self.terminal = wide(self.pos, "terminal")
        self.premium = wide(self.pos, "premium")
        self.md: list[str] = []
        self.figs: list[tuple[str, str]] = []
        self.blocks: list[tuple[str, Any]] = []  # ("md", text) | ("table", …) | ("figure", …)
        self.has_lsv = "ssr12_a1" in self.c
        self.has_hedge = self.pnl_h is not None and self.pnl_h.notna().any().any()

    # -- helpers -----------------------------------------------------------------------------

    def add(self, text: str) -> None:
        self.md.append(text)
        self.blocks.append(("md", text))

    def table(self, name: str, frame: pd.DataFrame, caption: str, index: bool = True) -> None:
        frame.to_csv(self.out / "tables" / f"{name}.csv", index=index)
        self.md.append(f"**Table {name}.** {caption}\n\n{md_table(frame, index)}\n")
        self.blocks.append(("table", (name, frame.reset_index() if index else frame, caption)))

    def figure(self, name: str, caption: str) -> None:
        path = self.out / "figures" / f"{name}.pdf"
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        self.figs.append((name, caption))
        self.md.append(f"*Figure {name}: {caption}* (`figures/{name}.pdf`)\n")
        self.blocks.append(("figure", (name, caption)))

    def sample_note(self, frame: pd.DataFrame) -> str:
        n = frame["entry"].nunique()
        lo, hi = frame["entry"].min(), frame["entry"].max()
        label = (
            "strict sample (eSSVI tolerances met)"
            if self.sample == "strict"
            else "all built entries"
        )
        return f"Sample: {n} entry dates {lo} to {hi}, {label}."

    def cellstat(
        self, frame: pd.DataFrame, series: pd.Series, months: int
    ) -> tuple[float, float, int, float]:
        g = frame.sort_values("entry")
        return stat(series.reindex(g["cell"]), months)

    # -- 1 coverage --------------------------------------------------------------------------

    def coverage(self) -> None:
        a = self.cells_all.drop_duplicates("entry")
        rows = []
        for name, lo, hi in bh.PERIODS:
            g = a[(a["entry"] >= lo) & (a["entry"] <= hi)]
            rows.append(
                {
                    "period": name,
                    "entry dates": len(g),
                    "surface carried (excluded)": int((~g["built"]).sum()),
                    "eSSVI tolerances failed": int((g["built"] & ~g["strict"]).sum()),
                    "strict sample": int(g["strict"].sum()),
                    "settlement uncertain": int(g["settlement_uncertain"].sum()),
                }
            )
        f = pd.DataFrame(rows).set_index("period")
        f.loc["all"] = f.sum()
        self.add("## 1. Data and coverage\n")
        self.table(
            "coverage",
            f,
            "Weekly entry dates by settlement period. An entry is excluded when its surface was "
            "carried from an earlier day (bad-day rule); the strict sample also drops the entries "
            "whose eSSVI residual exceeds the agreed tolerances (0.25 vol point 3m–2y, 0.20 6m–2y). "
            "The forward cross-check against the vendor's stkPx is not applied (its convention "
            "changes in 2024-04).",
        )
        fin = self.cells_all[self.cells_all["finished"].fillna(False).astype(bool)]
        by = fin.groupby("months").agg(
            trades=("cell", "size"),
            entries=("entry", "nunique"),
            knocked_daily=("tau1", lambda s: s.notna().mean()),
            knocked_cont=("tau2", lambda s: s.notna().mean()),
        )
        by["independent windows"] = [
            round(bh.effective_sample(int(e), 7.0, DAYS[int(m)]), 0)
            for m, e in by["entries"].items()
        ]
        by[["knocked_daily", "knocked_cont"]] = by[["knocked_daily", "knocked_cont"]].round(3)
        self.table(
            "trades",
            by,
            "Finished trades by maturity (all built entries, 18 barriers, two sides) and the share "
            "knocked on the realised path. Independent windows = entries × 7 days / maturity: "
            "inference rests on the 1-month and 3-month trades; the longer ones are indicative.",
        )

    # -- 2 prices through time ---------------------------------------------------------------

    def prices(self) -> None:
        self.add("## 2. Prices through time\n")
        c = self.c[self.c["barrier"].isin(SD)]
        rows = []
        for (m, side, b), g in c.groupby(["months", "side", "barrier"]):
            row: dict[str, Any] = {
                "maturity": f"{m}m",
                "side": SIDE[side],
                "barrier": b,
                "KO daily (LV)": f"{100 * g['lv_a1'].mean():.3f}",
                "KO cont. / daily": f"{(g['lv_a2'].sum() / g['lv_a1'].sum()):.3f}",
                "tight limit": f"{100 * g['P_B6'].mean():.3f}",
                "fly n=2": f"{100 * g['P_B5_2'].mean():.3f}",
                "KO / fly": f"{g['lv_a1'].sum() / g['P_B5_2'].sum():.3f}",
                "regret share": f"{1 - g['lv_a1'].sum() / g['P_B6'].sum():.3f}",
                "LB / KO": f"{g['LB'].sum() / g['lv_a1'].sum():.3f}",
            }
            if self.has_lsv:
                ok = g["ssr12_a1"].notna()
                for mark in ("ssr10", "ssr12", "ssr15"):
                    row[f"{mark} / LV"] = (
                        f"{g.loc[ok, f'{mark}_a1'].sum() / g.loc[ok, 'lv_a1'].sum():.3f}"
                    )
            rows.append(row)
        self.table(
            "prices",
            pd.DataFrame(rows),
            "Entry premiums in % of spot, means over the entries (standard-deviation barriers). "
            "Regret share = 1 − KO / tight limit: the part of the European knock-out the path "
            "condition removes. LB is the model-free lower bound. "
            + ("SSR columns: LSV daily knock-out over the local-vol one. " if self.has_lsv else "")
            + self.sample_note(c),
            index=False,
        )
        # figure: KO over fly and regret share through time, 3m, 1 sd
        _, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for side, ax in zip((1, -1), axes, strict=True):
            g = c[(c["months"] == 3) & (c["barrier"] == "s1") & (c["side"] == side)].sort_values(
                "entry"
            )
            t = pd.to_datetime(g["entry"])
            ax.plot(t, g["lv_a1"] / g["P_B5_2"], lw=0.8, label="KO / fly n=2 (local vol)")
            ax.plot(t, 1 - g["lv_a1"] / g["P_B6"], lw=0.8, label="regret share")
            if self.has_lsv:
                ax.plot(t, g["ssr12_a1"] / g["lv_a1"], lw=0.8, label="LSV 1.2 / local vol")
            ax.set_title(f"3m, barrier at 1 sd, {SIDE[side]} side")
            ax.legend(fontsize=7)
        self.figure("prices_time", "Knock-out against fly and tight limit through time (3m, 1 sd).")

    # -- 3 unhedged outcomes -----------------------------------------------------------------

    def outcomes(self, hedged: bool, percent: bool = False) -> None:
        src = self.pnl_h if hedged else self.pnl_u
        assert src is not None
        f = (
            self.fin[~self.fin["barrier"].isin(SD)]
            if percent
            else self.fin[self.fin["barrier"].isin(SD)]
        )
        if hedged:
            f = f[f["marks_complete"].fillna(False).astype(bool)]
        rows = []
        for (m, side, b), g in f.groupby(["months", "side", "barrier"]):
            g = g.sort_values("entry")
            row: dict[str, Any] = {"maturity": f"{m}m", "side": SIDE[side], "barrier": b}
            for p in MAIN:
                if p in src:
                    mean, se, _, _ = stat(src[p].reindex(g["cell"]), int(m))
                    row[NAMES[p]] = fmt(mean, se)
            _, _, n, eff = stat(src["A1"].reindex(g["cell"]), int(m))
            row["n"] = int(n)
            row["eff. n"] = round(eff, 0)
            rows.append(row)
        label = "delta-hedged" if hedged else "unhedged"
        self.table(
            f"outcomes_{label.replace('-', '_')}" + ("_percent" if percent else ""),
            pd.DataFrame(rows),
            f"Mean {label} P&L of buying each structure, % of spot ± block standard error "
            "(valued at expiry). KO premiums are the local-vol ones. " + self.sample_note(f),
            index=False,
        )

    # -- 11 ladder ---------------------------------------------------------------------------

    def ladder(self) -> None:
        self.add("## 11. The ladder: TR, OS, UP, W\n")
        f = self.fin[self.fin["barrier"].isin(SD)]
        prem, term = self.premium, self.terminal
        pieces = {
            "TR": ("B6", "A1"),
            "OS(2)": ("B5_2", "B6"),
            "OS(6)": ("B5_6", "B6"),
            "UP": ("B3", "B5_2"),
            "W": ("B5_2", "B4"),
        }
        rows = []
        for (m, side, b), g in f.groupby(["months", "side", "barrier"]):
            g = g.sort_values("entry")
            row: dict[str, Any] = {"maturity": f"{m}m", "side": SIDE[side], "barrier": b}
            df0 = g.set_index("cell")["DF0"]
            for name, (long_, short_) in pieces.items():
                price = (prem[long_] - prem[short_]).reindex(g["cell"])
                paid = (term[long_] - term[short_]).reindex(g["cell"]) * df0.reindex(g["cell"])
                mp, _, _, _ = stat(price, int(m))
                me, se, _, _ = stat(price - paid, int(m))
                row[f"{name} price"] = f"{100 * mp:.3f}"
                row[f"{name} priced−realised"] = fmt(me, se)
            rows.append(row)
        self.table(
            "ladder",
            pd.DataFrame(rows),
            "Each piece of the ladder: mean price at entry and mean (price minus discounted realised "
            "payoff) — what the seller of the piece earned — % of spot ± block standard error. TR "
            "(touch and return) = tight limit − knock-out (local-vol price); OS(n) = fly(n) − tight "
            "limit; UP = spread − fly(2); W = fly(2) − ratio. " + self.sample_note(f),
            index=False,
        )
        # through time: priced minus realised of TR and OS(2), 3m 1 sd, by year
        g = f[(f["months"] == 3) & (f["barrier"] == "s1")]
        _, axes = plt.subplots(1, 2, figsize=(10, 3.4))
        for side, ax in zip((1, -1), axes, strict=True):
            h = g[g["side"] == side].set_index("cell")
            for name in ("TR", "OS(2)", "UP", "W"):
                long_, short_ = pieces[name]
                e = (prem[long_] - prem[short_]).reindex(h.index) - (
                    term[long_] - term[short_]
                ).reindex(h.index) * h["DF0"]
                by = (100 * e).groupby(h["year"]).mean()
                ax.plot(by.index.astype(int), by.to_numpy(), marker="o", ms=3, lw=0.9, label=name)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(f"3m, 1 sd, {SIDE[side]} side: priced − realised, % of spot")
            ax.legend(fontsize=7)
        self.figure("ladder_time", "The seller's earnings on each piece by entry year (3m, 1 sd).")

    # -- 7 conditions ------------------------------------------------------------------------

    def conditions(self, cols: dict[str, str], title: str, name: str) -> None:
        self.add(f"## {title}\n")
        f = self.fin[self.fin["barrier"].isin(SD) & self.fin["months"].isin([1, 3])]
        diff = (self.pnl_u["B5_2"] - self.pnl_u["A1"]).rename("fly_minus_ko")
        ko = self.pnl_u["A1"]
        rows = []
        for col, label in cols.items():
            if col not in f:
                continue
            for (m, side), g in f.groupby(["months", "side"]):
                if col == "dist_sd":
                    groups = g["sd_bin"]
                    order = ["<0.5", "0.5-1", "1-1.5", ">1.5"]
                else:
                    # terciles against the history of the same condition up to the entry
                    per_entry = (
                        g.drop_duplicates("entry").sort_values("entry").set_index("entry")[col]
                    )
                    lab = bh.expanding_tercile(per_entry, min_history=104)
                    groups = g["entry"].map(lab)
                    order = ["low", "mid", "high", "no history"]
                for key in order:
                    h = g[groups == key].sort_values("entry")
                    if len(h) < 20:
                        continue
                    md_, sd_, n, eff = stat(diff.reindex(h["cell"]), int(m))
                    mk, sk, _, _ = stat(ko.reindex(h["cell"]), int(m))
                    rows.append(
                        {
                            "condition": label,
                            "maturity": f"{m}m",
                            "side": SIDE[side],
                            "group": key,
                            "fly − KO P&L": fmt(md_, sd_),
                            "KO P&L": fmt(mk, sk),
                            "n": n,
                            "eff. n": round(eff, 0),
                        }
                    )
        self.table(
            name,
            pd.DataFrame(rows),
            "Unhedged P&L of the fly (n = 2) minus the knock-out (local-vol premium), and of the "
            "knock-out alone, by group of each condition; % of spot ± block standard error; all "
            "standard-deviation barriers pooled. Groups are terciles against the condition's own "
            "history up to the entry (two years minimum); the distance uses fixed bins. "
            + self.sample_note(f),
            index=False,
        )

    # -- 8 decision rules --------------------------------------------------------------------

    def rules(self) -> None:
        self.add("## 8. Decision rules\n")
        f = self.fin[self.fin["barrier"].isin(SD)].copy()
        f = f.set_index("cell", drop=False)
        choices: dict[str, pd.Series] = {
            "always KO": pd.Series("A1", index=f.index),
            "always fly n=2": pd.Series("B5_2", index=f.index),
            "always tight limit": pd.Series("B6", index=f.index),
            "always ratio": pd.Series("B4", index=f.index),
        }
        if "iv_minus_rv" in f:
            mid = np.where(f["iv_minus_rv"] > 0, "A1", "B5_2")
            choices["1 threshold"] = pd.Series(
                np.where(f["dist_sd"] < 1.0, "B5_2", np.where(f["dist_sd"] > 1.5, "A1", mid)),
                index=f.index,
            )
        if "h_A1" in f:
            for alt, tag in (("B5_2", "fly"), ("B6", "tight limit")):
                hist = f["DF0"] * (f[f"h_{alt}"] - f["h_A1"])
                priced = f[f"P_{alt}"] - f["lv_a1"]
                choices[f"2 priced vs historical gap, LV ({tag})"] = pd.Series(
                    np.where(priced > hist, "A1", alt), index=f.index
                )
                if self.has_lsv:
                    priced_l = f[f"P_{alt}"] - f["ssr12_a1"]
                    s = pd.Series(np.where(priced_l > hist, "A1", alt), index=f.index)
                    choices[f"2 priced vs historical gap, LSV 1.2 ({tag})"] = s.where(
                        f["ssr12_a1"].notna()
                    )
        first = (
            f[(f["half"] == HALVES[0][0]) & f["lsv_gap_share"].notna()]
            if self.has_lsv
            else f.iloc[:0]
        )
        if len(first) >= 200:
            # rule 3: threshold on the LSV-minus-LV gap share, fitted on 2007–2016 only
            grid = np.quantile(first["lsv_gap_share"], np.linspace(0.05, 0.95, 19))
            best, best_x = -np.inf, float("nan")
            for x in grid:
                pick = np.where(first["lsv_gap_share"] > x, "A1", "B5_2")
                value = np.where(
                    pick == "A1",
                    self.pnl_u["A1"].reindex(first.index),
                    self.pnl_u["B5_2"].reindex(first.index),
                )
                if np.nanmean(value) > best:
                    best, best_x = float(np.nanmean(value)), float(x)
            s = pd.Series(np.where(f["lsv_gap_share"] > best_x, "A1", "B5_2"), index=f.index)
            choices[f"3 forward skew (gap share > {best_x:.3f}, fitted 2007–2016)"] = s.where(
                f["lsv_gap_share"].notna()
            )
        self.choices = choices
        rows = []
        srcs = [("unhedged", self.pnl_u)]
        if self.has_hedge:
            srcs.append(("hedged", self.pnl_h))
        for rule, pick in choices.items():
            for half, lo, hi in HALVES:
                for m in (1, 3, 6, 12):
                    g = f[(f["entry"] >= lo) & (f["entry"] <= hi) & (f["months"] == m)].sort_values(
                        "entry"
                    )
                    row: dict[str, Any] = {"rule": rule, "period": half, "maturity": f"{m}m"}
                    p = pick.reindex(g.index)
                    row["share KO"] = f"{(p == 'A1').sum() / max(p.notna().sum(), 1):.2f}"
                    for tag, src in srcs:
                        assert src is not None
                        chosen = pd.Series(np.nan, index=g.index)
                        other = pd.Series(np.nan, index=g.index)
                        for name in ("A1", "B5_2", "B6", "B4"):
                            mask = p == name
                            chosen[mask] = src[name].reindex(g.index)[mask]
                            alt = "B5_2" if name == "A1" else "A1"
                            other[mask] = src[alt].reindex(g.index)[mask]
                        mc, sc, n, _ = stat(chosen, m)
                        ms, ss, _, _ = stat(chosen - other, m)
                        row[f"{tag} P&L"] = fmt(mc, sc)
                        row[f"{tag} vs the other"] = fmt(ms, ss)
                        row[f"n {tag}"] = n
                    rows.append(row)
        self.table(
            "rules",
            pd.DataFrame(rows),
            "Each rule picks one structure per (entry, maturity, barrier, side); mean P&L of the pick "
            "and of the spread (long the pick, short the other of knock-out / fly), % of spot ± block "
            "standard error, standard-deviation barriers and both sides pooled. Rule 3's threshold "
            "is fitted on 2007–2016 and applied once to 2017–2026. " + self.sample_note(f),
            index=False,
        )

    # -- 9 break-even ------------------------------------------------------------------------

    def breakeven(self) -> None:
        self.add("## 9. Break-even knock-out premium\n")
        f = self.fin[self.fin["barrier"].isin(SD)].set_index("cell", drop=False)
        rows = []
        for (m, side, b), g in f.groupby(["months", "side", "barrier"]):
            row: dict[str, Any] = {"maturity": f"{m}m", "side": SIDE[side], "barrier": b}
            ko_pay = g["DF0"] * self.terminal["A1"].reindex(g.index)
            for n_ in (2, 6):
                p = f"B5_{n_}"
                star = ko_pay - (
                    g["DF0"] * self.terminal[p].reindex(g.index) - self.premium[p].reindex(g.index)
                )
                row[f"P* vs fly n={n_}, % spot"] = f"{100 * star.mean():.3f}"
                row[f"P* / LV (n={n_})"] = f"{star.mean() / g['lv_a1'].mean():.2f}"
                if self.has_lsv:
                    ok = g["ssr12_a1"].notna()
                    for mark in ("ssr10", "ssr12", "ssr15"):
                        row[f"P* / {mark} (n={n_})"] = (
                            f"{star[ok].mean() / g.loc[ok, f'{mark}_a1'].mean():.2f}"
                        )
            rows.append(row)
        self.table(
            "breakeven",
            pd.DataFrame(rows),
            "P* = the knock-out premium at which the knock-out would have done exactly as well as "
            "the fly on the realised paths (mean over the trades), in % of spot and as a multiple "
            "of the model premiums: above 1, the model premium was cheap against the fly. "
            + self.sample_note(f),
            index=False,
        )

    # -- value along the way, residual, risk -------------------------------------------------

    def along(self) -> None:
        self.add("## 5. Value along the way\n")
        f = self.fin[
            self.fin["barrier"].isin(SD) & self.fin["marks_complete"].fillna(False).astype(bool)
        ]
        rows = []
        for (m, side), g in f.groupby(["months", "side"]):
            for p in ("A1", "B5_2", "B6", "B4"):
                sub = self.pos[(self.pos["position"] == p) & self.pos["cell"].isin(g["cell"])]
                if sub.empty or "run_50" not in sub:
                    continue
                rows.append(
                    {
                        "maturity": f"{m}m",
                        "side": SIDE[side],
                        "structure": NAMES[p],
                        "at 25 %": f"{100 * sub['run_25'].mean():.3f}",
                        "at 50 %": f"{100 * sub['run_50'].mean():.3f}",
                        "at 75 %": f"{100 * sub['run_75'].mean():.3f}",
                        "end": f"{100 * sub['pnl_h'].mean():.3f}",
                        "mean best": f"{100 * sub['best'].mean():.3f}",
                        "mean worst": f"{100 * sub['worst'].mean():.3f}",
                        "5 % worst end": f"{100 * sub['pnl_h'].quantile(0.05):.3f}",
                    }
                )
        self.table(
            "along",
            pd.DataFrame(rows),
            "Hedged running P&L (mark at the day's surface or model, plus the hedge so far), mean over "
            "trades, % of spot, at fractions of the life; the best and worst points of the path. "
            + self.sample_note(f),
            index=False,
        )

    def residual(self) -> None:
        self.add("## 6. The dealer's residual: C7–C9 sold at the touch\n")
        sold = (
            self.pos[self.pos.get("sold", False).fillna(False).astype(bool)]
            if "sold" in self.pos
            else self.pos.iloc[:0]
        )
        sold = sold.merge(self.fin[["cell", "months", "side", "barrier", "w_rel"]], on="cell")
        sold = sold[sold["barrier"].isin(SD) & sold["sale"].notna()]
        rows = []
        for (m, side, p), g in sold.groupby(["months", "side", "position"]):
            x = g["sale"] / g["w_rel"]
            rows.append(
                {
                    "maturity": f"{m}m",
                    "side": SIDE[side],
                    "structure": p,
                    "n touched": len(g),
                    "mean sale, % spot": f"{100 * g['sale'].mean():.3f}",
                    "mean / w": f"{x.mean():.3f}",
                    "sd / w": f"{x.std():.3f}",
                    "5 %": f"{x.quantile(0.05):.3f}",
                    "median": f"{x.median():.3f}",
                    "95 %": f"{x.quantile(0.95):.3f}",
                }
            )
        self.table(
            "residual",
            pd.DataFrame(rows),
            "Value of the static hedges on the first daily knock (sold at that day's surface mid, "
            "snapshot spot): what a dealer hedged with them carries. In % of spot and as a share "
            "of the barrier distance w. Zero would be a perfect hedge. "
            + self.sample_note(self.fin),
            index=False,
        )

    def risk(self) -> None:
        self.add("## 12b. Risk profile at entry\n")
        c = self.c[self.c["barrier"].isin(SD)]
        rows = []
        for (m, side, b), g in c.groupby(["months", "side", "barrier"]):
            row = {"maturity": f"{m}m", "side": SIDE[side], "barrier": b}
            row["vega KO daily"] = f"{100 * g['lv_a1_vega'].mean():.4f}"
            row["vega KO cont."] = f"{100 * g['lv_a2_vega'].mean():.4f}"
            for p in ("B3", "B4", "B5_2", "B6"):
                row[f"vega {NAMES[p]}"] = f"{100 * g[f'vega_{p}'].mean():.4f}"
            row["theta KO daily"] = f"{100 * g['lv_a1_theta'].mean():.4f}"
            for p in ("B4", "B5_2", "B6"):
                row[f"theta {NAMES[p]}"] = f"{100 * g[f'theta_{p}'].mean():.4f}"
            rows.append(row)
        self.table(
            "risk",
            pd.DataFrame(rows),
            "Vega to a parallel +1 vol point move of the surface (local vol rebuilt for the "
            "knock-outs) and one-calendar-day theta, % of spot, means over the entries. "
            + self.sample_note(c),
            index=False,
        )

    def lsv_study(self) -> None:
        """Step 6: every output that depends on the knock-out's model, under the LSV (SSR 1.2)
        beside local vol, on the part of the history the LSV daily series covers."""
        if self.pos_lsv.empty or "pnl_h" not in self.pos_lsv:
            return
        lsv = self.pos_lsv.dropna(subset=["pnl_h"]).set_index(["cell", "position"])
        lv = self.pos[self.pos["position"].isin(["A1", "A2"])].set_index(["cell", "position"])
        both = lsv.join(lv, lsuffix="_lsv", rsuffix="_lv", how="inner").reset_index()
        both = both.merge(self.fin[["cell", "entry", "months", "side", "barrier"]], on="cell")
        both = both[both["barrier"].isin(SD)]
        if both.empty:
            return
        lo, hi = both["entry"].min(), both["entry"].max()
        self.add("## 4b. The LSV study: knock-outs marked and hedged under the LSV (SSR 1.2)\n")
        self.add(
            f"Coverage of the LSV daily series: {both['entry'].nunique()} entry dates from {lo} to "
            f"{hi} ({both['cell'].nunique()} knock-out trades with complete daily marks under both "
            "models). The leverage is recalibrated every day on the day's surface and on each "
            "bumped surface; the parameters are re-marked on the first trading day of each week.\n"
        )
        rows = []
        for (m, side, b, p), g in both.groupby(["months", "side", "barrier", "position"]):
            g = g.sort_values("entry")
            row: dict[str, Any] = {
                "maturity": f"{m}m",
                "side": SIDE[side],
                "barrier": b,
                "KO": NAMES[p],
            }
            for tag in ("lv", "lsv"):
                mu, su, _, _ = stat(g[f"pnl_u_{tag}"], int(m))
                mh, sh, _, _ = stat(g[f"pnl_h_{tag}"], int(m))
                row[f"premium {tag.upper()}"] = f"{100 * g[f'premium_{tag}'].mean():.3f}"
                row[f"unhedged {tag.upper()}"] = fmt(mu, su)
                row[f"hedged {tag.upper()}"] = fmt(mh, sh)
                row[f"hedged sd {tag.upper()}"] = f"{100 * g[f'pnl_h_{tag}'].std():.3f}"
                row[f"delta at entry {tag.upper()}"] = f"{g[f'delta0_{tag}'].mean():.3f}"
            rk = "remark_pnl_lsv" if "remark_pnl_lsv" in g else "remark_pnl"
            row["LSV re-mark days' P&L"] = f"{100 * g[rk].mean():.3f}"
            row["n"] = len(g)
            rows.append(row)
        self.table(
            "lsv_hedged",
            pd.DataFrame(rows),
            "The same knock-out trades under the two models: entry premium, mean unhedged and "
            "delta-hedged P&L (% of spot ± block standard error), the dispersion of the hedged P&L, "
            "the spot delta at entry, and the part of the LSV hedged P&L that arrives on parameter "
            "re-mark days. Each model's P&L uses its own premium, marks and sticky-strike delta.",
            index=False,
        )
        # the touch-and-return piece under each model
        rows = []
        a1 = both[both["position"] == "A1"].set_index("cell")
        fin = self.fin.set_index("cell")
        for (m, side, b), g in a1.groupby(["months", "side", "barrier"]):
            g = g.sort_values("entry")
            b6 = fin.loc[g.index, "P_B6"]
            paid = (self.terminal["B6"].reindex(g.index) - g["terminal_lv"]) * fin.loc[
                g.index, "DF0"
            ]
            row = {"maturity": f"{m}m", "side": SIDE[side], "barrier": b, "n": len(g)}
            for tag in ("lv", "lsv"):
                price = b6 - g[f"premium_{tag}"]
                me, se, _, _ = stat(price - paid, int(m))
                row[f"TR price {tag.upper()}"] = f"{100 * price.mean():.3f}"
                row[f"TR priced−realised {tag.upper()}"] = fmt(me, se)
            rows.append(row)
        self.table(
            "lsv_ladder",
            pd.DataFrame(rows),
            "The touch-and-return piece (tight limit minus knock-out), the only model-dependent "
            "piece of the ladder, priced under each model against its realised payoff.",
            index=False,
        )

    def limits(self) -> None:
        self.add("## 10. Limits\n")
        self.add(
            "- **Mid prices.** Every vanilla structure is priced at the mid of the day's eSSVI "
            "surface, not at market mids: on the pilot window the surface is within 0.1 vol point "
            "of the quoted mid at the strike, 0.2–0.6 at the barrier and up to 1.2 at the far fly "
            "strike, more than the quoted half-spread in almost every cell. The cost table adds "
            "half the quoted bid-ask; it does not correct this.\n"
            "- **Model prices for the knock-out.** No market price of the knock-out exists: its "
            "premium is the local-vol price (and the LSV price at three marks); the break-even "
            "premium says how far a dealer's price may be from it before the conclusion changes.\n"
            "- **Snapshot against close.** Strikes, marks and deltas use the vendor's snapshot "
            "(about 14 minutes before the close); barrier observation and the payoff use the "
            "official close, high and low. The knock day's hedge is closed at the official close.\n"
            "- **Weekly parameter re-marks.** The LSV parameters change on the first trading day of "
            "each week; the P&L of those days is reported apart.\n"
            "- **The hedge** is a forward to the trade's expiry, rebalanced once a day at the "
            "snapshot, with the sticky-strike delta; no transaction cost on the hedge.\n"
            "- **Settlement flags.** Before 2011 the settlement of the non-monthly SPX expiries is a "
            "rule (Friday expiries AM before 2010-12-01, quarter ends PM), December 2010 is flagged "
            "uncertain; from 2017-05-10 to 2021-05-27 the settlement of each monthly comes from the "
            "open interest.\n"
            "- **Surface quality.** The strict sample drops the entries whose eSSVI residual exceeds "
            "the agreed tolerances: 72 % of the entries, but 33–46 % of those of 2017-05 to 2021-05.\n"
            "- **Particles.** LSV leverages use 100,000 particles with a 1 % regression floor "
            "(pilot check 9: prices and deltas within tolerance of 200,000 on 216 cells).\n"
            "- **Monte Carlo.** 100,000 antithetic paths per (day, bump), the European knock-out as "
            "control variate; error at most 1 % of the value or 0.5 bp of spot on the pilot cells.\n"
            "- **Inference.** About 236 independent one-month windows and 78 three-month ones; 39 "
            "six-month and 19 one-year ones: the long maturities are indicative only.\n"
        )
        self.add("## References\n")
        self.add(
            "Brown, Hobson, Rogers (2001), Robust hedging of barrier options. Carr, Ellis, Gupta "
            "(1998), Static hedging of exotic options. Derman, Ergener, Kani (1995), Static options "
            "replication. Nalholm, Poulsen (2006), Static hedging of barrier options under general "
            "asset dynamics. Maruhn, Nalholm, Fengler (2011), Static hedges for reverse barrier "
            "options with robustness against skew risk. Broadie, Glasserman, Kou (1997), A continuity "
            "correction for discrete barrier options. Bergomi (2009), Smile dynamics IV. Bakshi, "
            "Madan, Panayotov (2010), Returns of claims on the upside and the viability of U-shaped "
            "pricing kernels. Hu, Liu (2022), on the returns of upside claims.\n"
        )

    def strategy(self) -> None:
        path = bh.RESULTS / "book.parquet"
        if not path.exists():
            return
        self.add("## 12a. Strategy statistics and risk\n")
        book = pd.read_parquet(path)
        ohlc = bh.load_ohlc()
        spx = np.log(ohlc["close"]).diff()
        vix_path = bh.HISTORY / "VIX.csv"
        vix = pd.read_csv(vix_path, dtype={"date": str}).set_index("date")["close"].diff()
        days = sorted(book["date"].unique())
        fin = self.fin.set_index("cell")
        rows = []
        for (m, side, b, p), g in book.groupby(["months", "side", "barrier", "position"]):
            if m not in (1, 3) or b not in ("s0.5", "s1", "s1.5"):
                continue
            s_ = g.set_index("date").reindex(days).fillna(0.0)
            trades = self.pos[
                (self.pos["position"] == p)
                & self.pos["cell"].isin(
                    fin.index[(fin["months"] == m) & (fin["side"] == side) & (fin["barrier"] == b)]
                )
            ]
            row: dict[str, Any] = {
                "maturity": f"{m}m",
                "side": SIDE[side],
                "barrier": b,
                "structure": NAMES[p],
            }
            for tag, col in (("unhedged", "du"), ("hedged", "dh")):
                x = s_[col]
                vol = x.std() * np.sqrt(252.0)
                cum = x.cumsum()
                month = x.groupby(x.index.str[:7]).sum()
                row[f"{tag} mean /y"] = f"{100 * 252 * x.mean():.2f}"
                row[f"{tag} vol /y"] = f"{100 * vol:.2f}"
                row[f"{tag} Sharpe"] = f"{252 * x.mean() / vol:.2f}" if vol > 0 else "–"
                row[f"{tag} max drawdown"] = f"{100 * (cum - cum.cummax()).min():.2f}"
                row[f"{tag} worst month"] = f"{100 * month.min():.2f}"
                row[f"{tag} skew (monthly)"] = f"{month.skew():.2f}"
            x = s_["dh"]
            reg = pd.DataFrame(
                {"y": x, "spx": spx.reindex(days), "vix": vix.reindex(days)}
            ).dropna()
            if len(reg) > 100:
                beta = np.linalg.lstsq(
                    np.column_stack([np.ones(len(reg)), reg["spx"], reg["vix"]]),
                    reg["y"],
                    rcond=None,
                )[0]
                row["hedged beta S&P"] = f"{beta[1]:.3f}"
                row["hedged beta VIX (per pt)"] = f"{100 * beta[2]:.4f}"
            if len(trades):
                row["5 % worst trade"] = f"{100 * trades['pnl_u'].quantile(0.05):.2f}"
                row["1 % worst trade"] = f"{100 * trades['pnl_u'].quantile(0.01):.2f}"
            if p == "B4" and len(trades):
                lost = trades["terminal"] < 0
                row["ratio: share beyond break-even"] = f"{lost.mean():.3f}"
                row["ratio: mean payoff there"] = (
                    f"{100 * trades.loc[lost, 'terminal'].mean():.2f}" if lost.any() else "–"
                )
            rows.append(row)
        frame = pd.DataFrame(rows).fillna("–")
        keys = ["maturity", "side", "barrier", "structure"]
        note = (
            "The rolling book that buys one unit at every weekly entry and holds to expiry, {} "
            "daily P&L, % of spot: annualised mean and volatility, Sharpe ratio, maximum drawdown, "
            "worst calendar month, skewness of monthly P&L{}. All built entries."
        )
        un = [c for c in frame.columns if c.startswith("unhedged")] + [
            c for c in frame.columns if "worst trade" in c or c.startswith("ratio:")
        ]
        self.table(
            "strategy_unhedged",
            frame[keys + un],
            note.format("unhedged", "; the worst trade outcomes; for the ratio, the share of trades "
                        "finishing with a negative payoff and the mean payoff there"),
            index=False,
        )  # fmt: skip
        hd = [c for c in frame.columns if c.startswith("hedged")]
        self.table(
            "strategy_hedged",
            frame[keys + hd],
            note.format("delta-hedged", "; betas to S&P log returns and to VIX changes"),
            index=False,
        )

    def costs(self) -> None:
        if "cost_B3" not in self.c:
            return
        self.add("## 12c. Cost sensitivity\n")
        f = self.fin[self.fin["barrier"].isin(SD)].set_index("cell", drop=False)
        rows = []
        for (m, side, b), g in f.groupby(["months", "side", "barrier"]):
            row: dict[str, Any] = {"maturity": f"{m}m", "side": SIDE[side], "barrier": b}
            for p in ("B3", "B4", "B5_2", "B6", "E13"):
                cost = g[f"cost_{p}"]
                prem = g[f"P_{p}"]
                net = self.pnl_u[p].reindex(g.index) - cost / g["DF0"]
                share = f"{100 * cost.sum() / prem.sum():.0f} %" if prem.sum() > 0 else "–"
                row[f"{NAMES[p]} cost"] = f"{100 * cost.mean():.3f} ({share})"
                row[f"{NAMES[p]} net"] = f"{100 * net.mean():.3f}"
            rows.append(row)
        self.table(
            "costs",
            pd.DataFrame(rows),
            "Cost of each vanilla structure = Σ over legs |quantity| × half the quoted bid-ask in "
            "vol at the nearest listed strike × the leg's vega, % of spot (share of the premium in "
            "brackets), and the mean unhedged P&L net of it. The tight limit's digital is costed as "
            "its ±0.5 % call spread (large quantities: it is not a tradable structure). No cost is "
            "assumed for the knock-out: its break-even premium says how much a dealer may charge "
            "before the fly wins. " + self.sample_note(f),
            index=False,
        )

    def events(self) -> None:
        if "knock_on_fomc" not in self.fin:
            return
        f = self.fin[(self.fin["months"] == 1) & self.fin["barrier"].isin(SD)]
        knocked = f[f["tau1"].notna()]
        rows = []
        for side, g in knocked.groupby("side"):
            base = f[f["side"] == side]
            rows.append(
                {
                    "side": SIDE[side],
                    "1m trades": len(base),
                    "knocked (daily)": len(g),
                    "knocks on a Fed meeting day": int(g["knock_on_fomc"].sum()),
                    "share": f"{g['knock_on_fomc'].mean():.3f}",
                    "Fed meeting days / trading days": f"{8 / 252:.3f}",
                    "trades with a meeting in the life": f"{(base['fomc_in_life'] > 0).mean():.2f}",
                }
            )
        self.table(
            "events",
            pd.DataFrame(rows),
            "Share of the daily knocks of the 1-month trades that fall on the last day of a Fed "
            "meeting (dates from federalreserve.gov), against the share of such days among trading "
            "days. CPI and payroll dates are not used: no official source was fetched. "
            + self.sample_note(f),
            index=False,
        )

    # -- 13 decision map, 14 today -----------------------------------------------------------

    def decision_map(self) -> None:
        self.add("## 13. Decision map\n")
        f = self.fin[self.fin["barrier"].isin(SD) & self.fin["months"].isin([1, 3])]
        diff = self.pnl_u["A1"] - self.pnl_u["B5_2"]
        rows = []
        for col, label in PRIMARY.items():
            if col not in f or col == "dist_sd":
                continue
            for (m, side, sdb), g in f.groupby(["months", "side", "sd_bin"]):
                per_entry = g.drop_duplicates("entry").sort_values("entry").set_index("entry")[col]
                lab = g["entry"].map(bh.expanding_tercile(per_entry, min_history=104))
                for key in ("low", "high"):
                    h = g[lab == key]
                    out: dict[str, Any] = {
                        "condition": label,
                        "maturity": f"{m}m",
                        "side": SIDE[side],
                        "distance": sdb,
                        "group": key,
                    }
                    signs = []
                    for half, lo, hi in HALVES:
                        hh = h[(h["entry"] >= lo) & (h["entry"] <= hi)].sort_values("entry")
                        mean, se, n, _ = stat(diff.reindex(hh["cell"]), int(m))
                        out[f"KO − fly, {half}"] = fmt(mean, se)
                        out[f"n {half}"] = n
                        signs.append(np.sign(mean) if n >= 20 else np.nan)
                    mean, se, _, _ = stat(diff.reindex(h.sort_values("entry")["cell"]), int(m))
                    out["preferred"] = "KO" if mean > 0 else "fly"
                    out["confirmed"] = (
                        "yes" if signs[0] == signs[1] else ("–" if np.isnan(signs).any() else "no")
                    )
                    rows.append(out)
        self.table(
            "decision_map",
            pd.DataFrame(rows),
            "For each maturity × distance bin × primary condition (low and high terciles): mean "
            "unhedged P&L of the knock-out minus the fly (n = 2), % of spot ± block standard error, "
            "in each half; the preferred structure over the whole sample; confirmed = the two "
            "halves agree in sign. " + self.sample_note(f),
            index=False,
        )

    def today(self) -> None:
        self.add("## 14. Today's reading\n")
        a = self.cells_all[self.cells_all["built"]]
        last = a["entry"].max()
        t = a[a["entry"] == last]
        metrics = [m for m in (*PRIMARY, "atm", "convexity", "term_slope", "vix", "vvix", "cor3m",
                               "lsv_gap_share", "regret_share", "ko_over_spread", "band_pos_lv") if m in a]  # fmt: skip
        rows = []
        for m_ in (1, 3):
            for side in (1, -1):
                now = t[(t["months"] == m_) & (t["side"] == side) & (t["barrier"] == "s1")]
                hist = a[(a["months"] == m_) & (a["side"] == side) & (a["barrier"] == "s1")]
                if now.empty:
                    continue
                for met in metrics:
                    v = float(now[met].iloc[0])
                    if not np.isfinite(v):
                        continue
                    pct = float((hist[met].dropna() <= v).mean())
                    rows.append(
                        {
                            "maturity": f"{m_}m",
                            "side": SIDE[side],
                            "metric": PRIMARY.get(met, met),
                            "value": f"{v:.4g}",
                            "percentile": f"{100 * pct:.0f}",
                        }
                    )
        self.table(
            "today",
            pd.DataFrame(rows),
            f"Every metric on the latest entry date ({last}), barrier at 1 sd, with its percentile "
            "against its own history of weekly entries.",
            index=False,
        )
        if hasattr(self, "choices"):
            rows = []
            for rule, pick in self.choices.items():
                p = pick.reindex(t["cell"])
                tt = t.set_index("cell")
                for m_ in (1, 3):
                    sel = p[(tt["months"] == m_) & tt["barrier"].isin(SD)]
                    if sel.notna().any():
                        rows.append(
                            {
                                "rule": rule,
                                "maturity": f"{m_}m",
                                **{
                                    f"{SIDE[int(tt.loc[c_, 'side'])]} {tt.loc[c_, 'barrier']}": NAMES.get(
                                        str(v), "–"
                                    )
                                    for c_, v in sel.items()
                                },
                            }
                        )
            if rows:
                self.table("today_rules", pd.DataFrame(rows).fillna("–"),
                           f"Each rule's choice on {last} (the latest entries are unfinished: the "
                           "rules use entry data only).", index=False)  # fmt: skip

    # -- assembly ----------------------------------------------------------------------------

    def build(self) -> None:
        title = "Barrier study on 2007–2026: knock-outs against ratios and flies"
        self.add(f"# {title}\n")
        self.add(
            f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}. Companion to the barrier-versus-vanilla "
            "study (`docs/barrier_vs_vanilla.md`): the same question on every weekly entry of the "
            "ORATS history. Amounts are per unit notional 1/S0, shown ×100 as % of the entry spot. "
            f"Main tables use the standard-deviation barriers and the {self.sample} sample.\n"
        )
        self.coverage()
        self.prices()
        self.add("## 3. Unhedged outcomes\n")
        self.outcomes(hedged=False)
        if self.has_hedge:
            self.add("## 4. Delta-hedged outcomes (local-vol sticky-strike delta)\n")
            self.outcomes(hedged=True)
            self.lsv_study()
            self.along()
            self.residual()
        self.conditions(PRIMARY, "7. Conditions at entry: primary", "conditions_primary")
        self.rules()
        self.breakeven()
        self.ladder()
        self.strategy()
        self.risk()
        self.costs()
        self.conditions(
            EXPLORATORY, "7b. Conditions at entry: exploratory", "conditions_exploratory"
        )
        self.events()
        self.decision_map()
        self.today()
        self.limits()
        self.add("## Appendix: percent barriers\n")
        self.outcomes(hedged=False, percent=True)
        if self.has_hedge:
            self.outcomes(hedged=True, percent=True)
        (self.out / "report.md").write_text("\n".join(self.md))

    def pdf(self) -> dict[str, Any]:
        parts: list[str] = []
        for kind, payload in self.blocks:
            if kind == "md":
                parts.append(lx.markdown_to_latex(payload))
            elif kind == "figure":
                name, caption = payload
                parts += [
                    r"\begin{figure}[htbp]",
                    r"\centering",
                    rf"\includegraphics[width=0.85\linewidth]{{figures/{name}.pdf}}",
                    rf"\caption{{{lx.latex_escape(caption)}}}",
                    r"\end{figure}",
                    r"\FloatBarrier",
                ]
            else:
                name, frame, caption = payload
                n = frame.shape[1]
                size = r"\tiny" if n > 12 else (r"\scriptsize" if n > 8 else r"\footnotesize")
                sep = "2pt" if n > 12 else "3pt"
                esc = lx.latex_escape
                # long text columns wrap (a paragraph column), the others stay as they are
                longest = [
                    max([len(str(c))] + [len(str(v)) for v in frame[c]]) for c in frame.columns
                ]
                spec = "".join(
                    (
                        (r">{\raggedright\arraybackslash}p{" + f"{min(0.11 * w_, 7.5):.1f}cm" + "}")
                        if w_ > 45
                        else "l"
                    )
                    for w_ in longest
                )
                head = " & ".join(
                    rf"\shortstack[l]{{{esc(str(c)).replace(' ', r' \\ ', 1) if n > 12 and len(str(c)) > 9 else esc(str(c))}}}"
                    for c in frame.columns
                )
                parts += [
                    r"\begingroup",
                    size,
                    rf"\setlength\tabcolsep{{{sep}}}",
                    rf"\noindent\textbf{{Table {esc(name)}.}} {lx.inline_to_latex(caption)}",
                    "",
                    rf"\begin{{longtable}}{{@{{}}{spec}@{{}}}}",
                    r"\toprule",
                    head + r" \\",
                    r"\midrule",
                    r"\endhead",
                ]
                for row in frame.itertuples(index=False):
                    parts.append(" & ".join(esc(str(v)) for v in row) + r" \\")
                parts += [r"\bottomrule", r"\end{longtable}", r"\endgroup", ""]
        body = "\n".join(parts)
        preamble = (
            (lx.TEX_PREAMBLE + "\\usepackage{array}\n")
            .replace(
                r"\usepackage[margin=2.5cm]{geometry}",
                r"\usepackage[landscape,margin=1.3cm]{geometry}",
            )
            .replace(r"\documentclass[11pt]{article}", r"\documentclass[10pt]{article}")
        )
        tex = preamble.rstrip("\n") + "\n\\begin{document}\n" + body + "\n\\end{document}\n"
        (self.out / "report.tex").write_text(tex)
        return lx.compile_latex(self.out, "report.tex")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", default=str(bh.OUT / "report"))
    ap.add_argument("--sample", default="strict", choices=["strict", "built"])
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    rep = Report(Path(args.out), args.sample)
    rep.build()
    print(f"report.md: {len(rep.md)} blocks, {len(rep.figs)} figures")
    if not args.no_pdf:
        res = rep.pdf()
        print({k: res[k] for k in ("status", "seconds", "pdf", "problems", "reason") if k in res})


if __name__ == "__main__":
    main()
