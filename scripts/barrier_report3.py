"""Report v3 of the barrier study: the sections of addendum 2 (BARRIER_STUDY_ADDENDUM2).

    python scripts/barrier_report3.py [--out outputs/interview/report] [--sample strict|built]

Everything of report v2 (``scripts/barrier_report2.py``) is kept.  Just before the verdict table
come the new sections, numbered as in the addendum with the prefix ``A2.`` (v2 already uses
11g–11j and 12a): the attribution of the hedged P&L into vol carry and the rest, the justified
share, the conditions and the rules ex vol carry, the touch frequencies against the models,
turnover and hedging costs, the skew sensitivity when it has run, the LSV-hedged comparison by
period, the strategy series of the rules, today's picks and the framework table.  The verdict
table then carries P12–P14, the findings are written at the top, and ``pm_framework.md`` (one
page) is written beside the report.  Nothing is priced: the inputs are ``barrier_results/``
(``positions_attrib.parquet`` and ``turnover.parquet`` from ``scripts/barrier_attrib.py``).
"""

# ruff: noqa: RUF001, E501 — report prose: typographic signs and long caption lines
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barrier_report as base
import barrier_report2 as r2

from volsto.studies import barrier_attrib as ba
from volsto.studies import barrier_history as bh
from volsto.studies import barrier_theory as bt

SD, SIDE, BLOCK, HALVES, NAMES = base.SD, base.SIDE, base.BLOCK, base.HALVES, base.NAMES
PERIODS = (("all", "2007", "2027"), *HALVES)
#: Structures of the attribution and cost tables, and the candidates of the framework table.
ATTRIB = ("A1", "A2", "B5_2", "B5_6", "B3", "B4", "B6")
PAIRS = (("B5_2", "fly n=2 − KO"), ("B6", "tight limit − KO"))
CANDIDATES = ("A1", "B5_2", "B5_6", "B4", "B6")
#: Conditions of the touch-frequency table (addendum 2 §3).
TOUCH_CONDITIONS = {
    "past 3-month return": "ret_3m",
    "vol premium": "vrp_ratio",
    "implied minus realised vol": "iv_minus_rv",
    "level of vol": "atm",
}
fmt, entry_stat, labels = r2.fmt, r2.entry_stat, r2.labels


def interval(x: tuple[float, ...]) -> str:
    return f"{x[0]:.2f} [{x[1]:.2f}, {x[2]:.2f}]" if np.isfinite(x[0]) else "–"


def interval_verdict(lo: float, hi: float) -> str:
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "inconclusive"
    return "confirmed" if (lo >= 0 and hi <= 1) else ("contradicted" if (hi < 0 or lo > 1) else "inconclusive")  # fmt: skip


class Report3(r2.Report2):
    def __init__(self, out: Path, sample: str = "strict") -> None:
        super().__init__(out, sample)
        att = self.read("positions_attrib.parquet")
        self.key: dict[str, Any] = {}
        self.pnl_hx: pd.DataFrame | None = None
        if att is not None:
            own = att[att["model"] != "lsv"]
            self.pnl_hx = base.wide(own, "pnl_hx")
            self.carry = base.wide(own, "carry")
            self.vega = base.wide(own, "vega")
            self.dsig = own.groupby("cell")["dsig"].first()
            self.att_lsv = att[att["model"] == "lsv"]
        self.turn = self.read("turnover.parquet")

    # -- helpers -----------------------------------------------------------------------------

    def col(self, g: pd.DataFrame, wide: pd.DataFrame, name: str) -> pd.Series:
        """Column ``name`` of a cell-indexed wide table, aligned on the rows of ``g``."""
        return pd.Series(wide[name].reindex(g["cell"]).to_numpy(), index=g.index)

    def es(self, g: pd.DataFrame, values: pd.Series, m: int) -> tuple[float, float, int]:
        return entry_stat(g, values, int(m))

    def hedged_frame(self) -> pd.DataFrame:
        """The finished sd-barrier trades with complete marks (every maturity)."""
        f = self.fin[
            self.fin["barrier"].isin(SD) & self.fin["marks_complete"].fillna(False).astype(bool)
        ].copy()
        f["touched"] = f["tau1"].notna().astype(float)
        f["touched2"] = f["tau2"].notna().astype(float)
        return f

    def shown(self, name: str, full: pd.DataFrame, shown: pd.DataFrame, caption: str) -> None:
        """A table whose CSV holds every row and whose report shows a subset."""
        self.table(name, shown, caption, index=False)
        full.to_csv(self.out / "tables" / f"{name}.csv", index=False)

    # -- A2.11g attribution ------------------------------------------------------------------

    def attribution(self) -> None:
        if self.pnl_hx is None or not self.has_hedge:
            return self.missing("A2.11g. Attribution", "positions_attrib.parquet does not exist")
        self.add("## A2.11g. Attribution of the hedged P&L: vol carry and the rest\n")
        self.add(
            "The delta-hedged P&L of a structure is read as *vol carry + forward-skew premium + touch "
            "cost + noise*. The vol carry is the entry vega (to a parallel move of one vol point) times "
            "the realised vol premium of the trade's life, `dsig` = realised close-to-close vol from "
            "entry to expiry minus the entry at-the-money vol, in vol points; it is a choice of vega "
            "sign that vanillas alone give. `ex vol` is the hedged P&L minus that carry: what the "
            "framework's question bears on.\n"
        )
        f = self.hedged_frame()
        h, hx, ca = self.pnl_h, self.pnl_hx, self.carry
        assert h is not None
        rows = []
        for (m, side, b), g in f.groupby(["months", "side", "barrier"]):
            dsig = pd.Series(self.dsig.reindex(g["cell"]).to_numpy(), index=g.index)
            d_mean, d_se, _ = self.es(g, dsig, m)
            for name in ATTRIB:
                st = [self.es(g, self.col(g, w, name), m) for w in (h, ca, hx)]
                rows.append(
                    {
                        "maturity": f"{m}m", "side": SIDE[side], "barrier": b,
                        "structure": NAMES[name],
                        "vega": f"{100 * self.col(g, self.vega, name).mean():.3f}",
                        "dsig": fmt(d_mean, d_se, 1.0, 2),
                        "hedged": fmt(*st[0][:2]), "carry": fmt(*st[1][:2]),
                        "ex vol": fmt(*st[2][:2]), "n": st[2][2],
                        "LSV − LV (sgn)": "", "Π_d LV": "", "Π_d LSV": "",
                    }
                )  # fmt: skip
                self.key[("attrib", int(m), int(side), b, name)] = tuple(x[0] for x in st) + tuple(x[1] for x in st)  # fmt: skip
            for name, label in PAIRS:
                st = [
                    self.es(g, self.col(g, w, name) - self.col(g, w, "A1"), m) for w in (h, ca, hx)
                ]
                row = {
                    "maturity": f"{m}m", "side": SIDE[side], "barrier": b, "structure": label,
                    "vega": f"{100 * (self.col(g, self.vega, name) - self.col(g, self.vega, 'A1')).mean():.3f}",
                    "dsig": fmt(d_mean, d_se, 1.0, 2),
                    "hedged": fmt(*st[0][:2]), "carry": fmt(*st[1][:2]),
                    "ex vol": fmt(*st[2][:2]), "n": st[2][2],
                    "LSV − LV (sgn)": "", "Π_d LV": "", "Π_d LSV": "",
                }  # fmt: skip
                if "lsv_minus_lv" in g:
                    row["LSV − LV (sgn)"] = f"{100 * (g['sgn'] * g['lsv_minus_lv']).mean():.3f}"
                    row["Π_d LV"] = f"{100 * g['Pi_d_lv'].mean():.3f}"
                    row["Π_d LSV"] = f"{100 * g['Pi_d_ssr12'].mean():.3f}"
                rows.append(row)
                self.key[("attrib", int(m), int(side), b, name + "-A1")] = tuple(x[0] for x in st) + tuple(x[1] for x in st)  # fmt: skip
        self.table(
            "attribution",
            pd.DataFrame(rows),
            "By side, maturity and barrier: the mean entry vega (% of spot per vol point), the mean "
            "realised vol premium of the life (`dsig`, vol points), and the mean delta-hedged P&L, "
            "its vol carry (vega × dsig) and the hedged P&L ex carry, % of spot ± block standard "
            "error over entry dates; for the pairs against the daily knock-out, the signed LSV-minus-"
            "local-vol premium and what each anchor charges for leaving C8 (Π_d) beside. Knock-outs "
            "at the local-vol premium, local-vol sticky-strike delta. " + self.sample_note(f),
            index=False,
        )

    # -- 2.3 the regression check ------------------------------------------------------------

    def carry_regression(self) -> None:
        if self.pnl_hx is None or not self.has_hedge:
            return
        self.add("### The carry approximation, checked by regression\n")
        f = self.hedged_frame()
        assert self.pnl_h is not None
        rows = []
        for (m, side, b), g in f.groupby(["months", "side", "barrier"]):
            g = g.sort_values("entry")
            for name in ("A1", "B5_2", "B3", "B6"):
                y = self.col(g, self.pnl_h, name).to_numpy(float)
                c = self.col(g, self.carry, name).to_numpy(float)
                X = np.column_stack([np.ones(len(g)), c, g["touched"].to_numpy(float)])
                fit = ba.clustered_ols(y, X, g["entry"].to_numpy(str), BLOCK[int(m)])
                co, se = fit["coef"], fit["se"]
                rows.append(
                    {
                        "maturity": f"{m}m", "side": SIDE[side], "barrier": b,
                        "structure": NAMES[name],
                        "a": fmt(co[0], se[0]), "b": fmt(co[1], se[1], 1.0, 2),
                        "c (touched)": fmt(co[2], se[2]), "R²": f"{fit['r2']:.2f}",
                        "b in [0.7, 1.3]": "yes" if 0.7 <= co[1] <= 1.3 else "no", "n": fit["n"],
                    }
                )  # fmt: skip
                self.key[("reg", int(m), int(side), b, name)] = (co, se, fit["r2"])
                if name == "A1" and int(m) == 3 and b == "s1" and "lsv_minus_lv" in g:
                    gap, gap_se, _ = self.es(g, g["sgn"] * g["lsv_minus_lv"], 3)
                    raw = float(g["lsv_minus_lv"].mean())
                    b_ok = 0.7 <= co[1] <= 1.3
                    a_ok = abs(co[0] - gap) < 2 * se[0]
                    b_out = (co[1] + 2 * se[1] < 0.7) or (co[1] - 2 * se[1] > 1.3)
                    self.verdicts.append(
                        {
                            "prediction": "P13",
                            "statistic": f"{SIDE[side]} 3m 1 sd: hedged KO = a + b·carry + c·touched; b = {co[1]:.2f} ± {se[1]:.2f}, R² {fit['r2']:.2f}; a (% spot) against sgn·(LSV − LV) = {100 * gap:.3f} ± {100 * gap_se:.3f}",
                            "estimate": f"{100 * co[0]:.3f} ± {100 * se[0]:.3f}",
                            "verdict": "confirmed" if (b_ok and a_ok) else ("contradicted" if b_out else "inconclusive"),
                            "note": f"confirmed = b in [0.7, 1.3] and a within two standard errors of the gap; contradicted = b two standard errors outside the band. Unsigned LSV − LV: {100 * raw:.3f}",
                        }
                    )  # fmt: skip
        self.table(
            "carry_regression",
            pd.DataFrame(rows),
            "Per trade: hedged P&L = a + b × carry + c × (knocked on the daily rule) + e, for the "
            "daily knock-out (local-vol premium), the fly n = 2, the spread and the tight limit; a "
            "and c in % of spot, ± block-bootstrap standard errors over entry dates. The vol-carry "
            "reading holds where b is in [0.7, 1.3] and R² is material; a is the mean P&L of the "
            "trades not knocked that the carry does not explain. " + self.sample_note(f),
            index=False,
        )

    # -- A2.11h the justified share ex vol ---------------------------------------------------

    def share_exvol(self) -> None:
        f = getattr(self, "frame_hedged", None)
        if f is None or self.pnl_hx is None or "lsv_minus_lv" not in f:
            return self.missing("A2.11h. The justified share ex vol", "needs the attribution and the LSV entry marks")  # fmt: skip
        self.add("## A2.11h. The justified share, ex vol carry\n")
        f = f[f["lsv_minus_lv"].notna()].copy()
        hx = self.pnl_hx
        f["be_fly_x"] = -f["DF0"] * (self.col(f, hx, "B5_2") - self.col(f, hx, "A1"))
        f["be_b6_x"] = -f["DF0"] * (self.col(f, hx, "B6") - self.col(f, hx, "A1"))
        f["be_fly"] = -f["DF0"] * f["fly_ko"]
        if self.touches is not None and "Pi_d_ssr12" in f:
            f = f.merge(self.touches[["cell", "resid_pv"]], on="cell", how="left")
            f["resid0"] = f["resid_pv"].where(f["tau1"].notna(), 0.0)
            f["lt_num"] = f["resid0"] - f["Pi_d_lv"]
            f["lt_den"] = f["Pi_d_ssr12"] - f["Pi_d_lv"]
        # the condition labels were built on frame_hedged's index: carry them by cell
        lab_by_cell = {
            label: pd.Series(lab.to_numpy(), index=self.frame_hedged.loc[lab.index, "cell"])
            for label, lab in getattr(self, "cond_labels", {}).items()
        }

        def lam(g: pd.DataFrame, num: str, den: str, m: int, seed: int = 9) -> tuple[float, ...]:
            per = g.groupby("entry")[[num, den]].mean().dropna().sort_index()
            return ba.ratio_interval(per[num], per[den], BLOCK[m], floor=bt.RATIO_FLOOR, seed=seed)

        rows = []
        self.lam_x: dict[tuple[Any, ...], tuple[float, ...]] = {}
        for (m, side), gs in f.groupby(["months", "side"]):
            for scope in (*SD, "sd pooled"):
                g = gs if scope == "sd pooled" else gs[gs["barrier"] == scope]
                buckets: list[tuple[str, pd.DataFrame]] = [("all", g)]
                if scope in ("s1", "sd pooled"):
                    for label, lab in lab_by_cell.items():
                        keys = lab.reindex(g["cell"]).to_numpy()
                        buckets += [(f"{label}: {k}", g[keys == k]) for k in ("low", "mid", "high")]
                for bucket, gg in buckets:
                    lx = lam(gg, "be_fly_x", "lsv_minus_lv", int(m))
                    lb = lam(gg, "be_b6_x", "lsv_minus_lv", int(m))
                    l0 = lam(gg, "be_fly", "lsv_minus_lv", int(m))
                    lt = lam(gg, "lt_num", "lt_den", int(m), seed=3) if "lt_num" in gg else (np.nan,) * 5  # fmt: skip
                    self.lam_x[(int(m), int(side), scope, bucket)] = lx
                    if bucket == "all":
                        self.lam_x[(int(m), int(side), scope, "touch")] = lt
                        self.lam_x[(int(m), int(side), scope, "with vol")] = l0
                    rows.append(
                        {
                            "maturity": f"{m}m", "side": SIDE[side], "barrier": scope,
                            "bucket": bucket,
                            "P*x − LV vs fly": f"{100 * lx[3]:.3f}" if np.isfinite(lx[3]) else "–",
                            "λ*x vs fly": interval(lx),
                            "P*x − LV vs tight": f"{100 * lb[3]:.3f}" if np.isfinite(lb[3]) else "–",
                            "λ*x vs tight": interval(lb),
                            "mean LSV − LV": f"{100 * lx[4]:.3f}" if np.isfinite(lx[4]) else "–",
                            "λ* with vol (v2)": interval(l0),
                            "λ_touch": interval(lt),
                            "entries": gg["entry"].nunique(),
                        }
                    )  # fmt: skip
                if scope == "s1":
                    self.lam_x[(int(m), int(side), "s1", "late")] = lam(
                        g[g["entry"] >= HALVES[1][1]], "be_fly_x", "lsv_minus_lv", int(m)
                    )
                if scope == "s1" and int(m) == 3:
                    lx = self.lam_x[(3, int(side), "s1", "all")]
                    lt = self.lam_x[(3, int(side), "s1", "touch")]
                    inside = np.isfinite(lt[0]) and lx[1] <= lt[0] <= lx[2]
                    self.verdicts.append(
                        {
                            "prediction": "P12",
                            "statistic": f"{SIDE[side]} 3m 1 sd: λ*x against the fly (hedged, ex vol carry); λ_touch of the same cell {interval(lt)}",
                            "estimate": interval(lx),
                            "verdict": interval_verdict(lx[1], lx[2]),
                            "note": f"confirmed = the 95 % interval inside [0, 1]; contradicted = wholly outside. The interval {'contains' if inside else 'does not contain'} λ_touch",
                        }
                    )  # fmt: skip
        full = pd.DataFrame(rows)
        self.shown(
            "share_exvol",
            full,
            full[(full["bucket"] == "all") | (full["barrier"] == "s1")],
            "The justified share recomputed on the hedged P&L ex vol carry: P*x − lv_a1 = mean of "
            "DF0·(pnl_hx of the knock-out − pnl_hx of X), % of spot, X the fly (n = 2) or the tight "
            "limit; λ*x = that over the mean LSV-minus-local-vol premium, with a 95 % block-"
            "bootstrap interval; per barrier and pooled. Beside it the share of report v2 (with the "
            "vol carry in) and λ_touch of the same cell. The pooled rows divide by a gap that is "
            "near zero at 1.5–2 sd: the per-barrier rows are the ones to read. Terciles of the "
            "primary conditions are shown at 1 sd (the CSV has them pooled as well). "
            + self.sample_note(f),
        )

    # -- A2.11i conditions ex vol ------------------------------------------------------------

    def conditions_exvol(self) -> None:
        f = getattr(self, "frame_hedged", None)
        if f is None or self.pnl_hx is None:
            return self.missing("A2.11i. Conditions ex vol", "needs the attribution")
        self.add("## A2.11i. Conditions at entry, hedged outcomes ex vol carry\n")
        f = f.copy()
        hx = self.pnl_hx
        f["fly_ko_x"] = self.col(f, hx, "B5_2") - self.col(f, hx, "A1")
        f["b6_ko_x"] = self.col(f, hx, "B6") - self.col(f, hx, "A1")
        f["ko_x"] = self.col(f, hx, "A1")
        rows = []
        for label, lab in self.cond_labels.items():
            for scope, sel in (
                ("sd pooled", f["barrier"].isin(SD)),
                ("1 sd", f["barrier"] == "s1"),
            ):
                for (m, side), g in f[sel].groupby(["months", "side"]):
                    for period, lo, hi in PERIODS:
                        gp = g[(g["entry"] >= lo) & (g["entry"] <= hi)]
                        row: dict[str, Any] = {
                            "condition": label, "barriers": scope, "maturity": f"{m}m",
                            "side": SIDE[side], "period": period,
                        }  # fmt: skip
                        stats = {}
                        for key in ("low", "mid", "high"):
                            gg = gp[lab.reindex(gp.index) == key]
                            a, s, n = self.es(gg, gg["fly_ko_x"], m)
                            stats[key] = (a, s, n, gg)
                            row[f"fly − KO, {key}"] = fmt(a, s)
                        for key in ("low", "high"):
                            gg = stats[key][3]
                            row[f"tight − KO, {key}"] = fmt(*self.es(gg, gg["b6_ko_x"], m)[:2])
                            row[f"KO, {key}"] = fmt(*self.es(gg, gg["ko_x"], m)[:2])
                        row["entries low / high"] = f"{stats['low'][2]} / {stats['high'][2]}"
                        rows.append(row)
                        if scope == "1 sd" and period == "all":
                            self.key[("cond_x", label, int(m), int(side))] = {
                                k: stats[k][:2] for k in stats
                            }
        self.table(
            "conditions_exvol",
            pd.DataFrame(rows),
            "The hedged conditions table of section 11e on the P&L ex vol carry: fly minus knock-"
            "out, tight limit minus knock-out and the knock-out alone, % of spot ± block standard "
            "error, by tercile of each primary condition; barriers pooled and at one standard "
            "deviation; whole sample and each half. " + self.sample_note(f),
            index=False,
        )
        self.frame_x = f

    # -- A2.11j rules ex vol -----------------------------------------------------------------

    def rule_picks(self) -> dict[str, tuple[pd.Series, pd.Series | None]]:
        """rule → (pick per cell, the extra premium the knock-out pays under that rule)."""
        out: dict[str, tuple[pd.Series, pd.Series | None]] = {
            rule: (pick, None) for rule, pick in getattr(self, "choices", {}).items()
        }
        fin = self.fin.set_index("cell")
        for lam0, pick in getattr(self, "rule4_today", {}).items():
            extra = lam0 * fin["lsv_minus_lv"] / fin["DF0"]
            out[f"4 roll rule, λ0 = {lam0:g}"] = (pick, extra)
        return out

    def rules_exvol(self) -> None:
        if self.pnl_hx is None or not hasattr(self, "choices"):
            return self.missing("A2.11j. Rules ex vol", "needs the attribution and the rules")
        self.add("## A2.11j. The rules, unhedged, hedged and ex vol carry\n")
        f = self.hedged_frame().set_index("cell", drop=False)
        srcs = (("unhedged", self.pnl_u), ("hedged", self.pnl_h), ("ex vol", self.pnl_hx))
        rows = []
        for rule, (pick, extra) in self.rule_picks().items():
            for period, lo, hi in PERIODS:
                for m in (1, 3, 6, 12):
                    g = f[(f["entry"] >= lo) & (f["entry"] <= hi) & (f["months"] == m)]
                    p = pick.reindex(g.index)
                    g, p = g[p.notna()], p[p.notna()]
                    if g.empty:
                        continue
                    row: dict[str, Any] = {
                        "rule": rule, "period": period, "maturity": f"{m}m",
                        "share KO": f"{(p == 'A1').mean():.2f}",
                    }  # fmt: skip
                    for tag, src in srcs:
                        assert src is not None
                        ko = src["A1"].reindex(g.index)
                        if extra is not None:
                            ko = ko - extra.reindex(g.index)
                        fly = src["B5_2"].reindex(g.index)
                        chosen = pd.Series(np.nan, index=g.index)
                        for name in ("B5_2", "B6", "B4"):
                            chosen[p == name] = src[name].reindex(g.index)[p == name]
                        chosen[p == "A1"] = ko[p == "A1"]
                        a, s, n = self.es(g, chosen, m)
                        row[f"{tag} P&L"] = fmt(a, s)
                        row[f"{tag} vs KO"] = fmt(*self.es(g, chosen - ko, m)[:2])
                        row[f"{tag} vs fly"] = fmt(*self.es(g, chosen - fly, m)[:2])
                        row["entries"] = n
                        self.key[("rule", rule, period, m, tag)] = (
                            a, s, *self.es(g, chosen - ko, m)[:2], *self.es(g, chosen - fly, m)[:2]
                        )  # fmt: skip
                    rows.append(row)
        self.table(
            "rules_exvol",
            pd.DataFrame(rows),
            "Each rule and each always-strategy: mean P&L of its pick, and against always-knock-out "
            "and always-fly (n = 2) on the same trades, unhedged, delta-hedged and hedged ex vol "
            "carry; % of spot ± block standard error over entry dates; standard-deviation barriers "
            "and both sides pooled, trades with complete marks. Rule 4 buys the knock-out at share "
            "λ0 between the anchors (its knock-out, and the 'vs KO' column, pay λ0·(LSV − LV)/DF0 "
            "more); the other rules buy it at the local-vol premium. " + self.sample_note(f),
            index=False,
        )

    # -- A2.11k touch frequencies ------------------------------------------------------------

    def touch_frequency(self) -> None:
        self.add("## A2.11k. Touch frequencies against the models\n")
        self.add(
            "How often the market actually touched, against what the price assumed. Realised shares "
            "are over finished trades, one per entry date, with block standard errors; the model "
            "columns are the means over the same trades of the probabilities at entry (risk-neutral: "
            "the gap to the realised share includes the drift and the vol premium).\n"
        )
        f = self.fin[self.fin["barrier"].isin(SD)].copy()
        f["k1"], f["k2"] = f["tau1"].notna().astype(float), f["tau2"].notna().astype(float)
        f["bey"], f["reg"] = f["beyond"].astype(float), f["regret"].astype(float)
        f["dig"] = (f["P_B3"] - f["P_B6"]) / (f["w_rel"] * f["DF0"])
        has_lsv = "ssr12_p1" in f and f["ssr12_p1"].notna().any()
        cond = {label: labels(f, col) for label, col in TOUCH_CONDITIONS.items() if col in f}

        def row_of(g: pd.DataFrame, m: int, head: dict[str, Any]) -> dict[str, Any]:
            k1, k2 = self.es(g, g["k1"], m), self.es(g, g["k2"], m)
            be, rg = self.es(g, g["bey"], m), self.es(g, g["reg"], m)
            d1 = self.es(g, g["k1"] - g["lv_p1"], m)
            d2 = self.es(g, g["k2"] - g["lv_p2"], m)
            row = {
                **head,
                "knocked daily": fmt(k1[0], k1[1], 1.0), "LV p1": f"{g['lv_p1'].mean():.3f}",
                "LSV p1": f"{g['ssr12_p1'].mean():.3f}" if has_lsv else "–",
                "daily − LV": fmt(d1[0], d1[1], 1.0),
                "knocked cont.": fmt(k2[0], k2[1], 1.0), "LV p2": f"{g['lv_p2'].mean():.3f}",
                "LSV p2": f"{g['ssr12_p2'].mean():.3f}" if has_lsv else "–",
                "cont. − LV": fmt(d2[0], d2[1], 1.0),
                "beyond": fmt(be[0], be[1], 1.0), "Dig(B)": f"{g['dig'].mean():.3f}",
                "regret": fmt(rg[0], rg[1], 1.0),
                "LV p1 − Dig": f"{(g['lv_p1'] - g['dig']).mean():.3f}",
                "LV p1 − beyond": f"{g['lv_p1'].mean() - be[0]:.3f}",
                "entries": k1[2],
            }  # fmt: skip
            return row

        main, conds = [], []
        for (m, side, b), g in f.groupby(["months", "side", "barrier"]):
            head = {"maturity": f"{m}m", "side": SIDE[side], "barrier": b}
            for period, lo, hi in PERIODS:
                gp = g[(g["entry"] >= lo) & (g["entry"] <= hi)]
                main.append(row_of(gp, int(m), {**head, "group": period}))
                if period == "all":
                    self.key[("touch", int(m), int(side), b)] = (
                        self.es(gp, gp["k1"], m), float(gp["lv_p1"].mean()),
                        self.es(gp, gp["k2"], m), float(gp["lv_p2"].mean()),
                        float(gp["ssr12_p1"].mean()) if has_lsv else np.nan,
                    )  # fmt: skip
                    if int(m) == 3 and b == "s1":
                        for tag, kcol, pcol in (("daily", "k1", "p1"), ("continuous", "k2", "p2")):
                            a, s, _ = self.es(gp, gp[kcol] - gp[f"lv_{pcol}"], 3)
                            z = abs(a / s) if s > 0 else np.inf
                            note = "confirmed = within two standard errors of zero; contradicted = beyond"
                            if has_lsv:
                                h_ = gp[gp[f"ssr12_{pcol}"].notna()]
                                al, sl, _ = self.es(h_, h_[kcol] - h_[f"ssr12_{pcol}"], 3)
                                note += f". Against the LSV ({pcol}): {al:.3f} ± {sl:.3f}"
                            self.verdicts.append(
                                {
                                    "prediction": "P14",
                                    "statistic": f"{SIDE[side]} 3m 1 sd: realised knock share minus lv_{pcol} ({tag} rule), probability",
                                    "estimate": f"{a:.3f} ± {s:.3f}",
                                    "verdict": "confirmed" if z < 2 else "contradicted",
                                    "note": note,
                                }
                            )
            for label, lab in cond.items():
                for key in ("low", "mid", "high"):
                    gg = g[lab.reindex(g.index) == key]
                    if gg["entry"].nunique() >= 20:
                        conds.append(row_of(gg, int(m), {**head, "group": f"{label}: {key}"}))
        full = pd.concat([pd.DataFrame(main), pd.DataFrame(conds)], ignore_index=True)
        cap = (
            "Realised share of trades knocked on the daily rule and on the continuous rule against "
            "the models' probabilities at entry (local vol; LSV at SSR 1.2); realised share finishing "
            "beyond the barrier against the surface digital Dig(B) (undiscounted, the one inside "
            "the tight limit); realised share of regret paths (knocked, finishing in the money and "
            "inside the barrier) against the model's knock probability minus the digital, and "
            "minus the realised share beyond. Probabilities, ± block standard error. "
        )
        self.shown("touch_frequency", full, pd.DataFrame(main), cap + "Whole sample and each half; "
                   "the CSV also holds the terciles of every maturity and barrier. " + self.sample_note(f))  # fmt: skip
        c = pd.DataFrame(conds)
        if len(c):
            c = c[c["maturity"].isin(["1m", "3m"]) & c["barrier"].isin(["s0.5", "s1", "s1.5"])]
            self.table(
                "touch_frequency_conditions",
                c.drop(
                    columns=["LV p2", "LSV p2", "cont. − LV", "knocked cont.", "LV p1 − beyond"]
                ),
                "The same by tercile (against its own history up to the entry, two years minimum) of "
                "the past 3-month return, the vol premium (`vrp_ratio`), implied minus realised vol "
                "and the level of vol; 1 and 3 months, barriers at 0.5, 1 and 1.5 sd.",
                index=False,
            )

    # -- A2.11l turnover and costs -----------------------------------------------------------

    def hedge_cost(self) -> None:
        if self.turn is None or not self.has_hedge:
            return self.missing("A2.11l. Hedge turnover and the cost of hedging", "turnover.parquet does not exist (the turnover pass has not finished)")  # fmt: skip
        self.add("## A2.11l. Hedge turnover and the cost of hedging\n")
        f = self.hedged_frame()
        tn = self.turn.pivot_table(index="cell", columns="position", values="turnover_notional", aggfunc="first")  # fmt: skip
        tr = self.turn.pivot_table(index="cell", columns="position", values="turnover", aggfunc="first")  # fmt: skip
        f = f[f["cell"].isin(tn.index)]
        h = self.pnl_h
        assert h is not None
        rows = []
        for (m, side, b), g in f.groupby(["months", "side", "barrier"]):
            for name in ATTRIB:
                t = self.col(g, tn, name)
                hp = self.col(g, h, name)
                row = {
                    "maturity": f"{m}m", "side": SIDE[side], "barrier": b,
                    "structure": NAMES[name],
                    "turnover": f"{self.col(g, tr, name).mean():.2f}",
                    "notional traded": f"{t.mean():.2f}",
                    "hedged": fmt(*self.es(g, hp, m)[:2]),
                }  # fmt: skip
                for tag, rate in ba.COST_RATES.items():
                    row[f"net of {tag}"] = fmt(*self.es(g, hp - rate * t, m)[:2])
                row["n"] = int(t.notna().sum())
                rows.append(row)
                self.key[("turn", int(m), int(side), b, name)] = float(t.mean())
            t = self.col(g, tn, "B5_2") - self.col(g, tn, "A1")
            hp = self.col(g, h, "B5_2") - self.col(g, h, "A1")
            row = {
                "maturity": f"{m}m", "side": SIDE[side], "barrier": b, "structure": "fly n=2 − KO",
                "turnover": f"{(self.col(g, tr, 'B5_2') - self.col(g, tr, 'A1')).mean():.2f}",
                "notional traded": f"{t.mean():.2f}", "hedged": fmt(*self.es(g, hp, m)[:2]),
            }  # fmt: skip
            for tag, rate in ba.COST_RATES.items():
                st = self.es(g, hp - rate * t, m)
                row[f"net of {tag}"] = fmt(*st[:2])
                self.key[("cost_pair", int(m), int(side), b, tag)] = st[:2]
            self.key[("cost_pair", int(m), int(side), b, "gross")] = self.es(g, hp, m)[:2]
            row["n"] = int(t.notna().sum())
            rows.append(row)
        self.table(
            "hedge_cost",
            pd.DataFrame(rows),
            "Turnover of the daily delta hedge per trade: |δ_0| + Σ|δ_t − δ_{t−1}| + |δ_last|, in "
            "units of delta (`turnover`) and weighted by the forward over the entry spot (`notional "
            "traded`, in multiples of the notional); the hedged P&L and the hedged P&L net of a "
            "cost of 0.5 bp and of 2 bp of the notional traded, % of spot ± block standard error. "
            "The pair's cost is the fly's minus the knock-out's (the two hedges are run "
            "separately). Local-vol sticky-strike deltas; the knock-out is hedged to the knock. "
            + self.sample_note(f),
            index=False,
        )

    # -- A2.11m skew sensitivity -------------------------------------------------------------

    def skew_vega(self) -> None:
        sk = self.read("skew_vega.parquet")
        if sk is None:
            return self.missing("A2.11m. Skew sensitivity at entry", "the skew bump (optional, monthly subset) has not run")  # fmt: skip
        self.add("## A2.11m. Skew sensitivity at entry (monthly subset, local vol)\n")
        c = self.c[["entry", "months", "side", "barrier"]].merge(sk, on=["entry", "months", "side", "barrier"])  # fmt: skip
        c = c[c["barrier"].isin(SD)]
        rows = []
        for (m, side, b), g in c.groupby(["months", "side", "barrier"]):
            row = {"maturity": f"{m}m", "side": SIDE[side], "barrier": b, "entries": len(g)}
            for name in ("A1", "B5_2", "B5_6", "B3", "B4", "B6"):
                row[NAMES[name]] = f"{100 * g[f'skewvega_{name}'].mean():.3f}"
            row["fly n=2 − KO"] = f"{100 * (g['skewvega_B5_2'] - g['skewvega_A1']).mean():.3f}"
            rows.append(row)
        self.table(
            "skew_vega",
            pd.DataFrame(rows),
            "Change of each entry premium for a tilt of the entry smile of one vol point at the "
            "barrier strike (plus on the call side, minus on the put side; minus/plus at the mirror "
            "strike; at-the-money unchanged), % of spot: the knock-out repriced under local vol with "
            "Dupire rebuilt and common random numbers, the vanilla structures off the tilted "
            "surface. The knock-out and the fly are both short the vol at the barrier; the sign and "
            "size of their difference say whether a view on the skew separates them. " + self.sample_note(c),
            index=False,
        )  # fmt: skip

    # -- 5.4 the LSV-hedged comparison by period ---------------------------------------------

    def lsv_periods(self) -> None:
        if self.pos_lsv.empty or "pnl_h" not in self.pos_lsv:
            return
        lsv = self.pos_lsv.dropna(subset=["pnl_h"]).set_index(["cell", "position"])
        lv = self.pos[self.pos["position"].isin(["A1", "A2"])].set_index(["cell", "position"])
        both = lsv.join(lv, lsuffix="_lsv", rsuffix="_lv", how="inner").reset_index()
        both = both.merge(self.fin[["cell", "entry", "months", "side", "barrier"]], on="cell")
        both = both[both["barrier"].isin(SD) & (both["position"] == "A1")]
        if both.empty:
            return
        self.add("### The LSV-hedged comparison by period (addendum 2 §5.4)\n")
        n_all = self.fin["entry"].nunique()
        self.add(
            f"The LSV daily study covers {both['entry'].nunique()} of the {n_all} entry dates of "
            f"this sample with finished trades, from {both['entry'].min()} to {both['entry'].max()} "
            f"({both['cell'].nunique()} daily knock-outs marked and hedged under both models).\n"
        )
        self.key["lsv_cover"] = (both["entry"].nunique(), n_all, both["entry"].min(), both["entry"].max())  # fmt: skip
        rows = []
        for (m, side), gs in both.groupby(["months", "side"]):
            for period, lo, hi in PERIODS:
                g = gs[(gs["entry"] >= lo) & (gs["entry"] <= hi)]
                if g["entry"].nunique() < 10:
                    continue
                st = {t: self.es(g, g[f"pnl_h_{t}"], m) for t in ("lv", "lsv")}
                d = self.es(g, g["pnl_h_lsv"] - g["pnl_h_lv"], m)
                rows.append(
                    {
                        "maturity": f"{m}m", "side": SIDE[side], "period": period,
                        "entries": st["lv"][2],
                        "premium LV": f"{100 * g['premium_lv'].mean():.3f}",
                        "premium LSV": f"{100 * g['premium_lsv'].mean():.3f}",
                        "hedged LV": fmt(*st["lv"][:2]), "hedged LSV": fmt(*st["lsv"][:2]),
                        "hedged LSV − LV": fmt(*d[:2]),
                        "minus premium difference": f"{-100 * (g['premium_lsv'] - g['premium_lv']).mean():.3f}",
                        "hedged sd LV": f"{100 * g['pnl_h_lv'].std():.3f}",
                        "hedged sd LSV": f"{100 * g['pnl_h_lsv'].std():.3f}",
                    }
                )  # fmt: skip
                self.key[("lsv", int(m), int(side), period)] = (d[0], d[1], float(g["pnl_h_lv"].std()), float(g["pnl_h_lsv"].std()), float((g["premium_lsv"] - g["premium_lv"]).mean()))  # fmt: skip
        self.table(
            "lsv_hedged_periods",
            pd.DataFrame(rows),
            "The daily knock-out bought, marked and delta-hedged under each model (its own premium, "
            "marks and sticky-strike delta), standard-deviation barriers pooled: mean hedged P&L "
            "under each, their difference (± block standard error over entry dates) beside minus the "
            "difference of the two premiums, and the dispersion of the hedged P&L under each; whole "
            "coverage and each half. The model moves the premium, not the hedge, where the "
            "difference of the P&Ls is minus the difference of the premiums and the dispersions "
            "are equal.",
            index=False,
        )

    # -- 5.2 the cross-tab -------------------------------------------------------------------

    def crosstab(self) -> None:
        f = getattr(self, "frame_x", None)
        if f is None or "vrp_ratio" not in f:
            return
        self.add("### SSR against the vol premium (addendum 2 §5.2, beside P7)\n")
        ssr = self.cond_labels.get("SSR up (call) / down (put)")
        vrp = self.cond_labels.get("vol premium")
        if ssr is None or vrp is None:
            return
        g = f[(f["months"] == 3) & (f["barrier"] == "s1")]
        rows = []
        for side in (1, -1):
            gs = g[g["side"] == side]
            for a in ("low", "mid", "high"):
                for b in ("low", "mid", "high"):
                    gg = gs[(ssr.reindex(gs.index) == a) & (vrp.reindex(gs.index) == b)]
                    rows.append(
                        {
                            "side": SIDE[side],
                            "SSR up / down tercile": a, "vol premium tercile": b,
                            "entries": gg["entry"].nunique(),
                            "fly − KO hedged": fmt(*self.es(gg, gg["fly_ko"], 3)[:2]),
                            "fly − KO ex vol": fmt(*self.es(gg, gg["fly_ko_x"], 3)[:2]),
                        }
                    )  # fmt: skip
        self.table(
            "ssr_vrp_crosstab",
            pd.DataFrame(rows),
            "3 months, barrier at 1 sd: terciles of the directional SSR (`ssr_up_3m_6m` on the call "
            "side, `ssr_dn_3m_6m` on the put side) against terciles of `vrp_ratio`; number of "
            "entries and the mean hedged P&L of fly minus knock-out, with and without the vol "
            "carry, % of spot ± block standard error. " + self.sample_note(g),
            index=False,
        )

    # -- 5.3 strategy series of the rules ----------------------------------------------------

    def rule_strategies(self) -> None:
        files = sorted((bh.RESULTS / "books").glob("*.parquet"))
        picks = {
            k: v for k, v in self.rule_picks().items()
            if k.startswith(("1 ", "always KO", "always fly")) or ("(fly)" in k and k.startswith("2 ")) or k.endswith("λ0 = 0.5")
        }  # fmt: skip
        if not files or not picks:
            return
        self.add("### Strategy series of the rules (addendum 2 §5.3)\n")
        keep_b = ["s0.5", "s1", "s1.5"]
        parts = []
        for p in files:
            b = pd.read_parquet(p)
            b = b[
                b["months"].isin([1, 3])
                & b["barrier"].isin(keep_b)
                & b["position"].isin(["A1", "B5_2"])
            ]
            if len(b):
                b = b.assign(
                    cell=p.stem
                    + "|"
                    + b["months"].astype(str)
                    + "|"
                    + b["side"].astype(str)
                    + "|"
                    + b["barrier"]
                )
                parts.append(b)
        book = pd.concat(parts, ignore_index=True)
        book = book[book["cell"].isin(set(self.fin["cell"]))]
        ohlc = bh.load_ohlc()
        spx = np.log(ohlc["close"]).diff()
        vix = (
            pd.read_csv(bh.HISTORY / "VIX.csv", dtype={"date": str})
            .set_index("date")["close"]
            .diff()
        )
        days = sorted(book["date"].unique())
        first_day = book.groupby("cell")["date"].transform("min")
        fin = self.fin.set_index("cell")
        rows = []
        for rule, (pick, extra) in picks.items():
            chosen = book[
                book["position"].to_numpy() == pick.reindex(book["cell"]).to_numpy()
            ].copy()
            if (
                extra is not None
            ):  # the extra premium of the knock-out, paid on the trade's first day
                pay = (chosen["position"] == "A1") & (
                    chosen["date"] == first_day.reindex(chosen.index)
                )
                e = extra.reindex(chosen["cell"]).to_numpy()
                for c_ in ("du", "dh"):
                    chosen[c_] = chosen[c_] - np.where(pay, e, 0.0)
            trade_u = pd.Series(np.nan, index=pick.dropna().index)
            for name in ("A1", "B5_2"):
                sel = pick.reindex(trade_u.index) == name
                trade_u[sel] = self.pnl_u[name].reindex(trade_u.index)[sel]
            if extra is not None:
                trade_u = trade_u - extra.reindex(trade_u.index).where(
                    pick.reindex(trade_u.index) == "A1", 0.0
                )
            for (m, side), g in chosen.groupby(["months", "side"]):
                daily = g.groupby("date")[["du", "dh"]].sum().reindex(days).fillna(0.0) / len(
                    keep_b
                )
                cells_ = fin.index[
                    (fin["months"] == m) & (fin["side"] == side) & fin["barrier"].isin(keep_b)
                ]
                tu = trade_u.reindex(cells_).dropna()
                tent = fin.loc[tu.index, "entry"]
                for period, lo, hi in PERIODS:
                    s_ = daily[(daily.index >= lo) & (daily.index <= hi + "z")]
                    row: dict[str, Any] = {
                        "rule": rule,
                        "maturity": f"{m}m",
                        "side": SIDE[side],
                        "period": period,
                    }
                    for tag, c_ in (("unhedged", "du"), ("hedged", "dh")):
                        x = s_[c_]
                        vol = x.std() * np.sqrt(252.0)
                        cum = x.cumsum()
                        month = x.groupby(x.index.str[:7]).sum()
                        row[f"{tag} mean /y"] = f"{100 * 252 * x.mean():.2f}"
                        row[f"{tag} vol /y"] = f"{100 * vol:.2f}"
                        row[f"{tag} Sharpe"] = f"{252 * x.mean() / vol:.2f}" if vol > 0 else "–"
                        row[f"{tag} max DD"] = f"{100 * (cum - cum.cummax()).min():.2f}"
                        row[f"{tag} worst month"] = f"{100 * month.min():.2f}"
                        row[f"{tag} skew"] = f"{month.skew():.2f}"
                    reg = pd.DataFrame(
                        {"y": s_["dh"], "spx": spx.reindex(s_.index), "vix": vix.reindex(s_.index)}
                    ).dropna()
                    if len(reg) > 100:
                        beta = np.linalg.lstsq(
                            np.column_stack([np.ones(len(reg)), reg["spx"], reg["vix"]]),
                            reg["y"],
                            rcond=None,
                        )[0]
                        row["beta S&P"] = f"{beta[1]:.3f}"
                        row["beta VIX /pt"] = f"{100 * beta[2]:.4f}"
                    t = tu[(tent >= lo) & (tent <= hi)]
                    if len(t):
                        row["5 % worst"] = f"{100 * t.quantile(0.05):.2f}"
                        row["1 % worst"] = f"{100 * t.quantile(0.01):.2f}"
                    rows.append(row)
        frame = pd.DataFrame(rows).fillna("–")
        self.table(
            "strategy_rules",
            frame,
            "The rolling book that follows each rule at every weekly entry of this sample and holds "
            "to expiry (one third of a unit on each of the barriers at 0.5, 1 and 1.5 sd): daily "
            "P&L unhedged and delta-hedged, % of spot — annualised mean and volatility, Sharpe "
            "ratio, maximum drawdown, worst calendar month, skewness of monthly P&L; betas of the "
            "hedged book to S&P log returns and to VIX changes; the 5 % and 1 % worst unhedged "
            "trades. Rule 4 at λ0 = 0.5 pays the knock-out's extra premium on the trade's first day. "
            "Beside the always-strategies on the same trades.",
            index=False,
        )

    # -- 5.1 today's picks -------------------------------------------------------------------

    def today3(self) -> None:
        a = self.cells_all[self.cells_all["built"]]
        if "n_spot" not in a or "dn_lsv" not in a or self.touches is None:
            return
        last = a["entry"].max()
        t = a[(a["entry"] == last) & a["barrier"].isin(SD)].sort_values(
            ["months", "side", "barrier"]
        )
        self.add("### Today's picks (addendum 2 §5.1)\n")
        tt = self.touches[["side", "tau", "skew_ratio"]].dropna()
        rho = {}
        for side in (1, -1):
            past = tt[(tt["side"] == side) & (tt["tau"] < last)]
            rho[side] = float(past["skew_ratio"].mean()) if len(past) >= 30 else 1.0
        rows = []
        self.today_pick: dict[tuple[int, int, str], dict[str, str]] = {}
        for r in t.itertuples():
            n_exp = r.n_spot * rho[int(r.side)]
            row: dict[str, Any] = {
                "maturity": f"{r.months}m", "side": SIDE[r.side], "barrier": r.barrier,
                "n_exp": f"{n_exp:.3f}", "ρ": f"{rho[int(r.side)]:.3f}",
            }  # fmt: skip
            picks: dict[str, str] = {}
            for lam0 in (0.0, 0.5, 1.0):
                ch = r.n_eff_lv + lam0 * r.dn_lsv
                if np.isfinite(ch) and np.isfinite(n_exp):
                    ko = ch < n_exp if r.side > 0 else ch > n_exp
                    picks[f"Rule 4, λ0={lam0:g}"] = "KO daily" if ko else "fly n=2"
                row[f"n_charged λ0={lam0:g}"] = f"{ch:.3f}" if np.isfinite(ch) else "–"
                row[f"Rule 4, λ0={lam0:g}"] = picks.get(f"Rule 4, λ0={lam0:g}", "–")
            if np.isfinite(getattr(r, "iv_minus_rv", np.nan)):
                mid = "KO daily" if r.iv_minus_rv > 0 else "fly n=2"
                picks["Rule 1"] = (
                    "fly n=2" if r.dist_sd < 1.0 else ("KO daily" if r.dist_sd > 1.5 else mid)
                )
            if np.isfinite(getattr(r, "h_A1", np.nan)) and np.isfinite(
                getattr(r, "h_B5_2", np.nan)
            ):
                hist = r.DF0 * (r.h_B5_2 - r.h_A1)
                picks["Rule 2, LV premium"] = (
                    "KO daily" if (r.P_B5_2 - r.lv_a1) > hist else "fly n=2"
                )
                if np.isfinite(getattr(r, "ssr12_a1", np.nan)):
                    picks["Rule 2, LSV premium"] = (
                        "KO daily" if (r.P_B5_2 - r.ssr12_a1) > hist else "fly n=2"
                    )
            for k in ("Rule 1", "Rule 2, LV premium", "Rule 2, LSV premium"):
                row[k] = picks.get(k, "–")
            self.today_pick[(int(r.months), int(r.side), r.barrier)] = picks
            rows.append(row)
        self.table(
            "today3",
            pd.DataFrame(rows),
            f"The picks on the latest entry date ({last}), from entry data only. Rule 4: n_charged = "
            "n_eff[LV] + λ0·dn_lsv against n_exp = n_spot·ρ (ρ the mean realised skew ratio at the "
            "touches before that date, same side); call side knock-out if n_charged < n_exp, put "
            "side the reverse. Rule 1: fly inside 1 sd, knock-out beyond 1.5 sd, between them the "
            "knock-out if implied exceeds realised vol. Rule 2: knock-out if the fly-minus-knock-out "
            "premium exceeds its historical outcome gap, with the knock-out at the local-vol or at "
            "the LSV premium.",
            index=False,
        )

    # -- A2.12 the framework table -----------------------------------------------------------

    def framework(self) -> None:
        f = getattr(self, "frame_hedged", None)
        if f is None or self.pnl_hx is None:
            return self.missing(
                "A2.12. The framework table", "needs the hedged conditions and the attribution"
            )
        self.add("## A2.12. The framework table\n")
        f = f.copy()
        f["touched"] = f["tau1"].notna().astype(float)
        srcs = (("unhedged", self.pnl_u), ("hedged", self.pnl_h), ("ex vol", self.pnl_hx))
        for tag, src in srcs:
            assert src is not None
            for name in CANDIDATES:
                f[f"{tag}|{name}"] = self.col(f, src, name)
        a = self.cells_all[self.cells_all["built"]]
        last = a["entry"].max()
        conds: list[tuple[str, str, dict[int, str], pd.Series]] = []
        for label, cols in r2.PRIMARY2.items():
            if label in self.cond_labels:
                conds.append((label, "primary", cols, self.cond_labels[label]))
        for col, label in base.EXPLORATORY.items():
            if col in f and f[col].notna().sum() > 500 and col not in {c[2][1] for c in conds}:
                conds.append((label, "exploratory", {1: col, -1: col}, labels(f, col)))

        def best(g: pd.DataFrame, tag: str, m: int) -> str:
            per = (
                g.groupby("entry")[[f"{tag}|{n}" for n in CANDIDATES]].mean().dropna().sort_index()
            )
            if len(per) < 20:
                return "–"
            means = per.mean().sort_values(ascending=False)
            top, second = means.index[0], means.index[1]
            _, se = bh.block_bootstrap((per[top] - per[second]).to_numpy(float), BLOCK[m], seed=7)
            return f"{NAMES[top.split('|')[1]]} (+{100 * (means[top] - means[second]):.3f} ± {100 * se:.3f})"

        rows = []
        for label, kind, cols, lab in conds:
            for scope, sel in (
                ("1 sd", f["barrier"] == "s1"),
                ("sd pooled", f["barrier"].isin(SD)),
            ):
                for (m, side), g in f[sel].groupby(["months", "side"]):
                    col = cols[int(side)]
                    hist = a[
                        (a["months"] == m) & (a["side"] == side) & (a["barrier"] == "s1")
                    ].sort_values("entry")
                    today = "–"
                    if (
                        col in hist
                        and len(hist)
                        and hist["entry"].iloc[-1] == last
                        and np.isfinite(hist[col].iloc[-1])
                    ):
                        pct = float(
                            (hist[col].iloc[:-1].dropna() <= float(hist[col].iloc[-1])).mean()
                        )
                        today = "low" if pct <= 1 / 3 else ("high" if pct > 2 / 3 else "mid")
                    for key in ("low", "mid", "high"):
                        gg = g[lab.reindex(g.index) == key]
                        late = gg[gg["entry"] >= HALVES[1][1]]
                        row = {
                            "side": SIDE[side], "maturity": f"{m}m", "barriers": scope,
                            "condition": label, "kind": kind, "tercile": key,
                            "entries": gg["entry"].nunique(),
                            "touch freq.": f"{self.es(gg, gg['touched'], m)[0]:.2f}" if len(gg) else "–",
                        }  # fmt: skip
                        for tag, _ in srcs:
                            row[f"best {tag}"] = best(gg, tag, int(m))
                        for tag, _ in srcs:
                            row[f"2017–26 {tag}"] = best(late, tag, int(m))
                        row["today"] = "← today" if key == today else ""
                        rows.append(row)
        full = pd.DataFrame(rows)
        self.framework_rows = full
        short = {"KO daily": "KO", "fly n=2": "fly2", "fly n=6": "fly6", "ratio 1x2": "ratio", "tight limit": "tight"}  # fmt: skip
        shown = full[(full["kind"] == "primary") & (full["barriers"] == "1 sd")].drop(
            columns=["kind", "barriers"]
        )
        shown = shown.replace(short, regex=True).replace(
            {"SSR up \\(call\\) / down \\(put\\)": "SSR up / down", "← today": "today"}, regex=True
        )
        self.shown(
            "framework",
            full,
            shown,
            "A read-out of the conditional tables, nothing fitted. Per side, maturity, condition and "
            "tercile (against the condition's own history up to the entry): the structure with the "
            "highest mean P&L among the daily knock-out (local-vol premium), the fly n = 2, the fly "
            "n = 6, the ratio and the tight limit — unhedged, delta-hedged, and hedged ex vol carry — "
            "with its margin over the runner-up (% of spot ± block standard error of that "
            "difference); the same on 2017–2026 alone; the realised daily touch frequency of the "
            f"tercile; and the tercile the condition is in on {last} (the row's structures are then "
            "the read-out for today). Shown: primary conditions, barrier at 1 sd, with KO = daily knock-out, fly2 / fly6 = fly n = 2 / 6, tight = tight limit; the CSV also holds "
            "the barriers pooled and the exploratory conditions. " + self.sample_note(f),
        )

    # -- the findings and the PM page --------------------------------------------------------

    def vd(self, pred: str, start: str) -> str:
        """``estimate (verdict)`` of the first verdict row of ``pred`` whose statistic starts
        with ``start``."""
        for v in self.verdicts:
            if v["prediction"] == pred and v["statistic"].startswith(start):
                return f"{v['estimate']} ({v['verdict']})"
        return "not evaluated"

    def att(self, m: int, side: int, b: str, name: str) -> str:
        """``hedged h, carry c, ex vol x ± se`` of one attribution row."""
        k = self.key.get(("attrib", m, side, b, name))
        if k is None:
            return "not available"
        return f"hedged {100 * k[0]:.3f}, carry {100 * k[1]:.3f}, ex vol {100 * k[2]:.3f} ± {100 * k[5]:.3f}"

    def rule_line(self, rule: str, period: str, m: int) -> str:
        out = []
        for tag in ("unhedged", "hedged", "ex vol"):
            k = self.key.get(("rule", rule, period, m, tag))
            if k is not None:
                out.append(
                    f"{tag} {100 * k[2]:+.3f} ± {100 * k[3]:.3f} vs KO, {100 * k[4]:+.3f} ± {100 * k[5]:.3f} vs fly"
                )
        return "; ".join(out) if out else "not available"

    def findings(self) -> str:
        K = self.key
        lx = getattr(self, "lam_x", {})
        dsig = self.dsig.reindex(
            self.hedged_frame().query("months == 3 and barrier == 's1' and side == 1")["cell"]
        )
        reg = K.get(("reg", 3, 1, "s1", "A1"))
        reg_s = K.get(("reg", 3, 1, "s1", "B3"))
        cx = K.get(("cond_x", "SSR up (call) / down (put)", 3, 1), {})
        tc, tp = K.get(("touch", 3, 1, "s1")), K.get(("touch", 3, -1, "s1"))
        cover = K.get("lsv_cover")
        l3 = K.get(("lsv", 3, 1, "all"))
        cost = K.get(("cost_pair", 3, 1, "s1", "2bp"))
        gross = K.get(("cost_pair", 3, 1, "s1", "gross"))
        lines = [
            "## Findings (report v3: with the attribution of addendum 2)\n",
            f"{'Strict sample' if self.sample == 'strict' else 'All built entries'}; % of the entry spot unless said otherwise; "
            "± is a block standard error over entry dates, [ ] a 95 % block-bootstrap interval. The verdict "
            "words are those of the verdict table (section 11j) and of no other rule.\n",
            f"1. **Two anchors.** The LSV-minus-local-vol premium of the knock-out, signed by side, at 3 months: call {self.vd('P1', 'call')}, put {self.vd('P1', 'put')}; it grows with maturity (section 11a).",
            f"2. **The gap is the price of forward skew at the barrier.** Slope of the gap on the roll value: call {self.vd('P2', 'call')}, put {self.vd('P2', 'put')}.",
            f"3. **The touch test sides with the stationary smile.** |n_real − n_stat| − |n_real − n_lv| at the knock day, 3 months: call {self.vd('P3', 'call 3m')}, put {self.vd('P3', 'put 3m')}.",
            f"4. **At-the-money vol at the touch against local vol's**, vol points, 3 months: call {self.vd('P4', 'call 3m')}, put {self.vd('P4', 'put 3m')}.",
            f"5. **Leaving the static hedge at an up touch costs** {self.vd('P5', 'call 3m')} per knocked trade at 3 months, from the smile.",
            "6. **What the hedged edge of the knock-out over the fly is made of (rewritten; P7).** "
            f"3 months, call side, barrier at 1 sd: the realised vol premium of the life averaged {dsig.mean():.2f} vol points; the knock-out is short vega, the fly is not. "
            f"Knock-out: {self.att(3, 1, 's1', 'A1')}. Fly n = 2: {self.att(3, 1, 's1', 'B5_2')}. Fly − knock-out: {self.att(3, 1, 's1', 'B5_2-A1')}. "
            "So part of the hedged gap is the vol premium a short-vega structure collected, and part is not. "
            + (
                f"Ex vol carry, fly − knock-out by tercile of the upside SSR at 1 sd: low {fmt(*cx['low'])}, high {fmt(*cx['high'])}. "
                if cx
                else ""
            )
            + f"P7's statistic is unchanged: {self.vd('P7', 'call 3m')} at 3 months, {self.vd('P7', 'call 1m')} at 1 month.",
            "7. **The justified share (rewritten; P8, P12).** With the vol carry in, the share at 3 months and 1 sd was "
            f"{interval(lx.get((3, 1, 's1', 'with vol'), (np.nan,) * 3))} on calls and {interval(lx.get((3, -1, 's1', 'with vol'), (np.nan,) * 3))} on puts. "
            f"Ex vol carry it is λ*x = {interval(lx.get((3, 1, 's1', 'all'), (np.nan,) * 3))} on calls and {interval(lx.get((3, -1, 's1', 'all'), (np.nan,) * 3))} on puts, "
            f"beside the touch-day share λ_touch = {interval(lx.get((3, 1, 's1', 'touch'), (np.nan,) * 3))} and {interval(lx.get((3, -1, 's1', 'touch'), (np.nan,) * 3))}. "
            f"P12: call {self.vd('P12', 'call')}, put {self.vd('P12', 'put')}. The pooled shares divide by a gap that is near zero at 1.5–2 sd; the per-barrier rows of A2.11h are the ones to read.",
            "8. **The model moves the premium, not the hedge.** "
            + (
                f"The LSV daily study covers {cover[0]} of {cover[1]} entry dates ({cover[2]} to {cover[3]}). "
                if cover
                else ""
            )
            + (
                f"3 months, call side, barriers pooled: hedged P&L under the LSV minus under local vol {fmt(l3[0], l3[1])} for a premium difference of {100 * l3[4]:.3f}; dispersion {100 * l3[3]:.3f} against {100 * l3[2]:.3f}."
                if l3
                else ""
            ),
            "9. **The rules (rewritten).** Against always-knock-out and always-fly on the same trades, 3 months, barriers and sides pooled. "
            f"Rule 4 at λ0 = 0.5: {self.rule_line('4 roll rule, λ0 = 0.5', 'all', 3)}; on 2017–2026: {self.rule_line('4 roll rule, λ0 = 0.5', '2017–2026', 3)}. "
            f"Always-knock-out: {self.rule_line('always KO', 'all', 3)}. "
            "The margin of always-knock-out over always-fly is what the vol carry and the rest add up to; the ex-vol columns are the part that is not a vega choice.",
            f"10. **Fly or ratio.** Unhedged fly − ratio at 1.5 sd, call side, 3 months: {self.vd('P10', 'call 3m s1.5')}; beyond one standard deviation on the call side the far call is worth almost nothing and the two are the same trade (section 11g).",
            "",
            "**Added by addendum 2.**\n",
            (
                f"- *The carry approximation per trade (P13).* Hedged knock-out = a + b·carry + c·touched at 3 months, 1 sd, call side: b = {reg[0][1]:.2f} ± {reg[1][1]:.2f}, R² {reg[2]:.2f}; for the spread b = {reg_s[0][1]:.2f} ± {reg_s[1][1]:.2f}, R² {reg_s[2]:.2f}. P13: call {self.vd('P13', 'call')}, put {self.vd('P13', 'put')}. The mean carry above is a first-order estimate; where b is far from 1 or R² small, it is to be read as such."
                if reg and reg_s
                else "- The regression check is not available."
            ),
            (
                f"- *Touch frequencies (P14).* 3 months, 1 sd, daily rule: realised knock share {tc[0][0]:.3f} ± {tc[0][1]:.3f} against local vol's {tc[1]:.3f} on calls, {tp[0][0]:.3f} ± {tp[0][1]:.3f} against {tp[1]:.3f} on puts. P14 (daily): call {self.vd('P14', 'call 3m 1 sd: realised knock share minus lv_p1')}, put {self.vd('P14', 'put 3m 1 sd: realised knock share minus lv_p1')}."
                if tc and tp
                else "- The touch frequencies are not available."
            ),
            (
                f"- *Hedging costs.* 3 months, 1 sd, call side: notional traded per trade {K.get(('turn', 3, 1, 's1', 'A1'), np.nan):.2f} for the knock-out against {K.get(('turn', 3, 1, 's1', 'B5_2'), np.nan):.2f} for the fly; fly − knock-out hedged {fmt(*gross)} gross, {fmt(*cost)} net of 2 bp of the notional traded."
                if cost and gross
                else "- Turnover and costs are not available in this render."
            ),
            "",
        ]
        return "\n".join(lines)

    def pm_page(self) -> None:
        """``pm_framework.md``: one page, the five pieces, the rules and the costs."""
        K, lx = self.key, getattr(self, "lam_x", {})
        nan3 = (np.nan,) * 3
        f = self.hedged_frame()
        f31 = f[(f["months"] == 3) & (f["barrier"] == "s1")]
        late = HALVES[1][1]
        a = self.cells_all[self.cells_all["built"]]
        last = a["entry"].max()
        now = a[(a["entry"] == last) & (a["months"] == 3) & (a["barrier"] == "s1")].set_index(
            "side"
        )

        def today(col: str, side: int, scale: float = 1.0, digits: int = 3) -> str:
            if col not in now or side not in now.index or not np.isfinite(now.loc[side, col]):
                return "–"
            return f"{scale * float(now.loc[side, col]):.{digits}f}"

        def mean(g: pd.DataFrame, s: pd.Series, late_only: bool = False) -> str:
            if late_only:
                keep = g["entry"] >= late
                g, s = g[keep], s[keep]
            return fmt(*self.es(g, s, 3)[:2])

        h, u, hx = self.pnl_h, self.pnl_u, self.pnl_hx
        assert h is not None
        out = [
            "# Knock-out or fly: the framework on one page\n",
            f"Barrier study, report v3, {'strict sample' if self.sample == 'strict' else 'all built entries'}: {f31['entry'].nunique()} weekly entries with finished 3-month trades, "
            f"{f31['entry'].min()} to {f31['entry'].max()}. Numbers are for 3 months and a barrier at one standard deviation, in % of the entry spot unless said "
            f'otherwise; ± is a block standard error over entry dates, [ ] a 95 % block-bootstrap interval; "late" is 2017–2026 alone; "today" is the entry of {last}. '
            "No claim goes beyond the verdict table of the report (section 11j).\n",
        ]
        for side in (1, -1):
            s = SIDE[side]
            g = f31[f31["side"] == side]
            if g.empty:
                continue
            out.append(f"## {s.capitalize()} side\n")
            gap = (
                g["sgn"] * g["lsv_minus_lv"]
                if "lsv_minus_lv" in g
                else pd.Series(np.nan, index=g.index)
            )
            out.append(
                "**1. Forward skew.** What the dealer's model charges for the smile the knock-out is exposed to at the barrier: local vol is one anchor, the LSV the other. "
                f"Signed LSV-minus-local-vol premium {mean(g, gap)} (late {mean(g, gap, True)}; P1 {self.vd('P1', s)}); slope on the roll value {self.vd('P2', s)}; "
                f"fair share between the anchors from the touch days λ_touch = {interval(lx.get((3, side, 's1', 'touch'), nan3))}, from the hedged P&L ex vol carry λ*x = {interval(lx.get((3, side, 's1', 'all'), nan3))} "
                f"(late {interval(lx.get((3, side, 's1', 'late'), nan3))}); P12 {self.vd('P12', s)}. "
                f"Today: gap {today('lsv_minus_lv', side, 100 * side)}, roll {today('roll', side)}, n_spot {today('n_spot', side)}.\n"
            )
            if hx is not None:
                ko_c, fly_c = self.col(g, self.carry, "A1"), self.col(g, self.carry, "B5_2")
                out.append(
                    "**2. Vega sign at the barrier.** A structure short vega collected the realised vol premium, a long one paid it; this is a choice vanillas alone give. "
                    f"Entry vega (% of spot per vol point): knock-out {100 * self.col(g, self.vega, 'A1').mean():.3f}, fly n = 2 {100 * self.col(g, self.vega, 'B5_2').mean():.3f}; "
                    f"realised minus implied vol of the life {fmt(*self.es(g, pd.Series(self.dsig.reindex(g['cell']).to_numpy(), index=g.index), 3)[:2], 1.0, 2)} vol points. "
                    f"Vol carry: knock-out {mean(g, ko_c)}, fly {mean(g, fly_c)} (late: {mean(g, ko_c, True)} and {mean(g, fly_c, True)}). "
                    f"Fly − knock-out, hedged {mean(g, self.col(g, h, 'B5_2') - self.col(g, h, 'A1'))}, of which carry {mean(g, fly_c - ko_c)}, ex vol {mean(g, self.col(g, hx, 'B5_2') - self.col(g, hx, 'A1'))} "
                    f"(late ex vol {mean(g, self.col(g, hx, 'B5_2') - self.col(g, hx, 'A1'), True)}). P13 {self.vd('P13', s)}. "
                    f"Today: vega of the knock-out {today('lv_a1_vega', side, 100)}, of the fly {today('vega_B5_2', side, 100)}; `vrp_ratio` {today('vrp_ratio', side, 1, 2)}.\n"
                )
            t = K.get(("touch", 3, side, "s1"))
            fin31 = self.fin[
                (self.fin["months"] == 3)
                & (self.fin["barrier"] == "s1")
                & (self.fin["side"] == side)
            ]
            k1 = fin31["tau1"].notna().astype(float)
            if t:
                out.append(
                    "**3. The touch.** How often the barrier was touched against what the price assumed, and what the touch cost. "
                    f"Realised daily knock share {t[0][0]:.3f} ± {t[0][1]:.3f} against local vol's {t[1]:.3f} and the LSV's {t[4]:.3f} "
                    f"(late: {fmt(*self.es(fin31[fin31['entry'] >= late], k1[fin31['entry'] >= late], 3)[:2], 1.0)} against {fin31.loc[fin31['entry'] >= late, 'lv_p1'].mean():.3f}); "
                    f"P14 {self.vd('P14', f'{s} 3m 1 sd: realised knock share minus lv_p1')}. "
                    f"At the knock day the smile was the stationary one rather than local vol's (P3 {self.vd('P3', f'{s} 3m')}); at-the-money vol minus local vol's {self.vd('P4', f'{s} 3m')} vol points"
                    + (
                        f"; leaving the static hedge cost {self.vd('P5', 'call 3m')} per knocked trade"
                        if side > 0
                        else ""
                    )
                    + f". Today: local-vol knock probability {today('lv_p1', side)}, LSV {today('ssr12_p1', side)}.\n"
                )
            w = g["P_B5_2"] - g["P_B4"]
            ru, fu = self.col(g, u, "B4"), self.col(g, u, "B5_2")
            out.append(
                "**4. Fly or ratio.** The fly is the ratio plus a far option W that caps the loss beyond the barrier. "
                f"Price of W {100 * w.mean():.3f}; fly − ratio unhedged {mean(g, fu - ru)} (late {mean(g, fu - ru, True)}); worst 1 % unhedged outcome: ratio {100 * ru.quantile(0.01):.2f}, fly {100 * fu.quantile(0.01):.2f}. "
                f"Today: W {today('P_B5_2', side, 100)} − {today('P_B4', side, 100)} (fly and ratio premiums).\n"
            )
            d21 = self.col(g, h, "A2") - self.col(g, h, "A1")
            p9 = self.vd("P9", f"{s} 3m")
            out.append(
                "**5. Daily or continuous monitoring.** The continuous knock-out is cheaper by about a barrier shift of 0.58·σ·√(1/252). "
                f"Premium ratio continuous over daily {(g['lv_a2'] / g['lv_a1'].where(g['lv_a1'] > 2e-4)).mean():.3f}; against the barrier-shift ratio, mean difference {p9}; "
                f"hedged P&L continuous minus daily {mean(g, d21)} (late {mean(g, d21, True)}). "
                f"Today: premiums {today('lv_a2', side, 100)} continuous, {today('lv_a1', side, 100)} daily.\n"
            )
        out.append("## The rules against the always-strategies\n")
        out.append(
            "3 months, standard-deviation barriers and both sides pooled, mean P&L of the pick against always-knock-out and always-fly on the same trades (whole sample; 2017–2026 is in table `rules_exvol`).\n"
        )
        for rule in (
            "1 threshold",
            "2 priced vs historical gap, LV (fly)",
            "2 priced vs historical gap, LSV 1.2 (fly)",
            "4 roll rule, λ0 = 0.5",
        ):
            if ("rule", rule, "all", 3, "hedged") in K:
                out.append(f"- **Rule {rule}**: {self.rule_line(rule, 'all', 3)}.")
        out.append("")
        out.append("## Costs\n")
        for side in (1, -1):
            gross, c05, c2 = (
                K.get(("cost_pair", 3, side, "s1", t)) for t in ("gross", "0.5bp", "2bp")
            )
            if gross:
                out.append(
                    f"- **{SIDE[side].capitalize()} side, hedging.** Notional traded by the daily hedge per trade: knock-out {K.get(('turn', 3, side, 's1', 'A1'), np.nan):.2f}, fly n = 2 {K.get(('turn', 3, side, 's1', 'B5_2'), np.nan):.2f}, "
                    f"tight limit {K.get(('turn', 3, side, 's1', 'B6'), np.nan):.2f} (multiples of the notional). Fly − knock-out hedged: {fmt(*gross)} gross, {fmt(*c05)} net of 0.5 bp, {fmt(*c2)} net of 2 bp per unit traded."
                )
        out.append(
            "- **Dealer break-even.** The knock-out has no market price; the premium at which it would only have matched the fly, hedged and ex vol carry, is the local-vol price plus "
            f"{100 * lx.get((3, 1, 's1', 'all'), (np.nan,) * 5)[3]:.3f} on calls and {100 * lx.get((3, -1, 's1', 'all'), (np.nan,) * 5)[3]:.3f} on puts (report A2.11h); "
            "the vanilla structures' bid-ask costs are in section 12c of the report.\n"
        )
        out.append(
            "*Limits.* The knock-out's premium is a model's. The vol carry is first order (entry vega × realised-minus-implied vol of the life); the regression of A2.11g says how well it holds per trade. "
            "The hedge is a forward to expiry rebalanced once a day at the snapshot. About 78 independent three-month windows carry the inference.\n"
        )
        (self.out / "pm_framework.md").write_text("\n".join(out))

    def build(self) -> None:
        super().build()
        try:
            text = self.findings()
        except Exception as exc:
            import traceback

            text = f"**The findings failed in this render:** `{type(exc).__name__}: {exc}`\n"
            print(f"findings failed:\n{traceback.format_exc()[-1500:]}")
        self.md.insert(2, text)
        self.blocks.insert(2, ("md", text))
        (self.out / "report.md").write_text("\n".join(self.md))
        try:
            self.pm_page()
        except Exception as exc:
            import traceback

            print(
                f"pm_framework.md failed: {type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}"
            )

    # -- assembly ----------------------------------------------------------------------------

    def verdict_table(self) -> None:
        for step in (
            self.attribution, self.carry_regression, self.share_exvol, self.conditions_exvol,
            self.crosstab, self.rules_exvol, self.rule_strategies, self.touch_frequency,
            self.hedge_cost, self.skew_vega, self.lsv_periods, self.today3, self.framework,
        ):  # fmt: skip
            try:
                step()
            except Exception as exc:  # a section that fails must not lose the report
                import traceback

                self.add(
                    f"**Section {step.__name__} failed in this render:** `{type(exc).__name__}: {exc}`\n"
                )
                print(f"section {step.__name__} failed:\n{traceback.format_exc()[-1500:]}")
        super().verdict_table()


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", default=str(bh.OUT / "report"))
    ap.add_argument("--sample", default="strict", choices=["strict", "built"])
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    if (out / "report.pdf").exists() and not (out / "report_v2.pdf").exists():
        shutil.copy2(out / "report.pdf", out / "report_v2.pdf")  # report v2 is kept
    rep = Report3(out, args.sample)
    rep.build()
    print(
        f"report v3: {len(rep.md)} blocks, {len(rep.figs)} figures, {len(rep.verdicts)} verdict rows"
    )
    if not args.no_pdf:
        res = rep.pdf()
        print({k: res[k] for k in ("status", "seconds", "pdf", "problems", "reason") if k in res})


if __name__ == "__main__":
    main()
