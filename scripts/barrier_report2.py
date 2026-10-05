"""Report v2 of the barrier study: the sections of addendum 1 (BARRIER_STUDY_ADDENDUM §9).

    python scripts/barrier_report2.py [--out outputs/interview/report] [--sample strict|built]

Everything of ``scripts/barrier_report.py`` is kept; after "The ladder" come the two anchors,
the forward-skew premium, the model view of the touch, the touch test, the hedged conditions,
the justified share, fly or ratio, the quote-based prices, Rule 4, the verdict on the eleven
predictions and the extended reading of today.  A section whose inputs do not exist yet is
replaced by one line saying so.

New tables aggregate per entry date first and bootstrap over entry dates (blocks of 5, 13, 26,
52 weekly entries by maturity), so that pooling barriers does not shrink the standard errors.
"""

# ruff: noqa: RUF001, E501 — report prose: typographic signs and long caption lines
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barrier_report as base

from volsto.studies import barrier_history as bh
from volsto.studies import barrier_theory as bt

SD, SIDE, BLOCK, HALVES, NAMES = base.SD, base.SIDE, base.BLOCK, base.HALVES, base.NAMES
GATE = bh.OUT / "gate.json"
#: The pre-registered primary conditions (addendum §7.5): label → column (per side where it differs).
PRIMARY2: dict[str, dict[int, str]] = {
    "SSR up (call) / down (put)": {1: "ssr_up_3m_6m", -1: "ssr_dn_3m_6m"},
    "spot-vol correlation": {1: "corr_vol_spot_3m", -1: "corr_vol_spot_3m"},
    "vol premium": {1: "vrp_ratio", -1: "vrp_ratio"},
    "level of vol": {1: "atm", -1: "atm"},
    "level of skew": {1: "n_T", -1: "n_T"},
    "roll": {1: "roll", -1: "roll"},
    "past 3-month return": {1: "ret_3m", -1: "ret_3m"},
}


def entry_stat(
    frame: pd.DataFrame, values: pd.Series, months: int, seed: int = 7
) -> tuple[float, float, int]:
    """Mean over entry dates of the per-entry mean of ``values`` (indexed like ``frame``), its
    block-bootstrap standard error (block = the maturity in weekly entries) and the number of
    entry dates."""
    per = values.groupby(frame["entry"]).mean().dropna().sort_index()
    mean, se = bh.block_bootstrap(per.to_numpy(float), BLOCK[int(months)], seed=seed)
    return mean, se, len(per)


def fmt(mean: float, se: float = float("nan"), scale: float = 100.0, digits: int = 3) -> str:
    return base.fmt(mean, se, scale, digits)


def labels(frame: pd.DataFrame, col: str) -> pd.Series:
    """Tercile of ``col`` against its own history up to the entry (two years minimum), per
    (maturity, side, barrier) series of entries."""
    out = pd.Series("no history", index=frame.index, dtype=object)
    for _, g in frame.groupby(["months", "side", "barrier"]):
        g = g.sort_values("entry")
        out.loc[g.index] = bh.expanding_tercile(g[col], min_history=104).to_numpy()
    return out


class Report2(base.Report):
    def __init__(self, out: Path, sample: str = "strict") -> None:
        super().__init__(out, sample)
        self.gate = json.loads(GATE.read_text()) if GATE.exists() else {}
        for attr in ("cells_all", "c", "fin"):
            setattr(self, attr, self.extend(getattr(self, attr)))
        self.touches = self.read("touches.parquet")
        self.quotes = self.read("quotes.parquet")
        self.verdicts: list[dict[str, Any]] = []
        self.n_conditions = (len(PRIMARY2) + 1, len(base.EXPLORATORY))

    @staticmethod
    def read(name: str) -> pd.DataFrame | None:
        p = bh.RESULTS / name
        return pd.read_parquet(p) if p.exists() else None

    def extend(self, c: pd.DataFrame) -> pd.DataFrame:
        """Add the addendum's columns: features2, the local-vol buckets, and the LSV-derived
        quantities of §4 (model prefix ``ssr12``)."""
        c = c.copy()
        key = ["entry", "months", "side", "barrier"]
        for name in ("features2.parquet", "buckets_lv.parquet"):
            extra = self.read(name)
            if extra is not None:
                extra = extra[[k for k in extra.columns if k in key or k not in c.columns]]
                c = c.merge(extra, on=key, how="left")
        c["sgn"] = c["side"].astype(float)
        c["P_C8_rel"] = c["P_C8"]
        if "A" in c and "ssr12_a2" in c:
            c["Pi_c_ssr12"] = c["sgn"] * (c["ssr12_a2"] - c["P_C8"])
            c["Pi_d_ssr12"] = c["sgn"] * (c["ssr12_a1"] - c["P_C8"])
            c["n_eff_ssr12"] = bt.guarded_ratio(c["Pi_c_ssr12"] - c["Pi_flat"], c["A"])
            c["dn_lsv"] = bt.guarded_ratio(c["Pi_c_ssr12"] - c["Pi_c_lv"], c["A"])
            c["lsv_minus_lv"] = c["ssr12_a1"] - c["lv_a1"]
            c["anchor_gap_c"] = c["sgn"] * (c["ssr12_a2"] - c["lv_a2"])
        return c

    def missing(self, title: str, what: str) -> None:
        self.add(f"## {title}\n")
        self.add(f"Not available in this render: {what}.\n")

    # -- 9.1 the two anchors -----------------------------------------------------------------

    def anchors(self) -> None:
        if "ssr12_a1" not in self.c or self.c["ssr12_a1"].notna().sum() == 0:
            return self.missing("11a. The two anchors", "the LSV entry marks have not run")
        self.add("## 11a. The two anchors: local vol and the LSV\n")
        est = self.gate.get("estimator") or "sorted (the library's default path)"
        self.add(
            f"Particle estimator selected by the gate: **{est}**"
            + (f" — {self.gate.get('reason')}" if self.gate.get("reason") else "")
            + ". Local vol and the LSV (desk mark, SSR 1.2) are the two anchors: least and most "
            "forward skew at the barrier.\n"
        )
        c = self.c[self.c["barrier"].isin(SD) & self.c["ssr12_a1"].notna()]
        rows = []
        for (m, side, b), g in c.groupby(["months", "side", "barrier"]):
            mean, se, n = entry_stat(g, g["sgn"] * g["lsv_minus_lv"], int(m))
            rows.append(
                {
                    "maturity": f"{m}m",
                    "side": SIDE[side],
                    "barrier": b,
                    "LV KO daily": f"{100 * g['lv_a1'].mean():.3f}",
                    "LSV / LV daily": f"{g['ssr12_a1'].sum() / g['lv_a1'].sum():.3f}",
                    "LSV / LV cont.": f"{g['ssr12_a2'].sum() / g['lv_a2'].sum():.3f}",
                    "sgn·(LSV − LV), % spot": fmt(mean, se),
                    "entries": n,
                }
            )
        self.table(
            "anchors",
            pd.DataFrame(rows),
            "The LSV knock-out against the local-vol one, every weekly entry with an LSV mark "
            "(ratio of mean premiums; signed difference with its block standard error). "
            + self.sample_note(c),
            index=False,
        )
        mo = (
            c[c.get("ssr10_a1", pd.Series(np.nan, index=c.index)).notna()]
            if "ssr10_a1" in c
            else c.iloc[:0]
        )
        if len(mo):
            rows = []
            for (m, side, b), g in mo.groupby(["months", "side", "barrier"]):
                rows.append(
                    {
                        "maturity": f"{m}m",
                        "side": SIDE[side],
                        "barrier": b,
                        **{
                            f"{k} / LV": f"{g[f'{k}_a1'].sum() / g['lv_a1'].sum():.3f}"
                            for k in ("ssr10", "ssr12", "ssr15")
                        },
                        "entries": g["entry"].nunique(),
                    }
                )
            self.table(
                "anchors_marks",
                pd.DataFrame(rows),
                "The three desk marks (SSR 1.0, 1.2, 1.5) on the monthly subset: daily knock-out "
                "over the local-vol one. They agree: the band is not a band.",
                index=False,
            )
        for side in (1, -1):
            by = []
            for m in (1, 3, 6, 12):
                g = c[(c["months"] == m) & (c["side"] == side)]
                mean, se, _ = entry_stat(g, g["sgn"] * g["lsv_minus_lv"], m)
                by.append((m, mean, se))
            inc = all(by[i][1] < by[i + 1][1] for i in range(3))
            self.verdict(
                "P1",
                f"{SIDE[side]}: mean sgn·(LSV − LV) by maturity "
                + ", ".join(f"{m}m {100 * a:.3f}±{100 * s:.3f}" for m, a, s in by)
                + (" (increasing)" if inc else " (not increasing)"),
                by[1][1],
                by[1][2],
                +1,
                note="verdict on the 3m estimate; all four maturities shown",
            )

    # -- 9.2 the forward-skew premium --------------------------------------------------------

    def skew_premium(self) -> None:
        if "A" not in self.c:
            return self.missing("11b. The forward-skew premium", "features2 has not run")
        self.add("## 11b. The forward-skew premium\n")
        c = self.c[self.c["barrier"].isin(SD)]
        has_lsv = "Pi_c_ssr12" in c and c["Pi_c_ssr12"].notna().any()
        rows = []
        for (m, side, b), g in c.groupby(["months", "side", "barrier"]):
            ko = g["lv_a1"].mean()
            row: dict[str, Any] = {
                "maturity": f"{m}m",
                "side": SIDE[side],
                "barrier": b,
                "Π_c LV": f"{100 * g['Pi_c_lv'].mean():.3f}",
                "Π_d LV": f"{100 * g['Pi_d_lv'].mean():.3f}",
                "Π_flat": f"{100 * g['Pi_flat'].mean():.3f}",
                "A": f"{100 * g['A'].mean():.3f}",
                "n_eff LV": f"{g['n_eff_lv'].median():.3f}",
                "n_spot": f"{g['n_spot'].median():.3f}",
                "roll": f"{g['roll'].median():.3f}",
                "roll value / KO": f"{g['roll_value'].mean() / ko:.2f}" if ko > 2e-5 else "–",
                "ratios NaN": int(g["n_eff_lv"].isna().sum()),
            }
            if has_lsv:
                row["Π_c LSV"] = f"{100 * g['Pi_c_ssr12'].mean():.3f}"
                row["Π_d LSV"] = f"{100 * g['Pi_d_ssr12'].mean():.3f}"
                row["n_eff LSV"] = f"{g['n_eff_ssr12'].median():.3f}"
                row["dn LSV"] = f"{g['dn_lsv'].median():.3f}"
            rows.append(row)
        self.table(
            "premium",
            pd.DataFrame(rows),
            "Π = sgn·(knock-out − C8), continuous (c) and daily (d), % of spot (means); Π_flat the "
            "same under a flat smile with the entry's carry; A the price of one unit of normalised "
            "skew at the barrier, % of spot; n_eff = (Π_c − Π_flat)/A the skew the model's price "
            "implies, n_spot the surface's, roll = n_spot − n_eff[LV] (medians); the roll's value "
            "as a share of the local-vol knock-out price; ratios whose denominator is under 0.2 bp "
            "are NaN and counted. " + self.sample_note(c),
            index=False,
        )
        if has_lsv:
            _, axes = plt.subplots(1, 2, figsize=(10, 3.6))
            for side, ax in zip((1, -1), axes, strict=True):
                g = c[(c["side"] == side) & c["anchor_gap_c"].notna() & c["roll_value"].notna()]
                x, y = g["roll_value"].to_numpy(float), g["anchor_gap_c"].to_numpy(float)
                slope = float(np.sum(x * y) / np.sum(x * x))
                # block bootstrap of the slope over entry dates
                per = (
                    g.groupby("entry")
                    .apply(
                        lambda h: pd.Series(
                            {
                                "xy": (h["roll_value"] * h["anchor_gap_c"]).sum(),
                                "xx": (h["roll_value"] ** 2).sum(),
                            }
                        )
                    )
                    .sort_index()
                )
                rng = np.random.default_rng(11)
                n, blk = len(per), 13
                starts = rng.integers(0, n, size=(1000, int(np.ceil(n / blk))))
                idx = ((starts[:, :, None] + np.arange(blk)[None, None, :]) % n).reshape(1000, -1)[
                    :, :n
                ]
                boots = per["xy"].to_numpy()[idx].sum(1) / per["xx"].to_numpy()[idx].sum(1)
                se = float(boots.std(ddof=1))
                ax.scatter(100 * x, 100 * y, s=2, alpha=0.25)
                lim = np.array([np.nanmin(100 * x), np.nanmax(100 * x)])
                ax.plot(lim, slope * lim, "r-", lw=1, label=f"slope {slope:.2f} ± {se:.2f}")
                ax.plot(lim, lim, "k--", lw=0.6, label="slope 1")
                ax.set_xlabel("roll value, % of spot")
                ax.set_ylabel("sgn·(LSV − LV), continuous KO, % of spot")
                ax.set_title(f"{SIDE[side]} side")
                ax.legend(fontsize=7)
                lo_, hi_ = slope - 2 * se, slope + 2 * se
                where = (
                    "inside"
                    if (lo_ >= 0.7 and hi_ <= 1.0)
                    else ("overlaps" if (hi_ >= 0.7 and lo_ <= 1.0) else "outside")
                )
                self.verdict(
                    "P2",
                    f"{SIDE[side]}: slope through the origin of sgn·(ssr12_a2 − lv_a2) on roll_value",
                    slope,
                    se,
                    +1,
                    scale=1.0,
                    note=f"sign rule; the slope ± 2 standard errors is {where} the predicted [0.7, 1]",
                )
            self.figure(
                "roll_scatter",
                "The LSV-minus-local-vol gap of the continuous knock-out against the roll value (prediction P2).",
            )
        # the normalised skew by tenor through time
        tenors = [k for k in ("n_1m", "n_2m", "n_3m", "n_6m", "n_12m") if k in self.cells_all]
        if tenors:
            d = self.cells_all.drop_duplicates("entry").set_index("entry")[tenors].sort_index()
            plt.figure(figsize=(10, 3.2))
            t = pd.to_datetime(d.index)
            for k in tenors:
                plt.plot(
                    t, d[k].rolling(8, min_periods=1).mean(), lw=0.8, label=k.replace("n_", "")
                )
            plt.legend(fontsize=7, ncol=5)
            plt.title("Normalised at-the-money skew n(θ) = −skew·√θ by tenor (8-week mean)")
            self.figure(
                "n_tenors",
                "Is the normalised skew flat across tenors? n(θ) at 1, 2, 3, 6 and 12 months through time.",
            )
            med = d.median()
            self.add(
                "Median n(θ) over the entries: "
                + ", ".join(f"{k.replace('n_', '')} {med[k]:.3f}" for k in tenors)
                + ".\n"
            )

    # -- 9.3 model view of the touch ---------------------------------------------------------

    def model_touch(self) -> None:
        if "lv_d_p1" not in self.c:
            return self.missing(
                "11c. Model view of the touch", "the bucket statistics have not run"
            )
        self.add("## 11c. Model view of the touch: local vol against the LSV\n")
        c = self.c[self.c["barrier"].isin(["s0.5", "s1", "s1.5"])]
        has_lsv = "ssr12_d_p1" in c and c["ssr12_d_p1"].notna().any()
        rows = []
        for (m, side, b), g in c.groupby(["months", "side", "barrier"]):
            for model in ("lv", "ssr12") if has_lsv else ("lv",):
                h = g[g[f"{model}_d_p1"].notna()]
                if h.empty:
                    continue
                row: dict[str, Any] = {
                    "maturity": f"{m}m",
                    "side": SIDE[side],
                    "barrier": b,
                    "model": "LV" if model == "lv" else "LSV",
                }
                for j in (1, 2, 3):
                    row[f"p{j} cont."] = f"{h[f'{model}_c_p{j}'].mean():.3f}"
                for j in (1, 2, 3):
                    row[f"d{j} daily"] = f"{h[f'{model}_d_d{j}'].mean():.3f}"
                for j in (1, 2, 3):
                    row[f"fly at touch {j}"] = (
                        f"{100 * (h[f'{model}_d_vB5_2{j}'] / h['K']).mean():.3f}"
                    )
                for j in (1, 2, 3):
                    row[f"tight limit at touch {j}"] = (
                        f"{100 * (h[f'{model}_d_vB6{j}'] / h['K']).mean():.3f}"
                    )
                w, kap = h["w_rel"], None
                if "sig_B" in h:
                    tl = np.column_stack([h[f"{model}_c_tl{j}"].fillna(0.0) for j in (1, 2, 3)])
                    pj = np.column_stack([h[f"{model}_c_p{j}"] for j in (1, 2, 3)])
                    kap = np.column_stack(
                        [
                            bt.kap(
                                1.0, (h["B"] / h["K"]).to_numpy(), h["sig_B"].to_numpy(), tl[:, j]
                            )
                            for j in range(3)
                        ]
                    )
                    a_paths = bt.ANNUITY_CONSTANT * w.to_numpy() * (pj * kap).sum(1)
                    row["A paths"] = f"{100 * np.nanmean(a_paths):.3f}"
                    row["A closed form"] = f"{100 * h['A'].mean():.3f}"
                rows.append(row)
        self.table(
            "model_touch",
            pd.DataFrame(rows),
            "From each model's pricing paths at entry, by third of the life in which the first "
            "touch falls: touch probabilities (continuous rule), probability of finishing beyond "
            "the barrier given a daily touch, and the expected payoff of the fly (n = 2) and of the "
            "tight limit given a daily touch there (% of spot); the annuity from the paths beside "
            "its closed form. " + self.sample_note(c),
            index=False,
        )

    # -- 9.4 the touch test ------------------------------------------------------------------

    def touch_test(self) -> None:
        t = self.touches
        if t is None or t.empty:
            return self.missing("11d. The touch test", "the touch-day measurements have not run")
        self.add("## 11d. The touch test\n")
        keep = self.fin[["cell", "strict", "built", "w_rel", "DF0", "half"]]
        t = t.merge(keep, on="cell")
        t = t[t[self.sample]]
        t["abs_stat"] = (t["n_real"] - t["n_stat"]).abs()
        t["abs_lv"] = (t["n_real"] - t["n_lv"]).abs()
        t["p3"] = t["abs_stat"] - t["abs_lv"]
        t["p4"] = t["sig_real"] - t["sig_lv"]
        t["skew_term"] = bt.ANNUITY_CONSTANT * t["w_rel"] * t["kap_touch"] * t["n_real"]
        self.add(
            f"{len(t)} trades knocked under the daily rule on {t['entry'].nunique()} entry dates "
            "(standard-deviation barriers; trades with fewer than 3 trading days left or a carried "
            "touch-day surface left out). θ is the time left; all skews use one centred difference "
            "of half-width max(0.5·σ·√θ, 1 %) around the forward.\n"
        )
        rows = []
        for (m, side, b), g in t.groupby(["months", "side", "barrier"]):
            row: dict[str, Any] = {
                "maturity": f"{m}m",
                "side": SIDE[side],
                "barrier": b,
                "n": len(g),
            }
            for col, name in (
                ("sig_real", "σ real"),
                ("sig_stat", "σ stationary"),
                ("sig_ss", "σ sticky strike"),
                ("sig_lv", "σ LV"),
            ):
                row[name] = f"{100 * g[col].mean():.2f}"
            for col, name in (
                ("n_real", "n real"),
                ("n_stat", "n stationary"),
                ("n_ss", "n sticky strike"),
                ("n_lv", "n LV"),
            ):
                row[name] = f"{g[col].mean():.3f}"
            a, s, _ = entry_stat(g, g["ssr_at_touch"], int(m))
            row["SSR at touch"] = fmt(a, s, 1.0, 2)
            a, s, _ = entry_stat(g, g["skew_ratio"], int(m))
            row["skew ratio"] = fmt(a, s, 1.0, 2)
            a, s, _ = entry_stat(g, g["resid_skew"], int(m))
            row["resid_skew"] = fmt(a, s)
            row["0.8·w·κ·n_real"] = f"{100 * g['skew_term'].mean():.3f}"
            for name in ("B6", "B5_2", "C8"):
                row[f"{NAMES.get(name, name)}: mark / LV"] = (
                    f"{100 * g[f'mark_{name}'].mean():.3f} / {100 * g[f'lv_{name}'].mean():.3f}"
                )
            rows.append(row)
        self.table(
            "touch",
            pd.DataFrame(rows),
            "On the knock day: at-the-money vol (%) and normalised skew of the touch-day surface "
            "against the stationary, sticky-strike and local-vol predictions made from the entry "
            "day; SSR at the touch (0 sticky moneyness, 1 sticky strike, 2 local vol) and the skew "
            "ratio n_real/n_stat (± block standard error); the smile part of the residual of "
            "'knock-out minus C8' against 0.8·w·κ·n_real; the day's marks of the tight limit, the "
            "fly and C8 against their local-vol restart values (% of spot, valued at expiry).",
            index=False,
        )
        # predictions P3, P4, P5
        for side in (1, -1):
            g = t[t["side"] == side]
            for m_ in (1, 3):
                gm = g[g["months"] == m_]
                a, s, n = entry_stat(gm, gm["p3"], m_)
                self.verdict(
                    "P3",
                    f"{SIDE[side]} {m_}m: mean |n_real − n_stat| − |n_real − n_lv| ({n} entry dates)",
                    a,
                    s,
                    -1,
                    scale=1.0,
                )
                a, s, n = entry_stat(gm, gm["p4"], m_)
                self.verdict(
                    "P4",
                    f"{SIDE[side]} {m_}m: mean σ_real − σ_lv at touches, vol points",
                    a,
                    s,
                    side,
                    scale=100.0,
                )
        gc = t[(t["side"] == 1) & (t["months"] == 3)]
        a, s, _ = entry_stat(gc, gc["resid_skew"], 3)
        self.verdict(
            "P5",
            f"call 3m: mean resid_skew on knocked trades (beside 0.8·w·κ·n_real = {100 * gc['skew_term'].mean():.3f})",
            a,
            s,
            +1,
        )
        gc = t[(t["side"] == 1) & (t["months"] == 1)]
        a, s, _ = entry_stat(gc, gc["resid_skew"], 1)
        self.verdict(
            "P5",
            f"call 1m: mean resid_skew on knocked trades (beside 0.8·w·κ·n_real = {100 * gc['skew_term'].mean():.3f})",
            a,
            s,
            +1,
        )
        # regression: the fly's and the tight limit's value at the touch on vol and skew
        rows = []
        for side in (1, -1):
            for name in ("B5_2", "B6"):
                g = t[(t["side"] == side) & t[f"mark_{name}"].notna()].copy()
                g["x1"] = g["sig_real"] * np.sqrt(g["theta"])
                y = g[f"mark_{name}"] / g["w_rel"]
                X = np.column_stack([np.ones(len(g)), g["x1"], g["n_real"]])
                ok = np.isfinite(X).all(1) & np.isfinite(y)
                beta = np.linalg.lstsq(X[ok], y[ok], rcond=None)[0]
                # block bootstrap over entry dates
                ents = np.array(sorted(g.loc[ok, "entry"].unique()))
                pos = {e: np.flatnonzero((g["entry"] == e).to_numpy() & ok) for e in ents}
                rng = np.random.default_rng(5)
                boots = []
                for _ in range(300):
                    st = rng.integers(0, len(ents), size=int(np.ceil(len(ents) / 13)))
                    pick = np.concatenate(
                        [
                            np.concatenate([pos[ents[(s0 + k) % len(ents)]] for k in range(13)])
                            for s0 in st
                        ]
                    )
                    boots.append(np.linalg.lstsq(X[pick], y.to_numpy()[pick], rcond=None)[0])
                se = np.std(boots, axis=0, ddof=1)
                rows.append(
                    {
                        "side": SIDE[side],
                        "structure": NAMES[name],
                        "n": int(ok.sum()),
                        "const": fmt(beta[0], se[0], 1.0),
                        "coef. on σ√θ": fmt(beta[1], se[1], 1.0),
                        "coef. on n_real": fmt(beta[2], se[2], 1.0),
                    }
                )
                if name == "B5_2":
                    z = abs(beta[2] / se[2]) if se[2] > 0 else np.inf
                    self.verdicts.append(
                        {
                            "prediction": "P6",
                            "statistic": f"{SIDE[side]}: coefficient of the fly's touch-day mark (over w) on n_real",
                            "estimate": f"{beta[2]:.3f} ± {se[2]:.3f}",
                            "verdict": "inconclusive" if z < 2 else "contradicted",
                            "note": "the prediction is 'no dependence': it cannot be confirmed by the sign rule; contradicted when two standard errors from zero",
                        }
                    )
                elif side == 1:
                    self.verdict(
                        "P6",
                        "call: coefficient of the tight limit's touch-day mark (over w) on n_real",
                        float(beta[2]),
                        float(se[2]),
                        -1,
                        scale=1.0,
                    )
        self.table(
            "touch_regression",
            pd.DataFrame(rows),
            "Knocked trades: the touch-day mark (over the barrier distance w) of the fly and of the "
            "tight limit regressed on σ_real·√θ and n_real; block-bootstrap standard errors over "
            "entry dates.",
            index=False,
        )
        self.lam_touch(t)

    def lam_touch(self, t: pd.DataFrame) -> None:
        """Per trade, the not-knocked ones as zeros: mean realised residual against Π_d under
        each anchor, and the realised share."""
        f = self.fin[self.fin["barrier"].isin(SD)].copy()
        if "Pi_d_lv" not in f:
            return
        f = f.merge(t[["cell", "resid_pv"]], on="cell", how="left")
        f["resid0"] = f["resid_pv"].where(f["tau1"].notna(), 0.0)  # knocked but skipped stays NaN
        has_lsv = "Pi_d_ssr12" in f and f["Pi_d_ssr12"].notna().any()
        rows = []
        self.lam_touch_values: dict[tuple[int, int], tuple[float, float, float]] = {}
        for (m, side), g in f.groupby(["months", "side"]):
            g = g[g["resid0"].notna() & (g["Pi_d_ssr12"].notna() if has_lsv else True)]
            per = (
                g.groupby("entry")[["resid0", "Pi_d_lv", *(["Pi_d_ssr12"] if has_lsv else [])]]
                .mean()
                .sort_index()
            )
            row: dict[str, Any] = {"maturity": f"{m}m", "side": SIDE[side], "entries": len(per)}
            row["mean resid"] = f"{100 * per['resid0'].mean():.3f}"
            row["Π_d LV"] = f"{100 * per['Pi_d_lv'].mean():.3f}"
            if has_lsv and len(per) > 30:
                row["Π_d LSV"] = f"{100 * per['Pi_d_ssr12'].mean():.3f}"
                arr = per.to_numpy(float)
                rng = np.random.default_rng(3)
                n, blk = len(arr), BLOCK[int(m)]
                starts = rng.integers(0, n, size=(2000, int(np.ceil(n / blk))))
                idx = ((starts[:, :, None] + np.arange(blk)[None, None, :]) % n).reshape(2000, -1)[
                    :, :n
                ]
                mu = arr[idx].mean(axis=1)
                den = mu[:, 2] - mu[:, 1]
                lam_b = np.where(np.abs(den) >= bt.RATIO_FLOOR, (mu[:, 0] - mu[:, 1]) / den, np.nan)
                d0 = per["Pi_d_ssr12"].mean() - per["Pi_d_lv"].mean()
                lam = (
                    (per["resid0"].mean() - per["Pi_d_lv"].mean()) / d0
                    if abs(d0) >= bt.RATIO_FLOOR
                    else np.nan
                )
                lo, hi = np.nanpercentile(lam_b, [2.5, 97.5])
                row["λ_touch"] = f"{lam:.2f}"
                row["95 % interval"] = f"[{lo:.2f}, {hi:.2f}]"
                self.lam_touch_values[(int(m), int(side))] = (float(lam), float(lo), float(hi))
            rows.append(row)
        self.table(
            "lam_touch",
            pd.DataFrame(rows).fillna("–"),
            "Per trade, the trades not knocked counted as zero: the realised residual of 'daily "
            "knock-out minus C8' (minus sgn times the mark of C8 on the knock day, discounted to "
            "the entry), against what each anchor charges for it (Π_d), % of spot; and the realised "
            "share λ_touch = (resid − Π_d[LV]) / (Π_d[LSV] − Π_d[LV]) with a block-bootstrap "
            "interval: 0 = local vol was right, 1 = the LSV was right. Standard-deviation barriers "
            "pooled.",
            index=False,
        )

    # -- 9.5 conditions, hedged ----------------------------------------------------------------

    def conditions_hedged(self) -> None:
        if not self.has_hedge:
            return self.missing("11e. Conditions, hedged", "the hedged outcomes do not exist")
        self.add("## 11e. Conditions at entry, hedged outcomes (the tests)\n")
        self.add(
            f"Pre-registered primary conditions: {self.n_conditions[0]} (the seven below and the "
            f"distance to the barrier); exploratory conditions examined elsewhere in this report: "
            f"{self.n_conditions[1]}. With that many, an isolated two-standard-error result among "
            "the exploratory ones is to be read as such.\n"
        )
        f = self.fin[
            self.fin["barrier"].isin(SD)
            & self.fin["months"].isin([1, 3])
            & self.fin["marks_complete"].fillna(False).astype(bool)
        ].copy()
        h = self.pnl_h
        assert h is not None
        f["fly_ko"] = (h["B5_2"] - h["A1"]).reindex(f["cell"]).to_numpy()
        f["b6_ko"] = (h["B6"] - h["A1"]).reindex(f["cell"]).to_numpy()
        f["ko"] = h["A1"].reindex(f["cell"]).to_numpy()
        rows = []
        self.cond_labels: dict[str, pd.Series] = {}
        for label, cols in PRIMARY2.items():
            if not all(col in f for col in cols.values()):
                continue
            f["_cond"] = np.where(f["side"] > 0, f[cols[1]], f[cols[-1]])
            lab = labels(f, "_cond")
            self.cond_labels[label] = lab
            for scope, sel in (
                ("sd pooled", f["barrier"].isin(SD)),
                ("1 sd", f["barrier"] == "s1"),
            ):
                for (m, side), g in f[sel].groupby(["months", "side"]):
                    for period, lo, hi in (("all", "2007", "2027"), *HALVES):
                        gp = g[(g["entry"] >= lo) & (g["entry"] <= hi)]
                        row: dict[str, Any] = {
                            "condition": label,
                            "barriers": scope,
                            "maturity": f"{m}m",
                            "side": SIDE[side],
                            "period": period,
                        }
                        stats = {}
                        for key in ("low", "mid", "high"):
                            gg = gp[lab.reindex(gp.index) == key]
                            a, s, n = entry_stat(gg, gg["fly_ko"], int(m))
                            stats[key] = (a, s, n, gg)
                            row[f"fly − KO, {key}"] = fmt(a, s)
                        for key in ("low", "high"):
                            gg = stats[key][3]
                            a, s, _ = entry_stat(gg, gg["b6_ko"], int(m))
                            row[f"tight − KO, {key}"] = fmt(a, s)
                            a, s, _ = entry_stat(gg, gg["ko"], int(m))
                            row[f"KO, {key}"] = fmt(a, s)
                        row["entries low / high"] = f"{stats['low'][2]} / {stats['high'][2]}"
                        rows.append(row)
                        if (
                            label.startswith("SSR up")
                            and side == 1
                            and scope == "sd pooled"
                            and period == "all"
                        ):
                            d = stats["high"][0] - stats["low"][0]
                            se = float(np.hypot(stats["high"][1], stats["low"][1]))
                            self.verdict(
                                "P7",
                                f"call {m}m: hedged fly − KO, top minus bottom tercile of ssr_up",
                                d,
                                se,
                                +1,
                            )
        self.table(
            "conditions_hedged",
            pd.DataFrame(rows),
            "Delta-hedged P&L (local-vol sticky-strike delta), % of spot ± block standard error over "
            "entry dates, of fly minus knock-out, tight limit minus knock-out and the knock-out "
            "alone, by tercile of each primary condition against its own history; barriers pooled "
            "and at one standard deviation; whole sample and each half. " + self.sample_note(f),
            index=False,
        )
        self.frame_hedged = f

    # -- 9.6 the justified share ---------------------------------------------------------------

    def share(self) -> None:
        f = getattr(self, "frame_hedged", None)
        if f is None or "lsv_minus_lv" not in f or f["lsv_minus_lv"].notna().sum() == 0:
            return self.missing(
                "11f. The justified share", "needs the hedged outcomes and the LSV entry marks"
            )
        self.add("## 11f. The justified share\n")
        f = f[f["lsv_minus_lv"].notna()].copy()
        f["be_fly"] = -f["DF0"] * f["fly_ko"]  # DF0·(pnl_h[KO] − pnl_h[fly])
        f["be_b6"] = -f["DF0"] * f["b6_ko"]

        def lam_of(g: pd.DataFrame, col: str, m: int) -> tuple[float, float, float, float]:
            per = g.groupby("entry")[[col, "lsv_minus_lv"]].mean().sort_index()
            arr = per.to_numpy(float)
            if len(arr) < 30:
                return (float("nan"),) * 4
            den0 = arr[:, 1].mean()
            lam = arr[:, 0].mean() / den0 if abs(den0) >= bt.RATIO_FLOOR else float("nan")
            rng = np.random.default_rng(9)
            n, blk = len(arr), BLOCK[m]
            starts = rng.integers(0, n, size=(2000, int(np.ceil(n / blk))))
            idx = ((starts[:, :, None] + np.arange(blk)[None, None, :]) % n).reshape(2000, -1)[
                :, :n
            ]
            mu = arr[idx].mean(axis=1)
            lb = np.where(np.abs(mu[:, 1]) >= bt.RATIO_FLOOR, mu[:, 0] / mu[:, 1], np.nan)
            lo, hi = np.nanpercentile(lb, [2.5, 97.5])
            return float(lam), float(lo), float(hi), float(arr[:, 0].mean())

        rows = []
        self.lam_star: dict[tuple[int, int, str, str], float] = {}
        for (m, side), g in f.groupby(["months", "side"]):
            row: dict[str, Any] = {"maturity": f"{m}m", "side": SIDE[side], "bucket": "all"}
            for col, name in (("be_fly", "fly"), ("be_b6", "tight limit")):
                lam, lo, hi, num = lam_of(g, col, int(m))
                row[f"P* − LV vs {name}, % spot"] = f"{100 * num:.3f}"
                row[f"λ* vs {name}"] = f"{lam:.2f} [{lo:.2f}, {hi:.2f}]"
                if name == "fly":
                    self.verdicts.append(
                        {
                            "prediction": "P8",
                            "statistic": f"{SIDE[side]} {m}m: λ* against the fly (hedged)",
                            "estimate": f"{lam:.2f} [{lo:.2f}, {hi:.2f}]",
                            "verdict": (
                                "confirmed"
                                if (lo >= 0 and hi <= 1)
                                else ("contradicted" if (hi < 0 or lo > 1) else "inconclusive")
                            ),
                            "note": "confirmed = the 95 % interval inside [0, 1]; contradicted = wholly outside",
                        }
                    )
            row["mean LSV − LV, % spot"] = f"{100 * g['lsv_minus_lv'].mean():.3f}"
            lt = getattr(self, "lam_touch_values", {}).get((int(m), int(side)))
            row["λ_touch"] = f"{lt[0]:.2f} [{lt[1]:.2f}, {lt[2]:.2f}]" if lt else "–"
            if lt:
                self.verdicts.append(
                    {
                        "prediction": "P8",
                        "statistic": f"{SIDE[side]} {m}m: λ_touch",
                        "estimate": f"{lt[0]:.2f} [{lt[1]:.2f}, {lt[2]:.2f}]",
                        "verdict": (
                            "confirmed"
                            if (lt[1] >= 0 and lt[2] <= 1)
                            else ("contradicted" if (lt[2] < 0 or lt[1] > 1) else "inconclusive")
                        ),
                        "note": "same rule",
                    }
                )
            rows.append(row)
            for label, lab in getattr(self, "cond_labels", {}).items():
                for key in ("low", "mid", "high"):
                    gg = g[lab.reindex(g.index) == key]
                    lam, lo, hi, num = lam_of(gg, "be_fly", int(m))
                    lam2, lo2, hi2, num2 = lam_of(gg, "be_b6", int(m))
                    self.lam_star[(int(m), int(side), label, key)] = lam
                    rows.append(
                        {
                            "maturity": f"{m}m",
                            "side": SIDE[side],
                            "bucket": f"{label}: {key}",
                            "P* − LV vs fly, % spot": f"{100 * num:.3f}",
                            "λ* vs fly": f"{lam:.2f} [{lo:.2f}, {hi:.2f}]",
                            "P* − LV vs tight limit, % spot": f"{100 * num2:.3f}",
                            "λ* vs tight limit": f"{lam2:.2f} [{lo2:.2f}, {hi2:.2f}]",
                            "mean LSV − LV, % spot": f"{100 * gg['lsv_minus_lv'].mean():.3f}",
                            "λ_touch": "–",
                        }
                    )
        self.table(
            "share",
            pd.DataFrame(rows),
            "Break-even knock-out premium from hedged P&L: P* − lv_a1 = mean of DF0·(hedged P&L of "
            "the knock-out − hedged P&L of X), X the fly (n = 2) or the tight limit; λ* = that over "
            "the mean LSV-minus-local-vol premium (0 = the local-vol price was the fair one, 1 = the "
            "LSV price), with a 95 % block-bootstrap interval; overall and by tercile of the primary "
            "conditions; standard-deviation barriers pooled. " + self.sample_note(f),
            index=False,
        )

    # -- 9.7 fly or ratio ----------------------------------------------------------------------

    def fly_or_ratio(self) -> None:
        self.add("## 11g. Fly or ratio: the upside call W\n")
        f = self.fin[self.fin["barrier"].isin(SD) & self.fin["months"].isin([1, 3])].copy()
        u, h = self.pnl_u, self.pnl_h
        f["W"] = f["P_B5_2"] - f["P_B4"]
        f["w_u"] = (u["B5_2"] - u["B4"]).reindex(f["cell"]).to_numpy()
        f["fly_u"], f["ratio_u"] = (
            u["B5_2"].reindex(f["cell"]).to_numpy(),
            u["B4"].reindex(f["cell"]).to_numpy(),
        )
        if h is not None:
            f["w_h"] = (h["B5_2"] - h["B4"]).reindex(f["cell"]).to_numpy()
        rows = []
        for (m, side, b), g in f.groupby(["months", "side", "barrier"]):
            a, s, _ = entry_stat(g, g["w_u"], int(m))
            row: dict[str, Any] = {
                "maturity": f"{m}m",
                "side": SIDE[side],
                "barrier": b,
                "W price": f"{100 * g['W'].mean():.3f}",
                "W / fly price": f"{g['W'].sum() / g['P_B5_2'].sum():.2f}",
                "fly − ratio unhedged": fmt(a, s),
            }
            if "w_h" in g:
                a2, s2, _ = entry_stat(g, g["w_h"], int(m))
                row["fly − ratio hedged"] = fmt(a2, s2)
            for q in (0.01, 0.05):
                row[f"worst {int(100 * q)} %: ratio / fly"] = (
                    f"{100 * g['ratio_u'].quantile(q):.2f} / {100 * g['fly_u'].quantile(q):.2f}"
                )
            if "upside_skew" in g:
                row["upside skew, vp"] = f"{100 * g['upside_skew'].mean():.2f}"
            if "rich_W_h2" in g:
                row["richness of W"] = f"{g['rich_W_h2'].median():.2f}"
            rows.append(row)
            if side == 1 and m == 3 and b in ("s1.5", "s2"):
                self.verdicts.append(
                    {
                        "prediction": "P10",
                        "statistic": f"call 3m {b}: price of W {100 * g['W'].mean():.3f} % of spot; unhedged fly − ratio",
                        "estimate": f"{100 * a:.3f} ± {100 * s:.3f}",
                        "verdict": (
                            "confirmed"
                            if (g["W"].mean() < 5e-4 and abs(a) < 2 * s)
                            else (
                                "contradicted"
                                if abs(a) >= 2 * s and g["W"].mean() >= 5e-4
                                else "inconclusive"
                            )
                        ),
                        "note": "confirmed = W below 5 bp of spot and fly − ratio within two standard errors of zero",
                    }
                )
        self.table(
            "fly_or_ratio",
            pd.DataFrame(rows),
            "The fly is the ratio plus the far option W (strike 2B − K): its price, its share of the "
            "fly's price, the P&L of fly minus ratio (% of spot ± block standard error), the worst "
            "1 % and 5 % unhedged outcomes of the ratio against the fly's, the upside skew (implied "
            "vol of the far strike minus at-the-money) and the richness of W (priced over "
            "historical gap at the forecast vol, median). " + self.sample_note(f),
            index=False,
        )

    # -- 9.8 quotes ----------------------------------------------------------------------------

    def quote_prices(self) -> None:
        q = self.quotes
        if q is None or q.empty:
            return self.missing(
                "11h. Quote-based prices",
                "the quote-based repricing has not run (last priority of the addendum)",
            )
        self.add("## 11h. Vanilla prices from the quotes\n")
        key = ["entry", "months", "side", "barrier"]
        f = self.fin[self.fin["barrier"].isin(SD)].merge(q, on=key, how="inner")
        rows = []
        for (m, side, b), g in f.groupby(["months", "side", "barrier"]):
            row: dict[str, Any] = {
                "maturity": f"{m}m",
                "side": SIDE[side],
                "barrier": b,
                "n": len(g),
            }
            for name in ("B3", "B4", "B5_2"):
                d = (g[f"Pq_{name}"] - g[f"P_{name}"]).abs()
                rel = d / g[f"P_{name}"].abs().where(g[f"P_{name}"].abs() > 2e-5)
                row[f"{NAMES[name]} |diff| bp mean / p90"] = (
                    f"{1e4 * d.mean():.2f} / {1e4 * d.quantile(0.9):.2f}"
                )
                row[f"{NAMES[name]} % of price"] = f"{100 * rel.median():.1f}"
            row["strikes beyond the listed range"] = int(g["Pq_B5_2"].isna().sum())
            for piece, (lo_, hi_) in (("UP", ("B3", "B5_2")), ("W", ("B5_2", "B4"))):
                paid = (self.terminal[lo_] - self.terminal[hi_]).reindex(g["cell"]).to_numpy() * g[
                    "DF0"
                ].to_numpy()
                for tag, pre in (("surface", "P_"), ("quotes", "Pq_")):
                    e = (g[f"{pre}{lo_}"] - g[f"{pre}{hi_}"]).to_numpy() - paid
                    row[f"{piece} priced−realised ({tag})"] = f"{100 * np.nanmean(e):.3f}"
            rows.append(row)
        self.table(
            "quotes",
            pd.DataFrame(rows),
            "Vanilla structures repriced from the quotes (implied vol interpolated between the two "
            "neighbouring listed strikes and, in total variance at fixed log-moneyness, between the "
            "two expiries around the trade's): absolute difference with the surface price in bp of "
            "spot (mean and 90th percentile) and in % of the price (median); and the ladder pieces "
            "UP and W, priced minus realised, with surface and with quote-based premiums.",
            index=False,
        )

    # -- 9.9 rule 4 ----------------------------------------------------------------------------

    def rule4(self) -> None:
        f = self.fin[self.fin["barrier"].isin(SD)].copy()
        if "dn_lsv" not in f or "n_spot" not in f or self.touches is None:
            return self.missing(
                "11i. Rule 4, the roll rule", "needs features2, the LSV marks and the touch test"
            )
        self.add("## 11i. Rule 4, the roll rule (pre-registered, nothing fitted)\n")
        # rho: the expanding mean of skew_ratio over touches that happened strictly before the entry, same side
        t = self.touches[["side", "tau", "skew_ratio"]].dropna().sort_values("tau")
        rho = pd.Series(1.0, index=f.index)
        for side in (1, -1):
            ts = t[t["side"] == side]
            taus, cs = ts["tau"].to_numpy(str), np.cumsum(ts["skew_ratio"].to_numpy(float))
            sel = f["side"] == side
            k = np.searchsorted(taus, f.loc[sel, "entry"].to_numpy(str), side="left")
            rho.loc[sel] = np.where(k >= 30, cs[np.maximum(k, 1) - 1] / np.maximum(k, 1), 1.0)
        f["rho"] = rho
        f["n_exp"] = f["n_spot"] * f["rho"]
        f = f[f["dn_lsv"].notna() & f["n_eff_lv"].notna()].set_index("cell", drop=False)
        srcs = [("unhedged", self.pnl_u)] + ([("hedged", self.pnl_h)] if self.has_hedge else [])
        rows = []
        self.rule4_today: dict[float, pd.Series] = {}
        for lam0 in (0.0, 0.5, 1.0):
            charged = f["n_eff_lv"] + lam0 * f["dn_lsv"]
            ko = np.where(f["side"] > 0, charged < f["n_exp"], charged > f["n_exp"])
            pick = pd.Series(np.where(ko, "A1", "B5_2"), index=f.index)
            extra = lam0 * f["lsv_minus_lv"] / f["DF0"]  # the knock-out bought at share lam0
            for half, lo, hi in HALVES:
                for m in (1, 3, 6, 12):
                    g = f[(f["entry"] >= lo) & (f["entry"] <= hi) & (f["months"] == m)]
                    row: dict[str, Any] = {
                        "λ0": lam0,
                        "period": half,
                        "maturity": f"{m}m",
                        "share KO": f"{(pick.reindex(g.index) == 'A1').mean():.2f}",
                    }
                    for tag, src in srcs:
                        assert src is not None
                        p_ko = src["A1"].reindex(g.index) - extra.reindex(g.index)
                        p_fly = src["B5_2"].reindex(g.index)
                        chosen = pd.Series(
                            np.where(pick.reindex(g.index) == "A1", p_ko, p_fly), index=g.index
                        )
                        a, s, n = entry_stat(g, chosen, m)
                        row[f"{tag} P&L of the pick"] = fmt(a, s)
                        a, s, _ = entry_stat(g, chosen - p_ko, m)
                        row[f"{tag} vs always KO"] = fmt(a, s)
                        a, s, _ = entry_stat(g, chosen - p_fly, m)
                        row[f"{tag} vs always fly"] = fmt(a, s)
                        row["entries"] = n
                    rows.append(row)
            self.rule4_today[lam0] = pick
        self.table(
            "rule4",
            pd.DataFrame(rows),
            "Rule 4 for a dealer price at share λ0 between the anchors: n_charged = n_eff[LV] + "
            "λ0·dn_lsv against n_exp = n_spot·ρ (ρ the expanding mean of the realised skew ratio at "
            "earlier touches on the same side, 1 below 30 touches). Call side: knock-out if "
            "n_charged < n_exp, else the fly; put side the reverse. The knock-out's P&L at share λ0 "
            "is its local-vol-premium P&L minus λ0·(LSV − LV)/DF0. % of spot ± block standard error "
            "over entry dates; standard-deviation barriers and both sides pooled. "
            + self.sample_note(f),
            index=False,
        )

    # -- 9.10 verdict --------------------------------------------------------------------------

    def verdict(
        self,
        pred: str,
        statistic: str,
        est: float,
        se: float,
        sign: int,
        scale: float = 100.0,
        note: str = "",
    ) -> None:
        """The rule of the addendum and no other: confirmed if the sign is as predicted and the
        estimate is at least two standard errors from zero; contradicted if opposite and at least
        two standard errors; otherwise inconclusive."""
        if not (np.isfinite(est) and np.isfinite(se) and se > 0):
            v = "inconclusive"
        elif abs(est) >= 2 * se:
            v = "confirmed" if np.sign(est) == np.sign(sign) else "contradicted"
        else:
            v = "inconclusive"
        self.verdicts.append(
            {
                "prediction": pred,
                "statistic": statistic,
                "estimate": f"{scale * est:.3f} ± {scale * se:.3f}",
                "verdict": v,
                "note": note,
            }
        )

    def other_predictions(self) -> None:
        """P9 (continuous over daily) and P11 (drift) from the tables already in hand."""
        c = self.c[self.c["barrier"].isin(SD) & (self.c["lv_a1"] > 2e-4)]
        if "sig_B" in c:
            for side in (1, -1):
                g = c[(c["side"] == side) & (c["months"] == 3) & c["sig_B"].notna()].copy()
                shift = np.exp(0.5826 * g["sig_B"] * np.sqrt(1 / 252))
                pred = []
                for r in g.itertuples():
                    b_shift = r.B * (shift[r.Index] if side > 0 else 1 / shift[r.Index])
                    rr, qq = bt.carry_rates(r.DF0, r.F0, r.K, r.T)
                    a = bt.ko_rr(side, r.K, r.K, r.B, r.T, r.sig_B, rr, qq)
                    d = bt.ko_rr(side, r.K, r.K, b_shift, r.T, r.sig_B, rr, qq)
                    pred.append(a / d if d > 0 else np.nan)
                g["pred"] = pred
                g["ratio"] = g["lv_a2"] / g["lv_a1"]
                a, s, _ = entry_stat(g, g["ratio"] - g["pred"], 3)
                z = abs(a / s) if s > 0 else np.inf
                self.verdicts.append(
                    {
                        "prediction": "P9",
                        "statistic": f"{SIDE[side]} 3m: mean a2/a1 = {g['ratio'].mean():.3f} against the barrier-shift ratio {np.nanmean(pred):.3f}; mean difference",
                        "estimate": f"{a:.4f} ± {s:.4f}",
                        "verdict": "inconclusive" if z < 2 else "contradicted",
                        "note": "the prediction is 'no difference': it cannot be confirmed by the sign rule; contradicted when two standard errors from zero (local-vol prices)",
                    }
                )
        if self.has_hedge:
            f = self.fin[
                self.fin["barrier"].isin(SD)
                & (self.fin["months"] == 3)
                & self.fin["marks_complete"].fillna(False).astype(bool)
            ]
            assert self.pnl_h is not None
            for side in (1, -1):
                g = f[f["side"] == side]
                du = (self.pnl_u["B5_2"] - self.pnl_u["A1"]).reindex(g["cell"]).to_numpy()
                dh = (self.pnl_h["B5_2"] - self.pnl_h["A1"]).reindex(g["cell"]).to_numpy()
                ret = g["close_T"].to_numpy() - 1.0
                ok = np.isfinite(du) & np.isfinite(dh) & np.isfinite(ret)
                X = np.column_stack([np.ones(ok.sum()), ret[ok], ret[ok] ** 2])
                r2 = {}
                for tag, y in (("unhedged", du[ok]), ("hedged", dh[ok])):
                    res = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
                    r2[tag] = 1 - res.var() / y.var()
                self.verdicts.append(
                    {
                        "prediction": "P11",
                        "statistic": f"{SIDE[side]} 3m: share of the variance of fly − KO explained by the trade's spot return (and its square)",
                        "estimate": f"unhedged {r2['unhedged']:.2f}, hedged {r2['hedged']:.2f}",
                        "verdict": (
                            "confirmed"
                            if r2["unhedged"] > 0.5 and r2["unhedged"] > r2["hedged"]
                            else (
                                "contradicted" if r2["unhedged"] < r2["hedged"] else "inconclusive"
                            )
                        ),
                        "note": "confirmed = more than half unhedged and more than hedged (no standard error: a variance share)",
                    }
                )

    def verdict_table(self) -> None:
        self.add("## 11j. Verdict on the predictions\n")
        if not self.verdicts:
            self.add("No prediction could be evaluated in this render.\n")
            return
        frame = pd.DataFrame(self.verdicts).sort_values(
            "prediction", key=lambda s: s.str[1:].astype(int), kind="stable"
        )
        self.table(
            "verdict",
            frame,
            "One row per statistic of each prediction. Verdict by one rule: confirmed if the sign is "
            "as predicted and the estimate is at least two standard errors from zero; contradicted "
            "if the sign is opposite and at least two standard errors; otherwise inconclusive "
            "(predictions stated as 'no difference' or as an interval carry their own rule in the "
            "note). Estimates in % of spot unless the statistic says otherwise.",
            index=False,
        )

    # -- 9.11 today ----------------------------------------------------------------------------

    def today2(self) -> None:
        a = self.cells_all[self.cells_all["built"]]
        if "n_spot" not in a:
            return
        self.add("## 14b. Today's reading, extended\n")
        last = a["entry"].max()
        t = a[(a["entry"] == last) & a["barrier"].isin(SD)]
        rows = []
        for r in t.sort_values(["months", "side", "barrier"]).itertuples():
            row: dict[str, Any] = {
                "maturity": f"{r.months}m",
                "side": SIDE[r.side],
                "barrier": r.barrier,
                "n_spot": f"{r.n_spot:.3f}",
                "n_eff LV": f"{r.n_eff_lv:.3f}",
                "roll": f"{r.roll:.3f}",
                "A, % spot": f"{100 * r.A:.3f}",
                "roll value": f"{100 * r.roll_value:.3f}",
            }
            if hasattr(r, "lsv_minus_lv") and np.isfinite(getattr(r, "lsv_minus_lv", np.nan)):
                row["LSV − LV"] = f"{100 * r.lsv_minus_lv:.3f}"
            for lam0, pick in getattr(self, "rule4_today", {}).items():
                row[f"Rule 4, λ0={lam0:g}"] = NAMES.get(str(pick.get(r.cell, "")), "–")
            rows.append(row)
        self.table(
            "today2",
            pd.DataFrame(rows).fillna("–"),
            f"The addendum's quantities on the latest entry date ({last}).",
            index=False,
        )
        rows = []
        for label, cols in PRIMARY2.items():
            for m in (1, 3):
                for side in (1, -1):
                    col = cols[side]
                    if col not in a:
                        continue
                    hist = a[
                        (a["months"] == m) & (a["side"] == side) & (a["barrier"] == "s1")
                    ].sort_values("entry")
                    if hist.empty or not np.isfinite(hist[col].iloc[-1]):
                        continue
                    v = float(hist[col].iloc[-1])
                    past = hist[col].iloc[:-1].dropna()
                    pct = float((past <= v).mean())
                    key = "low" if pct <= 1 / 3 else ("high" if pct > 2 / 3 else "mid")
                    lam = getattr(self, "lam_star", {}).get((m, side, label, key), float("nan"))
                    rows.append(
                        {
                            "condition": label,
                            "maturity": f"{m}m",
                            "side": SIDE[side],
                            "value": f"{v:.4g}",
                            "percentile": f"{100 * pct:.0f}",
                            "tercile": key,
                            "λ* of that tercile": f"{lam:.2f}" if np.isfinite(lam) else "–",
                        }
                    )
        if rows:
            self.table(
                "today2_conditions",
                pd.DataFrame(rows),
                f"Each primary condition on {last} (barrier at 1 sd): value, percentile against its history, and the justified share λ* of trades entered in the same tercile.",
                index=False,
            )

    # -- assembly ------------------------------------------------------------------------------

    def ladder(self) -> None:
        super().ladder()
        for step in (
            self.anchors,
            self.skew_premium,
            self.model_touch,
            self.touch_test,
            self.conditions_hedged,
            self.share,
            self.fly_or_ratio,
            self.quote_prices,
            self.rule4,
            self.other_predictions,
            self.verdict_table,
        ):
            try:
                step()
            except Exception as exc:  # a section that fails must not lose the report
                import traceback

                self.add(
                    f"**Section {step.__name__} failed in this render:** `{type(exc).__name__}: {exc}`\n"
                )
                print(f"section {step.__name__} failed:\n{traceback.format_exc()[-1200:]}")

    def today(self) -> None:
        super().today()
        try:
            self.today2()
        except Exception as exc:
            self.add(f"**Section today2 failed in this render:** `{type(exc).__name__}: {exc}`\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--out", default=str(bh.OUT / "report"))
    ap.add_argument("--sample", default="strict", choices=["strict", "built"])
    ap.add_argument("--no-pdf", action="store_true")
    args = ap.parse_args()
    rep = Report2(Path(args.out), args.sample)
    rep.build()
    print(
        f"report v2: {len(rep.md)} blocks, {len(rep.figs)} figures, {len(rep.verdicts)} verdict rows"
    )
    if not args.no_pdf:
        res = rep.pdf()
        print({k: res[k] for k in ("status", "seconds", "pdf", "problems", "reason") if k in res})


if __name__ == "__main__":
    main()
