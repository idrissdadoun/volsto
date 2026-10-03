"""The barrier-versus-vanilla study, stage 2 — the report (runner module of
``configs/studies/barrier_vs_vanilla/study.yaml``; the owner's question of 2026-10-03).  Reads
the tables that ``scripts/barrier_vs_vanilla_study.py`` writes under ``<outputs>/<params.dir>/``
(:mod:`volsto.studies.barrier_vs_vanilla`) and never prices or calibrates; the anchor,
decomposition, history and decision tables are required (a missing one prints the stage-1
command and the run exits 2), the map, Greeks, hedge and daily tables are reported when
present.  The theory is ``docs/barrier_vs_vanilla.md``.

Results tables (values ×100 in % of the spot unless stated): ``setup``; ``prices_up`` /
``prices_down`` (every barrier product under BS / LV / 1F / 2F and the surface's model-free
value of the European knock-outs and the vanilla structures); ``decomposition_up`` /
``decomposition_down`` (per maturity and level: the European knock-out, the daily and
continuous knock-outs, the regret value and share, the model spread, the touch and regret
probabilities, the spread / fly / ratio and the premium-matched alternatives); ``map_up`` /
``map_down`` (spot × time to expiry); ``greeks_<product>``; ``hedge``; ``daily_<product>``
(the 127-day series) and ``daily_summary``; ``history_<tag>`` (touch and regret frequencies
and the structures' payoffs per layer); ``decision`` (the framework's score per layer).

Checked by ``tests/test_barrier_vs_vanilla.py``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from volsto.studies.results import Column, FigureSpec, Results, ResultsBuilder, TableSpec
from volsto.studies.runner import Requirement, StudyContext, artefact_path
from volsto.studies.style import new_figure

TITLE = "Up-and-out calls against call ratios and flies (and the put mirror): the framework"
QUESTION = (
    "On the SPX 2022-12-30 desk mark and over 2022 H2 and 1990-2026, when does a call ratio or "
    "a call fly beat an up-and-out call (daily or continuous), and a put ratio or put fly a "
    "down-and-out put - by pure-vol, market and statistical metrics?"
)
REQUIRED_PARAMS: tuple[str, ...] = ("dir",)
OPTIONAL_PARAMS: tuple[str, ...] = ()
COMMAND = ".venv/bin/python scripts/barrier_vs_vanilla_study.py --parts {part}"
REQUIRED_FILES: dict[str, tuple[str, ...]] = {
    "anchor": ("anchor.csv", "decomposition.csv", "setup.csv"),
    "history": ("history.csv", "decision.csv"),
}
OPTIONAL_FILES: dict[str, tuple[str, ...]] = {
    "map": ("map.csv",),
    "greeks": ("greeks.csv",),
    "hedge": ("hedge.csv",),
    "daily": ("daily.csv",),
}
PCT = "% of spot"
DAILY_PRODUCTS: tuple[str, ...] = ("uoc 3m 110", "uoc 6m 110", "dop 3m 90", "dop 6m 90")
DAILY_COLUMNS: tuple[str, ...] = (
    "uoc_LV",
    "EKO",
    "regret_share",
    "fly",
    "fly_over_uoc",
    "ratio_matched",
    "distance_sd",
    "iv_minus_rv_3m",
    "atm_3m",
    "skew_3m",
    "curv_3m",
    "vix",
)
LAYER_ORDER: tuple[str, ...] = (
    "2F mark (Q)",
    "history",
    "history since anchor",
    "FHS at realised 3m",
    "FHS at implied ATM",
    "FHS at implied ATM, historical drift",
)


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_")


def validate_params(params: Mapping[str, Any]) -> None:
    if not isinstance(params["dir"], str) or not params["dir"]:
        raise ValueError("dir must be the stage-1 output directory relative to outputs")


def requirements(ctx: StudyContext) -> list[Requirement]:
    out = []
    for part, files in REQUIRED_FILES.items():
        for f in files:
            out.append(
                ctx.artefact_requirement(
                    f"{ctx.params['dir']}/{f}",
                    f"barrier-vs-vanilla {part}",
                    COMMAND.format(part=part),
                )
            )
    return out


def _path(ctx: StudyContext, name: str) -> Any:
    return artefact_path(ctx.outputs_root, f"{ctx.params['dir']}/{name}")


def _read(ctx: StudyContext, name: str) -> pd.DataFrame | None:
    p = _path(ctx, name)
    return pd.read_csv(p) if p.exists() else None


def _num(
    b: ResultsBuilder,
    table: str,
    row: str,
    col: str,
    v: Any,
    se: Any = float("nan"),
    *,
    unit: str = PCT,
    source: str,
    **kw: Any,
) -> None:
    """A number with its stderr when it has one, exact otherwise; NaN values are skipped."""
    v = float(v)
    if not math.isfinite(v):
        return
    se = float(se) if se is not None else float("nan")
    if math.isfinite(se) and se > 0:
        # a Monte Carlo value needs a unit: ratios and probabilities carry '1'
        b.add(table, row, col, v, se, unit=unit or "1", source=source, **kw)
    else:
        b.add_exact(table, row, col, v, unit=unit or "1", source=source, **kw)


def _label(tag: str, side: str, level: float) -> str:
    return f"{tag} {side} {round(100 * level)}%"


def compute(ctx: StudyContext) -> Results:
    b = ResultsBuilder()
    d = ctx.params["dir"]
    src = {
        k: f"artefact:{d}/{k}.csv"
        for k in (
            "anchor",
            "decomposition",
            "map",
            "greeks",
            "hedge",
            "daily",
            "history",
            "decision",
            "setup",
        )
    }
    anchor = pd.read_csv(_path(ctx, "anchor.csv"))
    dec = pd.read_csv(_path(ctx, "decomposition.csv"))
    hist = pd.read_csv(_path(ctx, "history.csv"))
    decision = pd.read_csv(_path(ctx, "decision.csv"))
    setup = pd.read_csv(_path(ctx, "setup.csv"))
    for k, v in zip(setup["key"], setup["value"], strict=True):
        if math.isfinite(float(v)):
            b.add_exact("setup", str(k), "value", float(v), unit="", source=src["setup"])
    b.add_exact(
        "setup",
        "barrier products",
        "value",
        float(anchor[anchor["family"] == "barrier"]["product"].nunique()),
        unit="",
        source=src["anchor"],
    )
    # prices
    for _, r in anchor[anchor["family"] != "probability"].iterrows():
        t = "prices_up" if r["side"] == "up" else "prices_down"
        _num(
            b,
            t,
            str(r["product"]),
            str(r["model"]),
            r["value"],
            r["stderr"],
            source=src["anchor"],
            axes={"tag": r["tag"], "level": r["level"], "role": r["role"]},
        )
    # decomposition
    mc_cols = (
        ("2F", "2F_stderr"),
        ("2F cont", "2F_cont_stderr"),
        ("EKO 2F", "EKO_2F_stderr"),
        ("2F - LV", "2F_minus_LV_stderr"),
    )
    exact_cols = (
        "BS",
        "LV",
        "1F",
        "EKO surface",
        "regret value",
        "regret share",
        "continuous discount",
        "model spread",
        "p touch",
        "p regret",
        "spread",
        "fly",
        "ratio 1x2",
        "fly over uoc",
        "fly units",
        "ratio matched",
        "ratio breakeven",
        "fly matched top",
        "fly matched max payoff",
    )
    keymap = {
        "2F cont": "2F_cont",
        "EKO 2F": "EKO_2F",
        "2F - LV": "2F_minus_LV",
        "EKO surface": "EKO_surface",
        "regret value": "regret_value",
        "regret share": "regret_share",
        "continuous discount": "continuous_discount",
        "model spread": "model_spread",
        "p touch": "p_touch",
        "p regret": "p_regret",
        "ratio 1x2": "ratio_1x2",
        "fly over uoc": "fly_over_uoc",
        "fly units": "fly_units_for_premium",
        "ratio matched": "ratio_matched",
        "ratio breakeven": "ratio_matched_breakeven",
        "fly matched top": "fly_matched_top",
        "fly matched max payoff": "fly_matched_max_payoff",
    }
    unitless = {
        "regret share",
        "p touch",
        "p regret",
        "fly over uoc",
        "fly units",
        "ratio matched",
        "ratio breakeven",
        "fly matched top",
    }
    for _, r in dec.iterrows():
        t = "decomposition_up" if r["side"] == "up" else "decomposition_down"
        row = _label(str(r["tag"]), str(r["side"]), float(r["level"]))
        ax = {"tag": r["tag"], "level": r["level"], "side": r["side"]}
        for col, sec in mc_cols:
            _num(
                b,
                t,
                row,
                col,
                r[keymap.get(col, col)],
                r[sec],
                source=src["decomposition"],
                axes=ax,
            )
        for col in exact_cols:
            _num(
                b,
                t,
                row,
                col,
                r[keymap.get(col, col)],
                unit=("" if col in unitless else PCT),
                source=src["decomposition"],
                axes=ax,
            )
    # map
    mp = _read(ctx, "map.csv")
    if mp is not None:
        for _, r in mp.iterrows():
            t = "map_up" if r["side"] == "up" else "map_down"
            row = f"tau {float(r['tau']):.3g}y spot {float(r['spot_rel']):.3g}"
            ax = {"tau": r["tau"], "spot_rel": r["spot_rel"], "side": r["side"]}
            _num(b, t, row, "uoc 2F", r["uoc_2F"], r["uoc_2F_stderr"], source=src["map"], axes=ax)
            for col, key, unit in (
                ("EKO", "EKO", PCT),
                ("regret value", "regret_value", PCT),
                ("spread", "spread", PCT),
                ("fly", "fly", PCT),
                ("ratio 1x2", "ratio_1x2", PCT),
                ("fly over uoc", "fly_over_uoc", ""),
                ("ratio matched", "ratio_matched", ""),
                ("distance sd", "distance_sd", ""),
            ):
                _num(b, t, row, col, r[key], unit=unit, source=src["map"], axes=ax)
    # greeks
    gk = _read(ctx, "greeks.csv")
    if gk is not None:
        for _, r in gk.iterrows():
            t = f"greeks_{slug(str(r['product']))}"
            parts = [str(r["group"]), str(r["name"])]
            for k in ("regime", "variant"):
                if k in r and isinstance(r[k], str) and r[k]:
                    parts.append(str(r[k]))
            scale = (
                float(r["scale"]) if "scale" in r and math.isfinite(float(r["scale"])) else 100.0
            )
            _num(
                b,
                t,
                " ".join(parts),
                "value",
                scale * float(r["value"]),
                scale * float(r["stderr"]),
                unit=f"{PCT} per unit",
                source=src["greeks"],
            )
    # hedge
    hd = _read(ctx, "hedge.csv")
    if hd is not None:
        for _, r in hd.iterrows():
            row = f"{r['product']} | {r['strategy']} | {r['world']}" + (
                f" | {r['regime']}"
                if "regime" in r and isinstance(r["regime"], str) and r["regime"] != "model"
                else ""
            )
            for col, key, sek in (
                ("std zc", "std_zc", "std_zc_stderr"),
                ("mean zc", "mean_zc", "mean_zc_stderr"),
                ("es01 tc", "es01_tc", "es01_tc_stderr"),
                ("std tc", "std_tc", "std_tc_stderr"),
            ):
                _num(
                    b,
                    "hedge",
                    row,
                    col,
                    r[key],
                    r[sek],
                    source=src["hedge"],
                    axes={"product": r["product"], "strategy": r["strategy"], "world": r["world"]},
                )
            _num(b, "hedge", row, "product std", r["product_std"], source=src["hedge"])
            _num(b, "hedge", row, "std ratio", r["std_ratio_zc"], unit="", source=src["hedge"])
            _num(b, "hedge", row, "costs", r["costs_mean"], source=src["hedge"])
    # daily
    dy = _read(ctx, "daily.csv")
    if dy is not None:
        for p in DAILY_PRODUCTS:
            sub = dy[dy["product"] == p]
            t = f"daily_{slug(p)}"
            for _, r in sub.iterrows():
                for col in DAILY_COLUMNS:
                    if col in r:
                        unit = PCT if col in ("uoc_LV", "EKO", "fly") else ""
                        se = r["uoc_LV_stderr"] if col == "uoc_LV" else float("nan")
                        _num(b, t, str(r["date"]), col, r[col], se, unit=unit, source=src["daily"])
        # summary per product over the dates
        for prod, sub in dy.groupby("product", sort=False):
            for col in ("regret_share", "fly_over_uoc", "ratio_matched", "distance_sd"):
                x = sub[col].to_numpy(dtype=np.float64)
                x = x[np.isfinite(x)]
                if x.size:
                    _num(
                        b,
                        "daily_summary",
                        str(prod),
                        f"{col} mean",
                        float(x.mean()),
                        unit="",
                        source=src["daily"],
                    )
                    _num(
                        b,
                        "daily_summary",
                        str(prod),
                        f"{col} min",
                        float(x.min()),
                        unit="",
                        source=src["daily"],
                    )
                    _num(
                        b,
                        "daily_summary",
                        str(prod),
                        f"{col} max",
                        float(x.max()),
                        unit="",
                        source=src["daily"],
                    )
            for col, key in (
                ("corr regret-distance", "distance_sd"),
                ("corr regret-ivrv", "iv_minus_rv_3m"),
            ):
                xy = sub[["regret_share", key]].dropna().to_numpy(dtype=np.float64)
                if key == "distance_sd":
                    xy[:, 1] = np.abs(xy[:, 1])  # the put side's distance is negative
                if len(xy) > 3:
                    _num(
                        b,
                        "daily_summary",
                        str(prod),
                        col,
                        float(np.corrcoef(xy.T)[0, 1]),
                        unit="",
                        source=src["daily"],
                    )
            # realised outcomes (entries whose horizon fits the history)
            if "realised_uoc" in sub:
                ok = sub.dropna(subset=["realised_uoc"])
                if len(ok):
                    _num(
                        b,
                        "daily_outcomes",
                        str(prod),
                        "entries",
                        float(len(ok)),
                        unit="",
                        source=src["daily"],
                    )
                    _num(
                        b,
                        "daily_outcomes",
                        str(prod),
                        "premium LV mean",
                        float(ok["uoc_LV"].mean()),
                        source=src["daily"],
                    )
                    for col in (
                        "realised_uoc",
                        "realised_eko",
                        "realised_spread",
                        "realised_fly",
                        "realised_ratio",
                    ):
                        _num(
                            b,
                            "daily_outcomes",
                            str(prod),
                            col.replace("realised_", "realised "),
                            float(ok[col].mean()),
                            source=src["daily"],
                        )
                    _num(
                        b,
                        "daily_outcomes",
                        str(prod),
                        "touched share",
                        float(ok["realised_touched"].mean()),
                        unit="",
                        source=src["daily"],
                    )
                    _num(
                        b,
                        "daily_outcomes",
                        str(prod),
                        "regret share realised",
                        float(ok["realised_regret"].mean()),
                        unit="",
                        source=src["daily"],
                    )
    # history
    for _, r in hist.iterrows():
        t = f"history_{r['tag']}"
        row = f"{_label(str(r['tag']), str(r['side']), float(r['level']))} | {r['layer']}" + (
            f" | {r['regime']} {r['group']}" if r["regime"] != "all" else ""
        )
        ax = {
            "tag": r["tag"],
            "side": r["side"],
            "level": r["level"],
            "layer": r["layer"],
            "regime": r["regime"],
            "group": r["group"],
        }
        for col in ("touched", "regret", "uoc", "eko", "spread", "fly", "ratio"):
            if col in r and math.isfinite(float(r[col])):
                sek = f"{col}_stderr"
                se = float(r[sek]) if sek in r else float("nan")
                scale = 100.0 if col not in ("touched", "regret") else 1.0
                _num(
                    b,
                    t,
                    row,
                    col,
                    scale * float(r[col]),
                    scale * se if math.isfinite(se) else se,
                    unit=("" if scale == 1.0 else PCT),
                    source=src["history"],
                    axes=ax,
                )
        if "n_eff" in r and math.isfinite(float(r["n_eff"])):
            _num(b, t, row, "n eff", float(r["n_eff"]), unit="", source=src["history"], axes=ax)
        if "fhs_terminal_vol" in r and math.isfinite(float(r["fhs_terminal_vol"])):
            _num(
                b,
                t,
                row,
                "fhs vol",
                float(r["fhs_terminal_vol"]),
                unit="",
                source=src["history"],
                axes=ax,
            )
    # decision
    for _, r in decision.iterrows():
        row = f"{_label(str(r['tag']), str(r['side']), float(r['level']))} | {r['layer']}"
        ax = {"tag": r["tag"], "side": r["side"], "level": r["level"], "layer": r["layer"]}
        for col, key in (("premium 2F", "premium_2F"),):
            _num(b, "decision", row, col, r[key], source=src["decision"], axes=ax)
        for col, key in (
            ("uoc payoff/premium", "uoc_payoff_to_premium"),
            ("fly payoff/premium", "fly_payoff_to_premium"),
            ("ratio payoff/premium", "ratio_payoff_to_premium"),
            ("edge uoc-fly", "edge_uoc_minus_fly"),
            ("touch gap P-Q", "touch_gap_P_minus_Q"),
            ("regret gap P-Q", "regret_gap_P_minus_Q"),
            ("model band share", "model_band_share"),
            ("hedge std share", "hedge_std_share"),
            ("hedge cost share", "hedge_cost_share"),
        ):
            if key in r:
                _num(b, "decision", row, col, r[key], unit="", source=src["decision"], axes=ax)
        b.add_exact(
            "decision",
            row,
            "verdict",
            1.0
            if r["verdict_vs_fly"] == "barrier"
            else (0.0 if r["verdict_vs_fly"] == "fly" else 0.5),
            unit="",
            source=src["decision"],
            note=str(r["verdict_vs_fly"]),
            axes=ax,
        )
    return b.build()


# --------------------------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------------------------


def _cols(results: Results, table: str, keys: tuple[str, ...] | None = None) -> tuple[Column, ...]:
    have = results.columns(table)
    return tuple(Column(c, c) for c in (keys or have) if c in have)


def tables(results: Results) -> list[TableSpec]:
    have = set(results.tables())
    specs: list[TableSpec] = []
    for side in ("up", "down"):
        t = f"decomposition_{side}"
        if t in have:
            specs.append(
                TableSpec(
                    t,
                    f"The {'up-and-out call' if side == 'up' else 'down-and-out put'} decomposed (x100 in % of spot): Black-Scholes at the ATM vol, LV, the reference 1F, the 2F desk mark (daily closes, strict) and its continuous twin; the European knock-out (surface: model-free; 2F: Monte Carlo); the regret value EKO - UOC(2F), its share of the EKO, the continuous discount, the LV / 1F / 2F spread; the 2F touch and regret probabilities; the spread, the midpoint fly and the 1x2 ratio off the surface; the fly-to-knock-out price ratio, the premium-matched ratio and its break-even (per unit spot), the premium-matched symmetric fly's top strike (per unit spot) and its maximum payoff.",
                    t,
                    _cols(
                        results,
                        t,
                        (
                            "BS",
                            "LV",
                            "1F",
                            "2F",
                            "2F cont",
                            "EKO surface",
                            "EKO 2F",
                            "regret share",
                            "continuous discount",
                            "model spread",
                            "p touch",
                            "p regret",
                            "spread",
                            "fly",
                            "ratio 1x2",
                            "fly over uoc",
                            "ratio matched",
                            "ratio breakeven",
                            "fly matched top",
                            "fly matched max payoff",
                        ),
                    ),
                    row_header="maturity side barrier",
                )
            )
        t = f"map_{side}"
        if t in have:
            specs.append(
                TableSpec(
                    t,
                    f"The {'6m 110% up-and-out call' if side == 'up' else '6m 90% down-and-out put'} across spot and time to expiry under the 2F mark (leverage held in spot, surface held in strike): the knock-out, the European knock-out, the midpoint fly, the fly-to-knock-out ratio, the premium-matched ratio, the distance to the barrier in standard deviations.",
                    t,
                    _cols(
                        results,
                        t,
                        (
                            "distance sd",
                            "uoc 2F",
                            "EKO",
                            "regret value",
                            "fly",
                            "fly over uoc",
                            "ratio matched",
                        ),
                    ),
                    row_header="state",
                )
            )
    for t in sorted(x for x in have if x.startswith("greeks_")):
        specs.append(
            TableSpec(
                t,
                f"Risk report of `{t[7:].replace('_', ' ')}` under the 2F mark (recalibrate mode), x100 in % of spot per bump unit.",
                t,
                (Column("value", "value"),),
                row_header="greek",
            )
        )
    if "hedge" in have:
        specs.append(
            TableSpec(
                "hedge",
                "Hedged P&L (x100 in % of spot, 2e4 world paths, daily): std and mean without costs, 1% expected shortfall and std with costs, the unhedged product's std, the std ratio and the mean costs; the barrier rows with `fly proxy` / `ratio proxy` hold the premium-matched vanilla structure short against the knock-out (static, optionally plus the delta); the barrier's own baselines come from the payoff study.",
                "hedge",
                _cols(
                    results,
                    "hedge",
                    ("std zc", "mean zc", "es01 tc", "std tc", "product std", "std ratio", "costs"),
                ),
                row_header="product | strategy | world",
            )
        )
    if "daily_summary" in have:
        specs.append(
            TableSpec(
                "daily_summary",
                "The 127 snapshots of 2022 H2 under each day's local vol: the regret share, the fly-to-knock-out ratio, the premium-matched ratio and the distance in standard deviations (mean, min, max over the dates) and the correlation of the regret share with the distance and with implied-minus-realised 3m vol.",
                "daily_summary",
                _cols(results, "daily_summary"),
                row_header="product",
            )
        )
    if "daily_outcomes" in have:
        specs.append(
            TableSpec(
                "daily_outcomes",
                "Realised outcomes of the trades entered on the snapshot dates whose horizon ends inside the history (x100 in % of spot): the LV premium at entry against the realised payoffs of the knock-out, the European knock-out, the spread, the midpoint fly and the 1x2 ratio, with the share of entries that touched and of regret paths.",
                "daily_outcomes",
                _cols(results, "daily_outcomes"),
                row_header="product",
            )
        )
    for tag in ("3m", "6m", "1y"):
        t = f"history_{tag}"
        if t in have:
            specs.append(
                TableSpec(
                    t,
                    f"The statistical layer at {tag}: the touch and regret frequencies and the structures' payoffs (x100 in % of spot, per unit notional) under the 2F mark (Q), the 1990-2026 history (overall, since the anchor, by VIX tercile, by the sign of implied-minus-realised vol and of the 3m momentum), and the filtered historical simulation at the anchor's realised 3m vol and implied ATM vol (demeaned) and with the historical drift; `n eff` the effective sample of overlapping windows, `fhs vol` the simulated terminal vol.",
                    t,
                    _cols(
                        results,
                        t,
                        (
                            "touched",
                            "regret",
                            "uoc",
                            "eko",
                            "spread",
                            "fly",
                            "ratio",
                            "n eff",
                            "fhs vol",
                        ),
                    ),
                    row_header="barrier | layer | regime",
                )
            )
    if "decision" in have:
        specs.append(
            TableSpec(
                "decision",
                "The framework's score per barrier and statistical layer: the 2F premium, the expected payoff per unit premium of the knock-out, of the premium-matched fly (midpoint fly scaled to the premium) and of the premium-matched ratio under the layer, the edge (knock-out minus fly), the P-minus-Q touch and regret gaps, the model band and the hedge std / cost as shares of the premium, and the verdict (1 = barrier, 0 = fly, 0.5 = inside the model band).",
                "decision",
                _cols(
                    results,
                    "decision",
                    (
                        "premium 2F",
                        "uoc payoff/premium",
                        "fly payoff/premium",
                        "ratio payoff/premium",
                        "edge uoc-fly",
                        "touch gap P-Q",
                        "regret gap P-Q",
                        "model band share",
                        "hedge std share",
                        "verdict",
                    ),
                ),
                row_header="barrier | layer",
            )
        )
    return specs


# --------------------------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------------------------


def _table_frame(results: Results, table: str) -> pd.DataFrame:
    f = results.frame
    return f[f["table"] == table]


def _draw_regret(results: Results) -> Figure:
    fig, axes = new_figure(1, 2, height=2.6)
    for ax, side in zip(np.atleast_1d(axes).ravel(), ("up", "down"), strict=False):
        t = f"decomposition_{side}"
        if t not in results.tables():
            ax.set_visible(False)
            continue
        for tag, marker in (("3m", "o"), ("6m", "s"), ("1y", "^")):
            xs, ys = [], []
            for row in results.rows(t):
                if not row.startswith(tag):
                    continue
                try:
                    share, _ = results.value(t, row, "regret share")
                except KeyError:
                    continue
                lvl = float(row.split()[-1].rstrip("%")) / 100.0
                xs.append(abs(math.log(lvl)))
                ys.append(share)
            ax.plot(xs, ys, marker=marker, lw=0.8, label=tag)
        ax.set_xlabel("|ln(B/S)|", fontsize=7)
        ax.set_ylabel("regret share (EKO - UOC)/EKO", fontsize=7)
        ax.set_title("up-and-out call" if side == "up" else "down-and-out put", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
    return fig


def _draw_map(results: Results) -> Figure:
    fig, axes = new_figure(1, 2, height=2.6)
    for ax, side in zip(np.atleast_1d(axes).ravel(), ("up", "down"), strict=False):
        t = f"map_{side}"
        if t not in results.tables():
            ax.set_visible(False)
            continue
        df = _table_frame(results, t)
        pts = df[df["column"] == "fly over uoc"]
        for tau in sorted({float(a["tau"]) for a in pts["axes"].map(_axes)}):
            sub = [
                (float(_axes(a)["spot_rel"]), v)
                for a, v in zip(pts["axes"], pts["value"], strict=True)
                if float(_axes(a)["tau"]) == tau
            ]
            sub.sort()
            ax.plot(
                [s for s, _ in sub],
                [v for _, v in sub],
                marker="o",
                lw=0.8,
                label=f"tau {tau:.3g}y",
            )
        ax.axhline(1.0, lw=0.6, color="0.5")
        ax.set_xlabel("spot / inception spot", fontsize=7)
        ax.set_ylabel("fly price / knock-out price", fontsize=7)
        ax.set_yscale("log")
        ax.set_title("6m 110% call" if side == "up" else "6m 90% put", fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
    return fig


def _axes(a: Any) -> dict[str, Any]:
    import json

    if isinstance(a, dict):
        return a
    try:
        return dict(json.loads(a)) if isinstance(a, str) and a else {}
    except ValueError:
        return {}


def _draw_daily(results: Results) -> Figure:
    prods = [p for p in DAILY_PRODUCTS if f"daily_{slug(p)}" in results.tables()]
    fig, axes = new_figure(len(prods), 1, height=1.9 * max(len(prods), 1))
    for ax, p in zip(np.atleast_1d(axes).ravel(), prods, strict=False):
        t = f"daily_{slug(p)}"
        df = _table_frame(results, t)
        for col, color in (("regret_share", "C0"), ("iv_minus_rv_3m", "C3")):
            s = df[df["column"] == col].sort_values("row")
            if s.empty:
                continue
            x = pd.to_datetime(s["row"])
            ax.plot(x, s["value"], lw=0.8, color=color, label=col.replace("_", " "))
        ax.set_title(p, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6, loc="upper left")
    return fig


def _draw_pq(results: Results) -> Figure:
    fig, ax = new_figure(height=2.8)
    labels, qs, hs, fs = [], [], [], []
    for tag in ("3m", "6m", "1y"):
        t = f"history_{tag}"
        if t not in results.tables():
            continue
        df = _table_frame(results, t)
        touched = df[df["column"] == "touched"]
        by = {str(r["row"]): float(r["value"]) for _, r in touched.iterrows()}
        for row in sorted({r.split(" | ")[0] for r in by}):
            q = by.get(f"{row} | 2F mark (Q)")
            h = by.get(f"{row} | history")
            f = by.get(f"{row} | FHS at implied ATM")
            if q is None:
                continue
            labels.append(row)
            qs.append(q)
            hs.append(h if h is not None else np.nan)
            fs.append(f if f is not None else np.nan)
    x = np.arange(len(labels))
    ax.bar(x - 0.25, qs, 0.25, label="2F mark (Q)")
    ax.bar(x, hs, 0.25, label="history 1990-2026")
    ax.bar(x + 0.25, fs, 0.25, label="FHS at implied ATM (demeaned)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=5)
    ax.set_ylabel("P(daily close touch)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6)
    return fig


def figures(results: Results) -> list[FigureSpec]:
    out = [
        FigureSpec(
            "regret_share",
            "The regret share (EKO - UOC)/EKO under the 2F mark against the log distance to the barrier, per maturity.",
            _draw_regret,
        ),
        FigureSpec(
            "touch_p_vs_q",
            "Touch probabilities: the 2F mark against the 1990-2026 history and the demeaned filtered historical simulation at the implied ATM vol.",
            _draw_pq,
        ),
    ]
    if any(t.startswith("map_") for t in results.tables()):
        out.append(
            FigureSpec(
                "map_fly_ratio",
                "The fly-to-knock-out price ratio across spot and time to expiry (2F mark): the switch point of a holder.",
                _draw_map,
            )
        )
    if any(t.startswith("daily_uoc") or t.startswith("daily_dop") for t in results.tables()):
        out.append(
            FigureSpec(
                "daily_series",
                "2022 H2 day by day: the regret share under each day's local vol and the implied-minus-realised 3m vol.",
                _draw_daily,
            )
        )
    return out


# --------------------------------------------------------------------------------------------
# narrative
# --------------------------------------------------------------------------------------------


def _fmt(v: float, se: float = float("nan"), digits: int = 2) -> str:
    return f"{v:.{digits}f}" if not math.isfinite(se) else f"{v:.{digits}f} +/- {se:.{digits}f}"


def _share_range(results: Results, side: str) -> tuple[str, str] | None:
    t = f"decomposition_{side}"
    if t not in results.tables():
        return None
    vals = []
    for row in results.rows(t):
        try:
            v, _ = results.value(t, row, "regret share")
        except KeyError:
            continue
        vals.append((v, row))
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    return f"{100 * lo[0]:.0f}% ({lo[1]})", f"{100 * hi[0]:.0f}% ({hi[1]})"


def _verdicts(results: Results) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    if "decision" not in results.tables():
        return out
    df = _table_frame(results, "decision")
    v = df[df["column"] == "verdict"]
    for _, r in v.iterrows():
        layer = str(r["row"]).split(" | ")[1]
        out.setdefault(layer, {"barrier": 0, "fly": 0, "indifferent": 0})
        key = "barrier" if r["value"] == 1.0 else ("fly" if r["value"] == 0.0 else "indifferent")
        out[layer][key] += 1
    return out


def _gap_summary(results: Results, layer: str, side: str) -> tuple[float, float, float] | None:
    """Mean touch gap P − Q, the ratio of the summed P and Q touch probabilities and the mean
    regret gap of ``side``'s barriers under ``layer``."""
    if "decision" not in results.tables():
        return None
    df = _table_frame(results, "decision")
    sum_p, sum_q, gaps, rgaps = 0.0, 0.0, [], []
    for row in results.rows("decision"):
        head, _, row_layer = row.partition(" | ")
        if row_layer != layer or f" {side} " not in head:
            continue
        sub = df[df["row"] == row]
        pq = {str(r["column"]): float(r["value"]) for _, r in sub.iterrows()}
        q = pq.get("touch gap P-Q")
        if q is None:
            continue
        tag, _, lvl = row.split(" | ")[0].split()
        try:
            pt, _ = results.value(f"decomposition_{side}", f"{tag} {side} {lvl}", "p touch")
        except KeyError:
            continue
        sum_q += pt
        sum_p += pt + q
        gaps.append(q)
        rgaps.append(pq.get("regret gap P-Q", float("nan")))
    if not gaps or sum_q <= 0:
        return None
    return float(np.mean(gaps)), sum_p / sum_q, float(np.nanmean(rgaps))


def _hedge_line(results: Results, product: str) -> str | None:
    if "hedge" not in results.tables():
        return None
    df = _table_frame(results, "hedge")
    std = df[df["column"] == "std zc"]
    rows = {
        str(r["row"]): float(r["value"])
        for _, r in std.iterrows()
        if str(r["row"]).startswith(product + " |")
    }
    if not rows:
        return None
    best = min(rows.items(), key=lambda kv: kv[1])
    delta = next((v for k, v in rows.items() if "| delta | 2F" in k and k.count("|") == 2), None)
    fly = next((v for k, v in rows.items() if "fly proxy static | 2F" in k), None)
    prod = next(
        (
            float(r["value"])
            for _, r in df[
                (df["column"] == "product std") & (df["row"].str.startswith(product + " |"))
            ].iterrows()
        ),
        float("nan"),
    )
    parts = [f"`{product}`: unhedged std {prod:.2f}% of spot"]
    if delta is not None:
        parts.append(f"delta alone {delta:.2f}")
    if fly is not None:
        parts.append(f"the premium-matched fly held short alone {fly:.2f}")
    parts.append(
        f"best `{best[0].split(' | ')[1]}` ({best[0].split(' | ')[2]} world) {best[1]:.2f}"
    )
    return "; ".join(parts) + "."


def _daily_line(results: Results) -> str | None:
    if "daily_summary" not in results.tables():
        return None
    rows = results.rows("daily_summary")
    cs, ci = [], []
    for r in rows:
        try:
            cs.append(results.value("daily_summary", r, "corr regret-distance")[0])
            ci.append(results.value("daily_summary", r, "corr regret-ivrv")[0])
        except KeyError:
            continue
    if not cs:
        return None
    return (
        f"Over the 127 days the regret share moves with the absolute distance to the barrier in standard "
        f"deviations (correlation {min(cs):.2f} to {max(cs):.2f} across the {len(cs)} products) and "
        f"with implied-minus-realised 3m vol ({min(ci):.2f} to {max(ci):.2f}) - both through the implied "
        f"vol level, which sets the distance: a higher implied vol brings the barrier closer and raises "
        f"the share of the European value the path condition removes."
    )


def narrative(results: Results) -> str:
    have = set(results.tables())
    lines = [
        "The theory, the metrics and the decision rule are in `docs/barrier_vs_vanilla.md`; this "
        "report reads the measured tables. Everything is on the SPX 2022-12-30 desk mark (2F LSV, "
        "step 0 from the snapshot's SABRW fits, ssr 1, eps 0.10), strike 100%, notional 1/spot, "
        "barriers observed on daily closes (strict) with a continuous twin, every Monte Carlo "
        "number with its standard error; the vanilla structures are priced off the surface.",
        "",
        "## 1. The decomposition on the anchor",
        "",
    ]
    for side in ("up", "down"):
        if f"decomposition_{side}" in have:
            lines += [f"{{{{table:decomposition_{side}}}}}", ""]
    rng_up, rng_dn = _share_range(results, "up"), _share_range(results, "down")
    if rng_up and rng_dn:
        lines.append(
            f"**Regret share.** On the call side the barrier removes from {rng_up[0]} to {rng_up[1]} "
            f"of the European knock-out's value; on the put side from {rng_dn[0]} to {rng_dn[1]}. "
            "The share falls with the distance and rises with the maturity (figure `regret_share`): "
            "a near barrier on a long expiry leaves the holder almost nothing of the European contract."
        )
    if "decomposition_up" in have:
        try:
            v2, se2 = results.value("decomposition_up", "6m up 110%", "2F")
            lv, _ = results.value("decomposition_up", "6m up 110%", "LV")
            bs, _ = results.value("decomposition_up", "6m up 110%", "BS")
            ek, _ = results.value("decomposition_up", "6m up 110%", "EKO surface")
            fl, _ = results.value("decomposition_up", "6m up 110%", "fly")
            top, _ = results.value("decomposition_up", "6m up 110%", "fly matched top")
            rm, _ = results.value("decomposition_up", "6m up 110%", "ratio matched")
            cont, _ = results.value("decomposition_up", "6m up 110%", "2F cont")
            d2, _ = results.value("decomposition_down", "6m down 90%", "2F")
            dlv, _ = results.value("decomposition_down", "6m down 90%", "LV")
            dbs, _ = results.value("decomposition_down", "6m down 90%", "BS")
            lines.append(
                f" The 6m 110% up-and-out call is worth {_fmt(v2, se2, 3)}% of spot under the 2F mark "
                f"({lv:.3f}% under local vol, {bs:.3f}% under Black-Scholes at the ATM vol, {cont:.3f}% "
                f"with continuous observation) against a model-free European knock-out of {ek:.3f}%: "
                f"the premium-matched symmetric fly tops out at {100 * top:.1f}% of spot (the midpoint "
                f"fly to the barrier costs {fl:.3f}%, {fl / v2:.1f} times the knock-out, and no fly on the "
                f"barrier's own strikes can be as cheap) and the premium-matched call ratio is 1x{rm:.2f}. "
                f"The model risk has opposite signs on the two sides: the 2F mark prices the up-and-out "
                f"call above local vol and Black-Scholes, the 6m 90% down-and-out put below them "
                f"({d2:.3f}% against {dlv:.3f}% and {dbs:.3f}%) - a stochastic-vol world reaches the up "
                f"barrier in a low-vol state (fewer returns) and the down barrier in a high-vol one."
            )
        except KeyError:
            pass
    lines += ["", "{{figure:regret_share}}", ""]
    if any(t.startswith("map_") for t in have):
        lines += ["## 2. When: the spot x time map", ""]
        lines += ["{{table:map_up}}", ""] if "map_up" in have else []
        lines += ["{{table:map_down}}", ""] if "map_down" in have else []
        lines += [
            "The fly-to-knock-out price ratio rises as the spot approaches the barrier and as the "
            "expiry shortens (figure `map_fly_ratio`): below one the knock-out is the dearer contract "
            "(far from the barrier, little time), above one the fly - the switch point of a holder whose "
            "view is unchanged.",
            "",
            "{{figure:map_fly_ratio}}",
            "",
        ]
    if any(t.startswith("greeks_") for t in have):
        lines += ["## 3. Greeks of the representatives", ""]
        for t in sorted(x for x in have if x.startswith("greeks_")):
            lines += [f"{{{{table:{t}}}}}", ""]
    if "hedge" in have:
        lines += ["## 4. Hedging cost", "", "{{table:hedge}}", ""]
        for p in (
            "uoc 6m 110",
            "dop 6m 90",
            "fly 6m 110",
            "pfly 6m 90",
            "ratio 6m 110",
            "pratio 6m 90",
        ):
            hl = _hedge_line(results, p)
            if hl:
                lines.append("* " + hl)
        lines.append("")
    if "daily_summary" in have:
        lines += ["## 5. 2022 H2 day by day", "", "{{table:daily_summary}}", ""]
        dl = _daily_line(results)
        if dl:
            lines += [dl, ""]
        if "daily_outcomes" in have:
            lines += [
                "{{table:daily_outcomes}}",
                "",
                "The realised outcomes of 2022 H2 are one draw of the path distribution (the rally of 2023): they illustrate the regret paths - the entries that touched the barrier and finished in the money - and are not a statistic.",
                "",
            ]
        lines += ["{{figure:daily_series}}", ""]
    lines += ["## 6. The statistical layer and the decision", ""]
    for tag in ("3m", "6m", "1y"):
        if f"history_{tag}" in have:
            lines += [f"{{{{table:history_{tag}}}}}", ""]
    lines += ["{{figure:touch_p_vs_q}}", ""]
    for layer in ("history", "FHS at implied ATM"):
        parts = []
        for side in ("up", "down"):
            g = _gap_summary(results, layer, side)
            if g:
                parts.append(
                    f"{'call' if side == 'up' else 'put'} side: touch probability P minus Q "
                    f"{g[0]:+.3f} on average over the nine barriers (summed P over summed Q "
                    f"{100 * g[1]:.0f}%), regret gap {g[2]:+.3f}"
                )
        if parts:
            lines.append(f"**{layer}.** " + "; ".join(parts) + ".")
    lines.append("")
    if "decision" in have:
        lines += ["{{table:decision}}", ""]
        vd = _verdicts(results)
        if vd:
            parts = [
                f"{layer}: barrier {c['barrier']}, fly {c['fly']}, indifferent {c['indifferent']}"
                for layer, c in vd.items()
            ]
            lines.append("**Verdicts per layer (18 barriers each).** " + "; ".join(parts) + ".")
            lines.append("")
    lines += [
        "**Reading.** The knock-out is the right trade when the mark's touch probability exceeds the statistical one by more than its model band and hedging cost - far barriers, short expiries, implied vol above realised; the fly when the barrier is near in standard deviations or the expiry long (the regret share dominates), or when the mark's own models disagree by more than the premium saved; the ratio only for a buyer who takes the unbounded upside loss, whose size the history and FHS tables give through the ratio's payoff per premium. On this anchor the drift-free statistical layers favour the fly on the call side and the knock-out on the put side: the mark's down-and-out puts are cheap against a drift-free path distribution (the skew prices the down touch above its statistical frequency), its up-and-out calls are not.",
    ]
    return "\n".join(x for x in lines if x is not None)


__all__ = [
    "OPTIONAL_PARAMS",
    "QUESTION",
    "REQUIRED_PARAMS",
    "TITLE",
    "compute",
    "figures",
    "narrative",
    "requirements",
    "tables",
    "validate_params",
]
