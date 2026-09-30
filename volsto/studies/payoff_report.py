"""The payoff study, stage 2 — the report (runner module of ``configs/studies/payoff/payoff.yaml``;
owner's request of 2026-09-27).  Reads the tables that ``scripts/payoff_study.py`` writes under
``<outputs>/<params.dir>/`` (:mod:`volsto.studies.payoff`) and never prices or calibrates; a
missing table prints the command that produces it and the run exits 2.

Results tables: ``setup``; ``prices`` (BS reference, LV, 1F, 2F per product, ×100 in the
product's unit); ``model_risk`` (the LV / 1F / 2F spread and 2F − LV, stderrs in quadrature: the
models' path sets are independent); ``greeks_<family>`` (the 2F risk report of the family's
representative, recalibrate mode — P1 held, leverage recalibrated — ×100 in the product's unit
per bump unit); ``dials`` (the price at each marking-dial mark minus the reference mark's, stderr
in quadrature: a bound, the marks share the seed and their correlation is not measured);
``rotation`` (the rotation greek ``(V(+1) − V(−1))/2`` and the convexity, P1 held);
``hedge_<family>`` (per strategy / world / delta regime: the hedged P&L std, mean, 1% quantile
and 1% expected shortfall with and without costs, and the std over the unhedged product's,
stderrs from the fourth moment and the bootstrap of
:func:`volsto.hedging.report.distribution_table`).

The narrative states, per family, the theoretical decomposition of the risks and the hedges it
suggests, then reads the measured comparison: the best hedge by P&L std without costs and by 1%
expected shortfall with costs, and the reserve (the mean P&L in the pure-LV world with the 2F
pricing model).  Checked by ``tests/test_payoff_study.py``.
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

TITLE = "The payoff study: reverse barriers, variance options, KO vars and corridors on SPX"
QUESTION = (
    "On the SPX 2022-12-30 desk mark, what are the study's products worth under LV / 1F / 2F, "
    "which risks do they carry, how do the marking dials and the skew rotation move them, and "
    "which hedge works best?"
)
REQUIRED_PARAMS: tuple[str, ...] = ("dir",)
OPTIONAL_PARAMS: tuple[str, ...] = ()
COMMAND = ".venv/bin/python scripts/payoff_study.py --parts {part}"
PART_FILES: dict[str, tuple[str, ...]] = {
    "prices": ("prices.csv", "model_risk.csv", "setup.csv"),
    "greeks": ("greeks.csv",),
    "dials": ("dials.csv",),
    "rotation": ("rotation.csv", "rotation_greek.csv"),
    "hedge": ("hedge.csv",),
}
FAMILIES: tuple[str, ...] = (
    "uoc",
    "dop",
    "dip",
    "put on var",
    "ko var",
    "up var",
    "down var",
    "vko",
)
FAMILY_TITLES: dict[str, str] = {
    "uoc": "Up-and-out call",
    "dop": "Down-and-out put",
    "dip": "Down-and-in put",
    "put on var": "Put on realised variance",
    "ko var": "Knock-out variance swap",
    "up var": "Up variance swap",
    "down var": "Down variance swap",
    "vko": "VKO put",
}
QUAD_NOTE = "stderr in quadrature (independent path sets)"
DIAL_NOTE = (
    "stderr in quadrature: a bound only if the marks' correlation is non-negative (they share "
    "the seed; the correlation is not measured)"
)
REFERENCE_MARK = (1.0, 0.10)


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_")


def validate_params(params: Mapping[str, Any]) -> None:
    if not isinstance(params["dir"], str) or not params["dir"]:
        raise ValueError("dir must be the stage-1 output directory relative to outputs")


def requirements(ctx: StudyContext) -> list[Requirement]:
    out = []
    for part, files in PART_FILES.items():
        for f in files:
            out.append(
                ctx.artefact_requirement(
                    f"{ctx.params['dir']}/{f}", f"payoff study {part}", COMMAND.format(part=part)
                )
            )
    return out


def _read(ctx: StudyContext, name: str) -> pd.DataFrame:
    return pd.read_csv(artefact_path(ctx.outputs_root, f"{ctx.params['dir']}/{name}"))


def _mc(
    b: ResultsBuilder,
    table: str,
    row: str,
    col: str,
    v: Any,
    se: Any,
    unit: str,
    *,
    source: str,
    **kw: Any,
) -> None:
    v, se = float(v), float(se)
    if not math.isfinite(v):
        return
    if not math.isfinite(se):
        b.add(
            table, row, col, v, None, exact=True, unit=unit, source=source, note="no stderr", **kw
        )
        return
    b.add(table, row, col, v, se, unit=unit, source=source, **kw)


UNIT_TAGS: dict[str, str] = {"% of spot": "pct", "vol pts (vega notional 1)": "vp"}
UNIT_TITLES: dict[str, str] = {"pct": "% of spot", "vp": "vol points of vega notional 1"}


def _tag(unit: str) -> str:
    return UNIT_TAGS.get(unit, slug(unit))


def compute(ctx: StudyContext) -> Results:
    b = ResultsBuilder()
    prices = _read(ctx, "prices.csv")
    mr = _read(ctx, "model_risk.csv")
    greeks = _read(ctx, "greeks.csv")
    dials = _read(ctx, "dials.csv")
    rot = _read(ctx, "rotation_greek.csv")
    hedge = _read(ctx, "hedge.csv")
    setup = _read(ctx, "setup.csv")
    fam = dict(zip(prices["product"], prices["family"], strict=False))
    d = ctx.params["dir"]
    sv = {str(k): float(v) for k, v in zip(setup["key"], setup["value"], strict=True)}
    spot = sv["spot"]
    for k, v in sv.items():
        if math.isfinite(v):
            b.add_exact("setup", k, "value", v, unit="", source=f"artefact:{d}/setup.csv")
    src = {k: f"artefact:{d}/{k}.csv" for k in ("prices", "model_risk", "greeks", "dials")}
    src |= {"rotation": f"artefact:{d}/rotation_greek.csv", "hedge": f"artefact:{d}/hedge.csv"}
    b.add_exact(
        "setup",
        "products",
        "value",
        float(prices["product"].nunique()),
        unit="",
        source=src["prices"],
    )
    b.add_exact(
        "setup",
        "price paths",
        "value",
        float(prices["n_paths"].max()),
        unit="",
        source=src["prices"],
    )
    b.add_exact("setup", "hedge runs", "value", float(len(hedge)), unit="", source=src["hedge"])
    for _, r in prices.iterrows():
        u = str(r["unit"])
        _mc(
            b,
            f"prices_{_tag(u)}",
            str(r["product"]),
            str(r["model"]),
            r["value"],
            r["stderr"],
            u,
            source=src["prices"],
            axes={"family": fam.get(r["product"], "")},
        )
    for _, r in mr.iterrows():
        p, u = str(r["product"]), str(r["unit"])
        t = f"model_risk_{_tag(u)}"
        sm = src["model_risk"]
        _mc(b, t, p, "2F", r["2F"], r["2F_stderr"], u, source=sm)
        _mc(b, t, p, "spread", r["spread"], r["spread_stderr"], u, source=sm, note=QUAD_NOTE)
        _mc(
            b,
            t,
            p,
            "2F - LV",
            r["2F_minus_LV"],
            r["2F_minus_LV_stderr"],
            u,
            source=sm,
            note=QUAD_NOTE,
        )
    for prod_name, g in greeks.groupby("product", sort=False):
        p = str(prod_name)
        f = slug(str(g["family"].iloc[0]))
        scale = float(g["scale"].iloc[0])
        unit = str(g["unit"].iloc[0])
        for _, r in g.iterrows():
            grp, name = str(r.get("group", "")), str(r.get("name", ""))
            tags = [
                str(r[k])
                for k in ("regime", "variant", "key")
                if k in r and isinstance(r[k], str) and r[k]
            ]
            row = f"{grp}: {name}" + (f" ({', '.join(tags)})" if tags else "")
            if grp == "product":
                _mc(
                    b,
                    "ko_probability",
                    str(p),
                    name,
                    float(r["value"]),
                    float(r["stderr"]),
                    "probability per unit ln S",
                    source=src["greeks"],
                )
                continue
            # the engine's delta / gamma are per unit (per unit²) of the index: shown per 1% spot
            # move — delta × 1% S, gamma × (1% S)² (the change of the 1% delta over a 1% move)
            conv = 1.0
            if grp == "delta":
                conv, row = 0.01 * spot, f"{row} per 1% spot"
            elif grp == "gamma":
                conv, row = (0.01 * spot) ** 2, f"{row} per (1% spot)^2"
            _mc(
                b,
                f"greeks_{f}",
                row,
                "value",
                conv * scale * float(r["value"]),
                conv * scale * float(r["stderr"]),
                unit,
                source=src["greeks"],
                axes={"product": str(p)},
            )
    ok = dials[dials["status"] != "infeasible"] if "status" in dials else dials
    ref = ok[(ok["ssr_target"] == REFERENCE_MARK[0]) & (ok["skew_eps"] == REFERENCE_MARK[1])]
    ref_v = {str(r["product"]): (float(r["value"]), float(r["stderr"])) for _, r in ref.iterrows()}
    for _, r in ok.iterrows():
        mark = (float(r["ssr_target"]), float(r["skew_eps"]))
        name = str(r["product"])
        if mark == REFERENCE_MARK or name not in ref_v:
            continue
        v0, s0 = ref_v[name]
        u = str(r["unit"])
        _mc(
            b,
            f"dials_{_tag(u)}",
            name,
            f"ssr {mark[0]:g} eps {mark[1]:g}",
            float(r["value"]) - v0,
            float(np.hypot(float(r["stderr"]), s0)),
            u,
            source=src["dials"],
            note=DIAL_NOTE,
        )
    for _, r in rot.iterrows():
        u = str(r["unit"])
        t = f"rotation_{_tag(u)}"
        _mc(
            b,
            t,
            str(r["product"]),
            "rotation greek",
            r["rotation_greek"],
            r["stderr_bound"],
            u,
            source=src["rotation"],
            note=QUAD_NOTE,
        )
        if math.isfinite(float(r["convexity"])):
            b.add(
                t,
                str(r["product"]),
                "convexity",
                float(r["convexity"]),
                None,
                exact=True,
                unit=u,
                source=src["rotation"],
                note="second difference at ±2 around 0, no stderr stored",
            )
    for _, r in hedge.iterrows():
        f = slug(fam.get(str(r["product"]), str(r["product"])))
        label = str(r["strategy"])
        if str(r["world"]) != "2F":
            label += f" [{r['world']} world]"
        if str(r["regime"]) != "model":
            label = f"delta ({r['regime']})"
        u = str(r["unit"])
        for stat in ("std", "mean", "q01", "es01"):
            for tg in ("zc", "tc"):
                _mc(
                    b,
                    f"hedge_{f}",
                    label,
                    f"{stat}_{tg}",
                    r[f"{stat}_{tg}"],
                    r[f"{stat}_{tg}_stderr"],
                    u,
                    source=src["hedge"],
                    axes={
                        "product": str(r["product"]),
                        "world": str(r["world"]),
                        "regime": str(r["regime"]),
                    },
                )
        b.add(
            f"hedge_{f}",
            label,
            "product std",
            float(r["product_std"]),
            None,
            exact=True,
            unit=u,
            source=src["hedge"],
            note="the unhedged product's P&L std on the same world paths",
        )
    return b.build()


def tables(results: Results) -> list[TableSpec]:
    specs: list[TableSpec] = []
    have = set(results.tables())
    for tag, title in UNIT_TITLES.items():
        if f"prices_{tag}" in have:
            specs.append(
                TableSpec(
                    f"prices_{tag}",
                    f"Prices x100 in {title}: Black-Scholes at the ATM vol of the maturity, local "
                    "vol, the reference 1F LSV and the 2F desk mark; one path set per model.",
                    f"prices_{tag}",
                    tuple(
                        Column(c, c.replace(" (ATM vol)", ""))
                        for c in results.columns(f"prices_{tag}")
                    ),
                    row_header="product",
                )
            )
        if f"model_risk_{tag}" in have:
            specs.append(
                TableSpec(
                    f"model_risk_{tag}",
                    f"Model risk x100 in {title}: the 2F price, the spread across LV / 1F / 2F "
                    f"and 2F - LV ({QUAD_NOTE}).",
                    f"model_risk_{tag}",
                    (Column("2F", "2F"), Column("spread", "spread"), Column("2F - LV", "2F - LV")),
                    row_header="product",
                )
            )
    for f in FAMILIES:
        t = f"greeks_{slug(f)}"
        if t in have:
            specs.append(
                TableSpec(
                    t,
                    f"{FAMILY_TITLES[f]}: the 2F risk report of the representative (recalibrate "
                    "mode: P1 held, leverage recalibrated after the bump), x100 in the product's "
                    "unit per bump unit; the barrier rows per 1% of the barrier (vol point of the "
                    "vol barrier for the VKO), the barrier-shift rows the price change with the "
                    "barrier moved away from the spot.",
                    t,
                    (Column("value", "value"),),
                    row_header="sensitivity",
                )
            )
    if "ko_probability" in have:
        specs.append(
            TableSpec(
                "ko_probability",
                "The knock-out probability's sensitivity to ln S under the model regime.",
                "ko_probability",
                tuple(Column(c, c) for c in results.columns("ko_probability")),
                row_header="product",
            )
        )
    for tag, title in UNIT_TITLES.items():
        if f"dials_{tag}" in have:
            specs.append(
                TableSpec(
                    f"dials_{tag}",
                    "Marking dials: the price at each (ssr_target, skew_eps) mark minus the "
                    f"reference mark (1.0, 0.10)'s, x100 in {title} ({DIAL_NOTE}).",
                    f"dials_{tag}",
                    tuple(Column(c, c) for c in results.columns(f"dials_{tag}")),
                    row_header="product",
                )
            )
        if f"rotation_{tag}" in have:
            specs.append(
                TableSpec(
                    f"rotation_{tag}",
                    "Shadow rotation with the P1 parameters held and the leverage recalibrated: "
                    "the rotation greek (V(+1) - V(-1))/2 and the convexity V(2) - 2V(0) + V(-2), "
                    f"x100 in {title}.",
                    f"rotation_{tag}",
                    tuple(Column(c, c) for c in results.columns(f"rotation_{tag}")),
                    row_header="product",
                )
            )
    for f in FAMILIES:
        t = f"hedge_{slug(f)}"
        if t in have:
            specs.append(
                TableSpec(
                    t,
                    f"{FAMILY_TITLES[f]}: hedged P&L per strategy (2F pricing model, daily "
                    "rebalancing; the 2F world unless marked), x100 in the product's unit: std, "
                    "mean and 1% expected shortfall, without (zc) and with (tc) half-spreads of "
                    "1 bp on the spot and 0.25 vol point on options and swaps.",
                    t,
                    (
                        Column("std_zc", "std zc"),
                        Column("std_tc", "std tc"),
                        Column("mean_zc", "mean zc"),
                        Column("mean_tc", "mean tc"),
                        Column("es01_zc", "ES1% zc"),
                        Column("es01_tc", "ES1% tc"),
                        Column("product std", "unhedged std"),
                    ),
                    row_header="strategy",
                )
            )
    return specs


def _hedge_ratios(results: Results, table: str) -> list[tuple[str, float]]:
    out = []
    for row in results.rows(table):
        s, _ = results.value(table, row, "std_zc")
        p, _ = results.value(table, row, "product std")
        if "world]" in row or not p:
            continue
        out.append((row, s / p))
    return out


def _draw_hedge(results: Results) -> Figure:
    fams = [f for f in FAMILIES if f"hedge_{slug(f)}" in results.tables()]
    ncol = 2
    nrow = max(1, math.ceil(len(fams) / ncol))
    fig, axes = new_figure(nrow, ncol, height=2.2 * nrow)
    axs = np.atleast_1d(axes).ravel()
    for ax, f in zip(axs, fams, strict=False):
        pairs = sorted(_hedge_ratios(results, f"hedge_{slug(f)}"), key=lambda x: x[1])
        ax.barh([p[0][:28] for p in pairs], [p[1] for p in pairs])
        ax.axvline(1.0, lw=0.8, color="0.4")
        ax.set_title(FAMILY_TITLES[f], fontsize=8)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("hedged std / unhedged std (2F world, zero cost)", fontsize=6)
    for ax in axs[len(fams) :]:
        ax.set_visible(False)
    return fig


def _draw_kovar(results: Results) -> Figure:
    fig, ax = new_figure()
    for tag in ("3m", "6m", "1y"):
        xs, ys, es = [], [], []
        for b in (101, 103, 105):
            row = f"ko var {tag} {b}"
            if row in results.rows("prices_vp"):
                v, se = results.value("prices_vp", row, "2F")
                xs.append(b)
                ys.append(v)
                es.append(se)
        if xs:
            ax.errorbar(xs, ys, yerr=es, marker="o", capsize=2, label=tag)
    ax.set_xlabel("barrier (% of spot)")
    ax.set_ylabel("2F value (vol pts, vega notional 1)")
    ax.legend(fontsize=7)
    return fig


def figures(results: Results) -> list[FigureSpec]:
    out = [
        FigureSpec(
            "hedge_std",
            "Hedged P&L std over the unhedged product's per strategy (2F world, zero cost).",
            _draw_hedge,
        )
    ]
    if "prices_vp" in results.tables() and any(
        r.startswith("ko var") for r in results.rows("prices_vp")
    ):
        out.append(
            FigureSpec(
                "kovar_prices",
                "Knock-out variance swaps settled at the knock-out, struck at the VS vol: the 2F "
                "value against the barrier per maturity.",
                _draw_kovar,
            )
        )
    return out


THEORY: dict[str, str] = {
    "uoc": (
        "A reverse barrier: long the call spread up to the barrier, short the digital at the "
        "barrier. Its delta and gamma change sign near the barrier, the vega is negative close to "
        "it (more vol, more knock-outs) and the value is extremely sensitive to the barrier "
        "level and the monitoring (the barrier-shift reserve). The skew sets the price of the "
        "digital: the rotation greek and the dials measure it. Hedges: the put-call-symmetry "
        "replication (the call spread plus the reflected puts beyond the barrier, unwound at the "
        "knock-out), which is exact in a symmetric, driftless world, corrected here for the carry "
        "and the daily monitoring; the delta hedge; the gamma preset."
    ),
    "dop": (
        "The mirror of the up-and-out call on the put side: long the put spread down to the "
        "barrier, short the digital put there; the skew makes the knock-out region expensive. "
        "Hedges as for the up-and-out call, with the reflected calls below the barrier."
    ),
    "dip": (
        "Vanilla minus the down-and-out put (in-out parity): long vega and skew; at the barrier "
        "it becomes the vanilla. Hedges: short the parity vanilla plus the knock-out's "
        "replication held long and dropped at the knock-in."
    ),
    "put on var": (
        "Convex in realised variance: long vol of vol, short the variance level (its variance "
        "delta is negative). A variance swap sized on the vega hedges the level; the residual is "
        "the vol-of-vol, which only an instrument convex in variance (the var-vol swap spread, "
        "options on variance or VIX options) can offset."
    ),
    "ko var": (
        "A variance swap stopped at the first close above the barrier and settled there: before "
        "the knock-out it is long realised variance like the swap, and the knock-out ends the "
        "accrual on up moves (where the variance is lowest under the skew). Its delta comes from "
        "the knock-out probability and the fixed leg paid pro rata to the knock-out time. With "
        "barriers at 101-105% the knock-out is likely and the swap behaves like a short-dated "
        "down-variance. Hedges: the stopped log contract (the 2/K^2 strip below the barrier "
        "minus its reflection above), the variance swap on the vega, the delta."
    ),
    "up var": (
        "The variance accrued while the spot is above the level: long gamma on the upside only, "
        "its delta the change in the time expected above the level. Carr-Lewis: the 2/K^2 calls "
        "above the level plus a delta traded only inside the corridor replicate it."
    ),
    "down var": (
        "The mirror on the downside, where the skew makes the variance expensive: the 2/K^2 puts "
        "below the level plus the corridor delta."
    ),
    "vko": (
        "A put that dies when the realised variance exhausts its budget: long the put minus a "
        "vol-knock-in put; short vol of vol through the knock-in. Hedges: short the underlying "
        "put (static), the delta, the variance swap and the risk reversal of the preset."
    ),
}


def _rows_2f(results: Results, table: str) -> list[str]:
    return [r for r in results.rows(table) if "world]" not in r]


def _best(results: Results, table: str, col: str) -> tuple[str, float, float] | None:
    """The best 2F-world row by ``col`` (std: lowest; expected shortfall: least negative)."""
    best = None
    for row in _rows_2f(results, table):
        try:
            v, se = results.value(table, row, col)
        except KeyError:
            continue
        key = v if col.startswith("std") else -v
        if best is None or key < best[0]:
            best = (key, row, v, se)
    return None if best is None else (best[1], best[2], best[3])


def _ties(results: Results, table: str, col: str, best: tuple[str, float, float]) -> list[str]:
    """The other rows within 2 combined stderrs of the best."""
    out = []
    for row in _rows_2f(results, table):
        if row == best[0]:
            continue
        try:
            v, se = results.value(table, row, col)
        except KeyError:
            continue
        if abs(v - best[1]) <= 2.0 * math.hypot(se, best[2]):
            out.append(row)
    return out


def hedge_findings(results: Results, table: str) -> list[str]:
    """The measured comparison of one family, read from the results (narrative lines)."""
    lines: list[str] = []
    b1 = _best(results, table, "std_zc")
    if b1 is None:
        return lines
    unhedged = results.value(table, b1[0], "product std")[0]
    tied = _ties(results, table, "std_zc", b1)
    if b1[1] >= unhedged:
        lines.append(
            f"No strategy lowers the P&L std below the unhedged product's ({unhedged:.4g}); the "
            f"least bad is **{b1[0]}**, {b1[1]:.4g} ± {b1[2]:.2g}."
        )
    else:
        tie = f" (tied within 2 stderr: {', '.join(tied)})" if tied else ""
        lines.append(
            f"Lowest hedged P&L std without costs: **{b1[0]}**, {b1[1]:.4g} ± {b1[2]:.2g}, "
            f"{100 * (1 - b1[1] / unhedged):.0f}% below the unhedged {unhedged:.4g}{tie}."
        )
    b2 = _best(results, table, "es01_tc")
    if b2 is not None:
        lines.append(
            f"Best 1% expected shortfall with costs: **{b2[0]}**, {b2[1]:.4g} ± {b2[2]:.2g}."
        )
    worse = [
        f"{r} ({results.value(table, r, 'std_zc')[0]:.3g})"
        for r in _rows_2f(results, table)
        if results.value(table, r, "std_zc")[0] > unhedged
    ]
    if worse:
        lines.append(f"Worse than unhedged: {', '.join(worse)}.")
    lv = [r for r in results.rows(table) if r == "delta [LV world]"]
    if lv:
        m, se = results.value(table, lv[0], "mean_zc")
        lines.append(
            f"Reserve (the delta hedge's mean P&L in the pure-LV world, 2F pricing): {m:+.4g} ± "
            f"{se:.2g} — the model's mispricing realised by a hedged book when the world is local "
            "vol."
        )
    return lines


def narrative(results: Results) -> str:
    lines = [
        "The book is priced on the SPX 2022-12-30 snapshot under the desk's marking fit (step 0 "
        "from the snapshot's SABRW fits, ssr 1, eps 0.10) and hedged with the 2F LSV as the "
        "pricing model. Every number carries its Monte Carlo standard error; the knock-out "
        "variance swaps settle at the knock-out (owner, 2026-09-27) and every barrier is "
        "observed on daily closes.",
        "",
    ]
    have = set(results.tables())
    for tag in UNIT_TITLES:
        for base in ("prices", "model_risk"):
            if f"{base}_{tag}" in have:
                lines += [f"{{{{table:{base}_{tag}}}}}", ""]
    for f in FAMILIES:
        lines += [f"## {FAMILY_TITLES[f]}", "", THEORY[f], ""]
        g = f"greeks_{slug(f)}"
        if g in results.tables():
            lines += [f"{{{{table:{g}}}}}", ""]
        h = f"hedge_{slug(f)}"
        if h in results.tables():
            lines += [f"{{{{table:{h}}}}}", ""]
            lines += hedge_findings(results, h)
            lines.append("")
    if "ko_probability" in have:
        lines += ["{{table:ko_probability}}", ""]
    lines += ["## Marking dials and skew rotation", ""]
    for tag in UNIT_TITLES:
        for base in ("dials", "rotation"):
            if f"{base}_{tag}" in have:
                lines += [f"{{{{table:{base}_{tag}}}}}", ""]
    lines += ["{{figure:hedge_std}}", ""]
    return "\n".join(lines)


__all__ = [
    "FAMILIES",
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
