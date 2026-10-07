"""The dispersion study, stage 2 — the report (runner module of
``configs/studies/dispersion/study.yaml``; the owner's question of 2026-10-03).  Reads the tables
``scripts/dispersion_study.py`` writes under ``<outputs>/<params.dir>/``
(:mod:`volsto.studies.dispersion`); never prices.  The theory is ``docs/dispersion_palladium.md``.

Results tables (per unit of basket notional, performances in fractions of the initial level):
``setup``; ``market`` (the trades under the market world: Monte Carlo and Gaussian);
``sens_<axis>`` (the trades across each parameter axis); ``expectations_<trade>`` (the P&L per
unit premium over the realised-correlation × realised-vol grid, the other departures at zero);
``departures`` (the same at the implied correlation and vols across the local-correlation,
correlation-uncertainty and jump departures); ``best`` (the best trade per cell, coded);
``history`` (the realised statistics and proxy P&Ls by regime).

Checked by ``tests/test_dispersion.py``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from volsto.studies.dispersion import TRADES
from volsto.studies.results import Column, FigureSpec, Results, ResultsBuilder, TableSpec
from volsto.studies.runner import Requirement, StudyContext, artefact_path
from volsto.studies.style import new_figure

TITLE = "Palladium (the call on dispersion) against single-name straddles and the basket straddle"
QUESTION = (
    "Given the market's parameters (single-name implied vols, implied correlation) and a view on "
    "their realised counterparts, when is the call on dispersion the better trade than the "
    "single-name straddles against the basket straddle?"
)
REQUIRED_PARAMS: tuple[str, ...] = ("dir",)
OPTIONAL_PARAMS: tuple[str, ...] = ()
COMMAND = ".venv/bin/python scripts/dispersion_study.py --parts {part}"
FILES: dict[str, tuple[str, ...]] = {
    "sensitivities": ("sensitivities.csv", "setup.csv"),
    "expectations": ("expectations.csv",),
    "history": ("history_summary.csv",),
}
UNIT = "fraction of basket notional"
AXES: tuple[str, ...] = (
    "rho",
    "vol_scale",
    "local_corr",
    "corr_sd",
    "jump_prob",
    "skew",
    "T",
    "n",
    "strike_fraction",
)
AXIS_TITLES: dict[str, str] = {
    "rho": "the correlation",
    "vol_scale": "the vol level (all names scaled)",
    "local_corr": "the local-correlation slope lambda (correlation rises when the basket falls)",
    "corr_sd": "the correlation uncertainty (the path's correlation drawn from rho +/- sd)",
    "jump_prob": "the idiosyncratic-event probability per name (jump sd 10%)",
    "skew": "the per-name skew (SSVI rho of each name's smile; local-vol names)",
    "T": "the horizon in years",
    "n": "the basket size (the first n names)",
    "strike_fraction": "the call on dispersion's strike as a fraction of the forward dispersion",
}
TRADE_CODES: dict[str, float] = {t: float(i) for i, t in enumerate(TRADES)}
SHORT: dict[str, str] = {
    "palladium": "palladium",
    "palladium call": "pall. call",
    "basket straddle": "basket strdl",
    "single straddles": "singles",
    "straddle package": "package",
    "straddle package (premium-neutral)": "package pn",
    "variance dispersion": "var disp",
}


def slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_")


def validate_params(params: Mapping[str, Any]) -> None:
    if not isinstance(params["dir"], str) or not params["dir"]:
        raise ValueError("dir must be the stage-1 output directory relative to outputs")


def requirements(ctx: StudyContext) -> list[Requirement]:
    out = []
    for part, files in FILES.items():
        for f in files:
            out.append(
                ctx.artefact_requirement(
                    f"{ctx.params['dir']}/{f}", f"dispersion {part}", COMMAND.format(part=part)
                )
            )
    return out


def _read(ctx: StudyContext, name: str) -> pd.DataFrame:
    return pd.read_csv(artefact_path(ctx.outputs_root, f"{ctx.params['dir']}/{name}"))


def _num(
    b: ResultsBuilder,
    table: str,
    row: str,
    col: str,
    v: Any,
    se: Any = float("nan"),
    *,
    unit: str = UNIT,
    source: str,
    **kw: Any,
) -> None:
    v = float(v)
    if not math.isfinite(v):
        return
    se = float(se) if se is not None else float("nan")
    if math.isfinite(se) and se > 0:
        # a Monte Carlo value needs a unit: ratios and probabilities carry '1'
        b.add(table, row, col, v, se, unit=unit or "1", source=source, **kw)
    else:
        b.add_exact(table, row, col, v, unit=unit or "1", source=source, **kw)


def compute(ctx: StudyContext) -> Results:
    b = ResultsBuilder()
    d = ctx.params["dir"]
    src = {
        k: f"artefact:{d}/{k}.csv"
        for k in ("sensitivities", "expectations", "history_summary", "setup")
    }
    setup = _read(ctx, "setup.csv")
    for k, v in zip(setup["key"], setup["value"], strict=True):
        try:
            x = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(x):
            b.add_exact("setup", str(k), "value", x, unit="", source=src["setup"])
    sens = _read(ctx, "sensitivities.csv")
    base = sens[sens["axis"] == "base"]
    for _, r in base.iterrows():
        _num(
            b,
            "market",
            str(r["trade"]),
            "Monte Carlo",
            r["value"],
            r["stderr"],
            source=src["sensitivities"],
        )
        _num(b, "market", str(r["trade"]), "Gaussian", r["gaussian"], source=src["sensitivities"])
    for axis in AXES:
        sub = sens[sens["axis"] == axis]
        if sub.empty:
            continue
        t = f"sens_{axis}"
        for _, r in sub.iterrows():
            col = f"{float(r['x']):g}"
            _num(
                b,
                t,
                str(r["trade"]),
                col,
                r["value"],
                r["stderr"],
                source=src["sensitivities"],
                axes={"x": float(r["x"]), "axis": axis},
            )
        if axis in ("rho", "vol_scale", "T", "n", "strike_fraction"):
            for _, r in sub.iterrows():
                _num(
                    b,
                    f"{t}_gaussian",
                    str(r["trade"]),
                    f"{float(r['x']):g}",
                    r["gaussian"],
                    source=src["sensitivities"],
                    axes={"x": float(r["x"]), "axis": axis},
                )
    exp = _read(ctx, "expectations.csv")
    plain = exp[(exp["local_corr"] == 0) & (exp["corr_sd"] == 0) & (exp["jump_prob"] == 0)]
    for trade in TRADES:
        t = f"expectations_{slug(trade)}"
        sub = plain[plain["trade"] == trade]
        for _, r in sub.iterrows():
            row = f"vols x{float(r['vol_scale']):g}"
            col = f"rho {float(r['d_rho']):+.1f}"
            _num(
                b,
                t,
                row,
                col,
                r["pnl_per_premium"] if float(r["premium"]) > 1e-9 else r["pnl_mean"],
                r["pnl_stderr"] / float(r["premium"])
                if float(r["premium"]) > 1e-9
                else r["pnl_stderr"],
                unit=("per unit premium" if float(r["premium"]) > 1e-9 else UNIT),
                source=src["expectations"],
                axes={"d_rho": float(r["d_rho"]), "vol_scale": float(r["vol_scale"])},
            )
    # departures at the implied parameters (d_rho 0, vols x1)
    dep = exp[(exp["d_rho"] == 0) & (exp["vol_scale"] == 1.0)]
    for _, r in dep.iterrows():
        row = f"lambda {float(r['local_corr']):g} | corr sd {float(r['corr_sd']):g} | jumps {float(r['jump_prob']):g}"
        prem = float(r["premium"])
        _num(
            b,
            "departures",
            row,
            str(r["trade"]),
            r["pnl_per_premium"] if prem > 1e-9 else r["pnl_mean"],
            (r["pnl_stderr"] / prem) if prem > 1e-9 else r["pnl_stderr"],
            unit=("per unit premium" if prem > 1e-9 else UNIT),
            source=src["expectations"],
        )
    # the best trade per cell (coded)
    firsts = exp.drop_duplicates(
        subset=["d_rho", "vol_scale", "local_corr", "corr_sd", "jump_prob"]
    )
    for _, r in firsts.iterrows():
        row = f"rho {float(r['d_rho']):+.1f} | vols x{float(r['vol_scale']):g} | lambda {float(r['local_corr']):g} | sd {float(r['corr_sd']):g} | jumps {float(r['jump_prob']):g}"
        b.add_exact(
            "best",
            row,
            "best per premium",
            TRADE_CODES[str(r["best_per_premium"])],
            unit="trade code",
            source=src["expectations"],
            note=str(r["best_per_premium"]),
            axes={
                "d_rho": float(r["d_rho"]),
                "vol_scale": float(r["vol_scale"]),
                "local_corr": float(r["local_corr"]),
                "corr_sd": float(r["corr_sd"]),
                "jump_prob": float(r["jump_prob"]),
            },
        )
        b.add_exact(
            "best",
            row,
            "best mean",
            TRADE_CODES[str(r["best_mean"])],
            unit="trade code",
            source=src["expectations"],
            note=str(r["best_mean"]),
        )
    hs = _read(ctx, "history_summary.csv")
    for _, r in hs.iterrows():
        row = f"{r['regime']} {r['group']}" if r["regime"] != "all" else "all"
        for col in (
            "dispersion",
            "straddle_package",
            "basket_abs",
            "singles_abs",
            "corr_realised",
            "corr_implied",
            "corr_premium",
            "vol_mean",
            "vol_basket",
            *tuple(f"pnl {t}" for t in TRADES),
        ):
            if col in r:
                unit = "" if col.startswith(("corr", "vol")) else UNIT
                _num(
                    b,
                    "history",
                    row,
                    col,
                    r[col],
                    r.get(f"{col}_stderr", float("nan")),
                    unit=unit,
                    source=src["history_summary"],
                )
        _num(b, "history", row, "n eff", r["n_eff"], unit="", source=src["history_summary"])
    return b.build()


def _cols(results: Results, table: str) -> tuple[Column, ...]:
    return tuple(Column(c, c) for c in results.columns(table))


def tables(results: Results) -> list[TableSpec]:
    have = set(results.tables())
    specs = [
        TableSpec(
            "market",
            "The trades under the market world (3m horizon, ten names, equal weights; vols = trailing 1y realised x 1.15, correlation = Cboe COR3M at the anchor): Monte Carlo value with its stderr and the Gaussian closed form, per unit of basket notional (the palladium call struck at 80% of the forward dispersion; the premium-neutral package scaled so it costs nothing).",
            "market",
            _cols(results, "market"),
            row_header="trade",
        )
    ]
    for axis in AXES:
        t = f"sens_{axis}"
        if t in have:
            specs.append(
                TableSpec(
                    t,
                    f"The trades across {AXIS_TITLES[axis]} (Monte Carlo; per unit of basket notional).",
                    t,
                    _cols(results, t),
                    row_header="trade",
                )
            )
    for trade in TRADES:
        t = f"expectations_{slug(trade)}"
        if t in have:
            specs.append(
                TableSpec(
                    t,
                    f"`{trade}` bought at the market world's price and realised under a world whose correlation is the implied one shifted by the column and whose vols are the implied ones scaled by the row (no local correlation, no correlation uncertainty, no jumps): expected P&L per unit premium (per unit notional for the zero-premium package).",
                    t,
                    _cols(results, t),
                    row_header="realised vols",
                )
            )
    if "departures" in have:
        specs.append(
            TableSpec(
                "departures",
                "At the implied correlation and vols, the trades' expected P&L per unit premium under the departures: the local-correlation slope, the correlation uncertainty and the idiosyncratic events.",
                "departures",
                _cols(results, "departures"),
                row_header="world",
            )
        )
    if "history" in have:
        specs.append(
            TableSpec(
                "history",
                "The ten-name basket 2006-2026 per 3m window, overall, by VIX tercile and by the basket's move: realised dispersion, the straddle package payoff, the basket's and names' absolute moves, the realised and Cboe-implied correlation and their gap, the mean single-name and basket vols, and the proxy ex-post P&L of each trade (premium = Gaussian value at COR3M and the trailing realised vols x 1.15; block stderrs).",
                "history",
                _cols(results, "history"),
                row_header="regime",
            )
        )
    return specs


def _frame(results: Results, table: str) -> pd.DataFrame:
    f = results.frame
    return f[f["table"] == table]


def _draw_rho(results: Results) -> Figure:
    fig, ax = new_figure(height=2.8)
    if "sens_rho" in results.tables():
        df = _frame(results, "sens_rho")
        for trade in (
            "palladium",
            "palladium call",
            "basket straddle",
            "straddle package",
            "straddle package (premium-neutral)",
        ):
            sub = df[df["row"] == trade].copy()
            if sub.empty:
                continue
            sub["x"] = sub["column"].astype(float)
            sub = sub.sort_values("x")
            ax.plot(sub["x"], sub["value"], marker="o", lw=0.8, label=trade)
        ax.axhline(0.0, lw=0.5, color="0.5")
        ax.set_xlabel("correlation", fontsize=7)
        ax.set_ylabel("value per unit basket notional", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
    return fig


def _draw_best(results: Results) -> Figure:
    fig, axes = new_figure(1, 2, height=2.8)
    axs = np.atleast_1d(axes).ravel()
    if "best" not in results.tables():
        return fig
    df = _frame(results, "best")
    df = df[df["column"] == "best per premium"]
    import json

    recs = []
    for _, r in df.iterrows():
        a = json.loads(r["axes"]) if isinstance(r["axes"], str) and r["axes"] else {}
        recs.append({**a, "code": float(r["value"])})
    g = pd.DataFrame(recs)
    for ax, (title, mask) in zip(
        axs,
        (
            ("no departures", (g["local_corr"] == 0) & (g["corr_sd"] == 0) & (g["jump_prob"] == 0)),
            (
                "jumps p 0.2, corr sd 0.2",
                (g["local_corr"] == 0) & (g["corr_sd"] > 0) & (g["jump_prob"] > 0),
            ),
        ),
        strict=False,
    ):
        sub = g[mask]
        piv = sub.pivot_table(index="vol_scale", columns="d_rho", values="code")
        im = ax.imshow(piv.to_numpy(), cmap="tab10", vmin=0, vmax=len(TRADES) - 1, aspect="auto")
        ax.set_xticks(range(len(piv.columns)))
        ax.set_xticklabels([f"{c:+.1f}" for c in piv.columns], fontsize=6)
        ax.set_yticks(range(len(piv.index)))
        ax.set_yticklabels([f"x{i:g}" for i in piv.index], fontsize=6)
        ax.set_xlabel("realised - implied correlation", fontsize=7)
        ax.set_ylabel("realised / implied vols", fontsize=7)
        ax.set_title(title, fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                code = piv.to_numpy()[i, j]
                if math.isfinite(code):
                    ax.text(
                        j,
                        i,
                        SHORT[TRADES[int(code)]],
                        ha="center",
                        va="center",
                        fontsize=5,
                        color="w",
                    )
        del im
    return fig


def _draw_history(results: Results) -> Figure:
    fig, ax = new_figure(height=2.8)
    if "history" in results.tables():
        df = _frame(results, "history")
        rows = [r for r in results.rows("history") if r.startswith("basket_move")]
        vals = []
        for r in rows:
            d, _ = results.value("history", r, "dispersion")
            s, _ = results.value("history", r, "straddle_package")
            vals.append((r.replace("basket_move ", ""), d, s))
        x = np.arange(len(vals))
        ax.bar(x - 0.2, [v[1] for v in vals], 0.4, label="dispersion D")
        ax.bar(x + 0.2, [v[2] for v in vals], 0.4, label="straddle package")
        ax.set_xticks(x)
        ax.set_xticklabels([v[0] for v in vals], fontsize=6)
        ax.set_ylabel("3m realised, fraction", fontsize=7)
        ax.set_xlabel("basket 3m move", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        del df
    return fig


def figures(results: Results) -> list[FigureSpec]:
    return [
        FigureSpec("sens_rho", "The trades against the correlation (market vols, 3m).", _draw_rho),
        FigureSpec(
            "best_trade",
            "The best trade per unit premium over the realised-correlation x realised-vol grid, without departures and with idiosyncratic events and correlation uncertainty.",
            _draw_best,
        ),
        FigureSpec(
            "history_moves",
            "Realised dispersion against the straddle package payoff by the basket's 3m move, 2006-2026.",
            _draw_history,
        ),
    ]


def _v(results: Results, table: str, row: str, col: str) -> float:
    try:
        return results.value(table, row, col)[0]
    except KeyError:
        return float("nan")


def narrative(results: Results) -> str:
    have = set(results.tables())
    lines = [
        "The theory and the decision table are in `docs/dispersion_palladium.md`; this report reads the measured tables. The basket is ten large caps (AAPL, MSFT, AMZN, NVDA, JPM, XOM, JNJ, PG, HD, UNH), equal weights, 3m horizon; the market world's single-name vols are the trailing 1y realised vols at 2022-12-30 times 1.15 (an input: single-name implied vols are not in the repository) and its correlation the Cboe COR3M of that date.",
        "",
        "## 1. The market world and the parameters",
        "",
        "{{table:market}}",
        "",
    ]
    pf, pk, bs, sp = (
        _v(results, "market", t, "Monte Carlo")
        for t in ("palladium", "palladium call", "basket straddle", "straddle package")
    )
    if all(math.isfinite(x) for x in (pf, pk, bs, sp)):
        lines.append(
            f"The palladium forward is worth {pf:.4f} per unit of basket notional, {pf / sp:.1f} times the straddle package ({sp:.4f}) it dominates path by path; the basket straddle {bs:.4f}; the call at 80% of the forward dispersion {pk:.4f}."
        )
        lines.append("")
    for axis in AXES:
        if f"sens_{axis}" in have:
            lines += [f"{{{{table:sens_{axis}}}}}", ""]
    lines += ["{{figure:sens_rho}}", "", "## 2. The framework: expectations against the market", ""]
    for trade in TRADES:
        t = f"expectations_{slug(trade)}"
        if t in have:
            lines += [f"{{{{table:{t}}}}}", ""]
    if "departures" in have:
        lines += ["{{table:departures}}", ""]
    lines += ["{{figure:best_trade}}", ""]
    # count the best trades
    if "best" in have:
        df = _frame(results, "best")
        df = df[df["column"] == "best per premium"]
        counts = df["note"].value_counts().to_dict()
        lines.append(
            "**Best trade per unit premium over the 120 cells.** "
            + "; ".join(f"{k}: {v}" for k, v in counts.items())
            + "."
        )
        lines.append("")
    dep: dict[str, dict[str, float]] = {}
    if "departures" in have:
        for row in results.rows("departures"):
            dep[row] = {c: _v(results, "departures", row, c) for c in results.columns("departures")}
    base_row = "lambda 0 | corr sd 0 | jumps 0"
    if base_row in dep:
        jm = dep.get("lambda 0 | corr sd 0 | jumps 0.2", {})
        lc = dep.get("lambda 3 | corr sd 0 | jumps 0", {})
        sd = dep.get("lambda 0 | corr sd 0.2 | jumps 0", {})
        nan = float("nan")
        lines.append(
            "**What the departures do at the implied parameters** (expected P&L per unit premium). "
            f"Idiosyncratic events (each name jumping with probability 0.2 by a 10% shock): palladium call {jm.get('palladium call', nan):+.2f}, straddle package {jm.get('straddle package', nan):+.2f}, basket straddle {jm.get('basket straddle', nan):+.2f}. "
            f"A local correlation that rises when the basket falls (lambda 3): palladium call {lc.get('palladium call', nan):+.2f}, straddle package {lc.get('straddle package', nan):+.2f}, basket straddle {lc.get('basket straddle', nan):+.2f} - the correlation falls in rallies, which is where the call on dispersion collects. "
            f"An uncertain realised correlation (+/- 0.2): palladium forward {sd.get('palladium', nan):+.2f} (concave in the correlation), palladium call {sd.get('palladium call', nan):+.2f}, straddle package {sd.get('straddle package', nan):+.2f}."
        )
        lines.append("")
    if "sens_rho" in have:
        slopes: dict[str, float] = {}
        for t in ("palladium", "palladium call", "straddle package", "basket straddle"):
            vals = sorted(
                (float(c), _v(results, "sens_rho", t, c)) for c in results.columns("sens_rho")
            )
            if len(vals) >= 2:
                slopes[t] = (vals[-1][1] - vals[0][1]) / (vals[-1][0] - vals[0][0])
        base = {t: _v(results, "market", t, "Monte Carlo") for t in slopes}
        parts = [
            f"{t} {slopes[t]:+.3f} per unit of correlation ({0.1 * slopes[t] / base[t]:+.0%} of its premium per 0.1)"
            for t in slopes
            if base.get(t, 0) > 0
        ]
        if parts:
            lines.append(
                "**Correlation sensitivity (market vols, 3m).** "
                + "; ".join(parts)
                + ": per unit premium the straddle package is the sharpest correlation instrument, the call on dispersion next, the forward the dullest."
            )
            lines.append("")
    lines += ["## 3. The history", "", "{{table:history}}", "", "{{figure:history_moves}}", ""]
    cr, ci = (
        _v(results, "history", "all", "corr_realised"),
        _v(results, "history", "all", "corr_implied"),
    )
    if math.isfinite(cr) and math.isfinite(ci):
        lines.append(
            f"Over 2006-2026 the Cboe-implied 3m correlation averaged {100 * ci:.0f}% against a realised {100 * cr:.0f}%: the correlation risk premium the straddle package carries. The palladium-to-package payoff ratio is largest when the basket is flat and smallest in large moves of either sign: the triangle gap closes when everything moves together."
        )
    return "\n".join(lines)


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
