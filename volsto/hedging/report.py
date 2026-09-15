"""Hedging reports (SPEC §8, M8 Part 4): the P&L distribution with standard errors, regime
breakdowns, the attribution of the hedged P&L and the residual-exposure report, in the study
table format with an Excel export.

* **Distribution**: mean (with its standard error), std (its standard error from the fourth
  moment), quantiles 1 / 5 / 50 / 95 / 99% (standard errors by a 200-draw bootstrap with a
  fixed seed), the worst ``n_worst`` paths with their realised vol, spot return, factor state at
  the end and termination date — for the costed and the zero-cost totals.
* **Regimes**: by realised-vol tercile of the world paths, by the realised skew-proxy change
  (the world's factor-state move when the world has factors, else the realised-vol change
  between the two halves of the life, noted), by early termination / barrier event.
* **Attribution** (aggregated over the life, per path then averaged, with standard errors):
  the product's revaluation explained by the regressed Greeks and the world's moves between
  consecutive dates — ``delta × ΔS``, ``½ gamma × ΔS²``, ``Σ_i ∂V/∂X_i × ΔX_i`` (the factor
  channel, which carries vega / vol-of-vol moves when the world has the pricing model's factor
  structure) — the hedge legs, costs, recalibration and the residual (unexplained, which
  includes the surface-level moves the desk cannot observe in a model world: vega waves, skew,
  curvature — reported as such).
* **Residual exposure** per date from Layer A (mean |residual| per target against the unhedged
  exposure).

Checked by ``tests/test_hedging.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.hedging.hedger import HedgeResult

FloatArray = NDArray[np.float64]
QUANTILES = (0.01, 0.05, 0.50, 0.95, 0.99)
BOOTSTRAP = 200
BOOTSTRAP_SEED = 2718


def _se_mean(x: FloatArray) -> float:
    return float(np.std(x, ddof=1) / np.sqrt(x.size)) if x.size > 1 else float("nan")


def _se_std(x: FloatArray) -> float:
    n = x.size
    if n < 4:
        return float("nan")
    m4 = float(np.mean((x - x.mean()) ** 4))
    s2 = float(np.var(x, ddof=1))
    return float(np.sqrt(max(m4 - s2 * s2, 0.0) / n) / (2.0 * np.sqrt(s2))) if s2 > 0 else 0.0


def _quantiles_with_se(x: FloatArray, qs: tuple[float, ...] = QUANTILES) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    base = np.quantile(x, qs)
    boots = np.empty((BOOTSTRAP, len(qs)))
    for b in range(BOOTSTRAP):
        boots[b] = np.quantile(rng.choice(x, size=x.size, replace=True), qs)
    return pd.DataFrame({"quantile": qs, "value": base, "stderr": boots.std(axis=0, ddof=1)})


def distribution_table(x: FloatArray, label: str) -> pd.DataFrame:
    rows = [
        {"series": label, "statistic": "mean", "value": float(np.mean(x)), "stderr": _se_mean(x)},
        {
            "series": label,
            "statistic": "std",
            "value": float(np.std(x, ddof=1)),
            "stderr": _se_std(x),
        },
    ]
    for rec in _quantiles_with_se(x).to_dict(orient="records"):
        rows.append(
            {
                "series": label,
                "statistic": f"q{round(float(rec['quantile']) * 100):02d}",
                "value": float(rec["value"]),
                "stderr": float(rec["stderr"]),
            }
        )
    return pd.DataFrame(rows)


def realised_vol(result: HedgeResult) -> FloatArray:
    """Annualised realised vol of each world path over the life (from the recorded columns)."""
    w = result.world_paths
    ls = w.log_spot
    T = float(w.times[-1] - w.times[0])
    r2 = np.sum(np.diff(ls, axis=1) ** 2, axis=1)
    return np.asarray(np.sqrt(r2 / max(T, 1e-12)), dtype=np.float64)


def skew_proxy_change(result: HedgeResult) -> tuple[FloatArray, str]:
    """The world's skew-proxy change over the life: the first factor's move when the world has
    factors (the model's skew-driving state), else the realised-vol change between the two halves
    of the life (noted)."""
    w = result.world_paths
    if w.n_factors >= 1:
        f = w.factors[:, :, 0]
        return np.asarray(f[:, -1] - f[:, 0]), "world factor X1 move over the life"
    ls = w.log_spot
    n = ls.shape[1]
    h = n // 2
    a = np.sum(np.diff(ls[:, : h + 1], axis=1) ** 2, axis=1)
    b = np.sum(np.diff(ls[:, h:], axis=1) ** 2, axis=1)
    return (
        np.asarray(np.sqrt(b) - np.sqrt(a)),
        "realised-vol change between the two halves (no world factors)",
    )


def regime_table(result: HedgeResult) -> pd.DataFrame:
    x = result.pnl_total
    rv = realised_vol(result)
    rows = []
    edges = np.quantile(rv, [1 / 3, 2 / 3])
    labels = np.where(rv <= edges[0], "low", np.where(rv <= edges[1], "mid", "high"))
    for lab in ("low", "mid", "high"):
        sel = labels == lab
        rows.append(
            {
                "regime": f"realised vol {lab}",
                "n": int(sel.sum()),
                "mean": float(x[sel].mean()),
                "stderr": _se_mean(x[sel]),
                "std": float(x[sel].std(ddof=1)) if sel.sum() > 1 else float("nan"),
                "realised_vol_mean": float(rv[sel].mean()),
            }
        )
    sk, note = skew_proxy_change(result)
    e2 = np.quantile(sk, [1 / 3, 2 / 3])
    lab2 = np.where(sk <= e2[0], "down", np.where(sk <= e2[1], "flat", "up"))
    for lab in ("down", "flat", "up"):
        sel = lab2 == lab
        rows.append(
            {
                "regime": f"skew proxy {lab} ({note})",
                "n": int(sel.sum()),
                "mean": float(x[sel].mean()),
                "stderr": _se_mean(x[sel]),
                "std": float(x[sel].std(ddof=1)) if sel.sum() > 1 else float("nan"),
                "realised_vol_mean": float(rv[sel].mean()),
            }
        )
    term = np.isfinite(result.termination)
    for lab, sel in (("early termination", term), ("no termination", ~term)):
        if sel.any():
            rows.append(
                {
                    "regime": lab,
                    "n": int(sel.sum()),
                    "mean": float(x[sel].mean()),
                    "stderr": _se_mean(x[sel]),
                    "std": float(x[sel].std(ddof=1)) if sel.sum() > 1 else float("nan"),
                    "realised_vol_mean": float(rv[sel].mean()),
                }
            )
    return pd.DataFrame(rows)


def worst_paths(result: HedgeResult, n: int = 10) -> pd.DataFrame:
    x = result.pnl_total
    order = np.argsort(x)[:n]
    rv = realised_vol(result)
    w = result.world_paths
    ret = np.exp(w.log_spot[:, -1] - w.log_spot[:, 0]) - 1.0
    rows = []
    for i in order:
        row: dict[str, Any] = {
            "path": int(i),
            "pnl": float(x[i]),
            "pnl_product": float(result.pnl_product[i]),
            "pnl_hedges": float(result.pnl_hedges[i].sum()),
            "costs": float(result.costs[i]),
            "realised_vol": float(rv[i]),
            "spot_return": float(ret[i]),
            "termination": float(result.termination[i]),
        }
        for f in range(w.n_factors):
            row[f"X{f + 1}_end"] = float(w.factors[i, -1, f])
        rows.append(row)
    return pd.DataFrame(rows)


def attribution_table(result: HedgeResult) -> pd.DataFrame:
    """The life-aggregated attribution (module docstring) with standard errors."""
    g = result.greeks_by_date
    m = result.moves_by_date
    n = result.n_paths
    rows = []
    explained = np.zeros(n)
    if "delta" in g and "dS" in m:
        d = np.sum(g["delta"] * m["dS"], axis=0)
        rows.append(
            {"component": "product delta x dS", "mean": float(d.mean()), "stderr": _se_mean(d)}
        )
        explained += d
    if "gamma" in g and "dS" in m:
        d = np.sum(0.5 * g["gamma"] * m["dS"] ** 2, axis=0)
        rows.append(
            {
                "component": "product 1/2 gamma x dS^2",
                "mean": float(d.mean()),
                "stderr": _se_mean(d),
            }
        )
        explained += d
    for k in ("dX1", "dX2"):
        if k in g and k in m:
            d = np.sum(g[k] * m[k], axis=0)
            rows.append(
                {
                    "component": f"product dV/{k} x {k} (factor channel)",
                    "mean": float(d.mean()),
                    "stderr": _se_mean(d),
                }
            )
            explained += d
    resid = result.pnl_product - explained
    rows.append(
        {
            "component": "product residual (higher order, unobserved surface moves)",
            "mean": float(resid.mean()),
            "stderr": _se_mean(resid),
        }
    )
    rows.append(
        {
            "component": "product total",
            "mean": float(result.pnl_product.mean()),
            "stderr": _se_mean(result.pnl_product),
        }
    )
    for j, name in enumerate(result.instruments):
        h = result.pnl_hedges[:, j]
        rows.append({"component": f"hedge {name}", "mean": float(h.mean()), "stderr": _se_mean(h)})
    rows.append(
        {
            "component": "recalibration",
            "mean": float(result.pnl_recalibration.mean()),
            "stderr": _se_mean(result.pnl_recalibration),
        }
    )
    rows.append(
        {
            "component": "costs (-)",
            "mean": -float(result.costs.mean()),
            "stderr": _se_mean(result.costs),
        }
    )
    rows.append(
        {
            "component": "total",
            "mean": float(result.pnl_total.mean()),
            "stderr": _se_mean(result.pnl_total),
        }
    )
    return pd.DataFrame(rows)


@dataclass
class HedgeReport:
    """The tables of one run (module docstring)."""

    result: HedgeResult
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        r = self.result
        self.tables["distribution"] = pd.concat(
            [
                distribution_table(r.pnl_total, "with costs"),
                distribution_table(r.pnl_zero_cost, "zero cost"),
            ],
            ignore_index=True,
        )
        self.tables["regimes"] = regime_table(r)
        self.tables["worst_paths"] = worst_paths(r)
        self.tables["attribution"] = attribution_table(r)
        self.tables["residual_exposure"] = r.residual
        self.tables["by_date"] = r.by_date
        self.tables["quantities"] = r.quantities
        if len(r.recalibrations):
            self.tables["recalibrations"] = r.recalibrations
        self.tables["settings"] = pd.DataFrame(
            [{"key": k, "value": str(v)} for k, v in {**r.settings, **r.budget}.items()]
        )

    def summary(self) -> str:
        r = self.result
        d = self.tables["distribution"]
        lines = [
            r.summary(),
            "P&L distribution:",
            d.round(6).to_string(index=False),
            "regimes:",
            self.tables["regimes"].round(6).to_string(index=False),
            "attribution:",
            self.tables["attribution"].round(6).to_string(index=False),
        ]
        res = self.tables["residual_exposure"]
        if len(res):
            cols = [c for c in res.columns if c.startswith(("residual:", "exposure:"))]
            lines += [
                "residual exposure (mean over dates of the mean |.| over alive paths):",
                res[cols].mean().round(6).to_string(),
            ]
        if r.pricing_notes:
            lines.append("notes: " + "; ".join(r.pricing_notes))
        return "\n".join(lines)

    def to_excel(self, path: str | Path) -> Path:
        path = Path(path)
        with pd.ExcelWriter(path) as xw:
            for name, table in self.tables.items():
                table.to_excel(xw, sheet_name=name[:31], index=False)
        return path


def hedge_report(result: HedgeResult) -> HedgeReport:
    return HedgeReport(result)


__all__ = [
    "BOOTSTRAP",
    "QUANTILES",
    "HedgeReport",
    "attribution_table",
    "distribution_table",
    "hedge_report",
    "realised_vol",
    "regime_table",
    "skew_proxy_change",
    "worst_paths",
]
