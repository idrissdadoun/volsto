"""Dispersion study, check C1 (spec §10): the forward rule and the vol convention.

    python scripts/disp_c1.py

On 20 random (ticker, entry date) pairs per asset class (single names, DJX, DIA), at the
listed expiry nearest three months: (i) for each forward candidate, the gap between the vol
implied by the call and by the put at the same strikes within ±10 % of the spot; (ii) the
study's at-the-money straddle against ``cValue + pValue`` at the nearest listed strike; (iii)
the study's vols minus ``smoothSmvVol`` at 1, 3, 6 and 12 months.  Writes
``outputs/dispersion/c1.json`` and prints the decision of §1.1.
"""

# ruff: noqa: E501, RUF001
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from volsto.market.bs import black_price, implied_vol
from volsto.studies import disp_data as dd
from volsto.studies import disp_smile as ds
from volsto.studies import disp_universe as du

CANDIDATES = ("rate", "rate_plus_residual", "dividends", "parity")


def call_put_gap(
    g: pd.DataFrame, spot: float, F: float, T: float, df: float
) -> tuple[float, float]:
    near = g[(g["strike"] / spot - 1.0).abs() <= 0.10]
    near = near[(near["cValue"] >= 0.05) & (near["pValue"] >= 0.05)]
    if len(near) < 2:
        return float("nan"), float("nan")
    K = near["strike"].to_numpy(float)
    vc = implied_vol(near["cValue"].to_numpy(float), F, K, T, 1.0, df)
    vp = implied_vol(near["pValue"].to_numpy(float), F, K, T, -1.0, df)
    gap = 100.0 * (vc - vp)
    gap = gap[np.isfinite(gap)]
    if gap.size == 0:
        return float("nan"), float("nan")
    return float(np.median(np.abs(gap))), float(np.median(gap))


def one(ticker: str, date: str) -> dict[str, float] | None:
    chain = dd.read_chains(date, [ticker]).get(ticker)
    if chain is None or chain.empty:
        return None
    spot = float(chain["stkPx"].iloc[0])
    day = pd.Timestamp(date)
    exps = sorted(chain["expirDate"].unique())
    days = np.array([(pd.Timestamp(e) - day).days for e in exps])
    ok = days >= 20
    if not ok.any():
        return None
    e = str(np.array(exps)[ok][int(np.argmin(np.abs(days[ok] - 91)))])
    g = chain[chain["expirDate"] == e].sort_values("strike").drop_duplicates("strike")
    T = (pd.Timestamp(e) - day).days / 365.0
    r = float(np.nanmedian(g["iRate"]))
    df = float(np.exp(-r * T))
    res = float(np.nanmedian(g["residualRateData"]))
    out: dict[str, float] = {"T": T, "residual": res, "rate": r}
    fwd = {
        "rate": spot * np.exp(r * T),
        "rate_plus_residual": spot * np.exp((r + (res if np.isfinite(res) else 0.0)) * T),
        "parity": ds.parity_forward(g, spot, df),
    }
    div = dd.projected_dividends(ticker, date)
    if div is not None:
        t, d = div
        inside = (t > 0) & (t <= T)
        fwd["dividends"] = spot * np.exp(r * T) - float(
            np.sum(d[inside] * np.exp(r * (T - t[inside])))
        )
    for name, F in fwd.items():
        a, s = call_put_gap(g, spot, float(F), T, df)
        out[f"gap_abs_{name}"], out[f"gap_signed_{name}"] = a, s
        out[f"carry_{name}"] = float(np.log(F / spot) / T)
    # (ii) the straddle at the nearest listed strike, with the study's convention under each rule
    i = int(np.argmin(np.abs(g["strike"].to_numpy(float) - spot)))
    K = float(g["strike"].iloc[i])
    listed = float(g["cValue"].iloc[i] + g["pValue"].iloc[i])
    for name, F in fwd.items():
        call = K >= F
        value = float(g["cValue"].iloc[i] if call else g["pValue"].iloc[i])
        v = float(implied_vol(value, F, K, T, 1.0 if call else -1.0, df))
        ours = float(black_price(F, K, T, v, 1.0, df) + black_price(F, K, T, v, -1.0, df))
        out[f"straddle_err_{name}"] = abs(ours / listed - 1.0) if listed > 0 else float("nan")
    return out


def vendor_gap(ticker: str, date: str, rule: str) -> dict[str, float]:
    chain = dd.read_chains(date, [ticker]).get(ticker)
    out: dict[str, float] = {}
    if chain is None or chain.empty:
        return out
    spot = float(chain["stkPx"].iloc[0])
    smiles = ds.expiry_smiles(
        chain,
        date,
        rule,
        dividends=dd.projected_dividends(ticker, date) if rule == "dividends" else None,
    )
    if not smiles:
        return out
    for label, T in (("1m", 30 / 365), ("3m", 91 / 365), ("6m", 182 / 365), ("12m", 1.0)):
        if smiles[-1].T < 0.8 * T:
            continue
        ours = ds.smile_at(smiles, spot, T).vol_at_strike(1.0)
        theirs = ds.smile_at(smiles, spot, T, vendor=True).vol_at_strike(1.0)
        out[label] = 100.0 * (ours - theirs)
    return out


def main() -> None:
    rng = np.random.default_rng(20261005)
    entries = dd.entry_dates(63)
    samples: dict[str, list[tuple[str, str]]] = {"single names": [], "DJX": [], "DIA": []}
    for d in rng.choice(entries, 20, replace=False):
        names = du.members_on(str(d))
        samples["single names"].append((str(rng.choice(names)), str(d)))
    for d in rng.choice(entries, 20, replace=False):
        samples["DJX"].append(("DJX", str(d)))
    for d in rng.choice(entries, 20, replace=False):
        samples["DIA"].append(("DIA", str(d)))
    report: dict[str, dict[str, float]] = {}
    rows = []
    for cls, pairs in samples.items():
        got = []
        for t, d in pairs:
            r = one(t, d)
            if r is not None:
                got.append({"class": cls, "ticker": t, "date": d, **r})
        frame = pd.DataFrame(got)
        rows.append(frame)
        report[cls] = {"n": len(frame)}
        for c in CANDIDATES:
            if f"gap_abs_{c}" in frame:
                report[cls][f"median |call − put vol| ({c}), vp"] = float(
                    frame[f"gap_abs_{c}"].median()
                )
                report[cls][f"median signed ({c}), vp"] = float(frame[f"gap_signed_{c}"].median())
                report[cls][f"median straddle error ({c})"] = float(
                    frame[f"straddle_err_{c}"].median()
                )
                report[cls][f"n ({c})"] = int(frame[f"gap_abs_{c}"].notna().sum())
    detail = pd.concat(rows, ignore_index=True)
    # the decision: for American legs the candidate with the smallest median gap (parity is not a candidate there)
    choice = {}
    for cls in ("single names", "DIA"):
        cands = {
            c: report[cls].get(f"median |call − put vol| ({c}), vp", np.inf)
            for c in ("rate", "rate_plus_residual", "dividends")
        }
        choice[cls] = min(cands, key=lambda c: cands[c])
    choice["DJX"] = "parity"
    for cls, pairs in samples.items():
        gaps = pd.DataFrame([vendor_gap(t, d, choice[cls]) for t, d in pairs])
        for label in ("1m", "3m", "6m", "12m"):
            if label in gaps:
                report[cls][f"our vol − smoothSmvVol at {label}, median vp"] = float(
                    gaps[label].median()
                )
    dd.OUT.mkdir(parents=True, exist_ok=True)
    detail.to_csv(dd.OUT / "c1_detail.csv", index=False)
    (dd.OUT / "c1.json").write_text(json.dumps({"choice": choice, "report": report}, indent=1))
    for cls, r in report.items():
        print(f"== {cls}")
        for k, v in r.items():
            print(f"   {k}: {v:.4f}" if isinstance(v, float) else f"   {k}: {v}")
    print("choice:", choice)


if __name__ == "__main__":
    main()
