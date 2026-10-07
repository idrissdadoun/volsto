"""Checks of addendum 2 (BARRIER_STUDY_ADDENDUM2 §10) on the new post-processing.

    python scripts/barrier_checks3.py [--report outputs/interview/report]

Reads ``positions_attrib.parquet``, ``turnover.parquet``, ``cells.parquet`` and the report's
``touch_frequency.csv``; writes ``outputs/interview/checks3.md`` (pass or fail per check, the
measured numbers beside).  No tolerance is tuned: each is the one the addendum states.
"""

# ruff: noqa: RUF001, E501
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barrier_attrib as att
import barrier_features as feat

from volsto.market.curves import ForwardCurve
from volsto.market.surface import ImpliedSurface
from volsto.studies import barrier_attrib as ba
from volsto.studies import barrier_history as bh

OUT = bh.OUT / "checks3.md"
FIVE = ["2008-10-06", "2012-06-04", "2017-03-06", "2020-03-02", "2024-01-02"]
_lines: list[str] = []


def say(text: str) -> None:
    print(text)
    _lines.append(text)


def verdict(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def check_carry_identity(a: pd.DataFrame) -> None:
    own = a[a["model"] != "lsv"]
    w = {
        c: own.pivot_table(index="cell", columns="position", values=c, aggfunc="first")
        for c in ("vega", "carry", "pnl_h", "pnl_hx")
    }
    dsig = own.groupby("cell")["dsig"].first().reindex(w["vega"].index)
    worst_c = worst_x = 0.0
    n = 0
    for name in ("B5_2", "B6"):
        pair_carry = (w["vega"][name] - w["vega"]["A1"]) * dsig
        e1 = (pair_carry - (w["carry"][name] - w["carry"]["A1"])).abs()
        pair_hx = (w["pnl_h"][name] - w["pnl_h"]["A1"]) - pair_carry
        e2 = (pair_hx - (w["pnl_hx"][name] - w["pnl_hx"]["A1"])).abs()
        worst_c, worst_x = max(worst_c, float(e1.max())), max(worst_x, float(e2.max()))
        n += int(e2.notna().sum())
    say(
        f"- **10.1 carry identity** ({n} pairs fly − KO and tight limit − KO): max |carry_pair − (carry_X − carry_KO)| = {worst_c:.1e}, "
        f"max |pnl_hx pair − (pnl_hx_X − pnl_hx_KO)| = {worst_x:.1e} (limit 1e-12): **{verdict(worst_c <= 1e-12 and worst_x <= 1e-12)}**"
    )


def check_zero_vega() -> None:
    cells = pd.read_parquet(bh.RESULTS / "cells.parquet")
    c = cells[
        cells["finished"].fillna(False).astype(bool)
        & cells["tau1"].isna()
        & cells["barrier"].str.startswith("s")
    ]
    c = c.sample(200, random_state=11).copy()  # the trades of check 11.4
    pos = pd.read_parquet(
        bh.RESULTS / "positions.parquet", columns=["cell", "position", "model", "pnl_h"]
    )
    pos = pos[pos["cell"].isin(set(c["cell"]))]
    for col in [x for x in c.columns if x.startswith("vega_") or x.endswith("_vega")]:
        c[col] = 0.0
    out = ba.attribute(pos, c, bh.load_ohlc()["close"])
    have = out["pnl_h"].notna() & out["rv_life"].notna()
    same = bool((out.loc[have, "pnl_hx"] == out.loc[have, "pnl_h"]).all())
    say(
        f"- **10.2 zero-vega control** (the 200 trades of check 11.4, {int(have.sum())} positions with a hedged P&L, every vega set to zero): "
        f"pnl_hx == pnl_h exactly on all of them: **{verdict(same and int(have.sum()) > 0)}**"
    )


def check_realised_vol() -> None:
    worst = 0.0
    for x, n in ((0.01, 21), (0.004, 63), (0.02, 126), (0.013, 15)):
        r = x * np.where(np.arange(n) % 2 == 0, 1.0, -1.0)
        closes = 100.0 * np.exp(np.concatenate(([0.0], np.cumsum(r))))
        worst = max(worst, abs(ba.realised_vol(closes) - x * np.sqrt(252.0)))
    say(
        f"- **10.3a realised vol, synthetic** (constant ±x daily log returns): max |rv − x·√252| = {worst:.1e} (limit 1e-10): **{verdict(worst <= 1e-10)}**"
    )
    # the existing trailing realised vols (features) where their window is exactly a trade's life
    close = bh.load_ohlc()["close"]
    days = list(close.index)
    idx = {d: i for i, d in enumerate(days)}
    cells = pd.read_parquet(
        bh.RESULTS / "cells.parquet", columns=["cell", "entry", "expiry", "months", "finished"]
    )
    cells = cells[cells["finished"].fillna(False).astype(bool)].drop_duplicates(
        ["entry", "expiry", "months"]
    )
    feats = pd.read_parquet(
        bh.RESULTS / "features.parquet", columns=["cell", "rv_1m", "rv_3m", "rv_6m", "rv_12m"]
    )
    feats["entry"] = feats["cell"].str[:10]
    feats = feats.drop_duplicates("entry").set_index("entry")
    diffs, same_est = [], []
    for r in cells.itertuples():
        i0, i1 = idx[r.entry], idx[r.expiry]
        n = feat.TRADING_DAYS[int(r.months)]
        if i1 - i0 != n or i1 + 1 >= len(days):
            continue
        nxt = days[i1 + 1]  # the features of this date use the n returns ending at the expiry close
        if nxt not in feats.index:
            continue
        old = float(feats.loc[nxt, f"rv_{int(r.months)}m"])
        mine = ba.life_realised_vol(close, r.entry, r.expiry)
        rets = np.diff(np.log(close.iloc[i0 : i1 + 1].to_numpy()))
        diffs.append(100.0 * abs(mine - old))
        same_est.append(100.0 * abs(float(np.std(rets, ddof=1) * np.sqrt(252.0)) - old))
    if diffs:
        d, s = np.array(diffs), np.array(same_est)
        say(
            f"- **10.3b realised vol, actual closes** ({d.size} lives whose closes are exactly the window of an existing trailing realised vol): "
            f"max |rv_life − existing| = {d.max():.3f} vol points, median {np.median(d):.3f}, {int((d > 0.1).sum())} above 0.1 (limit 0.1): **{verdict(bool(d.max() <= 0.1))}**. "
            f"The existing one removes the mean (ddof 1), rv_life does not (addendum 2 §2.1 and check 10.3a); with the existing estimator on the same closes the max difference is {s.max():.1e}"
        )
    else:
        say(
            "- **10.3b realised vol, actual closes**: no life coincides with a trailing window: not evaluated"
        )


def check_turnover(t: pd.DataFrame) -> None:
    worst = max(abs(ba.turnover([d] * n) - 2 * abs(d)) for d, n in ((0.3, 5), (-0.7, 40), (1.2, 1)))
    c = t[t["delta_constant"]]
    err = (
        float((c["turnover"] - 2.0 * c["delta_first"].abs()).abs().max())
        if len(c)
        else float("nan")
    )
    ok = worst <= 1e-12 and (not len(c) or err <= 1e-12)
    say(
        f"- **10.4 turnover** — synthetic constant deltas: max |turnover − 2|δ|| = {worst:.1e}; the {len(c)} stored hedges whose deltas are constant "
        f"({int((c['n_hedge'] == 1).sum())} of them held one interval): max error {err:.1e}: **{verdict(ok)}**"
    )


def check_touch(report: Path, sample: str) -> None:
    cells = pd.read_parquet(bh.RESULTS / "cells.parquet")
    c = cells[
        cells["finished"].fillna(False).astype(bool)
        & cells["barrier"].str.startswith("s")
        & ~cells["carried"].astype(bool)
    ]
    if sample == "strict":
        c = c[c["tolerances_ok"].astype(bool)]
    table = pd.read_csv(report / "tables" / "touch_frequency.csv")
    table = table[table["group"] == "all"]
    side = {"call": 1, "put": -1}
    worst, n = 0.0, 0
    for r in table.to_dict("records"):
        g = c[
            (c["months"] == int(str(r["maturity"])[:-1]))
            & (c["side"] == side[r["side"]])
            & (c["barrier"] == r["barrier"])
        ]
        shown = float(str(r["knocked daily"]).split(" ± ")[0])
        worst = max(worst, abs(shown - float(g["tau1"].notna().mean())))
        n += 1
    bad = int((c["tau1"].notna() & c["tau2"].isna()).sum())
    say(
        f"- **10.5 touch frequencies** ({sample} sample): the table's realised daily knock share against mean(tau1 is not null) on the same trades, {n} rows: "
        f"max difference {worst:.1e} (the table has three decimals); trades knocked on the daily rule and not on the continuous rule: {bad} of {len(c)}: "
        f"**{verdict(worst <= 5.1e-4 and bad == 0)}**"
    )


class _Flat(ImpliedSurface):
    def __init__(self) -> None:
        fc = ForwardCurve.flat(100.0, 0.03, 0.01)
        super().__init__(fc, fc.rate_curve, 10.0)

    def total_variance(self, k: Any, T: Any) -> Any:
        return 0.04 * np.asarray(T, dtype=np.float64) + 0.0 * np.asarray(k)


def check_tilt() -> None:
    base, T = _Flat(), 0.25
    F = float(base.forward(T))
    worst_atm = worst_b = 0.0
    for B, want in ((108.0, 0.01), (93.0, -0.01)):
        tilted = ba.TiltedSurface(base, float(np.log(B / F)))
        worst_atm = max(worst_atm, abs(float(tilted.implied_vol_k(0.0, T)) - 0.2))
        worst_b = max(worst_b, abs(float(tilted.implied_vol(B, T)) - 0.2 - want))
    ran = (bh.RESULTS / "skew_vega.parquet").exists()
    say(
        f"- **10.6 skew bump** — the tilt on a flat surface: at-the-money vol unchanged to {worst_atm:.1e}, vol at B off ±0.01 by {worst_b:.1e} (limit 1e-12): "
        f"**{verdict(worst_atm <= 1e-12 and worst_b <= 1e-12)}**; the bump itself (section 6, optional) {'has run' if ran else 'has not run'}"
    )


def check_repro(a: pd.DataFrame, t: pd.DataFrame) -> None:
    key = ["cell", "position", "model"]
    again = att.build_attrib(FIVE).sort_values(key).reset_index(drop=True)
    stored = a[a["cell"].str[:10].isin(FIVE)].sort_values(key).reset_index(drop=True)
    same_a = len(again) == len(stored) and again.equals(stored)
    with tempfile.TemporaryDirectory() as tmp:
        for e in FIVE:
            res = att.turnover_job((e, tmp))
            if "error" in res:
                say(f"  turnover re-run of {e} failed: {res['error']}")
        parts = [pd.read_parquet(p) for p in sorted(Path(tmp).glob("*.parquet"))]
        again_t = (
            pd.concat([p for p in parts if len(p)], ignore_index=True)
            .sort_values(["cell", "position"])
            .reset_index(drop=True)
        )
    stored_t = (
        t[t["cell"].str[:10].isin(FIVE)].sort_values(["cell", "position"]).reset_index(drop=True)
    )
    same_t = len(again_t) == len(stored_t) and again_t.equals(stored_t)
    say(
        f"- **10.7 reproducibility** (entries {', '.join(FIVE)}): attribution re-run identical on {len(stored)} rows: {verdict(same_a)}; "
        f"turnover re-run identical on {len(stored_t)} rows: {verdict(same_t)}; the touch frequencies are recomputed from the cells in 10.5: "
        f"**{verdict(same_a and same_t)}**"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--report", default=str(bh.OUT / "report"))
    ap.add_argument("--sample", default="strict", choices=["strict", "built"])
    args = ap.parse_args()
    a = pd.read_parquet(bh.RESULTS / "positions_attrib.parquet")
    t = pd.read_parquet(bh.RESULTS / "turnover.parquet")
    say(f"# Checks of addendum 2 (§10) — {pd.Timestamp.now():%Y-%m-%d %H:%M}\n")
    for check in (
        lambda: check_carry_identity(a),
        check_zero_vega,
        check_realised_vol,
        lambda: check_turnover(t),
        lambda: check_touch(Path(args.report), args.sample),
        check_tilt,
        lambda: check_repro(a, t),
    ):
        try:
            check()
        except Exception as exc:
            say(f"- a check raised {type(exc).__name__}: {exc}: **FAIL**")
    OUT.write_text("\n".join(_lines) + "\n")


if __name__ == "__main__":
    main()
