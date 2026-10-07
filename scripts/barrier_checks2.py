"""Checks on the addendum's new code (BARRIER_STUDY_ADDENDUM §11): each reported pass or fail,
nothing tuned.

    python scripts/barrier_checks2.py [--skip-flat] [--skip-repro]

Writes ``outputs/interview/checks2.md`` (appended to PROGRESS by hand or by the chain's log).
"""

# ruff: noqa: RUF001, E501 — messages of the checks: typographic signs, long lines
from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

for _v in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import barrier_history as run  # noqa: E402

from volsto.market.curves import ForwardCurve  # noqa: E402
from volsto.market.surface import ImpliedSurface  # noqa: E402
from volsto.studies import barrier_history as bh  # noqa: E402
from volsto.studies import barrier_theory as bt  # noqa: E402

KEY = ["entry", "months", "side", "barrier"]
lines: list[str] = []


def say(text: str) -> None:
    print(text)
    lines.append(text)


class Flat(ImpliedSurface):
    def __init__(self, vol: float, spot: float, r: float, q: float) -> None:
        fc = ForwardCurve.flat(spot, r, q)
        super().__init__(fc, fc.rate_curve, 5.0)
        self.vol = vol

    def total_variance(self, k: Any, T: Any) -> Any:
        return self.vol**2 * np.asarray(T, dtype=np.float64) + 0.0 * np.asarray(k, dtype=np.float64)


def check_identities() -> None:
    """11.1 and 11.2 on the stored bucket statistics, per model."""
    cells = pd.read_parquet(bh.RESULTS / "cells.parquet")
    for model, path, merge_cells in (
        ("local vol", bh.RESULTS / "buckets_lv.parquet", True),
        ("LSV (ssr12)", None, False),
    ):
        if model.startswith("LSV"):
            files = sorted((bh.RESULTS / "lsv2").glob("*.parquet"))
            if not files:
                say(f"- 11.1 / 11.2 {model}: no LSV bucket statistics yet")
                continue
            b = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
            pre, a1, a1_se, p1 = "ssr12", "ssr12_a1", "ssr12_a1_se", "ssr12_p1"
        else:
            if path is None or not path.exists():
                say(f"- 11.1 / 11.2 {model}: buckets_lv.parquet not there")
                continue
            b = pd.read_parquet(path)
            pre, a1, a1_se, p1 = "lv", "lv_a1", "lv_a1_se", "lv_p1"
        b = b.merge(
            cells[
                [*KEY, "K", "DF0", "P_B6", "P_B3", "B", *([a1, a1_se, p1] if merge_cells else [])]
            ],
            on=KEY,
        )
        b = b[b[f"{pre}_d_p1"].notna()]
        p = np.column_stack([b[f"{pre}_d_p{j}"] for j in (1, 2, 3)])
        v = np.nan_to_num(np.column_stack([b[f"{pre}_d_vB6{j}"] for j in (1, 2, 3)]))
        d = np.nan_to_num(np.column_stack([b[f"{pre}_d_d{j}"] for j in (1, 2, 3)]))
        K = b["K"].to_numpy(float)
        e1 = np.abs((p * v).sum(1) - (b[f"{pre}_bk_b6"] - b[f"{pre}_bk_a1"]).to_numpy()) / K
        e2 = np.abs(p.sum(1) - b[p1].to_numpy())
        e3 = np.abs((p * d).sum(1) - b[f"{pre}_bk_beyond"].to_numpy())
        ok1, ok2, ok3 = e1.max() < 1e-9, e2.max() < 1e-9, e3.max() < 1e-9
        say(
            f"- **11.1 touch identity, {model}** ({len(b)} cells): max |Σ p_j·vB6_j − (mean tight limit − "
            f"mean daily KO)| = {e1.max():.1e} of spot → {'PASS' if ok1 else 'FAIL'}; max |Σ p_j − p1| = "
            f"{e2.max():.1e} → {'PASS' if ok2 else 'FAIL'}"
        )
        # beside it: against the surface tight limit minus the controlled knock-out price
        lhs = (p * v).sum(1)
        rhs = ((b["P_B6"] - b[a1]) / b["DF0"]).to_numpy()
        se = (b[a1_se] / b["DF0"]).to_numpy()
        z = np.abs(lhs - rhs) / np.where(se > 0, se, np.nan)
        say(
            f"  beside it, against (P_B6 − {a1})/DF0 in Monte Carlo standard errors: median |z| "
            f"{np.nanmedian(z):.2f}, share beyond 2: {np.nanmean(z > 2):.3f}, beyond 3: {np.nanmean(z > 3):.3f} "
            f"(max difference {np.nanmax(np.abs(lhs - rhs) / K) * 1e4:.2f} bp of spot)"
        )
        w = np.abs(b["B"] - b["K"]).to_numpy(float)
        dig = ((b["P_B3"] - b["P_B6"]) / (w * b["DF0"])).to_numpy()
        say(
            f"- **11.2 one-touch identity, {model}**: max |Σ p_j·d_j − share of paths beyond B| = "
            f"{e3.max():.1e} → {'PASS' if ok3 else 'FAIL'}; beside the surface digital: mean path share "
            f"{b[f'{pre}_bk_beyond'].mean():.4f}, mean surface digital {np.nanmean(dig):.4f}"
        )


def check_flat() -> None:
    """11.3: on a flat 20 % surface the continuous knock-out is C8 (zero rates), and with carry
    Π_skew is zero — each within 3 standard errors, every cell."""
    days = [d.strftime("%Y-%m-%d") for d in pd.bdate_range("2030-01-02", "2031-01-10")]
    entry = days[0]
    times = np.array([bh.year_fraction(entry, d) for d in days[1:]])
    for label, r, q in (
        ("zero rates and dividends", 0.0, 0.0),
        ("r = 4.5 %, q = 1.5 %", 0.045, 0.015),
    ):
        surf = Flat(0.20, 100.0, r, q)
        rows = []
        for months in bh.MATURITY_MONTHS:
            expiry = bh.expiry_date(entry, months, days)
            T = bh.year_fraction(entry, expiry)
            for side in (1, -1):
                for name, B in bh.barrier_levels(side, 100.0, 0.20, T).items():
                    rows.append({"entry": entry, "months": months, "side": side, "barrier": name,
                                 "K": 100.0, "B": B, "expiry": expiry, "T": T})  # fmt: skip
        cells = pd.DataFrame(rows)
        legs = [
            (int(s), bh.structure_legs(int(s), 100.0, float(b), 100.0))
            for s, b in zip(cells["side"], cells["B"], strict=True)
        ]
        pieces = bh.expand_legs(legs)
        piece_T = cells["T"].to_numpy(float)[pieces["slot"] // len(bh.STRUCTURES)]
        v0 = bh.bulk_values(pieces, piece_T, surf)[0]
        df = np.array([float(surf.discount.df(t)) for t in cells["T"]])
        ko = run._ko_table(
            bh.local_vol_model(surf),
            times,
            20300102,
            cells,
            days[1:],
            v0[:, bh.STRUCTURES.index("B6")],
            df,
        )
        c8 = np.array(
            [
                bt.c8_bs(int(s), 100.0, 100.0, float(b), float(t), 0.20, r, q)
                for s, b, t in zip(cells["side"], cells["B"], cells["T"], strict=True)
            ]
        )
        flat = np.array(
            [
                bt.pi_flat(int(s), 100.0, 100.0, float(b), float(t), 0.20, r, q)
                for s, b, t in zip(cells["side"], cells["B"], cells["T"], strict=True)
            ]
        )
        pi_c = cells["side"].to_numpy(float) * (ko["a2"].to_numpy(float) - c8)
        stat = pi_c - flat  # Π_skew (Π_flat is zero when r = q)
        se = ko["a2_se"].to_numpy(float)
        z = np.abs(stat) / np.where(se > 0, se, np.nan)
        n_bad = int(np.nansum(z > 3))
        say(
            f"- **11.3 flat smile, {label}** ({len(cells)} cells, exact C8 with the exact digital): "
            f"|Π_skew[lv]| within 3 standard errors in {len(cells) - n_bad} of {len(cells)} cells; max |z| "
            f"{np.nanmax(z):.2f}; max |Π_skew| {np.nanmax(np.abs(stat)) * 100:.3f} bp of spot (spot 100) → "
            f"{'PASS' if n_bad == 0 else 'FAIL'}"
        )
        if r > 0:
            pick = cells[(cells["months"] == 6)]
            say(
                "  continuous knock-out, 6m, against Reiner–Rubinstein with carry: max |MC − closed form| "
                f"{np.max(np.abs(ko.loc[pick.index, 'a2'].to_numpy(float) - np.array([bt.ko_rr(int(s), 100.0, 100.0, float(b), float(t), 0.20, r, q) for s, b, t in zip(pick['side'], pick['B'], pick['T'], strict=True)]))):.4f}"
                " (spot 100)"
            )


def check_pi_d() -> None:
    """11.4: on 200 random trades not knocked, unhedged P&L of the daily knock-out minus that of
    C8 sold at the touch is (P_C8 − lv_a1)/DF0."""
    cells = pd.read_parquet(bh.RESULTS / "cells.parquet")
    pos = pd.read_parquet(
        bh.RESULTS / "positions.parquet", columns=["cell", "position", "model", "pnl_u"]
    )
    c = cells[
        cells["finished"].fillna(False).astype(bool)
        & cells["tau1"].isna()
        & cells["barrier"].str.startswith("s")
    ]
    c = c.sample(200, random_state=11)
    w = pos[pos["cell"].isin(c["cell"]) & (pos["model"] != "lsv")].pivot_table(
        index="cell", columns="position", values="pnl_u", aggfunc="first"
    )
    c = c.set_index("cell")
    want = (c["P_C8"] - c["lv_a1"]) / c["DF0"] / c["K"]
    err = (w["A1"] - w["C8"] - want).abs()
    say(
        f"- **11.4 exactness of Π_d** (200 random trades not knocked): max |pnl_u[KO] − pnl_u[C8] − "
        f"(P_C8 − lv_a1)/DF0| = {err.max():.2e} of spot → {'PASS' if err.max() < 1e-6 else 'FAIL'}"
    )


def check_repro() -> None:
    """11.7: sections 5 to 7 re-run on five entries give identical numbers."""
    import barrier_buckets as bk
    import barrier_features2 as f2
    import barrier_touches as tc

    entries = sorted(p.stem for p in (bh.RESULTS / "features2").glob("*.parquet"))
    pick = [entries[i] for i in np.linspace(0, len(entries) - 1, 5).astype(int)]
    with tempfile.TemporaryDirectory() as tmp:
        for mod, name in ((f2, "features2"), (bk, "buckets_lv"), (tc, "touches")):
            stored = bh.RESULTS / name
            mod.OUT = Path(tmp) / name
            worst = 0.0
            same = True
            for e in pick:
                job = {
                    "features2": f2.feature_job,
                    "buckets_lv": bk.bucket_job,
                    "touches": tc.touch_job,
                }[name]
                res = job(e)
                if "error" in res or not (stored / f"{e}.parquet").exists():
                    same = False
                    continue
                a, b = pd.read_parquet(stored / f"{e}.parquet"), pd.read_parquet(
                    mod.OUT / f"{e}.parquet"
                )
                if a.shape != b.shape:
                    same = False
                    continue
                num = a.select_dtypes("number").columns
                diff = a[num].to_numpy(float) - b[num].to_numpy(float)
                worst = max(worst, float(np.nanmax(np.abs(diff))) if diff.size else 0.0)
                same &= bool(
                    np.array_equal(a[num].to_numpy(float), b[num].to_numpy(float), equal_nan=True)
                )
            say(
                f"- **11.7 reproducibility, {name}** (entries {', '.join(pick)}): identical = {same}; "
                f"largest absolute difference {worst:.1e} → {'PASS' if same else 'FAIL'}"
            )
            mod.OUT = stored


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--skip-flat", action="store_true")
    ap.add_argument("--skip-repro", action="store_true")
    args = ap.parse_args()
    steps = [check_identities, check_pi_d]
    if not args.skip_flat:
        steps.append(check_flat)
    if not args.skip_repro:
        steps.append(check_repro)
    for step in steps:
        try:
            step()
        except Exception as exc:
            import traceback

            say(f"- {step.__name__}: could not run — {type(exc).__name__}: {exc}")
            print(traceback.format_exc()[-800:])
    say(
        "- 11.5 (n_eff[lv] against restarts at the bucket mid-dates, monthly subset): not run in this pass (last priority tier)"
    )
    qp = bh.RESULTS / "quotes.parquet"
    if not qp.exists():
        say("- 11.6 (quotes): the quote-based repricing has not run")
    else:
        q = pd.read_parquet(qp)
        n_all = n_bad = 0
        worst = 0.0
        for name in ("K", "B", "far", "sym"):
            ok = (100 * (q[f"vq_{name}"] - q[f"vs_{name}"])).abs() < 0.05
            rel = ((q[f"cq_{name}"] - q[f"cs_{name}"]).abs() / q[f"cs_{name}"].abs())[
                ok & (q[f"cs_{name}"] > 2e-5)
            ]
            n_all += len(rel)
            n_bad += int((rel > 0.01).sum())
            worst = max(worst, float(rel.max()) if len(rel) else 0.0)
        say(
            f"- **11.6 quotes**: {n_all} strikes whose surface fit error is below 0.05 vol point (and "
            f"whose vanilla is worth more than 0.2 bp); quote-based and surface prices differ by more "
            f"than 1 % of the price at {n_bad}; largest {100 * worst:.2f} % → {'PASS' if n_bad == 0 else 'FAIL'}"
        )
    (bh.OUT / "checks2.md").write_text("\n".join(lines) + "\n")
    write_summary()


def write_summary() -> None:
    """Addendum §12: at the end of phase F, a short summary at the top of PROGRESS — which
    estimator, what ran, what did not, the verdict table (written by the chain when the
    session is not there to write it)."""
    import json
    import time

    progress = bh.OUT / "PROGRESS.md"
    gate = json.loads((bh.OUT / "gate.json").read_text()) if (bh.OUT / "gate.json").exists() else {}
    res = bh.RESULTS
    counts = {
        "LSV entry files (lsv2)": len(list((res / "lsv2").glob("*.parquet"))),
        "LSV daily files (lsv2_daily)": len(list((res / "lsv2_daily").glob("*.parquet"))),
        "features2": len(list((res / "features2").glob("*.parquet"))),
        "local-vol buckets": len(list((res / "buckets_lv").glob("*.parquet"))),
        "touch files": len(list((res / "touches").glob("*.parquet"))),
        "quote files": len(list((res / "quotes").glob("*.parquet"))),
    }
    out = [
        f"# Summary at the end of phase F — {time.strftime('%Y-%m-%d %H:%M')} (written by the chain)",
        "",
        f"- Estimator: **{gate.get('estimator') or 'sorted (default path)'}**"
        + (f" — {gate.get('reason')}" if gate.get("reason") else ""),
        "- Ran: " + "; ".join(f"{k} {v}" for k, v in counts.items()),
        "- Reports: `report/report.pdf` (strict sample), `report_built/report.pdf` (all built entries)",
        "- Not run: check 11.5 (restarts at the bucket mid-dates); the LSV daily study is phase G/H",
        "- Checks of §11: see `checks2.md`: "
        + "; ".join(
            ln.split("**")[1] + (" PASS" if "PASS" in ln and "FAIL" not in ln else " FAIL")
            for ln in lines
            if ln.startswith("- **") and ("PASS" in ln or "FAIL" in ln)
        ),
        "",
    ]
    verdict = bh.OUT / "report" / "tables" / "verdict.csv"
    if verdict.exists():
        v = pd.read_csv(verdict)
        out += ["| prediction | statistic | estimate | verdict |", "|---|---|---|---|"]
        out += [
            f"| {r.prediction} | {r.statistic} | {r.estimate} | {r.verdict} |"
            for r in v.itertuples()
        ]
        out.append("")
    marker = "# Summary at the end of phase F"
    text = progress.read_text()
    if text.startswith(marker):  # replace an earlier summary
        text = text[text.index("\n---\n", 1) + 5 :] if "\n---\n" in text else text
    progress.write_text("\n".join(out) + "\n---\n\n" + text)


if __name__ == "__main__":
    main()
