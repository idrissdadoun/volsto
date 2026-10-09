"""PM results package of 2026-10-09, section V: validation status and caveats.

    python scripts/pm_validation.py [--s3-log <file>] [--runaway <folder>]

What it writes (``outputs/dispersion_lc/pm_update``): the part ``V_validation`` (records and
Markdown), ``tables/V_s3_gate.csv``, ``tables/V_runaway_paths.csv``,
``tables/V_runaway_ratios.csv`` and, so that the package holds its own sources, copies of the
two diagnostics' small files under ``diagnostics/s3`` and ``diagnostics/runaway`` (logs, JSON
results and the scratch scripts; not the saved paths).

Sources.  (1) The log of the slow test ``tests/test_local_correlation.py::
test_s3_identical_names`` at the production size (800000 paths), run on 2026-10-09 with the gate
the owner decided that day: the table of the calibrated model's index smile minus the smile under
``lambda = 1`` on the same paths.  (2) The verification of the runaway-path diagnostic (an
independent measurement on the stored development rows' model, bit for bit): the share of each
price carried by the paths on which some name ends above 3 times its spot.

Nothing is estimated here: the numbers are parsed from the log and read from the JSON results.
"""

# ruff: noqa: E501, RUF001
from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_common as pc

log = logging.getLogger("pm_validation")
SCRATCH = Path(
    "/private/tmp/claude-501/-Users-idrissdadoun-Code-volsto/75dd7f23-d74c-43a9-b99e-61c23722b22c/scratchpad"
)
S3_LOG = SCRATCH / "r4" / "s3-gate" / "s3_run2.log"
RUNAWAY = SCRATCH / "pm" / "verify" / "runaway"
S3_GATE_VP = 0.05
CELL = re.compile(r"(-?\d+\.\d+) \((\d+\.\d+)\)")
MULTS = (
    ("0.5", "050"),
    ("0.75", "075"),
    ("1.0", "100"),
    ("1.25", "125"),
    ("1.5", "150"),
    ("2.0", "200"),
)


def s3_table(text: str) -> pd.DataFrame:
    """The gate's table of the log: one row per (pillar, strike in sd) with the difference of the
    two smiles in vol points and its paired standard error."""
    lines = text.splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith("S3 gate (owner's decision 4")
    )
    sds = [float(x) for x in lines[start + 1].split()[1:]]
    rows = []
    for line in lines[start + 3 : start + 6]:
        T = float(line.split()[0])
        cells = CELL.findall(line)
        if len(cells) != len(sds):
            raise ValueError(f"S3 log: {len(cells)} cells for {len(sds)} strikes")
        rows += [
            {"T": T, "months": round(12 * T), "sd": sd, "diff_vp": float(v), "se_vp": float(se)}
            for sd, (v, se) in zip(sds, cells, strict=True)
        ]
    out = pd.DataFrame(rows)
    out["over_gate"] = out["diff_vp"].abs() > S3_GATE_VP
    return out


def num(x: float) -> str:
    """A ratio for a table cell: three decimals below 10, one above."""
    return f"{x:.3f}" if abs(x) < 10 else f"{x:.1f}"


def copy_sources(s3_log: Path, runaway: Path) -> tuple[Path, Path]:
    d3, dr = pc.PM / "diagnostics" / "s3", pc.PM / "diagnostics" / "runaway"
    d3.mkdir(parents=True, exist_ok=True)
    dr.mkdir(parents=True, exist_ok=True)
    shutil.copy2(s3_log, d3 / "test_s3_identical_names_production.log")
    for f in sorted(runaway.iterdir()):
        if f.suffix in (".py", ".json", ".log"):
            shutil.copy2(f, dr / f.name)
    return d3 / "test_s3_identical_names_production.log", dr


def build(s3_log: Path, runaway: Path) -> None:
    s3_copy, run_copy = copy_sources(s3_log, runaway)
    records: list[dict[str, Any]] = []
    md: list[str] = []
    add = md.append

    # V1: the S3 gate
    text = s3_copy.read_text()
    s3 = s3_table(text)
    pc.save_table(s3, "V_s3_gate")
    over = s3[s3["over_gate"]]
    inside = s3[s3["sd"] <= 2.0]
    worst_in = inside.loc[inside["diff_vp"].abs().idxmax()]
    floors = next(line for line in text.splitlines() if line.startswith("S3 smallest lambda"))
    trusted = next(line for line in text.splitlines() if line.startswith("S3 gate, reported"))
    src = f"{s3_copy} (log of tests/test_local_correlation.py::test_s3_identical_names, production size, run of 2026-10-09)"
    add("### V1. Synthetic test S3 (identical names): the gate decided on 9 Oct FAILS at +2.5 sd")
    add("")
    add(
        f"Gate (owner's decision 4): |index smile under the calibrated λ − index smile under λ = 1 imposed| ≤ {S3_GATE_VP:g} vol points on common paths, at 1m, 2m, 3m and the 11 strikes −2.5 … +2.5 sd (33 cells). "
        f"**Result: FAIL — {len(over)} of {len(s3)} cells are over the gate, all at +2.5 sd:** "
        + "; ".join(
            f"{int(r.months)}m {r.diff_vp:+.4f} ± {r.se_vp:.4f} vp" for r in over.itertuples()
        )
        + f". Every cell from −2.5 to +2.0 sd passes (largest |difference| {abs(worst_in['diff_vp']):.4f} ± {worst_in['se_vp']:.4f} vp at {int(worst_in['months'])}m, {worst_in['sd']:+.1f} sd). "
        "The same three cells are over on each of four pricing seeds (the test's author reran it; log lines in `diagnostics/s3`). The gate is as decided and has not been changed; the two floors on λ pass."
    )
    add("")
    add("| pillar | " + " | ".join(f"{sd:+.1f} sd" for sd in sorted(s3["sd"].unique())) + " |")
    add("|:--|" + "--:|" * s3["sd"].nunique())
    for months, g in s3.groupby("months"):
        g = g.sort_values("sd")
        add(
            f"| {int(months)}m | "
            + " | ".join(
                ("**" if r.over_gate else "")
                + f"{r.diff_vp:+.4f} ± {r.se_vp:.4f}"
                + ("**" if r.over_gate else "")
                for r in g.itertuples()
            )
            + " |"
        )
        for r in g.itertuples():
            records.append(pc.record(
                f"V.s3.gate.{int(months)}m.sd{r.sd:+.1f}", "V", f"S3: index smile under the calibrated lambda minus under lambda = 1, {int(months)}m, {r.sd:+.1f} sd", r.diff_vp, r.se_vp,
                date="synthetic (identical names)", tenor=f"{int(months)}m", unit="vol points", definition="basket implied vol of the calibrated model minus the same with lambda = 1 imposed, on the same 800000 pricing paths; paired standard error; gate 0.05 vp",
                budget="8e5 particles / 8e5 paths (the test's production size)", commit="a102ef8 (test file; the gate is the one of d4faa74)", source=src, notes="OVER THE GATE: FAIL" if r.over_gate else "within the gate",
            ))  # fmt: skip
    add("")
    add(
        f'Cells: difference in vol points ± paired standard error; bold = over the gate. From the log: "{floors.strip()}". "{trusted.strip()}".'
    )
    add("")
    add(
        "What it means for the Dow numbers: the +2.5 sd strike lies beyond the upper end of the range on which the particle estimate of λ is trusted (about +2.07 at-the-money sd in this test); above it the default tail rule gives a λ of about 0.98 instead of 1. "
        "The downside wing (−2.5 sd), where the Dow calibration binds, passes at 0.010 vp. The failure is recorded for the owner; nothing was tuned to remove it."
    )
    add("")
    records.append(pc.record(
        "V.s3.gate.cells_over", "V", "S3: cells over the 0.05 vp gate", float(len(over)), None, date="synthetic (identical names)", unit="count of 33 cells",
        definition="number of (pillar, strike) cells with |difference| above 0.05 vol points", budget="8e5 particles / 8e5 paths", commit="a102ef8", source=src, n=len(s3), notes="the gate FAILS (a count, no standard error)",
    ))  # fmt: skip

    # V2: other validation items
    add("### V2. Other validation items, as they stand at the freeze")
    add("")
    add(
        "- **Row gates** (no NaN, forward, index smile): no gating check fails on the four production rows of sections A and B. The index gate (0.15 vp at the money and at the 90 % strike) is waived when the wing binds; on 2026-10-02 and 2017-04-03 the error at the 90 % strike is outside it (section A/B flags tables)."
    )
    add(
        "- **Names' 2 % check** (Σ w E[R_i²] of the model against the listed strips): a reported diagnostic since the owner's decision 3, not a gate. Its Monte Carlo number is not pinned down in the call wing (V3)."
    )
    add(
        "- **Golden baseline S11** (`tests/golden/lcm_baseline_2026-10-02.json`): recorded on the old defaults (no calendar repair); it has NOT been re-recorded under the defaults of 9 Oct, and the slow Dow tests (C2 Dow, S5 Dow, S11) have NOT been rerun under them. The fast suites of the local correlation files pass (70 tests) at commit a102ef8."
    )
    add(
        "- **Section C's rows** are from the variant development pass at commit 5b4700b (names' calendar repair and unscreened fallback on; the index repair of decision 5 not yet in the code). On the dates where the index repair would drop a DJX slice at 3m (32 of 218 dates built in a specification-only check) the current defaults give a different specification from those rows."
    )
    add("")

    # V3: runaway paths
    add("### V3. Caveat on the high-strike calls: a few runaway paths carry them (LC and CC alike)")
    add("")
    path_rows: list[dict[str, Any]] = []
    ratio_rows: list[dict[str, Any]] = []
    done = []
    for f in sorted(run_copy.glob("result_*.json")):
        doc = json.loads(f.read_text())
        date = doc["date"]
        done.append(date)
        meta = json.loads((run_copy / f"meta_{date}.json").read_text())
        bud = f"{meta['n_paths']:.0e} paths, development budget".replace("e+0", "e")
        fsrc = f"{f} (independent verification on the model of rows/3m_development_repair/{date}.json, bit for bit)"
        for model in ("lc", "cc"):
            m = doc["models"][model]
            for key, row in m["rows"].items():
                tag = "ED" if key == "E[D]" else dict(MULTS)[key]
                path_rows.append({
                    "date": date, "model": model.upper(), "price_of": "E[D]" if key == "E[D]" else f"call at {key} x P_D", "price": row["price"], "price_se": row["se"],
                    "share_paths_above_3x": row["share_gt_3"], "share_paths_above_2x": row["share_gt_2"], "n_paths_above_3x": m["n_gt"]["3"], "n_paths_above_2x": m["n_gt"]["2"],
                    "n_in_the_money": row["n_itm"], "n_in_the_money_above_3x": row["n_itm_gt3"], "share_top_1_path": row["top1"], "share_top_10_paths": row["top10"], "share_top_100_paths": row["top100"],
                    "bit_for_bit_with_the_row": m["bit_for_bit"],
                })  # fmt: skip
                records.append(pc.record(
                    f"V.runaway.{date}.{model}.{tag}.share_above_3x", "V", f"share of {'E[D]' if key == 'E[D]' else 'the call at ' + key + ' x P_D'} carried by paths with a name above 3 times its spot, {model.upper()}",
                    row["share_gt_3"], None, date=date, unit="share of the price", definition="sum of the payoff over the paths on which max_i S_i(T)/S_i(0) > 3, over the sum over all paths",
                    budget=bud, commit="model of the stored row (5b4700b); measured at a102ef8", source=fsrc, n=int(m["n_gt"]["3"]),
                    notes=f"n = paths above 3x of {meta['n_paths']}; the 10 largest paths carry {100 * row['top10']:.1f} % of the price; no usable standard error where a handful of paths carry the price",
                ))  # fmt: skip
        for key, tag in MULTS:
            r = doc["ratios"][tag]
            ratio_rows.append(
                {
                    "date": date,
                    "strike_multiple": float(key),
                    "copula_call": r["copula"],
                    **{
                        k: r[k]
                        for k in (
                            "lc_over_cc",
                            "lc_over_cc_ex",
                            "lc_over_cop",
                            "lc_over_cop_ex",
                            "cc_over_cop",
                            "cc_over_cop_ex",
                        )
                    },
                }
            )
            for k, label in (
                ("lc_over_cop", "LC/copula"),
                ("lc_over_cc", "LC/CC"),
                ("cc_over_cop", "CC/copula"),
            ):
                records.append(pc.record(
                    f"V.runaway.{date}.{tag}.{k}_ex", "V", f"call at {key} x P_D, {label}, with the payoff of the paths above 3x (under either model) set to zero", r[f"{k}_ex"], None, date=date, unit="ratio",
                    definition="a sensitivity, not a corrected price: the ratio after removing the payoff of the paths on which some name ends above 3 times its spot under LC or CC; the copula's call is unchanged",
                    budget=bud, commit="model of the stored row (5b4700b); measured at a102ef8", source=fsrc, notes=f"on all paths: {r[k]:.4g}",
                ))  # fmt: skip
    paths = pd.DataFrame(path_rows)
    ratios = pd.DataFrame(ratio_rows)
    pc.save_table(paths, "V_runaway_paths")
    pc.save_table(ratios, "V_runaway_ratios")
    add(
        "Beyond the last listed call strike the names' Dupire local volatility is an extrapolation (several hundred per cent in places, up to the 500 % cap, alternating in time with the 1 % floor). A few paths per 200000 therefore end with one name at 3 to 40 times its spot in three months "
        "— above every strike listed for that name — and these paths carry most of the calls at 1.5 and 2 times the forward, under LC and under CC. The owner's decision 3 leaves the tails as they are; the numbers below say how much of each price they are. "
        f"Measured at the development budget on {' and '.join(done)} (the model of the stored development rows, bit for bit), independently of the analyst who first reported it."
    )
    add("")
    add(
        "| date | model | paths above 3x (of 200000) | E[D] | call 0.75x | call 1.0x | call 1.25x | call 1.5x | call 2.0x |"
    )
    add("|:--|:--|--:|--:|--:|--:|--:|--:|--:|")
    for (date, model), g in paths.groupby(["date", "model"], sort=False):
        s = g.set_index("price_of")["share_paths_above_3x"]
        add(
            f"| {date} | {model} | {int(g['n_paths_above_3x'].iloc[0])} | "
            + " | ".join(
                f"{100 * s[k]:.2f} %" if k == "E[D]" else f"{100 * s[k]:.1f} %"
                for k in (
                    "E[D]",
                    "call at 0.75 x P_D",
                    "call at 1.0 x P_D",
                    "call at 1.25 x P_D",
                    "call at 1.5 x P_D",
                    "call at 2.0 x P_D",
                )
            )
            + " |"
        )
    add("")
    add("Share of each price carried by the paths on which some name ends above 3 times its spot. The 10 largest paths alone carry, under LC: " + "; ".join(
        f"{date}: {100 * paths[(paths.date == date) & (paths.model == 'LC') & (paths.price_of == 'call at 1.5 x P_D')]['share_top_10_paths'].iloc[0]:.0f} % of the 1.5x call and "
        f"{100 * paths[(paths.date == date) & (paths.model == 'LC') & (paths.price_of == 'call at 2.0 x P_D')]['share_top_10_paths'].iloc[0]:.0f} % of the 2.0x call" for date in done) + ".")  # fmt: skip
    add("")
    add(
        "| date | strike | LC/copula: all paths → without them | LC/CC: all paths → without them | CC/copula: all paths → without them |"
    )
    add("|:--|:--|--:|--:|--:|")
    for r in ratios.itertuples():
        add(
            f"| {r.date} | {r.strike_multiple:g}x | {num(r.lc_over_cop)} → {num(r.lc_over_cop_ex)} | {r.lc_over_cc:.4f} → {r.lc_over_cc_ex:.4f} | {num(r.cc_over_cop)} → {num(r.cc_over_cop_ex)} |"
        )
    add("")
    add(
        "Sensitivity, not a corrected price: the same ratios with the payoff of those paths (under either model) set to zero. **Reading for the report:** the forward E[D] is not affected (0.2 %); the at-the-forward call moves by 2–3 %; "
        "the call at 1.25x by 8–19 %; the ratios at 1.5x and 2.0x, against CC and against the copula, are essentially these paths and should not be quoted as model results (LC/copula of 11 and 20 at 2.0x is entirely them). "
        "The same holds for the production rows of sections A and B (K_150, K_200) and for the K_150 column and the upper strikes of figure F2 in section C, where the mean across dates sits well above the median. "
        "Files: `tables/V_runaway_paths.csv`, `tables/V_runaway_ratios.csv`, `diagnostics/runaway/` (scripts, logs and JSON results of the measurement)."
    )
    pc.write_part("V_validation", records, "\n".join(md))
    pc.status(
        f"V: validation part written (S3 gate: {len(over)} of {len(s3)} cells over 0.05 vp, FAIL; runaway-path caveat on {', '.join(done)}); sources copied to diagnostics/."
    )
    log.info(
        "V_validation written: %d records; S3 cells over the gate: %d of %d",
        len(records),
        len(over),
        len(s3),
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--s3-log", default=str(S3_LOG))
    ap.add_argument("--runaway", default=str(RUNAWAY))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    build(Path(args.s3_log), Path(args.runaway))
    return 0


if __name__ == "__main__":
    sys.exit(main())
