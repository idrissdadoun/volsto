"""PM results package of 2026-10-09, section A, check (e): why the cross-dependent volatility
prototype moves the Palladium forward less than the fixed-kappa wing estimate.

    python scripts/pm_check_e.py

Input: the scans of ``scripts/cdv_scan.py`` (branch ``cross-dependent-vol``) under
``outputs/dispersion_lc/cdv/pm``: today (2026-10-02) at the production budget and the two other
dates of the overnight scan (2017-04-03, 2019-09-03) at the development budget, all at
``beta = 0, 3, 6`` with ``g_max = 3`` and M12's defaults of 2026-10-09.  A scan that is not
there is written as pending.

Output: the part ``A_check_e`` (records and Markdown) and ``tables/A_check_e_attribution.csv``,
``tables/A_check_e_regions.csv``.  Nothing is estimated here: every number is a field of a scan
file (``attribution_from_beta_0`` and the rows' ``regions``), whose definitions are in
``cdv_scan.attribution``:

    d ln E[D] = d ln kappa_h + (1/2) d ln E_h[V]          (names' second moment held at beta = 0)
    (1/2) d ln E_h[V] = wing estimate + short term,

so the gap of the realised move to the wing estimate is ``d ln kappa_h`` (kappa up) plus the
``short term`` (the basket's second moment still below the listed one).
"""

# ruff: noqa: E501, RUF001
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_common as pc

log = logging.getLogger("pm_check_e")
SCANS = pc.LC_OUT / "cdv" / "pm"
DATES = (("2026-10-02", "production"), ("2017-04-03", "development"), ("2019-09-03", "development"))
TERMS = (
    (
        "dln_ED",
        "move of the forward, d ln E[D]",
        "ln(E_beta[D]/E_0[D]), paired on the pricing paths",
    ),
    (
        "wing_estimate",
        "fixed-kappa wing estimate",
        "(1/2) ln((sum_w E_0[R_i^2] - M_B^listed)/E_0[V]): the basket's second moment taken to the listed one at unchanged names and kappa",
    ),
    ("gap_to_wing_estimate", "gap: move minus estimate", "d ln E[D] - wing estimate"),
    (
        "dln_kappa_names_held",
        "of which kappa up",
        "d ln kappa with the names' second moment held at its beta = 0 estimate",
    ),
    (
        "short_term_names_held",
        "of which the basket still short of the listed moment",
        "(1/2) ln((N_0 - E_beta[Rbar^2])/(N_0 - M_B^listed)), N_0 the names' second moment at beta = 0",
    ),
    (
        "names_drift",
        "measured drift of the names' second moment (zero in the model)",
        "(1/2) d ln E[V] minus the same with the names held: Monte Carlo noise of the names' upper tails",
    ),
    (
        "shortfall_closed",
        "share of the basket's shortfall closed",
        "(E_beta[Rbar^2] - E_0[Rbar^2])/(M_B^listed - E_0[Rbar^2])",
    ),
    ("kappa_base", "kappa at beta = 0", "E_0[D]/sqrt(E_0[V])"),
    ("kappa_names_held", "kappa at beta, names held", "E_beta[D]/sqrt(N_0 - E_beta[Rbar^2])"),
)
PCT = {
    "dln_ED",
    "wing_estimate",
    "gap_to_wing_estimate",
    "dln_kappa_names_held",
    "short_term_names_held",
    "names_drift",
    "shortfall_closed",
}


def fmt(name: str, v: float, se: float) -> str:
    if name in PCT:
        return (
            f"{100 * v:+.3f} ± {100 * se:.3f} %"
            if name != "shortfall_closed"
            else f"{100 * v:.1f} ± {100 * se:.1f} %"
        )
    return f"{v:.4f} ± {se:.4f}"


def build() -> None:
    records: list[dict[str, Any]] = []
    md: list[str] = []
    att_rows: list[dict[str, Any]] = []
    reg_rows: list[dict[str, Any]] = []
    add = md.append
    add(
        'Check (e): "The forward moved 0.4 %, not the 1.2 % I estimated at fixed κ. Two candidate reasons: E[R̄²] still short of M_B^listed beyond −2.5 sd, or κ up. Say which. Also give the other two dates of your scan."'
    )
    add("")
    answers: list[str] = []
    for date, budget in DATES:
        path = SCANS / f"cdv_scan_{date}_3m.json"
        add(f"**{date}** ({budget} budget)")
        add("")
        if not path.exists():
            add(f"pending: `{path.relative_to(pc.LC_OUT.parent.parent)}`")
            add("")
            continue
        doc = json.loads(path.read_text())
        if doc["budget"] != budget:
            raise ValueError(f"{path}: budget {doc['budget']}, expected {budget}")
        rec = doc["record"]
        rows = {float(r["beta"]): r for r in doc["rows"]}
        g_max = {float(r["g_max"]) for r in doc["rows"]}
        src = str(path)
        bud = f"{rec['n_particles']:.0e} particles / {rec['n_paths']:.0e} paths".replace("e+0", "e")
        atts = {float(a["beta"]): a for a in doc["attribution_from_beta_0"]}
        betas = sorted(atts)
        add("| term | " + " | ".join(f"β = {b:g}" for b in betas) + " |")
        add("|:--|" + "--:|" * len(betas))
        for name, label, definition in TERMS:
            cells = []
            for b in betas:
                v, se = atts[b][name], atts[b][f"{name}_se"]
                cells.append(fmt(name, v, se))
                records.append(pc.record(
                    f"A.check_e.{date}.beta{b:g}.{name}", "A", f"check (e): {label}, beta = {b:g}", v, se, date=date,
                    unit="log change" if name in PCT and name != "shortfall_closed" else ("share" if name == "shortfall_closed" else "ratio"),
                    definition=definition, budget=bud, commit=rec["git_commit"], source=src, notes=f"g_max {sorted(g_max)}; prototype branch cross-dependent-vol",
                ))  # fmt: skip
                att_rows.append(
                    {
                        "date": date,
                        "budget": budget,
                        "beta": b,
                        "term": name,
                        "label": label,
                        "value": v,
                        "se": se,
                    }
                )
            add(f"| {label} | " + " | ".join(cells) + " |")
        add("")
        if date != DATES[0][0]:
            core = (
                (
                    "clip_high_inner",
                    "clipped mass inside ±2.5 sd, at the cap (% of particles)",
                    100.0,
                    None,
                    3,
                ),
                (
                    "clip_low_inner",
                    "clipped mass inside ±2.5 sd, at λ = 0 (% of particles)",
                    100.0,
                    None,
                    3,
                ),
                ("ED", "E[D] (notional)", 1.0, "ED_se", 6),
                (
                    "ED_over_cc",
                    "E[D]/E_CC[D] (E_CC[D] of the development row, unpaired)",
                    1.0,
                    None,
                    5,
                ),
                ("kappa", "κ", 1.0, "kappa_se", 4),
                ("EV_over_EQV", "E[V]/EQV", 1.0, "EV_over_EQV_se", 4),
                ("E_Rbar2_over_listed", "E[R̄²]/M_B^listed", 1.0, "E_Rbar2_over_listed_se", 4),
                ("idx_-2.5", "index error at −2.5 sd (vol points)", 1.0, "idx_-2.5_se", 3),
                ("idx_-3.0", "index error at −3.0 sd (vol points)", 1.0, "idx_-3.0_se", 3),
                ("idx_-3.5", "index error at −3.5 sd (vol points)", 1.0, "idx_-3.5_se", 3),
            )
            add("| quantity | " + " | ".join(f"β = {b:g}" for b in sorted(rows)) + " |")
            add("|:--|" + "--:|" * len(rows))
            for key, label, scale, se_key, digits in core:
                cells = []
                for b in sorted(rows):
                    v = rows[b].get(key)
                    se = rows[b].get(se_key) if se_key else None
                    cells.append(pc.pm(None if v is None else scale * v, se, digits))
                    records.append(pc.record(
                        f"A.check_e.{date}.beta{b:g}.core.{key}", "A", f"CDV scan, {label}, beta = {b:g}", None if v is None else scale * v, se, date=date,
                        unit="", definition=f"field {key} of the scan row (cdv_scan.py); beta = 0 is the local correlation model", budget=bud, commit=rec["git_commit"], source=src,
                        notes=f"g_max {sorted(g_max)}" + ("; a calibration diagnostic, no standard error" if key.startswith("clip") else ""),
                    ))  # fmt: skip
                add(f"| {label} | " + " | ".join(cells) + " |")
            add("")
        labels = rows[0.0]["regions"]["labels"]
        add(
            "| basket second moment by region of the index return, model − target (1e-6) | "
            + " | ".join(labels)
            + " | total |"
        )
        add("|:--|" + "--:|" * (len(labels) + 1))
        for b in sorted(rows):
            reg = rows[b]["regions"]
            diff, se = reg["difference"], reg["model_se"]
            total = reg["total_model"] - reg["total_target"]
            add(
                f"| β = {b:g} | "
                + " | ".join(
                    f"{1e6 * d:+.0f} ± {1e6 * s:.0f}" for d, s in zip(diff, se, strict=True)
                )
                + f" | {1e6 * total:+.0f} ± {1e6 * reg['total_model_se']:.0f} |"
            )
            for lab, d, s in zip(labels, diff, se, strict=True):
                reg_rows.append(
                    {
                        "date": date,
                        "budget": budget,
                        "beta": b,
                        "region": lab,
                        "model_minus_target": d,
                        "se": s,
                    }
                )
                records.append(pc.record(
                    f"A.check_e.{date}.beta{b:g}.region.{lab}", "A", f"check (e): basket second moment, model minus target, {lab}, beta = {b:g}", d, s, date=date,
                    unit="units of squared return", definition="E[Rbar^2 - 1 restricted to the region of the index log-return, in at-the-money sd] of the model minus the same of the index target's strip",
                    budget=bud, commit=rec["git_commit"], source=src, notes="the se is the model's; the target's strip is exact",
                ))  # fmt: skip
        add("")
        clauses = []
        for b in betas:
            a = atts[b]
            reg = rows[b]["regions"]
            short = [-d for d in reg["difference"]]
            below = short[0] / sum(short[:4]) if sum(short[:4]) > 0 else float("nan")
            gap, k_part, s_part = (
                a["gap_to_wing_estimate"],
                a["dln_kappa_names_held"],
                a["short_term_names_held"],
            )
            move = f"β = {b:g}: forward {100 * a['dln_ED']:+.2f} % (± {100 * a['dln_ED_se']:.2f}) against a fixed-κ estimate of {100 * a['wing_estimate']:+.2f} % (± {100 * a['wing_estimate_se']:.2f})"
            kappa = f"κ {100 * k_part:+.2f} % (± {100 * a['dln_kappa_names_held_se']:.2f}; {a['kappa_base']:.4f} → {a['kappa_names_held']:.4f})"
            if a["wing_estimate"] > -2.0 * a["wing_estimate_se"]:
                # no measurable shortfall of the basket's second moment at beta = 0
                clauses.append(
                    f"{move}: there is no measurable shortfall to close on this date (the estimate is within two standard errors of zero), and the move is {kappa} plus the E[V] term {100 * a['half_dln_EV_names_held']:+.2f} %"
                )
            else:
                if k_part > 0.0 and s_part > 0.0:
                    larger = (
                        "κ is the larger part"
                        if k_part > s_part
                        else "the basket still short is the larger part"
                    )
                elif k_part <= 0.0:
                    larger = "κ falls and offsets the basket still short"
                else:
                    larger = "κ rises; the basket is not short"
                clauses.append(
                    f"{move}; the gap of {100 * gap:+.2f} % is {kappa} plus the basket still short of the listed moment {100 * s_part:+.2f} % (± {100 * a['short_term_names_held_se']:.2f}): {larger}; "
                    f"{100 * a['shortfall_closed']:.0f} % (± {100 * a['shortfall_closed_se']:.0f}) of the shortfall is closed and {100 * below:.0f} % of what remains below the money sits below −3.5 sd"
                )
                records.append(pc.record(
                    f"A.check_e.{date}.beta{b:g}.kappa_share_of_gap", "A", f"check (e): share of the gap to the wing estimate due to kappa, beta = {b:g}", k_part / gap, None, date=date, unit="share",
                    definition="d ln kappa (names held) / (d ln E[D] - wing estimate)", budget=bud, commit=rec["git_commit"], source=src, notes="ratio of two fields of the scan; no error computed",
                ))  # fmt: skip
                records.append(pc.record(
                    f"A.check_e.{date}.beta{b:g}.remaining_below_m35", "A", f"check (e): share of the remaining downside shortfall that sits below -3.5 sd, beta = {b:g}", below, None, date=date, unit="share",
                    definition="(target - model) below -3.5 sd over (target - model) summed over the four regions below the money", budget=bud, commit=rec["git_commit"], source=src, notes="ratio of fields of the scan; no error computed",
                ))  # fmt: skip
        answers.append(f"*{date} ({budget})* — " + ". ".join(clauses) + ".")
    add(
        "**Answer.** Both reasons hold, in proportions that depend on the date and on β (terms defined below; the clipped masses and index errors of today's scan are in A8)."
    )
    add("")
    for text in answers:
        add(f"- {text}")
    add("")
    add(
        'Terms: with the names\' second moment held at its β = 0 estimate (it is the same under every β in the model; its measured drift is the row "measured drift"), '
        "d ln E[D] = d ln κ + ½ d ln E[V] and ½ d ln E[V] = wing estimate + short term, so the gap to the wing estimate is κ up plus the basket still short (`cdv_scan.attribution`; paired on the pricing paths, delta-method errors). "
        "Prototype: `scripts/cdv_scan.py`, branch `cross-dependent-vol`, g_max = 3, M12's defaults of 2026-10-09. Files: `tables/A_check_e_attribution.csv`, `tables/A_check_e_regions.csv`."
    )
    pc.save_table(pd.DataFrame(att_rows), "A_check_e_attribution")
    pc.save_table(pd.DataFrame(reg_rows), "A_check_e_regions")
    pc.write_part("A_check_e", records, "\n".join(md))
    log.info("A_check_e written: %d records; answers for %d dates", len(records), len(answers))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    build()
