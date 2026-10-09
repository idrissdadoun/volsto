"""PM results package of 2026-10-09, section A, check (e): why the cross-dependent volatility
prototype moves the Palladium forward less than the fixed-kappa wing estimate.

    python scripts/pm_check_e.py

Input: the scans of ``scripts/cdv_scan.py`` (branch ``cross-dependent-vol``) under
``outputs/dispersion_lc/cdv/pm``: today (2026-10-02) at the production budget and the two other
dates of the overnight scan (2017-04-03, 2019-09-03) at the development budget, all at
``beta = 0, 3, 6`` with ``g_max = 3`` and M12's defaults of 2026-10-09; and, for ``E_CC[D]``
and the M12 row's ``E[D]``, the row of the same specification key under
``outputs/dispersion_lc/rows``.  A scan that is not there is written as pending.

Output: the part ``A_check_e`` (records and Markdown) and ``tables/A_check_e_attribution.csv``,
``tables/A_check_e_regions.csv``.  No path is simulated here.  The numbers are fields of the scan
files (``attribution_from_beta_0``, the rows and their ``regions``; definitions in
``cdv_scan.attribution`` and ``cdv_scan.basket_regions``) and closed-form functions of those
fields, each named where it is printed.

Two targets.  ``M_B^listed = sum_w M_i - EQV`` is the study's listed index second moment (field
``M_B_listed``).  The model is calibrated to another object, its own SVI index surface, whose
out-of-the-money strip gives, in the units of ``E[Rbar^2]``, the row field
``E_Rbar2_of_target``.  The stored attribution is against ``M_B^listed``:

    d ln E[D] = d ln kappa_h + (1/2) d ln E_h[V]          (names' second moment held at beta = 0)
    (1/2) d ln E_h[V] = wing estimate + short term,

so the gap of the realised move to the wing estimate is ``d ln kappa_h`` plus the ``short term``
(the basket's second moment still below the target).  With ``N0 = sum_w E_0[R_i^2]``,
``R_b = E_b[Rbar^2]`` and a target moment ``m``:

    wing estimate = (1/2) ln((N0 - m)/(N0 - R_0)),   short term = (1/2) ln((N0 - R_b)/(N0 - m)),
    shortfall closed = (R_b - R_0)/(m - R_0).

:func:`split_against` evaluates these; with ``m = M_B^listed`` it must return the stored fields
(asserted to 1e-9), and with ``m = E_Rbar2_of_target`` it gives the split against the model's own
index target (point values: no standard error is computed for them).
"""

# ruff: noqa: E501, RUF001
from __future__ import annotations

import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_common as pc

log = logging.getLogger("pm_check_e")
SCANS = pc.LC_OUT / "cdv" / "pm"
DATES = (("2026-10-02", "production"), ("2017-04-03", "development"), ("2019-09-03", "development"))
#: The folder of the M12 row with the scan's specification key, by budget.
ROW_DIRS = {"production": "3m_production", "development": "3m_development_repair"}
#: Decision 5's rule: a calibration is flagged when the clipped mass inside ±2.5 sd is above 1 %.
FLAG_LEVEL = 0.01
QUESTION = (
    "e) today at beta = 0 and beta = 3: E[V]/E^Q[V] with split; E[Rbar^2] vs M_B^listed; kappa; index smile errors "
    "-2.5 to -3.5 sd. The forward moved 0.4 %, not the 1.2 % I estimated at fixed kappa. Two candidate reasons: "
    "E[Rbar^2] still short of M_B^listed beyond -2.5 sd, or kappa up (to about 0.758 if the whole basket half of the "
    "E[V] excess went). Say which. Also give the other two dates of your scan."
)
HELD = "names' second moment held"
POINT = "point values, no standard error computed"
NO_SHORTFALL = "n/a (no shortfall against M_B^listed)"
#: name, label of the row, definition, source of the value ("att": attribution field with its _se).
TERMS = (
    (
        "dln_ED",
        "move of the forward, d ln E[D]",
        "ln(E_beta[D]/E_0[D]), paired on the pricing paths",
    ),
    (
        "wing_estimate",
        "fixed-κ wing estimate against M_B^listed",
        "(1/2) ln((sum_w E_0[R_i^2] - M_B^listed)/E_0[V]): the basket's second moment taken to the study's listed index second moment M_B^listed at unchanged names and kappa",
    ),
    ("gap_to_wing_estimate", "gap: move minus estimate", "d ln E[D] - wing estimate"),
    (
        "dln_kappa_names_held",
        f"of which κ ({HELD})",
        "d ln kappa with the names' second moment held at its beta = 0 estimate N_0: ln(kappa_h(beta)/kappa(0)), kappa_h(beta) = E_beta[D]/sqrt(N_0 - E_beta[Rbar^2])",
    ),
    (
        "short_term_names_held",
        f"of which the basket still short of M_B^listed ({HELD})",
        "(1/2) ln((N_0 - E_beta[Rbar^2])/(N_0 - M_B^listed)), N_0 the names' second moment at beta = 0",
    ),
    (
        "shortfall_closed",
        "share of the shortfall against M_B^listed closed",
        "(E_beta[Rbar^2] - E_0[Rbar^2])/(M_B^listed - E_0[Rbar^2])",
    ),
    (
        "half_dln_EV_names_held",
        f"½ d ln E[V] ({HELD}) = wing estimate + still short",
        "(1/2) ln((N_0 - E_beta[Rbar^2])/(N_0 - E_0[Rbar^2]))",
    ),
    (
        "names_drift",
        "measured drift of the names' second moment (zero in the model)",
        "(1/2) d ln E[V] as measured minus the same with the names' second moment held; zero in the model (each name keeps its law); its standard error is how far the paths confirm that",
    ),
    (
        "dln_kappa",
        "d ln κ as measured (names' second moment not held)",
        "ln(kappa(beta)/kappa(0)) with kappa = E[D]/sqrt(E[V]) measured under each beta",
    ),
    ("kappa_base", "κ at β = 0", "E_0[D]/sqrt(E_0[V])"),
    (
        "kappa_names_held",
        f"κ at β, {HELD}",
        "E_beta[D]/sqrt(N_0 - E_beta[Rbar^2])",
    ),
)
#: Printed in % with three decimals (log changes), in % with one decimal (the share), else as a level.
LOGS = {
    "dln_ED",
    "wing_estimate",
    "gap_to_wing_estimate",
    "dln_kappa_names_held",
    "short_term_names_held",
    "half_dln_EV_names_held",
    "names_drift",
    "dln_kappa",
}
#: The rows that mean nothing on a date with no shortfall against M_B^listed.
SHORTFALL_ROWS = ("short_term_names_held", "shortfall_closed")
REGION_HEADER = (
    "basket second moment E[(L − 1)²] by region of option strike (the out-of-the-money strip 2∫OTM(K)dK cut at "
    "−3.5, −2.5, −1.5, 0, +1.5 at-the-money sd), model − the index target's strip (1e-6)"
)
REGION_DEFINITION = (
    "part of E[(L - 1)^2] (L the basket level in forward moneyness) carried by the out-of-the-money options struck in the region "
    "(Carr-Madan strip 2 x integral of OTM(K) dK cut at -3.5, -2.5, -1.5, 0, +1.5 at-the-money sd): the model's minus the same part of the strip "
    "of the model's own SVI index surface (cdv_scan.basket_regions, region_second_moments). A region of strike, not of the index return: "
    "a path that ends below -3.5 sd adds to every region below the forward"
)


def p3(v: float, se: float | None = None, digits: int = 3) -> str:
    """A log change in %, signed, with its standard error when given."""
    if se is None:
        return f"{100 * v:+.{digits}f} %"
    return f"{100 * v:+.{digits}f} ± {100 * se:.{digits}f} %"


def share(v: float, se: float | None = None, digits: int = 1) -> str:
    """A share in %."""
    if se is None:
        return f"{100 * v:.{digits}f} %"
    return f"{100 * v:.{digits}f} ± {100 * se:.{digits}f} %"


def fmt(name: str, v: float, se: float) -> str:
    if name == "shortfall_closed":
        return share(v, se)
    if name in LOGS:
        return p3(v, se)
    return f"{v:.4f} ± {se:.4f}"


def split_against(n0: float, r0: float, rb: float, m: float) -> dict[str, float]:
    """The split of the module docstring against a target moment ``m`` (names held)."""
    return {
        "wing_estimate": 0.5 * math.log((n0 - m) / (n0 - r0)),
        "short_term_names_held": 0.5 * math.log((n0 - rb) / (n0 - m)),
        "shortfall_closed": (rb - r0) / (m - r0),
    }


def parts_difference(a: dict[str, Any]) -> tuple[float, float, bool]:
    """``d ln kappa_h`` minus the short term and its standard error from the stored errors.  The
    gap is their sum, so ``cov = (se_gap² − se_k² − se_s²)/2`` and ``se² = 2 se_k² + 2 se_s² −
    se_gap²``; when that is not positive, ``√(se_k² + se_s²)`` (third value ``True``)."""
    sk, ss, sg = (
        a["dln_kappa_names_held_se"],
        a["short_term_names_held_se"],
        a["gap_to_wing_estimate_se"],
    )
    var = 2.0 * sk * sk + 2.0 * ss * ss - sg * sg
    fallback = var <= 0.0
    se = math.sqrt(sk * sk + ss * ss) if fallback else math.sqrt(var)
    return a["dln_kappa_names_held"] - a["short_term_names_held"], se, fallback


def kappa_share(a: dict[str, Any]) -> tuple[float, float]:
    """``d ln kappa_h`` over the gap, with a delta-method error from the stored errors and the
    covariance of the two parts recovered from the gap's error (as :func:`parts_difference`)."""
    k, s, g = a["dln_kappa_names_held"], a["short_term_names_held"], a["gap_to_wing_estimate"]
    vk, vs = a["dln_kappa_names_held_se"] ** 2, a["short_term_names_held_se"] ** 2
    cov = (a["gap_to_wing_estimate_se"] ** 2 - vk - vs) / 2.0
    var = (s * s * vk + k * k * vs - 2.0 * k * s * cov) / g**4
    return k / g, math.sqrt(max(var, 0.0))


def below_share(reg: dict[str, Any]) -> tuple[float, float, float, float]:
    """Of the put strip's remaining shortfall against the index target (the four regions of
    strike below the forward, target minus model): the share in puts struck below −3.5 sd, its
    approximate error (regions treated as independent), the put strip's shortfall and its
    approximate error."""
    short = [-d for d in reg["difference"][:4]]
    ses = reg["model_se"][:4]
    den = sum(short)
    den_se = math.sqrt(sum(s * s for s in ses))
    if den <= 0.0:
        return float("nan"), float("nan"), den, den_se
    a, b = short[0], den - short[0]
    var = (b / den**2 * ses[0]) ** 2 + sum((a / den**2 * s) ** 2 for s in ses[1:])
    return a / den, math.sqrt(var), den, den_se


class Part:
    """What the builder accumulates."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.att_rows: list[dict[str, Any]] = []
        self.reg_rows: list[dict[str, Any]] = []


def one_date(date: str, budget: str, path: Path, part: Part) -> tuple[list[str], str, str]:
    """The tables of one date (Markdown lines), its bullet of the answer and its part of the
    answer's lead sentence."""
    doc = json.loads(path.read_text())
    if doc["budget"] != budget:
        raise ValueError(f"{path}: budget {doc['budget']}, expected {budget}")
    meta = doc["record"]
    rows = {float(r["beta"]): r for r in doc["rows"]}
    atts = {float(a["beta"]): a for a in doc["attribution_from_beta_0"]}
    all_betas, betas = sorted(rows), sorted(atts)
    g_max = sorted({float(r["g_max"]) for r in doc["rows"]})
    src, commit = str(path), meta["git_commit"]
    bud = f"{meta['n_particles']:.0e} particles / {meta['n_paths']:.0e} paths".replace("e+0", "e")
    proto = f"g_max {g_max}; prototype branch cross-dependent-vol"
    md: list[str] = []
    add = md.append

    def rec(suffix: str, quantity: str, v: float | None, se: float | None, unit: str, definition: str, notes: str, **kw: str) -> None:  # fmt: skip
        part.records.append(pc.record(
            f"A.check_e.{date}.{suffix}", "A", quantity, v, se, date=date, unit=unit, definition=definition, budget=bud,
            commit=kw.get("commit", commit), source=kw.get("source", src), notes=notes,
        ))  # fmt: skip

    # --- the M12 row of the same specification (E_CC[D], and its own E[D])
    row_path = pc.LC_OUT / "rows" / ROW_DIRS[budget] / f"{date}.json"
    m12 = json.loads(row_path.read_text())
    if m12["spec_key"] != meta["spec_key"]:
        raise ValueError(f"{row_path}: another specification key than the scan's")
    ed_cc, ed_cc_se, row_commit = float(m12["ED_cc"]), float(m12["ED_cc_se"]), m12["git_commit"]
    stored_cc = doc.get("E_CC_D_of_the_M12_row")
    if stored_cc is not None and abs(stored_cc - ed_cc) > 1e-15:
        raise ValueError(f"{row_path}: E_CC[D] is not the one the scan used")
    row_rel = row_path.relative_to(pc.LC_OUT.parent.parent)

    # --- the two targets and the bridge
    r0 = rows[0.0]
    m_b, n0, rbar0 = float(doc["M_B_listed"]), float(r0["sum_w_ER2"]), float(r0["E_Rbar2"])
    m_t = float(r0["E_Rbar2_of_target"])
    if any(abs(r["E_Rbar2_of_target"] - m_t) > 1e-15 for r in doc["rows"]):
        raise ValueError(f"{path}: the index target's moment differs between the rows")
    rel = m_t / m_b - 1.0
    level_gap = max(abs(r["E_Rbar2_of_model_level"] - r["E_Rbar2"]) for r in doc["rows"])
    f_b = float(r0["basket_forward"])
    add(
        f"Two targets (units of E[R̄²], 1e-6): M_B^listed = {1e6 * m_b:.1f} (the study's listed index second moment, Σ w M_i − EQV); "
        f"the model's own index target = {1e6 * m_t:.1f} (the strip of the SVI index surface the model is calibrated to, (F_B − 1)² + F_B²·E[(L − 1)²], F_B = {f_b:.4f}): "
        f"{100 * rel:+.2f} % against M_B^listed."
    )
    add("")
    rec("bridge.M_B_listed", "check (e): M_B^listed, the study's listed index second moment", m_b, None, "units of squared return",
        "sum_w M_i - EQV of the study's entry (field M_B_listed)", "an input of the study, not a Monte Carlo estimate")  # fmt: skip
    rec("bridge.index_target", "check (e): second moment of the model's own index target, in the units of E[Rbar^2]", m_t, None, "units of squared return",
        "(F_B - 1)^2 + F_B^2 x the out-of-the-money strip of the model's SVI index surface (row field E_Rbar2_of_target)", "a quadrature of the target surface, not a Monte Carlo estimate")  # fmt: skip
    rec("bridge.index_target_over_listed_minus_1", "check (e): the index target's second moment over M_B^listed, minus 1", rel, None, "ratio - 1",
        "E_Rbar2_of_target/M_B_listed - 1", "two inputs, not a Monte Carlo estimate")  # fmt: skip
    add("| second moment of the basket (1e-6) | " + " | ".join(f"β = {b:g}" for b in all_betas) + " |")  # fmt: skip
    add("|:--|" + "--:|" * len(all_betas))
    bridge = (
        ("E_Rbar2", "E[R̄²] under the model", lambda r: r["E_Rbar2"], "E[Rbar^2] under the model (row field E_Rbar2)"),
        ("short_of_listed", "M_B^listed − E[R̄²]", lambda r: m_b - r["E_Rbar2"], "M_B^listed - E[Rbar^2]: the shortfall against the study's listed index second moment"),
        ("short_of_index_target", "index target − E[R̄²]", lambda r: m_t - r["E_Rbar2"], "E_Rbar2_of_target - E[Rbar^2]: the shortfall against the model's own index target"),
    )  # fmt: skip
    for key, label, fn, definition in bridge:
        cells = []
        for b in all_betas:
            v, se = fn(rows[b]), rows[b]["E_Rbar2_se"]
            cells.append(f"{1e6 * v:.1f} ± {1e6 * se:.1f}")
            rec(f"beta{b:g}.bridge.{key}", f"check (e): {label}, beta = {b:g}", v, se, "units of squared return", definition, f"the se is that of E[Rbar^2] (row field E_Rbar2_se); {proto}")  # fmt: skip
        add(f"| {label} | " + " | ".join(cells) + " |")
    add("")

    # --- the core set
    def flagged(r: dict[str, Any]) -> bool:
        return max(r["clip_high_inner"], r["clip_low_inner"]) > FLAG_LEVEL

    cc_label = (
        f"E[D]/E_CC[D] (E_CC[D] of the {budget} row `{row_rel}`, commit {row_commit}; unpaired)"
    )
    cc_note = f"E_CC[D] {ed_cc:.6f} ± {ed_cc_se:.6f} of the M12 row of the same specification key ({row_path}, commit {row_commit}); unpaired: delta-method error from the two standard errors taken as independent (pm_common.ratio_se)"
    idx_def = "basket implied vol at the horizon minus the model's SVI index target at the strike, in vol points (row field idx_<k>, strikes in at-the-money sd)"
    #: key, label, value and se of a row, digits, definition, notes
    core: list[tuple[str, str, Any, int, str, str]] = [
        ("clip_high_inner", "clipped mass inside ±2.5 sd, at the cap (% of particles)", lambda r: (100.0 * r["clip_high_inner"], None), 3, "largest share of particles inside ±2.5 sd whose correlation is clipped at the cap (row field clip_high_inner)", "a calibration diagnostic, no standard error"),
        ("clip_low_inner", "clipped mass inside ±2.5 sd, at λ = 0 (% of particles)", lambda r: (100.0 * r["clip_low_inner"], None), 3, "the same at lambda = 0 (row field clip_low_inner)", "a calibration diagnostic, no standard error"),
        ("flagged", "flagged (clipped mass above 1 % inside ±2.5 sd, decision 5's rule)", lambda r: (1.0 if flagged(r) else 0.0, None), 0, "1 when clip_high_inner or clip_low_inner is above 0.01, else 0", "a rule on a calibration diagnostic, no standard error"),
        ("ED", "E[D] (notional)", lambda r: (r["ED"], r["ED_se"]), 6, "Palladium forward E[D] (row field ED)", ""),
        ("ED_over_cc", cc_label, lambda r: (r["ED"] / ed_cc, pc.ratio_se(r["ED"], r["ED_se"], ed_cc, ed_cc_se)), 5, "E[D] of the scan over E_CC[D] of the M12 row", cc_note),
        ("kappa", "κ as measured", lambda r: (r["kappa"], r["kappa_se"]), 4, "E[D]/sqrt(E[V]) measured under this beta (row fields kappa, kappa_se); the names' second moment is not held", ""),
        ("EV_over_EQV", "E[V]/EQV", lambda r: (r["EV_over_EQV"], r["EV_over_EQV_se"]), 4, "E[V] under the model over the listed EQV (row field EV_over_EQV)", ""),
        ("names_part", "split of E[V] − EQV: single-name part", lambda r: (r["names_part"], r["names_part_se"]), 6, "sum_w E[R_i^2] - sum_w M_i (row field names_part); E[V] - EQV = single-name part - basket part", ""),
        ("basket_part", "split of E[V] − EQV: basket part, E[R̄²] − M_B^listed", lambda r: (r["basket_part"], r["basket_part_se"]), 6, "E[Rbar^2] - M_B^listed (row field basket_part); E[V] - EQV = single-name part - basket part", ""),
        ("names_over_listed", "names' second moment over the listed strips, minus 1 (%)", lambda r: (100.0 * r["names_over_listed"], 100.0 * r["names_part_se"] / doc["sum_w_M"]) if "names_over_listed" in r else (None, None), 2, "100 x (sum_w E[R_i^2]/sum_w M_i - 1) (row field names_over_listed)", "se = names_part_se/sum_w_M"),
        ("E_Rbar2_over_listed", "E[R̄²]/M_B^listed", lambda r: (r["E_Rbar2_over_listed"], r["E_Rbar2_over_listed_se"]), 4, "E[Rbar^2] under the model over the study's listed index second moment (row field E_Rbar2_over_listed)", ""),
    ]  # fmt: skip
    for k in ("-2.5", "-3.0", "-3.5", "+2.5"):
        core.append((f"idx_{k}", f"index error at {k.replace('-', '−')} sd (vol points)", lambda r, k=k: (r[f"idx_{k}"], r[f"idx_{k}_se"]), 3, idx_def, ""))  # fmt: skip
    add("| quantity | " + " | ".join(f"β = {b:g}" for b in all_betas) + " |")
    add("|:--|" + "--:|" * len(all_betas))
    for key, label, fn, digits, definition, notes in core:
        cells = []
        for b in all_betas:
            v, se = fn(rows[b])
            if key == "ED_over_cc" and rows[b].get("ED_over_cc") is not None:
                if abs(rows[b]["ED_over_cc"] - v) > 1e-12:
                    raise ValueError(
                        f"{path}: ED_over_cc of beta {b:g} is not E[D]/E_CC[D] of the row"
                    )
                v = rows[b]["ED_over_cc"]
            cells.append(("yes" if v else "no") if key == "flagged" else pc.pm(v, se, digits))
            rec(f"beta{b:g}.core.{key}", f"CDV scan, {label}, beta = {b:g}", v, se, "", f"{definition}; beta = 0 is the local correlation model on the scan's paths",
                "; ".join(x for x in (f"g_max {g_max}", notes) if x),
                **({"commit": f"{commit} (scan), {row_commit} (row of E_CC[D])", "source": f"{src}; {row_path}"} if key == "ED_over_cc" else {}))  # fmt: skip
        add(f"| {label} | " + " | ".join(cells) + " |")
    add("")
    d_row = r0["ED"] - float(m12["ED_lc"])
    rec("beta0.ED_minus_M12_row", "check (e): E[D] of the scan at beta = 0 minus E_LC[D] of the M12 row", d_row, None, "notional", "row field ED at beta = 0 minus ED_lc of the M12 row of the same specification key",
        f"other pricing paths; no error of the difference is computed (scan ± {r0['ED_se']:.6f}, row ± {float(m12['ED_lc_se']):.6f}); {d_row / r0['ED_se']:+.2f} of the scan's standard error",
        commit=f"{commit} (scan), {row_commit} (row)", source=f"{src}; {row_path}")  # fmt: skip
    notes_core = [
        f"β = 0 of the scan is the local correlation model in the prototype's own pricing pass (other pricing paths than the M12 row's): its E[D] is {1e6 * d_row:+.1f} (1e-6) from the row's {float(m12['ED_lc']):.6f} ± {float(m12['ED_lc_se']):.6f} "
        f"({d_row / r0['ED_se']:+.1f} of the scan's standard error {r0['ED_se']:.6f}; no error of the difference is computed)."
    ]
    up = [b for b in all_betas if abs(rows[b]["idx_+2.5"]) > 3.0 * rows[b]["idx_+2.5_se"]]
    if up:
        notes_core.append(
            "Upside: the index error at +2.5 sd is more than three standard errors from zero at "
            + ", ".join(f"β = {b:g} ({rows[b]['idx_+2.5']:+.3f} ± {rows[b]['idx_+2.5_se']:.3f} vol points)" for b in up)
            + f" (β = 0: {r0['idx_+2.5']:+.3f} ± {r0['idx_+2.5_se']:.3f})."
        )  # fmt: skip
    add(
        " ".join(notes_core)
        + " Index errors: at the horizon, the model minus its own SVI index target."
    )
    add("")

    # --- the attribution against M_B^listed (stored fields) and the same against the index target
    no_shortfall = all(atts[b]["wing_estimate"] > -2.0 * atts[b]["wing_estimate_se"] for b in betas)
    target_unresolved = m_t - rbar0 <= 2.0 * r0["E_Rbar2_se"]
    add("| term (against M_B^listed; ± standard error) | " + " | ".join(f"β = {b:g}" for b in betas) + " |")  # fmt: skip
    add("|:--|" + "--:|" * len(betas))
    point: dict[float, dict[str, float]] = {}
    for b in betas:
        a, rb = atts[b], float(rows[b]["E_Rbar2"])
        again = split_against(n0, rbar0, rb, m_b)
        for name, v in again.items():
            if abs(v - a[name]) > 1e-9:
                raise ValueError(
                    f"{path}: {name} at beta {b:g} is not reproduced from the rows ({v} against {a[name]})"
                )
        if (
            abs(a["gap_to_wing_estimate"] - a["dln_kappa_names_held"] - a["short_term_names_held"])
            > 1e-9
        ):
            raise ValueError(f"{path}: the gap is not the sum of its two parts at beta {b:g}")
        point[b] = split_against(n0, rbar0, rb, m_t)
        point[b]["gap_to_wing_estimate"] = a["dln_ED"] - point[b]["wing_estimate"]
    for name, label, definition in TERMS:
        hide = no_shortfall and name in SHORTFALL_ROWS
        cells = []
        for b in betas:
            v, se = atts[b][name], atts[b][f"{name}_se"]
            cells.append(NO_SHORTFALL if hide else fmt(name, v, se))
            unit = (
                "share"
                if name == "shortfall_closed"
                else ("log change" if name in LOGS else "ratio")
            )
            note = proto + (
                "; no shortfall against M_B^listed on this date (the wing estimate is within two standard errors of zero or positive): not printed in the Markdown"
                if hide
                else ""
            )
            if not (hide and name == "shortfall_closed"):
                rec(
                    f"beta{b:g}.{name}",
                    f"check (e): {label}, beta = {b:g}",
                    v,
                    se,
                    unit,
                    definition,
                    note,
                )
            part.att_rows.append({"date": date, "budget": budget, "beta": b, "target": "M_B_listed", "term": name, "label": label, "value": v, "se": se,
                                  "flag": "no shortfall against M_B^listed: not printed" if hide else ""})  # fmt: skip
        add(f"| {label} | " + " | ".join(cells) + " |")
    add("| κ at β, as measured | " + " | ".join(f"{rows[b]['kappa']:.4f} ± {rows[b]['kappa_se']:.4f}" for b in betas) + " |")  # fmt: skip
    add("")
    add(f"| the same split against the model's own index target ({POINT}) | " + " | ".join(f"β = {b:g}" for b in betas) + " |")  # fmt: skip
    add("|:--|" + "--:|" * len(betas))
    unresolved = (
        "n/a (shortfall against the index target within two standard errors of zero at β = 0)"
    )
    point_rows = (
        ("wing_estimate", "fixed-κ wing estimate against the index target", "(1/2) ln((N_0 - m)/(N_0 - E_0[Rbar^2])), m the index target's moment E_Rbar2_of_target, N_0 = sum_w E_0[R_i^2]"),
        ("gap_to_wing_estimate", "gap: move minus estimate", "d ln E[D] - wing estimate against the index target"),
        ("dln_kappa_names_held", f"of which κ ({HELD}; the same as above)", ""),
        ("short_term_names_held", f"of which the basket still short of the index target ({HELD})", "(1/2) ln((N_0 - E_beta[Rbar^2])/(N_0 - m)), m the index target's moment E_Rbar2_of_target"),
        ("shortfall_closed", "share of the shortfall against the index target closed", "(E_beta[Rbar^2] - E_0[Rbar^2])/(m - E_0[Rbar^2]), m the index target's moment E_Rbar2_of_target"),
    )  # fmt: skip
    for name, label, definition in point_rows:
        hide = target_unresolved and name in SHORTFALL_ROWS
        cells = []
        for b in betas:
            if name == "dln_kappa_names_held":
                cells.append(p3(atts[b][name]))
                continue
            v = point[b][name]
            cells.append(
                unresolved if hide else (share(v) if name == "shortfall_closed" else p3(v))
            )
            if not hide:
                rec(f"beta{b:g}.index_target.{name}", f"check (e): {label}, beta = {b:g}", v, None, "share" if name == "shortfall_closed" else "log change", definition,
                    f"point value, no standard error computed: a closed form of the rows' moments (sum_w_ER2 at beta = 0, E_Rbar2, E_Rbar2_of_target) and of d ln E[D]; the same formula with m = M_B^listed returns the stored field to 1e-9; {proto}")  # fmt: skip
            part.att_rows.append({"date": date, "budget": budget, "beta": b, "target": "index_target", "term": name, "label": label, "value": v, "se": None,
                                  "flag": "shortfall against the index target within two standard errors of zero: not printed" if hide else "point value, no standard error computed"})  # fmt: skip
        add(f"| {label} | " + " | ".join(cells) + " |")
    add("")

    # --- kappa against the basket still short; kappa as measured
    compare: dict[float, dict[str, Any]] = {}
    lines = []
    for b in betas:
        a = atts[b]
        diff, dse, fallback = parts_difference(a)
        gap_open = abs(a["gap_to_wing_estimate"]) > 2.0 * a["gap_to_wing_estimate_se"]
        k, s = a["dln_kappa_names_held"], a["short_term_names_held"]
        c: dict[str, Any] = {
            "diff": diff,
            "se": dse,
            "fallback": fallback,
            "gap_open": gap_open,
            "split": not no_shortfall and gap_open and k > 0.0 and s > 0.0,
        }
        c["measured_resolved"] = abs(a["dln_kappa"]) > 2.0 * a["dln_kappa_se"]
        c["both"] = (
            c["split"]
            and k > 2.0 * a["dln_kappa_names_held_se"]
            and s > 2.0 * a["short_term_names_held_se"]
        )
        c["k_whole"] = a["kappa_base"] * math.exp(a["gap_to_wing_estimate"])
        if c["split"]:
            if abs(diff) > 3.0 * dse:
                c["verdict"] = (
                    "κ is the larger part"
                    if diff > 0.0
                    else "the basket still short is the larger part"
                )
            elif abs(diff) > 2.0 * dse:
                c["verdict"] = "the two parts are of the same order"
            else:
                c["verdict"] = "the two parts are not separated"
            c[
                "verdict"
            ] += f" (difference {p3(diff, dse)}{', error without the covariance' if fallback else ''})"
            c["share"], c["share_se"] = kappa_share(a)
            c["share_t"] = k / point[b]["gap_to_wing_estimate"]
            rec(f"beta{b:g}.kappa_share_of_gap", f"check (e): share of the gap to the wing estimate (against M_B^listed) due to kappa ({HELD}), beta = {b:g}", c["share"], None, "share",
                "d ln kappa (names held) / (d ln E[D] - wing estimate)", f"ratio of two fields of the scan; approximate delta-method error {c['share_se']:.4f} from the stored errors with the covariance of the two parts recovered from the gap's error (not stored as se)")  # fmt: skip
            rec(f"beta{b:g}.index_target.kappa_share_of_gap", f"check (e): share of the gap to the wing estimate against the index target due to kappa ({HELD}), beta = {b:g}", c["share_t"], None, "share",
                "d ln kappa (names held) / (d ln E[D] - wing estimate against the index target)", "point value, no standard error computed")  # fmt: skip
            rec(f"beta{b:g}.kappa_if_whole_gap", f"check (e): kappa if the whole gap to the wing estimate (against M_B^listed) were kappa, beta = {b:g}", c["k_whole"], None, "ratio",
                "kappa(0) x exp(d ln E[D] - wing estimate)", "a function of two stored fields; no error computed (the owner's bracket: about 0.758 today at beta = 3)")  # fmt: skip
        rec(f"beta{b:g}.kappa_minus_short", f"check (e): the kappa part minus the basket still short of M_B^listed ({HELD}), beta = {b:g}", diff, dse, "log change",
            "d ln kappa (names held) - short term (names held)", ("se = sqrt(se_k^2 + se_s^2): the recovered variance was not positive" if fallback else "se^2 = 2 se_k^2 + 2 se_s^2 - se_gap^2 (the gap is the sum of the two parts)") + ("; no shortfall against M_B^listed on this date" if no_shortfall else ""))  # fmt: skip
        compare[b] = c
        c["measured"] = (
            f"as measured κ {rows[b]['kappa']:.4f} ± {rows[b]['kappa_se']:.4f}, d ln κ {p3(a['dln_kappa'], a['dln_kappa_se'])}"
        )
        c["measured"] += (
            ""
            if c["measured_resolved"]
            else " (not resolved as measured: within two standard errors of zero)"
        )
        drift = f"measured drift of the names' second moment {p3(a['names_drift'], a['names_drift_se'])}"
        drift += (
            " (more than two standard errors from zero)"
            if abs(a["names_drift"]) > 2.0 * a["names_drift_se"]
            else ""
        )
        head = f"β = {b:g}{' (flagged)' if flagged(rows[b]) else ''}: "
        if no_shortfall:
            lines.append(
                f"{head}κ ({HELD}) {p3(k, a['dln_kappa_names_held_se'])}; {c['measured']}; {drift}."
            )
        elif c["split"]:
            lines.append(f"{head}κ part minus still short (against M_B^listed, {HELD}) {p3(diff, dse)}, {abs(diff) / dse:.1f} standard errors: {c['verdict'].split(' (')[0]}; "
                         f"against the index target {p3(k - point[b]['short_term_names_held'])} (point value); {c['measured']}; {drift}.")  # fmt: skip
        else:
            lines.append(f"{head}gap {p3(a['gap_to_wing_estimate'], a['gap_to_wing_estimate_se'])}{'' if gap_open else ', within two standard errors of zero'}: κ ({HELD}) {p3(k, a['dln_kappa_names_held_se'])} against the basket still short {p3(s, a['short_term_names_held_se'])}; "
                         f"no share of the gap is given; {c['measured']}; {drift}.")  # fmt: skip
    add("κ and the basket still short. " + " ".join(lines) + " The rule: \"larger part\" is written above three standard errors of the difference, \"of the same order\" between two and three.")  # fmt: skip
    add("")

    # --- the regions of option strike
    labels = r0["regions"]["labels"]
    add(f"| {REGION_HEADER} | " + " | ".join(f"struck {lab}" for lab in labels) + " | total |")
    add("|:--|" + "--:|" * (len(labels) + 1))
    for b in all_betas:
        reg = rows[b]["regions"]
        total, total_se = reg["total_model"] - reg["total_target"], reg["total_model_se"]
        add(f"| β = {b:g} | " + " | ".join(f"{1e6 * d:+.0f} ± {1e6 * s:.0f}" for d, s in zip(reg["difference"], reg["model_se"], strict=True)) + f" | {1e6 * total:+.0f} ± {1e6 * total_se:.0f} |")  # fmt: skip
        for lab, d, s in [
            *zip(labels, reg["difference"], reg["model_se"], strict=True),
            ("total", total, total_se),
        ]:
            part.reg_rows.append({"date": date, "budget": budget, "beta": b, "region": lab, "model_minus_target": d, "se": s})  # fmt: skip
            rec(f"beta{b:g}.region.{lab}", f"check (e): basket second moment E[(L-1)^2], options struck {lab}, model minus the index target's strip, beta = {b:g}", d, s, "units of squared level (forward moneyness)",
                REGION_DEFINITION if lab != "total" else "E[(L - 1)^2] of the model minus the whole strip of the model's own SVI index surface (regions total_model - total_target)", "the se is the model's; the target's strip is a quadrature")  # fmt: skip
    shares: dict[float, tuple[float, float, float, float]] = {}
    said: list[str] = []
    unsaid: list[str] = []
    for b in all_betas:
        shares[b] = below_share(rows[b]["regions"])
        sh, sh_se, den, den_se = shares[b]
        if not den > 2.0 * den_se:
            unsaid.append(f"β = {b:g} ({1e6 * den:.0f} ± {1e6 * den_se:.0f})")
        else:
            said.append(f"β = {b:g}: {share(sh, sh_se, 0)}")
            if b in atts:
                rec(f"beta{b:g}.remaining_below_m35", f"check (e): share of the remaining shortfall of the put strip against the index target that is in puts struck below -3.5 sd, beta = {b:g}", sh, sh_se, "share",
                    "(target - model) of the region of strike below -3.5 sd over (target - model) summed over the four regions of strike below the forward", "approximate error: the four regions treated as independent (they are computed on the same paths)")  # fmt: skip
    tail = f"The totals are in units of the level L; in the units of E[R̄²] they are multiplied by F_B² = {f_b * f_b:.4f}, and the level's moment is within {1e6 * level_gap:.1f} (1e-6) of the measured E[R̄²] (the names' carries differ)."
    text = ""
    if said:
        text += "Share of the remaining shortfall of the put strip (the four regions of strike below the forward, against the index target) that is in puts struck below −3.5 sd, ± an approximate error (regions treated as independent): " + "; ".join(said) + ". "  # fmt: skip
    if unsaid:
        text += "No share is given where the put strip's shortfall (1e-6, approximate error) is within two standard errors of zero: " + ", ".join(unsaid) + ". "  # fmt: skip
    add("")
    add(text + tail)
    add("")

    # --- the bullet of the answer and this date's part of the lead sentence
    def idx(b: float, k: str) -> str:
        return f"{rows[b][f'idx_{k}']:+.3f} ± {rows[b][f'idx_{k}_se']:.3f}"

    def flag(b: float) -> str:
        return " (flagged calibration)" if flagged(rows[b]) else ""

    def measured(b: float) -> str:
        text = f"κ {rows[b]['kappa']:.4f} ± {rows[b]['kappa_se']:.4f}, d ln κ {p3(atts[b]['dln_kappa'], atts[b]['dln_kappa_se'])}"
        return text if compare[b]["measured_resolved"] else text + " (not resolved as measured)"

    def held(b: float) -> str:
        a = atts[b]
        return f"κ ({HELD}) {p3(a['dln_kappa_names_held'], a['dln_kappa_names_held_se'])} ({a['kappa_base']:.4f} → {a['kappa_names_held']:.4f} ± {a['kappa_names_held_se']:.4f})"

    def short(b: float) -> str:
        return f"basket still short of M_B^listed {p3(atts[b]['short_term_names_held'], atts[b]['short_term_names_held_se'])}"

    def gap(b: float) -> str:
        a = atts[b]
        return f"gap {p3(a['gap_to_wing_estimate'], a['gap_to_wing_estimate_se'])}" + (
            "" if compare[b]["gap_open"] else " (within two standard errors of zero)"
        )

    def closed(b: float) -> str:
        return f"{share(atts[b]['shortfall_closed'], atts[b]['shortfall_closed_se'])} of the shortfall against M_B^listed closed"

    def against_target(b: float) -> str:
        t = point[b]
        text = (
            f"gap {p3(t['gap_to_wing_estimate'])}, κ ({HELD}) {p3(atts[b]['dln_kappa_names_held'])}"
        )
        if compare[b]["split"]:
            text += f" ({share(compare[b]['share_t'], None, 0)} of that gap)"
        return (
            text
            + f", basket still short of the index target {p3(t['short_term_names_held'])}, {share(t['shortfall_closed'], None, 0)} of that shortfall closed"
        )

    if no_shortfall:
        b0 = betas[0]
        w, w_se = atts[b0]["wing_estimate"], atts[b0]["wing_estimate_se"]
        wing_short = all(
            rows[0.0][f"idx_{k}"] < -2.0 * rows[0.0][f"idx_{k}_se"]
            for k in ("-2.5", "-3.0", "-3.5")
        )
        lead = f"on {date} there is no shortfall against M_B^listed and the move of the forward is κ and the E[V] term"
        bullet = (
            f"*{date} ({budget} budget)* — no shortfall against M_B^listed: E[R̄²]/M_B^listed = {r0['E_Rbar2_over_listed']:.4f} ± {r0['E_Rbar2_over_listed_se']:.4f} at β = 0 "
            f"(wing estimate {p3(w, w_se)}, {'within two standard errors of zero' if abs(w) <= 2.0 * w_se else 'positive'}); the move of the forward is κ and the E[V] term. "
            + " ".join(
                f"β = {b:g}{flag(b)}: forward {p3(atts[b]['dln_ED'], atts[b]['dln_ED_se'])} = {held(b)} + ½ d ln E[V] ({HELD}) {p3(atts[b]['half_dln_EV_names_held'], atts[b]['half_dln_EV_names_held_se'])}; as measured {measured(b)}."
                for b in betas
            )
            + f" Against the model's own index target the total is short by {1e6 * (m_t - rbar0):.0f} ± {1e6 * r0['E_Rbar2_se']:.0f} (1e-6) at β = 0"
            + (" (within two standard errors of zero)" if target_unresolved else "")
            + (", and the downside wing is short at β = 0: " if wing_short else "; the downside wing at β = 0: ")
            + "options struck below −1.5 sd "
            + ", ".join(f"{1e6 * d:+.0f} ± {1e6 * s:.0f}" for d, s in zip(r0["regions"]["difference"][:3], r0["regions"]["model_se"][:3], strict=True))
            + f" (1e-6; the three regions of strike), index errors {idx(0.0, '-2.5')}, {idx(0.0, '-3.0')}, {idx(0.0, '-3.5')} vol points at −2.5, −3.0, −3.5 sd, {100 * r0['clip_high_inner']:.1f} % of the particles clipped at the cap inside ±2.5 sd. "
            + f"At β = {b0:g} the same are "
            + ", ".join(f"{1e6 * d:+.0f} ± {1e6 * s:.0f}" for d, s in zip(rows[b0]["regions"]["difference"][:3], rows[b0]["regions"]["model_se"][:3], strict=True))
            + f"; {idx(b0, '-2.5')}, {idx(b0, '-3.0')}, {idx(b0, '-3.5')}; {100 * rows[b0]['clip_high_inner']:.1f} %."
        )  # fmt: skip
    elif date == DATES[0][0]:
        b0 = betas[0]
        a, c, t = atts[b0], compare[b0], point[b0]
        bullet = f"*{date} ({budget} budget; the date of the question)* — β = {b0:g}{flag(b0)}: the forward moved {p3(a['dln_ED'], a['dln_ED_se'])} against a fixed-κ wing estimate of {p3(a['wing_estimate'], a['wing_estimate_se'])}: {gap(b0)}. "
        lead = f"today at β = {b0:g} the split is in the first bullet"
        if c["split"]:
            same_order = (c["diff"] > 0.0) == (
                a["dln_kappa_names_held"] > t["short_term_names_held"]
            )
            robust = abs(c["diff"]) > 3.0 * c["se"] and same_order
            thirds = robust and 0.6 <= c["share"] <= 0.73 and 0.6 <= c["share_t"] <= 0.73
            bullet += (
                f"{'Both reasons hold. ' if c['both'] else ''}Against M_B^listed: {held(b0)} and the {short(b0)}; κ is {share(c['share'], c['share_se'], 0)} of the gap (approximate error); {c['verdict']}; {closed(b0)}. "
                f"The owner's bracket: \"about 0.758\" is κ(0)·exp(gap) = {c['k_whole']:.4f}, the value if the basket were not short and the whole gap were κ; with the {HELD} κ is {a['kappa_names_held']:.4f} ± {a['kappa_names_held_se']:.4f}. "
                f"Against the model's own index target ({POINT}): wing estimate {p3(t['wing_estimate'])}, {against_target(b0)}. "
            )
            if thirds:
                bullet += f"The answer does not depend on the target: κ ({HELD}) is about two thirds of the gap under both. "
                lead = f"today at β = {b0:g} {'both reasons hold: ' if c['both'] else ''}with the {HELD}, κ is about two thirds of the gap to the fixed-κ estimate and the basket still short of the target is the rest, against either target"
            elif robust:
                bullet += "The ordering of the two parts does not depend on the target. "
                lead = f"today at β = {b0:g}, with the {HELD}, the two parts of the gap have the same ordering against either target"
            else:
                bullet += "The ordering of the two parts depends on the target or is not resolved. "
                lead = f"today at β = {b0:g}, with the {HELD}, the ordering of the two parts of the gap depends on the target or is not resolved"
        else:
            bullet += f"Against M_B^listed: {held(b0)}; {short(b0)}; {closed(b0)}. Against the model's own index target ({POINT}): {against_target(b0)}. "
        bullet += f"As measured (names' second moment not held): {measured(b0)}."
        if shares[b0][2] > 2.0 * shares[b0][3]:
            bullet += f" Of the remaining shortfall of the put strip against the index target, {share(shares[b0][0], shares[b0][1], 0)} (approximate error) is in puts struck below −3.5 sd."
        for b in betas[1:]:
            bullet += f" β = {b:g}{flag(b)}: {gap(b)}; against M_B^listed {held(b)}, {short(b)}, {closed(b)}; as measured {measured(b)}."
    else:
        signs_listed = {b: compare[b]["diff"] > 0.0 for b in betas}
        signs_target = {
            b: atts[b]["dln_kappa_names_held"] > point[b]["short_term_names_held"] for b in betas
        }
        both = all(compare[b]["split"] for b in betas)
        firm = both and all(abs(compare[b]["diff"]) > 3.0 * compare[b]["se"] for b in betas)
        depends = [x for x, cond in (("β", len(set(signs_listed.values())) > 1), ("the target", any(signs_listed[b] != signs_target[b] for b in betas))) if cond]  # fmt: skip
        if firm and not depends:
            verdict = "both parts are positive, with the same ordering of the two at each β and against either target"
        elif both:
            order = (
                "of the same order"
                if all("same order" in compare[b]["verdict"] for b in betas)
                else "not separated at three standard errors"
            )
            verdict = (
                f"{'both reasons hold and ' if all(compare[b]['both'] for b in betas) else ''}the two parts are {order}"
                + (f" (which is larger depends on {' and on '.join(depends)})" if depends else "")
            )
        else:
            verdict = "the split differs between the β (numbers below)"
        lead = f"on {date} {verdict}"
        bullet = (
            f"*{date} ({budget} budget)* — {verdict}. Against M_B^listed, with the {HELD}: "
            + "; ".join(
                f"β = {b:g}{flag(b)}: forward {p3(atts[b]['dln_ED'], atts[b]['dln_ED_se'])}, {gap(b)} = κ {p3(atts[b]['dln_kappa_names_held'], atts[b]['dln_kappa_names_held_se'])} + {short(b)} "
                f"(κ part minus still short {p3(compare[b]['diff'], compare[b]['se'])}), {closed(b)}"
                for b in betas
            )
            + f" (wing estimate {p3(atts[betas[0]]['wing_estimate'], atts[betas[0]]['wing_estimate_se'])}). Against the model's own index target ({POINT}): wing estimate {p3(point[betas[0]]['wing_estimate'])}; "
            + "; ".join(f"β = {b:g}: {against_target(b)} (κ part minus still short {p3(atts[b]['dln_kappa_names_held'] - point[b]['short_term_names_held'])})" for b in betas)
            + ". As measured (names' second moment not held): "
            + "; ".join(f"β = {b:g}: {measured(b)}" for b in betas)
            + "."
        )  # fmt: skip
    return md, bullet, lead


def build() -> None:
    part = Part()
    blocks: list[str] = []
    answers: list[str] = []
    leads: list[str] = []
    for date, budget in DATES:
        path = SCANS / f"cdv_scan_{date}_3m.json"
        blocks += [f"**{date}** ({budget} budget)", ""]
        if not path.exists():
            blocks += [f"pending: `{path.relative_to(pc.LC_OUT.parent.parent)}`", ""]
            continue
        md, bullet, lead = one_date(date, budget, path, part)
        blocks += md
        answers.append(bullet)
        leads.append(lead)
    out = [f'Check (e), the owner\'s question: "{QUESTION}"', ""]
    lead = "; ".join(leads)
    out.append(f"**Answer.** {lead[:1].upper()}{lead[1:]}.")
    out.append("")
    out += [f"- {text}" for text in answers]
    out.append("")
    out.append(
        "How to read the numbers. "
        '(1) Two targets. "Shortfall", "wing estimate", "still short", "shortfall closed" and E[R̄²]/M_B^listed are against M_B^listed = Σ w M_i − EQV, the study\'s listed index second moment (rows with ± standard errors: fields of `cdv_scan.attribution`, paired on the pricing paths, delta method). '
        f"The model is calibrated to its own SVI index surface; the strip of that surface is the regions' target and gives the second table of each date ({POINT}: closed forms of the rows' moments, which return the stored fields to 1e-9 when M_B^listed is put in). The first lines of each date give both moments and their difference. "
        f'(2) κ. With the {HELD} at its β = 0 estimate N₀ (the model keeps each name\'s law, so this moment is the same under every β; the paths confirm it only to the standard error of the row "measured drift"), '
        "d ln E[D] = d ln κ + ½ d ln E[V] and ½ d ln E[V] = wing estimate + still short, so the gap to the wing estimate is the κ part plus the basket still short. The κ as measured (E[D]/√E[V] under each β, the names' second moment not held) is printed beside it; "
        '"not resolved as measured" means the measured d ln κ is within two standard errors of zero. '
        "(3) Regions. The regions are regions of option strike of the out-of-the-money strip of E[(L − 1)²], L the basket level in forward moneyness; a path that ends below −3.5 sd adds to every region below the forward. "
        '(4) Rows "still short" and "shortfall closed" are n/a on a date where the wing estimate is within two standard errors of zero or positive (the raw values are in the attribution CSV with a flag). '
        "Prototype: `scripts/cdv_scan.py`, branch `cross-dependent-vol`, g_max = 3, M12's defaults of 2026-10-09; β = 0 is the local correlation model. Files: `tables/A_check_e_attribution.csv`, `tables/A_check_e_regions.csv`."
    )
    out.append("")
    out += blocks
    pc.save_table(pd.DataFrame(part.att_rows), "A_check_e_attribution")
    pc.save_table(pd.DataFrame(part.reg_rows), "A_check_e_regions")
    pc.write_part("A_check_e", part.records, "\n".join(out))
    log.info("A_check_e written: %d records; answers for %d dates", len(part.records), len(answers))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    build()
