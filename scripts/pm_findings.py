"""PM results package of 2026-10-09: the findings that close ``NUMBERS.md`` (part ``Z_findings``).

    python scripts/pm_findings.py [--strict]

Each finding is one sentence whose numbers are read from the records of the other parts
(``parts/*.json``) by id, so a finding cannot disagree with the tables: no number is typed here.
A record that is missing or has no value is written ``[pending: <id>]`` (and ``--strict`` then
exits with an error: use it for the final assembly).  The only arithmetic done here is
``1 − ratio`` and a difference of two records, each written as its own record with the ids it is
made from.
"""

# ruff: noqa: E501, RUF001
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pm_common as pc

log = logging.getLogger("pm_findings")


class Book:
    """The records of every part but this one, by id; remembers what was missing."""

    def __init__(self) -> None:
        self.rec: dict[str, dict[str, Any]] = {}
        for f in sorted((pc.PM / "parts").glob("*.json")):
            if f.stem in ("Z_findings", "A_risk_today_raw"):
                continue
            for r in json.loads(f.read_text()):
                self.rec[r["id"]] = r
        self.missing: list[str] = []
        self.derived: list[dict[str, Any]] = []

    def get(self, rid: str) -> dict[str, Any] | None:
        r = self.rec.get(rid)
        if r is None or r.get("value") is None:
            self.missing.append(rid)
            return None
        return r

    def v(self, rid: str, digits: int = 4, scale: float = 1.0, sign: bool = False) -> str:
        """The value alone."""
        r = self.get(rid)
        if r is None:
            return f"[pending: {rid}]"
        return f"{scale * r['value']:{'+' if sign else ''}.{digits}f}"

    def pm(self, rid: str, digits: int = 4, scale: float = 1.0, sign: bool = False) -> str:
        """``value ± se`` (the value alone when the record has no error)."""
        r = self.get(rid)
        if r is None:
            return f"[pending: {rid}]"
        out = f"{scale * r['value']:{'+' if sign else ''}.{digits}f}"
        return out if r.get("se") is None else f"{out} ± {abs(scale) * r['se']:.{digits}f}"

    def n(self, rid: str) -> str:
        r = self.get(rid)
        return (
            f"[pending: {rid}]"
            if r is None
            else str(int(r["n"] if r.get("n") is not None else r["value"]))
        )

    def count(self, rid: str) -> str:
        r = self.get(rid)
        return f"[pending: {rid}]" if r is None else str(round(r["value"]))

    def ci(self, rid: str, digits: int = 3) -> str:
        """``value [lo, hi]`` from the records ``rid``, ``rid.ci95_lo`` and ``rid.ci95_hi``."""
        r, lo, hi = self.get(rid), self.get(f"{rid}.ci95_lo"), self.get(f"{rid}.ci95_hi")
        if r is None or lo is None or hi is None:
            return f"[pending: {rid}]"
        return f"{r['value']:.{digits}f} [{lo['value']:.{digits}f}, {hi['value']:.{digits}f}]"

    def discount(self, rid: str, digits: int = 1) -> str:
        """``100·(1 − ratio)`` in per cent, recorded as a derived number."""
        r = self.get(rid)
        if r is None:
            return f"[pending: {rid}]"
        value = 1.0 - r["value"]
        self.derived.append(pc.record(
            f"Z.discount.{rid}", "Z", f"1 minus ({r['quantity']})", value, r.get("se"), date=r.get("date", ""), unit="fraction", definition=f"1 - record {rid}",
            budget=r.get("budget", ""), commit=r.get("commit", ""), source=f"record {rid}", n=r.get("n"), notes="derived in the findings",
        ))  # fmt: skip
        return f"{100 * value:.{digits}f} %"


def findings(b: Book) -> list[str]:
    d0, d1, d2, d3 = "2026-10-02", "2019-09-03", "2017-04-03", "2008-07-07"
    h, e = "C.hist", f"A.check_e.{d0}.beta3"
    g, p, s = "C.arith.gap", "C.arith.payout", "C.arith.split"
    out = []
    out.append(
        f"**Today.** On 2026-10-02 (3m, production budget, defaults of 9 Oct) the local correlation model prices the Palladium forward at E[D] = {b.pm('A.ED.lc', 6)} of notional, "
        f"which is {b.pm('A.ratio.lc_over_cc', 4)} of its constant-correlation companion and {b.pm('A.ratio.lc_over_copula', 4)} of the copula's {b.v('A.ED.copula', 6)} "
        f"(CC/copula {b.pm('A.ratio.cc_over_copula', 4)}; listed-variance forward {b.v('A.ratio.listed_fwd', 4)} of the copula's); the ± are pricing Monte Carlo errors only — the same specification at the development budget gives LC/CC {b.pm('C.b.today_dev.lc_over_cc', 4)} — "
        "and model S did not converge on this date, so today has no valid S/copula."
    )
    out.append(
        f"**Reference dates (3m, production budget).** LC/CC is {b.pm(f'B.{d1}.ratio.lc_over_cc', 4)} on {d1}, {b.pm(f'B.{d2}.ratio.lc_over_cc', 4)} on {d2} and {b.pm(f'B.{d3}.ratio.lc_over_cc', 4)} on {d3}, "
        f"against {b.v(f'B.{d1}.ref.lc_over_cc', 4)}, {b.v(f'B.{d2}.ref.lc_over_cc', 4)} and {b.v(f'B.{d3}.ref.lc_over_cc', 4)} for the parametric reference implementation (another model and other inputs) and "
        f"{b.v(f'B.{d1}.ratio.s_over_copula', 4)}, {b.v(f'B.{d2}.ratio.s_over_copula', 4)} and {b.v(f'B.{d3}.ratio.s_over_copula', 4)} for model S over the copula; "
        f"the calendar repair of the names' slices (decision 1) moved LC/CC by {b.v(f'B.{d1}.new_minus_old.ratio.lc_over_cc', 4, sign=True)} and {b.v(f'B.{d2}.new_minus_old.ratio.lc_over_cc', 4, sign=True)} on the first two dates, "
        f"about forty times the printed Monte Carlo error, and by {b.v(f'B.{d0}.new_minus_old.ratio.lc_over_cc', 4, sign=True)} today."
    )
    out.append(
        f"**History.** On the {b.n(h + '.lc_over_copula.mean.all')} priced dates of the history table (3m, development budget, commit 5b4700b: decisions 1–2 on, decision 5 not; the table holds model S's {b.count(h + '.count.model_s_converged')} converged dates and 2026-10-02, "
        f"{b.count(h + '.count.failed')} failed, and {b.count(h + '.count.study_monthly_absent')} of the study's {b.count(h + '.count.study_monthly')} monthly dates have no row) LC/copula averages {b.pm(h + '.lc_over_copula.mean.all', 4)} "
        f"(median {b.v(h + '.lc_over_copula.median.all', 4)}, pooled {b.v(h + '.lc_over_copula.pooled.all', 4)}), LC/CC {b.pm(h + '.lc_over_cc.mean.all', 4)} and CC/copula {b.pm(h + '.cc_over_copula.mean.all', 4)}, "
        f"against model S/copula {b.pm(h + '.s_over_copula.mean.all', 4)} on the {b.n(h + '.s_over_copula.mean.all')} of them where model S converged (LC/copula {b.v(h + '.lc_over_copula.mean.S', 4)} on the same dates) and a listed-variance forward of {b.pm(h + '.listed_fwd_ratio.mean.all', 4)}: "
        f"LC takes {b.discount(h + '.lc_over_copula.mean.all')} off the copula's forward on average where model S takes {b.discount(h + '.s_over_copula.mean.all')} "
        f"(± = standard error across dates treated as independent; Newey–West at 6 lags, still a lower value: {b.v(h + '.lc_over_copula.mean_nw_se.all', 4)} for LC/copula, {b.v(h + '.s_over_copula.mean_nw_se.all', 4)} for S/copula)."
    )
    out.append(
        f"**Where LC and model S differ.** LC/CC − S/copula is {b.pm(h + '.y_check_d.mean.clip_low', 3, sign=True)} in the lowest tercile of M12's clipped mass, {b.pm(h + '.y_check_d.mean.clip_mid', 3, sign=True)} in the middle one and "
        f"{b.pm(h + '.y_check_d.mean.clip_high', 3, sign=True)} in the highest, and across the {b.n('C.d.S.rel.n')} dates it rises by {b.v('C.d.S.rel.clip_inner_max.coef', 3)} per unit of clipped mass (a fraction of the particles; "
        f"HC1 standard error {b.v('C.d.S.rel.clip_inner_max.se_hc1', 3)}, R² {b.v('C.d.S.rel.r2', 2)}): the gap sits on the dates where LC's correlation is clipped, and it is model S's discount that grows there "
        f"(S/copula {b.v(h + '.s_over_copula.mean.clip_low', 3)}, {b.v(h + '.s_over_copula.mean.clip_mid', 3)}, {b.v(h + '.s_over_copula.mean.clip_high', 3)} by tercile) while LC/CC does not move with the clipped mass "
        f"({b.v(h + '.lc_over_cc.mean.clip_low', 3)}, {b.v(h + '.lc_over_cc.mean.clip_mid', 3)}, {b.v(h + '.lc_over_cc.mean.clip_high', 3)}) — an association across dates, not a demonstration that the clipping is the cause."
    )
    out.append(
        f"**Stratified test of the cross-dependent volatility prototype (check f; development budget).** On the {b.n('D.high.lc_over_copula.mean')} dates where M12's clipped mass is largest, β = 3 and β = 6 take the forward over the copula's from {b.pm('D.high.lc_over_copula.mean', 4)} "
        f"to {b.pm('D.high.cdv3_over_copula.mean', 4)} and {b.pm('D.high.cdv6_over_copula.mean', 4)} (model S {b.pm('D.high.s_over_copula.mean', 4)}; listed-variance forward {b.pm('D.high.listed_fwd.mean', 4)}; means across the group's dates ± their standard error), "
        f"about a quarter and about a half of the distance to model S; but β was imposed, not calibrated, and the same β moves the forward on the {b.n('D.low.lc_over_copula.mean')} least-clipped dates: per date, CDV/CC − LC/CC is "
        f"{b.pm('D.low.cdv3_minus_lc_over_cc.mean', 4, sign=True)} and {b.pm('D.low.cdv6_minus_lc_over_cc.mean', 4, sign=True)} there against {b.pm('D.high.cdv3_minus_lc_over_cc.mean', 4, sign=True)} and {b.pm('D.high.cdv6_minus_lc_over_cc.mean', 4, sign=True)} on the binding dates, "
        f"so the part specific to the binding dates is {b.pm('D.high_minus_low.cdv3_minus_lc_over_cc', 4, sign=True)} at β = 3 ({b.pm('D.high_minus_low_unflagged.cdv3_minus_lc_over_cc', 4, sign=True)} without the two flagged dates) and "
        f"{b.pm('D.high_minus_low.cdv6_minus_lc_over_cc', 4, sign=True)} at β = 6 ({b.pm('D.high_minus_low_unflagged.cdv6_minus_lc_over_cc', 4, sign=True)}): the level moves towards model S on the binding dates, and the test does not give what a β calibrated date by date would."
    )
    out.append(
        f"**Check (e), today, production budget.** Closing the index wing with the prototype at β = 3 moves the forward by {b.pm(e + '.dln_ED', 2, 100, True)} %, not the {b.pm(e + '.wing_estimate', 2, 100, True)} % of the fixed-κ estimate: "
        f"κ rises ({b.pm(e + '.dln_kappa_names_held', 2, 100, True)} %, from {b.v(e + '.kappa_base', 4)} to {b.pm(e + '.kappa_names_held', 4)} with the names' second moment held at its β = 0 value; as measured {b.pm(e + '.core.kappa', 4)}) "
        f"and the basket's second moment is still short of the listed one ({b.pm(e + '.short_term_names_held', 2, 100, True)} %; {b.pm(e + '.shortfall_closed', 0, 100)} % of the shortfall closed), "
        "so today at β = 3 both reasons hold and κ is the larger; the proportions depend on the date and on β (A9)."
    )
    out.append(
        f"**Risk today (production budget).** The common delta is {b.pm('A.delta.lc_ss', 3, sign=True)} % of E_LC[D] per +1 % on every spot under LC against {b.pm('A.delta.cc_ss', 3, sign=True)} % of E_CC[D] under CC "
        f"(the model's own sticky strike: local vols in absolute spot, λ in absolute basket level, no recalibration; correlation channel {b.pm('A.delta.correlation_channel', 3, sign=True)}); "
        f"one vol point on all names is worth {b.pm('A.vega.names_recalibrated.pct_price', 3, sign=True)} % of the price with λ recalibrated ({b.pm('A.vega.names_held.pct_price', 3, sign=True)} % with λ held) and one on the index {b.pm('A.vega.index.pct_price', 3, sign=True)} %; "
        f"the skew vegas are {b.pm('A.vega.skew_rotation.pct_price', 3, sign=True)} % per +1 rota (the rotation defined in A7) and {b.pm('A.vega.skew_put90.pct_price', 3, sign=True)} % per vol point at the 90 % strike of the index; "
        f"over seven recalibrated variants of R_low and of the λ family the price moves by {b.v('A.model_risk.1_equi_0_10_particle.minus_base_pct', 3, sign=True)} % to {b.v('A.model_risk.3_equi_0_parametric.minus_base_pct', 3, sign=True)} %, "
        "the two ends being variants that are not calibrated like the base (A7)."
    )
    out.append(
        f"**The report's arithmetic at the LC price ({b.n(g + '.held.lc.intersection')} trades: the study's {b.count('C.arith.sample.trades.study_monthly')} monthly trades less the dates the LC pass did not price).** The held gap is {b.pm(g + '.held.lc.intersection', 2, sign=True)} % of notional at the LC price "
        f"(copula {b.pm(g + '.held.copula.intersection', 2, sign=True)}, model S {b.pm(g + '.held.model_s.intersection', 2, sign=True)}, on the same trades) and the hedged gap {b.pm(g + '.hedged.lc.intersection', 2, sign=True)} "
        f"(copula {b.pm(g + '.hedged.copula.intersection', 2, sign=True)}, model S {b.pm(g + '.hedged.model_s.intersection', 2, sign=True)}); the forward pays {b.ci(p + '.forward.lc.intersection')} per 1 of premium [95 % block-bootstrap interval] "
        f"(copula {b.ci(p + '.forward.copula.intersection')}, model S {b.ci(p + '.forward.model_s.intersection')}) and the call struck at the copula's forward price {b.ci(p + '.call_100.lc.intersection', 2)} "
        f"(copula {b.ci(p + '.call_100.copula.intersection', 2)}, model S {b.ci(p + '.call_100.model_s.intersection', 2)}); LC's price over payoff is {b.pm(s + '.price_over_payoff.lc.intersection', 3)}, "
        f"with a model part {b.pm(s + '.model_part_minus_copula.lc.intersection', 3, sign=True)} from the copula's (the ± are sampling errors over trades: Hansen–Hodrick for the gaps; "
        f"on these trades the hedged gap at the LC price is {b.v(g + '.hedged.lc.intersection.t_ratio', 1)} standard errors from zero and the forward's interval includes 1)."
    )
    out.append(
        f"**Calls.** On the {b.n(h + '.C_lc_over_copula_075.mean.S')} dates where both exist, the LC call is {b.pm(h + '.C_lc_over_copula_075.mean.S', 3)} of the copula's at 0.75 × the copula's forward, {b.pm(h + '.C_lc_over_copula_100.mean.S', 3)} at the forward and "
        f"{b.pm(h + '.C_lc_over_copula_125.mean.S', 3)} at 1.25 × (model S: {b.pm(h + '.C_S_over_copula_075.mean.S', 3)}, {b.pm(h + '.C_S_over_copula_100.mean.S', 3)}, {b.pm(h + '.C_S_over_copula_125.mean.S', 3)}; means across dates ± their standard error); "
        f"at 1.5 × and above the ratios should not be quoted as model results: measured at the development budget on two dates, the paths on which one name ends above 3 times its spot carry {b.v(f'V.runaway.{d0}.lc.150.share_above_3x', 0, 100)} % of the LC call at 1.5 × and "
        f"{b.v(f'V.runaway.{d0}.lc.200.share_above_3x', 0, 100)} % at 2 × on 2026-10-02 ({b.v('V.runaway.2023-02-06.lc.150.share_above_3x', 0, 100)} % and {b.v('V.runaway.2023-02-06.lc.200.share_above_3x', 0, 100)} % on 2023-02-06), "
        f"against {b.v(f'V.runaway.{d0}.lc.ED.share_above_3x', 2, 100)} % of E[D]."
    )
    out.append(
        f"**Validation.** The synthetic test S3 fails the gate decided on 9 Oct on {b.count('V.s3.gate.cells_over')} of {b.n('V.s3.gate.cells_over')} cells, all at +2.5 sd (largest {b.pm('V.s3.gate.1m.sd+2.5', 4, sign=True)} vol points against the 0.05 gate), and passes from −2.5 to +2.0 sd; "
        f"on the Dow today the model's index smile is {b.pm('A.idx_err.90', 2, sign=True)} vol points from its target (the model's own SVI fit of the DJX smile) at the 90 % strike and {b.pm('A.idx_err.m25', 2, sign=True)} at −2.5 sd, "
        f"with λ clipped at its cap on {b.v('A.clip.inner_high', 1)} % of the particles inside ±2.5 sd at its worst calibration slice; the golden baseline and the slow Dow tests have not been rerun under the defaults of 9 Oct."
    )
    over = [
        r
        for i, r in b.rec.items()
        if i.startswith("V.s3.gate.") and "sd" in i and str(r.get("notes", "")).startswith("OVER")
    ]
    way = (
        b.rec.get("D.high.lc_over_copula.mean"),
        b.rec.get("D.high.cdv3_over_copula.mean"),
        b.rec.get("D.high.cdv6_over_copula.mean"),
        b.rec.get("D.high.s_over_copula.mean"),
    )
    claims = (
        ("check (e): kappa is the larger part", lambda: b.rec[e + ".dln_kappa_names_held"]["value"] > b.rec[e + ".short_term_names_held"]["value"] > 0),
        ("the forward payout interval at the LC price includes 1", lambda: b.rec[p + ".forward.lc.intersection.ci95_lo"]["value"] < 1 < b.rec[p + ".forward.lc.intersection.ci95_hi"]["value"]),
        ("model-risk range: variants 1 and 3 are the lowest and the highest of seven", lambda: (
            len([i for i in b.rec if i.startswith("A.model_risk.") and i.endswith(".minus_base_pct")]) == 7
            and min(r["value"] for i, r in b.rec.items() if i.startswith("A.model_risk.") and i.endswith(".minus_base_pct")) == b.rec["A.model_risk.1_equi_0_10_particle.minus_base_pct"]["value"]
            and max(r["value"] for i, r in b.rec.items() if i.startswith("A.model_risk.") and i.endswith(".minus_base_pct")) == b.rec["A.model_risk.3_equi_0_parametric.minus_base_pct"]["value"])),
        ("S3: the largest cell is 1m at +2.5 sd", lambda: abs(b.rec["V.s3.gate.1m.sd+2.5"]["value"]) == max(abs(r["value"]) for i, r in b.rec.items() if i.startswith("V.s3.gate.") and "sd" in i)),
        ("S3: every cell over the gate is at +2.5 sd and every other cell is within it", lambda: len(over) == round(b.rec["V.s3.gate.cells_over"]["value"]) and all("sd+2.5" in r["id"] for r in over)),
        ("stratified: about a quarter and about a half of the distance to model S", lambda: 0.2 < (way[0]["value"] - way[1]["value"]) / (way[0]["value"] - way[3]["value"]) < 0.3 and 0.45 < (way[0]["value"] - way[2]["value"]) / (way[0]["value"] - way[3]["value"]) < 0.6),
        ("reference dates: the repair's move is about forty times the printed error on the first two dates", lambda: all(30 < abs(b.rec[f"B.{d}.new_minus_old.ratio.lc_over_cc"]["value"]) / b.rec[f"B.{d}.ratio.lc_over_cc"]["se"] < 50 for d in (d1, d2))),
        ("history: LC/CC does not move with the clipped mass (terciles within 0.012)", lambda: max(b.rec[f"{h}.lc_over_cc.mean.clip_{k}"]["value"] for k in ("low", "mid", "high")) - min(b.rec[f"{h}.lc_over_cc.mean.clip_{k}"]["value"] for k in ("low", "mid", "high")) < 0.012),
        ("history: S/copula falls with the clipped mass tercile", lambda: b.rec[h + ".s_over_copula.mean.clip_low"]["value"] > b.rec[h + ".s_over_copula.mean.clip_mid"]["value"] > b.rec[h + ".s_over_copula.mean.clip_high"]["value"]),
    )  # fmt: skip
    for label, test in claims:
        try:
            ok = bool(test())
        except (KeyError, TypeError):
            b.missing.append(f"claim not checked: {label}")
            continue
        if not ok:
            raise ValueError(f"a finding's sentence does not hold on its records: {label}")
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument(
        "--strict", action="store_true", help="fail when a record a finding reads is missing"
    )
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    book = Book()
    items = findings(book)
    md = "\n".join(f"{i}. {text}" for i, text in enumerate(items, 1))
    md += "\n\nEvery number above is a record of the sections it comes from (`numbers.json`); the findings add only `1 − ratio` where a discount is quoted."
    seen: set[str] = set()
    derived = [r for r in book.derived if not (r["id"] in seen or seen.add(r["id"]))]
    pc.write_part("Z_findings", derived, md)
    missing = sorted(set(book.missing))
    log.info("Z_findings written: %d findings; missing records: %s", len(items), missing or "none")
    if missing and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
