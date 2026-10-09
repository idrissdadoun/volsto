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
    out = []
    out.append(
        f"**Today.** On 2026-10-02 (production budget, decisions 1, 2 and 5 on) the local correlation model prices the Palladium forward at E[D] = {b.pm('A.ED.lc', 6)} of notional, "
        f"which is {b.pm('A.ratio.lc_over_cc', 4)} of its constant-correlation companion and {b.pm('A.ratio.lc_over_copula', 4)} of the copula's {b.v('A.ED.copula', 6)}, "
        f"with CC/copula at {b.pm('A.ratio.cc_over_copula', 4)} and the listed-variance forward at {b.v('A.ratio.listed_fwd', 4)} of the copula's; model S did not converge on this date, so today has no valid S/copula."
    )
    out.append(
        f"**History.** On the {b.n('C.hist.lc_over_copula.mean.all')} priced monthly dates of 2007–2026 (3m, development budget, decisions 1–2 on) LC/copula averages {b.pm('C.hist.lc_over_copula.mean.all', 4)} "
        f"(median {b.v('C.hist.lc_over_copula.median.all', 4)}, pooled {b.v('C.hist.lc_over_copula.pooled.all', 4)}), LC/CC {b.pm('C.hist.lc_over_cc.mean.all', 4)} and CC/copula {b.pm('C.hist.cc_over_copula.mean.all', 4)}, "
        f"against model S/copula {b.pm('C.hist.s_over_copula.mean.all', 4)} on its {b.n('C.hist.s_over_copula.mean.all')} dates and a listed-variance forward of {b.pm('C.hist.listed_fwd_ratio.mean.all', 4)}: "
        f"LC takes {b.discount('C.hist.lc_over_copula.mean.all')} off the copula's forward on average where model S takes {b.discount('C.hist.s_over_copula.mean.all')} (± across dates, the dates treated as independent)."
    )
    out.append(
        f"**Where LC and model S differ.** LC/CC − S/copula is {b.pm('C.hist.y_check_d.mean.clip_low', 3, sign=True)} in the lowest tercile of M12's clipped mass, {b.pm('C.hist.y_check_d.mean.clip_mid', 3, sign=True)} in the middle one and "
        f"{b.pm('C.hist.y_check_d.mean.clip_high', 3, sign=True)} in the highest, and across the {b.n('C.d.S.rel.n')} dates it rises by {b.v('C.d.S.rel.clip_inner_max.coef', 3)} per unit of clipped mass "
        f"(HC1 standard error {b.v('C.d.S.rel.clip_inner_max.se_hc1', 3)}, R² {b.v('C.d.S.rel.r2', 2)}): LC's smaller discount sits on the dates where its correlation is clipped."
    )
    out.append(
        f"**Stratified test of the cross-dependent volatility (check f).** On the {b.n('D.high.lc_over_copula.mean')} dates where M12's clipped mass is largest, the forward over the copula's is {b.pm('D.high.lc_over_copula.mean', 4)} under LC, "
        f"{b.pm('D.high.cdv3_over_copula.mean', 4)} under CDV at β = 3 and {b.pm('D.high.cdv6_over_copula.mean', 4)} at β = 6, against {b.pm('D.high.s_over_copula.mean', 4)} for model S and {b.pm('D.high.listed_fwd.mean', 4)} for the listed-variance forward, "
        f"with the clipped mass going from {b.pm('D.high.clip_lc.mean', 3)} to {b.pm('D.high.clip_cdv3.mean', 3)} and {b.pm('D.high.clip_cdv6.mean', 3)}; "
        f"on the {b.n('D.low.lc_over_copula.mean')} dates where it is smallest the same five ratios are {b.pm('D.low.lc_over_copula.mean', 4)}, {b.pm('D.low.cdv3_over_copula.mean', 4)}, {b.pm('D.low.cdv6_over_copula.mean', 4)}, "
        f"{b.pm('D.low.s_over_copula.mean', 4)} and {b.pm('D.low.listed_fwd.mean', 4)} (means across each group's dates ± their standard error, development budget); per date the prototype changes the forward over CC by {b.pm('D.high.cdv3_minus_lc_over_cc.mean', 4, sign=True)} (β = 3) and {b.pm('D.high.cdv6_minus_lc_over_cc.mean', 4, sign=True)} (β = 6) in the first group and by {b.pm('D.low.cdv3_minus_lc_over_cc.mean', 4, sign=True)} and {b.pm('D.low.cdv6_minus_lc_over_cc.mean', 4, sign=True)} in the second, so the part specific to the binding dates is {b.pm('D.high_minus_low.cdv3_minus_lc_over_cc', 4, sign=True)} and {b.pm('D.high_minus_low.cdv6_minus_lc_over_cc', 4, sign=True)}: the direction is the conjectured one, but at a fixed β the test does not separate the wing's effect from the cross-dependence's own."
    )
    e = f"A.check_e.{d0}.beta3"
    out.append(
        f"**Check (e), today, production budget.** Closing the index wing with the prototype at β = 3 moves the forward by {b.pm(e + '.dln_ED', 2, 100, True)} %, not the {b.pm(e + '.wing_estimate', 2, 100, True)} % of the fixed-κ estimate: "
        f"κ rises ({b.pm(e + '.dln_kappa_names_held', 2, 100, True)} %, from {b.v(e + '.kappa_base', 4)} to {b.pm(e + '.kappa_names_held', 4)}) and the basket's second moment is still short of the listed one "
        f"({b.pm(e + '.short_term_names_held', 2, 100, True)} %, {b.pm(e + '.shortfall_closed', 0, 100)} % of the shortfall closed), so both reasons hold and κ is the larger."
    )
    out.append(
        f"**Reference dates (production budget).** LC/CC is {b.pm(f'B.{d1}.ratio.lc_over_cc', 4)} on {d1}, {b.pm(f'B.{d2}.ratio.lc_over_cc', 4)} on {d2} and {b.pm(f'B.{d3}.ratio.lc_over_cc', 4)} on {d3}, "
        f"against {b.v(f'B.{d1}.ref.lc_over_cc', 4)}, {b.v(f'B.{d2}.ref.lc_over_cc', 4)} and {b.v(f'B.{d3}.ref.lc_over_cc', 4)} for the parametric reference implementation and "
        f"{b.v(f'B.{d1}.ratio.s_over_copula', 4)}, {b.v(f'B.{d2}.ratio.s_over_copula', 4)} and {b.v(f'B.{d3}.ratio.s_over_copula', 4)} for model S over the copula; "
        f"the calendar repair of decisions 1–2 moved LC/CC by {b.pm(f'B.{d1}.new_minus_old.ratio.lc_over_cc', 4, sign=True)} and {b.pm(f'B.{d2}.new_minus_old.ratio.lc_over_cc', 4, sign=True)} on the first two dates, far more than the Monte Carlo errors, "
        f"and by {b.pm(f'B.{d0}.new_minus_old.ratio.lc_over_cc', 4, sign=True)} today."
    )
    out.append(
        f"**Risk today.** The sticky-strike common delta is {b.pm('A.delta.lc_ss', 3, sign=True)} % of E[D] per +1 % under LC against {b.pm('A.delta.cc_ss', 3, sign=True)} % under CC (correlation channel {b.pm('A.delta.correlation_channel', 3, sign=True)}); "
        f"one vol point on all names is worth {b.pm('A.vega.names_recalibrated.pct_price', 3, sign=True)} % of the price with λ recalibrated ({b.pm('A.vega.names_held.pct_price', 3, sign=True)} % with λ held) and one on the index {b.pm('A.vega.index.pct_price', 3, sign=True)} %, "
        f"the skew vegas are {b.pm('A.vega.skew_rotation.pct_price', 3, sign=True)} % (rotation) and {b.pm('A.vega.skew_put90.pct_price', 3, sign=True)} % (90 % put), "
        f"and the model-risk range over seven recalibrated variants is {b.pm('A.model_risk.1_equi_0_10_particle.minus_base_pct', 3, sign=True)} % to {b.pm('A.model_risk.3_equi_0_parametric.minus_base_pct', 3, sign=True)} % of the price."
    )
    g, p, s = "C.arith.gap", "C.arith.payout", "C.arith.split"
    out.append(
        f"**The report's arithmetic at the LC price ({b.n(g + '.held.lc.intersection')} trades).** The held gap is {b.pm(g + '.held.lc.intersection', 2, sign=True)} % of notional at the LC price "
        f"(copula {b.pm(g + '.held.copula.intersection', 2, sign=True)}, model S {b.pm(g + '.held.model_s.intersection', 2, sign=True)}) and the hedged gap {b.pm(g + '.hedged.lc.intersection', 2, sign=True)} "
        f"(copula {b.pm(g + '.hedged.copula.intersection', 2, sign=True)}, model S {b.pm(g + '.hedged.model_s.intersection', 2, sign=True)}); the forward pays {b.ci(p + '.forward.lc.intersection')} per 1 of premium "
        f"(copula {b.ci(p + '.forward.copula.intersection')}, model S {b.ci(p + '.forward.model_s.intersection')}) and the call struck at the copula's forward price {b.ci(p + '.call_100.lc.intersection', 2)} "
        f"(copula {b.ci(p + '.call_100.copula.intersection', 2)}, model S {b.ci(p + '.call_100.model_s.intersection', 2)}); LC's price over payoff is {b.pm(s + '.price_over_payoff.lc.intersection', 3)} "
        f"with a model part of {b.pm(s + '.model_part.lc.intersection', 3)} (the ± are sampling errors over trades; at the LC price the hedged gap is within two standard errors of zero and the forward's payout interval includes 1)."
    )
    out.append(
        f"**Calls.** Over the history the LC call is {b.pm('C.hist.C_lc_over_copula_075.mean.all', 3)} of the copula's at 0.75 × the forward, {b.pm('C.hist.C_lc_over_copula_100.mean.all', 3)} at the forward and {b.pm('C.hist.C_lc_over_copula_125.mean.all', 3)} at 1.25 × "
        f"(model S: {b.pm('C.hist.C_S_over_copula_075.mean.S', 3)}, {b.pm('C.hist.C_S_over_copula_100.mean.S', 3)}, {b.pm('C.hist.C_S_over_copula_125.mean.S', 3)}); "
        f"at 1.5 × and above the LC and CC calls are carried by a few paths on which one name ends above 3 times its spot ({b.v(f'V.runaway.{d0}.lc.150.share_above_3x', 0, 100)} % of today's LC call at 1.5 × and {b.v(f'V.runaway.{d0}.lc.200.share_above_3x', 0, 100)} % at 2 ×, "
        f"against {b.v(f'V.runaway.{d0}.lc.ED.share_above_3x', 2, 100)} % of E[D]), so the ratios at those strikes should not be quoted as model results."
    )
    out.append(
        f"**Validation.** The synthetic test S3 with the gate decided on 9 Oct fails on {b.count('V.s3.gate.cells_over')} of 33 cells, all at +2.5 sd (largest {b.pm('V.s3.gate.1m.sd+2.5', 4, sign=True)} vol points against the 0.05 gate) and passes from −2.5 to +2.0 sd; "
        f"on the Dow today the calibrated index smile is {b.pm('A.idx_err.90', 2, sign=True)} vol points from its target at the 90 % strike and {b.pm('A.idx_err.m25', 2, sign=True)} at −2.5 sd, where the correlation is clipped at its cap on {b.v('A.clip.inner_high', 1)} % of the particles; "
        "the golden baseline and the slow Dow tests have not been rerun under the defaults of 9 Oct."
    )
    claims = (
        ("check (e): kappa is the larger part", lambda: b.rec[e + ".dln_kappa_names_held"]["value"] > b.rec[e + ".short_term_names_held"]["value"] > 0),
        ("hedged gap at the LC price within two standard errors of zero", lambda: abs(b.rec[g + ".hedged.lc.intersection"]["value"]) < 2 * b.rec[g + ".hedged.lc.intersection"]["se"]),
        ("forward payout interval at the LC price includes 1", lambda: b.rec[p + ".forward.lc.intersection.ci95_lo"]["value"] < 1 < b.rec[p + ".forward.lc.intersection.ci95_hi"]["value"]),
        ("model-risk range: variants 1 and 3 are the lowest and the highest", lambda: (
            min(r["value"] for i, r in b.rec.items() if i.startswith("A.model_risk.") and i.endswith(".minus_base_pct")) == b.rec["A.model_risk.1_equi_0_10_particle.minus_base_pct"]["value"]
            and max(r["value"] for i, r in b.rec.items() if i.startswith("A.model_risk.") and i.endswith(".minus_base_pct")) == b.rec["A.model_risk.3_equi_0_parametric.minus_base_pct"]["value"])),
        ("S3: the largest cell is 1m at +2.5 sd", lambda: abs(b.rec["V.s3.gate.1m.sd+2.5"]["value"]) == max(abs(r["value"]) for i, r in b.rec.items() if i.startswith("V.s3.gate.") and "sd" in i)),
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
