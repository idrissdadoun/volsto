"""Write the reference fixtures of test S4 (SPEC §8.7) — run once, by hand:

    python tests/golden/lcm_reference/make_fixture.py

Source: the stand-alone reference implementation's outputs ``outputs/interview/lcm_reference/
out/<tag>_params.json`` and ``<tag>_results.json`` (run of 2026-10-07; kept outside git), for the
tags ``today`` (2026-10-02), ``typical`` (2019-09-03), ``steep`` (2017-04-03) and
``typical_alt`` (2008-07-07).  Each fixture ``<tag>.json`` records the SHA-256 of its two source
files and holds what the regression needs and nothing else: the names' weights and their
two-parameter local vols (``sig0``, ``b``), the maturity and the number of steps, the fitted
correlations (``rho_cc``; ``rho0``, ``c``), the two index targets, and the anchors with their
standard errors (``[value, stderr]``; the ratios', calls' and deltas' errors are paired ones).

The fixtures are regression anchors, not library code, and are never regenerated to make a test
pass: a change of the reference is a new run with new files and new digests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[2] / "outputs" / "interview" / "lcm_reference" / "out"
TAGS = ("today", "typical", "steep", "typical_alt")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(tag: str) -> dict[str, object]:
    params_path, results_path = SOURCE / f"{tag}_params.json", SOURCE / f"{tag}_results.json"
    p = json.loads(params_path.read_text())
    r = json.loads(results_path.read_text())
    assert p["date"] == r["params"]["date"] and p["rho0"] == r["params"]["rho0"]
    return {
        "tag": tag,
        "date": p["date"],
        "source_sha256": {
            f"{tag}_params.json": _sha256(params_path),
            f"{tag}_results.json": _sha256(results_path),
        },
        "n_paths": p["n_paths"],
        "T": p["T"],
        "steps": p["steps"],
        "single": {k: p["single"][k] for k in ("tickers", "w", "sig0", "b")},
        "rho_cc": p["rho_cc"],
        "rho0": p["rho0"],
        "c": p["c"],
        "targets": {"sigB": p["sigB"], "v90B": p["v90B"]},
        "anchors": {
            "mom.cc.ED": r["mom"]["cc"]["ED"],
            "mom.lc.ED": r["mom"]["lc"]["ED"],
            "forward_ratio": r["forward_ratio"],
            "calls": {
                m: {
                    "K": c["K"],
                    "cc": c["cc"],
                    "lc": c["lc"],
                    "ratio": c["ratio"],
                    "diff": c["diff"],
                }
                for m, c in r["calls"].items()
            },
            "delta.cc": r["delta"]["cc"],
            "delta.lc": r["delta"]["lc"],
            "fit.lc.iv_str": [r["fit"]["lc"]["iv_str"], r["fit"]["lc"]["se_iv_str"]],
            "fit.lc.iv_put": [r["fit"]["lc"]["iv_put"], r["fit"]["lc"]["se_iv_put"]],
            "fit.cc.iv_put": [r["fit"]["cc"]["iv_put"], r["fit"]["cc"]["se_iv_put"]],
            "fit.cc.iv_str": [r["fit"]["cc"]["iv_str"], r["fit"]["cc"]["se_iv_str"]],
        },
    }


def main() -> None:
    for tag in TAGS:
        out = HERE / f"{tag}.json"
        if out.exists():
            raise SystemExit(f"{out} exists: the fixtures are written once")
        out.write_text(json.dumps(fixture(tag), indent=1) + "\n")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
