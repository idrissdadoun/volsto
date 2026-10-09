"""PM results package of 2026-10-09, section A: today's risk of the Palladium forward under the
calibrated local correlation model, in pieces that run side by side (SPEC §8.7, LC6/LC7).

    python scripts/pm_today_risk.py deltas   [--date 2026-10-02] [--budget production] [--root <dir>]
    python scripts/pm_today_risk.py vegas    ...
    python scripts/pm_today_risk.py modelrisk --variants 0,1,2   (indices into the list below)
    python scripts/pm_today_risk.py merge

Each piece writes ``<root>/outputs/dispersion_lc/pm_update/parts/raw/risk_<piece>.json`` and
``merge`` assembles ``parts/A_risk_today_raw.json``.  Every model comes through the local
correlation cache on the date's specification under the default configuration (``lcm.yaml``:
the owner's decisions of 2026-10-09 on); every difference is paired on the pricing seed.

* ``deltas``: the common deltas of the forward as elasticities — percent of the model's own
  ``E[D]`` per +1 % on every spot, arithmetic bump of 1 %, central — under sticky strike and
  sticky moneyness for LC and for its constant-correlation companion, and the decomposition
  (``volsto.risk.local_correlation.delta_decomposition``: homogeneity ``Δ^LC_sm``, skew channel
  ``Δ^CC_ss − Δ^CC_sm``, correlation channel ``(Δ^LC_ss − Δ^LC_sm) − (Δ^CC_ss − Δ^CC_sm)``, level
  term ``Δ^LC_sm − Δ^CC_sm``).  Also the index smile at the horizon against its target at −3.5 to
  −1.5 at-the-money standard deviations, at the money and at the 90 % strike.
* ``vegas``: every name's surface +1 vol point at once with ``λ`` recalibrated to the unchanged
  index smile, and with ``λ`` held; the index target +1 vol point; the two index skew vegas (the
  +1 rota; a 1 vol point tent at the 90 % strike).  Values are changes of ``E^LC[D]`` in units
  of notional per vol point (per +1 rota for the rotation).
* ``modelrisk``: the forward under another ``R_low`` or family, recalibrated — the variants are
  ``equi 0``, ``equi 0.10``, ``historical-scaled:252,0.05`` with the particle family (indices 0,
  1, 2) and ``equi 0``, ``equi 0.02``, ``equi 0.10``, ``historical-scaled:252,0.05`` with the
  two-parameter family (3, 4, 5, 6) — with the paired difference to the base.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lcm_diagnostics as lcd
import lcm_price as lp

from volsto.calibration.cache import code_version
from volsto.calibration.fit_records import FitRecords
from volsto.calibration.lc_cache import (
    LocalCorrelationCache,
    build_lc_market,
    lc_spec_key,
)
from volsto.market.bs import black_vega, implied_vol
from volsto.multi.family import historical_scaled_correlation
from volsto.multi.products import Palladium
from volsto.risk import local_correlation as lcr
from volsto.studies import disp_data as dd

log = logging.getLogger("pm_today_risk")
SDS = (-3.5, -3.0, -2.5, -2.0, -1.5, 0.0)
PIECES = ("deltas", "vegas", "modelrisk")


def setup(date: str, tenor: str, budget: str, root: Path) -> dict[str, Any]:
    cfg = lp.load_config()
    smoke = os.environ.get("PM_SMOKE")  # a tiny budget for testing the code path only
    if smoke:
        cfg["budgets"][budget] = {
            "n_particles": int(smoke),
            "n_paths": int(smoke),
            "companion_paths": int(smoke),
        }
    cache_root = root / cfg["cache"]
    cache = LocalCorrelationCache(cache_root)
    inp = lcd.load_inputs(date, tenor, cfg["index"])
    spec, info = lp.spec_for(inp, cfg, budget, FitRecords(cache_root / "svi_fits"))
    return {
        "cfg": cfg, "cache": cache, "inp": inp, "spec": spec, "info": info,
        "head": {
            "date": date, "tenor": tenor, "budget": budget, "commit": code_version(),
            "spec_key": lc_spec_key(spec), "n_particles": spec.lc.particle.n_particles,
            "n_paths": spec.sim.n_paths, "companion_paths": spec.lc.parametric.n_paths,
            "particle_seed": spec.lc.particle.seed, "pricing_seed": spec.sim.seed,
            "screen": info["screen"], "config_digest": lp.config_digest(cfg, tenor, budget),
        },
    }  # fmt: skip


def index_errors_at(levels: np.ndarray, surface: Any, T: float) -> list[dict[str, Any]]:
    """The basket's implied vol at the horizon minus the target's, vol points with standard
    errors on antithetic pair means, at the strikes of ``SDS`` and at the 90 % strike."""
    atm = float(surface.atm_vol(T))
    out = []
    for label, k in [(f"{m:+.1f}", m * atm * math.sqrt(T)) for m in SDS] + [("90%", math.log(0.9))]:
        cp = 1.0 if k >= 0 else -1.0
        price, se = lcd.pair_mean(np.maximum(cp * (levels - math.exp(k)), 0.0))
        vol = float(implied_vol(price, 1.0, math.exp(k), T, cp))
        target = float(surface.implied_vol_k(k, T))
        vega = float(black_vega(1.0, math.exp(k), T, vol)) if np.isfinite(vol) else float("nan")
        out.append({"strike": label, "k": k, "model_vol": vol, "target_vol": target,
                    "error_vp": 100 * (vol - target), "stderr_vp": 100 * se / vega if vega > 0 else float("nan")})  # fmt: skip
    return out


def run_deltas(s: dict[str, Any]) -> dict[str, Any]:
    spec, cache = s["spec"], s["cache"]
    T, w = spec.lc.particle.horizon, np.array(spec.weights)
    model, _ = cache.get_or_calibrate(spec)
    cc_spec = dataclasses.replace(spec, lc=dataclasses.replace(spec.lc, family="constant"))
    model_cc, diag_cc = cache.get_or_calibrate(cc_spec)
    forward = Palladium(w, 0.0, T, lp.ZERO)
    e_lc = lcr.LCRiskEngine(lcr.FixedLambdaBuilder(model, "LC"), spec.sim)
    e_cc = lcr.LCRiskEngine(lcr.FixedLambdaBuilder(model_cc, "CC"), spec.sim)
    d = lcr.delta_decomposition(e_lc, e_cc, forward, size=lcr.SPOT_BUMP, bump="arith")
    base_lc, base_cc = e_lc.priced(forward, e_lc.builder.base), e_cc.priced(
        forward, e_cc.builder.base
    )
    market = build_lc_market(spec)
    lc_pass = lp.Pass(model, spec.sim, w, [T])
    return {
        "base": {"ED_lc": [base_lc.mean, base_lc.stderr], "ED_cc": [base_cc.mean, base_cc.stderr]},
        "lambda_c": float(diag_cc.calibration["lambda"]) if diag_cc is not None else None,
        "deltas": {
            "lc_ss": list(d.lc_ss), "lc_sm": list(d.lc_sm), "cc_ss": list(d.cc_ss), "cc_sm": list(d.cc_sm),
            "homogeneity": list(d.homogeneity), "skew_channel": list(d.skew_channel),
            "correlation_channel": list(d.correlation_channel), "level_term": list(d.level_term),
            "residual": d.residual, "bump": "arith", "size": lcr.SPOT_BUMP,
            "unit": "percent of the model's own E[D] per +1 % on every spot",
        },
        "index_errors_T": index_errors_at(lc_pass.levels[:, -1], market.index_surface, T),
        "n_pricings": e_lc.n_pricings + e_cc.n_pricings,
    }  # fmt: skip


def sens(x: Any) -> dict[str, Any]:
    return {"name": x.name, "value": x.value, "stderr": x.stderr, "unit": x.unit,
            "unpaired_stderr": x.extra.get("unpaired_stderr"), "base_price": x.extra.get("base_price")}  # fmt: skip


def run_vegas(s: dict[str, Any]) -> dict[str, Any]:
    spec, cache = s["spec"], s["cache"]
    T, w = spec.lc.particle.horizon, np.array(spec.weights)
    forward = Palladium(w, 0.0, T, lp.ZERO)
    state = lcr.LCState(spec)
    builder = lcr.LCBuilder(cache, state)
    engine = lcr.LCRiskEngine(builder, spec.sim)
    out = []
    recal = lcr.name_vegas(engine, forward, state, names=(), all_names=True)
    held = lcr.name_vegas(engine, forward, state, lambda_mode="held", names=(), all_names=True)
    out.append({**sens(recal[-1]), "name": "vega[all names] recalibrated"})
    out.append({**sens(held[-1]), "name": "vega[all names] held"})
    out.append(sens(lcr.index_vega(engine, forward, state)))
    out += [
        sens(lcr.index_skew_vega(engine, forward, state, kind=k)) for k in ("rotation", "put90")
    ]
    base = engine.priced(forward, state)
    return {"vegas": out, "base": {"ED_lc": [base.mean, base.stderr]}, "n_calibrated": builder.n_missed,
            "n_pricings": engine.n_pricings}  # fmt: skip


def variant_specs(
    s: dict[str, Any],
) -> tuple[list[tuple[str, Any]], dict[str, Any], dict[str, Any]]:
    """The seven alternative specifications (module docstring), the historical-scaled matrices
    by cache key, and how that matrix was made."""
    spec, inp = s["spec"], s["inp"]
    w = np.array(spec.weights)
    panel = dd.prices().ffill()
    rets = np.log(panel[inp.names]).diff().loc[: inp.date].iloc[-252:]
    window = np.ascontiguousarray(rets.to_numpy(), dtype=np.float64)
    hist = historical_scaled_correlation(window, w, 0.05)
    source = hashlib.sha256(window.tobytes()).hexdigest()
    variants: list[tuple[str, Any]] = []
    matrices: dict[str, Any] = {}
    for family in ("particle", "parametric"):
        for label, changes, extra in (
            ("equi 0", {"rho_min": 0.0}, {}),
            ("equi 0.02", {"rho_min": 0.02}, {}),
            ("equi 0.10", {"rho_min": 0.10}, {}),
            (
                "historical-scaled:252,0.05",
                {"r_low": "historical-scaled:252,0.05"},
                {"r_low_source": source},
            ),
        ):
            other = dataclasses.replace(
                spec, lc=dataclasses.replace(spec.lc, family=family, **changes), **extra
            )
            if lc_spec_key(other) == lc_spec_key(spec):
                continue
            variants.append((f"{label} / {family}", other))
            if "r_low_source" in extra:
                matrices[lc_spec_key(other)] = hist.r_low
    info = {
        **hist.info(),
        "window": [str(rets.index[0])[:10], str(rets.index[-1])[:10]],
        "digest": source,
    }
    return variants, matrices, info


def run_modelrisk(s: dict[str, Any], which: Sequence[int]) -> dict[str, Any]:
    spec, cache = s["spec"], s["cache"]
    T, w = spec.lc.particle.horizon, np.array(spec.weights)
    forward = Palladium(w, 0.0, T, lp.ZERO)
    variants, matrices, info = variant_specs(s)
    chosen = [variants[i] for i in which]
    state = lcr.LCState(spec)
    builder = lcr.LCBuilder(cache, state, matrices=matrices)
    engine = lcr.LCRiskEngine(builder, spec.sim)
    rng = lcr.model_risk_range(engine, forward, state, chosen)
    base = engine.priced(forward, state)
    return {"rows": rng.rows, "labels": [v[0] for v in variants], "which": list(which), "historical_scaled": info,
            "base": {"ED_lc": [base.mean, base.stderr]}, "n_calibrated": builder.n_missed}  # fmt: skip


def merge(raw: Path, out: Path) -> dict[str, Any]:
    doc: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for path in sorted(raw.glob("risk_*.json")):
        piece = json.loads(path.read_text())
        for key in ("date", "tenor", "budget", "commit", "spec_key", "n_particles", "n_paths", "companion_paths",
                    "particle_seed", "pricing_seed", "config_digest", "screen"):  # fmt: skip
            if key in piece["head"]:
                if key in doc and doc[key] != piece["head"][key]:
                    raise ValueError(
                        f"{path.name}: {key} differs between the pieces: {doc[key]} / {piece['head'][key]}"
                    )
                doc[key] = piece["head"][key]
        body = piece["body"]
        doc.setdefault("pieces", {})[path.stem] = {
            "seconds": piece["seconds"],
            "written": piece["written"],
        }
        if "deltas" in body:
            doc.update(
                base=body["base"],
                deltas=body["deltas"],
                index_errors_T=body["index_errors_T"],
                lambda_c=body["lambda_c"],
            )
        if "vegas" in body:
            doc["vegas"] = body["vegas"]
            doc["vegas_base"] = body["base"]
        if "rows" in body:
            rows += body["rows"]
            doc["model_risk_base"] = body["base"]
            doc["historical_scaled"] = body["historical_scaled"]
            doc["model_risk_labels"] = body["labels"]
    if rows:
        doc["model_risk"] = {"rows": rows, "low": min(r["price"] for r in rows), "high": max(r["price"] for r in rows),
                             "n_variants": len(rows)}  # fmt: skip
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=lp.jsonable))
    tmp.replace(out)
    return doc


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("piece", choices=(*PIECES, "merge"))
    ap.add_argument("--date", default="2026-10-02")
    ap.add_argument("--tenor", default="3m")
    ap.add_argument("--budget", default="production", choices=("production", "development"))
    ap.add_argument("--variants", default="0,1,2,3,4,5,6")
    ap.add_argument("--root", default="/Users/idrissdadoun/Code/volsto-lc")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S", stream=sys.stdout
    )
    root = Path(args.root)
    pm_dir = root / "outputs" / "dispersion_lc" / "pm_update"
    raw = pm_dir / "parts" / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    if args.piece == "merge":
        doc = merge(raw, pm_dir / "parts" / "A_risk_today_raw.json")
        log.info("merged %s: keys %s", sorted(doc.get("pieces", {})), sorted(doc))
        return 0
    t0 = time.perf_counter()
    s = setup(args.date, args.tenor, args.budget, root)
    log.info("%s on %s %s (%s budget), key %s, commit %s", args.piece, args.date, args.tenor, args.budget,
             s["head"]["spec_key"][:12], s["head"]["commit"])  # fmt: skip
    name = args.piece
    if args.piece == "deltas":
        body = run_deltas(s)
    elif args.piece == "vegas":
        body = run_vegas(s)
    else:
        which = [int(x) for x in args.variants.split(",")]
        body = run_modelrisk(s, which)
        name = "modelrisk_" + "_".join(str(i) for i in which)
    doc = {
        "head": s["head"],
        "body": body,
        "seconds": time.perf_counter() - t0,
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = raw / f"risk_{name}.json"
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, default=lp.jsonable))
    tmp.replace(path)
    log.info("written %s in %.0f s", path, doc["seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
