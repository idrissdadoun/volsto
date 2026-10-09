"""Cross-dependent volatility prototype on the dispersion study's Dow basket: a scan of ``β``
(``volsto/multi/cdv.py``, ``docs/cross_dependent_vol.md``).

    python scripts/cdv_scan.py --date 2026-10-02 [--tenor 3m] [--betas 0,1,2,3,5]
        [--budget development] [--g-max 2.0] [--rows <M12 rows folder>] [--out <dir>]

For each ``β``: the particle calibration of ``λ`` and of the names' normalisations on the
date's specification (the one ``scripts/lcm_price.py`` builds: the same smiles, screen,
schedule, seeds and budget), then one pass of the pricing paths.  Reported with standard errors
on antithetic pair means: the clipped mass inside ±2.5 sd by side (how much of the index target
is out of reach), the index smile at the horizon against its target, the Palladium forward
``E[D]`` and its ratio to the constant-correlation companion's of the M12 row when that row
exists, ``E[V]`` against the listed ``E^Q[V]`` with its single-name and basket parts (the names
must keep their strips; the basket part is the wing), and the wall time.  ``β = 0`` is M12's
model on the calibration grid.  Outputs: a table in the log and
``<out>/cdv_scan_<date>_<tenor>.json`` (default ``outputs/dispersion_lc/cdv`` of this worktree;
never the study's own folders).
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lcm_diagnostics as lcd
import lcm_price as lp

from volsto.calibration.cache import code_version
from volsto.calibration.lc_cache import build_lc_market, lc_spec_key
from volsto.market.bs import black_vega, implied_vol
from volsto.multi.cdv import CrossDependence, calibrate_cdv, simulate_cdv

log = logging.getLogger("cdv_scan")
SD = (-2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5)


def smile_errors(level: np.ndarray, surface: Any, T: float) -> dict[str, tuple[float, float]]:
    """The basket's implied vol minus the target at the horizon, vol points with standard
    errors, at strikes in at-the-money standard deviations and at the 90 % strike."""
    atm = float(surface.atm_vol(T))
    out = {}
    for label, k in [(f"{m:+.1f}", m * atm * math.sqrt(T)) for m in SD] + [("90%", math.log(0.9))]:
        cp = 1.0 if k >= 0 else -1.0
        price, se = lcd.pair_mean(np.maximum(cp * (level - math.exp(k)), 0.0))
        target = float(surface.implied_vol_k(k, T))
        vol = float(implied_vol(price, 1.0, math.exp(k), T, cp))
        vega = float(black_vega(1.0, math.exp(k), T, target))
        out[label] = (100 * (vol - target), 100 * se / vega)
    return out


def scan(
    date: str, tenor: str, betas: Sequence[float], budget: str, g_max: float, rows_dir: str | None
) -> dict[str, Any]:
    cfg = lp.load_config()
    inp = lcd.load_inputs(date, tenor, cfg["index"])
    spec, _ = lp.spec_for(inp, cfg, budget, None)
    market = build_lc_market(spec)
    T = spec.lc.particle.horizon
    w = np.array(spec.weights)
    b1 = inp.entry["B1"]
    eqv = float(b1["EQV"])
    sum_wm = float(np.sum(inp.weights / inp.weights.sum() * np.asarray(inp.entry["legs"]["M"])))
    m_b = sum_wm - eqv
    row_file = None if rows_dir is None else Path(rows_dir) / f"{tenor}_{budget}" / f"{date}.json"
    ed_cc = (
        json.loads(row_file.read_text()).get("ED_cc")
        if row_file is not None and row_file.exists()
        else None
    )
    log.info(
        "=== cross-dependent scan %s %s (%s budget: N = %d, paths = %d; key of the M12 specification %s); "
        "E_CC[D] of the M12 row: %s",
        date, tenor, budget, spec.lc.particle.n_particles, spec.sim.n_paths, lc_spec_key(spec)[:12], ed_cc,
    )  # fmt: skip
    rows = []
    for beta in betas:
        t0 = time.perf_counter()
        dep = CrossDependence(float(beta), g_max=g_max, g_min=1.0 / g_max)
        res = calibrate_cdv(market.models, market.family, market.basket, market.index_surface,
                            market.index_lv, spec.lc.particle, spec.sim, spec.lc, dep)  # fmt: skip
        t1 = time.perf_counter()
        ls, kb = simulate_cdv(res, market.models, market.family, market.basket, spec.sim, [T])
        t2 = time.perf_counter()
        fwd = np.array(
            [float(m.forward_curve.forward(T)) / float(m.forward_curve.spot) for m in market.models]
        )
        spot0 = np.array([float(m.forward_curve.spot) for m in market.models])
        r = np.exp(ls[0]) / spot0[None, :] - 1.0
        rb = r @ w
        D = np.abs(r - rb[:, None]) @ w
        sq = (r * r) @ w
        level = np.exp(kb[0])
        errs = smile_errors(level, market.index_surface, T)
        ed, ed_se = lcd.pair_mean(D)
        ev, ev_se = lcd.pair_mean(sq - rb**2)
        names, names_se = lcd.pair_mean(sq)
        rbar2, rbar2_se = lcd.pair_mean(rb**2)
        kappa, kappa_se = lcd.pair_ratio_sqrt(D, sq - rb**2)
        f_err, f_se = lcd.pair_mean(level - 1.0)
        row = {
            "beta": float(beta), "g_max": g_max,
            "clip_high_inner": float(res.inner_high.max()), "clip_low_inner": float(res.inner_low.max()),
            "clip_high": float(res.clipped_high.max()), "clip_low": float(res.clipped_low.max()),
            "lambda_mean_T": float(res.lambda_mean[-1]), "scale_min": float(res.scale.min()), "scale_max": float(res.scale.max()),
            "ED": ed, "ED_se": ed_se, "ED_over_cc": None if ed_cc is None else ed / ed_cc,
            "EV_over_EQV": ev / eqv, "EV_over_EQV_se": ev_se / eqv,
            "names_part": names - sum_wm, "names_part_se": names_se, "names_over_listed": names / sum_wm - 1.0,
            "basket_part": rbar2 - m_b, "basket_part_se": rbar2_se, "kappa": kappa, "kappa_se": kappa_se,
            "forward_error": f_err, "forward_error_se": f_se,
            **{f"idx_{k}": v[0] for k, v in errs.items()}, **{f"idx_{k}_se": v[1] for k, v in errs.items()},
            "seconds_calibration": t1 - t0, "seconds_pricing": t2 - t1, "timings": res.timings,
            "mean_forward_check": float(np.max(np.abs(np.exp(ls[0]).mean(axis=0) / (spot0 * fwd) - 1.0))),
        }  # fmt: skip
        rows.append(row)
        log.info(
            "beta %.2f: clipped mass inside ±2.5 sd high %.4f, low %.4f; index at T: ATM %+.3f (%.3f), 90%% %+.3f (%.3f), -1.5 sd %+.3f, -2.5 sd %+.3f (%.3f), +1.5 sd %+.3f; "
            "E[D] %.6f (%.6f)%s; E[V]/E_Q[V] %.4f (%.4f): names %+.6f (%.6f), basket %+.6f (%.6f); kappa %.4f; scales [%.3f, %.3f]; %.0f + %.0f s",
            beta, row["clip_high_inner"], row["clip_low_inner"], errs["+0.0"][0], errs["+0.0"][1], errs["90%"][0], errs["90%"][1],
            errs["-1.5"][0], errs["-2.5"][0], errs["-2.5"][1], errs["+1.5"][0], ed, ed_se,
            "" if ed_cc is None else f" = {ed / ed_cc:.5f} x E_CC[D]", row["EV_over_EQV"], row["EV_over_EQV_se"],
            row["names_part"], row["names_part_se"], row["basket_part"], row["basket_part_se"], kappa,
            row["scale_min"], row["scale_max"], t1 - t0, t2 - t1,
        )  # fmt: skip
    table = pd.DataFrame(rows)
    cols = [
        "beta",
        "clip_high_inner",
        "clip_low_inner",
        "idx_+0.0",
        "idx_90%",
        "idx_-1.5",
        "idx_-2.5",
        "idx_+1.5",
        "ED",
        "ED_se",
        "ED_over_cc",
        "EV_over_EQV",
        "names_over_listed",
        "basket_part",
        "kappa",
    ]
    log.info("\n%s", table[cols].to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    return {
        "date": date, "tenor": tenor, "budget": budget, "rows": rows, "E_CC_D_of_the_M12_row": ed_cc,
        "EQV": eqv, "sum_w_M": sum_wm, "M_B_listed": m_b,
        "record": {"git_commit": code_version(), "spec_key": lc_spec_key(spec), "particle_seed": spec.lc.particle.seed,
                   "pricing_seed": spec.sim.seed, "n_particles": spec.lc.particle.n_particles, "n_paths": spec.sim.n_paths,
                   "schedule": repr(spec.sim.step_schedule)},
    }  # fmt: skip


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--date", default="2026-10-02")
    ap.add_argument("--tenor", default="3m")
    ap.add_argument("--betas", default="0,1,2,3,5")
    ap.add_argument("--budget", default="development", choices=("production", "development"))
    ap.add_argument("--g-max", type=float, default=2.0)
    ap.add_argument(
        "--rows", default=None, help="the rows folder of an M12 sweep (for the companion's E[D])"
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("volsto").setLevel(logging.WARNING)
    betas = [float(b) for b in args.betas.split(",")]
    doc = scan(args.date, args.tenor, betas, args.budget, args.g_max, args.rows)
    out = Path(args.out) if args.out else lp.ROOT / "outputs" / "dispersion_lc" / "cdv"
    if "outputs/dispersion_lc" not in str(out.resolve()):
        raise ValueError("outputs must go under outputs/dispersion_lc")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"cdv_scan_{args.date}_{args.tenor}.json"
    path.write_text(json.dumps(doc, indent=1, default=lp.jsonable))
    log.info("written %s", path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
