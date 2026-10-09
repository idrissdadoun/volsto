"""Local correlation model (SPEC §8.7, M12): the synthetic acceptance runs on another step
schedule, and the tail rule on the synthetic world — for information.

    python scripts/lcm_synthetic.py runs --schedule week [--tests s1 s2 s5 s10]
    python scripts/lcm_synthetic.py tail [--schedule lc]
    python scripts/lcm_synthetic.py where --case s1|s3 [--schedule lc]

The acceptance tests S1, S2, S5 and S10 of ``tests/test_local_correlation.py`` run on the model's
step schedule (quarter steps over the first two weeks, then daily: the owner's decision 2 of
2026-10-08).  ``runs`` executes the same test bodies — the same worlds, seeds, budgets and
gates — on another schedule (``week``: quarter steps over the first week only; ``daily``: 1/252
throughout, the specification's first choice) and prints their tables; a gate that fails is
reported, not raised.  Nothing here changes the model's schedule.

``tail`` compares the two ``λ`` tail rules, ``"regressions"`` and ``"flat"``, on the synthetic
world W5 at 3m (its analytic target): the Palladium forward and calls, the basket variance swap
on daily fixings and the far-wing index repricing, on the same particles and the same pricing
paths.

``where`` locates the two gates that fail on the model's schedule (SPEC §8.7, LC4 follow-up):
for S1 the clipped mass inside ±2.5 sd slice by slice, before and after the target's first
pillar, as designed and with a two-week pillar added to the target; for S3 the smallest ``λ̂``
slice by slice.

The numbers of SPEC §8.7 labelled "one-week variant", "tail rule, W5" and "where" come from
here; the Dow ones from ``scripts/lcm_diagnostics.py``.
"""

# ruff: noqa: E501
from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lcm_diagnostics as lcd  # noqa: E402
import test_local_correlation as t  # noqa: E402

from volsto.calibration import local_correlation as lcal  # noqa: E402
from volsto.config import SimConfig  # noqa: E402
from volsto.engine.grid import TimeGrid  # noqa: E402
from volsto.market.dupire import LocalVolSurface  # noqa: E402
from volsto.models.localvol import LocalVol  # noqa: E402
from volsto.multi.family import CorrelationFamily  # noqa: E402
from volsto.multi.lc_function import LocalCorrelationFunction  # noqa: E402
from volsto.multi.lc_model import LocalCorrelationModel  # noqa: E402

log = logging.getLogger("lcm_synthetic")

TESTS = {
    "s1": "test_s1_round_trip",
    "s2": "test_s2_constant_correlation_fixed_point",
    "s5": "test_s5_dt_halving_w5",
    "s10": "test_s10_carry_mode",
}


def run_tests(args: argparse.Namespace) -> None:
    schedule = lcd.SCHEDULES[args.schedule]
    t.ACCEPTANCE_SCHEDULE = schedule
    log.info(
        "=== the acceptance runs on the schedule %r (%s), for information", args.schedule, schedule
    )
    for name in args.tests:
        start = time.perf_counter()
        try:
            getattr(t, TESTS[name])()
            verdict = "every gate passes"
        except AssertionError as exc:
            first = str(exc).strip().splitlines()[0] if str(exc).strip() else "an assertion"
            verdict = f"a gate fails ({first[:160]})"
        log.info(
            "--- %s on the schedule %r: %s (%.0f s)",
            name.upper(),
            args.schedule,
            verdict,
            time.perf_counter() - start,
        )


def run_tail(args: argparse.Namespace) -> None:
    schedule = lcd.SCHEDULES[args.schedule]
    T = 0.25
    models = t.w5_models(T)
    fam = CorrelationFamily.equi(5)
    basket = t.w5_basket(models)
    surface, _ = t.w5_target(T)
    w = t.W5_WEIGHTS
    sim = t.production_sim(schedule)
    got = {}
    for tail in ("regressions", "flat"):
        res = t.calibrate(models, fam, basket, surface, T, sim=sim, lambda_tail=tail)
        model = LocalCorrelationModel(models, fam, res.lam, basket)
        # every step recorded: the basket's realised variance on the grid's fixings
        rec = SimConfig(
            n_paths=sim.n_paths,
            dt_max=schedule,
            seed=sim.seed,
            chunk_size=20_000,
            record_all_steps=True,
        )
        grid = TimeGrid.build(
            [T], rec.dt_max, calibration_grid=model.required_times(), record_all_steps=True
        )
        draws = model.draws_for(grid, rec.seed, rec.n_paths)
        d, rv = np.empty(rec.n_paths), np.empty(rec.n_paths)
        daily = [int(np.argmin(np.abs(grid.times - x))) for x in np.arange(0, 64) / 252.0]
        for p0, p1 in rec.chunk_ranges(grid.n_records * model.n_assets, 0):
            paths = model.simulate_chunk(grid, draws, p0, p1, rec.scheme)
            r = paths.performances(grid.n_records - 1)
            mean = r @ w
            d[p0:p1] = np.abs(r - mean[:, None]) @ w
            assert paths.aux is not None
            rv[p0:p1] = np.sum(np.diff(paths.aux["k_basket"][:, daily], axis=1) ** 2, axis=1) / T
        rep = lcal.reprice_index_smile(model, surface, sim, maturities=t.W5_3M_PILLARS)
        got[tail] = (res, d, rv, rep)
    a, b = got["regressions"], got["flat"]
    base = lcd.pair_mean(a[1])[0]
    rows = []
    for mlt in (0.0, 0.75, 1.0, 1.25, 1.5, 2.0):
        x, y = np.maximum(a[1] - mlt * base, 0.0), np.maximum(b[1] - mlt * base, 0.0)
        diff, se = lcd.pair_mean(y - x)
        rows.append({"quantity": "E[D]" if mlt == 0 else f"call {mlt:.2f}", "regressions": lcd.pair_mean(x)[0],
                     "se": lcd.pair_mean(x)[1], "flat": lcd.pair_mean(y)[0], "flat_minus_regressions": diff,
                     "paired_se": se, "rel_%": 100 * diff / lcd.pair_mean(x)[0]})  # fmt: skip
    diff, se = lcd.pair_mean(b[2] - a[2])
    va, vb = lcd.pair_mean(a[2])[0], lcd.pair_mean(b[2])[0]
    rows.append({"quantity": "basket variance swap (variance, daily fixings)", "regressions": va, "se": lcd.pair_mean(a[2])[1],
                 "flat": vb, "flat_minus_regressions": diff, "paired_se": se, "rel_%": 100 * diff / va})  # fmt: skip
    log.info("=== tail rule on W5 3m (schedule %s): N = %d, paths = %d\n%s", args.schedule, t.PRODUCTION, sim.n_paths,
             pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6g}"))  # fmt: skip
    log.info("basket variance swap vol strike: regressions %.4f vp, flat %.4f vp, difference %+.4f vp (paired se %.4f)",
             100 * np.sqrt(va), 100 * np.sqrt(vb), 100 * (np.sqrt(vb) - np.sqrt(va)), 100 * se / (2 * np.sqrt(va)))  # fmt: skip
    for tail, (res, _, _, rep) in got.items():
        far = rep.table[rep.table["sd"].abs() >= 2.0]
        cells = far.assign(
            cell=[
                f"{e:+.3f} ({s:.3f})"
                for e, s in zip(far["error_vp"], far["stderr_vp"], strict=True)
            ]
        )
        log.info("%s: clipped mass max low %.4f, high %.4f; index error at |sd| >= 2, vp (se):\n%s", tail,
                 res.clipped_low.max(), res.clipped_high.max(), cells.pivot(index="T", columns="sd", values="cell").to_string())  # fmt: skip


def where_s3() -> None:
    """S3 (five identical names, the truth ``λ ≡ 1``): the smallest ``λ̂`` on the trusted range,
    slice by slice, on two calibration seeds."""
    horizon = 0.25
    surface = t.w5_surfaces(2)[1]
    name = LocalVol(
        LocalVolSurface.from_implied(surface, t.lc_grid(horizon)), surface.forward_curve
    )
    models = [name] * 5
    fam = CorrelationFamily.equi(5, 0.02, 1.0)
    basket = t.w5_basket(models)
    for seed in t.NOISE_SEEDS[:2]:
        res = t.calibrate(models, fam, basket, surface, horizon, rho_max=1.0, seed=seed)
        k, times = res.lam.k_grid, res.grid.times
        low, at = [], []
        for j in range(1, res.lam.n_slices):
            inside = (k >= res.q_lo[j]) & (k <= res.q_hi[j])
            row = res.lam.values[j][inside]
            low.append(float(row.min()))
            at.append(float(k[inside][int(np.argmin(row))]))
        lows = np.array(low)
        early = times[1:] <= 10 / 252 + 1e-12
        log.info(
            "S3 seed %d: smallest lambda on the trusted range %.6f; %d of %d slices below 0.99; in the first two weeks "
            "min %.6f, after them min %.6f; smallest cloud mean of lambda %.6f",
            seed, lows.min(), int((lows < 0.99).sum()), lows.size, lows[early].min(), lows[~early].min(),
            res.lambda_mean[1:].min(),
        )  # fmt: skip
        for i in np.argsort(lows)[:6]:
            j = int(i) + 1
            sd = float(surface.atm_vol(max(times[j], 1 / 365))) * np.sqrt(times[j])
            log.info(
                "    t = %.5f (step %d): min lambda %.6f at k = %+.4f (%+.2f sd); bandwidth %.5f; cloud mean %.6f",
                times[j], j, lows[i], at[i], at[i] / sd, res.bandwidths[j - 1], res.lambda_mean[j],
            )  # fmt: skip


def where_s1() -> None:
    """S1 (the round trip at 1y): the clipped mass inside ±2.5 sd slice by slice — as designed
    (monthly pillars and 13m) and with a two-week pillar added to the target."""
    horizon, beyond = 1.0, 13 / 12
    pillars = tuple(m / 12 for m in range(1, 13))
    models = t.w5_models(beyond)
    fam = CorrelationFamily.equi(5)
    basket = t.w5_basket(models)

    def truth(k: np.ndarray) -> np.ndarray:
        return np.asarray(0.5 - 0.4 * np.tanh(np.asarray(k) / 0.25))

    kg = t.k_grid_of(models)
    lam_true = LocalCorrelationFunction([0.0, beyond], kg, np.tile(truth(kg), (2, 1)))
    true_model = LocalCorrelationModel(models, fam, lam_true, basket)
    for label, target in (
        ("as designed", (*pillars, beyond)),
        ("with a two-week pillar", (1 / 24, *pillars, beyond)),
    ):
        surface, _ = t.index_smile_by_simulation(
            true_model, t.production_sim(n_paths=2_000_000), target
        )
        res = t.calibrate(models, fam, basket, surface, horizon)
        times = res.grid.times
        inner = np.maximum(res.clipped_low_inner, res.clipped_high_inner)
        first = min(target)
        before = times < first - 1e-12
        rep = lcal.reprice_index_smile(
            true_model.with_lambda(res.lam), surface, t.production_sim(), maturities=pillars
        )
        log.info(
            "S1 %s (first pillar %.4f): clipped mass inside ±2.5 sd max %.4f at t = %.4f, %d slices over 1 %%; before the "
            "first pillar max %.4f, from it on max %.5f; repricing max %.3f vp inside ±1.5 sd, %.3f inside ±2.5 sd",
            label, first, inner.max(), times[int(np.argmax(inner))], int((inner > 0.01).sum()), inner[before].max(),
            inner[~before].max(), rep.max_abs_error(1.5), rep.max_abs_error(2.5),
        )  # fmt: skip
        for j in np.flatnonzero(inner > 0.002):
            sd = float(surface.atm_vol(times[j])) * np.sqrt(times[j])
            cuts = [
                (a, round(float(b / sd), 2), round(float(c / sd), 2))
                for a, b, c in res.clip_intervals(int(j))
            ]
            log.info(
                "    t = %.4f: inside %.4f (all of the cloud: %.4f); clipped on, in sd: %s; lambda_true at -2.5 sd %.3f, at -2 sd %.3f; cap %.4f",
                times[j], inner[j], res.clipped_high[j], cuts, float(truth(-2.5 * sd)), float(truth(-2.0 * sd)), fam.lambda_max,
            )  # fmt: skip


def run_where(args: argparse.Namespace) -> None:
    t.ACCEPTANCE_SCHEDULE = lcd.SCHEDULES[args.schedule]
    log.info(
        "=== where the gate of %s fails, on the schedule %r (%s)",
        args.case.upper(),
        args.schedule,
        t.ACCEPTANCE_SCHEDULE,
    )
    {"s1": where_s1, "s3": where_s3}[args.case]()


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("runs", help="the acceptance test bodies on another schedule")
    p.add_argument("--schedule", default="week", choices=sorted(lcd.SCHEDULES))
    p.add_argument("--tests", nargs="+", default=list(TESTS), choices=sorted(TESTS))
    p.set_defaults(run=run_tests)
    p = sub.add_parser("tail", help="the lambda tail rule on W5")
    p.add_argument("--schedule", default="lc", choices=sorted(lcd.SCHEDULES))
    p.set_defaults(run=run_tail)
    p = sub.add_parser("where", help="where the gates of S1 and S3 fail")
    p.add_argument("--case", required=True, choices=("s1", "s3"))
    p.add_argument("--schedule", default="lc", choices=sorted(lcd.SCHEDULES))
    p.set_defaults(run=run_where)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    logging.getLogger("volsto").setLevel(logging.WARNING)
    args.run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
