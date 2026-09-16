"""M8b study D — the conditional-Greeks delta estimator measured against the analytic delta
(SPEC §8.2, "Study D reinstated"; the owner's decision of 2026-09-16: reproduce the diagnosis's
decisive Black–Scholes measurements).

Black–Scholes bed (flat 20% surface, ``r = 2%``, ``q = 1%``, spot 100, the Black–Scholes builder:
nothing is ever calibrated), 1y ATM call, ``delta only`` with the spot, world = pricing, the
world paths and seed held while the pricing (regression) sample varies:

1. **accuracy** — the hedger's fitted delta on every world path and rebalancing date against the
   analytic ``DF(T) N(d₁) F(T)/F(t)``: bias and RMSE pooled, by date bucket and by forward
   log-moneyness ``ln(S_t F(T)/(F(t) K))``, at each ``--paths`` count, with the §7.11 delta
   control off and on (:meth:`~volsto.hedging.pricing.ConditionalPricer.delta_control`);
2. **inflation** — the hedged P&L std of the estimator's hedge against the analytic-delta hedge
   on the same world paths and dates (the ratio, with a path-pair bootstrap se), and the std of
   the delta-error leg ``Σ_k (Δ_est − Δ_analytic) ΔI_k``;
3. **regime identity** — the four §7.2 surface regimes and ``min_variance`` coincide with the
   ``model`` regime under Black–Scholes: the maximum per-path P&L difference (bit-identical:
   exactly 0), control off and on;
4. **delta control** — the per-date variance reduction of the delta target
   (``Fit.variance_reduction["delta"]`` with its path-bootstrap se, ``β``, the fit's R²) under
   Black–Scholes and, with ``--lsv``, under the M8b 2F marking LSV read from the leverage cache
   (``allow_calibrate=False``: a cache miss stops the section with the reason).

Every Monte Carlo number carries its standard error: bias / RMSE se's are world-path-pair
standard errors (ratio estimators for the buckets) *conditional on the pricing sample* — the
part of the bias common to every date and path (the Monte Carlo error of the CRN bump's own
mean) is not in them, which is why the bias is read across ``--paths``.  Outputs:
``<out>/delta_estimator.csv`` (long format: ``section, pricing_paths, control, quantity, bucket,
value, value_se``) and ``<out>/delta_estimator.md`` (the tables, the wall clock, "recalibrated:
no").

Usage::

    .venv/bin/python scripts/m8b_delta_estimator.py
    .venv/bin/python scripts/m8b_delta_estimator.py --paths 5000,20000 --lsv
"""

from __future__ import annotations

import argparse
import itertools
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import norm

from volsto.calibration.cache import CacheMissError
from volsto.config import (
    BergomiParams,
    CalibrationSpec,
    CurveConfig,
    MarketConfig,
    SimConfig,
    SSVIConfig,
)
from volsto.hedging import (
    ConditionalPricer,
    Costs,
    GreekTargetStrategy,
    Hedger,
    PricingContext,
    Schedule,
    Spot,
    Target,
)
from volsto.hedging.hedger import FREQUENCIES, HedgeResult
from volsto.hedging.pricing import union_grid
from volsto.hedging.report import _se_std
from volsto.hedging.strategies import DELTA_REGIMES
from volsto.market.bs import black_price
from volsto.market.curves import ForwardCurve
from volsto.models.base import Model
from volsto.products.vanilla import EuropeanOption
from volsto.risk.engine import RiskState

FloatArray = NDArray[np.float64]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "outputs" / "m8b"
#: the Black–Scholes bed of the diagnosis (``tests/helpers.py::flat_state``)
BS_VOL = 0.2
BS_RATE = 0.02
BS_DIV = 0.01
BS_SPOT = 100.0
STRIKE = 100.0
MATURITY = 1.0
#: pricing / world seeds of the diagnosis (the world seed held across the path-count scan)
PRICING_SEED = 3
WORLD_SEED = 4
#: the study-D budget: 2·10⁴ world paths, the pricing counts scanned below and at it
DEFAULT_WORLD_PATHS = 20_000
DEFAULT_PATHS = (5_000, 20_000)
#: the regime-identity check runs at this (small) path count
IDENTITY_PATHS = 4_000
#: dates of the per-date delta-control read (years)
CONTROL_DATES = (0.02, 0.25, 0.5, 0.75, 0.95)
#: date buckets (years) and forward log-moneyness buckets of the accuracy tables
DATE_EDGES = (0.0, 0.25, 0.5, 0.75, 0.95, 1.0)
MONEYNESS_EDGES = (-np.inf, -0.30, -0.15, -0.05, 0.05, 0.15, 0.30, np.inf)
#: path-pair bootstrap draws of the std-ratio se
BOOTSTRAP = 200
BOOTSTRAP_SEED = 5


def flat_bs_state() -> RiskState:
    """The flat 20% state (a one-factor model the Black–Scholes builder ignores)."""
    spec = CalibrationSpec(
        market=MarketConfig(
            BS_SPOT, CurveConfig((1.0,), (BS_RATE,)), CurveConfig((1.0,), (BS_DIV,))
        ),
        surface=SSVIConfig((1 / 12, 0.25, 0.5, 1.0, 2.0, 3.0), (BS_VOL,) * 6, 0.0, 0.0, 0.5, 10.0),
        model=BergomiParams.one_factor(1.0, 1.0, 0.0),
    )
    return RiskState(spec)


def analytic_delta(s: FloatArray, t: float, fc: ForwardCurve) -> FloatArray:
    """``∂(discounted call payoff value)/∂S_t = DF(T) N(d₁) F(T)/F(t)`` (time-0 money)."""
    ratio = float(fc.forward(MATURITY) / fc.forward(t))
    tau = MATURITY - t
    f = np.asarray(s) * ratio
    d1 = (np.log(f / STRIKE) + 0.5 * BS_VOL * BS_VOL * tau) / (BS_VOL * np.sqrt(tau))
    return np.asarray(float(fc.rate_curve.df(MATURITY)) * norm.cdf(d1) * ratio)


def world_columns(result: HedgeResult) -> list[int]:
    gt = result.world_paths.times
    return [int(np.argmin(np.abs(gt - t))) for t in result.dates]


def analytic_hedge(result: HedgeResult, fc: ForwardCurve) -> tuple[FloatArray, FloatArray]:
    """The exact Black–Scholes delta hedge's P&L on the run's world paths and dates
    (``tests/test_hedging.py::_exact_delta_hedge``) and the per-date analytic deltas
    ``(n_dates, n_paths)``."""
    w = result.world_paths
    s = np.exp(w.log_spot)
    cols = world_columns(result)
    df_T = float(fc.rate_curve.df(MATURITY))
    pnl = np.zeros(w.n_paths)
    deltas = np.zeros((len(cols), w.n_paths))
    for k, c in enumerate(cols):
        t = float(w.times[c])
        c_next = cols[k + 1] if k + 1 < len(cols) else w.n_cols - 1
        t_next = float(w.times[c_next])
        deltas[k] = analytic_delta(s[:, c], t, fc)
        q = deltas[k] * float(fc.forward(t) / fc.spot)  # units of the total-return spot
        pnl += q * (
            s[:, c_next] * float(fc.spot / fc.forward(t_next))
            - s[:, c] * float(fc.spot / fc.forward(t))
        )
    payoff = np.maximum(s[:, -1] - STRIKE, 0.0) * df_T
    price = float(black_price(fc.forward(MATURITY), STRIKE, MATURITY, BS_VOL, 1)) * df_T
    return payoff - price - pnl, deltas


def pair_sums(x: FloatArray) -> FloatArray:
    """Sums over the antithetic world-path pairs along the last axis."""
    m = x.shape[-1] // 2
    return np.asarray(x[..., 0 : 2 * m : 2] + x[..., 1 : 2 * m : 2])


def ratio_stat(num: FloatArray, den: FloatArray) -> tuple[float, float]:
    """``Σ num / Σ den`` over paths with its pair-level standard error (delta method)."""
    n_p, d_p = pair_sums(num), pair_sums(den)
    m = n_p.size
    total = float(d_p.sum())
    if total <= 0.0 or m < 2:
        return float("nan"), float("nan")
    r = float(n_p.sum()) / total
    resid = n_p - r * d_p
    se = float(np.sqrt(np.sum(resid**2) / (m * (m - 1))) / (total / m))
    return r, se


def std_se(x: FloatArray) -> tuple[float, float]:
    return float(np.std(x, ddof=1)), float(_se_std(x))


def std_ratio(a: FloatArray, b: FloatArray) -> tuple[float, float]:
    """``std(a)/std(b)`` on the same paths with a path-pair bootstrap se."""
    r = float(np.std(a, ddof=1) / np.std(b, ddof=1))
    m = a.size // 2
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    boot = np.empty(BOOTSTRAP)
    for i in range(BOOTSTRAP):
        pick = rng.integers(0, m, size=m)
        ix = np.concatenate([2 * pick, 2 * pick + 1])
        boot[i] = np.std(a[ix], ddof=1) / np.std(b[ix], ddof=1)
    return r, float(boot.std(ddof=1))


def run_hedge(
    ctx: PricingContext,
    product: EuropeanOption,
    regime: str,
    n_pricing: int,
    n_world: int,
    frequency: str,
    control: bool,
) -> HedgeResult:
    sim = SimConfig(
        n_paths=n_pricing,
        chunk_size=n_pricing,
        seed=PRICING_SEED,
        dt_max=min(FREQUENCIES[frequency], 1.0 / 52.0),
    )
    h = Hedger(
        ctx,
        ctx.model,
        Schedule(frequency),
        Costs(),
        sim=sim,
        world_paths=n_world,
        world_seed=WORLD_SEED,
        verbose=False,
        control_delta=control,
    )
    strat = GreekTargetStrategy(
        (Target("delta"),), [Spot()], delta_regime=regime, name=f"delta only ({regime})"
    )
    return h.run(product, strat)


class Rows:
    """The long-format output table."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        section: str,
        n: int,
        control: str,
        quantity: str,
        bucket: str,
        value: tuple[float, float],
    ) -> None:
        self.rows.append(
            {
                "section": section,
                "pricing_paths": n,
                "control": control,
                "quantity": quantity,
                "bucket": bucket,
                "value": value[0],
                "value_se": value[1],
            }
        )

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def accuracy(
    rows: Rows,
    ctx: PricingContext,
    product: EuropeanOption,
    paths: Sequence[int],
    n_world: int,
    frequency: str,
) -> None:
    fc = ctx.model.forward_curve
    for n in paths:
        for control in (False, True):
            tag = "on" if control else "off"
            t0 = time.perf_counter()
            r = run_hedge(ctx, product, "model", n, n_world, frequency, control)
            exact, d_an = analytic_hedge(r, fc)
            d_est = np.asarray(r.greeks_by_date["delta"])
            err = d_est - d_an
            w = r.world_paths
            cols = world_columns(r)
            ones = np.ones_like(err)
            rows.add("accuracy", n, tag, "bias", "all", ratio_stat(err.sum(0), ones.sum(0)))
            mse = ratio_stat((err**2).sum(0), ones.sum(0))
            rows.add("accuracy", n, tag, "rmse", "all", _rmse(mse))
            for lo, hi in itertools.pairwise(DATE_EDGES):
                sel = (r.dates >= lo) & (r.dates < hi)
                e = err[sel]
                b = f"t in [{lo:g}, {hi:g})"
                rows.add(
                    "accuracy", n, tag, "bias", b, ratio_stat(e.sum(0), np.ones_like(e).sum(0))
                )
                rows.add(
                    "accuracy",
                    n,
                    tag,
                    "rmse",
                    b,
                    _rmse(ratio_stat((e**2).sum(0), np.ones_like(e).sum(0))),
                )
            fwd_k = np.stack(
                [
                    w.log_spot[:, c]
                    + float(np.log(fc.forward(MATURITY) / fc.forward(float(t))))
                    - float(np.log(STRIKE))
                    for c, t in zip(cols, r.dates, strict=True)
                ]
            )
            for lo, hi in itertools.pairwise(MONEYNESS_EDGES):
                m = ((fwd_k > lo) & (fwd_k <= hi)).astype(np.float64)
                b = f"k in ({lo:g}, {hi:g}]"
                rows.add("accuracy", n, tag, "bias", b, ratio_stat((err * m).sum(0), m.sum(0)))
                rows.add(
                    "accuracy",
                    n,
                    tag,
                    "rmse",
                    b,
                    _rmse(ratio_stat((err**2 * m).sum(0), m.sum(0))),
                )
            # the hedged std against the analytic-delta bound, and the delta-error leg
            est = r.pnl_total
            leg = np.zeros(w.n_paths)
            for k, c in enumerate(cols):
                c_next = cols[k + 1] if k + 1 < len(cols) else w.n_cols - 1
                t, t_next = float(w.times[c]), float(w.times[c_next])
                i_now = np.exp(w.log_spot[:, c]) * float(fc.spot / fc.forward(t))
                i_next = np.exp(w.log_spot[:, c_next]) * float(fc.spot / fc.forward(t_next))
                leg += err[k] * float(fc.forward(t) / fc.spot) * (i_next - i_now)
            rows.add("inflation", n, tag, "std_estimator", "", std_se(est))
            rows.add("inflation", n, tag, "std_analytic", "", std_se(exact))
            rows.add("inflation", n, tag, "std_ratio", "", std_ratio(est, exact))
            rows.add("inflation", n, tag, "std_delta_error_leg", "", std_se(leg))
            # the median over dates of the delta target's variance reduction, with the median of
            # the per-date bootstrap se's
            red = r.budget.get("cv_delta_reduction_median", float("nan"))
            red_se = r.budget.get("cv_delta_reduction_se_median", float("nan"))
            rows.add("inflation", n, tag, "cv_delta_reduction_median", "", (red, red_se))
            print(
                f"[accuracy] {n} pricing paths, control {tag}: std {np.std(est, ddof=1):.4f} vs "
                f"analytic {np.std(exact, ddof=1):.4f}; {time.perf_counter() - t0:.0f} s",
                flush=True,
            )


def _rmse(mse: tuple[float, float]) -> tuple[float, float]:
    v, se = mse
    if not np.isfinite(v) or v <= 0.0:
        return float("nan"), float("nan")
    return float(np.sqrt(v)), float(se / (2.0 * np.sqrt(v)))


def identity(
    rows: Rows, ctx: PricingContext, product: EuropeanOption, n: int, frequency: str
) -> None:
    for control in (False, True):
        tag = "on" if control else "off"
        base = run_hedge(ctx, product, "model", n, n, frequency, control)
        rows.add("identity", n, tag, "std", "model", std_se(base.pnl_total))
        for regime in DELTA_REGIMES:
            if regime == "model":
                continue
            r = run_hedge(ctx, product, regime, n, n, frequency, control)
            diff = float(np.max(np.abs(r.pnl_total - base.pnl_total)))
            # an exact identity: the difference is a deterministic 0, its se is 0
            rows.add("identity", n, tag, "max_abs_pnl_difference", regime, (diff, 0.0))
            rows.add("identity", n, tag, "std", regime, std_se(r.pnl_total))
        print(f"[identity] {n} paths, control {tag}: done", flush=True)


def control_by_date(rows: Rows, label: str, model: Model, surface: Any, n: int) -> None:
    """The delta control's per-date variance reduction on a 1y ATM call under ``model``."""
    fc = model.forward_curve
    call = EuropeanOption(float(fc.spot), MATURITY, 1, fc.rate_curve)
    sim = SimConfig(n_paths=n, chunk_size=n, seed=PRICING_SEED, dt_max=1.0 / 252.0)
    dates = np.asarray(CONTROL_DATES)
    grid = union_grid([model], [call], dates, sim)
    t0 = time.perf_counter()
    pr = ConditionalPricer(model, [call], grid, sim, surface=surface, control_delta=True)
    for t in dates:
        f = pr.fit(0, float(t))
        b = f"t = {float(t):g}"
        if "delta" not in f.controlled:
            rows.add("control", n, label, "variance_reduction", b, (float("nan"), float("nan")))
            continue
        vr = (f.variance_reduction["delta"], f.variance_reduction_se["delta"])
        rows.add("control", n, label, "variance_reduction", b, vr)
        # beta and R² are fit diagnostics of this sample, reported without an se (as Fit does)
        rows.add("control", n, label, "beta", b, (f.beta["delta"], float("nan")))
        rows.add("control", n, label, "r2_controlled_target", b, (f.r2["delta"], float("nan")))
    pr.release()
    print(f"[control] {label}: {time.perf_counter() - t0:.0f} s", flush=True)


def markdown(df: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    for _, rec in df.iterrows():
        cells = []
        for c in df.columns:
            v = rec[c]
            cells.append(f"{v:.5g}" if isinstance(v, float | np.floating) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def pm(df: pd.DataFrame) -> pd.DataFrame:
    """``value ± se`` cells for the markdown."""
    out = df.copy()
    out["value ± se"] = [
        (
            "n/a (control off)"
            if not np.isfinite(v)
            else f"{v:.5g} ± {s:.2g}" if np.isfinite(s) else f"{v:.5g} (fit diagnostic, no se)"
        )
        for v, s in zip(out["value"], out["value_se"], strict=True)
    ]
    return out.drop(columns=["value", "value_se"])


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--paths", default=",".join(str(p) for p in DEFAULT_PATHS))
    ap.add_argument("--world-paths", type=int, default=DEFAULT_WORLD_PATHS)
    ap.add_argument("--frequency", default="daily", choices=tuple(FREQUENCIES))
    ap.add_argument("--identity-paths", type=int, default=IDENTITY_PATHS)
    ap.add_argument("--lsv", action="store_true", help="add the 2F marking LSV control section")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    paths = [int(x) for x in args.paths.split(",") if x.strip()]
    t_all = time.perf_counter()
    ctx = PricingContext.from_state(flat_bs_state(), None, "bs", label="BS 20%")
    call = EuropeanOption(STRIKE, MATURITY, 1, ctx.model.forward_curve.rate_curve)
    rows = Rows()
    accuracy(rows, ctx, call, paths, args.world_paths, args.frequency)
    identity(rows, ctx, call, args.identity_paths, args.frequency)
    control_by_date(rows, "BS", ctx.model, ctx.surface, max(paths))
    lsv_note = "2F marking LSV section not run (pass --lsv)"
    calibrations = 0
    if args.lsv:
        from volsto.studies.m8b import StudyConfig, StudyEnvironment

        env = StudyEnvironment(StudyConfig(allow_calibrate=False, verbose=False))
        try:
            lctx = env.pricing_ctx
            control_by_date(rows, "2F marking LSV", lctx.model, lctx.surface, max(paths))
            lsv_note = f"2F marking LSV from the cache at {env.cfg.n_particles} particles"
        except CacheMissError as exc:
            lsv_note = f"2F marking LSV section skipped: leverage not cached ({exc})"
        calibrations = env.calibrations
    wall = time.perf_counter() - t_all
    df = rows.frame()
    args.out.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out / "delta_estimator.csv", index=False)
    md = [
        "# M8b study D — the delta estimator against the analytic delta",
        "",
        f"- wall clock {wall:.0f} s; recalibrated: no ({calibrations} leverage calibrations)",
        f"- Black-Scholes bed: flat {BS_VOL:.0%}, r {BS_RATE:.0%}, q {BS_DIV:.0%}, spot "
        f"{BS_SPOT:g}; 1y ATM call; {args.frequency} rebalancing; world paths "
        f"{args.world_paths} (seed {WORLD_SEED}); pricing seed {PRICING_SEED}",
        f"- {lsv_note}",
        "- bias / RMSE se's: world-path-pair standard errors conditional on the pricing sample "
        "(the bias common to every date — the CRN bump mean's own Monte Carlo error — is read "
        "across the pricing path counts)",
        "",
    ]
    for section, title in (
        ("inflation", "Hedged std: estimator against the analytic-delta hedge (same paths)"),
        ("accuracy", "Delta bias and RMSE against the analytic delta (delta units)"),
        ("identity", "Regime identity under Black-Scholes (per-path P&L)"),
        ("control", "Delta control: per-date variance reduction of the delta target"),
    ):
        md += [f"## {title}", "", markdown(pm(df[df["section"] == section])), ""]
    (args.out / "delta_estimator.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"[m8b_delta_estimator] wrote {args.out / 'delta_estimator.csv'} and .md; {wall:.0f} s")


if __name__ == "__main__":
    main()
