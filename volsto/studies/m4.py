"""M4 headline comparison (SPEC §10 regression numbers, owner request at M4).

For every model calibrated to the same surface — pure local vol (ω = 0), the one-factor LSV
with ω ∈ {1, 2, 3} (ρ = −0.7, κ = 1.5) and the two-factor Table 8.2 set — price on one path set:

* the 1y-into-1y forward smile (out-of-the-money forward-start options on a moneyness grid, the
  ATM-forward strike included) → forward ATM vol and the put-wing table,
* the 1y → 2y forward variance swap (simulation grid) and forward vol swap (daily fixings),
* the study cliquets (monthly, local cap 2%, no local floor, global floor 0) of 1y and 2y.

Every number carries its standard error; cliquet prices are in % of notional.  The regression
test ``tests/test_m4_regression.py`` re-runs :func:`run_headline` with the same seed and
compares to the stored baselines with tolerance ``max(2 stderr, 0.02% of notional)``.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from volsto.analytics.conditional_variance import mean_and_stderr, pair_average, ratio_of_means
from volsto.analytics.forward_smile import (
    ForwardSmile,
    forward_ratio,
    forward_smile_from_prices,
    put_wing_table,
)
from volsto.calibration.cache import LeverageCache, build_market
from volsto.config import BergomiParams, CalibrationSpec, SimConfig
from volsto.engine.mc import MonteCarlo
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import SSVISurface
from volsto.models.base import Model
from volsto.models.localvol import LocalVol
from volsto.products.base import Product, daily_schedule
from volsto.products.cliquet import AdditiveCliquet
from volsto.products.conditional_variance import DownVar, KnockOutVarianceSwap, UpVar
from volsto.products.forward_start import ForwardStartOption
from volsto.products.variance import VarianceSwap, VolSwap
from volsto.products.vko import VolKnockOutPut

log = logging.getLogger(__name__)

HEADLINE_STRIKES: tuple[float, ...] = (0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2)
HEADLINE_OMEGAS: tuple[float, ...] = (1.0, 2.0, 3.0)
LV_NAME = "LV (ω=0)"
TWO_FACTOR_NAME = "2F Table 8.2"


def one_factor_name(omega: float) -> str:
    return f"1F ω={omega:g}"


def one_factor_variants(
    base: CalibrationSpec,
    omegas: Sequence[float] = HEADLINE_OMEGAS,
    kappa: float = 1.5,
    rho: float = -0.7,
) -> dict[str, CalibrationSpec]:
    """The base spec with the model replaced by the 1F set ``(ω, κ, ρ)`` for each ``ω``."""
    return {
        one_factor_name(w): dataclasses.replace(base, model=BergomiParams.one_factor(w, kappa, rho))
        for w in omegas
    }


def headline_models(
    cache: LeverageCache,
    base_1f: CalibrationSpec,
    spec_2f: CalibrationSpec | None,
    *,
    omegas: Sequence[float] = HEADLINE_OMEGAS,
    n_particles: int | None = None,
) -> tuple[dict[str, Model], SSVISurface]:
    """Pure LV plus the calibrated LSV models (cache hits or fresh calibrations).

    ``n_particles`` overrides the specs' particle count: production numbers (headline tables,
    regression baselines, viewer precompute) use 8·10⁵ with a single seed (owner decision at
    M4b acceptance), development paths keep the specs' 2·10⁵.
    """
    if n_particles is not None:
        base_1f = dataclasses.replace(
            base_1f, particle=dataclasses.replace(base_1f.particle, n_particles=n_particles)
        )
        if spec_2f is not None:
            spec_2f = dataclasses.replace(
                spec_2f, particle=dataclasses.replace(spec_2f.particle, n_particles=n_particles)
            )
    _, surface, _ = build_market(base_1f)
    models: dict[str, Model] = {
        LV_NAME: LocalVol(LocalVolSurface.from_implied(surface, base_1f.local_vol))
    }
    for name, spec in one_factor_variants(base_1f, omegas).items():
        t0 = time.perf_counter()
        models[name], _ = cache.get_or_calibrate(spec)
        log.info("%s ready in %.0f s", name, time.perf_counter() - t0)
    if spec_2f is not None:
        if spec_2f.surface != base_1f.surface or spec_2f.market != base_1f.market:
            raise ValueError("the 2F spec must target the same market and surface")
        t0 = time.perf_counter()
        models[TWO_FACTOR_NAME], _ = cache.get_or_calibrate(spec_2f)
        log.info("%s ready in %.0f s", TWO_FACTOR_NAME, time.perf_counter() - t0)
    return models, surface


@dataclass(frozen=True)
class HeadlineResult:
    """``table``: one row per model; ``smiles``: forward smiles; ``wing``: put-wing table."""

    table: pd.DataFrame
    smiles: dict[str, ForwardSmile]
    wing: pd.DataFrame
    t1: float
    t2: float
    n_paths: int

    def to_markdown(self) -> str:
        t = self.table
        cliquets = [c for c in t.columns if c.startswith("cliquet_") and not c.endswith("_stderr")]
        head = [f"{self.t1:g}y→{self.t2:g}y ATMF fwd vol", "fwd VS vol", "fwd vol-swap vol"]
        head += [f"{c.replace('cliquet_', 'cliquet ')} (% notional)" for c in cliquets]
        lines = ["| model | " + " | ".join(head) + " | wall (s) |", "|---" * (len(head) + 2) + "|"]
        for _, r in t.iterrows():
            cells = [
                f"{r[k] * 100:.2f} ± {r[k + '_stderr'] * 100:.2f}"
                for k in ("atm_vol", "vs_vol", "volswap_vol")
            ]
            cells += [f"{r[c]:.3f} ± {r[c + '_stderr']:.3f}" for c in cliquets]
            lines.append(f"| {r['model']} | " + " | ".join(cells) + f" | {r['wall_s']:.0f} |")
        if "upvar_100" in t.columns:
            lines += [
                "",
                "| model | up-var B=100% (fair vol) | down-var B=100% | KO var B=110% | P(KO) | "
                "VKO 12m 100% put @30% (% notional) | VKO discount | P(KO) |",
                "|---|---|---|---|---|---|---|---|",
            ]
            for _, r in t.iterrows():
                lines.append(
                    f"| {r['model']} | {r['upvar_100'] * 100:.2f} ± "
                    f"{r['upvar_100_stderr'] * 100:.2f}"
                    f" | {r['downvar_100'] * 100:.2f} ± {r['downvar_100_stderr'] * 100:.2f}"
                    f" | {r['kovar_110'] * 100:.2f} ± {r['kovar_110_stderr'] * 100:.2f}"
                    f" | {r['kovar_110_p_ko']:.3f} ± {r['kovar_110_p_ko_stderr']:.3f}"
                    f" | {r['vko_30']:.3f} ± {r['vko_30_stderr']:.3f}"
                    f" | {r['vko_30_discount']:.3f} ± {r['vko_30_discount_stderr']:.3f}"
                    f" | {r['vko_30_p_ko']:.3f} ± {r['vko_30_p_ko_stderr']:.3f} |"
                )
        w = self.wing
        names = [c[:-4] for c in w.columns if c.endswith("_vol")]
        lines += [
            "",
            "| k | " + " | ".join(names) + " | spread (vp) | spread z |",
            "|---" * (len(names) + 3) + "|",
        ]
        for _, r in w.iterrows():
            cells = [f"{r[n + '_vol'] * 100:.2f} ± {r[n + '_se'] * 100:.2f}" for n in names]
            lines.append(
                f"| {r['k']:.4g} | "
                + " | ".join(cells)
                + f" | {r['spread'] * 100:.2f} | {r['spread_z']:.1f} |"
            )
        return "\n".join(lines)


def run_headline(
    models: Mapping[str, Model],
    sim: SimConfig,
    *,
    t1: float = 1.0,
    t2: float = 2.0,
    strikes: Sequence[float] = HEADLINE_STRIKES,
    cliquet_maturities: Sequence[float] = (1.0, 2.0),
    per_year: int = 252,
    conditional: bool = True,
    vko_barrier: float = 0.30,
) -> HeadlineResult:
    """Price the headline set for every model on one path set each (same seed across models).

    With ``conditional`` (M4c) the set also carries the 1y daily up-var and down-var swaps at
    B = 100% of spot, the 1y up-and-out KO variance swap at B = 110% (fair vols, P(KO)) and the
    12m 100% VKO put at ``vko_barrier`` (price in % of notional, ratio to the vanilla put, P(KO)).
    """
    rows = []
    smiles: dict[str, ForwardSmile] = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        discount = model.forward_curve.rate_curve
        f_r = forward_ratio(model, t1, t2)
        ks = np.unique(np.append(np.asarray(strikes, dtype=np.float64), f_r))
        cps = np.where(ks >= f_r, 1, -1).astype(np.int64)
        products: list[Product] = [
            ForwardStartOption(t1, t2, float(k), int(c), discount) for k, c in zip(ks, cps)
        ]
        products.append(VarianceSwap([t1, t2], 0.0, discount, use_simulation_grid=True))
        products.append(VolSwap.daily(t2, 0.0, discount, start=t1, per_year=per_year))
        products += [AdditiveCliquet.study(T, discount) for T in cliquet_maturities]
        n_base = len(products)
        cond: dict[str, Product] = {}
        if conditional:
            times_d = daily_schedule(1.0, per_year)
            spot = model.spot
            up = UpVar(times_d, spot, 0.0, discount)
            down = DownVar(times_d, spot, 0.0, discount)
            ko = KnockOutVarianceSwap(times_d, 1.1 * spot, 0.0, discount)
            # notional 1/spot: prices as a fraction of the (spot) notional, reported in %
            vko = VolKnockOutPut(spot, 1.0, vko_barrier, times_d, discount, notional=1.0 / spot)
            cond = {
                "up_acc": up.leg("accrued"),
                "up_cnt": up.leg("count"),
                "down_acc": down.leg("accrued"),
                "down_cnt": down.leg("count"),
                "ko_acc": ko.leg("accrued"),
                "ko_cnt": ko.leg("count"),
                "ko_ko": ko.leg("ko"),
                "vko": vko,
                "vko_van": vko.vanilla(),
                "vko_ko": vko.leg("ko"),
            }
            products += list(cond.values())
        res = MonteCarlo(sim).price_many(products, model, keep_payoffs=conditional)
        n_k = ks.size
        df = float(discount.df(t2))
        smile = forward_smile_from_prices(t1, t2, f_r, ks, cps, res[:n_k], df, sim.n_paths)
        smiles[name] = smile
        atm_vol, atm_se = smile.atm
        k_var, se_var = res[n_k].mean / df, res[n_k].stderr / df
        vs_vol = float(np.sqrt(k_var))
        row: dict[str, float | str] = {
            "model": name,
            "atm_vol": atm_vol,
            "atm_vol_stderr": atm_se,
            "vs_vol": vs_vol,
            "vs_vol_stderr": se_var / (2.0 * vs_vol),
            "volswap_vol": res[n_k + 1].mean / df,
            "volswap_vol_stderr": res[n_k + 1].stderr / df,
        }
        for T, r in zip(cliquet_maturities, res[n_k + 2 : n_base], strict=True):
            tag = f"cliquet_{T:g}y"
            row[tag] = 100.0 * r.mean
            row[tag + "_stderr"] = 100.0 * r.stderr
        if conditional:
            pay = {
                name: pair_average(np.asarray(res[n_base + j].payoffs), sim.antithetic)
                for j, name in enumerate(cond)
            }
            for tag, acc, cnt in (
                ("upvar_100", "up_acc", "up_cnt"),
                ("downvar_100", "down_acc", "down_cnt"),
                ("kovar_110", "ko_acc", "ko_cnt"),
            ):
                k2, se2 = ratio_of_means(pay[acc], pay[cnt])
                row[tag] = float(np.sqrt(k2))
                row[tag + "_stderr"] = se2 / (2.0 * float(np.sqrt(k2)))
            row["kovar_110_p_ko"], row["kovar_110_p_ko_stderr"] = mean_and_stderr(pay["ko_ko"])
            price, van = pay["vko"], pay["vko_van"]
            row["vko_30"], row["vko_30_stderr"] = (100.0 * x for x in mean_and_stderr(price))
            ratio, se_ratio = ratio_of_means(price, van)
            row["vko_30_discount"], row["vko_30_discount_stderr"] = ratio, se_ratio
            row["vko_30_p_ko"], row["vko_30_p_ko_stderr"] = mean_and_stderr(pay["vko_ko"])
        row["wall_s"] = time.perf_counter() - t0
        rows.append(row)
        log.info("%s: %s", name, row)
    table = pd.DataFrame(rows)
    return HeadlineResult(table, smiles, put_wing_table(smiles), t1, t2, sim.n_paths)
