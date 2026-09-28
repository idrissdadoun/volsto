"""Product-specific risks (SPEC v2 §7.10): fixing risk, barrier risk and the realised-variance
exposure profile.

* :func:`fixing_risk` — for a forward-start option / straddle the delta, gamma and vega one
  business day before and one after the strike fixing ``T1`` with the state held (spot at
  the fixing = today's spot): before, the contract is the same forward start with ``T1 = 1d``;
  after, it is the vanilla struck at ``k·S₀`` with ``T2 − T1 − 1d`` to run (payment deferral
  carried as a discount-factor ratio on the notional).  For an additive cliquet the same per
  intermediate fixing ``t_j``: with the path held flat every fixed leg has return 0, so "before"
  is the cliquet on ``[0, 1d, 1d + (t_{j+1} − t_j), …]`` and "after" the cliquet on ``[0, (t_{j+1}
  − t_j) − 1d, …]`` with the same local / global bounds.  The jump ``after − before`` is the
  vega-to-delta conversion at the fixing (the forward start carries vega and almost no delta
  before ``T1`` and becomes a plain vanilla after it).
* :func:`barrier_sensitivity` — ``∂price/∂B`` of a knock-out variance swap or a spot barrier
  option (knock-out / knock-in; central, ±0.5% of the barrier, reported per unit spot and per 1%
  of the barrier) and ``∂price/∂H`` of the vol-knock-out / knock-in put per vol point of the vol
  barrier; :func:`barrier_shift_table` the price of a barrier option or knock-out variance swap
  with its barrier moved by a few signed relative shifts (positive: away from the spot) against
  the contractual price, paired — the barrier-shift reserve of a conservative mark;
  :func:`ko_probability_delta` gives ``∂P(KO)/∂ln S`` from the ``"ko"`` statistic leg under the
  ``"model"`` regime;
  :func:`barrier_profile` the model delta / gamma profile within ±5% of a spot barrier in 0.5%
  steps (:func:`~volsto.risk.profiles.spot_profile`).
* :func:`realised_variance_exposure` — the exposure of the price to the realised variance of
  each period bucket along the path: the period log-returns inside the bucket are rescaled by
  ``(1 ± ε)`` on the simulated paths (later log-spots shifted accordingly) and the symmetric
  second difference ``E[P₊ + P₋ − 2P]`` — the dollar-gamma-weighted variance of the bucket, the
  first-order spot effect cancelling — is normalised by the bucket's contribution to the
  annualised realised variance, ``2ε² E[Σ_{i∈b} r_i²]/(T − t₀)``.  For a variance swap with a
  daily schedule this is exactly ``N · DF(T)`` in every bucket (the payoff is linear in the
  squared returns), the identity the profile is checked against; it reads the fixing log-spots,
  so products integrating the simulation-grid variance are outside its scope.

Checked by ``tests/test_risk_product.py``.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.config import SimConfig
from volsto.engine.mc import MonteCarlo
from volsto.models.base import Model
from volsto.products.barrier import _BarrierOption
from volsto.products.base import Portfolio, Product
from volsto.products.cliquet import AdditiveCliquet
from volsto.products.conditional_variance import KnockOutVarianceSwap, StatisticLeg
from volsto.products.forward_start import ForwardStartOption, ForwardStartStraddle
from volsto.products.vanilla import EuropeanOption
from volsto.products.vko import VolKnockOutPut
from volsto.risk.engine import RiskEngine, RiskState, Sensitivity
from volsto.risk.greeks import BUSINESS_DAY, _spot_state, delta_gamma, vega
from volsto.risk.profiles import spot_profile

FloatArray = NDArray[np.float64]


# --------------------------------------------------------------------------------------------
# fixing risk
# --------------------------------------------------------------------------------------------


def _none_if_inf(x: float) -> float | None:
    return None if not np.isfinite(x) else float(x)


def _forward_start_legs(
    product: ForwardStartOption | ForwardStartStraddle, spot: float, dt: float
) -> tuple[Product, Product]:
    """(before, after) contracts around the strike fixing with the state held."""
    t1, t2, pay = product.T1, product.T2, product.pay_time
    if t1 <= dt:
        raise ValueError("the strike fixing must lie more than one roll step ahead")
    tau2 = t2 - t1 - dt
    tau_pay = pay - t1 - dt
    df_ratio = float(product.df(tau_pay) / product.df(tau2))
    k_abs = product.strike * spot
    n_after = product.notional / spot * df_ratio
    if isinstance(product, ForwardStartStraddle):
        before: Product = ForwardStartStraddle(
            dt, t2 - t1 + dt, product.strike, product.discount, product.notional, pay - t1 + dt
        )
        after: Product = Portfolio(
            [
                EuropeanOption(k_abs, tau2, 1, product.discount, n_after),
                EuropeanOption(k_abs, tau2, -1, product.discount, n_after),
            ]
        )
    else:
        before = ForwardStartOption(
            dt,
            t2 - t1 + dt,
            product.strike,
            product.cp,
            product.discount,
            product.notional,
            pay - t1 + dt,
        )
        after = EuropeanOption(k_abs, tau2, product.cp, product.discount, n_after)
    return before, after


def _cliquet_legs(cliquet: AdditiveCliquet, j: int, dt: float) -> tuple[Product, Product]:
    """(before, after) cliquets around the ``j``-th fixing with the path held flat."""
    ft = cliquet.fixing_times
    if not 0 < j < ft.size - 1:
        raise ValueError("j must index an intermediate fixing")
    rest = ft[j + 1 :] - ft[j]
    if rest[0] <= dt:
        raise ValueError("the next fixing lies inside the roll step")

    def make(times: FloatArray) -> AdditiveCliquet:
        return AdditiveCliquet(
            times,
            cliquet.discount,
            local_floor=_none_if_inf(cliquet.local_floor),
            local_cap=_none_if_inf(cliquet.local_cap),
            global_floor=_none_if_inf(cliquet.global_floor),
            global_cap=_none_if_inf(cliquet.global_cap),
            notional=cliquet.notional,
        )

    before = make(np.concatenate([[0.0, dt], rest + dt]))
    after = make(np.concatenate([[0.0], rest - dt]))
    return before, after


def _greeks_row(
    engine: RiskEngine, product: Product, state: RiskState, size: float, tag: str
) -> dict[str, float]:
    d, g = delta_gamma(engine, product, state, "model", size)
    v = vega(engine, product, state, "recalibrated", size)
    p = engine.price(product, state)
    return {
        f"price_{tag}": p.mean,
        f"delta_{tag}": d.value,
        f"delta_{tag}_stderr": d.stderr,
        f"gamma_{tag}": g.value,
        f"gamma_{tag}_stderr": g.stderr,
        f"vega_{tag}": v.value,
        f"vega_{tag}_stderr": v.stderr,
    }


def _jump_columns(row: dict[str, float]) -> dict[str, float]:
    out = dict(row)
    for g in ("delta", "gamma", "vega"):
        out[f"{g}_jump"] = row[f"{g}_after"] - row[f"{g}_before"]
        out[f"{g}_jump_stderr"] = float(
            np.hypot(row[f"{g}_after_stderr"], row[f"{g}_before_stderr"])
        )
    return out


def fixing_risk(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    dt: float = BUSINESS_DAY,
    size: float = 0.01,
) -> pd.DataFrame:
    """Delta / gamma (``"model"`` regime) and recalibrated vega one business day before and
    after each fixing with the state held, and the jump; one row per fixing (a forward start
    has one, a cliquet one per intermediate fixing)."""
    rows = []
    if isinstance(product, ForwardStartOption | ForwardStartStraddle):
        before, after = _forward_start_legs(product, state.spot, dt)
        row: dict[str, float] = {"fixing": product.T1}
        row.update(_greeks_row(engine, before, state, size, "before"))
        row.update(_greeks_row(engine, after, state, size, "after"))
        rows.append(_jump_columns(row))
    elif isinstance(product, AdditiveCliquet):
        ft = product.fixing_times
        for j in range(1, ft.size - 1):
            before, after = _cliquet_legs(product, j, dt)
            row = {"fixing": float(ft[j])}
            row.update(_greeks_row(engine, before, state, size, "before"))
            row.update(_greeks_row(engine, after, state, size, "after"))
            rows.append(_jump_columns(row))
    else:
        raise TypeError("fixing risk is defined for forward-start options and additive cliquets")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# barrier risk
# --------------------------------------------------------------------------------------------


def _with_barrier(product: KnockOutVarianceSwap, barrier: float) -> KnockOutVarianceSwap:
    return KnockOutVarianceSwap(
        product.fixing_times,
        barrier,
        product.strike_vol,
        product.discount,
        direction=product.direction,
        strict=product.strict,
        monitoring=product.monitoring,
        variant=product.variant,
        settlement=product.settlement,
        daily_cap=product.daily_cap,
        annualisation=product.annualisation,
        notional=product.notional,
        **product.state_kwargs(),
    )


def _barrier_level(product: KnockOutVarianceSwap | _BarrierOption) -> float:
    return float(product.barrier)


def _moved(
    product: KnockOutVarianceSwap | _BarrierOption, barrier: float
) -> KnockOutVarianceSwap | Product:
    if isinstance(product, KnockOutVarianceSwap):
        return _with_barrier(product, barrier)
    return product.with_barrier(barrier)


def _with_vol_barrier(product: VolKnockOutPut, vol_ko: float) -> VolKnockOutPut:
    return VolKnockOutPut(
        product.strike,
        product.T,
        vol_ko,
        product.fixing_times,
        product.discount,
        daily_cap=product.daily_cap,
        annualisation=product.annualisation,
        notional=product.notional,
        knock_in=product.knock_in,
    )


def barrier_sensitivity(
    engine: RiskEngine,
    product: KnockOutVarianceSwap | VolKnockOutPut | _BarrierOption,
    state: RiskState,
    size: float = 0.005,
) -> dict[str, Sensitivity]:
    """``∂price/∂B`` (knock-out variance swap or spot barrier option: central ±``size``
    relative; per unit spot and per 1% of the barrier) or ``∂price/∂H`` (vol knock-out /
    knock-in put: central ±``size`` in vol units; per vol point)."""
    out: dict[str, Sensitivity] = {}
    if isinstance(product, KnockOutVarianceSwap | _BarrierOption):
        b = _barrier_level(product)
        up, dn = _moved(product, b * (1 + size)), _moved(product, b * (1 - size))
        d_b = engine.paired(
            "dprice/dB",
            [
                (up, state, "recalibrate", 1.0 / (2 * b * size)),
                (dn, state, "recalibrate", -1.0 / (2 * b * size)),
            ],
            unit="per unit barrier",
            size=size,
            scheme="central",
        )
        out["dprice_dB"] = d_b
        out["dprice_dB_pct"] = dataclasses.replace(
            d_b,
            name="dprice/dB[1%]",
            value=d_b.value * 0.01 * b,
            stderr=d_b.stderr * 0.01 * b,
            unit="per 1% of barrier",
        )
    elif isinstance(product, VolKnockOutPut):
        h = product.vol_ko
        up_v, dn_v = _with_vol_barrier(product, h + size), _with_vol_barrier(product, h - size)
        out["dprice_dH"] = engine.paired(
            "dprice/dH",
            [
                (up_v, state, "recalibrate", 0.01 / (2 * size)),
                (dn_v, state, "recalibrate", -0.01 / (2 * size)),
            ],
            unit="per vol point of vol barrier",
            size=size,
            scheme="central",
        )
    else:
        raise TypeError(
            "barrier sensitivity is defined for KO variance swaps, VKO puts and barrier options"
        )
    return out


def barrier_shift_table(
    engine: RiskEngine,
    product: KnockOutVarianceSwap | _BarrierOption,
    state: RiskState,
    shifts: tuple[float, ...] = (-0.02, -0.01, -0.005, 0.005, 0.01, 0.02),
) -> pd.DataFrame:
    """The price with the barrier moved by each signed relative shift — positive moves it AWAY
    from the spot (an up barrier higher, a down barrier lower) — against the contractual price,
    as a paired difference (common paths, leverage recalibrated): ``shift, barrier, delta_price,
    stderr``.  For a long knock-out a shift away from the spot adds value, for a long knock-in it
    removes it: the table is the barrier-shift reserve of a conservative mark in either
    direction, never a single convention."""
    b = _barrier_level(product)
    sign = 1.0 if product.direction == "up" else -1.0
    rows = []
    for s in shifts:
        level = b * (1.0 + sign * float(s))
        d = engine.paired(
            f"price(B {s:+.1%})",
            [
                (_moved(product, level), state, "recalibrate", 1.0),
                (product, state, "recalibrate", -1.0),
            ],
            unit="price",
            size=float(s),
            scheme="forward",
        )
        rows.append(
            {"shift": float(s), "barrier": level, "delta_price": d.value, "stderr": d.stderr}
        )
    return pd.DataFrame(rows)


def ko_probability_delta(
    engine: RiskEngine,
    product: KnockOutVarianceSwap | VolKnockOutPut,
    state: RiskState,
    size: float = 0.01,
) -> Sensitivity:
    """``∂P(KO)/∂ln S`` from the ``"ko"`` statistic leg, central under the ``"model"`` regime."""
    leg = StatisticLeg(product, "ko")
    up, _ = _spot_state(state, "model", size)
    dn, _ = _spot_state(state, "model", -size)
    return engine.combination(
        "dP(KO)/dlnS",
        leg,
        [(up, "model", 1.0 / (2 * size)), (dn, "model", -1.0 / (2 * size))],
        unit="per unit ln S",
        size=size,
        scheme="central",
    )


def barrier_profile(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    barrier: float,
    *,
    width: float = 0.05,
    step: float = 0.005,
    size: float = 0.01,
) -> pd.DataFrame:
    """Model delta / gamma with the spot within ``±width`` of ``barrier`` in ``step`` steps."""
    rel = np.arange(-width, width + 1e-12, step)
    shifts = tuple(float(x) for x in barrier / state.spot * (1.0 + rel) - 1.0)
    frame = spot_profile(engine, product, state, shifts, regime="model", size=size, with_vega=False)
    frame.insert(1, "spot_over_barrier", frame["spot"] / barrier)
    return frame


# --------------------------------------------------------------------------------------------
# realised-variance exposure
# --------------------------------------------------------------------------------------------


def _period_buckets(fixings: FloatArray, n_buckets: int) -> list[tuple[int, int]]:
    """``(c_lo, c_hi)`` fixing-index ranges: periods ``c_lo+1 … c_hi`` whose end falls in each of
    ``n_buckets`` equal time buckets over ``[t₀, T]``."""
    t0, T = float(fixings[0]), float(fixings[-1])
    edges = t0 + (T - t0) * np.arange(1, n_buckets + 1) / n_buckets
    out = []
    lo = 0
    for e in edges:
        hi = int(np.searchsorted(fixings, e + 1e-12, side="right") - 1)
        if hi > lo:
            out.append((lo, hi))
            lo = hi
    return out


def realised_variance_exposure(
    product: Product,
    model: Model,
    sim: SimConfig,
    *,
    n_buckets: int = 12,
    eps: float = 0.05,
) -> pd.DataFrame:
    """Exposure of the price to the realised variance of each period bucket (module docstring):
    ``E[P₊ + P₋ − 2P] · (T − t₀) / (2ε² E[Σ_{i∈b} r_i²])`` with the bucket's period returns
    rescaled by ``1 ± ε`` on the paths; per unit of annualised variance."""
    fixings = np.asarray(product.fixing_times, dtype=np.float64)
    if fixings[0] > 0:
        fixings = np.concatenate([[0.0], fixings])
    buckets = _period_buckets(fixings, n_buckets)
    mc = MonteCarlo(sim)
    grid = mc.build_grid([product], model)
    draws = mc.draws_for(grid, model)
    idx = grid.fixing_index
    cols = idx.indices(fixings)
    num = [np.empty(0)] * len(buckets)
    den = [np.empty(0)] * len(buckets)
    num_parts: list[list[FloatArray]] = [[] for _ in buckets]
    den_parts: list[list[FloatArray]] = [[] for _ in buckets]
    life = float(fixings[-1] - fixings[0])
    for p0, p1 in sim.chunk_ranges(grid.n_records, model.n_factors):
        paths = model.simulate_chunk(grid, draws, p0, p1, sim.scheme)
        base = product.payoff(paths, idx)
        ls = paths.log_spot
        for b, (lo, hi) in enumerate(buckets):
            c_lo, c_hi = int(cols[lo]), int(cols[hi])
            seg = ls[:, c_lo : c_hi + 1] - ls[:, [c_lo]]
            tail = ls[:, [c_hi]] - ls[:, [c_lo]]
            vals = []
            sq = paths.sum_sq
            sq_seg = sq[:, c_lo : c_hi + 1] - sq[:, [c_lo]]
            sq_tail = sq[:, [c_hi]] - sq[:, [c_lo]]
            for s in (1.0 + eps, 1.0 - eps):
                mod = ls.copy()
                mod[:, c_lo : c_hi + 1] = ls[:, [c_lo]] + s * seg
                mod[:, c_hi + 1 :] = ls[:, c_hi + 1 :] + (s - 1.0) * tail
                # the squared-increment accumulator follows the same homothety (the barrier
                # products check it against the recorded log-returns, M6)
                sq_mod = sq.copy()
                sq_mod[:, c_lo : c_hi + 1] = sq[:, [c_lo]] + s * s * sq_seg
                sq_mod[:, c_hi + 1 :] = sq[:, c_hi + 1 :] + (s * s - 1.0) * sq_tail
                vals.append(
                    product.payoff(dataclasses.replace(paths, log_spot=mod, sum_sq=sq_mod), idx)
                )
            num_parts[b].append(vals[0] + vals[1] - 2.0 * base)
            r2 = np.diff(ls[:, cols[lo : hi + 1]], axis=1) ** 2
            den_parts[b].append(np.sum(r2, axis=1))
    rows = []
    for b, (lo, hi) in enumerate(buckets):
        num[b] = np.concatenate(num_parts[b])
        den[b] = np.concatenate(den_parts[b])
        if sim.antithetic:
            num[b] = 0.5 * (num[b][0::2] + num[b][1::2])
        scale = life / (2.0 * eps * eps * float(den[b].mean()))
        vals_b = num[b] * scale
        rows.append(
            {
                "bucket": b,
                "t_lo": float(fixings[lo]),
                "t_hi": float(fixings[hi]),
                "n_periods": hi - lo,
                "exposure": float(vals_b.mean()),
                "stderr": float(vals_b.std(ddof=1) / np.sqrt(vals_b.size)),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "barrier_profile",
    "barrier_sensitivity",
    "barrier_shift_table",
    "fixing_risk",
    "ko_probability_delta",
    "realised_variance_exposure",
]
