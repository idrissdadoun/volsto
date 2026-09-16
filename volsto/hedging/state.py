"""Hedge state of a product along recorded paths (SPEC §8, M8 Part 1 support): the extra
regression features, the alive flag and the settled value that make the conditional-pricing
regression of :mod:`volsto.hedging.pricing` Markov in the recorded state.

The conditional value of a product at a rebalancing date ``t`` is regressed on ``(ln S_t, X_t)``
plus the **product's own state** — the path-dependent quantities its remaining cash flows depend
on.  :func:`hedge_state` returns, for a product and a column ``t`` of a :class:`PathSet`:

* ``features`` ``(n_paths, m)`` — the extra regression features (empty for a vanilla);
* ``alive`` ``(n_paths,)`` — False once the product has terminated early (autocall called,
  barrier knocked out, knock-out variance swap stopped, VKO knocked out) so that the hedger
  unwinds its hedges and holds the settled value;
* ``settled`` ``(n_paths,)`` — the discounted (time-0 money) realised value of the product on the
  paths that are no longer alive (their total payoff, known at ``t``); NaN where alive.

Registered products (the hedge state is *derived from the payoff*, decision of SPEC §8 Part 3:
"each product's structure defines what needs hedging"):

======================================  ================================================
product                                 extra features / termination
======================================  ================================================
``EuropeanOption``, ``DigitalOption``   none
``ForwardStartOption / Straddle``,      ``u_start = ln(S_t/S_{T1})`` once ``t ≥ T1`` (0 before,
``FVA``                                 with the indicator ``1{t ≥ T1}``)
``VarianceSwap``, ``VolSwap``           accrued sum of squared fixing returns to ``t`` and
                                        the log return of the period in progress ``u``
``AdditiveCliquet``, ``ReverseCliquet``,  accumulated local sum of the completed periods and
``AccumulatedSumOption``                the period's log return so far ``u``
``ConditionalVarianceSwap``             accrued in-region squared returns, count and ``u``
``KnockOutVarianceSwap``                accrued squared returns and ``u``, alive = not stopped
                                        before ``t``
``VolKnockOutPut``                      accrued squared returns and ``u``; knock-out variant:
                                        alive = the variance budget not yet exhausted
``Autocall`` (and Phoenix)              alive = not autocalled at an observation date ``≤ t``;
                                        knock-in status, coupons in memory
``KnockOutOption``, ``KnockInOption``,  alive = not knocked at a monitoring date ``≤ t``
``_Touch``                              (discrete monitoring; continuous monitoring falls back
                                        to the discrete schedule with a note)
``Portfolio``                           the settled legs' discounted cash (one feature) plus the
                                        live legs' features de-duplicated; alive when any leg is
``CashFlow``                            none
======================================  ================================================

Any other product gets no extra feature and a note (``HedgeState.notes``): its conditional value
is then regressed on the spot and factor state only, which is exact for state-independent
payoffs and an approximation otherwise — reported, never silent.  **Why the period start
matters:** the conditional delta of a path-dependent object is the derivative of its value in
``S_t`` with the history held; the CRN spot bump of §7.11 scales the whole path, past fixings
included (a forward-start leg inside its period, a cliquet's current period, an accrued variance
swap then show a zero bump delta), so :mod:`volsto.hedging.pricing` differentiates the fitted value
in ``ln S_t`` for every object with a hedge state and needs the state to carry the fixed history
the remaining payoff depends on.

Checked by ``tests/test_hedging.py::test_hedge_state_features``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from volsto.engine.grid import FixingIndex
from volsto.engine.paths import PathSet
from volsto.products.base import CashFlow, Portfolio, Product

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
_TOL = 1e-9


@dataclass
class HedgeState:
    """The hedge state at one column (module docstring)."""

    t: float
    features: FloatArray
    alive: BoolArray
    settled: FloatArray
    names: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def n_paths(self) -> int:
        return int(self.alive.size)

    @property
    def any_terminated(self) -> bool:
        return bool((~self.alive).any())


def _empty(n: int, t: float, notes: tuple[str, ...] = ()) -> HedgeState:
    return HedgeState(
        float(t), np.zeros((n, 0)), np.ones(n, dtype=bool), np.full(n, np.nan), (), notes
    )


def _fixings_up_to(fixings: FloatArray, t: float) -> FloatArray:
    return np.asarray(fixings[fixings <= t + _TOL], dtype=np.float64)


def _u_period(paths: PathSet, idx: FixingIndex, fixings: FloatArray, t: float) -> FloatArray:
    """``ln S_t − ln S_start``, the log return of the period in progress (``S_start`` the last
    fixing ``≤ t``; 0 at a fixing date and before the first fixing) — a feature that moves one
    for one with ``ln S_t`` when the history is held, hence its ``u_`` prefix."""
    done = _fixings_up_to(fixings, t)
    col_t = idx[float(t)]
    if done.size == 0:
        return np.zeros(paths.n_paths)
    start = paths.log_spot_at(idx[float(done[-1])])
    return np.asarray(paths.log_spot_at(col_t) - start, dtype=np.float64)


def _accrued_sq(paths: PathSet, idx: FixingIndex, fixings: FloatArray, t: float) -> FloatArray:
    """Sum of squared log returns over the fixings ``≤ t`` (0 with fewer than two)."""
    done = _fixings_up_to(fixings, t)
    if done.size < 2:
        return np.zeros(paths.n_paths)
    ls = paths.log_spot_at(idx.indices(done))
    return np.asarray(np.sum(np.diff(ls, axis=1) ** 2, axis=1), dtype=np.float64)


def _terminated(
    paths: PathSet, idx: FixingIndex, product: Product, alive: BoolArray, t: float
) -> FloatArray:
    settled = np.full(paths.n_paths, np.nan)
    if (~alive).any():
        pay = product.payoff(paths, idx)
        settled[~alive] = pay[~alive]
    return settled


def hedge_state(product: Product, paths: PathSet, idx: FixingIndex, t: float) -> HedgeState:
    """The hedge state of ``product`` at time ``t`` (a recorded column of ``paths``)."""
    from volsto.products.autocall import Autocall
    from volsto.products.barrier import _BarrierBase
    from volsto.products.cliquet import AccumulatedSumOption, AdditiveCliquet, ReverseCliquet
    from volsto.products.conditional_variance import (
        ConditionalVarianceSwap,
        KnockOutVarianceSwap,
    )
    from volsto.products.forward_start import ForwardStartOption, ForwardStartStraddle
    from volsto.products.vanilla import DigitalOption, EuropeanOption
    from volsto.products.variance import FVA, VarianceSwap, VolSwap
    from volsto.products.vko import VolKnockOutPut

    n = paths.n_paths
    t = float(t)
    if isinstance(product, EuropeanOption | DigitalOption | CashFlow):
        return _empty(n, t)
    if isinstance(product, ForwardStartOption | ForwardStartStraddle | FVA):
        t1 = float(product.T1)
        if t < t1 - _TOL:
            return HedgeState(
                t,
                np.zeros((n, 2)),
                np.ones(n, dtype=bool),
                np.full(n, np.nan),
                ("started", "u_start"),
            )
        # the started leg's value depends on ln(S_t / S_{T1}) alone: a moving feature
        u = paths.log_spot_at(idx[t]) - paths.log_spot_at(idx[t1])
        feats = np.column_stack([np.ones(n), u])
        return HedgeState(
            t, feats, np.ones(n, dtype=bool), np.full(n, np.nan), ("started", "u_start")
        )
    if isinstance(product, VarianceSwap | VolSwap):
        acc = _accrued_sq(paths, idx, product.fixing_times, t)
        start = _u_period(paths, idx, product.fixing_times, t)
        return HedgeState(
            t,
            np.column_stack([acc, start]),
            np.ones(n, dtype=bool),
            np.full(n, np.nan),
            ("accrued_sq", "u_period"),
        )
    if isinstance(product, AccumulatedSumOption):
        return hedge_state(product.cliquet, paths, idx, t)
    if isinstance(product, AdditiveCliquet | ReverseCliquet):
        inner = product if isinstance(product, AdditiveCliquet) else product._inner()
        fx = inner.fixing_times
        done = _fixings_up_to(fx, t)
        if done.size < 2:
            acc = np.zeros(n)
        else:
            ls = paths.log_spot_at(idx.indices(done))
            r = np.exp(np.diff(ls, axis=1)) - 1.0
            acc = np.sum(np.clip(r, inner.local_floor, inner.local_cap), axis=1)
        start = _u_period(paths, idx, fx, t)
        return HedgeState(
            t,
            np.column_stack([acc, start]),
            np.ones(n, dtype=bool),
            np.full(n, np.nan),
            ("accumulated", "u_period"),
        )
    if isinstance(product, ConditionalVarianceSwap):
        done = _fixings_up_to(product.fixing_times, t)
        if done.size < 2:
            feats = np.zeros((n, 2))
        else:
            ls = paths.log_spot_at(idx.indices(done))
            ind = product.indicators(ls)
            r2 = product.squared_returns(ls)
            feats = np.column_stack([np.sum(r2 * ind, axis=1), np.sum(ind, axis=1)])
        start = _u_period(paths, idx, product.fixing_times, t)
        return HedgeState(
            t,
            np.column_stack([feats, start]),
            np.ones(n, dtype=bool),
            np.full(n, np.nan),
            ("accrued_in", "count_in", "u_period"),
        )
    if isinstance(product, KnockOutVarianceSwap):
        from volsto.products.conditional_variance import _in_region

        done = _fixings_up_to(product.fixing_times, t)
        alive = np.ones(n, dtype=bool)
        acc = np.zeros(n)
        if done.size >= 1:
            ls = paths.log_spot_at(idx.indices(done))
            hit = _in_region(ls, np.log(product.barrier), product.direction, product.strict)
            alive = ~hit.any(axis=1)  # a close beyond the barrier at a done fixing stops it
            if done.size >= 2:
                acc = np.sum(product.squared_returns(ls), axis=1)
        settled = _terminated(paths, idx, product, alive, t)
        start = _u_period(paths, idx, product.fixing_times, t)
        return HedgeState(
            t, np.column_stack([acc, start]), alive, settled, ("accrued_sq", "u_period")
        )
    if isinstance(product, VolKnockOutPut):
        acc = _accrued_sq(paths, idx, product.fixing_times, t)
        if product.daily_cap is not None:
            done = _fixings_up_to(product.fixing_times, t)
            if done.size >= 2:
                ls = paths.log_spot_at(idx.indices(done))
                acc = np.sum(product.squared_returns(ls), axis=1)
        budget = product.vol_ko**2 * product.n_returns / product.annualisation
        alive = np.ones(n, dtype=bool)
        if not product.knock_in:
            alive = acc < budget
        settled = _terminated(paths, idx, product, alive, t)
        start = _u_period(paths, idx, product.fixing_times, t)
        return HedgeState(
            t, np.column_stack([acc, start]), alive, settled, ("accrued_sq", "u_period")
        )
    if isinstance(product, Autocall):
        obs = product.observation_times
        past = obs[obs <= t + _TOL]
        alive = np.ones(n, dtype=bool)
        if past.size:
            spots = paths.spot_at(idx.indices(past))
            levels = product.autocall_levels[: past.size]
            alive = ~(spots >= levels[None, :]).any(axis=1)
        ki = product._breached_by(paths, idx, t).astype(np.float64)
        memory = np.zeros(n)
        if product.is_phoenix and product.memory and past.size:
            coupons = product.coupon_amounts(paths, idx)
            memory = product._memory_at(product._memory_state_by_date(coupons), obs, t)
        feats = np.column_stack([ki, memory])
        settled = _terminated(paths, idx, product, alive, t)
        return HedgeState(t, feats, alive, settled, ("ki", "memory"))
    if isinstance(product, _BarrierBase):
        notes: tuple[str, ...] = ()
        sched = np.asarray(product.schedule, dtype=np.float64)
        done = sched[sched <= t + _TOL]
        alive = np.ones(n, dtype=bool)
        if getattr(product, "monitoring", "discrete") != "discrete":
            notes = ("continuous barrier monitoring: the hedge state uses the discrete schedule",)
        if done.size:
            ls = paths.log_spot_at(idx.indices(done))
            from volsto.products.barrier import first_hit_index

            j = first_hit_index(
                ls, product.ln_barrier, product.direction, strict=bool(product.strict)
            )
            alive = j >= done.size
        knock = getattr(product, "knock", "out")
        if knock == "in":
            # a knocked-in option stays alive (it became the vanilla); the knock status is a feature
            feats = (~alive).astype(np.float64)[:, None]
            return HedgeState(
                t, feats, np.ones(n, dtype=bool), np.full(n, np.nan), ("knocked",), notes
            )
        settled = _terminated(paths, idx, product, alive, t)
        return HedgeState(t, np.zeros((n, 0)), alive, settled, (), notes)
    if isinstance(product, Portfolio):
        # the settled legs (every fixing <= t) are a known cash amount per path: one feature; the
        # live legs contribute their own features, identical columns de-duplicated (a straddle's
        # call and put share u_start); alive when any leg is alive
        settled_sum = np.zeros(n)
        cols: list[FloatArray] = []
        names: list[str] = []
        pnotes: list[str] = []
        alive = np.zeros(n, dtype=bool)
        any_live = False
        for i, (w, leg) in enumerate(zip(product.weights, product.legs, strict=True)):
            fx = np.asarray(leg.fixing_times, dtype=np.float64)
            if fx.size and float(fx.max()) <= t + _TOL:
                settled_sum += float(w) * leg.payoff(paths, idx)
                continue
            any_live = True
            p_ = hedge_state(leg, paths, idx, t)
            alive |= p_.alive
            pnotes += [nn for nn in p_.notes if nn not in pnotes]
            for jj, nm in enumerate(p_.names):
                column = p_.features[:, jj]
                if any(np.array_equal(column, c) for c in cols):
                    continue
                cols.append(column)
                names.append(f"leg{i}:{nm}")
        if not any_live:
            alive = np.ones(n, dtype=bool)
        feats = np.column_stack([settled_sum, *cols]) if cols else settled_sum[:, None]
        names = ["settled", *names]
        if len(names) > 8:
            pnotes.append(
                f"Portfolio hedge state with {len(names)} features: the regression basis is large"
            )
        settled = _terminated(paths, idx, product, alive, t)
        return HedgeState(t, feats, alive, settled, tuple(names), tuple(pnotes))
    return _empty(
        n,
        t,
        (
            f"{type(product).__name__}: no hedge state registered - the conditional value is "
            "regressed on the spot and factor state only",
        ),
    )


def state_features(
    product: Product, paths: PathSet, idx: FixingIndex, t: float
) -> tuple[FloatArray, HedgeState]:
    """``(ln S_t, X_t, product features)`` stacked, and the :class:`HedgeState`."""
    hs = hedge_state(product, paths, idx, t)
    col = idx[float(t)]
    base = [paths.log_spot_at(col)]
    if paths.n_factors:
        base.append(paths.factors_at(col))
    feats = np.column_stack([*base, hs.features]) if hs.features.shape[1] else np.column_stack(base)
    return np.asarray(feats, dtype=np.float64), hs


__all__ = ["HedgeState", "hedge_state", "state_features"]
