"""Single-underlying autocall and Phoenix notes (SPEC §6.6, M6 Part 2).

Term sheet (levels are fractions of the reference spot ``S_ref = spot_reference``; observation
dates ``T_1 < … < T_N``, maturity ``T_N``; ``K = S_ref`` is the put strike):

* Autocall event at ``T_i`` (``i > non_call_periods``): the first date with ``S_{T_i} ≥ AC_i
  S_ref`` redeems par at ``T_i`` and terminates (``AC_i = +inf`` disables the event at that date).
  Plain autocall (no coupon barrier): the redemption carries the coupon ``c_i`` (``c_i = i · c``
  for a float ``coupons = c`` — the growing coupon — or ``coupons[i-1]``, the total paid at that
  date, for a sequence).
* Phoenix (``coupon_barrier = CB``): at every observation date while alive (the autocall date
  included) the period coupon ``c_i`` (``c`` for a float, ``coupons[i-1]`` for a sequence) is
  paid if ``S_{T_i} ≥ CB S_ref``; with ``memory`` the coupons missed since the last payment are
  paid with it.  ``guaranteed_coupons``: the period coupon is paid at every date while alive,
  unconditionally (no coupon barrier, no memory).
* Final date, not autocalled: par less the geared put ``(K − S_{T_N})⁺ / K`` when the knock-in
  condition triggered; ``ki_type = "european"``: ``S_{T_N} < KI S_ref``; ``"american"``:
  ``min S < KI S_ref`` over the monitoring schedule (``ki_monitoring = "discrete"``, daily by
  default, the schedule ending at ``T_N``) or continuously (``"continuous"``: the Brownian-bridge
  survival weight of :func:`volsto.products.barrier.continuous_survival_weight` over the life,
  the knock-in weight being ``1 − survival``; every step must be recorded, so the instance sets
  ``requires_all_steps``).  Every discrete comparison with the knock-in level is made in spot
  space on ``paths.spot_at`` (strict ``S < B``), the same array the put payoff and the European
  components use, so the identity below is exact including at the barrier (``ln S < ln B`` and
  ``S < B`` disagree one ulp below ``B``); the continuous variant inherits the barrier module's
  convention (a recorded log-spot at or below ``ln B`` knocks).
* ``final_redemption`` (Phoenix only; an explicit convention, no market default is assumed
  silently): ``"knock_in"`` — the coupon decision at ``T_N`` and the knock-in redemption are
  independent, a path that knocked in and ends between ``CB`` and ``K`` receives the coupon and
  ``S_{T_N} / K`` (the standard term sheet); ``"coupon_barrier"`` — the literal SPEC §6.6
  sentence "par plus the final coupon if ``S_{T_N} ≥ CB``": the put loss applies only when
  ``S_{T_N} < CB S_ref``.  The two coincide for a European knock-in with ``KI ≤ CB`` (the only
  European case accepted under ``"coupon_barrier"``: above ``CB`` the stated level would never be
  the effective one).

Knock-in statistics: ``ki_breach`` is the barrier event over the life — the level breached at a
monitoring date ``≤`` the termination date (the autocall date's own fixing included; for the
European type the only monitoring date is ``T_N``, so autocalled paths never breach) — and
``ki_hit = ki_breach · 1{no autocall}`` is the event that drives the put leg for both types;
the analytics report ``P(KI) = E[ki_hit]`` and ``P(breach) = E[ki_breach]`` separately.

Discounted payoff per unit notional, ``D`` the product's discount factors (SPEC §6.6)::

    Σ_i D(T_i) 1{AC at i} (1 + c_i^AC)  +  Σ_i D(T_i) coupon_i
                                        +  D(T_N) 1{no AC} [1 − 1{KI} g (K − S_{T_N})⁺ / K]

with ``g = 1`` (``"knock_in"``) or ``g = 1{S_{T_N} < CB S_ref}`` (``"coupon_barrier"``),
evaluated on the path's first autocall date.  ``decompose()``: one :class:`ConditionalDigital`
per date (the autocall event conditional on survival), the :class:`BondLeg` (par at ``T_N`` on
survival), the :class:`CouponLeg` (Phoenix / guaranteed coupons) and the :class:`KIPutLeg`.  For
the European knock-in the put leg is the interview-thread identity ``(K − S_T)⁺ 1{S_T < B} =
(B − S_T)⁺ + (K − B) 1{S_T < B}`` — a put struck at ``B`` plus a cash-or-nothing put of size
``K − B`` at ``B``, both conditional on survival, never "a put struck at the barrier" — and
``decompose()`` returns the two components; the American knock-in has no static decomposition
and :meth:`KIPutLeg.european_counterpart` is the leg it is reported against.  All legs sum to the
payoff path by path.  Checked by ``tests/test_autocall.py``.
"""

from __future__ import annotations

import hashlib
import weakref
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.products.base import Product, daily_schedule, shift_times
from volsto.products.gap import (
    GapReport,
    GapSpec,
    LevelFactors,
    month_grid,
    regress_at_level,
)
from volsto.products.vanilla import DigitalOption, EuropeanOption

if TYPE_CHECKING:
    from volsto.engine.grid import FixingIndex
    from volsto.engine.paths import PathSet
    from volsto.market.curves import DiscountCurve

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]

KI_TYPES = ("european", "american")
KI_MONITORINGS = ("discrete", "continuous")
FINAL_REDEMPTIONS = ("knock_in", "coupon_barrier")
STATISTICS = (
    "ac_index",
    "autocalled",
    "life",
    "ki_breach",
    "ki_hit",
    "coupons_paid",
    "redemption",
    "put_loss",
)
_TOL = 1e-9
_Eval = tuple[dict[str, FloatArray], dict[str, FloatArray], FloatArray]
_SmartMemo = tuple["weakref.ref[PathSet]", "weakref.ref[FixingIndex]", _Eval]


def _unique_times(ts: ArrayLike) -> FloatArray:
    a = np.sort(np.atleast_1d(np.asarray(ts, dtype=np.float64)).ravel())
    if a.size == 0:
        return a
    keep = np.concatenate(([True], np.diff(a) > _TOL))
    return np.asarray(a[keep], dtype=np.float64)


def _per_date(value: float | Sequence[float] | FloatArray, n: int, name: str) -> FloatArray:
    a = np.asarray(value, dtype=np.float64)
    if a.ndim == 0:
        return np.full(n, float(a))
    if a.shape != (n,):
        raise ValueError(f"{name} must be a float or a sequence with one entry per date")
    return np.asarray(a, dtype=np.float64)


class Autocall(Product):
    """Autocall / Phoenix note (module docstring for the term sheet and the payoff).

    Args (keyword-only, levels as fractions of ``spot_reference``):
        observation_times: ``T_1 < … < T_N`` (years, positive); the last one is the maturity.
        discount: discounting curve (payments at the observation dates).
        spot_reference: ``S_ref`` — the initial fixing all levels refer to (required, > 0).
        coupons: ``c`` or a per-date sequence (see the module docstring for both conventions).
        ki_level: knock-in level ``KI ∈ (0, 1]``.
        ki_type: ``"european"`` or ``"american"``.
        autocall_barriers: ``AC`` (float) or ``AC_i`` per date (step-down); ``inf`` disables.
        coupon_barrier: ``CB`` (Phoenix) or ``None`` (plain autocall).
        memory: Phoenix memory feature.
        ki_monitoring: ``"discrete"`` / ``"continuous"`` for the American knock-in (``None`` for
            the European one).
        ki_fixing_times: monitoring schedule of the discrete American knock-in, default
            ``daily_schedule(T_N)`` (252 per year, ``t = 0`` included); must end at ``T_N``.
        guaranteed_coupons: period coupons paid at every date while alive.
        non_call_periods: the first ``k`` observation dates carry no autocall event.
        final_redemption: Phoenix final-date convention, ``"knock_in"`` (coupon decision and
            knock-in redemption independent) or ``"coupon_barrier"`` (the put loss only when
            ``S_{T_N} < CB S_ref``, the literal SPEC §6.6 sentence); see the module docstring.
            Must be ``"knock_in"`` without a coupon barrier.
        notional: scales every cash flow.
    """

    def __init__(
        self,
        observation_times: ArrayLike,
        discount: DiscountCurve,
        *,
        spot_reference: float,
        coupons: float | Sequence[float],
        ki_level: float,
        ki_type: str,
        autocall_barriers: float | Sequence[float] = 1.0,
        coupon_barrier: float | None = None,
        memory: bool = False,
        ki_monitoring: str | None = None,
        ki_fixing_times: ArrayLike | None = None,
        guaranteed_coupons: bool = False,
        non_call_periods: int = 0,
        final_redemption: str = "knock_in",
        gap: GapSpec | None = None,
        notional: float = 1.0,
    ) -> None:
        super().__init__(discount, notional)
        obs = np.atleast_1d(np.asarray(observation_times, dtype=np.float64)).ravel()
        if (
            obs.size < 1
            or not np.all(np.isfinite(obs))
            or obs[0] <= 0
            or np.any(np.diff(obs) <= _TOL)
        ):
            raise ValueError("observation_times must be finite, positive and strictly increasing")
        n = int(obs.size)
        if not np.isfinite(spot_reference) or spot_reference <= 0:
            raise ValueError("spot_reference must be positive")
        ac = _per_date(autocall_barriers, n, "autocall_barriers")
        if np.any(np.isnan(ac)) or np.any(ac <= 0):
            raise ValueError("autocall_barriers must be positive (inf disables a date)")
        if coupon_barrier is not None and (not np.isfinite(coupon_barrier) or coupon_barrier <= 0):
            raise ValueError("coupon_barrier must be positive")
        if memory and coupon_barrier is None:
            raise ValueError("memory coupons need a coupon_barrier (Phoenix)")
        if guaranteed_coupons and (coupon_barrier is not None or memory):
            raise ValueError("guaranteed coupons exclude a coupon barrier and memory")
        has_coupon_leg = coupon_barrier is not None or guaranteed_coupons
        c_raw = np.asarray(coupons, dtype=np.float64)
        if c_raw.ndim == 0:
            c = float(c_raw)
            if not np.isfinite(c) or c < 0:
                raise ValueError("coupons must be finite and non-negative")
            schedule = np.full(n, c) if has_coupon_leg else c * np.arange(1, n + 1, dtype=float)
            coupons_arg: float | tuple[float, ...] = c
        else:
            schedule = _per_date(c_raw, n, "coupons")
            if not np.all(np.isfinite(schedule)) or np.any(schedule < 0):
                raise ValueError("coupons must be finite and non-negative")
            coupons_arg = tuple(float(x) for x in schedule)
        if not np.isfinite(ki_level) or not 0.0 < ki_level <= 1.0:
            raise ValueError("ki_level must lie in (0, 1] (a fraction of spot_reference)")
        if ki_type not in KI_TYPES:
            raise ValueError(f"ki_type must be one of {KI_TYPES}")
        kf: FloatArray | None = None
        if ki_type == "european":
            if ki_monitoring is not None or ki_fixing_times is not None:
                raise ValueError(
                    "a European knock-in is observed at maturity only: no ki_monitoring or "
                    "ki_fixing_times"
                )
        else:
            if ki_monitoring not in KI_MONITORINGS:
                raise ValueError(f"an American knock-in needs ki_monitoring in {KI_MONITORINGS}")
            if ki_monitoring == "discrete":
                kf = (
                    daily_schedule(float(obs[-1]))
                    if ki_fixing_times is None
                    else _unique_times(ki_fixing_times)
                )
                if (
                    kf.size == 0
                    or not np.all(np.isfinite(kf))
                    or kf[0] < 0
                    or abs(kf[-1] - obs[-1]) > _TOL
                ):
                    raise ValueError(
                        "ki_fixing_times must be finite, non-negative and end at the maturity"
                    )
            elif ki_fixing_times is not None:
                raise ValueError("continuous monitoring takes no ki_fixing_times")
        ncp = int(non_call_periods)
        if ncp != non_call_periods or not 0 <= ncp <= n:
            raise ValueError("non_call_periods must be an integer in [0, N]")
        if final_redemption not in FINAL_REDEMPTIONS:
            raise ValueError(f"final_redemption must be one of {FINAL_REDEMPTIONS}")
        if final_redemption == "coupon_barrier":
            if coupon_barrier is None:
                raise ValueError(
                    "final_redemption='coupon_barrier' gates the put loss by the coupon barrier: "
                    "it needs a Phoenix (coupon_barrier)"
                )
            if ki_type == "european" and ki_level > coupon_barrier:
                raise ValueError(
                    "final_redemption='coupon_barrier' with a European knock-in above the coupon "
                    "barrier: the stated ki_level would never be the effective one (state the "
                    "coupon barrier as ki_level)"
                )
        self.observation_times = obs
        self.n_dates = n
        self.spot_reference = float(spot_reference)
        self.autocall_barriers = ac
        self.coupon_schedule = schedule
        self.coupons = coupons_arg
        self.coupon_barrier = None if coupon_barrier is None else float(coupon_barrier)
        self.memory = bool(memory)
        # gap convention (owner addendum at the M6 review): None = no shift; GapSpec(mode="fixed")
        # is the signed fixed shift of the monitored levels; "smart" = the state-dependent signed
        # gap of volsto.products.gap (two-pass evaluation, see _evaluate)
        if gap is not None and not isinstance(gap, GapSpec):
            raise TypeError("gap must be a GapSpec or None")
        if (
            gap is not None
            and gap.smart
            and ki_type == "american"
            and ki_monitoring == "continuous"
        ):
            raise NotImplementedError(
                "the smart gap is implemented for discrete knock-in monitoring (and the "
                "European knock-in); use ki_monitoring='discrete'"
            )
        self.gap = gap
        self.ki_level = float(ki_level)
        self.ki_type = ki_type
        self.ki_monitoring = ki_monitoring
        self.ki_fixing_times = kf
        self.guaranteed_coupons = bool(guaranteed_coupons)
        self.non_call_periods = ncp
        self.final_redemption = final_redemption
        # the continuous knock-in needs the spot at every simulation step (Brownian bridge)
        self.requires_all_steps = ki_monitoring == "continuous"

    # -- term sheet ----------------------------------------------------------------------------

    def _kwargs(self) -> dict[str, Any]:
        """Constructor arguments (for :meth:`replace`, :meth:`aged`, :meth:`with_discount`)."""
        return {
            "observation_times": self.observation_times.copy(),
            "discount": self.discount,
            "spot_reference": self.spot_reference,
            "coupons": self.coupons,
            "ki_level": self.ki_level,
            "ki_type": self.ki_type,
            "autocall_barriers": self.autocall_barriers.copy(),
            "coupon_barrier": self.coupon_barrier,
            "memory": self.memory,
            "gap": self.gap,
            "ki_monitoring": self.ki_monitoring,
            "ki_fixing_times": (
                None if self.ki_fixing_times is None else self.ki_fixing_times.copy()
            ),
            "guaranteed_coupons": self.guaranteed_coupons,
            "non_call_periods": self.non_call_periods,
            "final_redemption": self.final_redemption,
            "notional": self.notional,
        }

    def replace(self, **changes: Any) -> Autocall:
        """The same note with some constructor arguments changed (e.g. ``ki_level``, or
        ``ki_type="european", ki_monitoring=None, ki_fixing_times=None`` for the European
        counterpart of an American knock-in)."""
        return Autocall(**{**self._kwargs(), **changes})

    @property
    def is_phoenix(self) -> bool:
        return self.coupon_barrier is not None

    @property
    def has_coupon_leg(self) -> bool:
        """Coupons paid on the observation dates independently of the autocall redemption."""
        return self.is_phoenix or self.guaranteed_coupons

    @property
    def maturity_date(self) -> float:
        return float(self.observation_times[-1])

    @property
    def ki_barrier(self) -> float:
        """``B = KI · S_ref`` in spot units."""
        return self.ki_level * self.spot_reference

    @property
    def autocall_levels(self) -> FloatArray:
        return np.asarray(self.autocall_barriers * self.spot_reference, dtype=np.float64)

    @property
    def fixing_times(self) -> FloatArray:
        if self.ki_fixing_times is None:
            return self.observation_times
        return _unique_times(np.concatenate([self.observation_times, self.ki_fixing_times]))

    @property
    def pay_times(self) -> FloatArray:
        return self.observation_times

    # -- evaluation ----------------------------------------------------------------------------

    def _ki_breach(
        self,
        paths: PathSet,
        idx: FixingIndex,
        ac_index: IntArray,
        life: FloatArray,
        s_t: FloatArray,
        ki_levels: LevelFactors | None = None,
    ) -> FloatArray:
        """``ki_breach``: the knock-in level breached at a monitoring date ``≤ life`` (0/1; a
        weight in ``[0, 1]`` for the continuous variant).  Discrete comparisons are strict and in
        spot space (module docstring); ``s_t`` is the terminal spot column of the caller."""
        b = self.ki_barrier
        if self.ki_type == "european":
            # the only monitoring date is T_N: an autocalled path never observes it
            b_eff = (
                b
                if ki_levels is None
                else b * ki_levels.at(self.maturity_date, paths.n_paths)[:, 0]
            )
            return ((s_t < b_eff) & (ac_index > self.n_dates)).astype(np.float64)
        if self.ki_monitoring == "discrete":
            assert self.ki_fixing_times is not None
            spots = paths.spot_at(idx.indices(self.ki_fixing_times))
            within = self.ki_fixing_times[None, :] <= life[:, None] + _TOL
            b_eff = (
                b if ki_levels is None else b * ki_levels.at(self.ki_fixing_times, paths.n_paths)
            )
            return ((spots < b_eff) & within).any(axis=1).astype(np.float64)
        if ki_levels is not None:
            raise NotImplementedError("gap shifts with continuous knock-in monitoring")
        from volsto.products.barrier import continuous_survival_weight

        ln_b = float(np.log(b))
        n = self.n_dates
        w = np.zeros(paths.n_paths)
        end = np.minimum(ac_index, n)
        for i, t_i in enumerate(self.observation_times, start=1):
            sel = end == i
            if sel.any():
                surv = np.asarray(
                    continuous_survival_weight(paths, idx, ln_b, "down", 0.0, float(t_i)),
                    dtype=np.float64,
                )
                w[sel] = 1.0 - surv[sel]
        return np.asarray(np.clip(w, 0.0, 1.0), dtype=np.float64)

    def _evaluate(
        self, paths: PathSet, idx: FixingIndex
    ) -> tuple[dict[str, FloatArray], dict[str, FloatArray], FloatArray]:
        """Per-path statistics, discounted legs (notional included) and the total payoff.

        Gap conventions (:mod:`volsto.products.gap`): a fixed gap shifts every monitored level
        by ``fixed_shift``; the smart gap runs two passes — the unshifted evaluation gives the
        remaining cash flows per path, per-date regressions of those cash flows on the state give
        the seller's liabilities just inside / outside each level (``ΔV``), the sign rule and the
        sizing function give a per-path, per-date effective level, and the second pass evaluates
        the note on those levels (same paths: common random numbers).  The per-path ``ΔV`` and
        shifts are returned in the statistics (``gap_dv_ac_i``, ``gap_shift_ac_i``,
        ``gap_dv_ki_j``, ``gap_shift_ki_j`` for the knock-in regression grid dates) and
        summarised by :meth:`gap_report`.
        """
        if self.gap is None:
            return self._evaluate_levels(paths, idx, None, None)[:3]
        if not self.gap.smart:
            f = 1.0 + self.gap.fixed_shift
            ac = np.broadcast_to(self.autocall_levels * f, (paths.n_paths, self.n_dates)).copy()
            return self._evaluate_levels(paths, idx, ac, LevelFactors(fixed=f))[:3]
        # the two-pass evaluation is memoised on the path set (weak reference): the legs and
        # statistics priced on the same paths reuse it instead of repeating the regressions
        memo: _SmartMemo | None = getattr(self, "_smart_memo", None)
        if memo is not None and memo[0]() is paths and memo[1]() is idx:
            return memo[2]
        result = self._evaluate_smart(paths, idx)
        self._smart_memo: _SmartMemo = (weakref.ref(paths), weakref.ref(idx), result)
        return result

    def _evaluate_levels(
        self,
        paths: PathSet,
        idx: FixingIndex,
        ac_levels: FloatArray | None,
        ki_levels: LevelFactors | None,
    ) -> tuple[dict[str, FloatArray], dict[str, FloatArray], FloatArray, dict[str, FloatArray]]:
        """The evaluation on given levels — ``ac_levels`` ``(n_paths, N)`` effective autocall
        levels (``None``: the term sheet's), ``ki_levels`` the knock-in level factors (``None``:
        the term sheet's) — returning the statistics, the discounted legs, the payoff and the
        per-date matrices ``(n_paths, N)`` the smart gap needs: ``cash`` (undiscounted cash flow
        at each observation date, notional 1) and ``coupons`` (the coupon part of it)."""
        obs = self.observation_times
        n = self.n_dates
        n_paths = paths.n_paths
        s_ref = self.spot_reference
        spots = paths.spot_at(idx.indices(obs))  # (n_paths, N)
        dates = np.arange(1, n + 1)
        levels = self.autocall_levels[None, :] if ac_levels is None else ac_levels
        cond = spots >= levels
        if self.non_call_periods:
            cond[:, : self.non_call_periods] = False
        autocalled = cond.any(axis=1)
        ac_index = np.where(autocalled, np.argmax(cond, axis=1) + 1, n + 1).astype(np.int64)
        alive = dates[None, :] <= ac_index[:, None]  # alive at date i (autocall date included)
        ac_event = dates[None, :] == ac_index[:, None]
        life = np.where(autocalled, obs[np.minimum(ac_index, n) - 1], obs[-1])
        c = self.coupon_schedule
        if self.has_coupon_leg:
            if self.guaranteed_coupons:
                pay = alive
            else:
                assert self.coupon_barrier is not None
                pay = alive & (spots >= self.coupon_barrier * s_ref)
            if self.memory:
                cum = np.concatenate(([0.0], np.cumsum(c)))  # cum[j] = Σ_{m ≤ j} c_m, 1-based
                pos = np.where(pay, dates[None, :], 0)
                last = np.maximum.accumulate(pos, axis=1)  # last payment date ≤ i (0: none)
                prev = np.concatenate(
                    [np.zeros((n_paths, 1), dtype=np.int64), last[:, :-1]], axis=1
                )
                amount = np.where(pay, cum[dates][None, :] - cum[prev], 0.0)
            else:
                amount = np.where(pay, c[None, :], 0.0)
            ac_coupon = np.zeros(n)
        else:
            amount = np.zeros((n_paths, n))
            ac_coupon = c
        s_t = spots[:, -1]
        survive = ~autocalled
        breach = self._ki_breach(paths, idx, ac_index, life, s_t, ki_levels)
        ki = breach * survive  # the knock-in event of the life (module docstring)
        if self.final_redemption == "coupon_barrier":
            assert self.coupon_barrier is not None
            # the literal SPEC §6.6 reading: par plus the coupon when S_TN ≥ CB, the put loss
            # applies only below the coupon barrier (a no-op for a European knock-in, KI ≤ CB)
            loss_ind = ki * (s_t < self.coupon_barrier * s_ref)
        else:
            loss_ind = ki
        put_loss = np.maximum(s_ref - s_t, 0.0) / s_ref  # (K − S_T)⁺ / K
        df = self.df(obs)
        df_t = float(df[-1])
        nt = self.notional
        legs: dict[str, FloatArray] = {}
        for i in range(n):
            legs[f"autocall_{i + 1}"] = nt * float(df[i]) * (1.0 + ac_coupon[i]) * ac_event[:, i]
        legs["bond"] = nt * df_t * survive.astype(np.float64)
        legs["coupon"] = nt * np.sum(amount * df[None, :], axis=1)
        legs["put"] = -nt * df_t * loss_ind * put_loss
        if self.ki_type == "european":
            # the identity's two components on the effective (possibly per-path) level, so that
            # put = put_vanilla + put_digital holds path by path under every gap convention
            b = self.ki_barrier
            b_eff = b if ki_levels is None else b * ki_levels.at(self.maturity_date, n_paths)[:, 0]
            scale = -nt / s_ref * df_t * survive
            legs["put_vanilla"] = np.asarray(scale * np.maximum(b_eff - s_t, 0.0), dtype=np.float64)
            legs["put_digital"] = np.asarray(
                scale * (s_ref - b_eff) * (s_t < b_eff), dtype=np.float64
            )
        total = legs["bond"] + legs["coupon"] + legs["put"]
        for i in range(n):
            total = total + legs[f"autocall_{i + 1}"]
        cash = amount + ac_event * (1.0 + ac_coupon)[None, :]
        cash[:, -1] += survive * (1.0 - loss_ind * put_loss)
        stats = {
            "ac_index": ac_index.astype(np.float64),
            "autocalled": autocalled.astype(np.float64),
            "life": np.asarray(life, dtype=np.float64),
            "ki_breach": breach,
            "ki_hit": np.asarray(ki, dtype=np.float64),
            "coupons_paid": np.sum(amount, axis=1)
            + np.where(autocalled, ac_coupon[np.minimum(ac_index, n) - 1], 0.0),
            "redemption": np.where(autocalled, 1.0, 1.0 - loss_ind * put_loss),
            "put_loss": np.asarray(loss_ind * put_loss, dtype=np.float64),
        }
        extra = {
            "cash": np.asarray(cash, dtype=np.float64),
            "coupons": np.asarray(amount, dtype=np.float64),
        }
        return stats, legs, np.asarray(total, dtype=np.float64), extra

    def _evaluate_smart(
        self, paths: PathSet, idx: FixingIndex
    ) -> tuple[dict[str, FloatArray], dict[str, FloatArray], FloatArray]:
        """Two-pass smart gap (:mod:`volsto.products.gap`).  Pass 1 evaluates the note on the
        term-sheet levels and keeps the undiscounted cash flow of every path at every observation
        date.  **Autocall barrier** ``i``: for the paths alive at ``T_i`` the seller's liability
        when called is analytic, ``1 + c_i^AC`` (plain autocall) or ``1 +`` the date's coupon at
        the level (Phoenix: ``c_i`` plus the memory coupons when ``AC_i ≥ CB``, else nothing);
        the continuing liability is the §7.11 regression, over the paths continuing at ``T_i``,
        of their cash flows from ``T_i`` on (date-``T_i`` money) on ``(ln S_i − ln AC_i, X_i,
        knock-in status, memory state)``, evaluated at the level in each alive path's own state;
        ``ΔV = L_called − L_continuing``.  **Knock-in barrier**: on the regression grid dates of
        :func:`~volsto.products.gap.month_grid` (the single date ``T_N`` for the European type)
        two regressions of the cash flows from the date on — over the paths already knocked in
        and over the others — give, at the level, ``ΔV = L_knocked − L_alive`` for every path not
        yet settled; the factor holds until the next grid date (piecewise constant in time, per
        path).  Sign and size per path: :func:`~volsto.products.gap.conservative_shift` with the
        spec's sizing function; a date whose regression has fewer than ``min_paths`` paths on a
        side keeps the fixed shift (``n_fit = 0`` in the report).  Pass 2 evaluates the note on
        the effective levels with the same paths (common random numbers).  Each path's own
        variance / knock-in / memory state enters its ``ΔV``: the paths that reach an autocall
        level after a large move carry the variance state of that move, which is how the sign at
        one barrier differs between states (module docstring of :mod:`volsto.products.gap`)."""
        assert self.gap is not None and self.gap.smart
        gap = self.gap
        stats0, _legs0, _total0, extra = self._evaluate_levels(paths, idx, None, None)
        cash = extra["cash"]
        obs = self.observation_times
        n = self.n_dates
        n_paths = paths.n_paths
        f_fixed = 1.0 + gap.fixed_shift
        cols = idx.indices(obs)
        ln_spots = paths.log_spot_at(cols)  # (n_paths, N)
        df = np.asarray(self.df(obs), dtype=np.float64)
        ac_index = stats0["ac_index"].astype(np.int64)
        life = stats0["life"]
        ki_state = self._ki_state_by_date(paths, idx)  # breached at a fixing ≤ T_i (0/1)
        memory_state = self._memory_state_by_date(extra["coupons"])  # in memory after T_i
        ac_levels = np.broadcast_to(self.autocall_levels * f_fixed, (n_paths, n)).copy()
        gap_stats: dict[str, FloatArray] = {}
        report = GapReport()
        if "autocall" in gap.apply_to:
            for i in range(n):
                level = float(self.autocall_levels[i])
                if i < self.non_call_periods or not np.isfinite(level):
                    continue
                alive = ac_index > i  # not called at an earlier date (1-based dates ≤ i)
                cont = alive & (ac_index != i + 1)  # continuing at T_i
                # the continuing liability: cash flows from T_i on, in T_i money
                y = (cash[:, i:] * (df[None, i:] / df[i])).sum(axis=1)
                feats = np.column_stack(
                    [
                        ln_spots[:, i] - np.log(level),
                        paths.factors_at(int(cols[i])),
                        ki_state[:, i],
                        memory_state[:, i],
                    ]
                )
                v_cont, n_fit, se_cont = regress_at_level(gap, y, feats, cont, alive)
                called = 1.0 + self._called_coupon_at_level(i, memory_state)
                dv = np.full(n_paths, np.nan)
                shift = np.full(n_paths, gap.fixed_shift)
                if n_fit > 0:
                    dv_alive = called[alive] - v_cont
                    dv[alive] = dv_alive
                    shift[alive] = gap.shift(dv_alive, "up", se_cont)
                    ac_levels[:, i] = level * (1.0 + shift)
                gap_stats[f"gap_dv_ac_{i + 1}"] = dv
                gap_stats[f"gap_shift_ac_{i + 1}"] = shift
                near = alive & (np.abs(feats[:, 0]) <= gap.report_band)
                report.add(f"autocall_{i + 1}", float(obs[i]), dv, shift, n_fit, near=near)
        ki_levels = LevelFactors(fixed=f_fixed)
        if "ki" in gap.apply_to:
            b = self.ki_barrier
            if self.ki_type == "european":
                grid = np.array([obs[-1]])
            else:
                assert self.ki_fixing_times is not None
                grid = month_grid(self.ki_fixing_times, gap.ki_grid_months)
            gcols = idx.indices(grid)
            ln_grid = paths.log_spot_at(gcols)
            fac = np.full((n_paths, grid.size), f_fixed)
            for j, t in enumerate(grid):
                # the liabilities from the grid date on (a settlement at the date included: at
                # T_N the knock-in decides the redemption), in date money
                after = obs >= t - _TOL
                dfg = float(np.asarray(self.df(np.array([t])), dtype=np.float64)[0])
                y = (cash[:, after] * (df[None, after] / dfg)).sum(axis=1)
                alive = life >= t - _TOL  # not settled before the grid date
                breached = self._breached_by(paths, idx, float(t))
                feats = np.column_stack(
                    [
                        ln_grid[:, j] - np.log(b),
                        paths.factors_at(int(gcols[j])),
                        self._memory_at(memory_state, obs, float(t)),
                    ]
                )
                v_in, n_in, se_in = regress_at_level(gap, y, feats, alive & breached, alive)
                v_out, n_out, se_out = regress_at_level(gap, y, feats, alive & ~breached, alive)
                n_fit = min(n_in, n_out)
                dv = np.full(n_paths, np.nan)
                shift = np.full(n_paths, gap.fixed_shift)
                if n_fit > 0:
                    dv_alive = v_in - v_out  # knocked-in minus alive liability
                    dv[alive] = dv_alive
                    shift[alive] = gap.shift(dv_alive, "down", np.hypot(se_in, se_out))
                    fac[:, j] = 1.0 + shift
                gap_stats[f"gap_dv_ki_{j + 1}"] = dv
                gap_stats[f"gap_shift_ki_{j + 1}"] = shift
                near = alive & (np.abs(feats[:, 0]) <= gap.report_band)
                report.add("ki", float(t), dv, shift, n_fit, near=near)
            ki_levels = LevelFactors(fixed=f_fixed, grid=grid, factors=fac)
        stats, legs, total, _extra = self._evaluate_levels(paths, idx, ac_levels, ki_levels)
        stats.update(gap_stats)
        self._last_gap_report = report
        return stats, legs, total

    def gap_report(self, paths: PathSet, idx: FixingIndex) -> GapReport:
        """The per-barrier, per-date smart-gap summary on these paths (:class:`GapReport`;
        evaluates the note)."""
        if self.gap is None or not self.gap.smart:
            raise ValueError("gap_report needs a smart gap")
        self._evaluate_smart(paths, idx)
        return self._last_gap_report

    def _called_coupon_at_level(self, i: int, memory_state: FloatArray) -> FloatArray:
        """Coupon paid with the autocall redemption at ``T_i`` when the spot sits at the level:
        the growing / scheduled coupon (plain autocall, guaranteed coupons); for a Phoenix the
        period coupon plus the coupons in memory before the date when ``AC_i ≥ CB``, else 0."""
        n_paths = memory_state.shape[0]
        c_i = float(self.coupon_schedule[i])
        if not self.is_phoenix:
            return np.full(n_paths, c_i)
        assert self.coupon_barrier is not None
        if self.autocall_barriers[i] < self.coupon_barrier - _TOL:
            return np.zeros(n_paths)
        before = memory_state[:, i - 1] if i > 0 else np.zeros(n_paths)
        return np.asarray(c_i + before, dtype=np.float64)

    def _ki_state_by_date(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        """Knock-in level breached at a monitoring date ``≤ T_i`` (0/1 per path and date; the
        European type has no status before its single date)."""
        obs = self.observation_times
        n = self.n_dates
        out = np.zeros((paths.n_paths, n))
        if self.ki_type == "european":
            return out
        assert self.ki_fixing_times is not None
        hit = paths.spot_at(idx.indices(self.ki_fixing_times)) < self.ki_barrier
        for i in range(n):
            upto = self.ki_fixing_times <= obs[i] + _TOL
            if upto.any():
                out[:, i] = hit[:, upto].any(axis=1)
        return out

    def _breached_by(self, paths: PathSet, idx: FixingIndex, t: float) -> BoolArray:
        """Knock-in status at ``t``: a fixing ``≤ t`` below the level (European: ``S_T < B``
        when ``t`` is the maturity)."""
        if self.ki_type == "european":
            if t < self.maturity_date - _TOL:
                return np.zeros(paths.n_paths, dtype=bool)
            return np.asarray(paths.spot_at(idx[self.maturity_date]) < self.ki_barrier, dtype=bool)
        assert self.ki_fixing_times is not None
        upto = self.ki_fixing_times <= t + _TOL
        if not upto.any():
            return np.zeros(paths.n_paths, dtype=bool)
        spots = paths.spot_at(idx.indices(self.ki_fixing_times[upto]))
        return np.asarray((spots < self.ki_barrier).any(axis=1), dtype=bool)

    def _memory_state_by_date(self, coupons: FloatArray) -> FloatArray:
        """Coupons missed and still recoverable just after each observation date (Phoenix with
        memory; zeros otherwise): the scheduled coupons to date less the coupons paid to date
        (``coupons``: the per-date coupon amounts of the evaluation, ``(n_paths, N)``)."""
        n_paths, n = coupons.shape
        if not (self.is_phoenix and self.memory):
            return np.zeros((n_paths, n))
        cum_sched = np.cumsum(self.coupon_schedule)
        state = cum_sched[None, :] - np.cumsum(coupons, axis=1)
        return np.asarray(np.maximum(state, 0.0), dtype=np.float64)

    @staticmethod
    def _memory_at(memory_state: FloatArray, obs: FloatArray, t: float) -> FloatArray:
        """Memory state at ``t``: the state after the last observation date ``≤ t``."""
        before = np.flatnonzero(obs <= t + _TOL)
        if before.size == 0:
            return np.zeros(memory_state.shape[0])
        return np.asarray(memory_state[:, before[-1]], dtype=np.float64)

    def statistics(self, paths: PathSet, idx: FixingIndex) -> dict[str, FloatArray]:
        """Per-path undiscounted statistics: ``ac_index`` (first autocall date, 1-based, ``N + 1``
        when none), ``autocalled``, ``life`` (years), ``ki_breach`` (the level breached at a
        monitoring date ``≤ life``, 0/1; a weight in ``[0, 1]`` for the continuous knock-in),
        ``ki_hit = ki_breach · 1{no AC}`` (the event driving the put leg, both types),
        ``coupons_paid`` (total, fraction of notional), ``redemption`` (fraction of notional) and
        ``put_loss = 1{no AC} 1{KI} g (K − S_T)⁺ / K`` (``g`` the ``final_redemption`` gate)."""
        return self._evaluate(paths, idx)[0]

    def coupon_amounts(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        """Per-path, per-observation-date coupon amounts (fraction of notional, undiscounted,
        the memory catch-up included), shape ``(n_paths, N)`` — the ``extra["coupons"]`` of
        :meth:`_evaluate_levels` that :meth:`_evaluate_smart` derives its memory state from,
        which :meth:`statistics` does not expose.  The hedge state of a Phoenix
        (:func:`volsto.hedging.state.hedge_state`) reads its memory feature from it.  A fixed
        gap shifts the levels as :meth:`_evaluate` does; the smart gap's memory state is the
        unshifted pass-1 one.  Test: ``tests/test_hedging.py::test_hedge_basis_and_state_features``.
        """
        if self.gap is not None and not self.gap.smart:
            f = 1.0 + self.gap.fixed_shift
            ac = np.broadcast_to(self.autocall_levels * f, (paths.n_paths, self.n_dates)).copy()
            return self._evaluate_levels(paths, idx, ac, LevelFactors(fixed=f))[3]["coupons"]
        return self._evaluate_levels(paths, idx, None, None)[3]["coupons"]

    def leg_payoffs(self, paths: PathSet, idx: FixingIndex) -> dict[str, FloatArray]:
        """Discounted per-path leg cash flows (notional included): ``autocall_i``, ``bond``,
        ``coupon``, ``put`` and, for the European knock-in, ``put_vanilla`` / ``put_digital``
        (``put = put_vanilla + put_digital``).  The payoff is the sum of the first four kinds."""
        return self._evaluate(paths, idx)[1]

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        return self._evaluate(paths, idx)[2]

    # -- decomposition -------------------------------------------------------------------------

    def autocall_leg(self, date: int) -> ConditionalDigital:
        return ConditionalDigital(self, date)

    def bond_leg(self) -> BondLeg:
        return BondLeg(self)

    def coupon_leg(self) -> CouponLeg:
        return CouponLeg(self)

    def ki_put_leg(self) -> KIPutLeg:
        return KIPutLeg(self)

    def statistic(self, name: str, date: int | None = None) -> AutocallStatistic:
        """A product whose (undiscounted) payoff is the statistic ``name`` (or ``1{AC at date}``
        for ``name = "autocall_at"``)."""
        return AutocallStatistic(self, name, date)

    def decompose(self) -> list[Product]:
        legs: list[Product] = [ConditionalDigital(self, i) for i in range(1, self.n_dates + 1)]
        legs.append(BondLeg(self))
        if self.has_coupon_leg:
            legs.append(CouponLeg(self))
        if self.ki_type == "european":
            legs += [KIPutLeg(self, "vanilla"), KIPutLeg(self, "digital")]
        else:
            legs.append(KIPutLeg(self))
        return legs

    # -- state changes -------------------------------------------------------------------------

    def with_discount(self, discount: DiscountCurve) -> Autocall:
        return self.replace(discount=discount)

    def aged(self, dt: float) -> Autocall:
        """The note seen ``dt`` years later with the state held (SPEC v2 §7.3): observation dates
        move earlier by ``dt`` (one inside ``(0, dt]`` raises, the note would have fixed); knock-in
        monitoring dates inside ``(0, dt]`` were observed at the held spot and drop out (``t = 0``
        stays, so a spot already below the barrier still knocks in), the rest move earlier."""
        kf = None
        if self.ki_fixing_times is not None:
            t = self.ki_fixing_times
            kf = shift_times(t[(t <= 0.0) | (t > dt + 1e-12)], dt)
        return self.replace(
            observation_times=shift_times(self.observation_times, dt), ki_fixing_times=kf
        )

    def __repr__(self) -> str:
        dates = ", ".join(f"{t:g}" for t in self.observation_times)
        ac = ", ".join(f"{a:.4g}" for a in self.autocall_barriers)
        if isinstance(self.coupons, tuple):
            cpn = "[" + ", ".join(f"{c:.4g}" for c in self.coupons) + "]"
        else:
            cpn = f"{self.coupons:.4g}" + ("" if self.has_coupon_leg else " x i (growing)")
        if self.is_phoenix:
            kind = f"Phoenix: coupon {cpn} if S >= {self.coupon_barrier:.4g}"
            kind += " (memory)" if self.memory else " (no memory)"
            kind += (
                ", final put loss only below the coupon barrier"
                if self.final_redemption == "coupon_barrier"
                else ", final coupon and knock-in redemption independent"
            )
        elif self.guaranteed_coupons:
            kind = f"Autocall with guaranteed coupon {cpn}"
        else:
            kind = f"Autocall: coupon {cpn} paid at the autocall"
        if self.ki_type == "european":
            ki = f"European KI {self.ki_level:.4g} at maturity"
        elif self.ki_monitoring == "discrete":
            assert self.ki_fixing_times is not None
            kf = self.ki_fixing_times
            ki = f"American KI {self.ki_level:.4g} on {kf.size} discrete dates"
            if kf.size > 1:
                gaps = np.diff(kf)
                ki += f" from {kf[0]:g}y to {kf[-1]:g}y (step {np.median(gaps):.4g}y"
                if np.ptp(gaps) > _TOL:  # non-uniform schedule: name its content
                    digest = hashlib.sha1(np.round(kf, 12).tobytes()).hexdigest()[:8]
                    ki += f", non-uniform, sha1 {digest}"
                ki += ")"
            else:
                ki += f" at {kf[0]:g}y"
        else:
            ki = f"American KI {self.ki_level:.4g} continuous (Brownian bridge)"
        ncp = f", first {self.non_call_periods} dates non-call" if self.non_call_periods else ""
        text = (
            f"{kind}; observation dates [{dates}]y, autocall barriers [{ac}]{ncp}; {ki}; "
            f"put strike 100% geared 1:1; levels x spot_reference {self.spot_reference:g}; "
            f"notional {self.notional:g}"
        )
        if self.gap is not None:
            text += f"; {self.gap!r}"
        return text


def Phoenix(
    observation_times: ArrayLike,
    discount: DiscountCurve,
    *,
    spot_reference: float,
    coupon: float | Sequence[float],
    coupon_barrier: float,
    memory: bool,
    ki_level: float,
    ki_type: str,
    ki_monitoring: str | None = None,
    ki_fixing_times: ArrayLike | None = None,
    autocall_barriers: float | Sequence[float] = 1.0,
    non_call_periods: int = 0,
    final_redemption: str = "knock_in",
    notional: float = 1.0,
) -> Autocall:
    """Phoenix convenience constructor: the period ``coupon`` is paid when ``S ≥ coupon_barrier``
    (with or without ``memory``), autocall at ``autocall_barriers``, the final date settled per
    ``final_redemption`` (module docstring)."""
    return Autocall(
        observation_times,
        discount,
        spot_reference=spot_reference,
        coupons=coupon,
        ki_level=ki_level,
        ki_type=ki_type,
        autocall_barriers=autocall_barriers,
        coupon_barrier=coupon_barrier,
        memory=memory,
        ki_monitoring=ki_monitoring,
        ki_fixing_times=ki_fixing_times,
        non_call_periods=non_call_periods,
        final_redemption=final_redemption,
        notional=notional,
    )


# ---------------------------------------------------------------------------------------------
# legs
# ---------------------------------------------------------------------------------------------


class _AutocallLeg(Product):
    """One discounted leg of an autocall (``key`` into :meth:`Autocall.leg_payoffs`)."""

    def __init__(self, parent: Autocall, key: str, pay_times: FloatArray) -> None:
        super().__init__(parent.discount, 1.0)
        self.parent = parent
        self.key = key
        self._pay_times = np.asarray(pay_times, dtype=np.float64)
        self.requires_all_steps = parent.requires_all_steps

    @property
    def fixing_times(self) -> FloatArray:
        return self.parent.fixing_times

    @property
    def pay_times(self) -> FloatArray:
        return self._pay_times

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        return self.parent.leg_payoffs(paths, idx)[self.key]

    def _clone(self, parent: Autocall) -> _AutocallLeg:
        raise NotImplementedError

    def with_discount(self, discount: DiscountCurve) -> Product:
        return self._clone(self.parent.with_discount(discount))

    def aged(self, dt: float) -> Product:
        return self._clone(self.parent.aged(dt))


class ConditionalDigital(_AutocallLeg):
    """Autocall event at date ``i``: pays ``1 + c_i^AC`` (plain autocall) or ``1`` (Phoenix /
    guaranteed) at ``T_i`` if ``S_{T_i} ≥ AC_i S_ref`` and no earlier date autocalled — a digital
    conditional on survival (SPEC §6.6 "strip of digitals conditional on survival")."""

    def __init__(self, parent: Autocall, date: int) -> None:
        if not 1 <= date <= parent.n_dates:
            raise ValueError("date must lie in 1..N")
        self.date = int(date)
        super().__init__(parent, f"autocall_{date}", np.array([parent.observation_times[date - 1]]))

    @property
    def payout(self) -> float:
        """Amount paid on the event (per unit notional)."""
        c = 0.0 if self.parent.has_coupon_leg else float(self.parent.coupon_schedule[self.date - 1])
        return 1.0 + c

    def _clone(self, parent: Autocall) -> ConditionalDigital:
        return ConditionalDigital(parent, self.date)

    def __repr__(self) -> str:
        p = self.parent
        t_i = p.observation_times[self.date - 1]
        return (
            f"Autocall event leg date {self.date}/{p.n_dates} at {t_i:g}y: pays {self.payout:.4g} "
            f"if S >= {p.autocall_barriers[self.date - 1]:.4g} x S_ref and no earlier autocall, "
            f"of [{p!r}]"
        )


class BondLeg(_AutocallLeg):
    """Par at ``T_N`` on the paths that never autocalled (the zero-coupon bond leg)."""

    def __init__(self, parent: Autocall) -> None:
        super().__init__(parent, "bond", np.array([parent.maturity_date]))

    def _clone(self, parent: Autocall) -> BondLeg:
        return BondLeg(parent)

    def __repr__(self) -> str:
        return (
            f"Bond leg: par at {self.parent.maturity_date:g}y if not autocalled, of "
            f"[{self.parent!r}]"
        )


class CouponLeg(_AutocallLeg):
    """Phoenix (barrier, optional memory) or guaranteed coupons paid on the observation dates
    while alive; identically zero for the plain autocall (its coupons ride the autocall legs)."""

    def __init__(self, parent: Autocall) -> None:
        super().__init__(parent, "coupon", parent.observation_times)

    def _clone(self, parent: Autocall) -> CouponLeg:
        return CouponLeg(parent)

    def __repr__(self) -> str:
        return f"Coupon leg of [{self.parent!r}]"


class KIPutLeg(_AutocallLeg):
    """The knock-in put the investor is short: ``−1{no AC} 1{KI} g (K − S_{T_N})⁺ / K`` at
    ``T_N`` (``g`` the ``final_redemption`` gate, ``1`` for ``"knock_in"``).

    ``component = "full"`` is the leg; for the European knock-in ``"vanilla"`` is
    ``−1{no AC} (B − S_T)⁺ / K`` and ``"digital"`` is ``−1{no AC} (K − B) 1{S_T < B} / K``, the two
    terms of ``(K − S_T)⁺ 1{S_T < B} = (B − S_T)⁺ + (K − B) 1{S_T < B}``, exact path by path
    including at the barrier (every comparison in spot space, module docstring; ``decompose()``
    of the full leg; ``tests/test_autocall.py::test_black_scholes_closed_forms`` checks the
    vanilla term against ``bs_price`` and the digital against the cash-or-nothing closed form,
    ``test_continuous_knock_in`` the continuous American leg against the Reiner–Rubinstein
    down-and-in put of :func:`volsto.market.barrier_bs.bs_barrier_price`).
    :meth:`unconditional_components` gives the plain :class:`EuropeanOption` and
    :class:`DigitalOption` the two terms reduce to when nothing autocalls.
    """

    COMPONENTS = ("full", "vanilla", "digital")

    def __init__(self, parent: Autocall, component: str = "full") -> None:
        if component not in self.COMPONENTS:
            raise ValueError(f"component must be one of {self.COMPONENTS}")
        if component != "full" and parent.ki_type != "european":
            raise ValueError("only the European knock-in put splits into vanilla + digital")
        self.component = component
        key = {"full": "put", "vanilla": "put_vanilla", "digital": "put_digital"}[component]
        super().__init__(parent, key, np.array([parent.maturity_date]))

    def _clone(self, parent: Autocall) -> KIPutLeg:
        return KIPutLeg(parent, self.component)

    def decompose(self) -> list[Product] | None:
        if self.component != "full" or self.parent.ki_type != "european":
            return None
        return [KIPutLeg(self.parent, "vanilla"), KIPutLeg(self.parent, "digital")]

    def unconditional_components(self) -> tuple[EuropeanOption, DigitalOption]:
        """``put(B)`` and the ``(K − B)`` cash-or-nothing put at ``B`` (European knock-in), scaled
        by ``−notional / K``: the leg's components when no date autocalls."""
        p = self.parent
        if p.ki_type != "european":
            raise ValueError("the American knock-in put has no static decomposition")
        b, t, scale = p.ki_barrier, p.maturity_date, -p.notional / p.spot_reference
        return (
            EuropeanOption(b, t, -1, p.discount, scale),
            DigitalOption(b, t, -1, p.discount, p.spot_reference - b, scale),
        )

    def european_counterpart(self) -> KIPutLeg:
        """The same leg with the knock-in observed at maturity only (what the American knock-in
        put is reported against, SPEC §6.6)."""
        if self.parent.ki_type == "european":
            return self
        return KIPutLeg(
            self.parent.replace(ki_type="european", ki_monitoring=None, ki_fixing_times=None),
            self.component,
        )

    def __repr__(self) -> str:
        what = {
            "full": "knock-in put",
            "vanilla": "put struck at the KI level",
            "digital": "cash-or-nothing put at the KI level of size K - B",
        }[self.component]
        return f"KI put leg ({what}, short, geared 1:1) of [{self.parent!r}]"


class AutocallStatistic(Product):
    """Undiscounted per-path statistic of an autocall priced like a product (its "price" is the
    expectation): ``name`` in :data:`STATISTICS`, or ``"autocall_at"`` with a ``date`` for
    ``1{first autocall at date}`` (used by :mod:`volsto.analytics.autocall`)."""

    def __init__(self, parent: Autocall, name: str, date: int | None = None) -> None:
        super().__init__(parent.discount, 1.0)
        if name == "autocall_at":
            if date is None or not 1 <= date <= parent.n_dates + 1:
                raise ValueError("'autocall_at' needs a date in 1..N+1 (N+1: never)")
        elif name not in STATISTICS or date is not None:
            raise ValueError(f"name must be 'autocall_at' (with a date) or one of {STATISTICS}")
        self.parent = parent
        self.name = name
        self.date = None if date is None else int(date)
        self.requires_all_steps = parent.requires_all_steps

    @property
    def fixing_times(self) -> FloatArray:
        return self.parent.fixing_times

    @property
    def pay_times(self) -> FloatArray:
        return np.array([self.parent.maturity_date])

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        st = self.parent.statistics(paths, idx)
        if self.name == "autocall_at":
            return np.asarray(st["ac_index"] == self.date, dtype=np.float64)
        return st[self.name]

    def with_discount(self, discount: DiscountCurve) -> Product:
        return AutocallStatistic(self.parent.with_discount(discount), self.name, self.date)

    def aged(self, dt: float) -> Product:
        return AutocallStatistic(self.parent.aged(dt), self.name, self.date)

    def __repr__(self) -> str:
        tag = self.name if self.date is None else f"{self.name}[{self.date}]"
        return f"Statistic '{tag}' of [{self.parent!r}]"
