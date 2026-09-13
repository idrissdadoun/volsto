"""Risk for digital structures (SPEC v2 §6.7, extending §7.10; M6 Part 3): autocall spot profiles
at each observation date, the knock-in put's barrier risk, vega-T / skew-T of the structure and of
every leg, expected-life sensitivities and the likelihood-ratio cross-check on the digital legs.

**Spot profiles per observation date** (:func:`autocall_spot_profiles`).  The note is priced "as
seen one business day before ``T_i`` with the state held": today's surface and model, the path
held flat at today's spot up to ``T_i − dt`` (``dt = 1/252``) and the spot then shifted along a
grid straddling the autocall barrier of the date.  The held history is a market state, not a
path event: the profile is *conditional on the note being alive at* ``T_i − dt`` (with ``AC =
100%`` and ``S_ref = S₀`` a literally flat path would have autocalled at ``T_1``), so the earlier
observation dates are dropped; the earlier knock-in monitoring dates were observed at the held
spot, so an American knock-in is *alive* unless the held spot lies below the barrier (strict
``S < B`` for discrete monitoring, ``S ≤ B`` for the Brownian bridge) — in which case the residual
note is the knocked-in note, which is exactly the European knock-in at ``100%`` (``(K − S_T)⁺
1{S_T < K} = (K − S_T)⁺``; at the coupon barrier under ``final_redemption = "coupon_barrier"``);
a memory Phoenix whose held spot lies below the coupon barrier carries the missed coupons of the
dropped dates into its first residual coupon (the memory payment is ``Σ_{m ≤ j} c_m`` from the
start, so the carry is exact).  :func:`residual_note` builds the residual note with
:meth:`~volsto.products.autocall.Autocall.aged` for the first date (no history adjustment) and
explicitly otherwise — ``aged`` raises for any observation date inside the roll, the growing
coupon must be re-indexed (``c_j = j·c`` refers to the original date index), the non-call periods
reduced and the knock-in schedule shifted with the ``aged`` convention (monitoring dates in ``(0,
T_i − dt]`` drop out, ``t = 0`` stays, the rest move earlier); the construction is recorded.  Each
date's frame is the ``"model"``-regime :func:`~volsto.risk.profiles.spot_profile` (price, central
delta / gamma, sticky-leverage vega) on a grid of relative moves around the *monitored* autocall
level (default ±10% in 1% steps, refined to 0.25% within ±2%), where the digital's delta / gamma
spike shows.  The central log-spot stencil ``±size`` of the delta / gamma is itself a smoothing of
the one-day digital: it averages the jump over the band ``S e^{∓size}`` of width ``≈ 2·size·S``
(``attrs["stencil_width"]``), comparable to ``σ√dt ≈ 1.3%`` at 20% vol — at the default 1% the
peak delta reads about 10% below the digital's own delta (Black–Scholes: 0.0172 against 0.0190 at
the barrier), at 0.25% within 0.7% (the test), at twice the Monte Carlo noise; ``size`` is the
report parameter for that trade-off.

**Smoothing and barrier shift** (report parameters, recorded in every frame's ``attrs``).  The
:class:`~volsto.products.autocall.Autocall` class has no smoothing argument, and averaging two
notes with ``AC_i ∓ w/2`` is a two-step staircase, not a call spread.  The exact call spread of
the note's jump at the profile's date is :class:`SmoothedAutocall`: on the residual note the
autocall event of its first date splits the payoff path by path, ``P = 1{S_1 ≥ B} P_call + 1{S_1 <
B} P_cont`` with ``P_call`` the note whose first barrier is certain (``AC_1 → 0⁺``) and ``P_cont``
the note whose first event is disabled (``AC_1 = ∞``); the smoothed note replaces the indicator by
``θ(S) = clip((S − B + w/2)/w, 0, 1) = (1/w) ∫_{B−w/2}^{B+w/2} 1{S ≥ b} db`` — the uniform average
of the note over the barrier band, i.e. the unit-height spread of calls struck at ``B ∓ w/2``
(:class:`volsto.products.barrier.Digital`'s convention), applied to the *whole jump* of the note at
that date (redemption plus coupon versus the continuation value).  Only the autocall event at the
profile's date is smoothed; the later digitals and the knock-in digital are exact.  ``smoothing``
is the width ``w`` as a fraction of ``S_ref`` (the unit of every level).  ``barrier_shift`` moves
monitored levels only, by the signed fraction ``(1 + s)`` (:func:`shifted_levels`): the knock-in
level always, and the autocall barrier of the profile's date (``barrier_shift_scope = "date"``,
the integrator's convention) or every autocall barrier of the residual note (``"all"``); the put
strike ``K = S_ref``, the coupon barrier and the coupons never move (the desk convention of §6.5;
no direction is built in).  The profile grid is centred on the monitored level;
``spot_over_barrier`` is relative to the contractual level and ``spot_over_monitored`` to the
monitored one (the same two columns in :func:`ki_barrier_profile`).

**Knock-in put barrier risk** (§7.10, :func:`ki_put_barrier_risk`, :func:`ki_barrier_profile`).
``∂price/∂B`` by a central bump of the knock-in level (``±size`` relative, default 0.5%) through
:meth:`~volsto.risk.engine.RiskEngine.paired` on ``product.replace(ki_level=…)``, per unit spot and
per 1% of the barrier — of the put leg (the only leg that depends on ``B``, so it is the note's
``∂price/∂B`` too), of its vanilla / digital components for the European knock-in and of
``P(KI)``.  Black–Scholes (European knock-in, never autocalling; the interview-thread identity
``(K − S_T)⁺ 1{S_T < B} = (B − S_T)⁺ + (K − B) 1{S_T < B}``):

    d/dB [put(B) + (K − B) DF N(−d₂(B))] = (K − B) DF φ(d₂(B)) / (B σ √T) = (K − B) DF f_{S_T}(B),

the ``DF N(−d₂)`` terms of the two components cancelling (``∂put/∂K = DF N(−d₂)``); the put leg is
``−notional/K`` times that and ``∂P(KI)/∂B = f_{S_T}(B)``.  The profile is the model-regime delta /
gamma of the whole note within ±5% of the knock-in barrier in 0.5% steps
(:func:`~volsto.risk.product_risk.barrier_profile`, 21 rows); for the never-autocalling note the
price is ``DF − [put(S; B) + (K − B) DF N(−d₂(S; B))]/K``, so every row's delta is checked against
``[e^{−qT} N(−d₁) + (K − B) DF φ(d₂)/(S σ √T)]/K`` and its gamma against the derivative of that.

**Structure vega and skew** (:func:`structure_vega_skew`): :func:`~volsto.risk.ladders.vega_T`
waves and projections and the :func:`~volsto.risk.ladders.skew_T` entries for the note and for
every leg of ``decompose()`` (each leg is a product priced through the engine on the same grid and
draws as the note, so the legs' sensitivities add up to the note's exactly), plus the parallel vega
(wave ``n``) and the global rotation.  The skew bump is the fixed-ATM rotation of §7.6, ``σ → σ +
s κ(k) tent_i(T)``, ``κ(k) = k_cap tanh(k/k_cap)``, ``s`` sized so the 90/110 skew at the pillar
rises by 1 vp (put wing up, call wing down, ATM fixed).  Signs (autocalls are short vol, short
skew, long forward, SPEC §6.7): the parallel vega of the note is negative; the knock-in put leg is
short skew at the maturity pillar (a short put wing); an autocall digital at the money is *long*
skew at its own date (a digital call is a call spread, worth more when the skew steepens), so the
note's skew sign is date-dependent and the global rotation can carry either sign — reported per
leg and per pillar, not forced.  "Long forward" is the positive model delta and the negative repo
delta (``q + 1 bp`` lowers the forward with the discounting held); rho (``r + 1 bp``) also moves
the discounting of a par note and its sign is reported by the test.

**Expected life** (:func:`expected_life_sensitivity`): ``∂E[life]/∂ln S`` (central, ``"model"``
regime) and ``∂E[life]/∂σ`` (parallel +1 vp, sticky leverage) with
:class:`~volsto.products.autocall.AutocallStatistic` ``"life"`` as the priced product; a higher spot
autocalls earlier, so the spot sensitivity is negative.

**Likelihood-ratio cross-check** (§7.11, :func:`lr_cross_check`): :func:`~volsto.risk.estimators.
lr_delta` (spot normals aggregated over ``first_step``) and :func:`~volsto.risk.estimators.lr_vega`
of the note and of each digital leg (the conditional autocall digitals and the European knock-in
digital) against the ``"model"`` bump delta and the sticky-leverage bump vega from the engine at
the same state, with the z-score of :func:`~volsto.risk.estimators.compare`, the 3-stderr flag and
the ratio of the two standard errors.  Both sides discount with the state's rate curve (the
engine rebinds every product to it; the likelihood-ratio side is bound explicitly), so the
comparison is a Greek comparison whatever curve the note was built with.  The likelihood-ratio
error does not depend on the bump size, the bump's error grows as ``1/√h`` on a discontinuous
payoff; at the default 1% stencil the bump is the less noisy estimator of a 1–3y digital, the
ratio being the reported finding.

Checked by ``tests/test_risk_digital.py``.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from volsto.config import SimConfig, SurfacePerturbation
from volsto.models.base import Model
from volsto.products.autocall import (
    Autocall,
    AutocallStatistic,
    ConditionalDigital,
    KIPutLeg,
    _AutocallLeg,
)
from volsto.products.base import Product
from volsto.risk.engine import RiskEngine, RiskState, Sensitivity
from volsto.risk.estimators import compare, lr_delta, lr_vega
from volsto.risk.greeks import BUSINESS_DAY, _spot_state, delta_gamma, vega
from volsto.risk.ladders import PILLARS, skew_T, vega_T
from volsto.risk.product_risk import barrier_profile
from volsto.risk.profiles import spot_profile

if TYPE_CHECKING:
    from volsto.engine.grid import FixingIndex
    from volsto.engine.paths import PathSet
    from volsto.market.curves import DiscountCurve

FloatArray = NDArray[np.float64]
log = logging.getLogger(__name__)

_TOL = 1e-9
#: Autocall level that makes the event certain (``S ≥ 1e-12 · S_ref`` on every path).
CERTAIN_AUTOCALL = 1e-12
SKEW_BUMP = (
    "fixed-ATM rotation sigma -> sigma + s kappa(k) tent_i(T), kappa(k) = k_cap tanh(k/k_cap), "
    "s sized to +1 vp of 90/110 skew at the pillar (put wing up, call wing down)"
)
#: ``barrier_shift`` scopes of :func:`autocall_spot_profiles`: the autocall barrier of the
#: profile's date only (plus the knock-in level), or every autocall barrier of the residual note.
BARRIER_SHIFT_SCOPES = ("date", "all")


# --------------------------------------------------------------------------------------------
# term-sheet helpers
# --------------------------------------------------------------------------------------------


def _leg_label(leg: Product) -> str:
    """``autocall_i`` / ``bond`` / ``coupon`` / ``put`` / ``put_vanilla`` / ``put_digital``."""
    if isinstance(leg, _AutocallLeg):
        return leg.key
    return type(leg).__name__


def callable_dates(product: Autocall) -> list[int]:
    """1-based observation dates carrying an autocall event (after the non-call periods, finite
    barrier).  Checked by ``tests/test_risk_digital.py::test_residual_note_construction`` (a
    non-call first date drops out, the residual note re-indexes)."""
    ac = product.autocall_barriers
    return [
        i
        for i in range(1, product.n_dates + 1)
        if i > product.non_call_periods and np.isfinite(ac[i - 1])
    ]


def shifted_levels(
    product: Autocall, barrier_shift: float, *, dates: Sequence[int] | None = None
) -> Autocall:
    """The note with its *monitored* levels multiplied by ``1 + barrier_shift`` (a signed
    fraction, SPEC §6.5 convention): the knock-in level and the autocall barriers of ``dates``
    (1-based; ``None``: every date); the put strike ``S_ref``, the coupon barrier and the coupons
    do not move.  A shifted knock-in level above ``100%`` is refused by the term sheet.  Checked
    by ``tests/test_risk_digital.py::test_residual_note_construction`` (levels) and
    ``test_spot_profile_single_date_digital_black_scholes`` (the peak moves with the shift)."""
    if not isinstance(product, Autocall):
        raise TypeError("shifted_levels is defined for Autocall notes")
    if not np.isfinite(barrier_shift) or barrier_shift <= -1.0:
        raise ValueError("barrier_shift is a signed fraction of the levels and must exceed -1")
    if barrier_shift == 0.0:
        return product
    f = 1.0 + float(barrier_shift)
    ac = product.autocall_barriers.copy()
    if dates is None:
        ac *= f
    else:
        for d in dates:
            if int(d) != d or not 1 <= int(d) <= product.n_dates:
                raise ValueError("dates must be integers in 1..N")
            ac[int(d) - 1] *= f
    return product.replace(autocall_barriers=ac, ki_level=product.ki_level * f)


def residual_note(
    product: Autocall, date: int, *, spot: float, dt: float = BUSINESS_DAY
) -> tuple[Autocall, dict[str, Any]]:
    """The note as seen ``dt`` before observation date ``date`` (1-based) with the state held
    (module docstring): observation dates ``j ≥ date`` moved to ``T_j − (T_i − dt)`` (the first at
    ``dt``), the coupons of those dates (a growing coupon keeps its original index), the non-call
    periods reduced by ``date − 1``, the knock-in schedule shifted with the ``aged`` convention;
    the held path at ``spot`` decides the knock-in history (American: knocked in when the level
    was monitored before ``T_i − dt`` and ``spot`` lies below it — the residual note is then the
    European knock-in at ``100%``, or at the coupon barrier under ``final_redemption =
    "coupon_barrier"``) and the Phoenix memory carry (``spot`` below the coupon barrier: the
    coupons of the dropped dates were missed and are added to the first residual coupon).

    Returns the note and an info mapping: ``construction`` (``"aged"`` — the first date without a
    history adjustment, through :meth:`Autocall.aged` — or ``"explicit"``), ``date``, ``T``,
    ``valuation_time = T_i − dt``, ``ki_history`` (``"european"``, ``"alive"``, ``"knocked_in"``)
    and ``memory_carry``.  Raises when the date lies inside the roll step or the previous
    observation date sits inside the last ``dt`` before it.  Checked by
    ``tests/test_risk_digital.py::test_residual_note_construction`` (explicit versus ``aged`` path
    by path at the first date, the re-indexed coupons, the knocked-in and memory histories).
    """
    if not isinstance(product, Autocall):
        raise TypeError("residual_note is defined for Autocall notes")
    n = product.n_dates
    i = int(date)
    if i != date or not 1 <= i <= n:
        raise ValueError("date must be an integer in 1..N")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive")
    if not np.isfinite(spot) or spot <= 0:
        raise ValueError("spot must be positive")
    obs = product.observation_times
    t_i = float(obs[i - 1])
    t0 = t_i - dt
    if t0 <= 0.0:
        raise ValueError("the observation date must lie more than one roll step ahead")
    if i > 1 and float(obs[i - 2]) > t0 - _TOL:
        raise ValueError("the previous observation date lies inside the roll step before the date")
    s_ref = product.spot_reference
    ki_history = "european"
    if product.ki_type == "american":
        b = product.ki_barrier
        if product.ki_monitoring == "discrete":
            kf = product.ki_fixing_times
            assert kf is not None
            observed = bool(np.any(kf <= t0 + _TOL))
            knocked = observed and spot < b
        else:
            knocked = spot <= b
        ki_history = "knocked_in" if knocked else "alive"
    carry = 0.0
    if product.is_phoenix and product.memory and i > 1:
        assert product.coupon_barrier is not None
        if spot < product.coupon_barrier * s_ref:
            carry = float(np.sum(product.coupon_schedule[: i - 1]))
    info: dict[str, Any] = {
        "date": i,
        "T": t_i,
        "valuation_time": t0,
        "ki_history": ki_history,
        "memory_carry": carry,
    }
    if i == 1 and ki_history != "knocked_in":
        info["construction"] = "aged"
        return product.aged(t0), info
    changes: dict[str, Any] = {
        "observation_times": obs[i - 1 :] - t0,
        "autocall_barriers": product.autocall_barriers[i - 1 :].copy(),
        "non_call_periods": max(0, product.non_call_periods - (i - 1)),
    }
    schedule = product.coupon_schedule[i - 1 :].copy()
    schedule[0] += carry
    changes["coupons"] = tuple(float(c) for c in schedule)
    if ki_history == "knocked_in":
        level = 1.0 if product.final_redemption == "knock_in" else product.coupon_barrier
        changes.update(ki_type="european", ki_level=level, ki_monitoring=None, ki_fixing_times=None)
    elif product.ki_fixing_times is not None:
        t = product.ki_fixing_times
        kept = t[(t <= 0.0) | (t > t0 + _TOL)]
        changes["ki_fixing_times"] = np.where(kept > 0.0, kept - t0, 0.0)
    info["construction"] = "explicit"
    return product.replace(**changes), info


class SmoothedAutocall(Product):
    """A note whose autocall event at its *first* observation date is smoothed by a call spread of
    width ``width`` (spot units) around the barrier ``B = AC_1 S_ref`` (module docstring):

        payoff = θ(S_1) P_call + (1 − θ(S_1)) P_cont,  θ(S) = clip((S − B + w/2) / w, 0, 1),

    ``P_call`` the note with the first event certain (barrier :data:`CERTAIN_AUTOCALL`), ``P_cont``
    the note with it disabled (``AC_1 = ∞``); with ``θ → 1{S ≥ B}`` this is the note itself, path
    by path.  The notional rides on the inner notes.  Checked by ``tests/test_risk_digital.py``
    (``test_residual_note_construction`` for the split identity on synthetic paths,
    ``test_spot_profile_single_date_digital_black_scholes`` against the call-spread digital delta).
    """

    def __init__(self, note: Autocall, width: float) -> None:
        if not isinstance(note, Autocall):
            raise TypeError("SmoothedAutocall wraps an Autocall note")
        if note.non_call_periods >= 1 or not np.isfinite(note.autocall_barriers[0]):
            raise ValueError(
                "the first observation date carries no autocall event: nothing to smooth"
            )
        if not np.isfinite(width) or width <= 0:
            raise ValueError("width must be a positive amount in spot units")
        barrier = float(note.autocall_levels[0])
        if barrier - 0.5 * width <= 0:
            raise ValueError("the smoothing width must keep the lower strike B - w/2 positive")
        super().__init__(note.discount, 1.0)
        self.note = note
        self.width = float(width)
        self.barrier = barrier
        ac = note.autocall_barriers
        self.note_call = note.replace(
            autocall_barriers=np.concatenate([[CERTAIN_AUTOCALL], ac[1:]])
        )
        self.note_cont = note.replace(autocall_barriers=np.concatenate([[np.inf], ac[1:]]))
        self.requires_all_steps = note.requires_all_steps

    @property
    def fixing_times(self) -> FloatArray:
        return self.note.fixing_times

    @property
    def pay_times(self) -> FloatArray:
        return self.note.pay_times

    def weight(self, spot: FloatArray) -> FloatArray:
        """``θ(S)``: the call-spread weight of the autocall event."""
        w = self.width
        return np.asarray(np.clip((spot - self.barrier + 0.5 * w) / w, 0.0, 1.0), dtype=np.float64)

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        s1 = paths.spot_at(idx[float(self.note.observation_times[0])])
        th = self.weight(s1)
        return np.asarray(
            th * self.note_call.payoff(paths, idx) + (1.0 - th) * self.note_cont.payoff(paths, idx),
            dtype=np.float64,
        )

    def with_discount(self, discount: DiscountCurve) -> Product:
        return SmoothedAutocall(self.note.with_discount(discount), self.width)

    def aged(self, dt: float) -> Product:
        return SmoothedAutocall(self.note.aged(dt), self.width)

    def __repr__(self) -> str:
        t1 = float(self.note.observation_times[0])
        return (
            f"Call-spread smoothed autocall event at the first observation date {t1:g}y: width "
            f"{self.width:g} spot units (strikes B -/+ w/2 around B = {self.barrier:g}), later "
            f"digitals exact, of [{self.note!r}]"
        )


# --------------------------------------------------------------------------------------------
# spot profiles per observation date
# --------------------------------------------------------------------------------------------


def default_profile_shifts(
    width: float = 0.10, step: float = 0.01, fine_width: float = 0.02, fine_step: float = 0.0025
) -> tuple[float, ...]:
    """Relative moves around the barrier: ``±width`` in ``step`` steps refined to ``fine_step``
    within ``±fine_width`` (default 33 points).  Checked by ``tests/test_risk_digital.py::
    test_spot_profile_single_date_digital_black_scholes`` (the 33-point default grid) and
    ``test_residual_note_construction`` (validation)."""
    if not 0 < fine_step <= step or not 0 < fine_width <= width:
        raise ValueError("need 0 < fine_step <= step and 0 < fine_width <= width")
    coarse = np.arange(-width, width + 1e-12, step)
    fine = np.arange(-fine_width, fine_width + 1e-12, fine_step)
    grid = np.unique(np.round(np.concatenate([coarse, fine]), 10))
    return tuple(float(x) for x in grid)


def autocall_spot_profiles(
    engine: RiskEngine,
    product: Autocall,
    state: RiskState,
    *,
    dates: Sequence[int] | None = None,
    shifts: Sequence[float] | None = None,
    size: float = 0.01,
    smoothing: float = 0.0,
    barrier_shift: float = 0.0,
    barrier_shift_scope: str = "date",
    with_vega: bool = True,
    dt: float = BUSINESS_DAY,
) -> dict[int, pd.DataFrame]:
    """Model-regime spot profiles of the note one business day before each observation date
    (module docstring): ``{date: frame}`` with the :func:`~volsto.risk.profiles.spot_profile`
    columns (``shift`` relative to the state's spot, ``spot``, ``price``, ``delta``, ``gamma``,
    ``vega`` with standard errors, ``gamma_fd``) plus ``spot_over_barrier`` (contractual level)
    and ``spot_over_monitored``.  ``dates`` default to the observation dates with an autocall
    event (integers, no repeats); ``shifts`` are relative moves around the monitored barrier
    (:func:`default_profile_shifts`); ``size`` is the central log-spot stencil of the delta /
    gamma, itself a smoothing of the one-day digital over ``attrs["stencil_width"] ≈ 2·size·B``
    (module docstring: the 1% default reads the peak about 10% low, 0.25% within 0.7% at twice
    the noise); ``smoothing`` is the call-spread width as a fraction of ``S_ref`` (``0``: the
    exact digital); ``barrier_shift`` the signed fraction of the monitored levels and
    ``barrier_shift_scope`` (:data:`BARRIER_SHIFT_SCOPES`) which autocall barriers it moves —
    ``"date"``: the profile's date only, ``"all"``: every date of the residual note; the knock-in
    level moves in both.  ``frame.attrs`` records the residual note's construction, the
    contractual and monitored levels (``residual_autocall_barriers``, ``residual_ki_level``), the
    stencil, the smoothing and the shift.  Checked by ``tests/test_risk_digital.py::
    test_spot_profile_single_date_digital_black_scholes`` (single-date note against the
    Black–Scholes digital and call-spread digital stencils at 1% and 0.25%) and
    ``test_spot_profiles_multi_date_black_scholes`` (explicit residual notes, the smoothed
    residual note, the shift scopes)."""
    if not isinstance(product, Autocall):
        raise TypeError("autocall_spot_profiles is defined for Autocall notes")
    if not np.isfinite(smoothing) or smoothing < 0:
        raise ValueError("smoothing is a non-negative width as a fraction of spot_reference")
    if not np.isfinite(size) or size <= 0:
        raise ValueError("size must be positive")
    if barrier_shift_scope not in BARRIER_SHIFT_SCOPES:
        raise ValueError(f"barrier_shift_scope must be one of {BARRIER_SHIFT_SCOPES}")
    available = callable_dates(product)
    if dates is None:
        use = available
    else:
        use = []
        for d in dates:
            if isinstance(d, bool) or int(d) != d:
                raise ValueError("dates must be integer observation-date indices")
            if int(d) not in available:
                raise ValueError(f"date {d} carries no autocall event (non-call or disabled)")
            if int(d) in use:
                raise ValueError(f"date {d} is repeated")
            use.append(int(d))
    rel = default_profile_shifts() if shifts is None else tuple(float(x) for x in shifts)
    if not rel or any(not np.isfinite(r) or r <= -1.0 for r in rel):
        raise ValueError("shifts must be finite relative moves greater than -1")
    s_ref = product.spot_reference
    stencil = float(np.exp(size) - np.exp(-size))
    out: dict[int, pd.DataFrame] = {}
    for i in use:
        scoped = None if barrier_shift_scope == "all" else (i,)
        monitored = shifted_levels(product, barrier_shift, dates=scoped)
        note, info = residual_note(monitored, i, spot=state.spot, dt=dt)
        b_contract = float(product.autocall_levels[i - 1])
        b_mon = float(monitored.autocall_levels[i - 1])
        priced: Product = note if smoothing == 0.0 else SmoothedAutocall(note, smoothing * s_ref)
        spot_shifts = tuple(b_mon * (1.0 + r) / state.spot - 1.0 for r in rel)
        frame = spot_profile(
            engine, priced, state, spot_shifts, regime="model", size=size, with_vega=with_vega
        )
        frame.insert(1, "spot_over_barrier", frame["spot"] / b_contract)
        frame.insert(2, "spot_over_monitored", frame["spot"] / b_mon)
        frame.attrs.update(info)
        frame.attrs.update(
            {
                "autocall_barrier": b_contract,
                "monitored_barrier": b_mon,
                "barrier_shift": float(barrier_shift),
                "barrier_shift_scope": barrier_shift_scope,
                "residual_autocall_barriers": tuple(float(a) for a in note.autocall_barriers),
                "residual_ki_level": note.ki_level,
                "smoothing": float(smoothing),
                "smoothing_width": float(smoothing) * s_ref,
                "smoothed": (
                    f"autocall event at date {i} (call spread of width {smoothing * s_ref:g} "
                    "around the monitored barrier); later digitals and the knock-in exact"
                    if smoothing > 0
                    else "none"
                ),
                "regime": "model",
                "size": float(size),
                "stencil_width": b_mon * stencil,
                "stencil": (
                    f"central log-spot stencil +/-{size:g}: delta / gamma average the one-day "
                    f"digital over a band of width {b_mon * stencil:g} spot units at the barrier"
                ),
                "dt": float(dt),
                "residual_note": repr(priced),
            }
        )
        out[i] = frame
    return out


# --------------------------------------------------------------------------------------------
# knock-in put barrier risk
# --------------------------------------------------------------------------------------------


def _pct(s: Sensitivity, b: float) -> Sensitivity:
    return dataclasses.replace(
        s,
        name=f"{s.name}[1%]",
        value=s.value * 0.01 * b,
        stderr=s.stderr * 0.01 * b,
        unit="per 1% of barrier",
    )


def ki_put_barrier_risk(
    engine: RiskEngine, product: Autocall, state: RiskState, *, size: float = 0.005
) -> dict[str, Sensitivity]:
    """``∂price/∂B`` of the knock-in put leg by a central bump of the knock-in level ``±size``
    (relative) through :meth:`~volsto.risk.engine.RiskEngine.paired`, per unit spot
    (``dprice_dB``) and per 1% of the barrier (``dprice_dB_pct``); the European knock-in adds
    the vanilla / digital components (``dprice_dB_vanilla``, ``dprice_dB_digital`` and their
    ``_pct``); ``dpki_dB`` / ``dpki_dB_pct`` is ``∂P(KI)/∂B``.  The put leg is the only leg
    depending on ``B``, so ``dprice_dB`` is the note's.  Black–Scholes reference in the module
    docstring; checked by ``tests/test_risk_digital.py::test_ki_put_barrier_risk_black_scholes``.
    """
    if not isinstance(product, Autocall):
        raise TypeError("ki_put_barrier_risk is defined for Autocall notes")
    if not 0.0 < size < 1.0:
        raise ValueError("size is a relative bump of the knock-in level in (0, 1)")
    b = product.ki_barrier
    up = product.replace(ki_level=product.ki_level * (1.0 + size))
    dn = product.replace(ki_level=product.ki_level * (1.0 - size))
    c = 1.0 / (2.0 * b * size)
    extra = {"barrier": b, "ki_level": product.ki_level, "ki_type": product.ki_type}
    pairs: list[tuple[str, str, Product, Product, str]] = [
        ("dprice_dB", "dprice/dB", KIPutLeg(up), KIPutLeg(dn), "per unit barrier"),
    ]
    if product.ki_type == "european":
        pairs += [
            (
                "dprice_dB_vanilla",
                "dprice/dB[vanilla]",
                KIPutLeg(up, "vanilla"),
                KIPutLeg(dn, "vanilla"),
                "per unit barrier",
            ),
            (
                "dprice_dB_digital",
                "dprice/dB[digital]",
                KIPutLeg(up, "digital"),
                KIPutLeg(dn, "digital"),
                "per unit barrier",
            ),
        ]
    pairs.append(
        (
            "dpki_dB",
            "dP(KI)/dB",
            AutocallStatistic(up, "ki_hit"),
            AutocallStatistic(dn, "ki_hit"),
            "probability per unit barrier",
        )
    )
    out: dict[str, Sensitivity] = {}
    for key, name, p_up, p_dn, unit in pairs:
        s = engine.paired(
            name,
            [(p_up, state, "recalibrate", c), (p_dn, state, "recalibrate", -c)],
            unit=unit,
            size=size,
            scheme="central",
            extra=extra,
        )
        out[key] = s
        out[f"{key}_pct"] = _pct(s, b)
    return out


def ki_barrier_profile(
    engine: RiskEngine,
    product: Autocall,
    state: RiskState,
    *,
    width: float = 0.05,
    step: float = 0.005,
    size: float = 0.01,
    barrier_shift: float = 0.0,
) -> pd.DataFrame:
    """Model-regime delta / gamma profile of the whole note with the spot within ``±width`` of
    the (monitored) knock-in barrier in ``step`` steps (§7.10: ±5% at 0.5%, 21 rows), through
    :func:`~volsto.risk.product_risk.barrier_profile`; ``barrier_shift`` moves every monitored
    level (the knock-in level and every autocall barrier, :func:`shifted_levels`);
    ``spot_over_barrier`` is relative to the contractual knock-in level and
    ``spot_over_monitored`` to the monitored one, as in :func:`autocall_spot_profiles`.  Checked
    by ``tests/test_risk_digital.py::test_ki_put_barrier_risk_black_scholes`` (every row's delta
    and gamma against the Black–Scholes closed form of the module docstring; the columns under a
    shift)."""
    if not isinstance(product, Autocall):
        raise TypeError("ki_barrier_profile is defined for Autocall notes")
    monitored = shifted_levels(product, barrier_shift)
    frame = barrier_profile(
        engine, monitored, state, monitored.ki_barrier, width=width, step=step, size=size
    )
    frame.rename(columns={"spot_over_barrier": "spot_over_monitored"}, inplace=True)
    frame.insert(1, "spot_over_barrier", frame["spot"] / product.ki_barrier)
    frame.attrs.update(
        {
            "barrier": product.ki_barrier,
            "monitored_barrier": monitored.ki_barrier,
            "barrier_shift": float(barrier_shift),
            "regime": "model",
            "size": float(size),
        }
    )
    return frame


# --------------------------------------------------------------------------------------------
# vega-T and skew-T of the structure and its legs
# --------------------------------------------------------------------------------------------


def structure_vega_skew(
    engine: RiskEngine,
    product: Autocall,
    state: RiskState,
    pillars: Sequence[float] = PILLARS,
    size: float = 0.01,
    *,
    variant: str = "recalibrated",
    with_legs: bool = True,
) -> pd.DataFrame:
    """Vega-T waves / projections and skew-T entries of the note (``leg = "product"``) and of
    every leg of ``decompose()`` (the European knock-in adds the full put leg ``"put"``, the sum
    of ``put_vanilla`` and ``put_digital``); one row per ``(leg, pillar)`` with ``kind =
    "pillar"`` and a ``kind = "total"`` row per leg carrying the parallel vega (``vega_wave``),
    the sum of the projections (``vega_projection`` — identically the parallel vega, the
    projections telescoping on shared states, so it carries the parallel vega's standard error,
    not the quadrature sum of the correlated projections'), the global rotation (``skew``) and
    the sum of the skew ladder (``skew_ladder_sum``); ``skew_achieved`` is the achieved bump size
    after any halving.  ``frame.attrs`` keeps the :class:`~volsto.risk.engine.Sensitivity`
    objects of the totals and the bump definition (:data:`SKEW_BUMP`).  Checked by
    ``tests/test_risk_digital.py::test_sign_test_local_vol``."""
    if not isinstance(product, Autocall):
        raise TypeError("structure_vega_skew is defined for Autocall notes")
    ps = tuple(float(p) for p in pillars)
    items: list[tuple[str, Product]] = [("product", product)]
    if with_legs:
        legs = product.decompose()
        items += [(_leg_label(leg), leg) for leg in legs]
        if product.ki_type == "european":
            items.append(("put", KIPutLeg(product)))
    rows: list[dict[str, Any]] = []
    parallel: dict[str, Sensitivity] = {}
    global_skew: dict[str, Sensitivity] = {}
    for name, prod in items:
        vt = vega_T(engine, prod, state, ps, size, variant, with_tents=False)
        sk = skew_T(engine, prod, state, ps, size, variant)
        for j, p in enumerate(ps):
            rows.append(
                {
                    "leg": name,
                    "kind": "pillar",
                    "pillar": p,
                    "vega_wave": vt.waves[j].value,
                    "vega_wave_stderr": vt.waves[j].stderr,
                    "vega_projection": vt.projections[j].value,
                    "vega_projection_stderr": vt.projections[j].stderr,
                    "skew": sk.entries[j].value,
                    "skew_stderr": sk.entries[j].stderr,
                    "skew_achieved": sk.entries[j].size,
                    "skew_ladder_sum": np.nan,
                    "skew_ladder_sum_stderr": np.nan,
                }
            )
        assert sk.parallel is not None and sk.total is not None
        rows.append(
            {
                "leg": name,
                "kind": "total",
                "pillar": np.nan,
                "vega_wave": vt.parallel.value,
                "vega_wave_stderr": vt.parallel.stderr,
                "vega_projection": float(sum(p.value for p in vt.projections)),
                "vega_projection_stderr": vt.parallel.stderr,
                "skew": sk.parallel.value,
                "skew_stderr": sk.parallel.stderr,
                "skew_achieved": sk.parallel.size,
                "skew_ladder_sum": sk.total.value,
                "skew_ladder_sum_stderr": sk.total.stderr,
            }
        )
        parallel[name] = vt.parallel
        global_skew[name] = sk.parallel
    frame = pd.DataFrame(rows)
    frame.attrs.update(
        {
            "pillars": ps,
            "variant": variant,
            "size": float(size),
            "skew_bump": SKEW_BUMP,
            "parallel_vega": parallel,
            "global_skew": global_skew,
        }
    )
    return frame


# --------------------------------------------------------------------------------------------
# expected life
# --------------------------------------------------------------------------------------------


def expected_life_sensitivity(
    engine: RiskEngine,
    product: Autocall,
    state: RiskState,
    *,
    spot_size: float = 0.01,
    vol_size: float = 0.01,
) -> dict[str, Sensitivity]:
    """``life`` (``E[T_τ]`` in years), ``dlife_dlnS`` (central in log-spot under the ``"model"``
    regime, years per unit ``ln S``) and ``dlife_dsigma`` (parallel +1 vp with the leverage held,
    years per vol point) of the first-autocall time, with the standard error of the per-path CRN
    difference.  Checked by ``tests/test_risk_digital.py::
    test_expected_life_sensitivity_black_scholes`` (negative spot sensitivity at 3 stderr)."""
    if not isinstance(product, Autocall):
        raise TypeError("expected_life_sensitivity is defined for Autocall notes")
    if not (np.isfinite(spot_size) and spot_size > 0 and np.isfinite(vol_size) and vol_size > 0):
        raise ValueError("bump sizes must be finite and positive")
    life = AutocallStatistic(product, "life")
    base = engine.price(life, state)
    up, _ = _spot_state(state, "model", spot_size)
    dn, _ = _spot_state(state, "model", -spot_size)
    d_lns = engine.combination(
        "dE[life]/dlnS",
        life,
        [(up, "model", 1.0 / (2.0 * spot_size)), (dn, "model", -1.0 / (2.0 * spot_size))],
        unit="years per unit ln S",
        size=spot_size,
        scheme="central",
        extra={"regime": "model"},
    )
    bumped, achieved = engine.perturbed_state(
        state, lambda s: SurfacePerturbation("parallel", {"size": s}), vol_size, label="vega"
    )
    c = 0.01 / achieved
    d_vol = engine.combination(
        "dE[life]/dsigma",
        life,
        [(bumped, "sticky_leverage", c), (state, "recalibrate", -c)],
        unit="years per vol point",
        size=achieved,
        scheme="forward",
        extra={"variant": "sticky_leverage", "requested": vol_size},
    )
    return {
        "life": Sensitivity(
            "E[life]", base.mean, base.stderr, "years", 0.0, "none", (state.label,), base.n_paths
        ),
        "dlife_dlnS": d_lns,
        "dlife_dsigma": d_vol,
    }


# --------------------------------------------------------------------------------------------
# likelihood-ratio cross-check
# --------------------------------------------------------------------------------------------


def lr_cross_check(
    product: Autocall,
    model: Model,
    sim: SimConfig,
    *,
    engine: RiskEngine,
    state: RiskState,
    first_step: float | None = 1.0 / 52.0,
    spot_size: float | Sequence[float] = 0.01,
    vol_size: float | Sequence[float] = 0.01,
) -> pd.DataFrame:
    """Likelihood-ratio delta and vega (``model``, ``sim``) of the note and of each digital leg
    (``autocall_i`` conditional digitals; ``put_digital`` for a European knock-in) against the
    ``"model"`` bump delta and the sticky-leverage bump vega from ``engine`` at ``state`` (``model``
    must be the engine's base model at that state: the spots are checked).  Every product is
    bound to the model's rate curve for the likelihood-ratio pass, as the engine binds it for the
    bumps (module docstring).  One row per ``(leg, greek, requested_size)`` — several stencils
    share one likelihood-ratio pass, whose error does not depend on the stencil: ``lr``,
    ``lr_stderr``, ``bump``, ``bump_stderr``, ``z``, ``flag`` (``|z| > 3``,
    :func:`~volsto.risk.estimators.compare`), ``stderr_ratio = lr_stderr / bump_stderr``,
    ``requested_size`` (the stencil asked for; ``bump_size`` is the achieved one after any halving
    of the vol bump), ``first_step`` (the achieved aggregation window of the delta) and the
    simulated path counts.  Checked by ``tests/test_risk_digital.py::
    test_lr_cross_check_black_scholes`` (z-scores and error ratios; the curve binding: a note
    built on another curve gives the same rows)."""
    if not isinstance(product, Autocall):
        raise TypeError("lr_cross_check is defined for Autocall notes")
    if abs(model.spot - state.spot) > 1e-9 * state.spot:
        raise ValueError("model and state disagree on the spot: pass the engine's model at state")
    spot_sizes, vol_sizes = _sizes(spot_size), _sizes(vol_size)
    items: list[tuple[str, Product]] = [("product", product)]
    items += [(f"autocall_{i}", ConditionalDigital(product, i)) for i in callable_dates(product)]
    if product.ki_type == "european":
        items.append(("put_digital", KIPutLeg(product, "digital")))
    curve = model.forward_curve.rate_curve

    def row(
        name: str, greek: str, lr: Sensitivity, bump: Sensitivity, requested: float, tau: float
    ) -> dict[str, Any]:
        rep = compare(lr, bump)
        return {
            "leg": name,
            "greek": greek,
            "requested_size": requested,
            "bump_size": bump.size,
            **rep,
            "stderr_ratio": lr.stderr / bump.stderr if bump.stderr > 0 else np.nan,
            "first_step": tau,
            "lr_n_paths": sim.n_paths,
            "bump_n_paths": engine.sim.n_paths,
        }

    rows: list[dict[str, Any]] = []
    for name, prod in items:
        bound = prod.with_discount(curve)
        lr_d = lr_delta(bound, model, sim, first_step=first_step)
        lr_v = lr_vega(bound, model, sim)
        for h in spot_sizes:
            b_d, _ = delta_gamma(engine, prod, state, "model", h)
            rows.append(row(name, "delta", lr_d, b_d, h, lr_d.size))
        for v in vol_sizes:
            b_v = vega(engine, prod, state, "sticky_leverage", v)
            rows.append(row(name, "vega", lr_v, b_v, v, np.nan))
    return pd.DataFrame(rows)


def _sizes(size: float | Sequence[float]) -> tuple[float, ...]:
    """One or several bump sizes as floats; a numpy scalar counts as one size."""
    if isinstance(size, int | float | np.number):
        out: tuple[float, ...] = (float(size),)
    else:
        out = tuple(float(s) for s in size)
    if not out or any(not np.isfinite(s) or s <= 0 for s in out):
        raise ValueError("bump sizes must be finite and positive")
    return out


__all__ = [
    "BARRIER_SHIFT_SCOPES",
    "CERTAIN_AUTOCALL",
    "SKEW_BUMP",
    "SmoothedAutocall",
    "autocall_spot_profiles",
    "callable_dates",
    "default_profile_shifts",
    "expected_life_sensitivity",
    "ki_barrier_profile",
    "ki_put_barrier_risk",
    "lr_cross_check",
    "residual_note",
    "shifted_levels",
    "structure_vega_skew",
]
