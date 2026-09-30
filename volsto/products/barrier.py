"""Barrier machinery (SPEC §6.5, M6 Part 1): knock-in / knock-out options, one-touch / no-touch
and the smoothed cash-or-nothing digital, with discrete or continuous monitoring.

Conventions (all explicit constructor arguments, all reported in the term-sheet repr; an
argument that does not apply to the chosen monitoring mode raises ``ValueError`` instead of
being ignored):

* **Barrier and monitored level.**  ``B`` is the contractual barrier; the level actually
  monitored is ``B_eff = B (1 + barrier_shift)`` with ``barrier_shift`` a *signed* fraction of
  the barrier (default ``0``).  The shift is the desk convention for pricing digital risk
  conservatively; no direction convention is built in — the desk applies the sign per side.
  The strike and the rebate never move.
* **Knock rule.**  ``direction = "down"`` knocks when ``S ≤ B_eff`` (``x ≤ ln B_eff`` in
  log-spot), ``"up"`` when ``S ≥ B_eff``.  Under discrete monitoring the inequality is a
  *required* convention ``strict`` (no default, like the ``indicator`` of SPEC §6.1):
  ``strict=True`` knocks on ``<`` / ``>`` (the M4c close-to-close convention of the knock-out
  variance swap), ``strict=False`` on ``≤`` / ``≥`` (touching knocks).  The Brownian bridge is
  non-strict (touching is a measure-zero event), so ``strict`` must not be given with continuous
  monitoring.
* **Discrete monitoring** observes the closes on the monitoring schedule ``fixing_times``
  (default: daily from ``0`` to ``T``, so a breach at inception knocks, like the M4c swap);
  ``fixing_times=[T]`` is the European-at-maturity barrier.  ``survival`` and ``seed`` do not
  apply (the breach is an indicator) and must be left at ``"weight"`` / ``0``.
* **Continuous monitoring** (``requires_all_steps``): between consecutive recorded simulation
  steps the crossing probability of a Brownian bridge with the diffusion coefficient
  ``L(t, S) sqrt(V)`` frozen at the step start, ``p_i = exp(−2 (b − x_i)(b − x_{i+1}) /
  (σ_i² Δt_i))`` in log-spot (``σ_i² = paths.variance_at(step start column)``, ``b = ln
  B_eff``), ``p_i = 1`` when a recorded point is at or beyond the barrier.  The survival
  ``Π (1 − p_i)`` is used as a weight (``survival="weight"``, the lower-variance pricing form
  and the form the risk layer must use) or sampled (``"sampled"``: one uniform ``U`` per path,
  knocked at the first step whose cumulative survival falls to or below ``U`` — the exact joint
  law of the indicator and of the crossing step; ``seed`` applies to this form only).
  ``fixing_times`` restricts continuous monitoring to the window ``[min, max]`` of the given
  times (default ``[0, T]``).  The bridge is evaluated in blocks of :data:`STEP_BLOCK` steps
  with running products, so the temporaries are ``O(n_paths × STEP_BLOCK)`` rather than
  ``O(n_paths × n_steps)`` (the engine's chunk budget covers the recorded arrays only).
* **Rebate.**  A knock-out's ``rebate`` is paid at maturity (``rebate_timing="maturity"``) or
  at the knock-out (``"hit"``: on the breaching observation date for discrete monitoring; for
  continuous monitoring at the *midpoint* of the crossing step, in the weight form as the
  expected discount factor ``Σ_i (S_{i−1} − S_i) DF(t_i + Δt_i/2)`` over the cumulative survival
  ``S_i``).  The midpoint removes the first-order bias ``r R Δt/2`` of an end-of-step payment;
  what remains is ``r R`` times the mean offset of the crossing time from the step midpoint,
  second order in ``Δt`` for a crossing law symmetric within the step and invisible against the
  Reiner–Rubinstein ``F`` term at 3 stderr (``test_closed_forms_vs_bridge_monte_carlo``).  A
  knock-in's rebate is paid at maturity when the option never knocked in.  The timing must be
  given whenever ``rebate != 0``.
* **Sampled draws.**  The uniforms come from ``numpy.random.default_rng([seed, fingerprint])``
  with ``fingerprint`` the CRC-32 of the chunk's terminal log-spots: deterministic for a given
  path set (knock-in and knock-out built with the same ``seed`` share their draws, so in–out
  parity holds path by path in the sampled form as well) and distinct across chunks.  Caveat:
  the fingerprint changes under any spot / vol bump or pathwise perturbation, so the sampled
  form does **not** keep common random numbers across bumped states (Bernoulli noise enters a
  finite difference); bump risk must use ``survival="weight"``.  A bump-invariant, chunk-distinct
  stream needs a per-chunk offset from the engine (requested from the integrator).

Full-grid detection (the only full-grid evidence a :class:`~volsto.engine.paths.PathSet`
carries): a record interval is a single simulation step iff the kernel's ``Σ (Δ ln S)²``
accumulator increment (``realised_variance_grid``) equals the squared recorded log-return on
every path (every kernel updates ``ls += dx; sum_sq += dx²`` per step); an unrecorded sub-step
breaks the identity on essentially every path, so a coarser recording raises ``ValueError``
naming ``requires_all_steps``.  **Invariant:** any code that rewrites ``log_spot`` on a
``PathSet`` (pathwise perturbations such as the realised-variance exposure of the risk layer)
must rewrite ``sum_sq`` consistently — a bucket homothety ``x → x_lo + s (x − x_lo)`` maps the
accumulator to ``sq_lo + s² (sq − sq_lo)`` inside the bucket and shifts the later columns by
``(s² − 1)(sq_hi − sq_lo)`` — otherwise the continuous products raise on the perturbed set.

Checked by ``tests/test_barrier.py``: hand values of the bridge weights (including the
step-start variance convention), the weights on a stochastic-variance kernel, in–out parity path
by path (discrete, continuous weight and sampled), discrete → continuous convergence against the
Reiner–Rubinstein value with the Broadie–Glasserman–Kou shift as reference (common random
numbers), closed forms within 3 stderr, sampled versus weighted agreement with the sampled
variance predicted from the weights, the M4c knock-out variance swap rebuilt with
:func:`first_hit_index`, the barrier shift, and the digital smoothing identities.
"""

from __future__ import annotations

import zlib
from collections.abc import Iterator
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.products.base import CashFlow, Product, daily_schedule, parse_cp, shift_times
from volsto.products.gap import (
    GapReport,
    GapSpec,
    LevelFactors,
    month_grid,
    regress_at_level,
)
from volsto.products.vanilla import EuropeanOption

if TYPE_CHECKING:
    from volsto.engine.grid import FixingIndex
    from volsto.engine.paths import PathSet
    from volsto.market.curves import DiscountCurve

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

DIRECTIONS = ("up", "down")
MONITORINGS = ("discrete", "continuous")
SURVIVALS = ("weight", "sampled")
REBATE_TIMINGS = ("hit", "maturity")

#: Simulation steps per Brownian-bridge block (bounds the temporaries to ``n_paths × STEP_BLOCK``).
STEP_BLOCK = 64

_TOL = 1e-9


# --------------------------------------------------------------------------------------------
# Monitoring helpers
# --------------------------------------------------------------------------------------------


def first_hit_index(
    log_spots: ArrayLike, ln_barrier: float | FloatArray, direction: str, *, strict: bool
) -> IntArray:
    """Index of the first observation at or beyond the barrier along axis 1 of ``log_spots``
    (``(n_paths, n_obs)``), ``n_obs`` when the barrier is never breached.  ``ln_barrier`` is a
    scalar or an array broadcastable to ``log_spots`` (per-path, per-date effective levels of
    the smart gap).

    ``direction = "up"`` breaches when ``x ≥ ln_barrier`` (``>`` if ``strict``), ``"down"`` when
    ``x ≤ ln_barrier`` (``<`` if ``strict``).  With the closes ``S_0..S_N`` this is the ``j`` of
    the M4c knock-out variance swap (``τ = min(j, N)``); checked by
    ``tests/test_barrier.py::test_discrete_monitoring_matches_m4c_knock_out_variance_swap``.
    """
    ls = np.asarray(log_spots, dtype=np.float64)
    if ls.ndim != 2:
        raise ValueError("log_spots must have shape (n_paths, n_obs)")
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if direction == "up":
        hit = (ls > ln_barrier) if strict else (ls >= ln_barrier)
    else:
        hit = (ls < ln_barrier) if strict else (ls <= ln_barrier)
    n_obs = ls.shape[1]
    first = np.argmax(hit, axis=1)
    return np.asarray(np.where(hit.any(axis=1), first, n_obs), dtype=np.int64)


def _window_columns(paths: PathSet, idx: FixingIndex, t0: float, t1: float) -> tuple[int, int]:
    if not (0.0 <= t0 < t1):
        raise ValueError("need 0 <= t0 < t1 for the monitoring window")
    try:
        c0, c1 = idx[t0], idx[t1]
    except KeyError as exc:
        raise ValueError(
            f"monitoring window [{t0:g}, {t1:g}] must start and end at record times"
        ) from exc
    if paths.n_cols != len(idx) or not np.allclose(paths.times, idx.times, rtol=0, atol=_TOL):
        raise ValueError("the PathSet's columns do not match the FixingIndex")
    return c0, c1


def _bridge_blocks(
    paths: PathSet,
    times: FloatArray,
    ln_barrier: float,
    direction: str,
    c0: int,
    c1: int,
) -> Iterator[tuple[FloatArray, int]]:
    """Yield ``(1 − p_i over the steps starting at columns a..e−1, a)`` for consecutive blocks
    of at most :data:`STEP_BLOCK` steps covering ``[c0, c1]`` (module docstring), checking the
    full-grid accumulator identity block by block.  Reads the container only through its
    accessors (the asset axis is reserved for a second underlying)."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    b = float(ln_barrier)
    t0, t1 = float(times[c0]), float(times[c1])
    for a in range(c0, c1, STEP_BLOCK):
        e = min(a + STEP_BLOCK, c1)
        cols = np.arange(a, e + 1, dtype=np.int64)
        x = paths.log_spot_at(cols)
        dx = np.diff(x, axis=1)
        dsq = np.stack([paths.realised_variance_grid(c, c + 1) for c in range(a, e)], axis=1)
        if not np.allclose(dsq, dx * dx, rtol=1e-6, atol=1e-14):
            raise ValueError(
                "continuous monitoring needs every simulation step recorded: inside "
                f"[{t0:g}, {t1:g}] the PathSet's sum-of-squared-log-return accumulator does not "
                "match the squared recorded log-returns, i.e. the record columns are coarser "
                "than the simulation grid (set requires_all_steps on the product or "
                "SimConfig.record_all_steps) or log_spot was rewritten without sum_sq"
            )
        d = x - b
        alive = (d > 0.0) if direction == "down" else (d < 0.0)
        both = alive[:, :-1] & alive[:, 1:]
        denom = paths.variance_at(cols[:-1]) * np.diff(times[cols])[None, :]
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            p = np.exp(-2.0 * d[:, :-1] * d[:, 1:] / denom)
        p = np.where(denom > 0.0, p, 0.0)
        yield np.asarray(np.where(both, 1.0 - p, 0.0), dtype=np.float64), a


def bridge_step_survival(
    paths: PathSet, idx: FixingIndex, ln_barrier: float, direction: str, t0: float, t1: float
) -> tuple[FloatArray, IntArray]:
    """Per-step Brownian-bridge survival probabilities ``1 − p_i`` over the recorded simulation
    steps inside ``[t0, t1]``, shape ``(n_paths, n_steps)``, and the step-end columns.

    ``p_i = exp(−2 (b − x_i)(b − x_{i+1}) / (σ_i² Δt_i))`` with ``σ_i² = paths.variance_at(i)``
    (the step start) and ``Δt_i`` from the record times; ``p_i = 1`` (survival ``0``) when
    ``x_i`` or ``x_{i+1}`` is at or beyond the barrier (``x ≤ b`` for ``"down"``, ``x ≥ b`` for
    ``"up"``); a zero step variance gives ``p_i = 0`` between two alive points.  Raises
    ``ValueError`` (mentioning ``requires_all_steps``) when the path set does not carry every
    simulation step.  Materialises the full matrix (hand checks and diagnostics; the products
    and :func:`continuous_survival_weight` stream the blocks instead).  Checked by
    ``tests/test_barrier.py::test_bridge_weights_hand_values``.
    """
    c0, c1 = _window_columns(paths, idx, t0, t1)
    blocks = [s for s, _ in _bridge_blocks(paths, idx.times, ln_barrier, direction, c0, c1)]
    return np.concatenate(blocks, axis=1), np.arange(c0 + 1, c1 + 1, dtype=np.int64)


def _uniforms(paths: PathSet, seed: int) -> FloatArray:
    """One uniform per path, deterministic for the path set (module docstring, *Sampled draws*:
    keyed on the chunk's terminal log-spots, hence not common across bumped states)."""
    terminal = paths.log_spot_at(paths.n_cols - 1)
    fingerprint = zlib.crc32(np.ascontiguousarray(terminal).tobytes())
    rng = np.random.default_rng([int(seed), int(fingerprint)])
    return np.asarray(rng.random(paths.n_paths), dtype=np.float64)


def continuous_survival_weight(
    paths: PathSet,
    idx: FixingIndex,
    ln_barrier: float,
    direction: str,
    t0: float,
    t1: float,
    *,
    sample: bool = False,
    seed: int = 0,
) -> FloatArray:
    """Per-path Brownian-bridge survival over the recorded steps inside ``[t0, t1]``:
    ``Π_i (1 − p_i)`` (zero when a recorded log-spot is at or beyond the barrier), or with
    ``sample=True`` a 0/1 survival draw — ``1{U < Π (1 − p_i)}`` with one uniform ``U`` per
    path from ``default_rng([seed, fingerprint])`` (see the module docstring; the same rule the
    products use, so their sampled payoffs are reproducible from this function).  Raises
    ``ValueError`` naming ``requires_all_steps`` on a path set without every simulation step.
    Checked by ``tests/test_barrier.py`` (``test_bridge_weights_hand_values``,
    ``test_continuous_requires_all_steps``, ``test_bridge_on_stochastic_variance_kernel``,
    ``test_sampled_vs_weighted``).
    """
    c0, c1 = _window_columns(paths, idx, t0, t1)
    weight = np.ones(paths.n_paths)
    for surv, _ in _bridge_blocks(paths, idx.times, ln_barrier, direction, c0, c1):
        weight *= np.prod(surv, axis=1)
    if not sample:
        return np.asarray(weight, dtype=np.float64)
    return np.asarray((weight > _uniforms(paths, seed)).astype(np.float64), dtype=np.float64)


# --------------------------------------------------------------------------------------------
# Products
# --------------------------------------------------------------------------------------------


class _BarrierBase(Product):
    """Monitoring conventions shared by the barrier and touch products (module docstring).

    ``strict`` is required for ``monitoring="discrete"`` and must be ``None`` for
    ``"continuous"``; ``survival`` (default ``"weight"``) and ``seed`` (default ``0``, applies to
    ``survival="sampled"`` only) are continuous-monitoring conventions and must keep their
    defaults under discrete monitoring.  Checked by
    ``tests/test_barrier.py::test_validation_reprs_and_ageing``.
    """

    def __init__(
        self,
        barrier: float,
        direction: str,
        maturity: float,
        discount: DiscountCurve,
        *,
        monitoring: str,
        fixing_times: ArrayLike | None,
        barrier_shift: float,
        survival: str,
        strict: bool | None,
        seed: int,
        notional: float,
        gap: GapSpec | None = None,
    ) -> None:
        super().__init__(discount, notional)
        if not np.isfinite(barrier) or barrier <= 0 or not np.isfinite(maturity) or maturity <= 0:
            raise ValueError("barrier and maturity must be positive")
        # gap conventions (owner addendum at the M6 review): ``barrier_shift`` is the fixed
        # gap; ``gap=GapSpec(mode="fixed", fixed_shift=s)`` is the same thing spelled through
        # the spec (the two must agree when both are given); ``mode="smart"`` is the signed
        # state-dependent gap of volsto.products.gap with ``barrier_shift`` as its fallback
        if gap is not None:
            if not isinstance(gap, GapSpec):
                raise TypeError("gap must be a GapSpec or None")
            if gap.fixed_shift != 0.0 and barrier_shift != 0.0 and gap.fixed_shift != barrier_shift:
                raise ValueError("gap.fixed_shift and barrier_shift disagree: give one of them")
            if gap.fixed_shift != 0.0:
                barrier_shift = gap.fixed_shift
            if gap.smart and monitoring != "discrete":
                raise NotImplementedError(
                    "the smart gap is implemented for discrete monitoring (the Brownian bridge "
                    "has no per-date level to regress at)"
                )
        self.gap = gap
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
        if monitoring not in MONITORINGS:
            raise ValueError(f"monitoring must be one of {MONITORINGS}, got {monitoring!r}")
        if survival not in SURVIVALS:
            raise ValueError(f"survival must be one of {SURVIVALS}, got {survival!r}")
        if not np.isfinite(barrier_shift) or barrier_shift <= -1.0:
            raise ValueError("barrier_shift is a signed fraction of the barrier and must exceed -1")
        if int(seed) != seed or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if monitoring == "discrete":
            if strict is None:
                raise ValueError(
                    "strict is required for discrete monitoring: True knocks on S < B_eff / "
                    "S > B_eff (the M4c swap convention), False when touching knocks (<= / >=)"
                )
            if survival != "weight" or seed != 0:
                raise ValueError(
                    "survival and seed apply to continuous monitoring only (a discrete "
                    "close-to-close breach is an indicator): leave survival='weight', seed=0"
                )
        else:
            if strict is not None:
                raise ValueError(
                    "strict applies to discrete monitoring only (the Brownian bridge is "
                    "non-strict: touching the barrier is a measure-zero event)"
                )
            if survival == "weight" and seed != 0:
                raise ValueError("seed applies to survival='sampled' only")
        self.barrier = float(barrier)
        self.direction = direction
        self.T = float(maturity)
        self.monitoring = monitoring
        self.barrier_shift = float(barrier_shift)
        self.survival = survival
        self.strict: bool | None = None if strict is None else bool(strict)
        self.seed = int(seed)
        self.requires_all_steps = monitoring == "continuous"
        if monitoring == "discrete":
            sched = (
                daily_schedule(self.T)
                if fixing_times is None
                else np.unique(np.asarray(fixing_times, dtype=np.float64))
            )
            if sched.size == 0 or not np.all(np.isfinite(sched)):
                raise ValueError("the monitoring schedule must be a non-empty set of finite times")
            if sched[0] < 0.0 or sched[-1] > self.T + _TOL:
                raise ValueError("monitoring times must lie in [0, maturity]")
            self.schedule = np.minimum(sched, self.T)
        else:
            if fixing_times is None:
                t0, t1 = 0.0, self.T
            else:
                win = np.asarray(fixing_times, dtype=np.float64)
                if win.size == 0 or not np.all(np.isfinite(win)):
                    raise ValueError(
                        "the monitoring window must be a non-empty set of finite times"
                    )
                t0, t1 = float(win.min()), float(win.max())
            if not (0.0 <= t0 < t1 <= self.T + _TOL):
                raise ValueError("continuous monitoring needs a window 0 <= t0 < t1 <= maturity")
            self.schedule = np.array([t0, min(t1, self.T)])

    # -- conventions -------------------------------------------------------------------------

    @property
    def barrier_eff(self) -> float:
        """Monitored level ``B (1 + barrier_shift)``."""
        return self.barrier * (1.0 + self.barrier_shift)

    @property
    def ln_barrier(self) -> float:
        return float(np.log(self.barrier_eff))

    @property
    def fixing_times(self) -> FloatArray:
        return np.unique(np.concatenate([self.schedule, [self.T]]))

    def _kwargs(self) -> dict[str, object]:
        """Constructor keywords shared by every subclass (for ``aged`` / decompositions): the
        subset relevant to the monitoring mode, so a round trip never trips the mode checks."""
        kw: dict[str, object] = {
            "monitoring": self.monitoring,
            "fixing_times": self.schedule,
            "barrier_shift": self.barrier_shift,
            "gap": self.gap,
        }
        if self.monitoring == "discrete":
            kw["strict"] = self.strict
        else:
            kw["survival"] = self.survival
            if self.survival == "sampled":
                kw["seed"] = self.seed
        return kw

    def _aged_times(self, dt: float) -> tuple[float, FloatArray]:
        """Maturity and monitoring schedule ``dt`` later with the state held: monitoring dates
        inside ``(0, dt]`` were observed at the held spot and drop out (``t = 0`` stays), the
        rest move earlier — the convention shared with :class:`~volsto.products.autocall.Autocall`
        (a daily-monitored contract must be ageable by one business day for theta); a continuous
        window is shifted with its start clipped at 0."""
        T = float(shift_times([self.T], dt)[0])
        sched = self.schedule
        if self.monitoring == "discrete":
            kept = sched[(sched <= 0.0) | (sched > dt + 1e-12)]
            return T, shift_times(kept, dt)
        return T, np.array([max(float(sched[0]) - dt, 0.0), float(sched[1]) - dt])

    # -- monitoring --------------------------------------------------------------------------

    def monitor(self, paths: PathSet, idx: FixingIndex) -> tuple[FloatArray, FloatArray]:
        """Per path: the survival (probability or indicator that the barrier was never breached)
        and the expected discount factor to the knock time ``E[DF(τ) 1{τ ≤ T}]`` (used by
        rebates paid at the hit; continuous monitoring discounts from the midpoint of the
        crossing step, module docstring).  Checked by
        ``tests/test_barrier.py::test_rebate_timing_and_european_barrier``."""
        if self.monitoring == "discrete":
            if self.gap is not None and self.gap.smart:
                return self._monitor_smart(paths, idx)
            alive, hit_df, _ = self._monitor_discrete(paths, idx, self.ln_barrier)
            return alive, hit_df
        t0, t1 = float(self.schedule[0]), float(self.schedule[1])
        c0, c1 = _window_columns(paths, idx, t0, t1)
        times = idx.times
        n = paths.n_paths
        blocks = _bridge_blocks(paths, times, self.ln_barrier, self.direction, c0, c1)
        cum = np.ones(n)
        if self.survival == "weight":
            hit_df = np.zeros(n)
            for surv, a in blocks:
                k = surv.shape[1]
                block_cum = cum[:, None] * np.cumprod(surv, axis=1)
                prev = np.concatenate([cum[:, None], block_cum[:, :-1]], axis=1)
                df_mid = self.df(0.5 * (times[a : a + k] + times[a + 1 : a + k + 1]))
                hit_df += np.sum((prev - block_cum) * df_mid[None, :], axis=1)
                cum = block_cum[:, -1]
            return np.asarray(cum, dtype=np.float64), np.asarray(hit_df, dtype=np.float64)
        u = _uniforms(paths, self.seed)
        hit_col = np.full(n, -1, dtype=np.int64)  # start column of the crossing step
        for surv, a in blocks:
            block_cum = cum[:, None] * np.cumprod(surv, axis=1)
            dead = block_cum <= u[:, None]
            new = (hit_col < 0) & dead.any(axis=1)
            if new.any():
                hit_col[new] = a + np.argmax(dead[new], axis=1)
            cum = block_cum[:, -1]
        knocked = hit_col >= 0
        hc = np.maximum(hit_col, 0)
        df_mid = self.df(0.5 * (times[hc] + times[hc + 1]))
        alive = (~knocked).astype(np.float64)
        hit_df = np.where(knocked, df_mid, 0.0)
        return alive, np.asarray(hit_df, dtype=np.float64)

    def _monitor_discrete(
        self, paths: PathSet, idx: FixingIndex, ln_levels: float | FloatArray
    ) -> tuple[FloatArray, FloatArray, IntArray]:
        """Discrete monitoring on the log-levels ``ln_levels`` (a scalar, or ``(n_paths, n_obs)``
        per-path effective levels): survival indicator, discount factor to the hit, hit index."""
        assert self.strict is not None
        ls = paths.log_spot_at(idx.indices(self.schedule))
        j = first_hit_index(ls, ln_levels, self.direction, strict=self.strict)
        n_obs = self.schedule.size
        alive = (j == n_obs).astype(np.float64)
        dfs = self.df(self.schedule)
        hit_df = np.where(j < n_obs, dfs[np.minimum(j, n_obs - 1)], 0.0)
        return alive, np.asarray(hit_df, dtype=np.float64), j

    def _knocked_liability(
        self, paths: PathSet, idx: FixingIndex, t: float
    ) -> tuple[FloatArray, bool]:
        """Seller's liability per path once the barrier has been breached at ``t``, in ``t``
        money per unit notional and per unit of the barrier level, and whether it depends on
        the path (then it is regressed on the state like the continuing liability)."""
        raise NotImplementedError

    def _payoff_given(
        self, alive: FloatArray, hit_df: FloatArray, paths: PathSet, idx: FixingIndex
    ) -> FloatArray:
        """The discounted payoff (notional included) from a monitoring result."""
        raise NotImplementedError

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        alive, hit_df = self.monitor(paths, idx)
        return self._payoff_given(alive, hit_df, paths, idx)

    def _monitor_smart(self, paths: PathSet, idx: FixingIndex) -> tuple[FloatArray, FloatArray]:
        """Two-pass smart gap (:mod:`volsto.products.gap`) for a discretely monitored barrier:
        pass 1 monitors the fixed level (``barrier_shift``) and gives every path's discounted
        payoff; on the regression grid dates (:func:`~volsto.products.gap.month_grid` of the
        schedule, spacing ``ki_grid_months``) the continuing liability is the §7.11 regression
        of the payoff (date money, per unit notional and per unit of the barrier level) over the
        paths not knocked at or before the date on ``(ln S − ln B, X)``, evaluated at the level
        for the paths not knocked before the date; the knocked liability is the subclass's
        (rebate / payout — analytic — or the vanilla, regressed the same way); ``ΔV = L_knocked
        − L_continuing`` per path, the shift per path from :func:`conservative_shift` and the
        spec's sizing function, held until the next grid date.  Pass 2 monitors the per-path,
        per-date effective levels ``B (1 + shift)`` on the same paths."""
        assert self.gap is not None and self.gap.smart
        gap = self.gap
        n = paths.n_paths
        alive0, hit_df0, j0 = self._monitor_discrete(paths, idx, self.ln_barrier)
        pay0 = self._payoff_given(alive0, hit_df0, paths, idx) / (self.notional * self.barrier)
        sched = self.schedule
        grid = month_grid(sched, gap.ki_grid_months)
        g_index = np.searchsorted(sched, grid - _TOL)
        gcols = idx.indices(grid)
        ln_grid = paths.log_spot_at(gcols)
        dfs = np.asarray(self.df(grid), dtype=np.float64)
        fac = np.full((n, grid.size), 1.0 + self.barrier_shift)
        ln_b = float(np.log(self.barrier))
        report = GapReport()
        for k, t in enumerate(grid):
            g = int(g_index[k])
            alive = j0 >= g  # not knocked before the grid date
            cont = j0 > g  # not knocked at or before it
            y = pay0 / dfs[k]
            feats = np.column_stack([ln_grid[:, k] - ln_b, paths.factors_at(int(gcols[k]))])
            v_cont, n_c, se_c = regress_at_level(gap, y, feats, cont, alive)
            knocked, path_dependent = self._knocked_liability(paths, idx, float(t))
            if path_dependent:
                v_kn, n_k, se_k = regress_at_level(gap, knocked, feats, alive, alive)
            else:
                v_kn, n_k, se_k = knocked[alive], n_c, np.zeros(int(alive.sum()))
            n_fit = min(n_c, n_k)
            dv = np.full(n, np.nan)
            shift = np.full(n, self.barrier_shift)
            if n_fit > 0:
                dv_alive = v_kn - v_cont
                dv[alive] = dv_alive
                shift[alive] = gap.shift(dv_alive, self.direction, np.hypot(se_c, se_k))
                fac[:, k] = 1.0 + shift
            near = alive & (np.abs(feats[:, 0]) <= gap.report_band)
            report.add("barrier", float(t), dv, shift, n_fit, near=near)
        levels = LevelFactors(fixed=1.0 + self.barrier_shift, grid=grid, factors=fac)
        ln_levels = ln_b + np.log(levels.at(sched, n))
        alive_eff, hit_df_eff, _ = self._monitor_discrete(paths, idx, ln_levels)
        self._last_gap_report = report
        return alive_eff, hit_df_eff

    def gap_report(self, paths: PathSet, idx: FixingIndex) -> GapReport:
        """The per-date smart-gap summary on these paths (:class:`GapReport`; ``ΔV`` per unit
        of the barrier level; monitors the product)."""
        if self.gap is None or not self.gap.smart:
            raise ValueError("gap_report needs a smart gap")
        self._monitor_smart(paths, idx)
        return self._last_gap_report

    def _monitoring_repr(self) -> str:
        if self.monitoring == "discrete":
            op = {"up": ">" if self.strict else ">=", "down": "<" if self.strict else "<="}
            return (
                f"discrete monitoring ({self.schedule.size} obs from {self.schedule[0]:g}y to "
                f"{self.schedule[-1]:g}y, knocked when S {op[self.direction]} B_eff)"
            )
        seed = (
            f", seed {self.seed}, draws keyed on the path set (no CRN across bumps)"
            if self.survival == "sampled"
            else ""
        )
        return (
            f"continuous monitoring on [{self.schedule[0]:g}, {self.schedule[1]:g}]y "
            f"(Brownian bridge, survival {self.survival}{seed})"
        )

    def _barrier_repr(self) -> str:
        text = (
            f"barrier {self.barrier:g} (monitored {self.barrier_eff:g}, shift "
            f"{self.barrier_shift * 100:+g}%)"
        )
        if self.gap is not None and self.gap.smart:
            text += f" with {self.gap!r} (fixed shift is the fallback)"
        return text


def _validate_rebate(rebate: float, rebate_timing: str | None, knock: str) -> str | None:
    if not np.isfinite(rebate):
        raise ValueError("rebate must be finite")
    if rebate_timing is not None and rebate_timing not in REBATE_TIMINGS:
        raise ValueError(f"rebate_timing must be one of {REBATE_TIMINGS}, got {rebate_timing!r}")
    if rebate != 0.0 and rebate_timing is None:
        raise ValueError("rebate_timing ('hit' or 'maturity') is required when rebate != 0")
    if knock == "in" and rebate_timing == "hit":
        raise ValueError("a knock-in rebate is paid at maturity (rebate_timing='maturity')")
    return rebate_timing


class _BarrierOption(_BarrierBase):
    """Vanilla leg ``(cp (S_T − K))⁺`` plus the barrier and rebate conventions."""

    knock: str

    def __init__(
        self,
        strike: float,
        maturity: float,
        cp: int | str,
        barrier: float,
        direction: str,
        discount: DiscountCurve,
        *,
        monitoring: str,
        fixing_times: ArrayLike | None = None,
        rebate: float = 0.0,
        rebate_timing: str | None = None,
        barrier_shift: float = 0.0,
        survival: str = "weight",
        strict: bool | None = None,
        seed: int = 0,
        notional: float = 1.0,
        gap: GapSpec | None = None,
        seasoned: bool = False,
    ) -> None:
        super().__init__(
            barrier,
            direction,
            maturity,
            discount,
            monitoring=monitoring,
            fixing_times=fixing_times,
            barrier_shift=barrier_shift,
            survival=survival,
            strict=strict,
            seed=seed,
            notional=notional,
            gap=gap,
        )
        if not np.isfinite(strike) or strike <= 0:
            raise ValueError("strike must be positive")
        self.strike = float(strike)
        self.cp = parse_cp(cp)
        self.rebate = float(rebate)
        self.rebate_timing = _validate_rebate(self.rebate, rebate_timing, self.knock)
        #: set by :func:`volsto.products.seasoning.season` (dates count from the as-of date; the
        #: realised monitoring dates did not knock)
        self.seasoned = bool(seasoned)

    @property
    def is_seasoned(self) -> bool:
        return self.seasoned

    def vanilla_payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        """Undiscounted ``(cp (S_T − K))⁺`` per path."""
        S = paths.spot_at(idx[self.T])
        return np.asarray(np.maximum(self.cp * (S - self.strike), 0.0), dtype=np.float64)

    def vanilla(self) -> EuropeanOption:
        """The European option the barrier is written on."""
        return EuropeanOption(self.strike, self.T, self.cp, self.discount, self.notional)

    def _rebuild(
        self,
        cls: type[_BarrierOption],
        T: float,
        schedule: FloatArray,
        *,
        discount: DiscountCurve | None = None,
        seasoned: bool | None = None,
    ) -> Product:
        kw = self._kwargs()
        kw["fixing_times"] = schedule
        return cls(
            self.strike,
            T,
            self.cp,
            self.barrier,
            self.direction,
            self.discount if discount is None else discount,
            rebate=self.rebate,
            rebate_timing=self.rebate_timing,
            notional=self.notional,
            seasoned=self.seasoned if seasoned is None else seasoned,
            **kw,  # type: ignore[arg-type]
        )

    def aged(self, dt: float) -> Product:
        T, schedule = self._aged_times(dt)
        return self._rebuild(type(self), T, schedule)

    def with_barrier(self, barrier: float) -> Product:
        """The same contract with its barrier level at ``barrier`` (the monitoring, shift, gap and
        rebate conventions kept): the barrier bump of :func:`volsto.risk.product_risk.
        barrier_sensitivity` and the barrier-shift reserve."""
        kw = self._kwargs()
        return type(self)(
            self.strike,
            self.T,
            self.cp,
            float(barrier),
            self.direction,
            self.discount,
            rebate=self.rebate,
            rebate_timing=self.rebate_timing,
            notional=self.notional,
            seasoned=self.seasoned,
            **kw,  # type: ignore[arg-type]
        )

    def _repr(self, kind: str) -> str:
        opt = "Call" if self.cp > 0 else "Put"
        reb = f"rebate {self.rebate:g}"
        if self.rebate != 0.0:
            reb += f" at {self.rebate_timing}"
        return (
            f"{self.direction.capitalize()}-and-{kind} {opt}: strike {self.strike:g}, expiry "
            f"{self.T:g}y, {self._barrier_repr()}, {self._monitoring_repr()}, {reb}, "
            f"notional {self.notional:g}"
        )


class KnockOutOption(_BarrierOption):
    """Knock-out option: ``notional · [DF(T) 1{alive} (cp (S_T − K))⁺ + rebate · (DF(T) 1{knocked}
    or DF(τ) 1{τ ≤ T})]`` with the survival of the module docstring (weighted, sampled or
    discrete).  Reiner–Rubinstein under Black–Scholes with continuous monitoring
    (:func:`volsto.market.barrier_bs.bs_barrier_price`); checked by ``tests/test_barrier.py``.
    """

    knock = "out"

    def _payoff_given(
        self, alive: FloatArray, hit_df: FloatArray, paths: PathSet, idx: FixingIndex
    ) -> FloatArray:
        df_T = float(self.df(self.T))
        cf = df_T * alive * self.vanilla_payoff(paths, idx)
        if self.rebate != 0.0:
            timing = hit_df if self.rebate_timing == "hit" else df_T * (1.0 - alive)
            cf = cf + self.rebate * timing
        return np.asarray(self.notional * cf, dtype=np.float64)

    def _knocked_liability(
        self, paths: PathSet, idx: FixingIndex, t: float
    ) -> tuple[FloatArray, bool]:
        # the rebate: at the hit (date money) or at maturity (discounted to the date)
        value = 0.0
        if self.rebate != 0.0:
            value = self.rebate * (
                1.0 if self.rebate_timing == "hit" else float(self.df(self.T) / self.df(t))
            )
        return np.full(paths.n_paths, value / self.barrier), False

    def __repr__(self) -> str:
        return self._repr("out")


class KnockInOption(_BarrierOption):
    """Knock-in option: ``notional · DF(T) [1{knocked} (cp (S_T − K))⁺ + rebate 1{alive}]``;
    ``decompose()`` is the in–out parity ``vanilla − knock-out(rebate at maturity) + rebate``,
    exact path by path (``tests/test_barrier.py::test_in_out_parity_path_by_path``)."""

    knock = "in"

    def _payoff_given(
        self, alive: FloatArray, hit_df: FloatArray, paths: PathSet, idx: FixingIndex
    ) -> FloatArray:
        df_T = float(self.df(self.T))
        cf = df_T * ((1.0 - alive) * self.vanilla_payoff(paths, idx) + self.rebate * alive)
        return np.asarray(self.notional * cf, dtype=np.float64)

    def _knocked_liability(
        self, paths: PathSet, idx: FixingIndex, t: float
    ) -> tuple[FloatArray, bool]:
        # once knocked in the seller owes the vanilla: its terminal payoff, discounted to the
        # date, regressed on the state at the date
        scale = float(self.df(self.T) / self.df(t)) / self.barrier
        return np.asarray(scale * self.vanilla_payoff(paths, idx), dtype=np.float64), True

    def decompose(self) -> list[Product]:
        kw = self._kwargs()
        ko = KnockOutOption(
            self.strike,
            self.T,
            self.cp,
            self.barrier,
            self.direction,
            self.discount,
            rebate=self.rebate,
            rebate_timing="maturity" if self.rebate != 0.0 else None,
            notional=-self.notional,
            **kw,  # type: ignore[arg-type]
        )
        legs: list[Product] = [self.vanilla(), ko]
        if self.rebate != 0.0:
            legs.append(CashFlow(self.rebate, self.T, self.discount, self.notional))
        return legs

    def __repr__(self) -> str:
        return self._repr("in")


class _Touch(_BarrierBase):
    def __init__(
        self,
        barrier: float,
        maturity: float,
        direction: str,
        discount: DiscountCurve,
        *,
        monitoring: str,
        fixing_times: ArrayLike | None = None,
        payout: float = 1.0,
        barrier_shift: float = 0.0,
        survival: str = "weight",
        strict: bool | None = None,
        seed: int = 0,
        notional: float = 1.0,
        gap: GapSpec | None = None,
    ) -> None:
        super().__init__(
            barrier,
            direction,
            maturity,
            discount,
            monitoring=monitoring,
            fixing_times=fixing_times,
            barrier_shift=barrier_shift,
            survival=survival,
            strict=strict,
            seed=seed,
            notional=notional,
            gap=gap,
        )
        if not np.isfinite(payout):
            raise ValueError("payout must be finite")
        self.payout = float(payout)

    def aged(self, dt: float) -> Product:
        T, schedule = self._aged_times(dt)
        kw = self._kwargs()
        kw["fixing_times"] = schedule
        return type(self)(
            self.barrier,
            T,
            self.direction,
            self.discount,
            payout=self.payout,
            notional=self.notional,
            **kw,  # type: ignore[arg-type]
        )

    def _repr(self, kind: str) -> str:
        return (
            f"{kind} ({self.direction}): payout {self.payout:g} at {self.T:g}y, "
            f"{self._barrier_repr()}, {self._monitoring_repr()}, notional {self.notional:g}"
        )


class OneTouch(_Touch):
    """Pays ``notional · payout`` at ``T`` if the barrier was touched: ``DF(T) (1 − survival)``.
    Black–Scholes: :func:`volsto.market.barrier_bs.bs_one_touch_price`; checked by
    ``tests/test_barrier.py::test_closed_forms_vs_bridge_monte_carlo``."""

    def _payoff_given(
        self, alive: FloatArray, hit_df: FloatArray, paths: PathSet, idx: FixingIndex
    ) -> FloatArray:
        return np.asarray(
            self.notional * self.payout * float(self.df(self.T)) * (1.0 - alive), dtype=np.float64
        )

    def _knocked_liability(
        self, paths: PathSet, idx: FixingIndex, t: float
    ) -> tuple[FloatArray, bool]:
        value = self.payout * float(self.df(self.T) / self.df(t)) / self.barrier
        return np.full(paths.n_paths, value), False

    def __repr__(self) -> str:
        return self._repr("One-touch")


class NoTouch(_Touch):
    """Pays ``notional · payout`` at ``T`` if the barrier was never touched: ``DF(T) survival``
    (one-touch + no-touch = ``payout DF(T)`` path by path)."""

    def _payoff_given(
        self, alive: FloatArray, hit_df: FloatArray, paths: PathSet, idx: FixingIndex
    ) -> FloatArray:
        return np.asarray(
            self.notional * self.payout * float(self.df(self.T)) * alive, dtype=np.float64
        )

    def _knocked_liability(
        self, paths: PathSet, idx: FixingIndex, t: float
    ) -> tuple[FloatArray, bool]:
        return np.zeros(paths.n_paths), False

    def __repr__(self) -> str:
        return self._repr("No-touch")


class Digital(Product):
    """Cash-or-nothing digital with optional call-spread smoothing (the risk convention).

    ``smoothing = 0``: pays ``notional · payout`` at ``T`` if ``cp (S_T − K) > 0`` (the exact
    digital, identical to :class:`~volsto.products.vanilla.DigitalOption`).  ``smoothing = w >
    0``: the indicator is replaced by the unit-height spread of vanillas struck at ``K ∓ w/2``,
    ``[(cp (S_T − K + cp·w/2))⁺ − (cp (S_T − K − cp·w/2))⁺] / w`` — for a call
    ``[(S − (K − w/2))⁺ − (S − (K + w/2))⁺] / w``, for a put ``[((K + w/2) − S)⁺ − ((K − w/2) −
    S)⁺] / w`` — which equals the spread of two :class:`EuropeanOption` path by path
    (``decompose()``) and converges to the exact digital as ``w → 0`` (``E|smooth − exact|`` is
    ``payout DF w/4`` times the spot density at the strike, first order in ``w``).
    Black–Scholes: :func:`volsto.market.barrier_bs.bs_digital_price`.  Checked by
    ``tests/test_barrier.py::test_digital_smoothing``.
    """

    def __init__(
        self,
        strike: float,
        maturity: float,
        cp: int | str,
        discount: DiscountCurve,
        payout: float = 1.0,
        notional: float = 1.0,
        smoothing: float = 0.0,
    ) -> None:
        super().__init__(discount, notional)
        if not np.isfinite(strike) or strike <= 0 or not np.isfinite(maturity) or maturity <= 0:
            raise ValueError("strike and maturity must be positive")
        if not np.isfinite(payout):
            raise ValueError("payout must be finite")
        if not np.isfinite(smoothing) or smoothing < 0:
            raise ValueError("smoothing must be a non-negative width in spot units")
        if smoothing > 0 and strike - 0.5 * smoothing <= 0:
            raise ValueError("smoothing width must keep the lower strike K - w/2 positive")
        self.strike = float(strike)
        self.T = float(maturity)
        self.cp = parse_cp(cp)
        self.payout = float(payout)
        self.smoothing = float(smoothing)

    @property
    def fixing_times(self) -> FloatArray:
        return np.array([self.T])

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        S = paths.spot_at(idx[self.T])
        w = self.smoothing
        if w == 0.0:
            ind = (self.cp * (S - self.strike) > 0.0).astype(np.float64)
        else:
            lo = np.maximum(self.cp * (S - self.strike) + 0.5 * w, 0.0)
            hi = np.maximum(self.cp * (S - self.strike) - 0.5 * w, 0.0)
            ind = (lo - hi) / w
        return np.asarray(
            self.notional * self.payout * float(self.df(self.T)) * ind, dtype=np.float64
        )

    def decompose(self) -> list[Product] | None:
        """The vanilla spread for ``smoothing > 0`` (``None`` for the exact digital)."""
        if self.smoothing == 0.0:
            return None
        w = self.smoothing
        size = self.notional * self.payout / w
        k_lo, k_hi = self.strike - 0.5 * w, self.strike + 0.5 * w
        if self.cp > 0:
            return [
                EuropeanOption(k_lo, self.T, 1, self.discount, size),
                EuropeanOption(k_hi, self.T, 1, self.discount, -size),
            ]
        return [
            EuropeanOption(k_hi, self.T, -1, self.discount, size),
            EuropeanOption(k_lo, self.T, -1, self.discount, -size),
        ]

    def aged(self, dt: float) -> Product:
        return Digital(
            self.strike,
            float(shift_times([self.T], dt)[0]),
            self.cp,
            self.discount,
            self.payout,
            self.notional,
            self.smoothing,
        )

    def __repr__(self) -> str:
        kind = "Call" if self.cp > 0 else "Put"
        smooth = (
            "exact"
            if self.smoothing == 0.0
            else f"call-spread smoothing width {self.smoothing:g} (strikes K -/+ w/2)"
        )
        return (
            f"Digital {kind}: strike {self.strike:g}, expiry {self.T:g}y, payout {self.payout:g}, "
            f"{smooth}, notional {self.notional:g}"
        )
