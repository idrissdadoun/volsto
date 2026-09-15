"""Conditional pricing under the pricing model by regression (SPEC §8, M8 Part 1; the
conditional-Greeks engine of §7.11 generalised to every hedge object and every rebalancing
date).

**Accounting convention — time-0 money.**  Every value is discounted to ``t = 0`` with the
deterministic curves: a product's conditional value at a rebalancing date ``t_k`` is
``V_k = E[total discounted payoff | F_k]``, which includes cash flows already paid (they are
``F_k``-measurable) and equals the realised discounted payoff at maturity.  The hedge P&L of the
product leg is then ``Σ_k (V_{k+1} − V_k) = payoff − V_0`` **exactly** whatever the regression
error (the sum telescopes; only the per-date split carries the regression noise), and discounted
values are martingales under the pricing model, so no funding leg is needed.  The same holds for
each hedge instrument: a position ``q_k`` held over ``[t_k, t_{k+1}]`` earns ``q_k (I_{k+1} −
I_k)`` in time-0 money.

**Regression.**  The pricing model is simulated once (``n_paths`` of the pricing configuration)
on the union grid of the rebalancing dates and every object's fixings; at each date the object's
value is regressed on the **spline-tensor basis** :func:`hedge_basis` — a cubic spline in
``ln S_t`` (truncated-power form, ``n_knots`` interior knots at the alive paths' quantiles: a global
cubic cannot follow the near-step delta of a short-dated option, which is where the gamma and
the hedging noise sit; measured on the Black–Scholes daily delta hedge the global cubic gave a
P&L std 3.3× the exact-delta bound) tensored with a quadratic (with cross terms) in the other
features — in the object's **hedge state** — ``(ln S_t, X_t)`` plus the product's
own path-dependent state of :mod:`volsto.hedging.state` (accrued variance, accumulated cliquet
sum, knock-in status, …) — on the paths where the object is alive; a terminated path holds its
settled value.  **Greeks by hybrid CRN bumps.**  The spot bumps ``S₀ e^{±h}`` are simulated on
the same normals (leverage held in spot: the ``"model"`` regime).  For an object with a hedge
state the plain bumped path is the wrong object — it scales the whole path, past fixings
included, and measures the homogeneity direction (the cliquet's cap-call strip had a zero bump
delta inside every period) — and the derivative of the fitted value in ``ln S_t`` is hopeless
one day into a period (measured: the strip's fitted-gradient delta had a std of 0.43 against a
true 0.004, correlation 0 with the closed form — a regression cannot resolve a spot dependence
that runs through a one-day-old return feature).  The delta target at ``t`` is therefore the CRN
difference of the payoff on **hybrid paths**: the base path up to and including the column of
``t`` (the state, fixings at ``t`` included) spliced to the bumped path after it (a future sample
started from the bumped spot, cumulative accumulators rebased at the splice, the splice step's
squared log-return kept consistent for the continuous-monitoring check) — the conditional delta
with the history held, at the CRN variance; gamma is the second difference; both are normalised
by ``S_t/S₀`` (and its square) as §7.11.  For a state-free object the hybrid and the plain
bumped path coincide.  The **model bumps** are hybridised the same way (``hybrid_bumps``): a
bumped model's path has its own history, so the plain CRN difference of a path-dependent payoff
carries the history's sensitivity as noise (measured on the study cliquet's static replication
``cliquet + strip − put = Σ rᵢ``, whose vega is zero: the plain-bump vega exposure grew from 0.22
to 0.80 over the year and the variance-swap quantity it drove was noise); the bumped future is
re-anchored at the base spot at ``t`` and spliced onto the base history.  The fitted-gradient
estimator is kept as ``delta_estimator="gradient"`` for diagnostics.  Any number of
**bumped pricing models** (:class:`Bump`: a vol-shifted or surface-perturbed and
recalibrated model, a model-parameter bump, …) whose per-path payoff differences are regressed the
same way — each bump is one extra simulation on the same normals.  Sensitivities to the factor
state ``X_t`` come from the gradient of the fitted value polynomial (no extra simulation; noted
as such).

**Evaluation at the world's states.**  The fitted polynomials are evaluated at the world paths'
hedge states.  When the world model carries a different factor structure from the pricing model
(a local-vol world under a two-factor pricing model, or the reverse) the pricing factors are set
to their initial value 0 at every date — the desk's model sees no factor move it cannot observe
— and the report says so; same class and factor count pass the factors through.

Checked by ``tests/test_hedging.py`` (``test_conditional_pricer_bs``: the regressed value and
delta of a vanilla against Black–Scholes along the paths; the t = 0 value equals the Monte Carlo
price on the same draws).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volsto.config import SimConfig
from volsto.engine.grid import TimeGrid
from volsto.engine.paths import PathSet
from volsto.engine.rng import GaussianDraws
from volsto.hedging.state import HedgeState, hedge_state
from volsto.models.base import Model
from volsto.products.base import Product

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]
_TOL = 1e-9
#: fewer alive paths than this and the regression falls back to the alive mean (noted)
MIN_REGRESSION_PATHS = 50
#: interior knots of the ``ln S`` spline (at the alive paths' quantiles)
DEFAULT_KNOTS = 8
#: bump kinds: ``"model"`` (difference over ``unit``, central when ``dn`` is given), ``"spot"``
#: (a spot move under a delta regime: normalised like the CRN delta, by ``S_up − S_dn`` and
#: ``S_t/S₀``), ``"second"`` (second difference ``(up − 2 base + dn)/unit²``)
BUMP_KINDS = ("model", "spot", "second")


@dataclass(frozen=True)
class Bump:
    """A bumped pricing model for one target: ``value(up) − value(dn or base)`` over ``unit``.
    ``kind`` names the sensitivity (``"vega"``, ``"fwd_var:0.25-0.5"``, ``"skew_T:1"``, …)."""

    name: str
    up: Model
    dn: Model | None = None
    unit: float = 1.0
    description: str = ""
    kind: str = "model"

    def __post_init__(self) -> None:
        if self.unit == 0 or not np.isfinite(self.unit):
            raise ValueError("unit must be a finite non-zero number")
        if self.kind not in BUMP_KINDS:
            raise ValueError(f"kind must be one of {BUMP_KINDS}")
        if self.kind in ("second", "spot") and self.dn is None:
            raise ValueError(f"a {self.kind!r} bump needs both up and dn models")


def hedge_basis(x: FloatArray, knots: FloatArray, degree: int = 2) -> FloatArray:
    """``[spline(x₀)] ⊗ [1, x_i]  ∪  [x_i x_j]`` on standardised features: a cubic truncated-power
    spline ``[1, x, x², x³, (x − k)₊³ …]`` in the first feature (``ln S``) tensored with the
    *linear* terms of the others, plus the additive quadratic (cross) terms of the others when
    ``degree ≥ 2`` (``degree`` 0 gives the spline alone).  Columns ``(4 + n_knots)(1 + d) + d(d +
    1)/2``: 12 / 25 / 39 / 54 for ``d = 0 … 3`` other features with 8 knots.  The full
    spline ⊗ quadratic tensor (72 columns at ``d = 2``, 252 at ``d = 5``) was measured to overfit
    the noisy sensitivity targets in the sparse corners of the state space (sixth-order cross
    terms: single-path fitted vegas of ±40 against a true 0 on the study cliquet's net
    portfolio), which the linear interaction does not."""
    x = np.asarray(x, dtype=np.float64)
    n, d = x.shape
    x0 = x[:, 0]
    cols0 = [np.ones(n), x0, x0**2, x0**3] + [np.maximum(x0 - k, 0.0) ** 3 for k in knots]
    spline = np.column_stack(cols0)
    if d == 1 or degree == 0:
        return np.asarray(spline, dtype=np.float64)
    rest = x[:, 1:]
    lin = np.column_stack([np.ones(n)] + [rest[:, i] for i in range(d - 1)])
    parts = [(spline[:, :, None] * lin[:, None, :]).reshape(n, -1)]
    if degree >= 2:
        parts.append(
            np.column_stack(
                [rest[:, i] * rest[:, j] for i in range(d - 1) for j in range(i, d - 1)]
            )
        )
    return np.asarray(np.column_stack(parts), dtype=np.float64)


@dataclass
class Fit:
    """The regressions of one object at one date: coefficient vectors per quantity (``value``,
    ``delta`` and ``gamma`` — stored as the first and second derivatives in ``ln S_t``, converted
    per unit spot by :meth:`ConditionalPricer.evaluate` —, one per bump name), the feature
    standardisation, the kept columns and clip range, the spline knots, the alive paths' share
    and the fit diagnostics."""

    t: float
    degree: int
    mean: FloatArray
    scale: FloatArray
    coefficients: dict[str, FloatArray]
    r2: dict[str, float]
    n_alive: int
    n_paths: int
    fallback: bool = False
    knots: FloatArray = field(default_factory=lambda: np.zeros(0))
    state_constant: bool = True
    moving: tuple[int, ...] = (0,)
    columns: tuple[int, ...] = (0,)
    lo: FloatArray = field(default_factory=lambda: np.full(1, -np.inf))
    hi: FloatArray = field(default_factory=lambda: np.full(1, np.inf))

    def basis(self, features: FloatArray) -> FloatArray:
        """The design matrix: the kept feature columns (``columns`` — a state column constant
        across the pricing paths at this date, a ``started`` flag or the in-period return on a
        fixing date, carries nothing and is dropped) standardised, **clipped to the pricing
        paths' range** ``[lo, hi]`` (the fitted function is held constant beyond the tails it
        was fitted on: a cubic spline extrapolates violently — measured: one world path beyond
        the range turned the variance swap's fitted vega negative and its solved quantity into
        13 000) and expanded."""
        f = np.asarray(features, dtype=np.float64)[:, list(self.columns)]
        x = np.clip((f - self.mean) / self.scale, self.lo, self.hi)
        return hedge_basis(x, self.knots, self.degree)

    def predict(self, kind: str, features: FloatArray) -> FloatArray:
        return np.asarray(self.basis(features) @ self.coefficients[kind], dtype=np.float64)

    def gradient(self, kind: str, features: FloatArray, column: int) -> FloatArray:
        """Derivative of the prediction in raw feature ``column`` (central difference in the
        standardised feature; one-sided at the clip boundary so the gradient beyond the fitted
        range is the boundary gradient, not zero — measured: a zero vanna on the 0.2% of world
        paths beyond the range made the spanned three-instrument solve singular there)."""
        f = np.asarray(features, dtype=np.float64)
        if column not in self.columns:
            return np.zeros(f.shape[0])
        j = self.columns.index(column)
        z = np.clip((f[:, list(self.columns)] - self.mean) / self.scale, self.lo, self.hi)
        h = 1e-4
        up, dn = z.copy(), z.copy()
        up[:, j] = np.minimum(z[:, j] + h, self.hi[j])
        dn[:, j] = np.maximum(z[:, j] - h, self.lo[j])
        num = (
            hedge_basis(up, self.knots, self.degree) @ self.coefficients[kind]
            - hedge_basis(dn, self.knots, self.degree) @ self.coefficients[kind]
        )
        den = (up[:, j] - dn[:, j]) * float(self.scale[j])
        # a degenerate range (t = 0: every path at the same state) has no gradient
        return np.asarray(np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0), 0.0))

    def directional(self, kind: str, features: FloatArray) -> tuple[FloatArray, FloatArray]:
        """First and second derivatives of the prediction along ``ln S_t`` with the history held:
        every ``moving`` column (``ln S`` and the ``u_`` features) shifted together."""
        f = np.asarray(features, dtype=np.float64)
        h = 1e-3 * float(self.scale[0])
        up, dn = f.copy(), f.copy()
        for c in self.moving:
            up[:, c] += h
            dn[:, c] -= h
        pu, pm, pd = self.predict(kind, up), self.predict(kind, f), self.predict(kind, dn)
        return np.asarray((pu - pd) / (2 * h)), np.asarray((pu - 2.0 * pm + pd) / (h * h))


DELTA_ESTIMATORS = ("hybrid_crn", "gradient")
CLIP_QUANTILE = 0.001  # the basis is evaluated inside the pricing paths' [0.1%, 99.9%] range
# ridge on the normal equations, relative to each column's own diagonal entry (scale-free), the
# intercept unpenalised (so the t = 0 fit is exactly the sample mean and the product leg
# telescopes to payoff - V0 to round-off): a numerical stabiliser for near-collinear columns,
# not a smoother (a ridge of 1e-3 x n_paths on the raw columns crushed the last spline pieces
# and took the daily Black-Scholes delta hedge from 1.09x to 1.46x the exact-delta bound)
DEFAULT_RIDGE = 1e-6


@dataclass
class ObjectPayoffs:
    """Per-path discounted payoffs of one object under the base and the bumped models."""

    base: FloatArray
    up: FloatArray
    dn: FloatArray
    bumps: dict[str, tuple[FloatArray, FloatArray | None]]


@dataclass
class ConditionalPricer:
    """The pricing-model simulation shared by every object and date (module docstring)."""

    model: Model
    objects: list[Product]
    grid: TimeGrid
    sim: SimConfig
    bumps: tuple[Bump, ...] = ()
    degree: int = 2
    spot_size: float = 0.01
    seed: int | None = None
    n_knots: int = DEFAULT_KNOTS
    delta_estimator: str = "hybrid_crn"
    hybrid_bumps: bool = True
    ridge: float = DEFAULT_RIDGE
    paths: PathSet = field(init=False, repr=False)
    payoffs: list[ObjectPayoffs] = field(init=False, repr=False)
    bumped_paths: dict[str, PathSet] = field(init=False, repr=False, default_factory=dict)
    hybrids: dict[float, dict[str, FloatArray]] = field(
        init=False, repr=False, default_factory=dict
    )
    fits: dict[tuple[int, float], Fit] = field(init=False, repr=False, default_factory=dict)
    states: dict[tuple[int, float], HedgeState] = field(
        init=False, repr=False, default_factory=dict
    )
    notes: list[str] = field(init=False, default_factory=list)
    n_simulations: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if not self.objects:
            raise ValueError("no objects to price")
        names = [b.name for b in self.bumps]
        if len(set(names)) != len(names):
            raise ValueError("bump names must be unique")
        if self.delta_estimator not in DELTA_ESTIMATORS:
            raise ValueError(f"delta_estimator must be one of {DELTA_ESTIMATORS}")
        self._simulate()

    # -- simulation ----------------------------------------------------------------------------

    def _simulate(self) -> None:
        idx = self.grid.fixing_index
        mc_draws = GaussianDraws(
            self.sim.seed if self.seed is None else self.seed,
            self.sim.n_paths,
            self.grid.n_steps,
            self.model.n_brownians,
            self.sim.antithetic,
        )
        s0 = self.model.spot
        up = self.model.bump(spot=s0 * float(np.exp(self.spot_size)))
        dn = self.model.bump(spot=s0 * float(np.exp(-self.spot_size)))
        models: list[tuple[str, Model]] = [("base", self.model), ("up", up), ("dn", dn)]
        for b in self.bumps:
            models.append((f"bump:{b.name}:up", b.up))
            if b.dn is not None:
                models.append((f"bump:{b.name}:dn", b.dn))
        n_obj = len(self.objects)
        pay: dict[str, FloatArray] = {k: np.empty((self.sim.n_paths, n_obj)) for k, _ in models}
        parts: dict[str, list[PathSet]] = {k: [] for k, _ in models}
        for p0, p1 in self.sim.chunk_ranges(self.grid.n_records, self.model.n_factors):
            for key, m in models:
                ps = m.simulate_chunk(self.grid, mc_draws, p0, p1, self.sim.scheme)
                for j, obj in enumerate(self.objects):
                    pay[key][p0:p1, j] = obj.payoff(ps, idx)
                parts[key].append(ps)
        self.n_simulations = len(models)
        self.paths = PathSet.concat(parts.pop("base"))
        # every bumped path set is kept for the hybrid (history-held) targets at every date;
        # memory_bytes reports the footprint
        self.bumped_paths = {k: PathSet.concat(v) for k, v in parts.items()}
        self.payoffs = []
        for j in range(n_obj):
            bumps: dict[str, tuple[FloatArray, FloatArray | None]] = {}
            for b in self.bumps:
                bumps[b.name] = (
                    pay[f"bump:{b.name}:up"][:, j],
                    None if b.dn is None else pay[f"bump:{b.name}:dn"][:, j],
                )
            self.payoffs.append(
                ObjectPayoffs(pay["base"][:, j], pay["up"][:, j], pay["dn"][:, j], bumps)
            )
        self._s_up = up.spot
        self._s_dn = dn.spot

    # -- hybrid paths ----------------------------------------------------------------------------

    def _hybrid(self, bumped: PathSet, col: int, shift: float) -> PathSet:
        """The base path up to column ``col`` (the state at ``t``) spliced to the future of
        ``bumped`` **re-anchored** at the base spot (times ``e^shift``): ``ln S`` after the splice
        is the bumped path's increments from ``ln S_t^base + shift`` — a sample of the bumped
        model's future from the base state (exact where the increments do not depend on the
        level, second order otherwise), the bumped model's own history discarded.  The
        cumulative accumulators are rebased at the splice; the splice step's ``sum_sq`` increment
        is the hybrid's own squared log-return so the continuous-monitoring check holds."""
        base = self.paths
        m = base.n_cols
        ls = base.log_spot.copy()
        var = base.variance.copy()
        fac = base.factors.copy()
        iv = base.int_var.copy()
        ss = base.sum_sq.copy()
        if col + 1 < m:
            off = base.log_spot[:, col] + shift - bumped.log_spot[:, col]
            ls[:, col + 1 :] = bumped.log_spot[:, col + 1 :] + off[:, None]
            var[:, col + 1 :] = bumped.variance[:, col + 1 :]
            fac[:, col + 1 :, :] = bumped.factors[:, col + 1 :, :]
            iv[:, col + 1 :] = (
                base.int_var[:, [col]] + bumped.int_var[:, col + 1 :] - bumped.int_var[:, [col]]
            )
            jump = (ls[:, col + 1] - ls[:, col]) ** 2
            ss[:, col + 1 :] = (
                base.sum_sq[:, [col]]
                + jump[:, None]
                + bumped.sum_sq[:, col + 1 :]
                - bumped.sum_sq[:, [col + 1]]
            )
        return PathSet(base.times, ls, var, fac, iv, ss)

    def hybrid_payoffs(self, t: float) -> dict[str, FloatArray]:
        """Every object's payoff, ``(n_paths, n_objects)`` per bumped set (``"up"`` / ``"dn"``
        for the spot bumps, ``"bump:<name>:up"`` / ``":dn"`` for the model bumps), on the hybrid
        paths of date ``t``; the last three dates are cached."""
        key = float(t)
        if key in self.hybrids:
            return self.hybrids[key]
        col = self.idx[key]
        n_obj = len(self.objects)
        out: dict[str, FloatArray] = {}
        for name, bumped in self.bumped_paths.items():
            if name.startswith("bump:") and not self.hybrid_bumps:
                continue
            shift = {"up": self.spot_size, "dn": -self.spot_size}.get(name, 0.0)
            hyb = self._hybrid(bumped, col, shift)
            pay = np.empty((self.sim.n_paths, n_obj))
            for j, obj in enumerate(self.objects):
                pay[:, j] = obj.payoff(hyb, self.idx)
            out[name] = pay
        while len(self.hybrids) >= 3:
            del self.hybrids[next(iter(self.hybrids))]
        self.hybrids[key] = out
        return out

    @property
    def memory_bytes(self) -> int:
        """Footprint of the kept path sets (base and every bumped set)."""
        sets = [self.paths, *self.bumped_paths.values()]
        return int(
            sum(
                p.log_spot.nbytes
                + p.variance.nbytes
                + p.factors.nbytes
                + p.int_var.nbytes
                + p.sum_sq.nbytes
                for p in sets
            )
        )

    # -- features --------------------------------------------------------------------------------

    @property
    def idx(self) -> Any:
        return self.grid.fixing_index

    def features(self, obj_index: int, paths: PathSet, t: float) -> tuple[FloatArray, HedgeState]:
        """``(ln S_t, X_t, object state)`` on ``paths`` (pricing or world) for one object; a
        world with another factor structure gets the pricing factors set to 0 (module
        docstring)."""
        obj = self.objects[obj_index]
        col = self.idx[float(t)]
        hs = hedge_state(obj, paths, self.idx, t)
        base = [paths.log_spot_at(col)]
        nf = self.model.n_factors
        if nf:
            if paths.n_factors == nf:
                base.append(paths.factors_at(col))
            else:
                base.append(np.zeros((paths.n_paths, nf)))
        feats = (
            np.column_stack([*base, hs.features]) if hs.features.shape[1] else np.column_stack(base)
        )
        return np.asarray(feats, dtype=np.float64), hs

    # -- regression ----------------------------------------------------------------------------

    def fit(self, obj_index: int, t: float) -> Fit:
        """The regressions of object ``obj_index`` at date ``t`` (cached)."""
        key = (obj_index, float(t))
        if key in self.fits:
            return self.fits[key]
        feats, hs = self.features(obj_index, self.paths, t)
        self.states[key] = hs
        for note in hs.notes:
            if note not in self.notes:
                self.notes.append(note)
        pay = self.payoffs[obj_index]
        alive = hs.alive
        col = self.idx[float(t)]
        s_t = np.exp(self.paths.log_spot_at(col))
        ratio = s_t / self.model.spot
        targets: dict[str, FloatArray] = {"value": pay.base}
        hyb = (
            self.hybrid_payoffs(float(t))
            if (self.delta_estimator == "hybrid_crn" or self.hybrid_bumps)
            else {}
        )
        if self.delta_estimator == "hybrid_crn":
            p_up, p_dn = hyb["up"][:, obj_index], hyb["dn"][:, obj_index]
        else:
            p_up, p_dn = pay.up, pay.dn
        d_raw = (p_up - p_dn) / (self._s_up - self._s_dn)
        c_up = 2.0 / ((self._s_up - self.model.spot) * (self._s_up - self._s_dn))
        c_dn = 2.0 / ((self.model.spot - self._s_dn) * (self._s_up - self._s_dn))
        g_raw = c_up * p_up - (c_up + c_dn) * pay.base + c_dn * p_dn
        # regressed as derivatives in ln S_t (dollar delta / gamma): flat across the range for
        # a returns-type payoff, so the constant extrapolation beyond the fitted range holds on
        # the world's crash paths (a per-unit-spot delta ~ 1/S_prev clipped at the 0.1% quantile
        # was 2-3x too small on the local-vol world's -70..-90% paths, which then carried the
        # whole hedged variance of the study cliquet's static replication)
        d_ln = d_raw / ratio * s_t
        targets["delta"] = d_ln
        targets["gamma"] = g_raw / ratio**2 * s_t * s_t + d_ln
        for b in self.bumps:
            if self.hybrid_bumps:
                u = hyb[f"bump:{b.name}:up"][:, obj_index]
                d = hyb[f"bump:{b.name}:dn"][:, obj_index] if b.dn is not None else None
            else:
                u, d = pay.bumps[b.name]
            if b.kind == "second":
                assert d is not None
                targets[b.name] = (u - 2.0 * pay.base + d) / (b.unit * b.unit)
            elif b.kind == "spot":
                assert d is not None
                targets[b.name] = (u - d) / (b.up.spot - b.dn.spot) / ratio  # type: ignore[union-attr]
            else:
                targets[b.name] = (
                    (u - d) / (2.0 * b.unit) if d is not None else (u - pay.base) / b.unit
                )
        raw = feats[alive]
        n_alive = int(alive.sum())
        nf = self.model.n_factors
        moving = [0] + [1 + nf + i for i, nm in enumerate(hs.names) if nm.startswith("u_")]
        state_cols = raw[:, 1 + nf :] if raw.shape[1] > 1 + nf else np.zeros((raw.shape[0], 0))
        state_constant = bool(state_cols.shape[1] == 0 or np.all(state_cols.std(axis=0) == 0.0))
        coefficients: dict[str, FloatArray] = {}
        r2: dict[str, float] = {}
        keep = tuple([0] + [c for c in range(1, raw.shape[1]) if n_alive and raw[:, c].std() > 0.0])
        if n_alive >= MIN_REGRESSION_PATHS:
            raw = raw[:, list(keep)]
            mean = raw.mean(axis=0)
            scale = np.where(raw.std(axis=0) > 0, raw.std(axis=0), 1.0)
            z = (raw - mean) / scale
            lo, hi = np.quantile(z, CLIP_QUANTILE, axis=0), np.quantile(
                z, 1 - CLIP_QUANTILE, axis=0
            )
            z = np.clip(z, lo, hi)
            qs = np.linspace(0.0, 1.0, self.n_knots + 2)[1:-1]
            knots = np.unique(np.quantile(z[:, 0], qs)) if self.n_knots > 0 else np.zeros(0)
            x = hedge_basis(z, knots, self.degree)
            xtx = x.T @ x
            diag = np.diag(xtx).copy()
            pen = self.ridge * diag
            pen[diag == 0.0] = 1.0  # an all-zero column (t = 0, a constant state) gets beta = 0
            pen[0] = 0.0
            xtx = xtx + np.diag(pen)
            for kind, y_all in targets.items():
                y = y_all[alive]
                try:
                    beta = np.linalg.solve(xtx, x.T @ y)
                except np.linalg.LinAlgError:
                    beta = np.linalg.lstsq(x, y, rcond=None)[0]
                resid = y - x @ beta
                coefficients[kind] = np.asarray(beta)
                var_y = float(y.var())
                r2[kind] = 1.0 - float(resid.var()) / var_y if var_y > 0 else 1.0
            fit = Fit(
                float(t),
                self.degree,
                mean,
                scale,
                coefficients,
                r2,
                n_alive,
                alive.size,
                False,
                knots,
                state_constant,
                tuple(moving),
                keep,
                lo,
                hi,
            )
        else:
            # too few alive paths: constant fits at the alive means (or 0), noted
            mean = np.zeros(len(keep))
            scale = np.ones(len(keep))
            knots = np.zeros(0)
            nb = hedge_basis(np.zeros((1, len(keep))), knots, self.degree).shape[1]
            for kind, y_all in targets.items():
                beta = np.zeros(nb)
                beta[0] = float(y_all[alive].mean()) if n_alive else 0.0
                coefficients[kind] = beta
                r2[kind] = float("nan")
            fit = Fit(
                float(t),
                self.degree,
                mean,
                scale,
                coefficients,
                r2,
                n_alive,
                alive.size,
                True,
                knots,
                state_constant,
                tuple(moving),
                keep,
            )
            note = (
                f"{type(self.objects[obj_index]).__name__} at t={t:g}: {n_alive} alive paths "
                f"(< {MIN_REGRESSION_PATHS}), constant fit"
            )
            if note not in self.notes:
                self.notes.append(note)
        self.fits[key] = fit
        return fit

    # -- evaluation --------------------------------------------------------------------------

    def delta_method(self, obj_index: int, t: float) -> str:
        """``"crn"`` under the hybrid-CRN estimator (the default).  Under the ``"gradient"``
        estimator: ``"crn"`` when the object's state columns are constant across the pricing
        paths at ``t`` (a state-free object, ``t = 0``, a period product at a fixing) — there the
        plain CRN bump and the state derivative agree —, ``"gradient"`` otherwise."""
        if self.delta_estimator == "hybrid_crn":
            return "crn"
        return "crn" if self.fit(obj_index, t).state_constant else "gradient"

    def evaluate(
        self, obj_index: int, t: float, paths: PathSet, kinds: Sequence[str]
    ) -> tuple[dict[str, FloatArray], HedgeState]:
        """Conditional quantities of one object at ``t`` on ``paths`` (world or pricing):
        ``value`` (time-0 money), ``delta`` / ``gamma`` (per unit spot at ``t``; CRN or gradient by
        :meth:`delta_method`), the bump names, and ``dX<i>`` (the value gradient in the pricing
        factors).  Terminated paths carry the settled value and zero sensitivities."""
        fit = self.fit(obj_index, t)
        feats, hs = self.features(obj_index, paths, t)
        out: dict[str, FloatArray] = {}
        nf = self.model.n_factors
        method = self.delta_method(obj_index, t)
        s_t = np.exp(feats[:, 0])
        for kind in kinds:
            if kind.startswith("dX"):
                i = int(kind[2:])
                if i < 1 or i > nf:
                    raise ValueError(f"{kind}: the pricing model has {nf} factors")
                vals = fit.gradient("value", feats, i)
            elif kind in ("delta", "gamma") and method == "gradient":
                v_l, v_ll = fit.directional("value", feats)
                vals = v_l / s_t if kind == "delta" else (v_ll - v_l) / (s_t * s_t)
            elif kind == "delta":
                vals = fit.predict("delta", feats) / s_t
            elif kind == "gamma":
                vals = (fit.predict("gamma", feats) - fit.predict("delta", feats)) / (s_t * s_t)
            else:
                vals = fit.predict(kind, feats)
            if hs.any_terminated:
                dead = ~hs.alive
                vals = vals.copy()
                vals[dead] = hs.settled[dead] if kind == "value" else 0.0
            out[kind] = vals
        return out, hs

    def value_at_zero(self, obj_index: int) -> tuple[float, float]:
        """The ``t = 0`` value (mean of the discounted payoff on the pricing paths) and its
        standard error — the Monte Carlo price on these draws."""
        y = self.payoffs[obj_index].base
        if self.sim.antithetic:
            y = 0.5 * (y[0::2] + y[1::2])
        return float(y.mean()), float(y.std(ddof=1) / np.sqrt(y.size))


def union_grid(
    models: Sequence[Model], products: Sequence[Product], dates: FloatArray, sim: SimConfig
) -> TimeGrid:
    """The grid of the hedge: every product fixing, every rebalancing date and the models'
    required times, recording all steps when a product needs them."""
    fixings = np.unique(
        np.concatenate([np.asarray(dates, dtype=np.float64), *[p.fixing_times for p in products]])
    )
    fixings = fixings[fixings >= 0.0]
    cal = np.unique(np.concatenate([m.required_times() for m in models]))
    record_all = sim.record_all_steps or any(p.requires_all_steps for p in products)
    return TimeGrid.build(fixings, sim.dt_max, calibration_grid=cal, record_all_steps=record_all)


__all__ = [
    "BUMP_KINDS",
    "DEFAULT_KNOTS",
    "MIN_REGRESSION_PATHS",
    "Bump",
    "ConditionalPricer",
    "Fit",
    "ObjectPayoffs",
    "hedge_basis",
    "union_grid",
]
