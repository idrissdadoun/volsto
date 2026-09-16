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
re-anchored at the base spot at ``t`` and spliced onto the base history — for a **spot-kind**
bump (the §7.2 delta regimes, :meth:`~volsto.hedging.hedger.PricingContext.regime_delta_bump`)
at the base spot times the bump's own spot ratio ``e^{±h}``, so the regime delta measures the
bumped model's future from the bumped spot (re-anchoring both sides at the base spot removed the
spot move from the hybrid and drove the regime-delta target to ~0 — the bug fixed at the M8
acceptance).  The fitted-gradient
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

**Memory.**  Every bumped path set is kept for the hybrid targets at every date (``memory_bytes``:
≈ ``(3 + bump sides) × 5 arrays × n_paths × n_cols × 8`` bytes, 1.5 GB at 2·10⁴ paths, 260
columns and one two-sided bump).  ``stream_bumps=True`` (owner decision (d) at the M8 acceptance)
trades memory for disk: each bumped set is written per array as ``.npy`` files into a fresh
temporary directory (``tempfile.mkdtemp(prefix="volsto-bumps-")`` under ``scratch_dir``, else the
environment variable :data:`SCRATCH_ENV`, else the system temp) as soon as it is concatenated and
not kept; :meth:`ConditionalPricer.hybrid_payoffs` loads it with ``np.load(mmap_mode="r")``,
computes the hybrid payoffs and drops it.  ``memory_bytes`` then counts the resident arrays only
and ``streamed_bytes`` the on-disk footprint; :meth:`ConditionalPricer.close` (and ``__del__``,
best effort) removes the directory.  The two modes give identical results (the same arrays are
read back; ``test_stream_bumps_matches_in_memory``).

**Control variate on the difference (§7.11; owner decision (b) at the M8 acceptance).**  The
model-bump targets of the objects with a Black–Scholes proxy (:mod:`volsto.hedging.controls`:
vanillas, digitals, forward starts before ``T1`` and portfolios of those) are controlled with the
same payoff difference under a Black–Scholes **shadow** on the same spot normals (one per proxy
vol, the pricing model's forward curve; the shadow's future spliced at ``t`` like the hybrid
targets): ``y_i − β (c_i − E[c_i | state_i])`` with ``β = Cov(y, c − E[c|state])/Var(c −
E[c|state])`` on the alive paths of the date (the centred control's coefficient, so the
reduction is ``≥ 1`` by construction — :meth:`ConditionalPricer.controlled_target`, fixed
2026-09-16) and the conditional expectation the analytic Black value difference (exact for the
shadow, so the regression stays unbiased whatever ``β``).  Applies to the surface-driven bumps
(:class:`Bump` carries the bumped surfaces or the flat vol shifts) and — ``control_delta``, the
owner's decision of 2026-09-16, **off by default** (:data:`DEFAULT_CONTROL_DELTA`: measured to
hurt the 2F study-D hedges) — to the **hybrid-CRN delta target** and a §7.2 regime's delta
(:meth:`ConditionalPricer.delta_control`: the shadow payoff difference under ``S_t e^{±h}``,
whose conditional expectation is the Black finite-difference delta at the leg's proxy vol; the
regime's shadow reads the proxy vols of its moved surfaces); the gamma target and the parameter
bumps are untouched.  ``Fit.variance_reduction`` / ``Fit.beta`` report it per bump (the delta
control under ``"delta"``), ``control_variate=False`` switches every control off,
``control_delta=False`` the delta control only, and an object without a proxy is noted
once per class (cliquets, autocalls, barriers, variance products, accumulated-sum options).

Checked by ``tests/test_hedging.py`` (``test_conditional_pricer_bs``: the regressed value and
delta of a vanilla against Black–Scholes along the paths; the t = 0 value equals the Monte Carlo
price on the same draws; ``test_regime_delta_bump_reanchors_at_the_bumped_spot``: the
sticky-strike regime delta at ``t = 0`` equals the M5 ``delta_gamma`` estimator;
``test_control_variate_vega_vanilla_lv`` and
``test_control_variate_forward_risk_reversal_skew_tent``: the controlled and raw targets agree
in expectation and the variance reductions are reported).
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volsto.config import SimConfig
from volsto.engine.grid import TimeGrid
from volsto.engine.paths import PathSet
from volsto.engine.rng import GaussianDraws
from volsto.hedging.controls import (
    MIN_SHADOW_VOL,
    ProxyLeg,
    ShadowBrownian,
    proxy_legs,
    proxy_note,
)
from volsto.hedging.state import HedgeState, hedge_state
from volsto.models.base import Model
from volsto.products.base import Product
from volsto.risk.engine import model_regime_spot_bump

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
#: environment variable naming the scratch directory of streamed bump sets (``stream_bumps``);
#: an explicit ``scratch_dir`` wins, the system temp is the fallback
SCRATCH_ENV = "VOLSTO_SCRATCH"
#: prefix of the per-pricer scratch directories (``tempfile.mkdtemp``), so a crashed run's
#: leftovers are recognisable
STREAM_PREFIX = "volsto-bumps-"
#: the five arrays of a :class:`~volsto.engine.paths.PathSet` written per streamed set
_PATH_ARRAYS = ("log_spot", "variance", "factors", "int_var", "sum_sq")


@dataclass(frozen=True)
class Bump:
    """A bumped pricing model for one target: ``value(up) − value(dn or base)`` over ``unit``.
    ``kind`` names the sensitivity (``"vega"``, ``"fwd_var:0.25-0.5"``, ``"skew_T:1"``, …).

    A **surface-driven** bump carries what the §7.11 control variate needs to shadow it under
    Black–Scholes (:mod:`volsto.hedging.controls`): the perturbed states' implied surfaces
    ``up_surface`` / ``dn_surface`` (vega / volga parallel, forward-variance buckets, the
    ``skew_T`` / ``curvature_T`` tents — the proxy vol of a leg moves by the bumped surface's
    vol minus the base surface's at the leg's ``(k, T)``), or, for a bare Black–Scholes pricing
    model whose vega bump is the flat ``±VOL_BUMP``, the flat shifts ``vol_shift_up`` /
    ``vol_shift_dn``.  A model-parameter bump (``param:<name>``) carries neither and gets no
    control.  A **spot-kind** bump (a §7.2 delta regime) may carry the moved states' surfaces
    ``up_surface`` / ``dn_surface`` (both): the delta control
    (:meth:`ConditionalPricer.delta_control`) then shadows it at the proxy vol the regime's
    surface move gives the leg; without them it is controlled at the base proxy vol."""

    name: str
    up: Model
    dn: Model | None = None
    unit: float = 1.0
    description: str = ""
    kind: str = "model"
    up_surface: Any = None
    dn_surface: Any = None
    vol_shift_up: float | None = None
    vol_shift_dn: float | None = None

    def __post_init__(self) -> None:
        if self.unit == 0 or not np.isfinite(self.unit):
            raise ValueError("unit must be a finite non-zero number")
        if self.kind not in BUMP_KINDS:
            raise ValueError(f"kind must be one of {BUMP_KINDS}")
        if self.kind in ("second", "spot") and self.dn is None:
            raise ValueError(f"a {self.kind!r} bump needs both up and dn models")
        if self.up_surface is not None and self.vol_shift_up is not None:
            raise ValueError("a bump carries either surfaces or flat vol shifts, not both")
        if self.dn is None and (self.dn_surface is not None or self.vol_shift_dn is not None):
            raise ValueError("a one-sided bump cannot carry a dn surface or vol shift")
        if self.dn is not None and self.controllable and not self._has_dn_control:
            raise ValueError("a two-sided controllable bump needs the dn surface or vol shift too")
        if self.kind == "spot" and (self.vol_shift_up is not None or self.vol_shift_dn is not None):
            raise ValueError("a spot-kind bump carries the moved surfaces, not flat vol shifts")
        if self.kind == "spot" and (self.up_surface is None) != (self.dn_surface is None):
            raise ValueError("a spot-kind bump carries both moved surfaces or neither")

    @property
    def _has_dn_control(self) -> bool:
        return self.dn_surface is not None or self.vol_shift_dn is not None

    @property
    def controllable(self) -> bool:
        """Whether the bump carries a Black–Scholes shadow (surfaces or flat vol shifts) — a
        ``"model"`` or ``"second"`` kind; spot-kind bumps are controlled by the delta control
        instead (:meth:`ConditionalPricer.delta_control`, ``control_delta``)."""
        return self.kind in ("model", "second") and (
            self.up_surface is not None or self.vol_shift_up is not None
        )


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
    and the fit diagnostics.  The §7.11 control variate's report per bump name (the delta
    control's under ``"delta"``):
    ``variance_reduction`` (``Var(y) / Var(y − β (c − E[c | state]))`` on the alive paths; 1.0
    when the bump was not controlled), ``beta`` (0.0 when not controlled) and the ``controlled``
    names."""

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
    variance_reduction: dict[str, float] = field(default_factory=dict)
    variance_reduction_se: dict[str, float] = field(default_factory=dict)
    beta: dict[str, float] = field(default_factory=dict)
    controlled: tuple[str, ...] = ()

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
#: path-bootstrap draws (and seed) for the standard error of a control's variance reduction
CV_BOOTSTRAP_DRAWS = 64
CV_BOOTSTRAP_SEED = 7
CLIP_QUANTILE = 0.001  # the basis is evaluated inside the pricing paths' [0.1%, 99.9%] range
# ridge on the normal equations, relative to each column's own diagonal entry (scale-free), the
# intercept unpenalised (so the t = 0 fit is exactly the sample mean and the product leg
# telescopes to payoff - V0 to round-off): a numerical stabiliser for near-collinear columns,
# not a smoother (a ridge of 1e-3 x n_paths on the raw columns crushed the last spline pieces
# and took the daily Black-Scholes delta hedge from 1.09x to 1.46x the exact-delta bound)
DEFAULT_RIDGE = 1e-6
#: default of ``ConditionalPricer.control_delta`` / ``Hedger.control_delta``: the §7.11 control on
#: the hybrid-CRN delta target (and a §7.2 regime's delta) is OFF by default — the owner's rule of
#: 2026-09-16 is "on by default only if it is measured to help and never to hurt", and it was
#: measured to hurt.  M8b study-D vanilla (2F marking LSV, 2·10⁴ pricing and world paths, daily,
#: the same world paths on and off; hedged std on/off with a path-pair bootstrap se): model
#: 1.0019 ± 0.0007 and sticky_moneyness 1.0038 ± 0.0007 (worse), sticky_strike 0.9984 ± 0.0009,
#: sticky_skew 0.9987 ± 0.0008, min_variance 0.9952 ± 0.0017 — although the delta target's own
#: variance falls by a median 1.63 (inter-quartile 1.34–2.03) over the model row's dates.  Under
#: Black–Scholes it helps: the daily 1y ATM call's hedged std over the analytic-delta hedge goes
#: from 1.133 ± 0.006 to 1.074 ± 0.005 (``scripts/m8b_delta_estimator.py``).  Re-measured with the
#: centred-control coefficient (same budget and paths, 337 s, no calibration): model 1.0018 ±
#: 0.0007 and sticky_moneyness 1.0033 ± 0.0006 (still worse), sticky_strike 0.9979 ± 0.0008,
#: sticky_skew 0.9982 ± 0.0007, min_variance 0.9954 ± 0.0016 — the default stays OFF
DEFAULT_CONTROL_DELTA = False
#: the name of the controlled value regression (:meth:`ConditionalPricer.value_control`): never
#: used for the P&L (the ``value`` target stays the raw payoff, so ``t = 0`` is the Monte Carlo
#: price on the draws), only for the factor gradients of the minimum-variance delta
VALUE_CV = "value_cv"


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
    stream_bumps: bool = False
    scratch_dir: str | Path | None = None
    #: the §7.11 control variate on the difference for the surface-driven model bumps of the
    #: objects with a Black–Scholes proxy (:mod:`volsto.hedging.controls`; module docstring)
    control_variate: bool = True
    #: the §7.11 control on the **delta** target (:meth:`delta_control`; needs
    #: ``control_variate`` and the hybrid-CRN estimator): off switches the delta target back to
    #: the raw CRN difference while the bump targets stay controlled
    control_delta: bool = DEFAULT_CONTROL_DELTA
    #: fit the controlled value target :data:`VALUE_CV` next to ``value`` (needs
    #: ``control_variate``; the hedger turns it on for the ``min_variance`` delta regime only,
    #: which reads its factor gradient from it — one extra shadow pass per object and date)
    control_value: bool = False
    #: the base implied surface the proxy vols are read from (the pricing context's; a
    #: Black–Scholes pricing model uses its own ``vol`` instead; ``None`` without a state)
    surface: Any = None
    paths: PathSet = field(init=False, repr=False)
    payoffs: list[ObjectPayoffs] = field(init=False, repr=False)
    #: the bumped sets kept in memory (every set unless ``stream_bumps``)
    bumped_paths: dict[str, PathSet] = field(init=False, repr=False, default_factory=dict)
    #: the bumped sets' keys in simulation order (``"up"``, ``"dn"``, ``"bump:<name>:<side>"``)
    bump_keys: tuple[str, ...] = field(init=False, default=())
    #: re-anchoring log-shift of each bumped set's future at the splice (module docstring)
    bump_shifts: dict[str, float] = field(init=False, repr=False, default_factory=dict)
    streamed: dict[str, dict[str, Path]] = field(init=False, repr=False, default_factory=dict)
    stream_path: Path | None = field(init=False, default=None)
    hybrids: dict[float, dict[str, FloatArray]] = field(
        init=False, repr=False, default_factory=dict
    )
    fits: dict[tuple[int, float], Fit] = field(init=False, repr=False, default_factory=dict)
    notes: list[str] = field(init=False, default_factory=list)
    n_simulations: int = field(init=False, default=0)
    #: set by :meth:`release`: the resident sets are gone, the footprint they had is kept
    released: bool = field(init=False, default=False)
    released_bytes: int = field(init=False, default=0)
    released_streamed_bytes: int = field(init=False, default=0)
    #: the spot Brownian of the pricing draws at the record columns (``None`` when no control
    #: applies: control off, no controllable bump or no object with a proxy)
    brownian: ShadowBrownian | None = field(init=False, repr=False, default=None)
    #: the Black–Scholes proxy legs per object index (``None``: no proxy)
    proxies: dict[int, tuple[ProxyLeg, ...] | None] = field(
        init=False, repr=False, default_factory=dict
    )
    #: per ``(object, spot-bump name)``: the legs' shadow vols ``(up, dn)`` of the delta control
    #: (``"delta"`` for the hybrid-CRN spot bumps, else a §7.2 regime bump's name)
    delta_vols: dict[tuple[int, str], tuple[FloatArray, FloatArray]] = field(
        init=False, repr=False, default_factory=dict
    )
    #: per object: the legs' base proxy vols (the value control's)
    base_vols: dict[int, FloatArray] = field(init=False, repr=False, default_factory=dict)
    #: per ``(object, bump name)``: the legs' shadow vols ``(base, up, dn)`` (``dn`` ``None`` for
    #: a one-sided bump); ``None`` when the bump cannot be shadowed for that object
    shadow_vols: dict[tuple[int, str], tuple[FloatArray, FloatArray, FloatArray | None] | None] = (
        field(init=False, repr=False, default_factory=dict)
    )

    def __post_init__(self) -> None:
        if not self.objects:
            raise ValueError("no objects to price")
        names = [b.name for b in self.bumps]
        if len(set(names)) != len(names):
            raise ValueError("bump names must be unique")
        if self.delta_estimator not in DELTA_ESTIMATORS:
            raise ValueError(f"delta_estimator must be one of {DELTA_ESTIMATORS}")
        self._simulate()
        self._prepare_controls()

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
        # the "model" regime of every model class (LocalVol.bump(spot=) would hold the local vol
        # in k, the sticky-local-vol move: volsto.risk.engine.model_regime_spot_bump)
        up = model_regime_spot_bump(self.model, s0 * float(np.exp(self.spot_size)))
        dn = model_regime_spot_bump(self.model, s0 * float(np.exp(-self.spot_size)))
        models: list[tuple[str, Model]] = [("base", self.model), ("up", up), ("dn", dn)]
        # the re-anchoring shift of each bumped set's future at the splice: the CRN spot bumps
        # move the spot by ±h; a spot-kind bump (a delta regime) moves it by its own spot ratio
        # — re-anchored at the base spot the spot move would be removed from the hybrid and the
        # regime-delta target driven to ~0 —; a model bump leaves the spot where it is
        shifts: dict[str, float] = {"up": self.spot_size, "dn": -self.spot_size}
        for b in self.bumps:
            models.append((f"bump:{b.name}:up", b.up))
            if b.kind == "spot":
                shifts[f"bump:{b.name}:up"] = float(np.log(b.up.spot / s0))
            if b.dn is not None:
                models.append((f"bump:{b.name}:dn", b.dn))
                if b.kind == "spot":
                    shifts[f"bump:{b.name}:dn"] = float(np.log(b.dn.spot / s0))
        self.bump_shifts = shifts
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
        # every bumped path set is needed for the hybrid (history-held) targets at every date:
        # kept in memory (memory_bytes reports the footprint) or, with stream_bumps, written to
        # the scratch directory as soon as it is concatenated and memory-mapped back on demand
        self.bump_keys = tuple(parts)
        self.bumped_paths = {}
        self.streamed = {}
        if self.stream_bumps:
            base_dir = self.scratch_dir or os.environ.get(SCRATCH_ENV) or None
            self.stream_path = Path(tempfile.mkdtemp(prefix=STREAM_PREFIX, dir=base_dir))
        for i, k in enumerate(list(parts)):
            ps = PathSet.concat(parts.pop(k))
            if self.stream_path is None:
                self.bumped_paths[k] = ps
                continue
            files: dict[str, Path] = {}
            for arr in _PATH_ARRAYS:
                f = self.stream_path / f"{i:03d}_{arr}.npy"
                np.save(f, getattr(ps, arr))
                files[arr] = f
            self.streamed[k] = files
            del ps
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
        for name in self.bump_keys:
            if name.startswith("bump:") and not self.hybrid_bumps:
                continue
            bumped = self.bumped_set(name)
            hyb = self._hybrid(bumped, col, self.bump_shifts.get(name, 0.0))
            del bumped  # a streamed set is dropped as soon as the hybrid is built
            pay = np.empty((self.sim.n_paths, n_obj))
            for j, obj in enumerate(self.objects):
                pay[:, j] = obj.payoff(hyb, self.idx)
            out[name] = pay
        while len(self.hybrids) >= 3:
            del self.hybrids[next(iter(self.hybrids))]
        self.hybrids[key] = out
        return out

    def bumped_set(self, name: str) -> PathSet:
        """The bumped path set ``name``: the resident one, or a :class:`PathSet` on read-only
        memory maps of the streamed ``.npy`` files (``stream_bumps``)."""
        if name in self.bumped_paths:
            return self.bumped_paths[name]
        if name not in self.streamed:
            raise KeyError(f"no bumped path set {name!r}")
        files = self.streamed[name]
        arrays = {arr: np.load(files[arr], mmap_mode="r") for arr in _PATH_ARRAYS}
        return PathSet(self.paths.times, *(arrays[arr] for arr in _PATH_ARRAYS))

    @staticmethod
    def _set_bytes(p: PathSet) -> int:
        return int(
            p.log_spot.nbytes
            + p.variance.nbytes
            + p.factors.nbytes
            + p.int_var.nbytes
            + p.sum_sq.nbytes
        )

    @property
    def memory_bytes(self) -> int:
        """Footprint of the resident path sets (the base set and every bumped set kept in
        memory; a streamed set counts in :attr:`streamed_bytes` instead); after
        :meth:`release`, the footprint the pricer had."""
        if self.released:
            return self.released_bytes
        return sum(self._set_bytes(p) for p in [self.paths, *self.bumped_paths.values()])

    @property
    def streamed_bytes(self) -> int:
        """On-disk footprint of the streamed bumped sets (0 unless ``stream_bumps``); after
        :meth:`release`, the footprint they had."""
        if self.released:
            return self.released_streamed_bytes
        total = 0
        for files in self.streamed.values():
            for f in files.values():
                with contextlib.suppress(OSError):
                    total += f.stat().st_size
        return total

    def close(self) -> None:
        """Remove the scratch directory of the streamed sets (idempotent; the pricer cannot
        build new hybrids afterwards — its cached fits stay usable)."""
        d = self.stream_path
        self.stream_path = None
        self.streamed = {}
        if d is not None:
            shutil.rmtree(d, ignore_errors=True)

    def release(self) -> int:
        """Free the memory of a pricer the hedge loop has **replaced** (a recalibration rebuilt
        the pricing model and its pricer): the base and bumped path sets, the hybrid payoff
        cache and the shadow Brownian draws are dropped and the
        streamed sets removed (:meth:`close`).  What the end of a run still reads stays: the
        fits (coefficients, control-variate reductions), the payoffs (:meth:`value_at_zero`)
        and, through :attr:`memory_bytes` / :attr:`streamed_bytes`, the footprint the pricer
        had.  Returns the bytes released.  Idempotent; the pricer cannot regress a new date
        afterwards.  Measured need: the study-C autocall run (10 objects, 6 path sets of
        2·10⁴ paths, 5.2 GB per pricer) grew by one full pricer per refit — 26 GB resident on
        a 24 GB laptop after two refits."""
        if self.released:
            return 0
        n = self.memory_bytes
        self.released_bytes = n
        self.released_streamed_bytes = self.streamed_bytes
        self.released = True
        self.bumped_paths = {}
        self.hybrids = {}
        self.shadow_vols = {}
        self.delta_vols = {}
        self.base_vols = {}
        self.brownian = None
        del self.paths
        self.close()
        return n

    def __del__(self) -> None:  # best effort: a failed __post_init__ has no stream_path
        try:
            if getattr(self, "stream_path", None) is not None:
                self.close()
        except Exception:  # never raise from a finaliser
            pass

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

    # -- control variate (§7.11) -------------------------------------------------------------

    def _prepare_controls(self) -> None:
        """Register the objects' Black–Scholes proxies and build the shadow Brownian when at
        least one controllable bump meets at least one object with a proxy (module docstring;
        :mod:`volsto.hedging.controls`); one note per object class without a proxy."""
        self.proxies = {}
        self.shadow_vols = {}
        self.delta_vols = {}
        self.base_vols = {}
        self.brownian = None
        spot_bumps = [b for b in self.bumps if b.kind == "spot"]
        delta_on = bool(self.delta_controlled)
        value_on = bool(self.control_variate and self.control_value)
        if not self.control_variate or not (
            any(b.controllable for b in self.bumps) or delta_on or value_on
        ):
            return
        base_vol = float(self.model.vol) if hasattr(self.model, "vol") else None
        if base_vol is None and self.surface is None:
            note = "no Black-Scholes control: the pricer has no base surface (raw bump targets)"
            if note not in self.notes:
                self.notes.append(note)
            return
        any_proxy = False
        for j, obj in enumerate(self.objects):
            legs = proxy_legs(obj)
            self.proxies[j] = legs
            if legs is None:
                note = proxy_note(obj)
                if note not in self.notes:
                    self.notes.append(note)
                continue
            any_proxy = True
            for b in self.bumps:
                if b.controllable:
                    self.shadow_vols[(j, b.name)] = self._shadow_vols(legs, b, base_vol)
            base = self._base_vols(legs, base_vol)
            self.base_vols[j] = base
            if delta_on:
                # the hybrid-CRN spot bumps move the spot with the model held: the base vols
                self.delta_vols[(j, "delta")] = (base, base)
                for b in spot_bumps:
                    self.delta_vols[(j, b.name)] = (
                        self._moved_vols(legs, base, b.up_surface),
                        self._moved_vols(legs, base, b.dn_surface),
                    )
        if not any_proxy:
            return
        self.brownian = ShadowBrownian.from_draws(
            self.sim.seed if self.seed is None else self.seed,
            self.sim.n_paths,
            self.grid,
            self.sim.antithetic,
            self.sim.chunk_ranges(self.grid.n_records, self.model.n_factors),
        )

    def _shadow_vols(
        self, legs: Sequence[ProxyLeg], b: Bump, base_vol: float | None
    ) -> tuple[FloatArray, FloatArray, FloatArray | None] | None:
        """The legs' shadow vols under bump ``b``: base (the pricing model's own flat vol, else
        the base surface's vol at the leg's ``(k, T)``), up and dn (base + the bumped surface's
        vol minus the base surface's, or + the flat shift), floored at ``MIN_SHADOW_VOL``."""
        base = self._base_vols(legs, base_vol)

        def side(surface: Any, shift: float | None) -> FloatArray | None:
            if shift is not None:
                return np.asarray(np.maximum(base + shift, MIN_SHADOW_VOL))
            if surface is None or self.surface is None:
                return None
            delta = np.array([leg.vol(surface) - leg.vol(self.surface) for leg in legs])
            return np.asarray(np.maximum(base + delta, MIN_SHADOW_VOL))

        up = side(b.up_surface, b.vol_shift_up)
        if up is None:
            return None
        dn = side(b.dn_surface, b.vol_shift_dn) if b.dn is not None else None
        if b.dn is not None and dn is None:
            return None
        return base, up, dn

    def _base_vols(self, legs: Sequence[ProxyLeg], base_vol: float | None) -> FloatArray:
        """The legs' base proxy vols: the pricing model's own flat vol, else the base surface's
        vol at each leg's ``(k, T)``."""
        if base_vol is not None:
            return np.full(len(legs), base_vol)
        return np.array([leg.vol(self.surface) for leg in legs])

    def _moved_vols(self, legs: Sequence[ProxyLeg], base: FloatArray, surface: Any) -> FloatArray:
        """The legs' proxy vols after a §7.2 regime's spot move: base + [moved surface vol −
        base surface vol] at each leg's fixed strike (the moved surface is in the moved spot's
        coordinates, so a fixed strike reads the regime's vol there), floored at
        ``MIN_SHADOW_VOL``; the base vols when the bump carries no surface (a flat move: the
        model regime)."""
        if surface is None or self.surface is None:
            return base
        delta = np.array([leg.vol(surface) - leg.vol(self.surface) for leg in legs])
        return np.asarray(np.maximum(base + delta, MIN_SHADOW_VOL))

    @property
    def delta_controlled(self) -> bool:
        """Whether the delta control applies to this pricer's delta targets (``control_variate``
        and ``control_delta`` on, the hybrid-CRN estimator: the plain bumped path of the
        ``"gradient"`` estimator scales the whole history, which a shadow spliced at ``t`` does not
        follow)."""
        return bool(
            self.control_variate and self.control_delta and self.delta_estimator == "hybrid_crn"
        )

    def delta_control(
        self, obj_index: int, t: float, name: str, s_up: float, s_dn: float
    ) -> tuple[FloatArray, FloatArray] | None:
        """The per-path control of a delta target (the hybrid-CRN ``"delta"``, or the spot-kind
        bump ``name``) of one object at ``t`` and its analytic conditional expectation, scaled
        exactly like the target: the legs' shadow payoffs spliced at ``t`` from ``S_t e^{±h}`` —
        ``h`` the bump's own re-anchoring shift ``ln(s_up/S₀)`` / ``ln(s_dn/S₀)`` — at the up /
        down proxy vols, differenced over ``s_up − s_dn`` and converted to the dollar delta in
        ``ln S_t`` (``× S₀``, the target's ``/ (S_t/S₀) × S_t``).  The conditional expectation is
        the Black finite difference ``[V_BS(S_t e^{h_up}) − V_BS(S_t e^{h_dn})] × S₀/(s_up −
        s_dn)`` — exact for the shadow (the Black delta ``DF(T) N(d₁) F(T)/F(t)`` up to
        ``O(h²)``), so the regression stays unbiased whatever ``β``.  Under a Black–Scholes
        pricing model the shadow is the model and the controlled target is that finite
        difference to round-off.  ``None`` when no control applies (no proxy, a forward-start
        leg past its ``T1``: noted)."""
        legs = self.proxies.get(obj_index)
        vols = self.delta_vols.get((obj_index, name))
        if legs is None or vols is None or self.brownian is None:
            return None
        if not all(leg.active(float(t)) for leg in legs):
            note = proxy_note(self.objects[obj_index], float(t))
            if note not in self.notes:
                self.notes.append(note)
            return None
        col = self.idx[float(t)]
        fc = self.model.forward_curve
        ln_s_t = self.paths.log_spot_at(col)
        s0 = self.model.spot
        n = self.sim.n_paths

        def side(v: FloatArray, shift: float) -> tuple[FloatArray, FloatArray]:
            c = np.zeros(n)
            e = np.zeros(n)
            assert self.brownian is not None and legs is not None
            for leg, vol in zip(legs, v, strict=True):
                c += self.brownian.leg_payoff(leg, self.paths, self.idx, col, float(vol), fc, shift)
                e += leg.weight * leg.conditional_value(ln_s_t + shift, float(t), float(vol), fc)
            return c, e

        c_up, e_up = side(vols[0], float(np.log(s_up / s0)))
        c_dn, e_dn = side(vols[1], float(np.log(s_dn / s0)))
        k = s0 / (s_up - s_dn)
        return (c_up - c_dn) * k, (e_up - e_dn) * k

    def value_control(self, obj_index: int, t: float) -> tuple[FloatArray, FloatArray] | None:
        """The per-path control of the **value** target of one object at ``t`` — the legs'
        shadow payoffs spliced at ``t`` at the base proxy vols — and its analytic conditional
        expectation (the Black value given ``S_t``), or ``None`` (no proxy, a forward start past
        its ``T1``).  Its expectation depends on ``S_t`` only, so the controlled value target
        keeps its factor dependence while shedding the spot-driven payoff noise: the regression
        gradient in the factors (:data:`VALUE_CV`, the minimum-variance delta's ``∂V/∂X_i``) is
        read from it."""
        legs = self.proxies.get(obj_index)
        base = self.base_vols.get(obj_index)
        if legs is None or base is None or self.brownian is None:
            return None
        if not all(leg.active(float(t)) for leg in legs):
            return None
        col = self.idx[float(t)]
        fc = self.model.forward_curve
        ln_s_t = self.paths.log_spot_at(col)
        c = np.zeros(self.sim.n_paths)
        e = np.zeros(self.sim.n_paths)
        for leg, vol in zip(legs, base, strict=True):
            c += self.brownian.leg_payoff(leg, self.paths, self.idx, col, float(vol), fc)
            e += leg.weight * leg.conditional_value(ln_s_t, float(t), float(vol), fc)
        return c, e

    def control(self, obj_index: int, t: float, b: Bump) -> tuple[FloatArray, FloatArray] | None:
        """The per-path control ``c`` of bump ``b`` for one object at ``t`` and its analytic
        conditional expectation ``E[c | state]`` (``(n_paths,)`` each), scaled exactly like the
        target — ``(u − d)/(2 unit)``, ``(u − base)/unit`` or ``(u − 2 base + d)/unit²`` of the
        legs' shadow payoffs spliced at ``t`` —, or ``None`` when no control applies (no proxy,
        the bump not controllable, a forward-start leg past its ``T1``: noted)."""
        legs = self.proxies.get(obj_index)
        vols = self.shadow_vols.get((obj_index, b.name))
        if legs is None or vols is None or self.brownian is None:
            return None
        if not all(leg.active(float(t)) for leg in legs):
            note = proxy_note(self.objects[obj_index], float(t))
            if note not in self.notes:
                self.notes.append(note)
            return None
        col = self.idx[float(t)]
        fc = self.model.forward_curve
        ln_s_t = self.paths.log_spot_at(col)
        base_v, up_v, dn_v = vols
        n = self.sim.n_paths

        def side(v: FloatArray) -> tuple[FloatArray, FloatArray]:
            c = np.zeros(n)
            e = np.zeros(n)
            assert self.brownian is not None
            for leg, vol in zip(legs, v, strict=True):
                c += self.brownian.leg_payoff(leg, self.paths, self.idx, col, float(vol), fc)
                e += leg.weight * leg.conditional_value(ln_s_t, float(t), float(vol), fc)
            return c, e

        c_up, e_up = side(up_v)
        if b.kind == "second":
            assert dn_v is not None
            c_dn, e_dn = side(dn_v)
            c_0, e_0 = side(base_v)
            u2 = b.unit * b.unit
            return (c_up - 2.0 * c_0 + c_dn) / u2, (e_up - 2.0 * e_0 + e_dn) / u2
        if dn_v is not None:
            c_dn, e_dn = side(dn_v)
            return (c_up - c_dn) / (2.0 * b.unit), (e_up - e_dn) / (2.0 * b.unit)
        c_0, e_0 = side(base_v)
        return (c_up - c_0) / b.unit, (e_up - e_0) / b.unit

    @staticmethod
    def controlled_target(
        y: FloatArray, c: FloatArray, e: FloatArray, alive: BoolArray
    ) -> tuple[FloatArray, float, float, float]:
        """``y − β (c − e)`` with ``β = Cov(y, c − e)/Var(c − e)`` on the alive paths, the
        variance reduction ``Var(y)/Var(y − β (c − e))`` there (``inf`` when the controlled target
        is constant, 1.0 when ``y`` is), ``β``, and the bootstrap standard error of the reduction
        (:data:`CV_BOOTSTRAP_DRAWS` path resamples; 0 when there is nothing to estimate).

        The coefficient is the one of the **centred** control ``c − e`` (``e = E[c | state]``
        varies with the state): with it ``Var(y − β (c − e)) = Var(y) − Cov(y, c − e)²/Var(c − e)
        ≤ Var(y)`` on the alive paths by construction (the sample identity, same ``ddof``), so the
        reduction is ``≥ 1`` up to rounding.  The coefficient ``Cov(y, c)/Var(c)`` used before
        2026-09-16 minimises ``Var(y − β c)``, not the variance of the target actually returned,
        and can make it larger (``tests/test_hedging.py::
        test_controlled_target_coefficient_never_increases_the_variance``)."""
        ya = y[alive]
        za = np.asarray(c[alive] - e[alive], dtype=np.float64)
        var_z = float(za.var(ddof=1)) if za.size > 1 else 0.0
        beta = float(np.cov(ya, za, ddof=1)[0, 1] / var_z) if var_z > 0.0 else 0.0
        out = np.asarray(y - beta * (c - e), dtype=np.float64)
        var_y = float(ya.var(ddof=1)) if ya.size > 1 else 0.0
        ra = out[alive]
        var_r = float(ra.var(ddof=1)) if ya.size > 1 else 0.0
        ratio = 1.0 if var_y == 0.0 else (var_y / var_r if var_r > 0.0 else float("inf"))
        # standard error of the ratio of two sample variances: a path bootstrap (the ratio is
        # a Monte Carlo number and is never reported without one)
        se = 0.0
        if np.isfinite(ratio) and ratio != 1.0 and ya.size >= 8:
            rng = np.random.default_rng(CV_BOOTSTRAP_SEED)
            n = ya.size
            idx = rng.integers(0, n, size=(CV_BOOTSTRAP_DRAWS, n))
            vy = ya[idx].var(axis=1, ddof=1)
            vr = ra[idx].var(axis=1, ddof=1)
            good = vr > 0.0
            se = float((vy[good] / vr[good]).std(ddof=1)) if good.sum() > 1 else float("nan")
        return out, ratio, beta, se

    # -- regression ----------------------------------------------------------------------------

    def fit(self, obj_index: int, t: float) -> Fit:
        """The regressions of object ``obj_index`` at date ``t`` (cached)."""
        key = (obj_index, float(t))
        if key in self.fits:
            return self.fits[key]
        feats, hs = self.features(obj_index, self.paths, t)
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
        # the pure dollar gamma S_t² Γ: independent of the 'delta' key, which a delta-regime bump
        # (kind "spot", named "delta") replaces by the regime's dollar delta
        targets["gamma"] = g_raw / ratio**2 * s_t * s_t
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
                # the regime delta: per unit spot at t is (u - d) / (S_t (e^h - e^-h)) =
                # (u - d) / (S_up - S_dn) / ratio; regressed as the DOLLAR delta (x S_t) like the
                # hybrid-CRN delta above, since evaluate() divides every "delta" prediction by S_t
                assert d is not None
                targets[b.name] = (
                    (u - d) / (b.up.spot - b.dn.spot) / ratio * s_t  # type: ignore[union-attr]
                )
            else:
                targets[b.name] = (
                    (u - d) / (2.0 * b.unit) if d is not None else (u - pay.base) / b.unit
                )
        # the §7.11 control on the difference: the shadow Black-Scholes payoff difference on the
        # same normals, centred at its analytic conditional expectation, regressed out of the
        # bump target (variance reduction and beta reported per bump; 1.0 / 0.0 without one)
        variance_reduction: dict[str, float] = {}
        variance_reduction_se: dict[str, float] = {}
        betas: dict[str, float] = {}
        controlled: list[str] = []
        n_alive = int(alive.sum())
        # the delta target: the hybrid-CRN one, or the spot-kind bump that replaces it (a §7.2
        # regime: the same shadow from the regime's own spot ratio at its moved proxy vols)
        regime = next((b for b in self.bumps if b.kind == "spot" and b.name == "delta"), None)
        if self.delta_controlled and n_alive > 1:
            if regime is None:
                ce = self.delta_control(obj_index, float(t), "delta", self._s_up, self._s_dn)
            else:
                assert regime.dn is not None
                ce = self.delta_control(
                    obj_index, float(t), regime.name, regime.up.spot, regime.dn.spot
                )
            if ce is not None:
                (
                    targets["delta"],
                    variance_reduction["delta"],
                    betas["delta"],
                    variance_reduction_se["delta"],
                ) = self.controlled_target(targets["delta"], ce[0], ce[1], alive)
                controlled.append("delta")
        if self.control_variate and self.control_value and n_alive > 1:
            cv = self.value_control(obj_index, float(t))
            if cv is not None:
                (
                    targets[VALUE_CV],
                    variance_reduction[VALUE_CV],
                    betas[VALUE_CV],
                    variance_reduction_se[VALUE_CV],
                ) = self.controlled_target(pay.base, cv[0], cv[1], alive)
                controlled.append(VALUE_CV)
        for b in self.bumps:
            if b.kind == "spot" and b.name == "delta":
                variance_reduction.setdefault(b.name, 1.0)
                variance_reduction_se.setdefault(b.name, 0.0)
                betas.setdefault(b.name, 0.0)
                continue
            ce = self.control(obj_index, float(t), b) if n_alive > 1 else None
            if ce is None:
                variance_reduction[b.name], betas[b.name] = 1.0, 0.0
                variance_reduction_se[b.name] = 0.0
                continue
            (
                targets[b.name],
                variance_reduction[b.name],
                betas[b.name],
                variance_reduction_se[b.name],
            ) = self.controlled_target(targets[b.name], ce[0], ce[1], alive)
            controlled.append(b.name)
        raw = feats[alive]
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
        fit.variance_reduction = variance_reduction
        fit.variance_reduction_se = variance_reduction_se
        fit.beta = betas
        fit.controlled = tuple(controlled)
        self.fits[key] = fit
        return fit

    def cv_reductions(self, delta: bool = False) -> list[float]:
        """Every controlled bump's variance reduction over the fits computed so far (dates ×
        objects), for the run's ``cv_reduction_median``; ``delta=True``: the delta control's
        instead (``cv_delta_reduction_median``)."""
        return [
            f.variance_reduction[k]
            for f in self.fits.values()
            for k in f.controlled
            if k != VALUE_CV and (k == "delta") == delta
        ]

    def cv_reduction_ses(self, delta: bool = False) -> list[float]:
        """The bootstrap standard errors matching :meth:`cv_reductions`."""
        return [
            f.variance_reduction_se[k]
            for f in self.fits.values()
            for k in f.controlled
            if k != VALUE_CV and (k == "delta") == delta
        ]

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
                vals = fit.predict("gamma", feats) / (s_t * s_t)
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
    "SCRATCH_ENV",
    "STREAM_PREFIX",
    "Bump",
    "ConditionalPricer",
    "Fit",
    "ObjectPayoffs",
    "hedge_basis",
    "union_grid",
]
