"""PDE cross-check for the one-factor LSV (SPEC §9.1, M6 Part 4).

State ``(x = ln S, X)`` with ``θ = 0``.  The kernel's single OU factor ``dX = −k1 X dt + dW¹``
(``X_0 = 0``, unit diffusion; :mod:`volsto.models.bergomi` with ``α_0 = 1`` and ``x_t^t = X_t``)
drives the instantaneous variance ``V_t = ξ_t^t = g(t) e^{ω X_t}`` with
``g(t) = ξ_0^t exp(−½ ω² χ(t, t))``, ``χ(t, t) = (1 − e^{−2 k1 t})/(2 k1)`` and ``ω = 2ν`` —
exactly :meth:`~volsto.models.bergomi.BergomiSV.variance_from_factors` (the kernel's
``coef = ω α_θ w_1`` equals ``ω`` at ``θ = 0``; ``g`` and ``coef`` are taken from the kernel, not
re-derived).  The spot variance is ``L(t, S)² V_t`` with the calibrated leverage
(:class:`~volsto.models.leverage.LeverageFunction`: linear in ``t`` between slices, linear in
``k = ln S/F(t)``, flat outside), ``L ≡ 1`` for the pure SV kernel, and ``L² V ≡ σ_loc²(t, S)``
with ``ω = 0`` for local vol and Black–Scholes, where ``X`` drops out and the problem is
one-dimensional.  ``corr(dW^S, dW¹) = ρ = ρ_SX1``.

Backward Kolmogorov equation for the discounted value ``u(t, x, X)`` (owner's spec)::

    ∂_t u + (r − q − ½ L² V) ∂_x u − k1 X ∂_X u + ½ L² V ∂_xx u + ½ ∂_XX u
        + ρ L sqrt(V) ∂_xX u − r u = 0,          u(T, x, X) = payoff(e^x),

with ``r``, ``q`` the step-averaged instantaneous rates of the model's forward curve
(``∫ r dt / Δt`` over each time step, exact for the piecewise-flat curves).

Discretisation (in 't Hout & Foulon, *ADI finite difference schemes for option pricing in the
Heston model with correlation*, Int. J. Numer. Anal. Model. 7 (2010) 303–320):

* **Grids.** ``x`` on ``[x_lo, x_hi] = [min(ln S0, ln F_T) − w sd, max(ln S0, ln F_T) + w sd]``
  with ``sd² = ∫_0^T ξ_0`` (``σ² T`` for Black–Scholes, ``∫ σ_loc²(t, 0) dt`` for local vol) and
  ``w = x_width_sd`` (owner: 4), clustered around a chosen spot level by the sinh map
  ``x = x_c + c sinh(ξ)``, ``ξ`` uniform, ``c = x_cluster · (x_hi − x_lo)/2`` (their eq. 3.1
  in log-spot); a continuous barrier replaces the corresponding end of the domain so that a
  node sits exactly at ``ln B``.  ``X`` uniform on ``±X_width_sd · sqrt(1/(2 k1))`` (the OU's
  stationary standard deviation) with an odd point count so that ``X = 0`` is a node.
* **Space.** Second-order central differences on the non-uniform grid (their eqs. 2.6–2.8),
  the mixed derivative as the product of the two central first-derivative stencils.  Far-field
  boundaries: ``∂_SS u = 0`` (linear in ``S``: ``∂_xx u = ∂_x u``, exact asymptotically for
  vanillas and digitals), i.e. the row ``∂_t u + (r − q) ∂_x u − r u = 0`` with a one-sided first
  derivative; in ``X``: ``∂_XX u = 0`` with the one-sided (inward, upwind) drift stencil; the
  mixed term is dropped on all boundary rows.  A continuous barrier is a Dirichlet row
  ``u = rebate`` (paid at hit) or ``rebate · DF(t, T)`` (paid at maturity), re-imposed after
  every stage.
* **Payoff.** Sampled at the nodes, except within one cell of each declared kink where the
  nodal value is replaced by the cell average (Pooley, Vetzal & Forsyth 2003, *Convergence
  remedies for non-smooth payoffs*): the mean of the payoff over the cell between the
  neighbouring mid-points, integrated by 8-point Gauss–Legendre on each smooth piece — second
  order for kinks and jumps irrespective of their position relative to the nodes.  Away from
  the kinks nodal sampling is exact for ``a e^x + b``, the far-field asymptote of every
  vanilla, so put–call parity holds to ``1e-7`` relative; averaging every cell instead biases
  ``e^x`` by ``(h₊ − h₋)/4`` per cell on the graded grid (measured: parity residual ``4.5e-4``
  under Black–Scholes and ``4.8e-3`` under the ``ω = 3`` kernel at 1y, 200 points).
* **Domain truncation.** The far-field condition is exact only asymptotically; with the
  ``ω = 3`` kernel the wings carry weight and the owner's ``±4`` ATM standard deviations leave a
  truncation error of ``2.4e-4`` relative on the 1y ATM call and put and ``1.5e-4`` on the
  digital (``3e-5`` / ``2e-6`` at ``±6`` sd, ``7e-6`` at ``±8``; ``1e-4`` on the 3m call at
  ``±4``; nil under Black–Scholes), measured against ``±12`` sd with the central spacing held
  fixed — :meth:`LSV1FPDE.width_sensitivity` reports this for any payoff.
* **Time.** ``dt`` from the :class:`~volsto.config.StepSchedule` (the MC grid,
  :meth:`~volsto.engine.grid.TimeGrid.build` with the leverage slices as extra knots), marched
  backward with the Hundsdorfer–Verwer scheme (``scheme="hv"``, their eq. 2.13, default
  ``θ = ½ + √3/6``, unconditionally stable in 2D with mixed derivatives per in 't Hout & Welfert
  2009) or Craig–Sneyd (``scheme="cs"``, their eq. 2.12, default ``θ = ½``); ``A_1`` (the
  ``x`` operator, including ``−r u``) and ``A_2`` (the ``X`` operator) implicit, the mixed
  operator ``A_0`` explicit.  Rannacher start (Rannacher 1984; Pooley et al. 2003): the first
  ``n_rannacher`` steps after maturity are each split into two half-steps of the Douglas
  scheme with ``θ = 1`` (fully implicit in each direction, mixed term explicit), which damps
  the odd-even oscillations of the non-smooth payoff and restores second order in ``dt``.
  Tridiagonal solves: Thomas algorithm batched over the other coordinate
  (:func:`solve_tridiagonal_batch`, a pure array numba kernel).

Checked by ``tests/test_pde_lsv1f.py``: Black–Scholes vanilla / digital closed forms and
put–call parity, the 2D machinery at ``ω → 0``, the Reiner–Rubinstein down-and-out call, the
European knock-in put decomposition, pure 1F Bergomi versus Monte Carlo (also from a bumped
initial factor state), grid-convergence orders, the domain-width sensitivity; slow: the cached
reference 1F LSV versus Monte Carlo (call, put, digital, European knock-in put, bridge-form
continuous knock-out call).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np
from numpy.typing import NDArray

from volsto._numba import njit
from volsto.config import DEFAULT_STEP_SCHEDULE, StepSchedule
from volsto.engine.grid import TimeGrid
from volsto.market.curves import ForwardCurve
from volsto.models.base import Model
from volsto.models.bergomi import BergomiSV
from volsto.models.bs import BlackScholes
from volsto.models.localvol import LocalVol
from volsto.models.lsv import LSV

FloatArray = NDArray[np.float64]
Payoff = Callable[[FloatArray], FloatArray]

# standard Hundsdorfer–Verwer parameter θ = ½ + √3/6 (in 't Hout & Foulon 2010, §2)
HV_THETA = 0.5 + math.sqrt(3.0) / 6.0

_SCHEMES = ("hv", "cs")
_REBATE_AT = ("hit", "maturity")


# --------------------------------------------------------------------------------------------
# tridiagonal kernel
# --------------------------------------------------------------------------------------------


@njit(cache=True)
def solve_tridiagonal_batch(
    lower: FloatArray, diag: FloatArray, upper: FloatArray, rhs: FloatArray, out: FloatArray
) -> None:  # pragma: no cover - numba
    """Thomas algorithm along axis 0 for every column ``j``: for each ``j`` solve
    ``lower[i, j] y[i-1] + diag[i, j] y[i] + upper[i, j] y[i+1] = rhs[i, j]`` (``lower[0]`` and
    ``upper[n-1]`` unused).  Pure array function; no pivoting (the ADI matrices are diagonally
    dominant).  Checked against ``numpy.linalg.solve`` in ``tests/test_pde_lsv1f.py``."""
    n, m = diag.shape
    c = np.empty((n, m))
    d = np.empty((n, m))
    for j in range(m):
        c[0, j] = upper[0, j] / diag[0, j]
        d[0, j] = rhs[0, j] / diag[0, j]
    for i in range(1, n):
        for j in range(m):
            den = diag[i, j] - lower[i, j] * c[i - 1, j]
            c[i, j] = upper[i, j] / den
            d[i, j] = (rhs[i, j] - lower[i, j] * d[i - 1, j]) / den
    for j in range(m):
        out[n - 1, j] = d[n - 1, j]
    for i in range(n - 2, -1, -1):
        for j in range(m):
            out[i, j] = d[i, j] - c[i, j] * out[i + 1, j]


# --------------------------------------------------------------------------------------------
# grids and stencils
# --------------------------------------------------------------------------------------------


def sinh_grid(lo: float, hi: float, centre: float, n: int, c: float) -> FloatArray:
    """``n`` points on ``[lo, hi]`` clustered around ``centre`` (clipped into the interval):
    ``z = centre + c sinh(ξ)`` with ``ξ`` uniform (in 't Hout & Foulon 2010, eq. 3.1).  The end
    points are set exactly to ``lo`` and ``hi``."""
    if not lo < hi:
        raise ValueError("need lo < hi")
    if n < 3 or c <= 0:
        raise ValueError("need n >= 3 and c > 0")
    zc = min(max(centre, lo), hi)
    xi = np.linspace(math.asinh((lo - zc) / c), math.asinh((hi - zc) / c), n)
    z = zc + c * np.sinh(xi)
    z[0], z[-1] = lo, hi
    if np.any(np.diff(z) <= 0):  # pragma: no cover - guards float rounding at the ends
        raise ValueError("sinh grid is not strictly increasing")
    return np.asarray(z, dtype=np.float64)


def fd_weights(z: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Three-point weights ``(n, 3)`` on ``(z_{i-1}, z_i, z_{i+1})`` of the first and second
    derivative on a non-uniform grid (in 't Hout & Foulon 2010, eqs. 2.6–2.8); the boundary rows
    carry the one-sided two-point first derivative and a zero second derivative."""
    n = z.size
    w1 = np.zeros((n, 3))
    w2 = np.zeros((n, 3))
    if n < 2:
        return w1, w2
    hm = z[1:-1] - z[:-2]
    hp = z[2:] - z[1:-1]
    w1[1:-1, 0] = -hp / (hm * (hm + hp))
    w1[1:-1, 1] = (hp - hm) / (hm * hp)
    w1[1:-1, 2] = hm / (hp * (hm + hp))
    w2[1:-1, 0] = 2.0 / (hm * (hm + hp))
    w2[1:-1, 1] = -2.0 / (hm * hp)
    w2[1:-1, 2] = 2.0 / (hp * (hm + hp))
    h0 = z[1] - z[0]
    w1[0, 1], w1[0, 2] = -1.0 / h0, 1.0 / h0
    hn = z[-1] - z[-2]
    w1[-1, 0], w1[-1, 1] = -1.0 / hn, 1.0 / hn
    return w1, w2


def _apply_axis0(lower: FloatArray, diag: FloatArray, upper: FloatArray, u: FloatArray) -> Any:
    """``(tridiag · u)`` along axis 0; the ``(n, m)`` coefficient arrays act column by column."""
    out = diag * u
    out[1:] += lower[1:] * u[:-1]
    out[:-1] += upper[:-1] * u[1:]
    return out


def _project_payoff(
    payoff: Payoff, x: FloatArray, kinks_x: FloatArray, window: int = 1, n_gl: int = 8
) -> FloatArray:
    """Terminal condition on the ``x`` grid: ``payoff(e^{x_i})`` at the nodes, except within
    ``window`` cells of each kink where the value is the mean of the payoff over the node's cell
    (mid-point to mid-point, half cells at the ends), Gauss–Legendre on every smooth piece
    between the ``kinks_x`` (Pooley, Vetzal & Forsyth 2003).  See the module docstring for why
    the smooth part is sampled rather than averaged (exactness on ``a e^x + b``)."""
    vals = np.array(payoff(np.exp(x)), dtype=np.float64)
    if vals.shape != x.shape:
        raise ValueError("payoff must return one value per spot")
    if kinks_x.size == 0:
        return vals
    edges = np.concatenate(([x[0]], 0.5 * (x[:-1] + x[1:]), [x[-1]]))
    mark = np.zeros(x.size, dtype=bool)
    for kx in kinks_x:
        for i in np.nonzero((edges[:-1] <= kx) & (kx <= edges[1:]))[0]:
            mark[max(int(i) - window, 0) : int(i) + window + 1] = True
    gl_x, gl_w = np.polynomial.legendre.leggauss(n_gl)
    for i in np.nonzero(mark)[0]:
        a, b = float(edges[i]), float(edges[i + 1])
        inner = kinks_x[(kinks_x > a) & (kinks_x < b)]
        pts = np.concatenate(([a], np.sort(inner), [b]))
        total = 0.0
        for p0, p1 in pairwise(pts):
            if p1 <= p0:
                continue
            nodes = 0.5 * (p1 - p0) * gl_x + 0.5 * (p0 + p1)
            piece = np.asarray(payoff(np.exp(nodes)), dtype=np.float64)
            total += 0.5 * (p1 - p0) * float(gl_w @ piece)
        vals[i] = total / (b - a)
    return vals


def _interp_cubic_axis0(z: FloatArray, vals: FloatArray, z0: float) -> FloatArray:
    """Four-point Lagrange interpolation along axis 0 of ``vals`` at ``z0`` (linear when fewer
    than four nodes; the single value when there is one)."""
    n = z.size
    if n == 1:
        return np.asarray(vals[0], dtype=np.float64)
    if not z[0] <= z0 <= z[-1]:
        raise ValueError(f"{z0} lies outside the grid [{z[0]}, {z[-1]}]")
    if n < 4:
        i = min(int(np.searchsorted(z, z0, side="right") - 1), n - 2)
        w = (z0 - z[i]) / (z[i + 1] - z[i])
        return np.asarray((1.0 - w) * vals[i] + w * vals[i + 1], dtype=np.float64)
    i = min(max(int(np.searchsorted(z, z0, side="right") - 1), 1), n - 3)
    idx = np.arange(i - 1, i + 3)
    zz = z[idx]
    out = np.zeros_like(vals[0], dtype=np.float64)
    for a in range(4):
        w = 1.0
        for b in range(4):
            if a != b:
                w *= (z0 - zz[b]) / (zz[a] - zz[b])
        out = out + w * vals[idx[a]]
    return np.asarray(out, dtype=np.float64)


# --------------------------------------------------------------------------------------------
# model adapter
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Dynamics:
    """What the PDE needs from a model: ``local_var(t, x)`` (``L²`` for the LSV, ``σ_loc²`` for
    local vol, ``σ²`` for Black–Scholes), ``factor_var(t, X)`` (``V = g(t) e^{ω X}``, ``None``
    when there is no factor), the OU parameters and the ATM total variance for the domain."""

    label: str
    forward_curve: ForwardCurve
    local_var: Callable[[float, FloatArray], FloatArray]
    factor_var: Callable[[float, FloatArray], FloatArray] | None
    k1: float
    rho: float
    omega: float
    X0: float
    atm_variance: Callable[[float], float]
    required_times: FloatArray


def _dynamics(model: Model) -> _Dynamics:
    if not isinstance(model, (BlackScholes, LocalVol, BergomiSV, LSV)):
        raise TypeError(f"unsupported model for the 1F PDE: {type(model).__name__}")
    fc = model.forward_curve
    if isinstance(model, BlackScholes):
        s2 = model.vol**2

        def bs_var(t: float, x: FloatArray) -> FloatArray:
            return np.full(x.shape, s2)

        return _Dynamics(
            "BlackScholes", fc, bs_var, None, 0.0, 0.0, 0.0, 0.0, lambda T: s2 * T, np.empty(0)
        )
    if isinstance(model, LocalVol):
        lv = model.local_vol

        def lv_var(t: float, x: FloatArray) -> FloatArray:
            return lv.local_var_k(t, x - float(fc.log_forward(t)))

        def lv_total(T: float) -> float:
            return float(np.interp(0.0, lv.k_grid, lv.cumulative_var(T)[0]))

        return _Dynamics("LocalVol", fc, lv_var, None, 0.0, 0.0, 0.0, 0.0, lv_total, np.empty(0))
    kernel = model.kernel if isinstance(model, LSV) else model
    p = kernel.params
    if not p.is_one_factor:
        raise ValueError("the PDE cross-check needs the one-factor kernel (theta = 0)")
    coef = float(kernel.coef[0])
    xi0 = kernel.xi0

    def sv_factor(t: float, X: FloatArray) -> FloatArray:
        return np.asarray(float(kernel.g(t)) * np.exp(coef * X), dtype=np.float64)

    if isinstance(model, LSV):
        lev = model.leverage

        def lsv_local(t: float, x: FloatArray) -> FloatArray:
            L = lev(t, np.exp(x))
            return np.asarray(L * L, dtype=np.float64)

        local, label, req = lsv_local, "LSV", model.required_times()
    else:

        def one(t: float, x: FloatArray) -> FloatArray:
            return np.ones(x.shape)

        local, label, req = one, "BergomiSV", np.empty(0)
    return _Dynamics(
        label,
        fc,
        local,
        sv_factor,
        float(p.k1),
        float(p.rho_SX1),
        float(p.omega),
        float(kernel.x0[0]),
        lambda T: float(xi0.integral(0.0, T)),
        np.asarray(req, dtype=np.float64),
    )


# --------------------------------------------------------------------------------------------
# result
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PDEResult:
    """Solution at ``t = 0`` on the grid with the interpolated price at ``(S0, X0)``.

    Attributes:
        price: value at the model's spot and initial factor state.
        spot, X0, maturity: evaluation point and horizon.
        x, X: grid coordinates (``ln S`` and the factor); ``values`` is ``(n_x, n_X)``.
        n_steps: time steps taken (Rannacher half-steps counted).
        scheme, theta, n_rannacher: time-stepping settings.
        lower_barrier, upper_barrier, rebate, rebate_at: barrier conventions used
            (``rebate_at`` is ``"n/a"`` when the rebate is zero).
    """

    price: float
    spot: float
    X0: float
    maturity: float
    x: FloatArray
    X: FloatArray
    values: FloatArray
    n_steps: int
    scheme: str
    theta: float
    n_rannacher: int
    lower_barrier: float | None
    upper_barrier: float | None
    rebate: float
    rebate_at: str

    @property
    def n_x(self) -> int:
        return int(self.x.size)

    @property
    def n_X(self) -> int:
        return int(self.X.size)

    @property
    def dx_min(self) -> float:
        return float(np.diff(self.x).min())

    @property
    def dX(self) -> float:
        return float(np.diff(self.X).max()) if self.n_X > 1 else 0.0

    def price_at(self, S: float, X: float = 0.0) -> float:
        """Value at spot ``S`` and factor ``X`` (four-point Lagrange interpolation in each
        coordinate; ``X`` is ignored for factorless models)."""
        if S <= 0:
            raise ValueError("S must be positive")
        col = _interp_cubic_axis0(self.X, self.values.T, X) if self.n_X > 1 else self.values[:, 0]
        return float(_interp_cubic_axis0(self.x, col, math.log(S)))

    def __repr__(self) -> str:
        bar = ""
        if self.lower_barrier is not None:
            bar += f", lower barrier {self.lower_barrier:g}"
        if self.upper_barrier is not None:
            bar += f", upper barrier {self.upper_barrier:g}"
        if bar:
            bar += f", rebate {self.rebate:g} at {self.rebate_at}"
        return (
            f"PDEResult(price={self.price:.8g}, T={self.maturity:g}, grid {self.n_x}x{self.n_X}, "
            f"{self.n_steps} steps, {self.scheme} theta={self.theta:.4g}, "
            f"rannacher={self.n_rannacher}{bar})"
        )


# --------------------------------------------------------------------------------------------
# solver
# --------------------------------------------------------------------------------------------


class LSV1FPDE:
    """ADI finite-difference pricer for Black–Scholes, local vol, the pure 1F Bergomi kernel
    and the 1F LSV; see the module docstring for the equation, grids and schemes.

    Args:
        model: :class:`~volsto.models.bs.BlackScholes`, :class:`~volsto.models.localvol.LocalVol`,
            :class:`~volsto.models.bergomi.BergomiSV` or :class:`~volsto.models.lsv.LSV` (the
            kernel must be one-factor, ``θ = 0``).
        n_x: points in ``x = ln S`` (owner: 400).
        n_X: points in the factor ``X``, odd so that ``X = 0`` is a node (owner: 120, rounded
            up to 121); ignored (the grid is ``n_x × 1``) for factorless models.
        x_width_sd, X_width_sd: half-widths in ATM standard deviations of ``ln S_T`` and in
            stationary standard deviations of ``X`` (owner: 4 and 4).  The ATM standard
            deviation ignores the vol of vol: at ``ω = 3`` the ``±4`` sd domain truncates the
            1y vanilla by ``2.4e-4`` relative (``3e-5`` at ``±6``, see the module docstring and
            :meth:`width_sensitivity`); the default stays the owner's 4.
        schedule: :class:`~volsto.config.StepSchedule` or a uniform ``dt``.
        scheme: ``"hv"`` (Hundsdorfer–Verwer) or ``"cs"`` (Craig–Sneyd).
        theta: ADI parameter; ``None`` selects the scheme's standard value (``½ + √3/6`` for
            HV, ``½`` for CS).
        n_rannacher: number of initial steps replaced by two Douglas ``θ = 1`` half-steps each.
        x_cluster: sinh clustering parameter ``c`` as a fraction of the ``x`` half-width.
    """

    def __init__(
        self,
        model: Model,
        *,
        n_x: int = 400,
        n_X: int = 121,
        x_width_sd: float = 4.0,
        X_width_sd: float = 4.0,
        schedule: StepSchedule | float = DEFAULT_STEP_SCHEDULE,
        scheme: str = "hv",
        theta: float | None = None,
        n_rannacher: int = 2,
        x_cluster: float = 0.25,
    ) -> None:
        self.model = model
        self.dyn = _dynamics(model)
        if int(n_x) != n_x or n_x < 16:
            raise ValueError("n_x must be an integer >= 16")
        if int(n_X) != n_X or n_X < 5 or n_X % 2 == 0:
            raise ValueError("n_X must be an odd integer >= 5 (X = 0 must be a grid node)")
        if not (np.isfinite(x_width_sd) and x_width_sd > 0):
            raise ValueError("x_width_sd must be positive")
        if not (np.isfinite(X_width_sd) and X_width_sd > 0):
            raise ValueError("X_width_sd must be positive")
        if scheme not in _SCHEMES:
            raise ValueError(f"scheme must be one of {_SCHEMES}, got {scheme!r}")
        if theta is None:
            theta = HV_THETA if scheme == "hv" else 0.5
        if not 0.0 < theta <= 1.0:
            raise ValueError("theta must lie in (0, 1]")
        if int(n_rannacher) != n_rannacher or n_rannacher < 0:
            raise ValueError("n_rannacher must be a non-negative integer")
        if not (np.isfinite(x_cluster) and x_cluster > 0):
            raise ValueError("x_cluster must be positive")
        if isinstance(schedule, StepSchedule):
            self.schedule = schedule
        else:
            if not (np.isfinite(schedule) and schedule > 0):
                raise ValueError("schedule must be a StepSchedule or a positive dt")
            self.schedule = StepSchedule.uniform(float(schedule))
        self.n_x = int(n_x)
        self.n_X = int(n_X) if self.dyn.factor_var is not None else 1
        self.x_width_sd = float(x_width_sd)
        self.X_width_sd = float(X_width_sd)
        self.scheme = scheme
        self.theta = float(theta)
        self.n_rannacher = int(n_rannacher)
        self.x_cluster = float(x_cluster)

    # -- settings ------------------------------------------------------------------------------

    @property
    def settings(self) -> dict[str, Any]:
        return {
            "n_x": self.n_x,
            "n_X": self.n_X if self.dyn.factor_var is not None else 121,
            "x_width_sd": self.x_width_sd,
            "X_width_sd": self.X_width_sd,
            "schedule": self.schedule,
            "scheme": self.scheme,
            "theta": self.theta,
            "n_rannacher": self.n_rannacher,
            "x_cluster": self.x_cluster,
        }

    def with_settings(self, **changes: Any) -> LSV1FPDE:
        """Copy with some constructor settings changed."""
        return LSV1FPDE(self.model, **{**self.settings, **changes})

    @property
    def has_factor(self) -> bool:
        return self.dyn.factor_var is not None

    def __repr__(self) -> str:
        return (
            f"LSV1FPDE({self.dyn.label}, {self.n_x}x{self.n_X}, x +-{self.x_width_sd:g} sd, "
            f"X +-{self.X_width_sd:g} sd, {self.schedule!r}, {self.scheme} theta="
            f"{self.theta:.4g}, rannacher={self.n_rannacher})"
        )

    # -- pricing -------------------------------------------------------------------------------

    def price(
        self,
        payoff: Payoff,
        maturity: float,
        *,
        kinks: Sequence[float] = (),
        cluster_at: float | None = None,
        lower_barrier: float | None = None,
        upper_barrier: float | None = None,
        rebate: float = 0.0,
        rebate_at: str | None = None,
    ) -> PDEResult:
        """Price ``payoff(S_T)`` (undiscounted terminal cash flow, vectorised in ``S``) paid at
        ``maturity``, optionally continuously knocked out at ``lower_barrier`` / ``upper_barrier``
        (Dirichlet rows).  A non-zero ``rebate`` requires the payment convention ``rebate_at``
        (``"hit"`` or ``"maturity"``, no default); it is recorded as ``"n/a"`` when the rebate
        is zero.

        ``kinks`` lists the spot levels where the payoff is non-smooth (strikes, digital
        levels) for the cell-averaging projection; ``cluster_at`` is the spot level the ``x``
        grid clusters around (the spot when ``None``).
        """
        T = float(maturity)
        if not (np.isfinite(T) and T > 0):
            raise ValueError("maturity must be positive")
        S0 = self.dyn.forward_curve.spot
        if lower_barrier is not None and not 0.0 < lower_barrier < S0:
            raise ValueError("lower_barrier must lie in (0, spot)")
        if upper_barrier is not None and not upper_barrier > S0:
            raise ValueError("upper_barrier must exceed the spot")
        if not (np.isfinite(rebate) and rebate >= 0):
            raise ValueError("rebate must be non-negative")
        if rebate_at is not None and rebate_at not in _REBATE_AT:
            raise ValueError(f"rebate_at must be one of {_REBATE_AT}, got {rebate_at!r}")
        if rebate > 0 and rebate_at is None:
            raise ValueError("rebate_at ('hit' or 'maturity') is required when rebate > 0")
        rebate_at_used = rebate_at if rebate > 0 else "n/a"
        kk = np.asarray(kinks, dtype=np.float64).ravel()
        if np.any(kk <= 0):
            raise ValueError("kinks are spot levels and must be positive")
        centre = S0 if cluster_at is None else float(cluster_at)
        if centre <= 0:
            raise ValueError("cluster_at must be positive")

        x, X = self._grids(T, math.log(centre), lower_barrier, upper_barrier)
        dyn = self.dyn
        rates, divs = dyn.forward_curve.rate_curve, dyn.forward_curve.dividend_curve
        dirichlet_rows = ([0] if lower_barrier is not None else []) + (
            [x.size - 1] if upper_barrier is not None else []
        )
        df_T = float(rates.df(T))

        def rebate_value(t: float) -> float:
            if rebate == 0.0:
                return 0.0
            return rebate if rebate_at == "hit" else rebate * df_T / float(rates.df(t))

        def impose(u: FloatArray, t: float) -> FloatArray:
            for row in dirichlet_rows:
                u[row, :] = rebate_value(t)
            return u

        # terminal condition: nodal payoff, cell-averaged around the kinks, broadcast over X
        u0 = _project_payoff(payoff, x, np.log(kk))
        U = np.ascontiguousarray(np.repeat(u0[:, None], X.size, axis=1))
        impose(U, T)

        # time grid (MC schedule, leverage slices as knots) and the step list, backward
        grid = TimeGrid.build([T], self.schedule, calibration_grid=dyn.required_times)
        times = grid.times
        steps: list[tuple[float, float, str]] = []
        for n in range(times.size - 1, 0, -1):
            t_hi, t_lo = float(times[n]), float(times[n - 1])
            if times.size - 1 - n < self.n_rannacher:
                tm = 0.5 * (t_hi + t_lo)
                steps += [(t_hi, tm, "do1"), (tm, t_lo, "do1")]
            else:
                steps.append((t_hi, t_lo, self.scheme))

        op = _Operators(self, x, X, dirichlet_rows)
        layer_old = op.layer(T)
        for t_old, t_new, kind in steps:
            dt = t_old - t_new
            r = float(rates.integrated(t_new, t_old)) / dt
            q = float(divs.integrated(t_new, t_old)) / dt
            layer_new = op.layer(t_new)
            A1_old = op.build_A1(layer_old, r, q)
            A1_new = op.build_A1(layer_new, r, q)
            th = 1.0 if kind == "do1" else self.theta
            F_old = op.apply_A1(A1_old, U) + op.apply_A2(U) + op.apply_A0(layer_old, U)
            Y0 = U + dt * F_old
            Y1 = op.solve_x(A1_new, th * dt, impose(Y0 - th * dt * op.apply_A1(A1_old, U), t_new))
            Y2 = impose(op.solve_X(th * dt, Y1 - th * dt * op.apply_A2(U)), t_new)
            if kind == "do1":
                U = Y2
            elif kind == "hv":
                F_new = op.apply_A1(A1_new, Y2) + op.apply_A2(Y2) + op.apply_A0(layer_new, Y2)
                Yt0 = Y0 + 0.5 * dt * (F_new - F_old)
                rhs = impose(Yt0 - th * dt * op.apply_A1(A1_new, Y2), t_new)
                Yt1 = op.solve_x(A1_new, th * dt, rhs)
                U = impose(op.solve_X(th * dt, Yt1 - th * dt * op.apply_A2(Y2)), t_new)
            else:  # craig-sneyd
                Yt0 = Y0 + 0.5 * dt * (op.apply_A0(layer_new, Y2) - op.apply_A0(layer_old, U))
                rhs = impose(Yt0 - th * dt * op.apply_A1(A1_old, U), t_new)
                Yt1 = op.solve_x(A1_new, th * dt, rhs)
                U = impose(op.solve_X(th * dt, Yt1 - th * dt * op.apply_A2(U)), t_new)
            layer_old = layer_new

        res = PDEResult(
            float("nan"),
            S0,
            dyn.X0,
            T,
            x,
            X,
            U,
            len(steps),
            self.scheme,
            self.theta,
            self.n_rannacher,
            lower_barrier,
            upper_barrier,
            float(rebate),
            str(rebate_at_used),
        )
        return PDEResult(**{**res.__dict__, "price": res.price_at(S0, dyn.X0)})

    def width_sensitivity(
        self, payoff: Payoff, maturity: float, *, factor: float = 2.0, **price_kwargs: Any
    ) -> tuple[PDEResult, PDEResult]:
        """Domain-adequacy diagnostic: the price at the configured ``x_width_sd`` and at
        ``factor`` times it with ``n_x`` scaled so that the central spacing of the sinh grid is
        unchanged (``n_x − 1 → factor (n_x − 1)``; the wings get finer as well).  The difference
        is the truncation error of the far-field condition (plus the wing-discretisation
        change); measured values in the module docstring.  Checked by ``tests/test_pde_lsv1f.py``
        (pure 1F Bergomi, 3m call, ``±4`` versus ``±8`` sd within ``5e-4`` relative)."""
        if not (np.isfinite(factor) and factor > 1.0):
            raise ValueError("factor must exceed 1")
        base = self.price(payoff, maturity, **price_kwargs)
        wide = self.with_settings(
            x_width_sd=factor * self.x_width_sd, n_x=round(factor * (self.n_x - 1)) + 1
        )
        return base, wide.price(payoff, maturity, **price_kwargs)

    def _grids(
        self, T: float, x_centre: float, lower: float | None, upper: float | None
    ) -> tuple[FloatArray, FloatArray]:
        dyn = self.dyn
        x0 = math.log(dyn.forward_curve.spot)
        xf = float(dyn.forward_curve.log_forward(T))
        sd = math.sqrt(dyn.atm_variance(T))
        if not sd > 0:
            raise ValueError("the model's ATM total variance must be positive")
        x_lo = min(x0, xf) - self.x_width_sd * sd
        x_hi = max(x0, xf) + self.x_width_sd * sd
        x_lo = min(x_lo, x_centre - sd) if lower is None else math.log(lower)
        x_hi = max(x_hi, x_centre + sd) if upper is None else math.log(upper)
        c = self.x_cluster * 0.5 * (x_hi - x_lo)
        x = sinh_grid(x_lo, x_hi, x_centre, self.n_x, c)
        if dyn.factor_var is None:
            return x, np.zeros(1)
        half = self.X_width_sd * math.sqrt(0.5 / dyn.k1)
        X = np.linspace(-half, half, self.n_X)
        X[self.n_X // 2] = 0.0
        return x, X

    # -- convenience products --------------------------------------------------------------

    def vanilla(self, strike: float, maturity: float, cp: int | str) -> PDEResult:
        """``(cp (S_T − K))⁺``; cross-checks :class:`~volsto.products.vanilla.EuropeanOption`."""
        K, s = _strike(strike), _cp(cp)
        return self.price(
            lambda S: np.maximum(s * (S - K), 0.0), maturity, kinks=(K,), cluster_at=K
        )

    def digital(self, strike: float, maturity: float, cp: int | str) -> PDEResult:
        """Cash-or-nothing ``1{cp (S_T − K) > 0}`` (unit payout, no smoothing beyond the cell
        average); cross-checks :class:`~volsto.products.vanilla.DigitalOption`."""
        K, s = _strike(strike), _cp(cp)
        return self.price(
            lambda S: (s * (S - K) > 0.0).astype(np.float64), maturity, kinks=(K,), cluster_at=K
        )

    def knock_out_call(
        self,
        strike: float,
        maturity: float,
        barrier: float,
        direction: str,
        *,
        rebate: float = 0.0,
        rebate_at: str | None = None,
    ) -> PDEResult:
        """Continuously monitored knock-out call, ``direction`` ``"down"`` (``B < S0``) or
        ``"up"`` (``B > S0``): Dirichlet condition on the barrier line; a non-zero ``rebate``
        needs its payment convention ``rebate_at`` (``"hit"`` / ``"maturity"``).  Checked
        against Reiner–Rubinstein under Black–Scholes and the MC bridge form under the LSV."""
        K = _strike(strike)
        if direction not in ("down", "up"):
            raise ValueError("direction must be 'down' or 'up'")
        if barrier <= 0:
            raise ValueError("barrier must be positive")
        lo, up = (float(barrier), None) if direction == "down" else (None, float(barrier))
        return self.price(
            lambda S: np.maximum(S - K, 0.0),
            maturity,
            kinks=(K,),
            cluster_at=K,
            lower_barrier=lo,
            upper_barrier=up,
            rebate=rebate,
            rebate_at=rebate_at,
        )

    def european_ki_put(self, strike: float, maturity: float, barrier: float) -> PDEResult:
        """European knock-in put ``(K − S_T)⁺ 1{S_T < B}`` — the knock-in is observed at
        maturity only, so it is a terminal condition (kinks at ``B`` and ``K``).  Identity checked
        under Black–Scholes: ``= put(B) + (K − B) · digital put(B)`` for ``B ≤ K`` (SPEC §6.6)."""
        K = _strike(strike)
        if barrier <= 0:
            raise ValueError("barrier must be positive")
        B = float(barrier)
        return self.price(
            lambda S: np.where(S < B, np.maximum(K - S, 0.0), 0.0),
            maturity,
            kinks=(min(B, K), K),
            cluster_at=min(B, K),
        )


def _strike(strike: float) -> float:
    if not (np.isfinite(strike) and strike > 0):
        raise ValueError("strike must be positive")
    return float(strike)


def _cp(cp: int | str) -> float:
    if isinstance(cp, str):
        s = cp.lower()
        if s in ("c", "call"):
            return 1.0
        if s in ("p", "put"):
            return -1.0
        raise ValueError(f"cp must be 'call'/'put' or +1/-1, got {cp!r}")
    if cp in (1, -1):
        return float(cp)
    raise ValueError(f"cp must be 'call'/'put' or +1/-1, got {cp!r}")


# --------------------------------------------------------------------------------------------
# discrete operators
# --------------------------------------------------------------------------------------------


@dataclass
class _Layer:
    """Coefficient fields at one calendar time: ``sig2 = L² V`` and ``cmix = ρ L sqrt(V)``."""

    sig2: FloatArray
    cmix: FloatArray | None


@dataclass
class _Tridiag:
    lower: FloatArray
    diag: FloatArray
    upper: FloatArray


class _Operators:
    """``A_1`` (x, implicit), ``A_2`` (X, implicit, time-independent OU operator) and ``A_0``
    (mixed, explicit) on the ``(x, X)`` grid, with the batched Thomas solves."""

    def __init__(
        self, pde: LSV1FPDE, x: FloatArray, X: FloatArray, dirichlet_rows: list[int]
    ) -> None:
        self.pde = pde
        self.x, self.X = x, X
        self.n_x, self.n_X = x.size, X.size
        self.rows = dirichlet_rows
        self.w1x, self.w2x = fd_weights(x)
        self.w1X, self.w2X = fd_weights(X)
        self.interior_x = np.ones(self.n_x)
        self.interior_x[[0, -1]] = 0.0
        dyn = pde.dyn
        if self.n_X > 1:
            a2 = np.full(self.n_X, 0.5)
            a2[[0, -1]] = 0.0
            b2 = -dyn.k1 * X
            self.A2 = _Tridiag(
                b2 * self.w1X[:, 0] + a2 * self.w2X[:, 0],
                b2 * self.w1X[:, 1] + a2 * self.w2X[:, 1],
                b2 * self.w1X[:, 2] + a2 * self.w2X[:, 2],
            )
        else:
            self.A2 = _Tridiag(np.zeros(1), np.zeros(1), np.zeros(1))
        self._X_cache: dict[float, _Tridiag] = {}

    # -- coefficients ----------------------------------------------------------------------

    def layer(self, t: float) -> _Layer:
        dyn = self.pde.dyn
        lv = np.asarray(dyn.local_var(t, self.x), dtype=np.float64)
        if dyn.factor_var is None:
            return _Layer(lv[:, None] * np.ones((1, self.n_X)), None)
        fv = np.asarray(dyn.factor_var(t, self.X), dtype=np.float64)
        sig2 = lv[:, None] * fv[None, :]
        cmix = dyn.rho * np.sqrt(lv)[:, None] * np.sqrt(fv)[None, :]
        return _Layer(sig2, cmix)

    def build_A1(self, layer: _Layer, r: float, q: float) -> _Tridiag:
        a = 0.5 * layer.sig2 * self.interior_x[:, None]  # zero diffusion on the far-field rows
        b = (r - q) - a
        w1, w2 = self.w1x, self.w2x
        lower = b * w1[:, 0, None] + a * w2[:, 0, None]
        diag = b * w1[:, 1, None] + a * w2[:, 1, None] - r
        upper = b * w1[:, 2, None] + a * w2[:, 2, None]
        for row in self.rows:
            lower[row], diag[row], upper[row] = 0.0, 0.0, 0.0
        return _Tridiag(lower, diag, upper)

    # -- applications ----------------------------------------------------------------------

    def apply_A1(self, A1: _Tridiag, U: FloatArray) -> FloatArray:
        return np.asarray(_apply_axis0(A1.lower, A1.diag, A1.upper, U), dtype=np.float64)

    def apply_A2(self, U: FloatArray) -> FloatArray:
        if self.n_X == 1:
            return np.zeros_like(U)
        A2 = self.A2
        out = U * A2.diag[None, :]
        out[:, 1:] += U[:, :-1] * A2.lower[None, 1:]
        out[:, :-1] += U[:, 1:] * A2.upper[None, :-1]
        return np.asarray(out, dtype=np.float64)

    def apply_A0(self, layer: _Layer, U: FloatArray) -> FloatArray:
        if layer.cmix is None or self.n_X == 1:
            return np.zeros_like(U)
        w1x, w1X = self.w1x, self.w1X
        dx = np.zeros_like(U)
        dx[1:-1] = (
            w1x[1:-1, 0, None] * U[:-2] + w1x[1:-1, 1, None] * U[1:-1] + w1x[1:-1, 2, None] * U[2:]
        )
        out = np.zeros_like(U)
        out[:, 1:-1] = (
            w1X[None, 1:-1, 0] * dx[:, :-2]
            + w1X[None, 1:-1, 1] * dx[:, 1:-1]
            + w1X[None, 1:-1, 2] * dx[:, 2:]
        )
        return np.asarray(layer.cmix * out, dtype=np.float64)

    # -- solves ----------------------------------------------------------------------------

    def solve_x(self, A1: _Tridiag, tau: float, rhs: FloatArray) -> FloatArray:
        """``(I − τ A_1) y = rhs`` column by column."""
        out = np.empty_like(rhs)
        solve_tridiagonal_batch(
            np.ascontiguousarray(-tau * A1.lower),
            np.ascontiguousarray(1.0 - tau * A1.diag),
            np.ascontiguousarray(-tau * A1.upper),
            np.ascontiguousarray(rhs),
            out,
        )
        return out

    def solve_X(self, tau: float, rhs: FloatArray) -> FloatArray:
        """``(I − τ A_2) y = rhs`` row by row (identity for factorless models)."""
        if self.n_X == 1:
            return np.array(rhs, copy=True)
        key = round(tau, 15)
        M = self._X_cache.get(key)
        if M is None:
            rep = self.n_x
            M = _Tridiag(
                np.ascontiguousarray(np.repeat((-tau * self.A2.lower)[:, None], rep, axis=1)),
                np.ascontiguousarray(np.repeat((1.0 - tau * self.A2.diag)[:, None], rep, axis=1)),
                np.ascontiguousarray(np.repeat((-tau * self.A2.upper)[:, None], rep, axis=1)),
            )
            self._X_cache[key] = M
        outT = np.empty((self.n_X, self.n_x))
        solve_tridiagonal_batch(M.lower, M.diag, M.upper, np.ascontiguousarray(rhs.T), outT)
        return np.ascontiguousarray(outT.T)


# --------------------------------------------------------------------------------------------
# grid convergence
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ConvergenceLine:
    """Prices at successive halvings of one discretisation parameter.

    ``differences[i] = prices[i] − prices[i+1]``; ``observed_order = log2(D_0 / D_1)`` when three
    or more levels are available and the differences have the same sign;
    ``richardson = (2^p P_fine − P_coarse)/(2^p − 1)`` on the two finest levels with the
    nominal order ``p``.
    """

    direction: str
    resolutions: tuple[float, ...]
    prices: tuple[float, ...]
    differences: tuple[float, ...]
    observed_order: float | None
    nominal_order: float
    richardson: float

    def __repr__(self) -> str:
        ps = ", ".join(f"{p:.8g}" for p in self.prices)
        ds = ", ".join(f"{d:.3g}" for d in self.differences)
        o = "n/a" if self.observed_order is None else f"{self.observed_order:.2f}"
        return (
            f"{self.direction}: prices [{ps}], differences [{ds}], order {o}, "
            f"richardson {self.richardson:.8g}"
        )


@dataclass(frozen=True)
class ConvergenceTable:
    """Richardson study of a PDE price: one :class:`ConvergenceLine` per refined direction and
    the combined extrapolation ``base + Σ_d (richardson_d − base)`` (leading errors assumed
    additive across ``dx``, ``dX``, ``dt``)."""

    base_price: float
    lines: dict[str, ConvergenceLine] = field(default_factory=dict)

    @property
    def price_extrapolated(self) -> float:
        return self.base_price + sum(ln.richardson - self.base_price for ln in self.lines.values())

    @property
    def observed_orders(self) -> dict[str, float | None]:
        return {k: v.observed_order for k, v in self.lines.items()}

    def __repr__(self) -> str:
        body = "\n".join(f"  {ln!r}" for ln in self.lines.values())
        return (
            f"ConvergenceTable(base={self.base_price:.8g}, extrapolated="
            f"{self.price_extrapolated:.8g}\n{body}\n)"
        )


def _halved(schedule: StepSchedule) -> StepSchedule:
    return StepSchedule(schedule.breaks, tuple(0.5 * d for d in schedule.dts))


def convergence_table(
    pde: LSV1FPDE | Model,
    payoff: Payoff,
    maturity: float,
    *,
    levels: int = 3,
    directions: Sequence[str] = ("dx", "dX", "dt"),
    nominal_order: float = 2.0,
    **price_kwargs: Any,
) -> ConvergenceTable:
    """Halve ``dx`` (``n_x → 2 n_x − 1``, nested sinh grids), ``dX`` (``n_X → 2 n_X − 1``) and
    ``dt`` (the schedule's steps halved) separately, ``levels`` times each from the settings of
    ``pde`` (a :class:`LSV1FPDE`, or a model priced with the default settings), and report the
    observed orders and Richardson extrapolants; ``price_kwargs`` go to :meth:`LSV1FPDE.price`.
    ``dX`` is skipped for factorless models.  Checked by ``tests/test_pde_lsv1f.py`` (vanilla
    order in ``dx`` ≈ 2)."""
    if levels < 2:
        raise ValueError("need at least two levels")
    base = pde if isinstance(pde, LSV1FPDE) else LSV1FPDE(pde)
    for d in directions:
        if d not in ("dx", "dX", "dt"):
            raise ValueError(f"unknown direction {d!r}")
    base_price = base.price(payoff, maturity, **price_kwargs).price
    lines: dict[str, ConvergenceLine] = {}
    for d in directions:
        if d == "dX" and not base.has_factor:
            continue
        prices = [base_price]
        res: list[float] = []
        cur = base
        for lvl in range(levels):
            if d == "dx":
                res.append(float(cur.n_x))
            elif d == "dX":
                res.append(float(cur.n_X))
            else:
                res.append(cur.schedule.finest)
            if lvl == levels - 1:
                break
            if d == "dx":
                cur = cur.with_settings(n_x=2 * cur.n_x - 1)
            elif d == "dX":
                cur = cur.with_settings(n_X=2 * cur.n_X - 1)
            else:
                cur = cur.with_settings(schedule=_halved(cur.schedule))
            prices.append(cur.price(payoff, maturity, **price_kwargs).price)
        diffs = [prices[i] - prices[i + 1] for i in range(levels - 1)]
        order: float | None = None
        if levels >= 3 and diffs[-2] * diffs[-1] > 0 and abs(diffs[-1]) > 1e-14:
            order = float(np.log2(abs(diffs[-2]) / abs(diffs[-1])))
        w = 2.0**nominal_order
        rich = (w * prices[-1] - prices[-2]) / (w - 1.0)
        lines[d] = ConvergenceLine(
            d, tuple(res), tuple(prices), tuple(diffs), order, float(nominal_order), float(rich)
        )
    return ConvergenceTable(base_price, lines)


__all__ = [
    "HV_THETA",
    "LSV1FPDE",
    "ConvergenceLine",
    "ConvergenceTable",
    "PDEResult",
    "convergence_table",
    "fd_weights",
    "sinh_grid",
    "solve_tridiagonal_batch",
]
