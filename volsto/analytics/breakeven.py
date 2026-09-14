"""Break-even engine of the two-factor model: closed forms of the ATMF vol's sensitivities to
the state and of the two break-evens ``SpotVolCovar`` and ``VolVar`` (SPEC §15 Part 3, M7
addendum; the order-one machinery of Bergomi ch. 7 §7.4, ch. 8 eq. 8.54 and ch. 9 §9.2 written
in the break-even parametrisation of :mod:`volsto.analytics.reparam`).

Definitions (``σ̂_T`` the ATMF implied vol of maturity ``T``, ``σ_0`` the instantaneous spot
vol, all covariations per unit time)::

    SensiSpot(T) = σ_0 ∂σ̂_T/∂ln S_0      SensiX(T) = ∂σ̂_T/∂X_0     SensiY(T) = ∂σ̂_T/∂Y_0
    SpotVolCovar(T) = (1/(σ_0 dt)) <d ln S, dσ̂_T> = SensiSpot + ρ_SX SensiX + ρ_SY SensiY
    VolVar(T)       = (1/dt) <dσ̂_T, dσ̂_T>
                    = SensiSpot² + SensiX² + SensiY² + 2 ρ_SX SensiSpot SensiX
                      + 2 ρ_SY SensiSpot SensiY + 2 ρ_XY SensiX SensiY

in **absolute vol units** (``dσ̂``, not ``d ln σ̂``): the skew-stickiness ratio is then
``SSR_T = SpotVolCovar / (σ_0 Skew_T)`` with ``Skew_T = ∂σ̂_T/∂ln K`` (book eq. 9.3 / 12.50),
and ``vovol_T = sqrt(VolVar) / σ̂_T`` is the lognormal vol of the ATMF vol.

**σ_0.**  SPEC §15 defines ``σ_0 = L(0, S_0) sqrt(ξ_0^0)``, the model's instantaneous vol
(``L = 1`` for the naked model; :func:`sigma_0_of`); it is the ``σ_0`` of the state covariance of
:func:`volsto.analytics.smile_dynamics.atmf_vol_of_vol` (``initial_vol``), so every quantity of
this module and of the simulation estimate uses the same number.  The targets of
:mod:`volsto.calibration.targets` read it off a market surface as the one-month ATMF vol (the
proxy of the ``T → 0`` ATMF vol; 0.2200 against 0.2194 for the cached 2F LSV on the reference
SSVI); a comparison with a :class:`~volsto.calibration.targets.TargetSet` must pass
``sigma_0=targets.sigma_0`` to both sides.  The one-month VS vol (0.2609 on the reference SSVI,
19% above) is **not** σ_0 and is no longer a default anywhere.

**Most-likely-path level** (:func:`mlp_atmf_variance`).  ``σ̂_T² ≈ (1/T) ∫₀ᵀ E[V_t | S_T = F_T]
dt`` with ``V_t = ξ_t^t L²(t, S_t)``; at order one in the vol of vol ``(ln S_T, ln V_t)`` is
Gaussian, so with ``c̃_i(t) = ∫₀ᵗ sqrt(ξ_0^u L²(u, F_u)) e^{−k_i(t−u)} du`` (the spot/factor
covariance kernel), ``W_t = ∫₀ᵗ ξ_0^u L²(u, F_u) du`` and the local volatility along the ATM
path approximated by ``L²(t, S) ≈ exp(a(t) + b(t) ln S)``, ``b(t) = ∂ ln L²/∂ ln S`` at the
forward (the ATM-skew-invariant log-linear choice; naked model ``a = b = 0``)::

    E[V_t | S_T = F_T] = ξ_0^t L̂²_t exp( Σ_i ω_i e^{−k_i t} X_0^i + ½ Σ_i λ_i c̃_i(t)
                                          + b(t) Σ_i λ_i c̃_i(t) (1 − W_t/W_T)
                                          − ½ (Σ_i λ_i c̃_i(t))² / W_T )

The first two terms are the order-one (forward-versus-mean) conditioning; the last two are
**partial second-order** terms (the conditional variance of the Gaussian model): the
``−½ (Σλc̃)²/W_T`` term improves the level at ν ≤ 0.4 (MC on a 1F flat curve: −3.0 s.e. with it,
−11.5 s.e. without) and the whole form is 1% low-biased at ν = 1.  The term ``½ b² W_t (1 −
W_t/W_T)`` of the same Gaussian model is **dropped**: it is wrong at its own order because the
order-zero spot dynamics is not lognormal once ``b ≠ 0`` (Bachelier check ``L² = σ² F²/S²``, ``b
= −2`` exactly: exact ATM vol ``σ (1 + σ² T/24)``, the term predicts ``σ (1 + σ² T/6)``), and on
the reference surface ``|b| = 10–40`` makes it unusable (measured on the cached LSVs, 3M: level
0.2230 / 0.2398 with the term against 0.2142 / 0.2175 without and 0.2097 / 0.2098 simulated, 2F
Table 8.2 / 1F ω = 2).

**First-order sensitivities** (:func:`first_order_breakevens`) are the derivatives of that
exponent with the order-zero weights ``V̂_t = ξ_0^t L̂²_t`` and ``W_T = σ̂_T² T``::

    SensiX_i(T)  = ½ ω_i A_i(T) σ̂_T,   A_i = ∫₀ᵀ V̂_t e^{−k_i t} dt / W_T           (eq. 7.38)
    SensiSpot(T) = σ_0 σ̂_T B(T) / (2 W_T),   B = ∫₀ᵀ V̂_t b(t) dt          (0 for the naked model)
    Skew_T       = Σ_i λ_i J_i(T) + Skew^LV_T,   J_i = ∫₀ᵀ V̂_t c̃_i(t) dt / (2 σ̂_T³ T²)  (eq. 8.54)
    Skew^LV_T    = σ̂_T B_K(T) / (2 W_T),   B_K = ∫₀ᵀ V̂_t b(t) W_t/W_T dt   (0 for the naked model)

so that ``SpotVolCovar`` and ``Skew`` are affine in ``(λ1, λ2)`` at fixed ``(k1, k2)`` and
``VolVar`` is quadratic in ``(ω1, ω2)`` with ``ρ_XY`` from ``χ``; differentiating the closed form
directly in ``ln S_0, X_0, Y_0`` is the integration-by-parts route that never needs
``∂²σ̂/∂ln K²``.  ``Skew^LV`` is the local-volatility part of the skew (the strike derivative of
the same exponent), so for an LSV ``Skew`` is the model's total first-order skew and
:attr:`BreakEvenValues.ssr` its SSR; ``B/B_K`` is the local-vol SSR of the term structure
(book eqs. 12.53–12.54 with the leverage's slope as the local skew).

**Local branch on real surfaces** (``local`` given).  The closure is first order in ``b`` while
``b`` is O(10–50) at the short end of the reference SSVI (cached 2F leverage: −31 at 1e-3, −12
at 0.05, −6.2 at 0.1, −2.6 at 3M, −1.3 at 1Y in ``ln L²`` per ``ln S`` with ``h = 0.02``; the
Dupire slope is ``h``-dependent below ``t ≈ 0.05`` where the short end is not log-linear over
``h = 0.02``).  Measured against the spot-bump partial of the simulation (40k paths, eps 0.05):
the unrescaled ``SensiSpot`` overshoots by +22% / +47% (2F Table 8.2, 3M / 1Y) and +22% / +24%
(1F ω = 2), the pure-Dupire local-vol check by +25–31%.  When the :class:`LocalSlope` carries
the market ATM skew (``skew_market``, the surface the leverage was calibrated to) the engine
takes the book's route: ``SensiSpot`` is multiplied by ``Skew^mkt_T / (Σ λ_i J_i + Skew^LV_T)``
(the market skew over the total first-order skew, equal to ``Skew^mkt / Skew^LV_order1`` for a
pure local-vol model, where the ratio ``d σ̂/d ln S = R^LV × Skew^mkt`` is accurate to 2%) and
``Skew`` is the market skew: measured +5% / +26% (2F, 3M / 1Y; the 1Y residual is the ν = 1.74
short-factor error, −30% on ``SensiX``) and +4% / +5% (1F ω = 2).  Without ``skew_market`` the
unrescaled values are returned and carry the overshoot above; ``mlp_sensitivities`` applies the
same factor.

**``sigma_hat`` override.**  With the market ATMF vol passed as ``sigma_hat`` (the fit: the
order-two level correction is supplied by the market; for the calibrated LSV the ATMF vol *is*
the market's) the sensitivities keep their log-sensitivities and scale with the level:
``SensiX``, ``SensiSpot`` and ``Skew^LV`` by ``σ̂/σ̂_VS``.  The SV skew ``Σ λ_i J_i`` is **not**
rescaled: eq. 8.54 holds at the order-zero VS vol and the gate below measured the plain
order-one skew within 0.6–7% of the simulated skew at every node (ν 0.5–1.74) while a
``(σ̂_VS/σ̂)³`` rescale moved it the wrong way by +5–10% (ν 0.5), +16–29% (ν 1) and +47–84%
(ν 1.74) — the rescale was removed after that measurement.  Consequence for the break-even
fitter's marking mode: the first-order ``SSR = SpotVolCovar/(σ_0 Skew)`` of a naked kernel is
``(σ̂/σ̂_VS) R^{order one}`` with ``R^{order one} ∈ [1, 2]`` (eq. 9.21, ``→ 2`` as ``T → 0``), so an
``ssr_target = 1`` with a ±10% market-skew guard is infeasible at short maturities for any
diffusive model (measured on the reference SSVI: k1 → 20, |ρ_SXi| → 1, both guards binding, the
SSR dial inert) — recorded for the owner in :mod:`volsto.calibration.fit_2f` and SPEC §15.

**Accuracy (naked model, measured in ``tests/test_breakeven.py::test_engine_gate_analytic_vs_
simulation``, Table 8.2 correlations, flat 20% curve, 100k paths, first order with the
simulated ``σ̂``):** ``SpotVolCovar`` and ``VolVar`` within 0.3–1.4% at ν = 0.5, 0.1–3.1% at ν =
1, 1.5–8.4% at ν = 1.74 (3M–2Y); the short-factor ``SensiX`` −2 / −5% (ν = 0.5, 1Y / 2Y), −5 /
−10% (ν = 1), −13 / −22% (ν = 1.74), the long-factor ``SensiY`` within +2% throughout; the
order-one skew within 10% (0.6–7%) with the true skew *smaller* than the order one.  The
quadrature (``n_quad = 64`` outer, ``n_inner = 32`` inner Gauss–Legendre) is verified to 1e-4 on
smooth curves and to about 3e-3 on the kinked PCHIP strip of a stripped surface at 5–10y (the
outer rule does not see the strip's nodes); both far below the O(ν) model error.  Normalisation
note: the owner's ``J`` carries ``σ̂²`` where eq. 8.54 has ``σ̂³``; the code follows eq. 8.54
(``Skew`` in vol per unit log-strike, verified against the mixing-solution skew).

Checked by ``tests/test_breakeven.py`` (``test_engine_first_order_identities``,
``test_engine_affine_in_lambda``, ``test_mlp_matches_first_order_at_small_volvol``,
``test_local_slope_lsv_first_order``, ``test_engine_gate_analytic_vs_simulation``,
``test_engine_argument_validation``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

from volsto.analytics.reparam import BreakEvenParams, to_breakeven
from volsto.config import BergomiParams, SimConfig
from volsto.market.varswap import ForwardVarianceCurve

FloatArray = NDArray[np.float64]

#: relative tolerance on the maturity of a reused :class:`Kernels`
_TOL_T = 1e-12


@dataclass(frozen=True)
class LocalSlope:
    """Local-volatility factor along the ATM path: ``l2(t) = L²(t, F_t)`` and ``b(t) = ∂ ln L²
    / ∂ ln S`` at the forward (the ATM-skew-invariant log-linear approximation), plus the
    optional market ATM skew ``skew_market(T)`` of the surface the leverage was calibrated to
    (module docstring, local branch: with it the spot sensitivity is rescaled to the market
    skew and ``skew`` is the market skew).  ``None`` in the engine means the naked model
    (``l2 ≡ 1``, ``b ≡ 0``)."""

    l2: Callable[[FloatArray], FloatArray]
    b: Callable[[FloatArray], FloatArray]
    label: str = "local"
    skew_market: Callable[[float], float] | None = None


def local_slope_from_leverage(model: Any, h: float = 0.02, surface: Any = None) -> LocalSlope:
    """:class:`LocalSlope` of an :class:`~volsto.models.lsv.LSV` (or any model exposing
    ``leverage(t, S)`` and ``forward_curve``): central log-differences of ``ln L²`` of
    half-width ``h`` in ``ln S`` around the forward ``F_t`` (``h = 0.02`` is the log-moneyness
    window of the slope, not a converged derivative below ``t ≈ 0.05``, module docstring).
    With ``surface`` (the surface the leverage was calibrated to) the market ATM skew is
    attached (its analytic ``atm_skew`` when it has one, else
    :func:`volsto.market.surface.atm_skew_numeric`)."""
    lev = model.leverage
    fc = model.forward_curve

    def l2(t: FloatArray) -> FloatArray:
        t_ = np.atleast_1d(np.asarray(t, dtype=np.float64))
        f = np.asarray(fc.forward(t_), dtype=np.float64)
        return np.asarray([float(lev(float(ti), float(fi))) ** 2 for ti, fi in zip(t_, f)])

    def b(t: FloatArray) -> FloatArray:
        t_ = np.atleast_1d(np.asarray(t, dtype=np.float64))
        f = np.asarray(fc.forward(t_), dtype=np.float64)
        up = np.asarray([float(lev(float(ti), float(fi * np.exp(h)))) for ti, fi in zip(t_, f)])
        dn = np.asarray([float(lev(float(ti), float(fi * np.exp(-h)))) for ti, fi in zip(t_, f)])
        return np.asarray(2.0 * (np.log(up) - np.log(dn)) / (2.0 * h), dtype=np.float64)

    skew_market: Callable[[float], float] | None = None
    if surface is not None:
        fn = getattr(surface, "atm_skew", None)
        if callable(fn):

            def skew_market(T: float) -> float:
                return float(np.asarray(fn(float(T))))

        else:
            from volsto.market.surface import atm_skew_numeric

            def skew_market(T: float) -> float:
                return float(np.asarray(atm_skew_numeric(surface, float(T))).ravel()[0])

    return LocalSlope(l2, b, "leverage", skew_market)


def _gl(a: float, b: float, n: int) -> tuple[FloatArray, FloatArray]:
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w


@dataclass(frozen=True)
class Kernels:
    """The curve-dependent integrals of one maturity: nodes ``t`` and weights ``w`` on
    ``[0, T]``, ``v`` = ``V̂_t`` (order-zero variance along the ATM path), ``W_t``, ``c̃_i(t)``
    (``(n, 2)``), ``W_T``, ``σ̂_T`` (order zero), ``A_i``, ``J_i``, ``B = ∫ V̂ b dt``, the slope
    ``b`` at the nodes and ``BK = ∫ V̂ b W_t/W_T dt`` (the local skew integral)."""

    T: float
    t: FloatArray
    w: FloatArray
    v: FloatArray
    W_t: FloatArray
    c: FloatArray
    W_T: float
    sigma_hat: float
    A: FloatArray
    J: FloatArray
    B: float
    b: FloatArray
    BK: float = 0.0


def kernels(
    ks: tuple[float, float],
    xi0: ForwardVarianceCurve,
    T: float,
    *,
    local: LocalSlope | None = None,
    n_quad: int = 64,
    n_inner: int = 32,
) -> Kernels:
    """The integrals of the module docstring for mean reversions ``ks = (k1, k2)``."""
    if T <= 0:
        raise ValueError("T must be positive")
    t, w = _gl(0.0, T, n_quad)
    xi = np.asarray(xi0.xi0(t), dtype=np.float64)
    if local is None:
        l2 = np.ones_like(t)
        b = np.zeros_like(t)
    else:
        l2 = np.asarray(local.l2(t), dtype=np.float64)
        b = np.asarray(local.b(t), dtype=np.float64)
    v = xi * l2
    W_T = float(np.sum(w * v))
    W_t = np.array([float(xi0.total_variance(ti)) for ti in t]) if local is None else None
    if W_t is None:
        # cumulative ∫ V̂ with the local factor (inner quadrature per node)
        W_t = np.empty_like(t)
        for j, tj in enumerate(t):
            u, wu = _gl(0.0, float(tj), n_inner)
            W_t[j] = float(np.sum(wu * np.asarray(xi0.xi0(u)) * np.asarray(local.l2(u))))  # type: ignore[union-attr]
    c = np.empty((t.size, 2))
    for j, tj in enumerate(t):
        u, wu = _gl(0.0, float(tj), n_inner)
        s = np.sqrt(np.asarray(xi0.xi0(u), dtype=np.float64))
        if local is not None:
            s = s * np.sqrt(np.asarray(local.l2(u), dtype=np.float64))
        for i, k in enumerate(ks):
            c[j, i] = float(np.sum(wu * s * np.exp(-k * (tj - u))))
    sigma_hat = float(np.sqrt(W_T / T))
    A = np.array([float(np.sum(w * v * np.exp(-k * t))) / W_T for k in ks])
    J = np.array([float(np.sum(w * v * c[:, i])) / (2.0 * sigma_hat**3 * T * T) for i in range(2)])
    B = float(np.sum(w * v * b))
    BK = float(np.sum(w * v * b * W_t / W_T))
    return Kernels(float(T), t, w, v, np.asarray(W_t), c, W_T, sigma_hat, A, J, B, b, BK)


def _check_kern(K: Kernels, T: float) -> None:
    if abs(K.T - T) > _TOL_T * max(T, 1.0):
        raise ValueError(f"kern was built for maturity {K.T:g}, not {T:g}")


def sigma_0_of(xi0: ForwardVarianceCurve, local: LocalSlope | None = None) -> float:
    """``σ_0 = L(0, F_0) sqrt(ξ_0^0)`` (SPEC §15: the model's instantaneous spot vol, equal to
    :func:`volsto.analytics.smile_dynamics.initial_vol`; ``L = 1`` for the naked model)."""
    l2 = 1.0 if local is None else float(np.asarray(local.l2(np.array([0.0]))).ravel()[0])
    return float(np.sqrt(float(xi0.xi0(0.0)) * l2))


@dataclass(frozen=True)
class BreakEvenValues:
    """Break-evens of one maturity in absolute vol units (module docstring), with the
    per-quantity standard errors (zero for the closed forms).  ``skew_sv`` / ``skew_lv`` split
    the skew into its SV part ``Σ λ_i J_i`` and its local-vol part (closed forms only)."""

    T: float
    method: str
    sigma_hat: float
    sigma_0: float
    sensi_spot: float
    sensi_x: float
    sensi_y: float
    spot_vol_covar: float
    vol_var: float
    skew: float
    sensi_spot_se: float = 0.0
    sensi_x_se: float = 0.0
    sensi_y_se: float = 0.0
    spot_vol_covar_se: float = 0.0
    vol_var_se: float = 0.0
    skew_se: float = 0.0
    ssr_se: float = 0.0
    skew_sv: float = float("nan")
    skew_lv: float = float("nan")

    @property
    def vovol(self) -> float:
        """Lognormal vol of the ATMF vol ``sqrt(VolVar)/σ̂``."""
        return float(np.sqrt(max(self.vol_var, 0.0)) / self.sigma_hat)

    @property
    def ssr(self) -> float:
        """``SpotVolCovar / (σ_0 Skew)`` (book eq. 9.3 / 12.50)."""
        return float(self.spot_vol_covar / (self.sigma_0 * self.skew)) if self.skew != 0 else np.nan

    @property
    def correl(self) -> float:
        """Spot/ATMF-vol correlation ``SpotVolCovar / sqrt(VolVar)``."""
        return float(self.spot_vol_covar / np.sqrt(self.vol_var)) if self.vol_var > 0 else np.nan

    def as_dict(self) -> dict[str, float | str]:
        d: dict[str, float | str] = {
            "T": self.T,
            "method": self.method,
            "sigma_hat": self.sigma_hat,
            "sigma_0": self.sigma_0,
            "sensi_spot": self.sensi_spot,
            "sensi_x": self.sensi_x,
            "sensi_y": self.sensi_y,
            "spot_vol_covar": self.spot_vol_covar,
            "vol_var": self.vol_var,
            "skew": self.skew,
            "skew_sv": self.skew_sv,
            "skew_lv": self.skew_lv,
            "vovol": self.vovol,
            "correl": self.correl,
            "ssr": self.ssr,
        }
        if self.method != "first_order":
            d.update(
                {
                    "sensi_spot_se": self.sensi_spot_se,
                    "sensi_x_se": self.sensi_x_se,
                    "sensi_y_se": self.sensi_y_se,
                    "spot_vol_covar_se": self.spot_vol_covar_se,
                    "vol_var_se": self.vol_var_se,
                    "skew_se": self.skew_se,
                    "ssr_se": self.ssr_se,
                }
            )
        return d


def _be(p: BergomiParams | BreakEvenParams) -> BreakEvenParams:
    return p if isinstance(p, BreakEvenParams) else to_breakeven(p)


def _skew_terms(
    K: Kernels, be: BreakEvenParams, local: LocalSlope | None, sig: float
) -> tuple[float, float, float]:
    """``(skew_sv, skew_lv, factor)`` of the module docstring at the prefactor ``sig``: the SV
    skew ``Σ λ_i J_i`` (eq. 8.54 at the order-zero VS vol — the prefactor override does NOT
    touch it: the gate measured the rescaled variant off by +47–84% at ν = 1.74 while the plain
    order-one skew is within 6–7% of the mixing skew), the local skew (the market skew minus
    the SV skew when the slope carries ``skew_market``, else ``σ̂ B_K / (2 W_T)``) and the market
    rescale ``factor`` of the spot sensitivity (1 without ``skew_market``)."""
    lam = np.array([be.lambda1, be.lambda2])
    sv0 = float(lam @ K.J)  # at the order-zero σ̂ (eq. 8.54)
    skew_sv = sv0
    if local is None:
        return skew_sv, 0.0, 1.0
    lv0 = K.BK / (2.0 * K.sigma_hat * K.T)
    if local.skew_market is None:
        return skew_sv, sig * K.BK / (2.0 * K.W_T), 1.0
    s_mkt = float(local.skew_market(K.T))
    total0 = sv0 + lv0
    factor = s_mkt / total0 if total0 != 0.0 else float("nan")
    return skew_sv, s_mkt - skew_sv, factor


def first_order_breakevens(
    params: BergomiParams | BreakEvenParams,
    xi0: ForwardVarianceCurve,
    T: float,
    *,
    local: LocalSlope | None = None,
    sigma_0: float | None = None,
    sigma_hat: float | None = None,
    kern: Kernels | None = None,
    n_quad: int = 64,
) -> BreakEvenValues:
    """The first-order closed forms (module docstring) at ``X_0 = Y_0 = 0``; ``kern`` can be
    passed to reuse the kernels of ``(k1, k2, xi0, T)`` across parameter values (they do not
    depend on ``ω``, ``λ`` or ``χ``; a kernel of another maturity is refused).  ``sigma_0``
    defaults to :func:`sigma_0_of` (``L(0, F_0) sqrt(ξ_0^0)``); ``sigma_hat`` overrides the
    order-zero ATMF vol used as the prefactor (the market ATMF vol in the fit: the order-two
    level correction of the ATMF vol is then supplied by the market — a 3% effect on ``VolVar``
    at ν = 0.5, more at large ν): ``SensiX``, ``SensiSpot`` and the local skew scale with it;
    the SV skew keeps the order-zero VS-vol prefactor of eq. 8.54 (validated against the
    mixing skew; module docstring)."""
    be = _be(params)
    if kern is not None:
        _check_kern(kern, T)
    K = kern if kern is not None else kernels((be.k1, be.k2), xi0, T, local=local, n_quad=n_quad)
    s0 = sigma_0_of(xi0, local) if sigma_0 is None else float(sigma_0)
    sig = K.sigma_hat if sigma_hat is None else float(sigma_hat)
    om = np.array([be.omega1, be.omega2])
    sensi = 0.5 * om * K.A * sig  # ∂σ̂/∂X_i
    skew_sv, skew_lv, factor = _skew_terms(K, be, local, sig)
    sensi_spot = s0 * sig * K.B / (2.0 * K.W_T) * factor
    rho = np.array([be.rho_SX, be.rho_SY])
    svc = float(sensi_spot + rho @ sensi)
    volvar = float(
        sensi_spot**2
        + sensi[0] ** 2
        + sensi[1] ** 2
        + 2.0 * rho[0] * sensi_spot * sensi[0]
        + 2.0 * rho[1] * sensi_spot * sensi[1]
        + 2.0 * be.rho_XY * sensi[0] * sensi[1]
    )
    return BreakEvenValues(
        float(T),
        "first_order",
        sig,
        s0,
        float(sensi_spot),
        float(sensi[0]),
        float(sensi[1]),
        svc,
        volvar,
        skew_sv + skew_lv,
        skew_sv=skew_sv,
        skew_lv=skew_lv,
    )


def mlp_atmf_variance(
    params: BergomiParams | BreakEvenParams,
    xi0: ForwardVarianceCurve,
    T: float,
    *,
    x0: Sequence[float] = (0.0, 0.0),
    d_ln_s0: float = 0.0,
    local: LocalSlope | None = None,
    kern: Kernels | None = None,
    n_quad: int = 64,
) -> float:
    """``σ̂_T²`` by the most-likely-path closed form of the module docstring, for an initial
    factor state ``x0 = (X_0, Y_0)`` (two entries also for a one-factor model, the second one
    idle) and a spot shift ``d_ln_s0`` (the local factor ``L²(t, S)`` moves along with the
    path: ``ln L̂²_t → + b(t) d_ln_s0``; the naked model is spot-homogeneous)."""
    be = _be(params)
    x0_ = np.asarray(x0, dtype=np.float64)
    if x0_.shape != (2,):
        raise ValueError("x0 must have two entries (X_0, Y_0)")
    if kern is not None:
        _check_kern(kern, T)
    K = kern if kern is not None else kernels((be.k1, be.k2), xi0, T, local=local, n_quad=n_quad)
    om = np.array([be.omega1, be.omega2])
    lam = np.array([be.lambda1, be.lambda2])
    ks = np.array([be.k1, be.k2])
    lc = K.c @ lam  # Σ λ_i c̃_i(t)
    frac = 1.0 - K.W_t / K.W_T
    expo = (
        (np.exp(-np.outer(K.t, ks)) @ (om * x0_))
        + 0.5 * lc
        + K.b * lc * frac
        - 0.5 * lc**2 / K.W_T
        + K.b * d_ln_s0
    )
    return float(np.sum(K.w * K.v * np.exp(expo)) / T)


def mlp_sensitivities(
    params: BergomiParams | BreakEvenParams,
    xi0: ForwardVarianceCurve,
    T: float,
    *,
    eps: float = 1e-4,
    local: LocalSlope | None = None,
    sigma_0: float | None = None,
    n_quad: int = 64,
) -> BreakEvenValues:
    """The sensitivities by central differences of :func:`mlp_atmf_variance` in ``ln S_0``,
    ``X_0``, ``Y_0`` (an internal consistency check of the first-order forms: they agree to
    ``O(ω²)``; ``Skew`` and the market rescale of the spot sensitivity are taken from the
    first-order form at the MLP level)."""
    be = _be(params)
    K = kernels((be.k1, be.k2), xi0, T, local=local, n_quad=n_quad)
    s0 = sigma_0_of(xi0, local) if sigma_0 is None else float(sigma_0)

    def sig(x: Sequence[float], ds: float) -> float:
        return float(np.sqrt(mlp_atmf_variance(be, xi0, T, x0=x, d_ln_s0=ds, local=local, kern=K)))

    base = sig((0.0, 0.0), 0.0)
    d_spot = (sig((0.0, 0.0), eps) - sig((0.0, 0.0), -eps)) / (2 * eps)
    d_x = (sig((eps, 0.0), 0.0) - sig((-eps, 0.0), 0.0)) / (2 * eps)
    d_y = (sig((0.0, eps), 0.0) - sig((0.0, -eps), 0.0)) / (2 * eps)
    skew_sv, skew_lv, factor = _skew_terms(K, be, local, K.sigma_hat)  # order-zero level
    sensi_spot = s0 * d_spot * factor
    rho = np.array([be.rho_SX, be.rho_SY])
    svc = sensi_spot + rho[0] * d_x + rho[1] * d_y
    volvar = (
        sensi_spot**2
        + d_x**2
        + d_y**2
        + 2 * rho[0] * sensi_spot * d_x
        + 2 * rho[1] * sensi_spot * d_y
        + 2 * be.rho_XY * d_x * d_y
    )
    return BreakEvenValues(
        float(T),
        "mlp_fd",
        base,
        s0,
        sensi_spot,
        d_x,
        d_y,
        svc,
        volvar,
        skew_sv + skew_lv,
        skew_sv=skew_sv,
        skew_lv=skew_lv,
    )


def simulated_breakevens(
    model: Any,
    T: float,
    *,
    sim: SimConfig,
    eps: float = 0.05,
    sigma_0: float | None = None,
    skew_h: float = 0.01,
    scheme: str = "central",
) -> BreakEvenValues:
    """Break-evens by simulation: the state-bump partials of the ATMF vol of the M7 Part 1
    machinery (:func:`volsto.analytics.smile_dynamics.atmf_vol_of_vol`: central differences in
    ``ln S_0`` with the leverage held, ``X_0``, ``Y_0``, common random numbers) combined as in
    the module docstring, with delta-method standard errors from the per-path influence
    samples (the partials are estimated on common paths, so ``SpotVolCovar``'s error is that
    of ``σ_0 p_0 + Σ ρ_i p_i`` itself, not the root-sum-square of the partials' errors — 8%
    smaller on the cached 2F LSV, 20% larger for the naked Table 8.2 model whose factor
    partials are positively correlated); the skew from ±``skew_h`` strikes on the same paths;
    ``ssr`` carries its own standard error (``ssr_se``) and equals ``ATMFVolOfVol.ssr``.
    ``σ_0`` defaults to the model's instantaneous vol ``L(0, S_0) sqrt(ξ_0^0)`` (the state
    covariance's own); a caller's ``sigma_0`` replaces it in the spot slot of the covariance,
    so ``SpotVolCovar``, ``VolVar``, ``correl`` and ``ssr`` derive from one covariance."""
    from volsto.analytics.smile_dynamics import (
        SCHEMES,
        _atmf_quotes,
        _check,
        _grid_and_draws,
        _partials_from_quotes,
        _quadratic_form,
        _ratio,
        _se,
        _skew,
        factor_loadings,
        state_covariance,
    )

    _check(scheme, SCHEMES, "scheme")
    if eps <= 0 or T <= 0 or skew_h <= 0:
        raise ValueError("eps, T and skew_h must be positive")
    rho_s, _ = factor_loadings(model)
    nf = int(rho_s.size)
    Ts = np.array([float(T)])
    mc, grid, draws = _grid_and_draws(model, Ts, sim)
    base = _atmf_quotes(model, Ts, np.array([-skew_h, 0.0, skew_h]), mc, grid, draws)
    skew, skew_s = _skew(base, skew_h)
    vals, samps = _partials_from_quotes(model, Ts, eps, scheme, mc, grid, draws, base)
    p = np.asarray(vals[0], dtype=np.float64)  # (∂σ̂/∂ln S, ∂σ̂/∂X1[, ∂σ̂/∂X2])
    ps = np.asarray(samps[:, 0, :], dtype=np.float64)
    cov = np.array(state_covariance(model), dtype=np.float64)
    s0_model = float(np.sqrt(cov[0, 0]))
    s0 = s0_model if sigma_0 is None else float(sigma_0)
    if s0 != s0_model:
        cov[0, 0] = s0 * s0
        cov[0, 1:] = cov[1:, 0] = rho_s * s0
    volvar, volvar_se, _ = _quadratic_form(p, ps, cov)
    w = np.concatenate(([s0], rho_s))  # SpotVolCovar = σ_0 p_0 + Σ ρ_i p_i
    svc_s = ps @ w
    svc = float(w @ p)
    svc_se = _se(svc_s)
    _ssr, ssr_se = _ratio(svc / s0, float(skew[0]), svc_s / s0, skew_s[:, 0])
    se = np.array([_se(ps[:, j]) for j in range(ps.shape[1])])
    return BreakEvenValues(
        float(T),
        "simulation",
        float(base.iv[0, 1]),
        s0,
        float(s0 * p[0]),
        float(p[1]) if nf >= 1 else 0.0,
        float(p[2]) if nf >= 2 else 0.0,
        svc,
        float(volvar),
        float(skew[0]),
        float(s0 * se[0]),
        float(se[1]) if nf >= 1 else 0.0,
        float(se[2]) if nf >= 2 else 0.0,
        svc_se,
        float(volvar_se),
        _se(skew_s[:, 0]),
        float(ssr_se),
    )


def breakeven_table(
    params: BergomiParams | BreakEvenParams,
    xi0: ForwardVarianceCurve,
    Ts: ArrayLike,
    *,
    local: LocalSlope | None = None,
    sigma_0: float | None = None,
) -> pd.DataFrame:
    """First-order break-evens per maturity as a frame (closed forms: no wall clock, nothing
    calibrated)."""
    rows = [
        first_order_breakevens(params, xi0, float(T), local=local, sigma_0=sigma_0).as_dict()
        for T in np.atleast_1d(np.asarray(Ts, dtype=np.float64))
    ]
    return pd.DataFrame(rows)


__all__ = [
    "BreakEvenValues",
    "Kernels",
    "LocalSlope",
    "breakeven_table",
    "first_order_breakevens",
    "kernels",
    "local_slope_from_leverage",
    "mlp_atmf_variance",
    "mlp_sensitivities",
    "sigma_0_of",
    "simulated_breakevens",
]
