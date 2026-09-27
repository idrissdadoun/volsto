"""SABRW — the desk's in-house SABR smile, per expiry (owner's documents, SPEC §15 Part 3, *Step 0
on (e)SSVI surfaces and the desk's SABRW*).

**Smile.** SABR with ``β = 1`` read through its short-maturity (Berestycki–Busca–Florent) limit:
the implied vol at log-moneyness ``k = ln(K/F_T)`` is ``σ̂(k) = k / I(k)`` with
``I(k) = ∫_0^k dz / g(z)`` and the equivalent local vol

    g(x) = sqrt(σ² + 2ρνσ h(x) + ν² h(x)²).

Plain SABR has ``h(x) = x``.  SABRW replaces it by a continuous piecewise-linear ``h`` with slope 1
on a central zone ``(x_Td, x_Tu)`` and the slopes ``T_d, EX_d`` on the downside, ``T_u, EX_u`` on
the upside (:func:`h`)::

    h(x) = EX_d (x − x_EXd) + T_d (x_EXd − x_Td) + x_Td    x < x_EXd
           T_d (x − x_Td) + x_Td                          x_EXd ≤ x < x_Td
           x                                              x_Td ≤ x ≤ x_Tu
           T_u (x − x_Tu) + x_Tu                          x_Tu < x ≤ x_EXu
           EX_u (x − x_EXu) + T_u (x_EXu − x_Tu) + x_Tu   x > x_EXu

so ``I(k)`` is a sum of zone integrals, each closed-form (:func:`zone_integral`)::

    G(a, b) = [asinh((ρ + (ν/σ) h(b)) / √(1 − ρ²)) − asinh((ρ + (ν/σ) h(a)) / √(1 − ρ²))] / (h′ ν)

with ``h′`` the slope of the zone holding ``[a, b]`` (the documents write ``h′((a + b)/2)``).
In the central zone the smile is plain SABR, so the ATM level, skew and curvature are SABR's
(:func:`atm_triplet`) and the step-0 reduction returns the fitted ``(ρ, ν)`` exactly
(:func:`breakeven_from_triplet`).

**Zones** (:func:`zones`): ``x_Td = N⁻¹(−Δ_put) σ_ref √T`` with ``Δ_put = −15 %`` and ``σ_ref =
30 %`` (the documents' "≈": the ``½σ²T`` term of the put delta is left out — volsto's reading),
``x_Tu`` the plain SABR smile's minimum on ``k > 0`` (:func:`sabr_smile_minimum`; volsto's
reading of the documents' ``arginf σ̂^SABR``), ``x_EXd = 4 x_Td`` and ``x_EXu = 2 x_Tu``.

**Fit** (:func:`fit_sabrw`): weighted non-linear least squares on one expiry's quotes,
``Σ_i ((σ̂(k_i) − σ̂_mid,i) / ε_i)²``, under bounds (:data:`BOUNDS`).  The documents' solver is
Levenberg–Marquardt with bounds; volsto uses scipy's trust-region reflective method (``trf``, the
bounded least-squares solver it has).  The weights ``ε_i`` are the caller's (the documents do not
say which; volsto's measurement uses the quotes' half bid–ask spread in vol, SPEC §15 Part 3).

Not arbitrage-free by design (the documents' own account: calendar, butterfly), and not used as a
pricing surface: its place is the step-0 input of the marking calibration.  Checked by
``tests/test_sabrw.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares, minimize_scalar
from scipy.stats import norm

FloatArray = NDArray[np.float64]

ZONE_DELTA_PUT: Final[float] = -0.15
"""``Δ_put`` of the downside zone edge ``x_Td`` (a 15-delta put)."""
ZONE_SIGMA_REF: Final[float] = 0.30
"""Reference vol of ``x_Td``."""
ZONE_EXD_FACTOR: Final[float] = 4.0
"""``x_EXd = ZONE_EXD_FACTOR · x_Td``."""
ZONE_EXU_FACTOR: Final[float] = 2.0
"""``x_EXu = ZONE_EXU_FACTOR · x_Tu`` (the documents' "≈ 2")."""
SMILE_MIN_SEARCH: Final[float] = 3.0
"""Upper end of the search for the SABR smile's minimum on ``k > 0`` (in units of ``√T``)."""
SMILE_MIN_FLOOR: Final[float] = 0.01
"""Smallest ``x_Tu`` (in units of ``√T``): the zone edge when the SABR smile has no minimum on
``k > 0`` (``rho >= 0``), flagged by :func:`fit_sabrw`."""
BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "sigma": (0.01, 3.0),
    "rho": (-0.999, 0.999),
    "nu": (1e-3, 20.0),
    "t_d": (0.05, 10.0),
    "t_u": (0.05, 10.0),
    "ex_d": (0.05, 10.0),
    "ex_u": (0.05, 10.0),
}
"""Box of the fit (:func:`fit_sabrw`)."""
AT_BOUND_REL: Final[float] = 2e-3
"""A fitted parameter within this fraction of its box width from a bound is flagged."""
MIN_ZONE_QUOTES: Final[int] = 3
"""Fewest quotes a wing zone needs for its slope to be fitted; a slope with fewer is held at its
initial value (1: plain SABR extrapolation) and flagged — the documents' "wide open
configuration" otherwise leaves it to the solver's noise."""
PARAM_NAMES: Final[tuple[str, ...]] = ("sigma", "rho", "nu", "t_d", "t_u", "ex_d", "ex_u")


@dataclass(frozen=True)
class SabrwParams:
    """One expiry's SABRW parameters: ``sigma`` (the ATM level ``σ``), ``rho``, ``nu`` (the
    lognormal vol of vol), and the wing slopes ``t_d, t_u, ex_d, ex_u`` of ``h`` (1 everywhere is
    plain SABR)."""

    sigma: float
    rho: float
    nu: float
    t_d: float = 1.0
    t_u: float = 1.0
    ex_d: float = 1.0
    ex_u: float = 1.0

    def __post_init__(self) -> None:
        if not self.sigma > 0.0:
            raise ValueError("sigma must be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")
        if not self.nu > 0.0:
            raise ValueError("nu must be positive")
        for name in ("t_d", "t_u", "ex_d", "ex_u"):
            if not getattr(self, name) > 0.0:
                raise ValueError(f"{name} must be positive")

    def as_array(self) -> FloatArray:
        return np.array([getattr(self, n) for n in PARAM_NAMES], dtype=np.float64)

    @classmethod
    def from_array(cls, x: ArrayLike) -> SabrwParams:
        v = np.asarray(x, dtype=np.float64).ravel()
        if v.size != len(PARAM_NAMES):
            raise ValueError(f"expected {len(PARAM_NAMES)} parameters")
        return cls(*(float(a) for a in v))


@dataclass(frozen=True)
class SabrwZones:
    """The zone edges, ``x_exd < x_td < 0 < x_tu < x_exu``."""

    x_exd: float
    x_td: float
    x_tu: float
    x_exu: float

    def __post_init__(self) -> None:
        if not self.x_exd < self.x_td < 0.0 < self.x_tu < self.x_exu:
            raise ValueError("need x_exd < x_td < 0 < x_tu < x_exu")


def sabr_bbf_vol(k: ArrayLike, sigma: float, rho: float, nu: float) -> FloatArray:
    """The plain short-maturity SABR (``β = 1``) smile ``k / ∫_0^k dz/√(σ² + 2ρνσz + ν²z²)``
    (``σ`` at ``k = 0``, its limit)."""
    kk = np.asarray(k, dtype=np.float64)
    c = np.sqrt(1.0 - rho * rho)
    integral = (np.arcsinh((rho + nu / sigma * kk) / c) - np.arcsinh(rho / c)) / nu
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(kk) > 1e-12, kk / integral, sigma)
    return np.asarray(out, dtype=np.float64)


def sabr_smile_minimum(sigma: float, rho: float, nu: float, T: float) -> tuple[float, bool]:
    """``(x, interior)``: the minimum of the plain SABR smile on ``0 < k ≤ SMILE_MIN_SEARCH √T``,
    and whether it is an interior minimum.  Without one (``ρ ≥ 0``: the smile rises from the
    money) ``x`` is ``SMILE_MIN_FLOOR √T``."""
    if not T > 0.0:
        raise ValueError("T must be positive")
    lo, hi = SMILE_MIN_FLOOR * np.sqrt(T), SMILE_MIN_SEARCH * np.sqrt(T)
    res = minimize_scalar(
        lambda x: float(sabr_bbf_vol(np.array([x]), sigma, rho, nu)[0]),
        bounds=(lo, hi),
        method="bounded",
        options={"xatol": 1e-9 * max(1.0, hi)},
    )
    x = float(res.x)
    interior = lo * (1.0 + 1e-6) < x < hi * (1.0 - 1e-6)
    if not interior:
        return float(lo), False
    return x, True


def zones(params: SabrwParams, T: float) -> tuple[SabrwZones, bool]:
    """The zone edges of ``params`` at maturity ``T`` (module docstring) and whether ``x_Tu`` is
    an interior minimum of the SABR smile."""
    if not T > 0.0:
        raise ValueError("T must be positive")
    x_td = float(norm.ppf(-ZONE_DELTA_PUT)) * ZONE_SIGMA_REF * np.sqrt(T)
    x_td = -abs(x_td)
    x_tu, interior = sabr_smile_minimum(params.sigma, params.rho, params.nu, T)
    return (
        SabrwZones(ZONE_EXD_FACTOR * x_td, x_td, x_tu, ZONE_EXU_FACTOR * x_tu),
        interior,
    )


def h(x: ArrayLike, params: SabrwParams, z: SabrwZones) -> FloatArray:
    """The continuous piecewise-linear ``h`` of the module docstring."""
    xx = np.asarray(x, dtype=np.float64)
    down_exd = params.t_d * (z.x_exd - z.x_td) + z.x_td
    up_exu = params.t_u * (z.x_exu - z.x_tu) + z.x_tu
    out = np.select(
        [xx < z.x_exd, xx < z.x_td, xx <= z.x_tu, xx <= z.x_exu],
        [
            params.ex_d * (xx - z.x_exd) + down_exd,
            params.t_d * (xx - z.x_td) + z.x_td,
            xx,
            params.t_u * (xx - z.x_tu) + z.x_tu,
        ],
        default=params.ex_u * (xx - z.x_exu) + up_exu,
    )
    return np.asarray(out, dtype=np.float64)


def zone_integral(
    a: ArrayLike, b: ArrayLike, slope: ArrayLike, params: SabrwParams, z: SabrwZones
) -> FloatArray:
    """``G(a, b) = ∫_a^b dz / g(z)`` for ``[a, b]`` inside one zone of slope ``slope``."""
    c = np.sqrt(1.0 - params.rho * params.rho)
    s = params.nu / params.sigma
    ha, hb = h(a, params, z), h(b, params, z)
    num = np.arcsinh((params.rho + s * hb) / c) - np.arcsinh((params.rho + s * ha) / c)
    return np.asarray(num / (np.asarray(slope, dtype=np.float64) * params.nu), dtype=np.float64)


def local_vol_integral(k: ArrayLike, params: SabrwParams, z: SabrwZones) -> FloatArray:
    """``I(k) = ∫_0^k dz / g(z)``, summed over the zones between 0 and ``k``."""
    kk = np.asarray(k, dtype=np.float64)
    out = np.zeros_like(kk)
    # upside: (0, x_tu] slope 1, (x_tu, x_exu] slope t_u, beyond slope ex_u
    up = [(0.0, z.x_tu, 1.0), (z.x_tu, z.x_exu, params.t_u), (z.x_exu, np.inf, params.ex_u)]
    down = [(z.x_td, 0.0, 1.0), (z.x_exd, z.x_td, params.t_d), (-np.inf, z.x_exd, params.ex_d)]
    pos = kk > 0.0
    for lo, hi, slope in up:
        a = np.full(int(pos.sum()), lo)
        b = np.minimum(kk[pos], hi)
        live = b > a
        seg = np.zeros_like(b)
        seg[live] = zone_integral(a[live], b[live], slope, params, z)
        out[pos] += seg
    neg = kk < 0.0
    for lo, hi, slope in down:
        b = np.full(int(neg.sum()), hi)
        a = np.maximum(kk[neg], lo)
        live = a < b
        seg = np.zeros_like(a)
        seg[live] = zone_integral(a[live], b[live], slope, params, z)
        out[neg] -= seg
    return out


def sabrw_vol(k: ArrayLike, params: SabrwParams, T: float) -> FloatArray:
    """The SABRW implied vol at log-moneyness ``k`` and maturity ``T`` (``σ`` at ``k = 0``)."""
    z, _ = zones(params, T)
    kk = np.asarray(k, dtype=np.float64)
    integral = local_vol_integral(kk, params, z)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.where(np.abs(kk) > 1e-12, kk / integral, params.sigma)
    return np.asarray(out, dtype=np.float64)


def atm_triplet(params: SabrwParams) -> tuple[float, float, float]:
    """``(atf, skw, cvx)``: the ATM level, skew ``∂σ̂/∂k`` and curvature ``∂²σ̂/∂k²`` of the
    central zone (plain SABR): ``σ``, ``½ρν`` and ``ν²(1 − 3ρ²/2)/(3σ)`` — the Taylor expansion of
    the owner's documents."""
    p = params
    return (
        float(p.sigma),
        0.5 * p.rho * p.nu,
        p.nu * p.nu * (1.0 - 1.5 * p.rho * p.rho) / (3.0 * p.sigma),
    )


def breakeven_from_triplet(atf: float, skw: float, cvx: float) -> tuple[float, float]:
    """``(ν, ρ)`` from the ATM triplet: ``ν = √(6 skw² + 3 atf cvx)``, ``ρ = 2 skw / ν`` (the
    step-0 reduction at ``p = 0``; the inverse of :func:`atm_triplet`)."""
    rad = 6.0 * skw * skw + 3.0 * atf * cvx
    if not rad > 0.0:
        raise ValueError("the triplet has no SABR reading (6 skw² + 3 atf cvx <= 0)")
    nu = float(np.sqrt(rad))
    return nu, 2.0 * skw / nu


def zone_counts(k: ArrayLike, z: SabrwZones) -> dict[str, int]:
    """Quotes per wing zone, keyed by the zone's slope name (``ex_d``, ``t_d``, ``t_u``,
    ``ex_u``)."""
    kk = np.asarray(k, dtype=np.float64)
    return {
        "ex_d": int((kk < z.x_exd).sum()),
        "t_d": int(((kk >= z.x_exd) & (kk < z.x_td)).sum()),
        "t_u": int(((kk > z.x_tu) & (kk <= z.x_exu)).sum()),
        "ex_u": int((kk > z.x_exu).sum()),
    }


@dataclass(frozen=True)
class SabrwFit:
    """One expiry's fit (:func:`fit_sabrw`): the parameters, the zones, the weighted and plain
    RMS of the residuals in vol points, the number of quotes, the slopes held (their zone holds
    fewer than :data:`MIN_ZONE_QUOTES` quotes), the parameters at a bound, and the flags."""

    T: float
    params: SabrwParams
    zones: SabrwZones
    rms_vp: float
    weighted_rms: float
    n: int
    held: tuple[str, ...]
    at_bound: tuple[str, ...]
    flags: tuple[str, ...] = field(default_factory=tuple)


def fit_sabrw(
    k: ArrayLike,
    iv: ArrayLike,
    weights: ArrayLike,
    T: float,
    *,
    init: SabrwParams | None = None,
    max_nfev: int = 2000,
) -> SabrwFit:
    """Fit one expiry's quotes (log-moneyness ``k``, mid implied vols ``iv``) by weighted least
    squares ``Σ ((σ̂(k_i) − iv_i) / ε_i)²`` with ``ε = weights`` (module docstring), inside
    :data:`BOUNDS`.  ``init`` defaults to the quote nearest the money for ``σ``, ``ρ = −0.7``,
    ``ν = 1`` and unit slopes.  A wing slope whose zone (at the initial parameters) holds fewer
    than :data:`MIN_ZONE_QUOTES` quotes is held at its initial value; a change of the zones'
    coverage during the fit, a parameter within :data:`AT_BOUND_REL` of its box, a smile without
    an interior minimum and a solver failure are flagged."""
    kk = np.asarray(k, dtype=np.float64).ravel()
    vv = np.asarray(iv, dtype=np.float64).ravel()
    ee = np.asarray(weights, dtype=np.float64).ravel()
    if not (kk.shape == vv.shape == ee.shape) or kk.size < len(PARAM_NAMES):
        raise ValueError("k, iv and weights must match and hold at least 7 quotes")
    if not np.all(ee > 0.0):
        raise ValueError("weights must be positive")
    if not T > 0.0:
        raise ValueError("T must be positive")
    x0 = (
        init if init is not None else SabrwParams(float(vv[int(np.argmin(np.abs(kk)))]), -0.7, 1.0)
    ).as_array()
    lo = np.array([BOUNDS[n][0] for n in PARAM_NAMES])
    hi = np.array([BOUNDS[n][1] for n in PARAM_NAMES])
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
    z0, _ = zones(SabrwParams.from_array(x0), T)
    counts0 = zone_counts(kk, z0)
    held = tuple(n for n in ("t_d", "t_u", "ex_d", "ex_u") if counts0[n] < MIN_ZONE_QUOTES)
    free = np.array([n not in held for n in PARAM_NAMES])

    def full(xf: FloatArray) -> FloatArray:
        x = x0.copy()
        x[free] = xf
        return x

    def resid(xf: FloatArray) -> FloatArray:
        return (sabrw_vol(kk, SabrwParams.from_array(full(xf)), T) - vv) / ee

    sol = least_squares(
        resid, x0[free], bounds=(lo[free], hi[free]), method="trf", max_nfev=max_nfev
    )
    x_fit = full(sol.x)
    params = SabrwParams.from_array(x_fit)
    z, interior = zones(params, T)
    model = sabrw_vol(kk, params, T)
    width = hi - lo
    at_bound = tuple(
        n
        for n, v, a, b, w, f in zip(PARAM_NAMES, x_fit, lo, hi, width, free, strict=True)
        if f and (v - a < AT_BOUND_REL * w or b - v < AT_BOUND_REL * w)
    )
    flags: list[str] = []
    if held:
        flags.append(
            f"slopes held at their initial value (zone with < {MIN_ZONE_QUOTES} quotes): "
            f"{ {n: counts0[n] for n in held} }"
        )
    counts = zone_counts(kk, z)
    moved = [
        n
        for n in ("t_d", "t_u", "ex_d", "ex_u")
        if (counts[n] < MIN_ZONE_QUOTES) != (counts0[n] < MIN_ZONE_QUOTES)
    ]
    if moved:
        flags.append(f"zone coverage changed during the fit for {moved}: {counts0} -> {counts}")
    if at_bound:
        flags.append(f"parameters at a bound: {list(at_bound)}")
    if not interior:
        flags.append("the SABR smile has no minimum on k > 0: x_Tu at its floor")
    if not sol.success:
        flags.append(f"solver: {sol.message}")
    return SabrwFit(
        T=float(T),
        params=params,
        zones=z,
        rms_vp=float(100.0 * np.sqrt(np.mean((model - vv) ** 2))),
        weighted_rms=float(np.sqrt(np.mean(((model - vv) / ee) ** 2))),
        n=int(kk.size),
        held=held,
        at_bound=at_bound,
        flags=tuple(flags),
    )


@dataclass(frozen=True)
class SabrwTermStructure:
    """Step 0's triplet source from per-expiry SABRW fits (the ``Step0Triplets`` protocol of
    :mod:`volsto.calibration.targets`; SPEC §15 Part 3).  At maturity ``T`` the ATM level is
    ``atm_vol(T)`` — the pricing surface's: per-expiry SABR levels are not calendar-monotone —,
    and the skew and curvature are the fits' central-zone terms (:func:`atm_triplet`),
    interpolated linearly in ``T`` in the desk's 365-day quotes ``Smile_365 = 200 √T skw`` and
    ``Convex_365 = 100 T cvx`` and held flat beyond the fitted expiries (volsto's rule: the
    documents give none).  Fits on one maturity (SPX and SPXW on the same date) keep the one with
    more quotes."""

    T: tuple[float, ...]
    smile_365: tuple[float, ...]
    convex_365: tuple[float, ...]
    atm_vol: Callable[[float], float]
    label: str

    @classmethod
    def from_fits(
        cls, fits: Sequence[SabrwFit], atm_vol: Callable[[float], float]
    ) -> SabrwTermStructure:
        by_t: dict[float, SabrwFit] = {}
        for f in fits:
            kept = by_t.get(f.T)
            if kept is None or f.n > kept.n:
                by_t[f.T] = f
        if len(by_t) < 2:
            raise ValueError("a term structure needs fits on at least two maturities")
        ts = sorted(by_t)
        smile, convex = [], []
        for t in ts:
            _, skw, cvx = atm_triplet(by_t[t].params)
            smile.append(200.0 * float(np.sqrt(t)) * skw)
            convex.append(100.0 * t * cvx)
        label = f"SABRW fits of {len(ts)} expiries ({ts[0]:.3f}y to {ts[-1]:.3f}y)"
        return cls(tuple(ts), tuple(smile), tuple(convex), atm_vol, label)

    def triplet(self, T: float) -> tuple[float, float, float]:
        """``(atf, ∂σ̂/∂k, ∂²σ̂/∂k²)`` at ``T``."""
        t = float(T)
        if not t > 0.0:
            raise ValueError("T must be positive")
        sm = float(np.interp(t, self.T, self.smile_365))
        cv = float(np.interp(t, self.T, self.convex_365))
        return float(self.atm_vol(t)), sm / (200.0 * float(np.sqrt(t))), cv / (100.0 * t)


__all__ = [
    "AT_BOUND_REL",
    "BOUNDS",
    "MIN_ZONE_QUOTES",
    "PARAM_NAMES",
    "SMILE_MIN_FLOOR",
    "SMILE_MIN_SEARCH",
    "ZONE_DELTA_PUT",
    "ZONE_EXD_FACTOR",
    "ZONE_EXU_FACTOR",
    "ZONE_SIGMA_REF",
    "SabrwFit",
    "SabrwParams",
    "SabrwTermStructure",
    "SabrwZones",
    "atm_triplet",
    "breakeven_from_triplet",
    "fit_sabrw",
    "h",
    "local_vol_integral",
    "sabr_bbf_vol",
    "sabr_smile_minimum",
    "sabrw_vol",
    "zone_counts",
    "zone_integral",
    "zones",
]
