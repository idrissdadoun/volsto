"""Break-even reparametrisation of the two-factor Bergomi model (SPEC §15 Part 3, M7 addendum).

Book parameters ``(k1, k2, ν, θ, ρ_SX, ρ_SY, ρ_XY)`` ↔ break-even parameters
``(k1, k2, ω1, ω2, λ1, λ2, χ)``::

    ω1 = 2 ν α_θ (1 − θ)        ω2 = 2 ν α_θ θ
    λ1 = ρ_SX ω1                 λ2 = ρ_SY ω2
    χ  = (ρ_XY − ρ_SX ρ_SY) / (sqrt(1 − ρ_SX²) sqrt(1 − ρ_SY²))

with ``α_θ = ((1−θ)² + θ² + 2 ρ_XY θ (1−θ))^{−1/2}`` (book eq. 7.29).  ``ω_i`` is the loading
of factor ``i`` on ``ln ξ_t^t`` (``ω x_t^t = ω1 X¹ + ω2 X²`` at ``t = 0``, eq. 7.34 with
``ω = 2ν``), ``λ_i`` the spot-correlated part of that loading — the order-one ATMF skew and the
spot/vol break-even depend on the model through ``(k1, k2, λ1, λ2)`` only — and ``χ`` the
residual correlation of the two factors once their common spot component is removed (the
inverse of book eq. 8.56).  Inverse: ``θ = ω2 / (ω1 + ω2)``, ``ρ_Si = λ_i / ω_i``, ``ρ_XY =
ρ_SX ρ_SY + χ sqrt(1 − ρ_SX²) sqrt(1 − ρ_SY²)``, ``ν = (ω1 + ω2) / (2 α_θ)``.  Both directions
are checked round-trip to 1e-12 by ``tests/test_breakeven.py::test_reparam_round_trip`` on the
*identifiable* parameters: when a factor is absent (``θ ∈ {0, 1}``, i.e. ``ω_i = 0``) its
spot correlation and ``ρ_XY`` do not enter the model, so ``χ`` is set to 0 and the round trip
returns ``ρ_XY = 0`` and ``ρ_Si = 0`` for that factor whatever the input carried.  Neither
class enforces ``k1 > k2`` ("X1 is the short factor" is a naming convention, not a constraint).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from volsto.config import BergomiParams

_TOL = 1e-12


@dataclass(frozen=True)
class BreakEvenParams:
    """``(k1, k2, ω1, ω2, λ1, λ2, χ)`` — see the module docstring."""

    k1: float
    k2: float
    omega1: float
    omega2: float
    lambda1: float
    lambda2: float
    chi: float

    def __post_init__(self) -> None:
        if self.k1 <= 0 or self.k2 <= 0:
            raise ValueError("k1 and k2 must be positive")
        if self.omega1 < 0 or self.omega2 < 0 or self.omega1 + self.omega2 <= 0:
            raise ValueError("omega1, omega2 must be non-negative with a positive sum")
        if abs(self.lambda1) > self.omega1 * (1 + _TOL) + _TOL:
            raise ValueError("|lambda1| <= omega1 is required (|rho_SX| <= 1)")
        if abs(self.lambda2) > self.omega2 * (1 + _TOL) + _TOL:
            raise ValueError("|lambda2| <= omega2 is required (|rho_SY| <= 1)")
        if not -1.0 <= self.chi <= 1.0:
            raise ValueError("chi must lie in [-1, 1]")

    @property
    def rho_SX(self) -> float:
        return 0.0 if self.omega1 == 0.0 else self.lambda1 / self.omega1

    @property
    def rho_SY(self) -> float:
        return 0.0 if self.omega2 == 0.0 else self.lambda2 / self.omega2

    @property
    def rho_XY(self) -> float:
        r1, r2 = self.rho_SX, self.rho_SY
        return r1 * r2 + self.chi * math.sqrt(max(1.0 - r1 * r1, 0.0)) * math.sqrt(
            max(1.0 - r2 * r2, 0.0)
        )

    @property
    def theta(self) -> float:
        return self.omega2 / (self.omega1 + self.omega2)

    @property
    def alpha_theta(self) -> float:
        th = self.theta
        return 1.0 / math.sqrt((1 - th) ** 2 + th * th + 2.0 * self.rho_XY * th * (1 - th))

    @property
    def nu(self) -> float:
        return (self.omega1 + self.omega2) / (2.0 * self.alpha_theta)

    def to_book(self) -> BergomiParams:
        """Inverse map to the book parametrisation (:class:`BergomiParams`)."""
        return BergomiParams(
            self.nu, self.theta, self.k1, self.k2, self.rho_XY, self.rho_SX, self.rho_SY
        )


def alpha_theta_of(theta: float, rho_xy: float) -> float:
    """``α_θ`` from ``θ`` and ``ρ_XY`` (book eq. 7.29)."""
    return 1.0 / math.sqrt((1 - theta) ** 2 + theta * theta + 2.0 * rho_xy * theta * (1 - theta))


def to_breakeven(p: BergomiParams) -> BreakEvenParams:
    """Forward map ``(ν, θ, k1, k2, ρ_XY, ρ_SX, ρ_SY) → (k1, k2, ω1, ω2, λ1, λ2, χ)``.
    ``χ`` is undefined when a spot correlation is ±1 (set to 0 there: the factor is then
    perfectly spot-driven and ``ρ_XY`` is fixed by PSD) and irrelevant when a factor is absent
    (``θ ∈ {0, 1}``: set to 0, module docstring)."""
    a = alpha_theta_of(p.theta, p.rho12)
    om1 = 2.0 * p.nu * a * (1.0 - p.theta)
    om2 = 2.0 * p.nu * a * p.theta
    if om1 == 0.0 or om2 == 0.0:
        return BreakEvenParams(p.k1, p.k2, om1, om2, p.rho_SX1 * om1, p.rho_SX2 * om2, 0.0)
    d = math.sqrt(max(1.0 - p.rho_SX1**2, 0.0)) * math.sqrt(max(1.0 - p.rho_SX2**2, 0.0))
    chi = (p.rho12 - p.rho_SX1 * p.rho_SX2) / d if d > _TOL else 0.0
    chi = min(1.0, max(-1.0, chi))
    return BreakEvenParams(p.k1, p.k2, om1, om2, p.rho_SX1 * om1, p.rho_SX2 * om2, chi)


def from_breakeven(
    k1: float, k2: float, omega1: float, omega2: float, lambda1: float, lambda2: float, chi: float
) -> BergomiParams:
    """Inverse map (see :meth:`BreakEvenParams.to_book`)."""
    return BreakEvenParams(k1, k2, omega1, omega2, lambda1, lambda2, chi).to_book()


__all__ = ["BreakEvenParams", "alpha_theta_of", "from_breakeven", "to_breakeven"]
