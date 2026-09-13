"""Control variates (SPEC §5): vanilla and variance-swap controls with sample-estimated β.

For payoff ``X`` and controls ``C_j`` with known (discounted) expectations ``c_j``, the adjusted
estimator is ``Y = X − Σ_j β_j (C_j − c_j)`` with ``β = Var(C)⁻¹ Cov(C, X)`` estimated on the
sample (the O(1/n) bias is negligible at the path counts used).  The variance-reduction factor
``Var(X)/Var(Y)`` is reported.  Checked by ``tests/test_engine.py::test_control_variate``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from volsto.market.varswap import varswap_strike
from volsto.products.vanilla import EuropeanOption
from volsto.products.variance import VarianceSwap

if TYPE_CHECKING:
    from volsto.engine.grid import FixingIndex
    from volsto.engine.paths import PathSet
    from volsto.market.surface import ImpliedSurface
    from volsto.products.base import Product

FloatArray = NDArray[np.float64]


class ControlVariate(ABC):
    """A product with a known discounted expectation under the pricing model."""

    def __init__(self, product: Product, expectation: float) -> None:
        if not np.isfinite(expectation):
            raise ValueError("control expectation must be finite")
        self.product = product
        self.expectation = float(expectation)

    @property
    def fixing_times(self) -> FloatArray:
        return self.product.fixing_times

    def payoff(self, paths: PathSet, idx: FixingIndex) -> FloatArray:
        return self.product.payoff(paths, idx)

    @abstractmethod
    def __repr__(self) -> str: ...


class VanillaControl(ControlVariate):
    """European option whose price is known (from the target surface or a closed form)."""

    @classmethod
    def from_surface(
        cls, surface: ImpliedSurface, strike: float, maturity: float, cp: int = 1
    ) -> VanillaControl:
        """Expectation = surface price (valid for LV/LSV calibrated to that surface, SPEC §5)."""
        option = EuropeanOption(strike, maturity, cp, surface.discount)
        return cls(option, float(surface.price(strike, maturity, cp)))

    def __repr__(self) -> str:
        return f"VanillaControl({self.product!r}, expectation={self.expectation:.6g})"


class VarianceControl(ControlVariate):
    """Floating leg of a variance swap with the replication strike as expectation."""

    @classmethod
    def from_surface(cls, surface: ImpliedSurface, swap: VarianceSwap) -> VarianceControl:
        """Expectation ``DF(T) · notional · K_var(T)`` with the log-contract strike (SPEC §2.4)."""
        if swap.start > 0:
            raise ValueError("use a spot-start variance swap as control")
        leg = swap.floating_leg()
        k_var = varswap_strike(surface, swap.maturity)
        expectation = float(surface.discount.df(swap.maturity)) * swap.notional * k_var
        return cls(leg, expectation)

    def __repr__(self) -> str:
        return f"VarianceControl({self.product!r}, expectation={self.expectation:.6g})"


@dataclass(frozen=True)
class CVReport:
    """Diagnostics of a control-variate adjustment."""

    betas: FloatArray
    variance_reduction: float
    raw_mean: float
    raw_stderr: float

    def __repr__(self) -> str:
        return (
            f"CVReport(betas={np.round(self.betas, 4).tolist()}, "
            f"variance_reduction={self.variance_reduction:.3g}x, "
            f"raw={self.raw_mean:.6g} ± {self.raw_stderr:.2g})"
        )


def apply_controls(
    samples: FloatArray, control_samples: FloatArray, expectations: FloatArray
) -> tuple[FloatArray, CVReport]:
    """Return adjusted samples ``Y`` and a report; inputs are independent samples."""
    x = np.asarray(samples, dtype=np.float64)
    c = np.asarray(control_samples, dtype=np.float64)
    e = np.asarray(expectations, dtype=np.float64)
    n = x.size
    if c.shape != (n, e.size):
        raise ValueError("control_samples must have shape (n_samples, n_controls)")
    if n < 3:
        raise ValueError("need at least 3 samples for a control-variate fit")
    raw_mean = float(x.mean())
    raw_std = float(x.std(ddof=1))
    dc = c - c.mean(axis=0)
    dx = x - raw_mean
    cov_cc = dc.T @ dc / (n - 1)
    cov_cx = dc.T @ dx / (n - 1)
    try:
        beta = np.linalg.solve(cov_cc, cov_cx)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(cov_cc, cov_cx, rcond=None)[0]
    y = x - (c - e) @ beta
    adj_std = float(y.std(ddof=1))
    vr = (raw_std / adj_std) ** 2 if adj_std > 0 else float("inf")
    report = CVReport(
        betas=np.asarray(beta, dtype=np.float64),
        variance_reduction=float(vr),
        raw_mean=raw_mean,
        raw_stderr=raw_std / np.sqrt(n),
    )
    return np.asarray(y, dtype=np.float64), report
