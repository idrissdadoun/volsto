"""Vol-of-vol and model-parameter sensitivities (SPEC v2 §7.9).

Partial sensitivities to each of ``(ν, θ, k1, k2, ρ12, ρ_SX1, ρ_SX2)`` by central differences
with recalibration of the leverage (each bumped parameter set is a cache entry), and for ``ν``
also the sticky-leverage variant (``L`` held, the kernel re-parametrised).  Default bumps
(:func:`~volsto.risk.engine.default_params_bump`): 5% relative for ``ν, k1, k2``, 0.05 absolute
for ``θ`` and the correlations.  When one side of the central bump leaves the admissible set
(PSD correlation matrix of ``(W^S, W¹, W²)``, ``|ρ| ≤ 1``, ``θ ∈ [0, 1]`` — ``θ = 0`` of the
one-factor model is a boundary) the difference is one-sided towards the admissible side; the
bump is halved (up to four times) only when neither side is admissible.  The achieved bump and
the scheme are reported.  Output: one row per parameter with value, stderr, bump and scheme
under both variants.  Checked by ``tests/test_risk_profiles.py`` (structure and the zero
response of Black–Scholes; one-sided θ at the boundary; halving on a correlation beyond
``|ρ| ≤ 1``) and by the M5 budget run under the LSV.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from volsto.products.base import Product
from volsto.risk.engine import RiskEngine, RiskState, Sensitivity, default_params_bump

PARAMS: tuple[str, ...] = ("nu", "theta", "k1", "k2", "rho12", "rho_SX1", "rho_SX2")


def _valid(state: RiskState, name: str, value: float, label: str) -> RiskState | None:
    try:
        return state.with_params(label=label, **{name: value})
    except ValueError:
        return None


def _bump_terms(
    state: RiskState, name: str, h: float, mode: str, max_halvings: int = 4
) -> tuple[list[tuple[RiskState, str, float]], float, str]:
    """Difference terms for ``∂P/∂name``: central when both ``base ± h`` are admissible, else
    one-sided towards the admissible side (a parameter at a boundary, e.g. ``θ = 0`` of the
    one-factor model); ``h`` is halved while neither side is admissible."""
    base = float(getattr(state.spec.model, name))
    for _ in range(max_halvings + 1):
        up = _valid(state, name, base + h, f"{name}+{h:g}")
        dn = _valid(state, name, base - h, f"{name}-{h:g}")
        if up is not None and dn is not None:
            return [(up, mode, 1.0 / (2.0 * h)), (dn, mode, -1.0 / (2.0 * h))], h, "central"
        if up is not None:
            return [(up, mode, 1.0 / h), (state, mode, -1.0 / h)], h, "forward"
        if dn is not None:
            return [(state, mode, 1.0 / h), (dn, mode, -1.0 / h)], h, "backward"
        h *= 0.5
    raise ValueError(f"no admissible bump for {name} from {base:g}")


def parameter_sensitivity(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    name: str,
    size: float | None = None,
    variant: str = "recalibrated",
) -> Sensitivity:
    """``∂P/∂param`` by central differences (one-sided at a parameter boundary) under a leverage
    variant (``"recalibrated"`` or ``"sticky_leverage"``); ``Sensitivity.scheme`` records which."""
    if variant not in ("recalibrated", "sticky_leverage"):
        raise ValueError("variant must be 'recalibrated' or 'sticky_leverage'")
    h = default_params_bump(name, state.spec.model) if size is None else float(size)
    mode = "recalibrate" if variant == "recalibrated" else "sticky_leverage"
    terms, h, scheme = _bump_terms(state, name, h, mode)
    return engine.combination(
        f"d/d{name}[{variant}]",
        product,
        terms,
        unit=f"per unit {name}",
        size=h,
        scheme=scheme,
        extra={"variant": variant},
    )


def parameter_sensitivities(
    engine: RiskEngine,
    product: Product,
    state: RiskState,
    params: Sequence[str] = PARAMS,
    sizes: dict[str, float] | None = None,
    sticky_nu: bool = True,
) -> pd.DataFrame:
    """One row per parameter: recalibrated value / stderr, the sticky-leverage value / stderr
    (``ν`` only unless ``sticky_nu`` is False, others ``nan``), the achieved bump and scheme."""
    rows = []
    for name in params:
        size = (sizes or {}).get(name)
        s = parameter_sensitivity(engine, product, state, name, size, "recalibrated")
        row = {
            "param": name,
            "bump": s.size,
            "scheme": s.scheme,
            "value": s.value,
            "stderr": s.stderr,
            "value_sticky": np.nan,
            "stderr_sticky": np.nan,
        }
        if name == "nu" and sticky_nu:
            st = parameter_sensitivity(engine, product, state, name, size, "sticky_leverage")
            row["value_sticky"], row["stderr_sticky"] = st.value, st.stderr
        rows.append(row)
    return pd.DataFrame(rows)
