"""Finite-difference cross-checks (SPEC §9.1): the 1F LSV ADI solver."""

from __future__ import annotations

from volsto.pde.lsv1f import (
    HV_THETA,
    LSV1FPDE,
    ConvergenceLine,
    ConvergenceTable,
    PDEResult,
    convergence_table,
)

__all__ = [
    "HV_THETA",
    "LSV1FPDE",
    "ConvergenceLine",
    "ConvergenceTable",
    "PDEResult",
    "convergence_table",
]
