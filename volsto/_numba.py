"""Typed shim around numba's ``njit`` so mypy keeps the decorated function's signature.

Numba ships no type stubs; decorating with ``numba.njit`` directly makes the function ``Any``
under ``mypy --strict``.  This module re-exports a typed decorator factory.  Kernels stay pure
array functions (SPEC §11); all validation lives in the Python wrappers that call them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, ParamSpec, TypeVar

import numba

P = ParamSpec("P")
R = TypeVar("R")


def njit(**options: Any) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return ``numba.njit(**options)`` typed as a signature-preserving decorator."""

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        compiled: Callable[P, R] = numba.njit(**options)(fn)
        return compiled

    return decorator


prange: Any = numba.prange
