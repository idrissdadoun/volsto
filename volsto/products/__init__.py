"""Products (SPEC §6)."""

from __future__ import annotations

from volsto.products.base import Product, daily_schedule, parse_cp, uniform_schedule
from volsto.products.vanilla import DigitalOption, EuropeanOption
from volsto.products.variance import ForwardVarianceSwap, VarianceSwap, VolSwap

__all__ = [
    "DigitalOption",
    "EuropeanOption",
    "ForwardVarianceSwap",
    "Product",
    "VarianceSwap",
    "VolSwap",
    "daily_schedule",
    "parse_cp",
    "uniform_schedule",
]
