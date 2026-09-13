"""Products (SPEC §6)."""

from __future__ import annotations

from volsto.products.base import CashFlow, Product, daily_schedule, parse_cp, uniform_schedule
from volsto.products.cliquet import AccumulatedSumOption, AdditiveCliquet, Napoleon, ReverseCliquet
from volsto.products.forward_start import (
    ForwardStartOption,
    ForwardStartStraddle,
    forward_start_strip,
)
from volsto.products.vanilla import DigitalOption, EuropeanOption
from volsto.products.variance import FVA, ForwardVarianceSwap, VarianceSwap, VolSwap

__all__ = [
    "FVA",
    "AccumulatedSumOption",
    "AdditiveCliquet",
    "CashFlow",
    "DigitalOption",
    "EuropeanOption",
    "ForwardStartOption",
    "ForwardStartStraddle",
    "ForwardVarianceSwap",
    "Napoleon",
    "Product",
    "ReverseCliquet",
    "VarianceSwap",
    "VolSwap",
    "daily_schedule",
    "forward_start_strip",
    "parse_cp",
    "uniform_schedule",
]
