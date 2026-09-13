"""Products (SPEC §6)."""

from __future__ import annotations

from volsto.products.base import CashFlow, Product, daily_schedule, parse_cp, uniform_schedule
from volsto.products.cliquet import AccumulatedSumOption, AdditiveCliquet, Napoleon, ReverseCliquet
from volsto.products.conditional_variance import (
    ConditionalVarianceSwap,
    ConvexitySpread,
    DownVar,
    KnockOutVarianceSwap,
    StatisticLeg,
    UpVar,
)
from volsto.products.forward_start import (
    ForwardStartOption,
    ForwardStartStraddle,
    forward_start_strip,
)
from volsto.products.vanilla import DigitalOption, EuropeanOption
from volsto.products.variance import FVA, ForwardVarianceSwap, VarianceSwap, VolSwap
from volsto.products.vko import VolKnockOutPut

__all__ = [
    "FVA",
    "AccumulatedSumOption",
    "AdditiveCliquet",
    "CashFlow",
    "ConditionalVarianceSwap",
    "ConvexitySpread",
    "DigitalOption",
    "DownVar",
    "EuropeanOption",
    "ForwardStartOption",
    "ForwardStartStraddle",
    "ForwardVarianceSwap",
    "KnockOutVarianceSwap",
    "Napoleon",
    "Product",
    "ReverseCliquet",
    "StatisticLeg",
    "UpVar",
    "VarianceSwap",
    "VolKnockOutPut",
    "VolSwap",
    "daily_schedule",
    "forward_start_strip",
    "parse_cp",
    "uniform_schedule",
]
