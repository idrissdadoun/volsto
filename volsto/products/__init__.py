"""Products (SPEC §6)."""

from __future__ import annotations

from volsto.products.autocall import (
    Autocall,
    AutocallStatistic,
    BondLeg,
    ConditionalDigital,
    CouponLeg,
    KIPutLeg,
    Phoenix,
)
from volsto.products.barrier import (
    Digital,
    KnockInOption,
    KnockOutOption,
    NoTouch,
    OneTouch,
    bridge_step_survival,
    continuous_survival_weight,
    first_hit_index,
)
from volsto.products.base import (
    CashFlow,
    Portfolio,
    Product,
    daily_schedule,
    parse_cp,
    shift_times,
    uniform_schedule,
)
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
    "Autocall",
    "AutocallStatistic",
    "BondLeg",
    "CashFlow",
    "ConditionalDigital",
    "ConditionalVarianceSwap",
    "ConvexitySpread",
    "CouponLeg",
    "Digital",
    "DigitalOption",
    "DownVar",
    "EuropeanOption",
    "ForwardStartOption",
    "ForwardStartStraddle",
    "ForwardVarianceSwap",
    "KIPutLeg",
    "KnockInOption",
    "KnockOutOption",
    "KnockOutVarianceSwap",
    "Napoleon",
    "NoTouch",
    "OneTouch",
    "Phoenix",
    "Portfolio",
    "Product",
    "ReverseCliquet",
    "StatisticLeg",
    "UpVar",
    "VarianceSwap",
    "VolKnockOutPut",
    "VolSwap",
    "bridge_step_survival",
    "continuous_survival_weight",
    "daily_schedule",
    "first_hit_index",
    "forward_start_strip",
    "parse_cp",
    "shift_times",
    "uniform_schedule",
]
