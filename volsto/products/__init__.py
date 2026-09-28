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
from volsto.products.gap import (
    GAP_FUNCTIONS,
    GapReport,
    GapSpec,
    LevelFactors,
    conditional_value_at_level,
    conservative_shift,
    month_grid,
)
from volsto.products.seasoning import (
    RealisedCashFlow,
    RealisedHistory,
    Replay,
    Settled,
    SettledCash,
    replay,
    season,
)
from volsto.products.vanilla import DigitalOption, EuropeanOption
from volsto.products.variance import (
    FVA,
    ForwardVarianceSwap,
    VarianceOption,
    VarianceSwap,
    VolSwap,
)
from volsto.products.vko import VolKnockOutPut

__all__ = [
    "FVA",
    "GAP_FUNCTIONS",
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
    "GapReport",
    "GapSpec",
    "KIPutLeg",
    "KnockInOption",
    "KnockOutOption",
    "KnockOutVarianceSwap",
    "LevelFactors",
    "Napoleon",
    "NoTouch",
    "OneTouch",
    "Phoenix",
    "Portfolio",
    "Product",
    "RealisedCashFlow",
    "RealisedHistory",
    "Replay",
    "ReverseCliquet",
    "Settled",
    "SettledCash",
    "StatisticLeg",
    "UpVar",
    "VarianceOption",
    "VarianceSwap",
    "VolKnockOutPut",
    "VolSwap",
    "bridge_step_survival",
    "conditional_value_at_level",
    "conservative_shift",
    "continuous_survival_weight",
    "daily_schedule",
    "first_hit_index",
    "forward_start_strip",
    "month_grid",
    "parse_cp",
    "replay",
    "season",
    "shift_times",
    "uniform_schedule",
]
