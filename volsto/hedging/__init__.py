"""Hedging framework (SPEC §8, M8): conditional pricing by regression, hedge instruments,
payoff-derived Greek-targeting strategies, the hedger loop and the reports."""

from __future__ import annotations

from volsto.hedging.hedger import (
    Costs,
    Hedger,
    HedgeResult,
    PricingContext,
    RecalibrationRule,
    Schedule,
)
from volsto.hedging.instruments import (
    CapCallStrip,
    ConditionalVarianceSwap,
    Digital,
    ForwardStartRiskReversal,
    ForwardStartStraddle,
    ForwardStartVanilla,
    ForwardVarianceSwap,
    HedgeInstrument,
    Spot,
    StaticPortfolio,
    Vanilla,
    VarianceSwap,
    VolSwap,
    option_strip,
)
from volsto.hedging.pricing import Bump, ConditionalPricer
from volsto.hedging.report import HedgeReport, hedge_report
from volsto.hedging.state import HedgeState, hedge_state
from volsto.hedging.strategies import (
    PRESETS,
    CustomStrategy,
    GreekTargetStrategy,
    PresetContext,
    Target,
    default_strategy,
)

__all__ = [
    "PRESETS",
    "Bump",
    "CapCallStrip",
    "ConditionalPricer",
    "ConditionalVarianceSwap",
    "Costs",
    "CustomStrategy",
    "Digital",
    "ForwardStartRiskReversal",
    "ForwardStartStraddle",
    "ForwardStartVanilla",
    "ForwardVarianceSwap",
    "GreekTargetStrategy",
    "HedgeInstrument",
    "HedgeReport",
    "HedgeResult",
    "HedgeState",
    "Hedger",
    "PresetContext",
    "PricingContext",
    "RecalibrationRule",
    "Schedule",
    "Spot",
    "StaticPortfolio",
    "Target",
    "Vanilla",
    "VarianceSwap",
    "VolSwap",
    "default_strategy",
    "hedge_report",
    "hedge_state",
    "option_strip",
]
