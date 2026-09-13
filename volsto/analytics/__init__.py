"""Closed forms and semi-analytic tools (SPEC §3.3, §7).

The mixing solution lives in :mod:`volsto.analytics.mixing` (imported explicitly: it depends on
the Bergomi model, which itself uses the closed forms here)."""

from __future__ import annotations

from volsto.analytics.bergomi import (
    alpha_theta,
    atmf_skew_order1,
    atmf_skew_order1_flat,
    chi,
    cov_x_diag,
    cov_xi_diag,
    eta_u,
    forward_vs_vol_of_vol_flat,
    ssr_order1,
    ssr_order1_flat,
    var_integrated_variance,
    vs_vol_of_vol,
    vs_vol_of_vol_flat,
)
from volsto.analytics.conditional_variance import (
    FairStrike,
    VKOReport,
    fair_strike,
    lsv_minus_lv,
    mean_and_stderr,
    pair_average,
    ratio_of_means,
    strike_differential,
    vko_report,
)
from volsto.analytics.forward_smile import (
    ForwardSmile,
    ForwardVolComparison,
    forward_atm_vol,
    forward_ratio,
    forward_smile,
    forward_smile_from_prices,
    forward_vol_comparison,
    put_wing_table,
)

__all__ = [
    "FairStrike",
    "ForwardSmile",
    "ForwardVolComparison",
    "VKOReport",
    "alpha_theta",
    "atmf_skew_order1",
    "atmf_skew_order1_flat",
    "chi",
    "cov_x_diag",
    "cov_xi_diag",
    "eta_u",
    "fair_strike",
    "forward_atm_vol",
    "forward_ratio",
    "forward_smile",
    "forward_smile_from_prices",
    "forward_vol_comparison",
    "forward_vs_vol_of_vol_flat",
    "lsv_minus_lv",
    "mean_and_stderr",
    "pair_average",
    "put_wing_table",
    "ratio_of_means",
    "ssr_order1",
    "ssr_order1_flat",
    "strike_differential",
    "var_integrated_variance",
    "vko_report",
    "vs_vol_of_vol",
    "vs_vol_of_vol_flat",
]
