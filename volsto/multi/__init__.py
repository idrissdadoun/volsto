"""Multi-asset layer (the dispersion study, SPEC §8.5): correlated Brownians driving the
library's factor-free single-asset models (Black–Scholes, local vol), a path container with one
:class:`~volsto.engine.paths.PathSet` per asset, the Monte Carlo loop over it, and the basket /
dispersion products — the palladium (call on dispersion), basket options and straddles, the
single-name straddle package and the variance dispersion.

Products access the paths only through :class:`~volsto.multi.paths.MultiPathSet`; the single-
asset code is untouched (``PathSet`` keeps ``n_assets = 1``: the second underlying lives in a
second container, as SPEC §3.1 allowed).

The local correlation model (SPEC §8.7, M12) lives beside the constant-correlation one: the
affine correlation family ``ρ(λ) = (1 − λ)·R_low + λ·R_high``
(:class:`~volsto.multi.family.CorrelationFamily`) and its draws
(:class:`~volsto.multi.lc_draws.LocalCorrelationDraws`).
"""

from volsto.multi.analytics import (
    basket_vol,
    gaussian_dispersion_moments,
    gaussian_palladium_call,
    gaussian_palladium_forward,
    gaussian_straddle_dispersion,
    implied_correlation,
    pairwise_mean_correlation,
)
from volsto.multi.draws import CorrelatedDraws
from volsto.multi.family import CorrelationFamily
from volsto.multi.lc_draws import LocalCorrelationDraws
from volsto.multi.mc import MultiAssetMonteCarlo
from volsto.multi.model import MultiAssetModel
from volsto.multi.paths import MultiPathSet
from volsto.multi.products import (
    BasketOption,
    BasketStraddle,
    MultiAssetProduct,
    MultiPortfolio,
    Palladium,
    SingleNameStraddles,
    VarianceDispersion,
    dispersion_straddles,
)

__all__ = [
    "BasketOption",
    "BasketStraddle",
    "CorrelatedDraws",
    "CorrelationFamily",
    "LocalCorrelationDraws",
    "MultiAssetModel",
    "MultiAssetMonteCarlo",
    "MultiAssetProduct",
    "MultiPathSet",
    "MultiPortfolio",
    "Palladium",
    "SingleNameStraddles",
    "VarianceDispersion",
    "basket_vol",
    "dispersion_straddles",
    "gaussian_dispersion_moments",
    "gaussian_palladium_call",
    "gaussian_palladium_forward",
    "gaussian_straddle_dispersion",
    "implied_correlation",
    "pairwise_mean_correlation",
]
