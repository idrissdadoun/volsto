# volsto

Stochastic-volatility pricing library for single-underlying light-exotic and exotic parameter
studies: Black–Scholes, Dupire local vol, two-factor lognormal Bergomi forward-variance model,
LSV with particle-calibrated leverage, Monte Carlo with common random numbers, and viewers over a
precomputed parameter cache.  The full design is in [SPEC.md](SPEC.md).

Status: **M1** (market layer, BS, local vol, MC engine, vanilla / variance products).  The
quickstart below prices with local vol; the LSV calibration (M3) and cliquet (M4) steps land with
their milestones.

## Install

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
```

## Quickstart (M1)

```python
import numpy as np
from volsto.config import SimConfig
from volsto.engine import MonteCarlo, VanillaControl
from volsto.market import LocalVolSurface, implied_vol, varswap_strike, xi0_curve
from volsto.market.loaders import load_ssvi_surface
from volsto.models import LocalVol
from volsto.products import EuropeanOption, VarianceSwap

surface = load_ssvi_surface("configs/surfaces/reference_ssvi.yaml")   # SSVI + curves
local_vol = LocalVolSurface.from_implied(surface)                      # Dupire (Gatheral 1.10)
print(local_vol.check_positive())

model = LocalVol(local_vol)
mc = MonteCarlo(SimConfig(n_paths=100_000, dt_max=1 / 365, seed=1))

T, K = 1.0, float(surface.forward(1.0))
call = EuropeanOption(K, T, "call", surface.discount)
ctrl = VanillaControl.from_surface(surface, 0.95 * K, T)               # analytic control
res = mc.price(call, model, controls=[ctrl])
iv = implied_vol(res.mean, surface.forward(T), K, T, 1, surface.discount.df(T))
print(res, f"implied {iv:.4%} vs surface {surface.implied_vol(K, T):.4%}", res.cv)

vs = VarianceSwap.daily(T, strike=0.0, discount=surface.discount)       # floating leg
fair = mc.price(vs, model)
print("MC fair var strike", fair, "replication", varswap_strike(surface, T))
print("xi0(1y)", xi0_curve(surface, 3.0)(1.0))
```

Every Monte Carlo number is a `PriceResult(mean ± stderr)`; the library never returns a bare
float for a simulated quantity.

## Development

```bash
.venv/bin/black volsto tests && .venv/bin/ruff check volsto tests && .venv/bin/mypy && .venv/bin/pytest -m "not slow"
```

`pytest -m slow` runs the full-size checks.  `docs/Stochastic_Volatility_Modeling.pdf` (Bergomi)
is git-ignored on purpose; docstrings cite its equation numbers.
