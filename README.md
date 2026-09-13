# volsto

Stochastic-volatility pricing library for single-underlying light-exotic and exotic parameter
studies: Black–Scholes, Dupire local vol, two-factor lognormal Bergomi forward-variance model,
LSV with particle-calibrated leverage, Monte Carlo with common random numbers, and viewers over a
precomputed parameter cache.  The full design is in [SPEC.md](SPEC.md).

Status: **M3** — market layer, BS, local vol, MC engine, vanilla / variance products (M1); the
two-factor lognormal Bergomi forward-variance model with exact factor stepping, its closed forms
and the mixing-solution smile (M2); particle-method leverage calibration, the LSV model, §4.2
repricing diagnostics and the content-addressed leverage cache (M3).  The cliquet of the SPEC §11
quickstart lands with M4; the quickstart below calibrates the 1F LSV and prices a vanilla and a
variance swap with it.

## Install

Python 3.12 or later is required (numpy ≥ 2.5 dropped 3.11; mypy strict runs against 3.12).

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
```

## Quickstart

```python
from volsto.calibration import LeverageCache, reprice_surface
from volsto.calibration.cache import build_market
from volsto.config import CalibrationSpec, SimConfig, load_yaml
from volsto.engine import MonteCarlo
from volsto.market import implied_vol
from volsto.products import EuropeanOption, VarianceSwap

# 1F LSV (omega = 3, kappa = 1.5, rho = -0.7) on the reference SSVI surface: surface, curves,
# Bergomi kernel, particle settings, step schedule and scheme all come from one YAML spec
spec = load_yaml("configs/studies/lsv_reference_1f.yaml", CalibrationSpec)
cache = LeverageCache("cache")                      # content-addressed; the only entry point
model, _ = cache.get_or_calibrate(spec)             # ~30 s on a miss (2e5 particles, 3y), instant on a hit
_, surface, _ = build_market(spec)

mc = MonteCarlo(SimConfig(n_paths=200_000, seed=1))  # default schedule + scheme, as in calibration
T, K = 1.0, float(surface.forward(1.0))
call = mc.price(EuropeanOption(K, T, "call", surface.discount), model)
iv = implied_vol(call.mean, K, K, T, 1, surface.discount.df(T))
print(call, f"LSV implied {iv:.4%} vs surface {surface.implied_vol(K, T):.4%}")
print(mc.price(VarianceSwap.daily(T, 0.0, surface.discount), model))  # fair variance strike

report = reprice_surface(model, surface, SimConfig(n_paths=400_000, seed=2))  # §4.2 table
print(report.summary())
```

Local vol only (no calibration):

```python
from volsto.market import LocalVolSurface
from volsto.market.loaders import load_ssvi_surface
from volsto.models import LocalVol

surface = load_ssvi_surface("configs/surfaces/reference_ssvi.yaml")
model = LocalVol(LocalVolSurface.from_implied(surface))              # Dupire (Gatheral 1.10)
```

Every Monte Carlo number is a `PriceResult(mean ± stderr)`; the library never returns a bare
float for a simulated quantity.

Discretisation defaults (`SimConfig`): step schedule 1/1460 below 3m, 1/365 to 2y, 1/250 after,
and Platen's explicit weak order-2 spot step (`weak_order2=True`).  On the reference surface the
1m ATM local-vol repricing bias is 0.03 vol points at dt = 1/365 versus 0.36 for plain log-Euler;
`volsto.engine.refinement_study` reproduces this under common random numbers.  Plain log-Euler,
time-averaged variance and a weak predictor-corrector remain available as options.

## Bergomi two-factor model (M2)

```python
from volsto.analytics import atmf_skew_order1_flat, ssr_order1_flat, vs_vol_of_vol_flat
from volsto.analytics.mixing import mixing_atmf_skew, mixing_smile
from volsto.config import BergomiParams, load_yaml
from volsto.market import ForwardCurve, ForwardVarianceCurve
from volsto.models import BergomiSV

params = load_yaml("configs/models/bergomi_table_8_2.yaml", BergomiParams)   # book Table 8.2
model = BergomiSV(params, ForwardVarianceCurve.flat(0.04), ForwardCurve.flat(100.0, 0.0, 0.0))
skew, atmf_vol, err = mixing_atmf_skew(model, 1.0, h=0.02, n_paths=400_000)  # naked 2F smile
print(skew, atmf_skew_order1_flat(params, 1.0), ssr_order1_flat(params, 1.0), vs_vol_of_vol_flat(params, 1.0))
```

`BergomiParams.one_factor(omega, kappa, rho)` is the 1F model of the earlier studies (θ = 0).

## Calibration notes (M3)

`ParticleConfig` defaults were chosen by measurement on the reference surface: local-linear
kernel regression of E[ξ|S] (Nadaraya–Watson carries an h² m′f′/f design bias that skewed the
±10% repricing by 0.3 vol points), a plug-in ½h²m″ bias correction, a 2000-particle window floor
in the tails, and a saturating log-quadratic tail extrapolation (the flat rule mis-priced the 3m
+30% call by 1 vol point and variance swaps by 0.3).  Calibration and pricing share the kernel
step for step: every leverage lookup inside a step uses the slice at the step start.

## Development

```bash
.venv/bin/black volsto tests && .venv/bin/ruff check volsto tests && .venv/bin/mypy && .venv/bin/pytest -m "not slow"
```

`pytest -m slow` runs the full-size checks.  `docs/Stochastic_Volatility_Modeling.pdf` (Bergomi)
is git-ignored on purpose; docstrings cite its equation numbers.
