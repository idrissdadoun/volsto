# volsto

Stochastic-volatility pricing library for single-underlying light-exotic and exotic parameter
studies: Black–Scholes, Dupire local vol, two-factor lognormal Bergomi forward-variance model,
LSV with particle-calibrated leverage, Monte Carlo with common random numbers, and viewers over a
precomputed parameter cache.  The full design is in [SPEC.md](SPEC.md).

Status: **M4** — market layer, BS, local vol, MC engine, vanilla / variance products (M1); the
two-factor lognormal Bergomi forward-variance model with exact factor stepping, its closed forms
and the mixing-solution smile (M2); particle-method leverage calibration, the LSV model, §4.2
repricing diagnostics and the content-addressed leverage cache (M3); the HistoricalData.net
option-chain importer and SSVI/eSSVI fitter (M3b); forward-start options, the FVA, the cliquet
family with exact decompositions, forward-smile analytics and the headline study runner (M4);
the second-order SV spot step, the adaptive particle-regression grid and the restored
1/1460–1/365–1/250 schedule (M4b).

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
from volsto.analytics import forward_vol_comparison
from volsto.market import implied_vol
from volsto.products import AdditiveCliquet, EuropeanOption, VarianceSwap

# 1F LSV (omega = 3, kappa = 1.5, rho = -0.7) on the reference SSVI surface: surface, curves,
# Bergomi kernel, particle settings, step schedule and scheme all come from one YAML spec
spec = load_yaml("configs/studies/lsv_reference_1f.yaml", CalibrationSpec)
cache = LeverageCache("cache")                      # content-addressed; the only entry point
model, _ = cache.get_or_calibrate(spec)             # ~33 s on a miss (2e5 particles, 3y), instant on a hit
_, surface, _ = build_market(spec)

mc = MonteCarlo(SimConfig(n_paths=200_000, seed=1))  # default schedule + scheme, as in calibration
T, K = 1.0, float(surface.forward(1.0))
call = mc.price(EuropeanOption(K, T, "call", surface.discount), model)
iv = implied_vol(call.mean, K, K, T, 1, surface.discount.df(T))
print(call, f"LSV implied {iv:.4%} vs surface {surface.implied_vol(K, T):.4%}")
print(mc.price(VarianceSwap.daily(T, 0.0, surface.discount), model))  # fair variance strike

cliquet = AdditiveCliquet.study(1.0, surface.discount, notional=100.0)  # monthly, cap 2%, floor 0
print(cliquet, mc.price(cliquet, model))                                 # in % of notional
print(forward_vol_comparison(model, 1.0, 2.0, SimConfig(n_paths=200_000, seed=1)))  # 1y-into-1y

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
Platen's explicit weak order-2 spot step (`weak_order2=True`) and, for the Bergomi / LSV kernels,
the second-order SV step (`sv_order2=True`, M4b: exact factor increments first, trapezoidal
variance in the drift, explicit spot/variance cross terms).  On the reference surface the 1m ATM
local-vol repricing bias is 0.03 vol points at dt = 1/365 versus 0.36 for plain log-Euler, and the
pure 1F Bergomi model at dt = 1/365 matches the mixing solution within 0.03 vol points at 1m where
the frozen-variance step was 0.06 low on the variance swap and 0.14 flatter in the smile;
`volsto.engine.refinement_study` measures such biases under common random numbers.  Plain
log-Euler, the frozen-variance step, time-averaged variance and a weak predictor-corrector remain
available as options.

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

## Market data import (M3b)

```bash
volsto-import --vendor hdn --date 2022-09-15 --underlying SPX \
    --root data/hdn_sample/options_sample_2022H2 --out configs/surfaces/snapshots   # eSSVI by default; --ssvi for a single rho
```

Reads one HistoricalData.net daily CSV (34 columns), keeps the SPX/SPXW roots, derives implied
forwards by put–call-parity regression, builds an arbitrage-checked `GridSurface` from OTM mid
quotes, fits eSSVI (rho per pillar; `--ssvi` for a single rho) and writes a dated market YAML with provenance that runs through
calibration like the synthetic configs.  `scripts/capture_yfinance.py` saves today's SPX/SPY
chain from Yahoo in the same layout (`pip install -e ".[data]"`).

## Calibration notes (M3)

`ParticleConfig` defaults were chosen by measurement on the reference surface: local-linear
kernel regression of E[ξ|S] (Nadaraya–Watson carries an h² m′f′/f design bias that skewed the
±10% repricing by 0.3 vol points) on a grid that follows the particle cloud (M4b: a fixed grid
biased the 1m ATM vol 0.09 vol points low at ω = 3), a plug-in ½h²m″ bias correction differenced
on a bandwidth-wide stencil, a 2000-particle window floor in the tails, and a saturating
log-quadratic tail extrapolation (the flat rule mis-priced the 3m +30% call by 1 vol point and
variance swaps by 0.3).  Calibration and pricing share the kernel step for step: every leverage
lookup inside a step uses the slice at the step start.  A single calibration with 2·10⁵
particles carries ±0.035 vol points of seed noise per variance-swap pillar; use seed averages or
8·10⁵ particles for production figures (SPEC §4.2, M4b notes).

## Forward-start products and forward-smile analytics (M4)

`ForwardStartOption` pays `(cp (S_T2/S_T1 − k))⁺` (book §3.1; `k` is a moneyness, `t1 = 0` and
a deferred pay date are allowed), `ForwardStartStraddle`, `FVA` (the forward-start ATM-forward
straddle less its Black premium at the agreed vol, so the fair strike is the forward ATM vol),
`AdditiveCliquet` (local floor/cap, global floor/cap; `AdditiveCliquet.study(T)` is the study's
monthly 2%-capped, zero-floored structure), `ReverseCliquet` and `Napoleon`.  `decompose()` on
the cliquets returns cash + a strip of forward-start calls (long at `1 + LF`, short at `1 + LC`)
+ a put / call on the accumulated sum (`AccumulatedSumOption`); the identity holds path by path.

`volsto.analytics.forward_smile` inverts out-of-the-money forward-start prices with the model's
own forward ratio `F(T2)/F(T1)` (`forward_smile`, `forward_atm_vol`), prices the forward ATM
vol, the forward variance swap and the forward vol swap on one path set
(`forward_vol_comparison`) and lays forward smiles of several models side by side
(`put_wing_table`).  `scripts/m4_headline.py` (`volsto.studies.m4`) reproduces the M4 headline
table: LV and the 1F LSV for ω = 1, 2, 3 and the 2F Table 8.2 set on the reference surface.

## Development

```bash
.venv/bin/black volsto tests && .venv/bin/ruff check volsto tests && .venv/bin/mypy && .venv/bin/pytest -m "not slow"
```

`pytest -m slow` runs the full-size checks.  `docs/Stochastic_Volatility_Modeling.pdf` (Bergomi)
is git-ignored on purpose; docstrings cite its equation numbers.
