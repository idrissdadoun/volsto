# volsto

Stochastic-volatility pricing library for single-underlying light-exotic and exotic parameter
studies: Black–Scholes, Dupire local vol, two-factor lognormal Bergomi forward-variance model,
LSV with particle-calibrated leverage, Monte Carlo with common random numbers, and viewers over a
precomputed parameter cache.  The full design is in [SPEC.md](SPEC.md).

Status: **M10 built (pending review); M1–M9 accepted** — market layer, BS, local vol, MC engine, vanilla / variance products (M1); the
two-factor lognormal Bergomi forward-variance model with exact factor stepping, its closed forms
and the mixing-solution smile (M2); particle-method leverage calibration, the LSV model, §4.2
repricing diagnostics and the content-addressed leverage cache (M3); the HistoricalData.net
option-chain importer and SSVI/eSSVI fitter (M3b); forward-start options, the FVA, the cliquet
family with exact decompositions, forward-smile analytics and the headline study runner (M4);
the second-order SV spot step, the adaptive particle-regression grid and the restored
1/1460–1/365–1/250 schedule (M4b); conditional / corridor / knock-out variance swaps and the
volatility knock-out put with production-count (8e5-particle) headline baselines (M4c). Risk layer (M5): CRN bump-and-reprice engine with recalibration through the
leverage cache, five delta regimes, vega variants, theta split, vega-T waves, forward-variance /
skew / curvature ladders, spot and cliquet gamma profiles, parameter sensitivities, product risks,
likelihood-ratio / conditional / control-variate estimators, P&L attribution and `RiskReport`.
M6: barrier machinery (discrete and Brownian-bridge continuous monitoring, barrier shift,
Reiner–Rubinstein closed forms), autocall / Phoenix notes with exact leg decompositions and
analytics, and a 1F LSV ADI PDE cross-check (M6). Smile dynamics, SSR estimators and the P1
marking calibration by SABR break-evens (M7); the hedging framework and the M8b hedging studies
(M8); the sharded precompute, results store and Streamlit viewers (M9); the certified eSSVI
calendar repair, the study runner with LaTeX output, the study catalogue S1–S7 and the rolling
backtest (M10). Methodology: [docs/methodology.md](docs/methodology.md); studies:
[docs/studies.md](docs/studies.md).

## Install

Python 3.12 or later is required (numpy ≥ 2.5 dropped 3.11; mypy strict runs against 3.12).

```bash
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -e ".[dev]"
```

## Setup on a new machine

1. **Python 3.12.13** (`.python-version`; any 3.12.x works, 3.12.13 is the version the lock was
   frozen on). Install it with `uv python install 3.12.13` or Homebrew `python@3.12`.
2. **Install**, pinned to the frozen environment of `requirements-lock.txt` (every package of the
   working `.venv`, including the `viewers`, `data`, `studies` and `dev` extras):

   ```bash
   uv venv --python 3.12.13 .venv
   uv pip install --python .venv/bin/python -r requirements-lock.txt -e .
   ```

   To resolve fresh instead of from the lock: `uv pip install --python .venv/bin/python -e ".[dev,viewers,data,studies]"`.
3. **Tests**: `.venv/bin/python -m pytest -n auto -m "not slow"` (fast suite; drop the marker for the
   full-size tests). Tests never calibrate: a test whose leverage is not in `cache/` skips with the
   reason, and a test needing the vendor sample skips when `data/hdn_sample/` is absent.
4. **Tectonic** (`brew install tectonic`) compiles the study runner's `study.pdf`; a full TeX
   distribution is not needed. Without it the LaTeX check and its tests skip. Its first compile
   downloads the TeX package bundle and needs network access.
5. **Git-ignored data** (none of it is in the repository; restore it from the local archive
   `volsto-local-data.tar.gz`, extracted at the repository root):

   | Path | Contents |
   |---|---|
   | `data/hdn_sample/` | the HistoricalData.net option-chain sample 2022 H2 (licensed, local only) |
   | `data/orats_sample/` | the ORATS free one-day sample (`ORATS_SMV_Strikes_20240103.zip`; re-download: `docs/m11_part0.md`) |
   | `data/raw/`, `data/store/` | vendor archives as delivered and the Parquet store built from them (`VOLSTO_DATA_RAW`, `VOLSTO_DATA_STORE`; see "Vendor data") |
   | `data/history/` | yfinance and Cboe daily closes (`scripts/fetch_history.py` re-downloads them) |
   | `cache/` | the content-addressed leverage cache (hours to days of calibration) |
   | `outputs/` | the results store, study outputs and reports |
   | `docs/Stochastic_Volatility_Modeling.pdf` | Bergomi's book, copyrighted: place your own copy |

## Quickstart

### Command line: precompute a toy grid, run study S1, open the viewer

Run the commands below from the repository root. They install the optional extras, precompute a
small grid, run study S1 on it, and check the viewer.

The walk-through writes everything under a scratch root, `$QS`. This keeps the toy grid
(2·10⁴ particles, 1y horizon) away from the production store: the toy grid's local-vol
point has the same id as the one in `outputs/store`, so writing the toy there would replace that
point.

```bash
uv pip install --python .venv/bin/python -e ".[viewers,studies]"   # streamlit + plotly, matplotlib
source .venv/bin/activate
export QS=/tmp/volsto-quickstart

# 1. Precompute the toy grid (configs/grids/toy.yaml): the placeholder surface's LV point,
#    1F nu in {0.25, 0.5} (rho -0.7, kappa 1.5) and 2F Table 8.2 -- 3 leverage calibrations
#    at 2e4 particles, 4000 pricing paths, no risk tier
volsto-precompute --grid configs/grids/toy.yaml --store $QS/store --cache $QS/cache --dry-run
volsto-precompute --grid configs/grids/toy.yaml --store $QS/store --cache $QS/cache

# 2. Study S1 (forward vol and cliquets vs vol-of-vol) in fast mode on that store.
#    It reads the store and the cache and never calibrates.
volsto-study run configs/studies/catalogue/s1_fast.yaml \
    --grid configs/grids/toy.yaml --store $QS/store --cache $QS/cache \
    --outputs $QS/outputs --out $QS/studies/s1_forward_vol_fast

# 3. The viewer: render the eight pages headless (nothing served), then serve them
volsto-viewer --grid configs/grids/toy.yaml --store $QS/store --cache $QS/cache --outputs $QS/outputs --check
volsto-viewer --grid configs/grids/toy.yaml --store $QS/store --cache $QS/cache --outputs $QS/outputs
```

The last command starts Streamlit on port 8501 (`--port`, `--headless`); stop it with Ctrl-C.

**Where the study lands.** Everything is in `$QS/studies/s1_forward_vol_fast/`:

| File | Contents |
|---|---|
| `study.md` | the question, the narrative with the tables inlined, provenance |
| `study.pdf`, `study.tex` | compiled by Tectonic (see below) |
| `tables/*.tex` | the LaTeX tables |
| `figures/*.pdf`, `figures/*.png` | the figures |
| `results.parquet` | every number with its standard error |
| `manifest.json` | commit, cache keys, seeds, wall clock, `recalibrated: false` |

A fast-mode study carries a "FAST MODE" banner: the toy numbers check the plumbing, not the
models. `study.pdf` needs Tectonic (`brew install tectonic`); without it the LaTeX check is
skipped and the install line is printed, and `--no-latex-check` skips it on purpose.

To rebuild the documents from the stored numbers, or to re-execute the study and diff the numbers
at 2 standard errors:

```bash
volsto-study render $QS/studies/s1_forward_vol_fast
volsto-study rerun  $QS/studies/s1_forward_vol_fast
```

**Default paths**, used when a flag is omitted (all relative to the repository root):

| Command | Defaults |
|---|---|
| `volsto-precompute` | `--grid configs/grids/default.yaml --store outputs/store --cache cache` |
| `volsto-study run` | the config's `store: outputs/store`, `cache: cache`, `outputs: outputs`; output in `outputs/studies/<name>` |
| `volsto-viewer` | `--cache cache --store outputs/store --outputs outputs --snapshots configs/surfaces/snapshots --grid configs/grids/default.yaml`; also settable through `VOLSTO_CACHE` / `VOLSTO_STORE` / `VOLSTO_OUTPUTS` / `VOLSTO_SNAPSHOTS` / `VOLSTO_GRID` or `configs/viewer.yaml` |

The production S1 reads the repository store directly:
`volsto-study run configs/studies/catalogue/s1.yaml`.

**Measured** on 2026-09-16, with `NUMBA_NUM_THREADS=2` on a laptop shared with a Monte Carlo job:

| Step | Wall clock |
|---|---|
| Install (resolved with `--dry-run`: every extra already satisfied) | 0.3 s |
| Precompute dry run | 1.1 s |
| Precompute | 20.2 s (3 calibrations of 2.8–2.9 s each) |
| S1 fast | 3.2 s |
| Viewer `--check` | 8.1 s (8 pages ok) |
| `render` | 2.9 s |
| `rerun` | 2.6 s (351 numbers, nothing moved) |

On an empty cache the dry run charges a 140 s fallback per calibration and projects 0.12 h. That
figure overstates the toy run.

What each catalogue study answers, and how to run it: [docs/studies.md](docs/studies.md). The
model, calibration and conventions: [docs/methodology.md](docs/methodology.md). The production
grid on a rented VM: [docs/vm_grid_run.md](docs/vm_grid_run.md).

### Python API

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

## Forward-start, conditional-variance and VKO products (M4, M4c)

`ForwardStartOption` pays `(cp (S_T2/S_T1 − k))⁺` (book §3.1; `k` is a moneyness, `t1 = 0` and
a deferred pay date are allowed), `ForwardStartStraddle`, `FVA` (the forward-start ATM-forward
straddle less its Black premium at the agreed vol, so the fair strike is the forward ATM vol),
`AdditiveCliquet` (local floor/cap, global floor/cap; `AdditiveCliquet.study(T)` is the study's
monthly 2%-capped, zero-floored structure), `ReverseCliquet` and `Napoleon`.  `decompose()` on
the cliquets returns cash + a strip of forward-start calls (long at `1 + LF`, short at `1 + LC`)
+ a put / call on the accumulated sum (`AccumulatedSumOption`); the identity holds path by path.

`ConditionalVarianceSwap` (`UpVar` / `DownVar` desk helpers: accrue where the spot is above or
below a barrier on the previous, current or both closes; conditional or corridor convention),
`ConvexitySpread`, `KnockOutVarianceSwap` (up-and-out, close-to-close, variant b) and
`VolKnockOutPut` (put alive only if the life realised vol ends below the vol barrier) reuse the
fixing-date realised variance; `volsto.analytics.conditional_variance` gives their fair strikes
with delta-method errors, the LSV-minus-LV differential and the VKO analysis (barrier sweep of
the ratio to the vanilla put, P(KO), and the realised vol of the paths ending in the money).

`volsto.analytics.forward_smile` inverts out-of-the-money forward-start prices with the model's
own forward ratio `F(T2)/F(T1)` (`forward_smile`, `forward_atm_vol`), prices the forward ATM
vol, the forward variance swap and the forward vol swap on one path set
(`forward_vol_comparison`) and lays forward smiles of several models side by side
(`put_wing_table`).  `scripts/m4_headline.py` (`volsto.studies.m4`) reproduces the M4 headline
table: LV and the 1F LSV for ω = 1, 2, 3 and the 2F Table 8.2 set on the reference surface.

## Barriers, autocalls and the PDE cross-check (M6)

```python
from volsto.products import Autocall, KnockOutOption, Digital
from volsto.analytics import autocall_report
from volsto.pde import LSV1FPDE

ko = KnockOutOption(100.0, 1.0, "call", 90.0, "down", disc, monitoring="continuous")  # bridge weights
note = Autocall([1.0, 2.0, 3.0], disc, spot_reference=100.0, coupons=0.06, ki_level=0.6,
                ki_type="european", ki_monitoring="discrete", final_redemption="knock_in")
rep = autocall_report(note, model, sim)        # price, expected life, P(KI), legs, all with stderr
pde = LSV1FPDE(lsv_model)                      # theta = 0 kernels; 400 x 121 grid, HV ADI
print(pde.vanilla(100.0, 1.0, 1).price, pde.knock_out_call(100.0, 1.0, 90.0, "down").price)
```

Conventions are explicit arguments (`strict` for discrete barriers, `rebate_timing`, `survival`,
`final_redemption`); every Monte Carlo number carries its standard error; the decompositions
reprice path by path. Risk for the digital structures lives in `volsto.risk.digital_risk`
(autocall spot profiles with smoothing and barrier shift, KI-put barrier risk, per-leg vega-T
and skew, expected-life sensitivities, likelihood-ratio cross-check); the M6 headline rows in
`volsto.studies.m6` / `scripts/m6_headline.py`. Measured accuracies and the conventions are in
SPEC §6.8.

## Risk layer (M5)

```python
from volsto.calibration import LeverageCache
from volsto.config import CalibrationSpec, SimConfig, load_yaml
from volsto.products import EuropeanOption
from volsto.risk import LSVBuilder, RiskEngine, RiskState, delta_table, explain, risk_report
from volsto.risk.engine import surface_of

state = RiskState(load_yaml("configs/studies/lsv_reference_1f.yaml", CalibrationSpec))
engine = RiskEngine(LSVBuilder(LeverageCache("cache"), state), SimConfig(n_paths=200_000))
call = EuropeanOption(100.0, 1.0, 1, surface_of(state).discount)
print(delta_table(engine, call, state))          # model / sticky-strike / -moneyness / -skew / -local-vol
report = risk_report(engine, call, state)         # every sensitivity with stderr, bump specs, cache keys
print(report.summary()); report.to_excel("outputs/call_risk.xlsx")
pnl = explain(engine, call, state, state.with_spot(102.0), dt=1 / 252)   # sequential CRN attribution
```

Every bump is a `SurfacePerturbation` layer on the implied surface (arbitrage checks re-run,
halve-and-retry with the achieved size reported) or a state change (`with_spot`, `with_params`,
`with_rate_shift`, `with_x0`); surface and parameter bumps recalibrate the leverage through the
cache (each bumped state is a cache entry), spot and factor bumps do not (`"model"` regime).
Sensitivities are per-path CRN differences with the standard error of the difference.
Conventions and measured findings: SPEC §7 and §7.15 (forward-variance buckets are sized on the
log-contract strip; skew / curvature bumps use a saturating profile; the cliquet Bachelier
cross-check is an approximation even under Black–Scholes — the exact independent-legs reference
is `bs_cliquet_value_mc`).  `scripts/m5_budget.py` runs the full report under the reference LSV
and prints the recalibration count and wall clock (the viewer precompute budget).

The second vendor, ORATS, goes through the same steps from the Parquet store (see "Vendor
data"); its snapshots are written under the store, never under `configs/`:

```bash
volsto-import --vendor orats --date 2024-01-03 --underlying SPX   # -> $VOLSTO_DATA_STORE/orats/snapshots/
```

## Vendor data (M11)

Vendor archives never enter git. Two roots, each overridable by an environment variable or a
flag of `volsto-data`:

| Root | Variable | Default | Contents |
|---|---|---|---|
| raw | `VOLSTO_DATA_RAW` (`--raw`) | `data/raw` | vendor zips exactly as delivered (may be an external volume) |
| store | `VOLSTO_DATA_STORE` (`--store`) | `data/store` | the typed Parquet store derived from raw |

```bash
volsto-data status                                   # roots, free space, raw files, manifest
volsto-data fetch --vendor orats --profile <aws profile> --bucket <bucket> --prefix <prefix> --dry-run
volsto-data fetch --vendor orats --profile <aws profile> --bucket <bucket> --prefix <prefix>
volsto-data verify-raw --vendor orats                # raw manifest, trading-calendar and OPRA checks
volsto-data convert --vendor orats                   # one typed Parquet file per trading day
volsto-data verify --vendor orats                    # every Parquet file against its raw file
volsto-data extract --tickers SPX                    # one Parquet file per ticker, all dates
volsto-data sql "select count(*) from strikes"       # optional, DuckDB (pip install -e ".[data]")
```

Reading the store from Python (vendor-native columns; a missing date or ticker raises with the
exact `volsto-data` command that produces it; nothing is downloaded, converted or fitted):

```python
from volsto.market.store import available_dates, load_chain, load_range

available_dates("orats")                                   # ['2024-01-03', ...]
spx = load_chain("orats", "2024-01-03", "SPX")             # one day, one ticker
hist = load_range("orats", "SPX", "2024-01-01", "2024-12-31", columns=["trade_date", "strike", "cBidPx"])
```

`fetch` wraps `aws s3 sync` and takes the *name* of an AWS profile you configured; the code
never sees a credential. The store is a faithful typed copy (every vendor column, float64,
zstd 9), rebuildable from raw; both roots may sit on external volumes. `docs/data_runbook.md` is the download-day procedure; SPEC §18 has the
design and the measurements. Tests that need the ORATS one-day sample skip when
`data/orats_sample/` is absent.

## Development

```bash
.venv/bin/black volsto tests && .venv/bin/ruff check volsto tests && .venv/bin/mypy && .venv/bin/pytest -m "not slow"
```

`pytest -m slow` runs the full-size checks.  `docs/Stochastic_Volatility_Modeling.pdf` (Bergomi)
is git-ignored on purpose; docstrings cite its equation numbers.
