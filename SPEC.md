# volsto — Stochastic-volatility pricing library for light-exotic and exotic parameter studies

Specification v1.1 — 13 September 2026 (v1.1: §3.3/§4.4 aligned to Bergomi's book notation and equation numbers)
Owner: Idriss (hybrid trading desk). Implementer: Claude Code.

---

## 0. Purpose and scope

A single Python library that replaces the per-study re-implementation of the LSV Monte Carlo engine used in the forward-start / cliquet / FVA studies, and that supports interactive viewers (forward vol, forward skew, smile dynamics, product price grids) across vol-sto parameter values.

In scope for v1:
- Single underlying, equity-style (continuous dividend yield / repo), deterministic rates.
- Models: Black–Scholes, Dupire local vol, two-factor lognormal Bergomi forward-variance model (one-factor is the degenerate case), LSV = leverage function × SV kernel, Heston (comparison only).
- Particle-method leverage calibration to a target vanilla surface (SSVI parametrisation, or a market slice grid).
- Monte Carlo engine with common random numbers, antithetics, control variates, standard errors on every number.
- Products: vanilla, forward-start, variance swap, vol swap, FVA, cliquet family, single-underlying autocall / Phoenix, discrete and continuous barriers.
- Risk: bump-and-reprice Greeks, forward-variance vega ladder (sticky-leverage and recalibrated variants), spot-shift and gamma profiles, vol-of-vol sensitivity under recalibration, smile-dynamics diagnostics (SSR).
- Hedging simulation framework (pricing model vs realised-world model).
- Precomputed grids + Streamlit/Plotly viewers.
- Test suite with analytical identities and regression numbers from the existing studies.

Explicitly deferred: multi-asset / worst-of, discrete cash dividends, jumps, stochastic rates, adjoint Greeks. Design so that these can be added without changing the product interface (see §3 on path containers).

Language and stack: Python 3.11+, numpy, scipy, numba (path loop and kernel regression), pandas, pyarrow (cache), plotly + streamlit (viewers), pytest, pyyaml. No pandas inside numba kernels. Type hints everywhere. No global state.

---

## 1. Repository layout

```
volsto/
  pyproject.toml
  README.md
  volsto/
    __init__.py
    config.py            # dataclasses + YAML loading/validation for every config object
    market/
      curves.py          # discount factors, forward curve (r, q) — piecewise-flat and interpolated
      surface.py         # ImpliedSurface ABC; SSVISurface; GridSurface (market slices)
      dupire.py          # local vol from total-variance surface (Gatheral formula)
      varswap.py         # variance-swap strip by log-contract replication; xi0(T) curve
      bs.py              # Black–Scholes price, greeks, implied vol inversion (vectorised, robust)
    models/
      base.py            # Model ABC, ModelState, path simulation interface
      bs.py
      localvol.py
      bergomi.py         # BergomiSV (2F; 1F when theta=0), exact OU steps
      heston.py
      lsv.py             # LSV wrapper: leverage function L(t,S) over any SV kernel
      leverage.py        # LeverageFunction: (t,S) grid + interpolation + serialisation
    calibration/
      particle.py        # particle method for L(t,S)
      diagnostics.py     # repricing error vs target surface at pillars, var-swap check
      cache.py           # content-addressed cache of calibrated leverage functions
      ssr.py             # skew-stickiness ratio estimator and 2F fitting helpers
    engine/
      grid.py            # TimeGrid: union of product fixings + discretisation steps
      rng.py             # seeded Gaussian generation, antithetics, CRN keys
      mc.py              # MonteCarlo: run model on grid, return PathSet
      paths.py           # PathSet container (spot, V, factors, realised variance accumulators)
      cv.py              # control variates (vanilla, variance swap)
    products/
      base.py            # Product ABC: fixing schedule, payoff(PathSet) -> cashflows
      vanilla.py
      forward_start.py
      variance.py        # VarianceSwap, VolSwap, FVA, forward variance swap
      cliquet.py         # additive, local cap/floor, global floor/cap, reverse, Napoleon
      autocall.py        # single-underlying autocall, Phoenix (memory), KI put decomposition
      barrier.py         # discrete/continuous KI/KO, barrier shift, digital
    risk/
      greeks.py          # bump-and-reprice with CRN
      ladders.py         # forward-variance vega ladder, both variants
      profiles.py        # spot-shift and gamma profiles
      volsto_sens.py     # sensitivities to (omega, theta, k1, k2, rho...) with recalibration
      attribution.py     # P&L explain
    hedging/
      instruments.py     # hedge instruments priced under a pricing model
      hedger.py          # rebalancing loop, realised-world model, P&L distribution
      report.py
    analytics/
      forward_smile.py   # forward-start implied smiles, forward ATM vol vs forward var swap
      smile_dynamics.py  # conditional smile after spot move, SSR, vol-of-vol term structure
      var_decomp.py      # Var(V) decomposition (predictable at T1 vs within-period), closed forms
    pde/
      lsv1f.py           # 2D finite-difference pricer for the 1F degenerate case (validation only)
    studies/
      runner.py          # run a study from YAML, write parquet + LaTeX tables + figures
      latex.py
    viewers/
      precompute.py      # builds parameter grids into the cache
      app.py             # Streamlit entry point
      pages/             # one page per viewer (see §9)
  docs/
    Stochastic_Volatility_Modeling.pdf   # Bergomi — reference for §3.3, §4.1, §4.4; cite eq. numbers
  configs/
    surfaces/            # SSVI parameter files (including the one from the original study)
    models/
    studies/
  tests/
  notebooks/
```

---

## 2. Market data

### 2.1 Curves
`DiscountCurve(times, rates)` and `ForwardCurve(S0, r_curve, q_curve)`; `F(T) = S0 * exp(∫(r−q))`. Piecewise-flat rates by default; linear-in-log-DF interpolation. Everything in year fractions (ACT/365 fixed; a `Calendar` helper converts dates).

### 2.2 Implied surface
`ImpliedSurface` ABC with `total_variance(k, T)` where `k = ln(K/F(T))`, plus `implied_vol(K,T)`, `price(K,T,cp)`.

`SSVISurface(theta_T, rho, phi)` — Gatheral–Jacquier SSVI: `w(k,T) = θ_T/2 · (1 + ρ φ(θ_T) k + sqrt((φ(θ_T) k + ρ)² + 1 − ρ²))`, with `φ(θ) = η / (θ^γ (1+θ)^(1−γ))` (power-law) and `θ_T` an interpolated ATM total variance term structure. Must implement the no-arbitrage checks (`θφ(θ)(1+|ρ|) < 4`, `θφ(θ)²(1+|ρ|) ≤ 4`) and raise on violation.

`GridSurface` — market slices (K or delta, T) with arbitrage-free interpolation in total variance (linear in `w` along `k`, linear in `w` along `T` at fixed `k`), extrapolation flat in implied vol beyond wings. Enough for feeding a Bloomberg export later.

### 2.3 Dupire local vol
`LocalVolSurface.from_implied(surface)`:
`σ_loc²(k,T) = ∂_T w / (1 − k/w ∂_k w + ¼(−¼ − 1/w + k²/w²)(∂_k w)² + ½ ∂_kk w)`, derivatives by central finite differences on the analytic SSVI (step sizes configurable), floored at a small positive value, and stored on a (T, k) grid with bilinear interpolation. Provide `check_positive()` diagnostic.

### 2.4 Variance swaps and ξ₀
`varswap_strike(surface, T)`: log-contract replication `K_var(T) = (2/T) ∫ [P(K)/K² (K<F) + C(K)/K² (K>F)] e^{rT} dK`, adaptive quadrature in `k` with wide bounds. `xi0_curve(surface)`: forward variance `ξ₀(T) = d/dT [T · K_var(T)]`, computed on a fine T grid and interpolated; must be positive.

---

## 3. Models

### 3.1 Common interface
```python
class Model(ABC):
    n_factors: int                  # 0 (BS/LV), 1 or 2 (Bergomi), 1 (Heston)
    def simulate(self, grid: TimeGrid, rng: GaussianDraws, cfg: SimConfig) -> PathSet
    def instantaneous_variance(self, state) -> ndarray
    def initial_state(self, n_paths) -> ModelState
    def bump(self, **kwargs) -> "Model"    # returns a new model with parameters changed
```
`PathSet` holds arrays of shape `(n_paths, n_times)` for `log_spot`, `V` (instantaneous variance), `factors` `(n_paths, n_times, n_factors)`, plus accumulated realised variance `∫V dt` and realised log-return sum-of-squares between consecutive fixing indices. Products access only `PathSet` and a `fixing_index` map from dates to columns. `PathSet` is written so that a second underlying can later be added as an extra leading axis without touching products.

All simulation is log-Euler in the spot with the variance frozen over each step; factors are stepped exactly (§3.3). The spot step's variance is chosen by `SchemeConfig` (owner amendment after M1): time-averaged local/forward variance over the step (`local_var_time_average`), an optional weak predictor–corrector (`predictor_corrector`, drift θ = ½ with the Itô correction, diffusion weight `pc_eta`), or — the default — Platen's explicit weak order-2 step (`weak_order2`). Step sizes come from `SimConfig.dt_max`, a `StepSchedule` (default 1/2920 below 3m, 1/730 to 2y, 1/500 after since the pre-M4 amendment; the LSV step is first order in time and needed the halving, see §4.2) shared by pricing and calibration, with the union of fixing dates always included; `record_all_steps` records every grid step when a product needs the path between fixings.

### 3.2 Black–Scholes and local vol
Trivial. Local vol uses `σ_loc(t, S_t)` interpolated on the grid.

### 3.3 Two-factor lognormal Bergomi forward-variance model
Source: Bergomi, *Stochastic Volatility Modeling* (CRC, 2016), Chapter 7 — the book PDF is in `docs/`; cite equation numbers in docstrings. Notation follows the book exactly.

Factors (book eq. 7.30 and below), zero mean, `X¹₀ = X²₀ = 0`:
```
dX¹_t = −k1 X¹_t dt + dW¹_t
dX²_t = −k2 X²_t dt + dW²_t
corr(dW¹, dW²) = ρ12        (k1 > k2: X¹ is the short factor, X² the long factor)
```
Mixed Gaussian factor for maturity T (eq. 7.30):
```
x_t^T = α_θ [ (1−θ) e^{−k1(T−t)} X¹_t + θ e^{−k2(T−t)} X²_t ]
α_θ  = 1 / sqrt( (1−θ)² + θ² + 2 ρ12 θ (1−θ) )                       (eq. 7.29)
```
Forward variance (eqs. 7.32–7.35):
```
dξ_t^T = ω ξ_t^T dx_t^T,       ω = 2ν
ξ_t^T  = ξ_0^T exp( ω x_t^T − ½ ω² χ(t,T) )
χ(t,T) = ∫_{T−t}^{T} η²(u) du
       = α_θ² [ (1−θ)² e^{−2k1(T−t)} (1−e^{−2k1 t})/(2k1)
              + θ² e^{−2k2(T−t)} (1−e^{−2k2 t})/(2k2)
              + 2θ(1−θ)ρ12 e^{−(k1+k2)(T−t)} (1−e^{−(k1+k2)t})/(k1+k2) ]
η(u)   = α_θ sqrt( (1−θ)² e^{−2k1 u} + θ² e^{−2k2 u} + 2ρ12 θ(1−θ) e^{−(k1+k2)u} ),  η(0) = 1   (eq. 7.31)
```
`χ(t,T) = Var[x_t^T]`, so `E[ξ_t^T] = ξ_0^T` for all t ≤ T (martingale — test it). ν is the instantaneous lognormal volatility of a VS volatility of vanishing maturity; equivalently ω = 2ν is the lognormal vol of vol of `ξ_t^t`. **The ω of the earlier 1F studies is this ω** (ω = 3 ⇔ ν = 150%, the order of magnitude of the book's Set I). Store ν in configs, expose both.

Instantaneous variance of the SV kernel: `V_t = ξ_t^t`.

Spot (book §7.3.1 and §8.7), with leverage `L ≡ 1` for the pure SV model:
```
d ln S_t = (r_t − q_t − ½ L(t,S_t)² ξ_t^t) dt + L(t,S_t) sqrt(ξ_t^t) dW^S_t
corr(dW^S, dW¹) = ρ_SX1,   corr(dW^S, dW²) = ρ_SX2
```
The 3×3 correlation matrix of `(W^S, W¹, W²)` must be PSD; validate at construction. The book's admissibility parametrisation (eq. 8.56) `ρ_SX2 = ρ12 ρ_SX1 + χ_c sqrt(1−ρ12²) sqrt(1−ρ_SX1²)`, `χ_c ∈ [−1,1]`, is offered as an alternative constructor.

Exact simulation (book §7.3.1, eqs. 7.15–7.18) over a step δτ:
```
X^i_{τ+δτ} = e^{−k_i δτ} X^i_τ + δX^i
E[δX^i δX^j] = ρ_ij (1 − e^{−(k_i+k_j)δτ}) / (k_i + k_j)
E[δW^S δX^i] = ρ_iS (1 − e^{−k_i δτ}) / k_i
E[(δW^S)²]   = δτ
```
Build the 3×3 covariance of `(δW^S, δX¹, δX²)` per distinct step size, Cholesky it once per step size, draw jointly. The factors are exact; only the spot is log-Euler with `ξ_t^t` frozen over the step. When no spot is needed (pure variance payoffs) the factors can be sampled directly at the fixing dates with no intermediate stepping.

Degenerate 1F: `θ = 0` ⇒ `α_θ = 1`, `x_t^T = e^{−k1(T−t)} X¹_t`, `χ = e^{−2k1(T−t)}(1−e^{−2k1 t})/(2k1)`; this is exactly the 1F model of the earlier studies with κ = k1 and ρ = ρ_SX1. Test: with the same seed, `θ = 0` reproduces the 1F implementation path by path.

Parameters dataclass: `BergomiParams(nu, theta, k1, k2, rho12, rho_SX1, rho_SX2)`; `BergomiParams.one_factor(omega, kappa, rho)`. Reference parameter sets shipped in `configs/models/`: book Table 8.2 (`ν = 174%, θ = 0.245, k1 = 5.35, k2 = 0.28, ρ12 = 0, ρ_SX1 = −75.9%, ρ_SX2 = −48.7%`) and Table 7.1 Sets I–III (variance dynamics only).

Closed forms to implement in `analytics/`, all with `L ≡ 1`, used as tests:
- Instantaneous vol of a VS volatility, eq. 7.39 with `A_i` from eq. 7.38 (flat term structure: `A_i = (1−e^{−k_i(T−t)})/(k_i(T−t))`).
- Order-one ATMF skew for a flat VS term structure, eq. 8.55 / 9.20:
  `S_T = ν α_θ [ (1−θ) ρ_SX1 (k1T − (1−e^{−k1T}))/(k1T)² + θ ρ_SX2 (k2T − (1−e^{−k2T}))/(k2T)² ]`.
- Order-one SSR for a flat VS term structure, eq. 9.21; the general (sloping term structure) forms eqs. 9.18–9.19.
- Covariance of instantaneous variances (derived here from the OU covariances — verify against MC): for `u < v`,
  `Cov(x_u^u, x_v^v) = α_θ² [ (1−θ)² e^{−k1(v−u)} (1−e^{−2k1u})/(2k1) + θ² e^{−k2(v−u)} (1−e^{−2k2u})/(2k2) + θ(1−θ)ρ12 (1−e^{−(k1+k2)u})/(k1+k2) (e^{−k1(v−u)} + e^{−k2(v−u)}) ]`,
  `Cov(ξ_u^u, ξ_v^v) = ξ_0^u ξ_0^v (exp(ω² Cov(x_u^u, x_v^v)) − 1)`,
  `Var(∫_{T1}^{T2} ξ_t^t dt) = ∫∫ Cov(ξ_u^u, ξ_v^v) du dv` (this is the book's eq. 7.19–7.20 machinery specialised to the diagonal).
- The predictable/within-period decomposition `Var(V) = Var(E[V|F_{T1}]) + E[Var(V|F_{T1})]` from the original study: closed form for the pure model, MC regression estimator on the `T1` state for the LSV model.

Vanilla smiles of the pure SV model without spot simulation: implement the mixing solution of the book's Chapter 8 Appendix A (condition on the factor path, integrate the orthogonal spot Brownian analytically, average Black–Scholes prices with the effective variance and shifted forward). This is the cheap, low-noise way to get the "naked" 2F smile and its ATMF skew for the 2F fitting helper (§4.4) and for the eq. 8.55 test.

### 3.4 Heston
Standard, QE scheme (Andersen). Comparison only; leverage on top optional.

### 3.5 LSV wrapper
`LSV(kernel: Model, leverage: LeverageFunction)`. `L(t,S)` stored on a `(t, ln S)` grid, linear interpolation, flat extrapolation in S, and held constant on the last time slice beyond the calibration horizon. Serialises to `.npz` with its provenance metadata (§4.3).

---

## 4. Calibration

### 4.1 Particle method (Guyon–Henry-Labordère; book §12.2.5)
Target: `L(t,S)² = σ_loc²(t,S) / E[V_t | S_t = S]`.

Algorithm, on the simulation time grid:
1. At `t_0`, `L(0,S) = σ_loc(0,S)/sqrt(ξ_0^0)`.
2. Step all `N` particles from `t_i` to `t_{i+1}` with the current `L(t_i, ·)`.
3. At `t_{i+1}`, estimate `E[V | S]` by Nadaraya–Watson kernel regression in `ln S` (Gaussian or quartic kernel), bandwidth `h_i = c · σ_ref · sqrt(t_{i+1}) · N^{−1/5}` with `σ_ref` the ATM vol and a floor `h_min`; `c` default 1.5 (configurable). Evaluate on the leverage `ln S` grid; outside the particle cloud's 0.5%–99.5% quantiles hold `E[V|S]` flat.
4. Set `L(t_{i+1}, S)` and continue.

Numba `@njit(parallel=True)` for the kernel regression (O(N × n_grid); use sorted particles and a truncated kernel window to make it O(N + n_grid) when N is large). Default `N = 2·10⁵` particles, `dt = 1/365` up to 1y then 1/250, leverage `ln S` grid of 201 points spanning ±6 ATM standard deviations at the maturity.

Optional second pass: re-run with the calibrated `L` and a fresh seed and average the two `L` surfaces (reduces particle noise).

Implementation notes (M3, measured on the reference surface, see `ParticleConfig`): the regression is local-linear rather than Nadaraya–Watson (NW carries the design bias `h² m′ f′/f`, which with `c = 1.5` skewed the ±10% repricing by 0.3 vol points), with a plug-in `½ h² m″` curvature correction (otherwise a −0.10 vp level bias at 1y), a 2000-particle window floor in the tails, and `E[V|S]` extrapolated beyond the trusted quantiles with a saturating log-quadratic (the flat rule mis-priced the 3m +30% call by 1 vp and variance swaps by 0.3 vp). `E[V|S]` is estimated on the 201-point grid and interpolated onto the fine leverage grid (dk = 0.0025, the Dupire grid) where `σ_loc²` is resolved. All lookups within a simulation step use `L(t_n, ·)` (frozen-leverage rule), identically in calibration and pricing.

### 4.2 Diagnostics
`calibration/diagnostics.py`: reprice the target surface on a pillar grid (T ∈ {1m, 3m, 6m, 1y, 18m, 2y, 3y}, k ∈ ±{0, 0.05, 0.1, 0.2, 0.3}) with the calibrated LSV via MC (CRN, `n_paths ≥ 4·10⁵`), report implied-vol error in vol points with MC standard error, plus variance-swap strikes vs the replication values.

Acceptance table (updated after M1; pure local vol measured with the default scheme — Platen weak order 2 — and the default step schedule 1/1460 below 3m, 1/365 to 2y, 1/250 after; 800k paths; `tests/test_scheme.py`, `tests/test_surface.py`):

| Maturity | Pure LV vs SSVI, ATM | Pure LV, ±10% | Pure LV, ±20% | LSV (M3), ATM | LSV (M3), ±20% within 2.5 σ√T | LSV variance swap |
|---|---|---|---|---|---|---|
| 1m | ≤ 0.10 vp (measured 0.02) | ≤ 0.20 vp | — | ≤ 0.10 vp (measured 0.02) | ≤ 0.15 vp (±10% only: ±20% is 3.4 σ√T) | ≤ 0.10 vp (measured 0.04–0.08) |
| 3m | ≤ 0.10 vp (measured 0.05) | ≤ 0.20 vp | ≤ 0.10 vp | ≤ 0.10 vp (measured 0.02) | ≤ 0.15 vp (measured 0.09) | ≤ 0.10 vp (measured 0.04–0.08) |
| 6m | ≤ 0.10 vp (measured 0.04) | ≤ 0.20 vp | ≤ 0.10 vp | ≤ 0.10 vp (measured 0.02) | ≤ 0.15 vp (measured 0.08) | ≤ 0.10 vp (measured 0.02) |
| 1y–2y | ≤ 0.10 vp (measured 0.01–0.05) | ≤ 0.10 vp | ≤ 0.10 vp | ≤ 0.10 vp (measured 0.05) | ≤ 0.15 vp; put wing −20%/−30% ≤ 0.10 vp (measured 0.03–0.06) | ≤ 0.10 vp (measured 0.00–0.07) |
| 3y | — | — | ≤ 0.15 vp | ≤ 0.10 vp (measured 0.03) | ≤ 0.20 vp (measured 0.08) | open: +0.18–0.20 vp, far-put tail at the horizon edge — fix before M6 (KI put) |

LSV values measured after the pre-M4 amendments on the reference surface with the 1F (ω = 3, κ = 1.5, ρ = −0.7) and 2F (Table 8.2) kernels, 2·10⁵ particles, 3y horizon, the default step schedule 1/2920–1/730–1/500 and 4·10⁵ pricing paths on three pricing seeds (MC standard errors 0.02–0.03 vp ATM, 0.05–0.10 vp at ±20–30%). A cell counts as a violation only if its error exceeds both the tolerance and 3 MC standard errors (`CalibrationReport.passes`). Variance swaps: pure LV within 0.03 vol points at all pillars once the Dupire grid spans ±3.0 in log-moneyness (±1.5 truncated the far put wing: 2y −0.08, 3y −0.22 vp). Two pre-M4 findings fixed the LSV residuals reported after M3: (i) the tail window floor is now `max(2000, 0.01 N)` particles (k-NN, `min_window` / `min_window_fraction`), which made the variance-swap residual monotone in N — at the final defaults the 1y strike reads +0.06 / 0.00 / −0.10 vp for 5·10⁴ / 2·10⁵ / 8·10⁵ particles (it was −0.14 at 2·10⁵ and +0.23 at 8·10⁵ with a fixed 2000-particle floor), i.e. monotone but not yet flat within noise: small clouds are tail-noise rich, large clouds expose the remaining O(dt) shortfall at 2y (−0.08) — every pillar 1m–3y is within 0.10 vp at 2·10⁵ and 8·10⁵ particles, 5·10⁴ particles reach +0.11 at 18m and +0.15 at 3y; (ii) the residual level itself (1y–2y put wing 0.10–0.15 vp cheap, 1y variance swap 0.14 vp low, present for every N and bandwidth) is the O(dt) time error of the LSV step (leverage and SV variance frozen over the step): halving the schedule removed it (put wing ≤ 0.04, variance swaps within 0.08 on three seeds), so the default schedule was halved at the cost of 2× calibration (≈ 61 s for 3y at 2·10⁵ particles) and pricing time. The structural fix — a second-order SV spot step (Andersen-type, exact factor increments with the intra-step spot/variance covariance) — would recover the cost; it is scheduled with the 3y far-put-tail item before M6. The particle calibration uses exactly the pricing kernel, scheme options and step schedule (`SimConfig`), with every leverage lookup inside a step taken from the step-start slice (frozen-L rule) in both calibration and pricing. Known residual: the 2F far right tail at 1m (+20%, 3.4 σ√T, a sub-basis-point option) is 1.4 vp rich because E[ξ|S] must be extrapolated beyond the particle cloud there. For reference, plain log-Euler at uniform dt = 1/365 biased the 1m ATM vol by +0.36 vp and the θ = η = ½ predictor–corrector by +0.78 vp (curvature over-correction); the weak order-2 scheme gives +0.03 vp.

### 4.3 Cache
Content-addressed: key = SHA-256 of (surface params, curves, model params, particle config, seed, code version tag). Store leverage `.npz` + diagnostics `.json` + a `manifest.parquet` row. `get_or_calibrate(cfg)` is the only entry point studies and viewers use. Calibration must never run silently inside a viewer; the viewer reads the cache and reports what is missing.

### 4.4 SSR and 2F fitting helpers
Skew-stickiness ratio (book eq. 9.3 / 12.50):
```
R_T = (1/S_T) · E[dσ̂_T d ln S] / E[(d ln S)²]
```
with `S_T` the model's own ATMF skew `∂σ̂_T/∂ ln K` at the forward.

Estimator for the mixed (LSV) model, book §12.4.3: with a fixed leverage function the ATMF vol is a function of `(ln S, X¹, X²)`, so
```
R_T = (1/(S_T σ_0)) · [ σ_0 ∂σ̂_T/∂ ln S + ρ_SX1 ∂σ̂_T/∂X¹ + ρ_SX2 ∂σ̂_T/∂X² ],   σ_0 = L(0,S_0) sqrt(ξ_0^0)
```
evaluated with the book's single-reprice trick: bump the initial state jointly to `(ln S_0 + ε σ_0, ερ_SX1, ερ_SX2)`, reprice the ATMF option under CRN, invert to implied vol, divide the difference by `ε`. For the pure SV model the `∂/∂ ln S` term vanishes (book §9.8). Also implement the book's analytic decomposition eq. 12.52, `R_T = R_T^LV(Mkt) + (S_T^SV/S_T)[R_T^SV − R_T^LV(SV)]`, with `R^LV` from eq. 12.53–12.54 and `R^SV` from eq. 9.21, as a cross-check of the numerical estimator. A third, independent estimator (short-horizon simulation to `t = 1/52`, conditional ATMF vol by regression on `(ln S_t, X¹_t, X²_t)`, regression of `Δσ̂` on `Δ ln S`) is kept as a slow test.

Volatilities of ATMF volatilities: same state-bump partials, squared and combined with the instantaneous covariances (book §12.4.3, eq. 12.56).

`fit_2f_to_targets(targets)`: given a target SSR term structure (e.g. `R_T ≈ 1.2` for T in 1y–2y) and the market ATMF skew term structure, and a fixed ν and variance-dynamics set `(θ, k1, k2, ρ12)` (default: book Table 8.1), solve for `(ρ_SX1, ρ_SX2)` by least squares on the order-one formulas (eqs. 8.55 and 9.21 or their sloping-term-structure versions 9.18–9.19), then refine numerically with the mixing-solution smile, then calibrate leverage. Report the naked 2F ATMF skew against the market skew per maturity so the leverage correction stays small (the design intent: keep `S_T^SV` close to `S_T` so `R_T ≈ R_T^SV`, see eq. 12.52). Note the book's remark after eq. 8.55: rescaling `(ρ_SX1, ρ_SX2)` by a constant and ν by its inverse leaves the order-one skew unchanged — expose this degeneracy in the fitter's output.

---

## 5. Monte Carlo engine

- `TimeGrid.build(fixing_dates, dt_max, calibration_grid=None)`: union, sorted, with `fixing_index` map. When an LSV model is used the grid must contain the calibration time slices (or interpolate `L` in t — do the latter, linear in t).
- `GaussianDraws(seed, n_paths, n_steps, n_brownians, antithetic=True)`: PCG64 generator; the same `(seed, path index, step, brownian index)` always gives the same normal so CRN bumps are exact. Generate in blocks to bound memory; `n_paths` default 2·10⁵, chunk 5·10⁴.
- `MonteCarlo.price(product, model, grid, draws, cv=None) -> PriceResult(mean, stderr, n_paths, per_path_payoffs optional)`.
- Control variates: vanilla with the same maturity (analytic BS price under the model's implied vol at that strike from the target surface — valid because the LSV reprices the surface) and variance swap (replication strike). Coefficient estimated on the sample; report variance reduction.
- All results carry standard errors; the library never returns a bare float for a MC quantity.

---

## 6. Products

`Product` ABC: `fixing_dates`, `pay_dates`, `payoff(paths: PathSet, idx: FixingIndex) -> ndarray (n_paths,)` of discounted cash flows (discounting done inside via the curve), `decompose()` optional returning a list of simpler products whose sum reprices it (used for the KI-put and cliquet analyses).

| Family | Products | Notes |
|---|---|---|
| Vanilla | European call/put, digital | |
| Forward-start | `(S_T2/S_T1 − k)^+`, forward-start straddle, forward smile via strike grid | `analytics/forward_smile.py` inverts to forward implied vol using the model's own forward |
| Variance | variance swap (realised = Σ ln² returns on fixing grid, annualised), vol swap, forward variance swap `[T1,T2]`, FVA (forward-start straddle vs forward vol) | realised on **fixing** dates (daily by default), not the simulation grid |
| Cliquet | additive cliquet with local cap/floor, global cap/floor; reverse cliquet; Napoleon | the original study's structure = monthly, local cap 2%, no local floor, global floor 0, 1y and 2y |
| Autocall | single-underlying autocall (call barrier, coupon schedule), Phoenix with memory coupon, European or American KI put | `decompose()` returns put(K_ki) + digital put of size (100 − K_ki) at K_ki (the correct decomposition; never "a put struck at the barrier") |
| Barrier | discrete/continuous KI/KO calls and puts, barrier shift parameter, Brownian-bridge correction for continuous barriers | |

Every product has a `__repr__` that reads like a term sheet.

---

## 7. Risk and analytics

- `greeks.py`: delta, gamma, vega (parallel surface bump with recalibration or with sticky leverage — both), theta; all via CRN bump-and-reprice; report stderr of the difference, not of the levels.
- `ladders.py`: forward-variance vega ladder: bump `ξ_0` multiplicatively by `1 + ε` on bucket `[T_i, T_{i+1}]` (monthly buckets by default). Variant A "sticky leverage": keep `L`. Variant B "recalibrated": recalibrate `L` to the bumped surface (bumped surface = surface with the bucket variance-swap bump propagated to total variance at fixed skew shape — document the mapping). Output per-bucket vega per vol point of forward variance-swap vol, as in the original study (net ladder units: % of notional per vol point per forward month).
- `profiles.py`: spot-shift profiles of price/delta/gamma/vega (shift `S_0`, sticky leverage), gamma profile across the cliquet accumulated sum (reuse the Bachelier-call-on-remaining-capped-sum decomposition from the study as an analytic cross-check).
- `volsto_sens.py`: sensitivities to `(ω, θ, k1, k2, ρ1, ρ2, ρ12)` with recalibration of `L` (each point is a cache entry).
- `smile_dynamics.py`: conditional smile at `t` given a spot move (regression-based conditional pricing as in §4.4), SSR term structure, vol-of-vol term structure implied by the model (variance of the forward variance-swap strike at horizon `t`).
- `var_decomp.py`: as §3.3.
- `attribution.py`: P&L explain of a hedged position between two dates: delta, gamma, vega buckets, vol-of-vol, residual.

---

## 8. Hedging framework

Port the cliquet and FVA hedging studies onto a generic engine:
- `HedgeInstrument`: spot, vanilla (strike, maturity), variance swap, forward variance swap, forward-start straddle; each priced under the **pricing model** at rebalancing dates using the conditional-pricing regression (§4.4) or nested MC (configurable, regression default).
- `Hedger(strategy, pricing_model, world_model, schedule)`: paths are generated by the world model (may differ from the pricing model: different ω, θ, ρ, or a different model class); at each rebalancing date the strategy rebalances hedge quantities computed under the pricing model (delta, cap-call strip, net-sized forward variance swaps, etc.); P&L accumulated to maturity with transaction costs (bid/ask in vol points on vanillas and var swaps, bps on spot).
- Output: P&L distribution (mean, std, quantiles, worst paths), regime breakdown (by realised vol, by realised skew), tables and figures in the study format.
- First regression targets: cliquet strategy ranking and the "delta + q-weighted cap calls + net-sized var swap, monthly" result with mean P&L within ±0.2% of notional across regimes.

---

## 9. Viewers

`viewers/precompute.py` builds grids into the cache: default grid `ω ∈ {0, 0.5, 1, 1.5, 2, 2.5, 3}`, `ρ1 ∈ {−0.9, −0.7, −0.5, −0.3, 0}`, `k1 ∈ {0.5, 1.5, 4}`, 2F presets (a few `(θ, k1, k2, ρ12)` combinations including the SSR ≈ 1.2 fit), always including the 1F degenerate points so the old studies are recoverable. Precompute is a CLI with resume support.

Streamlit pages (all read-only over the cache, Plotly figures, parameter sliders snap to the grid):
1. **Surface & model** — target surface, Dupire local vol, leverage function heatmap, calibration error map.
2. **Forward smile** — forward-start smiles `T1 → T2` vs parameters; forward ATM vol vs forward variance-swap vs forward vol swap; the put-wing invariance from the study.
3. **Forward skew** — forward ATM skew and curvature term structure vs ω, ρ, θ, k's.
4. **Smile dynamics** — conditional smile after a spot move, SSR term structure, vol-of-vol term structure, `Var(V)` decomposition.
5. **Model risk** — one product, all models calibrated to the same surface, price with error bars, side by side. This is the page the studies are really about.
6. **Product grid** — price heatmaps over two chosen parameters; vega ladders; spot-shift profiles; term-sheet editor for the product.
7. **Hedging** — P&L distributions of stored hedging runs.

Excel export button on every table (openpyxl).

---

## 10. Tests

Unit and property tests (pytest, fast, `n_paths` small with loose tolerances; a `slow` marker for the full-size ones):
- BS: put-call parity, implied-vol round trip, Greeks vs finite differences.
- SSVI: no-arbitrage checks; Dupire local vol positive; local-vol MC reprices SSVI at pillars to within tolerance.
- Variance swap: replication strike equals `E[realised variance]` under local vol MC; `ξ_0` integrates back to the strip.
- Bergomi: `E[ξ_t^T] = ξ_0^T` (martingale) for 1F and 2F; exact step (eqs. 7.15–7.18) vs fine Euler; `χ(t,T)` (eq. 7.35) vs sample variance of `x_t^T`; `Cov(ξ_u^u, ξ_v^v)` closed form vs MC; `Var(∫ξ)` closed form vs MC; θ = 0 reproduces the 1F model path-by-path with the same seed; vol of VS vol from simulation vs eq. 7.39; with Table 8.2 parameters and a flat 20% VS term structure, the mixing-solution ATMF skew vs eq. 8.55 and the numerical SSR (§4.4) vs eq. 9.21 (book Figure 9.1 dots vs line — tolerance consistent with 'order one in ν' accuracy, i.e. agreement within a few percent relative at 1y, not exact); mixing-solution smile vs spot-simulation MC smile within 2 stderr.
- LSV: calibration acceptance criterion of §4.2 on the reference surface; variance swaps unchanged across ω (surface-calibrated invariance); LSV SSR numerical estimator vs the analytic decomposition eq. 12.52 (order-one accuracy).
- Identity tests from the study: ATM implied variance equals the dollar-gamma-weighted average of realised variance along the path (verify to ≤ 0.1 vol point); forward-start ATM vol below forward vol swap when ρ < 0.
- PDE cross-check: 1F LSV, 2D finite differences (ADI) on `(ln S, X1)`, vanilla and forward-start prices vs MC within 2 stderr.
- Products: cliquet decomposition (strip of capped calls + global-floor put) reprices the product; KI put decomposition reprices the autocall's put leg; discrete barrier converges to continuous with the bridge correction.
- Risk: CRN bump Greeks vs analytic BS Greeks in the BS model; vega ladder sums to the parallel vega.
- Regression tests (`slow`, recalled from the original study — must be re-verified against the study archive; the SSVI parameters and seeds must be taken from that archive, not guessed): with the study surface, 1F, `ρ = −0.7`, `κ = 1.5`: 1y-into-1y ATM forward vol 21.5% (LV) → 18.6% (ω = 3), forward variance-swap level 25.2–25.5% across ω; 1y capped cliquet (2% cap, global floor 0) 1.082% / 1.194% / 1.472% / 1.763% of notional for ω = 0/1/2/3 with stderr 0.004–0.009%; 2y version 0.639% → 1.718%. Tolerance: 2 stderr, or 0.02% of notional if the seeds cannot be matched.

---

## 11. Conventions and quality bar

- Every equation implemented has a docstring stating the formula, its source (derived / Bergomi / Gatheral / Guyon–HL), and the test that checks it.
- No silent defaults for model parameters; configs are explicit YAML, validated with clear errors.
- Numba kernels are pure functions on arrays; Python wrappers do validation.
- Standard errors everywhere; no MC number printed without one.
- Logging via `logging`, not print. A `--profile` flag on the study runner.
- Reproducibility: every artefact (cache entry, study output) records the git commit, config hash and seed.
- Style: black, ruff, mypy (strict on `volsto/`), 100-char lines.
- README with a 20-line quickstart that calibrates a 1F LSV on the reference surface and prices the study cliquet.

---

## 12. Milestones (in order; each ends with green tests)

M1. Market layer + BS + local vol + MC engine + vanilla/variance products + tests §10 (BS, SSVI, var swap).
M2. Bergomi 2F kernel with exact stepping + closed-form tests + 1F degeneracy test.
M3. Particle calibration + cache + diagnostics; reproduce the study's calibration accuracy.
M4. Forward-start, cliquet family, forward smile analytics; regression tests against the study numbers.
M5. Risk layer (Greeks, ladders, profiles, vol-sto sensitivities with recalibration).
M6. Autocall/Phoenix/barrier products with decompositions; PDE 1F cross-check.
M7. Smile dynamics, SSR estimators, 2F fitting helper; Var(V) decomposition.
M8. Hedging framework; port the cliquet and FVA hedging studies as regression tests.
M9. Precompute CLI + Streamlit viewers + Excel export.
M10. Study runner with LaTeX output; regenerate the original paper's tables from the library.

Open items for the owner (do not block M1–M3): the study archive (zip with SSVI parameters, seeds, tables) for the regression tests; the 2F target parameterisation (SSR target, which maturities); whether hedging transaction-cost assumptions should follow the original study or be re-specified.

---

## 13. Addendum — market data import (milestone M3b, after M3)

Added by the owner during M1 (13 September 2026).

Add `volsto/market/import_hdn.py`: importer for the HistoricalData.net EOD option-chain CSV format (34 columns, one file per trading day; schema at https://historicaldata.net/options.html). The free sample `options_sample_2022H2.zip` is in `./data/hdn_sample/` (git-ignored). Pipeline, one function per step, each testable:

1. `load_day(path, underlying) -> DataFrame`. Keep only SPX and SPXW roots for the index surface; drop rows with missing bid or ask; record `settlement_time` (AM/PM) and use it in the time-to-expiry convention. Use the rate curve shipped in the ZIP manifest.
2. `implied_forward(chain, expiry) -> F`: from put-call parity on the mid prices near ATM (regression of C − P on K), with the discount factor from the rate curve. Do not use the vendor's iv or Greeks as inputs; they are cross-checks only (the vendor documents that quotes across contracts are not synchronized snapshots).
3. `to_grid_surface(chain, forwards) -> GridSurface`: OTM options only, mid implied vols against the implied forward, liquidity filter (min bid, max relative bid/ask spread in vol terms, using `iv_bid`/`iv_ask` where present), and butterfly/calendar arbitrage checks on the retained points.
4. `fit_ssvi(grid_surface) -> SSVISurface`: `theta_T` from ATM total variance, global `(rho, eta, gamma)` by least squares in vol space with the no-arbitrage constraints enforced; report residuals per expiry. Add an eSSVI option (slice-dependent rho) behind a flag for later single-stock use.
5. `snapshot_config(date) -> YAML` market config (surface params, forward curve, rate curve, provenance: vendor, file checksum, filters used) so a dated market snapshot runs through calibration exactly like the synthetic configs.

Tests: on one sample day, implied forwards agree with the vendor's parity-based forward (`iv_flag == 0` rows) to within a few bp; SSVI residuals inside ±20% moneyness ≤ 0.3 vol points for T ≤ 2y; the fitted surface passes the SSVI no-arbitrage checks; a CLI `volsto-import --vendor hdn --date 2022-09-15 --underlying SPX` writes the config. Also add `scripts/capture_yfinance.py`: saves today's SPX/SPY chain in the same 34-column layout (blank where unavailable) so a daily cron can accumulate history.

Milestone order becomes: M1, M2, M3, **M3b (market data import)**, M4, … M10.

Implementation notes (M3b, measured on the 2022 H2 SPX sample): contracts are grouped by (root, expiration) as the vendor does — SPX (AM) and SPXW (PM) share expiration dates but differ by one day in `T`; the implied forward is the regression `C − P = a + b K` on two-sided pairs within ±10% of spot (`F = −a/b`, discount `−b`, delta-method standard error), which agrees with the vendor's closest-strike parity forward within 0.7 bp to 6m and 3.5 bp at 2y (the vendor discounts at the Treasury rate, the market-implied rate was 0.4–0.6% higher); our Black-76 inversion reproduces the vendor's `iv` on `iv_flag = 0` rows to a median 0.1 vp near the money. Butterfly pruning is iterative convexity of OTM prices in strike; the calendar check runs on the common quoted `k` range of consecutive slices (a global fixed point dropping the worst violating quote), which is also what `GridSurface` enforces. `θ_T` is fitted at the SPEC pillar tenors (1m, 3m, 6m, 1y, 18m, 2y, 3y) by isotonic least squares through the slices' ATM total variances; the global `(ρ, η, γ)` least squares (vega-weighted, |k| ≤ 0.25, expiries ≥ 3 weeks) enforces both SSVI butterfly conditions through a bounded reparametrisation of `η`; `--essvi` fits `ρ` at the pillars (piecewise-linear `ρ_T`, `ESSVISurface`). Fit quality on four sample days, inside ±20% moneyness: RMS 0.12–0.22 vp and max 0.4–1.0 vp from 3m to 2y (eSSVI RMS 0.08–0.16 from 6m); the 1–2 month weeklies of these high-volatility days are 1–7 vp off — a single power-law φ cannot follow them, so the §13 target of 0.3 vp holds in RMS from 3m but not as a maximum below 3m. The snapshot YAML has `market` + `ssvi` sections (loadable by `load_ssvi_surface`) and a `provenance` section (vendor, file and manifest SHA-256, filters, forwards, rate curve, fit residuals, code version). `scripts/capture_yfinance.py` writes the same 34-column layout (blank calculated columns, `iv_flag = 7`) plus a manifest with a user-supplied Treasury curve.

---

## 14. Addendum — smile dynamics for delta and gamma (amends §7 `greeks.py`, milestone M5)

Added by the owner during M1 (13 September 2026).

Delta and gamma take a `smile_dynamics` argument:

- `"model"` — bump `S0`, leverage `L` and factors held fixed, CRN. Default.
- `"sticky_strike"` — bump `S0`, rebuild the target surface with total variance held fixed per strike `K` (convert SSVI to a strike grid before the bump), recalibrate `L`, reprice. Cached.
- `"sticky_moneyness"` — bump `S0`, keep the SSVI parameters (surface fixed in `k = ln K/F`), recalibrate `L`, reprice. Cached.

Report all three side by side in risk reports; the hedger uses `"model"` unless the strategy config overrides it.

Test: for a vanilla, the `sticky_strike` delta equals the BS delta at the market vol, the `sticky_moneyness` delta equals BS delta minus vega × ATM skew / `S0` to first order, and the model delta lies between them with the ordering set by the sign of the SSR minus one; the ATM vol shift under `"model"` per unit log-spot move equals SSR × ATM skew (ties §4.4 to the delta).
