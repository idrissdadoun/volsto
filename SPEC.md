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
Build the 3×3 covariance of `(δW^S, δX¹, δX²)` per distinct step size, Cholesky it once per step size, draw jointly. The factors are exact. The spot step is the **second-order SV step** (M4b, `SchemeConfig.sv_order2`, default): the factors are advanced first so `ξ_{t+δ}` is known exactly; the log-spot drift uses the trapezoid of the variance; and the weak order-2 Itô–Taylor terms of the spot increment that involve the variance factors are added explicitly — `¼ b₀ Σ_i c_i (δW̃_i δW^S − ρ_Si δ)` (the intra-step spot/variance covariance, with `δW̃_i = δX^i − (e^{−k_i δ} − 1) X^i` the exact factor increment's Brownian part and `c_i = ω α_θ w_i`) and `½ δ b₀ [½ ∂_t ln g − ½ Σ c_i k_i X^i + ⅛ Σ c_i c_j ρ_ij + ½ (b_x/b₀) Σ ρ_Si c_i] δW^S`, with `b₀ = L(t, S) sqrt(ξ_t^t)`; the leverage stays frozen at `t_n` (§3.5) and Platen's weak order-2 supporting values handle the `S`-dependence (§3.1). Lévy areas are dropped (zero mean; their omission is `O(ω² δ/32)` in relative variance). Measured on the pure 1F model (ω = 3, κ = 1.5, ρ = −0.7, flat ξ₀ = 4%, 400k paths, dt = 1/365, 1m): the frozen-variance step (`sv_order2=False`) reprices the variance swap 0.06 vp low (its time-averaged prefactor is paired with start-of-step factors, a `−¼ ω² δ χ'(t)` bias in variance) and the smile 0.14 vp low at k = −0.1 / 0.07 high at +0.1 (missing within-step covariance); the second-order step matches the mixing solution within 0.03 vp at all three strikes and the variance swap within 0.01 vp; at 1y both steps agree within noise (`tests/test_scheme.py::test_second_order_sv_step`). When no spot is needed (pure variance payoffs) the factors can be sampled directly at the fixing dates with no intermediate stepping.

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

LSV values measured after the pre-M4 amendments on the reference surface with the 1F (ω = 3, κ = 1.5, ρ = −0.7) and 2F (Table 8.2) kernels, 2·10⁵ particles, 3y horizon, the then-default step schedule 1/2920–1/730–1/500 (M4b restored 1/1460–1/365–1/250, see the M4b notes below) and 4·10⁵ pricing paths on three pricing seeds (MC standard errors 0.02–0.03 vp ATM, 0.05–0.10 vp at ±20–30%). A cell counts as a violation only if its error exceeds both the tolerance and 3 MC standard errors (`CalibrationReport.passes`). Variance swaps: pure LV within 0.03 vol points at all pillars once the Dupire grid spans ±3.0 in log-moneyness (±1.5 truncated the far put wing: 2y −0.08, 3y −0.22 vp). Two pre-M4 findings fixed the LSV residuals reported after M3: (i) the tail window floor is now `max(2000, 0.01 N)` particles (k-NN, `min_window` / `min_window_fraction`), which made the variance-swap residual monotone in N — at the final defaults the 1y strike reads +0.06 / 0.00 / −0.10 vp for 5·10⁴ / 2·10⁵ / 8·10⁵ particles (it was −0.14 at 2·10⁵ and +0.23 at 8·10⁵ with a fixed 2000-particle floor), i.e. monotone but not yet flat within noise: small clouds are tail-noise rich, large clouds expose the remaining O(dt) shortfall at 2y (−0.08) — every pillar 1m–3y is within 0.10 vp at 2·10⁵ and 8·10⁵ particles, 5·10⁴ particles reach +0.11 at 18m and +0.15 at 3y; (ii) the residual level itself (1y–2y put wing 0.10–0.15 vp cheap, 1y variance swap 0.14 vp low, present for every N and bandwidth) is the O(dt) time error of the LSV step (leverage and SV variance frozen over the step): halving the schedule removed it (put wing ≤ 0.04, variance swaps within 0.08 on three seeds), so the default schedule was halved at the cost of 2× calibration (≈ 61 s for 3y at 2·10⁵ particles) and pricing time. The structural fix — a second-order SV spot step (Andersen-type, exact factor increments with the intra-step spot/variance covariance) — would recover the cost; it is scheduled with the 3y far-put-tail item before M6. The particle calibration uses exactly the pricing kernel, scheme options and step schedule (`SimConfig`), with every leverage lookup inside a step taken from the step-start slice (frozen-L rule) in both calibration and pricing. Known residual: the 2F far right tail at 1m (+20%, 3.4 σ√T, a sub-basis-point option) is 1.4 vp rich because E[ξ|S] must be extrapolated beyond the particle cloud there. For reference, plain log-Euler at uniform dt = 1/365 biased the 1m ATM vol by +0.36 vp and the θ = η = ½ predictor–corrector by +0.78 vp (curvature over-correction); the weak order-2 scheme gives +0.03 vp.

Implementation notes (M4b — second-order SV step; measured on the placeholder reference surface with the 1F ω = 3, κ = 1.5, ρ = −0.7 kernel unless stated).

1. *Kernel.* The Bergomi / LSV spot step is second order in the SV variance (§3.3): exact factor increments first, trapezoidal variance in the drift, explicit weak order-2 spot/variance cross terms; the leverage stays frozen at `t_n`. Pure 1F model at dt = 1/365, 1m: the variance-swap error goes from −0.06 vp (frozen step) to 0.00 and the smile matches the mixing solution within 0.03 vp (the frozen step was 0.14 vp flatter at k = −0.1). 2F Table 8.2 at 3m and 1y, dt = 1/365: within 2 combined stderr of the mixing solution run at dt = 1/2920 at all five strikes — after a bug in the first implementation (the factor-Brownian proxy subtracted the mean reversion twice and cancelled the `−½ Σ c_i k_i X_i` drift term; +0.12 vp at k = +5% for the 2F set) was caught by that test; `tests/test_bergomi.py` now carries an independent numpy transcription of the step. Under common random numbers (`refinement_study`, 1y, dt = 1/365 → 1/1460) the kernel alone — pure SV and a fixed steep leverage `L = e^{2.3 k}` — has step errors ≤ 0.03 vp with the frozen step and ≤ 0.01 vp with the second-order step on the variance swap, the ATM and the 80% put: the kernel was never the dominant term of the calibrated LSV's coarse-schedule error, because the calibration absorbs any variance-level bias of the step.
2. *Calibration.* Two changes in `calibrate_leverage`: the regression grid is adaptive — 201 points over the cloud's trusted quantile range at each slice instead of a fixed 0.025 spacing over the whole leverage range (at the first slices the cloud is 0.006 wide while `E[ξ|k]` varies like `exp(ω ρ k / (σ√t))`; the fixed grid biased the short end low: 1m ATM −0.09 vp, 3m −0.05 at ω = 3, on both schedules) — and the plug-in curvature correction differences `m` on a stencil of width ≈ h (neighbouring points of the dense grid amplified the regression noise by `(h/d)² ≈ 25` before the one-sided clip). Tested and dropped (≤ 0.01 vp effect): the step average of `σ_loc²` in the target and a linear-in-time end-of-step predictor of the next slice fed to the weak order-2 supporting values; the frozen-L rule of §3.5 stands.
3. *Calibration noise.* With `N = 2·10⁵` the variance-swap error of a single calibration fluctuates by ±0.035 vp (1σ) per pillar across particle seeds (12345 / 777 / 4242 / 99); the pre-M4 "−0.14 vp at 1y on the coarse schedule" was the low seed 12345 on top of the fixed regression grid. Acceptance figures must be read on seed averages or at `N = 8·10⁵` (whose particle set contains the `2·10⁵` one, position-addressed draws, so the two are correlated).
4. *Schedule* (three particle seeds, pricing seed 2, 4·10⁵ paths; errors in vol points, the LV pricing noise of seed 2 — up to −0.06 vp on 2y puts — not subtracted):

| configuration | VS 1m | 3m | 6m | 1y | 18m | 2y | 3y | ATM 1m | 1y | 18m | 2y | 3y | calibration |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| second order, 1/1460–1/365–1/250 (default) | +0.03 | +0.02 | −0.02 | −0.04 | −0.05 | −0.06 | 0.00 | −0.06 | −0.03 | −0.01 | 0.00 | +0.06 | 33 s |
| frozen, 1/1460–1/365–1/250 | −0.02 | +0.02 | −0.01 | 0.00 | −0.01 | −0.02 | +0.02 | −0.07 | −0.05 | −0.03 | −0.03 | +0.02 | 33 s |
| second order, 1/2920–1/730–1/500 | −0.03 | +0.01 | +0.01 | −0.02 | −0.03 | −0.04 | +0.03 | −0.02 | +0.07 | +0.08 | +0.06 | +0.06 | 67 s |
| frozen, 1/2920–1/730–1/500 | −0.05 | −0.01 | 0.00 | −0.03 | −0.03 | −0.05 | +0.03 | −0.02 | +0.06 | +0.07 | +0.05 | +0.05 | 67 s |

Single-run spread ±0.04 vp on the variance swap. The coarse schedule is the default again: it reprices as well on average at half the cost, and the halved schedule carries a systematic +0.05–0.08 vp ATM bias at 1y–3y with *either* step, i.e. the bias sits in the calibration at fine slices (the M3.2 record with the fixed regression grid had |ATM| ≤ 0.05 there), not in the spot step; its origin is open.
5. *Acceptance against the §12 M4b criteria.* Calibration time: met (33 s at `N = 2·10⁵`, 124 s at `8·10⁵`). Variance swap ≤ 0.05 vp at every pillar: met on seed averages at 1m–6m and 3y, missed by 0.01 at 1y–2y (−0.04 to −0.06), and not met by single runs at `N = 2·10⁵` because of the ±0.04 noise; the `8·10⁵` single run (seed 12345) gives 1m–6m within 0.03, 1y–2y −0.05 to −0.06, 3y +0.02. Put wing at −20% / −30%, 1y–3y: within 0.05 vp + 2 SE at every pillar (excess over the LV pricing baseline ≤ 0.03). Open items: 1m ATM −0.06 vp (all seeds, both schedules), 3y ATM +0.06 on the coarse schedule, the fine-schedule long-end bias — all calibration-side; a seed-averaged (or `8·10⁵`) calibration is the recommendation for production numbers until they are resolved.

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

Implementation notes (M4, measured on the placeholder reference surface). **Products.** `ForwardStartOption(t1, t2, k, cp, pay_time=None)` pays `(cp (S_T2/S_T1 − k))⁺`; `k` is a moneyness (book §3.1; `k = 1` is eq. 3.13), `t1 = 0` is allowed (a vanilla on `S_T2/S_0`) and the pay date may be deferred past `T2` so that every leg of a cliquet decomposition settles on the cliquet's maturity; a `k = 0` call pays the forward return itself. `ForwardStartStraddle` decomposes into the call and the put. `FVA(t1, t2, K_vol, moneyness=F(T2)/F(T1))` pays `|R − m| − Straddle_Black(m, m, τ, K_vol)` at `T2` (the relative-performance FVA of book §3.1.9, footnote 9 — the FX form `(S_T2 − k S_T1)⁺` of eq. 3.18 is not implemented), so its fair strike is exactly the forward ATM-forward implied vol. `AdditiveCliquet(fixings, local_floor, local_cap, global_floor, global_cap)` pays `clip(Σ clip(r_i, LF, LC), GF, GC)`; `decompose()` uses `clip(x, LF, LC) = LF + (x − LF)⁺ − (x − LC)⁺` (with no local floor `x = (R − 0)⁺ − 1`) and `clip(Σ, GF, GC) = Σ + (GF − Σ)⁺ − (Σ − GC)⁺`: cash + long forward-start calls struck `1 + LF` − short calls struck `1 + LC` + a put on the accumulated sum struck `GF` (`AccumulatedSumOption`) − a call struck `GC`; the identity is exact path by path (`tests/test_cliquet.py`). `AdditiveCliquet.study(T)` is the study structure (monthly, local cap 2%, no local floor, global floor 0). `ReverseCliquet(coupon, LF, GF)` = `C + clip(Σ clip(r_i, LF, 0), GF − C, ∞)` decomposes through the same identities; `Napoleon` (coupon + worst period return) has no additive decomposition. **Analytics.** `analytics/forward_smile.py` prices out-of-the-money forward-start options on a moneyness grid (ATM-forward strike added) and inverts them with the model's own forward ratio `F_R = F(T2)/F(T1)`, `τ = T2 − T1` and `DF(T2)`; vol standard errors are price standard errors over the Black vega. `forward_vol_comparison` prices on one path set the forward ATM-forward vol, the forward variance swap on the simulation grid (the calibration diagnostics' convention) and the forward vol swap on daily fixings; `put_wing_table` lays several models' forward smiles side by side with the spread across models per strike. The BS closed form (eqs. 3.1–3.2), path-wise parity, the FVA fair strike and the ordering ATM forward vol < vol-swap vol < variance-swap vol for `ρ < 0` (pure 1F SV, ω = 3: 14.6 < 15.4 < 20.2%) are the fast tests.

Headline comparison (M4; 400k paths, seed 2024, default schedule and scheme shared with calibration; leverage calibrated at N = 2·10⁵, 3y horizon, 60–66 s per model; pricing 21–37 s per model for the whole set; `scripts/m4_headline.py`, baselines in `tests/test_m4_regression.py` with tolerance max(2 stderr, 0.02% of notional / 0.02 vol points)):

| model | 1y→2y ATMF forward vol | forward VS vol | forward vol-swap vol | cliquet 1y (% notional) | cliquet 2y (% notional) |
|---|---|---|---|---|---|
| LV (ω = 0) | 21.51 ± 0.04 | 25.21 ± 0.02 | 22.52 ± 0.01 | 1.198 ± 0.004 | 0.744 ± 0.004 |
| 1F ω = 1 (ρ = −0.7, κ = 1.5) | 20.84 ± 0.04 | 25.24 ± 0.02 | 22.32 ± 0.01 | 1.317 ± 0.004 | 0.941 ± 0.005 |
| 1F ω = 2 | 19.74 ± 0.04 | 25.24 ± 0.03 | 21.41 ± 0.01 | 1.624 ± 0.005 | 1.442 ± 0.006 |
| 1F ω = 3 | 18.49 ± 0.04 | 25.18 ± 0.03 | 20.10 ± 0.02 | 1.966 ± 0.005 | 2.005 ± 0.006 |
| 2F Table 8.2 | 19.20 ± 0.03 | 25.19 ± 0.03 | 21.02 ± 0.01 | 1.740 ± 0.005 | 1.780 ± 0.006 |

The surface's own forward 1y→2y variance-swap level (log-contract replication) is 25.24%; every model is within 0.06 vol points of it.

M4b rerun of the same set (second-order SV step, schedule 1/1460–1/365–1/250, adaptive regression grid; calibrations 36–39 s per model, pricing 12–23 s per model; these are the regression baselines in `tests/test_m4_regression.py`):

| model | 1y→2y ATMF forward vol | forward VS vol | forward vol-swap vol | cliquet 1y (% notional) | cliquet 2y (% notional) |
|---|---|---|---|---|---|
| LV (ω = 0) | 21.44 ± 0.04 | 25.25 ± 0.02 | 22.53 ± 0.01 | 1.193 ± 0.004 | 0.754 ± 0.004 |
| 1F ω = 1 | 20.76 ± 0.04 | 25.23 ± 0.02 | 22.31 ± 0.01 | 1.314 ± 0.004 | 0.951 ± 0.005 |
| 1F ω = 2 | 19.69 ± 0.04 | 25.19 ± 0.03 | 21.38 ± 0.01 | 1.624 ± 0.005 | 1.459 ± 0.006 |
| 1F ω = 3 | 18.45 ± 0.04 | 25.09 ± 0.03 | 20.05 ± 0.02 | 1.968 ± 0.005 | 2.024 ± 0.006 |
| 2F Table 8.2 | 19.17 ± 0.03 | 25.16 ± 0.03 | 21.00 ± 0.01 | 1.742 ± 0.005 | 1.805 ± 0.006 |

What moved against the M4 table (M4b − M4, in units of the combined stderr): every entry by less than 2 σ except the 2y cliquet at ω = 2 / 3 / 2F (+0.017 / +0.019 / +0.025% of notional, 2.1 / 2.2 / 2.9 σ; pure LV moved +0.010, 1.8 σ, so about half is the schedule's effect on the monthly capped returns common to all models) and the ω = 3 forward vol swap (−0.05 vp, 2.3 σ). The forward ATM vols moved −0.03 to −0.07 vp (≤ 1.3 σ), the forward VS levels by ≤ 0.09 vp (≤ 1.9 σ), the 1y cliquets by ≤ 0.005%. The forward-smile put-wing invariance is unchanged: spread 0.20 vp at k = 0.8 (z = 1.7) against 2.9–3.8 vp at and above the money. Against the study numbers recalled in §10: the ATM forward vol (21.5% → 18.6% for ω = 0 → 3) and the forward VS level (25.2–25.5%) are reproduced on the placeholder surface to 0.1 vol points; the cliquet prices are about 10% higher than the study's at every ω (1y 1.198 / 1.317 / 1.624 / 1.966 vs 1.082 / 1.194 / 1.472 / 1.763; 2y 0.744 → 2.005 vs 0.639 → 1.718) while the ratios across ω agree to 1% (1y ×1.64 vs ×1.63, 2y ×2.70 vs ×2.69) — a level difference consistent with a surface or convention difference (rates, day count, the study's exact schedule) rather than a dynamics difference; it cannot be resolved without the archive. Put-wing invariance (1y-into-1y forward smile, strikes relative to `S_T1`): at `k = 0.8` the five models agree within 0.27 vol points (26.65–26.92, z = 2.2) whereas the spread is 1.2 at 0.9, 3.0 at the money, 3.8 at 1.1 and 3.5 at 1.2 — the model-invariant part of the forward smile is the put wing. Forward ATM vol < forward vol swap < forward variance swap holds for every model including pure LV (21.5 < 22.5 < 25.2).

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
- Regression tests (`slow`, recalled from the original study — must be re-verified against the study archive; the SSVI parameters and seeds must be taken from that archive, not guessed): with the study surface, 1F, `ρ = −0.7`, `κ = 1.5`: 1y-into-1y ATM forward vol 21.5% (LV) → 18.6% (ω = 3), forward variance-swap level 25.2–25.5% across ω; 1y capped cliquet (2% cap, global floor 0) 1.082% / 1.194% / 1.472% / 1.763% of notional for ω = 0/1/2/3 with stderr 0.004–0.009%; 2y version 0.639% → 1.718%. Tolerance: 2 stderr, or 0.02% of notional if the seeds cannot be matched. M4 status: the archive is still absent (tests skipped); the same set measured on the placeholder surface is the regression baseline (`tests/test_m4_regression.py`, `slow`), see the §6 notes.

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
M4b (owner, before M5; done 2026-09-13, see §4.2 notes — calibration time met, variance-swap criterion met on seed averages except 1y–2y by 0.01, single runs at 2·10⁵ particles are noise-limited). Second-order SV spot step (exact factor increments with the intra-step spot/variance covariance, Andersen-type), shared by calibration and pricing under the frozen-L-per-step rule. Acceptance on the previous coarse schedule (1/1460 – 1/365 – 1/250): variance-swap error ≤ 0.05 vol points at every pillar 1m–3y for N = 2·10⁵ and 8·10⁵, put wing ≤ 0.05 vol points at 2 SE, calibration time back to ~30 s; re-run the M4 tables and report what moved.
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

Implementation notes (M3b, measured on the 2022 H2 SPX sample): contracts are grouped by (root, expiration) as the vendor does — SPX (AM) and SPXW (PM) share expiration dates but differ by one day in `T`; the implied forward is the regression `C − P = a + b K` on two-sided pairs within ±10% of spot (`F = −a/b`, discount `−b`, delta-method standard error), which agrees with the vendor's closest-strike parity forward within 0.7 bp to 6m and 3.5 bp at 2y (the vendor discounts at the Treasury rate, the market-implied rate was 0.4–0.6% higher); our Black-76 inversion reproduces the vendor's `iv` on `iv_flag = 0` rows to a median 0.1 vp near the money. Butterfly pruning is iterative convexity of OTM prices in strike; the calendar check runs on the common quoted `k` range of consecutive slices (a global fixed point dropping the worst violating quote), which is also what `GridSurface` enforces. `θ_T` is fitted at the SPEC pillar tenors (1m, 3m, 6m, 1y, 18m, 2y, 3y) by isotonic least squares through the slices' ATM total variances; the global `(ρ, η, γ)` least squares (vega-weighted, |k| ≤ 0.25, expiries ≥ 3 weeks) enforces both SSVI butterfly conditions through a bounded reparametrisation of `η`; imported surfaces default to eSSVI — `ρ` fitted at the pillars (piecewise-linear `ρ_T`, `ESSVISurface`) — with `--ssvi` as the single-`ρ` opt-out; synthetic surfaces (`configs/surfaces/reference_ssvi.yaml`, the study parameters) stay plain SSVI. Expiries under 3m are reported in the residual tables but sit outside the acceptance region. Fit quality on four sample days, inside ±20% moneyness: RMS 0.12–0.22 vp and max 0.4–1.0 vp from 3m to 2y (eSSVI RMS 0.08–0.16 from 6m); the 1–2 month weeklies of these high-volatility days are 1–7 vp off — a single power-law φ cannot follow them, so the §13 target of 0.3 vp holds in RMS from 3m but not as a maximum below 3m. The snapshot YAML has `market` + `ssvi` sections (loadable by `load_ssvi_surface`) and a `provenance` section (vendor, file and manifest SHA-256, filters, forwards, rate curve, fit residuals, code version). `scripts/capture_yfinance.py` writes the same 34-column layout (blank calculated columns, `iv_flag = 7`) plus a manifest with a user-supplied Treasury curve.

---

## 14. Addendum — smile dynamics for delta and gamma (amends §7 `greeks.py`, milestone M5)

Added by the owner during M1 (13 September 2026).

Delta and gamma take a `smile_dynamics` argument:

- `"model"` — bump `S0`, leverage `L` and factors held fixed, CRN. Default.
- `"sticky_strike"` — bump `S0`, rebuild the target surface with total variance held fixed per strike `K` (convert SSVI to a strike grid before the bump), recalibrate `L`, reprice. Cached.
- `"sticky_moneyness"` — bump `S0`, keep the SSVI parameters (surface fixed in `k = ln K/F`), recalibrate `L`, reprice. Cached.

Report all three side by side in risk reports; the hedger uses `"model"` unless the strategy config overrides it.

Test: for a vanilla, the `sticky_strike` delta equals the BS delta at the market vol, the `sticky_moneyness` delta equals BS delta minus vega × ATM skew / `S0` to first order, and the model delta lies between them with the ordering set by the sign of the SSR minus one; the ATM vol shift under `"model"` per unit log-spot move equals SSR × ATM skew (ties §4.4 to the delta).

---

## 15. Addendum — conditional and knock-out variance products (amends §6; after M4b)

Add after the cliquet family; they reuse the realised-variance accumulators. Notation: daily closes `S_0..S_N` on the fixing schedule, `r_i = ln(S_i/S_{i-1})`, `A = 252`, barrier `B`, strike `K` quoted as a volatility, variance notional `N_var`.

New file `volsto/products/conditional_variance.py`:

1. `ConditionalVarianceSwap(barrier, side, indicator, convention, strict=True, daily_cap=None)`. `side`: `"up"` (accrue where `S > B`) or `"down"` (`S < B`). `indicator`, applied to the side's inequality: `"prev"` `I_i = 1{S_{i-1} in region}`; `"curr"` `I_i = 1{S_i in region}`; `"both"` `I_i = 1{S_{i-1} in region} · 1{S_i in region}`. Desk conventions (constructor helpers `UpVar`, `DownVar`): up-var uses `"prev"` or `"both"`; down-var uses `"curr"` or `"both"`. The indicator is a required argument on the base class, never defaulted. `D = Σ_i I_i`. Convention `"conditional"`: payoff `N_var [ (A/N) Σ_i r_i² I_i − K² D/N ]`, zero if `D = 0`. Convention `"corridor"`: payoff `N_var [ (A/N) Σ_i r_i² I_i − K² ]`. `strict` selects `>` / `<` versus `>=` / `<=`. `daily_cap c` replaces `r_i²` by `min(r_i², c²)`. `fair_strike()`: `K² = A E[Σ r_i² I_i] / E[D]` for conditional (ratio of expectations, stderr by the delta method), `A E[Σ r_i² I_i]/N` for corridor.
2. `ConvexitySpread(upvar, varswap, notional_ratio=1.0)`: long the conditional product, short the plain variance swap on the same schedule; `decompose()` returns the two legs. Report the strike differential `K_up² − K_var²`.
3. `KnockOutVarianceSwap(barrier, direction="up", strict=True)`. Close-to-close monitoring only: `j = min{ i : S_i > B }`, `τ = min(j, N)`. Variant (b) only: payoff `N_var [ (A/N) Σ_{i≤τ} r_i² − K² τ/N ]`, where the KO day's own return `r_j` accrues. `fair_strike()`: `K² = A E[Σ_{i≤τ} r_i²] / E[τ]`, delta-method stderr. Expose `P(KO)` and `E[τ]` as diagnostics. Continuous monitoring and variants (a)/(c) are deferred (raise `NotImplementedError`).

Analytics (`volsto/analytics/conditional_variance.py`): for each product, the LSV fair strike minus the pure local-vol fair strike on the same surface and seed, as a function of the model parameters; that difference is the study quantity.

Tests: Gyöngy invariance — the corridor-convention swap with a single-close indicator (`"prev"` for up, `"curr"` for down), priced on a daily grid, gives the same fair strike under LV and under LSV for ω ∈ {1, 2, 3} within 2 stderr, and equals the LV quantity `(A/N) Σ_i E[σ_loc²(t_i, S_{i-1}) dt · I_i]` computed on the LV paths. Complementarity — up `"prev"` + down `"prev"` with the same `B` and corridor convention equals the plain variance swap path by path; up `"both"` + down `"both"` equals the variance swap minus the variance on barrier-crossing days, path by path. Ordering on the reference (negatively skewed) surface — `K_up < K_var < K_down` for `B ∈ {90%, 100%, 110%}` of spot, and `K_KO > K_var` for `B ∈ {105%, 110%, 120%}`, all at 3 stderr, under LV and LSV (document in the test that the KO ordering is a property of negative skew and non-inverted term structure, not a theorem). KO var — `B → ∞` recovers the plain variance swap; the fair strike is monotone in `B`; `P(KO)` decreases in `B`. Conditional-convention `D/N` scaling — with `B` far outside the spot range the conditional up-var equals the plain variance swap. `decompose()` of `ConvexitySpread` reprices it.

Add these products to the M4 headline table (fair strikes with stderr for ω = 0/1/2/3 in 1F and for the Table 8.2 2F set; `B = 100%` for up/down var, `B = 110%` for KO var, 1y maturity, daily fixings).

---

## 16. Addendum — volatility knock-out put (amends §6; with the §15 products)

`VolKnockOutPut(strike, maturity, vol_ko, fixing_schedule, daily_cap=None, monitoring="maturity")` (same file as §15 or `volsto/products/vko.py`). Realised volatility over the life on the fixing schedule, `σ_real² = (A/N) Σ_{i=1}^N r_i²` (`r_i²` capped at `c²` if `daily_cap`). Payoff `(K − S_T)⁺ · 1{σ_real < vol_ko}`. `monitoring "maturity"` (default, the traded form): the condition is checked once at `T` on the full-life realised vol. `monitoring "running"` (flag): knock out on the first day the accrued variance `Σ r_i²` exceeds `vol_ko² N / A`, i.e. the option is dead as soon as the full-life realised vol can no longer be below the barrier; document that this is not the traded convention. Reference terms: 100% or 95% strike, 12m, `vol_ko = 30%` on SPX. `decompose()`: vanilla put minus the "vol-knock-in" put `(K − S_T)⁺ 1{σ_real ≥ vol_ko}`; reprices path by path. Report price, the ratio to the vanilla put (the "VKO discount") and `P(knock-out)`.

Why it is in the study: the price is `E[(K − S_T)⁺ 1{RV < H²}]`, the joint law of terminal spot and realised variance. Under local vol, realised variance is nearly a deterministic function of the path's spot levels, so the LV VKO price is close to a hard threshold on `S_T`; stochastic vol spreads RV conditional on `S_T`, and the spread is governed by ν and ρ. Expect the largest model dependence of any product in the library. The study quantity is the VKO discount versus the vanilla put as a function of `(ν, ρ, θ)` and of `vol_ko ∈ {25, 30, 35, 40}%`.

Tests: `vol_ko → ∞` recovers the vanilla put; `vol_ko → 0` gives zero; the price is monotone increasing in `vol_ko` and never above the vanilla put. `decompose()` reprices path by path. With the reference surface (ATM vol ~20%), 12m ATM put, `vol_ko = 30%`: the LSV discount is materially larger than the LV discount at ω = 2, 3 (assert the sign and report the values; no fixed number until the archive tests exist). `"running"` monitoring price ≤ `"maturity"` monitoring price path by path.

Add to the headline table: 12m 100% put VKO at `vol_ko = 30%`, price as % of notional, VKO discount versus vanilla, and `P(KO)`, for ω = 0/1/2/3 in 1F and the Table 8.2 2F set.

---

## 17. Addendum — M5 risk layer (extends §7 and §14; four parts, the later wins where they overlap: the vega-T "wave" block of part C replaces the vega-T paragraph of part B)

Reporting: same format as the milestone reports, plus the performance budget (number of recalibrations, wall clock) for a full `RiskReport`.

### 17.A General machinery, delta/gamma, vega, forward-variance ladder, profiles, parameter sensitivities, attribution

General machinery (`volsto/risk/engine.py`): `BumpSpec(name, apply: Model|Surface|State -> bumped, size, scheme="central"|"forward")` and a `RiskEngine` that prices base and bumped states under common random numbers (same seed, same grid) and returns `Sensitivity(value, stderr)` where `stderr` is that of the *difference*, estimated path by path. Any bump that changes the surface or a model parameter triggers a recalibration through `LeverageCache.get_or_calibrate`; bumps that change only `S0` or the factor state do not. Every recalibrating bump is a cache entry, so the second run of a ladder is free. Surface perturbations are an additive perturbation layer on `ImpliedSurface` (`δσ(k, T)` added in implied vol), reused by every bump and delta regime below, with the no-arbitrage checks re-run on the perturbed surface; if a check fails, halve the bump, retry and report. Default bump sizes: delta/gamma 1% of spot in log space (central three-point); vega 1 vol point of implied vol; forward-variance buckets +1 vol point of the bucket's forward VS vol; model parameters 5% relative (`ν, k1, k2`), 0.05 absolute (`θ`, correlations) with the PSD check re-run. All configurable.

Delta and gamma (`greeks.py`): `smile_dynamics` regimes as defined in 17.B (five regimes; `"model"` is the default and the one the hedger uses). Report all regimes side by side.

Vega (`greeks.py`): parallel bump, implied vol +1 vp at every `(K, T)`. Two variants: `"recalibrated"` (leverage refit to the bumped surface, default) and `"sticky_leverage"` (`L` held; only `ξ_0` moves with the bumped strip). Report both; the difference is the "leverage vega", a study quantity. Theta: price at `t + 1` business day with the surface held fixed in `(K, absolute expiry)`, factors at zero, forward rolled; only for products with no fixing in the roll window (raise otherwise). Test in the BS model: `θ + ½σ²S²Γ + r-terms = 0` for a vanilla within stderr.

Forward-variance vega ladder (`ladders.py`, `fwd_var_ladder`): buckets monthly to 1y, quarterly to 3y by default; configurable. Bump `ξ_0^T → (1+ε) ξ_0^T` for `T` in bucket `[T_i, T_{i+1}]`. Propagation to the surface: the VS total variance at maturity `T` shifts by `dW(T) = ε ∫_{bucket ∩ [0,T]} ξ_0`; apply the same shift additively to `w(k, T)` for every `k` (parallel shift in total variance per maturity, skew preserved in total-variance terms). Document this mapping. `ε` chosen so the bucket's forward VS vol rises by 1 vp. Units: % of notional per vol point of bucket forward VS vol. Variants `"recalibrated"` (default) and `"sticky_leverage"`. Test: for both variants the ladder sums to the corresponding parallel forward-variance bump within 2 stderr; for a plain variance swap the recalibrated ladder is flat and equals its analytic sensitivity.

Spot-shift profiles (`profiles.py`): `S0` shifts on a grid (default −30% to +30% in 2.5% steps), prices and model delta/gamma/vega at each shift under `"model"` dynamics by default (leverage fixed, factors at zero); optional other regimes, which recalibrate per shift (cached). Gamma profile as the second difference. Cliquet gamma profile also versus the accumulated sum at an intermediate date, reusing the Bachelier call-on-remaining-capped-sum decomposition from the original study as an analytic cross-check (exact under BS with independent legs; report the deviation under the LSV).

Vol-of-vol and model-parameter sensitivities (`volsto_sens.py`): partial sensitivities to each of `(ν, θ, k1, k2, ρ12, ρ_SX1, ρ_SX2)` with recalibration; for `ν` also the sticky-leverage variant. Central differences, PSD check on correlation bumps (reduce the bump if violated, report). Output as one table per product: value, stderr and the price per unit parameter under both variants.

P&L attribution (`attribution.py`): `explain(product, state_0, state_1)`: sequential CRN revaluation from `state_0` to `state_1` in the order spot (delta, gamma), surface (parallel vega, then vega-T waves, then skew/curvature), model parameters, factor state; residual = actual change minus the sum. States carry `S0`, surface, model params, factor values, date. Used by the hedger's regime breakdown in M8.

`RiskReport`: one object per product holding every sensitivity with stderr, the bump specs used, the cache keys of every recalibration, and `to_dataframe()` / `to_excel()`. Docstring must state: (i) wave projections equal single-pillar bumps only to first order (recalibration and vega convexity add a small cross term, tested within 2 stderr); (ii) an expiry between two pillars shows vega in both neighbouring projections, in proportion to its distance from each, because the bump lives at the pillars and the interpolation ramps it.

Performance: a full report (recalibrated fwd-var ladder of 18 buckets, vega-T waves, skew and curvature ladders, 5 regime deltas, parallel vega, 7 parameter sensitivities, cross-Greeks of 17.D) is on the order of 80–100 recalibrations. Print the count and the wall clock; that is the viewer precompute budget.

Tests (in addition to those inline above): BS model — delta, gamma, vega, theta versus analytic within 3 stderr for calls and puts; all delta regimes coincide in BS. Forward-start option before `T1` — model delta small relative to vega (report the ratio); sticky_moneyness delta ≈ 0 within stderr. Capped cliquet (study structure) — fwd-var ladder recalibrated vs sticky-leverage, reproduce the qualitative result of the original study (net ladder units: % of notional per vol point per forward month). Attribution — for a pure spot move the residual is at the gamma-cubed level (report), for a pure parallel vega move the residual is within 2 stderr.

### 17.B Skew and curvature risk, delta regimes

Skew and curvature risk (`ladders.py`, `skew_T` and `curvature_T`): pillars = the surface's expiry pillars (default 1m, 2m, 3m, 6m, 9m, 1y, 18m, 2y, 3y). `tent_i(T)` linear from 0 at `T_{i-1}` to 1 at `T_i` to 0 at `T_{i+1}`, flat beyond the last pillar and before the first. Skew bump `i`: `σ(k, T) → σ(k, T) + s · k · tent_i(T)`, with `s` chosen so that `σ(ln 0.9) − σ(ln 1.1)` at `T_i` rises by 1 vp (rotation around ATM). Units: % of notional per vol point of 90/110 skew at the pillar. Curvature bump `i`: `σ(k, T) → σ(k, T) + c · k² · tent_i(T)`, with `c` chosen so that `[σ(ln 0.9) + σ(ln 1.1)]/2 − σ(0)` at `T_i` rises by 1 vp. Units per vol point of 90/110 butterfly. Recalibrated and sticky-leverage variants; no-arbitrage checks as in 17.A. Tests: ATM vanilla has zero skew vega within stderr, a 90% put has positive skew vega; a 90/110 risk reversal has zero curvature vega within stderr, a 90/110 strangle has positive curvature vega; `skew_T` sums to a global rotation bump within 2 stderr.

Delta regimes (replaces §14 `smile_dynamics`): smile in log-moneyness `k = ln(K/S)`; `Δ = ln(S0_new / S0_old)`; `s_T` = ATM skew `dσ/dk` at maturity `T` from the surface.
- `"model"`: bump `S0`, `L` and factors fixed (default; the hedger's delta).
- `"sticky_strike"`: `σ_new(k, T) = σ_old(k + Δ, T)`.
- `"sticky_moneyness"`: `σ_new(k, T) = σ_old(k, T)`.
- `"sticky_skew"`: `σ_new(k, T) = σ_old(k, T) + s_T Δ` (ATM vol slides along the old smile; shape re-centred at the new spot with the same skew and curvature).
- `"sticky_local_vol"`: `σ_new(k, T) = σ_old(k, T) + 2 s_T Δ` (Derman's sticky-implied-tree regime, SSR = 2; reference point only).
All except `"model"` rebuild the surface as above, recalibrate `L` (cached) and reprice under CRN. Report all five deltas and gammas side by side. Tests: in BS all five coincide; for an ATM vanilla, sticky_strike and sticky_skew deltas agree to first order (difference bounded by the curvature term, report it); the ordering of ATM-vol shifts per unit `Δ` is `0 (moneyness) < s_T (strike, skew) < 2 s_T (local vol)`, and the `"model"` regime's shift equals `SSR_T · s_T` from §4.4 within tolerance (this test lands when the §4.4 estimator lands in M7; mark it skipped until then).

### 17.C Vega by maturity, desk "wave" convention (replaces any vega-T paragraph above)

Vega by maturity (`ladders.py`, `vega_T`): pillar bump `i`: `σ(k, T) → σ(k, T) + 0.01 · tent_i(T)`, `tent_i` as in 17.B (the bump lives at the pillars; interpolation ramps to the neighbours). Check the calendar condition on every ramp after bumping: at `σ = 20%` it requires `T_{i+1}/T_i ≥ 1.10`, which the default pillars satisfy; raise if a custom pillar set does not. Wave `j`: bump every pillar with `T_i ≤ T_j` simultaneously, i.e. the sum of tents `1..j`; wave `n` (all pillars) equals the parallel vega bump. Report both: cumulative vega per wave, and the vega projection at pillar `T_j = wave_j − wave_{j-1}`. Projections sum to the parallel vega by construction. Tests: the projection ladder equals the single-pillar tent ladder within 2 stderr; `wave_n` equals the parallel vega within 2 stderr. Recalibrated (default) and sticky-leverage variants. Units: % of notional per vol point.

### 17.D Additional Greeks

Cross-Greeks (`greeks.py`), all CRN central differences with stderr of the difference, under the `"model"` regime unless stated: vanna `d(vega)/d(ln S)` and `d(delta)/d(σ)`, both reported (they agree only in BS; the difference under the LSV is informative); volga `d(vega)/d(σ)`; charm and veta: one-business-day roll of delta and vega with the surface held in `(K, absolute expiry)`; rho: parallel +1 bp in rates; repo/dividend delta: +1 bp in `q`. Theta split into carry (rates/divs), pure decay (surface held in time-to-maturity) and roll-down (difference). Cross terms `d(delta)/d(ρ_SX1)`, `d(delta)/d(ρ_SX2)`, `d(delta)/d(ν)`: the hedge ratio's own dependence on the vol-sto parameters (recalibrated).

Product-specific (`risk/product_risk.py`): fixing risk — for forward-start and cliquet products, delta/gamma/vega at `T1` minus one day and `T1` plus one day (state held), reported as the jump; for cliquets, per fixing. Barrier risk — `d(price)/d(barrier)` for KO var and KI put; delta and gamma profile within ±5% of the barrier at 0.5% steps; for the VKO, `d(price)/d(H)` in vol points of the vol barrier and `d(P(KO))/d(ln S)`. Realised-variance exposure — expected dollar-gamma-weighted variance per fixing period along the path, as a profile; for a variance swap it must be flat and equal to the notional within stderr. Second-order forward-variance ladder — diagonal convexity per bucket (three-point second difference on the bucket bump); off-diagonal cross terms only on request.

Precision options (`risk/estimators.py`): likelihood-ratio delta and vega (score function on the Gaussian draws) for discontinuous payoffs (barriers, KO var, VKO, digitals), with the bump estimate as the cross-check; report both with stderr and flag when they differ by more than 3 stderr. Conditional Greeks at a future date `t`: regression of the discounted payoff and its CRN bumps on a polynomial basis in `(ln S_t, X1_t, X2_t)`, returning delta, gamma, vega as functions of the state; this is the hedger's engine in M8. Control variate on the difference: for vanilla-like products use the BS Greek at the market vol as control variate on (bumped minus base) and report the variance reduction.

Tests: vanna and volga in BS versus analytic within 3 stderr; realised-variance exposure flat for a variance swap; fixing-risk jump for a forward-start straddle is a vega-to-delta conversion (report); likelihood-ratio and bump delta agree within 3 stderr on a digital; conditional delta at `t` regressed equals the `t = 0` delta when `t → 0`.
