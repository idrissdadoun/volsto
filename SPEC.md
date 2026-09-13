# volsto — Stochastic-volatility pricing library for light-exotic and exotic parameter studies

Specification v2.0 — 14 September 2026
Consolidates v1.1 plus every addendum agreed during M1–M4 (scheme, calibration, importer, products, risk layer M5, fitting M7). Where a number here conflicts with a measured value already recorded in the repository's SPEC.md by the implementer, the measured value wins; where a definition conflicts, this document wins. §17 lists the changes.
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

Language and stack: Python 3.12+ (numpy ≥ 2.5 dropped 3.11), numpy, scipy, numba (path loop and kernel regression), pandas, pyarrow (cache), plotly + streamlit (viewers), pytest, pyyaml. No pandas inside numba kernels. Type hints everywhere. No global state.

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
      import_hdn.py      # HistoricalData.net EOD chain importer (§13)
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
      history.py         # historical estimators from a surface history (§15)
      fit_2f.py          # staged fit of the 2F parameters (§15)
      stability.py       # rolling refits and identifiability diagnostics (§15)
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
      conditional_variance.py  # up/down var, convexity spread, KO var (§6.1)
      vko.py             # volatility knock-out put (§6.2)
      autocall.py        # single-underlying autocall, Phoenix (memory), KI put decomposition
      barrier.py         # discrete/continuous KI/KO, barrier shift, digital
    risk/
      greeks.py          # bump-and-reprice with CRN
      ladders.py         # forward-variance vega ladder, both variants
      profiles.py        # spot-shift and gamma profiles
      volsto_sens.py     # sensitivities to (omega, theta, k1, k2, rho...) with recalibration
      attribution.py     # P&L explain
      engine.py          # BumpSpec / RiskEngine / RiskReport (§7.1, §7.13)
      product_risk.py    # fixing, barrier and realised-variance risks (§7.10)
      estimators.py      # likelihood-ratio, conditional and control-variate Greeks (§7.11)
    hedging/
      instruments.py     # hedge instruments priced under a pricing model
      hedger.py          # rebalancing loop, realised-world model, P&L distribution
      report.py
    analytics/
      forward_smile.py   # forward-start implied smiles, forward ATM vol vs forward var swap
      smile_dynamics.py  # conditional smile after spot move, SSR, vol-of-vol term structure
      var_decomp.py      # Var(V) decomposition (predictable at T1 vs within-period), closed forms
      conditional_variance.py  # LSV-minus-LV fair strikes vs parameters (§6.1)
      vix.py             # VIX futures/options by 2D quadrature in the 2F model (§15 Part 5)
      mixing.py          # mixing solution for pure-SV vanilla smiles (ch. 8 App. A)
    pde/
      lsv1f.py           # 2D finite-difference pricer for the 1F degenerate case (validation only)
    studies/
      m4.py              # headline study runner behind the M4/M4b tables (implemented)
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

`ESSVISurface` — SSVI with a per-pillar ρ_T (Hendriks–Martini conditions); the default for imported market surfaces (§13), since a single power-law φ cannot follow 1–2m index weeklies. Plain SSVI stays the default for synthetic surfaces.

`GridSurface` — market slices (K or delta, T) with arbitrage-free interpolation in total variance (linear in `w` along `k`, linear in `w` along `T` at fixed `k`), extrapolation flat in implied vol beyond wings. Enough for feeding a Bloomberg export later.

### 2.3 Dupire local vol
`LocalVolSurface.from_implied(surface)`:
`σ_loc²(k,T) = ∂_T w / (1 − k/w ∂_k w + ¼(−¼ − 1/w + k²/w²)(∂_k w)² + ½ ∂_kk w)`, derivatives by central finite differences on the analytic SSVI (step sizes configurable), floored at a small positive value, and stored on a (T, k) grid with bilinear interpolation: square-root-spaced t grid (400 points), k grid ±3.0 with dk = 0.0025 (measured: ±1.5 made 2y/3y variance swaps 0.08/0.22 vp low). Provide `check_positive()` diagnostic.

### 2.4 Variance swaps and ξ₀
`varswap_strike(surface, T)`: log-contract replication `K_var(T) = (2/T) ∫ [P(K)/K² (K<F) + C(K)/K² (K>F)] e^{rT} dK`, adaptive quadrature in `k` over `[−k_max, k_max]` with `k_max = max(25 σ_ATM √T, 3)` (measured: SSVI put wings decay slowly — `k_max = 2.4` at 1y truncated 0.01 vp on the reference surface, and a Dupire grid limited to ±1.5 priced the 2y/3y variance swaps 0.08/0.22 vp low). `xi0_curve(surface)`: forward variance `ξ₀(T) = d/dT [T · K_var(T)]` via a PCHIP interpolant of the strip so positivity and exact integration back to the strip hold by construction.

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

Factors are stepped exactly (§3.3). The spot step is Platen's explicit weak order-2 scheme by default (`scheme="weak2"`), with `log_euler`, `local_var_time_average` and `predictor_corrector` (Andersen θ = ½ with the Kloeden–Platen Itô drift correction) kept as options. Milstein is deliberately not offered: it raises strong, not weak, order and its correction term has zero mean. All leverage lookups within a step use the step-start slice of L ("frozen-L rule"), identically in the particle calibration and in pricing, through one shared stepping routine.

Time grid: `StepSchedule` on `SimConfig`, a piecewise-constant dt schedule shared by engine and calibration. Default after M4b: 1/1460 for t < 3m, 1/365 to 2y, 1/250 beyond (the finer 1/2920–1/730–1/500 schedule was needed before the second-order SV step of M4b). The union of fixing dates is always included. `SimConfig.record_all_steps` / `Product.requires_all_steps` make every grid step a record column (Brownian-bridge barriers, short-horizon SSR estimator), with a chunk memory budget.

Second-order SV spot step (M4b; cross-term compensator corrected at M6 — the exact step covariance `ρ_Si (1 − e^{−k_i δ})/k_i` in place of `ρ_Si δ`, whose difference was a first-order drift error linear in `k_i`, §4.2 M6 Part 0 item 7): exact factor increments with the intra-step spot/variance covariance (Andersen-type), removing the O(dt) error from freezing leverage and SV variance over the step; acceptance: on the coarse schedule, VS error ≤ 0.05 vp at every pillar 1m–3y for N = 2·10⁵ and 8·10⁵ particles, put wing ≤ 0.05 vp at 2 SE, 3y calibration ≈ 30 s. **Status (measured, §4.2 M4b notes):** accepted on seed averages — calibration 33 s at 2·10⁵ particles; variance swap within 0.05 vp at 1m–6m and 3y, −0.04 to −0.06 at 1y–2y; single runs at 2·10⁵ particles carry ±0.035 vp of particle-seed noise per pillar. The kernel's own step error at dt = 1/365 was ≤ 0.03 vp even with the frozen step (CRN refinement), so the second-order step's visible gains are on the pure SV model (1m: variance swap −0.06 → 0.00 vp, smile 0.14 vp flatter → within 0.03 vp of the mixing solution) while the calibrated-LSV improvements came from the calibration side.

Implemented scheme switches (measured on the reference surface, 1m ATM, dt = 1/365, 200k paths): `SchemeConfig` / `SimConfig` expose boolean flags rather than a `scheme` string — `weak_order2` (default; +0.03 vp), `local_var_time_average` (time-averaged local/forward variance, +0.29 vp alone), `predictor_corrector` with `pc_eta` (θ = η = ½, +0.78 vp: it over-corrects the local-variance curvature 2×, kept as an option), `local_var_time_eval` (`"start"` / `"midpoint"`, diagnostic), plain log-Euler (+0.36 vp) when the first three are off, and `sv_order2` (default True, the M4b step; off = the frozen-variance step of M2/M3). `volsto.engine.refinement_study` measures such biases under common random numbers (Talay–Tubaro extrapolant).

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
Target: `L(t,S)² = σ_loc²(t,S) / E[V_t | S_t = S]`, with `L` stored in `k = ln(S/F(t))`.

Algorithm on the shared `StepSchedule` with the shared spot step (frozen-L rule):
1. `L(0,·) = σ_loc(0,·)/sqrt(ξ_0^0)`.
2. Step all `N` particles with the current slice of `L`.
3. Estimate `E[V | S]` at the new slice by **local-linear kernel regression in k with a plug-in ½h²m″ curvature correction**, evaluated on a 201-point grid spanning the particle cloud's trusted `[q, 1−q]` quantile range at each slice (adaptive since M4b; a fixed grid over the whole leverage range biased the short end low, §4.2) and interpolated onto the fine leverage grid (dk = 0.0025, the Dupire grid), with the ½h²m″ term differenced on a bandwidth-wide stencil; bandwidth `h = c·σ_ref·sqrt(t)·N^{−1/5}` with `c = 1.5` and a k-NN window floor of `max(2000, 0.01 N)` particles (so the tail bandwidth does not shrink with N); saturating log-quadratic tails outside the cloud. Nadaraya–Watson with flat tails remains an option (measured 0.27 vp max error vs 0.11 for the default; VS +0.34 vs +0.07).
4. Set the new slice and continue. Optional second pass with a fresh seed, averaged.

Defaults: `N = 2·10⁵`, leverage grid floor ±2.5 in k (±2.08 made 3y variance swaps 0.18 vp rich), 3y horizon. Timings at the M4b scheme: ≈ 30 s per 3y calibration; the §9 default grid (105 points) ≈ 1 h. Production numbers — headline tables, regression baselines, viewer precompute — use `N = 8·10⁵` with a single seed (owner decision at M4b acceptance; ≈ 124 s per 3y calibration, so the §9 grid ≈ 4 h); development and fast paths keep `2·10⁵`.

Implementation notes (M3, measured on the reference surface, see `ParticleConfig`): the regression is local-linear rather than Nadaraya–Watson (NW carries the design bias `h² m′ f′/f`, which with `c = 1.5` skewed the ±10% repricing by 0.3 vol points), with a plug-in `½ h² m″` curvature correction (otherwise a −0.10 vp level bias at 1y), a 2000-particle window floor in the tails, and `E[V|S]` extrapolated beyond the trusted quantiles with a saturating log-quadratic (the flat rule mis-priced the 3m +30% call by 1 vp and variance swaps by 0.3 vp). `E[V|S]` is estimated on the 201-point grid and interpolated onto the fine leverage grid (dk = 0.0025, the Dupire grid) where `σ_loc²` is resolved. All lookups within a simulation step use `L(t_n, ·)` (frozen-leverage rule), identically in calibration and pricing.

### 4.2 Diagnostics
`calibration/diagnostics.py`: `CalibrationReport` repricing the target surface on pillars T ∈ {1m, 3m, 6m, 1y, 18m, 2y, 3y} × k ∈ ±{0, 0.05, 0.1, 0.2, 0.3} via MC (CRN, `n_paths ≥ 4·10⁵`), implied-vol error in vol points with MC stderr, plus variance-swap strikes vs replication. Acceptance is noise-aware (an error fails only if it exceeds both the tolerance and 3 stderr) and bounded at 2.5 ATM standard deviations; the tolerances and the measured values are the table below (the slow test's gate is `CalibrationReport.passes(0.15 vp, T ≤ 2y, |k| ≤ 0.2, z = 3, max_std = 2.5)`); variance swaps ≤ 0.10 vp at every pillar, ≤ 0.05 targeted after M4b (met on seed averages except 1y–2y at −0.04 to −0.06, see the M4b notes). The 2F 1m +20% cell (a 3.4σ sub-basis-point option) sits outside the acceptance region.

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
5. *Acceptance against the §12 M4b criteria.* Calibration time: met (33 s at `N = 2·10⁵`, 124 s at `8·10⁵`). Variance swap ≤ 0.05 vp at every pillar: met on seed averages at 1m–6m and 3y, missed by 0.01 at 1y–2y (−0.04 to −0.06), and not met by single runs at `N = 2·10⁵` because of the ±0.04 noise; the `8·10⁵` single run (seed 12345) gives 1m–6m within 0.03, 1y–2y −0.05 to −0.06, 3y +0.02. Put wing at −20% / −30%, 1y–3y: within 0.05 vp + 2 SE at every pillar (excess over the LV pricing baseline ≤ 0.03). Open items, tagged **calibration-side pass before M6**: 1m ATM −0.06 vp (all seeds, both schedules), 3y ATM +0.06 on the coarse schedule, the fine-schedule long-end bias of +0.05 to +0.08 vp at 1y–3y. Owner's prescribed first diagnostic for that pass (M4b acceptance): calibrate `L` on the daily slice grid but simulate on the fine schedule with `L` interpolated linearly in `t` between slices; if the long-end bias disappears, the cause is regression bias compounding with the slice count and the fix is bandwidth/stencil scaling with the slice spacing, not the scheme — report before changing anything. Production convention (owner decision at M4b acceptance): headline tables, regression baselines and viewer precompute use `8·10⁵` particles with a single seed; development and fast paths keep `2·10⁵`; the M4 regression baseline is re-recorded once at `8·10⁵` in M4c with the particle count noted in the baseline file.

**M6 Part 0 — calibration-side pass (2026-09-13, measured; in progress, stopped for the owner's decision).** Variants at 8·10⁵ particles, seeds 12345 / 777 / 4242, 4·10⁵ pricing paths (`scripts/m6_calibration_pass.py`, tables in `outputs/m6/part0_report.md`): A = calibrate and price on the coarse schedule (production); B = the coarse-calibrated `L` priced on the fine schedule with `L` interpolated linearly in `t` between the coarse slices (the diagnostic prescribed at the M4b acceptance); C = calibrate and price on the fine schedule; B2 = coarse `L` held per slice on the fine schedule; D = fine `L` priced on the coarse schedule. Findings so far:
1. *Self-consistency on the fine schedule (owner item 1).* The particle loop and the pricing kernel agree to the last bit (max |Δ ln S| = max |ΔX| = 0) at the slices nearest 1y, at 2y and at 3y on the fine and on the coarse schedule, on the calibration grid and on the engine-built grid (the engine grid coincides with the calibration grid when the fixings are slice times). Note: the calibration grid refines each schedule interval evenly, so its steps beyond 3m are 1/730.29 (fine) and 1/365.14 (coarse) and `t = 1.0` is not a slice on either schedule; a 1y fixing adds one step (harmless under the frozen rule, `L` interpolated at that single step start).
2. *ω = 0 isolation (owner item 2).* With ν = 0 the particle-calibrated LSV and the Dupire local vol priced on the same schedule agree within 0.005 vp at every strike from 1y to 3y (fine and coarse), and neither shows the long-end ATM bias on the fine schedule (LSV₀ ATM +0.008 / +0.020 / +0.013 / −0.031 vp at 1y / 18m / 2y / 3y). The +0.05–0.08 vp fine-schedule bias of the 1F set is therefore in the stochastic-volatility part of the step, not in the local-vol machinery or the regression.
3. *N-scaling (owner's ratio-bias mechanism).* The 1F residuals are unchanged from 2·10⁵ to 8·10⁵ particles on both schedules (fine ATM 1y +0.065 → +0.059, 3y +0.063 → +0.054; coarse ATM 1y −0.035 → −0.045, 3y +0.057 → +0.056; wings likewise), so the residual is not the `Var(X̂)/E[X̂]³` plug-in bias of the ratio `σ_loc²/Ê[V|S]` (which would fall 4×); the 3.2·10⁶ seed (12345, pricing seed 2) confirms it at 16× the particles: coarse 1y ATM −0.046 (2·10⁵ −0.050, 8·10⁵ −0.049), fine 1y ATM +0.058 (+0.057, +0.058), 18m +0.081 (+0.087, +0.079) — the same numbers, i.e. the same pricing realisation (item 4c).
4a. *Separating variants.* B2 (coarse `L` held per coarse slice, fine steps) reproduces B on both sets (1F ATM +0.067 / +0.055 / +0.034 at 1y / 18m / 2y; 2F 1y wings −0.08, ATM +0.08): the linear-in-`t` interpolation of `L` is not the driver — the finer steps under the same frozen-per-slice `L` are. D (fine `L` on coarse steps) is not realisable through the engine, whose grid always contains the leverage slices (grid ⊇ slices, §5): the run reproduced C to the last digit and is dropped. The 1y ATM moves +0.10 vp from A to B2 while the fixed-leverage CRN study at 2y–3y gives +0.02 per halving of a uniform 1/365 step, so the schedule dependence is concentrated where the two schedules differ most — the sub-3m region (1/1460 → 1/2920) and the [3m, 2y] steps — a decomposition by region is being measured.
4b. *CRN comparison of the production schedule against its global halving, same frozen-per-slice `L`, Brownian-consistent paths (`CoarsenedDraws`), 4·10⁵ paths, fixings at slice times.* 1F ω = 3 (8·10⁵ coarse `L`): fine − coarse = −0.007 / −0.010 / −0.011 / −0.017 vp ATM at 1y / 18m / 2y / 3y, +0.005 to +0.013 in the wings, −0.006 to −0.011 on the variance swaps (stderr 0.001–0.002): the 1F kernel's step effect at daily steps is ≤ 0.02 vp everywhere. 1F ω = 0: ≤ 0.02 vp (largest on the 3m variance swap, −0.018). 2F Table 8.2 (8·10⁵ coarse `L`): −0.015 / −0.029 / −0.047 / −0.061 / −0.072 / −0.105 vp ATM at 3m / 6m / 1y / 18m / 2y / 3y and +0.02 to +0.06 in the wings — a real, seed-independent step error of the 2F kernel at daily steps, about half of the 2F coarse-schedule residual. **Consequence for the A / B / C tables:** the paired step effect of the 1F set is an order of magnitude smaller than the +0.10 vp A → B move at 1y ATM read from those tables; the tables' MC standard error per cell is 0.03 vp ATM and 0.05 vp in the wings (4·10⁵ paths, seed 2 for every run), and that pricing realisation is *common* to the three particle seeds of a variant (the seed SE of 0.002–0.01 measures calibration noise only), so unpaired differences between variants of ±0.1 vp are two to three standard errors of the pricing noise; a pricing-seed scan (six seeds, 1F A and C) is being run to size it. The M4b acceptance residuals (1m ATM −0.06, 3y +0.06, fine long end +0.05–0.08) were measured the same way (pricing seed 2 throughout) and are subject to the same caveat.
4d. *Acceptance convention from M6 on (owner decision).* Repricing diagnostics average over pricing seeds: `reprice_surface_seeds(model, surface, sim, pricing_seeds)` (default six seeds in `scripts/m6_calibration_pass.py` and `scripts/m4b_acceptance.py`; the reported `stderr_vp` is the empirical standard error across seeds, `mc_stderr_vp` the quoted single-run MC error), or paired (CRN) estimates when two configurations are compared; single-seed residual tables are no longer an acceptance basis.
4c. *Pricing-seed scan (the decisive check).* The same coarse-calibrated 1F `L` (8·10⁵, particle seed 12345) repriced on its own schedule with pricing seeds 2–7 (4·10⁵ paths each) gives 1y ATM errors from −0.049 (seed 2, the Part 0 / M4b seed) to +0.042 vp (std 0.032, matching the quoted MC SE 0.027) and 2y ATM from −0.029 to +0.088 (std 0.043); the fine-calibrated `L` on the fine schedule likewise (1y ATM +0.058 at seed 2, −0.054 at seed 6, std 0.038). Averaged over the six pricing seeds (seed SE 0.013–0.02 vp): coarse ATM +0.002 / +0.018 / +0.025 / +0.041 at 1y / 18m / 2y / 3y, fine ATM +0.019 / +0.028 / +0.029 / +0.036; `k = −0.2` coarse −0.073 / −0.064 / −0.048 / −0.064, fine −0.033 / −0.024 / −0.045 / −0.032; variance swaps −0.107 / −0.100 / −0.092 / −0.010 (coarse) and −0.084 / −0.091 / −0.097 / −0.005 (fine), std 0.02. **Reading:** the long-end ATM residuals tagged at M4b ("fine +0.05–0.08", "coarse 1y −0.05, 3y +0.06") were the seed-2 pricing realisation, not calibration biases; averaged over pricing seeds the 1F set reprices the smile within +0.04 vp ATM and −0.07 vp in the wings on both schedules, the fine-schedule numbers sitting where the CRN kernel study puts the second-order step's own error (§4.2 item 6); the one 1F residual that is real, schedule- and `N`-independent is the **1y–2y variance swap at −0.10 vp** with the |k| ≤ 0.3 vanillas within noise, i.e. the far put wing beyond the regression's trusted quantiles (the log-quadratic tail), the item noted at M4b as "VS not flat across N"; the 3y variance swap is within 0.01.
4. *B ≈ C on both sets.* The coarse-calibrated `L` priced on the fine schedule reproduces the fine calibration's residual (1F: ATM +0.060 / +0.085 / +0.060 / +0.054 vs +0.059 / +0.080 / +0.052 / +0.054 at 1y / 18m / 2y / 3y), i.e. the calibrated leverage functions agree across schedules and the schedule dependence sits in the pricing kernel of the SV step; the owner's reading: the calibration absorbs the frozen-`L` step error, so an `L` calibrated at daily steps is only valid at daily steps (B is not a defect).
5. *2F Table 8.2 set (8·10⁵, seed-stable).* On the coarse schedule the calibrated 2F model misprices the long end far outside the gate: ATM +0.06 / +0.11 / +0.09 / +0.20 vp and the −20% / −30% wings −0.09 / −0.14 / −0.15 / −0.21 vp at 1y / 18m / 2y / 3y (variant A); the fine schedule halves it (C: ATM +0.05 / +0.07 / +0.08 / +0.09, wings −0.01 / +0.01 / −0.03 / −0.10) and B ≈ C again. The 2F kernel's step error at daily steps (k₁ = 5.35, ν = 1.74) is not absorbed by the calibration; the ν-halved set (ν = 0.87) and the fixed-leverage 2F convergence at 2y / 3y are being measured to test the `ν²` scaling of the frozen-`L` × fast-factor interaction.
6. *Fixed-leverage kernel convergence at 2y / 3y (1F, CRN, dt 1/365 → 1/730 → 1/1460, 4·10⁵ paths).* Pure SV, second-order step: ATM +0.012 / +0.006 vp per halving at 2y and +0.016 / +0.008 at 3y, wings −0.012 / −0.007 (first order: ≈ +0.03 vp ATM and −0.025 wings of total step error at dt = 1/365 and 3y). Pure SV, frozen step: converged within ±0.003 vp at both maturities (at long maturities the frozen step carries less step error than the second-order step, whose gain was the 1m smile, §4.2 M4b). Calibrated `L` (8·10⁵, coarse) held as the given function of `(t, k)` with the second-order step: ATM +0.022 / +0.010 per halving at 2y and 3y, variance swap +0.012 / +0.005, wings −0.004 to −0.009, i.e. ≈ +0.045 vp ATM and −0.02 wings of step error at dt = 1/365 — a quarter of the +0.07 to +0.10 ATM schedule dependence seen between variants A and B at 1y–2y, so most of A − B is the frozen-per-slice versus interpolated-in-`t` definition of `L`, which variant B2 isolates. The steep fixed `L = e^{2.3k}` rows are noise-dominated at 2y–3y (the variance explodes in the far wing) apart from the puts (−0.006 / −0.003 per halving).
7. *2F kernel (pure SV, `L ≡ 1`, CRN dt 1/365 → 1/730 → 1/1460, 4·10⁵ paths).* Second-order step, ν = 1.74: ATM +0.080 / +0.040 vp per halving at 2y and +0.102 / +0.052 at 3y, wings −0.045 / −0.023 and −0.053 / −0.027, variance swap +0.009 / +0.007 — a first-order weak error of ≈ +0.16 vp ATM at 2y and +0.2 vp at 3y at daily steps, −0.1 in the wings; the frozen step at the same maturities: within ±0.005 vp. This is the 2F coarse-schedule residual of item 5 (ATM +0.20, wings −0.21 at 3y) and its halving on the fine schedule: the second-order SV step of M4b, whose gain was the 1m pure-SV smile, carries an O(dt) error at long maturities that grows with the mean-reversion speed and the vol of vol (1F: +0.02 per halving at 2–3y, item 6; 2F k₁ = 5.35, ν = 1.74: +0.08–0.10) and that the leverage calibration does not absorb. The ν-halved 2F set (ν = 0.87, coarse / coarse, 8·10⁵, three seeds) roughly halves the residual (1y ATM +0.035 vs +0.06, 18m wings −0.09 vs −0.14, 2y wings −0.10 vs −0.15): not the ν² scaling of a pure frozen-`L` × fast-factor interaction — "something else in the 2F step" in the owner's dichotomy; the pure-SV kernel rows at ν = 0.87 give ATM +0.051 / +0.026 per halving at 2y and +0.067 / +0.034 at 3y, wings −0.028 / −0.014 and −0.035 / −0.018 — 0.64× the ν = 1.74 values, not 0.25× (a ν² law) nor 0.5× (linear); the frozen step is again within ±0.002 vp. Both steps converge to the same limit (2y ATM: frozen 20.653%, second-order 20.816 → 20.736 → 20.696, extrapolating to 20.656), the second-order step approaching it first-order from above. ν and k₁ sweep of the second-order step's first-order error (pure SV, 2y ATM, coarse − fine per halving of the daily step, 2·10⁵ paths, CRN): at k₁ = 5.35, ν = 0.435 / 0.87 / 1.74 / 2.61 → +0.028 / +0.051 / +0.080 / +0.086 vp (wings −0.015 / −0.028 / −0.045 / −0.049): sub-linear and saturating in ν; at ν = 1.74, k₁ = 1.5 / 3.0 / 5.35 / 10 → +0.013 / +0.038 / +0.080 / +0.164 (wings −0.009 / −0.023 / −0.045 / −0.089): close to linear in the mean-reversion speed; the 1F kernel at κ = 5.35 (ν = 1.5) gives +0.075 against +0.010 at κ = 1.5. The error is therefore driven by the factor's mean reversion within the step. **Diagnosis (owner decision: option (b)).** The cross term `¼ b₀ Σ c_i (dwt_i dW^S − ρ_Si δ)` already uses the Brownian part of the exact OU increment (`dwt_i = ∫ e^{−k_i(δ−u)} dW^i`, my item-7 duplication hypothesis was wrong), but its compensator was `ρ_Si δ` while `E[dwt_i dW^S] = ρ_Si (1 − e^{−k_i δ})/k_i`: the difference `ρ_Si k_i δ²/2` per step is a drift of `ln S` of `⅛ b₀ ρ_Si c_i k_i δ` per unit time — a first-order weak error linear in `k_i` and in the correlation × vol of vol, which moves the ATM call up and the puts down (a forward error, the signature seen in every CRN table) and which the leverage calibration cannot absorb. The fix (M6) uses the exact step covariance `chol[1+i,0]·chol[0,0]` as the compensator; the numpy transcription in `tests/test_bergomi.py` carries the same change; the 1F / 2F mixing-solution tests at 1m / 3m / 1y pass unchanged. Acceptance per the owner: pure 2F SV, `L ≡ 1`, CRN halving of the daily step, 2y and 3y ATM and wings ≤ 0.01 vp per halving at `(ν, k₁) = (1.74, 5.35)` and `(1.74, 10)`; then the 2F set recalibrated at 8·10⁵ and the residual table rerun with six pricing seeds (gate 0.05 vp at 2 SE, 1y–3y, both sets). Fallback (a): the frozen step as default with the second-order step kept as a labelled option. **Acceptance met (2026-09-13):** with the exact-covariance compensator the second-order step's CRN error per halving of the daily step on the pure 2F SV kernel (`L ≡ 1`, 4·10⁵ paths) is, at `(ν, k₁) = (1.74, 5.35)`: 2y ATM −0.0060 ± 0.0012 vp (was +0.080), `k = −0.2` −0.0019, `k = −0.3` −0.0015; 3y ATM −0.0055, wings −0.0003 / −0.0008; at `(1.74, 10)`: 2y ATM −0.0062 (was +0.164), wings −0.0026 / −0.0027; 3y ATM −0.0058, wings −0.0014 / −0.0006 — worst 0.0062 vp against the 0.01 criterion; the frozen step's rows at the same points stay within ±0.005 (its wings −0.002 to −0.005); the variance swaps +0.008 to +0.013 per halving for the second-order step (the criterion is on the vanillas). The 1F / 2F mixing-solution tests at 1m / 3m / 1y and the numpy transcription pass with the corrected compensator; the code tag is bumped to `m6`, every cached leverage function is recalibrated at 8·10⁵ on the production schedule, and the residual tables are rerun with six pricing seeds (items 4d / 3 of the owner's decision).

8. *Gate after the step fix (tag `m6`, production schedule, 8·10⁵ particles, three particle seeds × six pricing seeds, `outputs/m6/calibration_pass_fixed_step`).* Seed-averaged implied-vol errors (vp; the SE across pricing seeds is 0.013–0.028 per cell, across particle seeds 0.002–0.011):

| set | T | ATM | k = −0.2 | k = −0.3 | VS |
|---|---|---|---|---|---|
| 1F ω = 3 | 1y | −0.010 | −0.050 | −0.055 | −0.080 |
| 1F | 18m | −0.007 | −0.039 | −0.047 | −0.086 |
| 1F | 2y | +0.001 | −0.024 | −0.033 | −0.084 |
| 1F | 3y | +0.005 | −0.032 | −0.035 | −0.009 |
| 2F Table 8.2 | 1y | −0.000 | −0.034 | −0.039 | −0.035 |
| 2F | 18m | −0.014 | −0.026 | −0.027 | −0.035 |
| 2F | 2y | −0.020 | −0.000 | −0.002 | −0.024 |
| 2F | 3y | −0.025 | −0.005 | −0.004 | +0.056 |

**The gate (ATM and −20% / −30% wings within 0.05 vp at 2 SE from 1y to 3y, both sets) is met**: every 2F cell is within 0.04 (the 3y wings went from −0.21 to −0.005 vp with the corrected step), the 1F ATM cells within 0.01, and the 1F 1y wings sit at the 0.05 edge (−0.050 / −0.055, within at 2 SE). What remains is the far-wing item: the 1F variance swap at −0.08 vp from 1y to 2y (the −0.10 of item 4c, now measured on six pricing seeds) with the 2F set at −0.035 there and +0.056 at 3y — the tails beyond the regression's trusted quantiles.
10. *Headline and regression baselines on the corrected step.* `scripts/m4_headline.py` rerun at 8·10⁵ particles (tag `m6`): every one of the 80 baseline entries of `tests/test_m4_regression.py` moved by at most 2.4 stderr against the M4c record (one beyond 1.5: the 2F 1y→2y vol-swap vol, +0.04 vp); the 1y→2y ATMF forward vol reads 21.46 / 20.81 / 19.69 / 18.44 (LV, ω = 1 / 2 / 3) and 19.17 (2F), the 1y cliquet 1.199 / 1.320 / 1.628 / 1.970 / 1.743 % of notional, the KO variance swap 29.07 / 28.65 / 27.66 / 26.46 vol points, the VKO ratio at 30% 0.325 / 0.310 / 0.324 / 0.343 — the M4/M4c conclusions stand; baselines re-recorded (`outputs/m6/baseline_moves_m6.csv`).
9. *Far-wing tails (owner item 4, bounded attempt; 1F, 8·10⁵, production schedule, three particle seeds × six pricing seeds).* Continuing `ln E[V|S]` beyond the trusted quantiles with the model's own conditional slope — the whole-cloud fit (`cloud_slope`) or the pure-SV Gaussian slope (`sv_slope`, which overstates the measured pure-SV slope by 28% at ω = 3) — moves the residual the *wrong way*: variance swap 1y / 18m / 2y −0.182 / −0.182 / −0.162 (`cloud_slope`) and −0.237 / −0.246 / −0.243 (`sv_slope`) against −0.080 / −0.086 / −0.084 with the default saturating log-quadratic; the −30% put at 1y −0.071 / −0.079 against −0.055. Both model-consistent tails raise `E[V|S]` in the far put wing (β < 0), lower `L` there and take more variance out of the wing, while the M3 flat rule (+0.34 vp on the variance swap) errs the other way: the far-put-wing `E[V|S]` the surface needs lies *between* the flat and the saturating-quadratic continuations, i.e. below the model's own conditional slope. **Recorded as a known far-wing bias and closed for M6**: 1y–2y variance swap −0.08 vp (1F ω = 3), 2F −0.035 at 1y–2y and +0.056 at 3y, with the |k| ≤ 0.3 smile within the gate; the default tail stays `log_quadratic`, the two model-consistent tails remain available as `ParticleConfig.tail_extrapolation` options. No code change has been made to the calibration or the kernel pending the owner's decision; the measured options are (a) the frozen SV step for pricing and calibration (converged within ±0.005 vp at 2–3y for both sets; loses the M4b 1m pure-SV gain, which the calibration absorbs for the LSV), (b) a corrected second-order step (the first-order error grows with k₁ and ν and is absent from the frozen step, which points at the step's cross / L⁰ terms rather than the factor dynamics), or (c) a finer schedule for the 2F set (halving to 1/730 halves the error to ≈ 0.1 vp at 3y, at twice the cost).

### 4.3 Cache
Content-addressed: key = SHA-256 of (surface params, curves, model params, particle config, seed, calibration code tag). The code tag is manual (so the precompute is not invalidated by every commit); `code_tag_guard.json` stores hashes of the particle, leverage, LSV, Bergomi and local-vol modules and a test fails when they change without a tag bump. Store leverage `.npz` + diagnostics `.json` + a `manifest.parquet` row. `get_or_calibrate(cfg)` is the only entry point studies and viewers use. Calibration must never run silently inside a viewer; the viewer reads the cache and reports what is missing.

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

- `TimeGrid.build(fixing_dates, dt_max, calibration_grid=None)`: union, sorted, with `fixing_index` map. When an LSV model is used the grid contains the calibration time slices (`Model.required_times()`), so that the frozen-L rule of §3.1 applies identically in calibration and pricing; interpolating `L` linearly in `t` on a finer grid is the diagnostic prescribed for the calibration-side pass (§4.2 M4b notes), not the pricing rule.
- `GaussianDraws(seed, n_paths, n_steps, n_brownians, antithetic=True)`: PCG64 generator; the same `(seed, path index, step, brownian index)` always gives the same normal so CRN bumps are exact. Generate in blocks to bound memory; `n_paths` default 2·10⁵, chunk 5·10⁴.
- `MonteCarlo.price(product, model, grid, draws, cv=None) -> PriceResult(mean, stderr, n_paths, per_path_payoffs optional)`.
- Control variates: vanilla with the same maturity (analytic BS price under the model's implied vol at that strike from the target surface — valid because the LSV reprices the surface) and variance swap (replication strike). Coefficient estimated on the sample; report variance reduction.
- All results carry standard errors; the library never returns a bare float for a MC quantity. `engine/stats.py` provides batch-means standard errors for variance-type statistics.
- Realised-variance accumulators: `int_var` is a left-point Riemann sum (its exact discrete moments are exposed by `BergomiSV.integrated_variance_moments` for tests); realised variance of products is always the sum of squared log returns on **fixing** dates.

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

### 6.1 Conditional and knock-out variance products (`products/conditional_variance.py`)
Notation: daily closes `S_0..S_N` on the fixing schedule, `r_i = ln(S_i/S_{i−1})`, `A = 252`, barrier `B`, strike `K` quoted as a volatility, variance notional `N_var`.

**ConditionalVarianceSwap(barrier, side, indicator, convention, strict=True, daily_cap=None).** `side` "up" (accrue where S > B) or "down" (S < B). `indicator` is required, never defaulted: `"prev"` I_i = 1{S_{i−1} in region}; `"curr"` I_i = 1{S_i in region}; `"both"` I_i = 1{S_{i−1} in region}·1{S_i in region}. Desk conventions: up-var "prev" (T−1) or "both" (T & T−1); down-var "curr" (T) or "both". `D = Σ I_i`.
- conditional: `Payoff = N_var [ (A/N) Σ r_i² I_i − K² D/N ]`, zero if D = 0 (quoted as `(D/N)(σ_cond² − K²)` with `σ_cond² = (A/D) Σ r_i² I_i`).
- corridor: `Payoff = N_var [ (A/N) Σ r_i² I_i − K² ]`.
`strict` selects >/< vs ≥/≤; `daily_cap` c replaces r_i² by min(r_i², c²). `fair_strike()`: `K² = A E[Σ r_i² I_i]/E[D]` (conditional; ratio of expectations, delta-method stderr) or `A E[Σ r_i² I_i]/N` (corridor). Ordering on a negatively skewed surface: `K_up < K_var < K_down` for any B (Gyöngy: the strike averages local variance over the accrual region).

**ConvexitySpread(upvar, varswap, notional_ratio=1.0)**: long the conditional product, short the plain variance swap; `decompose()` returns the legs; reports `K_up² − K_var²`.

**KnockOutVarianceSwap(barrier, direction="up", strict=True)**: up-and-out, close-to-close monitoring, `j = min{i : S_i > B}`, `τ = min(j, N)`, variant (b) only: `Payoff = N_var [ (A/N) Σ_{i≤τ} r_i² − K² τ/N ]` (the KO day's return accrues). `fair_strike()`: `K² = A E[Σ_{i≤τ} r_i²]/E[τ]`; diagnostics P(KO), E[τ]. Continuous monitoring and variants (a) nothing paid / (c) unscaled strike raise NotImplemented. Note for the study document: `K_KO > K_var` holds under negative skew and a non-inverted term structure (survival-weighted average of local variance over lower spot states) but is not a theorem — it can flip for symmetric smiles or inverted term structures; the model dependence enters through the hitting probabilities, which differ between models sharing all marginals.

Analytics (`analytics/conditional_variance.py`): LSV fair strike minus pure-LV fair strike, same surface and seed, versus model parameters.

Tests: Gyöngy invariance (corridor, single-close indicator, daily grid: same strike under LV and LSV for ω ∈ {1,2,3} within 2 stderr, equal to `(A/N) Σ E[σ_loc²(t_i,S_{i−1}) dt · I_i]` on LV paths); complementarity (up "prev" + down "prev" corridor = variance swap path by path; up "both" + down "both" = variance swap minus crossing-day variance); ordering at 3 stderr for B ∈ {90,100,110}% (up/down) and {105,110,120}% (KO); KO limits (B → ∞ recovers the var swap; strike monotone in B; P(KO) decreasing in B); conditional up-var with B outside the spot range equals the var swap; ConvexitySpread decompose reprices.

### 6.2 Volatility knock-out put (`products/vko.py`)
**VolKnockOutPut(strike, maturity, vol_ko, fixing_schedule, daily_cap=None)**: `σ_real² = (A/N) Σ r_i²` over the life; `Payoff = (K − S_T)⁺ · 1{σ_real < vol_ko}`. The condition is checked once at maturity only (M4c review: a running check against the full-life budget is the same event because the accrued sum is monotone, so the `monitoring` flag was dropped); the knock-out-time statistic `ko_time` — the first fixing at which the accrued variance exceeds `vol_ko² N/A`, i.e. the day the knock-out becomes certain — is kept for risk and hedging. Reference terms: 100% or 95% strike, 12m, vol_ko = 30% on SPX. `decompose()`: vanilla put minus the vol-knock-in put; reprices path by path. Reports price, ratio to the vanilla put (the "VKO discount") and P(KO).

**Measured outputs (M4c review, owner decision):** the sign of the LSV-versus-LV difference is not a model property — spreading the realised variance around the local-vol level is sign-indefinite for `E[(K − S_T)⁺ 1{RV < H²}]` — so the headline runner and `analytics.conditional_variance.vko_analysis` report, on one path set per model, (a) the `vol_ko` sweep {20, 25, 30, 35, 40}% of the VKO/vanilla ratio for the 12m 100% put with stderr, and (b) the distribution of the full-life realised vol conditional on `S_T < K` (10/50/90 percentiles and `P(σ_real > vol_ko | ITM)` per barrier). **The LSV-versus-LV direction depends on where the barrier sits relative to the ITM-conditional realised-vol distribution**: below its bulk the LSV's wider spread saves more in-the-money paths (dearer VKO), above it the wider spread knocks more of them out (cheaper VKO). Numbers in §6.4.

Background for the study document: a longstanding FX/equity exotic; Citi sold ≈ $30bn of 95–100% strike, 12m, 30–40 vol-barrier VKO puts from late 2021 on the view of a slow, low-realised-vol correction; 2022 delivered exactly that (SPX −20%, VIX 16–37), holders made 7–10× premium, and dealer hedging of "spot down, vol down" exposure is credited with flattening 2022 skew. The product is a pure bet on the realised spot–vol relationship versus the one the surface implies; in the 2F LSV that is governed by the SSR and ν, so the VKO discount ties directly to §7.14/§15 smile-dynamics analytics, and 2022 H2 is the natural backtest window.

Tests: vol_ko → ∞ recovers the vanilla put, vol_ko → 0 gives zero, monotone in vol_ko and ≤ vanilla; decompose reprices; LSV discount larger than LV discount at ω = 2, 3 (sign only until archive tests exist); "running" ≤ "maturity" path by path.

Headline table additions: fair strikes (B = 100% up/down var, B = 110% KO var, 1y, daily fixings) and the 12m 100% VKO put at vol_ko = 30% (price, discount, P(KO)), for ω = 0/1/2/3 in 1F and the Table 8.2 2F set.

### 6.5 Barrier machinery (`products/barrier.py`, M6 Part 1)
Monitoring: `"discrete"` on the fixing schedule (daily by default) or `"continuous"`. Continuous monitoring uses the Brownian-bridge crossing probability between recorded steps with the local diffusion coefficient `L(t,S) sqrt(V)` frozen per step, i.e. per step `p_i = exp(−2 (b − x_i)(b − x_{i+1}) / (σ_i² Δt))` in log-spot, and the survival indicator is the product of `(1 − p_i)` sampled or, for pricing, used as a weight (both reported; the weighted form has lower variance). Requires `record_all_steps`. Barrier shift: `barrier_shift` in % of barrier, applied to the monitored level only, the desk convention for pricing digitals conservatively; default 0, reported in the term-sheet repr. Products: `KnockOutOption` / `KnockInOption` (call/put, up/down, in/out, European-at-maturity or American/continuous), `Digital` (cash-or-nothing, optional call-spread smoothing width for risk), `OneTouch` / `NoTouch`. Tests: in-out parity (KI + KO = vanilla, path by path); discrete → continuous convergence as the monitoring frequency increases, matching the bridge form; BS closed forms (Reiner–Rubinstein) within 3 stderr for continuous barriers; KO var with the barrier machinery equals the M4c close-to-close KO var when `monitoring="discrete"` on daily fixings.

### 6.6 Autocall and Phoenix (`products/autocall.py`, `analytics/autocall.py`, M6 Part 2)
Term sheet (levels in % of S₀; observation dates `T_1..T_N`, maturity `T_N`): at `T_i`, if `S_{T_i} ≥ AC_i` (autocall barrier, default 100%), redeem 100% + coupon `c_i` (`c_i = i × c` for a growing coupon, or a fixed schedule) at `T_i` and terminate. Phoenix variant: at each `T_i` a coupon `c` is paid if `S_{T_i} ≥ CB` (coupon barrier, default 70–80%), with memory (unpaid coupons paid when a later coupon is paid) or without; autocall as above. At `T_N` if not autocalled (market-standard knock-in reading, owner decision at the M6 review): the final coupon is paid if `S_{T_N} ≥ CB` (Phoenix) or `≥ AC_N`, and the capital is redeemed at 100% unless the KI event occurred, in which case at `100% − (K − S_{T_N})⁺ / K` with `K = 100%` (the put leg, geared 1:1) — the coupon decision and the knock-in redemption are independent (`final_redemption="knock_in"`; the `"coupon_barrier"` reading, capital loss only below the coupon barrier, remains an option). Headline Phoenix convention: coupon 6% per annual observation with memory, CB 70%, American KI 60% daily; the term-sheet repr states the coupon, the memory and the reading. KI condition: `"european"` (`S_{T_N} < KI`) or `"american"` (min over the monitoring schedule < KI, discrete daily or continuous per §6.5); KI default 60–70%. Optional: guaranteed coupons, step-down autocall barriers, a first non-call period, redemption at par plus coupon on the final date. Payoff `= Σ_i D(T_i) [AC event at i] (1 + c_i)` + coupon legs + final leg, on the path's first autocall date. `decompose()`: the autocall event leg as a strip of digitals conditional on survival, the coupon leg and the KI put leg; European KI: `(K − S_T)⁺ 1{S_T ≤ B} = (B − S_T)⁺ + (K − B) 1{S_T ≤ B}` — put(B) plus a digital put of size `(K − B)` at `B`, never "a put struck at the barrier"; the American KI has no static decomposition and is reported as put leg vs its European counterpart. Reprices path by path. Analytics: expected life; probability of autocall at each date; probability of KI; price attribution by leg; the LSV-minus-LV table versus model parameters for the product and each leg; forward-skew exposure (price change under a forward-skew bump at each observation date). Tests: `decompose()` reprices; `AC → ∞` and `KI → 0` reduce the product to a zero-coupon bond plus the coupon legs; American KI ≥ European KI put leg; price monotone decreasing in the KI level and in vol (parallel bump); the KI put under BS matches the barrier closed form; the interview-thread decomposition identity holds exactly under BS.

### 6.7 Risk for digital structures (extends §7.10, M6 Part 3)
Autocall delta / gamma / vega profiles versus spot at each observation date (state held, spot shifted), showing the digital at the autocall barrier; the call-spread smoothing width and the barrier shift as report parameters; vega-T waves and `skew_T` for the whole structure and per leg (autocalls are short vol, short skew, long forward — signs verified in a test); expected-life sensitivity to spot and vol; likelihood-ratio Greeks (§7.11) as the cross-check on the digitals.

### 6.8 M6 implementation notes (Parts 1, 2, 4 — 2026-09-13, pending owner review)

Built by three parallel implementers with a three-lens adversarial review and a fix pass each (barrier machinery, autocall / Phoenix, 1F PDE); integrated by hand (exports, `Portfolio.requires_all_steps`, the shared ageing convention, the `sum_sq` consistency of the M5 realised-variance exposure). Conventions and deviations recorded for the review:

**Barrier machinery (`products/barrier.py`, `market/barrier_bs.py`, `tests/test_barrier.py`, 16 fast tests).**
- Discrete monitoring requires `strict` per trade (`strict=False`: touching knocks, `S ≤ B_eff`; `strict=True`: the M4c knock-out variance-swap convention); no library default (the §6.1 "indicator is never defaulted" rule). Mode-irrelevant arguments raise (`survival` / `seed` under discrete monitoring).
- Continuous monitoring: Brownian-bridge weights with `σ_i²` the recorded instantaneous variance at the *start* of the step (tested against a hand transcription on the Bergomi kernel to 4e-16); `survival="weight"` (default for risk) or `"sampled"` (one uniform per path, the exact joint law of the crossing time; the uniforms are keyed on the path set, so the sampled form has no common random numbers across bumps — documented in the repr); the payoff detects a coarser-than-step recording from the kernel's `sum_sq` accumulator and raises naming `requires_all_steps` (any code that rewrites `log_spot` must rewrite `sum_sq` by the same homothety — the M5 realised-variance exposure now does). Evaluated in 64-step blocks (peak temporaries 0.7 of one recorded array).
- `barrier_shift` is a signed fraction, `B_eff = B (1 + shift)` on the monitored level only (repr shows both; no direction convention built in). Rebates: `rebate_timing` required whenever `rebate ≠ 0` (`"hit"` / `"maturity"`; knock-ins accept `"maturity"` only); a continuous "hit" rebate is discounted from the midpoint of the crossing step (removes the `r R dt/2` first-order bias; residual second order). For `monitoring="continuous"` the fixing times are the monitoring window `[t₀, t₁]` (partial-time barriers for free); the discrete default is `daily_schedule(T)` from 0 (inception breach knocks); `fixing_times=[T]` is the European-at-maturity barrier.
- `Digital(strike, T, cp, discount, payout, notional, smoothing)`: `smoothing = w > 0` is the unit-height call spread on `K ∓ w/2` (exact equality with the vanilla spread path by path; the exact digital at `w = 0`; the existing `DigitalOption` is untouched). `KnockInOption.decompose()` = vanilla − knock-out (+ cash flow for a rebate), exact path by path in every mode.
- Closed forms (`bs_barrier_price`, `bs_hit_probability`, `bs_hit_discount`, `bs_one_touch_price`, `bs_no_touch_price`, `bs_digital_price`): Reiner–Rubinstein (1991), reproduce all 36 entries of Haug's Table 4-13 to table rounding, the image identities to 1.7e-14, the killed-density quadrature to 1.6e-14; `bs_hit_discount` raises when `μ² + 2r/σ² < 0`.
- Measured (BS σ = 20%, 1y, dt = 1/250, 40k paths): eight continuous products within 3 stderr of the closed forms (|z| ≤ 1.0); discrete → continuous convergence 12 / 52 / 252 / 1008 observations per year on one path set, with the Broadie–Glasserman–Kou barrier shift `H e^{∓0.5826 σ √Δt}` reproducing the discrete prices under CRN within 0.3–1.3 stderr from 52 per year on (−1.6 stderr at 12 per year, the size of the dropped next-order term); weighted / sampled stderr ratio 0.96 at daily, 0.84 at monthly monitoring; the knock-out variance swap of §6.1 reproduced path by path (`first_hit_index`, both strictness conventions). On the cached 1F LSV: down-and-out call K = S, H = 0.9 S continuous 6.116 ± 0.050 (weight) / 6.112 ± 0.051 (sampled), daily discrete 6.548 ± 0.051.
- Ageing (integrator decision, applied to barrier and autocall alike; `Product.aged` docstring): monitoring dates inside `(0, dt]` were observed at the held spot and drop out (`t = 0` stays), contractual fixings and maturities inside the window still raise.

**Autocall and Phoenix (`products/autocall.py`, `analytics/autocall.py`, `tests/test_autocall.py`, 8 fast tests).**
- `Autocall(observation_times, discount, *, spot_reference, autocall_barriers, coupons, coupon_barrier, memory, ki_level ∈ (0, 1], ki_type, ki_monitoring, ki_fixing_times, guaranteed_coupons, non_call_periods, final_redemption, notional)` with `Phoenix(...)` convenience; levels relative to an explicit `spot_reference`; the KI put geared 1:1 on `S₀` with `K = 100%` (owner's clarification). Coupons: plain autocall `c_i = i × c` paid with the redemption; Phoenix / guaranteed coupons in a separate coupon leg (memory requires a coupon barrier). Knock-in as an event of the life: `ki_breach` (level breached at a monitoring date up to the termination date) and `ki_hit = ki_breach × 1{no autocall}` drives the put leg and `P(KI)`; `P(breach)` reported separately. Continuous American KI through `continuous_survival_weight` (weighted form, seed 0). `decompose()` = conditional digitals (autocall events), bond leg, coupon leg, KI put leg (`KIPutLeg.european_counterpart()`, `unconditional_components()` = put(B) and `(K − B)` digital put at B), exact path by path.
- **Owner decision needed — final-date Phoenix reading** (`final_redemption`): `"knock_in"` (default, market standard: the coupon decision at `T_N` and the knock-in redemption are independent) vs `"coupon_barrier"` (the literal §6.6 sentence: par plus coupon whenever `S_{T_N} ≥ CB`, the put loss only below the coupon barrier). Under BS the headline Phoenix prices 0.99211 ± 0.00075 vs 0.99692 ± 0.00075 (put leg −0.0514 vs −0.0466); `P(KI)` and the expected life are unchanged.
- Analytics: `autocall_report` (price, expected life, autocall probability per date, `P(KI)`, `P(breach)`, leg attribution, all with stderr on one path set), `lsv_minus_lv_table(models, sim, products, reference)` (one row per model × product; the models run on their own grids, so the differences are not paired), `forward_skew_exposure(product, model_factory, sim)` = the sensitivity to the §7.6 skew tent at each observation date (per vol point of 90/110 skew, halving on the arbitrage checks).
- Measured (BS 20% on flat(100, 2%, 1%), 40k paths): 3y annual autocall (AC 100%, c = 6% growing, European KI 60%) price 0.9818 ± 0.0007, expected life 1.918 ± 0.002 y, `P(KI)` 0.073 ± 0.001, `P(autocall at T₁)` 0.4806 ± 0.0007 vs `N(d₂)` = 0.4801; European KI put leg −0.03706 ± 0.00060 vs put(B) + (K − B) digital = −0.03756 (0.8 σ); continuous American KI put (never-autocalling) −0.05845 ± 0.00065 vs Reiner–Rubinstein −0.05912 (z = 1.0), ordering continuous ≤ daily ≤ European path by path; 3y Phoenix (CB 70%, memory, American KI 60% daily) 0.9921 ± 0.0008, life 1.921 y, `P(KI)` 0.135 ± 0.002. Forward-skew exposure under LV: the KI put leg is short skew at 3y (−0.0109 ± 0.0003 per vol point vs −0.0113 static), the digital legs long skew at the early dates — the "autocalls are short skew" sign of §6.7 depends on the bump (fixed-ATM rotation vs put-wing steepening); Part 3 reports it per leg.

**PDE cross-check (`pde/lsv1f.py`, `tests/test_pde_lsv1f.py`, 9 fast + 1 slow tests).**
- `LSV1FPDE(model, *, n_x=400, n_X=121, x_width_sd=4, X_width_sd=4, schedule, scheme ∈ {"hv", "cs"}, theta)` for `BlackScholes`, `LocalVol` (1-D, `X` dropped) and `LSV` with `θ = 0` (ω = 2ν and `ξ_t^t = ξ₀^t e^{ω X_t − ½ω² Var X_t}` taken from the kernel's own conventions); Hundsdorfer–Verwer (θ = ½ + √3/6) or Craig–Sneyd, two Rannacher half-steps, sinh grid clustered at the strike / barrier (a node exactly at `ln B`), nodal payoff with local kink averaging (a global cell-averaging projection was tried and biased put–call parity by 5e-3; replaced), numba Thomas solver for the per-column tridiagonals, dt from the `StepSchedule` with the leverage slices as knots. `n_X` must be odd so `X = 0` is a node (**SPEC §9.1 "400 × 120" → 400 × 121**, owner to accept or amend). `rebate_at` required whenever `rebate > 0`.
- Measured: BS 1y ATM 200 points / dt 1/100: call −3.7e-5 rel, digital 1e-6; observed orders dx 1.99, dt 1.99 (pure Bergomi 2-D: dx 1.99, dX 1.97, dt 1.8); Reiner–Rubinstein down-and-out −2.7e-5 rel; European KI put identity −6e-5; pure 1F Bergomi 3m ATM vs MC z = +0.1 / −0.5 (call / digital), the `x₀ ≠ 0` read-out exact. Reference 1F LSV 1y (slow test, cached model, MC 2·10⁵ paths seed 7): call 8.4778 vs 8.4882 ± 0.0155 (z −0.7), put z +0.5, digital 0.60278 vs 0.60306 ± 0.00078 (a five-run pooled MC of 1.4·10⁶ paths gives 0.60287 ± 0.00029, z −0.3 — the digital's criterion is 3 stderr on the single seed, a deviation from §9.1's 2 stderr), European KI put 6.8301 vs 6.8118 ± 0.0290 (z +0.6), continuous KO call (bridge form) 6.1228 vs 6.1489 ± 0.0159 (z −1.6; three-seed pooled gap −0.014, 1.5 σ — open). Truncation at ±4 ATM sd: 2.4e-4 relative on the 1y vanillas at ω = 3 (3e-5 at ±6 sd) — **default width for factor models open**. The PDE exposes the Dupire grid's own residual against the SSVI surface: 3m −10% put −0.05 vp, 1y −10% −0.012 vp, ATM ≤ 0.007 vp, invariant under PDE refinement (the local-vol grid, not the PDE, is the limit). Cost 400 × 121 × 641 steps ≈ 5–7 s per price.

**Risk for digital structures (`risk/digital_risk.py`, `tests/test_risk_digital.py`, 7 fast tests — M6 Part 3; conventions adopted, owner to review with the M6 report).**
- `autocall_spot_profiles(engine, note, state, *, dates, shifts, size, smoothing, barrier_shift, barrier_shift_scope)`: the note one business day before each observation date with the path held flat at today's spot (date 1 through `aged`, later dates as an explicit residual note with the coupon schedule re-indexed, the non-call periods reduced and the knock-in history represented: a held spot below an American KI level makes it the knocked-in note, a memory Phoenix held below the coupon barrier carries the missed coupons — path-by-path identities tested), then the model-regime spot profile on a grid straddling `AC_i` (±10% in 1% steps, 0.25% inside ±2%). `smoothing = w` replaces the profile date's autocall event by the exact call spread of width `w · S_ref` (`SmoothedAutocall`: `θ(S) = clip((S − B + w/2)/w, 0, 1)` on the path-by-path split of the note into its called and continuing halves; later digitals stay exact); `barrier_shift` moves the monitored levels by the signed fraction — the profile date's barrier and the KI level (`scope="date"`, default) or every autocall barrier (`"all"`), never the put strike or the coupon barrier; the ±`size` log-stencil itself averages the one-day digital over ≈ 2·size·B (10% low at 1%, 0.7% at 0.25%; recorded in `attrs`).
- `ki_put_barrier_risk` (central bump of `ki_level`, per unit spot and per 1% of the barrier, with the vanilla / digital components and `∂P(KI)/∂B`) and `ki_barrier_profile` (±5% in 0.5% steps, 21 rows) — the §7.10 spot-barrier KI put items; `structure_vega_skew` (vega-T waves / projections and `skew_T` for the note and every leg; legs add up to the note to 1e-9 per entry); `expected_life_sensitivity` (`∂E[life]/∂ln S` model regime, `∂E[life]/∂σ` sticky leverage, through the `life` statistic leg); `lr_cross_check(product, model, sim, *, engine, state, first_step, spot_size, vol_size)` (likelihood-ratio delta / vega of the note and its digital legs against the bump estimates, z-scores, several stencils in one pass).
- Measured (BS 20%, flat(100, 2%, 1%), 40k paths): one-day digital delta at the barrier 0.01725 ± 0.00010 vs the analytic 1% stencil 0.01718 (exact derivative 0.01900); smoothed w = 2%: 0.01591 ± 0.00007 vs 0.01582; KI put `∂price/∂B` −0.002636 ± 0.000203 per unit spot vs the closed form −0.002759 (z 0.6); `E[life]` 1.918 ± 0.002 y, `∂E[life]/∂ln S` −3.76 ± 0.08 y. **Signs under local vol on the reference surface (3y annual autocall, 40k paths):** parallel vega −0.0053 ± 0.0002 per vp (short vol, 31 σ); skew −0.0031 ± 0.0003 at 3y and −0.0045 globally under the fixed-ATM rotation (short skew; the 1y date is flat because the 1y digital's long skew, +0.020, offsets the later legs); delta +0.0070 ± 0.0002 (long forward); repo delta −5.9e-5 ± 0.7e-5 per bp; **rho is negative**, −9.4e-5 ± 0.4e-5 per bp (22 σ): the funded par note's discounting (bond leg −2.1e-4, duration ≈ E[life]) outweighs the equity legs' +1.2e-4 ± 0.15e-4 — the "long forward" of §6.7 holds for the equity legs and the delta, not for the note's rho. **Likelihood ratio vs bump on the 1–3y digitals (BS, 10⁵ paths, first_step 1/52):** all |z| ≤ 2.7, but the LR standard error is *larger* than the 1% bump's on every digital leg (delta 1.3–2.6×, vega 2–11×) and beats it only below a ≈ 0.5% stencil (0.2%: 0.56–1.11× for delta); the report should carry a fine-stencil column rather than rely on the LR as the precise estimator at daily-step first-week aggregation.

**M6 headline rows (`studies/m6.py`, `scripts/m6_headline.py`, `tests/test_m6_headline.py`, 14 fast tests).** `headline_products(discount, spot)`: the 3y annual autocall (AC 100%, coupon 6% p.a. growing, European KI 60%) and the 3y Phoenix (CB 70%, memory, American KI 60% daily, **coupon 6% per period — the owner gave no rate; the Part 2 test bed used 5%, the two are labelled**); `run_m6_headline(models, sim, products, reference)` prices both notes on one path set per model (the union grid, the Phoenix's daily grid) with price, expected life, `P(KI)`, `P(breach)`, `P(no autocall)`, the per-date autocall probabilities and every leg, all with stderr; LSV − LV differences are root-sum-square (models on their own grids, not paired); the CSV keeps prices as fractions of notional, the markdown renders %; `headline.meta.json` records seed, particle count, paths, code version. Verified: legs add up to the price and `leg:autocall_i = DF(T_i)(1 + c_i) P(AC at i)` exactly; `P(AC at T₁) = N(d₂)` within 3 stderr; 25% vol lowers the price and raises `P(KI)` at 20 σ. **Production run (8·10⁵ particles, tag `m6`, 4·10⁵ paths, seed 2024, `outputs/m6/headline`), prices in % of notional, one path set per model shared by the two notes, differences to LV root-sum-square (not paired):**

| model | autocall 3y price | E[life] (y) | P(KI) | Phoenix 3y price | E[life] | P(KI) | P(breach) |
|---|---|---|---|---|---|---|---|
| LV (ω = 0) | 95.82 ± 0.03 | 1.655 | 0.1111 ± 0.0005 | 96.43 ± 0.03 | 1.655 | 0.1811 ± 0.0005 | 0.2105 |
| 1F ω = 1 | 96.17 ± 0.03 | 1.658 | 0.1051 | 96.75 ± 0.03 | 1.658 | 0.1817 | 0.2052 |
| 1F ω = 2 | 96.44 ± 0.03 | 1.664 | 0.0998 | 97.09 ± 0.03 | 1.664 | 0.1788 | 0.1977 |
| 1F ω = 3 | 96.61 ± 0.03 | 1.670 | 0.0962 | 97.40 ± 0.03 | 1.670 | 0.1736 | 0.1892 |
| 2F Table 8.2 | 96.53 ± 0.03 | 1.664 | 0.0980 | 97.20 ± 0.03 | 1.664 | 0.1776 | 0.1935 |

Reading: on the same surface the LSV notes are worth 0.35 / 0.63 / 0.79 % of notional more than under local vol for ω = 1 / 2 / 3 (0.71 for the 2F set) on the autocall and 0.32 / 0.66 / 0.97 (0.77) on the Phoenix — the knock-in put leg shrinks (autocall put leg −6.71 → −5.62 % from LV to ω = 3; Phoenix American put −8.24 → −7.38) faster than the coupon and autocall legs change, `P(KI)` falls from 0.111 to 0.096 (autocall) while the European-observation `P(AC at 1y)` is model-independent (0.616 ± 0.001 everywhere: a 1y digital reprices the surface) and the later autocall probabilities fall with ω (the forward ATM skew flattens); the expected life is 1.66–1.67 y in every model. The Phoenix's American daily KI is hit almost twice as often as the European one (`P(breach)` 0.19–0.21 vs 0.10–0.11) and the American put leg is 1.5–1.8 % of notional dearer than its European counterpart. The 2F set sits between ω = 2 and ω = 3, as in the M4 headline. A `tests/test_m6_regression.py` baseline can pin these rows (`regression_keys`) once the owner confirms the two term-sheet conventions.

Conventions adopted (owner: "proceed with your defaults … I'll review them with the M6 report"): (i) Phoenix `final_redemption="knock_in"` (coupon decision and knock-in redemption independent); (ii) `strict` required per trade for discrete barriers (no library default); (iii) sampled survival keyed on the path set (no CRN across bumps; the weighted form is the risk default); (iv) continuous hit rebate discounted from the step midpoint; (v) PDE grid 400 × 121 with the ±4 sd width; (vi) 3-stderr digital criterion in the PDE slow test on a single seed (pooled evidence at 2 stderr).

### 6.3 M4 notes (implemented)
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



### 6.4 M4c notes (implemented)
`products/conditional_variance.py`: `ConditionalVarianceSwap(fixing_times, barrier, side, indicator, convention, strike_vol, discount, *, strict=True, daily_cap=None, annualisation=252, notional=1)` — the indicator is required; `UpVar(...)` / `DownVar(...)` supply the desk defaults (`"prev"` / `"curr"`, `"both"` allowed, the other indicator rejected); realised quantities on the fixing dates, `A/N` annualisation; the payoff is zero when `D = 0` (both terms vanish). `ConvexitySpread(upvar, varswap, notional_ratio)` requires the two legs to share the fixing schedule and the variance swap to be realised on the fixings. `KnockOutVarianceSwap(fixing_times, barrier, strike_vol, discount, *, direction="up", strict=True, monitoring="close", variant="b", ...)`: the monitoring includes `S_0` (a breach at inception gives `τ = 0`, zero payoff); `monitoring != "close"` and variants (a)/(c) raise `NotImplementedError`. Every product exposes per-path `statistics()` (`accrued` already annualised, `count = D/N` or `τ/N`, and `ko`, `tau` for the knock-out swap) through `leg(name)` products priced on the same path set; `analytics/conditional_variance.py` turns them into `fair_strike()` (ratio of means with the delta-method standard error, `count` / `p_ko` / `e_tau` as extras), `strike_differential()` (joint delta method on one path set), `lsv_minus_lv()` (same seed, independent path sets) and `vko_report()` (price, ratio to the vanilla put, `P(KO)`).

`products/vko.py`: `VolKnockOutPut(strike, maturity, vol_ko, fixing_times, discount, *, daily_cap=None, annualisation=252, notional=1, knock_in=False)`; `decompose()` = vanilla put − the knock-in put, exact path by path; statistics `alive`, `ko`, `ko_time`, `rv`, `put`, `itm`. The "running" flag of the first implementation was dropped at the M4c review (it had the same terminal payoff as the maturity check; only the knock-out time differed); `ko_time` stays. `analytics.conditional_variance.vko_analysis(vko, model, sim, barriers)` returns the barrier sweep of the ratio to the vanilla put (delta-method stderr), the prices, `P(KO)`, `P(σ_real > h | ITM)` and the 10/50/90 percentiles of `σ_real` given `S_T < K`, all from the per-path `put`, `rv`, `itm` statistics of one simulation; `vko_report` gives the single-barrier summary.

Headline table extension (M4c; production convention: 8·10⁵ particles per calibration, single seed, 400k pricing paths, seed 2024; the legacy columns moved by at most 1.2σ against the 2·10⁵ M4b baseline; all numbers below are the regression baselines in `tests/test_m4_regression.py`):

| model | 1y→2y ATMF fwd vol | fwd VS vol | fwd vol-swap vol | cliquet 1y % | cliquet 2y % |
|---|---|---|---|---|---|
| LV (ω=0) | 21.46 ± 0.04 | 25.25 ± 0.02 | 22.54 ± 0.01 | 1.199 ± 0.004 | 0.749 ± 0.004 |
| 1F ω=1 | 20.82 ± 0.04 | 25.25 ± 0.02 | 22.32 ± 0.01 | 1.321 ± 0.004 | 0.954 ± 0.005 |
| 1F ω=2 | 19.70 ± 0.04 | 25.18 ± 0.03 | 21.39 ± 0.01 | 1.629 ± 0.005 | 1.464 ± 0.006 |
| 1F ω=3 | 18.45 ± 0.04 | 25.12 ± 0.03 | 20.07 ± 0.02 | 1.971 ± 0.005 | 2.033 ± 0.006 |
| 2F Table 8.2 | 19.22 ± 0.03 | 25.14 ± 0.03 | 20.98 ± 0.01 | 1.750 ± 0.005 | 1.812 ± 0.006 |

| model | up-var B=100% (fair vol) | down-var B=100% | KO var B=110% | P(KO) | VKO 12m 100% put @30% (% notional) | P(KO) |
|---|---|---|---|---|---|---|
| LV (ω=0) | 15.04 ± 0.00 | 34.92 ± 0.03 | 29.07 ± 0.03 | 0.657 ± 0.001 | 2.437 ± 0.009 | 0.175 ± 0.001 |
| 1F ω=1 | 15.05 ± 0.00 | 34.99 ± 0.03 | 28.65 ± 0.03 | 0.645 ± 0.001 | 2.330 ± 0.009 | 0.178 ± 0.001 |
| 1F ω=2 | 15.06 ± 0.01 | 34.88 ± 0.03 | 27.66 ± 0.03 | 0.623 ± 0.001 | 2.431 ± 0.009 | 0.184 ± 0.001 |
| 1F ω=3 | 15.07 ± 0.01 | 34.75 ± 0.04 | 26.45 ± 0.03 | 0.594 ± 0.001 | 2.570 ± 0.010 | 0.188 ± 0.001 |
| 2F Table 8.2 | 15.09 ± 0.01 | 34.91 ± 0.03 | 27.42 ± 0.02 | 0.617 ± 0.001 | 2.098 ± 0.008 | 0.182 ± 0.001 |

| model | VKO/vanilla @20% | VKO/vanilla @25% | VKO/vanilla @30% | VKO/vanilla @35% | VKO/vanilla @40% | σ_real given ITM: p10 / p50 / p90 | P(ITM) | P(σ_real > 20% given ITM) | P(σ_real > 25% given ITM) | P(σ_real > 30% given ITM) | P(σ_real > 35% given ITM) | P(σ_real > 40% given ITM) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| LV (ω=0) | 0.055 ± 0.000 | 0.176 ± 0.001 | 0.325 ± 0.001 | 0.476 ± 0.002 | 0.614 ± 0.002 | 18.1 / 27.5 / 45.7 | 0.384 | 0.820 | 0.599 | 0.414 | 0.274 | 0.174 |
| 1F ω=1 | 0.053 ± 0.000 | 0.164 ± 0.001 | 0.310 ± 0.001 | 0.466 ± 0.002 | 0.608 ± 0.002 | 18.1 / 27.9 / 45.5 | 0.384 | 0.829 | 0.615 | 0.425 | 0.278 | 0.174 |
| 1F ω=2 | 0.067 ± 0.001 | 0.178 ± 0.001 | 0.324 ± 0.001 | 0.474 ± 0.002 | 0.612 ± 0.002 | 17.0 / 27.5 / 46.0 | 0.384 | 0.796 | 0.596 | 0.414 | 0.276 | 0.177 |
| 1F ω=3 | 0.088 ± 0.001 | 0.203 ± 0.001 | 0.343 ± 0.001 | 0.485 ± 0.002 | 0.614 ± 0.002 | 15.5 / 26.8 / 46.9 | 0.384 | 0.749 | 0.562 | 0.399 | 0.274 | 0.183 |
| 2F Table 8.2 | 0.036 ± 0.000 | 0.132 ± 0.001 | 0.280 ± 0.001 | 0.448 ± 0.002 | 0.605 ± 0.002 | 19.0 / 28.5 / 45.0 | 0.382 | 0.865 | 0.654 | 0.442 | 0.280 | 0.170 |

Reading: the up-var (`"prev"` indicator) is Gyöngy-flat across models (15.04–15.09%, the residual spread is the within-day structure of the daily return); the down-var (`"curr"` indicator conditions on the end of the return) moves by 0.2 vp; the knock-out swap is the model-dependent one — LSV − LV fair vol −0.4 / −1.4 / −2.6 vp for ω = 1 / 2 / 3 and −1.7 vp for the 2F set, with `P(KO)` falling from 0.66 to 0.59 — while `K_KO > K_var` (25.2%) holds throughout. **VKO:** at 30% the ratio to the vanilla put is 0.325 under LV and 0.310 / 0.324 / 0.343 at ω = 1 / 2 / 3, 0.280 for the 2F set — no uniform LSV-versus-LV sign (the owner withdrew the sign claim at the M4c review). The sweep shows the mechanism: the ITM-conditional realised vol has median 27–28.5% and a 10–90 band of 15.5–19% to 45–47% in every model, so a 30% barrier sits at its centre (`P(σ_real > 30% | ITM)` 0.40–0.44); the 1F LSV widens the band as ω grows (p10 18.1 → 15.5%, p90 45.7 → 46.9%), which saves in-the-money paths below the barrier — the LSV ratio exceeds LV's at 20–25% for ω ≥ 2 (0.088 vs 0.055 at 20% for ω = 3) and the ordering fades by 35–40% where all ratios converge to 0.61 — while the 2F set narrows it (p10 19.0%, p90 45.0%) and is cheaper at every barrier up to 35%.

Tests (`tests/test_conditional_variance.py`): hand values for every indicator / convention / strictness / daily cap; complementarity path by path (up `"prev"` strict + down `"prev"` non-strict, corridor = the variance swap; `"both"` variants = variance swap minus crossing-day variance); knock-out swap `τ` logic (knock-out day accrues, inception breach), `B → ∞` = variance swap path by path; VKO decomposition, `vol_ko → ∞` / `→ 0` limits, monotonicity in the barrier path by path, running = maturity; Black–Scholes fair strikes = σ for the conditional and knock-out swaps (iid returns, optional stopping) and `σ² E[D]/N` for the corridor, zero convexity-spread differential; local vol on the reference surface: `K_up < K_var < K_down` for B ∈ {90, 100, 110}% and `K_KO > K_var` for B ∈ {105, 110, 120}% at 3 stderr (a property of negative skew and a non-inverted term structure, not a theorem), VKO monotone and below the vanilla; slow: Gyöngy invariance of the single-close corridor swaps under LV and the LSV for ω ∈ {1, 2, 3} within 2 stderr and the LV identity `(A/N) Σ E[σ_loc²(t_{i−1}, S_{i−1}) δ I_i]`, the orderings under the LSV,; the VKO barrier sweep and ITM-conditional realised vol under BS (monotone, saturating at 1, centred on σ).

---

## 7. Risk and analytics (M5 — detailed)

### 7.1 General machinery (`risk/engine.py`)
- `BumpSpec(name, apply, size, scheme="central"|"forward")` and a `RiskEngine` pricing base and bumped states under common random numbers, returning `Sensitivity(value, stderr)` with the stderr of the *difference*, path by path.
- Bumps that change the surface or a model parameter recalibrate through `LeverageCache.get_or_calibrate` (each is a cache entry; the second run of a ladder is free); bumps of S0 or the factor state do not.
- Surface perturbations are an additive layer on `ImpliedSurface` (δσ(k,T) in implied vol), with the no-arbitrage checks re-run; on failure halve the bump, retry, report.
- Default bump sizes: delta/gamma 1% of spot in log space (central three-point); vega 1 vol point; forward-variance buckets +1 vp of the bucket's forward VS vol; model parameters 5% relative (ν, k1, k2), 0.05 absolute (θ, correlations) with the PSD check re-run. Configurable.

### 7.2 Delta and gamma regimes (`risk/greeks.py`)
Smile in `k = ln(K/S)`, `Δ = ln(S0_new/S0_old)`, `s_T` = ATM skew dσ/dk at maturity T from the surface.
- `"model"`: bump S0, L and factors fixed — default, the hedger's delta.
- `"sticky_strike"`: `σ_new(k,T) = σ_old(k + Δ, T)`.
- `"sticky_moneyness"`: `σ_new(k,T) = σ_old(k, T)`.
- `"sticky_skew"`: `σ_new(k,T) = σ_old(k,T) + s_T Δ` (ATM vol slides along the old smile; shape re-centred with the same skew and curvature).
- `"sticky_local_vol"`: `σ_new(k,T) = σ_old(k,T) + 2 s_T Δ` (Derman's sticky-implied-tree regime, SSR = 2; reference only).
All but "model" rebuild the surface, recalibrate L (cached), reprice under CRN. Report all five deltas and gammas side by side. Tests: coincide in BS; ATM vanilla sticky_strike vs sticky_skew agree to first order (report the curvature term); ATM-vol shift per unit Δ ordered 0 < s_T < 2 s_T; the "model" shift equals `SSR_T · s_T` (§4.4, lands with M7).

### 7.3 Vega, theta (`risk/greeks.py`)
- Parallel vega: +1 vp at every (K,T); variants `"recalibrated"` (default) and `"sticky_leverage"` (L held, ξ₀ moves with the strip); the difference is the leverage vega, a study quantity.
- Theta: t + 1 business day, surface held in (K, absolute expiry), factors at zero, forward rolled; only for products with no fixing in the roll window. Split into carry (rates/divs), pure decay (surface held in time-to-maturity) and roll-down. BS test: θ + ½σ²S²Γ + rate terms = 0.

### 7.4 Vega by maturity — desk "wave" convention (`risk/ladders.py`, `vega_T`)
Pillars: the surface's expiry pillars (default 1m, 2m, 3m, 6m, 9m, 1y, 18m, 2y, 3y). `tent_i(T)` linear 0 → 1 → 0 over (T_{i−1}, T_i, T_{i+1}), flat beyond the ends. Pillar bump i: `σ → σ + 0.01·tent_i(T)`; check the calendar condition on every ramp (`(σ+0.01)² T_i ≤ σ² T_{i+1}`, i.e. `T_{i+1}/T_i ≥ 1.10` at σ = 20%; raise for pillar sets that violate it). Wave j = all pillars with T_i ≤ T_j bumped together; wave n = parallel vega. Report cumulative vega per wave and the projection at T_j = wave_j − wave_{j−1}; projections sum to the parallel vega. Recalibrated (default) and sticky-leverage variants; units % of notional per vol point. Tests: projections equal the single-pillar tent ladder within 2 stderr; wave_n equals parallel vega within 2 stderr. `RiskReport` docstring states that (i) projections equal single-pillar bumps only to first order (recalibration and vega convexity add a cross term), (ii) an expiry between pillars shows vega in both neighbouring projections in proportion to distance.

### 7.5 Forward-variance vega ladder (`risk/ladders.py`, `fwd_var_ladder`)
Buckets monthly to 1y, quarterly to 3y. Bump `ξ₀^T → (1+ε)ξ₀^T` on the bucket; propagated to the surface as `dW(T) = ε ∫_{bucket∩[0,T]} ξ₀` added to `w(k,T)` for every k (parallel shift in total variance per maturity; skew preserved in total-variance terms). ε sized so the bucket's forward VS vol rises 1 vp; units % of notional per vol point of bucket forward VS vol. Variants recalibrated / sticky-leverage. Tests: ladder sums to the parallel forward-variance bump (both variants, 2 stderr); flat and analytic for a plain variance swap. Second-order (diagonal convexity per bucket, three-point) on request.

### 7.6 Skew and curvature risk (`risk/ladders.py`, `skew_T`, `curvature_T`)
Skew bump i: `σ → σ + s·k·tent_i(T)`, s sized so `σ(ln 0.9) − σ(ln 1.1)` at T_i rises 1 vp (rotation around ATM); units per vol point of 90/110 skew. Curvature bump i: `σ → σ + c·k²·tent_i(T)`, c sized so `[σ(ln 0.9)+σ(ln 1.1)]/2 − σ(0)` rises 1 vp; units per vol point of 90/110 butterfly. No-arbitrage checks as §7.1; both variants. Tests: ATM vanilla zero skew vega, 90% put positive; 90/110 risk reversal zero curvature vega, strangle positive; `skew_T` sums to a global rotation within 2 stderr.

### 7.7 Cross-Greeks (`risk/greeks.py`)
Vanna as both `∂vega/∂ln S` and `∂delta/∂σ` (equal only in BS; the LSV difference is informative); volga `∂vega/∂σ`; charm and veta (one-business-day rolls, surface in (K, absolute expiry)); rho (+1 bp rates); repo/dividend delta (+1 bp in q). Cross terms `∂delta/∂ρ_SX1`, `∂delta/∂ρ_SX2`, `∂delta/∂ν` (recalibrated): the hedge ratio's own dependence on the vol-sto parameters.

### 7.8 Spot-shift profiles (`risk/profiles.py`)
S0 grid −30% to +30% in 2.5% steps; price, model delta/gamma/vega per shift under "model" dynamics (other regimes optional, recalibrated per shift, cached); gamma profile as second difference. Cliquet gamma profile versus the accumulated sum at an intermediate date, with the Bachelier call-on-remaining-capped-sum decomposition of the original study as an *approximate* cross-check: the sum of capped monthly returns is not Gaussian, so the formula was never exact under BS (owner's correction at the M5 review); the exact BS reference is the independent-legs simulation, and both the BS and the LSV deviations are reported (§7.15).

### 7.9 Model-parameter sensitivities (`risk/volsto_sens.py`)
Partials to each of (ν, θ, k1, k2, ρ12, ρ_SX1, ρ_SX2) with recalibration; ν also sticky-leverage. Central differences, PSD check on correlation bumps. One table per product.

### 7.10 Product-specific risks (`risk/product_risk.py`)
- Fixing risk: forward-start and cliquet delta/gamma/vega at T1 − 1d and T1 + 1d (state held), reported as the jump; per fixing for cliquets.
- Barrier risk: `∂price/∂B` for KO var and for the spot-barrier knock-in put of the autocall (M6 Part 3; owner's clarification at the M5 review — the M5 implementation covered the VKO knock-in variant, which stays as built); delta and gamma within ±5% of the barrier at 0.5% steps; VKO `∂price/∂H` (vol points of the vol barrier) and `∂P(KO)/∂ln S`.
- Realised-variance exposure: expected dollar-gamma-weighted variance per fixing period along the path; flat and equal to notional for a variance swap.

### 7.11 Precision options (`risk/estimators.py`)
Likelihood-ratio delta and vega for discontinuous payoffs (barriers, KO var, VKO, digitals) with the bump estimate as cross-check (flag > 3 stderr disagreement); conditional Greeks at a future date by regression of payoff and CRN bumps on a polynomial basis in (ln S_t, X¹_t, X²_t) — the hedger's engine; control variate on the difference (BS Greek at market vol for vanilla-like products), with variance reduction reported.

### 7.12 P&L attribution (`risk/attribution.py`)
`explain(product, state_0, state_1)`: sequential CRN revaluation in the order spot (delta, gamma), surface (parallel vega, vega-T waves, skew/curvature), model parameters, factor state; residual = actual minus sum. States carry S0, surface, model params, factor values, date. Tests: pure spot move residual at the gamma-cubed level (report); pure parallel vega move residual within 2 stderr.

### 7.13 RiskReport and budget
One object per product with every sensitivity, stderr, bump specs, cache keys; `to_dataframe()`, `to_excel()`. A full report is ≈ 80–100 recalibrations; print count and wall clock — it is the viewer precompute budget. Owner decision (M6 Part 0 review): the 8·10⁵ budget run is skipped; the 2·10⁵ count with the 4× per-calibration scaling (33 s → 130–140 s) is the M9 estimate. **Measured (2026-09-13, `scripts/m5_budget.py`, 1y ATM call, 1F reference LSV, 2·10⁵ particles and paths, seed 2024, default configuration — 9 pillars × 3 ladders, 20 forward-variance buckets × 2 variants, 5 regimes, 7 parameters, cross-Greeks, theta):** 89 recalibrations (76 cache misses), 134 pricings, 127 sensitivities, 80 min wall clock on a machine shared with other calibration jobs (about 50 min idle: 33–40 s per calibration plus 8 s per pricing at 2·10⁵ paths). At 8·10⁵ particles the calibrations scale 4× (130–140 s each): ≈ 3–3.5 h per product-state for the viewer precompute, dominated by the 76 distinct calibrations, which are shared across products at the same state (the study cliquet's two forward-variance ladders on the same states added 0 recalibrations and 13 min of pricing). Cliquet forward-variance ladders (study structure, 1y, % of notional per vol point of bucket forward VS vol): recalibrated sum −0.142 ± 0.001 (parallel −0.144 ± 0.001), sticky-leverage sum −0.034 ± 0.000 — the recalibrated ladder is front-loaded (−0.028 in the first month falling to −0.007 at 1y) and 4× the sticky one; the buckets beyond the 1y maturity carry nothing. The M5 budget runs found and fixed two report-path defects under the LSV builder (the unfloored ATM-skew shift of the sticky-skew / sticky-local-vol regimes and the `"model"` mode of a perturbed state), §7.15.

### 7.14 Smile-dynamics analytics (`analytics/smile_dynamics.py`, `analytics/var_decomp.py`)
As §4.4 and §15 Part 1: numerical SSR (LSV joint-bump, pure SV), eq. 12.52 decomposition, short-horizon regression estimator; vols of ATMF vols (eq. 12.56); conditional smile after a spot move at horizon t; vol-of-vol term structure; Var(V) decomposition (closed form pure SV, regression estimator LSV).

### 7.15 M5 implementation notes (2026-09-13)

Layout: `risk/engine.py` (`RiskState`, builders `LSVBuilder` / `LVBuilder` / `BSBuilder`, `RiskEngine` with `perturbed_state` (halve-and-retry on the arbitrage checks, achieved size reported), `combination` / `paired` (per-path CRN estimator with the stderr of the difference), `Sensitivity`, `product_key`), `risk/greeks.py` (five delta regimes, vega variants, theta split, cross-Greeks), `risk/ladders.py` (`vega_T`, `fwd_var_ladder`, `fwd_var_convexity`, `skew_T`, `curvature_T`), `risk/profiles.py`, `risk/volsto_sens.py`, `risk/product_risk.py`, `risk/estimators.py`, `risk/attribution.py`, `risk/report.py` (`RiskReport`, `risk_report`); `scripts/m5_budget.py`; tests `tests/test_risk_{greeks,ladders,profiles,product,estimators,attribution,report}.py` (fast on the Black–Scholes bed and local vol; slow: LSV budget report, control variate under LV).

Deviations from the text above and measured findings:
- **Bumps** are `SurfacePerturbation` layers (`parallel`, `tent`, `skew_tent`, `curvature_tent`, `shift_k`, `atm_shift`, `total_variance`, `roll`, `table`, `composite`) plus `RiskState.with_spot / with_params / with_rate_shift / with_x0`; there is no separate `BumpSpec` class — the bump specification lives in the `Sensitivity` (size, scheme, states, extras). Surface bumps are hashed into the leverage-cache key (`CalibrationSpec.perturbation`).
- **Forward-variance ladder sizing (§7.5).** The ξ₀-scaling `ε = [(σ_b+0.01)² − σ_b²]/σ_b²` moves the log-contract strip by a smile-dependent factor, 1.17 on the reference surface (K_var 6.0% against an ATM variance of 4.0%): the first monthly bucket of a 1y variance swap measured 0.04975 ± 0.00056 against the naive 0.0434 (−14%). `bucket_epsilon` now solves ε on the strip (two secant steps) so the bucket's forward VS vol rises exactly 1 vp, and the analytic reference of `varswap_bucket_sensitivity` is the model-independent strip difference `N · DF · [(σ_b+0.01)² − σ_b²] · Δt_b/T`.
- **Skew / curvature profile (§7.6).** `s·k` and `c·k²` break the butterfly / calendar checks at the short pillars; the implemented profile is `κ(k) = k_cap tanh(k/k_cap)` (k_cap = 0.5, linear near the money, saturating in the wings) with the same 90/110 normalisation; the checks still halve the skew bump once from 1y and the curvature bump 2–4 times (achieved sizes reported, sensitivities per requested unit).
- **Parameter boundaries (§7.9).** θ = 0 of the one-factor model admits no central bump: the difference is one-sided towards the admissible side (`scheme` records `forward` / `backward`), halving only when neither side is admissible (|ρ| > 1, PSD).
- **Cliquet gamma profile (§7.8).** The local-linear regression carries the ½h²m″ smoothing bias (+0.1% of notional at h = 0.03 where m″ reaches 3–4 per unit²); the value is corrected with the stencil second difference (the calibration's device). The Bachelier call-on-remaining-capped-sum is *not* exact in Black–Scholes: the Gaussian-sum step is an approximation even with independent legs (the remaining sum is bounded above by the caps with an open left tail). Measured against the exact independent-legs reference `bs_cliquet_value_mc`: the Bachelier value exceeds the true conditional value by up to 0.18% of notional at the 6m fixing of the study cliquet — 20–25% of the value around A ≈ 0 (value ≈ 1% of notional), 4% at the top of the range. The regression reproduces the exact reference within 1.5% of the value.
- **Fixing risk (§7.10).** "State held" legs are built explicitly (before: the same forward start with T1 = 1d; after: the vanilla at k·S₀; cliquets: the remaining structure with zero fixed returns), not through `aged()`, which raises for fixings inside the roll window. Black–Scholes: the forward-start straddle's delta jumps 0 → 0.116 (per unit S₀, notional 100) and gamma 0 → 0.039 with vega unchanged (0.0011 jump) — the vega-to-delta conversion ratio (delta jump / vega before) is 0.15.
- **Barrier risk (§7.10).** The library had no spot-barrier knock-in put at M5; the M5 implementation covered the VKO's knock-in variant (`knock_in=True`) through `∂price/∂H`, which stays as built; the owner clarified at the review that the §7.10 KI put is the autocall's spot-barrier knock-in put, delivered with M6 Part 3. `∂price/∂B` for the knock-out variance swap is central at ±0.5% of the barrier; `∂P(KO)/∂ln S` comes from the `"ko"` statistic leg; the ±5% profile in 0.5% steps is the model-regime spot profile.
- **Realised-variance exposure (§7.10).** Defined pathwise: the bucket's period log-returns are rescaled by 1 ± ε on the simulated paths (later log-spots shifted), the symmetric second difference E[P₊ + P₋ − 2P] is the dollar-gamma-weighted variance of the bucket (first-order spot effect cancelling), normalised by 2ε² E[Σ_b r_i²]/(T − t₀). Exactly N · DF(T) in every bucket for a daily variance swap (identity; tested within 3 stderr, flat to 1%).
- **Likelihood ratio (§7.11).** The exact first-transition score has variance ∝ 1/δ₀ and is useless on the 1/1460 first step; `lr_delta` aggregates the spot normals over the first week (score of the constant-coefficient approximation: exact under Black–Scholes and for τ = δ₀, O(τ) bias under LSV that the bump cross-check bounds); `lr_vega` is the per-step parallel instantaneous-vol score on a record-all-steps grid (frozen coefficients; the comparator is the sticky-leverage bump vega). Digital call (BS, 2·10⁵ paths): LR delta 0.02783 ± 0.00026 vs bump 0.02810 ± 0.00031 (z = −0.67), both within 3.5 stderr of the analytic value; LR vega within 3.5 stderr.
- **Conditional Greeks (§7.11).** The per-path CRN differences are normalised by S_t/S₀ (tangent-process form: exact for spot-homogeneous dynamics, the leverage-held-in-spot approximation for the LSV); the raw S₀ targets are kept for the t = 0 Greeks. A quadratic basis cannot follow the vanilla delta's sigmoid (RMSE 0.05 against the Black–Scholes delta at 6m); the default is cubic (RMSE 0.02–0.03; conditional value within 4% of the analytic price).
- **Control variate (§7.11).** The Black–Scholes control at the market implied vol of (K, T) runs on the engine's grid with the same normals (column 0 of the draws is the spot normal in every kernel); it needs the LV/LSV engine (under `BSBuilder` the control is the estimator). Measured under local vol on the reference surface (40k paths, 1y options): the control reduces the **vega** variance by 10× (ATM call, β = 0.76) and 2.9× (90% put, β = 1.3), but leaves **delta** and **gamma** essentially uncontrolled — VR 1.1–1.3 for the delta under the model, sticky-moneyness and sticky-strike moves alike, with β < 0 for the ATM call (the local vol held in spot makes the up path run at a lower vol than the down path, so the pathwise LV difference anti-correlates with the Black–Scholes one on the deep in-the-money paths) and β ≈ 0.04 for the gamma. The control is a vega device under smile dynamics.
- **Attribution (§7.12).** The spot step is the sticky-moneyness move of the state's own surface configuration (SSVI in k); the time step is explained by decay + carry (the roll-down sits in the surface step since `state_1` carries the observed end surface); `detail="parallel"` (parallel vega × mean ATM change) or `"ladders"` (tents × per-pillar ATM changes, skew and curvature ladders × 90/110 changes). Black–Scholes 1y ATM call: pure +2% spot → residual −1.4·10⁻⁴ against the analytic third-order term −4.6·10⁻⁴ (stderr 1.0·10⁻³, delta term 1.109, gamma term 0.038); pure +0.5 vp → residual 2.8·10⁻⁵ (stderr 1.6·10⁻³).
- **Engine details.** The pricing memo is keyed by the product's attributes at full precision (`product_key`; reprs round floats), the calibration key and the mode; products are rebound to the state's rate curve (rho moves discounting); `BSBuilder` counts only `recalibrate`-mode builds; a `Portfolio` product (weighted legs) was added to `products/base.py`; `openpyxl` moved into the core dependencies for `RiskReport.to_excel`. `bergomi.py` / `lsv.py` gained the initial factor state `x0` (numerics unchanged with `x0 = None`; code tag kept at `m4b`, guard refreshed).
- **Budget (§7.13).** See the table below (filled by `scripts/m5_budget.py`).


---

## 8. Hedging framework

Port the cliquet and FVA hedging studies onto a generic engine:
- `HedgeInstrument`: spot, vanilla (strike, maturity), variance swap, forward variance swap, forward-start straddle; each priced under the **pricing model** at rebalancing dates using the conditional-pricing regression (§4.4) or nested MC (configurable, regression default).
- `Hedger(strategy, pricing_model, world_model, schedule)`: paths are generated by the world model (may differ from the pricing model: different ω, θ, ρ, or a different model class); at each rebalancing date the strategy rebalances hedge quantities computed under the pricing model (delta, cap-call strip, net-sized forward variance swaps, etc.); P&L accumulated to maturity with transaction costs (bid/ask in vol points on vanillas and var swaps, bps on spot).
- Output: P&L distribution (mean, std, quantiles, worst paths), regime breakdown (by realised vol, by realised skew), tables and figures in the study format.
- First regression targets: cliquet strategy ranking and the "delta + q-weighted cap calls + net-sized var swap, monthly" result with mean P&L within ±0.2% of notional across regimes.

---

## 9. Viewers

`viewers/precompute.py` builds grids into the cache: default grid `ω ∈ {0, 0.5, 1, 1.5, 2, 2.5, 3}`, `ρ1 ∈ {−0.9, −0.7, −0.5, −0.3, 0}`, `k1 ∈ {0.5, 1.5, 4}`, 2F presets (a few `(θ, k1, k2, ρ12)` combinations including the SSR ≈ 1.2 fit), always including the 1F degenerate points so the old studies are recoverable. Precompute is a CLI with resume support. Owner additions (recorded at the M4c review, for M9): the precompute CLI takes an explicit list of grid points and a worker count, so a grid can be sharded across cores or machines and resumed — `volsto-precompute --shard i/n` runs the i-th of n interleaved shards of the point list; the cache is relocatable (relative paths only, manifest-driven), so grids can be computed on a rented multi-core VM and synced to a laptop; production entries use 8·10⁵ particles (§11).

Streamlit pages (all read-only over the cache, Plotly figures, parameter sliders snap to the grid):
1. **Surface & model** — target surface, Dupire local vol, leverage function heatmap, calibration error map.
2. **Forward smile** — forward-start smiles `T1 → T2` vs parameters; forward ATM vol vs forward variance-swap vs forward vol swap; the put-wing invariance from the study.
3. **Forward skew** — forward ATM skew and curvature term structure vs ω, ρ, θ, k's.
4. **Smile dynamics** — conditional smile after a spot move, SSR term structure, vol-of-vol term structure, `Var(V)` decomposition.
5. **Model risk** — one product, all models calibrated to the same surface, price with error bars, side by side. This is the page the studies are really about.
6. **Product grid** — price heatmaps over two chosen parameters; vega ladders; spot-shift profiles; term-sheet editor for the product.
7. **Hedging** — P&L distributions of stored hedging runs.

Excel export button on every table (openpyxl).

### 9.1 PDE cross-check for the 1F LSV (`pde/lsv1f.py`, M6 Part 4)
State `(x = ln S, X)` with `θ = 0`: `V_t = ξ_0^t exp(ω X_t − ½ ω² (1 − e^{−2k1 t})/(2k1))`, `L = L(t, x)`. Backward PDE `∂_t u + (r − q − ½ L² V) ∂_x u − k1 X ∂_X u + ½ L² V ∂_xx u + ½ ∂_XX u + ρ L sqrt(V) ∂_xX u − r u = 0`, ADI (Hundsdorfer–Verwer or Craig–Sneyd) on a non-uniform grid, `x ∈ ±4 sd`, `X ∈ ±4 sqrt(1/(2k1))`, 400 × 120 points, dt from the `StepSchedule`. Continuous barriers as Dirichlet conditions. Validation: vanilla and digital vs MC within 2 stderr; continuous KO call and European KI put vs MC (bridge form) within 2 stderr; a 1y ATM VKO is out of scope (needs a third state variable). Grid convergence reported (Richardson in dx, dX, dt). Implemented grid: 400 × 121 (odd `n_X` so `X = 0` is a node) with the x half-width ±6 ATM sd for factor models and ±4 for BS / LV (owner decision at the M6 review); measured truncation on the pure 1F set (ω = 3, κ = 1.5, ρ = −0.7, ξ₀ = 4%, central spacing fixed): 1y call 6.905522 / 6.906973 / 6.907117 / 6.907168 at ±4 / ±6 / ±8 / ±12 sd (relative 2.4e-4 → 3e-5 → 7e-6), put 5.920395 / 5.921861 / 5.922006 / 5.922053, digital 0.597621 / 0.597707 / 0.597710 / 0.597708 (1.5e-4 → 5e-6), 3m call 3.720283 / 3.720620 / 3.720655 / 3.720671.

Headline additions (M6): a 3y annual autocall (AC 100%, c = 6% p.a., European KI 60%) and a 3y Phoenix (CB 70%, memory, American KI 60% daily): price, expected life, P(KI), for ω = 0/1/2/3 in 1F and the Table 8.2 2F set, 8·10⁵ particles.

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
- Scheme tests: pure LV vs SSVI at 1m/3m/6m ATM within 0.1 vp on the default schedule; bias halves when dt halves; Richardson diagnostic 2P(dt/2) − P(dt) under CRN; the calibrated LSV simulated with the calibration seed reproduces the particle cloud at the horizon to 1e-12; acceptance tests price on fresh seeds.
- M4 regression baseline (placeholder surface, 400k paths, seed 2024; `tests/test_m4_regression.py`), re-recorded at M4b on the second-order step and the 1/1460–1/365–1/250 schedule with 2·10⁵ particles: 1y→2y ATMF forward vol 21.44/20.76/19.69/18.45 (ω = 0/1/2/3, 1F ρ = −0.7, κ = 1.5) and 19.17 (2F Table 8.2); forward VS 25.1–25.3; capped cliquet 1y 1.193/1.314/1.624/1.968 % and 2y 0.754/0.951/1.459/2.024 %; tolerance max(2 stderr, 0.02% of notional / 0.02 vol points). Against the M4 record every entry moved by less than 2 stderr except the 2y cliquet at ω ≥ 2 and the 2F set (+0.017–0.025% of notional, 2.1–2.9σ) and the ω = 3 forward vol swap (−0.05 vp, 2.3σ). Re-recorded at 8·10⁵ particles in M4c (`HEADLINE_N_PARTICLES` in the test file): 21.46/20.82/19.70/18.45 and 19.22 forward vol, forward VS 25.1–25.3, cliquet 1y 1.199/1.321/1.629/1.971 % and 2y 0.749/0.954/1.464/2.033 % — at most 1.2σ from the 2·10⁵ record — plus the §6.4 conditional-variance and VKO columns. The study-archive tests are skipped until `study_archive/` is present.
- Regression tests (`slow`, recalled from the original study — must be re-verified against the study archive; the SSVI parameters and seeds must be taken from that archive, not guessed): with the study surface, 1F, `ρ = −0.7`, `κ = 1.5`: 1y-into-1y ATM forward vol 21.5% (LV) → 18.6% (ω = 3), forward variance-swap level 25.2–25.5% across ω; 1y capped cliquet (2% cap, global floor 0) 1.082% / 1.194% / 1.472% / 1.763% of notional for ω = 0/1/2/3 with stderr 0.004–0.009%; 2y version 0.639% → 1.718%. Tolerance: 2 stderr, or 0.02% of notional if the seeds cannot be matched.

---

## 11. Conventions and quality bar

- Every equation implemented has a docstring stating the formula, its source (derived / Bergomi / Gatheral / Guyon–HL), and the test that checks it.
- No silent defaults for model parameters; configs are explicit YAML, validated with clear errors.
- Numba kernels are pure functions on arrays; Python wrappers do validation.
- Standard errors everywhere; no MC number printed without one.
- Logging via `logging`, not print. A `--profile` flag on the study runner.
- Reproducibility: every artefact (cache entry, study output) records the git commit, config hash and seed.
- Particle counts (owner decision, M4b): production numbers — headline tables, regression baselines, viewer precompute — use 8·10⁵ particles, single seed; development and fast paths use 2·10⁵. Baseline files state the particle count.
- Style: black, ruff, mypy (strict on `volsto/`), 100-char lines.
- README with a 20-line quickstart that calibrates a 1F LSV on the reference surface and prices the study cliquet.

---

## 12. Milestones (in order; each ends with green tests and a commit — standing rule: commit every green milestone without asking)

M1. Market layer + BS + local vol + MC engine + vanilla/variance products. **Done.**
M2. Bergomi 2F kernel, exact stepping, closed-form tests, 1F degeneracy, mixing solution. **Done.**
M3. Particle calibration + cache + diagnostics (+ M3.2 tail/scheme fixes, code-tag guard). **Done.**
M3b. HistoricalData.net importer, eSSVI, snapshot configs (§13). **Done.**
M4. Forward-start, cliquet family, FVA, forward smile analytics, headline table, regression baseline. **Done.**
M4b. Second-order SV spot step shared by calibration and pricing (§3.1); rerun the M4 tables and report what moved. **Done, accepted 2026-09-13** on seed averages (1y–2y variance swap at −0.04 to −0.06 vp noted); open residuals tagged "calibration-side pass before M6" (§4.2 M4b notes).
M4c. Conditional/KO variance products and the VKO put (§6.1–6.2), headline table extended. **Done and accepted 2026-09-13** (§6.4); at the review the VKO sign claim was withdrawn in favour of the measured barrier sweep and ITM-conditional realised-vol distribution (§6.2), and the "running" flag dropped.
M5. Risk layer (§7). **Done and accepted 2026-09-13** (§7.15 notes; the seven recorded deviations approved; the cliquet Bachelier check relabelled an approximation; budget runs at 2·10⁵ and 8·10⁵ particles follow M6 Part 0 and are recorded in §7.15).
M6. Autocall/Phoenix/barrier products with decompositions (§6.5–6.7); PDE 1F cross-check (§9.1); Part 0 first: the calibration-side pass of §4.2. **Done 2026-09-13, pending owner review** — Part 0 closed with the owner's option (b): the second-order SV step's cross-term compensator corrected (a first-order drift error linear in the mean-reversion speed; acceptance worst 0.0062 vp per halving), code tag `m6`, the gate met for both reference sets on six pricing seeds, the 1y–2y variance swap −0.08 vp recorded as a known far-wing bias (§4.2 M6 Part 0 notes); Parts 1–4 in §6.8 / §9.1 with the conventions listed for review; M6 headline rows in §6.8; regression baselines re-recorded.
M7. Smile dynamics, SSR estimators, historical estimators, 2F fitting, stability, VIX check (§15).
M8. Hedging framework; port the cliquet and FVA hedging studies as regression tests.
M9. Precompute CLI + Streamlit viewers + Excel export.
M10. Study runner with LaTeX output; regenerate the original paper's tables; backtest study on the market history (§15 Part 4).

Open items for the owner: the original study archive (SSVI parameters, seeds, tables) — the M4 cliquet baseline is ≈ 10–16% above the study at every ω with ratios across ω agreeing to 1%, consistent with a surface difference; the paid EOD archive for a multi-year backtest.

---

## 13. Market data import (M3b, implemented)

`market/import_hdn.py`: importer for the HistoricalData.net EOD option-chain CSV (34 columns, one file per trading day; the free `options_sample_2022H2.zip` in `./data/hdn_sample/`, git-ignored, uses the paid format). Pipeline: `load_day` (SPX and SPXW roots for the index surface, both quotes required, AM/PM settlement in the time-to-expiry convention, manifest Treasury curve); `implied_forward` (regression of C − P on K near the money; the vendor's parity forward as cross-check — their iv and Greeks are never inputs, since quotes across contracts are not synchronised snapshots); `to_grid_surface` (OTM mids, liquidity filter using iv_bid/iv_ask, butterfly and calendar pruning); `fit_ssvi` (θ at the SPEC pillars by isotonic least squares, global (ρ, η, γ) constrained; **eSSVI per-pillar ρ is the default for imported surfaces**, `--ssvi` opts out); `snapshot_config` (dated YAML with provenance: vendor, checksum, filters). CLI `volsto-import --vendor hdn --date … --underlying SPX`. Slices are grouped by (root, expiration) because SPX AM and SPXW PM share dates; fits use expiries from 3 weeks and |k| ≤ 0.25, vega-weighted; expiries under 3m are reported outside the acceptance region. `scripts/capture_yfinance.py` writes today's SPX/SPY chain in the same layout for daily accumulation.

Measured on 2022-09-15: implied forwards within 0.7 bp (≤ 6m) / 3.5 bp (2y) of the vendor's; SSVI RMS 0.20 vp inside ±20% from 3m; eSSVI RMS 0.17 / 0.08 vp (3m–2y / 6m–2y); 1–2m weeklies 1–4 vp off under SSVI; ≈ 1 s per day.

Implementation notes (M3b, measured on the 2022 H2 SPX sample): contracts are grouped by (root, expiration) as the vendor does — SPX (AM) and SPXW (PM) share expiration dates but differ by one day in `T`; the implied forward is the regression `C − P = a + b K` on two-sided pairs within ±10% of spot (`F = −a/b`, discount `−b`, delta-method standard error), which agrees with the vendor's closest-strike parity forward within 0.7 bp to 6m and 3.5 bp at 2y (the vendor discounts at the Treasury rate, the market-implied rate was 0.4–0.6% higher); our Black-76 inversion reproduces the vendor's `iv` on `iv_flag = 0` rows to a median 0.1 vp near the money. Butterfly pruning is iterative convexity of OTM prices in strike; the calendar check runs on the common quoted `k` range of consecutive slices (a global fixed point dropping the worst violating quote), which is also what `GridSurface` enforces. `θ_T` is fitted at the SPEC pillar tenors (1m, 3m, 6m, 1y, 18m, 2y, 3y) by isotonic least squares through the slices' ATM total variances; the global `(ρ, η, γ)` least squares (vega-weighted, |k| ≤ 0.25, expiries ≥ 3 weeks) enforces both SSVI butterfly conditions through a bounded reparametrisation of `η`; imported surfaces default to eSSVI — `ρ` fitted at the pillars (piecewise-linear `ρ_T`, `ESSVISurface`) — with `--ssvi` as the single-`ρ` opt-out; synthetic surfaces (`configs/surfaces/reference_ssvi.yaml`, the study parameters) stay plain SSVI. Expiries under 3m are reported in the residual tables but sit outside the acceptance region. Fit quality on four sample days, inside ±20% moneyness: RMS 0.12–0.22 vp and max 0.4–1.0 vp from 3m to 2y (eSSVI RMS 0.08–0.16 from 6m); the 1–2 month weeklies of these high-volatility days are 1–7 vp off — a single power-law φ cannot follow them, so the §13 target of 0.3 vp holds in RMS from 3m but not as a maximum below 3m. The snapshot YAML has `market` + `ssvi` sections (loadable by `load_ssvi_surface`) and a `provenance` section (vendor, file and manifest SHA-256, filters, forwards, rate curve, fit residuals, code version). `scripts/capture_yfinance.py` writes the same 34-column layout (blank calculated columns, `iv_flag = 7`) plus a manifest with a user-supplied Treasury curve.

Data sourcing decisions: build against the free sample; buy the HistoricalData.net full archive ($799 one-time, daily updates $79/month) for a multi-year backtest; ThetaData's free tier (1y EOD) as a second-source check; rates from FRED/SOFR or the desk's OIS curve; forwards and dividends always implied per expiry from put-call parity on the chain.

---

## 14. (superseded — the delta regimes are now §7.2)

---

## 15. Smile dynamics, SSR and fitting the 2F parameters (M7)

### Part 1 — smile-dynamics analytics
Numerical SSR at t = 0 for the LSV (book §12.4.3 joint-bump trick) and for pure SV (§9.8); the analytic decomposition eq. 12.52 with R^LV from 12.53–12.54 and R^SV from 9.21; the slow short-horizon regression estimator. Tests as §4.4, plus enabling the M5 test "model-regime ATM-vol shift per unit Δ = SSR_T·s_T". Vols of ATMF vols (eq. 12.56). Conditional smile after a spot move at horizon t; Var(V) decomposition.

### Part 2 — historical estimators from a surface history (`calibration/history.py`)
Input `SurfaceHistory`: dated snapshots (importer configs or synthetic) with forward curves; pillars at constant time-to-maturity T ∈ {1m, 3m, 6m, 1y, 2y, 3y}. Per date and pillar: `vs_vol` (log-contract strip), `atm_vol` (k = 0), `skew` (dσ/dk at k = 0), `ln_spot`. Estimators with Newey–West standard errors:
1. `volvol_hist(T) = sqrt(252)·std(Δ ln vs_vol(·,T))` and the cross-pillar correlation matrix of Δ ln vs_vol.
2. `SSR_hist(T) = slope(T) / mean skew(·,T)`, slope from regressing Δ atm_vol(·,T) on Δ ln_spot (book eq. 9.3 read historically); report R².
3. Mean and latest ATMF skew term structure.
4. Realised spot/vol correlation per pillar (diagnostic).
Windows: 250 days default for 1 and 3, 60 days for the SSR (regime-dependent), both reported; rolling versions for Part 4.

Test (identifiability): simulate the pure 2F model daily for 3 years (Table 8.2, flat 20% VS curve), compute each day's VS vols from the factor state and ATMF vols/skews via the mixing solution, feed the estimators: `volvol_hist` recovers eq. 7.39, `SSR_hist` recovers eq. 9.21 (order-one accuracy, 2 SE), cross-pillar correlations recover eq. 7.20.

### Part 3 — fitting (`calibration/fit_2f.py`)
Parametrisation: book notation; stage 2 uses the eq. 8.56 form (ρ_SX1, χ) so PSD holds by construction; ν stored, ω = 2ν exposed.

**Stage 1 — variance dynamics (ν, θ, k1, k2 | ρ12).** Targets: `volvol_hist(T)` at the pillars (log space), optionally VIX-implied vol-of-vol at horizons ≤ 9m (Part 5) with its own weight. Model: instantaneous vol of VS vol, eq. 7.39 (A_i from 7.38; the non-flat-curve form if the window's mean VS curve is sloping). Weighted LS in log(volvol), weights 1/SE². ρ12 fixed by default (0 or user value); option `from_correlation`: fit ρ12 by matching the model's corr(Δ ln vs_vol_3m, Δ ln vs_vol_2y) (eq. 7.20 machinery) to the historical one, iterating twice. Bounds: ν ∈ [0.3, 4], θ ∈ [0, 1], k1 ∈ [1, 20], k2 ∈ [0.05, 1.5], k1 > k2; report Jacobian singular values and flag k2 as bound-driven when insensitive. Degeneracy check: rerun from Table 7.1 Sets I–III as starting points and report all optima — Bergomi's Table 7.1 shows that for a given decay exponent many parameter sets fit the vol-of-vol curve equally well; the correlation target is what separates them (§7.4.2: higher forward-variance vols go with lower correlations; correlations are invariant to shifting all k's).

**Stage 2 — spot/vol correlations (ρ_SX1, ρ_SX2), stage 1 frozen.** Targets: ATMF skew term structure (latest surface for a pricing date) and `SSR_hist(T)` (60-day window). Model: order-one skew eq. 8.55 (9.18 for a sloping curve) and SSR eq. 9.21 (9.19), then a refinement replacing the skew formula by the mixing-solution skew of the naked 2F model. Weighted LS on both curves (default weights put skew residuals in vol points per unit k / 10 and SSR residuals in units of 0.1 on the same scale; SSR pillars beyond 1y down-weighted). Outputs: fitted correlations and the naked-skew-vs-market-skew table (the leverage absorbs the residual; small means the SV carries the skew, as intended). Forward-skew consensus marks (Totem) are not used: not available on the desk and noisy; Bergomi's point that mixed models can be parametrised for given *future* skews is applied as a stage-3 check instead.

**Stage 3 — validation on the calibrated LSV.** Calibrate leverage; report (a) mean |L − 1| over the grid, (b) model SSR (numerical) vs `SSR_hist`, (c) model vol-of-vol of VS vols with leverage vs `volvol_hist`, (d) forward-start ATM vol and forward 90/110 skew at 1y-into-1y and 2y-into-1y, (e) the headline product table on this fit. Nothing is refit here.

API: `fit_2f(history, pricing_date, config) -> FitResult(params, stage1_report, stage2_report, stage3_report, config_yaml)`; the YAML is a loadable model config with provenance (window, settings, objectives).

Tests: recovery on the synthetic history (ν, θ, k1 within 10%, k2 within its sensitivity band; ρ_SX1, ρ_SX2 within 0.05); degeneracy (three Table 7.1 starts, near-identical objectives with ρ12 free, unique optimum with the correlation target); real-data end-to-end on the 2022 H2 sample (no fixed numbers).

Rationale: in an LSV the leverage absorbs the vanilla surface, so vanillas identify none of the seven parameters; each group is pinned by a non-vanilla observable and fitted in stages (Bergomi ch. 7, ch. 12 conclusion; Guyon 2019/2022 on VIX-based calibration of 1F/2F Bergomi; the 2025 quantisation study reporting stable daily 2F parameters from VIX futures and options). VIX pins ν and the fast factor only (futures liquid to ~9m, options to ~6m read a 1m forward variance), so the historical route is the base for the whole term structure and VIX a short-end check for SPX.

### Part 4 — stability (`calibration/stability.py`)
Rolling refits (stage 1 monthly on a trailing 250-day window, stage 2 daily) → parameter time series with SEs; flag parameters whose day-to-day changes exceed their SE band consistently (unidentified, not informative); plot the objective along the ν ⟷ correlations degeneracy direction. Used by the backtest study.

### Part 5 — VIX in the 2F model (`analytics/vix.py`, check only)
`VIX²_T = (1/Δ) ∫_T^{T+Δ} ξ_T^u du`, Δ = 30/365; in the 2F model `ξ_T^u` is explicit in (X¹_T, X²_T) (eqs. 7.30–7.35), so VIX futures and calls price by 2D Gauss–Hermite quadrature (book §7.7.2). Implied vol-of-vol at horizon T from the ATM VIX option. Tests: quadrature vs factor Monte Carlo within 2 stderr; VIX² futures vs forward 1m VS variance up to the convexity term; short-horizon implied vol-of-vol → eq. 7.39 as T → 0. Optional stage-1 target for SPX only, fitting futures and ATM VIX vols never the wings (book §7.7.4: the lognormal 2F model under-produces the VIX smile's upward skew).

---

## 16. Desk conventions recorded
- "P1 deco" = this LSV two-factor Bergomi model. ω = 2ν; the earlier studies' ω = 3 is ν = 150%.
- Surfaces: the desk marks with SABRW (in-house); the library keeps SSVI/eSSVI (public, arbitrage-free by construction, analytic Dupire); `ImpliedSurface` is an ABC so another parametrisation can be added without touching calibration.
- Vega by maturity is by cumulative "waves" (§7.4). Delta regimes include sticky skew (§7.2).
- Up-var indicator "T−1" or "T & T−1"; down-var "T" or "T & T−1"; KO var is up-and-out, close-to-close, variant (b) (§6.1).

---

## 17. Change log v1.1 → v2.0
- §2: Dupire grid (√t spacing, ±3.0), replication bounds ±25 sd, PCHIP ξ₀, eSSVI class.
- §3.1: Platen weak-2 default, frozen-L rule, StepSchedule, record_all_steps, M4b second-order SV step; Milstein explicitly excluded.
- §4: local-linear particle regression with curvature correction and k-NN floor, L in k, leverage grid ±2.5, noise-aware acceptance, code-tag guard.
- §5: batch-means stderr, accumulator conventions.
- §6: conditional/KO variance products, VKO put, M4 implementation notes.
- §7: full M5 risk layer (regimes incl. sticky skew and sticky local vol, vega-T waves, forward-variance ladder, skew/curvature, cross-Greeks, product risks, precision estimators, attribution).
- §12: milestone status, M4b/M4c inserted, standing commit rule.
- §13: importer as implemented and data sourcing decisions.
- §14: superseded by §7.2.
- §15: M7 smile dynamics, historical estimators, staged 2F fitting, stability, VIX check.
- §16: desk conventions.
- 2026-09-13 M5: §7.15 implementation notes (fwd-var sizing on the log-contract strip, saturating skew/curvature profile, one-sided boundary bumps, cliquet Bachelier deviation measured, fixing/barrier/realised-variance definitions, LR aggregation, cubic conditional basis, attribution conventions, budget); §12 M5 status.
- 2026-09-13 merge (this file): the implementer's measured tables and notes re-applied over v2.0 — §2.4 replication bounds, §3.1 scheme switches and M4b status, §3.3 second-order step, §4.1 M3 notes and production particle count, §4.2 acceptance table, pre-M4 findings and M4b notes, §5 grid rule, §6.3 M4/M4b tables, §10 M4b baseline, §11 particle-count convention, §12 M4b status, §13 M3b notes.
