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

Second-order SV spot step (M4b): exact factor increments with the intra-step spot/variance covariance (Andersen-type), removing the O(dt) error from freezing leverage and SV variance over the step; acceptance: on the coarse schedule, VS error ≤ 0.05 vp at every pillar 1m–3y for N = 2·10⁵ and 8·10⁵ particles, put wing ≤ 0.05 vp at 2 SE, 3y calibration ≈ 30 s. **Status (measured, §4.2 M4b notes):** accepted on seed averages — calibration 33 s at 2·10⁵ particles; variance swap within 0.05 vp at 1m–6m and 3y, −0.04 to −0.06 at 1y–2y; single runs at 2·10⁵ particles carry ±0.035 vp of particle-seed noise per pillar. The kernel's own step error at dt = 1/365 was ≤ 0.03 vp even with the frozen step (CRN refinement), so the second-order step's visible gains are on the pure SV model (1m: variance swap −0.06 → 0.00 vp, smile 0.14 vp flatter → within 0.03 vp of the mixing solution) while the calibrated-LSV improvements came from the calibration side.

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
S0 grid −30% to +30% in 2.5% steps; price, model delta/gamma/vega per shift under "model" dynamics (other regimes optional, recalibrated per shift, cached); gamma profile as second difference. Cliquet gamma profile versus the accumulated sum at an intermediate date, with the Bachelier call-on-remaining-capped-sum decomposition of the original study as analytic cross-check (exact in BS with independent legs; report the LSV deviation).

### 7.9 Model-parameter sensitivities (`risk/volsto_sens.py`)
Partials to each of (ν, θ, k1, k2, ρ12, ρ_SX1, ρ_SX2) with recalibration; ν also sticky-leverage. Central differences, PSD check on correlation bumps. One table per product.

### 7.10 Product-specific risks (`risk/product_risk.py`)
- Fixing risk: forward-start and cliquet delta/gamma/vega at T1 − 1d and T1 + 1d (state held), reported as the jump; per fixing for cliquets.
- Barrier risk: `∂price/∂B` for KO var and KI put; delta and gamma within ±5% of the barrier at 0.5% steps; VKO `∂price/∂H` (vol points of the vol barrier) and `∂P(KO)/∂ln S`.
- Realised-variance exposure: expected dollar-gamma-weighted variance per fixing period along the path; flat and equal to notional for a variance swap.

### 7.11 Precision options (`risk/estimators.py`)
Likelihood-ratio delta and vega for discontinuous payoffs (barriers, KO var, VKO, digitals) with the bump estimate as cross-check (flag > 3 stderr disagreement); conditional Greeks at a future date by regression of payoff and CRN bumps on a polynomial basis in (ln S_t, X¹_t, X²_t) — the hedger's engine; control variate on the difference (BS Greek at market vol for vanilla-like products), with variance reduction reported.

### 7.12 P&L attribution (`risk/attribution.py`)
`explain(product, state_0, state_1)`: sequential CRN revaluation in the order spot (delta, gamma), surface (parallel vega, vega-T waves, skew/curvature), model parameters, factor state; residual = actual minus sum. States carry S0, surface, model params, factor values, date. Tests: pure spot move residual at the gamma-cubed level (report); pure parallel vega move residual within 2 stderr.

### 7.13 RiskReport and budget
One object per product with every sensitivity, stderr, bump specs, cache keys; `to_dataframe()`, `to_excel()`. A full report is ≈ 80–100 recalibrations; print count and wall clock — it is the viewer precompute budget.

### 7.14 Smile-dynamics analytics (`analytics/smile_dynamics.py`, `analytics/var_decomp.py`)
As §4.4 and §15 Part 1: numerical SSR (LSV joint-bump, pure SV), eq. 12.52 decomposition, short-horizon regression estimator; vols of ATMF vols (eq. 12.56); conditional smile after a spot move at horizon t; vol-of-vol term structure; Var(V) decomposition (closed form pure SV, regression estimator LSV).

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
M5. Risk layer (§7).
M6. Autocall/Phoenix/barrier products with decompositions; PDE 1F cross-check; the calibration-side pass of §4.2 (owner's diagnostic first) and the 1F put-wing re-check precede the KI put.
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
- 2026-09-13 merge (this file): the implementer's measured tables and notes re-applied over v2.0 — §2.4 replication bounds, §3.1 scheme switches and M4b status, §3.3 second-order step, §4.1 M3 notes and production particle count, §4.2 acceptance table, pre-M4 findings and M4b notes, §5 grid rule, §6.3 M4/M4b tables, §10 M4b baseline, §11 particle-count convention, §12 M4b status, §13 M3b notes.
