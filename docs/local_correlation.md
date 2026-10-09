# The local correlation model

A theory note for milestone M12. The as-built record — conventions, tests, measured numbers,
review items — is SPEC §8.7; this note explains why the model is what it is and what it says
about the Palladium. Bergomi's *Stochastic Volatility Modeling* is cited by chapter. Other
sources are those of the owner's specification and were not re-read for this note; they are
named where their result is used.

Tags: **[measured]** a number produced by this repository (the command or file is given),
**[derived]** an identity or an estimate obtained on paper here, **[decision]** a convention.

Numbers on market data come from the dispersion study's Dow basket (30 names, price weights,
DJX as the index) on four dates: 2026-10-02 ("today"), 2019-09-03, 2017-04-03 (steep index
skew, low vols) and 2008-07-07. Unless a line says otherwise they are 3-month numbers from
`scripts/lcm_price.py` at the development budget (2·10⁵ particles and paths; today's also at
the production budget of 8·10⁵), with Monte Carlo standard errors on antithetic pair means.

---

## 1. The model (SPEC §8.7, LC2–LC3)

Each name keeps the model that already reprices its own smile: a Dupire local volatility
`σ_i(t, k_i)` in forward log-moneyness `k_i = ln(S_i/F_i(t))`, built from SVI slices fitted to
its listed expiries (Bergomi ch. 2; SPEC §2.3, §8.7 LC1). The names are tied together by a
correlation matrix that depends on the state through one scalar:

    ρ(λ) = (1 − λ)·R_low + λ·R_high,        λ = λ(t, k_B) ∈ [0, λ_max],

where `k_B = ln(B_t/F_B(t))` is the log-moneyness of the basket `B_t = Σ_i w_i S_i(t)/F_i(t)`
(the "performance" basket: a martingale with `F_B ≡ 1`) and `R_low`, `R_high` are two fixed
correlation matrices. The default is `R_low` an equicorrelation at 0.02 and `R_high = 11ᵀ`, for
which `ρ(λ)` is the equicorrelation `ρ_t = 0.02 + 0.98·λ`.

The Brownian increments of a step are built as

    dW = √(1 − λ)·L_low ε + √λ·L_high η,

with `ε ∈ ℝⁿ` and `η ∈ ℝʳ` independent standard normals, `L_low L_lowᵀ = R_low` and
`L_high L_highᵀ = R_high`. Conditionally on the state at the start of the step the covariance is
`ρ(λ)` [derived]. With `R_high = 11ᵀ`, `η` is one common factor: the model is "idiosyncratic
noise plus a market factor whose weight depends on where the index is".

With `u_i = ω_i σ_i` (`ω_i = w_i X_i/B` the current weights), the instantaneous variance of the
basket is affine in `λ` [derived]:

    v_B = uᵀρ(λ)u = a + λ·b,    a = uᵀR_low u,    b = uᵀ(R_high − R_low)u ≥ 0.

## 2. Gyöngy and the formula for λ (SPEC §8.7, LC4)

Gyöngy's theorem (1986; Bergomi ch. 2 uses it for local volatility and ch. 12 for the
local-stochastic models): an Itô process has the same one-dimensional marginals as the Markov
diffusion whose drift and variance are its conditional drift and variance. Applied to `ln B`:
the basket has the marginals of the listed index — it reprices the whole index smile — if and
only if

    E[v_B(t) | k_B] = σ_B²(t, k_B)      for every t up to the horizon,

`σ_B` being the Dupire local volatility of the index smile. Because `λ` is a function of
`(t, k_B)` it comes out of the conditional expectation, and the condition can be solved for it:

    λ(t, k) = (σ_B²(t, k) − E[a | k_B = k]) / E[b | k_B = k],

clipped to `[0, λ_max]`. The formula is the multi-asset analogue of the leverage function
`L² = σ_Dup²/E[V | S]` of a local-stochastic volatility model (Bergomi ch. 12; SPEC §4.1): the
unknown appears on the right through the law of the process, a McKean–Vlasov equation.

For the equicorrelation family the formula reads `ρ_t = (σ_B² − E[Σu_i²|k]) / E[(Σu_i)² − Σu_i²|k]`:
the Cboe implied-correlation formula applied to local variances and conditioned on the index
level [derived].

The clip matters. Where the formula asks for `λ > λ_max` the index is more volatile at that
level than fully correlated names can make it; where it asks for `λ < 0`, less volatile than
nearly uncorrelated ones. The mass of the cloud on which the clip binds is recorded at every
step ("clipped mass"); it is the measure of how far the index smile is outside the model's
reach (§8).

## 3. The affine family

**Positive semi-definite by construction [derived].** For `λ ∈ [0, 1]`, `ρ(λ)` has a unit
diagonal and `xᵀρ(λ)x = (1 − λ)·xᵀR_low x + λ·xᵀR_high x ≥ 0`. It is a valid correlation
matrix in every state, with no repair step. This is the point of Guyon's "local correlation
families" (Risk, 2014): one scalar function is enough to match the index smile, and choosing
the family is choosing which admissible model one gets.

**Admissibility.** `v_B` must increase with `λ` for the inversion to be monotone: `b ≥ 0`
whenever `R_high − R_low ≥ 0` entrywise, since `u_i ≥ 0`. With `R_high = 11ᵀ` every `R_low`
qualifies. A calibration is admissible where the unclipped `λ*` stays inside `[0, λ_max]`.

**What the family does not pin down.** The index smile gives one function of `(t, k)`. It
fixes the level of correlation as a function of the index level and nothing about how that
correlation is distributed among the pairs: that is the choice of `R_low` and `R_high`. A
historical-scaled `R_low` keeps the sector structure, the equicorrelation does not. Products
on the whole basket are little exposed to the choice; a sub-basket product is, and must carry
the model-risk range of SPEC §8.7 (LC6).

**Langnau's pathwise λ.** Langnau (Risk, 2010) solves `v_B = σ_B²(t, k_B)` path by path:
`λ = (σ_B² − a)/b` with the path's own `a` and `b`. The basket is then exactly a one-dimensional
local-volatility process. Here `λ` is a function of `(t, k_B)` only and the condition holds in
conditional expectation: the marginals of the basket are the index's, its dynamics are not
Markov in `k_B`. The two are different members of the admissible set and price exotics
differently.

## 4. The particle method and the frozen rule (SPEC §8.7, LC3–LC4)

`λ` is computed forward in time on a cloud of `N` simulated paths (Guyon and Henry-Labordère's
particle method; Bergomi ch. 12): at `t_j` the conditional expectations `E[a | k]` and
`E[b | k]` are estimated by kernel regression on the cloud, `λ(t_j, ·)` follows from the
formula, and the cloud is moved to `t_{j+1}` with that `λ`.

**Frozen over a step [decision].** Over `[t_j, t_{j+1}]` every path uses `λ(t_j, k_B(t_j))`.
Two consequences [derived]:

* *Each name keeps its law exactly.* The mixed normal `Z_i = √(1−λ)(L_low ε)_i + √λ(L_high η)_i`
  is standard normal conditionally on the start of the step, whatever `λ` is. Name `i`'s
  discrete path has exactly the law of the single-asset local-volatility scheme on the same
  grid: single-name vanillas are repriced at every step size, not only in the limit
  [measured: 450 cells on the thirty Dow names, largest |z| 3.60, test C2].
* *The per-name weak order 2 scheme is the multi-dimensional one* for a correlation that is
  constant over the step.

What is neglected is the change of the mixing inside the step, a first-order weak error in the
correlation — the analogue of the frozen leverage (SPEC §3.1). It is largest in the first
weeks, where the cloud is narrow; the schedule therefore uses quarter-day steps over the first
two weeks and daily steps after [decision of the owner; measured: halving every step moves the
Dow forward by −0.008 % ± 0.004 and the index smile by at most 0.045 vp].

**The target is averaged over the step.** The regression at `t_j` is matched to the mean of the
index local variance over `[t_j, t_{j+1}]`, not its value at `t_j`, which removes most of the
first-order bias of the frozen rule on the smile.

**Calibration and pricing share one kernel.** The cloud is advanced by the pricing kernel on
the same streams: the calibrated model simulated with the calibration's seed reproduces the
final cloud bit for bit (test I2), and `λ ≡ 0` reproduces the constant-correlation model
`MultiAssetModel(R_low)` bit for bit (test I1).

No existence or uniqueness theorem covers the equation; the particle method is the numerical
definition of the model, and the step, particle-count and bandwidth checks (S5–S7) are its
practical guarantee.

## 5. Why constant correlation overprices the Palladium

The Palladium forward pays the dispersion `D = Σ_i w_i |R_i − R̄|`, with `R_i` the names'
returns and `R̄ = Σ w_i R_i` the basket's. Its companion in variance is

    V = Σ_i w_i (R_i − R̄)² = Σ_i w_i R_i² − R̄².

The comparison is between the calibrated model (LC) and its constant-correlation companion
(CC): the same local volatilities with the one constant correlation that reprices the index
at-the-money straddle at the horizon.

1. **The first term is the same.** `Σ w_i E[R_i²]` is fixed by the single-name smiles, which
   both models reprice (§4).
2. **The second term is not.** A model that reprices the index skew has a fatter left tail for
   the basket than a constant-correlation model marked at the money, so `E[R̄²]` is higher
   [measured: +17 %, +11 %, +21 %, +2 % on the four dates, LC over CC].
3. **So `E[V]` is lower** [measured, `E^LC[V]/E^CC[V]`: 0.955, 0.909, 0.893, 0.984].
4. **`κ = E[D]/√E[V]` hardly moves** [measured, LC over CC: 0.993, 1.006, 1.000, 0.988].
5. **So the forward falls by about the square root of the fall of `E[V]`** [measured,
   `E^LC[D]/E^CC[D]`: 0.9704 ± 0.0002, 0.9587 ± 0.0002, 0.9452 ± 0.0003, 0.9807 ± 0.0001; at
   the production budget today 0.97008 ± 0.00011].

| | today | 2019-09-03 | 2017-04-03 | 2008-07-07 |
|---|---|---|---|---|
| `E^LC[R̄²] / E^CC[R̄²]` | 1.17 | 1.11 | 1.21 | 1.02 |
| `E^LC[V] / E^CC[V]` | 0.955 | 0.909 | 0.893 | 0.984 |
| its square root | 0.977 | 0.953 | 0.945 | 0.992 |
| `κ_LC / κ_CC` | 0.993 | 1.006 | 1.000 | 0.988 |
| `E^LC[D] / E^CC[D]` | 0.970 | 0.959 | 0.945 | 0.981 |
| the study's model S over its copula | 0.974 (not converged) | 0.944 | 0.876 | 0.980 |
| the reference implementation (its own world) | 0.974 | 0.945 | 0.911 | 0.976 |

(The `E[R̄²]` ratio uses `Σ w E[R_i²]` of the LC run for both models, by point 1. Model S is
the study's skewed one-factor model, `outputs/dispersion/model_s_3m.parquet`; the reference
implementation has no carry and a two-parameter `λ`.)

Path by path `D ≤ √V` (Cauchy–Schwarz with the weights), and with Jensen `E[D] ≤ √E[V]`: the
cap is model-free once `E[V]` is known [derived].

**The constant-correlation companion is the study's copula.** `E^CC[D]` is within 0.7 % of the
study's copula forward `P_D` on the four dates (0.9993, 0.9933, 1.0030, 1.0029) [measured] —
two implementations that share only the data. The LC/CC ratio is therefore also, to that
accuracy, the correction the calibrated model applies to the study's price.

**Where the model stops short.** On the Dow the calibrated model does not reach the index's
downside wing (§8), so its `E[R̄²]` is below the listed index strip's and the fall of the
forward is understated. The sweep reports two bracketing numbers beside `E^LC[D]`:
`ED_wing = κ_LC·√(Σ w E^LC[R_i²] − M_B^listed)`, which replaces the model's basket second
moment by the listed one, and `ED_eqv = κ_LC·√EQV`, which replaces both terms by the listed
strips [measured today at the production budget: LC 0.9701, `ED_wing` 0.9587 ± 0.0005,
`ED_eqv` 0.9471 ± 0.0012, as ratios to `E^CC[D]`; on 2017-04-03, development budget: 0.945,
0.902, 0.886].

## 6. The calls can move either way

A call on the dispersion pays `(D − K)⁺`; what matters is in which states the dispersion is
large.

* Under CC with skewed local volatilities, the names' volatilities rise together in a sell-off
  and the correlation does not: sell-offs carry more dispersion.
* LC raises the correlation in sell-offs (where the index skew asks for it) and lowers it in
  rallies: it removes dispersion from the left and adds some on the right.

The profile `E[D/B | bucket]/E[D/B]` on the study's buckets of the basket return shows it
[measured, lowest bucket (below −10 %), LC against CC: 1.00 against 1.18 today, 1.37 against
1.59, 1.29 against 1.58, 1.19 against 1.31; highest bucket (above +10 %): 1.08 against 1.02,
0.96 against 0.80, 1.04 against 0.91, 0.99 against 0.88].

Which effect wins at a far strike depends on how steep the single-name skews are relative to
the index's. Calls at multiples of the CC forward, LC over CC [measured]:

| strike | today | 2019-09-03 | 2017-04-03 | 2008-07-07 |
|---|---|---|---|---|
| 0.75 × | 0.911 | 0.840 | 0.800 | 0.930 |
| 1 × | 0.903 | 0.611 | 0.602 | 0.892 |
| 1.25 × | 0.953 | 0.295 | 0.368 | 0.927 |
| 1.5 × | 1.018 ± 0.022 | 0.112 ± 0.004 | 0.259 ± 0.010 | 0.892 ± 0.020 |

Today the names' vols are high and their skews flat relative to the index: the far calls lose
little or gain. On the two dates with steep skews the far calls are worth a small fraction of
their constant-correlation price. The far calls also depend on the single-name upside wings,
which the quotes do not pin down (the names' SVI second moment is 1.3 % above the study's
listed strips today, almost all of it above the last listed strike; SPEC §8.7, second
follow-up, decision 5).

## 7. Hedging (SPEC §8.7, LC6)

**Sticky strike.** If the local volatilities are held in absolute spot and `λ` in absolute
index level, a rise of every spot moves the basket to a region of lower `λ`: correlation
falls and dispersion rises. The common delta of the forward, as the percentage of its price per
+1 % on every spot, splits exactly into

* homogeneity, +1 (the payoff is homogeneous of degree one in the spots at fixed moneyness);
* the single-name skew channel, `Δ^CC_ss − 1` < 0 (each name's volatility falls along its skew);
* the correlation channel, `Δ^LC_ss − Δ^CC_ss` > 0.

[measured, LC against CC: today +2.29 against +0.35 (production: +2.2937 ± 0.0045 against
+0.3492 ± 0.0027, skew −0.651, correlation +1.944); 2019-09-03 +1.33 against −2.40;
2017-04-03 +1.75 against −2.57; 2008-07-07 +1.47 against −0.51. On the reference's world for
today the same decomposition is +1.000, −0.804, +2.543 against the reference's +1.000, −0.804,
+2.545, test S4.]

The sign of the total differs between the models on three dates of four: a constant-correlation
book is short the index against its Palladium where the calibrated model is long.

**Sticky moneyness.** If everything moves with the spots the forward's delta is the homogeneous
+1 % in both models. The market lies between the two regimes.

**Vega.** A name's vega depends on what is held when its smile moves: the index smile
(`λ` recalibrated — the vega at a fixed index smile, with the correlation re-marked) or `λ`
(the index smile then moves with the name). The index vega moves from the at-the-money level,
where the constant-correlation model has all of it, towards the index skew, where `λ(t, k)`
gets its shape. `scripts/lcm_price.py --risk full` measures the per-name vegas both ways, the
index vega, two index skew vegas and the model-risk range for today (SPEC §8.7, LC7 results).

## 8. Limits

* **The index downside wing on the Dow** (SPEC §8.7, owner's decision 7 of the second round).
  Local correlation on own-level local volatilities cannot generate today's DJX downside
  skew: the clip at `λ_max` binds on 1.8 % of the mass inside ±2.5 sd today and on 13 to 15 %
  on 2019-09-03 and 2017-04-03, and the model's index vol is 0.5 vp low at −1.5 sd and 1.5 vp
  low at −2.5 sd at 3m today [measured]. It is not a data inconsistency (the DJX puts are
  inside the comonotonic bound of the thirty marginals), not the names' wing fits (within
  0.2 vp of their quotes), and not the cap (raising `ρ_max` from 0.98 to 0.995 moves the wing
  by 0.04 vp). In a sell-off the index needs the names' volatilities to rise *because the
  index fell*, not only because each name fell: cross-dependent volatility (Guyon, Risk 2016),
  per-name stochastic volatility with correlated factors, or common jumps.
* **Deterministic correlation given the index.** Given `(t, k_B)` the correlation has no
  randomness of its own, so the model understates the variability of dispersion: far calls on
  dispersion are too cheap relative to stochastic-correlation models fitted to the same smiles
  (sellers are short volga and short correlation gamma). An uncertain-`λ` overlay would turn
  this into a reserve; it is not built.
* **Sub-baskets.** `λ`'s state is the index, so every constituent must be simulated, and the
  correlation inside the subset is the choice of `R_low` and `R_high`, not the market's.
* **Existence and uniqueness** of the McKean equation are open; see §4.
* **Data.** The single-name upside wings and the long-dated index expiries are extrapolations
  or thin quotes: 12m and 24m numbers carry the study's caveat, and on 2026-10-02 the DJX
  slices beyond one year are not consistent with the shorter ones in the wings (SPEC §8.7,
  [review] LC4G-g).
* **Smaller ones.** Discrete dividends; pricing beyond the calibration horizon is refused; the
  performance basket freezes the weights of the forward-normalised names (below 1 % of a weight
  at 3m).

---

## Sources

As tagged in the owner's specification (`LOCAL_CORRELATION_SPEC.md`, §12); none was re-read
for this note. Gyöngy (1986), mimicking the one-dimensional marginals of an Itô process.
Dupire (1994). Guyon and Henry-Labordère (2012), the particle method. Guyon (2014), local
correlation families. Langnau (2010), a dynamic model for correlation. Guyon (2016),
cross-dependent volatility. Koster and Oeltz (2019), pairwise local correlation. Barthe
(2020), on the Palladium as sold. Bergomi, *Stochastic Volatility Modeling*: ch. 2 (local
volatility), ch. 5 (variance swaps and the log contract), ch. 12 (local-stochastic volatility
and the particle method).
