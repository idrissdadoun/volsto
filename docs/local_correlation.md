# The local correlation model

A theory note for milestone M12. The as-built record — conventions, tests, measured numbers,
review items — is SPEC §8.7; this note explains why the model is what it is and what it says
about the Palladium. Bergomi's *Stochastic Volatility Modeling* is cited by chapter. Other
sources are those of the owner's specification and were not re-read for this note; they are
named where their result is used. Updated 2026-10-09 (the owner's fourth round, SPEC §8.7).

Tags: **[measured]** a number produced by this repository (the command or file is given),
**[derived]** an identity or an estimate obtained on paper here, **[decision]** a convention.

Numbers on market data come from the dispersion study's Dow basket (30 names, price weights, DJX
as the index) on four dates: 2026-10-02 ("today"), 2019-09-03, 2017-04-03 (steep index skew, low
vols) and 2008-07-07. Unless a line says otherwise they are 3-month numbers from
`scripts/lcm_price.py` at the production budget (8·10⁵ particles and paths; development budget:
2·10⁵), with Monte Carlo standard errors on antithetic pair means. Numbers of the fourth round
(§8) are copied from the results package `outputs/dispersion_lc/pm_update/` (`NUMBERS.md`, cited
by its sections A, B, C, D, V; `1y/NUMBERS_1Y.md`) or from SPEC §8.7. They are not all at one
commit: the rows of the four dates are at the **defaults of 2026-10-09** (commit d4faa74);
today's risk run is at 0155bac (same library); the history and the report's arithmetic are at
5b4700b, before decision 5 of §8 was in the code; the cross-dependent prototype is on branch
`cross-dependent-vol` at d4dd888, with the defaults of 2026-10-09 merged in. Numbers at the
**old defaults** (commit c69ffab: no calendar repair, no fallback) are labelled.

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
step. At a calibration slice `t_j` (a date of the calibration's time grid, §4) the mass clipped
on one side, at 0 or at the cap, is the share of all the particles that are inside ±2.5 sd of
the basket and whose `λ` is clipped on that side. An sd is an at-the-money standard deviation:
inside means `|k_B| ≤ 2.5·σ_ATM,B(t_j)·√t_j`, with `σ_ATM,B(t_j)` the index at-the-money vol at
`t_j`. The denominator is the whole cloud, not the particles inside ±2.5 sd. Each side is taken
at its worst slice. The **clipped mass** of a calibration is the larger of the two sides, not
their sum. It is the measure of how far the index smile is outside the model's reach (§9).

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
  [measured, old defaults: 450 cells on the thirty Dow names, largest |z| 3.60, test C2].
* *The per-name weak order 2 scheme is the multi-dimensional one* for a correlation that is
  constant over the step.

What is neglected is the change of the mixing inside the step, a first-order weak error in the
correlation — the analogue of the frozen leverage (SPEC §3.1). It is largest in the first
weeks, where the cloud is narrow; the schedule therefore uses quarter-day steps over the first
two weeks and daily steps after [decision of the owner; measured, old defaults: halving every
step moves the Dow forward by −0.008 % ± 0.004 and the index smile by at most 0.045 vol points
(vp)].

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

**The identical-names test S3: the gate FAILS at +2.5 sd** [measured, production size; package
V1]. With identical names the calibration should return `λ = 1`. The gate [decision 4 of the
owner, 2026-10-09] is |index smile under the calibrated `λ̂` − index smile under `λ ≡ 1`| ≤ 0.05
vp on common paths, at 1m, 2m, 3m and 11 strikes from −2.5 to +2.5 sd. 3 of the 33 cells are
over, all at +2.5 sd: −0.0923 ± 0.0145 vp at 1m, −0.0589 ± 0.0146 at 2m, −0.0751 ± 0.0164 at 3m;
every cell from −2.5 to +2.0 sd passes (largest 0.0255 ± 0.0054). The same three cells are over
on each of four pricing seeds. The gate is not changed. Reported, not gated: +2.5 sd is above
the range on which `λ̂` is trusted (its upper end, the 99.5 % quantile of the particle cloud, is
at +2.06 to +2.08 sd at the three maturities; above it the default tail rule gives `λ` about
0.98 at +2.5 sd instead of 1). With `λ` set to 1 above the trusted range no cell is over the
gate (largest 0.0101 ± 0.0013 vp; a scratch diagnostic). The test's two floors on the smallest
`λ̂` of the trusted range pass (0.9889 against 0.985 over the quarter-day steps, 0.9986 against
0.99 after). What the failure costs on the Dow is not measured. The error against the analytic
target is reported apart: on common random numbers it is at most 0.069 vp at Δt and 0.076 at
Δt/2 inside ±1.5 sd (0.156 and 0.150 inside ±2.5 sd), and the smile moves by at most
0.0136 ± 0.0049 vp: the Δt halving showed that it is not a time-step error but the single name's
own repricing error.

## 5. Why constant correlation overprices the Palladium

The Palladium forward pays the dispersion `D = Σ_i w_i |R_i − R̄|`, with `R_i` the names'
returns and `R̄ = Σ w_i R_i` the basket's. Its companion in variance is

    V = Σ_i w_i (R_i − R̄)² = Σ_i w_i R_i² − R̄².

The comparison is between the calibrated model (LC) and its constant-correlation companion
(CC): the same local volatilities with the one constant correlation that reprices the index
at-the-money straddle at the horizon. Points 2–5 and the table are at the old defaults.

1. **The first term is the same.** `Σ w_i E[R_i²]` is fixed by the single-name smiles, which
   both models reprice (§4).
2. **The second term is not.** A model that reprices the index skew has a fatter left tail for
   the basket than a constant-correlation model marked at the money, so `E[R̄²]` is higher
   [measured: +18 %, +12 %, +21 %, +2 % on the four dates, LC over CC].
3. **So `E[V]` is lower** [measured, `E^LC[V]/E^CC[V]`: 0.953, 0.908, 0.893, 0.985].
4. **`κ = E[D]/√E[V]` hardly moves** [measured, LC over CC: 0.994, 1.006, 1.000, 0.988].
5. **So the forward falls by about the square root of the fall of `E[V]`** [measured,
   `E^LC[D]/E^CC[D]`: 0.97008 ± 0.00011, 0.95816 ± 0.00010, 0.94535 ± 0.00013,
   0.98075 ± 0.00007].

| | today | 2019-09-03 | 2017-04-03 | 2008-07-07 |
|---|---|---|---|---|
| `E^LC[R̄²] / E^CC[R̄²]` | 1.18 | 1.12 | 1.21 | 1.02 |
| `E^LC[V] / E^CC[V]` | 0.953 | 0.908 | 0.893 | 0.985 |
| its square root | 0.976 | 0.953 | 0.945 | 0.992 |
| `κ_LC / κ_CC` | 0.994 | 1.006 | 1.000 | 0.988 |
| `E^LC[D] / E^CC[D]` | 0.970 | 0.958 | 0.945 | 0.981 |
| the same at the defaults of 2026-10-09 (§8) | 0.970 | 0.954 | 0.940 | 0.981 |
| the study's model S over its copula | none: not converged (0.973 printed for information) | 0.944 | 0.876 | 0.980 |
| the reference implementation (its own world) | 0.974 | 0.945 | 0.911 | 0.976 |

(The `E[R̄²]` ratio uses `Σ w E[R_i²]` of the LC run for both models, by point 1. Model S is
the study's skewed one-factor model, `outputs/dispersion/model_s_3m.parquet`; the reference
implementation has no carry and a two-parameter `λ`.)

Path by path `D ≤ √V` (Cauchy–Schwarz with the weights), and with Jensen `E[D] ≤ √E[V]`: the
cap is model-free once `E[V]` is known [derived].

**The constant-correlation companion against the study's copula.** `E^CC[D]` over the study's
copula forward `P_D` is 0.9992, 0.9937, 1.0029, 1.0022 on the four dates at the old defaults
[measured] — two implementations that share only the data; §8 has the ratio at the defaults of
2026-10-09. The earlier sentence, that LC/CC is "to that accuracy" the correction applied to the
study's price, is superseded (2026-10-09, check (b)): LC/copula is reported beside LC/CC (§8).

**Where the model stops short.** On the Dow the calibrated model does not reach the index's
downside wing (§9): today and on 2017-04-03 its `E[R̄²]` is below the listed index strip
`M_B^listed` (ratios 0.9314 ± 0.0029 and 0.8941 ± 0.0028, B1). The rows report two numbers
beside `E^LC[D]`: `ED_wing = κ_LC·√(Σ w E^LC[R_i²] − M_B^listed)`, which replaces the model's
basket second moment by the listed one, and `ED_eqv = κ_LC·√EQV`, which replaces both terms by
the listed strips (`EQV`: the names' listed strips minus the index strip) [measured: the table
of §8]. They assume κ unchanged. Check (e) looked at κ today in the cross-dependent prototype at
`β = 3` (production budget, A9; the prototype and `β` are described in §8): with the names'
second moment held at its `β = 0` value κ goes from 0.7522 to 0.7562 ± 0.0013
(+0.529 ± 0.023 %); as measured it is 0.7546 ± 0.0018 (+0.328 ± 0.262 %, not resolved). The
earlier sentence, that the price of a model that fits the wing "lies between the LC number and
these brackets", is superseded (2026-10-09).

## 6. The calls: two effects of opposite sign

A call on the dispersion pays `(D − K)⁺`; what matters is in which states the dispersion is
large.

* Under CC with skewed local volatilities, the names' volatilities rise together in a sell-off
  and the correlation does not: sell-offs carry more dispersion.
* LC raises the correlation in sell-offs (where the index skew asks for it) and lowers it in
  rallies: it removes dispersion from the left and adds some on the right.

The profile `E[D/B | bucket]/E[D/B]` on the study's buckets of the basket return shows it
[measured, old defaults, lowest bucket (below −10 %), LC against CC: 1.01 against 1.18 today,
1.37 against 1.59, 1.29 against 1.58, 1.18 against 1.31; highest bucket (above +10 %): 1.08
against 1.02, 0.96 against 0.80, 1.04 against 0.91, 0.99 against 0.88].

How the two net out at a far strike depends on how steep the single-name skews are relative to
the index's. Calls at multiples of the CC forward, LC over CC [measured, production budget, B2;
in brackets, the old defaults]:

| strike | today | 2019-09-03 | 2017-04-03 | 2008-07-07 |
|---|---|---|---|---|
| 0.75 × | 0.9099 ± 0.0003 (0.910) | 0.8236 ± 0.0004 (0.838) | 0.7824 ± 0.0004 (0.800) | 0.9303 ± 0.0002 (0.930) |
| 1 × | 0.9034 ± 0.0007 (0.901) | 0.5841 ± 0.0008 (0.609) | 0.5815 ± 0.0009 (0.602) | 0.8940 ± 0.0007 (0.893) |
| 1.25 × † | 0.9562 ± 0.0024 (0.950) | 0.2635 ± 0.0013 (0.293) | 0.3402 ± 0.0019 (0.369) | 0.9375 ± 0.0028 (0.934) |
| 1.5 × † | 1.0110 ± 0.0101 (1.003 ± 0.010) | 0.0832 ± 0.0015 (0.106 ± 0.002) | 0.2073 ± 0.0045 (0.262 ± 0.005) | 0.9290 ± 0.0108 (0.921 ± 0.011) |

† Runaway paths (V3; §9). Measured at the development budget on 2026-10-02 and 2023-02-06 only,
at multiples of the copula's forward `P_D` (the table's strikes are multiples of the CC
forward): the paths on which one name ends above 3 times its spot carry 28.6 % and 61.8 % of the
LC call at 1.5 × and 7.6 % and 17.8 % at 1.25 × (CC call: 27.8 % and 21.0 % at 1.5 ×, 7.0 % and
5.8 % at 1.25 ×). Not measured on the three reference dates nor at the production budget. The
ratios at 1.5 × are not to be quoted as model results; those at 1.25 × are to be read with care.

Today the names' vols are high and their skews flat relative to the index: the calls lose less
than on the two dates with steep skews (LC over CC 0.9034 ± 0.0007 at 1 ×, 0.9562 ± 0.0024 at
1.25 ×, the latter to be read with care). No gain is shown: every ratio at 0.75 ×, 1 × and
1.25 × is below 1 on the four dates, and the only ratio above 1 is today's at 1.5 ×
(1.0110 ± 0.0101), which is not to be quoted. On the two dates with steep skews, 2019-09-03 and
2017-04-03, the call at 1 × is worth 0.5841 ± 0.0008 and 0.5815 ± 0.0009 of its
constant-correlation price and the call at 1.25 × 0.2635 ± 0.0013 and 0.3402 ± 0.0019 (to be
read with care). The far calls also depend on the single-name surfaces away from the quotes: on
the upside wings (the names' SVI second moment is 1.3 % above the study's listed strips today,
almost all of it above the last listed strike; SPEC §8.7, second follow-up, decision 5), and on
the calendar crossings of the names' weekly slices — on 2022-04-04 dropping the crossing slices
moves the calls at 1.25 and 1.5 times `P_D` † by −14 % and −33 %, and their LC/CC ratios from
0.50 to 0.46 and from 0.46 to 0.37 (SPEC §8.7, LC7 addendum; the † applies: the pair at 1.5
times is not to be quoted as a model result, the pair at 1.25 times is to be read with care).
The table is with that repair, the brackets and the profile without.

## 7. Hedging (SPEC §8.7, LC6)

**Sticky strike.** If the local volatilities are held in absolute spot and `λ` in absolute
index level, a rise of every spot moves the basket to a region of lower `λ`: correlation
falls and dispersion rises. This is the model's own delta, without recalibration, not the
desk regime in which implied vols are held by strike. The common delta of the forward, as the
percentage of its price per +1 % on every spot, splits exactly into

* homogeneity, +1 (the payoff is homogeneous of degree one in the spots at fixed moneyness);
* the single-name skew channel, `Δ^CC_ss − 1` < 0 (each name's volatility falls along its skew);
* the correlation channel, `Δ^LC_ss − Δ^CC_ss` > 0.

[measured: the table of §8 has the four dates at the defaults of 2026-10-09 (old defaults:
package B4). On the reference's world for today the same decomposition is +1.000, −0.804, +2.543
against the reference's +1.000, −0.804, +2.545, test S4.]

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
§8 has today's all-names vega both ways, index vega, two index skew vegas and model-risk range
at the defaults of 2026-10-09, from the package's risk run (`scripts/pm_today_risk.py`);
per-name vegas are not reported there.

## 8. Results of the fourth round (2026-10-09)

**Defaults and status** [decisions of the owner, 2026-10-09; defaults since commit d4faa74].
*Decision 1*, the calendar repair of the names' slices: when the total variance falls from one
slice to the next inside their central ±2 sd range, one of the two is dropped (the one that is
not a third Friday, else the shorter). *Decision 5*: the same rule on the DJX slices; it also
defines a flag for the long-dated runs: a date is **flagged for the clipped mass** when the mass
clipped on either side (at `λ = 0` or at the cap, §2) exceeds 1 % inside ±2.5 sd.
*Decision 2*: a name with no expiry passing the quote screen (the rules that admit a listed
expiry) is kept unscreened and its date is **flagged**; "flagged date" without qualifier is
decision 2's flag. *Decision 3*: the names' 2 % check (the model's `Σ w E[R_i²]` within 2 % of
the listed strips) is a reported diagnostic, not a gate: a priced row is `check` when
`check_no_nan`, `check_forward` or `check_index` fails, and `ok` otherwise. `check_index`, the
index gate, asks an index error (the model's index implied vol minus that of its target, its
own SVI fit of the DJX smile) within 0.15 vp at the money and at 90 % of the forward; it is
waived when the wing binds (clipped mass above 1 %). *Decision 4*: the S3 gate (§4).

**Today and the three reference dates** (3m, production budget; package A, B; the ± are pricing
Monte Carlo errors, §9). The listed-variance forward is `√(EQV/EV)`: the copula's κ on the
listed dispersion variance `EQV` over `P_D`; `EV` is the copula's `E[V]`. `ED_wing` and
`ED_eqv`: §5. Deltas (sticky strike, % of the price per +1 % on every spot) and channels: §7.

| | today | 2019-09-03 | 2017-04-03 | 2008-07-07 |
|---|---|---|---|---|
| `E^LC[D]`, units of notional | 0.106201 ± 0.000034 | 0.070643 ± 0.000020 | 0.052710 ± 0.000014 | 0.097847 ± 0.000025 |
| LC/CC | 0.96996 ± 0.00011 | 0.95401 ± 0.00011 | 0.93977 ± 0.00013 | 0.98079 ± 0.00007 |
| LC/copula | 0.96899 ± 0.00031 | 0.94351 ± 0.00027 | 0.93826 ± 0.00027 | 0.98304 ± 0.00025 |
| CC/copula | 0.99900 ± 0.00033 | 0.98899 ± 0.00031 | 0.99839 ± 0.00030 | 1.00229 ± 0.00027 |
| model S/copula | none: not converged | 0.94400 | 0.87608 | 0.98001 |
| listed-variance forward | 0.96468 | 0.94689 | 0.89469 | 0.99109 |
| `ED_wing` over `E^CC[D]` | 0.95846 ± 0.00046 | 0.95882 ± 0.00133 | 0.89848 ± 0.00109 | 0.98430 ± 0.00076 |
| `ED_eqv` over `E^CC[D]` | 0.94767 ± 0.00117 | 0.95561 ± 0.00030 | 0.89250 ± 0.00028 | 0.97638 ± 0.00032 |
| clipped mass, % of the particles | 1.898 | 11.488 | 14.386 | 2.065 |
| index error at 90 % of the forward, vp | −0.462 ± 0.030 | −0.078 ± 0.035 | −1.467 ± 0.025 | −0.042 ± 0.036 |
| index error at −2.5 sd, vp | −1.491 ± 0.037 | −0.239 ± 0.044 | −2.053 ± 0.028 | −0.227 ± 0.049 |
| delta of the forward, LC | +2.310 ± 0.005 | +1.600 ± 0.003 | +2.072 ± 0.005 | +1.463 ± 0.001 |
| delta of the forward, CC | +0.356 ± 0.003 | −2.361 ± 0.002 | −2.511 ± 0.002 | −0.516 ± 0.001 |
| skew channel / correlation channel | −0.644 ± 0.003 / +1.954 ± 0.004 | −3.361 ± 0.002 / +3.961 ± 0.003 | −3.511 ± 0.002 / +4.583 ± 0.005 | −1.516 ± 0.001 / +1.979 ± 0.001 |
| LC/CC, new minus old defaults (B4) | −0.00013 ± 0.00015 | −0.00415 ± 0.00015 | −0.00557 ± 0.00019 | +0.00004 ± 0.00010 |

The four rows are `ok`: no gating check fails. The index gate is waived on all four because the
wing binds on each; on 2026-10-02 and 2017-04-03 the error at 90 % of the forward is outside it
(−0.462 ± 0.030 and −1.467 ± 0.025 vp), on 2019-09-03 and 2008-07-07 inside (−0.078 ± 0.035 and
−0.042 ± 0.036 vp). No name is kept unscreened and decision 5 drops no DJX slice on these dates:
new minus old is the effect of decision 1 (its ± is an upper bound), 38.4 and 41.6 times the
printed error of LC/CC on 2019-09-03 and 2017-04-03.

**Today** (A). `E^CC[D]` = 0.109490 ± 0.000035, `P_D` = 0.109599 ± 0.000008; model S did not
converge, so today has no S/copula. Vegas (A7; today's risk run, `scripts/pm_today_risk.py`,
commit 0155bac; one-sided +1 bumps, % of that run's base price 0.106176 ± 0.000034, not the
table's 0.106201): all names +1 vp, +4.256 ± 0.005 with `λ` recalibrated to the unchanged index
smile and +3.033 ± 0.005 with `λ` held; index +1 vp, −2.268 ± 0.004; index skew, −0.249 ± 0.003
per vp at 90 % of the forward and −0.130 ± 0.005 per unit of the rotation bump of A7; the index
and skew vegas are those of a model whose cap binds. Model risk (A7; the same run and the same
base price), `R_low` and the `λ` family changed and the model recalibrated: range (a), all seven
variants, −0.150 % to +1.057 % of the price, from a calibration whose mass clipped at `λ = 0` is
above 1 % (`R_low` an equicorrelation at 0.10: 66.309 % of the particles inside ±2.5 sd) to a
two-parameter `λ` fitted to two index vols; range (b), the two particle variants whose mass
clipped at `λ = 0` is not above 1 % (both are clipped at the cap like the base), −0.143 % to
+0.003 %, the like-for-like one (the ranges have no standard error of their own).

**History** (3m, development budget; package C). Not at the defaults of 2026-10-09: commit
5b4700b, with decisions 1–2 and without decision 5. Samples: the 218 dates on which model S
converged, and today; 216 priced, 3 failed (2008-12-01, 2009-03-02, 2023-07-03), 8 flagged
(decision 2's flag); model S on 215. Of the 216 priced, under the status rule above, 205 are
`ok` and 11 `check`: 9 on `check_no_nan` for a non-finite diagnostic column, 2 FAIL the index
gate (2024-02-05 and 2024-05-06); all 216 are in the means. Mean ± standard error across dates
taken as independent (in brackets, Newey–West at 6 lags, a lower value: the error still grows
with the lag except for LC/CC; at 24 lags it is 0.0040 for LC/copula, 0.0017 for LC/CC, 0.0086
for model S/copula, 0.0082 for the listed-variance forward):

| | n | mean | median | without the flagged dates (n) |
|---|---|---|---|---|
| LC/copula | 216 | 0.9644 ± 0.0018 (0.0029) | 0.9645 | 0.9640 ± 0.0018 (208) |
| LC/CC | 216 | 0.9681 ± 0.0012 (0.0016) | 0.9657 | 0.9677 ± 0.0012 (208) |
| model S/copula | 215 | 0.9431 ± 0.0023 (0.0051) | 0.9471 | 0.9419 ± 0.0023 (207) |
| listed-variance forward | 216 | 0.9415 ± 0.0023 (0.0049) | 0.9429 | 0.9401 ± 0.0022 (208) |

By tercile of the clipped mass (low, middle, high; the 215 dates with model S: 71, 72, 72)
`LC/CC − S/copula` is +0.001 ± 0.003, +0.017 ± 0.002, +0.057 ± 0.004; S/copula is
0.9720 ± 0.0019, 0.9456 ± 0.0028, 0.9121 ± 0.0035 and LC/CC 0.9731 ± 0.0018, 0.9626 ± 0.0015,
0.9688 ± 0.0027. Check (d) regresses that difference across the 215 dates on the clipped mass (a
fraction of the particles) and on `(E^LC[R̄²] − M_B^listed)/M_B^listed`: +0.278 (HC1 standard
error 0.030) and −0.065 (HC1 t −1.0), R² 0.443 (C.7). It shows that the difference rises with
the clipped mass, through model S's discount: LC/CC has no trend (slope −0.006, HC1 0.021). It
does not show that the clipping causes the difference: it is an association across dates.

**The report's arithmetic at the LC price** (package C; development budget, the history's rows
at commit 5b4700b, not the defaults of 2026-10-09; 214 trades, 7 of them flagged). A trade is
one monthly entry date of the study at 3m with its outcome: the study's 217 monthly trades less
the three dates the LC pass did not price. The gap is the Palladium forward minus the
vega-neutral package; held: to expiry; hedged: delta-hedged daily with the study's hedge; the ±
of a gap is the Hansen–Hodrick standard error over trades (2 lags). The held gap is
−0.113 ± 0.221 % of notional at the LC price (copula −0.366 ± 0.220, model S +0.027 ± 0.224) and
the hedged gap −0.199 ± 0.113 (−0.452 ± 0.117 and −0.059 ± 0.115), which is 1.8 of its standard
errors from zero. The forward pays 1.037 [0.991, 1.083] per 1 of premium at the LC price (copula
1.001 [0.958, 1.045], model S 1.057 [1.011, 1.106]; 95 % block-bootstrap intervals, the LC one
including 1). Without the 7 flagged trades (207 trades) the held gap at the LC price is
−0.076 ± 0.229, the hedged gap −0.218 ± 0.107 (2.0 standard errors from 0; its block-bootstrap
interval excludes 0) and the forward pays 1.046 [1.001, 1.095], a lower bound within bootstrap
noise of 1 (0.998 to 1.004 over the bootstrap seeds 1 to 40). On the 214 trades neither result
supports a claim about a sign or "above 1". Price over payoff = listed-option part × model part;
the listed-option part has no model price in it. LC's model part is −0.0353 ± 0.0025 from the
copula's on the same trades (0.9981 ± 0.0105 against 1.0335 ± 0.0106; these ± are bootstrap
standard deviations over trades); the difference holds on the sub-samples, the level does not
(0.9719 to 1.0100).

**The cross-dependent prototype today** (pointer; branch `cross-dependent-vol`,
`docs/cross_dependent_vol.md` there). Each name's local volatility is multiplied by
`g = clip(e^{−β k_B})` (upper bound `g_max = 3` in these runs) and renormalised; `β = 0` is this
model; `β` is imposed, not calibrated.
Check (e), today, production budget (A8, A9): from `β = 0` to `β = 3` the clipped mass goes from
1.898 % to 0.000 % and the forward moves by −0.453 ± 0.009 %, against −1.240 ± 0.052 % for the
estimate at fixed κ. Both reasons hold. In the model the names' second moment is the same under
every `β` (each name keeps its law); the paths give it with noise. With it held at its `β = 0`
value, κ accounts for +0.529 ± 0.023 % (0.7522 → 0.7562 ± 0.0013) and the basket's second moment
still short of `M_B^listed` for +0.258 ± 0.064 % (79.4 ± 4.3 % of the shortfall closed); κ is
the larger part (difference +0.271 ± 0.078 %). As measured, the names' second moment not held,
κ is 0.7546 ± 0.0018 against 0.7522 ± 0.0012 at `β = 0` (+0.328 ± 0.262 %): its rise is not
resolved. Two targets are involved. The estimate at fixed κ, the shortfall and its share closed
are against `M_B^listed`, the study's listed index second moment. The model is calibrated to
its own index target (its SVI fit of the DJX smile), whose second moment is 0.27 % above
`M_B^listed` today. Against that target (A9; point values, no standard error) the estimate is
−1.287 %, κ's part is the same +0.529 % and the basket's is +0.305 %, with 76.6 % of the
shortfall closed.
Check (f), development budget (D; means across the group's dates ± their standard error, the
dates taken as independent): on the 20 dates with the largest clipped mass in the history's
table (commit 5b4700b) the forward over the copula's goes from 0.9648 ± 0.0069 under LC to
0.9500 ± 0.0065 at `β = 3` and 0.9326 ± 0.0064 at `β = 6`, against 0.9064 ± 0.0081 for model S
and 0.9358 ± 0.0100 for the listed-variance forward. The clipped mass is under 1 % on 0 of the
20 dates at either `β`, and the same `β` also moves the forward on the 20 least-clipped dates
(the prototype's forward over CC's minus LC/CC, per date: −0.0084 ± 0.0029 at `β = 3` and
−0.0368 ± 0.0061 at `β = 6` there, against −0.0149 ± 0.0027 and −0.0326 ± 0.0044 on the 20
most-clipped): the test gives the direction and not the size.

## 9. Limits

* **What "status ok" does not mean**: that the index smile is repriced. The index gate is waived
  when the wing binds, on 201 of the 216 priced dates of the history and on all four dates of
  §8; on the 205 `ok` dates of the history the index error at 90 % of the forward exceeds
  0.15 vp in size on 163 and 1 vp on 41 (C.0).
* **Units of the clipped mass** (defined in §2; the larger side, not the sum of the two): the
  package prints % of the particles in A and B, a fraction in C, D and on the 1y page (today
  1.898 % = 0.019).
* **The pricing Monte Carlo error is not the total error**: it leaves out the calibration's own
  noise, the budget and the screen. Same specification, development against production budget:
  LC/CC differs by 0.0005 to 0.0019 on the four dates; decision 1 moves it by −0.00415 and
  −0.00557 on two of them (§8). LC/CC is known to three decimals, not five.
* **Runaway paths carry the high-strike calls** (V3). On 2026-10-02 at the development budget
  174 of 200000 paths under LC (181 under CC) end with one name above 3 times its spot, beyond
  its listed strikes, where the local volatility is an extrapolation. Under LC they carry 0.22 %
  of `E[D]`, 1.8 % of the call struck at the copula's forward, 7.6 % at 1.25 ×, 28.6 % at 1.5 ×
  and 82.9 % at 2 × (2023-02-06: 0.22 %, 2.9 %, 17.8 %, 61.8 %, 89.1 %): the ratios at 1.5 × and
  above are not model results. Not measured on the reference dates, nor at 12m. The tails are
  unchanged, as decided.
* **The names' second moment is not pinned down in the call wing** (V4; decision 3's question).
  What puts `Σ w E[R_i²]` above the listed strips is the calls beyond the last listed strike:
  +490 ± 116 of +527 ± 137 (in 1e-6; +2.04 % of the strips) on 2026-10-02 at the development
  budget, +455 ± 135 of +478 ± 151 (+2.75 %) on the median failing date, 2023-02-06. The number
  depends on the run: on the `λ = 0` paths of 2026-10-02, +0.68 % of the strips (± 54 in 1e-6)
  on 2·10⁵ paths and +2.23 % (± 206) on 8·10⁵; on the production row, +1.68 ± 0.22 %. So
  the ± of κ, `E[V]/EQV`, `ED_wing` and `ED_eqv` is not their uncertainty.
* **12m: open points for the owner** (1y page). On 2026-10-02 the rule of decision 5 drops the
  DJX slices at 0.707y and 0.959y (SVI rms 0.26 and 0.09 vp; both cross the 1.208y slice, all
  three are third Fridays): the 1y target is an interpolation between the slices at 0.460y and
  1.208y (rms 0.17 and 1.34 vp), the opposite of [review] LC4G-g. With decisions 1, 2 and 5
  against the old defaults (development budget; masses as fractions of the particles) the mass
  clipped at `λ = 0` goes from 0.1044 to 0.0220 and the mass at the cap from 0.0178 to 0.0185.
  The two runs have different targets: the repair moves the 12m target vol at −2.5 sd (each
  run's own sd) from 26.60 % to 21.78 % (−4.81 vp), a downside skew 0.52 times that of the old
  defaults (target vol at −2.5 sd minus at the money: 5.65 vp against 10.91 vp); it is not a
  better fit to the same target. The date is still flagged for the clipped mass (decision 5's
  flag, §8: either side above 0.01 inside ±2.5 sd; not decision 2's flag). LC/CC goes from
  0.97690 ± 0.00025 to 0.99220 ± 0.00015 (0.99277 ± 0.00009 at the production budget). Of the
  20 yearly dates (development budget) 19 are flagged for the clipped mass: the summary without
  them is one date.
* **Not rerun under the defaults of 2026-10-09** (V2): the golden baseline S11, recorded on the
  old defaults, and the slow Dow tests (C2 Dow, S5 Dow, S11). S3 fails (§4).
* **The index downside wing on the Dow** (SPEC §8.7, owner's decision 7 of the second round).
  Local correlation on own-level local volatilities cannot generate today's DJX downside skew:
  the clip at `λ_max` binds on 1.898 % of the particles inside ±2.5 sd today and on 11.488 % and
  14.386 % on 2019-09-03 and 2017-04-03, and the model's index vol is 0.486 ± 0.030 vp low at
  −1.5 sd and 1.491 ± 0.037 vp low at −2.5 sd at 3m today [measured, A5, B1]. It is not a data
  inconsistency (the DJX puts are inside the comonotonic bound of the thirty marginals), not the
  names' wing fits (within 0.2 vp of their quotes on average), and not the cap (raising `ρ_max`
  from 0.98 to 0.995 moves the wing error by 0.04 vp) [the three diagnostics: measured, old
  defaults, SPEC §8.7 second round]. In a sell-off the index needs the names' volatilities to
  rise *because the index fell*, not only because each name fell: cross-dependent volatility
  (Guyon, Risk 2016), per-name stochastic volatility with correlated factors, or common jumps.
  The earlier version of this note read the old sweep (208 priced dates, old defaults; SPEC
  §8.7, LC7 results) as "what this model misses of model S's correction is the wing it does not
  reach"; that reading is superseded (2026-10-09) by check (d), §8: an association across dates,
  the cause not shown. A prototype of the first extension is on branch `cross-dependent-vol`
  (§8, last paragraph).
* **Deterministic correlation given the index.** Given `(t, k_B)` the correlation has no
  randomness of its own, so the model understates the variability of dispersion: far calls on
  dispersion are too cheap relative to stochastic-correlation models fitted to the same smiles
  (sellers are short volga and short correlation gamma). An uncertain-`λ` overlay would turn
  this into a reserve; it is not built.
* **Sub-baskets.** `λ`'s state is the index, so every constituent must be simulated, and the
  correlation inside the subset is the choice of `R_low` and `R_high`, not the market's.
* **Existence and uniqueness** of the McKean equation are open; see §4.
* **Data.** The names' weekly slices cross in calendar on most dates from 2011 on, and the
  Dupire surface is floored there: with the crossing slices dropped (the default since
  2026-10-09; before, an option, off by default) the forward falls by 0.5 % on average over the
  old sweep and the call at 1.25 times `P_D` by 10 %, the forward's LC/CC by 0.002 (SPEC §8.7,
  LC7 addendum and results). The single-name upside wings and the long-dated index expiries are
  extrapolations or thin quotes: 12m and 24m numbers carry the study's caveat, and on 2026-10-02
  the DJX slices beyond one year are not consistent with the shorter ones in the wings (SPEC
  §8.7, [review] LC4G-g; the 12m point above).
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
