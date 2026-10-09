# Cross-dependent volatility on top of the local correlation model — a prototype

Branch `cross-dependent-vol` (from the head of M12, `local-correlation`). Nothing here changes
M12: the prototype is one new module (`volsto/multi/cdv.py`), its tests
(`tests/test_cross_dependent_vol.py`) and one script (`scripts/cdv_scan.py`); it is off by
default and nothing imports it. Tags as in `docs/local_correlation.md`: **[measured]**,
**[derived]**, **[decision]**.

## 1. Why

M12's finding on the Dow (SPEC §8.7, owner's decision 7 of the second round): local
correlation on own-level local volatilities cannot generate the DJX downside skew. With
`σ_i = σ_i(t, S_i)` the basket's variance in a sell-off can rise through the names' own skews
and through the correlation, and the correlation is capped at 1. On 2026-10-02 the cap binds
on 1.8 % of the mass inside ±2.5 sd and the model's index vol is 1.5 vp low at −2.5 sd at 3m;
on 2017-04-03 it binds on 15 %.

What the index smile says in those states is that the names' volatilities are higher *because
the index is lower*, whatever each name did. Guyon's cross-dependent volatility (Risk, 2016)
puts that into the model: a name's volatility depends on the index level as well as on its own.

## 2. The model

    σ_i(t, S_i, B) = σ_Dup,i(t, k_i) · g(k_B) · s_i(t, k_i),
    g(k) = clip(e^{−β k}, 1/g_max, g_max),        s_i(t, k) = 1 / √E[g(k_B(t))² | k_i(t) = k],

with `k_i` and `k_B` the forward log-moneyness of the name and of the basket, `σ_Dup,i` the
name's Dupire local volatility, `β ≥ 0` one number and `g_max` a clip [decision: 2].

**Each name keeps its smile [derived].** By Gyöngy's theorem name `i` has the marginals of its
Dupire diffusion if and only if `E[σ_i² | S_i] = σ_Dup,i²`, that is
`s_i² · E[g² | k_i] = 1`. The normalisation is a conditional expectation per name and per
time, estimated on the particle cloud — exactly the leverage function of a local-stochastic
volatility model, with `g(k_B)²` in the place of the stochastic variance (Bergomi ch. 12;
SPEC §4.1).

**The index.** With `m_i = g(k_B)·s_i(t, k_i)` and `u_i = ω_i σ_Dup,i m_i`, the basket's
variance is still affine in the local correlation, `v_B = a + λ b`, and `λ(t, k)` follows from
the formula of M12 with these `a` and `b`. The calibration is M12's particle loop with thirty
more regressions per step.

**What β does [derived].** For small `β` and names that are jointly lognormal-like,
`E[g² | k_i] ≈ exp(−2β·E[k_B | k_i])`: the normalisation removes from each name the part of
`g` that its own level explains, and what is left, `g·s_i`, is the index move *not* explained
by the name. In a sell-off of the index every name's volatility is raised by that residual,
so the basket's variance rises at a given correlation, and less correlation is asked for where
M12 hits its cap.

**Discretisation [decision].** `λ` and `m_i` are frozen over a step at their start-of-step
values, like `λ` in M12 and the leverage in the LSV; `m_i²` is passed to the library's shared
step as the variance factor where the local correlation kernel passes 1.

**β = 0 is M12.** `g ≡ 1`, `s_i ≡ 1`, `m_i² = 1.0` exactly, and every statement of the kernel
and of the calibration loop is M12's: the calibrated `λ`, the raw `λ*`, the clipped masses and
the final particle cloud are equal to `calibrate_local_correlation`'s **bit for bit**
[measured: `test_beta_zero_is_the_local_correlation_model`, W5 at 3m, 5·10⁴ particles].

## 3. Tests on W5 (five names with skews, 3m; 5·10⁴ particles, 10⁵ pricing paths)

| test | what | measured |
|---|---|---|
| `test_beta_zero_is_the_local_correlation_model` | `β = 0` against M12 | bit for bit |
| `test_normalisation_of_a_constant_is_one` | the regression of a constant | 1e-12 (1e-10 for a constant other than 1: the tail extrapolation) |
| `test_names_keep_their_smiles` | C2 at `β = 3`: every name's 3m vanillas at −1, 0, +1 sd | worst 0.40 vp against the analytic smile (gate 0.25 vp + 4 se), worst 0.13 vp against the same name under `β = 0` on the same normals (gate 0.15 vp + 4 paired se) |
| `test_index_smile_and_correlation_at_beta_3` | the basket reprices the index target at `β = 3`; `λ` on the downside | inside 0.30 vp + 4 se at ±1.5 sd; `λ` at −1 sd at 3m: 0.574 against 0.746 at `β = 0` |
| `test_synthetic_truth_round_trip` | a truth with `β = 3` and constant `λ = 0.5`, recovered | `λ` within 0.010 of 0.5 inside ±1.5 sd from 1m on, no clipped mass; with `β = 0` the same target needs `λ(−1.5 sd) − λ(+1.5 sd) = +0.61` and clips on 2.6 % of the mass inside ±2.5 sd |

The last line is the Dow's situation in miniature: a world whose index skew comes from
cross-dependence looks, to a model with own-level volatilities, like a correlation that must
rise steeply in sell-offs and runs into its cap.

The names' smiles at `β = 3` are within 0.13 vp of M12's, not within its Monte Carlo error:
the normalisation is estimated by regression and frozen over a step, as the leverage is. This
is the prototype's main numerical cost and would need the LSV's care (bandwidth, a second
pass, the step in the first weeks) before any use.

## 4. The Dow on 2026-10-02 at 3m: a scan of β

`scripts/cdv_scan.py --date 2026-10-02 --betas 0,1,2,3,5` (the specification of
`scripts/lcm_price.py` at the development budget: 2·10⁵ particles and paths; thirty names;
`g_max = 2`).

| β | clipped mass inside ±2.5 sd | index error at 3m (vp): at the money / 90 % / −1.5 sd / −2.5 sd | `E[D]` | `E[D]/E^CC[D]` | `Σ w E[R_i²]` over the listed strips | basket part of `E[V] − E^Q[V]` | κ |
|---|---|---|---|---|---|---|---|
| 0 (M12) | 1.82 % | −0.005 / −0.420 / −0.444 / −1.442 | 0.106246 | 0.9703 | +2.98 % | −0.00046 | 0.7469 |
| 1 | 0.85 % | +0.002 / −0.238 / −0.252 / −0.805 | 0.106070 | 0.9687 | +2.28 % | −0.00028 | 0.7524 |
| 2 | 0.25 % | +0.013 / −0.119 / −0.126 / −0.401 | 0.105887 | 0.9670 | +2.38 % | −0.00013 | 0.7534 |
| 3 | 0.00 % | +0.019 / −0.051 / −0.056 / −0.242 | 0.105790 | 0.9662 | +2.33 % | −0.00009 | 0.7539 |
| 5 | 0.00 % | +0.019 / −0.055 / −0.060 / −0.266 | 0.105322 | 0.9618 | +4.45 % | −0.00011 | 0.7399 |

[measured; standard errors: 0.033 vp at the money, 0.06 to 0.07 at the 90 % strike and at −1.5 sd,
0.064 to 0.085 at −2.5 sd; 0.00007 on `E[D]`; 0.6 % on the names' second moment; `E^CC[D]` =
0.109500 is the constant-correlation companion's of the M12 sweep row at the same budget.
Calibration 21 s at `β = 0` and 90 s at `β > 0` — the thirty regressions per step — plus 15 s of
pricing, on 4 threads.]

**What the scan shows.**

* `β = 0` reproduces the M12 row: clipped mass 1.82 %, the wing 1.44 vp low at −2.5 sd,
  `E[D]` 0.106246 ± 0.000067 against the row's 0.106280 ± 0.000068 (a different pricing grid).
* The clipped mass and the wing error fall together: at `β = 3` no particle inside ±2.5 sd is
  clipped and the index smile is repriced within its Monte Carlo error at −1.5 sd (−0.06 ± 0.07)
  and to −0.24 ± 0.09 vp at −2.5 sd. The basket part of `E[V] − E^Q[V]` goes from −0.00046 to
  −0.00009: the model's basket second moment is the listed strip's.
* The names keep their second moment up to `β = 3` (+2.3 % against +3.0 % at `β = 0`, errors
  0.6 %); at `β = 5` it drifts (+4.5 %) and κ moves: the normalisation, estimated by regression
  and clipped to its theoretical range `[1/g_max, g_max]`, is no longer accurate enough there.
* **The Palladium forward moves little:** 0.9703 → 0.9662 of the constant-correlation forward
  between `β = 0` and `β = 3` (−0.43 % ± 0.09). The two bracketing numbers of the M12 rows —
  `ED_wing` 0.9587 and `ED_eqv` 0.9471 of `E^CC[D]` — assume that reaching the listed basket
  second moment leaves κ unchanged; in the cross-dependent model κ rises (0.747 → 0.754)
  because the names' volatilities rise together in the sell-off, which adds dispersion there
  while the basket's tail is fattened. On this date the forward of a model that reaches the
  wing is much nearer to M12's than to the bracketing numbers.

**2017-04-03, where the wing binds on 15 % of the mass** (`--date 2017-04-03 --betas 0,3,6,10
--g-max 3`; the same budget; `E^CC[D]` = 0.056315 from the M12 sweep row):

| β | clipped mass inside ±2.5 sd (high / low) | index error at 3m (vp): at the money / 90 % / −1.5 sd / −2.5 sd | `E[D]` | `E[D]/E^CC[D]` | `Σ w E[R_i²]` over the listed strips | basket part of `E[V] − E^Q[V]` | κ |
|---|---|---|---|---|---|---|---|
| 0 (M12) | 15.3 % / 1.1 % | −0.03 / −1.58 / −1.00 / −2.20 | 0.053211 | 0.9449 | +1.7 % | −0.00044 | 0.7604 |
| 3 | 8.7 % / 0.4 % | −0.00 / −0.88 / −0.54 / −1.24 | 0.052675 | 0.9354 | +1.6 % | −0.00022 | 0.7710 |
| 6 | 4.4 % / 0.0 % | +0.01 / −0.43 / −0.27 / −0.63 | 0.052100 | 0.9252 | +1.7 % | −0.00009 | 0.7725 |
| 10 | 9.2 % / 0.0 % | +0.02 / −0.33 / −0.19 / −0.53 | 0.051463 | 0.9139 | +1.9 % | −0.00010 | 0.7609 |

[measured; standard errors 0.02 vp at the money, 0.04 to 0.06 in the wing, 0.00003 on `E[D]`.]
Here the steeper index skew needs a stronger cross-dependence: `β = 6` takes the clipped mass
from 15.3 % to 4.4 % and the error at −2.5 sd from −2.2 to −0.6 vp, and the forward from
0.945 to 0.925 of the constant-correlation forward (M12's bracketing numbers on this date:
`ED_wing` 0.902, `ED_eqv` 0.886; the reference implementation in its own world 0.911; the
study's model S 0.876). At `β = 10` the clipped mass rises again (9.2 %, now where `g` is at
its clip and on the upside, where `g < 1` lowers the names' volatilities and more correlation is
asked for): one exponential with a clip is not the right shape for both wings.

Two dates, one budget, one shape of `g`: an indication, not a result. What the two scans agree
on: the names keep their second moment, the basket's reaches the listed strip's as the clipped
mass falls, and the forward falls further than M12's but by less than the bracketing formulas.

## 5. What this is and is not

* **It is** a demonstration, on synthetic data and on one date, that one number of
  cross-dependence moves the index's downside wing into the model's reach while each name
  keeps its smile, with M12 as the exact `β = 0` case.
* **It is not** a calibrated model: `β` is scanned, not fitted, and `g` is one exponential with
  a clip. A calibrated version would choose `g(t, k)` — for instance the smallest
  cross-dependence that brings the clipped mass under 1 % at every step — and would then need
  a rule for what the index smile no longer pins down (the split between `g` and `λ` is not
  identified by the index and the names' smiles alone: both raise the basket's variance in a
  sell-off; they differ in what they do to dispersion).
* **The Palladium.** Cross-dependence and correlation are not substitutes for a dispersion
  product: a common rise of the names' volatilities in a sell-off *raises* dispersion there,
  a rise of correlation *lowers* it. The scan shows how the forward moves with `β` at an
  unchanged index smile; the choice between them is model risk of first order for the product
  and should be reported as a range, like the `R_low` range of M12.
* **Not built:** a cache, the products layer (the simulation returns the names' log-spots),
  blocks of steps in the kernel, risk, `g` depending on time, the second pass.
