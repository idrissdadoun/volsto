# Cross-dependent volatility on top of the local correlation model — a prototype

Branch `cross-dependent-vol` (from the head of M12, `local-correlation`). Nothing here changes
M12: the prototype is one new module (`volsto/multi/cdv.py`), its tests
(`tests/test_cross_dependent_vol.py`) and one script (`scripts/cdv_scan.py`); it is off by
default and nothing imports it. Tags as in `docs/local_correlation.md`: **[measured]**,
**[derived]**, **[decision]**.

**State on 2026-10-09.** The branch carries M12's defaults of 9 Oct (merge d4dd888 of
`local-correlation`: calendar repair of the names' and of the index's crossing slices; a name with
no expiry passing the quote screen is kept unscreened and its date flagged). A second script,
`scripts/cdv_stratified.py`, was added, and `scripts/cdv_scan.py` also reports the attribution and
the regions of §6. Sections 1 to 5 predate the merge and are kept as a record: their scans are on
the old defaults (no calendar repair, no fallback), with `g_max = 2` on 2026-10-02 and `g_max = 3`
on the two other dates. The runs of 9 Oct (§6, §7) use the new defaults, `g_max = 3` and `β`
imposed at 0, 3 and 6; their numbers are copied from the M12 worktree's results package,
`~/Code/volsto-lc/outputs/dispersion_lc/pm_update/NUMBERS.md`, and cited by its sections ([A3],
[A8], [A9], [D.1] to [D.3], [V1], [V2]) and by its reader notes. Validation of the code
underneath, as it stands: M12's synthetic test S3 FAILS the gate decided on 9 Oct at +2.5 sd, and
the golden baseline and the slow Dow tests have not been rerun under the defaults of 9 Oct (§8).

**Budgets, `±` and flags in §6 to §8.** Budgets: production = 8·10⁵ particles and 8·10⁵ pricing
paths (the constant-correlation companion fitted on 4·10⁵ paths); development = 2·10⁵ and 2·10⁵
(companion on 10⁵). `±`: in §6, the pricing Monte Carlo standard error given the calibrated model
and the expiry screen, paired on the pricing paths for a change between two `β`. It contains
neither the calibration's own noise (each `β` is its own particle calibration), nor the budget,
nor the sensitivity to the screen [reader note 1]; for scale, between the two budgets and on the
same specification M12's LC/CC differs by 0.0005 to 0.0019 on the four dates of the package's
sections A and B. In §7's tables, the standard error of a mean across dates, the dates taken as
independent. In §3 to §5 the errors are those stated there. Two flags of the owner's decisions of
9 Oct are used and kept apart [reader note 17]: decision 5's (a calibration whose clipped mass
inside ±2.5 sd is above 1 %; "flagged" in §6) and decision 2's (a date on which a name is kept
unscreened; "date with an unscreened name" in §7).

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
name's Dupire local volatility, `β ≥ 0` one number and `g_max` a clip [decision: 2, the module's
default; 3 in the runs of 2026-10-09, §6 and §7].

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

*Scans made before 9 Oct on the old defaults (no calendar repair, no fallback), `g_max` as stated
for each date; kept as a record. The statements that §6 supersedes are marked.*

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
  −0.00009: the model's basket second moment is the listed strip's. [Superseded 2026-10-09: at
  the production budget, `g_max = 3` and the defaults of 9 Oct, the basket is still short at
  `β = 3`: basket part −0.000100 ± 0.000025 [A8], `E[R̄²]/M_B^listed` = 0.9853 ± 0.0036 (§6).]
* The names keep their second moment up to `β = 3` (+2.3 % against +3.0 % at `β = 0`, errors
  0.6 %); at `β = 5` it drifts (+4.5 %) and κ moves: the normalisation, estimated by regression
  and clipped to its theoretical range `[1/g_max, g_max]`, is no longer accurate enough there.
* **The Palladium forward moves little:** 0.9703 → 0.9662 of the constant-correlation forward
  between `β = 0` and `β = 3` (−0.43 % ± 0.09). The two bracketing numbers of the M12 rows —
  `ED_wing` 0.9587 and `ED_eqv` 0.9471 of `E^CC[D]` — assume that reaching the listed basket
  second moment leaves κ unchanged; in the cross-dependent model κ rises (0.747 → 0.754)
  because the names' volatilities rise together in the sell-off, which adds dispersion there
  while the basket's tail is fattened. On this date the forward of a model that reaches the
  wing is much nearer to M12's than to the bracketing numbers. [2026-10-09: §6 measures this at
  the production budget, `g_max = 3` and the defaults of 9 Oct. As measured κ goes from
  0.7522 ± 0.0012 to 0.7546 ± 0.0018, `d ln κ` +0.328 ± 0.262 %, not resolved; with the names'
  second moment held it rises by +0.529 ± 0.023 %. κ is one of two reasons; the other is that
  the basket's second moment is still short of the listed one at `β = 3`.]

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

**2019-09-03** (`--betas 0,3,6 --g-max 3`; the clipped mass of M12 is 13 % there, at short
maturities, and the 3m wing is already near: −0.41 vp at −2.5 sd): `β` = 0 / 3 / 6 → clipped mass
inside ±2.5 sd 13.3 / 6.9 / 3.9 %; index at −2.5 sd −0.41 / −0.14 / −0.13 vp; `E[D]/E^CC[D]`
0.9594 / 0.9632 / 0.9439 (errors 0.0007); κ 0.766 / 0.773 / 0.756; the names' second moment
over the listed strips +1.0 / +0.8 / +1.0 %. The forward is not monotone in `β` on this date:
where the index wing costs little to reach, what `β` changes is mostly how dispersion is
distributed over index levels, and the family spans about ±1 % of the forward.

Three dates, one budget, one shape of `g`: an indication, not a result. What the scans agree on:
the names keep their second moment and the basket's reaches the listed strip's as the clipped
mass falls. Where the wing is far out of M12's reach (2026-10-02, 2017-04-03) the forward falls
further than M12's, by less than the bracketing formulas; where it is near (2019-09-03) the
forward moves by about ±1 % either way. [Superseded 2026-10-09 for "the basket's reaches the
listed strip's": on 2026-10-02 and on 2017-04-03 it is still short at `β = 3` and at `β = 6` (§6).]

## 5. What this is and is not

*Written before 9 Oct; §8 is the list after the checks of 9 Oct.*

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

## 6. Check (e): why the forward moves less than the fixed-κ estimate (2026-10-09)

The owner's question (SPEC §8.7): from `β = 0` to `β = 3` the forward moved by 0.4 % (§4), not by
the 1.2 % estimated at fixed κ. Is `E[R̄²]` still short of `M_B^listed`, or is κ up?

**Definitions [derived; `cdv_scan.attribution`, `test_attribution_identities`].**
`V = Σ w R_i² − R̄²` (`R_i` the names' returns to the horizon, `R̄ = Σ w R_i`), `κ = E[D]/√E[V]`;
`E^Q[V]` and `M_i`: the study's listed dispersion variance and listed second moment of name `i`;
`M_B^listed = Σ w M_i − E^Q[V]`: the study's listed index second moment; `E⁰`, `E^β`: expectations
at `β = 0` and at `β`; `d`: the change between them on the same pricing paths; `N⁰ = Σ w E⁰[R_i²]`.

    d ln E[D] = d ln κ + ½ d ln E[V]                                  (an identity)
    wing estimate = ½ ln((N⁰ − M_B^listed) / E⁰[V])                   (M12's `ED_wing`)
    E_h[V] = N⁰ − E^β[R̄²],    κ_h = E^β[D] / √E_h[V]                  (names held)
    d ln E[D] = d ln κ_h + ½ d ln E_h[V],    ½ d ln E_h[V] = wing estimate + still short
    still short = ½ ln((N⁰ − E^β[R̄²]) / (N⁰ − M_B^listed))

The wing estimate is the fixed-κ estimate: the move of the forward if the basket's second moment
went to `M_B^listed` with κ and the names unchanged. Names held: the model keeps each name's law,
so `Σ w E[R_i²]` is set to `N⁰` under every `β`. The gap, move minus estimate, is then
`d ln κ_h + still short`: κ up, or the basket still short (`E^β[R̄²] < M_B^listed`). "As measured"
is `E[D]/√E[V]` on each `β`'s own paths. Standard errors: delta method, the two models paired.

**Two targets.** The model is calibrated to its own SVI fit of the index smile, whose strip is
above `M_B^listed` by 0.27 % on 2026-10-02, 4.34 % on 2017-04-03 and 1.49 % on 2019-09-03 [A9].
The attribution lines (wing estimate, gap, still short, share closed) and `E[R̄²]/M_B^listed` are
against `M_B^listed` and carry a `±`; for the same split against the model's own target the
package has point values only. The regions (`cdv_scan.basket_regions`) and the index errors, which
carry a `±`, are against the model's own target: the regions cut that strip, `2∫ OTM(K) dK`, by
option strike in at-the-money sd, in units of the squared level (the basket in forward moneyness;
times `F_B²` = 1.0138 on 2026-10-02 in the units of `E[R̄²]`); the index errors are the model
minus its own SVI target at the horizon. "The shortfall" is against `M_B^listed` unless the
model's own target is named.

**2026-10-02, production budget** [measured: A8, A9; `scripts/cdv_scan.py` at commit d4dd888,
`g_max = 3`]. `β = 0` is M12: its `E[D]` equals that of M12's risk run of the day to 1e-12 [A8].
It is another pricing run than the production row of section A, the row the package says to quote
for today: same specification and seeds; in the code the row's pricing pass observes the basket at
the index target's slices before the horizon and at the horizon, the risk run and the scan at the
horizon alone (read in the code, not confirmed by a rerun). The row prints κ 0.7526 ± 0.0009,
`E[R̄²]/M_B^listed` 0.9314 ± 0.0029 and an index error of −1.491 ± 0.037 vp at −2.5 sd; `E[D]` is
0.106201 ± 0.000034 in the row and 0.106176 ± 0.000034 here, 0.7 of one standard error apart (no
error of the difference is computed) [A3; A8; A, "Consistency of the inputs"; A9; reader note 2].

| quantity | β = 0 | β = 3 | β = 6 |
|---|---|---|---|
| clipped mass inside ±2.5 sd at the cap, % of particles (0.000 at `λ = 0` for the three); flagged (decision 5's rule) when above 1 % | 1.898, flagged | 0.000, not flagged | 2.192, flagged |
| index error against the model's own target at −2.5 / −3.0 / −3.5 sd (vp) | −1.519 ± 0.032 / −2.223 ± 0.032 / −2.972 ± 0.032 | −0.301 ± 0.044 / −0.398 ± 0.048 / −0.490 ± 0.052 | −0.230 ± 0.044 / −0.315 ± 0.048 / −0.405 ± 0.052 |
| index error against the model's own target at +2.5 sd (vp) | +0.117 ± 0.053 | +0.071 ± 0.050 | −0.635 ± 0.057 |
| `E[R̄²]/M_B^listed` | 0.9286 ± 0.0029 | 0.9853 ± 0.0036 | 0.9859 ± 0.0038 |
| `E[V]/E^Q[V]` | 1.0484 ± 0.0034 | 1.0321 ± 0.0050 | 1.0868 ± 0.0452 |
| κ as measured | 0.7522 ± 0.0012 | 0.7546 ± 0.0018 | 0.7288 ± 0.0151 |
| κ, names held | 0.7522 ± 0.0012 | 0.7562 ± 0.0013 | 0.7494 ± 0.0013 |
| move of the forward, `d ln E[D]` | — | −0.453 ± 0.009 % | −1.354 ± 0.017 % |
| wing estimate against `M_B^listed` | — | −1.240 ± 0.052 % | −1.240 ± 0.052 % |
| gap: move minus estimate | — | +0.786 ± 0.055 % | −0.114 ± 0.058 % |
| of which κ, names held | — | +0.529 ± 0.023 % | −0.362 ± 0.043 % |
| of which the basket still short of `M_B^listed` | — | +0.258 ± 0.064 % | +0.248 ± 0.067 % |
| share of the shortfall closed, `(E^β[R̄²] − E⁰[R̄²])/(M_B^listed − E⁰[R̄²])` | — | 79.4 ± 4.3 % | 80.2 ± 4.7 % |
| `d ln κ` as measured | — | +0.328 ± 0.262 % | −3.155 ± 2.075 % |
| against the model's own target, point values: estimate / gap / still short / share closed | — | −1.287 % / +0.833 % / +0.305 % / 76.6 % | −1.287 % / −0.067 % / +0.295 % / 77.3 % |
| model minus the model's own index target's strip (10⁻⁶), options struck below −3.5 sd / −3.5 to −2.5 / −2.5 to −1.5 / −1.5 to 0; total of the six regions | −272 ± 2 / −107 ± 2 / −87 ± 3 / −43 ± 8; −501 ± 20 | −80 ± 5 / −21 ± 3 / −20 ± 4 / −7 ± 9; −119 ± 25 | −88 ± 4 / −17 ± 3 / −15 ± 4 / −4 ± 9; −115 ± 26 |

**The `±` of the table.** The `±` of κ as measured and of `E[V]/E^Q[V]` are the pricing Monte
Carlo errors of this run and are not their uncertainty. The names' second moment is not pinned
down in the call wing: on 2026-10-02 the model's is 0.7 % to 2.2 % above the listed strips
depending on the run and the budget. Section A's production row, the same model in another
pricing run, prints κ 0.7526 ± 0.0009 and `E[V]/E^Q[V]` 1.0476 ± 0.0026 against 0.7522 ± 0.0012
and 1.0484 ± 0.0034 in the `β = 0` column here [reader note 8; A3; A8]. The `±` of the index
errors, on the three dates, are the scan's: the Monte Carlo error of the option price divided by
the Black vega at the target vol (`cdv_scan.smile_errors`), not the delta-method error of the
model's implied vol, which divides by the Black vega at the model vol. On 2026-10-02 at `β = 0` they are 14 to 32 % smaller than the
latter (± 0.032 against ± 0.037, ± 0.042 and ± 0.047 in M12's risk run at −2.5, −3.0 and
−3.5 sd); at `β = 3` and `β = 6` the size of the effect is not computed, and the package gives it
for no other date [A8, footnote to the index-error lines].

* `β = 3`: both reasons hold. With the names held, κ is 67 ± 6 % of the gap (approximate error)
  and the basket still short the rest; κ part minus still short is +0.271 ± 0.078 %, 3.5 standard
  errors: κ is the larger part (63 % of the gap against the model's own target, a point value).
  The owner's "about 0.758" is the value of κ if the whole gap were κ (0.7581).
* Not resolved at `β = 3`: κ as measured, whose `d ln κ` is within two standard errors of zero.
  That κ rises rests on holding the names' second moment; measured, that moment is not pinned
  down in the call wing (the package's reader note 8).
* `β = 6` is a flagged calibration and loses the upside wing. The forward moves further with no
  more of the shortfall closed, and the gap is within two standard errors of zero. κ and `E[V]`
  as measured are not resolved; `E[D]` is resolved.

**The two other dates, development budget** [measured: A9; `g_max = 3`; every calibration is
flagged by decision 5's rule (clipped mass above 1 %), `β = 0` included]. Pairs of values are
`β = 3` / `β = 6`.

* 2017-04-03: move −1.042 ± 0.013 % / −1.993 ± 0.022 %, estimate −4.533 ± 0.250 %. Both reasons
  hold: κ with the names held +1.364 ± 0.053 % / +1.654 ± 0.070 %, basket still short
  +2.128 ± 0.298 % / +0.886 ± 0.314 % (54.2 ± 3.9 % / 81.2 ± 5.7 % of the shortfall closed). The
  two parts are of the same order: κ part minus still short is −0.764 ± 0.342 % (2.2 standard
  errors) / +0.768 ± 0.373 % (2.1). Against the model's own target the basket still short is the
  larger part at both `β` (point values +4.100 % / +2.858 %): the ranking depends on `β` and on
  the target. κ as measured is resolved (`d ln κ` +1.361 ± 0.010 % / +1.636 ± 0.021 %).
* 2019-09-03: no shortfall against `M_B^listed` (`E[R̄²]/M_B^listed` = 1.0097 ± 0.0059 at
  `β = 0`; estimate +0.478 ± 0.286 %, within two standard errors of zero). The move,
  +0.706 ± 0.025 % / −1.285 ± 0.042 %, is κ with the names held, +0.955 ± 0.045 % /
  −1.056 ± 0.058 %, plus `½ d ln E_h[V]`, −0.249 ± 0.036 % / −0.229 ± 0.042 %. Against the model's
  own index target `E[R̄²]` is short by 43 ± 49 (10⁻⁶) at `β = 0`, within two standard errors of
  zero, and the downside wing is short at `β = 0`: model minus that target's strip for options
  struck below −3.5 sd / −3.5 to −2.5 / −2.5 to −1.5 sd, −26 ± 6 / −18 ± 5 / −21 ± 9 (10⁻⁶); index
  errors −0.263 ± 0.086, −0.340 ± 0.096, −0.421 ± 0.107 vp at −2.5, −3.0, −3.5 sd; 11.4 % of the
  particles clipped at the cap inside ±2.5 sd. At `β = 3` the same are −7 ± 7 / −8 ± 5 / −11 ± 9;
  −0.121 ± 0.089, −0.149 ± 0.099, −0.163 ± 0.112 vp; 2.4 %. κ as measured is resolved: `d ln κ`
  +1.064 ± 0.013 % / −1.105 ± 0.025 %. It differs from the names-held value by the measured drift
  of the names' second moment (zero in the model, and set to zero with the names held), which at
  `β = 3` changes `½ d ln E[V]` by −0.109 ± 0.049 %, more than two standard errors from zero
  (`β = 6`: +0.049 ± 0.070 %).

## 7. Check (f): the stratified test (2026-10-09)

The owner's question (SPEC §8.7): on the dates where M12 binds, does closing the wing take its
discount from about 3 % towards model S's 6 %?

**Design [D].** The selection table is the 3m development pass of M12's history at commit 5b4700b
(the names' calendar repair and the fallback on, the index repair not yet in the code). Among its
priced rows: the 20 dates with the largest clipped mass (high, "the binding dates"; the 20th is
0.2277) and the 20 with the smallest (low; at most 0.0145, 15 of them under 1 %). Clipped mass:
the larger of the two one-sided masses of particles (`λ` clipped at 0; at its cap), each at its
worst calibration slice, inside ±2.5 sd, here as a fraction. Run: `scripts/cdv_stratified.py
--fixed 3,6 --g-max 3` at commit d4dd888, development budget, 40 of 40 dates priced. `--fixed`
imposes the two `β` on every date; the script's other mode, a search for the smallest `β` of a
grid with a clipped mass of 1 % or less, has no run in the package. The request (f) as recorded in
SPEC §8.7 asked for that search (on each date the smallest `β` of the grid with a clipped mass of
1 % or less inside ±2.5 sd and twice that `β`; if none reaches 1 %, "the best, flagged"); the run
of the package answers another design, 3 and 6 on every date with no search. Per date CC (`λ`
constant), LC (the prototype at `β = 0`), `β = 3` and `β = 6` are priced on the same paths; κ is
as measured.

Means across a group's dates ± the standard error of the mean, the dates taken as independent
[D.1, D.1b]. Cells with three values are LC / `β = 3` / `β = 6`. Copula: the dispersion study's
model; model S: its skewed model; listed-variance forward: `√(E^Q[V]/E^cop[V])`, the copula's κ
times `√E^Q[V]` over the copula's forward. Date with an unscreened name (decision 2's flag): a
name is kept unscreened on it (2008-11-03 and 2009-04-06, both low). This is not §6's flag
(decision 5's rule, clipped mass above 1 %): in §6's sense every date of the high group is
flagged, under LC, at `β = 3` and at `β = 6` [D.3]. The index error is against the model's own
index target; `E[R̄²]/M_B^listed` is against the study's listed moment (§6, two targets).

| quantity | high (n 20) | low, all dates (n 20) | low, without the 2 dates with an unscreened name (n 18) |
|---|---|---|---|
| clipped mass | 0.2684 ± 0.0151 / 0.2336 ± 0.0193 / 0.2280 ± 0.0290 | 0.0052 ± 0.0011 / 0.0234 ± 0.0120 / 0.1459 ± 0.0403 | 0.0058 ± 0.0012 / 0.0119 ± 0.0055 / 0.1001 ± 0.0281 |
| index error at −2.5 sd, against the model's own target (vp) | −2.02 ± 0.33 / −0.76 ± 0.17 / −0.57 ± 0.14 | −0.30 ± 0.10 / −0.16 ± 0.05 / −0.11 ± 0.04 | −0.33 ± 0.11 / −0.17 ± 0.05 / −0.13 ± 0.05 |
| `E[R̄²]/M_B^listed` | 0.9650 ± 0.0089 / 1.0000 ± 0.0070 / 1.0043 ± 0.0067 | 0.9669 ± 0.0234 / 0.9740 ± 0.0237 / 0.9669 ± 0.0246 | 0.9637 ± 0.0260 / 0.9722 ± 0.0263 / 0.9708 ± 0.0270 |
| `E[V]/E^Q[V]` | 1.1087 ± 0.0221 / 1.0440 ± 0.0129 / 1.0469 ± 0.0133 | 1.0657 ± 0.0226 / 1.0656 ± 0.0229 / 1.1117 ± 0.0329 | 1.0702 ± 0.0250 / 1.0686 ± 0.0255 / 1.1097 ± 0.0364 |
| forward over the copula's | 0.9648 ± 0.0069 / 0.9500 ± 0.0065 / 0.9326 ± 0.0064 | 0.9887 ± 0.0091 / 0.9803 ± 0.0088 / 0.9520 ± 0.0094 | 0.9879 ± 0.0101 / 0.9828 ± 0.0096 / 0.9539 ± 0.0099 |
| model S over the copula; listed-variance forward | 0.9064 ± 0.0081; 0.9358 ± 0.0100 | 0.9767 ± 0.0042; 0.9761 ± 0.0039 | 0.9752 ± 0.0045; 0.9743 ± 0.0042 |
| forward over CC's | 0.9754 ± 0.0054 / 0.9605 ± 0.0047 / 0.9428 ± 0.0044 | 0.9874 ± 0.0039 / 0.9790 ± 0.0033 / 0.9506 ± 0.0039 | 0.9866 ± 0.0043 / 0.9814 ± 0.0031 / 0.9525 ± 0.0032 |
| per date, forward over CC's at `β` minus LC's: `β = 3`; `β = 6` | −0.0149 ± 0.0027; −0.0326 ± 0.0044 | −0.0084 ± 0.0029; −0.0368 ± 0.0061 | −0.0051 ± 0.0020; −0.0341 ± 0.0058 |
| LC to `β = 3`, per date, in %: `d ln E[D]`; `d ln κ`; `½ d ln E[V]` | −1.53 ± 0.28; +1.37 ± 0.46; −2.91 ± 0.44 | −0.85 ± 0.30; −0.84 ± 0.40; −0.01 ± 0.14 | −0.51 ± 0.20; −0.43 ± 0.31; −0.08 ± 0.14 |
| LC to `β = 6`, the same | −3.39 ± 0.45; −0.62 ± 0.59; −2.77 ± 0.56 | −3.80 ± 0.64; −5.73 ± 0.97; +1.93 ± 0.68 | −3.51 ± 0.60; −5.13 ± 0.97; +1.62 ± 0.69 |

**Level, on the binding dates [D.2].** `β = 3` and `β = 6` take M12's discount to the copula from
3.5 % to 5.0 % and 6.7 %, against model S's 9.4 %: about a quarter and about a half of the
distance (25 % and 55 % on the group means; 24 % and 52 % on the per-date medians; no standard
error computed for these fractions). At `β = 6` the forward over the copula's is still
0.0262 ± 0.0062 above model S's. The clipped mass goes under 1 % on none of the 20 dates at
either `β` (smallest values 0.047 and 0.011). On the group mean `E[R̄²]/M_B^listed` goes from
0.9650 ± 0.0089 to 1.0000 ± 0.0070 and 1.0043 ± 0.0067 (per date at `β = 3` it runs from 0.916 to
1.055 [D.3]); this is against the study's listed moment, not the model's own index target (§6,
two targets). The index error at −2.5 sd, which is against the model's own target, goes from
−2.02 ± 0.33 to −0.76 ± 0.17 and −0.57 ± 0.14 vp (table).

**Against the least-clipped dates [D.2].** The same `β` moves the forward there too. High minus
low of the per-date difference: at `β = 3`, −0.0065 ± 0.0040 on all dates (1.6 standard errors)
and −0.0098 ± 0.0034 without the 2 dates with an unscreened name (2.9); at `β = 6`,
+0.0043 ± 0.0075 (0.6) and +0.0015 ± 0.0073 (0.2). **Channels [D.1b].** On the binding dates the
fall comes through `E[V]`, with κ up at `β = 3` and 1.0 standard errors from unchanged at `β = 6`.
On the low group the fall goes through κ at `β = 6` (`d ln κ` −5.73 ± 0.97 %, with `½ d ln E[V]`
+1.93 ± 0.68 %). At `β = 3` `d ln κ` is −0.84 ± 0.40 % on all dates and −0.43 ± 0.31 % without the
2 dates with an unscreened name, the latter within two standard errors of zero. κ is as measured
and is not resolved on 8 of the 20 low dates (caveat iv). There the prototype itself clips `λ`
(table, line 1).

**Caveats on the groups [D.1 to D.3].** (i) Selection table against this run, which recomputes
M12 on the defaults of 9 Oct: on 3 high dates M12's clipped mass is then below the table's 20th
largest (2015-11-02 0.320 → 0.092, 2016-10-03 0.279 → 0.216, 2023-09-05 0.259 → 0.192); without
them (n = 17) the per-date differences are −0.0171 ± 0.0028 (`β = 3`) and −0.0370 ± 0.0042
(`β = 6`). 1 high date has status `check` in the selection table (2012-06-04: the gate
`check_no_nan` fails). (ii) The low group is not a group where nothing binds: 5 of its dates are
above the 1 % line, 2 fail the index gate (0.15 vol points at the money and at 90 % of the
forward) in the selection table (2024-02-05, 2024-05-06), and on 3 the model's index second moment
is more than 15 % from the listed one (ratios to the copula not like for like). (iii) 7 of the 20
high dates are in 2020, 5 of the low dates in 2008 and 5 in 2024; clustered by calendar year the
standard error of high minus low is 0.0038 at `β = 3` and 0.0106 at `β = 6`, and the high group's
listed-variance forward has a clustered standard error of 0.0155 against 0.0100. (iv) κ and
`E[V]/E^Q[V]` are not resolved on 11 of the 40 dates (3 high, 8 low: a κ with a Monte Carlo error
of 0.005 or more); the forward ratios are resolved (largest Monte Carlo error 0.0011).

**Conclusion, as far as D.2 goes.** In level, on the binding dates, `β = 3` and `β = 6` move M12's
discount towards model S's. At `β = 3` the move on the binding dates is through `E[V]`. Its
difference from the move on the other dates is −0.0065 ± 0.0040 (1.6 standard errors: not
resolved) on all dates and −0.0098 ± 0.0034 (2.9) without the 2 dates with an unscreened name. At
`β = 6` the total move is the same in both groups to within one standard error, through different
channels. `β` was imposed and the clipped mass goes under 1 % on no binding date: the test gives
the direction of the level, not its size, which needs `β` calibrated date by date.

## 8. What the prototype does not do; next steps (2026-10-09)

* **`β` is imposed, not calibrated**, and `g` is one exponential with a clip. Neither check says
  what a `β` calibrated date by date to the index wing would give: that calibration is the next
  step. Where the two targets of §6 differ, it has to state which one it uses.
* **The upside wing at large `β`.** At `β = 6` the index error at +2.5 sd is −0.635 ± 0.057 vp on
  2026-10-02 (production budget) and −1.249 ± 0.056 on 2019-09-03 (+0.143 ± 0.108 on 2017-04-03;
  both at the development budget) [A9]. A `g` of another shape above the forward is not built.
* **The names' second moment under `g_max = 3`.** On 2026-10-02 at `β = 6` (production budget),
  `Σ w E[R_i²]` is 6.01 ± 3.36 % above the listed strips against 1.67 ± 0.28 % at `β = 0` [A9] (a
  pricing error of one run: reader note 8): the paths neither show nor exclude that the names keep
  it. On 2019-09-03 at `β = 3` (development budget) the measured drift of the names' second moment
  changes `½ d ln E[V]` by −0.109 ± 0.049 %, more than two standard errors from zero [A9]. The
  tests of §3 are at `β = 3` with the module's default `g_max = 2`; the names' smiles are not
  tested at `g_max = 3` or at `β = 6`, and "names held" in §6 assumes what such a test would check.
* **Validation of the code underneath [V1, V2].** M12's synthetic test S3 FAILS the gate decided
  on 9 Oct (the index smile under the calibrated `λ` within 0.05 vp of the smile under `λ ≡ 1` on
  common paths, at 1m, 2m, 3m and 11 strikes from −2.5 to +2.5 sd) on 3 of 33 cells, all at
  +2.5 sd: −0.0923 ± 0.0145 vp at 1m, −0.0589 ± 0.0146 at 2m, −0.0751 ± 0.0164 at 3m; every cell
  from −2.5 to +2.0 sd passes. The strike is above the range on which the particle estimate of `λ`
  is trusted (the 0.5 % to 99.5 % quantiles of the particle cloud; its upper end is at +2.06 to
  +2.08 sd at the three pillars), where the default tail rule applies; it is the only tail rule
  the prototype implements (`lambda_tail = "regressions"`). What this implies for the Dow numbers
  has not been measured; §6 and the bullet on the upside wing quote index errors at +2.5 sd. The
  golden baseline S11 and the slow Dow tests have not been rerun under the defaults of 9 Oct.
* **Not done:** the production budget on dates other than 2026-10-02; and, as in §5, a cache, the
  products layer, blocks of steps in the kernel, risk, `g` depending on time, the second pass.
