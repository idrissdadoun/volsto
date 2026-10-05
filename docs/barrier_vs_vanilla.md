# Up-and-out calls against call ratios and call flies: the decision framework

*The theory behind `volsto/studies/barrier_vs_vanilla.py` (SPEC §8.4). The measured results
are in `outputs/studies/barrier_vs_vanilla/study.md`. Everything below holds for the
down-and-out put against put ratios and put flies with the signs mirrored; the up side is
written out.*

## 1. The three contracts as views on one event

Take a strike `K` at the spot, a barrier `B > K`, an expiry `T`. The three bullish, capped
structures a desk puts side by side are:

| contract | payoff at `T` | what it costs the buyer beyond its payoff |
|---|---|---|
| up-and-out call `UOC(K, B)` | `(S_T − K)⁺ · 1{S never closed above B}` | nothing beyond the knock-out: it is the cheapest; it can pay zero after being deep in the money |
| call fly `K / m / B` (`m = (K+B)/2`) | `(S_T − K)⁺ − 2 (S_T − m)⁺ + (S_T − B)⁺` | the payoff fades to zero *continuously* between `m` and `B`; nothing lost on the path |
| call ratio `1 × n` (`K`, `m`) | `(S_T − K)⁺ − n (S_T − m)⁺` | the upside is sold without a cap: a loss grows without bound above the break-even `m + (m − K)/(n − 1)` |

The cleanest way to see what the barrier removes is to insert the **European knock-out**
`EKO(K, B) = (S_T − K)⁺ 1{S_T < B}` between the vanilla structures and the barrier. It is the
same contract with the barrier observed at expiry only, hence an upper bound for both the daily
and the continuous knock-out, and it is **model-free**: with the skew-adjusted digital `D(B) =
−∂C/∂K |_B`,

    EKO(K, B) = C(K) − C(B) − (B − K) · D(B)   (call spread minus the cliff).

Three model-free quantities then frame the whole question:

* the **call spread** `C(K) − C(B)`: the European version without the cliff;
* the **cliff** `(B − K) D(B)`: what the European knock-out gives up at the barrier, a digital
  priced off the smile's slope at `B` (a steeper smile at `B` makes the digital cheaper on the
  call side, which makes the EKO dearer);
* the **fly** `K/m/B`: the call spread with the cliff replaced by a slope, i.e. the EKO's
  payoff "sanded down" from `m` to `B`; it is always worth *less* than the EKO (it pays less
  on `(m, B)`) and more than zero.

And one model-dependent quantity, the **regret value**

    R = EKO(K, B) − UOC(K, B) = E[ (S_T − K)⁺ · 1{S_T < B} · 1{touched} ]:

the value of the paths that closed above `B` at some point and came back to finish between
`K` and `B` — exactly the paths on which the barrier throws away a payoff the European
contract keeps. The knock-out's price is the European knock-out's minus the regret value; the
*regret share* `R / EKO` is the fraction of the European value the path dependence destroys.
It is the single most useful number of the framework: it is small when the barrier is far in
standard deviations and the expiry short, and it tends to one when the barrier is close or
the expiry long (the paths that end between `K` and `B` have almost always visited `B`). The
measured values on the SPX 2022-12-30 mark run from a few percent (3m, 120%) to above 90%
(1y, 90% put).

The ordering is therefore, for every model and every path,

    UOC_continuous ≤ UOC_daily ≤ EKO ≤ call spread,        fly ≤ EKO,

with the **premium-matched** alternatives defined by solving the vanilla structure that costs
the knock-out's premium: the ratio `n = (C(K) − P_UOC)/C(m)` in closed form; the fly by
shrinking its width — a fly spanning the whole `[K, B]` with its inner strike at `K` is worth
at least `EKO(K, B)` (its payoff dominates a fly whose wing is pushed to `B`), so **no fly on
the barrier's own strikes can match a knock-out's premium**; the matched fly is the symmetric
one `K / (K+w)/2 / w` whose top `w` makes the price equal, and `w` lands well inside the
barrier (measured: 108–131% of spot for barriers at 110–120% — the knock-out buys exposure up
to `B` for the price of a fly that stops at `w < B`). That is the first lesson of market
practice: *the knock-out is not a cheap fly; it is a cheap call spread with a path condition
attached*, and the fair comparison is "cap at `w` for sure" against "cap at `B` unless the
path touched it".

## 2. Intuition for the knock-out premium and what moves it

Write `UOC = EKO − R`. The three pieces respond to the market in different ways.

**The European part (spread minus cliff).** Pure vanilla risk: the call spread's value is the
probability-weighted mass between `K` and `B`, the cliff is the density at `B` times the width.
Both read off today's smile: level (ATM vol of the expiry), skew (the digital at `B` carries
`−∂σ/∂K · vega(B)`: on the call side, a smile that *rises* towards `B` raises the digital and
cuts the EKO; the SPX call wing is flat to slightly rising, the put wing steep) and curvature.
The fly's value is the density around `m` (positive curvature of the smile makes the fly
cheap, the "fly is the curvature" rule); the 1×2 ratio's value is `C(K) − 2C(m)`, long the
level and short the mid-strike, so it is the structure most exposed to the smile's slope
between `K` and `m` (on the put side the 1×2 put ratio *sells* the rich low strike: it is a
short-skew structure, which is why put ratios are the desk's classic skew-selling trade).

**The regret value `R`.** This is where the model enters. `R` is the value of touch-and-return
paths. It grows with:

* the *touch probability* `P(τ_B ≤ T)` — rises with vol and with time, falls with the distance
  `ln(B/S)/(σ√T)`;
* the *return probability* after a touch — the conditional law of `S_T` given a touch at
  `τ`, i.e. the forward smile seen from the barrier at the first-hitting time: under a flat
  Black–Scholes world it is symmetric (half the touched paths come back below `B` by reflection);
  under local vol the vol *at* the barrier drives it — on the SPX call side the local vol
  decreases with spot, so a path that reaches `B` moves more slowly from there, and comes back
  less often: local vol gives a lower `R` and a *lower* UOC than Black–Scholes? No — the measured
  LV price is *higher* than BS at the ATM vol (0.40 vs 0.29% of spot on the 6m 110%), because
  the first effect dominates: the lower vol above the spot cuts the touch probability itself;
* the *vol-of-vol and the spot-vol correlation* — under the 2F mark the knock-out is worth
  more than under local vol on the call side (0.56 vs 0.40%): a stochastic-vol world with
  negative spot-vol correlation reaches an up barrier in a *low*-vol state, where the return
  probability is low, and the forward skew stays steep (local vol flattens the forward smile).
  On the put side the ordering flips: the down barrier is reached in a high-vol state and the
  paths return more often, so the 2F down-and-out put is worth *less* than the local-vol one
  (0.15 vs 0.22%): the model risk of the two sides has opposite signs, and the market
  practitioner's rule "stochastic vol makes reverse knock-outs dearer" is true only on the
  call side.

**The daily-versus-continuous discount.** Continuous observation knocks out on intraday
touches; the Broadie–Glasserman–Kou shift `B e^{−0.5826 σ √Δt}` says a daily barrier at `B`
is a continuous barrier about `0.58 σ √(1/252)` closer — 0.9% of spot at 25% vol. Measured on
the mark, the continuous knock-out is 10–50% cheaper than the daily one, more so for the
near barriers. A vanilla structure has no such convention; a desk comparing a daily UOC with a
fly should also compare it with its continuous twin, which is the one the structuring desk
usually sells.

## 3. The framework: three metric families and one decision

The buyer's question is never "which is cheaper" — the premium-matched structures cost the
same by construction. It is: *on which paths does each pay, and what is my view of the path
distribution against the one the mark implies?* Three families of metrics feed that view.

### 3.1 Pure-vol metrics (off the surface and the mark)

| metric | read as | moves the decision towards |
|---|---|---|
| ATM vol of the expiry, implied minus realised (1m / 3m) | the touch probability the mark charges against the one the recent past delivered | implied ≫ realised: the barrier (the mark over-prices the knock-out, the buyer is paid for the path risk); implied ≪ realised: the fly |
| distance to the barrier in standard deviations `ln(B/S)/(σ√T)` | the regret share | near (< 1 sd): the fly or ratio (the barrier keeps little); far (> 1.5 sd): the barrier (regret share small, the knock-out is almost the spread) |
| 90–110 skew and the fly curvature of the expiry | the price of the digital at `B` and of the fly's mid strike | high curvature: flies dear → barrier; steep put skew: put ratios cheap → ratio on the down side |
| term slope (1y − 3m ATM) | the forward vol the barrier lives in after a touch | inverted (short vol high): the barrier's return probability is over-charged → barrier |
| model spread LV / 1F / 2F of the knock-out, and 2F − LV | the model-risk band: the knock-out's price moves with the forward skew and vol-of-vol; the vanilla's does not | a band above the premium saved by the barrier makes the barrier's cheapness unreliable → fly |
| the regret share `R / EKO` under the mark | how much of the European value the path condition removes | small: barrier; large: fly (or the European knock-out itself when it trades) |

### 3.2 Market metrics (costs and hedgeability)

* **Hedging cost.** The knock-out's delta explodes near the barrier (the digital at `B` shows
  up as a gamma that flips sign) and its vega turns negative there; the measured hedged P&L
  std under the 2F world is of the order of the premium itself for the 6m 110% call on the
  delta alone, and the put-call-symmetry static replication cuts it by about half — while
  a fly is a static package whose hedged residual is a fraction of a percent of spot. A
  desk that *holds* the structure (no delta hedge) needs no such cost; a desk that hedges
  must add the knock-out's hedging cost (transaction costs of the daily delta plus the
  residual std) to its premium. The study's hedge table gives both numbers.
* **Model-risk reserve.** The knock-out's mark moves with the P1 parameters; the desk buying
  it is paid the model band (the 1F / 2F / LV spread) only if its own view of the forward skew
  is better than the market's. The vanilla structure carries no reserve.
* **Liquidity.** Listed vanillas of the expiry make the fly and the ratio exit-able in the
  screen; the knock-out is an OTC contract whose unwind is marked by the dealer — in practice
  the main reason desks replace reverse knock-outs with flies in liquid names, and keep the
  knock-out where the fly's three legs would be wide (single names, long expiries).
* **Convention.** Daily close observation (the study's default) versus continuous; the
  barrier shift the dealer applies (the study's `barrier_shift_table` of the payoff study
  gives the reserve); strict versus touching breaches.

### 3.3 Statistical metrics (the real-world path distribution)

The knock-out and the fly differ by the paths on which they pay. The model's touch and
regret probabilities are risk-neutral; the buyer's are statistical. Two estimators, both in
`volsto.studies.history_stats`:

* **Regime-conditional history** over 1990–2026 (SPX daily closes): for each horizon and
  barrier level, the frequency of a daily-close touch, of a regret path, and the realised
  payoff of every structure per unit of spot — overall and by VIX tercile, by the sign of
  implied-minus-realised vol at entry and by the sign of the past 3m return. Overlapping
  windows: the standard errors use `n / horizon` effective samples.
* **Filtered historical simulation** at the chosen vol: the history's daily returns
  standardised by their trailing EWMA vol, rescaled to the target vol (today's realised or
  implied), resampled in horizon-length blocks, drift removed. It keeps the path *shapes* of
  history (clustering, trends, the negative daily autocorrelation that makes a 3m realised
  vol lower than the daily one) and discards the lognormal assumption; the same table with the
  historical drift kept is the bull's view.

The three statistical readings the framework uses:

* **touch gap** `P^P(touch) − P^Q(touch)`: negative when the market over-charges the
  knock-out (the measured unconditional history touches the 110% barrier in 3m about half as
  often as the 2F mark implies; the demeaned FHS at the implied vol touches about as often);
* **regret gap** `P^P(regret) − P^Q(regret)`: the same for the paths that matter;
* the **expected payoff per unit premium** of each structure under the statistical layer,
  `E^P[payoff]/premium` — the fair value is 1 under the mark's own measure; a structure above
  1 is one the buyer's view favours.

### 3.4 The decision

For a buyer at premium `P` (the knock-out's), with `A` the premium-matched vanilla alternative:

    edge = E^P[UOC payoff]/P − E^P[A payoff]/P

Trade the **knock-out** when `edge` exceeds the model-risk band (the LV/1F/2F spread over the
premium) plus the hedging-cost share; trade the **fly** (defined risk) when `edge < 0`; and
between the two, the question is indifferent and liquidity decides. The **ratio** enters as
the alternative only for a buyer who accepts the unbounded upside loss: its statistical
metric is the probability of finishing above the break-even and the expected shortfall
there, both read off the same history and FHS tables.

**When** (timing) is the same rule evaluated along the trade's life or along a date series:
the study's spot × time-to-expiry map shows the knock-out's price collapsing as the spot
approaches the barrier (the fly-to-knock-out price ratio rising from about 0.6 at 10% below
the barrier to 5 at the barrier), i.e. the point at which the holder of a knock-out should
switch into the fly if the view is still bullish; the 127-day series of 2022 H2 shows the
regret share and the premium ratio moving with the distance in standard deviations and with
implied-minus-realised vol — the two state variables that carry most of the variation.

## 4. Market-practice check

* Structuring desks sell reverse knock-outs (the UOC with `B > K`, the DOP with `H < K`) as
  the cheap way to buy a capped view; their mark is the knock-out's LV/LSV price plus a
  barrier shift. The study's decomposition `UOC = spread − cliff − regret` is the trader's
  standard way to read the quote: compare the quoted premium with the model-free `EKO` (both
  the spread and the digital are listed), the difference is the regret value the dealer
  charges, to be set against one's own touch view.
* Call ratios (1×2, 1×3) are the desk's zero-cost or credit structures in low-vol-of-vol,
  pinned markets; they are *short* the upside wing, so the metric that governs them is the
  upside tail (the 3m-momentum regime of the history table: a trending market makes the
  ratio's unbounded loss the dominant risk).
* Flies are the defined-risk version; "flies are cheap when the smile is flat" is the
  curvature metric above; the fly's three legs are the reason it is the liquid-name choice.
* The put side is where the skew enters: a put ratio sells the rich low strike (short skew),
  a down-and-out put is short a digital on the steep side of the smile; the down-and-out
  put's regret share is the largest of the book (the measured 91% at 1y 90%: the European
  down-and-out put is worth almost nothing once the path condition is added), so on the put
  side the vanilla alternatives win far more often than on the call side.
* Convention risk: a desk comparing a *daily* knock-out with a *continuous* one must shift
  the barrier (BGK); the study prices both.

## 5. What the library builds for this

* `volsto/products/structures.py`: the vanilla structures as portfolios, their model-free
  surface price, the premium matching (ratio closed form, fly width, fly wing), the European
  knock-out and the skew-adjusted digital.
* `volsto/studies/history_stats.py`: the barrier path statistics, the regret indicator, the
  regime tables, the filtered historical simulation.
* `volsto/studies/barrier_vs_vanilla.py`: the book, the anchor decomposition, the spot × time
  map, the Greeks, the hedge comparison (including the fly and the ratio as static proxies of
  the knock-out), the 127-day backdated series, the statistical layer, the decision table.

## 6. What 2007–2026 changes

Sections 1 to 5 rest on one date, 127 days of 2022 and the spot history. The same question was
then put to every weekly entry of the ORATS history: 1,031 entry dates from 2007-01-03 to
2026-09-28, strike at the snapshot spot, maturities 1 / 3 / 6 / 12 months, barriers at 0.25 to
2 standard deviations and at fixed percentages, both sides — 74,232 trades, every vanilla
structure priced off the day's surface, the knock-outs under local vol and under the desk LSV
mark (SSR 1.2), all of them marked and delta-hedged every day to expiry. The specification is
`outputs/interview/BARRIER_STUDY_SPEC.md` with its addendum; the code is
`volsto/studies/barrier_history.py`, `volsto/studies/barrier_theory.py` and
`scripts/barrier_*.py`; the report is `outputs/interview/report/report.pdf` (strict sample:
the 744 entries whose surface meets the eSSVI tolerances) and `report_built/` (all 1,028 built
entries). The numbers below are from the strict sample, in % of the entry spot, with block
standard errors over entry dates. The LSV daily series (every knock-out re-marked and hedged
each day under the LSV, leverage recalibrated daily) covers the entries from 2021-06-01 so far:
221 entry dates, 7,380 trades; the render is that of 2026-10-05 05:03.

**What it confirms.**

- *Two anchors, not a band.* The three desk marks (SSR 1.0, 1.2, 1.5) price the knock-outs alike;
  local vol and the LSV are the low and high anchors of the up-and-out call and the reverse for
  the down-and-out put. The signed gap LSV − local vol grows with maturity: 0.014, 0.063, 0.123,
  0.209 at 1, 3, 6 and 12 months on the call side, the same on the put side. It is the price of
  forward skew at the barrier: regressed on the "roll value" (the skew the local-vol price does
  not charge for, times the annuity of the barrier) the slope is 0.72 ± 0.02 on calls and
  0.62 ± 0.03 on puts.
- *The touch test sides with the stationary smile.* On the 8,741 trades knocked on the daily
  rule, the skew on the knock day is within a few percent of what the entry-day smile predicts
  by moneyness (ratio 0.94–0.98 on up touches, 1.06–1.25 on down touches) and well away from
  local vol's prediction, on both sides and at 1 and 3 months. The at-the-money vol at up
  touches is 0.8–1.2 vol points above local vol's and 1.7 below at down touches. Getting out of
  the static hedge C8 at an up touch costs the holder of the knock-out 0.12 (1m) and 0.28 (3m)
  per knocked trade because of the smile, close to 0.8·w·κ·n.

- *The model moves the premium, not the hedge.* On the 221 entries from 2021-06 marked and
  hedged every day under both models (daily knock-outs, standard-deviation barriers), the
  dispersion of the hedged P&L is the same under the LSV and under local vol (0.53 against 0.53
  at 1m, 0.70 against 0.72 at 3m, 0.94 against 0.97 at 6m on the call side; within 0.02 on the
  put side), and the mean hedged P&L of the buyer differs by about the difference of the two
  premiums: −0.02, −0.09, −0.19 on calls and +0.01, +0.05, +0.09 on puts at 1, 3 and 6 months,
  for premium differences of +0.02, +0.08, +0.15 and −0.01, −0.05, −0.10.

**What it contradicts or qualifies.**

- *Hedged, the knock-out bought at the local-vol price beat the fly.* Section 3.4 expected the
  fly to win when the vol responds strongly to rallies. Delta-hedged, fly (n = 2) minus knock-out
  is negative in every tercile of every primary condition on the call side (−0.05 to −0.32), and
  most negative in the top tercile of the upside skew-stickiness ratio (−0.20 ± 0.03 at 1m against
  −0.08 ± 0.02 in the bottom tercile). On the put side the difference is small (0 to −0.1).
- *The justified premium is above both anchors on the call side.* The knock-out premium at which
  the knock-out would only have matched the fly, hedged, is local vol plus 0.077 at 1m and 0.179
  at 3m, where the LSV adds 0.014 and 0.063: a share of 5.6 [3.8, 7.3] and 2.8 [1.9, 3.8] of the
  gap between the anchors. On the put side the share is −0.9 [−1.4, −0.3] at 3m. The touch-day
  share (what leaving C8 at the knock cost, against what each anchor charges) is between the
  anchors at 3 months (0.63 on calls, 0.67 on puts) and outside at 1 month.
- *Rule 4* (knock-out when the skew the dealer charges is below the skew expected at the touch,
  nothing fitted) beats always-fly by 0.02–0.10 hedged in both halves, and does not beat
  always-knock-out.
- *Fly or ratio.* Beyond one standard deviation on the call side the far call W is worth less than
  0.02 and ratio and fly are the same trade. Near the barrier the ratio earned 0.2–0.35 more than
  the fly, hedged, at 3 months (barriers at 0.25 and 0.5 standard deviations), for a worst 1 %
  unhedged outcome of −6 to −16 against −0.4 to −2.3 for the fly.

**Limits to read these with.** The knock-out has no market price: its premium is a model's. The
vanilla structures are priced on the fitted surface, which misses the quoted mids by 2–5 bp of
spot on a spread or a fly at 3 months (quote-based table of the report): of the size of several
of the differences above. The hedge is a forward to expiry rebalanced once a day at the
snapshot, without cost. About 236 independent one-month windows and 78 three-month ones carry
the inference; the six-month and one-year results are indicative. The verdict on the eleven
pre-registered predictions is the report's table `verdict`.

