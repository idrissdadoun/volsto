# Palladium (the call on dispersion) against single-name straddles and the basket straddle

*The theory behind `volsto/studies/dispersion.py` (SPEC §8.5). The measured results are in
`outputs/studies/dispersion/study.md`.*

## 1. The contracts

A basket of `n` names with weights `w_i` (`Σ w_i = 1`), performances `r_i = S_i(T)/S_i(0) − 1`
over the horizon `T`, basket performance `r_B = Σ w_i r_i`. The **dispersion** of the owner's
definition is

    D = Σ_i w_i |r_i − r_B|,

the weighted sum of the names' absolute performances against the basket's. The **palladium**
is the call on it, `(D − K)⁺`; `K = 0` is the dispersion forward, the strike is usually quoted
as a fraction of the forward dispersion (the study takes 80%).

The alternatives a desk weighs against it:

* the **single-name straddle package** `Σ w_i |r_i|` (one at-the-money straddle per name,
  the basket's weights);
* the **basket straddle** `|r_B|`;
* their difference, the **straddle dispersion trade** `Σ w_i |r_i| − λ_B |r_B|`, long the
  names' straddles and short the basket's, with `λ_B = 1` (basket weights) or `λ_B` chosen
  premium-neutral (the package costs nothing: the market world's `λ_B ≈ 1.4`);
* the delta-hedged analogue, the **variance dispersion** `Σ w_i RV_i − RV_B` (what the straddle
  package becomes when every leg is delta-hedged daily: the realised variances).

## 2. The identity that frames the question

Path by path, the triangle inequality gives

    Σ w_i |r_i| − |r_B|  ≤  D  ≤  Σ w_i |r_i| + |r_B|.

The straddle dispersion trade is the *lower bound* of the palladium's payoff. The gap between
the two is

    D − (Σ w_i |r_i| − |r_B|) = 2 Σ_{i : sign(r_i) ≠ sign(r_B)} w_i · min(|r_i|, |r_B|)
                                (plus the names that moved with the basket but less than it),

non-zero only on the names that moved *against* the basket (or less than it in the same
direction): the palladium pays for every name's deviation from the basket, the straddle trade
only for the excess of the names' moves over the basket's. When every name moves with the
basket and more than it (a crash where everything falls but by different amounts), the two
coincide; when names move in opposite directions (sector rotation, idiosyncratic events, a
flat basket with busy components), the palladium pays and the straddle trade pays little.

That is the whole intuition: **the palladium is a call on *relative* moves; the straddle
package is a call on *excess absolute* moves.** The palladium forward is worth much more than
the straddle trade — on the market world of the study about 2.7 times — and the question is
whether the extra premium buys the paths the buyer expects.

## 3. Closed forms and the parameters

With `r_i ≈ σ_i W_i(T)` jointly normal (`E|X| = σ √(2T/π)` for `X ~ N(0, σ²T)`):

    palladium forward     E[D] = √(2T/π) · Σ_i w_i σ_{i−B},     σ_{i−B}² = σ_i² + σ_B² − 2 Cov(r_i, r_B)/T
    straddle dispersion   E[Σ w_i|r_i| − |r_B|] = √(2T/π) · (Σ_i w_i σ_i − σ_B)
    variance dispersion   Σ_i w_i σ_i² − σ_B²
    basket vol            σ_B² = Σ_ij w_i w_j ρ_ij σ_i σ_j

Equal vols `σ` and a constant correlation `ρ` make the parameters explicit (`n` large):
`σ_B ≈ σ √ρ`, `σ_{i−B} ≈ σ √(1 − ρ)`, so

    palladium forward ≈ σ √(1−ρ) · √(2T/π),      straddle dispersion ≈ σ (1 − √ρ) · √(2T/π).

The ratio `√(1−ρ)/(1−√ρ)` is 2.4 at `ρ = 0.5` and 3.7 at `ρ = 0.7`. The sensitivities per
unit of correlation are `−σ/(2√(1−ρ))` for the palladium and `−σ/(2√ρ)` for the straddle
trade: at high correlation the straddle trade is the more correlation-sensitive *per unit of
notional*, the palladium the more sensitive *in absolute value* — and per unit of **premium**
the straddle trade wins by far, since its premium is a fraction of the palladium's. Both are
linear in the vol level: a view on the vol level alone does not separate them; a view on the
*ratio* of single-name to basket vol — i.e. on correlation — does.

The call on dispersion adds a second-order parameter the forward has not: the **vol of
correlation**. `D` is concave in `ρ` (the forward *falls* when the realised correlation is
uncertain around its mean), but the call struck near the forward is convex and gains from
correlation uncertainty (the study's `corr_sd` axis: the call-to-forward ratio rises with
it). The palladium call is therefore the instrument of a view that *correlation will move a
lot*, in either direction, whereas the straddle trade and the forward are views that it will
*fall*.

The three departures from the constant-correlation lognormal world that the study prices:

* **local correlation** `λ` — `ρ_t = ρ − λ (B_t − 1)`: correlation rises when the basket
  falls (the cross-asset side of the index skew; what makes index puts dear relative to
  single-name puts). It fattens the basket's down tail (the basket straddle gains) and takes
  dispersion away precisely when the single-name vol is high;
* **uncertain correlation** `ρ_sd` — the realised correlation of the horizon drawn from
  `ρ ± ρ_sd`: the vol-of-correlation axis above;
* **idiosyncratic events** `(p_J, σ_J)` — a name jumps once over the horizon with
  probability `p_J`: dispersion without basket vol, the palladium's best case (measured: the
  jumps raise the palladium forward several times more than the basket straddle).

## 4. The framework: parameters and expectations

The market world is a set of *implied* parameters: the names' implied vols `σ_i^impl`, the
implied correlation `ρ^impl` (the index-implied correlation, or the Cboe COR3M index for the
S&P 500), the dealer's correlation skew and the palladium's implied vol-of-correlation (read
off the call's price against its forward). The buyer's expectations are the same parameters
under the real-world measure. The framework compares the trades on the **expected P&L per unit
of premium** when bought at the market world's prices and realised under the expectation
world, with the P&L std and the loss probability beside it. The study's `expectations` table
runs the grid

    realised correlation ρ^impl − 0.2 … + 0.2   ×   realised vols × {0.8, 1, 1.2}
    × local correlation λ ∈ {0, 3}   ×   ρ_sd ∈ {0, 0.2}   ×   p_J ∈ {0, 0.2}

and names the best trade per cell. The readings:

| expectation | favoured trade | why |
|---|---|---|
| realised correlation below implied, vols near implied | **straddle dispersion trade** (premium-neutral) | the cheapest exposure to `ρ`: the correlation risk premium (COR3M above realised by about 10 points on average over 2006–2026) is captured with no vol-level bet |
| realised correlation below implied *and* names moving against each other (rotation, idiosyncratic news: `p_J > 0`) | **palladium call** | the triangle gap pays: the call on `D` at 80% of the forward pays the relative moves the straddle trade misses, and the convexity in `ρ` adds when the correlation realises far from its mean |
| realised correlation above implied (a systemic move, `λ > 0`) | **basket straddle** | everything moves together: the basket vol realises above the implied, dispersion is small relative to the basket move; the palladium and the straddle trade both lose |
| single-name vols above implied, correlation unchanged | **single-name straddles** outright (or the package) | a vol-level view, not a dispersion view |
| large uncertainty about where correlation will realise | **palladium call** over the forward | the convexity: the call-to-forward ratio rises with `ρ_sd` |

Two parameters a buyer should always check before paying for a palladium rather than the
straddle trade: the **basket's size and heterogeneity** (dispersion is the sum over names:
a small basket with one dominant name is a bet on that name, and a basket of very different
vols is dominated by the high-vol names — the study's `n` and vol axes), and the **horizon**
(all forwards scale with `√T`, the correlation risk premium and the jump probability
accumulate with `T`: longer palladiums are the more "event" products).

## 5. Statistical layer

For the ten-name large-cap basket of the study (`data/history`, 2006–2026 with the Cboe
COR3M), per 3-month window: the realised dispersion `D`, the straddle package payoff, the
basket's absolute move, the realised pairwise correlation, the implied correlation at entry.
The patterns that matter for the framework:

* the **correlation risk premium**: COR3M sits above the realised 3m correlation on average
  (the gap is the straddle trade's carry) and widens in high-VIX regimes;
* the **palladium / straddle-trade payoff ratio** by regime: it is largest when the basket is
  flat (the `−3..3%` bucket: the names move, the basket does not) and smallest in large
  basket moves of either sign — the triangle gap closes when everything moves together;
* realised dispersion rises with the VIX tercile *less* than the basket's absolute move does:
  high-vol regimes are high-correlation regimes.

The ex-post P&L of each trade at proxy market prices (Gaussian closed forms at COR3M and the
trailing realised vols times an assumed implied-to-realised ratio of 1.15 — single-name
implied vols are not in the repository, and the ratio is an input) completes the picture by
regime; the sign of these P&Ls depends on that ratio, their *ranking* across regimes much
less.

## 6. Market-practice check

* Dispersion desks run the straddle (or variance-swap) version vega- or correlation-weighted:
  short index vol, long single-name vol, sized so the vega nets to zero; the P&L is realised
  correlation against implied, with a vol-of-vol residual. The premium-neutral package of the
  study is that trade in terminal form.
* The palladium is a dealer product (a "call on dispersion" sold to funds wanting a clean
  long-dispersion payoff without the delta and vega management of the straddle book); dealers
  price it on a multi-asset local-vol model with a correlation skew, and hedge it with the
  straddle / variance book — the triangle gap is the dealer's residual, which is why it is
  quoted above its Gaussian value.
* The straddle trade is a *delta* trade until hedged: unhedged, both the package and the
  palladium are terminal payoffs; delta-hedged daily, the package becomes the variance
  dispersion, and the comparison is palladium (terminal, path-independent) against variance
  dispersion (realised, path-dependent, its P&L accruing with the daily returns).
* The basket straddle alone is the systemic-move trade; it is the right instrument when the
  expectation is "vol up, correlation up" (a crash), where the palladium pays least.

## 7. What the library builds for this

* `volsto/multi/`: correlated draws from the library's CRN streams (one seed per asset: a
  correlation bump leaves the names' draws unchanged), the multi-asset model over the
  factor-free single-asset kernels (Black–Scholes or Dupire local vol per name, so each name
  can carry its own skew), the path container, the Monte Carlo loop, the products (palladium,
  basket option / straddle, single-name straddles, the straddle package, the variance
  dispersion) and the Gaussian closed forms (including the folded-normal moments behind the
  Bachelier price of the call on dispersion).
* `volsto/studies/dispersion.py`: the worlds (local, uncertain correlation, jumps on the
  one-factor representation), the trades, the market world from the data, the sensitivities,
  the expectations grid, the history.
