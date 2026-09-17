# volsto methodology

This page describes what the library computes and how, for a reader who has not followed the
build: the model, the calibrations (the leverage by the particle method, and the two-factor
parameters by SABR break-evens with its two dials), the conventions, and the known limits of the
first-order break-even engine. `SPEC.md` is the authoritative record; section numbers below
(§) refer to it. Every number quoted here is a measured value recorded in SPEC, with its section.
Bergomi, *Stochastic Volatility Modeling* (CRC, 2016), is cited by chapter and equation number
only ("book eq. 7.30").

**volsto is its own system.** It has its own surface parametrisation (SSVI and eSSVI, not the
desk's in-house surface), its own conventions (below), and it treats the SSR mark and the skew
tolerance as free inputs. It is a research and risk tool. It is not a replica of any production
marking engine, and no number it produces is meant to match one (§15 Part 3, owner's decisions;
§16).

Contents

1. [Conventions](#1-conventions)
2. [Market layer](#2-market-layer)
3. [Models](#3-models)
4. [Simulation](#4-simulation)
5. [Leverage calibration (particle method)](#5-leverage-calibration-particle-method)
6. [The two-factor marking calibration (SABR break-evens)](#6-the-two-factor-marking-calibration-sabr-break-evens)
7. [Smile dynamics and the SSR](#7-smile-dynamics-and-the-ssr)
8. [Risk, attribution and hedging](#8-risk-attribution-and-hedging)
9. [Known first-order limits of the break-even engine](#9-known-first-order-limits-of-the-break-even-engine)
10. [Other known limits](#10-other-known-limits)
11. [Entry points](#11-entry-points)

---

## 1. Conventions

Each convention is stated once here; the rest of the page uses it without restating it.

| Item | Convention | Where |
|---|---|---|
| Time | Year fractions, ACT/365 fixed. $t$ is the current time and $T$ a maturity, both in years. | §2.1 |
| Trading days | 252 per year. This covers the realised-variance annualisation $A = 252$, the daily fixing schedule, the trading index of a seasoned trade ($j = 252\,t$), one business day ($1/252$, `volsto.risk.greeks.BUSINESS_DAY`) for theta and for the backtest attribution step, and $\sqrt{252}$ for historical vol of vol. | §6.1, §6.10, §7.3, §10.3, §15 Part 2 |
| Simulation steps | $1/1460$ below 3 months, $1/365$ to 2 years, $1/250$ beyond. These are step sizes, not a day count. | §3.1 |
| SABR quotes (step 0) | "365-day" quotes on ACT/365 maturities (section 6.1 of this page). | §15 Part 3 |
| Spot, forward, rates | $S_t$ is the spot. $r$ is the rate and $q$ the continuous dividend yield or repo, both deterministic. $F(T) = S_0 \exp\int_0^T (r - q)$ is the forward. | §2.1 |
| Log-moneyness | Surfaces, local vol and the leverage use forward log-moneyness: $k = \ln(K/F(T))$ for a strike $K$, and $k = \ln(S/F(t))$ for the leverage's spot axis. The delta regimes of §7.2 write the smile in spot log-moneyness $\ln(K/S)$. | §2.2, §4.1, §7.2 |
| Implied vol | $\hat\sigma(k, T)$ is the Black implied vol, a decimal (0.20 means 20 %). $w(k,T) = \hat\sigma^2 T$ is the total implied variance. ATMF (at the money forward) means $K = F(T)$, i.e. $k = 0$. | §2.2 |
| Vol point (vp) | 0.01 of implied vol: 20 % to 21 % is +1 vp. Variance-swap strikes are quoted as a vol, in vol points. Vega is per vol point. | §6.1, §7.3 |
| ATM skew | $\mathcal S_T = \partial\hat\sigma_T/\partial \ln K$ at the forward, i.e. $\partial\hat\sigma/\partial k$ at $k = 0$. It is negative for an equity put skew. | §4.4, §15 Part 3 |
| 90/110 skew | $\hat\sigma(\ln 0.9) - \hat\sigma(\ln 1.1)$ in vol points, positive for a put skew. The butterfly is $[\hat\sigma(\ln 0.9) + \hat\sigma(\ln 1.1)]/2 - \hat\sigma(0)$. | §7.6 |
| SSR | $R_T = \dfrac{1}{\mathcal S_T}\,\dfrac{E[d\hat\sigma_T\, d\ln S]}{E[(d\ln S)^2]}$ (book eq. 9.3 / 12.50). Sticky strike gives $R = 1$ and sticky local vol gives $R \approx 2$ (Derman). SABR dynamics give $R = 2$. The order-one SSR of the pure two-factor model lies in $[1, 2]$ (book eq. 9.21). | §4.4, §7.2, §15 Part 3 |
| Vol of vol | $\nu$ is the lognormal vol of a variance-swap vol of vanishing maturity, and $\omega = 2\nu$ is the lognormal vol of vol of the instantaneous variance. Configs store $\nu$. The earlier studies' $\omega = 3$ is $\nu = 1.5$. | §3.3, §16 |
| Prices | A product's value is in units of its `notional`. With `notional=100` it reads in % of notional. The M6 headline notes use notional 1, i.e. fractions of notional. The results store's M6 cells are fractions stored under a "% notional" label, and study S4 converts them (§10.2). The backtest stores product units and a `scale` / `unit` column (× 100 gives % of notional, % of the inception spot, or vol points of vega notional). | §6.8, §10.2, §10.3 |
| Hedging units | Autocall, Phoenix, cliquet and VKO in % of notional. The study-D vanilla in % of spot. The FVA in vol points × notional. The KO variance swap in vol points of vega notional: the variance P&L divided by $2K_{\text{vol}}$, with $K_{\text{vol}}$ the pricing model's fair strike. | §8.2 |
| Desk P&L sign | The desk is **short** the note. The hedger prices the product long, and every "desk" number is its negative (`volsto.studies.m8b.to_desk_pnl`). For the shadow rotation, fee = P1 price − LV price, and the desk P&L per +1 rota = −(d fee). The backtest's P&L is $V(d) - V(d-1)$ + flows, i.e. the change in the product's value (section 10 of this page). | §8.2, §15 Part 3, §10.3 |
| Rota | +1 rota is an ATM-skew move of $2/\sqrt{T}$ vol points per unit log-moneyness, with $T$ floored at 1 month. It is applied through the saturating profile $k_{\text{cap}}\tanh(k/k_{\text{cap}})$ with $k_{\text{cap}} = 0.5$. Positive means puts up (the skew steepens). At 6 months +1 rota is +0.56 vp of 90/110 skew (0.568 without the saturation). The code constant is `volsto.risk.shadow_rotation.ROTATION_CONVENTION`. | §15 Part 3 |
| Bumps | Delta and gamma: 1 % of spot in log space, central. Vega: 1 vp. Forward-variance buckets: +1 vp of the bucket's forward variance-swap vol. Model parameters: 5 % relative for $\nu, k_1, k_2$ and 0.05 absolute for $\theta$ and the correlations. | §7.1 |
| Errors | No Monte Carlo number without its standard error. A difference priced on common paths carries the paired error. A difference of two stored estimates carries the quadrature error, labelled an upper bound. | §5, §10.2, §11 |
| Particle counts | Production (headline tables, baselines, grid precompute): $8\cdot10^5$ particles, single seed. Development: $2\cdot10^5$. The test-suite toy grids use $2\cdot10^4$. | §11, §9.2 |
| Seeds | Normals are addressed by (seed, path, step, Brownian), so bumps under common random numbers are exact. The M4/M6 baseline pricing seed is 2024. | §5, §9.2 |

---

## 2. Market layer

**Curves** (`volsto.market.curves`): `DiscountCurve`, `ForwardCurve`, piecewise-flat rates by
default (§2.1).

**Implied surfaces** (`volsto.market.surface`). `ImpliedSurface` exposes the total variance
$w(k, T)$. `SSVISurface` is the Gatheral–Jacquier SSVI (§2.2):

$$
w(k,T) = \frac{\theta_T}{2}\Big(1 + \rho\,\varphi(\theta_T)\,k + \sqrt{(\varphi(\theta_T)\,k + \rho)^2 + 1 - \rho^2}\Big),
\qquad
\varphi(\theta) = \frac{\eta}{\theta^{\gamma}(1+\theta)^{1-\gamma}}
$$

Here $\theta_T$ is the ATM total variance, interpolated linearly in $T$ between pillars (flat
forward variance). $\rho$ is the smile correlation, and $\eta, \gamma$ set the power law. The
butterfly conditions are checked at construction.

`ESSVISurface` replaces $\rho$ by a per-pillar $\rho_T$, piecewise linear in $T$, and is the
default for imported market surfaces. Plain SSVI remains the default for synthetic surfaces
(§2.2, §13). A surface is built from a config only by `surface_from_config`, so an eSSVI
snapshot is never flattened to a single $\rho$ on its way to a calibration or a risk bump (§13.2).

**Calendar repair** (`volsto.market.import_hdn.repair_calendar`, `volsto.market.surface.certify_calendar`,
§13.1). For every eSSVI surface the importer returns without a fallback, the exact derivative
$\partial_T w(k,T) \ge 10^{-4}$ per year holds for every $|k| \le 3$ and every $T$ from one day
to the surface's last maturity. This is proven by branch and bound on the closed-form
$\partial_T w$ (`essvi_dw_dt`), not sampled. The repair refits only $\rho_T$ (with $\gamma$ in its
box and $\eta$ capped) and freezes $\theta_T$. On the 127-day 2022 H2 SPX sample: 127/127 days
certified, 25 of them repaired and 102 bit-equal, with no fallback (§13.1).

**Local volatility** (`volsto.market.dupire.LocalVolSurface.from_implied`, §2.3). Dupire in total
variance, derivatives by central differences on the analytic surface:

$$
\sigma_{\text{loc}}^2(k,T) = \frac{\partial_T w}{1 - \frac{k}{w}\partial_k w + \frac14\left(-\frac14 - \frac1w + \frac{k^2}{w^2}\right)(\partial_k w)^2 + \frac12\partial_{kk} w}
$$

It is floored at a small positive value and stored on a square-root-spaced grid of 400 times and
$k \in [-3, 3]$ with $dk = 0.0025$. The measured reason for the width: $\pm 1.5$ priced 2y/3y
variance swaps 0.08/0.22 vp low (§2.3).

**Variance swaps and $\xi_0$** (`volsto.market.varswap`, §2.4). `varswap_strike` is log-contract
replication, with $P$ and $C$ the put and call prices at strike $K$:

$$
K_{\text{var}}(T) = \frac{2}{T}\int\left[\frac{P(K)}{K^2}\mathbf 1_{K<F} + \frac{C(K)}{K^2}\mathbf 1_{K>F}\right] e^{rT}\,dK
$$

The integration uses $k_{\max} = \max(25\,\sigma_{\text{ATM}}\sqrt T, 3)$. `xi0_curve` gives the
initial forward variance curve $\xi_0(T) = \frac{d}{dT}\big[T\,K_{\text{var}}(T)\big]$, through a
PCHIP interpolant of the strip, which keeps it positive and integrating back to the strip.

**Importer** (`volsto-import`, `volsto.market.import_hdn`, §13). It reads HistoricalData.net
end-of-day chains, implies forwards by put–call-parity regression, prunes butterfly and calendar
violations, and fits $\theta_T$ at the pillars (1m, 3m, 6m, 1y, 18m, 2y, 3y) by isotonic least
squares. The global parameters are fitted by vega-weighted least squares on $|k| \le 0.25$ and
expiries from 3 weeks. The vendor's implied vols and Greeks are never inputs.

---

## 3. Models

All models implement `volsto.models.base.Model` and produce a `PathSet`.

**Two-factor lognormal Bergomi** (`volsto.models.bergomi.BergomiSV`, §3.3; book ch. 7,
eqs. 7.29–7.35). Two Ornstein–Uhlenbeck factors start at zero:

$$
dX^i_t = -k_i X^i_t\,dt + dW^i_t,\quad i = 1, 2,\qquad d\langle W^1, W^2\rangle_t = \rho_{12}\,dt,\qquad k_1 > k_2
$$

$X^1$ is the short factor. The Gaussian driver of maturity $T$ is

$$
x_t^T = \alpha_\theta\Big[(1-\theta)\,e^{-k_1(T-t)}X^1_t + \theta\,e^{-k_2(T-t)}X^2_t\Big],\qquad
\alpha_\theta = \big((1-\theta)^2 + \theta^2 + 2\rho_{12}\theta(1-\theta)\big)^{-1/2}
$$

The forward variances $\xi_t^T$ (the fair variance at $t$ for the instant $T$) evolve as

$$
\xi_t^T = \xi_0^T \exp\Big(\omega\, x_t^T - \tfrac12\omega^2\chi(t,T)\Big),\qquad \omega = 2\nu,\qquad \chi(t,T) = \operatorname{Var}[x_t^T]
$$

so $E[\xi_t^T] = \xi_0^T$. The instantaneous variance is $V_t = \xi_t^t$. The parameters are
`BergomiParams(nu, theta, k1, k2, rho12, rho_SX1, rho_SX2)`, where $\theta \in [0, 1]$ weights the
long factor and $\rho_{SX_i} = \operatorname{corr}(dW^S, dW^i)$. The $3\times3$ correlation matrix
is checked positive semi-definite.

**Spot and leverage (LSV)** (`volsto.models.lsv.LSV`, §3.3, §3.5). $W^S$ is the spot Brownian
motion and $L$ the leverage function:

$$
d\ln S_t = \Big(r_t - q_t - \tfrac12 L(t,S_t)^2\,\xi_t^t\Big)dt + L(t,S_t)\sqrt{\xi_t^t}\,dW^S_t
$$

- **Pure SV**: $L \equiv 1$.
- **One-factor model**: $\theta = 0$. `BergomiParams.one_factor(omega, kappa, rho)` sets
  $\kappa = k_1$ and $\rho = \rho_{SX_1}$, the model of the earlier studies.
- **Local vol**: $\nu = 0$. A grid's $\nu = 0$ point is the surface's pure Dupire model
  (`volsto.models.localvol.LocalVol`), never a particle calibration (§9.2).
- `LeverageFunction` stores $L$ on a $(t, k)$ grid. It interpolates linearly, extrapolates flat
  in $S$, and holds the last slice beyond the calibration horizon (§3.5).
- "P1" is the desk's name for this two-factor LSV (§16).

**Break-even parametrisation** (`volsto.analytics.reparam`, §15 Part 3). Write
$X = X^1$, $Y = X^2$, $\rho_{SX} = \rho_{SX_1}$, $\rho_{SY} = \rho_{SX_2}$ and
$\rho_{XY} = \rho_{12}$. The map is

$$
\omega_1 = 2\nu\alpha_\theta(1-\theta),\quad \omega_2 = 2\nu\alpha_\theta\theta,\quad
\lambda_1 = \rho_{SX}\,\omega_1,\quad \lambda_2 = \rho_{SY}\,\omega_2,\quad
\chi = \frac{\rho_{XY} - \rho_{SX}\rho_{SY}}{\sqrt{1-\rho_{SX}^2}\sqrt{1-\rho_{SY}^2}}
$$

- $\omega_i$ is factor $i$'s loading on $\ln V_t$.
- $\lambda_i$ is the spot-correlated part of that loading.
- $\chi$ is the residual factor correlation.
- The inverse is $\theta = \omega_2/(\omega_1+\omega_2)$, $\rho_{Si} = \lambda_i/\omega_i$ and
  $\nu = (\omega_1+\omega_2)/(2\alpha_\theta)$. The round trip holds to $10^{-12}$.
- The order-one ATMF skew and spot/vol covariance depend on $(k_1, k_2, \lambda_1, \lambda_2)$
  only.

Black–Scholes (`volsto.models.bs.BlackScholes`) is available as a test bed. Heston, listed in
§0 and §3.4, has no module in `volsto/models/`.

---

## 4. Simulation

`volsto.engine.MonteCarlo` prices a product on a `TimeGrid`. The grid is the union of the
product's fixings, the step schedule and, for an LSV, the leverage's calibration slices (§5).
It returns `PriceResult(mean, stderr)`.

- **Factors** are stepped exactly (book eqs. 7.15–7.18). The joint covariance of the spot and
  factor increments is Cholesky-factorised once per step size (§3.3).
- **Spot step**: Platen's explicit weak order-2 scheme by default, with the second-order SV step
  (`SchemeConfig.sv_order2`). That step advances the factors first, uses the trapezoidal variance
  in the drift, and adds the spot/variance cross terms with the exact step covariance as
  compensator (§3.1, §3.3, §4.2 M6 Part 0 item 7).
- **Frozen-leverage rule**: every leverage lookup inside a step uses the step-start slice
  $L(t_n,\cdot)$, identically in pricing and calibration. `volsto.models.lsv.step_lsv_block` is
  the one stepping routine both use (§3.1, §3.5).
- **Realised variance** of a product is the sum of squared log returns on its fixing dates
  (daily by default), never on the simulation grid (§5).

---

## 5. Leverage calibration (particle method)

**Target** (Guyon–Henry-Labordère; book §12.2.5; §4.1). The leverage makes the LSV reprice the
target surface:

$$
L(t,S)^2 = \frac{\sigma_{\text{loc}}^2(t,S)}{E[V_t \mid S_t = S]}
$$

**Algorithm** (`volsto.calibration.particle.calibrate_leverage`, §4.1):

1. Start from $L(0,\cdot) = \sigma_{\text{loc}}(0,\cdot)/\sqrt{\xi_0^0}$.
2. Step $N$ particles with the pricing kernel and the current slice.
3. Estimate $E[V \mid S]$ at the new slice by local-linear kernel regression in $k$:
   - a plug-in $\tfrac12 h^2 m''$ curvature correction;
   - bandwidth $h = 1.5\,\sigma_{\text{ref}}\sqrt t\,N^{-1/5}$;
   - a window floor of $\max(2000, 0.01N)$ particles;
   - a 201-point grid over the cloud's trusted quantile range, interpolated onto the Dupire grid;
   - a saturating log-quadratic continuation beyond the trusted quantiles (default
     `ParticleConfig.tail_extrapolation = "log_quadratic"`).
4. Set the slice and continue.

**Cache** (`volsto.calibration.cache.LeverageCache`, §4.3). The key is the SHA-256 of the
calibration spec (market, surface, perturbation, model, particle settings, local-vol grid, step
schedule, scheme) plus the manual code tag `CALIBRATION_CODE_TAG` (`m6`). `get_or_calibrate` is
the only entry point. A study or viewer reads with `allow_calibrate=False` and reports a miss
with the command that fills it. Every file is written atomically (§9.2).

**Where calibration is allowed.** Only three commands calibrate:

- `volsto-precompute`, for the viewers' grid (§9.2);
- `volsto-backtest run`, for the backtest (§10.3);
- the research scripts that say so.

`calibrate_leverage` refuses while a study runs (§10.1).

**Cost** (§4.1, §4.2 M4b item 5, §9.2, §10.3):

| Particles | 3y calibration | Source |
|---|---|---|
| $2\cdot10^5$ | about 33 s | §4.1, §4.2 M4b item 5 |
| $8\cdot10^5$ | 124 s at M4b; 138 s median of the cache manifest | §4.2 M4b item 5, §9.2, §10.3 |

**Accuracy**, measured after the M6 step fix (§4.2 item 8) with $8\cdot10^5$ particles, three
particle seeds and six pricing seeds:

- **Gate met.** ATM and −20 % / −30 % wings are within 0.05 vp at 2 standard errors from 1y to
  3y for the 1F $\omega = 3$ and 2F Table 8.2 sets.
- **Known far-wing bias** (§4.2 item 9):
  - 1F: the 1y–2y variance swap is at −0.08 vp;
  - 2F: −0.035 vp at 1y–2y and +0.056 vp at 3y.

  The far put wing lies beyond the regression's trusted quantiles. The two model-consistent tail
  continuations moved the residual the wrong way (1y–2y variance swap −0.162 to −0.246 vp) and
  are kept only as options.
- **Single-run noise.** A single $2\cdot10^5$ calibration carries ±0.035 vp of particle-seed
  noise per variance-swap pillar (§4.2 M4b item 3).

`volsto.calibration.diagnostics.reprice_surface` produces the repricing table
(`CalibrationReport`, §4.2).

---

## 6. The two-factor marking calibration (SABR break-evens)

`volsto.calibration.fit_2f.fit_2f_marking(surface, BreakEvenFitConfig(...), ssr_target=...)`
fits the seven two-factor parameters to one surface (§15 Part 3). The leverage is then
calibrated for the fitted set as in section 5. The fit itself takes seconds and calibrates nothing.
The method has a step 0 (the SABR reduction), a step 1 (the targets), the P1 break-evens, and
two minimisations (steps 2 and 3).

### 6.0 Break-even observables

**Symbols.**

- $\hat\sigma_T$ is the ATMF implied vol of maturity $T$.
- $\sigma_0 = L(0,S_0)\sqrt{\xi_0^0}$ is the model's instantaneous vol. The targets use the
  1-month ATMF vol as its market proxy: 0.2200 against 0.2194 for the cached 2F LSV on the
  reference surface.
- $\mathrm{SensiSpot}(T) = \sigma_0\,\partial\hat\sigma_T/\partial\ln S_0$.
- $\mathrm{Sensi}X_i(T) = \partial\hat\sigma_T/\partial X^i_0$, also written
  $\mathrm{Sensi}X = \mathrm{Sensi}X_1$ and $\mathrm{Sensi}Y = \mathrm{Sensi}X_2$.
- $\mathcal S_T$ is the model's ATM skew.

All quantities are in absolute vol units ($d\hat\sigma$, not $d\ln\hat\sigma$) per unit time
(`volsto.analytics.breakeven`, §15 Part 3). The two break-evens are

$$
\mathrm{SpotVolCovar}(T) = \frac{\langle d\ln S, d\hat\sigma_T\rangle}{\sigma_0\,dt}
= \mathrm{SensiSpot} + \rho_{SX}\,\mathrm{Sensi}X + \rho_{SY}\,\mathrm{Sensi}Y
$$

$$
\mathrm{VolVar}(T) = \frac{\langle d\hat\sigma_T, d\hat\sigma_T\rangle}{dt}
= \mathrm{SensiSpot}^2 + \mathrm{Sensi}X^2 + \mathrm{Sensi}Y^2
+ 2\rho_{SX}\,\mathrm{SensiSpot}\,\mathrm{Sensi}X + 2\rho_{SY}\,\mathrm{SensiSpot}\,\mathrm{Sensi}Y + 2\rho_{XY}\,\mathrm{Sensi}X\,\mathrm{Sensi}Y
$$

With these, $\mathrm{SSR}_T = \mathrm{SpotVolCovar}(T)/(\sigma_0\,\mathcal S_T)$.
`simulated_breakevens` gives the same quantities by finite differences on common paths. It is
the truth check of the closed forms.

### 6.1 Step 0: SABR reduction per pillar

`volsto.calibration.targets.sabr_reduce` works on each pillar $T$ of the surface. Let
$x = k/\sqrt T$ be the normalised log-moneyness. The library's 365-day quotes are

$$
\mathrm{Atf}_{365} = 100\,\hat\sigma(0,T),\quad
\mathrm{Smile}_{365} = 100\cdot 2\,\frac{\partial\hat\sigma}{\partial x},\quad
\mathrm{Convex}_{365} = 100\,\frac{\partial^2\hat\sigma}{\partial x^2}
$$

They convert to

$$
\mathrm{atf} = \frac{\mathrm{Atf}_{365}}{100},\quad
\mathrm{smi} = \frac{\mathrm{Smile}_{365}}{100\cdot 2\sqrt T} = \frac{\partial\hat\sigma}{\partial k},\quad
\mathrm{cvx} = \frac{\mathrm{Convex}_{365}}{100\,T\,(\mathrm{atf}/\mathrm{atf}_{\text{ref}})^p}
$$

with $\mathrm{atf}_{\text{ref}} = 0.3$ and $p = 1$ (`sabrw_power`). Then

$$
\nu_{\text{SABR}} = \sqrt{6\,\mathrm{smi}^2 + 3\,\mathrm{atf}\,\mathrm{cvx}},\quad
\mathrm{VoV}_{\text{SABR}} = \mathrm{atf}\,\nu_{\text{SABR}},\quad
\mathrm{Corr}_{\text{SABR}} = \frac{2\,\mathrm{smi}}{\nu_{\text{SABR}}},\quad
\mathrm{Skew}_{\text{SABR}} = \mathrm{smi}
$$

- $\nu_{\text{SABR}}$ is the lognormal SABR vol of vol (the Hagan $\beta = 1$ inversion, exact at
  $p = 0$).
- $\mathrm{VoV}_{\text{SABR}}$ is the absolute vol of vol.
- $\mathrm{Skew}_{\text{SABR}}$ keeps the sign of $\partial\hat\sigma/\partial k$, which is
  negative for equities.

**Radicand guard.** $6\,\mathrm{smi}^2 + 3\,\mathrm{atf}\,\mathrm{cvx}$ is floored at
$6\,\mathrm{smi}^2(1 - c)$ with $c = 0.5$ (`radicand_floor`). The guard is logged and flagged
when it fires (§15 Part 3, decision iii). Other flags, none silent:

- a correlation above 1 in magnitude, clipped;
- no analytic ATM skew (central differences are used instead);
- $\nu_{\text{SABR}}^2 T > 1$.

### 6.2 Step 1: break-even targets (the SSR dial)

`marking_targets` computes, per pillar,

$$
\mathrm{VoV}_{BE}(T) = \frac{\mathrm{atf}_{3M}}{\mathrm{atf}_T}\cdot\frac{\mathrm{ssr\_target}(T)}{2}\cdot\mathrm{VoV}_{\text{SABR}}(T),\qquad
\mathrm{Corr}_{BE} = \mathrm{Corr}_{\text{SABR}}
$$

$$
\mathrm{SpotVolCovar}_{\text{target}} = \mathrm{Corr}_{BE}\,\mathrm{VoV}_{BE},\qquad
\mathrm{VolVar}_{\text{target}} = \mathrm{VoV}_{BE}^2
$$

- **The SSR dial.** SABR dynamics are SSR 2, so `ssr_target = 2` reproduces them;
  `ssr_target = 1` halves $\mathrm{VoV}_{\text{SABR}}$ (sticky strike). `ssr_target` is a scalar,
  a per-pillar value or a curve (default 1). It enters $\mathrm{VoV}_{BE}$ only. It is not a
  target for the calibrated LSV's realised SSR, which is reported as a diagnostic (§15 Part 3,
  decision 1).
- **Implied SSR.** Before smoothing, the targets imply the SSR
  $\mathrm{ssr\_target}\cdot\mathrm{atf}_{3M}/\sigma_0$: 0.955 × ssr on the reference surface and
  1.084 × ssr on SPX 2022-12-30.
- **SmoothBreakEven** (on by default). $\ln\mathrm{VoV}_{BE}$ is replaced by its least-squares
  polynomial in $\ln T$ of degree $\min(2, n-2)$ over the $n$ pillars. The raw curve is kept, and
  an adjustment beyond 5 % is flagged.
- **Pillars.** They run from 3M to 10Y inside the surface. Pillars below 3M are dropped with a
  flag (`mat_min`), and so are pillars beyond the surface. SPX is quoted to 3Y, so its 5Y and 10Y
  pillars are dropped.

### 6.3 The P1 break-evens (first order)

**Symbols.**

- $W_T = \int_0^T \xi_0^t\,dt = \bar\sigma_T^2\,T$. Here $\bar\sigma_T$ is the order-zero
  (variance-swap) vol of the surface's $\xi_0$.
- $A_i(T) = \int_0^T \xi_0^t\, e^{-k_i t}\,dt \,/\, W_T$ (book eq. 7.38).
- $\tilde c_i(t) = \int_0^t \sqrt{\xi_0^u}\,e^{-k_i(t-u)}\,du$.
- $J_i(T) = \int_0^T \xi_0^t\,\tilde c_i(t)\,dt \,/\, (2\bar\sigma_T^3T^2)$ (book eq. 8.54).
  The kernel's order-one skew at maturity $T$ is $\lambda_1J_1(T) + \lambda_2J_2(T)$.
- $\mathrm{atf}_t$ is the market ATMF vol at maturity $t$ (step 0).
- $\mathcal S^{\text{mkt}}_t$ is the market ATM skew at maturity $t$.
- $\sigma^2(t) = \frac{d}{dt}(\mathrm{atf}_t^2\,t)$ is the market forward ATMF variance, and
  $f(t) = \sigma^2(t)/(\mathrm{atf}_t\,\mathrm{atf}_T)$.
- $\lambda\cdot J_t = \lambda_1J_1(t) + \lambda_2J_2(t)$ and
  $\lambda\cdot A_T = \lambda_1A_1(T) + \lambda_2A_2(T)$.

`fit_2f.P1Maps` computes (§15 Part 3):

$$
\mathrm{Sensi}X_i(T) = \tfrac12\,\omega_i\,A_i(T)\,\mathrm{atf}_T,\qquad
\mathrm{Skew}_{\text{naked}}(T) = \lambda_1J_1(T) + \lambda_2J_2(T)
$$

$$
\mathrm{SensiSpot}(T) = \sigma_0\Big[(\mathcal S^{\text{mkt}}_T - \lambda\cdot J_T) + \frac1T\int_0^T f(t)\,(\mathcal S^{\text{mkt}}_t - \lambda\cdot J_t)\,dt\Big]
$$

$$
\mathrm{SpotVolCovar}_{P1} = \mathrm{SensiSpot} + \tfrac12\,\mathrm{atf}_T\,\lambda\cdot A_T
$$

$$
\mathrm{VolVar}_{P1} = \mathrm{SensiSpot}^2 + \mathrm{SensiSpot}\,\mathrm{atf}_T\,\lambda\cdot A_T + \mathrm{Sensi}X_1^2 + \mathrm{Sensi}X_2^2 + 2\rho_{12}\,\mathrm{Sensi}X_1\,\mathrm{Sensi}X_2
$$

**Reading of these formulas.**

- $\mathrm{SensiSpot}$ is the spot sensitivity the leverage generates when it absorbs the skew
  residual $\mathcal S^{\text{mkt}} - \mathrm{Skew}_{\text{naked}}$ (book eq. 12.52 in covariance
  form). It is zero when the naked skew equals the market skew at every maturity. This
  leverage-included reading was confirmed by the owner (decision i).
- The market ATMF vol $\mathrm{atf}_T$ is the prefactor of the sensitivities. The naked skew
  keeps the order-zero vol $\bar\sigma_T$ of eq. 8.54.
- $\mathrm{SpotVolCovar}_{P1}$ and $\mathrm{Skew}_{\text{naked}}$ are affine in
  $(\lambda_1,\lambda_2)$ at fixed $(k_1,k_2)$.
- The full quadratic form of VolVar is used. The half-weight cross terms of the owner's slide are
  recorded as an unresolved normalisation (decision ii).

### 6.4 Step 2: first minimisation (the skew_eps dial)

`fit_first` solves, over $(k_1, \lambda_1, \lambda_2)$ with $k_2 = 0.2$ fixed (§15 Part 3):

$$
\min \sum_i w_i\,\big(\mathrm{SpotVolCovar}_{P1}(T_i) - \mathrm{SpotVolCovar}_{\text{target}}(T_i)\big)^2
$$

subject to

$$
\mathrm{Skew}_{\text{naked}}(T) \in \big[(1 \mp \varepsilon)\,\mathrm{Skew}_{\text{SABR}}(T)\big]\ \text{at } T_s = 1\text{Y},\ T_l = 5\text{Y},
\qquad |\lambda_1| + |\lambda_2| \le 2\,\nu_{\text{cap}}
$$

- **Weights.** $w_i = 1/\mathrm{target}_i^2$ by default ("relative"); "uniform" and explicit
  per-pillar values are also accepted.
- **The skew_eps dial.** $\varepsilon$ = `skew_eps`, a float or a pair
  $(\varepsilon_s, \varepsilon_l)$, default 0.10. This is the two-point hard skew constraint,
  and it leaves the short end free.
- **Moved constraint maturity.** A constraint maturity that is not a fitted pillar moves to the
  nearest one, with a note (3Y on SPX).
- **The box.** It is the feasibility condition of step 3.
- **Solver.** At each $k_1$ the problem is an exact 2-D quadratic programme. $k_1$ runs on a
  geometric grid in $[0.3, 20]$, then a bounded refinement.
- **Soft mode.** `skew_mode="soft"` replaces the constraint by an all-pillar penalty with weight
  10. `"auto"`, the default, is two-point in marking mode and soft in historical mode
  (decision v).

**Status: interior, binding or infeasible** (`FitResult.status`). A fit never rails silently.

- **Binding.** For $\varepsilon \ge 0$ the two skew slabs always intersect, so `ssr_target` cannot
  empty the constraint set. An incompatible (`ssr_target`, `skew_eps`) pair **binds**, and the
  fit carries a message naming the maturity, the edge, the naked vs market skew and the achieved
  vs target SpotVolCovar.
- **Infeasible.** This needs the $\nu$ box. The fit returns the least-violation solution with its
  message.
- **SPX binding map.** On SPX every pair of
  $\{1, 1.25, 1.5, 1.75, 2\}\times\{0.05, 0.1, 0.2, 0.3\}$ binds, except ssr 1.75 with
  $\varepsilon \ge 0.2$ (§15 Part 3).
- **The default grid's marks.** They are `ssr_target` ∈ {0.75, 1, 1.25, 1.5} × `skew_eps` ∈
  {0.05, 0.10, 0.20} on three SPX snapshots (§9.2).

### 6.5 Step 3: second minimisation

`fit_second` solves, over $(\omega_1, \omega_2, \chi)$ with $(k_1, k_2, \lambda_1, \lambda_2)$
fixed:

$$
\min \sum_i w_i\,\big(\mathrm{VolVar}_{P1}(T_i) - \mathrm{VolVar}_{\text{target}}(T_i)\big)^2
\quad\text{s.t.}\quad \omega_i \ge |\lambda_i|,\ \ \nu \le \nu_{\text{cap}},\ \ \chi \in [-0.99, 0.99]
$$

- **Solver.** SLSQP from ten starts, then the inverse reparametrisation (section 3).
- **Correlation kept at $\rho_{\text{SABR}}$.** In marking mode the VolVar target is rebuilt
  from the achieved covariance:
  $\mathrm{VolVar}_{\text{target}} = (\mathrm{SpotVolCovar}_{P1}/\mathrm{Corr}_{BE})^2$.
  A missed covariance therefore never forces
  $|\rho| = 1$, and the requested $\mathrm{VoV}_{BE}^2$ is reported beside it (decision 3).
- **The cap.** $\nu_{\text{cap}}$ = `nu_cap`, default 3.5 since decision viii (was 2.5). When
  step 2's box or step 3's cap binds, the fit logs and attaches the owner's warning (quoted in
  section 9, item 2).
- **Collapse flag.** $|\rho_{12}| > 0.9$ is flagged as the two-factor structure collapsing
  (decision vii).

### 6.6 Stage 3 and the reported diagnostics

`stage3_validation` (optional; `Stage3Inputs`) calibrates the leverage for the fitted set and
reports:

- the mean $|L-1|$ inside ±2 ATM standard deviations per slice;
- the LSV's numerical SSR (a diagnostic);
- the naked mixing skew;
- the simulated SpotVolCovar and VolVar;
- the forward 90/110 skew at 1y-into-1y and 2y-into-1y against the spot 1y skew. This is a
  diagnostic, not a target (decision vi).

**Stage-3 assertion.** The simulated break-evens must lie within 10 % (`stage3_tolerance`) of
the first-order break-evens **at the fitted parameters**, i.e. the engine bias. The miss of the
targets by a binding fit is reported, not asserted (§15 Part 3, decision viii as restricted on
2026-09-15). `fit_2f` raises `BreakEvenValidationError` when the assertion fails.

**Iteration against simulation.** `iterate_against_simulation=k` refits $k$ times with the
targets divided by the simulated/analytic ratio, and stops when the worst gap grows (section 9,
item 9).

**Always reported** (`FitResult.summary()`):

- the parameters;
- the naked skew at the constraint maturities against the market, with any binding message;
- the free short-end naked skew;
- per pillar, the target and achieved SpotVolCovar and VolVar, the correlations, the first-order
  P1 SSR and the SSR the targets imply.

**Historical mode** (`fit_2f_historical`, not the default; §15 Part 3, step 1). The targets come
from a surface history (`volsto.calibration.history.SurfaceHistory`, §15 Part 2).
$\mathrm{volvol}_{\text{hist}}(T)$ is the annualised standard deviation of daily changes in the
log variance-swap vol of maturity $T$, and $\mathrm{SSR}_{\text{hist}}(T)$ is the historical SSR
of section 7. The targets are

$$
\mathrm{VolVar}_{\text{target}}(T) = \big(\mathrm{atf}_T\,\mathrm{volvol}_{\text{hist}}(T)\big)^2,\qquad
\mathrm{SpotVolCovar}_{\text{target}}(T) = \mathrm{SSR}_{\text{hist}}(T)\,\sigma_0\,\mathcal S^{\text{mkt}}_T
$$

The empirical VolVar is kept in step 3. The marking fit has no standard errors by construction:
its targets carry no sampling error (§10.3).

### 6.7 The shadow rotation

`volsto.risk.shadow_rotation.rotation_shadow_sensitivity` (§15 Part 3) computes, for +1 rota:

- **Usual rotation.** Rotate the surface, hold the P1 set, recalibrate the leverage.
- **Recalibrated rotation.** Rotate the surface, redo steps 0–3 under a policy, recalibrate the
  leverage. The policies are:
  - `sabr_linked`: the break-evens follow the rotated SABR reduction;
  - `sticky_breakeven`: $\mathrm{SpotVolCovar}_{\text{target}}$ and $\mathrm{Corr}_{BE}$ are held.
- **Shadow** = recalibrated − usual, by central differences under common random numbers.

**Measured** on the SPX 2022-12-30 3y autocall at (ssr 1, eps 0.10), with $2\cdot10^5$ paths,
in % of notional per rota. The desk P&L shadow is:

| Policy | Desk P&L shadow |
|---|---|
| `sabr_linked` | −0.0539 ± 0.0066 |
| `sticky_breakeven` | −0.0737 ± 0.0072 |

The fee rises when the marking set follows the skew, which is a loss for the short note (§15
Part 3).

---

## 7. Smile dynamics and the SSR

- **Numerical SSR of a model** (`volsto.analytics.smile_dynamics.ssr_numerical`,
  `ssr_numerical_many`; book §12.4.3; §4.4, §15 Part 1). Bump the initial state jointly to
  $(\ln S_0 + \varepsilon\sigma_0,\ \varepsilon\rho_{SX_1},\ \varepsilon\rho_{SX_2})$, reprice
  the ATMF option under common random numbers, and divide by the model's own skew.
- **Cross-checks.** `ssr_decomposition` implements book eq. 12.52 (with eqs. 12.53–12.54 and
  9.21). `ssr_short_horizon` is an independent regression estimator.
- **Historical SSR** (`SurfaceHistory.ssr_hist`, §15 Part 2). It is the slope of daily ATM-vol
  changes on $\Delta\ln S$, divided by the window's mean skew, with Newey–West errors. The windows
  are 60 and 100 days on the 127-day sample.
- **Raw-slice discriminator** (`volsto.calibration.raw_history.discriminator`). It recomputes
  the historical SSR from local quadratic fits of the raw quotes, with no surface
  parametrisation. It tests whether an SSR below 1 belongs to the quotes or to the fit (§15 M7
  notes, §13.1).

---

## 8. Risk, attribution and hedging

**Risk** (`volsto.risk`, §7):

- `RiskEngine` with the builders `LSVBuilder`, `LVBuilder` and `BSBuilder` prices base and bumped
  states under common random numbers. Every sensitivity is a `Sensitivity(value, stderr)`.
- Surface and parameter bumps recalibrate the leverage through the cache. Spot and factor bumps
  do not.
- **Delta regimes** (`volsto.risk.greeks`, §7.2), with $\Delta = \ln(S_{0,\text{new}}/S_{0,\text{old}})$
  and $s_T$ the ATM skew:

  | Regime | New smile |
  |---|---|
  | `model` | leverage held in spot, factors at zero |
  | `sticky_strike` | $\hat\sigma(k+\Delta)$ |
  | `sticky_moneyness` | $\hat\sigma(k)$ |
  | `sticky_skew` | $\hat\sigma(k) + s_T\Delta$ |
  | `sticky_local_vol` | $\hat\sigma(k) + 2 s_T\Delta$ |

- **Ladders** (`volsto.risk.ladders`): vega by maturity as cumulative "waves" (§7.4),
  forward-variance buckets (§7.5), and skew and curvature tents (§7.6).
- `risk_report` assembles a full report of about 80–100 recalibrations per product (§7.13).

**Attribution** (`volsto.risk.attribution.explain`, §7.12, §7.12.1). A sequential revaluation
under common random numbers, in the order spot, rates, surface, parameters, factors, time. With
`mode="sticky_leverage"`:

- every intermediate state is priced on the start state's leverage, held in forward
  log-moneyness, so there is no calibration between the endpoints;
- the Greeks are model Greeks with the leverage frozen, not recalibrated Greeks;
- the leverage refit appears only in a final `recalibration` bucket;
- the buckets sum to the P&L to $10^{-14}$ (§10.3).

**Hedging** (`volsto.hedging`, §8).

- `Hedger` simulates a world model and rebalances hedges computed under a pricing model, with
  conditional prices by regression (`ConditionalPricer`). Deltas and gammas come from
  history-held bumps under common random numbers.
- The **model-mismatch reserve** of a world is its hedged leakage minus the leakage when the
  world equals the pricing model. The engine's own baseline is not zero (§8.2).
- `delta_regime="min_variance"` hedges with the pricing model's minimum-variance spot delta
  (§8.1).

---

## 9. Known first-order limits of the break-even engine

The P1 break-evens of section 6.3 are first order in the vol of vol. Everything below is
measured and recorded in SPEC. Read a fit's numbers with it.

**1. The engine gate** (§15 Part 3, `scripts/m7_breakeven_gate.py`). The table gives the relative
discrepancy analytic/simulation − 1, as a mean over the grid with the range in brackets. The grid
is Table 8.2 plus a $\rho_{12} = +0.5$ set, flat and sloping curves, $2\cdot10^5$ paths and
$T \in$ {3M, 6M, 1Y, 2Y}.

| $\nu$ | SpotVolCovar | VolVar | Skew (order one) |
|---|---|---|---|
| 0.5 | −0.3 % (−1.0 … +0.3) | +0.1 % (−0.7 … +0.9) | +1.3 % |
| 1.0 | −1.4 % (−2.7 … 0.0) | −1.0 % (−2.7 … +0.9) | +3.3 % |
| 1.74 | −4.7 % (−8.0 … −1.3) | −5.4 % (−8.8 … −2.4) | +7.5 % |
| 2.5 | −10.1 % (−16.5 … −3.8) | −13.2 % (−19.3 … −7.3) | +13.6 % |

- The error grows with $\nu$.
- The short factor carries it: −13 % / −21 % at $\nu = 1.74$ at 1Y / 2Y. The long factor is
  within 2 %.
- In LSV mode, on the cached LSVs, the engine is within ±7 % on SpotVolCovar and ±13 % on VolVar
  at 3M / 1Y. The exception is 1F $\omega = 3$ at 1Y, at −17 % / −28 % (§15 Part 3).

**2. The $\nu$ cap and its warning.** When a fit binds the cap, it attaches the owner's warning,
"nu at cap X; first-order break-even engine ~15% biased beyond ~4; raise only if stage-3
simulation validates" (`NU_CAP_WARNING`, §15 Part 3, decision 2). The cap is `nu_cap` = 3.5,
raised from 2.5 by decision viii.

**3. Stage 3 finds the bias well below $\nu \approx 4$** (§15 Part 3, re-run of 2026-09-15). The
assertion tolerance is **10 %** on the engine bias alone. At $\nu \ge 1.9$ the first-order VolVar
is 17–52 % below the simulation, and the 1Y SpotVolCovar 17–23 %. Verdicts:

| Fit | $\nu$ | Verdict | Engine bias at 3M / 1Y |
|---|---|---|---|
| SPX (1.0, 0.10) | 1.94 | **FAIL** | SVC +10 % / +17 %; VolVar +17 % / +29 % |
| SPX (1.5, 0.05) | 1.39 | **PASS** | SVC ≤ 5.5 %; VolVar ≤ 9.5 %; target misses ≤ 2.6 % |
| placeholder (1.0, 0.10) | 3.50 | **FAIL** | SVC +8 % / +23 %; VolVar +17 % / +52 % |
| placeholder (1.5, 0.05) | 3.50 | **FAIL** | VolVar +9 % / +17 %; SVC +4 % / +8 % |

- For the SPX (1.0, 0.10) fit, the 3M covariance miss of +11 % is reported, not asserted.
- The SPX (1.0, 0.10) mark is the M8b reference mark and the backtest's mark. The backtest
  therefore runs with stage 3 off and marks the first-order SSR (§10.3).
- Whether to lower the threshold or correct the engine's bias is open (§15 Part 3, open question
  c).

**4. The placeholder surface binds at the cap.** On the placeholder (reference) SSVI both owner
pairs bind with $\nu$ at the cap of 3.5 (§15 Part 3, $\nu_{\text{cap}} = 3.5$ re-run):

| Pair | $\nu$ | $\theta$ | $k_1$ | $\rho_{SX_1}$ / $\rho_{SX_2}$ | $\rho_{12}$ | Max first-order SVC miss (3M) | Mean $\lvert L-1\rvert$ |
|---|---|---|---|---|---|---|---|
| (1.0, 0.10) | 3.50 | 0.046 | 9.82 | −1.00 / −0.99 | +0.99 | +97 % | 0.418 |
| (1.5, 0.05) | 3.50 | 0.065 | 13.2 | −1.00 / −1.00 | +1.00 | +31 % | 0.315 |

- Both fits carry the collapse flag.
- The placeholder is unsuitable for marking work: $\nu$ past the cap, $\rho = -1$,
  $\rho_{12} = +1$, $\theta \approx 0.06$ (decision vii). Marking studies use the SPX snapshots,
  and the placeholder stays for the M4/M6 baselines.
- The test suite's toy marking grid (placeholder, $2\cdot10^4$ particles) binds at the cap in
  every fit, with $\rho_{SX_1} \approx -1$ and $\rho_{12} \approx 0.97$–1.0 (§10.2).

**5. First-order SSR versus the calibrated LSV** (§15 Part 3). The first-order P1 SSR
understates the calibrated LSV's numerical SSR by 10–20 % beyond 3M. At `ssr_target = 1` the
calibrated LSV realises SSR 1.13–1.43 on SPX and 1.49–2.32 on the reference surface. SPX (1.0,
0.10), by pillar:

| Pillar | 3M | 6M | 1Y | 2Y | 3Y |
|---|---|---|---|---|---|
| LSV numerical SSR | 1.33 | 1.13 | 1.16 | 1.34 | 1.43 |
| First-order SSR | 1.20 | 0.95 | 0.99 | 1.15 | 1.23 |

The SSR the targets imply is 1.08. `ssr_target` is a dial on the targets, not a realised-SSR
guarantee (decision 1).

**6. Binding at first order is not binding exactly.** The exact (mixing) naked skew is 5–7 points
closer to the market at 1Y than the first-order one. A constraint that binds at first order is
therefore inside $\varepsilon$ exactly (§15 Part 3).

**7. The free short end.** The two-point constraint leaves the naked skew below 1Y free. On SPX
(1.0, 0.10) the first-order short-end gaps are +68 % / +57 % / +33 % at 1M / 3M / 6M. The exact
naked skew gap is +47 % at 3M and +3.8 % at 1Y. The leverage carries the rest: mean $|L-1|$ is
0.156 (§15 Part 3). A tight skew cannot make $|L-1|$ small on a steep short skew (decision 4).

**8. The radicand guard.** It is
$6\,\mathrm{smi}^2 + 3\,\mathrm{atf}\,\mathrm{cvx} \ge 6\,\mathrm{smi}^2(1-c)$
with $c = 0.5$. It never fired on the 127 SPX 2022 H2 snapshots nor on
the placeholder (§15 Part 3, decision iii). With $c > 1/3$ the guard can still produce
$|\mathrm{Corr}_{\text{SABR}}| > 1$, which is then clipped with a flag
(`volsto.calibration.targets`).

**9. Iterating against simulation does not fix a binding fit** (§15 Part 3). On the binding SPX
(1.0, 0.10) fit, `iterate_against_simulation=2` **diverges**:

| Quantity | Iteration 0 → 1 → 2 |
|---|---|
| $\nu$ | 1.94 → 2.08 → 2.28 |
| SpotVolCovar engine bias | 17 % → 27 % |
| VolVar gap vs target | 0.31 → 1.53 |

Meanwhile $\rho$ goes to −1 and $\rho_{12}$ to +1. The loop now stops and keeps the best
iteration. On the interior SPX (1.75, 0.20) fit the first correction overshoots (VolVar gap
18.6 %, $\nu$ 1.24) and the guard reverts to iteration 0.

**10. Identifiability in historical mode** (§15 Part 3, historical recovery; owner's tolerances
$\nu, \theta, k_1$ 10 %, $\rho$ 0.05). On a three-year synthetic history:

- The default two-point configuration misses $\rho_{SX_1}$: −0.664, off by 0.095. The soft
  all-pillar penalty at weight 10 passes, which is why `"auto"` is soft in historical mode.
- On a one-year history the two-point fit gives $\nu$ 1.422 (−18 %) with every SpotVolCovar
  within 2 %. Five covariance targets do not pin $(k_1, \lambda)$ once the short-end skew is free.

**11. The SSR < 1 finding** (2022 H2 SPX; §15 M7 implementation notes, §13.1). The historical SSR
is:

- 0.81–0.84 ± 0.07–0.11 on 60 days at every pillar to 1y;
- 0.86–0.92 on 100 days.

Both are below the floor $R \ge 1$ of local vol and of the two-factor LSV. An `ssr_target` below 1
cannot be met by the model class.

**12. The discriminator verdict: "surface artefact".** The M10 Part 0 re-run on the 127 repaired
eSSVI days (§13.1) gives a fitted 60-day SSR of

| Pillar | 1m | 3m | 6m | 1y |
|---|---|---|---|---|
| Fitted 60-day SSR | 0.874 ± 0.117 | 0.800 ± 0.080 | 0.799 ± 0.069 | 0.834 ± 0.068 |

- At 3m–1y the raw-quote SSR agrees with the fitted one: 0.80 / 0.79 / 0.77 raw against
  0.80 / 0.80 / 0.83 fitted.
- At 1m the raw SSR is 0.942 ± 0.153, and $0.942 + 2\times0.153 = 1.249$ is not below 1.
- By the letter of the owner's gate (every pillar up to 1y), the verdict is therefore
  **"surface artefact"**. The 0.85 SSR is supported at 3m–1y and not established at 1m.
- The hedging study's historical world stays skipped on that verdict (§8.2, §13.1).

**13. The 3y skew on SPX is extrapolated.** On 118 of the 127 sample days the last eSSVI ATM
pillar is at or below 2y. The pillar counts are 5 / 6 / 7 on 1 / 117 / 9 days, and the median
longest quoted maturity is 2.25y. The 3y skew is therefore flat-forward-variance extrapolation
whatever the calendar repair (§13.1, item 3). On SPX the 5Y skew constraint moves to 3Y
(section 6.4), i.e. onto that extrapolation.

**14. No standard errors on a marking fit.** Its targets carry no sampling error, so the fit
has no standard errors and `flag_unidentified` cannot flag a marking parameter series. The
backtest's stability flags come from the historical-mode rolling fit instead (§10.3, §15 Part 4).

---

## 10. Other known limits

**Surface fit.**

- `fit_ssvi` clamps $\theta_T$ beyond the last pillar in its residuals, while the surface it
  returns extrapolates. The reported fit errors therefore describe a slightly different surface:
  median RMS 0.1957 vp reported against 0.1384 vp on the returned surface ($|k| \le 0.2$, 3m–3y).
  The fix is deferred because it would change every fitted snapshot and cache key (§13.1).
- The 1–2 month weeklies of high-volatility days are 1–7 vp off under a single power-law
  $\varphi$ (§13).

**Ladders on eSSVI.** The curvature ladder's butterfly bump needs more than the default 4
halvings on 62 of the 127 eSSVI days (5 on 51, 6 on 11). A ladders risk run on eSSVI surfaces
uses `max_halvings=6` (§13.2, §7.12.1).

**Particle calibration.** The far-wing bias of section 5 applies. SPEC §4.2 also records, from
before M4, that the 2F far right tail at 1m (+20 %, 3.4 standard deviations, a sub-basis-point
option) is 1.4 vp rich, because $E[\xi \mid S]$ is extrapolated beyond the particle cloud there.

**Sticky-leverage attribution.** Its Greeks miss the local-vol part of a surface bump. The
recalibrated vega of a 1y study cliquet ladder is 4× the sticky one (§7.13, §7.12.1).

**Backtest sample.** The 2022 H2 sample is one regime and 127 days. Studies on it are proofs of
concept (§10.3).

**Backtest P&L sign.** SPEC §10.3 defines the P&L of $(d-1, d]$ as $V(d) - V(d-1)$ + the flows
dated $d$, i.e. the change in the trade's value. It does not apply the desk-short sign used by
the hedging studies and the shadow rotation. Negate it for a desk that is short the book.

---

## 11. Entry points

| Concept | Entry point |
|---|---|
| Surface from a config; eSSVI | `volsto.market.surface.surface_from_config`, `ESSVISurface` |
| Calendar certificate and repair | `volsto.market.surface.certify_calendar`, `volsto.market.import_hdn.repair_calendar` |
| Local vol | `volsto.market.dupire.LocalVolSurface.from_implied`, `volsto.models.LocalVol` |
| Variance swaps, $\xi_0$ | `volsto.market.varswap.varswap_strike`, `xi0_curve` |
| Two-factor kernel | `volsto.models.bergomi.BergomiSV`, `volsto.config.BergomiParams` |
| LSV, leverage | `volsto.models.lsv.LSV`, `volsto.models.leverage.LeverageFunction` |
| Particle calibration | `volsto.calibration.particle.calibrate_leverage` (via the cache only) |
| Leverage cache | `volsto.calibration.cache.LeverageCache.get_or_calibrate`, `build_market` |
| Repricing diagnostics | `volsto.calibration.diagnostics.reprice_surface` |
| Step 0 and step 1 | `volsto.calibration.targets.sabr_reduce`, `marking_targets`, `historical_targets` |
| Break-even engine | `volsto.analytics.breakeven.first_order_breakevens`, `simulated_breakevens` |
| Reparametrisation | `volsto.analytics.reparam.to_breakeven`, `from_breakeven` |
| Marking fit (steps 2–3) | `volsto.calibration.fit_2f.fit_2f_marking`, `BreakEvenFitConfig`, `FitResult` |
| Stage 3 | `volsto.calibration.fit_2f.stage3_validation`, `breakeven_check` |
| Historical fit, stability | `fit_2f_historical`, `volsto.calibration.stability.rolling_fit`, `flag_unidentified` |
| SSR (model) | `volsto.analytics.smile_dynamics.ssr_numerical`, `ssr_decomposition` |
| SSR (history) | `volsto.calibration.history.SurfaceHistory`, `estimate_history` |
| Discriminator | `volsto.calibration.raw_history.discriminator` |
| Shadow rotation | `volsto.risk.shadow_rotation.rotation_shadow_sensitivity`, `ROTATION_CONVENTION` |
| Risk | `volsto.risk.RiskEngine`, `LSVBuilder`, `delta_table`, `risk_report` |
| Attribution | `volsto.risk.attribution.explain` |
| Seasoned trades | `volsto.products.seasoning.season`, `replay` |
| Hedging | `volsto.hedging.Hedger`, `volsto.studies.m8b` |
| Grid precompute, viewer | `volsto-precompute`, `volsto-viewer` (`volsto.viewers`) |
| Studies, backtest | `volsto-study`, `volsto-backtest` (see [studies.md](studies.md)) |
