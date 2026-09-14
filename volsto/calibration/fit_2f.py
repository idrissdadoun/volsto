"""Break-even fit of the two-factor Bergomi model with a **soft skew penalty** (SPEC §15 Part 3,
M7 addendum and the owner's M7 Part 3 redesign; Bergomi ch. 7 §7.4, ch. 8 eq. 8.54, ch. 9
§9.2, ch. 12 eq. 12.52, in the break-even parametrisation of :mod:`volsto.analytics.reparam`).

**Motivation (owner).**  Forcing the naked skew to equal the market skew exactly pins
``(λ1, λ2)``, which pins the spot/vol covariance, which pins the SSR — leaving no freedom to
target the SSR, and driving the leverage to ``L ≡ 1`` (an inert local-vol layer, defeating the
LSV).  Making the skew a *soft penalty* frees the SSR **and** gives the leverage its proper job:
the stochastic-volatility kernel carries the dynamics (SSR, vol of vol), the leverage absorbs the
residual skew mismatch so that the surface still reprices.  The four hard skew inequalities of
the previous design (``eps_skew``, ``skew_guard``) are deleted.  *Note:* the owner referred to a
"gamma-multiplier approach from the last message" — no gamma multiplier exists in the committed
code (commit adb673a), so there was nothing to drop.

**Targets** (:mod:`volsto.calibration.targets`, absolute vol units): ``correl_target(T) =
ρ_SABR(T)``; ``ssr_target(T)`` a user input, scalar or curve, default 1.0; ``SpotVolCovar_target
= ssr_target σ_0 Skew_market`` (``σ_0 = atf(1M)``); ``VolVar`` from the vol of vol backed out as
before (absolute ``VoV_SABR = ν_SABR atf``).

**Premise warning — the leverage is not dynamically neutral** (book eq. 12.52): ``R_LSV =
R^LV(Mkt) + (S^SV_T / S_T) [R^SV − R^LV(SV)]``.  Absorbing skew in the leverage *adds* local-vol
spot/vol covariance, so the calibrated LSV's SSR is not the naked kernel's.  On the reference
SSVI the order-one ``R^LV(Mkt)`` is 3.01 / 3.01 / 3.03 at 3M / 6M / 1Y (pure-Dupire Monte Carlo
2.53 / 2.61 / 2.65) and the four cached LSVs (1F ω = 1, 2, 3; 2F Table 8.2) show a numerical SSR
of 2.2–2.7 at 3M–1Y whatever the kernel, while their naked kernels' SSR is 1.4–1.9.  Hence two
measures of "the SSR the fit achieves" (``ssr_measure``), affine in ``λ`` at fixed ``(k1, k2)``::

    naked:  SVC_model(λ) = c ½ pref (λ1 A1 + λ2 A2)                                  (eq. 7.38)
    lsv:    SVC_model(λ) = σ_0 S_T R^LV(Mkt)_T + c ½ pref λ·A − σ_0 [λ·J_T + λ·I_T]  (eq. 12.52)
            R^LV(Mkt)_T = 1 + (1/T) ∫₀ᵀ f(t) S^mkt_t dt / S^mkt_T,  I_T = (1/T) ∫₀ᵀ f(t) J_t dt
    achieved SSR = SVC_model / (σ_0 S^mkt_T),  Skew_naked = λ·J_T                    (eq. 8.54)

with ``J_t`` the kernel's order-one skew at maturity ``t`` for every ``t ∈ (0, T]``, ``f(t) =
σ²(t)/(σ̂_t σ̂_T)`` on the market ATMF curve (``term_structure="atmf"``, book p. 475; ``"flat"``
and ``"vs"`` change the SSR by 1–2%), ``pref = atf_T`` and ``c = 1`` under the ``"market"``
prefactor, ``pref = σ̂_VS,T`` and ``c = σ_0/sqrt(ξ₀(0))`` under ``"model"`` (the book pair of
``R^SV``).  The lsv form reduces to the naked one when the market skew term structure is the
kernel's own, and to ``σ_0 S R^LV(Mkt)`` at ``λ = 0`` (``test_lsv_measure_identities``).  The time
integrals use ``t = T u^p`` with Gauss-Legendre in ``u`` (``n_ts = 64``), ``p = clip(1/(1 − γ), 2,
4)`` from the market skew's short-end exponent ``γ`` (:func:`ts_substitution_power`: ``p = 2`` on
the reference SSVI and SPX, ``p = 4`` on a historical power-law extension at its clip ``γ =
0.75``, where ``p = 2`` loses 3e-2 on ``R^LV``); the defaults are converged to about 6e-4 on
``R^LV(Mkt)`` (the error decays as ``n_ts^−2`` from the weekly knots of the ATMF curve in ``f``;
``test_term_structure_quadrature``).

**SSR measure** (``ssr_measure``, default ``"auto"``, :func:`resolve_ssr_measure`): ``"lsv"`` on
targets with a surface skew term structure (marking mode), ``"naked"`` otherwise (historical
mode; below).  Evidence for lsv in marking mode: against the numerical LSV SSR of the four cached
LSVs (ν ≤ 1.74, naked/market skew ratio about 1) at 20 maturity points (3M–3Y) the lsv prediction
is within 8.3% RMS (worst +15%, 1F ω = 1 at 3M; 2F Table 8.2 within −8.4% to +6.6%), whereas the
naked measure is off by −40% to −84% (it cannot hit an LSV SSR target).  D1 of the design notes
carried an extra factor ``σ_0/σ_0^SV`` on the naked covariance: with the market-ATMF prefactor it
double counts the level (10.8% RMS, worst −23%) and is dropped (``c = 1``); pairing ``atf_T`` with
``sqrt(ξ₀(0))`` is wrong by up to −14%.  ``lv_ssr="numerical"`` (V2) replaces ``R^LV(Mkt)`` by a
pure-Dupire Monte Carlo (:func:`local_vol_ssr_numerical`; use 2·10⁵ paths and ``dt ≤ 1/400``) and
scales ``I_T`` by ``c_T = (R^LV_num − 1)/(R^LV_o1 − 1)`` (0.76 / 0.80 / 0.82 at 3M / 6M / 1Y): 3.7%
mean, 7.9% max error on the cached LSVs; the default stays ``"order1"``.  Every fit reports
**both** achieved SSRs (``ssr_achieved_lsv``, ``ssr_achieved_naked_vs_market``) and the naked
kernel's own first-order SSR ``svc_naked / (σ_0 Skew_naked)`` (``ssr_naked_own``, −8% to +1%
against its numerical SSR); stage 3's numerical LSV SSR is the truth.

**First-order accuracy — every SSR, floor and message value of this module is a first-order
estimate.**  The validation above holds at ν ≤ 1.74 and a skew ratio about 1.  At the operating
points the earlier default (``ν_cap = 5``) selected (ν 4–5, ``k1`` 17–20), stage 3 (review; 2·10⁵
particles, 10⁵ pricing paths, standard errors about 0.02) measured the lsv prediction **below**
the calibrated LSV's numerical SSR at 3M / 6M / 1Y / 2Y: weight 1, target 1 (ν 5, skew ratio
1.37) 1.346 / 1.051 / 1.014 / 1.092 against 1.707 / 1.442 / 1.285 / 1.290 (−21% / −27% / −21% /
−15%); weight 100, target 1 1.739 / 1.568 / 1.548 / 1.565 against 1.939 / 1.763 / 1.732 / 1.801
(−10% to −13%); weight 0.1, target 1.6 1.730 / 1.561 / 1.548 / 1.580 against 1.924 / 1.742 /
1.707 / 1.772 (−9% to −11%).  The pieces of eq. 12.52 were not the error (numerical ``R^SV``,
skew and ``R^LV`` in the same formula give 0.94 at 3M against 1.71): the expansion linear in
``S^SV − S^mkt`` breaks when the leverage is strongly counter-skewed (actual mean ``|L − 1|``
0.368).  The D5 VolVar model was low too (3M 0.0449 against 0.0669 ± 0.0015).  On SPX 2022-12-30
the lsv prediction was 3–5% low for the skew-tight fit and 5–14% low for the unclamped fits.
Hence the default ``nu_cap = nu_flag = 2.5`` and :attr:`FirstFit.first_order_valid` (``ν_min ≤
nu_flag`` and skew ratio ``≤`` :data:`FIRST_ORDER_SKEW_RATIO_LIMIT` = 1.2), with a note on every
fit outside that regime (:func:`first_order_regime_note`).  The cap does **not** make the
first-order SSRs accurate, it keeps them less wrong: the trade-off study at ``ν_cap = 2.5``
(``scripts/m7_skew_tradeoff.py``, eight fits on the reference SSVI, every one with the ν limit
binding, 2·10⁵ particles recalibrated, 10⁵ pricing paths, standard errors 0.018–0.029; 1546 s)
measured the lsv prediction against the numerical LSV SSR at 3M / 6M / 1Y / 2Y / 3Y within
+1% / −7% / −8% / −11% / −14% for the skew-tight fits (ratio about 1.03; e.g. weight 100, target
1: 2.249 / 1.898 / 1.702 / 1.617 / 1.574 against 2.231 / 2.041 / 1.861 / 1.831 / 1.823) and −1% /
−13% / −19% / −18% / −14% at skew ratio 1.21 (weights 1 and 0.1, target 1: 2.299 / 1.770 / 1.334
/ 1.149 / 1.139 against 2.328 / 2.037 / 1.654 / 1.408 / 1.327).  The first-order values under-state
the LSV SSR beyond 3M by about 10–20% in this regime; stage 3 remains the truth.

**First minimisation** — ``(k1, λ1, λ2)``, ``k2`` fixed (0.2), the owner's penalised least squares
in dimensionless units::

    min  Σ_i wc_i (SSR_model_i(k1, λ) − ssr_target_i)²  +  w_skew Σ_i (Skew_naked_i / S^mkt_i − 1)²

(``SSR_model_i = SVC_model_i / (σ_0 S^mkt_i)``: the covariance residual normalised by ``σ_0
|S^mkt_i|``, ``wc_i`` uniform by default).  The design note D2 normalised by ``|SVC_target_i|``
instead, which makes the SSR weight scale as ``1/ssr_target²`` (a 7.6x change of the dial across
0.8–2.2); the fixed normalisation keeps ``skew_weight`` a scale-free dial at every target
(``weights_ssr="relative"`` restores D2's form).  Both residuals are affine in ``λ`` at fixed
``k1``: the inner problem is a stacked 2-D least squares solved exactly (:func:`_qp2`, candidate
enumeration) on the ν-feasibility polytope ``|λ1| + |λ2| ≤ 2 ν_cap`` (4 rows): the smallest ``ν``
compatible with ``λ`` is ``|λ1 + λ2|/2`` (``Cov(W^S, ω1 X1 + ω2 X2) = λ1 + λ2 ≤ 2ν``), equal to
``(|λ1| + |λ2|)/2`` for same-sign loadings — every fit measured — and the polytope is conservative
for opposite signs (it excludes cancelling loadings).  The outer ``k1`` runs on a coarse
geometric grid then a bounded scalar refinement; the ``k1`` profile is reported
(:func:`k1_profile`).  **Standard errors** (:func:`_first_stderr`): in marking mode none (NaN with
a note: the targets are not noisy observations and the penalised objective, which scales with the
arbitrary ``w_skew``, is not a residual variance); in historical mode the sandwich ``(JᵀJ)⁻¹ Jᵀ Σ J
(JᵀJ)⁻¹`` on the stacked residuals with ``J = [∂r/∂k1, ∂r/∂λ]`` and ``Σ`` the SSR target variances
on the SSR rows, 0 on the skew rows (the pricing date's skew is observed); the second fit likewise
(VolVar standard errors in historical mode, NaN in marking mode).  ``skew_weight`` (default 1.0)
is the owner's skew/leverage dial.  At ``ssr_target = 1`` on the reference SSVI (pillars 3M–3Y,
unclamped) the weights 100 / 10 / 1 / 0.1 give: lsv achieved SSR 1.808 / 1.694 / 1.538 / 1.538,
mean naked-skew gap 0.019 / 0.072 / 0.170 / 0.170 (the ν limit binds at every weight, so 1 and
0.1 give the same fit); naked 1.405 / 1.334 / 1.128 / 1.020, gap 0.014 / 0.048 / 0.192 / 0.299.
Flags: ``k1`` at a bound, the ν limit binding, ``ν > nu_flag``, the first-order regime.

**Attainable floor / ceiling and the clamp** (:func:`attainable_ssr`).  The first minimisation
is run at the target curve ``r · shape`` for every ``r`` of ``np.arange(0, 3.01, 0.1)`` and the
request (``shape`` the request normalised by its mean, 1 for a constant target — scalar and
curve targets share one definition).  The **floor / ceiling are the lowest / highest mean
achieved SSR over the scan** (achieved-SSR units, the ``Y`` of the message), with the scan target
attaining them, the binding limits (``k1`` bound, ν limit) and a scan-edge flag when the extreme
sits at the end of the scan with the achieved SSR still moving by more than 0.1 per unit of
target (then the extreme belongs to the scan range, not to the model).  Tracking segments (mean,
and every pillar, within ``ssr_tol``) and ``k1`` basin jumps between neighbouring scan targets
are reported separately; nothing is bisected or assumed contiguous.  The previous definition —
the smallest target tracked within ``ssr_tol`` — was a shrinkage threshold of the penalised fit,
not a floor: lower SSRs were attainable below it, the clamp raised the achieved SSR (1.130 →
1.273) and made the SSR dial inert (targets 0.6–1.2 returned one model).  Since the achieved SSR
is close to affine in ``r`` with a slope set by the weight, the floor is where the limits
saturate it — with ``ν_cap = 2.5`` the lsv floors on the reference SSVI are 1.779 / 1.508 / 1.408
/ 1.352 at weights 100 / 10 / 1 / 0.1 (ν limit binding at each; ceilings 1.963 / 2.330 / 2.905 /
2.991), naked 1.379 / 1.131 / 0.439 / 0.067 (ceilings 1.457 / 1.742 / 2.523 / 2.903; the naked
band at weight 100 is interior and saturated); SPX 2022-12-30 (1M–1Y) lsv 1.801 / 1.203 / 0.720
/ 0.668, naked 1.728 / 1.240 / 0.391 / 0.053.  :func:`fit_2f` always attaches the band and a
tracking detail (achieved SSR, tracking error, worst pillar).  When the mean request ``X`` lies
below the floor (above the ceiling) the fit refits the first minimisation at the scan target that
attains it, builds the VolVar target at ``Y`` (a scan target of 0 would ask for no vol of vol) and
sets :attr:`FitResult.message` to the owner's text exactly, ``"ssr_target=X below attainable
floor Y at skew_weight=W; fitted at Y. Lower the skew weight to reach lower SSR (naked skew will
diverge further from market, leverage will do more)."`` (``X``, ``Y`` to 3 decimals, ``W`` to 3
significant digits positional, :func:`format_skew_weight`); ``Y`` equals
:attr:`AttainableSSR.floor` and the refit's achieved SSR, and the clamp never moves the achieved
SSR away from ``X`` (the scan includes ``X``).  With ``clamp_to_attainable=False`` the fit stays at
``X`` and the message says so (:data:`FLOOR_MESSAGE_UNCLAMPED`).  Reference SSVI at the defaults
and ``ssr_target = 1``: "ssr_target=1.000 below attainable floor 1.408 at skew_weight=1; fitted at
1.408. …" (unclamped 1.538).

**Second minimisation** — ``(ω1, ω2, χ)`` on ``VolVar`` with ``(k1, k2, λ1, λ2)`` frozen::

    VolVar_i = SensiX_i² + SensiY_i² + 2 ρ_XY SensiX_i SensiY_i
               [+ SensiSpot_i² + 2 SensiSpot_i ½ pref λ·A_i   under the lsv measure]

with ``SensiX_i = ½ ω1 A1_i pref``, ``ρ_Si = λ_i/ω_i``, ``ρ_XY = ρ_SX ρ_SY + χ sqrt(1−ρ_SX²)
sqrt(1−ρ_SY²)`` and ``SensiSpot_i = (SVC_lsv_i − SVC_naked_i) / c = σ_0 [S_T + I^mkt_T − λ·J_T −
λ·I_T] / c`` the local-vol spot sensitivity of eq. 12.52 in ``SensiX`` units, so that the
decomposition implies ``c (SensiSpot + Σ ρ_Si SensiX_i) = SVC_model`` under both prefactors
(:meth:`MeasureMaps.implied_covariance`, ``test_volvar_decomposition_consistent``; D5: a
consistency choice; stage 3's simulated VolVar of the LSV is the check; at order one it overshoots
the numerical spot bump by +10–31%, V2 brings it to −4..+8%).  ``volvar_target="achieved"``
(default, marking mode) rebuilds the target from the *achieved* covariance, ``VolVar_target_i =
(SVC_model_i atf_i A_i / (σ_0 ρ_SABR_i))²`` (it reduces to the requested ``(½ ssr atf ν_SABR A)²``
when the covariance is on target; historical mode keeps the requested target with a note).
Bounds ``ω_i ≥ |λ_i|``, ``ω_i ≤ omega_max``, ``χ ∈ [−0.99, 0.99]``, a soft penalty above ``ν_cap``;
``scipy.optimize.least_squares`` from nine starts; the book parameters follow from the inverse
reparametrisation and the paired risk regime is ``sticky_strike``.  On the reference SSVI most
fits put ``χ`` on a bound (reported in ``bound_flags``).  The per-pillar table reports
``correl_target`` and ``correl_implied = SVC_model atf A / (σ_0 sqrt(VolVar_model))`` and the
absolute ``volvol_target`` / ``volvol_model``.

**Prefactor convention** (``sigma_hat_prefactor``, unchanged): the SV skew is eq. 8.54 at the
kernel's order-zero VS vol under both conventions (the gate measured it within 6–7% of the
mixing skew at ν = 1.74); the sensitivities take the market ATMF vol (``"market"``, default) or
the kernel's VS vol (``"model"``, with ``c = σ_0/sqrt(ξ₀(0))`` above, 0.85 on the reference SSVI).

**Historical mode** has no skew term structure below the first pillar: the lsv measure would
integrate a power-law interpolation of the pillar skews, which overshoots the flattening Bergomi
skew below 1M.  On the three-year mixing history (k2 = 0.28, weight 1) lsv gives ν 1.732, θ 0.270,
k1 7.20, ρ −0.920 / −0.573 (outside the owner's tolerances; a flat extension beyond the pillars
fixes ρ, −0.735 / −0.489, but not ν 1.528 and k1 4.68) against ν 1.739, θ 0.252, k1 5.71, ρ −0.710 /
−0.487 when fed the kernel's own skew shape.  The ``"auto"`` measure is therefore **naked** in
historical mode (the recovery tests assert on the default): weight 1 gives ν 1.770, θ 0.248, k1
5.58, ρ_SX1 −0.728, ρ_SX2 −0.477, ρ12 +0.11 (``"market"``; true 1.74, 0.245, 5.35, −0.759,
−0.487, 0 — inside the owner's 10% / 0.05) and ν 1.687, θ 0.214, k1 5.98, ρ −0.795 / −0.496
(``"model"``); on the one-year order-one history ν 1.656, θ 0.240, k1 5.34, ρ −0.814 / −0.518
(explicit lsv: ν 1.544, θ 0.257, k1 5.61, ρ −0.880 / −0.578).  Weight 10 already moves k1 to 5.93
(outside 10%) on the three-year history.

**Leverage diagnostics without calibration** (:func:`leverage_proxy`): ``L²(t, k) = σ²_Dup(t, k)
/ E[V_t | k]`` with the convexity-adjusted Gaussian regression ``ln E[V_t | k] = ln ξ₀(t) + β_t (k
+ W_t/2) − β_t² W_t/2``, ``β_t = C_t/W_t``, ``C_t = Σ_i λ_i ∫₀ᵗ sqrt(ξ₀(u)) e^{−VarY(u)/8}
e^{−k_i(t−u)} du`` (``E[sqrt V] = sqrt(ξ₀) e^{−Var(ln V)/8}``), averaged like stage 3 (mean of
``|L − 1|`` over ``|k| ≤ 2 atf(t) sqrt(t)`` per time of the calibration grid, then over times).
Against the cached reference leverages (8·10⁵ particles, ν ≤ 1.74): actual 0.240 / 0.151 / 0.110
/ 0.112, proxy 0.256 / 0.170 / 0.140 / 0.132 (+6 / +12 / +28 / +18%) for 1F ω = 1, 2, 3 and 2F
Table 8.2.  Against calibrated fits at ν = 5 (review, 2·10⁵ particles) it failed: actual 0.368 /
0.074 / 0.074 against proxy 0.173 / 0.135 / 0.141 (−53% at skew ratio 1.37, +82%, +91%), i.e. a
proxy spread of 0.135–0.173 for an actual spread of 0.074–0.368, and the skew-tight calibrated
leverage (0.074) lies well below the proxy's apparent floor of about 0.13 — that floor is an
artefact of the proxy, not a property of near-pure-SV leverages.  The proxy is therefore flagged
invalid above a skew ratio of 1.2 (:data:`PROXY_SKEW_RATIO_LIMIT`) or ``ν > nu_flag``, is printed
and never asserted on; the tests assert ``|L − 1|`` on the calibrated study leverages.  The
skew-only form of design note D6 (local slope ``2 S + T S'``) was measured at +40–70% and is not
shipped.  :func:`mean_abs_leverage_deviation` is the same statistic on a calibrated leverage.

**Trade-off study** (:func:`skew_weight_tradeoff`): the fit at one ``ssr_target`` for several
skew weights, tabulating the first-order achieved SSR under both measures, the naked-skew gap,
the attainable floor / ceiling, the leverage proxy and — with stage-3 inputs (cached leverages
via ``model=``, or a calibration per weight in a script) — the actual mean ``|L − 1|`` and the
numerical LSV SSR.  ``scripts/m7_skew_tradeoff.py`` writes each fitted parameter set as a
:class:`~volsto.config.CalibrationSpec` YAML with the fit provenance (:func:`write_tradeoff_spec`,
``configs/studies/m7_skew_tradeoff/``), calibrates it into the cache (plus a second particle seed
:data:`TRADEOFF_NOISE_SEED` at the tightest and loosest weight for the particle noise) and prints
the numerical LSV SSR next to each floor; tests read them back with :func:`load_tradeoff_specs`
and ``allow_calibrate=False``.  Measured (``ν_cap = 2.5``; |L − 1| with the two-seed particle
noise 0.0015–0.0029): at target 1 weights 100 / 10 / 1 / 0.1 give mean ``|L − 1|`` 0.201 / 0.241 /
0.286 / 0.286 (weights 1 and 0.1 are the same fit) against the proxy 0.135 / 0.147 / 0.180 / 0.180;
at target 1.5, 0.196 / 0.205 / 0.209 / 0.224 against 0.134 / 0.137 / 0.144 / 0.186.  The leverage
grows as the skew weight loosens, but even the skew-tight fits keep ``|L − 1|`` about 0.20: the
near-pure-SV limit is not ``L ≈ 1`` on this surface.

**Stage 3 — validation, nothing refit** (:func:`stage3_validation`): the leverage is calibrated
on the fitted parameters (or a ``model=`` override is used without calibrating) and the report
carries (a) the mean ``|L − 1|``, (b) the LSV's numerical SSR (state bump,
:func:`~volsto.analytics.smile_dynamics.ssr_numerical_many`) against ``ssr_target`` and the fit's
first-order ``ssr_achieved_lsv`` / ``ssr_achieved_naked_vs_market``, (c) the naked kernel's skew
by the exact mixing derivative against the market skew and the first-order naked skew, (d) the
LSV's ``SpotVolCovar`` and ``VolVar`` by simulation with standard errors and z-scores, (e) the
forward ATM vol and 90/110 skew, optionally the M6 headline table.  Every report states its wall
clock and whether it recalibrated.

**Conventions to confirm (owner).**

1. *Floor definition and clamp.*  The floor is the lowest first-order mean SSR over the scan
   (targets 0–3), refit at the scan target attaining it.  The owner's "where the objective stops
   improving / bounds bind" is met where a limit saturates the achieved SSR (every lsv floor
   above); where nothing saturates (naked at weights 1 / 0.1) the floor is the value at scan target
   0 and is flagged as a scan-edge artefact.  Alternatives: invert ``r → achieved`` so the achieved
   SSR hits ``X`` exactly whenever attainable (changes the meaning of the owner's target), or keep
   the clamp off by default.  ``clamp_to_attainable`` stays True since the clamp can no longer raise
   the SSR or freeze the dial.
2. *Ceiling message wording* (:data:`CEILING_MESSAGE`) is the implementer's and neutral: under the
   previous definition the ceilings were not monotone in the weight (1.893 at 100, 1.802 at 10).
   Under the present one the measured ceilings rise as the weight loosens (reference lsv 1.963 /
   2.330 / 2.905 / 2.991, naked 1.457 / 1.742 / 2.523 / 2.903; SPX lsv 1.975 / 2.309 / 2.830 /
   2.962), so an owner-symmetric "Lower the skew weight to reach higher SSR" would hold there, but
   it is not guaranteed and the loose ceilings sit at the scan edge.
3. *``nu_cap`` default 2.5* (was 5): at ν 5 the first-order SSRs under-state the LSV SSR by
   10–27%, at 2.5 by up to about 20% beyond 3M (tables above) — less wrong, not validated.  With
   2.5 an LSV SSR near 1 is not attainable on the reference SSVI (first-order floor 1.35–1.78,
   numerical LSV SSR of the study fits 1.33–2.33); ``nu_cap=5`` reaches lower first-order values
   that stage 3 does not confirm (numerical 1.29–1.71 for the target-1 fit at weight 1).
4. *``volvar_target`` default ``"achieved"``* (the owner asked for "VolVar via the vovol backed
   out as before", i.e. ``"requested"``).  Ratio achieved / requested VolVar target at target 1,
   weight 1, clamp off (review, ``ν_cap = 5``): lsv 1.811 / 1.104 / 1.028 / 1.192 / 1.320 at 3M–3Y,
   naked 1.578 / 1.221 / 1.109 / 1.170 / 1.269; the correlation implied by the fit is still not
   ``ρ_SABR`` (−0.929 … −0.776 against −0.808), since three parameters fit five VolVars.  With
   ``"requested"`` and a missed SSR target the second fit puts ``ω`` on ``ω_i = |λ_i|``
   (``|ρ_Si| = 1``; measured on every lsv weight at target 1 with ``ν_cap = 2.5``).  Kept as is
   pending the owner.
5. *``ssr_measure="auto"``* resolves to naked in historical mode (above); an explicit ``"lsv"``
   there is allowed and noted.

Checked by ``tests/test_fit_2f.py`` (affine maps against the engine, the exact QP and the ν
polytope, the term-structure quadrature, the lsv-measure identities, the VolVar decomposition,
the floor / ceiling messages, the attainable band, SSR tracking at a loose weight, the tight
weight, the monotone trade-off, the study specs and cached study leverages, the leverage proxy
against the cached leverages, historical recovery with the default measure, the rolling fit,
stage 3 on the cached Table 8.2 LSV).
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from numpy.typing import NDArray
from scipy.optimize import least_squares, minimize_scalar

from volsto.analytics.breakeven import Kernels, _gl
from volsto.analytics.reparam import BreakEvenParams, to_breakeven
from volsto.calibration.history import WINDOW_SSR, WINDOW_VOL, SurfaceHistory
from volsto.calibration.targets import (
    DEFAULT_TARGET_PILLARS,
    SABR_CURVATURE_H,
    TargetSet,
    historical_targets,
    marking_targets,
)
from volsto.config import BergomiParams, from_mapping, to_mapping
from volsto.market.varswap import ForwardVarianceCurve, xi0_curve

log = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

#: the fixed slow mean reversion (owner default; Table 8.2 has 0.28)
DEFAULT_K2 = 0.2
#: weight of the soft residual ``(ν − ν_cap)⁺`` in the second minimisation (per unit of ν;
#: with relative weights a 0.1 excess in ν costs as much as a 100% relative VolVar error)
NU_PENALTY_WEIGHT = 10.0
#: the risk regime paired with a break-even fit (SPEC §15 Part 3)
RISK_REGIME = "sticky_strike"
WEIGHT_KINDS = ("uniform", "relative")
PREFACTOR_KINDS = ("market", "model")
#: the SSR the first minimisation targets (module docstring): the calibrated LSV's first-order
#: SSR (eq. 12.52, default) or the naked kernel's spot/vol covariance over the market skew
SSR_MEASURES = ("lsv", "naked")
#: the accepted values of ``BreakEvenFitConfig.ssr_measure``: ``"auto"`` resolves to ``"lsv"``
#: on targets with a surface skew term structure (marking mode) and to ``"naked"`` otherwise
#: (historical mode; module docstring, :func:`resolve_ssr_measure`)
SSR_MEASURE_CHOICES = ("auto", *SSR_MEASURES)
#: term-structure factor ``f(t)`` of the eq. 12.52 integrals
TERM_STRUCTURE_KINDS = ("atmf", "flat", "vs")
#: the VolVar target of the second minimisation: rebuilt from the achieved SpotVolCovar so that
#: the SpotVolCovar/VolVar pair keeps ``correl = ρ_SABR`` (marking mode, default; owner decision
#: pending, module docstring "Conventions to confirm"), or the requested one (the vol of vol backed
#: out of the requested SSR)
VOLVAR_TARGET_KINDS = ("achieved", "requested")
#: ``R^LV(Mkt)`` of the lsv measure: order one, or a pure-Dupire Monte Carlo (V2 correction)
LV_SSR_KINDS = ("order1", "numerical")
DEFAULT_SKEW_WEIGHT = 1.0
DEFAULT_SSR_TOL = 0.05
#: the ν level above which the first-order engine and the lsv measure are outside their measured
#: accuracy (module docstring); also the default hard ν limit of the first minimisation
DEFAULT_NU_FLAG = 2.5
#: the scan of :func:`attainable_ssr` (mean target of the scanned curve)
DEFAULT_SSR_GRID: tuple[float, ...] = tuple(
    float(x) for x in np.round(np.arange(0.0, 3.01, 0.1), 10)
)
#: a scan edge counts as saturated when the mean achieved SSR moves by less than this per unit of
#: target between the edge and its neighbour (:func:`attainable_ssr`)
DEFAULT_SATURATION_SLOPE = 0.1
#: neighbouring scan rows whose ``k1`` differ by more than this factor are reported as a basin
#: jump (:func:`attainable_ssr`)
K1_BASIN_JUMP_RATIO = 2.0
#: the trade-off sweep, skew-tight to skew-loose
DEFAULT_TRADEOFF_WEIGHTS: tuple[float, ...] = (100.0, 10.0, 1.0, 0.1)
#: the second particle seed of the study's particle-noise calibrations (the tightest and loosest
#: weight recalibrated with it; ``scripts/m7_skew_tradeoff.py --noise``, read by the tests)
TRADEOFF_NOISE_SEED = 54321
#: above this naked-to-market skew ratio the first-order SSRs and the leverage proxy are flagged
#: (module docstring: measured breakdown of the linear eq. 12.52 expansion at ratio 1.37)
FIRST_ORDER_SKEW_RATIO_LIMIT = 1.2
#: the leverage proxy is flagged invalid above this naked-to-market skew ratio
PROXY_SKEW_RATIO_LIMIT = FIRST_ORDER_SKEW_RATIO_LIMIT
#: tolerance on ``ν − nu_limit`` before the proxy is flagged (the second fit's soft ν cap
#: overshoots by about 0.01)
PROXY_NU_TOL = 0.05
#: standard errors above this are reported as NaN (numerically unidentified; :mod:`stability`)
MAX_FINITE_SE = 1e3
#: the owner's floor message (M7 Part 3 redesign), filled with X, Y (3 decimals) and W (3
#: significant digits, positional: :func:`format_skew_weight`)
FLOOR_MESSAGE = (
    "ssr_target={x:.3f} below attainable floor {y:.3f} at skew_weight={w}; fitted at {y:.3f}. "
    "Lower the skew weight to reach lower SSR (naked skew will diverge further from market, "
    "leverage will do more)."
)
#: the ceiling message (implementer's wording, neutral: the measured ceiling is not monotone in
#: the skew weight; owner sign-off pending, module docstring "Conventions to confirm")
CEILING_MESSAGE = (
    "ssr_target={x:.3f} above attainable ceiling {y:.3f} at skew_weight={w}; fitted at {y:.3f}. "
    "The ceiling is not monotone in the skew weight: see attainable_ssr for the band at other "
    "weights."
)
#: the variants when ``clamp_to_attainable=False`` (the fit is kept at X)
FLOOR_MESSAGE_UNCLAMPED = (
    "ssr_target={x:.3f} below attainable floor {y:.3f} at skew_weight={w}; not clamped "
    "(clamp_to_attainable=False): fitted at {x:.3f}, achieved {z:.3f}."
)
CEILING_MESSAGE_UNCLAMPED = (
    "ssr_target={x:.3f} above attainable ceiling {y:.3f} at skew_weight={w}; not clamped "
    "(clamp_to_attainable=False): fitted at {x:.3f}, achieved {z:.3f}."
)
#: the largest power of the ``t = T u^p`` substitution of the term-structure integrals
#: (:func:`ts_substitution_power`): smooth for skews ``S ~ t^−γ`` up to ``γ = 1 − 1/p = 0.75``,
#: the historical-mode exponent clip
TS_SUBSTITUTION_POWER = 4
_POLYTOPE_LABELS = ("l1+l2<=2nu_cap", "l1-l2<=2nu_cap", "-l1+l2<=2nu_cap", "-l1-l2<=2nu_cap")
_TOL_T = 1e-9
_TOL_ACTIVE = 1e-9


# --------------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BreakEvenFitConfig:
    """Settings of the two minimisations (module docstring).

    ``pillars`` not present in the target set are dropped with a note; ``k1_bounds[0]`` must
    exceed ``k2 + k1_min_gap``; ``k1_grid`` is the number of points of the coarse geometric grid
    before the bounded refinement.  ``skew_weight`` is the owner's skew/leverage dial ``w_skew``
    (``≥ 0``; 1.0 by default, the middle of the transition measured in the module docstring),
    ``ssr_measure`` one of :data:`SSR_MEASURE_CHOICES` (``"auto"``: lsv on surface targets,
    naked on historical targets; :func:`resolve_ssr_measure`), ``ssr_tol`` the tracking tolerance
    (achieved against requested SSR) of the tracking bands and the tracking detail,
    ``clamp_to_attainable`` whether a request below the attainable floor (above the ceiling) is
    refit at the scan target that attains it, with the owner's message, ``term_structure`` the
    ``f(t)`` of
    the eq. 12.52 integrals (:data:`TERM_STRUCTURE_KINDS`), ``lv_ssr`` the ``R^LV(Mkt)`` of the
    lsv measure (:data:`LV_SSR_KINDS`; ``"numerical"`` needs a :class:`LocalVolSSR`),
    ``weights_ssr`` / ``weights_volvar`` are ``"uniform"`` or ``"relative"`` (``1/target²``),
    ``volvar_target`` one of :data:`VOLVAR_TARGET_KINDS`, ``sigma_hat_prefactor`` ``"market"``
    or ``"model"``; ``nu_cap`` is the hard feasibility limit of the first minimisation
    (``|λ1| + |λ2| ≤ 2 ν_cap``) and the soft cap of the second — default
    :data:`DEFAULT_NU_FLAG` = 2.5, the edge of the regime where the first-order SSRs were
    validated against stage 3 (module docstring; at ν = 5 they under-state the LSV's numerical
    SSR by 10–27%) —, ``nu_flag`` the level above which the first-order engine's accuracy is
    flagged; ``omega_max`` bounds the loadings of the second
    minimisation; ``n_quad`` / ``n_inner`` are the quadrature orders of the pillar kernels
    (:func:`volsto.analytics.breakeven.kernels`) and ``n_ts`` / ``n_quad_ts`` / ``n_inner_ts``
    those of the term-structure integrals (``t = T u^p``, Gauss-Legendre in ``u``;
    :func:`ts_substitution_power`)."""

    pillars: tuple[float, ...] = DEFAULT_TARGET_PILLARS
    k2: float = DEFAULT_K2
    k1_bounds: tuple[float, float] = (0.3, 20.0)
    k1_min_gap: float = 0.05
    k1_grid: int = 25
    skew_weight: float = DEFAULT_SKEW_WEIGHT
    ssr_measure: str = "auto"
    ssr_tol: float = DEFAULT_SSR_TOL
    clamp_to_attainable: bool = True
    term_structure: str = "atmf"
    lv_ssr: str = "order1"
    weights_ssr: str = "uniform"
    weights_volvar: str = "relative"
    volvar_target: str = "achieved"
    sigma_hat_prefactor: str = "market"
    nu_cap: float = DEFAULT_NU_FLAG
    nu_flag: float = DEFAULT_NU_FLAG
    chi_bounds: tuple[float, float] = (-0.99, 0.99)
    omega_max: float = 20.0
    n_quad: int = 64
    n_inner: int = 32
    n_ts: int = 64
    n_quad_ts: int = 32
    n_inner_ts: int = 24

    def __post_init__(self) -> None:
        if len(self.pillars) < 2 or any(t <= 0 for t in self.pillars):
            raise ValueError("pillars must be at least two positive maturities")
        if self.k2 <= 0:
            raise ValueError("k2 must be positive")
        lo, hi = self.k1_bounds
        if not lo < hi:
            raise ValueError("k1_bounds must be increasing")
        if lo < self.k2 + self.k1_min_gap:
            raise ValueError(
                f"k1_bounds[0] = {lo:g} must exceed k2 + k1_min_gap = {self.k2 + self.k1_min_gap:g}"
            )
        if self.k1_grid < 3:
            raise ValueError("k1_grid must be at least 3")
        if not (math.isfinite(self.skew_weight) and self.skew_weight >= 0.0):
            raise ValueError("skew_weight must be a finite non-negative number")
        if self.ssr_measure not in SSR_MEASURE_CHOICES:
            raise ValueError(f"ssr_measure must be one of {SSR_MEASURE_CHOICES}")
        if not self.ssr_tol > 0:
            raise ValueError("ssr_tol must be positive")
        if self.term_structure not in TERM_STRUCTURE_KINDS:
            raise ValueError(f"term_structure must be one of {TERM_STRUCTURE_KINDS}")
        if self.lv_ssr not in LV_SSR_KINDS:
            raise ValueError(f"lv_ssr must be one of {LV_SSR_KINDS}")
        for w in (self.weights_ssr, self.weights_volvar):
            if w not in WEIGHT_KINDS:
                raise ValueError(f"weights must be one of {WEIGHT_KINDS}, got {w!r}")
        if self.volvar_target not in VOLVAR_TARGET_KINDS:
            raise ValueError(f"volvar_target must be one of {VOLVAR_TARGET_KINDS}")
        if self.sigma_hat_prefactor not in PREFACTOR_KINDS:
            raise ValueError(f"sigma_hat_prefactor must be one of {PREFACTOR_KINDS}")
        if self.nu_cap <= 0 or self.nu_flag <= 0:
            raise ValueError("nu_cap and nu_flag must be positive")
        clo, chi_ = self.chi_bounds
        if not -1.0 <= clo < chi_ <= 1.0:
            raise ValueError("chi_bounds must be increasing inside [-1, 1]")
        if self.omega_max <= 2.0 * self.nu_cap:
            raise ValueError(
                "omega_max must exceed 2 nu_cap (the first fit allows |lambda_i| <= 2 nu_cap)"
            )
        if min(self.n_quad, self.n_inner, self.n_ts, self.n_quad_ts, self.n_inner_ts) < 4:
            raise ValueError("quadrature orders must be at least 4")


def resolve_ssr_measure(cfg: BreakEvenFitConfig, targets: TargetSet) -> tuple[str, str]:
    """The SSR measure the fit uses and a note: an explicit ``cfg.ssr_measure`` is kept;
    ``"auto"`` gives ``"lsv"`` when ``targets.term_structure_source == "surface"`` (marking mode:
    the eq. 12.52 integrals read the surface's skew on ``(0, T]``) and ``"naked"`` otherwise
    (historical mode: the lsv measure would integrate a power-law interpolation of the pillar
    skews, which fails the SPEC recovery test — module docstring).  Checked by
    ``tests/test_fit_2f.py::test_config_validation`` and the recovery tests."""
    if cfg.ssr_measure != "auto":
        return cfg.ssr_measure, ""
    src = targets.term_structure_source
    measure = "lsv" if src == "surface" else "naked"
    return measure, f"ssr_measure 'auto' resolved to '{measure}' (term_structure_source '{src}')"


def _resolved_config(cfg: BreakEvenFitConfig, targets: TargetSet) -> tuple[BreakEvenFitConfig, str]:
    measure, note = resolve_ssr_measure(cfg, targets)
    return (cfg if measure == cfg.ssr_measure else replace(cfg, ssr_measure=measure)), note


def format_skew_weight(w: float) -> str:
    """``W`` of the owner's message: 3 significant digits in positional notation (``1000`` not
    ``1e+03``, ``0.1``, ``1``)."""
    return str(
        np.format_float_positional(float(w), precision=3, unique=True, fractional=False, trim="-")
    )


# --------------------------------------------------------------------------------------------
# kernels along k1: the quadrature of ``kernels`` precomputed once per pillar
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PillarQuad:
    """The ``k``-independent part of :func:`~volsto.analytics.breakeven.kernels` for one
    maturity of the naked model (same Gauss-Legendre nodes and inner nodes, so ``A(k)`` and
    ``J(k)`` reproduce ``kernels((k1, k2), xi0, T).A / .J`` to round-off — checked by
    ``tests/test_fit_2f.py::test_affine_maps_match_engine``)."""

    T: float
    t: FloatArray
    w: FloatArray
    v: FloatArray
    W_T: float
    sigma_hat: float
    U: FloatArray  # (n_quad, n_inner) inner nodes
    S: FloatArray  # (n_quad, n_inner) inner weights times sqrt(xi0(u))

    def A(self, k: float) -> float:
        """``A(k) = ∫₀ᵀ ξ₀ e^{−kt} dt / W_T`` (eq. 7.38)."""
        return float(np.sum(self.w * self.v * np.exp(-k * self.t)) / self.W_T)

    def c(self, k: float) -> FloatArray:
        """``c̃(t_j; k) = ∫₀^{t_j} sqrt(ξ₀^u) e^{−k(t_j − u)} du`` at the nodes."""
        return np.asarray(
            np.sum(self.S * np.exp(-k * (self.t[:, None] - self.U)), axis=1), dtype=np.float64
        )

    def J(self, k: float) -> float:
        """``J(k) = ∫₀ᵀ ξ₀ c̃(t; k) dt / (2 σ̂³ T²)`` (eq. 8.54)."""
        return float(np.sum(self.w * self.v * self.c(k)) / (2.0 * self.sigma_hat**3 * self.T**2))


def pillar_quad(xi0: ForwardVarianceCurve, T: float, *, n_quad: int, n_inner: int) -> PillarQuad:
    """Precompute :class:`PillarQuad` for ``T`` on ``xi0`` (naked model)."""
    if T <= 0:
        raise ValueError("T must be positive")
    t, w = _gl(0.0, T, n_quad)
    v = np.asarray(xi0.xi0(t), dtype=np.float64)
    W_T = float(np.sum(w * v))
    U = np.empty((t.size, n_inner))
    S = np.empty((t.size, n_inner))
    for j, tj in enumerate(t):
        u, wu = _gl(0.0, float(tj), n_inner)
        U[j] = u
        S[j] = wu * np.sqrt(np.asarray(xi0.xi0(u), dtype=np.float64))
    return PillarQuad(float(T), t, w, v, W_T, float(np.sqrt(W_T / T)), U, S)


@dataclass(frozen=True)
class AffineMaps:
    """The naked affine maps at fixed ``(k1, k2)`` for the fitted pillars: ``svc(λ) = a @ λ``
    (``½ pref (λ1 A1 + λ2 A2)``, eq. 7.38) and ``skew(λ) = j @ λ`` (``λ1 J1 + λ2 J2``, eq.
    8.54) with ``a, j`` of shape ``(n, 2)``; ``prefactor`` names the convention and ``sig`` the
    vol that multiplies the sensitivities (``atf`` under ``"market"``, ``sigma_hat`` under
    ``"model"``)."""

    k1: float
    k2: float
    T: FloatArray
    atf: FloatArray
    a: FloatArray
    j: FloatArray
    A: FloatArray  # (n, 2) raw kernels
    J: FloatArray
    sigma_hat: FloatArray
    prefactor: str = "market"

    @property
    def sig(self) -> FloatArray:
        return self.atf if self.prefactor == "market" else self.sigma_hat

    def svc(self, lam: FloatArray) -> FloatArray:
        return np.asarray(self.a @ lam, dtype=np.float64)

    def skew(self, lam: FloatArray) -> FloatArray:
        return np.asarray(self.j @ lam, dtype=np.float64)


def _maps(
    k1: float,
    k2: float,
    T: FloatArray,
    atf: FloatArray,
    A: FloatArray,
    J: FloatArray,
    sig: FloatArray,
    prefactor: str,
) -> AffineMaps:
    if prefactor not in PREFACTOR_KINDS:
        raise ValueError(f"prefactor must be one of {PREFACTOR_KINDS}")
    # the SV skew is eq. 8.54 at the kernel's order-zero VS vol under both conventions (the
    # cube rescale (sig/atf)^3 was measured to move the skew the wrong way: gate table); the
    # conventions differ in the prefactor of the sensitivities only
    j = J.copy()
    pref = atf if prefactor == "market" else sig
    a = 0.5 * pref[:, None] * A
    return AffineMaps(float(k1), float(k2), T, atf, a, j, A, J, sig, prefactor)


def affine_maps(
    quads: Sequence[PillarQuad],
    atf: FloatArray,
    k1: float,
    k2: float,
    prefactor: str = "market",
) -> AffineMaps:
    """:class:`AffineMaps` from the precomputed quadratures and the market ATMF vols."""
    A = np.array([[q.A(k1), q.A(k2)] for q in quads])
    J = np.array([[q.J(k1), q.J(k2)] for q in quads])
    sig = np.array([q.sigma_hat for q in quads])
    T = np.array([q.T for q in quads])
    return _maps(k1, k2, T, atf, A, J, sig, prefactor)


def affine_maps_from_kernels(
    ks: tuple[float, float],
    kerns: Sequence[Kernels],
    atf: FloatArray,
    prefactor: str = "market",
) -> AffineMaps:
    """The same maps read from :func:`~volsto.analytics.breakeven.kernels` objects (the slow
    reference route; used by the tests to check :class:`PillarQuad`)."""
    A = np.array([k.A for k in kerns])
    J = np.array([k.J for k in kerns])
    sig = np.array([k.sigma_hat for k in kerns])
    T = np.array([k.T for k in kerns])
    return _maps(ks[0], ks[1], T, atf, A, J, sig, prefactor)


# --------------------------------------------------------------------------------------------
# the term-structure integrals of eq. 12.52 (lsv measure)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TermStructureBank:
    """The ``k``-independent quadrature of ``(1/T) ∫₀ᵀ f(t) S_t dt`` for every fitted pillar:
    nodes ``t = T u^p`` (shape ``(n, m)``, ``p`` of :func:`ts_substitution_power`), weights ``p
    u^{p−1} w_u f(t)`` (``f`` included), the factor
    ``f(t)``, the market skew at the nodes, and — flattened over the ``n m`` node maturities —
    the eq. 8.54 quadrature of the naked kernel's order-one skew ``J(t; k)`` at every node (the
    same construction as :class:`PillarQuad`, vectorised).  Checked against
    :class:`PillarQuad` and against :func:`~volsto.analytics.smile_dynamics.ssr_decomposition` by
    ``tests/test_fit_2f.py::test_lsv_measure_identities``."""

    T: FloatArray
    t: FloatArray
    weight: FloatArray
    f: FloatArray
    skew_market_nodes: FloatArray
    tq: FloatArray  # (n m, n_quad)
    wv: FloatArray  # (n m, n_quad) weights times xi0
    U: FloatArray  # (n m, n_quad, n_inner)
    S: FloatArray  # (n m, n_quad, n_inner)
    denom: FloatArray  # (n m,)  2 σ̂_t³ t²
    kind: str

    def J_nodes(self, k: float) -> FloatArray:
        """``J(t; k)`` (eq. 8.54 at maturity ``t``) at every node, shape ``(n, m)``."""
        c = np.sum(self.S * np.exp(-k * (self.tq[:, :, None] - self.U)), axis=2)
        return np.asarray(
            (np.sum(self.wv * c, axis=1) / self.denom).reshape(self.t.shape), dtype=np.float64
        )

    def integral(self, k: float) -> FloatArray:
        """``(1/T) ∫₀ᵀ f(t) J(t; k) dt`` per pillar."""
        return np.asarray(np.sum(self.weight * self.J_nodes(k), axis=1), dtype=np.float64)

    @property
    def I_market(self) -> FloatArray:
        """``(1/T) ∫₀ᵀ f(t) S^mkt_t dt`` per pillar."""
        return np.asarray(np.sum(self.weight * self.skew_market_nodes, axis=1), dtype=np.float64)


def term_structure_bank(
    targets: TargetSet,
    T: FloatArray,
    xi0: ForwardVarianceCurve,
    *,
    kind: str,
    n_ts: int,
    n_quad: int,
    n_inner: int,
    power: float | None = None,
) -> TermStructureBank:
    """Build :class:`TermStructureBank` (``f(t)`` per :data:`TERM_STRUCTURE_KINDS`: ``"flat"``
    1; ``"atmf"`` ``σ²(t)/(σ̂_t σ̂_T)`` on the targets' ATMF curve; ``"vs"`` the same on ``xi0``).
    """
    if kind not in TERM_STRUCTURE_KINDS:
        raise ValueError(f"kind must be one of {TERM_STRUCTURE_KINDS}")
    T = np.asarray(T, dtype=np.float64)
    u, wu = np.polynomial.legendre.leggauss(n_ts)
    u = 0.5 * (u + 1.0)
    wu = 0.5 * wu
    p = ts_substitution_power(targets) if power is None else float(power)
    t = T[:, None] * (u**p)[None, :]
    weight = np.broadcast_to(p * u ** (p - 1) * wu, t.shape)
    if kind == "flat":
        f = np.ones_like(t)
    else:
        curve = targets.atmf_curve(max(float(T.max()), 1.5 / 12.0)) if kind == "atmf" else xi0
        sig_t = np.sqrt(np.asarray(curve.total_variance(t), dtype=np.float64) / t)
        sig_T = np.sqrt(np.asarray(curve.total_variance(T), dtype=np.float64) / T)
        f = np.asarray(curve.xi0(t), dtype=np.float64) / (sig_t * sig_T[:, None])
    tau = t.ravel()
    xq, wq = np.polynomial.legendre.leggauss(n_quad)
    xq, wq = 0.5 * (xq + 1.0), 0.5 * wq
    xi, wi = np.polynomial.legendre.leggauss(n_inner)
    xi, wi = 0.5 * (xi + 1.0), 0.5 * wi
    tq = tau[:, None] * xq[None, :]
    w = tau[:, None] * wq[None, :]
    v = np.asarray(xi0.xi0(tq), dtype=np.float64)
    W = np.sum(w * v, axis=1)
    sig = np.sqrt(W / tau)
    U = tq[:, :, None] * xi[None, None, :]
    S = (
        tq[:, :, None]
        * wi[None, None, :]
        * np.sqrt(np.asarray(xi0.xi0(U.ravel()), dtype=np.float64)).reshape(U.shape)
    )
    denom = 2.0 * sig**3 * tau * tau
    skew_nodes = targets.market_skew(tau).reshape(t.shape)
    return TermStructureBank(T, t, weight * f, f, skew_nodes, tq, w * v, U, S, denom, kind)


def ts_substitution_power(targets: TargetSet) -> float:
    """The power ``p`` of the substitution ``t = T u^p`` of the term-structure integrals:
    ``p = clip(1/(1 − γ), 2, 4)`` with ``γ = −d ln|S| / d ln t`` the market skew's short-end
    exponent measured between ``t = 1e-4`` and ``1e-3``.  For ``S ~ t^−γ`` the market integrand
    in ``u`` is ``u^{p(1−γ)−1}``, constant at ``p = 1/(1 − γ)``; the kernel's own skew is finite
    at ``t → 0`` (integrand ``u^{p−1}``, polynomial at ``p = 2`` and 4).  The reference SSVI (``γ =
    0.50``) and SPX 2022-12-30 (``γ = 0.23``) get ``p = 2``; the historical power-law extension at
    its exponent clip ``γ = 0.75`` gets ``p = 4`` (plain ``p = 2`` loses −6.8e-3 there).  Checked by
    ``tests/test_fit_2f.py::test_term_structure_quadrature``."""
    t = np.array([1e-4, 1e-3])
    S = np.abs(targets.market_skew(t))
    gamma = float(-np.log(S[1] / S[0]) / np.log(t[1] / t[0]))
    return float(np.clip(1.0 / (1.0 - min(gamma, 0.9)), 2.0, float(TS_SUBSTITUTION_POWER)))


@dataclass(frozen=True)
class LocalVolSSR:
    """The pure-Dupire numerical SSR per pillar, ``R^LV_num(T) = slope / S^mkt_T`` (the V2
    correction of the lsv measure, module docstring), with its standard error and settings."""

    T: FloatArray
    R: FloatArray
    R_stderr: FloatArray
    n_paths: int
    dt_max: float
    eps: float
    wall_seconds: float

    def at(self, T: float) -> tuple[float, float]:
        i = int(np.argmin(np.abs(self.T - T)))
        if abs(self.T[i] - T) > _TOL_T:
            raise KeyError(f"no local-vol SSR at T = {T:g}")
        return float(self.R[i]), float(self.R_stderr[i])


def local_vol_ssr_numerical(
    surface: Any,
    pillars: Sequence[float],
    *,
    sim: Any,
    eps: float = 0.05,
    local_vol: Any | None = None,
) -> LocalVolSSR:
    """``R^LV_num(T)`` by :func:`~volsto.analytics.smile_dynamics.ssr_numerical_many` on the
    Dupire local-vol model of ``surface`` (one simulation for all pillars; the verification
    measured a +3.4% bias at 3M with ``dt_max = 0.01``, so use ``dt_max ≤ 1/400``); the slope is
    divided by the *market* skew (not the Monte Carlo skew) to remove the skew noise."""
    from volsto.analytics.smile_dynamics import ssr_numerical_many
    from volsto.market.dupire import LocalVolSurface
    from volsto.models.localvol import LocalVol

    t0 = time.perf_counter()
    lvs = local_vol if local_vol is not None else LocalVolSurface.from_implied(surface)
    rows = ssr_numerical_many(LocalVol(lvs), list(pillars), eps=eps, sim=sim)
    T = np.array([r.T for r in rows])
    S = np.asarray(surface.atm_skew(T), dtype=np.float64)
    R = np.array([r.slope for r in rows]) / S
    se = np.array([r.slope_stderr for r in rows]) / np.abs(S)
    dt = sim.dt_max if isinstance(sim.dt_max, float) else float(sim.dt_max.finest)
    return LocalVolSSR(T, R, se, int(sim.n_paths), float(dt), float(eps), time.perf_counter() - t0)


@dataclass(frozen=True)
class MeasureMaps:
    """Both SSR measures at fixed ``(k1, k2)`` (module docstring), affine in ``λ``:

    * ``svc_naked(λ) = c · ½ pref (λ·A)`` with the level ratio ``c = σ_0 / σ_0^pair``
      (``σ_0^pair = σ_0`` under ``"market"``, ``sqrt(ξ₀(0))`` under ``"model"``);
    * ``svc_lsv(λ) = σ_0 S^mkt_T R^LV(Mkt)_T + svc_naked(λ) − σ_0 [λ·J_T + c_T λ·I_T]`` — eq.
      12.52 in covariance form, ``R^LV(Mkt) = 1 + I^mkt_T / S^mkt_T`` at order one (``c_T = 1``)
      or the Dupire Monte Carlo value with ``c_T = (R^LV_num − 1)/(R^LV_o1 − 1)``;
    * ``skew_naked(λ) = λ·J_T``.
    """

    naked: AffineMaps
    I: FloatArray  # (n, 2) term-structure rows
    skew_market: FloatArray
    I_market: FloatArray
    sigma_0: float
    level_ratio: float
    r_lv_market: FloatArray
    lv_scale: FloatArray

    @property
    def T(self) -> FloatArray:
        return self.naked.T

    @property
    def k1(self) -> float:
        return self.naked.k1

    def lsv_rows(self) -> tuple[FloatArray, FloatArray]:
        a = self.level_ratio * self.naked.a - self.sigma_0 * (
            self.naked.j + self.lv_scale[:, None] * self.I
        )
        b = self.sigma_0 * self.skew_market * self.r_lv_market
        return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)

    def rows(self, measure: str) -> tuple[FloatArray, FloatArray]:
        """``(a, b)`` with ``svc_model(λ) = a @ λ + b`` under ``measure``."""
        if measure == "naked":
            return self.level_ratio * self.naked.a, np.zeros(self.T.size)
        if measure == "lsv":
            return self.lsv_rows()
        raise ValueError(f"measure must be one of {SSR_MEASURES}")

    def svc(self, lam: FloatArray, measure: str) -> FloatArray:
        a, b = self.rows(measure)
        return np.asarray(a @ lam + b, dtype=np.float64)

    def svc_naked(self, lam: FloatArray) -> FloatArray:
        return self.svc(lam, "naked")

    def svc_lsv(self, lam: FloatArray) -> FloatArray:
        return self.svc(lam, "lsv")

    def skew_naked(self, lam: FloatArray) -> FloatArray:
        return self.naked.skew(lam)

    def ssr(self, lam: FloatArray, measure: str) -> FloatArray:
        """Achieved SSR ``svc_model / (σ_0 S^mkt)`` under ``measure``."""
        return np.asarray(self.svc(lam, measure) / (self.sigma_0 * self.skew_market))

    def ssr_naked_own(self, lam: FloatArray) -> FloatArray:
        """The naked kernel's own first-order SSR ``svc_naked / (σ_0 λ·J)``."""
        return np.asarray(self.svc_naked(lam) / (self.sigma_0 * self.skew_naked(lam)))

    def sensi_spot_lv(self, lam: FloatArray) -> FloatArray:
        """The local-vol spot sensitivity ``(svc_lsv − svc_naked) / c`` (D5 of the module
        docstring), in the units of ``SensiX`` so that the VolVar decomposition implies
        ``c (SensiSpot + ρ_S1 SensiX + ρ_S2 SensiY) = svc_lsv`` under both prefactors
        (:meth:`implied_covariance`; ``c`` the level ratio)."""
        return np.asarray((self.svc_lsv(lam) - self.svc_naked(lam)) / self.level_ratio)

    def implied_covariance(self, lam: FloatArray, sensi_spot: FloatArray | None) -> FloatArray:
        """``c (SensiSpot + Σ_i ρ_Si SensiX_i)`` with ``ρ_Si SensiX_i = ½ sig λ_i A_i``: the
        SpotVolCovar implied by the second fit's VolVar decomposition (equals ``svc_model`` of the
        measure whose ``sensi_spot`` is passed — None for naked; checked by
        ``tests/test_fit_2f.py::test_volvar_decomposition_consistent``)."""
        rho_sensi = 0.5 * self.naked.sig * (self.naked.A @ lam)
        spot = 0.0 if sensi_spot is None else sensi_spot
        return np.asarray(self.level_ratio * (spot + rho_sensi), dtype=np.float64)


# --------------------------------------------------------------------------------------------
# the inner problem: 2-D least squares on the nu-feasibility polytope
# --------------------------------------------------------------------------------------------


def _weights(kind: str, target: FloatArray) -> FloatArray:
    if kind == "uniform":
        return np.ones_like(target)
    if np.any(target == 0.0) or not np.all(np.isfinite(target)):
        raise ValueError("relative weights need finite, non-zero targets")
    return np.asarray(1.0 / target**2, dtype=np.float64)


def _qp2(
    H: FloatArray, g: FloatArray, G: FloatArray, h: FloatArray
) -> tuple[FloatArray, tuple[int, ...], bool]:
    """``argmin ½ xᵀHx − gᵀx`` over ``G x ≤ h`` for ``x ∈ R²`` by candidate enumeration
    (convex QP: the optimum is the unconstrained point, the equality optimum on one edge or a
    vertex).  Returns ``(x, active constraint indices, feasible)``; when no candidate is
    feasible the unconstrained solution comes back with ``feasible=False``."""
    scale = 1.0 + np.abs(h)

    def feasible(x: FloatArray) -> bool:
        return bool(np.all(G @ x <= h + _TOL_ACTIVE * scale))

    def obj(x: FloatArray) -> float:
        return float(0.5 * x @ H @ x - g @ x)

    def active(x: FloatArray) -> tuple[int, ...]:
        return tuple(int(i) for i in np.flatnonzero(np.abs(G @ x - h) <= 1e-7 * scale))

    try:
        x0 = np.linalg.solve(H, g)
    except np.linalg.LinAlgError:
        x0 = np.linalg.lstsq(H, g, rcond=None)[0]
    cands: list[tuple[float, FloatArray]] = []
    if feasible(x0):
        cands.append((obj(x0), x0))
    m = G.shape[0]
    for i in range(m):
        K = np.zeros((3, 3))
        K[:2, :2] = H
        K[:2, 2] = G[i]
        K[2, :2] = G[i]
        rhs = np.array([g[0], g[1], h[i]])
        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            continue
        x = sol[:2]
        if feasible(x):
            cands.append((obj(x), x))
    for i in range(m):
        for j in range(i + 1, m):
            Gij = G[[i, j]]
            if abs(np.linalg.det(Gij)) <= 1e-14 * (np.abs(Gij).max() ** 2 + 1e-300):
                continue
            x = np.linalg.solve(Gij, h[[i, j]])
            if feasible(x):
                cands.append((obj(x), x))
    if not cands:
        return np.asarray(x0, dtype=np.float64), (), False
    best = min(cands, key=lambda c: c[0])[1]
    return np.asarray(best, dtype=np.float64), active(best), True


@dataclass(frozen=True)
class InnerSolution:
    """The inner least squares at one ``k1``: ``λ``, the penalised objective and its two parts
    (SSR residuals, skew penalty), the active ν-feasibility rows (indices into
    ``[λ1+λ2, λ1−λ2, −λ1+λ2, −λ1−λ2] ≤ 2 ν_cap``), feasibility, the unconstrained-LS covariance
    of ``λ`` and the maps."""

    k1: float
    lam: FloatArray
    objective: float
    objective_ssr: float
    objective_skew: float
    active: tuple[int, ...]
    feasible: bool
    cov: FloatArray
    maps: MeasureMaps


@dataclass(frozen=True)
class _FirstProblem:
    quads: tuple[PillarQuad, ...]
    bank: TermStructureBank
    T: FloatArray
    atf: FloatArray
    sigma_0: float
    level_ratio: float
    skew_market: FloatArray
    r_lv_order1: FloatArray
    r_lv_market: FloatArray
    lv_scale: FloatArray
    ssr_target: FloatArray
    ssr_se: FloatArray
    weights_kind: str
    skew_weight: float
    measure: str
    nu_cap: float
    k2: float
    prefactor: str
    cache: dict[float, MeasureMaps] = field(default_factory=dict, compare=False, repr=False)

    @property
    def wc(self) -> FloatArray:
        return _weights(self.weights_kind, self.ssr_target)

    def with_ssr(self, ssr: FloatArray, skew_weight: float | None = None) -> _FirstProblem:
        """The same problem (sharing the maps cache) at another SSR target / skew weight."""
        return replace(
            self,
            ssr_target=np.asarray(ssr, dtype=np.float64),
            skew_weight=self.skew_weight if skew_weight is None else float(skew_weight),
        )

    def maps(self, k1: float) -> MeasureMaps:
        key = float(k1)
        mm = self.cache.get(key)
        if mm is None:
            naked = affine_maps(self.quads, self.atf, key, self.k2, self.prefactor)
            I = np.stack([self.bank.integral(key), self.bank.integral(self.k2)], axis=1)
            mm = MeasureMaps(
                naked,
                I,
                self.skew_market,
                self.bank.I_market,
                self.sigma_0,
                self.level_ratio,
                self.r_lv_market,
                self.lv_scale,
            )
            if len(self.cache) > 4096:
                self.cache.clear()
            self.cache[key] = mm
        return mm

    @property
    def has_noise(self) -> bool:
        """Whether the SSR targets carry sampling standard errors (historical mode)."""
        se = self.ssr_se
        return bool(se.size and np.all(np.isfinite(se)) and np.all(se > 0))

    @property
    def noise_cov(self) -> FloatArray:
        """Covariance of the stacked residuals from the target noise: ``wc_i se_i²`` on the SSR
        rows, 0 on the skew rows (the pricing date's skew is observed, not estimated)."""
        n = self.T.size
        return np.diag(np.concatenate((self.wc * self.ssr_se**2, np.zeros(n))))

    def stacked(self, k1: float) -> tuple[MeasureMaps, FloatArray, FloatArray]:
        """``(maps, Rm, y)`` with the stacked residuals ``Rm λ − y`` at ``k1``: the SSR rows
        ``sqrt(wc_i) (SSR_model_i − r_i)`` then the skew rows ``sqrt(w_skew) (λ·J_i/S_i − 1)``."""
        mm = self.maps(k1)
        a, b = mm.rows(self.measure)
        scale = self.sigma_0 * self.skew_market
        swc = np.sqrt(self.wc)
        R1 = swc[:, None] * a / scale[:, None]
        y1 = swc * (self.ssr_target - b / scale)
        sw = math.sqrt(self.skew_weight)
        R2 = sw * mm.naked.j / self.skew_market[:, None]
        y2 = np.full(self.T.size, sw)
        return mm, np.vstack((R1, R2)), np.concatenate((y1, y2))

    def solve(self, k1: float) -> InnerSolution:
        mm, Rm, y = self.stacked(k1)
        n = self.T.size
        H = 2.0 * Rm.T @ Rm
        g = 2.0 * Rm.T @ y
        G = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
        h = np.full(4, 2.0 * self.nu_cap)
        lam, active, feas = _qp2(H, g, G, h)
        r = Rm @ lam - y
        o1 = float(r[:n] @ r[:n])
        o2 = float(r[n:] @ r[n:])
        cov = _sandwich(Rm, self.noise_cov) if self.has_noise else np.full((2, 2), np.nan)
        return InnerSolution(
            float(k1), lam, o1 + o2, o1, o2, active, feas, np.asarray(cov, dtype=np.float64), mm
        )


def _sandwich(J: FloatArray, sigma: FloatArray) -> FloatArray:
    """``(JᵀJ)⁻¹ Jᵀ Σ J (JᵀJ)⁻¹``: the covariance of the least-squares estimate of ``min |r|²``
    with Jacobian ``J`` when the residuals carry the noise covariance ``Σ`` (NaN when ``JᵀJ`` is
    singular)."""
    try:
        N = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        return np.full((J.shape[1], J.shape[1]), np.nan)
    return np.asarray(N @ J.T @ sigma @ J @ N, dtype=np.float64)


def _select_pillars(targets: TargetSet, pillars: Sequence[float]) -> tuple[FloatArray, list[str]]:
    """Indices into ``targets.pillars`` of the requested pillars; the absent ones are noted."""
    tp = np.asarray(targets.pillars, dtype=np.float64)
    idx: list[int] = []
    dropped: list[float] = []
    for T in sorted(float(t) for t in pillars):
        i = int(np.argmin(np.abs(tp - T)))
        if abs(tp[i] - T) <= _TOL_T:
            idx.append(i)
        else:
            dropped.append(T)
    notes = []
    if dropped:
        notes.append(f"pillars absent from the targets dropped: {dropped}")
    if len(idx) < 2:
        raise ValueError(f"the fit needs at least two pillars, {len(idx)} available")
    return np.asarray(idx, dtype=np.int64), notes


def _first_problem(
    targets: TargetSet,
    cfg: BreakEvenFitConfig,
    xi0: ForwardVarianceCurve,
    lv_ssr: LocalVolSSR | None = None,
) -> tuple[_FirstProblem, list[str]]:
    cfg, measure_note = _resolved_config(cfg, targets)
    idx, notes = _select_pillars(targets, cfg.pillars)
    if measure_note:
        notes.append(measure_note)
    T = np.asarray(targets.pillars, dtype=np.float64)[idx]
    quads = tuple(pillar_quad(xi0, float(t), n_quad=cfg.n_quad, n_inner=cfg.n_inner) for t in T)
    bank = term_structure_bank(
        targets,
        T,
        xi0,
        kind=cfg.term_structure,
        n_ts=cfg.n_ts,
        n_quad=cfg.n_quad_ts,
        n_inner=cfg.n_inner_ts,
    )
    skew = np.asarray(targets.skew_target, dtype=np.float64)[idx]
    if np.any(skew == 0) or not np.all(np.isfinite(skew)):
        raise ValueError("the fit needs finite, non-zero market skews at every pillar")
    s0 = float(targets.sigma_0)
    level_ratio = (
        1.0 if cfg.sigma_hat_prefactor == "market" else s0 / math.sqrt(float(xi0.xi0(0.0)))
    )
    r_lv_o1 = 1.0 + bank.I_market / skew
    r_lv = r_lv_o1.copy()
    lv_scale = np.ones(T.size)
    if cfg.ssr_measure == "lsv" and targets.term_structure_source != "surface":
        notes.append(
            "lsv measure: the market skew on (0, T] is the pillar skews interpolated as a power "
            "law in T (historical mode; no surface) - the eq. 12.52 local integrals carry that "
            "interpolation, which fails the SPEC recovery test (module docstring); the 'auto' "
            "measure is naked here"
        )
    if cfg.lv_ssr == "numerical":
        if lv_ssr is None:
            raise ValueError("lv_ssr='numerical' needs a LocalVolSSR (local_vol_ssr_numerical)")
        r_lv = np.array([lv_ssr.at(float(t))[0] for t in T])
        lv_scale = (r_lv - 1.0) / (r_lv_o1 - 1.0)
        notes.append(
            f"R^LV(Mkt) from a pure-Dupire Monte Carlo ({lv_ssr.n_paths} paths, dt "
            f"{lv_ssr.dt_max:.4g}): {np.round(r_lv, 3).tolist()} vs order one "
            f"{np.round(r_lv_o1, 3).tolist()}; local integrals scaled by "
            f"{np.round(lv_scale, 3).tolist()}"
        )
    ssr = np.asarray(targets.ssr_target, dtype=np.float64)[idx]
    svc_se = np.asarray(targets.spot_vol_covar_se, dtype=np.float64)[idx]
    prob = _FirstProblem(
        quads,
        bank,
        T,
        np.asarray(targets.atf, dtype=np.float64)[idx],
        s0,
        float(level_ratio),
        skew,
        r_lv_o1,
        r_lv,
        lv_scale,
        ssr,
        np.abs(svc_se / (s0 * skew)),
        cfg.weights_ssr,
        float(cfg.skew_weight),
        cfg.ssr_measure,
        float(cfg.nu_cap),
        float(cfg.k2),
        cfg.sigma_hat_prefactor,
    )
    return prob, notes


def _k1_grid(cfg: BreakEvenFitConfig) -> FloatArray:
    lo, hi = cfg.k1_bounds
    return np.asarray(np.geomspace(lo, hi, cfg.k1_grid), dtype=np.float64)


def _optimise_k1(
    prob: _FirstProblem, cfg: BreakEvenFitConfig
) -> tuple[InnerSolution, FloatArray, list[InnerSolution]]:
    """Coarse geometric grid in ``k1``, bounded scalar refinement around the best grid point."""
    grid = _k1_grid(cfg)
    sols = [prob.solve(float(k)) for k in grid]
    objs = np.array([s.objective for s in sols])
    i = int(np.argmin(objs))
    lo = float(grid[max(i - 1, 0)])
    hi = float(grid[min(i + 1, grid.size - 1)])
    best = sols[i]
    if hi > lo:
        res = minimize_scalar(
            lambda k: prob.solve(float(k)).objective,
            bounds=(lo, hi),
            method="bounded",
            options={"xatol": 1e-5 * hi},
        )
        cand = prob.solve(float(res.x))
        if cand.objective <= best.objective:
            best = cand
    return best, grid, sols


def k1_profile(
    targets: TargetSet,
    cfg: BreakEvenFitConfig,
    xi0: ForwardVarianceCurve,
    k1s: Sequence[float] | FloatArray | None = None,
    *,
    lv_ssr: LocalVolSSR | None = None,
) -> pd.DataFrame:
    """The first-minimisation objective along ``k1`` (default: the config's coarse grid) with
    the inner optimum per point: columns ``k1, objective, objective_ssr, objective_skew,
    lambda1, lambda2, nu_min, n_active``."""
    prob, _ = _first_problem(targets, cfg, xi0, lv_ssr)
    grid = _k1_grid(cfg) if k1s is None else np.asarray(k1s, dtype=np.float64)
    return _profile_frame([prob.solve(float(k)) for k in grid])


def _profile_frame(sols: Sequence[InnerSolution]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "k1": [s.k1 for s in sols],
            "objective": [s.objective for s in sols],
            "objective_ssr": [s.objective_ssr for s in sols],
            "objective_skew": [s.objective_skew for s in sols],
            "lambda1": [float(s.lam[0]) for s in sols],
            "lambda2": [float(s.lam[1]) for s in sols],
            "nu_min": [0.5 * abs(float(np.sum(s.lam))) for s in sols],
            "n_active": [len(s.active) for s in sols],
        }
    )


@dataclass(frozen=True)
class FirstFit:
    """Result of the first minimisation: ``(k1, λ1, λ2)``, the penalised objective and its
    parts, the standard errors, the active ν-feasibility rows, the ``k1`` profile, the maps and
    the per-pillar table (``T, atf, ssr_target, svc_target, svc_model, svc_naked, svc_lsv,
    ssr_achieved`` (under the configured measure), ``ssr_achieved_lsv,
    ssr_achieved_naked_vs_market, ssr_naked_own, r_lv_market, skew_market, skew_naked,
    skew_gap_rel``)."""

    k1: float
    lambda1: float
    lambda2: float
    objective: float
    objective_ssr: float
    objective_skew: float
    k1_se: float
    lambda_cov: FloatArray
    active: tuple[int, ...]
    feasible: bool
    k1_at_bound: bool
    profile: pd.DataFrame
    table: pd.DataFrame
    maps: MeasureMaps
    measure: str
    skew_weight: float
    notes: tuple[str, ...]
    wall_seconds: float
    nu_flag: float = DEFAULT_NU_FLAG

    @property
    def lam(self) -> FloatArray:
        return np.array([self.lambda1, self.lambda2])

    @property
    def lambda1_se(self) -> float:
        return float(np.sqrt(self.lambda_cov[0, 0]))

    @property
    def lambda2_se(self) -> float:
        return float(np.sqrt(self.lambda_cov[1, 1]))

    @property
    def nu_min(self) -> float:
        """The smallest ``ν`` compatible with ``(λ1, λ2)``: ``|λ1 + λ2|/2`` (module docstring)."""
        return 0.5 * abs(self.lambda1 + self.lambda2)

    @property
    def nu_limit_binding(self) -> bool:
        return len(self.active) > 0

    @property
    def active_labels(self) -> tuple[str, ...]:
        return tuple(_POLYTOPE_LABELS[i] for i in self.active)

    @property
    def ssr_achieved_mean(self) -> float:
        return float(self.table["ssr_achieved"].mean())

    @property
    def mean_skew_gap(self) -> float:
        """Mean over pillars of ``|skew_naked / skew_market − 1|``."""
        return float(np.mean(np.abs(self.table["skew_gap_rel"])))

    @property
    def max_skew_ratio(self) -> float:
        """Largest naked-to-market skew ratio over the pillars."""
        return float(np.max(self.table["skew_naked"] / self.table["skew_market"]))

    @property
    def first_order_valid(self) -> bool:
        """Whether the fit lies in the regime where the first-order SSRs were validated against
        stage 3 (``ν_min ≤ nu_flag`` and skew ratio ``≤`` :data:`FIRST_ORDER_SKEW_RATIO_LIMIT`;
        ``nu_flag`` of the config)."""
        return (
            self.nu_min <= self.nu_flag * (1.0 + 1e-6)
            and self.max_skew_ratio <= FIRST_ORDER_SKEW_RATIO_LIMIT
        )


def _first_table(prob: _FirstProblem, lam: FloatArray, mm: MeasureMaps) -> pd.DataFrame:
    svc_model = mm.svc(lam, prob.measure)
    skew_naked = mm.skew_naked(lam)
    return pd.DataFrame(
        {
            "T": prob.T,
            "atf": prob.atf,
            "ssr_target": prob.ssr_target,
            "svc_target": prob.ssr_target * prob.sigma_0 * prob.skew_market,
            "svc_model": svc_model,
            "svc_naked": mm.svc_naked(lam),
            "svc_lsv": mm.svc_lsv(lam),
            "ssr_achieved": mm.ssr(lam, prob.measure),
            "ssr_achieved_lsv": mm.ssr(lam, "lsv"),
            "ssr_achieved_naked_vs_market": mm.ssr(lam, "naked"),
            "ssr_naked_own": mm.ssr_naked_own(lam),
            "r_lv_market": prob.r_lv_market,
            "skew_market": prob.skew_market,
            "skew_naked": skew_naked,
            "skew_gap_rel": skew_naked / prob.skew_market - 1.0,
        }
    )


def _first_fit_from(
    prob: _FirstProblem, cfg: BreakEvenFitConfig, notes: list[str], t0: float
) -> FirstFit:
    best, _grid, sols = _optimise_k1(prob, cfg)
    notes = list(notes)
    k1 = best.k1
    at_bound = bool(k1 <= cfg.k1_bounds[0] * (1 + 1e-6) or k1 >= cfg.k1_bounds[1] * (1 - 1e-6))
    if at_bound:
        notes.append(f"k1 = {k1:.4g} sits on a bound of {cfg.k1_bounds}")
    if best.active:
        notes.append(
            f"nu feasibility limit |lambda1| + |lambda2| <= 2 nu_cap = {2 * cfg.nu_cap:g} binds"
        )
    nu_min = 0.5 * abs(float(np.sum(best.lam)))
    if nu_min > cfg.nu_flag:
        notes.append(
            f"nu >= |lambda1 + lambda2|/2 = {nu_min:.2f} > nu_flag {cfg.nu_flag:g}: outside the "
            "first-order engine's measured accuracy (-5% at nu 1.74, -10% at 2.5)"
        )
    if not best.feasible:
        notes.append("nu feasibility polytope infeasible (numerical): unconstrained lambda kept")
    k1_se, lam_cov, se_notes = _first_stderr(prob, best, cfg.k1_bounds, at_bound)
    notes += se_notes
    table = _first_table(prob, best.lam, best.maps)
    ratio = table["skew_naked"].to_numpy() / table["skew_market"].to_numpy()
    if prob.measure == "lsv" and ratio[0] > 1.0:
        notes.append(
            f"naked skew steeper than market at T = {prob.T[0]:g} (ratio {ratio[0]:.2f}): "
            "the leverage is counter-skewed there (eq. 12.52: a larger S^SV/S lowers the "
            "LSV SSR when R^SV < R^LV(SV))"
        )
    if nu_min > cfg.nu_flag * (1.0 + 1e-6) or float(np.max(ratio)) > FIRST_ORDER_SKEW_RATIO_LIMIT:
        notes.append(first_order_regime_note(nu_min, float(np.max(ratio)), cfg.nu_flag))
    return FirstFit(
        k1,
        float(best.lam[0]),
        float(best.lam[1]),
        best.objective,
        best.objective_ssr,
        best.objective_skew,
        k1_se,
        lam_cov,
        best.active,
        best.feasible,
        at_bound,
        _profile_frame(sols),
        table,
        best.maps,
        prob.measure,
        prob.skew_weight,
        tuple(notes),
        time.perf_counter() - t0,
        float(cfg.nu_flag),
    )


def fit_first(
    targets: TargetSet,
    cfg: BreakEvenFitConfig,
    xi0: ForwardVarianceCurve,
    *,
    lv_ssr: LocalVolSSR | None = None,
) -> FirstFit:
    """The first minimisation (module docstring): coarse geometric grid in ``k1``, bounded
    scalar refinement around the best grid point, exact inner QP on the ν polytope."""
    t0 = time.perf_counter()
    prob, notes = _first_problem(targets, cfg, xi0, lv_ssr)
    return _first_fit_from(prob, cfg, notes, t0)


def first_order_regime_note(nu: float, skew_ratio: float, nu_flag: float) -> str:
    """The warning attached to first-order SSRs outside the validated regime (module docstring,
    "First-order accuracy"): ``ν > nu_flag`` or a naked-to-market skew ratio above
    :data:`FIRST_ORDER_SKEW_RATIO_LIMIT`."""
    return (
        f"first-order SSRs outside the validated regime (nu {nu:.2f} vs nu_flag {nu_flag:g}, max "
        f"naked/market skew ratio {skew_ratio:.2f} vs {FIRST_ORDER_SKEW_RATIO_LIMIT:g}): stage 3 "
        "measured the lsv-measure SSR 10-27% below the calibrated LSV's numerical SSR at nu 5 "
        "(ratio 1.37); read the achieved SSRs, the floor and the message as first-order "
        "estimates and check stage 3"
    )


def _first_stderr(
    prob: _FirstProblem, best: InnerSolution, bounds: tuple[float, float], at_bound: bool
) -> tuple[float, FloatArray, list[str]]:
    """Standard errors of ``(k1, λ1, λ2)`` (module docstring): NaN in marking mode (the targets
    are not noisy observations); in historical mode the sandwich ``(JᵀJ)⁻¹ Jᵀ Σ J (JᵀJ)⁻¹`` on
    the stacked residuals (SSR rows with the target standard errors, skew rows noiseless), ``J =
    [∂r/∂k1 at fixed λ, Rm]`` by a central difference in ``k1``; ``λ`` alone at fixed ``k1`` when
    ``k1`` sits on a bound.  The penalised objective is never used as a residual variance."""
    notes: list[str] = []
    nan2 = np.full((2, 2), np.nan)
    if not prob.has_noise:
        notes.append(
            "no standard errors for (k1, lambda): the targets carry no sampling error (marking "
            "mode) and the penalised objective is not a residual variance"
        )
        return float("nan"), nan2, notes
    k1 = best.k1
    _, Rm, _ = prob.stacked(k1)
    sigma = prob.noise_cov
    if at_bound:
        notes.append("k1 standard error: k1 at a bound; lambda standard errors at fixed k1")
        return float("nan"), _sandwich(Rm, sigma), notes
    h = 1e-4 * k1
    lo, hi = max(k1 - h, bounds[0]), min(k1 + h, bounds[1])
    _, Ra, ya = prob.stacked(lo)
    _, Rb, yb = prob.stacked(hi)
    dr = ((Rb @ best.lam - yb) - (Ra @ best.lam - ya)) / (hi - lo)
    cov = _sandwich(np.column_stack((dr, Rm)), sigma)
    if best.active:
        notes.append("standard errors ignore the active nu limit (kink possible)")
    k1_var = float(cov[0, 0])
    k1_se = math.sqrt(k1_var) if np.isfinite(k1_var) and k1_var >= 0 else float("nan")
    return k1_se, np.asarray(cov[1:, 1:], dtype=np.float64), notes


# --------------------------------------------------------------------------------------------
# second minimisation
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class SecondFit:
    """Result of the second minimisation: ``(ω1, ω2, χ)``, the weighted objective, standard
    errors from the Jacobian, bound flags, the number of starts and distinct optima, the
    per-pillar table (``T, volvar_target, volvar_target_requested, volvar_model, volvol_target,
    volvol_model`` — absolute vols of the ATMF vol —, ``sensi_spot_lv, correl_target,
    correl_implied``)."""

    omega1: float
    omega2: float
    chi: float
    objective: float
    stderr: dict[str, float]
    bound_flags: tuple[str, ...]
    n_starts: int
    n_distinct: int
    nu_penalised: bool
    table: pd.DataFrame
    notes: tuple[str, ...]
    wall_seconds: float


def _volvar_model(
    x: FloatArray,
    lam: FloatArray,
    A: FloatArray,
    sig: FloatArray,
    spot: FloatArray | None = None,
) -> tuple[FloatArray, float]:
    """``VolVar`` per pillar and ``ν`` for ``x = (ω1, ω2, χ)`` at fixed ``λ`` (module
    docstring); ``sig`` is the prefactor vol of the sensitivities and ``spot`` the local-vol spot
    sensitivity of the lsv measure in ``SensiX`` units (:meth:`MeasureMaps.sensi_spot_lv`;
    ``VolVar += spot² + 2 spot ½ sig λ·A``, since ``ρ_Si SensiX_i = ½ sig λ_i A_i``)."""
    om1, om2, chi = float(x[0]), float(x[1]), float(x[2])
    r1 = lam[0] / om1
    r2 = lam[1] / om2
    rxy = r1 * r2 + chi * math.sqrt(max(1.0 - r1 * r1, 0.0)) * math.sqrt(max(1.0 - r2 * r2, 0.0))
    sx = 0.5 * om1 * A[:, 0] * sig
    sy = 0.5 * om2 * A[:, 1] * sig
    vv = sx * sx + sy * sy + 2.0 * rxy * sx * sy
    if spot is not None:
        vv = vv + spot * spot + 2.0 * spot * (0.5 * sig * (A @ lam))
    th = om2 / (om1 + om2)
    alpha = 1.0 / math.sqrt((1 - th) ** 2 + th * th + 2.0 * rxy * th * (1 - th))
    return np.asarray(vv, dtype=np.float64), (om1 + om2) / (2.0 * alpha)


def _correl_implied(
    svc: FloatArray, atf: FloatArray, anchor: FloatArray, sigma_0: float, volvar: FloatArray
) -> FloatArray:
    """``SpotVolCovar · atf · A / (σ_0 · sqrt(VolVar))`` — the SABR-normalised spot/vol
    correlation of the targets' convention (``= ρ_SABR`` on marking targets)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.asarray(svc * atf * anchor / (sigma_0 * np.sqrt(volvar)), dtype=np.float64)


def fit_second(targets: TargetSet, cfg: BreakEvenFitConfig, first: FirstFit) -> SecondFit:
    """The second minimisation (module docstring) from several starts (``ω_i = max(|λ_i|/0.7,
    0.05)`` scaled by 0.5, 1, 2 and ``χ ∈ {0, −0.6, +0.6}``); the best optimum is kept and the
    number of distinct optima (objective within 1% and parameters within 1e-3) reported.  The
    VolVar target follows ``cfg.volvar_target`` and the model follows ``first.measure``."""
    t0 = time.perf_counter()
    idx, _ = _select_pillars(targets, cfg.pillars)
    notes: list[str] = []
    vv_req = np.asarray(targets.vol_var, dtype=np.float64)[idx]
    vv_se = np.asarray(targets.vol_var_se, dtype=np.float64)[idx]
    atf = first.maps.naked.atf
    anchor = (
        np.asarray(targets.anchor, dtype=np.float64)[idx]
        if targets.anchor.size == targets.pillars.size
        else np.ones(idx.size)
    )
    correl_t = (
        np.asarray(targets.correl_target, dtype=np.float64)[idx]
        if targets.correl_target.size == targets.pillars.size
        else np.full(idx.size, np.nan)
    )
    svc_ach = first.table["svc_model"].to_numpy()
    if (
        cfg.volvar_target == "achieved"
        and targets.mode == "marking"
        and np.all(np.isfinite(correl_t) & (correl_t != 0))
    ):
        vovol_t = svc_ach * atf * anchor / (targets.sigma_0 * correl_t)
        vv_t = np.asarray(vovol_t * vovol_t, dtype=np.float64)
    else:
        vv_t = vv_req
        if cfg.volvar_target == "achieved":
            notes.append(
                "volvar_target='achieved' needs marking targets with rho_SABR: the requested "
                "VolVar is kept"
            )
    w = _weights(cfg.weights_volvar, vv_t)
    sw = np.sqrt(w)
    lam = first.lam
    A = first.maps.naked.A
    sig = first.maps.naked.sig
    # the spot sensitivity in SensiX units (divided by the level ratio c): the decomposition then
    # implies c (spot + rho.Sensi) = svc_model under both prefactors (implied_covariance)
    spot = first.maps.sensi_spot_lv(lam) if first.measure == "lsv" else None
    lower = np.array([abs(lam[0]) + 1e-9, abs(lam[1]) + 1e-9, cfg.chi_bounds[0]])
    upper = np.array([cfg.omega_max, cfg.omega_max, cfg.chi_bounds[1]])
    if np.any(lower >= upper):
        raise ValueError("|lambda_i| exceeds omega_max: enlarge omega_max")

    def resid(x: FloatArray) -> FloatArray:
        vv, nu = _volvar_model(x, lam, A, sig, spot)
        r = sw * (vv - vv_t)
        pen = NU_PENALTY_WEIGHT * max(nu - cfg.nu_cap, 0.0)
        return np.concatenate((r, [pen]))

    base = np.maximum(np.abs(lam) / 0.7, 0.05)
    starts = []
    for scale in (1.0, 2.0, 0.5):
        for chi0 in (0.0, -0.6, 0.6):
            x0 = np.array([base[0] * scale, base[1] * scale, chi0])
            starts.append(np.clip(x0, lower * (1 + 1e-6) + 1e-9, upper * (1 - 1e-6)))
    results = []
    for x0 in starts:
        res = least_squares(resid, x0, bounds=(lower, upper), method="trf", x_scale="jac")
        results.append(res)
    results.sort(key=lambda r: float(r.cost))
    best = results[0]
    x = np.asarray(best.x, dtype=np.float64)
    distinct = 1
    for r in results[1:]:
        if abs(r.cost - best.cost) <= 0.01 * max(best.cost, 1e-12) and not np.allclose(
            r.x, best.x, atol=1e-3, rtol=1e-3
        ):
            distinct += 1
    vv_m, nu = _volvar_model(x, lam, A, sig, spot)
    r_fit = sw * (vv_m - vv_t)
    objective = float(np.sum(r_fit * r_fit))
    n = vv_t.size
    Jm = np.asarray(best.jac, dtype=np.float64)[:n]
    names = ("omega1", "omega2", "chi")
    stderr: dict[str, float] = dict.fromkeys(names, float("nan"))
    try:
        JtJ_inv = np.linalg.inv(Jm.T @ Jm)
        if np.all(np.isfinite(vv_se)) and np.all(vv_se > 0):
            cov = JtJ_inv @ (Jm.T @ ((w * vv_se * vv_se)[:, None] * Jm)) @ JtJ_inv
        else:
            cov = np.full((3, 3), np.nan)
            notes.append(
                "no standard errors for (omega1, omega2, chi): the VolVar targets carry no "
                "sampling error (marking mode)"
            )
        stderr = {nm: float(np.sqrt(cov[i, i])) for i, nm in enumerate(names)}
    except np.linalg.LinAlgError:
        notes.append("singular Jacobian in the second fit: no standard errors")
    flags = []
    for i, nm in enumerate(names):
        if x[i] <= lower[i] * (1 + 1e-6) + 1e-9:
            flags.append(f"{nm} at lower bound")
        if x[i] >= upper[i] * (1 - 1e-6):
            flags.append(f"{nm} at upper bound")
    penalised = nu > cfg.nu_cap + 1e-9
    if penalised:
        notes.append(f"nu {nu:.3f} above the cap {cfg.nu_cap:g}: penalty active")
    table = pd.DataFrame(
        {
            "T": first.maps.T,
            "volvar_target": vv_t,
            "volvar_target_requested": vv_req,
            "volvar_model": vv_m,
            "volvol_target": np.sqrt(vv_t),
            "volvol_model": np.sqrt(vv_m),
            "sensi_spot_lv": np.zeros(n) if spot is None else spot,
            "correl_target": correl_t,
            "correl_implied": _correl_implied(svc_ach, atf, anchor, targets.sigma_0, vv_m),
        }
    )
    return SecondFit(
        float(x[0]),
        float(x[1]),
        float(x[2]),
        objective,
        stderr,
        tuple(flags),
        len(starts),
        distinct,
        bool(penalised),
        table,
        tuple(notes),
        time.perf_counter() - t0,
    )


# --------------------------------------------------------------------------------------------
# leverage diagnostics: actual (a calibrated leverage) and first-order proxy
# --------------------------------------------------------------------------------------------


def mean_abs_leverage_deviation(
    leverage: Any, surface: Any, *, n_sd: float = 2.0
) -> tuple[float, pd.DataFrame]:
    """Stage-3 statistic of a calibrated :class:`~volsto.models.leverage.LeverageFunction`: per
    leverage time ``t > 0`` the mean of ``|L(t, k) − 1|`` over the grid points with ``|k| ≤ n_sd
    atf(t) sqrt(t)``, and the unweighted mean over times (columns ``t, mean_abs_L_minus_1,
    n``)."""
    rows = []
    for i, t in enumerate(leverage.times):
        if t <= 0:
            continue
        sd = float(surface.atm_vol(t)) * np.sqrt(t)
        inside = np.abs(leverage.k_grid) <= n_sd * sd
        vals = leverage.values[i][inside]
        if vals.size == 0:
            continue
        rows.append(
            {
                "t": float(t),
                "mean_abs_L_minus_1": float(np.mean(np.abs(vals - 1.0))),
                "n": int(inside.sum()),
            }
        )
    table = pd.DataFrame(rows)
    mean = float(table["mean_abs_L_minus_1"].mean()) if len(table) else float("nan")
    return mean, table


@dataclass(frozen=True)
class LeverageProxy:
    """First-order leverage proxy (module docstring): the stage-3 statistic of ``L²(t, k) =
    σ²_Dup(t, k) / E[V_t | k]``, per time and overall, with its validity flag and notes."""

    mean_abs_l_minus_1: float
    table: pd.DataFrame
    buckets: pd.DataFrame
    valid: bool
    notes: tuple[str, ...]
    wall_seconds: float


def _ctilde_uniform(t: FloatArray, m: FloatArray, k: float) -> FloatArray:
    """``∫₀ᵗ m(u) e^{−k(t−u)} du`` on the grid ``t`` (exact for ``m`` piecewise constant at the
    interval mid-values): ``c_i = c_{i−1} e^{−k h} + m̄ (1 − e^{−k h})/k``."""
    h = np.diff(t)
    mbar = 0.5 * (m[1:] + m[:-1])
    inc = mbar * (-np.expm1(-k * h)) / k
    out = np.zeros_like(t)
    # c_i = e^{-k t_i} sum_j inc_j e^{k t_j}; stable in the grid's range (k t <= 60)
    acc = np.cumsum(inc * np.exp(k * t[1:]))
    out[1:] = acc * np.exp(-k * t[1:])
    return out


def leverage_proxy(
    params: BergomiParams | BreakEvenParams,
    surface: Any,
    xi0: ForwardVarianceCurve,
    *,
    horizon: float = 3.0,
    times: FloatArray | None = None,
    local_vol: Any | None = None,
    particle: Any | None = None,
    n_sd: float = 2.0,
    skew_ratio: float | None = None,
    nu_limit: float = DEFAULT_NU_FLAG,
    n_fine: int = 6001,
) -> LeverageProxy:
    """The convexity-adjusted Gaussian-regression leverage proxy (module docstring, verification
    evidence): ``L²(t, k) = σ²_Dup(t, k) / E[V_t | k]`` with ``ln E[V_t | k] = ln ξ₀(t) + β_t (k +
    W_t/2) − β_t² W_t/2``, ``β_t = C_t / W_t``, ``W_t = ∫₀ᵗ ξ₀``, ``C_t = Σ_i λ_i ∫₀ᵗ sqrt(ξ₀(u))
    e^{−VarY(u)/8} e^{−k_i(t−u)} du`` and ``VarY(u) = Σ_ij ω_i ω_j ρ_ij (1 − e^{−(k_i+k_j)u}) /
    (k_i + k_j)``; the statistic of :func:`mean_abs_leverage_deviation` on the particle
    calibration's time grid (default: ``TimeGrid.build([horizon], SimConfig().dt_max)``, the
    grid of a default calibration) and its Dupire ``k`` grid (``leverage_grid_config`` of a
    default :class:`~volsto.config.ParticleConfig`).  It is marked invalid when ``skew_ratio``
    (the fit's largest naked-to-market skew ratio) exceeds :data:`PROXY_SKEW_RATIO_LIMIT` or
    ``ν`` exceeds ``nu_limit`` (+ :data:`PROXY_NU_TOL`).  Measured against the four cached
    reference leverages (ν ≤ 1.74, ratio about 1): +6 / +12 / +28 / +18% (1F ω = 1, 2, 3; 2F
    Table 8.2); against calibrated fits at ν = 5 (review, 2·10⁵ particles): −53% (ratio 1.37),
    +82% and +91% (ratio about 1) — hence both limits.  A printed diagnostic, not a test
    oracle."""
    t_start = time.perf_counter()
    be = to_breakeven(params) if isinstance(params, BergomiParams) else params
    notes: list[str] = []
    if local_vol is None:
        from volsto.calibration.particle import leverage_grid_config
        from volsto.config import ParticleConfig
        from volsto.market.dupire import LocalVolSurface

        pc = particle if particle is not None else ParticleConfig(horizon=horizon)
        local_vol = LocalVolSurface.from_implied(surface, leverage_grid_config(surface, pc))
    if times is None:
        from volsto.config import SimConfig
        from volsto.engine.grid import TimeGrid

        times = np.asarray(TimeGrid.build([horizon], SimConfig().dt_max).times, dtype=np.float64)
    ts = np.asarray(times, dtype=np.float64)
    ts = ts[ts > 0]
    t_hi = float(ts.max())
    t_mid = min(0.05, 0.5 * t_hi)
    tf = np.unique(
        np.concatenate((np.linspace(0.0, t_mid, 2001), np.linspace(t_mid, t_hi, n_fine)))
    )
    xi_f = np.asarray(xi0.xi0(tf), dtype=np.float64)
    ks = (be.k1, be.k2)
    om = (be.omega1, be.omega2)
    rho = ((1.0, be.rho_XY), (be.rho_XY, 1.0))
    var_y = np.zeros_like(tf)
    for i in range(2):
        for j in range(2):
            kk = ks[i] + ks[j]
            var_y += om[i] * om[j] * rho[i][j] * (-np.expm1(-kk * tf)) / kk
    m = np.sqrt(xi_f) * np.exp(-var_y / 8.0)
    C = be.lambda1 * _ctilde_uniform(tf, m, be.k1) + be.lambda2 * _ctilde_uniform(tf, m, be.k2)
    W = np.concatenate(([0.0], np.cumsum(0.5 * np.diff(tf) * (xi_f[1:] + xi_f[:-1]))))
    beta = C / np.maximum(W, 1e-14)
    b_t = np.interp(ts, tf, beta)
    W_t = np.interp(ts, tf, W)
    xi_t = np.asarray(xi0.xi0(ts), dtype=np.float64)
    sd = np.asarray(surface.atm_vol(ts), dtype=np.float64) * np.sqrt(ts)
    kg = np.asarray(local_vol.k_grid, dtype=np.float64)
    mask = np.abs(kg)[None, :] <= n_sd * sd[:, None]
    ti, kj = np.nonzero(mask)
    kk_ = kg[kj]
    ln_ev = np.log(xi_t[ti]) + b_t[ti] * (kk_ + 0.5 * W_t[ti]) - 0.5 * b_t[ti] ** 2 * W_t[ti]
    loc = np.asarray(local_vol.local_var_k(ts[ti], kk_), dtype=np.float64)
    L = np.exp(0.5 * (np.log(loc) - ln_ev))
    counts = np.bincount(ti, minlength=ts.size)
    sums = np.bincount(ti, weights=np.abs(L - 1.0), minlength=ts.size)
    keep = counts > 0
    per_t = sums[keep] / counts[keep]
    table = pd.DataFrame({"t": ts[keep], "mean_abs_L_minus_1": per_t, "n": counts[keep]})
    edges = ((0.0, 0.05), (0.05, 0.25), (0.25, 1.0), (1.0, np.inf))
    buckets = pd.DataFrame(
        [
            {
                "t_lo": lo,
                "t_hi": hi,
                "mean_abs_L_minus_1": float(
                    table.loc[(table["t"] > lo) & (table["t"] <= hi), "mean_abs_L_minus_1"].mean()
                ),
            }
            for lo, hi in edges
        ]
    )
    mean = float(np.mean(per_t)) if per_t.size else float("nan")
    valid = True
    if skew_ratio is not None and skew_ratio > PROXY_SKEW_RATIO_LIMIT:
        valid = False
        notes.append(
            f"naked-to-market skew ratio {skew_ratio:.2f} > {PROXY_SKEW_RATIO_LIMIT:g}: the "
            "proxy's exponential regression is outside its validity (measured -53% at 1.37)"
        )
    if be.nu > nu_limit + PROXY_NU_TOL:
        valid = False
        notes.append(
            f"nu {be.nu:.2f} > {nu_limit:g}: the proxy is outside its validated range (measured "
            "+82% / +91% at nu 5)"
        )
    if not np.isfinite(mean) or mean > 1.0:
        valid = False
        notes.append(f"proxy mean |L - 1| = {mean:.3g}: outside the first-order regime")
    return LeverageProxy(mean, table, buckets, valid, tuple(notes), time.perf_counter() - t_start)


# --------------------------------------------------------------------------------------------
# stage 3
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Stage3Inputs:
    """What the validation needs beyond the fit: the pricing date's implied surface, the
    particle / simulation settings, the optional local-vol config, the pricing configuration,
    the pillars of each check and, optionally, an already calibrated ``model`` (then nothing
    is calibrated and ``recalibrated`` is ``False``; the model's own kernel is used for the
    naked-skew check).  ``mixing_paths`` / ``mixing_seed`` / ``mixing_dt`` set the exact
    mixing derivative of check (c)."""

    surface: Any  # ImpliedSurface
    particle: Any  # ParticleConfig
    sim: Any  # SimConfig (calibration schedule)
    pricing_sim: Any  # SimConfig
    local_vol: Any | None = None
    ssr_pillars: tuple[float, ...] = (0.25, 1.0)
    breakeven_pillars: tuple[float, ...] = (0.25, 1.0)
    forward_starts: tuple[tuple[float, float], ...] = ((1.0, 2.0), (2.0, 3.0))
    headline: bool = False
    eps: float = 0.05
    model: Any | None = None
    mixing_paths: int = 100_000
    mixing_seed: int = 11
    mixing_dt: float = 1.0 / 365.0


@dataclass(frozen=True)
class Stage3Report:
    """Stage-3 tables (module docstring) with the wall clocks and the recalibration flag."""

    mean_abs_l_minus_1: float
    leverage_table: pd.DataFrame
    ssr_table: pd.DataFrame
    skew_table: pd.DataFrame
    breakeven_table: pd.DataFrame
    forward_table: pd.DataFrame
    headline: pd.DataFrame | None
    calibration_seconds: float
    wall_seconds: float
    recalibrated: bool
    n_particles: int
    n_paths: int

    def summary(self) -> str:
        lines = [
            f"stage 3: mean |L - 1| {self.mean_abs_l_minus_1:.4f}; recalibrated: "
            f"{'yes' if self.recalibrated else 'no'} (calibration "
            f"{self.calibration_seconds:.0f} s, {self.n_particles} particles); pricing "
            f"{self.n_paths} paths; wall clock {self.wall_seconds:.0f} s",
            "numerical SSR of the LSV (the truth) vs ssr_target and the fit's first-order SSRs:",
            self.ssr_table.round(4).to_string(index=False),
            "naked skew (mixing derivative) vs market skew:",
            self.skew_table.round(4).to_string(index=False),
            "simulated break-evens of the LSV vs targets:",
            self.breakeven_table.round(5).to_string(index=False),
        ]
        if len(self.forward_table):
            lines += ["forward smiles:", self.forward_table.round(4).to_string(index=False)]
        return "\n".join(lines)


def _target_at(targets: TargetSet, T: float, values: FloatArray) -> float:
    tp = np.asarray(targets.pillars, dtype=np.float64)
    i = int(np.argmin(np.abs(tp - T)))
    return float(values[i]) if abs(tp[i] - T) <= _TOL_T else float("nan")


def _table_at(table: pd.DataFrame | None, T: float, column: str) -> float:
    if table is None or column not in table:
        return float("nan")
    tt = table["T"].to_numpy(dtype=float)
    i = int(np.argmin(np.abs(tt - T)))
    return float(table[column].iloc[i]) if abs(tt[i] - T) <= _TOL_T else float("nan")


def stage3_validation(
    params: BergomiParams,
    inputs: Stage3Inputs,
    targets: TargetSet,
    *,
    fit_table: pd.DataFrame | None = None,
) -> Stage3Report:
    """Stage 3 of the module docstring: calibrate the leverage on ``inputs.surface`` for
    ``params`` (or take ``inputs.model``) and report the validation tables; nothing is refit.
    With the fit's per-pillar ``fit_table`` the SSR table also carries the first-order
    ``ssr_achieved_lsv`` / ``ssr_achieved_naked_vs_market`` next to the numerical LSV SSR."""
    from volsto.analytics.breakeven import simulated_breakevens
    from volsto.analytics.forward_smile import forward_smile
    from volsto.analytics.smile_dynamics import ssr_numerical_many
    from volsto.calibration.history import mixing_atmf_batch
    from volsto.calibration.particle import calibrate_leverage
    from volsto.models.bergomi import BergomiSV
    from volsto.models.lsv import LSV

    t_all = time.perf_counter()
    surface = inputs.surface
    if inputs.model is None:
        fc = surface.forward_curve
        xi0 = xi0_curve(surface, min(surface.max_maturity, max(inputs.particle.horizon + 1.0, 5.0)))
        kernel = BergomiSV(params, xi0, fc)
        t0 = time.perf_counter()
        result = calibrate_leverage(
            surface, kernel, inputs.particle, inputs.sim, local_vol_cfg=inputs.local_vol
        )
        cal_s = time.perf_counter() - t0
        lsv = LSV(kernel, result.leverage)
        recalibrated = True
        n_particles = int(inputs.particle.n_particles)
    else:
        lsv = inputs.model
        kernel = lsv.kernel
        cal_s = 0.0
        recalibrated = False
        n_particles = int(lsv.leverage.metadata.get("n_particles", 0))
    # (a) mean |L - 1| over the leverage grid inside +/-2 sd of log-moneyness per time
    mean_abs, lev_table = mean_abs_leverage_deviation(lsv.leverage, surface)
    psim = inputs.pricing_sim
    # (b) numerical SSR of the LSV vs ssr_target and the fit's first-order measures
    ssr_rows = ssr_numerical_many(lsv, list(inputs.ssr_pillars), eps=inputs.eps, sim=psim)
    ssr_t = np.array([_target_at(targets, r.T, targets.ssr_target) for r in ssr_rows])
    ssr_m = np.array([r.R for r in ssr_rows])
    ssr_se = np.array([r.R_stderr for r in ssr_rows])
    ssr_table = pd.DataFrame(
        {
            "T": [r.T for r in ssr_rows],
            "ssr_model": ssr_m,
            "ssr_model_se": ssr_se,
            "ssr_target": ssr_t,
            "z": (ssr_m - ssr_t) / ssr_se,
            "ssr_achieved_lsv": [_table_at(fit_table, r.T, "ssr_achieved_lsv") for r in ssr_rows],
            "ssr_achieved_naked_vs_market": [
                _table_at(fit_table, r.T, "ssr_achieved_naked_vs_market") for r in ssr_rows
            ],
            "skew_lsv": [r.skew for r in ssr_rows],
            "skew_lsv_se": [r.skew_stderr for r in ssr_rows],
        }
    )
    # (c) naked skew (exact mixing derivative) vs the market skew target
    nf = int(kernel.n_factors)
    skew_rows = []
    for T in inputs.breakeven_pillars:
        st = _target_at(targets, float(T), targets.skew_target)
        try:
            mb = mixing_atmf_batch(
                kernel,
                float(T),
                np.array([0.0]),
                np.zeros((1, nf)),
                n_paths=inputs.mixing_paths,
                seed=inputs.mixing_seed,
                dt=inputs.mixing_dt,
            )
            sk, sk_se, av, note = float(mb.skew[0]), float(mb.skew_se[0]), float(mb.atm_vol[0]), ""
        except ValueError as exc:  # the mixing solution needs |spot/vol correlation| < 1
            sk, sk_se, av, note = float("nan"), float("nan"), float("nan"), str(exc)
        skew_rows.append(
            {
                "T": float(T),
                "skew_naked": sk,
                "skew_naked_se": sk_se,
                "skew_target": st,
                "gap": sk / st - 1.0 if st else float("nan"),
                "skew_naked_first_order": _table_at(fit_table, float(T), "skew_naked"),
                "atmf_vol_naked": av,
                "note": note,
            }
        )
    skew_table = pd.DataFrame(skew_rows)
    # (d) simulated break-evens of the LSV vs the targets
    be_rows = []
    for T in inputs.breakeven_pillars:
        b = simulated_breakevens(lsv, float(T), sim=psim, eps=inputs.eps, sigma_0=targets.sigma_0)
        svc_t = _target_at(targets, float(T), targets.spot_vol_covar)
        vv_t = _target_at(targets, float(T), targets.vol_var)
        atf_t = _target_at(targets, float(T), targets.atf)
        be_rows.append(
            {
                "T": float(T),
                "svc_sim": b.spot_vol_covar,
                "svc_se": b.spot_vol_covar_se,
                "svc_target": svc_t,
                "svc_z": (
                    (b.spot_vol_covar - svc_t) / b.spot_vol_covar_se
                    if b.spot_vol_covar_se > 0
                    else float("nan")
                ),
                "volvar_sim": b.vol_var,
                "volvar_se": b.vol_var_se,
                "volvar_target": vv_t,
                "volvar_z": (b.vol_var - vv_t) / b.vol_var_se if b.vol_var_se > 0 else float("nan"),
                "vovol_sim": b.vovol,
                "vovol_target": np.sqrt(vv_t) / atf_t if atf_t > 0 else float("nan"),
                "ssr_sim": b.ssr,
                "atmf_vol_sim": b.sigma_hat,
            }
        )
    be_table = pd.DataFrame(be_rows)
    # (e) forward smiles
    fwd_rows = []
    for t1, t2 in inputs.forward_starts:
        sm = forward_smile(lsv, t1, t2, [0.9, 1.1], psim)  # the ATM-forward strike is added
        i_atm = int(np.argmin(np.abs(sm.strikes - sm.forward_ratio)))
        i_lo = int(np.argmin(np.abs(sm.strikes - 0.9)))
        i_hi = int(np.argmin(np.abs(sm.strikes - 1.1)))
        fwd_rows.append(
            {
                "t1": t1,
                "t2": t2,
                "atm_fwd_vol": float(sm.vols[i_atm]),
                "atm_fwd_vol_se": float(sm.vol_stderr[i_atm]),
                "skew_90_110": float(sm.vols[i_lo] - sm.vols[i_hi]),
                "skew_90_110_se": float(np.hypot(sm.vol_stderr[i_lo], sm.vol_stderr[i_hi])),
            }
        )
    forward_table = pd.DataFrame(fwd_rows)
    headline = None
    if inputs.headline:
        from volsto.studies.m6 import run_m6_headline

        headline = run_m6_headline({"fit": lsv}, psim).table
    return Stage3Report(
        mean_abs,
        lev_table,
        ssr_table,
        skew_table,
        be_table,
        forward_table,
        headline,
        cal_s,
        time.perf_counter() - t_all,
        recalibrated,
        n_particles,
        int(psim.n_paths),
    )


# --------------------------------------------------------------------------------------------
# attainable SSR band (floor / ceiling)
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class AttainableSSR:
    """The SSR band attainable at one skew weight (:func:`attainable_ssr`), in **achieved-SSR
    units** (mean over pillars of the first-order SSR under ``ssr_measure``).

    ``scan``: one row per scanned mean target ``r`` (the curve ``r · shape``; ``kind`` ``grid``,
    ``request`` or ``grid+request``) with ``ssr_achieved_mean, tracking_error`` (mean achieved −
    ``r``), ``max_abs_pillar_error, k1, k1_at_bound, lambda1, lambda2, nu_min, nu_limit_binding,
    mean_skew_gap, max_skew_ratio, objective`` and ``ssr_<T>`` per pillar.  ``floor`` /
    ``ceiling``: the lowest / highest mean achieved SSR over the scan, attained at the scan
    target ``floor_target`` / ``ceiling_target``; ``floor_at_scan_edge`` /
    ``ceiling_at_scan_edge`` when that target is the end of the scan and the achieved SSR still
    moves there by more than ``saturation_slope`` per unit of target (the floor is then a
    property of the scan range, not of the model); ``floor_bounds`` / ``ceiling_bounds`` the
    limits binding there (``k1`` bound, ν limit).  ``tracking_segments``: the contiguous ranges of
    scan targets whose mean tracking error is within ``ssr_tol``; ``strict_tracking_segments`` the
    same with every pillar within ``ssr_tol``; ``basin_jumps``: neighbouring scan rows whose
    ``k1`` differ by more than :data:`K1_BASIN_JUMP_RATIO` (``(r_lo, r_hi, k1_lo, k1_hi)``).
    ``pillar_band``: per pillar the lowest / highest SSR reached and the first / last scan target
    tracking within ``ssr_tol`` (with the number of tracking segments)."""

    skew_weight: float
    ssr_measure: str
    ssr_tol: float
    shape: FloatArray
    scan: pd.DataFrame
    pillar_band: pd.DataFrame
    floor: float
    floor_target: float
    ceiling: float
    ceiling_target: float
    floor_at_scan_edge: bool
    ceiling_at_scan_edge: bool
    floor_bounds: tuple[str, ...]
    ceiling_bounds: tuple[str, ...]
    tracking_segments: tuple[tuple[float, float], ...]
    strict_tracking_segments: tuple[tuple[float, float], ...]
    basin_jumps: tuple[tuple[float, float, float, float], ...]
    saturation_slope: float
    wall_seconds: float

    def summary(self) -> str:
        def seg(s: tuple[tuple[float, float], ...]) -> str:
            return ", ".join(f"[{a:.2f}, {b:.2f}]" for a, b in s) or "none"

        def edge(flag: bool, bounds: tuple[str, ...]) -> str:
            parts = (["scan edge, not saturated"] if flag else []) + list(bounds)
            return f" ({'; '.join(parts)})" if parts else ""

        jumps = ", ".join(
            f"r {a:.2f}->{b:.2f}: k1 {c:.3g}->{d:.3g}" for a, b, c, d in self.basin_jumps
        )
        return (
            f"attainable SSR (first order, {self.ssr_measure} measure, skew_weight "
            f"{format_skew_weight(self.skew_weight)}): floor {self.floor:.3f} at scan target "
            f"{self.floor_target:.3f}{edge(self.floor_at_scan_edge, self.floor_bounds)}, ceiling "
            f"{self.ceiling:.3f} at scan target {self.ceiling_target:.3f}"
            f"{edge(self.ceiling_at_scan_edge, self.ceiling_bounds)}; tracking band (mean within "
            f"ssr_tol {self.ssr_tol:g}) {seg(self.tracking_segments)}, every pillar "
            f"{seg(self.strict_tracking_segments)}; k1 basin jumps: {jumps or 'none'}; wall "
            f"clock {self.wall_seconds:.1f} s; recalibrated: no\n"
            + self.pillar_band.round(3).to_string(index=False)
        )


def _bound_labels(row: Any) -> tuple[str, ...]:
    out = []
    if row["k1_at_bound"]:
        out.append(f"k1 at bound ({row['k1']:.3g})")
    if row["nu_limit_binding"]:
        out.append(f"nu limit binding (nu_min {row['nu_min']:.3f})")
    return tuple(out)


def _scan_row(
    prob: _FirstProblem, cfg: BreakEvenFitConfig, r: float, shape: FloatArray, kind: str
) -> dict[str, Any]:
    target = r * shape
    p = prob.with_ssr(target)
    best, _, _ = _optimise_k1(p, cfg)
    R = best.maps.ssr(best.lam, prob.measure)
    s = best.maps.skew_naked(best.lam) / prob.skew_market
    k1 = best.k1
    row: dict[str, Any] = {
        "ssr_target": float(r),
        "kind": kind,
        "ssr_achieved_mean": float(np.mean(R)),
        "tracking_error": float(np.mean(R) - r),
        "max_abs_pillar_error": float(np.max(np.abs(R - target))),
        "k1": k1,
        "k1_at_bound": bool(
            k1 <= cfg.k1_bounds[0] * (1 + 1e-6) or k1 >= cfg.k1_bounds[1] * (1 - 1e-6)
        ),
        "lambda1": float(best.lam[0]),
        "lambda2": float(best.lam[1]),
        "nu_min": 0.5 * abs(float(np.sum(best.lam))),
        "nu_limit_binding": bool(best.active),
        "mean_skew_gap": float(np.mean(np.abs(s - 1.0))),
        "max_skew_ratio": float(np.max(s)),
        "objective": best.objective,
    }
    for T, x in zip(prob.T, R):
        row[f"ssr_{T:g}"] = float(x)
    return row


def _segments(r: FloatArray, ok: NDArray[np.bool_]) -> tuple[tuple[float, float], ...]:
    out: list[tuple[float, float]] = []
    start: int | None = None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == ok.size - 1):
            end = i if flag else i - 1
            out.append((float(r[start]), float(r[end])))
            start = None
    return tuple(out)


def attainable_ssr(
    targets: TargetSet | Any,
    cfg: BreakEvenFitConfig | None = None,
    xi0: ForwardVarianceCurve | None = None,
    *,
    ssr_grid: Sequence[float] | FloatArray = DEFAULT_SSR_GRID,
    shape: Sequence[float] | FloatArray | None = None,
    include: Sequence[float] = (),
    saturation_slope: float = DEFAULT_SATURATION_SLOPE,
    lv_ssr: LocalVolSSR | None = None,
    anchor_power: float = 1.0,
    h: float = SABR_CURVATURE_H,
) -> AttainableSSR:
    """The achievable SSR band at ``cfg.skew_weight`` (owner's SSR floor reporting; module
    docstring "Attainable floor / ceiling"): the first minimisation is run at the SSR target
    curve ``r · shape`` for every mean target ``r`` of ``ssr_grid`` and ``include`` (``shape``
    per fitted pillar, default constant 1; :func:`fit_2f` passes the request's normalised curve
    and its mean, so that the band always contains the unclamped fit).  The floor / ceiling are
    the lowest / highest **mean achieved SSR** over the scan — the SSR the fitter can deliver at
    this weight, in the units of the message — with the scan target attaining them, the
    scan-edge / saturation flag and the binding limits.  Tracking segments (mean and every
    pillar within ``cfg.ssr_tol``) and ``k1`` basin jumps are reported separately; nothing is
    bisected or assumed contiguous.  ``targets`` is a :class:`TargetSet` or a surface (marking
    targets on ``cfg.pillars`` with ``anchor_power`` / ``h``; ``ξ₀`` its variance-swap strip to the
    last pillar when ``xi0`` is None).  Only the first minimisation runs (about a second);
    nothing is calibrated.  Checked by ``tests/test_fit_2f.py::test_attainable_ssr_band``."""
    t0 = time.perf_counter()
    c = cfg or BreakEvenFitConfig()
    if not isinstance(targets, TargetSet):
        surface = targets
        targets = marking_targets(surface, c.pillars, anchor_power=anchor_power, h=h)
        if xi0 is None:
            xi0 = xi0_curve(surface, float(min(surface.max_maturity, max(targets.pillars))))
    if xi0 is None:
        raise ValueError("attainable_ssr needs the forward-variance curve xi0 of the targets")
    c, _ = _resolved_config(c, targets)
    prob, _ = _first_problem(targets, c, xi0, lv_ssr)
    sh = np.ones(prob.T.size) if shape is None else np.asarray(shape, dtype=np.float64)
    if sh.shape != prob.T.shape or not np.all(np.isfinite(sh)):
        raise ValueError("shape must be finite with one value per fitted pillar")
    grid = np.asarray(ssr_grid, dtype=np.float64)
    if grid.size < 2 or np.any(np.diff(grid) <= 0):
        raise ValueError("ssr_grid must be increasing with at least two points")
    extra = [float(x) for x in include if not np.any(np.abs(grid - float(x)) <= 1e-12)]
    inc = np.asarray([float(x) for x in include], dtype=np.float64)
    rows = [
        _scan_row(
            prob, c, float(r), sh, "grid+request" if np.any(np.abs(inc - r) <= 1e-12) else "grid"
        )
        for r in grid
    ]
    rows += [_scan_row(prob, c, x, sh, "request") for x in extra]
    scan = pd.DataFrame(rows).sort_values("ssr_target", kind="stable").reset_index(drop=True)
    r = scan["ssr_target"].to_numpy(dtype=float)
    ach = scan["ssr_achieved_mean"].to_numpy(dtype=float)
    i_lo, i_hi = int(np.argmin(ach)), int(np.argmax(ach))

    def at_edge(i: int, end: int, nb: int) -> bool:
        if i != end:
            return False
        slope = abs(ach[nb] - ach[end]) / abs(r[nb] - r[end])
        return bool(slope > saturation_slope)

    floor_edge = at_edge(i_lo, 0, 1)
    ceiling_edge = at_edge(i_hi, r.size - 1, r.size - 2)
    tol = c.ssr_tol
    err = scan["tracking_error"].to_numpy(dtype=float)
    ok = np.abs(err) <= tol
    strict = scan["max_abs_pillar_error"].to_numpy(dtype=float) <= tol
    k1 = scan["k1"].to_numpy(dtype=float)
    jumps = tuple(
        (float(r[i]), float(r[i + 1]), float(k1[i]), float(k1[i + 1]))
        for i in range(r.size - 1)
        if max(k1[i], k1[i + 1]) > K1_BASIN_JUMP_RATIO * min(k1[i], k1[i + 1])
    )
    band_rows = []
    for T, s_i in zip(prob.T, sh):
        col = scan[f"ssr_{T:g}"].to_numpy(dtype=float)
        inside = np.abs(col - r * s_i) <= tol
        idx_in = np.flatnonzero(inside)
        band_rows.append(
            {
                "T": float(T),
                "ssr_min_achieved": float(col.min()),
                "ssr_max_achieved": float(col.max()),
                "first_tracking_target": float(r[idx_in[0]]) if idx_in.size else float("nan"),
                "last_tracking_target": float(r[idx_in[-1]]) if idx_in.size else float("nan"),
                "n_tracking_segments": len(_segments(r, inside)),
            }
        )
    return AttainableSSR(
        float(c.skew_weight),
        c.ssr_measure,
        float(tol),
        sh,
        scan,
        pd.DataFrame(band_rows),
        float(ach[i_lo]),
        float(r[i_lo]),
        float(ach[i_hi]),
        float(r[i_hi]),
        floor_edge,
        ceiling_edge,
        _bound_labels(scan.iloc[i_lo]),
        _bound_labels(scan.iloc[i_hi]),
        _segments(r, ok),
        _segments(r, strict),
        jumps,
        float(saturation_slope),
        time.perf_counter() - t0,
    )


# --------------------------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FitResult:
    """The fit: book and break-even parameters, the fitted targets (at the requested SSR, or at
    the floor / ceiling level ``Y`` when clamped — the second fit's VolVar target), both
    minimisations, the per-pillar table (``T, atf, ssr_requested, ssr_target`` (the first
    minimisation's target: the scan target ``r*`` when clamped), ``ssr_fitted`` (the fitted
    targets' SSR), ``svc_target, svc_model, ssr_achieved, ssr_achieved_lsv,
    ssr_achieved_naked_vs_market, ssr_naked_own, r_lv_market, skew_market, skew_naked,
    skew_gap_rel, correl_target, correl_implied, volvar_target, volvar_model, volvol_target,
    volvol_model, sensi_spot_lv``; every SSR first order), the resolved config, the paired risk
    regime, the optional stage 3, the wall clock, whether a leverage was recalibrated, the
    owner's floor / ceiling ``message`` (None inside the attainable band) with its details (always
    the tracking error and the band), the attainable band, the leverage proxy when a surface was
    given."""

    params: BergomiParams
    breakeven: BreakEvenParams
    targets: TargetSet
    xi0: ForwardVarianceCurve
    config: BreakEvenFitConfig
    first: FirstFit
    second: SecondFit
    table: pd.DataFrame
    risk_regime: str
    stage3: Stage3Report | None
    wall_seconds: float
    recalibrated: bool
    pricing_date: pd.Timestamp | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)
    message: str | None = None
    message_details: tuple[str, ...] = field(default_factory=tuple)
    ssr_requested: FloatArray = field(default_factory=lambda: np.zeros(0))
    attainable: AttainableSSR | None = None
    leverage_proxy: LeverageProxy | None = None
    lv_ssr: LocalVolSSR | None = None

    @property
    def ssr_achieved_mean(self) -> float:
        """Mean over pillars of the SSR achieved under the configured measure."""
        return float(self.table["ssr_achieved"].mean())

    @property
    def ssr_achieved_lsv_mean(self) -> float:
        return float(self.table["ssr_achieved_lsv"].mean())

    @property
    def ssr_achieved_naked_mean(self) -> float:
        return float(self.table["ssr_achieved_naked_vs_market"].mean())

    @property
    def mean_skew_gap(self) -> float:
        return self.first.mean_skew_gap

    @property
    def config_yaml(self) -> str:
        """A loadable model config: ``model`` (:class:`BergomiParams`, ``load_yaml(path,
        BergomiParams, section="model")``), ``breakeven`` (:class:`BreakEvenParams`), the
        ``risk_regime`` and the provenance (mode, targets, settings, achieved SSRs, message,
        objectives, standard errors, flags)."""
        f, s = self.first, self.second
        tg = self.targets
        cfg = self.config
        doc: dict[str, Any] = {
            "model": to_mapping(self.params),
            "breakeven": to_mapping(self.breakeven),
            "risk_regime": self.risk_regime,
            "provenance": {
                "mode": tg.mode,
                "pricing_date": (
                    None if self.pricing_date is None else str(self.pricing_date.date())
                ),
                "pillars": [float(t) for t in f.maps.T],
                "sigma_0": float(tg.sigma_0),
                "ssr_requested": [float(x) for x in self.ssr_requested],
                "ssr_fitted": [float(x) for x in tg.ssr_target],
                "ssr_first_target": [float(x) for x in f.table["ssr_target"]],
                "anchor_power": (
                    None if not np.isfinite(tg.anchor_power) else float(tg.anchor_power)
                ),
                "k2_fixed": float(cfg.k2),
                "k1_bounds": [float(x) for x in cfg.k1_bounds],
                "skew_weight": float(cfg.skew_weight),
                "ssr_measure": cfg.ssr_measure,
                "ssr_tol": float(cfg.ssr_tol),
                "clamp_to_attainable": bool(cfg.clamp_to_attainable),
                "term_structure": cfg.term_structure,
                "term_structure_source": tg.term_structure_source,
                "lv_ssr": cfg.lv_ssr,
                "volvar_target": cfg.volvar_target,
                "weights": {"ssr": cfg.weights_ssr, "volvar": cfg.weights_volvar},
                "nu_cap": float(cfg.nu_cap),
                "sigma_hat_prefactor": cfg.sigma_hat_prefactor,
                "nu_flag": float(cfg.nu_flag),
                "message": self.message,
                "message_details": list(self.message_details),
                "attainable": (
                    None
                    if self.attainable is None
                    else {
                        "floor": float(self.attainable.floor),
                        "floor_target": float(self.attainable.floor_target),
                        "floor_at_scan_edge": bool(self.attainable.floor_at_scan_edge),
                        "ceiling": float(self.attainable.ceiling),
                        "ceiling_target": float(self.attainable.ceiling_target),
                        "ceiling_at_scan_edge": bool(self.attainable.ceiling_at_scan_edge),
                    }
                ),
                "first_order_valid": bool(f.first_order_valid),
                "achieved": {
                    "ssr_measure_mean": self.ssr_achieved_mean,
                    "ssr_lsv": [float(x) for x in self.table["ssr_achieved_lsv"]],
                    "ssr_naked_vs_market": [
                        float(x) for x in self.table["ssr_achieved_naked_vs_market"]
                    ],
                    "ssr_naked_own": [float(x) for x in self.table["ssr_naked_own"]],
                    "skew_gap_rel": [float(x) for x in self.table["skew_gap_rel"]],
                    "leverage_proxy_mean_abs_L_minus_1": (
                        None
                        if self.leverage_proxy is None
                        else float(self.leverage_proxy.mean_abs_l_minus_1)
                    ),
                },
                "first": {
                    "objective": float(f.objective),
                    "objective_ssr": float(f.objective_ssr),
                    "objective_skew": float(f.objective_skew),
                    "k1_se": float(f.k1_se),
                    "lambda1_se": float(f.lambda1_se),
                    "lambda2_se": float(f.lambda2_se),
                    "nu_limit_active": list(f.active_labels),
                    "k1_at_bound": bool(f.k1_at_bound),
                    "notes": list(f.notes),
                },
                "second": {
                    "objective": float(s.objective),
                    "stderr": {k: float(v) for k, v in s.stderr.items()},
                    "bound_flags": list(s.bound_flags),
                    "n_starts": int(s.n_starts),
                    "n_distinct_optima": int(s.n_distinct),
                    "nu_penalised": bool(s.nu_penalised),
                    "notes": list(s.notes),
                },
                "target_flags": list(tg.flags),
                "notes": list(self.notes),
                "stage3_mean_abs_L_minus_1": (
                    None if self.stage3 is None else float(self.stage3.mean_abs_l_minus_1)
                ),
                "recalibrated": bool(self.recalibrated),
                "wall_seconds": float(self.wall_seconds),
            },
        }
        return str(yaml.safe_dump(doc, sort_keys=False))

    def write_yaml(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.config_yaml, encoding="utf-8")
        return p

    def summary(self) -> str:
        p, b, f, s = self.params, self.breakeven, self.first, self.second
        cfg = self.config
        when = "" if self.pricing_date is None else f" @ {self.pricing_date.date()}"
        lines = []
        if self.message:
            lines.append(f"MESSAGE: {self.message}")
        lines += [f"  - {d}" for d in self.message_details]
        proxy = (
            ""
            if self.leverage_proxy is None
            else f"; leverage proxy mean |L - 1| {self.leverage_proxy.mean_abs_l_minus_1:.3f}"
            + ("" if self.leverage_proxy.valid else " (invalid)")
        )
        lines += [
            f"fit_2f [{self.targets.mode}]{when}: skew_weight "
            f"{format_skew_weight(cfg.skew_weight)}, measure {cfg.ssr_measure}; first-order "
            f"achieved SSR mean {self.ssr_achieved_mean:.3f} (lsv {self.ssr_achieved_lsv_mean:.3f}"
            f", naked vs market {self.ssr_achieved_naked_mean:.3f}; "
            f"{'validated regime' if f.first_order_valid else 'OUTSIDE the validated regime'})"
            f", mean naked-skew gap {self.mean_skew_gap:.3f}{proxy}",
            f"nu {p.nu:.3f} theta {p.theta:.3f} "
            f"k1 {p.k1:.3f} (se {f.k1_se:.3f}) k2 {p.k2:.3f} (fixed) rho12 {p.rho12:+.3f} "
            f"rho_SX1 {p.rho_SX1:+.3f} rho_SX2 {p.rho_SX2:+.3f}; "
            f"break-even omega1 {b.omega1:.3f} omega2 {b.omega2:.3f} "
            f"lambda1 {b.lambda1:+.3f} (se {f.lambda1_se:.3f}) lambda2 {b.lambda2:+.3f} "
            f"(se {f.lambda2_se:.3f}) chi {b.chi:+.3f} (se {s.stderr['chi']:.3f}); "
            f"objectives {f.objective:.3e} (ssr {f.objective_ssr:.3e}, skew "
            f"{f.objective_skew:.3e}) / {s.objective:.3e}; nu limit {list(f.active_labels)}; "
            f"bounds {list(s.bound_flags)}; risk regime {self.risk_regime}; "
            f"wall clock {self.wall_seconds:.1f} s; recalibrated: "
            f"{'yes' if self.recalibrated else 'no'}",
            self.table.round(5).to_string(index=False),
        ]
        if self.notes or f.notes or s.notes:
            lines.append("notes: " + "; ".join((*self.notes, *f.notes, *s.notes)))
        if self.stage3 is not None:
            lines.append(self.stage3.summary())
        return "\n".join(lines)


def _clamp(
    targets: TargetSet,
    cfg: BreakEvenFitConfig,
    xi0: ForwardVarianceCurve,
    first: FirstFit,
    idx: NDArray[Any],
    lv_ssr: LocalVolSSR | None,
) -> tuple[TargetSet, TargetSet, FirstFit, str | None, list[str], AttainableSSR]:
    """The attainable band, the floor / ceiling clamp and the message of :func:`fit_2f` (module
    docstring).  Returns ``(first-fit targets, fitted targets, first fit, message, details,
    band)``: the first-fit targets carry the scan target ``r* · shape`` that attains the floor
    (ceiling) when clamped, the fitted targets the SSR level ``Y · shape`` the message reports —
    the VolVar target of the second fit is built on it (a scan target of 0 would ask for no vol
    of vol)."""
    requested = np.asarray(targets.ssr_target, dtype=np.float64)[idx]
    x = float(np.mean(requested))
    if abs(x) <= 1e-12:
        if float(np.ptp(requested)) > 1e-12:
            raise ValueError("a curve ssr_target needs a non-zero mean (the scan scales its shape)")
        shape = np.ones(requested.size)
    else:
        shape = requested / x
    att = attainable_ssr(targets, cfg, xi0, shape=shape, include=(x,), lv_ssr=lv_ssr)
    z = first.ssr_achieved_mean
    tab = first.table
    err = tab["ssr_achieved"].to_numpy() - requested
    worst = int(np.argmax(np.abs(err)))
    m = cfg.ssr_measure
    curve = float(np.ptp(requested)) > 1e-12
    details: list[str] = [
        f"first-order mean {m}-measure SSR {z:.3f} at ssr_target={x:.3f}"
        f"{' (curve mean)' if curve else ''}: tracking error {z - x:+.3f} "
        f"({'within' if abs(z - x) <= cfg.ssr_tol else 'beyond'} ssr_tol={cfg.ssr_tol:g}), worst "
        f"pillar T={float(tab['T'].iloc[worst]):g} {err[worst]:+.3f}",
        att.summary().splitlines()[0],
    ]
    if not first.first_order_valid:
        details.append(first_order_regime_note(first.nu_min, first.max_skew_ratio, cfg.nu_flag))
    below, above = x < att.floor, x > att.ceiling
    first_targets, fitted_targets, new_first, message = targets, targets, first, None
    w = format_skew_weight(cfg.skew_weight)
    if below or above:
        y = att.floor if below else att.ceiling
        r_star = att.floor_target if below else att.ceiling_target
        edge = att.floor_at_scan_edge if below else att.ceiling_at_scan_edge
        word = "floor" if below else "ceiling"
        if edge:
            details.append(
                f"the {word} sits at the end of the scan and is not saturated: a wider ssr_grid "
                f"reaches {'lower' if below else 'higher'} SSR"
            )
        if cfg.clamp_to_attainable:
            full = np.asarray(targets.ssr_target, dtype=np.float64).copy()
            full[idx] = r_star * shape
            first_targets = targets.with_ssr_target(full)
            new_first = fit_first(first_targets, cfg, xi0, lv_ssr=lv_ssr)
            full_y = np.asarray(targets.ssr_target, dtype=np.float64).copy()
            full_y[idx] = y * shape
            fitted_targets = targets.with_ssr_target(full_y)
            template = FLOOR_MESSAGE if below else CEILING_MESSAGE
            message = template.format(x=x, y=y, w=w)
            details.append(
                f"clamped: the first minimisation refit at the scan target {r_star:.3f}"
                f"{' x the requested shape' if curve else ''}, which attains the {word} "
                f"(mean achieved {new_first.ssr_achieved_mean:.3f}, unclamped {z:.3f}); the "
                f"VolVar target is built at the fitted SSR {y:.3f}"
            )
        else:
            template = FLOOR_MESSAGE_UNCLAMPED if below else CEILING_MESSAGE_UNCLAMPED
            message = template.format(x=x, y=y, z=z, w=w)
    tab = new_first.table
    details.append(
        "first-order achieved SSR per pillar: "
        + ", ".join(
            f"T={T:g}: {m} {r:.3f} (lsv {rl:.3f}, naked vs market {rn:.3f})"
            for T, r, rl, rn in zip(
                tab["T"],
                tab["ssr_achieved"],
                tab["ssr_achieved_lsv"],
                tab["ssr_achieved_naked_vs_market"],
            )
        )
    )
    s = (tab["skew_naked"] / tab["skew_market"]).to_numpy()
    if m == "lsv" and np.any(s > 1.0):
        details.append(
            "naked skew steeper than market at T = "
            f"{[float(t) for t in tab['T'][s > 1.0]]} (ratios {np.round(s[s > 1.0], 2).tolist()}): "
            "the leverage is counter-skewed there, as eq. 12.52 requires for a low LSV SSR"
        )
    return first_targets, fitted_targets, new_first, message, details, att


def fit_2f(
    targets: TargetSet,
    xi0: ForwardVarianceCurve,
    cfg: BreakEvenFitConfig | None = None,
    *,
    stage3: Stage3Inputs | None = None,
    pricing_date: pd.Timestamp | None = None,
    lv_ssr: LocalVolSSR | None = None,
    proxy_surface: Any | None = None,
) -> FitResult:
    """Both minimisations on ``targets`` with the forward-variance curve ``xi0``, the attainable
    band with the floor / ceiling clamp and the owner's message, the leverage proxy when
    ``proxy_surface`` is given and stage 3 when ``stage3`` is given (module docstring).  The
    ``"auto"`` SSR measure is resolved on ``targets`` (:func:`resolve_ssr_measure`) and the
    resolved config is returned in :attr:`FitResult.config`."""
    t0 = time.perf_counter()
    c, measure_note = _resolved_config(cfg or BreakEvenFitConfig(), targets)
    idx, notes = _select_pillars(targets, c.pillars)
    if measure_note:
        notes.append(measure_note)
    requested = np.asarray(targets.ssr_target, dtype=np.float64)[idx].copy()
    first = fit_first(targets, c, xi0, lv_ssr=lv_ssr)
    _, fitted, first, message, details, att = _clamp(targets, c, xi0, first, idx, lv_ssr)
    second = fit_second(fitted, c, first)
    be = BreakEvenParams(
        first.k1, c.k2, second.omega1, second.omega2, first.lambda1, first.lambda2, second.chi
    )
    params = be.to_book()
    ft, st = first.table, second.table
    table = pd.DataFrame(
        {
            "T": ft["T"],
            "atf": ft["atf"],
            "ssr_requested": requested,
            "ssr_target": ft["ssr_target"],
            "ssr_fitted": np.asarray(fitted.ssr_target, dtype=np.float64)[idx],
            "svc_target": ft["svc_target"],
            "svc_model": ft["svc_model"],
            "ssr_achieved": ft["ssr_achieved"],
            "ssr_achieved_lsv": ft["ssr_achieved_lsv"],
            "ssr_achieved_naked_vs_market": ft["ssr_achieved_naked_vs_market"],
            "ssr_naked_own": ft["ssr_naked_own"],
            "r_lv_market": ft["r_lv_market"],
            "skew_market": ft["skew_market"],
            "skew_naked": ft["skew_naked"],
            "skew_gap_rel": ft["skew_gap_rel"],
            "correl_target": st["correl_target"],
            "correl_implied": st["correl_implied"],
            "volvar_target": st["volvar_target"],
            "volvar_model": st["volvar_model"],
            "volvol_target": st["volvol_target"],
            "volvol_model": st["volvol_model"],
            "sensi_spot_lv": st["sensi_spot_lv"],
        }
    )
    proxy = None
    if proxy_surface is not None:
        proxy = leverage_proxy(
            params, proxy_surface, xi0, skew_ratio=first.max_skew_ratio, nu_limit=c.nu_flag
        )
    s3 = None
    recalibrated = False
    if stage3 is not None:
        s3 = stage3_validation(params, stage3, fitted, fit_table=table)
        recalibrated = s3.recalibrated
    return FitResult(
        params,
        be,
        fitted,
        xi0,
        c,
        first,
        second,
        table,
        RISK_REGIME,
        s3,
        time.perf_counter() - t0,
        recalibrated,
        pricing_date,
        tuple(notes),
        message,
        tuple(details),
        requested,
        att,
        proxy,
        lv_ssr,
    )


def naked_kernel(result: FitResult, forward_curve: Any) -> Any:
    """The fitted pure-SV kernel :class:`~volsto.models.bergomi.BergomiSV` on the fit's ``ξ₀``
    (for the numerical SSR / mixing checks of the naked model)."""
    from volsto.models.bergomi import BergomiSV

    return BergomiSV(result.params, result.xi0, forward_curve)


def fit_2f_marking(
    surface: Any,
    cfg: BreakEvenFitConfig | None = None,
    *,
    ssr_target: float | Mapping[float, float] | Callable[[float], float] = 1.0,
    anchor_power: float = 1.0,
    stage3: Stage3Inputs | None = None,
    h: float = SABR_CURVATURE_H,
    proxy: bool = True,
    lv_sim: Any | None = None,
) -> FitResult:
    """Marking mode on a surface: targets by :func:`~volsto.calibration.targets.
    marking_targets` (pillars of the config inside the surface's range), ``ξ₀`` the surface's
    variance-swap strip to the last fitted pillar; ``proxy`` computes the first-order
    leverage proxy (a Dupire surface, about a second); ``lv_sim`` is the
    :class:`~volsto.config.SimConfig` of the pure-Dupire SSR run when ``cfg.lv_ssr ==
    "numerical"``."""
    c = cfg or BreakEvenFitConfig()
    targets = marking_targets(
        surface, c.pillars, ssr_target=ssr_target, anchor_power=anchor_power, h=h
    )
    t_max = float(min(surface.max_maturity, max(targets.pillars)))
    xi0 = xi0_curve(surface, t_max)
    lv = None
    if c.lv_ssr == "numerical":
        if lv_sim is None:
            raise ValueError("cfg.lv_ssr='numerical' needs lv_sim (the Dupire SSR simulation)")
        idx, _ = _select_pillars(targets, c.pillars)
        lv = local_vol_ssr_numerical(surface, targets.pillars[idx].tolist(), sim=lv_sim)
    return fit_2f(
        targets,
        xi0,
        c,
        stage3=stage3,
        lv_ssr=lv,
        proxy_surface=surface if proxy else None,
    )


def _curve_from_vs(pillars: FloatArray, vs_vol: FloatArray) -> ForwardVarianceCurve:
    """Forward-variance curve through the pillar VS vols (total variance ``T σ̂_T²`` interpolated
    by :class:`ForwardVarianceCurve`, flat VS vol before the first and beyond the last pillar,
    made monotone by a tiny ramp)."""
    mats = np.concatenate(([0.5 * pillars[0]], pillars, [pillars[-1] + 1.0, pillars[-1] + 5.0]))
    vs = np.concatenate(([vs_vol[0]], vs_vol, [vs_vol[-1], vs_vol[-1]]))
    W = vs * vs * mats
    W = np.maximum.accumulate(W + 1e-12 * np.arange(W.size))
    return ForwardVarianceCurve(mats, W)


def fit_2f_historical(
    history: SurfaceHistory,
    cfg: BreakEvenFitConfig | None = None,
    *,
    end: pd.Timestamp | str | None = None,
    window_vol: int = WINDOW_VOL,
    window_ssr: int = WINDOW_SSR,
    stage3: Stage3Inputs | None = None,
) -> FitResult:
    """Historical mode on a :class:`~volsto.calibration.history.SurfaceHistory` at ``end``
    (default: the last date): targets by :func:`~volsto.calibration.targets.historical_targets`
    on the config's pillars present in the history, ``ξ₀`` from the pricing date's VS vols at
    all the history's pillars (:func:`_curve_from_vs`); the lsv measure's market skew on ``(0,
    T]`` is the power-law interpolation of the pillar skews (flagged)."""
    c = cfg or BreakEvenFitConfig()
    e = history.date_index(end)
    end_ts = history.dates[e]
    hp = np.asarray(history.pillars, dtype=np.float64)
    pillars = [float(T) for T in c.pillars if np.any(np.abs(hp - float(T)) <= _TOL_T)]
    if len(pillars) < 2:
        raise ValueError(f"fewer than two of the config pillars {c.pillars} are in the history")
    targets = historical_targets(
        history, pillars, end=end_ts, window_vol=window_vol, window_ssr=window_ssr
    )
    xi0 = _curve_from_vs(hp, np.asarray(history.vs_vol.to_numpy()[e], dtype=np.float64))
    return fit_2f(targets, xi0, c, stage3=stage3, pricing_date=end_ts)


# --------------------------------------------------------------------------------------------
# skew-weight trade-off study
# --------------------------------------------------------------------------------------------


def skew_weight_tradeoff(
    surface_or_targets: Any,
    cfg: BreakEvenFitConfig | None = None,
    xi0: ForwardVarianceCurve | None = None,
    *,
    weights: Sequence[float] = DEFAULT_TRADEOFF_WEIGHTS,
    ssr_target: float = 1.0,
    stage3: Stage3Inputs | Mapping[float, Stage3Inputs] | None = None,
    clamp: bool = False,
    proxy: bool = True,
    lv_ssr: LocalVolSSR | None = None,
    anchor_power: float = 1.0,
    h: float = SABR_CURVATURE_H,
) -> pd.DataFrame:
    """The owner's trade-off study: the fit at ``ssr_target`` for every skew weight in
    ``weights`` (skew-tight to skew-loose), one row per weight with ``skew_weight,
    ssr_measure, ssr_target, ssr_first_target, ssr_fitted`` (after the clamp when ``clamp``),
    ``message`` (the unclamped variant when ``clamp`` is False and the request lies outside the
    band), the first-order achieved SSR under both measures (mean and ``_<T>`` per pillar) and
    the naked kernel's own, ``first_order_valid`` (:attr:`FirstFit.first_order_valid`),
    ``mean_skew_gap`` (mean ``|S_naked/S_mkt − 1|``), ``max_skew_ratio``, ``k1, k1_at_bound, nu,
    nu_limit_binding, lambda1, lambda2, omega1, omega2, chi``, the attainable ``ssr_floor`` /
    ``ssr_ceiling`` (achieved-SSR units, the values of the message) with their scan targets and
    scan-edge flags, the leverage proxy ``mean_abs_L_minus_1_proxy`` and its validity (``proxy``,
    needs a surface), and with ``stage3`` (one input for every weight, or a mapping weight →
    input, e.g. cached study leverages via ``model=``) the actual ``mean_abs_L_minus_1`` and the
    numerical LSV SSR per pillar with standard errors; ``recalibrated`` and ``wall_seconds`` per
    row.  ``lv_ssr`` is forwarded to every fit (needed when ``cfg.lv_ssr == "numerical"``).
    Clamping is off by default so that the trade-off at the requested target is visible.  Checked
    by ``tests/test_fit_2f.py::test_tradeoff_sweep_monotone``."""
    base = cfg or BreakEvenFitConfig()
    if base.lv_ssr == "numerical" and lv_ssr is None:
        raise ValueError("cfg.lv_ssr='numerical' needs lv_ssr (local_vol_ssr_numerical)")
    surface = None
    if isinstance(surface_or_targets, TargetSet):
        targets = surface_or_targets.with_ssr_target(ssr_target)
        if xi0 is None:
            raise ValueError("a TargetSet needs its xi0")
    else:
        surface = surface_or_targets
        targets = marking_targets(
            surface, base.pillars, ssr_target=ssr_target, anchor_power=anchor_power, h=h
        )
        if xi0 is None:
            xi0 = xi0_curve(surface, float(min(surface.max_maturity, max(targets.pillars))))
    rows = []
    for w in weights:
        t0 = time.perf_counter()
        c = replace(base, skew_weight=float(w), clamp_to_attainable=bool(clamp))
        s3 = stage3.get(float(w)) if isinstance(stage3, Mapping) else stage3
        r = fit_2f(
            targets,
            xi0,
            c,
            stage3=s3,
            lv_ssr=lv_ssr,
            proxy_surface=surface if (proxy and surface is not None) else None,
        )
        tab = r.table
        att = r.attainable
        row: dict[str, Any] = {
            "skew_weight": float(w),
            "ssr_measure": r.config.ssr_measure,
            "ssr_target": float(ssr_target),
            "ssr_first_target": float(tab["ssr_target"].mean()),
            "ssr_fitted": float(tab["ssr_fitted"].mean()),
            "message": r.message,
            "ssr_achieved": r.ssr_achieved_mean,
            "ssr_achieved_lsv": r.ssr_achieved_lsv_mean,
            "ssr_achieved_naked_vs_market": r.ssr_achieved_naked_mean,
            "ssr_naked_own": float(tab["ssr_naked_own"].mean()),
            "first_order_valid": r.first.first_order_valid,
            "mean_skew_gap": r.mean_skew_gap,
            "max_skew_ratio": r.first.max_skew_ratio,
            "k1": r.params.k1,
            "k1_at_bound": r.first.k1_at_bound,
            "nu": r.params.nu,
            "nu_limit_binding": r.first.nu_limit_binding,
            "lambda1": r.breakeven.lambda1,
            "lambda2": r.breakeven.lambda2,
            "omega1": r.breakeven.omega1,
            "omega2": r.breakeven.omega2,
            "chi": r.breakeven.chi,
            "second_bound_flags": ";".join(r.second.bound_flags),
        }
        for T, a, b in zip(tab["T"], tab["ssr_achieved_lsv"], tab["ssr_achieved_naked_vs_market"]):
            row[f"ssr_lsv_{T:g}"] = float(a)
            row[f"ssr_naked_{T:g}"] = float(b)
        for T, g in zip(tab["T"], tab["skew_gap_rel"]):
            row[f"skew_gap_{T:g}"] = float(g)
        if att is not None:
            row.update(
                {
                    "ssr_floor": att.floor,
                    "ssr_floor_target": att.floor_target,
                    "ssr_floor_at_scan_edge": att.floor_at_scan_edge,
                    "ssr_ceiling": att.ceiling,
                    "ssr_ceiling_target": att.ceiling_target,
                    "ssr_ceiling_at_scan_edge": att.ceiling_at_scan_edge,
                }
            )
        if r.leverage_proxy is not None:
            row["mean_abs_L_minus_1_proxy"] = r.leverage_proxy.mean_abs_l_minus_1
            row["proxy_valid"] = r.leverage_proxy.valid
        row["mean_abs_L_minus_1"] = (
            float("nan") if r.stage3 is None else r.stage3.mean_abs_l_minus_1
        )
        if r.stage3 is not None:
            for _, x in r.stage3.ssr_table.iterrows():
                row[f"ssr_lsv_numerical_{x['T']:g}"] = float(x["ssr_model"])
                row[f"ssr_lsv_numerical_se_{x['T']:g}"] = float(x["ssr_model_se"])
        row["recalibrated"] = r.recalibrated
        row["wall_seconds"] = time.perf_counter() - t0
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# study specs (configs/studies/m7_skew_tradeoff): written by the study script, read by tests
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class TradeoffSpec:
    """One fitted parameter set of the trade-off study: the :class:`~volsto.config.
    CalibrationSpec` (section ``spec``, the key of the leverage cache) and the fit provenance
    (section ``fit``: ``config`` — a loadable :class:`BreakEvenFitConfig` —, ``ssr_target``,
    ``anchor_power``, ``breakeven``, the achieved SSRs, the message)."""

    path: Path
    spec: Any
    fit: dict[str, Any]

    @property
    def skew_weight(self) -> float:
        return float(self.fit["config"]["skew_weight"])

    @property
    def ssr_target(self) -> float:
        return float(self.fit["ssr_target"])

    @property
    def config(self) -> BreakEvenFitConfig:
        return from_mapping(BreakEvenFitConfig, self.fit["config"], path=str(self.path))

    @property
    def breakeven(self) -> BreakEvenParams:
        return BreakEvenParams(**{k: float(v) for k, v in self.fit["breakeven"].items()})


def tradeoff_spec_document(
    result: FitResult, base_spec: Any, *, n_particles: int, ssr_target: float
) -> dict[str, Any]:
    """The YAML document of :class:`TradeoffSpec` for ``result`` on ``base_spec`` (its market,
    surface, particle and simulation settings; the model replaced by the fitted parameters and
    the particle count by ``n_particles``)."""
    spec = replace(
        base_spec,
        model=result.params,
        particle=replace(base_spec.particle, n_particles=int(n_particles)),
    )
    tab = result.table
    return {
        "spec": to_mapping(spec),
        "fit": {
            "config": to_mapping(result.config),
            "ssr_target": float(ssr_target),
            "ssr_fitted": [float(x) for x in result.targets.ssr_target],
            "anchor_power": (
                None
                if not np.isfinite(result.targets.anchor_power)
                else float(result.targets.anchor_power)
            ),
            "breakeven": to_mapping(result.breakeven),
            "message": result.message,
            "pillars": [float(t) for t in tab["T"]],
            "ssr_achieved_lsv": [float(x) for x in tab["ssr_achieved_lsv"]],
            "ssr_achieved_naked_vs_market": [float(x) for x in tab["ssr_achieved_naked_vs_market"]],
            "skew_gap_rel": [float(x) for x in tab["skew_gap_rel"]],
            "leverage_proxy_mean_abs_L_minus_1": (
                None
                if result.leverage_proxy is None
                else float(result.leverage_proxy.mean_abs_l_minus_1)
            ),
        },
    }


def write_tradeoff_spec(
    result: FitResult, base_spec: Any, path: str | Path, *, n_particles: int, ssr_target: float
) -> Path:
    """Write :func:`tradeoff_spec_document` of ``result`` to ``path`` (directories created): the
    study script's hand-off to the tests (design note D7).  The ``spec`` section is the
    :class:`~volsto.config.CalibrationSpec` whose cache key the script calibrates and the tests
    read with ``allow_calibrate=False``; the ``fit`` section the provenance that lets a test
    re-run the fit (:class:`TradeoffSpec`).  Checked by
    ``tests/test_fit_2f.py::test_tradeoff_spec_round_trip``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    doc = tradeoff_spec_document(result, base_spec, n_particles=n_particles, ssr_target=ssr_target)
    p.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return p


def load_tradeoff_specs(directory: str | Path) -> list[TradeoffSpec]:
    """Every ``*.yaml`` of ``directory`` with ``spec`` and ``fit`` sections, sorted by
    decreasing skew weight (skew-tight first); an absent directory gives an empty list."""
    from volsto.config import CalibrationSpec, load_yaml

    d = Path(directory)
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.yaml")):
        raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping) or "spec" not in raw or "fit" not in raw:
            continue
        spec = load_yaml(p, CalibrationSpec, section="spec")
        out.append(TradeoffSpec(p, spec, dict(raw["fit"])))
    return sorted(out, key=lambda e: -e.skew_weight)


__all__ = [
    "CEILING_MESSAGE",
    "CEILING_MESSAGE_UNCLAMPED",
    "DEFAULT_K2",
    "DEFAULT_NU_FLAG",
    "DEFAULT_SATURATION_SLOPE",
    "DEFAULT_SKEW_WEIGHT",
    "DEFAULT_SSR_GRID",
    "DEFAULT_SSR_TOL",
    "DEFAULT_TRADEOFF_WEIGHTS",
    "FIRST_ORDER_SKEW_RATIO_LIMIT",
    "FLOOR_MESSAGE",
    "FLOOR_MESSAGE_UNCLAMPED",
    "K1_BASIN_JUMP_RATIO",
    "LV_SSR_KINDS",
    "MAX_FINITE_SE",
    "NU_PENALTY_WEIGHT",
    "PREFACTOR_KINDS",
    "PROXY_NU_TOL",
    "PROXY_SKEW_RATIO_LIMIT",
    "RISK_REGIME",
    "SSR_MEASURES",
    "SSR_MEASURE_CHOICES",
    "TERM_STRUCTURE_KINDS",
    "TRADEOFF_NOISE_SEED",
    "TS_SUBSTITUTION_POWER",
    "VOLVAR_TARGET_KINDS",
    "WEIGHT_KINDS",
    "AffineMaps",
    "AttainableSSR",
    "BreakEvenFitConfig",
    "FirstFit",
    "FitResult",
    "InnerSolution",
    "LeverageProxy",
    "LocalVolSSR",
    "MeasureMaps",
    "PillarQuad",
    "SecondFit",
    "Stage3Inputs",
    "Stage3Report",
    "TermStructureBank",
    "TradeoffSpec",
    "affine_maps",
    "affine_maps_from_kernels",
    "attainable_ssr",
    "first_order_regime_note",
    "fit_2f",
    "fit_2f_historical",
    "fit_2f_marking",
    "fit_first",
    "fit_second",
    "format_skew_weight",
    "k1_profile",
    "leverage_proxy",
    "load_tradeoff_specs",
    "local_vol_ssr_numerical",
    "mean_abs_leverage_deviation",
    "naked_kernel",
    "pillar_quad",
    "resolve_ssr_measure",
    "skew_weight_tradeoff",
    "stage3_validation",
    "term_structure_bank",
    "tradeoff_spec_document",
    "write_tradeoff_spec",
]
