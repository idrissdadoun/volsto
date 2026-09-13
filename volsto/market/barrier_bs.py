"""Black–Scholes closed forms for continuously monitored barrier options, touch options and
cash-or-nothing digitals (SPEC §6.5, M6 Part 1).

Source: Reiner & Rubinstein, "Breaking down the barriers", *Risk* 4(8), 1991, in the notation of
Haug, *The Complete Guide to Option Pricing Formulas*, 2nd ed., §4.17.1 (standard barrier
options), §4.19.1 (cash-or-nothing) and the reflection-principle hitting probabilities of a
drifted Brownian motion.  Flat continuously compounded rate ``r`` and dividend yield ``q``
(cost of carry ``b = r − q``), barrier ``H``, strike ``K``, rebate ``R``:

    μ = (b − σ²/2)/σ²,  λ = sqrt(μ² + 2r/σ²),  s = σ√T
    x1 = ln(S/K)/s + (1 + μ) s,   x2 = ln(S/H)/s + (1 + μ) s
    y1 = ln(H²/(S K))/s + (1 + μ) s,   y2 = ln(H/S)/s + (1 + μ) s,   z = ln(H/S)/s + λ s
    A = φ S e^{(b−r)T} N(φ x1) − φ K e^{−rT} N(φ x1 − φ s)
    B = φ S e^{(b−r)T} N(φ x2) − φ K e^{−rT} N(φ x2 − φ s)
    C = φ S e^{(b−r)T} (H/S)^{2(μ+1)} N(η y1) − φ K e^{−rT} (H/S)^{2μ} N(η y1 − η s)
    D = φ S e^{(b−r)T} (H/S)^{2(μ+1)} N(η y2) − φ K e^{−rT} (H/S)^{2μ} N(η y2 − η s)
    E = R e^{−rT} [N(η x2 − η s) − (H/S)^{2μ} N(η y2 − η s)]        (rebate at expiry if no hit)
    F = R [(H/S)^{μ+λ} N(η z) + (H/S)^{μ−λ} N(η z − 2 η λ s)]         (rebate paid at the hit)

with ``φ = ±1`` (call / put) and ``η = +1`` (down barrier) / ``−1`` (up barrier).  The eight
combinations are assembled in :func:`bs_barrier_price`; ``E/R`` is the discounted no-hit
probability used by the touch options and ``F/R`` the expected discount factor to the first
hitting time.  Every function broadcasts over its numeric arguments; ``cp``, ``direction`` and
``knock`` are scalar conventions.

Checked by ``tests/test_barrier.py::test_closed_form_identities``: the knock-out prices against a
quadrature of the reflection-principle density of ``ln S_T`` killed at the barrier (independent of
the ``A..D`` assembly, which makes in–out parity and the ``K = H`` continuity structural), the image
identities ``DIC(K ≥ H) = (H/S)^{2μ} C_BS(H²/S, K)`` and ``UIP(K ≤ H) = (H/S)^{2μ} P_BS(H²/S, K)``,
in–out parity against :func:`volsto.market.bs.bs_price`, barrier limits (``H → 0`` / ``H → ∞``
give the vanilla or zero, ``H = S`` gives the rebate or the vanilla), one-touch + no-touch =
discount factor, the hit probability against an independent reflection-principle transcription,
the ``F`` term against a quadrature of ``e^{−rt}`` times the first-passage density at ``r > 0``,
the digital as the limit of a call spread with an ``O(w²)`` bias, and
``test_closed_forms_vs_bridge_monte_carlo`` (Monte Carlo bridge prices within 3 stderr).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import ndtr

FloatArray = NDArray[np.float64]

DIRECTIONS = ("up", "down")
KNOCKS = ("in", "out")
REBATE_TIMINGS = ("hit", "maturity")


def _arrays(*xs: ArrayLike) -> tuple[FloatArray, ...]:
    return tuple(np.asarray(x, dtype=np.float64) for x in xs)


def _eta(direction: str) -> float:
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    return 1.0 if direction == "down" else -1.0


def _cp(cp: ArrayLike) -> FloatArray:
    phi = np.asarray(cp, dtype=np.float64)
    if not np.all(np.isin(phi, (1.0, -1.0))):
        raise ValueError("cp must be +1 (call) or -1 (put)")
    return phi


def _validate_positive(**named: FloatArray) -> None:
    for name, value in named.items():
        if np.any(~np.isfinite(value)) or np.any(value <= 0):
            raise ValueError(f"{name} must be positive and finite")


def _validate_finite(**named: FloatArray) -> None:
    for name, value in named.items():
        if np.any(~np.isfinite(value)):
            raise ValueError(f"{name} must be finite")


def _already_knocked(S: FloatArray, H: FloatArray, direction: str) -> NDArray[np.bool_]:
    """Spot at or beyond the barrier at inception (the formulas assume the alive side)."""
    return (S <= H) if direction == "down" else (S >= H)


def _no_hit_probability(
    S: FloatArray, H: FloatArray, T: FloatArray, vol: FloatArray, b: FloatArray, direction: str
) -> FloatArray:
    """``P(barrier never hit before T)`` for ``ln S`` with drift ``b − σ²/2``: the bracket of
    the Reiner–Rubinstein ``E`` term, ``N(η x2 − η s) − (H/S)^{2μ} N(η y2 − η s)``."""
    eta = _eta(direction)
    s = vol * np.sqrt(T)
    mu = (b - 0.5 * vol * vol) / (vol * vol)
    x2 = np.log(S / H) / s + (1.0 + mu) * s
    y2 = np.log(H / S) / s + (1.0 + mu) * s
    p = ndtr(eta * x2 - eta * s) - (H / S) ** (2.0 * mu) * ndtr(eta * y2 - eta * s)
    return np.asarray(np.where(_already_knocked(S, H, direction), 0.0, p), dtype=np.float64)


def bs_hit_probability(
    S: ArrayLike,
    H: ArrayLike,
    T: ArrayLike,
    vol: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    direction: str,
) -> FloatArray:
    """Risk-neutral probability that the spot touches ``H`` before ``T`` (``1`` when the spot
    starts at or beyond the barrier).  Reflection principle for ``ln S`` with drift
    ``r − q − σ²/2``; checked by ``tests/test_barrier.py::test_closed_form_identities``."""
    S_, H_, T_, v_, r_, q_ = _arrays(S, H, T, vol, r, q)
    _validate_positive(S=S_, H=H_, T=T_, vol=v_)
    _validate_finite(r=r_, q=q_)
    return np.asarray(
        1.0 - _no_hit_probability(S_, H_, T_, v_, r_ - q_, direction), dtype=np.float64
    )


def bs_hit_discount(
    S: ArrayLike,
    H: ArrayLike,
    T: ArrayLike,
    vol: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    direction: str,
) -> FloatArray:
    """``E[e^{−rτ} 1{τ ≤ T}]`` with ``τ`` the first hitting time of ``H``: the Reiner–Rubinstein
    ``F`` term per unit rebate (a one-touch paid at the hit); ``1`` when already at the barrier.
    Needs ``μ² + 2r/σ² ≥ 0`` (always true for ``r ≥ 0``; raises ``ValueError`` otherwise instead
    of returning NaN).  Checked by ``tests/test_barrier.py::test_closed_form_identities``
    (quadrature of ``e^{−rt}`` against the first-passage density at ``r > 0``)."""
    S_, H_, T_, v_, r_, q_ = _arrays(S, H, T, vol, r, q)
    _validate_positive(S=S_, H=H_, T=T_, vol=v_)
    _validate_finite(r=r_, q=q_)
    eta = _eta(direction)
    b = r_ - q_
    s = v_ * np.sqrt(T_)
    mu = (b - 0.5 * v_ * v_) / (v_ * v_)
    radicand = mu * mu + 2.0 * r_ / (v_ * v_)
    if np.any(radicand < 0.0):
        raise ValueError(
            "a rebate paid at the hit needs mu^2 + 2 r / sigma^2 >= 0 (Reiner-Rubinstein F term); "
            "the rate is too negative for this volatility"
        )
    lam = np.sqrt(radicand)
    z = np.log(H_ / S_) / s + lam * s
    ratio = H_ / S_
    f = ratio ** (mu + lam) * ndtr(eta * z) + ratio ** (mu - lam) * ndtr(
        eta * z - 2.0 * eta * lam * s
    )
    return np.asarray(np.where(_already_knocked(S_, H_, direction), 1.0, f), dtype=np.float64)


def bs_barrier_price(
    S: ArrayLike,
    K: ArrayLike,
    H: ArrayLike,
    T: ArrayLike,
    vol: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    cp: ArrayLike,
    direction: str,
    knock: str,
    rebate: ArrayLike = 0.0,
    rebate_timing: str | None = None,
) -> FloatArray:
    """Reiner–Rubinstein price of a continuously monitored single-barrier option.

    ``cp = ±1`` (call / put), ``direction`` in ``("up", "down")``, ``knock`` in ``("in",
    "out")``.  A knock-out's rebate is paid at the hit (``rebate_timing="hit"``, the ``F`` term)
    or at maturity (``"maturity"``: ``R e^{−rT} P(hit)``); a knock-in's rebate is paid at
    maturity when the option never knocked in (the ``E`` term; ``rebate_timing`` must be
    ``"maturity"`` or ``None``).  The timing is required whenever ``rebate != 0`` (no silent
    convention).  A spot already at or beyond the barrier prices as the knocked contract: the
    rebate (out) or the vanilla (in).  Checked by ``tests/test_barrier.py``
    (``test_closed_form_identities``, ``test_closed_forms_vs_bridge_monte_carlo``).
    """
    S_, K_, H_, T_, v_, r_, q_, R_ = _arrays(S, K, H, T, vol, r, q, rebate)
    _validate_positive(S=S_, K=K_, H=H_, T=T_, vol=v_)
    _validate_finite(r=r_, q=q_, rebate=R_)
    if knock not in KNOCKS:
        raise ValueError(f"knock must be one of {KNOCKS}, got {knock!r}")
    eta = _eta(direction)
    phi = _cp(cp)
    if np.any(R_ != 0.0) and rebate_timing is None:
        raise ValueError("rebate_timing ('hit' or 'maturity') is required when rebate != 0")
    if rebate_timing is not None and rebate_timing not in REBATE_TIMINGS:
        raise ValueError(f"rebate_timing must be one of {REBATE_TIMINGS}, got {rebate_timing!r}")
    if knock == "in" and rebate_timing == "hit":
        raise ValueError("a knock-in rebate is paid at maturity (rebate_timing='maturity')")

    b = r_ - q_
    s = v_ * np.sqrt(T_)
    mu = (b - 0.5 * v_ * v_) / (v_ * v_)
    ratio = H_ / S_
    pow_hi = ratio ** (2.0 * (mu + 1.0))
    pow_lo = ratio ** (2.0 * mu)
    fwd = S_ * np.exp((b - r_) * T_)
    dfk = K_ * np.exp(-r_ * T_)
    x1 = np.log(S_ / K_) / s + (1.0 + mu) * s
    x2 = np.log(S_ / H_) / s + (1.0 + mu) * s
    y1 = np.log(H_ * H_ / (S_ * K_)) / s + (1.0 + mu) * s
    y2 = np.log(H_ / S_) / s + (1.0 + mu) * s
    A = phi * fwd * ndtr(phi * x1) - phi * dfk * ndtr(phi * x1 - phi * s)
    B = phi * fwd * ndtr(phi * x2) - phi * dfk * ndtr(phi * x2 - phi * s)
    C = phi * fwd * pow_hi * ndtr(eta * y1) - phi * dfk * pow_lo * ndtr(eta * y1 - eta * s)
    D = phi * fwd * pow_hi * ndtr(eta * y2) - phi * dfk * pow_lo * ndtr(eta * y2 - eta * s)

    k_above = K_ >= H_
    call = phi > 0
    down = eta > 0
    if knock == "in":
        # down-in call: K>H: C | K<H: A-B+D ; up-in call: K>H: A | K<H: B-C+D
        # down-in put:  K>H: B-C+D | K<H: A ; up-in put: K>H: A-B+D | K<H: C
        opt = np.where(
            call,
            np.where(down, np.where(k_above, C, A - B + D), np.where(k_above, A, B - C + D)),
            np.where(down, np.where(k_above, B - C + D, A), np.where(k_above, A - B + D, C)),
        )
        reb = R_ * np.exp(-r_ * T_) * _no_hit_probability(S_, H_, T_, v_, b, direction)
    else:
        # down-out call: K>H: A-C | K<H: B-D ; up-out call: K>H: 0 | K<H: A-B+C-D
        # down-out put:  K>H: A-B+C-D | K<H: 0 ; up-out put: K>H: B-D | K<H: A-C
        zero = np.zeros_like(A)
        opt = np.where(
            call,
            np.where(down, np.where(k_above, A - C, B - D), np.where(k_above, zero, A - B + C - D)),
            np.where(down, np.where(k_above, A - B + C - D, zero), np.where(k_above, B - D, A - C)),
        )
        if rebate_timing == "hit":
            reb = R_ * bs_hit_discount(S_, H_, T_, v_, r_, q_, direction)
        else:
            reb = R_ * np.exp(-r_ * T_) * (1.0 - _no_hit_probability(S_, H_, T_, v_, b, direction))
    knocked = _already_knocked(S_, H_, direction)
    # A is the vanilla: a knocked-in contract is the vanilla, a knocked-out one the rebate
    value = np.where(knocked, A if knock == "in" else reb, opt + reb)
    return np.asarray(value, dtype=np.float64)


def bs_one_touch_price(
    S: ArrayLike,
    H: ArrayLike,
    T: ArrayLike,
    vol: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    direction: str,
    payout: ArrayLike = 1.0,
) -> FloatArray:
    """One-touch paid at expiry: ``payout e^{−rT} P(hit)`` (``R e^{−rT} − E`` in the notation of
    the module docstring).  Checked by ``test_closed_form_identities`` (one-touch + no-touch =
    discount factor) and ``test_closed_forms_vs_bridge_monte_carlo``."""
    S_, H_, T_, v_, r_, q_, P_ = _arrays(S, H, T, vol, r, q, payout)
    _validate_finite(payout=P_)
    p_hit = bs_hit_probability(S_, H_, T_, v_, r_, q_, direction)
    return np.asarray(P_ * np.exp(-r_ * T_) * p_hit, dtype=np.float64)


def bs_no_touch_price(
    S: ArrayLike,
    H: ArrayLike,
    T: ArrayLike,
    vol: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    direction: str,
    payout: ArrayLike = 1.0,
) -> FloatArray:
    """No-touch paid at expiry: ``payout e^{−rT} P(no hit)`` (the ``E`` term per unit rebate)."""
    S_, H_, T_, v_, r_, q_, P_ = _arrays(S, H, T, vol, r, q, payout)
    _validate_finite(payout=P_)
    p_hit = bs_hit_probability(S_, H_, T_, v_, r_, q_, direction)
    return np.asarray(P_ * np.exp(-r_ * T_) * (1.0 - p_hit), dtype=np.float64)


def bs_digital_price(
    S: ArrayLike,
    K: ArrayLike,
    T: ArrayLike,
    vol: ArrayLike,
    r: ArrayLike,
    q: ArrayLike,
    cp: ArrayLike,
    payout: ArrayLike = 1.0,
) -> FloatArray:
    """Cash-or-nothing digital ``payout e^{−rT} N(cp d2)``, ``d2 = (ln(S/K) + (b − σ²/2)T)/(σ√T)``
    (Haug §4.19.1).  Checked by ``test_closed_form_identities`` (limit of the call spread)."""
    S_, K_, T_, v_, r_, q_, P_ = _arrays(S, K, T, vol, r, q, payout)
    _validate_positive(S=S_, K=K_, T=T_, vol=v_)
    _validate_finite(r=r_, q=q_, payout=P_)
    phi = _cp(cp)
    s = v_ * np.sqrt(T_)
    d2 = (np.log(S_ / K_) + (r_ - q_ - 0.5 * v_ * v_) * T_) / s
    return np.asarray(P_ * np.exp(-r_ * T_) * ndtr(phi * d2), dtype=np.float64)
