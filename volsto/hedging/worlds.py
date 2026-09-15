"""World models of the M8b hedging studies (SPEC §8.1 "Skew-shock world"; study C: "same book as
B, pricing = marking fit, world = pricing but with a skew shock: +1, +2, +3 rota applied linearly
over 5 business days mid-life, then held").

**Construction.**  The world is the base LSV model (the pricing model: base surface, base P1 set,
base leverage ``L_base``) before ``t0``; over the shock window ``[t0, t_end]``,
``t_end = t0 + days / days_per_year``, its leverage blends **linearly in time** from ``L_base``
to the leverage of the rotated surface, and it holds the rotated leverage afterwards:

    L_world(t, k) = (1 − w(t)) L_base(t, k) + w(t) L̃_rot(t, k),
    w(t) = clip((t − t0) / (t_end − t0), 0, 1),

per leverage slice ``t_i``, with slices **added at ``t0`` and ``t_end``** when absent (both
leverages interpolated there first, :meth:`~volsto.models.leverage.LeverageFunction.rows`), so
the blend is exact at the shock boundaries and, the slices being required grid times of the LSV
model, the simulation grid contains them.  ``L_rot`` is the leverage calibrated on the **rotated
surface** — the base surface under the M7 rotation perturbation of ``rota`` rotas
(:func:`~volsto.risk.shadow_rotation.rotation_perturbation`: one rota steepens the 90/110 skew by
:func:`~volsto.risk.shadow_rotation.rota_skew_vol_points` — 0.5600 vp at 6M with the ``k_cap``
saturation; ``2/sqrt(T) ln(110/90)`` = 0.5676 is the small-``k`` limit —, ATM vols unchanged)
with the **base parameters**
— read from the leverage cache through :class:`~volsto.risk.engine.LSVBuilder` (a study script
may calibrate on a miss; tests never do).

**The shock is applied through the leverage; the kernel is the base kernel throughout** (the
2F parameters and ``ξ₀`` of the base surface).  The rotation preserves the ATM vols exactly
(``δσ(0, T) = 0``) but **not** the forward-variance curve: the variance-swap strip integrates the
steepened put wing, so on the reference surface +1 rota raises ``ξ₀`` by 5.7% at 1M, 3.6% at 6M,
3.1% at 1Y and 1.9% at 5Y (in variance; +3 rota: 19.7% / 11.4% / 9.6% / 5.7%), measured in
``tests/test_hedging_worlds.py``.  Using ``L_rot`` on the base kernel as it is would therefore
leave the post-shock world short of the rotated surface's variance level by that much.  Because
``ξ₀`` enters the lognormal Bergomi kernel only as the deterministic multiplier ``g(t) =
ξ₀(t) exp(−½ω²χ(t, t))`` of ``ξ_t^t`` (the factor dynamics do not depend on ``ξ₀``), the
rotated model ``LSV(kernel_rot, L_rot)`` is in law **exactly** ``LSV(kernel_base, L̃_rot)`` with
``L̃_rot(t, k) = L_rot(t, k) · sqrt(ξ₀_rot(t) / ξ₀_base(t))``; the world blends towards
``L̃_rot`` (``xi0_rescale=True``, the default: the post-shock world is then the rotated LSV model
up to the within-step treatment of the slowly varying ratio — the frozen-leverage rule holds
``L̃`` at the step start while the second-order SV step carries ``½ ∂_t ln g``; the ratio's
log-slope on the reference surface at +1 rota peaks near 0.8/yr around 1M–2M, i.e. ``½ ∂_t
ln(ratio) δt ≈ 1.6·10⁻³`` of the variance per daily step there, and is 0.014/yr (3·10⁻⁵ per
step) beyond 6M; measured on a synthetic pair of kernels in the test).  ``xi0_rescale=False``
keeps the raw ``L_rot`` (the leverage-only
approximation, for reports that want to show the difference).  Beyond the leverage horizon the
last slice — hence the ratio at the horizon — is held.

**Mid-shock the world is an interpolation, not a calibrated model**: between ``t0`` and ``t_end``
neither surface is repriced; only the boundaries are exact.  This is the intended reading of
"applied linearly over 5 business days".

**Pure local-vol analogue** (:func:`lv_shock_world`): the same construction on the Dupire
local vol of the two surfaces (:class:`~volsto.risk.engine.LVBuilder`, no leverage, no kernel,
hence no ``ξ₀`` question), blending the **local variance** ``σ_loc²`` linearly in time on the
shared Dupire grid with slices added at the boundaries.  The variance is blended because
:class:`~volsto.market.dupire.LocalVolSurface` interpolates the variance linearly between slices
(as :class:`~volsto.models.leverage.LeverageFunction` interpolates ``L``, which is why the LSV
world blends ``L``): the world's variance is then exactly linear across the window whatever the
slice density, whereas a vol blend would hold only at the slices (measured on a coarse synthetic
grid without an interior slice: the midpoint variance was the variance midpoint either way).

Checked by ``tests/test_hedging_worlds.py`` (blend weights and boundaries on synthetic
leverages, the blended LSV simulates and its vanilla price sits between the two endpoints, the
``ξ₀``-rescale identity on two kernels, the rotated surface's skew / ATM / ``ξ₀`` moves, the
cache-backed miss raised cleanly without calibrating).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.calibration.cache import LeverageCache
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import ImpliedSurface, perturbed_surface
from volsto.models.base import Model
from volsto.models.leverage import LeverageFunction
from volsto.models.localvol import LocalVol
from volsto.models.lsv import LSV
from volsto.risk.engine import LSVBuilder, LVBuilder, RiskState, surface_of
from volsto.risk.shadow_rotation import ROTA_K_CAP, ROTA_T_MIN, rotation_perturbation

FloatArray = NDArray[np.float64]

#: the owner's shock window: "applied linearly over 5 business days mid-life, then held"
SHOCK_DAYS = 5
#: business days per year: the hedger's daily schedule (``FREQUENCIES["daily"] = 1/252``)
DAYS_PER_YEAR = 252
#: a boundary within this of an existing slice reuses the slice (no near-duplicate grid time;
#: the hedger's date tolerance)
_SLICE_TOL = 1e-9
_LN_90_110 = (math.log(0.9), math.log(1.1))


# --------------------------------------------------------------------------------------------
# blending
# --------------------------------------------------------------------------------------------


def shock_weights(times: ArrayLike, t0: float, t_end: float) -> FloatArray:
    """``w(t) = clip((t − t0) / (t_end − t0), 0, 1)``: 0 before the shock, 1 after."""
    if not t_end > t0 >= 0.0:
        raise ValueError(f"need 0 <= t0 < t_end, got t0={t0}, t_end={t_end}")
    t = np.asarray(times, dtype=np.float64)
    return np.asarray(np.clip((t - t0) / (t_end - t0), 0.0, 1.0), dtype=np.float64)


def blend_times(times: ArrayLike, t0: float, t_end: float) -> tuple[FloatArray, int]:
    """``times`` with ``t0`` and ``t_end`` added when no slice lies within :data:`_SLICE_TOL`;
    returns the sorted grid and the number of slices added."""
    t = np.asarray(times, dtype=np.float64).ravel()
    added = [x for x in (t0, t_end) if np.min(np.abs(t - x)) > _SLICE_TOL]
    return np.asarray(np.sort(np.concatenate([t, added])), dtype=np.float64), len(added)


def _blend_rows(
    times: FloatArray,
    rows_base: Callable[[FloatArray], FloatArray],
    rows_target: Callable[[FloatArray], FloatArray],
    t0: float,
    t_end: float,
) -> tuple[FloatArray, FloatArray, int]:
    grid, n_added = blend_times(times, t0, t_end)
    w = shock_weights(grid, t0, t_end)
    a = rows_base(grid)
    b = rows_target(grid)
    v = (1.0 - w)[:, None] * a + w[:, None] * b
    return grid, np.asarray(v, dtype=np.float64), n_added


def blend_leverage(
    base: LeverageFunction, target: LeverageFunction, *, t0: float, t_end: float, **metadata: Any
) -> LeverageFunction:
    """``(1 − w) L_base + w L_target`` slice by slice (module docstring), on the shared grid, with
    slices added at ``t0`` and ``t_end``; both leverages must share ``(times, k_grid)``."""
    if not (
        base.times.shape == target.times.shape
        and np.allclose(base.times, target.times, rtol=0.0, atol=_SLICE_TOL)
    ):
        raise ValueError(
            "blend_leverage needs the same slice times on both leverages "
            f"({base.n_slices} vs {target.n_slices} slices, horizons {base.horizon:g} / "
            f"{target.horizon:g}): the two calibrations must share the particle schedule"
        )
    if not (
        base.k_grid.shape == target.k_grid.shape
        and np.allclose(base.k_grid, target.k_grid, rtol=0.0, atol=1e-12)
    ):
        raise ValueError(
            "blend_leverage needs the same k grid on both leverages "
            f"({base.k_grid.size} points over [{base.k_grid[0]:g}, {base.k_grid[-1]:g}] vs "
            f"{target.k_grid.size} over [{target.k_grid[0]:g}, {target.k_grid[-1]:g}])"
        )
    grid, values, n_added = _blend_rows(base.times, base.rows, target.rows, t0, t_end)
    meta = {
        **base.metadata,
        "blend": {"t0": float(t0), "t_end": float(t_end), "slices_added": n_added},
        **metadata,
    }
    return LeverageFunction(grid, base.k_grid, values, base.forward_curve, meta)


def blend_local_vol(
    base: LocalVolSurface, target: LocalVolSurface, *, t0: float, t_end: float
) -> LocalVolSurface:
    """``σ_loc² = (1 − w) σ_base² + w σ_target²`` (the local **variance**, the quantity the surface
    interpolates in time; module docstring) on the shared Dupire grid with slices added at ``t0``
    and ``t_end``."""
    if base.t_grid.shape != target.t_grid.shape or not np.allclose(
        base.t_grid, target.t_grid, rtol=0.0, atol=_SLICE_TOL
    ):
        raise ValueError("blend_local_vol needs the same t grid on both local-vol surfaces")
    if base.k_grid.shape != target.k_grid.shape or not np.allclose(
        base.k_grid, target.k_grid, rtol=0.0, atol=1e-12
    ):
        raise ValueError("blend_local_vol needs the same k grid on both local-vol surfaces")

    grid, var, _ = _blend_rows(base.t_grid, base.var_at_times, target.var_at_times, t0, t_end)
    return LocalVolSurface(grid, base.k_grid, var, base.forward_curve, None)


# --------------------------------------------------------------------------------------------
# the rotated surface
# --------------------------------------------------------------------------------------------


def shocked_surface(
    base_surface: ImpliedSurface,
    rota: float,
    *,
    t_min: float = ROTA_T_MIN,
    k_cap: float = ROTA_K_CAP,
) -> ImpliedSurface:
    """``base_surface`` rotated by ``rota`` rotas (arbitrage-checked), for reports."""
    return perturbed_surface(rotation_perturbation(rota, t_min=t_min, k_cap=k_cap), base_surface)


def skew_90_110(surface: ImpliedSurface, T: float) -> float:
    """``σ(ln 0.9, T) − σ(ln 1.1, T)`` in vol points."""
    lo, hi = _LN_90_110
    return float(100.0 * (surface.implied_vol_k(lo, T) - surface.implied_vol_k(hi, T)))


def shock_state(
    base: RiskState, rota: float, *, t_min: float = ROTA_T_MIN, k_cap: float = ROTA_K_CAP
) -> RiskState:
    """The rotated state (base parameters, the rotation added to the perturbation layer)."""
    return base.with_perturbation(
        rotation_perturbation(rota, t_min=t_min, k_cap=k_cap), label=f"rota{rota:+g}"
    )


def _window(t0: float, days: int, days_per_year: int) -> float:
    if days <= 0 or days_per_year <= 0:
        raise ValueError("days and days_per_year must be positive")
    if t0 < 0.0:
        raise ValueError("t0 must be non-negative")
    return float(t0) + float(days) / float(days_per_year)


# --------------------------------------------------------------------------------------------
# worlds
# --------------------------------------------------------------------------------------------


def skew_shock_world(
    base: RiskState,
    cache: LeverageCache,
    *,
    rota: float,
    t0: float,
    days: int = SHOCK_DAYS,
    days_per_year: int = DAYS_PER_YEAR,
    allow_calibrate: bool = True,
    t_min: float = ROTA_T_MIN,
    k_cap: float = ROTA_K_CAP,
    xi0_rescale: bool = True,
) -> tuple[Model, dict[str, Any]]:
    """The skew-shock LSV world of study C (module docstring): the base model before ``t0``,
    its leverage blended linearly to the rotated surface's over ``[t0, t0 + days/days_per_year]``
    and held afterwards, on the **base kernel** (``ξ₀`` of the base surface; the rotated
    surface's ``ξ₀`` absorbed into the leverage when ``xi0_rescale``).

    Both leverages come from ``cache`` through :class:`~volsto.risk.engine.LSVBuilder`; with
    ``allow_calibrate=False`` a missing entry raises
    :class:`~volsto.calibration.cache.CacheMissError` (tests never calibrate).  Returns the
    model and a metadata mapping: ``rota``, ``t0``, ``t_end``, the base and rotated cache keys,
    ``calibrated`` (whether any leverage was calibrated in this call) and ``n_calibrations``,
    ``leverage_rel_change_mean`` (mean ``|L_rot / L_base − 1|`` over the shared grid),
    ``xi0_rel_change_max`` (max ``|ξ₀_rot / ξ₀_base − 1|`` at the slice times), the 6M 90/110
    skew of the two surfaces in vol points and the slices added."""
    t_end = _window(t0, days, days_per_year)
    rot = shock_state(base, rota, t_min=t_min, k_cap=k_cap)
    base_cached = cache.has(base.spec)
    rot_cached = cache.has(rot.spec)
    builder = LSVBuilder(cache, base, allow_calibrate=allow_calibrate)
    base_model = builder.base_model
    rot_model = builder.build(rot, "recalibrate")
    assert isinstance(base_model, LSV) and isinstance(rot_model, LSV)
    l_base, l_rot = base_model.leverage, rot_model.leverage
    if l_base.values.shape != l_rot.values.shape:
        raise ValueError(
            "the base and rotated leverages live on different grids "
            f"({l_base!r} vs {l_rot!r}); the rotation preserves the ATM vols, so the particle "
            "schedule or the leverage grid settings of the two cache entries differ"
        )
    t = l_base.times
    xi_ratio = np.asarray(rot_model.kernel.xi0.xi0(t) / base_model.kernel.xi0.xi0(t))
    target = (
        l_rot.with_values(l_rot.values * np.sqrt(xi_ratio)[:, None], xi0_rescaled=True)
        if xi0_rescale
        else l_rot
    )
    lev = blend_leverage(
        l_base,
        target,
        t0=t0,
        t_end=t_end,
        world="skew_shock",
        rota=float(rota),
        rotated_key=rot.key,
        xi0_rescale=bool(xi0_rescale),
    )
    world = base_model.bump(leverage=lev)
    base_surface, rot_surface = surface_of(base), surface_of(rot)
    meta: dict[str, Any] = {
        "world": "skew_shock_lsv",
        "rota": float(rota),
        "t0": float(t0),
        "t_end": t_end,
        "days": int(days),
        "days_per_year": int(days_per_year),
        "base_key": base.key,
        "rotated_key": rot.key,
        "base_cached_before": base_cached,
        "rotated_cached_before": rot_cached,
        "calibrated": bool(builder.n_cache_misses > 0),
        "n_calibrations": int(builder.n_cache_misses),
        "leverage_rel_change_mean": float(np.mean(np.abs(l_rot.values / l_base.values - 1.0))),
        "xi0_rel_change_max": float(np.max(np.abs(xi_ratio - 1.0))),
        "xi0_rescale": bool(xi0_rescale),
        "slices_added": int(lev.metadata["blend"]["slices_added"]),
        "n_slices": lev.n_slices,
        "skew_90_110_6m_base_vp": skew_90_110(base_surface, 0.5),
        "skew_90_110_6m_rotated_vp": skew_90_110(rot_surface, 0.5),
    }
    return world, meta


def lv_shock_world(
    base: RiskState,
    *,
    rota: float,
    t0: float,
    days: int = SHOCK_DAYS,
    days_per_year: int = DAYS_PER_YEAR,
    t_min: float = ROTA_T_MIN,
    k_cap: float = ROTA_K_CAP,
) -> tuple[Model, dict[str, Any]]:
    """The pure local-vol skew-shock world (module docstring): the Dupire local vol of the base
    surface before ``t0``, blended linearly in the local **variance** ``σ_loc²`` (module
    docstring) to the rotated surface's over the window and held afterwards; two Dupire builds,
    no leverage, no calibration."""
    t_end = _window(t0, days, days_per_year)
    rot = shock_state(base, rota, t_min=t_min, k_cap=k_cap)
    builder = LVBuilder(base)
    base_model = builder.base_model
    rot_model = builder.build(rot, "recalibrate")
    assert isinstance(base_model, LocalVol) and isinstance(rot_model, LocalVol)
    lv = blend_local_vol(base_model.local_vol, rot_model.local_vol, t0=t0, t_end=t_end)
    world = base_model.bump(local_vol=lv)
    vb, vr = base_model.local_vol.local_var, rot_model.local_vol.local_var
    base_surface, rot_surface = surface_of(base), surface_of(rot)
    meta: dict[str, Any] = {
        "world": "skew_shock_lv",
        "rota": float(rota),
        "t0": float(t0),
        "t_end": t_end,
        "days": int(days),
        "days_per_year": int(days_per_year),
        "base_key": base.key,
        "rotated_key": rot.key,
        "calibrated": False,
        "n_calibrations": 0,
        "local_vol_rel_change_mean": float(np.mean(np.abs(np.sqrt(vr / vb) - 1.0))),
        "n_slices": int(lv.t_grid.size),
        "slices_added": int(lv.t_grid.size - base_model.local_vol.t_grid.size),
        "skew_90_110_6m_base_vp": skew_90_110(base_surface, 0.5),
        "skew_90_110_6m_rotated_vp": skew_90_110(rot_surface, 0.5),
    }
    return world, meta


__all__ = [
    "DAYS_PER_YEAR",
    "SHOCK_DAYS",
    "blend_leverage",
    "blend_local_vol",
    "blend_times",
    "lv_shock_world",
    "shock_state",
    "shock_weights",
    "shocked_surface",
    "skew_90_110",
    "skew_shock_world",
]
