"""The joint step kernel of the local correlation model (SPEC §8.7, M12 parts LC3–LC4).

:class:`~volsto.multi.model.MultiAssetModel` steps the assets one after another; that is
impossible here, because ``λ`` at ``t_j`` needs every asset at ``t_j``.
:func:`lc_diffuse_block` loops over paths (``prange``), then steps, then assets.  For path
``p`` and step ``j``:

1. **Basket state** (read before any asset moves in the step): ``B = Σ_i w_i·exp(ls_i −
   s_i(t_j))`` and ``k_B = ln B − ln F_B(t_j)``, with the shifts ``s_i`` and ``ln F_B`` of the
   :class:`~volsto.multi.lc_model.BasketSpec`.
2. **λ**: ``λ = min(max(interp_uniform(λ(t_j, ·), k_B), 0), λ_max)``, frozen over the step;
   ``c₁ = √(1 − λ)``, ``c₂ = √λ``.
3. **Mixed normals**: ``x_i = (L_low ε)_i`` (the equicorrelation prefix or
   :func:`~volsto.multi.draws.lower_row_dot`: the summation order of ``CorrelatedDraws``),
   ``y_i = (L_high η)_i`` (:func:`~volsto.multi.lc_draws.high_row_dot`), ``z_i = c₁·x_i +
   c₂·y_i``.
4. **Asset steps**: the library's shared :func:`~volsto.models.localvol.spot_step` on each
   asset's own local variance, then the accumulators and the record columns — the statements
   of :func:`~volsto.models.localvol.diffuse_block`, in its order.

**Why λ is frozen over the step** (derived, SPEC §8.7): ``c₁`` and ``c₂`` are known at
``t_j`` with ``c₁² + c₂² = 1`` and each mixed term has variance 1, so conditionally on the past
``z_i ~ N(0, 1)``: the ``z_{i,j}`` of an asset are i.i.d. standard normals and its discrete path
has exactly the law of the single-asset scheme on the same grid — every single-name vanilla is
the single-asset local-vol price at every ``Δt``.  Re-evaluating ``λ`` at a predictor or
supporting value (which depends on ``z``) would break this, so no scheme does.  Within the
step the mixing is constant, each asset's diffusion depends on its own state only, and the
per-asset Platen weak order-2 step on its own ``z_i`` is the multi-dimensional weak order-2
step for a constant correlation; what is neglected is the dependence of the mixing on the
state inside the step, a first-order weak error in the correlation (measured by the Δt check
of the calibration).

**Identities** (``tests/test_local_correlation.py``, bit for bit).  With ``λ ≡ 0``: ``c₁ =
1.0``, ``c₂ = 0.0`` and ``z_i = x_i`` exactly, and step 4 is ``diffuse_block``'s arithmetic —
the kernel reproduces ``MultiAssetModel`` on ``CorrelatedDraws(R_low)`` (I1).  Each asset's
path is ``LocalVol.simulate_chunk`` on its recorded ``z_i`` (I3).  The particle calibration
calls this kernel one step at a time (I2).

:func:`lc_ab` computes, per particle, the two terms of the basket variance ``v_B = a + λ·b``
that the calibration regresses on the basket log-moneyness.

Pure functions on arrays (SPEC §11); the Python wrappers validate.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from volsto._numba import njit, prange
from volsto.models.localvol import interp_uniform, spot_step
from volsto.multi.draws import lower_row_dot
from volsto.multi.lc_draws import high_row_dot

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@njit(parallel=True, cache=True)
def lc_diffuse_block(
    log_spot: FloatArray,
    int_var: FloatArray,
    sum_sq: FloatArray,
    lam_int: FloatArray,
    eps: FloatArray,
    eta: FloatArray,
    t_nodes: FloatArray,
    ln_f: FloatArray,
    drifts: FloatArray,
    step_record: IntArray,
    k0: float,
    dk: float,
    var_a: FloatArray,
    var_b: FloatArray,
    var_rec: FloatArray,
    mode: int,
    pc_eta: float,
    weights: FloatArray,
    shifts: FloatArray,
    ln_fb: FloatArray,
    lam_rows: FloatArray,
    lam_k0: float,
    lam_dk: float,
    lam_max: float,
    low_equi: bool,
    d_low: FloatArray,
    ell_low: FloatArray,
    l_low: FloatArray,
    l_high: FloatArray,
    out_log_spot: FloatArray,
    out_var: FloatArray,
    out_int_var: FloatArray,
    out_sum_sq: FloatArray,
    out_lam_int: FloatArray,
    out_k_basket: FloatArray,
    record_z: bool,
    out_z: FloatArray,
    z_step0: int,
) -> None:  # pragma: no cover - numba
    """Advance all paths and all assets over a block of steps (module docstring).

    State, updated in place: ``log_spot``, ``int_var``, ``sum_sq`` of shape ``(n_paths, n)`` and
    ``lam_int`` ``(n_paths,)`` (the cumulative ``Σ λ·Δt``).  Draws: ``eps`` ``(n_paths, n_block,
    n)``, ``eta`` ``(n_paths, n_block, r)``.  Per asset ``i``: ``ln_f[i]`` (``n_block + 1``
    nodes), ``drifts[i]``, and the variance rows ``var_a[i, j]``, ``var_b[i, j]``,
    ``var_rec[i, j]`` on the shared uniform grid ``(k0, dk)``, as in ``diffuse_block``.
    Basket: ``weights`` ``(n,)``, ``shifts`` ``(n, n_block + 1)``, ``ln_fb`` ``(n_block + 1,)``.
    ``lam_rows[j]`` is the start-of-step row of ``λ`` on its own uniform grid ``(lam_k0,
    lam_dk)``.  Mixing: ``low_equi`` selects the equicorrelation prefix (``d_low``,
    ``ell_low``) or the general factor ``l_low``; ``l_high`` is ``(n, r)``.  Outputs at record
    columns (``step_record[j] ≥ 0``): the assets' ``(n, n_paths, n_cols)`` arrays, the
    cumulative ``λ`` integral and the basket log-moneyness at ``t_{j+1}``, ``(n_paths,
    n_cols)``.  With ``record_z`` the mixed normals go to ``out_z[p, z_step0 + j, i]``.
    """
    n_paths = log_spot.shape[0]
    n = log_spot.shape[1]
    nb = eps.shape[1]
    for p in prange(n_paths):
        for j in range(nb):
            dt = t_nodes[j + 1] - t_nodes[j]
            # 1. the basket state at the step start
            level = 0.0
            for i in range(n):
                level += weights[i] * np.exp(log_spot[p, i] - shifts[i, j])
            kb = np.log(level) - ln_fb[j]
            # 2. lambda, frozen over the step
            lam = interp_uniform(lam_k0, lam_dk, lam_rows[j], kb)
            lam = min(max(lam, 0.0), lam_max)
            c1 = np.sqrt(1.0 - lam)
            c2 = np.sqrt(lam)
            er = eps[p, j]
            hr = eta[p, j]
            col = step_record[j]
            pref = 0.0
            for i in range(n):
                # 3. the mixed normal of asset i
                if low_equi:
                    x = pref + d_low[i] * er[i]
                    pref = pref + ell_low[i] * er[i]
                else:
                    x = lower_row_dot(l_low, er, i)
                zj = c1 * x + c2 * high_row_dot(l_high, hr, i)
                # 4. the asset's step: diffuse_block's statements
                ls = log_spot[p, i]
                iv = int_var[p, i]
                sq = sum_sq[p, i]
                dls, v, _b0, _bbx = spot_step(
                    ls,
                    dt,
                    drifts[i, j],
                    zj,
                    k0,
                    dk,
                    var_a[i, j],
                    var_b[i, j],
                    1.0,
                    1.0,
                    ln_f[i, j],
                    ln_f[i, j + 1],
                    mode,
                    pc_eta,
                )
                ls += dls
                iv += v * dt
                sq += dls * dls
                if col >= 0:
                    out_log_spot[i, p, col] = ls
                    out_var[i, p, col] = interp_uniform(k0, dk, var_rec[i, j], ls - ln_f[i, j + 1])
                    out_int_var[i, p, col] = iv
                    out_sum_sq[i, p, col] = sq
                log_spot[p, i] = ls
                int_var[p, i] = iv
                sum_sq[p, i] = sq
                if record_z:
                    out_z[p, z_step0 + j, i] = zj
            lam_int[p] += lam * dt
            if col >= 0:
                end = 0.0
                for i in range(n):
                    end += weights[i] * np.exp(log_spot[p, i] - shifts[i, j + 1])
                out_lam_int[p, col] = lam_int[p]
                out_k_basket[p, col] = np.log(end) - ln_fb[j + 1]


@njit(parallel=True, cache=True)
def lc_ab(
    log_spot: FloatArray,
    ln_f: FloatArray,
    weights: FloatArray,
    shifts: FloatArray,
    ln_fb: float,
    k0: float,
    dk: float,
    var_rows: FloatArray,
    low_equi: bool,
    rho_low: float,
    r_low: FloatArray,
    high_ones: bool,
    l_high: FloatArray,
    out_a: FloatArray,
    out_b: FloatArray,
    out_k: FloatArray,
) -> None:  # pragma: no cover - numba
    """Per particle, the two terms of the basket variance ``v_B = a + λ·b`` and the basket
    log-moneyness (SPEC §8.7; :class:`~volsto.multi.family.CorrelationFamily`):

        k_i = ls_i − ln F_i,   ω_i = w_i·exp(ls_i − s_i)/B,   u_i = ω_i·√var_rows[i](k_i)
        a = uᵀ R_low u          ((1 − ρ)·Σ u_i² + ρ·(Σ u_i)² for the equicorrelation)
        b = uᵀ R_high u − a     (uᵀ R_high u = (Σ u_i)² for 11ᵀ, Σ_m (Σ_i L_high[i, m]·u_i)² else)

    ``log_spot`` is ``(n_particles, n)``; ``ln_f``, ``weights`` and ``shifts`` are ``(n,)`` at
    the slice time; ``var_rows`` is ``(n, n_k)``, each name's variance on the shared grid
    ``(k0, dk)`` (its average over the step the row will govern).  Sums run over ``i`` (then
    ``j``) ascending.  Checked against the matrix form by ``tests/test_local_correlation.py::
    test_lc_ab_matches_the_matrix_form``."""
    n_paths = log_spot.shape[0]
    n = log_spot.shape[1]
    r = l_high.shape[1]
    simple = low_equi and high_ones
    for p in prange(n_paths):
        level = 0.0
        for i in range(n):
            level += weights[i] * np.exp(log_spot[p, i] - shifts[i])
        out_k[p] = np.log(level) - ln_fb
        s1 = 0.0
        s2 = 0.0
        if simple:
            for i in range(n):
                ls = log_spot[p, i]
                omega = weights[i] * np.exp(ls - shifts[i]) / level
                ui = omega * np.sqrt(interp_uniform(k0, dk, var_rows[i], ls - ln_f[i]))
                s1 += ui
                s2 += ui * ui
            a = (1.0 - rho_low) * s2 + rho_low * (s1 * s1)
            out_a[p] = a
            out_b[p] = s1 * s1 - a
        else:
            u = np.empty(n)
            for i in range(n):
                ls = log_spot[p, i]
                omega = weights[i] * np.exp(ls - shifts[i]) / level
                u[i] = omega * np.sqrt(interp_uniform(k0, dk, var_rows[i], ls - ln_f[i]))
                s1 += u[i]
                s2 += u[i] * u[i]
            if low_equi:
                a = (1.0 - rho_low) * s2 + rho_low * (s1 * s1)
            else:
                a = 0.0
                for i in range(n):
                    for j in range(n):
                        a += u[i] * r_low[i, j] * u[j]
            if high_ones:
                high = s1 * s1
            else:
                high = 0.0
                for m in range(r):
                    proj = 0.0
                    for i in range(n):
                        proj += l_high[i, m] * u[i]
                    high += proj * proj
            out_a[p] = a
            out_b[p] = high - a


__all__ = ["lc_ab", "lc_diffuse_block"]
