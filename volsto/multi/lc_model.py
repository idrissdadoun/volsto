"""The local correlation model (SPEC §8.7, M12 part LC3): Dupire local-vol names whose Brownian
correlation matrix ``ρ(λ) = (1 − λ)·R_low + λ·R_high`` depends on the basket through a scalar
``λ(t, k)``.

**The basket state** (:class:`BasketSpec`).  ``B_t = Σ_i w_i·S_i(t)·e^{−s_i(t)}`` with forward
``F_B(t)`` and log-moneyness ``k = ln(B_t / F_B(t))``; the two modes differ in the deterministic
shifts only (the names and the products are the same):

* ``"performance"`` (default, the dispersion study's convention): ``s_i(t) = ln F_i(t)`` and
  ``F_B ≡ 1``, so ``B_t = Σ w_i X_i`` with ``X_i = S_i/F_i(t)`` a martingale; then ``dB/B = Σ ω_i
  σ_i dW_i`` with the current weights ``ω_i = w_i X_i / B`` and the basket's instantaneous
  variance is ``v_B = Σ_ij ω_i ω_j σ_i σ_j ρ_ij`` exactly (derived).
* ``"carry"``: ``s_i ≡ ln S_i(0)`` and ``F_B(t) = Σ w_i F_i(t)/S_i(0)``: ``B`` is the price
  basket normalised to 1 (the index without its divisor); ``ln(B/F_B)`` then has the stochastic
  drift ``δ_t = Σ (ω_i − θ_i)·μ_i`` (``θ_i = w_i f_i/F_B`` the forward weights, ``μ_i`` the
  names' carry), which the calibration ignores and measures.

The shifts are those of the curves the specification is built with — the **base state's**: a
spot bump of a name's model moves the basket's starting point (``λ`` held in absolute basket
level, the "sticky strike" regime of the risk layer), it does not move the shifts.

**The model** (:class:`LocalCorrelationModel`) exposes what
:class:`~volsto.multi.mc.MultiAssetMonteCarlo` uses (the :class:`~volsto.multi.model.MultiModel`
protocol).  All names share one uniform ``k`` grid (validated: the joint kernel holds every
table on one grid, with O(1) lookups); the pricing grid contains every ``λ`` slice
(:meth:`LocalCorrelationModel.required_times`), so ``λ`` is piecewise constant per step in
pricing exactly as in the calibration; a product beyond the last slice is refused — ``λ`` is
never extrapolated silently.

Checked by ``tests/test_local_correlation.py`` (identities I1, I3, I4, I5, I9).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Final

import numpy as np
from numpy.typing import ArrayLike, NDArray

from volsto.config import SchemeConfig
from volsto.engine.grid import TimeGrid
from volsto.engine.paths import PathSet
from volsto.market.curves import ForwardCurve
from volsto.models.localvol import LocalVol, step_variance_tables
from volsto.models.lsv import scheme_mode
from volsto.multi.family import CorrelationFamily
from volsto.multi.lc_draws import LocalCorrelationDraws
from volsto.multi.lc_function import LocalCorrelationFunction
from volsto.multi.lc_kernel import lc_diffuse_block
from volsto.multi.paths import MultiPathSet

FloatArray = NDArray[np.float64]

BASKET_MODES: Final[tuple[str, ...]] = ("performance", "carry")
#: The weights of the basket state sum to 1 within this (the basket starts at ``k = 0``).
WEIGHT_SUM_TOL: Final[float] = 1e-12
#: Default memory budget of one block of draws and tables, in MiB: the default of
#: ``SimConfig.chunk_memory_mb`` (``simulate_chunk`` receives the scheme, not the ``SimConfig``).
BLOCK_MEMORY_MB: Final[int] = 512
#: A grid ending this far beyond the last ``λ`` slice is still the horizon (rounding of a time).
HORIZON_TOL: Final[float] = 1e-9


class BasketSpec:
    """The basket state the correlation depends on (module docstring): the weights ``w_i ≥ 0``
    (summing to 1), the mode, and the names' forward curves of the base state."""

    def __init__(
        self, weights: ArrayLike, mode: str, forward_curves: Sequence[ForwardCurve]
    ) -> None:
        w = np.asarray(weights, dtype=np.float64).ravel()
        if w.size == 0 or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("basket weights must be finite and non-negative")
        if abs(float(w.sum()) - 1.0) > WEIGHT_SUM_TOL:
            raise ValueError(f"basket weights must sum to 1 (sum = {float(w.sum())!r})")
        if mode not in BASKET_MODES:
            raise ValueError(f"basket mode must be one of {BASKET_MODES}, got {mode!r}")
        if len(forward_curves) != w.size:
            raise ValueError("one forward curve per basket weight")
        self.weights = np.ascontiguousarray(w)
        self.mode = mode
        self.forward_curves = tuple(forward_curves)

    @property
    def n(self) -> int:
        return int(self.weights.size)

    def log_forwards(self, times: ArrayLike) -> FloatArray:
        """``ln F_i(t)`` of the base curves: ``(n, len(times))``."""
        t = np.atleast_1d(np.asarray(times, dtype=np.float64))
        return np.ascontiguousarray(
            np.stack(
                [np.asarray(fc.log_forward(t), dtype=np.float64) for fc in self.forward_curves]
            )
        )

    def shifts(self, times: ArrayLike) -> FloatArray:
        """``s_i(t)``: ``(n, len(times))`` — ``ln F_i(t)`` in performance mode, ``ln S_i(0)`` in
        carry mode."""
        t = np.atleast_1d(np.asarray(times, dtype=np.float64))
        if self.mode == "performance":
            return self.log_forwards(t)
        spots = np.log(np.array([fc.spot for fc in self.forward_curves]))
        return np.ascontiguousarray(np.repeat(spots[:, None], t.size, axis=1))

    def log_basket_forward(self, times: ArrayLike) -> FloatArray:
        """``ln F_B(t)``: zero in performance mode, ``ln Σ w_i F_i(t)/S_i(0)`` in carry mode."""
        t = np.atleast_1d(np.asarray(times, dtype=np.float64))
        if self.mode == "performance":
            return np.zeros(t.size)
        ratios = np.exp(self.log_forwards(t) - self.shifts(t))
        return np.asarray(np.log(self.weights @ ratios), dtype=np.float64)

    def log_moneyness(self, log_spots: ArrayLike, t: float) -> FloatArray:
        """``k = ln(B/F_B(t))`` for log-spots of shape ``(…, n)`` at time ``t``."""
        ls = np.asarray(log_spots, dtype=np.float64)
        level = np.exp(ls - self.shifts([t])[:, 0]) @ self.weights
        return np.asarray(np.log(level) - self.log_basket_forward([t])[0], dtype=np.float64)

    def with_mode(self, mode: str) -> BasketSpec:
        return BasketSpec(self.weights, mode, self.forward_curves)

    def __repr__(self) -> str:
        return f"BasketSpec(n={self.n}, mode={self.mode!r})"


class LocalCorrelationModel:
    """``models[i]`` (a :class:`~volsto.models.localvol.LocalVol`) drives asset ``i``; the
    Brownians have the correlation ``family.correlation(λ)`` with ``λ = lam(t, k)`` read on the
    basket state of ``basket`` (module docstring)."""

    def __init__(
        self,
        models: Sequence[LocalVol],
        family: CorrelationFamily,
        lam: LocalCorrelationFunction,
        basket: BasketSpec,
        names: Sequence[str] | None = None,
        *,
        block_memory_mb: int = BLOCK_MEMORY_MB,
    ) -> None:
        if not models:
            raise ValueError("at least one asset")
        for i, m in enumerate(models):
            if not isinstance(m, LocalVol):
                raise ValueError(
                    f"asset {i}: the local correlation model drives Dupire local-vol names "
                    f"(a Black-Scholes name is a LocalVol with a flat table); got "
                    f"{type(m).__name__}"
                )
        first = models[0].local_vol
        for i, m in enumerate(models):
            lv = m.local_vol
            if (lv.k_grid.size, lv.k0, lv.dk) != (first.k_grid.size, first.k0, first.dk):
                raise ValueError(
                    f"asset {i}: every name must share one uniform k grid (one LocalVolConfig); "
                    f"got k0={lv.k0}, dk={lv.dk}, n_k={lv.k_grid.size} against k0={first.k0}, "
                    f"dk={first.dk}, n_k={first.k_grid.size}"
                )
        n = len(models)
        if family.n != n or basket.n != n:
            raise ValueError("models, family and basket must agree on the number of assets")
        if block_memory_mb <= 0:
            raise ValueError("block_memory_mb must be positive")
        self.models = list(models)
        self.family = family
        self.lam = lam
        self.basket = basket
        self.block_memory_mb = int(block_memory_mb)
        self.names = (
            tuple(str(x) for x in names)
            if names is not None
            else tuple(f"asset{i}" for i in range(n))
        )
        if len(self.names) != n:
            raise ValueError("one name per asset")

    # -- the MultiModel protocol ---------------------------------------------------------------

    @property
    def n_assets(self) -> int:
        return len(self.models)

    @property
    def spots(self) -> FloatArray:
        return np.array([m.spot for m in self.models])

    def forwards(self, T: float) -> FloatArray:
        return np.array([float(m.forward_curve.forward(T)) for m in self.models])

    def required_times(self) -> FloatArray:
        """The ``λ`` slices (and the names' own required times, none for local vol): the pricing
        grid contains them, so the frozen rule is the calibration's."""
        parts = [np.asarray(m.required_times(), dtype=np.float64) for m in self.models]
        return np.unique(np.concatenate([self.lam.times, *parts]))

    def draws_for(
        self, grid: TimeGrid, seed: int, n_paths: int, antithetic: bool = True
    ) -> LocalCorrelationDraws:
        return LocalCorrelationDraws(seed, n_paths, grid.n_steps, self.family, antithetic)

    def _block_steps(self, n_paths: int, step_block: int) -> int:
        """``step_block`` lowered until one block (``ε``, ``η`` and the three variance tables)
        fits the memory budget; no numerical effect (every quantity is per path and step)."""
        n = self.n_assets
        n_k = int(self.models[0].local_vol.k_grid.size)
        per_step = 8 * (n_paths * (n + self.family.rank_high) + 3 * n * n_k)
        return int(max(1, min(step_block, (self.block_memory_mb * 2**20) // per_step)))

    def simulate_chunk(
        self,
        grid: TimeGrid,
        draws: LocalCorrelationDraws,
        p0: int,
        p1: int,
        scheme: SchemeConfig,
        *,
        step_block: int = 64,
        record_normals: bool = False,
    ) -> MultiPathSet:
        """Paths ``[p0, p1)`` on ``grid``: one :class:`~volsto.engine.paths.PathSet` per asset
        and ``aux`` with ``"lam_int"`` (the cumulative ``Σ λ·Δt``) and ``"k_basket"`` (the
        basket log-moneyness) at the record columns.  ``record_normals`` (a test hook) adds
        ``aux["normals"]``, the mixed normals ``(n_paths, n_steps, n_assets)``."""
        n = self.n_assets
        if draws.n_assets != n or draws.family.rank_high != self.family.rank_high:
            raise ValueError("draws and model disagree on the number of assets or of factors")
        if grid.horizon > self.lam.horizon + HORIZON_TOL:
            raise ValueError(
                f"pricing to {grid.horizon:.6g}y is beyond the calibration horizon of the local "
                f"correlation ({self.lam.horizon:.6g}y): refused, lambda is never extrapolated"
            )
        n_paths = p1 - p0
        times = grid.times
        n_cols = grid.n_records
        fam = self.family
        out_ls = np.empty((n, n_paths, n_cols))
        out_var = np.empty((n, n_paths, n_cols))
        out_iv = np.empty((n, n_paths, n_cols))
        out_sq = np.empty((n, n_paths, n_cols))
        out_lam = np.empty((n_paths, n_cols))
        out_kb = np.empty((n_paths, n_cols))
        ls = np.empty((n_paths, n))
        ln_f = np.empty((n, times.size))
        for i, m in enumerate(self.models):
            state = m.initial_state(n_paths)
            out_ls[i, :, 0] = state.log_spot
            out_var[i, :, 0] = state.variance
            ls[:, i] = state.log_spot
            ln_f[i] = np.asarray(m.forward_curve.log_forward(times), dtype=np.float64)
        out_iv[:, :, 0] = 0.0
        out_sq[:, :, 0] = 0.0
        drifts = np.ascontiguousarray(np.diff(ln_f, axis=1))
        shifts = self.basket.shifts(times)
        ln_fb = self.basket.log_basket_forward(times)
        out_lam[:, 0] = 0.0
        out_kb[:, 0] = self.basket.log_moneyness(ls, 0.0)
        iv = np.zeros((n_paths, n))
        sq = np.zeros((n_paths, n))
        lam_int = np.zeros(n_paths)
        mode = scheme_mode(scheme)
        lv0 = self.models[0].local_vol
        out_z = np.empty((n_paths, grid.n_steps, n)) if record_normals else np.empty((1, 1, 1))
        nb_max = self._block_steps(n_paths, step_block)
        for s0 in range(0, grid.n_steps, nb_max):
            s1 = min(s0 + nb_max, grid.n_steps)
            t_nodes = times[s0 : s1 + 1]
            rec = grid.step_record[s0:s1]
            any_rec = bool(np.any(rec >= 0))
            tables = [step_variance_tables(m.local_vol, t_nodes, scheme) for m in self.models]
            var_a = np.ascontiguousarray(np.stack([t[0] for t in tables]))
            var_b = np.ascontiguousarray(np.stack([t[1] for t in tables]))
            var_rec = (
                np.ascontiguousarray(
                    np.stack([m.local_vol.var_at_times(t_nodes[1:]) for m in self.models])
                )
                if any_rec
                else var_a
            )
            lc_diffuse_block(
                ls,
                iv,
                sq,
                lam_int,
                draws.eps_block(s0, s1, p0, p1),
                draws.eta_block(s0, s1, p0, p1),
                t_nodes,
                np.ascontiguousarray(ln_f[:, s0 : s1 + 1]),
                np.ascontiguousarray(drifts[:, s0:s1]),
                rec,
                lv0.k0,
                lv0.dk,
                var_a,
                var_b,
                var_rec,
                mode,
                scheme.pc_eta,
                self.basket.weights,
                np.ascontiguousarray(shifts[:, s0 : s1 + 1]),
                np.ascontiguousarray(ln_fb[s0 : s1 + 1]),
                self.lam.step_rows(t_nodes),
                self.lam.k0,
                self.lam.dk,
                fam.lambda_max,
                fam.low_equi,
                fam.d_low,
                fam.ell_low,
                fam.l_low,
                fam.l_high,
                out_ls,
                out_var,
                out_iv,
                out_sq,
                out_lam,
                out_kb,
                record_normals,
                out_z,
                s0,
            )
        no_factors = np.empty((n_paths, n_cols, 0))
        assets = tuple(
            PathSet(
                times=grid.record_times,
                log_spot=out_ls[i],
                variance=out_var[i],
                factors=no_factors,
                int_var=out_iv[i],
                sum_sq=out_sq[i],
            )
            for i in range(n)
        )
        aux: dict[str, FloatArray] = {"lam_int": out_lam, "k_basket": out_kb}
        if record_normals:
            aux["normals"] = out_z
        return MultiPathSet(assets, self.names, aux)

    # -- variants ------------------------------------------------------------------------------

    def with_lambda(self, lam: LocalCorrelationFunction) -> LocalCorrelationModel:
        """The same names, family and basket state with another ``λ`` (the constant-correlation
        companion, a parametric ``λ``, a bump): common random numbers are preserved."""
        return LocalCorrelationModel(
            self.models,
            self.family,
            lam,
            self.basket,
            self.names,
            block_memory_mb=self.block_memory_mb,
        )

    def with_models(self, models: Sequence[LocalVol]) -> LocalCorrelationModel:
        """Other single-name models (a spot or a vol bump of a name) with ``λ`` and the basket
        state held."""
        return LocalCorrelationModel(
            models,
            self.family,
            self.lam,
            self.basket,
            self.names,
            block_memory_mb=self.block_memory_mb,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "n_assets": self.n_assets,
            "names": list(self.names),
            "family": self.family.describe(),
            "mode": self.basket.mode,
            "lambda_horizon": self.lam.horizon,
            "lambda_slices": self.lam.n_slices,
        }

    def __repr__(self) -> str:
        return (
            f"LocalCorrelationModel(n_assets={self.n_assets}, {self.family!r}, {self.lam!r}, "
            f"{self.basket!r})"
        )


__all__ = ["BASKET_MODES", "BasketSpec", "LocalCorrelationModel"]
