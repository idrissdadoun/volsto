"""On-demand what-if of the forward smile and skew (owner's decision of 2026-09-30): a marking
state (by default the desk's SPX 2022-12-30 mark) with edited stochastic-volatility parameters,
both leverages calibrated at a reduced particle count (:data:`WHATIF_PARTICLES`, 100 000 — the
owner's choice) and the forward smiles of every window priced on **common random numbers**; the
change of each strike's implied vol, of the forward ATM level, skew and curvature is reported
with its **paired** standard error (the per-path payoff differences), which is far smaller than
either level's error — a parameter move is measured, not drowned in the levels' noise.

This is the one viewer module that calibrates (the read API, :mod:`volsto.viewers.api`, still
never does): each leverage goes through :meth:`~volsto.calibration.cache.LeverageCache.
get_or_calibrate` in a what-if cache (``<outputs>/whatif/cache``), so a parameter set computed
once is a cache hit afterwards, and every run is stored as JSON under ``<outputs>/whatif/runs``
keyed by its request.  Measured on the desk mark (2026-09-30, horizon 2y): a calibration takes
8.8 s at 5·10⁴ particles and 26 s at 2·10⁵; the 1y→1y forward smile 0.8 s at 2·10⁴ paths and
2.0 s at 5·10⁴, its 95/105 skew level ±1.7 vp per unit log-strike at 5·10⁴ paths.

**Method.**  Per window ``T1 → T2`` the forward-start options on the forward-moneyness strikes
(plus the ATM forward ``F(T2)/F(T1)``) of every window are priced on one path set per model
(:meth:`~volsto.engine.mc.MonteCarlo.price_many`, keeping the payoffs), inverted to Black vols on
the ratio (:func:`~volsto.analytics.forward_smile.forward_smile_from_prices`); the change's
price error is the antithetic-paired standard error of the per-path differences, converted to
vol through the base smile's Black vega.  The forward ATM level, skew ``d vol / d ln K`` and
curvature ``d² vol / d ln K²`` are the weighted quadratic fit in log forward moneyness of the
viewer's forward-skew page (weights ``1/stderr²``); the change's fit is the same regression on
the vol changes with their paired errors (a linear functional: its slope is the skew change).
The quadratic's errors assume independent strikes: conservative for the positively correlated
errors of one path set.

Checked by ``tests/test_viewers_whatif.py`` (Black–Scholes pairs, no calibration).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volsto.analytics.forward_smile import forward_ratio, forward_smile_from_prices
from volsto.config import SimConfig
from volsto.engine.mc import MonteCarlo, summarize
from volsto.market.bs import black_vega
from volsto.models.base import Model
from volsto.products.forward_start import ForwardStartOption

#: the owner's particle count for the what-if leverages (2026-09-30)
WHATIF_PARTICLES = 100_000
WHATIF_PATHS = 50_000
WHATIF_SEED = 2024
#: the stochastic-volatility parameters the page edits (BergomiParams fields)
PARAMS: tuple[str, ...] = ("nu", "theta", "k1", "k2", "rho12", "rho_SX1", "rho_SX2")
DEFAULT_WINDOWS: tuple[tuple[float, float], ...] = ((0.25, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 3.0))
DEFAULT_STRIKES: tuple[float, ...] = (0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20)
#: the desk's mark the page starts from (repository-relative)
DESK_MARK = "configs/studies/m7_p1_marking/spx_ssr1_eps0.1.yaml"


@dataclass(frozen=True)
class WhatIfRequest:
    """A what-if: the base marking spec (repository-relative path), the parameter changes, the
    windows, strikes and budgets."""

    base_spec: str = DESK_MARK
    changes: tuple[tuple[str, float], ...] = ()
    windows: tuple[tuple[float, float], ...] = DEFAULT_WINDOWS
    strikes: tuple[float, ...] = DEFAULT_STRIKES
    n_particles: int = WHATIF_PARTICLES
    n_paths: int = WHATIF_PATHS
    seed: int = WHATIF_SEED

    def __post_init__(self) -> None:
        unknown = [k for k, _ in self.changes if k not in PARAMS]
        if unknown:
            raise ValueError(f"unknown parameters {unknown}; editable: {list(PARAMS)}")
        if not self.windows or any(not (0.0 <= a < b) for a, b in self.windows):
            raise ValueError("windows must be non-empty pairs 0 <= T1 < T2")
        if len(self.strikes) < 3 or any(k <= 0 for k in self.strikes):
            raise ValueError("need at least three positive strikes (forward moneyness)")
        if self.n_particles < 1000 or self.n_paths < 2 or self.n_paths % 2:
            raise ValueError("n_particles >= 1000 and an even n_paths >= 2 (antithetic pairs)")

    @property
    def horizon(self) -> float:
        return float(max(b for _, b in self.windows))

    def key(self) -> str:
        payload = json.dumps(dataclasses.asdict(self), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class WhatIfResult:
    """The smiles (one row per window and strike), the fits (base / moved / change per window)
    and the run's record."""

    request: WhatIfRequest
    smiles: pd.DataFrame
    fits: pd.DataFrame
    info: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {
                "request": dataclasses.asdict(self.request),
                "smiles": self.smiles.to_dict(orient="list"),
                "fits": self.fits.to_dict(orient="list"),
                "info": self.info,
            },
            default=float,
        )

    @classmethod
    def from_json(cls, text: str) -> WhatIfResult:
        d = json.loads(text)
        r = d["request"]
        req = WhatIfRequest(
            base_spec=r["base_spec"],
            changes=tuple((str(k), float(v)) for k, v in r["changes"]),
            windows=tuple((float(a), float(b)) for a, b in r["windows"]),
            strikes=tuple(float(k) for k in r["strikes"]),
            n_particles=int(r["n_particles"]),
            n_paths=int(r["n_paths"]),
            seed=int(r["seed"]),
        )
        return cls(req, pd.DataFrame(d["smiles"]), pd.DataFrame(d["fits"]), dict(d["info"]))


def _quadratic(
    x: np.ndarray, y: np.ndarray, se: np.ndarray
) -> tuple[float, float, float, float, float, float, float] | None:
    """Weighted quadratic ``y = a + b x + c x²`` (weights ``1/se²``): ``a, se_a, b, se_b, 2c,
    se_2c, chi²/dof`` or ``None`` with fewer than three usable points."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(se) & (se > 0)
    x, y, se = x[ok], y[ok], se[ok]
    if x.size < 3 or np.unique(x).size < 3:
        return None
    design = np.column_stack([np.ones(x.size), x, x * x])
    w = 1.0 / (se * se)
    normal = design.T @ (w[:, None] * design)
    beta = np.linalg.solve(normal, design.T @ (w * y))
    cov = np.linalg.inv(normal)
    resid = y - design @ beta
    dof = x.size - 3
    chi2 = float(np.sum(w * resid * resid)) / dof if dof > 0 else math.nan
    return (
        float(beta[0]),
        float(math.sqrt(cov[0, 0])),
        float(beta[1]),
        float(math.sqrt(cov[1, 1])),
        float(2.0 * beta[2]),
        float(2.0 * math.sqrt(cov[2, 2])),
        chi2,
    )


def _products(
    model: Model, windows: Sequence[tuple[float, float]], strikes: Sequence[float]
) -> tuple[list[ForwardStartOption], list[tuple[float, float, float, np.ndarray, np.ndarray]]]:
    """The forward-start options of every window and, per window, ``(t1, t2, F_R, strikes, cps)``
    (the ATM forward strike added)."""
    discount = model.forward_curve.rate_curve
    products: list[ForwardStartOption] = []
    layout = []
    for t1, t2 in windows:
        f_r = forward_ratio(model, t1, t2)
        ks = np.unique(np.append(np.asarray(strikes, dtype=np.float64), f_r))
        cps = np.where(ks >= f_r, 1, -1).astype(np.int64)
        layout.append((float(t1), float(t2), float(f_r), ks, cps))
        products += [
            ForwardStartOption(t1, t2, float(k), int(c), discount) for k, c in zip(ks, cps)
        ]
    return products, layout


def compare(
    base: Model,
    moved: Model,
    windows: Sequence[tuple[float, float]],
    strikes: Sequence[float],
    sim: SimConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Forward smiles of ``base`` and ``moved`` on common random numbers and their change with
    the paired standard error (module docstring): ``(smiles, fits)``."""
    products, layout = _products(base, windows, strikes)
    res_b = MonteCarlo(sim).price_many(products, base, keep_payoffs=True)
    res_m = MonteCarlo(sim).price_many(products, moved, keep_payoffs=True)
    df_curve = base.forward_curve.rate_curve
    rows: list[dict[str, Any]] = []
    fits: list[dict[str, Any]] = []
    i = 0
    for t1, t2, f_r, ks, cps in layout:
        n = ks.size
        rb, rm = res_b[i : i + n], res_m[i : i + n]
        i += n
        df = float(df_curve.df(t2))
        sb = forward_smile_from_prices(t1, t2, f_r, ks, cps, rb, df, sim.n_paths)
        sm = forward_smile_from_prices(t1, t2, f_r, ks, cps, rm, df, sim.n_paths)
        tau = t2 - t1
        vega = black_vega(f_r, ks, tau, np.where(np.isfinite(sb.vols), sb.vols, 0.0), df)
        d_vol = np.asarray(sm.vols - sb.vols, dtype=np.float64)
        d_se = np.full(n, np.nan)
        for j in range(n):
            pb, pm = rb[j].payoffs, rm[j].payoffs
            if pb is None or pm is None or not (vega[j] > 0):
                continue
            d_se[j] = summarize(np.asarray(pm) - np.asarray(pb), sim.antithetic).stderr / vega[j]
        x = np.log(ks / f_r)
        for j in range(n):
            rows.append(
                {
                    "window": f"{t1:g}y→{t2:g}y",
                    "t1": t1,
                    "t2": t2,
                    "strike": float(ks[j]),
                    "log_moneyness": float(x[j]),
                    "base_vol": float(sb.vols[j]),
                    "base_vol_stderr": float(sb.vol_stderr[j]),
                    "moved_vol": float(sm.vols[j]),
                    "moved_vol_stderr": float(sm.vol_stderr[j]),
                    "change": float(d_vol[j]),
                    "change_stderr": float(d_se[j]),
                }
            )
        for which, y, se in (
            ("base", sb.vols, sb.vol_stderr),
            ("moved", sm.vols, sm.vol_stderr),
            ("change", d_vol, d_se),
        ):
            q = _quadratic(x, np.asarray(y, dtype=float), np.asarray(se, dtype=float))
            fits.append(
                {
                    "window": f"{t1:g}y→{t2:g}y",
                    "t1": t1,
                    "t2": t2,
                    "which": which,
                    "atm": math.nan if q is None else q[0],
                    "atm_stderr": math.nan if q is None else q[1],
                    "skew": math.nan if q is None else q[2],
                    "skew_stderr": math.nan if q is None else q[3],
                    "curvature": math.nan if q is None else q[4],
                    "curvature_stderr": math.nan if q is None else q[5],
                    "chi2_dof": math.nan if q is None else q[6],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(fits)


def base_state(request: WhatIfRequest, repo_root: Path) -> Any:
    """The base marking state at the request's particle count and horizon (at least the window's
    last date)."""
    from volsto.calibration.fit_2f import load_fit_spec
    from volsto.risk.engine import RiskState

    fs = load_fit_spec(repo_root / request.base_spec)
    horizon = max(request.horizon, 0.25)
    spec = dataclasses.replace(
        fs.spec,
        particle=dataclasses.replace(
            fs.spec.particle, n_particles=int(request.n_particles), horizon=float(horizon)
        ),
    )
    return RiskState(spec, None, "what-if base")


def base_params(request: WhatIfRequest, repo_root: Path) -> dict[str, float]:
    st = base_state(request, repo_root)
    return {k: float(getattr(st.spec.model, k)) for k in PARAMS}


def run_whatif(
    request: WhatIfRequest,
    outputs_root: Path,
    repo_root: Path,
    *,
    allow_calibrate: bool = True,
    calibrate: Callable[[Any], Model] | None = None,
) -> WhatIfResult:
    """Calibrate (or read from the what-if cache) the base and moved leverages and compare
    their forward smiles (module docstring); the result is stored under
    ``<outputs>/whatif/runs/<key>.json`` and read back on a second call.  ``calibrate`` replaces
    the cache (tests inject a model without calibrating)."""
    runs = outputs_root / "whatif" / "runs"
    path = runs / f"{request.key()}.json"
    if path.exists():
        return WhatIfResult.from_json(path.read_text())
    st0 = base_state(request, repo_root)
    st1 = (
        st0.with_params(label="what-if moved", **dict(request.changes)) if request.changes else st0
    )
    info: dict[str, Any] = {
        "base_params": {k: float(getattr(st0.spec.model, k)) for k in PARAMS},
        "moved_params": {k: float(getattr(st1.spec.model, k)) for k in PARAMS},
        "base_key": st0.key,
        "moved_key": st1.key,
        "n_particles": request.n_particles,
        "n_paths": request.n_paths,
        "seed": request.seed,
    }
    t0 = time.perf_counter()
    if calibrate is None:
        from volsto.calibration.cache import LeverageCache

        cache = LeverageCache(outputs_root / "whatif" / "cache")
        hits = [cache.has(st0.spec), cache.has(st1.spec)]

        def calibrate(st: Any) -> Model:
            model, _ = cache.get_or_calibrate(st.spec, allow_calibrate=allow_calibrate)
            return model

        info["cache_hits"] = hits
    m0 = calibrate(st0)
    t_cal0 = time.perf_counter() - t0
    m1 = m0 if st1 is st0 else calibrate(st1)
    info["calibration_seconds"] = [t_cal0, time.perf_counter() - t0 - t_cal0]
    sim = SimConfig(
        n_paths=int(request.n_paths),
        chunk_size=min(int(request.n_paths), 20_000),
        seed=int(request.seed),
        dt_max=1.0 / 52.0,
    )
    t1 = time.perf_counter()
    smiles, fits = compare(m0, m1, request.windows, request.strikes, sim)
    info["pricing_seconds"] = time.perf_counter() - t1
    result = WhatIfResult(request, smiles, fits, info)
    runs.mkdir(parents=True, exist_ok=True)
    path.write_text(result.to_json())
    return result


def stored_runs(outputs_root: Path) -> list[WhatIfResult]:
    """Every stored what-if run, newest first."""
    runs = outputs_root / "whatif" / "runs"
    if not runs.is_dir():
        return []
    files = sorted(runs.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for p in files:
        try:
            out.append(WhatIfResult.from_json(p.read_text()))
        except (ValueError, KeyError):
            continue
    return out


def changes_label(changes: Mapping[str, float] | Sequence[tuple[str, float]]) -> str:
    items = changes.items() if isinstance(changes, Mapping) else changes
    return ", ".join(f"{k} {v:g}" for k, v in items) or "no change"


__all__ = [
    "DEFAULT_STRIKES",
    "DEFAULT_WINDOWS",
    "DESK_MARK",
    "PARAMS",
    "WHATIF_PARTICLES",
    "WHATIF_PATHS",
    "WhatIfRequest",
    "WhatIfResult",
    "base_params",
    "base_state",
    "changes_label",
    "compare",
    "run_whatif",
    "stored_runs",
]
