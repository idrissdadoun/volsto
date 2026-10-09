"""Content-addressed cache of calibrated local correlation functions (SPEC §8.7, M12 part LC4)
— the counterpart of :mod:`volsto.calibration.cache` for ``λ(t, k)``.

**Key** (:func:`lc_spec_key`): the SHA-256 of the canonical encoding
(:func:`volsto.calibration.fit_records.canonical`) of ``{"spec": spec.key_payload(),
"lc_code_tag": LC_CODE_TAG}``.  :meth:`~volsto.config.LocalCorrelationSpec.key_payload` holds
everything that determines ``λ`` and nothing else (pricing seed, path count and chunking are
left out).  The SVI parameters in the key are recorded fits, read back on every machine and
never refitted (``<root>/svi_fits``, :meth:`LocalCorrelationCache.svi_records`; SPEC §13.3–13.4).

**Entry** ``<root>/<key>/``: ``spec.json``, ``r_low.npy`` (the ``R_low`` matrix: a
``"historical-scaled"`` matrix is machine-dependent in its last bits, so it is stored and read
back, and the key hashes the digest of its data instead), ``lambda_surface.parquet`` (plot
data, particle family), ``lambda.npz`` (**the commit point**), ``diagnostics.json``; one row per
entry in ``manifest.parquet``.  Every file is written through
:func:`volsto.calibration.cache.atomic_write`; the manifest is rewritten under an advisory lock,
as the leverage cache's.

:meth:`LocalCorrelationCache.get_or_calibrate` is the only entry point for studies and risk;
with ``allow_calibrate=False`` a miss raises :class:`~volsto.calibration.cache.CacheMissError`
naming the key.  :func:`build_lc_market` builds the model's skeleton from a specification (the
counterpart of ``cache.build_market``).

**Code tag.**  :data:`~volsto.calibration.local_correlation.LC_CODE_TAG` is bumped whenever a
change can move a calibrated ``λ``; the sources of :data:`LC_GUARDED_MODULES` are hashed
against it in ``lc_code_tag_guard.json`` (:func:`check_lc_guard`,
``tests/test_local_correlation.py::test_lc_code_tag_guard``), like the leverage tag (SPEC §4.3).
"""

from __future__ import annotations

import dataclasses
import datetime as _dt
import fcntl
import hashlib
import json
import logging
import platform
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray

import volsto
from volsto._numba import num_threads
from volsto.calibration.cache import CacheMissError, atomic_write, code_version
from volsto.calibration.fit_records import FitRecords, canonical
from volsto.calibration.local_correlation import (
    LC_CODE_TAG,
    LCCalibrationResult,
    calibrate_constant_lambda,
    calibrate_local_correlation,
    calibrate_parametric_lambda,
    lambda_surface_frame,
    reprice_index_smile,
)
from volsto.config import LocalCorrelationSpec, SimConfig, SviSurfaceConfig, to_mapping
from volsto.engine.grid import TimeGrid
from volsto.market.curves import DiscountCurve, ForwardCurve
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import ArbitrageError, ImpliedSurface, perturbed_surface
from volsto.market.svi_slices import SviSlices
from volsto.models.localvol import LocalVol
from volsto.multi.family import (
    CorrelationFamily,
    family_from_spec,
    file_sha256,
    load_correlation_matrix,
    parse_r_high_spec,
    parse_r_low_spec,
)
from volsto.multi.lc_function import LocalCorrelationFunction
from volsto.multi.lc_model import BasketSpec, LocalCorrelationModel

FloatArray = NDArray[np.float64]
log = logging.getLogger(__name__)

#: The entry's commit point (written after its other files, before the report and the row).
LAMBDA_NAME: Final[str] = "lambda.npz"
SPEC_NAME: Final[str] = "spec.json"
DIAGNOSTICS_NAME: Final[str] = "diagnostics.json"
SURFACE_NAME: Final[str] = "lambda_surface.parquet"
R_LOW_NAME: Final[str] = "r_low.npy"
MANIFEST_NAME: Final[str] = "manifest.parquet"
MANIFEST_LOCK_NAME: Final[str] = "manifest.parquet.lock"
#: Sub-directory of the cache that holds the SVI slice fit records.
SVI_FITS_DIR: Final[str] = "svi_fits"
#: A listed index expiry whose forward is further than this from the basket's is flagged: a
#: dividend or divisor problem, not a smile feature.
ALIGNMENT_FLAG: Final[float] = 0.01

#: Modules whose source can move a calibrated ``λ``: the kernel, the model, the draws, the
#: function, the family and the mixing; the normal draws; the shared spot step; the Dupire
#: surface and the SVI slices it is built from; the calibration and the leverage's estimators
#: it reuses.
LC_GUARDED_MODULES: Final[tuple[str, ...]] = (
    "volsto/multi/lc_kernel.py",
    "volsto/multi/lc_model.py",
    "volsto/multi/lc_draws.py",
    "volsto/multi/lc_function.py",
    "volsto/multi/family.py",
    "volsto/multi/draws.py",
    "volsto/engine/rng.py",
    "volsto/models/localvol.py",
    "volsto/market/dupire.py",
    "volsto/market/svi_slices.py",
    "volsto/calibration/local_correlation.py",
    "volsto/calibration/particle.py",
    "volsto/calibration/binned.py",
)
LC_GUARD_FILE = Path(__file__).resolve().parent / "lc_code_tag_guard.json"


def lc_spec_key(spec: LocalCorrelationSpec, code_tag: str = LC_CODE_TAG) -> str:
    """The cache key of ``spec`` under ``code_tag`` (module docstring)."""
    payload = {"spec": spec.key_payload(), "lc_code_tag": code_tag}
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


# --------------------------------------------------------------------------------------------
# the market of a specification
# --------------------------------------------------------------------------------------------


@dataclass
class LCMarket:
    """The model's skeleton and its inputs' diagnostics (:func:`build_lc_market`).

    ``arbitrage[name]`` is the SVI surface's :class:`~volsto.market.svi_slices.ArbitrageReport`
    as a mapping (``"index"`` for the target) and ``flagged`` the names whose report has a
    violation; ``dupire[name]`` the floored fraction, the smallest denominator and the smallest
    ``∂_T w`` of the Dupire surface; ``alignment`` one row per listed index expiry — ``T``,
    ``delta = ln(Σ w_i f_i(T)) − ln(F_I(T)/I_0)`` and whether ``|delta|`` exceeds 1 %."""

    names: tuple[str, ...]
    models: list[LocalVol]
    surfaces: list[ImpliedSurface]
    family: CorrelationFamily
    basket: BasketSpec
    index_surface: ImpliedSurface
    index_lv: LocalVolSurface
    arbitrage: dict[str, dict[str, Any]] = field(default_factory=dict)
    flagged: list[str] = field(default_factory=list)
    dupire: dict[str, dict[str, float]] = field(default_factory=dict)
    alignment: list[dict[str, Any]] = field(default_factory=list)

    def summary(self) -> dict[str, Any]:
        """The JSON-able diagnostics of the inputs (stored with a cache entry)."""
        floored = [d["floored_fraction"] for k, d in self.dupire.items() if k != "index"]
        return {
            "flagged": list(self.flagged),
            "arbitrage_ok": not self.flagged,
            "violations": {k: self.arbitrage[k]["violations"] for k in self.flagged},
            "dupire": self.dupire,
            "max_floored_fraction_names": float(max(floored)) if floored else 0.0,
            "floored_fraction_index": self.dupire.get("index", {}).get("floored_fraction", 0.0),
            "alignment": self.alignment,
            "alignment_flagged": bool(any(a["flagged"] for a in self.alignment)),
            "family": self.family.describe(),
            "mode": self.basket.mode,
        }


def _svi_surface(cfg: SviSurfaceConfig, curve: ForwardCurve) -> SviSlices:
    return SviSlices(np.array(cfg.times), np.array(cfg.params), curve, cfg.max_maturity)


def _check_file_digest(label: str, path: str, expected: str | None) -> None:
    """A matrix file read for ``label`` must have the digest the specification (and so the
    cache key) holds."""
    digest = file_sha256(path)
    if digest != expected:
        raise ValueError(
            f"{label}: the file {path} has SHA-256 {digest}, not the specification's "
            f"{label}_source ({expected}): the matrix changed under its path"
        )


#: The central range of the visited-range arbitrage report, in standard deviations of the cloud.
VISITED_CENTRAL_SD: Final[float] = 3.0


def visited_arbitrage(
    spec: LocalCorrelationSpec,
    result: LCCalibrationResult,
    *,
    n_k: int = 401,
    central_sd: float = VISITED_CENTRAL_SD,
) -> dict[str, Any]:
    """The arbitrage report of every SVI surface of ``spec`` on the range its particles visited
    (owner's decision of 2026-10-08, SPEC §8.7; the report on ``|k| ≤ 1`` of
    :func:`build_lc_market` flags far wings no particle reaches).

    Slice ``s`` of name ``i`` is checked on ``[min, max]`` of the name's forward log-moneyness
    over the calibration cloud at the grid time nearest ``min(T_s, horizon)`` (never the
    degenerate ``t_0``); a pair of slices on the union of their two ranges; the index target
    likewise on the basket's range.  The extremes of 8·10⁵ particles are four to five standard
    deviations out, so the same report is also run on the central range, the cloud mean
    ``± central_sd`` standard deviations (inside the visited range): a violation there is where
    the particles are.

    Returns ``{"flagged": [labels], "violations": {label: [lines]}, "flagged_central": [labels],
    "violations_central": {label: [lines]}, "ranges": {label: {"T", "k_lo", "k_hi",
    "k_lo_central", "k_hi_central"}}}``."""
    times = result.grid.times
    horizon = float(times[-1])

    def rows(slice_times: Sequence[float]) -> list[int]:
        return [max(1, int(np.argmin(np.abs(times - min(float(t), horizon))))) for t in slice_times]

    curves = [ForwardCurve.from_config(m) for m in spec.markets]
    flat = ForwardCurve(1.0, DiscountCurve.flat(0.0), DiscountCurve.flat(0.0))
    out: dict[str, Any] = {
        "central_sd": float(central_sd),
        "flagged": [],
        "violations": {},
        "flagged_central": [],
        "violations_central": {},
        "ranges": {},
    }
    r = result
    cases = [
        (name, cfg, curve, i)
        for i, (name, cfg, curve) in enumerate(zip(spec.names, spec.surfaces, curves, strict=True))
    ]
    cases.append(("index", spec.index_surface, flat, -1))
    for label, cfg, curve, i in cases:
        if i >= 0:
            lo, hi = r.name_k_min[:, i], r.name_k_max[:, i]
            mu, sd = r.name_k_mean[:, i], r.name_k_std[:, i]
        else:
            lo, hi, mu, sd = r.basket_k_min, r.basket_k_max, r.basket_k_mean, r.basket_k_std
        j = rows(cfg.times)
        k_lo, k_hi = [float(lo[x]) for x in j], [float(hi[x]) for x in j]
        c_lo = [max(float(mu[x] - central_sd * sd[x]), a) for x, a in zip(j, k_lo, strict=True)]
        c_hi = [min(float(mu[x] + central_sd * sd[x]), b) for x, b in zip(j, k_hi, strict=True)]
        surface = _svi_surface(cfg, curve)
        full = surface.arbitrage_report(n_k=n_k, k_lo=k_lo, k_hi=k_hi)
        central = surface.arbitrage_report(n_k=n_k, k_lo=c_lo, k_hi=c_hi)
        out["ranges"][label] = {
            "T": [float(t) for t in cfg.times],
            "k_lo": k_lo,
            "k_hi": k_hi,
            "k_lo_central": c_lo,
            "k_hi_central": c_hi,
        }
        if not full.ok:
            out["flagged"].append(label)
            out["violations"][label] = full.violations()
        if not central.ok:
            out["flagged_central"].append(label)
            out["violations_central"][label] = central.violations()
    return out


def _dupire_summary(lv: LocalVolSurface) -> dict[str, float]:
    d = lv.check_positive()
    return {
        "floored_fraction": float(d.floored_fraction),
        "min_denominator": float(d.min_denominator),
        "min_dw_dt": float(d.min_dw_dt),
    }


def _basket_forward_curve(basket: BasketSpec, curves: Sequence[ForwardCurve]) -> ForwardCurve:
    """The curve the index target sits on: ``F_B ≡ 1`` in performance mode; in carry mode the
    curve through ``F_B(t) = Σ w_i F_i(t)/S_i(0)`` at the names' own curve pillars (log-linear
    between them).  The calibration reads ``ln F_B`` from the basket itself; this curve is the
    surface's label."""
    zero = DiscountCurve.flat(0.0)
    if basket.mode == "performance":
        return ForwardCurve(1.0, zero, zero)
    pillars = np.unique(
        np.concatenate(
            [np.concatenate([fc.rate_curve.times, fc.dividend_curve.times]) for fc in curves]
        )
    )
    forwards = np.exp(basket.log_basket_forward(pillars))
    return ForwardCurve.from_forwards(1.0, pillars, forwards, zero)


def build_lc_market(
    spec: LocalCorrelationSpec,
    *,
    r_low_matrix: ArrayLike | None = None,
    r_high_matrix: ArrayLike | None = None,
) -> LCMarket:
    """The names' local-vol models, the correlation family, the basket state and the index
    target of ``spec`` (the counterpart of ``cache.build_market``).

    Each name: its forward curve from its market, its SVI surface (with its perturbation, if
    any), the Dupire local variance on the shared grid ``spec.local_vol``.  The index target:
    its SVI surface in its own forward moneyness, used at the basket's forward moneyness
    (SPEC §8.7: aligned in forward moneyness, not in strike), with its perturbation; its Dupire
    surface on ``spec.lc.lambda_grid`` (default: the shared grid) is the ``λ`` grid.

    ``r_low_matrix`` / ``r_high_matrix``: the matrices of a non-default family — for a
    ``"matrix:<path>"`` specification the file is read when no matrix is given, and its SHA-256
    must be the specification's ``r_low_source`` / ``r_high_source`` (the key hashes the digest:
    a file that changed under its path is refused); a ``"historical-scaled"`` ``R_low`` must be
    given (the cache passes the stored one).

    Arbitrage policy ``spec.lc.arbitrage``: ``"raise"`` raises
    :class:`~volsto.market.surface.ArbitrageError` on the first surface with a butterfly or
    calendar violation; ``"flag"`` records it and goes on (the Dupire floor and cap apply).
    """
    lc = spec.lc
    curves = [ForwardCurve.from_config(m) for m in spec.markets]
    pert = spec.perturbations or (None,) * spec.n_names
    market_arb: dict[str, dict[str, Any]] = {}
    flagged: list[str] = []
    dupire: dict[str, dict[str, float]] = {}

    def checked(label: str, base: SviSlices) -> None:
        report = base.arbitrage_report()
        market_arb[label] = report.to_dict()
        if not report.ok:
            if lc.arbitrage == "raise":
                raise ArbitrageError(f"{label}: " + "; ".join(report.violations()))
            flagged.append(label)

    surfaces: list[ImpliedSurface] = []
    models: list[LocalVol] = []
    for name, cfg, curve, p in zip(spec.names, spec.surfaces, curves, pert, strict=True):
        base = _svi_surface(cfg, curve)
        checked(name, base)
        surface = perturbed_surface(p, base, check=False)
        lv = LocalVolSurface.from_implied(surface, spec.local_vol)
        dupire[name] = _dupire_summary(lv)
        surfaces.append(surface)
        models.append(LocalVol(lv, curve))
    basket = BasketSpec(np.array(spec.weights), lc.mode, curves)
    index_base = _svi_surface(spec.index_surface, _basket_forward_curve(basket, curves))
    checked("index", index_base)
    index_surface = perturbed_surface(spec.index_perturbation, index_base, check=False)
    index_lv = LocalVolSurface.from_implied(index_surface, lc.lambda_grid or spec.local_vol)
    dupire["index"] = _dupire_summary(index_lv)
    # the family
    low_kind, low_args = parse_r_low_spec(lc.r_low)
    high_kind, high_args = parse_r_high_spec(lc.r_high)
    if low_kind == "matrix" and r_low_matrix is None:
        _check_file_digest("r_low", low_args["path"], spec.r_low_source)
        r_low_matrix = load_correlation_matrix(low_args["path"], spec.names)
    if low_kind == "historical-scaled" and r_low_matrix is None:
        raise ValueError(
            f"r_low = {lc.r_low!r}: the matrix must be given (historical_scaled_correlation on the "
            "window's returns; the cache stores it with the entry and reads it back)"
        )
    if high_kind == "matrix" and r_high_matrix is None:
        _check_file_digest("r_high", high_args["path"], spec.r_high_source)
        r_high_matrix = load_correlation_matrix(high_args["path"], spec.names)
    family = family_from_spec(
        spec.n_names,
        r_low=lc.r_low,
        r_high=lc.r_high,
        rho_min=lc.rho_min,
        rho_max=lc.rho_max,
        lambda_max=lc.lambda_max,
        r_low_matrix=r_low_matrix,
        r_high_matrix=r_high_matrix,
    )
    # alignment of the listed index forwards with the basket's
    alignment: list[dict[str, Any]] = []
    w = np.array(spec.weights)
    for t_e, ratio in spec.index_forward_ratios:
        f_basket = float(w @ np.array([float(fc.forward(t_e)) / fc.spot for fc in curves]))
        delta = float(np.log(f_basket) - np.log(ratio))
        alignment.append(
            {"T": float(t_e), "delta": delta, "flagged": bool(abs(delta) > ALIGNMENT_FLAG)}
        )
    return LCMarket(
        spec.names,
        models,
        surfaces,
        family,
        basket,
        index_surface,
        index_lv,
        market_arb,
        flagged,
        dupire,
        alignment,
    )


# --------------------------------------------------------------------------------------------
# diagnostics of an entry
# --------------------------------------------------------------------------------------------


@dataclass
class LCDiagnostics:
    """What a cache entry records beside ``λ``: the calibration's summary (clipped masses,
    trusted ranges, timings — or the parametric / constant fit), the inputs' diagnostics
    (arbitrage flags, Dupire floors, the alignment of the index forwards), the index repricing
    report when one was run, and the reproducibility record (git commit, key, code tag, seeds,
    particle and path counts, threads, machine)."""

    calibration: dict[str, Any]
    market: dict[str, Any]
    record: dict[str, Any]
    index_report: dict[str, Any] | None = None

    @property
    def max_clipped_mass(self) -> float:
        return float(self.calibration.get("max_clipped_mass", float("nan")))

    @property
    def max_index_error_vp(self) -> float | None:
        if self.index_report is None:
            return None
        return float(self.index_report["max_abs_error_vp_1p5sd"])

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration": self.calibration,
            "market": self.market,
            "record": self.record,
            "index_report": self.index_report,
        }

    @classmethod
    def from_dict(cls, doc: dict[str, Any]) -> LCDiagnostics:
        return cls(doc["calibration"], doc["market"], doc["record"], doc.get("index_report"))


def reproducibility_record(spec: LocalCorrelationSpec, key: str) -> dict[str, Any]:
    """The record every artefact of the model carries (SPEC §11, §8.7)."""
    return {
        "git_commit": code_version(),
        "volsto_version": volsto.__version__,
        "key": key,
        "lc_code_tag": LC_CODE_TAG,
        "label": spec.label,
        "particle_seed": spec.lc.particle.seed,
        "pricing_seed": spec.sim.seed,
        "n_particles": spec.lc.particle.n_particles,
        "n_paths": spec.sim.n_paths,
        "threads": num_threads(),
        "machine": platform.machine(),
        "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
    }


# --------------------------------------------------------------------------------------------
# the cache
# --------------------------------------------------------------------------------------------


def _is_complete_npz(path: Path) -> bool:
    """``path`` exists and is a complete zip archive (every member's CRC checks): a file cut
    short by a kill is a miss, as in the leverage cache."""
    try:
        if not path.is_file():
            return False
        with zipfile.ZipFile(path) as zf:
            return zf.testzip() is None
    except (OSError, zipfile.BadZipFile, EOFError, ValueError):
        return False


class LocalCorrelationCache:
    """Directory-backed cache of calibrated ``λ`` (module docstring).  The root is not created
    by a reader."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    # -- addressing ----------------------------------------------------------------------------

    def key(self, spec: LocalCorrelationSpec) -> str:
        return lc_spec_key(spec)

    def entry_dir(self, spec: LocalCorrelationSpec) -> Path:
        return self.root / self.key(spec)

    def has(self, spec: LocalCorrelationSpec) -> bool:
        """A committed entry: a complete ``lambda.npz``."""
        return _is_complete_npz(self.entry_dir(spec) / LAMBDA_NAME)

    def svi_records(self) -> FitRecords:
        """The SVI slice fit records that travel with this cache (``<root>/svi_fits``)."""
        return FitRecords(self.root / SVI_FITS_DIR)

    # -- read ----------------------------------------------------------------------------------

    def load(self, spec: LocalCorrelationSpec) -> LocalCorrelationFunction:
        p = self.entry_dir(spec) / LAMBDA_NAME
        if not p.exists():
            raise CacheMissError(self.key(spec))
        return LocalCorrelationFunction.load(p)

    def load_diagnostics(self, spec: LocalCorrelationSpec) -> LCDiagnostics | None:
        p = self.entry_dir(spec) / DIAGNOSTICS_NAME
        if not p.exists():
            return None
        return LCDiagnostics.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def load_r_low(self, spec: LocalCorrelationSpec) -> FloatArray | None:
        p = self.entry_dir(spec) / R_LOW_NAME
        if not p.exists():
            return None
        return np.asarray(np.load(p, allow_pickle=False), dtype=np.float64)

    def manifest(self) -> pd.DataFrame:
        path = self.root / MANIFEST_NAME
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    # -- write ---------------------------------------------------------------------------------

    def store(
        self,
        spec: LocalCorrelationSpec,
        lam: LocalCorrelationFunction,
        diagnostics: LCDiagnostics,
        *,
        r_low: FloatArray,
        frame: pd.DataFrame | None = None,
    ) -> Path:
        """Write the entry: ``spec.json``, ``r_low.npy``, the plot data, then ``lambda.npz``
        (the commit point), then the diagnostics and the manifest row."""
        key = self.key(spec)
        d = self.root / key
        d.mkdir(parents=True, exist_ok=True)
        lam.metadata.update(
            {
                "cache_key": key,
                "git_commit": diagnostics.record["git_commit"],
                "lc_code_tag": LC_CODE_TAG,
                "created_utc": diagnostics.record["created_utc"],
            }
        )
        spec_text = json.dumps(to_mapping(spec), indent=1, default=str)
        atomic_write(d / SPEC_NAME, lambda tmp: tmp.write_text(spec_text))
        atomic_write(d / R_LOW_NAME, lambda tmp: np.save(tmp, r_low))
        if frame is not None:
            atomic_write(d / SURFACE_NAME, lambda tmp: frame.to_parquet(tmp, index=False))
        # a report left from an earlier lambda of this key does not describe the new one
        (d / DIAGNOSTICS_NAME).unlink(missing_ok=True)
        atomic_write(d / LAMBDA_NAME, lam.save)  # the commit point
        self.write_diagnostics(spec, diagnostics)
        self._append_manifest(key, spec, diagnostics)
        return d

    def write_diagnostics(self, spec: LocalCorrelationSpec, diagnostics: LCDiagnostics) -> Path:
        text = json.dumps(diagnostics.to_dict(), indent=1, default=str)
        return atomic_write(
            self.entry_dir(spec) / DIAGNOSTICS_NAME, lambda tmp: tmp.write_text(text)
        )

    def _append_manifest(
        self, key: str, spec: LocalCorrelationSpec, diagnostics: LCDiagnostics
    ) -> None:
        cal = diagnostics.calibration
        row: dict[str, Any] = {
            "key": key,
            "created_utc": diagnostics.record["created_utc"],
            "git_commit": diagnostics.record["git_commit"],
            "lc_code_tag": LC_CODE_TAG,
            "label": spec.label,
            "n_names": spec.n_names,
            "family": spec.lc.family,
            "r_low": spec.lc.r_low,
            "mode": spec.lc.mode,
            "n_particles": spec.lc.particle.n_particles,
            "seed": spec.lc.particle.seed,
            "horizon": spec.lc.particle.horizon,
            "max_clipped_mass": float(cal.get("max_clipped_mass", float("nan"))),
            "max_index_error_vp": diagnostics.max_index_error_vp,
            "wall_time": float(cal.get("wall_time", float("nan"))),
        }
        row_df = pd.DataFrame([row])
        self._update_manifest(
            lambda old: (
                row_df
                if old.empty
                else pd.concat([old[old["key"] != key], row_df], ignore_index=True)
            )
        )

    def _update_manifest(self, update: Callable[[pd.DataFrame], pd.DataFrame]) -> pd.DataFrame:
        """``update(current manifest)`` under the lock, published atomically (the leverage
        cache's discipline)."""
        self.root.mkdir(parents=True, exist_ok=True)
        path = self.root / MANIFEST_NAME
        with open(self.root / MANIFEST_LOCK_NAME, "a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                old = pd.read_parquet(path) if path.exists() else pd.DataFrame({"key": []})
                new = update(old)
                atomic_write(path, lambda tmp: new.to_parquet(tmp, index=False))
                return new
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    # -- entry point ---------------------------------------------------------------------------

    def get_or_calibrate(
        self,
        spec: LocalCorrelationSpec,
        *,
        allow_calibrate: bool = True,
        run_diagnostics: bool = False,
        pricing_sim: SimConfig | None = None,
        r_low_matrix: ArrayLike | None = None,
        r_high_matrix: ArrayLike | None = None,
    ) -> tuple[LocalCorrelationModel, LCDiagnostics | None]:
        """The local correlation model of ``spec``, calibrating and storing on a miss.

        A hit reads ``λ`` and the stored ``R_low`` back and calibrates nothing.  On a miss with
        ``allow_calibrate=False`` :class:`~volsto.calibration.cache.CacheMissError` names the
        key.  ``run_diagnostics`` reprices the index smile at the target's slices up to the
        horizon with ``pricing_sim`` (default ``spec.sim``) and stores the report.
        ``r_low_matrix``: the matrix of a ``"historical-scaled"`` ``R_low`` (needed on a miss
        only).  The family ``spec.lc.family`` selects the calibration: the particle method,
        the two-parameter fit (at ``parametric.maturity``, default the horizon) or the constant
        that reprices the at-the-money straddle at the horizon — the last two on
        ``parametric.n_paths`` paths of the calibration seed.
        """
        key = self.key(spec)
        if self.has(spec):
            log.info("local correlation cache hit %s", key[:12])
            # a matrix R_low is read back from the entry (never recomputed, never re-read from
            # its file); the equicorrelation is rebuilt from rho_min
            equi = parse_r_low_spec(spec.lc.r_low)[0] == "equi"
            stored = None if equi else self.load_r_low(spec)
            market = build_lc_market(spec, r_low_matrix=stored, r_high_matrix=r_high_matrix)
            model = LocalCorrelationModel(
                market.models, market.family, self.load(spec), market.basket, spec.names
            )
            return model, self.load_diagnostics(spec)
        if not allow_calibrate:
            raise CacheMissError(f"no calibrated local correlation for key {key}")
        log.info("local correlation cache miss %s: calibrating (%s)", key[:12], spec.lc.family)
        market = build_lc_market(spec, r_low_matrix=r_low_matrix, r_high_matrix=r_high_matrix)
        lam, calibration, frame = _calibrate(spec, market)
        model = LocalCorrelationModel(market.models, market.family, lam, market.basket, spec.names)
        report = None
        if run_diagnostics:
            horizon = spec.lc.particle.horizon
            pillars = [t for t in spec.index_surface.times if t < horizon - 1e-9] + [horizon]
            report = reprice_index_smile(
                model, market.index_surface, pricing_sim or spec.sim, maturities=pillars
            ).to_dict()
        diagnostics = LCDiagnostics(
            calibration, market.summary(), reproducibility_record(spec, key), report
        )
        self.store(spec, lam, diagnostics, r_low=market.family.r_low, frame=frame)
        return model, diagnostics


def _calibrate(
    spec: LocalCorrelationSpec, market: LCMarket
) -> tuple[LocalCorrelationFunction, dict[str, Any], pd.DataFrame | None]:
    """``(λ, calibration summary, plot data)`` of ``spec`` by its family."""
    lc = spec.lc
    horizon = lc.particle.horizon
    if lc.family == "particle":
        result = calibrate_local_correlation(
            market.models,
            market.family,
            market.basket,
            market.index_surface,
            market.index_lv,
            lc.particle,
            spec.sim,
            lc,
        )
        summary = {
            "family": "particle",
            **result.summary(),
            "arbitrage_visited": visited_arbitrage(spec, result),
        }
        return result.lam, summary, lambda_surface_frame(result)
    # the two fitted families: a skeleton on the calibration grid's slices
    times = TimeGrid.build([horizon], spec.sim.dt_max).times
    k_grid = market.index_lv.k_grid
    zero = LocalCorrelationFunction.constant(0.0, times, k_grid)
    skeleton = LocalCorrelationModel(market.models, market.family, zero, market.basket, spec.names)
    fit_sim = dataclasses.replace(spec.sim, n_paths=lc.parametric.n_paths, seed=lc.particle.seed)
    if lc.family == "constant":
        value = calibrate_constant_lambda(skeleton, market.index_surface, horizon, fit_sim)
        meta = {"seed": lc.particle.seed, "n_paths": lc.parametric.n_paths, "code_tag": LC_CODE_TAG}
        lam = LocalCorrelationFunction.constant(value, times, k_grid, meta)
        return lam, {"family": "constant", "lambda": value, "max_clipped_mass": 0.0}, None
    maturity = lc.parametric.maturity if lc.parametric.maturity is not None else horizon
    fit = calibrate_parametric_lambda(
        skeleton, market.index_surface, maturity, lc.parametric, spec.sim, seed=lc.particle.seed
    )
    meta = {
        "seed": lc.particle.seed,
        "n_paths": lc.parametric.n_paths,
        "code_tag": LC_CODE_TAG,
        "converged": fit.converged,
        "rho0": fit.rho0,
        "c": fit.c,
    }
    lam = LocalCorrelationFunction.parametric(fit.lam, times, k_grid, meta)
    return lam, {"family": "parametric", **fit.to_dict(), "max_clipped_mass": 0.0}, None


# --------------------------------------------------------------------------------------------
# code-tag guard
# --------------------------------------------------------------------------------------------


def lc_source_hash() -> str:
    """SHA-256 of the concatenated source of :data:`LC_GUARDED_MODULES` (line endings
    normalised) — the construction of the leverage guard."""
    root = Path(volsto.__file__).resolve().parents[1]
    h = hashlib.sha256()
    for rel in LC_GUARDED_MODULES:
        text = (root / rel).read_text(encoding="utf-8").replace("\r\n", "\n")
        h.update(rel.encode())
        h.update(b"\0")
        h.update(text.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def read_lc_guard() -> dict[str, str]:
    """Stored ``{lc_code_tag: source_hash}``; a missing file is an empty mapping."""
    if LC_GUARD_FILE.exists():
        return {str(k): str(v) for k, v in json.loads(LC_GUARD_FILE.read_text()).items()}
    return {}


def write_lc_guard() -> Path:
    """Record the current source hash under the current tag (after bumping the tag, or after a
    change proven not to move any ``λ`` — say which in the commit)."""
    guard = read_lc_guard()
    guard[LC_CODE_TAG] = lc_source_hash()
    LC_GUARD_FILE.write_text(json.dumps(guard, indent=1, sort_keys=True) + "\n")
    return LC_GUARD_FILE


def check_lc_guard() -> None:
    """Raise if the guarded sources changed without a bump of ``LC_CODE_TAG``
    (``tests/test_local_correlation.py::test_lc_code_tag_guard``).  To accept a change: bump
    ``LC_CODE_TAG`` in ``volsto/calibration/local_correlation.py`` and run ``python -c "from
    volsto.calibration.lc_cache import write_lc_guard; write_lc_guard()"``."""
    stored = read_lc_guard().get(LC_CODE_TAG)
    current = lc_source_hash()
    if stored is None:
        raise AssertionError(
            f"no stored source hash for LC_CODE_TAG={LC_CODE_TAG!r}; run write_lc_guard()"
        )
    if stored != current:
        raise AssertionError(
            "local correlation sources changed but LC_CODE_TAG "
            f"({LC_CODE_TAG!r}) was not bumped: stored {stored[:12]}…, current {current[:12]}… — "
            "bump the tag and run write_lc_guard()"
        )


__all__ = [
    "LC_GUARDED_MODULES",
    "LCDiagnostics",
    "LCMarket",
    "LocalCorrelationCache",
    "build_lc_market",
    "check_lc_guard",
    "lc_source_hash",
    "lc_spec_key",
    "read_lc_guard",
    "reproducibility_record",
    "visited_arbitrage",
    "write_lc_guard",
]
