"""Content-addressed cache of calibrated leverage functions (SPEC §4.3).

Key = SHA-256 of the canonical JSON of :meth:`~volsto.config.CalibrationSpec.key_payload` plus
the calibration code tag (:data:`~volsto.calibration.particle.CALIBRATION_CODE_TAG`, bumped when
the calibration numerics change).  Each entry directory holds ``leverage.npz`` (with provenance
metadata: git commit, config hash, seeds) and optionally ``diagnostics.json``; ``manifest.parquet``
at the cache root has one row per entry.  ``get_or_calibrate`` is the only entry point studies and
viewers use; a viewer passes ``allow_calibrate=False`` and reports what is missing.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

import volsto
from volsto.calibration.diagnostics import CalibrationReport, reprice_surface
from volsto.calibration.particle import CALIBRATION_CODE_TAG, calibrate_leverage
from volsto.config import CalibrationSpec, SimConfig, to_mapping
from volsto.market.curves import ForwardCurve
from volsto.market.surface import SSVISurface
from volsto.market.varswap import xi0_curve
from volsto.models.bergomi import BergomiSV
from volsto.models.leverage import LeverageFunction
from volsto.models.lsv import LSV

log = logging.getLogger(__name__)


class CacheMissError(KeyError):
    """The requested calibration is not in the cache."""


def code_version() -> str:
    """Short git commit of the working tree when available, else the package version."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(volsto.__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return f"volsto-{volsto.__version__}"


def spec_key(spec: CalibrationSpec, code_tag: str = CALIBRATION_CODE_TAG) -> str:
    payload = {"spec": spec.key_payload(), "code_tag": code_tag}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(blob).hexdigest()


def build_market(spec: CalibrationSpec) -> tuple[ForwardCurve, SSVISurface, BergomiSV]:
    """Forward curve, SSVI surface and pure SV kernel (ξ₀ from the variance-swap strip)."""
    fc = ForwardCurve.from_config(spec.market)
    surface = SSVISurface.from_config(spec.surface, fc, fc.rate_curve)
    t_max = min(surface.max_maturity, max(spec.particle.horizon + 1.0, 5.0))
    xi0 = xi0_curve(surface, t_max)
    return fc, surface, BergomiSV(spec.model, xi0, fc)


class LeverageCache:
    """Directory-backed cache; see module docstring."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # -- addressing ----------------------------------------------------------------------------

    def key(self, spec: CalibrationSpec) -> str:
        return spec_key(spec)

    def entry_dir(self, spec: CalibrationSpec) -> Path:
        return self.root / self.key(spec)

    def has(self, spec: CalibrationSpec) -> bool:
        return (self.entry_dir(spec) / "leverage.npz").exists()

    # -- read / write --------------------------------------------------------------------------

    def load(self, spec: CalibrationSpec) -> LeverageFunction:
        p = self.entry_dir(spec) / "leverage.npz"
        if not p.exists():
            raise CacheMissError(self.key(spec))
        return LeverageFunction.load(p)

    def load_report(self, spec: CalibrationSpec) -> CalibrationReport | None:
        p = self.entry_dir(spec) / "diagnostics.json"
        return CalibrationReport.load(p) if p.exists() else None

    def store(
        self,
        spec: CalibrationSpec,
        leverage: LeverageFunction,
        report: CalibrationReport | None = None,
    ) -> Path:
        key = self.key(spec)
        d = self.root / key
        d.mkdir(parents=True, exist_ok=True)
        leverage.metadata.update(
            {
                "cache_key": key,
                "git_commit": code_version(),
                "code_tag": CALIBRATION_CODE_TAG,
                "created_utc": _dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
                "spec": spec.key_payload(),
            }
        )
        leverage.save(d / "leverage.npz")
        (d / "spec.json").write_text(json.dumps(to_mapping(spec), indent=1, default=str))
        if report is not None:
            report.save(d / "diagnostics.json")
        self._append_manifest(key, spec, leverage, report)
        return d

    def _append_manifest(
        self,
        key: str,
        spec: CalibrationSpec,
        leverage: LeverageFunction,
        report: CalibrationReport | None,
    ) -> None:
        m = spec.model
        row: dict[str, Any] = {
            "key": key,
            "created_utc": leverage.metadata.get("created_utc"),
            "git_commit": leverage.metadata.get("git_commit"),
            "code_tag": CALIBRATION_CODE_TAG,
            "horizon": spec.particle.horizon,
            "n_particles": spec.particle.n_particles,
            "seed": spec.particle.seed,
            "nu": m.nu,
            "theta": m.theta,
            "k1": m.k1,
            "k2": m.k2,
            "rho12": m.rho12,
            "rho_SX1": m.rho_SX1,
            "rho_SX2": m.rho_SX2,
            "surface_rho": spec.surface.rho,
            "surface_eta": spec.surface.eta,
            "surface_gamma": spec.surface.gamma,
            "wall_time": float(leverage.metadata.get("wall_time", float("nan"))),
            "max_abs_error_vp": report.max_abs_error() if report is not None else None,
        }
        path = self.root / "manifest.parquet"
        df = pd.DataFrame([row])
        if path.exists():
            old = pd.read_parquet(path)
            old = old[old["key"] != key]
            df = pd.concat([old, df], ignore_index=True)
        df.to_parquet(path, index=False)

    def manifest(self) -> pd.DataFrame:
        path = self.root / "manifest.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()

    # -- entry point ---------------------------------------------------------------------------

    def get_or_calibrate(
        self,
        spec: CalibrationSpec,
        *,
        allow_calibrate: bool = True,
        run_diagnostics: bool = False,
        diagnostics_sim: SimConfig | None = None,
    ) -> tuple[LSV, CalibrationReport | None]:
        """Return the LSV model for ``spec``, calibrating and storing on a miss.

        Viewers call with ``allow_calibrate=False`` (calibration never runs silently there) and
        get :class:`CacheMissError` naming the key.
        """
        _, surface, kernel = build_market(spec)
        if self.has(spec):
            log.info("leverage cache hit %s", self.key(spec)[:12])
            return LSV(kernel, self.load(spec)), self.load_report(spec)
        if not allow_calibrate:
            raise CacheMissError(f"no calibrated leverage for key {self.key(spec)}")
        log.info("leverage cache miss %s: calibrating", self.key(spec)[:12])
        result = calibrate_leverage(
            surface, kernel, spec.particle, spec.sim, local_vol_cfg=spec.local_vol
        )
        model = LSV(kernel, result.leverage)
        report = None
        if run_diagnostics:
            report = reprice_surface(model, surface, diagnostics_sim or spec.sim)
        self.store(spec, result.leverage, report)
        return model, report
