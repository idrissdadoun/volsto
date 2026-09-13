"""M4 regression tests (SPEC §10, owner request at M3.2/M3b acceptance).

Part A — the original study's numbers (its SSVI surface and seeds) can only be checked against
``study_archive/``; those tests are skipped until the archive is present.

Part B — the same headline set measured on the **placeholder** reference surface at M4 with
:func:`volsto.studies.m4.run_headline` (400k paths, seed 2024, the default schedule and scheme
shared with calibration).  Tolerance ``max(2 stderr, 0.02% of notional)`` for prices and
``max(2 stderr, 0.02 vol points)`` for volatilities; with the same seed the run is
deterministic, so this guards the numerics (scheme, calibration, products) rather than the noise.
Slow: it needs the four leverage calibrations (cache hits after the first run).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from volsto.calibration import LeverageCache
from volsto.config import CalibrationSpec, SimConfig, load_yaml
from volsto.market import ForwardCurve
from volsto.models import BlackScholes
from volsto.studies.m4 import (
    HEADLINE_OMEGAS,
    LV_NAME,
    TWO_FACTOR_NAME,
    headline_models,
    one_factor_name,
    run_headline,
)

ROOT = Path(__file__).resolve().parents[1]
STUDY_ARCHIVE = ROOT / "study_archive"
SPEC_1F = ROOT / "configs" / "studies" / "lsv_reference_1f.yaml"
SPEC_2F = ROOT / "configs" / "studies" / "lsv_reference_2f.yaml"

HEADLINE_N_PATHS = 400_000
HEADLINE_SEED = 2024

# SPEC §10 numbers recalled from the original study (its own surface and seeds): 1y-into-1y ATM
# forward vol, forward variance-swap level and the 1y / 2y capped cliquets for ω = 0/1/2/3.
STUDY_NUMBERS = {
    "atm_vol": {0.0: 0.215, 3.0: 0.186},
    "vs_vol_range": (0.252, 0.255),
    "cliquet_1y": {0.0: 1.082, 1.0: 1.194, 2.0: 1.472, 3.0: 1.763},
    "cliquet_2y": {0.0: 0.639, 3.0: 1.718},
}

# Placeholder-surface baselines measured at M4 (scripts/m4_headline.py, 400k paths, seed 2024;
# SPEC §6 notes): value, stderr — vols in absolute units, cliquets in % of notional.
PLACEHOLDER_BASELINES: dict[str, dict[str, tuple[float, float]]] = {
    "LV (ω=0)": {
        "atm_vol": (0.215121, 0.000447),
        "vs_vol": (0.252111, 0.000196),
        "volswap_vol": (0.225172, 0.000121),
        "cliquet_1y": (1.19772, 0.00407),
        "cliquet_2y": (0.74375, 0.00394),
    },
    "1F ω=1": {
        "atm_vol": (0.208359, 0.000384),
        "vs_vol": (0.25237, 0.000211),
        "volswap_vol": (0.22315, 0.00012),
        "cliquet_1y": (1.31741, 0.00425),
        "cliquet_2y": (0.940793, 0.00452),
    },
    "1F ω=2": {
        "atm_vol": (0.197386, 0.000362),
        "vs_vol": (0.252391, 0.000263),
        "volswap_vol": (0.21411, 0.000128),
        "cliquet_1y": (1.62409, 0.00456),
        "cliquet_2y": (1.44226, 0.00553),
    },
    "1F ω=3": {
        "atm_vol": (0.184944, 0.000374),
        "vs_vol": (0.251793, 0.000335),
        "volswap_vol": (0.201011, 0.000152),
        "cliquet_1y": (1.96646, 0.00465),
        "cliquet_2y": (2.00505, 0.0061),
    },
    "2F Table 8.2": {
        "atm_vol": (0.192008, 0.000336),
        "vs_vol": (0.251856, 0.000268),
        "volswap_vol": (0.210247, 0.000132),
        "cliquet_1y": (1.73973, 0.00482),
        "cliquet_2y": (1.77974, 0.00612),
    },
}


def _tol(se: float, floor: float) -> float:
    return max(2.0 * se, floor)


def test_run_headline_smoke(forward_curve: ForwardCurve) -> None:
    """Fast: the study runner's table, smiles and markdown on a Black–Scholes model."""
    model = BlackScholes(0.2, forward_curve)
    sim = SimConfig(n_paths=20_000, dt_max=1.0 / 50.0, chunk_size=10_000, seed=3)
    result = run_headline({"bs": model}, sim, cliquet_maturities=(1.0,))
    row = result.table.iloc[0]
    assert row["model"] == "bs" and abs(row["atm_vol"] - 0.2) < 3.5 * row["atm_vol_stderr"]
    assert abs(row["vs_vol"] - 0.2) < 3.5 * row["vs_vol_stderr"] + 1e-6
    assert row["cliquet_1y_stderr"] > 0 and "cliquet_2y" not in result.table.columns
    assert set(result.smiles) == {"bs"} and len(result.wing) == 8  # 7 strikes + ATMF
    md = result.to_markdown()
    assert md.startswith("| model |") and "| k |" in md


@pytest.mark.slow
@pytest.mark.skipif(not STUDY_ARCHIVE.exists(), reason="study archive absent (SPEC §10 open item)")
def test_study_archive_numbers() -> None:
    """Requires ``study_archive/surface.yaml`` (market + ssvi sections) and
    ``study_archive/seeds.yaml`` (n_paths, seed); tolerance 2 stderr, else 0.02% of notional."""
    surface_file = STUDY_ARCHIVE / "surface.yaml"
    seeds_file = STUDY_ARCHIVE / "seeds.yaml"
    if not (surface_file.exists() and seeds_file.exists()):
        pytest.fail(
            "study_archive/ is present but surface.yaml / seeds.yaml are missing: add the study's "
            "SSVI parameters and simulation seeds, then wire them into this test"
        )
    pytest.fail("study archive found: port the study surface and seeds into this test (M4 TODO)")


@pytest.mark.slow
@pytest.mark.skipif(not PLACEHOLDER_BASELINES, reason="baselines not yet recorded")
def test_headline_regression_on_placeholder_surface() -> None:
    base = load_yaml(SPEC_1F, CalibrationSpec)
    spec_2f = load_yaml(SPEC_2F, CalibrationSpec)
    models, _ = headline_models(LeverageCache(ROOT / "cache"), base, spec_2f)
    sim = dataclasses.replace(base.sim, n_paths=HEADLINE_N_PATHS, seed=HEADLINE_SEED)
    result = run_headline(models, sim)
    table = result.table.set_index("model")
    for name, baseline in PLACEHOLDER_BASELINES.items():
        row = table.loc[name]
        for key, (value, _se_base) in baseline.items():
            new, se = float(row[key]), float(row[key + "_stderr"])
            floor = 0.02 if key.startswith("cliquet") else 0.0002  # % notional / vol points
            assert abs(new - value) <= _tol(se, floor), (name, key, new, se, value)
    # the study's qualitative findings on this surface
    names = [LV_NAME] + [one_factor_name(w) for w in HEADLINE_OMEGAS]
    atm = table.loc[names, "atm_vol"].to_numpy()
    assert np.all(np.diff(atm) < 0), atm  # forward ATM vol falls with ω
    cliq = table.loc[names, "cliquet_1y"].to_numpy()
    assert np.all(np.diff(cliq) > 0), cliq  # the capped cliquet gains value with ω
    assert TWO_FACTOR_NAME in table.index
    # SPEC §10 identity: forward ATM vol < forward vol-swap vol < forward variance-swap vol for
    # every calibrated LSV model (ρ < 0), well beyond the noise
    for name in [*names[1:], TWO_FACTOR_NAME]:
        r = table.loc[name]
        gap = 2.0 * float(np.hypot(r["atm_vol_stderr"], r["volswap_vol_stderr"]))
        assert r["atm_vol"] < r["volswap_vol"] - gap < r["vs_vol"], (name, r.to_dict())
