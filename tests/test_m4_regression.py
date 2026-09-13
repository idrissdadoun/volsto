"""M4 regression tests (SPEC §10, owner request at M3.2/M3b acceptance).

Part A — the original study's numbers (its SSVI surface and seeds) can only be checked against
``study_archive/``; those tests are skipped until the archive is present.

Part B — the same headline set measured on the **placeholder** reference surface at M4 with
:func:`volsto.studies.m4.run_headline` (400k paths, seed 2024, the default schedule and scheme
shared with calibration; baselines re-recorded at M4b).  Tolerance ``max(2 stderr, 0.02% of notional)`` for prices and
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
# Production convention (owner decision at M4b acceptance): baselines are recorded with 8e5
# particles per calibration, single seed; development paths keep 2e5.
HEADLINE_N_PARTICLES = 800_000

# SPEC §10 numbers recalled from the original study (its own surface and seeds): 1y-into-1y ATM
# forward vol, forward variance-swap level and the 1y / 2y capped cliquets for ω = 0/1/2/3.
STUDY_NUMBERS = {
    "atm_vol": {0.0: 0.215, 3.0: 0.186},
    "vs_vol_range": (0.252, 0.255),
    "cliquet_1y": {0.0: 1.082, 1.0: 1.194, 2.0: 1.472, 3.0: 1.763},
    "cliquet_2y": {0.0: 0.639, 3.0: 1.718},
}

# Placeholder-surface baselines recorded at M4c with the production convention — 8e5 particles per
# calibration, single seed (HEADLINE_N_PARTICLES), 400k pricing paths, seed 2024, second-order SV
# step, schedule 1/1460–1/365–1/250, adaptive regression grid (scripts/m4_headline.py
# --n-particles 800000; SPEC §6.3–6.4).  Units: vols and fair strikes in absolute vol units,
# cliquets and the VKO price in % of notional, ratios/probabilities as such; the VKO barrier
# sweep ratios vko_ratio_{20..40} are the §6.2 measured outputs (no sign claim).  Versus the 2e5 M4b
# record every legacy entry moved by at most 1.2 stderr.
PLACEHOLDER_BASELINES: dict[str, dict[str, tuple[float, float]]] = {
    "LV (ω=0)": {
        "atm_vol": (0.214576, 0.000446),
        "vs_vol": (0.252529, 0.000197),
        "volswap_vol": (0.225388, 0.000121),
        "cliquet_1y": (1.19878, 0.00408),
        "cliquet_2y": (0.748585, 0.00395),
        "upvar_100": (0.150365, 2.64e-05),
        "downvar_100": (0.349239, 0.000279),
        "kovar_110": (0.29066, 0.000257),
        "kovar_110_p_ko": (0.657295, 0.00052),
        "vko_30": (2.43677, 0.00915),
        "vko_30_p_ko": (0.174662, 0.0006),
        "vko_ratio_20": (0.055343, 0.000458),
        "vko_ratio_25": (0.175671, 0.000962),
        "vko_ratio_30": (0.324817, 0.0014),
        "vko_ratio_35": (0.475881, 0.00168),
        "vko_ratio_40": (0.613968, 0.00181),
    },
    "1F ω=1": {
        "atm_vol": (0.208179, 0.000383),
        "vs_vol": (0.252462, 0.00021),
        "volswap_vol": (0.223237, 0.000119),
        "cliquet_1y": (1.32081, 0.00426),
        "cliquet_2y": (0.953597, 0.00454),
        "upvar_100": (0.150464, 3.54e-05),
        "downvar_100": (0.349862, 0.000288),
        "kovar_110": (0.286518, 0.000258),
        "kovar_110_p_ko": (0.645005, 0.000511),
        "vko_30": (2.32991, 0.00884),
        "vko_30_p_ko": (0.178475, 0.000605),
        "vko_ratio_20": (0.0529788, 0.000449),
        "vko_ratio_25": (0.163882, 0.000919),
        "vko_ratio_30": (0.310134, 0.00136),
        "vko_ratio_35": (0.466046, 0.00167),
        "vko_ratio_40": (0.60826, 0.00181),
    },
    "1F ω=2": {
        "atm_vol": (0.197029, 0.000357),
        "vs_vol": (0.251775, 0.000255),
        "volswap_vol": (0.213857, 0.000126),
        "cliquet_1y": (1.62934, 0.00457),
        "cliquet_2y": (1.46409, 0.00556),
        "upvar_100": (0.150565, 6.21e-05),
        "downvar_100": (0.348766, 0.000317),
        "kovar_110": (0.276623, 0.000264),
        "kovar_110_p_ko": (0.623413, 0.000507),
        "vko_30": (2.43056, 0.00916),
        "vko_30_p_ko": (0.183965, 0.000613),
        "vko_ratio_20": (0.0668763, 0.000518),
        "vko_ratio_25": (0.178308, 0.00097),
        "vko_ratio_30": (0.323758, 0.00139),
        "vko_ratio_35": (0.473652, 0.00168),
        "vko_ratio_40": (0.612228, 0.00181),
    },
    "1F ω=3": {
        "atm_vol": (0.184497, 0.000367),
        "vs_vol": (0.251221, 0.000317),
        "volswap_vol": (0.20071, 0.00015),
        "cliquet_1y": (1.97127, 0.00466),
        "cliquet_2y": (2.0333, 0.00612),
        "upvar_100": (0.150664, 9.74e-05),
        "downvar_100": (0.347464, 0.000356),
        "kovar_110": (0.264509, 0.000268),
        "kovar_110_p_ko": (0.59421, 0.000534),
        "vko_30": (2.5702, 0.00961),
        "vko_30_p_ko": (0.18799, 0.000618),
        "vko_ratio_20": (0.0882828, 0.000613),
        "vko_ratio_25": (0.203183, 0.00105),
        "vko_ratio_30": (0.342678, 0.00144),
        "vko_ratio_35": (0.484613, 0.0017),
        "vko_ratio_40": (0.613533, 0.0018),
    },
    "2F Table 8.2": {
        "atm_vol": (0.192215, 0.000333),
        "vs_vol": (0.251384, 0.000264),
        "volswap_vol": (0.209819, 0.000131),
        "cliquet_1y": (1.74981, 0.00483),
        "cliquet_2y": (1.81152, 0.00616),
        "upvar_100": (0.150884, 5.49e-05),
        "downvar_100": (0.349131, 0.000284),
        "kovar_110": (0.274175, 0.00024),
        "kovar_110_p_ko": (0.617415, 0.000517),
        "vko_30": (2.09821, 0.0082),
        "vko_30_p_ko": (0.18169, 0.00061),
        "vko_ratio_20": (0.0363604, 0.000347),
        "vko_ratio_25": (0.13183, 0.000786),
        "vko_ratio_30": (0.28033, 0.00127),
        "vko_ratio_35": (0.448196, 0.00164),
        "vko_ratio_40": (0.605087, 0.00182),
    },
}


def _tol(se: float, floor: float) -> float:
    return max(2.0 * se, floor)


def test_run_headline_smoke(forward_curve: ForwardCurve) -> None:
    """Fast: the study runner's table, smiles and markdown on a Black–Scholes model."""
    model = BlackScholes(0.2, forward_curve)
    sim = SimConfig(n_paths=20_000, dt_max=1.0 / 50.0, chunk_size=10_000, seed=3)
    # barriers around sigma so that both knock-out outcomes occur in the sample
    result = run_headline(
        {"bs": model}, sim, cliquet_maturities=(1.0,), vko_barriers=(0.19, 0.205, 0.30, 0.40)
    )
    row = result.table.iloc[0]
    assert row["model"] == "bs" and abs(row["atm_vol"] - 0.2) < 3.5 * row["atm_vol_stderr"]
    assert abs(row["vs_vol"] - 0.2) < 3.5 * row["vs_vol_stderr"] + 1e-6
    assert row["cliquet_1y_stderr"] > 0 and "cliquet_2y" not in result.table.columns
    assert set(result.smiles) == {"bs"} and len(result.wing) == 8  # 7 strikes + ATMF
    # M4c columns: flat BS -> conditional and KO fair vols equal sigma; VKO discount in (0, 1)
    for key in ("upvar_100", "downvar_100", "kovar_110"):
        assert abs(row[key] - 0.2) < 3.5 * row[key + "_stderr"], (key, row[key])
    assert 0.0 < row["vko_ratio_20"] < row["vko_ratio_40"] <= 1.0 and 0.0 < row["p_itm"] < 1.0
    assert row["vko_30"] > 0 and row["vko_30_stderr"] > 0 and row["vko_30_p_ko"] == 0.0
    assert row["itm_rv_p10"] < row["itm_rv_p50"] < row["itm_rv_p90"]
    md = result.to_markdown()
    assert md.startswith("| model |") and "| k |" in md and "VKO/vanilla @30%" in md


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
    models, _ = headline_models(
        LeverageCache(ROOT / "cache"), base, spec_2f, n_particles=HEADLINE_N_PARTICLES
    )
    sim = dataclasses.replace(base.sim, n_paths=HEADLINE_N_PATHS, seed=HEADLINE_SEED)
    result = run_headline(models, sim)
    table = result.table.set_index("model")
    for name, baseline in PLACEHOLDER_BASELINES.items():
        row = table.loc[name]
        for key, (value, _se_base) in baseline.items():
            new, se = float(row[key]), float(row[key + "_stderr"])
            if key.startswith("cliquet") or key == "vko_30":
                floor = 0.02  # % of notional
            elif key.endswith("_p_ko") or key.startswith("vko_ratio_"):
                floor = 0.002  # ratio / probability
            else:
                floor = 0.0002  # vol points (absolute vol units)
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
