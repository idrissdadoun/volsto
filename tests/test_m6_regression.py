"""M6 headline regression (SPEC §6.8, owner decision at the M6 review): the 3y annual autocall and
the 3y Phoenix (``volsto.studies.m6.headline_products``) under LV, the 1F LSV at ω = 1, 2, 3 and the
2F Table 8.2 set, priced by ``run_m6_headline_seeds`` and compared cell by cell with the baseline
recorded here.  Baseline convention: 800000 particles per calibration (single particle seed,
the specs' 12345), 400000 pricing paths averaged over the pricing seeds [2024, 2025, 2026, 2027, 2028, 2029] (the stderr is the
standard error of that mean across seeds), code tag "m6" (corrected second-order SV step);
tolerance ``max(2 stderr, 0.02% of notional)`` for prices and legs (fractions of notional), the
same absolute floor 0.0002 for probabilities and 0.002 y for the expected life.  Rerunning costs
about six times the headline (six pricing seeds): slow.

Baseline provenance: git commit b89a35c; model configs configs/studies/lsv_reference_1f.yaml
(1F: {'nu': 1.5, 'theta': 0.0, 'k1': 1.5, 'k2': 1.5, 'rho12': 0.0, 'rho_SX1': -0.7, 'rho_SX2': 0.0}; the ω = 1, 2 rows replace ν by ω/2 with κ = 1.5, ρ = −0.7 as in
volsto.studies.m4.one_factor_variants; ω = 0 is the pure Dupire local vol of the same surface)
and configs/studies/lsv_reference_2f.yaml (2F Table 8.2: {'nu': 1.74, 'theta': 0.245, 'k1': 5.35, 'k2': 0.28, 'rho12': 0.0, 'rho_SX1': -0.759, 'rho_SX2': -0.487}); the reference SSVI surface
configs/surfaces/reference_ssvi.yaml; the recompute uses the same six pricing seeds, so the
per-cell comparison is paired (common random numbers per seed) and measures a real change, not
seed noise.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from volsto.calibration import LeverageCache
from volsto.config import CalibrationSpec, load_yaml
from volsto.studies.m4 import LV_NAME, headline_models
from volsto.studies.m6 import run_m6_headline_seeds

ROOT = Path(__file__).resolve().parents[1]
SPEC_1F = ROOT / "configs" / "studies" / "lsv_reference_1f.yaml"
SPEC_2F = ROOT / "configs" / "studies" / "lsv_reference_2f.yaml"
N_PARTICLES = 800000
N_PATHS = 400000
PRICING_SEEDS = [2024, 2025, 2026, 2027, 2028, 2029]

M6_BASELINES: dict[tuple[str, str], dict[str, tuple[float, float]]] = {
    ("LV (ω=0)", "autocall 3y"): {
        "price": (0.958175, 0.00015),
        "expected_life": (1.65483, 0.000249),
        "p_ki": (0.111262, 0.000224),
        "p_breach": (0.111262, 0.000224),
        "p_no_autocall": (0.220669, 0.000188),
        "p_autocall_1": (0.615744, 0.000161),
        "p_autocall_2": (0.113686, 0.000171),
        "p_autocall_3": (0.0499004, 7.48e-05),
        "leg:autocall_1": (0.639765, 0.000167),
        "leg:autocall_2": (0.122336, 0.000184),
        "leg:autocall_3": (0.0554534, 8.31e-05),
        "leg:bond": (0.207818, 0.000177),
        "leg:put_digital": (-0.0419129, 8.44e-05),
        "leg:put_vanilla": (-0.0252846, 6.44e-05),
    },
    ("LV (ω=0)", "phoenix 3y"): {
        "price": (0.964451, 0.000135),
        "expected_life": (1.65483, 0.000249),
        "p_ki": (0.180321, 0.000214),
        "p_breach": (0.209864, 0.000198),
        "p_no_autocall": (0.220669, 0.000188),
        "p_autocall_1": (0.615744, 0.000161),
        "p_autocall_2": (0.113686, 0.000171),
        "p_autocall_3": (0.0499004, 7.48e-05),
        "leg:autocall_1": (0.603552, 0.000158),
        "leg:autocall_2": (0.109229, 0.000165),
        "leg:autocall_3": (0.0469944, 7.04e-05),
        "leg:bond": (0.207818, 0.000177),
        "leg:coupon": (0.0791677, 1.52e-05),
        "leg:put": (-0.08231, 0.000119),
        "leg:put_european": (-0.0671975, 0.000145),
    },
    ("1F ω=1", "autocall 3y"): {
        "price": (0.961417, 0.000182),
        "expected_life": (1.65769, 0.00028),
        "p_ki": (0.105439, 0.00029),
        "p_breach": (0.105439, 0.00029),
        "p_no_autocall": (0.225308, 0.00015),
        "p_autocall_1": (0.615781, 0.000135),
        "p_autocall_2": (0.110746, 7.27e-05),
        "p_autocall_3": (0.048165, 7.05e-05),
        "leg:autocall_1": (0.639803, 0.00014),
        "leg:autocall_2": (0.119172, 7.83e-05),
        "leg:autocall_3": (0.0535249, 7.83e-05),
        "leg:bond": (0.212187, 0.000141),
        "leg:put_digital": (-0.0397194, 0.000109),
        "leg:put_vanilla": (-0.0235502, 7.58e-05),
    },
    ("1F ω=1", "phoenix 3y"): {
        "price": (0.967319, 0.000157),
        "expected_life": (1.65769, 0.00028),
        "p_ki": (0.181665, 0.000125),
        "p_breach": (0.205513, 0.000149),
        "p_no_autocall": (0.225308, 0.00015),
        "p_autocall_1": (0.615781, 0.000135),
        "p_autocall_2": (0.110746, 7.27e-05),
        "p_autocall_3": (0.048165, 7.05e-05),
        "leg:autocall_1": (0.603588, 0.000132),
        "leg:autocall_2": (0.106403, 6.99e-05),
        "leg:autocall_3": (0.0453601, 6.64e-05),
        "leg:bond": (0.212187, 0.000141),
        "leg:coupon": (0.0796137, 4.42e-05),
        "leg:put": (-0.0798329, 0.000127),
        "leg:put_european": (-0.0632696, 0.000181),
    },
    ("1F ω=2", "autocall 3y"): {
        "price": (0.964104, 0.000126),
        "expected_life": (1.66359, 0.000263),
        "p_ki": (0.100415, 0.000206),
        "p_breach": (0.100415, 0.000206),
        "p_no_autocall": (0.232917, 0.000166),
        "p_autocall_1": (0.615582, 0.000164),
        "p_autocall_2": (0.105247, 0.000154),
        "p_autocall_3": (0.0462542, 0.000116),
        "leg:autocall_1": (0.639597, 0.000171),
        "leg:autocall_2": (0.113254, 0.000166),
        "leg:autocall_3": (0.0514014, 0.000129),
        "leg:bond": (0.219353, 0.000157),
        "leg:put_digital": (-0.0378268, 7.74e-05),
        "leg:put_vanilla": (-0.0216743, 6.9e-05),
    },
    ("1F ω=2", "phoenix 3y"): {
        "price": (0.970603, 0.000129),
        "expected_life": (1.66359, 0.000263),
        "p_ki": (0.179154, 0.000128),
        "p_breach": (0.19819, 0.000162),
        "p_no_autocall": (0.232917, 0.000166),
        "p_autocall_1": (0.615582, 0.000164),
        "p_autocall_2": (0.105247, 0.000154),
        "p_autocall_3": (0.0462542, 0.000116),
        "leg:autocall_1": (0.603393, 0.000161),
        "leg:autocall_2": (0.10112, 0.000148),
        "leg:autocall_3": (0.0435605, 0.00011),
        "leg:bond": (0.219353, 0.000157),
        "leg:coupon": (0.0800702, 3.46e-05),
        "leg:put": (-0.076893, 0.000109),
        "leg:put_european": (-0.059501, 0.000139),
    },
    ("1F ω=3", "autocall 3y"): {
        "price": (0.965842, 8.3e-05),
        "expected_life": (1.67083, 0.000387),
        "p_ki": (0.0965087, 0.000121),
        "p_breach": (0.0965087, 0.000121),
        "p_no_autocall": (0.241778, 0.000135),
        "p_autocall_1": (0.615457, 0.000192),
        "p_autocall_2": (0.0982504, 0.000125),
        "p_autocall_3": (0.0445137, 0.00016),
        "leg:autocall_1": (0.639467, 0.0002),
        "leg:autocall_2": (0.105726, 0.000134),
        "leg:autocall_3": (0.0494673, 0.000178),
        "leg:bond": (0.227698, 0.000127),
        "leg:put_digital": (-0.0363554, 4.56e-05),
        "leg:put_vanilla": (-0.0201609, 5.66e-05),
    },
    ("1F ω=3", "phoenix 3y"): {
        "price": (0.973701, 9.56e-05),
        "expected_life": (1.67083, 0.000387),
        "p_ki": (0.173682, 7.54e-05),
        "p_breach": (0.189684, 0.000174),
        "p_no_autocall": (0.241778, 0.000135),
        "p_autocall_1": (0.615457, 0.000192),
        "p_autocall_2": (0.0982504, 0.000125),
        "p_autocall_3": (0.0445137, 0.00016),
        "leg:autocall_1": (0.603271, 0.000188),
        "leg:autocall_2": (0.094398, 0.00012),
        "leg:autocall_3": (0.0419215, 0.000151),
        "leg:bond": (0.227698, 0.000127),
        "leg:coupon": (0.0804935, 2.58e-05),
        "leg:put": (-0.0740804, 8.55e-05),
        "leg:put_european": (-0.0565163, 9.07e-05),
    },
    ("2F Table 8.2", "autocall 3y"): {
        "price": (0.965157, 0.000105),
        "expected_life": (1.66621, 0.000542),
        "p_ki": (0.0981354, 0.000138),
        "p_breach": (0.0981354, 0.000138),
        "p_no_autocall": (0.236064, 0.000287),
        "p_autocall_1": (0.615493, 0.000301),
        "p_autocall_2": (0.102805, 0.000234),
        "p_autocall_3": (0.0456383, 0.000109),
        "leg:autocall_1": (0.639504, 0.000313),
        "leg:autocall_2": (0.110626, 0.000252),
        "leg:autocall_3": (0.0507171, 0.000121),
        "leg:bond": (0.222317, 0.000271),
        "leg:put_digital": (-0.0369682, 5.21e-05),
        "leg:put_vanilla": (-0.021039, 5.15e-05),
    },
    ("2F Table 8.2", "phoenix 3y"): {
        "price": (0.971934, 0.0001),
        "expected_life": (1.66621, 0.000542),
        "p_ki": (0.177934, 0.00019),
        "p_breach": (0.19403, 0.000218),
        "p_no_autocall": (0.236064, 0.000287),
        "p_autocall_1": (0.615493, 0.000301),
        "p_autocall_2": (0.102805, 0.000234),
        "p_autocall_3": (0.0456383, 0.000109),
        "leg:autocall_1": (0.603305, 0.000295),
        "leg:autocall_2": (0.0987736, 0.000225),
        "leg:autocall_3": (0.0429806, 0.000102),
        "leg:bond": (0.222317, 0.000271),
        "leg:coupon": (0.080377, 4.2e-05),
        "leg:put": (-0.0758194, 7.65e-05),
        "leg:put_european": (-0.0580072, 9.76e-05),
    },
}


def _tol(key: str, se: float) -> float:
    # 0.02% of notional (fractions) / 0.0002 in probability; 0.002 y for the expected life
    floor = 0.002 if key == "expected_life" else 0.0002
    return max(2.0 * se, floor)


@pytest.mark.slow
def test_m6_headline_regression() -> None:
    base = load_yaml(SPEC_1F, CalibrationSpec)
    spec_2f = load_yaml(SPEC_2F, CalibrationSpec)
    models, _ = headline_models(
        LeverageCache(ROOT / "cache"), base, spec_2f, n_particles=N_PARTICLES
    )
    sim = dataclasses.replace(base.sim, n_paths=N_PATHS, seed=PRICING_SEEDS[0])
    result = run_m6_headline_seeds(models, sim, PRICING_SEEDS, reference=LV_NAME)
    table = result.table.set_index(["model", "product"])
    failures = []
    for (model, product), cells in M6_BASELINES.items():
        row = table.loc[(model, product)]
        for key, (value, _se_base) in cells.items():
            new, se = float(row[key]), float(row[key + "_stderr"])
            if abs(new - value) > _tol(key, se):
                failures.append((model, product, key, new, se, value))
    assert not failures, failures
    # the qualitative reading of SPEC §6.8: the LSV notes are worth more than under LV and the
    # knock-in probability falls with the vol of vol
    for product in ("autocall 3y", "phoenix 3y"):
        prices = [table.loc[(m, product), "price"] for m in [LV_NAME, "1F ω=1", "1F ω=2", "1F ω=3"]]
        assert np.all(np.diff(prices) > 0), (product, prices)
