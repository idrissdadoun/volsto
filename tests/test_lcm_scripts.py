"""The scripts of the local correlation study (SPEC §8.7, M12 part LC7) after the owner's
decisions of 2026-10-09: the columns ``scripts/lcm_price.py`` derives from a row (the
listed-variance forward, LC and CC over the copula, the flags), its status rule (the names'
2 % check is a diagnostic, not a gate), the flags on the sweep's line (``scripts/disp_lcm.py``)
and the steps of the driver (``scripts/lcm_sweep.sh``).

Fast: no model is built and no data of the study is read.  The sweep's order, the configuration
and the delta method are checked in ``tests/test_local_correlation.py``.
"""

from __future__ import annotations

import json
import math
import pickle
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def lcm_scripts() -> tuple[Any, Any]:
    """``scripts/lcm_price.py`` and ``scripts/disp_lcm.py`` as modules."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import disp_lcm
    import lcm_price

    return lcm_price, disp_lcm


def hand_row(**changes: Any) -> dict[str, Any]:
    """A row with round numbers: ``√(EQV/EV) = 0.14/0.15``, ``κ_cop = P_D/√EV = 0.11/0.15``."""
    row: dict[str, Any] = {
        "date": "2020-01-03", "tenor": "3m", "status": "ok", "reason": "",
        "EQV": 0.0196, "EV_copula": 0.0225, "P_D_copula": 0.11,
        "ED_lc": 0.1045, "ED_lc_se": 0.00011, "ED_cc": 0.1089, "ED_cc_se": 0.00022,
        "n_names_unscreened": 0, "names_unscreened": "",
        "clip_low_inner_max": 0.0, "clip_high_inner_max": 0.0,
    }  # fmt: skip
    row.update(changes)
    return row


def test_derived_columns() -> None:
    """``lcm_price.derived_columns`` on a hand-made row (owner's checks (b) and (c) of
    2026-10-09): ``listed_fwd_ratio = √(EQV/EV_copula)`` and ``listed_fwd = P_D·√(EQV/EV)``,
    which is the copula's ``κ = P_D/√EV`` times ``√EQV``; ``lc_over_copula = ED_lc/P_D`` and
    ``cc_over_copula = ED_cc/P_D`` with the errors of the numerators over ``P_D``; the keys are
    ``DERIVED_COLUMNS``, the row is left as it was and the result goes through JSON."""
    lp, _ = lcm_scripts()
    row = hand_row()
    before = dict(row)
    out = lp.derived_columns(row)
    assert row == before and tuple(out) == lp.DERIVED_COLUMNS
    assert out["listed_fwd_ratio"] == pytest.approx(0.14 / 0.15, rel=1e-14)
    kappa_cop = row["P_D_copula"] / math.sqrt(row["EV_copula"])
    assert out["listed_fwd"] == pytest.approx(kappa_cop * math.sqrt(row["EQV"]), rel=1e-14)
    assert out["listed_fwd"] == pytest.approx(0.11 * 0.14 / 0.15, rel=1e-14)
    assert out["listed_fwd"] / row["P_D_copula"] == pytest.approx(out["listed_fwd_ratio"])
    assert out["lc_over_copula"] == pytest.approx(0.95, rel=1e-14)
    assert out["lc_over_copula_se"] == pytest.approx(0.001, rel=1e-14)
    assert out["cc_over_copula"] == pytest.approx(0.99, rel=1e-14)
    assert out["cc_over_copula_se"] == pytest.approx(0.002, rel=1e-14)
    flags = ("flag_unscreened", "flag_clip_low", "flag_clip_high", "flag_clip", "indicative")
    assert all(out[k] is False for k in flags)
    assert json.loads(json.dumps(out)) == out
    print("\nderived columns of the hand-made row")
    for k, v in out.items():
        print(f"  {k:20s} {v}")


def test_derived_flags_at_their_thresholds() -> None:
    """The flags (owner's decisions 2 and 5 of 2026-10-09): ``flag_unscreened`` when a name is
    kept unscreened; ``flag_clip_low`` / ``flag_clip_high`` when the clipped mass inside ±2.5 sd
    at ``λ = 0``, resp. at the cap, exceeds 1 % — exactly 1 % is not flagged, the next float
    is — and ``flag_clip`` when either is; ``indicative`` for the 24m tenor only.  The flags are
    Python ``bool`` also when the row holds numpy numbers."""
    lp, _ = lcm_scripts()
    assert lp.CLIP_FLAG_MASS == 0.01 and lp.INDICATIVE_TENORS == ("24m",)
    above = float(np.nextafter(0.01, 1.0))
    below = float(np.nextafter(0.01, 0.0))
    cases = [  # (low, high) -> (flag_clip_low, flag_clip_high, flag_clip)
        ((0.0, 0.0), (False, False, False)),
        ((0.01, 0.01), (False, False, False)),
        ((below, below), (False, False, False)),
        ((above, 0.01), (True, False, True)),
        ((0.01, above), (False, True, True)),
        ((0.25, 0.37), (True, True, True)),
    ]
    print("\nclip_low_inner_max  clip_high_inner_max  flag_clip_low  flag_clip_high  flag_clip")
    for (low, high), expected in cases:
        out = lp.derived_columns(hand_row(clip_low_inner_max=low, clip_high_inner_max=high))
        got = (out["flag_clip_low"], out["flag_clip_high"], out["flag_clip"])
        print(f"  {low!r:18}  {high!r:19}  {got[0]!s:13}  {got[1]!s:14}  {got[2]}")
        assert got == expected, (low, high)
    for n, expected_flag in ((0, False), (1, True), (3, True)):
        out = lp.derived_columns(hand_row(n_names_unscreened=n))
        assert out["flag_unscreened"] is expected_flag, n
        assert out["flag_clip"] is False  # the flags are independent
    for tenor, expected_flag in (("3m", False), ("12m", False), ("24m", True)):
        assert lp.derived_columns(hand_row(tenor=tenor))["indicative"] is expected_flag, tenor
    out = lp.derived_columns(
        hand_row(
            n_names_unscreened=np.int64(2),
            clip_low_inner_max=np.float64(0.02),
            clip_high_inner_max=np.float64(0.005),
        )
    )
    for k in ("flag_unscreened", "flag_clip_low", "flag_clip_high", "flag_clip", "indicative"):
        assert type(out[k]) is bool, k
    assert (out["flag_unscreened"], out["flag_clip_low"], out["flag_clip_high"]) == (
        True,
        True,
        False,
    )


def test_derived_columns_of_incomplete_rows() -> None:
    """A row of an earlier pass or a failed row (no ``n_names_unscreened``, no prices, ``None``
    or NaN where a table had no value): every key of ``DERIVED_COLUMNS`` is there, a quantity
    whose inputs are absent or not finite is NaN, a flag is ``False`` — never an exception.
    The listed-variance forward needs the entry's three numbers only."""
    lp, _ = lcm_scripts()
    failed = {"date": "2023-07-03", "tenor": "3m", "status": "failed", "reason": "ValueError: x"}
    out = lp.derived_columns(failed)
    assert tuple(out) == lp.DERIVED_COLUMNS
    for k in lp.DERIVED_COLUMNS:
        if k.startswith("flag_") or k == "indicative":
            assert out[k] is False, k
        else:
            assert math.isnan(out[k]), k
    entry_only = {"tenor": "24m", "EQV": 0.0196, "EV_copula": 0.0225, "P_D_copula": 0.11}
    out = lp.derived_columns(entry_only)
    assert out["listed_fwd_ratio"] == pytest.approx(0.14 / 0.15, rel=1e-14)
    assert out["listed_fwd"] == pytest.approx(0.11 * 0.14 / 0.15, rel=1e-14)
    assert math.isnan(out["lc_over_copula"]) and math.isnan(out["cc_over_copula_se"])
    assert out["indicative"] is True and out["flag_clip"] is False
    nan = float("nan")
    for changes in (
        {"EV_copula": 0.0},
        {"EV_copula": -0.01},
        {"EQV": -0.01},
        {"EQV": nan},
        {"EV_copula": None},
        {"EQV": "n/a"},
    ):
        out = lp.derived_columns(hand_row(**changes))
        assert math.isnan(out["listed_fwd_ratio"]) and math.isnan(out["listed_fwd"]), changes
        assert out["lc_over_copula"] == pytest.approx(0.95)  # P_D is there: the ratios stay
    for bad in (0.0, nan, None):
        out = lp.derived_columns(hand_row(P_D_copula=bad))
        assert out["listed_fwd_ratio"] == pytest.approx(0.14 / 0.15)  # it does not read P_D
        assert all(
            math.isnan(out[k])
            for k in ("listed_fwd", "lc_over_copula", "lc_over_copula_se", "cc_over_copula")
        ), bad
    out = lp.derived_columns(
        hand_row(clip_low_inner_max=nan, clip_high_inner_max=None, n_names_unscreened=nan)
    )
    assert (out["flag_clip_low"], out["flag_clip_high"], out["flag_clip"]) == (False, False, False)
    assert out["flag_unscreened"] is False


def test_row_status() -> None:
    """``lcm_price.row_status`` (owner's decision 3 of 2026-10-09: the names' 2 % check "stays
    a reported diagnostic, not a gate"): a row whose only failing check is ``check_names`` is
    ``ok`` with no reason; ``check`` remains for the no-NaN, forward and index checks, and the
    reason names the gates that fail, in their order, never ``check_names``."""
    lp, _ = lcm_scripts()
    assert lp.GATING_CHECKS == ("check_no_nan", "check_forward", "check_index")
    passed = {"check_no_nan": True, "check_forward": True, "check_index": True, "check_names": True}  # fmt: skip
    assert lp.row_status(passed) == ("ok", "")
    assert lp.row_status({**passed, "check_names": False}) == ("ok", "")
    print("\nfailing checks -> status, reason")
    for gate in lp.GATING_CHECKS:
        for names in (True, False):
            status = lp.row_status({**passed, gate: False, "check_names": names})
            print(f"  {gate}{'' if names else ' + check_names'} -> {status}")
            assert status == ("check", f"sanity checks: {gate}")
    every = dict.fromkeys(passed, False)
    assert lp.row_status(every) == (
        "check",
        "sanity checks: check_no_nan, check_forward, check_index",
    )
    two = {**passed, "check_no_nan": False, "check_index": False}
    assert lp.row_status(two) == ("check", "sanity checks: check_no_nan, check_index")
    # numpy booleans (a row read back from a table) count as booleans
    assert lp.row_status({k: np.bool_(v) for k, v in two.items()})[0] == "check"
    assert lp.row_status({k: np.bool_(True) for k in passed}) == ("ok", "")
    # the gates are required: a row without them is not given a status here
    with pytest.raises(KeyError):
        lp.row_status({"check_names": True})


def test_failed_columns(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``lcm_price.failed_columns`` — what a failed row carries beside its reason: the
    listed-variance forward of the study's entry (basket B1: ``√(EQV/EV)`` and ``P_D`` times
    it), which needs nothing of the model, the model's ratios NaN and the flags ``False``; a
    date without an entry gives NaN and no exception.  The entry is read where the study keeps
    it (``<study outputs>/entries/<tenor>/<date>.pkl``), here a folder of the test."""
    lp, _ = lcm_scripts()
    monkeypatch.setattr(lp.dd, "OUT", tmp_path)
    folder = tmp_path / "entries" / "24m"
    folder.mkdir(parents=True)
    entry = {"B1": {"EQV": 0.0196, "EV": 0.0225, "P_D": 0.11}, "B2": {"EQV": 1.0, "EV": 1.0}}
    with (folder / "2020-01-03.pkl").open("wb") as fh:
        pickle.dump(entry, fh)
    out = lp.failed_columns("2020-01-03", "24m")
    assert tuple(out) == lp.DERIVED_COLUMNS
    assert out["listed_fwd_ratio"] == pytest.approx(0.14 / 0.15, rel=1e-14)
    assert out["listed_fwd"] == pytest.approx(0.11 * 0.14 / 0.15, rel=1e-14)
    assert math.isnan(out["lc_over_copula"]) and math.isnan(out["cc_over_copula"])
    assert out["indicative"] is True
    assert not (out["flag_unscreened"] or out["flag_clip"])
    missing = lp.failed_columns("2020-01-06", "24m")
    assert math.isnan(missing["listed_fwd_ratio"]) and math.isnan(missing["listed_fwd"])
    other = lp.failed_columns("2020-01-03", "3m")  # the entry of another tenor is not read
    assert math.isnan(other["listed_fwd_ratio"]) and other["indicative"] is False
    json.dumps(out)  # the failed row is written as JSON


def test_sweep_line_names_the_flags() -> None:
    """``disp_lcm.flags_text``: the pass's one line per date names ``flag_unscreened`` (with the
    names), ``flag_clip`` (with the clipped mass at ``λ = 0`` and at the cap) and ``indicative``
    when they are set, and nothing when none is or when the row has no such column (a row of an
    earlier pass)."""
    lp, sweep = lcm_scripts()
    row = hand_row()
    assert sweep.flags_text({**row, **lp.derived_columns(row)}) == ""
    assert sweep.flags_text({"date": "2020-01-03", "status": "ok"}) == ""
    row = hand_row(n_names_unscreened=1, names_unscreened="TRV")
    assert sweep.flags_text({**row, **lp.derived_columns(row)}) == "flag_unscreened (TRV)"
    row = hand_row(clip_low_inner_max=0.0, clip_high_inner_max=0.01896)
    assert sweep.flags_text({**row, **lp.derived_columns(row)}) == (
        "flag_clip (low 0.0000, high 0.0190)"
    )
    row = hand_row(
        tenor="24m",
        n_names_unscreened=2,
        names_unscreened="TRV,DOW",
        clip_low_inner_max=0.0312,
        clip_high_inner_max=0.2,
    )
    text = sweep.flags_text({**row, **lp.derived_columns(row)})
    print("\n" + text)
    assert text == "flag_unscreened (TRV,DOW), flag_clip (low 0.0312, high 0.2000), indicative"


def test_sweep_driver_steps() -> None:
    """``scripts/lcm_sweep.sh`` holds the steps of the owner's order of 2026-10-09 and no other:
    ``P`` (3m production pass on the default configuration, deltas, the variance swap, the
    report), ``F`` (today's full risk at production budget, to
    ``risk_full_2026-10-02_3m.json``), ``T`` (today alone at 12m and at 24m, development budget,
    no risk), ``Y`` (12m development pass, monthly dates), ``Z`` (24m, development budget, the
    reference dates only), ``N`` (3m development pass on ``lcm_norepair.yaml``, tag
    ``norepair``; optional, not a default step).  The script parses (``sh -n``)."""
    script = SCRIPTS / "lcm_sweep.sh"
    done = subprocess.run(["sh", "-n", str(script)], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    text = script.read_text()
    labels = re.findall(r"^\s{4}(\S+)\) ", text, flags=re.MULTILINE)
    assert labels == ["P", "F", "T", "Y", "Z", "N", "*"]
    assert 'STEPS=${2:-"P F T Y Z"}' in text and "lcm_repair.yaml" not in text
    body = text[text.index("for step in $STEPS") :]
    steps = {
        m.group(1): " ".join(m.group(2).replace("\\\n", " ").split())
        for m in re.finditer(r"^\s{4}(\w)\) (.*?) ;;", body, flags=re.MULTILINE | re.DOTALL)
    }
    assert set(steps) == {"P", "F", "T", "Y", "Z", "N"}
    for label, command in steps.items():
        print(f"\n{label}: {command}")
        assert '--root "$ROOT"' in command, label
    assert steps["P"] == (
        "run scripts/disp_lcm.py --tenor 3m --dates monthly --budget production --risk deltas "
        '--varswap --workers "$WORKERS" --root "$ROOT"'
    )
    assert "--config" not in steps["P"] and "--no-report" not in steps["P"]
    assert steps["F"].startswith("run scripts/lcm_price.py --date 2026-10-02 --tenor 3m ")
    assert "--budget production --risk full" in steps["F"]
    assert '--row-out "$ROOT/outputs/dispersion_lc/risk_full_2026-10-02_3m.json"' in steps["F"]
    today = steps["T"].split("run ")[1:]
    assert [re.search(r"--tenor (\w+)", c).group(1) for c in today] == ["12m", "24m"]  # type: ignore[union-attr]
    for command in today:
        assert "--dates today --budget development --risk none" in command
    assert "--tenor 12m --dates monthly --budget development --risk none" in steps["Y"]
    assert "--tenor 24m --dates reference --budget development --risk none" in steps["Z"]
    assert "--tenor 3m --dates monthly --budget development --risk none" in steps["N"]
    assert "--config configs/studies/dispersion/lcm_norepair.yaml --tag norepair" in steps["N"]
    for label in ("P", "Y", "Z", "N"):
        assert '--workers "$WORKERS"' in steps[label], label
    # what the steps ask of the driver exists: the dates' keywords, the tag and the tenors
    _, sweep = lcm_scripts()
    assert sweep.TODAY == "2026-10-02" and sweep.TODAY in sweep.REFERENCE_DATES
    assert len(sweep.REFERENCE_DATES) == 4
