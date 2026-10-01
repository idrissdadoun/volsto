"""The what-if of the forward smile and skew (``volsto/viewers/whatif.py``, page 9; owner's
decision of 2026-09-30): on a Black–Scholes pair (20% → 22% vol) the forward smiles are flat at
their vols, the change is +2 vp at every strike with a paired error far below the levels' errors
(common random numbers), its skew change is zero within its error; the request validates and keys;
a run stores its JSON and reads it back; the page renders without computing.  Never calibrates:
the LSV step is replaced by injected Black–Scholes models."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from volsto.config import SimConfig
from volsto.market.curves import ForwardCurve
from volsto.models.bs import BlackScholes
from volsto.viewers import whatif as wi
from volsto.viewers.grid import REPO_ROOT


def _fc() -> ForwardCurve:
    return ForwardCurve.flat(100.0, 0.03, 0.01)


def test_black_scholes_pair_change_is_paired() -> None:
    fc = _fc()
    base, moved = BlackScholes(0.20, fc), BlackScholes(0.22, fc)
    sim = SimConfig(n_paths=20_000, chunk_size=20_000, seed=7, dt_max=1.0 / 52.0)
    smiles, fits = wi.compare(base, moved, ((0.5, 1.0), (1.0, 2.0)), wi.DEFAULT_STRIKES, sim)
    assert set(smiles["window"]) == {"0.5y→1y", "1y→2y"}
    ok = np.isfinite(smiles["base_vol"]) & np.isfinite(smiles["change_stderr"])
    s = smiles[ok]
    assert len(s) >= 16
    # flat Black-Scholes smiles at their vols, within 4 of their own errors
    assert np.all(np.abs(s["base_vol"] - 0.20) < 4 * s["base_vol_stderr"] + 1e-4)
    assert np.all(np.abs(s["moved_vol"] - 0.22) < 4 * s["moved_vol_stderr"] + 1e-4)
    # the change: +2 vp everywhere, its paired error far below the unpaired one
    assert np.all(np.abs(s["change"] - 0.02) < 4 * s["change_stderr"] + 1e-4)
    unpaired = np.hypot(s["base_vol_stderr"], s["moved_vol_stderr"])
    assert np.all(s["change_stderr"] < 0.5 * unpaired), (s["change_stderr"] / unpaired).max()
    ch = fits[fits["which"] == "change"]
    assert np.all(np.abs(ch["atm"] - 0.02) < 4 * ch["atm_stderr"] + 1e-4)
    assert np.all(np.abs(ch["skew"]) < 4 * ch["skew_stderr"] + 1e-4)
    print(fits.round(5).to_string())


def test_request_validation_and_key() -> None:
    r = wi.WhatIfRequest()
    assert r.n_particles == wi.WHATIF_PARTICLES == 100_000 and r.horizon == 3.0
    assert r.key() == wi.WhatIfRequest().key()
    assert r.key() != wi.WhatIfRequest(changes=(("nu", 2.0),)).key()
    with pytest.raises(ValueError, match="unknown parameters"):
        wi.WhatIfRequest(changes=(("omega", 1.0),))
    with pytest.raises(ValueError, match="even n_paths"):
        wi.WhatIfRequest(n_paths=1001)
    with pytest.raises(ValueError, match="windows"):
        wi.WhatIfRequest(windows=((1.0, 0.5),))
    assert wi.changes_label((("nu", 2.0), ("rho12", 0.1))) == "nu 2, rho12 0.1"


def test_run_stores_and_reloads_without_calibrating(tmp_path: Path) -> None:
    """The base and moved models injected (vol 20% for the base, 22% when ν moves): the run
    stores ``<outputs>/whatif/runs/<key>.json`` and a second call reads it back unchanged."""
    fc = _fc()
    req = wi.WhatIfRequest(
        changes=(("nu", 3.0),), windows=((1.0, 2.0),), n_particles=1000, n_paths=4000
    )
    base_nu = wi.base_params(req, REPO_ROOT)["nu"]
    calls: list[float] = []

    def inject(state: object) -> BlackScholes:
        nu = float(state.spec.model.nu)  # type: ignore[attr-defined]
        calls.append(nu)
        return BlackScholes(0.20 if math.isclose(nu, base_nu) else 0.22, fc)

    res = wi.run_whatif(req, tmp_path, REPO_ROOT, calibrate=inject)
    assert calls == [base_nu, 3.0]
    assert res.info["moved_params"]["nu"] == 3.0 and res.info["base_params"]["nu"] == base_nu
    stored = tmp_path / "whatif" / "runs" / f"{req.key()}.json"
    assert stored.exists()
    again = wi.run_whatif(req, tmp_path, REPO_ROOT, calibrate=inject)
    assert calls == [base_nu, 3.0]  # read back, nothing recomputed
    assert again.smiles["change"].tolist() == pytest.approx(res.smiles["change"].tolist())
    assert [r.request.key() for r in wi.stored_runs(tmp_path)] == [req.key()]


def test_page_renders_without_running(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The page renders headless (title, form, Run button) and computes nothing until Run."""
    from streamlit.testing.v1 import AppTest

    from volsto.viewers.pages import PAGES_DIR

    monkeypatch.setenv("VOLSTO_OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setenv("VOLSTO_STORE", str(tmp_path / "store"))
    monkeypatch.setenv("VOLSTO_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("VOLSTO_VIEWER_CONFIG", str(tmp_path / "no_viewer.yaml"))
    at = AppTest.from_file(str(PAGES_DIR / "9_what_if.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.value for t in at.title] == ["What-if"]
    assert any(b.label == "Run" for b in at.button)
    assert not (tmp_path / "outputs" / "whatif").exists()
    assert not (tmp_path / "cache").exists()
