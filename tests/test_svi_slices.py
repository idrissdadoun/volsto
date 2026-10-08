"""M12 part LC1 (SPEC §8.7): the SVI slices promoted from the dispersion study's check C8 into
``volsto/market/svi_slices.py``.

V1: the library's fit, slice and surface equal C8's, bit for bit, against the frozen copy of
the C8 code (``tests/_c8_reference.py``) on synthetic smiles and — slow, when the study's data
is present — on the smiles of the thirty Dow names and DJX on 2026-10-02.  Also: the analytic
derivatives, the interpolation in time, the arbitrage report on a crossed calendar and on a
butterfly violation, the slice selection, and the fit records (the key sees every ulp; a
recorded fit is read back and never fitted again)."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import _c8_reference as c8
import numpy as np
import pytest

from volsto.calibration import fit_records as fr
from volsto.config import LocalVolConfig
from volsto.market import svi_slices as sv
from volsto.market.curves import ForwardCurve
from volsto.market.dupire import LocalVolSurface
from volsto.market.surface import ArbitrageError

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "outputs" / "dispersion"
TODAY = "2026-10-02"

#: Gatheral & Jacquier (2014), Example 3.1 (Axel Vogt's slice): raw SVI parameters with a
#: negative density in the call wing.
VOGT = (-0.0410, 0.1331, 0.3060, 0.3586, 0.4153)  # (a, b, rho, m, sigma)


def _smile(
    T: float,
    atm: float,
    skew: float,
    curv: float,
    n: int,
    half_width: float,
    noise: float,
    seed: int,
) -> c8.Smile:
    """A listed-looking smile: ``n`` strikes over ``±half_width`` in log-moneyness (not centred
    on the forward), vol ``atm + skew·k + curv·k²`` with multiplicative noise."""
    rng = np.random.default_rng(seed)
    k = np.sort(rng.uniform(-half_width, 0.8 * half_width, n))
    vol = (atm + skew * k + curv * k * k) * (1.0 + noise * rng.standard_normal(n))
    return c8.Smile(T, k, np.maximum(vol, 0.02))


def _synthetic_smiles() -> list[c8.Smile]:
    return [
        _smile(30 / 365, 0.22, -0.60, 1.5, 25, 0.25, 0.0, 1),
        _smile(60 / 365, 0.25, -0.45, 0.9, 31, 0.35, 0.01, 2),
        _smile(91 / 365, 0.24, -0.35, 0.6, 41, 0.45, 0.02, 3),
        _smile(182 / 365, 0.26, -0.28, 0.4, 35, 0.60, 0.01, 4),
        _smile(365 / 365, 0.27, -0.20, 0.2, 29, 0.80, 0.03, 5),
        _smile(14 / 365, 0.45, -1.20, 6.0, 17, 0.12, 0.02, 6),  # every strike inside the floor
        _smile(45 / 365, 0.15, -0.30, 0.5, 9, 0.50, 0.0, 7),  # wide strikes, narrow band
        c8.Smile(0.25, np.array([-0.3, 0.1, 0.4]), np.array([0.3, 0.2, 0.22])),  # < 5: all fitted
        c8.Smile(0.5, np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5]), np.full(6, 0.2)),  # one-sided
    ]


def _same_fit(e: c8.Smile) -> sv.SviSliceFit:
    ref_params, ref_rms = c8.fit_svi(e)
    got = sv.fit_svi_slice(e.k, e.vol, e.T)
    np.testing.assert_array_equal(np.array(got.params), ref_params)
    assert got.rms_vp == ref_rms and got.T == e.T
    np.testing.assert_array_equal(sv.svi_total_variance(got.params, e.k), c8.svi(ref_params, e.k))
    return got


def test_svi_matches_c8() -> None:
    """V1 on synthetic smiles: every fitted parameter, the fit error, the slice, the surface on
    a ``(k, T)`` mesh (before the first slice, between slices, after the last, one slice only)
    and the Dupire local variance built from it are C8's, bit for bit."""
    smiles = _synthetic_smiles()
    fits = [_same_fit(e) for e in smiles]
    # the strike selection: the three-strike smile is fitted whole; the one-sided smile keeps
    # the five strikes inside 3 sd (0.42); the 14-day smile sits inside the 0.15 floor; the wide
    # low-vol smile has fewer than five strikes inside its band (0.158), so all nine are fitted;
    # the 3m smile loses the strikes beyond 3 sd (0.36)
    assert [f.n_points for f in fits[-2:]] == [3, 5]
    assert fits[5].n_points == 17
    assert int((np.abs(smiles[6].k) <= 0.158).sum()) < 5 and fits[6].n_points == 9
    assert fits[2].n_points < smiles[2].k.size and fits[2].k_lo > -0.37
    # the surface: the first five maturities are increasing
    use, used = smiles[:5], fits[:5]
    times = np.array([e.T for e in use])
    params = np.stack([np.array(f.params) for f in used])
    fc = ForwardCurve.flat(1.0, 0.03, 0.01)
    lib = sv.SviSlices(times, params, fc, max_maturity=1.05)
    ref = c8.SviSlices(times, params, fc, max_maturity=1.05)
    k = np.linspace(-1.5, 1.5, 61)[None, :]
    T = np.concatenate([[1 / 365, 0.05], times, 0.5 * (times[:-1] + times[1:]), [1.02, 1.05]])
    np.testing.assert_array_equal(
        lib.total_variance(k, T[:, None]), ref.total_variance(k, T[:, None])
    )
    assert float(lib.total_variance(0.1, 0.3)) == float(ref.total_variance(0.1, 0.3))  # scalars
    one = sv.SviSlices(times[:1], params[:1], fc, max_maturity=1.0)
    one_ref = c8.SviSlices(times[:1], params[:1], fc, max_maturity=1.0)
    np.testing.assert_array_equal(
        one.total_variance(k, T[:, None]), one_ref.total_variance(k, T[:, None])
    )
    cfg = LocalVolConfig(t_min=1 / 365, t_max=0.27, n_t=40, k_min=-2.0, k_max=2.0, n_k=401)
    a, b = LocalVolSurface.from_implied(lib, cfg), LocalVolSurface.from_implied(ref, cfg)
    np.testing.assert_array_equal(a.local_var, b.local_var)
    assert a.diagnostics == b.diagnostics
    with pytest.raises(ValueError):
        sv.SviSlices(times[::-1], params, fc, 1.05)
    with pytest.raises(ValueError):
        sv.SviSlices(times, params[:, :4], fc, 1.05)
    with pytest.raises(ValueError):
        sv.fit_svi_slice([0.1, 0.0], [0.2, 0.2], 0.25)  # strikes not increasing


def _study_data_present() -> bool:
    from volsto.studies import disp_data as dd

    return (
        (STUDY / "prices.parquet").exists()
        and (STUDY / "entries" / "3m" / f"{TODAY}.pkl").exists()
        and dd.day_path(TODAY).exists()
    )


@pytest.mark.slow
@pytest.mark.skipif(not _study_data_present(), reason="the dispersion study's data is absent")
def test_svi_matches_c8_on_recorded_smiles() -> None:
    """V1 on recorded smiles: every listed expiry of the thirty Dow names and of DJX on
    2026-10-02 (the study's smiles, parity forwards), fitted by the library and by the frozen
    C8 code — the same parameters and the same error, bit for bit."""
    from volsto.studies import disp_data as dd
    from volsto.studies import disp_smile as ds

    with (STUDY / "entries" / "3m" / f"{TODAY}.pkl").open("rb") as fh:
        names = list(pickle.load(fh)["names"])
    chains = dd.read_chains(TODAY, [*names, "DJX"])
    panel = dd.prices()
    n = 0
    worst = 0.0
    for t in [*names, "DJX"]:
        px = float(panel.at[TODAY, t]) if t in panel.columns else float("nan")
        smiles = ds.expiry_smiles(chains[t], TODAY, "parity", spot=px if np.isfinite(px) else None)
        assert smiles, t
        for e in smiles:
            worst = max(worst, _same_fit(e).rms_vp)
            n += 1
    assert n > 200
    print(
        f"{n} expiry smiles on {TODAY}: library fit = C8 fit bit for bit; worst rms {worst:.2f} vp"
    )


def test_svi_derivatives() -> None:
    """``svi_derivatives`` against central differences of ``svi_total_variance``."""
    k = np.linspace(-1.2, 1.2, 97)
    h = 1e-4
    for p in ((0.01, 0.08, -0.6, 0.02, 0.15), (0.0, 0.2, 0.3, -0.1, 0.4), VOGT):
        w, w1, w2 = sv.svi_derivatives(p, k)
        np.testing.assert_array_equal(w, sv.svi_total_variance(p, k))
        up, dn = sv.svi_total_variance(p, k + h), sv.svi_total_variance(p, k - h)
        np.testing.assert_allclose(w1, (up - dn) / (2 * h), rtol=0, atol=1e-8)
        np.testing.assert_allclose(w2, (up - 2 * w + dn) / (h * h), rtol=0, atol=2e-6)


def test_slices_interpolate_linearly_in_time() -> None:
    """Between two slices the total variance is linear in ``T`` at fixed ``k``; before the first
    and after the last it is proportional to ``T`` (the implied vol at fixed ``k`` is flat); a
    slice below the floor is floored."""
    fc = ForwardCurve.flat(1.0, 0.0, 0.0)
    p = np.array([[0.004, 0.05, -0.5, 0.0, 0.2], [0.010, 0.07, -0.4, 0.0, 0.25]])
    s = sv.SviSlices([0.25, 0.75], p, fc, 2.0)
    k = np.linspace(-0.5, 0.5, 11)
    w0, w1 = sv.svi_total_variance(p[0], k), sv.svi_total_variance(p[1], k)
    np.testing.assert_allclose(s.total_variance(k, 0.25), w0, rtol=1e-15)
    np.testing.assert_allclose(s.total_variance(k, 0.75), w1, rtol=1e-15)
    np.testing.assert_allclose(s.total_variance(k, 0.5), 0.5 * (w0 + w1), rtol=1e-14)
    np.testing.assert_allclose(s.total_variance(k, 0.1), w0 * 0.1 / 0.25, rtol=1e-14)
    np.testing.assert_allclose(s.total_variance(k, 1.5), w1 * 1.5 / 0.75, rtol=1e-14)
    np.testing.assert_allclose(s.implied_vol_k(k, 1.5), s.implied_vol_k(k, 0.75), rtol=1e-14)
    np.testing.assert_allclose(s.implied_vol_k(k, 0.01), s.implied_vol_k(k, 0.25), rtol=1e-14)
    assert s.n_slices == 2 and "SviSlices" in repr(s)
    # a slice that crosses zero in the put wing is floored there
    neg = sv.SviSlices([0.5], [[-0.02, 0.1, 0.9, 0.0, 0.05]], fc, 1.0)
    assert float(sv.svi_total_variance(neg.params[0], -0.5)) < 0
    assert float(neg.total_variance(-0.5, 0.5)) == sv.W_FLOOR


def test_arbitrage_report_detects_violations() -> None:
    """A clean surface passes; a calendar crossed in the put wing and a slice with a negative
    density (Gatheral–Jacquier's Example 3.1) are reported and raise ``ArbitrageError``."""
    fc = ForwardCurve.flat(1.0, 0.0, 0.0)
    clean = sv.SviSlices(
        [0.25, 0.5], [[0.004, 0.05, -0.5, 0.0, 0.2], [0.010, 0.07, -0.4, 0.0, 0.25]], fc, 1.0
    )
    rep = clean.arbitrage_report()
    assert rep.ok and rep.violations() == [] and rep.n_k == 401 and rep.k_range == 1.0
    assert len(rep.min_g) == 2 and len(rep.min_calendar) == 1 and rep.n_floored == (0, 0)
    assert min(rep.min_g) > 0 and rep.min_calendar[0] > 0
    clean.check_no_arbitrage()
    assert json.loads(json.dumps(rep.to_dict()))["ok"] is True
    # calendar: the second slice is above the first at the money and below it in the put wing
    crossed = sv.SviSlices(
        [0.25, 0.5], [[0.002, 0.10, -0.6, 0.0, 0.1], [0.010, 0.03, -0.3, 0.0, 0.1]], fc, 1.0
    )
    assert float(crossed.total_variance(0.0, 0.5)) > float(crossed.total_variance(0.0, 0.25))
    rep = crossed.arbitrage_report()
    assert rep.butterfly_ok and not rep.calendar_ok and not rep.ok
    assert rep.min_calendar[0] < 0 and rep.argmin_calendar[0] == -1.0
    assert len(rep.violations()) == 1 and rep.violations()[0].startswith("calendar")
    with pytest.raises(ArbitrageError, match="calendar"):
        crossed.check_no_arbitrage()
    # butterfly
    vogt = sv.SviSlices([1.0], [VOGT], fc, 2.0)
    rep = vogt.arbitrage_report(k_range=1.5, n_k=601)
    assert not rep.butterfly_ok and rep.calendar_ok and rep.min_g[0] < -1e-3
    assert rep.argmin_g[0] > 0.5  # the negative density sits in the call wing
    with pytest.raises(ArbitrageError, match="butterfly"):
        vogt.check_no_arbitrage(k_range=1.5)
    # the butterfly function of the report is the density condition: finite-difference check
    k = np.linspace(-1.5, 1.5, 601)
    w = sv.svi_total_variance(VOGT, k)
    wk = np.gradient(w, k)
    wkk = np.gradient(wk, k)
    g = (1.0 - k * wk / (2.0 * w)) ** 2 - 0.25 * wk * wk * (1.0 / w + 0.25) + 0.5 * wkk
    assert rep.min_g[0] == pytest.approx(g[5:-5].min(), abs=2e-4)
    with pytest.raises(ValueError):
        clean.arbitrage_report(n_k=2)


class _Expiry:
    def __init__(self, e: c8.Smile, T: float) -> None:
        self.T, self.k, self.vol = T, e.k, e.vol


def test_fit_svi_surface_selects_c8_slices(tmp_path: Path) -> None:
    """C8's selection: expiries from ten days, those up to the horizon and the next two; the
    surface ends at ``max(last slice, horizon) + 0.05``; with records, one record per slice."""
    base = _synthetic_smiles()[2]
    days = (5, 20, 50, 80, 120, 200, 300, 400)
    expiries = [_Expiry(base, d / 365) for d in days]
    fc = ForwardCurve.flat(1.0, 0.02, 0.0)
    surface, fits = sv.fit_svi_surface(expiries, fc, horizon=0.25, origin="test")
    np.testing.assert_array_equal(surface.times, np.array([20, 50, 80, 120, 200]) / 365)
    assert [f.T for f in fits] == list(surface.times)
    assert surface.max_maturity == max(200 / 365, 0.25) + 0.05
    for f, e in zip(fits, expiries[1:6], strict=True):
        np.testing.assert_array_equal(surface.params[surface.times == f.T][0], c8.fit_svi(e)[0])
    # a horizon beyond every expiry: the surface still covers it
    far, _ = sv.fit_svi_surface(expiries, fc, horizon=2.0, origin="test")
    assert far.n_slices == 7 and far.max_maturity == 2.05
    LocalVolSurface.from_implied(
        far, LocalVolConfig(t_min=1 / 365, t_max=2.02, n_t=30, k_min=-2.0, k_max=2.0, n_k=201)
    )
    with pytest.raises(ValueError, match="no listed expiry"):
        sv.fit_svi_surface(expiries[:1], fc, horizon=0.25, origin="test")
    records = fr.FitRecords(tmp_path / "svi_fits")
    recorded, fits2 = sv.fit_svi_surface(expiries, fc, horizon=0.25, records=records, origin="test")
    assert len(records.keys()) == 5 and fits2 == fits
    np.testing.assert_array_equal(recorded.params, surface.params)


def test_svi_fit_records(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The record key sees every ulp of what the fit reads and nothing else; a recorded fit is
    read back bit for bit and never fitted again; the record carries the SVI code tag, and the
    marking-fit records keep theirs (``FitRecords.put``'s default)."""
    e = _synthetic_smiles()[1]
    settings = sv.svi_fit_settings()
    assert settings == {"width_sd": 3.0, "min_width": 0.15, "min_points": 5}
    key = sv.svi_fit_key(e.T, e.k, e.vol, settings)
    assert len(key) == 64 and key == sv.svi_fit_key(e.T, list(e.k), tuple(e.vol), dict(settings))
    seen = {key}
    for i in range(e.k.size):
        for arr in ("k", "vol"):
            k, vol = e.k.copy(), e.vol.copy()
            target = k if arr == "k" else vol
            target[i] = np.nextafter(target[i], np.inf)
            seen.add(sv.svi_fit_key(e.T, k, vol, settings))
    seen.add(sv.svi_fit_key(float(np.nextafter(e.T, 1.0)), e.k, e.vol, settings))
    for name, value in (("width_sd", 3.5), ("min_width", 0.2), ("min_points", 6)):
        seen.add(sv.svi_fit_key(e.T, e.k, e.vol, {**settings, name: value}))
    assert len(seen) == 1 + 2 * e.k.size + 1 + 3
    payload = {"inputs": sv.svi_fit_inputs(e.T, e.k, e.vol, settings), "fit_code_tag": "svi2"}
    import hashlib

    assert hashlib.sha256(fr.canonical(payload).encode()).hexdigest() != key  # the tag is keyed
    # round trip
    records = fr.FitRecords(tmp_path / "cache" / "lc" / "svi_fits")
    assert sv.recorded_svi_fit(e.k, e.vol, e.T, records=None, origin="x") == sv.fit_svi_slice(
        e.k, e.vol, e.T
    )
    assert records.keys() == []
    first = sv.recorded_svi_fit(e.k, e.vol, e.T, records=records, origin="test_svi")
    assert first == sv.fit_svi_slice(e.k, e.vol, e.T) and records.keys() == [key]
    doc = records.get(key)
    assert doc is not None and doc["fit_code_tag"] == sv.SVI_FIT_CODE_TAG == "svi1"
    assert doc["origin"] == "test_svi" and sorted(doc["inputs"]) == ["T", "k", "settings", "vol"]
    assert sv.SviSliceFit.from_summary(doc["fit"]) == first

    def no_fit(*args: object, **kwargs: object) -> sv.SviSliceFit:
        raise AssertionError("a recorded slice was fitted again")

    monkeypatch.setattr(sv, "fit_svi_slice", no_fit)
    again = sv.recorded_svi_fit(e.k, e.vol, e.T, records=records, origin="another machine")
    assert again == first and records.get(key)["origin"] == "test_svi"  # type: ignore[index]
    # other settings are another record (and would fit)
    with pytest.raises(AssertionError, match="fitted again"):
        sv.recorded_svi_fit(e.k, e.vol, e.T, records=records, origin="x", min_width=0.2)
    # FitRecords.put: the default tag is the marking fit's, as before the keyword existed
    other = records.put("ab" * 32, {"a": 1}, {"params": [1.0]}, origin="default tag")
    assert other["fit_code_tag"] == fr.FIT_CODE_TAG != sv.SVI_FIT_CODE_TAG
