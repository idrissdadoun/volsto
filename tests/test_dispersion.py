"""The dispersion study's stage 1 (``volsto/studies/dispersion.py``): the one-factor simulator
against the library's multi-asset layer and the Gaussian closed forms, the worlds' departures
(local correlation, uncertain correlation, jumps) in the expected directions, the trades'
bounds, the history summary on synthetic closes.  No data files, no calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volsto.multi import gaussian_palladium_forward
from volsto.multi.draws import constant_correlation
from volsto.studies.dispersion import (
    TRADES,
    World,
    gaussian_values,
    history_summary,
    simulate_library,
    simulate_one_factor,
    trade_payoffs,
    trade_values,
)

VOLS = (0.25, 0.30, 0.35, 0.40, 0.45)


def test_one_factor_matches_library_and_closed_forms() -> None:
    w = World(VOLS, 0.5, T=0.25)
    from volsto.config import SimConfig

    lib = simulate_library(w, SimConfig(n_paths=40_000, chunk_size=20_000, seed=3, dt_max=1 / 52))
    one = simulate_one_factor(w, 40_000, seed=5, steps=13)
    a = trade_values(lib, call_strike=0.05, package_scale=1.2)
    b = trade_values(one, call_strike=0.05, package_scale=1.2)
    g = gaussian_values(w, 0.05, 1.2)
    for t in TRADES:
        assert abs(a[t][0] - b[t][0]) < 3 * float(np.hypot(a[t][1], b[t][1])) + 0.01 * abs(g[t])
    assert (
        abs(
            a["palladium"][0]
            - gaussian_palladium_forward(VOLS, constant_correlation(5, 0.5), w.w, 0.25)
        )
        < 0.02 * g["palladium"]
    )
    assert (
        abs(b["variance dispersion"][0] - g["variance dispersion"])
        < 3 * b["variance dispersion"][1] + 2e-3
    )
    # realised correlation of the one-factor paths
    lr = np.log1p(one.perf)
    c = np.corrcoef(lr.T)
    assert abs(float(np.mean(c[np.triu_indices(5, 1)])) - 0.5) < 0.03


def test_departures_move_the_trades_as_expected() -> None:
    base = World(VOLS, 0.5, T=0.25)
    b = trade_values(
        simulate_one_factor(base, 40_000, 1, steps=13), call_strike=0.05, package_scale=1.2
    )
    # local correlation: the correlation rises when the basket falls — dispersion falls in
    # down moves, rises in up moves; the basket straddle gains (fatter down tail)
    loc = trade_values(
        simulate_one_factor(base.replace(local_corr=3.0), 40_000, 1, steps=13),
        call_strike=0.05,
        package_scale=1.2,
    )
    assert loc["basket straddle"][0] > b["basket straddle"][0] - 3 * b["basket straddle"][1]
    # uncertain correlation: the forward is concave in ρ (the mixture lowers it); the call
    # struck at the forward dispersion is convex and gains relative to the forward
    g = gaussian_values(base, 0.0, 1.0)["palladium"]
    b_atm = trade_values(
        simulate_one_factor(base, 40_000, 1, steps=13), call_strike=g, package_scale=1.2
    )
    unc = trade_values(
        simulate_one_factor(base.replace(corr_sd=0.3), 40_000, 1, steps=13),
        call_strike=g,
        package_scale=1.2,
    )
    assert unc["palladium"][0] < b["palladium"][0]
    assert (
        unc["palladium call"][0] / unc["palladium"][0]
        > b_atm["palladium call"][0] / b_atm["palladium"][0]
    )
    # idiosyncratic jumps: dispersion up more than the basket straddle
    jmp = trade_values(
        simulate_one_factor(base.replace(jump_prob=0.3, jump_sd=0.15), 40_000, 1, steps=13),
        call_strike=0.05,
        package_scale=1.2,
    )
    assert jmp["palladium"][0] > b["palladium"][0] + 3 * b["palladium"][1]
    assert (jmp["palladium"][0] - b["palladium"][0]) > (
        jmp["basket straddle"][0] - b["basket straddle"][0]
    )
    with pytest.raises(ValueError):
        simulate_one_factor(
            World(VOLS, tuple(tuple(r) for r in constant_correlation(5, 0.5))), 100, 0
        )


def test_trade_bounds_path_by_path() -> None:
    sim = simulate_one_factor(
        World(VOLS, 0.4, T=0.5, jump_prob=0.2, jump_sd=0.1), 5000, 9, steps=26
    )
    pay = trade_payoffs(sim, sim.world.w, call_strike=0.03, package_scale=1.0)
    assert np.all(pay["palladium"] >= pay["straddle package"] - 1e-12)
    assert np.all(pay["palladium"] <= pay["single straddles"] + pay["basket straddle"] + 1e-12)
    assert np.all(pay["palladium call"] <= pay["palladium"])


def test_history_summary_groups() -> None:
    n = 400
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "dispersion": rng.random(n),
            "straddle_package": rng.random(n),
            "basket_abs": rng.random(n),
            "singles_abs": rng.random(n),
            "corr_realised": rng.random(n),
            "corr_implied": rng.random(n),
            "corr_premium": rng.random(n),
            "vol_mean": rng.random(n),
            "vol_basket": rng.random(n),
            "vix_tercile": rng.choice(["low", "mid", "high"], n),
            "basket_move": rng.choice(["-3..3%", "> 10%"], n),
            **{f"pnl {t}": rng.normal(size=n) for t in TRADES},
        }
    )
    s = history_summary(df)
    assert set(s["regime"]) == {"all", "vix_tercile", "basket_move"} and len(s) == 1 + 3 + 2
    assert np.isfinite(s["pnl palladium_stderr"]).all()
