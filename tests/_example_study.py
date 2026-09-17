"""Example study module of ``tests/test_study_runner.py`` (the runner-module protocol of
:mod:`volsto.studies.runner`): European calls under Black–Scholes priced by Monte Carlo
(2·10⁴ antithetic paths, one exact log-Euler step) beside their closed form — one Monte Carlo
column with its stderr, exact columns, one table, one figure.  No leverage, no calibration.

``params.requires`` declares extra requirements (``[{kind, id, what}]``) so the tests can make
a point, a leverage or an artefact missing; the artefact's command is
:data:`ARTEFACT_COMMAND`.

``params.attack`` (optional, :data:`ATTACKS`) makes ``compute`` try to calibrate or write the
cache by one route, for the guard and observation tests.  The direct calls pass ``None``
arguments to ``calibrate_leverage``: should the guard ever fail, they raise on the arguments
instead of calibrating; the one real calibration route (``bypass_cache``) uses a 5000-particle,
0.1y spec.  ``late_*`` attacks start a thread that acts while the figure is drawn (render).
"""

from __future__ import annotations

import concurrent.futures as cf
import dataclasses
import multiprocessing as mp
import os
import subprocess
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from volsto.calibration.cache import LeverageCache
from volsto.calibration.guard import ENV_VAR, CalibrationForbiddenError
from volsto.calibration.particle import calibrate_leverage as _early_alias
from volsto.config import CalibrationSpec, ConfigError, SimConfig, load_yaml
from volsto.engine.mc import MonteCarlo
from volsto.market.bs import bs_price
from volsto.market.curves import ForwardCurve
from volsto.models.bs import BlackScholes
from volsto.models.leverage import LeverageFunction
from volsto.products.vanilla import DigitalOption, EuropeanOption
from volsto.studies import style
from volsto.studies.results import (
    DIMENSIONLESS,
    Column,
    FigureSpec,
    Results,
    ResultsBuilder,
    TableSpec,
)
from volsto.studies.runner import Requirement, StudyContext

TITLE = "Example study: Black-Scholes calls by Monte Carlo"
QUESTION = "Does a 2e4-path Monte Carlo reproduce the Black-Scholes call prices within 2 stderr?"
REQUIRED_PARAMS = ("vol", "maturity", "strikes", "n_paths", "requires")
OPTIONAL_PARAMS = ("attack", "extra_rows")
#: The environment variable naming the directory a test's external writer process watches
#: (``wait_for_writer``: compute writes ``go`` there and waits for ``done``).
WRITER_ENV = "VOLSTO_TEST_WRITER_DIR"
WIDE_COLUMNS = 12
ATTACKS = (
    "bypass_cache",
    "process_pool",
    "thread_pool",
    "swallow",
    "late_calibrate_in_render",
    "write_root",
    "late_write_in_render",
    "ctx_cache_hit",
    "undeclared_miss",
    "escaped_child",
    "wait_for_writer",
)
#: The late-thread state shared by compute and the figure drawer.
LATE: dict[str, Any] = {"event": None, "thread": None, "log": []}
TINY_SPEC = Path(__file__).resolve().parents[1] / "configs" / "studies" / "lsv_reference_1f.yaml"
SPOT, RATE, DIVIDEND = 100.0, 0.02, 0.01
ARTEFACT_COMMAND = ".venv/bin/python scripts/m8b.py --study D"
TABLE = "calls"


def validate_params(params: Mapping[str, Any]) -> None:
    if not params["strikes"]:
        raise ConfigError("strikes must not be empty")
    if params.get("attack") is not None and params["attack"] not in ATTACKS:
        raise ConfigError(f"attack must be one of {ATTACKS}")


def requirements(ctx: StudyContext) -> list[Requirement]:
    out: list[Requirement] = []
    for r in ctx.params["requires"]:
        if r["kind"] == "point":
            out.append(ctx.point_requirement(r["id"], r["what"]))
        elif r["kind"] == "leverage":
            out.append(ctx.leverage_requirement(r["id"], r["what"]))
        else:
            out.append(ctx.artefact_requirement(r["id"], r["what"], ARTEFACT_COMMAND))
    return out


def tiny_spec(seed_shift: int = 0) -> CalibrationSpec:
    """The 1F reference spec at 5000 particles over 0.1y (``seed_shift`` makes another key)."""
    spec = load_yaml(TINY_SPEC, CalibrationSpec)
    particle = dataclasses.replace(
        spec.particle,
        n_particles=5000,
        horizon=0.1,
        min_window=500,
        seed=spec.particle.seed + seed_shift,
    )
    return dataclasses.replace(spec, particle=particle)


def flat_leverage(spec: CalibrationSpec) -> LeverageFunction:
    """A flat L = 1 on a small grid (written to a cache by the tests; no calibration)."""
    fc = ForwardCurve.from_config(spec.market)
    return LeverageFunction(
        [0.0, spec.particle.horizon], [-1.0, -0.5, 0.0, 0.5, 1.0], [[1.0] * 5] * 2, fc
    )


def direct_calibration() -> None:
    """``calibrate_leverage`` through an alias bound at import time (``None`` arguments)."""
    _early_alias(None, None, None, None)  # type: ignore[arg-type]


def _child_calibration() -> str:
    from volsto.calibration.particle import calibrate_leverage

    calibrate_leverage(None, None, None, None)  # type: ignore[arg-type]
    return "not refused"


def _late(action: str, root: str) -> None:
    event = LATE["event"]
    assert event is not None
    event.wait(timeout=120)
    try:
        if action == "calibrate":
            direct_calibration()
        else:
            spec = tiny_spec(seed_shift=21)
            LeverageCache(root).store(spec, flat_leverage(spec))
        LATE["log"].append(f"late {action}: done")
    except Exception as exc:
        LATE["log"].append(f"late {action}: {type(exc).__name__}")


def _attack(ctx: StudyContext, attack: str) -> None:
    root = ctx.cache_root
    if attack == "bypass_cache":
        LeverageCache(root).get_or_calibrate(tiny_spec(seed_shift=20))  # allow_calibrate=True
    elif attack == "process_pool":
        with cf.ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as pool:
            pool.submit(_child_calibration).result()
    elif attack == "thread_pool":
        with cf.ThreadPoolExecutor(1) as pool:
            pool.submit(direct_calibration).result()
    elif attack == "swallow":
        try:
            direct_calibration()
        except CalibrationForbiddenError:
            ctx.log.info("the refusal was caught")
    elif attack in ("late_calibrate_in_render", "late_write_in_render"):
        LATE["event"] = threading.Event()
        LATE["log"] = []
        action = "calibrate" if attack == "late_calibrate_in_render" else "write"
        thread = threading.Thread(target=_late, args=(action, str(root)), daemon=True)
        LATE["thread"] = thread
        thread.start()
    elif attack == "write_root":
        spec = tiny_spec(seed_shift=22)
        LeverageCache(root).store(spec, flat_leverage(spec))
    elif attack == "ctx_cache_hit":
        from volsto.risk.engine import LSVBuilder, RiskState

        ctx.cache.get_or_calibrate(tiny_spec(), allow_calibrate=True)
        LSVBuilder(ctx.cache, RiskState(tiny_spec()), allow_calibrate=True)
    elif attack == "undeclared_miss":
        ctx.leverage(tiny_spec(seed_shift=23), "undeclared tiny leverage")
    elif attack == "escaped_child":
        # a child whose environment drops the prohibition (but keeps the run's marker directory)
        code = (
            "from volsto.calibration.particle import calibrate_leverage as c\n"
            "try:\n    c(None, None, None, None)\nexcept Exception:\n    pass\n"
        )
        subprocess.run(
            [sys.executable, "-c", code], env={**os.environ, ENV_VAR: "0"}, check=True, timeout=300
        )
    elif attack == "wait_for_writer":
        directory = Path(os.environ[WRITER_ENV])
        (directory / "go").write_text("go")
        t0 = time.monotonic()
        while not (directory / "done").exists():
            if time.monotonic() - t0 > 120:
                raise TimeoutError("the external writer did not finish")
            time.sleep(0.05)


def compute(ctx: StudyContext) -> Results:
    p = ctx.params
    if p.get("attack"):
        _attack(ctx, str(p["attack"]))
    vol, T, n_paths = float(p["vol"]), float(p["maturity"]), int(p["n_paths"])
    fc = ForwardCurve.flat(SPOT, RATE, DIVIDEND)
    model = BlackScholes(vol, fc)
    sim = SimConfig(n_paths=n_paths, dt_max=T, chunk_size=n_paths, seed=ctx.seed("pricing"))
    strikes = [float(k) for k in p["strikes"]]
    calls = [EuropeanOption(k, T, "call", fc.rate_curve) for k in strikes]
    digitals = [DigitalOption(k, T, "call", fc.rate_curve) for k in strikes]
    prices = MonteCarlo(sim).price_many([*calls, *digitals], model)
    ctx.record("n_paths", n_paths)
    ctx.log.info("priced %d products on %d paths", len(prices), n_paths)
    b = ResultsBuilder()
    for i, k in enumerate(strikes):
        row = f"K={k:g}"
        axes = {"strike": k}
        b.add_exact(TABLE, row, "strike", k, unit="", source="computed", axes=axes)
        b.add_mc(TABLE, row, "mc_price", prices[i], unit="price", source="computed", axes=axes)
        closed = float(bs_price(SPOT, k, T, vol, RATE, DIVIDEND, 1))
        b.add_exact(TABLE, row, "bs_price", closed, unit="price", source="computed", axes=axes)
        b.add_mc(
            TABLE,
            row,
            "digital",
            prices[len(strikes) + i],
            unit=DIMENSIONLESS,
            source="computed",
            axes=axes,
        )
    b.add_exact("setup", "budget", "n_paths", n_paths, unit="", source="computed")
    for i in range(int(p.get("extra_rows") or 0)):  # a long and wide synthetic table
        for j in range(WIDE_COLUMNS):
            b.add(
                "wide",
                f"row {i:03d}",
                f"c{j:02d}",
                100.0 + i + j / 100,
                0.05,
                unit="vol pts",
                source="computed",
                axes={"block": i // 20},
            )
    return b.build()


def tables(results: Results) -> list[TableSpec]:
    extra = []
    if "wide" in results.tables():
        cols = tuple(Column(f"c{j:02d}", f"a long column header {j}") for j in range(WIDE_COLUMNS))
        extra.append(TableSpec("wide", "A long and wide table", "wide", cols, row_header="row"))
    return [
        *extra,
        TableSpec(
            name="calls",
            caption="European calls: Monte Carlo vs closed form (σ = 20%, T = 1y, r_f 2%)",  # noqa: RUF001
            table=TABLE,
            columns=(
                Column("strike", "strike K", digits=4),
                Column("mc_price", "MC price"),
                Column("bs_price", "BS price", digits=6),
                Column("digital", "digital (disc.)"),
            ),
            row_header="strike",
        ),
    ]


def _draw(results: Results) -> Any:
    if LATE["event"] is not None:  # a late attack acts now, while the run renders
        LATE["event"].set()
        LATE["thread"].join(timeout=120)
        LATE["event"] = None
    fig, ax = style.new_figure()
    mc = results.pivot(TABLE)
    strikes = mc["strike"].to_numpy()
    diff = mc["mc_price"].to_numpy() - mc["bs_price"].to_numpy()
    ax.axhline(0.0, color=style.INK["spine"], linewidth=0.8)
    style.mc_errorbar(ax, strikes, diff, mc["mc_price_stderr"], label="Monte Carlo - BS")
    ax.set_xlabel("strike")
    ax.set_ylabel("price difference")
    ax.set_title("Monte Carlo minus Black-Scholes (error bars: 1 stderr)")
    return fig


def figures(results: Results) -> list[FigureSpec]:
    return [
        FigureSpec(
            "mc_vs_bs", "Monte Carlo call price minus the closed form, ±1 stderr bars.", _draw
        )
    ]


def narrative(results: Results) -> str:
    z = []
    for row in results.rows(TABLE):
        mc, se = results.value(TABLE, row, "mc_price")
        bs, _ = results.value(TABLE, row, "bs_price")
        z.append(abs(mc - bs) / se)
    return "\n".join(
        [
            "## Result",
            "",
            f"The largest deviation is **{max(z):.2f}** stderr over {len(z)} strikes "
            "(model: \\(\\sigma\\) flat, ω = 0; `bs_price` is exact, 50% of the rows are ITM_ish, "
            "a $5 note).",
            "",
            "{{table:calls}}",
            "",
            "- one exact log-Euler step",
            "- antithetic pairs",
            "",
            "| check | value |",
            "|---|---|",
            f"| max z | {float(np.max(z)):.2f} |",
        ]
    )
