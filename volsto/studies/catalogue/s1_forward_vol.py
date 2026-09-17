"""S1 — forward vol and cliquets versus vol-of-vol (SPEC §10.2, owner's M10 Part 2; the original
paper's study): the M4 headline table, the forward smiles, the put-wing invariance and the capped
cliquet ladder across the 1F ν grid and the 2F presets, with the regeneration of the M4
regression baseline.

**Data.**  Every number is read from the M9 results store (``source = store:<point id>``: the
M4 headline columns of ``products``, the ``forward_smile`` and ``forward_vols`` tables, the
``points`` provenance).  A selected point missing from the store is, with ``price_missing:
true``, priced here from its cached leverage with :func:`volsto.studies.m4.run_headline` (the
1y → 2y window, the cliquets) and :func:`volsto.analytics.forward_smile.forward_smile` /
``forward_vol_comparison`` (the 2y → 3y window) — exactly the precompute's step 2–3, on the
config's paths and ``pricing`` seed (``source = cache:<key>``, ``computed`` for the local vol);
with ``price_missing: false`` a missing point is a requirement (the run exits 2 with the
precompute line) and nothing is priced.  ``volsto-study run configs/studies/catalogue/s1.yaml
--set price_missing=false`` is the store-only run.

**Cross-model differences** (the LSV-minus-LV forward smile per strike, the cliquet ratios)
are differences of stored estimates: their stderr is the quadrature error.  The estimates share
random numbers and the store keeps no per-path samples, so the correlation is not measured and
the quadrature error is not the exact error (it bounds it only if the correlation is
non-negative); each z is d over that error.  The put-wing reading compares the largest wing
difference with the smallest at-the-money one, which guards the claim against the choice of
model.  A number whose product ends beyond the calibration horizon (a forward window by the
store's flag; a cliquet, and the 1y → 2y headline keys when no flag is at hand, against the
point's horizon) says so in its note, in every table that shows it.  The
"capped-cliquet ladder" is read as the study cliquet at its fixed 2% cap across the models (an
interpretation, pending the owner).

**Params** (all required)::

    models: {surface, lv, one_factor, two_factor}   # catalogue._common.validate_selection
    headline: {rho: -0.7, kappa: 1.5}              # the 1F slice of the headline table / smiles
    cliquet_maturities: [1.0, 2.0]                 # the store's cliquet columns
    price_missing: true                            # price absent store points from the cache
    pricing: {n_paths: 400000}                     # paths of that inline pricing
    baseline: {file: tests/test_m4_regression.py, variable: PLACEHOLDER_BASELINES} | null

**Regeneration.**  With a ``baseline``, the file is read as data (``ast``; nothing is imported
from the tests): ``PLACEHOLDER_BASELINES`` ``{model: {key: (value, stderr)}}`` and the
``HEADLINE_N_PARTICLES`` / ``HEADLINE_N_PATHS`` / ``HEADLINE_SEED`` it was recorded with.  Per
key present in both, the tables ``regression_vol`` / ``regression_price`` / ``regression_ratio``
carry the store value, the baseline, the difference (stderr: the two errors in quadrature — the
"combined se"; with the same seed and leverage the two estimates are the same computation, so a
non-zero difference is a code change or the baseline file's rounding, not noise, and the
combined se is only a scale), ``|d| / combined se``, the regression
tolerance of the test (``max(2 × store stderr, floor)``, floors 0.02 vol points, 0.02 % of
notional, 0.002 for ratios and probabilities) and whether ``|d|`` is within it;
``regression_meta`` compares the particle counts, paths and seeds.

Checked by ``tests/test_catalogue_s1_s4.py`` (the fast config on the toy store, the inline-pricing
path on a store copy missing a point, the store-only switch).
"""

from __future__ import annotations

import ast
import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from volsto.config import ConfigError
from volsto.studies import style
from volsto.studies.catalogue._common import (
    LV_LABEL,
    QUADRATURE_NOTE,
    VP,
    ModelPoint,
    axis,
    axis_text,
    band,
    baseline_name,
    empty_panel,
    leverage_requirements,
    load_model,
    model_source,
    point_requirements,
    pricing_sim,
    rss,
    select_models,
    series_by,
    store_source,
    stored_ids,
    validate_selection,
)
from volsto.studies.latex import format_value_text
from volsto.studies.results import (
    DIMENSIONLESS,
    Column,
    FigureSpec,
    Results,
    ResultsBuilder,
    TableSpec,
    records,
)
from volsto.studies.runner import Requirement, StudyContext
from volsto.viewers.grid import REPO_ROOT

TITLE = "S1 - Forward volatility and capped cliquets versus vol-of-vol"
QUESTION = (
    "How do the forward smile, the forward volatilities and the capped cliquets move with "
    "vol-of-vol when every model is calibrated to the same smile?"
)
REQUIRED_PARAMS = (
    "models",
    "headline",
    "cliquet_maturities",
    "price_missing",
    "pricing",
    "baseline",
)
OPTIONAL_PARAMS: tuple[str, ...] = ()

#: The store's forward-start windows (:data:`volsto.viewers.precompute.FORWARD_WINDOWS`).
WINDOWS: tuple[tuple[float, float], ...] = ((1.0, 2.0), (2.0, 3.0))
#: The M4 headline keys of the store's ``products`` table (1y → 2y window).
FWD_KEYS: tuple[tuple[str, str], ...] = (
    ("atm_vol", "fwd_atm_vol"),
    ("vs_vol", "fwd_vs"),
    ("volswap_vol", "fwd_volswap"),
)
HEADLINE_KEYS = frozenset(k for k, _ in FWD_KEYS)
#: Maturity (years) of the M4c conditional and VKO keys (``volsto.studies.m4`` prices them at 1y).
M4_PRODUCT_T = 1.0
VOL = "vol pts"
PCT = "% notional"
#: Regression floors of ``tests/test_m4_regression.py`` in the tables' units.
FLOOR_VOL_VP = 0.02
FLOOR_PRICE_PCT = 0.02
FLOOR_RATIO = 0.002
REGRESSION_TABLES: tuple[str, ...] = ("regression_vol", "regression_price", "regression_ratio")
BASELINE_CONSTANTS: tuple[str, ...] = ("HEADLINE_N_PARTICLES", "HEADLINE_N_PATHS", "HEADLINE_SEED")
#: Rows of the ``setup`` table.
SETUP_ROWS: tuple[str, ...] = (
    "models",
    "read from the store",
    "priced from the cache",
    "headline rho",
    "headline kappa",
)


def window_tag(t1: float, t2: float) -> str:
    return f"{t1:g}y{t2:g}y"


def window_label(t1: float, t2: float) -> str:
    return f"{t1:g}y→{t2:g}y"


def cliquet_key(T: float) -> str:
    return f"cliquet_{T:g}y"


# --------------------------------------------------------------------------------------------
# params and requirements
# --------------------------------------------------------------------------------------------


def validate_params(params: Mapping[str, Any]) -> None:
    validate_selection(params["models"])
    h = params["headline"]
    if not isinstance(h, Mapping) or set(h) != {"rho", "kappa"}:
        raise ConfigError("headline: expected {rho, kappa}")
    mats = params["cliquet_maturities"]
    if not isinstance(mats, list) or not mats or any(float(T) <= 0 for T in mats):
        raise ConfigError("cliquet_maturities: expected a non-empty list of positive maturities")
    if not isinstance(params["price_missing"], bool):
        raise ConfigError("price_missing: expected true or false")
    pr = params["pricing"]
    if not isinstance(pr, Mapping) or set(pr) != {"n_paths"}:
        raise ConfigError("pricing: expected {n_paths}")
    if int(pr["n_paths"]) <= 0 or int(pr["n_paths"]) % 2:
        raise ConfigError("pricing.n_paths: a positive even number (antithetic pairs)")
    b = params["baseline"]
    if b is not None and (not isinstance(b, Mapping) or set(b) != {"file", "variable"}):
        raise ConfigError("baseline: expected null or {file, variable}")


def requirements(ctx: StudyContext) -> list[Requirement]:
    """A stored point per model; with ``price_missing`` a model absent from the store needs its
    cached leverage instead (the local vol needs nothing)."""
    models = select_models(ctx, ctx.params["models"])
    if ctx.params["baseline"] is not None:
        read_baseline(ctx.params["baseline"])  # a missing / unreadable file is a config error
    if not ctx.params["price_missing"]:
        return point_requirements(ctx, models)
    have = stored_ids(ctx)
    present = [mp for mp in models if mp.id in have]
    absent = [mp for mp in models if mp.id not in have]
    return point_requirements(ctx, present) + leverage_requirements(ctx, absent)


# --------------------------------------------------------------------------------------------
# baseline
# --------------------------------------------------------------------------------------------


def read_baseline(spec: Mapping[str, Any]) -> dict[str, Any]:
    """``{"values": {model: {key: (v, se)}}, "constants": {...}, "path", "sha256"}`` from the
    baseline file, read as data (module docstring)."""
    path = Path(str(spec["file"]))
    path = path if path.is_absolute() else REPO_ROOT / path
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"baseline file {path}: {exc}") from exc
    wanted = {str(spec["variable"]), *BASELINE_CONSTANTS}
    found: dict[str, Any] = {}
    for node in ast.parse(text).body:
        target: ast.expr | None = None
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            target = node.target
            value = node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            value = node.value
        if isinstance(target, ast.Name) and target.id in wanted:
            found[target.id] = ast.literal_eval(value)
    missing = sorted(wanted - set(found))
    if missing:
        raise ConfigError(f"baseline file {path}: {missing} not found as literal assignments")
    return {
        "values": found[str(spec["variable"])],
        "constants": {k: found[k] for k in BASELINE_CONSTANTS},
        "path": str(spec["file"]),
        "sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def key_group(key: str) -> tuple[str, str, float, float]:
    """``(results table, unit, scale to the unit, floor in the unit)`` of a baseline key (the
    floors of ``tests/test_m4_regression.py``)."""
    if key.startswith("cliquet") or key == "vko_30":
        return "regression_price", PCT, 1.0, FLOOR_PRICE_PCT
    if key.endswith("_p_ko") or key.startswith("vko_ratio_"):
        return "regression_ratio", DIMENSIONLESS, 1.0, FLOOR_RATIO
    return "regression_vol", VOL, VP, FLOOR_VOL_VP


# --------------------------------------------------------------------------------------------
# compute
# --------------------------------------------------------------------------------------------


def price_point(
    ctx: StudyContext, mp: ModelPoint, n_paths: int, maturities: tuple[float, ...]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """The store's ``products`` (M4 rows), ``forward_smile`` and ``forward_vols`` frames of one
    point, priced here from the cached leverage (module docstring)."""
    from volsto.analytics.forward_smile import forward_smile
    from volsto.studies.m4 import HEADLINE_STRIKES, run_headline
    from volsto.viewers.precompute import _fvc_row, _m4_product_rows, _smile_rows

    model = load_model(ctx, mp)
    sim = pricing_sim(ctx, mp, n_paths)
    horizon = float(mp.point.spec.particle.horizon)
    (t1a, t2a), (t1b, t2b) = WINDOWS
    ctx.log.info("%s: not in the store, pricing from %s at %d paths", mp.label, mp.id[:12], n_paths)
    head = run_headline(
        {mp.label: model},
        sim,
        t1=t1a,
        t2=t2a,
        cliquet_maturities=maturities,
        conditional=False,
    )
    r0 = head.table.iloc[0]
    products = pd.DataFrame(_m4_product_rows(r0))
    smiles = pd.concat(
        [
            _smile_rows(head.smiles[mp.label], horizon),
            _smile_rows(forward_smile(model, t1b, t2b, HEADLINE_STRIKES, sim), horizon),
        ],
        ignore_index=True,
    )
    fvols = pd.DataFrame(
        [
            {
                "t1": t1a,
                "t2": t2a,
                "beyond_horizon": bool(t2a > horizon + 1e-9),
                "fwd_atm_vol": float(r0["atm_vol"]),
                "fwd_atm_vol_stderr": float(r0["atm_vol_stderr"]),
                "fwd_vs": float(r0["vs_vol"]),
                "fwd_vs_stderr": float(r0["vs_vol_stderr"]),
                "fwd_volswap": float(r0["volswap_vol"]),
                "fwd_volswap_stderr": float(r0["volswap_vol_stderr"]),
            },
            _fvc_row(model, t1b, t2b, sim, horizon),
        ]
    )
    return products, smiles, fvols


def _strike_row(k: float, log_m: float) -> str:
    return f"k={k:.4g}" + (" (ATMF)" if abs(log_m) < 1e-12 else "")


def _key_values(products: pd.DataFrame) -> dict[str, tuple[float, float]]:
    if products.empty:
        return {}
    return {str(r["key"]): (float(r["value"]), float(r["value_stderr"])) for r in records(products)}


def _window(frame: pd.DataFrame, t1: float, t2: float) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame[np.isclose(frame["t1"], t1) & np.isclose(frame["t2"], t2)]


@dataclass(frozen=True)
class _Frames:
    """One model's store-shaped frames and where they came from."""

    products: pd.DataFrame
    smile: pd.DataFrame
    fvols: pd.DataFrame
    source: str
    stored: bool


def compute(ctx: StudyContext) -> Results:
    p = ctx.params
    models = select_models(ctx, p["models"])
    maturities = tuple(float(T) for T in p["cliquet_maturities"])
    n_inline = int(p["pricing"]["n_paths"])
    have = stored_ids(ctx)
    reader = ctx.store()
    frames: dict[str, _Frames] = {}
    for mp in models:
        if mp.id in have:
            frames[mp.label] = _Frames(
                reader.products(mp.id),
                reader.forward_smile(mp.id),
                reader.forward_vols(mp.id),
                store_source(mp.id),
                True,
            )
        elif p["price_missing"]:
            prod, sm, fv = price_point(ctx, mp, n_inline, maturities)
            frames[mp.label] = _Frames(prod, sm, fv, model_source(mp), False)
        else:  # pragma: no cover - requirements() makes this unreachable
            raise ConfigError(f"{mp.label}: not in the store and price_missing is false")
    stored_points = {mp.id: reader.point(mp.id) for mp in models if frames[mp.label].stored}
    n_priced = sum(1 for f in frames.values() if not f.stored)
    lv = next((mp for mp in models if mp.is_lv), None)
    b = ResultsBuilder()

    # -- provenance per model ----------------------------------------------------------------
    budgets: dict[str, tuple[float, float, float]] = {}
    for mp in models:
        fr = frames[mp.label]
        rec = stored_points.get(mp.id)
        if rec is not None:
            n_part = float(rec["n_particles"])
            paths, seed = float(rec["pricing_n_paths"]), float(rec["pricing_seed"])
        else:
            n_part = float(mp.point.spec.particle.n_particles)
            paths, seed = float(n_inline), float(ctx.seed("pricing"))
        budgets[mp.label] = (n_part, paths, seed)
        for col, val in (
            ("from_store", 1.0 if fr.stored else 0.0),
            ("n_particles", math.nan if mp.is_lv else n_part),
            ("n_paths", paths),
            ("seed", seed),
            ("horizon", float(mp.point.spec.particle.horizon)),
        ):
            b.add_exact("provenance", mp.label, col, val, unit="", source=fr.source, axes=mp.axes)

    # -- headline and cliquet ladder -----------------------------------------------------------
    missing_cells: list[str] = []
    lv_vals = _key_values(frames[lv.label].products) if lv is not None else {}
    lv_source = frames[lv.label].source if lv is not None else ""
    for mp in models:
        fr = frames[mp.label]
        vals = _key_values(fr.products)
        for key, col in FWD_KEYS:
            v, s = vals.get(key, (math.nan, math.nan))
            if not math.isfinite(v):
                missing_cells.append(f"{mp.label} {key}")
            b.add(
                "headline",
                mp.label,
                col,
                VP * v,
                VP * s,
                unit=VOL,
                source=fr.source,
                note=_key_beyond(fr, mp, key),
                axes=mp.axes,
            )
        for T in maturities:
            key = cliquet_key(T)
            v, s = vals.get(key, (math.nan, math.nan))
            if not math.isfinite(v):
                missing_cells.append(f"{mp.label} {key}")
            beyond = _key_beyond(fr, mp, key)
            for table in ("headline", "cliquet_ladder"):
                b.add(
                    table,
                    mp.label,
                    key,
                    v,
                    s,
                    unit=PCT,
                    source=fr.source,
                    note=beyond,
                    axes=mp.axes,
                )
            if lv is None:
                continue
            col = f"ratio_{T:g}y"
            if mp.is_lv:
                b.add_exact(
                    "cliquet_ladder",
                    mp.label,
                    col,
                    1.0,
                    unit=DIMENSIONLESS,
                    source=fr.source,
                    note=_join("the reference itself", beyond),
                    axes=mp.axes,
                )
                continue
            lv_v, lv_s = lv_vals.get(key, (math.nan, math.nan))
            lv_beyond = _key_beyond(frames[lv.label], lv, key)
            lv_beyond = f"reference: {lv_beyond}" if lv_beyond else ""
            if not (math.isfinite(v) and math.isfinite(lv_v) and v != 0.0):
                b.add(
                    "cliquet_ladder",
                    mp.label,
                    col,
                    math.nan,
                    None,
                    unit=DIMENSIONLESS,
                    source=fr.source,
                    note=_join(beyond, lv_beyond),
                    axes=mp.axes,
                )
                continue
            ratio = v / lv_v
            b.add(
                "cliquet_ladder",
                mp.label,
                col,
                ratio,
                abs(ratio) * rss(s / v, lv_s / lv_v),
                unit=DIMENSIONLESS,
                source=f"{fr.source};{lv_source}",
                note=_join("relative errors in quadrature", QUADRATURE_NOTE, beyond, lv_beyond),
                axes=mp.axes,
            )

    # -- forward vols per window ---------------------------------------------------------------
    for mp in models:
        fr = frames[mp.label]
        for r in records(fr.fvols):
            t1, t2 = float(r["t1"]), float(r["t2"])
            row = f"{mp.label} | {window_label(t1, t2)}"
            axes = {**mp.axes, "t1": t1, "t2": t2}
            flagged, how = _beyond(r, mp, t2)
            for col in ("fwd_atm_vol", "fwd_vs", "fwd_volswap"):
                b.add(
                    "forward_vols",
                    row,
                    col,
                    VP * float(r[col]),
                    VP * float(r[f"{col}_stderr"]),
                    unit=VOL,
                    source=fr.source,
                    note=how,
                    axes=axes,
                )
            b.add_exact(
                "forward_vols",
                row,
                "beyond_horizon",
                1.0 if flagged else 0.0,
                unit="",
                source=fr.source,
                note=how,
                axes=axes,
            )

    # -- forward smiles and the pairwise LSV minus LV differences -------------------------------
    head = p["headline"]
    for t1, t2 in WINDOWS:
        tag = window_tag(t1, t2)
        lv_rows: dict[str, dict[str, Any]] = {}
        if lv is not None:
            for r in records(_window(frames[lv.label].smile, t1, t2)):
                lv_rows[_strike_row(float(r["strike_moneyness"]), float(r["log_moneyness"]))] = r
        for mp in models:
            fr = frames[mp.label]
            for r in records(_window(fr.smile, t1, t2)):
                k, lm = float(r["strike_moneyness"]), float(r["log_moneyness"])
                row = _strike_row(k, lm)
                iv, se = float(r["iv"]), float(r["iv_stderr"])
                flagged, how = _beyond(r, mp, t2)
                axes = {
                    **mp.axes,
                    "k": k,
                    "log_moneyness": lm,
                    "t1": t1,
                    "t2": t2,
                    "beyond_horizon": float(flagged),
                }
                b.add(
                    f"smile_{tag}",
                    row,
                    mp.label,
                    VP * iv,
                    VP * se,
                    unit=VOL,
                    source=fr.source,
                    note=how,
                    axes=axes,
                )
                ref = lv_rows.get(row)
                if mp.is_lv or ref is None or lv is None:
                    continue
                lv_flagged, lv_how = _beyond(ref, lv, t2)
                lv_how = f"reference: {lv_how}" if lv_flagged else ""
                wing_axes = {**axes, "beyond_horizon": float(flagged or lv_flagged)}
                d = VP * (iv - float(ref["iv"]))
                quad = VP * rss(se, float(ref["iv_stderr"]))
                both = f"{fr.source};{lv_source}"
                tail = "".join(f"; {x}" for x in (how, lv_how) if x)
                b.add(
                    f"wing_vs_lv_{tag}",
                    row,
                    mp.label,
                    d,
                    quad,
                    unit=VOL,
                    source=both,
                    note="LSV minus LV; " + QUADRATURE_NOTE + tail,
                    axes=wing_axes,
                )
                b.add_exact(
                    f"wing_z_{tag}",
                    row,
                    mp.label,
                    d / quad if quad > 0 else math.nan,
                    unit="",
                    source="computed",
                    note="d over the quadrature error (not an exact significance)" + tail,
                    axes={**wing_axes, "headline": float(_in_headline(mp.axes, head))},
                )
            if mp.is_lv:
                for r in records(_window(fr.smile, t1, t2)):
                    k = float(r["strike_moneyness"])
                    row = _strike_row(k, float(r["log_moneyness"]))
                    lv_flagged, how = _beyond(r, mp, t2)
                    b.add(
                        f"wing_lv_{tag}",
                        row,
                        "lv_iv",
                        VP * float(r["iv"]),
                        VP * float(r["iv_stderr"]),
                        unit=VOL,
                        source=fr.source,
                        note=how,
                        axes={"k": k, "t1": t1, "t2": t2, "beyond_horizon": float(lv_flagged)},
                    )

    # -- setup ---------------------------------------------------------------------------------
    setup = (
        float(len(models)),
        float(len(models) - n_priced),
        float(n_priced),
        float(head["rho"]),
        float(head["kappa"]),
    )
    for row, val in zip(SETUP_ROWS, setup, strict=True):
        b.add_exact("setup", row, "value", val, unit="", source="computed")
    b.add_exact(
        "setup",
        "missing store cells",
        "value",
        float(len(missing_cells)),
        unit="",
        source="computed",
        note="; ".join(missing_cells[:20]) + (" ..." if len(missing_cells) > 20 else ""),
    )

    # -- regeneration against the M4 baseline ----------------------------------------------
    if p["baseline"] is not None:
        base = read_baseline(p["baseline"])
        ctx.record(
            "baseline",
            {"path": base["path"], "sha256": base["sha256"], "constants": base["constants"]},
        )
        _regression(b, models, frames, budgets, base)

    ctx.record("n_paths", {"inline": n_inline if n_priced else None, "priced_models": n_priced})
    ctx.record("store_models", len(models) - n_priced)
    return b.build()


def _beyond_suffix(mp: ModelPoint) -> str:
    if mp.is_lv:
        return " (the local vol has no leverage: only the flag's convention)"
    return " (the last leverage slice is held beyond it)"


def _beyond(rec: Mapping[str, Any], mp: ModelPoint, t2: float) -> tuple[bool, str]:
    """Whether a stored (or inline) window ends beyond the calibration horizon, and the note
    every number of it carries."""
    flag = rec.get("beyond_horizon")
    horizon = float(mp.point.spec.particle.horizon)
    if flag is None or (isinstance(flag, float) and math.isnan(flag)):
        beyond = t2 > horizon + 1e-9
        how = "t2 beyond the calibration horizon (derived: the store predates the flag)"
    else:
        beyond = bool(flag)
        how = "the store's flag: t2 beyond the grid's calibration horizon"
    return beyond, how + _beyond_suffix(mp) if beyond else ""


def _key_beyond(fr: _Frames, mp: ModelPoint, key: str) -> str:
    """The beyond-horizon note of a store key (empty when within): the headline keys follow the
    1y → 2y window's flag, a cliquet and the 1y M4c / VKO products compare their maturity with
    the point's horizon."""
    t1, t2 = WINDOWS[0]
    if key in HEADLINE_KEYS:
        win = _window(fr.fvols, t1, t2)
        rec = records(win)[0] if len(win) else {}
        return _beyond(rec, mp, t2)[1]
    end = float(key[len("cliquet_") : -1]) if key.startswith("cliquet_") else M4_PRODUCT_T
    horizon = float(mp.point.spec.particle.horizon)
    if end <= horizon + 1e-9:
        return ""
    return (
        f"the product's {end:g}y maturity is beyond the {horizon:g}y calibration horizon "
        "(against the point's horizon)" + _beyond_suffix(mp)
    )


def _join(*notes: str) -> str:
    return "; ".join(n for n in notes if n)


def _in_headline(axes: Mapping[str, Any], headline: Mapping[str, Any]) -> bool:
    mode = axes.get("mode")
    if mode in ("lv", "two_factor"):
        return True
    return (
        mode == "one_factor"
        and math.isclose(float(axes["rho"]), float(headline["rho"]))
        and math.isclose(float(axes["kappa"]), float(headline["kappa"]))
    )


def _regression(
    b: ResultsBuilder,
    models: list[ModelPoint],
    frames: Mapping[str, _Frames],
    budgets: Mapping[str, tuple[float, float, float]],
    base: Mapping[str, Any],
) -> None:
    values: Mapping[str, Mapping[str, tuple[float, float]]] = base["values"]
    const = base["constants"]
    b_part = float(const["HEADLINE_N_PARTICLES"])
    b_paths = float(const["HEADLINE_N_PATHS"])
    b_seed = float(const["HEADLINE_SEED"])
    note_base = (
        f"baseline {base['path']} ({b_part:.0f} particles, {b_paths:.0f} paths, "
        f"seed {b_seed:.0f})"
    )
    for mp in models:
        name = baseline_name(mp)
        if name is None or name not in values:
            continue
        fr = frames[mp.label]
        vals = _key_values(fr.products)
        n_part, paths, seed = budgets[mp.label]
        # the local vol has no particles: only paths and seed can differ
        match = (mp.is_lv or n_part == b_part) and paths == b_paths and seed == b_seed
        meta_axes = {**mp.axes, "baseline_model": name}
        for col, val in (
            ("n_particles", math.nan if mp.is_lv else n_part),
            ("n_particles_baseline", b_part),
            ("n_paths", paths),
            ("n_paths_baseline", b_paths),
            ("seed", seed),
            ("seed_baseline", b_seed),
            ("match", 1.0 if match else 0.0),
        ):
            b.add_exact(
                "regression_meta", name, col, val, unit="", source=fr.source, axes=meta_axes
            )
        for key, (bv, bs) in values[name].items():
            if key not in vals:
                continue
            table, unit, scale, floor = key_group(key)
            sv, ss = vals[key]
            row = f"{name} / {key}"
            axes = {**mp.axes, "baseline_model": name, "key": key}
            d = scale * (sv - float(bv))
            comb = scale * rss(ss, float(bs))
            tol = max(2.0 * scale * ss, floor)
            exact_unit = "" if unit == DIMENSIONLESS else unit
            beyond = _key_beyond(fr, mp, key)
            b.add(
                table,
                row,
                "store",
                scale * sv,
                scale * ss,
                unit=unit,
                source=fr.source,
                note=beyond,
                axes=axes,
            )
            b.add(
                table,
                row,
                "baseline",
                scale * float(bv),
                scale * float(bs),
                unit=unit,
                source="computed",
                note=_join(note_base, beyond),
                axes=axes,
            )
            b.add(
                table,
                row,
                "diff",
                d,
                comb,
                unit=unit,
                source=fr.source,
                note=_join(
                    "store minus baseline; stderr = the two errors in quadrature, only a scale "
                    "(the same seed and leverage make the two the same computation)",
                    beyond,
                ),
                axes=axes,
            )
            b.add_exact(
                table,
                row,
                "n_se",
                abs(d) / comb if comb > 0 else math.nan,
                unit="",
                source="computed",
                note=_join(
                    "|d| / combined stderr (a deterministic function of the estimates)", beyond
                ),
                axes=axes,
            )
            b.add_exact(
                table,
                row,
                "tolerance",
                tol,
                unit=exact_unit,
                source="computed",
                note=_join("max(2 x store stderr, floor) of tests/test_m4_regression.py", beyond),
                axes=axes,
            )
            b.add_exact(
                table,
                row,
                "within",
                1.0 if abs(d) <= tol else 0.0,
                unit="",
                source="computed",
                note=beyond,
                axes=axes,
            )


# --------------------------------------------------------------------------------------------
# tables, figures, narrative (from the results only)
# --------------------------------------------------------------------------------------------


def _headline_setup(results: Results) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in ("headline rho", "headline kappa"):
        out[row] = results.value("setup", row, "value")[0]
    return out


def headline_rows(results: Results, table: str) -> tuple[str, ...]:
    """The rows of ``table`` in the headline set (LV, the 1F slice, the 2F presets)."""
    h = _headline_setup(results)
    head = {"rho": h["headline rho"], "kappa": h["headline kappa"]}
    return tuple(r for r in results.rows(table) if _in_headline(results.axes(table, r), head))


def _wing_models(results: Results, tag: str) -> list[str]:
    """The headline LSV models of the ``wing_vs_lv_<tag>`` table (at most 8)."""
    if not _has(results, f"wing_vs_lv_{tag}"):
        return []
    head = set(headline_rows(results, "headline"))
    return [c for c in results.columns(f"wing_vs_lv_{tag}") if c in head][:8]


def _cliquet_cols(results: Results) -> list[str]:
    return [c for c in results.columns("headline") if c.startswith("cliquet_")]


def _has(results: Results, table: str) -> bool:
    return table in results.tables()


def tables(results: Results) -> list[TableSpec]:
    specs: list[TableSpec] = []
    cliq = _cliquet_cols(results)
    head_rows = headline_rows(results, "headline")
    specs.append(
        TableSpec(
            name="headline",
            caption=(
                "M4 headline set: 1y into 1y forward ATMF vol, forward variance-swap and "
                "vol-swap vols (vol points), and the study cliquets (monthly, local cap 2%, "
                "global floor 0; % of notional), per model calibrated to the same surface."
            ),
            table="headline",
            columns=(
                Column("fwd_atm_vol", "ATMF fwd", unit="vp"),
                Column("fwd_vs", "fwd VS", unit="vp"),
                Column("fwd_volswap", "fwd vol swap", unit="vp"),
                *(Column(c, c.replace("cliquet_", "cliquet "), unit="%") for c in cliq),
            ),
            rows=head_rows,
            row_header="model",
        )
    )
    fv_rows = tuple(
        r
        for r in results.rows("forward_vols")
        if _in_headline(results.axes("forward_vols", r), _hdict(results))
    )
    specs.append(
        TableSpec(
            name="forward_vols",
            caption=(
                "Forward volatilities per window, vol points (beyond = 1: the window ends beyond "
                "the calibration horizon, priced on the last leverage slice)."
            ),
            table="forward_vols",
            columns=(
                Column("fwd_atm_vol", "ATMF fwd", unit="vp"),
                Column("fwd_vs", "fwd VS", unit="vp"),
                Column("fwd_volswap", "fwd vol swap", unit="vp"),
                Column("beyond_horizon", "beyond", digits=1),
            ),
            rows=fv_rows,
            row_header="model | window",
        )
    )
    for t1, t2 in WINDOWS:
        tag = window_tag(t1, t2)
        if _has(results, f"smile_{tag}"):
            labels = [c for c in results.columns(f"smile_{tag}")]
            head = set(headline_rows(results, "headline"))
            cols = [c for c in labels if c in head][:8]
            specs.append(
                TableSpec(
                    name=f"smile_{tag}",
                    caption=(
                        f"Forward smile {window_label(t1, t2)} (strikes relative to S at "
                        f"{t1:g}y; implied vols from out-of-the-money forward-start options)."
                    ),
                    table=f"smile_{tag}",
                    columns=tuple(Column(c, c) for c in cols),
                    row_header="strike",
                )
            )
        head_lsv = [c for c in _wing_models(results, tag)]
        if head_lsv:
            specs.append(
                TableSpec(
                    name=f"wing_vs_lv_{tag}",
                    caption=(
                        f"Forward smile {window_label(t1, t2)}: LSV minus LV implied vol per "
                        "strike, vol points; stderr: the quadrature error (the two stored "
                        "estimates share random numbers, whose correlation is not measured "
                        "here, so this is not the exact error of the difference)."
                    ),
                    table=f"wing_vs_lv_{tag}",
                    columns=tuple(Column(c, c) for c in head_lsv),
                    row_header="strike",
                )
            )
            specs.append(
                TableSpec(
                    name=f"wing_z_{tag}",
                    caption=(
                        f"Forward smile {window_label(t1, t2)}: the same differences over their "
                        "quadrature error (not exact significances)."
                    ),
                    table=f"wing_z_{tag}",
                    columns=tuple(Column(c, c, digits=2) for c in head_lsv),
                    row_header="strike",
                )
            )
    h = _hdict(results)
    ladder_rows = tuple(
        r
        for r in results.rows("cliquet_ladder")
        if results.axes("cliquet_ladder", r).get("mode") != "one_factor"
        or math.isclose(float(results.axes("cliquet_ladder", r)["kappa"]), h["kappa"])
    )
    ladder_cols = [c for c in results.columns("cliquet_ladder")]
    specs.append(
        TableSpec(
            name="cliquet_ladder",
            caption=(
                "Capped cliquet ladder (the study cliquet at its fixed 2% local cap, across the "
                "models) at the headline kappa: price in % of notional and ratio to the local-vol "
                "price (relative errors in quadrature, not the exact error of the ratio)."
            ),
            table="cliquet_ladder",
            columns=tuple(
                (
                    Column(c, c.replace("cliquet_", "cliquet "), unit="%", digits=4)
                    if c.startswith("cliquet_")
                    else Column(c, c.replace("ratio_", "/ LV "), digits=4)
                )
                for c in ladder_cols
            ),
            rows=ladder_rows,
            row_header="model",
        )
    )
    specs.append(
        TableSpec(
            name="provenance",
            caption=(
                "Where each model's numbers come from: 1 = the results store, 0 = priced here "
                "from the cached leverage; particles, pricing paths and seed."
            ),
            table="provenance",
            columns=(
                Column("from_store", "store", digits=1),
                Column("n_particles", "particles", digits=6),
                Column("n_paths", "paths", digits=6),
                Column("seed", "seed", digits=6),
                Column("horizon", "horizon [y]", digits=3),
            ),
            rows=headline_rows(results, "provenance"),
            row_header="model",
        )
    )
    for t in REGRESSION_TABLES:
        if not _has(results, t):
            continue
        kind = t.split("_")[1]
        unit = {"vol": "vp", "price": "%", "ratio": ""}[kind]
        specs.append(
            TableSpec(
                name=t,
                caption=(
                    f"Regeneration of the M4 regression baseline ({kind} keys"
                    + {"vol": ", vol points", "price": ", % of notional", "ratio": ""}[kind]
                    + "): store value, baseline, and their difference with the combined stderr."
                ),
                table=t,
                columns=(
                    Column("store", "store", unit=unit),
                    Column("baseline", "baseline", unit=unit),
                    Column("diff", "d", unit=unit),
                ),
                row_header="model / key",
            )
        )
        specs.append(
            TableSpec(
                name=f"{t}_check",
                caption=(
                    f"Regeneration of the M4 regression baseline ({kind} keys), the check: "
                    "|d| over the combined stderr, the test's tolerance max(2 store stderr, "
                    "floor) and 1 when |d| is within it."
                ),
                table=t,
                columns=(
                    Column("n_se", "|d| / se", digits=3),
                    Column("tolerance", "tolerance", unit=unit, digits=3),
                    Column("within", "within", digits=1),
                ),
                row_header="model / key",
            )
        )
    if _has(results, "regression_meta"):
        specs.append(
            TableSpec(
                name="regression_meta",
                caption=(
                    "Budgets of the store numbers against the baseline's (1 = particles, paths "
                    "and seed all match)."
                ),
                table="regression_meta",
                columns=tuple(
                    Column(
                        c,
                        c.replace("n_particles", "particles")
                        .replace("_baseline", " base")
                        .replace("n_paths", "paths")
                        .replace("_", " "),
                        digits=7,
                    )
                    for c in results.columns("regression_meta")
                ),
                row_header="baseline model",
            )
        )
    return specs


def _hdict(results: Results) -> dict[str, float]:
    h = _headline_setup(results)
    return {"rho": h["headline rho"], "kappa": h["headline kappa"]}


def _slice_1f(
    long: pd.DataFrame, h: Mapping[str, float], *, kappa_only: bool = False
) -> pd.DataFrame:
    """The LV rows and the 1F rows of the headline slice (or of the headline kappa only)."""
    mode = axis_text(long, "mode")
    rho, kappa = axis(long, "rho"), axis(long, "kappa")
    in_slice = np.isclose(kappa, h["kappa"])
    if not kappa_only:
        in_slice &= np.isclose(rho, h["rho"])
    return long[(mode == "lv") | ((mode == "one_factor") & in_slice)]


def _two_factor(long: pd.DataFrame) -> pd.DataFrame:
    return long[axis_text(long, "mode") == "two_factor"]


def _draw_fwd_vs_nu(results: Results) -> Any:
    fig, axes = style.new_figure(1, 2, sharey=True)
    h = _hdict(results)
    names = {"fwd_atm_vol": "ATMF fwd vol", "fwd_vs": "fwd VS", "fwd_volswap": "fwd vol swap"}
    long_all = results.long("forward_vols")
    for ax, (t1, t2) in zip(axes, WINDOWS):
        long = long_all[np.isclose(axis(long_all, "t1"), t1) & np.isclose(axis(long_all, "t2"), t2)]
        long = long[long["column"].isin(list(names))]
        one = _slice_1f(long, h)
        if one.empty:
            empty_panel(ax, "no LV or 1F point at the headline slice")
            continue
        one = one.assign(axis_nu=axis(one, "nu"))
        n = series_by(ax, "axis_nu", [(names[c], one[one["column"] == c]) for c in names])
        two = _two_factor(long)
        for i, r in enumerate(records(two[two["column"] == "fwd_atm_vol"])):
            label = f"ATMF {str(r['row']).split(' | ')[0]}"
            band(ax, float(r["value"]), float(r["stderr"]), label, min(n + i, 7))
        ax.set_title(f"{window_label(t1, t2)} (1F at rho {h['rho']:g}, kappa {h['kappa']:g})")
        ax.set_xlabel("nu (omega = 2 nu; nu = 0 is LV)")
        ax.set_ylabel("vol [vol pts]")
        ax.legend()
    return fig


def _draw_smiles(results: Results) -> Any:
    fig, axes = style.new_figure(1, 2, sharey=True)
    head = set(headline_rows(results, "headline"))
    for ax, (t1, t2) in zip(axes, WINDOWS):
        tag = window_tag(t1, t2)
        if not _has(results, f"smile_{tag}"):
            empty_panel(ax, f"no {window_label(t1, t2)} smile in the results")
            continue
        long = results.long(f"smile_{tag}")
        long = long.assign(axis_k=axis(long, "k"))
        labels = [c for c in results.columns(f"smile_{tag}") if c in head][:8]
        series_by(ax, "axis_k", [(lab, long[long["column"] == lab]) for lab in labels])
        ax.set_title(f"Forward smile {window_label(t1, t2)}")
        ax.set_xlabel("strike (fraction of S at t1)")
        ax.set_ylabel("implied vol [vol pts]")
        ax.legend()
    return fig


def _draw_wing(results: Results) -> Any:
    fig, axes = style.new_figure(1, 2, sharey=True)
    head = set(headline_rows(results, "headline"))
    for ax, (t1, t2) in zip(axes, WINDOWS):
        tag = window_tag(t1, t2)
        if not _has(results, f"wing_vs_lv_{tag}"):
            empty_panel(ax, "no local-vol reference in the selection")
            continue
        long = results.long(f"wing_vs_lv_{tag}")
        long = long.assign(axis_k=axis(long, "k"))
        labels = [c for c in results.columns(f"wing_vs_lv_{tag}") if c in head][:8]
        ax.axhline(0.0, color=style.INK["spine"], linewidth=0.8)
        series_by(ax, "axis_k", [(lab, long[long["column"] == lab]) for lab in labels])
        ax.set_title(f"LSV minus LV, {window_label(t1, t2)}")
        ax.set_xlabel("strike (fraction of S at t1)")
        ax.set_ylabel("implied vol difference [vol pts]")
        ax.legend()
    return fig


def _rho_groups(long: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """One series per rho, each starting from the LV rows at nu = 0."""
    mode = axis_text(long, "mode")
    lv = long[mode == "lv"]
    one = long[mode == "one_factor"]
    rho = axis(one, "rho")
    groups = [
        (f"rho {r:g}", pd.concat([lv, one[np.isclose(rho, r)]]))
        for r in sorted({float(x) for x in rho})
    ]
    if not groups and not lv.empty:
        groups = [("LV", lv)]
    return [(lab, g.assign(axis_nu=axis(g, "nu"))) for lab, g in groups]


def _draw_cliquets(results: Results) -> Any:
    cols = [c for c in results.columns("cliquet_ladder") if c.startswith("cliquet_")]
    fig, axes = style.new_figure(1, max(len(cols), 1))
    axes_list = list(np.atleast_1d(axes))
    h = _hdict(results)
    for ax, col in zip(axes_list, cols):
        long = results.long("cliquet_ladder", col)
        n = series_by(ax, "axis_nu", _rho_groups(_slice_1f(long, h, kappa_only=True)))
        for i, r in enumerate(records(_two_factor(long))):
            band(ax, float(r["value"]), float(r["stderr"]), str(r["row"]), min(n + i, 7))
        ax.set_title(f"{col.replace('cliquet_', 'cliquet ')} (1F at kappa {h['kappa']:g})")
        ax.set_xlabel("nu (nu = 0 is LV)")
        ax.set_ylabel("price [% notional]")
        ax.legend()
    return fig


def _draw_grid(results: Results) -> Any:
    long = results.long("headline", "fwd_atm_vol")
    mode = axis_text(long, "mode")
    kappas = sorted({float(x) for x in axis(long[mode == "one_factor"], "kappa")})
    fig, axes = style.new_figure(1, max(len(kappas), 1), sharey=True)
    axes_list = list(np.atleast_1d(axes))
    if not kappas:
        empty_panel(axes_list[0], "no 1F point in the selection")
        return fig
    for ax, kappa in zip(axes_list, kappas):
        sub = long[
            (mode == "lv") | ((mode == "one_factor") & np.isclose(axis(long, "kappa"), kappa))
        ]
        series_by(ax, "axis_nu", _rho_groups(sub))
        ax.set_title(f"kappa {kappa:g}")
        ax.set_xlabel("nu (nu = 0 is LV)")
        ax.set_ylabel("1y->2y ATMF fwd vol [vol pts]")
        ax.legend()
    return fig


def _draw_regression(results: Results) -> Any:
    fig, ax = style.new_figure()
    x0 = 0
    ticks: list[str] = []
    for i, t in enumerate(t for t in REGRESSION_TABLES if _has(results, t)):
        piv = results.pivot(t)
        ratio = (piv["diff"].abs() / piv["tolerance"]).to_numpy(dtype=float)
        xs = np.arange(x0, x0 + len(ratio), dtype=float)
        style.mc_errorbar(
            ax,
            xs,
            ratio,
            np.zeros_like(ratio),
            exact=np.ones_like(ratio, dtype=bool),
            series=i,
            label=t.replace("regression_", "") + " keys",
            line=False,
        )
        ticks += list(piv.index)
        x0 += len(ratio)
    if not ticks:
        empty_panel(ax, "no baseline comparison in the results")
        return fig
    ax.axhline(1.0, color=style.INK["spine"], linewidth=0.8, linestyle="--")
    ax.set_xlabel("baseline key (model / key, table order)")
    ax.set_ylabel("|store - baseline| / tolerance")
    ax.set_title("Regeneration against the M4 baseline (<= 1: within the regression tolerance)")
    ax.legend()
    return fig


def figures(results: Results) -> list[FigureSpec]:
    specs = [
        FigureSpec(
            "forward_vols_vs_nu",
            "Forward ATM vol, forward variance-swap vol and forward vol-swap vol against nu "
            "(1F headline slice; nu = 0 is the local vol), error bars 1 stderr; dashed lines "
            "with a 1-stderr band: the 2F presets' ATMF forward vol.",
            _draw_fwd_vs_nu,
        ),
        FigureSpec(
            "forward_smiles",
            "Forward smiles of the headline models, error bars 1 stderr.",
            _draw_smiles,
        ),
        FigureSpec(
            "put_wing_vs_lv",
            "Forward implied vol minus the local vol's, per strike, for the headline LSV models; "
            "error bars 1 quadrature error (of two stored estimates that share random numbers, "
            "whose correlation is not measured here: not the exact error).",
            _draw_wing,
        ),
        FigureSpec(
            "cliquet_vs_nu",
            "Capped cliquet prices against nu per rho at the headline kappa (nu = 0 is the local "
            "vol), error bars 1 stderr; dashed lines with a 1-stderr band: the 2F presets.",
            _draw_cliquets,
        ),
        FigureSpec(
            "fwd_atm_vol_grid",
            "1y into 1y ATMF forward vol against nu, one panel per kappa, one series per rho "
            "(the whole 1F grid of the selection), error bars 1 stderr.",
            _draw_grid,
        ),
    ]
    if any(_has(results, t) for t in REGRESSION_TABLES):
        specs.append(
            FigureSpec(
                "regression_ratio",
                "|store - baseline| over the regression tolerance per key (a deterministic "
                "function of the two estimates, whose errors the tolerance already carries: no "
                "error bars); points above the dashed line are outside the tolerance.",
                _draw_regression,
            )
        )
    return specs


#: The phrase every beyond-horizon note of this study contains.
BEYOND_PHRASE = "calibration horizon"


def _horizon_lines(results: Results) -> list[str]:
    """How many numbers lie beyond the calibration horizon (from the notes)."""
    noted = 0
    for t in results.tables():
        long = results.long(t)
        noted += int(long["note"].fillna("").str.contains(BEYOND_PHRASE, regex=False).sum())
    windows = results.pivot("forward_vols") if _has(results, "forward_vols") else None
    n_win = 0 if windows is None else int((windows["beyond_horizon"] == 1.0).sum())
    if not noted:
        return []
    total = 0 if windows is None else len(windows)
    return [
        f"**{n_win} of {total} (model, window) forward-vol rows end beyond the calibration "
        f"horizon**, and {noted} numbers across the tables belong to a product that does (a "
        "window by the store's flag, a cliquet by its maturity): each says so in its note.",
        "",
    ]


def narrative(results: Results) -> str:
    setup = {r: results.value("setup", r, "value")[0] for r in results.rows("setup")}
    n_models = int(setup["models"])
    n_store = int(setup["read from the store"])
    n_priced = int(setup["priced from the cache"])
    lines = [
        "## Data",
        "",
        f"{n_models} models calibrated to the same surface: {n_store} read from the M9 results "
        f"store (`source = store:<point id>`), {n_priced} priced here from their cached leverage "
        "(`source = cache:<key>`, `computed` for the local vol) with the precompute's own "
        "functions (`run_headline`, `forward_smile`, `forward_vol_comparison`). Nothing was "
        "calibrated. The `provenance` table lists the particle count, the paths and the seed "
        "behind every row.",
        "",
    ]
    missing = int(setup["missing store cells"])
    if missing:
        note = str(results.record("setup", "missing store cells", "value")["note"])
        lines += [
            f"**{missing} cell(s) are absent from the store** and shown as `--`: {note}.",
            "",
        ]
    lines += _horizon_lines(results)
    lines += [
        "The original study's archive (its SSVI parameters, seeds and tables) is absent (SPEC "
        "§12 open item), so the placeholder-surface baseline of `tests/test_m4_regression.py` "
        "stands in for the paper's tables. The study's recalled numbers — 1y into 1y ATM "
        "forward vol 21.5% (LV) to 18.6% (ω = 3), forward variance swap 25.2-25.5%, 1y "
        "cliquet 1.082 / 1.194 / 1.472 / 1.763% and 2y 0.639% to 1.718% for ω = 0 / 1 / 2 / "
        "3 — were measured on the study's own surface, not this one.",
        "",
        "## Headline",
        "",
        "{{table:headline}}",
        "",
    ]
    head = headline_rows(results, "headline")
    lv = LV_LABEL if LV_LABEL in head else None
    ones = [r for r in head if results.axes("headline", r).get("mode") == "one_factor"]
    if lv is not None and ones:
        last = max(ones, key=lambda r: float(results.axes("headline", r)["nu"]))
        a0, s0 = results.value("headline", lv, "fwd_atm_vol")
        a1, s1 = results.value("headline", last, "fwd_atm_vol")
        v0, e0 = results.value("headline", lv, "fwd_vs")
        v1, e1 = results.value("headline", last, "fwd_vs")
        da, dv = a1 - a0, v1 - v0
        da_se, dv_se = rss(s0, s1), rss(e0, e1)
        reading = (
            "the forward ATM vol moves by more than ten times the forward variance-swap vol"
            if abs(da) - 2 * da_se > 10 * (abs(dv) + 2 * dv_se)
            else "the two moves are not separated by a factor of ten at 2 stderr"
        )
        lines += [
            f"From the local vol to {last}, the 1y into 1y ATMF forward vol moves from "
            f"{format_value_text(a0, s0)} to {format_value_text(a1, s1)} vol points "
            f"({da:+.2f}, quadrature error {da_se:.2f}) while the forward variance-swap vol moves "
            f"from {format_value_text(v0, e0)} to {format_value_text(v1, e1)} ({dv:+.2f}, "
            f"quadrature error {dv_se:.2f}): {reading}.",
            "",
        ]
        for col in _cliquet_cols(results):
            c0, t0 = results.value("headline", lv, col)
            c1, t1 = results.value("headline", last, col)
            if math.isfinite(c0) and math.isfinite(c1):
                lines.append(
                    f"- {col.replace('cliquet_', 'cliquet ')}: {format_value_text(c0, t0)} (LV) "
                    f"to {format_value_text(c1, t1)} % of notional ({last})."
                )
        lines.append("")
    lines += [
        "{{figure:forward_vols_vs_nu}}",
        "",
        "## Forward smiles and the put wing",
        "",
    ]
    for t1, t2 in WINDOWS:
        tag = window_tag(t1, t2)
        models = _wing_models(results, tag)
        if not models:
            continue
        long = results.long(f"wing_vs_lv_{tag}")
        long = long[long["column"].isin(models)].assign(k=lambda f: axis(f, "k"))
        ks = sorted({float(x) for x in long["k"]})
        wing_k = ks[0]
        atm_k = min(ks, key=lambda x: abs(math.log(x)))
        wing = long[np.isclose(long["k"], wing_k)]
        atm = long[np.isclose(long["k"], atm_k)]
        w = wing.iloc[int(np.argmax(np.abs(wing["value"].to_numpy(float))))]
        a = atm.iloc[int(np.argmin(np.abs(atm["value"].to_numpy(float))))]
        w_v, w_s = abs(float(w["value"])), float(w["stderr"])
        a_v, a_s = abs(float(a["value"])), float(a["stderr"])
        verdict = (
            "the models agree most closely in the put wing"
            if w_v + 2.0 * w_s < a_v - 2.0 * a_s
            else "the wing is **not** resolved as the closer region at 2 stderr"
        )
        lines.append(
            f"- {window_label(t1, t2)}: the largest |LSV - LV| over the {len(models)} headline "
            f"LSV models is {format_value_text(w_v, w_s)} vol points at {w['row']} "
            f"({w['column']}); the smallest at {a['row']} is {format_value_text(a_v, a_s)} "
            f"({a['column']}): {verdict}."
        )
    lines += [
        "",
        "Each difference carries the quadrature error of the two stored estimates. The two "
        "share random numbers and the store keeps no per-path samples, so their correlation "
        "is not measured here: the quadrature error is not the exact error of the difference, "
        "and the z tables are d over that error, not exact significances. Comparing the "
        "largest wing difference with the smallest at-the-money one guards the claim against "
        "the choice of model, not against that error.",
        "",
    ]
    for t1, t2 in WINDOWS:
        tag = window_tag(t1, t2)
        if _wing_models(results, tag):
            lines += [f"{{{{table:wing_vs_lv_{tag}}}}}", "", f"{{{{table:wing_z_{tag}}}}}", ""]
    lines += [
        "",
        "{{figure:put_wing_vs_lv}}",
        "",
        "## The capped cliquet ladder",
        "",
        "Interpretation, pending the owner: the ladder is read as the study cliquet at its fixed "
        "2% local cap across the vol-of-vol grid (and the 2F presets), not as a ladder over cap "
        "levels.",
        "",
        "{{table:cliquet_ladder}}",
        "",
        "{{figure:cliquet_vs_nu}}",
        "",
    ]
    if any(_has(results, t) for t in REGRESSION_TABLES):
        n_keys = 0
        n_within = 0
        worst = (0.0, "")
        for t in REGRESSION_TABLES:
            if not _has(results, t):
                continue
            piv = results.pivot(t)
            n_keys += len(piv)
            n_within += int(piv["within"].sum())
            nse = piv["n_se"].to_numpy(dtype=float)
            if np.isfinite(nse).any() and float(np.nanmax(nse)) > worst[0]:
                worst = (float(np.nanmax(nse)), str(piv.index[int(np.nanargmax(nse))]))
        meta = results.pivot("regression_meta") if _has(results, "regression_meta") else None
        all_match = bool(meta is not None and (meta["match"] == 1.0).all())
        lines += [
            "## Regeneration of the M4 baseline",
            "",
            f"{n_within} of {n_keys} keys are within the regression tolerance of "
            "`tests/test_m4_regression.py`; the largest |d| / combined stderr is "
            f"{worst[0]:.2f} ({worst[1]}). "
            + (
                "The store's particle count, paths and seed are the baseline's, so the two "
                "numbers are the same computation: a non-zero difference would be a code change."
                if all_match
                else "**The budgets differ from the baseline's** (`regression_meta`): the "
                "comparison is then between different estimates, not a regression check."
            )
            + (
                " The non-zero differences are the baseline file's six-significant-digit "
                "rounding."
                if all_match and worst[0] < 0.05
                else ""
            ),
            "",
            "{{table:regression_meta}}",
            "",
        ]
        for t in REGRESSION_TABLES:
            if _has(results, t):
                lines += [f"{{{{table:{t}}}}}", "", f"{{{{table:{t}_check}}}}", ""]
        lines += ["{{figure:regression_ratio}}", ""]
    return "\n".join(lines)
