# ruff: noqa: N999 — Streamlit multipage file name
"""Page 9 — What-if: forward smile & skew (owner's decision of 2026-09-30; SPEC §9): start from a
marking state (default: the desk's SPX 2022-12-30 mark), edit the stochastic-volatility parameters
(ν, θ, k1, k2, ρ12, ρ_SX1, ρ_SX2), choose the forward windows, press **Run**: both leverages are
calibrated at 100 000 particles (the owner's choice; :data:`~volsto.viewers.whatif.
WHATIF_PARTICLES`) — about 15–25 s each, a cache hit the second time — and the forward smiles are
priced on common random numbers; the page shows the base and moved smiles with their errors, the
change per strike with its **paired** error, and the forward ATM level / skew / curvature of base,
moved and change (:mod:`volsto.viewers.whatif` states the method).

The only page that computes: it calibrates small leverages into ``<outputs>/whatif/cache`` and
stores each run under ``<outputs>/whatif/runs`` (reloaded instantly from the run history); the
other eight pages and the read API stay read-only.  Production-quality numbers (8·10⁵ particles,
4·10⁵ paths) remain the precompute's.  Every table has the Excel export, every figure the PNG /
SVG download.  Rendered headless (without pressing Run) by ``tests/test_viewers_whatif.py``.
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from volsto.viewers import whatif as wi
from volsto.viewers.components import (
    error_trace,
    figure_with_download,
    page_header,
    table_with_export,
)
from volsto.viewers.config import ViewerConfig
from volsto.viewers.grid import REPO_ROOT
from volsto.viewers.pages import run_if_streamlit

PARAM_LABELS: dict[str, str] = {
    "nu": "nu (vol of vol)",
    "theta": "theta (factor mix)",
    "k1": "k1 (fast mean reversion)",
    "k2": "k2 (slow mean reversion)",
    "rho12": "rho12 (factor correlation)",
    "rho_SX1": "rho_SX1 (spot-fast factor)",
    "rho_SX2": "rho_SX2 (spot-slow factor)",
}
VP = 100.0


def _bases() -> list[str]:
    root = REPO_ROOT / "configs" / "studies" / "m7_p1_marking"
    found = sorted(str(p.relative_to(REPO_ROOT)) for p in root.glob("spx_*.yaml"))
    return found or [wi.DESK_MARK]


def _form(cfg: ViewerConfig) -> wi.WhatIfRequest | None:
    bases = _bases()
    default = bases.index(wi.DESK_MARK) if wi.DESK_MARK in bases else 0
    base = st.selectbox("Base marking state", bases, index=default, key="wi_base")
    try:
        params = wi.base_params(wi.WhatIfRequest(base_spec=base), REPO_ROOT)
    except (OSError, ValueError, KeyError) as exc:
        st.error(f"cannot read the base state {base}: {exc}")
        return None
    st.caption("Base parameters: " + ", ".join(f"{k} {v:.4g}" for k, v in params.items()))
    cols = st.columns(4)
    moved: dict[str, float] = {}
    for i, k in enumerate(wi.PARAMS):
        with cols[i % 4]:
            moved[k] = float(
                st.number_input(
                    PARAM_LABELS[k],
                    value=float(params[k]),
                    format="%.4f",
                    key=f"wi_{base}_{k}",
                )
            )
    labels = {f"{a:g}y→{b:g}y": (a, b) for a, b in wi.DEFAULT_WINDOWS}
    chosen = st.multiselect("Forward windows", list(labels), default=list(labels), key="wi_windows")
    with st.expander("Budget"):
        n_particles = int(
            st.number_input(
                "leverage particles", value=wi.WHATIF_PARTICLES, step=50_000, key="wi_np"
            )
        )
        n_paths = int(
            st.number_input(
                "pricing paths (even)", value=wi.WHATIF_PATHS, step=10_000, key="wi_paths"
            )
        )
    changes = tuple(
        (k, v) for k, v in moved.items() if abs(v - params[k]) > 1e-12 * max(1.0, abs(params[k]))
    )
    if not chosen:
        st.info("choose at least one forward window")
        return None
    try:
        return wi.WhatIfRequest(
            base_spec=base,
            changes=changes,
            windows=tuple(labels[c] for c in chosen),
            n_particles=n_particles,
            n_paths=n_paths + (n_paths % 2),
        )
    except ValueError as exc:
        st.error(str(exc))
        return None


def _show(res: wi.WhatIfResult) -> None:
    info = res.info
    st.subheader(f"Result — {wi.changes_label(res.request.changes)}")
    cal = info.get("calibration_seconds", [float("nan"), float("nan")])
    st.caption(
        f"{res.request.n_particles:,} particles, {res.request.n_paths:,} paths, seed "
        f"{res.request.seed}; calibration {cal[0]:.0f} s + {cal[1]:.0f} s (cache hits "
        f"{info.get('cache_hits', '?')}), pricing "
        f"{info.get('pricing_seconds', float('nan')):.0f} s; values in vol points, the change's "
        "error paired (common random numbers)"
    )
    fits = res.fits.copy()
    for c in ("atm", "atm_stderr", "skew", "skew_stderr", "curvature", "curvature_stderr"):
        fits[c] = fits[c] * VP
    table_with_export(
        fits[
            [
                "window",
                "which",
                "atm",
                "atm_stderr",
                "skew",
                "skew_stderr",
                "curvature",
                "curvature_stderr",
                "chi2_dof",
            ]
        ].round(4),
        "what-if forward fits",
        caption="Forward ATM level, skew (vp per unit log-strike) and curvature, base / moved / "
        "change; the change's errors from the paired strike errors.",
    )
    for window, g in res.smiles.groupby("window", sort=False):
        fig = go.Figure()
        fig.add_trace(
            error_trace(
                (g["log_moneyness"]).tolist(),
                (g["base_vol"] * VP).tolist(),
                (g["base_vol_stderr"] * VP).tolist(),
                name="base",
                unit="vp",
            )
        )
        fig.add_trace(
            error_trace(
                (g["log_moneyness"]).tolist(),
                (g["moved_vol"] * VP).tolist(),
                (g["moved_vol_stderr"] * VP).tolist(),
                name="moved",
                unit="vp",
            )
        )
        fig.update_layout(
            title=f"Forward smile {window}",
            xaxis_title="log forward moneyness",
            yaxis_title="implied vol (vp)",
        )
        figure_with_download(fig, f"what-if smile {window}")
        ch = go.Figure()
        ch.add_trace(
            error_trace(
                (g["log_moneyness"]).tolist(),
                (g["change"] * VP).tolist(),
                (g["change_stderr"] * VP).tolist(),
                name="moved - base (paired)",
                unit="vp",
            )
        )
        ch.add_hline(y=0.0, line_width=1, line_color="grey")
        ch.update_layout(
            title=f"Change of the forward smile {window}",
            xaxis_title="log forward moneyness",
            yaxis_title="vol change (vp)",
        )
        figure_with_download(ch, f"what-if change {window}")
    table_with_export(res.smiles.round(6), "what-if forward smiles")


def render(cfg: ViewerConfig) -> None:
    page_header(
        "What-if",
        cfg,
        subtitle="Forward smile & skew under edited stochastic-vol parameters: press Run and "
        "both leverages are calibrated at 100 000 particles (about 15-25 s each, cached "
        "afterwards), the forward smiles priced on common random numbers. The only page that "
        "computes.",
    )
    req = _form(cfg)
    run = st.button("Run", type="primary", disabled=req is None, key="wi_run")
    if run and req is not None:
        with st.spinner("calibrating the leverages and pricing the forward smiles…"):
            try:
                st.session_state["wi_result"] = wi.run_whatif(req, cfg.outputs_root, REPO_ROOT)
            except Exception as exc:  # a degenerate parameter set fails its calibration
                st.error(f"the what-if failed: {type(exc).__name__}: {exc}")
    history = wi.stored_runs(cfg.outputs_root)
    if history:
        names = [
            f"{r.request.key()} · {wi.changes_label(r.request.changes)} · "
            f"{len(r.request.windows)} windows"
            for r in history
        ]
        pick = st.selectbox("Run history", ["(latest run)", *names], key="wi_hist")
        if pick != "(latest run)":
            st.session_state["wi_result"] = history[names.index(pick)]
    res = st.session_state.get("wi_result")
    if res is not None:
        _show(res)
    else:
        st.info("No run yet: edit the parameters and press Run.")


run_if_streamlit(render)
