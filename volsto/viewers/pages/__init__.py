"""The viewer pages (SPEC §9 / §9.2, M9 Part 2; owner's ``viewers/pages/``, here
``volsto/viewers/pages/``): one Streamlit page per module, registered in :data:`PAGES` and
served by :mod:`volsto.viewers.app` through ``st.navigation``.

Page convention (every page file): a module docstring stating what it shows and which test
renders it, ``def render(cfg: ViewerConfig) -> None`` doing all the work from the read API
(:mod:`volsto.viewers.api`) — never calibrating, never simulating —, and the module-level guard
``run_if_streamlit(render)`` which calls ``render(ViewerConfig.from_env())`` only when a
Streamlit script-run context exists (``streamlit run`` / ``st.navigation`` / ``AppTest``), so
importing a page or type-checking it renders nothing.  The file names follow the Streamlit
``<n>_<name>.py`` multipage convention (hence the ``N999`` noqa per file).

Checked by ``tests/test_viewers_api.py`` (``test_pages_registry``, ``test_pages_render``).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from volsto.viewers.config import ViewerConfig

PAGES_DIR = Path(__file__).resolve().parent

#: ``(file name, title, icon)`` in sidebar order — the owner's eight pages.
PAGES: tuple[tuple[str, str, str], ...] = (
    ("1_surface_model.py", "Surface & model", ":material/grid_on:"),
    ("2_forward_smile.py", "Forward smile", ":material/show_chart:"),
    ("3_forward_skew.py", "Forward skew", ":material/trending_down:"),
    ("4_smile_dynamics.py", "Smile dynamics", ":material/timeline:"),
    ("5_model_risk.py", "Model risk", ":material/compare_arrows:"),
    ("6_product_grid.py", "Product grid", ":material/apps:"),
    ("7_marking.py", "Marking / calibration", ":material/tune:"),
    ("8_hedging.py", "Hedging", ":material/account_balance:"),
)


def page_paths() -> list[Path]:
    return [PAGES_DIR / f for f, _, _ in PAGES]


def in_streamlit() -> bool:
    """True inside a Streamlit script run (``streamlit run``, ``st.navigation``, ``AppTest``)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:  # pragma: no cover - streamlit is a viewers extra
        return False
    return get_script_run_ctx(suppress_warning=True) is not None


def run_if_streamlit(render: Callable[[ViewerConfig], None]) -> None:
    """The page guard: render with :meth:`ViewerConfig.from_env` under Streamlit only."""
    if in_streamlit():
        render(ViewerConfig.from_env())
