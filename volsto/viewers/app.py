"""``volsto-viewer`` — the Streamlit app over the results store and the leverage cache (SPEC
§9.2 "Read API and app", M9 Parts 2–3; owner's ``viewers/app.py``, here
``volsto/viewers/app.py`` with the console script in ``pyproject.toml``).

Three entry modes of the same file:

* **launcher** (``volsto-viewer [--cache R] [--store R] [--outputs R] [--snapshots R]
  [--grid P] [--config Y] [--port N] [--headless]`` or ``python -m volsto.viewers.app``):
  :func:`main` resolves the :class:`~volsto.viewers.config.ViewerConfig` (defaults ← YAML ← env
  ← flags), exports it as the ``VOLSTO_*`` environment variables and executes ``streamlit run
  <this file> -- <flags>`` in a subprocess, so the pages read the same configuration through
  :meth:`ViewerConfig.from_env`;
* **check** (``volsto-viewer --check [--check-timeout S]`` with the same root flags):
  :func:`check_pages` renders every registered page headless through ``streamlit.testing.v1
  .AppTest`` against the configured roots — nothing is served, nothing is computed — and
  :func:`main` prints one line per page (wall clock, dataframes, figures, download buttons,
  notices, the exception when one was raised) and exits 0 when no page raised, 1 otherwise
  (:func:`exit_code`).  An empty store is a valid configuration for the check: the pages then
  print the ``volsto-precompute`` command and the check says the store is empty;
* **Streamlit script** (what ``streamlit run`` executes): :func:`streamlit_main` parses the
  flags after ``--`` again (belt and braces when the app is started with ``streamlit run``
  directly), sets the environment for the pages and registers the eight pages of
  :data:`volsto.viewers.pages.PAGES` with ``st.navigation`` (files ``<n>_<name>.py``, each
  rendering through its ``run_if_streamlit(render)`` guard).

Read-only by construction: the app imports nothing that calibrates; a missing point or study
file is reported by the page with the command that produces it.

Checked by ``tests/test_viewers_api.py`` (``test_pages_registry``, ``test_app_help``) and
``tests/test_viewers_app.py`` (``test_check_pages_in_process``, ``test_viewer_check_cli``).
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from volsto.viewers.config import (
    CLI_FLAGS,
    DEFAULT_CONFIG_FILE,
    ENV_CONFIG_FILE,
    ENV_VARS,
    ViewerConfig,
    build_parser,
)
from volsto.viewers.pages import PAGES, PAGES_DIR, in_streamlit

APP_FILE = Path(__file__).resolve()
APP_TITLE = "volsto viewer"
#: Per-page timeout of ``--check`` (seconds).  Page 1 builds the SSVI + Dupire surface (≈ 3–7 s
#: measured) and the first figure probes the kaleido engine; the rest render well under a second
#: on a small store.  Generous so a cold laptop or a large store never fails the check on time.
CHECK_TIMEOUT_S = 300.0


def page_list() -> list[tuple[Path, str, str]]:
    """``(path, title, icon)`` of the registered pages, in sidebar order."""
    return [(PAGES_DIR / f, title, icon) for f, title, icon in PAGES]


def app_parser() -> argparse.ArgumentParser:
    """The ``volsto-viewer`` parser: the configuration flags of
    :func:`~volsto.viewers.config.build_parser` plus the launcher-only ``--check`` options."""
    p = build_parser()
    p.add_argument(
        "--check",
        action="store_true",
        help=(
            "render every page headless (streamlit AppTest) against the configured roots, print "
            "one line per page and exit 0 when no page raised, 1 otherwise; nothing is served, "
            "nothing is computed"
        ),
    )
    p.add_argument(
        "--check-timeout",
        type=float,
        default=CHECK_TIMEOUT_S,
        help=f"per-page timeout of --check in seconds (default {CHECK_TIMEOUT_S:g})",
    )
    return p


def parse_args(argv: Sequence[str] | None = None) -> tuple[ViewerConfig, argparse.Namespace]:
    """:func:`app_parser` on top of :meth:`ViewerConfig.from_sources` — the resolution of
    :meth:`ViewerConfig.from_args` (defaults ← YAML ← environment ← flags) with the extra
    launcher flags accepted."""
    args = app_parser().parse_args(argv)
    yaml_path = args.config or os.environ.get(ENV_CONFIG_FILE) or DEFAULT_CONFIG_FILE
    overrides = {k: getattr(args, k) for k in ENV_VARS}
    return ViewerConfig.from_sources(yaml_path=yaml_path, overrides=overrides), args


def streamlit_command(cfg: ViewerConfig, *, port: int, headless: bool) -> list[str]:
    """The ``streamlit run`` argv the launcher executes (flags forwarded after ``--``)."""
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_FILE),
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    if headless:
        cmd += ["--server.headless", "true"]
    cmd.append("--")
    for field_name, flag in CLI_FLAGS.items():
        cmd += [flag, str(getattr(cfg, field_name))]
    return cmd


def streamlit_main(argv: Sequence[str] | None = None) -> None:
    """The Streamlit script: configuration → environment → navigation."""
    import streamlit as st

    cfg, _ = ViewerConfig.from_args(list(argv) if argv is not None else sys.argv[1:])
    os.environ.update(cfg.to_env())
    st.set_page_config(page_title=APP_TITLE, layout="wide", page_icon=":material/insights:")
    pages = [
        st.Page(str(path), title=title, icon=icon, default=(i == 0))
        for i, (path, title, icon) in enumerate(page_list())
    ]
    with st.sidebar:
        st.markdown(f"**{APP_TITLE}** — read-only over the results store and the leverage cache")
        st.caption(f"store `{cfg.store_root}`")
        st.caption(f"cache `{cfg.cache_root}`")
    st.navigation(pages).run()


# --------------------------------------------------------------------------------------------
# --check: headless render of every page
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PageCheck:
    """One page's headless render: what the check prints (:func:`format_checks`)."""

    page: str
    title: str
    ok: bool
    seconds: float
    n_dataframes: int
    n_figures: int
    n_downloads: int
    n_notices: int
    exceptions: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return dataclasses.asdict(self)


def check_pages(
    cfg: ViewerConfig,
    *,
    timeout: float = CHECK_TIMEOUT_S,
    pages: Sequence[tuple[Path, str, str]] | None = None,
    config_file: str | Path | None = None,
) -> list[PageCheck]:
    """Render ``pages`` (default: every registered page) headless with ``AppTest`` against
    ``cfg`` and return one :class:`PageCheck` per page.  The configuration reaches the pages
    the way the app passes it — the ``VOLSTO_*`` environment variables (and
    ``VOLSTO_VIEWER_CONFIG`` when ``config_file`` is given); the environment is restored
    afterwards.  Counts: ``st.dataframe`` elements, plotly charts, download buttons (Excel /
    PNG / SVG), notices (``warning`` + ``info`` + ``code`` blocks — where a page names the
    command that produces a missing point).  Never raises for a page failure: the failure is the
    ``ok = False`` row with the exception text."""
    from streamlit.testing.v1 import AppTest

    saved = {k: os.environ.get(k) for k in [*ENV_VARS.values(), ENV_CONFIG_FILE]}
    os.environ.update(cfg.to_env())
    if config_file is not None:
        os.environ[ENV_CONFIG_FILE] = str(config_file)
    results: list[PageCheck] = []
    try:
        for path, title, _icon in pages if pages is not None else page_list():
            t0 = time.perf_counter()
            exceptions: list[str] = []
            n_df = n_fig = n_dl = n_notice = 0
            try:
                at = AppTest.from_file(str(path), default_timeout=timeout).run()
            except Exception as exc:  # a script that cannot even start (syntax, timeout)
                exceptions.append(f"{type(exc).__name__}: {exc}")
            else:
                exceptions.extend(str(e.value) for e in at.exception)
                n_df = len(at.dataframe)
                n_fig = len(at.get("plotly_chart"))
                n_dl = len(at.get("download_button"))
                n_notice = len(at.warning) + len(at.info) + len(at.code)
            results.append(
                PageCheck(
                    path.name,
                    title,
                    not exceptions,
                    time.perf_counter() - t0,
                    n_df,
                    n_fig,
                    n_dl,
                    n_notice,
                    tuple(exceptions),
                )
            )
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return results


def exit_code(results: Sequence[PageCheck]) -> int:
    """0 when every page rendered without an exception, 1 otherwise."""
    return 0 if results and all(r.ok for r in results) else 1


def format_checks(results: Sequence[PageCheck]) -> str:
    """The check report: one line per page and the verdict."""
    lines = [
        f"{'page':<22} {'status':<6} {'wall':>7} {'tables':>6} {'figures':>7} {'downloads':>9} "
        f"{'notices':>7}"
    ]
    for r in results:
        lines.append(
            f"{r.page:<22} {'ok' if r.ok else 'FAIL':<6} {r.seconds:>6.2f}s {r.n_dataframes:>6} "
            f"{r.n_figures:>7} {r.n_downloads:>9} {r.n_notices:>7}"
        )
        for text in r.exceptions:
            lines.append(f"    exception: {text}")
    n_fail = sum(not r.ok for r in results)
    total = sum(r.seconds for r in results)
    verdict = "all pages rendered" if n_fail == 0 else f"{n_fail} page(s) FAILED"
    lines.append(f"check: {len(results)} pages, {verdict}, {total:.1f} s wall")
    return "\n".join(lines)


def run_check(cfg: ViewerConfig, *, timeout: float, config_file: str | Path | None) -> int:
    """``volsto-viewer --check``: print the configuration, the store summary and the report."""
    from volsto.viewers.api import precompute_command, store_summary

    print("volsto-viewer --check: rendering every page headless (nothing served, nothing computed)")
    for k, v in cfg.as_dict().items():
        print(f"  {k}: {v}", flush=True)
    s = store_summary(cfg)
    if s.n_points == 0:
        print(
            "  store: EMPTY (0 points) — the pages print the precompute command: "
            f"{precompute_command(cfg)}"
        )
    else:
        print(
            f"  store: {s.n_points} points, {s.n_runs} runs, grid '{s.grid_name}', "
            f"{s.n_particles} particles, code tag {s.code_tag}"
        )
    results = check_pages(cfg, timeout=timeout, config_file=config_file)
    print(format_checks(results), flush=True)
    return exit_code(results)


def main(argv: Sequence[str] | None = None) -> int:
    """Launcher: resolve the configuration and start ``streamlit run`` (returns its exit code),
    or run the headless check with ``--check``."""
    if in_streamlit():  # executed by ``streamlit run`` — not a launcher call
        streamlit_main(argv)
        return 0
    cfg, args = parse_args(argv)
    if args.check:
        return run_check(cfg, timeout=float(args.check_timeout), config_file=args.config)
    env = {**os.environ, **cfg.to_env()}
    cmd = streamlit_command(cfg, port=int(args.port), headless=bool(args.headless))
    print(f"volsto-viewer: {' '.join(cmd)}", flush=True)
    for k, v in cfg.as_dict().items():
        print(f"  {k}: {v}", flush=True)
    try:
        return int(subprocess.call(cmd, env=env))
    except KeyboardInterrupt:  # pragma: no cover - interactive stop
        return 130


if in_streamlit():
    streamlit_main()
elif __name__ == "__main__":  # pragma: no cover
    sys.exit(main())


__all__ = [
    "APP_FILE",
    "APP_TITLE",
    "CHECK_TIMEOUT_S",
    "PageCheck",
    "app_parser",
    "build_parser",
    "check_pages",
    "exit_code",
    "format_checks",
    "main",
    "page_list",
    "parse_args",
    "run_check",
    "streamlit_command",
]
